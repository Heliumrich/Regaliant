//! GPU memory residency (RENDERER.md, "Residency design"; `DS_RESIDENT=1`).
//!
//! Stage 1, the foundation; nothing is bound from the mirror yet:
//! - The page table: per 4 KiB guest page a state (`ST_*`), kept in 64 MiB
//!   chunks made on first use (guest addresses are host addresses).
//! - Arming: a page whose contents the mirror holds is made read-only
//!   (`arm`); the CPU's next write faults (`on_fault`, asked first by the
//!   exception handler), and the page becomes writable and device-invalid
//!   again. Arm before copying a page out: a write in between then faults and
//!   invalidates the copy. Only pages that are plain read-write now are
//!   armed; imported pages must never be (protecting pages imported into the
//!   GPU lost the device: `track`); a guest protection change, commit or
//!   decommit resets the pages it covers (`remapped`). A page's arming and
//!   disarming hold its `ST_BUSY` bit, so a write fault racing either waits
//!   for the protection change instead of being taken for a crash.
//! - The mirror: one sparse buffer over guest [MIRROR_BASE, MIRROR_END), its
//!   64 KiB pages bound on first use (`back`) to device memory taken from
//!   64 MiB blocks.
//! - Staging copies into and out of the mirror (`upload_now`,
//!   `readback_now`): synchronous here (the boot self-test).
//!
//! Stage 2: read-only buffer bindings of pages the mirror holds bind it
//! (`Exec::resident_bind`). A page is uploaded by the batch that first binds
//! it: claimed (`claim_pages`: its upload pending), armed, copied (memcpy)
//! into the slot's staging ring and from there into the mirror in the
//! batch's prologue; once that batch is submitted (`submitted`) every later
//! batch may bind it (each batch begins behind a full barrier). While an
//! upload is pending no other batch uploads or binds the page (two pending
//! copies could land in either order). A page written again within
//! VOLATILE_FRAMES of its last write fault is volatile: never armed again
//! (each fault and re-arm costs microseconds), it keeps the old placements.
//! Pages an import covers are host class (`mark_host`, called before every
//! import): protecting imported pages lost the device. Writes that cannot
//! fault (file reads: the kernel fails them on a read-only page, "Invalid
//! access to memory location") announce themselves (`host_write_begin` and
//! `_end`, from `track::host_write`'s guard): their pages are disarmed and
//! not armed again until the write ends.

use super::vk::ctx;
use ash::vk;
use parking_lot::{Mutex, RwLock};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicU8, Ordering};

pub const PAGE: u64 = 4096;
/// Page states per chunk of the table: 64 MiB of guest memory.
const CHUNK_SHIFT: u32 = 26;
const CHUNK_PAGES: usize = 1 << (CHUNK_SHIFT - 12);
/// Guest span the mirror covers (the title's direct memory and the GPU
/// buffers seen at the courtyard lie in [0x10_0000_0000, 0x15_0000_0000)).
pub const MIRROR_BASE: u64 = 0x10_0000_0000;
pub const MIRROR_END: u64 = 0x18_0000_0000;
/// Device memory blocks the mirror's pages are bound from.
const BLOCK: u64 = 64 << 20;

/// The mirror holds the page's current contents (no CPU write since).
pub const ST_VALID: u8 = 1;
/// Made read-only by us: the CPU's next write faults.
pub const ST_ARMED: u8 = 2;
/// A protection change of the page is in progress (`arm`, `on_fault`).
const ST_BUSY: u8 = 4;
/// Host class: imported or written by CP writes the GPU performs; never armed.
pub const ST_HOST: u8 = 8;
/// Written by the CPU in nearby frames again and again: never armed.
pub const ST_VOLATILE: u8 = 16;
/// A page's upload sequence while the batch copying it is not submitted.
const PENDING: u64 = u64::MAX;
/// A write fault this many frames after the page's last one makes it volatile.
const VOLATILE_FRAMES: u32 = 8;

struct Chunk {
    pages: Box<[AtomicU8]>,
    /// The batch that uploaded the page (PENDING: not submitted yet; 0: none).
    seqs: Box<[AtomicU64]>,
    /// The frame of the page's last write fault, plus one (0: none).
    faulted: Box<[AtomicU32]>,
    /// Writes that cannot fault in progress over the page (`host_write_begin`).
    writing: Box<[AtomicU8]>,
}

/// Chunks by guest address >> CHUNK_SHIFT; never removed (16 KiB each).
static TABLE: RwLock<Option<HashMap<u64, &'static Chunk>>> = RwLock::new(None);
static ON: AtomicBool = AtomicBool::new(false);

/// Counters (`status`): pages armed, write faults taken, arm calls, their
/// protection calls and ns, mirror pages backed, uploads, readbacks, resets,
/// fault races, pages claimed for upload, pages made volatile, pages made
/// host class, bindings served from the mirror, and refusals by REFUSE_NAMES.
static STATS: [AtomicU64; 18] = [const { AtomicU64::new(0) }; 18];
/// Protection calls and ns that disarmed (the others armed).
const S_UNPROTECTS: usize = 14;
const S_UNPROTECT_NS: usize = 15;
/// Host writes announced, and pages disarmed by them.
const S_HOST_WRITES: usize = 16;
const S_HOST_WRITE_PAGES: usize = 17;
const S_CLAIMED: usize = 9;
const S_VOLATILE: usize = 10;
const S_HOSTED: usize = 11;
const S_BINDS: usize = 12;
const S_BIND_PAGES: usize = 13;
/// Why a read-only binding was not served from the mirror (`refused`).
pub const REFUSE_NAMES: [&str; 8] = ["outside the mirror or too large", "GPU or CP writes pending", "host or volatile pages", "another batch's upload pending", "imported pages", "staging full", "arming failed", "mirror pages not bound"];
static REFUSED: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];

/// Counts a refusal (`REFUSE_NAMES` index).
pub fn refused(why: usize) {
    REFUSED[why.min(7)].fetch_add(1, Ordering::Relaxed);
}

/// Counts a binding served from the mirror and its pages.
pub fn served(pages: u64) {
    STATS[S_BINDS].fetch_add(1, Ordering::Relaxed);
    STATS[S_BIND_PAGES].fetch_add(pages, Ordering::Relaxed);
}
const S_ARMED: usize = 0;
const S_FAULTS: usize = 1;
const S_PROTECTS: usize = 2;
const S_PROTECT_NS: usize = 3;
const S_BACKED: usize = 4;
const S_UPLOADS: usize = 5;
const S_READBACKS: usize = 6;
const S_RESETS: usize = 7;
const S_RACES: usize = 8;

/// `DS_RESIDENT=1`: residency is wanted (the device enables sparse buffers).
pub fn wanted() -> bool {
    static W: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *W.get_or_init(|| std::env::var("DS_RESIDENT").ok().as_deref() == Some("1"))
}

/// Residency runs (wanted, and the mirror was made).
pub fn enabled() -> bool {
    ON.load(Ordering::Acquire)
}

fn chunk(addr: u64, create: bool) -> Option<&'static Chunk> {
    let key = addr >> CHUNK_SHIFT;
    if let Some(c) = TABLE.read().as_ref().and_then(|t| t.get(&key).copied()) {
        return Some(c);
    }
    if !create {
        return None;
    }
    let mut g = TABLE.write();
    let t = g.get_or_insert_with(HashMap::new);
    Some(*t.entry(key).or_insert_with(|| {
        Box::leak(Box::new(Chunk {
            pages: (0..CHUNK_PAGES).map(|_| AtomicU8::new(0)).collect(),
            seqs: (0..CHUNK_PAGES).map(|_| AtomicU64::new(0)).collect(),
            faulted: (0..CHUNK_PAGES).map(|_| AtomicU32::new(0)).collect(),
            writing: (0..CHUNK_PAGES).map(|_| AtomicU8::new(0)).collect(),
        }))
    }))
}

fn idx(addr: u64) -> usize {
    ((addr >> 12) as usize) & (CHUNK_PAGES - 1)
}

fn slot(c: &Chunk, addr: u64) -> &AtomicU8 {
    &c.pages[idx(addr)]
}

/// The state of the page holding `addr` (0 when untracked).
pub fn state(addr: u64) -> u8 {
    chunk(addr, false).map_or(0, |c| slot(c, addr).load(Ordering::Acquire) & !ST_BUSY)
}

/// Marks [base, base+len) host class: never armed (disarmed now if armed).
pub fn mark_host(base: u64, len: u64) {
    let (lo, hi) = (base & !(PAGE - 1), (base + len).next_multiple_of(PAGE));
    disarm_range(lo, hi);
    let mut p = lo;
    let mut n = 0;
    while p < hi {
        if let Some(c) = chunk(p, true) {
            if slot(c, p).fetch_or(ST_HOST, Ordering::AcqRel) & ST_HOST == 0 {
                n += 1;
            }
        }
        p += PAGE;
    }
    STATS[S_HOSTED].fetch_add(n, Ordering::Relaxed);
}

fn protect(a: u64, n: u64, prot: u32) -> bool {
    use windows_sys::Win32::System::Memory::{VirtualProtect, PAGE_READONLY};
    let t = std::time::Instant::now();
    let mut old = 0u32;
    let ok = unsafe { VirtualProtect(a as _, n as usize, prot, &mut old) } != 0;
    let k = if prot == PAGE_READONLY { 0 } else { S_UNPROTECTS - S_PROTECTS };
    STATS[S_PROTECTS + k].fetch_add(1, Ordering::Relaxed);
    STATS[S_PROTECT_NS + k].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    ok
}

/// Whether [a, a+n) is committed plain read-write memory now.
fn plain_rw(a: u64, n: u64) -> bool {
    use windows_sys::Win32::System::Memory::{VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT, PAGE_READWRITE};
    let mut p = a;
    while p < a + n {
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        if unsafe { VirtualQuery(p as _, &mut info, std::mem::size_of::<MEMORY_BASIC_INFORMATION>()) } == 0 {
            return false;
        }
        if info.State != MEM_COMMIT || info.Protect != PAGE_READWRITE {
            return false;
        }
        p = info.BaseAddress as u64 + info.RegionSize as u64;
    }
    true
}

/// Arms the pages of [base, base+len) (rounded out to pages): read-only,
/// device-valid. Pages of the host class, busy or already armed are left;
/// a run that is not plain read-write memory is not armed. Returns the
/// pages armed.
pub fn arm(base: u64, len: u64) -> u64 {
    use windows_sys::Win32::System::Memory::PAGE_READONLY;
    let (lo, hi) = (base & !(PAGE - 1), (base + len).next_multiple_of(PAGE));
    let mut armed = 0;
    let mut p = lo;
    // Runs of pages taken busy, protected with one call each.
    let mut run: Option<(u64, u64)> = None;
    let flush = |run: &mut Option<(u64, u64)>, armed: &mut u64| {
        if let Some((a, e)) = run.take() {
            let ok = plain_rw(a, e - a) && protect(a, e - a, PAGE_READONLY);
            let mut q = a;
            while q < e {
                if let Some(c) = chunk(q, false) {
                    let s = slot(c, q);
                    if ok {
                        s.store((s.load(Ordering::Acquire) | ST_ARMED | ST_VALID) & !ST_BUSY, Ordering::Release);
                    } else {
                        s.fetch_and(!ST_BUSY, Ordering::AcqRel);
                    }
                }
                q += PAGE;
            }
            if ok {
                *armed += (e - a) / PAGE;
            }
        }
    };
    while p < hi {
        let c = chunk(p, true).expect("chunk");
        let s = slot(c, p);
        let cur = s.load(Ordering::Acquire);
        let take = cur & (ST_HOST | ST_BUSY | ST_ARMED) == 0 && c.writing[idx(p)].load(Ordering::Acquire) == 0 && s.compare_exchange(cur, cur | ST_BUSY, Ordering::AcqRel, Ordering::Acquire).is_ok();
        if take {
            run = Some(match run {
                Some((a, e)) if e == p => (a, p + PAGE),
                other => {
                    let mut r = other;
                    flush(&mut r, &mut armed);
                    (p, p + PAGE)
                }
            });
        } else {
            flush(&mut run, &mut armed);
        }
        p += PAGE;
    }
    flush(&mut run, &mut armed);
    STATS[S_ARMED].fetch_add(armed, Ordering::Relaxed);
    armed
}

/// Disarms the armed pages of [lo, hi) (page-aligned): writable again and
/// device-invalid. Runs of armed pages are taken busy and made writable with
/// one call each (an import window covers up to 16,384 pages).
fn disarm_range(lo: u64, hi: u64) {
    use windows_sys::Win32::System::Memory::PAGE_READWRITE;
    let release = |a: u64, e: u64| {
        protect(a, e - a, PAGE_READWRITE);
        let mut q = a;
        while q < e {
            if let Some(c) = chunk(q, false) {
                slot(c, q).fetch_and(!(ST_ARMED | ST_VALID | ST_BUSY), Ordering::AcqRel);
            }
            q += PAGE;
        }
    };
    let mut run: Option<(u64, u64)> = None;
    let mut p = lo;
    while p < hi {
        let Some(c) = chunk(p, false) else {
            if let Some((a, e)) = run.take() {
                release(a, e);
            }
            p = ((p >> CHUNK_SHIFT) + 1) << CHUNK_SHIFT;
            continue;
        };
        let s = slot(c, p);
        let took = loop {
            let cur = s.load(Ordering::Acquire);
            if cur & ST_BUSY != 0 {
                std::hint::spin_loop();
                continue;
            }
            if cur & ST_ARMED == 0 {
                break false;
            }
            if s.compare_exchange(cur, cur | ST_BUSY, Ordering::AcqRel, Ordering::Acquire).is_ok() {
                break true;
            }
        };
        match (took, run) {
            (true, Some((a, e))) if e == p => run = Some((a, p + PAGE)),
            (true, r) => {
                if let Some((a, e)) = r {
                    release(a, e);
                }
                run = Some((p, p + PAGE));
            }
            (false, Some((a, e))) => {
                release(a, e);
                run = None;
            }
            (false, None) => {}
        }
        p += PAGE;
    }
    if let Some((a, e)) = run {
        release(a, e);
    }
}

/// A guest commit, decommit or protection change of [base, base+len): its
/// pages are no longer armed or valid (the new protection is the guest's).
pub fn remapped(base: u64, len: u64) {
    if !enabled() || len == 0 {
        return;
    }
    let (lo, hi) = (base & !(PAGE - 1), base.saturating_add(len).next_multiple_of(PAGE));
    let mut p = lo;
    while p < hi {
        let Some(c) = chunk(p, false) else {
            p = ((p >> CHUNK_SHIFT) + 1) << CHUNK_SHIFT;
            continue;
        };
        let s = slot(c, p);
        loop {
            let cur = s.load(Ordering::Acquire);
            if cur & ST_BUSY != 0 {
                std::hint::spin_loop();
                continue;
            }
            if cur & (ST_ARMED | ST_VALID) == 0 || s.compare_exchange(cur, cur & !(ST_ARMED | ST_VALID), Ordering::AcqRel, Ordering::Acquire).is_ok() {
                break;
            }
        }
        p += PAGE;
    }
    STATS[S_RESETS].fetch_add(1, Ordering::Relaxed);
}

/// An access violation at `addr` (`write`: a write): handled when it is a
/// write to a page we armed (it becomes writable and device-invalid) or one
/// that another thread disarmed meanwhile. Called by the exception handler
/// before anything else; takes no lock a faulting thread of ours could hold
/// (the table's read lock only).
pub fn on_fault(addr: u64, write: bool) -> bool {
    use windows_sys::Win32::System::Memory::PAGE_READWRITE;
    if !write || !enabled() {
        return false;
    }
    let page = addr & !(PAGE - 1);
    let Some(c) = chunk(page, false) else { return false };
    let s = slot(c, page);
    loop {
        let cur = s.load(Ordering::Acquire);
        if cur & ST_BUSY != 0 {
            // Armed or disarmed by another thread right now: retry after.
            std::hint::spin_loop();
            continue;
        }
        if cur & ST_ARMED == 0 {
            // Disarmed meanwhile (the write can be retried) or not ours.
            if plain_rw(page, PAGE) {
                STATS[S_RACES].fetch_add(1, Ordering::Relaxed);
                return true;
            }
            return false;
        }
        if s.compare_exchange(cur, cur | ST_BUSY, Ordering::AcqRel, Ordering::Acquire).is_ok() {
            protect(page, PAGE, PAGE_READWRITE);
            // Written again soon after the last write: volatile from now on.
            let frame = super::FRAMES.load(Ordering::Relaxed) as u32 + 1;
            let last = c.faulted[idx(page)].swap(frame, Ordering::AcqRel);
            let volatile = last != 0 && frame.wrapping_sub(last) <= VOLATILE_FRAMES;
            s.store((cur & !(ST_ARMED | ST_VALID | ST_BUSY)) | if volatile { ST_VOLATILE } else { 0 }, Ordering::Release);
            STATS[S_FAULTS].fetch_add(1, Ordering::Relaxed);
            if volatile {
                STATS[S_VOLATILE].fetch_add(1, Ordering::Relaxed);
            }
            return true;
        }
    }
}

/// What a read-only binding may do with a page (`page`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Page {
    /// The mirror holds it (uploaded by a submitted batch).
    Ready,
    /// A batch not submitted yet uploads it (usable by that batch only).
    Pending,
    /// It needs an upload (`claim_pages`).
    Upload,
    /// Host class, volatile or busy: not from the mirror.
    Refuse,
}

/// The page holding `addr` for a read-only binding.
pub fn page(addr: u64) -> Page {
    let Some(c) = chunk(addr, false) else { return Page::Upload };
    let i = idx(addr);
    let s = c.pages[i].load(Ordering::Acquire);
    if s & (ST_HOST | ST_VOLATILE | ST_BUSY) != 0 || c.writing[i].load(Ordering::Acquire) != 0 {
        return Page::Refuse;
    }
    let seq = c.seqs[i].load(Ordering::Acquire);
    if seq == PENDING {
        Page::Pending
    } else if s & (ST_VALID | ST_ARMED) == ST_VALID | ST_ARMED && seq != 0 {
        Page::Ready
    } else {
        Page::Upload
    }
}

/// Claims the uploads of `pages` (ascending, each `Page::Upload` when
/// looked at): their uploads become pending and they are armed (read-only,
/// device-valid), so the caller may copy them out. Returns how many were
/// claimed, in order: the caller uploads those and gives up on the rest.
pub fn claim_pages(pages: &[u64]) -> usize {
    // Take the uploads (stop at the first another thread holds).
    let mut taken = 0;
    for &p in pages {
        let c = chunk(p, true).expect("chunk");
        let i = idx(p);
        let cur = c.seqs[i].load(Ordering::Acquire);
        if cur == PENDING || c.seqs[i].compare_exchange(cur, PENDING, Ordering::AcqRel, Ordering::Acquire).is_err() {
            break;
        }
        taken += 1;
    }
    // Arm them, contiguous runs with one call each (pages still armed are
    // unchanged since: valid again).
    let mut k = 0;
    while k < taken {
        let mut e = k + 1;
        while e < taken && pages[e] == pages[e - 1] + PAGE {
            e += 1;
        }
        arm(pages[k], pages[e - 1] + PAGE - pages[k]);
        k = e;
    }
    // The claim holds up to the first page that could not be armed; the
    // pages after it are given up (never back to an older upload: device
    // memory may hold a copy older than what the host has now).
    let mut n = 0;
    for (j, &p) in pages[..taken].iter().enumerate() {
        let c = chunk(p, false).expect("chunk");
        let i = idx(p);
        let s = c.pages[i].load(Ordering::Acquire);
        let ok = j == n
            && s & ST_ARMED != 0
            && s & (ST_HOST | ST_VOLATILE) == 0
            && c.writing[i].load(Ordering::Acquire) == 0;
        if ok {
            c.pages[i].fetch_or(ST_VALID, Ordering::AcqRel);
            n += 1;
        } else {
            c.pages[i].fetch_and(!ST_VALID, Ordering::AcqRel);
            c.seqs[i].store(0, Ordering::Release);
        }
    }
    STATS[S_CLAIMED].fetch_add(n as u64, Ordering::Relaxed);
    n
}

/// A write that cannot fault (a file read into guest memory) begins over
/// [base, base+len): its pages in the mirror's span are disarmed and stay
/// out of the mirror until `host_write_end`.
pub fn host_write_begin(base: u64, len: u64) {
    if !enabled() || len == 0 {
        return;
    }
    let (lo, hi) = ((base & !(PAGE - 1)).max(MIRROR_BASE), base.saturating_add(len).next_multiple_of(PAGE).min(MIRROR_END));
    if lo >= hi {
        return;
    }
    let mut p = lo;
    while p < hi {
        if let Some(c) = chunk(p, true) {
            c.writing[idx(p)].fetch_add(1, Ordering::AcqRel);
        }
        p += PAGE;
    }
    disarm_range(lo, hi);
    STATS[S_HOST_WRITES].fetch_add(1, Ordering::Relaxed);
    STATS[S_HOST_WRITE_PAGES].fetch_add((hi - lo) / PAGE, Ordering::Relaxed);
}

/// The write announced by `host_write_begin(base, len)` ended: its pages
/// may be armed again (the mirror's copies of them are stale).
pub fn host_write_end(base: u64, len: u64) {
    if !enabled() || len == 0 {
        return;
    }
    let (lo, hi) = ((base & !(PAGE - 1)).max(MIRROR_BASE), base.saturating_add(len).next_multiple_of(PAGE).min(MIRROR_END));
    let mut p = lo;
    while p < hi {
        if let Some(c) = chunk(p, false) {
            let i = idx(p);
            c.pages[i].fetch_and(!ST_VALID, Ordering::AcqRel);
            let _ = c.writing[i].fetch_update(Ordering::AcqRel, Ordering::Acquire, |w| w.checked_sub(1));
        }
        p += PAGE;
    }
}

/// The batch that uploads `pages` was submitted as `seq`.
pub fn submitted(pages: &[u64], seq: u64) {
    for &p in pages {
        if let Some(c) = chunk(p, false) {
            let _ = c.seqs[idx(p)].compare_exchange(PENDING, seq, Ordering::AcqRel, Ordering::Acquire);
        }
    }
}

/// The batch that would have uploaded `pages` was dropped: they need an
/// upload again.
pub fn dropped(pages: &[u64]) {
    for &p in pages {
        if let Some(c) = chunk(p, false) {
            let i = idx(p);
            if c.seqs[i].compare_exchange(PENDING, 0, Ordering::AcqRel, Ordering::Acquire).is_ok() {
                c.pages[i].fetch_and(!ST_VALID, Ordering::AcqRel);
            }
        }
    }
}

/// The mirror buffer and its sparse page size.
pub fn mirror() -> Option<(vk::Buffer, u64)> {
    if !enabled() {
        return None;
    }
    MIRROR.get().map(|m| (m.buf, m.page))
}

/// The mirror: a sparse buffer over [MIRROR_BASE, MIRROR_END).
struct Mirror {
    buf: vk::Buffer,
    /// Sparse block size (the alignment of its memory requirements).
    page: u64,
    mem_type: u32,
    /// Device memory blocks and the next free offset in the last one.
    blocks: Mutex<(Vec<vk::DeviceMemory>, u64)>,
    /// One bit per mirror page: bound to memory.
    backed: Box<[AtomicU64]>,
}

static MIRROR: std::sync::OnceLock<Mirror> = std::sync::OnceLock::new();

fn make_mirror() -> Result<Mirror, String> {
    let c = ctx();
    if !c.sparse_buffers {
        return Err("sparse buffers not enabled".into());
    }
    let size = MIRROR_END - MIRROR_BASE;
    let usage = vk::BufferUsageFlags::STORAGE_BUFFER
        | vk::BufferUsageFlags::TRANSFER_SRC
        | vk::BufferUsageFlags::TRANSFER_DST
        | vk::BufferUsageFlags::INDEX_BUFFER
        | vk::BufferUsageFlags::INDIRECT_BUFFER;
    let mut bi = vk::BufferCreateInfo::default().size(size).usage(usage).flags(vk::BufferCreateFlags::SPARSE_BINDING | vk::BufferCreateFlags::SPARSE_RESIDENCY);
    if c.families.len() > 1 {
        bi = bi.sharing_mode(vk::SharingMode::CONCURRENT).queue_family_indices(&c.families);
    }
    let buf = unsafe { c.device.create_buffer(&bi, None) }.map_err(|e| format!("vkCreateBuffer (sparse, {} GiB): {e}", size >> 30))?;
    let req = unsafe { c.device.get_buffer_memory_requirements(buf) };
    let Some(mem_type) = c.memory_type(req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL) else {
        unsafe { c.device.destroy_buffer(buf, None) };
        return Err("no device-local memory type for the mirror".into());
    };
    let pages = (size / req.alignment) as usize;
    crate::info!(
        "resident: mirror of guest {MIRROR_BASE:#x}..{MIRROR_END:#x} ({} GiB, sparse pages of {} KiB, memory type {mem_type}; device sparse address space {} GiB)",
        size >> 30,
        req.alignment >> 10,
        c.limits.sparse_address_space_size >> 30
    );
    Ok(Mirror { buf, page: req.alignment, mem_type, blocks: Mutex::new((Vec::new(), BLOCK)), backed: (0..pages.div_ceil(64)).map(|_| AtomicU64::new(0)).collect() })
}

/// Binds device memory to the mirror's pages over guest [base, base+len)
/// that have none yet (synchronously: the bind completes before return).
pub fn back(base: u64, len: u64) -> Result<(), String> {
    let m = MIRROR.get().ok_or("no mirror")?;
    if base < MIRROR_BASE || base + len > MIRROR_END {
        return Err(format!("{base:#x}+{len:#x} is outside the mirror"));
    }
    let c = ctx();
    let first = (base - MIRROR_BASE) / m.page;
    let last = (base + len - 1 - MIRROR_BASE) / m.page;
    let mut g = m.blocks.lock();
    let mut binds = Vec::new();
    for p in first..=last {
        let (w, bit) = ((p / 64) as usize, 1u64 << (p % 64));
        if m.backed[w].load(Ordering::Acquire) & bit != 0 {
            continue;
        }
        if g.1 + m.page > BLOCK {
            let mem = unsafe { c.device.allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(BLOCK).memory_type_index(m.mem_type), None) }
                .map_err(|e| format!("vkAllocateMemory (mirror block): {e}"))?;
            g.0.push(mem);
            g.1 = 0;
        }
        let mem = *g.0.last().expect("block");
        binds.push(vk::SparseMemoryBind::default().resource_offset(p * m.page).size(m.page).memory(mem).memory_offset(g.1));
        g.1 += m.page;
    }
    if binds.is_empty() {
        return Ok(());
    }
    let bufs = [vk::SparseBufferMemoryBindInfo::default().buffer(m.buf).binds(&binds)];
    let info = vk::BindSparseInfo::default().buffer_binds(&bufs);
    let fence = unsafe { c.device.create_fence(&vk::FenceCreateInfo::default(), None) }.map_err(|e| format!("vkCreateFence: {e}"))?;
    let r = {
        let q = c.queue.lock();
        unsafe { c.device.queue_bind_sparse(*q, &[info], fence) }
    };
    let r = r.map_err(|e| format!("vkQueueBindSparse: {e}")).and_then(|_| unsafe { c.device.wait_for_fences(&[fence], true, u64::MAX) }.map_err(|e| format!("bind fence: {e}")));
    unsafe { c.device.destroy_fence(fence, None) };
    r?;
    for b in &binds {
        let p = b.resource_offset / m.page;
        m.backed[(p / 64) as usize].fetch_or(1u64 << (p % 64), Ordering::AcqRel);
    }
    STATS[S_BACKED].fetch_add(binds.len() as u64, Ordering::Relaxed);
    Ok(())
}

/// Runs `record` in a one-off command buffer on the main queue and waits.
fn one_off(record: impl FnOnce(vk::CommandBuffer)) -> Result<(), String> {
    let c = ctx();
    unsafe {
        let pool = c.device.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(c.queue_family), None).map_err(|e| format!("pool: {e}"))?;
        let r = (|| {
            let cmd = c.device.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(pool).command_buffer_count(1)).map_err(|e| format!("cmd: {e}"))?[0];
            c.device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).map_err(|e| format!("begin: {e}"))?;
            record(cmd);
            c.device.end_command_buffer(cmd).map_err(|e| format!("end: {e}"))?;
            let fence = c.device.create_fence(&vk::FenceCreateInfo::default(), None).map_err(|e| format!("fence: {e}"))?;
            let r = c.submit_cmds(&[cmd], &[], &[], fence).and_then(|_| c.device.wait_for_fences(&[fence], true, u64::MAX).map_err(|e| format!("wait: {e}")));
            c.device.destroy_fence(fence, None);
            r
        })();
        c.device.destroy_command_pool(pool, None);
        r
    }
}

/// Copies `data` into the mirror at guest address `at` (backed first).
pub fn upload_now(at: u64, data: &[u8]) -> Result<(), String> {
    let m = MIRROR.get().ok_or("no mirror")?;
    let n = data.len() as u64;
    back(at, n)?;
    let c = ctx();
    let (sbuf, smem, ptr) = c.host_buffer(n.max(4), vk::BufferUsageFlags::TRANSFER_SRC)?;
    unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len()) };
    let r = one_off(|cmd| unsafe {
        c.device.cmd_copy_buffer(cmd, sbuf, m.buf, &[vk::BufferCopy { src_offset: 0, dst_offset: at - MIRROR_BASE, size: n }]);
    });
    unsafe {
        c.device.destroy_buffer(sbuf, None);
        c.device.free_memory(smem, None);
    }
    STATS[S_UPLOADS].fetch_add(1, Ordering::Relaxed);
    r
}

/// Reads `n` bytes of the mirror at guest address `at` (backed).
pub fn readback_now(at: u64, n: u64) -> Result<Vec<u8>, String> {
    let m = MIRROR.get().ok_or("no mirror")?;
    let c = ctx();
    let (dbuf, dmem, ptr) = c.host_buffer(n.max(4), vk::BufferUsageFlags::TRANSFER_DST)?;
    let r = one_off(|cmd| unsafe {
        c.device.cmd_copy_buffer(cmd, m.buf, dbuf, &[vk::BufferCopy { src_offset: at - MIRROR_BASE, dst_offset: 0, size: n }]);
    });
    let out = r.map(|_| unsafe { std::slice::from_raw_parts(ptr, n as usize) }.to_vec());
    unsafe {
        c.device.destroy_buffer(dbuf, None);
        c.device.free_memory(dmem, None);
    }
    STATS[S_READBACKS].fetch_add(1, Ordering::Relaxed);
    out
}

/// Makes the mirror and runs the self-test (after Vulkan initialization,
/// when `wanted`). On failure residency stays off.
pub fn init() {
    if !wanted() || MIRROR.get().is_some() {
        return;
    }
    match make_mirror() {
        Ok(m) => {
            let _ = MIRROR.set(m);
        }
        Err(e) => {
            crate::error!("resident: DS_RESIDENT=1 but no mirror: {e}");
            return;
        }
    }
    ON.store(true, Ordering::Release);
    match self_test() {
        Ok(s) => crate::info!("resident: self-test passed ({s})"),
        Err(e) => {
            crate::error!("resident: self-test failed: {e}; residency off");
            ON.store(false, Ordering::Release);
        }
    }
}

/// Write faults on armed pages of a private region, and a pattern through
/// the mirror's last page and back.
fn self_test() -> Result<String, String> {
    use windows_sys::Win32::System::Memory::{VirtualAlloc, VirtualFree, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE};
    let n = 16 * PAGE;
    let a = unsafe { VirtualAlloc(std::ptr::null(), n as usize, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE) } as u64;
    if a == 0 {
        return Err("VirtualAlloc".into());
    }
    let r = (|| {
        let armed = arm(a, 8 * PAGE);
        if armed != 8 {
            return Err(format!("armed {armed} of 8 pages"));
        }
        let faults = STATS[S_FAULTS].load(Ordering::Relaxed);
        // Writes to pages 0 and 3 (armed) fault and disarm them; page 9 is not armed.
        for i in [0u64, 3, 9] {
            unsafe { std::ptr::write_volatile((a + i * PAGE + 8) as *mut u64, 0x5EED_0000 + i) };
        }
        let taken = STATS[S_FAULTS].load(Ordering::Relaxed) - faults;
        let st: Vec<u8> = (0..10).map(|i| state(a + i * PAGE)).collect();
        if taken != 2 || st[0] != 0 || st[3] != 0 || st[1] != ST_ARMED | ST_VALID || st[9] != 0 {
            return Err(format!("{taken} faults, states {st:?}"));
        }
        if (0..3u64).map(|i| unsafe { std::ptr::read_volatile((a + i * PAGE * 3 + 8) as *const u64) }).collect::<Vec<_>>() != [0x5EED_0000, 0x5EED_0003, 0] {
            return Err("written values".into());
        }
        Ok(())
    })();
    disarm_range(a, a + n);
    unsafe { VirtualFree(a as _, 0, MEM_RELEASE) };
    r?;
    let m = MIRROR.get().ok_or("no mirror")?;
    let at = MIRROR_END - m.page;
    let data: Vec<u8> = (0..m.page).map(|i| (i * 7 + 3) as u8).collect();
    upload_now(at, &data)?;
    let back = readback_now(at, m.page)?;
    if back != data {
        let first = back.iter().zip(&data).position(|(x, y)| x != y).unwrap_or(0);
        return Err(format!("mirror readback differs from byte {first}"));
    }
    Ok(format!("2 write faults on 8 armed pages; {} KiB through the mirror and back", m.page >> 10))
}

/// Status line.
pub fn status() -> String {
    if !wanted() {
        return String::new();
    }
    let v: Vec<u64> = STATS.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    let refused: Vec<String> = REFUSE_NAMES.iter().zip(REFUSED.iter()).map(|(n, c)| format!("{n} {}", c.load(Ordering::Relaxed))).collect();
    format!(
        "    residency ({}): {} bindings from the mirror ({} pages), {} pages uploaded; {} pages armed, {} write faults ({} races), {} volatile, {} host class; {} arming calls ({:.1} us each), {} disarming calls ({:.1} us each); {} host writes ({} pages); {} mirror pages backed, {} uploads, {} readbacks, {} remaps; refused: {}\n",
        if enabled() { "on" } else { "off" },
        v[S_BINDS],
        v[S_BIND_PAGES],
        v[S_CLAIMED],
        v[S_ARMED],
        v[S_FAULTS],
        v[S_RACES],
        v[S_VOLATILE],
        v[S_HOSTED],
        v[S_PROTECTS],
        v[S_PROTECT_NS] as f64 / v[S_PROTECTS].max(1) as f64 / 1000.0,
        v[S_UNPROTECTS],
        v[S_UNPROTECT_NS] as f64 / v[S_UNPROTECTS].max(1) as f64 / 1000.0,
        v[S_HOST_WRITES],
        v[S_HOST_WRITE_PAGES],
        v[S_BACKED],
        v[S_UPLOADS],
        v[S_READBACKS],
        v[S_RESETS],
        refused.join(", ")
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows_sys::Win32::System::Memory::{VirtualAlloc, VirtualFree, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE};

    /// Arming, a simulated write fault and disarming, without the handler.
    #[test]
    fn arm_fault_disarm() {
        ON.store(true, Ordering::Release);
        let n = 8 * PAGE;
        let a = unsafe { VirtualAlloc(std::ptr::null(), n as usize, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE) } as u64;
        assert_ne!(a, 0);
        assert_eq!(arm(a + 10, 2 * PAGE), 3);
        assert_eq!(state(a), ST_ARMED | ST_VALID);
        assert_eq!(state(a + 3 * PAGE), 0);
        assert!(!plain_rw(a, PAGE));
        // The handler's work for a write to page 1.
        assert!(on_fault(a + PAGE + 5, true));
        assert_eq!(state(a + PAGE), 0);
        assert!(plain_rw(a + PAGE, PAGE));
        unsafe { std::ptr::write_volatile((a + PAGE) as *mut u32, 7) };
        // Reads are not ours, nor pages never armed.
        assert!(!on_fault(a, false));
        assert!(on_fault(a + 3 * PAGE, true), "a writable page: retried as a race");
        // Host class pages are not armed; marking disarms.
        mark_host(a, PAGE);
        assert_eq!(state(a), ST_HOST);
        assert!(plain_rw(a, PAGE));
        assert_eq!(arm(a, PAGE), 0);
        // A remap resets armed pages' state (the guest's protection applies).
        remapped(a + 2 * PAGE, PAGE);
        assert_eq!(state(a + 2 * PAGE), 0);
        disarm_range(a, a + n);
        unsafe { VirtualFree(a as _, 0, MEM_RELEASE) };
    }

    /// Claims, submission, and a write that cannot fault, on pages inside
    /// the mirror's span (skipped when that address is taken).
    #[test]
    fn claim_submit_host_write() {
        ON.store(true, Ordering::Release);
        let want = MIRROR_END - (1 << 20);
        let n = 8 * PAGE;
        let a = unsafe { VirtualAlloc(want as _, n as usize, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE) } as u64;
        if a != want {
            return;
        }
        let pages: Vec<u64> = (0..4).map(|i| a + i * PAGE).collect();
        assert_eq!(page(a), Page::Upload);
        assert_eq!(claim_pages(&pages), 4);
        assert_eq!(page(a), Page::Pending);
        assert_eq!(state(a + PAGE), ST_ARMED | ST_VALID);
        // Another claim of a pending page fails.
        assert_eq!(claim_pages(&pages[..1]), 0);
        submitted(&pages, 7);
        assert_eq!(page(a + 2 * PAGE), Page::Ready);
        // A file read over pages 1..2: disarmed and kept out until it ends.
        host_write_begin(a + PAGE + 100, PAGE);
        assert_eq!(page(a + PAGE), Page::Refuse);
        assert!(plain_rw(a + PAGE, 2 * PAGE));
        assert_eq!(claim_pages(&[a + PAGE]), 0);
        host_write_end(a + PAGE + 100, PAGE);
        assert_eq!(page(a + PAGE), Page::Upload);
        assert_eq!(page(a + 3 * PAGE), Page::Ready);
        // A dropped batch's pages need an upload again.
        assert_eq!(claim_pages(&[a + PAGE]), 1);
        dropped(&[a + PAGE]);
        assert_eq!(page(a + PAGE), Page::Upload);
        disarm_range(a, a + n);
        unsafe { VirtualFree(a as _, 0, MEM_RELEASE) };
    }
}
