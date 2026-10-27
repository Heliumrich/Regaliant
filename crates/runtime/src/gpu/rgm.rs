//! Resident GPU-written data, stage W1 (RENDERER.md "Resident GPU-written
//! data"; `tune rgm`, boot).
//!
//! GPU-written buffer data stays in video memory across batches and queues:
//! two sparse buffers over the guest span [BASE, END), the working copy W,
//! which every GPU use inside a resident granule binds (written or
//! read-only), and the reference R, what guest memory holds as far as the GPU
//! side knows. A granule (64 KiB of guest memory) becomes resident when a
//! written binding first covers it: copied in from guest memory into W and R.
//! The batch that writes W stores the dwords it changed (W differs from R)
//! into guest memory and R when it ends (`Exec::rgm_merge`,
//! `rdna::kernels::resident_merge`), so guest memory is as current after each
//! writing batch as with shadows, and the CPU side (the title, the command
//! processors, stage preparation, the image cache) is unchanged.
//!
//! CPU writes: a granule's write marks are taken once per synchronization
//! epoch, by its first use in it (`precache::take_written_for`, which keeps
//! the copy cache and texture regions informed); pages found written, and
//! pages our own writes into guest memory changed (CP writes, fills, file
//! reads, mapping changes, GPU writes by other paths: `invalidate`), are dirty:
//! copied in again from guest memory by the next batch that uses the granule.
//! Copies in are not exclusive: every batch that uses a granule with dirty
//! pages copies them in itself (in its prologue guest memory holds every
//! earlier batch's merged writes, so the copies agree), and the pages stay
//! dirty until such a batch is submitted with no newer write found since
//! (the dirty generation); a granule made resident by a written binding is
//! resident for others once that batch is submitted. Batches that go before
//! it on the queue copy for themselves.

use super::vk::ctx;
use ash::vk;
use parking_lot::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

/// The guest span W and R cover (the title's direct memory and its GPU
/// buffers: `resident::MIRROR_BASE`..`MIRROR_END`).
pub const BASE: u64 = super::resident::MIRROR_BASE;
pub const END: u64 = super::resident::MIRROR_END;
/// A granule: residency, claims and write-mark epochs are per granule, its
/// pages' dirtiness per 4 KiB page (16 bits).
pub const GRANULE: u64 = 64 << 10;
pub const PAGE: u64 = 4096;
/// The whole granule's page mask.
pub const ALL_PAGES: u16 = 0xFFFF;
/// Largest binding placed on W.
pub const MAX_BIND: u64 = 64 << 20;
/// Flips a demoted granule takes the old paths (`Rgm::demote`).
const DEMOTE_FLIPS: u64 = 600;
/// Video memory W and R may take together.
const BUDGET: u64 = 8 << 30;
/// Device memory blocks their sparse pages are bound from.
const BLOCK: u64 = 64 << 20;

/// W and R hold the granule's data as of its last copy-in or merge (a
/// submitted batch made it resident).
const RESIDENT: u64 = 1 << 63;
/// A thread is taking its write marks (`take_lock`).
const TAKING: u64 = 1 << 62;
/// Pages to copy in again (CPU writes taken, our writes into guest memory).
const DIRTY_SHIFT: u32 = 32;
const DIRTY: u64 = (ALL_PAGES as u64) << DIRTY_SHIFT;
/// Bumped whenever pages turn dirty (14 bits, wrapping): a batch that
/// copied them in clears them on submission only if none turned dirty since.
const GEN_SHIFT: u32 = 48;
const GEN: u64 = 0x3FFF << GEN_SHIFT;
/// The synchronization epoch its write marks were last taken in (low 32 bits).
const EPOCH: u64 = 0xFFFF_FFFF;

/// Counters (`status_line`).
pub static STATS: [AtomicU64; 18] = [const { AtomicU64::new(0) }; 18];
pub const S_BIND_W: usize = 0;
pub const S_BIND_R: usize = 1;
pub const S_RESIDENT: usize = 2;
pub const S_COPY_PAGES: usize = 3;
pub const S_COPY_INLINE: usize = 4;
pub const S_TAKES: usize = 5;
pub const S_TAKE_PAGES: usize = 6;
pub const S_TAKE_NS: usize = 7;
pub const S_MERGES: usize = 8;
pub const S_MERGE_RANGES: usize = 9;
pub const S_MERGE_BYTES: usize = 10;
pub const S_INVAL_PAGES: usize = 11;
pub const S_DROPPED: usize = 12;
pub const S_EARLY_MERGES: usize = 13;
pub const S_BACKED_BYTES: usize = 14;
pub const S_DEMOTED: usize = 15;
pub const S_TABLE_UPDATES: usize = 16;
pub const S_CPU_PAGES: usize = 17;
/// Why a binding inside the span took its old path (`refused`).
pub const REFUSE_NAMES: [&str; 11] = [
    "outside the span, unbounded or too large",
    "(unused)",
    "read-only over granules not resident",
    "written earlier in the batch by other paths",
    "not one import",
    "homes or kept windows there",
    "no video memory left",
    "CP writes pending",
    "resident shadows there",
    "no table room",
    "demoted",
];
static REFUSED: [AtomicU64; 11] = [const { AtomicU64::new(0) }; 11];

pub fn refused(why: usize) {
    REFUSED[why.min(REFUSED.len() - 1)].fetch_add(1, Ordering::Relaxed);
}

/// W and R, their sparse memory and the granule table.
pub struct Rgm {
    pub w: vk::Buffer,
    pub r: vk::Buffer,
    pub w_addr: u64,
    pub r_addr: u64,
    /// Sparse page size (the buffers' memory alignment).
    page: u64,
    mem_type: u32,
    /// Device memory blocks and the next free offset in the last one.
    blocks: Mutex<(Vec<vk::DeviceMemory>, u64)>,
    /// One bit per sparse page: bound to memory (in both buffers).
    backed: Box<[AtomicU64]>,
    /// Per granule: RESIDENT, DIRTY pages, their generation and the epoch.
    table: Box<[AtomicU64]>,
    /// Per granule: the flip until which it takes the old paths (`demote`).
    demoted: Box<[std::sync::atomic::AtomicU32]>,
}

/// `tune rgm` asked for at boot (the device then enables sparse buffers).
pub fn wanted() -> bool {
    super::tune::rgm() != 0
}

static RGM: std::sync::OnceLock<Option<Rgm>> = std::sync::OnceLock::new();

/// W and R while `tune rgm` is on (made at the first call).
pub fn get() -> Option<&'static Rgm> {
    if super::tune::rgm() == 0 {
        return None;
    }
    RGM.get_or_init(|| match make() {
        Ok(r) => Some(r),
        Err(e) => {
            crate::warn!("rgm: resident GPU-written data unavailable: {e}");
            None
        }
    })
    .as_ref()
}

fn make() -> Result<Rgm, String> {
    let c = ctx();
    if !c.sparse_buffers {
        return Err("sparse buffers not enabled (DS_RGM must be set at boot)".into());
    }
    if !c.device_addresses {
        return Err("no buffer device addresses".into());
    }
    let size = END - BASE;
    let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::INDIRECT_BUFFER;
    let mut bi = vk::BufferCreateInfo::default().size(size).usage(c.addressable(usage)).flags(vk::BufferCreateFlags::SPARSE_BINDING | vk::BufferCreateFlags::SPARSE_RESIDENCY);
    if c.families.len() > 1 {
        bi = bi.sharing_mode(vk::SharingMode::CONCURRENT).queue_family_indices(&c.families);
    }
    let w = unsafe { c.device.create_buffer(&bi, None) }.map_err(|e| format!("vkCreateBuffer (W, sparse, {} GiB): {e}", size >> 30))?;
    let r = match unsafe { c.device.create_buffer(&bi, None) } {
        Ok(r) => r,
        Err(e) => {
            unsafe { c.device.destroy_buffer(w, None) };
            return Err(format!("vkCreateBuffer (R, sparse, {} GiB): {e}", size >> 30));
        }
    };
    let req = unsafe { c.device.get_buffer_memory_requirements(w) };
    let Some(mem_type) = c.memory_type(req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL) else {
        unsafe {
            c.device.destroy_buffer(w, None);
            c.device.destroy_buffer(r, None);
        }
        return Err("no device-local memory type".into());
    };
    let page = req.alignment.max(PAGE);
    if BLOCK % page != 0 || !page.is_power_of_two() {
        return Err(format!("sparse page size {page:#x} does not divide the memory blocks"));
    }
    let (w_addr, r_addr) = (c.buffer_address(w), c.buffer_address(r));
    if w_addr == 0 || r_addr == 0 {
        return Err("W or R has no device address".into());
    }
    let pages = (size / page) as usize;
    crate::info!(
        "rgm: resident GPU-written data over guest {BASE:#x}..{END:#x}: W and R ({} GiB each, sparse pages of {} KiB, memory type {mem_type}, at most {} GiB of video memory)",
        size >> 30,
        page >> 10,
        BUDGET >> 30
    );
    Ok(Rgm {
        w,
        r,
        w_addr,
        r_addr,
        page,
        mem_type,
        blocks: Mutex::new((Vec::new(), BLOCK)),
        backed: (0..pages.div_ceil(64)).map(|_| AtomicU64::new(0)).collect(),
        table: (0..((END - BASE) / GRANULE) as usize).map(|_| AtomicU64::new(0)).collect(),
        demoted: (0..((END - BASE) / GRANULE) as usize).map(|_| std::sync::atomic::AtomicU32::new(0)).collect(),
    })
}

/// `s` with its dirty generation bumped.
#[inline]
fn bump(s: u64) -> u64 {
    (s & !GEN) | ((s + (1 << GEN_SHIFT)) & GEN)
}

/// The granule of guest address `a` (inside the span).
#[inline]
pub fn granule(a: u64) -> usize {
    ((a - BASE) / GRANULE) as usize
}

/// The guest address of granule `g`.
#[inline]
pub fn granule_addr(g: usize) -> u64 {
    BASE + g as u64 * GRANULE
}

/// The pages of [a, a+n) inside granule `g` as its page mask.
pub fn page_mask(g: usize, a: u64, n: u64) -> u16 {
    let ga = granule_addr(g);
    let lo = a.max(ga);
    let hi = (a + n).min(ga + GRANULE);
    if lo >= hi {
        return 0;
    }
    let (p0, p1) = ((lo - ga) / PAGE, (hi - 1 - ga) / PAGE);
    let bits = (1u32 << (p1 - p0 + 1)) - 1;
    (bits << p0) as u16
}

/// A batch's copy in of a granule: its index, the pages, the dirty
/// generation they were dirty in, and whether the granule was resident
/// (a first copy in makes it resident when the batch is submitted).
#[derive(Clone, Copy, Debug)]
pub struct Claim {
    pub g: usize,
    pub pages: u16,
    pub dgen: u64,
    pub was_resident: bool,
}

/// What a binding finds in a granule (`Rgm::look`).
pub enum Look {
    /// Resident with no dirty page: bind W as it is.
    Clean,
    /// Copy these pages in (all of them for a first copy in) before its use.
    Copy(Claim),
    /// Not resident and not to be made resident by this use.
    Absent,
}

impl Rgm {
    #[inline]
    pub fn state(&self, g: usize) -> u64 {
        self.table[g].load(Ordering::Acquire)
    }

    /// Whether granule `g` is resident (a read-only binding's precondition).
    #[inline]
    pub fn usable(&self, g: usize) -> bool {
        self.state(g) & RESIDENT != 0
    }

    /// Whether granule `g` takes the old paths for now (`demote`).
    #[inline]
    pub fn is_demoted(&self, g: usize) -> bool {
        let until = self.demoted[g].load(Ordering::Relaxed);
        until != 0 && until as u64 > super::FRAMES.load(Ordering::Relaxed)
    }

    /// Granules over [a, a+n) whose W data a use of guest memory in place
    /// needed in the batch that wrote it: they take the old paths for
    /// DEMOTE_FLIPS flips (a merge before every such use costs more than W
    /// saves) and are copied in whole when they come back.
    pub fn demote(&self, a: u64, n: u64) {
        let end = a.saturating_add(n).min(END);
        let a = a.max(BASE);
        if n == 0 || a >= end {
            return;
        }
        let now = super::FRAMES.load(Ordering::Relaxed);
        let until = (now + DEMOTE_FLIPS).min(u32::MAX as u64) as u32;
        for g in granule(a)..=granule(end - 1) {
            let prev = self.demoted[g].swap(until, Ordering::Relaxed);
            if prev as u64 <= now {
                STATS[S_DEMOTED].fetch_add(1, Ordering::Relaxed);
            }
            // Not resident from now on (a batch submitted later that copied
            // it in does not make it resident again: `submitted`).
            self.table[g].fetch_and(!RESIDENT, Ordering::AcqRel);
        }
    }

    /// Holds granules [g0, g1] for a take of their write marks (in ascending
    /// order, so threads cannot deadlock): another thread's take of one of
    /// them is waited for, since its pages are noted only when it ends.
    pub fn take_lock(&self, g0: usize, g1: usize) {
        for g in g0..=g1 {
            let mut spins = 0u32;
            loop {
                let s = self.state(g);
                if s & TAKING == 0 && self.table[g].compare_exchange_weak(s, s | TAKING, Ordering::AcqRel, Ordering::Acquire).is_ok() {
                    break;
                }
                spins += 1;
                if spins < 64 {
                    std::hint::spin_loop();
                } else {
                    std::thread::yield_now();
                }
            }
        }
    }

    /// Lets granules [g0, g1] go after a take (`take_lock`).
    pub fn take_unlock(&self, g0: usize, g1: usize) {
        for g in g0..=g1 {
            self.table[g].fetch_and(!TAKING, Ordering::AcqRel);
        }
    }

    /// Whether granule `g`'s write marks were taken in `epoch`.
    #[inline]
    pub fn checked(&self, g: usize, epoch: u64) -> bool {
        self.state(g) & EPOCH == epoch & EPOCH
    }

    /// Granules [g0, g1] had their write marks taken in `epoch`; `written`
    /// lists the pages found written (guest addresses; None: unknown, all).
    pub fn note_taken(&self, g0: usize, g1: usize, epoch: u64, written: Option<&[u64]>) {
        for g in g0..=g1 {
            let ga = granule_addr(g);
            let mask: u16 = match written {
                None => ALL_PAGES,
                Some(w) => w.iter().filter(|&&p| p >= ga && p < ga + GRANULE).fold(0, |m, &p| m | 1 << ((p - ga) / PAGE)),
            };
            let _ = self.table[g].fetch_update(Ordering::AcqRel, Ordering::Acquire, |s| {
                let s = (s & !EPOCH) | (epoch & EPOCH);
                Some(if mask == 0 { s } else { bump(s) | ((mask as u64) << DIRTY_SHIFT) })
            });
        }
    }

    /// Granule `g` for a binding (`make`: a written binding, which makes a
    /// granule not resident yet resident: copied in whole).
    pub fn look(&self, g: usize, make: bool) -> Look {
        if self.is_demoted(g) {
            return Look::Absent;
        }
        let s = self.state(g);
        let dgen = s & GEN;
        if s & RESIDENT == 0 {
            return if make { Look::Copy(Claim { g, pages: ALL_PAGES, dgen, was_resident: false }) } else { Look::Absent };
        }
        match ((s & DIRTY) >> DIRTY_SHIFT) as u16 {
            0 => Look::Clean,
            pages => Look::Copy(Claim { g, pages, dgen, was_resident: true }),
        }
    }

    /// The batch holding `claims` was submitted: its copies precede later
    /// batches. Granules it copied in whole are resident (unless demoted
    /// meanwhile); the pages it copied are clean unless pages turned dirty
    /// since its look.
    pub fn submitted(&self, claims: &[Claim]) {
        for cl in claims {
            let demoted = self.is_demoted(cl.g);
            let _ = self.table[cl.g].fetch_update(Ordering::AcqRel, Ordering::Acquire, |s| {
                let mut s = s;
                if !cl.was_resident && !demoted {
                    s |= RESIDENT;
                }
                if s & GEN == cl.dgen {
                    s &= !((cl.pages as u64) << DIRTY_SHIFT);
                }
                Some(s)
            });
            if !cl.was_resident {
                STATS[S_RESIDENT].fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// The batch holding `claims` was dropped: its copies never ran (the
    /// pages stay dirty, a granule it was to make resident stays not).
    pub fn dropped(&self, claims: &[Claim]) {
        STATS[S_DROPPED].fetch_add(claims.len() as u64, Ordering::Relaxed);
    }

    /// Guest [a, a+n) was written other than through W (CP writes, fills,
    /// file reads, mapping changes, GPU writes by other paths): the
    /// resident or claimed granules' pages there are copied in again by
    /// their next use.
    pub fn invalidate(&self, a: u64, n: u64) -> u64 {
        let d = self.invalidate_quiet(a, n);
        STATS[S_INVAL_PAGES].fetch_add(d, Ordering::Relaxed);
        d
    }

    /// `invalidate` without counting the pages as dirtied by our writes.
    pub fn invalidate_quiet(&self, a: u64, n: u64) -> u64 {
        let end = a.saturating_add(n).min(END);
        let a = a.max(BASE);
        if n == 0 || a >= end {
            return 0;
        }
        let mut dirtied = 0;
        for g in granule(a)..=granule(end - 1) {
            let mask = page_mask(g, a, end - a);
            let mut changed = false;
            let _ = self.table[g].fetch_update(Ordering::AcqRel, Ordering::Acquire, |s| {
                if s & RESIDENT == 0 {
                    changed = false;
                    return None;
                }
                changed = true;
                Some(bump(s) | ((mask as u64) << DIRTY_SHIFT))
            });
            if changed {
                dirtied += mask.count_ones() as u64;
            }
        }
        dirtied
    }

    /// Binds device memory to W's and R's sparse pages over granules
    /// [g0, g1] that have none yet (synchronously). False: over the budget.
    pub fn back(&self, g0: usize, g1: usize) -> Result<bool, String> {
        let first = ((granule_addr(g0) - BASE) / self.page) as usize;
        let last = ((granule_addr(g1) + GRANULE - 1 - BASE) / self.page) as usize;
        let unbacked = |p: usize| self.backed[p / 64].load(Ordering::Acquire) & (1u64 << (p % 64)) == 0;
        if !(first..=last).any(unbacked) {
            return Ok(true);
        }
        let c = ctx();
        let mut g = self.blocks.lock();
        // (Under the lock: another thread may have backed them meanwhile.)
        let newly: Vec<usize> = (first..=last).filter(|&p| unbacked(p)).collect();
        if newly.is_empty() {
            return Ok(true);
        }
        if STATS[S_BACKED_BYTES].load(Ordering::Relaxed) + 2 * self.page * newly.len() as u64 > BUDGET {
            return Ok(false);
        }
        let mut binds = [Vec::new(), Vec::new()];
        for &p in &newly {
            for b in &mut binds {
                if g.1 + self.page > BLOCK {
                    let mut fl = vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
                    let ai = vk::MemoryAllocateInfo::default().allocation_size(BLOCK).memory_type_index(self.mem_type).push_next(&mut fl);
                    let mem = unsafe { c.device.allocate_memory(&ai, None) }.map_err(|e| format!("vkAllocateMemory (rgm block): {e}"))?;
                    g.0.push(mem);
                    g.1 = 0;
                }
                let mem = *g.0.last().expect("block");
                b.push(vk::SparseMemoryBind::default().resource_offset(p as u64 * self.page).size(self.page).memory(mem).memory_offset(g.1));
                g.1 += self.page;
            }
        }
        let bufs = [vk::SparseBufferMemoryBindInfo::default().buffer(self.w).binds(&binds[0]), vk::SparseBufferMemoryBindInfo::default().buffer(self.r).binds(&binds[1])];
        let info = vk::BindSparseInfo::default().buffer_binds(&bufs);
        let fence = unsafe { c.device.create_fence(&vk::FenceCreateInfo::default(), None) }.map_err(|e| format!("vkCreateFence: {e}"))?;
        let r = {
            let q = c.queue.lock();
            unsafe { c.device.queue_bind_sparse(*q, &[info], fence) }
        };
        let r = r.map_err(|e| format!("vkQueueBindSparse: {e}")).and_then(|_| unsafe { c.device.wait_for_fences(&[fence], true, u64::MAX) }.map_err(|e| format!("bind fence: {e}")));
        unsafe { c.device.destroy_fence(fence, None) };
        r?;
        for &p in &newly {
            self.backed[p / 64].fetch_or(1u64 << (p % 64), Ordering::AcqRel);
        }
        STATS[S_BACKED_BYTES].fetch_add(2 * self.page * newly.len() as u64, Ordering::Relaxed);
        Ok(true)
    }
}

/// Merges before a use in place and our writes that dirtied resident pages,
/// by call site: [merges, invalidations, pages dirtied] (status, while on).
static SITES: Mutex<Option<std::collections::HashMap<(&'static str, u32), [u64; 3]>>> = Mutex::new(None);

/// Counts an event at the caller's call site (`k`: 0 a merge before a use in
/// place, 1 an invalidation dirtying `pages`).
#[track_caller]
pub fn note_site(k: usize, pages: u64) {
    note_site_at(std::panic::Location::caller(), k, pages);
}

/// `note_site` at a given site.
pub fn note_site_at(l: &'static std::panic::Location<'static>, k: usize, pages: u64) {
    let mut g = SITES.lock();
    let e = g.get_or_insert_with(Default::default).entry((l.file(), l.line())).or_insert([0; 3]);
    e[k] += 1;
    e[2] += pages;
}

/// Diagnostics: the first 40 writers by other paths (shader, queue) that
/// dirtied 64 or more resident pages at once.
pub fn note_big_writer(shader: u64, queue: &str, a: u64, n: u64, pages: u64) {
    static SEEN: Mutex<Option<std::collections::HashSet<(u64, u64)>>> = Mutex::new(None);
    let mut g = SEEN.lock();
    let s = g.get_or_insert_with(Default::default);
    if s.len() < 40 && s.insert((shader, a)) {
        crate::info!("rgm: {queue} shader {shader:#x} writes {a:#x}+{n:#x} other than through W: {pages} resident pages dirtied");
    }
}

/// Our writes into guest memory (`precache::invalidate`'s callers: CP
/// writes, file reads, mapping changes): resident pages there are dirty.
#[track_caller]
pub fn invalidate(a: u64, n: u64) {
    if let Some(r) = RGM.get().and_then(|r| r.as_ref()) {
        let d = r.invalidate(a, n);
        if d > 0 {
            note_site(1, d);
        }
    }
}

/// Pages a write-mark take (anyone's: the copy cache, homes, textures, ours)
/// found written by the CPU: dirty in their resident granules (the take
/// reset their marks, so no later take sees them).
pub fn cpu_written(pages: &[u64]) {
    let Some(r) = RGM.get().and_then(|r| r.as_ref()) else { return };
    let mut n = 0;
    for &p in pages {
        if p >= BASE && p < END {
            n += r.invalidate_quiet(p, PAGE);
        }
    }
    STATS[S_CPU_PAGES].fetch_add(n, Ordering::Relaxed);
}

/// A submitted batch's writes outside its W writes (`w`): resident pages
/// there are dirty again (they were dirtied when recorded, but a copy in by
/// a batch submitted earlier may have run before them).
pub fn invalidate_except(writes: &super::ranges::Ranges, w: &super::ranges::Ranges) {
    let Some(r) = RGM.get().and_then(|r| r.as_ref()) else { return };
    for (a, n) in writes.iter() {
        if a.saturating_add(n) <= BASE || a >= END || w.contains(a, n) {
            continue;
        }
        let d = r.invalidate(a, n);
        if d > 0 {
            note_site(1, d);
        }
    }
}

/// The merge kernel (`rdna::kernels::resident_merge`): binding 0 its table;
/// push constants {at, n, W, R}.
pub fn merge_pipe() -> Result<&'static (vk::Pipeline, vk::PipelineLayout, vk::DescriptorSetLayout), String> {
    static PIPE: std::sync::OnceLock<Result<(vk::Pipeline, vk::PipelineLayout, vk::DescriptorSetLayout), String>> = std::sync::OnceLock::new();
    PIPE.get_or_init(|| {
        let c = ctx();
        unsafe {
            let bindings = [vk::DescriptorSetLayoutBinding::default().binding(0).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE)];
            // Pushed (no descriptor-pool space: merges are not reserved for).
            let set_layout = c
                .device
                .create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().flags(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR).bindings(&bindings), None)
                .map_err(|e| format!("rgm merge set layout: {e}"))?;
            let layouts = [set_layout];
            let ranges = [vk::PushConstantRange { stage_flags: vk::ShaderStageFlags::COMPUTE, offset: 0, size: 24 }];
            let layout = c.device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts).push_constant_ranges(&ranges), None).map_err(|e| format!("rgm merge layout: {e}"))?;
            let code = rdna::kernels::resident_merge();
            let module = c.device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None).map_err(|e| format!("rgm merge module: {e}"))?;
            let stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(c"main");
            let ci = vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout);
            let pipeline = super::pcache::with_cache('i', 10, |cache| c.device.create_compute_pipelines(cache, &[ci], None)).map_err(|(_, e)| format!("rgm merge pipeline: {e}"));
            c.device.destroy_shader_module(module, None);
            Ok((pipeline?[0], layout, set_layout))
        }
    })
    .as_ref()
    .map_err(|e| e.clone())
}

/// One status line ("" while off).
pub fn status_line(flips: u64) -> String {
    if RGM.get().and_then(|r| r.as_ref()).is_none() {
        return String::new();
    }
    let s = |i: usize| STATS[i].load(Ordering::Relaxed);
    let per = |x: u64| x as f64 / flips.max(1) as f64;
    let refused: Vec<String> = REFUSE_NAMES.iter().zip(&REFUSED).filter(|(_, c)| c.load(Ordering::Relaxed) > 0).map(|(n, c)| format!("{n} {}", c.load(Ordering::Relaxed))).collect();
    format!(
        "    resident GPU-written data (rgm): bindings on W {:.1} written / {:.1} read a flip ({:.1} indirect dispatches' arguments); {} granules resident ({} MiB of video memory for W and R); copy-ins {:.1} pages a flip ({} copies recorded in place); write-mark takes {:.1} a flip ({:.0} us each, {:.1} pages found written a flip); merges {:.1} a flip ({:.1} ranges, {:.2} MiB a flip, {} before a use in place); pages dirtied by our writes {:.1} a flip, by CPU writes others' takes found {:.1}; claims dropped {}; granules demoted {}; old paths: {}\n",
        per(s(S_BIND_W)),
        per(s(S_BIND_R)),
        per(super::exec::RGM_ARGS.load(Ordering::Relaxed)),
        s(S_RESIDENT),
        s(S_BACKED_BYTES) >> 20,
        per(s(S_COPY_PAGES)),
        s(S_COPY_INLINE),
        per(s(S_TAKES)),
        s(S_TAKE_NS) as f64 / 1e3 / s(S_TAKES).max(1) as f64,
        per(s(S_TAKE_PAGES)),
        per(s(S_MERGES)),
        per(s(S_MERGE_RANGES)),
        per(s(S_MERGE_BYTES)) / (1u64 << 20) as f64,
        s(S_EARLY_MERGES),
        per(s(S_INVAL_PAGES)),
        per(s(S_CPU_PAGES)),
        s(S_DROPPED),
        s(S_DEMOTED),
        if refused.is_empty() { "none".to_string() } else { refused.join(", ") }
    ) + &sites_line(flips)
}

/// The sites of merges before uses in place and of our writes that dirtied
/// resident pages, the busiest first.
fn sites_line(flips: u64) -> String {
    let g = SITES.lock();
    let Some(m) = g.as_ref() else { return String::new() };
    let mut v: Vec<(&(&'static str, u32), &[u64; 3])> = m.iter().collect();
    v.sort_by(|a, b| (b.1[0] + b.1[1]).cmp(&(a.1[0] + a.1[1])));
    let per = |x: u64| x as f64 / flips.max(1) as f64;
    let parts: Vec<String> = v
        .iter()
        .take(12)
        .map(|((f, l), e)| format!("{}:{l} merges {:.1} / dirtied {:.1} ({:.0} pages) a flip", f.rsplit(['/', '\\']).next().unwrap_or(f), per(e[0]), per(e[1]), per(e[2])))
        .collect();
    format!("    rgm sites: {}\n", parts.join("; "))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn page_masks() {
        let g = 3;
        let ga = granule_addr(g);
        assert_eq!(page_mask(g, ga, GRANULE), ALL_PAGES);
        assert_eq!(page_mask(g, ga + 4096, 1), 0b10);
        assert_eq!(page_mask(g, ga + 4095, 2), 0b11);
        assert_eq!(page_mask(g, ga - 100, 200), 0b1);
        assert_eq!(page_mask(g, ga + GRANULE - 1, 100), 0x8000);
        assert_eq!(page_mask(g, ga + GRANULE, 100), 0);
    }
}
