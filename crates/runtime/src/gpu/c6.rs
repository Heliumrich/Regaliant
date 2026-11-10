//! Front end 3's C6 (FRONTEND3.md "C6"; `DS_C6=1`, opt-in while it is
//! built): shaders resolve their read-only buffers, and the scalar loads on
//! their chains, from the mirror where the resident set says the mirror
//! holds guest memory, else from guest memory through the page table,
//! reporting the range (`rdna::translate::Spec::c6`). This module keeps the
//! resident set: a bit per 64 KiB granule of the mirror's span on the GPU,
//! and its shadow here.
//! - Granules join from the shaders' reports (`admit_round`, after each
//!   coherence pass): outside the GPU arena they are armed first (a CPU write
//!   then faults and ends the arming; the next pass takes such pages:
//!   `hot_runs`), inside it the pass scans them anyway.
//! - Before each batch the submission thread records, on the main queue,
//!   the copies that make the resident set current (`coverage`): newly
//!   admitted granules, and resident pages a take found written since
//!   (`note_invalidated`), then the set's new bits. Compute batches wait for
//!   the main queue's progress at their submission, so one set serves both.
//! - A batch with C6 commands waits for a coherence pass newer than the
//!   epoch its commands want (`kco`): every CPU write before that epoch was
//!   then found by a take, and its page is in the copies.

use super::resident::{MIRROR_BASE, PAGE};
use super::vk::{ctx, Ctx};
use ash::vk;
use parking_lot::Mutex;
use rdna::translate::{C6_FB_BYTES, C6_FB_ENTRIES, C6_GRANULE_SHIFT, C6_RES_WORDS, PT_BASE, PT_END};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

const GRANULE: u64 = 1 << C6_GRANULE_SHIFT;
const WORDS: usize = C6_RES_WORDS as usize;

/// `DS_C6=1` (read once).
pub fn wanted() -> bool {
    static W: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *W.get_or_init(|| std::env::var("DS_C6").ok().as_deref() == Some("1"))
}

/// Stages C6 resolves (`DS_C6_KINDS`: bits 1 compute, 2 vertex, 4 pixel;
/// all by default).
pub fn stage_wanted(stage: rdna::translate::Stage) -> bool {
    static K: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    let k = *K.get_or_init(|| std::env::var("DS_C6_KINDS").ok().and_then(|v| v.parse().ok()).unwrap_or(7));
    let bit = match stage {
        rdna::translate::Stage::Compute => 1,
        rdna::translate::Stage::Fragment => 4,
        _ => 2,
    };
    k & bit != 0
}

/// Whether C6 runs: wanted, front end 2 and the coherence service on, the
/// resident set made.
pub fn on() -> bool {
    wanted() && super::fe2::on() && super::kco::covering() && bufs().is_some()
}

struct Bufs {
    res: vk::Buffer,
    _res_mem: vk::DeviceMemory,
    res_va: u64,
    _fb: vk::Buffer,
    _fb_mem: vk::DeviceMemory,
    fb_va: u64,
    fb_ptr: *mut u32,
}

unsafe impl Send for Bufs {}
unsafe impl Sync for Bufs {}

static BUFS: std::sync::OnceLock<Option<Bufs>> = std::sync::OnceLock::new();

fn bufs() -> Option<&'static Bufs> {
    // (Not before Vulkan is up: front end 2's takes run from the start.)
    if !wanted() || super::vk::try_ctx().is_none() || super::resident::mirror_buffer().is_none() {
        return None;
    }
    BUFS.get_or_init(|| match make() {
        Ok(b) => {
            crate::info!(
                "c6: resident set ({} KiB at device address {:#x}) and feedback ring ({} KiB at {:#x}) made; the mirror at {:#x}, the guest page table at {:#x}",
                WORDS * 4 >> 10,
                b.res_va,
                C6_FB_BYTES >> 10,
                b.fb_va,
                super::resident::mirror_address(),
                super::pagetable::address()
            );
            Some(b)
        }
        Err(e) => {
            crate::error!("c6: {e}; C6 off");
            None
        }
    })
    .as_ref()
}

fn make() -> Result<Bufs, String> {
    let c = ctx();
    if !c.device_addresses {
        return Err("no device addresses".into());
    }
    let bytes = (WORDS * 4) as u64;
    let (res, res_mem) = c.device_buffer(bytes, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST)?;
    super::resident::one_off_now(|cmd| unsafe { c.device.cmd_fill_buffer(cmd, res, 0, bytes, 0) })?;
    let res_va = c.buffer_address(res);
    let (fb, fb_mem, fb_ptr) = c.host_buffer(C6_FB_BYTES, vk::BufferUsageFlags::STORAGE_BUFFER)?;
    unsafe { std::ptr::write_bytes(fb_ptr, 0, C6_FB_BYTES as usize) };
    let fb_va = c.buffer_address(fb);
    Ok(Bufs { res, _res_mem: res_mem, res_va, _fb: fb, _fb_mem: fb_mem, fb_va, fb_ptr: fb_ptr as *mut u32 })
}

/// The address block header's C6 words (`rdna::translate::C6_HDR_*`): the
/// mirror's device address of guest PT_BASE, the resident set's, the
/// feedback ring's.
pub fn header() -> [u32; 6] {
    let Some(b) = bufs() else { return [0; 6] };
    let m = super::resident::mirror_address() + (PT_BASE - MIRROR_BASE);
    [m as u32, (m >> 32) as u32, b.res_va as u32, (b.res_va >> 32) as u32, b.fb_va as u32, (b.fb_va >> 32) as u32]
}

/// The resident set as the batches submitted so far leave it.
static SHADOW: [AtomicU32; WORDS] = [const { AtomicU32::new(0) }; WORDS];
/// Granules admitted, waiting for the copies that make them resident.
static ADMIT: Mutex<Vec<u32>> = Mutex::new(Vec::new());
/// Resident pages to copy again (a take found them written, or guest memory
/// changed without a write mark).
static REUP: Mutex<Vec<u64>> = Mutex::new(Vec::new());
/// Granules to leave the set (their memory unmapped or the mirror's backing
/// dropped): their bits cleared at the next coverage.
static GONE: Mutex<Vec<u32>> = Mutex::new(Vec::new());
/// The feedback ring's entries read so far.
static CURSOR: Mutex<u64> = Mutex::new(0);

/// [reports read, reports lost to the ring's capacity, granules admitted,
/// refused (no import window over them), refused (not armed), pages copied
/// by coverage (admissions), pages copied again (resident pages written),
/// pages left out (held or GPU-written in flight), coverages, ns of
/// coverage, waits for a pass, ns waited, granules gone, hot pages taken,
/// reports by kind 1, 2, 3].
pub static STATS: [AtomicU64; 17] = [const { AtomicU64::new(0) }; 17];

fn granule_of(a: u64) -> Option<u32> {
    (PT_BASE..PT_END).contains(&a).then(|| ((a - PT_BASE) >> C6_GRANULE_SHIFT) as u32)
}

/// Whether granule `g` is in the resident set (as submitted).
pub fn resident(g: u32) -> bool {
    SHADOW.get((g >> 5) as usize).is_some_and(|w| w.load(Ordering::Acquire) >> (g & 31) & 1 != 0)
}

fn resident_page(p: u64) -> bool {
    granule_of(p).is_some_and(resident)
}

/// Front end 2's takes found these pages written (`fe2::cpu_written`):
/// resident ones are copied again before the next batch.
pub fn note_invalidated(pages: &[u64]) {
    if !on() {
        return;
    }
    let mut v: Vec<u64> = pages.iter().copied().filter(|&p| resident_page(p)).collect();
    if !v.is_empty() {
        REUP.lock().append(&mut v);
    }
}

/// Guest [base, base+len) changed without a write mark (file reads, CP
/// writes the CPU performs, mapping changes: `fe2::invalidate`): its
/// resident pages are copied again (or leave the set when no import window
/// holds them then).
pub fn note_invalidated_range(base: u64, len: u64) {
    if !on() || len == 0 {
        return;
    }
    let (Some(g0), Some(g1)) = (granule_of(base), granule_of(base + len - 1)) else { return };
    if !(g0..=g1).any(resident) {
        return;
    }
    let mut v = Vec::new();
    let mut p = base & !(PAGE - 1);
    while p < base + len {
        if resident_page(p) {
            v.push(p);
        }
        p += PAGE;
    }
    REUP.lock().append(&mut v);
}

/// The mirror's backing over [base, base+len) is dropped (`resident`'s
/// unbacking): the granules leave the set.
pub fn note_gone(base: u64, len: u64) {
    if !on() || len == 0 {
        return;
    }
    let (Some(g0), Some(g1)) = (granule_of(base), granule_of(base + len - 1)) else { return };
    let v: Vec<u32> = (g0..=g1).filter(|&g| resident(g)).collect();
    if !v.is_empty() {
        GONE.lock().extend(v);
    }
}

/// Whether any granule of [base, base+len) is resident (the mirror must
/// keep its backing).
pub fn any_resident(base: u64, len: u64) -> bool {
    if !on() || len == 0 {
        return false;
    }
    match (granule_of(base), granule_of(base + len - 1)) {
        (Some(g0), Some(g1)) => (g0..=g1).any(resident),
        _ => false,
    }
}

/// Granules whose arming failed, and the round they may be tried again.
static REFUSED: Mutex<std::collections::BTreeMap<u32, u64>> = Mutex::new(std::collections::BTreeMap::new());
/// `admit_round`s so far.
static ROUNDS: AtomicU64 = AtomicU64::new(0);
/// Resident granules outside the arena found not armed by the last pass.
static HOT: Mutex<Vec<u32>> = Mutex::new(Vec::new());

/// Granules front end 2's CPU path placed for reading, to join the set
/// (`note_placed`; at most PLACED_MAX waiting).
static PLACED: Mutex<Vec<u32>> = Mutex::new(Vec::new());
const PLACED_MAX: usize = 1 << 16;

/// Front end 2 placed guest [lo, hi) for reading (`Exec::fe2_place`): its
/// granules join the resident set (ranges up to 256 MiB), so the shaders
/// resolving their own buffers find what the CPU path reads.
pub fn note_placed(lo: u64, hi: u64) {
    if !on() || hi <= lo || hi - lo > 256 << 20 {
        return;
    }
    let (Some(g0), Some(g1)) = (granule_of(lo), granule_of(hi - 1)) else { return };
    if (g0..=g1).all(resident) {
        return;
    }
    let mut p = PLACED.lock();
    if p.len() < PLACED_MAX {
        p.extend((g0..=g1).filter(|&g| !resident(g)));
    }
}

/// Reads the shaders' reports: the granules they read through the page
/// table, deduplicated into `out`.
fn drain_feedback(out: &mut Vec<u32>) {
    let Some(b) = bufs() else { return };
    let mut cur = CURSOR.lock();
    let count = unsafe { std::ptr::read_volatile(b.fb_ptr) } as u64;
    // (The counter wraps at 2^32: entries are read modulo the ring.)
    let mut c32 = *cur as u32;
    let lost = count.wrapping_sub(c32 as u64) as u32;
    if lost > C6_FB_ENTRIES {
        STATS[1].fetch_add((lost - C6_FB_ENTRIES) as u64, Ordering::Relaxed);
        c32 = (count as u32).wrapping_sub(C6_FB_ENTRIES);
    }
    while c32 != count as u32 {
        let slot = (c32 & (C6_FB_ENTRIES - 1)) as usize;
        let e = unsafe { b.fb_ptr.add(4 + 4 * slot) };
        let kind = unsafe { std::ptr::read_volatile(e.add(3)) };
        if kind == 0 {
            // (Not written yet: read next time.)
            break;
        }
        let a = unsafe { std::ptr::read_volatile(e) } as u64 | (unsafe { std::ptr::read_volatile(e.add(1)) } as u64) << 32;
        let n = unsafe { std::ptr::read_volatile(e.add(2)) } as u64;
        unsafe { std::ptr::write_volatile(e.add(3), 0) };
        STATS[0].fetch_add(1, Ordering::Relaxed);
        if (1..=3).contains(&kind) {
            STATS[13 + kind as usize].fetch_add(1, Ordering::Relaxed);
        }
        if n > 0 && n <= 256 << 20 {
            if let (Some(g0), Some(g1)) = (granule_of(a), granule_of(a + n - 1)) {
                out.extend(g0..=g1);
            }
        }
        c32 = c32.wrapping_add(1);
    }
    *cur = c32 as u64;
    out.sort_unstable();
    out.dedup();
}

/// After a coherence pass (`kco`): the reported granules are admitted
/// (outside the GPU arena armed through the host first, and taken; a granule
/// with any page not armed then, or without one import window over it, is
/// left out until it is reported again).
pub fn admit_round() {
    if !on() {
        return;
    }
    let mut want = Vec::new();
    drain_feedback(&mut want);
    want.append(&mut PLACED.lock());
    want.sort_unstable();
    want.dedup();
    if want.is_empty() {
        return;
    }
    let arena = crate::engine::arena_owner();
    let pass = ROUNDS.fetch_add(1, Ordering::Relaxed);
    // Resident granules a write took out of the arming last pass: armed
    // again (the next write faults again; a hot one stays taken each pass).
    let hot = std::mem::take(&mut *HOT.lock());
    for g in hot.into_iter().take(64) {
        let a = PT_BASE + ((g as u64) << C6_GRANULE_SHIFT);
        let _ = super::fe2::arm_granule(a, a + GRANULE);
    }
    let pending: std::collections::HashSet<u32> = ADMIT.lock().iter().copied().collect();
    let mut ok = Vec::new();
    for g in want {
        if resident(g) || pending.contains(&g) {
            continue;
        }
        let a = PT_BASE + ((g as u64) << C6_GRANULE_SHIFT);
        // (Imported now when no window holds it: the shaders' reads of what
        // the set does not hold go through the guest page table.)
        if !super::exec::import_covers(a, GRANULE) && super::exec::import_span(a, GRANULE).is_none() {
            STATS[3].fetch_add(1, Ordering::Relaxed);
            continue;
        }
        if REFUSED.lock().get(&g).is_some_and(|&until| pass < until) {
            continue;
        }
        if arena == 0 || crate::engine::owner(a) != arena {
            if !super::fe2::arm_granule(a, a + GRANULE) {
                STATS[4].fetch_add(1, Ordering::Relaxed);
                // (Asked again after 64 rounds, not at every pass.)
                REFUSED.lock().insert(g, pass + 64);
                continue;
            }
        }
        // (The mirror's backing for it: copies into unbacked pages vanish.)
        if super::resident::back(a, GRANULE).is_err() {
            STATS[3].fetch_add(1, Ordering::Relaxed);
            continue;
        }
        ok.push(g);
    }
    if !ok.is_empty() {
        STATS[2].fetch_add(ok.len() as u64, Ordering::Relaxed);
        ADMIT.lock().extend(ok);
    }
}

/// Resident pages outside the GPU arena that are not armed now (a write
/// ended their arming, or front end 4 released them): the coherence pass
/// takes them, as runs into `out`.
pub fn hot_runs(out: &mut Vec<(u64, u64)>) {
    out.clear();
    if !on() {
        return;
    }
    let arena = crate::engine::arena_owner();
    let mut hot = Vec::new();
    for (w, word) in SHADOW.iter().enumerate() {
        let mut bits = word.load(Ordering::Acquire);
        while bits != 0 {
            let k = bits.trailing_zeros();
            bits &= bits - 1;
            let g = (w as u32) << 5 | k;
            let a = PT_BASE + ((g as u64) << C6_GRANULE_SHIFT);
            if arena != 0 && crate::engine::owner(a) == arena {
                continue;
            }
            if super::fe2::granule_armed(a, a + GRANULE) {
                continue;
            }
            hot.push(g);
            match out.last_mut() {
                Some(l) if l.1 == a => l.1 = a + GRANULE,
                _ => out.push((a, a + GRANULE)),
            }
        }
    }
    let n: u64 = out.iter().map(|r| (r.1 - r.0) / PAGE).sum();
    STATS[13].fetch_add(n, Ordering::Relaxed);
    *HOT.lock() = hot;
}

/// Coverage command buffers in flight (freed once their fence signals), with
/// the staging ring's bytes they read ([start, end)).
struct Flight {
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
    start: u64,
}

/// The coverage's staging ring: the CPU copies the pages into it when the
/// batch is submitted (after the coherence pass the batch waits for), and
/// the coverage's copies read it, so no import window has to outlive them.
const RING: u64 = 64 << 20;

struct Submitter {
    pool: vk::CommandPool,
    flights: std::collections::VecDeque<Flight>,
    ring: vk::Buffer,
    _ring_mem: vk::DeviceMemory,
    ring_ptr: *mut u8,
    head: u64,
}

unsafe impl Send for Submitter {}

impl Submitter {
    /// Frees the flights that completed (in order: one queue); with `wait`,
    /// waits for the oldest first and frees it alone.
    fn retire(&mut self, c: &Ctx, wait: bool) {
        while let Some(f) = self.flights.front() {
            let done = unsafe { c.device.get_fence_status(f.fence) }.unwrap_or(false);
            if !done {
                if !wait {
                    break;
                }
                let _ = unsafe { c.device.wait_for_fences(&[f.fence], true, u64::MAX) };
            }
            let f = self.flights.pop_front().unwrap();
            unsafe {
                c.device.destroy_fence(f.fence, None);
                c.device.free_command_buffers(self.pool, &[f.cmd]);
            }
            if wait {
                break;
            }
        }
    }

    /// `n` bytes of the ring (n <= RING), waiting for flights when it is full.
    fn alloc(&mut self, c: &Ctx, n: u64) -> u64 {
        loop {
            self.retire(c, false);
            let tail = self.flights.front().map(|f| f.start);
            let at = match tail {
                None => Some(if self.head + n <= RING { self.head } else { 0 }),
                Some(t) if self.head >= t => {
                    if self.head + n <= RING {
                        Some(self.head)
                    } else if n <= t {
                        Some(0)
                    } else {
                        None
                    }
                }
                Some(t) => (self.head + n <= t).then_some(self.head),
            };
            if let Some(a) = at {
                self.head = a + n;
                return a;
            }
            self.retire(c, true);
        }
    }
}

static SUBMITTER: Mutex<Option<Submitter>> = Mutex::new(None);

/// The submission thread, before a batch (in sequence order): a batch with
/// C6 commands wanting epoch `want` (0: none) first waits for a coherence
/// pass that began in that epoch or later; then the copies that make the
/// resident set current and its new bits go to the main queue ahead of the
/// batch (a compute batch waits for the main queue's progress).
pub(super) fn coverage(c: &Ctx, want: u32) -> Result<(), String> {
    if !on() {
        return Ok(());
    }
    if want != 0 && super::kco::covered_epoch() < want {
        let t = std::time::Instant::now();
        STATS[10].fetch_add(1, Ordering::Relaxed);
        super::kco::trigger();
        while super::kco::covered_epoch() < want {
            if t.elapsed() > std::time::Duration::from_secs(5) {
                crate::warn!("c6: no coherence pass for epoch {want} within 5 s (last {})", super::kco::covered_epoch());
                break;
            }
            std::thread::sleep(std::time::Duration::from_micros(50));
        }
        STATS[11].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
    let admits = std::mem::take(&mut *ADMIT.lock());
    let mut reup = std::mem::take(&mut *REUP.lock());
    let gone = std::mem::take(&mut *GONE.lock());
    if admits.is_empty() && reup.is_empty() && gone.is_empty() {
        return Ok(());
    }
    let t = std::time::Instant::now();
    let b = bufs().ok_or("no resident set")?;
    let (mbuf, _) = super::resident::mirror_buffer().ok_or("no mirror")?;
    // Pages to copy: every page of the admitted granules, the written
    // resident pages; held ones (front end 4: the mirror holds their newest
    // words) and those GPU work in flight writes (the mirror will hold its
    // words) left out.
    let mut pages: Vec<u64> = Vec::with_capacity(admits.len() * 16 + reup.len());
    for &g in &admits {
        let a = PT_BASE + ((g as u64) << C6_GRANULE_SHIFT);
        pages.extend((0..GRANULE / PAGE).map(|k| a + k * PAGE));
    }
    let n_admit = pages.len();
    reup.sort_unstable();
    reup.dedup();
    pages.extend(reup.iter().copied().filter(|&p| resident_page(p)));
    pages.sort_unstable();
    pages.dedup();
    let mut left = 0u64;
    let mut dropped: Vec<u32> = Vec::new();
    let mut copied: Vec<u64> = Vec::with_capacity(pages.len());
    // (By runs of consecutive pages: one query each for GPU work in flight
    // and for the mapping, page by page only where a run's answer is mixed.)
    let mut i = 0;
    while i < pages.len() {
        let mut j = i + 1;
        while j < pages.len() && pages[j] == pages[j - 1] + PAGE {
            j += 1;
        }
        let (a, b) = (pages[i], pages[j - 1] + PAGE);
        let busy = super::exec::inflight_latest(&[(a, b - a)]).is_some();
        let readable = crate::mem::os_readable(a, b - a);
        for &p in &pages[i..j] {
            if super::fe2::c4_held(p) || (busy && super::exec::inflight_latest(&[(p, PAGE)]).is_some()) {
                left += 1;
                continue;
            }
            if !readable && !crate::mem::os_readable(p, PAGE) {
                // (Unmapped meanwhile: the granule leaves the set.)
                if let Some(g) = granule_of(p) {
                    dropped.push(g);
                }
                continue;
            }
            copied.push(p);
        }
        i = j;
    }
    dropped.sort_unstable();
    dropped.dedup();
    STATS[5].fetch_add(n_admit as u64, Ordering::Relaxed);
    STATS[6].fetch_add((pages.len() - n_admit.min(pages.len())) as u64, Ordering::Relaxed);
    STATS[7].fetch_add(left, Ordering::Relaxed);
    // The set's words after this coverage.
    let mut words: std::collections::BTreeMap<u32, u32> = std::collections::BTreeMap::new();
    for &g in &admits {
        if dropped.contains(&g) {
            continue;
        }
        let w = g >> 5;
        let v = words.entry(w).or_insert_with(|| SHADOW[w as usize].load(Ordering::Acquire));
        *v |= 1 << (g & 31);
    }
    for &g in gone.iter().chain(dropped.iter()) {
        let w = g >> 5;
        let v = words.entry(w).or_insert_with(|| SHADOW[w as usize].load(Ordering::Acquire));
        *v &= !(1 << (g & 31));
    }
    STATS[12].fetch_add((gone.len() + dropped.len()) as u64, Ordering::Relaxed);
    let mut s = SUBMITTER.lock();
    let sub = match s.as_mut() {
        Some(x) => x,
        None => {
            let pool = unsafe { c.device.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(c.queue_family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER), None) }.map_err(|e| format!("c6 pool: {e}"))?;
            let (ring, ring_mem, ring_ptr) = c.host_buffer(RING, vk::BufferUsageFlags::TRANSFER_SRC)?;
            s.insert(Submitter { pool, flights: std::collections::VecDeque::new(), ring, _ring_mem: ring_mem, ring_ptr, head: 0 })
        }
    };
    // The pages into the staging ring (in pieces of at most the ring).
    let max_pages = (RING / PAGE) as usize;
    let mut copies: Vec<vk::BufferCopy> = Vec::with_capacity(copied.len());
    let mut first_at = u64::MAX;
    for part in copied.chunks(max_pages) {
        let at = sub.alloc(c, part.len() as u64 * PAGE);
        first_at = first_at.min(at);
        for (k, &p) in part.iter().enumerate() {
            let src = at + k as u64 * PAGE;
            unsafe { std::ptr::copy_nonoverlapping(p as *const u8, sub.ring_ptr.add(src as usize), PAGE as usize) };
            match copies.last_mut() {
                Some(lc) if lc.src_offset + lc.size == src && lc.dst_offset + lc.size == p - MIRROR_BASE => lc.size += PAGE,
                _ => copies.push(vk::BufferCopy { src_offset: src, dst_offset: p - MIRROR_BASE, size: PAGE }),
            }
        }
    }
    if first_at == u64::MAX {
        first_at = sub.head;
    }
    let cmd = unsafe { c.device.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(sub.pool).command_buffer_count(1)) }.map_err(|e| format!("c6 command buffer: {e}"))?[0];
    let fence = unsafe { c.device.create_fence(&vk::FenceCreateInfo::default(), None) }.map_err(|e| format!("c6 fence: {e}"))?;
    unsafe {
        c.device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).ok();
        // (After everything before on the queue: work in flight may still
        // read the old words of a page the CPU rewrote only after it, as the
        // title synchronizes.)
        let pre = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::MEMORY_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::TRANSFER_READ);
        c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[pre], &[], &[]);
        if !copies.is_empty() {
            c.device.cmd_copy_buffer(cmd, sub.ring, mbuf, &copies);
        }
        if !words.is_empty() {
            let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
            c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[mb], &[], &[]);
            for (&w, &v) in &words {
                c.device.cmd_update_buffer(cmd, b.res, w as u64 * 4, &v.to_le_bytes());
            }
        }
        let post = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
        c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::ALL_COMMANDS, vk::DependencyFlags::empty(), &[post], &[], &[]);
        c.device.end_command_buffer(cmd).ok();
    }
    // (On the main queue with its progress signal: a compute batch waits
    // for it.)
    c.submit_timeline(false, &[cmd], &[], &[], fence)?;
    sub.flights.push_back(Flight { cmd, fence, start: first_at });
    drop(s);
    for (&w, &v) in &words {
        SHADOW[w as usize].store(v, Ordering::Release);
    }
    STATS[8].fetch_add(1, Ordering::Relaxed);
    STATS[9].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    Ok(())
}

/// Granules in the set now.
fn resident_count() -> u64 {
    SHADOW.iter().map(|w| w.load(Ordering::Relaxed).count_ones() as u64).sum()
}

/// The status line ("" when off).
pub fn status() -> String {
    if !on() {
        return String::new();
    }
    let s: Vec<u64> = STATS.iter().map(|x| x.load(Ordering::Relaxed)).collect();
    format!(
        "    C6 resident set: {} stages bound since boot; {} granules now ({} MiB); reports {} (lost {}; through the page table {}, no window in one piece {}, outside the span {}), granules admitted {}, refused (no import window) {}, refused (not armed) {}; coverages {} ({:.1} us each), pages copied for admissions {} and again for writes {}, left out (held or written in flight) {}; waits for a pass {} ({:.1} ms), granules gone {}, hot pages taken {}\n",
        super::exec::C6_STAGES.load(Ordering::Relaxed),
        resident_count(),
        resident_count() * GRANULE >> 20,
        s[0],
        s[1],
        s[14],
        s[15],
        s[16],
        s[2],
        s[3],
        s[4],
        s[8],
        s[9] as f64 / s[8].max(1) as f64 / 1000.0,
        s[5],
        s[6],
        s[7],
        s[10],
        s[11] as f64 / 1e6,
        s[12],
        s[13]
    )
}

/// Statistics start again (`profile reset`).
pub fn reset() {
    for x in STATS.iter() {
        x.store(0, Ordering::Relaxed);
    }
}
