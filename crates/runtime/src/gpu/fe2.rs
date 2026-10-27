//! Front end 2 (docs/FRONTEND2.md; `DS_FE2=1`): the GPU front end on one
//! GPU address space. Every buffer GPU work uses inside the mirror's span
//! ([MIRROR_BASE, MIRROR_END)) is bound from the mirror M, the sparse buffer
//! `resident` makes (backed on use); this module keeps which pages M holds
//! current and runs the write-back's kernel.
//!
//! - A page is valid when M holds what the CPU last wrote there (as of its
//!   last upload) and every GPU write since: GPU work writes only M, and the
//!   write-back (`rdna::kernels::mirror_scatter`, recorded by the executor
//!   before each CP write the GPU performs and at each batch's end) stores
//!   the dwords it marked into guest memory, so guest memory is current at
//!   batch boundaries for the CPU, the command processors and the image
//!   cache.
//! - CPU writes are found by write watching. A valid page that is not armed
//!   is trusted for the synchronization epoch (`precache::epoch`) a take
//!   found it unwritten in; every take, ours or another's (`precache` applies
//!   them all under one protocol), clears the valid bit of the pages it
//!   reports (`marks_taken`), as do file reads, CP writes the CPU performs
//!   and mapping changes (`invalidate`).
//! - Hosted (`--host`): pages quiet for ARM_QUIET flips (doubled each time an
//!   arming ended in a write, up to ARM_MAX_LEVEL) and found clean by takes
//!   in ARM_USES epochs since, not written by GPU work, inside import
//!   windows at least SETTLE flips old (protecting a new import's pages lost
//!   the device: FINDINGS.md 2026-10-26) are armed read-only through the
//!   host; while the host's bit stays set (no write fault, disarm or mapping
//!   change since) such a page needs no take. Each fault stops the whole
//!   process for 15-50 us: while the host answers more than FAULT_BURST a
//!   flip (loads rewrite memory armed before them), nothing is armed for
//!   ARM_HOLD flips.
//! - Uploads are GPU copies from the import windows in the batch's
//!   prologue, on the main queue after every earlier batch, so after their
//!   write-backs (a copy the CPU made now could predate GPU writes of
//!   batches in flight and undo them in M). A batch's uploads count for
//!   other batches once it is queued (`uploaded`): batches queued later run
//!   after it; until then other batches upload such pages themselves.

use super::resident::{MIRROR_BASE, MIRROR_END};
use super::vk::ctx;
use ash::vk;
use parking_lot::Mutex;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU32, AtomicU64, AtomicU8, Ordering};

const PAGE: u64 = 4096;
/// Page states per chunk: 64 MiB of the span.
const CHUNK_SHIFT: u32 = 26;
const CHUNK_PAGES: usize = 1 << (CHUNK_SHIFT - 12);
const WORDS: usize = CHUNK_PAGES / 64;
const CHUNKS: usize = ((MIRROR_END - MIRROR_BASE) >> CHUNK_SHIFT) as usize;
/// Flips a page stays unwritten before it is armed the first time.
const ARM_QUIET: u64 = 30;
const ARM_MAX_LEVEL: u8 = 6;
/// Epochs a page must have been found clean in since its last write.
const ARM_USES: u8 = 8;
/// Faults a flip that pause arming, and for how many flips.
const FAULT_BURST: u64 = 200;
const ARM_HOLD: u64 = 120;
/// Flips an import window exists before pages in it are armed.
pub const SETTLE: u64 = 60;
/// Runs to check at most this many pages apart get one take (a take's
/// query costs ~0.75 us per MiB but microseconds, often tens of them with
/// other queries in flight, to make).
const TAKE_GAP: u64 = 256;

// The host's armed-page words cover the same 64 pages as ours.
const _: () = assert!((MIRROR_BASE - crate::mem::GUEST_BASE) % (64 * PAGE) == 0);

struct Chunk {
    /// M holds the page's current contents.
    valid: [AtomicU64; WORDS],
    /// Armed by us through the host (the host's bit says whether still).
    armed: [AtomicU64; WORDS],
    /// Written by GPU work (never armed: the write-back stores into it).
    gpu: [AtomicU64; WORDS],
    /// The epoch every valid page of the word was found clean in (a take
    /// over the whole word), else 0.
    wclean: [AtomicU32; WORDS],
    /// Bumped by every invalidation of a page of the word.
    wgen: [AtomicU32; WORDS],
    /// The epoch (low 32 bits) a take found the page clean in.
    clean: Box<[AtomicU32]>,
    /// The flip, plus one, a CPU write to the page was last found.
    wrote: Box<[AtomicU32]>,
    /// Armings of the page that ended in a write (each doubles the quiet
    /// time before the next).
    level: Box<[AtomicU8]>,
    /// Epochs a take found the page clean in since its last write (at most
    /// ARM_USES).
    uses: Box<[AtomicU8]>,
}

static TABLE: [AtomicPtr<Chunk>; CHUNKS] = [const { AtomicPtr::new(std::ptr::null_mut()) }; CHUNKS];
static ON: AtomicBool = AtomicBool::new(false);

/// `DS_FE2=1` (read once: shader specs follow it, so the pipelines
/// precompiled at a session's start match).
pub fn wanted() -> bool {
    static W: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *W.get_or_init(|| std::env::var("DS_FE2").ok().as_deref() == Some("1"))
}

/// Front end 2 runs (wanted, and its self-test passed).
#[inline]
pub fn on() -> bool {
    ON.load(Ordering::Acquire)
}

/// Whether guest [base, base+size) lies in the mirror's span.
#[inline]
pub fn in_span(base: u64, size: u64) -> bool {
    base >= MIRROR_BASE && base.saturating_add(size) <= MIRROR_END
}

fn chunk(addr: u64, create: bool) -> Option<&'static Chunk> {
    if !(MIRROR_BASE..MIRROR_END).contains(&addr) {
        return None;
    }
    let slot = &TABLE[((addr - MIRROR_BASE) >> CHUNK_SHIFT) as usize];
    let p = slot.load(Ordering::Acquire);
    if !p.is_null() {
        return Some(unsafe { &*p });
    }
    if !create {
        return None;
    }
    let new = Box::into_raw(Box::new(Chunk {
        valid: std::array::from_fn(|_| AtomicU64::new(0)),
        armed: std::array::from_fn(|_| AtomicU64::new(0)),
        gpu: std::array::from_fn(|_| AtomicU64::new(0)),
        wclean: std::array::from_fn(|_| AtomicU32::new(0)),
        wgen: std::array::from_fn(|_| AtomicU32::new(0)),
        clean: (0..CHUNK_PAGES).map(|_| AtomicU32::new(0)).collect(),
        wrote: (0..CHUNK_PAGES).map(|_| AtomicU32::new(0)).collect(),
        level: (0..CHUNK_PAGES).map(|_| AtomicU8::new(0)).collect(),
        uses: (0..CHUNK_PAGES).map(|_| AtomicU8::new(0)).collect(),
    }));
    match slot.compare_exchange(std::ptr::null_mut(), new, Ordering::AcqRel, Ordering::Acquire) {
        Ok(_) => Some(unsafe { &*new }),
        Err(cur) => {
            drop(unsafe { Box::from_raw(new) });
            Some(unsafe { &*cur })
        }
    }
}

/// Page index of `addr` in its chunk.
#[inline]
fn idx(addr: u64) -> usize {
    ((addr >> 12) as usize) & (CHUNK_PAGES - 1)
}

/// The bits of pages [a, e) within the 64-page word holding `a` (e at most
/// that word's end).
#[inline]
fn word_mask(a: u64, e: u64) -> u64 {
    let first = idx(a) % 64;
    let n = ((e - a) / PAGE) as usize;
    if n >= 64 {
        u64::MAX
    } else {
        ((1u64 << n) - 1) << first
    }
}

/// The end of the 64-page word holding `a`, at most `hi`.
#[inline]
fn word_end(a: u64, hi: u64) -> u64 {
    ((a & !(64 * PAGE - 1)) + 64 * PAGE).min(hi)
}

/// Counters (`status`).
static STATS: [AtomicU64; 24] = [const { AtomicU64::new(0) }; 24];
const S_ENSURES: usize = 0;
const S_ENSURE_PAGES: usize = 1;
const S_FAST: usize = 2;
const S_TAKES: usize = 3;
const S_TAKE_WRITTEN: usize = 4;
const S_UPLOADS: usize = 5;
const S_UPLOAD_PAGES: usize = 6;
const S_UPLOAD_FAILED: usize = 7;
const S_ARM_RUNS: usize = 8;
const S_ARMED: usize = 9;
const S_ARM_ENDED: usize = 10;
const S_ARM_FAILED: usize = 11;
const S_INVAL_CALLS: usize = 12;
const S_INVAL_TAKEN: usize = 13;
const S_SCATTERS: usize = 14;
const S_CP_MIRROR: usize = 15;
const S_VALIDATED: usize = 16;
const S_RACED: usize = 17;
const S_MEMCPY: usize = 19;
const S_UNSCATTERABLE: usize = 20;
const S_TAKE_NS: usize = 21;
const S_SLOW_NS: usize = 22;
const S_TAKE_FAILED: usize = 23;

/// Placement checks by size (pages: <16, <256, <4096, more): [checks,
/// pages, takes, take ns].
static BY_SIZE: [[AtomicU64; 4]; 4] = [const { [const { AtomicU64::new(0) }; 4] }; 4];

fn size_class(pages: u64) -> usize {
    match pages {
        0..16 => 0,
        16..256 => 1,
        256..4096 => 2,
        _ => 3,
    }
}

/// Counts a write-back the executor recorded.
pub fn note_scatter() {
    STATS[S_SCATTERS].fetch_add(1, Ordering::Relaxed);
}

/// Counts a CP write the GPU performs that also went into the mirror.
pub fn note_cp_mirror() {
    STATS[S_CP_MIRROR].fetch_add(1, Ordering::Relaxed);
}

/// Counts pages copied into the mirror by the CPU (no import window).
pub fn note_memcpy(pages: u64) {
    STATS[S_MEMCPY].fetch_add(pages, Ordering::Relaxed);
}

/// Counts written bindings whose pages no import window covers (their GPU
/// writes cannot go back).
pub fn note_unscatterable() {
    STATS[S_UNSCATTERABLE].fetch_add(1, Ordering::Relaxed);
}

/// Pages of [a, e) (one word) that need no check now in epoch `ep`: valid,
/// and armed (the host's bit still set) or found clean in `ep`.
#[inline]
fn word_ok(c: &Chunk, a: u64, e: u64, ep: u32) -> u64 {
    let w = idx(a) / 64;
    let mask = word_mask(a, e);
    let v = c.valid[w].load(Ordering::Acquire) & mask;
    if v == 0 {
        return 0;
    }
    let ours = c.armed[w].load(Ordering::Acquire) & v;
    let armed = if ours != 0 { ours & crate::hosted::armed_word(a & !(64 * PAGE - 1)) } else { 0 };
    let rest = v & !armed;
    if rest == 0 {
        return armed;
    }
    if c.wclean[w].load(Ordering::Acquire) == ep {
        return v;
    }
    let mut ok = armed;
    let base = w * 64;
    let mut r = rest;
    while r != 0 {
        let b = r.trailing_zeros() as usize;
        r &= r - 1;
        if c.clean[base + b].load(Ordering::Acquire) == ep {
            ok |= 1u64 << b;
        }
    }
    ok
}

thread_local! {
    /// Scratch of `ensure`: runs to check, pages a take reported.
    static SCRATCH: std::cell::RefCell<(Vec<(u64, u64)>, Vec<u64>)> = const { std::cell::RefCell::new((Vec::new(), Vec::new())) };
}

/// Makes M current over the pages of [lo, hi) (page-aligned, inside the
/// span) for GPU work recorded now in epoch `ep` (`precache::epoch`, low
/// bits). Pages not known current and not `mine` (uploaded by the caller's
/// batch: current for it) are checked for CPU writes by a take (armed first
/// where `armable` allows: the import window over them is settled), and
/// those M lacks are uploaded by `upload(a, b)` (the GPU copy of guest
/// [a, b) into M, recorded into the caller's prologue; false: it could not).
/// `gpu_writes`: GPU work will write the range (its pages are never armed).
/// Err: an upload failed (M lacks some of the range).
pub fn ensure(lo: u64, hi: u64, ep: u32, gpu_writes: bool, mine: &dyn Fn(u64) -> bool, upload: &mut dyn FnMut(u64, u64) -> bool, armable: &dyn Fn(u64, u64) -> bool) -> Result<(), &'static str> {
    STATS[S_ENSURES].fetch_add(1, Ordering::Relaxed);
    STATS[S_ENSURE_PAGES].fetch_add((hi - lo) / PAGE, Ordering::Relaxed);
    let class = &BY_SIZE[size_class((hi - lo) / PAGE)];
    class[0].fetch_add(1, Ordering::Relaxed);
    class[1].fetch_add((hi - lo) / PAGE, Ordering::Relaxed);
    SCRATCH.with(|s| {
        let mut s = s.borrow_mut();
        let (bad, written) = &mut *s;
        bad.clear();
        // The pages that need a check, as runs.
        let mut a = lo;
        while a < hi {
            let e = word_end(a, hi);
            let c = chunk(a, true).expect("inside the span");
            let w = idx(a) / 64;
            if gpu_writes {
                c.gpu[w].fetch_or(word_mask(a, e), Ordering::AcqRel);
            }
            let need = word_mask(a, e) & !word_ok(c, a, e, ep);
            let mut r = need;
            while r != 0 {
                let b = r.trailing_zeros() as u64;
                // The run of set bits from b.
                let run = (!(r >> b)).trailing_zeros().min(64 - b as u32) as u64;
                let pa = (a & !(64 * PAGE - 1)) + b * PAGE;
                let pe = pa + run * PAGE;
                r &= !(if run >= 64 { u64::MAX } else { ((1u64 << run) - 1) << b });
                // Pages the caller's batch uploaded need nothing.
                let mut x = pa;
                while x < pe {
                    if mine(x) {
                        x += PAGE;
                        continue;
                    }
                    let mut y = x + PAGE;
                    while y < pe && !mine(y) {
                        y += PAGE;
                    }
                    match bad.last_mut() {
                        Some(l) if l.1 + TAKE_GAP * PAGE >= x => l.1 = y,
                        _ => bad.push((x, y)),
                    }
                    x = y;
                }
            }
            a = e;
        }
        if bad.is_empty() {
            STATS[S_FAST].fetch_add(1, Ordering::Relaxed);
            return Ok(());
        }
        let t0 = std::time::Instant::now();
        let frame = super::FRAMES.load(Ordering::Relaxed);
        for &(a, b) in bad.iter() {
            note_arming_ended(a, b);
            if crate::hosted::on() && !gpu_writes {
                try_arm(a, b, frame, armable);
            }
            // The take: pages written since their last take lose their
            // valid bit (`marks_taken`, for any take's pages).
            written.clear();
            let tt = std::time::Instant::now();
            let r = super::precache::take_written_for(a, b - a, written, None, crate::mem::WW_WHO_FE2);
            let tns = tt.elapsed().as_nanos() as u64;
            STATS[S_TAKE_NS].fetch_add(tns, Ordering::Relaxed);
            STATS[S_TAKES].fetch_add(1, Ordering::Relaxed);
            class[2].fetch_add(1, Ordering::Relaxed);
            class[3].fetch_add(tns, Ordering::Relaxed);
            STATS[S_TAKE_WRITTEN].fetch_add(written.len() as u64, Ordering::Relaxed);
            if r.is_none() {
                STATS[S_TAKE_FAILED].fetch_add(1, Ordering::Relaxed);
                invalidate(a, b - a);
            }
            // Uploads of the pages M lacks.
            let mut x = a;
            while x < b {
                let c = chunk(x, true).expect("inside the span");
                let i = idx(x);
                if c.valid[i / 64].load(Ordering::Acquire) >> (i % 64) & 1 != 0 || mine(x) {
                    x += PAGE;
                    continue;
                }
                let mut y = x + PAGE;
                while y < b && !mine(y) && chunk(y, true).is_some_and(|c| c.valid[idx(y) / 64].load(Ordering::Acquire) >> (idx(y) % 64) & 1 == 0) {
                    y += PAGE;
                }
                if !upload(x, y) {
                    STATS[S_UPLOAD_FAILED].fetch_add(1, Ordering::Relaxed);
                    STATS[S_SLOW_NS].fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    return Err("an upload into the mirror failed");
                }
                STATS[S_UPLOADS].fetch_add(1, Ordering::Relaxed);
                STATS[S_UPLOAD_PAGES].fetch_add((y - x) / PAGE, Ordering::Relaxed);
                x = y;
            }
            // Valid pages are clean in this epoch (uploaded ones become
            // valid when the caller's batch is queued: `uploaded`).
            set_clean(a, b, ep, false);
        }
        STATS[S_SLOW_NS].fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        Ok(())
    })
}

/// Marks pages of [a, b) clean in epoch `ep` (whole words: `wclean` too);
/// `validate`: they become valid as well. A page seen for the first time
/// counts as written now (its quiet time before arming starts here: a
/// take cannot tell what the CPU will write next into memory it just
/// filled).
fn set_clean(a: u64, b: u64, ep: u32, validate: bool) {
    let f = super::FRAMES.load(Ordering::Relaxed) as u32 + 1;
    let mut x = a;
    while x < b {
        let e = word_end(x, b);
        let Some(c) = chunk(x, true) else { break };
        let (i, w) = (idx(x), idx(x) / 64);
        let mask = word_mask(x, e);
        for k in i..i + ((e - x) / PAGE) as usize {
            // A take's first look at the page in this epoch counts as a use.
            if !validate && c.clean[k].swap(ep, Ordering::AcqRel) != ep {
                let u = &c.uses[k];
                if u.load(Ordering::Relaxed) < ARM_USES {
                    u.fetch_add(1, Ordering::Relaxed);
                }
            } else if validate {
                c.clean[k].store(ep, Ordering::Release);
            }
            if c.wrote[k].load(Ordering::Relaxed) == 0 {
                let _ = c.wrote[k].compare_exchange(0, f, Ordering::AcqRel, Ordering::Relaxed);
            }
        }
        if validate {
            c.valid[w].fetch_or(mask, Ordering::AcqRel);
        }
        if mask == u64::MAX {
            c.wclean[w].store(ep, Ordering::Release);
        }
        x = e;
    }
}

/// Pages of [a, b) we armed whose host bit is gone (a write fault, a disarm
/// or a mapping change): no longer ours; the next arming waits longer.
fn note_arming_ended(a: u64, b: u64) {
    let mut x = a;
    while x < b {
        let e = word_end(x, b);
        let Some(c) = chunk(x, false) else { return };
        let w = idx(x) / 64;
        let ours = c.armed[w].load(Ordering::Acquire) & word_mask(x, e);
        if ours != 0 {
            let gone = ours & !crate::hosted::armed_word(x & !(64 * PAGE - 1));
            if gone != 0 {
                c.armed[w].fetch_and(!gone, Ordering::AcqRel);
                let base = w * 64;
                let mut r = gone;
                while r != 0 {
                    let k = r.trailing_zeros() as usize;
                    r &= r - 1;
                    let l = &c.level[base + k];
                    let _ = l.fetch_update(Ordering::AcqRel, Ordering::Acquire, |v| Some((v + 1).min(ARM_MAX_LEVEL)));
                }
                STATS[S_ARM_ENDED].fetch_add(gone.count_ones() as u64, Ordering::Relaxed);
            }
        }
        x = e;
    }
}

/// Writes that cannot fault in progress (file reads into guest memory):
/// their pages are not armed meanwhile (the kernel fails a write into a
/// read-only page instead of faulting).
static HOST_WRITES: Mutex<Vec<(u64, u64)>> = Mutex::new(Vec::new());
static HOST_WRITES_N: AtomicU64 = AtomicU64::new(0);

/// A write that cannot fault begins over guest [base, base+len)
/// (`track::host_write`; its pages were disarmed).
pub fn host_write_begin(base: u64, len: u64) {
    if !on() || len == 0 {
        return;
    }
    HOST_WRITES.lock().push((base, base.saturating_add(len)));
    HOST_WRITES_N.fetch_add(1, Ordering::AcqRel);
    invalidate(base, len);
}

/// That write ended: its pages hold new contents.
pub fn host_write_end(base: u64, len: u64) {
    if !on() || len == 0 {
        return;
    }
    let mut g = HOST_WRITES.lock();
    if let Some(i) = g.iter().position(|&(a, e)| a == base && e == base.saturating_add(len)) {
        g.swap_remove(i);
        HOST_WRITES_N.fetch_sub(1, Ordering::AcqRel);
    }
    drop(g);
    invalidate(base, len);
}

fn host_writing(a: u64, b: u64) -> bool {
    HOST_WRITES_N.load(Ordering::Acquire) != 0 && HOST_WRITES.lock().iter().any(|&(x, e)| x < b && a < e)
}

/// Arms the quiet pages of [a, b) not armed now and not written by GPU
/// work, by runs inside settled import windows (`armable`), plain
/// read-write memory only.
fn try_arm(a: u64, b: u64, frame: u64, armable: &dyn Fn(u64, u64) -> bool) {
    if arming_held(frame) {
        return;
    }
    let quiet = |c: &Chunk, i: usize| {
        let wrote = c.wrote[i].load(Ordering::Acquire) as u64;
        let need = ARM_QUIET << c.level[i].load(Ordering::Acquire);
        wrote != 0 && frame.saturating_sub(wrote - 1) >= need && c.uses[i].load(Ordering::Relaxed) >= ARM_USES
    };
    let mut x = a;
    while x < b {
        // The next run of candidates from x.
        let cand = |p: u64| {
            let Some(c) = chunk(p, false) else { return false };
            let (i, w) = (idx(p), idx(p) / 64);
            let bit = 1u64 << (i % 64);
            c.armed[w].load(Ordering::Acquire) & bit == 0 && c.gpu[w].load(Ordering::Acquire) & bit == 0 && quiet(c, i)
        };
        if !cand(x) {
            x += PAGE;
            continue;
        }
        let mut y = x + PAGE;
        while y < b && cand(y) {
            y += PAGE;
        }
        arm_run(x, y, armable);
        x = y;
    }
}

/// The flip the host's fault count was last sampled at, the count then,
/// and the flip arming is held until.
static FAULT_FLIP: AtomicU64 = AtomicU64::new(0);
static FAULT_BASE: AtomicU64 = AtomicU64::new(0);
static HOLD_UNTIL: AtomicU64 = AtomicU64::new(0);
static HOLDS: AtomicU64 = AtomicU64::new(0);

/// Whether arming waits (faults came in a burst lately).
fn arming_held(frame: u64) -> bool {
    let last = FAULT_FLIP.load(Ordering::Acquire);
    if frame != last && FAULT_FLIP.compare_exchange(last, frame, Ordering::AcqRel, Ordering::Relaxed).is_ok() {
        let now = crate::hosted::faults();
        let before = FAULT_BASE.swap(now, Ordering::AcqRel);
        if now.saturating_sub(before) > FAULT_BURST * frame.saturating_sub(last).max(1) {
            HOLD_UNTIL.store(frame + ARM_HOLD, Ordering::Release);
            HOLDS.fetch_add(1, Ordering::Relaxed);
        }
    }
    frame < HOLD_UNTIL.load(Ordering::Acquire)
}

fn arm_run(x: u64, y: u64, armable: &dyn Fn(u64, u64) -> bool) {
    if !armable(x, y) || host_writing(x, y) {
        return;
    }
    // No mapping change meanwhile (`resident::remap_guard`).
    let _readers = crate::mem::GUEST_READERS.read_recursive();
    let _arming = super::resident::arm_guard();
    if !plain_rw(x, y - x) {
        return;
    }
    STATS[S_ARM_RUNS].fetch_add(1, Ordering::Relaxed);
    if crate::hosted::arm(x, y - x).is_err() {
        STATS[S_ARM_FAILED].fetch_add(1, Ordering::Relaxed);
        return;
    }
    let mut p = x;
    while p < y {
        let e = word_end(p, y);
        if let Some(c) = chunk(p, false) {
            c.armed[idx(p) / 64].fetch_or(word_mask(p, e), Ordering::AcqRel);
        }
        p = e;
    }
    STATS[S_ARMED].fetch_add((y - x) / PAGE, Ordering::Relaxed);
}

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

/// Generations of the words the runs touch, in order (taken right before
/// their batch is queued: `uploaded`).
pub fn generations(runs: &[(u64, u64)], out: &mut Vec<u32>) {
    out.clear();
    for &(a, b) in runs {
        let mut x = a;
        while x < b {
            let e = word_end(x, b);
            out.push(chunk(x, true).map_or(0, |c| c.wgen[idx(x) / 64].load(Ordering::Acquire)));
            x = e;
        }
    }
}

/// The batch whose prologue uploads `runs` was queued: their pages are
/// valid and clean in epoch `ep` (loaded before it was queued), except in
/// words invalidated since `gens` (`generations`) were taken (the upload may
/// have run before that write).
pub fn uploaded(runs: &[(u64, u64)], gens: &[u32], ep: u32) {
    let mut k = 0;
    for &(a, b) in runs {
        let mut x = a;
        while x < b {
            let e = word_end(x, b);
            let g = gens.get(k).copied();
            k += 1;
            if let Some(c) = chunk(x, true) {
                if Some(c.wgen[idx(x) / 64].load(Ordering::Acquire)) == g {
                    set_clean(x, e, ep, true);
                    STATS[S_VALIDATED].fetch_add((e - x) / PAGE, Ordering::Relaxed);
                } else {
                    STATS[S_RACED].fetch_add((e - x) / PAGE, Ordering::Relaxed);
                }
            }
            x = e;
        }
    }
}

/// A write-watch take over guest [base, base+size) (anyone's: `precache`):
/// the pages it reports (None: unknown, all of it) are no longer valid.
pub fn marks_taken(r: Option<()>, base: u64, size: u64, pages: &[u64]) {
    if !on() {
        return;
    }
    match r {
        Some(()) => cpu_written(pages),
        None => invalidate(base, size),
    }
}

/// CPU writes to these pages were found (by a take).
pub fn cpu_written(pages: &[u64]) {
    if !on() || pages.is_empty() {
        return;
    }
    let f = super::FRAMES.load(Ordering::Relaxed) as u32 + 1;
    let mut n = 0;
    for &p in pages {
        let Some(c) = chunk(p, false) else { continue };
        let (i, w) = (idx(p), idx(p) / 64);
        c.valid[w].fetch_and(!(1u64 << (i % 64)), Ordering::AcqRel);
        c.wgen[w].fetch_add(1, Ordering::AcqRel);
        c.wrote[i].store(f, Ordering::Release);
        c.uses[i].store(0, Ordering::Relaxed);
        n += 1;
    }
    STATS[S_INVAL_TAKEN].fetch_add(n, Ordering::Relaxed);
}

/// Guest [base, base+len) may have changed without a write mark (file
/// reads, CP writes the CPU performs, mapping changes): M no longer holds
/// its pages.
pub fn invalidate(base: u64, len: u64) {
    if !on() || len == 0 {
        return;
    }
    let lo = (base & !(PAGE - 1)).max(MIRROR_BASE);
    let hi = base.saturating_add(len).next_multiple_of(PAGE).min(MIRROR_END);
    if lo >= hi {
        return;
    }
    STATS[S_INVAL_CALLS].fetch_add(1, Ordering::Relaxed);
    let f = super::FRAMES.load(Ordering::Relaxed) as u32 + 1;
    let mut x = lo;
    while x < hi {
        let e = word_end(x, hi);
        if let Some(c) = chunk(x, false) {
            let w = idx(x) / 64;
            c.valid[w].fetch_and(!word_mask(x, e), Ordering::AcqRel);
            c.wgen[w].fetch_add(1, Ordering::AcqRel);
            for k in idx(x)..idx(x) + ((e - x) / PAGE) as usize {
                c.wrote[k].store(f, Ordering::Release);
                c.uses[k].store(0, Ordering::Relaxed);
            }
            x = e;
        } else {
            // No chunk: nothing valid in it.
            x = ((x >> CHUNK_SHIFT) + 1) << CHUNK_SHIFT;
        }
    }
}

/// The write-back's pipeline, its one descriptor set (the written-bit map
/// and the counters never change) and the counters (host visible).
struct Scatter {
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    set: vk::DescriptorSet,
    counters: *mut u32,
}

unsafe impl Send for Scatter {}
unsafe impl Sync for Scatter {}

static SCATTER: std::sync::OnceLock<Scatter> = std::sync::OnceLock::new();

fn make_scatter(marks: vk::Buffer) -> Result<Scatter, String> {
    let c = ctx();
    unsafe {
        let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..2)
            .map(|b| vk::DescriptorSetLayoutBinding::default().binding(b).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE))
            .collect();
        let set_layout = c.device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None).map_err(|e| format!("scatter set layout: {e}"))?;
        let layouts = [set_layout];
        let ranges = [vk::PushConstantRange { stage_flags: vk::ShaderStageFlags::COMPUTE, offset: 0, size: 16 }];
        let layout = c.device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts).push_constant_ranges(&ranges), None).map_err(|e| format!("scatter layout: {e}"))?;
        let code = rdna::kernels::mirror_scatter();
        let module = c.device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None).map_err(|e| format!("scatter module: {e}"))?;
        let stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(c"main");
        let ci = vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout);
        let pipeline = super::pcache::with_cache('i', 20, |cache| c.device.create_compute_pipelines(cache, &[ci], None)).map_err(|(_, e)| format!("scatter pipeline: {e}"));
        c.device.destroy_shader_module(module, None);
        let pipeline = pipeline?[0];
        let sizes = [vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 2 }];
        let pool = c.device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&sizes), None).map_err(|e| format!("scatter pool: {e}"))?;
        let set = c.device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&layouts)).map_err(|e| format!("scatter set: {e}"))?[0];
        let bytes = 4 * rdna::kernels::SCATTER_COUNTERS as u64;
        let (cbuf, _cmem, cptr) = c.host_buffer(bytes, vk::BufferUsageFlags::STORAGE_BUFFER)?;
        std::ptr::write_bytes(cptr, 0, bytes as usize);
        let infos = [vk::DescriptorBufferInfo { buffer: marks, offset: 0, range: super::resident::MARK_BYTES }, vk::DescriptorBufferInfo { buffer: cbuf, offset: 0, range: bytes }];
        let writes: Vec<vk::WriteDescriptorSet> = infos
            .iter()
            .enumerate()
            .map(|(k, info)| vk::WriteDescriptorSet::default().dst_set(set).dst_binding(k as u32).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(std::slice::from_ref(info)))
            .collect();
        c.device.update_descriptor_sets(&writes, &[]);
        Ok(Scatter { pipeline, layout, set, counters: cptr as *mut u32 })
    }
}

/// Records the write-back into `cmd` (outside a render pass): a barrier
/// after all earlier work, the scatter over the touched-page list, the
/// list emptied, a barrier before all later work. Returns the pipeline it
/// bound (the caller's bound compute state is gone).
pub fn record_scatter(cmd: vk::CommandBuffer) -> Option<vk::Pipeline> {
    let s = SCATTER.get()?;
    let marks = super::resident::marks()?;
    let (mbuf, _) = super::resident::mirror_buffer()?;
    let c = ctx();
    let m = c.buffer_address(mbuf);
    let t = super::pagetable::address();
    let pc: Vec<u8> = [m as u32, (m >> 32) as u32, t as u32, (t >> 32) as u32].iter().flat_map(|x| x.to_le_bytes()).collect();
    unsafe {
        let all = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::MEMORY_WRITE).dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
        c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::DRAW_INDIRECT, vk::DependencyFlags::empty(), &[all], &[], &[]);
        c.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, s.pipeline);
        c.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, s.layout, 0, &[s.set], &[]);
        c.device.cmd_push_constants(cmd, s.layout, vk::ShaderStageFlags::COMPUTE, 0, &pc);
        c.device.cmd_dispatch_indirect(cmd, marks, rdna::kernels::MARK_LIST_AT as u64 * 4);
        let to_fill = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
        c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[to_fill], &[], &[]);
        c.device.cmd_fill_buffer(cmd, marks, rdna::kernels::MARK_LIST_AT as u64 * 4, 4, 0);
        let after = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE | vk::AccessFlags::HOST_READ);
        c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::ALL_COMMANDS | vk::PipelineStageFlags::HOST, vk::DependencyFlags::empty(), &[after], &[], &[]);
    }
    note_scatter();
    Some(s.pipeline)
}

/// The write-back's counters (pages, dwords, pages without a mapping,
/// overflowed lists), as the GPU left them so far.
fn counters() -> [u32; 4] {
    let Some(s) = SCATTER.get() else { return [0; 4] };
    // SAFETY: host-visible words the kernel adds to atomically.
    std::array::from_fn(|k| unsafe { std::ptr::read_volatile(s.counters.add(k)) })
}

/// Makes the mirror, the written-bit map, the guest page table and the
/// write-back, switches off the old placement mechanisms, and runs the
/// self-test (after Vulkan initialization, when `wanted`). On failure front
/// end 2 stays off.
pub fn init() {
    if !wanted() || on() {
        return;
    }
    // The old placement and coherence mechanisms are off under it, and so
    // are the recording schemes that exist to hide their costs.
    const OFF: [&str; 14] = ["par", "past_waits", "past_runs", "long_runs", "frame_ahead", "forward_copies", "copy_update", "homes", "keep_small", "draw_args_gpu", "gpu_loads", "gpu_chains", "shadow_args", "pre_share"];
    for name in OFF {
        if !super::tune::set(name, 0) {
            crate::warn!("fe2: no switch {name}");
        }
    }
    let made = match super::resident::make_for_fe2() {
        Ok(s) => s,
        Err(e) => {
            crate::error!("fe2: DS_FE2=1 but no mirror: {e}; front end 2 off");
            return;
        }
    };
    if super::pagetable::address() == 0 {
        crate::error!("fe2: no guest page table (device addresses); front end 2 off");
        return;
    }
    let Some(marks) = super::resident::marks() else {
        crate::error!("fe2: no written-bit map; front end 2 off");
        return;
    };
    match make_scatter(marks) {
        Ok(s) => {
            let _ = SCATTER.set(s);
        }
        Err(e) => {
            crate::error!("fe2: write-back kernel: {e}; front end 2 off");
            return;
        }
    }
    match self_test() {
        Ok(s) => {
            ON.store(true, Ordering::Release);
            crate::info!("fe2: front end 2 on ({made}; {s}; hosted arming {}; switched off: {})", if crate::hosted::on() { "on" } else { "off (not under the host)" }, OFF.join(", "));
        }
        Err(e) => crate::error!("fe2: the write-back's self-test failed: {e}; front end 2 off"),
    }
}

/// The write-back over one page: guest memory a private page imported like
/// a window, mapped into the guest page table at the span's last page; the
/// mirror's page a pattern; five dwords marked. The marked ones must land,
/// the others keep what the CPU wrote, and the marks and the touched bit
/// must be cleared.
fn self_test() -> Result<String, String> {
    use windows_sys::Win32::System::Memory::{VirtualAlloc, VirtualFree, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE};
    let c = ctx();
    let marks = super::resident::marks().ok_or("no written-bit map")?;
    let table = super::pagetable::buffer().ok_or("no page table")?;
    let guest = MIRROR_END - PAGE;
    let page = (guest - MIRROR_BASE) / PAGE;
    let pattern: Vec<u8> = (0..1024u32).flat_map(|i| (0x5000_0000 + i).to_le_bytes()).collect();
    super::resident::upload_now(guest, &pattern)?;
    let n = 64 << 10;
    let h = unsafe { VirtualAlloc(std::ptr::null(), n, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE) } as u64;
    if h == 0 {
        return Err("VirtualAlloc".into());
    }
    let r = (|| {
        let words = h as *mut u32;
        for i in 0..1024 {
            unsafe { words.add(i).write_volatile(0xAAAA_AAAA) };
        }
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
        let (hb, hm, hva) = c.import_host(h, PAGE, usage)?;
        let dwords = [0u32, 5, 31, 32, 1023];
        let mut mask = [0u32; 32];
        for d in dwords {
            mask[(d / 32) as usize] |= 1 << (d % 32);
        }
        let mask_bytes: Vec<u8> = mask.iter().flat_map(|x| x.to_le_bytes()).collect();
        let list: Vec<u8> = [1u32, 1, 1, 0, page as u32].iter().flat_map(|x| x.to_le_bytes()).collect();
        let pbit_at = 4 * (rdna::kernels::MARK_PAGE_BITS_AT as u64 + page / 32);
        let pbit = (1u32 << (page % 32)).to_le_bytes();
        let before = counters();
        let (rb, rbm, rbp) = c.host_buffer(256, vk::BufferUsageFlags::TRANSFER_DST)?;
        let r = super::resident::one_off_now(|cmd| unsafe {
            c.device.cmd_update_buffer(cmd, table, page * 8, &hva.to_le_bytes());
            c.device.cmd_update_buffer(cmd, marks, page * 128, &mask_bytes);
            c.device.cmd_update_buffer(cmd, marks, pbit_at, &pbit);
            c.device.cmd_update_buffer(cmd, marks, rdna::kernels::MARK_LIST_AT as u64 * 4, &list);
            let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
            c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::ALL_COMMANDS, vk::DependencyFlags::empty(), &[mb], &[], &[]);
            record_scatter(cmd);
            // What the kernel left in the map, then the table entry cleared.
            c.device.cmd_copy_buffer(cmd, marks, rb, &[vk::BufferCopy { src_offset: page * 128, dst_offset: 0, size: 128 }, vk::BufferCopy { src_offset: pbit_at, dst_offset: 128, size: 4 }, vk::BufferCopy { src_offset: rdna::kernels::MARK_LIST_AT as u64 * 4, dst_offset: 132, size: 4 }]);
            c.device.cmd_update_buffer(cmd, table, page * 8, &0u64.to_le_bytes());
        });
        let after = counters();
        let left: Vec<u32> = (0..35).map(|k| unsafe { std::ptr::read_volatile((rbp as *const u32).add(k)) }).collect();
        unsafe {
            c.device.destroy_buffer(rb, None);
            c.device.free_memory(rbm, None);
            c.device.destroy_buffer(hb, None);
            c.device.free_memory(hm, None);
        }
        r?;
        for i in 0..1024u32 {
            let got = unsafe { words.add(i as usize).read_volatile() };
            let want = if dwords.contains(&i) { 0x5000_0000 + i } else { 0xAAAA_AAAA };
            if got != want {
                return Err(format!("dword {i}: {got:#x}, not {want:#x}"));
            }
        }
        if left[..32].iter().any(|&w| w != 0) {
            return Err("mask words left set".into());
        }
        if left[32] & (1 << (page % 32)) != 0 {
            return Err("the page's touched bit left set".into());
        }
        if left[33] != 0 {
            return Err(format!("the list's count left at {}", left[33]));
        }
        if after[0] != before[0] + 1 || after[1] != before[1] + 5 || after[2] != before[2] {
            return Err(format!("counters {before:?} -> {after:?}"));
        }
        Ok(format!("write-back self-test passed: 5 marked dwords of an imported page written back, 1019 left"))
    })();
    unsafe { VirtualFree(h as _, 0, MEM_RELEASE) };
    r
}

/// One status line ("" when off).
pub fn status() -> String {
    if !on() {
        return String::new();
    }
    let s: Vec<u64> = STATS.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    let k = counters();
    let ms = |ns: u64| ns as f64 / 1e6;
    let sizes: Vec<String> = ["<64K", "<1M", "<16M", ">=16M"]
        .iter()
        .zip(BY_SIZE.iter())
        .map(|(n, c)| {
            let v: Vec<u64> = c.iter().map(|x| x.load(Ordering::Relaxed)).collect();
            format!("{n} {} ({} MiB; {} takes, {:.1} ms)", v[0], (v[1] * PAGE) >> 20, v[2], ms(v[3]))
        })
        .collect();
    format!(
        "    front end 2: {} placement checks ({} pages; {} needed nothing), {} takes ({} pages written, {} failed; {:.1} ms), checking {:.1} ms; uploads {} ({} pages, {} MiB), {} failed, {} pages valid once queued ({} raced a write); hosted arming: {} pages in {} runs ({} failed), {} armings ended by writes, {} holds for fault bursts; invalidated: {} ranges, {} pages by takes; write-backs recorded {}: {} pages, {} dwords, {} pages unmapped (lost), {} lists past their capacity; CP writes also into the mirror {}, pages copied by the CPU {}, written bindings without an import {}\n    front end 2's placement checks by size: {}\n",
        s[S_ENSURES],
        s[S_ENSURE_PAGES],
        s[S_FAST],
        s[S_TAKES],
        s[S_TAKE_WRITTEN],
        s[S_TAKE_FAILED],
        ms(s[S_TAKE_NS]),
        ms(s[S_SLOW_NS]),
        s[S_UPLOADS],
        s[S_UPLOAD_PAGES],
        (s[S_UPLOAD_PAGES] * PAGE) >> 20,
        s[S_UPLOAD_FAILED],
        s[S_VALIDATED],
        s[S_RACED],
        s[S_ARMED],
        s[S_ARM_RUNS],
        s[S_ARM_FAILED],
        s[S_ARM_ENDED],
        HOLDS.load(Ordering::Relaxed),
        s[S_INVAL_CALLS],
        s[S_INVAL_TAKEN],
        s[S_SCATTERS],
        k[0],
        k[1],
        k[2],
        k[3],
        s[S_CP_MIRROR],
        s[S_MEMCPY],
        s[S_UNSCATTERABLE],
        sizes.join(", "),
    )
}
