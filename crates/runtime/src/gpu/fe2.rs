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
//!   in ARM_USES epochs since, inside import windows at least SETTLE flips
//!   old (protecting a new import's pages lost the device: FINDINGS.md
//!   2026-10-26) are armed read-only through the host; while the host's bit
//!   stays set (no write fault, disarm or mapping change since) such a page
//!   needs no take. Pages GPU work writes are not armed (they are the
//!   pages the GPU writes through their imports; `DS_FE2_ARM_GPU=1`: armed
//!   too, as GPU writes do not fault and M holds them). Each fault
//!   stops the whole process for 15-50 us: a thread reads the faults the
//!   host answered (`watch`), and a 64 MiB chunk with CHUNK_BURST of them
//!   within BURST_MS (or with some while all chunks together had ALL_BURST:
//!   a load rewriting memory armed before it) has its armed pages disarmed
//!   and its arming held CHUNK_HOLD flips (doubled for each burst within
//!   HOLD_MEMORY flips of its last hold, up to CHUNK_HOLD_MAX_LEVEL); the
//!   other chunks keep theirs.
//! - A take only spares uploads of pages that did not change: small runs of
//!   pages found written within HOT_FLIPS flips (the title's rings of
//!   per-draw data, rewritten every frame) are uploaded without one.
//! - With `DS_FE2_DEFER=1` checks wait for the batch's close (`ensure` with
//!   a list, `resolve`; off by default):
//!   the runs a batch's bindings need checked count as its own meanwhile,
//!   and at its close they are merged into few takes (write-watch queries
//!   queue behind each other in the kernel: ~13 us each with the parallel
//!   workers taking too) whose uploads go into its prologue, which runs
//!   before all of its work.
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
/// Faults on armed pages within BURST_MS that disarm a chunk (its own, or
/// all chunks' together while it had some), the flips its arming then
/// waits, and how a burst within HOLD_MEMORY flips of the chunk's last hold
/// doubles that (up to CHUNK_HOLD_MAX_LEVEL times).
const CHUNK_BURST: u32 = 16;
const ALL_BURST: u32 = 64;
const BURST_MS: u128 = 100;
const CHUNK_HOLD: u64 = 60;
/// (5 until 2026-10-27 late night: holds of up to 1,920 flips after a load
/// left its pools unarmed long after it, their ranges taken every epoch;
/// the pages' own doubling keeps rewritten ones unarmed.)
const CHUNK_HOLD_MAX_LEVEL: u8 = 2;
const HOLD_MEMORY: u64 = 600;
/// Flips an import window exists before pages in it are armed.
pub const SETTLE: u64 = 60;
/// Runs of at most HOT_PAGES pages found written within HOT_FLIPS flips are
/// uploaded without a take (larger ones, as loads write them, would be
/// uploaded again at every epoch where a take finds them unchanged).
const HOT_FLIPS: u64 = 8;
const HOT_PAGES: u64 = 16;
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
    /// Written by GPU work (never armed, unless `DS_FE2_ARM_GPU=1`).
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
    /// Epochs a take looked at the word in since a write to one of its pages
    /// was last found (at most ARM_USES), and the last epoch counted. (Per
    /// page until 2026-10-27 late night: every take over a pool of ~113 MB
    /// then updated ~27,000 pages one by one.)
    uses: [AtomicU8; WORDS],
    use_ep: [AtomicU32; WORDS],
    /// Pages whose `wrote` is set (a page seen for the first time counts as
    /// written then: `set_clean`).
    seen: [AtomicU64; WORDS],
    /// Bumped by every invalidation in the chunk and every arming that
    /// ended there (the summary below goes with it).
    cgen: AtomicU32,
    /// The chunk's summary: every page of it needed no check as of (`cgen`
    /// << 32 | epoch, or SUM_ARMED for every page armed: any epoch) and the
    /// host's events of its leaf (`hosted::leaf_events`); 0: none.
    sum: AtomicU64,
    sum_host: AtomicU64,
    /// No page of the chunk is armed before this flip (after a burst of
    /// faults: `burst`), and the doublings of its last hold.
    hold_until: AtomicU64,
    hold_level: AtomicU8,
}

/// `Chunk::sum`'s epoch part when every page of the chunk is armed.
const SUM_ARMED: u32 = u32::MAX;

/// Ranges per thread whose last check found every page fine (`RANGES`).
const RANGE_CACHE: usize = 32;
/// Ranges over more chunks or host leaves than these are not kept there.
const FINE_CHUNKS: usize = 8;
const FINE_LEAVES: usize = 5;

/// A range a thread found needing no check: the generations of the chunks
/// it covers (`Chunk::cgen`: every invalidation and ended arming there bumps
/// them) and the host's events of its leaves (`hosted::leaf_events`) as
/// they were before that check, and its epoch (SUM_ARMED: every page was
/// armed, any epoch). While they are unchanged it still needs none. (Until
/// 2026-10-27 late night one generation over the whole span: queue 0x50's
/// pools of ~113 MB were scanned again after any invalidation anywhere,
/// ~3 us a binding.)
#[derive(Clone, Copy)]
struct Fine {
    lo: u64,
    hi: u64,
    gens: [u32; FINE_CHUNKS],
    host: [u64; FINE_LEAVES],
    ep: u32,
    /// Its words were marked GPU-written (a written binding's check).
    marked: bool,
}

thread_local! {
    /// This thread's ranges found fine, most recent first.
    static RANGES: std::cell::RefCell<Vec<Fine>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// The generations of the chunks over [lo, hi) and the host's events of its
/// leaves (None: more than FINE_CHUNKS or FINE_LEAVES of them).
#[inline]
fn snapshot(lo: u64, hi: u64) -> Option<([u32; FINE_CHUNKS], [u64; FINE_LEAVES])> {
    let (c0, c1) = (((lo - MIRROR_BASE) >> CHUNK_SHIFT) as usize, ((hi - 1 - MIRROR_BASE) >> CHUNK_SHIFT) as usize);
    if c1 - c0 >= FINE_CHUNKS {
        return None;
    }
    let mut gens = [0u32; FINE_CHUNKS];
    for (k, g) in (c0..=c1).zip(gens.iter_mut()) {
        let p = TABLE[k].load(Ordering::Acquire);
        // (A chunk made later starts at generation 0.)
        if !p.is_null() {
            *g = unsafe { &*p }.cgen.load(Ordering::Acquire);
        }
    }
    let mut host = [0u64; FINE_LEAVES];
    let mut a = lo;
    let mut k = 0;
    while a < hi {
        if k == FINE_LEAVES {
            return None;
        }
        host[k] = crate::hosted::leaf_events(a);
        k += 1;
        a = crate::hosted::leaf_end(a);
    }
    Some((gens, host))
}

/// Whether [lo, hi) lies in a range this thread found fine and nothing
/// changed there since.
#[inline]
fn range_fine(lo: u64, hi: u64, ep: u32, marked: bool) -> bool {
    RANGES.with(|r| {
        let mut r = r.borrow_mut();
        let Some(i) = r.iter().position(|f| f.lo <= lo && hi <= f.hi && (f.marked || !marked) && (f.ep == ep || f.ep == SUM_ARMED) && snapshot(f.lo, f.hi).is_some_and(|(g, h)| g == f.gens && h == f.host)) else {
            if callers_on() {
                // Why not (the closest reason among the ranges covering it).
                let covering: Vec<&Fine> = r.iter().filter(|f| f.lo <= lo && hi <= f.hi && (f.marked || !marked)).collect();
                let why = if covering.is_empty() {
                    C_NO_RANGE
                } else if !covering.iter().any(|f| f.ep == ep || f.ep == SUM_ARMED) {
                    C_EPOCH
                } else if covering.iter().any(|f| snapshot(f.lo, f.hi).is_some_and(|(g, _)| g == f.gens)) {
                    C_HOST
                } else {
                    C_GENS
                };
                caller_count(why, 1);
            }
            return false;
        };
        if i > 0 {
            let e = r.remove(i);
            r.insert(0, e);
        }
        true
    })
}

/// `DS_FE2_CALLERS=1` (diagnosis): placement checks counted by caller
/// (`set_caller`), with why the range cache missed.
pub fn callers_on() -> bool {
    static W: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *W.get_or_init(|| std::env::var("DS_FE2_CALLERS").ok().as_deref() == Some("1"))
}

const CALLER_NAMES: [&str; 4] = ["graphics CP", "its parallel workers", "queue 0x50", "other compute queues"];
/// Per caller: [checks, range-cache hits, misses with no range covering,
/// misses on another epoch, misses on chunk generations, misses on host
/// events, words scanned, takes, TSC cycles in all, placements (the
/// executor's `fe2_place`), their TSC cycles].
static BY_CALLER: [[AtomicU64; 11]; 4] = [const { [const { AtomicU64::new(0) }; 11] }; 4];
const C_PLACES: usize = 9;
const C_PLACE_CYCLES: usize = 10;

/// Clears the checks by caller (`profile reset`).
pub fn reset_callers() {
    for c in BY_CALLER.iter().flatten() {
        c.store(0, Ordering::Relaxed);
    }
}

/// A placement took `cycles` (TSC; `DS_FE2_CALLERS`).
pub fn note_place(cycles: u64) {
    caller_count(C_PLACES, 1);
    caller_count(C_PLACE_CYCLES, cycles);
}
const C_CHECKS: usize = 0;
const C_HITS: usize = 1;
const C_NO_RANGE: usize = 2;
const C_EPOCH: usize = 3;
const C_GENS: usize = 4;
const C_HOST: usize = 5;
const C_WORDS: usize = 6;
const C_TAKES: usize = 7;
const C_CYCLES: usize = 8;

thread_local! {
    static CALLER: std::cell::Cell<usize> = const { std::cell::Cell::new(3) };
}

/// The caller class of an executor named `name` (`DS_FE2_CALLERS`).
pub fn set_caller(name: &str) {
    let k = match name {
        "gpu-gfx" => 0,
        n if n.starts_with("gpu-gfx-par") => 1,
        "gpu-acb0x50" => 2,
        _ => 3,
    };
    CALLER.with(|c| c.set(k));
}

fn caller_count(what: usize, n: u64) {
    BY_CALLER[CALLER.with(|c| c.get())][what].fetch_add(n, Ordering::Relaxed);
}

fn note_range_fine(lo: u64, hi: u64, gens: [u32; FINE_CHUNKS], host: [u64; FINE_LEAVES], ep: u32, marked: bool) {
    RANGES.with(|r| {
        let mut r = r.borrow_mut();
        r.retain(|f| !(lo <= f.lo && f.hi <= hi && (marked || !f.marked)));
        r.insert(0, Fine { lo, hi, gens, host, ep, marked });
        r.truncate(RANGE_CACHE);
    })
}

static TABLE: [AtomicPtr<Chunk>; CHUNKS] = [const { AtomicPtr::new(std::ptr::null_mut()) }; CHUNKS];
static ON: AtomicBool = AtomicBool::new(false);

/// `DS_FE2=1` (read once: shader specs follow it, so the pipelines
/// precompiled at a session's start match).
pub fn wanted() -> bool {
    static W: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *W.get_or_init(|| std::env::var("DS_FE2").ok().as_deref() == Some("1"))
}

/// `DS_FE2_DEFER=1`: checks wait for the batch's close (`resolve`); off by
/// default: at the gate it made queue 0x50's many small batches late with
/// their label writes (FINDINGS.md 2026-10-27 night).
pub fn defer_checks() -> bool {
    static W: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *W.get_or_init(|| std::env::var("DS_FE2_DEFER").ok().as_deref() == Some("1"))
}

/// `DS_FE2_ARM_GPU=1`: pages GPU work writes are armed like others (GPU
/// writes reach guest memory by the write-back, which does not fault, and M
/// holds them already). Off by default: they are the pages the GPU writes
/// through their imports (write-backs, labels, timestamps), and the two GPU
/// timeouts of 2026-10-27 late night (driver resets, nvlddmkm 153) came in
/// the builds that armed them; protecting an import's pages before the
/// driver pinned them lost the device before (FINDINGS.md 2026-10-26).
fn arm_gpu() -> bool {
    static W: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *W.get_or_init(|| std::env::var("DS_FE2_ARM_GPU").ok().as_deref() == Some("1"))
}

/// Where pages may be armed (the executor's import list): calls its third
/// argument with the parts of [a, b) inside settled import windows, the
/// list held meanwhile; false: the list was busy.
pub type Armable = dyn Fn(u64, u64, &mut dyn FnMut(u64, u64)) -> bool;

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

#[inline]
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
    Some(make_chunk(slot))
}

/// A new chunk in `slot` (or the one another thread put there first); out
/// of line: its initializer's frame would make every `chunk` call probe the
/// stack.
#[cold]
#[inline(never)]
fn make_chunk(slot: &'static AtomicPtr<Chunk>) -> &'static Chunk {
    let new = Box::into_raw(Box::new(Chunk {
        valid: std::array::from_fn(|_| AtomicU64::new(0)),
        armed: std::array::from_fn(|_| AtomicU64::new(0)),
        gpu: std::array::from_fn(|_| AtomicU64::new(0)),
        wclean: std::array::from_fn(|_| AtomicU32::new(0)),
        wgen: std::array::from_fn(|_| AtomicU32::new(0)),
        clean: (0..CHUNK_PAGES).map(|_| AtomicU32::new(0)).collect(),
        wrote: (0..CHUNK_PAGES).map(|_| AtomicU32::new(0)).collect(),
        level: (0..CHUNK_PAGES).map(|_| AtomicU8::new(0)).collect(),
        uses: std::array::from_fn(|_| AtomicU8::new(0)),
        use_ep: std::array::from_fn(|_| AtomicU32::new(0)),
        seen: std::array::from_fn(|_| AtomicU64::new(0)),
        cgen: AtomicU32::new(0),
        sum: AtomicU64::new(0),
        sum_host: AtomicU64::new(0),
        hold_until: AtomicU64::new(0),
        hold_level: AtomicU8::new(0),
    }));
    match slot.compare_exchange(std::ptr::null_mut(), new, Ordering::AcqRel, Ordering::Acquire) {
        Ok(_) => unsafe { &*new },
        Err(cur) => {
            drop(unsafe { Box::from_raw(new) });
            unsafe { &*cur }
        }
    }
}

const CHUNK_BYTES: u64 = 1 << CHUNK_SHIFT;

/// Whether chunk `c` (at `a`) has a summary current in epoch `ep`.
#[inline]
fn summary_ok(c: &Chunk, a: u64, ep: u32) -> bool {
    let s = c.sum.load(Ordering::Acquire);
    if s == 0 || (s >> 32) as u32 != c.cgen.load(Ordering::Acquire) {
        return false;
    }
    let e = s as u32;
    (e == SUM_ARMED || e == ep) && c.sum_host.load(Ordering::Acquire) == crate::hosted::leaf_events(a)
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
static STATS: [AtomicU64; 31] = [const { AtomicU64::new(0) }; 31];
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
/// Chunks a placement passed by their summary, placements its thread's
/// range cache passed.
const S_SUMMARY: usize = 24;
const S_RANGE_HITS: usize = 25;
/// Runs uploaded without a take (written lately) and their pages.
const S_HOT_RUNS: usize = 26;
const S_HOT_PAGES: usize = 27;
/// Runs deferred to their batch's close, closes that resolved some, and the
/// merged runs they took.
const S_DEFERRED: usize = 28;
const S_RESOLVES: usize = 29;
const S_RESOLVED_RUNS: usize = 30;

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
/// `gpu_writes`: GPU work will write the range (its pages are marked so,
/// and never armed; with `DS_FE2_ARM_GPU=1` it is checked like any).
/// Err: an upload failed (M lacks some of the range).
/// With `defer`, runs that need a take are appended there instead (the
/// caller counts their pages as its batch's own until `resolve` at its
/// close).
pub fn ensure(lo: u64, hi: u64, ep: u32, gpu_writes: bool, mine: &dyn Fn(u64) -> bool, upload: &mut dyn FnMut(u64, u64) -> bool, armable: &Armable, defer: Option<&mut Vec<(u64, u64)>>) -> Result<(), &'static str> {
    if !callers_on() {
        return ensure_inner(lo, hi, ep, gpu_writes, mine, upload, armable, defer);
    }
    let t0 = unsafe { core::arch::x86_64::_rdtsc() };
    caller_count(C_CHECKS, 1);
    let r = ensure_inner(lo, hi, ep, gpu_writes, mine, upload, armable, defer);
    caller_count(C_CYCLES, unsafe { core::arch::x86_64::_rdtsc() }.wrapping_sub(t0));
    r
}

#[allow(clippy::too_many_arguments)]
fn ensure_inner(lo: u64, hi: u64, ep: u32, gpu_writes: bool, mine: &dyn Fn(u64) -> bool, upload: &mut dyn FnMut(u64, u64) -> bool, armable: &Armable, mut defer: Option<&mut Vec<(u64, u64)>>) -> Result<(), &'static str> {
    STATS[S_ENSURES].fetch_add(1, Ordering::Relaxed);
    STATS[S_ENSURE_PAGES].fetch_add((hi - lo) / PAGE, Ordering::Relaxed);
    let class = &BY_SIZE[size_class((hi - lo) / PAGE)];
    class[0].fetch_add(1, Ordering::Relaxed);
    class[1].fetch_add((hi - lo) / PAGE, Ordering::Relaxed);
    // Written ranges mark their pages GPU-written (kept unarmed) unless
    // `DS_FE2_ARM_GPU=1`; a range found fine with its marks set passes the
    // cache like a read one (marking every word on every use, ~430 words
    // for a pool of ~113 MB, made queue 0x50's dispatches slow).
    let gpu_writes = gpu_writes && !arm_gpu();
    if range_fine(lo, hi, ep, gpu_writes) {
        STATS[S_RANGE_HITS].fetch_add(1, Ordering::Relaxed);
        STATS[S_FAST].fetch_add(1, Ordering::Relaxed);
        if callers_on() {
            caller_count(C_HITS, 1);
        }
        return Ok(());
    }
    // (Before the scan: a change during it shows as a new value.)
    let snap = snapshot(lo, hi);
    SCRATCH.with(|s| {
        let mut s = s.borrow_mut();
        let (bad, written) = &mut *s;
        bad.clear();
        // The pages that need a check, as runs (a whole chunk by its
        // summary when it has a current one).
        let mut a = lo;
        // (The chunk being scanned whole: its generation and host events
        // before the scan, and whether every page so far was ok and armed.)
        let mut whole: Option<(u64, u32, u64, bool, bool)> = None;
        // Every page of the range armed so far (the host's bit set), and no
        // page passed for being the caller's upload.
        let mut all_armed = true;
        let mut mine_none = true;
        // (Words looked at: `DS_FE2_CALLERS`.)
        let mut words = 0u64;
        while a < hi {
            if a & (CHUNK_BYTES - 1) == 0 && a + CHUNK_BYTES <= hi {
                let c = chunk(a, true).expect("inside the span");
                if !gpu_writes && summary_ok(c, a, ep) {
                    STATS[S_SUMMARY].fetch_add(1, Ordering::Relaxed);
                    if c.sum.load(Ordering::Acquire) as u32 != SUM_ARMED {
                        all_armed = false;
                    }
                    a += CHUNK_BYTES;
                    continue;
                }
                whole = Some((a, c.cgen.load(Ordering::Acquire), crate::hosted::leaf_events(a), true, true));
            }
            let e = word_end(a, hi);
            let c = chunk(a, true).expect("inside the span");
            let w = idx(a) / 64;
            words += 1;
            if gpu_writes {
                c.gpu[w].fetch_or(word_mask(a, e), Ordering::AcqRel);
            }
            let need = word_mask(a, e) & !word_ok(c, a, e, ep);
            if all_armed && need == 0 {
                let armed = c.armed[w].load(Ordering::Acquire) & crate::hosted::armed_word(a & !(64 * PAGE - 1));
                all_armed = armed & word_mask(a, e) == word_mask(a, e);
            }
            if let Some(wh) = whole.as_mut() {
                if need != 0 {
                    wh.3 = false;
                } else if wh.4 {
                    // Every page of the word armed (the host's bit set)?
                    let armed = c.armed[w].load(Ordering::Acquire) & crate::hosted::armed_word(a & !(64 * PAGE - 1));
                    wh.4 = armed == u64::MAX;
                }
            }
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
                        mine_none = false;
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
            // A chunk scanned whole without a page to check: its summary.
            if let Some((start, g, h, ok, armed)) = whole {
                if a == start + CHUNK_BYTES {
                    if ok {
                        let c = chunk(start, false).expect("scanned");
                        c.sum_host.store(h, Ordering::Release);
                        c.sum.store((g as u64) << 32 | if armed { SUM_ARMED } else { ep } as u64, Ordering::Release);
                    }
                    whole = None;
                }
            }
        }
        if callers_on() {
            caller_count(C_WORDS, words);
        }
        if bad.is_empty() {
            STATS[S_FAST].fetch_add(1, Ordering::Relaxed);
            // (Not for ranges with pages the caller's batch uploaded: they are
            // fine for its batch only, until it is queued.)
            if let (true, Some((gens, host))) = (mine_none, snap) {
                note_range_fine(lo, hi, gens, host, if all_armed { SUM_ARMED } else { ep }, gpu_writes);
            }
            return Ok(());
        }
        let t0 = std::time::Instant::now();
        let frame = super::FRAMES.load(Ordering::Relaxed);
        for &(a, b) in bad.iter() {
            note_arming_ended(a, b);
            // Every page written lately: uploaded as it is, without a take.
            if hot_run(a, b, frame) {
                STATS[S_HOT_RUNS].fetch_add(1, Ordering::Relaxed);
                let mut x = a;
                while x < b {
                    if mine(x) {
                        x += PAGE;
                        continue;
                    }
                    let mut y = x + PAGE;
                    while y < b && !mine(y) {
                        y += PAGE;
                    }
                    if !upload(x, y) {
                        STATS[S_UPLOAD_FAILED].fetch_add(1, Ordering::Relaxed);
                        STATS[S_SLOW_NS].fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                        return Err("an upload into the mirror failed");
                    }
                    STATS[S_UPLOADS].fetch_add(1, Ordering::Relaxed);
                    STATS[S_UPLOAD_PAGES].fetch_add((y - x) / PAGE, Ordering::Relaxed);
                    STATS[S_HOT_PAGES].fetch_add((y - x) / PAGE, Ordering::Relaxed);
                    x = y;
                }
                continue;
            }
            if let Some(d) = defer.as_deref_mut() {
                d.push((a, b));
                STATS[S_DEFERRED].fetch_add(1, Ordering::Relaxed);
                continue;
            }
            if let Err(e) = check_run(a, b, ep, frame, !gpu_writes, mine, upload, armable, written, class) {
                STATS[S_SLOW_NS].fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                return Err(e);
            }
        }
        STATS[S_SLOW_NS].fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        Ok(())
    })
}

/// Arms (`arm`: hosted, where allowed) and takes [a, b), uploads its pages
/// M lacks (not `mine`), and marks its valid pages clean in epoch `ep`.
#[allow(clippy::too_many_arguments)]
fn check_run(a: u64, b: u64, ep: u32, frame: u64, arm: bool, mine: &dyn Fn(u64) -> bool, upload: &mut dyn FnMut(u64, u64) -> bool, armable: &Armable, written: &mut Vec<u64>, class: &[AtomicU64; 4]) -> Result<(), &'static str> {
    take_run(a, b, frame, arm, armable, written, class);
    fix_run(a, b, ep, mine, upload)
}

/// Arms (`arm`: hosted, where allowed) and takes [a, b): pages written since
/// their last take lose their valid bit.
fn take_run(a: u64, b: u64, frame: u64, arm: bool, armable: &Armable, written: &mut Vec<u64>, class: &[AtomicU64; 4]) {
    if arm && crate::hosted::on() {
        try_arm(a, b, frame, armable);
    }
    // The take: pages written since their last take lose their valid bit
    // (`marks_taken`, for any take's pages).
    written.clear();
    let tt = std::time::Instant::now();
    let r = super::precache::take_written_for(a, b - a, written, None, crate::mem::WW_WHO_FE2);
    let tns = tt.elapsed().as_nanos() as u64;
    STATS[S_TAKE_NS].fetch_add(tns, Ordering::Relaxed);
    STATS[S_TAKES].fetch_add(1, Ordering::Relaxed);
    if callers_on() {
        caller_count(C_TAKES, 1);
    }
    class[2].fetch_add(1, Ordering::Relaxed);
    class[3].fetch_add(tns, Ordering::Relaxed);
    STATS[S_TAKE_WRITTEN].fetch_add(written.len() as u64, Ordering::Relaxed);
    if r.is_none() {
        STATS[S_TAKE_FAILED].fetch_add(1, Ordering::Relaxed);
        invalidate(a, b - a);
    }
}

/// Uploads the pages of [a, b) M lacks (not `mine`) and marks its valid
/// pages clean in epoch `ep` (after its take).
fn fix_run(a: u64, b: u64, ep: u32, mine: &dyn Fn(u64) -> bool, upload: &mut dyn FnMut(u64, u64) -> bool) -> Result<(), &'static str> {
    // Uploads of the pages M lacks, as runs of consecutive ones (word by
    // word: only the invalid pages are looked at one by one).
    let mut failed = false;
    let mut run: Option<(u64, u64)> = None;
    let mut flush = |run: &mut Option<(u64, u64)>, failed: &mut bool| {
        if let Some((x, y)) = run.take() {
            if !upload(x, y) {
                // (The rest of the range is still made current.)
                STATS[S_UPLOAD_FAILED].fetch_add(1, Ordering::Relaxed);
                *failed = true;
            } else {
                STATS[S_UPLOADS].fetch_add(1, Ordering::Relaxed);
                STATS[S_UPLOAD_PAGES].fetch_add((y - x) / PAGE, Ordering::Relaxed);
            }
        }
    };
    let mut x = a;
    while x < b {
        let e = word_end(x, b);
        let c = chunk(x, true).expect("inside the span");
        let lacks = word_mask(x, e) & !c.valid[idx(x) / 64].load(Ordering::Acquire);
        let wa = x & !(64 * PAGE - 1);
        let mut r = lacks;
        while r != 0 {
            let p = wa + r.trailing_zeros() as u64 * PAGE;
            r &= r - 1;
            if mine(p) {
                flush(&mut run, &mut failed);
                continue;
            }
            match run.as_mut() {
                Some(l) if l.1 == p => l.1 = p + PAGE,
                _ => {
                    flush(&mut run, &mut failed);
                    run = Some((p, p + PAGE));
                }
            }
        }
        x = e;
    }
    flush(&mut run, &mut failed);
    // Valid pages are clean in this epoch (uploaded ones become valid when
    // the caller's batch is queued: `uploaded`).
    set_clean(a, b, ep, false);
    if failed {
        return Err("an upload into the mirror failed");
    }
    Ok(())
}

/// The checks a batch's bindings deferred (`ensure` with a list), at its
/// close: `runs` merged where less than TAKE_GAP pages apart, each armed
/// where allowed, taken, its pages M lacks and not `mine` (uploaded by the
/// batch already) uploaded into its prologue (`upload`), its valid pages
/// marked clean in the epoch now. `runs` is left empty.
pub fn resolve(runs: &mut Vec<(u64, u64)>, mine: &dyn Fn(u64) -> bool, upload: &mut dyn FnMut(u64, u64) -> bool, armable: &Armable) -> Result<(), &'static str> {
    if runs.is_empty() {
        return Ok(());
    }
    let t0 = std::time::Instant::now();
    runs.sort_unstable();
    // Envelopes for the takes: runs less than TAKE_GAP pages apart with
    // mapped memory between them (a take over a hole fails); each
    // envelope's index range into `runs`.
    let mut merged: Vec<(u64, u64, usize, usize)> = Vec::with_capacity(runs.len());
    for (k, &(a, b)) in runs.iter().enumerate() {
        match merged.last_mut() {
            Some(l) if a <= l.1 || (l.1 + TAKE_GAP * PAGE >= a && crate::mem::readable_span(l.1).is_some_and(|(_, e)| e >= a)) => {
                l.1 = l.1.max(b);
                l.3 = k + 1;
            }
            _ => merged.push((a, b, k, k + 1)),
        }
    }
    STATS[S_RESOLVES].fetch_add(1, Ordering::Relaxed);
    STATS[S_RESOLVED_RUNS].fetch_add(merged.len() as u64, Ordering::Relaxed);
    let ep = super::precache::epoch() as u32;
    let frame = super::FRAMES.load(Ordering::Relaxed);
    let r = SCRATCH.with(|s| {
        let mut s = s.borrow_mut();
        let written = &mut s.1;
        // (A run that fails leaves the others to be made current still.)
        let mut r = Ok(());
        for &(a, b, i, j) in &merged {
            note_arming_ended(a, b);
            take_run(a, b, frame, true, armable, written, &BY_SIZE[size_class((b - a) / PAGE)]);
            // Only the pages the batch bound need M current.
            for &(x, y) in &runs[i..j] {
                if let Err(e) = fix_run(x, y, ep, mine, upload) {
                    r = Err(e);
                }
            }
        }
        r
    });
    runs.clear();
    STATS[S_SLOW_NS].fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
    r
}

/// Whether [a, b) is at most HOT_PAGES pages, every one found written
/// within HOT_FLIPS flips.
fn hot_run(a: u64, b: u64, frame: u64) -> bool {
    if (b - a) / PAGE > HOT_PAGES {
        return false;
    }
    let mut x = a;
    while x < b {
        let Some(c) = chunk(x, false) else { return false };
        let wrote = c.wrote[idx(x)].load(Ordering::Relaxed) as u64;
        if wrote == 0 || frame.saturating_sub(wrote - 1) >= HOT_FLIPS {
            return false;
        }
        x += PAGE;
    }
    true
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
        // A whole word goes by its stamp (`word_ok` reads the pages' own only
        // when the word's is older).
        let whole = mask == u64::MAX;
        if !whole {
            for k in i..i + ((e - x) / PAGE) as usize {
                c.clean[k].store(ep, Ordering::Release);
            }
        }
        // A take's first look at the word in this epoch counts as a use.
        if !validate && c.use_ep[w].swap(ep, Ordering::AcqRel) != ep {
            let u = &c.uses[w];
            if u.load(Ordering::Relaxed) < ARM_USES {
                u.fetch_add(1, Ordering::Relaxed);
            }
        }
        let unseen = mask & !c.seen[w].load(Ordering::Acquire);
        if unseen != 0 {
            let mut r = unseen;
            while r != 0 {
                let k = w * 64 + r.trailing_zeros() as usize;
                r &= r - 1;
                let _ = c.wrote[k].compare_exchange(0, f, Ordering::AcqRel, Ordering::Relaxed);
            }
            c.seen[w].fetch_or(unseen, Ordering::AcqRel);
        }
        if validate {
            c.valid[w].fetch_or(mask, Ordering::AcqRel);
        }
        if whole {
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
                c.cgen.fetch_add(1, Ordering::AcqRel);
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
    {
        // (An arming that checked before this is done by now, and the
        // caller's disarm after this undoes it: `arm_part`.)
        let _lock = crate::hosted::arm_lock();
        HOST_WRITES.lock().push((base, base.saturating_add(len)));
        HOST_WRITES_N.fetch_add(1, Ordering::AcqRel);
    }
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

/// Arms the quiet pages of [a, b) not armed now (nor written by GPU work,
/// unless `DS_FE2_ARM_GPU=1`) outside held chunks, by runs inside settled
/// import windows (`armable`), plain read-write memory only.
fn try_arm(a: u64, b: u64, frame: u64, armable: &Armable) {
    let arm_gpu = arm_gpu();
    // Why a page is not a candidate (None: it is one).
    let why_not = |p: u64| -> Option<usize> {
        let Some(c) = chunk(p, false) else { return Some(R_QUIET) };
        let (i, w) = (idx(p), idx(p) / 64);
        let bit = 1u64 << (i % 64);
        if c.armed[w].load(Ordering::Acquire) & bit != 0 {
            return Some(R_ARMED);
        }
        if frame < c.hold_until.load(Ordering::Acquire) {
            return Some(R_HELD);
        }
        if !arm_gpu && c.gpu[w].load(Ordering::Acquire) & bit != 0 {
            return Some(R_GPU);
        }
        let wrote = c.wrote[i].load(Ordering::Acquire) as u64;
        let need = ARM_QUIET << c.level[i].load(Ordering::Acquire);
        if wrote == 0 || frame.saturating_sub(wrote - 1) < need {
            return Some(R_QUIET);
        }
        if c.uses[w].load(Ordering::Relaxed) < ARM_USES {
            return Some(R_USES);
        }
        None
    };
    let mut x = a;
    while x < b {
        // The next run of candidates from x.
        if let Some(r) = why_not(x) {
            NOT_ARMED[r].fetch_add(1, Ordering::Relaxed);
            x += PAGE;
            continue;
        }
        let cand = |p: u64| why_not(p).is_none();
        let mut y = x + PAGE;
        while y < b && cand(y) {
            y += PAGE;
        }
        arm_run(x, y, frame, armable);
        x = y;
    }
}

/// Pages checked by takes that were not armed, by reason: armed already
/// (by us, its host bit set), written by GPU work (`DS_FE2_ARM_GPU`),
/// written within their quiet time, found clean in fewer than ARM_USES
/// epochs, outside settled import windows, not plain read-write (or a file
/// read in progress), in a chunk held after a burst of faults, the import
/// list busy.
static NOT_ARMED: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];
const R_ARMED: usize = 0;
const R_GPU: usize = 1;
const R_QUIET: usize = 2;
const R_USES: usize = 3;
const R_WINDOW: usize = 4;
const R_RW: usize = 5;
const R_HELD: usize = 6;
const R_BUSY: usize = 7;

/// The faults `watch` read: [on pages of the span, those that ended an
/// arming of ours (its level raised there), bursts, pages they disarmed,
/// faults outside the span, ring overflows seen].
static FAULTS: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];
const F_SPAN: usize = 0;
const F_ENDED: usize = 1;
const F_BURSTS: usize = 2;
const F_DISARMED: usize = 3;
const F_OUTSIDE: usize = 4;
const F_LOST: usize = 5;

/// Reads the faults the host answered, every millisecond (a thread of its
/// own, hosted runs only): each ends its page's arming (the next waits
/// longer) and restarts its quiet time; a chunk with CHUNK_BURST of them
/// within BURST_MS, or with some while all chunks together had ALL_BURST,
/// is disarmed and held (`burst`).
fn watch() {
    let mut got: Vec<(u64, bool, u64)> = Vec::new();
    let mut seen = crate::hosted::faults();
    // This window's faults per chunk, whether the chunk burst in it, the
    // chunks with any, and all of them.
    let mut counts = vec![0u32; CHUNKS];
    let mut fired = vec![false; CHUNKS];
    let mut touched: Vec<usize> = Vec::new();
    let mut total = 0u32;
    let mut window = std::time::Instant::now();
    let mut logged = 0u32;
    loop {
        std::thread::sleep(std::time::Duration::from_millis(1));
        if window.elapsed().as_millis() >= BURST_MS {
            for &k in &touched {
                counts[k] = 0;
                fired[k] = false;
            }
            touched.clear();
            total = 0;
            window = std::time::Instant::now();
        }
        let n = crate::hosted::faults();
        if n == seen {
            continue;
        }
        seen = n;
        got.clear();
        if !crate::hosted::drain_faults(&mut got) {
            FAULTS[F_LOST].fetch_add(1, Ordering::Relaxed);
        }
        let frame = super::FRAMES.load(Ordering::Relaxed);
        for &(page, _, rip) in &got {
            let Some(c) = chunk(page, false) else {
                FAULTS[F_OUTSIDE].fetch_add(1, Ordering::Relaxed);
                continue;
            };
            FAULTS[F_SPAN].fetch_add(1, Ordering::Relaxed);
            let (i, w) = (idx(page), idx(page) / 64);
            let bit = 1u64 << (i % 64);
            // (A take's `note_arming_ended` may have seen it first.)
            if c.armed[w].fetch_and(!bit, Ordering::AcqRel) & bit != 0 {
                let _ = c.level[i].fetch_update(Ordering::AcqRel, Ordering::Acquire, |v| Some((v + 1).min(ARM_MAX_LEVEL)));
                c.cgen.fetch_add(1, Ordering::AcqRel);
                FAULTS[F_ENDED].fetch_add(1, Ordering::Relaxed);
                STATS[S_ARM_ENDED].fetch_add(1, Ordering::Relaxed);
            }
            c.wrote[i].store(frame as u32 + 1, Ordering::Release);
            c.uses[w].store(0, Ordering::Relaxed);
            let k = ((page - MIRROR_BASE) >> CHUNK_SHIFT) as usize;
            if counts[k] == 0 {
                touched.push(k);
            }
            counts[k] += 1;
            total += 1;
            if logged < 40 {
                logged += 1;
                crate::info!("fe2: write fault on armed page {page:#x} at {}", describe_code(rip));
            }
        }
        for &k in &touched {
            if !fired[k] && (counts[k] >= CHUNK_BURST || total >= ALL_BURST) {
                fired[k] = true;
                burst(k, counts[k], total, frame);
            }
        }
    }
}

/// Where code at `rip` is (the title's or a host module's).
fn describe_code(rip: u64) -> String {
    if (crate::mem::GUEST_BASE..crate::mem::GUEST_END).contains(&rip) {
        crate::exception::describe_guest(rip)
    } else {
        crate::exception::host_symbol(rip)
    }
}

/// Chunk `k` had `n` faults within BURST_MS (`total` in all chunks): its
/// armed pages are disarmed and arming there waits CHUNK_HOLD flips (twice
/// as long for each burst within HOLD_MEMORY flips of the last hold's end).
fn burst(k: usize, n: u32, total: u32, frame: u64) {
    let base = MIRROR_BASE + ((k as u64) << CHUNK_SHIFT);
    let Some(c) = chunk(base, false) else { return };
    let until = c.hold_until.load(Ordering::Acquire);
    let level = if until != 0 && frame < until + HOLD_MEMORY { (c.hold_level.load(Ordering::Acquire) + 1).min(CHUNK_HOLD_MAX_LEVEL) } else { 0 };
    c.hold_level.store(level, Ordering::Release);
    let hold = CHUNK_HOLD << level;
    c.hold_until.store(frame + hold, Ordering::Release);
    let mut pages = 0u64;
    {
        // No mapping change meanwhile, no arming between our bits and the
        // host's (ours go first: a take seeing the host's gone and ours set
        // would count the arming as ended by a write).
        let _readers = crate::mem::GUEST_READERS.read_recursive();
        let _arming = super::resident::arm_guard();
        let _lock = crate::hosted::arm_lock();
        let mut pend: Option<(u64, u64)> = None;
        for w in 0..WORDS {
            let ours = c.armed[w].load(Ordering::Acquire);
            if ours == 0 {
                continue;
            }
            let wa = base + (w as u64) * 64 * PAGE;
            let live = ours & crate::hosted::armed_word(wa);
            if live == 0 {
                continue;
            }
            c.armed[w].fetch_and(!live, Ordering::AcqRel);
            let mut r = live;
            while r != 0 {
                let b = r.trailing_zeros() as u64;
                let run = (!(r >> b)).trailing_zeros().min(64 - b as u32) as u64;
                r &= !(if run >= 64 { u64::MAX } else { ((1u64 << run) - 1) << b });
                let (x, y) = (wa + b * PAGE, wa + (b + run) * PAGE);
                pages += run;
                pend = match pend {
                    Some((p, q)) if q == x => Some((p, y)),
                    Some((p, q)) => {
                        crate::hosted::disarm(p, q - p);
                        Some((x, y))
                    }
                    None => Some((x, y)),
                };
            }
        }
        if let Some((p, q)) = pend {
            crate::hosted::disarm(p, q - p);
        }
        c.cgen.fetch_add(1, Ordering::AcqRel);
    }
    let b = FAULTS[F_BURSTS].fetch_add(1, Ordering::Relaxed);
    FAULTS[F_DISARMED].fetch_add(pages, Ordering::Relaxed);
    if b < 40 {
        crate::info!("fe2: {n} faults within {BURST_MS} ms in {base:#x}+64M ({total} in all): {pages} armed pages there disarmed, arming there held {hold} flips");
    }
}

/// Arms the parts of the candidate run [x, y) inside settled import windows.
fn arm_run(x: u64, y: u64, frame: u64, armable: &Armable) {
    let mut inside = 0u64;
    let listed = armable(x, y, &mut |p, q| {
        inside += q - p;
        arm_part(p, q, frame);
    });
    if !listed {
        NOT_ARMED[R_BUSY].fetch_add((y - x) / PAGE, Ordering::Relaxed);
    } else if inside < y - x {
        NOT_ARMED[R_WINDOW].fetch_add((y - x - inside) / PAGE, Ordering::Relaxed);
    }
}

/// Arms [x, y) (inside settled import windows, the import list held: an
/// import over it waits, then disarms it).
fn arm_part(x: u64, y: u64, frame: u64) {
    // No mapping change meanwhile (`resident::remap_guard`), and no other
    // arm or disarm between the checks and the arming (`hosted::arm_lock`:
    // a file read beginning now waits, then disarms what is armed).
    let _readers = crate::mem::GUEST_READERS.read_recursive();
    let _arming = super::resident::arm_guard();
    let _lock = crate::hosted::arm_lock();
    if host_writing(x, y) || !plain_rw(x, y - x) {
        NOT_ARMED[R_RW].fetch_add((y - x) / PAGE, Ordering::Relaxed);
        return;
    }
    if held(x, y, frame) {
        NOT_ARMED[R_HELD].fetch_add((y - x) / PAGE, Ordering::Relaxed);
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

/// Whether a chunk of [x, y) holds its arming at `frame` (`burst`).
fn held(x: u64, y: u64, frame: u64) -> bool {
    let mut p = x & !(CHUNK_BYTES - 1);
    while p < y {
        if chunk(p, false).is_some_and(|c| frame < c.hold_until.load(Ordering::Acquire)) {
            return true;
        }
        p += CHUNK_BYTES;
    }
    false
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
        c.cgen.fetch_add(1, Ordering::AcqRel);
        c.wrote[i].store(f, Ordering::Release);
        c.uses[w].store(0, Ordering::Relaxed);
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
            c.cgen.fetch_add(1, Ordering::AcqRel);
            for k in idx(x)..idx(x) + ((e - x) / PAGE) as usize {
                c.wrote[k].store(f, Ordering::Release);
            }
            c.uses[w].store(0, Ordering::Relaxed);
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
    // (Parallel runs stay: their chunks run in program order, each with
    // its own uploads and write-backs; so does recording past label waits
    // and runs: held batches are deferred ones, their uploads valid once
    // they are queued, and a dropped stretch's never happen. So do the
    // per-view copies as buffer updates, forwarded to the CPU-side reads of
    // their blocks: the updates write the mirror too.)
    // (The host census and mirror mode would read the host's faults too.)
    const OFF: [&str; 11] = ["long_runs", "frame_ahead", "homes", "keep_small", "draw_args_gpu", "gpu_loads", "gpu_chains", "shadow_args", "pre_share", "host_census", "gpu_mirror"];
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
            if crate::hosted::on() {
                let _ = std::thread::Builder::new().name("fe2-faults".into()).spawn(watch);
            }
            crate::info!(
                "fe2: front end 2 on ({made}; {s}; hosted arming {}{}; switched off: {})",
                if crate::hosted::on() { "on" } else { "off (not under the host)" },
                if arm_gpu() { "" } else { ", GPU-written pages never armed" },
                OFF.join(", ")
            );
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
        "    front end 2: {} placement checks ({} pages; {} needed nothing; {} chunks passed by their summary, {} placements by the range cache; {} runs deferred to {} batch closes, taken as {} runs), {} takes ({} pages written, {} failed; {:.1} ms), checking {:.1} ms; uploads {} ({} pages, {} MiB; runs written lately, without a take: {}, {} pages), {} failed, {} pages valid once queued ({} raced a write); hosted arming: {} pages in {} runs ({} failed), {} armings ended by writes; faults read {} ({} ending an arming of ours, {} outside the span, {} ring overflows), {} chunk bursts ({} pages disarmed); invalidated: {} ranges, {} pages by takes; write-backs recorded {}: {} pages, {} dwords, {} pages unmapped (lost), {} lists past their capacity; CP writes also into the mirror {}, pages copied by the CPU {}, written bindings without an import {}\n    front end 2's placement checks by size: {}\n    front end 2's pages taken but not armed: {}\n",
        s[S_ENSURES],
        s[S_ENSURE_PAGES],
        s[S_FAST],
        s[S_SUMMARY],
        s[S_RANGE_HITS],
        s[S_DEFERRED],
        s[S_RESOLVES],
        s[S_RESOLVED_RUNS],
        s[S_TAKES],
        s[S_TAKE_WRITTEN],
        s[S_TAKE_FAILED],
        ms(s[S_TAKE_NS]),
        ms(s[S_SLOW_NS]),
        s[S_UPLOADS],
        s[S_UPLOAD_PAGES],
        (s[S_UPLOAD_PAGES] * PAGE) >> 20,
        s[S_HOT_RUNS],
        s[S_HOT_PAGES],
        s[S_UPLOAD_FAILED],
        s[S_VALIDATED],
        s[S_RACED],
        s[S_ARMED],
        s[S_ARM_RUNS],
        s[S_ARM_FAILED],
        s[S_ARM_ENDED],
        FAULTS[F_SPAN].load(Ordering::Relaxed),
        FAULTS[F_ENDED].load(Ordering::Relaxed),
        FAULTS[F_OUTSIDE].load(Ordering::Relaxed),
        FAULTS[F_LOST].load(Ordering::Relaxed),
        FAULTS[F_BURSTS].load(Ordering::Relaxed),
        FAULTS[F_DISARMED].load(Ordering::Relaxed),
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
        ["armed already", "GPU-written", "written lately", "too few clean epochs", "no settled window", "not plain read-write", "chunk held", "import list busy"]
            .iter()
            .zip(NOT_ARMED.iter())
            .map(|(n, c)| format!("{n} {}", c.load(Ordering::Relaxed)))
            .collect::<Vec<_>>()
            .join(", "),
    ) + &callers_line()
}

/// The placement checks by caller (`DS_FE2_CALLERS=1`; "" otherwise).
pub fn callers_line() -> String {
    if !callers_on() {
        return String::new();
    }
    let hz = super::threadcpu::tsc_hz();
    let parts: Vec<String> = CALLER_NAMES
        .iter()
        .zip(BY_CALLER.iter())
        .map(|(n, c)| {
            let v: Vec<u64> = c.iter().map(|x| x.load(Ordering::Relaxed)).collect();
            format!(
                "{n}: {} placements ({:.1} ms), {} checks ({} by the range cache; missed: no range {}, another epoch {}, chunk generations {}, host events {}), {} words scanned, {} takes, {:.1} ms",
                v[C_PLACES],
                v[C_PLACE_CYCLES] as f64 * 1e3 / hz,
                v[C_CHECKS],
                v[C_HITS],
                v[C_NO_RANGE],
                v[C_EPOCH],
                v[C_GENS],
                v[C_HOST],
                v[C_WORDS],
                v[C_TAKES],
                v[C_CYCLES] as f64 * 1e3 / hz
            )
        })
        .collect();
    format!("    front end 2's checks by caller: {}\n", parts.join("; "))
}
