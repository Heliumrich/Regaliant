//! Front end 2 (docs/FRONTEND2.md; the default since 2026-10-30, `DS_FE2=0`
//! the old front end as a fallback): the GPU front end on one GPU address
//! space. Every buffer GPU work uses inside the mirror's span
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
//! - CPU writes are found by write watching. A take stamps the pages it
//!   found unwritten with the synchronization epoch (`precache::epoch`) read
//!   before it; a valid page that is not armed is trusted by a check whose
//!   queue's view (`views_on`: a compute queue's last synchronization point;
//!   the epoch now for the graphics queue and its workers) is at most its
//!   stamp. Every take, ours or another's (`precache` applies them all under
//!   one protocol), clears the valid bit of the pages it reports
//!   (`marks_taken`), as do file reads, CP writes the CPU performs and
//!   mapping changes (`invalidate`).
//! - Hosted (the default, `hosted::wants_host`): pages quiet for ARM_QUIET flips (doubled each time an
//!   arming ended in a write, up to ARM_MAX_LEVEL) and found clean by takes
//!   in ARM_USES epochs since, inside import windows at least SETTLE flips
//!   old (protecting a new import's pages lost the device: FINDINGS.md
//!   2026-10-26) are armed read-only through the host; while the host's bit
//!   stays set (no write fault, disarm or mapping change since) such a page
//!   needs no take. Pages GPU work writes are not armed (they are the
//!   pages the GPU writes through their imports; `DS_FE2_ARM_GPU=1`: armed
//!   too, as GPU writes do not fault and M holds them). Each fault
//!   stops the whole process for 15-50 us, and the host answers it for its
//!   page's whole 64-page word (`hosted::set_word_mode`): a thread reads the
//!   faults the host answered (`watch`; the word's pages take the faulted
//!   page's backoff level), and a 64 MiB chunk with CHUNK_BURST of them
//!   within BURST_MS (or with some while all chunks together had ALL_BURST:
//!   a load rewriting memory armed before it) has its armed pages disarmed
//!   and its arming held CHUNK_HOLD flips (doubled for each burst within
//!   HOLD_MEMORY flips of its last hold, up to CHUNK_HOLD_MAX_LEVEL); the
//!   other chunks keep theirs.
//! - A take only spares uploads of pages that did not change: small runs of
//!   pages found written within HOT_FLIPS flips (the title's rings of
//!   per-draw data, rewritten every frame) are uploaded without one.
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
/// doubles that (up to CHUNK_HOLD_MAX_LEVEL times). A burst is a load-like
/// storm: 64 faults in 100 ms cost ~1.7 ms of stopped process. (16 and 64
/// until 2026-10-28 small hours: at the gate, ~1.2 faults a flip in
/// clusters still held ~20 chunks a window, and their pages were taken every
/// epoch. 256 and 1024, tried the same night, let the page-by-page backoff
/// learn the gate's writes alone: 52 faults a flip against 6, 12 flips/s.)
const CHUNK_BURST: u32 = 64;
const ALL_BURST: u32 = 256;
const BURST_MS: u128 = 100;
/// Holds double only while bursts follow each other (a load going on): the
/// gate's storms, ~20 s apart, held their chunks 120 flips each with the
/// doubling counted over 600 flips, and a held chunk's pages are taken
/// every epoch; 8-flip holds made arming churn (~300 pages armed and
/// disarmed a flip, faults 6 -> 10 a flip; 2026-10-28).
const CHUNK_HOLD: u64 = 60;
const CHUNK_HOLD_MAX_LEVEL: u8 = 1;
const HOLD_MEMORY: u64 = 120;
/// Flips an import window exists before pages in it are armed.
pub const SETTLE: u64 = 60;
/// Runs of at most HOT_PAGES pages found written within HOT_FLIPS flips, or
/// within their quiet time before arming (`hot_quiet`), are uploaded without a
/// take (larger ones, as loads write them, would be uploaded again at every
/// epoch where a take finds them unchanged). A take of a page within its
/// quiet time found it unchanged 93% of the time over a 1-1 walk (~1,880
/// takes a flip, 3.6 us each, 87% of ranges under 64 KiB: FINDINGS.md
/// 2026-10-28 night), and each move of a view repeats it; an upload is a
/// copy region in the batch's prologue.
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
    /// Armed and taken since (by one of our takes that began with the page
    /// armed): from then on only a write fault may change it; a take that
    /// finds such a page written found a write the protection missed
    /// (`take_run`: a hole).
    varmed: [AtomicU64; WORDS],
    /// Written by GPU work (never armed, unless `DS_FE2_ARM_GPU=1`). Bits
    /// are only ever set.
    gpu: [AtomicU64; WORDS],
    /// Every `gpu` bit of the chunk set (a written range's scan covered it
    /// whole): its scans need set none, so its summary serves them too.
    gpu_all: AtomicBool,
    /// The epoch every valid page of the word was found clean in (a take
    /// over the whole word), else 0.
    wclean: [AtomicU32; WORDS],
    /// At least the newest epoch any page of the word has in `clean` (a
    /// take over part of the word): older than a check's requirement, no
    /// page of it meets it by its own stamp.
    wmax: [AtomicU32; WORDS],
    /// Bumped by every invalidation of a page of the word.
    wgen: [AtomicU32; WORDS],
    /// The epoch (low 32 bits) a take found the page clean in.
    clean: Box<[AtomicU32]>,
    /// The flip, plus one, a CPU write to the page was last found.
    wrote: Box<[AtomicU32]>,
    /// Bumped by every CPU write to the page found (and every invalidation
    /// of it): run-shared uploads hold while it stays (`run_shared`).
    pgen: Box<[AtomicU32]>,
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
    /// ended there (the summary below goes with it). On a cache line of its
    /// own: every range-cache hit over the chunk reads it, and it lay beside
    /// the summary that scans store.
    cgen: Line<AtomicU32>,
    /// The chunk's summary: every page of it needed no check as of (`cgen`
    /// << 32 | the requirement it met (`ensure`'s `want`: met for any lower
    /// one too), or SUM_ARMED for every page armed: any) and the host's
    /// events of its leaf (`hosted::leaf_events`); 0: none.
    sum: AtomicU64,
    sum_host: AtomicU64,
    /// No page of the chunk is armed before this flip (after a burst of
    /// faults: `burst`), and the doublings of its last hold.
    hold_until: AtomicU64,
    hold_level: AtomicU8,
}

/// A value on a cache line of its own (`Chunk::cgen`).
#[repr(align(64))]
struct Line<T>(T);

impl<T> std::ops::Deref for Line<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.0
    }
}

/// `Chunk::sum`'s epoch part when every page of the chunk is armed.
const SUM_ARMED: u32 = u32::MAX;

/// A queue without a view of its own (`view_cell`): its checks need pages
/// current as of the epoch now (the graphics queue and its workers).
pub const NO_VIEW: u32 = u32::MAX;

/// Per-queue views (`DS_FE2_VIEWS`, on by default; 0: every check needs the
/// epoch now): a compute queue's checks need pages current as of its own
/// last synchronization point (a title submission, a wait the CPU released:
/// the epoch then; a wait on a label another queue's CP wrote: that queue's
/// view, the epoch now for queues without one), not as of every queue's.
/// The epoch ends at any queue's (~260 times a flip at the gate), and queue
/// 0x50 checked its pools again after each. A take's pages are stamped with
/// the epoch before it, so a page stamped at or after a queue's view
/// reflects every CPU write before that queue's synchronization point.
static VIEWS: Mutex<Vec<(String, std::sync::Arc<AtomicU32>)>> = Mutex::new(Vec::new());

pub fn views_on() -> bool {
    static W: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *W.get_or_init(|| std::env::var("DS_FE2_VIEWS").ok().as_deref() != Some("0"))
}

/// `DS_FE2_GFX_VIEWS` (on by default; 0: the graphics queue and its
/// workers need the epoch now, as before 2026-10-28 night): the graphics
/// queue has a view of its own too (its submissions, waits the CPU
/// released, REWINDs, waits in parallel runs the CPU released; the waits on
/// other queues' labels adopt theirs), and each parallel run's workers
/// check as of the view its command processor had when it planned the run.
/// While the camera moved its checks re-took the pages of every epoch any
/// queue ended (~3,400 takes a flip in a 1-1 walk window).
pub fn gfx_views() -> bool {
    static W: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *W.get_or_init(|| std::env::var("DS_FE2_GFX_VIEWS").ok().as_deref() != Some("0"))
}

/// The view of the queue (or parallel worker) named `name` (made at
/// NO_VIEW).
pub fn view_cell(name: &str) -> std::sync::Arc<AtomicU32> {
    let mut g = VIEWS.lock();
    if let Some((_, c)) = g.iter().find(|(n, _)| n == name) {
        return c.clone();
    }
    let c = std::sync::Arc::new(AtomicU32::new(NO_VIEW));
    g.push((name.to_string(), c.clone()));
    c
}

/// The view of the queue named `name` (None: it has none).
pub fn view_of(name: &str) -> Option<u32> {
    let g = VIEWS.lock();
    g.iter().find(|(n, _)| n == name).map(|(_, c)| c.load(Ordering::Acquire)).filter(|&v| v != NO_VIEW)
}

/// A queue synchronized: its view becomes at least `v`; whether that
/// raised it.
pub fn sync_view(cell: &AtomicU32, v: u32) -> bool {
    let old = cell.fetch_update(Ordering::AcqRel, Ordering::Acquire, |c| Some(if c == NO_VIEW { v } else { c.max(v) })).unwrap_or(NO_VIEW);
    VIEW_SYNCS[0].fetch_add(1, Ordering::Relaxed);
    let raised = old == NO_VIEW || old < v;
    if raised {
        VIEW_SYNCS[1].fetch_add(1, Ordering::Relaxed);
    }
    raised
}

/// Views synchronized, and those it raised (`status`).
static VIEW_SYNCS: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
/// Views raised by the queues' own synchronization (`Cp::fe2_sync`): the
/// graphics command processor's, the compute queues' (`status`).
static VIEW_RAISES: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// Queue `name`'s own synchronization raised its view (`status`).
pub fn note_raise(compute: bool) {
    VIEW_RAISES[compute as usize].fetch_add(1, Ordering::Relaxed);
}

/// Ranges per thread whose last check found every page fine (`RANGES`).
const RANGE_CACHE: usize = 32;
/// Ranges over more chunks or host leaves than these are not kept there.
const FINE_CHUNKS: usize = 8;
const FINE_LEAVES: usize = 5;

/// A range a thread found needing no check: the generations of the chunks
/// it covers (`Chunk::cgen`: every invalidation of a valid page and every
/// ended arming there bumps them) and the host's events of its leaves
/// (`hosted::leaf_events`) as they were before that check, and the
/// requirement it met (`ensure`'s `want`; SUM_ARMED: every page was armed,
/// any). While they are unchanged it still needs none. (Until 2026-10-27
/// late night one generation over the whole span: queue 0x50's pools of
/// ~113 MB were scanned again after any invalidation anywhere, ~3 us a
/// binding.)
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
    /// This thread's exact ranges found fine (`EXACT_SLOTS`, by `exact_slot`;
    /// an empty slot has `hi` 0).
    static EXACT: std::cell::RefCell<Vec<Fine>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Slots of each thread's table of exact ranges found fine, beside the
/// RANGE_CACHE ranges that may cover a binding: a frame's distinct bindings
/// outnumber those (the graphics CP found no covering range in 42% of its
/// ~15,850 checks a flip at Sec09 Firebridge, 2026-10-28 night). A slot
/// holds the last range hashed there (`DS_FE2_EXACT=0`: none).
const EXACT_SLOTS: usize = 4096;

fn exact_on() -> bool {
    static W: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *W.get_or_init(|| std::env::var("DS_FE2_EXACT").ok().as_deref() != Some("0"))
}

/// The slot of page-aligned [lo, hi) in `EXACT`.
#[inline]
fn exact_slot(lo: u64, hi: u64) -> usize {
    let h = (lo >> 12).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (hi >> 12).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    (h >> (64 - EXACT_SLOTS.trailing_zeros())) as usize
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
fn range_fine(lo: u64, hi: u64, want: u32, marked: bool) -> bool {
    // The exact range first.
    let mut exact_why = None;
    if exact_on() {
        let hit = EXACT.with(|t| {
            let mut t = t.borrow_mut();
            let k = exact_slot(lo, hi);
            let f = t.get(k)?;
            if f.lo != lo || f.hi != hi || (marked && !f.marked) {
                return None;
            }
            match changed(f) {
                Some(c) => {
                    t[k].hi = 0;
                    Some(Err(c))
                }
                None if f.ep >= want => Some(Ok(())),
                None => Some(Err(C_EPOCH)),
            }
        });
        match hit {
            Some(Ok(())) => return true,
            Some(Err(c)) => exact_why = Some(c),
            None => {}
        }
    }
    RANGES.with(|r| {
        let mut r = r.borrow_mut();
        // (Why not, for `DS_FE2_CALLERS`: the last covering range's reason.)
        let mut why = exact_why.unwrap_or(C_NO_RANGE);
        let mut i = 0;
        while i < r.len() {
            let f = &r[i];
            if f.lo <= lo && hi <= f.hi && (f.marked || !marked) {
                let ep_ok = f.ep >= want;
                match changed(f) {
                    // Generations and events only grow: it never holds again.
                    Some(c) => {
                        why = c;
                        r.remove(i);
                        continue;
                    }
                    None if ep_ok => {
                        // (One step forward a hit: no move of the whole list.)
                        if i > 0 {
                            r.swap(i, i - 1);
                        }
                        return true;
                    }
                    None => why = C_EPOCH,
                }
            }
            i += 1;
        }
        if callers_on() {
            caller_count(why, 1);
        }
        false
    })
}

/// What changed in `f`'s chunks (C_GENS) or host leaves (C_HOST) since it
/// was found fine, compared as read (None: nothing).
#[inline]
fn changed(f: &Fine) -> Option<usize> {
    let (c0, c1) = (((f.lo - MIRROR_BASE) >> CHUNK_SHIFT) as usize, ((f.hi - 1 - MIRROR_BASE) >> CHUNK_SHIFT) as usize);
    for (k, &g) in (c0..=c1).zip(f.gens.iter()) {
        let p = TABLE[k].load(Ordering::Acquire);
        // (A chunk made later starts at generation 0.)
        let now = if p.is_null() { 0 } else { unsafe { &*p }.cgen.load(Ordering::Acquire) };
        if now != g {
            return Some(C_GENS);
        }
    }
    let (mut a, mut k) = (f.lo, 0);
    while a < f.hi && k < FINE_LEAVES {
        if crate::hosted::leaf_events(a) != f.host[k] {
            return Some(C_HOST);
        }
        k += 1;
        a = crate::hosted::leaf_end(a);
    }
    None
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
static BY_CALLER: [[AtomicU64; 22]; 4] = [const { [const { AtomicU64::new(0) }; 22] }; 4];
/// The same for placements made while a batch's (or parallel chunk's) first
/// command is bound (`set_first_command`): first commands cost 100-216 us
/// against ~20 us for later ones at Castlewall's arrival, most of it
/// binding (FINDINGS.md 2026-10-31).
static BY_CALLER_FIRST: [[AtomicU64; 22]; 4] = [const { [const { AtomicU64::new(0) }; 22] }; 4];

thread_local! {
    /// The command being bound opens its batch (`set_first_command`).
    static FIRST_COMMAND: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Whether the command this thread binds now opens its batch (the
/// executor's draws and dispatches; `DS_FE2_CALLERS`).
#[inline]
pub fn set_first_command(first: bool) {
    if callers_on() {
        FIRST_COMMAND.with(|f| f.set(first));
    }
}
const C_PLACES: usize = 9;
const C_PLACE_CYCLES: usize = 10;
/// A placement's parts outside `ensure` (TSC cycles): finding its mapping
/// and backing the mirror, beginning the batch and the written range's
/// import, the bookkeeping after `ensure` (its uploads' pages, deferred CP
/// writes).
const C_PART_MAP: usize = 11;
const C_PART_BEGIN: usize = 12;
const C_PART_AFTER: usize = 13;
/// Of C_PART_BEGIN, the batch's begin alone (the rest: the written range's
/// import, `exec::span`).
const C_PART_BATCH: usize = 18;
/// The checks' parts (TSC cycles): scanning words, arming (`try_arm`),
/// takes, making runs current (`fix_run`).
const C_SCAN: usize = 14;
const C_ARM: usize = 15;
const C_TAKE: usize = 16;
const C_FIX: usize = 17;
/// More of the checks' parts: the range caches' lookup, following another
/// thread's take (`follow_take`, `followed`), uploads made without a take
/// (pages written lately, pages a take of the epoch found lacking).
const C_CACHE: usize = 19;
const C_FOLLOW: usize = 20;
const C_UPLOAD: usize = 21;

/// The TSC when `DS_FE2_CALLERS` is on, else 0.
#[inline]
fn tsc() -> u64 {
    if callers_on() {
        unsafe { core::arch::x86_64::_rdtsc() }
    } else {
        0
    }
}

/// Adds the cycles since `t0` (`tsc`) to part `what`.
#[inline]
fn caller_since(what: usize, t0: u64) {
    if t0 != 0 {
        caller_count(what, unsafe { core::arch::x86_64::_rdtsc() }.wrapping_sub(t0));
    }
}

/// Cycles of a placement's parts (`DS_FE2_CALLERS`): map, begin (of it
/// the batch's begin alone), after.
pub fn note_place_parts(map: u64, begin: u64, batch: u64, after: u64) {
    caller_count(C_PART_MAP, map);
    caller_count(C_PART_BEGIN, begin);
    caller_count(C_PART_BATCH, batch);
    caller_count(C_PART_AFTER, after);
}

/// Clears the checks by caller (`profile reset`).
pub fn reset_callers() {
    reset_up_census();
    for c in BY_CALLER.iter().flatten().chain(BY_CALLER_FIRST.iter().flatten()).chain(QUIET_TAKES.iter().flatten()).chain(BY_SOURCE.iter().flatten()) {
        c.store(0, Ordering::Relaxed);
    }
}

/// `DS_FE2_QUIET_CENSUS=1` (diagnosis): each take classed by whether every
/// page of its range is GPU-written with no CPU write found for QUIET_FLIPS
/// flips (what view-backed GPU-quiet mappings would spare: FRONTEND2.md
/// "GPU-quiet mappings on sections"), by caller (`set_caller`).
pub fn quiet_census_on() -> bool {
    static W: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *W.get_or_init(|| std::env::var("DS_FE2_QUIET_CENSUS").ok().as_deref() == Some("1"))
}

/// `DS_FE2_UPCENSUS=1` (diagnosis): every page uploaded into the mirror
/// with the flip and its generation (`Chunk::pgen`: CPU writes takes found,
/// invalidations) at its last upload, so an upload of a page unchanged since
/// an upload of it in the same flip shows: a copy of what another batch's
/// prologue holds already (pages become valid only when their batch is
/// queued, and a parallel run's chunks check the same ranges together), by
/// caller and by the path that uploaded it (`UP_KINDS`). Pages written lately
/// are uploaded without a take, so their generation misses writes since the
/// last take: theirs may be new contents.
pub fn up_census_on() -> bool {
    static W: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *W.get_or_init(|| std::env::var("DS_FE2_UPCENSUS").ok().as_deref() == Some("1"))
}

/// The paths that upload pages into the mirror (`up_census_on`).
const UP_KINDS: [&str; 5] = ["other", "written lately", "a take of the epoch looked at", "after another's take", "after a take"];

thread_local! {
    /// The path uploading now (`UP_KINDS`).
    static UP_KIND: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Sets this thread's upload path until dropped (`up_census_on`).
struct UpKind(usize);

impl UpKind {
    #[inline]
    fn new(k: usize) -> UpKind {
        if !up_census_on() {
            return UpKind(usize::MAX);
        }
        UpKind(UP_KIND.with(|c| c.replace(k)))
    }
}

impl Drop for UpKind {
    #[inline]
    fn drop(&mut self) {
        if self.0 != usize::MAX {
            UP_KIND.with(|c| c.set(self.0));
        }
    }
}

/// Per page of the span: (flip + 1) << 32 | its generation at its last
/// upload; 0: never uploaded.
static UP_LAST: std::sync::OnceLock<Box<[AtomicU64]>> = std::sync::OnceLock::new();
/// By caller and path: [pages uploaded, of them unchanged since an upload
/// in the same flip, unchanged since their last upload in an earlier flip].
static UP_COUNTS: [[[AtomicU64; 3]; 5]; 4] = [const { [const { [const { AtomicU64::new(0) }; 3] }; 5] }; 4];
/// The flip being counted and the flips with the most pages uploaded again
/// (`up_census_on`).
struct UpFlips {
    flip: u64,
    pages: u64,
    again: u64,
    /// This flip's pages uploaded again by caller and path.
    by: [[u64; 5]; 4],
    /// (again, pages, flip, its largest callers and paths), most first.
    worst: Vec<(u64, u64, u64, String)>,
    flips: u64,
}

static UP_FLIPS: Mutex<UpFlips> = Mutex::new(UpFlips { flip: 0, pages: 0, again: 0, by: [[0; 5]; 4], worst: Vec::new(), flips: 0 });

/// Pages [a, b) are uploaded into the mirror by this thread's batch
/// (`up_census_on`).
pub fn upload_census(a: u64, b: u64) {
    if HELD_MAP.get().is_some() {
        // (The hold census: pages uploaded while held.)
        let n = (a & !(PAGE - 1)..b).step_by(PAGE as usize).filter(|&p| crate::hosted::held(p)).count() as u64;
        if n > 0 {
            HOLD[H_UPLOADED].fetch_add(n, Ordering::Relaxed);
        }
    }
    if !up_census_on() {
        return;
    }
    // (Zeroed pages: only those of pages uploaded become resident.)
    let last = UP_LAST.get_or_init(|| unsafe { Box::<[AtomicU64]>::new_zeroed_slice(((MIRROR_END - MIRROR_BASE) / PAGE) as usize).assume_init() });
    let flip = super::FRAMES.load(Ordering::Relaxed) + 1;
    let (mut n, mut again, mut later) = (0u64, 0u64, 0u64);
    let mut p = (a & !(PAGE - 1)).max(MIRROR_BASE);
    let hi = b.min(MIRROR_END);
    while p < hi {
        let pg = chunk(p, false).map_or(0, |c| c.pgen[idx(p)].load(Ordering::Acquire));
        let old = last[((p - MIRROR_BASE) / PAGE) as usize].swap(flip << 32 | pg as u64, Ordering::AcqRel);
        n += 1;
        if old != 0 && old as u32 == pg {
            if old >> 32 == flip {
                again += 1;
            } else {
                later += 1;
            }
        }
        p += PAGE;
    }
    let (caller, kind) = (CALLER.with(|c| c.get()), UP_KIND.with(|c| c.get()));
    let c = &UP_COUNTS[caller][kind];
    c[0].fetch_add(n, Ordering::Relaxed);
    c[1].fetch_add(again, Ordering::Relaxed);
    c[2].fetch_add(later, Ordering::Relaxed);
    let mut g = UP_FLIPS.lock();
    if g.flip != flip {
        if g.flip != 0 {
            // (Its three largest callers and paths of pages uploaded again.)
            let mut by: Vec<(u64, usize, usize)> = (0..4).flat_map(|i| (0..5).map(move |k| (i, k))).map(|(i, k)| (g.by[i][k], i, k)).filter(|x| x.0 != 0).collect();
            by.sort_unstable_by(|x, y| y.cmp(x));
            let top: Vec<String> = by.iter().take(3).map(|&(v, i, k)| format!("{} {} {}", CALLER_NAMES[i], UP_KINDS[k], v)).collect();
            let row = (g.again, g.pages, g.flip, top.join(", "));
            g.worst.push(row);
            g.worst.sort_unstable_by(|x, y| (y.0, y.1, y.2).cmp(&(x.0, x.1, x.2)));
            g.worst.truncate(8);
            g.flips += 1;
        }
        (g.flip, g.pages, g.again, g.by) = (flip, 0, 0, [[0; 5]; 4]);
    }
    g.pages += n;
    g.again += again;
    g.by[caller][kind] += again;
}

/// Run-shared uploads (`Exec::fe2_share`): per page of the span, the
/// parallel run and part whose batch claimed it for its copy and the page's
/// generation then (`Chunk::pgen`, low 16 bits): run << 24 | part << 16 |
/// generation; 0: none. Of a run, the earliest part's claim is kept.
static RUN_UP: std::sync::OnceLock<Box<[AtomicU64]>> = std::sync::OnceLock::new();
/// Per 64-page word of the span, the run that last claimed a page there
/// (`run_claim`): a lookup for another run needs no page entry (every page a
/// check needs is looked up, and at an arrival the page entries' loads cost
/// the workers ~6% more CPU a flip; 2026-10-31). Runs of other queues
/// claiming in the same word take it from each other: their lookups then
/// miss, and those pages are copied again, as without sharing.
static RUN_WORD: std::sync::OnceLock<Box<[AtomicU64]>> = std::sync::OnceLock::new();
/// Runs that claimed a page, by their low bits (`run_any`): a check by a
/// part of a run without claims passes its pages by whole runs as a batch
/// without uploads does (`ensure`'s `mine_any`).
static RUN_ANY: [AtomicU64; 256] = [const { AtomicU64::new(0) }; 256];
/// [page lookups that found a page claimed by an earlier part or batch of
/// the run, pages claimed, pages left to an earlier part's claim].
static RUN_SHARED: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
/// Parts past this share nothing (the entry's field is 8 bits).
pub const RUN_CHUNKS: u32 = 254;

fn run_up() -> &'static [AtomicU64] {
    // (Zeroed pages: only those of pages a run uploads become resident.)
    RUN_UP.get_or_init(|| unsafe { Box::<[AtomicU64]>::new_zeroed_slice(((MIRROR_END - MIRROR_BASE) / PAGE) as usize).assume_init() })
}

fn run_word() -> &'static [AtomicU64] {
    RUN_WORD.get_or_init(|| unsafe { Box::<[AtomicU64]>::new_zeroed_slice(((MIRROR_END - MIRROR_BASE) / PAGE / 64) as usize).assume_init() })
}

#[inline]
fn page_gen16(p: u64) -> u64 {
    chunk(p, false).map_or(0, |c| c.pgen[idx(p)].load(Ordering::Acquire) as u64 & 0xffff)
}

/// Claims pages [a, b) (page-aligned) for the batch of part `chunk` of
/// parallel run `run` (`Exec::fe2_share`; `chunk` at most RUN_CHUNKS;
/// part 0 the main command processor's own, before the workers' chunks): a
/// page an earlier part or batch of the run claimed, unchanged since, is
/// left to that claim; the others are claimed and pushed onto `out` as runs,
/// for copies into this batch's prologue. One atomic step per page: the
/// workers that followed one take wake together, and looked up before any of
/// them had claimed, each would copy every page. Pages outside the span are
/// pushed as they are.
pub fn run_claim(a: u64, b: u64, run: u64, chunk: u32, out: &mut Vec<(u64, u64)>) {
    let mut push = |x: u64, y: u64| match out.last_mut() {
        Some(l) if l.1 == x => l.1 = y,
        _ => out.push((x, y)),
    };
    if a < MIRROR_BASE || b > MIRROR_END {
        push(a, b);
        return;
    }
    let t = run_up();
    let words = run_word();
    let r = run & 0xff_ffff_ffff;
    let any = &RUN_ANY[(r & 255) as usize];
    if any.load(Ordering::Relaxed) != r {
        any.store(r, Ordering::Release);
    }
    let (mut claimed, mut left) = (0u64, 0u64);
    let mut p = a;
    while p < b {
        // (The word's tag before the page's claim: a lookup that sees the
        // claim sees the tag.)
        let w = &words[((p - MIRROR_BASE) / PAGE / 64) as usize];
        if w.load(Ordering::Relaxed) != r {
            w.store(r, Ordering::Release);
        }
        let g = page_gen16(p);
        let new = r << 24 | (chunk as u64) << 16 | g;
        let e = &t[((p - MIRROR_BASE) / PAGE) as usize];
        let won = e
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |old| {
                let earlier = old >> 24 == r && (old >> 16 & 0xff) as u32 <= chunk && old & 0xffff == g;
                (!earlier).then_some(new)
            })
            .is_ok();
        if won {
            claimed += 1;
            push(p, p + PAGE);
        } else {
            left += 1;
        }
        p += PAGE;
    }
    RUN_SHARED[1].fetch_add(claimed, Ordering::Relaxed);
    RUN_SHARED[2].fetch_add(left, Ordering::Relaxed);
}

/// Whether a part of parallel run `run` claimed a page (`run_claim`; false
/// as well once a run 256 ids later claimed one).
#[inline]
pub fn run_any(run: u64) -> bool {
    RUN_ANY[(run & 255) as usize].load(Ordering::Acquire) == run & 0xff_ffff_ffff
}

/// Whether page `p` is claimed by a batch of part `chunk` or an earlier part
/// of parallel run `run`, unchanged since (`run_claim`):
/// that batch is queued before the asking one, its copies landing first.
#[inline]
pub fn run_shared(p: u64, run: u64, chunk: u32) -> bool {
    let (Some(t), Some(words)) = (RUN_UP.get(), RUN_WORD.get()) else { return false };
    if !(MIRROR_BASE..MIRROR_END).contains(&p) {
        return false;
    }
    let i = ((p - MIRROR_BASE) / PAGE) as usize;
    if words[i / 64].load(Ordering::Acquire) != run & 0xff_ffff_ffff {
        return false;
    }
    let v = t[i].load(Ordering::Acquire);
    let hit = v >> 24 == run & 0xff_ffff_ffff && (v >> 16 & 0xff) as u32 <= chunk && v & 0xffff == page_gen16(p);
    if hit {
        // (Counted per thread: every worker adding to one word made the
        // lookups slow.)
        RUN_HITS.with(|c| {
            let n = c.get() + 1;
            if n >= 4096 {
                RUN_SHARED[0].fetch_add(n, Ordering::Relaxed);
                c.set(0);
            } else {
                c.set(n);
            }
        });
    }
    hit
}

thread_local! {
    /// This thread's lookups served not yet added to RUN_SHARED.
    static RUN_HITS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Status part: run-shared uploads (`Exec::fe2_share`).
fn run_share_line() -> String {
    let [hits, claimed, left] = [0, 1, 2].map(|k| RUN_SHARED[k].load(Ordering::Relaxed));
    if hits == 0 && claimed == 0 {
        return String::new();
    }
    format!("    front end 2's run-shared uploads (`Exec::fe2_share`): {claimed} pages claimed by parallel runs' parts for their copies, {left} left to an earlier part's claim at the copy; {hits} page lookups found a page claimed by an earlier part or batch of the run\n")
}

/// Clears the upload census's counts (`profile reset`; the pages' last
/// uploads stay).
fn reset_up_census() {
    for c in UP_COUNTS.iter().flatten().flatten() {
        c.store(0, Ordering::Relaxed);
    }
    *UP_FLIPS.lock() = UpFlips { flip: 0, pages: 0, again: 0, by: [[0; 5]; 4], worst: Vec::new(), flips: 0 };
}

/// The upload census (`DS_FE2_UPCENSUS=1`; "" otherwise).
fn up_census_line() -> String {
    if !up_census_on() {
        return String::new();
    }
    let mib = |pages: u64| pages as f64 * PAGE as f64 / (1u64 << 20) as f64;
    let mut parts = Vec::new();
    let mut all = [0u64; 3];
    for (name, rows) in CALLER_NAMES.iter().zip(UP_COUNTS.iter()) {
        let mut kinds = Vec::new();
        for (kind, c) in UP_KINDS.iter().zip(rows.iter()) {
            let v: Vec<u64> = c.iter().map(|x| x.load(Ordering::Relaxed)).collect();
            if v[0] == 0 {
                continue;
            }
            for k in 0..3 {
                all[k] += v[k];
            }
            kinds.push(format!("{kind} {} pages ({:.0} MiB; unchanged since an upload in the flip {}, since one before it {})", v[0], mib(v[0]), v[1], v[2]));
        }
        if !kinds.is_empty() {
            parts.push(format!("{name}: {}", kinds.join(", ")));
        }
    }
    let g = UP_FLIPS.lock();
    let flips = g.flips.max(1);
    let worst: Vec<String> = g.worst.iter().map(|(again, pages, flip, top)| format!("flip {} {} of {} pages ({:.0} of {:.0} MiB; {top})", flip - 1, again, pages, mib(*again), mib(*pages))).collect();
    format!(
        "    front end 2's uploads into the mirror (`DS_FE2_UPCENSUS`): {} pages ({:.1} MiB a flip over {} flips), {} unchanged since an upload in the same flip ({:.1} MiB a flip), {} since one in an earlier flip; by caller and path: {}; most uploaded again in a flip: {}\n",
        all[0],
        mib(all[0]) / flips as f64,
        flips,
        all[1],
        mib(all[1]) / flips as f64,
        all[2],
        parts.join("; "),
        worst.join(", ")
    )
}

/// Flips without a CPU write found (`Chunk::wrote`; a page's first sight
/// counts as one) that make a GPU-written page quiet (`quiet_census_on`).
const QUIET_FLIPS: u64 = 300;

/// Per caller: [takes, ns, quiet takes, their ns] (`quiet_census_on`).
static QUIET_TAKES: [[AtomicU64; 4]; 4] = [const { [const { AtomicU64::new(0) }; 4] }; 4];

/// Whether every page of [a, b) is GPU-written and quiet (`QUIET_FLIPS`).
fn quiet_range(a: u64, b: u64, frame: u64) -> bool {
    let mut x = a;
    while x < b {
        let Some(c) = chunk(x, false) else { return false };
        let i = idx(x);
        if c.gpu[i / 64].load(Ordering::Relaxed) >> (i % 64) & 1 == 0 {
            return false;
        }
        let w = c.wrote[i].load(Ordering::Relaxed) as u64;
        if w != 0 && frame.saturating_sub(w - 1) < QUIET_FLIPS {
            return false;
        }
        x += PAGE;
    }
    true
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
    let k = CALLER.with(|c| c.get());
    BY_CALLER[k][what].fetch_add(n, Ordering::Relaxed);
    if FIRST_COMMAND.with(|f| f.get()) {
        BY_CALLER_FIRST[k][what].fetch_add(n, Ordering::Relaxed);
    }
}

fn note_range_fine(lo: u64, hi: u64, gens: [u32; FINE_CHUNKS], host: [u64; FINE_LEAVES], ep: u32, marked: bool) {
    if exact_on() {
        EXACT.with(|t| {
            let mut t = t.borrow_mut();
            if t.is_empty() {
                t.resize(EXACT_SLOTS, Fine { lo: 0, hi: 0, gens: [0; FINE_CHUNKS], host: [0; FINE_LEAVES], ep: 0, marked: false });
            }
            t[exact_slot(lo, hi)] = Fine { lo, hi, gens, host, ep, marked };
        });
    }
    RANGES.with(|r| {
        let mut r = r.borrow_mut();
        r.retain(|f| !(lo <= f.lo && f.hi <= hi && (marked || !f.marked)));
        r.insert(0, Fine { lo, hi, gens, host, ep, marked });
        r.truncate(RANGE_CACHE);
    });
}

static TABLE: [AtomicPtr<Chunk>; CHUNKS] = [const { AtomicPtr::new(std::ptr::null_mut()) }; CHUNKS];
static ON: AtomicBool = AtomicBool::new(false);

/// Front end 2 wanted: the default since 2026-10-30 (the user's direction);
/// `DS_FE2=0` runs the old front end, kept as a fallback only. Read once:
/// shader specs follow it, so the pipelines precompiled at a session's start
/// match.
pub fn wanted() -> bool {
    static W: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *W.get_or_init(|| std::env::var("DS_FE2").ok().as_deref() != Some("0"))
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

/// Diagnosis (`DS_FE2_CHECK_COPIES=1`): copy kernels' dispatches outside the
/// graphics CP checked after their batches (`Exec::dispatch_inner`).
pub fn check_copies() -> bool {
    static W: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *W.get_or_init(|| std::env::var("DS_FE2_CHECK_COPIES").ok().as_deref() == Some("1"))
}

/// `DS_FE2_STRETCH_AUDIT=1` (diagnosis): at its label every stretch past a
/// label wait whose checks needed an older epoch than its label does is
/// checked again (`Exec::past_current_for`) and counted stale or not, as
/// waits passed on GPU writes check them, but kept either way.
pub fn stretch_audit() -> bool {
    static W: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *W.get_or_init(|| std::env::var("DS_FE2_STRETCH_AUDIT").ok().as_deref() == Some("1"))
}

/// Held past a label wait, the graphics command processor's checks need the
/// epoch now, not its view (`DS_FE2_STRETCH_VIEW=1`: the view, as before
/// 2026-10-28 night's fix; `Exec::fe2_place_inner`).
pub fn stretch_epoch() -> bool {
    static W: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *W.get_or_init(|| std::env::var("DS_FE2_STRETCH_VIEW").ok().as_deref() != Some("1"))
}

/// Small runs are uploaded without a take while their pages are within their
/// quiet time, not only HOT_FLIPS after a write (`hot_run`; `DS_FE2_HOT_QUIET=0`:
/// HOT_FLIPS only, as before 2026-10-28).
fn hot_quiet() -> bool {
    static W: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *W.get_or_init(|| std::env::var("DS_FE2_HOT_QUIET").ok().as_deref() != Some("0"))
}

/// Where pages may be armed (the executor's import list): calls its third
/// argument with the parts of [a, b) inside settled import windows (asked
/// again under the arm lock: `arm_part`); false: the list was busy.
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
        varmed: std::array::from_fn(|_| AtomicU64::new(0)),
        gpu: std::array::from_fn(|_| AtomicU64::new(0)),
        gpu_all: AtomicBool::new(false),
        wclean: std::array::from_fn(|_| AtomicU32::new(0)),
        wmax: std::array::from_fn(|_| AtomicU32::new(0)),
        wgen: std::array::from_fn(|_| AtomicU32::new(0)),
        clean: (0..CHUNK_PAGES).map(|_| AtomicU32::new(0)).collect(),
        wrote: (0..CHUNK_PAGES).map(|_| AtomicU32::new(0)).collect(),
        pgen: (0..CHUNK_PAGES).map(|_| AtomicU32::new(0)).collect(),
        level: (0..CHUNK_PAGES).map(|_| AtomicU8::new(0)).collect(),
        uses: std::array::from_fn(|_| AtomicU8::new(0)),
        use_ep: std::array::from_fn(|_| AtomicU32::new(0)),
        seen: std::array::from_fn(|_| AtomicU64::new(0)),
        cgen: Line(AtomicU32::new(0)),
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

/// Whether chunk `c` (at `a`) has a summary that meets `want`.
#[inline]
fn summary_ok(c: &Chunk, a: u64, want: u32) -> bool {
    let s = c.sum.load(Ordering::Acquire);
    if s == 0 || (s >> 32) as u32 != c.cgen.load(Ordering::Acquire) {
        return false;
    }
    let e = s as u32;
    e >= want && c.sum_host.load(Ordering::Acquire) == crate::hosted::leaf_events(a)
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
static STATS: [AtomicU64; 34] = [const { AtomicU64::new(0) }; 34];
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
// (28-30: the checks deferred to batch closes, `DS_FE2_DEFER`, removed on
// 2026-11-01.)
// (31, 32: runs uploaded after another take of their epoch, `tune
// fe2_taken_upload`, removed on 2026-11-01.)
/// View pages the residency armed at its takes, armed here too (`adopt_armed`).
const S_ADOPTED: usize = 33;

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

/// Pages of [lo, hi) GPU work wrote (as far as front end 2 knows: its
/// write marks' pages; `fe2census`).
pub fn gpu_pages(lo: u64, hi: u64) -> u64 {
    let (mut x, hi) = ((lo & !(PAGE - 1)).max(MIRROR_BASE), hi.min(MIRROR_END));
    let mut n = 0;
    while x < hi {
        let e = word_end(x, hi);
        if let Some(c) = chunk(x, false) {
            n += (c.gpu[idx(x) / 64].load(Ordering::Relaxed) & word_mask(x, e)).count_ones() as u64;
        }
        x = e;
    }
    n
}

/// Counts pages copied into the mirror by the CPU (no import window).
pub fn note_memcpy(pages: u64) {
    STATS[S_MEMCPY].fetch_add(pages, Ordering::Relaxed);
}

/// A page of a view window the residency armed through the host at a take
/// (`resident::take_written`): armed for front end 2 as if armed here. Its
/// own arming refuses such pages (read-only already), so without this they
/// were taken again after every view raise (2026-10-31, views boot
/// -135103). A take arms a page only after reporting it or finding it clean
/// and armed, and every take's written pages reach `marks_taken`, so the
/// host's bit set at a take leaves no write unseen; a fault clears it, and
/// `watch` ends the arming here as for pages armed here.
pub fn adopt_armed(page: u64) {
    if !on() {
        return;
    }
    let Some(c) = chunk(page, false) else { return };
    let i = idx(page);
    c.armed[i / 64].fetch_or(1u64 << (i % 64), Ordering::AcqRel);
    STATS[S_ADOPTED].fetch_add(1, Ordering::Relaxed);
}

/// Counts written bindings whose pages no import window covers (their GPU
/// writes cannot go back).
pub fn note_unscatterable() {
    STATS[S_UNSCATTERABLE].fetch_add(1, Ordering::Relaxed);
}

/// Pages of [a, e) (one word) that need no check for requirement `want`:
/// valid, and armed (the host's bit still set) or found clean at or after
/// epoch `want`.
#[inline]
fn word_ok(c: &Chunk, a: u64, e: u64, want: u32) -> u64 {
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
    if c.wclean[w].load(Ordering::Acquire) >= want {
        return v;
    }
    // (A stamp stored after this read only makes the check conservative.)
    if c.wmax[w].load(Ordering::Acquire) < want {
        return armed;
    }
    let mut ok = armed;
    let base = w * 64;
    let mut r = rest;
    while r != 0 {
        let b = r.trailing_zeros() as usize;
        r &= r - 1;
        if c.clean[base + b].load(Ordering::Acquire) >= want {
            ok |= 1u64 << b;
        }
    }
    ok
}

/// The checks' commonest counts, kept per thread and added to STATS and
/// BY_SIZE every LOCAL_FLUSH checks (shared atomics on every check, from
/// every recording thread, bounced their cache lines: ~45,000 checks a flip
/// at the vista, four to six adds each). Slots: checks, their pages, range
/// cache hits, checks needing nothing, chunks passed by their summary, then
/// checks and pages per size class.
const L_CHECKS: usize = 0;
const L_PAGES: usize = 1;
const L_HITS: usize = 2;
const L_FAST: usize = 3;
const L_SUMMARY: usize = 4;
const L_CLASS: usize = 5;
const LOCAL_SLOTS: usize = L_CLASS + 8;
const LOCAL_FLUSH: u64 = 256;

thread_local! {
    static LOCAL: std::cell::RefCell<[u64; LOCAL_SLOTS]> = const { std::cell::RefCell::new([0; LOCAL_SLOTS]) };
}

#[inline]
fn local_add(k: usize, n: u64) {
    LOCAL.with(|l| l.borrow_mut()[k] += n);
}

/// A check of `pages` pages in size class `class` begins (the thread's
/// counts go to the shared ones every LOCAL_FLUSH checks).
#[inline]
fn local_check(class: usize, pages: u64) {
    LOCAL.with(|l| {
        let mut l = l.borrow_mut();
        l[L_CHECKS] += 1;
        l[L_PAGES] += pages;
        l[L_CLASS + 2 * class] += 1;
        l[L_CLASS + 2 * class + 1] += pages;
        if l[L_CHECKS] >= LOCAL_FLUSH {
            for (k, s) in [(L_CHECKS, S_ENSURES), (L_PAGES, S_ENSURE_PAGES), (L_HITS, S_RANGE_HITS), (L_FAST, S_FAST), (L_SUMMARY, S_SUMMARY)] {
                STATS[s].fetch_add(std::mem::take(&mut l[k]), Ordering::Relaxed);
            }
            for c in 0..4 {
                BY_SIZE[c][0].fetch_add(std::mem::take(&mut l[L_CLASS + 2 * c]), Ordering::Relaxed);
                BY_SIZE[c][1].fetch_add(std::mem::take(&mut l[L_CLASS + 2 * c + 1]), Ordering::Relaxed);
            }
        }
    });
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
/// `want`: pages must reflect every CPU write before epoch `want`
/// began (the caller's view, `views_on`); `ep`: the epoch now, read before
/// any take (what takes stamp their pages with).
#[allow(clippy::too_many_arguments)]
pub fn ensure(lo: u64, hi: u64, want: u32, ep: u32, gpu_writes: bool, mine: &dyn Fn(u64) -> bool, mine_any: bool, upload: &mut dyn FnMut(u64, u64) -> bool, armable: &Armable) -> Result<(), &'static str> {
    if !callers_on() {
        return ensure_inner(lo, hi, want, ep, gpu_writes, mine, mine_any, upload, armable);
    }
    let t0 = unsafe { core::arch::x86_64::_rdtsc() };
    caller_count(C_CHECKS, 1);
    source_count(B_CHECKS, 1);
    let r = ensure_inner(lo, hi, want, ep, gpu_writes, mine, mine_any, upload, armable);
    let cycles = unsafe { core::arch::x86_64::_rdtsc() }.wrapping_sub(t0);
    caller_count(C_CYCLES, cycles);
    source_count(B_CYCLES, cycles);
    r
}

thread_local! {
    /// The stage whose bindings this thread places (`StageSource`; for
    /// `DS_FE2_CALLERS`): 0 none, else `Prepared::ahead`.
    static SOURCE: std::cell::Cell<u8> = const { std::cell::Cell::new(0) };
}

const SOURCE_NAMES: [&str; 3] = ["placements of no stage", "stages it prepared", "stages a worker prepared"];
/// The graphics CP's checks by the stage they bind (`SOURCE`;
/// `DS_FE2_CALLERS`): [stage bindings, checks, range-cache hits, TSC cycles,
/// runs taken, their pages, runs only of pages a take of the epoch wanted
/// had looked at and found lacking, runs with a valid page not looked at
/// since that epoch began (and no lacking one), runs with a page M lacks not
/// looked at since; pages not looked at since: stamped one epoch before,
/// older].
static BY_SOURCE: [[AtomicU64; 11]; 3] = [const { [const { AtomicU64::new(0) }; 11] }; 3];
const B_BINDINGS: usize = 0;
const B_CHECKS: usize = 1;
const B_HITS: usize = 2;
const B_CYCLES: usize = 3;
const B_TAKES: usize = 4;
const B_PAGES: usize = 5;
const B_LACKING: usize = 6;
const B_VALID_STALE: usize = 7;
const B_INVALID_STALE: usize = 8;
const B_ONE_BEHIND: usize = 9;
const B_OLDER: usize = 10;

/// The stage whose bindings this thread places, until dropped (`SOURCE`).
pub struct StageSource;

impl StageSource {
    pub fn set(ahead: u8) -> StageSource {
        if callers_on() {
            SOURCE.with(|s| s.set(ahead.min(2)));
            source_count(B_BINDINGS, 1);
        }
        StageSource
    }
}

impl Drop for StageSource {
    fn drop(&mut self) {
        if callers_on() {
            SOURCE.with(|s| s.set(0));
        }
    }
}

/// Counts `n` for the graphics CP's checks of the stage being bound.
fn source_count(what: usize, n: u64) {
    if CALLER.with(|c| c.get()) == 0 {
        BY_SOURCE[SOURCE.with(|s| s.get()) as usize][what].fetch_add(n, Ordering::Relaxed);
    }
}

/// Why the pages of a run about to be taken need it, for a check wanting
/// epoch `want`: [pages M lacks that a take of that epoch looked at, valid
/// pages not looked at since, pages M lacks not looked at since, of those
/// two stamped one epoch before `want`] (pages in the run that need nothing,
/// merged by `TAKE_GAP`, are not counted).
fn take_reasons(a: u64, b: u64, want: u32) -> [u64; 4] {
    let mut k = [0u64; 4];
    let mut x = a;
    while x < b {
        let e = word_end(x, b);
        let Some(c) = chunk(x, false) else { break };
        let w = idx(x) / 64;
        let need = word_mask(x, e) & !word_ok(c, x, e, want);
        let valid = c.valid[w].load(Ordering::Acquire);
        let wclean = c.wclean[w].load(Ordering::Acquire);
        let mut r = need;
        while r != 0 {
            let bit = r.trailing_zeros() as usize;
            r &= r - 1;
            let stamp = wclean.max(c.clean[w * 64 + bit].load(Ordering::Acquire));
            if valid >> bit & 1 != 0 {
                k[1] += 1;
            } else if stamp >= want {
                k[0] += 1;
                continue;
            } else {
                k[2] += 1;
            }
            if stamp.wrapping_add(1) == want {
                k[3] += 1;
            }
        }
        x = e;
    }
    k
}

/// A run the graphics CP took, with `take_reasons`' counts.
fn note_source_take(a: u64, b: u64, k: [u64; 4]) {
    let c = &BY_SOURCE[SOURCE.with(|s| s.get()) as usize];
    c[B_TAKES].fetch_add(1, Ordering::Relaxed);
    c[B_PAGES].fetch_add((b - a) / PAGE, Ordering::Relaxed);
    let class = if k[2] != 0 {
        B_INVALID_STALE
    } else if k[1] != 0 {
        B_VALID_STALE
    } else {
        B_LACKING
    };
    c[class].fetch_add(1, Ordering::Relaxed);
    c[B_ONE_BEHIND].fetch_add(k[3], Ordering::Relaxed);
    c[B_OLDER].fetch_add(k[1] + k[2] - k[3], Ordering::Relaxed);
}

#[allow(clippy::too_many_arguments)]
fn ensure_inner(lo: u64, hi: u64, want: u32, ep: u32, gpu_writes: bool, mine: &dyn Fn(u64) -> bool, mine_any: bool, upload: &mut dyn FnMut(u64, u64) -> bool, armable: &Armable) -> Result<(), &'static str> {
    // (No page of the caller's batch to pass over: runs go in whole.)
    let mine_fast = !mine_any;
    let sc = size_class((hi - lo) / PAGE);
    local_check(sc, (hi - lo) / PAGE);
    let class = &BY_SIZE[sc];
    // Written ranges mark their pages GPU-written (kept unarmed) unless
    // `DS_FE2_ARM_GPU=1`; a range found fine with its marks set passes the
    // cache like a read one (marking every word on every use, ~430 words
    // for a pool of ~113 MB, made queue 0x50's dispatches slow).
    // (Pages of the mixed range's view windows too: their imports are of
    // the aliases, `mem::view_rules`.)
    let gpu_writes = gpu_writes && !arm_gpu() && !crate::mem::view_span(lo, hi);
    let tc = tsc();
    let fine = range_fine(lo, hi, want, gpu_writes);
    caller_since(C_CACHE, tc);
    if fine {
        local_add(L_HITS, 1);
        local_add(L_FAST, 1);
        if callers_on() {
            caller_count(C_HITS, 1);
            source_count(B_HITS, 1);
        }
        return Ok(());
    }
    // (Before the scan: a change during it shows as a new value.)
    let snap = snapshot(lo, hi);
    let ts = tsc();
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
                if (!gpu_writes || c.gpu_all.load(Ordering::Acquire)) && summary_ok(c, a, want) {
                    local_add(L_SUMMARY, 1);
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
                // (Set bits stay set: the chunks of a run start together and
                // scan the same pools, and a write per word made each scan's
                // words move between their cores.)
                let m = word_mask(a, e);
                if c.gpu[w].load(Ordering::Acquire) & m != m {
                    c.gpu[w].fetch_or(m, Ordering::AcqRel);
                }
            }
            let need = word_mask(a, e) & !word_ok(c, a, e, want);
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
                if mine_fast {
                    match bad.last_mut() {
                        Some(l) if l.1 + TAKE_GAP * PAGE >= pa => l.1 = pe,
                        _ => bad.push((pa, pe)),
                    }
                    continue;
                }
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
                    let c = chunk(start, false).expect("scanned");
                    // (Stored only when they change: threads rescanning a
                    // chunk kept its line moving between their cores.)
                    let sum = (g as u64) << 32 | if armed { SUM_ARMED } else { want } as u64;
                    if ok && (c.sum.load(Ordering::Relaxed) != sum || c.sum_host.load(Ordering::Relaxed) != h) {
                        c.sum_host.store(h, Ordering::Release);
                        c.sum.store(sum, Ordering::Release);
                    }
                    if gpu_writes && !c.gpu_all.load(Ordering::Relaxed) {
                        c.gpu_all.store(true, Ordering::Release);
                    }
                    whole = None;
                }
            }
        }
        if callers_on() {
            caller_count(C_WORDS, words);
            caller_since(C_SCAN, ts);
        }
        if bad.is_empty() {
            local_add(L_FAST, 1);
            // (Not for ranges with pages the caller's batch uploaded: they are
            // fine for its batch only, until it is queued.)
            if let (true, Some((gens, host))) = (mine_none, snap) {
                note_range_fine(lo, hi, gens, host, if all_armed { SUM_ARMED } else { want }, gpu_writes);
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
                let _k = UpKind::new(1);
                let tu = tsc();
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
                caller_since(C_UPLOAD, tu);
                continue;
            }
            // Another thread's take around this run, for a requirement at
            // least ours, instead of one of our own (`follow_take`).
            let tf = tsc();
            if follow_take(a, b, want) {
                let r = {
                    let _k = UpKind::new(3);
                    followed(a, b, want, mine, upload)
                };
                caller_since(C_FOLLOW, tf);
                match r {
                    Ok(true) => continue,
                    Ok(false) => {}
                    Err(e) => {
                        STATS[S_SLOW_NS].fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                        return Err(e);
                    }
                }
            } else {
                caller_since(C_FOLLOW, tf);
            }
            // (Why the graphics CP takes it, for `DS_FE2_CALLERS`.)
            let reasons = (callers_on() && CALLER.with(|c| c.get()) == 0).then(|| take_reasons(a, b, want));
            if let Err(e) = check_run(a, b, ep, frame, !gpu_writes, mine, upload, armable, written, class) {
                STATS[S_SLOW_NS].fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                return Err(e);
            }
            if let Some(k) = reasons {
                note_source_take(a, b, k);
            }
        }
        STATS[S_SLOW_NS].fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        // (A long one is a hitch cause: `gpu::hitch`.)
        super::hitch::note_since(super::hitch::Kind::Mirror, t0, || format!("check of {lo:#x}..{hi:#x}: {} runs, {} pages", bad.len(), bad.iter().map(|r| (r.1 - r.0) / PAGE).sum::<u64>()));
        Ok(())
    })
}

/// Arms (`arm`: hosted, where allowed) and takes [a, b), uploads its pages
/// M lacks (not `mine`), and marks its valid pages clean in epoch `ep`.
#[allow(clippy::too_many_arguments)]
fn check_run(a: u64, b: u64, ep: u32, frame: u64, arm: bool, mine: &dyn Fn(u64) -> bool, upload: &mut dyn FnMut(u64, u64) -> bool, armable: &Armable, written: &mut Vec<u64>, class: &[AtomicU64; 4]) -> Result<(), &'static str> {
    let (ta, tb) = widen(a, b);
    // (Until its pages are stamped: `follow_take`.)
    let _taking = Taking::new(ta, tb, ep);
    take_run(a, b, ta, tb, frame, arm, armable, written, class);
    let t = tsc();
    let r = fix_run(a, b, ep, mine, upload);
    // The words around a small run were taken too: clean in this epoch.
    if ta < a {
        set_clean(ta, a, ep, false);
    }
    if b < tb {
        set_clean(b, tb, ep, false);
    }
    caller_since(C_FIX, t);
    r
}

/// Runs of at most WIDE_TAKE pages are taken with their whole 64-page words,
/// within the readable mappings around them (`widen`): a take costs its call,
/// not its pages, and a word's other pages are then clean in its epoch, so
/// later checks of small ranges beside it in the same view need no take of
/// their own (at the gate ~3,650 takes a flip were of ranges under 64 KiB with
/// GPU-written pages unarmed: 2026-10-28; `DS_FE2_WIDE_TAKES=0`: the run
/// alone).
const WIDE_TAKE: u64 = 16;

fn wide_takes() -> bool {
    static W: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *W.get_or_init(|| std::env::var("DS_FE2_WIDE_TAKES").ok().as_deref() != Some("0"))
}

/// The range a check's take of [a, b) covers (`WIDE_TAKE`): the run of
/// GPU-written pages around it within its words and mappings. (Whole words
/// took the CPU's per-frame data beside small ranges too: a take that finds
/// written pages resets them, its costly part, and the gate's takes went
/// from ~4,100 to ~2,250 a flip but from ~27 to ~31 ms; 2026-10-28.)
fn widen(a: u64, b: u64) -> (u64, u64) {
    if (b - a) / PAGE > WIDE_TAKE || !wide_takes() {
        return (a, b);
    }
    // (`tune fe2_wide_pages`: the window in pages, whole words.)
    let w = super::tune::fe2_wide_pages().max(64).next_multiple_of(64) * PAGE;
    let (wa, wb) = (a - a % w, b.next_multiple_of(w));
    let Some((s, e)) = crate::mem::readable_span(a).filter(|&(s, e)| s <= a && b <= e) else { return (a, b) };
    let (lo, hi) = (wa.max(s.next_multiple_of(PAGE)), wb.min(e & !(PAGE - 1)));
    // (Past one word only pages a take saw before: a first look at a page
    // counts as a write, `set_clean`, and its next checks would upload it.)
    let seen_only = w > 64 * PAGE;
    let gpu = |p: u64| chunk(p, false).is_some_and(|c| (c.gpu[idx(p) / 64].load(Ordering::Relaxed) & if seen_only { c.seen[idx(p) / 64].load(Ordering::Relaxed) } else { u64::MAX }) >> (idx(p) % 64) & 1 != 0);
    let (mut ta, mut tb) = (a, b);
    while ta > lo && gpu(ta - PAGE) {
        ta -= PAGE;
    }
    while tb < hi && gpu(tb) {
        tb += PAGE;
    }
    (ta, tb)
}

/// Arms [a, b) (`arm`: hosted, where allowed) and takes [ta, tb) (around it:
/// `widen`): pages written since their last take lose their valid bit.
#[allow(clippy::too_many_arguments)]
fn take_run(a: u64, b: u64, ta: u64, tb: u64, frame: u64, arm: bool, armable: &Armable, written: &mut Vec<u64>, class: &[AtomicU64; 4]) {
    if arm && crate::hosted::on() {
        let t = tsc();
        try_arm(a, b, frame, armable);
        caller_since(C_ARM, t);
    }
    // The take: pages written since their last take lose their valid bit
    // (`marks_taken`, for any take's pages).
    written.clear();
    let quiet = quiet_census_on().then(|| quiet_range(ta, tb, frame));
    hole_snapshot(ta, tb);
    let tt = std::time::Instant::now();
    let t = tsc();
    let r = super::precache::take_written_for(ta, tb - ta, written, None, crate::mem::WW_WHO_FE2);
    caller_since(C_TAKE, t);
    if r.is_some() {
        hole_check(written);
    }
    if wb_buckets() {
        for &pg in written.iter() {
            note_wb_pages(&WB_CPU_PAGES, pg, PAGE);
        }
    }
    super::hitch::note_since(super::hitch::Kind::Mirror, tt, || format!("take of {ta:#x}..{tb:#x} ({} pages)", (tb - ta) / PAGE));
    let tns = tt.elapsed().as_nanos() as u64;
    STATS[S_TAKE_NS].fetch_add(tns, Ordering::Relaxed);
    STATS[S_TAKES].fetch_add(1, Ordering::Relaxed);
    if callers_on() {
        caller_count(C_TAKES, 1);
    }
    class[2].fetch_add(1, Ordering::Relaxed);
    class[3].fetch_add(tns, Ordering::Relaxed);
    if let Some(q) = quiet {
        let s = &QUIET_TAKES[CALLER.with(|c| c.get())];
        s[0].fetch_add(1, Ordering::Relaxed);
        s[1].fetch_add(tns, Ordering::Relaxed);
        if q {
            s[2].fetch_add(1, Ordering::Relaxed);
            s[3].fetch_add(tns, Ordering::Relaxed);
        }
    }
    STATS[S_TAKE_WRITTEN].fetch_add(written.len() as u64, Ordering::Relaxed);
    if super::fe2census::on() {
        super::fe2census::note_take(ta, tb, written, CALLER.with(|c| c.get()));
    }
    if r.is_none() {
        STATS[S_TAKE_FAILED].fetch_add(1, Ordering::Relaxed);
        invalidate(ta, tb - ta);
    } else if take_all() {
        // Diagnosis: as if the take found every page written (what takes
        // on section-backed memory report for pages they cannot arm).
        invalidate(ta, tb - ta);
    }
}

thread_local! {
    /// `take_run`'s words with armed pages as its take began: (word address,
    /// pages armed and taken since, pages armed).
    static HOLE_WORDS: std::cell::RefCell<Vec<(u64, u64, u64)>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Pages our takes found written while armed and taken since (no fault
/// ended the arming: a write the protection missed, whose page the mirror
/// would have kept stale where it trusts armed pages).
static HOLES: AtomicU64 = AtomicU64::new(0);

/// Before a take of [ta, tb): its pages armed (the host's bit set), and
/// those of them taken since their arming (`Chunk::varmed`).
fn hole_snapshot(ta: u64, tb: u64) {
    HOLE_WORDS.with(|h| {
        let mut h = h.borrow_mut();
        h.clear();
        let (mut x, hi) = (ta.max(MIRROR_BASE), tb.min(MIRROR_END));
        while x < hi {
            let e = word_end(x, hi);
            if let Some(c) = chunk(x, false) {
                let w = idx(x) / 64;
                let armed = c.armed[w].load(Ordering::Acquire) & word_mask(x, e);
                if armed != 0 {
                    let wa = x & !(64 * PAGE - 1);
                    let now = armed & crate::hosted::armed_word(wa);
                    if now != 0 {
                        h.push((wa, c.varmed[w].load(Ordering::Acquire) & now, now));
                    }
                }
            }
            x = e;
        }
    });
}

/// After that take (it found `written`): a page armed and taken since
/// before it began, found written and still armed and taken since (no
/// fault cleared its bit, no new arming began) is a hole; the pages armed
/// as it began and still armed are taken since their arming now.
fn hole_check(written: &[u64]) {
    HOLE_WORDS.with(|h| {
        let h = h.borrow();
        if h.is_empty() {
            return;
        }
        for &p in written {
            let wa = p & !(64 * PAGE - 1);
            let Ok(k) = h.binary_search_by_key(&wa, |e| e.0) else { continue };
            let bit = 1u64 << (idx(p) % 64);
            if h[k].1 & bit == 0 {
                continue;
            }
            let Some(c) = chunk(p, false) else { continue };
            let w = idx(p) / 64;
            if c.varmed[w].load(Ordering::Acquire) & c.armed[w].load(Ordering::Acquire) & crate::hosted::armed_word(wa) & bit != 0 {
                let n = HOLES.fetch_add(1, Ordering::Relaxed);
                if n < 32 {
                    // (try_lock: the caller may hold the memory map.)
                    let map = crate::mem::MEM.try_lock().map_or_else(|| "busy".into(), |m| m.find(p).map_or_else(|| "none".into(), |v| format!("{:#x}+{:#x} {:?} '{}'", v.start, v.size, v.kind, v.name)));
                    crate::warn!("fe2: a take found page {p:#x} written while armed and taken since (no write fault; flip {}, thread {}; mapping {map})", super::FRAMES.load(Ordering::Relaxed), std::thread::current().name().unwrap_or("?"));
                }
            }
        }
        for &(wa, _, now) in h.iter() {
            let Some(c) = chunk(wa, false) else { continue };
            let w = idx(wa) / 64;
            let live = now & c.armed[w].load(Ordering::Acquire) & crate::hosted::armed_word(wa);
            if live != 0 {
                c.varmed[w].fetch_or(live, Ordering::AcqRel);
            }
        }
    });
}

/// `DS_FE2_TAKE_ALL=1` (diagnosis): every take invalidates its whole range,
/// so its pages, GPU-written ones included, are uploaded into the mirror
/// again in every epoch they are checked in.
fn take_all() -> bool {
    static W: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *W.get_or_init(|| std::env::var("DS_FE2_TAKE_ALL").ok().as_deref() == Some("1"))
}

/// Uploads the pages of [a, b) M lacks (not `mine`) and marks its valid
/// pages clean in epoch `ep` (after its take).
fn fix_run(a: u64, b: u64, ep: u32, mine: &dyn Fn(u64) -> bool, upload: &mut dyn FnMut(u64, u64) -> bool) -> Result<(), &'static str> {
    let (_, failed) = {
        let _k = UpKind::new(4);
        upload_lacking(a, b, mine, upload)
    };
    // Valid pages are clean in this epoch (uploaded ones become valid when
    // the caller's batch is queued: `uploaded`).
    let tc = std::time::Instant::now();
    set_clean(a, b, ep, false);
    super::hitch::note_since(super::hitch::Kind::Mirror, tc, || format!("clean marks of {a:#x}..{b:#x}"));
    if failed {
        return Err("an upload into the mirror failed");
    }
    Ok(())
}

/// Diagnosis (`exec`'s import audit, `DS_IMPORT_AUDIT`): up to `n` pages M
/// holds as current that are armed now (a CPU write faults, which ends the
/// arming) and were never written by GPU work: such a page reads in M as in
/// guest memory. `seed` varies the sample.
pub fn audit_sample(n: usize, seed: u64) -> Vec<u64> {
    let mut out = Vec::with_capacity(n);
    let mut h = seed | 1;
    let mut next = || {
        h ^= h << 13;
        h ^= h >> 7;
        h ^= h << 17;
        h
    };
    for _ in 0..n * 256 {
        if out.len() >= n {
            break;
        }
        let r = next();
        let ci = (r % CHUNKS as u64) as usize;
        let p = TABLE[ci].load(Ordering::Acquire);
        if p.is_null() {
            continue;
        }
        let c = unsafe { &*p };
        let w = ((r >> 40) % WORDS as u64) as usize;
        let wa = MIRROR_BASE + ((ci as u64) << CHUNK_SHIFT) + w as u64 * 64 * PAGE;
        let bits = audit_bits(c, w, wa);
        if bits == 0 {
            continue;
        }
        let mut b = bits;
        for _ in 0..next() % bits.count_ones() as u64 {
            b &= b - 1;
        }
        out.push(wa + b.trailing_zeros() as u64 * PAGE);
    }
    out
}

fn audit_bits(c: &Chunk, w: usize, wa: u64) -> u64 {
    c.valid[w].load(Ordering::Acquire) & c.armed[w].load(Ordering::Acquire) & crate::hosted::armed_word(wa) & !c.gpu[w].load(Ordering::Acquire)
}

/// Whether page `p` (`audit_sample`) is still current in M, armed and never
/// GPU-written.
pub fn audit_still(p: u64) -> bool {
    let Some(c) = chunk(p, false) else { return false };
    audit_bits(c, idx(p) / 64, p & !(64 * PAGE - 1)) & (1u64 << (idx(p) % 64)) != 0
}

/// Uploads the pages of [a, b) M lacks (not `mine`), as runs of consecutive
/// ones (word by word: only the invalid pages are looked at one by one):
/// (pages, whether an upload failed; the rest of the range is still made
/// current).
fn upload_lacking(a: u64, b: u64, mine: &dyn Fn(u64) -> bool, upload: &mut dyn FnMut(u64, u64) -> bool) -> (u64, bool) {
    let tu = std::time::Instant::now();
    let mut failed = false;
    let mut pages = 0u64;
    let mut run: Option<(u64, u64)> = None;
    let mut flush = |run: &mut Option<(u64, u64)>, failed: &mut bool, pages: &mut u64| {
        if let Some((x, y)) = run.take() {
            if !upload(x, y) {
                STATS[S_UPLOAD_FAILED].fetch_add(1, Ordering::Relaxed);
                *failed = true;
            } else {
                STATS[S_UPLOADS].fetch_add(1, Ordering::Relaxed);
                STATS[S_UPLOAD_PAGES].fetch_add((y - x) / PAGE, Ordering::Relaxed);
                *pages += (y - x) / PAGE;
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
                flush(&mut run, &mut failed, &mut pages);
                continue;
            }
            match run.as_mut() {
                Some(l) if l.1 == p => l.1 = p + PAGE,
                _ => {
                    flush(&mut run, &mut failed, &mut pages);
                    run = Some((p, p + PAGE));
                }
            }
        }
        x = e;
    }
    flush(&mut run, &mut failed, &mut pages);
    // (Long parts are hitch causes: `gpu::hitch`.)
    super::hitch::note_since(super::hitch::Kind::Mirror, tu, || format!("uploads of {a:#x}..{b:#x} into the mirror"));
    (pages, failed)
}

/// Our takes in flight, as `check_run` makes them: (taken range, the epoch
/// its pages are stamped with, id). The chunks of a parallel run start
/// together, and each one's first command checked the same GPU-written
/// ranges at the run's view: every worker took them, one after another
/// (~10 commands a flip at ~1.3 ms each on the workers at the vista,
/// 2026-10-28). A check of a run inside one of them, for a requirement it
/// meets, waits for it and uses what it found (`follow_take`, `followed`).
static TAKING: Mutex<Vec<(u64, u64, u32, u64)>> = Mutex::new(Vec::new());
static TAKING_IDS: AtomicU64 = AtomicU64::new(1);
/// Bumped as each of them ends (the waiters watch it).
static TAKING_ENDED: AtomicU64 = AtomicU64::new(0);
/// [runs that followed another's take without one of their own, pages they
/// uploaded, runs that followed and still needed a take].
static FOLLOW: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// `DS_FE2_FOLLOW=0` (A/B): every check takes for itself, as before
/// 2026-10-28 evening.
fn follow_on() -> bool {
    static W: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *W.get_or_init(|| std::env::var("DS_FE2_FOLLOW").ok().as_deref() != Some("0"))
}

/// Takes in flight from here until dropped.
struct Taking(u64);

impl Taking {
    fn new(ta: u64, tb: u64, ep: u32) -> Taking {
        let id = TAKING_IDS.fetch_add(1, Ordering::Relaxed);
        TAKING.lock().push((ta, tb, ep, id));
        Taking(id)
    }
}

impl Drop for Taking {
    fn drop(&mut self) {
        TAKING.lock().retain(|e| e.3 != self.0);
        TAKING_ENDED.fetch_add(1, Ordering::AcqRel);
    }
}

/// Whether another thread's take covering [a, b) whose stamps meet `want`
/// was in flight: waited for until it ended (its clean stamps set, the
/// pages it found written invalid).
fn follow_take(a: u64, b: u64, want: u32) -> bool {
    if !follow_on() {
        return false;
    }
    let id = {
        let g = TAKING.lock();
        match g.iter().find(|e| e.0 <= a && b <= e.1 && e.2 >= want) {
            Some(e) => e.3,
            None => return false,
        }
    };
    let mut k = 0u32;
    loop {
        // (Read before the list: an end after it shows as a change.)
        let seen = TAKING_ENDED.load(Ordering::Acquire);
        if !TAKING.lock().iter().any(|e| e.3 == id) {
            return true;
        }
        while TAKING_ENDED.load(Ordering::Acquire) == seen {
            k += 1;
            if k < 64 {
                std::hint::spin_loop();
            } else {
                std::thread::yield_now();
            }
        }
    }
}

/// After another's take over [a, b) (`follow_take`): Ok(true) when nothing
/// needs a take of our own (pages valid and clean for `want`; pages not
/// valid, which that take found written or the mirror never held, uploaded
/// here as after a take: they become valid and clean when the caller's
/// batch is queued, `uploaded`); Ok(false) when a valid page is not clean
/// for `want`.
fn followed(a: u64, b: u64, want: u32, mine: &dyn Fn(u64) -> bool, upload: &mut dyn FnMut(u64, u64) -> bool) -> Result<bool, &'static str> {
    let mut x = a;
    while x < b {
        let e = word_end(x, b);
        let c = chunk(x, true).expect("inside the span");
        let need = word_mask(x, e) & !word_ok(c, x, e, want);
        if need & c.valid[idx(x) / 64].load(Ordering::Acquire) != 0 {
            FOLLOW[2].fetch_add(1, Ordering::Relaxed);
            return Ok(false);
        }
        x = e;
    }
    let (pages, failed) = upload_lacking(a, b, mine, upload);
    FOLLOW[0].fetch_add(1, Ordering::Relaxed);
    FOLLOW[1].fetch_add(pages, Ordering::Relaxed);
    if failed {
        return Err("an upload into the mirror failed");
    }
    Ok(true)
}

/// The runs of pages of [lo, hi) (page-aligned) each found written within
/// its hot window (`hot_run`'s): a stretch past a label wait uploads them
/// as they are when it runs, whatever its checks found, since the CPU may
/// write them again before its label (`Exec::fe2_place_inner`; the
/// title's per-frame blocks, rewritten after the graphics queue's
/// submission and before the compute queue's whose label it waits on).
pub fn hot_pages(lo: u64, hi: u64, out: &mut Vec<(u64, u64)>) {
    let frame = super::FRAMES.load(Ordering::Relaxed);
    let quiet = hot_quiet();
    let mut start: Option<u64> = None;
    let mut x = lo;
    while x < hi {
        let hot = chunk(x, false).is_some_and(|c| {
            let i = idx(x);
            let wrote = c.wrote[i].load(Ordering::Relaxed) as u64;
            let window = if quiet { HOT_FLIPS.max(ARM_QUIET << c.level[i].load(Ordering::Relaxed)) } else { HOT_FLIPS };
            wrote != 0 && frame.saturating_sub(wrote - 1) < window
        });
        match (hot, start) {
            (true, None) => start = Some(x),
            (false, Some(s)) => {
                out.push((s, x));
                start = None;
            }
            _ => {}
        }
        x += PAGE;
    }
    if let Some(s) = start {
        out.push((s, hi));
    }
}

/// Whether [a, b) is at most HOT_PAGES pages, every one found written
/// within HOT_FLIPS flips.
fn hot_run(a: u64, b: u64, frame: u64) -> bool {
    if (b - a) / PAGE > HOT_PAGES {
        return false;
    }
    let quiet = hot_quiet();
    let mut x = a;
    while x < b {
        let Some(c) = chunk(x, false) else { return false };
        let i = idx(x);
        let wrote = c.wrote[i].load(Ordering::Relaxed) as u64;
        let window = if quiet { HOT_FLIPS.max(ARM_QUIET << c.level[i].load(Ordering::Relaxed)) } else { HOT_FLIPS };
        if wrote == 0 || frame.saturating_sub(wrote - 1) >= window {
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
            c.wmax[w].fetch_max(ep, Ordering::AcqRel);
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

/// Pages of [a, b) we armed whose host bit is gone (a write fault, its
/// word's, a disarm or a mapping change): no longer ours; the next arming
/// waits longer, its quiet time from now (the fault watcher may not have
/// seen the fault yet).
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
                let f = super::FRAMES.load(Ordering::Relaxed) as u32 + 1;
                let mut r = gone;
                while r != 0 {
                    let k = r.trailing_zeros() as usize;
                    r &= r - 1;
                    let l = &c.level[base + k];
                    let _ = l.fetch_update(Ordering::AcqRel, Ordering::Acquire, |v| Some((v + 1).min(ARM_MAX_LEVEL)));
                    c.wrote[base + k].store(f, Ordering::Release);
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
    hold_begin(base, len, "a file read");
    invalidate(base, len);
}

/// That write ended: its pages hold new contents.
pub fn host_write_end(base: u64, len: u64) {
    if !on() || len == 0 {
        return;
    }
    hold_end(base, len);
    invalidate(base, len);
}

/// Guest [base, base+len) armed by none until `hold_end` (`HOST_WRITES`).
fn hold_begin(base: u64, len: u64, who: &str) {
    // (An arming that checked before this is done by now, and the caller's
    // disarm after this undoes it: `arm_part`.)
    let t = std::time::Instant::now();
    let _lock = crate::hosted::arm_lock();
    super::hitch::note_since(super::hitch::Kind::Mirror, t, || format!("{who} into {base:#x}+{len:#x} waiting for an arming"));
    HOST_WRITES.lock().push((base, base.saturating_add(len)));
    HOST_WRITES_N.fetch_add(1, Ordering::AcqRel);
}

fn hold_end(base: u64, len: u64) {
    let mut g = HOST_WRITES.lock();
    if let Some(i) = g.iter().position(|&(a, e)| a == base && e == base.saturating_add(len)) {
        g.swap_remove(i);
        HOST_WRITES_N.fetch_sub(1, Ordering::AcqRel);
    }
}

/// Guest [base, base+len) held unarmed while this lives (`hold_begin`).
pub struct Hold(Option<(u64, u64)>);

impl Drop for Hold {
    fn drop(&mut self) {
        if let Some((base, len)) = self.0 {
            hold_end(base, len);
        }
    }
}

/// An import of guest [base, base+len) begins (`exec::import_span`, before
/// its disarm and the driver's call): its pages are armed by none until it
/// is listed, young (the import list is no longer held while arming).
pub fn hold(base: u64, len: u64) -> Hold {
    if !on() || len == 0 {
        return Hold(None);
    }
    hold_begin(base, len, "an import");
    Hold(Some((base, len)))
}

fn host_writing(a: u64, b: u64) -> bool {
    HOST_WRITES_N.load(Ordering::Acquire) != 0 && HOST_WRITES.lock().iter().any(|&(x, e)| x < b && a < e)
}

/// Arms the quiet pages of [a, b) not armed now (nor written by GPU work,
/// unless `DS_FE2_ARM_GPU=1`) outside held chunks, by runs inside settled
/// import windows (`armable`), plain read-write memory only.
fn try_arm(a: u64, b: u64, frame: u64, armable: &Armable) {
    let arm_gpu = arm_gpu();
    // A word at a time, the pages not armed counted here and added once
    // (until 2026-10-28 page by page, each such page a shared atomic add:
    // ~170,000 pages of held chunks a flip at the gate). A page's reason is
    // the first of: armed already, its chunk held, GPU-written, its word
    // found clean in too few epochs, its quiet time.
    let mut not = [0u64; 8];
    // The run of candidates being gathered (armed when a page that is not
    // one ends it).
    let mut run: Option<(u64, u64)> = None;
    let mut x = a;
    while x < b {
        let e = word_end(x, b);
        let Some(c) = chunk(x, false) else {
            not[R_QUIET] += (e - x) / PAGE;
            if let Some((p, q)) = run.take() {
                arm_run(p, q, frame, armable);
            }
            x = e;
            continue;
        };
        let w = idx(x) / 64;
        let mask = word_mask(x, e);
        let armed = c.armed[w].load(Ordering::Acquire) & mask;
        not[R_ARMED] += armed.count_ones() as u64;
        let mut cand = mask & !armed;
        if cand != 0 && frame < c.hold_until.load(Ordering::Acquire) {
            not[R_HELD] += cand.count_ones() as u64;
            cand = 0;
        }
        if cand != 0 && !arm_gpu && !hold_census() && !crate::mem::view_window(x) {
            let g = c.gpu[w].load(Ordering::Acquire) & cand;
            not[R_GPU] += g.count_ones() as u64;
            cand &= !g;
        }
        if cand != 0 && c.uses[w].load(Ordering::Relaxed) < ARM_USES {
            not[R_USES] += cand.count_ones() as u64;
            cand = 0;
        }
        let base = w * 64;
        let mut r = cand;
        while r != 0 {
            let k = r.trailing_zeros() as usize;
            r &= r - 1;
            let wrote = c.wrote[base + k].load(Ordering::Acquire) as u64;
            let need = ARM_QUIET << c.level[base + k].load(Ordering::Acquire);
            if wrote == 0 || frame.saturating_sub(wrote - 1) < need {
                not[R_QUIET] += 1;
                cand &= !(1u64 << k);
            }
        }
        // The word's runs of candidates (a run goes on across words).
        let wa = x & !(64 * PAGE - 1);
        let (mut pos, end) = ((idx(x) % 64) as u32, (idx(x) % 64) as u32 + ((e - x) / PAGE) as u32);
        while pos < end {
            let rest = cand >> pos;
            if rest & 1 != 0 {
                let ones = (!rest).trailing_zeros().min(end - pos);
                let (p, q) = (wa + pos as u64 * PAGE, wa + (pos + ones) as u64 * PAGE);
                match run.as_mut() {
                    Some(l) if l.1 == p => l.1 = q,
                    _ => {
                        if let Some((s, t)) = run.replace((p, q)) {
                            arm_run(s, t, frame, armable);
                        }
                    }
                }
                pos += ones;
            } else {
                if let Some((s, t)) = run.take() {
                    arm_run(s, t, frame, armable);
                }
                pos += if rest == 0 { end - pos } else { rest.trailing_zeros().min(end - pos) };
            }
        }
        x = e;
    }
    if let Some((p, q)) = run.take() {
        arm_run(p, q, frame, armable);
    }
    for (k, &n) in not.iter().enumerate() {
        if n != 0 {
            NOT_ARMED[k].fetch_add(n, Ordering::Relaxed);
        }
    }
}

/// Pages checked by takes that were not armed, by reason: armed already
/// (by us, its host bit set), written by GPU work (`DS_FE2_ARM_GPU`),
/// written within their quiet time, found clean in fewer than ARM_USES
/// epochs, outside settled import windows, not plain read-write (or a file
/// read or an import in progress), in a chunk held after a burst of faults,
/// the import list busy.
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
static FAULTS: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];
const F_SPAN: usize = 0;
const F_ENDED: usize = 1;
const F_BURSTS: usize = 2;
const F_DISARMED: usize = 3;
const F_OUTSIDE: usize = 4;
const F_LOST: usize = 5;
/// Pages of ours the host made read-write with a faulted page's word
/// (`word_ended`).
const F_WORD_DISARMED: usize = 6;

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
    // (The code of each chunk's first fault in the window: the burst log.)
    let mut first_rip = vec![0u64; CHUNKS];
    let mut touched: Vec<usize> = Vec::new();
    let mut total = 0u32;
    let mut window = std::time::Instant::now();
    let mut logged = 0u32;
    // `DS_FE2_FAULT_LOG=1` (diagnosis): every fault as "flip page code"
    // into the run's fe2-faults.txt.
    let mut flog = (std::env::var("DS_FE2_FAULT_LOG").ok().as_deref() == Some("1"))
        .then(|| crate::run_dir().and_then(|d| std::fs::File::create(d.join("fe2-faults.txt")).ok()))
        .flatten()
        .map(std::io::BufWriter::new);
    let mut names: std::collections::HashMap<u64, String> = std::collections::HashMap::new();
    // (This read's words with an arming a fault ended, and the level.)
    let mut hit: Vec<(u64, u8)> = Vec::new();
    let mut reconciled = std::time::Instant::now();
    loop {
        std::thread::sleep(std::time::Duration::from_millis(1));
        if HELD_MAP.get().is_some() {
            hold_tick(&mut reconciled);
        }
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
        hit.clear();
        if let Some(f) = flog.as_mut() {
            use std::io::Write;
            for &(page, _, rip) in &got {
                let name = names.entry(rip).or_insert_with(|| describe_code(rip));
                let _ = writeln!(f, "{frame} {page:#x} {name}");
            }
            let _ = f.flush();
        }
        for &(page, write, rip) in &got {
            if write && wb_buckets() {
                note_wb_pages(&WB_CPU_PAGES, page, PAGE);
            }
            note_wb_lazy_fault(page, write, rip);
            if held_mark(page, false) {
                note_hold_fault(page, write, rip);
            }
            // (Pages of view windows: the residency tracks their writes, and
            // only the page written turns volatile.)
            if write && crate::mem::view_window(page) {
                super::resident::host_fault(page);
            }
            let Some(c) = chunk(page, false) else {
                FAULTS[F_OUTSIDE].fetch_add(1, Ordering::Relaxed);
                continue;
            };
            FAULTS[F_SPAN].fetch_add(1, Ordering::Relaxed);
            let (i, w) = (idx(page), idx(page) / 64);
            let bit = 1u64 << (i % 64);
            // (A take's `note_arming_ended` may have seen it first.)
            if c.armed[w].fetch_and(!bit, Ordering::AcqRel) & bit != 0 {
                let v = c.level[i].fetch_update(Ordering::AcqRel, Ordering::Acquire, |v| Some((v + 1).min(ARM_MAX_LEVEL))).unwrap_or(0);
                c.cgen.fetch_add(1, Ordering::AcqRel);
                FAULTS[F_ENDED].fetch_add(1, Ordering::Relaxed);
                STATS[S_ARM_ENDED].fetch_add(1, Ordering::Relaxed);
                let wa = page & !(64 * PAGE - 1);
                let lv = (v + 1).min(ARM_MAX_LEVEL);
                match hit.iter_mut().find(|h| h.0 == wa) {
                    Some(h) => h.1 = h.1.max(lv),
                    None => hit.push((wa, lv)),
                }
            }
            c.wrote[i].store(frame as u32 + 1, Ordering::Release);
            c.uses[w].store(0, Ordering::Relaxed);
            let k = ((page - MIRROR_BASE) >> CHUNK_SHIFT) as usize;
            if counts[k] == 0 {
                touched.push(k);
                first_rip[k] = rip;
            }
            counts[k] += 1;
            total += 1;
            if logged < 40 {
                logged += 1;
                crate::info!("fe2: write fault on armed page {page:#x} at {}", describe_code(rip));
            }
        }
        for &(wa, lv) in &hit {
            let n = word_ended(wa, lv, frame);
            FAULTS[F_WORD_DISARMED].fetch_add(n, Ordering::Relaxed);
        }
        for &k in &touched {
            if !fired[k] && (counts[k] >= CHUNK_BURST || total >= ALL_BURST) {
                fired[k] = true;
                burst(k, counts[k], total, frame, first_rip[k]);
            }
        }
    }
}

/// A fault ended an arming in the 64-page word at `wa`, and the host's
/// answer made the word's other armed pages read-write too
/// (`hosted::set_word_mode`): those of ours are no longer armed, their
/// quiet time restarts and their level becomes at least `level` (the
/// faulted page's). Titles write runs of pages (at the gate a word's pages
/// faulted one after another within the millisecond, each page written once
/// in ~200 flips), and the backoff learnt page by page. The pages.
fn word_ended(wa: u64, level: u8, frame: u64) -> u64 {
    let Some(c) = chunk(wa, false) else { return 0 };
    let w = idx(wa) / 64;
    let gone = c.armed[w].load(Ordering::Acquire) & !crate::hosted::armed_word(wa);
    if gone == 0 {
        return 0;
    }
    c.armed[w].fetch_and(!gone, Ordering::AcqRel);
    let base = w * 64;
    let mut r = gone;
    while r != 0 {
        let k = r.trailing_zeros() as usize;
        r &= r - 1;
        c.wrote[base + k].store(frame as u32 + 1, Ordering::Release);
        c.level[base + k].fetch_max(level, Ordering::AcqRel);
    }
    c.cgen.fetch_add(1, Ordering::AcqRel);
    gone.count_ones() as u64
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
fn burst(k: usize, n: u32, total: u32, frame: u64, rip: u64) {
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
    // (The first 40, then every 16th.)
    if b < 40 || b % 16 == 0 {
        crate::info!("fe2: burst {b}: {n} faults within {BURST_MS} ms in {base:#x}+64M ({total} in all; the first at {}): {pages} armed pages there disarmed, arming there held {hold} flips", describe_code(rip));
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

/// Arms [x, y) (inside settled import windows, asked again under the arm
/// lock: an import beginning meanwhile waits for it, then disarms it).
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
    // (Still inside settled imports: the caller looked without the arm lock,
    // and an import listed since is young.)
    if super::exec::fe2_settled_over(x, y) != Some(true) {
        NOT_ARMED[R_WINDOW].fetch_add((y - x) / PAGE, Ordering::Relaxed);
        return;
    }
    if held(x, y, frame) {
        NOT_ARMED[R_HELD].fetch_add((y - x) / PAGE, Ordering::Relaxed);
        return;
    }
    if HELD_MAP.get().is_some() {
        hold_part(x, y);
        return;
    }
    STATS[S_ARM_RUNS].fetch_add(1, Ordering::Relaxed);
    let t = std::time::Instant::now();
    let r = crate::hosted::arm(x, y - x);
    super::hitch::note_since(super::hitch::Kind::Mirror, t, || format!("arming {x:#x}+{:#x}", y - x));
    if r.is_err() {
        STATS[S_ARM_FAILED].fetch_add(1, Ordering::Relaxed);
        return;
    }
    let mut p = x;
    while p < y {
        let e = word_end(p, y);
        if let Some(c) = chunk(p, false) {
            // (Not taken since this arming yet: `take_run`.)
            c.varmed[idx(p) / 64].fetch_and(!word_mask(p, e), Ordering::AcqRel);
            c.armed[idx(p) / 64].fetch_or(word_mask(p, e), Ordering::AcqRel);
        }
        p = e;
    }
    STATS[S_ARMED].fetch_add((y - x) / PAGE, Ordering::Relaxed);
}

/// `arm_part` under the hold census (its checks passed, the locks held):
/// the GPU-written pages of [x, y) held (`hosted::hold`, their bits in the
/// census's map), the others armed against writes, run by run.
fn hold_part(x: u64, y: u64) {
    // (0: not GPU-written, armed; 1: GPU-written and written back lately,
    // held; 2: GPU-written, not written back lately, left alone.)
    let frame = super::FRAMES.load(Ordering::Relaxed);
    let quiet = hold_quiet();
    let kind = |p: u64| -> u8 {
        let Some(c) = chunk(p, false) else { return 0 };
        if c.gpu[idx(p) / 64].load(Ordering::Acquire) >> (idx(p) % 64) & 1 == 0 {
            0
        } else if !wb_recent(p) || never_hold(p) {
            2
        } else if quiet > 0 && frame.saturating_sub((c.wrote[idx(p)].load(Ordering::Acquire) as u64).saturating_sub(1)) < quiet {
            3
        } else {
            1
        }
    };
    let mut p = x;
    while p < y {
        let k = kind(p);
        let mut q = p + PAGE;
        while q < y && kind(q) == k {
            q += PAGE;
        }
        if k == 2 || k == 3 {
            if k == 2 {
                HOLD_STALE.fetch_add((q - p) / PAGE, Ordering::Relaxed);
            } else {
                HOLD_YOUNG.fetch_add((q - p) / PAGE, Ordering::Relaxed);
            }
            p = q;
            continue;
        }
        let g = k == 1;
        STATS[S_ARM_RUNS].fetch_add(1, Ordering::Relaxed);
        let t = std::time::Instant::now();
        let r = if g { crate::hosted::hold(p, q - p) } else { crate::hosted::arm(p, q - p) };
        super::hitch::note_since(super::hitch::Kind::Mirror, t, || format!("{} {p:#x}+{:#x}", if g { "holding" } else { "arming" }, q - p));
        if r.is_err() {
            STATS[S_ARM_FAILED].fetch_add(1, Ordering::Relaxed);
            if g {
                HOLD[H_FAILED].fetch_add(1, Ordering::Relaxed);
            }
            p = q;
            continue;
        }
        let mut a = p;
        while a < q {
            let e = word_end(a, q);
            if let Some(c) = chunk(a, false) {
                c.varmed[idx(a) / 64].fetch_and(!word_mask(a, e), Ordering::AcqRel);
                c.armed[idx(a) / 64].fetch_or(word_mask(a, e), Ordering::AcqRel);
            }
            a = e;
        }
        if g {
            let mut n = 0u64;
            for a in (p..q).step_by(PAGE as usize) {
                if !held_mark(a, true) {
                    n += 1;
                }
            }
            HOLD[H_PAGES].fetch_add(n, Ordering::Relaxed);
            HOLD[H_RUNS].fetch_add(1, Ordering::Relaxed);
            HOLD[H_HELD_NOW].fetch_add(n, Ordering::Relaxed);
        }
        STATS[S_ARMED].fetch_add((q - p) / PAGE, Ordering::Relaxed);
        p = q;
    }
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
        let bit = 1u64 << (i % 64);
        // (The chunk's generation only for a page that was valid: ranges
        // found fine cover valid pages only, and every thread's range cache
        // reads it; an upload in flight still sees the word's generation.)
        if c.valid[w].fetch_and(!bit, Ordering::AcqRel) & bit != 0 {
            c.cgen.fetch_add(1, Ordering::AcqRel);
        }
        c.wgen[w].fetch_add(1, Ordering::AcqRel);
        c.wrote[i].store(f, Ordering::Release);
        c.pgen[i].fetch_add(1, Ordering::AcqRel);
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
            let mask = word_mask(x, e);
            // (The chunk's generation only where a page was valid, as
            // `cpu_written`.)
            if c.valid[w].fetch_and(!mask, Ordering::AcqRel) & mask != 0 {
                c.cgen.fetch_add(1, Ordering::AcqRel);
            }
            c.wgen[w].fetch_add(1, Ordering::AcqRel);
            for k in idx(x)..idx(x) + ((e - x) / PAGE) as usize {
                c.wrote[k].store(f, Ordering::Release);
                c.pgen[k].fetch_add(1, Ordering::AcqRel);
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
/// and the counters never change), the counters (host visible), whether
/// its kernel empties the touched-page list itself and the counters it
/// adds to in video memory (copied into the host-visible ones every
/// COUNTER_COPY_EVERY write-backs).
struct Scatter {
    pipeline: vk::Pipeline,
    /// The touched-page list made from the touched bits
    /// (`rdna::kernels::mirror_compact`, `tune touch_bits`).
    compact: vk::Pipeline,
    layout: vk::PipelineLayout,
    /// Its descriptor sets: the main queue's written-bit map, the compute
    /// family's queue's (null without it).
    set: vk::DescriptorSet,
    set_c: vk::DescriptorSet,
    counters: *mut u32,
    host: vk::Buffer,
    /// The counters in video memory (two host-memory atomics a page written
    /// back cost ~2.3 ms of GPU a vista frame), copied into `host` now and
    /// then.
    dev: vk::Buffer,
}

/// Write-backs between copies of the video-memory counters into the
/// host-visible ones (the status line's counts lag by at most that many).
const COUNTER_COPY_EVERY: u64 = 64;

static SCATTERS_RECORDED: AtomicU64 = AtomicU64::new(0);

unsafe impl Send for Scatter {}
unsafe impl Sync for Scatter {}

static SCATTER: std::sync::OnceLock<Scatter> = std::sync::OnceLock::new();

fn make_scatter(marks: vk::Buffer, marks_c: Option<vk::Buffer>) -> Result<Scatter, String> {
    let c = ctx();
    unsafe {
        let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..2)
            .map(|b| vk::DescriptorSetLayoutBinding::default().binding(b).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE))
            .collect();
        let set_layout = c.device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None).map_err(|e| format!("scatter set layout: {e}"))?;
        let layouts = [set_layout];
        // (32 bytes: the mirror's and the table's addresses; the rest
        // unused.)
        let ranges = [vk::PushConstantRange { stage_flags: vk::ShaderStageFlags::COMPUTE, offset: 0, size: 32 }];
        let layout = c.device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts).push_constant_ranges(&ranges), None).map_err(|e| format!("scatter layout: {e}"))?;
        // (The kernel empties the touched-page list itself; `DS_WB_BUCKETS`
        // also counts by bucket.)
        // (`DS_FE2_HOLD`: the hold census's kernel with its map in host
        // memory, read by address.)
        if hold_census() && HELD_MAP.get().is_none() {
            let bytes = (HELD_MAP_WORDS * 4) as u64;
            match c.host_buffer(bytes, vk::BufferUsageFlags::STORAGE_BUFFER) {
                Ok((buf, _mem, ptr)) => {
                    std::ptr::write_bytes(ptr, 0, bytes as usize);
                    let addr = c.buffer_address(buf);
                    if addr != 0 {
                        let _ = HELD_MAP.set(HeldMap { _buf: buf, words: ptr as usize, addr });
                    } else {
                        crate::warn!("fe2: hold census off: no buffer device addresses");
                    }
                }
                Err(e) => crate::warn!("fe2: hold census off: {e}"),
            }
        }
        let code = if wb_buckets() {
            rdna::kernels::mirror_scatter_counting()
        } else if wb_same_mode() == 1 {
            rdna::kernels::mirror_scatter_same_census()
        } else if HELD_MAP.get().is_some() {
            rdna::kernels::mirror_scatter_hold_census()
        } else {
            rdna::kernels::mirror_scatter_reset()
        };
        let make = |code: &[u32], key: u64| -> Result<vk::Pipeline, String> {
            let module = c.device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(code), None).map_err(|e| format!("scatter module: {e}"))?;
            let stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(c"main");
            let ci = vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout);
            let pipeline = super::pcache::with_cache('i', key, |cache| c.device.create_compute_pipelines(cache, &[ci], None)).map_err(|(_, e)| format!("scatter pipeline: {e}"));
            c.device.destroy_shader_module(module, None);
            Ok(pipeline?[0])
        };
        let pipeline = make(&code, if wb_buckets() { 27 } else if wb_same_mode() == 1 { 34 } else if HELD_MAP.get().is_some() { 28 } else { 20 })?;
        if wb_same_mode() == 2 && SAME_SKIP.get().is_none() {
            // (The tables first: zeroed, and the buffer of zeros.)
            let mut tables = Vec::new();
            for _ in [Some(marks), marks_c].into_iter().flatten() {
                let (b, _m, a) = c.device_buffer_addressed(HASH_TABLE_BYTES, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST)?;
                if a == 0 {
                    return Err("the hash tables need device addresses".into());
                }
                tables.push((b, a));
            }
            let (zeros, _zm) = c.device_buffer(HASH_ZERO_BYTES, vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST)?;
            let tb: Vec<vk::Buffer> = tables.iter().map(|t| t.0).collect();
            super::resident::one_off_now(|cmd| {
                for &t in &tb {
                    c.device.cmd_fill_buffer(cmd, t, 0, vk::WHOLE_SIZE, 0);
                }
                c.device.cmd_fill_buffer(cmd, zeros, 0, vk::WHOLE_SIZE, 0);
            })?;
            let _ = HASHES.set(HashTables { tables, zeros });
            let _ = SAME_SKIP.set(make(&rdna::kernels::mirror_scatter_same_skip(), 36)?);
        }
        let compact = make(&rdna::kernels::mirror_compact(), 24)?;
        let sizes = [vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 4 }];
        let pool = c.device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(2).pool_sizes(&sizes), None).map_err(|e| format!("scatter pool: {e}"))?;
        let bytes = 4 * rdna::kernels::SCATTER_COUNTERS as u64;
        let (cbuf, _cmem, cptr) = c.host_buffer(bytes, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST)?;
        std::ptr::write_bytes(cptr, 0, bytes as usize);
        let (dev, _dmem) = c.device_buffer(bytes, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST)?;
        super::resident::one_off_now(|cmd| c.device.cmd_fill_buffer(cmd, dev, 0, bytes, 0))?;
        // One set per map; the counters are shared (atomics).
        let make_set = |m: vk::Buffer| -> Result<vk::DescriptorSet, String> {
            let set = c.device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&layouts)).map_err(|e| format!("scatter set: {e}"))?[0];
            let infos = [vk::DescriptorBufferInfo { buffer: m, offset: 0, range: super::resident::MARK_BYTES }, vk::DescriptorBufferInfo { buffer: dev, offset: 0, range: bytes }];
            let writes: Vec<vk::WriteDescriptorSet> = infos
                .iter()
                .enumerate()
                .map(|(k, info)| vk::WriteDescriptorSet::default().dst_set(set).dst_binding(k as u32).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(std::slice::from_ref(info)))
                .collect();
            c.device.update_descriptor_sets(&writes, &[]);
            Ok(set)
        };
        let set = make_set(marks)?;
        let set_c = match marks_c {
            Some(m) => make_set(m)?,
            None => vk::DescriptorSet::null(),
        };
        Ok(Scatter { pipeline, compact, layout, set, set_c, counters: cptr as *mut u32, host: cbuf, dev })
    }
}

/// Some translation set touched bits alone (`tune touch_bits`): from then
/// on write-backs make the touched-page list from the bits first.
static TOUCH_BITS_EVER: AtomicBool = AtomicBool::new(false);

/// Write-backs that made the list from the touched bits: [write-backs,
/// words of touched bits scanned].
static COMPACTS: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

pub fn note_touch_bits() {
    if !TOUCH_BITS_EVER.load(Ordering::Relaxed) {
        TOUCH_BITS_EVER.store(true, Ordering::Relaxed);
        crate::info!("fe2: translations set touched bits alone from now on (`tune touch_bits`): write-backs make the touched-page list from them");
    }
}

/// `DS_WB_BUCKETS=1` (diagnosis): the write-back counts the dwords and
/// pages it writes back per 1 MiB of the span (`rdna::kernels::
/// mirror_scatter_counting`, the map's aux region), and status reports
/// class those buckets by the pages the CPU wrote (takes, write faults) and
/// the command processors read (`sync_for`) since the last report: how much
/// of the volume lies where neither touches it (pages a write-back could
/// leave out while their reads are detected).
pub fn wb_buckets() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_WB_BUCKETS").ok().as_deref() == Some("1"))
}

/// Whether the write-back compares pages with guest memory (`wb_same_mode`
/// 1 or 2).
pub fn wb_same() -> bool {
    wb_same_mode() != 0
}

/// The write-back's handling of pages it would store unchanged: 2 (the
/// default since 2026-11-05) leaves them out
/// (`rdna::kernels::mirror_scatter_same_skip`: a 64-bit hash of the page's
/// mirror copy against the one kept at its last store; FINDINGS.md 2026-11-05
/// small hours), the plain write-back's pipeline made too and the control
/// `wbs on|off` choosing (in-run A/Bs); `DS_WB_SAME=1` (diagnosis) counts
/// them by reading guest memory's copy (`mirror_scatter_same_census`);
/// `DS_WB_SAME=0`, and the diagnoses with write-back kernels of their own
/// (`DS_WB_BUCKETS=1`, `DS_FE2_HOLD=1`), store every page.
pub fn wb_same_mode() -> u8 {
    static M: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *M.get_or_init(|| {
        if wb_buckets() || std::env::var("DS_FE2_HOLD").ok().as_deref() == Some("1") {
            return 0;
        }
        std::env::var("DS_WB_SAME").ok().and_then(|v| v.parse().ok()).filter(|&v: &u8| v <= 2).unwrap_or(2)
    })
}

/// `DS_WB_SAME=2`: the skipping write-back's pipeline, and whether it is used
/// (`wbs on|off`).
static SAME_SKIP: std::sync::OnceLock<vk::Pipeline> = std::sync::OnceLock::new();
static SAME_SKIP_ON: AtomicBool = AtomicBool::new(true);

/// `DS_WB_SAME=2`: the pages' hashes at their last store, a table per
/// written-bit map (device-local, 8 bytes a page of the span: buffer,
/// address), and a buffer of zeros the uploads' clears copy from.
struct HashTables {
    tables: Vec<(vk::Buffer, u64)>,
    zeros: vk::Buffer,
}

static HASHES: std::sync::OnceLock<HashTables> = std::sync::OnceLock::new();
const HASH_TABLE_BYTES: u64 = (MIRROR_END - MIRROR_BASE) / PAGE * 8;
const HASH_ZERO_BYTES: u64 = 64 << 10;
/// Tables to clear before their family's next write-back (mapping changes,
/// `wbs on`): a bit per table.
static HASH_CLEAR: AtomicU32 = AtomicU32::new(0);

/// Guest [base, base+len)'s mapping changed: its pages' kept hashes dropped
/// before each family's next write-back (an upload may not precede GPU work
/// that only writes there).
pub fn hash_clear_range(base: u64, len: u64) {
    if HASHES.get().is_none() || len == 0 {
        return;
    }
    let lo = base.max(MIRROR_BASE) & !(PAGE - 1);
    let hi = base.saturating_add(len).min(MIRROR_END).next_multiple_of(PAGE);
    if lo < hi {
        let mut q = HASH_CLEAR_RANGES.lock();
        for l in q.iter_mut() {
            l.push((lo, hi));
        }
    }
}

/// Ranges whose hashes to clear, a list per table.
static HASH_CLEAR_RANGES: parking_lot::Mutex<[Vec<(u64, u64)>; 2]> = parking_lot::Mutex::new([Vec::new(), Vec::new()]);

/// Every page's kept hash dropped before the next write-backs (`wbs on`: the
/// tables were not kept meanwhile). (Guest memory changed under a page --
/// a CPU write, a mapping change -- reaches the mirror only by an upload,
/// which clears the page's hash: `hash_clear_uploads`.)
pub fn hash_clear_all() {
    if HASHES.get().is_some() {
        HASH_CLEAR.store(u32::MAX, Ordering::Release);
    }
}

/// Uploads into the mirror were recorded into `cmd` (`exec::record_copies`):
/// their pages' kept hashes cleared there too, after them.
pub fn hash_clear_uploads(cmd: vk::CommandBuffer, copies: &[(vk::Buffer, vk::Buffer, vk::BufferCopy)]) {
    let Some(h) = HASHES.get() else { return };
    let Some((mbuf, _)) = super::resident::mirror_buffer() else { return };
    // (The uploads' pages as disjoint runs: several uploads may cover a page.)
    let mut runs: Vec<(u64, u64)> = copies.iter().filter(|c| c.1 == mbuf).map(|c| (c.2.dst_offset / PAGE, (c.2.dst_offset + c.2.size).div_ceil(PAGE))).collect();
    runs.sort_unstable();
    let mut merged: Vec<(u64, u64)> = Vec::with_capacity(runs.len());
    for (a, b) in runs {
        match merged.last_mut() {
            Some(l) if a <= l.1 => l.1 = l.1.max(b),
            _ => merged.push((a, b)),
        }
    }
    let mut regions: Vec<vk::BufferCopy> = Vec::new();
    for (p0, p1) in merged {
        let mut p = p0;
        while p < p1 {
            let n = (p1 - p).min(HASH_ZERO_BYTES / 8);
            regions.push(vk::BufferCopy { src_offset: 0, dst_offset: p * 8, size: n * 8 });
            p += n;
        }
    }
    if regions.is_empty() {
        return;
    }
    let c = ctx();
    unsafe {
        // (Transfer stages only: the copy queue records these too.)
        let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
        c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[mb], &[], &[]);
        for &(t, _) in &h.tables {
            c.device.cmd_copy_buffer(cmd, h.zeros, t, &regions);
        }
    }
}

/// Control `wbs on|off` (`DS_WB_SAME=2`).
pub fn wbs_control(what: &str) -> Result<String, String> {
    if SAME_SKIP.get().is_none() {
        return Err("wbs: no skipping write-back (`DS_WB_SAME=2`)".into());
    }
    match what {
        "on" => {
            // (The tables were not kept while off.)
            hash_clear_all();
            SAME_SKIP_ON.store(true, Ordering::Release);
            Ok("write-backs leave out pages stored unchanged".into())
        }
        "off" => {
            SAME_SKIP_ON.store(false, Ordering::Release);
            Ok("write-backs store every page".into())
        }
        _ => Err("usage: wbs on|off".into()),
    }
}

/// Status line of `DS_WB_SAME` (since the last one).
fn wb_same_line() -> String {
    if !wb_same() || SCATTER.get().is_none() {
        return String::new();
    }
    static LAST: parking_lot::Mutex<(u64, u64, u64, u64, u64)> = parking_lot::Mutex::new((0, 0, 0, 0, 0));
    let frame = super::FRAMES.load(Ordering::Relaxed);
    let now = [WB64[0].load(Ordering::Relaxed), WB64[1].load(Ordering::Relaxed), WB64[2].load(Ordering::Relaxed), WB64[3].load(Ordering::Relaxed)];
    let mut l = LAST.lock();
    let flips = frame.saturating_sub(l.0).max(1) as f64;
    let d = [now[0].saturating_sub(l.1), now[1].saturating_sub(l.2), now[2].saturating_sub(l.3), now[3].saturating_sub(l.4)];
    *l = (frame, now[0], now[1], now[2], now[3]);
    format!(
        "    write-backs unchanged (`DS_WB_SAME={}`{}, over {flips:.0} flips): {:.0} of {:.0} pages a flip ({:.1}%) {} every marked dword as guest memory held it, {:.3} of {:.3} M dwords ({:.1}%)\n",
        wb_same_mode(),
        if wb_same_mode() == 2 { if SAME_SKIP_ON.load(Ordering::Relaxed) { ", on" } else { ", off" } } else { "" },
        d[3] as f64 / flips,
        d[0] as f64 / flips,
        100.0 * d[3] as f64 / d[0].max(1) as f64,
        if wb_same_mode() == 2 { "left out, holding" } else { "stored" },
        d[2] as f64 / 1e6 / flips,
        if wb_same_mode() == 2 { (d[1] + d[2]) as f64 / 1e6 / flips } else { d[1] as f64 / 1e6 / flips },
        if wb_same_mode() == 2 { 100.0 * d[2] as f64 / (d[1] + d[2]).max(1) as f64 } else { 100.0 * d[2] as f64 / d[1].max(1) as f64 }
    )
}

/// `DS_FE2_HOLD=1` (diagnosis, the hold census; FRONTEND2.md "GPU-held
/// pages"; hosted runs): the GPU-written pages the arming picks are held
/// through the host (no-access, their held bits set) instead of armed
/// read-only, the host answering their faults at once as for armed pages
/// (no thread waits; their words are written back as ever); the write-back
/// counts the dwords holding would leave out (`rdna::kernels::
/// mirror_scatter_hold_census`, a bitmap of held pages in host memory);
/// each fault on a held page is counted by origin and access, with the wait
/// an exact release would have had for GPU work writing the page then.
pub fn hold_census() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_FE2_HOLD").ok().as_deref() == Some("1") && !wb_buckets())
}

/// The hold census's bitmap: a bit per page of the span (bit i of word i /
/// 32), in host memory the write-back reads at `addr`.
struct HeldMap {
    _buf: vk::Buffer,
    words: usize,
    addr: u64,
}

static HELD_MAP: std::sync::OnceLock<HeldMap> = std::sync::OnceLock::new();
const HELD_MAP_WORDS: usize = ((MIRROR_END - MIRROR_BASE) / PAGE / 32) as usize;

/// Sets (or clears) page `page`'s bit in the hold census's map: whether it
/// was set.
fn held_mark(page: u64, on: bool) -> bool {
    let Some(m) = HELD_MAP.get() else { return false };
    if !(MIRROR_BASE..MIRROR_END).contains(&page) {
        return false;
    }
    let i = ((page - MIRROR_BASE) / PAGE) as usize;
    // SAFETY: HELD_MAP_WORDS host-memory words, mapped for the process's life.
    let w = unsafe { &*(m.words as *const AtomicU32).add(i / 32) };
    let bit = 1u32 << (i % 32);
    let old = if on { w.fetch_or(bit, Ordering::AcqRel) } else { w.fetch_and(!bit, Ordering::AcqRel) };
    old & bit != 0
}

/// The hold census's counts: pages held, runs, runs that failed; faults on
/// held pages [the title, ours, other code] x [read, write]; of them with
/// GPU work writing the page in flight, none in flight; waits resolved,
/// their total and longest (us); holds ended otherwise (disarms, mapping
/// changes, word answers: the reconcile).
static HOLD: [AtomicU64; 17] = [const { AtomicU64::new(0) }; 17];
/// (Pages uploaded into the mirror while held.)
const H_UPLOADED: usize = 16;
/// GPU-written pages the census did not hold: not written back lately.
static HOLD_STALE: AtomicU64 = AtomicU64::new(0);
/// Pages written back within the last two periods of `hold_thread` (a bit
/// per page of the span), and how many at its last look.
static WB_RECENT: std::sync::OnceLock<Box<[AtomicU32]>> = std::sync::OnceLock::new();
static WB_RECENT_PAGES: AtomicU64 = AtomicU64::new(0);

/// `DS_FE2_HOLD_ONCE=1` (the hold census): a page that faulted while held
/// is never held again (`NO_HOLD`, a bit per page of the span).
fn hold_once() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_FE2_HOLD_ONCE").ok().as_deref() == Some("1"))
}

static NO_HOLD: std::sync::OnceLock<Box<[AtomicU32]>> = std::sync::OnceLock::new();
static NO_HOLD_PAGES: AtomicU64 = AtomicU64::new(0);

fn no_hold() -> &'static [AtomicU32] {
    NO_HOLD.get_or_init(|| (0..HELD_MAP_WORDS).map(|_| AtomicU32::new(0)).collect())
}

/// `DS_FE2_HOLD_QUIET=<flips>` (the hold census): a GPU-written page is held
/// only after that many flips without a CPU write seen on it (0: the
/// arming's own quiet time).
fn hold_quiet() -> u64 {
    static Q: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *Q.get_or_init(|| std::env::var("DS_FE2_HOLD_QUIET").ok().and_then(|v| v.parse().ok()).unwrap_or(0))
}

/// Pages the census did not hold for want of its own quiet time.
static HOLD_YOUNG: AtomicU64 = AtomicU64::new(0);

/// Whether page `p` faulted while held before (`hold_once`).
fn never_hold(p: u64) -> bool {
    if !hold_once() || !(MIRROR_BASE..MIRROR_END).contains(&p) {
        return false;
    }
    let i = ((p - MIRROR_BASE) / PAGE) as usize;
    no_hold()[i / 32].load(Ordering::Relaxed) >> (i % 32) & 1 != 0
}

/// Whether page `p` was written back lately (`WB_RECENT`).
fn wb_recent(p: u64) -> bool {
    let Some(r) = WB_RECENT.get() else { return false };
    if !(MIRROR_BASE..MIRROR_END).contains(&p) {
        return false;
    }
    let i = ((p - MIRROR_BASE) / PAGE) as usize;
    r[i / 32].load(Ordering::Relaxed) >> (i % 32) & 1 != 0
}

/// The hold census's thread: every 500 ms the written-back bits of both maps'
/// aux regions read back and cleared (a write-back racing it may lose its
/// bit to the clear), `WB_RECENT` made of this period's and the last one's.
fn hold_thread() {
    let c = ctx();
    let words = HELD_MAP_WORDS as u64;
    let at = rdna::kernels::MARK_AUX_AT as u64 * 4;
    let recent = WB_RECENT.get_or_init(|| (0..HELD_MAP_WORDS).map(|_| AtomicU32::new(0)).collect());
    let Ok((hbuf, _hmem, hptr)) = c.host_buffer(2 * words * 4, vk::BufferUsageFlags::TRANSFER_DST) else {
        crate::warn!("fe2: hold census: no host buffer for the written-back bits");
        return;
    };
    let mut prev = vec![0u32; HELD_MAP_WORDS];
    loop {
        std::thread::sleep(std::time::Duration::from_millis(500));
        let maps: Vec<vk::Buffer> = [false, true].iter().filter_map(|&on_c| super::resident::marks_for(on_c)).collect();
        let r = super::resident::one_off_now(|cmd| unsafe {
            for (k, &m) in maps.iter().enumerate() {
                c.device.cmd_copy_buffer(cmd, m, hbuf, &[vk::BufferCopy { src_offset: at, dst_offset: k as u64 * words * 4, size: words * 4 }]);
            }
            let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_READ).dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
            c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[mb], &[], &[]);
            for &m in &maps {
                c.device.cmd_fill_buffer(cmd, m, at, words * 4, 0);
            }
        });
        if r.is_err() {
            continue;
        }
        let mut n = 0u64;
        for i in 0..HELD_MAP_WORDS {
            let mut cur = 0u32;
            for k in 0..maps.len() {
                cur |= unsafe { std::ptr::read_volatile((hptr as *const u32).add(k * HELD_MAP_WORDS + i)) };
            }
            let v = cur | prev[i];
            recent[i].store(v, Ordering::Relaxed);
            n += v.count_ones() as u64;
            prev[i] = cur;
        }
        WB_RECENT_PAGES.store(n, Ordering::Relaxed);
    }
}
const H_PAGES: usize = 0;
const H_RUNS: usize = 1;
const H_FAILED: usize = 2;
const H_FAULTS: usize = 3;
const H_INFLIGHT: usize = 9;
const H_NONE: usize = 10;
const H_WAITS: usize = 11;
const H_WAIT_US: usize = 12;
const H_WAIT_MAX: usize = 13;
const H_OTHER: usize = 14;
const H_HELD_NOW: usize = 15;
/// Faults on held pages by their page's quiet time at the fault (flips since
/// the last CPU write seen on it): < 64, < 128, ... < 8192, more.
static HOLD_AGES: [AtomicU64; 9] = [const { AtomicU64::new(0) }; 9];

/// Faults on held pages by (instruction, a write) since the last status
/// line.
static HOLD_RIPS: parking_lot::Mutex<Option<std::collections::HashMap<(u64, bool), u64>>> = parking_lot::Mutex::new(None);
/// The waits being timed: (the writer's sequence, the fault's time).
static HOLD_PENDING: parking_lot::Mutex<Vec<(u64, std::time::Instant)>> = parking_lot::Mutex::new(Vec::new());

/// A fault on a held page (`watch`): counted by origin and access, the
/// first 40 logged with their code; the GPU work writing the page then.
fn note_hold_fault(page: u64, write: bool, rip: u64) {
    let (b, e) = crate::exception::host_image();
    let origin = if (crate::mem::GUEST_BASE..crate::mem::GUEST_END).contains(&rip) {
        0
    } else if (b..e).contains(&rip) {
        1
    } else {
        2
    };
    let n = HOLD[H_FAULTS + 2 * origin + write as usize].fetch_add(1, Ordering::Relaxed);
    HOLD[H_HELD_NOW].fetch_sub(1, Ordering::Relaxed);
    if hold_once() && (MIRROR_BASE..MIRROR_END).contains(&page) {
        let i = ((page - MIRROR_BASE) / PAGE) as usize;
        if no_hold()[i / 32].fetch_or(1 << (i % 32), Ordering::Relaxed) >> (i % 32) & 1 == 0 {
            NO_HOLD_PAGES.fetch_add(1, Ordering::Relaxed);
        }
    }
    *HOLD_RIPS.lock().get_or_insert_with(Default::default).entry((rip, write)).or_insert(0) += 1;
    if let Some(c) = chunk(page, false) {
        let frame = super::FRAMES.load(Ordering::Relaxed);
        let age = frame.saturating_sub((c.wrote[idx(page)].load(Ordering::Acquire) as u64).saturating_sub(1));
        let k = ((64 - age.max(1).leading_zeros()) as usize).saturating_sub(6).min(8);
        HOLD_AGES[k].fetch_add(1, Ordering::Relaxed);
    }
    let writer = super::exec::inflight_writer(page, PAGE);
    match writer {
        Some(seq) => {
            HOLD[H_INFLIGHT].fetch_add(1, Ordering::Relaxed);
            HOLD_PENDING.lock().push((seq, std::time::Instant::now()));
        }
        None => {
            HOLD[H_NONE].fetch_add(1, Ordering::Relaxed);
        }
    }
    static LOGGED: AtomicU64 = AtomicU64::new(0);
    if LOGGED.fetch_add(1, Ordering::Relaxed) < 40 {
        crate::info!("fe2: hold census: a {} of held page {page:#x} by {} ({}; this kind's {} so far){}", if write { "write" } else { "read" }, ["the title", "ours", "other code"][origin], describe_code(rip), n + 1, if writer.is_some() { "; GPU work writing it in flight" } else { "" });
    }
}

/// The hold census's tick (`watch`, every millisecond): the waits whose
/// writer completed timed; every second, map bits of pages no longer held
/// (their holds ended without a fault of their own) cleared and counted.
fn hold_tick(last_reconcile: &mut std::time::Instant) {
    {
        let mut p = HOLD_PENDING.lock();
        if !p.is_empty() {
            let done = super::exec::done_now();
            p.retain(|&(seq, t)| {
                if seq > done {
                    return true;
                }
                let us = t.elapsed().as_micros() as u64;
                HOLD[H_WAITS].fetch_add(1, Ordering::Relaxed);
                HOLD[H_WAIT_US].fetch_add(us, Ordering::Relaxed);
                HOLD[H_WAIT_MAX].fetch_max(us, Ordering::Relaxed);
                false
            });
        }
    }
    if last_reconcile.elapsed().as_millis() < 1000 {
        return;
    }
    *last_reconcile = std::time::Instant::now();
    let Some(m) = HELD_MAP.get() else { return };
    // SAFETY: as `held_mark`.
    let words = unsafe { std::slice::from_raw_parts(m.words as *const AtomicU32, HELD_MAP_WORDS) };
    let mut ended = 0u64;
    for (i, w) in words.iter().enumerate() {
        let mut bits = w.load(Ordering::Acquire);
        while bits != 0 {
            let k = bits.trailing_zeros();
            bits &= bits - 1;
            let page = MIRROR_BASE + (i as u64 * 32 + k as u64) * PAGE;
            if !crate::hosted::held(page) && w.fetch_and(!(1u32 << k), Ordering::AcqRel) & (1u32 << k) != 0 {
                ended += 1;
            }
        }
    }
    if ended > 0 {
        HOLD[H_OTHER].fetch_add(ended, Ordering::Relaxed);
        HOLD[H_HELD_NOW].fetch_sub(ended, Ordering::Relaxed);
    }
}

/// Status line of the hold census (since the last one: per flip).
fn hold_line() -> String {
    if HELD_MAP.get().is_none() {
        return String::new();
    }
    static LAST: parking_lot::Mutex<(u64, [u64; 17], u64, u64, u64)> = parking_lot::Mutex::new((0, [0; 17], 0, 0, 0));
    let frame = super::FRAMES.load(Ordering::Relaxed);
    let now: [u64; 17] = std::array::from_fn(|k| HOLD[k].load(Ordering::Relaxed));
    let (dw, pg, all) = (WB64[2].load(Ordering::Relaxed), WB64[3].load(Ordering::Relaxed), WB64[1].load(Ordering::Relaxed));
    let mut l = LAST.lock();
    let flips = frame.saturating_sub(l.0).max(1) as f64;
    let d = |k: usize| now[k].saturating_sub(l.1[k]);
    let (ddw, dpg, dall) = (dw.saturating_sub(l.2), pg.saturating_sub(l.3), all.saturating_sub(l.4));
    let waits = d(H_WAITS);
    let line = format!(
        "    hold census (`DS_FE2_HOLD`, over {flips:.0} flips): pages held now {} ({} held in {} runs since, {} runs failed); held pages in the write-backs {:.3} M dwords and {:.0} pages a flip ({:.1}% of the dwords written back); faults on held pages {:.3} a flip: the title {} reads / {} writes, ours {} / {}, other code {} / {}; GPU work writing the page in flight at {} of them (until it completed {:.2} ms on average, {:.1} ms at most so far, over {waits}), none at {}; holds ended otherwise (disarms, mapping changes, word answers) {}; pages uploaded into the mirror while held {}; pages written back within the last second {}, GPU-written candidates not held (not written back lately, or faulted before: `DS_FE2_HOLD_ONCE`, {} pages) {}, not quiet for `DS_FE2_HOLD_QUIET` {}\n",
        now[H_HELD_NOW] as i64,
        d(H_PAGES),
        d(H_RUNS),
        d(H_FAILED),
        ddw as f64 / 1e6 / flips,
        dpg as f64 / flips,
        100.0 * ddw as f64 / dall.max(1) as f64,
        (H_FAULTS..H_FAULTS + 6).map(d).sum::<u64>() as f64 / flips,
        d(H_FAULTS),
        d(H_FAULTS + 1),
        d(H_FAULTS + 2),
        d(H_FAULTS + 3),
        d(H_FAULTS + 4),
        d(H_FAULTS + 5),
        d(H_INFLIGHT),
        d(H_WAIT_US) as f64 / 1000.0 / waits.max(1) as f64,
        now[H_WAIT_MAX] as f64 / 1000.0,
        d(H_NONE),
        d(H_OTHER),
        d(H_UPLOADED),
        WB_RECENT_PAGES.load(Ordering::Relaxed),
        NO_HOLD_PAGES.load(Ordering::Relaxed),
        HOLD_STALE.load(Ordering::Relaxed),
        HOLD_YOUNG.load(Ordering::Relaxed)
    );
    *l = (frame, now, dw, pg, all);
    let mut rips: Vec<((u64, bool), u64)> = HOLD_RIPS.lock().take().map(|m| m.into_iter().collect()).unwrap_or_default();
    rips.sort_by(|a, b| b.1.cmp(&a.1));
    if rips.is_empty() {
        return line;
    }
    let top: Vec<String> = rips.iter().take(8).map(|&((rip, w), n)| format!("{n} {} at {}", if w { "writes" } else { "reads" }, describe_code(rip))).collect();
    let ages: Vec<String> = HOLD_AGES.iter().map(|a| a.swap(0, Ordering::Relaxed).to_string()).collect();
    line + &format!("    hold census: faults by code: {} ({} places); by the page's quiet time (flips; < 64, < 128, ..., < 8192, more): {}\n", top.join(", "), rips.len(), ages.join(" "))
}

/// `DS_WB_BUCKETS`: pages of the span the CPU wrote, and those the command
/// processors read, since the last report (one bit a page).
static WB_CPU_PAGES: std::sync::OnceLock<Box<[AtomicU64]>> = std::sync::OnceLock::new();
static WB_READ_PAGES: std::sync::OnceLock<Box<[AtomicU64]>> = std::sync::OnceLock::new();
/// The flip of the last report.
static WB_FLIP: AtomicU64 = AtomicU64::new(0);
/// The buckets the last report found untouched (`wbb lazy`).
static WB_UNTOUCHED: parking_lot::Mutex<Vec<u32>> = parking_lot::Mutex::new(Vec::new());
/// `wbb lazy`'s buckets (a bit each) and the page runs it armed no-access.
static WB_LAZY_BITS: std::sync::OnceLock<Box<[AtomicU64]>> = std::sync::OnceLock::new();
static WB_LAZY_RUNS: parking_lot::Mutex<Vec<(u64, u64)>> = parking_lot::Mutex::new(Vec::new());
/// Faults the host answered on those pages: [title, ours, other] x [read,
/// write], then pages armed, runs failed.
static WB_LAZY_FAULTS: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];

fn wb_lazy_bits() -> &'static [AtomicU64] {
    WB_LAZY_BITS.get_or_init(|| (0..rdna::kernels::WB_BUCKETS as usize / 64).map(|_| AtomicU64::new(0)).collect())
}

/// A fault the host answered (`watch`): counted when its page lies in a lazy
/// bucket (`wbb lazy`), the first 30 logged with their code.
fn note_wb_lazy_fault(page: u64, write: bool, rip: u64) {
    if !wb_buckets() || !(MIRROR_BASE..MIRROR_END).contains(&page) {
        return;
    }
    let k = ((page - MIRROR_BASE) >> (rdna::kernels::WB_BUCKET_SHIFT as u64 + 12)) as usize;
    if wb_lazy_bits()[k / 64].load(Ordering::Relaxed) & (1 << (k % 64)) == 0 {
        return;
    }
    let (b, e) = crate::exception::host_image();
    let origin = if (crate::mem::GUEST_BASE..crate::mem::GUEST_END).contains(&rip) {
        0
    } else if (b..e).contains(&rip) {
        1
    } else {
        2
    };
    let n = WB_LAZY_FAULTS[2 * origin + write as usize].fetch_add(1, Ordering::Relaxed);
    static LOGGED: AtomicU64 = AtomicU64::new(0);
    if LOGGED.fetch_add(1, Ordering::Relaxed) < 30 {
        crate::info!("fe2: wbb lazy: a {} of {page:#x} by {} ({}; this kind's {} so far)", if write { "write" } else { "read" }, ["the title", "ours", "other code"][origin], describe_code(rip), n + 1);
    }
}

/// Control `wbb lazy` (diagnosis, wrong results where the title or a
/// command processor reads such pages): the counting write-back leaves out
/// the stores of the pages of the buckets the last report found untouched
/// (their marks cleared, their dwords counted); `wbb eager`: none again.
pub fn wbb_control(what: &str) -> Result<String, String> {
    if !wb_buckets() {
        return Err("wbb needs DS_WB_BUCKETS=1".into());
    }
    let n = (rdna::kernels::WB_BUCKETS / 32) as usize;
    let mut bits = vec![0u32; n];
    let count = match what {
        "lazy" => {
            let u = WB_UNTOUCHED.lock();
            for &k in u.iter() {
                bits[(k / 32) as usize] |= 1 << (k % 32);
            }
            u.len()
        }
        "eager" => 0,
        _ => return Err("usage: wbb lazy|eager".into()),
    };
    // The bucket bits for the fault count, and their pages armed no-access
    // (those nothing else armed, in runs) or disarmed again.
    let lb = wb_lazy_bits();
    for (i, w) in lb.iter().enumerate() {
        let v = (bits[2 * i] as u64) | (bits[2 * i + 1] as u64) << 32;
        w.store(v, Ordering::Relaxed);
    }
    let mut runs = WB_LAZY_RUNS.lock();
    for &(a, n) in runs.iter() {
        crate::hosted::disarm_if_armed(a, n);
    }
    runs.clear();
    if what == "lazy" && crate::hosted::on() {
        for &k in WB_UNTOUCHED.lock().iter() {
            let lo = MIRROR_BASE + ((k as u64) << (rdna::kernels::WB_BUCKET_SHIFT as u64 + 12));
            let hi = lo + (1u64 << (rdna::kernels::WB_BUCKET_SHIFT as u64 + 12));
            let mut p = lo;
            while p < hi {
                // A run of pages nothing armed, readable and writable now.
                if crate::hosted::armed(p) || !crate::mem::os_readable(p, PAGE) {
                    p += PAGE;
                    continue;
                }
                let mut q = p + PAGE;
                while q < hi && !crate::hosted::armed(q) && crate::mem::os_readable(q, PAGE) {
                    q += PAGE;
                }
                if crate::hosted::arm_noaccess(p, q - p).is_ok() {
                    runs.push((p, q - p));
                    WB_LAZY_FAULTS[6].fetch_add((q - p) / PAGE, Ordering::Relaxed);
                } else {
                    WB_LAZY_FAULTS[7].fetch_add(1, Ordering::Relaxed);
                }
                p = q;
            }
        }
    }
    let bytes: Vec<u8> = bits.iter().flat_map(|x| x.to_le_bytes()).collect();
    let at = rdna::kernels::WB_LAZY_AT as u64 * 4;
    let c = ctx();
    for on_c in [false, true] {
        let Some(marks) = super::resident::marks_for(on_c) else { continue };
        super::resident::one_off_now(|cmd| unsafe {
            c.device.cmd_update_buffer(cmd, marks, at, &bytes);
        })?;
    }
    Ok(format!("{count} buckets lazy"))
}

fn wb_bits(b: &std::sync::OnceLock<Box<[AtomicU64]>>) -> &[AtomicU64] {
    b.get_or_init(|| (0..(MIRROR_END - MIRROR_BASE) / PAGE / 64).map(|_| AtomicU64::new(0)).collect())
}

fn note_wb_pages(b: &std::sync::OnceLock<Box<[AtomicU64]>>, a: u64, n: u64) {
    let bits = wb_bits(b);
    let lo = a.max(MIRROR_BASE);
    let hi = a.saturating_add(n.max(1)).min(MIRROR_END);
    let mut p = (lo - MIRROR_BASE) / PAGE;
    while lo < hi && MIRROR_BASE + p * PAGE < hi {
        bits[(p / 64) as usize].fetch_or(1 << (p % 64), Ordering::Relaxed);
        p += 1;
    }
}

/// `DS_WB_BUCKETS`: reads the command processors make of guest memory
/// (`Exec::sync_for`).
pub fn note_wb_reads(reads: &[(u64, u64)]) {
    if wb_buckets() {
        for &(a, n) in reads {
            note_wb_pages(&WB_READ_PAGES, a, n);
        }
    }
}

/// Status part (`DS_WB_BUCKETS`): the window's write-back volume by bucket
/// class (the counts read back from both written-bit maps' aux regions,
/// which are cleared; a write-back racing the read-back may be lost to it),
/// the largest buckets nothing else touched by mapping.
fn wb_buckets_line() -> String {
    if !wb_buckets() {
        return String::new();
    }
    let c = ctx();
    let nb = rdna::kernels::WB_BUCKETS as usize;
    let words = 2 * nb as u64;
    let at = rdna::kernels::MARK_AUX_AT as u64 * 4;
    let mut gpu = vec![0u64; 2 * nb];
    for on_c in [false, true] {
        let Some(marks) = super::resident::marks_for(on_c) else { continue };
        let Ok((hbuf, hmem, hptr)) = c.host_buffer(words * 4, vk::BufferUsageFlags::TRANSFER_DST) else {
            return "    write-back buckets (`DS_WB_BUCKETS`): no host buffer\n".into();
        };
        let r = super::resident::one_off_now(|cmd| unsafe {
            c.device.cmd_copy_buffer(cmd, marks, hbuf, &[vk::BufferCopy { src_offset: at, dst_offset: 0, size: words * 4 }]);
            let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_READ).dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
            c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[mb], &[], &[]);
            c.device.cmd_fill_buffer(cmd, marks, at, words * 4, 0);
        });
        if r.is_ok() {
            for (i, g) in gpu.iter_mut().enumerate() {
                *g += unsafe { std::ptr::read_volatile((hptr as *const u32).add(i)) } as u64;
            }
        }
        unsafe {
            c.device.destroy_buffer(hbuf, None);
            c.device.free_memory(hmem, None);
        }
        if let Err(e) = r {
            return format!("    write-back buckets (`DS_WB_BUCKETS`): read-back failed: {e}\n");
        }
    }
    let frame = super::FRAMES.load(Ordering::Relaxed);
    let flips = frame.saturating_sub(WB_FLIP.swap(frame, Ordering::Relaxed)).max(1) as f64;
    let (cpu, rd) = (wb_bits(&WB_CPU_PAGES), wb_bits(&WB_READ_PAGES));
    let per = (1usize << rdna::kernels::WB_BUCKET_SHIFT) / 64;
    // Classes: neither, read by a CP, CPU-written, both: [buckets, dwords, pages].
    let mut cls = [[0u64; 3]; 4];
    let mut quiet: Vec<(u64, u32)> = Vec::new();
    let (mut cpu_pages_mixed, mut pages_mixed) = (0u64, 0u64);
    for k in 0..nb {
        let (dw, pg) = (gpu[k], gpu[nb + k]);
        if dw == 0 && pg == 0 {
            continue;
        }
        let cw: u32 = (0..per).map(|j| cpu[k * per + j].load(Ordering::Relaxed).count_ones()).sum();
        let rw: u32 = (0..per).map(|j| rd[k * per + j].load(Ordering::Relaxed).count_ones()).sum();
        let ci = (rw > 0) as usize + 2 * (cw > 0) as usize;
        cls[ci][0] += 1;
        cls[ci][1] += dw;
        cls[ci][2] += pg;
        if ci == 0 {
            quiet.push((dw, k as u32));
        } else if cw > 0 {
            cpu_pages_mixed += cw as u64;
            pages_mixed += 1 << rdna::kernels::WB_BUCKET_SHIFT;
        }
    }
    for b in [cpu, rd] {
        for x in b.iter() {
            x.store(0, Ordering::Relaxed);
        }
    }
    let (dw_all, pg_all): (u64, u64) = (cls.iter().map(|x| x[1]).sum(), cls.iter().map(|x| x[2]).sum());
    let names = ["untouched", "read by our CPs", "CPU-written", "CPU-written and read"];
    let parts: Vec<String> = cls
        .iter()
        .zip(names)
        .map(|(x, n)| format!("{n}: {} buckets, {:.0}% of dwords ({:.2} M a flip), {:.0}% of pages ({:.0} a flip)", x[0], 100.0 * x[1] as f64 / dw_all.max(1) as f64, x[1] as f64 / flips / 1e6, 100.0 * x[2] as f64 / pg_all.max(1) as f64, x[2] as f64 / flips))
        .collect();
    // The untouched buckets by mapping, the largest first.
    let mut by_map: std::collections::HashMap<(u64, u64, String), u64> = std::collections::HashMap::new();
    *WB_UNTOUCHED.lock() = quiet.iter().map(|q| q.1).collect();
    for &(dw, k) in &quiet {
        let a = MIRROR_BASE + ((k as u64) << (rdna::kernels::WB_BUCKET_SHIFT + 12));
        *by_map.entry(crate::mem::vma_label(a).unwrap_or((a, 0, "?".into()))).or_insert(0) += dw;
    }
    let mut maps: Vec<_> = by_map.into_iter().collect();
    maps.sort_by(|a, b| b.1.cmp(&a.1));
    let top: Vec<String> = maps.iter().take(8).map(|((s, z, n), dw)| format!("{n} {s:#x}+{z:#x} {:.2} M a flip", *dw as f64 / flips / 1e6)).collect();
    let lf: Vec<u64> = WB_LAZY_FAULTS.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    format!(
        "    write-back buckets (`DS_WB_BUCKETS`, 1 MiB, over {flips:.0} flips): {:.2} M dwords and {:.0} pages a flip; {}; the CPU wrote {:.1}% of the pages of the buckets it wrote; untouched by mapping: {}; `wbb lazy` pages armed no-access {} ({} runs failed), faults on them so far: the title {} reads / {} writes, ours {} / {}, other code {} / {}\n",
        dw_all as f64 / flips / 1e6,
        pg_all as f64 / flips,
        parts.join("; "),
        100.0 * cpu_pages_mixed as f64 / pages_mixed.max(1) as f64,
        top.join(", "),
        lf[6],
        lf[7],
        lf[0],
        lf[1],
        lf[2],
        lf[3],
        lf[4],
        lf[5]
    )
}

/// Records the write-back into `cmd` (outside a render pass): a barrier
/// after all earlier work, (`tune touch_bits`) the touched-page list made
/// from the touched bits of `written` (the guest range the executor's
/// written bindings since its last write-back span) and a barrier, the
/// scatter over the touched-page list, the list emptied (by the kernel
/// itself), a barrier before all later work. `on_c`: a batch on the compute
/// family's queue, with its written-bit map (ordered async compute).
/// Returns the pipeline it bound (the caller's bound compute state is gone).
pub fn record_scatter(cmd: vk::CommandBuffer, written: Option<(u64, u64)>, on_c: bool) -> Option<vk::Pipeline> {
    let s = SCATTER.get()?;
    let marks = super::resident::marks_for(on_c)?;
    let set = if on_c { s.set_c } else { s.set };
    let (mbuf, _) = super::resident::mirror_buffer()?;
    let c = ctx();
    let m = c.buffer_address(mbuf);
    let t = super::pagetable::address();
    let mut h = HELD_MAP.get().map_or(0, |m| m.addr);
    // (`DS_WB_SAME=2`: the family's hash table; cleared first when asked.)
    if let Some(ht) = HASHES.get() {
        let k = usize::from(on_c).min(ht.tables.len() - 1);
        h = ht.tables[k].1;
        let bit = 1u32 << k;
        let all = HASH_CLEAR.load(Ordering::Acquire) & bit != 0 && HASH_CLEAR.fetch_and(!bit, Ordering::AcqRel) & bit != 0;
        let ranges = std::mem::take(&mut HASH_CLEAR_RANGES.lock()[k]);
        if all || !ranges.is_empty() {
            unsafe {
                if all {
                    ctx().device.cmd_fill_buffer(cmd, ht.tables[k].0, 0, vk::WHOLE_SIZE, 0);
                } else {
                    for (lo, hi) in ranges {
                        ctx().device.cmd_fill_buffer(cmd, ht.tables[k].0, (lo - MIRROR_BASE) / PAGE * 8, (hi - lo) / PAGE * 8, 0);
                    }
                }
                let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
                ctx().device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &[mb], &[], &[]);
            }
        }
    }
    let pc: Vec<u8> = [m as u32, (m >> 32) as u32, t as u32, (t >> 32) as u32, h as u32, (h >> 32) as u32, 0, 0].iter().flat_map(|x| x.to_le_bytes()).collect();
    let pipeline = s.pipeline;
    unsafe {
        let all = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::MEMORY_WRITE).dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
        c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::DRAW_INDIRECT, vk::DependencyFlags::empty(), &[all], &[], &[]);
        if let Some((lo, hi)) = written.filter(|&(lo, hi)| lo < hi && TOUCH_BITS_EVER.load(Ordering::Relaxed)) {
            // The touched-bit words of the pages in [lo, hi), 32 pages a word.
            let span_words = (MIRROR_END - MIRROR_BASE) / PAGE / 32;
            let w0 = (lo.max(MIRROR_BASE) - MIRROR_BASE) / PAGE / 32;
            let w1 = ((hi.min(MIRROR_END) - MIRROR_BASE).div_ceil(PAGE)).div_ceil(32).min(span_words);
            if w0 < w1 {
                let cpc: Vec<u8> = [w0 as u32, w1 as u32, 0, 0].iter().flat_map(|x| x.to_le_bytes()).collect();
                c.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, s.compact);
                c.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, s.layout, 0, &[set], &[]);
                c.device.cmd_push_constants(cmd, s.layout, vk::ShaderStageFlags::COMPUTE, 0, &cpc);
                c.device.cmd_dispatch(cmd, ((w1 - w0) as u32).div_ceil(rdna::kernels::COMPACT_LOCAL_SIZE), 1, 1);
                let listed = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::INDIRECT_COMMAND_READ);
                c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::DRAW_INDIRECT, vk::DependencyFlags::empty(), &[listed], &[], &[]);
                COMPACTS[0].fetch_add(1, Ordering::Relaxed);
                COMPACTS[1].fetch_add(w1 - w0, Ordering::Relaxed);
            }
        }
        // (`DS_WB_SAME=2`: the skipping write-back while `wbs on`.)
        let pipeline = match SAME_SKIP.get() {
            Some(&p) if SAME_SKIP_ON.load(Ordering::Acquire) => p,
            _ => pipeline,
        };
        c.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
        c.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, s.layout, 0, &[set], &[]);
        c.device.cmd_push_constants(cmd, s.layout, vk::ShaderStageFlags::COMPUTE, 0, &pc);
        c.device.cmd_dispatch_indirect(cmd, marks, rdna::kernels::MARK_LIST_AT as u64 * 4);
        if SCATTERS_RECORDED.fetch_add(1, Ordering::Relaxed) % COUNTER_COPY_EVERY == COUNTER_COPY_EVERY - 1 {
            record_counter_copy(s, cmd);
        }
        // (The kernel emptied the list; the barrier's first scope holds the
        // dispatch's read of its arguments, a logically earlier stage.)
        let after = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE | vk::AccessFlags::HOST_READ);
        c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::ALL_COMMANDS | vk::PipelineStageFlags::HOST, vk::DependencyFlags::empty(), &[after], &[], &[]);
    }
    note_scatter();
    Some(pipeline)
}

/// Records a copy of the write-back's video-memory counters into the
/// host-visible ones (after the write-backs before it; the barrier after
/// the write-back that follows makes it visible to the host).
unsafe fn record_counter_copy(s: &Scatter, cmd: vk::CommandBuffer) {
    let d = s.dev;
    let c = ctx();
    let to_copy = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_READ);
    let bytes = 4 * rdna::kernels::SCATTER_COUNTERS as u64;
    let after = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::HOST_READ);
    unsafe {
        c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[to_copy], &[], &[]);
        c.device.cmd_copy_buffer(cmd, d, s.host, &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size: bytes }]);
        c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::HOST, vk::DependencyFlags::empty(), &[after], &[], &[]);
    }
}

/// The buffer the write-back kernel counts into (its pages and dwords
/// first: `rdna::kernels::SCATTER_COUNTERS`), in video memory.
pub fn counter_buffer() -> Option<vk::Buffer> {
    SCATTER.get().map(|s| s.dev)
}

/// Whether front end 2 is on with its write-back kernel (which empties the
/// touched-page list itself).
pub fn scatter_resets() -> bool {
    on() && SCATTER.get().is_some()
}

/// The write-back's counters summed in 64 bits (the kernel's are 32-bit
/// and wrap within seconds at an arrival's volume: ~10 million dwords a
/// flip), sampled at every flip (`census_sample`): pages, dwords and four
/// counters the lazy write-back study used (FINDINGS.md 2026-10-31 night),
/// then the last values seen of the six.
static WB64: [AtomicU64; 12] = [const { AtomicU64::new(0) }; 12];

/// Adds the counters' growth since the last sample (at each flip, from one
/// thread; the status line reads the sums alone, two samplers could race
/// on the last values).
pub fn census_sample() {
    if !on() || SCATTER.get().is_none() {
        return;
    }
    let k = counters();
    for (i, now) in [k[0], k[1], k[4], k[5], k[6], k[7]].into_iter().enumerate() {
        let prev = WB64[6 + i].swap(now as u64, Ordering::Relaxed) as u32;
        WB64[i].fetch_add(now.wrapping_sub(prev) as u64, Ordering::Relaxed);
    }
}

/// The write-back's counters (pages, dwords, pages without a mapping,
/// overflowed lists, the census's dwords and unchanged ones), as the GPU left them so far (as of the last
/// copy).
fn counters() -> [u32; rdna::kernels::SCATTER_COUNTERS as usize] {
    let Some(s) = SCATTER.get() else { return [0; rdna::kernels::SCATTER_COUNTERS as usize] };
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
    // (GPU-resolved chains stay as set: stage F3, they read the mirror at
    // guest offsets, `Spec::pt_linear`, their per-item ranges placed per
    // command, `exec::fe2_gpu_ranges`.)
    // (So do long runs, off by default: a chunk recorded before an earlier
    // one still uploads what it finds invalid, its uploads read guest memory
    // when they execute, after the earlier chunks' write-backs, and its pages
    // are valid only once it is queued, in program order; its CPU reads are
    // checked at its turn against the run's earlier writes as before.)
    const OFF: [&str; 4] = ["homes", "shadow_args", "pre_share", "host_census"];
    for name in OFF {
        if !super::tune::set(name, 0) {
            crate::warn!("fe2: no switch {name}");
        }
    }
    // Compute queues' synchronization-free runs are recorded in parallel
    // (four workers, flushes kept inside runs, runs from 64 commands: +6-10%
    // at the vista once checks follow each other's takes; a loss in the
    // default front end, FINDINGS.md 2026-10-28 night; `cp::compute_workers`,
    // without switches since 2026-11-01 but the run length), and waits passed on
    // another queue's promise adopt its writer's view instead of ending an
    // epoch (+3.6% at the vista with stretches checked at their labels,
    // FINDINGS.md 2026-10-29), and waits that release nothing new (gone
    // past, a counter's value passed on again) end none (epoch ends -38%,
    // takes -10%, level), and the write-back's kernel empties its
    // touched-page list itself (level to +1%), counts into video memory
    // (GPU busy -2.3 ms a frame, +2-3%) and counts as a full barrier, CP
    // writes right after it recording none before them (level to +0.3%;
    // FINDINGS.md 2026-10-29 morning): those five without switches since
    // 2026-11-01; the parallel runs' length unless set explicitly.
    const PAR: [(&str, u64); 1] = [("par_min_compute", 64)];
    for (name, v) in PAR {
        if !super::tune::set_default(name, v) {
            crate::warn!("fe2: no switch {name}");
        }
    }
    let made = match super::resident::make_for_fe2() {
        Ok(s) => s,
        Err(e) => {
            crate::error!("fe2: no mirror: {e}; front end 2 off");
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
    match make_scatter(marks, super::resident::marks_for(true)) {
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
            if std::env::var("DS_FE2_SCATTER_BENCH").ok().is_some_and(|v| !v.is_empty() && v != "0") {
                match scatter_bench() {
                    Ok(l) => crate::info!("fe2: {l}"),
                    Err(e) => crate::warn!("fe2: write-back bench: {e}"),
                }
            }
            ON.store(true, Ordering::Release);
            if crate::hosted::on() {
                // (`DS_FE2_WORD=0`, diagnosis: the host answers a fault for
                // its page only.)
                if std::env::var("DS_FE2_WORD").ok().as_deref() != Some("0") {
                    crate::hosted::set_word_mode();
                }
                if hold_census() {
                    crate::hosted::set_held_answered();
                    if HELD_MAP.get().is_some() {
                        let _ = std::thread::Builder::new().name("fe2-hold".into()).spawn(hold_thread);
                    }
                }
                let _ = std::thread::Builder::new().name("fe2-faults".into()).spawn(watch);
            }
            crate::info!(
                "fe2: front end 2 on ({made}; {s}; hosted arming {}{}; switched off: {}; compute runs recorded in parallel on 4 workers, from {} commands)",
                if crate::hosted::on() { "on" } else { "off (not under the host)" },
                if arm_gpu() { "" } else { ", GPU-written pages never armed" },
                OFF.join(", "),
                super::tune::par_min_compute()
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
            record_scatter(cmd, None, false);
            if let Some(s) = SCATTER.get() {
                record_counter_copy(s, cmd);
            }
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

/// `DS_FE2_SCATTER_BENCH=1` (diagnosis, at front end 2's start): the
/// write-back's GPU time on synthetic inputs as the segment profile counts
/// it (from after the barrier behind its inputs to after its last barrier;
/// the median of BENCH_REPS): an empty list, then 64 private pages (`=<n>`:
/// n, 64-16384 in words of 32) imported and mapped at the span's end (as
/// the self-test's page) listed without marks and with 1, 32 consecutive,
/// 32 spread (one in 32) and all 1024 dwords of each marked; then, for the
/// rate of stores into host memory by other means, the same pages copied
/// from the mirror into the import and the import filled (each between the
/// same barriers). Its pages and marks are cleared again after.
fn scatter_bench() -> Result<String, String> {
    use windows_sys::Win32::System::Memory::{VirtualAlloc, VirtualFree, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE};
    let want: u64 = std::env::var("DS_FE2_SCATTER_BENCH").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    let bench_pages: u64 = if want <= 1 { 64 } else { want.clamp(64, 16384).next_multiple_of(32) };
    const BENCH_REPS: usize = 15;
    // (`cmd_update_buffer` takes at most 64 KiB.)
    let update = |cmd: vk::CommandBuffer, buf: vk::Buffer, at: u64, data: &[u8]| {
        for (k, part) in data.chunks(65536).enumerate() {
            unsafe { ctx().device.cmd_update_buffer(cmd, buf, at + 65536 * k as u64, part) };
        }
    };
    let c = ctx();
    let marks = super::resident::marks().ok_or("no written-bit map")?;
    let table = super::pagetable::buffer().ok_or("no page table")?;
    let guest = MIRROR_END - bench_pages * PAGE;
    let p0 = (guest - MIRROR_BASE) / PAGE;
    if p0 % 32 != 0 {
        return Err("its pages do not start a word of touched bits".into());
    }
    super::resident::back(guest, bench_pages * PAGE)?;
    let mirror = super::resident::mirror_buffer().ok_or("no mirror")?.0;
    let h = unsafe { VirtualAlloc(std::ptr::null(), (bench_pages * PAGE) as usize, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE) } as u64;
    if h == 0 {
        return Err("VirtualAlloc".into());
    }
    let r = (|| {
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
        let (hb, hm, hva) = c.import_host(h, bench_pages * PAGE, usage)?;
        let entries: Vec<u8> = (0..bench_pages).flat_map(|i| (hva + i * PAGE).to_le_bytes()).collect();
        let n = 2 * BENCH_REPS as u32;
        let qp = match unsafe { c.device.create_query_pool(&vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(n), None) } {
            Ok(q) => q,
            Err(e) => {
                unsafe {
                    c.device.destroy_buffer(hb, None);
                    c.device.free_memory(hm, None);
                }
                return Err(format!("query pool: {e}"));
            }
        };
        let period = c.limits.timestamp_period as f64;
        let bits_at = 4 * (rdna::kernels::MARK_PAGE_BITS_AT as u64 + p0 / 32);
        let names = ["an empty list", "pages without marks", "1 dword a page", "32 consecutive dwords a page", "32 dwords a page one in 32", "every dword", "copied whole", "filled", "driver memory filled"];
        // (Host memory the driver allocated, for case 8.)
        let (drv, drv_mem, _) = c.host_buffer(bench_pages * PAGE, vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC)?;
        let pattern = |k: usize| -> [u32; 32] {
            let mut m = [0u32; 32];
            match k {
                2 => m[0] = 1,
                3 => m[0] = u32::MAX,
                4 => m = [1; 32],
                5 => m = [u32::MAX; 32],
                _ => {}
            }
            m
        };
        let mut lines = Vec::new();
        let res = (|| {
            for (k, name) in names.iter().enumerate() {
                // (6 and 7: no write-back; the list stays empty.)
                let listed = if k == 0 || k >= 6 { 0 } else { bench_pages };
                let mask: Vec<u8> = pattern(k).iter().flat_map(|x| x.to_le_bytes()).collect();
                let masks: Vec<u8> = (0..bench_pages).flat_map(|_| mask.iter().copied()).collect();
                let touched: Vec<u8> = (0..bench_pages / 32).flat_map(|_| (if listed > 0 { u32::MAX } else { 0u32 }).to_le_bytes()).collect();
                let list: Vec<u8> = [listed as u32, 1, 1, 0].into_iter().chain((0..listed).map(|i| (p0 + i) as u32)).flat_map(|x| x.to_le_bytes()).collect();
                let bytes = bench_pages * PAGE;
                super::resident::one_off_now(|cmd| unsafe {
                    c.device.cmd_reset_query_pool(cmd, qp, 0, n);
                    update(cmd, table, p0 * 8, &entries);
                    for r in 0..BENCH_REPS as u32 {
                        update(cmd, marks, p0 * 128, &masks);
                        update(cmd, marks, bits_at, &touched);
                        update(cmd, marks, rdna::kernels::MARK_LIST_AT as u64 * 4, &list);
                        let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
                        c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::ALL_COMMANDS, vk::DependencyFlags::empty(), &[mb], &[], &[]);
                        c.device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, qp, 2 * r);
                        if k < 6 {
                            record_scatter(cmd, None, false);
                        } else {
                            let all = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::MEMORY_WRITE).dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
                            c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[all], &[], &[]);
                            if k == 6 {
                                c.device.cmd_copy_buffer(cmd, mirror, hb, &[vk::BufferCopy { src_offset: guest - MIRROR_BASE, dst_offset: 0, size: bytes }]);
                            } else {
                                c.device.cmd_fill_buffer(cmd, if k == 8 { drv } else { hb }, 0, bytes, 0);
                            }
                            let after = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE | vk::AccessFlags::HOST_READ);
                            c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::ALL_COMMANDS | vk::PipelineStageFlags::HOST, vk::DependencyFlags::empty(), &[after], &[], &[]);
                        }
                        c.device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, qp, 2 * r + 1);
                        // (The next round's inputs after it.)
                        let next = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::MEMORY_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
                        c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[next], &[], &[]);
                    }
                })?;
                let mut t = vec![0u64; n as usize];
                unsafe { c.device.get_query_pool_results(qp, 0, &mut t, vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT) }.map_err(|e| format!("query: {e}"))?;
                let mut us: Vec<f64> = (0..BENCH_REPS).map(|r| t[2 * r + 1].wrapping_sub(t[2 * r]) as f64 * period / 1000.0).collect();
                us.sort_by(|a, b| a.total_cmp(b));
                let m = us[BENCH_REPS / 2];
                // Bytes stored into host memory: the marked dwords, or all of it.
                let stored = match k {
                    2 => bench_pages * 4,
                    3 | 4 => bench_pages * 128,
                    5..=8 => bytes,
                    _ => 0,
                };
                if stored >= 1 << 20 {
                    lines.push(format!("{name} {m:.1} us ({:.1} GB/s)", stored as f64 / (m * 1000.0)));
                } else {
                    lines.push(format!("{name} {m:.1} us"));
                }
            }
            // The copy queue's engine storing the pages into the import:
            // BENCH_REPS copies in one submission, timed on the CPU.
            if let Some(q) = c.copy.as_ref() {
                let bytes = bench_pages * PAGE;
                unsafe {
                    let pool = c.device.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(q.family), None).map_err(|e| format!("copy pool: {e}"))?;
                    let r = (|| {
                        let cb = c.device.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(pool).command_buffer_count(1)).map_err(|e| format!("copy cmd: {e}"))?[0];
                        c.device.begin_command_buffer(cb, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).map_err(|e| format!("begin: {e}"))?;
                        for _ in 0..BENCH_REPS {
                            c.device.cmd_copy_buffer(cb, mirror, hb, &[vk::BufferCopy { src_offset: guest - MIRROR_BASE, dst_offset: 0, size: bytes }]);
                            let ww = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::HOST_READ);
                            c.device.cmd_pipeline_barrier(cb, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::HOST, vk::DependencyFlags::empty(), &[ww], &[], &[]);
                        }
                        c.device.end_command_buffer(cb).map_err(|e| format!("end: {e}"))?;
                        let fence = c.device.create_fence(&vk::FenceCreateInfo::default(), None).map_err(|e| format!("fence: {e}"))?;
                        let t0 = std::time::Instant::now();
                        let sent = c.submit_timeline(true, &[cb], &[], &[], fence);
                        let waited = sent.and_then(|_| c.device.wait_for_fences(&[fence], true, 10_000_000_000).map_err(|e| format!("wait: {e}")));
                        let us = t0.elapsed().as_secs_f64() * 1e6 / BENCH_REPS as f64;
                        c.device.destroy_fence(fence, None);
                        waited?;
                        lines.push(format!("copied whole on the copy queue {us:.1} us ({:.1} GB/s, CPU-timed)", bytes as f64 / (us * 1000.0)));
                        Ok::<(), String>(())
                    })();
                    c.device.destroy_command_pool(pool, None);
                    r?;
                }
            }
            Ok::<(), String>(())
        })();
        unsafe {
            c.device.destroy_buffer(drv, None);
            c.device.free_memory(drv_mem, None);
        }
        // Its pages unmapped and their marks cleared again, whatever happened.
        let zeros_t = vec![0u8; entries.len()];
        let zeros_m = vec![0u8; (bench_pages * 128) as usize];
        let zeros_b = vec![0u8; (bench_pages / 32 * 4) as usize];
        let cleared = super::resident::one_off_now(|cmd| unsafe {
            update(cmd, table, p0 * 8, &zeros_t);
            update(cmd, marks, p0 * 128, &zeros_m);
            update(cmd, marks, bits_at, &zeros_b);
            c.device.cmd_update_buffer(cmd, marks, rdna::kernels::MARK_LIST_AT as u64 * 4, &[0u8; 4]);
        });
        unsafe {
            c.device.destroy_query_pool(qp, None);
            c.device.destroy_buffer(hb, None);
            c.device.free_memory(hm, None);
        }
        res?;
        cleared?;
        Ok(format!("write-back bench ({bench_pages} pages, median of {BENCH_REPS}): {}", lines.join(", ")))
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
    let spans = super::exec::fe2_span_census();
    let ends = super::precache::epoch_ends();
    let ms = |ns: u64| ns as f64 / 1e6;
    let sizes: Vec<String> = ["<64K", "<1M", "<16M", ">=16M"]
        .iter()
        .zip(BY_SIZE.iter())
        .map(|(n, c)| {
            let v: Vec<u64> = c.iter().map(|x| x.load(Ordering::Relaxed)).collect();
            format!("{n} {} ({} MiB; {} takes, {:.1} ms)", v[0], (v[1] * PAGE) >> 20, v[2], ms(v[3]))
        })
        .collect();
    let w: Vec<u64> = WB64.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    let compacts = if COMPACTS[0].load(Ordering::Relaxed) > 0 { format!("    front end 2's touched-page lists made from touched bits (`tune touch_bits`): {} write-backs, {} words of bits scanned\n", COMPACTS[0].load(Ordering::Relaxed), COMPACTS[1].load(Ordering::Relaxed)) } else { String::new() };
    format!(
        "    front end 2: {} placement checks ({} pages; {} needed nothing; {} chunks passed by their summary, {} placements by the range cache), {} takes ({} pages written, {} failed; {:.1} ms), checking {:.1} ms; uploads {} ({} pages, {} MiB; runs written lately, without a take: {}, {} pages), {} failed, {} pages valid once queued ({} raced a write); hosted arming: {} pages in {} runs ({} failed; {} view pages armed at the residency's takes adopted), {} armings ended by writes, {} writes found on pages armed and taken since (holes); checks that followed another's take {} ({} pages uploaded without a take of their own, {} took anyway); faults read {} ({} ending an arming of ours, {} outside the span, {} ring overflows), {} chunk bursts ({} pages disarmed), {} more pages unarmed with a faulted page's word; invalidated: {} ranges, {} pages by takes; write-backs recorded {}: {} pages, {} dwords, {} pages unmapped (lost), {} lists past their capacity (before GPU-performed CP writes {}, at batch closes {}, {} of them with no label write and no completion action; CP writes with work to write back {} of labels a CP of ours waits on, {} left to the batch's close, {} with their guest-memory copy at the close); CP writes also into the mirror {}, pages copied by the CPU {}, written bindings without an import {}; imports {} live ({} uploaded from, {} settled for arming); views synchronized {} times ({} raised; by the graphics CP's own synchronization {}, the compute queues' {}; epochs ended by {} title submissions, {} waits the CPU released, {} REWIND, {} waits passed on GPU writes ended none, {} passed again on a counter's value, {} gone past ended none); the mirror's video memory {} MiB (no placement in the last 1,800 flips: {} MiB; unbound since boot {} MiB, {} blocks freed, {} cold words kept by placements), its written-bit maps' {} MiB\n    front end 2's placement checks by size: {}\n    front end 2's pages taken but not armed: {}\n",
        s[S_ENSURES],
        s[S_ENSURE_PAGES],
        s[S_FAST],
        s[S_SUMMARY],
        s[S_RANGE_HITS],
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
        s[S_ADOPTED],
        s[S_ARM_ENDED],
        HOLES.load(Ordering::Relaxed),
        FOLLOW[0].load(Ordering::Relaxed),
        FOLLOW[1].load(Ordering::Relaxed),
        FOLLOW[2].load(Ordering::Relaxed),
        FAULTS[F_SPAN].load(Ordering::Relaxed),
        FAULTS[F_ENDED].load(Ordering::Relaxed),
        FAULTS[F_OUTSIDE].load(Ordering::Relaxed),
        FAULTS[F_LOST].load(Ordering::Relaxed),
        FAULTS[F_BURSTS].load(Ordering::Relaxed),
        FAULTS[F_DISARMED].load(Ordering::Relaxed),
        FAULTS[F_WORD_DISARMED].load(Ordering::Relaxed),
        s[S_INVAL_CALLS],
        s[S_INVAL_TAKEN],
        s[S_SCATTERS],
        w[0],
        w[1],
        k[2],
        k[3],
        super::exec::SCATTER_SITES[0].load(Ordering::Relaxed),
        super::exec::SCATTER_SITES[1].load(Ordering::Relaxed),
        super::exec::SCATTER_SITES[2].load(Ordering::Relaxed),
        super::exec::SCATTER_SITES[3].load(Ordering::Relaxed),
        super::exec::SCATTER_SITES[4].load(Ordering::Relaxed),
        super::exec::SCATTER_SITES[5].load(Ordering::Relaxed),
        s[S_CP_MIRROR],
        s[S_MEMCPY],
        s[S_UNSCATTERABLE],
        spans.0,
        spans.1,
        spans.2,
        VIEW_SYNCS[0].load(Ordering::Relaxed),
        VIEW_SYNCS[1].load(Ordering::Relaxed),
        VIEW_RAISES[0].load(Ordering::Relaxed),
        VIEW_RAISES[1].load(Ordering::Relaxed),
        ends[0],
        ends[1],
        ends[2],
        super::cp::EPOCH_GPU_WAITS.load(Ordering::Relaxed),
        super::cp::QUIET_WAITS[0].load(Ordering::Relaxed),
        super::cp::QUIET_WAITS[1].load(Ordering::Relaxed),
        super::resident::backed_mib(),
        super::resident::cold_mib(1800),
        super::resident::unbacked_status().0,
        super::resident::unbacked_status().1,
        super::resident::unbacked_status().2,
        super::resident::marks_mib(),
        sizes.join(", "),
        ["armed already", "GPU-written", "written lately", "too few clean epochs", "no settled window", "not plain read-write", "chunk held", "import list busy"]
            .iter()
            .zip(NOT_ARMED.iter())
            .map(|(n, c)| format!("{n} {}", c.load(Ordering::Relaxed)))
            .collect::<Vec<_>>()
            .join(", "),
    ) + &census_line()
        + &run_share_line()
        + &callers_line()
        + &compacts
        + &wb_buckets_line()
        + &hold_line()
        + &wb_same_line()
}

/// One status line: the mirror's pages by state now (a scan of every
/// chunk's words): valid, armed (the host's bit set), GPU-written, and those
/// found written (or first seen) within the last 8, 60 and 600 flips with the
/// runs they form (the volatile set a coherence without per-binding checks
/// would have to keep current: FRONTEND2.md, F3).
fn census_line() -> String {
    let frame = super::FRAMES.load(Ordering::Relaxed);
    let (mut valid, mut armed, mut gpu) = (0u64, 0u64, 0u64);
    let mut lately = [0u64; 3];
    let mut runs = [0u64; 3];
    let windows = [8u64, 60, 600];
    for (k, slot) in TABLE.iter().enumerate() {
        let p = slot.load(Ordering::Acquire);
        if p.is_null() {
            continue;
        }
        let c = unsafe { &*p };
        let base = MIRROR_BASE + ((k as u64) << CHUNK_SHIFT);
        let mut prev = [false; 3];
        for w in 0..WORDS {
            let v = c.valid[w].load(Ordering::Relaxed);
            valid += v.count_ones() as u64;
            armed += (c.armed[w].load(Ordering::Relaxed) & crate::hosted::armed_word(base + (w as u64) * 64 * PAGE)).count_ones() as u64;
            gpu += c.gpu[w].load(Ordering::Relaxed).count_ones() as u64;
            if c.seen[w].load(Ordering::Relaxed) == 0 {
                prev = [false; 3];
                continue;
            }
            for b in 0..64 {
                let wrote = c.wrote[w * 64 + b].load(Ordering::Relaxed) as u64;
                for (i, &n) in windows.iter().enumerate() {
                    let hit = wrote != 0 && frame.saturating_sub(wrote - 1) < n;
                    if hit {
                        lately[i] += 1;
                        if !prev[i] {
                            runs[i] += 1;
                        }
                    }
                    prev[i] = hit;
                }
            }
        }
    }
    let mib = |pages: u64| (pages * PAGE) >> 20;
    format!(
        "    front end 2's pages now: valid {} ({} MiB), armed {} ({} MiB), GPU-written {}; written or first seen within 8 / 60 / 600 flips: {} / {} / {} pages ({} / {} / {} MiB) in {} / {} / {} runs
",
        valid,
        mib(valid),
        armed,
        mib(armed),
        gpu,
        lately[0],
        lately[1],
        lately[2],
        mib(lately[0]),
        mib(lately[1]),
        mib(lately[2]),
        runs[0],
        runs[1],
        runs[2]
    )
}

/// The placement checks by caller (`DS_FE2_CALLERS=1`; "" otherwise).
pub fn callers_line() -> String {
    let mut quiet = up_census_line();
    if quiet_census_on() {
        let parts: Vec<String> = CALLER_NAMES
            .iter()
            .zip(QUIET_TAKES.iter())
            .map(|(n, c)| {
                let v: Vec<u64> = c.iter().map(|x| x.load(Ordering::Relaxed)).collect();
                format!("{n} {} of {} ({:.1} of {:.1} ms)", v[2], v[0], v[3] as f64 / 1e6, v[1] as f64 / 1e6)
            })
            .collect();
        quiet += &format!("    front end 2's takes of ranges all GPU-written with no CPU write found for {QUIET_FLIPS} flips (what view-backed mappings would spare), by caller: {}\n", parts.join("; "));
    }
    if !callers_on() {
        return quiet;
    }
    let hz = super::threadcpu::tsc_hz();
    let line = |rows: &[[AtomicU64; 22]; 4]| -> Vec<String> {
        CALLER_NAMES
        .iter()
        .zip(rows.iter())
        .map(|(n, c)| {
            let v: Vec<u64> = c.iter().map(|x| x.load(Ordering::Relaxed)).collect();
            format!(
                "{n}: {} placements ({:.1} ms; outside the checks: mapping and backing {:.1}, batch and import {:.1} (the batch's begin {:.1}), after {:.1}), {} checks ({} by the range cache; missed: no range {}, another epoch {}, chunk generations {}, host events {}), {} words scanned, {} takes, {:.1} ms (range caches {:.1}, scans {:.1}, arming {:.1}, takes {:.1}, runs made current {:.1}, following other takes {:.1}, uploads without a take {:.1}, the rest {:.1})",
                v[C_PLACES],
                v[C_PLACE_CYCLES] as f64 * 1e3 / hz,
                v[C_PART_MAP] as f64 * 1e3 / hz,
                v[C_PART_BEGIN] as f64 * 1e3 / hz,
                v[C_PART_BATCH] as f64 * 1e3 / hz,
                v[C_PART_AFTER] as f64 * 1e3 / hz,
                v[C_CHECKS],
                v[C_HITS],
                v[C_NO_RANGE],
                v[C_EPOCH],
                v[C_GENS],
                v[C_HOST],
                v[C_WORDS],
                v[C_TAKES],
                v[C_CYCLES] as f64 * 1e3 / hz,
                v[C_CACHE] as f64 * 1e3 / hz,
                v[C_SCAN] as f64 * 1e3 / hz,
                v[C_ARM] as f64 * 1e3 / hz,
                v[C_TAKE] as f64 * 1e3 / hz,
                v[C_FIX] as f64 * 1e3 / hz,
                v[C_FOLLOW] as f64 * 1e3 / hz,
                v[C_UPLOAD] as f64 * 1e3 / hz,
                v[C_CYCLES].saturating_sub(v[C_CACHE] + v[C_SCAN] + v[C_ARM] + v[C_TAKE] + v[C_FIX] + v[C_FOLLOW] + v[C_UPLOAD]) as f64 * 1e3 / hz
            )
        })
        .collect()
    };
    let parts = line(&BY_CALLER);
    let firsts = line(&BY_CALLER_FIRST);
    let sources: Vec<String> = SOURCE_NAMES
        .iter()
        .zip(BY_SOURCE.iter())
        .map(|(n, c)| {
            let v: Vec<u64> = c.iter().map(|x| x.load(Ordering::Relaxed)).collect();
            format!(
                "{n}: {} bindings, {} checks ({} by the range cache), {:.1} ms, {} runs taken ({} pages; only pages found lacking by a take of the epoch wanted {}, with a valid page not looked at since {}, with a page M lacks not looked at since {}; pages not looked at since: one epoch before {}, older {})",
                v[B_BINDINGS],
                v[B_CHECKS],
                v[B_HITS],
                v[B_CYCLES] as f64 * 1e3 / hz,
                v[B_TAKES],
                v[B_PAGES],
                v[B_LACKING],
                v[B_VALID_STALE],
                v[B_INVALID_STALE],
                v[B_ONE_BEHIND],
                v[B_OLDER]
            )
        })
        .collect();
    format!(
        "    front end 2's checks by caller: {}\n    of them at batches' and chunks' first commands: {}\n    the graphics CP's checks by the stage they bind: {}\n{quiet}",
        parts.join("; "),
        firsts.join("; "),
        sources.join("; ")
    )
}
