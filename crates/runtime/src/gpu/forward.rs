//! Forwarded copies (`tune forward_copies`). The title fills its per-view
//! constant blocks (the graphics queue's 0x1008880010 at the courtyard gate:
//! a 4x4 matrix at +0xc0, image descriptors at +0xa00..+0xc00) with a small
//! compute kernel that copies CPU-staged words, `if (i < cb[0]) dst[i] =
//! src[i % cb[1]]` over 32-bit elements (cs 0x1008881200 in 01.007.000).
//! Draws then resolve descriptors and constants from the block, and the
//! command processor waited for the copy to complete on the GPU before each
//! draw that read a freshly copied block (~1 ms each, ~12 ms per frame at the
//! gate: FINDINGS.md 2026-10-14). The copy's result is known when it is
//! recorded (its control block and source are written by the CPU): later
//! reads of the destination by the graphics queue's command processor and
//! its parallel workers (stage preparations and their checks) take these
//! words instead of waiting. The copy itself still runs on the GPU in order.
//! An entry ends when other GPU work or a CP write touches its destination
//! (later in the stream: the words would be stale), when a reader finds a
//! word that is neither the destination's old word (the copy has not run)
//! nor the copied one (it has): the title's CPU rewrote it, or once its
//! batch completed: memory then holds the words, which are compared
//! (`STATS`); a destination whose copy did not land is not forwarded again
//! (unless a few words missed, or its inputs were unchanged: `completed`).

use parking_lot::{Mutex, RwLock};
use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use super::shaders::{Prepared, Resolved};

/// The words a copy leaves at `base` as the command processor computed them.
#[derive(Clone)]
struct Entry {
    id: u64,
    base: u64,
    words: Arc<[u32]>,
    /// The batch that performs the copy (0 while it is being recorded).
    seq: u64,
    program: u64,
    hash: u64,
    /// What it was computed from: the source's words and the control block.
    from: Inputs,
    /// The destination's words when the copy was recorded.
    old: Arc<[u32]>,
    /// The words of earlier copies into the same destination that the
    /// GPU may not have run yet when this one was recorded (at most
    /// EARLIER_MAX): memory holds one of them until this copy runs, which
    /// is not a sign of the title's rewrite (several copies into one block
    /// are recorded before the GPU runs the first; such reads had dropped
    /// the entry, and later readers waited for the GPU).
    earlier: Arc<[Arc<[u32]>]>,
    /// Diagnostics: how the dispatch bound its destination, the recording
    /// queue, and the home write-backs so far when it was recorded.
    how: &'static str,
    queue: Arc<str>,
    write_backs: u64,
    /// The flip it was recorded at (diagnostics).
    flip: u64,
}

/// A copy's inputs as read when it was recorded.
#[derive(Clone)]
pub struct Inputs {
    src: u64,
    stride: u64,
    src_words: Vec<u32>,
    cb: u64,
    count: u64,
    modulus: u64,
}

impl Inputs {
    /// The words they were read as, by where: the control block's two, the source's records
    /// (a scout keeps them for its content check, `Exec::note_content`).
    pub fn words(&self) -> Vec<(u64, Vec<u32>)> {
        let mut v = vec![(self.cb, vec![self.count as u32, self.modulus as u32])];
        if self.stride == 4 {
            v.push((self.src, self.src_words.clone()));
        } else {
            v.extend(self.src_words.iter().enumerate().map(|(j, &w)| (self.src + j as u64 * self.stride, vec![w])));
        }
        v
    }
}

static ENTRIES: RwLock<Vec<Entry>> = RwLock::new(Vec::new());
static COUNT: AtomicUsize = AtomicUsize::new(0);
static GEN: AtomicU64 = AtomicU64::new(1);
static NEXT_ID: AtomicU64 = AtomicU64::new(1);
/// Where the live entries' destinations lie, [SPAN[0], SPAN[1]) (set with
/// every change of `ENTRIES`, under its lock): a read outside it and the
/// thread's held entries overlaps none, and `uncovered` answers without
/// making the thread's copy of the entries again (made again after every
/// change: ~1,400 a flip at the vista, ~3-4% of the graphics command
/// processor's time, 2026-10-28).
static SPAN: [AtomicU64; 2] = [AtomicU64::new(u64::MAX), AtomicU64::new(0)];

/// `SPAN` for the entries `g` (under the entries' lock).
fn note_span(g: &[Entry]) {
    let (lo, hi) = g.iter().fold((u64::MAX, 0u64), |(lo, hi), e| (lo.min(e.base), hi.max(e.base + 4 * e.words.len() as u64)));
    SPAN[0].store(lo, Ordering::Release);
    SPAN[1].store(hi, Ordering::Release);
}
/// At most this many entries live (the oldest goes first).
const MAX_ENTRIES: usize = 64;

/// [copies forwarded, entries ended by a later write, entries retired after
/// their batch, words served, stage reads that needed no wait thanks to
/// them, words compared equal after the copy ran, words still as before the
/// copy then (it did not land), copies not forwarded (inputs GPU work still
/// writes, odd formats or sizes), entries ended because a reader found the
/// title had rewritten the destination, words the title rewrote after the
/// copy ran, entries not compared because their destination lies in a
/// video-memory home].
pub static STATS: [AtomicU64; 11] = [const { AtomicU64::new(0) }; 11];

/// A copy kernel's words checked against memory once its batch completed
/// (`DS_FE2_CHECK_COPIES`). Never served to readers.
struct Check {
    id: u64,
    base: u64,
    words: Vec<u32>,
    /// The destination's words when the copy was recorded.
    old: Vec<u32>,
    /// Its batch (0 until submitted).
    seq: u64,
    /// What it checks (a worker's buffer update, or a copy left a dispatch:
    /// `add_check`) and the recording queue (diagnostics).
    what: &'static str,
    queue: Arc<str>,
}

static CHECKS: Mutex<Vec<Check>> = Mutex::new(Vec::new());
static CHECK_COUNT: AtomicUsize = AtomicUsize::new(0);
/// Live checks per 64 KiB page of their destinations, by a hash of the
/// page: a write whose pages all count 0 ends none (no lock taken).
static CHECK_PAGES: [std::sync::atomic::AtomicU32; 1024] = [const { std::sync::atomic::AtomicU32::new(0) }; 1024];

fn check_pages(base: u64, n: u64) -> impl Iterator<Item = &'static std::sync::atomic::AtomicU32> {
    let (first, last) = (base >> 16, (base + n.max(1) - 1) >> 16);
    // At most every slot once (a range over 1024 pages counts them all).
    let pages = (last - first + 1).min(1024);
    (0..pages).map(move |k| &CHECK_PAGES[((first + k).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 54) as usize])
}

fn check_untrack(c: &Check) {
    for slot in check_pages(c.base, 4 * c.words.len() as u64) {
        slot.fetch_sub(1, Ordering::AcqRel);
    }
}
/// At most this many checks wait for their batches (the oldest goes first).
const MAX_CHECKS: usize = 4096;
/// The workers' buffer updates: [recorded, words, words equal after their
/// batch completed, words still as before (it did not land), words other
/// than both (rewritten since), checks ended by a later write].
pub static CHECK_STATS: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

/// Resource indices of a recognized copy kernel.
#[derive(Clone, Copy)]
struct Roles {
    src: usize,
    dst: usize,
    cb: usize,
}

/// The copy kernel's code hash once recognized (0: not yet).
static COPY_HASH: AtomicU64 = AtomicU64::new(0);

/// Recent copy-kernel dispatches of every queue (diagnostics of copies that
/// did not land): (queue, destination, bytes, flip), the newest last.
static RECENT_COPIES: Mutex<std::collections::VecDeque<(Arc<str>, u64, u64, u64)>> = Mutex::new(std::collections::VecDeque::new());

/// A dispatch of `p` on queue `queue` (`exec`, every queue): a copy
/// kernel's goes into `RECENT_COPIES` with its destination.
pub fn note_dispatch(p: &Prepared, groups: [u32; 3], queue: &str) {
    let h = COPY_HASH.load(Ordering::Acquire);
    if h == 0 || p.code_hash != h {
        return;
    }
    let Some(r) = roles(p) else { return };
    let Some(Resolved::Buffer(dst)) = p.found.resources.get(r.dst) else { return };
    let bytes = (groups[0] as u64 * 64).min(dst.num_records as u64) * 4;
    let mut g = RECENT_COPIES.lock();
    if g.len() >= 512 {
        g.pop_front();
    }
    g.push_back((Arc::from(queue), dst.base, bytes, super::FRAMES.load(Ordering::Relaxed)));
}

/// The recent copies over [a, a+n) (`RECENT_COPIES`), newest first.
fn copies_over(a: u64, n: u64) -> String {
    let g = RECENT_COPIES.lock();
    let v: Vec<String> = g.iter().rev().filter(|c| c.1 < a + n && a < c.1 + c.2).take(8).map(|c| format!("{} {:#x}+{:#x} at flip {}", c.0, c.1, c.2, c.3)).collect();
    if v.is_empty() { "none in the ring".into() } else { v.join("; ") }
}

/// Recognition per program code hash (Some: the copy kernel).
static PROGRAMS: Mutex<Option<HashMap<u64, Option<Roles>>>> = Mutex::new(None);
/// (program hash, destination) pairs whose forwarded words once differed
/// from memory: not forwarded again.
static DISTRUSTED: Mutex<Option<HashSet<(u64, u64)>>> = Mutex::new(None);

thread_local! {
    /// This thread reads guest memory for the graphics queue.
    static HERE: Cell<bool> = const { Cell::new(false) };
    /// This thread's copy of the entries: (generation, [(base, bytes,
    /// copied words, old words)], the span [lo, hi) they lie in).
    static LOCAL: RefCell<(u64, Vec<LocalEntry>, (u64, u64), Option<Arc<Frozen>>)> = const { RefCell::new((0, Vec::new(), (0, 0), None)) };
}

/// Diagnostics (traced waits): the forwarding state over [a, a+n) as this
/// thread sees it.
pub fn describe(a: u64, n: u64) -> String {
    let here = HERE.with(|h| h.get());
    let global: Vec<String> = ENTRIES.read().iter().filter(|e| e.base < a + n && a < e.base + 4 * e.words.len() as u64).map(|e| format!("{:#x}+{:#x} batch {}", e.base, 4 * e.words.len(), e.seq)).collect();
    let local = if active() { with_local_span(|v, _| v.iter().filter(|e| e.0 < a + n && a < e.0 + e.1).count()) } else { 0 };
    format!("forwarding here {here}, entries {} (global: {}), this thread's copy {local}", COUNT.load(Ordering::Acquire), if global.is_empty() { "none".to_string() } else { global.join(", ") })
}

/// The graphics command processor's own view while it records past a
/// parallel run (`Exec::past_begin`, `tune forward_held`): the run's workers
/// record earlier commands at the same time, so the copies it forwards meanwhile
/// stay its own (`entries`), and the global entries its writes end are only
/// hidden from it (`hidden`, by id); the stretch's commit publishes both
/// (`hold_commit`), its drop forgets them (`hold_drop`).
#[derive(Default)]
struct Held {
    entries: Vec<Entry>,
    hidden: Vec<u64>,
    /// Bumped at each change (this thread's view is cached by it and GEN); begun at a value
    /// no other view has (`next_version`).
    version: u64,
    /// A scout's view (`hold_begin_scout`): the published entries as its record found them.
    frozen: Option<Arc<Frozen>>,
    /// `DS_SCOUT_TRACE`: its entries that ended (destination, words, the write that ended
    /// them: its start and bytes; 0, 0: replaced or dropped).
    gone: Vec<(u64, usize, u64, u64)>,
}

/// `DS_SCOUT_TRACE=<n>` (diagnosis): the first n CPU reads of scouts and their workers that
/// their view did not cover although copies were forwarded around them (`DEST_SPAN`), with
/// the view's entries on the read's page and the ones that ended there (`trace_uncovered`).
pub fn scout_trace() -> bool {
    static ON: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_SCOUT_TRACE").ok().and_then(|v| v.parse().ok()).unwrap_or(0)) != 0
}

/// Where copies of at most 4,096 words were forwarded so far: [lowest destination, highest
/// end) (the per-view blocks; `scout_trace`).
static DEST_SPAN: [AtomicU64; 2] = [AtomicU64::new(u64::MAX), AtomicU64::new(0)];

/// A scout's or its worker's CPU reads `reads` that its view left uncovered (`scout_trace`).
pub fn trace_uncovered(reads: &[(u64, u64)], who: &str, stages: [u64; 2]) {
    static N: AtomicU64 = AtomicU64::new(0);
    let (lo, hi) = (DEST_SPAN[0].load(Ordering::Relaxed), DEST_SPAN[1].load(Ordering::Relaxed));
    let Some(&(a, n)) = reads.iter().find(|&&(a, n)| a < hi && lo < a + n) else { return };
    // (`DS_SCOUT_TRACE_FROM=<flip>`: from that flip on.)
    static FROM: std::sync::OnceLock<(u64, u64)> = std::sync::OnceLock::new();
    let num = |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<u64>().ok()).unwrap_or(0);
    let (limit, from) = *FROM.get_or_init(|| (num("DS_SCOUT_TRACE"), num("DS_SCOUT_TRACE_FROM")));
    if super::FRAMES.load(Ordering::Relaxed) < from || N.fetch_add(1, Ordering::Relaxed) >= limit {
        return;
    }
    let (page, end) = (a & !0xFFF, (a + n).next_multiple_of(0x1000));
    let line = HELD.with(|h| match h.borrow().as_ref() {
        None => "no held view".to_string(),
        Some(x) => {
            let own: Vec<String> = x.entries.iter().filter(|e| e.base < end && page < e.base + 4 * e.words.len() as u64).map(|e| format!("{:#x}+{:#x}{}", e.base, 4 * e.words.len(), if e.how == HOW_CP_WRITE { " (CP write)" } else { "" })).collect();
            let gone: Vec<String> = x.gone.iter().filter(|g| g.0 < end && page < g.0 + 4 * g.1 as u64).map(|g| format!("{:#x}+{:#x} by {:#x}+{:#x}", g.0, 4 * g.1, g.2, g.3)).collect();
            let frozen: Vec<String> = x.frozen.as_ref().map_or_else(Vec::new, |f| f.entries.iter().filter(|e| e.1 < end && page < e.1 + 4 * e.2.len() as u64).map(|e| format!("{:#x}+{:#x}{}", e.1, 4 * e.2.len(), if x.hidden.contains(&e.0) { " (hidden)" } else { "" })).collect());
            format!("its entries on the page: [{}]; ended there: [{}]; published when it began: [{}]; {} entries in all, {} hidden", own.join(", "), gone.join(", "), frozen.join(", "), x.entries.len(), x.hidden.len())
        }
    });
    crate::info!("forward: scout trace: {who} read {a:#x}+{n:#x} from memory (stages {:#x} {:#x}; {} uncovered reads here): {line}", stages[0], stages[1], reads.len());
}

/// The published entries as a scout's record found them when it began (id, destination,
/// words). A scout records a later submission than the main command processor, which goes on
/// publishing copies of the submissions between: the scout reads through this snapshot for
/// its whole record (so every read of it sees one state); the words it takes from it are kept
/// with its other binding reads and checked by content at its turn (`Exec::note_content`,
/// `content_changed`). Its own entries (its copies and CP writes) come first. Both are served
/// as they are: memory under them moves with the earlier submissions' copies meanwhile, so
/// `before_copy` says nothing here.
struct Frozen {
    entries: Vec<(u64, u64, Arc<[u32]>)>,
}

/// How a thread's copy of an entry is served (`LocalEntry`'s last field): checked against
/// memory (`before_copy`), or as it is: a scout's own entry, or entry k of its snapshot
/// (`SERVE_FROZEN` + k).
const SERVE_CHECKED: u8 = 0;
const SERVE_OWN: u8 = 1;
const SERVE_FROZEN: u8 = 2;

/// A held view's first version: no two views of a thread share one (a view's cached copy is
/// found by its version: a new view that began at 0 again could find the last one's copy).
fn next_version() -> u64 {
    static V: AtomicU64 = AtomicU64::new(1 << 24);
    V.fetch_add(1 << 24, Ordering::Relaxed)
}

thread_local! {
    static HELD: RefCell<Option<Held>> = const { RefCell::new(None) };
}

/// Whether this thread records with a held view (`hold_begin`).
pub fn held() -> bool {
    HELD.with(|h| h.borrow().is_some())
}

/// A held stretch begins on this thread (the graphics CP past a run).
pub fn hold_begin() {
    HELD.with(|h| *h.borrow_mut() = Some(Held { version: next_version(), ..Default::default() }));
}

/// A scout's record begins on this thread (`Exec::scout_begin`): held, reading through the
/// published entries as they are now (`Frozen`).
pub fn hold_begin_scout() {
    let entries: Vec<(u64, u64, Arc<[u32]>)> = ENTRIES.read().iter().map(|e| (e.id, e.base, e.words.clone())).collect();
    HELD.with(|h| *h.borrow_mut() = Some(Held { version: next_version(), frozen: Some(Arc::new(Frozen { entries })), ..Default::default() }));
}

/// The scout's record on this thread is committed (`Cp::scout_commit`, in place of
/// `hold_commit`): `writes`, what its batches write, ends the published entries they overlap
/// (its writes come after them in program order; entries published since its record began
/// are not in its view, so its recording could not hide them), then its copies become
/// everyone's. Each one's old and earlier words are taken anew: what memory can hold there
/// before the copy runs is what it holds now and the words of the published copies this
/// record's writes end, not what the scout found when it recorded.
pub fn hold_commit_scout(writes: &[(u64, u64)]) {
    let Some(held) = HELD.with(|h| h.borrow_mut().take()) else { return };
    let over = |base: u64, len: usize| writes.iter().any(|&(a, n)| base < a + n && a < base + 4 * len as u64);
    let mut g = ENTRIES.write();
    let (gone, keep): (Vec<Entry>, Vec<Entry>) = std::mem::take(&mut *g).into_iter().partition(|e| held.hidden.contains(&e.id) || over(e.base, e.words.len()));
    *g = keep;
    for e in held.entries.into_iter().filter(|e| e.how != HOW_CP_WRITE && e.how != HOW_WALKED) {
        let n = e.words.len();
        let end = e.base + 4 * n as u64;
        let old: Arc<[u32]> = if super::cp::guest_readable(e.base, 4 * n as u64) {
            let mut v = vec![0u32; n];
            unsafe { read_words(e.base, &mut v) };
            v.into()
        } else {
            Arc::from(Vec::new())
        };
        let mut earlier: Vec<Arc<[u32]>> = Vec::new();
        for x in gone.iter().filter(|x| x.base == e.base && x.words.len() == n) {
            earlier.extend(x.earlier.iter().cloned());
            earlier.push(x.words.clone());
        }
        earlier.extend(e.earlier.iter().cloned());
        if earlier.len() > 2 * EARLIER_MAX {
            earlier.drain(..earlier.len() - 2 * EARLIER_MAX);
        }
        g.retain(|x| !(x.base < end && e.base < x.base + 4 * x.words.len() as u64));
        if g.len() >= MAX_ENTRIES {
            g.remove(0);
        }
        g.push(Entry { old, earlier: earlier.into(), ..e });
        HELD_STATS[1].fetch_add(1, Ordering::Relaxed);
    }
    COUNT.store(g.len(), Ordering::Release);
    note_span(&g);
    GEN.fetch_add(1, Ordering::AcqRel);
}

/// A held view off its thread (`park`, `unpark`): a scout's finished record's, until the
/// thread that commits or drops the record takes it.
pub struct Parked(Held);

/// Takes this thread's held view off it (None: it has none).
pub fn park() -> Option<Parked> {
    HELD.with(|h| h.borrow_mut().take()).map(Parked)
}

/// This thread reads and commits through `p` from now on (it had no held view).
pub fn unpark(p: Parked) {
    HELD.with(|h| *h.borrow_mut() = Some(p.0));
}

/// A scout's record refused at a gate after earlier segments of it were committed: `writes`
/// (what the record writes) ends the published entries they overlap, as its commit would
/// have; its own entries are forgotten with this thread's view (readers of its committed
/// copies wait for their batches).
pub fn hold_end_scout(writes: &[(u64, u64)]) {
    let Some(held) = HELD.with(|h| h.borrow_mut().take()) else { return };
    let over = |base: u64, len: usize| writes.iter().any(|&(a, n)| base < a + n && a < base + 4 * len as u64);
    let mut g = ENTRIES.write();
    let before = g.len();
    g.retain(|e| !(held.hidden.contains(&e.id) || over(e.base, e.words.len())));
    if g.len() != before {
        COUNT.store(g.len(), Ordering::Release);
        note_span(&g);
        GEN.fetch_add(1, Ordering::AcqRel);
    }
}

thread_local! {
    /// The span of this thread's held view's own entries, by the view's version
    /// (`uncovered_by_own`): (version, lowest destination, highest end).
    static OWN_SPAN: Cell<(u64, u64, u64)> = const { Cell::new((u64::MAX, 0, 0)) };
}

/// Appends to `out` the parts of [a, a + n) that this thread's held view's own entries do not
/// cover (a scout's: its copies and CP writes, before the reader in program order); without a
/// held view the whole range.
pub fn uncovered_by_own(a: u64, n: u64, out: &mut Vec<(u64, u64)>) {
    let end = a + n;
    HELD.with(|h| {
        let h = h.borrow();
        let Some(x) = h.as_ref() else {
            out.push((a, n));
            return;
        };
        let (v, lo, hi) = OWN_SPAN.with(|s| s.get());
        let (lo, hi) = if v == x.version {
            (lo, hi)
        } else {
            let (lo, hi) = x.entries.iter().fold((u64::MAX, 0u64), |(lo, hi), e| (lo.min(e.base), hi.max(e.base + 4 * e.words.len() as u64)));
            OWN_SPAN.with(|s| s.set((x.version, lo, hi)));
            (lo, hi)
        };
        if end <= lo || hi <= a {
            out.push((a, n));
            return;
        }
        let mut over: Vec<(u64, u64)> = x.entries.iter().filter(|e| e.base < end && a < e.base + 4 * e.words.len() as u64).map(|e| (a.max(e.base), end.min(e.base + 4 * e.words.len() as u64))).collect();
        if over.is_empty() {
            out.push((a, n));
            return;
        }
        over.sort_unstable();
        let mut at = a;
        for (s, e) in over {
            if s > at {
                out.push((at, s - at));
            }
            at = at.max(e);
        }
        if at < end {
            out.push((at, end - at));
        }
    });
}

/// The parts of `ranges` no published entry covers: what a reader in program order takes
/// from guest memory there (a scout's content check waits for their unfinished writers
/// first, `exec::words_now`).
pub fn uncovered_published(ranges: &[(u64, u64)]) -> Vec<(u64, u64)> {
    let mut out = Vec::new();
    if COUNT.load(Ordering::Acquire) == 0 {
        out.extend_from_slice(ranges);
        return out;
    }
    let g = ENTRIES.read();
    for &(a, n) in ranges {
        let end = a + n;
        let mut over: Vec<(u64, u64)> = g.iter().filter(|e| e.base < end && a < e.base + 4 * e.words.len() as u64).map(|e| (a.max(e.base), end.min(e.base + 4 * e.words.len() as u64))).collect();
        over.sort_unstable();
        let mut at = a;
        for (s, e) in over {
            if s > at {
                out.push((at, s - at));
            }
            at = at.max(e);
        }
        if at < end {
            out.push((at, end - at));
        }
    }
    out
}

/// What a reader in program order takes at [base, base + 4 n) now: the published entries'
/// words over guest memory (as `overlay` serves them); None where that is not known: the
/// range unreadable, or a part no entry covers still written by unfinished work (`busy`).
pub(super) fn reader_words(base: u64, n: usize, busy: &impl Fn(&[(u64, u64)]) -> bool) -> Option<Vec<u32>> {
    if !super::cp::guest_readable(base, 4 * n as u64) {
        return None;
    }
    let end = base + 4 * n as u64;
    // (The entries held still while memory is read: one that retires meanwhile would leave
    // its words neither here nor there.)
    let g = ENTRIES.read();
    let mut covered: Vec<(u64, u64)> = g.iter().filter(|e| e.base < end && base < e.base + 4 * e.words.len() as u64).map(|e| (base.max(e.base), end.min(e.base + 4 * e.words.len() as u64))).collect();
    covered.sort_unstable();
    let mut left: Vec<(u64, u64)> = Vec::new();
    let mut at = base;
    for &(s, e) in &covered {
        if s > at {
            left.push((at, s - at));
        }
        at = at.max(e);
    }
    if at < end {
        left.push((at, end - at));
    }
    if !left.is_empty() && busy(&left) {
        return None;
    }
    let mut out = vec![0u32; n];
    unsafe { read_words(base, &mut out) };
    for e in g.iter() {
        let (lo, hi) = (base.max(e.base), end.min(e.base + 4 * e.words.len() as u64));
        if lo >= hi {
            continue;
        }
        let ok = (lo..hi).step_by(4).all(|a| {
            let (i, k) = (((a - base) / 4) as usize, ((a - e.base) / 4) as usize);
            out[i] == e.words[k] || before_copy(&e.old, &e.earlier, k, out[i])
        });
        if !ok {
            // (A reader would end the entry and take memory, which unfinished work may write.)
            return None;
        }
        for a in (lo..hi).step_by(4) {
            out[((a - base) / 4) as usize] = e.words[((a - e.base) / 4) as usize];
        }
    }
    Some(out)
}

/// The held stretch is committed: its entries become global (submitted with
/// its batches next, `submitted`) and the global entries its writes ended go.
pub fn hold_commit() {
    if let Some(held) = HELD.with(|h| h.borrow_mut().take()) {
        publish(held);
    }
}

/// A held stretch's entries become global and the global entries its
/// writes ended go (`hold_commit`).
fn publish(held: Held) {
    if held.entries.is_empty() && held.hidden.is_empty() {
        return;
    }
    let mut g = ENTRIES.write();
    g.retain(|e| !held.hidden.contains(&e.id));
    // (A held CP write's words are its thread's alone, `add_cp_write`: once its batch is
    // submitted, readers find the write as any CP write.)
    for e in held.entries.into_iter().filter(|e| e.how != HOW_CP_WRITE && e.how != HOW_WALKED) {
        let end = e.base + 4 * e.words.len() as u64;
        g.retain(|x| !(x.base < end && e.base < x.base + 4 * x.words.len() as u64));
        if g.len() >= MAX_ENTRIES {
            g.remove(0);
        }
        g.push(e);
        HELD_STATS[1].fetch_add(1, Ordering::Relaxed);
    }
    COUNT.store(g.len(), Ordering::Release);
    note_span(&g);
    GEN.fetch_add(1, Ordering::AcqRel);
}

/// The held stretch is dropped (recorded again from where it began): what it
/// forwarded is forgotten, the global entries are as they were.
pub fn hold_drop() {
    if let Some(held) = HELD.with(|h| h.borrow_mut().take()) {
        HELD_STATS[2].fetch_add(held.entries.len() as u64, Ordering::Relaxed);
    }
}

/// [copies forwarded in held stretches, published at their commit, dropped].
pub static HELD_STATS: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// The calling thread records for the graphics queue (its command processor
/// or one of its parallel workers): its reads may take forwarded words.
pub fn enable_here() {
    HERE.with(|h| h.set(true));
}

/// Whether reads on this thread may take forwarded words now (cheap).
#[inline]
pub fn active() -> bool {
    (COUNT.load(Ordering::Acquire) != 0 || HELD.with(|h| h.borrow().as_ref().is_some_and(|x| !x.entries.is_empty() || x.frozen.as_ref().is_some_and(|f| !f.entries.is_empty())))) && HERE.with(|h| h.get())
}

type LocalEntry = (u64, u64, Arc<[u32]>, Arc<[u32]>, Arc<[Arc<[u32]>]>, u8);

/// Earlier copies' words an entry keeps (`Entry::earlier`).
const EARLIER_MAX: usize = 8;

/// Entries a write ended (`written`): destination, words, and their words
/// with their own earlier ones, for the next copy into that destination
/// (`add`).
static RECENT: Mutex<Vec<(u64, usize, Arc<[Arc<[u32]>]>)>> = Mutex::new(Vec::new());
const RECENT_MAX: usize = 16;

/// Whether guest memory's word `mem` at word `k` of an entry is one it can
/// hold before the entry's copy runs: the old word or an earlier copy's.
#[inline]
fn before_copy(old: &[u32], earlier: &[Arc<[u32]>], k: usize, mem: u32) -> bool {
    old.get(k) == Some(&mem) || earlier.iter().any(|e| e.get(k) == Some(&mem))
}

/// This thread's copy of the entries with the span they lie in (a read
/// outside it overlaps none: the entries are destinations within the
/// per-view block).
fn with_local_span<R>(f: impl FnOnce(&[LocalEntry], (u64, u64)) -> R) -> R {
    with_local_view(|v, span, _| f(v, span))
}

/// `with_local_span`, with a scout's snapshot (its entries are in the copy, served as they
/// are: `SERVE_FROZEN`).
fn with_local_view<R>(f: impl FnOnce(&[LocalEntry], (u64, u64), Option<&Frozen>) -> R) -> R {
    // A held view's own version is mixed in (its entries and hidden ones); a scout's view
    // does not follow the published entries at all.
    let (version, frozen) = HELD.with(|h| h.borrow().as_ref().map_or((0, false), |x| (x.version, x.frozen.is_some())));
    let generation = if frozen { version.wrapping_mul(0xC2B2_AE3D_27D4_EB4F) ^ 0x5C0F } else { GEN.load(Ordering::Acquire).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ version.wrapping_mul(0xC2B2_AE3D_27D4_EB4F) };
    LOCAL.with(|l| {
        let mut l = l.borrow_mut();
        if l.0 != generation {
            let local = |e: &Entry, serve: u8| -> LocalEntry { (e.base, 4 * e.words.len() as u64, e.words.clone(), e.old.clone(), e.earlier.clone(), serve) };
            let (v, fr): (Vec<LocalEntry>, Option<Arc<Frozen>>) = HELD.with(|h| match h.borrow().as_ref() {
                None => (ENTRIES.read().iter().map(|e| local(e, SERVE_CHECKED)).collect(), None),
                Some(x) => {
                    let under = |base: u64, len: usize| x.entries.iter().any(|p| p.base < base + 4 * len as u64 && base < p.base + 4 * p.words.len() as u64);
                    match x.frozen.as_ref() {
                        None => {
                            let g = ENTRIES.read();
                            (x.entries.iter().chain(g.iter().filter(|e| !x.hidden.contains(&e.id) && !under(e.base, e.words.len()))).map(|e| local(e, SERVE_CHECKED)).collect(), None)
                        }
                        Some(fr) => {
                            let none: Arc<[u32]> = Arc::from(Vec::new());
                            let none_earlier: Arc<[Arc<[u32]>]> = Arc::from(Vec::new());
                            let mut v: Vec<LocalEntry> = x.entries.iter().map(|e| local(e, SERVE_OWN)).collect();
                            v.extend(fr.entries.iter().enumerate().filter(|(_, e)| !x.hidden.contains(&e.0) && !under(e.1, e.2.len())).map(|(k, e)| (e.1, 4 * e.2.len() as u64, e.2.clone(), none.clone(), none_earlier.clone(), SERVE_FROZEN + k as u8)));
                            (v, Some(fr.clone()))
                        }
                    }
                }
            });
            l.1 = v;
            l.3 = fr;
            l.2 = l.1.iter().fold((u64::MAX, 0), |(lo, hi), e| (lo.min(e.0), hi.max(e.0 + e.1)));
            l.0 = generation;
        }
        f(&l.1, l.2, l.3.as_deref())
    })
}

/// The entry at `base` is stale: a reader found a destination word that is
/// neither the old nor the copied one (the title's CPU rewrote it).
fn drop_stale(base: u64) {
    let in_held = HELD.with(|h| {
        let mut h = h.borrow_mut();
        let Some(x) = h.as_mut() else { return false };
        let before = x.entries.len();
        x.entries.retain(|e| e.base != base);
        if x.entries.len() != before {
            x.version += 1;
            true
        } else {
            false
        }
    });
    if in_held {
        STATS[8].fetch_add(1, Ordering::Relaxed);
        return;
    }
    let mut g = ENTRIES.write();
    let before = g.len();
    if trace_on(base, 4) {
        trace_life(format_args!("entry at {base:#x} dropped as stale (a word neither old nor copied)"));
    }
    g.retain(|e| e.base != base);
    if g.len() != before {
        STATS[8].fetch_add(1, Ordering::Relaxed);
        COUNT.store(g.len(), Ordering::Release);
        note_span(&g);
        GEN.fetch_add(1, Ordering::AcqRel);
    }
}

/// The word a reader takes at `addr` (4-byte aligned) given `mem`, what
/// guest memory holds there: the copied word while memory still holds the
/// destination's old word (the copy has not run) or that word itself.
#[inline]
pub fn word_or(addr: u64, mem: u32) -> u32 {
    if !active() {
        return mem;
    }
    let hit = with_local_span(|v, (lo, hi)| {
        if addr < lo || hi <= addr {
            return None;
        }
        v.iter().find(|e| e.0 <= addr && addr < e.0 + e.1).map(|e| {
            let k = ((addr - e.0) / 4) as usize;
            (e.0, e.2[k], e.5 != SERVE_CHECKED || before_copy(&e.3, &e.4, k, mem))
        })
    });
    match hit {
        Some((_, w, known)) if mem == w || known => {
            STATS[3].fetch_add(1, Ordering::Relaxed);
            w
        }
        Some((base, ..)) => {
            drop_stale(base);
            mem
        }
        None => mem,
    }
}

/// Replaces the words of `out` (read at `addr` from guest memory) that
/// entries cover, entry by entry: all of an entry's words in range, or none
/// when one of them shows the title rewrote the destination (the entry ends).
pub fn overlay(addr: u64, out: &mut [u32]) {
    if !active() {
        return;
    }
    let end = addr + 4 * out.len() as u64;
    let mut stale = Vec::new();
    let n = with_local_span(|v, (lo, hi)| {
        if end <= lo || hi <= addr {
            return 0;
        }
        let mut n = 0u64;
        for (base, bytes, words, old, earlier, serve) in v {
            let (lo, hi) = (addr.max(*base), end.min(base + bytes));
            if lo >= hi {
                continue;
            }
            let ok = *serve != SERVE_CHECKED
                || (lo..hi).step_by(4).all(|a| {
                    let (i, k) = (((a - addr) / 4) as usize, ((a - base) / 4) as usize);
                    out[i] == words[k] || before_copy(old, earlier, k, out[i])
                });
            if !ok {
                stale.push(*base);
                continue;
            }
            for a in (lo..hi).step_by(4) {
                out[((a - addr) / 4) as usize] = words[((a - base) / 4) as usize];
                n += 1;
            }
        }
        n
    });
    for base in stale {
        drop_stale(base);
    }
    if n > 0 {
        STATS[3].fetch_add(n, Ordering::Relaxed);
    }
}

/// `reads` without the parts entries cover (the reader took the forwarded
/// words there, so GPU work writing them need not complete first). None:
/// nothing covered (use `reads` as they are).
pub fn uncovered(reads: &[(u64, u64)]) -> Option<Vec<(u64, u64)>> {
    if !active() {
        return None;
    }
    // Reads outside every entry need not make this thread's copy. (A span
    // read while it changes can only miss a cover: the reads are then
    // waited for as without forwarding.)
    let (lo, hi) = (SPAN[0].load(Ordering::Acquire), SPAN[1].load(Ordering::Acquire));
    // (A scout's view does not follow the published entries: its own span decides, below.)
    let held = HELD.with(|h| h.borrow().as_ref().is_some_and(|x| x.frozen.is_some() || x.entries.iter().any(|e| reads.iter().any(|&(a, n)| e.base < a + n && a < e.base + 4 * e.words.len() as u64))));
    if !held && !reads.iter().any(|&(a, n)| a < hi && lo < a + n) {
        return None;
    }
    with_local_span(|v, (lo, hi)| {
        if !reads.iter().any(|&(a, n)| a < hi && lo < a + n && v.iter().any(|e| e.0 < a + n && a < e.0 + e.1)) {
            return None;
        }
        let mut out = Vec::with_capacity(reads.len());
        let mut covered = 0u64;
        // The entries over one read, clipped to it (one buffer for all: this
        // allocated a vector per read and entry, ~1.6% of the graphics
        // command processor's time at the vista, 2026-10-28).
        let mut over: Vec<(u64, u64)> = Vec::new();
        for &(a, n) in reads {
            let end = a + n;
            over.clear();
            over.extend(v.iter().filter(|e| e.0 < end && a < e.0 + e.1).map(|e| (e.0.max(a), (e.0 + e.1).min(end))));
            if over.is_empty() {
                out.push((a, n));
                continue;
            }
            // Pieces of [a, end) outside every entry, in order.
            over.sort_unstable();
            let (mut at, before) = (a, out.len());
            for &(s, e) in &over {
                if s > at {
                    out.push((at, s - at));
                }
                at = at.max(e);
            }
            if at < end {
                out.push((at, end - at));
            }
            if out.len() == before {
                covered += 1;
            }
        }
        STATS[4].fetch_add(covered, Ordering::Relaxed);
        Some(out)
    })
}

/// GPU work or a CP write recorded later in the stream writes [a, a + n):
/// entries it overlaps end.
#[track_caller]
pub fn written(a: u64, n: u64) {
    if n != 0 && CHECK_COUNT.load(Ordering::Acquire) != 0 && (n > 1024 << 16 || check_pages(a, n).any(|slot| slot.load(Ordering::Acquire) != 0)) {
        let mut g = CHECKS.lock();
        let before = g.len();
        g.retain(|c| {
            let keep = !(c.base < a + n && a < c.base + 4 * c.words.len() as u64);
            if !keep {
                check_untrack(c);
            }
            keep
        });
        CHECK_STATS[5].fetch_add((before - g.len()) as u64, Ordering::Relaxed);
        CHECK_COUNT.store(g.len(), Ordering::Release);
    }
    if n == 0 {
        return;
    }
    // A held view: its own entries end; global ones are hidden from it.
    let handled = HELD.with(|h| {
        let mut h = h.borrow_mut();
        let Some(x) = h.as_mut() else { return false };
        let hit = |b: u64, len: usize| b < a + n && a < b + 4 * len as u64;
        let before = x.entries.len();
        if x.frozen.is_some() && scout_trace() && x.gone.len() < 512 {
            let ended: Vec<(u64, usize, u64, u64)> = x.entries.iter().filter(|e| hit(e.base, e.words.len())).map(|e| (e.base, e.words.len(), a, n)).collect();
            x.gone.extend(ended);
        }
        x.entries.retain(|e| !hit(e.base, e.words.len()));
        let mut changed = x.entries.len() != before;
        if let Some(fr) = x.frozen.clone() {
            // (A scout: the entries its record found are hidden from it; the published ones
            // its writes end are found at its commit, `hold_commit_scout`.)
            for e in fr.entries.iter().filter(|e| hit(e.1, e.2.len())) {
                if !x.hidden.contains(&e.0) {
                    x.hidden.push(e.0);
                    changed = true;
                }
            }
        } else if COUNT.load(Ordering::Acquire) != 0 {
            for e in ENTRIES.read().iter().filter(|e| hit(e.base, e.words.len())) {
                if !x.hidden.contains(&e.id) {
                    x.hidden.push(e.id);
                    changed = true;
                }
            }
        }
        if changed {
            x.version += 1;
            STATS[1].fetch_add(1, Ordering::Relaxed);
        }
        true
    });
    if handled || COUNT.load(Ordering::Acquire) == 0 {
        return;
    }
    let mut g = ENTRIES.write();
    let before = g.len();
    if trace_on(a, n) {
        for e in g.iter().filter(|e| e.base < a + n && a < e.base + 4 * e.words.len() as u64) {
            trace_end(e.base, 4 * e.words.len() as u64, e.seq, a, n);
        }
    }
    if super::tune::forward_earlier() {
        // The next copy into the same destination takes them as words memory
        // may hold before it runs (usually this write is that copy's own).
        let mut r = RECENT.lock();
        for e in g.iter().filter(|e| e.base < a + n && a < e.base + 4 * e.words.len() as u64) {
            r.retain(|x| x.0 != e.base);
            let mut v: Vec<Arc<[u32]>> = e.earlier.iter().cloned().collect();
            v.push(e.words.clone());
            if v.len() > EARLIER_MAX {
                v.drain(..v.len() - EARLIER_MAX);
            }
            r.push((e.base, e.words.len(), v.into()));
            if r.len() > RECENT_MAX {
                r.remove(0);
            }
        }
    }
    g.retain(|e| !(e.base < a + n && a < e.base + 4 * e.words.len() as u64));
    let ended = before - g.len();
    if ended > 0 {
        STATS[1].fetch_add(ended as u64, Ordering::Relaxed);
        COUNT.store(g.len(), Ordering::Release);
        note_span(&g);
        GEN.fetch_add(1, Ordering::AcqRel);
    }
}

/// `DS_TRACE_FORWARD` (a destination range) from `DS_TRACE_NOTE_FROM` (a
/// flip) on: whether a write over [a, a+n) is traced.
fn trace_on(a: u64, n: u64) -> bool {
    static AT: std::sync::OnceLock<(u64, u64, u64)> = std::sync::OnceLock::new();
    let (lo, hi, from) = *AT.get_or_init(|| {
        let v = std::env::var("DS_TRACE_FORWARD").unwrap_or_default();
        let num = |x: &str| u64::from_str_radix(x.trim().trim_start_matches("0x"), 16).ok();
        let (lo, hi) = match v.split_once('-') {
            Some((x, y)) => (num(x).unwrap_or(0), num(y).unwrap_or(0)),
            None => num(&v).map_or((0, 0), |x| (x, x + 1)),
        };
        let from = std::env::var("DS_TRACE_NOTE_FROM").ok();
        (lo, hi, if from.as_deref() == Some("sync") { u64::MAX } else { from.and_then(|v| v.parse().ok()).unwrap_or(0) })
    });
    // `DS_TRACE_NOTE_FROM=sync`: while `tune trace_sync` traces waits.
    let now = if from == u64::MAX { super::tune::get_trace_sync() } else { super::FRAMES.load(Ordering::Relaxed) >= from };
    lo != 0 && a < hi && lo < a.saturating_add(n) && now
}

/// An entry a later write ended (`trace_on`), with the caller of `written`
/// (who wrote).
#[track_caller]
fn trace_end(base: u64, bytes: u64, seq: u64, a: u64, n: u64) {
    let l = std::panic::Location::caller();
    trace_life(format_args!("entry {base:#x}+{bytes:#x} (batch {seq}) ended by a write of {a:#x}+{n:#x} (at {}:{})", l.file(), l.line()));
}

/// The first 300 traced entry events (`trace_on`), with the thread and flip.
fn trace_life(what: std::fmt::Arguments) {
    static N: AtomicU64 = AtomicU64::new(0);
    if N.fetch_add(1, Ordering::Relaxed) < 300 {
        crate::info!("forward: {what} (flip {})", super::FRAMES.load(Ordering::Relaxed));
    }
}

/// The copy kernel's roles, if the program is it (cached by code hash).
/// `p` is the copy kernel (profiling: its dispatch steps apart).
pub fn is_copy_kernel(p: &Prepared) -> bool {
    roles(p).is_some()
}

fn roles(p: &Prepared) -> Option<Roles> {
    let insts = &p.found.insts;
    let key = p.code_hash;
    let known = PROGRAMS.lock().get_or_insert_with(HashMap::new).get(&key).copied();
    if let Some(r) = known {
        if r.is_some() {
            note_copy_addr(p.addr);
        }
        return r;
    }
    let r = recognize(insts, &p.found.tr.plan);
    if r.is_some() {
        note_copy_addr(p.addr);
        COPY_HASH.store(key, Ordering::Release);
        crate::info!("gpu: cs {:#x} (code hash {key:#018x}) is a guarded modulo copy: its results are forwarded to later reads (`tune forward_copies`)", p.addr);
    }
    PROGRAMS.lock().get_or_insert_with(HashMap::new).insert(key, r);
    r
}

/// Addresses of the programs recognized as the copy kernel (0: free): what a walk ahead of
/// the recording looks for (`Cp::walk_copy`).
static COPY_ADDRS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

fn note_copy_addr(addr: u64) {
    for c in &COPY_ADDRS {
        match c.compare_exchange(0, addr, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => return,
            Err(a) if a == addr => return,
            Err(_) => {}
        }
    }
}

/// Whether the compute program at `addr` was recognized as the copy kernel.
#[inline]
pub fn is_copy_addr(addr: u64) -> bool {
    addr != 0 && COPY_ADDRS.iter().any(|c| c.load(Ordering::Relaxed) == addr)
}

/// `if (i < cb[0]) dst[i] = src[i % cb[1]]`, as the title's compiler emits
/// it for 64-lane groups (the unsigned remainder through a reciprocal),
/// with nothing else touching memory.
fn recognize(insts: &[rdna::Inst], plan: &rdna::analysis::Plan) -> Option<Roles> {
    const SEQUENCE: [&str; 26] = [
        "v_lshl_add_u32",
        "s_buffer_load_dword",
        "v_cmpx_gt_u32",
        "s_cbranch_execz",
        "s_buffer_load_dword",
        "v_cvt_f32_u32",
        "s_sub_i32",
        "v_rcp_iflag_f32",
        "v_mul_f32",
        "v_cvt_u32_f32",
        "v_readfirstlane_b32",
        "s_mul_i32",
        "s_mul_hi_u32",
        "v_add_u32",
        "v_mul_hi_u32",
        "v_mul_lo_u32",
        "v_sub_u32",
        "v_cmp_le_u32",
        "v_subrev_u32",
        "v_cndmask_b32",
        "v_cmp_le_u32",
        "v_subrev_u32",
        "v_cndmask_b32",
        "buffer_load_format_x",
        "buffer_store_format_x",
        "s_endpgm",
    ];
    let body: Vec<&rdna::Inst> = insts.iter().filter(|i| !matches!(i.name(), Some("s_inst_prefetch" | "s_waitcnt" | "s_nop"))).collect();
    if body.len() != SEQUENCE.len() || body.iter().zip(SEQUENCE).any(|(i, s)| i.name() != Some(s)) {
        return None;
    }
    // The control block's two words at offsets 0 and 4 of one descriptor.
    let (l0, l1) = (body[1], body[4]);
    if l0.smem_sbase() != l1.smem_sbase() || l0.smem_offset() != 0 || l1.smem_offset() != 4 {
        return None;
    }
    let cb = plan.inst_res.get(&l0.pc)?.0;
    let src = plan.inst_res.get(&body[23].pc)?.0;
    let dst = plan.inst_res.get(&body[24].pc)?.0;
    (cb != src && cb != dst && src != dst && plan.resources.get(dst)?.written && !plan.resources.get(src)?.written).then_some(Roles { src, dst, cb })
}

/// Formats a copy moves unchanged: 32_UINT, 32_SINT, 32_FLOAT.
fn plain32(format: u8) -> bool {
    matches!(format, 20..=22)
}

/// Reads guest words [a, a + 4 * out.len()) (4-byte aligned, readable) as
/// the title may be writing them: volatile loads of 16 bytes where they fit
/// (each 4-byte word whole, as a word-at-a-time read would see it), the
/// rest a word at a time.
unsafe fn read_words(a: u64, out: &mut [u32]) {
    let mut k = 0usize;
    // (To a 16-byte boundary first: aligned vector loads.)
    while k < out.len() && (a + 4 * k as u64) % 16 != 0 {
        out[k] = unsafe { std::ptr::read_volatile((a + 4 * k as u64) as *const u32) };
        k += 1;
    }
    while k + 4 <= out.len() {
        let q = unsafe { std::ptr::read_volatile((a + 4 * k as u64) as *const [u32; 4]) };
        out[k..k + 4].copy_from_slice(&q);
        k += 4;
    }
    while k < out.len() {
        out[k] = unsafe { std::ptr::read_volatile((a + 4 * k as u64) as *const u32) };
        k += 1;
    }
}

/// The words a dispatch of `p` with `groups` (64-lane groups along X) will
/// leave at its destination, when `p` is the copy kernel and its control
/// block and source can be read now (`busy`: GPU work or a deferred CP write
/// still writes one of those ranges). (destination base, words).
pub fn compute(p: &Prepared, groups: [u32; 3], local: [u32; 3], busy: impl Fn(&[(u64, u64)]) -> bool) -> Option<(u64, Vec<u32>, Inputs)> {
    compute_in(p, groups, local, busy, false)
}

/// `compute`; with `view`, the control block and the source are read through this thread's
/// forwarded words (`overlay`: a scout's held CP writes and copies precede the dispatch in
/// program order, `add_cp_write`), and only their parts no entry covers must be free of
/// unfinished writes.
pub fn compute_in(p: &Prepared, groups: [u32; 3], local: [u32; 3], busy: impl Fn(&[(u64, u64)]) -> bool, view: bool) -> Option<(u64, Vec<u32>, Inputs)> {
    let r = roles(p)?;
    let buf = |i: usize| match p.found.resources.get(i) {
        Some(Resolved::Buffer(v)) => Some(*v),
        _ => None,
    };
    // (With a view, the scouts' census counts why a copy is not forwarded: `VIEW_REFUSALS`.)
    let why = |k: usize| {
        if view {
            VIEW_REFUSALS[k].fetch_add(1, Ordering::Relaxed);
        }
    };
    why(0);
    let (src, dst, cb) = (buf(r.src)?, buf(r.dst)?, buf(r.cb)?);
    if DISTRUSTED.lock().as_ref().is_some_and(|d| d.contains(&(p.code_hash, dst.base))) {
        why(1);
        return None;
    }
    let refuse = || {
        STATS[7].fetch_add(1, Ordering::Relaxed);
        None
    };
    if local != [64, 1, 1] || groups[1] != 1 || groups[2] != 1 || !plain32(src.format) || !plain32(dst.format) || dst.stride != 4 || src.stride < 4 || src.add_tid || dst.add_tid {
        why(2);
        return refuse();
    }
    if !super::cp::guest_readable(cb.base, 8) {
        why(3);
        return refuse();
    }
    let mut nm = [unsafe { std::ptr::read_volatile(cb.base as *const u32) }, unsafe { std::ptr::read_volatile((cb.base + 4) as *const u32) }];
    if view {
        overlay(cb.base, &mut nm);
    }
    let (n, m) = (nm[0] as u64, nm[1] as u64);
    let count = n.min(groups[0] as u64 * 64).min(dst.num_records as u64);
    let in_src = m.min(src.num_records as u64);
    if m == 0 || count == 0 || count > 1 << 16 || in_src > 1 << 16 {
        why(4);
        if view {
            static N: AtomicU64 = AtomicU64::new(0);
            if N.fetch_add(1, Ordering::Relaxed) < 12 {
                crate::info!("forward: a scout's copy not computed for its counts: n {n:#x}, m {m:#x}, groups {groups:?}, destination {:#x} ({} records, stride {}), source {:#x} ({} records)", dst.base, dst.num_records, dst.stride, src.base, src.num_records);
            }
        }
        return refuse();
    }
    let src_bytes = (in_src - 1) * src.stride as u64 + 4;
    let inputs = [(cb.base, 8), (src.base, src_bytes)];
    let left = if view { uncovered(&inputs) } else { None };
    if busy(left.as_deref().unwrap_or(&inputs)) || super::homes::any_stale(&inputs) || !super::cp::guest_readable(src.base, src_bytes) {
        why(if busy(&[inputs[0]]) { 5 } else { 6 });
        trace_refusal(dst.base, 4 * count, &inputs, &busy);
        return refuse();
    }
    why(7);
    let src_words: Vec<u32> = if src.stride == 4 {
        let mut v = vec![0u32; in_src as usize];
        unsafe { read_words(src.base, &mut v) };
        if view {
            overlay(src.base, &mut v);
        }
        v
    } else {
        (0..in_src)
            .map(|j| {
                let a = src.base + j * src.stride as u64;
                let w = unsafe { std::ptr::read_volatile(a as *const u32) };
                if view { word_or(a, w) } else { w }
            })
            .collect()
    };
    // dst[i] = src[i % m], zero past the source's records: the source's
    // words a period at a time (a division per word was most of this).
    let mut words: Vec<u32> = Vec::with_capacity(count as usize);
    while (words.len() as u64) < count {
        let k = (count - words.len() as u64).min(m) as usize;
        let have = k.min(src_words.len());
        words.extend_from_slice(&src_words[..have]);
        words.resize(words.len() + (k - have), 0);
    }
    Some((dst.base, words, Inputs { src: src.base, stride: src.stride as u64, src_words, cb: cb.base, count: n, modulus: m }))
}

/// `DS_TRACE_FORWARD=<hex address>[-<hex end>]`: the first 60 copies into
/// the range that could not be forwarded, with which input was still
/// written (diagnosis: what a reader of the copy then waits for).
fn trace_refusal(dst: u64, bytes: u64, inputs: &[(u64, u64); 2], busy: &impl Fn(&[(u64, u64)]) -> bool) {
    static AT: std::sync::OnceLock<(u64, u64)> = std::sync::OnceLock::new();
    let (lo, hi) = *AT.get_or_init(|| {
        let v = std::env::var("DS_TRACE_FORWARD").unwrap_or_default();
        let num = |x: &str| u64::from_str_radix(x.trim().trim_start_matches("0x"), 16).ok();
        match v.split_once('-') {
            Some((a, b)) => (num(a).unwrap_or(0), num(b).unwrap_or(0)),
            None => num(&v).map_or((0, 0), |x| (x, x + 1)),
        }
    });
    if lo == 0 || dst >= hi || dst + bytes <= lo {
        return;
    }
    // From `DS_TRACE_NOTE_FROM` (a flip) on, as the write-note trace.
    static FROM: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    if super::FRAMES.load(Ordering::Relaxed) < *FROM.get_or_init(|| std::env::var("DS_TRACE_NOTE_FROM").ok().and_then(|v| v.parse().ok()).unwrap_or(0)) {
        return;
    }
    static N: AtomicU64 = AtomicU64::new(0);
    if N.fetch_add(1, Ordering::Relaxed) >= 60 {
        return;
    }
    let what = |r: (u64, u64)| {
        let mut v = Vec::new();
        if busy(&[r]) {
            v.push("written by GPU work or a CP write not done");
        }
        if super::homes::any_stale(&[r]) {
            v.push("a home's newer data");
        }
        if v.is_empty() { "free".to_string() } else { v.join(", ") }
    };
    crate::info!(
        "forward: copy into {dst:#x}+{bytes:#x} not forwarded: counts {:#x}+8 {}, source {:#x}+{:#x} {} (flip {})",
        inputs[0].0,
        what(inputs[0]),
        inputs[1].0,
        inputs[1].1,
        what(inputs[1]),
        super::FRAMES.load(Ordering::Relaxed)
    );
}

/// Starts forwarding `words` at `base` (written by the dispatch of `program`
/// just recorded, after its own writes were noted); returns the entry's id.
/// Copies forwarded by the engine allocator owning their destination's MiB
/// (`engine::owner`; 0: none known), with their words (status line): which
/// allocations hold the per-view blocks (front end 4's targets).
static BY_OWNER: [[AtomicU64; 2]; 256] = [const { [const { AtomicU64::new(0) }; 2] }; 256];

pub fn add(base: u64, words: Vec<u32>, program: u64, hash: u64, from: Inputs, how: &'static str, queue: &str) -> u64 {
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    let o = &BY_OWNER[crate::engine::owner(base) as usize];
    o[0].fetch_add(1, Ordering::Relaxed);
    o[1].fetch_add(words.len() as u64, Ordering::Relaxed);
    let n = words.len() as u64;
    if n <= 4096 {
        DEST_SPAN[0].fetch_min(base, Ordering::Relaxed);
        DEST_SPAN[1].fetch_max(base + 4 * n, Ordering::Relaxed);
    }
    let old: Arc<[u32]> = if super::cp::guest_readable(base, 4 * n) {
        let mut v = vec![0u32; n as usize];
        unsafe { read_words(base, &mut v) };
        v.into()
    } else {
        Arc::from(Vec::new())
    };
    if held() {
        return add_held(id, base, words, program, hash, from, how, queue, old);
    }
    let mut g = ENTRIES.write();
    let end = base + 4 * words.len() as u64;
    // An entry this copy replaces (same destination and length, not retired:
    // the GPU may not have run its copy yet): its words and its own earlier
    // ones remain what memory can hold before this copy runs.
    let earlier: Arc<[Arc<[u32]>]> = match g.iter().find(|e| e.base == base && e.words.len() as u64 == n) {
        Some(e) if super::tune::forward_earlier() => {
            let mut v: Vec<Arc<[u32]>> = e.earlier.iter().cloned().collect();
            v.push(e.words.clone());
            if v.len() > EARLIER_MAX {
                v.drain(..v.len() - EARLIER_MAX);
            }
            v.into()
        }
        // One this copy's own write ended a moment ago (`written`).
        _ if super::tune::forward_earlier() => {
            let mut r = RECENT.lock();
            match r.iter().position(|x| x.0 == base && x.1 as u64 == n) {
                Some(i) => r.remove(i).2,
                None => Arc::from(Vec::new()),
            }
        }
        _ => Arc::from(Vec::new()),
    };
    g.retain(|e| !(e.base < end && base < e.base + 4 * e.words.len() as u64));
    if g.len() >= MAX_ENTRIES {
        g.remove(0);
    }
    if trace_on(base, 4 * n) {
        trace_life(format_args!("entry {id} {base:#x}+{:#x} made by {queue}", 4 * n));
    }
    g.push(Entry { id, base, words: words.into(), seq: 0, program, hash, from, old, earlier, how, queue: queue.into(), write_backs: super::homes::WRITE_BACKS.load(Ordering::Relaxed), flip: super::FRAMES.load(Ordering::Relaxed) });
    COUNT.store(g.len(), Ordering::Release);
    note_span(&g);
    GEN.fetch_add(1, Ordering::AcqRel);
    STATS[0].fetch_add(1, Ordering::Relaxed);
    id
}

/// Copies computed through a view (`compute_in`: scouts): [copy kernel dispatches, destination
/// distrusted, odd shapes, control block unreadable, counts zero or large, control block still
/// written, source still written or stale, computed].
pub static VIEW_REFUSALS: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];

/// Entries a held view keeps at most (a scout's holds the CP writes and copies of its own
/// submission and of the ones walked before it).
const HELD_MAX: usize = 192;

/// Room for one more entry in a held view: past `MAX_ENTRIES` the oldest CP write's words go
/// first (labels mostly), at `HELD_MAX` the oldest entry.
fn held_room(entries: &mut Vec<Entry>) {
    if entries.len() < MAX_ENTRIES {
        return;
    }
    if let Some(k) = entries.iter().position(|e| e.how == HOW_CP_WRITE) {
        if entries.len() >= HELD_MAX || entries.iter().filter(|e| e.how == HOW_CP_WRITE).count() >= HELD_MAX - MAX_ENTRIES {
            entries.remove(k);
        }
    } else if entries.len() >= HELD_MAX {
        entries.remove(0);
    }
}

/// `Entry::how` of a held CP write's words (`add_cp_write`).
const HOW_CP_WRITE: &str = "a held CP write";

/// A CP write a scout's executor recorded into a held batch (`Exec::gpu_cp_write`): the GPU
/// performs it there, so memory holds the old words until the record is committed and run, but
/// the words are known. They are served to this thread's later reads as a copy's are (the
/// write precedes them in program order): stage preparations, and the copy kernel's control
/// block and source (`compute_in`), which else count as written by unfinished work. The entry is
/// the thread's own (`publish` leaves it out) and ends with the next write over it (`written`).
pub fn add_cp_write(base: u64, words: &[u32]) {
    if words.is_empty() || base % 4 != 0 || !held() || !HERE.with(|h| h.get()) {
        return;
    }
    let n = words.len() as u64;
    let end = base + 4 * n;
    let old: Arc<[u32]> = if super::cp::guest_readable(base, 4 * n) {
        let mut v = vec![0u32; n as usize];
        unsafe { read_words(base, &mut v) };
        v.into()
    } else {
        Arc::from(Vec::new())
    };
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    HELD.with(|h| {
        let mut h = h.borrow_mut();
        let Some(x) = h.as_mut() else { return };
        x.entries.retain(|e| !(e.base < end && base < e.base + 4 * e.words.len() as u64));
        held_room(&mut x.entries);
        x.entries.push(Entry {
            id,
            base,
            words: words.to_vec().into(),
            seq: 0,
            program: 0,
            hash: 0,
            from: Inputs { src: 0, stride: 0, src_words: Vec::new(), cb: 0, count: 0, modulus: 0 },
            old,
            earlier: Arc::from(Vec::new()),
            how: HOW_CP_WRITE,
            queue: "".into(),
            write_backs: 0,
            flip: super::FRAMES.load(Ordering::Relaxed),
        });
        x.version += 1;
    });
    CP_WRITE_ENTRIES.fetch_add(1, Ordering::Relaxed);
}

/// Held CP writes forwarded (`add_cp_write`).
pub static CP_WRITE_ENTRIES: AtomicU64 = AtomicU64::new(0);

/// `Entry::how` of a copy an earlier queued submission will make, computed by a walk ahead of
/// its recording (`add_walked`).
const HOW_WALKED: &str = "walked ahead";

/// A copy an earlier queued submission of the graphics queue will make, computed by a scout's
/// walk of that submission (`Cp::walk_copy`; its control block and source are the title's CPU
/// data or CP writes the walk forwarded): an entry of this thread's held view, so that the
/// scout reads the per-view block as the submissions before its own leave it. Never published
/// (`publish`); checked against what the real copy forwarded when the scout's turn comes
/// (`walked`, `global_words`).
pub fn add_walked(base: u64, words: Vec<u32>, program: u64, hash: u64, from: Inputs) -> u64 {
    WALKED_ENTRIES.fetch_add(1, Ordering::Relaxed);
    add(base, words, program, hash, from, HOW_WALKED, "walk")
}

/// Copies computed by walks ahead (`add_walked`).
pub static WALKED_ENTRIES: AtomicU64 = AtomicU64::new(0);

/// The walked copies this thread's held view holds now (`add_walked`): destination and words.
pub fn walked() -> Vec<(u64, Arc<[u32]>)> {
    HELD.with(|h| h.borrow().as_ref().map_or_else(Vec::new, |x| x.entries.iter().filter(|e| e.how == HOW_WALKED).map(|e| (e.base, e.words.clone())).collect()))
}

/// What a reader without a held view takes at [base, base + 4 n) now: guest memory with the
/// published entries' words over it (as `overlay` gives them). None: unreadable.
pub fn global_words(base: u64, n: usize) -> Option<Vec<u32>> {
    if !super::cp::guest_readable(base, 4 * n as u64) {
        return None;
    }
    let mut out = vec![0u32; n];
    unsafe { read_words(base, &mut out) };
    let end = base + 4 * n as u64;
    if COUNT.load(Ordering::Acquire) != 0 {
        for e in ENTRIES.read().iter() {
            let (lo, hi) = (base.max(e.base), end.min(e.base + 4 * e.words.len() as u64));
            if lo >= hi {
                continue;
            }
            let ok = (lo..hi).step_by(4).all(|a| {
                let (i, k) = (((a - base) / 4) as usize, ((a - e.base) / 4) as usize);
                out[i] == e.words[k] || before_copy(&e.old, &e.earlier, k, out[i])
            });
            if ok {
                for a in (lo..hi).step_by(4) {
                    out[((a - base) / 4) as usize] = e.words[((a - e.base) / 4) as usize];
                }
            }
        }
    }
    Some(out)
}

/// A held view's entries for another thread (`snapshot`, `adopt`).
pub struct Snapshot {
    entries: Vec<Entry>,
    hidden: Vec<u64>,
    frozen: Option<Arc<Frozen>>,
    gone: Vec<(u64, usize, u64, u64)>,
}

/// This thread's held view as it is now (None: it has none): what a scout's held run hands
/// its workers (`par::Run::held_fwd`), whose chunks read the per-view block as the scout does.
pub fn snapshot() -> Option<Arc<Snapshot>> {
    HELD.with(|h| h.borrow().as_ref().map(|x| Arc::new(Snapshot { entries: x.entries.clone(), hidden: x.hidden.clone(), frozen: x.frozen.clone(), gone: x.gone.clone() })))
}

/// This thread reads through a copy of `s` until `hold_drop` (a scout's worker for a chunk).
pub fn adopt(s: &Snapshot) {
    HELD.with(|h| *h.borrow_mut() = Some(Held { entries: s.entries.clone(), hidden: s.hidden.clone(), version: next_version(), frozen: s.frozen.clone(), gone: s.gone.clone() }));
}

/// `add` on a thread with a held view: the entry is its own; its earlier
/// words come from the entry it replaces there, else from the global one
/// over the destination (hidden from it or not), else from `RECENT`.
#[allow(clippy::too_many_arguments)]
fn add_held(id: u64, base: u64, words: Vec<u32>, program: u64, hash: u64, from: Inputs, how: &'static str, queue: &str, old: Arc<[u32]>) -> u64 {
    let n = words.len() as u64;
    let end = base + 4 * n;
    let chain = |e: &Entry| -> Arc<[Arc<[u32]>]> {
        let mut v: Vec<Arc<[u32]>> = e.earlier.iter().cloned().collect();
        v.push(e.words.clone());
        if v.len() > EARLIER_MAX {
            v.drain(..v.len() - EARLIER_MAX);
        }
        v.into()
    };
    HELD.with(|h| {
        let mut h = h.borrow_mut();
        let x = h.as_mut().expect("held view");
        let earlier: Arc<[Arc<[u32]>]> = if !super::tune::forward_earlier() {
            Arc::from(Vec::new())
        } else if let Some(e) = x.entries.iter().find(|e| e.base == base && e.words.len() as u64 == n) {
            chain(e)
        } else if let Some(e) = ENTRIES.read().iter().find(|e| e.base == base && e.words.len() as u64 == n) {
            chain(e)
        } else {
            let mut r = RECENT.lock();
            match r.iter().position(|y| y.0 == base && y.1 as u64 == n) {
                Some(i) => r.remove(i).2,
                None => Arc::from(Vec::new()),
            }
        };
        x.entries.retain(|e| !(e.base < end && base < e.base + 4 * e.words.len() as u64));
        held_room(&mut x.entries);
        if trace_on(base, 4 * n) {
            trace_life(format_args!("entry {id} {base:#x}+{:#x} made by {queue} (held)", 4 * n));
        }
        x.entries.push(Entry { id, base, words: words.into(), seq: 0, program, hash, from, old, earlier, how, queue: queue.into(), write_backs: super::homes::WRITE_BACKS.load(Ordering::Relaxed), flip: super::FRAMES.load(Ordering::Relaxed) });
        x.version += 1;
    });
    STATS[0].fetch_add(1, Ordering::Relaxed);
    HELD_STATS[0].fetch_add(1, Ordering::Relaxed);
    id
}

/// Starts a check of a parallel worker's buffer update, or of a copy left a
/// dispatch on a queue whose copies are not forwarded (`fe2::check_copies`:
/// a stale source in the mirror shows as words still as before) (`Check`);
/// its id.
pub fn add_check(base: u64, words: Vec<u32>, what: &'static str, queue: &str) -> u64 {
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    let n = words.len() as u64;
    let old: Vec<u32> = if super::cp::guest_readable(base, 4 * n) { (0..n).map(|k| unsafe { std::ptr::read_volatile((base + 4 * k) as *const u32) }).collect() } else { Vec::new() };
    CHECK_STATS[0].fetch_add(1, Ordering::Relaxed);
    CHECK_STATS[1].fetch_add(n, Ordering::Relaxed);
    let mut g = CHECKS.lock();
    if g.len() >= MAX_CHECKS {
        let c = g.remove(0);
        check_untrack(&c);
    }
    for slot in check_pages(base, 4 * n) {
        slot.fetch_add(1, Ordering::AcqRel);
    }
    g.push(Check { id, base, words, old, seq: 0, what, queue: queue.into() });
    CHECK_COUNT.store(g.len(), Ordering::Release);
    id
}

/// Drops the entries of batches that were never submitted (what the
/// graphics command processor recorded past a label wait, dropped).
pub fn drop_entries(ids: &[u64]) {
    if ids.is_empty() || COUNT.load(Ordering::Acquire) == 0 {
        return;
    }
    let mut g = ENTRIES.write();
    let before = g.len();
    for e in g.iter().filter(|e| ids.contains(&e.id) && trace_on(e.base, 4 * e.words.len() as u64)) {
        trace_life(format_args!("entry {} {:#x} dropped (its batch was not submitted)", e.id, e.base));
    }
    g.retain(|e| !ids.contains(&e.id));
    if g.len() != before {
        COUNT.store(g.len(), Ordering::Release);
        note_span(&g);
        GEN.fetch_add(1, Ordering::AcqRel);
    }
}

/// Drops the checks of batches that were never submitted.
pub fn drop_checks(ids: &[u64]) {
    if ids.is_empty() || CHECK_COUNT.load(Ordering::Acquire) == 0 {
        return;
    }
    let mut g = CHECKS.lock();
    g.retain(|c| {
        let keep = !ids.contains(&c.id);
        if !keep {
            check_untrack(c);
        }
        keep
    });
    CHECK_COUNT.store(g.len(), Ordering::Release);
}

/// The batch holding these entries' copies was submitted as `seq`.
pub fn submitted(ids: &[u64], seq: u64) {
    if ids.is_empty() {
        return;
    }
    if CHECK_COUNT.load(Ordering::Acquire) != 0 {
        let mut g = CHECKS.lock();
        for c in g.iter_mut() {
            if c.seq == 0 && ids.contains(&c.id) {
                c.seq = seq;
            }
        }
    }
    let mut g = ENTRIES.write();
    for e in g.iter_mut() {
        if ids.contains(&e.id) {
            e.seq = seq;
            if trace_on(e.base, 4 * e.words.len() as u64) {
                trace_life(format_args!("entry {} {:#x} submitted in batch {seq}", e.id, e.base));
            }
        }
    }
}

/// Batches up to `done` completed: their entries' words are compared with
/// memory (the GPU copied them by now) and the entries end.
pub fn completed(done: u64) {
    if CHECK_COUNT.load(Ordering::Acquire) != 0 {
        complete_checks(done);
    }
    if COUNT.load(Ordering::Acquire) == 0 {
        return;
    }
    let due: Vec<Entry> = {
        let mut g = ENTRIES.write();
        if !g.iter().any(|e| e.seq != 0 && e.seq <= done) {
            return;
        }
        let (due, keep): (Vec<Entry>, Vec<Entry>) = std::mem::take(&mut *g).into_iter().partition(|e| e.seq != 0 && e.seq <= done);
        for e in due.iter().filter(|e| trace_on(e.base, 4 * e.words.len() as u64)) {
            trace_life(format_args!("entry {} {:#x} retired (batch {} done, {done} completed)", e.id, e.base, e.seq));
        }
        *g = keep;
        COUNT.store(g.len(), Ordering::Release);
        note_span(&g);
        GEN.fetch_add(1, Ordering::AcqRel);
        due
    };
    for e in due {
        STATS[2].fetch_add(1, Ordering::Relaxed);
        if !super::cp::guest_readable(e.base, 4 * e.words.len() as u64) {
            continue;
        }
        // A destination in a video-memory home: the copy wrote the home,
        // and guest memory holds the old words until a reader on the CPU side
        // has the home written back (`homes`): nothing to compare here (these
        // were all the "did not land" reports of 5-2 loads, FINDINGS.md
        // 2026-10-16).
        if super::homes::overlaps(e.base, 4 * e.words.len() as u64) {
            STATS[10].fetch_add(1, Ordering::Relaxed);
            continue;
        }
        let mut differ = 0u64;
        let mut rewritten = 0u64;
        let mut first = None;
        for (k, &w) in e.words.iter().enumerate() {
            let now = unsafe { std::ptr::read_volatile((e.base + 4 * k as u64) as *const u32) };
            if now == w {
                continue;
            }
            if e.old.get(k) == Some(&now) && e.old[k] != w {
                differ += 1;
                first.get_or_insert((k, w, now));
            } else {
                rewritten += 1;
            }
        }
        STATS[9].fetch_add(rewritten, Ordering::Relaxed);
        STATS[5].fetch_add(e.words.len() as u64 - differ - rewritten, Ordering::Relaxed);
        if let Some((k, w, now)) = first {
            STATS[6].fetch_add(differ, Ordering::Relaxed);
            // What changed since the copy was recorded: its source, its
            // control block, or neither (then something else wrote the
            // destination, or the GPU copied other words).
            let f = &e.from;
            let src_now: Vec<u32> = if super::cp::guest_readable(f.src, (f.src_words.len() as u64).saturating_sub(1) * f.stride + 4) {
                (0..f.src_words.len() as u64).map(|j| unsafe { std::ptr::read_volatile((f.src + j * f.stride) as *const u32) }).collect()
            } else {
                Vec::new()
            };
            let src_changed = src_now.iter().zip(&f.src_words).filter(|(a, b)| a != b).count();
            let cb_now = if super::cp::guest_readable(f.cb, 8) { unsafe { [std::ptr::read_volatile(f.cb as *const u32), std::ptr::read_volatile((f.cb + 4) as *const u32)] } } else { [0, 0] };
            // A few words (at most 2, or 1 in 512) as before, or any number
            // with the copy's source and control block as when it was
            // recorded: memory missed them, not the forwarded result (the GPU
            // copied what the readers were given; a write of the previous
            // frame's block over the copy's in the two Doran boots of
            // 2026-11-06, 4 words of the per-view block and a whole 110-word
            // copy in three boots of 2026-11-07): the destination stays
            // forwarded, its readers taking the words the copy computes
            // rather than memory's (refusing it never mended memory and
            // slowed those boots 5-20% for good). A destination missing words
            // with unchanged inputs a fourth time within MISS_WINDOW flips is
            // refused (a copy the forwarding computed wrongly would miss at
            // every frame; such misses came about once a boot-hour, the
            // mirror then holding the title's later words, 2026-11-07, so a
            // count for the whole session would refuse a long session's block
            // in the end). Logged and counted (`TOLERATED`).
            let few = differ <= 2.max(e.words.len() as u64 / 512);
            let same_inputs = !src_now.is_empty() && src_changed == 0 && cb_now[0] as u64 == f.count && cb_now[1] as u64 == f.modulus;
            let misses = if few || !same_inputs {
                0
            } else {
                let flip = super::FRAMES.load(Ordering::Relaxed);
                let mut g = MISSES.lock();
                let c = g.get_or_insert_with(HashMap::new).entry((e.hash, e.base)).or_insert((0, flip));
                // (A window of MISS_WINDOW flips from its first miss.)
                if flip.saturating_sub(c.1) > MISS_WINDOW {
                    *c = (0, flip);
                }
                c.0 += 1;
                c.0
            };
            let keep = few || (same_inputs && misses <= 3);
            if keep {
                TOLERATED[0].fetch_add(1, Ordering::Relaxed);
                TOLERATED[1].fetch_add(differ, Ordering::Relaxed);
            }
            static LOGGED: AtomicU64 = AtomicU64::new(0);
            if LOGGED.fetch_add(1, Ordering::Relaxed) < 20 {
                crate::warn!(
                    "gpu: forwarded copy to {:#x} (cs {:#x}, batch {}): {differ} of {} words still as before the copy after it ran, first dword {k}: forwarded {w:#x}, memory {now:#x}; source {:#x} ({} words, stride {}): {src_changed} words changed since recorded{}; control block {:#x}: count {} modulus {} when recorded, {} {} now; last CP write to the source: {}; destination bound {} by {}, in a home now: {}, home write-backs since: {}; {}",
                    e.base,
                    e.program,
                    e.seq,
                    e.words.len(),
                    f.src,
                    f.src_words.len(),
                    f.stride,
                    if src_now.is_empty() { " (unreadable now)" } else { "" },
                    f.cb,
                    f.count,
                    f.modulus,
                    cb_now[0],
                    cb_now[1],
                    super::cp::last_write_info(f.src).unwrap_or_else(|| "none".into()),
                    e.how,
                    e.queue,
                    super::homes::overlaps(e.base, 4 * e.words.len() as u64),
                    super::homes::WRITE_BACKS.load(Ordering::Relaxed).saturating_sub(e.write_backs),
                    if few {
                        "the destination stays forwarded (a few words)".to_string()
                    } else if keep {
                        format!("the destination stays forwarded (its inputs as recorded: memory missed the copy's words; {misses} of 3 such within {MISS_WINDOW} flips tolerated)")
                    } else {
                        "this destination is no longer forwarded".to_string()
                    }
                );
                crate::warn!("gpu: the recent copies over {:#x} (recorded at flip {}, now {}): {}", e.base, e.flip, super::FRAMES.load(Ordering::Relaxed), copies_over(e.base, 4 * e.words.len() as u64));
                // (Whether the title unmapped the destination since: GPU
                // writes into a window imported before would not reach the
                // memory mapped there now.)
                let unmaps = crate::mem::unmaps_covering(e.base);
                crate::warn!("gpu: unmaps covering {:#x}: {}", e.base, if unmaps.is_empty() { "none noted".to_string() } else { unmaps.join("; ") });
                // The same words in the mirror (where the GPU's copy wrote
                // them): the copy's there means the write-back lost them,
                // the old ones that the copy never wrote there (later copies
                // into the block may have written it since). Read on a thread
                // of its own: the completion thread must not wait for the GPU
                // (batches may wait for its signals).
                let a = (e.base + 4 * k as u64) & !15;
                let first = ((a - e.base) / 4) as usize;
                let copied: Vec<u32> = e.words.iter().skip(first).take(4).copied().collect();
                let before: Vec<u32> = e.old.iter().skip(first).take(4).copied().collect();
                if (super::resident::MIRROR_BASE..super::resident::MIRROR_END).contains(&a) {
                    let _ = std::thread::Builder::new().name("forward-check".into()).spawn(move || {
                        let hex = |v: &[u32]| v.iter().map(|w| format!("{w:#x}")).collect::<Vec<_>>().join(" ");
                        match super::resident::readback_now(a, 16) {
                            Ok(b) => {
                                let m: Vec<u32> = b.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                                crate::warn!("gpu: the mirror at {a:#x} after the copy that did not land: {} (the copy's {}, before it {})", hex(&m), hex(&copied), hex(&before));
                            }
                            Err(err) => crate::warn!("gpu: the mirror at {a:#x} not read back: {err}"),
                        }
                    });
                }
            }
            if !keep {
                DISTRUSTED.lock().get_or_insert_with(HashSet::new).insert((e.hash, e.base));
            }
        }
    }
}

/// Copies whose words still as before after they ran (a few, or with their
/// inputs unchanged) kept their destination forwarded: [copies, words].
static TOLERATED: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
/// (program hash, destination) pairs: copies whose words memory missed with
/// their inputs unchanged, and the flip their window began (the fourth within
/// MISS_WINDOW flips refuses the destination).
static MISSES: Mutex<Option<HashMap<(u64, u64), (u32, u64)>>> = Mutex::new(None);

/// Flips over which a destination's misses with unchanged inputs are counted
/// (~1-2 minutes of play).
const MISS_WINDOW: u64 = 3600;

/// Batches up to `done` completed: their workers' buffer updates are
/// compared with memory.
fn complete_checks(done: u64) {
    let due: Vec<Check> = {
        let mut g = CHECKS.lock();
        if !g.iter().any(|c| c.seq != 0 && c.seq <= done) {
            return;
        }
        let (due, keep): (Vec<Check>, Vec<Check>) = std::mem::take(&mut *g).into_iter().partition(|c| c.seq != 0 && c.seq <= done);
        *g = keep;
        for c in &due {
            check_untrack(c);
        }
        CHECK_COUNT.store(g.len(), Ordering::Release);
        due
    };
    for c in due {
        if !super::cp::guest_readable(c.base, 4 * c.words.len() as u64) {
            continue;
        }
        let (mut equal, mut stale, mut other) = (0u64, 0u64, 0u64);
        for (k, &w) in c.words.iter().enumerate() {
            let now = unsafe { std::ptr::read_volatile((c.base + 4 * k as u64) as *const u32) };
            if now == w {
                equal += 1;
            } else if c.old.get(k) == Some(&now) {
                stale += 1;
            } else {
                other += 1;
            }
        }
        CHECK_STATS[2].fetch_add(equal, Ordering::Relaxed);
        CHECK_STATS[3].fetch_add(stale, Ordering::Relaxed);
        CHECK_STATS[4].fetch_add(other, Ordering::Relaxed);
        if stale > 0 {
            static LOGGED: AtomicU64 = AtomicU64::new(0);
            if LOGGED.fetch_add(1, Ordering::Relaxed) < 20 {
                crate::warn!("gpu: {} to {:#x} ({}, batch {}): {stale} of {} words still as before it after it ran", c.what, c.base, c.queue, c.seq, c.words.len());
            }
        }
    }
}

/// The status line ("" when nothing was forwarded).
pub fn status_line() -> String {
    let s: Vec<u64> = STATS.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    if s[0] == 0 && s[7] == 0 {
        return String::new();
    }
    let names: std::collections::HashMap<u8, String> = crate::engine::allocators().into_iter().map(|a| (a.id, a.name)).collect();
    let mut owners: Vec<(u8, u64, u64)> = (0..256usize).map(|i| (i as u8, BY_OWNER[i][0].load(Ordering::Relaxed), BY_OWNER[i][1].load(Ordering::Relaxed))).filter(|x| x.1 > 0).collect();
    owners.sort_by(|a, b| b.1.cmp(&a.1));
    let owners: Vec<String> = owners.iter().take(6).map(|&(i, n, w)| format!("{} {n} ({w} words)", names.get(&i).cloned().unwrap_or_else(|| format!("owner {i}")))).collect();
    let k: Vec<u64> = CHECK_STATS.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    format!(
        "    forwarded copies: {} made ({} not: inputs still written or odd shapes), {} ended by later writes, {} by the title's rewrites, {} retired after their batch ({} into a video-memory home, not compared); {} words served, {} stage reads with no wait for the copy; after the copies ran {} words equal, {} as before the copy (it did not land; {} copies of a few such words or unchanged inputs kept forwarded, {} words), {} rewritten since; checked after their batches {} ({} words; {} words equal, {} as before, {} rewritten; {} checks ended by later writes); past parallel runs {} (published {}, dropped {}); destinations by engine allocator: {}\n",
        s[0], s[7], s[1], s[8], s[2], s[10], s[3], s[4], s[5], s[6],
        TOLERATED[0].load(Ordering::Relaxed),
        TOLERATED[1].load(Ordering::Relaxed),
        s[9], k[0], k[1], k[2], k[3], k[4], k[5],
        HELD_STATS[0].load(Ordering::Relaxed),
        HELD_STATS[1].load(Ordering::Relaxed),
        HELD_STATS[2].load(Ordering::Relaxed),
        owners.join(", ")
    )
}
