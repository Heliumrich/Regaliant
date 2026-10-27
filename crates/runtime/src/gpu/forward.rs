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
//! (`STATS`); a destination whose copy did not land is not forwarded again.

use parking_lot::{Mutex, RwLock};
use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use super::shaders::{Prepared, Resolved};

/// The words a copy leaves at `base` as the command processor computed them.
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
    /// Whether the copy's source or control block overlaps [a, a + n).
    pub fn overlaps(&self, a: u64, n: u64) -> bool {
        let src_end = self.src + (self.src_words.len() as u64).saturating_sub(1) * self.stride + 4;
        (self.src < a + n && a < src_end) || (self.cb < a + n && a < self.cb + 8)
    }
}

static ENTRIES: RwLock<Vec<Entry>> = RwLock::new(Vec::new());
static COUNT: AtomicUsize = AtomicUsize::new(0);
static GEN: AtomicU64 = AtomicU64::new(1);
static NEXT_ID: AtomicU64 = AtomicU64::new(1);
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

/// Copies recorded as buffer updates of their computed words instead of
/// dispatches (`tune copy_update`; `Exec::copy_as_update`): [recorded, words
/// written, computed but recorded as dispatches (a home, a resident shadow
/// or unmapped memory at the destination)].
pub static UPDATES: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// A copy a parallel worker of the graphics queue recorded as a buffer
/// update (`tune copy_update 2`): compared with memory once its batch
/// completed. Never served to readers: the workers record their chunks out
/// of program order.
struct Check {
    id: u64,
    base: u64,
    words: Vec<u32>,
    /// The destination's words when the copy was recorded.
    old: Vec<u32>,
    /// Its batch (0 until submitted).
    seq: u64,
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

/// Recognition per program code hash (Some: the copy kernel).
static PROGRAMS: Mutex<Option<HashMap<u64, Option<Roles>>>> = Mutex::new(None);
/// The addresses the copy kernel was met at (a long run's planner forwards
/// their dispatches ahead: `plan`).
static COPY_ADDRS: RwLock<Option<HashSet<u64>>> = RwLock::new(None);

/// Whether the program at `addr` was recognized as the copy kernel.
pub fn is_copy_program(addr: u64) -> bool {
    COPY_ADDRS.read().as_ref().is_some_and(|s| s.contains(&addr))
}
/// (program hash, destination) pairs whose forwarded words once differed
/// from memory: not forwarded again.
static DISTRUSTED: Mutex<Option<HashSet<(u64, u64)>>> = Mutex::new(None);

thread_local! {
    /// This thread reads guest memory for the graphics queue.
    static HERE: Cell<bool> = const { Cell::new(false) };
    /// This thread's copy of the entries: (generation, [(base, bytes,
    /// copied words, old words)], the span [lo, hi) they lie in).
    static LOCAL: RefCell<(u64, Vec<LocalEntry>, (u64, u64))> = const { RefCell::new((0, Vec::new(), (0, 0))) };
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
    /// Bumped at each change (this thread's view is cached by it and GEN).
    version: u64,
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
    HELD.with(|h| *h.borrow_mut() = Some(Held::default()));
}

/// The held stretch is committed: its entries become global (submitted with
/// its batches next, `submitted`) and the global entries its writes ended go.
/// (A long run's chunk publishes its own copies; the planner's it was given
/// are the run's earlier chunks' to publish.)
pub fn hold_commit() {
    let Some(held) = HELD.with(|h| h.borrow_mut().take()) else { return };
    if held.entries.is_empty() && held.hidden.is_empty() {
        return;
    }
    let mut g = ENTRIES.write();
    g.retain(|e| !held.hidden.contains(&e.id));
    for e in held.entries.into_iter().filter(|e| e.how != PLANNED) {
        let end = e.base + 4 * e.words.len() as u64;
        g.retain(|x| !(x.base < end && e.base < x.base + 4 * x.words.len() as u64));
        if g.len() >= MAX_ENTRIES {
            g.remove(0);
        }
        g.push(e);
        HELD_STATS[1].fetch_add(1, Ordering::Relaxed);
    }
    COUNT.store(g.len(), Ordering::Release);
    GEN.fetch_add(1, Ordering::AcqRel);
}

/// The held stretch is dropped (recorded again from where it began): what it
/// forwarded is forgotten, the global entries are as they were.
pub fn hold_drop() {
    if let Some(held) = HELD.with(|h| h.borrow_mut().take()) {
        HELD_STATS[2].fetch_add(held.entries.len() as u64, Ordering::Relaxed);
    }
}

/// The pieces of [s, e) outside every range of `cover` ([start, end)).
fn outside(s: u64, e: u64, cover: &[(u64, u64)]) -> Vec<(u64, u64)> {
    let mut pieces = vec![(s, e)];
    for &(lo, hi) in cover {
        let mut next = Vec::with_capacity(pieces.len() + 1);
        for (a, b) in pieces {
            if b <= lo || hi <= a {
                next.push((a, b));
                continue;
            }
            if a < lo {
                next.push((a, lo));
            }
            if hi < b {
                next.push((hi, b));
            }
        }
        pieces = next;
    }
    pieces
}

/// The CPU reads of a long parallel run's chunk (`Exec::long`, a held view
/// on this thread) by how they were served: parts this thread's own copies
/// cover are left out (its chunk's, earlier in program order), parts the
/// planner's copies it was given or the global entries cover go to the
/// second list (right unless the run's earlier chunks wrote there otherwise
/// than by those copies: the planner forwarded every copy of the run it
/// could, the global entries were current when the run began), the rest to
/// the first.
pub fn split_long_reads(reads: &[(u64, u64)]) -> (Vec<(u64, u64)>, Vec<(u64, u64)>) {
    let (own, mut global, hidden): (Vec<(u64, u64)>, Vec<(u64, u64)>, Vec<u64>) = HELD.with(|h| match h.borrow().as_ref() {
        Some(x) => (
            x.entries.iter().filter(|e| e.how != PLANNED).map(|e| (e.base, e.base + 4 * e.words.len() as u64)).collect(),
            x.entries.iter().filter(|e| e.how == PLANNED).map(|e| (e.base, e.base + 4 * e.words.len() as u64)).collect(),
            x.hidden.clone(),
        ),
        None => (Vec::new(), Vec::new(), Vec::new()),
    });
    if COUNT.load(Ordering::Acquire) != 0 {
        global.extend(ENTRIES.read().iter().filter(|e| !hidden.contains(&e.id)).map(|e| (e.base, e.base + 4 * e.words.len() as u64)));
    }
    if own.is_empty() && global.is_empty() {
        return (reads.to_vec(), Vec::new());
    }
    let (mut plain, mut fwd) = (Vec::new(), Vec::new());
    for &(a, n) in reads {
        for (s, e) in outside(a, a + n, &own) {
            let rest = outside(s, e, &global);
            let mut at = s;
            for &(rs, re) in &rest {
                if at < rs {
                    fwd.push((at, rs - at));
                }
                plain.push((rs, re - rs));
                at = re;
            }
            if at < e {
                fwd.push((at, e - at));
            }
        }
    }
    (plain, fwd)
}

/// `Entry::how` of the copies a long run's planner forwarded (`plan`).
const PLANNED: &str = "planned";

/// A copy a long parallel run's planner forwarded ahead (`Cp::par_plan`):
/// its packet, destination and words, and what an entry of it holds.
#[derive(Clone)]
pub struct Planned {
    pub at: u64,
    pub base: u64,
    pub words: Arc<[u32]>,
    program: u64,
    hash: u64,
    from: Inputs,
    old: Arc<[u32]>,
    earlier: Arc<[Arc<[u32]>]>,
}

/// Forwards the copy `p` dispatched at packet `at` with `groups` ahead of
/// recording (a long run's planner; `busy` as `compute`'s). `prior`: the
/// run's copies planned before it: theirs (and the global entry's) are the
/// words memory may hold at the destination before this one runs.
pub fn plan(p: &Prepared, at: u64, groups: [u32; 3], local: [u32; 3], busy: impl Fn(&[(u64, u64)]) -> bool, prior: &[Planned]) -> Option<Planned> {
    let (base, words, from) = compute(p, groups, local, busy)?;
    let n = words.len() as u64;
    let old: Arc<[u32]> = if super::cp::guest_readable(base, 4 * n) {
        (0..n).map(|k| unsafe { std::ptr::read_volatile((base + 4 * k) as *const u32) }).collect()
    } else {
        Arc::from(Vec::new())
    };
    let mut v: Vec<Arc<[u32]>> = match ENTRIES.read().iter().find(|e| e.base == base && e.words.len() as u64 == n) {
        Some(e) => {
            let mut v: Vec<Arc<[u32]>> = e.earlier.iter().cloned().collect();
            v.push(e.words.clone());
            v
        }
        None => Vec::new(),
    };
    v.extend(prior.iter().filter(|q| q.base == base && q.words.len() as u64 == n).map(|q| q.words.clone()));
    if v.len() > EARLIER_MAX {
        v.drain(..v.len() - EARLIER_MAX);
    }
    PLANNED_STATS[0].fetch_add(1, Ordering::Relaxed);
    Some(Planned { at, base, words: words.into(), program: p.addr, hash: p.code_hash, from, old, earlier: v.into() })
}

impl Planned {
    /// The same copy as one before all of a run chained after its own
    /// (`Cp::chain_next`): at packet 0.
    pub fn before_all(&self) -> Planned {
        Planned { at: 0, ..self.clone() }
    }
}

/// [copies a long run's planner forwarded, copies a worker computed
/// otherwise than planned].
pub static PLANNED_STATS: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// The planned copies before packet `start` (latest per destination): what
/// a long run's chunk starting there inherits.
pub fn inherited(planned: &[Planned], start: u64) -> Vec<Planned> {
    let mut v: Vec<Planned> = Vec::new();
    for q in planned.iter().filter(|q| q.at < start) {
        let end = q.base + 4 * q.words.len() as u64;
        v.retain(|x| !(x.base < end && q.base < x.base + 4 * x.words.len() as u64));
        v.push(q.clone());
    }
    v
}

/// A held view on this thread that starts with `inherited` (a long run's
/// chunk: the copies its planner forwarded before the chunk).
pub fn hold_begin_with(inherited: &[Planned]) {
    let entries: Vec<Entry> = inherited
        .iter()
        .map(|q| Entry {
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            base: q.base,
            words: q.words.clone(),
            seq: 0,
            program: q.program,
            hash: q.hash,
            from: q.from.clone(),
            old: q.old.clone(),
            earlier: q.earlier.clone(),
            how: PLANNED,
            queue: Arc::from("planner"),
            write_backs: super::homes::WRITE_BACKS.load(Ordering::Relaxed),
        })
        .collect();
    let version = (!entries.is_empty()) as u64;
    HELD.with(|h| *h.borrow_mut() = Some(Held { entries, hidden: Vec::new(), version }));
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
    (COUNT.load(Ordering::Acquire) != 0 || HELD.with(|h| h.borrow().as_ref().is_some_and(|x| !x.entries.is_empty()))) && HERE.with(|h| h.get())
}

type LocalEntry = (u64, u64, Arc<[u32]>, Arc<[u32]>, Arc<[Arc<[u32]>]>);

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
    // A held view's own generation is mixed in (its entries and hidden ones).
    let held_gen = HELD.with(|h| h.borrow().as_ref().map_or(0, |x| x.version.wrapping_add(1) << 40));
    let generation = GEN.load(Ordering::Acquire) ^ held_gen;
    LOCAL.with(|l| {
        let mut l = l.borrow_mut();
        if l.0 != generation {
            let g = ENTRIES.read();
            l.1 = HELD.with(|h| match h.borrow().as_ref() {
                None => g.iter().map(|e| (e.base, 4 * e.words.len() as u64, e.words.clone(), e.old.clone(), e.earlier.clone())).collect(),
                Some(x) => x
                    .entries
                    .iter()
                    .chain(g.iter().filter(|e| !x.hidden.contains(&e.id) && !x.entries.iter().any(|p| p.base < e.base + 4 * e.words.len() as u64 && e.base < p.base + 4 * p.words.len() as u64)))
                    .map(|e| (e.base, 4 * e.words.len() as u64, e.words.clone(), e.old.clone(), e.earlier.clone()))
                    .collect(),
            });
            l.2 = l.1.iter().fold((u64::MAX, 0), |(lo, hi), e| (lo.min(e.0), hi.max(e.0 + e.1)));
            l.0 = generation;
        }
        f(&l.1, l.2)
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
            (e.0, e.2[k], before_copy(&e.3, &e.4, k, mem))
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
        for (base, bytes, words, old, earlier) in v {
            let (lo, hi) = (addr.max(*base), end.min(base + bytes));
            if lo >= hi {
                continue;
            }
            let ok = (lo..hi).step_by(4).all(|a| {
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
    with_local_span(|v, (lo, hi)| {
        if !reads.iter().any(|&(a, n)| a < hi && lo < a + n && v.iter().any(|e| e.0 < a + n && a < e.0 + e.1)) {
            return None;
        }
        let mut out = Vec::with_capacity(reads.len());
        let mut covered = 0u64;
        for &(a, n) in reads {
            // Pieces of [a, a + n) outside every entry.
            let mut pieces = vec![(a, a + n)];
            for (base, bytes, ..) in v {
                let (lo, hi) = (*base, base + bytes);
                let mut next = Vec::with_capacity(pieces.len() + 1);
                for (s, e) in pieces {
                    if e <= lo || hi <= s {
                        next.push((s, e));
                        continue;
                    }
                    if s < lo {
                        next.push((s, lo));
                    }
                    if hi < e {
                        next.push((hi, e));
                    }
                }
                pieces = next;
            }
            if pieces.is_empty() {
                covered += 1;
            }
            out.extend(pieces.into_iter().map(|(s, e)| (s, e - s)));
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
        x.entries.retain(|e| !hit(e.base, e.words.len()));
        let mut changed = x.entries.len() != before;
        if COUNT.load(Ordering::Acquire) != 0 {
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
fn roles(p: &Prepared) -> Option<Roles> {
    let insts = &p.found.insts;
    let key = p.code_hash;
    let known = PROGRAMS.lock().get_or_insert_with(HashMap::new).get(&key).copied();
    if let Some(r) = known {
        if r.is_some() && !is_copy_program(p.addr) {
            COPY_ADDRS.write().get_or_insert_with(HashSet::new).insert(p.addr);
        }
        return r;
    }
    let r = recognize(insts, &p.found.tr.plan);
    if r.is_some() {
        COPY_ADDRS.write().get_or_insert_with(HashSet::new).insert(p.addr);
    }
    if r.is_some() {
        crate::info!("gpu: cs {:#x} (code hash {key:#018x}) is a guarded modulo copy: its results are forwarded to later reads (`tune forward_copies`)", p.addr);
    }
    PROGRAMS.lock().get_or_insert_with(HashMap::new).insert(key, r);
    r
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

/// The words a dispatch of `p` with `groups` (64-lane groups along X) will
/// leave at its destination, when `p` is the copy kernel and its control
/// block and source can be read now (`busy`: GPU work or a deferred CP write
/// still writes one of those ranges). (destination base, words).
pub fn compute(p: &Prepared, groups: [u32; 3], local: [u32; 3], busy: impl Fn(&[(u64, u64)]) -> bool) -> Option<(u64, Vec<u32>, Inputs)> {
    let r = roles(p)?;
    let buf = |i: usize| match p.found.resources.get(i) {
        Some(Resolved::Buffer(v)) => Some(*v),
        _ => None,
    };
    let (src, dst, cb) = (buf(r.src)?, buf(r.dst)?, buf(r.cb)?);
    if DISTRUSTED.lock().as_ref().is_some_and(|d| d.contains(&(p.code_hash, dst.base))) {
        return None;
    }
    let refuse = || {
        STATS[7].fetch_add(1, Ordering::Relaxed);
        None
    };
    if local != [64, 1, 1] || groups[1] != 1 || groups[2] != 1 || !plain32(src.format) || !plain32(dst.format) || dst.stride != 4 || src.stride < 4 || src.add_tid || dst.add_tid {
        return refuse();
    }
    if !super::cp::guest_readable(cb.base, 8) {
        return refuse();
    }
    let n = unsafe { std::ptr::read_volatile(cb.base as *const u32) } as u64;
    let m = unsafe { std::ptr::read_volatile((cb.base + 4) as *const u32) } as u64;
    let count = n.min(groups[0] as u64 * 64).min(dst.num_records as u64);
    let in_src = m.min(src.num_records as u64);
    if m == 0 || count == 0 || count > 1 << 16 || in_src > 1 << 16 {
        return refuse();
    }
    let src_bytes = (in_src - 1) * src.stride as u64 + 4;
    let inputs = [(cb.base, 8), (src.base, src_bytes)];
    if busy(&inputs) || super::homes::any_stale(&inputs) || super::keep::any_dirty(&inputs) || !super::cp::guest_readable(src.base, src_bytes) {
        trace_refusal(dst.base, 4 * count, &inputs, &busy);
        return refuse();
    }
    let src_words: Vec<u32> = (0..in_src).map(|j| unsafe { std::ptr::read_volatile((src.base + j * src.stride as u64) as *const u32) }).collect();
    let words = (0..count)
        .map(|i| {
            let j = i % m;
            if j < src.num_records as u64 { src_words[j as usize] } else { 0 }
        })
        .collect();
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
        if super::keep::any_dirty(&[r]) {
            v.push("a kept window's newer data");
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
pub fn add(base: u64, words: Vec<u32>, program: u64, hash: u64, from: Inputs, how: &'static str, queue: &str) -> u64 {
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    let n = words.len() as u64;
    let old: Arc<[u32]> = if super::cp::guest_readable(base, 4 * n) {
        (0..n).map(|k| unsafe { std::ptr::read_volatile((base + 4 * k) as *const u32) }).collect()
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
    g.push(Entry { id, base, words: words.into(), seq: 0, program, hash, from, old, earlier, how, queue: queue.into(), write_backs: super::homes::WRITE_BACKS.load(Ordering::Relaxed) });
    COUNT.store(g.len(), Ordering::Release);
    GEN.fetch_add(1, Ordering::AcqRel);
    STATS[0].fetch_add(1, Ordering::Relaxed);
    id
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
        if x.entries.len() >= MAX_ENTRIES {
            x.entries.remove(0);
        }
        if trace_on(base, 4 * n) {
            trace_life(format_args!("entry {id} {base:#x}+{:#x} made by {queue} (held)", 4 * n));
        }
        x.entries.push(Entry { id, base, words: words.into(), seq: 0, program, hash, from, old, earlier, how, queue: queue.into(), write_backs: super::homes::WRITE_BACKS.load(Ordering::Relaxed) });
        x.version += 1;
    });
    STATS[0].fetch_add(1, Ordering::Relaxed);
    HELD_STATS[0].fetch_add(1, Ordering::Relaxed);
    id
}

/// Starts a check of a parallel worker's buffer update (`Check`); its id.
pub fn add_check(base: u64, words: Vec<u32>) -> u64 {
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
    g.push(Check { id, base, words, old, seq: 0 });
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
            static LOGGED: AtomicU64 = AtomicU64::new(0);
            if LOGGED.fetch_add(1, Ordering::Relaxed) < 20 {
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
                crate::warn!(
                    "gpu: forwarded copy to {:#x} (cs {:#x}, batch {}): {differ} of {} words still as before the copy after it ran, first dword {k}: forwarded {w:#x}, memory {now:#x}; source {:#x} ({} words, stride {}): {src_changed} words changed since recorded{}; control block {:#x}: count {} modulus {} when recorded, {} {} now; last CP write to the source: {}; destination bound {} by {}, in a home now: {}, home write-backs since: {}; this destination is no longer forwarded",
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
                    super::homes::WRITE_BACKS.load(Ordering::Relaxed).saturating_sub(e.write_backs)
                );
            }
            DISTRUSTED.lock().get_or_insert_with(HashSet::new).insert((e.hash, e.base));
        }
    }
}

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
                crate::warn!("gpu: a worker's buffer update to {:#x} (batch {}): {stale} of {} words still as before it after it ran", c.base, c.seq, c.words.len());
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
    let u: Vec<u64> = UPDATES.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    let k: Vec<u64> = CHECK_STATS.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    format!(
        "    forwarded copies: {} made ({} not: inputs still written or odd shapes), {} ended by later writes, {} by the title's rewrites, {} retired after their batch ({} into a video-memory home, not compared); {} words served, {} stage reads with no wait for the copy; after the copies ran {} words equal, {} as before the copy (it did not land), {} rewritten since; recorded as buffer updates {} ({} words; {} as dispatches); by the workers {} ({} words; after their batches {} words equal, {} as before, {} rewritten; {} checks ended by later writes); past parallel runs {} (published {}, dropped {})\n",
        s[0], s[7], s[1], s[8], s[2], s[10], s[3], s[4], s[5], s[6], s[9], u[0], u[1], u[2], k[0], k[1], k[2], k[3], k[4], k[5],
        HELD_STATS[0].load(Ordering::Relaxed),
        HELD_STATS[1].load(Ordering::Relaxed),
        HELD_STATS[2].load(Ordering::Relaxed)
    )
}
