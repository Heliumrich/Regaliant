//! Scouts: the graphics queue's title submissions recorded ahead of its
//! command processor (docs/SCOUTS.md, ROUTE60.md step 1).
//!
//! The graphics command processor runs ~21 submissions a flip one after the
//! other while each waits ~6 ms in the queue and the chunk workers idle
//! (FINDINGS.md 2026-11-14 night). A scout (`gpu-gfx-s<k>`: a command
//! processor with an executor and a worker pool of its own) takes a queued
//! submission the main command processor has not reached and records it
//! held, from a start state walked from the stream (`Cp::walk_submission`),
//! as the main command processor records a stretch past a label wait
//! (`Cp::scout_run`).
//!
//! Census stage (`tune scouts <n>`, `DS_SCOUTS`; 0, the default: none): a
//! record is checked when the main command processor reaches its submission
//! (the start state against the command processor's own, the CPU reads
//! against everything written since the scout began, the images against
//! pending clears and reloads) and then always dropped; the main command
//! processor records every submission as before. The status line says how
//! far scouts get, what ends their records and what the checks find: what
//! the hand-over (the next stage) will meet.

use super::cp::{Cp, WalkState};
use parking_lot::{Condvar, Mutex};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// Scouts wanted (`tune scouts`; at most 4).
pub fn count() -> usize {
    (super::tune::scouts() as usize).min(4)
}

static STARTED: AtomicBool = AtomicBool::new(false);

/// Whether scouts run (started with the graphics queue: `start`).
#[inline]
pub fn on() -> bool {
    STARTED.load(Ordering::Relaxed)
}

/// A queued submission as the scouts see it.
struct Entry {
    index: u64,
    bufs: Vec<(u64, u32)>,
    at: Instant,
    /// The state at its end, walked (Err: the walk refused a packet; None:
    /// not walked yet), and the label writes it will make.
    end: Option<Result<Arc<WalkState>, &'static str>>,
    labels: Arc<Vec<(u64, u64, u8)>>,
    slot: Option<Arc<Slot>>,
}

/// The submission the main command processor runs now: the walk's base.
struct Base {
    bufs: Vec<(u64, u32)>,
    state: Arc<WalkState>,
    end: Option<Result<Arc<WalkState>, &'static str>>,
    labels: Arc<Vec<(u64, u64, u8)>>,
}

#[derive(Default)]
struct Lanes {
    queued: VecDeque<Entry>,
    base: Option<Base>,
}

static LANES: Mutex<Option<Lanes>> = Mutex::new(None);
static CV: Condvar = Condvar::new();

/// A claimed submission: what the main command processor leaves for its
/// scout when it reaches the submission (its turn).
pub struct Slot {
    turn: Mutex<Option<Turn>>,
    cv: Condvar,
    /// The scout's journal (`exec::scout_journal_begin`; 0: not begun).
    journal: AtomicU64,
}

struct Turn {
    /// The main command processor's own state at the submission's start.
    state: Arc<WalkState>,
    at: Instant,
}

struct Job {
    index: u64,
    bufs: Vec<(u64, u32)>,
    at: Instant,
    start: Arc<WalkState>,
    future: Arc<Vec<(u64, u64, u8)>>,
    slot: Arc<Slot>,
    /// Submissions between the main command processor's and this one when
    /// it was claimed.
    ahead: usize,
}

/// Census counters (`status`).
static CLAIMED: AtomicU64 = AtomicU64::new(0);
static RECORDED: AtomicU64 = AtomicU64::new(0);
static WHOLE: AtomicU64 = AtomicU64::new(0);
static IN_TIME: AtomicU64 = AtomicU64::new(0);
static LATE_NS: AtomicU64 = AtomicU64::new(0);
static EARLY_NS: AtomicU64 = AtomicU64::new(0);
static REC_NS: AtomicU64 = AtomicU64::new(0);
static COMMANDS: AtomicU64 = AtomicU64::new(0);
static RUNS: AtomicU64 = AtomicU64::new(0);
static BATCHES: AtomicU64 = AtomicU64::new(0);
static CHUNK_BATCHES: AtomicU64 = AtomicU64::new(0);
static READS: AtomicU64 = AtomicU64::new(0);
static AHEAD: AtomicU64 = AtomicU64::new(0);
static DWORDS: AtomicU64 = AtomicU64::new(0);
static DWORDS_DONE: AtomicU64 = AtomicU64::new(0);
static WAITS: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];
/// Checks at the turn: [records checked, start states that differed, records
/// with CPU reads written meanwhile, those reads, records with images
/// pending, records every check passed, of them whole submissions, dwords of
/// the records every check passed, records whose only reads written meanwhile
/// were textures', records every check but those passed, of them whole
/// submissions, their dwords].
static CHECKS: [AtomicU64; 12] = [const { AtomicU64::new(0) }; 12];
static UNWALKED: AtomicU64 = AtomicU64::new(0);
static MISSED: AtomicU64 = AtomicU64::new(0);
static ENDS: Mutex<Vec<(&'static str, u64, u64)>> = Mutex::new(Vec::new());
static GATES: Mutex<Vec<(String, u64)>> = Mutex::new(Vec::new());
static CONFLICTS: Mutex<Vec<(String, u64)>> = Mutex::new(Vec::new());
static WALK_ERRS: Mutex<Vec<(&'static str, u64)>> = Mutex::new(Vec::new());
static LOGGED: AtomicU64 = AtomicU64::new(0);

fn bump<K: PartialEq + Clone>(l: &Mutex<Vec<(K, u64)>>, k: &K) {
    let mut g = l.lock();
    match g.iter_mut().find(|e| e.0 == *k) {
        Some(e) => e.1 += 1,
        None => g.push((k.clone(), 1)),
    }
}

/// A new window for the census (`profile reset`).
pub fn reset_stats() {
    for c in [&CLAIMED, &RECORDED, &WHOLE, &IN_TIME, &LATE_NS, &EARLY_NS, &REC_NS, &COMMANDS, &RUNS, &BATCHES, &CHUNK_BATCHES, &READS, &AHEAD, &DWORDS, &DWORDS_DONE, &UNWALKED, &MISSED] {
        c.store(0, Ordering::Relaxed);
    }
    for c in WAITS.iter().chain(CHECKS.iter()) {
        c.store(0, Ordering::Relaxed);
    }
    ENDS.lock().clear();
    GATES.lock().clear();
    CONFLICTS.lock().clear();
    WALK_ERRS.lock().clear();
    LOGGED.store(0, Ordering::Relaxed);
    for c in &super::exec::SCOUT_TEXTURES {
        c.store(0, Ordering::Relaxed);
    }
}

/// The graphics queue starts: its scouts' threads (`tune scouts`; none
/// without front end 2 or GPU execution).
pub fn start() {
    let n = count();
    if n == 0 || !super::fe2::on() || !super::exec::enabled() || STARTED.swap(true, Ordering::SeqCst) {
        return;
    }
    *LANES.lock() = Some(Lanes::default());
    for k in 0..n {
        let name = format!("gpu-gfx-s{k}");
        let tname = name.clone();
        let r = crate::thread::spawn_host(
            &name,
            8 << 20,
            Box::new(move || {
                crate::log::set_thread_name(&tname);
                crate::exception::arm_watch_current_thread(false);
                super::profile_this_thread("par");
                super::forward::enable_here();
                run(&tname);
            }),
        );
        if let Err(e) = r {
            crate::warn!("gpu[{name}]: no thread for a scout ({e})");
        }
    }
    crate::info!("gpu: {n} scout(s) record the graphics queue's submissions ahead (census: checked at their turn, then dropped; docs/SCOUTS.md)");
}

/// The title queued graphics submission `index` (under the queue's lock: in
/// order).
pub fn queued(index: u64, bufs: &[(u64, u32)], at: Instant) {
    if !on() {
        return;
    }
    let mut g = LANES.lock();
    if let Some(l) = g.as_mut() {
        l.queued.push_back(Entry { index, bufs: bufs.to_vec(), at, end: None, labels: Arc::new(Vec::new()), slot: None });
    }
    drop(g);
    CV.notify_all();
}

/// The main command processor takes submission `index` (about to run it):
/// its own state there becomes the walk's base, and the submission's scout,
/// if it has one, gets its turn (its journal ends here).
pub fn main_takes(index: u64, bufs: &[(u64, u32)], cp: &Cp) {
    if !on() {
        return;
    }
    let state = Arc::new(cp.walk_start());
    let mut g = LANES.lock();
    let Some(l) = g.as_mut() else { return };
    // (Entries before it: submissions the queue's worker ran another way, a
    // replay's control entries.)
    while l.queued.front().is_some_and(|e| e.index < index) {
        let e = l.queued.pop_front().unwrap();
        if let Some(s) = e.slot {
            give_turn(&s, state.clone());
        }
    }
    let slot = match l.queued.front() {
        Some(e) if e.index == index => l.queued.pop_front().and_then(|e| e.slot),
        _ => None,
    };
    l.base = Some(Base { bufs: bufs.to_vec(), state: state.clone(), end: None, labels: Arc::new(Vec::new()) });
    drop(g);
    if let Some(s) = slot {
        give_turn(&s, state);
    }
    CV.notify_all();
}

fn give_turn(s: &Slot, state: Arc<WalkState>) {
    let j = s.journal.load(Ordering::Acquire);
    if j != 0 {
        super::exec::scout_journal_freeze(j);
    }
    *s.turn.lock() = Some(Turn { state, at: Instant::now() });
    s.cv.notify_all();
}

/// Walks `bufs` from `from`: the state at their end and the label writes
/// they make.
fn walk(from: &WalkState, bufs: &[(u64, u32)]) -> (Result<Arc<WalkState>, &'static str>, Arc<Vec<(u64, u64, u8)>>) {
    let mut w = from.clone();
    w.labels.clear();
    w.flip = None;
    match Cp::walk_submission(&mut w, bufs) {
        Ok(_) => {
            let labels = Arc::new(std::mem::take(&mut w.labels));
            (Ok(Arc::new(w)), labels)
        }
        Err(e) => {
            bump(&WALK_ERRS, &e);
            (Err(e), Arc::new(Vec::new()))
        }
    }
}

/// The oldest queued submission behind the queue's head that no scout has
/// claimed and whose start state the walk reaches; waits for one.
fn take_job() -> Option<Job> {
    let mut g = LANES.lock();
    loop {
        if super::shutting_down() {
            return None;
        }
        if let Some(l) = g.as_mut() {
            let mut job = None;
            if let Some(b) = l.base.as_mut() {
                if b.end.is_none() {
                    let (end, labels) = walk(&b.state, &b.bufs);
                    b.end = Some(end);
                    b.labels = labels;
                }
                let mut prev = b.end.clone().unwrap();
                let mut future: Vec<(u64, u64, u8)> = b.labels.as_ref().clone();
                for (k, e) in l.queued.iter_mut().enumerate() {
                    let Ok(start) = prev.clone() else {
                        UNWALKED.fetch_add(1, Ordering::Relaxed);
                        break;
                    };
                    if e.end.is_none() {
                        let (end, labels) = walk(&start, &e.bufs);
                        e.end = Some(end);
                        e.labels = labels;
                    }
                    // (The queue's head is the main command processor's: it records that one
                    // itself while the scouts record the ones behind it.)
                    if e.slot.is_none() && k >= 1 {
                        let slot = Arc::new(Slot { turn: Mutex::new(None), cv: Condvar::new(), journal: AtomicU64::new(0) });
                        e.slot = Some(slot.clone());
                        job = Some(Job { index: e.index, bufs: e.bufs.clone(), at: e.at, start, future: Arc::new(future), slot, ahead: k });
                        break;
                    }
                    future.extend(e.labels.iter().copied());
                    prev = e.end.clone().unwrap();
                }
            }
            if job.is_some() {
                return job;
            }
        }
        CV.wait_for(&mut g, std::time::Duration::from_millis(200));
    }
}

fn run(name: &str) {
    let mut cp = Cp::new(name, false);
    loop {
        super::prio::apply();
        let Some(job) = take_job() else { return };
        CLAIMED.fetch_add(1, Ordering::Relaxed);
        AHEAD.fetch_add(job.ahead as u64, Ordering::Relaxed);
        // (The main command processor reached it before the scout began.)
        if job.slot.turn.lock().is_some() {
            MISSED.fetch_add(1, Ordering::Relaxed);
            continue;
        }
        let jid = super::exec::scout_journal_begin();
        job.slot.journal.store(jid, Ordering::Release);
        // (A turn given between the look above and the journal's id.)
        if job.slot.turn.lock().is_some() {
            super::exec::scout_journal_freeze(jid);
        }
        let t0 = Instant::now();
        let tl = super::timeline::mark();
        let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| cp.scout_run(&job.start, job.index, job.at, &job.bufs, job.future.clone()))).unwrap_or(None);
        let done = Instant::now();
        super::timeline::end(name, tl, || format!("scout: sub {}", job.index));
        // Its turn: the main command processor reaches the submission.
        let turn = {
            let mut g = job.slot.turn.lock();
            loop {
                if let Some(t) = g.take() {
                    break Some(t);
                }
                if super::shutting_down() {
                    break None;
                }
                job.slot.cv.wait_for(&mut g, std::time::Duration::from_millis(100));
            }
        };
        let (writes, who) = super::exec::scout_journal_end(jid);
        if let (Some(out), Some(turn)) = (out.as_ref(), turn.as_ref()) {
            note(&cp, &job, out, turn, t0, done, &writes, &who);
        }
        cp.scout_drop();
    }
}

#[allow(clippy::too_many_arguments)]
fn note(cp: &Cp, job: &Job, out: &super::cp::ScoutOut, turn: &Turn, t0: Instant, done: Instant, writes: &super::ranges::Ranges, who: &[(u64, u64, String)]) {
    RECORDED.fetch_add(1, Ordering::Relaxed);
    REC_NS.fetch_add((done - t0).as_nanos() as u64, Ordering::Relaxed);
    COMMANDS.fetch_add(out.commands, Ordering::Relaxed);
    RUNS.fetch_add(out.runs as u64, Ordering::Relaxed);
    BATCHES.fetch_add(out.batches as u64, Ordering::Relaxed);
    CHUNK_BATCHES.fetch_add(out.chunk_batches as u64, Ordering::Relaxed);
    READS.fetch_add(out.reads as u64, Ordering::Relaxed);
    for (c, w) in WAITS.iter().zip(out.waits) {
        c.fetch_add(w, Ordering::Relaxed);
    }
    // How far it got: the dwords before its end of the submission's.
    let total: u64 = job.bufs.iter().map(|b| b.1 as u64).sum();
    let (why, at, buf) = out.ended;
    let last = job.bufs.len() - 1;
    let whole = why == "buffer end" && buf == last;
    let before: u64 = job.bufs.iter().take(buf).map(|b| b.1 as u64).sum::<u64>() + job.bufs.get(buf).map_or(0, |b| if at >= b.0 { ((at - b.0) / 4).min(b.1 as u64) } else { 0 });
    let got = if whole { total } else { before.min(total) };
    DWORDS.fetch_add(total, Ordering::Relaxed);
    DWORDS_DONE.fetch_add(got, Ordering::Relaxed);
    if whole {
        WHOLE.fetch_add(1, Ordering::Relaxed);
    }
    {
        let mut g = ENDS.lock();
        match g.iter_mut().find(|e| e.0 == why) {
            Some(e) => {
                e.1 += 1;
                e.2 += total - got;
            }
            None => g.push((why, 1, total - got)),
        }
    }
    if let Some((_, writer)) = &out.gate {
        bump(&GATES, writer);
    }
    if done <= turn.at {
        IN_TIME.fetch_add(1, Ordering::Relaxed);
        EARLY_NS.fetch_add((turn.at - done).as_nanos() as u64, Ordering::Relaxed);
    } else {
        LATE_NS.fetch_add((done - turn.at).as_nanos() as u64, Ordering::Relaxed);
    }
    // The checks.
    CHECKS[0].fetch_add(1, Ordering::Relaxed);
    let start_ok = Cp::walk_states_agree(&job.start, &turn.state);
    if !start_ok {
        CHECKS[1].fetch_add(1, Ordering::Relaxed);
        if LOGGED.fetch_add(1, Ordering::Relaxed) < 24 {
            let (regs, fields) = job.start.diff(&turn.state);
            crate::info!("gpu[{}]: scout census: submission {} started from a state that differs from the command processor's: {} registers (first {:?}; space 0 context, 1 SH, 2 uconfig), fields and stack depth agree {fields}", cp.name, job.index, regs.len(), &regs[..regs.len().min(6)]);
        }
    }
    let c = cp.scout_check(writes, who);
    if c.conflicts > 0 {
        CHECKS[2].fetch_add(1, Ordering::Relaxed);
        CHECKS[3].fetch_add(c.conflicts as u64, Ordering::Relaxed);
        for (kind, _, _, by) in &c.first {
            let by = by.split(" [").next().unwrap_or("").to_string();
            let by = if by.contains("-par") { by.split("-par").next().unwrap_or("").to_string() + " workers" } else { by };
            bump(&CONFLICTS, &format!("{kind} written by {}", if by.is_empty() { "?" } else { &by }));
        }
        if LOGGED.fetch_add(1, Ordering::Relaxed) < 40 {
            let l: Vec<String> = c.first.iter().map(|(k, a, n, by)| format!("{k} {a:#x}+{n:#x} written by {by}")).collect();
            crate::info!("gpu[{}]: scout census: submission {} ({} dwords, {} ahead): {} of its {} CPU reads were written before its turn: {}", cp.name, job.index, total, job.ahead, c.conflicts, out.reads, l.join("; "));
        }
    }
    if c.images {
        CHECKS[4].fetch_add(1, Ordering::Relaxed);
    }
    if start_ok && c.conflicts == 0 && !c.images {
        CHECKS[5].fetch_add(1, Ordering::Relaxed);
        if whole {
            CHECKS[6].fetch_add(1, Ordering::Relaxed);
        }
        CHECKS[7].fetch_add(got, Ordering::Relaxed);
    }
    if c.conflicts > 0 && c.conflicts == c.textures {
        CHECKS[8].fetch_add(1, Ordering::Relaxed);
    }
    if start_ok && c.conflicts == c.textures && !c.images {
        CHECKS[9].fetch_add(1, Ordering::Relaxed);
        if whole {
            CHECKS[10].fetch_add(1, Ordering::Relaxed);
        }
        CHECKS[11].fetch_add(got, Ordering::Relaxed);
    }
}

/// The census's status lines ("" without scouts).
pub fn status() -> String {
    if !on() {
        return String::new();
    }
    let v = |c: &AtomicU64| c.load(Ordering::Relaxed);
    let rec = v(&RECORDED).max(1);
    let list = |l: &Mutex<Vec<(String, u64)>>| {
        let mut g = l.lock().clone();
        g.sort_by(|a, b| b.1.cmp(&a.1));
        g.iter().take(10).map(|(k, n)| format!("{k} {n}")).collect::<Vec<_>>().join(", ")
    };
    let mut ends = ENDS.lock().clone();
    ends.sort_by(|a, b| b.1.cmp(&a.1));
    let ends: Vec<String> = ends.iter().take(12).map(|(w, n, left)| format!("{w} {n} ({} K dwords left)", left / 1000)).collect();
    let mut werr = WALK_ERRS.lock().clone();
    werr.sort_by(|a, b| b.1.cmp(&a.1));
    let werr: Vec<String> = werr.iter().map(|(w, n)| format!("{w} {n}")).collect();
    format!(
        "    scouts (census, `tune scouts` {}): {} submissions claimed ({:.1} ahead of the command processor on average; {} reached by it before their scout began), {} recorded in {} ms ({:.2} ms each): {} whole, {:.0}% of their dwords ({} K of {} K); records ended by: {}; {} done before their turn ({:.2} ms early on average), the others {:.2} ms late; a record: {:.1} commands of its own, {:.2} held runs, {:.1} batches, {:.1} chunk batches, {:.0} CPU reads; waits passed on memory {}, a promise {}, its own held write {}, an earlier queued submission's write {}, gates {} (by the label's last writer: {})\n    scouts' checks at the turn: {} records, start state differed {}, CPU reads written meanwhile in {} ({} reads; first by kind and writer: {}), images with clears or reloads pending {}; textures bound {} ({} read from guest memory); every check passed {} ({} whole submissions, {} K dwords: {:.0}% of the claimed submissions' dwords); with textures' reads left out ({} records had only those) {} ({} whole, {} K dwords: {:.0}%); the walk stopped at {} submissions ({})\n",
        count(),
        v(&CLAIMED),
        v(&AHEAD) as f64 / v(&CLAIMED).max(1) as f64,
        v(&MISSED),
        v(&RECORDED),
        v(&REC_NS) / 1_000_000,
        v(&REC_NS) as f64 / 1e6 / rec as f64,
        v(&WHOLE),
        100.0 * v(&DWORDS_DONE) as f64 / v(&DWORDS).max(1) as f64,
        v(&DWORDS_DONE) / 1000,
        v(&DWORDS) / 1000,
        ends.join(", "),
        v(&IN_TIME),
        v(&EARLY_NS) as f64 / 1e6 / v(&IN_TIME).max(1) as f64,
        v(&LATE_NS) as f64 / 1e6 / (rec - v(&IN_TIME).min(rec)).max(1) as f64,
        v(&COMMANDS) as f64 / rec as f64,
        v(&RUNS) as f64 / rec as f64,
        v(&BATCHES) as f64 / rec as f64,
        v(&CHUNK_BATCHES) as f64 / rec as f64,
        v(&READS) as f64 / rec as f64,
        v(&WAITS[0]),
        v(&WAITS[1]),
        v(&WAITS[2]),
        v(&WAITS[3]),
        v(&WAITS[4]),
        list(&GATES),
        v(&CHECKS[0]),
        v(&CHECKS[1]),
        v(&CHECKS[2]),
        v(&CHECKS[3]),
        list(&CONFLICTS),
        v(&CHECKS[4]),
        v(&super::exec::SCOUT_TEXTURES[0]),
        v(&super::exec::SCOUT_TEXTURES[1]),
        v(&CHECKS[5]),
        v(&CHECKS[6]),
        v(&CHECKS[7]) / 1000,
        100.0 * v(&CHECKS[7]) as f64 / v(&DWORDS).max(1) as f64,
        v(&CHECKS[8]),
        v(&CHECKS[9]),
        v(&CHECKS[10]),
        v(&CHECKS[11]) / 1000,
        100.0 * v(&CHECKS[11]) as f64 / v(&DWORDS).max(1) as f64,
        v(&UNWALKED),
        werr.join(", ")
    )
}
