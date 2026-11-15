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

/// Scouts wanted (`tune scouts`; at most 8).
pub fn count() -> usize {
    (super::tune::scouts() as usize).min(8)
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

/// A scout's seat: a command processor with an executor and a worker pool of its own. It
/// records one submission and keeps the record (its held batches, its executor's notes)
/// until the main command processor has taken it over, or left it and its scout dropped it.
/// A scout has several (`tune scout_seats`): it goes on to the next submission while its
/// records wait for their turns (with one seat it idled until each turn, and the main command
/// processor found less than half the submissions recorded: run 20261115-000926).
struct Seat {
    cp: Mutex<Cp>,
    /// A record lives in it: set by its scout when it claims a submission for it, cleared by
    /// the main command processor when it took the record over or by the scout when it
    /// dropped it.
    busy: AtomicBool,
}

/// A claimed submission: its record in the making or made, and what the main command
/// processor leaves for its scout when it reaches the submission without taking the record
/// over.
pub struct Slot {
    seat: Arc<Seat>,
    index: u64,
    bufs: Vec<(u64, u32)>,
    /// When the title queued it.
    at: Instant,
    /// The state its record starts from (walked).
    start: Arc<WalkState>,
    /// Submissions between the main command processor's and this one when it was claimed.
    ahead: usize,
    /// What its scout made of it (None while it records, and once taken).
    made: Mutex<Option<Made>>,
    made_cv: Condvar,
    done: AtomicBool,
    /// The main command processor's turn, when it left the record to its scout (the census,
    /// a record not finished, a hand-over refused): the scout notes and drops it (`sweep`).
    turn: Mutex<Option<Turn>>,
    /// The scout's journal (`exec::scout_journal_begin`; 0: not begun, or ended).
    journal: AtomicU64,
    /// The ranges of the walked copies its view started from (`Cp::scout_run`; None: not
    /// walked yet).
    seeds: Mutex<Option<Vec<(u64, usize)>>>,
    /// The main command processor went on without the record (`Cp::held_halted`).
    cancel: Arc<AtomicBool>,
}

/// A finished record: what `Cp::scout_run` returned (None: nothing recorded), the forward
/// view it read through (parked: the thread that commits or drops it takes it), its times.
struct Made {
    out: Option<super::cp::ScoutOut>,
    view: Option<super::forward::Parked>,
    t0: Instant,
    done: Instant,
}

/// What the main command processor does with a submission after its scout's turn (`turn`).
#[derive(Clone, Copy)]
pub enum Handed {
    /// It runs the submission itself (no scout, the census, or a record refused).
    No,
    /// The scout's record was the whole submission.
    Whole,
    /// The record ended before the submission's end: on from this buffer and packet.
    From(usize, u64),
}

struct Turn {
    /// The main command processor's own state at the submission's start.
    state: Arc<WalkState>,
    at: Instant,
    /// What a reader takes at the scout's seeds' ranges now (`forward::global_words`; None:
    /// the turn came before the scout had walked them).
    seed_now: Option<Vec<Option<Vec<u32>>>>,
    /// The census's turn (`tune scouts_commit` 0, or the main command processor held work of
    /// its own): the record is checked and counted before it is dropped.
    census: bool,
    /// The record's own numbers were counted at the turn (a hand-over refused).
    noted: bool,
}

struct Job {
    slot: Arc<Slot>,
    future: Arc<Vec<(u64, u64, u8)>>,
    /// The submissions before it that are not recorded for sure (the one
    /// the main command processor runs, the queued ones between): each
    /// one's start state and buffers, in order (`DS_SCOUT_SEED=1`).
    pre: Vec<(Arc<WalkState>, Vec<(u64, u32)>)>,
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
/// submissions, their dwords, records with walked copies in their view, of
/// them records whose walked copies differ from what a reader takes at the
/// turn, records whose turn came before they were walked].
static CHECKS: [AtomicU64; 15] = [const { AtomicU64::new(0) }; 15];
static UNWALKED: AtomicU64 = AtomicU64::new(0);
static MISSED: AtomicU64 = AtomicU64::new(0);
static ENDS: Mutex<Vec<(&'static str, u64, u64)>> = Mutex::new(Vec::new());
static GATES: Mutex<Vec<(String, u64)>> = Mutex::new(Vec::new());
static CONFLICTS: Mutex<Vec<(String, u64)>> = Mutex::new(Vec::new());
static WALK_ERRS: Mutex<Vec<(&'static str, u64)>> = Mutex::new(Vec::new());
static LOGGED: AtomicU64 = AtomicU64::new(0);
/// Hand-overs (`tune scouts_commit`): [turns with a hand-over asked, records taken over
/// whole, taken over as a prefix, refused, dwords taken over, dwords of the submissions asked
/// for, ns the main command processor waited for its scouts, ns of the scouts' commits]; the
/// refusals by reason.
static HAND: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];
static REFUSED: Mutex<Vec<(&'static str, u64)>> = Mutex::new(Vec::new());
/// Records not finished when the main command processor reached their submission with a
/// hand-over wanted (it went on without them: `tune scouts_wait`).
static LATE: AtomicU64 = AtomicU64::new(0);
/// Gates (`Cp::scout_gate`): [records taken over up to a gate, segments behind gates taken
/// over, segments refused at their gate (the rest of the record dropped)].
static GATED: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// The record the main command processor holds at a gate (`turn`, `more`): taken over up to
/// there, the rest waits for the gate's wait to pass.
struct Pending {
    slot: Arc<Slot>,
    made: Made,
    bufs: Vec<(u64, u32)>,
    state: Arc<WalkState>,
}

static PENDING: Mutex<Option<Pending>> = Mutex::new(None);

fn bump<K: PartialEq + Clone>(l: &Mutex<Vec<(K, u64)>>, k: &K) {
    let mut g = l.lock();
    match g.iter_mut().find(|e| e.0 == *k) {
        Some(e) => e.1 += 1,
        None => g.push((k.clone(), 1)),
    }
}

/// A new window for the census (`profile reset`).
pub fn reset_stats() {
    for c in [&CLAIMED, &RECORDED, &WHOLE, &IN_TIME, &LATE_NS, &EARLY_NS, &REC_NS, &COMMANDS, &RUNS, &BATCHES, &CHUNK_BATCHES, &READS, &AHEAD, &DWORDS, &DWORDS_DONE, &UNWALKED, &MISSED, &LATE] {
        c.store(0, Ordering::Relaxed);
    }
    for c in WAITS.iter().chain(CHECKS.iter()).chain(HAND.iter()).chain(GATED.iter()) {
        c.store(0, Ordering::Relaxed);
    }
    REFUSED.lock().clear();
    super::cp::SCOUT_REFUSED_READS.lock().clear();
    ENDS.lock().clear();
    GATES.lock().clear();
    CONFLICTS.lock().clear();
    WALK_ERRS.lock().clear();
    LOGGED.store(0, Ordering::Relaxed);
    for c in super::exec::SCOUT_TEXTURES.iter().chain(super::forward::VIEW_REFUSALS.iter()) {
        c.store(0, Ordering::Relaxed);
    }
    super::forward::CP_WRITE_ENTRIES.store(0, Ordering::Relaxed);
    super::forward::WALKED_ENTRIES.store(0, Ordering::Relaxed);
    for c in super::forward::FROZEN_STATS.iter() {
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
    crate::info!("gpu: {n} scout(s) record the graphics queue's submissions ahead (`tune scouts_commit` {}: 0 the census, checked at their turn and dropped; 1 handed over; docs/SCOUTS.md)", super::tune::scouts_commit() as u8);
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

/// The main command processor takes submission `index` (about to run it): its own state
/// there becomes the walk's base, and the submission's record, if a scout made one, has its
/// turn. With `tune scouts_commit` a finished record is taken over when its checks pass: the
/// main command processor's work so far goes to the GPU first (`Cp::scout_turn`), then it
/// checks and commits the record itself, on its own thread, with the seat's command
/// processor (`Cp::scout_commit`; the scout is busy with a later submission), and goes on
/// from where the record ended (`Cp::scout_adopt`). Otherwise the record is left to its scout
/// (the census's notes, its drop) and the main command processor records the submission.
pub fn turn(index: u64, bufs: &[(u64, u32)], cp: &mut Cp) -> Handed {
    if !on() {
        return Handed::No;
    }
    let state = Arc::new(cp.walk_start());
    let slot = {
        let mut g = LANES.lock();
        let Some(l) = g.as_mut() else { return Handed::No };
        // (Entries before it: submissions the queue's worker ran another way, a
        // replay's control entries.)
        while l.queued.front().is_some_and(|e| e.index < index) {
            let e = l.queued.pop_front().unwrap();
            if let Some(s) = e.slot {
                leave(&s, &state, false, false);
            }
        }
        let slot = match l.queued.front() {
            Some(e) if e.index == index => l.queued.pop_front().and_then(|e| e.slot),
            _ => None,
        };
        l.base = Some(Base { bufs: bufs.to_vec(), state: state.clone(), end: None, labels: Arc::new(Vec::new()) });
        slot
    };
    let Some(slot) = slot else {
        CV.notify_all();
        return Handed::No;
    };
    if !super::tune::scouts_commit() {
        leave(&slot, &state, true, false);
        return Handed::No;
    }
    // A record its scout has not finished is not waited for (`tune scouts_wait`: or only so
    // long): the scout is told to stop, and the main command processor records the submission.
    if !slot.done.load(Ordering::Acquire) {
        let wait = super::tune::scouts_wait();
        if wait > 0 {
            let until = Instant::now() + std::time::Duration::from_micros(wait);
            let mut g = slot.made.lock();
            while !slot.done.load(Ordering::Acquire) && Instant::now() < until && !super::shutting_down() {
                slot.made_cv.wait_until(&mut g, until);
            }
        }
        if !slot.done.load(Ordering::Acquire) {
            LATE.fetch_add(1, Ordering::Relaxed);
            slot.cancel.store(true, Ordering::Relaxed);
            leave(&slot, &state, false, false);
            return Handed::No;
        }
    }
    // (Its own work to the GPU first: the record is checked against it and follows it. None:
    // it holds work back itself, a stretch past a run or a label wait: nothing to hand over.)
    let Some((need, last, cpu)) = cp.scout_turn() else {
        leave(&slot, &state, true, false);
        return Handed::No;
    };
    let Some(mut made) = slot.made.lock().take() else {
        leave(&slot, &state, false, false);
        return Handed::No;
    };
    let Some(out) = made.out.as_ref() else {
        *slot.made.lock() = Some(made);
        leave(&slot, &state, false, false);
        return Handed::No;
    };
    HAND[0].fetch_add(1, Ordering::Relaxed);
    HAND[5].fetch_add(bufs.iter().map(|b| b.1 as u64).sum(), Ordering::Relaxed);
    let t = Instant::now();
    note_record(&slot, out, t, made.t0, made.done);
    let tl = super::timeline::mark();
    // Its journal: everything written since its scout began. It ends here, unless the record
    // has gates (its later segments are checked against what is written until they pass).
    let writes = if out.gates > 0 { super::exec::scout_journal_snapshot(slot.journal.load(Ordering::Acquire)) } else { super::exec::scout_journal_end(slot.journal.swap(0, Ordering::AcqRel)).0 };
    let seeds_differ = !out.seeds.is_empty() && out.seeds.iter().any(|x| super::forward::global_words(x.0, x.1.len()).as_deref() != Some(&x.1[..]));
    let r = if !Cp::walk_states_agree(&slot.start, &state) {
        Err("start state")
    } else if seeds_differ {
        Err("walked copies")
    } else {
        let mut scp = slot.seat.cp.lock();
        // (The view its record read through: this thread's for the commit, which publishes it.)
        if let Some(v) = made.view.take() {
            super::forward::unpark(v);
        }
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| scp.scout_commit(need, (last, cpu), &writes))).unwrap_or(Err("failed"));
        // (Refused, or more of it waits behind a gate: its view stays with it.)
        if !matches!(&r, Ok(end) if !end.gate) {
            made.view = super::forward::park();
        }
        r
    };
    HAND[7].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    super::timeline::end(&cp.name, tl, || format!("scout's record of sub {index}: {}", match &r {
        Ok(_) => "taken over",
        Err(e) => e,
    }));
    match r {
        Ok(end) => {
            let (stop, gate) = (end.stop, end.gate);
            cp.scout_adopt(end);
            if gate {
                // (The rest of the record waits behind its gate: `more`.)
                GATED[0].fetch_add(1, Ordering::Relaxed);
                cp.scout_gate = stop.map(|x| x.1);
                *PENDING.lock() = Some(Pending { slot: slot.clone(), made, bufs: bufs.to_vec(), state: state.clone() });
            } else {
                // (Its seat is free again.)
                slot.seat.busy.store(false, Ordering::Release);
                CV.notify_all();
            }
            match stop {
                None => {
                    HAND[1].fetch_add(1, Ordering::Relaxed);
                    HAND[4].fetch_add(bufs.iter().map(|b| b.1 as u64).sum(), Ordering::Relaxed);
                    Handed::Whole
                }
                Some((buf, at)) => {
                    HAND[2].fetch_add(1, Ordering::Relaxed);
                    let before: u64 = bufs.iter().take(buf).map(|b| b.1 as u64).sum::<u64>() + bufs.get(buf).map_or(0, |b| if at >= b.0 { ((at - b.0) / 4).min(b.1 as u64) } else { 0 });
                    HAND[4].fetch_add(before, Ordering::Relaxed);
                    Handed::From(buf, at)
                }
            }
        }
        Err(why) => {
            HAND[3].fetch_add(1, Ordering::Relaxed);
            bump(&REFUSED, &why);
            *slot.made.lock() = Some(made);
            leave(&slot, &state, false, true);
            Handed::No
        }
    }
}

/// The main command processor passed the gate of the record it holds (`Pending`; `Cp::run`
/// calls this behind the gate's wait): the record's next segment is checked and committed as
/// its first was at the turn, and the main command processor goes on from where that one
/// ends: the buffer and packet returned (the next gate's wait, the record's stop, or the
/// submission's end). None: the segment was refused (the rest of the record is dropped; the
/// main command processor records on behind the wait itself), or no record waits.
pub fn more(cp: &mut Cp) -> Option<(usize, u64)> {
    let Pending { slot, mut made, bufs, state } = PENDING.lock().take()?;
    let t = Instant::now();
    let tl = super::timeline::mark();
    let r = match cp.scout_turn() {
        None => Err("the command processor held work at its gate"),
        Some((need, last, cpu)) => {
            let writes = super::exec::scout_journal_snapshot(slot.journal.load(Ordering::Acquire));
            let mut scp = slot.seat.cp.lock();
            if let Some(v) = made.view.take() {
                super::forward::unpark(v);
            }
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| scp.scout_commit(need, (last, cpu), &writes))).unwrap_or(Err("failed"));
            match &r {
                // (More of it waits behind the next gate: its view stays with it.)
                Ok(end) if end.gate => made.view = super::forward::park(),
                Ok(_) => {}
                // (What it committed before ends the published copies under its writes; its
                // own are forgotten.)
                Err(_) => super::forward::hold_end_scout(&scp.scout_all_writes()),
            }
            r
        }
    };
    HAND[7].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    super::timeline::end(&cp.name, tl, || format!("scout's record of sub {} behind a gate: {}", slot.index, match &r {
        Ok(_) => "taken over",
        Err(e) => e,
    }));
    match r {
        Ok(end) => {
            let (stop, gate) = (end.stop, end.gate);
            cp.scout_adopt(end);
            GATED[1].fetch_add(1, Ordering::Relaxed);
            // (Where it goes on: the record's stop, or the submission's end.)
            let to = stop.unwrap_or_else(|| {
                let k = bufs.len().saturating_sub(1);
                (k, bufs.get(k).map_or(0, |b| b.0 + 4 * b.1 as u64))
            });
            if gate {
                cp.scout_gate = Some(to.1);
                *PENDING.lock() = Some(Pending { slot, made, bufs, state });
            } else {
                let _ = super::exec::scout_journal_end(slot.journal.swap(0, Ordering::AcqRel));
                slot.seat.busy.store(false, Ordering::Release);
                CV.notify_all();
            }
            Some(to)
        }
        Err(why) => {
            GATED[2].fetch_add(1, Ordering::Relaxed);
            bump(&REFUSED, &why);
            *slot.made.lock() = Some(made);
            leave(&slot, &state, false, true);
            None
        }
    }
}

/// The main command processor leaves the record it holds at a gate without taking more of it
/// (it could not wait there plainly, a shutdown): what was committed stays, the rest is
/// dropped by its scout.
pub fn abandon() {
    let Some(Pending { slot, mut made, state, .. }) = PENDING.lock().take() else { return };
    if let Some(v) = made.view.take() {
        super::forward::unpark(v);
    }
    let writes = slot.seat.cp.lock().scout_all_writes();
    super::forward::hold_end_scout(&writes);
    GATED[2].fetch_add(1, Ordering::Relaxed);
    *slot.made.lock() = Some(made);
    leave(&slot, &state, false, true);
}

/// The main command processor leaves a record to its scout at its turn (`Turn`): the
/// record's journal takes no more writes, and the scout notes and drops it (`sweep`).
fn leave(s: &Slot, state: &Arc<WalkState>, census: bool, noted: bool) {
    let j = s.journal.load(Ordering::Acquire);
    if j != 0 {
        super::exec::scout_journal_freeze(j);
    }
    let seed_now = s.seeds.lock().as_ref().map(|v| v.iter().map(|&(b, n)| super::forward::global_words(b, n)).collect());
    *s.turn.lock() = Some(Turn { state: state.clone(), at: Instant::now(), seed_now, census, noted });
    CV.notify_all();
}

/// Whether scouts walk the submissions before their own with effects (`Cp::scout_run`,
/// `DS_SCOUT_SEED=1`): only then are those submissions' buffers handed to them.
fn scout_seed_wanted() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_SCOUT_SEED").ok().as_deref() == Some("1"))
}

/// Walks `bufs` from `from`: the state at their end and the label writes
/// they make.
fn walk(from: &WalkState, bufs: &[(u64, u32)]) -> (Result<Arc<WalkState>, &'static str>, Arc<Vec<(u64, u64, u8)>>) {
    let mut w = from.clone();
    w.labels.clear();
    w.flip = None;
    match Cp::walk_submission(&mut w, bufs, false) {
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

/// A queued submission for `seat` (free): the newest no scout has claimed whose start state
/// the walk reaches (`tune scouts_far`; else the oldest behind the queue's head). None: none
/// now (waited a moment for one).
fn take_job(seat: &Arc<Seat>) -> Option<Job> {
    let mut g = LANES.lock();
    for round in 0..2 {
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
                // Every queued submission's end state, as far as the walk gets; the one to take:
                // the newest unclaimed one (`tune scouts_far`: the most time before its turn),
                // or the oldest behind the queue's head (the head is the main command
                // processor's: it records that one itself meanwhile).
                let far = super::tune::scouts_far();
                let mut prev = b.end.clone().unwrap();
                let mut pick: Option<usize> = None;
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
                    if e.slot.is_none() && k >= 1 {
                        pick = Some(k);
                        if !far {
                            break;
                        }
                    }
                    prev = e.end.clone().unwrap();
                }
                if let Some(k) = pick {
                    let mut future: Vec<(u64, u64, u8)> = b.labels.as_ref().clone();
                    let mut pre = vec![(b.state.clone(), b.bufs.clone())];
                    let mut start = b.end.clone().unwrap();
                    for e in l.queued.iter().take(k) {
                        if scout_seed_wanted() {
                            if let Ok(s) = &start {
                                pre.push((s.clone(), e.bufs.clone()));
                            }
                        }
                        future.extend(e.labels.iter().copied());
                        start = e.end.clone().unwrap();
                    }
                    if let (Ok(start), Some(e)) = (start, l.queued.get_mut(k)) {
                        seat.busy.store(true, Ordering::Release);
                        let slot = Arc::new(Slot {
                            seat: seat.clone(),
                            index: e.index,
                            bufs: e.bufs.clone(),
                            at: e.at,
                            start,
                            ahead: k,
                            made: Mutex::new(None),
                            made_cv: Condvar::new(),
                            done: AtomicBool::new(false),
                            turn: Mutex::new(None),
                            journal: AtomicU64::new(0),
                            seeds: Mutex::new(None),
                            cancel: Arc::new(AtomicBool::new(false)),
                        });
                        e.slot = Some(slot.clone());
                        job = Some(Job { slot, future: Arc::new(future), pre });
                    }
                }
            }
            if job.is_some() {
                return job;
            }
        }
        if round == 0 {
            CV.wait_for(&mut g, std::time::Duration::from_millis(50));
        }
    }
    None
}

/// Seats a scout has (`tune scout_seats`, 1 to 8).
fn seats() -> usize {
    (super::tune::scout_seats() as usize).clamp(1, 8)
}

/// A scout's thread: it records queued submissions into its free seats, one after the
/// other, and finishes with the records the main command processor left (`sweep`).
fn run(name: &str) {
    let seats: Vec<Arc<Seat>> = (0..seats()).map(|i| Arc::new(Seat { cp: Mutex::new(Cp::new(&format!("{name}{}", (b'a' + i as u8) as char), false)), busy: AtomicBool::new(false) })).collect();
    // Its records the main command processor has neither taken over nor left yet.
    let mut mine: Vec<Arc<Slot>> = Vec::new();
    loop {
        super::prio::apply();
        if super::shutting_down() {
            return;
        }
        sweep(&mut mine);
        let Some(seat) = seats.iter().find(|x| !x.busy.load(Ordering::Acquire)) else {
            // (Every seat holds a record that waits for its turn.)
            let mut g = LANES.lock();
            CV.wait_for(&mut g, std::time::Duration::from_millis(20));
            continue;
        };
        let Some(job) = take_job(seat) else { continue };
        let slot = job.slot.clone();
        CLAIMED.fetch_add(1, Ordering::Relaxed);
        AHEAD.fetch_add(slot.ahead as u64, Ordering::Relaxed);
        let t0 = Instant::now();
        // (The main command processor reached it before the scout began: nothing recorded.)
        if slot.turn.lock().is_some() {
            MISSED.fetch_add(1, Ordering::Relaxed);
            *slot.made.lock() = Some(Made { out: None, view: None, t0, done: t0 });
            slot.done.store(true, Ordering::Release);
            mine.push(slot);
            continue;
        }
        let jid = super::exec::scout_journal_begin(&seat.cp.lock().name);
        slot.journal.store(jid, Ordering::Release);
        // (A turn left between the look above and the journal's id.)
        if slot.turn.lock().is_some() {
            super::exec::scout_journal_freeze(jid);
        }
        let tl = super::timeline::mark();
        let (out, view) = {
            let mut cp = seat.cp.lock();
            let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                cp.scout_run(&slot.start, slot.index, slot.at, &slot.bufs, job.future.clone(), &job.pre, &|ranges| *slot.seeds.lock() = Some(ranges), slot.cancel.clone())
            }))
            .unwrap_or(None);
            // (Its view leaves this thread with the record: the next record's begins.)
            (out, super::forward::park())
        };
        let done = Instant::now();
        super::timeline::end(name, tl, || format!("scout: sub {}", slot.index));
        {
            let mut g = slot.made.lock();
            *g = Some(Made { out, view, t0, done });
            slot.done.store(true, Ordering::Release);
        }
        slot.made_cv.notify_all();
        mine.push(slot);
    }
}

/// A scout finishes with its records the main command processor is done with: one taken over
/// left its seat free already; one left to the scout (`Turn`) is counted (the census: checked
/// as at a hand-over) and dropped, and its seat is free again.
fn sweep(mine: &mut Vec<Arc<Slot>>) {
    mine.retain(|slot| {
        if !slot.seat.busy.load(Ordering::Acquire) {
            return false;
        }
        let Some(turn) = slot.turn.lock().take() else { return true };
        let made = slot.made.lock().take();
        let (writes, who) = super::exec::scout_journal_end(slot.journal.swap(0, Ordering::AcqRel));
        let mut cp = slot.seat.cp.lock();
        if let Some(Made { out: Some(out), t0, done, .. }) = made.as_ref() {
            if !turn.noted {
                note_record(slot, out, turn.at, *t0, *done);
            }
            if turn.census {
                note_checks(&cp, slot, out, &turn, &writes, &who);
            }
        }
        // (Its parked view goes with it.)
        drop(made);
        cp.scout_drop();
        slot.seat.busy.store(false, Ordering::Release);
        false
    });
}

/// How far a record got: the dwords of its submission, those before its end, whether it is
/// the whole submission.
fn progress(job: &Slot, out: &super::cp::ScoutOut) -> (u64, u64, bool) {
    let total: u64 = job.bufs.iter().map(|b| b.1 as u64).sum();
    let (why, at, buf) = out.ended;
    let whole = why == "buffer end" && buf == job.bufs.len() - 1;
    let before: u64 = job.bufs.iter().take(buf).map(|b| b.1 as u64).sum::<u64>() + job.bufs.get(buf).map_or(0, |b| if at >= b.0 { ((at - b.0) / 4).min(b.1 as u64) } else { 0 });
    (total, if whole { total } else { before.min(total) }, whole)
}

/// A record's own numbers (the census and hand-overs alike).
fn note_record(job: &Slot, out: &super::cp::ScoutOut, turn_at: Instant, t0: Instant, done: Instant) {
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
    let (total, got, whole) = progress(job, out);
    let why = out.ended.0;
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
    if done <= turn_at {
        IN_TIME.fetch_add(1, Ordering::Relaxed);
        EARLY_NS.fetch_add((turn_at - done).as_nanos() as u64, Ordering::Relaxed);
    } else {
        LATE_NS.fetch_add((done - turn_at).as_nanos() as u64, Ordering::Relaxed);
    }
}

/// The census's checks of a record at its turn (counted; the record is dropped after them).
fn note_checks(cp: &Cp, job: &Slot, out: &super::cp::ScoutOut, turn: &Turn, writes: &super::ranges::Ranges, who: &[(u64, u64, String)]) {
    let (total, got, whole) = progress(job, out);
    CHECKS[0].fetch_add(1, Ordering::Relaxed);
    let start_ok = Cp::walk_states_agree(&job.start, &turn.state);
    if !start_ok {
        CHECKS[1].fetch_add(1, Ordering::Relaxed);
        if LOGGED.fetch_add(1, Ordering::Relaxed) < 24 {
            let (regs, fields) = job.start.diff(&turn.state);
            crate::info!("gpu[{}]: scout census: submission {} started from a state that differs from the command processor's: {} registers (first {:?}; space 0 context, 1 SH, 2 uconfig), fields and stack depth agree {fields}", cp.name, job.index, regs.len(), &regs[..regs.len().min(6)]);
        }
    }
    // The walked copies its view started from, against what a reader takes now.
    let mut seeds_ok = true;
    if !out.seeds.is_empty() {
        CHECKS[12].fetch_add(1, Ordering::Relaxed);
        match turn.seed_now.as_ref() {
            Some(now) if now.len() == out.seeds.len() => {
                let bad = out.seeds.iter().zip(now).filter(|(s, n)| n.as_deref() != Some(&s.1[..])).count();
                if bad > 0 {
                    seeds_ok = false;
                    CHECKS[13].fetch_add(1, Ordering::Relaxed);
                    if LOGGED.fetch_add(1, Ordering::Relaxed) < 40 {
                        let first = out.seeds.iter().zip(now).find(|(s, n)| n.as_deref() != Some(&s.1[..]));
                        let d = first.map(|(s, n)| {
                            let k = n.as_ref().and_then(|n| n.iter().zip(s.1.iter()).position(|(a, b)| a != b));
                            format!("{:#x}+{:#x}: first word that differs {:?} (walked {:#x?}, now {:#x?})", s.0, 4 * s.1.len(), k, k.map(|k| s.1[k]), k.and_then(|k| n.as_ref().map(|n| n[k])))
                        });
                        crate::info!("gpu[{}]: scout census: submission {} ({} ahead): {bad} of its {} walked copies differ from what a reader takes at its turn: {}", cp.name, job.index, job.ahead, out.seeds.len(), d.unwrap_or_default());
                    }
                }
            }
            _ => {
                seeds_ok = false;
                CHECKS[14].fetch_add(1, Ordering::Relaxed);
            }
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
    if start_ok && seeds_ok && c.conflicts == 0 && !c.images {
        CHECKS[5].fetch_add(1, Ordering::Relaxed);
        if whole {
            CHECKS[6].fetch_add(1, Ordering::Relaxed);
        }
        CHECKS[7].fetch_add(got, Ordering::Relaxed);
    }
    if c.conflicts > 0 && c.conflicts == c.textures {
        CHECKS[8].fetch_add(1, Ordering::Relaxed);
    }
    if start_ok && seeds_ok && c.conflicts == c.textures && !c.images {
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
    let hand = if v(&HAND[0]) == 0 {
        String::new()
    } else {
        let mut r = REFUSED.lock().clone();
        r.sort_by(|a, b| b.1.cmp(&a.1));
        let r: Vec<String> = r.iter().map(|(w, n)| format!("{w} {n}")).collect();
        format!(
            "    scouts' hand-overs (`tune scouts_commit`): {} asked: {} records taken over whole, {} as a prefix, {} refused ({}); {} K of the asked submissions' {} K dwords taken over ({:.0}%); the command processor waited {:.0} us for its scout on average ({} ms in all), a scout's commit {:.0} us; records that read published copies {} ({} of them checked at the turn, {} records refused for one); {} records not finished at their turn (not waited for: `tune scouts_wait`); {} records taken over up to a gate, {} segments behind gates taken over, {} refused there; the reads that refused records: {}\n",
            v(&HAND[0]),
            v(&HAND[1]),
            v(&HAND[2]),
            v(&HAND[3]),
            r.join(", "),
            v(&HAND[4]) / 1000,
            v(&HAND[5]) / 1000,
            100.0 * v(&HAND[4]) as f64 / v(&HAND[5]).max(1) as f64,
            v(&HAND[6]) as f64 / 1e3 / v(&HAND[0]).max(1) as f64,
            v(&HAND[6]) / 1_000_000,
            v(&HAND[7]) as f64 / 1e3 / v(&HAND[0]).max(1) as f64,
            v(&super::forward::FROZEN_STATS[0]),
            v(&super::forward::FROZEN_STATS[1]),
            v(&super::forward::FROZEN_STATS[2]),
            v(&LATE),
            v(&GATED[0]),
            v(&GATED[1]),
            v(&GATED[2]),
            {
                let mut g = super::cp::SCOUT_REFUSED_READS.lock().clone();
                g.sort_by(|a, b| b.1.cmp(&a.1));
                g.iter().take(8).map(|(k, n)| format!("{k} {n}")).collect::<Vec<_>>().join(", ")
            }
        )
    };
    format!(
        "    scouts (census, `tune scouts` {}): {} submissions claimed ({:.1} ahead of the command processor on average; {} reached by it before their scout began), {} recorded in {} ms ({:.2} ms each): {} whole, {:.0}% of their dwords ({} K of {} K); records ended by: {}; {} done before their turn ({:.2} ms early on average), the others {:.2} ms late; a record: {:.1} commands of its own, {:.2} held runs, {:.1} batches, {:.1} chunk batches, {:.0} CPU reads; waits passed on memory {}, a promise {}, its own held write {}, an earlier queued submission's write {}, gates {} (by the label's last writer: {})\n    scouts' checks at the turn: {} records, start state differed {}, CPU reads written meanwhile in {} ({} reads; first by kind and writer: {}), images with clears or reloads pending {}; textures bound {} ({} read from guest memory); held CP writes forwarded {}, copies walked ahead {} (records with walked copies {}, differing at the turn {}, turn before the walk {}), copy kernel dispatches {} (computed {}; not: distrusted {}, shapes {}, control block unreadable {}, counts {}, control block still written {}, source still written {}); every check passed {} ({} whole submissions, {} K dwords: {:.0}% of the claimed submissions' dwords); with textures' reads left out ({} records had only those) {} ({} whole, {} K dwords: {:.0}%); the walk stopped at {} submissions ({})\n",
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
        v(&super::forward::CP_WRITE_ENTRIES),
        v(&super::forward::WALKED_ENTRIES),
        v(&CHECKS[12]),
        v(&CHECKS[13]),
        v(&CHECKS[14]),
        v(&super::forward::VIEW_REFUSALS[0]),
        v(&super::forward::VIEW_REFUSALS[7]),
        v(&super::forward::VIEW_REFUSALS[1]),
        v(&super::forward::VIEW_REFUSALS[2]),
        v(&super::forward::VIEW_REFUSALS[3]),
        v(&super::forward::VIEW_REFUSALS[4]),
        v(&super::forward::VIEW_REFUSALS[5]),
        v(&super::forward::VIEW_REFUSALS[6]),
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
    ) + &hand
}
