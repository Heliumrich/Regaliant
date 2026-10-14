//! Parallel recording of synchronization-free runs (STATUS: Next steps 1,
//! parallel recording).
//!
//! A long run of commands without synchronization packets between them is
//! split at command boundaries into chunks. The main command processor
//! executes the first chunk itself; the others form a queue (`Run`) that the
//! workers take in program order: a worker is a command processor of its own
//! (`Cp`, started from a copy of the register state at the chunk's start)
//! whose executor is deferred (`Exec::deferred`): it records the chunk's
//! draws and dispatches with the same code into batches that it closes, then
//! waits for the chunk's turn, submits them, posts the state at the chunk's
//! end and takes the next chunk not taken. The main command processor gives
//! a chunk its turn once its own work before the chunk went to the GPU
//! (`Cp::par_collect`) and continues with the state at the chunk's end; a
//! chunk that ran to its end passes the turn to the next itself, so the GPU
//! starts on a run while its later chunks are still being recorded (`tune
//! par_split`: chunks per recorder). A command the worker cannot record
//! without flushing or waiting for its own unsubmitted work stops the chunk
//! there; the main command processor executes the rest of that chunk itself
//! and then gives the next chunk its turn.

use super::cp::{ChunkState, Cp};
use parking_lot::{Condvar, Mutex};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

/// Chunk workers wanted (`tune par`; 0: runs are recorded by the command
/// processor alone). By default a third of the logical CPUs, 2 to 8: on 24
/// logical CPUs 6, 7 and 8 workers recorded +2.0%, +2.2% and +2.4% against 5
/// (FINDINGS.md 2026-10-14); fewer CPUs keep fewer threads busy beside the
/// title's 13 job workers.
pub fn workers() -> usize {
    match super::tune::par() {
        1000 => {
            static AUTO: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
            *AUTO.get_or_init(|| std::thread::available_parallelism().map_or(5, |n| (n.get() / 3).clamp(2, 8)))
        }
        v => (v as usize).min(8),
    }
}

/// Chunks per recorder (the main command processor and each worker) a run
/// is split into (`tune par_split`; 1: one each, all submitted when the
/// main command processor's own chunk is done).
pub fn split() -> usize {
    (super::tune::par_split() as usize).clamp(1, 16)
}

/// The main command processor's chunk relative to a worker's, in 1/1000
/// (`tune par_balance`): raised after a split run in which it waited for
/// the workers when collecting their batches, lowered after one in which it
/// did not (the workers start after the planning and with cold per-thread
/// caches; it waited ~6.6% of its time with equal chunks, profile run
/// 20261012-001629).
static MAIN_WEIGHT: AtomicU64 = AtomicU64::new(1000);

/// The main command processor's chunk relative to a worker's.
pub fn main_weight() -> f64 {
    let fixed = super::tune::par_weight();
    if fixed > 0 {
        fixed.min(4000) as f64 / 1000.0
    } else if super::tune::par_balance() {
        MAIN_WEIGHT.load(Ordering::Relaxed) as f64 / 1000.0
    } else {
        1.0
    }
}

/// A split run was collected; the main command processor waited `ns` for
/// its workers in all.
pub fn note_main_wait(ns: u64) {
    if !super::tune::par_balance() {
        return;
    }
    let w = MAIN_WEIGHT.load(Ordering::Relaxed);
    let next = if ns > 20_000 { (w + 25).min(4000) } else { w.saturating_sub(25).max(500) };
    MAIN_WEIGHT.store(next, Ordering::Relaxed);
}

/// A chunk as its worker left it.
pub struct Done {
    /// Where it stopped (None: it ran to its end).
    pub stopped: Option<u64>,
    /// The command-processor state where it stopped or ended.
    pub state: ChunkState,
    /// The sequence of its last batch (None: it submitted none).
    pub last: Option<u64>,
    /// Why it stopped ("" when it ran to its end).
    pub why: &'static str,
}

/// The chunks of a parallel run that the workers record (all but the main
/// command processor's), in program order.
pub struct Run {
    /// Chunk i: its start state (taken by the worker that records it), its
    /// first packet and the packet after it.
    chunks: Vec<(Mutex<Option<Box<ChunkState>>>, u64, u64)>,
    /// The next chunk not taken.
    next: AtomicUsize,
    /// Chunks whose turn came: chunk i submits once this exceeds i.
    turn: AtomicUsize,
    /// Chunks posted (in order: chunk i's once this exceeds i).
    posted: AtomicUsize,
    results: Mutex<Vec<Option<Done>>>,
    cancelled: AtomicBool,
    /// Workers started on the run that have not left it.
    active: AtomicUsize,
    lock: Mutex<()>,
    cv: Condvar,
    /// When the run was planned (the workers' start latency).
    born: std::time::Instant,
}

impl Run {
    /// The run of `chunks` ([start state, first packet, end]).
    pub fn new(chunks: Vec<(Box<ChunkState>, u64, u64)>) -> Run {
        let n = chunks.len();
        Run {
            chunks: chunks.into_iter().map(|(s, a, b)| (Mutex::new(Some(s)), a, b)).collect(),
            next: AtomicUsize::new(0),
            turn: AtomicUsize::new(0),
            posted: AtomicUsize::new(0),
            results: Mutex::new((0..n).map(|_| None).collect()),
            cancelled: AtomicBool::new(false),
            active: AtomicUsize::new(0),
            lock: Mutex::new(()),
            cv: Condvar::new(),
            born: std::time::Instant::now(),
        }
    }

    fn notify(&self) {
        let _g = self.lock.lock();
        self.cv.notify_all();
    }

    /// Waits until `ready` holds (spinning briefly first: the chunks of a run
    /// are handed over within microseconds of each other); how long (ns).
    fn wait_until(&self, ready: impl Fn() -> bool) -> u64 {
        if ready() {
            return 0;
        }
        let t = std::time::Instant::now();
        let mut spins = 0;
        while spins < 4096 && !ready() {
            std::hint::spin_loop();
            spins += 1;
        }
        if !ready() {
            let mut g = self.lock.lock();
            while !ready() {
                self.cv.wait(&mut g);
            }
        }
        t.elapsed().as_nanos() as u64
    }

    /// Chunk `i` may submit (the main command processor's work before it went
    /// to the GPU).
    pub fn grant(&self, i: usize) {
        if self.turn.fetch_max(i + 1, Ordering::AcqRel) <= i {
            self.notify();
        }
    }

    /// Waits for chunk `i` to be posted: it and how long the wait took (ns).
    pub fn collect(&self, i: usize) -> (Done, u64) {
        let ns = self.wait_until(|| self.posted.load(Ordering::Acquire) > i);
        WAIT_NS.fetch_add(ns, Ordering::Relaxed);
        let d = self.results.lock()[i].take().expect("parallel chunk posted twice");
        (d, ns)
    }

    /// Abandons the run: chunks not yet submitted are dropped by their
    /// workers; returns once every worker left it.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
        self.notify();
        self.wait_until(|| self.active.load(Ordering::Acquire) == 0);
    }

    fn leave(&self) {
        if self.active.fetch_sub(1, Ordering::AcqRel) == 1 {
            self.notify();
        }
    }
}

pub struct Worker {
    /// The worker's command processor (locked by the worker while it runs a
    /// chunk).
    cp: Mutex<Cp>,
    /// The run to take chunks of.
    job: Mutex<Option<Arc<Run>>>,
    cv: Condvar,
}

/// Statistics: runs split, chunks handed to workers, chunks that stopped,
/// commands per chunk..., main command processor's waits for workers (ns),
/// workers' waits for their chunks' turns (ns).
pub static RUNS: AtomicU64 = AtomicU64::new(0);
pub static CHUNKS: AtomicU64 = AtomicU64::new(0);
pub static STOPS: AtomicU64 = AtomicU64::new(0);
pub static COMMANDS: AtomicU64 = AtomicU64::new(0);
pub static WAIT_NS: AtomicU64 = AtomicU64::new(0);
pub static TURN_NS: AtomicU64 = AtomicU64::new(0);
/// Workers' time per phase (ns): from the run's planning to their first
/// chunk of it (per worker and run), setting a chunk's state up, recording
/// it, closing its batch, submitting and posting it; and the count of first
/// chunks.
static PHASE_NS: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];
static STARTS: AtomicU64 = AtomicU64::new(0);
pub static CANCELLED: AtomicU64 = AtomicU64::new(0);
static STOP_WHY: Mutex<Vec<(&'static str, u64)>> = Mutex::new(Vec::new());

/// Runs the planner saw, by length (commands): [<32, <64, <192, <512,
/// <2048, more] x [runs, commands]; and by what ended them (PM4 opcode,
/// 0x100 + private operation, 0x130 buffer end, 0x131 limit, 0x133
/// predicated) x [runs, commands].
static LENGTHS: Mutex<[[u64; 2]; 6]> = Mutex::new([[0; 2]; 6]);
static ENDS: Mutex<Vec<(usize, u64, u64)>> = Mutex::new(Vec::new());

/// A run of `n` commands ended by `why` (planner statistics).
pub fn note_run(n: usize, why: usize) {
    let b = match n {
        0..32 => 0,
        32..64 => 1,
        64..192 => 2,
        192..512 => 3,
        512..2048 => 4,
        _ => 5,
    };
    let mut g = LENGTHS.lock();
    g[b][0] += 1;
    g[b][1] += n as u64;
    let mut e = ENDS.lock();
    match e.iter_mut().find(|x| x.0 == why) {
        Some(x) => {
            x.1 += 1;
            x.2 += n as u64;
        }
        None => e.push((why, 1, n as u64)),
    }
}

/// A chunk stopped early for `why`.
pub fn note_stop(why: &'static str) {
    let mut g = STOP_WHY.lock();
    match g.iter_mut().find(|e| e.0 == why) {
        Some(e) => e.1 += 1,
        None => g.push((why, 1)),
    }
}

/// The workers of a queue's command processor (created on first use).
pub struct Pool {
    workers: Vec<Arc<Worker>>,
}

impl Pool {
    pub fn new() -> Pool {
        Pool { workers: Vec::new() }
    }

    /// Worker `i` of the command processor `name` (started on first use);
    /// None if it cannot be made.
    pub fn worker(&mut self, i: usize, name: &str, compute: bool) -> Option<Arc<Worker>> {
        while self.workers.len() <= i {
            let n = self.workers.len();
            let wname = format!("{name}-par{n}");
            let mut cp = Cp::new(&wname, compute);
            if let Err(e) = cp.make_worker_exec() {
                crate::warn!("gpu[{wname}]: no executor for a parallel worker ({e})");
                return None;
            }
            let w = Arc::new(Worker { cp: Mutex::new(cp), job: Mutex::new(None), cv: Condvar::new() });
            let w2 = w.clone();
            let tname = wname.clone();
            let r = crate::thread::spawn_host(
                &wname,
                8 << 20,
                Box::new(move || {
                    crate::log::set_thread_name(&tname);
                    crate::exception::arm_watch_current_thread(false);
                    super::profile_this_thread("par");
                    run_worker(&w2, &tname);
                }),
            );
            if let Err(e) = r {
                crate::warn!("gpu[{wname}]: no thread for a parallel worker ({e})");
                return None;
            }
            self.workers.push(w);
        }
        Some(self.workers[i].clone())
    }
}

impl Default for Pool {
    fn default() -> Pool {
        Pool::new()
    }
}

fn run_worker(w: &Worker, name: &str) {
    loop {
        super::prio::apply();
        let run = {
            let mut g = w.job.lock();
            loop {
                if let Some(r) = g.take() {
                    break r;
                }
                w.cv.wait(&mut g);
            }
        };
        take_chunks(w, &run, name);
        run.leave();
    }
}

/// Records, submits and posts chunks of `run` until none is left or the run
/// is cancelled.
fn take_chunks(w: &Worker, run: &Run, name: &str) {
    let mut first = true;
    let lap = |k: usize, t: &mut std::time::Instant| {
        let now = std::time::Instant::now();
        PHASE_NS[k].fetch_add((now - *t).as_nanos() as u64, Ordering::Relaxed);
        *t = now;
    };
    loop {
        if run.cancelled.load(Ordering::Acquire) {
            return;
        }
        let i = run.next.fetch_add(1, Ordering::AcqRel);
        let Some((state, from, to)) = run.chunks.get(i) else { return };
        let Some(state) = state.lock().take() else { return };
        let (from, to) = (*from, *to);
        let mut t = std::time::Instant::now();
        if std::mem::take(&mut first) {
            PHASE_NS[0].fetch_add((t - run.born).as_nanos() as u64, Ordering::Relaxed);
            STARTS.fetch_add(1, Ordering::Relaxed);
        }
        let mut cp = w.cp.lock();
        cp.set_chunk_state(&state);
        cp.chunk_start();
        lap(1, &mut t);
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| cp.run_chunk(from, to)));
        let failed = r.is_err();
        let stopped = r.unwrap_or(Some(from));
        lap(2, &mut t);
        cp.chunk_finish();
        lap(3, &mut t);
        let ready = || run.turn.load(Ordering::Acquire) > i || run.cancelled.load(Ordering::Acquire);
        if !ready() {
            let tl = super::timeline::mark();
            TURN_NS.fetch_add(run.wait_until(ready), Ordering::Relaxed);
            super::timeline::end(name, tl, || "wait: chunk turn".into());
        }
        let mut t = std::time::Instant::now();
        let granted = run.turn.load(Ordering::Acquire) > i;
        if !granted || failed {
            // Cancelled before its turn, or failed (the main command
            // processor executes it from its start): never submitted.
            cp.worker_discard();
        }
        if !granted {
            return;
        }
        let done = if failed {
            Done { stopped, state: *state, last: None, why: "failed" }
        } else {
            let last = cp.worker_submit();
            let why = if stopped.is_some() { cp.worker_stop_why() } else { "" };
            Done { stopped, state: cp.chunk_state(), last, why }
        };
        drop(cp);
        run.results.lock()[i] = Some(done);
        run.posted.store(i + 1, Ordering::Release);
        if stopped.is_none() {
            // The next chunk follows this one's end directly.
            run.turn.fetch_max(i + 2, Ordering::AcqRel);
        }
        run.notify();
        lap(4, &mut t);
    }
}

impl Worker {
    /// Takes chunks of `run` (`Run::active` counts it).
    pub fn start(&self, run: Arc<Run>) {
        run.active.fetch_add(1, Ordering::AcqRel);
        *self.job.lock() = Some(run);
        self.cv.notify_all();
    }
}

/// Clears the statistics (`profile reset`: the status line and the profile
/// report then cover the window).
pub fn reset_stats() {
    for c in [&RUNS, &CHUNKS, &STOPS, &COMMANDS, &WAIT_NS, &TURN_NS, &CANCELLED, &STARTS] {
        c.store(0, Ordering::Relaxed);
    }
    for c in &PHASE_NS {
        c.store(0, Ordering::Relaxed);
    }
    STOP_WHY.lock().clear();
    *LENGTHS.lock() = [[0; 2]; 6];
    ENDS.lock().clear();
}

/// One status line ("" before any parallel run).
pub fn status() -> String {
    let runs = RUNS.load(Ordering::Relaxed);
    if runs == 0 && ENDS.lock().is_empty() {
        return String::new();
    }
    let lengths = *LENGTHS.lock();
    let names = ["<32", "<64", "<192", "<512", "<2048", ">=2048"];
    let lens: Vec<String> = names.iter().zip(lengths.iter()).map(|(n, x)| format!("{n} {}/{}", x[0], x[1])).collect();
    let mut ends = ENDS.lock().clone();
    ends.sort_by(|a, b| b.2.cmp(&a.2));
    let end_name = |w: usize| match w {
        0x130 => "buffer end".to_string(),
        0x131 => "limit".to_string(),
        0x133 => "predicated".to_string(),
        w if w >= 0x200 => format!("event {:#x}", w - 0x200),
        w if w >= 0x100 => format!("private {:#x}", w - 0x100),
        w => format!("{w:#x}"),
    };
    let ends: Vec<String> = ends.iter().take(10).map(|(w, r, c)| format!("{} {r}/{c}", end_name(*w))).collect();
    let mut why = STOP_WHY.lock().clone();
    why.sort_by(|a, b| b.1.cmp(&a.1));
    let why: Vec<String> = why.iter().map(|(w, n)| format!("{w} {n}")).collect();
    format!(
        "    parallel runs: {runs} runs, {} chunks to workers ({} commands; {} per recorder), {} stopped early ({}), {} cancelled; main waited {} ms for workers (its chunk {:.2}x a worker's), workers {} ms for their turns ({} batch slots added), per chunk: start latency {:.0} us (first chunks), setup {:.1} us, recording {:.0} us ({:.0} commands), closing {:.0} us, submitting and posting {:.1} us; runs seen by length (runs/commands) {}; by end {}\n",
        CHUNKS.load(Ordering::Relaxed),
        COMMANDS.load(Ordering::Relaxed),
        split(),
        STOPS.load(Ordering::Relaxed),
        why.join(", "),
        CANCELLED.load(Ordering::Relaxed),
        WAIT_NS.load(Ordering::Relaxed) / 1_000_000,
        main_weight(),
        TURN_NS.load(Ordering::Relaxed) / 1_000_000,
        super::exec::WORKER_SLOTS.load(Ordering::Relaxed),
        PHASE_NS[0].load(Ordering::Relaxed) as f64 / STARTS.load(Ordering::Relaxed).max(1) as f64 / 1e3,
        PHASE_NS[1].load(Ordering::Relaxed) as f64 / CHUNKS.load(Ordering::Relaxed).max(1) as f64 / 1e3,
        PHASE_NS[2].load(Ordering::Relaxed) as f64 / CHUNKS.load(Ordering::Relaxed).max(1) as f64 / 1e3,
        COMMANDS.load(Ordering::Relaxed) as f64 / CHUNKS.load(Ordering::Relaxed).max(1) as f64,
        PHASE_NS[3].load(Ordering::Relaxed) as f64 / CHUNKS.load(Ordering::Relaxed).max(1) as f64 / 1e3,
        PHASE_NS[4].load(Ordering::Relaxed) as f64 / CHUNKS.load(Ordering::Relaxed).max(1) as f64 / 1e3,
        lens.join(", "),
        ends.join(", ")
    )
}
