//! Parallel recording of synchronization-free runs (STATUS: Next steps 1,
//! parallel recording).
//!
//! A long run of commands without synchronization packets between them is
//! split at command boundaries into chunks. The main command processor
//! executes the first chunk itself; each later chunk goes to a worker: a
//! command processor of its own (`Cp`, started from a copy of the register
//! state at the chunk's start) whose executor is deferred (`Exec::deferred`):
//! it records the chunk's draws and dispatches with the same code into
//! batches that it closes but does not submit. The main command processor
//! then submits each chunk's batches in program order after its own work
//! (`Cp::par_collect`) and continues with the state at the chunk's end. A
//! command the worker cannot record without flushing or waiting for its own
//! unsubmitted work stops the chunk there; the main command processor
//! executes the rest of that chunk itself.

use super::cp::{ChunkState, Cp};
use parking_lot::{Condvar, Mutex};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

/// Chunk workers wanted (`tune par`; 0: runs are recorded by the command
/// processor alone).
pub fn workers() -> usize {
    (super::tune::par() as usize).min(8)
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
    if super::tune::par_balance() {
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

struct Job {
    state: Box<ChunkState>,
    from: u64,
    to: u64,
}

pub struct Worker {
    /// The worker's command processor (locked by the worker while it runs a
    /// chunk, by the main command processor while it takes the results).
    pub cp: Mutex<Cp>,
    job: Mutex<Option<Job>>,
    cv: Condvar,
    /// The finished chunk: where it stopped (None: it ran to its end).
    done: Mutex<Option<Option<u64>>>,
    /// Set once `done` holds the finished chunk (spun on without the lock).
    finished: AtomicBool,
    done_cv: Condvar,
}

/// Statistics: runs split, chunks handed to workers, chunks that stopped,
/// commands per chunk..., main command processor's waits for workers (ns).
pub static RUNS: AtomicU64 = AtomicU64::new(0);
pub static CHUNKS: AtomicU64 = AtomicU64::new(0);
pub static STOPS: AtomicU64 = AtomicU64::new(0);
pub static COMMANDS: AtomicU64 = AtomicU64::new(0);
pub static WAIT_NS: AtomicU64 = AtomicU64::new(0);
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
            let w = Arc::new(Worker { cp: Mutex::new(cp), job: Mutex::new(None), cv: Condvar::new(), done: Mutex::new(None), finished: AtomicBool::new(false), done_cv: Condvar::new() });
            let w2 = w.clone();
            let tname = wname.clone();
            let r = crate::thread::spawn_host(
                &wname,
                8 << 20,
                Box::new(move || {
                    crate::log::set_thread_name(&tname);
                    crate::exception::arm_watch_current_thread(false);
                    super::profile_this_thread("par");
                    run_worker(&w2);
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

fn run_worker(w: &Worker) {
    loop {
        super::prio::apply();
        let job = {
            let mut g = w.job.lock();
            loop {
                if let Some(j) = g.take() {
                    break j;
                }
                w.cv.wait(&mut g);
            }
        };
        let stopped = {
            let mut cp = w.cp.lock();
            cp.set_chunk_state(&job.state);
            cp.chunk_start();
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| cp.run_chunk(job.from, job.to))).unwrap_or(Some(job.from));
            cp.chunk_finish();
            r
        };
        *w.done.lock() = Some(stopped);
        w.finished.store(true, Ordering::Release);
        w.done_cv.notify_all();
    }
}

impl Worker {
    /// Starts the chunk [from, to) from `state`.
    pub fn start(&self, state: Box<ChunkState>, from: u64, to: u64) {
        *self.done.lock() = None;
        self.finished.store(false, Ordering::Release);
        *self.job.lock() = Some(Job { state, from, to });
        self.cv.notify_all();
    }

    /// Waits for the chunk: where it stopped (None: it ran to its end), and
    /// how long it waited (ns).
    pub fn wait(&self) -> (Option<u64>, u64) {
        let t = std::time::Instant::now();
        let mut g = self.done.lock();
        // A chunk usually ends within microseconds of the main command
        // processor's own; spin a little before blocking.
        if g.is_none() {
            drop(g);
            let mut spins = 0;
            if super::tune::lean_cmd() {
                // The flag, not the lock the worker takes to post its result
                // (2.6% of the main command processor's busy samples went to
                // this loop's locking, run 20261012-213847).
                while spins < 4096 && !self.finished.load(Ordering::Acquire) {
                    std::hint::spin_loop();
                    spins += 1;
                }
            } else {
                while spins < 4096 && self.done.lock().is_none() {
                    std::hint::spin_loop();
                    spins += 1;
                }
            }
            g = self.done.lock();
        }
        while g.is_none() {
            self.done_cv.wait(&mut g);
        }
        let ns = t.elapsed().as_nanos() as u64;
        WAIT_NS.fetch_add(ns, Ordering::Relaxed);
        (g.take().unwrap(), ns)
    }
}

/// Clears the statistics (`profile reset`: the status line and the profile
/// report then cover the window).
pub fn reset_stats() {
    for c in [&RUNS, &CHUNKS, &STOPS, &COMMANDS, &WAIT_NS, &CANCELLED] {
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
        "    parallel runs: {runs} runs, {} chunks to workers ({} commands), {} stopped early ({}), {} cancelled; main waited {} ms for workers (its chunk {:.2}x a worker's); runs seen by length (runs/commands) {}; by end {}\n",
        CHUNKS.load(Ordering::Relaxed),
        COMMANDS.load(Ordering::Relaxed),
        STOPS.load(Ordering::Relaxed),
        why.join(", "),
        CANCELLED.load(Ordering::Relaxed),
        WAIT_NS.load(Ordering::Relaxed) / 1_000_000,
        main_weight(),
        lens.join(", "),
        ends.join(", ")
    )
}
