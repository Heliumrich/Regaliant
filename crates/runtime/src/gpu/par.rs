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
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Chunk workers wanted (`tune par`; 0: runs are recorded by the command
/// processor alone).
pub fn workers() -> usize {
    (super::tune::par() as usize).min(8)
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
            let w = Arc::new(Worker { cp: Mutex::new(cp), job: Mutex::new(None), cv: Condvar::new(), done: Mutex::new(None), done_cv: Condvar::new() });
            let w2 = w.clone();
            let tname = wname.clone();
            let r = crate::thread::spawn_host(
                &wname,
                8 << 20,
                Box::new(move || {
                    crate::log::set_thread_name(&tname);
                    crate::exception::arm_watch_current_thread(false);
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
        w.done_cv.notify_all();
    }
}

impl Worker {
    /// Starts the chunk [from, to) from `state`.
    pub fn start(&self, state: Box<ChunkState>, from: u64, to: u64) {
        *self.done.lock() = None;
        *self.job.lock() = Some(Job { state, from, to });
        self.cv.notify_all();
    }

    /// Waits for the chunk: where it stopped (None: it ran to its end).
    pub fn wait(&self) -> Option<u64> {
        let t = std::time::Instant::now();
        let mut g = self.done.lock();
        // A chunk usually ends within microseconds of the main command
        // processor's own; spin a little before blocking.
        if g.is_none() {
            drop(g);
            let mut spins = 0;
            while spins < 4096 && self.done.lock().is_none() {
                std::hint::spin_loop();
                spins += 1;
            }
            g = self.done.lock();
        }
        while g.is_none() {
            self.done_cv.wait(&mut g);
        }
        WAIT_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        g.take().unwrap()
    }
}

/// One status line ("" before any parallel run).
pub fn status() -> String {
    let runs = RUNS.load(Ordering::Relaxed);
    if runs == 0 {
        return String::new();
    }
    format!(
        "    parallel runs: {runs} runs, {} chunks to workers ({} commands), {} stopped early, {} cancelled; main waited {} ms for workers\n",
        CHUNKS.load(Ordering::Relaxed),
        COMMANDS.load(Ordering::Relaxed),
        STOPS.load(Ordering::Relaxed),
        CANCELLED.load(Ordering::Relaxed),
        WAIT_NS.load(Ordering::Relaxed) / 1_000_000
    )
}
