//! Pipelines compiled ahead of their first use (`tune compile_ahead`).
//!
//! The first use of a pipeline stalls its command processor while the
//! driver compiles it: 0.2-2.9 s for the title's large compute programs
//! (40,000-450,000 SPIR-V words; the user's play test of 2026-10-16 stalled
//! 27 s in four minutes), and such uses come in bursts: the programs of an
//! effect or an area appear together in one command stream, each compiled
//! only when execution reaches it. When a dispatch's pipeline is missing,
//! its command processor scans the commands after it
//! (`Cp::prefetch_compiles`) and hands their compute stages to these
//! workers, which prepare each with the registers the scan saw (as
//! `gpu::ahead` does) and create its pipeline if it is missing
//! (`exec::precompile`). Meanwhile the command processor creates its own,
//! and later waits for the ones in flight instead of compiling them again
//! (`exec::compute_pipe`). Compute stages that a command processor prepares
//! while it waits on a label (`tune wait_ahead`) and whose pipeline is
//! missing are handed over too, already prepared (`submit_prepared`). A
//! preparation made from inputs that differ at execution only costs a
//! compile: execution prepares its stages itself.

use super::ahead::Slot;
use super::shaders::{self, Prepared, RegRead, StageKind};
use parking_lot::{Condvar, Mutex};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

/// A stage to prepare from scanned registers, or one prepared already.
enum Job {
    Regs(Arc<Slot>),
    Prepared(Box<Prepared>),
}

struct Pool {
    jobs: Mutex<VecDeque<Job>>,
    cv: Condvar,
}

static POOL: Pool = Pool { jobs: Mutex::new(VecDeque::new()), cv: Condvar::new() };
/// Translations of the prepared stages queued (`submit_prepared`).
static QUEUED: Mutex<Vec<u64>> = Mutex::new(Vec::new());
static WORKERS: AtomicUsize = AtomicUsize::new(0);

/// Statistics: indices into STATS.
const SCANS: usize = 0;
const SCAN_NS: usize = 1;
const JOBS: usize = 2;
const PREPARED: usize = 3;
const FAILED: usize = 4;
const COMPILED: usize = 5;
const COMPILE_NS: usize = 6;
const PRESENT: usize = 7;
const DROPPED: usize = 8;
const AT_WAITS: usize = 9;
static STATS: [AtomicU64; 10] = [const { AtomicU64::new(0) }; 10];

/// Most compute programs one scan hands over (16 lost the gain: a level's
/// first frames meet hundreds of programs; of those compiled ahead with 64,
/// 93% were used, run 20261016-214851).
const MAX_JOBS: usize = 64;

/// Whether missing pipelines are compiled ahead (`tune compile_ahead`:
/// worker threads; 0 off).
pub fn enabled() -> bool {
    super::tune::compile_ahead() > 0
}

/// `DS_COMPILE_AHEAD_MODE` (diagnosis): bit 0 scans at first uses, bit 1
/// stages prepared at label waits (default 3: both); bit 2 drops what a
/// scan found, bit 3 prepares it without creating pipelines.
fn mode() -> u32 {
    static M: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *M.get_or_init(|| std::env::var("DS_COMPILE_AHEAD_MODE").ok().and_then(|v| v.parse().ok()).unwrap_or(3))
}

/// Scans at first uses wanted.
pub fn at_first_uses() -> bool {
    enabled() && mode() & 1 != 0
}

/// Stages prepared at label waits handed over.
pub fn at_waits() -> bool {
    enabled() && mode() & 2 != 0
}

fn ensure_workers() {
    let want = super::tune::compile_ahead().min(16) as usize;
    while WORKERS.load(Ordering::Acquire) < want {
        let n = WORKERS.fetch_add(1, Ordering::AcqRel);
        if n >= want {
            WORKERS.fetch_sub(1, Ordering::AcqRel);
            break;
        }
        let name = format!("gpu-compile{n}");
        let r = crate::thread::spawn_host(
            &name.clone(),
            8 << 20,
            Box::new(move || {
                crate::log::set_thread_name(&name);
                crate::exception::arm_watch_current_thread(false);
                super::profile_this_thread("compile");
                worker();
            }),
        );
        if let Err(e) = r {
            crate::warn!("gpu: no compile worker ({e})");
            WORKERS.fetch_sub(1, Ordering::AcqRel);
            break;
        }
    }
}

fn worker() {
    loop {
        let job = {
            let mut q = POOL.jobs.lock();
            loop {
                if let Some(s) = q.pop_front() {
                    break s;
                }
                POOL.cv.wait(&mut q);
            }
        };
        // A panic must not end the worker (its jobs would wait forever).
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match job {
            Job::Regs(slot) => run(&slot),
            Job::Prepared(p) => {
                compile(&p);
                QUEUED.lock().retain(|&k| k != p.tr.key);
            }
        }));
    }
}

fn run(slot: &Slot) {
    let regs = slot.regs();
    let p = match shaders::prepare_compute(regs, slot.wave64()) {
        Ok(p) if !regs.missed.load(Ordering::Relaxed) => p,
        _ => {
            STATS[FAILED].fetch_add(1, Ordering::Relaxed);
            return;
        }
    };
    STATS[PREPARED].fetch_add(1, Ordering::Relaxed);
    if mode() & 8 != 0 {
        return;
    }
    compile(&p);
}

fn compile(p: &Prepared) {
    let t = std::time::Instant::now();
    match super::exec::precompile(p) {
        Some(true) => {
            STATS[COMPILED].fetch_add(1, Ordering::Relaxed);
            STATS[COMPILE_NS].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
        Some(false) => {
            STATS[PRESENT].fetch_add(1, Ordering::Relaxed);
        }
        None => {
            STATS[FAILED].fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// Hands the compute stages of a scan (`Cp::prefetch_compiles`, after
/// `ns` of scanning) to the workers, the first of each program only.
pub fn submit(slots: Vec<Arc<Slot>>, ns: u64) {
    STATS[SCANS].fetch_add(1, Ordering::Relaxed);
    STATS[SCAN_NS].fetch_add(ns, Ordering::Relaxed);
    let mut seen: Vec<(u64, bool)> = Vec::new();
    let mut jobs = Vec::new();
    for s in slots {
        if s.regs().kind != StageKind::Compute {
            continue;
        }
        let r = s.regs();
        let addr = ((r.sh(0x20C) as u64) | ((r.sh(0x20D) as u64 & 0xFF) << 32)) << 8;
        let k = (addr, s.wave64());
        if addr == 0 || seen.contains(&k) {
            continue;
        }
        seen.push(k);
        if jobs.len() >= MAX_JOBS {
            STATS[DROPPED].fetch_add(1, Ordering::Relaxed);
            continue;
        }
        jobs.push(s);
    }
    if jobs.is_empty() || mode() & 4 != 0 {
        return;
    }
    ensure_workers();
    STATS[JOBS].fetch_add(jobs.len() as u64, Ordering::Relaxed);
    POOL.jobs.lock().extend(jobs.into_iter().map(Job::Regs));
    POOL.cv.notify_all();
}

/// Hands a compute stage prepared while its command processor waited on a
/// label, whose pipeline is missing, to the workers (ahead of the queued
/// scans: it is the nearest to execution).
pub fn submit_prepared(p: Prepared) {
    // Once while queued (each label wait prepares the stages after it).
    {
        let mut q = QUEUED.lock();
        if q.contains(&p.tr.key) {
            return;
        }
        q.push(p.tr.key);
    }
    ensure_workers();
    STATS[AT_WAITS].fetch_add(1, Ordering::Relaxed);
    POOL.jobs.lock().push_front(Job::Prepared(Box::new(p)));
    POOL.cv.notify_one();
}

/// One status line ("" before any scan).
pub fn status() -> String {
    let s: Vec<u64> = STATS.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    if s[SCANS] == 0 && s[AT_WAITS] == 0 {
        return String::new();
    }
    let (waits, wait_ns, dups) = super::exec::inflight_stats();
    format!(
        "    pipelines compiled ahead: {} scans at first uses ({} ms), {} programs handed to {} workers ({} over the per-scan limit) and {} stages prepared at label waits, {} prepared, {} pipelines created ({} ms of compiles), {} present already, {} failed or not prepared; command processors waited {} times for a pipeline in flight ({} ms), {} compiles of one pipeline avoided\n",
        s[SCANS],
        s[SCAN_NS] / 1_000_000,
        s[JOBS],
        WORKERS.load(Ordering::Relaxed),
        s[DROPPED],
        s[AT_WAITS],
        s[PREPARED],
        s[COMPILED],
        s[COMPILE_NS] / 1_000_000,
        s[PRESENT],
        s[FAILED],
        waits,
        wait_ns / 1_000_000,
        dups
    )
}
