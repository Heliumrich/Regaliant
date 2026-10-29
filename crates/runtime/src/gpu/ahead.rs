//! Stage preparations made ahead of execution by worker threads.
//!
//! Within a run of commands without synchronization packets between them
//! (97% of the graphics queue's commands in the courtyard; STATUS: where the
//! graphics command processor's time goes) the title orders nothing, so the
//! shader inputs a command reads cannot depend on GPU writes of the same run.
//! After a run's first command the command processor scans the rest of the
//! run (`Cp::scan_ahead`: register writes and draw/dispatch packets only) and
//! hands each later command's stages to worker threads with a snapshot of
//! the registers they read (`StageRegs`). At execution a preparation is used
//! when the registers, the program, the image cache's lookup generation and
//! the guest words it read are unchanged (`take`); otherwise, or when no
//! worker has started it yet, the stage is prepared on the command processor
//! as before. The result is the same either way: a preparation is a function
//! of exactly those inputs.

use super::shaders::{self, Prepared, RegRead, StageKind, StageRegs};
use parking_lot::{Condvar, Mutex};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

/// Slot states.
const QUEUED: u32 = 0;
const RUNNING: u32 = 1;
const DONE: u32 = 2;
/// Not wanted any more (taken over by the command processor or dropped).
const CANCELLED: u32 = 3;

/// One stage preparation handed to the workers.
pub struct Slot {
    regs: StageRegs,
    wave64: bool,
    /// Its bindings' placement checks are made where it is prepared, after
    /// its result is posted (`tune fe2_precheck`).
    precheck: bool,
    state: AtomicU32,
    result: Mutex<Option<Result<Prepared, String>>>,
    cv: Condvar,
}

impl Slot {
    pub fn new(regs: StageRegs, wave64: bool) -> Arc<Slot> {
        Self::new_checked(regs, wave64, false)
    }

    /// `new`, its placement checks made where it is prepared when
    /// `precheck` (the graphics CP's stages under front end 2).
    pub fn new_checked(regs: StageRegs, wave64: bool, precheck: bool) -> Arc<Slot> {
        Arc::new(Slot { regs, wave64, precheck, state: AtomicU32::new(QUEUED), result: Mutex::new(None), cv: Condvar::new() })
    }

    /// The registers its preparation reads (`gpu::compile`).
    pub fn regs(&self) -> &StageRegs {
        &self.regs
    }

    pub fn wave64(&self) -> bool {
        self.wave64
    }

    /// Not needed any more (a worker skips it, or drops its result).
    pub fn cancel(&self) {
        if self.state.compare_exchange(QUEUED, CANCELLED, Ordering::AcqRel, Ordering::Acquire).is_ok() {
            STATS[DROPPED].fetch_add(1, Ordering::Relaxed);
        }
    }
}

struct Pool {
    jobs: Mutex<VecDeque<Arc<Slot>>>,
    cv: Condvar,
}

static POOL: Pool = Pool { jobs: Mutex::new(VecDeque::new()), cv: Condvar::new() };
static WORKERS: AtomicUsize = AtomicUsize::new(0);

/// Statistics: indices into STATS.
const SCANS: usize = 0;
const JOBS: usize = 1;
const USED: usize = 2;
const TAKEN_OVER: usize = 3;
const STALE_REGS: usize = 4;
const STALE_PROGRAM: usize = 5;
const STALE_IMAGES: usize = 6;
const STALE_WORDS: usize = 7;
const ERRORS: usize = 8;
const DROPPED: usize = 9;
const WAITED_NS: usize = 10;
const WAITS: usize = 11;
const SCAN_NS: usize = 12;
/// Stages a command processor prepared while it waited on a label
/// (`prepare_here`, `tune wait_ahead`) and the time that took (ns); scans
/// made at such waits and the commands they found.
const HERE: usize = 13;
const HERE_NS: usize = 14;
const WAIT_SCANS: usize = 15;
const WAIT_CMDS: usize = 16;
/// Stages whose placement checks were made where they were prepared, and
/// those checks' ns (`tune fe2_precheck`).
const PRECHECKS: usize = 17;
const PRECHECK_NS: usize = 18;
static STATS: [AtomicU64; 19] = [const { AtomicU64::new(0) }; 19];

/// Failed preparations by stage kind and reason (the error's words, numbers
/// and addresses left out).
static FAILED_WHY: Mutex<Option<std::collections::HashMap<String, u64>>> = Mutex::new(None);

fn note_failure(kind: StageKind, e: &str) {
    let words: Vec<&str> = e.split_whitespace().map(|w| if w.chars().any(|c| c.is_ascii_digit()) { "#" } else { w }).take(10).collect();
    let k = format!("{kind:?}: {}", words.join(" "));
    *FAILED_WHY.lock().get_or_insert_with(Default::default).entry(k).or_default() += 1;
}

/// Why scans ended: by PM4 opcode (0..0xFF), private operation (0x100 +
/// op), or END_* below.
static STOPS: [AtomicU64; 0x140] = [const { AtomicU64::new(0) }; 0x140];
pub const END_BUFFER: usize = 0x130;
pub const END_MAX: usize = 0x131;
pub const END_INVALID: usize = 0x132;
pub const END_PREDICATED: usize = 0x133;
pub const END_HEADER: usize = 0x134;
pub const END_ARGS: usize = 0x135;

/// A scan ended for `why` after `ns` on the command processor.
pub fn note_scan(why: usize, ns: u64) {
    STOPS[why.min(STOPS.len() - 1)].fetch_add(1, Ordering::Relaxed);
    STATS[SCAN_NS].fetch_add(ns, Ordering::Relaxed);
}

/// Worker threads wanted (`tune ahead`; 0: preparations are not made ahead).
pub fn enabled() -> bool {
    super::tune::ahead() > 0
}

/// Whether a command processor (a compute queue's: `compute`) hands its
/// stages to the workers (`tune ahead_queues`).
pub fn enabled_for(compute: bool) -> bool {
    enabled()
        && match super::tune::ahead_queues() {
            1 => compute,
            2 => !compute,
            _ => true,
        }
}

fn ensure_workers() {
    let want = super::tune::ahead().min(16) as usize;
    while WORKERS.load(Ordering::Acquire) < want {
        let n = WORKERS.fetch_add(1, Ordering::AcqRel);
        if n >= want {
            WORKERS.fetch_sub(1, Ordering::AcqRel);
            break;
        }
        let name = format!("gpu-prep{n}");
        let r = crate::thread::spawn_host(
            &name.clone(),
            8 << 20,
            Box::new(move || {
                crate::log::set_thread_name(&name);
                crate::exception::arm_watch_current_thread(false);
                super::profile_this_thread("prep");
                worker();
            }),
        );
        if let Err(e) = r {
            crate::warn!("gpu: no preparation worker ({e})");
            WORKERS.fetch_sub(1, Ordering::AcqRel);
            break;
        }
    }
}

fn worker() {
    loop {
        let slot = {
            let mut q = POOL.jobs.lock();
            loop {
                if let Some(s) = q.pop_front() {
                    break s;
                }
                POOL.cv.wait(&mut q);
            }
        };
        if slot.state.compare_exchange(QUEUED, RUNNING, Ordering::AcqRel, Ordering::Acquire).is_err() {
            continue;
        }
        run_slot(&slot);
    }
}

/// The compute stage a finished slot prepared, if it succeeded (a copy;
/// `gpu::compile` creates its pipeline ahead of use).
pub fn prepared_compute(slot: &Slot) -> Option<Prepared> {
    if slot.regs.kind != StageKind::Compute || slot.state.load(Ordering::Acquire) != DONE || slot.regs.missed.load(Ordering::Relaxed) {
        return None;
    }
    slot.result.lock().as_ref().and_then(|r| r.as_ref().ok().cloned())
}

/// Prepares `slot` on this thread if nothing took it yet (a command
/// processor while it waits on a label, `tune wait_ahead`); whether it did.
pub fn prepare_here(slot: &Slot) -> bool {
    if slot.state.compare_exchange(QUEUED, RUNNING, Ordering::AcqRel, Ordering::Acquire).is_err() {
        return false;
    }
    let t = std::time::Instant::now();
    run_slot(slot);
    STATS[HERE_NS].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    STATS[HERE].fetch_add(1, Ordering::Relaxed);
    true
}

/// A scan made at a label wait (`Cp::scan_through`) found `commands` and
/// ended for `why` (as `note_scan`) after `ns`.
pub fn note_wait_scan(commands: usize, why: usize, ns: u64) {
    STOPS[why.min(STOPS.len() - 1)].fetch_add(1, Ordering::Relaxed);
    STATS[SCAN_NS].fetch_add(ns, Ordering::Relaxed);
    STATS[WAIT_SCANS].fetch_add(1, Ordering::Relaxed);
    STATS[WAIT_CMDS].fetch_add(commands as u64, Ordering::Relaxed);
}

/// Runs a slot taken for preparation (state RUNNING) and posts its result.
fn run_slot(slot: &Slot) {
    {
        // A panic must not leave the slot running (the command processor
        // would wait for it forever).
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match slot.regs.kind {
            StageKind::Vertex => shaders::prepare_ngg(&slot.regs),
            StageKind::Pixel => shaders::prepare_ps(&slot.regs),
            StageKind::Compute => shaders::prepare_compute(&slot.regs, slot.wave64),
        }))
        .unwrap_or_else(|_| Err("preparation panicked".to_string()));
        let r = if slot.regs.missed.load(Ordering::Relaxed) {
            // Every such preparation is redone on the command processor:
            // `shaders::reg_lists` must list what the stage's preparation reads.
            let reg = slot.regs.missed_reg.load(Ordering::Relaxed);
            static WARNED: AtomicU32 = AtomicU32::new(0);
            if WARNED.fetch_or(1 << slot.regs.kind as u32, Ordering::Relaxed) & (1 << slot.regs.kind as u32) == 0 {
                crate::warn!("gpu: a {:?} stage preparation read register {reg:#x} outside its snapshot (shaders::reg_lists): such preparations made ahead are not used", slot.regs.kind);
            }
            Err("a register outside the snapshot was read".to_string())
        } else {
            r
        };
        // (Its bindings, for the checks made after the result is posted.)
        let found = r.as_ref().ok().filter(|_| slot.precheck).map(|p| p.found.clone());
        *slot.result.lock() = Some(r);
        slot.state.store(DONE, Ordering::Release);
        slot.cv.notify_all();
        if let Some(found) = found {
            let t = std::time::Instant::now();
            super::exec::fe2_precheck_stage(&found);
            STATS[PRECHECKS].fetch_add(1, Ordering::Relaxed);
            STATS[PRECHECK_NS].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
    }
}

/// Hands `slots` to the workers.
pub fn submit(slots: &[Arc<Slot>]) {
    if slots.is_empty() {
        return;
    }
    ensure_workers();
    STATS[SCANS].fetch_add(1, Ordering::Relaxed);
    STATS[JOBS].fetch_add(slots.len() as u64, Ordering::Relaxed);
    POOL.jobs.lock().extend(slots.iter().cloned());
    POOL.cv.notify_all();
}

/// The preparation made ahead in `slot` if it holds for the live registers
/// `live` (and `wave64` for compute stages) and the memory now; None: the
/// caller prepares the stage itself.
pub fn take<R: RegRead + ?Sized>(slot: &Slot, live: &R, wave64: bool) -> Option<Prepared> {
    if !slot.regs.matches(live) || (slot.regs.kind == StageKind::Compute && slot.wave64 != wave64) {
        slot.cancel();
        STATS[STALE_REGS].fetch_add(1, Ordering::Relaxed);
        return None;
    }
    // Not started yet: preparing it here is quicker than waiting.
    if slot.state.compare_exchange(QUEUED, CANCELLED, Ordering::AcqRel, Ordering::Acquire).is_ok() {
        STATS[TAKEN_OVER].fetch_add(1, Ordering::Relaxed);
        return None;
    }
    // Running: it usually finishes within microseconds, sooner than a
    // blocked thread would be woken.
    let mut spins = 0;
    while slot.state.load(Ordering::Acquire) == RUNNING && spins < 4096 {
        std::hint::spin_loop();
        spins += 1;
    }
    let r = {
        let mut g = slot.result.lock();
        if g.is_none() {
            let t = std::time::Instant::now();
            while g.is_none() {
                slot.cv.wait(&mut g);
            }
            STATS[WAITED_NS].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
            STATS[WAITS].fetch_add(1, Ordering::Relaxed);
        }
        g.take()?
    };
    let p = match r {
        Ok(p) => p,
        Err(e) => {
            STATS[ERRORS].fetch_add(1, Ordering::Relaxed);
            note_failure(slot.regs.kind, &e);
            return None;
        }
    };
    if !shaders::program(p.addr).is_ok_and(|(insts, _)| Arc::ptr_eq(&insts, &p.insts.0)) {
        STATS[STALE_PROGRAM].fetch_add(1, Ordering::Relaxed);
        return None;
    }
    // Depth or color views of the sampled images were chosen with the image
    // cache's state (`images::Images::depth_view`).
    if !shaders::depth_views_hold(&p) {
        STATS[STALE_IMAGES].fetch_add(1, Ordering::Relaxed);
        return None;
    }
    if !shaders::words_unchanged(&p) {
        STATS[STALE_WORDS].fetch_add(1, Ordering::Relaxed);
        return None;
    }
    STATS[USED].fetch_add(1, Ordering::Relaxed);
    Some(p)
}

/// One status line ("" before any scan).
pub fn status() -> String {
    let s: Vec<u64> = STATS.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    if s[SCANS] == 0 && s[WAIT_SCANS] == 0 {
        return String::new();
    }
    let mut stops: Vec<(u64, usize)> = STOPS.iter().enumerate().map(|(i, c)| (c.load(Ordering::Relaxed), i)).filter(|x| x.0 > 0).collect();
    stops.sort_by(|a, b| b.cmp(a));
    let name = |i: usize| match i {
        END_BUFFER => "buffer end".to_string(),
        END_MAX => "limit".to_string(),
        END_INVALID => "unmapped".to_string(),
        END_PREDICATED => "predicated".to_string(),
        END_HEADER => "other header".to_string(),
        END_ARGS => "indirect arguments".to_string(),
        i if i >= 0x100 => format!("private {:#x}", i - 0x100),
        i => format!("{i:#x}"),
    };
    let stops: Vec<String> = stops.iter().take(8).map(|(n, i)| format!("{} {n}", name(*i))).collect();
    let mut why: Vec<(String, u64)> = FAILED_WHY.lock().as_ref().map(|m| m.iter().map(|(k, v)| (k.clone(), *v)).collect()).unwrap_or_default();
    why.sort_by(|a, b| b.1.cmp(&a.1));
    let why: Vec<String> = why.iter().take(6).map(|(k, n)| format!("{n} {k}")).collect();
    format!(
        "    preparations ahead: {} scans ({} ms; ended by {}), {} stages handed to {} workers, {} scans at label waits found {} commands, {} stages prepared while waiting ({} ms), {} used, {} prepared here instead (not started), stale: {} registers, {} program, {} GPU images, {} words; {} failed ({}), {} dropped; {} waits for a worker ({} ms); placement checks made where prepared {} ({} ms)\n",
        s[SCANS],
        s[SCAN_NS] / 1_000_000,
        stops.join(", "),
        s[JOBS],
        WORKERS.load(Ordering::Relaxed),
        s[WAIT_SCANS],
        s[WAIT_CMDS],
        s[HERE],
        s[HERE_NS] / 1_000_000,
        s[USED],
        s[TAKEN_OVER],
        s[STALE_REGS],
        s[STALE_PROGRAM],
        s[STALE_IMAGES],
        s[STALE_WORDS],
        s[ERRORS],
        why.join("; "),
        s[DROPPED],
        s[WAITS],
        s[WAITED_NS] / 1_000_000,
        s[PRECHECKS],
        s[PRECHECK_NS] / 1_000_000
    )
}
