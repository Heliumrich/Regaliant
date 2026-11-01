//! Stage preparations made ahead of execution.
//!
//! While a command processor waits on a label it scans the commands after
//! the wait (`Cp::scan_through`: register writes and draw/dispatch packets)
//! and prepares their stages from a snapshot of the registers they read
//! (`StageRegs`, `prepare_here`); compile-ahead prepares scanned compute
//! stages the same way (`gpu::compile`). At execution a preparation is used
//! when the registers, the program, the image cache's lookup generation and
//! the guest words it read are unchanged (`take`); otherwise the stage is
//! prepared again. The result is the same either way: a preparation is a
//! function of exactly those inputs. (Worker threads preparing the runs
//! after each run's first command, `tune ahead`, cost CPU for no frame time
//! and were removed on 2026-11-01.)

use super::shaders::{self, Prepared, RegRead, StageKind, StageRegs};
use parking_lot::{Condvar, Mutex};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;

/// Slot states.
const QUEUED: u32 = 0;
const RUNNING: u32 = 1;
const DONE: u32 = 2;
/// Not wanted any more (taken over by the command processor or dropped).
const CANCELLED: u32 = 3;

/// One stage preparation made ahead.
pub struct Slot {
    regs: StageRegs,
    wave64: bool,
    state: AtomicU32,
    result: Mutex<Option<Result<Prepared, String>>>,
    cv: Condvar,
}

impl Slot {
    pub fn new(regs: StageRegs, wave64: bool) -> Arc<Slot> {
        Arc::new(Slot { regs, wave64, state: AtomicU32::new(QUEUED), result: Mutex::new(None), cv: Condvar::new() })
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

/// Statistics: indices into STATS.
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
static STATS: [AtomicU64; 17] = [const { AtomicU64::new(0) }; 17];

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
        *slot.result.lock() = Some(r);
        slot.state.store(DONE, Ordering::Release);
        slot.cv.notify_all();
    }
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
    let mut p = match r {
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
    // (Made ahead.)
    p.ahead = 2;
    Some(p)
}

/// One status line ("" before any scan).
pub fn status() -> String {
    let s: Vec<u64> = STATS.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    if s[WAIT_SCANS] == 0 {
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
        "    preparations ahead: {} scans at label waits ({} ms; ended by {}) found {} commands, {} stages prepared while waiting ({} ms), {} used, {} prepared again instead (not started), stale: {} registers, {} program, {} GPU images, {} words; {} failed ({}), {} dropped; {} waits for one in progress ({} ms)\n",
        s[WAIT_SCANS],
        s[SCAN_NS] / 1_000_000,
        stops.join(", "),
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
        s[WAITED_NS] / 1_000_000
    )
}
