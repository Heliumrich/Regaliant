//! Measurement (`DS_PROBE_XWAIT=1`): could the graphics queue's command
//! processor record past its waits for labels a compute queue writes? At
//! the gate the graphics CP stops at a label queue 0x50 writes after a burst
//! of work and goes on once 0x50 has recorded that write (`promises`): ~13
//! ms per frame of one thread waiting while another records (FINDINGS.md
//! 2026-10-20). Recording on speculatively (its batches held back until the
//! label's write is recorded) only pays if the commands after such a wait do
//! not read on the CPU what the writer queue's work during the wait writes.
//!
//! A window opens when the graphics CP starts waiting on a label a compute
//! queue wrote last; the writes that queue's bindings record until the wait
//! ends are collected; after it, the CPU reads the graphics queue's
//! recorders check against pending GPU writes (`Exec::sync_for`) are tested
//! against them until the graphics CP's next such wait. Per wait: its
//! length, and the time and commands from its end to the first dependent
//! read (none: the whole window). `status` sums them up.

use super::ranges::Ranges;
use parking_lot::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
/// A window is open (fast check for the hooks).
static ACTIVE: AtomicBool = AtomicBool::new(false);

pub fn on() -> bool {
    *ON.get_or_init(|| std::env::var("DS_PROBE_XWAIT").ok().as_deref() == Some("1"))
}

struct Window {
    /// The queue that wrote the awaited label last.
    writer: String,
    started: Instant,
    /// The wait ended (then reads are tested), and its length in us.
    ended: Option<Instant>,
    wait_us: u64,
    /// The writer queue's written bindings during the wait.
    writes: Ranges,
    /// Graphics CP commands recorded since the wait ended.
    cmds: u64,
    /// The first dependent read after the wait: (us after its end, commands, why).
    first_dep: Option<(u64, u64, usize)>,
}

static WINDOW: Mutex<Option<Window>> = Mutex::new(None);

/// Per writer queue: [waits, wait us, waits with a dependent read, us of the
/// waits before their first dependent read (overlappable), unused, commands
/// before the first dependent read (summed over the waits with one), waits
/// whose window held no writes].
static STATS: Mutex<Vec<(String, [u64; 7])>> = Mutex::new(Vec::new());
/// Dependent reads by kind (`Exec::sync_for`'s why).
static DEP_WHY: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

fn close(w: Window) {
    let mut g = STATS.lock();
    let i = match g.iter().position(|e| e.0 == w.writer) {
        Some(i) => i,
        None => {
            g.push((w.writer.clone(), [0; 7]));
            g.len() - 1
        }
    };
    let s = &mut g[i].1;
    s[0] += 1;
    s[1] += w.wait_us;
    if w.writes.is_empty() {
        s[6] += 1;
    }
    match w.first_dep {
        Some((us, cmds, _)) => {
            s[2] += 1;
            s[3] += us.min(w.wait_us);
            s[5] += cmds;
        }
        None => s[3] += w.wait_us,
    }
}

/// The graphics CP starts waiting on a label `writer` wrote last.
pub fn wait_start(writer: &str) {
    if !writer.starts_with("gpu-acb") {
        return;
    }
    let mut g = WINDOW.lock();
    if let Some(w) = g.take() {
        if w.ended.is_some() {
            close(w);
        }
    }
    *g = Some(Window { writer: writer.to_string(), started: Instant::now(), ended: None, wait_us: 0, writes: Ranges::default(), cmds: 0, first_dep: None });
    ACTIVE.store(true, Ordering::Release);
}

/// The graphics CP's wait ended.
pub fn wait_end() {
    if !ACTIVE.load(Ordering::Acquire) {
        return;
    }
    let mut g = WINDOW.lock();
    if let Some(w) = g.as_mut() {
        if w.ended.is_none() {
            w.wait_us = w.started.elapsed().as_micros() as u64;
            w.ended = Some(Instant::now());
        }
    }
}

/// A binding of queue `name` writes [base, base+size).
pub fn note_write(name: &str, base: u64, size: u64) {
    if !ACTIVE.load(Ordering::Acquire) {
        return;
    }
    let mut g = WINDOW.lock();
    if let Some(w) = g.as_mut() {
        if w.ended.is_none() && w.writer == name {
            w.writes.insert(base, size);
        }
    }
}

/// The graphics CP records a draw or dispatch.
pub fn note_command(name: &str) {
    if !ACTIVE.load(Ordering::Acquire) || name != "gpu-gfx" {
        return;
    }
    if let Some(w) = WINDOW.lock().as_mut() {
        if w.ended.is_some() && w.first_dep.is_none() {
            w.cmds += 1;
        }
    }
}

/// A recorder of the graphics queue (`name` "gpu-gfx...") checks CPU reads
/// against pending GPU writes (`why`: `Exec::sync_for`'s reason).
pub fn note_reads(name: &str, reads: &[(u64, u64)], why: usize) {
    if !ACTIVE.load(Ordering::Acquire) || !name.starts_with("gpu-gfx") {
        return;
    }
    let mut g = WINDOW.lock();
    let Some(w) = g.as_mut() else { return };
    let Some(end) = w.ended else { return };
    if w.writes.is_empty() || !w.writes.overlaps_any(reads) {
        return;
    }
    DEP_WHY[why.min(5)].fetch_add(1, Ordering::Relaxed);
    if w.first_dep.is_none() {
        w.first_dep = Some((end.elapsed().as_micros() as u64, w.cmds, why));
    }
}

/// Starts the totals again (`profile reset`).
pub fn reset() {
    STATS.lock().clear();
    for c in &DEP_WHY {
        c.store(0, Ordering::Relaxed);
    }
}

/// For the status report and profile windows.
pub fn status() -> String {
    if !on() {
        return String::new();
    }
    let g = STATS.lock();
    let mut out = String::new();
    for (q, s) in g.iter() {
        let avg_cmds = if s[2] > 0 { s[5] as f64 / s[2] as f64 } else { 0.0 };
        out.push_str(&format!(
            "    graphics CP waits for labels of {q}: {} ({:.1} ms), {} with a CPU read of the queue's writes during the wait in the commands after it (~{avg_cmds:.0} commands in first), {} with no writes; overlappable before the first such read {:.1} ms ({:.0}%)\n",
            s[0],
            s[1] as f64 / 1000.0,
            s[2],
            s[6],
            s[3] as f64 / 1000.0,
            100.0 * s[3] as f64 / s[1].max(1) as f64
        ));
    }
    let why: Vec<u64> = DEP_WHY.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    out.push_str(&format!("    dependent reads by kind [CP read, dispatch shader read, dispatch texture, draw shader read, draw texture, draw indices]: {why:?}\n"));
    out
}
