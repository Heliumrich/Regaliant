//! Stage H2 of the resident front end (RENDERER.md): how often CPU code
//! touches guest memory that protection through the host would watch, by
//! arming it and counting the faults the host answers (`tune host_census`,
//! hosted runs only). Observation only: write-backs and write-watch checks
//! stay as they are; a faulting page is made read-write again by the host.
//!
//! - bit 1, static data: the ranges of shared batch-start copies (`precache`)
//!   that stayed clean for CLEAN_FLIPS flips are armed read-only: a write
//!   faults (what write tracking by protection would see, stage H4);
//! - bit 2, GPU-written data: windows of up to MAX_WINDOW bytes a batch wrote
//!   are armed no-access when the batch completes: any CPU access faults
//!   (what lazy write-backs would have to answer, stage H3).
//!
//! A window that faulted is armed again COOLDOWN flips later at the
//! earliest, so the census costs at most one fault per window per COOLDOWN
//! flips. Faults are counted by class, by access (read, write) and by whose
//! code made them (the title's, ours, other modules').

use parking_lot::Mutex;
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};

const PAGE: u64 = 4096;
/// Flips a shared copy stays clean before it is armed.
const CLEAN_FLIPS: u64 = 30;
/// Largest GPU-written window armed.
const MAX_WINDOW: u64 = 4 << 20;
/// Smallest GPU-written range armed (labels and timestamps are smaller).
const MIN_WINDOW: u64 = 256;
/// Flips before a window that faulted is armed again.
const COOLDOWN: u64 = 30;

const STATIC: usize = 0;
const GPU: usize = 1;
const CLASS_NAMES: [&str; 2] = ["static data (shared copies clean 30 flips, read-only)", "GPU-written windows (no-access after their batch)"];
const ORIGIN_NAMES: [&str; 3] = ["the title", "ours", "other modules"];

#[derive(Default)]
struct Window {
    class: usize,
    size: u64,
    /// Times armed, times faulted.
    armed: u64,
    faulted: u64,
    /// Not armed again before this flip.
    quiet_until: u64,
    /// Armed now.
    live: bool,
}

#[derive(Default)]
struct Census {
    /// Armed page -> its window's base.
    pages: HashMap<u64, u64>,
    windows: HashMap<u64, Window>,
    /// GPU-written windows of submitted batches, armed when they complete.
    pending: VecDeque<(u64, Vec<(u64, u64)>)>,
    /// Faults by class, access (read, write) and origin.
    faults: [[[u64; 3]; 2]; 2],
    /// Fault lines logged so far.
    logged: u32,
    /// Arming calls that failed.
    failed: u64,
}

static CENSUS: Mutex<Option<Census>> = Mutex::new(None);
/// Write-watch takes of shared copies armed as static data: [takes, ns,
/// takes that found pages written].
static ARMED_TAKES: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
/// Faults drained (for the status line's rate).
static DRAINED: AtomicU64 = AtomicU64::new(0);
static FROM_FLIP: AtomicU64 = AtomicU64::new(u64::MAX);

fn bits() -> u64 {
    if !crate::hosted::on() {
        return 0;
    }
    crate::gpu::tune::host_census()
}

fn flip() -> u64 {
    crate::gpu::FRAMES.load(Ordering::Relaxed)
}

fn arm_window(c: &mut Census, class: usize, base: u64, size: u64) {
    let (lo, hi) = (base & !(PAGE - 1), (base + size).next_multiple_of(PAGE));
    let now = flip();
    let w = c.windows.entry(lo).or_insert_with(|| Window { class, size: hi - lo, ..Default::default() });
    if w.live || w.quiet_until > now || w.class != class {
        return;
    }
    // Pages already armed by another window stay with it.
    if (lo..hi).step_by(PAGE as usize).any(|p| c.pages.contains_key(&p)) {
        return;
    }
    let r = if class == GPU { crate::hosted::arm_noaccess(lo, hi - lo) } else { crate::hosted::arm(lo, hi - lo) };
    if r.is_err() {
        c.failed += 1;
        return;
    }
    let w = c.windows.get_mut(&lo).unwrap();
    w.live = true;
    w.armed += 1;
    for p in (lo..hi).step_by(PAGE as usize) {
        c.pages.insert(p, lo);
    }
    FROM_FLIP.fetch_min(now, Ordering::Relaxed);
}

/// A shared batch-start copy of guest [base, base+size) found current, made
/// at flip `born` (`precache::lookup`).
pub fn copy_used(base: u64, size: u64, born: u64) {
    if bits() & 1 == 0 || flip().saturating_sub(born) < CLEAN_FLIPS {
        return;
    }
    // Recording threads call this per binding: a contended census is skipped.
    let Some(mut g) = CENSUS.try_lock() else { return };
    let c = g.get_or_insert_with(Census::default);
    arm_window(c, STATIC, base, size);
}

/// Whether the shared copy of guest [base, base+size) is armed as static
/// data now (a contended census answers no).
pub fn static_live(base: u64, size: u64) -> bool {
    if bits() & 1 == 0 {
        return false;
    }
    let _ = size;
    let Some(g) = CENSUS.try_lock() else { return false };
    g.as_ref().and_then(|c| c.windows.get(&(base & !(PAGE - 1)))).is_some_and(|w| w.live && w.class == STATIC)
}

/// A write-watch take of an armed static copy (`precache::lookup`): its
/// time, and whether it found pages written.
pub fn note_armed_take(ns: u64, written: bool) {
    ARMED_TAKES[0].fetch_add(1, Ordering::Relaxed);
    ARMED_TAKES[1].fetch_add(ns, Ordering::Relaxed);
    ARMED_TAKES[2].fetch_add(written as u64, Ordering::Relaxed);
}

/// Batch `seq` (submitted) writes these guest ranges.
pub fn gpu_wrote(seq: u64, ranges: &crate::gpu::ranges::Ranges) {
    if bits() & 2 == 0 {
        return;
    }
    // (start, length); label and timestamp writes (the command processors
    // and the title poll them by design) are left out.
    let v: Vec<(u64, u64)> = ranges.iter().filter(|&(_, n)| n >= MIN_WINDOW && n <= MAX_WINDOW).collect();
    if v.is_empty() {
        return;
    }
    CENSUS.lock().get_or_insert_with(Census::default).pending.push_back((seq, v));
}

/// Batches up to `seq` completed (the completion thread): their windows are
/// armed, and the faults the host answered are counted.
pub fn completed(seq: u64) {
    if bits() == 0 {
        return;
    }
    let mut g = CENSUS.lock();
    let c = g.get_or_insert_with(Census::default);
    while c.pending.front().is_some_and(|(s, _)| *s <= seq) {
        let (_, v) = c.pending.pop_front().unwrap();
        for (s, n) in v {
            arm_window(c, GPU, s, n);
        }
    }
    drain_into(c);
}

fn origin(rip: u64) -> usize {
    let (b, e) = crate::exception::host_image();
    if (crate::mem::GUEST_BASE..crate::mem::GUEST_END).contains(&rip) {
        0
    } else if (b..e).contains(&rip) {
        1
    } else {
        2
    }
}

fn drain_into(c: &mut Census) {
    let mut got = Vec::new();
    crate::hosted::drain_faults(&mut got);
    if got.is_empty() {
        return;
    }
    DRAINED.fetch_add(got.len() as u64, Ordering::Relaxed);
    let now = flip();
    for (page, write, rip) in got {
        let Some(base) = c.pages.get(&page).copied() else { continue };
        let Some(w) = c.windows.get_mut(&base) else { continue };
        let o = origin(rip);
        c.faults[w.class][write as usize][o] += 1;
        w.faulted += 1;
        w.live = false;
        w.quiet_until = now + COOLDOWN;
        let (lo, hi) = (base, base + w.size);
        // The window's other pages are disarmed too (one fault per window).
        crate::hosted::disarm(lo, hi - lo);
        for p in (lo..hi).step_by(PAGE as usize) {
            c.pages.remove(&p);
        }
        if c.logged < 40 {
            c.logged += 1;
            let at = if o == 0 { crate::exception::describe_guest(rip) } else { crate::exception::host_symbol(rip) };
            crate::info!("host census: {} {:#x} in {} window {lo:#x}+{:#x} by {} at {at}", if write { "write" } else { "read" }, page, ["static", "GPU-written"][w.class], hi - lo, ORIGIN_NAMES[o]);
        }
    }
}

/// Guest [base, base+len) was remapped (`hosted::forget` cleared its bits):
/// the windows over it are no longer armed; their pages outside it are
/// disarmed (those inside keep the protection the change gave them).
pub fn forget(base: u64, len: u64) {
    drop_windows(base, len, false);
}

/// Kernel-mode I/O on guest [base, base+len) failed on armed pages: the
/// windows over it are disarmed whole.
pub fn release(base: u64, len: u64) {
    drop_windows(base, len, true);
}

fn drop_windows(base: u64, len: u64, whole: bool) {
    if bits() == 0 {
        return;
    }
    let mut g = CENSUS.lock();
    let Some(c) = g.as_mut() else { return };
    let (lo, hi) = (base & !(PAGE - 1), base.saturating_add(len));
    let mut gone: Vec<u64> = c.pages.iter().filter(|(p, _)| **p >= lo && **p < hi).map(|(_, b)| *b).collect();
    gone.sort_unstable();
    gone.dedup();
    for b in gone {
        let Some(w) = c.windows.get_mut(&b) else { continue };
        w.live = false;
        let (wl, wh) = (b, b + w.size);
        if whole {
            crate::hosted::disarm(wl, wh - wl);
        } else {
            if wl < lo {
                crate::hosted::disarm(wl, lo - wl);
            }
            if wh > hi {
                let from = hi.next_multiple_of(PAGE);
                if wh > from {
                    crate::hosted::disarm(from, wh - from);
                }
            }
        }
        for p in (wl..wh).step_by(PAGE as usize) {
            c.pages.remove(&p);
        }
    }
}

/// Status line ("" when off).
pub fn status() -> String {
    if bits() == 0 {
        return String::new();
    }
    let g = CENSUS.lock();
    let Some(c) = g.as_ref() else { return String::new() };
    let flips = flip().saturating_sub(FROM_FLIP.load(Ordering::Relaxed)).max(1);
    let mut s = format!("    host census (`tune host_census` {}, {} flips, {} faults drained, {} arming calls failed):\n", bits(), flips, DRAINED.load(Ordering::Relaxed), c.failed);
    let t: Vec<u64> = ARMED_TAKES.iter().map(|a| a.load(Ordering::Relaxed)).collect();
    if t[0] > 0 {
        s += &format!(
            "      write-watch takes of shared copies armed as static data (what protection would leave out): {} ({:.2} a flip, {:.1} us each, {:.2} ms a flip), {} found pages written\n",
            t[0],
            t[0] as f64 / flips as f64,
            t[1] as f64 / 1000.0 / t[0] as f64,
            t[1] as f64 / 1e6 / flips as f64,
            t[2]
        );
    }
    for (k, name) in CLASS_NAMES.iter().enumerate() {
        let ws: Vec<&Window> = c.windows.values().filter(|w| w.class == k && w.armed > 0).collect();
        if ws.is_empty() {
            continue;
        }
        let armed: u64 = ws.iter().map(|w| w.armed).sum();
        let faulted = ws.iter().filter(|w| w.faulted > 0).count();
        let mib: u64 = ws.iter().map(|w| w.size).sum::<u64>() >> 20;
        let f = c.faults[k];
        let by: Vec<String> = (0..3).map(|o| format!("{} {}/{}", ORIGIN_NAMES[o], f[0][o], f[1][o])).collect();
        let total: u64 = f.iter().flatten().sum();
        s += &format!(
            "      {name}: {} windows ({mib} MiB) armed {armed} times, {faulted} of them faulted; faults {total} ({:.2} a flip; reads/writes by {})\n",
            ws.len(),
            total as f64 / flips as f64,
            by.join(", ")
        );
    }
    s
}
