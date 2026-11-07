//! Census of the graphics queue's runs through their handshakes
//! (`DS_HS_CENSUS=1`, diagnosis; STATUS.md next step 1): what a run that
//! went on through cache actions, partial flushes and label handshakes
//! would meet. The graphics command processor's stream is cut into
//! stretches at the packets such runs would still end at (`Cp::
//! hs_census_packet`); within a stretch each serial command and each
//! parallel run is a segment, and the GPU writes its commands record
//! (`Exec::note_write`) are logged by segment. A CPU-side read
//! (`Exec::sync_for`) that overlaps a write of an earlier segment of its
//! stretch is counted by why it reads and by whether a forwarded copy covers
//! it now: in a merged run its writer would be an earlier chunk, recorded
//! by another thread, perhaps not yet.

use parking_lot::Mutex;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

pub fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_HS_CENSUS").ok().as_deref() == Some("1"))
}

/// The newest segment id (0: none).
static SEG: AtomicU64 = AtomicU64::new(1);
/// The stretch's GPU writes: start, end, segment, writer program.
static LOG: Mutex<Vec<(u64, u64, u64, u64)>> = Mutex::new(Vec::new());
/// Stretches, segments, reads checked, reads hitting an earlier segment's
/// write with a forwarded copy covering it, without; writes logged, writes
/// dropped (log full).
static N: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];
/// Uncovered hits by why (`Exec::sync_for`'s reasons: CP read, dispatch
/// shader read, dispatch texture, draw shader read, draw texture, draw
/// indices).
static WHY: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];
/// Covered hits by why.
static WHY_FWD: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];
/// Uncovered hits by writer and reader program (the most frequent shown).
static PAIRS: Mutex<Option<std::collections::HashMap<(u64, u64, usize), u64>>> = Mutex::new(None);
const LOG_MAX: usize = 1 << 15;

/// A packet a merged run would still end at: the stretch ends.
pub fn boundary() {
    let mut g = LOG.lock();
    if !g.is_empty() {
        g.clear();
    }
    N[0].fetch_add(1, Ordering::Relaxed);
}

/// A new segment (a serial command or a parallel run): its id.
pub fn next_segment() -> u64 {
    N[1].fetch_add(1, Ordering::Relaxed);
    SEG.fetch_add(1, Ordering::Relaxed) + 1
}

/// A GPU write a command of segment `seg` records.
pub fn wrote(a: u64, n: u64, seg: u64, writer: u64) {
    let mut g = LOG.lock();
    if g.len() >= LOG_MAX {
        N[6].fetch_add(1, Ordering::Relaxed);
        return;
    }
    g.push((a, a.saturating_add(n), seg, writer));
    N[5].fetch_add(1, Ordering::Relaxed);
}

/// CPU-side reads by a command of segment `seg` (`why` as `Exec::sync_for`).
pub fn reads(reads: &[(u64, u64)], seg: u64, why: usize, reader: u64) {
    let g = LOG.lock();
    for &(a, n) in reads {
        N[2].fetch_add(1, Ordering::Relaxed);
        let Some(&(_, _, _, writer)) = g.iter().rev().find(|&&(w, e, s, _)| s < seg && a < e && w < a.saturating_add(n)) else { continue };
        let covered = super::forward::uncovered(&[(a, n)]).is_some_and(|v| v.is_empty());
        let k = why.min(5);
        if covered {
            N[3].fetch_add(1, Ordering::Relaxed);
            WHY_FWD[k].fetch_add(1, Ordering::Relaxed);
        } else {
            N[4].fetch_add(1, Ordering::Relaxed);
            WHY[k].fetch_add(1, Ordering::Relaxed);
            *PAIRS.lock().get_or_insert_with(Default::default).entry((writer, reader, k)).or_insert(0) += 1;
        }
    }
}

/// The census as a status line (with `profile report`).
pub fn line() -> String {
    if !on() {
        return String::new();
    }
    let flips = (super::FRAMES.load(Ordering::Relaxed)).max(1) as f64;
    let v: Vec<u64> = N.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    const NAMES: [&str; 6] = ["CP read", "dispatch shader read", "dispatch texture", "draw shader read", "draw texture", "draw indices"];
    let by = |w: &[AtomicU64; 6]| NAMES.iter().zip(w.iter()).map(|(n, c)| format!("{n} {}", c.load(Ordering::Relaxed))).collect::<Vec<_>>().join(", ");
    let mut pairs: Vec<((u64, u64, usize), u64)> = PAIRS.lock().as_ref().map(|m| m.iter().map(|(k, v)| (*k, *v)).collect()).unwrap_or_default();
    pairs.sort_by(|a, b| b.1.cmp(&a.1));
    let top: Vec<String> = pairs.iter().take(6).map(|((w, r, k), c)| format!("{w:#x} -> {r:#x} ({}) {c}", NAMES[*k])).collect();
    let mut writers: std::collections::HashMap<u64, (u64, u64)> = Default::default();
    for ((w, _, _), c) in &pairs {
        let e = writers.entry(*w).or_insert((0, 0));
        e.0 += c;
        e.1 += 1;
    }
    let mut writers: Vec<(u64, (u64, u64))> = writers.into_iter().collect();
    writers.sort_by(|a, b| b.1.0.cmp(&a.1.0));
    let top = format!("{}; by writer ({} writers): {}", top.join("; "), writers.len(), writers.iter().take(10).map(|(w, (c, r))| format!("{w:#x} {c} ({r} readers)")).collect::<Vec<_>>().join(", "));
    format!(
        "    handshake census (since boot, {:.0} flips): {} stretches, {} segments, {} reads checked; reads of an earlier segment's GPU write: {} covered by forwarded copies ({}), {} not ({}); {} writes logged, {} dropped; uncovered by writer -> reader: {}\n",
        flips,
        v[0],
        v[1],
        v[2],
        v[3],
        by(&WHY_FWD),
        v[4],
        by(&WHY),
        v[5],
        v[6],
        top
    )
}
