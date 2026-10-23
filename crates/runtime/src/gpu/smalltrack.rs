//! Diagnosis (`DS_SMALL_TRACK=1`): who uses the guest memory of small
//! written-buffer shadows (windows under 256 KiB) after their merge.
//!
//! The merges of these windows cost ~13% of a gate frame's GPU time in
//! replay (FINDINGS.md 2026-10-23); keeping them in video memory instead
//! needs to know which users read the guest copy: the writing queue's own
//! later bindings, other queues' bindings (by placement), and the CPU-side
//! reads `Exec::sync_for` serves (CP reads, shader preparation, textures,
//! draw indices). Each window is registered at its first merge; every later
//! binding and read overlapping it is counted. `report` gives the windows
//! merged most and totals over all of them.

use parking_lot::Mutex;
use std::collections::{BTreeMap, HashMap};

/// Windows below this size are tracked (the merges the A/B priced).
pub const SMALL: u64 = 256 << 10;

pub fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_SMALL_TRACK").ok().as_deref() == Some("1"))
}

#[derive(Default)]
struct Win {
    /// Queue of the first merge.
    writer: usize,
    merges: u64,
    /// Merges by another queue than the first one.
    merges_foreign: u64,
    /// Bindings by the writing queue: [shadowed (written), read-only].
    own: [u64; 2],
    /// The writing queue's read-only bindings by placement.
    own_ro: [u64; 11],
    /// Bindings by other queues: written, then read-only by placement
    /// (exec's PLACEMENTS order).
    foreign_written: u64,
    foreign_ro: [u64; 11],
    /// Foreign read-only bindings by their size: [up to 256 KiB, up to 16
    /// MiB, larger], and the last tight one's shader.
    foreign_sizes: [u64; 3],
    foreign_shader: u64,
    /// CPU-side reads by size, as for bindings.
    cpu_sizes: [u64; 3],
    /// Foreign read-only bindings over 256 KiB that are unbounded pointers.
    foreign_unbounded: u64,
    /// CP writes into it (CPU-performed effects and GPU-performed writes)
    /// and fills.
    cp_writes: u64,
    fills: u64,
    /// CPU-side reads (`sync_for` kinds), by the writing queue / others.
    cpu_own: [u64; 6],
    cpu_foreign: [u64; 6],
}

#[derive(Default)]
struct State {
    /// The flip the window began at.
    first_flip: u64,
    queues: Vec<String>,
    wins: HashMap<(u64, u64), Win>,
    /// Window starts: (start, size) keys by start.
    starts: BTreeMap<u64, Vec<u64>>,
}

static ST: Mutex<Option<State>> = Mutex::new(None);

fn queue_id(s: &mut State, q: &str) -> usize {
    // A parallel worker's chunks count as its command processor's.
    let q = q.find("-par").map_or(q, |i| &q[..i]);
    match s.queues.iter().position(|x| x == q) {
        Some(i) => i,
        None => {
            s.queues.push(q.to_string());
            s.queues.len() - 1
        }
    }
}

/// The tracked windows overlapping [a, a+n).
fn overlapping(s: &State, a: u64, n: u64) -> Vec<(u64, u64)> {
    let end = a.saturating_add(n);
    let mut out = Vec::new();
    for (&start, sizes) in s.starts.range(a.saturating_sub(SMALL)..end) {
        for &size in sizes {
            if start + size > a {
                out.push((start, size));
            }
        }
    }
    out
}

/// A small shadow's window merged back by `queue`.
pub fn merged(guest: u64, range: u64, queue: &str) {
    if range >= SMALL {
        return;
    }
    let mut g = ST.lock();
    let s = g.get_or_insert_with(|| State { first_flip: super::FRAMES.load(std::sync::atomic::Ordering::Relaxed), ..Default::default() });
    let q = queue_id(s, queue);
    let key = (guest, range);
    if !s.wins.contains_key(&key) {
        s.starts.entry(guest).or_default().push(range);
        s.wins.insert(key, Win { writer: q, ..Default::default() });
    }
    let w = s.wins.get_mut(&key).unwrap();
    w.merges += 1;
    if w.writer != q {
        w.merges_foreign += 1;
    }
}

/// A buffer binding of [base, base+size) by `queue`: written (shadowed or
/// not) or read-only with placement index `placement`.
fn size_class(n: u64) -> usize {
    if n <= SMALL {
        0
    } else if n <= 16 << 20 {
        1
    } else {
        2
    }
}

pub fn bound(base: u64, size: u64, written: bool, placement: usize, queue: &str, shader: u64, unbounded: bool) {
    if size == 0 {
        return;
    }
    let mut g = ST.lock();
    let Some(s) = g.as_mut() else { return };
    let hits = overlapping(s, base, size);
    if hits.is_empty() {
        return;
    }
    let q = queue_id(s, queue);
    for key in hits {
        let w = s.wins.get_mut(&key).unwrap();
        if w.writer == q {
            w.own[if written { 0 } else { 1 }] += 1;
            if !written {
                w.own_ro[placement.min(10)] += 1;
            }
        } else if written {
            w.foreign_written += 1;
        } else {
            w.foreign_ro[placement.min(10)] += 1;
            w.foreign_sizes[size_class(size)] += 1;
            if size <= SMALL {
                w.foreign_shader = shader;
            } else if unbounded {
                w.foreign_unbounded += 1;
            }
        }
    }
}

/// CPU-side reads of guest memory served by `Exec::sync_for` (kind `why`).
pub fn cpu_read(reads: &[(u64, u64)], why: usize, queue: &str) {
    let mut g = ST.lock();
    let Some(s) = g.as_mut() else { return };
    let mut hits: Vec<(u64, u64)> = Vec::new();
    let mut largest = 0;
    for &(a, n) in reads {
        for k in overlapping(s, a, n) {
            largest = largest.max(n);
            if !hits.contains(&k) {
                hits.push(k);
            }
        }
    }
    if hits.is_empty() {
        return;
    }
    let q = queue_id(s, queue);
    for key in hits {
        let w = s.wins.get_mut(&key).unwrap();
        if w.writer == q {
            w.cpu_own[why.min(5)] += 1;
        } else {
            w.cpu_foreign[why.min(5)] += 1;
        }
        w.cpu_sizes[size_class(largest)] += 1;
    }
}

/// A CP write (`fill`: a fill) into [a, a+n).
pub fn cp_write(a: u64, n: u64, fill: bool) {
    let mut g = ST.lock();
    let Some(s) = g.as_mut() else { return };
    for key in overlapping(s, a, n) {
        let w = s.wins.get_mut(&key).unwrap();
        if fill {
            w.fills += 1;
        } else {
            w.cp_writes += 1;
        }
    }
}

/// The windows merged most (per flip) and totals.
pub fn report() -> String {
    let g = ST.lock();
    let Some(s) = g.as_ref() else { return String::new() };
    let flips = super::FRAMES.load(std::sync::atomic::Ordering::Relaxed).saturating_sub(s.first_flip);
    let per = |x: u64| x as f64 / flips.max(1) as f64;
    let mut v: Vec<(&(u64, u64), &Win)> = s.wins.iter().collect();
    v.sort_by(|a, b| b.1.merges.cmp(&a.1.merges));
    const KINDS: [&str; 6] = ["cp", "cs reads", "cs textures", "draw reads", "draw textures", "draw indices"];
    let kinds = |c: &[u64; 6]| -> String {
        let parts: Vec<String> = c.iter().enumerate().filter(|(_, n)| **n > 0).map(|(k, n)| format!("{} {:.1}", KINDS[k], per(*n))).collect();
        if parts.is_empty() { "-".into() } else { parts.join(", ") }
    };
    let mut out = format!("    small shadow windows after their merge ({} windows, merges {:.1} per flip):\n", s.wins.len(), per(s.wins.values().map(|w| w.merges).sum()));
    let tot = |f: &dyn Fn(&Win) -> u64| s.wins.values().map(f).sum::<u64>();
    let with = |f: &dyn Fn(&Win) -> bool| s.wins.values().filter(|w| f(w)).count();
    out += &format!(
        "      totals per flip: own shadowed binds {:.1}, own read-only {:.1}, foreign written {:.1}, foreign read-only {:.1}, foreign merges {:.1}; CPU reads own [{}], foreign [{}]; windows with foreign binds {}, with CPU reads {}\n",
        per(tot(&|w| w.own[0])),
        per(tot(&|w| w.own[1])),
        per(tot(&|w| w.foreign_written)),
        per(tot(&|w| w.foreign_ro.iter().sum())),
        per(tot(&|w| w.merges_foreign)),
        kinds(&std::array::from_fn(|k| tot(&|w| w.cpu_own[k]))),
        kinds(&std::array::from_fn(|k| tot(&|w| w.cpu_foreign[k]))),
        with(&|w| w.foreign_written + w.foreign_ro.iter().sum::<u64>() > 0),
        with(&|w| w.cpu_own.iter().chain(w.cpu_foreign.iter()).sum::<u64>() > 0),
    );
    out += &format!(
        "      tight users (at most 256 KiB) per flip: foreign read-only binds {:.1} in {} windows, CPU reads {:.1} in {} windows; merges of windows with neither {:.1}\n",
        per(tot(&|w| w.foreign_sizes[0])),
        with(&|w| w.foreign_sizes[0] > 0),
        per(tot(&|w| w.cpu_sizes[0])),
        with(&|w| w.cpu_sizes[0] > 0),
        per(s.wins.values().filter(|w| w.foreign_sizes[0] == 0 && w.cpu_sizes[0] == 0 && w.foreign_written == 0).map(|w| w.merges).sum()),
    );
    out += &format!(
        "      merges per flip of windows: with CPU reads {:.1}, with CP writes {:.1}, with fills {:.1}, with none of them {:.1}; large foreign read-only binds that are unbounded pointers {:.1} of {:.1}; CP writes {:.1}, fills {:.1} per flip\n",
        per(s.wins.values().filter(|w| w.cpu_sizes.iter().sum::<u64>() > 0).map(|w| w.merges).sum()),
        per(s.wins.values().filter(|w| w.cp_writes > 0).map(|w| w.merges).sum()),
        per(s.wins.values().filter(|w| w.fills > 0).map(|w| w.merges).sum()),
        per(s.wins.values().filter(|w| w.cpu_sizes.iter().sum::<u64>() == 0 && w.cp_writes == 0 && w.fills == 0).map(|w| w.merges).sum()),
        per(tot(&|w| w.foreign_unbounded)),
        per(tot(&|w| w.foreign_sizes[1] + w.foreign_sizes[2])),
        per(tot(&|w| w.cp_writes)),
        per(tot(&|w| w.fills)),
    );
    const PL: [&str; 11] = ["host", "snapshot", "pre-copy", "resident", "mirror", "shadow", "shadow reused", "none", "scratch", "home", "mirrored"];
    for &(&(a, n), w) in v.iter().take(24) {
        let ro: Vec<String> = w.foreign_ro.iter().enumerate().filter(|(_, c)| **c > 0).map(|(k, c)| format!("{} {:.1}", PL[k], per(*c))).collect();
        let own_ro: Vec<String> = w.own_ro.iter().enumerate().filter(|(_, c)| **c > 0).map(|(k, c)| format!("{} {:.1}", PL[k], per(*c))).collect();
        out += &format!(
            "      {a:#x}+{n:#x} ({}): merges {:.1} ({:.1} foreign); own binds {:.1} shadowed, {:.1} read-only [{}]; foreign {:.1} written, read-only [{}] by size {:.1}/{:.1}/{:.1} (tight: shader {:#x}); CPU reads own [{}], foreign [{}] by size {:.1}/{:.1}/{:.1}\n",
            s.queues.get(w.writer).map(String::as_str).unwrap_or("?"),
            per(w.merges),
            per(w.merges_foreign),
            per(w.own[0]),
            per(w.own[1]),
            if own_ro.is_empty() { "-".into() } else { own_ro.join(", ") },
            per(w.foreign_written),
            if ro.is_empty() { "-".into() } else { ro.join(", ") },
            per(w.foreign_sizes[0]),
            per(w.foreign_sizes[1]),
            per(w.foreign_sizes[2]),
            w.foreign_shader,
            kinds(&w.cpu_own),
            kinds(&w.cpu_foreign),
            per(w.cpu_sizes[0]),
            per(w.cpu_sizes[1]),
            per(w.cpu_sizes[2]),
        );
    }
    out
}

/// Forgets every window (`profile reset`).
pub fn reset() {
    *ST.lock() = None;
}
