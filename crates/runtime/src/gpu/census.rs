//! Diagnostics (`DS_PROFILE_GPU`): the buffer uses of each 64 MiB window of
//! the residency span (`resident::MIRROR_BASE..MIRROR_END`) since the
//! profile's last reset. A video-memory mirror in default mode must keep
//! the pages it holds out of GPU imports (protecting imported pages lost the
//! device), and imports cover whole windows: the census tells how many of
//! the read-only bindings lie in windows that GPU work or the command
//! processors also write (which keep needing imports until written data
//! moves into the mirror too).

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

/// The census runs (`DS_CENSUS=1` with `DS_PROFILE_GPU=1`): its notes (page
/// bitmaps per binding) cost a cull's bind step +4.2 us and CPU per command
/// +7% (2026-10-23), so profiles of per-step costs run without it.
pub fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| super::exec::profiling_on() && std::env::var("DS_CENSUS").ok().as_deref() == Some("1"))
}

/// Use kinds (bits of a window's flags).
pub const RO: u32 = 1;
pub const RO_UNSIZED: u32 = 2;
pub const WRITTEN: u32 = 4;
pub const CP_WRITE: u32 = 8;
pub const FILL: u32 = 16;
pub const INDEX: u32 = 32;
const KINDS: usize = 6;
const NAMES: [&str; KINDS] = ["read-only", "read-only unsized", "written", "CP writes", "fills", "index buffers"];

const SHIFT: u32 = 26;
const BASE: u64 = super::resident::MIRROR_BASE;
const WINDOWS: usize = ((super::resident::MIRROR_END - super::resident::MIRROR_BASE) >> SHIFT) as usize;

static FLAGS: [AtomicU32; WINDOWS] = [const { AtomicU32::new(0) }; WINDOWS];
/// Uses per window and kind.
static COUNTS: [[AtomicU64; KINDS]; WINDOWS] = [const { [const { AtomicU64::new(0) }; KINDS] }; WINDOWS];
/// Uses outside the span, per kind.
static OUTSIDE: [AtomicU64; KINDS] = [const { AtomicU64::new(0) }; KINDS];

/// Notes a use of guest [base, base+size) of `kind` (one bit).
pub fn note(base: u64, size: u64, kind: u32) {
    let k = kind.trailing_zeros() as usize;
    if k >= KINDS {
        return;
    }
    note_pages(base, size, kind);
    let end = base.saturating_add(size.max(1));
    if base < BASE || end > super::resident::MIRROR_END {
        OUTSIDE[k].fetch_add(1, Ordering::Relaxed);
        return;
    }
    let (lo, hi) = (((base - BASE) >> SHIFT) as usize, ((end - 1 - BASE) >> SHIFT) as usize);
    for w in lo..=hi.min(WINDOWS - 1) {
        FLAGS[w].fetch_or(kind, Ordering::Relaxed);
        COUNTS[w][k].fetch_add(1, Ordering::Relaxed);
    }
}

pub fn reset() {
    for w in 0..WINDOWS {
        FLAGS[w].store(0, Ordering::Relaxed);
        for k in 0..KINDS {
            COUNTS[w][k].store(0, Ordering::Relaxed);
        }
    }
    for c in &OUTSIDE {
        c.store(0, Ordering::Relaxed);
    }
    reset_pages();
}

fn kinds(f: u32) -> String {
    (0..KINDS).filter(|k| f & (1 << k) != 0).map(|k| NAMES[k]).collect::<Vec<_>>().join("+")
}

/// The census as profile lines.
pub fn status() -> String {
    let writes = WRITTEN | CP_WRITE | FILL;
    let reads = RO | RO_UNSIZED | INDEX;
    let mut used = 0;
    let (mut clean_w, mut mixed_w) = (0, 0);
    // Read uses (by kind) in windows without writes / with writes.
    let mut clean = [0u64; KINDS];
    let mut mixed = [0u64; KINDS];
    let mut rows: Vec<(u64, usize, u32)> = Vec::new();
    for w in 0..WINDOWS {
        let f = FLAGS[w].load(Ordering::Relaxed);
        if f == 0 {
            continue;
        }
        used += 1;
        let c: Vec<u64> = COUNTS[w].iter().map(|x| x.load(Ordering::Relaxed)).collect();
        let r: u64 = (0..KINDS).filter(|&k| (1u32 << k) & reads != 0).map(|k| c[k]).sum();
        if f & reads != 0 {
            if f & writes != 0 {
                mixed_w += 1;
            } else {
                clean_w += 1;
            }
        }
        for k in 0..KINDS {
            if (1u32 << k) & reads != 0 {
                if f & writes != 0 {
                    mixed[k] += c[k];
                } else {
                    clean[k] += c[k];
                }
            }
        }
        rows.push((r, w, f));
    }
    rows.sort_by(|a, b| b.0.cmp(&a.0));
    let part = |v: &[u64; KINDS]| {
        [RO, RO_UNSIZED, INDEX].iter().map(|&b| format!("{} {}", NAMES[b.trailing_zeros() as usize], v[b.trailing_zeros() as usize])).collect::<Vec<_>>().join(", ")
    };
    let top: Vec<String> = rows
        .iter()
        .take(16)
        .map(|&(r, w, f)| {
            let c: Vec<u64> = COUNTS[w].iter().map(|x| x.load(Ordering::Relaxed)).collect();
            let wr: u64 = (0..KINDS).filter(|&k| (1u32 << k) & writes != 0).map(|k| c[k]).sum();
            format!("{:#x} {} reads {r} writes {wr}", BASE + ((w as u64) << SHIFT), kinds(f))
        })
        .collect();
    let out: Vec<String> = (0..KINDS).map(|k| format!("{} {}", NAMES[k], OUTSIDE[k].load(Ordering::Relaxed))).collect();
    format!(
        "    buffer uses by 64 MiB window (census): {used} windows used; reads in {clean_w} windows nothing writes: {}; reads in {mixed_w} windows also written (bindings, CP writes, fills): {}; outside the span: {}\n      busiest windows: {}\n",
        part(&clean),
        part(&mixed),
        out.join(", "),
        top.join("; ")
    ) + &pages_status()
}

/// Pages (4 KiB) of the span GPU work or the command processors wrote since
/// the reset: a bitmap, made on first use.
static PAGES: std::sync::OnceLock<Box<[AtomicU64]>> = std::sync::OnceLock::new();
/// Read uses (per kind) whose pages were / were not written so far.
static PAGE_HIT: [[AtomicU64; KINDS]; 2] = [const { [const { AtomicU64::new(0) }; KINDS] }; 2];

fn pages() -> &'static [AtomicU64] {
    PAGES.get_or_init(|| (0..((super::resident::MIRROR_END - BASE) >> 18)).map(|_| AtomicU64::new(0)).collect())
}

/// Page-granular census: writes mark their pages, reads count whether any
/// of their pages was written before (stable frame to frame at the gate).
pub fn note_pages(base: u64, size: u64, kind: u32) {
    let k = kind.trailing_zeros() as usize;
    let end = base.saturating_add(size.max(1));
    if k >= KINDS || base < BASE || end > super::resident::MIRROR_END {
        return;
    }
    let (lo, hi) = ((base - BASE) >> 12, (end - 1 - BASE) >> 12);
    let bm = pages();
    if kind & (WRITTEN | CP_WRITE | FILL) != 0 {
        let mut p = lo;
        while p <= hi {
            let (w, b) = ((p >> 6) as usize, p & 63);
            let n = (64 - b).min(hi - p + 1);
            let mask = if n == 64 { u64::MAX } else { ((1u64 << n) - 1) << b };
            bm[w].fetch_or(mask, Ordering::Relaxed);
            p += n;
        }
        return;
    }
    let mut p = lo;
    let mut hit = false;
    while p <= hi {
        let (w, b) = ((p >> 6) as usize, p & 63);
        let n = (64 - b).min(hi - p + 1);
        let mask = if n == 64 { u64::MAX } else { ((1u64 << n) - 1) << b };
        if bm[w].load(Ordering::Relaxed) & mask != 0 {
            hit = true;
            break;
        }
        p += n;
    }
    PAGE_HIT[hit as usize][k].fetch_add(1, Ordering::Relaxed);
}

pub fn reset_pages() {
    if let Some(bm) = PAGES.get() {
        for w in bm.iter() {
            w.store(0, Ordering::Relaxed);
        }
    }
    for row in &PAGE_HIT {
        for c in row {
            c.store(0, Ordering::Relaxed);
        }
    }
}

pub fn pages_status() -> String {
    let v = |h: usize, b: u32| PAGE_HIT[h][b.trailing_zeros() as usize].load(Ordering::Relaxed);
    format!(
        "    reads by whether GPU work or CP writes wrote any of their 4 KiB pages (census): read-only {} clean / {} written, unsized {} / {}, index buffers {} / {}\n",
        v(0, RO),
        v(1, RO),
        v(0, RO_UNSIZED),
        v(1, RO_UNSIZED),
        v(0, INDEX),
        v(1, INDEX)
    )
}
