//! The GPU arena's coherence passes as a census (`tune kco`, off by default;
//! FRONTEND3.md "2026-11-09 late night"): front end 3 needs the arena's CPU
//! writes found per batch, and a pass over its whole scope (~73 MiB,
//! `engine::coherence_runs`) costs 150-240 us. Sealing would arm the pages a
//! pass found unwritten for a while, so that passes scan only new and hot
//! pages and a late write faults instead. Whether that pays depends on how
//! old a page is when the CPU writes it: here, after each title submission
//! (coalesced), a helper takes the scope (write watch with a reset, the
//! found pages invalidated in front end 2's mirror as any take does) and
//! records, per page found written, the passes since the page was last
//! allocated (from the allocation table's growth per MiB) or found written.
//! Nothing consumes the passes: front end 2's checks run as without them.

use parking_lot::{Condvar, Mutex};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

/// Triggers not yet answered by a pass started after them.
static PENDING: Mutex<u64> = Mutex::new(0);
/// Pages found written 2 or more passes after their last allocation or
/// write, since the last reset (`engine::late_writes_by_site`).
static LATE: Mutex<Vec<u64>> = Mutex::new(Vec::new());
static CV: Condvar = Condvar::new();

/// [passes, triggers, runs taken, pages taken, written pages found, ns of
/// passes, takes failed, pages newly allocated, pages active within the last
/// two passes (allocated or written, summed per pass), then written pages
/// by age in passes: <= 1, 2, 3-4, 5-8, 9-32, 33+].
pub static STATS: [AtomicU64; 15] = [const { AtomicU64::new(0) }; 15];
const S_PASSES: usize = 0;
const S_TRIGGERS: usize = 1;
const S_RUNS: usize = 2;
const S_PAGES: usize = 3;
const S_FOUND: usize = 4;
const S_NS: usize = 5;
const S_FAILED: usize = 6;
const S_NEW: usize = 7;
const S_ACTIVE: usize = 8;
const S_AGE: usize = 9;

/// Whether the passes run (front end 2 on, `tune kco`).
pub fn on() -> bool {
    super::tune::kco() && super::fe2::on()
}

/// A title submission was made: a pass should start after it.
pub fn trigger() {
    if !on() {
        return;
    }
    static STARTED: std::sync::Once = std::sync::Once::new();
    STARTED.call_once(|| {
        let _ = crate::thread::spawn_host("gpu-kco", 1 << 20, Box::new(run));
    });
    STATS[S_TRIGGERS].fetch_add(1, Ordering::Relaxed);
    let mut g = PENDING.lock();
    *g += 1;
    CV.notify_one();
}

/// The age census's state: per arena MiB its allocation-table entry at the
/// last pass, per page the pass it was last allocated in and last found
/// written in.
#[derive(Default)]
struct Ages {
    mib: HashMap<u64, u64>,
    allocated: HashMap<u64, u64>,
    written: HashMap<u64, u64>,
    /// Pages allocated or written in each of the last two passes.
    recent: [u64; 2],
}

fn run() {
    crate::log::set_thread_name("gpu-kco");
    let mut scope: Vec<(u64, u64)> = Vec::new();
    let mut written: Vec<u64> = Vec::new();
    let mut ages = Ages::default();
    let mut pass = 0u64;
    loop {
        {
            let mut g = PENDING.lock();
            while *g == 0 {
                CV.wait(&mut g);
            }
            *g = 0;
        }
        if !on() {
            continue;
        }
        pass += 1;
        let t = std::time::Instant::now();
        crate::engine::coherence_runs(&mut scope);
        // Pages allocated since the last pass: a MiB's high-water mark grew,
        // or its generation moved (all of it allocated anew).
        let mut new_pages = 0u64;
        let mut mibs: Vec<u64> = Vec::new();
        for &(a, b) in &scope {
            let mut m = a >> 20;
            while m << 20 < b {
                mibs.push(m);
                m += 1;
            }
        }
        mibs.dedup();
        for m in mibs {
            let e = crate::engine::alloc_entry_of(m);
            let old = ages.mib.insert(m, e).unwrap_or(0);
            if old == e {
                continue;
            }
            let (h_new, h_old) = (e & 0x1F_FFFF, old & 0x1F_FFFF);
            let from = if (e >> 32) != (old >> 32) { 0 } else { h_old.min(h_new) };
            let mut p = ((m << 20) + from) & !(super::resident::PAGE - 1);
            while p < (m << 20) + h_new {
                ages.allocated.insert(p, pass);
                new_pages += 1;
                p += super::resident::PAGE;
            }
        }
        let (mut pages, mut found, mut failed, mut runs) = (0u64, 0u64, 0u64, 0u64);
        let mut buckets = [0u64; 6];
        for &(a, b) in &scope {
            let (a, b) = (a & !(super::resident::PAGE - 1), b.next_multiple_of(super::resident::PAGE));
            if !super::fe2::in_span(a, b - a) {
                continue;
            }
            runs += 1;
            let mut x = a;
            while x < b {
                let y = (x + (16 << 20)).min(b);
                written.clear();
                if super::precache::take_written_for(x, y - x, &mut written, None, crate::mem::WW_WHO_FE2).is_some() {
                    pages += (y - x) / super::resident::PAGE;
                    found += written.len() as u64;
                    for &p in &written {
                        let last = ages.allocated.get(&p).copied().unwrap_or(0).max(ages.written.get(&p).copied().unwrap_or(0));
                        let age = if last == 0 { u64::MAX } else { pass - last };
                        buckets[match age {
                            0..=1 => 0,
                            2 => 1,
                            3..=4 => 2,
                            5..=8 => 3,
                            9..=32 => 4,
                            _ => 5,
                        }] += 1;
                        // (Late writes, for their allocations' call sites.)
                        if age >= 2 && age != u64::MAX {
                            let mut l = LATE.lock();
                            if l.len() < 1 << 20 {
                                l.push(p);
                            }
                        }
                        ages.written.insert(p, pass);
                    }
                } else {
                    failed += 1;
                }
                x = y;
            }
        }
        let active = new_pages + found;
        let window = active + ages.recent[0];
        ages.recent = [active, ages.recent[0]];
        if ages.allocated.len() > 1 << 21 || ages.written.len() > 1 << 21 {
            ages.allocated.clear();
            ages.written.clear();
        }
        STATS[S_PASSES].fetch_add(1, Ordering::Relaxed);
        STATS[S_RUNS].fetch_add(runs, Ordering::Relaxed);
        STATS[S_PAGES].fetch_add(pages, Ordering::Relaxed);
        STATS[S_FOUND].fetch_add(found, Ordering::Relaxed);
        STATS[S_FAILED].fetch_add(failed, Ordering::Relaxed);
        STATS[S_NEW].fetch_add(new_pages, Ordering::Relaxed);
        STATS[S_ACTIVE].fetch_add(window, Ordering::Relaxed);
        for (i, &n) in buckets.iter().enumerate() {
            STATS[S_AGE + i].fetch_add(n, Ordering::Relaxed);
        }
        STATS[S_NS].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
}

/// The status line ("" before any pass).
pub fn status() -> String {
    let s: Vec<u64> = STATS.iter().map(|x| x.load(Ordering::Relaxed)).collect();
    if s[S_PASSES] == 0 {
        return String::new();
    }
    let p = s[S_PASSES] as f64;
    let w = s[S_FOUND].max(1) as f64;
    format!(
        "    arena coherence passes (census): {} passes for {} title submissions, {:.1} runs and {:.0} pages a pass, {:.1} us a pass, {} takes failed; a pass: {:.1} pages newly allocated, {:.1} written found, {:.1} pages active within two passes; written pages by passes since their last allocation or write: <=1 {:.1}%, 2 {:.1}%, 3-4 {:.1}%, 5-8 {:.1}%, 9-32 {:.1}%, older or never seen {:.1}%; {}\n",
        s[S_PASSES],
        s[S_TRIGGERS],
        s[S_RUNS] as f64 / p,
        s[S_PAGES] as f64 / p,
        s[S_NS] as f64 / p / 1000.0,
        s[S_FAILED],
        s[S_NEW] as f64 / p,
        s[S_FOUND] as f64 / p,
        s[S_ACTIVE] as f64 / p,
        100.0 * s[S_AGE] as f64 / w,
        100.0 * s[S_AGE + 1] as f64 / w,
        100.0 * s[S_AGE + 2] as f64 / w,
        100.0 * s[S_AGE + 3] as f64 / w,
        100.0 * s[S_AGE + 4] as f64 / w,
        100.0 * s[S_AGE + 5] as f64 / w,
        crate::engine::cb_line()
    ) + &crate::engine::late_writes_by_site(&LATE.lock())
}

/// Statistics start again (`profile reset`).
pub fn reset() {
    for x in STATS.iter() {
        x.store(0, Ordering::Relaxed);
    }
    LATE.lock().clear();
}
