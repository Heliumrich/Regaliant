//! `DS_FE2_CENSUS=1` (diagnosis): what front end 2's takes cover, for the
//! choice between coherence per region and a mixed backing (STATUS.md,
//! FRONTEND2.md "F3"). Per guest mapping of the mirror's span (as mapped at
//! the window's start; takes go to the mapping holding their start): the
//! takes by caller, the epochs they were made in, the pages they covered
//! and the pages they reported written; per page reported written, the
//! flips it was; per queue, its view raises. `profile report` adds the
//! distinct written pages, those written in at least half or a tenth of the
//! window's flips, and the GPU-written pages front end 2 knows of in each
//! mapping; `profile reset` starts a new window.

use super::resident::{MIRROR_BASE, MIRROR_END};
use parking_lot::{Mutex, RwLock};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

const PAGE: u64 = 4096;
/// Mappings counted at most (the rest go to the last slot; the vista's
/// span holds more than 4,096).
const SLOTS: usize = 1 << 16;

pub fn on() -> bool {
    static W: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *W.get_or_init(|| std::env::var("DS_FE2_CENSUS").ok().as_deref() == Some("1"))
}

/// One mapping's counts: takes by caller (`fe2::CALLER_NAMES`), the
/// distinct epochs taken in and the last (+1), pages covered, pages
/// reported written.
struct Counts {
    takes: [AtomicU64; 4],
    epochs: AtomicU64,
    last_ep: AtomicU32,
    pages: AtomicU64,
    written: AtomicU64,
}

const ZERO: Counts = Counts { takes: [const { AtomicU64::new(0) }; 4], epochs: AtomicU64::new(0), last_ep: AtomicU32::new(0), pages: AtomicU64::new(0), written: AtomicU64::new(0) };
static COUNTS: [Counts; SLOTS] = [ZERO; SLOTS];

/// The window's mappings in the span: (start, end, kind, protection, name),
/// sorted; slot = index (SLOTS - 1: none of them or past the table).
struct Window {
    maps: Vec<(u64, u64, crate::mem::VmaKind, u32, String)>,
    /// The console's memory type of each (`Vma::mtype`).
    mtypes: Vec<i32>,
    flip0: u64,
}

static WINDOW: RwLock<Option<Window>> = RwLock::new(None);
/// Pages reported written: (the last flip + 1, flips written).
static PAGES: Mutex<Option<HashMap<u64, (u32, u32)>>> = Mutex::new(None);
/// View raises by queue.
static RAISES: Mutex<Option<HashMap<String, u64>>> = Mutex::new(None);
/// Takes outside every mapping of the window, and takes reaching past the
/// end of their mapping.
static ODD: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// Starts a window (`profile reset`, and the first use).
pub fn reset() {
    if !on() {
        return;
    }
    let (maps, mtypes): (Vec<_>, Vec<i32>) = {
        let st = crate::mem::MEM.lock();
        st.vmas.values().filter(|v| v.start < MIRROR_END && v.end() > MIRROR_BASE && v.kind != crate::mem::VmaKind::Reserved).map(|v| ((v.start, v.end(), v.kind, v.prot, v.name.clone()), v.mtype)).unzip()
    };
    let mut w = WINDOW.write();
    for c in &COUNTS {
        for t in &c.takes {
            t.store(0, Ordering::Relaxed);
        }
        c.epochs.store(0, Ordering::Relaxed);
        c.last_ep.store(0, Ordering::Relaxed);
        c.pages.store(0, Ordering::Relaxed);
        c.written.store(0, Ordering::Relaxed);
    }
    for o in &ODD {
        o.store(0, Ordering::Relaxed);
    }
    *PAGES.lock() = Some(HashMap::new());
    *RAISES.lock() = Some(HashMap::new());
    *w = Some(Window { maps, mtypes, flip0: super::FRAMES.load(Ordering::Relaxed) });
}

fn slot_of(w: &Window, a: u64) -> Option<usize> {
    let k = w.maps.partition_point(|m| m.0 <= a);
    (k > 0 && a < w.maps[k - 1].1).then(|| (k - 1).min(SLOTS - 1))
}

/// A take of [ta, tb) by `caller` reported `written` (`fe2::take_run`).
pub fn note_take(ta: u64, tb: u64, written: &[u64], caller: usize) {
    if WINDOW.read().is_none() {
        reset();
    }
    let g = WINDOW.read();
    let Some(w) = g.as_ref() else { return };
    let ep = super::precache::epoch() as u32 + 1;
    let Some(s) = slot_of(w, ta) else {
        ODD[0].fetch_add(1, Ordering::Relaxed);
        return;
    };
    if tb > w.maps[s].1 {
        ODD[1].fetch_add(1, Ordering::Relaxed);
    }
    let c = &COUNTS[s];
    c.takes[caller.min(3)].fetch_add(1, Ordering::Relaxed);
    if c.last_ep.swap(ep, Ordering::Relaxed) != ep {
        c.epochs.fetch_add(1, Ordering::Relaxed);
    }
    c.pages.fetch_add((tb - ta) / PAGE, Ordering::Relaxed);
    if written.is_empty() {
        return;
    }
    c.written.fetch_add(written.len() as u64, Ordering::Relaxed);
    let flip = super::FRAMES.load(Ordering::Relaxed) as u32 + 1;
    let mut m = PAGES.lock();
    let m = m.get_or_insert_with(HashMap::new);
    for &p in written {
        let e = m.entry(p).or_insert((0, 0));
        if e.0 != flip {
            *e = (flip, e.1 + 1);
        }
    }
}

/// Queue `name`'s own synchronization raised its view (`Cp::fe2_sync`).
pub fn note_raise(name: &str) {
    if let Some(m) = RAISES.lock().as_mut() {
        *m.entry(name.to_string()).or_default() += 1;
    }
}

/// The window's census (`profile report`; "" when off).
pub fn report() -> String {
    if !on() {
        return String::new();
    }
    let g = WINDOW.read();
    let Some(w) = g.as_ref() else { return String::new() };
    let flips = (super::FRAMES.load(Ordering::Relaxed) - w.flip0).max(1);
    let f = flips as f64;
    // Written pages by mapping: distinct, at least half the flips, a tenth.
    let mut wr = vec![[0u64; 3]; w.maps.len().min(SLOTS)];
    if let Some(m) = PAGES.lock().as_ref() {
        for (&p, &(_, n)) in m {
            if let Some(s) = slot_of(w, p) {
                wr[s][0] += 1;
                if n as u64 * 2 >= flips {
                    wr[s][1] += 1;
                }
                if n as u64 * 10 >= flips {
                    wr[s][2] += 1;
                }
            }
        }
    }
    struct Row {
        s: usize,
        takes: [u64; 4],
        epochs: u64,
        pages: u64,
        written: u64,
        gpu: u64,
    }
    let mut rows: Vec<Row> = (0..w.maps.len().min(SLOTS))
        .map(|s| {
            let c = &COUNTS[s];
            let (a, e) = (w.maps[s].0.max(MIRROR_BASE), w.maps[s].1.min(MIRROR_END));
            Row {
                s,
                takes: std::array::from_fn(|k| c.takes[k].load(Ordering::Relaxed)),
                epochs: c.epochs.load(Ordering::Relaxed),
                pages: c.pages.load(Ordering::Relaxed),
                written: c.written.load(Ordering::Relaxed),
                gpu: super::fe2::gpu_pages(a, e),
            }
        })
        .collect();
    let all: u64 = rows.iter().map(|r| r.takes.iter().sum::<u64>()).sum();
    let mut out = String::new();
    let raises: Vec<String> = RAISES.lock().as_ref().map_or(Vec::new(), |m| {
        let mut v: Vec<_> = m.iter().collect();
        v.sort_by(|a, b| b.1.cmp(a.1));
        v.iter().map(|(n, k)| format!("{n} {:.1}", **k as f64 / f)).collect()
    });
    out.push_str(&format!(
        "    fe2 census over {flips} flips: {:.0} takes a flip in {} of the span's {} mappings ({:.1} outside every mapping, {:.1} past their mapping's end); view raises a flip by queue: {}\n",
        all as f64 / f,
        rows.iter().filter(|r| r.takes.iter().sum::<u64>() > 0).count(),
        w.maps.len(),
        ODD[0].load(Ordering::Relaxed) as f64 / f,
        ODD[1].load(Ordering::Relaxed) as f64 / f,
        raises.join(", ")
    ));
    // By GPU write permission, then by kind of mapping.
    let mut by = |name: &str, keep: &dyn Fn(&(u64, u64, crate::mem::VmaKind, u32, String)) -> bool| {
        let mut t = [0u64; 9];
        for r in &rows {
            if !keep(&w.maps[r.s]) {
                continue;
            }
            t[0] += 1;
            t[1] += w.maps[r.s].1 - w.maps[r.s].0;
            t[2] += r.takes.iter().sum::<u64>();
            t[3] += r.epochs;
            t[4] += r.written;
            t[5] += wr[r.s][0];
            t[6] += wr[r.s][1];
            t[7] += wr[r.s][2];
            t[8] += r.gpu;
        }
        out.push_str(&format!(
            "    fe2 census, {name}: {} mappings of {} MiB; {:.0} takes a flip in {:.0} mapping-epochs a flip; {:.0} pages reported written a flip, {} distinct ({} in half the flips or more, {} in a tenth or more); {} pages GPU work wrote\n",
            t[0],
            t[1] >> 20,
            t[2] as f64 / f,
            t[3] as f64 / f,
            t[4] as f64 / f,
            t[5],
            t[6],
            t[7],
            t[8]
        ));
    };
    by("GPU-writable mappings", &|m| m.3 & crate::mem::PROT_GPU_WRITE != 0);
    by("other mappings", &|m| m.3 & crate::mem::PROT_GPU_WRITE == 0);
    // By the console's memory type: where GPU-written pages lie and whether
    // the CPU writes the same mappings (a write-back left out for memory
    // the CPU reads slowly on the console, if no CPU writes meet it).
    let mut types: std::collections::BTreeMap<i32, [u64; 7]> = std::collections::BTreeMap::new();
    for r in &rows {
        let t = types.entry(w.mtypes.get(r.s).copied().unwrap_or(-1)).or_default();
        t[0] += 1;
        t[1] += w.maps[r.s].1 - w.maps[r.s].0;
        t[2] += r.gpu;
        t[3] += wr[r.s][0];
        if r.gpu > 0 {
            t[4] += 1;
            if wr[r.s][0] > 0 {
                t[5] += 1;
                t[6] += r.gpu;
            }
        }
    }
    for (mt, t) in &types {
        out.push_str(&format!(
            "    fe2 census, memory type {mt}: {} mappings of {} MiB; {} pages GPU work wrote, {} distinct pages the CPU wrote; {} mappings with GPU-written pages, {} of them also written by the CPU ({} of their GPU-written pages)\n",
            t[0],
            t[1] >> 20,
            t[2],
            t[3],
            t[4],
            t[5],
            t[6]
        ));
    }
    // The mappings with the most takes, then the most pages written.
    let line = |r: &Row, wr: &[u64; 3]| {
        let m = &w.maps[r.s];
        let prot = format!(
            "{}{}{}/{}{}",
            if m.3 & crate::mem::PROT_CPU_READ != 0 { "r" } else { "-" },
            if m.3 & crate::mem::PROT_CPU_WRITE != 0 { "w" } else { "-" },
            if m.3 & crate::mem::PROT_CPU_EXEC != 0 { "x" } else { "-" },
            if m.3 & crate::mem::PROT_GPU_READ != 0 { "r" } else { "-" },
            if m.3 & crate::mem::PROT_GPU_WRITE != 0 { "w" } else { "-" }
        );
        format!(
            "      {:#x}+{} MiB {:?} cpu/gpu {prot} '{}': takes {:.1} a flip (graphics CP {:.1}, workers {:.1}, queue 0x50 {:.1}, other {:.1}) in {:.1} epochs a flip, {:.0} pages each; written {:.1} pages a flip, {} distinct ({} in half the flips, {} in a tenth); GPU-written pages {}\n",
            m.0,
            (m.1 - m.0) >> 20,
            m.2,
            m.4,
            r.takes.iter().sum::<u64>() as f64 / f,
            r.takes[0] as f64 / f,
            r.takes[1] as f64 / f,
            r.takes[2] as f64 / f,
            r.takes[3] as f64 / f,
            r.epochs as f64 / f,
            r.pages as f64 / r.takes.iter().sum::<u64>().max(1) as f64,
            r.written as f64 / f,
            wr[0],
            wr[1],
            wr[2],
            r.gpu
        )
    };
    rows.sort_by_key(|r| std::cmp::Reverse(r.takes.iter().sum::<u64>()));
    out.push_str("    fe2 census, mappings by takes:\n");
    for r in rows.iter().take(24).filter(|r| r.takes.iter().sum::<u64>() > 0) {
        out.push_str(&line(r, &wr[r.s]));
    }
    rows.sort_by_key(|r| std::cmp::Reverse(r.written));
    out.push_str("    fe2 census, mappings by pages written:\n");
    for r in rows.iter().take(12).filter(|r| r.written > 0) {
        out.push_str(&line(r, &wr[r.s]));
    }
    rows.sort_by_key(|r| std::cmp::Reverse(r.gpu));
    out.push_str("    fe2 census, mappings by GPU-written pages:\n");
    for r in rows.iter().take(12).filter(|r| r.gpu > 0) {
        out.push_str(&line(r, &wr[r.s]));
    }
    out
}
