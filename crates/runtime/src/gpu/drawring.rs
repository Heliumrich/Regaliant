//! Diagnosis (`DS_DRAW_RING=1`, 2026-11-18 night): the indexed indirect draws of the last frames
//! as the command processors recorded them (the flip they were queued for, the recorder, the
//! programs, the arguments' address and words, the index buffer), in a bounded ring. When the
//! frame watch sees a flash of 20 cells or more, the ring is written to
//! `<run dir>/drawring-<ms>.txt` (at most 8 dumps): the flash frame's draws against its
//! neighbours'. Built for the player's Nexus clip (huge triangles for one frame, only with the
//! graphics queue's parallel recording and the title's index culling).

use parking_lot::Mutex;
use std::collections::VecDeque;
use std::io::Write;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU32, Ordering::Relaxed};

struct Entry {
    flip: u64,
    ms: f64,
    thread: &'static str,
    vs: u64,
    ps: u64,
    args: u64,
    w: [u32; 5],
    index_base: u64,
    index_count: u32,
}

/// About 30 frames of the Nexus's heaviest views.
const CAP: usize = 120_000;
static RING: Mutex<VecDeque<Entry>> = Mutex::new(VecDeque::new());
static DUMPS: AtomicU32 = AtomicU32::new(0);

pub fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_DRAW_RING").ok().as_deref() == Some("1"))
}

thread_local! {
    static NAME: &'static str = Box::leak(std::thread::current().name().unwrap_or("?").to_string().into_boxed_str());
}

/// An indexed indirect draw read its arguments `w` at `args` (`Cp`'s DRAW_INDEX_INDIRECT).
pub fn note(vs: u64, ps: u64, args: u64, w: [u32; 5], index_base: u64, index_count: u32) {
    let flip = crate::hle::libs::videoout::QUEUED_FLIP.load(Relaxed) + 1;
    let thread = NAME.with(|n| *n);
    let e = Entry { flip, ms: crate::log::elapsed_ms(), thread, vs, ps, args, w, index_base, index_count };
    let mut g = RING.lock();
    if g.len() >= CAP {
        g.pop_front();
    }
    g.push_back(e);
}

/// The frame watch saw a flash at `at` ms: the ring to a file.
pub fn dump(at: f64) {
    if !on() || DUMPS.fetch_add(1, Relaxed) >= 8 {
        return;
    }
    let Some(dir) = crate::run_dir() else { return };
    let path = dir.join(format!("drawring-{at:.0}.txt"));
    let Ok(f) = std::fs::File::create(&path) else { return };
    let mut f = std::io::BufWriter::new(f);
    let g = RING.lock();
    let _ = writeln!(f, "# flash at {at:.0} ms; {} indexed indirect draws: flip ms thread vs ps args count instances first_index base_vertex first_instance index_base index_count", g.len());
    for e in g.iter() {
        let _ = writeln!(
            f,
            "{} {:.1} {} {:#x} {:#x} {:#x} {} {} {} {} {} {:#x} {}",
            e.flip, e.ms, e.thread, e.vs, e.ps, e.args, e.w[0], e.w[1], e.w[2], e.w[3] as i32, e.w[4], e.index_base, e.index_count
        );
    }
    crate::info!("draw ring: {} draws written to {}", g.len(), path.display());
}
