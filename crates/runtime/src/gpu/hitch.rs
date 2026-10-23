//! Hitches: presented frames far longer than the frames before them, logged
//! with the long pieces of work that overlapped them (pipeline creation,
//! shader translation, texture uploads, CPU waits for the GPU, long command
//! processor waits, file I/O, single long recordings), so a frame-time trace
//! says why a frame was long and not only that it was: the frame-rate goal is
//! judged by the worst frames (STATUS.md). Every run's log gets `hitch:`
//! lines; the window statistics (`frametimes`, the scripts' `dsport-frames:`
//! marks) count the hitches and their excess time by cause.

use parking_lot::Mutex;
use std::collections::VecDeque;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    /// Pipeline creation (the driver's compile).
    Pipeline = 0,
    /// A shader's translation to SPIR-V.
    Translate,
    /// Texture uploads (detiling on the CPU, staging copies).
    Upload,
    /// A CPU thread waiting for GPU work to complete.
    GpuWait,
    /// The graphics command processor waiting for a label (the title's CPU
    /// or another queue); the compute queues' waits for the graphics queue
    /// are the frame's structure, not a cause, and not noted.
    CpWait,
    /// File reads and writes of the title's threads.
    File,
    /// One command's recording, beyond the parts above.
    Record,
}

const KINDS: usize = 7;
const NAMES: [&str; KINDS + 1] = ["pipeline", "translation", "upload", "GPU wait", "CP wait", "file", "recording", "unexplained"];

/// Work shorter than this is not kept (us).
const MIN_US: u64 = 2_000;
/// Recent work kept for the next reports.
const KEEP: usize = 1024;

struct Event {
    start: u64,
    end: u64,
    kind: Kind,
    thread: String,
    what: String,
}

struct State {
    events: VecDeque<Event>,
    /// The last presented intervals (us), for the median.
    recent: VecDeque<u32>,
    last_flip: u64,
    /// The window's hitches, their excess over the median (us) and that
    /// excess by the cause that overlapped it most.
    hitches: u64,
    excess: u64,
    by_kind: [u64; KINDS + 1],
    worst: Vec<(u64, u64)>,
}

static STATE: Mutex<State> = Mutex::new(State {
    events: VecDeque::new(),
    recent: VecDeque::new(),
    last_flip: 0,
    hitches: 0,
    excess: 0,
    by_kind: [0; KINDS + 1],
    worst: Vec::new(),
});

/// The clock the presented flips are timed with (us since start).
pub fn now_us() -> u64 {
    crate::hle::kernel::time::process_time_us()
}

/// Notes work of `kind` from `start` to `end` (us, `now_us`) on this thread
/// for the hitch reports (kept if at least 2 ms long; `what` is called only
/// then).
pub fn note(kind: Kind, start: u64, end: u64, what: impl FnOnce() -> String) {
    if end.saturating_sub(start) < MIN_US {
        return;
    }
    let e = Event { start, end, kind, thread: crate::log::thread_name(), what: what() };
    let mut s = STATE.lock();
    if s.events.len() >= KEEP {
        s.events.pop_front();
    }
    s.events.push_back(e);
}

/// `note` for work that started at `t0` and ends now.
pub fn note_since(kind: Kind, t0: std::time::Instant, what: impl FnOnce() -> String) {
    let us = t0.elapsed().as_micros() as u64;
    if us < MIN_US {
        return;
    }
    let end = now_us();
    note(kind, end.saturating_sub(us), end, what);
}

/// A presented flip (`pacing::note_flip`): flip `n` shown at `us`. A frame
/// half again as long as the median of the last 60 and at least a vblank
/// over it is a hitch: logged with the work that overlapped it (from a
/// median before its start, as work just before a frame delays it too).
pub fn flip(n: u64, us: u64) {
    let mut s = STATE.lock();
    let prev = s.last_flip;
    s.last_flip = us;
    if prev == 0 || us <= prev {
        return;
    }
    let d = us - prev;
    let median = {
        let mut v: Vec<u32> = s.recent.iter().copied().collect();
        v.sort_unstable();
        v.get(v.len() / 2).copied().unwrap_or(0) as u64
    };
    if s.recent.len() >= 60 {
        s.recent.pop_front();
    }
    s.recent.push_back(d.min(u32::MAX as u64) as u32);
    if s.recent.len() < 9 || d * 2 < median * 3 || d < median + 16_667 {
        return;
    }
    let lo = prev.saturating_sub(median);
    // Per kind the time any of its work ran (concurrent work on several
    // threads counted once), clipped to the frame and the median before it.
    let mut spans: [Vec<(u64, u64)>; KINDS] = Default::default();
    let mut over: Vec<(u64, &Event)> = Vec::new();
    for e in s.events.iter().filter(|e| e.end > lo && e.start < us) {
        let (a, b) = (e.start.max(lo), e.end.min(us));
        spans[e.kind as usize].push((a, b));
        over.push((b - a, e));
    }
    let mut sum = [0u64; KINDS];
    for (k, v) in spans.iter_mut().enumerate() {
        v.sort_unstable();
        let mut end = 0u64;
        for &(a, b) in v.iter() {
            let a = a.max(end);
            if b > a {
                sum[k] += b - a;
                end = b;
            }
        }
    }
    over.sort_by(|a, b| b.0.cmp(&a.0));
    let excess = d - median;
    let top = (0..KINDS).max_by_key(|&k| sum[k]).filter(|&k| sum[k] * 4 >= excess).unwrap_or(KINDS);
    let kinds: Vec<String> = (0..KINDS).filter(|&k| sum[k] > 0).map(|k| format!("{} {:.0} ms", NAMES[k], sum[k] as f64 / 1e3)).collect();
    let longest: Vec<String> = over
        .iter()
        .take(4)
        .map(|(o, e)| format!("{} {:.0} ms on {} ({})", NAMES[e.kind as usize], *o as f64 / 1e3, e.thread, e.what))
        .collect();
    crate::info!(
        "hitch: flip {n} took {:.0} ms (median {:.0}): {}; {}",
        d as f64 / 1e3,
        median as f64 / 1e3,
        if kinds.is_empty() { "no long work noted meanwhile".to_string() } else { kinds.join(", ") },
        if longest.is_empty() { String::new() } else { format!("longest: {}", longest.join("; ")) }
    );
    s.hitches += 1;
    s.excess += excess;
    s.by_kind[top] += excess;
    s.worst.push((d, n));
}

/// The window's hitches (one line), and a new window.
pub fn window_report_and_reset() -> String {
    let mut s = STATE.lock();
    let line = if s.hitches == 0 {
        "no hitches".to_string()
    } else {
        let by: Vec<String> = (0..=KINDS).filter(|&k| s.by_kind[k] > 0).map(|k| format!("{} {:.0}", NAMES[k], s.by_kind[k] as f64 / 1e3)).collect();
        s.worst.sort_by(|a, b| b.0.cmp(&a.0));
        let w: Vec<String> = s.worst.iter().take(3).map(|&(d, f)| format!("{:.0} ms (flip {f})", d as f64 / 1e3)).collect();
        format!("{} hitches, {:.0} ms over the median (by cause, ms: {}); longest {}", s.hitches, s.excess as f64 / 1e3, by.join(", "), w.join(", "))
    };
    s.hitches = 0;
    s.excess = 0;
    s.by_kind = [0; KINDS + 1];
    s.worst.clear();
    line
}
