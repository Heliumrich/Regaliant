//! Diagnosis (`DS_RUN_PROBE=1`): the ceiling of the graphics command
//! processor recording past its own parallel runs (STATUS.md, next action).
//!
//! At the end of each parallel run the graphics command processor waits for
//! the run's chunks (`Cp::par_collect`). Were it to go on recording instead,
//! its batches after the run would still have to follow the chunks' batches:
//! it would wait for them at its next submission, flush or deferred CP effect
//! (`stop`), and before reading on the CPU what the run's chunks wrote
//! (`reads`: their batches' writes against the reads it checks for pending
//! GPU writes). So per run the wait it could hide is the smaller of its
//! wait and the time from the run's end to that next stop. Per frame (flip
//! to flip): the waits and that bound; by the kind of stop: how often it
//! came and how far after the run's end.

use parking_lot::Mutex;
use std::collections::HashMap;
use std::time::Instant;

pub fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_RUN_PROBE").ok().as_deref() == Some("1"))
}

#[derive(Default)]
struct State {
    /// The last run's wait for its chunks (ns), when it ended and what its
    /// chunks wrote.
    pending: Option<(u64, Instant, super::ranges::Ranges)>,
    /// This frame's waits, hideable part (ns) and runs.
    wait: u64,
    hidden: u64,
    runs: u32,
    /// Per frame: (ms waited, ms hideable, runs).
    frames: Vec<(f32, f32, u32)>,
    /// By what ended a run's window: [runs, ns of window, ns hideable].
    by_why: HashMap<&'static str, [u64; 3]>,
}

static ST: Mutex<Option<State>> = Mutex::new(None);

/// The graphics command processor collected a run's last chunk, having
/// waited `wait_ns` for the run's chunks.
pub fn run_end(wait_ns: u64, writes: super::ranges::Ranges) {
    let mut g = ST.lock();
    let s = g.get_or_insert_with(State::default);
    if s.pending.is_some() {
        close(s, "next run");
    }
    s.pending = Some((wait_ns, Instant::now(), writes));
}

/// The graphics command processor checks `reads` for pending GPU writes
/// (it reads them on the CPU now).
pub fn reads(reads: &[(u64, u64)]) {
    let mut g = ST.lock();
    if let Some(s) = g.as_mut() {
        if s.pending.as_ref().is_some_and(|p| p.2.overlaps_any(reads)) {
            close(s, "reads the run's writes");
        }
    }
}

/// The graphics command processor submits (`why`), flushes or defers a CP
/// effect: work after a run would wait for its chunks here.
pub fn stop(why: &'static str) {
    let mut g = ST.lock();
    if let Some(s) = g.as_mut() {
        if s.pending.is_some() {
            close(s, why);
        }
    }
}

fn close(s: &mut State, why: &'static str) {
    let Some((wait, t, _)) = s.pending.take() else { return };
    let window = t.elapsed().as_nanos() as u64;
    let hidden = wait.min(window);
    s.wait += wait;
    s.hidden += hidden;
    s.runs += 1;
    let e = s.by_why.entry(why).or_default();
    e[0] += 1;
    e[1] += window;
    e[2] += hidden;
}

/// The graphics command processor reached a flip.
pub fn flip() {
    let mut g = ST.lock();
    let s = g.get_or_insert_with(State::default);
    if s.frames.len() < 1 << 16 {
        s.frames.push((s.wait as f32 / 1e6, s.hidden as f32 / 1e6, s.runs));
    }
    s.wait = 0;
    s.hidden = 0;
    s.runs = 0;
}

pub fn reset() {
    if let Some(s) = ST.lock().as_mut() {
        s.frames.clear();
        s.by_why.clear();
    }
}

pub fn report() -> String {
    let g = ST.lock();
    let Some(s) = g.as_ref() else { return String::new() };
    if s.frames.is_empty() {
        return String::new();
    }
    let pct = |mut v: Vec<f32>, q: f32| -> f32 {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[((v.len() - 1) as f32 * q) as usize]
    };
    let wait: Vec<f32> = s.frames.iter().map(|f| f.0).collect();
    let hidden: Vec<f32> = s.frames.iter().map(|f| f.1).collect();
    let runs: Vec<f32> = s.frames.iter().map(|f| f.2 as f32).collect();
    let mut why: Vec<(&&str, &[u64; 3])> = s.by_why.iter().collect();
    why.sort_by(|a, b| b.1[2].cmp(&a.1[2]));
    let why: Vec<String> = why.iter().map(|(w, x)| format!("{w} {} (window {:.0} us, hideable {:.0} us each)", x[0], x[1] as f64 / 1e3 / x[0].max(1) as f64, x[2] as f64 / 1e3 / x[0].max(1) as f64)).collect();
    format!(
        "    run probe ({} frames): the graphics CP's waits for its parallel runs' chunks per frame p10 {:.2}, p50 {:.2}, p90 {:.2} ms over p50 {:.0} runs; hideable behind its own recording up to its next submission, flush or deferred CP effect p10 {:.2}, p50 {:.2}, p90 {:.2} ms; runs' windows ended by: {}\n",
        s.frames.len(),
        pct(wait.clone(), 0.1),
        pct(wait.clone(), 0.5),
        pct(wait, 0.9),
        pct(runs, 0.5),
        pct(hidden.clone(), 0.1),
        pct(hidden.clone(), 0.5),
        pct(hidden, 0.9),
        why.join(", ")
    )
}
