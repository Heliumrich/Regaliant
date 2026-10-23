//! Diagnosis (`DS_FRAME_PROBE=1`): the ceiling of recording the next frame's
//! graphics stream ahead, on a second recorder whose batches wait until the
//! current frame's are submitted (STATUS.md, next action).
//!
//! Per frame of the graphics command processor (flip to flip): when it
//! first flushes and waits for its own work (where such a recorder would
//! have to stop: it cannot submit), and the registers whose value at the
//! frame's start differs from the previous frame's start (what such a
//! recorder, started from the previous frame's starting state, would have
//! wrong) that the frame has not set yet when its first draw or dispatch
//! comes ("risky": used with the wrong value unless that command ignores
//! them).

use parking_lot::Mutex;
use std::time::{Duration, Instant};

pub fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_FRAME_PROBE").ok().as_deref() == Some("1"))
}

#[derive(Default)]
struct State {
    start: Option<Instant>,
    first_flush: Option<(Duration, &'static str)>,
    /// The register state at the previous flip: [ctx, sh, uc].
    prev: Option<[Vec<u32>; 3]>,
    /// Registers differing from the frame before at this frame's start, and
    /// whether the frame has set them since.
    diff: Vec<(usize, u32, bool)>,
    commands: u64,
    /// The frame's first command came: risky registers counted.
    risky: Option<usize>,
    risky_regs: std::collections::HashMap<(usize, u32), u64>,
    /// Per frame: (ms to the first own flush or None, frame ms, registers
    /// differing, risky, commands before the first flush).
    frames: Vec<(Option<f32>, f32, usize, usize, u64)>,
    whys: std::collections::HashMap<&'static str, u64>,
}

static ST: Mutex<Option<State>> = Mutex::new(None);

/// The graphics command processor reached a flip with registers `regs`.
pub fn flip(regs: [&[u32]; 3]) {
    let mut g = ST.lock();
    let s = g.get_or_insert_with(State::default);
    let now = Instant::now();
    if let Some(t) = s.start {
        let first = s.first_flush.map(|(d, _)| d.as_secs_f32() * 1000.0);
        let risky = s.risky.unwrap_or(s.diff.iter().filter(|d| !d.2).count());
        let before = s.commands;
        if s.frames.len() < 1 << 16 {
            s.frames.push((first, (now - t).as_secs_f32() * 1000.0, s.diff.len(), risky, before));
        }
    }
    let cur: [Vec<u32>; 3] = [regs[0].to_vec(), regs[1].to_vec(), regs[2].to_vec()];
    s.diff.clear();
    if let Some(p) = &s.prev {
        for k in 0..3 {
            for (i, (a, b)) in cur[k].iter().zip(&p[k]).enumerate() {
                if a != b {
                    s.diff.push((k, i as u32, false));
                }
            }
        }
    }
    s.prev = Some(cur);
    s.start = Some(now);
    s.first_flush = None;
    s.commands = 0;
    s.risky = None;
}

/// The graphics command processor set a register.
pub fn reg_written(space: usize, index: u32) {
    let mut g = ST.lock();
    let Some(s) = g.as_mut() else { return };
    if s.risky.is_none() {
        for d in s.diff.iter_mut() {
            if d.0 == space && d.1 == index {
                d.2 = true;
            }
        }
    }
}

/// A draw or dispatch of the graphics command processor.
pub fn command() {
    let mut g = ST.lock();
    let Some(s) = g.as_mut() else { return };
    if s.first_flush.is_none() {
        s.commands += 1;
    }
    if s.risky.is_none() && s.start.is_some() {
        let risky: Vec<(usize, u32)> = s.diff.iter().filter(|d| !d.2).map(|d| (d.0, d.1)).collect();
        for r in &risky {
            *s.risky_regs.entry(*r).or_default() += 1;
        }
        s.risky = Some(risky.len());
    }
}

/// The graphics command processor flushed and waited for its own work.
pub fn flushed(why: &'static str) {
    let mut g = ST.lock();
    let Some(s) = g.as_mut() else { return };
    if s.first_flush.is_none() {
        if let Some(t) = s.start {
            let why = if why.is_empty() { "flush" } else { why };
            s.first_flush = Some((t.elapsed(), why));
            *s.whys.entry(why).or_default() += 1;
        }
    }
}

pub fn report() -> String {
    let g = ST.lock();
    let Some(s) = g.as_ref() else { return String::new() };
    if s.frames.is_empty() {
        return String::new();
    }
    let pct = |mut v: Vec<f32>, q: f32| -> f32 {
        if v.is_empty() {
            return f32::NAN;
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[((v.len() - 1) as f32 * q) as usize]
    };
    let n = s.frames.len();
    let flushed: Vec<f32> = s.frames.iter().filter_map(|f| f.0).collect();
    let share: Vec<f32> = s.frames.iter().map(|f| f.0.map_or(1.0, |x| x / f.1)).collect();
    let frame: Vec<f32> = s.frames.iter().map(|f| f.1).collect();
    let diff: Vec<f32> = s.frames.iter().map(|f| f.2 as f32).collect();
    let risky: Vec<f32> = s.frames.iter().map(|f| f.3 as f32).collect();
    let cmds: Vec<f32> = s.frames.iter().map(|f| f.4 as f32).collect();
    let mut whys: Vec<(&&str, &u64)> = s.whys.iter().collect();
    whys.sort_by(|a, b| b.1.cmp(a.1));
    let mut regs: Vec<(&(usize, u32), &u64)> = s.risky_regs.iter().collect();
    regs.sort_by(|a, b| b.1.cmp(a.1));
    let regs: Vec<String> = regs.iter().take(12).map(|((k, i), c)| format!("{}:{i:#x} {c}", ["ctx", "sh", "uc"][*k])).collect();
    format!(
        "    frame probe ({n} frames): first own flush after p10 {:.1}, p50 {:.1}, p90 {:.1} ms ({} frames without one) of frames p50 {:.1} ms; share of the frame before it p10 {:.2}, p50 {:.2}, p90 {:.2}; commands before it p50 {:.0}; why: {}\n      registers differing from the frame before at the start p50 {:.0}, p90 {:.0}; not yet set at the first command p50 {:.0}, p90 {:.0}, max {:.0}; most often: {}\n",
        pct(flushed.clone(), 0.1),
        pct(flushed.clone(), 0.5),
        pct(flushed.clone(), 0.9),
        n - flushed.len(),
        pct(frame, 0.5),
        pct(share.clone(), 0.1),
        pct(share.clone(), 0.5),
        pct(share, 0.9),
        pct(cmds, 0.5),
        whys.iter().map(|(w, c)| format!("{w} {c}")).collect::<Vec<_>>().join(", "),
        pct(diff.clone(), 0.5),
        pct(diff, 0.9),
        pct(risky.clone(), 0.5),
        pct(risky.clone(), 0.9),
        pct(risky, 1.0),
        regs.join(", ")
    )
}

pub fn reset() {
    let mut g = ST.lock();
    if let Some(s) = g.as_mut() {
        s.frames.clear();
        s.whys.clear();
        s.risky_regs.clear();
    }
}
