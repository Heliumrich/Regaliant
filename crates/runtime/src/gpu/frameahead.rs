//! Frame-ahead recording (RENDERER.md, `tune frame_ahead`): a second
//! graphics command processor (`gpu-gfx-ahead`) records the next frame
//! while the graphics command processor finishes the current one.
//!
//! The graphics command processor posts its state at the start of each
//! title submission (`post`). The recorder walks the packets from there
//! through the submissions queued behind it to the current frame's flip
//! (`Cp::walk_state`: the next frame's start state, exact in 4,159 of 4,161
//! frames, FINDINGS.md 2026-10-23) and records the packets after the flip
//! with an executor that holds its batches (`Exec::set_held`), until it
//! stops: before anything it cannot do ahead of the current frame's work
//! (`Cp::held_refuses`, CPU-side CP effects, image-cache notes, fast clears,
//! storage images, its own unsubmitted writes read, a label wait that does
//! not pass), at the end of what is queued, or once the graphics command
//! processor reached the flip (`flipped`).
//!
//! Stage 1 (this): what it recorded is dropped after the flip; counters
//! say how far it got and whether a hand-over would have held: the guest
//! ranges it read on the CPU against what the graphics command processor
//! wrote meanwhile (`a_wrote`), and its start state against the real one
//! at the flip.

use super::cp::{Cp, WalkState};
use super::ranges::Ranges;
use parking_lot::{Condvar, Mutex};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

pub fn on() -> bool {
    super::tune::frame_ahead() != 0
}

/// The recorder's command-processor state while it records a frame ahead
/// (`Cp::held`).
pub struct HeldCp {
    /// Flips (`gpu::FRAMES`) before the current frame's: that frame is done
    /// once the count moves past this.
    pub frames: u64,
    /// What the current frame's rest writes last where values are known
    /// before execution (address -> value, bytes; 0 bytes: a timestamp):
    /// a wait on it sees that value (that work comes first).
    pub labels: HashMap<u64, (u64, u8)>,
    /// Why and where (packet address) it stopped.
    pub stop: Option<(&'static str, u64)>,
    /// Draws and dispatches recorded.
    pub commands: u64,
    /// Label waits passed: [on its own writes, on the current frame's,
    /// on memory, on another queue's promise, after polling].
    pub waits: [u64; 5],
    /// A wait that does not pass at once is polled while the frame lasts
    /// (false: it stops there; the graphics command processor recording
    /// past a run, `Cp::past_start`).
    pub poll: bool,
}

/// The graphics command processor's state at the start of a submission.
#[derive(Clone)]
struct Post {
    state: Arc<WalkState>,
    index: u64,
    bufs: Vec<(u64, u32)>,
    /// Flips before it.
    frames: u64,
    /// When it was queued.
    at: Instant,
}

struct Shared {
    post: Option<Post>,
}

static SH: Mutex<Shared> = Mutex::new(Shared { post: None });
/// Wakes the recorder: a post, a submission, a flip.
static CV: Condvar = Condvar::new();
static STARTED: std::sync::Once = std::sync::Once::new();

/// The graphics command processor's writes are collected (from when the
/// recorder begins a frame to that frame's flip).
static COLLECTING: AtomicBool = AtomicBool::new(false);
static A_WRITES: Mutex<Ranges> = Mutex::new(Ranges::new());
/// The flips the graphics command processor has finished its part of
/// (`flipped`), and its state at the last one while collecting.
static FLIPPED: AtomicU64 = AtomicU64::new(0);
/// When the graphics command processor reached its last flip.
static FLIP_AT: Mutex<Option<Instant>> = Mutex::new(None);
/// Where the recorder stopped in the next frame (packet address; 0: none)
/// and the flip count the graphics command processor reaches it with
/// (`a_reached`: when it got there itself, the time a hand-over saves).
pub static TARGET: AtomicU64 = AtomicU64::new(0);
static TARGET_FRAMES: AtomicU64 = AtomicU64::new(0);
/// [times the graphics command processor reached the recorder's stop, ns
/// from its flip]
static REACHED: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
static REACH_MS: Mutex<Vec<f32>> = Mutex::new(Vec::new());

/// The graphics command processor is at the packet where the recorder
/// stopped (`TARGET`).
pub fn a_reached() {
    if super::FRAMES.load(Ordering::Acquire) != TARGET_FRAMES.load(Ordering::Acquire) {
        return;
    }
    TARGET.store(0, Ordering::Relaxed);
    if let Some(t) = *FLIP_AT.lock() {
        let ns = t.elapsed().as_nanos() as u64;
        REACHED[0].fetch_add(1, Ordering::Relaxed);
        REACHED[1].fetch_add(ns, Ordering::Relaxed);
        let mut v = REACH_MS.lock();
        if v.len() < 1 << 16 {
            v.push(ns as f32 / 1e6);
        }
    }
}
/// At that flip: its count, the state, the time, and the images with a
/// fast clear or an overwrite reload pending.
static AT_FLIP: Mutex<Option<(u64, WalkState, Instant, Vec<super::images::ImageKey>)>> = Mutex::new(None);

/// [frames begun, commands recorded ahead, frames not walked, frames whose
/// CPU reads the current frame's rest wrote, frames whose start state
/// differed at the flip, recording time (ns), the flip to the recorder's
/// stop when the flip stopped it (ns, count), batches held, frames given
/// up before beginning (the frame flipped first), frames that used images
/// with the current frame's fast clears or overwrites still pending at the
/// flip]
static STATS: [AtomicU64; 11] = [const { AtomicU64::new(0) }; 11];
static STOPS: Mutex<Vec<(&'static str, u64)>> = Mutex::new(Vec::new());
static WALK_ERRORS: Mutex<Vec<(&'static str, u64)>> = Mutex::new(Vec::new());
/// Label waits passed by kind (`HeldCp::waits`).
static WAITS: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];
/// Per frame begun: (commands recorded, ms recording) (bounded).
static PER_FRAME: Mutex<Vec<(u32, f32)>> = Mutex::new(Vec::new());
/// Label waits it polled, by label and the queue that wrote it last:
/// [polled, passed, ns polled, commands recorded before], the last value
/// waited for, and what the writer queue was doing when the poll began.
type PollStats = ([u64; 4], u64, HashMap<String, u64>);
static POLLS: Mutex<Option<HashMap<(u64, String), PollStats>>> = Mutex::new(None);

/// The recorder polled a wait on label `a` for `reference` (`writer`: the
/// queue that wrote it last; `state`: what it was doing then) for
/// `waited`; whether it passed; commands it had recorded before.
pub fn note_poll(a: u64, reference: u64, writer: String, waited: Duration, passed: bool, commands: u64, state: String) {
    let mut g = POLLS.lock();
    let m = g.get_or_insert_with(HashMap::new);
    if m.len() > 4096 {
        return;
    }
    let e = m.entry((a, writer)).or_insert(([0; 4], 0, HashMap::new()));
    e.0[0] += 1;
    e.0[1] += passed as u64;
    e.0[2] += waited.as_nanos() as u64;
    e.0[3] += commands;
    e.1 = reference;
    if e.2.len() < 64 || e.2.contains_key(&state) {
        *e.2.entry(state).or_default() += 1;
    }
}

fn count(list: &Mutex<Vec<(&'static str, u64)>>, why: &'static str) {
    let mut g = list.lock();
    match g.iter_mut().find(|e| e.0 == why) {
        Some(e) => e.1 += 1,
        None => g.push((why, 1)),
    }
}

/// The graphics command processor starts title submission `index` (its
/// command buffers `bufs`).
pub fn post(cp: &Cp, index: u64, bufs: &[(u64, u32)], at: Instant) {
    STARTED.call_once(|| {
        if let Err(e) = crate::thread::spawn_host("gpu-gfx-ahead", 8 << 20, Box::new(recorder)) {
            crate::warn!("gpu: no frame-ahead recorder thread ({e})");
        }
    });
    let state = Arc::new(cp.walk_start());
    let frames = super::FRAMES.load(Ordering::SeqCst);
    let mut g = SH.lock();
    g.post = Some(Post { state, index, bufs: bufs.to_vec(), frames, at });
    CV.notify_all();
}

/// A title submission was queued on the graphics queue.
pub fn submitted() {
    if on() {
        let _g = SH.lock();
        CV.notify_all();
    }
}

/// The graphics command processor reached a flip (`gpu::FRAMES` counted
/// it): its writes before it were collected.
pub fn flipped(cp: &Cp) {
    let n = super::FRAMES.load(Ordering::SeqCst);
    if COLLECTING.swap(false, Ordering::SeqCst) {
        let pending = if super::images::any_pending() { super::images::with(|im| im.pending_keys()) } else { Vec::new() };
        *AT_FLIP.lock() = Some((n, cp.walk_start(), Instant::now(), pending));
    }
    *FLIP_AT.lock() = Some(Instant::now());
    FLIPPED.store(n, Ordering::SeqCst);
    let _g = SH.lock();
    CV.notify_all();
}

/// Whether the flip after `frames` flips came (the recorder's frame is the
/// one after it).
pub fn frame_finished(frames: u64) -> bool {
    super::FRAMES.load(Ordering::Acquire) > frames
}

/// The graphics command processor (or a parallel worker of its) wrote
/// `r` (batches at their submission, CP writes).
pub fn a_wrote(r: &[(u64, u64)]) {
    if COLLECTING.load(Ordering::Acquire) {
        let mut g = A_WRITES.lock();
        for &(a, n) in r {
            g.insert(a, n);
        }
    }
}

pub fn a_wrote_ranges(r: &Ranges) {
    if COLLECTING.load(Ordering::Acquire) && !r.is_empty() {
        A_WRITES.lock().extend(r);
    }
}

/// The graphics queue's title submissions queued with index above `after`:
/// (index, buffers, when queued).
fn queued_after(after: u64) -> Vec<(u64, Vec<(u64, u32)>, Instant)> {
    let q = super::queue(super::GFX);
    let g = q.pending.lock();
    g.iter().filter(|s| s.ctl.is_none() && s.index > after).map(|s| (s.index, s.bufs.clone(), s.at)).collect()
}

/// A buffer of the next frame: (submission index, when it was queued,
/// address, dwords).
type NextBuf = (u64, Instant, u64, u32);

/// Where the next frame starts: walked from the post to the flip. The
/// walked state, the next frame's buffers queued so far and the flip's
/// submission index; None: the flip or the packets after it are not queued
/// yet; Err: the frame cannot be walked.
fn find_next(post: &Post) -> Result<Option<(WalkState, Vec<NextBuf>, u64)>, &'static str> {
    let mut subs: Vec<(u64, Vec<(u64, u32)>, Instant)> = vec![(post.index, post.bufs.clone(), post.at)];
    subs.extend(queued_after(post.index));
    let mut w = (*post.state).clone();
    for (k, (index, bufs, t)) in subs.iter().enumerate() {
        for (j, &(a, n)) in bufs.iter().enumerate() {
            match Cp::walk_state(&mut w, a, n) {
                Ok(false) => {}
                Ok(true) => {
                    let (at, len) = w.flip.ok_or("flip without its position")?;
                    let end = a + 4 * n as u64;
                    let mut next: Vec<NextBuf> = Vec::new();
                    if at + len < end {
                        next.push((*index, *t, at + len, ((end - at - len) / 4) as u32));
                    }
                    next.extend(bufs[j + 1..].iter().map(|&(b, m)| (*index, *t, b, m)));
                    for (i2, b2, t2) in &subs[k + 1..] {
                        next.extend(b2.iter().map(|&(b, m)| (*i2, *t2, b, m)));
                    }
                    if next.is_empty() {
                        return Ok(None);
                    }
                    return Ok(Some((w, next, *index)));
                }
                Err(e) => return Err(e),
            }
        }
    }
    Ok(None)
}

fn recorder() {
    crate::log::set_thread_name("gpu-gfx-ahead");
    crate::exception::arm_watch_current_thread(false);
    super::profile_this_thread("ahead");
    super::shaders::set_compute_queue(false);
    // Vulkan comes up with the first GPU work (the title's first graphics
    // submissions come before it).
    while super::vk::try_ctx().is_none() || !super::exec::enabled() {
        if super::shutting_down() {
            return;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    let mut cp = Cp::new("gpu-gfx-ahead", false);
    if let Err(e) = cp.make_held_exec() {
        crate::warn!("gpu: frame-ahead recorder without an executor ({e})");
        return;
    }
    crate::info!("gpu: frame-ahead recorder started (`tune frame_ahead`: stage 1, what it records is dropped)");
    // The last frame recorded ahead or given up (`Post::frames`).
    let mut done_frames = u64::MAX;
    loop {
        let post = {
            let mut g = SH.lock();
            CV.wait_for(&mut g, Duration::from_millis(2));
            g.post.clone()
        };
        if super::shutting_down() {
            return;
        }
        let Some(post) = post else { continue };
        if !on() || post.frames == done_frames || super::capture::capturing() || super::replay::replaying() || frame_finished(post.frames) {
            continue;
        }
        let (w, next, flip_sub) = match find_next(&post) {
            Ok(Some(x)) => x,
            Ok(None) => continue,
            Err(e) => {
                count(&WALK_ERRORS, e);
                STATS[2].fetch_add(1, Ordering::Relaxed);
                done_frames = post.frames;
                continue;
            }
        };
        // The current frame's writes from here to its flip.
        A_WRITES.lock().clear();
        COLLECTING.store(true, Ordering::SeqCst);
        if frame_finished(post.frames) {
            COLLECTING.store(false, Ordering::SeqCst);
            STATS[9].fetch_add(1, Ordering::Relaxed);
            continue;
        }
        done_frames = post.frames;
        record_frame(&mut cp, &post, w, next, flip_sub);
    }
}

/// Records the next frame from `w` (its first buffers `next`) until a stop,
/// waits for the current frame's flip, checks and drops what it recorded.
fn record_frame(cp: &mut Cp, post: &Post, w: WalkState, next: Vec<NextBuf>, flip_sub: u64) {
    STATS[0].fetch_add(1, Ordering::Relaxed);
    let t0 = Instant::now();
    let mut labels: HashMap<u64, (u64, u8)> = HashMap::new();
    for &(a, v, n) in &w.labels {
        labels.insert(a, (v, n));
    }
    let h = Box::new(HeldCp { frames: post.frames, labels, stop: None, commands: 0, waits: [0; 5], poll: true });
    cp.held_begin(&w, h);
    let stopped = |cp: &Cp| cp.held.as_ref().is_some_and(|h| h.stop.is_some());
    let mut last_sub = flip_sub;
    let mut ran_out = false;
    for &(sub, t, a, n) in &next {
        if sub != last_sub {
            cp.new_submission();
            last_sub = sub;
        }
        cp.submitted_at = Some(t);
        cp.next_in_submission = true;
        cp.run(a, n);
        if stopped(cp) {
            break;
        }
    }
    // Later submissions as they come, while the current frame lasts.
    while !stopped(cp) {
        let more = queued_after(last_sub);
        match more.into_iter().next() {
            Some((index, bufs, at)) if index == last_sub + 1 => {
                cp.new_submission();
                cp.submitted_at = Some(at);
                last_sub = index;
                for (a, n) in bufs {
                    cp.next_in_submission = true;
                    cp.run(a, n);
                    if stopped(cp) {
                        break;
                    }
                }
            }
            _ => {
                if frame_finished(post.frames) || super::shutting_down() {
                    ran_out = true;
                    break;
                }
                let mut g = SH.lock();
                CV.wait_for(&mut g, Duration::from_millis(1));
            }
        }
    }
    let held = cp.held_end();
    let rec_ns = t0.elapsed().as_nanos() as u64;
    // Where the graphics command processor would take over.
    if let Some((_, at)) = held.as_ref().and_then(|h| h.stop) {
        TARGET_FRAMES.store(post.frames + 1, Ordering::Release);
        TARGET.store(at, Ordering::Release);
    }
    let (why, commands, waits) = match &held {
        Some(h) => (h.stop.map_or(if ran_out { "end of the queue" } else { "frame done" }, |s| s.0), h.commands, h.waits),
        None => ("no state", 0, [0; 5]),
    };
    count(&STOPS, why);
    STATS[1].fetch_add(commands, Ordering::Relaxed);
    STATS[5].fetch_add(rec_ns, Ordering::Relaxed);
    for (k, n) in waits.iter().enumerate() {
        WAITS[k].fetch_add(*n, Ordering::Relaxed);
    }
    {
        let mut p = PER_FRAME.lock();
        if p.len() < 1 << 16 {
            p.push((commands.min(u32::MAX as u64) as u32, rec_ns as f32 / 1e6));
        }
    }
    // The flip: the current frame's writes are complete.
    let target = post.frames + 1;
    while FLIPPED.load(Ordering::SeqCst) < target && !super::shutting_down() {
        let mut g = SH.lock();
        CV.wait_for(&mut g, Duration::from_millis(2));
    }
    let at_flip = AT_FLIP.lock().take();
    if why == "frame done" {
        if let Some((_, _, t, _)) = &at_flip {
            STATS[6].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
            STATS[7].fetch_add(1, Ordering::Relaxed);
        }
    }
    let Some(e) = cp.held_exec() else { return };
    STATS[8].fetch_add(e.closed_count() as u64, Ordering::Relaxed);
    {
        let writes = A_WRITES.lock();
        if writes.overlaps_any(e.held_reads()) {
            STATS[3].fetch_add(1, Ordering::Relaxed);
        }
    }
    if let Some((n, actual, _, pending)) = &at_flip {
        if *n == target {
            let (d, fields) = w.diff(actual);
            if !d.is_empty() || !fields {
                STATS[4].fetch_add(1, Ordering::Relaxed);
            }
            if pending.iter().any(|k| e.held_keys().contains(k)) {
                STATS[10].fetch_add(1, Ordering::Relaxed);
            }
        }
    }
    // Stage 1: dropped.
    e.discard_closed();
}

/// Clears the statistics (`profile reset`).
pub fn reset() {
    for c in STATS.iter().chain(WAITS.iter()) {
        c.store(0, Ordering::Relaxed);
    }
    STOPS.lock().clear();
    WALK_ERRORS.lock().clear();
    PER_FRAME.lock().clear();
    *POLLS.lock() = None;
    for c in &REACHED {
        c.store(0, Ordering::Relaxed);
    }
    REACH_MS.lock().clear();
}

/// One profile line ("" before any frame).
pub fn status() -> String {
    let n = |i: usize| STATS[i].load(Ordering::Relaxed);
    if n(0) == 0 && n(2) == 0 {
        return String::new();
    }
    let list = |l: &Mutex<Vec<(&'static str, u64)>>| {
        let mut v = l.lock().clone();
        v.sort_by(|a, b| b.1.cmp(&a.1));
        v.iter().map(|(w, c)| format!("{w} {c}")).collect::<Vec<_>>().join(", ")
    };
    let per = PER_FRAME.lock().clone();
    let pct = |mut v: Vec<f32>, q: f64| -> f32 {
        if v.is_empty() {
            return 0.0;
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[((v.len() - 1) as f64 * q) as usize]
    };
    let cmds: Vec<f32> = per.iter().map(|x| x.0 as f32).collect();
    let ms: Vec<f32> = per.iter().map(|x| x.1).collect();
    let w: Vec<u64> = WAITS.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    let reach: Vec<f32> = REACH_MS.lock().clone();
    let polls = {
        let g = POLLS.lock();
        let mut v: Vec<(&(u64, String), &PollStats)> = g.as_ref().map(|m| m.iter().collect()).unwrap_or_default();
        v.sort_by(|a, b| b.1.0[0].cmp(&a.1.0[0]));
        v.iter()
            .take(6)
            .map(|((a, q), (x, r, st))| {
                let mut st: Vec<(&String, &u64)> = st.iter().collect();
                st.sort_by(|a, b| b.1.cmp(a.1));
                let st: Vec<String> = st.iter().take(3).map(|(s, n)| format!("{n}x {s}")).collect();
                format!("{a:#x} ({q}, last for {r:#x}) {} polled, {} passed, {:.2} ms each after {:.0} commands (writer then: {})", x[0], x[1], x[2] as f64 / 1e6 / x[0].max(1) as f64, x[3] as f64 / x[0].max(1) as f64, st.join(" | "))
            })
            .collect::<Vec<_>>()
            .join("; ")
    };
    format!(
        "    frame-ahead recording (stage 1, dropped): {} frames begun ({} not walked: {}; {} given up), {} commands recorded ahead (per frame p10 {:.0}, p50 {:.0}, p90 {:.0}; recording p50 {:.1}, p90 {:.1} ms), {} batches held; stops: {}; label waits passed on its own writes {}, the frame's {}, memory {}, other queues' promises {}, after polling {}; would not hold: CPU reads written by the frame's rest {}, start state differing {}, images with the frame's clears or overwrites pending {}; the flip to its stop {:.2} ms ({}); the graphics command processor reached its stop {} times, p10 {:.1}, p50 {:.1}, p90 {:.1} ms after its flip; waits polled: {}\n",
        n(0),
        n(2),
        list(&WALK_ERRORS),
        n(9),
        n(1),
        pct(cmds.clone(), 0.1),
        pct(cmds.clone(), 0.5),
        pct(cmds, 0.9),
        pct(ms.clone(), 0.5),
        pct(ms, 0.9),
        n(8),
        list(&STOPS),
        w[0],
        w[1],
        w[2],
        w[3],
        w[4],
        n(3),
        n(4),
        n(10),
        n(6) as f64 / 1e6 / n(7).max(1) as f64,
        n(7),
        REACHED[0].load(Ordering::Relaxed),
        pct(reach.clone(), 0.1),
        pct(reach.clone(), 0.5),
        pct(reach, 0.9),
        if polls.is_empty() { "-".to_string() } else { polls }
    )
}
