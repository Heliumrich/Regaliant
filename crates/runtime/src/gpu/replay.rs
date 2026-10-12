//! Frame replay (`replay <iterations> [<switch> <a> <b>]` in the control
//! file): a repeatable benchmark of the GPU front end on one frame of the
//! running title, without the title's CPU work and with the same commands in
//! every measurement.
//!
//! 1. After a submission with a flip (a frame's end), the title's next
//!    submission is held (its submit call waits) until every queue has
//!    completed its work, GPU work and deferred effects included
//!    (`Ctl::Finish`).
//! 2. The title then runs one frame as usual: its submissions are recorded
//!    in each queue's order up to the next one with a flip, and the
//!    submission after that is held until the queues are idle again. For
//!    every location a CP write of the frame targets (labels, CP data writes
//!    up to `MAX_WRITE` bytes) its contents before the frame's first write
//!    to it are kept. If the queues do not go idle within `QUIET` (a wait
//!    the title releases only after the held submission), the hold is
//!    lifted and the next frame is tried.
//! 3. Each iteration restores those locations and each queue's register
//!    state of the frame's start, queues the frame's submissions again and
//!    waits until all queues have completed them. Flips, display-buffer
//!    waits and EOP interrupts are skipped. Waits the title's CPU released
//!    while the frame ran are not released now: a wait on a label no CP
//!    write of the frame targets passes at once (only the CPU writes it: as
//!    if the CPU were done), and one on a label the frame writes that still
//!    blocks after `WAIT_LIMIT` passes ("forced", listed) and passes at once
//!    in later iterations.
//! 4. The locations get their contents of step 2's end back, the queues
//!    their register states, and the held submission proceeds.
//!
//! Each iteration reports its wall time from queueing to completion, the
//! CPU time of the front end's threads by kind (`threadcpu`), its commands,
//! and the GPU time inside batches when batch stamps are on
//! (`DS_PROFILE_GPU=1`). With `<switch> <a> <b>` the switch alternates per
//! iteration (a first); the summary compares medians without each value's
//! first iteration (pipelines of a new setting compile there). The frame's
//! command buffers (top levels) get their contents at submission back
//! before each iteration (some change after it: the differences are listed
//! once) and their contents at the hold afterwards; other memory the frame
//! reads is replayed as it is then (the title does not run while it is
//! held).

use parking_lot::{Condvar, Mutex};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use super::{threadcpu, Ctl};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Phase {
    Off,
    /// Waiting for a submission with a flip.
    Armed,
    /// The title's next submission waits: the frame to record starts there.
    Start,
    /// The title's submissions are recorded up to one with a flip.
    Recording,
    /// The title's next submission waits until the replay ends.
    End,
}

struct State {
    phase: Phase,
    /// Title submissions waiting in `hold`.
    held: u32,
    /// The recorded frame: (queue, command buffers, their contents at
    /// submission).
    frame: Vec<(u32, Vec<(u64, u32)>, Vec<Vec<u8>>)>,
}

static ST: Mutex<State> = Mutex::new(State { phase: Phase::Off, held: 0, frame: Vec::new() });
static CV: Condvar = Condvar::new();
/// A replay is armed or running (checked without the lock per submission).
static ACTIVE: AtomicBool = AtomicBool::new(false);
/// CP writes are noted (step 2).
static RECORDING: AtomicBool = AtomicBool::new(false);
/// Iterations are running (step 3).
static REPLAYING: AtomicBool = AtomicBool::new(false);
/// Locations CP writes of the recorded frame target and their contents
/// before its first write to each, in the order first written.
static BEFORE: Mutex<(Vec<(u64, Vec<u8>)>, Option<std::collections::HashSet<(u64, u64)>>)> = Mutex::new((Vec::new(), None));
/// CP writes of the recorded frame too large to restore (DMA copies).
static UNRESTORED: AtomicU64 = AtomicU64::new(0);
/// The 8-byte words the restored locations cover (iterations).
static RESTORED: Mutex<Option<std::collections::HashSet<u64>>> = Mutex::new(None);
/// Waits forced in an iteration: (packet address, what it waited for).
static FORCED: Mutex<Vec<(u64, String)>> = Mutex::new(Vec::new());
/// Labels only the CPU writes that waits passed on: (packet, label).
static CPU_PASSED: Mutex<Vec<(u64, u64)>> = Mutex::new(Vec::new());
/// Waits forced, waits passed as forced before, and waits passed on labels
/// only the CPU writes, so far.
static FORCED_N: AtomicU64 = AtomicU64::new(0);
static PASSED_N: AtomicU64 = AtomicU64::new(0);
static CPU_N: AtomicU64 = AtomicU64::new(0);

/// Largest CP write restored per iteration.
const MAX_WRITE: u64 = 256;
/// How long the queues may take to go idle while the title is held.
const QUIET: Duration = Duration::from_secs(5);
/// How long a wait on a label the frame writes may block in an iteration
/// before it passes.
const WAIT_LIMIT: Duration = Duration::from_secs(2);

/// Whether replay iterations are running: the command processors skip
/// flips, display-buffer waits and EOP interrupts.
pub fn replaying() -> bool {
    REPLAYING.load(Ordering::Relaxed)
}

/// The title's submission path: waits while the replay holds the title.
pub fn hold() {
    if !ACTIVE.load(Ordering::Relaxed) {
        return;
    }
    let mut g = ST.lock();
    let phase = g.phase;
    if !matches!(phase, Phase::Start | Phase::End) {
        return;
    }
    g.held += 1;
    CV.notify_all();
    while g.phase == phase && !super::shutting_down() {
        CV.wait_for(&mut g, Duration::from_millis(100));
    }
    g.held -= 1;
}

/// A title submission being queued (under its queue's lock: in queue order).
pub fn note_submitted(queue: u32, bufs: &[(u64, u32)]) {
    if !ACTIVE.load(Ordering::Relaxed) {
        return;
    }
    let mut g = ST.lock();
    match g.phase {
        Phase::Armed => {
            if super::has_flip(bufs) {
                g.phase = Phase::Start;
                CV.notify_all();
            }
        }
        // Submissions that passed the hold before it began belong to the frame.
        Phase::Recording | Phase::End => {
            let copies = bufs.iter().map(|&(a, n)| if super::cp::guest_readable(a, 4 * n as u64) { read(a, 4 * n as usize) } else { Vec::new() }).collect();
            g.frame.push((queue, bufs.to_vec(), copies));
            if g.phase == Phase::Recording && super::has_flip(bufs) {
                g.phase = Phase::End;
                CV.notify_all();
            }
        }
        _ => {}
    }
}

/// A CP write of `len` bytes at `a` (recorded or performed): while a frame
/// is recorded, the location's contents before the frame's first write.
pub fn note_cp_write(a: u64, len: u64) {
    if !RECORDING.load(Ordering::Relaxed) {
        return;
    }
    if len > MAX_WRITE {
        UNRESTORED.fetch_add(1, Ordering::Relaxed);
        return;
    }
    let mut g = BEFORE.lock();
    let (list, seen) = &mut *g;
    if !seen.get_or_insert_with(Default::default).insert((a, len)) || !super::cp::guest_readable(a, len) {
        return;
    }
    list.push((a, read(a, len as usize)));
}

/// A wait at packet `at` on `label` about to block in an iteration: passes
/// if it was forced before or no CP write of the frame targets the label.
pub fn pass_wait(at: u64, label: u64) -> bool {
    if !replaying() {
        return false;
    }
    if FORCED.lock().iter().any(|f| f.0 == at) {
        PASSED_N.fetch_add(1, Ordering::Relaxed);
        return true;
    }
    if RESTORED.lock().as_ref().is_some_and(|r| r.contains(&(label & !7))) {
        return false;
    }
    CPU_N.fetch_add(1, Ordering::Relaxed);
    let mut g = CPU_PASSED.lock();
    if g.len() < 64 && !g.iter().any(|x| x.0 == at) {
        g.push((at, label));
    }
    true
}

/// A wait blocking in an iteration since `since`: passes after `WAIT_LIMIT`.
pub fn force_wait(at: u64, since: Instant, what: &str) -> bool {
    if !replaying() || since.elapsed() < WAIT_LIMIT {
        return false;
    }
    FORCED_N.fetch_add(1, Ordering::Relaxed);
    let mut g = FORCED.lock();
    if !g.iter().any(|f| f.0 == at) {
        g.push((at, what.to_string()));
    }
    true
}

fn read(a: u64, len: usize) -> Vec<u8> {
    unsafe { std::slice::from_raw_parts(a as *const u8, len) }.to_vec()
}

/// Writes locations back, the earlier first-written last (their contents
/// stand where locations overlap).
fn write_back(list: &[(u64, Vec<u8>)]) {
    for (a, b) in list.iter().rev() {
        let n = b.len() as u64;
        super::precache::invalidate(*a, n);
        unsafe { std::ptr::copy_nonoverlapping(b.as_ptr(), *a as *mut u8, b.len()) };
        super::homes::cpu_written(*a, n);
        super::mirror::invalidate(&[(*a, n)]);
    }
}

/// The command buffers' contents as (address, bytes): from the copies, or
/// as they are now.
fn buffers(frame: &[(u32, Vec<(u64, u32)>, Vec<Vec<u8>>)], now: bool) -> Vec<(u64, Vec<u8>)> {
    let mut v = Vec::new();
    for (_, bufs, copies) in frame {
        for (&(a, n), c) in bufs.iter().zip(copies) {
            if !c.is_empty() {
                v.push((a, if now { read(a, 4 * n as usize) } else { c.clone() }));
            }
        }
    }
    v
}

/// The dwords of the command buffers that differ from their copies (a few).
fn buffer_changes(rec: &[(u64, Vec<u8>)], now: &[(u64, Vec<u8>)]) -> (usize, Vec<String>) {
    let (mut n, mut first) = (0, Vec::new());
    for ((a, r), (_, c)) in rec.iter().zip(now) {
        for (k, (x, y)) in r.chunks_exact(4).zip(c.chunks_exact(4)).enumerate() {
            if x != y {
                n += 1;
                if first.len() < 8 {
                    let (x, y) = (u32::from_le_bytes(x.try_into().unwrap()), u32::from_le_bytes(y.try_into().unwrap()));
                    first.push(format!("{:#x} {x:#x}->{y:#x}", a + 4 * k as u64));
                }
            }
        }
    }
    (n, first)
}

fn set_phase(p: Phase) {
    ST.lock().phase = p;
    CV.notify_all();
}

/// Waits until `phase` holds a title submission.
fn wait_held(phase: Phase, limit: Duration) -> bool {
    let t0 = Instant::now();
    let mut g = ST.lock();
    while !(g.phase == phase && g.held > 0) {
        if t0.elapsed() > limit || super::shutting_down() {
            return false;
        }
        CV.wait_for(&mut g, Duration::from_millis(50));
    }
    true
}

fn wait_idle(limit: Duration) -> bool {
    let t0 = Instant::now();
    while !super::queues_idle() {
        if t0.elapsed() > limit || super::shutting_down() {
            return false;
        }
        std::thread::sleep(Duration::from_micros(200));
    }
    true
}

/// Every queue completes its work, GPU work and deferred effects included.
fn quiesce(limit: Duration) -> bool {
    for q in super::queue_ids() {
        super::enqueue(q, Vec::new(), Some(Ctl::Finish));
    }
    wait_idle(limit)
}

fn to_queues(c: Ctl) {
    for q in super::queue_ids() {
        super::enqueue(q, Vec::new(), Some(c));
    }
}

#[derive(Clone, Copy)]
struct Sample {
    at: Instant,
    cycles: [u64; threadcpu::KINDS.len()],
    draws: u64,
    dispatches: u64,
    packets: u64,
    batch_ns: u64,
    forced: u64,
    passed: u64,
    cpu_passed: u64,
}

fn sample() -> Sample {
    Sample {
        at: Instant::now(),
        cycles: threadcpu::snapshot(),
        draws: super::cp::DRAWS.load(Ordering::Relaxed),
        dispatches: super::cp::DISPATCHES.load(Ordering::Relaxed),
        packets: super::cp::PACKETS.load(Ordering::Relaxed),
        batch_ns: super::exec::batch_ns(),
        forced: FORCED_N.load(Ordering::Relaxed),
        passed: PASSED_N.load(Ordering::Relaxed),
        cpu_passed: CPU_N.load(Ordering::Relaxed),
    }
}

/// One iteration's measurements.
struct Run {
    value: Option<u64>,
    wall_ms: f64,
    /// Thread CPU ms by kind (`threadcpu::KINDS`).
    cpu: [f64; threadcpu::KINDS.len()],
    commands: u64,
}

impl Run {
    fn cpu_all(&self) -> f64 {
        self.cpu.iter().sum()
    }
    fn us_per_command(&self) -> f64 {
        self.cpu_all() * 1000.0 / self.commands.max(1) as f64
    }
}

/// Arms a replay of `iterations` iterations (see the module documentation).
pub fn start(iterations: u32, ab: Option<(String, u64, u64)>) -> Result<(), String> {
    if !super::exec::enabled() {
        return Err("GPU execution is off".into());
    }
    if let Some((name, _, _)) = &ab {
        if super::tune::value(name).is_none() {
            return Err(format!("unknown switch '{name}'"));
        }
    }
    {
        let mut g = ST.lock();
        if g.phase != Phase::Off || ACTIVE.load(Ordering::SeqCst) {
            return Err("a replay is in progress".into());
        }
        g.phase = Phase::Armed;
        g.frame.clear();
    }
    ACTIVE.store(true, Ordering::SeqCst);
    let r = crate::thread::spawn_host(
        "gpu-replay",
        1 << 20,
        Box::new(move || {
            crate::log::set_thread_name("gpu-replay");
            drive(iterations, ab);
            RECORDING.store(false, Ordering::SeqCst);
            REPLAYING.store(false, Ordering::SeqCst);
            set_phase(Phase::Off);
            ACTIVE.store(false, Ordering::SeqCst);
        }),
    );
    if let Err(e) = r {
        set_phase(Phase::Off);
        ACTIVE.store(false, Ordering::SeqCst);
        return Err(format!("no replay thread: {e}"));
    }
    Ok(())
}

/// Steps 1 and 2: a frame recorded between two idle points; false if none.
fn record() -> bool {
    for attempt in 1..=3 {
        if !wait_held(Phase::Start, Duration::from_secs(30)) {
            crate::warn!("replay: no flip followed by a submission within 30 s: stopped");
            return false;
        }
        if !quiesce(QUIET) {
            crate::warn!("replay: attempt {attempt}: the queues did not go idle within {} s while the title was held; trying the next frame", QUIET.as_secs());
            set_phase(Phase::Armed);
            continue;
        }
        to_queues(Ctl::Save(0));
        {
            let mut g = BEFORE.lock();
            g.0.clear();
            g.1 = None;
        }
        UNRESTORED.store(0, Ordering::Relaxed);
        ST.lock().frame.clear();
        RECORDING.store(true, Ordering::SeqCst);
        set_phase(Phase::Recording);
        if !wait_held(Phase::End, Duration::from_secs(30)) {
            RECORDING.store(false, Ordering::SeqCst);
            crate::warn!("replay: the recorded frame did not end with a flip within 30 s: stopped");
            return false;
        }
        let quiet = quiesce(QUIET);
        RECORDING.store(false, Ordering::SeqCst);
        if quiet {
            return true;
        }
        crate::warn!("replay: attempt {attempt}: the recorded frame did not complete within {} s while the title was held; trying the next frame", QUIET.as_secs());
        set_phase(Phase::Armed);
    }
    crate::warn!("replay: no frame recorded in 3 attempts: stopped");
    false
}

fn drive(iterations: u32, ab: Option<(String, u64, u64)>) {
    crate::info!("replay: armed for {iterations} iteration(s){}", ab.as_ref().map(|(n, a, b)| format!(", {n} alternating {a} / {b}")).unwrap_or_default());
    if !record() {
        return;
    }
    let frame = ST.lock().frame.clone();
    let before = std::mem::take(&mut BEFORE.lock().0);
    let end: Vec<(u64, Vec<u8>)> = before.iter().map(|(a, b)| (*a, read(*a, b.len()))).collect();
    let bufs_rec = buffers(&frame, false);
    let bufs_end = buffers(&frame, true);
    let (changed, first) = buffer_changes(&bufs_rec, &bufs_end);
    if changed > 0 {
        crate::info!("replay: {changed} dwords of the frame's command buffers changed after their submission (restored for each iteration): {}", first.join(", "));
    }
    to_queues(Ctl::Save(1));
    let mut per_queue: Vec<(u32, usize, u64)> = Vec::new();
    for (q, bufs, _) in &frame {
        let dw: u64 = bufs.iter().map(|b| b.1 as u64).sum();
        match per_queue.iter_mut().find(|x| x.0 == *q) {
            Some(x) => {
                x.1 += 1;
                x.2 += dw;
            }
            None => per_queue.push((*q, 1, dw)),
        }
    }
    let queues: Vec<String> = per_queue.iter().map(|(q, n, dw)| format!("{} {n} ({} KiB)", if *q == super::GFX { "gfx".to_string() } else { format!("acb{q:#x}") }, dw * 4 >> 10)).collect();
    crate::info!(
        "replay: frame recorded at flip {}: {} submissions ({}); {} locations restored per iteration, {} larger CP writes not",
        super::FRAMES.load(Ordering::Relaxed),
        frame.len(),
        queues.join(", "),
        before.len(),
        UNRESTORED.load(Ordering::Relaxed)
    );
    let orig = ab.as_ref().and_then(|(n, _, _)| super::tune::value(n));
    FORCED.lock().clear();
    CPU_PASSED.lock().clear();
    *RESTORED.lock() = Some(before.iter().flat_map(|(a, b)| (*a & !7..*a + b.len() as u64).step_by(8)).collect());
    let mut runs: Vec<Run> = Vec::new();
    REPLAYING.store(true, Ordering::SeqCst);
    for i in 0..iterations {
        let value = ab.as_ref().map(|(name, a, b)| {
            let v = if i % 2 == 0 { *a } else { *b };
            super::tune::set(name, v);
            v
        });
        write_back(&bufs_rec);
        write_back(&before);
        to_queues(Ctl::Restore(0));
        let s0 = sample();
        for (q, bufs, _) in &frame {
            super::enqueue(*q, bufs.clone(), None);
        }
        let done = quiesce(Duration::from_secs(60));
        let s1 = sample();
        if !done {
            crate::warn!("replay: iteration {} did not complete within 60 s: stopped", i + 1);
            // Waits still pass after WAIT_LIMIT while the queues drain.
            wait_idle(Duration::from_secs(60));
            break;
        }
        let mut cpu = [0.0; threadcpu::KINDS.len()];
        for (k, c) in cpu.iter_mut().enumerate() {
            *c = threadcpu::ms(s1.cycles[k].saturating_sub(s0.cycles[k]));
        }
        let draws = s1.draws - s0.draws;
        let dispatches = s1.dispatches - s0.dispatches;
        let r = Run { value, wall_ms: (s1.at - s0.at).as_secs_f64() * 1000.0, cpu, commands: draws + dispatches };
        let kinds: Vec<String> = threadcpu::KINDS.iter().zip(&r.cpu).map(|(k, c)| format!("{k} {c:.1}")).collect();
        let gpu = s1.batch_ns - s0.batch_ns;
        crate::info!(
            "replay {}/{iterations}{}: {:.1} ms; thread CPU ms {} (all {:.1}; {:.2} us per command); {draws} draws, {dispatches} dispatches, {} packets{}{}",
            i + 1,
            value.map(|v| format!(" ({}={v})", ab.as_ref().unwrap().0)).unwrap_or_default(),
            r.wall_ms,
            kinds.join(", "),
            r.cpu_all(),
            r.us_per_command(),
            s1.packets - s0.packets,
            if gpu > 0 { format!("; GPU in batches {:.1} ms", gpu as f64 / 1e6) } else { String::new() },
            format!("; waits passed on CPU-written labels {}, forced {}, passed as forced before {}", s1.cpu_passed - s0.cpu_passed, s1.forced - s0.forced, s1.passed - s0.passed)
        );
        runs.push(r);
    }
    REPLAYING.store(false, Ordering::SeqCst);
    // The title's view as at the hold.
    write_back(&bufs_end);
    write_back(&end);
    to_queues(Ctl::Restore(1));
    wait_idle(QUIET);
    if let (Some((name, _, _)), Some(v)) = (&ab, orig) {
        super::tune::set(name, v);
    }
    for (at, what) in FORCED.lock().iter() {
        crate::info!("replay: forced wait at packet {at:#x}: {what}");
    }
    let cpu: Vec<String> = CPU_PASSED.lock().iter().map(|(at, l)| format!("{l:#x} (packet {at:#x})")).collect();
    if !cpu.is_empty() {
        crate::info!("replay: waits passed on labels no CP write of the frame targets: {}", cpu.join(", "));
    }
    *RESTORED.lock() = None;
    summary(&runs, ab.as_ref().map(|x| x.0.as_str()));
}

fn median(mut v: Vec<f64>) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    if n % 2 == 1 { v[n / 2] } else { (v[n / 2 - 1] + v[n / 2]) / 2.0 }
}

/// Medians per value, without each value's first iteration when it has more.
fn summary(runs: &[Run], name: Option<&str>) {
    let mut values: Vec<Option<u64>> = Vec::new();
    for r in runs {
        if !values.contains(&r.value) {
            values.push(r.value);
        }
    }
    let mut medians: Vec<(Option<u64>, f64, f64)> = Vec::new();
    for v in &values {
        let all: Vec<&Run> = runs.iter().filter(|r| r.value == *v).collect();
        let used: Vec<&Run> = if all.len() > 1 { all[1..].to_vec() } else { all.clone() };
        let wall = median(used.iter().map(|r| r.wall_ms).collect());
        let cpu: Vec<String> = threadcpu::KINDS.iter().enumerate().map(|(k, kind)| format!("{kind} {:.1}", median(used.iter().map(|r| r.cpu[k]).collect()))).collect();
        let us = median(used.iter().map(|r| r.us_per_command()).collect());
        let label = match (name, v) {
            (Some(n), Some(x)) => format!("{n}={x}"),
            _ => "frame".to_string(),
        };
        crate::info!(
            "replay summary {label}: median of {} iteration(s): {wall:.1} ms (min {:.1}, max {:.1}); thread CPU ms {} (all {:.1}); {us:.2} us per command",
            used.len(),
            used.iter().map(|r| r.wall_ms).fold(f64::MAX, f64::min),
            used.iter().map(|r| r.wall_ms).fold(0.0, f64::max),
            cpu.join(", "),
            median(used.iter().map(|r| r.cpu_all()).collect())
        );
        medians.push((*v, wall, us));
    }
    if let (Some(n), [a, b]) = (name, medians.as_slice()) {
        crate::info!(
            "replay summary {n}: {}/{}: wall {:.3}, CPU per command {:.3}",
            b.0.unwrap_or(0),
            a.0.unwrap_or(0),
            b.1 / a.1.max(1e-9),
            b.2 / a.2.max(1e-9)
        );
    }
}
