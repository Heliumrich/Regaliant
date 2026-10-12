//! CPU time of the GPU front end's threads from their cycle counters, by
//! kind: the queues' command processors ("gfx", "acb"), the parallel
//! recording workers ("par"), stage preparation ("prep"), batch submission
//! ("submit"), batch completion ("complete") and texture scrubbing
//! ("scrub"). QueryThreadCycleTime counts
//! the time-stamp counter while the thread runs, exactly (thread times from
//! GetThreadTimes advance in clock ticks of ~15.6 ms), and does not include
//! time other processes take from the GPU: per-command costs compare across
//! windows with different outside load, where frame rates do not. The
//! counter's rate is measured once against the system clock.

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::time::Instant;

pub const KINDS: [&str; 7] = ["gfx", "acb", "par", "prep", "submit", "complete", "scrub"];

/// Thread handles by host thread id, kept open (the threads live as long as
/// the process).
static HANDLES: Mutex<Option<HashMap<u32, usize>>> = Mutex::new(None);

fn thread_cycles(tid: u32) -> Option<u64> {
    use windows_sys::Win32::System::Threading::{OpenThread, THREAD_QUERY_LIMITED_INFORMATION};
    use windows_sys::Win32::System::WindowsProgramming::QueryThreadCycleTime;
    let mut g = HANDLES.lock();
    let m = g.get_or_insert_with(HashMap::new);
    let h = match m.get(&tid) {
        Some(&h) => h,
        None => {
            let h = unsafe { OpenThread(THREAD_QUERY_LIMITED_INFORMATION, 0, tid) };
            if h.is_null() {
                return None;
            }
            m.insert(tid, h as usize);
            h as usize
        }
    };
    let mut c = 0u64;
    (unsafe { QueryThreadCycleTime(h as _, &mut c) } != 0).then_some(c)
}

/// Time-stamp counter cycles per second (measured over 20 ms at first use).
pub fn tsc_hz() -> f64 {
    static HZ: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *HZ.get_or_init(|| {
        let (c0, t0) = (unsafe { core::arch::x86_64::_rdtsc() }, Instant::now());
        std::thread::sleep(std::time::Duration::from_millis(20));
        let c1 = unsafe { core::arch::x86_64::_rdtsc() };
        (c1.wrapping_sub(c0)) as f64 / t0.elapsed().as_secs_f64()
    })
}

/// Milliseconds of `cycles`.
pub fn ms(cycles: u64) -> f64 {
    cycles as f64 * 1000.0 / tsc_hz()
}

/// Cycles so far per kind (indexed like `KINDS`).
pub fn snapshot() -> [u64; KINDS.len()] {
    let mut out = [0u64; KINDS.len()];
    for (kind, tid) in super::cpu_threads() {
        if let (Some(k), Some(c)) = (KINDS.iter().position(|&x| x == kind), thread_cycles(tid)) {
            out[k] += c;
        }
    }
    out
}

/// Commands (draws and dispatches) executed so far.
pub fn commands() -> u64 {
    super::cp::DRAWS.load(Ordering::Relaxed) + super::cp::DISPATCHES.load(Ordering::Relaxed)
}

struct Window {
    at: Instant,
    cycles: [u64; KINDS.len()],
    flips: u64,
    commands: u64,
}

static WINDOW: Mutex<Option<Window>> = Mutex::new(None);

/// Starts a profile window (`profile reset`).
pub fn reset_window() {
    let w = Window { at: Instant::now(), cycles: snapshot(), flips: super::FRAMES.load(Ordering::Relaxed), commands: commands() };
    *WINDOW.lock() = Some(w);
}

/// The window's thread CPU time per flip and per command (`profile report`).
pub fn window_line() -> String {
    let g = WINDOW.lock();
    let Some(w) = g.as_ref() else { return String::new() };
    let now = snapshot();
    let d: Vec<u64> = (0..KINDS.len()).map(|i| now[i].saturating_sub(w.cycles[i])).collect();
    let flips = super::FRAMES.load(Ordering::Relaxed).saturating_sub(w.flips).max(1);
    let cmds = commands().saturating_sub(w.commands).max(1);
    let all: u64 = d.iter().sum();
    let parts: Vec<String> = KINDS.iter().zip(&d).map(|(k, &c)| format!("{k} {:.1}", ms(c) / flips as f64)).collect();
    format!(
        "    thread CPU in the window, ms per flip: {} (all {:.1}); {:.2} us per command over all threads; {flips} flips, {cmds} commands in {:.1} s\n",
        parts.join(", "),
        ms(all) / flips as f64,
        ms(all) * 1000.0 / cmds as f64,
        w.at.elapsed().as_secs_f64()
    )
}
