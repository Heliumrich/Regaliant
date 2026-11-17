//! The display's own vertical blanks, to pace the title's emulated vblank to the screen
//! (`hle::libs::videoout`): a thread waits for each vblank of the monitor the game's window is
//! on (`D3DKMTWaitForVerticalBlankEvent`) and keeps when the last one came, how many there were
//! and the refresh period measured over the last second.
//!
//! The title flips at every emulated vblank. A timer at 59.94 Hz cannot be even on a 165 Hz
//! screen (its frames are shown alternately 2 and 3 refreshes: 12 and 18 ms, a judder at a
//! steady 60 fps; the user, 2026-11-15: "near 60 it didn't feel all that smooth"), and against
//! a 120 Hz screen it drifts (a frame shown three refreshes every ~9 s). So at every refresh
//! rate the title's vblank ticks at each of the display's own (`lock`), and the frame limit
//! is a whole number of them between flips (`refreshes_for`): every frame at the limit stays
//! that many refreshes, and a frame done late is shown one refresh later, not a whole frame.
//!
//! On a variable-refresh screen (G-Sync, FreeSync) the vblanks follow the frames presented, so
//! a lock would feed back on itself: `vrr::active` (the driver's word where there is one, else
//! refresh intervals well above the mode's period here, `vrr_seen`) keeps the lock off, and
//! the frame limit is the vblank thread's own timer there (the screen shows each frame as it
//! comes).

use parking_lot::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering::Relaxed};
use std::time::{Duration, Instant};

#[repr(C)]
struct OpenAdapterFromHdc {
    hdc: windows_sys::Win32::Graphics::Gdi::HDC,
    h_adapter: u32,
    luid_low: u32,
    luid_high: i32,
    vidpn_source_id: u32,
}

#[repr(C)]
struct WaitForVblank {
    h_adapter: u32,
    h_device: u32,
    vidpn_source_id: u32,
}

#[repr(C)]
struct CloseAdapter {
    h_adapter: u32,
}

#[link(name = "gdi32")]
unsafe extern "system" {
    fn D3DKMTOpenAdapterFromHdc(p: *mut OpenAdapterFromHdc) -> i32;
    fn D3DKMTWaitForVerticalBlankEvent(p: *const WaitForVblank) -> i32;
    fn D3DKMTCloseAdapter(p: *const CloseAdapter) -> i32;
}

/// The display's vblanks as last seen.
#[derive(Clone, Copy, Debug)]
pub struct Snap {
    /// When the last vblank came and its number.
    pub last: Instant,
    pub count: u64,
    /// The refresh period measured over the last second (up to 128 vblanks).
    pub period: Duration,
    /// The mode's refresh rate (Hz, as Windows gives it: whole numbers).
    pub mode_hz: u32,
}

struct State {
    device: String,
    snap: Option<Snap>,
    /// The last vblanks seen (up to 129: when, and their number) and how many came since the
    /// clock opened the display (a wait woken late counts the vblanks it missed).
    times: std::collections::VecDeque<(Instant, u64)>,
    count: u64,
}

static STATE: Mutex<Option<State>> = Mutex::new(None);
static STARTED: AtomicBool = AtomicBool::new(false);
/// Variable refresh suspected (`vrr`): until when the lock stays off.
static VRR_UNTIL: Mutex<Option<Instant>> = Mutex::new(None);
static VRR_LOGGED: AtomicBool = AtomicBool::new(false);
/// Locks made and given up (status).
pub static LOCKS: AtomicU32 = AtomicU32::new(0);

/// Starts the clock's thread (once; without a window it waits for one). It also runs with the
/// lock off (`DS_VBLANK_LOCK=0`): the presents' cadence is measured against it.
pub fn start() {
    if STARTED.swap(true, Relaxed) {
        return;
    }
    std::thread::Builder::new().name("display-clock".into()).spawn(run).ok();
}

/// The vblanks as last seen, when the clock follows the window's monitor and has a second of them.
pub fn snapshot() -> Option<Snap> {
    STATE.lock().as_ref().and_then(|s| s.snap)
}

/// The lock: the display's vblanks as last seen, for the title's vblank to tick at each of
/// them, when the clock has a second of them at the mode's own period (a gap or a stretched
/// refresh is not a lock's) and variable refresh has not been seen here. Diagnosis:
/// `DS_VBLANK_LOCK=0` never locks; `=<n>` forces flips n refreshes apart whatever the limit
/// (`forced_refreshes`).
pub fn lock() -> Option<Snap> {
    if forced().is_some_and(|n| n == 0) {
        return None;
    }
    let s = snapshot()?;
    if vrr_seen() {
        return None;
    }
    let hz = 1.0 / s.period.as_secs_f64();
    if s.mode_hz > 0 && (hz / s.mode_hz as f64 - 1.0).abs() > 0.02 {
        return None;
    }
    Some(s)
}

fn forced() -> Option<u32> {
    static FORCED: std::sync::OnceLock<Option<u32>> = std::sync::OnceLock::new();
    *FORCED.get_or_init(|| std::env::var("DS_VBLANK_LOCK").ok().and_then(|v| v.trim_start_matches('f').parse::<u32>().ok()))
}

/// `DS_VBLANK_LOCK=<n>` (diagnosis): flips this many refreshes apart on the display's grid.
pub fn forced_refreshes() -> Option<u32> {
    forced().filter(|&n| n > 0)
}

/// Whether refreshes that follow the presents were seen lately (`note`): variable refresh, by
/// the display's own evidence (`vrr::active` asks the driver first).
pub fn vrr_seen() -> bool {
    VRR_UNTIL.lock().is_some_and(|t| Instant::now() < t)
}

/// How many of a display's refreshes a frame is shown for under a frame limit: the fewest
/// that keep the rate at or under the limit (5% over counts as at it: 60 fps on a 60.0 or a
/// 62 Hz screen is every refresh). 60 fps: 1 at 60 Hz, 2 at 120, 3 at 144 (48 fps) and 165
/// (55 fps), 4 at 240; no whole number of refreshes makes 60 where the rate is no multiple of
/// it, and an uneven mix of them is what the limit is there to avoid.
pub fn refreshes_for(display_hz: f64, limit_fps: f64) -> u32 {
    if !(display_hz > 0.0 && limit_fps > 0.0) {
        return 1;
    }
    let n = ((display_hz / limit_fps * 0.95).ceil() as u32).clamp(1, 16);
    // (Where that loses more than three frames in ten, 60 on a 72 or 75 Hz screen: every
    // refresh instead.)
    if n > 1 && display_hz / (n as f64) < limit_fps * 0.7 { n - 1 } else { n }
}

/// The refreshes between flips the vblank thread keeps now (0: its own timer), for the status.
pub static FLIPS_APART: AtomicU32 = AtomicU32::new(0);

fn monitor_hz(device: &str) -> u32 {
    if device == SIM_DEVICE {
        return sim_hz().map_or(0, |h| h.round() as u32);
    }
    crate::display::current_mode(device).map_or(0, |m| m.hz)
}

/// `DS_DISPLAY_SIM=<hz>` (tests, 24 to 500): the display's vblanks come from a timer at this
/// rate instead of a monitor's, also in runs without a window, so that the pacing on a
/// display's grid (the lock, the refreshes per frame, the game's step) runs at any refresh
/// rate where no such screen is at hand or the screen is off. What a real screen adds (the
/// swapchain's and the compositor's timing) is not in it.
pub fn sim_hz() -> Option<f64> {
    static SIM: std::sync::OnceLock<Option<f64>> = std::sync::OnceLock::new();
    *SIM.get_or_init(|| std::env::var("DS_DISPLAY_SIM").ok().and_then(|v| v.trim().parse::<f64>().ok()).filter(|h| (24.0..=500.0).contains(h)))
}

const SIM_DEVICE: &str = "simulated display";

/// The simulated display's vblanks: each at its own time on the timer's line.
fn run_sim(hz: f64) {
    crate::info!("display clock: a simulated {hz} Hz display (DS_DISPLAY_SIM): its vblanks from a timer");
    *STATE.lock() = Some(State { device: SIM_DEVICE.into(), snap: None, times: Default::default(), count: 0 });
    let period = Duration::from_secs_f64(1.0 / hz);
    let start = Instant::now();
    let mut k: u32 = 1;
    loop {
        let at = start + period * k;
        let now = Instant::now();
        if at > now {
            crate::thread::short_sleep(at - now);
        }
        // (Woken late by a period or more: the moment itself, and `note` counts the vblanks
        // missed.)
        let woke = Instant::now();
        note(SIM_DEVICE, if woke.saturating_duration_since(at) < period { at } else { woke });
        k = ((woke.saturating_duration_since(start).as_secs_f64() * hz) as u32 + 1).max(k + 1);
    }
}

fn run() {
    crate::log::set_thread_name("display-clock");
    // (It only waits: woken at once, it sees each vblank on time under the title's load.)
    unsafe {
        use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_TIME_CRITICAL};
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);
    }
    if let Some(hz) = sim_hz() {
        run_sim(hz);
    }
    let mut opened: Option<(String, u32, u32)> = None; // device, adapter, source
    let mut failures = 0u32;
    // Waits that failed in a row (a screen that is off or taken by another program's exclusive
    // fullscreen has no vblanks to wait for: tried again every two seconds, said once).
    let mut wait_failures = 0u32;
    loop {
        let want = crate::gpu::vk::window::current_monitor().map(|(_, d)| d);
        let Some(device) = want else {
            std::thread::sleep(Duration::from_millis(500));
            continue;
        };
        if opened.as_ref().is_none_or(|o| o.0 != device) {
            if let Some((_, a, _)) = opened.take() {
                unsafe { D3DKMTCloseAdapter(&CloseAdapter { h_adapter: a }) };
            }
            let wide: Vec<u16> = device.encode_utf16().chain(std::iter::once(0)).collect();
            let hdc = unsafe { windows_sys::Win32::Graphics::Gdi::CreateDCW(std::ptr::null(), wide.as_ptr(), std::ptr::null(), std::ptr::null()) };
            if hdc.is_null() {
                failures += 1;
                if failures <= 3 {
                    crate::warn!("display clock: no device context for {device}");
                }
                std::thread::sleep(Duration::from_secs(2));
                continue;
            }
            let mut oa = OpenAdapterFromHdc { hdc, h_adapter: 0, luid_low: 0, luid_high: 0, vidpn_source_id: 0 };
            let r = unsafe { D3DKMTOpenAdapterFromHdc(&mut oa) };
            unsafe { windows_sys::Win32::Graphics::Gdi::DeleteDC(hdc) };
            if r != 0 {
                failures += 1;
                if failures <= 3 {
                    crate::warn!("display clock: {device}'s adapter not opened (status {r:#x})");
                }
                std::thread::sleep(Duration::from_secs(2));
                continue;
            }
            if wait_failures == 0 {
                crate::info!("display clock: following {device}'s vblanks ({} Hz mode)", monitor_hz(&device));
            }
            opened = Some((device.clone(), oa.h_adapter, oa.vidpn_source_id));
            *STATE.lock() = Some(State { device: device.clone(), snap: None, times: Default::default(), count: 0 });
        }
        let (_, adapter, source) = opened.clone().unwrap();
        let w = WaitForVblank { h_adapter: adapter, h_device: 0, vidpn_source_id: source };
        let r = unsafe { D3DKMTWaitForVerticalBlankEvent(&w) };
        let now = Instant::now();
        if r != 0 {
            // The adapter changed or was lost, or the screen shows nothing of the desktop's
            // (0xC01E0006, "present occluded": the monitor is off, or another program holds it
            // in exclusive fullscreen): opened again, at once the first time, then every two
            // seconds. The title's vblank keeps its own timer meanwhile.
            wait_failures += 1;
            if wait_failures == 2 {
                crate::info!("display clock: {device}'s vblanks cannot be waited for (status {r:#x}{}): the title's vblank keeps its own timer; tried again every two seconds", if r as u32 == 0xC01E_0006 { ", the screen is off or held by another program" } else { "" });
            }
            unsafe { D3DKMTCloseAdapter(&CloseAdapter { h_adapter: adapter }) };
            opened = None;
            *STATE.lock() = None;
            std::thread::sleep(Duration::from_millis(if wait_failures > 1 { 2000 } else { 200 }));
            continue;
        }
        if wait_failures > 1 {
            crate::info!("display clock: following {device}'s vblanks again");
        }
        wait_failures = 0;
        note(&device, now);
    }
}

/// One vblank of `device` at `now`.
fn note(device: &str, now: Instant) {
    // The mode's rate, read again every second (a mode change starts the measure again).
    static MODE: Mutex<Option<(String, u32, Instant)>> = Mutex::new(None);
    let mode_hz = {
        let mut m = MODE.lock();
        if m.as_ref().is_none_or(|(d, _, at)| d != device || at.elapsed() > Duration::from_secs(1)) {
            let hz = monitor_hz(device);
            let changed = m.as_ref().is_some_and(|(d, h, _)| d == device && *h != hz);
            *m = Some((device.to_string(), hz, now));
            if changed {
                if let Some(s) = STATE.lock().as_mut() {
                    s.times.clear();
                    s.snap = None;
                }
                // (A mode change stalls the vblanks: not variable refresh.)
                *VRR_UNTIL.lock() = None;
                crate::info!("display clock: {device} now at {hz} Hz");
            }
        }
        m.as_ref().map_or(0, |x| x.1)
    };
    let mut g = STATE.lock();
    let Some(s) = g.as_mut() else { return };
    if s.device != device {
        return;
    }
    // The vblanks since the last one seen: one, or more where the wait woke late (the gap in
    // the mode's periods). A refresh far longer than the mode's period, many times a second, is
    // variable refresh (the screen waits for frames; a late wake is rare at this thread's
    // priority): eight within a second keep the lock off for ten.
    let mut step = 1u64;
    if let (Some(&(prev, _)), true) = (s.times.back(), mode_hz > 0) {
        let nominal = 1.0 / mode_hz as f64;
        let gap = now.duration_since(prev).as_secs_f64();
        if gap < 0.25 {
            step = ((gap / nominal).round() as u64).max(1);
        }
        if gap > nominal * 1.6 && gap < 0.25 {
            static LONG: Mutex<Vec<Instant>> = Mutex::new(Vec::new());
            let mut l = LONG.lock();
            l.retain(|t| now.duration_since(*t) < Duration::from_secs(1));
            l.push(now);
            if l.len() >= 8 {
                *VRR_UNTIL.lock() = Some(now + Duration::from_secs(10));
                if !VRR_LOGGED.swap(true, Relaxed) {
                    crate::info!("display clock: {device}'s refreshes vary ({:.1} ms against {:.1}): variable refresh or stalls; the vblank keeps its own timer while they do", gap * 1e3, nominal * 1e3);
                }
            }
        }
    }
    s.count += step;
    s.times.push_back((now, s.count));
    while s.times.len() > 129 {
        s.times.pop_front();
    }
    // (The period once half a second of vblanks or more is in.)
    let (t0, c0) = s.times[0];
    s.snap = (s.count - c0 >= 60).then(|| Snap { last: now, count: s.count, period: now.duration_since(t0) / (s.count - c0) as u32, mode_hz });
}

/// The presents' cadence in the display's refreshes: per present (`note_present`, the vblank
/// thread), how many of the display's vblanks passed since the previous one (1 to 7, 8 and
/// more). A compositor or a FIFO swapchain shows a present at the next refresh, so a frame is
/// shown that many refreshes: even pacing is one bucket (2 at 120 Hz), a 60 Hz timer against a
/// 165 Hz screen splits 2 and 3. Counted while the clock runs, the window seen or not.
static CADENCE: Mutex<([u64; 9], Option<u64>)> = Mutex::new(([0; 9], None));

/// The cadence counted afresh (`profile reset`).
pub fn reset_cadence() {
    *CADENCE.lock() = ([0; 9], None);
}

/// A frame presented now (the vblank thread).
pub fn note_present() {
    let Some(count) = STATE.lock().as_ref().map(|s| s.count) else { return };
    let mut c = CADENCE.lock();
    if let Some(prev) = c.1 {
        let d = count.saturating_sub(prev).clamp(0, 8) as usize;
        c.0[d] += 1;
    }
    c.1 = Some(count);
}

/// The frame-time windows' clause (`gpu::pacing::report_and_reset`): how many of the display's
/// refreshes lay between the window's presents (even pacing: one count), counted afresh.
pub fn window_clause() -> String {
    let (c, _) = std::mem::replace(&mut *CADENCE.lock(), ([0; 9], None));
    let total: u64 = c.iter().sum();
    let Some(s) = snapshot().filter(|_| total > 0) else { return String::new() };
    let parts: Vec<String> = c.iter().enumerate().filter(|(_, n)| **n > 0).map(|(i, n)| format!("{}{}:{n}", i, if i == 8 { "+" } else { "" })).collect();
    format!("; display refreshes between presents ({:.0} Hz, {}): {}", 1.0 / s.period.as_secs_f64(), match FLIPS_APART.load(Relaxed) {
        0 => "timer".into(),
        n => format!("locked each, flips {n} apart"),
    }, parts.join(" "))
}

/// Status: the clock's display and rate, the lock's state, the presents' cadence.
pub fn status_line() -> String {
    let Some(s) = snapshot() else { return crate::vrr::status_line() };
    let apart = FLIPS_APART.load(Relaxed);
    let cadence = {
        let c = CADENCE.lock();
        let total: u64 = c.0.iter().sum();
        if total == 0 {
            String::new()
        } else {
            let parts: Vec<String> = c.0.iter().enumerate().filter(|(_, n)| **n > 0).map(|(i, &n)| format!("{}{}: {:.1}%", i, if i == 8 { "+" } else { "" }, 100.0 * n as f64 / total as f64)).collect();
            format!("; presents {total}, refreshes between them {}", parts.join(", "))
        }
    };
    format!(
        "    display clock: {:.3} Hz measured ({} Hz mode), {} vblanks; the title's vblank {}{}{}\n",
        1.0 / s.period.as_secs_f64(),
        s.mode_hz,
        s.count,
        match apart {
            0 if std::env::var("DS_VBLANK_LOCK").ok().as_deref() == Some("0") => "keeps its own timer (DS_VBLANK_LOCK=0)".into(),
            0 => "keeps its own timer (variable refresh, a diagnosis rate, or no steady second of vblanks yet)".into(),
            n => format!("ticks at each of them, flips at least {n} apart ({:.2} frames a second at most)", 1.0 / s.period.as_secs_f64() / n as f64),
        },
        if VRR_LOGGED.load(Relaxed) { "; variable refresh or stalls seen" } else { "" },
        cadence
    ) + &crate::vrr::status_line()
}

#[cfg(test)]
mod tests {
    use super::refreshes_for;

    #[test]
    fn a_frame_limit_is_a_whole_number_of_refreshes() {
        let n60 = |hz: f64| refreshes_for(hz, 59.94);
        // (Multiples of 60, as Windows' modes measure: 59.94 to 60.5 a step.)
        assert_eq!([n60(59.94), n60(60.0), n60(60.5), n60(119.88), n60(120.0), n60(239.76), n60(240.0), n60(360.0)], [1, 1, 1, 2, 2, 4, 4, 6]);
        // No multiple: the fewest refreshes at or under the limit (48, 55 and 50 fps).
        assert_eq!([n60(144.0), n60(165.0), n60(100.0)], [3, 3, 2]);
        // 72 and 75 Hz: every refresh rather than 36 or 37.5 fps.
        assert_eq!([n60(72.0), n60(75.0), n60(50.0)], [1, 1, 1]);
        let n30 = |hz: f64| refreshes_for(hz, 29.97);
        assert_eq!([n30(60.0), n30(120.0), n30(144.0), n30(165.0), n30(240.0)], [2, 4, 5, 6, 8]);
    }
}
