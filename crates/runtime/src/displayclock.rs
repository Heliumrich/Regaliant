//! The display's own vertical blanks, to pace the title's emulated vblank to the screen
//! (`hle::libs::videoout`): a thread waits for each vblank of the monitor the game's window is
//! on (`D3DKMTWaitForVerticalBlankEvent`) and keeps when the last one came, how many there were
//! and the refresh period measured over the last second.
//!
//! The title flips at every emulated vblank. A timer at 59.94 Hz cannot be even on a 165 Hz
//! screen (its frames are shown alternately 2 and 3 refreshes: 12 and 18 ms, a judder at a
//! steady 60 fps; the user, 2026-11-15: "near 60 it didn't feel all that smooth"), and against
//! a 120 Hz screen it drifts (a frame shown three refreshes every ~9 s). Ticked at every Nth
//! real vblank where the refresh is N times ~60 Hz (`lock`), each frame stays N refreshes.
//!
//! On a variable-refresh screen (G-Sync, FreeSync) the vblanks follow the frames presented, so
//! a lock would feed back on itself: refresh intervals well above the mode's period mark it
//! (`vrr`), and the vblank keeps its own timer there (the screen then shows each frame as it
//! comes, which is even by itself).

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

/// The lock: the display's vblanks a 60 Hz frame spans (`n`, the flips' least distance) and how
/// many of them the title's vblank ticks at (`step`), with the clock's state. Where the refresh
/// is a whole multiple of ~60 Hz (within 1%: 59.94 to 60.5 Hz a step) the vblank ticks at each
/// of the display's (`step` 1, the fine grid): at 120 Hz a frame done late shows after 25 ms
/// instead of 33 (2 and 3 refreshes rather than 2 and 4), one of 17-25 ms keeps 40 fps instead
/// of 30. None where the refresh is no such multiple, the clock has no second of vblanks yet, or
/// variable refresh is suspected. Diagnosis: `DS_VBLANK_LOCK=<n>` ticks at every nth vblank
/// whatever the rate (3 on a 165 Hz screen: 55 Hz), `=f<n>` at each with flips n apart (`f3` at
/// 165 Hz: 55 fps on its finer grid), `=0` never; `DS_VBLANK_FINE=0` ticks every nth (the coarse
/// lock) where the fine one would.
pub fn lock() -> Option<(u32, u32, Snap)> {
    // (Forced: Some((n, fine)); n 0: never.)
    static FORCED: std::sync::OnceLock<Option<(u32, bool)>> = std::sync::OnceLock::new();
    let forced = *FORCED.get_or_init(|| {
        let v = std::env::var("DS_VBLANK_LOCK").ok()?;
        match v.strip_prefix('f') {
            Some(n) => n.parse::<u32>().ok().map(|n| (n, true)),
            None => v.parse::<u32>().ok().map(|n| (n, false)),
        }
    });
    if forced.is_some_and(|f| f.0 == 0) {
        return None;
    }
    let s = snapshot()?;
    if VRR_UNTIL.lock().is_some_and(|t| Instant::now() < t) {
        return None;
    }
    let hz = 1.0 / s.period.as_secs_f64();
    // (The measured period must be the mode's: a gap or a stretched refresh is not a lock's.)
    if s.mode_hz > 0 && (hz / s.mode_hz as f64 - 1.0).abs() > 0.02 {
        return None;
    }
    if let Some((n, fine)) = forced {
        return Some((n, if fine { 1 } else { n }, s));
    }
    static FINE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let fine = *FINE.get_or_init(|| std::env::var("DS_VBLANK_FINE").ok().as_deref() != Some("0"));
    let n = (hz / 59.94).round();
    let per = hz / n;
    (n >= 1.0 && n <= 8.0 && (59.94 * 0.999..=60.5).contains(&per)).then_some((n as u32, if fine { 1 } else { n as u32 }, s))
}

fn monitor_hz(device: &str) -> u32 {
    crate::display::current_mode(device).map_or(0, |m| m.hz)
}

fn run() {
    crate::log::set_thread_name("display-clock");
    // (It only waits: woken at once, it sees each vblank on time under the title's load.)
    unsafe {
        use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_TIME_CRITICAL};
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);
    }
    let mut opened: Option<(String, u32, u32)> = None; // device, adapter, source
    let mut failures = 0u32;
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
            crate::info!("display clock: following {device}'s vblanks ({} Hz mode)", monitor_hz(&device));
            opened = Some((device.clone(), oa.h_adapter, oa.vidpn_source_id));
            *STATE.lock() = Some(State { device: device.clone(), snap: None, times: Default::default(), count: 0 });
        }
        let (_, adapter, source) = opened.clone().unwrap();
        let w = WaitForVblank { h_adapter: adapter, h_device: 0, vidpn_source_id: source };
        let r = unsafe { D3DKMTWaitForVerticalBlankEvent(&w) };
        let now = Instant::now();
        if r != 0 {
            // (The adapter changed or was lost: opened again.)
            unsafe { D3DKMTCloseAdapter(&CloseAdapter { h_adapter: adapter }) };
            opened = None;
            *STATE.lock() = None;
            std::thread::sleep(Duration::from_millis(200));
            continue;
        }
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
    format!("; display refreshes between presents ({:.0} Hz, {}): {}", 1.0 / s.period.as_secs_f64(), match lock() {
        Some((n, step, _)) if step < n => format!("locked each, flips {n} apart"),
        Some((n, _, _)) => format!("locked every {n}"),
        None => "timer".into(),
    }, parts.join(" "))
}

/// Status: the clock's display and rate, the lock's state, the presents' cadence.
pub fn status_line() -> String {
    let Some(s) = snapshot() else { return String::new() };
    let lock = lock();
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
        match lock {
            Some((n, step, _)) if step < n => format!("ticks at each of them, flips at least {n} apart"),
            Some((n, _, _)) => format!("ticks at every {n} of them"),
            None if std::env::var("DS_VBLANK_LOCK").ok().as_deref() == Some("0") => "keeps its own timer (DS_VBLANK_LOCK=0)".into(),
            None => "keeps its own timer (the refresh is no whole multiple of ~60 Hz, or varies)".into(),
        },
        if VRR_LOGGED.load(Relaxed) { "; variable refresh or stalls seen" } else { "" },
        cadence
    )
}
