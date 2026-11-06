//! Output window (Win32) running its own message loop thread.
//!
//! Keyboard: key states by virtual-key code (`key_down`), with the left and
//! right Shift, Ctrl and Alt keys apart as well as together.
//!
//! Mouse (`input`): device motion from WM_INPUT (Raw Input: the counts the
//! mouse reports, no pointer acceleration), its buttons and wheel. A click
//! into the client area captures the cursor: hidden and held at the client
//! area's centre while the window is focused, so motion only turns the
//! camera. Leaving the window (Alt+Tab) releases it until the window is
//! focused again; F1 releases it or takes it back. Buttons and motion count
//! only while it is captured (the capturing click is not an attack).
//!
//! Modes: the window is per-monitor DPI aware (its client area in physical
//! pixels, which the presenter fits the frame to) and opens at 1280x720
//! scaled by its monitor's DPI, inside the work area and centred
//! (`DS_WINDOW_SIZE=<w>x<h>`: that client size in pixels). Alt+Enter or F11
//! switches between it and borderless fullscreen on the window's monitor
//! (`DS_FULLSCREEN=1`: fullscreen from the start, `0` windowed; otherwise the SETTINGS
//! screen's PC tab, Window Mode, which Alt+Enter updates); neither reaches the game
//! as a key. `DS_WINDOW_BEHIND=1` (test runs on a shared desktop): shown
//! without activation at the bottom of the z-order, fullscreen too.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicI64, AtomicU32, Ordering};
use windows_sys::Win32::Foundation::{HWND, LPARAM, LRESULT, POINT, RECT, WPARAM};
use windows_sys::Win32::Graphics::Gdi::{ClientToScreen, GetMonitorInfoW, MONITOR_DEFAULTTONEAREST, MONITORINFO, MonitorFromWindow};
use windows_sys::Win32::UI::HiDpi::{AdjustWindowRectExForDpi, DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2, GetDpiForWindow, SetThreadDpiAwarenessContext};
use windows_sys::Win32::System::LibraryLoader::GetModuleHandleW;
use windows_sys::Win32::UI::WindowsAndMessaging::*;

pub struct Window {
    pub hwnd: isize,
    pub hinstance: isize,
}

static WINDOW: OnceLock<Window> = OnceLock::new();
pub static CLOSED: AtomicBool = AtomicBool::new(false);
pub static RESIZED: AtomicBool = AtomicBool::new(false);
pub static CLIENT_W: AtomicU32 = AtomicU32::new(1280);
pub static CLIENT_H: AtomicU32 = AtomicU32::new(720);
/// Virtual-key states (true = down).
pub static KEYS: [AtomicBool; 256] = [const { AtomicBool::new(false) }; 256];

/// Mouse motion not yet taken (`input`), in device counts.
pub static MOUSE_DX: AtomicI64 = AtomicI64::new(0);
pub static MOUSE_DY: AtomicI64 = AtomicI64::new(0);
/// Wheel movement not yet taken, in WHEEL_DELTA (120) units per notch.
pub static MOUSE_WHEEL: AtomicI32 = AtomicI32::new(0);
/// Mouse buttons down: bit 0 left, 1 right, 2 middle, 3 and 4 the side
/// buttons (X1, X2).
pub static MOUSE_BUTTONS: AtomicU32 = AtomicU32::new(0);
/// The cursor is captured (hidden and held in the client area).
pub static CAPTURED: AtomicBool = AtomicBool::new(false);
static FOCUSED: AtomicBool = AtomicBool::new(false);
/// Whether a focused window captures the cursor: from the first click into
/// the client area on (not at launch: the window appears while the title
/// boots); F1 toggles.
static CAPTURE_WANTED: AtomicBool = AtomicBool::new(false);
/// Diagnosis (control channel `capture`): input as if captured without a
/// focused window (headless runs, injected motion).
pub static CAPTURE_TEST: AtomicBool = AtomicBool::new(false);
/// The last absolute position of a mouse reporting absolute coordinates
/// (remote sessions, tablets; i64::MIN: none yet).
static ABS_X: AtomicI64 = AtomicI64::new(i64::MIN);
static ABS_Y: AtomicI64 = AtomicI64::new(i64::MIN);
/// Borderless fullscreen (`apply_fullscreen`).
pub static FULLSCREEN: AtomicBool = AtomicBool::new(false);
/// The windowed placement fullscreen returns to.
static PLACEMENT: parking_lot::Mutex<Option<WINDOWPLACEMENT>> = parking_lot::Mutex::new(None);
/// Asks the UI thread for a mode (wParam 0 windowed, 1 fullscreen, 2 the
/// other one).
const WM_DS_MODE: u32 = WM_APP + 1;
/// `DS_WINDOW_BEHIND=1`: never activated or raised (test runs).
static BEHIND: AtomicBool = AtomicBool::new(false);
/// Asks the UI thread for a client size (wParam width, lParam height).
const WM_DS_SIZE: u32 = WM_APP + 2;

pub fn get() -> Option<&'static Window> {
    WINDOW.get()
}

pub fn key_down(vk: u8) -> bool {
    KEYS[vk as usize].load(Ordering::Relaxed)
}

/// The left or right key of a Shift, Ctrl or Alt key message (VK_LSHIFT ..
/// VK_RMENU): Shift by its scan code, Ctrl and Alt by the extended-key bit.
fn sided(wp: WPARAM, lp: LPARAM) -> Option<u8> {
    let extended = (lp >> 24) & 1 != 0;
    match wp as u8 {
        0x10 => Some(if (lp >> 16) & 0xFF == 0x36 { 0xA1 } else { 0xA0 }),
        0x11 => Some(if extended { 0xA3 } else { 0xA2 }),
        0x12 => Some(if extended { 0xA5 } else { 0xA4 }),
        _ => None,
    }
}

fn set_key(wp: WPARAM, lp: LPARAM, down: bool) {
    KEYS[(wp & 0xFF) as usize].store(down, Ordering::Relaxed);
    if let Some(k) = sided(wp, lp) {
        KEYS[k as usize].store(down, Ordering::Relaxed);
        // The common key is down while either side is.
        let (l, r) = match wp as u8 {
            0x10 => (0xA0, 0xA1),
            0x11 => (0xA2, 0xA3),
            _ => (0xA4, 0xA5),
        };
        KEYS[(wp & 0xFF) as usize].store(key_down(l) || key_down(r), Ordering::Relaxed);
    }
}

/// Captures the cursor while the window is focused and capture is wanted
/// (held at the client area's centre: Raw Input keeps reporting motion),
/// else releases it.
unsafe fn update_capture(hwnd: HWND) {
    if FOCUSED.load(Ordering::Relaxed) && CAPTURE_WANTED.load(Ordering::Relaxed) {
        let mut r: RECT = unsafe { std::mem::zeroed() };
        unsafe { GetClientRect(hwnd, &mut r) };
        let mut p = POINT { x: (r.left + r.right) / 2, y: (r.top + r.bottom) / 2 };
        unsafe { ClientToScreen(hwnd, &mut p) };
        let c = RECT { left: p.x, top: p.y, right: p.x + 1, bottom: p.y + 1 };
        unsafe { ClipCursor(&c) };
        if !CAPTURED.swap(true, Ordering::SeqCst) {
            unsafe { SetCursor(std::ptr::null_mut()) };
            MOUSE_DX.store(0, Ordering::Relaxed);
            MOUSE_DY.store(0, Ordering::Relaxed);
            ABS_X.store(i64::MIN, Ordering::Relaxed);
        }
    } else if CAPTURED.swap(false, Ordering::SeqCst) {
        unsafe {
            ClipCursor(std::ptr::null());
            SetCursor(LoadCursorW(std::ptr::null_mut(), IDC_ARROW));
        }
        MOUSE_BUTTONS.store(0, Ordering::Relaxed);
        MOUSE_WHEEL.store(0, Ordering::Relaxed);
    }
}

/// Asks the window for borderless fullscreen (`Some(true)`), windowed
/// (`Some(false)`) or the other one (`None`); from any thread.
pub fn set_fullscreen(on: Option<bool>) {
    if let Some(w) = WINDOW.get() {
        let mode = match on {
            Some(false) => 0,
            Some(true) => 1,
            None => 2,
        };
        unsafe { PostMessageW(w.hwnd as HWND, WM_DS_MODE, mode, 0) };
    }
}

/// Asks the windowed window for a client size in pixels (ignored in
/// fullscreen); from any thread.
pub fn set_client_size(w: u32, h: u32) {
    if let Some(win) = WINDOW.get() {
        unsafe { PostMessageW(win.hwnd as HWND, WM_DS_SIZE, w as WPARAM, h as LPARAM) };
    }
}

/// UI thread: borderless fullscreen on the window's monitor, or back to the
/// windowed placement it had.
unsafe fn apply_fullscreen(hwnd: HWND, on: bool) {
    if on == FULLSCREEN.load(Ordering::Relaxed) {
        return;
    }
    unsafe {
        if on {
            let mut wp: WINDOWPLACEMENT = std::mem::zeroed();
            wp.length = std::mem::size_of::<WINDOWPLACEMENT>() as u32;
            let mut mi: MONITORINFO = std::mem::zeroed();
            mi.cbSize = std::mem::size_of::<MONITORINFO>() as u32;
            if GetWindowPlacement(hwnd, &mut wp) == 0 || GetMonitorInfoW(MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST), &mut mi) == 0 {
                crate::warn!("window: no fullscreen ({})", std::io::Error::last_os_error());
                return;
            }
            *PLACEMENT.lock() = Some(wp);
            FULLSCREEN.store(true, Ordering::Relaxed);
            SetWindowLongPtrW(hwnd, GWL_STYLE, (WS_POPUP | WS_VISIBLE) as isize);
            let r = mi.rcMonitor;
            let behind = BEHIND.load(Ordering::Relaxed);
            let (after, quiet) = if behind { (HWND_BOTTOM, SWP_NOACTIVATE) } else { (HWND_TOP, 0) };
            SetWindowPos(hwnd, after, r.left, r.top, r.right - r.left, r.bottom - r.top, SWP_NOOWNERZORDER | SWP_FRAMECHANGED | quiet);
            crate::info!("window: borderless fullscreen {}x{}", r.right - r.left, r.bottom - r.top);
        } else {
            FULLSCREEN.store(false, Ordering::Relaxed);
            SetWindowLongPtrW(hwnd, GWL_STYLE, (WS_OVERLAPPEDWINDOW | WS_VISIBLE) as isize);
            let saved = PLACEMENT.lock().take();
            if let Some(mut wp) = saved {
                if BEHIND.load(Ordering::Relaxed) && wp.showCmd == SW_SHOWNORMAL as u32 {
                    wp.showCmd = SW_SHOWNOACTIVATE as u32;
                }
                SetWindowPlacement(hwnd, &wp);
            }
            SetWindowPos(hwnd, std::ptr::null_mut(), 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOOWNERZORDER | SWP_FRAMECHANGED | SWP_NOACTIVATE);
            crate::info!("window: windowed");
        }
    }
}

/// WM_INPUT: mouse motion, buttons and wheel while the cursor is captured.
unsafe fn raw_input(lp: LPARAM) {
    use windows_sys::Win32::UI::Input::*;
    let mut raw: RAWINPUT = unsafe { std::mem::zeroed() };
    let mut size = std::mem::size_of::<RAWINPUT>() as u32;
    let n = unsafe { GetRawInputData(lp as HRAWINPUT, RID_INPUT, &mut raw as *mut RAWINPUT as *mut _, &mut size, std::mem::size_of::<RAWINPUTHEADER>() as u32) };
    if n == 0 || n == u32::MAX || raw.header.dwType != RIM_TYPEMOUSE || !CAPTURED.load(Ordering::Relaxed) {
        return;
    }
    let m = unsafe { raw.data.mouse };
    // Keyboard and mouse in use (the button prompts): motion past a few counts, a button, the wheel.
    if m.lLastX.abs() + m.lLastY.abs() > 3 && m.usFlags & MOUSE_MOVE_ABSOLUTE == 0 || unsafe { m.Anonymous.Anonymous.usButtonFlags } != 0 {
        crate::input::note_device(crate::input::DEVICE_KBM);
    }
    if m.usFlags & MOUSE_MOVE_ABSOLUTE == 0 {
        MOUSE_DX.fetch_add(m.lLastX as i64, Ordering::Relaxed);
        MOUSE_DY.fetch_add(m.lLastY as i64, Ordering::Relaxed);
    } else {
        // 0..65535 over the (virtual) desktop: motion in pixels.
        let virt = m.usFlags & MOUSE_VIRTUAL_DESKTOP != 0;
        let (w, h) = unsafe {
            if virt {
                (GetSystemMetrics(SM_CXVIRTUALSCREEN), GetSystemMetrics(SM_CYVIRTUALSCREEN))
            } else {
                (GetSystemMetrics(SM_CXSCREEN), GetSystemMetrics(SM_CYSCREEN))
            }
        };
        let x = m.lLastX as i64 * w as i64 / 65535;
        let y = m.lLastY as i64 * h as i64 / 65535;
        let (px, py) = (ABS_X.swap(x, Ordering::Relaxed), ABS_Y.swap(y, Ordering::Relaxed));
        if px != i64::MIN {
            MOUSE_DX.fetch_add(x - px, Ordering::Relaxed);
            MOUSE_DY.fetch_add(y - py, Ordering::Relaxed);
        }
    }
    let f = unsafe { m.Anonymous.Anonymous.usButtonFlags } as u32;
    for (down, up, bit) in [(0x1, 0x2, 0), (0x4, 0x8, 1), (0x10, 0x20, 2), (0x40, 0x80, 3), (0x100, 0x200, 4)] {
        if f & down != 0 {
            MOUSE_BUTTONS.fetch_or(1 << bit, Ordering::Relaxed);
        }
        if f & up != 0 {
            MOUSE_BUTTONS.fetch_and(!(1 << bit), Ordering::Relaxed);
        }
    }
    if f & 0x400 != 0 {
        MOUSE_WHEEL.fetch_add(unsafe { m.Anonymous.Anonymous.usButtonData } as i16 as i32, Ordering::Relaxed);
    }
}

unsafe extern "system" fn wndproc(hwnd: HWND, msg: u32, wp: WPARAM, lp: LPARAM) -> LRESULT {
    match msg {
        WM_CLOSE => {
            CLOSED.store(true, Ordering::SeqCst);
            0
        }
        WM_SIZE => {
            let w = (lp & 0xFFFF) as u32;
            let h = ((lp >> 16) & 0xFFFF) as u32;
            if w > 0 && h > 0 {
                CLIENT_W.store(w, Ordering::Relaxed);
                CLIENT_H.store(h, Ordering::Relaxed);
                RESIZED.store(true, Ordering::SeqCst);
            }
            if wp as u32 == SIZE_MINIMIZED {
                FOCUSED.store(false, Ordering::Relaxed);
            }
            unsafe { update_capture(hwnd) };
            0
        }
        WM_MOVE => {
            unsafe { update_capture(hwnd) };
            0
        }
        WM_ACTIVATE => {
            // Active and not minimized.
            FOCUSED.store((wp & 0xFFFF) as u32 != WA_INACTIVE && (wp >> 16) & 0xFFFF == 0, Ordering::Relaxed);
            unsafe { update_capture(hwnd) };
            unsafe { DefWindowProcW(hwnd, msg, wp, lp) }
        }
        WM_SETCURSOR if CAPTURED.load(Ordering::Relaxed) && (lp & 0xFFFF) as u32 == HTCLIENT => {
            unsafe { SetCursor(std::ptr::null_mut()) };
            1
        }
        WM_LBUTTONDOWN | WM_RBUTTONDOWN | WM_MBUTTONDOWN if !CAPTURED.load(Ordering::Relaxed) => {
            // A click into the client area takes the cursor again.
            CAPTURE_WANTED.store(true, Ordering::Relaxed);
            FOCUSED.store(true, Ordering::Relaxed);
            unsafe { update_capture(hwnd) };
            0
        }
        WM_INPUT => {
            unsafe { raw_input(lp) };
            unsafe { DefWindowProcW(hwnd, msg, wp, lp) }
        }
        WM_KEYDOWN | WM_SYSKEYDOWN => {
            if wp as u16 == VK_F4_CODE && msg == WM_SYSKEYDOWN {
                // Alt+F4 closes the window.
                return unsafe { DefWindowProcW(hwnd, msg, wp, lp) };
            }
            if (wp as u16 == VK_RETURN_CODE && msg == WM_SYSKEYDOWN) || wp as u16 == VK_F11_CODE {
                // Alt+Enter, F11: windowed / borderless fullscreen.
                if (lp >> 30) & 1 == 0 {
                    let on = !FULLSCREEN.load(Ordering::Relaxed);
                    unsafe { apply_fullscreen(hwnd, on) };
                    // The PC tab's Window Mode follows (its menu shows the mode in use).
                    let _ = crate::pcsettings::store::set_by_name("Setting_PC_WindowMode", if on { "1" } else { "0" });
                }
                return 0;
            }
            if wp as u16 == VK_F1_CODE && (lp >> 30) & 1 == 0 {
                CAPTURE_WANTED.fetch_xor(true, Ordering::Relaxed);
                unsafe { update_capture(hwnd) };
            }
            set_key(wp, lp, true);
            crate::input::note_device(crate::input::DEVICE_KBM);
            0
        }
        WM_KEYUP | WM_SYSKEYUP => {
            set_key(wp, lp, false);
            0
        }
        // Alt with a key (Alt is walk): no menu mnemonic and its beep; Alt+Space
        // still opens the window menu.
        WM_SYSCHAR if wp != b' ' as WPARAM => 0,
        WM_DS_MODE => {
            let on = match wp {
                0 => false,
                1 => true,
                _ => !FULLSCREEN.load(Ordering::Relaxed),
            };
            unsafe { apply_fullscreen(hwnd, on) };
            0
        }
        WM_DS_SIZE if !FULLSCREEN.load(Ordering::Relaxed) => {
            let mut r = RECT { left: 0, top: 0, right: wp as i32, bottom: lp as i32 };
            unsafe {
                AdjustWindowRectExForDpi(&mut r, WS_OVERLAPPEDWINDOW, 0, 0, GetDpiForWindow(hwnd).max(96));
                SetWindowPos(hwnd, std::ptr::null_mut(), 0, 0, r.right - r.left, r.bottom - r.top, SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE);
            }
            crate::info!("window: client size {wp}x{lp} asked");
            0
        }
        WM_DPICHANGED => {
            // Moved to a monitor of another DPI: the size Windows suggests
            // (fullscreen keeps the monitor's).
            if !FULLSCREEN.load(Ordering::Relaxed) {
                let r = unsafe { &*(lp as *const RECT) };
                unsafe { SetWindowPos(hwnd, std::ptr::null_mut(), r.left, r.top, r.right - r.left, r.bottom - r.top, SWP_NOZORDER | SWP_NOACTIVATE) };
            }
            0
        }
        WM_KILLFOCUS => {
            for k in KEYS.iter() {
                k.store(false, Ordering::Relaxed);
            }
            FOCUSED.store(false, Ordering::Relaxed);
            unsafe { update_capture(hwnd) };
            0
        }
        _ => unsafe { DefWindowProcW(hwnd, msg, wp, lp) },
    }
}

const VK_F1_CODE: u16 = 0x70;
const VK_F4_CODE: u16 = 0x73;
const VK_F11_CODE: u16 = 0x7A;
const VK_RETURN_CODE: u16 = 0x0D;

/// `DS_WINDOW_SIZE=<w>x<h>`: the client size asked for (pixels).
fn size_override() -> Option<(i32, i32)> {
    let v = std::env::var("DS_WINDOW_SIZE").ok()?;
    let size = v.trim().split_once(['x', 'X']).and_then(|(w, h)| Some((w.trim().parse::<i32>().ok()?, h.trim().parse::<i32>().ok()?)));
    match size {
        Some((w, h)) if (160..=16384).contains(&w) && (90..=16384).contains(&h) => Some((w, h)),
        _ => {
            crate::warn!("DS_WINDOW_SIZE={v}: not a size (WxH); the default is used");
            None
        }
    }
}

/// UI thread, before the window is shown: its client area at `width`x`height`
/// scaled by the monitor's DPI (or `DS_WINDOW_SIZE`), at most the work area
/// (aspect kept), centred in it.
unsafe fn place_initial(hwnd: HWND, style: WINDOW_STYLE, width: u32, height: u32) {
    unsafe {
        let dpi = GetDpiForWindow(hwnd).max(96);
        let mut mi: MONITORINFO = std::mem::zeroed();
        mi.cbSize = std::mem::size_of::<MONITORINFO>() as u32;
        let have_area = GetMonitorInfoW(MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST), &mut mi) != 0;
        let area = mi.rcWork;
        let frame = |w: i32, h: i32| {
            let mut r = RECT { left: 0, top: 0, right: w, bottom: h };
            AdjustWindowRectExForDpi(&mut r, style, 0, 0, dpi);
            (r.right - r.left, r.bottom - r.top)
        };
        let asked = size_override();
        let (mut w, mut h) = asked.unwrap_or(((width * dpi / 96) as i32, (height * dpi / 96) as i32));
        if have_area && asked.is_none() {
            // Shrink (aspect kept) until the framed window fits the work area.
            let (fw, fh) = frame(w, h);
            let (aw, ah) = (area.right - area.left, area.bottom - area.top);
            if fw > aw || fh > ah {
                let s = ((aw - (fw - w)) as f64 / w as f64).min((ah - (fh - h)) as f64 / h as f64);
                (w, h) = (((w as f64 * s) as i32).max(160), ((h as f64 * s) as i32).max(90));
            }
        }
        let (fw, fh) = frame(w, h);
        let (x, y) = if have_area { (area.left + ((area.right - area.left - fw) / 2).max(0), area.top + ((area.bottom - area.top - fh) / 2).max(0)) } else { (0, 0) };
        SetWindowPos(hwnd, std::ptr::null_mut(), x, y, fw, fh, SWP_NOZORDER | SWP_NOACTIVATE);
        let mut c: RECT = std::mem::zeroed();
        GetClientRect(hwnd, &mut c);
        CLIENT_W.store((c.right - c.left).max(1) as u32, Ordering::Relaxed);
        CLIENT_H.store((c.bottom - c.top).max(1) as u32, Ordering::Relaxed);
        crate::info!("window: client {}x{} at DPI {dpi}", c.right - c.left, c.bottom - c.top);
    }
}

fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

/// Creates the window on a dedicated thread and waits until it exists.
pub fn create(title: &str, width: u32, height: u32) -> Result<&'static Window, String> {
    if let Some(w) = WINDOW.get() {
        return Ok(w);
    }
    let (tx, rx) = std::sync::mpsc::channel::<Result<(isize, isize), String>>();
    let title = title.to_string();
    std::thread::Builder::new()
        .name("ui".into())
        .spawn(move || unsafe {
            crate::log::set_thread_name("ui");
            let hinst = GetModuleHandleW(std::ptr::null());
            let class = wide("dsport_window");
            let wc = WNDCLASSEXW {
                cbSize: std::mem::size_of::<WNDCLASSEXW>() as u32,
                style: CS_HREDRAW | CS_VREDRAW,
                lpfnWndProc: Some(wndproc),
                cbClsExtra: 0,
                cbWndExtra: 0,
                hInstance: hinst,
                hIcon: std::ptr::null_mut(),
                hCursor: LoadCursorW(std::ptr::null_mut(), IDC_ARROW),
                hbrBackground: std::ptr::null_mut(),
                lpszMenuName: std::ptr::null(),
                lpszClassName: class.as_ptr(),
                hIconSm: std::ptr::null_mut(),
            };
            RegisterClassExW(&wc);
            // Per-monitor DPI awareness for this thread's windows only (the
            // text entry dialog keeps the system's scaling).
            SetThreadDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
            let style = WS_OVERLAPPEDWINDOW;
            let mut rect = windows_sys::Win32::Foundation::RECT { left: 0, top: 0, right: width as i32, bottom: height as i32 };
            AdjustWindowRect(&mut rect, style, 0);
            let t = wide(&title);
            let hwnd = CreateWindowExW(
                0,
                class.as_ptr(),
                t.as_ptr(),
                style,
                CW_USEDEFAULT,
                CW_USEDEFAULT,
                rect.right - rect.left,
                rect.bottom - rect.top,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                hinst,
                std::ptr::null(),
            );
            if hwnd.is_null() {
                let _ = tx.send(Err("CreateWindowExW failed".into()));
                return;
            }
            CLIENT_W.store(width, Ordering::Relaxed);
            CLIENT_H.store(height, Ordering::Relaxed);
            place_initial(hwnd, style, width, height);
            // The mouse as a Raw Input device of this window (motion
            // without pointer acceleration; only while it is in front).
            let dev = windows_sys::Win32::UI::Input::RAWINPUTDEVICE { usUsagePage: 0x01, usUsage: 0x02, dwFlags: 0, hwndTarget: hwnd };
            if windows_sys::Win32::UI::Input::RegisterRawInputDevices(&dev, 1, std::mem::size_of::<windows_sys::Win32::UI::Input::RAWINPUTDEVICE>() as u32) == 0 {
                crate::warn!("window: no raw mouse input ({}): no mouse look", std::io::Error::last_os_error());
            }
            if std::env::var("DS_WINDOW_BEHIND").is_ok_and(|v| v.trim() == "1") {
                BEHIND.store(true, Ordering::Relaxed);
                ShowWindow(hwnd, SW_SHOWNOACTIVATE);
                SetWindowPos(hwnd, HWND_BOTTOM, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE);
            } else {
                ShowWindow(hwnd, SW_SHOW);
            }
            // `DS_FULLSCREEN` (1 or 0) over the PC tab's Window Mode.
            let fullscreen = match std::env::var("DS_FULLSCREEN") {
                Ok(v) => v.trim() == "1",
                Err(_) => crate::pcsettings::window_fullscreen().unwrap_or(false),
            };
            if fullscreen {
                apply_fullscreen(hwnd, true);
            }
            let _ = tx.send(Ok((hwnd as isize, hinst as isize)));
            let mut msg: MSG = std::mem::zeroed();
            while GetMessageW(&mut msg, std::ptr::null_mut(), 0, 0) > 0 {
                TranslateMessage(&msg);
                DispatchMessageW(&msg);
            }
        })
        .map_err(|e| format!("ui thread: {e}"))?;
    let (hwnd, hinstance) = rx.recv().map_err(|e| format!("ui thread died: {e}"))??;
    WINDOW.set(Window { hwnd, hinstance }).ok();
    Ok(WINDOW.get().unwrap())
}
