//! Monitors and display modes: the PC tab's Monitor, Resolution and Refresh Rate, exclusive
//! fullscreen's mode changes (`gpu::vk::window`, `gpu::vk::present`) and each monitor's HDR
//! state (Windows' advanced color).
//!
//! The choices are kept in display.txt beside pcsettings.txt: `monitor=<GDI device name>`
//! (empty: the primary monitor), `resolution=<w>x<h>` (empty: the desktop's) and
//! `refresh=<Hz>` (0: the desktop's rate, or the highest one at another resolution). The
//! menu's lists are made once, at the start, from the monitors then and the chosen one's modes.

use parking_lot::Mutex;
use std::path::PathBuf;
use std::sync::OnceLock;
use windows_sys::Win32::Devices::Display::*;
use windows_sys::Win32::Foundation::{LPARAM, RECT};
use windows_sys::Win32::Graphics::Gdi::*;

/// A monitor as Windows lists it now.
#[derive(Clone, Debug, Default)]
pub struct Monitor {
    /// The GDI device name (`\\.\DISPLAY1`).
    pub device: String,
    /// The monitor's own name (its EDID's), else the device name.
    pub name: String,
    /// Its area on the desktop: left, top, right, bottom.
    pub rect: [i32; 4],
    pub primary: bool,
    /// Windows' HDR (advanced color) supported and turned on for it.
    pub hdr_supported: bool,
    pub hdr_on: bool,
    /// The brightness Windows gives SDR content on it in HDR mode, in nits (80 by default).
    pub sdr_white_nits: f32,
}

/// A display mode (32 bits per pixel, not interlaced).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Mode {
    pub w: u32,
    pub h: u32,
    pub hz: u32,
}

#[derive(Default)]
struct Choice {
    monitor: String,
    res: Option<(u32, u32)>,
    hz: u32,
}

static CHOICE: Mutex<Option<Choice>> = Mutex::new(None);
static PATH: Mutex<Option<PathBuf>> = Mutex::new(None);
/// The mode set for exclusive fullscreen and its device, to put back (`restore`).
static CHANGED: Mutex<Option<String>> = Mutex::new(None);

/// The menu's options, made at the start.
struct Lists {
    monitors: Vec<Monitor>,
    resolutions: Vec<(u32, u32)>,
    rates: Vec<u32>,
}

static LISTS: OnceLock<Lists> = OnceLock::new();

fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

fn from_wide(w: &[u16]) -> String {
    let n = w.iter().position(|&c| c == 0).unwrap_or(w.len());
    String::from_utf16_lossy(&w[..n])
}

/// Per GDI device name: the monitor's own name and its HDR state (DisplayConfig).
fn targets() -> Vec<(String, String, bool, bool, f32)> {
    let mut out = Vec::new();
    unsafe {
        let (mut np, mut nm) = (0u32, 0u32);
        if GetDisplayConfigBufferSizes(QDC_ONLY_ACTIVE_PATHS, &mut np, &mut nm) != 0 {
            return out;
        }
        let mut paths: Vec<DISPLAYCONFIG_PATH_INFO> = vec![std::mem::zeroed(); np as usize];
        let mut modes: Vec<DISPLAYCONFIG_MODE_INFO> = vec![std::mem::zeroed(); nm as usize];
        if QueryDisplayConfig(QDC_ONLY_ACTIVE_PATHS, &mut np, paths.as_mut_ptr(), &mut nm, modes.as_mut_ptr(), std::ptr::null_mut()) != 0 {
            return out;
        }
        paths.truncate(np as usize);
        for p in &paths {
            let mut src: DISPLAYCONFIG_SOURCE_DEVICE_NAME = std::mem::zeroed();
            src.header.r#type = DISPLAYCONFIG_DEVICE_INFO_GET_SOURCE_NAME;
            src.header.size = std::mem::size_of::<DISPLAYCONFIG_SOURCE_DEVICE_NAME>() as u32;
            src.header.adapterId = p.sourceInfo.adapterId;
            src.header.id = p.sourceInfo.id;
            if DisplayConfigGetDeviceInfo(&mut src.header) != 0 {
                continue;
            }
            let device = from_wide(&src.viewGdiDeviceName);
            let mut tgt: DISPLAYCONFIG_TARGET_DEVICE_NAME = std::mem::zeroed();
            tgt.header.r#type = DISPLAYCONFIG_DEVICE_INFO_GET_TARGET_NAME;
            tgt.header.size = std::mem::size_of::<DISPLAYCONFIG_TARGET_DEVICE_NAME>() as u32;
            tgt.header.adapterId = p.targetInfo.adapterId;
            tgt.header.id = p.targetInfo.id;
            let name = if DisplayConfigGetDeviceInfo(&mut tgt.header) == 0 { from_wide(&tgt.monitorFriendlyDeviceName) } else { String::new() };
            let mut aci: DISPLAYCONFIG_GET_ADVANCED_COLOR_INFO = std::mem::zeroed();
            aci.header.r#type = DISPLAYCONFIG_DEVICE_INFO_GET_ADVANCED_COLOR_INFO;
            aci.header.size = std::mem::size_of::<DISPLAYCONFIG_GET_ADVANCED_COLOR_INFO>() as u32;
            aci.header.adapterId = p.targetInfo.adapterId;
            aci.header.id = p.targetInfo.id;
            let (supported, on) = if DisplayConfigGetDeviceInfo(&mut aci.header) == 0 {
                let v = aci.Anonymous.value;
                // Bit 0 advanced color supported, bit 1 enabled, bit 3 force-disabled.
                (v & 1 != 0, v & 2 != 0 && v & 8 == 0)
            } else {
                (false, false)
            };
            let mut white: DISPLAYCONFIG_SDR_WHITE_LEVEL = std::mem::zeroed();
            white.header.r#type = DISPLAYCONFIG_DEVICE_INFO_GET_SDR_WHITE_LEVEL;
            white.header.size = std::mem::size_of::<DISPLAYCONFIG_SDR_WHITE_LEVEL>() as u32;
            white.header.adapterId = p.targetInfo.adapterId;
            white.header.id = p.targetInfo.id;
            // In thousandths of 80 nits.
            let nits = if DisplayConfigGetDeviceInfo(&mut white.header) == 0 && white.SDRWhiteLevel > 0 { white.SDRWhiteLevel as f32 / 1000.0 * 80.0 } else { 80.0 };
            out.push((device, name, supported, on, nits));
        }
    }
    out
}

unsafe extern "system" fn enum_monitor(h: HMONITOR, _dc: HDC, _r: *mut RECT, data: LPARAM) -> windows_sys::core::BOOL {
    let list = unsafe { &mut *(data as *mut Vec<Monitor>) };
    let mut mi: MONITORINFOEXW = unsafe { std::mem::zeroed() };
    mi.monitorInfo.cbSize = std::mem::size_of::<MONITORINFOEXW>() as u32;
    if unsafe { GetMonitorInfoW(h, &mut mi as *mut MONITORINFOEXW as *mut MONITORINFO) } != 0 {
        let r = mi.monitorInfo.rcMonitor;
        let device = from_wide(&mi.szDevice);
        list.push(Monitor {
            name: device.clone(),
            device,
            rect: [r.left, r.top, r.right, r.bottom],
            primary: mi.monitorInfo.dwFlags & windows_sys::Win32::UI::WindowsAndMessaging::MONITORINFOF_PRIMARY != 0,
            sdr_white_nits: 80.0,
            ..Default::default()
        });
    }
    1
}

/// The monitors Windows lists now, the primary first, then left to right.
pub fn monitors() -> Vec<Monitor> {
    let mut list: Vec<Monitor> = Vec::new();
    unsafe { EnumDisplayMonitors(std::ptr::null_mut(), std::ptr::null(), Some(enum_monitor), &mut list as *mut Vec<Monitor> as LPARAM) };
    let t = targets();
    for m in list.iter_mut() {
        if let Some((_, name, sup, on, nits)) = t.iter().find(|x| x.0.eq_ignore_ascii_case(&m.device)) {
            if !name.is_empty() {
                m.name = name.clone();
            }
            (m.hdr_supported, m.hdr_on, m.sdr_white_nits) = (*sup, *on, *nits);
        }
    }
    list.sort_by_key(|m| (!m.primary, m.rect[0], m.rect[1]));
    list
}

fn devmode() -> DEVMODEW {
    let mut dm: DEVMODEW = unsafe { std::mem::zeroed() };
    dm.dmSize = std::mem::size_of::<DEVMODEW>() as u16;
    dm
}

/// A monitor's modes (32 bits per pixel, progressive), largest and fastest first.
pub fn modes(device: &str) -> Vec<Mode> {
    let dev = wide(device);
    let mut out = Vec::new();
    let mut i = 0u32;
    loop {
        let mut dm = devmode();
        if unsafe { EnumDisplaySettingsExW(dev.as_ptr(), i, &mut dm, 0) } == 0 {
            break;
        }
        i += 1;
        let interlaced = unsafe { dm.Anonymous2.dmDisplayFlags } & DM_INTERLACED != 0;
        if dm.dmBitsPerPel == 32 && !interlaced && dm.dmPelsWidth >= 640 && dm.dmPelsHeight >= 480 && dm.dmDisplayFrequency > 1 {
            out.push(Mode { w: dm.dmPelsWidth, h: dm.dmPelsHeight, hz: dm.dmDisplayFrequency });
        }
        if i > 4096 {
            break;
        }
    }
    out.sort_by(|a, b| (b.w * b.h, b.w, b.hz).cmp(&(a.w * a.h, a.w, a.hz)));
    out.dedup();
    out
}

/// A monitor's mode now.
pub fn current_mode(device: &str) -> Option<Mode> {
    let dev = wide(device);
    let mut dm = devmode();
    (unsafe { EnumDisplaySettingsExW(dev.as_ptr(), ENUM_CURRENT_SETTINGS, &mut dm, 0) } != 0).then_some(Mode { w: dm.dmPelsWidth, h: dm.dmPelsHeight, hz: dm.dmDisplayFrequency })
}

/// Its mode in the registry (what Windows puts back when the port's mode ends).
fn registry_mode(device: &str) -> Option<Mode> {
    let dev = wide(device);
    let mut dm = devmode();
    (unsafe { EnumDisplaySettingsExW(dev.as_ptr(), ENUM_REGISTRY_SETTINGS, &mut dm, 0) } != 0).then_some(Mode { w: dm.dmPelsWidth, h: dm.dmPelsHeight, hz: dm.dmDisplayFrequency })
}

/// The chosen monitor (display.txt), else the primary one; None without any monitor.
pub fn target() -> Option<Monitor> {
    let list = monitors();
    let want = CHOICE.lock().as_ref().map(|c| c.monitor.clone()).unwrap_or_default();
    list.iter().find(|m| !want.is_empty() && m.device.eq_ignore_ascii_case(&want)).or_else(|| list.iter().find(|m| m.primary)).or(list.first()).cloned()
}

/// The mode exclusive fullscreen takes on `device`: the chosen resolution and rate matched to
/// its modes (the nearest rate at that resolution); None keeps the desktop's mode.
pub fn wanted_mode(device: &str) -> Option<Mode> {
    let (res, hz) = CHOICE.lock().as_ref().map(|c| (c.res, c.hz)).unwrap_or((None, 0));
    let desk = registry_mode(device).or_else(|| current_mode(device))?;
    let (w, h) = res.unwrap_or((desk.w, desk.h));
    let at: Vec<Mode> = modes(device).into_iter().filter(|m| m.w == w && m.h == h).collect();
    if at.is_empty() {
        crate::warn!("display: {w}x{h} is not a mode of {device}: the desktop's mode stays");
        return None;
    }
    let rate = if hz != 0 {
        at.iter().min_by_key(|m| (m.hz as i64 - hz as i64).abs()).map(|m| m.hz)
    } else if (w, h) == (desk.w, desk.h) {
        Some(desk.hz)
    } else {
        at.iter().map(|m| m.hz).max()
    }?;
    let m = Mode { w, h, hz: rate };
    (m != desk).then_some(m)
}

/// Sets `device` to `mode` for exclusive fullscreen (until `restore`); false when Windows
/// refuses it.
pub fn set_mode(device: &str, mode: Mode) -> bool {
    if current_mode(device) == Some(mode) {
        return true;
    }
    let dev = wide(device);
    let mut dm = devmode();
    dm.dmPelsWidth = mode.w;
    dm.dmPelsHeight = mode.h;
    dm.dmDisplayFrequency = mode.hz;
    dm.dmBitsPerPel = 32;
    dm.dmFields = DM_PELSWIDTH | DM_PELSHEIGHT | DM_DISPLAYFREQUENCY | DM_BITSPERPEL;
    let r = unsafe { ChangeDisplaySettingsExW(dev.as_ptr(), &dm, std::ptr::null_mut(), CDS_FULLSCREEN, std::ptr::null()) };
    if r == DISP_CHANGE_SUCCESSFUL {
        *CHANGED.lock() = Some(device.to_string());
        crate::info!("display: {device} set to {}x{} at {} Hz", mode.w, mode.h, mode.hz);
        true
    } else {
        crate::warn!("display: {device} refused {}x{} at {} Hz (code {r})", mode.w, mode.h, mode.hz);
        false
    }
}

/// Puts back the desktop's mode on the monitor `set_mode` changed (exclusive fullscreen ends,
/// the game loses focus or quits); from any thread.
pub fn restore() {
    let Some(device) = CHANGED.lock().take() else { return };
    let dev = wide(&device);
    let r = unsafe { ChangeDisplaySettingsExW(dev.as_ptr(), std::ptr::null(), std::ptr::null_mut(), 0, std::ptr::null()) };
    crate::info!("display: {device} back to its desktop mode{}", if r == DISP_CHANGE_SUCCESSFUL { String::new() } else { format!(" (code {r})") });
}

/// `restore` on the way out of the process (a quit or a fatal error): waits at most 200 ms for
/// the lock (a thread that died holding it must not keep the process alive).
pub fn restore_on_exit() {
    let Some(mut g) = CHANGED.try_lock_for(std::time::Duration::from_millis(200)) else { return };
    let Some(device) = g.take() else { return };
    drop(g);
    let dev = wide(&device);
    unsafe { ChangeDisplaySettingsExW(dev.as_ptr(), std::ptr::null(), std::ptr::null_mut(), 0, std::ptr::null()) };
}

/// Whether the port changed a monitor's mode (exclusive fullscreen holds it).
pub fn mode_changed() -> bool {
    CHANGED.lock().is_some()
}

fn save() {
    let Some(p) = PATH.lock().clone() else { return };
    let text = {
        let g = CHOICE.lock();
        let c = g.as_ref();
        let monitor = c.map(|c| c.monitor.clone()).unwrap_or_default();
        let res = c.and_then(|c| c.res).map(|(w, h)| format!("{w}x{h}")).unwrap_or_default();
        let hz = c.map_or(0, |c| c.hz);
        format!("# Display choices (the PC tab's Monitor, Resolution and Refresh Rate); written by the port\nmonitor={monitor}\nresolution={res}\nrefresh={hz}\n")
    };
    if let Err(e) = std::fs::write(&p, text) {
        crate::warn!("display: write {}: {e}", p.display());
    }
}

/// Reads display.txt (`path`) and makes the menu's lists.
pub fn init(path: PathBuf) {
    let mut c = Choice::default();
    if let Ok(text) = std::fs::read_to_string(&path) {
        for line in text.lines() {
            let line = line.trim();
            if line.starts_with('#') {
                continue;
            }
            let Some((k, v)) = line.split_once('=') else { continue };
            let v = v.trim();
            match k.trim() {
                "monitor" => c.monitor = v.to_string(),
                "resolution" => c.res = v.split_once('x').and_then(|(w, h)| Some((w.trim().parse().ok()?, h.trim().parse().ok()?))),
                "refresh" => c.hz = v.parse().unwrap_or(0),
                _ => {}
            }
        }
    }
    *CHOICE.lock() = Some(c);
    *PATH.lock() = Some(path);
    let mons = monitors();
    let t = target();
    let (mut resolutions, mut rates) = (Vec::new(), Vec::new());
    if let Some(t) = &t {
        for m in modes(&t.device) {
            if !resolutions.contains(&(m.w, m.h)) {
                resolutions.push((m.w, m.h));
            }
            if !rates.contains(&m.hz) {
                rates.push(m.hz);
            }
        }
        rates.sort_by(|a, b| b.cmp(a));
        let cur = current_mode(&t.device);
        crate::info!(
            "display: {} monitor(s); {} ({}) {}x{} at {} Hz, {} modes, HDR {}",
            mons.len(),
            t.name,
            t.device,
            cur.map_or(0, |m| m.w),
            cur.map_or(0, |m| m.h),
            cur.map_or(0, |m| m.hz),
            resolutions.len(),
            if t.hdr_on { format!("on (SDR white {:.0} nits)", t.sdr_white_nits) } else if t.hdr_supported { "off (supported)".into() } else { "not supported".into() }
        );
    }
    let _ = LISTS.set(Lists { monitors: mons, resolutions, rates });
}

fn lists() -> Option<&'static Lists> {
    LISTS.get()
}

fn short(s: &str, n: usize) -> String {
    if s.chars().count() > n { s.chars().take(n - 3).collect::<String>() + "..." } else { s.to_string() }
}

/// The Monitor list's options (the menu's), made at the start.
pub fn monitor_options() -> Vec<String> {
    let l = lists().map(|l| l.monitors.as_slice()).unwrap_or(&[]);
    if l.is_empty() {
        return vec!["Primary".into()];
    }
    l.iter().enumerate().map(|(i, m)| short(&format!("{}: {}", i + 1, m.name), 26)).collect()
}

/// The Resolution list's options: the desktop's, then the start monitor's resolutions.
pub fn resolution_options() -> Vec<String> {
    std::iter::once("Desktop".to_string()).chain(lists().map(|l| l.resolutions.as_slice()).unwrap_or(&[]).iter().map(|(w, h)| format!("{w} x {h}"))).collect()
}

/// The Refresh Rate list's options: the desktop's (or the highest), then the rates.
pub fn refresh_options() -> Vec<String> {
    std::iter::once("Automatic".to_string()).chain(lists().map(|l| l.rates.as_slice()).unwrap_or(&[]).iter().map(|hz| format!("{hz} Hz"))).collect()
}

/// The option the Monitor list shows (the chosen monitor's, else the primary's).
pub fn monitor_index() -> u32 {
    let Some(l) = lists() else { return 0 };
    let want = CHOICE.lock().as_ref().map(|c| c.monitor.clone()).unwrap_or_default();
    l.monitors.iter().position(|m| !want.is_empty() && m.device.eq_ignore_ascii_case(&want)).or_else(|| l.monitors.iter().position(|m| m.primary)).unwrap_or(0) as u32
}

pub fn resolution_index() -> u32 {
    let Some(l) = lists() else { return 0 };
    let res = CHOICE.lock().as_ref().and_then(|c| c.res);
    res.and_then(|r| l.resolutions.iter().position(|&x| x == r)).map_or(0, |i| i as u32 + 1)
}

pub fn refresh_index() -> u32 {
    let Some(l) = lists() else { return 0 };
    let hz = CHOICE.lock().as_ref().map_or(0, |c| c.hz);
    if hz == 0 {
        return 0;
    }
    l.rates.iter().position(|&x| x == hz).map_or(0, |i| i as u32 + 1)
}

/// The menu changed the Monitor (option `i`): kept, and the window moves.
pub fn set_monitor_index(i: u32) {
    let Some(m) = lists().and_then(|l| l.monitors.get(i as usize)) else { return };
    if let Some(c) = CHOICE.lock().as_mut() {
        c.monitor = if m.primary { String::new() } else { m.device.clone() };
    }
    save();
    crate::info!("display: monitor {} ({})", m.name, m.device);
    crate::gpu::vk::window::reapply_mode();
}

pub fn set_resolution_index(i: u32) {
    let res = if i == 0 { None } else { lists().and_then(|l| l.resolutions.get(i as usize - 1).copied()) };
    if let Some(c) = CHOICE.lock().as_mut() {
        c.res = res;
    }
    save();
    crate::gpu::vk::window::reapply_mode();
}

pub fn set_refresh_index(i: u32) {
    let hz = if i == 0 { 0 } else { lists().and_then(|l| l.rates.get(i as usize - 1).copied()).unwrap_or(0) };
    if let Some(c) = CHOICE.lock().as_mut() {
        c.hz = hz;
    }
    save();
    crate::gpu::vk::window::reapply_mode();
}

/// Whether the monitor the window is on has Windows' HDR turned on now, and its SDR white.
pub fn hdr_now(device: &str) -> Option<f32> {
    targets().into_iter().find(|t| t.0.eq_ignore_ascii_case(device)).filter(|t| t.3).map(|t| t.4)
}
