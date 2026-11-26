//! The performance overlay (the Graphics & Display list's Performance Overlay: Off, Simple,
//! Full): text in the picture's top-right corner, drawn by the presenter into the frame it shows
//! (`gpu::vk::present`), so it adds no window and no plane of its own (an overlay on a plane of
//! its own costs smoothness on displays without spare planes, such as those on DSC with GPUs
//! before NVIDIA's 50 series).
//!
//! Simple: frames a second and the mean frame time ("60 fps  16.7 ms"). Full adds the 1% and
//! 0.1% lows (the frame times at the 99th and 99.9th percentiles of the last 10 seconds, as
//! frames a second), the CPU use of the threads that bound the frame rate (each as a share of one
//! core: the game's main thread and the port's graphics command processor, which records the
//! game's GPU work), the GPU's use and clock (NVIDIA's NVAPI; other vendors: not shown), video
//! memory (this process's, VK_EXT_memory_budget), memory (this process's and the system's), and
//! the presentation (the present mode and the window: exclusive or borderless fullscreen can be
//! flipped by the display without composition, a window is composed by DWM; an estimate from the
//! window, Windows does not tell an application).
//!
//! The text is rendered with GDI (as `textbox`) four times a second at most; the presenter blits
//! it over each frame.

use parking_lot::Mutex;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU8, Ordering::Relaxed};
use std::time::{Duration, Instant};

/// The overlay chosen: 0 off, 1 simple, 2 full (`pcsettings::apply`).
pub static MODE: AtomicU8 = AtomicU8::new(0);

pub fn mode() -> u8 {
    MODE.load(Relaxed).min(2)
}

/// The window of frame times kept for the lows.
const WINDOW: Duration = Duration::from_secs(10);

struct Frames {
    last: Option<Instant>,
    /// (when, frame time in ms), oldest first, the last `WINDOW`.
    times: VecDeque<(Instant, f32)>,
}

static FRAMES: Mutex<Frames> = Mutex::new(Frames { last: None, times: VecDeque::new() });

/// A frame of the game's was flipped (`hle::libs::videoout`, the display's thread).
pub fn note_flip() {
    if mode() == 0 {
        // (Starts afresh when the overlay is turned on.)
        let mut f = FRAMES.lock();
        if f.last.is_some() {
            f.last = None;
            f.times.clear();
        }
        return;
    }
    let now = Instant::now();
    let mut f = FRAMES.lock();
    if let Some(prev) = f.last {
        let ms = now.duration_since(prev).as_secs_f32() * 1000.0;
        // (A pause, a load: not a frame.)
        if ms < 1000.0 {
            f.times.push_back((now, ms));
        }
    }
    f.last = Some(now);
    while f.times.front().is_some_and(|(t, _)| now.duration_since(*t) > WINDOW) {
        f.times.pop_front();
    }
}

/// Frames a second and their mean time over the last second; the 1% and 0.1% lows (frames a
/// second at the 99th and 99.9th percentile frame times) of the window.
fn frame_stats() -> Option<(f32, f32, f32, f32)> {
    let f = FRAMES.lock();
    let now = Instant::now();
    let recent: Vec<f32> = f.times.iter().filter(|(t, _)| now.duration_since(*t) <= Duration::from_secs(1)).map(|x| x.1).collect();
    if recent.is_empty() {
        return None;
    }
    let mean = recent.iter().sum::<f32>() / recent.len() as f32;
    let mut all: Vec<f32> = f.times.iter().map(|x| x.1).collect();
    drop(f);
    all.sort_by(|a, b| a.total_cmp(b));
    let pct = |p: f32| all[((all.len() as f32 * p) as usize).min(all.len() - 1)];
    Some((1000.0 / mean, mean, 1000.0 / pct(0.99), 1000.0 / pct(0.999)))
}

// ---------------------------------------------------------------------------
// CPU: the bounding threads' share of a core.

struct Cpu {
    at: Instant,
    looked: Option<Instant>,
    /// (label, host thread id, cycles at `at`).
    threads: Vec<(&'static str, u32, u64)>,
    shares: Vec<(&'static str, f32)>,
}

static CPU: Mutex<Option<Cpu>> = Mutex::new(None);

fn cpu_shares() -> Vec<(&'static str, f32)> {
    let mut g = CPU.lock();
    let c = g.get_or_insert_with(|| Cpu { at: Instant::now(), looked: None, threads: Vec::new(), shares: Vec::new() });
    // The threads, looked up again every 5 s (made at boot).
    if c.looked.is_none_or(|t| t.elapsed() > Duration::from_secs(5)) {
        c.looked = Some(Instant::now());
        let mut found: Vec<(&'static str, u32)> = Vec::new();
        if let Some(t) = crate::thread::THREADS.try_lock() {
            if let Some(tid) = t.iter().filter(|g| g.exited.try_lock().is_some_and(|e| e.is_none())).find(|g| g.name() == "GAME_MainThread").map(|g| g.host_tid.load(Relaxed) as u32) {
                found.push(("Main", tid));
            }
        }
        if let Some(tid) = crate::gpu::cpu_threads().iter().find(|t| t.0 == "gfx").map(|t| t.1) {
            found.push(("Render", tid));
        }
        if found.iter().map(|x| x.1).ne(c.threads.iter().map(|x| x.1)) {
            c.threads = found.into_iter().filter(|x| x.1 != 0).map(|(l, tid)| (l, tid, crate::gpu::threadcpu::thread_cycles_of(tid).unwrap_or(0))).collect();
            c.at = Instant::now();
            return Vec::new();
        }
    }
    let secs = c.at.elapsed().as_secs_f64();
    if secs >= 0.2 {
        let hz = crate::gpu::threadcpu::tsc_hz();
        c.shares = c
            .threads
            .iter_mut()
            .filter_map(|(label, tid, last)| {
                let now = crate::gpu::threadcpu::thread_cycles_of(*tid)?;
                let share = (now.wrapping_sub(*last) as f64 / hz / secs * 100.0) as f32;
                *last = now;
                Some((*label, share.clamp(0.0, 100.0)))
            })
            .collect();
        c.at = Instant::now();
    }
    c.shares.clone()
}

// ---------------------------------------------------------------------------
// GPU: NVIDIA's NVAPI (nvapi64.dll, the driver's): use and graphics clock.

type QueryInterface = unsafe extern "C" fn(u32) -> *const core::ffi::c_void;
type Init = unsafe extern "C" fn() -> i32;
type EnumGpus = unsafe extern "C" fn(*mut usize, *mut u32) -> i32;
type GpuCall = unsafe extern "C" fn(usize, *mut core::ffi::c_void) -> i32;

struct Nvapi {
    gpu: usize,
    pstates: Option<GpuCall>,
    clocks: Option<GpuCall>,
}

// SAFETY: a driver handle and function pointers, used under the lock.
unsafe impl Send for Nvapi {}

/// NV_GPU_DYNAMIC_PSTATES_INFO_EX: version, flags, 8 domains (present, percent); domain 0 the
/// graphics engine.
#[repr(C)]
struct Pstates {
    version: u32,
    flags: u32,
    util: [[u32; 2]; 8],
}

/// NV_GPU_CLOCK_FREQUENCIES_V2: version, flags (clock type 0: current), 32 domains (present,
/// kHz); domain 0 graphics, 4 memory.
#[repr(C)]
struct Clocks {
    version: u32,
    flags: u32,
    domain: [[u32; 2]; 32],
}

static NVAPI: Mutex<Option<Option<Nvapi>>> = Mutex::new(None);

fn nvapi_load() -> Option<Nvapi> {
    use windows_sys::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryW};
    let name: Vec<u16> = "nvapi64.dll".encode_utf16().chain(std::iter::once(0)).collect();
    unsafe {
        let h = LoadLibraryW(name.as_ptr());
        if h.is_null() {
            return None;
        }
        let qi = GetProcAddress(h, c"nvapi_QueryInterface".as_ptr().cast())?;
        let qi: QueryInterface = std::mem::transmute(qi);
        let get = |id: u32| {
            let p = qi(id);
            (!p.is_null()).then_some(p)
        };
        let init: Init = std::mem::transmute(get(0x0150_E828)?);
        if init() != 0 {
            return None;
        }
        let enum_gpus: EnumGpus = std::mem::transmute(get(0xE5AC_921F)?);
        let mut gpus = [0usize; 64];
        let mut n = 0u32;
        if enum_gpus(gpus.as_mut_ptr(), &mut n) != 0 || n == 0 {
            return None;
        }
        let pstates = get(0x60DE_D2ED).map(|p| std::mem::transmute::<_, GpuCall>(p));
        let clocks = get(0xDCB6_16C3).map(|p| std::mem::transmute::<_, GpuCall>(p));
        crate::info!("performance overlay: NVAPI loaded ({n} GPU(s); use {}, clocks {})", pstates.is_some(), clocks.is_some());
        Some(Nvapi { gpu: gpus[0], pstates, clocks })
    }
}

/// The GPU's use (%) and graphics clock (MHz), each when known.
fn gpu_stats() -> (Option<u32>, Option<u32>) {
    let mut g = NVAPI.lock();
    let Some(n) = g.get_or_insert_with(nvapi_load).as_ref() else { return (None, None) };
    let mut use_pct = None;
    if let Some(f) = n.pstates {
        let mut p = Pstates { version: std::mem::size_of::<Pstates>() as u32 | 1 << 16, flags: 0, util: [[0; 2]; 8] };
        if unsafe { f(n.gpu, (&mut p as *mut Pstates).cast()) } == 0 && p.util[0][0] & 1 != 0 {
            use_pct = Some(p.util[0][1].min(100));
        }
    }
    let mut mhz = None;
    if let Some(f) = n.clocks {
        let mut c = Clocks { version: std::mem::size_of::<Clocks>() as u32 | 2 << 16, flags: 0, domain: [[0; 2]; 32] };
        if unsafe { f(n.gpu, (&mut c as *mut Clocks).cast()) } == 0 && c.domain[0][0] & 1 != 0 {
            mhz = Some(c.domain[0][1] / 1000);
        }
    }
    (use_pct, mhz)
}

// ---------------------------------------------------------------------------
// Memory.

/// This process's private bytes, and the system's memory in use and in all.
fn ram() -> (u64, u64, u64) {
    use windows_sys::Win32::System::ProcessStatus::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS_EX};
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut pmc: PROCESS_MEMORY_COUNTERS_EX = unsafe { std::mem::zeroed() };
    pmc.cb = std::mem::size_of::<PROCESS_MEMORY_COUNTERS_EX>() as u32;
    let own = if unsafe { GetProcessMemoryInfo(GetCurrentProcess(), (&mut pmc as *mut PROCESS_MEMORY_COUNTERS_EX).cast(), pmc.cb) } != 0 { pmc.PrivateUsage as u64 } else { 0 };
    let mut ms: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    ms.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
    if unsafe { GlobalMemoryStatusEx(&mut ms) } == 0 {
        return (own, 0, 0);
    }
    (own, ms.ullTotalPhys - ms.ullAvailPhys, ms.ullTotalPhys)
}

fn gib(b: u64) -> f64 {
    b as f64 / (1u64 << 30) as f64
}

// ---------------------------------------------------------------------------
// The text.

/// The lines to show now (refreshed four times a second; None: off), and their generation.
pub fn lines() -> Option<(u64, Vec<String>)> {
    static CACHE: Mutex<Option<(Instant, u64, u8, Vec<String>)>> = Mutex::new(None);
    let mode = mode();
    if mode == 0 {
        return None;
    }
    let mut g = CACHE.lock();
    if let Some((at, generation, m, lines)) = g.as_ref() {
        if *m == mode && at.elapsed() < Duration::from_millis(250) {
            return Some((*generation, lines.clone()));
        }
    }
    let generation = g.as_ref().map_or(1, |c| c.1 + 1);
    let mut out = Vec::new();
    let stats = frame_stats();
    out.push(match stats {
        Some((fps, ms, _, _)) => format!("{fps:.0} fps  {ms:.1} ms"),
        None => "-- fps".to_string(),
    });
    if mode == 2 {
        if let Some((_, _, low1, low01)) = stats {
            out.push(format!("1% low {low1:.0}  0.1% low {low01:.0}"));
        }
        let cpu = cpu_shares();
        if !cpu.is_empty() {
            out.push(format!("CPU {}", cpu.iter().map(|(l, s)| format!("{l} {s:.0}%")).collect::<Vec<_>>().join("  ")));
        }
        match gpu_stats() {
            (Some(u), Some(m)) => out.push(format!("GPU {u}%  {m} MHz")),
            (Some(u), None) => out.push(format!("GPU {u}%")),
            (None, Some(m)) => out.push(format!("GPU {m} MHz")),
            (None, None) => {}
        }
        if let Some((used, budget)) = crate::gpu::vk::try_ctx().and_then(|c| c.device_budget()) {
            out.push(format!("VRAM {:.1} / {:.1} GB", gib(used), gib(budget)));
        }
        let (own, used, total) = ram();
        out.push(format!("RAM {:.1} GB  system {:.1} / {:.0} GB", gib(own), gib(used), gib(total)));
        out.push(crate::gpu::vk::present::presentation_line());
    }
    *g = Some((Instant::now(), generation, mode, out.clone()));
    Some((generation, out))
}

// ---------------------------------------------------------------------------
// Drawing (GDI, as `textbox`).

/// Design units at 1080 lines: the text's height, a line's pitch, the padding.
const TEXT_H: f32 = 22.0;
const LINE: f32 = 27.0;
const PAD: f32 = 10.0;

/// The overlay's lines as top-down BGRA pixels (alpha 255) for a picture `frame_h` pixels high:
/// light text on a dark panel, as small as the text.
pub fn render(lines: &[String], frame_h: u32) -> (Vec<u8>, u32, u32) {
    use windows_sys::Win32::Foundation::{RECT, SIZE};
    use windows_sys::Win32::Graphics::Gdi::*;
    let s = (frame_h as f32 / 1080.0).max(0.5);
    let px = |v: f32| (v * s).round() as i32;
    unsafe {
        let dc = CreateCompatibleDC(std::ptr::null_mut());
        if dc.is_null() {
            return (Vec::new(), 0, 0);
        }
        let face: Vec<u16> = "Consolas".encode_utf16().chain(std::iter::once(0)).collect();
        let font = CreateFontW(-px(TEXT_H), 0, 0, 0, 600, 0, 0, 0, DEFAULT_CHARSET as _, OUT_TT_PRECIS as _, CLIP_DEFAULT_PRECIS as _, ANTIALIASED_QUALITY as _, (FIXED_PITCH as u32 | FF_MODERN as u32) as _, face.as_ptr());
        let old_font = SelectObject(dc, font as _);
        // The panel's size: the widest line.
        let mut width = 0;
        for l in lines {
            let t: Vec<u16> = l.encode_utf16().collect();
            let mut sz = SIZE { cx: 0, cy: 0 };
            if !t.is_empty() {
                GetTextExtentPoint32W(dc, t.as_ptr(), t.len() as i32, &mut sz);
            }
            width = width.max(sz.cx);
        }
        let w = (width + 2 * px(PAD)).max(1) as u32;
        let h = (px(LINE) * lines.len() as i32 + 2 * px(PAD) - (px(LINE) - px(TEXT_H))).max(1) as u32;
        let mut out = vec![0u8; (w * h * 4) as usize];
        let mut bmi: BITMAPINFO = std::mem::zeroed();
        bmi.bmiHeader.biSize = std::mem::size_of::<BITMAPINFOHEADER>() as u32;
        bmi.bmiHeader.biWidth = w as i32;
        bmi.bmiHeader.biHeight = -(h as i32);
        bmi.bmiHeader.biPlanes = 1;
        bmi.bmiHeader.biBitCount = 32;
        bmi.bmiHeader.biCompression = BI_RGB;
        let mut bits: *mut core::ffi::c_void = std::ptr::null_mut();
        let bmp = CreateDIBSection(dc, &bmi, DIB_RGB_COLORS, &mut bits, std::ptr::null_mut(), 0);
        if bmp.is_null() || bits.is_null() {
            SelectObject(dc, old_font);
            DeleteObject(font as _);
            DeleteDC(dc);
            return (out, w, h);
        }
        let old_bmp = SelectObject(dc, bmp as _);
        let back = CreateSolidBrush(0x0010_0e0c);
        FillRect(dc, &RECT { left: 0, top: 0, right: w as i32, bottom: h as i32 }, back);
        DeleteObject(back as _);
        SetBkMode(dc, TRANSPARENT as _);
        for (i, l) in lines.iter().enumerate() {
            let mut t: Vec<u16> = l.encode_utf16().collect();
            if t.is_empty() {
                continue;
            }
            // The first line (the frame rate) bright, the others softer.
            SetTextColor(dc, if i == 0 { 0x0080_ff80 } else { 0x00e0_e0e0 });
            let top = px(PAD) + i as i32 * px(LINE);
            let mut r = RECT { left: px(PAD), top, right: w as i32 - px(PAD), bottom: top + px(LINE) };
            DrawTextW(dc, t.as_mut_ptr(), t.len() as i32, &mut r, DT_SINGLELINE | DT_LEFT | DT_TOP | DT_NOPREFIX);
        }
        GdiFlush();
        std::ptr::copy_nonoverlapping(bits as *const u8, out.as_mut_ptr(), out.len());
        for p in out.chunks_exact_mut(4) {
            p[3] = 255;
        }
        SelectObject(dc, old_bmp);
        SelectObject(dc, old_font);
        DeleteObject(bmp as _);
        DeleteObject(font as _);
        DeleteDC(dc);
        (out, w, h)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn struct_sizes() {
        // NVAPI's own sizes (their versions carry them).
        assert_eq!(std::mem::size_of::<super::Pstates>(), 72);
        assert_eq!(std::mem::size_of::<super::Clocks>(), 264);
    }
}
