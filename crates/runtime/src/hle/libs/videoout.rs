//! libSceVideoOut (PS5 API): display buffer sets, flip queue, vblank pacing,
//! output status and color settings.
//!
//! A vblank thread executes queued flips at 59.94 Hz (scaled by the flip rate);
//! with `tune flip_grid` 0 it shows a ready flip as soon as the flip rate
//! allows, between the vblanks the title sees.
//! Until the GPU renders into display buffers, presentation shows black: the
//! guest display memory is not produced by the title's GPU work yet, so
//! nothing else would be truthful. Color settings are recorded, not applied.
//!
//! Structures (PS5 layouts; see docs/PROVENANCE.md): BufferAttribute2 (0x50),
//! FlipStatus, VblankStatus, OutputStatus {u32 resolution, u32 dynamicRange,
//! u64 refreshRate, u64 flags, u64 reserved[3]}, ColorSettings {f32 gamma, ...}.

use crate::hle::Registry;
use crate::{reg, sysv};
use parking_lot::{Condvar, Mutex};
use std::collections::VecDeque;
use std::time::{Duration, Instant};

const ERR_INVALID_VALUE: i32 = 0x8029_0001u32 as i32;
const ERR_INVALID_ADDRESS: i32 = 0x8029_0002u32 as i32;
const ERR_INVALID_INDEX: i32 = 0x8029_0004u32 as i32;
const ERR_RESOURCE_BUSY: i32 = 0x8029_0009u32 as i32;
const ERR_INVALID_HANDLE: i32 = 0x8029_000Bu32 as i32;
pub const ERR_FLIP_QUEUE_FULL: i32 = 0x8029_0012u32 as i32;

/// Output reported to the title: 1080p SDR at 59.94 Hz (resolution 1 = 1080p,
/// 2 = 2160p; dynamic range 1 = SDR). A lower output keeps GPU emulation cost
/// down during bring-up; it is a choice of this port, not a console default.
const OUTPUT_RESOLUTION: u32 = 1;
const OUTPUT_DYNAMIC_RANGE: u32 = 1;
const REFRESH_59_94HZ: u64 = 3;

#[derive(Clone, Copy, Debug, Default)]
pub struct BufferAttr {
    pub pixel_format: u64,
    pub tiling: u32,
    pub aspect: u32,
    pub width: u32,
    pub height: u32,
    pub pitch: u32,
    pub option: u64,
    pub dcc_control: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct DisplayBuffer {
    pub addr: u64,
    pub metadata: u64,
    pub set: i32,
}

struct Flip {
    index: i32,
    arg: i64,
    submit_tsc: u64,
    /// The GPU work before it completed (`reserve_flip` queues it before,
    /// `commit_flip` marks it): only a ready flip at the front is shown.
    ready: bool,
    /// Unique per reserved flip (0: submitted ready).
    ticket: u64,
}

pub struct Port {
    pub buffers: [Option<DisplayBuffer>; 16],
    pub sets: [Option<BufferAttr>; 16],
    flip_rate: u32,
    pending: VecDeque<Flip>,
    count: u64,
    process_time: u64,
    tsc: u64,
    flip_arg: i64,
    submit_tsc: u64,
    current: i32,
    last_flip_vblank: u64,
    vblanks: u64,
    vblank_time: u64,
    vblank_tsc: u64,
    gamma: f32,
}

static PORT: Mutex<Option<Port>> = Mutex::new(None);
static FLIP_CV: Condvar = Condvar::new();
static VBLANK_CV: Condvar = Condvar::new();

fn vblank_thread() {
    crate::log::set_thread_name("vblank");
    // DS_VBLANK_HZ=<hz> (diagnosis, 30-1000): vblanks at another rate, so
    // flips are not quantized to 16.7 ms steps and A/Bs see frame cost below
    // a vblank (the title flips at every vblank, `sceVideoOutSetFlipRate(0)`;
    // it may pace or time its frames by them: not for behavior checks).
    let hz = std::env::var("DS_VBLANK_HZ").ok().and_then(|v| v.parse::<f64>().ok()).filter(|h| (30.0..=1000.0).contains(h));
    let period = match hz {
        Some(h) => {
            crate::info!("vblank at {h} Hz (DS_VBLANK_HZ, diagnosis)");
            Duration::from_nanos((1e9 / h) as u64)
        }
        None => Duration::from_nanos(16_683_333), // 59.94 Hz
    };
    // Flips off the vblank grid (`tune flip_grid` 0): the thread also wakes
    // every FLIP_TICK to show a ready flip as soon as the title's flip rate
    // allows (time since the last flip), while the title's vblanks (count,
    // events, waits) stay at the emulated rate.
    const FLIP_TICK: Duration = Duration::from_millis(1);
    let mut next = Instant::now() + period;
    let mut last_flip = Instant::now();
    let mut last_cap = 0;
    loop {
        // The PC settings' console variables (changed values, objects the title builds later).
        crate::pcsettings::convar::tick();
        let grid = crate::gpu::tune::flip_grid();
        let now = Instant::now();
        let wake = if grid { next } else { next.min(now + FLIP_TICK) };
        if wake > now {
            crate::thread::short_sleep(wake - now);
        }
        let now = Instant::now();
        let tick = now >= next;
        if tick {
            next += period;
        }
        let mut show: Option<DisplayBuffer> = None;
        let mut flipped = None;
        {
            let mut g = PORT.lock();
            let Some(p) = g.as_mut() else { continue };
            if tick {
                p.vblanks += 1;
                p.vblank_time = crate::hle::kernel::time::process_time_us();
                p.vblank_tsc = unsafe { core::arch::x86_64::_rdtsc() };
            }
            let mut interval = p.flip_rate as u64 + 1;
            // A frame cap (`tune frame_cap`): at least 1/cap between flips,
            // on the grid the whole number of vblanks that keeps under it.
            let cap = crate::gpu::tune::frame_cap();
            let min_gap = if cap > 0 { Duration::from_secs_f64(1.0 / cap as f64) } else { Duration::ZERO };
            if cap > 0 {
                interval = interval.max((min_gap.as_secs_f64() / period.as_secs_f64() - 0.01).ceil().max(1.0) as u64);
            }
            if cap != last_cap {
                last_cap = cap;
                crate::info!("frame cap {} (`tune frame_cap`): flips at least {interval} vblank(s) apart", if cap > 0 { format!("{cap} fps") } else { "off".into() });
            }
            // On the grid: at a vblank, `interval` vblanks after the last
            // flip. Off it: once `interval` vblank periods (and 1/cap) have
            // passed since the last flip (half a millisecond early is close
            // enough).
            let due = if grid { tick && p.vblanks - p.last_flip_vblank >= interval } else { now.duration_since(last_flip) + Duration::from_micros(500) >= (period * interval as u32).max(min_gap) };
            if p.pending.front().is_some_and(|f| f.ready) && due {
                let f = p.pending.pop_front().unwrap();
                p.current = f.index;
                p.count += 1;
                p.flip_arg = f.arg;
                p.submit_tsc = f.submit_tsc;
                if tick {
                    p.process_time = p.vblank_time;
                    p.tsc = p.vblank_tsc;
                } else {
                    p.process_time = crate::hle::kernel::time::process_time_us();
                    p.tsc = unsafe { core::arch::x86_64::_rdtsc() };
                }
                p.last_flip_vblank = p.vblanks;
                last_flip = now;
                if p.count <= 3 || p.count % 600 == 0 {
                    crate::info!("flip #{} -> buffer {} (arg {:#x})", p.count, f.index, f.arg);
                }
                if f.index >= 0 {
                    show = p.buffers[f.index as usize];
                }
                crate::gpu::FRAMES.store(p.count, std::sync::atomic::Ordering::Relaxed);
                crate::gpu::pacing::note_flip(p.count as u64, p.process_time);
                flipped = Some(p.count);
                FLIP_CV.notify_all();
            }
            if tick {
                VBLANK_CV.notify_all();
            }
        }
        if let Some(n) = flipped {
            crate::gpu::timeline::flipped(n as u64);
        }
        if let Some(b) = show {
            let (n, attr) = PORT.lock().as_ref().map(|p| (p.count, p.sets.get(b.set as usize).copied().flatten())).unwrap_or((0, None));
            if let Some(a) = attr {
                crate::gpu::capture::note_display_size(a.width, a.height);
            }
            maybe_screenshot(&b, n);
            present_buffer(&b);
        }
    }
}

fn port<R>(handle: i32, f: impl FnOnce(&mut Port) -> R) -> Result<R, i32> {
    if handle != 1 {
        return Err(ERR_INVALID_HANDLE);
    }
    let mut g = PORT.lock();
    match g.as_mut() {
        Some(p) => Ok(f(p)),
        None => Err(ERR_INVALID_HANDLE),
    }
}

/// Queues a flip of `index` (-1 = blank). Used by sceVideoOutSubmitFlip and by
/// GPU flips (sceAgcDcbSetFlip) when the command processor reaches them.
pub fn submit_flip(handle: i32, index: i32, arg: i64) -> i32 {
    port(handle, |p| {
        if index >= 16 || (index >= 0 && p.buffers[index as usize].is_none()) {
            return ERR_INVALID_INDEX;
        }
        if p.pending.len() >= 16 {
            return ERR_FLIP_QUEUE_FULL;
        }
        p.pending.push_back(Flip { index, arg, submit_tsc: unsafe { core::arch::x86_64::_rdtsc() }, ready: true, ticket: 0 });
        0
    })
    .unwrap_or_else(|e| e)
}

/// A flip queued now and shown only once `commit_flip` marks it ready (a
/// GPU queue's FLIP packet, `tune flip_async`: the GPU work before it is
/// still running). It counts towards the queue's limit and keeps its buffer
/// busy at once. Ok(ticket), or the error `submit_flip` would give.
pub fn reserve_flip(handle: i32, index: i32, arg: i64) -> Result<u64, i32> {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    port(handle, |p| {
        if index >= 16 || (index >= 0 && p.buffers[index as usize].is_none()) {
            return Err(ERR_INVALID_INDEX);
        }
        if p.pending.len() >= 16 {
            return Err(ERR_FLIP_QUEUE_FULL);
        }
        let ticket = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        p.pending.push_back(Flip { index, arg, submit_tsc: unsafe { core::arch::x86_64::_rdtsc() }, ready: false, ticket });
        Ok(ticket)
    })
    .unwrap_or_else(Err)
}

/// The GPU work before a reserved flip completed: it may be shown.
pub fn commit_flip(ticket: u64) {
    let mut g = PORT.lock();
    if let Some(p) = g.as_mut() {
        if let Some(f) = p.pending.iter_mut().find(|f| f.ticket == ticket) {
            f.ready = true;
        }
    }
}

/// Whether display buffer `index` is scanned out or waiting in the flip
/// queue (GPU `WaitUntilSafeForRendering` waits until it is neither).
pub fn buffer_busy(handle: i32, index: i32) -> bool {
    port(handle, |p| p.current == index || p.pending.iter().any(|f| f.index == index)).unwrap_or(false)
}

pub fn display_buffer(index: i32) -> Option<(DisplayBuffer, BufferAttr)> {
    let g = PORT.lock();
    let p = g.as_ref()?;
    let b = p.buffers.get(index as usize).copied().flatten()?;
    let a = p.sets.get(b.set as usize).copied().flatten()?;
    Some((b, a))
}

unsafe fn read_attr(attr: *const u8) -> BufferAttr {
    unsafe {
        let r32 = |o: usize| *(attr.add(o) as *const u32);
        let r64 = |o: usize| *(attr.add(o) as *const u64);
        BufferAttr {
            tiling: r32(0x04),
            aspect: r32(0x08),
            width: r32(0x0C),
            height: r32(0x10),
            pitch: r32(0x14),
            option: r64(0x18),
            pixel_format: r64(0x20),
            dcc_control: r32(0x30),
        }
    }
}

sysv! {
    fn sceVideoOutOpen(user: i32, bus: i32, index: i32, param: *const u8) -> i32 {
        if bus != 0 { return ERR_INVALID_VALUE; }
        let mut g = PORT.lock();
        if g.is_some() { return ERR_RESOURCE_BUSY; }
        *g = Some(Port {
            buffers: [None; 16],
            sets: [None; 16],
            flip_rate: 0,
            pending: VecDeque::new(),
            count: 0,
            process_time: 0,
            tsc: 0,
            flip_arg: 0,
            submit_tsc: 0,
            current: -1,
            last_flip_vblank: 0,
            vblanks: 0,
            vblank_time: 0,
            vblank_tsc: 0,
            gamma: 1.0,
        });
        drop(g);
        crate::thread::spawn_host("vblank", 1 << 20, Box::new(vblank_thread)).expect("vblank thread");
        crate::info!("sceVideoOutOpen -> 1");
        1
    }
    fn sceVideoOutClose(h: i32) -> i32 { port(h, |_| 0).unwrap_or_else(|e| e) }
    fn sceVideoOutSetFlipRate(h: i32, rate: i32) -> i32 {
        port(h, |p| {
            p.flip_rate = rate.clamp(0, 2) as u32;
            crate::info!("sceVideoOutSetFlipRate({rate})");
            0
        }).unwrap_or_else(|e| e)
    }
    fn sceVideoOutColorSettingsSetGamma_(settings: *mut f32, gamma: f32, size: u64) -> i32 {
        if settings.is_null() { return ERR_INVALID_ADDRESS; }
        if !(0.1..=2.0).contains(&gamma) { return ERR_INVALID_VALUE; }
        *settings = gamma;
        0
    }
    fn sceVideoOutAdjustColor_(h: i32, settings: *const f32, size: u64) -> i32 {
        if settings.is_null() { return ERR_INVALID_ADDRESS; }
        let gamma = *settings;
        port(h, |p| {
            p.gamma = gamma;
            crate::info!("sceVideoOutAdjustColor: gamma {gamma} (recorded; not applied to presentation yet)");
            0
        }).unwrap_or_else(|e| e)
    }
    fn sceVideoOutGetOutputStatus(h: i32, st: *mut u8) -> i32 {
        if st.is_null() { return ERR_INVALID_ADDRESS; }
        port(h, |_| {
            std::ptr::write_bytes(st, 0, 0x30);
            *(st as *mut u32) = OUTPUT_RESOLUTION;
            *(st.add(4) as *mut u32) = OUTPUT_DYNAMIC_RANGE;
            *(st.add(8) as *mut u64) = REFRESH_59_94HZ;
            0
        }).unwrap_or_else(|e| e)
    }
    fn sceVideoOutSetBufferAttribute2(attr: *mut u8, pixel_format: u64, tiling: u32, width: u32, height: u32, option: u64, dcc_control: u32, dcc_clear: u64) {
        if attr.is_null() { return; }
        std::ptr::write_bytes(attr, 0, 0x50);
        *(attr.add(0x04) as *mut u32) = tiling;
        *(attr.add(0x08) as *mut u32) = 0; // aspect ratio 16:9
        *(attr.add(0x0C) as *mut u32) = width;
        *(attr.add(0x10) as *mut u32) = height;
        *(attr.add(0x14) as *mut u32) = width; // pitch in pixels
        *(attr.add(0x18) as *mut u64) = option;
        *(attr.add(0x20) as *mut u64) = pixel_format;
        *(attr.add(0x28) as *mut u64) = dcc_clear;
        *(attr.add(0x30) as *mut u32) = dcc_control;
    }
    fn sceVideoOutRegisterBuffers2(h: i32, set: i32, start: i32, bufs: *const u64, num: i32, attr: *const u8, category: i32, option: u64) -> i32 {
        if bufs.is_null() || attr.is_null() { return ERR_INVALID_ADDRESS; }
        if !(0..16).contains(&set) || start < 0 || num <= 0 || start + num > 16 { return ERR_INVALID_INDEX; }
        let a = read_attr(attr);
        port(h, |p| {
            p.sets[set as usize] = Some(a);
            let mut addrs = Vec::new();
            for i in 0..num as usize {
                // SceVideoOutBuffers entries: {data, metadata, reserved[2]}
                // (32 bytes; the title fills them at eboot+0x90a99b/0x90a9c6).
                let data = *bufs.add(i * 4);
                let meta = *bufs.add(i * 4 + 1);
                p.buffers[start as usize + i] = Some(DisplayBuffer { addr: data, metadata: meta, set });
                addrs.push((data, meta));
            }
            crate::info!(
                "sceVideoOutRegisterBuffers2 set {set} [{start}..{}) {}x{} pitch {} format {:#x} tiling {} option {:#x} dcc {:#x} category {category} buffers {:x?}",
                start + num, a.width, a.height, a.pitch, a.pixel_format, a.tiling, a.option, a.dcc_control, addrs
            );
            0
        }).unwrap_or_else(|e| e)
    }
    fn sceVideoOutSubmitChangeBufferAttribute2(h: i32, set: i32, attr: *const u8, option: u64) -> i32 {
        if attr.is_null() { return ERR_INVALID_ADDRESS; }
        if !(0..16).contains(&set) { return ERR_INVALID_INDEX; }
        let a = read_attr(attr);
        port(h, |p| {
            p.sets[set as usize] = Some(a);
            crate::info!("sceVideoOutSubmitChangeBufferAttribute2 set {set}: {}x{} format {:#x}", a.width, a.height, a.pixel_format);
            0
        }).unwrap_or_else(|e| e)
    }
    fn sceVideoOutUnregisterBuffers(h: i32, set: i32) -> i32 {
        if !(0..16).contains(&set) { return ERR_INVALID_INDEX; }
        port(h, |p| {
            p.sets[set as usize] = None;
            for b in p.buffers.iter_mut() {
                if b.is_some_and(|x| x.set == set) { *b = None; }
            }
            0
        }).unwrap_or_else(|e| e)
    }
    fn sceVideoOutSubmitFlip(h: i32, index: i32, mode: u32, arg: i64) -> i32 {
        submit_flip(h, index, arg)
    }
    fn sceVideoOutIsFlipPending(h: i32) -> i32 {
        port(h, |p| p.pending.len() as i32).unwrap_or_else(|e| e)
    }
    fn sceVideoOutWaitVblank(h: i32) -> i32 {
        if h != 1 { return ERR_INVALID_HANDLE; }
        let mut g = PORT.lock();
        let Some(start) = g.as_ref().map(|p| p.vblanks) else { return ERR_INVALID_HANDLE };
        while g.as_ref().is_some_and(|p| p.vblanks == start) {
            VBLANK_CV.wait_for(&mut g, Duration::from_millis(50));
        }
        0
    }
    fn sceVideoOutGetFlipStatus(h: i32, st: *mut u8) -> i32 {
        if st.is_null() { return ERR_INVALID_ADDRESS; }
        port(h, |p| {
            std::ptr::write_bytes(st, 0, 0x88);
            *(st as *mut u64) = p.count;
            *(st.add(0x08) as *mut u64) = p.process_time;
            *(st.add(0x18) as *mut i64) = p.flip_arg;
            *(st.add(0x28) as *mut u64) = p.tsc;
            *(st.add(0x30) as *mut i32) = 0; // gcQueueNum
            *(st.add(0x34) as *mut i32) = p.pending.len() as i32;
            *(st.add(0x38) as *mut i32) = p.current;
            *(st.add(0x40) as *mut u64) = p.submit_tsc;
            0
        }).unwrap_or_else(|e| e)
    }
    fn sceVideoOutSetWindowModeMargins(h: i32, top: i32, bottom: i32) -> i32 {
        port(h, |_| {
            crate::info!("sceVideoOutSetWindowModeMargins(top {top}, bottom {bottom}) (recorded only)");
            0
        }).unwrap_or_else(|e| e)
    }
}

pub fn register(r: &mut Registry) {
    reg!(r, "libSceVideoOut", [
        sceVideoOutOpen, sceVideoOutClose, sceVideoOutSetFlipRate, sceVideoOutColorSettingsSetGamma_,
        sceVideoOutAdjustColor_, sceVideoOutGetOutputStatus, sceVideoOutSetBufferAttribute2,
        sceVideoOutRegisterBuffers2, sceVideoOutSubmitChangeBufferAttribute2, sceVideoOutUnregisterBuffers,
        sceVideoOutSubmitFlip, sceVideoOutIsFlipPending, sceVideoOutWaitVblank, sceVideoOutGetFlipStatus,
        sceVideoOutSetWindowModeMargins,
    ]);
}

/// The size display buffer `b` is registered with (its set's attribute).
fn registered_size(b: &DisplayBuffer) -> Option<(u32, u32)> {
    PORT.lock().as_ref().and_then(|p| p.sets.get(b.set as usize).copied().flatten()).map(|a| (a.width, a.height))
}

/// Presents a flipped display buffer: the GPU image the title rendered at
/// the buffer's address (`DS_GPU=1`), else a black frame of the buffer's size
/// (never raw guest memory, which is tiled).
fn present_buffer(b: &DisplayBuffer) {
    if crate::gpu::vk::try_ctx().is_none() {
        return;
    }
    if crate::gpu::exec::enabled() {
        let size = registered_size(b);
        if let Some((image, format, width, height)) = crate::gpu::images::with(|im| im.rendered_at(b.addr, size)) {
            // Diagnosis: the image each buffer presents, logged when it changes.
            static LAST: Mutex<Vec<(u64, u64)>> = Mutex::new(Vec::new());
            let id = ash::vk::Handle::as_raw(image);
            let changed = {
                let mut last = LAST.lock();
                match last.iter_mut().find(|(a, _)| *a == b.addr) {
                    Some(e) if e.1 == id => false,
                    Some(e) => {
                        e.1 = id;
                        true
                    }
                    None => {
                        last.push((b.addr, id));
                        true
                    }
                }
            };
            if changed {
                let all = crate::gpu::images::with(|im| im.at_base_report(b.addr));
                crate::info!("present: buffer {:#x} shows {format:?} {width}x{height} (registered {size:?}); images there: {all}", b.addr);
            }
            crate::gpu::vk::present::present(crate::gpu::vk::present::Source::Image { image, format, layout: ash::vk::ImageLayout::GENERAL, width, height });
            return;
        }
    }
    let (w, h) = PORT
        .lock()
        .as_ref()
        .and_then(|p| p.sets.get(b.set as usize).copied().flatten())
        .map(|a| (a.width.clamp(1, 8192), a.height.clamp(1, 8192)))
        .unwrap_or((1920, 1080));
    let data = vec![0u8; (w * h * 4) as usize];
    crate::gpu::vk::present::present(crate::gpu::vk::present::Source::Pixels { data, width: w, height: h });
}

/// Screenshot requested through the control channel (file name stem).
pub static SHOT_REQUEST: Mutex<Option<String>> = Mutex::new(None);

/// Screenshots being converted, encoded and written on threads of their own
/// (`maybe_screenshot`); `wait_shots` waits for them at shutdown.
static SHOTS_PENDING: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Waits (at most 10 s) for the screenshots still being written.
pub fn wait_shots() {
    let t = std::time::Instant::now();
    while SHOTS_PENDING.load(std::sync::atomic::Ordering::Acquire) > 0 && t.elapsed() < std::time::Duration::from_secs(10) {
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
}

fn maybe_screenshot(b: &DisplayBuffer, flip: u64) {
    let requested = SHOT_REQUEST.lock().take();
    let listed = std::env::var("DS_SCREENSHOT").is_ok_and(|list| list.split(',').any(|s| s.trim().parse::<u64>().ok() == Some(flip)));
    if requested.is_none() && !listed {
        return;
    }
    // The buffer is read back here (as flipped); its conversion, PNG encoding
    // and write run on a thread of their own: on this thread they held the
    // flips ~0.5 s per 4K shot (the boss fights' worst frames, 2026-11-01
    // night).
    let read = || -> Result<(Vec<u8>, ash::vk::Format, u32, u32, std::path::PathBuf), String> {
        if crate::gpu::vk::try_ctx().is_none() || !crate::gpu::exec::enabled() {
            return Err("GPU execution is off (DS_GPU=1 renders frames)".into());
        }
        let (raw, format, w, h) = crate::gpu::images::with(|im| -> Result<(Vec<u8>, ash::vk::Format, u32, u32), String> {
            let (image, format, w, h) = im.rendered_at(b.addr, registered_size(b)).ok_or("nothing was rendered to the buffer")?;
            Ok((im.read_back(image, format, w, h)?, format, w, h))
        })?;
        let dir = crate::run_dir().ok_or("no run directory")?;
        let name = match &requested {
            Some(n) if !n.is_empty() => format!("{n}.png"),
            _ => format!("screenshot-{flip}.png"),
        };
        Ok((raw, format, w, h, dir.join(name)))
    };
    let addr = b.addr;
    match read() {
        Ok((raw, format, w, h, path)) => {
            SHOTS_PENDING.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
            let write = move || {
                let r = crate::gpu::images::to_rgba8(format, &raw)
                    .ok_or_else(|| format!("no RGBA8 conversion for {format:?}"))
                    .and_then(|rgba| std::fs::write(&path, crate::gpu::png::encode(w, h, &rgba)).map_err(|e| e.to_string()));
                match r {
                    Ok(()) => crate::info!("screenshot of flip {flip} (buffer {addr:#x}): {}", path.display()),
                    Err(e) => crate::warn!("screenshot of flip {flip} (buffer {addr:#x}) unavailable: {e}"),
                }
                SHOTS_PENDING.fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
            };
            if let Err(e) = std::thread::Builder::new().name("shot-write".into()).spawn(write) {
                SHOTS_PENDING.fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
                crate::warn!("screenshot of flip {flip} (buffer {addr:#x}) unavailable: no writer thread ({e})");
            }
        }
        Err(e) => crate::warn!("screenshot of flip {flip} (buffer {addr:#x}) unavailable: {e}"),
    }
}
