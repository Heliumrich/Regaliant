//! libSceVideoOut: display buffers, flip queue and flip/vblank events.
//! A vblank thread executes queued flips at 60 Hz (scaled by the flip rate).

use crate::hle::Registry;
use crate::{reg, sysv};
use parking_lot::{Condvar, Mutex};
use std::collections::VecDeque;
use std::time::{Duration, Instant};

const ERR_INVALID_HANDLE: i32 = 0x8029_000Bu32 as i32;
const ERR_INVALID_INDEX: i32 = 0x8029_0004u32 as i32;
const ERR_FLIP_QUEUE_FULL: i32 = 0x8029_0012u32 as i32;
const EVENT_FLIP: u64 = 0;

#[derive(Clone, Copy, Debug, Default)]
pub struct BufferAttr {
    pub pixel_format: u32,
    pub tiling: u32,
    pub aspect: u32,
    pub width: u32,
    pub height: u32,
    pub pitch: u32,
    pub option: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct DisplayBuffer {
    pub addr: u64,
    pub attr: BufferAttr,
}

struct Flip {
    index: i32,
    arg: i64,
    submit_tsc: u64,
    /// Requested through sceGnmSubmitAndFlipCommandBuffers.
    gpu: bool,
}

pub struct Port {
    pub buffers: [Option<DisplayBuffer>; 16],
    groups: u32,
    flip_rate: u32,
    flip_events: Vec<(u64, u64)>,
    pending: VecDeque<Flip>,
    pending_per_buffer: [u32; 16],
    /// GPU flips submitted with their command buffers but not yet reached by the GPU.
    gc_unreached: u32,
    count: u64,
    process_time: u64,
    tsc: u64,
    flip_arg: i64,
    submit_tsc: u64,
    current: i32,
    last_flip_vblank: u64,
    vblanks: u64,
}

static PORT: Mutex<Option<Port>> = Mutex::new(None);
static FLIP_CV: Condvar = Condvar::new();

fn vblank_thread() {
    crate::log::set_thread_name("vblank");
    let period = Duration::from_nanos(16_666_667);
    let mut next = Instant::now() + period;
    loop {
        let now = Instant::now();
        if next > now {
            std::thread::sleep(next - now);
        }
        next += period;
        let mut fired: Vec<(u64, u64, i64)> = Vec::new();
        let mut show: Option<DisplayBuffer> = None;
        {
            let mut g = PORT.lock();
            let Some(p) = g.as_mut() else { continue };
            p.vblanks += 1;
            let interval = p.flip_rate as u64 + 1;
            if !p.pending.is_empty() && p.vblanks - p.last_flip_vblank >= interval {
                let f = p.pending.pop_front().unwrap();
                if f.index >= 0 {
                    p.pending_per_buffer[f.index as usize] -= 1;
                }
                p.current = f.index;
                p.count += 1;
                p.flip_arg = f.arg;
                p.submit_tsc = f.submit_tsc;
                p.process_time = crate::hle::kernel::time::process_time_us();
                p.tsc = unsafe { core::arch::x86_64::_rdtsc() };
                p.last_flip_vblank = p.vblanks;
                if p.count <= 3 || p.count % 600 == 0 {
                    crate::info!("flip #{} -> buffer {} (arg {:#x})", p.count, f.index, f.arg);
                }
                if f.index >= 0 {
                    show = p.buffers[f.index as usize];
                }
                for &(eq, ud) in &p.flip_events {
                    fired.push((eq, ud, f.arg));
                }
                FLIP_CV.notify_all();
            }
        }
        if let Some(b) = show {
            let n = PORT.lock().as_ref().map(|p| p.count).unwrap_or(0);
            maybe_screenshot(&b, n);
            present_buffer(&b);
        }
        for (eq, _ud, arg) in fired {
            if let Some(q) = unsafe { crate::hle::kernel::equeue::from_handle(eq) } {
                q.trigger(EVENT_FLIP, crate::hle::kernel::equeue::EVFILT_DISPLAY, Some(arg as u64), None);
            }
        }
    }
}

fn submit_flip(handle: i32, index: i32, arg: i64, gpu: bool) -> i32 {
    if handle != 1 {
        return ERR_INVALID_HANDLE;
    }
    let mut g = PORT.lock();
    let Some(p) = g.as_mut() else { return ERR_INVALID_HANDLE };
    if gpu {
        p.gc_unreached = p.gc_unreached.saturating_sub(1);
    }
    if index >= 16 || (index >= 0 && p.buffers[index as usize].is_none()) {
        return ERR_INVALID_INDEX;
    }
    if p.pending.len() >= 16 {
        return ERR_FLIP_QUEUE_FULL;
    }
    if index >= 0 {
        p.pending_per_buffer[index as usize] += 1;
    }
    p.pending.push_back(Flip { index, arg, submit_tsc: unsafe { core::arch::x86_64::_rdtsc() }, gpu });
    0
}

/// Registers a flip submitted together with command buffers; it stays
/// pending (gcQueueNum/flipPendingNum) until the GPU reaches it and the
/// vblank thread executes it.
pub fn gpu_flip_requested(handle: i32) {
    if handle != 1 {
        return;
    }
    if let Some(p) = PORT.lock().as_mut() {
        p.gc_unreached += 1;
    }
}

/// Flip requested by the GPU after its command buffers completed.
pub fn gpu_flip(handle: i32, index: i32, _mode: u32, arg: i64) {
    let r = submit_flip(handle, index, arg, true);
    if r != 0 {
        crate::error!("GPU flip of buffer {index} failed: {r:#x}");
    }
}

/// Blocks (GPU side) until no flip of `index` is pending.
pub fn wait_flip_done(handle: i32, index: i32) {
    if handle != 1 || !(0..16).contains(&index) {
        return;
    }
    let mut g = PORT.lock();
    let start = Instant::now();
    let deadline = start + Duration::from_secs(5);
    let mut warned = false;
    while g.as_ref().is_some_and(|p| p.pending_per_buffer[index as usize] > 0) {
        if !warned && start.elapsed() > Duration::from_millis(250) {
            warned = true;
            crate::warn!("GPU wait for flip of buffer {index} blocked for 250 ms (pending {:?})", g.as_ref().map(|p| p.pending.len()));
        }
        if FLIP_CV.wait_until(&mut g, deadline.min(Instant::now() + Duration::from_millis(100))).timed_out() && Instant::now() >= deadline {
            crate::warn!("wait for flip of buffer {index} timed out");
            return;
        }
    }
}

pub fn display_buffer(index: i32) -> Option<DisplayBuffer> {
    PORT.lock().as_ref().and_then(|p| p.buffers.get(index as usize).copied().flatten())
}

sysv! {
    fn sceVideoOutOpen(user: i32, bus: i32, index: i32, param: *const u8) -> i32 {
        if bus != 0 { return 0x8029_0002u32 as i32; }
        let mut g = PORT.lock();
        if g.is_some() { return 0x8029_0009u32 as i32; } // resource busy
        *g = Some(Port {
            buffers: [None; 16],
            groups: 0,
            flip_rate: 0,
            flip_events: Vec::new(),
            pending: VecDeque::new(),
            pending_per_buffer: [0; 16],
            gc_unreached: 0,
            count: 0,
            process_time: 0,
            tsc: 0,
            flip_arg: 0,
            submit_tsc: 0,
            current: -1,
            last_flip_vblank: 0,
            vblanks: 0,
        });
        drop(g);
        crate::thread::spawn_host("vblank", 1 << 20, Box::new(vblank_thread)).expect("vblank thread");
        crate::info!("sceVideoOutOpen -> 1");
        1
    }
    fn sceVideoOutClose(h: i32) -> i32 {
        if h != 1 { return ERR_INVALID_HANDLE; }
        0
    }
    fn sceVideoOutSetFlipRate(h: i32, rate: i32) -> i32 {
        let mut g = PORT.lock();
        let Some(p) = g.as_mut() else { return ERR_INVALID_HANDLE };
        p.flip_rate = rate.clamp(0, 2) as u32;
        crate::info!("sceVideoOutSetFlipRate({rate})");
        0
    }
    fn sceVideoOutAddFlipEvent(eq: u64, h: i32, udata: u64) -> i32 {
        let mut g = PORT.lock();
        let Some(p) = g.as_mut() else { return ERR_INVALID_HANDLE };
        let Some(q) = crate::hle::kernel::equeue::from_handle(eq) else { return 0x8029_0001u32 as i32 };
        q.add(EVENT_FLIP, crate::hle::kernel::equeue::EVFILT_DISPLAY, 0, udata);
        p.flip_events.push((eq, udata));
        0
    }
    fn sceVideoOutSetBufferAttribute(attr: *mut u32, fmt: u32, tiling: u32, aspect: u32, w: u32, h: u32, pitch: u32) {
        let v = [fmt, tiling, aspect, w, h, pitch, 0, 0, 0, 0];
        for (i, x) in v.iter().enumerate() { *attr.add(i) = *x; }
    }
    fn sceVideoOutRegisterBuffers(h: i32, start: i32, addrs: *const u64, num: i32, attr: *const u32) -> i32 {
        let mut g = PORT.lock();
        let Some(p) = g.as_mut() else { return ERR_INVALID_HANDLE };
        if start < 0 || num <= 0 || start + num > 16 || addrs.is_null() || attr.is_null() { return ERR_INVALID_INDEX; }
        let a = BufferAttr {
            pixel_format: *attr, tiling: *attr.add(1), aspect: *attr.add(2), width: *attr.add(3),
            height: *attr.add(4), pitch: *attr.add(5), option: *attr.add(6),
        };
        for i in 0..num as usize {
            p.buffers[start as usize + i] = Some(DisplayBuffer { addr: *addrs.add(i), attr: a });
        }
        let group = p.groups;
        p.groups += 1;
        crate::info!(
            "sceVideoOutRegisterBuffers [{}..{}) {}x{} pitch {} format {:#x} tiling {} -> group {} addrs {:x?}",
            start, start + num, a.width, a.height, a.pitch, a.pixel_format, a.tiling, group,
            (0..num as usize).map(|i| *addrs.add(i)).collect::<Vec<_>>()
        );
        group as i32
    }
    fn sceVideoOutSubmitFlip(h: i32, index: i32, mode: u32, arg: i64) -> i32 {
        submit_flip(h, index, arg, false)
    }
    fn sceVideoOutGetFlipStatus(h: i32, st: *mut u64) -> i32 {
        let g = PORT.lock();
        let Some(p) = g.as_ref() else { return ERR_INVALID_HANDLE };
        if st.is_null() { return 0x8029_0001u32 as i32; }
        *st = p.count;
        *st.add(1) = p.process_time;
        *st.add(2) = p.tsc;
        *st.add(3) = p.flip_arg as u64;
        *st.add(4) = p.submit_tsc;
        *st.add(5) = 0;
        let i = st.add(6) as *mut i32;
        let gpu_queued = p.pending.iter().filter(|f| f.gpu).count() as u32;
        *i = (p.gc_unreached + gpu_queued) as i32; // gcQueueNum
        *i.add(1) = (p.gc_unreached as usize + p.pending.len()) as i32; // flipPendingNum
        *i.add(2) = p.current;
        *i.add(3) = 0;
        0
    }
    fn sceVideoOutGetResolutionStatus(h: i32, st: *mut u32) -> i32 {
        if st.is_null() { return 0x8029_0001u32 as i32; }
        *st = 1920; *st.add(1) = 1080; *st.add(2) = 1920; *st.add(3) = 1080;
        *(st.add(4) as *mut u64) = 3; // 59.94 Hz class
        *(st.add(6) as *mut f32) = 50.0;
        0
    }
}

pub fn register(r: &mut Registry) {
    reg!(r, "libSceVideoOut", [
        sceVideoOutOpen, sceVideoOutClose, sceVideoOutSetFlipRate, sceVideoOutAddFlipEvent,
        sceVideoOutSetBufferAttribute, sceVideoOutRegisterBuffers, sceVideoOutSubmitFlip, sceVideoOutGetFlipStatus,
        sceVideoOutGetResolutionStatus,
    ]);
}

/// Presents a display buffer. Until the GPU renders into Vulkan images, the
/// guest memory contents are shown directly (linear interpretation).
fn present_buffer(b: &DisplayBuffer) {
    if crate::gpu::vk::try_ctx().is_none() {
        return;
    }
    let (w, h, pitch) = (b.attr.width, b.attr.height, b.attr.pitch.max(b.attr.width));
    if w == 0 || h == 0 || w > 8192 || h > 8192 {
        return;
    }
    let mut data = vec![0u8; (w * h * 4) as usize];
    unsafe {
        for y in 0..h as usize {
            let src = (b.addr + (y as u64) * pitch as u64 * 4) as *const u8;
            std::ptr::copy_nonoverlapping(src, data.as_mut_ptr().add(y * w as usize * 4), w as usize * 4);
        }
    }
    crate::gpu::vk::present::present(crate::gpu::vk::present::Source::Pixels { data, width: w, height: h });
}

/// DS_SCREENSHOT=n1,n2,... writes the displayed image of those flips as PNG.
/// Screenshot requested through the control channel (file name stem).
pub static SHOT_REQUEST: Mutex<Option<String>> = Mutex::new(None);

fn maybe_screenshot(b: &DisplayBuffer, flip: u64) {
    let requested = SHOT_REQUEST.lock().take();
    let listed = std::env::var("DS_SCREENSHOT").is_ok_and(|list| list.split(',').any(|s| s.trim().parse::<u64>().ok() == Some(flip)));
    if requested.is_none() && !listed {
        return;
    }
    // No GPU renderer exists yet: the display buffer contents are not produced
    // by the title's GPU work, so there is nothing truthful to capture.
    crate::warn!("screenshot of flip {flip} ({:?}) unavailable: GPU rendering is not implemented yet (buffer {:#x})", requested, b.addr);
}
