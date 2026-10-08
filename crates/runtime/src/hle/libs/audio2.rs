//! libSceAudioOut2: PS5 audio output contexts, users and ports.
//!
//! Model (interface facts, see docs/PROVENANCE.md): a context mixes "grains"
//! of `num_grains` samples (default 512 at 48 kHz). The title points each
//! port at one grain of PCM with `PortSetAttributes` (attribute 0 = PCM
//! pointer, 1 = per-channel gains), latches the grain with `ContextAdvance`
//! and submits it with `ContextPush`, which blocks while `queue_depth` grains
//! are queued: that blocking is the title's audio clock.
//!
//! Each pushed grain is mixed to stereo f32 here and handed to the host sink
//! (`crate::audio`). Object (3D) ports are mixed to the center (no
//! spatialization yet: explicit stand-in, logged once).

use crate::hle::Registry;
use crate::{reg, sysv};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};

const ERR_INVALID_PARAM: i32 = 0x8026_8001u32 as i32;
const ERR_NOT_READY: i32 = 0x8026_8008u32 as i32;
const ERR_INVALID_PORT: i32 = 0x8026_8009u32 as i32;

const ATTR_PCM: u32 = 0;
const ATTR_GAIN: u32 = 1;

struct Context {
    num_grains: u32,
    queue_depth: u32,
}

struct Port {
    ctx: u32,
    port_type: u16,
    format: u32,
    pcm: u64,
    latched: u64,
    gains: [f32; 16],
}

static NEXT: AtomicU32 = AtomicU32::new(1);
static CONTEXTS: Mutex<Option<HashMap<u32, Context>>> = Mutex::new(None);
static PORTS: Mutex<Option<HashMap<u32, Port>>> = Mutex::new(None);

fn channels(format: u32) -> usize {
    match (format >> 8) & 0xFF {
        0 => 2,
        n => (n as usize).min(16),
    }
}

fn is_object(port_type: u16) -> bool {
    port_type & 0xFF00 == 0x0100
}

/// Mixes one grain of a port into a stereo accumulator.
unsafe fn mix_port(p: &Port, n: usize, out: &mut [f32]) {
    if p.latched == 0 {
        return;
    }
    let ch = channels(p.format);
    let float = p.format & 0x7F == 0;
    let sample = |i: usize| -> f32 {
        unsafe {
            if float {
                *((p.latched as *const f32).add(i))
            } else {
                *((p.latched as *const i16).add(i)) as f32 / 32768.0
            }
        }
    };
    for f in 0..n {
        let base = f * ch;
        let (l, r) = match ch {
            1 => {
                let v = sample(base) * p.gains[0];
                (v * 0.7071, v * 0.7071)
            }
            _ => {
                // Front left/right; other channels (center, LFE, surrounds) fold down.
                let mut l = sample(base) * p.gains[0];
                let mut r = sample(base + 1) * p.gains[1];
                if ch >= 8 {
                    let c = sample(base + 2) * p.gains[2] * 0.7071;
                    let (sl, sr) = (sample(base + 4) * p.gains[4], sample(base + 5) * p.gains[5]);
                    let (bl, br) = (sample(base + 6) * p.gains[6], sample(base + 7) * p.gains[7]);
                    l += c + 0.7071 * (sl + bl);
                    r += c + 0.7071 * (sr + br);
                }
                (l, r)
            }
        };
        out[2 * f] += l;
        out[2 * f + 1] += r;
    }
}

sysv! {
    fn sceAudioOut2Initialize() -> i32 {
        crate::audio::init();
        0
    }
    /// SceAudioOut2ContextParam: {max_ports, max_object_ports, guarantee_object_ports,
    /// queue_depth, num_grains, flags, reserved[10]}.
    fn sceAudioOut2ContextResetParam(p: *mut u32) -> i32 {
        if p.is_null() { return ERR_INVALID_PARAM; }
        std::ptr::write_bytes(p as *mut u8, 0, 64);
        *p = 256;
        *p.add(1) = 256;
        *p.add(3) = 4;
        *p.add(4) = 512;
        *p.add(5) = 1;
        0
    }
    fn sceAudioOut2ContextQueryMemory(p: *const u32, size: *mut u64) -> i32 {
        if p.is_null() || size.is_null() { return ERR_INVALID_PARAM; }
        let depth = if *p.add(3) == 0 { 4 } else { *p.add(3) };
        *size = 0x10000 + depth as u64 * 0x590;
        0
    }
    fn sceAudioOut2ContextCreate(p: *const u32, mem: u64, mem_size: u64, out: *mut u32) -> i32 {
        if p.is_null() || out.is_null() { return ERR_INVALID_PARAM; }
        let depth = if *p.add(3) == 0 { 4 } else { *p.add(3) };
        let grains = if *p.add(4) == 0 { 512 } else { *p.add(4) };
        let h = NEXT.fetch_add(1, Ordering::Relaxed);
        CONTEXTS.lock().get_or_insert_with(HashMap::new).insert(h, Context { num_grains: grains, queue_depth: depth });
        crate::info!("sceAudioOut2ContextCreate -> {h} ({grains} samples/grain, queue depth {depth}, max ports {}, objects {})", *p, *p.add(1));
        *out = h;
        0
    }
    fn sceAudioOut2ContextDestroy(ctx: u32) -> i32 {
        CONTEXTS.lock().get_or_insert_with(HashMap::new).remove(&ctx);
        PORTS.lock().get_or_insert_with(HashMap::new).retain(|_, p| p.ctx != ctx);
        0
    }
    fn sceAudioOut2ContextSetAttributes(ctx: u32, attrs: u64, num: u32) -> i32 { 0 }
    /// Latches the PCM pointers set for this grain.
    fn sceAudioOut2ContextAdvance(ctx: u32) -> i32 {
        for p in PORTS.lock().get_or_insert_with(HashMap::new).values_mut().filter(|p| p.ctx == ctx) {
            p.latched = p.pcm;
        }
        0
    }
    fn sceAudioOut2ContextPush(ctx: u32, blocking: u32) -> i32 {
        let Some((n, depth)) = CONTEXTS.lock().get_or_insert_with(HashMap::new).get(&ctx).map(|c| (c.num_grains as usize, c.queue_depth)) else {
            return ERR_INVALID_PARAM;
        };
        if blocking == 0 && crate::audio::queued_grains() >= depth as usize {
            return ERR_NOT_READY;
        }
        let mut mix = vec![0f32; 2 * n];
        {
            let ports = PORTS.lock();
            for p in ports.as_ref().into_iter().flat_map(|m| m.values()).filter(|p| p.ctx == ctx) {
                if is_object(p.port_type) {
                    crate::hle::stub_log("AudioOut2 object (3D) ports are mixed without spatialization", 0);
                }
                mix_port(p, n, &mut mix);
            }
        }
        // Blocks while `depth` grains are queued (the title's pacing).
        crate::audio::push_grain(mix, depth as usize);
        0
    }
    fn sceAudioOut2ContextGetQueueLevel(ctx: u32, level: *mut u32, available: *mut u32) -> i32 {
        let depth = CONTEXTS.lock().get_or_insert_with(HashMap::new).get(&ctx).map(|c| c.queue_depth).unwrap_or(4);
        let q = (crate::audio::queued_grains() as u32).min(depth);
        if !level.is_null() { *level = q; }
        if !available.is_null() { *available = depth - q; }
        0
    }
    fn sceAudioOut2UserCreate(user: u32, out: *mut u32) -> i32 {
        if out.is_null() { return ERR_INVALID_PARAM; }
        *out = 0x100 + user;
        0
    }
    fn sceAudioOut2UserDestroy(h: u32) -> i32 { 0 }
    /// SceAudioOut2PortParam: {u16 port_type, u16 pad, u32 data_format,
    /// u32 sampling_freq, u32 flags, u32 user_handle, ...}.
    fn sceAudioOut2PortCreate(ctx: u32, p: *const u8, out: *mut u32) -> i32 {
        if p.is_null() || out.is_null() { return ERR_INVALID_PARAM; }
        let port_type = *(p as *const u16);
        let format = *(p.add(4) as *const u32);
        let freq = *(p.add(8) as *const u32);
        if freq != 48000 {
            crate::warn!("sceAudioOut2PortCreate: unsupported sampling frequency {freq}");
            return ERR_INVALID_PARAM;
        }
        if !CONTEXTS.lock().get_or_insert_with(HashMap::new).contains_key(&ctx) { return ERR_INVALID_PARAM; }
        let h = NEXT.fetch_add(1, Ordering::Relaxed);
        PORTS.lock().get_or_insert_with(HashMap::new).insert(h, Port { ctx, port_type, format, pcm: 0, latched: 0, gains: [1.0; 16] });
        crate::debug!("sceAudioOut2PortCreate ctx {ctx} type {port_type:#x} format {format:#x} ({} ch) -> {h}", channels(format));
        *out = h;
        0
    }
    fn sceAudioOut2PortDestroy(port: u32) -> i32 {
        PORTS.lock().get_or_insert_with(HashMap::new).remove(&port);
        0
    }
    /// SceAudioOut2Attribute: {u32 id, i32 reserved, const void* value, u64 value_size}.
    fn sceAudioOut2PortSetAttributes(port: u32, attrs: *const u8, num: u32) -> i32 {
        if num != 0 && attrs.is_null() { return ERR_INVALID_PARAM; }
        let mut g = PORTS.lock();
        let Some(p) = g.get_or_insert_with(HashMap::new).get_mut(&port) else { return ERR_INVALID_PORT };
        for i in 0..num as usize {
            let a = attrs.add(i * 24);
            let id = *(a as *const u32);
            let value = *(a.add(8) as *const u64);
            let size = *(a.add(16) as *const u64);
            match id {
                ATTR_PCM => p.pcm = if value == 0 { 0 } else { *(value as *const u64) },
                ATTR_GAIN if value != 0 => {
                    let n = (size as usize / 4).min(16);
                    for c in 0..n { p.gains[c] = *((value as *const f32).add(c)); }
                }
                _ => {}
            }
        }
        0
    }
    /// SceAudioOut2PortState: {u16 output, u8 channels, u8, i16 volume, u16 reroute, u32 flags, ...}.
    fn sceAudioOut2PortGetState(port: u32, st: *mut u8) -> i32 {
        if st.is_null() { return ERR_INVALID_PARAM; }
        let ch = PORTS.lock().get_or_insert_with(HashMap::new).get(&port).map(|p| channels(p.format)).unwrap_or(2);
        std::ptr::write_bytes(st, 0, 0x40);
        *(st as *mut u16) = 1; // primary output connected
        *st.add(2) = ch as u8;
        *(st.add(4) as *mut i16) = 127;
        0
    }
    /// Stereo speakers at +-30 degrees (the host output is mixed to stereo).
    fn sceAudioOut2GetSpeakerInfo(info: *mut u8, flags: u32) -> i32 {
        if info.is_null() { return ERR_INVALID_PARAM; }
        std::ptr::write_bytes(info, 0, 16 + 64);
        *(info.add(4) as *mut u32) = 0x3;
        *(info.add(16) as *mut i16) = -30;
        *(info.add(20) as *mut i16) = 30;
        0
    }
    fn sceAudioOut2GetSystemState(st: *mut u8) -> i32 {
        if st.is_null() { return ERR_INVALID_PARAM; }
        std::ptr::write_bytes(st, 0, 64);
        0
    }
}

pub fn register(r: &mut Registry) {
    reg!(r, "libSceAudioOut2", [
        sceAudioOut2Initialize, sceAudioOut2ContextResetParam, sceAudioOut2ContextQueryMemory,
        sceAudioOut2ContextCreate, sceAudioOut2ContextDestroy, sceAudioOut2ContextSetAttributes,
        sceAudioOut2ContextAdvance, sceAudioOut2ContextPush, sceAudioOut2ContextGetQueueLevel,
        sceAudioOut2UserCreate, sceAudioOut2UserDestroy, sceAudioOut2PortCreate, sceAudioOut2PortDestroy,
        sceAudioOut2PortSetAttributes, sceAudioOut2PortGetState, sceAudioOut2GetSpeakerInfo, sceAudioOut2GetSystemState,
    ]);
}
