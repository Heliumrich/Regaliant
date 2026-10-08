//! Audio output (AudioOut), game pad (see `crate::input`), mouse, mic, voice.
//!
//! AudioOut currently paces output in real time without producing sound (the
//! host audio backend comes later); this keeps FMOD's mixer timing correct.

use crate::hle::Registry;
use crate::{fixed, reg, sysv};
use parking_lot::Mutex;
use std::time::{Duration, Instant};
use windows_sys::Win32::UI::Input::XboxController::{XInputSetState, XINPUT_VIBRATION};

// ---------------------------------------------------------------------------
// AudioOut

#[derive(Clone)]
struct Port {
    kind: i32,
    len: u32,
    freq: u32,
    format: u32,
    next: Instant,
    volume: [i32; 8],
}

static PORTS: Mutex<Vec<Option<Port>>> = Mutex::new(Vec::new());

fn port_channels(format: u32) -> u32 {
    match format & 0xFF {
        0 | 3 => 1,
        1 | 4 => 2,
        _ => 8,
    }
}

// ---------------------------------------------------------------------------
// Pad

#[repr(C)]
#[derive(Clone, Copy)]
struct PadData {
    buttons: u32,
    lx: u8,
    ly: u8,
    rx: u8,
    ry: u8,
    l2: u8,
    r2: u8,
    pad0: [u8; 2],
    orientation: [f32; 4],
    acceleration: [f32; 3],
    angular_velocity: [f32; 3],
    touch: [u8; 24],
    connected: u8,
    pad1: [u8; 3],
    timestamp: u64,
    ext: [u8; 16],
    connected_count: u8,
    reserve: [u8; 2],
    unique_len: u8,
    unique: [u8; 12],
}

fn read_pad() -> PadData {
    let mut d: PadData = unsafe { std::mem::zeroed() };
    let st = crate::input::pad_state();
    d.buttons = st.buttons;
    d.lx = st.lx;
    d.ly = st.ly;
    d.rx = st.rx;
    d.ry = st.ry;
    d.l2 = st.l2;
    d.r2 = st.r2;
    d.orientation = [0.0, 0.0, 0.0, 1.0];
    d.acceleration = [0.0, 9.8, 0.0];
    d.connected = 1;
    d.connected_count = 1;
    d.timestamp = crate::hle::kernel::time::process_time_us();
    d
}

sysv! {
    fn sceAudioOutInit() -> i32 { 0 }
    fn sceAudioOutOpen(user: i32, kind: i32, index: i32, len: u32, freq: u32, param: u32) -> i32 {
        if len == 0 || len % 256 != 0 || freq != 48000 { return 0x8026_0008u32 as i32; }
        let mut ports = PORTS.lock();
        let p = Port { kind, len, freq, format: param, next: Instant::now(), volume: [32768; 8] };
        let idx = match ports.iter().position(|p| p.is_none()) { Some(i) => { ports[i] = Some(p); i } None => { ports.push(Some(p)); ports.len() - 1 } };
        crate::info!("sceAudioOutOpen type {kind} len {len} freq {freq} format {:#x} ({} ch) -> port {}", param, port_channels(param), idx + 1);
        idx as i32 + 1
    }
    fn sceAudioOutClose(h: i32) -> i32 {
        let mut ports = PORTS.lock();
        match ports.get_mut((h - 1) as usize) { Some(p @ Some(_)) => { *p = None; 0 } _ => 0x8026_0004u32 as i32 }
    }
    fn sceAudioOutOutput(h: i32, ptr: *const u8) -> i32 {
        let wait = {
            let mut ports = PORTS.lock();
            let Some(Some(p)) = ports.get_mut((h - 1) as usize) else { return 0x8026_0004u32 as i32 };
            let dur = Duration::from_secs_f64(p.len as f64 / p.freq as f64);
            let now = Instant::now();
            let start = if p.next > now { p.next } else { now };
            p.next = start + dur;
            start.saturating_duration_since(now)
        };
        if !wait.is_zero() { std::thread::sleep(wait); }
        0
    }
    fn sceAudioOutSetVolume(h: i32, flags: u32, vol: *const i32) -> i32 {
        let mut ports = PORTS.lock();
        let Some(Some(p)) = ports.get_mut((h - 1) as usize) else { return 0x8026_0004u32 as i32 };
        for i in 0..8 { if flags & (1 << i) != 0 && !vol.is_null() { p.volume[i] = *vol.add(i); } }
        0
    }
    fn sceAudioOutGetPortState(h: i32, st: *mut u8) -> i32 {
        let ports = PORTS.lock();
        let Some(Some(p)) = ports.get((h - 1) as usize) else { return 0x8026_0004u32 as i32 };
        std::ptr::write_bytes(st, 0, 32);
        *(st as *mut u16) = 1; // connected primary
        *st.add(2) = port_channels(p.format).min(8) as u8;
        *(st.add(4) as *mut i16) = 127;
        0
    }

    // ---- Pad
    fn scePadOpen(user: i32, kind: i32, index: i32, param: *const u8) -> i32 {
        if user != super::app::USER_ID { return 0x80920005u32 as i32; }
        crate::info!("scePadOpen user {user} type {kind} index {index} -> 1 (XInput {}; keyboard via window)", if crate::input::xinput_state().is_some() { "connected" } else { "not detected" });
        1
    }
    fn scePadReadState(h: i32, data: *mut PadData) -> i32 {
        if h != 1 || data.is_null() { return 0x80920003u32 as i32; }
        std::ptr::write_unaligned(data, read_pad());
        0
    }
    /// Buffered read: this runtime samples the current state, so one entry is
    /// returned per call (ScePadData is 120 bytes, as the title's buffer confirms).
    fn scePadRead(h: i32, data: *mut PadData, num: i32) -> i32 {
        if h != 1 || data.is_null() || num < 1 { return 0x80920003u32 as i32; }
        std::ptr::write_unaligned(data, read_pad());
        1
    }
    fn scePadGetHandle(user: i32, kind: i32, index: i32) -> i32 {
        if user != super::app::USER_ID { return 0x80920005u32 as i32; }
        1
    }
    /// DualSense light bar / adaptive triggers / motion sensors: accepted; the host
    /// controller (XInput) has no equivalent (logged once).
    fn scePadSetLightBar(h: i32, color: *const u8) -> i32 { crate::hle::stub_log("scePadSetLightBar (no host light bar)", 0); 0 }
    fn scePadSetTriggerEffect(h: i32, param: *const u8) -> i32 { crate::hle::stub_log("scePadSetTriggerEffect (no adaptive triggers on the host pad)", 0); 0 }
    fn scePadSetMotionSensorState(h: i32, enable: i32) -> i32 { crate::hle::stub_log("scePadSetMotionSensorState (no motion sensors mapped)", 0); 0 }
    fn scePadGetControllerInformation(h: i32, info: *mut u8) -> i32 {
        if info.is_null() { return 0x80920001u32 as i32; }
        std::ptr::write_bytes(info, 0, 28);
        *(info as *mut f32) = 44.86;
        *(info.add(4) as *mut u16) = 1920;
        *(info.add(6) as *mut u16) = 943;
        *info.add(8) = 13;
        *info.add(9) = 13;
        *info.add(10) = 0; // local connection
        *info.add(11) = 1; // connected count
        *info.add(12) = 1; // connected
        0
    }
    fn scePadSetVibration(h: i32, param: *const u8) -> i32 {
        if !param.is_null() {
            let mut v = XINPUT_VIBRATION { wLeftMotorSpeed: (*param as u16) * 257, wRightMotorSpeed: (*param.add(1) as u16) * 257 };
            XInputSetState(0, &mut v);
        }
        0
    }
}

fixed! {
    scePadInit => 0,
    scePadClose => 0,
    scePadResetOrientation => 0,
    scePadSetAngularVelocityDeadbandState => 0,
    scePadSetTiltCorrectionState => 0,
    sceMouseInit => 0,
    sceMouseOpen => 0x80DF_0003u32,
    sceMouseClose => 0x80DF_0003u32,
    sceMouseRead => 0x80DF_0003u32,
    sceAudioInOpen => 0x8026_0007u32,
    sceAudioInClose => 0x8026_0007u32,
    sceAudioInInput => 0x8026_0007u32,
    sceVoiceInit => 0,
    sceVoiceEnd => 0,
    sceVoiceStart => 0,
    sceVoiceStop => 0,
    sceVoiceCreatePort => 0x8082_0001u32,
    sceVoiceDeletePort => 0x8082_0001u32,
    sceVoiceConnectIPortToOPort => 0x8082_0001u32,
    sceVoiceDisconnectIPortFromOPort => 0x8082_0001u32,
    sceVoiceGetPortInfo => 0x8082_0001u32,
    sceVoiceReadFromOPort => 0x8082_0001u32,
    sceVoiceWriteToIPort => 0x8082_0001u32,
}

pub fn register(r: &mut Registry) {
    reg!(r, "libSceAudioOut", [
        sceAudioOutInit, sceAudioOutOpen, sceAudioOutClose, sceAudioOutOutput, sceAudioOutSetVolume, sceAudioOutGetPortState,
    ]);
    reg!(r, "libScePad", [
        scePadInit, scePadOpen, scePadClose, scePadReadState, scePadGetControllerInformation, scePadSetVibration,
        scePadRead, scePadGetHandle, scePadSetLightBar, scePadSetTriggerEffect, scePadSetMotionSensorState,
        scePadResetOrientation, scePadSetAngularVelocityDeadbandState, scePadSetTiltCorrectionState,
    ]);
    reg!(r, "libSceMouse", [sceMouseInit, sceMouseOpen, sceMouseClose, sceMouseRead]);
    reg!(r, "libSceAudioIn", [sceAudioInOpen, sceAudioInClose, sceAudioInInput]);
    reg!(r, "libSceVoice", [
        sceVoiceInit, sceVoiceEnd, sceVoiceStart, sceVoiceStop, sceVoiceCreatePort, sceVoiceDeletePort,
        sceVoiceConnectIPortToOPort, sceVoiceDisconnectIPortFromOPort, sceVoiceGetPortInfo, sceVoiceReadFromOPort,
        sceVoiceWriteToIPort,
    ]);
}
