//! Raw mouse look in the title's gameplay camera (PPSA01341 01.007.000;
//! FINDINGS.md 2026-10-21 evening).
//!
//! The camera system (its update writes the camera's frame at
//! eboot+0x79793b) drives a rig (eboot+0x181b00; mode at +0x84) whose
//! controller at +0x60 runs the gameplay camera: eboot+0x191d70 builds a work
//! context and eboot+0x196d40 runs its stages on it. The stick reader
//! (eboot+0x18bd60) stores the camera stick in the context: +0x120 drives the
//! pitch, +0x124 the yaw, +0x130 is their magnitude and +0x118 "the stick
//! moved this frame", which makes the stages treat the frame as manual
//! camera input (the angles taken against the pivot at +0x100, the auto
//! camera's timers reset). eboot+0x18c590 derives the controller's pitch
//! (+0x130, clamped between +0x1D8 and +0x1D4) and yaw (+0x134) from its
//! positions each frame; eboot+0x18cf00 adds the stick times the frame time
//! times the camera speeds (+0x1EC pitch, +0x1F0 yaw; faster after a ramp at
//! full horizontal tilt), wraps the yaw and clamps the pitch, and the stages
//! after it place the camera from those angles.
//!
//! Raw mouse look (`mouse_look_raw`, `input`): `stick_hook` (in the stick
//! reader) gives the title a negligible stick when mouse motion waits and the
//! stick rests, so the frame counts as manual input, and `apply_hook` (at the
//! end of eboot+0x18cf00) adds the mouse counts to the angles: linear, without
//! smoothing, acceleration or dead zone, independent of the frame rate, the
//! same angle per count on both axes (`input::camera_radians_per_count`).
//! Only while the controller takes stick input freely (it reads a device,
//! +0x2DC, and the context allows manual control, +0x74 bit 9); otherwise the
//! mouse moves the stick as before (`live`).

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// The hooks are in place (`titlefix::apply`).
static READY: AtomicBool = AtomicBool::new(false);
/// The last camera update took stick input freely.
static FREE: AtomicBool = AtomicBool::new(false);
/// When the stick reader last ran (ms since start; 0: never).
static LAST_MS: AtomicU64 = AtomicU64::new(0);
/// Controller fields: pitch and yaw (radians), the pitch limits, manual
/// input's pitch, the base pitch and yaw speeds (radians per second at full
/// tilt), the fast yaw speed, the input device flag.
const PITCH: u64 = 0x130;
const YAW: u64 = 0x134;
const MANUAL_PITCH: u64 = 0x140;
const PITCH_MAX: u64 = 0x1D4;
const PITCH_MIN: u64 = 0x1D8;
const PITCH_SPEED: u64 = 0x1EC;
const YAW_SPEED: u64 = 0x1F0;
const YAW_SPEED_FAST: u64 = 0x208;
const DEVICE: u64 = 0x2DC;
/// Context fields: flags (bit 9: manual control), "the stick moved".
const CTX_FLAGS: u64 = 0x74;
const CTX_STICK_MOVED: u64 = 0x118;
/// A tilt the title takes as input but that turns nothing measurable
/// (1e-20 of full tilt; a normal float).
const TINY_TILT: u32 = 0x1E3C_E508;
/// [camera updates, updates with free stick input, updates given the tiny
/// tilt, updates that took mouse motion, counts x, counts y (absolute)].
static STATS: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];
/// The controllers seen (the first few logged).
static SEEN: parking_lot::Mutex<Vec<u64>> = parking_lot::Mutex::new(Vec::new());

/// Called once both hooks are installed.
pub fn enable() {
    READY.store(true, Ordering::Release);
}

fn now_ms() -> u64 {
    static T0: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    T0.get_or_init(std::time::Instant::now).elapsed().as_millis() as u64 + 1
}

/// The gameplay camera takes stick input freely (its last update, within
/// 250 ms): mouse motion goes to `apply_hook` rather than to the stick.
pub fn live() -> bool {
    let t = LAST_MS.load(Ordering::Relaxed);
    READY.load(Ordering::Acquire) && FREE.load(Ordering::Relaxed) && t != 0 && now_ms().saturating_sub(t) < 250
}

fn rd32(a: u64) -> u32 {
    unsafe { (a as *const u32).read_volatile() }
}

fn rdf(a: u64) -> f32 {
    unsafe { (a as *const f32).read_volatile() }
}

/// In the stick reader (eboot+0x18bd99, after it loaded the stick: rdx its
/// two axes as f32 bits, rbx the context).
pub extern "sysv64" fn stick_hook(regs: *const u64, _arg: u32) {
    let regs = regs as *mut u64;
    let ctx = unsafe { *regs.add(3) };
    let ctl = unsafe { (ctx as *const u64).read_volatile() };
    STATS[0].fetch_add(1, Ordering::Relaxed);
    let free = ctl != 0 && unsafe { ((ctl + DEVICE) as *const u8).read_volatile() } == 1 && rd32(ctx + CTX_FLAGS) & 0x200 != 0;
    FREE.store(free, Ordering::Relaxed);
    LAST_MS.store(now_ms(), Ordering::Relaxed);
    if !free {
        return;
    }
    STATS[1].fetch_add(1, Ordering::Relaxed);
    {
        let mut g = SEEN.lock();
        if !g.contains(&ctl) && g.len() < 16 {
            g.push(ctl);
            crate::info!(
                "camhook: controller {ctl:#x}: pitch {:.4} yaw {:.4} (limits {:.4} {:.4}), speeds pitch {:.4} yaw {:.4} fast yaw {:.4} rad/s",
                rdf(ctl + PITCH),
                rdf(ctl + YAW),
                rdf(ctl + PITCH_MIN),
                rdf(ctl + PITCH_MAX),
                rdf(ctl + PITCH_SPEED),
                rdf(ctl + YAW_SPEED),
                rdf(ctl + YAW_SPEED_FAST)
            );
        }
    }
    let stick = unsafe { *regs.add(2) };
    if stick & 0x7FFF_FFFF_7FFF_FFFF == 0 && crate::input::camera_motion_pending() {
        // x (the pitch's axis) tiny, y unchanged.
        unsafe { *regs.add(2) = (stick & 0xFFFF_FFFF_0000_0000) | TINY_TILT as u64 };
        STATS[2].fetch_add(1, Ordering::Relaxed);
    }
}

/// `DS_CAMDEBUG=1`: frames left to log after manual input (`apply_hook`).
static DEBUG_LEFT: AtomicU64 = AtomicU64::new(0);

fn debug() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("DS_CAMDEBUG").is_some())
}

/// Diagnosis line: the frame's angles and the controller's position vectors
/// (x, z of +0xD0, +0xE0, +0x100).
fn debug_line(ctx: u64, ctl: u64, what: &str) {
    let v = |o: u64| format!("({:.3} {:.3})", rdf(ctl + o), rdf(ctl + o + 8));
    crate::info!(
        "camhook dbg: {what} manual {} flags {:#x} pitch {:.5} yaw {:.5} D0 {} E0 {} 100 {}",
        unsafe { ((ctx + CTX_STICK_MOVED) as *const u8).read_volatile() },
        rd32(ctx + CTX_FLAGS),
        rdf(ctl + PITCH),
        rdf(ctl + YAW),
        v(0xD0),
        v(0xE0),
        v(0x100)
    );
}

/// At the end of the stick stage (eboot+0x18d1c4, every path; rbx the
/// context): the mouse counts as angles, when the frame took manual input
/// freely.
pub extern "sysv64" fn apply_hook(regs: *const u64, _arg: u32) {
    let ctx = unsafe { *regs.add(3) };
    let ctl = unsafe { (ctx as *const u64).read_volatile() };
    if ctl == 0 {
        return;
    }
    let manual = unsafe { ((ctx + CTX_STICK_MOVED) as *const u8).read_volatile() } == 1;
    if debug() {
        if manual {
            DEBUG_LEFT.store(12, Ordering::Relaxed);
        }
        if DEBUG_LEFT.load(Ordering::Relaxed) > 0 {
            DEBUG_LEFT.fetch_sub(1, Ordering::Relaxed);
            debug_line(ctx, ctl, "stage end");
        }
    }
    if rd32(ctx + CTX_FLAGS) & 0x200 == 0 || !manual {
        return;
    }
    let Some((dx, dy)) = crate::input::take_camera_motion() else { return };
    if dx == 0 && dy == 0 {
        return;
    }
    if debug() {
        crate::info!("camhook dbg: mouse {dx} {dy}");
    }
    STATS[3].fetch_add(1, Ordering::Relaxed);
    STATS[4].fetch_add(dx.unsigned_abs(), Ordering::Relaxed);
    STATS[5].fetch_add(dy.unsigned_abs(), Ordering::Relaxed);
    let (yaw_per, pitch_per) = crate::input::camera_radians_per_count(rdf(ctl + YAW_SPEED));
    let pi = std::f32::consts::PI;
    let mut yaw = rdf(ctl + YAW) + dx as f32 * yaw_per;
    yaw -= (yaw / (2.0 * pi)).round() * 2.0 * pi;
    let mut pitch = rdf(ctl + PITCH) + dy as f32 * pitch_per;
    let (lo, hi) = (rdf(ctl + PITCH_MIN), rdf(ctl + PITCH_MAX));
    if lo.is_finite() && hi.is_finite() && lo <= hi {
        pitch = pitch.clamp(lo, hi);
    }
    unsafe {
        ((ctl + YAW) as *mut f32).write_volatile(yaw);
        ((ctl + PITCH) as *mut f32).write_volatile(pitch);
        // The stage copied the pitch for manual input before.
        ((ctl + MANUAL_PITCH) as *mut f32).write_volatile(pitch);
    }
    if debug() {
        debug_line(ctx, ctl, "mouse applied");
    }
}

/// Camera rigs seen at their update's entry (`DS_CAMDEBUG=1`): (rig, calls).
static RIGS: parking_lot::Mutex<Vec<(u64, u64)>> = parking_lot::Mutex::new(Vec::new());

/// Diagnosis hook at the camera rig update's entry (rdi the rig).
pub extern "sysv64" fn rig_hook(regs: *const u64, _arg: u32) {
    let rig = unsafe { *regs.add(7) };
    let mut g = RIGS.lock();
    if let Some(r) = g.iter_mut().find(|r| r.0 == rig) {
        r.1 += 1;
    } else if g.len() < 8 {
        g.push((rig, 1));
    }
}

/// The controllers seen and their pitch and yaw now, and with
/// `DS_CAMDEBUG=1` the rigs: mode, controllers (+0x60, +0x68) and their
/// fields +0x124..+0x13c (debug `cam`).
pub fn describe() -> String {
    let q = |a: u64| unsafe { (a as *const u64).read_volatile() };
    let parts: Vec<String> = SEEN.lock().iter().map(|&c| format!("{c:#x} pitch {:.5} yaw {:.5}", rdf(c + PITCH), rdf(c + YAW))).collect();
    let rigs: Vec<String> = RIGS
        .lock()
        .iter()
        .map(|&(r, n)| {
            let ctl = |c: u64| if c == 0 { "0".to_string() } else { format!("{c:#x} [124] {:.4} [128] {:.4} [130] {:.4} [134] {:.4} [138] {:.4} [13c] {:.4}", rdf(c + 0x124), rdf(c + 0x128), rdf(c + 0x130), rdf(c + 0x134), rdf(c + 0x138), rdf(c + 0x13c)) };
            format!("rig {r:#x} ({n} updates) mode {} ctl60 {} ctl68 {}", rd32(r + 0x84), ctl(q(r + 0x60)), ctl(q(r + 0x68)))
        })
        .collect();
    format!("{} (live {}); {}", parts.join(", "), live(), rigs.join("; "))
}

/// Status line.
pub fn status() -> String {
    if !READY.load(Ordering::Acquire) {
        return String::new();
    }
    let v: Vec<u64> = STATS.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    format!(
        "    raw mouse look (camera hooks, {}): {} camera updates, {} with free stick input, {} given a tiny tilt, {} took mouse motion ({} / {} counts x / y); controllers {:x?}\n",
        if crate::input::raw_mouse_look() { "on" } else { "off" },
        v[0],
        v[1],
        v[2],
        v[3],
        v[4],
        v[5],
        SEEN.lock().as_slice()
    )
}
