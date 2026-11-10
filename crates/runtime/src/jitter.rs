//! The title's TAA jitter (PPSA01341 01.007.000): the per-view function at eboot+0x9ed8d0
//! returns the projection's sub-pixel offset for a view. It reads the frame's copy of the
//! render settings (`[rdi+0x68]`): with `r_aaForceJitterIndex` (+0xAAC) negative it jitters
//! only while `r_aaMethod` (+0xAA0) is not Off and `r_aaEnableTemporalProjectMatrixJitter`
//! (+0xAA8) is set; its index is that forced one, else the frame counter (`[[rdi+0x10]+0x78]`),
//! modulo the phase count the caller passes (`edx`, at most 47; 0 means 47); its points are the
//! Halton (2, 3) sequence from the first point on (a 47-entry table it fills on first use at
//! eboot+0x2f35820, the first ones rounded to four decimals); the offset is
//! `(2 * point - 1) * scale` with the scale pair at `[rsi]` (clip-space units per pixel).
//!
//! A hook at the function's entry repeats that computation and keeps the camera's offset per
//! title frame (its frame counter; the other callers, with 16 phases and world-space scales,
//! are left out). The upscalers ask for the frame whose TAA they replace by the port's flip
//! count (`for_flip`): the two counters differ by a constant, taken from the oldest kept frame
//! at the first request and again whenever no kept frame matches (a TAA recorded again after a
//! dropped stretch finds the same frame). `DS_JITTER_LOG=<n>` logs the first `n` calls.

use parking_lot::Mutex;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};

/// Module-relative addresses (01.007.000).
const TABLE: u64 = 0x2f35820;
const TABLE_GUARD: u64 = 0x2f35998;
const ENTRIES: u64 = 47;

/// One jittered view: the offset in pixels (x right, y as the title's clip space: up), the
/// phase index and count.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Jitter {
    pub x: f32,
    pub y: f32,
    pub index: u32,
    pub phases: u32,
    /// The scale pair the title passed (its clip-space units per pixel, diagnosis).
    pub scale: (f32, f32),
    /// The title's frame counter when the view was made.
    pub frame: u64,
}

static QUEUE: Mutex<VecDeque<Jitter>> = Mutex::new(VecDeque::new());
static CALLS: AtomicU64 = AtomicU64::new(0);
static LOG_LEFT: AtomicU64 = AtomicU64::new(u64::MAX);

fn halton(mut i: u64, b: u64) -> f32 {
    let (mut f, mut r) = (1.0f64, 0.0f64);
    while i > 0 {
        f /= b as f64;
        r += f * (i % b) as f64;
        i /= b;
    }
    r as f32
}

/// The title's point `i` (0-based): its table once filled, else the same Halton point.
fn point(base: u64, i: u64) -> (f32, f32) {
    let filled = base != 0 && unsafe { ((base + TABLE_GUARD) as *const u8).read_volatile() } != 0;
    if filled {
        let p = (base + TABLE + 8 * i) as *const u32;
        let (x, y) = unsafe { (p.read_volatile(), p.add(1).read_volatile()) };
        (f32::from_bits(x), f32::from_bits(y))
    } else {
        (halton(i + 1, 2), halton(i + 1, 3))
    }
}

/// eboot+0x9ed8d0 entry: rdi the view, rsi the scale pair, edx the phase count.
pub extern "sysv64" fn hook(regs: *const u64, _arg: u32) {
    let r = |i: usize| unsafe { *regs.add(i) };
    let (rdx, rsi, rdi) = (r(2) as u32, r(6), r(7));
    let base = crate::pcsettings::convar::base();
    let Some(j) = (unsafe { compute(base, rdi, rsi, rdx) }) else { return };
    // Only the camera's projection: a scale of half a pixel in clip space (other systems call
    // the same function with 16 phases and world-space scales).
    let main = j.scale.0 > 0.0 && j.scale.0 < 0.01 && j.scale.1 > 0.0 && j.scale.1 < 0.01;
    if !main {
        return;
    }
    let mut q = QUEUE.lock();
    // (One camera view a frame: a second call in the same frame replaces the first.)
    if q.back().is_some_and(|l| l.frame == j.frame) {
        q.pop_back();
    }
    q.push_back(j);
    while q.len() > 16 {
        q.pop_front();
    }
    drop(q);
    let n = CALLS.fetch_add(1, Ordering::Relaxed);
    let mut left = LOG_LEFT.load(Ordering::Relaxed);
    if left == u64::MAX {
        left = std::env::var("DS_JITTER_LOG").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(0);
        LOG_LEFT.store(left, Ordering::Relaxed);
    }
    if left > 0 {
        LOG_LEFT.store(left - 1, Ordering::Relaxed);
        let caller = unsafe { *(r(4) as *const u64) };
        crate::info!("jitter: call {n} from eboot+{:#x}: index {} of {} -> {:+.4}, {:+.4} px (scale {:.6}, {:.6})", caller.wrapping_sub(base), j.index, j.phases, j.x, j.y, j.scale.0, j.scale.1);
    }
}

/// The function's result for these arguments, in pixels; None where it returns no jitter.
unsafe fn compute(base: u64, rdi: u64, rsi: u64, edx: u32) -> Option<Jitter> {
    unsafe {
        let settings = ((rdi + 0x68) as *const u64).read_volatile();
        if settings == 0 {
            return None;
        }
        let forced = ((settings + 0xaac) as *const i32).read_volatile();
        if forced < 0 {
            let aa = ((settings + 0xaa0) as *const i32).read_volatile();
            let on = ((settings + 0xaa8) as *const u8).read_volatile();
            if aa == 0 || on != 1 {
                return None;
            }
        }
        let phases = if edx == 0 { ENTRIES } else { (edx as u64).min(ENTRIES) };
        let frame_obj = ((rdi + 0x10) as *const u64).read_volatile();
        if frame_obj == 0 {
            return None;
        }
        let frame = ((frame_obj + 0x78) as *const u32).read_volatile() as u64;
        let index = if forced >= 0 { forced as u64 } else { frame };
        let i = index % phases;
        let (px, py) = point(base, i);
        let (sx, sy) = (f32::from_bits((rsi as *const u32).read_volatile()), f32::from_bits(((rsi + 4) as *const u32).read_volatile()));
        // Clip space spans 2 units over the image: with the scale a pixel's width in clip
        // space halved (1 / size), the offset in pixels is (2p - 1) / 2 (the logged scale tells).
        Some(Jitter { x: px - 0.5, y: py - 0.5, index: i as u32, phases: phases as u32, scale: (sx, sy), frame })
    }
}

/// The title's frame counter minus the port's flip count (i64::MIN: not known yet).
static OFFSET: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(i64::MIN);
static RECALIBRATED: AtomicU64 = AtomicU64::new(0);

/// The jitter of the frame whose TAA is recorded after `flips` flips (None: no jittered frame
/// kept, as with the title's TAA off).
pub fn for_flip(flips: u64) -> Option<Jitter> {
    let mut q = QUEUE.lock();
    let off = OFFSET.load(Ordering::Relaxed);
    let want = |off: i64| (flips as i64).wrapping_add(off);
    if off != i64::MIN {
        if let Some(j) = q.iter().find(|j| j.frame as i64 == want(off)).copied() {
            // (Frames older than the one before it are done.)
            while q.front().is_some_and(|f| (f.frame as i64) < want(off) - 1) {
                q.pop_front();
            }
            return Some(j);
        }
    }
    // The oldest kept frame is this one: the counters' difference taken from it.
    let j = *q.front()?;
    OFFSET.store(j.frame as i64 - flips as i64, Ordering::Relaxed);
    let n = RECALIBRATED.fetch_add(1, Ordering::Relaxed);
    if n < 8 {
        crate::info!("jitter: title frame {} paired with flip {flips} ({} kept)", j.frame, q.len());
    }
    Some(j)
}

/// Views jittered so far (diagnosis).
pub fn calls() -> u64 {
    CALLS.load(Ordering::Relaxed)
}
