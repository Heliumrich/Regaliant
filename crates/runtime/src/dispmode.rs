//! The title's display mode as a render-resolution choice (PPSA01341
//! 01.007.000; FINDINGS.md 2026-10-21 evening).
//!
//! The title's video-out object picks its render and display targets by a
//! mode (+0x3C), set each frame from eboot+0x9fe6b2 through eboot+0x90a600:
//! 3 unless the settings (0x2F13B40) say otherwise (+0xD38, then +0xD30, the
//! options' value: 3 at the gate). eboot+0x90a600 takes the render and
//! output sizes per mode: 1 renders 1600x900 for a 1920x1080 output, 2
//! 1920x1080 for 1920x1080, 3 2560x1440 for 3840x2160, 4 3840x2160 for
//! 3840x2160, 5 2688x1512 for 3840x2160 (with a flag); others 1920x1080. The
//! title starts in mode 2 and switches to 3 when the options load.
//!
//! Two more readers make the same choice each frame: eboot+0xb26030 keeps the
//! effective mode at the settings' +0xD28 and calls eboot+0x934260 when it
//! changes, and eboot+0xde88c0 derives "a 4K output mode" (3-5), the UI's
//! scale (mode 2 held at the display call alone drew the HUD at twice its
//! size, cut off at the right).
//!
//! `DS_DISPLAY_MODE=<1-5>` holds the mode from the first frame: at all three
//! the register holding the options' value (+0xD30, which has the last say)
//! becomes the chosen mode just before its `test`/`cmovne` (`hook_rax` at
//! eboot+0x9fe6a8 and eboot+0xb2606e, `hook_r15` at eboot+0xde88c0). The
//! option itself is left alone (the title saves its options).

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

/// The chosen mode (0: the title's own).
static MODE: AtomicU32 = AtomicU32::new(0);
/// The chosen render size (width << 32 | height; 0: the mode's own).
static RENDER: AtomicU64 = AtomicU64::new(0);
/// Times the render size was set.
static RENDER_SET: AtomicU64 = AtomicU64::new(0);
/// Frames the mode was held.
static HELD: AtomicU64 = AtomicU64::new(0);

/// Render and output sizes of a mode (eboot+0x90a600's table).
pub fn sizes(mode: u32) -> ((u32, u32), (u32, u32)) {
    match mode {
        1 => ((1600, 900), (1920, 1080)),
        3 => ((2560, 1440), (3840, 2160)),
        4 => ((3840, 2160), (3840, 2160)),
        5 => ((2688, 1512), (3840, 2160)),
        _ => ((1920, 1080), (1920, 1080)),
    }
}

/// Reads `DS_DISPLAY_MODE` and `DS_RENDER_SIZE` (called before the hooks are
/// installed).
pub fn init() {
    if let Ok(v) = std::env::var("DS_RENDER_SIZE") {
        let size = v.trim().split_once(['x', 'X']).and_then(|(w, h)| Some((w.trim().parse::<u32>().ok()?, h.trim().parse::<u32>().ok()?)));
        match size {
            Some((w, h)) if (320..=7680).contains(&w) && (180..=4320).contains(&h) => {
                RENDER.store((w as u64) << 32 | h as u64, Ordering::Relaxed);
                crate::info!("render size {w}x{h} (DS_RENDER_SIZE): the title's render targets at that size, its output as the mode has it");
            }
            _ => crate::warn!("DS_RENDER_SIZE={v}: not a size (WxH, 320x180 to 7680x4320); the mode's own is used"),
        }
    }
    let Ok(v) = std::env::var("DS_DISPLAY_MODE") else { return };
    match v.trim().parse::<u32>() {
        Ok(m @ 1..=5) => {
            let ((rw, rh), (ow, oh)) = sizes(m);
            MODE.store(m, Ordering::Relaxed);
            crate::info!("display mode {m} held (DS_DISPLAY_MODE): renders {rw}x{rh} for a {ow}x{oh} output");
        }
        Ok(0) => {}
        _ => crate::warn!("DS_DISPLAY_MODE={v}: not a mode (1-5); the title's own is used"),
    }
}

/// Before a mode choice's last `cmovne`: the register holding the options'
/// value (`regs` index: rax 0, r15 15) becomes the chosen mode.
fn hold(regs: *const u64, reg: usize) {
    let m = MODE.load(Ordering::Relaxed);
    if m == 0 {
        return;
    }
    unsafe { *(regs as *mut u64).add(reg) = m as u64 };
    HELD.fetch_add(1, Ordering::Relaxed);
}

/// eboot+0x9fe6a8 (the display mode call) and eboot+0xb2606e (+0xD28).
pub extern "sysv64" fn hook_rax(regs: *const u64, _arg: u32) {
    hold(regs, 0);
}

/// eboot+0xde88c0 (the UI's 4K output test).
pub extern "sysv64" fn hook_r15(regs: *const u64, _arg: u32) {
    hold(regs, 15);
}

/// eboot+0x90a6fc, in set display mode after it built the sizes it hands the
/// render resolution manager (eboot+0x90c430): the render size at
/// [rbp-0x4C] (width) and [rbp-0x48] (height) becomes the chosen one; the
/// output size ([rbp-0x3C]) stays the mode's.
pub extern "sysv64" fn hook_render_size(regs: *const u64, _arg: u32) {
    let r = RENDER.load(Ordering::Relaxed);
    if r == 0 {
        return;
    }
    let rbp = unsafe { *regs.add(5) };
    let (w, h) = ((r >> 32) as u32, r as u32);
    let at = rbp.wrapping_sub(0x4C);
    let old = unsafe { ((at as *const u32).read_volatile(), ((at + 4) as *const u32).read_volatile()) };
    unsafe {
        (at as *mut u32).write_volatile(w);
        ((at + 4) as *mut u32).write_volatile(h);
    }
    if RENDER_SET.fetch_add(1, Ordering::Relaxed) < 8 {
        crate::info!("render size {w}x{h} set (the mode's was {}x{})", old.0, old.1);
    }
}

/// eboot+0xa4f2e9, after the frame's copy of the settings (memcpy of 0xD8A
/// bytes to r14+0x390): the copy's option (+0xD30) becomes the chosen mode.
pub extern "sysv64" fn hook_snapshot(regs: *const u64, _arg: u32) {
    let m = MODE.load(Ordering::Relaxed);
    if m == 0 {
        return;
    }
    let r14 = unsafe { *regs.add(14) };
    if r14 != 0 {
        unsafe { ((r14 + 0x390 + 0xD30) as *mut u32).write_volatile(m) };
    }
}

/// Status line.
pub fn status() -> String {
    let (m, r) = (MODE.load(Ordering::Relaxed), RENDER.load(Ordering::Relaxed));
    let mut s = String::new();
    if m != 0 {
        s += &format!("    display mode {m} held ({} choices)\n", HELD.load(Ordering::Relaxed));
    }
    if r != 0 {
        s += &format!("    render size {}x{} set {} times\n", r >> 32, r as u32, RENDER_SET.load(Ordering::Relaxed));
    }
    s
}
