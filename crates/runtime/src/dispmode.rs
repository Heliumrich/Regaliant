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
//! The output size (experiment, `DS_OUTPUT_SIZE=<w>x<h>`): modes 3-5 output
//! 3840x2160, a size that appears at five places of the video-out object
//! and one of the UI: the sizes handed to the render resolution manager
//! (+0x14 of the size block built by eboot+0x90a600 and by the constructor
//! eboot+0x90ad80; +0x0C, 3840x2160 in every mode, stays: the largest size),
//! the two display targets the constructor makes (eboot+0x90af74,
//! eboot+0x90b061; their 1920x1080 views for modes 1-2 stay), the current
//! mode's sizes as the getter eboot+0x90ab90 reports them from its per-mode
//! tables (the UI makes its target from them, eboot+0xdedc59), and the UI's
//! canvas (eboot+0xde85a7: 3840x2160 for modes 3-5, else 1920x1080, unless
//! an override is set). Each is replaced where it is 3840x2160.
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
/// The chosen output size for modes 3-5 (width << 32 | height; 0: the
/// title's 3840x2160).
static OUTPUT: AtomicU64 = AtomicU64::new(0);
/// Times the output size was set: mode change, constructor, targets, UI
/// canvas, size getter.
static OUTPUT_SET: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];

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
    if let Ok(v) = std::env::var("DS_OUTPUT_SIZE") {
        let size = v.trim().split_once(['x', 'X']).and_then(|(w, h)| Some((w.trim().parse::<u32>().ok()?, h.trim().parse::<u32>().ok()?)));
        match size {
            Some((w, h)) if (1920..=3840).contains(&w) && (1080..=2160).contains(&h) => {
                OUTPUT.store((w as u64) << 32 | h as u64, Ordering::Relaxed);
                crate::info!("output size {w}x{h} (DS_OUTPUT_SIZE, experiment): the title's 3840x2160 output, display targets and UI canvas at that size");
            }
            _ => crate::warn!("DS_OUTPUT_SIZE={v}: not a size (WxH, 1920x1080 to 3840x2160); the title's own is used"),
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

/// The packed size at `at` (two u32) becomes the chosen output size where it
/// is 3840x2160 (`site`: the OUTPUT_SET counter).
fn swap_output(at: u64, site: usize) {
    let o = OUTPUT.load(Ordering::Relaxed);
    if o == 0 || at == 0 {
        return;
    }
    let p = at as *mut u32;
    unsafe {
        if (p.read_volatile(), p.add(1).read_volatile()) == (3840, 2160) {
            p.write_volatile((o >> 32) as u32);
            p.add(1).write_volatile(o as u32);
            if OUTPUT_SET[site].fetch_add(1, Ordering::Relaxed) < 4 {
                crate::info!("output size {}x{} set ({})", o >> 32, o as u32, ["mode change", "constructor", "display target", "UI canvas"][site]);
            }
        }
    }
}

/// eboot+0x90a6fc, in set display mode after it built the sizes it hands the
/// render resolution manager (eboot+0x90c430): the render size at
/// [rbp-0x4C] (width) and [rbp-0x48] (height) becomes the chosen one; the
/// output size ([rbp-0x3C]) the chosen one where it is 3840x2160.
pub extern "sysv64" fn hook_render_size(regs: *const u64, _arg: u32) {
    let rbp = unsafe { *regs.add(5) };
    swap_output(rbp.wrapping_sub(0x3C), 0);
    let r = RENDER.load(Ordering::Relaxed);
    if r == 0 {
        return;
    }
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

/// eboot+0x90b278, in the video-out constructor before it hands its size
/// block (r15) to eboot+0x91c680 and the render resolution manager: the
/// output size (+0x14).
pub extern "sysv64" fn hook_ctor_sizes(regs: *const u64, _arg: u32) {
    let r15 = unsafe { *regs.add(15) };
    swap_output(r15.wrapping_add(0x14), 1);
}

/// eboot+0x90af7e and eboot+0x90b06b, the constructor's display targets
/// (edi width, esi height): the chosen output size.
pub extern "sysv64" fn hook_targets(regs: *const u64, _arg: u32) {
    let o = OUTPUT.load(Ordering::Relaxed);
    let (w, h) = unsafe { (*regs.add(7) as u32, *regs.add(6) as u32) };
    if o == 0 || (w, h) != (3840, 2160) {
        return;
    }
    unsafe {
        *(regs as *mut u64).add(7) = o >> 32;
        *(regs as *mut u64).add(6) = o & 0xFFFF_FFFF;
    }
    if OUTPUT_SET[2].fetch_add(1, Ordering::Relaxed) < 4 {
        crate::info!("output size {}x{} set (display target)", o >> 32, o as u32);
    }
}

/// eboot+0x90ac57, in the current mode's size getter (eboot+0x90ab90)
/// before it stores the output size (r9d width, r10d height).
pub extern "sysv64" fn hook_size_getter(regs: *const u64, _arg: u32) {
    let o = OUTPUT.load(Ordering::Relaxed);
    let (w, h) = unsafe { (*regs.add(9) as u32, *regs.add(10) as u32) };
    if o == 0 || (w, h) != (3840, 2160) {
        return;
    }
    unsafe {
        *(regs as *mut u64).add(9) = o >> 32;
        *(regs as *mut u64).add(10) = o & 0xFFFF_FFFF;
    }
    if OUTPUT_SET[4].fetch_add(1, Ordering::Relaxed) < 4 {
        crate::info!("output size {}x{} set (size getter)", o >> 32, o as u32);
    }
}

/// eboot+0xde85db, the UI's canvas size (two f32 at [rbp-0x30]) before its
/// layout pass: the chosen output size where it is 3840x2160.
pub extern "sysv64" fn hook_ui_canvas(regs: *const u64, _arg: u32) {
    let o = OUTPUT.load(Ordering::Relaxed);
    if o == 0 {
        return;
    }
    let p = unsafe { *regs.add(5) }.wrapping_sub(0x30) as *mut f32;
    unsafe {
        if (p.read_volatile(), p.add(1).read_volatile()) == (3840.0, 2160.0) {
            p.write_volatile((o >> 32) as f32);
            p.add(1).write_volatile(o as u32 as f32);
            if OUTPUT_SET[3].fetch_add(1, Ordering::Relaxed) < 4 {
                crate::info!("output size {}x{} set (UI canvas)", o >> 32, o as u32);
            }
        }
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
    let o = OUTPUT.load(Ordering::Relaxed);
    if o != 0 {
        let n: Vec<u64> = OUTPUT_SET.iter().map(|c| c.load(Ordering::Relaxed)).collect();
        s += &format!("    output size {}x{} set: mode changes {}, constructor {}, display targets {}, UI canvas {}, size getter {}\n", o >> 32, o as u32, n[0], n[1], n[2], n[3], n[4]);
    }
    s
}
