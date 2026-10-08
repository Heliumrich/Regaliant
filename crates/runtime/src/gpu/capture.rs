//! Frame capture: the GPU state of every draw and dispatch submitted between
//! two flips, written to `<run dir>/capture-<flip>.txt`, plus each distinct
//! shader program to `<run dir>/shaders/<address>.bin`. Armed with the
//! `capture` control command or `DS_CAPTURE=<flip number>`.

use super::cp::{Cp, Space};
use parking_lot::Mutex;
use std::collections::BTreeSet;
use std::fmt::Write;
use std::sync::atomic::{AtomicU64, Ordering};

/// Flip number at which to start capturing (u64::MAX = not armed).
static ARMED: AtomicU64 = AtomicU64::new(u64::MAX);

struct Capture {
    flip: u64,
    text: String,
    shaders: BTreeSet<u64>,
    events: u64,
}

static ACTIVE: Mutex<Option<Capture>> = Mutex::new(None);

pub fn init_from_env() {
    if let Some(n) = std::env::var("DS_CAPTURE").ok().and_then(|v| v.trim().parse::<u64>().ok()) {
        ARMED.store(n, Ordering::SeqCst);
        crate::info!("capture armed for flip {n}");
    }
}

/// Arms a capture of the frame after the next flip.
pub fn arm_next() {
    ARMED.store(super::FRAMES.load(Ordering::SeqCst) + 1, Ordering::SeqCst);
}

pub fn capturing() -> bool {
    ACTIVE.try_lock().is_some_and(|g| g.is_some())
}

/// Called by the graphics queue for each flip packet (`flip` = flips so far).
pub fn on_flip(flip: u64) {
    let mut g = ACTIVE.lock();
    if let Some(c) = g.take() {
        finish(c);
        return;
    }
    let armed = ARMED.load(Ordering::SeqCst);
    if armed != u64::MAX && flip >= armed {
        ARMED.store(u64::MAX, Ordering::SeqCst);
        *g = Some(Capture { flip, text: format!("# frame after flip {flip}\n"), shaders: BTreeSet::new(), events: 0 });
        crate::info!("capture: recording the frame after flip {flip}");
    }
}

fn finish(c: Capture) {
    let Some(dir) = crate::run_dir() else { return };
    let path = dir.join(format!("capture-{}.txt", c.flip));
    let _ = std::fs::write(&path, &c.text);
    let sdir = dir.join("shaders");
    let _ = std::fs::create_dir_all(&sdir);
    let mut dumped = 0;
    for &a in &c.shaders {
        let size = crate::hle::libs::agc::shader_code_size(a).unwrap_or_else(|| scan_to_endpgm(a));
        if size == 0 || !crate::gpu::cp::guest_readable(a, size as u64) {
            continue;
        }
        let bytes = unsafe { std::slice::from_raw_parts(a as *const u8, size) };
        if std::fs::write(sdir.join(format!("{a:012x}.bin")), bytes).is_ok() {
            dumped += 1;
        }
    }
    crate::info!("capture: {} event(s) -> {} ({dumped} shader binaries in shaders/)", c.events, path.display());
}

/// Fallback size: up to the first s_endpgm (gfx10 0xBF810000), at most 64 KiB.
fn scan_to_endpgm(a: u64) -> usize {
    for i in 0..0x4000u64 {
        if !crate::gpu::cp::guest_readable(a + 4 * i, 4) {
            return 0;
        }
        if unsafe { *((a + 4 * i) as *const u32) } == 0xBF81_0000 {
            return 4 * (i as usize + 1);
        }
    }
    0x10000
}

fn sh_addr(cp: &Cp, lo: u32) -> u64 {
    ((cp.reg(Space::Sh, lo) as u64) << 8) | (((cp.reg(Space::Sh, lo + 1) & 0xFF) as u64) << 40)
}

/// Program, resources and user data of one hardware stage.
fn stage(cp: &Cp, out: &mut String, shaders: &mut BTreeSet<u64>, name: &str, lo: u32, rsrc: u32, user: u32) {
    let a = sh_addr(cp, lo);
    if a == 0 {
        return;
    }
    shaders.insert(a);
    let rsrc1 = cp.reg(Space::Sh, rsrc);
    let rsrc2 = cp.reg(Space::Sh, rsrc + 1);
    let n = (((rsrc2 >> 1) & 0x1F) | (((rsrc2 >> 27) & 1) << 5)).min(32);
    let ud: Vec<String> = (0..n).map(|i| format!("{:08x}", cp.reg(Space::Sh, user + i))).collect();
    let _ = writeln!(out, "    {name} {a:#x} rsrc1 {rsrc1:#010x} rsrc2 {rsrc2:#010x} user[{n}] {}", ud.join(" "));
}

fn f(v: u32) -> f32 {
    f32::from_bits(v)
}

/// Records one draw (`what` names the packet and its arguments).
pub fn draw(cp: &Cp, what: &str) {
    let mut g = ACTIVE.lock();
    let Some(c) = g.as_mut() else { return };
    c.events += 1;
    let mut s = String::new();
    let ctx = |a: u32| cp.reg(Space::Ctx, (a - 0x28000) / 4);
    let _ = writeln!(
        s,
        "draw #{} [{}] {what} prim {} stages {:#x} ge_cntl {:#x}",
        c.events,
        cp.name,
        cp.reg(Space::Uc, 0x242),
        ctx(0x28B54),
        cp.reg(Space::Uc, 0x25B)
    );
    // NGG primitive shaders: program in the ES registers, resources and user
    // data in the GS ones (gfx10.3 register database: RSRC1/2_GS 0x8A/0x8B,
    // USER_DATA_GS_0 0x8C).
    for (name, lo, rsrc, user) in [("ps", 0x08, 0x0A, 0x0C), ("vs", 0x48, 0x4A, 0x4C), ("gs", 0x88, 0x8A, 0x8C), ("es", 0xC8, 0x8A, 0x8C), ("hs", 0x108, 0x10A, 0x10C), ("ls", 0x148, 0x14A, 0x14C)] {
        stage(cp, &mut s, &mut c.shaders, name, lo, rsrc, user);
    }
    let mask = ctx(0x28238);
    for i in 0..8u32 {
        let info = ctx(0x28C70 + 0x3C * i);
        if (info >> 2) & 0x1F == 0 || (mask >> (4 * i)) & 0xF == 0 {
            continue;
        }
        let base = ((ctx(0x28C60 + 0x3C * i) as u64) << 8) | ((ctx(0x28E40 + 4 * i) as u64 & 0xFF) << 40);
        let a2 = ctx(0x28EC0 + 4 * i);
        let a3 = ctx(0x28EE0 + 4 * i);
        let _ = writeln!(
            s,
            "    cb{i} {base:#x} info {info:#010x} view {:#x} attrib {:#x} {}x{} sw_mode {} type {} mask {:x}",
            ctx(0x28C6C + 0x3C * i),
            ctx(0x28C74 + 0x3C * i),
            ((a2 >> 14) & 0x3FFF) + 1,
            (a2 & 0x3FFF) + 1,
            (a3 >> 14) & 0x1F,
            (a3 >> 24) & 3,
            (mask >> (4 * i)) & 0xF
        );
    }
    let zinfo = ctx(0x28040);
    if zinfo & 3 != 0 {
        let _ = writeln!(s, "    depth {:#x} z_info {zinfo:#010x} size {:#x} control {:#x}", (ctx(0x28048) as u64) << 8, ctx(0x2801C), ctx(0x28800));
    }
    let _ = writeln!(
        s,
        "    viewport scale ({}, {}, {}) offset ({}, {}, {}) scissor {:#x}..{:#x} col_format {:#x} shader_mask {:#x} cb_control {:#x} blend0 {:#x} ps_input {:#x}/{:#x}",
        f(ctx(0x2843C)),
        f(ctx(0x28444)),
        f(ctx(0x2844C)),
        f(ctx(0x28440)),
        f(ctx(0x28448)),
        f(ctx(0x28450)),
        ctx(0x28204),
        ctx(0x28208),
        ctx(0x28714),
        ctx(0x2823C),
        ctx(0x28808),
        ctx(0x28780),
        ctx(0x286CC),
        ctx(0x286D0)
    );
    c.text.push_str(&s);
}

/// Records one dispatch.
pub fn dispatch(cp: &Cp, what: &str) {
    let mut g = ACTIVE.lock();
    let Some(c) = g.as_mut() else { return };
    c.events += 1;
    let mut s = String::new();
    let _ = writeln!(
        s,
        "dispatch #{} [{}] {what} threads {}x{}x{}",
        c.events,
        cp.name,
        cp.reg(Space::Sh, 0x207) & 0xFFFF,
        cp.reg(Space::Sh, 0x208) & 0xFFFF,
        cp.reg(Space::Sh, 0x209) & 0xFFFF
    );
    stage(cp, &mut s, &mut c.shaders, "cs", 0x20C, 0x212, 0x240);
    c.text.push_str(&s);
}
