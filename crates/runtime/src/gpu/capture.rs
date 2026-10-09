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
    /// Textures referenced in the frame (dumped as PNG when viewable).
    textures: std::collections::BTreeMap<u64, super::desc::Tsharp>,
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
/// Frames still to capture after the current one (DS_CAPTURE_COUNT).
static MORE: AtomicU64 = AtomicU64::new(0);

/// Flips so far (graphics queue).
static FLIPS: AtomicU64 = AtomicU64::new(0);

pub fn flips() -> u64 {
    FLIPS.load(Ordering::Relaxed)
}

pub fn on_flip(flip: u64) {
    FLIPS.store(flip, Ordering::Relaxed);
    let mut g = ACTIVE.lock();
    if let Some(c) = g.take() {
        // Consecutive captures: text only (no texture, target or draw dumps).
        let more = MORE.load(Ordering::SeqCst);
        let last = more == 0;
        finish(c, last);
        if !last {
            MORE.store(more - 1, Ordering::SeqCst);
            *g = Some(Capture { flip, text: format!("# frame after flip {flip}
"), shaders: BTreeSet::new(), events: 0, textures: Default::default() });
        }
        return;
    }
    let armed = ARMED.load(Ordering::SeqCst);
    if armed != u64::MAX && flip >= armed {
        ARMED.store(u64::MAX, Ordering::SeqCst);
        MORE.store(std::env::var("DS_CAPTURE_COUNT").ok().and_then(|v| v.parse::<u64>().ok()).unwrap_or(1).saturating_sub(1), Ordering::SeqCst);
        *g = Some(Capture { flip, text: format!("# frame after flip {flip}\n"), shaders: BTreeSet::new(), events: 0, textures: Default::default() });
        crate::info!("capture: recording the frame after flip {flip}");
    }
}

fn finish(c: Capture, dumps: bool) {
    let Some(dir) = crate::run_dir() else { return };
    let path = dir.join(format!("capture-{}.txt", c.flip));
    let _ = std::fs::write(&path, &c.text);
    let dumped = c.shaders.iter().filter(|&&a| save_program(a).is_some()).count();
    let tex = if dumps { dump_textures(&dir, &c.textures) } else { 0 };
    // Render targets as rendered (GPU execution): the graphics queue's work
    // of the frame completed before its flip.
    let rts = if dumps && super::exec::enabled() { super::images::with(|im| im.dump_targets(&dir.join("targets"))) } else { 0 };
    crate::info!("capture: {} event(s) -> {} ({dumped} shader binaries in shaders/, {tex} texture(s) in textures/, {rts} render target(s) in targets/)", c.events, path.display());
}

/// Writes viewable textures (RGBA8/R8/RG8 in standard swizzles) of the frame
/// as PNG files, detiled with `tiling` (mip 0, slice 0): a visual check of
/// the swizzle implementation.
fn dump_textures(dir: &std::path::Path, textures: &std::collections::BTreeMap<u64, super::desc::Tsharp>) -> usize {
    let tdir = dir.join("textures");
    let mut n = 0;
    for (base, t) in textures {
        if n >= 48 || t.width < 16 || t.height < 16 || !matches!(t.sw_mode, 1 | 5 | 9) {
            continue;
        }
        let (elem_log2, expand): (u32, fn(&[u8]) -> [u8; 4]) = match t.format {
            56 | 130 => (2, |p| [p[0], p[1], p[2], p[3]]),
            1 => (0, |p| [p[0], p[0], p[0], 255]),
            14 => (1, |p| [p[0], p[1], 0, 255]),
            _ => continue,
        };
        let levels = (t.max_mip + 1).max(t.last_level + 1).max(1);
        let Some(l) = super::tiling::layout(t.sw_mode, elem_log2, t.width, t.height, levels, false) else { continue };
        let size = l.size(1);
        if !crate::gpu::cp::guest_readable(*base, size) {
            continue;
        }
        let src = unsafe { std::slice::from_raw_parts(*base as *const u8, size as usize) };
        let mut lin = Vec::new();
        l.detile(src, 0, 0, &mut lin);
        let e = 1usize << elem_log2;
        let mut rgba = Vec::with_capacity(lin.len() / e * 4);
        for px in lin.chunks_exact(e) {
            rgba.extend_from_slice(&expand(px));
        }
        let _ = std::fs::create_dir_all(&tdir);
        let name = format!("{base:010x}_{}x{}_f{}_sw{}_l{levels}.png", t.width, t.height, t.format, t.sw_mode);
        if std::fs::write(tdir.join(name), super::png::encode(t.width, t.height, &rgba)).is_ok() {
            n += 1;
        }
    }
    n
}

/// Fallback size: up to the first s_endpgm (gfx10 0xBF810000), at most 64 KiB.
/// Writes the shader program at `a` to `<run dir>/shaders/<address>.bin`.
pub fn save_program(a: u64) -> Option<std::path::PathBuf> {
    let sdir = crate::run_dir()?.join("shaders");
    let size = crate::hle::libs::agc::shader_code_size(a).unwrap_or_else(|| scan_to_endpgm(a));
    if size == 0 || !crate::gpu::cp::guest_readable(a, size as u64) {
        return None;
    }
    let _ = std::fs::create_dir_all(&sdir);
    let bytes = unsafe { std::slice::from_raw_parts(a as *const u8, size) };
    let path = sdir.join(format!("{a:012x}.bin"));
    std::fs::write(&path, bytes).ok().map(|_| path)
}

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

/// First records of the vertex stage's formatted buffers (vertex data).
fn vertex_data(out: &mut String, p: &super::shaders::Prepared) {
    for (i, r) in p.resources.iter().enumerate() {
        let super::shaders::Resolved::Buffer(v) = r else { continue };
        if v.stride == 0 || v.stride > 64 || v.format == 0 {
            continue;
        }
        let n = v.num_records.min(6) as u64;
        if !crate::gpu::cp::guest_readable(v.base, n * v.stride as u64) {
            continue;
        }
        let mut recs = Vec::new();
        for k in 0..n {
            let a = v.base + k * v.stride as u64;
            let words: Vec<u32> = (0..(v.stride as u64 / 4).min(4)).map(|j| unsafe { std::ptr::read_volatile((a + 4 * j) as *const u32) }).collect();
            recs.push(match v.format {
                // 32-bit float components.
                64 | 74 | 77 | 22 => format!("({})", words.iter().map(|w| format!("{}", f32::from_bits(*w))).collect::<Vec<_>>().join(", ")),
                _ => format!("[{}]", words.iter().map(|w| format!("{w:08x}")).collect::<Vec<_>>().join(" ")),
            });
        }
        let _ = writeln!(out, "      es {i} data: {}", recs.join(" "));
    }
}

fn note_textures(map: &mut std::collections::BTreeMap<u64, super::desc::Tsharp>, p: &super::shaders::Prepared) {
    for r in &p.resources {
        if let super::shaders::Resolved::Image(t) = r {
            map.entry(t.base).or_insert(*t);
        }
    }
}

fn f(v: u32) -> f32 {
    f32::from_bits(v)
}

/// `DS_CAPTURE_DRAWS=1`: also dump each captured draw's first color target
/// after the draw (draws/<event>_<address>.png).
pub fn draw_dumps() -> bool {
    dumps_env() && capturing()
}

/// `DS_CAPTURE_DISPATCHES=1`: also dump the images each captured dispatch
/// writes, after it (dispatches/<flip>_<event>_<shader>_<image>.png).
pub fn dispatch_dumps() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_CAPTURE_DISPATCHES").ok().as_deref() == Some("1")) && capturing()
}

fn dumps_env() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_CAPTURE_DRAWS").ok().as_deref() == Some("1"))
}

fn dump_draw_target(cp: &Cp, event: u64) -> Option<String> {
    let s = super::exec::color_surface(cp, 0).ok().flatten()?;
    let dir = crate::run_dir()?.join("draws");
    let _ = std::fs::create_dir_all(&dir);
    let r = super::images::with(|im| -> Result<(Vec<u8>, u32, u32), String> {
        let t = im.target(&s)?;
        let raw = im.read_back(t.image, t.key.format, t.key.width, t.key.height)?;
        Ok((super::images::to_rgba8(t.key.format, &raw).ok_or("format")?, t.key.width, t.key.height))
    });
    match r {
        Ok((rgba, w, h)) => {
            let name = format!("{event:04}_{:010x}.png", s.base);
            std::fs::write(dir.join(&name), super::png::encode(w, h, &rgba)).ok()?;
            Some(name)
        }
        Err(e) => Some(format!("(dump failed: {e})")),
    }
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
        "draw #{} [{}] {what} prim {} stages {:#x} ge_cntl {:#x} -> {}",
        c.events,
        cp.name,
        cp.reg(Space::Uc, 0x242),
        ctx(0x28B54),
        cp.reg(Space::Uc, 0x25B),
        cp.last_draw
    );
    if dumps_env() {
        if let Some(n) = dump_draw_target(cp, c.events) {
            let _ = writeln!(s, "    target after draw: {n}");
        }
    }
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
        // Metadata: DCC (INFO bit 28) and CMASK surfaces, fast-clear color.
        let dcc = ((ctx(0x28C94 + 0x3C * i) as u64) << 8) | ((ctx(0x28EA0 + 4 * i) as u64 & 0xFF) << 40);
        let cmask = ((ctx(0x28C7C + 0x3C * i) as u64) << 8) | ((ctx(0x28E60 + 4 * i) as u64 & 0xFF) << 40);
        let _ = writeln!(
            s,
            "    cb{i} {base:#x} info {info:#010x} view {:#x} attrib {:#x} {}x{} sw_mode {} type {} mask {:x} dcc {dcc:#x} ctl {:#x} cmask {cmask:#x} clear {:#x} {:#x}",
            ctx(0x28C6C + 0x3C * i),
            ctx(0x28C74 + 0x3C * i),
            ((a2 >> 14) & 0x3FFF) + 1,
            (a2 & 0x3FFF) + 1,
            (a3 >> 14) & 0x1F,
            (a3 >> 24) & 3,
            (mask >> (4 * i)) & 0xF,
            ctx(0x28C78 + 0x3C * i),
            ctx(0x28C8C + 0x3C * i),
            ctx(0x28C90 + 0x3C * i)
        );
    }
    for (name, r) in [("es", super::shaders::prepare_ngg(cp)), ("ps", super::shaders::prepare_ps(cp))] {
        match r {
            Ok(p) => {
                for l in super::shaders::describe_resources(&p) {
                    let _ = writeln!(s, "      {name} {l}");
                }
                if super::exec::enabled() {
                    for r in &p.resources {
                        if let super::shaders::Resolved::Image(t) = r {
                            let d = super::images::with(|im| im.describe(t));
                            let _ = writeln!(s, "      {name} image {:#x}: {d}", t.base);
                        }
                    }
                }
                if name == "es" {
                    vertex_data(&mut s, &p);
                }
                for l in super::shaders::describe_loads(&p) {
                    let _ = writeln!(s, "      {name} {l}");
                }
                // The SPIR-V used at run time (specialized to this draw).
                if let Some(dir) = crate::run_dir() {
                    let sdir = dir.join("shaders");
                    let _ = std::fs::create_dir_all(&sdir);
                    let path = sdir.join(format!("{:012x}-{:016x}.spv", p.addr, p.tr.key));
                    if !path.exists() {
                        let bytes: Vec<u8> = p.tr.out.spirv.iter().flat_map(|w| w.to_le_bytes()).collect();
                        let _ = std::fs::write(&path, bytes);
                    }
                    let _ = writeln!(s, "      {name} spirv {}", path.file_name().unwrap().to_string_lossy());
                }
                note_textures(&mut c.textures, &p);
            }
            Err(e) => {
                let _ = writeln!(s, "      {name} resources unresolved: {e}");
            }
        }
    }
    let zinfo = ctx(0x28040);
    if zinfo & 3 != 0 {
        // HTILE (Z_INFO bit 29), fast-clear values.
        let htile = ((ctx(0x28014) as u64) << 8) | ((ctx(0x28078) as u64 & 0xFF) << 40);
        let _ = writeln!(
            s,
            "    depth {:#x} z_info {zinfo:#010x} stencil {:#x} s_info {:#x} size {:#x} control {:#x} htile {htile:#x} clear z {} s {:#x}",
            (ctx(0x28048) as u64) << 8,
            ((ctx(0x2804C) as u64) << 8) | ((ctx(0x2806C) as u64 & 0xFF) << 40),
            ctx(0x28044),
            ctx(0x2801C),
            ctx(0x28800),
            f32::from_bits(ctx(0x2802C)),
            ctx(0x28028)
        );
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
    let inputs: Vec<String> = (0..(ctx(0x286D8) & 0x3F).min(8)).map(|n| format!("{:#x}", ctx(0x28644 + 4 * n))).collect();
    let _ = writeln!(s, "    ps_in_control {:#x} input_cntl [{}] z_format {:#x}", ctx(0x286D8), inputs.join(" "), ctx(0x28710));
    // Raster, depth/stencil and blend state.
    let blends: Vec<String> = (0..8u32).filter(|i| (mask >> (4 * i)) & 0xF != 0).map(|i| format!("{i}:{:#x}", ctx(0x28780 + 4 * i))).collect();
    let _ = writeln!(
        s,
        "    state vte {:#x} clip {:#x} su_sc_mode {:#x} sc_mode0 {:#x} aa {:#x} win_off {:#x} | screen_sc {:#x}..{:#x} generic_sc {:#x}..{:#x} vport_sc {:#x}..{:#x} zrange {}..{} | db_render {:#x} db_shader {:#x} depth_ctl {:#x} stencil_ctl {:#x} ref {:#x}/{:#x} z_read {:#x} s_info {:#x} | target_mask {mask:#x} blend [{}] blend_color ({}, {}, {}, {}) | index_type {:#x} instances {} reset_en {:#x} gb_clip {}/{}",
        ctx(0x28818),
        ctx(0x28810),
        ctx(0x28814),
        ctx(0x28A48),
        ctx(0x28BE0),
        ctx(0x28200),
        ctx(0x28030),
        ctx(0x28034),
        ctx(0x28240),
        ctx(0x28244),
        ctx(0x28250),
        ctx(0x28254),
        f(ctx(0x282D0)),
        f(ctx(0x282D4)),
        ctx(0x28000),
        ctx(0x2880C),
        ctx(0x28800),
        ctx(0x2842C),
        ctx(0x28430),
        ctx(0x28434),
        ((ctx(0x28048) as u64) << 8) | ((ctx(0x28068) as u64 & 0xFF) << 40),
        ctx(0x28044),
        blends.join(" "),
        f(ctx(0x28414)),
        f(ctx(0x28418)),
        f(ctx(0x2841C)),
        f(ctx(0x28420)),
        cp.reg(Space::Uc, 0x243),
        cp.num_instances(),
        ctx(0x28A94),
        f(ctx(0x28BE8)),
        f(ctx(0x28BF0))
    );
    c.text.push_str(&s);
}

/// Records a synchronization or memory packet.
pub fn packet(cp: &Cp, what: &str) {
    let mut g = ACTIVE.lock();
    let Some(c) = g.as_mut() else { return };
    let _ = writeln!(c.text, "packet [{}] {what}", cp.name);
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
    let dumps = std::env::var("DS_CAPTURE_DISPATCHES").ok().as_deref() == Some("1");
    match super::shaders::prepare_compute(cp, true) {
        Ok(p) => {
            for l in super::shaders::describe_resources(&p) {
                let _ = writeln!(s, "      {l}");
            }
            note_textures(&mut c.textures, &p);
            if dumps && super::exec::enabled() {
                for (i, r) in p.resources.iter().enumerate() {
                    let super::shaders::Resolved::Image(t) = r else { continue };
                    if !p.tr.plan.resources[i].written {
                        continue;
                    }
                    let Some(dir) = crate::run_dir().map(|d| d.join("dispatches")) else { continue };
                    let _ = std::fs::create_dir_all(&dir);
                    let name = format!("{}_{:04}_{:010x}_{:010x}.png", c.flip, c.events, p.addr, t.base);
                    let r = super::images::with(|im| im.snapshot(t));
                    match r {
                        Ok((rgba, w, h, stats)) => {
                            let _ = std::fs::write(dir.join(&name), super::png::encode(w, h, &rgba));
                            let _ = writeln!(s, "      image after dispatch: {name}{}", stats.map(|x| format!(" ({x})")).unwrap_or_default());
                        }
                        Err(e) => {
                            let _ = writeln!(s, "      image after dispatch: {name} failed: {e}");
                        }
                    }
                }
            }
        }
        Err(e) => {
            let _ = writeln!(s, "      resources unresolved: {e}");
        }
    }
    c.text.push_str(&s);
}
