//! Draws (docs/RENDERER.md milestone 3): the NGG primitive shader runs as a
//! Vulkan vertex shader, the pixel shader as the fragment shader, rect lists
//! are expanded by a generated geometry shader. Color and depth targets come
//! from the CB/DB registers and live in the image cache (shared with textures
//! by guest address); draws record into the queue's command buffer inside
//! dynamic-rendering instances that last while the attachments stay the same.

use super::*;
use crate::gpu::images::{self, Surface, Target};
use std::hash::{Hash, Hasher};
use std::sync::atomic::AtomicU32;

pub static DRAWN: AtomicU64 = AtomicU64::new(0);
/// Draws without visible effect here (metadata passes, all writes disabled).
pub static DRAW_NOOP: AtomicU64 = AtomicU64::new(0);
pub static DRAW_SKIPPED: AtomicU64 = AtomicU64::new(0);
/// Indexed indirect draws recorded with the GPU reading their arguments.
pub static GPU_ARG_DRAWS: AtomicU64 = AtomicU64::new(0);
/// Of them, those recorded as a repeat of the draw before (`draw_repeat`).
pub static REPEAT_DRAWS: AtomicU64 = AtomicU64::new(0);

/// A draw with GPU-read arguments that another can repeat (`draw_repeat`):
/// the batch and `cmd_uses` right after it was recorded, its index binding
/// (base, index size, bytes; the home's buffer and offset bound) and the
/// arguments' home buffer; or (`cpu`) a draw whose arguments the CPU read,
/// drawn in `cpu_args_base` mode with its indices in a home (`index_off`:
/// the offset bound last).
#[derive(Clone, Copy)]
pub(crate) struct Repeat {
    batch: u64,
    uses: u64,
    index: (u64, u32, u64),
    index_buf: vk::Buffer,
    index_off: u64,
    args_buf: vk::Buffer,
    cpu: bool,
}

const MAX_RT: usize = 8;

/// Active dynamic-rendering instance of an executor.
/// Dynamic graphics state recorded in the current command buffer: per
/// state its value plus one (0: not recorded yet), the bound pipeline, the
/// viewport, scissor, depth bias and blend constants (all of them as bits).
/// A new command buffer starts with nothing recorded; only draws record
/// graphics state, and every graphics pipeline makes the same states
/// dynamic. Re-setting unchanged state cost ~1.2 us per draw (~25 calls, run
/// 20261011-133502).
#[derive(Default)]
pub(crate) struct DynCache {
    words: [u64; DYN_SLOTS],
    pipeline: vk::Pipeline,
    /// The compute pipeline bound last in this command buffer.
    pub(crate) cs_pipeline: vk::Pipeline,
    /// The push descriptors in this command buffer, per bind point
    /// (graphics, compute; `bind_descriptors`).
    pub(crate) pushed: [super::PushedSet; 2],
    viewport: Option<[u32; 6]>,
    scissor: Option<[i32; 4]>,
    bias: Option<[u32; 3]>,
    blend: Option<[u32; 4]>,
}

const DYN_SLOTS: usize = 22;

impl DynCache {
    /// Whether state `slot` must be recorded with value `v` (and notes it).
    fn set(&mut self, slot: usize, v: u64) -> bool {
        let w = v.wrapping_add(1);
        if self.words[slot] == w {
            return false;
        }
        self.words[slot] = w;
        true
    }
}

pub(super) struct Rendering {
    pub colors: [Option<Target>; MAX_RT],
    pub depth: Option<Target>,
    pub area: (u32, u32),
    /// Color attachments its draws wrote, and those they sampled; the depth
    /// attachment's depth and stencil aspects written and sampled (one
    /// instance never does both to one aspect of an attachment: `tune
    /// feedback_exact`).
    pub written: Vec<vk::Image>,
    pub sampled: Vec<vk::Image>,
    pub ds_written: [bool; 2],
    pub ds_sampled: [bool; 2],
}

/// Arguments of one draw packet.
pub struct DrawArgs {
    /// Vertices (non-indexed) or indices.
    pub count: u32,
    pub instances: u32,
    /// Guest address of the first index and the index size in bytes.
    pub index: Option<(u64, u32)>,
    /// First vertex (non-indexed) or vertex offset (indexed); None: VGT_INDX_OFFSET.
    pub base_vertex: Option<i32>,
    /// Added to the instance id VGPR (indirect draws whose packet names no
    /// start-instance SGPR; see the CP).
    pub first_instance: u32,
    /// An indexed indirect draw whose arguments the GPU reads (`tune
    /// draw_args_gpu`); `count`, `instances`, `base_vertex` and
    /// `first_instance` are then unused and `index` is the index buffer's
    /// start.
    pub gpu_args: Option<GpuArgs>,
    /// The vertex stage's user-data word that takes the start instance from
    /// `gl_BaseInstance`: an indexed indirect draw whose arguments the CPU
    /// read, drawn with `first_instance` as Vulkan's first instance (`tune
    /// cpu_args_base`); None: any start instance is in that word's register.
    pub base_inst: Option<u16>,
}

/// Arguments read by the GPU (`DrawArgs::gpu_args`).
#[derive(Clone, Copy, Debug)]
pub struct GpuArgs {
    /// Guest address of the DrawIndexIndirect record (in a video-memory home).
    pub addr: u64,
    /// The vertex stage's user-data word that receives the start instance.
    pub inst_slot: u16,
    /// The index buffer's size in bytes (VGT_INDEX_BUFFER_SIZE), in a home.
    pub index_bytes: u64,
}

/// `draw` outcome when the vertex stage needs the start instance on the CPU
/// (`shaders::uses_user_word`): the command processor reads the arguments
/// and draws again.
pub const NEEDS_CPU_ARGS: &str = "indirect arguments needed on the CPU";

fn ctx_reg(cp: &Cp, a: u32) -> u32 {
    cp.reg(Space::Ctx, (a - 0x28000) / 4)
}

fn f32r(v: u32) -> f32 {
    f32::from_bits(v)
}

/// CB_BLENDn_CONTROL blend factor.
fn blend_factor(f: u32) -> vk::BlendFactor {
    use vk::BlendFactor as B;
    match f {
        0 => B::ZERO,
        1 => B::ONE,
        2 => B::SRC_COLOR,
        3 => B::ONE_MINUS_SRC_COLOR,
        4 => B::SRC_ALPHA,
        5 => B::ONE_MINUS_SRC_ALPHA,
        6 => B::DST_ALPHA,
        7 => B::ONE_MINUS_DST_ALPHA,
        8 => B::DST_COLOR,
        9 => B::ONE_MINUS_DST_COLOR,
        10 => B::SRC_ALPHA_SATURATE,
        13 => B::CONSTANT_COLOR,
        14 => B::ONE_MINUS_CONSTANT_COLOR,
        15 => B::SRC1_COLOR,
        16 => B::ONE_MINUS_SRC1_COLOR,
        17 => B::SRC1_ALPHA,
        18 => B::ONE_MINUS_SRC1_ALPHA,
        19 => B::CONSTANT_ALPHA,
        20 => B::ONE_MINUS_CONSTANT_ALPHA,
        _ => B::ONE,
    }
}

/// CB_BLENDn_CONTROL combine function.
fn blend_op(f: u32) -> vk::BlendOp {
    match f {
        1 => vk::BlendOp::SUBTRACT,
        2 => vk::BlendOp::MIN,
        3 => vk::BlendOp::MAX,
        4 => vk::BlendOp::REVERSE_SUBTRACT,
        _ => vk::BlendOp::ADD,
    }
}

/// DB_DEPTH_CONTROL ZFUNC / STENCILFUNC.
fn compare_op(f: u32) -> vk::CompareOp {
    match f & 7 {
        0 => vk::CompareOp::NEVER,
        1 => vk::CompareOp::LESS,
        2 => vk::CompareOp::EQUAL,
        3 => vk::CompareOp::LESS_OR_EQUAL,
        4 => vk::CompareOp::GREATER,
        5 => vk::CompareOp::NOT_EQUAL,
        6 => vk::CompareOp::GREATER_OR_EQUAL,
        _ => vk::CompareOp::ALWAYS,
    }
}

/// DB_STENCIL_CONTROL operation (ONES/logic ops approximated).
fn stencil_op(op: u32) -> vk::StencilOp {
    match op & 0xF {
        1 => vk::StencilOp::ZERO,
        3 | 4 => vk::StencilOp::REPLACE,
        5 => vk::StencilOp::INCREMENT_AND_CLAMP,
        6 => vk::StencilOp::DECREMENT_AND_CLAMP,
        7 => vk::StencilOp::INVERT,
        8 => vk::StencilOp::INCREMENT_AND_WRAP,
        9 => vk::StencilOp::DECREMENT_AND_WRAP,
        _ => vk::StencilOp::KEEP,
    }
}

/// EQUAL depth test diagnosis (`DS_DEPTH_EQUAL_BIAS` or the `eqbias`
/// control command): f32 bits, NaN = off, infinity = always pass.
pub static EQUAL_BIAS: AtomicU32 = AtomicU32::new(0x7FC0_0000);

pub fn set_equal_bias(v: Option<f32>) {
    EQUAL_BIAS.store(v.unwrap_or(f32::NAN).to_bits(), Ordering::Relaxed);
}

/// Draws whose vertex shader is on this list are skipped (`skipvs` control
/// command; diagnosis).
static SKIP_VS: parking_lot::Mutex<Option<std::collections::HashSet<u64>>> = parking_lot::Mutex::new(None);
/// Whether the list has entries (read by every draw without the lock).
static SKIP_VS_ANY: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Adds or removes a vertex shader address; returns whether it is listed now.
pub fn toggle_skip_vs(addr: u64) -> bool {
    let mut g = SKIP_VS.lock();
    let s = g.get_or_insert_with(Default::default);
    let listed = if s.remove(&addr) {
        false
    } else {
        s.insert(addr);
        true
    };
    SKIP_VS_ANY.store(!s.is_empty(), Ordering::Relaxed);
    listed
}

fn equal_bias() -> Option<f32> {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        if let Ok(v) = std::env::var("DS_DEPTH_EQUAL_BIAS") {
            set_equal_bias(if v == "always" { Some(f32::INFINITY) } else { v.parse().ok() });
        }
    });
    Some(f32::from_bits(EQUAL_BIAS.load(Ordering::Relaxed))).filter(|v| !v.is_nan())
}

/// Color target n from CB_COLORn_* (None: no target bound).
pub(crate) fn color_surface(cp: &Cp, n: u32) -> Result<Option<Surface>, String> {
    let r = |o: u32| ctx_reg(cp, 0x28C60 + 0x3C * n + o);
    let info = r(0x10);
    let fmt = (info >> 2) & 0x1F;
    if fmt == 0 {
        return Ok(None);
    }
    let format = images::cb_format(fmt, (info >> 8) & 7, (info >> 11) & 3).ok_or_else(|| format!("color target format {fmt} number type {} swap {} unsupported", (info >> 8) & 7, (info >> 11) & 3))?;
    let base = ((r(0) as u64) << 8) | ((ctx_reg(cp, 0x28E40 + 4 * n) as u64 & 0xFF) << 40);
    let a2 = ctx_reg(cp, 0x28EC0 + 4 * n);
    let a3 = ctx_reg(cp, 0x28EE0 + 4 * n);
    let view = r(0xC);
    // DCC (INFO.DCC_ENABLE): fast clears write only its metadata.
    let meta = if info & (1 << 28) != 0 { ((r(0x34) as u64) << 8) | ((ctx_reg(cp, 0x28EA0 + 4 * n) as u64 & 0xFF) << 40) } else { 0 };
    Ok(Some(Surface {
        base,
        format,
        width: ((a2 >> 14) & 0x3FFF) + 1,
        height: (a2 & 0x3FFF) + 1,
        layers: (a3 & 0x1FFF) + 1,
        levels: (a2 >> 28) + 1,
        ty3d: (a3 >> 24) & 3 == 2,
        mip: (view >> 26) & 0xF,
        slice: view & 0x1FFF,
        stencil_base: 0,
        meta,
        clear: [r(0x2C), r(0x30)],
    }))
}

/// Depth/stencil surface from DB_* (None: none bound).
pub(crate) fn depth_surface(cp: &Cp) -> Option<Surface> {
    let zinfo = ctx_reg(cp, 0x28040);
    let sinfo = ctx_reg(cp, 0x28044);
    let format = match (zinfo & 3, sinfo & 1) {
        (0, 0) => return None,
        (1, 0) => vk::Format::D16_UNORM,
        (3, 0) => vk::Format::D32_SFLOAT,
        (_, 1) => vk::Format::D32_SFLOAT_S8_UINT,
        _ => return None,
    };
    // DB_STENCIL_READ_BASE: the stencil plane lives at its own address.
    let stencil_base = ((ctx_reg(cp, 0x2804C) as u64) << 8) | ((ctx_reg(cp, 0x2806C) as u64 & 0xFF) << 40);
    let base = if zinfo & 3 != 0 { ((ctx_reg(cp, 0x28048) as u64) << 8) | ((ctx_reg(cp, 0x28068) as u64 & 0xFF) << 40) } else { stencil_base };
    let size = ctx_reg(cp, 0x2801C);
    let view = ctx_reg(cp, 0x28008);
    let slice = (view & 0x7FF) | (((view >> 11) & 3) << 11);
    let slice_max = ((view >> 13) & 0x7FF) | (((view >> 30) & 3) << 11);
    Some(Surface {
        base,
        format,
        width: (size & 0x3FFF) + 1,
        height: ((size >> 16) & 0x3FFF) + 1,
        layers: slice_max.max(slice) + 1,
        levels: ((zinfo >> 16) & 0xF) + 1,
        ty3d: false,
        mip: (view >> 26) & 0xF,
        slice,
        stencil_base: if sinfo & 1 != 0 { stencil_base } else { 0 },
        // HTILE (Z_INFO.TILE_SURFACE_ENABLE); DB_DEPTH_CLEAR, DB_STENCIL_CLEAR.
        meta: if zinfo & (1 << 29) != 0 { ((ctx_reg(cp, 0x28014) as u64) << 8) | ((ctx_reg(cp, 0x28078) as u64 & 0xFF) << 40) } else { 0 },
        clear: [ctx_reg(cp, 0x2802C), ctx_reg(cp, 0x28028)],
    })
}

/// Intersection of the enabled scissors (screen, window, generic, viewport 0),
/// clipped to the render area. Empty (unprogrammed) scissors are ignored.
fn scissor(cp: &Cp, area: (u32, u32)) -> (i32, i32, i32, i32) {
    let mut s = (0i32, 0i32, area.0 as i32, area.1 as i32);
    let vport = ctx_reg(cp, 0x28A48) & 2 != 0;
    for (tl, br, bits) in [(0x28030, 0x28034, 0xFFFF), (0x28204, 0x28208, 0x7FFF), (0x28240, 0x28244, 0x7FFF), (0x28250, 0x28254, 0x7FFF)] {
        if tl == 0x28250 && !vport {
            continue;
        }
        let (a, b) = (ctx_reg(cp, tl), ctx_reg(cp, br));
        let (x0, y0, x1, y1) = ((a & bits) as i32, ((a >> 16) & bits) as i32, (b & bits) as i32, ((b >> 16) & bits) as i32);
        if x1 > x0 && y1 > y0 {
            s = (s.0.max(x0), s.1.max(y0), s.2.min(x1), s.3.min(y1));
        }
    }
    s
}

/// Pipeline state that is baked into a graphics pipeline.
#[derive(Hash, Clone, Copy, PartialEq, Eq)]
struct GfxKey {
    vs: u64,
    ps: u64,
    rect_gs: Option<u32>,
    colors: [i32; MAX_RT],
    blend: [u32; MAX_RT],
    masks: [u8; MAX_RT],
    depth: i32,
    topology_class: u8,
    depth_clamp: bool,
}

thread_local! {
    /// The last graphics pipeline key this thread hashed, and its hash.
    static LAST_GFX_KEY: std::cell::Cell<Option<(GfxKey, u64)>> = const { std::cell::Cell::new(None) };
}

/// A graphics pipeline's state apart from its stages (`GfxKey` without the
/// translation keys; `gpu::precompile` keeps it with the stages' specs).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct GfxState {
    pub rect: bool,
    pub colors: [i32; MAX_RT],
    pub blend: [u32; MAX_RT],
    pub masks: [u8; MAX_RT],
    pub depth: i32,
    pub topology_class: u8,
    pub depth_clamp: bool,
}

/// Creates the graphics pipeline of translations made ahead from shader
/// objects (`gpu::precompile`) with `state`, under the key its draws look
/// up: Some(true) when created here, Some(false) when it existed or was in
/// flight, None when creating it failed.
pub(crate) fn precompile_gfx_translations(vs: StageTr, ps: Option<StageTr>, state: &GfxState) -> Option<bool> {
    let key = GfxKey {
        vs: vs.0.key,
        ps: ps.map_or(0, |p| p.0.key),
        rect_gs: state.rect.then_some(vs.0.out.param_mask),
        colors: state.colors,
        blend: state.blend,
        masks: state.masks,
        depth: state.depth,
        topology_class: state.topology_class,
        depth_clamp: state.depth_clamp,
    };
    let bindings: u32 = super::descriptor_bindings(&vs.0.out).chain(ps.iter().flat_map(|p| super::descriptor_bindings(&p.0.out))).map(|b| super::binding_count(b.kind)).sum();
    let push_mode = super::super::tune::push_desc() && bindings <= ctx().max_push;
    let hkey = super::push_key(hash_gfx_key(&key), push_mode);
    if gfx_known(hkey) {
        return Some(false);
    }
    let created = std::cell::Cell::new(false);
    gfx_pipe_with(&key, hkey, vs, ps, push_mode, &created).ok().map(|_| created.get())
}

/// A key's hash (the persistent cache file's name, so its hasher stays).
fn hash_gfx_key(key: &GfxKey) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    key.hash(&mut h);
    h.finish()
}

/// The registers a draw's baked pipeline state is made of, as a scan of the
/// commands ahead saw them (`gfx_key_ahead`): context registers and the
/// primitive type (a uconfig register).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct GfxRegs {
    cb_color_control: u32,
    clip: u32,
    col_format: u32,
    target_mask: u32,
    zinfo: u32,
    sinfo: u32,
    dctl: u32,
    db_render: u32,
    sctl: u32,
    refm: u32,
    refm_bf: u32,
    cb_info: [u32; MAX_RT],
    blend: [u32; MAX_RT],
    prim: u32,
}

impl GfxRegs {
    /// From a context register reader (`ctx(0x28xxx)`) and VGT_PRIMITIVE_TYPE.
    pub(crate) fn capture(ctx: &dyn Fn(u32) -> u32, prim: u32) -> GfxRegs {
        GfxRegs {
            cb_color_control: ctx(0x28808),
            clip: ctx(0x28810),
            col_format: ctx(0x28714),
            target_mask: ctx(0x28238) & ctx(0x2823C),
            zinfo: ctx(0x28040),
            sinfo: ctx(0x28044),
            dctl: ctx(0x28800),
            db_render: ctx(0x28000),
            sctl: ctx(0x2842C),
            refm: ctx(0x28430),
            refm_bf: ctx(0x28434),
            cb_info: std::array::from_fn(|n| ctx(0x28C60 + 0x3C * n as u32 + 0x10)),
            blend: std::array::from_fn(|n| ctx(0x28780 + 4 * n as u32)),
            prim,
        }
    }

    /// Whether two draws' baked state would be made from the same values
    /// (one compile ahead per such pair of stages and state).
    pub(crate) fn same(&self, o: &GfxRegs) -> bool {
        self == o
    }

    pub(crate) fn hash64(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.hash(&mut h);
        h.finish()
    }
}

/// The pipeline key and its cache hash (with the push-descriptor mode) of a
/// draw with these registers and stages as `draw_inner` makes them; None
/// when such a draw records nothing or is not supported. Compiles ahead of
/// use (`gpu::compile`) take it; a key that differs at execution only cost
/// a compile.
fn gfx_key_ahead(g: &GfxRegs, vs: &Prepared, ps: Option<&Prepared>) -> Option<(GfxKey, u64, bool)> {
    let cb_mode = (g.cb_color_control >> 4) & 7;
    if cb_mode > 1 || g.clip & (1 << 22) != 0 {
        return None;
    }
    let (class, rect) = match g.prim & 0x3F {
        1 => (0u8, false),
        2 | 3 => (1, false),
        4..=6 => (2, false),
        7 | 0x11 => (2, true),
        _ => return None,
    };
    let mrt_mask = ps.map(|p| p.tr.out.mrt_mask).unwrap_or(0);
    let mut colors = [vk::Format::UNDEFINED.as_raw(); MAX_RT];
    let mut blend = [0u32; MAX_RT];
    let mut masks = [0u8; MAX_RT];
    let mut any_color = false;
    if cb_mode == 1 {
        for i in 0..MAX_RT {
            let m = (g.target_mask >> (4 * i)) & 0xF;
            if m == 0 || mrt_mask & (1 << i) == 0 || (g.col_format >> (4 * i)) & 0xF == 0 {
                continue;
            }
            let info = g.cb_info[i];
            let fmt = (info >> 2) & 0x1F;
            if fmt == 0 {
                continue;
            }
            colors[i] = images::cb_format(fmt, (info >> 8) & 7, (info >> 11) & 3)?.as_raw();
            blend[i] = g.blend[i];
            masks[i] = m as u8;
            any_color = true;
        }
    }
    let dformat = match (g.zinfo & 3, g.sinfo & 1) {
        (0, 0) => None,
        (1, 0) => Some(vk::Format::D16_UNORM),
        (3, 0) => Some(vk::Format::D32_SFLOAT),
        (_, 1) => Some(vk::Format::D32_SFLOAT_S8_UINT),
        _ => None,
    };
    let (has_z, has_s) = match dformat {
        Some(f) => (images::aspects(f).contains(vk::ImageAspectFlags::DEPTH) && g.zinfo & 3 != 0, images::has_stencil(f)),
        None => (false, false),
    };
    let z_test = has_z && g.dctl & 2 != 0;
    let z_write = z_test && g.dctl & 4 != 0;
    let s_test = has_s && g.dctl & 1 != 0;
    let s_write = s_test && (g.refm | g.refm_bf) & 0x00FF_0000 != 0 && g.sctl & 0xFF_FFFF != 0;
    let clear_z = has_z && g.db_render & 1 != 0;
    let clear_s = has_s && g.db_render & 2 != 0;
    let depth = if z_test || s_test || clear_z || clear_s { dformat } else { None };
    let mem_writes = vs.tr.plan.resources.iter().any(|r| r.written) || ps.is_some_and(|p| p.tr.plan.resources.iter().any(|r| r.written));
    if !any_color && !z_write && !s_write && !clear_z && !clear_s && !mem_writes {
        return None;
    }
    if !any_color && depth.is_none() && !super::super::tune::attachless_draws() {
        return None;
    }
    let key = GfxKey {
        vs: vs.tr.key,
        ps: ps.map(|p| p.tr.key).unwrap_or(0),
        rect_gs: rect.then_some(vs.tr.out.param_mask),
        colors,
        blend,
        masks,
        depth: depth.map(|d| d.as_raw()).unwrap_or(vk::Format::UNDEFINED.as_raw()),
        topology_class: class,
        depth_clamp: g.clip & (3 << 26) != 0,
    };
    let push_mode = match ps {
        Some(p) => super::push_fits(&[vs, p]),
        None => super::push_fits(&[vs]),
    };
    Some((key, super::push_key(hash_gfx_key(&key), push_mode), push_mode))
}

/// Graphics pipelines being created, by cache hash: a thread that needs
/// one waits for it instead of compiling it again (compiles ahead of use).
static GFX_FLIGHTS: parking_lot::Mutex<Option<HashMap<u64, std::sync::Arc<PipeFlight>>>> = parking_lot::Mutex::new(None);
/// Draws that waited for a pipeline in flight: count, nanoseconds, and
/// compiles that saved.
pub static GFX_FLIGHT_WAITS: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
/// Draws whose graphics pipeline was missing and the nanoseconds their
/// command processors spent creating it or waiting for it (first uses).
pub static GFX_FIRST_USES: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

fn gfx_known(hkey: u64) -> bool {
    gfx_pipe(hkey).is_some() || GFX_FLIGHTS.lock().as_ref().is_some_and(|m| m.contains_key(&hkey))
}

/// Creates the graphics pipeline of a draw scanned ahead of its use
/// (`gpu::compile` workers): Some(true) when created here, Some(false) when
/// it existed or was in flight, None when such a draw records nothing or
/// creating it failed.
pub(crate) fn precompile_gfx(g: &GfxRegs, vs: &Prepared, ps: Option<&Prepared>) -> Option<bool> {
    let (key, hkey, push_mode) = gfx_key_ahead(g, vs, ps)?;
    if gfx_known(hkey) {
        return Some(false);
    }
    let created = std::cell::Cell::new(false);
    let r = gfx_pipe_with(&key, hkey, (&**vs.tr, vs.addr), ps.map(|p| (&**p.tr, p.addr)), push_mode, &created);
    if compile_trace() && created.get() {
        AHEAD_KEYS.lock().get_or_insert_with(HashSet::new).insert(hkey);
        crate::info!("compile ahead: vs {:#x} ps {:#x} key {hkey:#x} created", vs.addr, ps.map_or(0, |p| p.addr));
    }
    r.ok().map(|_| created.get())
}

/// The graphics pipeline of `key` (created on first use; one in flight on
/// another thread is waited for).
/// A stage of a graphics pipeline: its translation and program address.
type StageTr<'a> = (&'a super::super::shaders::Translated, u64);

fn gfx_pipe_with(key: &GfxKey, hkey: u64, vs: StageTr, ps: Option<StageTr>, push_mode: bool, created: &std::cell::Cell<bool>) -> Result<Pipe, String> {
    if let Some(x) = gfx_pipe(hkey) {
        return Ok(x);
    }
    let (mine, flight) = {
        let mut g = GFX_FLIGHTS.lock();
        let m = g.get_or_insert_with(HashMap::new);
        match m.get(&hkey) {
            Some(f) => (false, f.clone()),
            None => {
                let f = std::sync::Arc::new(PipeFlight::new());
                m.insert(hkey, f.clone());
                (true, f)
            }
        }
    };
    if !mine {
        let t = std::time::Instant::now();
        flight.waited();
        let r = {
            let mut d = flight.done.lock();
            while d.is_none() {
                flight.cv.wait(&mut d);
            }
            d.clone().unwrap()
        };
        GFX_FLIGHT_WAITS[0].fetch_add(1, Ordering::Relaxed);
        GFX_FLIGHT_WAITS[1].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        if r.is_ok() {
            GFX_FLIGHT_WAITS[2].fetch_add(1, Ordering::Relaxed);
            return r;
        }
        return gfx_pipe_create(key, hkey, vs, ps, push_mode, created);
    }
    let stored = gfx_pipe(hkey);
    // A panic must not leave the threads waiting for it blocked.
    let r = match stored {
        Some(x) => Ok(x),
        None => std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| gfx_pipe_create(key, hkey, vs, ps, push_mode, created))).unwrap_or_else(|_| Err("pipeline creation panicked".to_string())),
    };
    *flight.done.lock() = Some(r.clone());
    flight.cv.notify_all();
    if let Some(m) = GFX_FLIGHTS.lock().as_mut() {
        m.remove(&hkey);
    }
    r
}

fn gfx_pipe_create(key: &GfxKey, hkey: u64, vs: StageTr, ps: Option<StageTr>, push_mode: bool, created: &std::cell::Cell<bool>) -> Result<Pipe, String> {
    created.set(true);
    let t0 = std::time::Instant::now();
    let p = graphics_pipeline(key, hkey, vs, ps, push_mode)?;
    // What a later session makes ahead when the title creates these
    // shaders' objects (`gpu::precompile`).
    if super::super::tune::precompile() > 0 {
        let state = GfxState { rect: key.rect_gs.is_some(), colors: key.colors, blend: key.blend, masks: key.masks, depth: key.depth, topology_class: key.topology_class, depth_clamp: key.depth_clamp };
        super::super::precompile::learn_gfx(vs, ps, &state);
    }
    super::PIPES_CREATED[1].fetch_add(1, Ordering::Relaxed);
    super::PIPES_NS[1].fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
    // Pipelines whose shader pair an earlier pipeline had (other render
    // state only: what pipeline libraries would link without compiling the
    // shaders again).
    if !GFX_PAIRS.lock().insert((key.vs, key.ps, key.rect_gs, key.depth_clamp)) {
        GFX_REPEAT_PAIRS[0].fetch_add(1, Ordering::Relaxed);
        GFX_REPEAT_PAIRS[1].fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
    if p.push {
        super::PIPES_PUSH[1].fetch_add(1, Ordering::Relaxed);
    }
    let ms = t0.elapsed().as_millis();
    if ms >= 200 {
        crate::info!("gpu: graphics pipeline for vs {:#x} ps {:#x} took {ms} ms", vs.1, ps.map_or(0, |p| p.1));
    }
    Ok(keep_gfx_pipe(hkey, p))
}

fn set_layout_bindings(stages: &[(&super::super::shaders::Translated, vk::ShaderStageFlags)]) -> Vec<vk::DescriptorSetLayoutBinding<'static>> {
    let mut v = Vec::new();
    for (tr, flags) in stages {
        for b in super::descriptor_bindings(&tr.out) {
            v.push(vk::DescriptorSetLayoutBinding::default().binding(b.binding).descriptor_type(descriptor_type(b.kind)).descriptor_count(super::binding_count(b.kind)).stage_flags(*flags));
        }
    }
    v
}

/// Creates a graphics pipeline for dynamic rendering (`push_mode`: a push
/// descriptor layout if it fits, `push_layout`), through the pipeline cache
/// file of `hkey`.
/// Shader pairs graphics pipelines were created for (`GFX_REPEAT_PAIRS`).
static GFX_PAIRS: parking_lot::Mutex<std::collections::BTreeSet<(u64, u64, Option<u32>, bool)>> = parking_lot::Mutex::new(std::collections::BTreeSet::new());
/// Graphics pipelines created for a shader pair already seen with other
/// render state, and their nanoseconds (status).
pub static GFX_REPEAT_PAIRS: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// Render area of instances without attachments (`tune attachless_draws`).
const ATTACHLESS_AREA: u32 = 16384;
/// Draws recorded without attachments (status).
pub static ATTACHLESS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn graphics_pipeline(key: &GfxKey, hkey: u64, vs: StageTr, ps: Option<StageTr>, push_mode: bool) -> Result<Pipe, String> {
    let c = ctx();
    unsafe {
        let mut stages_p = vec![(vs.0, vk::ShaderStageFlags::VERTEX)];
        if let Some(p) = ps {
            stages_p.push((p.0, vk::ShaderStageFlags::FRAGMENT));
        }
        let bindings = set_layout_bindings(&stages_p);
        let (push, flags) = super::push_layout(&bindings, push_mode);
        let set_layout = c.device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().flags(flags).bindings(&bindings), None).map_err(|e| format!("set layout: {e}"))?;
        let layouts = [set_layout];
        let ranges = super::sgpr_pc_ranges(vk::ShaderStageFlags::VERTEX, vs.0.out.sgpr_pc.is_some() || ps.is_some_and(|p| p.0.out.sgpr_pc.is_some()));
        let layout = c.device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts).push_constant_ranges(ranges), None).map_err(|e| format!("pipeline layout: {e}"))?;
        let module = |code: &[u32]| c.device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&super::module_code(code)), None).map_err(|e| format!("shader module: {e}"));
        let mut modules = vec![(vk::ShaderStageFlags::VERTEX, module(&vs.0.out.spirv)?)];
        if let Some(mask) = key.rect_gs {
            modules.push((vk::ShaderStageFlags::GEOMETRY, module(&rdna::translate::gs::rect_list_gs(mask))?));
        }
        if let Some(p) = ps {
            modules.push((vk::ShaderStageFlags::FRAGMENT, module(&p.0.out.spirv)?));
        }
        let stages: Vec<vk::PipelineShaderStageCreateInfo> = modules.iter().map(|(s, m)| vk::PipelineShaderStageCreateInfo::default().stage(*s).module(*m).name(c"main")).collect();
        let vi = vk::PipelineVertexInputStateCreateInfo::default();
        let topo = match key.topology_class {
            0 => vk::PrimitiveTopology::POINT_LIST,
            1 => vk::PrimitiveTopology::LINE_LIST,
            _ => vk::PrimitiveTopology::TRIANGLE_LIST,
        };
        let ia = vk::PipelineInputAssemblyStateCreateInfo::default().topology(topo);
        let vps = vk::PipelineViewportStateCreateInfo::default().viewport_count(1).scissor_count(1);
        let rs = vk::PipelineRasterizationStateCreateInfo::default().depth_clamp_enable(key.depth_clamp).polygon_mode(vk::PolygonMode::FILL).line_width(1.0);
        let ms = vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let ds = vk::PipelineDepthStencilStateCreateInfo::default();
        let n = key.colors.iter().rposition(|&f| f != vk::Format::UNDEFINED.as_raw()).map(|i| i + 1).unwrap_or(0);
        let formats: Vec<vk::Format> = key.colors[..n].iter().map(|&f| vk::Format::from_raw(f)).collect();
        let atts: Vec<vk::PipelineColorBlendAttachmentState> = (0..n)
            .map(|i| {
                let b = key.blend[i];
                let enable = b & (1 << 30) != 0 && formats[i] != vk::Format::UNDEFINED;
                let sep = b & (1 << 29) != 0;
                let (cs, cf, cd) = (b & 0x1F, (b >> 5) & 7, (b >> 8) & 0x1F);
                let (asrc, af, ad) = if sep { ((b >> 16) & 0x1F, (b >> 21) & 7, (b >> 24) & 0x1F) } else { (cs, cf, cd) };
                vk::PipelineColorBlendAttachmentState::default()
                    .blend_enable(enable)
                    .src_color_blend_factor(blend_factor(cs))
                    .dst_color_blend_factor(blend_factor(cd))
                    .color_blend_op(blend_op(cf))
                    .src_alpha_blend_factor(blend_factor(asrc))
                    .dst_alpha_blend_factor(blend_factor(ad))
                    .alpha_blend_op(blend_op(af))
                    .color_write_mask(vk::ColorComponentFlags::from_raw(key.masks[i] as u32))
            })
            .collect();
        let cb = vk::PipelineColorBlendStateCreateInfo::default().attachments(&atts);
        let dyns = [
            vk::DynamicState::VIEWPORT,
            vk::DynamicState::SCISSOR,
            vk::DynamicState::LINE_WIDTH,
            vk::DynamicState::DEPTH_BIAS,
            vk::DynamicState::BLEND_CONSTANTS,
            vk::DynamicState::STENCIL_COMPARE_MASK,
            vk::DynamicState::STENCIL_WRITE_MASK,
            vk::DynamicState::STENCIL_REFERENCE,
            vk::DynamicState::CULL_MODE,
            vk::DynamicState::FRONT_FACE,
            vk::DynamicState::PRIMITIVE_TOPOLOGY,
            vk::DynamicState::DEPTH_TEST_ENABLE,
            vk::DynamicState::DEPTH_WRITE_ENABLE,
            vk::DynamicState::DEPTH_COMPARE_OP,
            vk::DynamicState::DEPTH_BOUNDS_TEST_ENABLE,
            vk::DynamicState::STENCIL_TEST_ENABLE,
            vk::DynamicState::STENCIL_OP,
            vk::DynamicState::DEPTH_BIAS_ENABLE,
            vk::DynamicState::PRIMITIVE_RESTART_ENABLE,
            vk::DynamicState::RASTERIZER_DISCARD_ENABLE,
        ];
        let dy = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dyns);
        let depth = vk::Format::from_raw(key.depth);
        let mut rinfo = vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&formats).depth_attachment_format(depth);
        if images::has_stencil(depth) {
            rinfo = rinfo.stencil_attachment_format(depth);
        }
        let gpci = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vi)
            .input_assembly_state(&ia)
            .viewport_state(&vps)
            .rasterization_state(&rs)
            .multisample_state(&ms)
            .depth_stencil_state(&ds)
            .color_blend_state(&cb)
            .dynamic_state(&dy)
            .layout(layout)
            .flags(crate::gpu::pstats::flags())
            .push_next(&mut rinfo);
        let r = crate::gpu::pcache::with_cache('g', hkey, |cache| c.device.create_graphics_pipelines(cache, &[gpci], None));
        for (_, m) in modules {
            c.device.destroy_shader_module(m, None);
        }
        let pipeline = r.map_err(|(_, e)| format!("graphics pipeline: {e}"))?[0];
        crate::gpu::pstats::report(pipeline, &format!("vs {:#x} ps {:#x}", vs.1, ps.map_or(0, |p| p.1)));
        Ok(Pipe { pipeline, layout, set_layout, push })
    }
}

/// Pixel shaders whose draws are limited to one pixel (control `tiny`), and
/// whether there are any (read by every draw without the lock).
pub static TINY_SCISSOR: parking_lot::Mutex<Vec<u64>> = parking_lot::Mutex::new(Vec::new());
pub static TINY_ANY: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

impl Exec {
    /// Executes a draw with the graphics state of `cp`; returns what
    /// happened (frame captures).
    pub fn draw(&mut self, cp: &Cp, args: DrawArgs) -> std::borrow::Cow<'static, str> {
        if DEVICE_LOST.load(Ordering::Relaxed) {
            DRAW_SKIPPED.fetch_add(1, Ordering::Relaxed);
            return "skipped (device lost)".into();
        }
        if super::super::tune::skip_cmds() & 1 != 0 {
            DRAW_SKIPPED.fetch_add(1, Ordering::Relaxed);
            return "skipped (`tune skip_cmds`)".into();
        }
        let outcome = match self.draw_inner(cp, &args) {
            Ok(true) => {
                DRAWN.fetch_add(1, Ordering::Relaxed);
                "drawn".into()
            }
            Ok(false) => {
                DRAW_NOOP.fetch_add(1, Ordering::Relaxed);
                "no effect".into()
            }
            Err(e) if e == super::STOPPED => "stopped (parallel run)".into(),
            Err(e) if e == NEEDS_CPU_ARGS => NEEDS_CPU_ARGS.into(),
            Err(e) => {
                DRAW_SKIPPED.fetch_add(1, Ordering::Relaxed);
                let s = format!("skipped: {e}");
                self.report(e);
                s.into()
            }
        };
        if sync_mode() {
            self.flush();
            if DEVICE_LOST.load(Ordering::Relaxed) {
                let vs = ((cp.reg(Space::Sh, 0xC8) as u64) | ((cp.reg(Space::Sh, 0xC9) as u64 & 0xFF) << 32)) << 8;
                let ps = ((cp.reg(Space::Sh, 0x08) as u64) | ((cp.reg(Space::Sh, 0x09) as u64 & 0xFF) << 32)) << 8;
                crate::error!("gpu[{}]: device lost by a draw (vs {vs:#x}, ps {ps:#x})", self.name);
            }
        }
        self.split_if_gpu_idle();
        outcome
    }

    /// Records an indexed indirect draw whose arguments the GPU reads at
    /// `args` as one more `vkCmdDrawIndexedIndirect` on the state the draw
    /// before left (pipeline, push descriptors, dynamic state, index buffer,
    /// rendering instance; `tune draw_repeat`): only when that draw was
    /// repeatable, nothing was recorded into the batch since, the index
    /// binding is the same and both homes still hold their ranges without a
    /// copy in (one would be recorded first: the draw is then recorded in
    /// full). The command processor calls it only when no packet but such
    /// draws came between them, so the registers are the same. False: not
    /// taken.
    pub fn draw_repeat(&mut self, args: u64, index: (u64, u32), index_bytes: u64) -> bool {
        let Some(r) = self.repeat.take() else { return false };
        if r.cpu || DEVICE_LOST.load(Ordering::Relaxed) || !self.recording || self.stopped || self.batch_count != r.batch || self.cmd_uses.get() != r.uses || self.rendering.is_none() || (index.0, index.1, index_bytes) != r.index || sync_mode() {
            return false;
        }
        let Some(ah) = self.home_bind(args, 20, false, HomeUse::Sized) else { return false };
        let Some(ih) = self.home_bind(index.0, index_bytes, false, HomeUse::Index) else { return false };
        if self.stopped || self.batch_count != r.batch || self.cmd_uses.get() != r.uses || ah.0 != r.args_buf || (ih.0, ih.1) != (r.index_buf, r.index_off) {
            return false;
        }
        unsafe { ctx().device.cmd_draw_indexed_indirect(self.slots[self.cur].cmd, ah.0, ah.1, 1, 20) };
        GPU_ARG_DRAWS.fetch_add(1, Ordering::Relaxed);
        REPEAT_DRAWS.fetch_add(1, Ordering::Relaxed);
        DRAWN.fetch_add(1, Ordering::Relaxed);
        self.args_read = true;
        self.repeat = Some(r);
        true
    }

    /// The same for an indexed indirect draw whose arguments the CPU read
    /// (`tune cpu_args_base`): `index` is its first index's address and the
    /// index size, the others its arguments. Its indices must lie in the
    /// home the draw before bound (bound again at their offset).
    pub fn draw_repeat_cpu(&mut self, index: (u64, u32), count: u32, instances: u32, base_vertex: i32, first_instance: u32) -> bool {
        let Some(mut r) = self.repeat.take() else { return false };
        if !r.cpu || DEVICE_LOST.load(Ordering::Relaxed) || !self.recording || self.stopped || self.batch_count != r.batch || self.cmd_uses.get() != r.uses || self.rendering.is_none() || index.1 != r.index.1 || sync_mode() {
            return false;
        }
        if count == 0 || instances == 0 {
            // Nothing to draw (the full path's "no effect").
            DRAW_NOOP.fetch_add(1, Ordering::Relaxed);
            self.repeat = Some(r);
            return true;
        }
        let bytes = count as u64 * index.1 as u64;
        let Some(ih) = self.home_bind(index.0, bytes, false, HomeUse::Index) else { return false };
        if self.stopped || self.batch_count != r.batch || self.cmd_uses.get() != r.uses || ih.0 != r.index_buf {
            return false;
        }
        let cmd = self.slots[self.cur].cmd;
        let c = ctx();
        unsafe {
            if ih.1 != r.index_off {
                let ty = if index.1 == 4 { vk::IndexType::UINT32 } else { vk::IndexType::UINT16 };
                c.device.cmd_bind_index_buffer(cmd, ih.0, ih.1, ty);
                r.index_off = ih.1;
            }
            c.device.cmd_draw_indexed(cmd, count, instances, 0, base_vertex, first_instance);
        }
        REPEAT_DRAWS.fetch_add(1, Ordering::Relaxed);
        DRAWN.fetch_add(1, Ordering::Relaxed);
        self.repeat = Some(r);
        true
    }

    /// Whether the draw just recorded can be repeated (`draw_repeat`).
    pub fn repeat_ready(&self) -> bool {
        self.repeat.is_some_and(|r| r.uses == self.cmd_uses.get() && r.batch == self.batch_count)
    }

    /// Ends the active rendering instance; its attachment writes become
    /// visible to later commands.
    #[track_caller]
    pub(super) fn end_rendering(&mut self) {
        if self.rendering.take().is_some() {
            unsafe { ctx().device.cmd_end_rendering(self.cmd()) };
            self.ops[super::OP_PASS_ENDS] += 1;
            if super::super::tune::skip_barriers() & 1 == 0 {
                self.barrier();
            }
            self.barrier_wanted = false;
            self.dispatched = false;
        }
    }

    /// Ends the active rendering instance for a caller that records a full
    /// barrier next, before anything else touches memory: that barrier
    /// makes the attachment writes visible (`tune merge_barriers`; ending
    /// with a barrier of its own drained the GPU twice in a row).
    #[track_caller]
    pub(super) fn end_rendering_before_barrier(&mut self) {
        if !super::super::tune::merge_barriers() {
            self.end_rendering();
        } else if self.rendering.take().is_some() {
            unsafe { ctx().device.cmd_end_rendering(self.cmd()) };
            self.ops[super::OP_PASS_ENDS] += 1;
            MERGED_PASS_BARRIERS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// `end_rendering_before_barrier`, then that barrier.
    #[track_caller]
    pub(super) fn end_rendering_then_barrier(&mut self) {
        self.end_rendering_before_barrier();
        self.barrier();
    }

    /// Ends the active rendering instance for a dispatch or for draws to
    /// other attachments: with `tune lazy_passes` without a barrier of its
    /// own (the title flushes and waits before work that uses what the
    /// instance drew, which arms the lazy barrier; draws after dispatches
    /// wait for them), else as `end_rendering`.
    #[track_caller]
    pub(super) fn end_rendering_lazy(&mut self) {
        if !super::super::tune::lazy_passes() {
            self.end_rendering();
        } else if self.rendering.take().is_some() {
            unsafe { ctx().device.cmd_end_rendering(self.cmd()) };
            self.ops[super::OP_PASS_ENDS] += 1;
        }
    }

    fn begin_rendering(&mut self, colors: [Option<Target>; MAX_RT], depth: Option<Target>, area: (u32, u32)) {
        let n = colors.iter().rposition(Option::is_some).map(|i| i + 1).unwrap_or(0);
        let color_infos: Vec<vk::RenderingAttachmentInfo> = colors[..n]
            .iter()
            .map(|t| match t {
                Some(t) => vk::RenderingAttachmentInfo::default().image_view(t.view).image_layout(vk::ImageLayout::GENERAL).load_op(vk::AttachmentLoadOp::LOAD).store_op(vk::AttachmentStoreOp::STORE),
                None => vk::RenderingAttachmentInfo::default().image_view(vk::ImageView::null()),
            })
            .collect();
        let depth_info = depth.map(|d| vk::RenderingAttachmentInfo::default().image_view(d.view).image_layout(vk::ImageLayout::GENERAL).load_op(vk::AttachmentLoadOp::LOAD).store_op(vk::AttachmentStoreOp::STORE));
        let mut ri = vk::RenderingInfo::default()
            .render_area(vk::Rect2D { offset: vk::Offset2D::default(), extent: vk::Extent2D { width: area.0, height: area.1 } })
            .layer_count(1)
            .color_attachments(&color_infos);
        if let (Some(d), Some(t)) = (&depth_info, &depth) {
            if images::aspects(t.format).contains(vk::ImageAspectFlags::DEPTH) {
                ri = ri.depth_attachment(d);
            }
            if images::has_stencil(t.format) {
                ri = ri.stencil_attachment(d);
            }
        }
        super::note_vk(super::VK_RENDER_PASSES);
        unsafe { ctx().device.cmd_begin_rendering(self.cmd(), &ri) };
        self.rendering = Some(Rendering { colors, depth, area, written: Vec::new(), sampled: Vec::new(), ds_written: [false; 2], ds_sampled: [false; 2] });
    }

    /// Records the draw; Ok(false) when it has no effect here.
    fn draw_inner(&mut self, cp: &Cp, a: &DrawArgs) -> Result<bool, String> {
        let gpu = a.gpu_args;
        if gpu.is_none() && (a.count == 0 || a.instances == 0) {
            return Ok(false);
        }
        // Phase times (DS_PROFILE_GPU only: a clock read costs ~20 ns).
        let t0 = super::profiling().then(std::time::Instant::now);
        let since = |t0: Option<std::time::Instant>| t0.map_or(std::time::Duration::ZERO, |t| t.elapsed());
        let kind = if self.deferred { 2 } else if cp.in_parallel_run() { 1 } else { 0 };
        let first = self.batch_shaders.is_empty();
        let mut lap = super::Laps::for_draw(kind, first);
        let reg = |r: u32| ctx_reg(cp, r);
        // CB_COLOR_CONTROL.MODE: 0 color disabled, 1 normal; others are
        // metadata passes (fast-clear eliminate, FMASK/DCC decompress, resolve)
        // that have nothing to do on uncompressed single-sample images.
        let cb_mode = (reg(0x28808) >> 4) & 7;
        if cb_mode > 1 {
            if cb_mode == 3 {
                self.report("MSAA resolve pass not implemented (skipped)".into());
            }
            return Ok(false);
        }
        let clip = reg(0x28810);
        if clip & (1 << 22) != 0 {
            return Ok(false); // DX_RASTERIZATION_KILL
        }
        if SKIP_VS_ANY.load(Ordering::Relaxed) && SKIP_VS.lock().as_ref().is_some_and(|s| s.contains(&super::super::shaders::ngg_addr(cp))) {
            return Err("vertex shader on the skip list (diagnosis)".into());
        }
        let prim = cp.reg(Space::Uc, 0x242) & 0x3F;
        let (topology, class, rect) = match prim {
            1 => (vk::PrimitiveTopology::POINT_LIST, 0u8, false),
            2 => (vk::PrimitiveTopology::LINE_LIST, 1, false),
            3 => (vk::PrimitiveTopology::LINE_STRIP, 1, false),
            4 => (vk::PrimitiveTopology::TRIANGLE_LIST, 2, false),
            5 => (vk::PrimitiveTopology::TRIANGLE_FAN, 2, false),
            6 => (vk::PrimitiveTopology::TRIANGLE_STRIP, 2, false),
            // Agc rect list (7, see PROVENANCE) and the hardware RECTLIST.
            7 | 0x11 => (vk::PrimitiveTopology::TRIANGLE_LIST, 2, true),
            p => return Err(format!("primitive type {p:#x} not supported (draw skipped)")),
        };
        let aa = reg(0x28BE0);
        if aa & 7 != 0 {
            self.report(format!("MSAA ({} samples) rendered single-sampled", 1 << (aa & 7)));
        }
        // Shader stages.
        let ps_addr = ((cp.reg(Space::Sh, 0x08) as u64) | ((cp.reg(Space::Sh, 0x09) as u64 & 0xFF) << 32)) << 8;
        // Stages prepared ahead by the workers (`gpu::ahead`) when they still
        // hold, else here. A draw whose arguments the GPU reads takes the
        // start instance on the GPU (prepared here).
        // The start instance from gl_BaseInstance: arguments the GPU reads,
        // or CPU-read ones drawn with it as Vulkan's first instance.
        let base_slot = gpu.map(|g| g.inst_slot).or(a.base_inst);
        let prep_vs = |cp: &Cp| match base_slot {
            Some(s) => shaders::with_base_instance(s, || shaders::prepare_ngg(cp)),
            None => shaders::prepare_ngg(cp),
        };
        let [pre_vs, pre_ps] = cp.take_ahead();
        let pre_vs = if base_slot.is_some() { None } else { pre_vs };
        let mut vs = match pre_vs.and_then(|s| crate::gpu::ahead::take(&s, cp, true)) {
            Some(p) => p,
            None => prep_vs(cp)?,
        };
        if let Some(slot) = base_slot {
            if shaders::uses_user_word(&vs, slot) {
                shaders::note_cpu_args_program(vs.addr);
                return Err(NEEDS_CPU_ARGS.into());
            }
        }
        lap.mark(0);
        let mut ps = if ps_addr != 0 {
            match pre_ps.and_then(|s| crate::gpu::ahead::take(&s, cp, true)) {
                Some(p) => Some(p),
                None => Some(shaders::prepare_ps(cp)?),
            }
        } else {
            None
        };
        lap.mark(1);
        // Data GPU work still writes (the title's per-draw transforms, written
        // by a dispatch just before): read by the shaders themselves (`tune
        // gpu_loads 3`; data the title writes stays captured when recorded).
        if super::super::tune::gpu_loads_mode() == 3 {
            if !vs.data_loads.is_empty() && self.would_wait(&shaders::data_load_ranges(&vs)) {
                shaders::note_gpu_loads_program(vs.addr);
                vs = shaders::with_gpu_loads(|| prep_vs(cp))?;
                super::HYBRID_LOADS[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            if ps.as_ref().is_some_and(|p| !p.data_loads.is_empty() && self.would_wait(&shaders::data_load_ranges(p))) {
                shaders::note_gpu_loads_program(ps.as_ref().map_or(0, |p| p.addr));
                ps = Some(shaders::with_gpu_loads(|| shaders::prepare_ps(cp))?);
                super::HYBRID_LOADS[1].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
        self.sync_readers = [vs.addr, ps.as_ref().map_or(0, |p| p.addr)];
        // Each stage's reads in place (no combined copy per draw), unless the
        // stages are used as prepared (`speculate`).
        let patches = match &ps {
            Some(p) => self.speculate(&[&vs, p]),
            None => self.speculate(&[&vs]),
        };
        let waited = patches.is_none() && (self.sync_for_stage(&vs.reads, SYNC_DRAW_READS) | ps.as_ref().is_some_and(|p| self.sync_for_stage(&p.reads, SYNC_DRAW_READS)));
        if self.stopped {
            return Err(super::STOPPED.into());
        }
        if waited {
            if let Some(c) = self.sync_culprit.take() {
                let mut st: Vec<&Prepared> = vec![&vs];
                if let Some(p) = &ps {
                    st.push(p);
                }
                super::explain_wait(&self.name, c, &st);
            }
            let same = shaders::words_unchanged(&vs) && ps.as_ref().is_none_or(shaders::words_unchanged);
            super::READ_WAIT_WORDS[if same { 0 } else { 1 }].fetch_add(1, Ordering::Relaxed);
            vs = prep_vs(cp)?;
            ps = if ps_addr != 0 { Some(shaders::prepare_ps(cp)?) } else { None };
        }
        lap.mark(2);
        // Measurement: what differs from the last draw of the same programs
        // on this executor (batching candidates; profiling only).
        if super::profiling() {
            let sig = draw_sig(&vs, ps.as_ref());
            note_draw_diff(&mut self.last_draw_sig, sig);
        }
        let t_prep = since(t0);
        let mut textures = std::mem::take(&mut self.tex_scratch);
        textures.clear();
        super::texture_ranges_into(&vs, &mut textures);
        if let Some(p) = &ps {
            super::texture_ranges_into(p, &mut textures);
        }
        self.sync_textures(&textures, SYNC_DRAW_TEXTURES);
        self.tex_scratch = textures;
        if self.stopped {
            return Err(super::STOPPED.into());
        }
        lap.mark(3);
        // Indices in a video-memory home (`homes`; one is made for indices
        // that GPU work wrote through a large binding) are bound from it and
        // read in order on the GPU; others are waited for if GPU work writes
        // them.
        let index_bytes = |size: u32| gpu.map_or(a.count as u64 * size as u64, |g| g.index_bytes);
        let index_home = match a.index {
            Some((addr, size)) => {
                let bytes = index_bytes(size);
                let h = self.home_bind(addr, bytes, false, super::HomeUse::Index);
                if h.is_none() {
                    if gpu.is_some() {
                        return Err(format!("index buffer {addr:#x}+{bytes:#x} of a draw reading its arguments on the GPU is not in a video-memory home (draw skipped)"));
                    }
                    self.sync_for(&[(addr, bytes)], SYNC_DRAW_INDICES);
                }
                if self.stopped {
                    return Err(super::STOPPED.into());
                }
                h
            }
            None => None,
        };
        // Arguments the GPU reads: from their video-memory home.
        let args_home = match gpu {
            Some(g) => {
                let h = self.home_bind(g.addr, 20, false, super::HomeUse::Sized);
                if self.stopped {
                    return Err(super::STOPPED.into());
                }
                match h {
                    Some(h) => Some(h),
                    None => return Err(format!("indirect arguments {:#x} are no longer in a video-memory home (draw skipped)", g.addr)),
                }
            }
            None => None,
        };
        lap.mark(4);
        // Color targets: enabled in CB_TARGET_MASK and CB_SHADER_MASK,
        // exported by the pixel shader with a non-zero export format.
        let mrt_mask = ps.as_ref().map(|p| p.tr.out.mrt_mask).unwrap_or(0);
        let col_format = reg(0x28714);
        let target_mask = reg(0x28238) & reg(0x2823C);
        let mut colors: [Option<Target>; MAX_RT] = [None; MAX_RT];
        let mut blend = [0u32; MAX_RT];
        let mut masks = [0u8; MAX_RT];
        if cb_mode == 1 {
            for i in 0..MAX_RT as u32 {
                let m = (target_mask >> (4 * i)) & 0xF;
                if m == 0 || mrt_mask & (1 << i) == 0 || (col_format >> (4 * i)) & 0xF == 0 {
                    continue;
                }
                let Some(s) = color_surface(cp, i)? else { continue };
                let t = images::target(&s)?;
                if colors.iter().flatten().any(|c| c.view == t.view) {
                    self.report(format!("color target {:#x} bound to two MRT slots (second ignored)", s.base));
                    continue;
                }
                colors[i as usize] = Some(t);
                blend[i as usize] = reg(0x28780 + 4 * i);
                masks[i as usize] = m as u8;
            }
        }
        lap.mark(5);
        // Depth/stencil.
        let dctl = reg(0x28800);
        let db_render = reg(0x28000);
        let dsurf = depth_surface(cp);
        let (has_z, has_s) = match &dsurf {
            Some(s) => (images::aspects(s.format).contains(vk::ImageAspectFlags::DEPTH) && reg(0x28040) & 3 != 0, images::has_stencil(s.format)),
            None => (false, false),
        };
        let z_test = has_z && dctl & 2 != 0;
        let z_write = z_test && dctl & 4 != 0;
        let s_test = has_s && dctl & 1 != 0;
        let sctl = reg(0x2842C);
        let refm = reg(0x28430);
        let refm_bf = reg(0x28434);
        let s_write = s_test && (refm | refm_bf) & 0x00FF_0000 != 0 && sctl & 0xFF_FFFF != 0;
        let clear_z = has_z && db_render & 1 != 0;
        let clear_s = has_s && db_render & 2 != 0;
        if z_test || s_test || clear_z || clear_s {
            // A view of a later slice than the depth image has: grow it
            // (after the work that used it) instead of a separate image.
            let s = dsurf.as_ref().unwrap();
            if let Some(old) = images::depth_to_grow(s.base, s.width, s.height, s.format, s.layers) {
                self.end_rendering();
                self.flush_for("depth growth");
                if self.stopped {
                    return Err(super::STOPPED.into());
                }
                images::with(|im| im.grow_depth(old, s.layers))?;
            }
        }
        let depth = if z_test || s_test || clear_z || clear_s { Some(images::target(dsurf.as_ref().unwrap())?) } else { None };
        lap.mark(6);
        let mem_writes = vs.tr.plan.resources.iter().any(|r| r.written) || ps.as_ref().is_some_and(|p| p.tr.plan.resources.iter().any(|r| r.written));
        if colors.iter().all(Option::is_none) && !z_write && !s_write && !clear_z && !clear_s && !mem_writes {
            return Ok(false);
        }
        let attachless = colors.iter().all(Option::is_none) && depth.is_none();
        if attachless && !super::super::tune::attachless_draws() {
            return Err("draw with only memory writes (no attachments) not supported yet (skipped)".into());
        }
        let mut area = (u32::MAX, u32::MAX);
        for t in colors.iter().flatten().chain(depth.iter()) {
            area = (area.0.min(t.width), area.1.min(t.height));
        }
        if attachless {
            // Memory side effects only: a rendering instance without
            // attachments, shared by such draws; each draw's scissors (from
            // its registers) limit what it rasterizes.
            area = (ATTACHLESS_AREA, ATTACHLESS_AREA);
            ATTACHLESS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        let (x0, y0, x1, y1) = scissor(cp, area);
        if x1 <= x0 || y1 <= y0 {
            return Ok(false);
        }
        // Pipeline.
        let depth_clamp = clip & (3 << 26) != 0;
        let key = GfxKey {
            vs: vs.tr.key,
            ps: ps.as_ref().map(|p| p.tr.key).unwrap_or(0),
            rect_gs: rect.then_some(vs.tr.out.param_mask),
            colors: colors.map(|c| c.map(|t| t.format.as_raw()).unwrap_or(vk::Format::UNDEFINED.as_raw())),
            blend,
            masks,
            depth: depth.map(|d| d.format.as_raw()).unwrap_or(vk::Format::UNDEFINED.as_raw()),
            topology_class: class,
            depth_clamp,
        };
        let push_mode = match &ps {
            Some(p) => super::push_fits(&[&vs, p]),
            None => super::push_fits(&[&vs]),
        };
        // Consecutive draws mostly share their pipeline state: the key's
        // hash (also the persistent cache file's name, so its hasher stays)
        // is computed again only when the key changed on this thread.
        let hkey = {
            let last = LAST_GFX_KEY.with(|c| c.get()).filter(|(k, _)| *k == key).map(|(_, h)| h);
            let h = last.unwrap_or_else(|| {
                let h = hash_gfx_key(&key);
                LAST_GFX_KEY.with(|c| c.set(Some((key, h))));
                h
            });
            super::push_key(h, push_mode)
        };
        // DS_COMPILE_TRACE: the key a scan ahead would make from these
        // registers (`gfx_key_ahead`) checked against this one.
        if compile_trace() {
            let g = GfxRegs::capture(&|r| ctx_reg(cp, r), cp.reg(Space::Uc, 0x242));
            let ahead = gfx_key_ahead(&g, &vs, ps.as_ref());
            if ahead.map(|a| a.1) != Some(hkey) {
                static N: AtomicU64 = AtomicU64::new(0);
                if N.fetch_add(1, Ordering::Relaxed) < 40 {
                    let why = match ahead {
                        None => "none".to_string(),
                        Some((k, _, pm)) => format!(
                            "colors {} blend {} masks {} depth {} class {} clamp {} rect {} push {}",
                            k.colors == key.colors,
                            k.blend == key.blend,
                            k.masks == key.masks,
                            k.depth == key.depth,
                            k.topology_class == key.topology_class,
                            k.depth_clamp == key.depth_clamp,
                            k.rect_gs == key.rect_gs,
                            pm == push_mode
                        ),
                    };
                    crate::info!("compile ahead: key check: vs {:#x} ps {ps_addr:#x} key {hkey:#x} differs from the ahead form ({why}; colors {:?} against {:?})", vs.addr, key.colors, ahead.map(|a| a.0.colors));
                }
            }
        }
        let cached = gfx_pipe(hkey);
        let Pipe { pipeline, layout, set_layout, push } = match cached {
            Some(x) => {
                if compile_trace() && AHEAD_KEYS.lock().get_or_insert_with(HashSet::new).remove(&hkey) {
                    crate::info!("used ahead: vs {:#x} ps {ps_addr:#x} key {hkey:#x}", vs.addr);
                }
                x
            }
            None => {
                // A first use: the pipelines of the draws after this one are
                // compiled meanwhile (`gpu::compile`).
                let t = std::time::Instant::now();
                if crate::gpu::compile::gfx_at_first_uses() && !gfx_known(hkey) {
                    cp.prefetch_compiles();
                }
                let created = std::cell::Cell::new(false);
                let p = gfx_pipe_with(&key, hkey, (&**vs.tr, vs.addr), ps.as_ref().map(|p| (&**p.tr, p.addr)), push_mode, &created)?;
                if crate::gpu::compile::gfx_at_first_uses() {
                    crate::gpu::compile::note_draw(vs.addr, ps_addr, &GfxRegs::capture(&|r| ctx_reg(cp, r), cp.reg(Space::Uc, 0x242)));
                }
                if compile_trace() {
                    crate::info!("first use: vs {:#x} ps {ps_addr:#x} key {hkey:#x} after {} ms{}", vs.addr, t.elapsed().as_millis(), if created.get() { "" } else { " (made by another thread)" });
                }
                GFX_FIRST_USES[0].fetch_add(1, Ordering::Relaxed);
                GFX_FIRST_USES[1].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
                p
            }
        };
        let t_pipe = since(t0);
        lap.mark(7);
        // Profile: the step's parts by draw kind ([draws, grow ns, check
        // ns, pending, resolve ns]).
        let hz_t = profiling().then(std::time::Instant::now);
        // Depth arrays sampled with more layers than rendered so far.
        match &ps {
            Some(p) => self.grow_sampled_depth(&[&vs, p]),
            None => self.grow_sampled_depth(&[&vs]),
        }
        let hz_t1 = hz_t.map(|_| std::time::Instant::now());
        // Fast clears (metadata writes), reloads and block copies of the
        // attachments and textures.
        let mut used = std::mem::take(&mut self.keys_scratch);
        used.clear();
        used.extend(colors.iter().flatten().chain(depth.iter()).map(|t| t.key));
        let pending = match &ps {
            Some(p) => super::image_hazards(&[&vs, p], &mut used),
            None => super::image_hazards(&[&vs], &mut used),
        };
        let hz_t2 = hz_t.map(|_| std::time::Instant::now());
        if self.stopped {
            return Err(super::STOPPED.into());
        }
        if pending {
            self.resolve_fast_clears(&used);
            self.resolve_overwrites(&used);
            self.sync_block_textures(&used);
            self.sync_depth_views(&used);
            if self.stopped {
                return Err(super::STOPPED.into());
            }
        }
        if let (Some(t0), Some(t1), Some(t2)) = (hz_t, hz_t1, hz_t2) {
            let h = &super::HAZARD_STEPS[kind];
            h[0].fetch_add(1, Ordering::Relaxed);
            h[1].fetch_add((t1 - t0).as_nanos() as u64, Ordering::Relaxed);
            h[2].fetch_add((t2 - t1).as_nanos() as u64, Ordering::Relaxed);
            h[3].fetch_add(pending as u64, Ordering::Relaxed);
            h[4].fetch_add(t2.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
        self.keys_scratch = used;
        lap.mark(8);
        // Resources.
        match &ps {
            Some(p) => self.reserve(&[&vs, p], push),
            None => self.reserve(&[&vs], push),
        }
        if self.stopped {
            return Err(super::STOPPED.into());
        }
        if let Some(h) = &index_home {
            // Kept until the batch that draws completes (once per batch).
            if !self.keep_home(h.0, h.3.as_ref()) {
                return Err("the index buffer's video-memory home was dropped during the draw (draw skipped)".into());
            }
        }
        if let Some(h) = &args_home {
            if !self.keep_home(h.0, h.3.as_ref()) {
                return Err("the indirect arguments' video-memory home was dropped during the draw (draw skipped)".into());
            }
        }
        // Indices in guest memory that a resident shadow holds newer data for.
        if let Some((addr, size)) = a.index {
            let t = profiling().then(std::time::Instant::now);
            self.flush_resident(Some(&[(addr, index_bytes(size))]))?;
            if let Some(t) = t {
                super::BEGIN_STATS[2].fetch_add(1, Ordering::Relaxed);
                super::BEGIN_STATS[3].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
            }
        }
        lap.mark(9);
        // The vector is kept for the next draw (empty: it drains at the end).
        let mut bounds = std::mem::take(&mut self.draw_bounds);
        bounds.clear();
        for k in 0..2 {
            let p: &mut Prepared = match k {
                0 => &mut vs,
                _ => match ps.as_mut() {
                    Some(p) => p,
                    None => break,
                },
            };
            let mut b = self.bind_stage(p, false)?;
            if self.stopped {
                return Err(super::STOPPED.into());
            }
            let (ring, at) = self.bind_sgpr(p, &mut b)?;
            if let Some(pt) = &patches {
                self.patch_sgpr(pt, k, ring, at);
            }
            bounds.push(b);
            lap.mark(10 + k.min(1));
        }
        if profiling() {
            // Buffer placements of the draw's stages (vertex stage first), under
            // the draw's profile key.
            let all: Vec<_> = bounds.iter().flat_map(|b| b.placements.iter().copied()).collect();
            super::note_placements(super::DRAW_KEY | if ps_addr != 0 { ps_addr } else { vs.addr }, [0; 3], false, &all);
        }
        let set = if push { None } else { Some(self.alloc_set(set_layout)?) };
        lap.mark(12);
        if let Some(set) = set {
            self.write_set(set, &bounds);
        }
        lap.mark(13);
        let t_bind = since(t0);
        // Rendering instance: reuse it while the attachments stay the same
        // and no attachment is both written and sampled in it (Vulkan's
        // rule for an image that is an attachment and a descriptor at once;
        // `tune feedback_exact`, else any sampled attachment ends it: ~266
        // instances per gate frame, each with a full barrier).
        let writes_ds = z_write || s_write || clear_z || clear_s;
        // This draw's writes of the depth attachment's [depth, stencil].
        let ds_writes = [z_write || clear_z, s_write || clear_s];
        let draw_writes = |img: vk::Image| colors.iter().flatten().any(|t| t.image == img) || (writes_ds && depth.as_ref().is_some_and(|d| d.image == img));
        let sampled_now = |img: vk::Image| bounds.iter().any(|b| b.images.contains(&img));
        let stored_now = |img: vk::Image| bounds.iter().any(|b| b.stored.contains(&img));
        // Its sampling of the depth attachment's [depth, stencil] (any view
        // that is not R8_UINT reads depth).
        let ds_sampled_now = |img: vk::Image| [bounds.iter().any(|b| b.images.contains(&img) && !b.stencil_views.contains(&img)), bounds.iter().any(|b| b.stencil_views.contains(&img))];
        let exact = super::super::tune::feedback_exact();
        let mut samples_attachment = false;
        let feedback = self.rendering.as_ref().is_some_and(|r| {
            let color = r.colors.iter().flatten().any(|t| {
                let a = t.image;
                let sampled = sampled_now(a);
                samples_attachment |= sampled;
                if !exact {
                    return sampled;
                }
                stored_now(a) || (sampled && (draw_writes(a) || r.written.contains(&a))) || (draw_writes(a) && r.sampled.contains(&a))
            });
            let ds = r.depth.as_ref().is_some_and(|t| {
                let a = t.image;
                let sampled = sampled_now(a);
                samples_attachment |= sampled;
                if !exact {
                    return sampled;
                }
                let now = ds_sampled_now(a);
                stored_now(a) || (0..2).any(|k| (now[k] && (ds_writes[k] || r.ds_written[k])) || (ds_writes[k] && r.ds_sampled[k]))
            });
            color || ds
        });
        let same = self.rendering.as_ref().is_some_and(|r| r.colors == colors && r.depth == depth);
        if samples_attachment && same {
            super::FEEDBACK[feedback as usize].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            // What the sampled attachment is and how this draw uses it
            // (statistics): [color written by the draw, color not written,
            // depth with depth writes, with stencil writes only, with a
            // clear only, depth not written, stored to].
            let r = self.rendering.as_ref().unwrap();
            let mut kinds = [false; 7];
            for t in r.colors.iter().flatten() {
                if sampled_now(t.image) {
                    kinds[if draw_writes(t.image) { 0 } else { 1 }] = true;
                }
                if stored_now(t.image) {
                    kinds[6] = true;
                }
            }
            if r.depth.as_ref().is_some_and(|d| sampled_now(d.image)) {
                kinds[if z_write { 2 } else if s_write { 3 } else if clear_z || clear_s { 4 } else { 5 }] = true;
            }
            for (k, &on) in kinds.iter().enumerate() {
                if on {
                    super::FEEDBACK_KINDS[k].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
            }
            if super::profiling_on() {
                let ps_addr = ps.as_ref().map_or(0, |p| p.addr);
                *super::FEEDBACK_PS.lock().get_or_insert_with(std::collections::HashMap::new).entry(ps_addr).or_default() += 1;
            }
        }
        if !same || feedback {
            if feedback {
                self.end_rendering();
            } else {
                self.end_rendering_lazy();
            }
            self.barrier_if_wanted(true);
            self.begin_rendering(colors, depth, area);
        } else if super::super::tune::race_checks() && self.barrier_wanted {
            super::UNORDERED[4].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        if super::super::tune::race_checks() {
            self.note_unordered(&bounds, &[vs.addr, ps_addr], true);
        }
        if let Some(r) = self.rendering.as_mut() {
            for t in r.colors.iter().flatten() {
                let a = t.image;
                if draw_writes(a) && !r.written.contains(&a) {
                    r.written.push(a);
                }
                if sampled_now(a) && !r.sampled.contains(&a) {
                    r.sampled.push(a);
                }
            }
            if let Some(a) = r.depth.as_ref().map(|t| t.image) {
                let now = ds_sampled_now(a);
                for k in 0..2 {
                    r.ds_written[k] |= ds_writes[k];
                    r.ds_sampled[k] |= now[k];
                }
            }
        }
        lap.mark(14);
        let c = ctx();
        let cmd = self.cmd();
        crate::gpu::vk::checkpoint(cmd, crate::gpu::vk::mark::DRAW, ps_addr);
        // Viewport 0 (PA_CL_VPORT_*): the hardware window transform maps
        // directly onto a Vulkan viewport (negative height for a flipped Y).
        let vte = reg(0x28818);
        let (xs, xo, ys, yo, zs, zo) = (f32r(reg(0x2843C)), f32r(reg(0x28440)), f32r(reg(0x28444)), f32r(reg(0x28448)), f32r(reg(0x2844C)), f32r(reg(0x28450)));
        let vp = if vte & 0x3F == 0x3F && xs != 0.0 && ys != 0.0 {
            vk::Viewport { x: xo - xs, y: yo - ys, width: 2.0 * xs, height: 2.0 * ys, min_depth: zo.clamp(0.0, 1.0), max_depth: (zo + zs).clamp(0.0, 1.0) }
        } else {
            self.report(format!("PA_CL_VTE_CNTL {vte:#x} (viewport transform disabled) approximated"));
            vk::Viewport { x: 0.0, y: 0.0, width: area.0 as f32, height: area.1 as f32, min_depth: 0.0, max_depth: 1.0 }
        };
        let dc = &mut self.dyncache;
        let untimed;
        unsafe {
            let rebound = dc.pipeline != pipeline;
            if rebound {
                super::note_vk(super::VK_GFX_PIPELINES);
                c.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);
                dc.pipeline = pipeline;
            }
            super::bind_descriptors(cmd, vk::PipelineBindPoint::GRAPHICS, layout, set, &bounds, &mut dc.pushed[0], rebound);
            super::push_sgpr_bases(cmd, layout, false, &bounds);
            let vpw = [vp.x.to_bits(), vp.y.to_bits(), vp.width.to_bits(), vp.height.to_bits(), vp.min_depth.to_bits(), vp.max_depth.to_bits()];
            if dc.viewport != Some(vpw) {
                c.device.cmd_set_viewport(cmd, 0, &[vp]);
                dc.viewport = Some(vpw);
            }
            let mut sc = vk::Rect2D { offset: vk::Offset2D { x: x0, y: y0 }, extent: vk::Extent2D { width: (x1 - x0) as u32, height: (y1 - y0) as u32 } };
            // Diagnosis (control `tiny <ps>`): those draws cover one pixel
            // (their vertex work stays, pixel work goes).
            if TINY_ANY.load(Ordering::Relaxed) && TINY_SCISSOR.lock().contains(&ps_addr) {
                sc.extent = vk::Extent2D { width: 1, height: 1 };
            }
            let scw = [sc.offset.x, sc.offset.y, sc.extent.width as i32, sc.extent.height as i32];
            if dc.scissor != Some(scw) {
                c.device.cmd_set_scissor(cmd, 0, &[sc]);
                dc.scissor = Some(scw);
            }
            // Depth/stencil clears requested through DB_RENDER_CONTROL.
            if clear_z || clear_s {
                let mut aspect = vk::ImageAspectFlags::empty();
                if clear_z {
                    aspect |= vk::ImageAspectFlags::DEPTH;
                }
                if clear_s {
                    aspect |= vk::ImageAspectFlags::STENCIL;
                }
                let ca = vk::ClearAttachment {
                    aspect_mask: aspect,
                    color_attachment: 0,
                    clear_value: vk::ClearValue { depth_stencil: vk::ClearDepthStencilValue { depth: f32r(reg(0x2802C)), stencil: reg(0x28028) & 0xFF } },
                };
                c.device.cmd_clear_attachments(cmd, &[ca], &[vk::ClearRect { rect: sc, base_array_layer: 0, layer_count: 1 }]);
                if colors.iter().all(Option::is_none) {
                    return Ok(true);
                }
            }
            let mode = reg(0x28814);
            let mut cull = vk::CullModeFlags::NONE;
            if mode & 1 != 0 && !rect {
                cull |= vk::CullModeFlags::FRONT;
            }
            if mode & 2 != 0 && !rect {
                cull |= vk::CullModeFlags::BACK;
            }
            if dc.set(0, cull.as_raw() as u64) {
                c.device.cmd_set_cull_mode(cmd, cull);
            }
            let front = if mode & 4 != 0 { vk::FrontFace::CLOCKWISE } else { vk::FrontFace::COUNTER_CLOCKWISE };
            if dc.set(1, front.as_raw() as u64) {
                c.device.cmd_set_front_face(cmd, front);
            }
            if dc.set(2, topology.as_raw() as u64) {
                c.device.cmd_set_primitive_topology(cmd, topology);
            }
            let restart = reg(0x28A94) & 1 != 0 && matches!(prim, 3 | 5 | 6) && a.index.is_some();
            if dc.set(3, restart as u64) {
                c.device.cmd_set_primitive_restart_enable(cmd, restart);
            }
            if dc.set(4, 0) {
                c.device.cmd_set_rasterizer_discard_enable(cmd, false);
            }
            if dc.set(5, z_test as u64) {
                c.device.cmd_set_depth_test_enable(cmd, z_test);
            }
            if dc.set(6, z_write as u64) {
                c.device.cmd_set_depth_write_enable(cmd, z_write);
            }
            // DS_DEPTH_EQUAL_BIAS=<units> (diagnosis): EQUAL tests without
            // depth writes pass a little in front of the stored depth instead
            // (reversed Z when the depth clear value is 0); `always`: they
            // always pass (do the draw's fragments exist at all?).
            let eq_bias = equal_bias().filter(|_| z_test && !z_write && (dctl >> 4) & 7 == 2);
            let reversed = f32r(reg(0x2802C)) == 0.0;
            let zop = match eq_bias {
                Some(b) if b.is_infinite() => vk::CompareOp::ALWAYS,
                Some(_) => {
                    if reversed {
                        vk::CompareOp::GREATER_OR_EQUAL
                    } else {
                        vk::CompareOp::LESS_OR_EQUAL
                    }
                }
                None => compare_op(dctl >> 4),
            };
            if dc.set(7, zop.as_raw() as u64) {
                c.device.cmd_set_depth_compare_op(cmd, zop);
            }
            let eq_bias = eq_bias.filter(|b| b.is_finite());
            if dc.set(8, 0) {
                c.device.cmd_set_depth_bounds_test_enable(cmd, false);
            }
            if dc.set(9, s_test as u64) {
                c.device.cmd_set_stencil_test_enable(cmd, s_test);
            }
            let back = dctl & (1 << 7) != 0;
            let ffunc = compare_op(dctl >> 8);
            let bfunc = if back { compare_op(dctl >> 20) } else { ffunc };
            let sop = |w: u32, f: vk::CompareOp| (stencil_op(w), stencil_op(w >> 4), stencil_op(w >> 8), f);
            let pack = |o: (vk::StencilOp, vk::StencilOp, vk::StencilOp, vk::CompareOp)| (o.0.as_raw() as u64) | (o.1.as_raw() as u64) << 16 | (o.2.as_raw() as u64) << 32 | (o.3.as_raw() as u64) << 48;
            let fo = sop(sctl, ffunc);
            if dc.set(10, pack(fo)) {
                c.device.cmd_set_stencil_op(cmd, vk::StencilFaceFlags::FRONT, fo.0, fo.1, fo.2, fo.3);
            }
            let bs = if back { sctl >> 12 } else { sctl };
            let bo = sop(bs, bfunc);
            if dc.set(11, pack(bo)) {
                c.device.cmd_set_stencil_op(cmd, vk::StencilFaceFlags::BACK, bo.0, bo.1, bo.2, bo.3);
            }
            let rb = if back { refm_bf } else { refm };
            for (slot, face, v) in [
                (12, vk::StencilFaceFlags::FRONT, refm & 0xFF),
                (13, vk::StencilFaceFlags::BACK, rb & 0xFF),
            ] {
                if dc.set(slot, v as u64) {
                    c.device.cmd_set_stencil_reference(cmd, face, v);
                }
            }
            for (slot, face, v) in [
                (14, vk::StencilFaceFlags::FRONT, (refm >> 8) & 0xFF),
                (15, vk::StencilFaceFlags::BACK, (rb >> 8) & 0xFF),
            ] {
                if dc.set(slot, v as u64) {
                    c.device.cmd_set_stencil_compare_mask(cmd, face, v);
                }
            }
            for (slot, face, v) in [
                (16, vk::StencilFaceFlags::FRONT, (refm >> 16) & 0xFF),
                (17, vk::StencilFaceFlags::BACK, (rb >> 16) & 0xFF),
            ] {
                if dc.set(slot, v as u64) {
                    c.device.cmd_set_stencil_write_mask(cmd, face, v);
                }
            }
            // PA_SU_SC_MODE_CNTL.POLY_OFFSET_FRONT/BACK_ENABLE with
            // PA_SU_POLY_OFFSET_* (scale in 1/16 units).
            let poly_offset = mode & (3 << 11) != 0;
            let bias_on = poly_offset || eq_bias.is_some();
            if dc.set(18, bias_on as u64) {
                c.device.cmd_set_depth_bias_enable(cmd, bias_on);
            }
            let bias = if let (Some(units), false) = (eq_bias, poly_offset) {
                Some((if reversed { units } else { -units }, 0.0, 0.0))
            } else if poly_offset {
                let (sr, or) = if mode & (1 << 11) != 0 { (0x28B80, 0x28B84) } else { (0x28B88, 0x28B8C) };
                Some((f32r(reg(or)), f32r(reg(0x28B7C)), f32r(reg(sr)) / 16.0))
            } else {
                None
            };
            if let Some((k, clamp, slope)) = bias {
                let bw = [k.to_bits(), clamp.to_bits(), slope.to_bits()];
                if dc.bias != Some(bw) {
                    c.device.cmd_set_depth_bias(cmd, k, clamp, slope);
                    dc.bias = Some(bw);
                }
            }
            if dc.set(20, 1) {
                c.device.cmd_set_line_width(cmd, 1.0);
            }
            let bc = [reg(0x28414), reg(0x28418), reg(0x2841C), reg(0x28420)];
            if dc.blend != Some(bc) {
                c.device.cmd_set_blend_constants(cmd, &[f32r(bc[0]), f32r(bc[1]), f32r(bc[2]), f32r(bc[3])]);
                dc.blend = Some(bc);
            }
            let base_vertex = a.base_vertex.unwrap_or(reg(0x28408) as i32);
            lap.mark(15);
            // DS_PROFILE_GPU: timestamps around the draw, keyed by its shader.
            let qp = self.slots[self.cur].queries;
            let ti = self.batch_shaders.len() as u32;
            // Full-screen passes (a triangle or quad, one instance) count as
            // large: their cost is the pixels they shade.
            let work = if gpu.is_some() {
                0
            } else if a.count <= 4 && a.instances == 1 {
                u64::MAX
            } else {
                a.count as u64 * a.instances as u64
            };
            let timed = qp != vk::QueryPool::null() && ti < super::MAX_TIMED && super::stamp_worth(Some(work), 16384);
            untimed = !timed;
            if timed {
                for k in 0..2 {
                    c.device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, qp, super::STAMPS * ti + k);
                }
            }
            // Index buffers: a copy in video memory (the input assembler
            // fetches indices again for every instance; from system memory
            // across PCIe an instanced foliage draw took 4-8 ms, 0.2-0.3 ms
            // from video memory), made by the GPU at the batch's start, else
            // by the CPU now.
            if super::profiling() {
                if let Some((addr, size)) = a.index {
                    super::super::census::note(addr, a.count as u64 * size as u64, super::super::census::INDEX);
                }
            }
            let snap = index_home.as_ref().map(|h| (h.0, h.1, 0)).or_else(|| {
                a.index.and_then(|(addr, size)| {
                    let bytes = a.count as u64 * size as u64;
                    self.pre_copy(addr, bytes).or_else(|| self.snapshot(addr, bytes, crate::gpu::tune::snap_index()))
                })
            });
            match a.index {
                Some((_, size)) if snap.is_some() => {
                    let (sbuf, soff, sbias) = snap.unwrap();
                    super::INDEX_SNAPS.fetch_add(1, Ordering::Relaxed);
                    super::INDEX_SNAP_BYTES.fetch_add(a.count as u64 * size as u64, Ordering::Relaxed);
                    let ty = if size == 4 { vk::IndexType::UINT32 } else { vk::IndexType::UINT16 };
                    c.device.cmd_bind_index_buffer(cmd, sbuf, soff + sbias, ty);
                    match &args_home {
                        Some(h) => {
                            GPU_ARG_DRAWS.fetch_add(1, Ordering::Relaxed);
                            c.device.cmd_draw_indexed_indirect(cmd, h.0, h.1, 1, 20);
                            self.args_read = true;
                        }
                        None => c.device.cmd_draw_indexed(cmd, a.count, a.instances, 0, base_vertex, a.first_instance),
                    }
                }
                Some((addr, size)) => {
                    let bytes = a.count as u64 * size as u64;
                    let span = super::span(addr, bytes);
                    let Some((buffer, start, _)) = span else {
                        return Err(format!("index buffer {addr:#x}+{bytes:#x} is not mapped GPU memory (draw skipped)"));
                    };
                    let ty = if size == 4 { vk::IndexType::UINT32 } else { vk::IndexType::UINT16 };
                    if let Some(s) = super::CP_WRITES.latest(&[(addr, bytes)]) {
                        self.submit_after = self.submit_after.max(s);
                    }
                    c.device.cmd_bind_index_buffer(cmd, buffer, addr - start, ty);
                    c.device.cmd_draw_indexed(cmd, a.count, a.instances, 0, base_vertex, a.first_instance);
                }
                None => c.device.cmd_draw(cmd, a.count, a.instances, base_vertex as u32, a.first_instance),
            }
            if timed {
                for k in 2..4 {
                    c.device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, qp, super::STAMPS * ti + k);
                }
                self.batch_shaders.push(super::DRAW_KEY | if ps_addr != 0 { ps_addr } else { vs.addr });
            } else if super::profiling() {
                // Batch GPU time by command (`note_batch_by`).
                self.batch_shaders.push(super::DRAW_KEY | if ps_addr != 0 { ps_addr } else { vs.addr });
            }
        }
        lap.mark(16);
        // Another draw like it can be recorded on the state this one left
        // (`draw_repeat`): GPU-read arguments, indices and arguments in
        // homes, no buffer writes, no timestamps around it.
        let no_writes = bounds.iter().all(|b| b.writes.is_empty());
        let repeat = match (gpu, &args_home, &index_home, a.index) {
            (Some(g), Some(ah), Some(ih), Some((addr, size))) if untimed && no_writes && self.recording && super::super::tune::draw_repeat() && !sync_mode() => {
                Some(Repeat { batch: self.batch_count, uses: 0, index: (addr, size, g.index_bytes), index_buf: ih.0, index_off: ih.1, args_buf: ah.0, cpu: false })
            }
            (None, None, Some(ih), Some((_, size))) if a.base_inst.is_some() && untimed && no_writes && self.recording && super::super::tune::draw_repeat() && !sync_mode() => {
                Some(Repeat { batch: self.batch_count, uses: 0, index: (0, size, 0), index_buf: ih.0, index_off: ih.1, args_buf: vk::Buffer::null(), cpu: true })
            }
            _ => None,
        };
        for b in bounds.drain(..) {
            if !b.writes.is_empty() {
                crate::gpu::mirror::invalidate(&b.writes);
            }
            for &(base, n) in &b.writes {
                crate::gpu::track::gpu_write(base, n);
            }
            if !b.sized_writes.is_empty() {
                crate::gpu::images::with(|im| im.note_overwrites(&b.sized_writes));
            }
            for &(w, n) in &b.writes {
                self.note_write(w, n);
            }
            self.recycle(b);
        }
        self.draw_bounds = bounds;
        // (After everything this draw recorded.)
        self.repeat = repeat.map(|r| Repeat { uses: self.cmd_uses.get(), ..r });
        lap.mark(17);
        if super::profiling() {
            super::note_phases(&super::DRAW_PHASES, [t_prep, t_pipe, t_bind, since(t0)]);
            super::note_place(self.deferred, self.batch_shaders.len().saturating_sub(1), [t_prep, t_pipe, t_bind, since(t0)]);
            if first {
                super::DRAW_STEPS_FIRST[18].fetch_add(1, Ordering::Relaxed);
            } else {
                super::DRAW_STEPS[kind][18].fetch_add(1, Ordering::Relaxed);
            }
        }
        Ok(true)
    }
}

/// Draws after a draw of the same programs on the same executor, by what
/// differs: [nothing, the user data only, buffer resources (images the
/// same), image resources]; and draws of other programs (profiling).
static DRAW_DIFFS: [std::sync::atomic::AtomicU64; 5] = [const { std::sync::atomic::AtomicU64::new(0) }; 5];

/// A draw's programs and hashes of its buffer resources, image and sampler
/// resources and user data.
fn draw_sig(vs: &shaders::Prepared, ps: Option<&shaders::Prepared>) -> [u64; 5] {
    use std::hash::{Hash, Hasher};
    let (mut hb, mut hi, mut hu) = (std::collections::hash_map::DefaultHasher::new(), std::collections::hash_map::DefaultHasher::new(), std::collections::hash_map::DefaultHasher::new());
    for p in std::iter::once(vs).chain(ps) {
        for r in p.resources.iter() {
            match r {
                shaders::Resolved::Buffer(v) => v.hash(&mut hb),
                shaders::Resolved::Global(a, n) => (a, n).hash(&mut hb),
                shaders::Resolved::Image(t) => t.hash(&mut hi),
                shaders::Resolved::ImageTable(e) => e.hash(&mut hi),
                shaders::Resolved::Sampler(sm) => sm.hash(&mut hi),
            }
        }
        let n = p.sgpr_data.len().min(rdna::translate::SGPR_DATA_USER as usize);
        p.sgpr_data[..n].hash(&mut hu);
    }
    [vs.addr, ps.map_or(0, |p| p.addr), hb.finish(), hi.finish(), hu.finish()]
}

fn note_draw_diff(last: &mut [u64; 5], sig: [u64; 5]) {
    use std::sync::atomic::Ordering::Relaxed;
    let k = if sig[0] != last[0] || sig[1] != last[1] {
        4
    } else if sig[2] == last[2] && sig[3] == last[3] {
        if sig[4] == last[4] { 0 } else { 1 }
    } else if sig[3] == last[3] {
        2
    } else {
        3
    };
    DRAW_DIFFS[k].fetch_add(1, Relaxed);
    *last = sig;
}

/// Status part: `DRAW_DIFFS`.
pub(super) fn draw_diff_line() -> String {
    use std::sync::atomic::Ordering::Relaxed;
    let v: Vec<u64> = DRAW_DIFFS.iter().map(|c| c.load(Relaxed)).collect();
    if v.iter().sum::<u64>() == 0 {
        return String::new();
    }
    format!(
        "    draws after a draw of the same programs (same executor): nothing different {}, the user data only {}, buffer resources (same images) {}, image resources {}; after other programs {}\n",
        v[0], v[1], v[2], v[3], v[4]
    )
}
