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
}

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
#[derive(Hash)]
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

fn set_layout_bindings(stages: &[(&Prepared, vk::ShaderStageFlags)]) -> Vec<vk::DescriptorSetLayoutBinding<'static>> {
    let mut v = Vec::new();
    for (p, flags) in stages {
        for b in &p.tr.out.bindings {
            v.push(vk::DescriptorSetLayoutBinding::default().binding(b.binding).descriptor_type(descriptor_type(b.kind)).descriptor_count(super::binding_count(b.kind)).stage_flags(*flags));
        }
    }
    v
}

/// Creates a graphics pipeline for dynamic rendering (`push_mode`: a push
/// descriptor layout if it fits, `push_layout`), through the pipeline cache
/// file of `hkey`.
fn graphics_pipeline(key: &GfxKey, hkey: u64, vs: &Prepared, ps: Option<&Prepared>, push_mode: bool) -> Result<Pipe, String> {
    let c = ctx();
    unsafe {
        let mut stages_p = vec![(vs, vk::ShaderStageFlags::VERTEX)];
        if let Some(p) = ps {
            stages_p.push((p, vk::ShaderStageFlags::FRAGMENT));
        }
        let bindings = set_layout_bindings(&stages_p);
        let (push, flags) = super::push_layout(&bindings, push_mode);
        let set_layout = c.device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().flags(flags).bindings(&bindings), None).map_err(|e| format!("set layout: {e}"))?;
        let layouts = [set_layout];
        let layout = c.device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts), None).map_err(|e| format!("pipeline layout: {e}"))?;
        let module = |code: &[u32]| c.device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(code), None).map_err(|e| format!("shader module: {e}"));
        let mut modules = vec![(vk::ShaderStageFlags::VERTEX, module(&vs.tr.out.spirv)?)];
        if let Some(mask) = key.rect_gs {
            modules.push((vk::ShaderStageFlags::GEOMETRY, module(&rdna::translate::gs::rect_list_gs(mask))?));
        }
        if let Some(p) = ps {
            modules.push((vk::ShaderStageFlags::FRAGMENT, module(&p.tr.out.spirv)?));
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
        crate::gpu::pstats::report(pipeline, &format!("vs {:#x} ps {:#x}", vs.addr, ps.map_or(0, |p| p.addr)));
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
    pub fn draw(&mut self, cp: &Cp, args: DrawArgs) -> String {
        if DEVICE_LOST.load(Ordering::Relaxed) {
            DRAW_SKIPPED.fetch_add(1, Ordering::Relaxed);
            return "skipped (device lost)".into();
        }
        let outcome = match self.draw_inner(cp, &args) {
            Ok(true) => {
                DRAWN.fetch_add(1, Ordering::Relaxed);
                "drawn".to_string()
            }
            Ok(false) => {
                DRAW_NOOP.fetch_add(1, Ordering::Relaxed);
                "no effect".to_string()
            }
            Err(e) if e == super::STOPPED => "stopped (parallel run)".to_string(),
            Err(e) => {
                DRAW_SKIPPED.fetch_add(1, Ordering::Relaxed);
                let s = format!("skipped: {e}");
                self.report(e);
                s
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

    /// Ends the active rendering instance; its attachment writes become
    /// visible to later commands.
    pub(super) fn end_rendering(&mut self) {
        if self.rendering.take().is_some() {
            unsafe { ctx().device.cmd_end_rendering(self.cmd()) };
            global_barrier(self.cmd());
            self.barrier_wanted = false;
            self.dispatched = false;
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
        unsafe { ctx().device.cmd_begin_rendering(self.cmd(), &ri) };
        self.rendering = Some(Rendering { colors, depth, area });
    }

    /// Records the draw; Ok(false) when it has no effect here.
    fn draw_inner(&mut self, cp: &Cp, a: &DrawArgs) -> Result<bool, String> {
        if a.count == 0 || a.instances == 0 {
            return Ok(false);
        }
        // Phase times (DS_PROFILE_GPU only: a clock read costs ~20 ns).
        let t0 = super::profiling().then(std::time::Instant::now);
        let since = |t0: Option<std::time::Instant>| t0.map_or(std::time::Duration::ZERO, |t| t.elapsed());
        let mut lap = super::Laps::new();
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
        // hold, else here.
        let [pre_vs, pre_ps] = cp.take_ahead();
        let mut vs = match pre_vs.and_then(|s| crate::gpu::ahead::take(&s, cp, true)) {
            Some(p) => p,
            None => shaders::prepare_ngg(cp)?,
        };
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
        self.sync_readers = [vs.addr, ps.as_ref().map_or(0, |p| p.addr)];
        // Each stage's reads in place (no combined copy per draw), unless the
        // stages are used as prepared (`speculate`).
        let patches = match &ps {
            Some(p) => self.speculate(&[&vs, p]),
            None => self.speculate(&[&vs]),
        };
        let waited = patches.is_none() && (self.sync_for(&vs.reads, SYNC_DRAW_READS) | ps.as_ref().is_some_and(|p| self.sync_for(&p.reads, SYNC_DRAW_READS)));
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
            vs = shaders::prepare_ngg(cp)?;
            ps = if ps_addr != 0 { Some(shaders::prepare_ps(cp)?) } else { None };
        }
        lap.mark(2);
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
        let index_home = match a.index {
            Some((addr, size)) => {
                let bytes = a.count as u64 * size as u64;
                let h = self.home_bind(addr, bytes, false, super::HomeUse::Index);
                if h.is_none() {
                    self.sync_for(&[(addr, bytes)], SYNC_DRAW_INDICES);
                }
                if self.stopped {
                    return Err(super::STOPPED.into());
                }
                h
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
        if colors.iter().all(Option::is_none) && depth.is_none() {
            // Memory side effects only: a render area is still needed.
            return Err("draw with only memory writes (no attachments) not supported yet (skipped)".into());
        }
        let mut area = (u32::MAX, u32::MAX);
        for t in colors.iter().flatten().chain(depth.iter()) {
            area = (area.0.min(t.width), area.1.min(t.height));
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
        let hkey = {
            let mut h = std::collections::hash_map::DefaultHasher::new();
            key.hash(&mut h);
            super::push_key(h.finish(), push_mode)
        };
        let cached = gfx_pipe(hkey);
        let Pipe { pipeline, layout, set_layout, push } = match cached {
            Some(x) => x,
            None => {
                let t0 = std::time::Instant::now();
                let p = graphics_pipeline(&key, hkey, &vs, ps.as_ref(), push_mode)?;
                super::PIPES_CREATED[1].fetch_add(1, Ordering::Relaxed);
                super::PIPES_NS[1].fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                if p.push {
                    super::PIPES_PUSH[1].fetch_add(1, Ordering::Relaxed);
                }
                let ms = t0.elapsed().as_millis();
                if ms >= 200 {
                    crate::info!("gpu: graphics pipeline for vs {:#x} ps {ps_addr:#x} took {ms} ms", vs.addr);
                }
                keep_gfx_pipe(hkey, p)
            }
        };
        let t_pipe = since(t0);
        lap.mark(7);
        // Depth arrays sampled with more layers than rendered so far.
        match &ps {
            Some(p) => self.grow_sampled_depth(&[&vs, p]),
            None => self.grow_sampled_depth(&[&vs]),
        }
        // Fast clears (metadata writes), reloads and block copies of the
        // attachments and textures.
        let attached: Vec<images::ImageKey> = colors.iter().flatten().chain(depth.iter()).map(|t| t.key).collect();
        let (used, pending) = match &ps {
            Some(p) => super::image_hazards(&[&vs, p], &attached),
            None => super::image_hazards(&[&vs], &attached),
        };
        if self.stopped {
            return Err(super::STOPPED.into());
        }
        if pending {
            self.resolve_fast_clears(&used);
            self.resolve_overwrites(&used);
            self.sync_block_textures(&used);
            if self.stopped {
                return Err(super::STOPPED.into());
            }
        }
        lap.mark(8);
        // Resources.
        let mut stages: Vec<&mut Prepared> = vec![&mut vs];
        if let Some(p) = ps.as_mut() {
            stages.push(p);
        }
        self.reserve(&stages.iter().map(|p| &**p).collect::<Vec<_>>(), push);
        if self.stopped {
            return Err(super::STOPPED.into());
        }
        if let Some(h) = &index_home {
            // Kept until the batch that draws completes.
            self.slots[self.cur].homes.push(h.3.clone());
        }
        // Indices in guest memory that a resident shadow holds newer data for.
        if let Some((addr, size)) = a.index {
            let t = profiling().then(std::time::Instant::now);
            self.flush_resident(Some(&[(addr, a.count as u64 * size as u64)]))?;
            if let Some(t) = t {
                super::BEGIN_STATS[2].fetch_add(1, Ordering::Relaxed);
                super::BEGIN_STATS[3].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
            }
        }
        lap.mark(9);
        // The vector is kept for the next draw (empty: it drains at the end).
        let mut bounds = std::mem::take(&mut self.draw_bounds);
        bounds.clear();
        for (k, p) in stages.iter_mut().enumerate() {
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
            super::note_placements(super::DRAW_KEY | if ps_addr != 0 { ps_addr } else { stages[0].addr }, [0; 3], false, &all);
        }
        let set = if push { None } else { Some(self.alloc_set(set_layout)?) };
        lap.mark(12);
        if let Some(set) = set {
            self.write_set(set, &bounds);
        }
        lap.mark(13);
        let t_bind = since(t0);
        // Rendering instance: reuse it while the attachments stay the same
        // and the draw does not sample one of them.
        let feedback = self.rendering.as_ref().is_some_and(|r| r.colors.iter().flatten().chain(r.depth.iter()).any(|t| bounds.iter().any(|b| b.images.contains(&t.image))));
        let same = self.rendering.as_ref().is_some_and(|r| r.colors == colors && r.depth == depth);
        if !same || feedback {
            self.end_rendering();
            self.barrier_if_wanted(true);
            self.begin_rendering(colors, depth, area);
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
        unsafe {
            if dc.pipeline != pipeline {
                c.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);
                dc.pipeline = pipeline;
            }
            super::bind_descriptors(cmd, vk::PipelineBindPoint::GRAPHICS, layout, set, &bounds);
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
            let timed = qp != vk::QueryPool::null() && ti < super::MAX_TIMED && super::command_stamps();
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
                    c.device.cmd_draw_indexed(cmd, a.count, a.instances, 0, base_vertex, a.first_instance);
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
        lap.mark(17);
        if super::profiling() {
            super::note_phases(&super::DRAW_PHASES, [t_prep, t_pipe, t_bind, since(t0)]);
        }
        Ok(true)
    }
}
