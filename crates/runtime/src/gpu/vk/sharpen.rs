//! The PC settings' Sharpening: AMD's RCAS (FSR 1's robust contrast-adaptive sharpening, without
//! FSR 1's upscaling; `rdna::kernels::rcas`) over the finished picture, after the upscaler (or
//! the game's TAA) and the game's tone mapping, before the interface is composited over it: on
//! the target of the title's post-processing pass (bloom, depth of field, colour grading; at the
//! output size while an upscaler takes the picture up to it: `gpu::postscale`), which the
//! interface composite (ps 0x100888d100, 0x100888d900 with HDR output) then reads with the
//! interface layer. The target is blitted into an image of ours (RGBA16F), sharpened into a
//! second one and blitted back. The menus and the HUD stay as they are.
//!
//! Until 2026-11-26 the presenter sharpened the whole shown frame with a kernel of its own
//! (interface included), and the game's own sharpening of its TAA's history ran as the title
//! set it; that one is now always off (`pcsettings::apply`).
//!
//! HDR: the HDR10 display target holds ST 2084 (PQ) values, which RCAS sharpens as they are
//! (perceptual, in [0, 1], like gamma-encoded SDR); an sRGB target's values are encoded to sRGB
//! for it, and a float (scRGB) one's to PQ, then decoded (the kernel's `encoding`). The images
//! between are RGBA16F (finer than the 10-bit target: no banding), the result stays within its
//! neighbourhood's range (no clipping, no halos).

use super::ctx;
use ash::vk;
use parking_lot::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};

/// The strength (f32 bits; 0: off; 1: RCAS's full strength).
pub static AMOUNT: AtomicU32 = AtomicU32::new(0);

pub fn amount() -> f32 {
    let v = f32::from_bits(AMOUNT.load(Ordering::Relaxed));
    if v.is_finite() && v > 0.0 { v.min(1.0) } else { 0.0 }
}

/// Passes recorded (the status line).
pub static PASSES: AtomicU64 = AtomicU64::new(0);

const FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;

struct Img {
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
}

struct Pass {
    /// (Kept for the pass's life: the push descriptors' set layout.)
    _dsl: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    /// The two images (input, result) at their size.
    imgs: Option<(Img, Img, u32, u32)>,
    /// Images of an earlier size (frames in flight may still use them; the display target's size
    /// is fixed for a session, so this stays short).
    retired: Vec<Img>,
    /// The formats checked for blits: (format, usable).
    formats: Vec<(vk::Format, bool)>,
}

unsafe impl Send for Pass {}

static PASS: Mutex<Option<Pass>> = Mutex::new(None);
/// The pass could not be made: no sharpening this session.
static FAILED: AtomicBool = AtomicBool::new(false);

unsafe fn new_pass() -> Result<Pass, String> {
    let c = ctx();
    let d = &c.device;
    unsafe {
        let bindings = [0, 1].map(|b| vk::DescriptorSetLayoutBinding::default().binding(b).descriptor_type(vk::DescriptorType::STORAGE_IMAGE).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE));
        let dsl = d
            .create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().flags(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR).bindings(&bindings), None)
            .map_err(|e| format!("rcas set layout: {e}"))?;
        let pcr = [vk::PushConstantRange { stage_flags: vk::ShaderStageFlags::COMPUTE, offset: 0, size: 16 }];
        let dsls = [dsl];
        let layout = d.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&dsls).push_constant_ranges(&pcr), None).map_err(|e| format!("rcas layout: {e}"))?;
        let code = rdna::kernels::rcas();
        let module = d.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None).map_err(|e| format!("rcas module: {e}"))?;
        let stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(c"main");
        let pipeline = d.create_compute_pipelines(vk::PipelineCache::null(), &[vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout)], None).map_err(|(_, e)| format!("rcas pipeline: {e}"))?[0];
        d.destroy_shader_module(module, None);
        Ok(Pass { _dsl: dsl, layout, pipeline, imgs: None, retired: Vec::new(), formats: Vec::new() })
    }
}

unsafe fn image(w: u32, h: u32) -> Result<Img, String> {
    let c = ctx();
    unsafe {
        let ici = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(FORMAT)
            .extent(vk::Extent3D { width: w, height: h, depth: 1 })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::STORAGE);
        let image = c.device.create_image(&ici, None).map_err(|e| e.to_string())?;
        let memory = match c.alloc(c.device.get_image_memory_requirements(image), vk::MemoryPropertyFlags::DEVICE_LOCAL) {
            Ok(m) => m,
            Err(e) => {
                c.device.destroy_image(image, None);
                return Err(e);
            }
        };
        if let Err(e) = c.device.bind_image_memory(image, memory, 0) {
            c.device.destroy_image(image, None);
            c.device.free_memory(memory, None);
            return Err(e.to_string());
        }
        let range = vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 };
        let view = match c.device.create_image_view(&vk::ImageViewCreateInfo::default().image(image).view_type(vk::ImageViewType::TYPE_2D).format(FORMAT).subresource_range(range), None) {
            Ok(v) => v,
            Err(e) => {
                c.device.destroy_image(image, None);
                c.device.free_memory(memory, None);
                return Err(e.to_string());
            }
        };
        Ok(Img { image, memory, view })
    }
}

/// How the kernel reads the target's values (see `rcas.comp`): 0 as they are (UNORM: gamma or
/// PQ encoded), 1 linear from an sRGB format, 2 linear float (scRGB).
fn encoding(f: vk::Format) -> u32 {
    let name = format!("{f:?}");
    if name.ends_with("_SRGB") || name.contains("_SRGB_") {
        1
    } else if name.contains("FLOAT") {
        2
    } else {
        0
    }
}

/// One pass to record: the display target and our two images.
#[derive(Clone, Copy)]
pub struct Job {
    target: vk::Image,
    w: u32,
    h: u32,
    a: (vk::Image, vk::ImageView),
    b: (vk::Image, vk::ImageView),
    fresh: bool,
    strength: f32,
    encoding: u32,
}

/// The pass over `target` (the composite's colour attachment, GENERAL, `w`x`h`, `format`): its
/// images made when needed. None when it cannot run (logged once).
pub fn prepare(target: vk::Image, format: vk::Format, w: u32, h: u32) -> Option<Job> {
    let strength = amount();
    if strength <= 0.0 || FAILED.load(Ordering::Relaxed) || w == 0 || h == 0 {
        return None;
    }
    let mut g = PASS.lock();
    if g.is_none() {
        match unsafe { new_pass() } {
            Ok(p) => *g = Some(p),
            Err(e) => {
                crate::warn!("sharpening: no RCAS pass ({e}): no sharpening");
                FAILED.store(true, Ordering::Relaxed);
                return None;
            }
        }
    }
    let p = g.as_mut().unwrap();
    let usable = match p.formats.iter().find(|(f, _)| *f == format) {
        Some(&(_, ok)) => ok,
        None => {
            let c = ctx();
            let f = unsafe { c.instance.get_physical_device_format_properties(c.pdev, format) }.optimal_tiling_features;
            let ok = f.contains(vk::FormatFeatureFlags::BLIT_SRC | vk::FormatFeatureFlags::BLIT_DST);
            p.formats.push((format, ok));
            if ok {
                crate::info!("sharpening: RCAS on the {w}x{h} {format:?} display target before the interface (encoding {})", encoding(format));
            } else {
                crate::warn!("sharpening: the display target's {format:?} cannot be blitted here: no sharpening on it");
            }
            ok
        }
    };
    if !usable {
        return None;
    }
    let mut fresh = false;
    if p.imgs.as_ref().is_none_or(|(_, _, iw, ih)| (*iw, *ih) != (w, h)) {
        let a = match unsafe { image(w, h) } {
            Ok(a) => a,
            Err(e) => {
                crate::warn!("sharpening: no {w}x{h} image ({e}): no sharpening");
                FAILED.store(true, Ordering::Relaxed);
                return None;
            }
        };
        let b = match unsafe { image(w, h) } {
            Ok(b) => b,
            Err(e) => {
                p.retired.push(a);
                crate::warn!("sharpening: no {w}x{h} image ({e}): no sharpening");
                FAILED.store(true, Ordering::Relaxed);
                return None;
            }
        };
        if let Some((oa, ob, _, _)) = p.imgs.replace((a, b, w, h)) {
            p.retired.push(oa);
            p.retired.push(ob);
        }
        fresh = true;
    }
    let (a, b, _, _) = p.imgs.as_ref().unwrap();
    Some(Job { target, w, h, a: (a.image, a.view), b: (b.image, b.view), fresh, strength, encoding: encoding(format) })
}

/// Records the pass into `cmd` (outside rendering): the target blitted into our input image,
/// sharpened into the result, blitted back, with barriers before and after.
pub fn record(cmd: vk::CommandBuffer, j: Job) {
    if PASSES.fetch_add(1, Ordering::Relaxed) == 0 {
        crate::info!("sharpening: first RCAS pass (strength {:.2})", j.strength);
    }
    super::defer::external(cmd, Box::new(move |cb| unsafe { record_now(cb, j) }));
}

unsafe fn record_now(cb: vk::CommandBuffer, j: Job) {
    let c = ctx();
    let g = PASS.lock();
    let Some(p) = g.as_ref() else { return };
    let range = vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 };
    let layers = vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 };
    let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::MEMORY_WRITE).dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
    let to_general = |img: vk::Image| {
        vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::SHADER_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(img)
            .subresource_range(range)
    };
    // (Their contents are written whole each time: the earlier ones are not kept.)
    let fresh = [to_general(j.a.0), to_general(j.b.0)];
    let whole = vk::ImageBlit {
        src_subresource: layers,
        src_offsets: [vk::Offset3D { x: 0, y: 0, z: 0 }, vk::Offset3D { x: j.w as i32, y: j.h as i32, z: 1 }],
        dst_subresource: layers,
        dst_offsets: [vk::Offset3D { x: 0, y: 0, z: 0 }, vk::Offset3D { x: j.w as i32, y: j.h as i32, z: 1 }],
    };
    let d = &c.device;
    unsafe {
        // The picture's draws (and the last pass's reads of our images) done.
        d.cmd_pipeline_barrier(cb, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &[mb], &[], if j.fresh { &fresh } else { &[] });
        // (Same size: a copy with the format's conversion.)
        d.cmd_blit_image(cb, j.target, vk::ImageLayout::GENERAL, j.a.0, vk::ImageLayout::GENERAL, &[whole], vk::Filter::NEAREST);
        d.cmd_pipeline_barrier(cb, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &[mb], &[], &[]);
        d.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, p.pipeline);
        let infos = [
            [vk::DescriptorImageInfo { sampler: vk::Sampler::null(), image_view: j.a.1, image_layout: vk::ImageLayout::GENERAL }],
            [vk::DescriptorImageInfo { sampler: vk::Sampler::null(), image_view: j.b.1, image_layout: vk::ImageLayout::GENERAL }],
        ];
        let writes = [
            vk::WriteDescriptorSet::default().dst_binding(0).descriptor_type(vk::DescriptorType::STORAGE_IMAGE).image_info(&infos[0]),
            vk::WriteDescriptorSet::default().dst_binding(1).descriptor_type(vk::DescriptorType::STORAGE_IMAGE).image_info(&infos[1]),
        ];
        c.push_desc.cmd_push_descriptor_set(cb, vk::PipelineBindPoint::COMPUTE, p.layout, 0, &writes);
        let mut pc = [0u8; 16];
        pc[0..4].copy_from_slice(&j.w.to_le_bytes());
        pc[4..8].copy_from_slice(&j.h.to_le_bytes());
        pc[8..12].copy_from_slice(&j.strength.to_le_bytes());
        pc[12..16].copy_from_slice(&j.encoding.to_le_bytes());
        d.cmd_push_constants(cb, p.layout, vk::ShaderStageFlags::COMPUTE, 0, &pc);
        let n = rdna::kernels::RCAS_LOCAL;
        d.cmd_dispatch(cb, j.w.div_ceil(n), j.h.div_ceil(n), 1);
        d.cmd_pipeline_barrier(cb, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[mb], &[], &[]);
        d.cmd_blit_image(cb, j.b.0, vk::ImageLayout::GENERAL, j.target, vk::ImageLayout::GENERAL, &[whole], vk::Filter::NEAREST);
        // The composite draws over it next.
        d.cmd_pipeline_barrier(cb, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::ALL_COMMANDS, vk::DependencyFlags::empty(), &[mb], &[], &[]);
    }
}

/// The status line (none while no pass ran).
pub fn status_line() -> String {
    let n = PASSES.load(Ordering::Relaxed);
    if n == 0 {
        return String::new();
    }
    format!("    sharpening (RCAS before the interface): {n} passes, strength {:.2}\n", amount())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encodings() {
        assert_eq!(encoding(vk::Format::A2B10G10R10_UNORM_PACK32), 0);
        assert_eq!(encoding(vk::Format::B8G8R8A8_UNORM), 0);
        assert_eq!(encoding(vk::Format::R8G8B8A8_SRGB), 1);
        assert_eq!(encoding(vk::Format::B8G8R8A8_SRGB), 1);
        assert_eq!(encoding(vk::Format::R16G16B16A16_SFLOAT), 2);
        assert_eq!(encoding(vk::Format::B10G11R11_UFLOAT_PACK32), 2);
    }
}
