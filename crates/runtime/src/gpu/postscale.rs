//! The title's post-processing pass at the output size (`upscale::post_pass`): the tone-mapped
//! picture the title makes at the render size (bloom, depth of field and colour grading over the
//! TAA output), made at the output size instead into a target of ours while an upscaler takes the
//! picture up to it (`Exec::draw_inner` redirects the draw, its viewport and scissor scaled), so
//! the interface composite gets it pixel for pixel rather than scaling it up. Afterwards
//! (`record_after`) the Sharpening runs on it and it is taken down into the title's own target,
//! for anything else that reads that.

use super::vk::ctx;
use ash::vk;
use parking_lot::Mutex;

struct Img {
    image: vk::Image,
    _memory: vk::DeviceMemory,
    view: vk::ImageView,
    format: vk::Format,
    w: u32,
    h: u32,
}

// SAFETY: plain handles.
unsafe impl Send for Img {}

static TARGET: Mutex<Option<Img>> = Mutex::new(None);
/// Targets of an earlier size or format (frames in flight may still use them; the output size is
/// fixed for a session, so this stays short).
static RETIRED: Mutex<Vec<Img>> = Mutex::new(Vec::new());

/// Our target at `w`x`h` in `format` (the title's target's): its image, its view and whether it
/// was made now (its layout then still UNDEFINED: `init_layout`).
pub fn target(format: vk::Format, w: u32, h: u32) -> Result<(vk::Image, vk::ImageView, bool), String> {
    let mut g = TARGET.lock();
    if let Some(t) = g.as_ref() {
        if (t.format, t.w, t.h) == (format, w, h) {
            return Ok((t.image, t.view, false));
        }
    }
    let c = ctx();
    unsafe {
        let f = c.instance.get_physical_device_format_properties(c.pdev, format).optimal_tiling_features;
        if !f.contains(vk::FormatFeatureFlags::COLOR_ATTACHMENT | vk::FormatFeatureFlags::SAMPLED_IMAGE | vk::FormatFeatureFlags::BLIT_SRC) {
            return Err(format!("{format:?} cannot be rendered to, sampled and blitted here"));
        }
        let ici = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D { width: w, height: h, depth: 1 })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST);
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
        let view = match c.device.create_image_view(&vk::ImageViewCreateInfo::default().image(image).view_type(vk::ImageViewType::TYPE_2D).format(format).subresource_range(range), None) {
            Ok(v) => v,
            Err(e) => {
                c.device.destroy_image(image, None);
                c.device.free_memory(memory, None);
                return Err(e.to_string());
            }
        };
        crate::info!("upscaler: post-processing target {w}x{h} {format:?} made");
        if let Some(old) = g.replace(Img { image, _memory: memory, view, format, w, h }) {
            RETIRED.lock().push(old);
        }
        Ok((image, view, true))
    }
}

/// Records into `cmd` (outside rendering) our new target's move to GENERAL.
pub fn init_layout(cmd: vk::CommandBuffer, image: vk::Image) {
    super::vk::defer::external(cmd, Box::new(move |cb| {
        let b = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE | vk::AccessFlags::COLOR_ATTACHMENT_READ)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 });
        unsafe { ctx().device.cmd_pipeline_barrier(cb, vk::PipelineStageFlags::TOP_OF_PIPE, vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT, vk::DependencyFlags::empty(), &[], &[], &[b]) };
    }));
}

/// A post-processing pass recorded: the title's own target and, when it ran at the output size,
/// ours.
#[derive(Clone, Copy)]
pub struct Done {
    pub title: (vk::Image, vk::Format, u32, u32),
    pub ours: Option<(vk::Image, u32, u32)>,
}

/// After the pass (outside rendering): the Sharpening on the picture it made, then ours taken
/// down into the title's target (linear where the format allows).
pub fn record_after(cmd: vk::CommandBuffer, d: Done) {
    let (sharp_img, sw, sh) = match d.ours {
        Some((i, w, h)) => (i, w, h),
        None => (d.title.0, d.title.2, d.title.3),
    };
    if let Some(job) = super::vk::sharpen::prepare(sharp_img, d.title.1, sw, sh) {
        super::vk::sharpen::record(cmd, job);
    }
    let Some((ours, ow, oh)) = d.ours else { return };
    let (title, format, tw, th) = d.title;
    let c = ctx();
    let linear = unsafe { c.instance.get_physical_device_format_properties(c.pdev, format) }.optimal_tiling_features.contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR | vk::FormatFeatureFlags::BLIT_DST);
    let filter = if linear { vk::Filter::LINEAR } else { vk::Filter::NEAREST };
    super::vk::defer::external(cmd, Box::new(move |cb| {
        let layers = vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 };
        let blit = vk::ImageBlit {
            src_subresource: layers,
            src_offsets: [vk::Offset3D { x: 0, y: 0, z: 0 }, vk::Offset3D { x: ow as i32, y: oh as i32, z: 1 }],
            dst_subresource: layers,
            dst_offsets: [vk::Offset3D { x: 0, y: 0, z: 0 }, vk::Offset3D { x: tw as i32, y: th as i32, z: 1 }],
        };
        let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::MEMORY_WRITE).dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
        let d = &ctx().device;
        unsafe {
            d.cmd_pipeline_barrier(cb, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[mb], &[], &[]);
            d.cmd_blit_image(cb, ours, vk::ImageLayout::GENERAL, title, vk::ImageLayout::GENERAL, &[blit], filter);
            d.cmd_pipeline_barrier(cb, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::ALL_COMMANDS, vk::DependencyFlags::empty(), &[mb], &[], &[]);
        }
    }));
}
