//! The upscalers' output beside the title's TAA (`gpu::upscale`, `Exec::taa_upscale`). The
//! title's TAA runs as before: its output's alpha packs per-pixel data (from the depth and
//! stencil) that later passes read, which no upscaler writes. The upscaler writes an image of its
//! own at the output's size, and a kernel then copies its colour into the TAA's output, keeping
//! that alpha and applying the exposure the TAA applies to its input (its output is pre-exposed:
//! the first float of the TAA's own constant buffer; `rdna::kernels::rgb_merge`).

use super::vk::ctx;
use ash::vk;
use parking_lot::Mutex;

const FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;

struct Img {
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
    w: u32,
    h: u32,
}

struct Merge {
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    target: Option<Img>,
    /// Images of earlier sizes (kept: frames in flight may still use them; sizes change rarely).
    retired: Vec<Img>,
}

unsafe impl Send for Merge {}

static MERGE: Mutex<Option<Merge>> = Mutex::new(None);

unsafe fn new_merge() -> Result<Merge, String> {
    let c = ctx();
    let d = &c.device;
    unsafe {
        let bindings = [0, 1, 2].map(|b| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(b)
                .descriptor_type(if b == 2 { vk::DescriptorType::STORAGE_BUFFER } else { vk::DescriptorType::STORAGE_IMAGE })
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        });
        let dsl = d
            .create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().flags(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR).bindings(&bindings), None)
            .map_err(|e| format!("merge set layout: {e}"))?;
        let pcr = [vk::PushConstantRange { stage_flags: vk::ShaderStageFlags::COMPUTE, offset: 0, size: 8 }];
        let dsls = [dsl];
        let layout = d.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&dsls).push_constant_ranges(&pcr), None).map_err(|e| format!("merge layout: {e}"))?;
        let code = rdna::kernels::rgb_merge();
        let module = d.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None).map_err(|e| format!("merge module: {e}"))?;
        let stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(c"main");
        let pipeline = d.create_compute_pipelines(vk::PipelineCache::null(), &[vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout)], None).map_err(|(_, e)| format!("merge pipeline: {e}"))?[0];
        d.destroy_shader_module(module, None);
        Ok(Merge { layout, pipeline, target: None, retired: Vec::new() })
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
            .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED);
        let image = c.device.create_image(&ici, None).map_err(|e| e.to_string())?;
        let memory = match c.alloc(c.device.get_image_memory_requirements(image), vk::MemoryPropertyFlags::DEVICE_LOCAL) {
            Ok(m) => m,
            Err(e) => {
                c.device.destroy_image(image, None);
                return Err(e);
            }
        };
        c.device.bind_image_memory(image, memory, 0).map_err(|e| e.to_string())?;
        let range = vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 };
        let view = c.device.create_image_view(&vk::ImageViewCreateInfo::default().image(image).view_type(vk::ImageViewType::TYPE_2D).format(FORMAT).subresource_range(range), None).map_err(|e| e.to_string())?;
        Ok(Img { image, memory, view, w, h })
    }
}

/// The upscaler's output image at `w`x`h` (made or remade; GENERAL layout once `record` ran).
pub fn target(w: u32, h: u32) -> Result<super::upscale::Tex, String> {
    let mut g = MERGE.lock();
    if g.is_none() {
        *g = Some(unsafe { new_merge() }?);
    }
    let m = g.as_mut().unwrap();
    if m.target.as_ref().is_none_or(|t| (t.w, t.h) != (w, h)) {
        let img = unsafe { image(w, h) }?;
        if let Some(old) = m.target.replace(img) {
            m.retired.push(old);
        }
        crate::info!("upscaler: output image {w}x{h} made");
    }
    let t = m.target.as_ref().unwrap();
    Ok(super::upscale::Tex {
        image: t.image,
        view: t.view,
        range: vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 },
        format: FORMAT,
        size: (w, h),
        usage: vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
        layout: vk::ImageLayout::GENERAL,
    })
}

/// The output image's first use in a frame: GENERAL layout from UNDEFINED (its contents are the
/// upscaler's each frame).
pub fn begin(cb: vk::CommandBuffer, image: vk::Image) {
    let c = ctx();
    let b = vk::ImageMemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::empty())
        .dst_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::SHADER_READ)
        .old_layout(vk::ImageLayout::UNDEFINED)
        .new_layout(vk::ImageLayout::GENERAL)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 });
    // (Also orders the title's TAA dispatch, recorded just before, ahead of what follows.)
    let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::MEMORY_WRITE).dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
    unsafe { c.device.cmd_pipeline_barrier(cb, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::ALL_COMMANDS, vk::DependencyFlags::empty(), &[mb], &[], &[b]) };
}

/// Records into `cb` the merge of the upscaler's output into the TAA's (`out_view`, a storage
/// view of it, `w`x`h`; `exposure` the TAA's constant buffer as it binds it), with barriers before
/// and after.
pub fn record(cb: vk::CommandBuffer, out_view: vk::ImageView, w: u32, h: u32, exposure: vk::DescriptorBufferInfo) {
    let c = ctx();
    // (The upscaler's evaluation, recorded just before, failed: the TAA's output stays.)
    if !super::upscale::last_eval_ok() {
        return;
    }
    let g = MERGE.lock();
    let Some(m) = g.as_ref() else { return };
    let Some(t) = m.target.as_ref() else { return };
    let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::MEMORY_WRITE).dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
    unsafe {
        c.device.cmd_pipeline_barrier(cb, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &[mb], &[], &[]);
        c.device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, m.pipeline);
        let infos = [
            [vk::DescriptorImageInfo { sampler: vk::Sampler::null(), image_view: t.view, image_layout: vk::ImageLayout::GENERAL }],
            [vk::DescriptorImageInfo { sampler: vk::Sampler::null(), image_view: out_view, image_layout: vk::ImageLayout::GENERAL }],
        ];
        let ebuf = [exposure];
        let writes = [
            vk::WriteDescriptorSet::default().dst_binding(0).descriptor_type(vk::DescriptorType::STORAGE_IMAGE).image_info(&infos[0]),
            vk::WriteDescriptorSet::default().dst_binding(1).descriptor_type(vk::DescriptorType::STORAGE_IMAGE).image_info(&infos[1]),
            vk::WriteDescriptorSet::default().dst_binding(2).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&ebuf),
        ];
        c.push_desc.cmd_push_descriptor_set(cb, vk::PipelineBindPoint::COMPUTE, m.layout, 0, &writes);
        let pc = [w, h];
        c.device.cmd_push_constants(cb, m.layout, vk::ShaderStageFlags::COMPUTE, 0, std::slice::from_raw_parts(pc.as_ptr().cast::<u8>(), 8));
        let local = rdna::kernels::RGB_MERGE_LOCAL;
        c.device.cmd_dispatch(cb, w.div_ceil(local), h.div_ceil(local), 1);
        c.device.cmd_pipeline_barrier(cb, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::ALL_COMMANDS, vk::DependencyFlags::empty(), &[mb], &[], &[]);
    }
}
