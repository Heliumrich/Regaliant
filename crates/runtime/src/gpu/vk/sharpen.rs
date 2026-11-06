//! The PC settings' sharpening of the presented image (the SETTINGS screen's PC tab,
//! `pcsettings`): the frame is blitted at the size it is shown into an image, sharpened by a
//! compute kernel (`rdna::kernels::sharpen`: each pixel moved away from its neighbours' mean,
//! within its 3x3 neighbourhood's range) into a second one, and that is blitted to the window.

use super::ctx;
use ash::vk;
use std::sync::atomic::{AtomicU32, Ordering};

/// The amount (f32 bits; 0: off).
pub static AMOUNT: AtomicU32 = AtomicU32::new(0);

pub fn amount() -> f32 {
    let v = f32::from_bits(AMOUNT.load(Ordering::Relaxed));
    if v.is_finite() && v > 0.0 { v.min(4.0) } else { 0.0 }
}

const FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;

struct Img {
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
}

pub struct Sharpen {
    dsl: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
    imgs: Option<(Img, Img, u32, u32)>,
}

unsafe impl Send for Sharpen {}

impl Sharpen {
    pub unsafe fn new() -> Result<Sharpen, String> {
        let c = ctx();
        let d = &c.device;
        unsafe {
            let bindings = [0, 1].map(|b| vk::DescriptorSetLayoutBinding::default().binding(b).descriptor_type(vk::DescriptorType::STORAGE_IMAGE).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE));
            let dsl = d.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None).map_err(|e| format!("sharpen set layout: {e}"))?;
            let pcr = [vk::PushConstantRange { stage_flags: vk::ShaderStageFlags::COMPUTE, offset: 0, size: 12 }];
            let dsls = [dsl];
            let layout = d.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&dsls).push_constant_ranges(&pcr), None).map_err(|e| format!("sharpen layout: {e}"))?;
            let code = rdna::kernels::sharpen();
            let module = d.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None).map_err(|e| format!("sharpen module: {e}"))?;
            let stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(c"main");
            let pipeline = d.create_compute_pipelines(vk::PipelineCache::null(), &[vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout)], None).map_err(|(_, e)| format!("sharpen pipeline: {e}"))?[0];
            d.destroy_shader_module(module, None);
            let sizes = [vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_IMAGE, descriptor_count: 2 }];
            let pool = d.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&sizes), None).map_err(|e| format!("sharpen pool: {e}"))?;
            let set = d.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&dsls)).map_err(|e| format!("sharpen set: {e}"))?[0];
            Ok(Sharpen { dsl, layout, pipeline, pool, set, imgs: None })
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
            c.device.bind_image_memory(image, memory, 0).map_err(|e| e.to_string())?;
            let range = vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 };
            let view = c
                .device
                .create_image_view(&vk::ImageViewCreateInfo::default().image(image).view_type(vk::ImageViewType::TYPE_2D).format(FORMAT).subresource_range(range), None)
                .map_err(|e| e.to_string())?;
            Ok(Img { image, memory, view })
        }
    }

    unsafe fn drop_imgs(&mut self) {
        let c = ctx();
        if let Some((a, b, _, _)) = self.imgs.take() {
            for i in [a, b] {
                unsafe {
                    c.device.destroy_image_view(i.view, None);
                    c.device.destroy_image(i.image, None);
                    c.device.free_memory(i.memory, None);
                }
            }
        }
    }

    /// The two images at `w`x`h` (`idle`: waits for frames still using the old ones); returns
    /// (input image, output image, made now), both in GENERAL layout once `begin` has run.
    pub unsafe fn ensure(&mut self, w: u32, h: u32, idle: impl FnOnce()) -> Result<(vk::Image, vk::Image, bool), String> {
        if let Some((a, b, iw, ih)) = &self.imgs {
            if (*iw, *ih) == (w, h) {
                return Ok((a.image, b.image, false));
            }
        }
        idle();
        unsafe {
            self.drop_imgs();
            let a = Self::image(w, h)?;
            let b = match Self::image(w, h) {
                Ok(b) => b,
                Err(e) => {
                    let c = ctx();
                    c.device.destroy_image_view(a.view, None);
                    c.device.destroy_image(a.image, None);
                    c.device.free_memory(a.memory, None);
                    return Err(e);
                }
            };
            let infos = [a.view, b.view].map(|v| [vk::DescriptorImageInfo { sampler: vk::Sampler::null(), image_view: v, image_layout: vk::ImageLayout::GENERAL }]);
            let writes = [0usize, 1].map(|k| vk::WriteDescriptorSet::default().dst_set(self.set).dst_binding(k as u32).descriptor_type(vk::DescriptorType::STORAGE_IMAGE).image_info(&infos[k]));
            ctx().device.update_descriptor_sets(&writes, &[]);
            let out = (a.image, b.image, true);
            self.imgs = Some((a, b, w, h));
            crate::info!("present: sharpening at {w}x{h}");
            Ok(out)
        }
    }

    fn barrier(img: vk::Image, old: vk::ImageLayout, src: (vk::PipelineStageFlags, vk::AccessFlags), dst: (vk::PipelineStageFlags, vk::AccessFlags), cmd: vk::CommandBuffer) {
        let range = vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 };
        let b = vk::ImageMemoryBarrier::default()
            .image(img)
            .old_layout(old)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_access_mask(src.1)
            .dst_access_mask(dst.1)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .subresource_range(range);
        unsafe { ctx().device.cmd_pipeline_barrier(cmd, src.0, dst.0, vk::DependencyFlags::empty(), &[], &[], &[b]) };
    }

    /// Before the frame is blitted into the input image: both images to GENERAL, after any
    /// earlier frame's reads and writes of them.
    pub fn begin(&self, cmd: vk::CommandBuffer, fresh: bool) {
        let Some((a, b, _, _)) = &self.imgs else { return };
        let old = if fresh { vk::ImageLayout::UNDEFINED } else { vk::ImageLayout::GENERAL };
        let prior = (vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::COMPUTE_SHADER, vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::SHADER_READ);
        Self::barrier(a.image, old, prior, (vk::PipelineStageFlags::TRANSFER, vk::AccessFlags::TRANSFER_WRITE), cmd);
        Self::barrier(b.image, old, prior, (vk::PipelineStageFlags::COMPUTE_SHADER, vk::AccessFlags::SHADER_WRITE), cmd);
    }

    /// The input image (blitted) sharpened into the output image, ready to be blitted from.
    pub fn record(&self, cmd: vk::CommandBuffer, amount: f32) {
        let Some((a, b, w, h)) = &self.imgs else { return };
        let d = &ctx().device;
        Self::barrier(a.image, vk::ImageLayout::GENERAL, (vk::PipelineStageFlags::TRANSFER, vk::AccessFlags::TRANSFER_WRITE), (vk::PipelineStageFlags::COMPUTE_SHADER, vk::AccessFlags::SHADER_READ), cmd);
        let mut pc = [0u8; 12];
        pc[0..4].copy_from_slice(&w.to_le_bytes());
        pc[4..8].copy_from_slice(&h.to_le_bytes());
        pc[8..12].copy_from_slice(&amount.to_le_bytes());
        unsafe {
            d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            d.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, self.layout, 0, &[self.set], &[]);
            d.cmd_push_constants(cmd, self.layout, vk::ShaderStageFlags::COMPUTE, 0, &pc);
            let n = rdna::kernels::SHARPEN_LOCAL;
            d.cmd_dispatch(cmd, w.div_ceil(n), h.div_ceil(n), 1);
        }
        Self::barrier(b.image, vk::ImageLayout::GENERAL, (vk::PipelineStageFlags::COMPUTE_SHADER, vk::AccessFlags::SHADER_WRITE), (vk::PipelineStageFlags::TRANSFER, vk::AccessFlags::TRANSFER_READ), cmd);
    }
}

impl Drop for Sharpen {
    fn drop(&mut self) {
        unsafe {
            self.drop_imgs();
            let d = &ctx().device;
            d.destroy_pipeline(self.pipeline, None);
            d.destroy_pipeline_layout(self.layout, None);
            d.destroy_descriptor_pool(self.pool, None);
            d.destroy_descriptor_set_layout(self.dsl, None);
        }
    }
}
