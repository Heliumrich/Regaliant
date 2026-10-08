//! Guest images on the GPU: one VkImage per guest surface (address,
//! format, size, layers, levels), shared by every use of that memory as a
//! texture, storage image or render target. Images first used as textures
//! are uploaded from guest memory (detiled with `tiling`); images the GPU
//! writes are owned by the GPU from then on. Samplers come from S#.
//!
//! All images stay in VK_IMAGE_LAYOUT_GENERAL; work is ordered by the
//! executor's global memory barriers.

use super::desc::Tsharp;
use super::tiling;
use super::vk::ctx;
use ash::vk;
use parking_lot::Mutex;
use std::collections::HashMap;

/// Vulkan format of a gfx10 image FORMAT.
pub fn vk_format(fmt: u16) -> Option<vk::Format> {
    use vk::Format as F;
    Some(match fmt {
        1 => F::R8_UNORM,
        2 => F::R8_SNORM,
        5 => F::R8_UINT,
        6 => F::R8_SINT,
        7 => F::R16_UNORM,
        8 => F::R16_SNORM,
        11 => F::R16_UINT,
        12 => F::R16_SINT,
        13 => F::R16_SFLOAT,
        14 => F::R8G8_UNORM,
        15 => F::R8G8_SNORM,
        18 => F::R8G8_UINT,
        19 => F::R8G8_SINT,
        20 => F::R32_UINT,
        21 => F::R32_SINT,
        22 => F::R32_SFLOAT,
        23 => F::R16G16_UNORM,
        24 => F::R16G16_SNORM,
        27 => F::R16G16_UINT,
        28 => F::R16G16_SINT,
        29 => F::R16G16_SFLOAT,
        36 => F::B10G11R11_UFLOAT_PACK32,
        50 => F::A2B10G10R10_UNORM_PACK32,
        54 => F::A2B10G10R10_UINT_PACK32,
        56 => F::R8G8B8A8_UNORM,
        57 => F::R8G8B8A8_SNORM,
        60 => F::R8G8B8A8_UINT,
        61 => F::R8G8B8A8_SINT,
        62 => F::R32G32_UINT,
        63 => F::R32G32_SINT,
        64 => F::R32G32_SFLOAT,
        65 => F::R16G16B16A16_UNORM,
        66 => F::R16G16B16A16_SNORM,
        69 => F::R16G16B16A16_UINT,
        70 => F::R16G16B16A16_SINT,
        71 => F::R16G16B16A16_SFLOAT,
        75 => F::R32G32B32A32_UINT,
        76 => F::R32G32B32A32_SINT,
        77 => F::R32G32B32A32_SFLOAT,
        128 => F::R8_SRGB,
        129 => F::R8G8_SRGB,
        130 => F::R8G8B8A8_SRGB,
        132 => F::E5B9G9R9_UFLOAT_PACK32,
        133 => F::B5G6R5_UNORM_PACK16,
        169 => F::BC1_RGBA_UNORM_BLOCK,
        170 => F::BC1_RGBA_SRGB_BLOCK,
        171 => F::BC2_UNORM_BLOCK,
        172 => F::BC2_SRGB_BLOCK,
        173 => F::BC3_UNORM_BLOCK,
        174 => F::BC3_SRGB_BLOCK,
        175 => F::BC4_UNORM_BLOCK,
        176 => F::BC4_SNORM_BLOCK,
        177 => F::BC5_UNORM_BLOCK,
        178 => F::BC5_SNORM_BLOCK,
        179 => F::BC6H_UFLOAT_BLOCK,
        180 => F::BC6H_SFLOAT_BLOCK,
        181 => F::BC7_UNORM_BLOCK,
        182 => F::BC7_SRGB_BLOCK,
        _ => return None,
    })
}

/// Bytes per element: texel, or 4x4 block of compressed formats (log2).
pub fn elem_log2(fmt: u16) -> Option<u32> {
    let f = rdna::format::gfx10(fmt)?;
    Some(f.bytes().trailing_zeros())
}

fn is_block(fmt: u16) -> bool {
    (169..=182).contains(&fmt)
}

/// UNORM equivalent of sRGB formats (storage views cannot be sRGB).
fn storage_format(f: vk::Format) -> vk::Format {
    match f {
        vk::Format::R8G8B8A8_SRGB => vk::Format::R8G8B8A8_UNORM,
        vk::Format::R8G8_SRGB => vk::Format::R8G8_UNORM,
        vk::Format::R8_SRGB => vk::Format::R8_UNORM,
        x => x,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ImageKey {
    pub base: u64,
    pub format: vk::Format,
    pub width: u32,
    pub height: u32,
    /// Array layers (cube faces), or depth of 3D images.
    pub layers: u32,
    pub levels: u32,
    pub ty3d: bool,
    pub cube: bool,
}

pub struct Image {
    pub image: vk::Image,
    pub key: ImageKey,
    views: HashMap<(vk::Format, vk::ImageViewType, u32, u32, u32, u32, [u8; 4]), vk::ImageView>,
    /// Written by the GPU: guest memory is no longer its source.
    pub gpu_written: bool,
    /// Guest bytes the upload came from (re-checked once per frame).
    source: Option<(u64, u64, u32, u16)>,
    hash: u64,
    checked_frame: u64,
}

pub struct Images {
    map: HashMap<ImageKey, Image>,
    samplers: HashMap<[u32; 4], vk::Sampler>,
    pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
    reported: HashMap<String, u64>,
    pub uploads: u64,
}

unsafe impl Send for Images {}

pub static IMAGES: Mutex<Option<Images>> = Mutex::new(None);

/// Runs `f` with the image cache.
pub fn with<R>(f: impl FnOnce(&mut Images) -> R) -> R {
    let mut g = IMAGES.lock();
    if g.is_none() {
        let c = ctx();
        unsafe {
            let pool = c
                .device
                .create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(c.queue_family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER), None)
                .expect("image pool");
            let cmd = c.device.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(pool).command_buffer_count(1)).expect("image cmd")[0];
            let fence = c.device.create_fence(&vk::FenceCreateInfo::default(), None).expect("image fence");
            *g = Some(Images { map: HashMap::new(), samplers: HashMap::new(), pool, cmd, fence, reported: HashMap::new(), uploads: 0 });
        }
    }
    f(g.as_mut().unwrap())
}

fn hash_bytes(b: &[u8]) -> u64 {
    // FNV-1a over 64-bit words (cheap; detects content changes).
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for c in b.chunks(8) {
        let mut w = [0u8; 8];
        w[..c.len()].copy_from_slice(c);
        h ^= u64::from_le_bytes(w);
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h
}

/// Sampled hash of a guest range: every 4 KiB page's first 256 bytes.
fn sample_hash(base: u64, size: u64) -> u64 {
    let mut h = 0u64;
    let mut a = base;
    while a < base + size {
        let n = 256.min(base + size - a) as usize;
        let s = unsafe { std::slice::from_raw_parts(a as *const u8, n) };
        h = h.rotate_left(5) ^ hash_bytes(s);
        a += 4096;
    }
    h
}

impl Images {
    fn report(&mut self, msg: String) {
        let n = self.reported.entry(msg.clone()).or_insert(0);
        *n += 1;
        if *n == 1 {
            crate::warn!("images: {msg}");
        }
    }

    /// Creates the VkImage for `key` (GENERAL layout, contents undefined).
    fn create(&mut self, key: ImageKey) -> Result<vk::Image, String> {
        let c = ctx();
        let props = unsafe { c.instance.get_physical_device_format_properties(c.pdev, key.format) };
        let f = props.optimal_tiling_features;
        let mut usage = vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC;
        if f.contains(vk::FormatFeatureFlags::SAMPLED_IMAGE) {
            usage |= vk::ImageUsageFlags::SAMPLED;
        }
        let sprops = unsafe { c.instance.get_physical_device_format_properties(c.pdev, storage_format(key.format)) };
        if sprops.optimal_tiling_features.contains(vk::FormatFeatureFlags::STORAGE_IMAGE) {
            usage |= vk::ImageUsageFlags::STORAGE;
        }
        if f.contains(vk::FormatFeatureFlags::COLOR_ATTACHMENT) {
            usage |= vk::ImageUsageFlags::COLOR_ATTACHMENT;
        }
        let mut flags = vk::ImageCreateFlags::MUTABLE_FORMAT | vk::ImageCreateFlags::EXTENDED_USAGE;
        if key.cube && key.layers % 6 == 0 {
            flags |= vk::ImageCreateFlags::CUBE_COMPATIBLE;
        }
        let (ty, extent, layers) = if key.ty3d {
            (vk::ImageType::TYPE_3D, vk::Extent3D { width: key.width, height: key.height, depth: key.layers }, 1)
        } else {
            (vk::ImageType::TYPE_2D, vk::Extent3D { width: key.width, height: key.height, depth: 1 }, key.layers)
        };
        let ci = vk::ImageCreateInfo::default()
            .flags(flags)
            .image_type(ty)
            .format(key.format)
            .extent(extent)
            .mip_levels(key.levels)
            .array_layers(layers)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        unsafe {
            let image = c.device.create_image(&ci, None).map_err(|e| format!("vkCreateImage {key:?}: {e}"))?;
            let req = c.device.get_image_memory_requirements(image);
            let mem = c.alloc(req, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
            c.device.bind_image_memory(image, mem, 0).map_err(|e| format!("bind image: {e}"))?;
            Ok(image)
        }
    }

    /// Records commands into the cache's command buffer and waits for them.
    fn one_shot(&mut self, f: impl FnOnce(vk::CommandBuffer)) {
        let c = ctx();
        unsafe {
            c.device.begin_command_buffer(self.cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).ok();
            f(self.cmd);
            c.device.end_command_buffer(self.cmd).ok();
            match c.submit(self.cmd, &[], &[], self.fence) {
                Ok(()) => {
                    if let Err(e) = c.device.wait_for_fences(&[self.fence], true, 10_000_000_000) {
                        if !super::exec::DEVICE_LOST.swap(true, std::sync::atomic::Ordering::SeqCst) {
                            crate::error!("images: upload/transition failed: {e}; GPU execution stops");
                        }
                    }
                }
                Err(e) => {
                    if !super::exec::DEVICE_LOST.swap(true, std::sync::atomic::Ordering::SeqCst) {
                        crate::error!("images: submit failed: {e}; GPU execution stops");
                    }
                }
            }
            c.device.reset_fences(&[self.fence]).ok();
            c.device.reset_command_buffer(self.cmd, vk::CommandBufferResetFlags::empty()).ok();
        }
    }

    fn to_general(&mut self, image: vk::Image) {
        self.one_shot(|cmd| unsafe {
            let b = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::GENERAL)
                .image(image)
                .subresource_range(vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: vk::REMAINING_MIP_LEVELS, base_array_layer: 0, layer_count: vk::REMAINING_ARRAY_LAYERS })
                .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
            ctx().device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, vk::PipelineStageFlags::ALL_COMMANDS, vk::DependencyFlags::empty(), &[], &[], &[b]);
        });
    }

    /// Uploads all mips/layers of a guest texture (detiled) into `image`.
    fn upload(&mut self, image: vk::Image, key: ImageKey, t: &Tsharp) -> Result<(u64, u64), String> {
        if key.ty3d {
            return Err("3D (thick) texture layouts are not implemented".into());
        }
        let el = elem_log2(t.format).ok_or("unknown element size")?;
        let block = is_block(t.format);
        let layers = if key.ty3d { 1 } else { key.layers };
        let lay = tiling::layout(t.sw_mode, el, key.width, key.height, key.levels, block).ok_or_else(|| format!("swizzle mode {} not supported", t.sw_mode))?;
        if tiling::is_xor(t.sw_mode) {
            self.report(format!("texture with XOR swizzle mode {} uploaded with an assumed pipe count (2^{})", t.sw_mode, tiling::pipes_log2()));
        }
        let size = lay.size(layers);
        if !super::cp::guest_readable(t.base, size) {
            return Err(format!("texture {:#x}+{size:#x} not mapped", t.base));
        }
        let src = unsafe { std::slice::from_raw_parts(t.base as *const u8, size as usize) };
        // Detile every (mip, layer) into one staging buffer.
        let mut staging: Vec<u8> = Vec::new();
        let mut regions = Vec::new();
        let mut tmp = Vec::new();
        for mip in 0..key.levels as usize {
            for layer in 0..layers {
                lay.detile(src, mip, layer, &mut tmp);
                let off = staging.len() as u64;
                staging.extend_from_slice(&tmp);
                let m = &lay.mips[mip];
                let (w, h) = (m.texel_w, m.texel_h);
                regions.push(
                    vk::BufferImageCopy::default()
                        .buffer_offset(off)
                        .buffer_row_length(if block { m.width * 4 } else { m.width })
                        .buffer_image_height(if block { m.height * 4 } else { m.height })
                        .image_subresource(vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: mip as u32, base_array_layer: layer, layer_count: 1 })
                        .image_extent(vk::Extent3D { width: w, height: h, depth: 1 }),
                );
            }
        }
        let c = ctx();
        let (buf, mem, ptr) = c.host_buffer(staging.len().max(4) as u64, vk::BufferUsageFlags::TRANSFER_SRC)?;
        unsafe { std::ptr::copy_nonoverlapping(staging.as_ptr(), ptr, staging.len()) };
        self.one_shot(|cmd| unsafe {
            let b = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::GENERAL)
                .image(image)
                .subresource_range(vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: vk::REMAINING_MIP_LEVELS, base_array_layer: 0, layer_count: vk::REMAINING_ARRAY_LAYERS })
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
            c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &[b]);
            c.device.cmd_copy_buffer_to_image(cmd, buf, image, vk::ImageLayout::GENERAL, &regions);
        });
        unsafe {
            c.device.destroy_buffer(buf, None);
            c.device.unmap_memory(mem);
            c.device.free_memory(mem, None);
        }
        self.uploads += 1;
        Ok((t.base, size))
    }

    /// Key of the image a T# describes.
    pub fn key_of(t: &Tsharp) -> Option<ImageKey> {
        let format = vk_format(t.format)?;
        // The resource's mip count (T# MAX_MIP), not the view's range.
        let levels = (t.max_mip + 1).max(t.last_level + 1).max(1);
        let (ty3d, cube) = (t.ty == 10, t.ty == 11);
        let layers = match t.ty {
            10 | 11 | 12 | 13 | 15 => t.depth.max(1),
            _ => 1,
        };
        Some(ImageKey { base: t.base, format, width: t.width, height: t.height, layers, levels, ty3d, cube })
    }

    /// Image for a T#: an existing GPU image at that address with a
    /// compatible size, or a texture uploaded from guest memory.
    fn image_for(&mut self, t: &Tsharp, frame: u64) -> Result<ImageKey, String> {
        let key = Self::key_of(t).ok_or_else(|| format!("image format {} unsupported", t.format))?;
        if self.map.contains_key(&key) {
            self.refresh(key, t, frame)?;
            return Ok(key);
        }
        // A GPU-written image of the same address and size (other format).
        if let Some(k) = self.map.iter().find(|(k, i)| i.gpu_written && k.base == key.base && k.width == key.width && k.height == key.height && k.layers >= key.layers).map(|(k, _)| *k) {
            return Ok(k);
        }
        let image = self.create(key)?;
        let source = match self.upload(image, key, t) {
            Ok(s) => Some(s),
            Err(e) => {
                self.report(format!("texture {:#x} ({}x{} format {} sw_mode {}): {e}; contents undefined", t.base, t.width, t.height, t.format, t.sw_mode));
                self.to_general(image);
                None
            }
        };
        let hash = source.map(|(b, n)| sample_hash(b, n)).unwrap_or(0);
        self.map.insert(key, Image { image, key, views: HashMap::new(), gpu_written: false, source: source.map(|(b, n)| (b, n, t.sw_mode, t.format)), hash, checked_frame: frame });
        Ok(key)
    }

    /// Re-uploads a CPU-sourced texture whose guest bytes changed (checked
    /// once per frame).
    fn refresh(&mut self, key: ImageKey, t: &Tsharp, frame: u64) -> Result<(), String> {
        let (image, check) = {
            let img = self.map.get_mut(&key).unwrap();
            if img.gpu_written || img.checked_frame == frame {
                return Ok(());
            }
            img.checked_frame = frame;
            (img.image, img.source)
        };
        let Some((b, n, _, _)) = check else { return Ok(()) };
        if !super::cp::guest_readable(b, n) {
            return Ok(());
        }
        let h = sample_hash(b, n);
        if h != self.map[&key].hash {
            self.upload(image, key, t)?;
            self.map.get_mut(&key).unwrap().hash = h;
        }
        Ok(())
    }

    /// View of a T# (format, type, levels/layers, swizzle) for sampling or storage.
    pub fn view(&mut self, t: &Tsharp, storage: bool, frame: u64) -> Result<vk::ImageView, String> {
        let key = self.image_for(t, frame)?;
        let req = vk_format(t.format).unwrap();
        let fmt = if storage { storage_format(req) } else { req };
        let levels = (t.last_level + 1).max(1);
        let _ = levels;
        let (vt, base_layer, layer_count) = if key.ty3d {
            (vk::ImageViewType::TYPE_3D, 0, 1)
        } else {
            match t.ty {
                11 if t.depth > 6 && !storage => (vk::ImageViewType::CUBE_ARRAY, 0, t.depth / 6 * 6),
                11 if !storage => (vk::ImageViewType::CUBE, 0, 6),
                11 => (vk::ImageViewType::TYPE_2D_ARRAY, 0, t.depth),
                12 | 13 | 15 => (vk::ImageViewType::TYPE_2D_ARRAY, t.base_array, t.depth.saturating_sub(t.base_array).max(1)),
                _ => (vk::ImageViewType::TYPE_2D, 0, 1),
            }
        };
        let sel = |s: u16| -> u8 { (s & 7) as u8 };
        let swz = if storage { [4, 5, 6, 7] } else { [sel(t.dst_sel), sel(t.dst_sel >> 3), sel(t.dst_sel >> 6), sel(t.dst_sel >> 9)] };
        let vkey = (fmt, vt, t.base_level, levels.saturating_sub(t.base_level).max(1), base_layer, layer_count, swz);
        let img = self.map.get_mut(&key).unwrap();
        if storage {
            img.gpu_written = true;
        }
        if let Some(&v) = img.views.get(&vkey) {
            return Ok(v);
        }
        let comp = |s: u8| match s {
            0 => vk::ComponentSwizzle::ZERO,
            1 => vk::ComponentSwizzle::ONE,
            4 => vk::ComponentSwizzle::R,
            5 => vk::ComponentSwizzle::G,
            6 => vk::ComponentSwizzle::B,
            _ => vk::ComponentSwizzle::A,
        };
        let levels_avail = img.key.levels;
        let base_level = t.base_level.min(levels_avail - 1);
        let ci = vk::ImageViewCreateInfo::default()
            .image(img.image)
            .view_type(vt)
            .format(fmt)
            .components(vk::ComponentMapping { r: comp(swz[0]), g: comp(swz[1]), b: comp(swz[2]), a: comp(swz[3]) })
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: base_level,
                level_count: if storage { 1 } else { (levels.saturating_sub(base_level)).clamp(1, levels_avail - base_level) },
                base_array_layer: base_layer,
                layer_count,
            });
        let mut usage = vk::ImageViewUsageCreateInfo::default().usage(if storage { vk::ImageUsageFlags::STORAGE } else { vk::ImageUsageFlags::SAMPLED });
        let ci = ci.push_next(&mut usage);
        let v = unsafe { ctx().device.create_image_view(&ci, None) }.map_err(|e| format!("image view {vkey:?}: {e}"))?;
        img.views.insert(vkey, v);
        Ok(v)
    }

    /// Sampler of an S#.
    pub fn sampler(&mut self, s: [u32; 4]) -> Result<vk::Sampler, String> {
        if let Some(&v) = self.samplers.get(&s) {
            return Ok(v);
        }
        let b = |v: u32, lo: u32, n: u32| (v >> lo) & ((1 << n) - 1);
        let addr = |m: u32| match m {
            0 => vk::SamplerAddressMode::REPEAT,
            1 => vk::SamplerAddressMode::MIRRORED_REPEAT,
            2 => vk::SamplerAddressMode::CLAMP_TO_EDGE,
            3 => vk::SamplerAddressMode::MIRROR_CLAMP_TO_EDGE,
            4 | 6 => vk::SamplerAddressMode::CLAMP_TO_BORDER,
            _ => vk::SamplerAddressMode::MIRROR_CLAMP_TO_EDGE,
        };
        let filt = |m: u32| if m & 1 != 0 { vk::Filter::LINEAR } else { vk::Filter::NEAREST };
        let aniso = b(s[0], 9, 3);
        let compare = b(s[0], 12, 3);
        let mip = b(s[2], 26, 2);
        let border = match b(s[3], 30, 2) {
            1 => vk::BorderColor::FLOAT_OPAQUE_BLACK,
            2 => vk::BorderColor::FLOAT_OPAQUE_WHITE,
            _ => vk::BorderColor::FLOAT_TRANSPARENT_BLACK,
        };
        let lod_bias = ((b(s[2], 0, 14) << 18) as i32 >> 18) as f32 / 256.0;
        let min_lod = b(s[1], 0, 12) as f32 / 256.0;
        let max_lod = b(s[1], 12, 12) as f32 / 256.0;
        let ci = vk::SamplerCreateInfo::default()
            .mag_filter(filt(b(s[2], 20, 2)))
            .min_filter(filt(b(s[2], 22, 2)))
            .mipmap_mode(if mip == 2 { vk::SamplerMipmapMode::LINEAR } else { vk::SamplerMipmapMode::NEAREST })
            .address_mode_u(addr(b(s[0], 0, 3)))
            .address_mode_v(addr(b(s[0], 3, 3)))
            .address_mode_w(addr(b(s[0], 6, 3)))
            .mip_lod_bias(lod_bias.clamp(-15.0, 15.0))
            .anisotropy_enable(aniso > 0 && b(s[2], 22, 2) >= 2)
            .max_anisotropy((1u32 << aniso.min(4)) as f32)
            .compare_enable(compare != 0)
            .compare_op(match compare {
                1 => vk::CompareOp::LESS,
                2 => vk::CompareOp::EQUAL,
                3 => vk::CompareOp::LESS_OR_EQUAL,
                4 => vk::CompareOp::GREATER,
                5 => vk::CompareOp::NOT_EQUAL,
                6 => vk::CompareOp::GREATER_OR_EQUAL,
                7 => vk::CompareOp::ALWAYS,
                _ => vk::CompareOp::NEVER,
            })
            .min_lod(min_lod)
            .max_lod(if mip == 0 { min_lod + 0.25 } else { max_lod.max(min_lod) })
            .border_color(border)
            .unnormalized_coordinates(false);
        let v = unsafe { ctx().device.create_sampler(&ci, None) }.map_err(|e| format!("sampler: {e}"))?;
        self.samplers.insert(s, v);
        Ok(v)
    }

    pub fn count(&self) -> usize {
        self.map.len()
    }
}
