//! GPU resources backing guest memory: the image cache (render targets and
//! textures share images by guest address), the per-batch upload ring and
//! samplers.

use super::desc::{self, Ssharp, Tsharp};
use super::tiling;
use super::vk::{Ctx, ctx};
use ash::vk;
use std::collections::HashMap;

pub const RING_SIZE: u64 = 256 << 20;
/// Device-local ring for buffers written by shaders (strided GPU writes to
/// host memory are very slow); copied back to the host ring per batch.
pub const DEV_RING_SIZE: u64 = 512 << 20;

pub struct Ring {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub ptr: *mut u8,
    pub size: u64,
    pub used: u64,
    pub align: u64,
    pub seg_lo: u64,
    pub seg_hi: u64,
}

unsafe impl Send for Ring {}

impl Ring {
    pub fn new(c: &Ctx, size: u64) -> Result<Ring, String> {
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::INDEX_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST
            | vk::BufferUsageFlags::UNIFORM_BUFFER;
        let (buffer, memory, ptr) = c.host_buffer(size, usage)?;
        let align = c.limits.min_storage_buffer_offset_alignment.max(16);
        Ok(Ring { buffer, memory, ptr, size, used: 0, align, seg_lo: 0, seg_hi: size })
    }
    /// Device-local ring (no CPU mapping: `write*` must not be used).
    pub fn new_device(c: &Ctx, size: u64) -> Result<Ring, String> {
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
        let (buffer, memory) = c.device_buffer(size, usage)?;
        let align = c.limits.min_storage_buffer_offset_alignment.max(16);
        Ok(Ring { buffer, memory, ptr: std::ptr::null_mut(), size, used: 0, align, seg_lo: 0, seg_hi: size })
    }
    /// Allocates `n` bytes in the current segment; returns the offset or None
    /// if the segment is full.
    pub fn alloc(&mut self, n: u64) -> Option<u64> {
        let off = self.used.next_multiple_of(self.align);
        if off + n.max(4) > self.seg_hi {
            return None;
        }
        self.used = off + n.max(4);
        Some(off)
    }
    /// Makes segment `i` of `count` current (its previous contents must no
    /// longer be in use by the GPU).
    pub fn set_segment(&mut self, i: usize, count: usize) {
        let seg = (self.size / count as u64) & !0xFFFF;
        self.seg_lo = seg * i as u64;
        self.seg_hi = self.seg_lo + seg;
        self.used = self.seg_lo;
    }
    pub fn remaining(&self) -> u64 {
        self.seg_hi.saturating_sub(self.used)
    }
    pub fn segment_size(&self) -> u64 {
        self.seg_hi - self.seg_lo
    }
    pub fn write(&mut self, off: u64, data: &[u8]) {
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), self.ptr.add(off as usize), data.len()) };
    }
    /// Copies guest memory into the ring (unmapped bytes read as zero).
    pub fn write_guest(&mut self, off: u64, addr: u64, n: u64) {
        let dst = unsafe { std::slice::from_raw_parts_mut(self.ptr.add(off as usize), n as usize) };
        copy_from_guest(dst, addr);
    }
}

/// Copies from guest memory, zero-filling bytes that are not mapped.
pub fn copy_from_guest(dst: &mut [u8], addr: u64) {
    let n = dst.len() as u64;
    if addr < crate::mem::GUEST_BASE || addr + n > crate::mem::GUEST_END {
        dst.fill(0);
        return;
    }
    let st = crate::mem::MEM.lock();
    let mut a = addr;
    while a < addr + n {
        let i = (a - addr) as usize;
        match st.find(a) {
            Some(v) if v.kind != crate::mem::VmaKind::Reserved => {
                let end = v.end().min(addr + n);
                let len = (end - a) as usize;
                unsafe { std::ptr::copy_nonoverlapping(a as *const u8, dst[i..].as_mut_ptr(), len) };
                a = end;
            }
            _ => {
                // Unmapped: zero up to the next mapping or the end.
                let next = st.vmas.range(a..).next().map(|(_, v)| v.start).unwrap_or(addr + n).min(addr + n);
                let next = next.max(a + 1);
                dst[i..(next - addr) as usize].fill(0);
                a = next;
            }
        }
    }
}

pub fn copy_to_guest(src: &[u8], addr: u64) -> bool {
    let n = src.len() as u64;
    let st = crate::mem::MEM.lock();
    match st.find(addr) {
        Some(v) if v.kind != crate::mem::VmaKind::Reserved && addr + n <= v.end() => {
            drop(st);
            unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), addr as *mut u8, src.len()) };
            true
        }
        _ => false,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ImageKey {
    pub base: u64,
    pub width: u32,
    pub height: u32,
    pub layers: u32,
    pub levels: u32,
    pub format: vk::Format,
    pub depth: bool,
    pub ty3d: bool,
}

pub struct Image {
    pub key: ImageKey,
    pub image: vk::Image,
    pub memory: vk::DeviceMemory,
    pub views: HashMap<(vk::Format, u16, bool, u32, u32), vk::ImageView>,
    /// Contents were produced by the GPU (do not re-upload from guest memory).
    pub gpu_written: bool,
    /// Hash of guest memory at last upload (CPU-sourced images).
    pub uploaded_hash: u64,
    pub upload_batch: u64,
    pub storage_ok: bool,
    /// Tile mode index of the guest surface (render targets), for write-back.
    pub tile_index: Option<u8>,
    /// GPU contents are newer than guest memory (rendered since last write-back).
    pub guest_stale: bool,
    /// Renderer draw tick of the last write as a render target (0 = never).
    pub last_write: u64,
    /// Frame (flip count) of the last write as a render target.
    pub last_write_frame: u64,
    /// Compression metadata surface (HTILE of a depth target, CMASK of a
    /// fast-clear color target): guest (base, bytes).
    pub meta: Option<(u64, u64)>,
    /// GPU work wrote the metadata surface since it was last examined.
    pub meta_dirty: bool,
    /// GPU buffer writes (compute stores, GDS/DMA transfers) changed the
    /// guest surface after the image was last rendered: guest memory is
    /// newer and is reloaded into the image before its next use.
    pub guest_newer: bool,
}

pub fn is_depth(f: vk::Format) -> bool {
    matches!(f, vk::Format::D16_UNORM | vk::Format::D32_SFLOAT | vk::Format::D32_SFLOAT_S8_UINT | vk::Format::D24_UNORM_S8_UINT)
}

fn storage_capable(c: &Ctx, f: vk::Format) -> bool {
    let p = unsafe { c.instance.get_physical_device_format_properties(c.pdev, f) };
    p.optimal_tiling_features.contains(vk::FormatFeatureFlags::STORAGE_IMAGE)
}

fn attachment_capable(c: &Ctx, f: vk::Format) -> bool {
    let p = unsafe { c.instance.get_physical_device_format_properties(c.pdev, f) };
    p.optimal_tiling_features.intersects(vk::FormatFeatureFlags::COLOR_ATTACHMENT | vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT)
}

pub fn create_image(key: ImageKey) -> Result<Image, String> {
    let c = ctx();
    let depth = is_depth(key.format);
    let mut usage = vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC;
    if attachment_capable(c, key.format) {
        usage |= if depth { vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT } else { vk::ImageUsageFlags::COLOR_ATTACHMENT };
    }
    let storage_ok = !depth && storage_capable(c, key.format);
    if storage_ok {
        usage |= vk::ImageUsageFlags::STORAGE;
    }
    let mut flags = vk::ImageCreateFlags::empty();
    if !depth {
        flags |= vk::ImageCreateFlags::MUTABLE_FORMAT;
    }
    if !key.ty3d && key.layers >= 6 && key.width == key.height {
        flags |= vk::ImageCreateFlags::CUBE_COMPATIBLE;
    }
    let ici = vk::ImageCreateInfo::default()
        .flags(flags)
        .image_type(if key.ty3d { vk::ImageType::TYPE_3D } else { vk::ImageType::TYPE_2D })
        .format(key.format)
        .extent(vk::Extent3D { width: key.width, height: key.height, depth: if key.ty3d { key.layers } else { 1 } })
        .mip_levels(key.levels)
        .array_layers(if key.ty3d { 1 } else { key.layers })
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(usage)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    unsafe {
        let image = c.device.create_image(&ici, None).map_err(|e| format!("vkCreateImage {key:?}: {e}"))?;
        let req = c.device.get_image_memory_requirements(image);
        let memory = c.alloc(req, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
        c.device.bind_image_memory(image, memory, 0).map_err(|e| e.to_string())?;
        Ok(Image { key, image, memory, views: HashMap::new(), gpu_written: false, uploaded_hash: 0, upload_batch: u64::MAX, storage_ok, tile_index: None, guest_stale: false, last_write: 0, last_write_frame: 0, meta: None, meta_dirty: false, guest_newer: false })
    }
}

pub fn aspect(f: vk::Format) -> vk::ImageAspectFlags {
    match f {
        vk::Format::D16_UNORM | vk::Format::D32_SFLOAT => vk::ImageAspectFlags::DEPTH,
        vk::Format::D32_SFLOAT_S8_UINT | vk::Format::D24_UNORM_S8_UINT => vk::ImageAspectFlags::DEPTH,
        _ => vk::ImageAspectFlags::COLOR,
    }
}

impl Image {
    /// View with a (compatible) format, T# dst_sel swizzle and dimensionality.
    /// `base_layer`/`layer_count` select the viewed slices (array views).
    pub fn view(&mut self, fmt: vk::Format, dst_sel: u16, view_type: vk::ImageViewType, storage: bool, base_layer: u32, layer_count: u32) -> Result<vk::ImageView, String> {
        let total = if self.key.ty3d { 1 } else { self.key.layers };
        let base_layer = base_layer.min(total - 1);
        let is_cube = matches!(view_type, vk::ImageViewType::CUBE | vk::ImageViewType::CUBE_ARRAY);
        if is_cube && total < 6 {
            return Err(format!("cube view of an image with {total} layers"));
        }
        // Cube views need whole cubes inside the image.
        let base_layer = if is_cube { (base_layer / 6 * 6).min(total / 6 * 6 - 6) } else { base_layer };
        let layer_count = match view_type {
            vk::ImageViewType::TYPE_2D | vk::ImageViewType::TYPE_1D | vk::ImageViewType::TYPE_3D => 1,
            vk::ImageViewType::CUBE => 6,
            vk::ImageViewType::CUBE_ARRAY => (layer_count / 6 * 6).clamp(6, (total - base_layer) / 6 * 6),
            _ => layer_count.clamp(1, total - base_layer),
        };
        let key = (fmt, (if storage { 0 } else { dst_sel }) ^ ((view_type.as_raw() as u16) << 12), storage, base_layer, layer_count);
        if let Some(&v) = self.views.get(&key) {
            return Ok(v);
        }
        let c = ctx();
        let sel = |s: u16| match s {
            0 => vk::ComponentSwizzle::ZERO,
            1 => vk::ComponentSwizzle::ONE,
            4 => vk::ComponentSwizzle::R,
            5 => vk::ComponentSwizzle::G,
            6 => vk::ComponentSwizzle::B,
            7 => vk::ComponentSwizzle::A,
            _ => vk::ComponentSwizzle::IDENTITY,
        };
        let comps = if storage || dst_sel == 0 || is_depth(fmt) {
            vk::ComponentMapping::default()
        } else {
            vk::ComponentMapping { r: sel(dst_sel & 7), g: sel((dst_sel >> 3) & 7), b: sel((dst_sel >> 6) & 7), a: sel((dst_sel >> 9) & 7) }
        };
        let mut usage_ci = vk::ImageViewUsageCreateInfo::default().usage(if storage { vk::ImageUsageFlags::STORAGE } else { vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::COLOR_ATTACHMENT });
        let vci = vk::ImageViewCreateInfo::default()
            .image(self.image)
            .view_type(view_type)
            .format(fmt)
            .components(comps)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: aspect(fmt),
                base_mip_level: 0,
                level_count: if storage { 1 } else { self.key.levels },
                base_array_layer: base_layer,
                layer_count,
            })
            .push_next(&mut usage_ci);
        let v = unsafe { c.device.create_image_view(&vci, None) }.map_err(|e| format!("vkCreateImageView {fmt:?} {view_type:?}: {e}"))?;
        self.views.insert(key, v);
        Ok(v)
    }
}

/// Bits per element for upload sizing.
pub fn element_bits(dfmt: u8) -> (u32, u32) {
    let (bytes, block) = desc::format_bytes(dfmt);
    (bytes * 8, block)
}

/// Linearizes a texture's base level (all layers) from guest memory.
/// Returns tightly packed data (rows of `w_elems` elements) per layer.
pub fn read_texture(t: &Tsharp, layers: u32) -> Vec<u8> {
    let (bpp, block) = element_bits(t.dfmt);
    let we = t.width.div_ceil(block);
    let he = t.height.div_ceil(block);
    let pitch_e = t.pitch.max(t.width).div_ceil(block);
    let mode = tiling::tile_mode(t.tiling_index);
    let (pw, ph) = tiling::padded(pitch_e, he, bpp, mode);
    let slice = tiling::slice_bytes(pw, ph, bpp);
    let mut out = vec![0u8; (we as usize) * (he as usize) * (bpp as usize / 8) * layers as usize];
    let per_layer = we as usize * he as usize * bpp as usize / 8;
    let mut src = vec![0u8; slice as usize];
    for l in 0..layers {
        copy_from_guest(&mut src, t.base + slice * l as u64);
        tiling::detile_slice(&src, &mut out[l as usize * per_layer..(l as usize + 1) * per_layer], we, he, pw, ph, bpp, mode);
    }
    out
}

pub fn hash_guest(addr: u64, n: u64) -> u64 {
    let mut buf = vec![0u8; n.min(64 << 20) as usize];
    copy_from_guest(&mut buf, addr);
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for c in buf.chunks(8) {
        let mut w = [0u8; 8];
        w[..c.len()].copy_from_slice(c);
        h = (h ^ u64::from_le_bytes(w)).wrapping_mul(0x100_0000_01b3);
        h ^= h >> 29;
    }
    h
}

pub fn texture_bytes(t: &Tsharp, layers: u32) -> u64 {
    let (bpp, block) = element_bits(t.dfmt);
    let pitch_e = t.pitch.max(t.width).div_ceil(block);
    let he = t.height.div_ceil(block);
    let (pw, ph) = tiling::padded(pitch_e, he, bpp, tiling::tile_mode(t.tiling_index));
    tiling::slice_bytes(pw, ph, bpp) * layers as u64
}

pub struct Samplers {
    map: HashMap<[u32; 4], vk::Sampler>,
}

impl Samplers {
    pub fn new() -> Samplers {
        Samplers { map: HashMap::new() }
    }
    pub fn get(&mut self, s: &Ssharp) -> Result<vk::Sampler, String> {
        if let Some(&v) = self.map.get(&s.0) {
            return Ok(v);
        }
        let c = ctx();
        let filt = |f: u32| if f == 0 { vk::Filter::NEAREST } else { vk::Filter::LINEAR };
        let aniso = s.min_filter() >= 2 && s.max_aniso() > 0;
        let sci = vk::SamplerCreateInfo::default()
            .mag_filter(filt(s.mag_filter()))
            .min_filter(filt(s.min_filter() & 1))
            .mipmap_mode(if s.mip_filter() == 2 { vk::SamplerMipmapMode::LINEAR } else { vk::SamplerMipmapMode::NEAREST })
            .address_mode_u(desc::vk_address_mode(s.clamp(0)))
            .address_mode_v(desc::vk_address_mode(s.clamp(1)))
            .address_mode_w(desc::vk_address_mode(s.clamp(2)))
            .mip_lod_bias(s.lod_bias())
            .anisotropy_enable(aniso)
            .max_anisotropy((1u32 << s.max_aniso()).min(16) as f32)
            .compare_enable(s.compare_func() != 0)
            .compare_op(desc::vk_compare(s.compare_func()))
            .min_lod(s.min_lod())
            .max_lod(if s.mip_filter() == 0 { 0.25 } else { s.max_lod().max(s.min_lod()) })
            .border_color(match s.border_type() {
                0 => vk::BorderColor::FLOAT_TRANSPARENT_BLACK,
                1 => vk::BorderColor::FLOAT_OPAQUE_BLACK,
                _ => vk::BorderColor::FLOAT_OPAQUE_WHITE,
            })
            .unnormalized_coordinates(false);
        let v = unsafe { c.device.create_sampler(&sci, None) }.map_err(|e| format!("vkCreateSampler: {e}"))?;
        self.map.insert(s.0, v);
        Ok(v)
    }
}
