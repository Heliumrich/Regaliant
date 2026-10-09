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

/// Bytes per texel (per 4x4 block for compressed formats); 0 if unknown.
pub fn texel_bytes(f: vk::Format) -> u32 {
    use vk::Format as F;
    match f {
        F::R8_UNORM | F::R8_SNORM | F::R8_UINT | F::R8_SINT | F::R8_SRGB | F::S8_UINT => 1,
        F::R16_UNORM | F::R16_SNORM | F::R16_UINT | F::R16_SINT | F::R16_SFLOAT | F::R8G8_UNORM | F::R8G8_SNORM | F::R8G8_UINT | F::R8G8_SINT | F::R8G8_SRGB | F::B5G6R5_UNORM_PACK16 | F::D16_UNORM => 2,
        F::R32_UINT | F::R32_SINT | F::R32_SFLOAT | F::R16G16_UNORM | F::R16G16_SNORM | F::R16G16_UINT | F::R16G16_SINT | F::R16G16_SFLOAT | F::B10G11R11_UFLOAT_PACK32 | F::A2B10G10R10_UNORM_PACK32 | F::A2B10G10R10_UINT_PACK32
        | F::A2R10G10B10_UNORM_PACK32 | F::R8G8B8A8_UNORM | F::R8G8B8A8_SNORM | F::R8G8B8A8_UINT | F::R8G8B8A8_SINT | F::R8G8B8A8_SRGB | F::B8G8R8A8_UNORM | F::B8G8R8A8_SRGB | F::E5B9G9R9_UFLOAT_PACK32 | F::D32_SFLOAT => 4,
        F::R32G32_UINT | F::R32G32_SINT | F::R32G32_SFLOAT | F::R16G16B16A16_UNORM | F::R16G16B16A16_SNORM | F::R16G16B16A16_UINT | F::R16G16B16A16_SINT | F::R16G16B16A16_SFLOAT | F::D32_SFLOAT_S8_UINT => 8,
        F::R32G32B32A32_UINT | F::R32G32B32A32_SINT | F::R32G32B32A32_SFLOAT => 16,
        F::BC1_RGBA_UNORM_BLOCK | F::BC1_RGBA_SRGB_BLOCK | F::BC4_UNORM_BLOCK | F::BC4_SNORM_BLOCK => 8,
        F::BC2_UNORM_BLOCK | F::BC2_SRGB_BLOCK | F::BC3_UNORM_BLOCK | F::BC3_SRGB_BLOCK | F::BC5_UNORM_BLOCK | F::BC5_SNORM_BLOCK | F::BC6H_UFLOAT_BLOCK | F::BC6H_SFLOAT_BLOCK | F::BC7_UNORM_BLOCK | F::BC7_SRGB_BLOCK => 16,
        _ => 0,
    }
}

pub fn is_depth(f: vk::Format) -> bool {
    matches!(f, vk::Format::D16_UNORM | vk::Format::D32_SFLOAT | vk::Format::D32_SFLOAT_S8_UINT | vk::Format::D16_UNORM_S8_UINT | vk::Format::D24_UNORM_S8_UINT | vk::Format::S8_UINT)
}

pub fn has_stencil(f: vk::Format) -> bool {
    matches!(f, vk::Format::D32_SFLOAT_S8_UINT | vk::Format::D16_UNORM_S8_UINT | vk::Format::D24_UNORM_S8_UINT | vk::Format::S8_UINT)
}

/// All aspects of a format.
pub fn aspects(f: vk::Format) -> vk::ImageAspectFlags {
    match f {
        vk::Format::D16_UNORM | vk::Format::D32_SFLOAT => vk::ImageAspectFlags::DEPTH,
        vk::Format::S8_UINT => vk::ImageAspectFlags::STENCIL,
        f if is_depth(f) => vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL,
        _ => vk::ImageAspectFlags::COLOR,
    }
}

fn is_compressed(f: vk::Format) -> bool {
    (vk::Format::BC1_RGB_UNORM_BLOCK.as_raw()..=vk::Format::BC7_SRGB_BLOCK.as_raw()).contains(&f.as_raw())
}

/// Two formats can view the same image (MUTABLE_FORMAT): same texel size,
/// both color, both compressed or both not.
fn view_compatible(a: vk::Format, b: vk::Format) -> bool {
    a == b || (texel_bytes(a) != 0 && texel_bytes(a) == texel_bytes(b) && !is_depth(a) && !is_depth(b) && is_compressed(a) == is_compressed(b))
}

/// CB color format (CB_COLORn_INFO FORMAT, NUMBER_TYPE, COMP_SWAP) -> Vulkan format.
pub fn cb_format(fmt: u32, ntype: u32, swap: u32) -> Option<vk::Format> {
    use vk::Format as F;
    // NUMBER_TYPE: 0 unorm, 1 snorm, 4 uint, 5 sint, 6 srgb, 7 float.
    let f = match (fmt, ntype) {
        (1, 0) => F::R8_UNORM,
        (1, 1) => F::R8_SNORM,
        (1, 4) => F::R8_UINT,
        (1, 5) => F::R8_SINT,
        (1, 6) => F::R8_SRGB,
        (2, 0) => F::R16_UNORM,
        (2, 1) => F::R16_SNORM,
        (2, 4) => F::R16_UINT,
        (2, 5) => F::R16_SINT,
        (2, 7) => F::R16_SFLOAT,
        (3, 0) => F::R8G8_UNORM,
        (3, 1) => F::R8G8_SNORM,
        (3, 4) => F::R8G8_UINT,
        (3, 5) => F::R8G8_SINT,
        (3, 6) => F::R8G8_SRGB,
        (4, 4) => F::R32_UINT,
        (4, 5) => F::R32_SINT,
        (4, 7) => F::R32_SFLOAT,
        (5, 0) => F::R16G16_UNORM,
        (5, 1) => F::R16G16_SNORM,
        (5, 4) => F::R16G16_UINT,
        (5, 5) => F::R16G16_SINT,
        (5, 7) => F::R16G16_SFLOAT,
        (6, 7) => F::B10G11R11_UFLOAT_PACK32,
        (9, 0) => F::A2B10G10R10_UNORM_PACK32,
        (9, 4) => F::A2B10G10R10_UINT_PACK32,
        (10, 0) => F::R8G8B8A8_UNORM,
        (10, 1) => F::R8G8B8A8_SNORM,
        (10, 4) => F::R8G8B8A8_UINT,
        (10, 5) => F::R8G8B8A8_SINT,
        (10, 6) => F::R8G8B8A8_SRGB,
        (11, 4) => F::R32G32_UINT,
        (11, 5) => F::R32G32_SINT,
        (11, 7) => F::R32G32_SFLOAT,
        (12, 0) => F::R16G16B16A16_UNORM,
        (12, 1) => F::R16G16B16A16_SNORM,
        (12, 4) => F::R16G16B16A16_UINT,
        (12, 5) => F::R16G16B16A16_SINT,
        (12, 7) => F::R16G16B16A16_SFLOAT,
        (14, 4) => F::R32G32B32A32_UINT,
        (14, 5) => F::R32G32B32A32_SINT,
        (14, 7) => F::R32G32B32A32_SFLOAT,
        (16, 0) => F::B5G6R5_UNORM_PACK16,
        _ => return None,
    };
    // COMP_SWAP 1 (ALT) on 4-component formats stores BGRA.
    Some(match (f, swap) {
        (_, 0) => f,
        (F::R8G8B8A8_UNORM, 1) => F::B8G8R8A8_UNORM,
        (F::R8G8B8A8_SRGB, 1) => F::B8G8R8A8_SRGB,
        (F::A2B10G10R10_UNORM_PACK32, 1) => F::A2R10G10B10_UNORM_PACK32,
        _ => return None,
    })
}

pub fn half_to_f32(h: u16) -> f32 {
    let s = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let e = ((h >> 10) & 0x1F) as i32;
    let m = (h & 0x3FF) as f32;
    s * match e {
        0 => m * 2f32.powi(-24),
        31 => f32::INFINITY,
        _ => (1.0 + m / 1024.0) * 2f32.powi(e - 15),
    }
}

/// Converts tightly packed texels to RGBA8 (screenshots, dumps).
pub fn to_rgba8(format: vk::Format, raw: &[u8]) -> Option<Vec<u8>> {
    use vk::Format as F;
    let unit = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
    Some(match format {
        F::R8G8B8A8_UNORM | F::R8G8B8A8_SRGB => raw.to_vec(),
        F::B8G8R8A8_UNORM | F::B8G8R8A8_SRGB => raw.chunks_exact(4).flat_map(|p| [p[2], p[1], p[0], p[3]]).collect(),
        F::A2B10G10R10_UNORM_PACK32 | F::A2R10G10B10_UNORM_PACK32 => raw
            .chunks_exact(4)
            .flat_map(|p| {
                let v = u32::from_le_bytes([p[0], p[1], p[2], p[3]]);
                let lo = ((v & 0x3FF) >> 2) as u8;
                let g = (((v >> 10) & 0x3FF) >> 2) as u8;
                let hi = (((v >> 20) & 0x3FF) >> 2) as u8;
                let a = ((v >> 30) * 85) as u8;
                if format == F::A2B10G10R10_UNORM_PACK32 { [lo, g, hi, a] } else { [hi, g, lo, a] }
            })
            .collect(),
        F::R16G16B16A16_SFLOAT => raw
            .chunks_exact(8)
            .flat_map(|p| {
                let h = |i: usize| half_to_f32(u16::from_le_bytes([p[2 * i], p[2 * i + 1]]));
                [unit(h(0)), unit(h(1)), unit(h(2)), unit(h(3))]
            })
            .collect(),
        F::R8_UNORM => raw.iter().flat_map(|&v| [v, v, v, 255]).collect(),
        F::R32_SFLOAT => raw
            .chunks_exact(4)
            .flat_map(|p| {
                let v = unit(f32::from_le_bytes([p[0], p[1], p[2], p[3]]));
                [v, v, v, 255]
            })
            .collect(),
        F::R16G16_SFLOAT => raw
            .chunks_exact(4)
            .flat_map(|p| [unit(half_to_f32(u16::from_le_bytes([p[0], p[1]]))), unit(half_to_f32(u16::from_le_bytes([p[2], p[3]]))), 0, 255])
            .collect(),
        F::R32_UINT => raw.chunks_exact(4).flat_map(|p| [p[0], p[1], p[2], 255]).collect(),
        F::R8_UINT => raw.iter().flat_map(|&v| [v, v, v, 255]).collect(),
        F::R8G8_UINT | F::R8G8_UNORM => raw.chunks_exact(2).flat_map(|p| [p[0], p[1], 0, 255]).collect(),
        F::R16_SFLOAT => raw
            .chunks_exact(2)
            .flat_map(|p| {
                let v = unit(half_to_f32(u16::from_le_bytes([p[0], p[1]])));
                [v, v, v, 255]
            })
            .collect(),
        _ => return None,
    })
}

/// Per-channel statistics of a float image (finite min / max / mean, NaN
/// and infinity counts): HDR targets clamp to white or black as PNG.
pub fn float_stats(format: vk::Format, raw: &[u8]) -> Option<String> {
    use vk::Format as F;
    let (channels, half) = match format {
        F::R16G16B16A16_SFLOAT => (4, true),
        F::R16G16_SFLOAT => (2, true),
        F::R16_SFLOAT => (1, true),
        F::R32G32B32A32_SFLOAT => (4, false),
        F::R32G32_SFLOAT => (2, false),
        F::R32_SFLOAT => (1, false),
        _ => return None,
    };
    let size = if half { 2 } else { 4 };
    let mut st = vec![(f32::INFINITY, f32::NEG_INFINITY, 0f64, 0u64, 0u64, 0u64); channels];
    for texel in raw.chunks_exact(size * channels) {
        for (c, s) in st.iter_mut().enumerate() {
            let p = &texel[c * size..];
            let v = if half { half_to_f32(u16::from_le_bytes([p[0], p[1]])) } else { f32::from_le_bytes([p[0], p[1], p[2], p[3]]) };
            if v.is_nan() {
                s.4 += 1;
            } else if v.is_infinite() {
                s.5 += 1;
            } else {
                s.0 = s.0.min(v);
                s.1 = s.1.max(v);
                s.2 += v as f64;
                s.3 += 1;
            }
        }
    }
    let parts: Vec<String> = st
        .iter()
        .zip("rgba".chars())
        .map(|(s, n)| {
            let mean = if s.3 > 0 { s.2 / s.3 as f64 } else { 0.0 };
            format!("{n} [{:.4}, {:.4}] mean {mean:.4} nan {} inf {}", s.0, s.1, s.4, s.5)
        })
        .collect();
    Some(parts.join("; "))
}

/// A render target or depth/stencil attachment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Target {
    pub key: ImageKey,
    pub image: vk::Image,
    pub view: vk::ImageView,
    /// Attachment (view) format.
    pub format: vk::Format,
    /// Size of the attached mip level.
    pub width: u32,
    pub height: u32,
}

/// Color or depth surface described by CB/DB registers.
#[derive(Clone, Copy, Debug)]
pub struct Surface {
    pub base: u64,
    pub format: vk::Format,
    pub width: u32,
    pub height: u32,
    /// Array layers of the resource (3D: depth).
    pub layers: u32,
    pub levels: u32,
    pub ty3d: bool,
    pub mip: u32,
    pub slice: u32,
    /// Address of the stencil plane (depth/stencil surfaces with stencil).
    pub stencil_base: u64,
    /// Metadata a fast clear writes (DCC of color, HTILE of depth; 0: none).
    pub meta: u64,
    /// Clear values: CB_COLOR_CLEAR_WORD0/1, or DB_DEPTH_CLEAR and DB_STENCIL_CLEAR.
    pub clear: [u32; 2],
}

/// A fast clear found pending for an image: its metadata and clear values.
pub struct FastClear {
    pub key: ImageKey,
    pub image: vk::Image,
    pub meta: u64,
    pub clear: [u32; 2],
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
    /// Views by (format, type, base mip, mips, base layer, layers, swizzle,
    /// usage: 0 sampled, 1 storage, 2 attachment). Usage is part of the key:
    /// a view created for one usage is not valid for another.
    views: HashMap<(vk::Format, vk::ImageViewType, u32, u32, u32, u32, [u8; 4], u8), vk::ImageView>,
    /// Written by the GPU: guest memory is no longer its source.
    pub gpu_written: bool,
    /// Write-tracking region of the guest bytes the upload came from.
    source: Option<u32>,
    /// Sampled hashes of those bytes, one per page slice (see slice_hash).
    hashes: [u64; SLICES],
    /// Frame of the last slice check.
    checked_frame: u64,
}

/// Pages of a texture are checked for CPU writes in SLICES interleaved
/// slices, one slice per frame (a change is seen within SLICES frames).
const SLICES: usize = 8;

/// Hash of `n` bytes at `a` (four independent lanes over 64-bit words).
fn hash_bytes(a: u64, n: usize) -> u64 {
    let mut h = [0x9E37_79B9_7F4A_7C15u64, 0xC2B2_AE3D_27D4_EB4F, 0x1656_67B1_9E37_79F9, 0x27D4_EB2F_1656_67C5];
    let words = n / 8;
    let p = a as *const u64;
    let mut i = 0;
    while i + 4 <= words {
        for (l, hl) in h.iter_mut().enumerate() {
            let x = unsafe { p.add(i + l).read_unaligned() };
            *hl = (*hl ^ x).wrapping_mul(0x0000_0100_0000_01B3).rotate_left(29);
        }
        i += 4;
    }
    let mut tail = 0u64;
    for k in i * 8..n {
        tail = tail.rotate_left(8) ^ unsafe { *((a + k as u64) as *const u8) } as u64;
    }
    h[0] ^ h[1].rotate_left(17) ^ h[2].rotate_left(31) ^ h[3].rotate_left(47) ^ tail.wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

/// Sampled hash of one slice of a guest range: the first 256 bytes of every
/// SLICES-th 4 KiB page starting at page `slice`.
fn slice_hash(base: u64, size: u64, slice: usize) -> u64 {
    if !super::cp::guest_readable(base, size) {
        return 0;
    }
    let mut h = 0u64;
    let mut a = base + 4096 * slice as u64;
    while a < base + size {
        let n = 256.min(base + size - a) as usize;
        h = h.rotate_left(5) ^ hash_bytes(a, n);
        a += 4096 * SLICES as u64;
    }
    h
}

fn all_hashes(base: u64, size: u64) -> [u64; SLICES] {
    std::array::from_fn(|s| slice_hash(base, size, s))
}

pub struct Images {
    map: HashMap<ImageKey, Image>,
    /// Addresses with a GPU-written image (render targets, storage images,
    /// stencil planes).
    gpu_bases: std::collections::HashSet<u64>,
    /// Depth/stencil image of each rendered stencil plane address.
    stencil_of: HashMap<u64, ImageKey>,
    /// Surface of each metadata address (DCC, HTILE) and its clear values.
    meta_of: HashMap<u64, (ImageKey, [u32; 2])>,
    /// Images whose metadata a GPU or CP write covered since (fast clears).
    pending_clears: HashMap<ImageKey, u64>,
    samplers: HashMap<([u32; 4], bool), vk::Sampler>,
    pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
    reported: HashMap<String, u64>,
    pub uploads: u64,
    pub reuploads: u64,
    pub upload_bytes: u64,
    pub upload_ns: u64,
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
            *g = Some(Images { map: HashMap::new(), gpu_bases: Default::default(), stencil_of: HashMap::new(), meta_of: HashMap::new(), pending_clears: HashMap::new(), samplers: HashMap::new(), pool, cmd, fence, reported: HashMap::new(), uploads: 0, reuploads: 0, upload_bytes: 0, upload_ns: 0 });
        }
    }
    f(g.as_mut().unwrap())
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
        if !is_depth(key.format) {
            let sprops = unsafe { c.instance.get_physical_device_format_properties(c.pdev, storage_format(key.format)) };
            if sprops.optimal_tiling_features.contains(vk::FormatFeatureFlags::STORAGE_IMAGE) {
                usage |= vk::ImageUsageFlags::STORAGE;
            }
        }
        if f.contains(vk::FormatFeatureFlags::COLOR_ATTACHMENT) {
            usage |= vk::ImageUsageFlags::COLOR_ATTACHMENT;
        }
        if f.contains(vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT) {
            usage |= vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT;
        }
        let mut flags = if is_depth(key.format) { vk::ImageCreateFlags::empty() } else { vk::ImageCreateFlags::MUTABLE_FORMAT | vk::ImageCreateFlags::EXTENDED_USAGE };
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
                            crate::error!("images: upload/transition failed: {e}; GPU execution stops; checkpoints: {}", super::vk::checkpoint_report());
                        }
                    }
                }
                Err(e) => {
                    if !super::exec::DEVICE_LOST.swap(true, std::sync::atomic::Ordering::SeqCst) {
                        crate::error!("images: submit failed: {e}; GPU execution stops; checkpoints: {}", super::vk::checkpoint_report());
                    }
                }
            }
            c.device.reset_fences(&[self.fence]).ok();
            c.device.reset_command_buffer(self.cmd, vk::CommandBufferResetFlags::empty()).ok();
        }
    }

    /// Moves a new image to GENERAL and zero-fills it (depth: 0, stencil: 0).
    fn to_general(&mut self, image: vk::Image, format: vk::Format) {
        self.one_shot(|cmd| unsafe {
            let range = vk::ImageSubresourceRange { aspect_mask: aspects(format), base_mip_level: 0, level_count: vk::REMAINING_MIP_LEVELS, base_array_layer: 0, layer_count: vk::REMAINING_ARRAY_LAYERS };
            let b = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::GENERAL)
                .image(image)
                .subresource_range(range)
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
            let d = &ctx().device;
            d.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &[b]);
            if is_depth(format) {
                d.cmd_clear_depth_stencil_image(cmd, image, vk::ImageLayout::GENERAL, &vk::ClearDepthStencilValue { depth: 0.0, stencil: 0 }, &[range]);
            } else if !is_compressed(format) {
                d.cmd_clear_color_image(cmd, image, vk::ImageLayout::GENERAL, &vk::ClearColorValue::default(), &[range]);
            }
            let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
            d.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::ALL_COMMANDS, vk::DependencyFlags::empty(), &[mb], &[], &[]);
        });
    }

    /// Uploads all mips/layers of a guest texture (detiled) into `image`.
    fn upload(&mut self, image: vk::Image, key: ImageKey, t: &Tsharp) -> Result<(u64, u64), String> {
        let t0 = std::time::Instant::now();
        let r = self.upload_inner(image, key, t);
        if let Ok((_, n)) = &r {
            self.upload_bytes += n;
        }
        self.upload_ns += t0.elapsed().as_nanos() as u64;
        r
    }

    fn upload_inner(&mut self, image: vk::Image, key: ImageKey, t: &Tsharp) -> Result<(u64, u64), String> {
        let first = !self.map.contains_key(&key);
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
        let base = t.base;
        self.one_shot(|cmd| unsafe {
            super::vk::checkpoint(cmd, super::vk::mark::UPLOAD, base);
            // A first upload starts from UNDEFINED; a re-upload keeps the
            // layout and waits for earlier work using the image.
            let b = vk::ImageMemoryBarrier::default()
                .old_layout(if first { vk::ImageLayout::UNDEFINED } else { vk::ImageLayout::GENERAL })
                .new_layout(vk::ImageLayout::GENERAL)
                .image(image)
                .subresource_range(vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: vk::REMAINING_MIP_LEVELS, base_array_layer: 0, layer_count: vk::REMAINING_ARRAY_LAYERS })
                .src_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
            c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &[b]);
            c.device.cmd_copy_buffer_to_image(cmd, buf, image, vk::ImageLayout::GENERAL, &regions);
            let a = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
            c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::ALL_COMMANDS, vk::DependencyFlags::empty(), &[a], &[], &[]);
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

    /// A rendered depth surface a texture of `key` reads (sampling only):
    /// same address and size, R32 float over D32, R16 unorm over D16, R8 uint
    /// over the stencil plane. Render targets are not written back, so guest
    /// memory under them is stale (a reversed-Z depth of 0 is the far plane,
    /// stencil 0 no material: the lighting saw only sky).
    fn depth_alias(&self, key: &ImageKey) -> Option<ImageKey> {
        use vk::Format as F;
        if key.format == F::R8_UINT {
            return self.stencil_of.get(&key.base).filter(|k| k.width == key.width && k.height == key.height && k.layers >= key.layers && self.map.contains_key(k)).copied();
        }
        let depths: &[vk::Format] = match key.format {
            F::R32_SFLOAT => &[F::D32_SFLOAT, F::D32_SFLOAT_S8_UINT],
            F::R16_UNORM => &[F::D16_UNORM, F::D16_UNORM_S8_UINT],
            _ => return None,
        };
        self.map
            .iter()
            .filter(|(k, i)| {
                i.gpu_written && k.base == key.base && k.width == key.width && k.height == key.height && k.layers >= key.layers && !k.ty3d && !key.ty3d && depths.contains(&k.format)
            })
            .map(|(k, _)| *k)
            .max_by_key(|k| k.layers)
    }

    /// The cached image a texture of `key` uses, and how it was found. A
    /// GPU-written image of the same address and size (other format,
    /// rendered depth or stencil) holds newer data than guest memory: render
    /// targets are not written back, so an upload made before the surface
    /// was first rendered (e.g. the depth buffer read at boot) is stale.
    fn lookup(&self, key: &ImageKey, storage: bool) -> Option<(ImageKey, &'static str)> {
        let exact = self.map.get(key).map(|i| i.gpu_written);
        if exact == Some(true) {
            return Some((*key, "exact"));
        }
        if self.gpu_bases.contains(&key.base) {
            if let Some(k) = self.alias(key, true) {
                return Some((k, "alias"));
            }
            if !storage {
                if let Some(k) = self.depth_alias(key) {
                    return Some((k, if key.format == vk::Format::R8_UINT { "stencil of" } else { "depth of" }));
                }
            }
        }
        exact.map(|_| (*key, "exact"))
    }

    /// Notes writes (GPU buffer writes, CP DMA) that cover the metadata of a
    /// rendered surface: fast clears, resolved at the surface's next use.
    pub fn note_meta_writes(&mut self, writes: &[(u64, u64)]) {
        if self.meta_of.is_empty() {
            return;
        }
        for &(b, n) in writes {
            for (&meta, &(key, _)) in &self.meta_of {
                if meta >= b && meta < b.saturating_add(n) {
                    self.pending_clears.insert(key, meta);
                }
            }
        }
    }

    /// Whether any of `keys` has a fast clear pending.
    pub fn clears_pending(&self, keys: &[ImageKey]) -> bool {
        !self.pending_clears.is_empty() && keys.iter().any(|k| self.pending_clears.contains_key(k))
    }

    /// Takes the pending fast clears of `keys`.
    pub fn take_clears(&mut self, keys: &[ImageKey]) -> Vec<FastClear> {
        let mut out = Vec::new();
        for k in keys {
            let Some(meta) = self.pending_clears.remove(k) else { continue };
            let (Some(img), Some(&(_, clear))) = (self.map.get(k), self.meta_of.get(&meta)) else { continue };
            if out.iter().any(|c: &FastClear| c.key == *k) {
                continue;
            }
            out.push(FastClear { key: *k, image: img.image, meta, clear });
        }
        out
    }

    /// The cached image a T# would use now (fast-clear checks; no side effects).
    pub fn resolved_key(&self, t: &Tsharp, storage: bool) -> Option<ImageKey> {
        self.lookup(&Self::key_of(t)?, storage).map(|(k, _)| k)
    }

    /// Image for a T#: an existing GPU image at that address with a
    /// compatible size, or a texture uploaded from guest memory.
    fn image_for(&mut self, t: &Tsharp, frame: u64, storage: bool) -> Result<ImageKey, String> {
        let key = Self::key_of(t).ok_or_else(|| format!("image format {} unsupported", t.format))?;
        if let Some((k, how)) = self.lookup(&key, storage) {
            if how == "exact" && !self.map[&k].gpu_written {
                self.refresh(key, t, frame)?;
            }
            return Ok(k);
        }
        let image = self.create(key)?;
        let source = match self.upload(image, key, t) {
            Ok(s) => Some(s),
            Err(e) => {
                self.report(format!("texture {:#x} ({}x{} format {} sw_mode {}): {e}; contents undefined", t.base, t.width, t.height, t.format, t.sw_mode));
                self.to_general(image, key.format);
                None
            }
        };
        let _ = frame;
        // CPU writes to the source re-upload it (write tracking).
        let region = source.and_then(|(b, n)| super::track::watch(b, n));
        if source.is_some() && region.is_none() {
            self.report(format!("texture {:#x} is not write-tracked (read-only or unmapped pages): later CPU updates are not seen", t.base));
        }
        let hashes = source.map(|(b, n)| all_hashes(b, n)).unwrap_or([0; SLICES]);
        self.map.insert(key, Image { image, key, views: HashMap::new(), gpu_written: false, source: region, hashes, checked_frame: frame });
        Ok(key)
    }

    /// Re-uploads a CPU-sourced texture whose guest bytes were written
    /// since its upload (write tracking: CPU faults, file reads, GPU writes).
    fn refresh(&mut self, key: ImageKey, t: &Tsharp, frame: u64) -> Result<(), String> {
        let img = &self.map[&key];
        if img.gpu_written {
            return Ok(());
        }
        let (image, Some(region)) = (img.image, img.source) else { return Ok(()) };
        let (b, n) = super::track::range(region).unwrap_or((0, 0));
        if super::track::take_gpu_written(region) {
            // GPU buffer writes (e.g. per-frame clears): re-upload only if
            // the contents changed.
            let h = all_hashes(b, n);
            if h != self.map[&key].hashes {
                self.reuploads += 1;
                self.report(format!("texture {:#x} {}x{} re-uploaded (GPU write)", t.base, t.width, t.height));
                self.upload(image, key, t)?;
                self.map.get_mut(&key).unwrap().hashes = h;
            }
        }
        if !super::track::protected() && self.map[&key].checked_frame != frame {
            // CPU writes: one slice of the sampled hash per frame.
            let img = self.map.get_mut(&key).unwrap();
            img.checked_frame = frame;
            let s = (frame as usize) % SLICES;
            if slice_hash(b, n, s) != img.hashes[s] {
                self.reuploads += 1;
                self.report(format!("texture {:#x} {}x{} re-uploaded (CPU write)", t.base, t.width, t.height));
                self.upload(image, key, t)?;
                self.map.get_mut(&key).unwrap().hashes = all_hashes(b, n);
            }
        }
        if let Some(why) = super::track::take_dirty(region) {
            // Re-armed first: writes during the upload mark it dirty again.
            super::track::rearm(region);
            self.reuploads += 1;
            if self.reuploads <= 12 {
                crate::info!(
                    "images: re-upload of {:#x} ({}x{}x{} format {} type {} sw_mode {} levels {} key {:?}): {why:?}",
                    t.base, t.width, t.height, t.depth, t.format, t.ty, t.sw_mode, t.max_mip + 1, key
                );
            }
            let msg = format!("texture {:#x} {}x{} re-uploaded ({})", t.base, t.width, t.height, match why {
                super::track::Why::Fault(_) => "CPU write",
                super::track::Why::HostWrite(_) => "file read or import",
                super::track::Why::GpuWrite(..) => "GPU write",
                super::track::Why::Remap(..) => "mapping change",
                super::track::Why::Unmapped => "unmapped",
                super::track::Why::None => "?",
            });
            self.report(msg);
            self.upload(image, key, t)?;
            self.map.get_mut(&key).unwrap().hashes = all_hashes(b, n);
        }
        Ok(())
    }

    /// An existing image at `key`'s address with the same size that can be
    /// viewed in `key`'s format (optionally only GPU-written ones).
    fn alias(&self, key: &ImageKey, gpu_written_only: bool) -> Option<ImageKey> {
        self.map
            .iter()
            .filter(|(k, i)| {
                (!gpu_written_only || i.gpu_written)
                    && k.base == key.base
                    && k.width == key.width
                    && k.height == key.height
                    && k.layers >= key.layers
                    && k.levels >= key.levels
                    && k.ty3d == key.ty3d
                    && view_compatible(k.format, key.format)
            })
            .map(|(k, _)| *k)
            .max_by_key(|k| (k.format == key.format, k.levels, k.layers))
    }

    /// Image for a render target or depth surface: the image of that exact
    /// surface, an alias of it (same address and size, compatible format), or
    /// a new zero-filled image. Marked GPU-written.
    fn surface_image(&mut self, s: &Surface) -> Result<ImageKey, String> {
        let key = ImageKey { base: s.base, format: s.format, width: s.width, height: s.height, layers: s.layers.max(1), levels: s.levels.max(1), ty3d: s.ty3d, cube: false };
        let k = if self.map.contains_key(&key) {
            key
        } else if let Some(k) = self.alias(&key, false) {
            k
        } else {
            let image = self.create(key)?;
            self.to_general(image, key.format);
            self.map.insert(key, Image { image, key, views: HashMap::new(), gpu_written: true, source: None, hashes: [0; SLICES], checked_frame: 0 });
            self.gpu_bases.insert(key.base);
            crate::info!("images: new render target {:#x} {}x{} {:?} at flip {}", key.base, key.width, key.height, key.format, super::FRAMES.load(std::sync::atomic::Ordering::Relaxed));
            // DS_CAPTURE_NEW_RT=<hex base>: capture the frame after a render
            // target at or above that address is created.
            static TRIGGER: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
            if let Some(t) = *TRIGGER.get_or_init(|| std::env::var("DS_CAPTURE_NEW_RT").ok().and_then(|v| u64::from_str_radix(v.trim_start_matches("0x"), 16).ok())) {
                if key.base >= t {
                    super::capture::arm_next();
                }
            }
            key
        };
        let img = self.map.get_mut(&k).unwrap();
        if !img.gpu_written {
            crate::info!("images: {:#x} {}x{} {:?} becomes a render target", k.base, k.width, k.height, k.format);
        }
        // GPU-owned from now on: guest memory is no longer its source.
        img.gpu_written = true;
        self.gpu_bases.insert(k.base);
        if s.stencil_base != 0 && has_stencil(k.format) {
            self.stencil_of.insert(s.stencil_base, k);
            self.gpu_bases.insert(s.stencil_base);
        }
        if s.meta != 0 {
            self.meta_of.insert(s.meta, (k, s.clear));
        }
        img.source = None;
        Ok(k)
    }

    /// Attachment for a color or depth/stencil surface (one mip level, one layer).
    pub fn target(&mut self, s: &Surface) -> Result<Target, String> {
        let k = self.surface_image(s)?;
        let img = self.map.get_mut(&k).unwrap();
        let mip = s.mip.min(img.key.levels - 1);
        let layer = if img.key.ty3d { 0 } else { s.slice.min(img.key.layers - 1) };
        let vkey = (s.format, vk::ImageViewType::TYPE_2D, mip, 1, layer, 1, [4u8, 5, 6, 7], 2u8);
        let view = match img.views.get(&vkey) {
            Some(&v) => v,
            None => {
                let depth = is_depth(s.format);
                let ci = vk::ImageViewCreateInfo::default()
                    .image(img.image)
                    .view_type(if img.key.ty3d { vk::ImageViewType::TYPE_2D_ARRAY } else { vk::ImageViewType::TYPE_2D })
                    .format(s.format)
                    .subresource_range(vk::ImageSubresourceRange { aspect_mask: aspects(s.format), base_mip_level: mip, level_count: 1, base_array_layer: layer, layer_count: 1 });
                let mut usage = vk::ImageViewUsageCreateInfo::default().usage(if depth { vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT } else { vk::ImageUsageFlags::COLOR_ATTACHMENT });
                let ci = if depth { ci } else { ci.push_next(&mut usage) };
                let v = unsafe { ctx().device.create_image_view(&ci, None) }.map_err(|e| format!("attachment view {vkey:?}: {e}"))?;
                img.views.insert(vkey, v);
                v
            }
        };
        Ok(Target { key: k, image: img.image, view, format: s.format, width: (img.key.width >> mip).max(1), height: (img.key.height >> mip).max(1) })
    }

    /// The GPU image last rendered at `base` (display buffers).
    pub fn rendered_at(&self, base: u64) -> Option<(vk::Image, vk::Format, u32, u32)> {
        self.map
            .values()
            .filter(|i| i.key.base == base && i.gpu_written && !is_depth(i.key.format))
            .max_by_key(|i| i.key.width as u64 * i.key.height as u64)
            .map(|i| (i.image, i.key.format, i.key.width, i.key.height))
    }

    /// Reads mip 0 / layer 0 of a color image back (tightly packed texels).
    pub fn read_back(&mut self, image: vk::Image, format: vk::Format, width: u32, height: u32) -> Result<Vec<u8>, String> {
        if texel_bytes(format) == 0 || is_compressed(format) || is_depth(format) {
            return Err(format!("read back of {format:?} unsupported"));
        }
        self.read_back_aspect(image, format, vk::ImageAspectFlags::COLOR, width, height)
    }

    /// Reads mip 0 / layer 0 of one aspect back: color texels, depth as 32-bit
    /// float (D32 formats) or 16-bit unorm (D16), stencil as bytes.
    pub fn read_back_aspect(&mut self, image: vk::Image, format: vk::Format, aspect: vk::ImageAspectFlags, width: u32, height: u32) -> Result<Vec<u8>, String> {
        let bpp = if aspect == vk::ImageAspectFlags::STENCIL {
            1
        } else if aspect == vk::ImageAspectFlags::DEPTH {
            if matches!(format, vk::Format::D16_UNORM | vk::Format::D16_UNORM_S8_UINT) { 2 } else { 4 }
        } else {
            texel_bytes(format) as u64
        };
        let size = bpp * width as u64 * height as u64;
        let c = ctx();
        let (buf, mem, ptr) = c.host_buffer(size, vk::BufferUsageFlags::TRANSFER_DST)?;
        self.one_shot(|cmd| unsafe {
            let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::MEMORY_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_READ);
            c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[mb], &[], &[]);
            let region = vk::BufferImageCopy::default()
                .image_subresource(vk::ImageSubresourceLayers { aspect_mask: aspect, mip_level: 0, base_array_layer: 0, layer_count: 1 })
                .image_extent(vk::Extent3D { width, height, depth: 1 });
            c.device.cmd_copy_image_to_buffer(cmd, image, vk::ImageLayout::GENERAL, buf, &[region]);
            let hb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::HOST_READ);
            c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::HOST, vk::DependencyFlags::empty(), &[hb], &[], &[]);
        });
        let out = unsafe { std::slice::from_raw_parts(ptr, size as usize).to_vec() };
        unsafe {
            c.device.destroy_buffer(buf, None);
            c.device.unmap_memory(mem);
            c.device.free_memory(mem, None);
        }
        Ok(out)
    }

    /// Writes the GPU-written color images (mip 0, layer 0) to `dir` as PNG
    /// files named <address>_<width>x<height>_<format>.png (frame captures).
    pub fn dump_targets(&mut self, dir: &std::path::Path) -> usize {
        let mut keys: Vec<ImageKey> = self.map.values().filter(|i| i.gpu_written && !is_depth(i.key.format)).map(|i| i.key).collect();
        keys.sort_by_key(|k| (k.base, k.format.as_raw()));
        let _ = std::fs::create_dir_all(dir);
        let mut n = 0;
        let mut stats = String::new();
        for k in keys.into_iter().take(256) {
            let image = self.map[&k].image;
            let Ok(raw) = self.read_back(image, k.format, k.width, k.height) else { continue };
            let name = format!("{:010x}_{}x{}_{:?}", k.base, k.width, k.height, k.format);
            if let Some(s) = float_stats(k.format, &raw) {
                stats.push_str(&format!("{name}: {s}\n"));
            }
            let Some(rgba) = to_rgba8(k.format, &raw) else { continue };
            if std::fs::write(dir.join(format!("{name}.png")), super::png::encode(k.width, k.height, &rgba)).is_ok() {
                n += 1;
            }
        }
        // Depth surfaces: depth statistics and the stencil values present.
        let mut dkeys: Vec<ImageKey> = self.map.values().filter(|i| i.gpu_written && is_depth(i.key.format)).map(|i| i.key).collect();
        dkeys.sort_by_key(|k| k.base);
        for k in dkeys.into_iter().take(64) {
            let image = self.map[&k].image;
            let name = format!("{:010x}_{}x{}_{:?}", k.base, k.width, k.height, k.format);
            if aspects(k.format).contains(vk::ImageAspectFlags::DEPTH) {
                if let Ok(raw) = self.read_back_aspect(image, k.format, vk::ImageAspectFlags::DEPTH, k.width, k.height) {
                    if raw.len() as u64 == 4 * k.width as u64 * k.height as u64 {
                        if let Some(s) = float_stats(vk::Format::R32_SFLOAT, &raw) {
                            stats.push_str(&format!("{name} depth: {s}\n"));
                        }
                        // Depth as gray, scaled to the largest value.
                        let d: Vec<f32> = raw.chunks_exact(4).map(|p| f32::from_le_bytes([p[0], p[1], p[2], p[3]])).collect();
                        let max = d.iter().copied().filter(|v| v.is_finite()).fold(0f32, f32::max).max(f32::MIN_POSITIVE);
                        let gray: Vec<u8> = d.iter().flat_map(|&v| {
                            let g = ((v / max).clamp(0.0, 1.0) * 255.0) as u8;
                            [g, g, g, 255]
                        }).collect();
                        let _ = std::fs::write(dir.join(format!("{name}_depth.png")), super::png::encode(k.width, k.height, &gray));
                    }
                }
            }
            if has_stencil(k.format) {
                if let Ok(raw) = self.read_back_aspect(image, k.format, vk::ImageAspectFlags::STENCIL, k.width, k.height) {
                    let mut hist = [0u64; 256];
                    for &v in &raw {
                        hist[v as usize] += 1;
                    }
                    let mut top: Vec<(u64, usize)> = hist.iter().enumerate().filter(|&(_, &c)| c > 0).map(|(v, &c)| (c, v)).collect();
                    top.sort_unstable_by(|a, b| b.cmp(a));
                    let list: Vec<String> = top.iter().take(12).map(|(c, v)| format!("{v:#04x}: {c}")).collect();
                    stats.push_str(&format!("{name} stencil: {}\n", list.join(", ")));
                    let gray: Vec<u8> = raw.iter().flat_map(|&v| [v, v, v, 255]).collect();
                    let _ = std::fs::write(dir.join(format!("{name}_stencil.png")), super::png::encode(k.width, k.height, &gray));
                }
            }
        }
        let _ = std::fs::write(dir.join("stats.txt"), stats);
        n
    }

    /// Whether a T# resolves to an image the GPU owns (written by GPU work;
    /// never re-uploaded from guest memory).
    pub fn gpu_owned(&self, t: &Tsharp) -> bool {
        let Some(key) = Self::key_of(t) else { return false };
        match self.map.get(&key) {
            Some(i) => i.gpu_written,
            None => self.alias(&key, true).is_some(),
        }
    }

    /// The cached image a T# resolves to (mip 0, layer 0): raw texels in its
    /// Vulkan format, width, height and bytes per texel (diagnostics).
    pub fn snapshot_raw(&mut self, t: &Tsharp) -> Result<(Vec<u8>, u32, u32, u32), String> {
        let want = Self::key_of(t).ok_or("unsupported format")?;
        let key = match self.lookup(&want, false) {
            Some((k, _)) => k,
            None => self.alias(&want, false).ok_or("not cached")?,
        };
        let image = self.map[&key].image;
        if is_depth(key.format) {
            // Rendered depth or stencil read as a texture (R32F / R16 / R8 uint).
            let aspect = if want.format == vk::Format::R8_UINT { vk::ImageAspectFlags::STENCIL } else { vk::ImageAspectFlags::DEPTH };
            let raw = self.read_back_aspect(image, key.format, aspect, key.width, key.height)?;
            let texel = (raw.len() as u64 / (key.width as u64 * key.height as u64).max(1)) as u32;
            return Ok((raw, key.width, key.height, texel));
        }
        let raw = self.read_back(image, key.format, key.width, key.height)?;
        Ok((raw, key.width, key.height, texel_bytes(key.format)))
    }

    /// The cached image a T# resolves to (mip 0, layer 0) as RGBA8, and the
    /// statistics of float formats (captures).
    pub fn snapshot(&mut self, t: &Tsharp) -> Result<(Vec<u8>, u32, u32, Option<String>), String> {
        let key = Self::key_of(t).ok_or("unsupported format")?;
        let key = if self.map.contains_key(&key) { key } else { self.alias(&key, false).ok_or("not cached")? };
        let image = self.map[&key].image;
        let raw = self.read_back(image, key.format, key.width, key.height)?;
        Ok((to_rgba8(key.format, &raw).ok_or("format")?, key.width, key.height, float_stats(key.format, &raw)))
    }

    /// Which cached image a T# resolves to now (frame captures).
    pub fn describe(&self, t: &Tsharp) -> String {
        let Some(key) = Self::key_of(t) else { return "unsupported format".into() };
        let all: Vec<String> = self
            .map
            .values()
            .filter(|i| i.key.base == t.base)
            .map(|i| format!("{}x{}x{} {:?} l{} m{}{}{}", i.key.width, i.key.height, i.key.layers, i.key.format, i.key.layers, i.key.levels, if i.gpu_written { " gpu" } else { "" }, if i.key == key { " =key" } else { "" }))
            .collect();
        let pick = match self.lookup(&key, false) {
            Some((k, "exact")) => format!("exact{}", if self.map[&k].gpu_written { "" } else { " (uploaded)" }),
            Some((k, how)) => format!("{how} {:#x} {:?} l{} m{}", k.base, k.format, k.layers, k.levels),
            None => "new upload".into(),
        };
        format!("{pick}; images at base: [{}]", all.join(", "))
    }

    /// View of a T# (format, type, levels/layers, swizzle) for sampling or storage.
    pub fn view(&mut self, t: &Tsharp, storage: bool, frame: u64) -> Result<(vk::ImageView, vk::Image), String> {
        let key = self.image_for(t, frame, storage)?;
        let req = vk_format(t.format).unwrap();
        let depth = is_depth(key.format);
        let stencil = depth && req == vk::Format::R8_UINT;
        if depth && storage {
            return Err(format!("texture {:#x} aliases a depth surface (storage access not supported)", t.base));
        }
        // Rendered depth is sampled through a view of its depth aspect (the
        // value arrives in R, like the 32-bit float texture the title reads).
        let fmt = if depth { key.format } else if storage { storage_format(req) } else { req };
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
        let vkey = (fmt, vt, t.base_level, levels.saturating_sub(t.base_level).max(1), base_layer, layer_count, swz, storage as u8 | (stencil as u8) << 1);
        let img = self.map.get_mut(&key).unwrap();
        if storage {
            if !img.gpu_written {
                crate::info!("images: {:#x} {}x{} {:?} becomes a storage image (was uploaded: {})", key.base, key.width, key.height, key.format, img.source.is_some());
            }
            img.gpu_written = true;
            img.source = None;
            self.gpu_bases.insert(key.base);
        }
        let img = self.map.get_mut(&key).unwrap();
        if let Some(&v) = img.views.get(&vkey) {
            return Ok((v, img.image));
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
                aspect_mask: if stencil { vk::ImageAspectFlags::STENCIL } else if depth { vk::ImageAspectFlags::DEPTH } else { vk::ImageAspectFlags::COLOR },
                base_mip_level: base_level,
                level_count: if storage { 1 } else { (levels.saturating_sub(base_level)).clamp(1, levels_avail - base_level) },
                base_array_layer: base_layer,
                layer_count,
            });
        let mut usage = vk::ImageViewUsageCreateInfo::default().usage(if storage { vk::ImageUsageFlags::STORAGE } else { vk::ImageUsageFlags::SAMPLED });
        let ci = ci.push_next(&mut usage);
        let v = unsafe { ctx().device.create_image_view(&ci, None) }.map_err(|e| format!("image view {vkey:?}: {e}"))?;
        img.views.insert(vkey, v);
        Ok((v, img.image))
    }

    /// Sampler of an S#; `compare`: used by depth-compare instructions (the
    /// S# compare function applies only to those).
    pub fn sampler(&mut self, s: [u32; 4], compare_use: bool) -> Result<vk::Sampler, String> {
        if let Some(&v) = self.samplers.get(&(s, compare_use)) {
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
            .compare_enable(compare_use)
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
        self.samplers.insert((s, compare_use), v);
        Ok(v)
    }

    pub fn count(&self) -> usize {
        self.map.len()
    }

    /// One status line (hang reports).
    pub fn status(&self) -> String {
        format!("    images: {} cached, {} uploads ({} re-uploads; {} MiB, {} ms detiling and copying)\n", self.map.len(), self.uploads, self.reuploads, self.upload_bytes >> 20, self.upload_ns / 1_000_000)
    }
}
