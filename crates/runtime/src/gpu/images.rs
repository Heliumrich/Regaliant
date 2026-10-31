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
use super::fx::Map as HashMap;

/// Changes whenever what `Images::lookup` finds for a key may change (images
/// added or removed, an image turning GPU-written, stencil links): the
/// per-thread lookup memos (`tls`) and the global write notes depend on it.
pub static LOOKUP_GEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
/// Changes when the GPU-written images (render targets, storage images,
/// their aliases at the same addresses) change: what `Images::depth_view`
/// answers for a texture depends only on them (texture keys have color
/// formats; uploads, which stream all the time, do not move it).
pub static GPU_IMAGES_GEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

fn gpu_images_changed() {
    GPU_IMAGES_GEN.fetch_add(1, std::sync::atomic::Ordering::Release);
}

/// Fast clears and overwritten images pending (published at the end of
/// every `with`): draws check them without the lock while both are 0.
static PENDING: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Whether any fast clear or overwrite reload is pending.
pub fn any_pending() -> bool {
    PENDING.load(std::sync::atomic::Ordering::Acquire) != 0
}

/// The images with a fast clear or an overwrite reload pending (published
/// by `with`), and its generation.
static PENDING_SET: Mutex<Option<std::sync::Arc<super::fx::Set<ImageKey>>>> = Mutex::new(None);
static PENDING_GEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

thread_local! {
    static PENDING_TLS: std::cell::RefCell<(u64, std::sync::Arc<super::fx::Set<ImageKey>>)> = std::cell::RefCell::new((u64::MAX, std::sync::Arc::new(Default::default())));
}

/// Whether any of `keys` has a fast clear or an overwrite reload pending
/// (`clears_pending || overwrites_pending` without the cache's lock).
pub fn pending_hit(keys: &[ImageKey]) -> bool {
    if !any_pending() {
        return false;
    }
    let g = PENDING_GEN.load(std::sync::atomic::Ordering::Acquire);
    PENDING_TLS.with(|t| {
        let mut t = t.borrow_mut();
        if t.0 != g {
            *t = (g, PENDING_SET.lock().clone().unwrap_or_default());
        }
        keys.iter().any(|k| t.1.contains(k))
    })
}

type SurfaceKey = (u64, i32, u32, u32, u32, u32, bool, u32, u32, u64, u64, [u32; 2]);

fn surface_key(s: &Surface) -> SurfaceKey {
    (s.base, s.format.as_raw(), s.width, s.height, s.layers, s.levels, s.ty3d, s.mip, s.slice, s.stencil_base, s.meta, s.clear)
}

/// Per-thread memos of the lookups every draw repeats (the recording
/// threads of a parallel run all wanted the cache's one lock, which made
/// every draw step 1.5-2x slower with three workers). Keys, targets, depth
/// growth and GPU-written bases hold while LOOKUP_GEN does (`gen`); views
/// also need the frame and the tracker's generation (`view`'s refresh);
/// samplers are never dropped.
#[derive(Default)]
struct Tls {
    lookup_gen: u64,
    keys: HashMap<([u32; 8], bool), Option<ImageKey>>,
    targets: HashMap<SurfaceKey, Target>,
    grows: HashMap<(u64, u32, u32, i32, u32), Option<ImageKey>>,
    gpu_base: HashMap<u64, bool>,
    depth_feed: HashMap<u64, bool>,
    views: HashMap<[u32; 8], (u64, u64, u64, vk::ImageView, vk::Image, UseStamp)>,
    samplers: HashMap<([u32; 4], bool), vk::Sampler>,
}

thread_local! {
    static TLS: std::cell::RefCell<Tls> = std::cell::RefCell::new(Tls::default());
}

/// `f` on this thread's memos, those valid for one LOOKUP_GEN emptied if
/// it moved.
fn tls<R>(f: impl FnOnce(&mut Tls) -> R) -> R {
    TLS.with(|t| {
        let mut t = t.borrow_mut();
        let g = LOOKUP_GEN.load(std::sync::atomic::Ordering::Acquire);
        if t.lookup_gen != g {
            t.lookup_gen = g;
            t.keys.clear();
            t.targets.clear();
            t.grows.clear();
            t.gpu_base.clear();
            t.depth_feed.clear();
        }
        f(&mut t)
    })
}

/// Memo statistics (DS_PROFILE_GPU; `profile report`): view lookups [hit,
/// not memoized, other frame, image-cache generation moved, write-tracking
/// generation moved], key lookups [hit, miss], and the generations at the
/// last `profile reset` [lookup, tracking].
static MEMO_STATS: [std::sync::atomic::AtomicU64; 7] = [const { std::sync::atomic::AtomicU64::new(0) }; 7];
static MEMO_GENS: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];
static TRACK_WHY_AT_RESET: [std::sync::atomic::AtomicU64; 6] = [const { std::sync::atomic::AtomicU64::new(0) }; 6];

fn memo_note(k: usize) {
    if super::exec::profiling_on() {
        MEMO_STATS[k].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Clears the memo statistics (`profile reset`).
pub fn reset_memo_stats() {
    for c in &MEMO_STATS {
        c.store(0, std::sync::atomic::Ordering::Relaxed);
    }
    MEMO_GENS[0].store(LOOKUP_GEN.load(std::sync::atomic::Ordering::Relaxed), std::sync::atomic::Ordering::Relaxed);
    MEMO_GENS[1].store(super::track::GEN.load(std::sync::atomic::Ordering::Relaxed), std::sync::atomic::Ordering::Relaxed);
    for (r, c) in TRACK_WHY_AT_RESET.iter().zip(super::track::GEN_WHY.iter()) {
        r.store(c.load(std::sync::atomic::Ordering::Relaxed), std::sync::atomic::Ordering::Relaxed);
    }
}

/// One profile line on the per-thread memos.
pub fn memo_stats_line() -> String {
    let s: Vec<u64> = MEMO_STATS.iter().map(|c| c.load(std::sync::atomic::Ordering::Relaxed)).collect();
    format!(
        "    image memos: views hit {}, missed {} (not memoized {}, other frame {}, image cache changed {}, write tracking changed {}); keys hit {}, missed {}; generations since reset: image cache {}, write tracking {} (forced dirty, page not writable, CPU write fault, host write, GPU write, remap: {:?})\n",
        s[0],
        s[1] + s[2] + s[3] + s[4],
        s[1],
        s[2],
        s[3],
        s[4],
        s[5],
        s[6],
        LOOKUP_GEN.load(std::sync::atomic::Ordering::Relaxed).saturating_sub(MEMO_GENS[0].load(std::sync::atomic::Ordering::Relaxed)),
        super::track::GEN.load(std::sync::atomic::Ordering::Relaxed).saturating_sub(MEMO_GENS[1].load(std::sync::atomic::Ordering::Relaxed)),
        super::track::GEN_WHY.iter().zip(TRACK_WHY_AT_RESET.iter()).map(|(c, r)| c.load(std::sync::atomic::Ordering::Relaxed).saturating_sub(r.load(std::sync::atomic::Ordering::Relaxed))).collect::<Vec<u64>>()
    )
}

/// `Images::view` from this thread's memo (sampled views only).
pub fn view_hit(t: &Tsharp, frame: u64) -> Option<(vk::ImageView, vk::Image)> {
    let track = super::track::GEN.load(std::sync::atomic::Ordering::Acquire);
    let lookup = LOOKUP_GEN.load(std::sync::atomic::Ordering::Acquire);
    let any_frame = scrubbing();
    let e = tls(|m| m.views.get(&t.raw).copied());
    let k = match e {
        None => 1,
        Some(e) if !(any_frame || e.0 == frame) => 2,
        Some(e) if e.1 != lookup => 3,
        Some(e) if e.2 != track => 4,
        Some(_) => 0,
    };
    memo_note(k);
    if k == 0 {
        e.map(|e| {
            e.5.stamp(frame);
            (e.3, e.4)
        })
    } else {
        None
    }
}

/// `Images::sampler` from this thread's memo.
pub fn sampler_hit(s: [u32; 4], compare: bool) -> Option<vk::Sampler> {
    tls(|m| m.samplers.get(&(s, compare)).copied())
}

/// `Images::resolved_key` from this thread's memo (None: not memoized).
pub fn key_hit(t: &Tsharp, storage: bool) -> Option<Option<ImageKey>> {
    let r = tls(|m| m.keys.get(&(t.raw, storage)).copied());
    memo_note(if r.is_some() { 5 } else { 6 });
    r
}

/// `Images::target`, from this thread's memo when it holds.
pub fn target(s: &Surface) -> Result<Target, String> {
    let k = surface_key(s);
    if let Some(t) = tls(|m| m.targets.get(&k).copied()) {
        t.used.stamp(super::FRAMES.load(std::sync::atomic::Ordering::Relaxed));
        return Ok(t);
    }
    let g = LOOKUP_GEN.load(std::sync::atomic::Ordering::Acquire);
    let t = with(|im| im.target(s))?;
    tls(|m| {
        if m.lookup_gen == g {
            m.targets.insert(k, t);
        }
    });
    Ok(t)
}

/// `Images::depth_to_grow`, from this thread's memo when it holds.
pub fn depth_to_grow(base: u64, width: u32, height: u32, format: vk::Format, layers: u32) -> Option<ImageKey> {
    let k = (base, width, height, format.as_raw(), layers);
    if let Some(r) = tls(|m| m.grows.get(&k).copied()) {
        return r;
    }
    let g = LOOKUP_GEN.load(std::sync::atomic::Ordering::Acquire);
    let r = with(|im| im.depth_to_grow(base, width, height, Images::depth_family(format), layers));
    tls(|m| {
        if m.lookup_gen == g {
            m.grows.insert(k, r);
        }
    });
    r
}

/// Whether block copies may be pending for `keys` (`block_copies_possible`
/// from this thread's memo of GPU-written bases).
pub fn block_copies_possible(keys: &[ImageKey]) -> bool {
    let cands: Vec<u64> = keys.iter().filter(|k| is_compressed(k.format) && !k.ty3d).map(|k| k.base).collect();
    if cands.is_empty() {
        return false;
    }
    let known: Option<bool> = tls(|m| {
        let mut any = false;
        for b in &cands {
            match m.gpu_base.get(b) {
                Some(&x) => any |= x,
                None => return None,
            }
        }
        Some(any)
    });
    if let Some(x) = known {
        return x;
    }
    let g = LOOKUP_GEN.load(std::sync::atomic::Ordering::Acquire);
    let flags: Vec<bool> = with(|im| cands.iter().map(|b| im.gpu_bases.contains(b)).collect());
    tls(|m| {
        if m.lookup_gen == g {
            for (b, f) in cands.iter().zip(&flags) {
                m.gpu_base.insert(*b, *f);
            }
        }
    });
    flags.iter().any(|&f| f)
}

/// The color format GPU work writes a depth surface's depth plane through
/// (the title's compute shaders store depth as R32 float texels under D32,
/// R16 unorm under D16).
fn depth_feed_format(depth: vk::Format) -> Option<vk::Format> {
    use vk::Format as F;
    match depth {
        F::D32_SFLOAT | F::D32_SFLOAT_S8_UINT => Some(F::R32_SFLOAT),
        F::D16_UNORM | F::D16_UNORM_S8_UINT => Some(F::R16_UNORM),
        _ => None,
    }
}

/// Whether depth copies may be pending for `keys` (`depth_feed_possible`
/// from this thread's memo of depth-feed bases).
pub fn depth_copies_possible(keys: &[ImageKey]) -> bool {
    let cands: Vec<u64> = keys.iter().filter(|k| depth_feed_format(k.format).is_some()).map(|k| k.base).collect();
    if cands.is_empty() {
        return false;
    }
    let known: Option<bool> = tls(|m| {
        let mut any = false;
        for b in &cands {
            match m.depth_feed.get(b) {
                Some(&x) => any |= x,
                None => return None,
            }
        }
        Some(any)
    });
    if let Some(x) = known {
        return x;
    }
    let g = LOOKUP_GEN.load(std::sync::atomic::Ordering::Acquire);
    let flags: Vec<bool> = with(|im| cands.iter().map(|b| im.depth_feeds.contains(b)).collect());
    tls(|m| {
        if m.lookup_gen == g {
            for (b, f) in cands.iter().zip(&flags) {
                m.depth_feed.insert(*b, *f);
            }
        }
    });
    flags.iter().any(|&f| f)
}

/// A depth image's depth plane to refresh from the color image GPU work
/// stored it through (`Images::take_depth_copies`).
pub struct DepthCopy {
    pub key: ImageKey,
    pub dst: vk::Image,
    pub src: vk::Image,
    pub width: u32,
    pub height: u32,
    pub layers: u32,
    /// Bytes per texel of the source (and of the buffer the copy passes through).
    pub texel: u32,
    pub seq: u64,
}

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
/// Bytes of guest memory a T#'s resource spans (all mips and layers), if
/// its format and swizzle mode are known.
pub fn guest_size(t: &Tsharp) -> Option<u64> {
    let key = Images::key_of(t)?;
    let el = elem_log2(t.format)?;
    if key.ty3d {
        return tiling::layout_3d(t.sw_mode, el, key.width, key.height, key.layers).map(|l| l.size());
    }
    tiling::layout(t.sw_mode, el, key.width, key.height, key.levels, is_block(t.format)).map(|l| l.size(key.layers))
}

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

/// GPU-owned images cleared right after a zero fill of their memory
/// (`Images::zero_filled`).
pub static ZERO_FILL_CLEARS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

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
        31 if m == 0.0 => f32::INFINITY,
        31 => f32::NAN,
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
        // Integer targets also as f32 bits (probed values, packed data).
        F::R32_SFLOAT | F::R32_UINT | F::R32_SINT => (1, false),
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
    /// The image's use stamp (memoized targets stamp it too).
    pub used: UseStamp,
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
    /// The metadata's first word, known from the fill that wrote it (no read
    /// back after the GPU work).
    pub known: Option<u32>,
}

/// Blocks of a block-compressed texture that GPU work stored through an
/// uncompressed view of its memory (one texel per block), to be copied in.
pub struct BlockCopy {
    pub key: ImageKey,
    /// The view of the whole chain (null when only the tail was written):
    /// its first `levels` mips.
    pub src: vk::Image,
    pub dst: vk::Image,
    /// Block grid of mip 0.
    pub width: u32,
    pub height: u32,
    pub levels: u32,
    pub layers: u32,
    /// The mip tail stored through a view of the tail block alone (one
    /// swizzle block at the surface's base, a single level the size of the
    /// block in elements: the title's transcoders write the small mips
    /// there): the view and, per tail mip, (mip, x, y, width, height) in
    /// blocks (where the mip sits inside the tail block).
    pub tail: Option<(vk::Image, Vec<(u32, u32, u32, u32, u32)>)>,
    seq: u64,
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
    /// Generation of its last binding as a storage image (see `Images::write_seq`).
    write_gen: u64,
    /// Block-compressed textures: generation of the block view last copied
    /// in (see `take_block_copies`).
    block_gen: u64,
    /// Guest bytes of its layout and the T# that described it exactly (0 /
    /// None: not seen through a T#): GPU buffer writes over GPU-owned images.
    span: u64,
    layout: Option<Tsharp>,
    /// GPU buffer writes or CP DMA covered all of its guest bytes since its
    /// last use: guest memory holds its contents (`reload_overwritten`).
    overwritten: bool,
    /// The shader that last bound it as a storage image (diagnosis:
    /// `fillimg cs=`).
    writer: u64,
    /// Its device memory and bytes, and the frame of its last use.
    mem: vk::DeviceMemory,
    bytes: u64,
    used: UseStamp,
}

/// An image's use stamp: the frame of its last lookup, memoized lookups
/// included, so that the least recently used images can go when video
/// memory runs short (`Images::evict`). Leaked (8 bytes per image ever
/// made): the memos read it without the cache's lock.
#[derive(Clone, Copy)]
pub struct UseStamp(&'static std::sync::atomic::AtomicU64);

impl UseStamp {
    fn new(frame: u64) -> Self {
        UseStamp(Box::leak(Box::new(std::sync::atomic::AtomicU64::new(frame))))
    }

    /// Notes a use in `frame` (one store per image and frame).
    #[inline]
    pub fn stamp(&self, frame: u64) {
        if self.0.load(std::sync::atomic::Ordering::Relaxed) < frame {
            self.0.store(frame, std::sync::atomic::Ordering::Relaxed);
        }
    }

    fn get(&self) -> u64 {
        self.0.load(std::sync::atomic::Ordering::Relaxed)
    }
}

impl PartialEq for UseStamp {
    fn eq(&self, o: &Self) -> bool {
        std::ptr::eq(self.0, o.0)
    }
}

impl Eq for UseStamp {}

impl std::fmt::Debug for UseStamp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "used at {}", self.get())
    }
}

/// An image the cache dropped (`Images::evict`, `grow_depth`), destroyed
/// once no batch can use it any more (`exec::bury_images`).
pub struct Evicted {
    pub image: vk::Image,
    pub mem: vk::DeviceMemory,
    pub views: Vec<vk::ImageView>,
}

/// Video memory of the cached images, bytes.
static IMAGE_BYTES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Images evicted: [count, bytes, eviction rounds].
static EVICTED: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];
/// An uploaded image is evicted under pressure only when unused for this
/// many frames. GPU-written ones, whose contents guest memory does not
/// hold, are never evicted: evicting those unused for 600 frames (data a
/// level load wrote and drew much later) lost 4-2's architecture in a
/// stress tour, and taking images over memory the title unmapped for dead
/// broke every level (the title maps its direct memory again with its
/// contents; 2026-10-19).
const EVICT_AGE: u64 = 120;
/// Video-memory budget checks at most this often.
const EVICT_CHECK_MS: u64 = 500;

/// When this process uses more video memory than `tune vram_high` percent of
/// the device-local budget, evicts the least recently used images until
/// about `vram_low` percent (completion thread, at most every
/// EVICT_CHECK_MS; skipped while the cache is busy).
pub fn maybe_evict() {
    use std::sync::atomic::Ordering::Relaxed;
    if !super::tune::image_evict() {
        return;
    }
    static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    static LAST: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let ms = START.get_or_init(std::time::Instant::now).elapsed().as_millis() as u64 + EVICT_CHECK_MS;
    if ms < LAST.load(Relaxed) + EVICT_CHECK_MS {
        return;
    }
    LAST.store(ms, Relaxed);
    let frame = super::FRAMES.load(Relaxed);
    let Some((used, budget)) = ctx().device_budget() else { return };
    if budget == 0 || used <= budget / 100 * super::tune::vram_high() {
        return;
    }
    let want = used.saturating_sub(budget / 100 * super::tune::vram_low());
    let Some(mut g) = IMAGES.try_lock() else { return };
    let Some(im) = g.as_mut() else { return };
    let out = im.evict(frame, want);
    // What holds the memory (the first reports).
    let largest = if EVICTED[2].load(Relaxed) < 5 { format!("; largest: {}", im.largest(6)) } else { String::new() };
    drop(g);
    if out.0.is_empty() {
        return;
    }
    let rounds = EVICTED[2].fetch_add(1, Relaxed) + 1;
    if rounds <= 20 || rounds % 100 == 0 {
        crate::info!(
            "images: video memory {} of budget {} MiB at flip {frame}: evicted {} textures ({} MiB), {} images cached ({} MiB){largest}",
            used >> 20,
            budget >> 20,
            out.0.len(),
            out.1 >> 20,
            out.2,
            IMAGE_BYTES.load(Relaxed) >> 20
        );
    }
    super::exec::bury_images(out.0);
}

/// Two GPU timestamps for an upload's command buffer: when the GPU reached
/// it and when it finished (`exec::batch_stamps`; null without them).
fn upload_stamps() -> vk::QueryPool {
    if !super::exec::batch_stamps() {
        return vk::QueryPool::null();
    }
    unsafe { ctx().device.create_query_pool(&vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(2), None) }.unwrap_or(vk::QueryPool::null())
}

/// Records an upload command buffer's start (k = 0, after a reset) or end
/// (k = 1) timestamp.
unsafe fn stamp(cmd: vk::CommandBuffer, qp: vk::QueryPool, k: u32) {
    if qp == vk::QueryPool::null() {
        return;
    }
    let d = &ctx().device;
    unsafe {
        if k == 0 {
            d.cmd_reset_query_pool(cmd, qp, 0, 2);
        }
        d.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, qp, k);
    }
}

/// Uploads whose GPU span went to the hitch reports.
static UPLOAD_SPANS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// A completed upload's span on the GPU, for the hitch reports' GPU busy
/// time (`hitch::gpu_batch`, queue "uploads").
fn note_upload_span(qp: vk::QueryPool, what: impl FnOnce() -> String) {
    if qp == vk::QueryPool::null() {
        return;
    }
    let mut t = [0u64; 2];
    if unsafe { ctx().device.get_query_pool_results(qp, 0, &mut t, vk::QueryResultFlags::TYPE_64) }.is_ok() {
        let n = UPLOAD_SPANS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        super::hitch::gpu_batch("uploads", n, t[0], t[1], 1, 0, what);
    }
}

/// A host-visible buffer texture data is uploaded from (`Images::staging`).
struct Staging {
    buf: vk::Buffer,
    mem: vk::DeviceMemory,
    ptr: *mut u8,
    cap: u64,
}

/// What an upload in flight holds until the GPU is done with it: its host
/// staging buffer, the video-memory buffers it used (GPU detiling, or the
/// copy queue's destination), and the copy queue's command buffer that
/// filled them.
struct Held {
    staging: Option<Staging>,
    dev: Vec<(vk::Buffer, vk::DeviceMemory, u64)>,
    copy_cmd: Option<vk::CommandBuffer>,
}

impl Held {
    fn staging(s: Staging) -> Held {
        Held { staging: Some(s), dev: Vec::new(), copy_cmd: None }
    }

    fn bytes(&self) -> u64 {
        self.staging.as_ref().map_or(0, |s| s.cap) + self.dev.iter().map(|b| b.2).sum::<u64>()
    }
}

/// Staging buffers kept for reuse: up to this many bytes, each of at most
/// STAGING_POOLED bytes (larger ones are made for their upload alone).
/// None: a host-visible staging buffer kept after its upload made the
/// driver refuse guest-memory imports thousands of times a session
/// (FINDINGS.md 2026-10-24; with 128 MiB kept, `upload_async` failed 53,832
/// imports in a tour, none kept 48 like the baseline).
const STAGING_KEEP: u64 = 0;
const STAGING_POOLED: u64 = 32 << 20;
/// Uploads in flight, and their staging bytes, before the next one waits
/// for the oldest. Unbounded bytes (up to 32 uploads of ~85 MB arrays, freed
/// only at a later upload) made guest-memory imports fail by the ten
/// thousand and preceded device losses (runs 20261013-083710, -084645).
const MAX_INFLIGHT: usize = 32;
/// The same for uploads with a copy-queue part (`tune upload_copy`, GPU
/// detiling): their bytes are bounded by DETILE_INFLIGHT; an arrival's
/// hundreds of small textures waited for the oldest at 32.
const MAX_INFLIGHT_COPY: usize = 256;
const MAX_INFLIGHT_BYTES: u64 = 128 << 20;
/// Uploads that waited for an earlier one (`MAX_INFLIGHT`), and staging
/// buffers made.
static UPLOAD_WAITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static STAGING_MADE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// `tune upload_batch`: uploads detiled on the CPU that would wait for their
/// transfer on the main queue are recorded into one command buffer they all
/// share, submitted ahead of the next batch (`flush_uploads`, from
/// `exec::submit_batch`) or of the cache's own next submission, and not
/// waited for; their staging buffers are freed once it completes. Every
/// batch goes to the one main queue (`tune alt_queue` aside), so a batch
/// that binds such a texture runs after its upload (the upload's last
/// barrier holds all later work), as does any batch an image buried after
/// it waits for.
struct Batched {
    pool: vk::CommandPool,
    /// The open command buffer, its fence, the staging it reads, its bytes.
    open: Option<(vk::CommandBuffer, vk::Fence, Vec<(vk::Buffer, vk::DeviceMemory)>, u64)>,
    inflight: std::collections::VecDeque<(vk::CommandBuffer, vk::Fence, Vec<(vk::Buffer, vk::DeviceMemory)>, u64)>,
    spare: Vec<(vk::CommandBuffer, vk::Fence)>,
}

static BATCHED: Mutex<Option<Batched>> = Mutex::new(None);
/// The open command buffer and those in flight (`flush_uploads` returns at
/// once at 0).
static BATCHED_LIVE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Uploads recorded batched, command buffers submitted, waits for one in
/// flight (`BATCHED_MAX_BYTES`).
static BATCHED_STATS: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];
/// Staging bytes recorded or in flight at most before an upload waits for
/// the oldest command buffer.
const BATCHED_MAX_BYTES: u64 = 256 << 20;
/// Staging bytes the open command buffer reads at most before it is
/// submitted at once (no batch submitted meanwhile: level loads).
const BATCHED_OPEN_BYTES: u64 = 64 << 20;

fn free_batched(held: Vec<(vk::Buffer, vk::DeviceMemory)>) {
    let c = ctx();
    for (buf, mem) in held {
        unsafe {
            c.device.destroy_buffer(buf, None);
            c.device.unmap_memory(mem);
            c.device.free_memory(mem, None);
        }
    }
}

impl Batched {
    /// Submits the open command buffer, if any, to the main queue.
    fn submit_open(&mut self) {
        let Some((cmd, fence, held, bytes)) = self.open.take() else { return };
        let c = ctx();
        unsafe { c.device.end_command_buffer(cmd).ok() };
        match c.submit(cmd, &[], &[], fence) {
            Ok(()) => {
                self.inflight.push_back((cmd, fence, held, bytes));
                BATCHED_STATS[1].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            Err(e) => {
                if !super::exec::DEVICE_LOST.swap(true, std::sync::atomic::Ordering::SeqCst) {
                    crate::error!("images: batched uploads' submit failed: {e}; GPU execution stops; checkpoints: {}", super::vk::checkpoint_report());
                }
                free_batched(held);
                unsafe { c.device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty()).ok() };
                self.spare.push((cmd, fence));
                BATCHED_LIVE.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
    }

    /// Frees what completed command buffers held, waiting for the oldest
    /// while more than `max_bytes` are recorded or in flight.
    fn reap(&mut self, max_bytes: u64) {
        let c = ctx();
        while let Some(&(_, fence, _, _)) = self.inflight.front() {
            let done = unsafe { c.device.get_fence_status(fence) }.unwrap_or(true);
            if !done {
                let bytes: u64 = self.inflight.iter().map(|x| x.3).sum::<u64>() + self.open.as_ref().map_or(0, |o| o.3);
                if bytes <= max_bytes {
                    break;
                }
                BATCHED_STATS[2].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let t0 = std::time::Instant::now();
                unsafe { c.device.wait_for_fences(&[fence], true, 10_000_000_000).ok() };
                super::hitch::note_since(super::hitch::Kind::GpuWait, t0, || format!("batched texture uploads in flight ({} MiB)", bytes >> 20));
            }
            let Some((cmd, fence, held, _)) = self.inflight.pop_front() else { break };
            free_batched(held);
            unsafe {
                c.device.reset_fences(&[fence]).ok();
                c.device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty()).ok();
            }
            self.spare.push((cmd, fence));
            BATCHED_LIVE.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

/// Records an upload (`f`) into the shared open command buffer; its staging
/// (buffer, memory) is freed once that completes.
fn record_batched(staging: (vk::Buffer, vk::DeviceMemory), bytes: u64, f: &dyn Fn(vk::CommandBuffer)) -> Result<(), String> {
    let c = ctx();
    let mut g = BATCHED.lock();
    if g.is_none() {
        let pool = unsafe { c.device.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(c.queue_family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER), None) }.map_err(|e| format!("batched uploads' pool: {e}"))?;
        *g = Some(Batched { pool, open: None, inflight: Default::default(), spare: Vec::new() });
    }
    let Some(b) = g.as_mut() else { return Err("no batched uploads".into()) };
    b.reap(BATCHED_MAX_BYTES.saturating_sub(bytes));
    if b.open.is_none() {
        let (cmd, fence) = match b.spare.pop() {
            Some(x) => x,
            None => unsafe {
                let cmd = c.device.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(b.pool).command_buffer_count(1)).map_err(|e| format!("batched uploads' command buffer: {e}"))?[0];
                (cmd, c.device.create_fence(&vk::FenceCreateInfo::default(), None).map_err(|e| format!("batched uploads' fence: {e}"))?)
            },
        };
        unsafe { c.device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).ok() };
        b.open = Some((cmd, fence, Vec::new(), 0));
        BATCHED_LIVE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    let Some(o) = b.open.as_mut() else { return Err("no open command buffer".into()) };
    f(o.0);
    o.2.push(staging);
    o.3 += bytes;
    BATCHED_STATS[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if o.3 >= BATCHED_OPEN_BYTES {
        b.submit_open();
    }
    Ok(())
}

/// Submits the shared upload command buffer to the main queue if one is
/// open (`tune upload_batch`: ahead of every batch, `exec::submit_batch`,
/// and of the cache's own submissions), and frees what completed ones held.
pub fn flush_uploads() {
    if BATCHED_LIVE.load(std::sync::atomic::Ordering::Relaxed) == 0 {
        return;
    }
    let mut g = BATCHED.lock();
    let Some(b) = g.as_mut() else { return };
    b.submit_open();
    b.reap(u64::MAX);
}

/// `tune upload_predetile`: a first upload of a 2D or array texture detiled
/// on the CPU, prepared before the image cache's lock by the recording
/// thread that binds it (`predetile`, from `Exec::bind_stage`): its mapped
/// parts read under the guest-reader lock and detiled into a staging buffer
/// of its own, its parts' sums and slices' hashes taken from the same bytes.
/// The cache was locked through every upload's detiling, and every other
/// recording thread's texture lookups, hazards and targets waited behind
/// it (5.6 s of waits over Castlewall's arrival, 4.1 s of them the graphics
/// workers', for 0.53 s of such holds; FINDINGS.md 2026-10-31). Under the
/// lock such an upload only records its copy (`Images::upload_prepared`).
/// One another thread uploaded first, or one a mapping change overtook
/// (`mem::MAP_GEN`), is dropped. `DS_PREDETILE_CHECK=1`: the cache's own
/// path runs as before and compares its bytes, regions and sums with the
/// prepared ones.
pub struct Prepared2D {
    key: ImageKey,
    raw: [u32; 8],
    map_gen: u64,
    size: u64,
    buf: vk::Buffer,
    mem: vk::DeviceMemory,
    ptr: *mut u8,
    total: usize,
    regions: Vec<vk::BufferImageCopy>,
    sums: Vec<u64>,
    skipped: u32,
    detiled: usize,
    hashes: [u64; SLICES],
}

thread_local! {
    /// This thread's prepared uploads for the stage it binds.
    static PREPARED: std::cell::RefCell<Vec<Prepared2D>> = const { std::cell::RefCell::new(Vec::new()) };
    /// A prepared upload the cache's own path compares with
    /// (`DS_PREDETILE_CHECK`).
    static CHECKED: std::cell::RefCell<Option<Prepared2D>> = const { std::cell::RefCell::new(None) };
}

/// Prepared uploads: made, their MiB, ns spent, used, dropped; checked,
/// differing in regions, in bytes, in sums.
static PREDETILED: [std::sync::atomic::AtomicU64; 9] = [const { std::sync::atomic::AtomicU64::new(0) }; 9];

/// Keys the cache holds, a bit per hash (set by `map_insert`, cleared by
/// `map_remove`): a texture whose bit is set is not prepared (a hint: a
/// collision costs a preparation dropped or one not made).
static PRESENT: [std::sync::atomic::AtomicU64; 4096] = [const { std::sync::atomic::AtomicU64::new(0) }; 4096];

fn present_bit(key: &ImageKey) -> (usize, u64) {
    let h = key.base.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ ((key.width as u64) << 32 | key.height as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F) ^ ((key.format.as_raw() as u64) << 16 | (key.layers as u64) << 40 | key.levels as u64);
    let b = (h >> 46) as usize & (4096 * 64 - 1);
    (b / 64, 1u64 << (b % 64))
}

fn predetile_check() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_PREDETILE_CHECK").ok().as_deref() == Some("1"))
}

fn free_prepared(p: Prepared2D) {
    let c = ctx();
    unsafe {
        c.device.destroy_buffer(p.buf, None);
        c.device.unmap_memory(p.mem);
        c.device.free_memory(p.mem, None);
    }
}

/// Prepares the first upload of the texture `t` names if it will need one
/// detiled on the CPU (`tune upload_predetile`; before the cache's lock).
pub fn predetile_for(t: &Tsharp) {
    use std::sync::atomic::Ordering::Relaxed;
    // (Known to this thread, or to the cache.)
    if key_hit(t, false).is_some_and(|k| k.is_some()) {
        return;
    }
    let Some(key) = Images::key_of(t) else { return };
    let (w, bit) = present_bit(&key);
    if key.ty3d || PRESENT[w].load(Relaxed) & bit != 0 || PREPARED.with(|p| p.borrow().iter().any(|x| x.key == key)) {
        return;
    }
    if super::tune::upload_copy() || super::tune::upload_async() || upload_check() {
        return;
    }
    let t0 = std::time::Instant::now();
    if let Some(p) = predetile(t, key) {
        PREDETILED[0].fetch_add(1, Relaxed);
        PREDETILED[1].fetch_add(p.total as u64, Relaxed);
        PREDETILED[2].fetch_add(t0.elapsed().as_nanos() as u64, Relaxed);
        super::hitch::note_since(super::hitch::Kind::Upload, t0, || format!("{}x{}x{} {:?} {} KiB, before the cache's lock", key.width, key.height, key.layers, key.format, p.total >> 10));
        PREPARED.with(|v| v.borrow_mut().push(p));
    }
}

/// The CPU path of a first upload (`Images::upload_inner`'s, which it
/// follows: parts mapped in the table read under the guest-reader lock,
/// the others zero-filled), into a staging buffer of its own. None: not
/// one to prepare (GPU detiling's size, a mapping change while choosing,
/// nothing to upload, no staging).
fn predetile(t: &Tsharp, key: ImageKey) -> Option<Prepared2D> {
    let el = elem_log2(t.format)?;
    let block = is_block(t.format);
    let layers = key.layers;
    let lay = tiling::layout(t.sw_mode, el, key.width, key.height, key.levels, block)?;
    let size = lay.size(layers);
    if let n @ 1.. = super::tune::gpu_detile() {
        if size >= n << 20 {
            return None;
        }
    }
    let elem = 1u64 << el;
    let range = |mip: usize, layer: u32| -> (u64, u64) {
        let m = &lay.mips[mip];
        let start = layer as u64 * lay.slice_size + if m.tail.is_some() { 0 } else { m.offset };
        (start, if m.tail.is_some() { 1u64 << lay.block_log2 } else { m.pitch as u64 * m.padded_height as u64 * elem })
    };
    let g0 = crate::mem::MAP_GEN.load(std::sync::atomic::Ordering::SeqCst);
    let table_whole = super::cp::valid(t.base, size) && crate::mem::host_readable(t.base, size);
    let n_parts = key.levels as usize * layers as usize;
    let table_parts: Vec<bool> = if table_whole {
        Vec::new()
    } else {
        (0..n_parts)
            .map(|i| {
                let (s, n) = range(i / layers as usize, (i % layers as usize) as u32);
                super::cp::valid(t.base + s, n) && crate::mem::host_readable(t.base + s, n)
            })
            .collect()
    };
    let readers = crate::mem::GUEST_READERS.read();
    if !crate::mem::table_view_held(g0) {
        return None;
    }
    let mapped = |mip: usize, layer: u32| table_whole || table_parts[mip * layers as usize + layer as usize];
    let sums_on = super::tune::upload_parts();
    let mut sums: Vec<u64> = if sums_on { vec![SUB_NONE; n_parts] } else { Vec::new() };
    let mut end = 0u64;
    let mut regions = Vec::new();
    let mut zeroed: Vec<(usize, u32, usize)> = Vec::new();
    let mut parts: Vec<(usize, u32, usize, usize)> = Vec::new();
    let mut total = 0usize;
    let mut skipped = 0u32;
    for mip in 0..key.levels as usize {
        for layer in 0..layers {
            let m = &lay.mips[mip];
            let len = m.width as usize * m.height as usize * elem as usize;
            if !mapped(mip, layer) {
                if sums_on {
                    sums[mip * layers as usize + layer as usize] = SUB_UNMAPPED;
                }
                skipped += 1;
                zeroed.push((mip, layer, len));
                continue;
            }
            let (s, n) = range(mip, layer);
            end = end.max(s + n);
            parts.push((mip, layer, total, len));
            regions.push(
                vk::BufferImageCopy::default()
                    .buffer_offset(total as u64)
                    .buffer_row_length(if block { m.width * 4 } else { m.width })
                    .buffer_image_height(if block { m.height * 4 } else { m.height })
                    .image_subresource(vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: mip as u32, base_array_layer: layer, layer_count: 1 })
                    .image_extent(vk::Extent3D { width: m.texel_w, height: m.texel_h, depth: 1 }),
            );
            total += len;
        }
    }
    if regions.is_empty() && zeroed.is_empty() {
        return None;
    }
    let src = unsafe { std::slice::from_raw_parts(t.base as *const u8, end as usize) };
    let zero_len = zeroed.iter().map(|z| z.2).max().unwrap_or(0);
    let (detiled, zero_off) = (total, total);
    total += zero_len;
    let (buf, mem, ptr) = ctx().host_buffer(total.max(4) as u64, vk::BufferUsageFlags::TRANSFER_SRC).ok()?;
    let dst = unsafe { std::slice::from_raw_parts_mut(ptr, total) };
    // (As the cache's path: parts under 1 MiB on several threads once they
    // add up to 1 MiB.)
    let small: Vec<(usize, u32, usize, usize)> = parts.iter().copied().filter(|p| p.3 < 1 << 20).collect();
    let small_bytes: usize = small.iter().map(|p| p.3).sum();
    for &(mip, layer, off, len) in &parts {
        if len >= 1 << 20 || small_bytes < 1 << 20 {
            lay.detile_into(src, mip, layer, &mut dst[off..off + len]);
        }
    }
    if small_bytes >= 1 << 20 {
        let threads = super::pool::threads();
        let mut rest: &mut [u8] = &mut dst[..];
        let mut base = 0usize;
        let mut jobs: Vec<Vec<(usize, u32, &mut [u8])>> = (0..threads).map(|_| Vec::new()).collect();
        let mut sorted = small;
        sorted.sort_by_key(|p| p.2);
        for (k, &(mip, layer, off, len)) in sorted.iter().enumerate() {
            let (_, tail) = std::mem::take(&mut rest).split_at_mut(off - base);
            let (mine, after) = tail.split_at_mut(len);
            rest = after;
            base = off + len;
            jobs[k % threads].push((mip, layer, mine));
        }
        let lay = &lay;
        let jobs: Vec<parking_lot::Mutex<Vec<(usize, u32, &mut [u8])>>> = jobs.into_iter().map(parking_lot::Mutex::new).collect();
        super::pool::for_each(jobs.len(), &|k| {
            for (mip, layer, out) in jobs[k].lock().iter_mut() {
                lay.detile_into(src, *mip, *layer, out);
            }
        });
    }
    dst[zero_off..].fill(0);
    if sums_on {
        let todo: Vec<(usize, u64, u64)> = parts
            .iter()
            .map(|&(mip, layer, _, _)| {
                let (s, n) = range(mip, layer);
                (mip * layers as usize + layer as usize, t.base + s, n)
            })
            .collect();
        let threads = if detiled >= 1 << 20 { super::pool::threads() } else { 1 };
        for (i, h) in hash_parts(&todo, threads) {
            sums[i] = h;
        }
    }
    // The slices' hashes from the same bytes (the cache takes them after
    // the upload).
    let hashes = all_hashes(t.base, size);
    drop(readers);
    if zero_len > 0 {
        for &(mip, layer, _) in &zeroed {
            let m = &lay.mips[mip];
            regions.push(
                vk::BufferImageCopy::default()
                    .buffer_offset(zero_off as u64)
                    .buffer_row_length(if block { m.width * 4 } else { m.width })
                    .buffer_image_height(if block { m.height * 4 } else { m.height })
                    .image_subresource(vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: mip as u32, base_array_layer: layer, layer_count: 1 })
                    .image_extent(vk::Extent3D { width: m.texel_w, height: m.texel_h, depth: 1 }),
            );
        }
    }
    Some(Prepared2D { key, raw: t.raw, map_gen: g0, size, buf, mem, ptr, total, regions, sums, skipped, detiled, hashes })
}

/// This thread's prepared upload of `key` for `t` while still current (a
/// mapping change since drops it).
fn take_prepared(key: &ImageKey, t: &Tsharp) -> Option<Prepared2D> {
    let p = PREPARED.with(|v| {
        let mut v = v.borrow_mut();
        let i = v.iter().position(|p| p.key == *key)?;
        Some(v.swap_remove(i))
    })?;
    if p.raw != t.raw || p.map_gen != crate::mem::MAP_GEN.load(std::sync::atomic::Ordering::SeqCst) {
        PREDETILED[4].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        free_prepared(p);
        return None;
    }
    Some(p)
}

/// Frees this thread's prepared uploads the binding did not use (another
/// thread uploaded them first, or the cache found another image).
pub fn discard_prepared() {
    let left: Vec<Prepared2D> = PREPARED.with(|v| std::mem::take(&mut *v.borrow_mut()));
    for p in left {
        PREDETILED[4].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        free_prepared(p);
    }
}

/// `DS_PREDETILE_CHECK=1`: the cache's own first upload compared with the
/// prepared one (`CHECKED`): staging size and regions, bytes, sums.
fn check_prepared(key: &ImageKey, total: usize, regions: &[vk::BufferImageCopy], dst: &[u8]) {
    use std::sync::atomic::Ordering::Relaxed;
    let Some(p) = CHECKED.with(|c| c.borrow_mut().take()) else { return };
    if p.key != *key {
        free_prepared(p);
        return;
    }
    PREDETILED[5].fetch_add(1, Relaxed);
    let same_region = |a: &vk::BufferImageCopy, b: &vk::BufferImageCopy| {
        a.buffer_offset == b.buffer_offset
            && a.buffer_row_length == b.buffer_row_length
            && a.buffer_image_height == b.buffer_image_height
            && a.image_subresource.mip_level == b.image_subresource.mip_level
            && a.image_subresource.base_array_layer == b.image_subresource.base_array_layer
            && a.image_extent.width == b.image_extent.width
            && a.image_extent.height == b.image_extent.height
    };
    let regions_ok = p.total == total && p.regions.len() == regions.len() && p.regions.iter().zip(regions).all(|(a, b)| same_region(a, b));
    let bytes_ok = regions_ok && unsafe { std::slice::from_raw_parts(p.ptr, p.total) } == &dst[..total];
    let sums_ok = SUB_SUMS.lock().as_ref().and_then(|m| m.get(key)).is_none_or(|s| *s == p.sums);
    for (ok, k) in [(regions_ok, 6), (bytes_ok, 7), (sums_ok, 8)] {
        if !ok && PREDETILED[k].fetch_add(1, Relaxed) < 8 {
            crate::warn!("images: prepared upload of {:#x} {}x{}x{} {:?} differs from the cache's own ({})", key.base, key.width, key.height, key.layers, key.format, ["regions", "bytes", "sums"][k - 6]);
        }
    }
    free_prepared(p);
}

/// The prepared uploads' status ("" before any).
fn predetile_line() -> String {
    use std::sync::atomic::Ordering::Relaxed;
    let v: Vec<u64> = PREDETILED.iter().map(|c| c.load(Relaxed)).collect();
    let skipped = SKIPPED_REUPLOADS.load(Relaxed);
    let skipped = if skipped > 0 { format!("; re-uploads skipped (`DS_SKIP_REUPLOADS`) {skipped}") } else { String::new() };
    if v[0] == 0 {
        return skipped;
    }
    let check = if v[5] > 0 { format!("; checked {} (differing: regions {}, bytes {}, sums {})", v[5], v[6], v[7], v[8]) } else { String::new() };
    format!("; detiled before the lock (`tune upload_predetile`) {} ({} MiB, {} ms), used {}, dropped {}{check}{skipped}", v[0], v[1] >> 20, v[2] / 1_000_000, v[3], v[4])
}

/// `DS_SKIP_REUPLOADS=1` (diagnosis, timing only: textures keep stale
/// contents): `Images::refresh` makes no re-upload, to bound what moving
/// re-uploads out of the cache's lock could gain (FINDINGS.md 2026-10-31).
fn skip_reuploads() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_SKIP_REUPLOADS").ok().as_deref() == Some("1"))
}

static SKIPPED_REUPLOADS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Pages of a texture are checked for CPU writes in SLICES interleaved
/// slices, one slice per frame (a change is seen within SLICES frames).
const SLICES: usize = 8;

/// Re-uploads after mapping changes (diagnostics), by kind of change
/// (`track::REMAP_*`): [count, with the sampled hashes unchanged, covering
/// part of the texture only, upload ns].
static REMAP_UPLOADS: [[std::sync::atomic::AtomicU64; 4]; 3] = [const { [const { std::sync::atomic::AtomicU64::new(0) }; 4] }; 3];
/// Partial re-uploads (`tune upload_parts`): [uploads, (mip, layer)s
/// uploaded, (mip, layer)s left out, bytes detiled].
static PART_UPLOADS: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];
/// Bytes detiled by 2D uploads (all of them).
static DETILED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// DS_UPLOAD_CHECK=1: every upload hashes all guest bytes of each mapped
/// (mip, layer); a partial re-upload checks that the ones it leaves out are
/// unchanged since the last upload ([checked, changed]): what the written
/// ranges and write-watch marks miss.
fn upload_check() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_UPLOAD_CHECK").ok().as_deref() == Some("1"))
}
static SUB_HASHES: Mutex<Option<HashMap<ImageKey, (std::time::Instant, Vec<u64>)>>> = Mutex::new(None);
/// Hashes of each (mip, layer)'s guest bytes at its last upload, by texture
/// (`tune upload_parts`; SUB_UNMAPPED: not mapped then, SUB_NONE: unknown).
static SUB_SUMS: Mutex<Option<HashMap<ImageKey, Vec<u64>>>> = Mutex::new(None);
const SUB_NONE: u64 = 0xAAAA_AAAA_AAAA_AAAA;
/// Partial re-uploads' candidates left out as unchanged (their hash).
static SUB_SAME: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static SUB_CHECKS: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];
const SUB_UNMAPPED: u64 = 0x5555_5555_5555_5555;

/// 2D uploads' time (diagnostics), ns: detiling, the staging buffer (made,
/// filled, freed), the transfer (submitted and waited for: the GPU's queue
/// drains first).
static UPLOAD_SPLIT: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];
/// Uploads whose pages were checked in the mapping table and held (no
/// mapping change before the reader lock), and those checked again with the
/// operating system (`mem::table_view_held`).
static UPLOAD_TABLE: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// 2D uploads detiled on the GPU (`tune gpu_detile`): [uploads, (mip,
/// layer)s, bytes copied from guest memory, bytes detiled, uploads left to
/// the CPU (no import window), ns recording and waiting, (mip, layer)s
/// compared with the CPU's detiling (`DS_GPU_DETILE_CHECK=1`), of them
/// differing, of them whose guest bytes changed during the upload, video-
/// memory buffers made, uploads whose transfer ran on the copy queue].
static GPU_DETILE: [std::sync::atomic::AtomicU64; 11] = [const { std::sync::atomic::AtomicU64::new(0) }; 11];

/// CPU-detiled uploads whose staging buffer the copy queue copied into
/// video memory (`tune upload_copy`).
static COPY_UPLOADS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// `DS_ZERO_CENSUS=1` (diagnosis): re-uploaded (mip, layer)s by cause (0 none,
/// 1 a commit, 2 another mapping change, 3 a file read, 4 a CPU write, 5 a GPU
/// write, 6 other): [parts, parts all zero in guest memory, bytes, zero
/// bytes, upload ns].
static ZERO_CENSUS: [[std::sync::atomic::AtomicU64; 5]; 7] = [const { [const { std::sync::atomic::AtomicU64::new(0) }; 5] }; 7];

fn zero_census() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_ZERO_CENSUS").ok().as_deref() == Some("1"))
}

thread_local! {
    /// The cause of the re-upload in progress (`ZERO_CENSUS`).
    static REUPLOAD_CAUSE: std::cell::Cell<u8> = const { std::cell::Cell::new(0) };
}

/// Status part: `ZERO_CENSUS` ("" when off).
pub fn zero_census_line() -> String {
    if !zero_census() {
        return String::new();
    }
    use std::sync::atomic::Ordering::Relaxed;
    const NAMES: [&str; 7] = ["first", "commit", "other mapping change", "file read", "CPU write", "GPU write", "other"];
    let parts: Vec<String> = (1..7)
        .filter(|&k| ZERO_CENSUS[k][0].load(Relaxed) > 0)
        .map(|k| {
            let v: Vec<u64> = ZERO_CENSUS[k].iter().map(|c| c.load(Relaxed)).collect();
            format!("{} {} parts ({} all zero; {} MiB, {} MiB zero; {} ms)", NAMES[k], v[0], v[1], v[2] >> 20, v[3] >> 20, v[4] / 1_000_000)
        })
        .collect();
    format!("    re-uploaded parts by cause (`DS_ZERO_CENSUS`): {}\n", parts.join("; "))
}

/// `DS_GPU_DETILE_CHECK=1`: every upload detiled on the GPU reads back the
/// tiled bytes it copied and the rows it made; the CPU's detiling of that
/// copy must give the same rows, (mip, layer) by (mip, layer), and guest
/// bytes that differ from the copy changed during the upload.
fn gpu_detile_check() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_GPU_DETILE_CHECK").ok().as_deref() == Some("1"))
}

/// Video-memory buffers kept for GPU detiling (tiled bytes, packed rows) up
/// to this size each and DETILE_POOL in all; larger uploads make and free
/// their own. Uploads detiled on the GPU are in flight (not waited for) with
/// up to DETILE_INFLIGHT bytes of such buffers.
const DETILE_KEEP: u64 = 128 << 20;
const DETILE_POOL: u64 = 256 << 20;
const DETILE_INFLIGHT: u64 = 512 << 20;

struct DetilePipe {
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
}

/// The detiling kernel's pipeline (`rdna::kernels::detile`; push
/// descriptors: binding 0 the tiled bytes, 1 the packed rows).
fn detile_pipe() -> Result<&'static DetilePipe, String> {
    static PIPE: std::sync::OnceLock<Result<DetilePipe, String>> = std::sync::OnceLock::new();
    PIPE.get_or_init(|| unsafe {
        let c = ctx();
        let d = &c.device;
        let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..2)
            .map(|b| vk::DescriptorSetLayoutBinding::default().binding(b).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE))
            .collect();
        let set_layout = d
            .create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().flags(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR).bindings(&bindings), None)
            .map_err(|e| format!("detile set layout: {e}"))?;
        let ranges = [vk::PushConstantRange { stage_flags: vk::ShaderStageFlags::COMPUTE, offset: 0, size: 4 * rdna::kernels::DETILE_PC_WORDS as u32 }];
        let layout = d.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&[set_layout]).push_constant_ranges(&ranges), None).map_err(|e| format!("detile layout: {e}"))?;
        let code = rdna::kernels::detile();
        let module = d.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None).map_err(|e| format!("detile module: {e}"))?;
        let stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(c"main");
        let ci = vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout);
        let pipeline = d.create_compute_pipelines(vk::PipelineCache::null(), &[ci], None).map_err(|(_, e)| format!("detile pipeline: {e}"));
        d.destroy_shader_module(module, None);
        Ok(DetilePipe { pipeline: pipeline?[0], layout })
    })
    .as_ref()
    .map_err(|e| e.clone())
}

/// Hashes of (index, guest address, bytes) parts, `threads` groups at a
/// time on the helper pool (`pool`).
fn hash_parts(todo: &[(usize, u64, u64)], threads: usize) -> Vec<(usize, u64)> {
    let per = todo.len().div_ceil(threads.max(1)).max(1);
    let out: Vec<std::sync::atomic::AtomicU64> = todo.iter().map(|_| std::sync::atomic::AtomicU64::new(0)).collect();
    super::pool::for_each(todo.len().div_ceil(per), &|c| {
        for k in c * per..((c + 1) * per).min(todo.len()) {
            let (_, a, n) = todo[k];
            out[k].store(hash_bytes(a, n as usize), std::sync::atomic::Ordering::Relaxed);
        }
    });
    todo.iter().zip(out).map(|(t, h)| (t.0, h.into_inner())).collect()
}

/// Hash of `n` bytes at `a` (four independent lanes over 64-bit words).
pub(super) fn hash_bytes(a: u64, n: usize) -> u64 {
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
/// SLICES-th 4 KiB page starting at page `slice`. Unmapped pages count as
/// a marker (partially resident textures map only some of their mips).
fn slice_hash(base: u64, size: u64, slice: usize) -> u64 {
    let mut h = 0u64;
    let mut a = base + 4096 * slice as u64;
    while a < base + size {
        let n = 256.min(base + size - a) as usize;
        let v = if super::cp::guest_readable(a, n as u64) { hash_bytes(a, n) } else { 0x5555_5555_5555_5555 };
        h = h.rotate_left(5) ^ v;
        a += 4096 * SLICES as u64;
    }
    h
}

/// Sampled hash of a guest range (diagnostics): all of it up to 4 KiB, else
/// 256 bytes at 16 evenly spaced points and its last 256 bytes.
pub(crate) fn sampled_hash(base: u64, size: u64) -> u64 {
    let piece = |a: u64, n: u64| if super::cp::guest_readable(a, n) { hash_bytes(a, n as usize) } else { 0x5555_5555_5555_5555 };
    if size <= 4096 {
        return piece(base, size);
    }
    let mut h = 0u64;
    for k in 0..16 {
        h = h.rotate_left(5) ^ piece(base + (size - 256) * k / 16, 256);
    }
    h.rotate_left(5) ^ piece(base + size - 256, 256)
}

fn all_hashes(base: u64, size: u64) -> [u64; SLICES] {
    std::array::from_fn(|s| slice_hash(base, size, s))
}

pub struct Images {
    map: HashMap<ImageKey, Image>,
    /// The keys of `map` by base address (`alias` lookups).
    by_base: HashMap<u64, Vec<ImageKey>>,
    /// Addresses with a GPU-written image (render targets, storage images,
    /// stencil planes).
    gpu_bases: super::fx::Set<u64>,
    /// Depth/stencil image of each rendered stencil plane address.
    stencil_of: HashMap<u64, ImageKey>,
    /// Surface of each metadata address (DCC, HTILE) and its clear values.
    /// Fast-clear metadata address -> (image, clear value); ordered for the
    /// range lookups of `note_meta_writes`.
    meta_of: std::collections::BTreeMap<u64, (ImageKey, [u32; 2])>,
    /// The writes `note_writes` last applied, with LOOKUP_GEN and note_gen
    /// then: the per-item culls report the same two ranges thousands of
    /// times per frame, and noting them again changes nothing until a
    /// pending clear or overwrite is taken or the images change.
    last_note: Option<(Vec<(u64, u64)>, Vec<(u64, u64)>, u64, u64)>,
    note_gen: u64,
    /// Sampled views by T#: (frame, LOOKUP_GEN, track::GEN, view, image).
    /// Within a frame, while no image was created, removed or re-aliased and
    /// no tracked texture source was written, `view` would do nothing but
    /// return the same view (draws bind the same textures thousands of times
    /// per frame; each full lookup took the tracker's lock three times).
    view_memo: HashMap<[u32; 8], (u64, u64, u64, vk::ImageView, vk::Image, UseStamp)>,
    /// `resolved_key` by (T#, storage): (LOOKUP_GEN, key).
    key_memo: HashMap<([u32; 8], bool), (u64, Option<ImageKey>)>,
    /// Images whose metadata a GPU or CP write covered since (fast clears).
    /// Images whose fast-clear metadata was written: (metadata address, its
    /// first word when the writer's value is known: `note_meta_fill`).
    pending_clears: HashMap<ImageKey, (u64, Option<u32>)>,
    samplers: HashMap<([u32; 4], bool), vk::Sampler>,
    pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
    /// The one-shot's GPU timestamps (`upload_stamps`).
    qp: vk::QueryPool,
    /// Staging buffers not in use (`staging`), uploads submitted without
    /// waiting (command buffer, fence, its timestamps, the staging buffer it
    /// reads; oldest first) and the command buffers, fences and timestamps
    /// of completed ones.
    staging_free: Vec<Staging>,
    inflight: std::collections::VecDeque<(vk::CommandBuffer, vk::Fence, vk::QueryPool, Held)>,
    spare_cmds: Vec<(vk::CommandBuffer, vk::Fence, vk::QueryPool)>,
    /// GPU detiling's video-memory buffers not in use: (buffer, memory,
    /// capacity).
    detile_free: Vec<(vk::Buffer, vk::DeviceMemory, u64)>,
    /// GPU detiling's transfers on the copy queue: its command pool (made
    /// on first use), spare command buffers, and the timeline its copies
    /// signal with `copy_value` (the main queue's part waits on it).
    copy_pool: vk::CommandPool,
    copy_spare: Vec<vk::CommandBuffer>,
    copy_timeline: vk::Semaphore,
    copy_value: u64,
    reported: HashMap<String, u64>,
    pub uploads: u64,
    pub reuploads: u64,
    pub upload_bytes: u64,
    pub upload_ns: u64,
    /// Counts storage-image bindings (write generations).
    write_seq: u64,
    /// GPU-owned images with a known guest span, by address (see
    /// `note_overwrites`), the largest span, and how many are marked.
    owned: std::collections::BTreeMap<u64, Vec<ImageKey>>,
    max_span: u64,
    pending_overwrites: usize,
    /// Images marked overwritten (with `pending_clears`' keys: the set
    /// `pending_hit` reads), and a count of changes to both (published at
    /// the end of `with` when it moved).
    overwritten_keys: super::fx::Set<ImageKey>,
    pending_change: u64,
    pending_published: u64,
    /// Textures the scrubber visits in turn (`scrub_batch`), rebuilt when
    /// LOOKUP_GEN moves, and its position.
    scrub_keys: Vec<ImageKey>,
    scrub_gen: u64,
    scrub_at: usize,
    /// GPU-owned images reloaded from guest memory after buffer writes
    /// covered them: cleared (all zero) / uploaded.
    pub overwrite_clears: u64,
    pub overwrite_uploads: u64,
    /// Images reported for buffer writes covering only part of them.
    partial_seen: super::fx::Set<u64>,
    /// Zero images for unused image table entries, by view type.
    dummies: HashMap<vk::ImageViewType, vk::ImageView>,
    /// Depth images grown (status line).
    pub depth_grows: u64,
    /// Bases of R32 float / R16 unorm images bound as storage: compute
    /// views of a depth plane (`depth_copies`).
    depth_feeds: super::fx::Set<u64>,
}

unsafe impl Send for Images {}

pub static IMAGES: Mutex<Option<Images>> = Mutex::new(None);

thread_local! {
    /// The shader whose bindings are being resolved on this thread
    /// (diagnostics: which pass first writes an uploaded image).
    pub static BINDING_SHADER: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Runs `f` with the image cache.
/// Texture scrubbing (`tune tex_scrub`): a background thread samples the
/// guest bytes of uploaded textures for CPU writes (`scrub_batch`) instead
/// of the first draw that binds each in a frame (`refresh`); texture views
/// then stay valid across frames (`view_hit`).
pub fn scrubbing() -> bool {
    super::tune::tex_scrub()
}

static SCRUB_CHECKS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Write-watch scrubbing (`tex_scrub` 2): takes and the pages they covered.
static SCRUB_TAKES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static SCRUB_TAKE_PAGES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static SCRUB_EDGE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static SCRUB_DIRTY: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn scrubber() {
    super::profile_this_thread("scrub");
    loop {
        let marks = super::tune::tex_scrub_mode() == 2;
        std::thread::sleep(std::time::Duration::from_millis(if marks { super::tune::scrub_ms() } else { 2 }));
        if !scrubbing() || super::shutting_down() {
            continue;
        }
        if marks {
            // Write-watch marks of the next textures' pages: taking them
            // tells the write tracker which pages the CPU wrote (the
            // texture's own region included: re-uploaded at its next use).
            // In pieces of at most 16 MiB (the copy cache's lock is held
            // during a take).
            let work = with(|im| im.scrub_regions(64));
            let mut pages = Vec::new();
            for (b, n) in work {
                let mut a = b;
                while a < b + n {
                    let len = (b + n - a).min(16 << 20);
                    pages.clear();
                    SCRUB_TAKES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    SCRUB_TAKE_PAGES.fetch_add(len / 4096, std::sync::atomic::Ordering::Relaxed);
                    let _ = super::precache::take_written_for(a, len, &mut pages, None, 3);
                    SCRUB_DIRTY.fetch_add((!pages.is_empty()) as u64, std::sync::atomic::Ordering::Relaxed);
                    // Written pages only where the texture shares a page with
                    // other data (its first and last page).
                    let (first, last) = (b & !4095, (b + n - 1) & !4095);
                    if !pages.is_empty() && pages.iter().all(|&p| p == first || p == last) {
                        SCRUB_EDGE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                    a += len;
                }
            }
            continue;
        }
        let work = with(|im| im.scrub_batch(32));
        for (region, b, n, s, stored) in work {
            SCRUB_CHECKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if slice_hash(b, n, s) != stored {
                SCRUB_DIRTY.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                super::track::sampled_dirty(region);
            }
        }
    }
}

/// One status line ("" before the scrubber checked anything).
pub fn scrub_status() -> String {
    let takes = SCRUB_TAKES.load(std::sync::atomic::Ordering::Relaxed);
    if takes > 0 {
        return format!(
            "    texture scrubbing by write-watch marks: {takes} takes over {} pages, {} found CPU writes ({} only on a first or last page)\n",
            SCRUB_TAKE_PAGES.load(std::sync::atomic::Ordering::Relaxed),
            SCRUB_DIRTY.load(std::sync::atomic::Ordering::Relaxed),
            SCRUB_EDGE.load(std::sync::atomic::Ordering::Relaxed)
        );
    }
    let n = SCRUB_CHECKS.load(std::sync::atomic::Ordering::Relaxed);
    if n == 0 {
        return String::new();
    }
    format!("    texture scrubbing: {n} slices checked, {} changed (re-uploaded at their next use)\n", SCRUB_DIRTY.load(std::sync::atomic::Ordering::Relaxed))
}

/// Follows `Images::note_gen` (read without the lock: `Exec`'s write-note
/// memo).
pub static NOTE_GEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Pages (64 KiB, hashed) holding a GPU-owned image's bytes or a fast-clear
/// metadata address (`Images::owned`, `meta_of`), marked when indexed and
/// never cleared (both only grow): buffer writes on no marked page have
/// nothing for `note_writes` and skip the image cache's lock (`may_note`;
/// every writing dispatch took it, the most waited-for lock after the copy
/// cache's, 2026-10-15).
const NOTE_BUCKETS: usize = 1 << 16;
static NOTE_PAGES: [std::sync::atomic::AtomicBool; NOTE_BUCKETS] = [const { std::sync::atomic::AtomicBool::new(false) }; NOTE_BUCKETS];

fn note_bucket(page: u64) -> usize {
    (page.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 48) as usize
}

fn mark_note_pages(base: u64, len: u64) {
    if len == 0 {
        return;
    }
    let (first, last) = (base >> 16, (base + len - 1) >> 16);
    if last - first >= NOTE_BUCKETS as u64 {
        for b in NOTE_PAGES.iter() {
            b.store(true, std::sync::atomic::Ordering::Release);
        }
        return;
    }
    for page in first..=last {
        NOTE_PAGES[note_bucket(page)].store(true, std::sync::atomic::Ordering::Release);
    }
}

/// Whether `note_writes` may have something to do for these writes (false:
/// certainly not; when a mark races a write, that write was not ordered
/// after the image's indexing under the lock either).
pub fn may_note(writes: &[(u64, u64)]) -> bool {
    if !super::tune::note_filter() {
        return true;
    }
    writes.iter().any(|&(b, n)| {
        if n == 0 {
            return false;
        }
        let (first, last) = (b >> 16, (b + n - 1) >> 16);
        last - first >= 1024 || (first..=last).any(|p| NOTE_PAGES[note_bucket(p)].load(std::sync::atomic::Ordering::Acquire))
    })
}

/// `DS_IMAGE_HOLDS=1` (diagnosis): the image cache's holds of 200 us or more
/// (`with`): count, total and longest ns, by the holder's thread class
/// (graphics CP, its workers, compute queues' CPs and workers, others); and
/// its waits of 1 us or more by the waiter's class: count, ns.
static HOLDS: [[std::sync::atomic::AtomicU64; 3]; 4] = [const { [const { std::sync::atomic::AtomicU64::new(0) }; 3] }; 4];
static WAITS: [[std::sync::atomic::AtomicU64; 2]; 4] = [const { [const { std::sync::atomic::AtomicU64::new(0) }; 2] }; 4];

/// `image_for`'s steps under the lock (`DS_IMAGE_HOLDS`): refresh of an
/// image held, create, upload (prepared), upload (the cache's own), write
/// watch, hashes; each [count, ns, count of 200 us or more].
static STEPS: [[std::sync::atomic::AtomicU64; 3]; 6] = [const { [const { std::sync::atomic::AtomicU64::new(0) }; 3] }; 6];

fn note_step(k: usize, t: Option<std::time::Instant>) {
    use std::sync::atomic::Ordering::Relaxed;
    if let Some(t) = t {
        let ns = t.elapsed().as_nanos() as u64;
        STEPS[k][0].fetch_add(1, Relaxed);
        STEPS[k][1].fetch_add(ns, Relaxed);
        if ns >= 200_000 {
            STEPS[k][2].fetch_add(1, Relaxed);
        }
    }
}

/// The long holds (HOLDS) by the call site of `with`: count, ns.
static HOLD_SITES: Mutex<Option<HashMap<(&'static str, u32), (u64, u64)>>> = Mutex::new(None);

thread_local! {
    /// This thread's class for HOLDS and WAITS (255: not looked up yet).
    static HOLD_CLASS: std::cell::Cell<u8> = const { std::cell::Cell::new(255) };
}

fn holds_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_IMAGE_HOLDS").ok().as_deref() == Some("1"))
}

fn hold_class() -> usize {
    HOLD_CLASS.with(|c| {
        if c.get() == 255 {
            let n = crate::log::thread_name();
            c.set(match n.as_str() {
                "gpu-gfx" => 0,
                n if n.starts_with("gpu-gfx-par") => 1,
                n if n.starts_with("gpu-acb") => 2,
                _ => 3,
            });
        }
        c.get() as usize
    })
}

/// Clears the hold census (`profile reset`, through `lockstat`).
pub fn reset_holds() {
    for c in HOLDS.iter().flatten().chain(WAITS.iter().flatten()).chain(STEPS.iter().flatten()) {
        c.store(0, std::sync::atomic::Ordering::Relaxed);
    }
    *HOLD_SITES.lock() = None;
}

/// The hold census for the contended-locks status line ("" when off).
pub fn holds_part() -> String {
    if !holds_on() {
        return String::new();
    }
    use std::sync::atomic::Ordering::Relaxed;
    let names = ["graphics CP", "its workers", "compute queues", "others"];
    let parts: Vec<String> = names
        .iter()
        .zip(HOLDS.iter())
        .map(|(n, h)| format!("{n} {} ({} ms, longest {:.1} ms)", h[0].load(Relaxed), h[1].load(Relaxed) / 1_000_000, h[2].load(Relaxed) as f64 / 1e6))
        .collect();
    let waits: Vec<String> = names.iter().zip(WAITS.iter()).map(|(n, w)| format!("{n} {} ({} ms)", w[0].load(Relaxed), w[1].load(Relaxed) / 1_000_000)).collect();
    let mut sites: Vec<((&'static str, u32), (u64, u64))> = HOLD_SITES.lock().as_ref().map(|m| m.iter().map(|(k, v)| (*k, *v)).collect()).unwrap_or_default();
    sites.sort_by_key(|s| std::cmp::Reverse(s.1 .1));
    let sites: Vec<String> = sites.iter().take(8).map(|((f, l), (n, ns))| format!("{f}:{l} {n} ({} ms)", ns / 1_000_000)).collect();
    let step_names = ["refresh", "create", "prepared upload", "own upload", "write watch", "hashes"];
    let steps: Vec<String> = step_names.iter().zip(STEPS.iter()).map(|(n, s)| format!("{n} {} ({} ms, {} of 200 us or more)", s[0].load(Relaxed), s[1].load(Relaxed) / 1_000_000, s[2].load(Relaxed))).collect();
    format!("; image cache held 200 us or more (`DS_IMAGE_HOLDS`): {}; waited for by: {}; long holds by site: {}; image_for's steps: {}", parts.join(", "), waits.join(", "), sites.join(", "), steps.join(", "))
}

#[track_caller]
pub fn with<R>(f: impl FnOnce(&mut Images) -> R) -> R {
    static SCRUBBER: std::sync::Once = std::sync::Once::new();
    SCRUBBER.call_once(|| {
        if let Err(e) = crate::thread::spawn_host("gpu-texscrub", 1 << 20, Box::new(scrubber)) {
            crate::warn!("images: no texture scrubber thread ({e})");
        }
    });
    let asked = holds_on().then(std::time::Instant::now);
    let mut g = super::lockstat::IMAGE_CACHE.lock(&IMAGES);
    let held_from = holds_on().then(std::time::Instant::now);
    if let (Some(a), Some(b)) = (asked, held_from) {
        let ns = (b - a).as_nanos() as u64;
        if ns >= 1000 {
            let w = &WAITS[hold_class()];
            w[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            w[1].fetch_add(ns, std::sync::atomic::Ordering::Relaxed);
        }
    }
    if g.is_none() {
        let c = ctx();
        unsafe {
            let pool = c
                .device
                .create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(c.queue_family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER), None)
                .expect("image pool");
            let cmd = c.device.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(pool).command_buffer_count(1)).expect("image cmd")[0];
            let fence = c.device.create_fence(&vk::FenceCreateInfo::default(), None).expect("image fence");
            let qp = upload_stamps();
            *g = Some(Images { qp, map: HashMap::default(), by_base: HashMap::default(), gpu_bases: Default::default(), stencil_of: HashMap::default(), meta_of: Default::default(), last_note: None, note_gen: 0, view_memo: HashMap::default(), key_memo: HashMap::default(), pending_clears: HashMap::default(), samplers: HashMap::default(), pool, cmd, fence, staging_free: Vec::new(), inflight: Default::default(), spare_cmds: Vec::new(), detile_free: Vec::new(), copy_pool: vk::CommandPool::null(), copy_spare: Vec::new(), copy_timeline: vk::Semaphore::null(), copy_value: 0, reported: HashMap::default(), uploads: 0, reuploads: 0, upload_bytes: 0, upload_ns: 0, write_seq: 0, owned: Default::default(), max_span: 0, pending_overwrites: 0, overwritten_keys: Default::default(), pending_change: 0, pending_published: 0, scrub_keys: Vec::new(), scrub_gen: 0, scrub_at: 0, overwrite_clears: 0, overwrite_uploads: 0, partial_seen: Default::default(), dummies: HashMap::default(), depth_grows: 0, depth_feeds: Default::default() });
        }
    }
    let im = g.as_mut().unwrap();
    let r = f(&mut *im);
    PENDING.store((im.pending_clears.len() + im.pending_overwrites) as u64, std::sync::atomic::Ordering::Release);
    if im.pending_change != im.pending_published {
        im.pending_published = im.pending_change;
        let set: super::fx::Set<ImageKey> = im.pending_clears.keys().copied().chain(im.overwritten_keys.iter().copied()).collect();
        *PENDING_SET.lock() = Some(std::sync::Arc::new(set));
        PENDING_GEN.fetch_add(1, std::sync::atomic::Ordering::Release);
    }
    if let Some(t) = held_from {
        let ns = t.elapsed().as_nanos() as u64;
        if ns >= 200_000 {
            use std::sync::atomic::Ordering::Relaxed;
            let h = &HOLDS[hold_class()];
            h[0].fetch_add(1, Relaxed);
            h[1].fetch_add(ns, Relaxed);
            h[2].fetch_max(ns, Relaxed);
            let l = std::panic::Location::caller();
            let mut s = HOLD_SITES.lock();
            let e = s.get_or_insert_with(HashMap::default).entry((l.file(), l.line())).or_insert((0, 0));
            e.0 += 1;
            e.1 += ns;
        }
    }
    r
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
    fn create(&mut self, key: ImageKey) -> Result<(vk::Image, vk::DeviceMemory, u64), String> {
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
            let t0 = std::time::Instant::now();
            let image = c.device.create_image(&ci, None).map_err(|e| format!("vkCreateImage {key:?}: {e}"))?;
            let req = c.device.get_image_memory_requirements(image);
            let mem = c.alloc(req, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
            c.device.bind_image_memory(image, mem, 0).map_err(|e| format!("bind image: {e}"))?;
            super::hitch::note_since(super::hitch::Kind::Upload, t0, || format!("image creation {}x{} ({} MiB)", key.width, key.height, req.size >> 20));
            Ok((image, mem, req.size))
        }
    }

    /// Records commands into the cache's command buffer and waits for them.
    /// A staging buffer of at least `n` bytes: the smallest free one that
    /// fits, else a new one (rounded up to a power of two from 1 MiB, so
    /// they are reused, up to STAGING_POOLED).
    fn staging(&mut self, n: u64) -> Result<Staging, String> {
        if let Some(i) = self.staging_free.iter().enumerate().filter(|(_, s)| s.cap >= n).min_by_key(|(_, s)| s.cap).map(|(i, _)| i) {
            return Ok(self.staging_free.swap_remove(i));
        }
        let cap = if n > STAGING_POOLED { n } else { n.max(1 << 20).next_power_of_two() };
        let t0 = std::time::Instant::now();
        let (buf, mem, ptr) = ctx().host_buffer(cap, vk::BufferUsageFlags::TRANSFER_SRC)?;
        super::hitch::note_since(super::hitch::Kind::Upload, t0, || format!("a new staging buffer of {} MiB", cap >> 20));
        STAGING_MADE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(Staging { buf, mem, ptr, cap })
    }

    /// A staging buffer the GPU is done with: kept for reuse or freed.
    fn release_staging(&mut self, s: Staging) {
        let kept: u64 = self.staging_free.iter().map(|s| s.cap).sum();
        if s.cap <= STAGING_POOLED && kept + s.cap <= STAGING_KEEP {
            self.staging_free.push(s);
            return;
        }
        let d = &ctx().device;
        unsafe {
            d.destroy_buffer(s.buf, None);
            d.unmap_memory(s.mem);
            d.free_memory(s.mem, None);
        }
    }

    /// Recycles the uploads the GPU completed (in submission order); with
    /// more than `max` in flight or more than `max_bytes` of their staging
    /// buffers, waits for the oldest.
    fn reap(&mut self, max: usize, max_bytes: u64) {
        let c = ctx();
        while let Some(&(_, fence, _, _)) = self.inflight.front() {
            let done = unsafe { c.device.get_fence_status(fence) }.unwrap_or(true);
            if !done {
                let bytes: u64 = self.inflight.iter().map(|x| x.3.bytes()).sum();
                if self.inflight.len() <= max && bytes <= max_bytes {
                    break;
                }
                UPLOAD_WAITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let t0 = std::time::Instant::now();
                let r = unsafe { c.device.wait_for_fences(&[fence], true, 10_000_000_000) };
                super::hitch::note_since(super::hitch::Kind::GpuWait, t0, || format!("earlier texture uploads ({} in flight, {} MiB)", self.inflight.len(), bytes >> 20));
                if let Err(e) = r {
                    if !super::exec::DEVICE_LOST.swap(true, std::sync::atomic::Ordering::SeqCst) {
                        crate::error!("images: upload failed: {e}; GPU execution stops; checkpoints: {}", super::vk::checkpoint_report());
                    }
                }
            }
            let (cmd, fence, qp, held) = self.inflight.pop_front().unwrap();
            note_upload_span(qp, || match (&held.staging, held.dev.is_empty()) {
                (Some(s), true) => format!(": texture upload from a {} KiB staging buffer", s.cap >> 10),
                (Some(s), false) => format!(": texture upload from a {} KiB staging buffer through the copy queue", s.cap >> 10),
                (None, _) => format!(": texture upload detiled on the GPU ({} KiB of buffers)", held.dev.iter().map(|b| b.2).sum::<u64>() >> 10),
            });
            unsafe {
                c.device.reset_fences(&[fence]).ok();
                c.device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty()).ok();
            }
            self.spare_cmds.push((cmd, fence, qp));
            self.release_held(held);
        }
    }

    /// Records and submits a transfer reading staging buffer `s` without
    /// waiting for it: on the one queue it runs after the work submitted
    /// before it and before batches submitted later, and its barriers order
    /// it against the image's other uses; `reap` recycles it.
    fn submit_upload(&mut self, s: Staging, f: impl FnOnce(vk::CommandBuffer)) {
        self.submit_held(Held::staging(s), MAX_INFLIGHT_BYTES, None, f);
    }

    /// What an upload the GPU is done with (or that was not submitted) held.
    fn release_held(&mut self, held: Held) {
        if let Some(s) = held.staging {
            self.release_staging(s);
        }
        self.release_detile(held.dev);
        if let Some(x) = held.copy_cmd {
            unsafe { ctx().device.reset_command_buffer(x, vk::CommandBufferResetFlags::empty()).ok() };
            self.copy_spare.push(x);
        }
    }

    /// The copy queue's part of an upload: `record` on one of its command
    /// buffers, submitted to signal the timeline's next value. Returns the
    /// command buffer and that value (None: nothing was submitted).
    fn submit_copy_part(&mut self, pool: vk::CommandPool, sem: vk::Semaphore, record: impl FnOnce(vk::CommandBuffer)) -> Option<(vk::CommandBuffer, u64)> {
        let c = ctx();
        let x = match self.copy_spare.pop() {
            Some(x) => x,
            None => unsafe { c.device.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(pool).command_buffer_count(1)).ok()?[0] },
        };
        let ok = unsafe {
            c.device.begin_command_buffer(x, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).ok();
            record(x);
            c.device.end_command_buffer(x).ok();
            c.submit_timeline(true, &[x], &[], &[(sem, self.copy_value + 1)], vk::Fence::null()).is_ok()
        };
        if !ok {
            unsafe { c.device.reset_command_buffer(x, vk::CommandBufferResetFlags::empty()).ok() };
            self.copy_spare.push(x);
            return None;
        }
        self.copy_value += 1;
        Some((x, self.copy_value))
    }

    /// The copy queue's part of GPU detiling: its command pool and timeline,
    /// made on first use (None: no copy queue, or `tune copy_queue` 0).
    fn copy_parts(&mut self) -> Option<(vk::CommandPool, vk::Semaphore)> {
        let c = ctx();
        let q = c.copy.as_ref().filter(|_| super::tune::copy_queue())?;
        if self.copy_pool == vk::CommandPool::null() {
            unsafe {
                let pool = c.device.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(q.family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER), None).ok()?;
                let mut tci = vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE).initial_value(0);
                let Ok(sem) = c.device.create_semaphore(&vk::SemaphoreCreateInfo::default().push_next(&mut tci), None) else {
                    c.device.destroy_command_pool(pool, None);
                    return None;
                };
                self.copy_pool = pool;
                self.copy_timeline = sem;
            }
        }
        Some((self.copy_pool, self.copy_timeline))
    }

    /// `submit_upload` for what `s` holds, with up to `max_bytes` of such
    /// buffers in flight, after the copy queue's timeline reaches `wait`
    /// (semaphore, value) when given.
    fn submit_held(&mut self, s: Held, max_bytes: u64, wait: Option<(vk::Semaphore, u64)>, f: impl FnOnce(vk::CommandBuffer)) {
        // (After the batched uploads recorded before it, `tune upload_batch`.)
        flush_uploads();
        let max = if s.copy_cmd.is_some() || !s.dev.is_empty() { MAX_INFLIGHT_COPY } else { MAX_INFLIGHT };
        self.reap(max - 1, max_bytes.saturating_sub(s.bytes()));
        let c = ctx();
        let (cmd, fence, qp) = match self.spare_cmds.pop() {
            Some(x) => x,
            None => unsafe {
                let cmd = c.device.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(self.pool).command_buffer_count(1)).expect("upload cmd")[0];
                (cmd, c.device.create_fence(&vk::FenceCreateInfo::default(), None).expect("upload fence"), upload_stamps())
            },
        };
        unsafe {
            c.device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).ok();
            stamp(cmd, qp, 0);
            f(cmd);
            stamp(cmd, qp, 1);
            c.device.end_command_buffer(cmd).ok();
            let r = match wait {
                Some((sem, v)) => c.submit_timeline(false, &[cmd], &[(sem, v, vk::PipelineStageFlags2::ALL_COMMANDS)], &[], fence),
                None => c.submit(cmd, &[], &[], fence),
            };
            match r {
                Ok(()) => self.inflight.push_back((cmd, fence, qp, s)),
                Err(e) => {
                    if !super::exec::DEVICE_LOST.swap(true, std::sync::atomic::Ordering::SeqCst) {
                        crate::error!("images: submit failed: {e}; GPU execution stops; checkpoints: {}", super::vk::checkpoint_report());
                    }
                    c.device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty()).ok();
                    self.spare_cmds.push((cmd, fence, qp));
                    self.release_held(s);
                }
            }
        }
    }

    fn one_shot(&mut self, f: impl FnOnce(vk::CommandBuffer)) {
        self.one_shot_after(None, f)
    }

    /// `one_shot` after the copy queue's timeline reaches `wait` (semaphore,
    /// value) when given.
    fn one_shot_after(&mut self, wait: Option<(vk::Semaphore, u64)>, f: impl FnOnce(vk::CommandBuffer)) {
        // (Its timestamps: `stamp`, `note_upload_span`. After the batched
        // uploads recorded before it, `tune upload_batch`.)
        flush_uploads();
        let c = ctx();
        unsafe {
            c.device.begin_command_buffer(self.cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).ok();
            stamp(self.cmd, self.qp, 0);
            f(self.cmd);
            stamp(self.cmd, self.qp, 1);
            c.device.end_command_buffer(self.cmd).ok();
            let r = match wait {
                Some((sem, v)) => c.submit_timeline(false, &[self.cmd], &[(sem, v, vk::PipelineStageFlags2::ALL_COMMANDS)], &[], self.fence),
                None => c.submit(self.cmd, &[], &[], self.fence),
            };
            match r {
                Ok(()) => {
                    let t0 = std::time::Instant::now();
                    let r = c.device.wait_for_fences(&[self.fence], true, 10_000_000_000);
                    super::hitch::note_since(super::hitch::Kind::GpuWait, t0, || "a texture cache one-shot (upload or transition)".to_string());
                    if r.is_ok() {
                        note_upload_span(self.qp, || ": texture cache one-shot (upload or transition)".to_string());
                    }
                    if let Err(e) = r {
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

    /// Value statistics of every cached image at guest address `base` (mip
    /// 0, all layers or slices; float formats), with the largest magnitude
    /// of the first channel per layer/slice; BC1 textures: each layer's mean
    /// endpoint color and share of all-zero blocks (diagnosis: `imgstat`).
    pub fn stats_at(&mut self, base: u64) -> Vec<String> {
        let imgs: Vec<(vk::Image, ImageKey)> = self.map.values().filter(|i| i.key.base == base).map(|i| (i.image, i.key)).collect();
        let mut out = Vec::new();
        for (image, k) in imgs {
            let bpp = texel_bytes(k.format) as u64;
            let half = matches!(k.format, vk::Format::R16G16B16A16_SFLOAT | vk::Format::R16G16_SFLOAT | vk::Format::R16_SFLOAT);
            let full = matches!(k.format, vk::Format::R32G32B32A32_SFLOAT | vk::Format::R32G32_SFLOAT | vk::Format::R32_SFLOAT);
            let bc1 = matches!(k.format, vk::Format::BC1_RGBA_SRGB_BLOCK | vk::Format::BC1_RGBA_UNORM_BLOCK);
            let unorm8 = matches!(k.format, vk::Format::R8_UNORM | vk::Format::R8G8_UNORM | vk::Format::R8G8B8A8_UNORM | vk::Format::B8G8R8A8_UNORM);
            let unorm16 = matches!(k.format, vk::Format::R16_UNORM | vk::Format::R16G16_UNORM | vk::Format::R16G16B16A16_UNORM);
            if bpp == 0 || !(half || full || bc1 || unorm8 || unorm16) {
                out.push(format!("{base:#x} {}x{}x{} {:?}: not a float format", k.width, k.height, k.layers, k.format));
                continue;
            }
            let (layers, depth) = if k.ty3d { (1, k.layers) } else { (k.layers, 1) };
            let slice = if bc1 { bpp * k.width.div_ceil(4) as u64 * k.height.div_ceil(4) as u64 } else { bpp * k.width as u64 * k.height as u64 };
            let size = slice * (layers * depth) as u64;
            let c = ctx();
            let Ok((buf, mem, ptr)) = c.host_buffer(size, vk::BufferUsageFlags::TRANSFER_DST) else {
                out.push(format!("{base:#x}: no staging buffer"));
                continue;
            };
            self.one_shot(|cmd| unsafe {
                let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::MEMORY_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_READ);
                c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[mb], &[], &[]);
                let region = vk::BufferImageCopy::default()
                    .image_subresource(vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: layers })
                    .image_extent(vk::Extent3D { width: k.width, height: k.height, depth });
                c.device.cmd_copy_image_to_buffer(cmd, image, vk::ImageLayout::GENERAL, buf, &[region]);
                let hb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::HOST_READ);
                c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::HOST, vk::DependencyFlags::empty(), &[hb], &[], &[]);
            });
            let raw = unsafe { std::slice::from_raw_parts(ptr, size as usize).to_vec() };
            unsafe {
                c.device.destroy_buffer(buf, None);
                c.device.unmap_memory(mem);
                c.device.free_memory(mem, None);
            }
            if bc1 {
                let rgb = |c: u16| [((c >> 11) & 31) as f32 / 31.0, ((c >> 5) & 63) as f32 / 63.0, (c & 31) as f32 / 31.0];
                let per: Vec<String> = raw
                    .chunks(slice as usize)
                    .enumerate()
                    .map(|(l, s)| {
                        let (mut sum, mut zero, mut n) = ([0f64; 3], 0usize, 0usize);
                        for b in s.chunks_exact(8) {
                            n += 1;
                            if b.iter().all(|&x| x == 0) {
                                zero += 1;
                            }
                            let (c0, c1) = (rgb(u16::from_le_bytes([b[0], b[1]])), rgb(u16::from_le_bytes([b[2], b[3]])));
                            for i in 0..3 {
                                sum[i] += (c0[i] + c1[i]) as f64 / 2.0;
                            }
                        }
                        let n = n.max(1) as f64;
                        format!("{l}:{:.2}/{:.2}/{:.2}{}", sum[0] / n, sum[1] / n, sum[2] / n, if zero > 0 { format!(" z{:.0}%", 100.0 * zero as f64 / n) } else { String::new() })
                    })
                    .collect();
                out.push(format!("{base:#x} {}x{}x{} {:?}: mean endpoint rgb per layer: {}", k.width, k.height, k.layers, k.format, per.join(" ")));
                continue;
            }
            if unorm8 || unorm16 {
                // Per channel mean and share of zero texels, per layer.
                let w = if unorm16 { 2 } else { 1 };
                let max = if unorm16 { 65535.0 } else { 255.0 };
                let per: Vec<String> = raw
                    .chunks(slice as usize)
                    .enumerate()
                    .map(|(l, s)| {
                        let n = s.len() / bpp as usize;
                        let ch = bpp as usize / w;
                        let mut sum = [0u64; 4];
                        let mut zero = 0usize;
                        for t in s.chunks_exact(bpp as usize) {
                            if t.iter().all(|&x| x == 0) {
                                zero += 1;
                            }
                            for (i, c) in t.chunks_exact(w).enumerate() {
                                sum[i] += if w == 2 { u16::from_le_bytes([c[0], c[1]]) as u64 } else { c[0] as u64 };
                            }
                        }
                        let means: Vec<String> = (0..ch).map(|i| format!("{:.3}", sum[i] as f64 / max / n.max(1) as f64)).collect();
                        format!("{l}:{}{}", means.join("/"), if zero > 0 { format!(" z{:.0}%", 100.0 * zero as f64 / n.max(1) as f64) } else { String::new() })
                    })
                    .collect();
                out.push(format!("{base:#x} {}x{}x{} {:?}: channel means per layer: {}", k.width, k.height, k.layers, k.format, per.join(" ")));
                continue;
            }
            let st = float_stats(k.format, &raw).unwrap_or_default();
            let maxima: Vec<String> = raw
                .chunks(slice as usize)
                .map(|s| {
                    let m = s
                        .chunks_exact(bpp as usize)
                        .map(|t| if half { half_to_f32(u16::from_le_bytes([t[0], t[1]])) } else { f32::from_le_bytes([t[0], t[1], t[2], t[3]]) })
                        .fold(0f32, |m, v| if v.is_nan() { f32::NAN } else { m.max(v.abs()) });
                    format!("{m:.3}")
                })
                .collect();
            out.push(format!("{base:#x} {}x{}x{} {:?}: {st}; first-channel max per {}: {}", k.width, k.height, k.layers, k.format, if k.ty3d { "slice" } else { "layer" }, maxima.join(" ")));
        }
        out
    }

    /// Fills every cached image at guest address `base` (all layers and
    /// mips) with one color value (diagnosis: `fillimg`; experiments with
    /// GPU-written inputs). Returns how many images.
    pub fn fill_at(&mut self, base: u64, value: f32) -> usize {
        let imgs: Vec<(vk::Image, ImageKey)> = self.map.values().filter(|i| i.key.base == base).map(|i| (i.image, i.key)).collect();
        let c = ctx();
        for (image, k) in &imgs {
            if is_depth(k.format) {
                continue;
            }
            let layers = if k.ty3d { 1 } else { k.layers };
            let range = vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: k.levels.max(1), base_array_layer: 0, layer_count: layers };
            let color = vk::ClearColorValue { float32: [value; 4] };
            self.one_shot(|cmd| unsafe {
                let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::MEMORY_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
                c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[mb], &[], &[]);
                c.device.cmd_clear_color_image(cmd, *image, vk::ImageLayout::GENERAL, &color, &[range]);
                let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
                c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::ALL_COMMANDS, vk::DependencyFlags::empty(), &[mb], &[], &[]);
            });
        }
        imgs.len()
    }

    /// Guest addresses of the images a shader last bound as storage
    /// (diagnosis: `fillimg cs=<shader>`).
    pub fn written_by(&self, shader: u64) -> Vec<u64> {
        let mut v: Vec<u64> = self.map.values().filter(|i| i.writer == shader).map(|i| i.key.base).collect();
        v.sort_unstable();
        v.dedup();
        v
    }

    /// Uploads the CPU-sourced textures at guest address `base` (all of
    /// them without one) again at their next use (diagnosis: `reupload`);
    /// returns how many.
    pub fn force_reupload(&mut self, base: Option<u64>) -> usize {
        let mut n = 0;
        for img in self.map.values() {
            if img.gpu_written || base.is_some_and(|b| b != img.key.base) {
                continue;
            }
            if let Some(region) = img.source {
                super::track::force_dirty(region);
                n += 1;
            }
        }
        n
    }

    /// Zero-fills every cached image at guest address `base` (diagnosis:
    /// `clearimg` control command); returns how many.
    pub fn clear_at(&mut self, base: u64) -> usize {
        let images: Vec<(vk::Image, vk::Format)> = self.map.values().filter(|i| i.key.base == base).map(|i| (i.image, i.key.format)).collect();
        for &(image, format) in &images {
            self.one_shot(|cmd| unsafe {
                let range = vk::ImageSubresourceRange { aspect_mask: aspects(format), base_mip_level: 0, level_count: vk::REMAINING_MIP_LEVELS, base_array_layer: 0, layer_count: vk::REMAINING_ARRAY_LAYERS };
                let d = &ctx().device;
                let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
                d.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[mb], &[], &[]);
                if is_depth(format) {
                    d.cmd_clear_depth_stencil_image(cmd, image, vk::ImageLayout::GENERAL, &vk::ClearDepthStencilValue { depth: 0.0, stencil: 0 }, &[range]);
                } else if !is_compressed(format) {
                    d.cmd_clear_color_image(cmd, image, vk::ImageLayout::GENERAL, &vk::ClearColorValue::default(), &[range]);
                }
                let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
                d.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::ALL_COMMANDS, vk::DependencyFlags::empty(), &[mb], &[], &[]);
            });
        }
        images.len()
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
    /// A video-memory buffer of at least `size` bytes for GPU detiling: the
    /// smallest free one large enough, else a new one (sizes rounded up to
    /// powers of two up to DETILE_KEEP; larger ones exact and not kept).
    fn detile_take(&mut self, size: u64) -> Result<(vk::Buffer, vk::DeviceMemory, u64), String> {
        if let Some(i) = self.detile_free.iter().enumerate().filter(|(_, b)| b.2 >= size).min_by_key(|(_, b)| b.2).map(|(i, _)| i) {
            return Ok(self.detile_free.swap_remove(i));
        }
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
        let cap = if size > DETILE_KEEP { size } else { size.next_power_of_two().clamp(1 << 20, DETILE_KEEP) };
        let t0 = std::time::Instant::now();
        let (b, m) = ctx().device_buffer(cap, usage)?;
        super::hitch::note_since(super::hitch::Kind::Upload, t0, || format!("a new GPU detiling buffer of {} MiB", cap >> 20));
        GPU_DETILE[9].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok((b, m, cap))
    }

    /// GPU detiling buffers the GPU is done with: kept for reuse (up to
    /// DETILE_POOL in all) or freed.
    fn release_detile(&mut self, v: Vec<(vk::Buffer, vk::DeviceMemory, u64)>) {
        for b in v {
            let kept: u64 = self.detile_free.iter().map(|b| b.2).sum();
            if b.2 <= DETILE_KEEP && kept + b.2 <= DETILE_POOL {
                self.detile_free.push(b);
                continue;
            }
            unsafe {
                ctx().device.destroy_buffer(b.0, None);
                ctx().device.free_memory(b.1, None);
            }
        }
    }

    /// The wanted, mapped (mip, layer)s `parts` of a 2D (array) texture
    /// detiled on the GPU (`tune gpu_detile`): their tiled bytes copied from
    /// the guest-memory import windows into video memory, detiled there into
    /// packed rows (`rdna::kernels::detile`, the CPU detiler's addresses)
    /// and copied into `image` with `part_regions` (one per part, in order;
    /// their buffer offsets are replaced), the unmapped parts of a first
    /// upload (`zeroed`) from a zero-filled area, the bytes read through the
    /// import `windows` (start, end, buffer) made before the guest-reader
    /// lock, which the caller holds (nothing here may take the mapping
    /// table's lock). Submitted without waiting (`submit_held`; the GPU
    /// reads guest memory later, through imports that outlive a remap until
    /// settled, as front end 2's uploads do), or waited for with
    /// `DS_GPU_DETILE_CHECK=1`. Err: nothing was recorded (the caller
    /// detiles on the CPU). Returns the bytes detiled.
    #[allow(clippy::too_many_arguments)]
    fn upload_gpu(&mut self, image: vk::Image, first: bool, lay: &tiling::Layout, t: &Tsharp, block: bool, parts: &[(usize, u32, usize, usize)], zeroed: &[(usize, u32, usize)], part_regions: &[vk::BufferImageCopy], range: &dyn Fn(usize, u32) -> (u64, u64), windows: &[(u64, u64, vk::Buffer)]) -> Result<u64, String> {
        use std::sync::atomic::Ordering::Relaxed;
        use rdna::kernels::{DETILE_LOCAL_SIZE, DETILE_PC_WORDS};
        let t0 = std::time::Instant::now();
        let pipe = detile_pipe()?;
        let pat = tiling::pattern(lay.sw, lay.elem_log2).ok_or("no swizzle pattern")?;
        let mut masks = [0u32; 16];
        for (i, m) in masks.iter_mut().enumerate().take(lay.block_log2 as usize) {
            *m = (pat[i] & 0xFFFF) as u32 | (((pat[i] >> 16) & 0xFFFF) as u32) << 16;
        }
        // The parts' tiled bytes, packed (a layer's mip-tail mips share its
        // tail block).
        let mut tiled: Vec<(u64, u64, u64)> = Vec::new();
        let mut t_of: HashMap<u64, u64> = HashMap::default();
        let mut t_total = 0u64;
        let mut src_at = Vec::with_capacity(parts.len());
        for &(mip, layer, _, _) in parts {
            let (s, n) = range(mip, layer);
            let at = match t_of.get(&s) {
                Some(&at) => at,
                None => {
                    let at = t_total;
                    tiled.push((t.base + s, n, at));
                    t_of.insert(s, at);
                    t_total = (t_total + n).next_multiple_of(256);
                    at
                }
            };
            src_at.push(at);
        }
        if t_total > u32::MAX as u64 {
            return Err(format!("{t_total} tiled bytes"));
        }
        // Their copies from the import windows, split at window boundaries.
        let mut copies: Vec<(vk::Buffer, vk::BufferCopy)> = Vec::new();
        for &(g, n, at) in &tiled {
            let end = g + n;
            let mut x = g;
            while x < end {
                let &(start, wend, buf) = windows.iter().find(|&&(ws, we, _)| ws <= x && x < we).ok_or_else(|| format!("no import window over {x:#x}"))?;
                let e = end.min(wend);
                copies.push((buf, vk::BufferCopy { src_offset: x - start, dst_offset: at + (x - g), size: e - x }));
                x = e;
            }
        }
        // Packed rows at 16-byte offsets (whole dwords per part whatever the
        // element size), then the zero area.
        let mut l_at = Vec::with_capacity(parts.len());
        let mut l_total = 0u64;
        for &(_, _, _, len) in parts {
            l_at.push(l_total);
            l_total = (l_total + len as u64).next_multiple_of(16);
        }
        let zero_len = zeroed.iter().map(|z| z.2 as u64).max().unwrap_or(0).next_multiple_of(16);
        let zero_at = l_total;
        l_total += zero_len;
        let mut regions: Vec<vk::BufferImageCopy> = part_regions.iter().zip(&l_at).map(|(r, &o)| r.buffer_offset(o)).collect();
        for &(mip, layer, _) in zeroed {
            let m = &lay.mips[mip];
            regions.push(
                vk::BufferImageCopy::default()
                    .buffer_offset(zero_at)
                    .buffer_row_length(if block { m.width * 4 } else { m.width })
                    .buffer_image_height(if block { m.height * 4 } else { m.height })
                    .image_subresource(vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: mip as u32, base_array_layer: layer, layer_count: 1 })
                    .image_extent(vk::Extent3D { width: m.texel_w, height: m.texel_h, depth: 1 }),
            );
        }
        // Each part's dispatch: its push constants and grid.
        let (lw, lh) = (lay.block_w.trailing_zeros(), lay.block_h.trailing_zeros());
        let mut dispatches: Vec<(Vec<u8>, u32, u32)> = Vec::with_capacity(parts.len());
        let mut detiled = 0u64;
        for (i, &(mip, _, _, len)) in parts.iter().enumerate() {
            let m = &lay.mips[mip];
            let (tx, ty) = m.tail.unwrap_or((0, 0));
            let count = (len as u64).div_ceil(4);
            let groups = count.div_ceil(DETILE_LOCAL_SIZE as u64).max(1);
            let gx = groups.min(65535);
            let gy = groups.div_ceil(gx);
            let mut pc = [0u32; DETILE_PC_WORDS];
            pc[0] = src_at[i] as u32;
            pc[1] = (l_at[i] / 4) as u32;
            pc[2] = m.width;
            pc[3] = m.height;
            pc[4] = tx;
            pc[5] = ty;
            pc[6] = (m.pitch / lay.block_w).max(1);
            pc[7] = lay.block_log2 | lw << 8 | lh << 16 | lay.elem_log2 << 24;
            pc[8] = count as u32;
            pc[9] = (gx * DETILE_LOCAL_SIZE as u64) as u32;
            pc[10] = t_total as u32;
            pc[12..28].copy_from_slice(&masks);
            dispatches.push((pc.iter().flat_map(|w| w.to_le_bytes()).collect(), gx as u32, gy as u32));
            detiled += len as u64;
        }
        let c = ctx();
        let tbuf = self.detile_take(t_total.max(256))?;
        let lbuf = match self.detile_take(l_total.max(16)) {
            Ok(b) => b,
            Err(e) => {
                self.release_detile(vec![tbuf]);
                return Err(e);
            }
        };
        let (tb, lb) = (tbuf.0, lbuf.0);
        // (Rows, then the tiled bytes.)
        let readback = if gpu_detile_check() {
            match c.host_buffer(l_total.max(16) + t_total.max(256), vk::BufferUsageFlags::TRANSFER_DST) {
                Ok(r) => Some(r),
                Err(e) => {
                    self.release_detile(vec![tbuf, lbuf]);
                    return Err(e);
                }
            }
        } else {
            None
        };
        let rb = readback.as_ref().map(|r| r.0);
        let rows_bytes = l_total.max(16);
        let base = t.base;
        // The transfer from guest memory on the copy queue (beside the
        // frames' work on the main queue), the rest on the main queue once
        // it is done (checked runs wait for that part); all on the main queue
        // without a copy queue.
        let on_copy = self.copy_parts();
        let record_copies = |cmd: vk::CommandBuffer| unsafe {
            let mut k = 0;
            while k < copies.len() {
                let src = copies[k].0;
                let mut regs = Vec::new();
                while k < copies.len() && copies[k].0 == src {
                    regs.push(copies[k].1);
                    k += 1;
                }
                c.device.cmd_copy_buffer(cmd, src, tb, &regs);
            }
        };
        let record = |cmd: vk::CommandBuffer| unsafe {
            super::vk::checkpoint(cmd, super::vk::mark::UPLOAD, base);
            let b = vk::ImageMemoryBarrier::default()
                .old_layout(if first { vk::ImageLayout::UNDEFINED } else { vk::ImageLayout::GENERAL })
                .new_layout(vk::ImageLayout::GENERAL)
                .image(image)
                .subresource_range(vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: vk::REMAINING_MIP_LEVELS, base_array_layer: 0, layer_count: vk::REMAINING_ARRAY_LAYERS })
                .src_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
            c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &[b]);
            if on_copy.is_none() {
                record_copies(cmd);
            }
            if zero_len > 0 {
                c.device.cmd_fill_buffer(cmd, lb, zero_at, zero_len, 0);
            }
            let m = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
            c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &[m], &[], &[]);
            c.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipe.pipeline);
            let ti = [vk::DescriptorBufferInfo { buffer: tb, offset: 0, range: vk::WHOLE_SIZE }];
            let li = [vk::DescriptorBufferInfo { buffer: lb, offset: 0, range: vk::WHOLE_SIZE }];
            let w = [
                vk::WriteDescriptorSet::default().dst_binding(0).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&ti),
                vk::WriteDescriptorSet::default().dst_binding(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&li),
            ];
            c.push_descriptors(cmd, vk::PipelineBindPoint::COMPUTE, pipe.layout, 0, &w);
            for (pc, gx, gy) in &dispatches {
                c.device.cmd_push_constants(cmd, pipe.layout, vk::ShaderStageFlags::COMPUTE, 0, pc);
                c.device.cmd_dispatch(cmd, *gx, *gy, 1);
            }
            let m = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_READ);
            c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[m], &[], &[]);
            if let Some(rb) = rb {
                c.device.cmd_copy_buffer(cmd, lb, rb, &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size: l_total.max(16) }]);
                c.device.cmd_copy_buffer(cmd, tb, rb, &[vk::BufferCopy { src_offset: 0, dst_offset: rows_bytes, size: t_total.max(256) }]);
                let h = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::HOST_READ);
                c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::HOST, vk::DependencyFlags::empty(), &[h], &[], &[]);
            }
            c.device.cmd_copy_buffer_to_image(cmd, lb, image, vk::ImageLayout::GENERAL, &regions);
            let a = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
            c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::ALL_COMMANDS, vk::DependencyFlags::empty(), &[a], &[], &[]);
        };
        // Submitted without waiting (the one queue runs it after the work
        // submitted before and before batches submitted later; its barriers
        // order it against the image's other uses), unless checked.
        if readback.is_some() && on_copy.is_none() {
            self.one_shot(record);
            self.release_detile(vec![tbuf, lbuf]);
        } else if let Some((pool, sem)) = on_copy {
            match self.submit_copy_part(pool, sem, record_copies) {
                Some((x, v)) => {
                    GPU_DETILE[10].fetch_add(1, Relaxed);
                    let held = Held { staging: None, dev: vec![tbuf, lbuf], copy_cmd: Some(x) };
                    if readback.is_some() {
                        self.one_shot_after(Some((sem, v)), record);
                        self.release_held(held);
                    } else {
                        self.submit_held(held, DETILE_INFLIGHT, Some((sem, v)), record);
                    }
                }
                None => {
                    self.release_detile(vec![tbuf, lbuf]);
                    return Err("the copy queue refused the transfer".into());
                }
            }
        } else {
            self.submit_held(Held { staging: None, dev: vec![tbuf, lbuf], copy_cmd: None }, DETILE_INFLIGHT, None, record);
        }
        // DS_GPU_DETILE_CHECK=1: each part against the CPU's detiling of the
        // tiled bytes the GPU copied (laid out at their texture offsets).
        if let Some((rbuf, rmem, rptr)) = readback {
            let got = unsafe { std::slice::from_raw_parts(rptr, l_total as usize) };
            let copied = unsafe { std::slice::from_raw_parts(rptr.add(rows_bytes as usize), t_total as usize) };
            let end = tiled.iter().map(|&(g, n, _)| g + n - t.base).max().unwrap_or(0);
            let mut src = vec![0u8; end as usize];
            let mut changed = 0u64;
            for &(g, n, at) in &tiled {
                let (s, n) = ((g - t.base) as usize, n as usize);
                let mine = &copied[at as usize..at as usize + n];
                src[s..s + n].copy_from_slice(mine);
                if unsafe { std::slice::from_raw_parts(g as *const u8, n) } != mine {
                    changed += 1;
                }
            }
            GPU_DETILE[8].fetch_add(changed, Relaxed);
            let mut differ = 0u64;
            for (i, &(mip, layer, _, len)) in parts.iter().enumerate() {
                let mut want = vec![0u8; len];
                lay.detile_into(&src, mip, layer, &mut want);
                let have = &got[l_at[i] as usize..l_at[i] as usize + len];
                if want != have {
                    differ += 1;
                    if GPU_DETILE[7].load(Relaxed) + differ <= 8 {
                        let at = want.iter().zip(have).position(|(a, b)| a != b).unwrap_or(0);
                        let m = &lay.mips[mip];
                        crate::warn!(
                            "images: GPU detiling of {:#x} {:?} mip {mip} layer {layer} ({}x{} elements of {} bytes, swizzle {}, tail {:?}) differs from the CPU's at byte {at} of {len}",
                            t.base,
                            t.format,
                            m.width,
                            m.height,
                            1u32 << lay.elem_log2,
                            lay.sw,
                            m.tail
                        );
                    }
                }
            }
            GPU_DETILE[6].fetch_add(parts.len() as u64, Relaxed);
            GPU_DETILE[7].fetch_add(differ, Relaxed);
            unsafe {
                c.device.destroy_buffer(rbuf, None);
                c.device.unmap_memory(rmem);
                c.device.free_memory(rmem, None);
            }
        }
        GPU_DETILE[0].fetch_add(1, Relaxed);
        GPU_DETILE[1].fetch_add(parts.len() as u64, Relaxed);
        GPU_DETILE[2].fetch_add(tiled.iter().map(|x| x.1).sum::<u64>(), Relaxed);
        GPU_DETILE[3].fetch_add(detiled, Relaxed);
        GPU_DETILE[5].fetch_add(t0.elapsed().as_nanos() as u64, Relaxed);
        Ok(detiled)
    }

    /// A first upload prepared before the lock (`predetile`): its copy
    /// recorded (into the uploads' shared command buffer, `tune
    /// upload_batch`, else waited for), with the counts and sums the cache's
    /// own path keeps.
    fn upload_prepared(&mut self, image: vk::Image, key: ImageKey, t: &Tsharp, p: Prepared2D) -> Result<(u64, u64), String> {
        let t0 = std::time::Instant::now();
        PREDETILED[3].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        DETILED.fetch_add(p.detiled as u64, std::sync::atomic::Ordering::Relaxed);
        if super::tune::upload_parts() {
            SUB_SUMS.lock().get_or_insert_with(HashMap::default).insert(key, p.sums.clone());
        }
        if p.skipped > 0 {
            self.report(format!("partially resident texture: {} (mip, layer) not mapped, zero-filled", p.skipped));
        }
        let c = ctx();
        let (buf, base, regions) = (p.buf, t.base, &p.regions);
        let record = |cmd: vk::CommandBuffer| unsafe {
            super::vk::checkpoint(cmd, super::vk::mark::UPLOAD, base);
            let b = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::GENERAL)
                .image(image)
                .subresource_range(vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: vk::REMAINING_MIP_LEVELS, base_array_layer: 0, layer_count: vk::REMAINING_ARRAY_LAYERS })
                .src_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
            c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &[b]);
            c.device.cmd_copy_buffer_to_image(cmd, buf, image, vk::ImageLayout::GENERAL, regions);
            let a = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
            c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::ALL_COMMANDS, vk::DependencyFlags::empty(), &[a], &[], &[]);
        };
        let size = p.size;
        let batched = super::tune::upload_batch() && record_batched((p.buf, p.mem), p.total.max(4) as u64, &record).is_ok();
        if !batched {
            self.one_shot(record);
            free_prepared(p);
        }
        self.uploads += 1;
        self.upload_bytes += size;
        self.upload_ns += t0.elapsed().as_nanos() as u64;
        Ok((base, size))
    }

    fn upload(&mut self, image: vk::Image, key: ImageKey, t: &Tsharp) -> Result<(u64, u64), String> {
        self.upload_parts(image, key, t, None, true)
    }

    /// Uploads the mapped (mip, layer)s of a guest texture: all of them with
    /// `all` or without written ranges `only` (and on a first upload, and
    /// for 3D textures); otherwise, with `tune upload_parts`, those that
    /// `only` or the CPU's writes since the last upload (write-watch marks)
    /// overlap.
    fn upload_parts(&mut self, image: vk::Image, key: ImageKey, t: &Tsharp, only: Option<&[(u64, u64)]>, all: bool) -> Result<(u64, u64), String> {
        let t0 = std::time::Instant::now();
        let r = self.upload_inner(image, key, t, only, all);
        if zero_census() {
            let cause = REUPLOAD_CAUSE.with(|c| c.get()) as usize;
            ZERO_CENSUS[cause.min(6)][4].fetch_add(t0.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        }
        if let Ok((_, n)) = &r {
            self.upload_bytes += n;
            super::hitch::note_since(super::hitch::Kind::Upload, t0, || format!("{}x{}x{}{} {:?} {} KiB", key.width, key.height, key.layers, if key.ty3d { " 3D" } else { "" }, key.format, n >> 10));
        }
        self.upload_ns += t0.elapsed().as_nanos() as u64;
        r
    }

    fn upload_inner(&mut self, image: vk::Image, key: ImageKey, t: &Tsharp, only: Option<&[(u64, u64)]>, all: bool) -> Result<(u64, u64), String> {
        let first = !self.map.contains_key(&key);
        let region = self.map.get(&key).and_then(|i| i.source);
        let precise = super::tune::upload_parts() && !first && !all && only.is_some();
        if key.ty3d {
            return self.upload_3d(image, key, t, first);
        }
        let el = elem_log2(t.format).ok_or("unknown element size")?;
        let block = is_block(t.format);
        let layers = if key.ty3d { 1 } else { key.layers };
        let lay = tiling::layout(t.sw_mode, el, key.width, key.height, key.levels, block).ok_or_else(|| format!("swizzle mode {} not supported", t.sw_mode))?;
        if tiling::is_xor(t.sw_mode) {
            self.report(format!("texture with XOR swizzle mode {} uploaded with an assumed pipe count (2^{})", t.sw_mode, tiling::pipes_log2()));
        }
        let size = lay.size(layers);
        // GPU detiling for textures of at least `tune gpu_detile` MiB.
        let gpu = match super::tune::gpu_detile() {
            0 => false,
            n => size >= n << 20,
        };
        // Guest memory where a video-memory home holds newer GPU-written data
        // (`homes`): this upload reads the older bytes (diagnostics).
        if super::homes::any() && super::homes::any_stale(&[(t.base, size)]) {
            let n = super::homes::IMAGE_STALE[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if n < 20 {
                crate::warn!(
                    "images: {} {:#x} {}x{}x{} {:?} uploaded from guest memory where home {} holds GPU-written data not written back",
                    if first { "new image" } else { "image" },
                    t.base,
                    key.width,
                    key.height,
                    layers,
                    t.format,
                    super::homes::describe(t.base, size)
                );
            }
        }
        // Streamed textures are partially resident: the title maps single
        // layers of texture arrays (light cookies), and within a layer the
        // mip tail and the smaller mips (at the start of the chain). Every
        // mapped (mip, layer) is uploaded; on the first upload the others
        // are zero-filled, never left as undefined image memory (which holds
        // whatever the device memory held before). Later mappings or writes
        // re-upload (write tracking).
        let elem = 1u64 << el;
        let range = |mip: usize, layer: u32| -> (u64, u64) {
            let m = &lay.mips[mip];
            let start = layer as u64 * lay.slice_size + if m.tail.is_some() { 0 } else { m.offset };
            (start, if m.tail.is_some() { 1u64 << lay.block_log2 } else { m.pitch as u64 * m.padded_height as u64 * elem })
        };
        // A partial re-upload (`precise`): the (mip, layer)s the written
        // ranges (mappings, file reads) or the CPU's writes since the last
        // take overlap. The title maps a layer of a streamed array at a time
        // and fills mapped layers and small mips with CPU writes later. The
        // marks are taken before the bytes are read (later writes are seen
        // next time); taking resets them, so the copy cache and other
        // textures on those pages are told (`precache::take_written_for`).
        let mut written: Vec<(u64, u64)> = Vec::new();
        let t_take = std::time::Instant::now();
        let precise = precise && {
            let mut pages = Vec::new();
            let ok = super::precache::take_written_for(t.base, size, &mut pages, region, 4).is_some();
            for p in pages {
                match written.last_mut() {
                    Some(r) if r.0 + r.1 == p => r.1 += 4096,
                    _ => written.push((p, 4096)),
                }
            }
            ok
        };
        let take_ms = t_take.elapsed().as_secs_f64() * 1e3;
        // Pages checked and read under the guest-reader lock: the title may
        // unmap layers of a streamed array meanwhile (`mem::GUEST_READERS`).
        // Checked in the mapping table first, before the lock (which must
        // not take the table's), and again with the operating system only
        // when a mapping changed in between (`mem::table_view_held`).
        let t_detile = std::time::Instant::now();
        let g0 = crate::mem::MAP_GEN.load(std::sync::atomic::Ordering::SeqCst);
        let table_whole = super::cp::valid(t.base, size) && crate::mem::host_readable(t.base, size);
        let n_parts = key.levels as usize * layers as usize;
        let table_parts: Vec<bool> = if table_whole {
            Vec::new()
        } else {
            (0..n_parts)
                .map(|i| {
                    let (s, n) = range(i / layers as usize, (i % layers as usize) as u32);
                    super::cp::valid(t.base + s, n) && crate::mem::host_readable(t.base + s, n)
                })
                .collect()
        };
        // GPU detiling: the import windows over the mapped parts, made now
        // (making one takes the mapping table's lock, which the guest-reader
        // lock's holders must not: decommits hold it while they wait for
        // them).
        let mut windows: Vec<(u64, u64, vk::Buffer)> = Vec::new();
        if gpu {
            for i in 0..n_parts {
                if !(table_whole || table_parts[i]) {
                    continue;
                }
                let (s, n) = range(i / layers as usize, (i % layers as usize) as u32);
                let (mut x, end) = (t.base + s, t.base + s + n);
                while x < end {
                    if let Some(&(_, we, _)) = windows.iter().find(|&&(ws, we, _)| ws <= x && x < we) {
                        x = we.min(end);
                        continue;
                    }
                    let e = ((x & !(super::exec::WINDOW - 1)) + super::exec::WINDOW).min(end);
                    match super::exec::import_window(x, e - x) {
                        Some((b, ws, we)) if ws <= x && e <= we => {
                            windows.push((ws, we, b));
                            x = we.min(end);
                        }
                        _ => break,
                    }
                }
            }
        }
        let t_table = std::time::Instant::now();
        let readers = crate::mem::GUEST_READERS.read();
        let t_locked = std::time::Instant::now();
        let held = crate::mem::table_view_held(g0);
        UPLOAD_TABLE[usize::from(!held)].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let whole = if held { table_whole } else { super::cp::valid(t.base, size) && crate::mem::os_readable(t.base, size) };
        let mapped = |mip: usize, layer: u32| -> bool {
            if whole {
                return true;
            }
            if held {
                return table_parts[mip * layers as usize + layer as usize];
            }
            let (s, n) = range(mip, layer);
            super::cp::valid(t.base + s, n) && crate::mem::os_readable(t.base + s, n)
        };
        let candidate = |mip: usize, layer: u32| -> bool {
            if !precise {
                return true;
            }
            let (s, n) = range(mip, layer);
            let (a, e) = (t.base + s, t.base + s + n);
            // `written` is ascending and disjoint.
            let i = written.partition_point(|&(pb, pl)| pb + pl <= a);
            written.get(i).is_some_and(|&(pb, _)| pb < e) || only.is_some_and(|parts| parts.iter().any(|&(pb, pl)| pb < e && a < pb.saturating_add(pl)))
        };
        // Of the candidates, those whose bytes changed since their last
        // upload (hashes kept per (mip, layer)): the title rewrites texture
        // memory with the same data (16,143 of 16,146 parts re-uploaded after
        // CPU writes were unchanged, run 20261013-082344). Mapping is checked
        // for candidates only (one memory query each; a streamed array has
        // thousands of (mip, layer)s, most of them not candidates).
        let sums_on = super::tune::upload_parts();
        let n_sub = key.levels as usize * layers as usize;
        let mut sums: Vec<u64> = if sums_on {
            SUB_SUMS.lock().as_ref().and_then(|m| m.get(&key).cloned()).filter(|v| v.len() == n_sub).unwrap_or_else(|| vec![SUB_NONE; n_sub])
        } else {
            Vec::new()
        };
        let t_query = std::time::Instant::now();
        // A re-upload's candidates are hashed first, on several threads when
        // they are large (one by one, a 1,200-part array's re-upload spent
        // 50 ms here).
        let mut pre_hash: Vec<u64> = Vec::new();
        if sums_on && !first {
            let todo: Vec<(usize, u64, u64)> = (0..n_sub)
                .filter_map(|i| {
                    let (mip, layer) = (i / layers as usize, (i % layers as usize) as u32);
                    if !candidate(mip, layer) || !mapped(mip, layer) {
                        return None;
                    }
                    let (s, n) = range(mip, layer);
                    Some((i, t.base + s, n))
                })
                .collect();
            let bytes: u64 = todo.iter().map(|x| x.2).sum();
            pre_hash = vec![SUB_NONE; n_sub];
            let threads = if bytes >= 1 << 20 { super::pool::threads() } else { 1 };
            for (i, h) in hash_parts(&todo, threads) {
                pre_hash[i] = h;
            }
        }
        let mut want = vec![false; n_sub];
        let mut use_mapped = vec![false; n_sub];
        let mut end = 0u64;
        for mip in 0..key.levels as usize {
            for layer in 0..layers {
                let i = mip * layers as usize + layer as usize;
                if !candidate(mip, layer) {
                    continue;
                }
                if !mapped(mip, layer) {
                    want[i] = true;
                    if sums_on {
                        sums[i] = SUB_UNMAPPED;
                    }
                    continue;
                }
                let (s, n) = range(mip, layer);
                // (A first upload hashes its parts after the detiling, on
                // several threads: one by one here it cost as long as
                // detiling a large array.)
                if sums_on && !first {
                    let h = pre_hash[i];
                    if precise && sums[i] == h {
                        SUB_SAME.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        continue;
                    }
                    sums[i] = h;
                }
                want[i] = true;
                use_mapped[i] = true;
                end = end.max(s + n);
            }
        }
        UPLOAD_SPLIT[3].fetch_add(t_query.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        let t_chosen = std::time::Instant::now();
        let wanted = |mip: usize, layer: u32| want[mip * layers as usize + layer as usize];
        if upload_check() {
            let now: Vec<u64> = (0..key.levels as usize)
                .flat_map(|mip| (0..layers).map(move |layer| (mip, layer)))
                .map(|(mip, layer)| {
                    let (s, n) = range(mip, layer);
                    if mapped(mip, layer) { hash_bytes(t.base + s, n as usize) } else { SUB_UNMAPPED }
                })
                .collect();
            let mut g = SUB_HASHES.lock();
            let map = g.get_or_insert_with(HashMap::default);
            if let (true, Some((at, prev))) = (precise, map.get(&key)) {
                for (i, (&a, &b)) in prev.iter().zip(&now).enumerate() {
                    let (mip, layer) = (i / layers as usize, (i % layers as usize) as u32);
                    if wanted(mip, layer) && a != SUB_UNMAPPED && b != SUB_UNMAPPED {
                        // Uploaded again: did its bytes change?
                        SUB_CHECKS[2].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if a == b {
                            SUB_CHECKS[3].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        }
                    }
                    if wanted(mip, layer) || b == SUB_UNMAPPED {
                        continue;
                    }
                    // Written since this upload's take: uploaded next time.
                    let (s, n) = range(mip, layer);
                    if region.is_some_and(|r| super::track::pending_overlaps(r, t.base + s, n)) {
                        continue;
                    }
                    SUB_CHECKS[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if a != b && SUB_CHECKS[1].fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 20 {
                        let ev = super::track::events_since(t.base + s, n, *at);
                        let takes = region.map(|r| super::track::takes_since(r, *at)).unwrap_or_default();
                        crate::warn!(
                            "upload check: texture {:#x} {}x{}x{} mip {mip} layer {layer} ({:#x}+{n:#x}): changed although no written range overlaps it (ranges {only:x?}); last upload {} ms ago; writes noted on it since: {}; region {region:?} taken since: {}",
                            t.base, key.width, key.height, layers, t.base + s, at.elapsed().as_millis(), if ev.is_empty() { "none".to_string() } else { ev.join("; ") }, takes.join(" | ")
                        );
                    }
                }
            }
            map.insert(key, (std::time::Instant::now(), now));
        }
        if end == 0 && precise {
            // Nothing mapped changed.
            if sums_on {
                SUB_SUMS.lock().get_or_insert_with(HashMap::default).insert(key, sums);
            }
            PART_UPLOADS[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            PART_UPLOADS[2].fetch_add(key.levels as u64 * layers as u64, std::sync::atomic::Ordering::Relaxed);
            return Ok((t.base, size));
        }
        if end == 0 && !first {
            return Err(format!("texture {:#x}+{size:#x} not mapped", t.base));
        }
        // Detiling reads only mapped regions (holes between them are not read).
        let src = unsafe { std::slice::from_raw_parts(t.base as *const u8, end as usize) };
        let mut skipped = 0;
        // Every wanted, mapped (mip, layer) is detiled straight into the
        // staging buffer at its offset (sizes first: no intermediate copies,
        // which took as long as the detiling, FINDINGS.md 2026-10-23);
        // unmapped ones of a first upload copy from one shared zero area at
        // the end.
        let mut regions = Vec::new();
        let mut zeroed: Vec<(usize, u32, usize)> = Vec::new();
        let mut parts: Vec<(usize, u32, usize, usize)> = Vec::new();
        let mut total = 0usize;
        let (mut parts_done, mut parts_left) = (0u64, 0u64);
        for mip in 0..key.levels as usize {
            for layer in 0..layers {
                if !wanted(mip, layer) {
                    parts_left += 1;
                    continue;
                }
                parts_done += 1;
                if !use_mapped[mip * layers as usize + layer as usize] {
                    skipped += 1;
                    if first {
                        let m = &lay.mips[mip];
                        let bytes = m.width as usize * m.height as usize * elem as usize;
                        zeroed.push((mip, layer, bytes));
                    }
                    continue;
                }
                let m = &lay.mips[mip];
                let len = m.width as usize * m.height as usize * elem as usize;
                let off = total as u64;
                parts.push((mip, layer, total, len));
                total += len;
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
        if regions.is_empty() && zeroed.is_empty() {
            return Err(format!("texture {:#x}+{size:#x}: no mip mapped", t.base));
        }
        if zero_census() && !first {
            use std::sync::atomic::Ordering::Relaxed;
            let cause = (REUPLOAD_CAUSE.with(|c| c.get()) as usize).min(6);
            for &(mip, layer, _, len) in &parts {
                let (s, n) = range(mip, layer);
                let zero = ((s + n) as usize) <= src.len() && src[s as usize..(s + n) as usize].chunks(8).all(|c| c.iter().all(|&b| b == 0));
                ZERO_CENSUS[cause][0].fetch_add(1, Relaxed);
                ZERO_CENSUS[cause][2].fetch_add(len as u64, Relaxed);
                if zero {
                    ZERO_CENSUS[cause][1].fetch_add(1, Relaxed);
                    ZERO_CENSUS[cause][3].fetch_add(len as u64, Relaxed);
                }
            }
        }
        let zero_len = zeroed.iter().map(|z| z.2).max().unwrap_or(0);
        let (detiled, zero_off) = (total, total);
        total += zero_len;
        // (Only while no mapping changed since the windows were made.)
        if gpu && held {
            match self.upload_gpu(image, first, &lay, t, block, &parts, &zeroed, &regions, &range, &windows) {
                Ok(bytes) => {
                    // A first upload's sums, as the CPU path makes them (a
                    // part rewritten with the same bytes is left out later).
                    if sums_on && first {
                        let todo: Vec<(usize, u64, u64)> = parts
                            .iter()
                            .map(|&(mip, layer, _, _)| {
                                let (s, n) = range(mip, layer);
                                (mip * layers as usize + layer as usize, t.base + s, n)
                            })
                            .collect();
                        let threads = if bytes >= 1 << 20 { super::pool::threads() } else { 1 };
                        for (i, h) in hash_parts(&todo, threads) {
                            sums[i] = h;
                        }
                    }
                    drop(readers);
                    if sums_on {
                        SUB_SUMS.lock().get_or_insert_with(HashMap::default).insert(key, sums);
                    }
                    UPLOAD_SPLIT[0].fetch_add(t_detile.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
                    DETILED.fetch_add(bytes, std::sync::atomic::Ordering::Relaxed);
                    if precise {
                        use std::sync::atomic::Ordering::Relaxed;
                        PART_UPLOADS[0].fetch_add(1, Relaxed);
                        PART_UPLOADS[1].fetch_add(parts_done, Relaxed);
                        PART_UPLOADS[2].fetch_add(parts_left, Relaxed);
                        PART_UPLOADS[3].fetch_add(bytes, Relaxed);
                    }
                    if skipped > 0 {
                        self.report(format!("partially resident texture: {skipped} (mip, layer) not mapped, {}", if first { "zero-filled" } else { "left as they were" }));
                    }
                    self.uploads += 1;
                    return Ok((t.base, size));
                }
                Err(e) => {
                    if GPU_DETILE[4].fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 8 {
                        crate::warn!("images: texture {:#x} {}x{}x{} {:?} detiled on the CPU: {e}", t.base, key.width, key.height, layers, t.format);
                    }
                }
            }
        }
        // (`tune upload_copy`: the staging buffer's copy on the copy queue,
        // not waited for.)
        let via_copy = if super::tune::upload_copy() { self.copy_parts() } else { None };
        let asynchronous = super::tune::upload_async() || via_copy.is_some();
        let st = if asynchronous {
            self.staging(total.max(4) as u64)?
        } else {
            // Uploads waited for get a staging buffer of their own, freed
            // after them: one kept across uploads made the driver refuse to
            // import guest memory windows thousands of times a session
            // (2026-10-24: 3,000-75,000 failed imports against 48, then
            // unmapped textures and device losses at cold boots; the pooled
            // buffers of `tune upload_async` likewise).
            let (buf, mem, ptr) = ctx().host_buffer(total.max(4) as u64, vk::BufferUsageFlags::TRANSFER_SRC)?;
            Staging { buf, mem, ptr, cap: total.max(4) as u64 }
        };
        let t_alloc = std::time::Instant::now();
        let dst = unsafe { std::slice::from_raw_parts_mut(st.ptr, total) };
        // Parts of under 1 MiB (array layers, small mips) are detiled on
        // several threads at once (`detile_into` splits larger ones itself):
        // a 360-layer array of 288 KiB layers took 91 ms one part at a time.
        let small: Vec<(usize, u32, usize, usize)> = parts.iter().copied().filter(|p| p.3 < 1 << 20).collect();
        let small_bytes: usize = small.iter().map(|p| p.3).sum();
        for &(mip, layer, off, len) in &parts {
            if len >= 1 << 20 || small_bytes < 1 << 20 {
                lay.detile_into(src, mip, layer, &mut dst[off..off + len]);
            }
        }
        if small_bytes >= 1 << 20 {
            let threads = super::pool::threads();
            // Disjoint destination slices, handed out in order of offset.
            let mut rest: &mut [u8] = &mut dst[..];
            let mut base = 0usize;
            let mut jobs: Vec<Vec<(usize, u32, &mut [u8])>> = (0..threads).map(|_| Vec::new()).collect();
            let mut sorted = small;
            sorted.sort_by_key(|p| p.2);
            for (k, &(mip, layer, off, len)) in sorted.iter().enumerate() {
                let (_, tail) = std::mem::take(&mut rest).split_at_mut(off - base);
                let (mine, after) = tail.split_at_mut(len);
                rest = after;
                base = off + len;
                jobs[k % threads].push((mip, layer, mine));
            }
            let lay = &lay;
            let jobs: Vec<parking_lot::Mutex<Vec<(usize, u32, &mut [u8])>>> = jobs.into_iter().map(parking_lot::Mutex::new).collect();
            super::pool::for_each(jobs.len(), &|k| {
                for (mip, layer, out) in jobs[k].lock().iter_mut() {
                    lay.detile_into(src, *mip, *layer, out);
                }
            });
        }
        let t_detiled = std::time::Instant::now();
        dst[zero_off..].fill(0);
        // A first upload's sums (for later partial re-uploads), hashed here
        // on several threads rather than one by one before the detiling.
        if sums_on && first {
            let todo: Vec<(usize, u64, u64)> = parts
                .iter()
                .map(|&(mip, layer, _, _)| {
                    let (s, n) = range(mip, layer);
                    (mip * layers as usize + layer as usize, t.base + s, n)
                })
                .collect();
            let threads = if detiled >= 1 << 20 { super::pool::threads() } else { 1 };
            for (i, h) in hash_parts(&todo, threads) {
                sums[i] = h;
            }
        }
        // Guest memory is no longer read.
        drop(readers);
        UPLOAD_SPLIT[0].fetch_add(t_detile.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        DETILED.fetch_add(detiled as u64, std::sync::atomic::Ordering::Relaxed);
        if sums_on {
            SUB_SUMS.lock().get_or_insert_with(HashMap::default).insert(key, sums);
        }
        if precise {
            use std::sync::atomic::Ordering::Relaxed;
            PART_UPLOADS[0].fetch_add(1, Relaxed);
            PART_UPLOADS[1].fetch_add(parts_done, Relaxed);
            PART_UPLOADS[2].fetch_add(parts_left, Relaxed);
            PART_UPLOADS[3].fetch_add(detiled as u64, Relaxed);
        }
        if zero_len > 0 {
            let off = zero_off as u64;
            for &(mip, layer, _) in &zeroed {
                let m = &lay.mips[mip];
                regions.push(
                    vk::BufferImageCopy::default()
                        .buffer_offset(off)
                        .buffer_row_length(if block { m.width * 4 } else { m.width })
                        .buffer_image_height(if block { m.height * 4 } else { m.height })
                        .image_subresource(vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: mip as u32, base_array_layer: layer, layer_count: 1 })
                        .image_extent(vk::Extent3D { width: m.texel_w, height: m.texel_h, depth: 1 }),
                );
            }
        }
        if skipped > 0 {
            self.report(format!(
                "partially resident texture: {skipped} (mip, layer) not mapped, {}",
                if first { "zero-filled" } else { "left as they were" }
            ));
        }
        // (`DS_PREDETILE_CHECK`: a prepared upload compared with this one.)
        if first {
            check_prepared(&key, total, &regions, dst);
        }
        let c = ctx();
        let t_stage = std::time::Instant::now();
        let (buf, mem) = (st.buf, st.mem);
        let base = t.base;
        let t_copy = std::time::Instant::now();
        UPLOAD_SPLIT[1].fetch_add((t_copy - t_stage).as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        let record = |cmd: vk::CommandBuffer| unsafe {
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
        };
        // Through the copy queue: the staging buffer copied there into video
        // memory, from which the main queue copies it into the image once
        // that is done (the same regions).
        if let Some((pool, sem)) = via_copy {
            let bytes = (total.max(4) as u64).next_multiple_of(4);
            if let Ok(dev) = self.detile_take(bytes) {
                let lb = dev.0;
                let to_dev = |x: vk::CommandBuffer| unsafe { c.device.cmd_copy_buffer(x, buf, lb, &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size: total.max(4) as u64 }]) };
                if let Some((x, v)) = self.submit_copy_part(pool, sem, to_dev) {
                    let record_dev = |cmd: vk::CommandBuffer| unsafe {
                        super::vk::checkpoint(cmd, super::vk::mark::UPLOAD, base);
                        let b = vk::ImageMemoryBarrier::default()
                            .old_layout(if first { vk::ImageLayout::UNDEFINED } else { vk::ImageLayout::GENERAL })
                            .new_layout(vk::ImageLayout::GENERAL)
                            .image(image)
                            .subresource_range(vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: vk::REMAINING_MIP_LEVELS, base_array_layer: 0, layer_count: vk::REMAINING_ARRAY_LAYERS })
                            .src_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)
                            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
                        c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &[b]);
                        c.device.cmd_copy_buffer_to_image(cmd, lb, image, vk::ImageLayout::GENERAL, &regions);
                        let a = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
                        c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::ALL_COMMANDS, vk::DependencyFlags::empty(), &[a], &[], &[]);
                    };
                    self.submit_held(Held { staging: Some(st), dev: vec![dev], copy_cmd: Some(x) }, DETILE_INFLIGHT, Some((sem, v)), record_dev);
                    COPY_UPLOADS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    UPLOAD_SPLIT[2].fetch_add(t_copy.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
                    self.uploads += 1;
                    return Ok((t.base, size));
                }
                self.release_detile(vec![dev]);
            }
        }
        if asynchronous {
            self.submit_upload(st, record);
            UPLOAD_SPLIT[2].fetch_add(t_copy.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
            self.uploads += 1;
            return Ok((t.base, size));
        }
        // (`tune upload_batch`: into the command buffer the uploads share,
        // not waited for; its own staging freed once that completes.)
        if super::tune::upload_batch() {
            match record_batched((st.buf, st.mem), total.max(4) as u64, &record) {
                Ok(()) => {
                    UPLOAD_SPLIT[2].fetch_add(t_copy.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
                    self.uploads += 1;
                    return Ok((t.base, size));
                }
                Err(e) => {
                    static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
                    if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                        crate::warn!("images: upload not batched: {e}");
                    }
                }
            }
        }
        self.one_shot(record);
        let t_free = std::time::Instant::now();
        UPLOAD_SPLIT[2].fetch_add((t_free - t_copy).as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        // The GPU is done with it (one_shot waited).
        let _ = (buf, mem);
        unsafe {
            c.device.destroy_buffer(st.buf, None);
            c.device.unmap_memory(st.mem);
            c.device.free_memory(st.mem, None);
        }
        UPLOAD_SPLIT[1].fetch_add(t_free.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        // Uploads of 50 ms or more: where their time went (diagnosis).
        let ms = |a: std::time::Instant, b: std::time::Instant| (b - a).as_secs_f64() * 1e3;
        let now = std::time::Instant::now();
        if ms(t_detile, now) >= 50.0 {
            crate::info!(
                "images: upload of {:#x} {}x{}x{} {:?} ({} parts, {} KiB) took {:.0} ms: write marks {take_ms:.0}, choosing {:.0} (table {:.0}, reader lock {:.0}, the rest: queries, hashes), staging buffer {:.0}, detiling {:.0}, transfer {:.0}, freeing {:.0}",
                t.base,
                key.width,
                key.height,
                layers,
                t.format,
                parts.len(),
                detiled >> 10,
                ms(t_detile, now),
                ms(t_detile, t_chosen),
                ms(t_detile, t_table),
                ms(t_table, t_locked),
                ms(t_chosen, t_alloc),
                ms(t_alloc, t_detiled),
                ms(t_copy, t_free),
                ms(t_free, now)
            );
        }
        self.uploads += 1;
        Ok((t.base, size))
    }

    /// Uploads a 3D texture (one mip, thick standard swizzle).
    fn upload_3d(&mut self, image: vk::Image, key: ImageKey, t: &Tsharp, first: bool) -> Result<(u64, u64), String> {
        if !tiling::is_thick_3d(t.sw_mode) || key.levels > 1 || is_block(t.format) {
            return Err(format!("3D texture layout not implemented (swizzle mode {}, {} mips)", t.sw_mode, key.levels));
        }
        let el = elem_log2(t.format).ok_or("unknown element size")?;
        let lay = tiling::layout_3d(t.sw_mode, el, key.width, key.height, key.layers).ok_or_else(|| format!("3D swizzle mode {} not supported", t.sw_mode))?;
        let size = lay.size();
        // As for 2D textures: the mapping table before the lock, the
        // operating system after it only when a mapping changed meanwhile.
        let g0 = crate::mem::MAP_GEN.load(std::sync::atomic::Ordering::SeqCst);
        let table = super::cp::valid(t.base, size) && crate::mem::host_readable(t.base, size);
        let readers = crate::mem::GUEST_READERS.read();
        let held = crate::mem::table_view_held(g0);
        UPLOAD_TABLE[usize::from(!held)].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let ok = if held { table } else { super::cp::valid(t.base, size) && crate::mem::os_readable(t.base, size) };
        if !ok {
            return Err(format!("3D texture {:#x}+{size:#x} not mapped", t.base));
        }
        let src = unsafe { std::slice::from_raw_parts(t.base as *const u8, size as usize) };
        // Detiled straight into the staging buffer (no intermediate copy).
        let c = ctx();
        let len = lay.width as usize * lay.height as usize * lay.depth as usize * (1usize << el);
        let (buf, mem, ptr) = c.host_buffer(len.max(4) as u64, vk::BufferUsageFlags::TRANSFER_SRC)?;
        lay.detile_into(src, unsafe { std::slice::from_raw_parts_mut(ptr, len) });
        drop(readers);
        let region = vk::BufferImageCopy::default()
            .image_subresource(vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 })
            .image_extent(vk::Extent3D { width: key.width, height: key.height, depth: key.layers });
        let base = t.base;
        self.one_shot(|cmd| unsafe {
            super::vk::checkpoint(cmd, super::vk::mark::UPLOAD, base);
            let b = vk::ImageMemoryBarrier::default()
                .old_layout(if first { vk::ImageLayout::UNDEFINED } else { vk::ImageLayout::GENERAL })
                .new_layout(vk::ImageLayout::GENERAL)
                .image(image)
                .subresource_range(vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: vk::REMAINING_MIP_LEVELS, base_array_layer: 0, layer_count: vk::REMAINING_ARRAY_LAYERS })
                .src_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
            c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &[b]);
            c.device.cmd_copy_buffer_to_image(cmd, buf, image, vk::ImageLayout::GENERAL, &[region]);
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

    /// A 1x1 zero image view of a dimension (unused image table entries).
    pub fn dummy_view(&mut self, dim: rdna::translate::ImageDim) -> Result<vk::ImageView, String> {
        use rdna::translate::ImageDim as D;
        let (vt, layers, ty3d) = match dim {
            D::D1 | D::D2 | D::D2Msaa => (vk::ImageViewType::TYPE_2D, 1, false),
            D::D1Array | D::D2Array | D::D2MsaaArray => (vk::ImageViewType::TYPE_2D_ARRAY, 1, false),
            D::Cube => (vk::ImageViewType::CUBE, 6, false),
            D::CubeArray => (vk::ImageViewType::CUBE_ARRAY, 6, false),
            D::D3 => (vk::ImageViewType::TYPE_3D, 1, true),
        };
        if let Some(&v) = self.dummies.get(&vt) {
            return Ok(v);
        }
        let key = ImageKey { base: 0, format: vk::Format::R8G8B8A8_UNORM, width: 1, height: 1, layers, levels: 1, ty3d, cube: layers == 6 };
        let (image, _, _) = self.create(key)?;
        self.to_general(image, key.format);
        let ci = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vt)
            .format(key.format)
            .subresource_range(vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: if ty3d { 1 } else { layers } });
        let v = unsafe { ctx().device.create_image_view(&ci, None) }.map_err(|e| format!("dummy view {vt:?}: {e}"))?;
        self.dummies.insert(vt, v);
        Ok(v)
    }

    /// Key of the image a T# describes.
    pub fn key_of(t: &Tsharp) -> Option<ImageKey> {
        let format = vk_format(t.format)?;
        // The resource's mip count (T# MAX_MIP), not the view's range: a
        // view may name levels past it, which the hardware clamps (the
        // particle lighting atlas's writers of levels 1-3 bind levels
        // m..m+3 of its 4-level resource). Keyed by their range, each such
        // writer got an image of its own that the atlas's readers never
        // saw: smoke lit from those levels came out black (4-2's fog).
        // A zero MAX_MIP with a longer view keeps the view's count (no such
        // descriptor seen: possibly a MAX_MIP not filled in).
        let levels = if t.max_mip > 0 { t.max_mip + 1 } else { (t.last_level + 1).max(1) };
        let (ty3d, cube) = (t.ty == 10, t.ty == 11);
        let layers = match t.ty {
            10 | 11 | 12 | 13 | 15 => t.depth.max(1),
            _ => 1,
        };
        Some(ImageKey { base: t.base, format, width: t.width, height: t.height, layers, levels, ty3d, cube })
    }

    /// The depth image at `base` (size, format family, mips) with the most
    /// layers, when it has fewer than `layers` and none has enough. Depth
    /// views name only their slice range (DB_DEPTH_VIEW), so a shadow array
    /// rendered slice by slice, or sampled whole, makes its image grow
    /// (`grow_depth`) instead of splitting into one image per view.
    pub fn depth_to_grow(&self, base: u64, width: u32, height: u32, depths: &[vk::Format], layers: u32) -> Option<ImageKey> {
        let same = |k: &ImageKey| k.width == width && k.height == height && !k.ty3d && depths.contains(&k.format);
        let at = self.by_base.get(&base)?;
        if at.iter().any(|k| same(k) && k.layers >= layers) {
            return None;
        }
        at.iter().filter(|k| same(k) && k.layers < layers && self.map.get(k).is_some_and(|i| i.gpu_written)).copied().max_by_key(|k| k.layers)
    }

    /// Depth formats a sampled texture of `format` can read (R32 float over
    /// D32, R16 unorm over D16), or a depth surface's own format.
    pub fn depth_family(format: vk::Format) -> &'static [vk::Format] {
        use vk::Format as F;
        match format {
            F::R32_SFLOAT | F::D32_SFLOAT | F::D32_SFLOAT_S8_UINT => &[F::D32_SFLOAT, F::D32_SFLOAT_S8_UINT],
            F::R16_UNORM | F::D16_UNORM | F::D16_UNORM_S8_UINT => &[F::D16_UNORM, F::D16_UNORM_S8_UINT],
            _ => &[],
        }
    }

    /// Replaces depth image `old` by one with `layers` layers that holds its
    /// contents (layers above stay cleared). The caller has completed the
    /// work that used `old`.
    pub fn grow_depth(&mut self, old: ImageKey, layers: u32) -> Result<ImageKey, String> {
        let new_key = ImageKey { layers, ..old };
        let (image, mem, bytes) = self.create(new_key)?;
        self.to_general(image, new_key.format);
        let Some(prev) = self.map_remove(&old) else { return Err("grow_depth: image gone".into()) };
        let aspect = aspects(old.format);
        self.one_shot(|cmd| unsafe {
            let d = &ctx().device;
            let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::MEMORY_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE);
            d.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[mb], &[], &[]);
            let regions: Vec<vk::ImageCopy> = (0..old.levels)
                .map(|m| {
                    let sub = vk::ImageSubresourceLayers { aspect_mask: aspect, mip_level: m, base_array_layer: 0, layer_count: old.layers };
                    vk::ImageCopy { src_subresource: sub, src_offset: vk::Offset3D::default(), dst_subresource: sub, dst_offset: vk::Offset3D::default(), extent: vk::Extent3D { width: (old.width >> m).max(1), height: (old.height >> m).max(1), depth: 1 } }
                })
                .collect();
            d.cmd_copy_image(cmd, prev.image, vk::ImageLayout::GENERAL, image, vk::ImageLayout::GENERAL, &regions);
            let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
            d.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::ALL_COMMANDS, vk::DependencyFlags::empty(), &[mb], &[], &[]);
        });
        self.map_insert(new_key, Image { image, key: new_key, views: HashMap::default(), gpu_written: true, source: None, hashes: [0; SLICES], checked_frame: 0, write_gen: prev.write_gen, block_gen: 0, span: 0, layout: None, overwritten: false, writer: 0, mem, bytes, used: UseStamp::new(super::FRAMES.load(std::sync::atomic::Ordering::Relaxed)) });
        for v in self.stencil_of.values_mut() {
            if *v == old {
                *v = new_key;
            }
        }
        for v in self.meta_of.values_mut() {
            if v.0 == old {
                v.0 = new_key;
            }
        }
        if let Some(n) = self.pending_clears.remove(&old) {
            self.pending_clears.insert(new_key, n);
            self.pending_change += 1;
        }
        self.note_gen += 1;
        NOTE_GEN.fetch_add(1, std::sync::atomic::Ordering::Release);
        // Destroyed once the batches that may name it are done.
        super::exec::bury_images(vec![Evicted { image: prev.image, mem: prev.mem, views: prev.views.values().copied().collect() }]);
        self.depth_grows += 1;
        if self.depth_grows <= 16 {
            crate::info!("images: depth image {:#x} {}x{} {:?} grown from {} to {} layers", old.base, old.width, old.height, old.format, old.layers, layers);
        }
        Ok(new_key)
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
        // The images at this address (it scanned the whole cache per lookup).
        self.by_base
            .get(&key.base)?
            .iter()
            .filter(|k| {
                self.map.get(k).is_some_and(|i| i.gpu_written) && k.width == key.width && k.height == key.height && k.layers >= key.layers && !k.ty3d && !key.ty3d && depths.contains(&k.format)
            })
            .copied()
            .max_by_key(|k| k.layers)
    }

    /// The cached image a texture of `key` uses, and how it was found. A
    /// GPU-written image of the same address and size (other format,
    /// rendered depth or stencil) holds newer data than guest memory: render
    /// targets are not written back, so an upload made before the surface
    /// was first rendered (e.g. the depth buffer read at boot) is stale.
    /// Whether binding `t` may read its guest memory now: a texture not cached
    /// yet or one uploaded from guest memory (refreshed when written there).
    /// Images the GPU owns are read from their Vulkan image; buffer writes
    /// over them are reloaded separately (`reload_overwritten`, which waits
    /// for the writers itself).
    pub fn reads_guest(&self, t: &Tsharp) -> bool {
        let Some(key) = Self::key_of(t) else { return true };
        match self.lookup(&key, false) {
            Some((k, _)) => !self.map.get(&k).is_some_and(|i| i.gpu_written),
            None => true,
        }
    }

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

    /// Block copies pending for the block-compressed textures among `keys`:
    /// the title transcodes streamed textures with compute shaders that store
    /// each 16- or 8-byte block as one R32G32B32A32 / R32G32 texel of a view
    /// whose size is the block grid. Images here do not share memory, so the
    /// blocks are copied into the texture (size-compatible formats) once per
    /// write generation. Mips with partial blocks are left out.
    /// Whether `take_block_copies` may find something for `keys`
    /// (block-compressed images whose guest base GPU work wrote).
    pub fn block_copies_possible(&self, keys: &[ImageKey]) -> bool {
        !self.gpu_bases.is_empty() && keys.iter().any(|k| is_compressed(k.format) && !k.ty3d && self.gpu_bases.contains(&k.base))
    }

    pub fn take_block_copies(&mut self, keys: &[ImageKey]) -> Vec<BlockCopy> {
        let out = self.block_copies(keys);
        for c in &out {
            if let Some(i) = self.map.get_mut(&c.key) {
                i.block_gen = c.seq;
            }
        }
        out
    }

    /// Whether `depth_copies` may find something for `keys` (depth
    /// attachments over memory GPU work wrote through a color view).
    pub fn depth_feed_possible(&self, keys: &[ImageKey]) -> bool {
        !self.depth_feeds.is_empty() && keys.iter().any(|k| depth_feed_format(k.format).is_some() && self.depth_feeds.contains(&k.base))
    }

    /// Depth images among `keys` whose depth plane GPU work stored through a
    /// color view of the same address and size since it was last copied in:
    /// the title builds its half-resolution depth with a compute shader
    /// (R32 float storage), then depth-tests particles against that memory
    /// as a D32 surface (after a full-screen HTILE resummarize, which has no
    /// counterpart here). Images here do not share memory, so the D32 image
    /// kept its clear (0, the far plane of a reversed-Z depth) and every
    /// particle passed the test: fire and embers showed through walls.
    /// Once per write generation of the source (its storage bindings), so
    /// depth the attachment's own draws write afterwards is kept.
    pub fn depth_copies(&self, keys: &[ImageKey]) -> Vec<DepthCopy> {
        let mut out: Vec<DepthCopy> = Vec::new();
        for k in keys {
            let Some(feed) = depth_feed_format(k.format) else { continue };
            if k.ty3d || !self.depth_feeds.contains(&k.base) || out.iter().any(|c| c.key == *k) {
                continue;
            }
            let Some(dst) = self.map.get(k) else { continue };
            let src = self
                .by_base
                .get(&k.base)
                .into_iter()
                .flatten()
                .filter_map(|kk| self.map.get(kk))
                .filter(|i| i.gpu_written && i.key.format == feed && !i.key.ty3d && i.key.width == k.width && i.key.height == k.height && i.write_gen > dst.block_gen)
                .max_by_key(|i| i.write_gen);
            let Some(src) = src else { continue };
            out.push(DepthCopy {
                key: *k,
                dst: dst.image,
                src: src.image,
                width: k.width,
                height: k.height,
                layers: k.layers.min(src.key.layers).max(1),
                texel: if feed == vk::Format::R32_SFLOAT { 4 } else { 2 },
                seq: src.write_gen,
            });
        }
        out
    }

    pub fn take_depth_copies(&mut self, keys: &[ImageKey]) -> Vec<DepthCopy> {
        let out = self.depth_copies(keys);
        for c in &out {
            if let Some(i) = self.map.get_mut(&c.key) {
                i.block_gen = c.seq;
            }
        }
        out
    }

    /// The mip tail of block-compressed texture `k` stored through a view
    /// of the tail block (`BlockCopy::tail`), newer than its last copy: the
    /// view, its write generation and the tail mips with whole blocks.
    fn tail_blocks(&self, k: &ImageKey, dst: &Image, bytes: u32) -> Option<(vk::Image, u64, Vec<(u32, u32, u32, u32, u32)>)> {
        let t = dst.layout.as_ref()?;
        let el = elem_log2(t.format)?;
        let lay = tiling::layout(t.sw_mode, el, k.width, k.height, k.levels, true)?;
        let regions: Vec<(u32, u32, u32, u32, u32)> = lay
            .mips
            .iter()
            .enumerate()
            .filter(|(_, m)| m.texel_w % 4 == 0 && m.texel_h % 4 == 0)
            .filter_map(|(i, m)| m.tail.map(|(x, y)| (i as u32, x, y, m.width, m.height)))
            .collect();
        if regions.is_empty() {
            return None;
        }
        let src = self
            .by_base
            .get(&k.base)
            .into_iter()
            .flatten()
            .filter_map(|kk| self.map.get(kk))
            .filter(|i| i.gpu_written && !is_compressed(i.key.format) && !i.key.ty3d && i.key.levels == 1)
            .filter(|i| texel_bytes(i.key.format) == bytes && i.key.width == lay.block_w && i.key.height == lay.block_h && i.write_gen > dst.block_gen)
            .max_by_key(|i| i.write_gen)?;
        Some((src.image, src.write_gen, regions))
    }

    /// The block copies `take_block_copies` would take (nothing marked).
    pub fn block_copies(&self, keys: &[ImageKey]) -> Vec<BlockCopy> {
        let mut out: Vec<BlockCopy> = Vec::new();
        for k in keys {
            if !is_compressed(k.format) || k.ty3d || out.iter().any(|c| c.key == *k) {
                continue;
            }
            if !self.gpu_bases.contains(&k.base) {
                continue;
            }
            let Some(dst) = self.map.get(k) else { continue };
            let (bw, bh) = (k.width.div_ceil(4), k.height.div_ceil(4));
            let bytes = texel_bytes(k.format);
            let src = self
                .by_base
                .get(&k.base)
                .into_iter()
                .flatten()
                .filter_map(|kk| self.map.get(kk))
                .filter(|i| i.gpu_written && !is_compressed(i.key.format) && !i.key.ty3d)
                .filter(|i| texel_bytes(i.key.format) == bytes && i.key.width == bw && i.key.height == bh && i.write_gen > dst.block_gen)
                .max_by_key(|i| i.write_gen);
            let tail = if super::tune::block_tails() { self.tail_blocks(k, dst, bytes) } else { None };
            if src.is_none() && tail.is_none() {
                continue;
            }
            // Mips with whole blocks, before the tail when a tail view
            // supplies the tail (the two copies must not overlap).
            let first_tail = tail.as_ref().and_then(|(_, _, r)| r.first().map(|x| x.0)).unwrap_or(u32::MAX);
            let mut levels = 0;
            if let Some(src) = src {
                while levels < src.key.levels.min(k.levels).min(first_tail) && (k.width >> levels) % 4 == 0 && (k.height >> levels) % 4 == 0 && (k.width >> levels) > 0 && (k.height >> levels) > 0 {
                    levels += 1;
                }
            }
            if levels == 0 && tail.is_none() {
                continue;
            }
            let seq = src.map_or(0, |s| s.write_gen).max(tail.as_ref().map_or(0, |t| t.1));
            out.push(BlockCopy {
                key: *k,
                src: if levels > 0 { src.map_or(vk::Image::null(), |s| s.image) } else { vk::Image::null() },
                dst: dst.image,
                width: bw,
                height: bh,
                levels,
                layers: src.map_or(1, |s| s.key.layers.min(k.layers).max(1)),
                tail: tail.map(|(img, _, r)| (img, r)),
                seq,
            });
        }
        out
    }

    /// Notes writes (GPU buffer writes, CP DMA) that cover the metadata of a
    /// rendered surface: fast clears, resolved at the surface's next use.
    /// GPU buffer writes: fast-clear metadata (`note_meta_writes`) and
    /// GPU-owned images overwritten (`note_overwrites`, the sized writes).
    pub fn note_writes(&mut self, writes: &[(u64, u64)], sized: &[(u64, u64)]) {
        let lookup = LOOKUP_GEN.load(std::sync::atomic::Ordering::Acquire);
        if let Some((w, s, g, n)) = &self.last_note {
            if *g == lookup && *n == self.note_gen && w == writes && s == sized {
                return;
            }
        }
        self.note_meta_writes(writes);
        self.note_overwrites(sized);
        self.last_note = Some((writes.to_vec(), sized.to_vec(), lookup, self.note_gen));
    }

    pub fn note_meta_writes(&mut self, writes: &[(u64, u64)]) {
        if self.meta_of.is_empty() {
            return;
        }
        for &(b, n) in writes {
            for (&meta, &(key, _)) in self.meta_of.range(b..b.saturating_add(n)) {
                self.pending_clears.insert(key, (meta, None));
                self.pending_change += 1;
            }
        }
    }

    /// What `note_writes` (then `note_meta_fill` for a fill of `fill`)
    /// would note, without noting it (the frame-ahead recorder keeps its
    /// own: `Exec::held_note`): the fast clears made pending (image,
    /// metadata address, the value when the fill gives it; later entries
    /// win) and the GPU-owned images overwritten.
    pub fn notes_for(&self, writes: &[(u64, u64)], sized: &[(u64, u64)], fill: Option<((u64, u64), u32)>) -> (Vec<(ImageKey, u64, Option<u32>)>, Vec<ImageKey>) {
        let mut clears = Vec::new();
        if !self.meta_of.is_empty() {
            for &(b, n) in writes {
                for (&meta, &(key, _)) in self.meta_of.range(b..b.saturating_add(n)) {
                    clears.push((key, meta, None));
                }
            }
            if let Some(((base, len), v)) = fill {
                let end = base.saturating_add(len);
                for (&meta, &(key, _)) in self.meta_of.range(base..end) {
                    if meta + 4 <= end {
                        clears.push((key, meta, Some(v)));
                    }
                }
            }
        }
        let mut over = Vec::new();
        if !self.owned.is_empty() {
            for &(b, n) in sized {
                let end = b.saturating_add(n);
                for (_, v) in self.owned.range(b.saturating_sub(self.max_span)..end) {
                    for &k in v {
                        let Some(img) = self.map.get(&k) else { continue };
                        let span = img.span;
                        if k.base + span <= b || !img.gpu_written {
                            continue;
                        }
                        if b <= k.base && k.base + span <= end && n <= 2 * span + (64 << 10) && !over.contains(&k) {
                            over.push(k);
                        }
                    }
                }
            }
        }
        (clears, over)
    }

    /// The GPU-owned images a zero fill of [base, base+len) clears as
    /// `zero_filled` chooses them, nothing noted (the frame-ahead recorder).
    pub fn zero_fill_targets(&self, base: u64, len: u64) -> Vec<(ImageKey, vk::Image, vk::Format)> {
        let mut out = Vec::new();
        if len == 0 || self.owned.is_empty() {
            return out;
        }
        let end = base.saturating_add(len);
        for (_, v) in self.owned.range(base.saturating_sub(self.max_span)..end) {
            for &k in v {
                let Some(img) = self.map.get(&k) else { continue };
                let span = img.span;
                if !img.gpu_written || is_compressed(k.format) || !(base <= k.base && k.base + span <= end && len <= 2 * span + (64 << 10)) {
                    continue;
                }
                if !out.iter().any(|x: &(ImageKey, vk::Image, vk::Format)| x.0 == k) {
                    out.push((k, img.image, k.format));
                }
            }
        }
        out
    }

    /// Notes a held executor's own fast clears and overwritten images once
    /// its batches go to the GPU (`Exec::past_commit`): after the notes of
    /// the work before them.
    pub fn apply_held(&mut self, clears: &std::collections::HashMap<ImageKey, (u64, Option<u32>)>, over: &std::collections::HashSet<ImageKey>) {
        for (k, &(meta, v)) in clears {
            self.pending_clears.insert(*k, (meta, v));
            self.pending_change += 1;
        }
        for k in over {
            let Some(img) = self.map.get_mut(k) else { continue };
            if !img.overwritten {
                img.overwritten = true;
                self.overwritten_keys.insert(*k);
                self.pending_change += 1;
                self.pending_overwrites += 1;
            }
        }
        self.note_gen += 1;
        NOTE_GEN.fetch_add(1, std::sync::atomic::Ordering::Release);
    }

    /// The fast clear of image `k` with metadata at `meta` (`take_clears`'s
    /// entry for a clear the frame-ahead recorder noted itself).
    pub fn fast_clear_of(&self, k: ImageKey, meta: u64, known: Option<u32>) -> Option<FastClear> {
        let (Some(img), Some(&(_, clear))) = (self.map.get(&k), self.meta_of.get(&meta)) else { return None };
        Some(FastClear { key: k, image: img.image, meta, clear, known })
    }

    /// The fill kernel stored `v` in every word of [base, base+len): the fast
    /// clears pending on metadata inside it read `v` (after `note_writes`
    /// for the same dispatch, which marked them unknown).
    pub fn note_meta_fill(&mut self, base: u64, len: u64, v: u32) {
        if self.meta_of.is_empty() {
            return;
        }
        let end = base.saturating_add(len);
        for (&meta, &(key, _)) in self.meta_of.range(base..end) {
            if meta + 4 <= end {
                self.pending_clears.insert(key, (meta, Some(v)));
                // A write noted again over it makes it unknown again: not a
                // repeat for the memos of `note_writes` and `Exec::last_notes`.
                self.note_gen += 1;
                NOTE_GEN.fetch_add(1, std::sync::atomic::Ordering::Release);
            }
        }
    }

    /// Whether every pending fast clear among `keys` has a known value.
    pub fn clears_known(&self, keys: &[ImageKey]) -> bool {
        keys.iter().all(|k| self.pending_clears.get(k).is_none_or(|&(_, v)| v.is_some()))
    }

    /// The images with a fast clear pending.
    pub fn pending_clear_keys(&self) -> Vec<ImageKey> {
        self.pending_clears.keys().copied().collect()
    }

    /// The images with a fast clear or an overwrite reload pending.
    pub fn pending_keys(&self) -> Vec<ImageKey> {
        self.pending_clears.keys().chain(self.overwritten_keys.iter()).copied().collect()
    }

    /// Whether any of `keys` has a fast clear pending.
    pub fn clears_pending(&self, keys: &[ImageKey]) -> bool {
        !self.pending_clears.is_empty() && keys.iter().any(|k| self.pending_clears.contains_key(k))
    }

    /// Takes the pending fast clears of `keys`.
    pub fn take_clears(&mut self, keys: &[ImageKey]) -> Vec<FastClear> {
        let mut out = Vec::new();
        for k in keys {
            let Some((meta, known)) = self.pending_clears.remove(k) else { continue };
            self.pending_change += 1;
            self.note_gen += 1;
            NOTE_GEN.fetch_add(1, std::sync::atomic::Ordering::Release);
            let (Some(img), Some(&(_, clear))) = (self.map.get(k), self.meta_of.get(&meta)) else { continue };
            if out.iter().any(|c: &FastClear| c.key == *k) {
                continue;
            }
            out.push(FastClear { key: *k, image: img.image, meta, clear, known });
        }
        out
    }

    /// The cached image a T# would use now (fast-clear checks; no side effects).
    /// Whether a T# binds as a depth/stencil view (a rendered depth surface):
    /// the translation compares in hardware there and emulates comparisons
    /// on color views.
    pub fn depth_view(&mut self, t: &Tsharp) -> bool {
        self.resolved_key(t, false).is_some_and(|k| is_depth(k.format))
    }

    pub fn resolved_key(&mut self, t: &Tsharp, storage: bool) -> Option<ImageKey> {
        let lookup = LOOKUP_GEN.load(std::sync::atomic::Ordering::Acquire);
        if let Some(&(g, k)) = self.key_memo.get(&(t.raw, storage)) {
            if g == lookup {
                tls(|m| {
                    if m.lookup_gen == lookup {
                        m.keys.insert((t.raw, storage), k);
                    }
                });
                return k;
            }
        }
        let k = self.lookup(&Self::key_of(t)?, storage).map(|(k, _)| k);
        if self.key_memo.len() >= 1 << 16 {
            self.key_memo.clear();
        }
        self.key_memo.insert((t.raw, storage), (lookup, k));
        tls(|m| {
            if m.lookup_gen == lookup {
                m.keys.insert((t.raw, storage), k);
            }
        });
        k
    }

    /// Image for a T#: an existing GPU image at that address with a
    /// compatible size, or a texture uploaded from guest memory.
    /// Whether the image a storage binding of `t` writes exists and is
    /// GPU-owned already: binding it changes no ownership (a long parallel
    /// run's chunk may then bind it, `Exec::long`).
    pub fn storage_owned(&self, t: &Tsharp) -> bool {
        let Some(key) = Self::key_of(t) else { return false };
        self.lookup(&key, true).is_some_and(|(k, _)| self.map.get(&k).is_some_and(|i| i.gpu_written))
    }

    fn image_for(&mut self, t: &Tsharp, frame: u64, storage: bool) -> Result<ImageKey, String> {
        let key = Self::key_of(t).ok_or_else(|| format!("image format {} unsupported", t.format))?;
        let timed = || holds_on().then(std::time::Instant::now);
        if let Some((k, how)) = self.lookup(&key, storage) {
            if how == "exact" && !self.map[&k].gpu_written {
                let ts = timed();
                self.refresh(key, t, frame)?;
                note_step(0, ts);
            }
            if how == "exact" {
                self.note_layout(k, t);
            }
            return Ok(k);
        }
        let ts = timed();
        let (image, mem, bytes) = self.create(key)?;
        note_step(1, ts);
        let ts = timed();
        // (`tune upload_predetile`: this thread's prepared upload, its copy
        // recorded now; checked instead with `DS_PREDETILE_CHECK`.)
        let mut pre_hashes = None;
        let prepared = take_prepared(&key, t);
        let uploaded = match prepared {
            Some(p) if !predetile_check() => {
                pre_hashes = Some(p.hashes);
                let r = self.upload_prepared(image, key, t, p);
                note_step(2, ts);
                r
            }
            Some(p) => {
                CHECKED.with(|c| *c.borrow_mut() = Some(p));
                let r = self.upload(image, key, t);
                if let Some(p) = CHECKED.with(|c| c.borrow_mut().take()) {
                    free_prepared(p);
                }
                r
            }
            None => {
                let r = self.upload(image, key, t);
                note_step(3, ts);
                r
            }
        };
        let source = match uploaded {
            Ok(s) => Some(s),
            Err(e) => {
                self.report(format!("texture {:#x} ({}x{} format {} sw_mode {}): {e}; contents undefined", t.base, t.width, t.height, t.format, t.sw_mode));
                self.to_general(image, key.format);
                None
            }
        };
        let _ = frame;
        // CPU writes to the source re-upload it (write tracking).
        let ts = timed();
        let region = source.and_then(|(b, n)| super::track::watch(b, n));
        note_step(4, ts);
        if source.is_some() && region.is_none() {
            self.report(format!("texture {:#x} is not write-tracked (read-only or unmapped pages): later CPU updates are not seen", t.base));
        }
        let ts = timed();
        let hashes = match (source, pre_hashes) {
            (Some(_), Some(h)) => h,
            _ => source.map(|(b, n)| all_hashes(b, n)).unwrap_or([0; SLICES]),
        };
        note_step(5, ts);
        self.map_insert(key, Image { image, key, views: HashMap::default(), gpu_written: false, source: region, hashes, checked_frame: frame, write_gen: 0, block_gen: 0, span: 0, layout: None, overwritten: false, writer: 0, mem, bytes, used: UseStamp::new(frame) });
        self.note_layout(key, t);
        Ok(key)
    }

    /// Remembers the guest layout of an image from a T# that describes it
    /// exactly (its span and a T# to upload it with again).
    fn note_layout(&mut self, k: ImageKey, t: &Tsharp) {
        let Some(img) = self.map.get_mut(&k) else { return };
        if img.layout.is_some() || k.ty3d {
            return;
        }
        let Some(el) = elem_log2(t.format) else { return };
        if let Some(lay) = tiling::layout(t.sw_mode, el, k.width, k.height, k.levels, is_block(t.format)) {
            img.span = lay.size(k.layers);
            img.layout = Some(*t);
            self.index_owned(k);
        }
    }

    /// Indexes a GPU-owned image with a known span (`note_overwrites`).
    fn index_owned(&mut self, k: ImageKey) {
        let Some(img) = self.map.get(&k) else { return };
        if !img.gpu_written || img.span == 0 {
            return;
        }
        let span = img.span;
        let v = self.owned.entry(k.base).or_default();
        if !v.contains(&k) {
            mark_note_pages(k.base, span);
            v.push(k);
            self.max_span = self.max_span.max(span);
            self.note_gen += 1;
            NOTE_GEN.fetch_add(1, std::sync::atomic::Ordering::Release);
        }
    }

    /// Notes GPU buffer writes and CP DMA writes (sized ranges, not unbounded
    /// pointers) over GPU-owned images. The title clears some of its render
    /// targets and volumes through a buffer view of their memory (the
    /// volumetric fog's density volume is filled with zeros by a compute
    /// shader every frame, then fog volumes are drawn into its slices);
    /// images here do not share memory with buffers, so an image whose guest
    /// bytes a write covers entirely takes its contents from guest memory at
    /// its next use (`reload_overwritten`). Writes over part of an image are
    /// reported, not applied: the rest of its guest bytes are stale.
    pub fn note_overwrites(&mut self, writes: &[(u64, u64)]) {
        if writes.is_empty() || self.owned.is_empty() {
            return;
        }
        let mut partial = Vec::new();
        for &(b, n) in writes {
            let end = b.saturating_add(n);
            let keys: Vec<ImageKey> = self.owned.range(b.saturating_sub(self.max_span)..end).flat_map(|(_, v)| v.iter().copied()).collect();
            for k in keys {
                let Some(img) = self.map.get_mut(&k) else { continue };
                let span = img.span;
                if k.base + span <= b || !img.gpu_written {
                    continue;
                }
                // Covering it, and not a much larger range that happens to
                // contain it (a pool or heap binding).
                if b <= k.base && k.base + span <= end && n <= 2 * span + (64 << 10) {
                    if !img.overwritten {
                        img.overwritten = true;
                        self.overwritten_keys.insert(k);
                        self.pending_change += 1;
                        self.pending_overwrites += 1;
                    }
                } else if self.partial_seen.len() < 64 && self.partial_seen.insert(k.base) {
                    partial.push(format!(
                        "GPU buffer write {b:#x}+{n:#x} covers part of GPU-owned image {:#x}+{span:#x} ({}x{}x{} {:?}): not applied",
                        k.base, k.width, k.height, k.layers, k.format
                    ));
                }
            }
        }
        for m in partial {
            crate::info!("images: {m}");
        }
    }

    /// The fill kernel stores zeros over [base, base+len) (`tune
    /// zero_fill_clears`): the GPU-owned images it covers as
    /// `note_overwrites` counts covering are cleared by the caller right after
    /// the fill, in its batch (returned), instead of being marked overwritten
    /// (which made their next user wait for the GPU, scan their guest bytes
    /// for zeros and clear them: ~6,300 per gate run, ~0.4 ms each on the
    /// graphics command processor). Compressed formats keep that path.
    pub fn zero_filled(&mut self, base: u64, len: u64) -> Vec<(vk::Image, vk::Format)> {
        let mut out = Vec::new();
        if len == 0 || self.owned.is_empty() {
            return out;
        }
        let end = base.saturating_add(len);
        let keys: Vec<ImageKey> = self.owned.range(base.saturating_sub(self.max_span)..end).flat_map(|(_, v)| v.iter().copied()).collect();
        for k in keys {
            let Some(img) = self.map.get_mut(&k) else { continue };
            let span = img.span;
            if !img.gpu_written || is_compressed(k.format) || !(base <= k.base && k.base + span <= end && len <= 2 * span + (64 << 10)) {
                continue;
            }
            if img.overwritten {
                img.overwritten = false;
                self.overwritten_keys.remove(&k);
                self.pending_overwrites = self.pending_overwrites.saturating_sub(1);
            }
            self.pending_change += 1;
            self.note_gen += 1;
            NOTE_GEN.fetch_add(1, std::sync::atomic::Ordering::Release);
            self.overwrite_clears += 1;
            ZERO_FILL_CLEARS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            out.push((img.image, k.format));
        }
        out
    }

    /// Evicts uploaded images not used for EVICT_AGE frames, least recently
    /// used first, until `want` bytes are freed: (the images to destroy,
    /// bytes, images left). A texture evicted is uploaded again from guest
    /// memory at its next use.
    /// The `n` largest cached images, one clause each (eviction reports).
    fn largest(&self, n: usize) -> String {
        let mut v: Vec<(&ImageKey, &Image)> = self.map.iter().collect();
        v.sort_by(|a, b| b.1.bytes.cmp(&a.1.bytes));
        let parts: Vec<String> = v
            .iter()
            .take(n)
            .map(|(k, i)| format!("{} MiB at {:#x} {}x{}x{} {:?} {} mips{}{}", i.bytes >> 20, k.base, k.width, k.height, k.layers, k.format, k.levels, if k.ty3d { " 3D" } else { "" }, if i.gpu_written { " GPU-written" } else { "" }))
            .collect();
        parts.join(", ")
    }

    pub fn evict(&mut self, frame: u64, want: u64) -> (Vec<Evicted>, u64, usize) {
        let mut cands: Vec<(u64, ImageKey)> = self
            .map
            .iter()
            .filter(|(_, i)| !i.gpu_written && frame.saturating_sub(i.used.get()) >= EVICT_AGE)
            .map(|(k, i)| (i.used.get(), *k))
            .collect();
        cands.sort_unstable_by_key(|c| c.0);
        let (mut out, mut freed) = (Vec::new(), 0u64);
        for (_, k) in cands {
            if freed >= want {
                break;
            }
            if let Some((e, bytes, _)) = self.drop_image(&k) {
                freed += bytes;
                out.push(e);
            }
        }
        EVICTED[0].fetch_add(out.len() as u64, std::sync::atomic::Ordering::Relaxed);
        EVICTED[1].fetch_add(freed, std::sync::atomic::Ordering::Relaxed);
        (out, freed, self.map.len())
    }

    /// Removes an image from the cache and every index naming it: (the
    /// image to destroy, its bytes, whether GPU-written).
    fn drop_image(&mut self, k: &ImageKey) -> Option<(Evicted, u64, bool)> {
        let img = self.map_remove(k)?;
        if let Some(v) = self.owned.get_mut(&k.base) {
            v.retain(|x| x != k);
            if v.is_empty() {
                self.owned.remove(&k.base);
            }
        }
        self.overwritten_keys.remove(k);
        if img.overwritten {
            self.pending_overwrites = self.pending_overwrites.saturating_sub(1);
        }
        self.pending_clears.remove(k);
        self.pending_change += 1;
        self.stencil_of.retain(|_, v| *v != *k);
        self.meta_of.retain(|_, v| v.0 != *k);
        if img.gpu_written && !self.by_base.get(&k.base).is_some_and(|ks| ks.iter().any(|x| self.map.get(x).is_some_and(|i| i.gpu_written))) {
            self.gpu_bases.remove(&k.base);
        }
        if let Some(id) = img.source {
            super::track::unwatch(id);
        }
        self.last_note = None;
        self.note_gen += 1;
        NOTE_GEN.fetch_add(1, std::sync::atomic::Ordering::Release);
        let views = img.views.values().copied().collect();
        Some((Evicted { image: img.image, mem: img.mem, views }, img.bytes, img.gpu_written))
    }

    /// Whether [base, base+len) is wholly the memory of GPU-written images
    /// that `zero_filled` clears, with no other image over any of it: a
    /// zero fill of the range then need not store into guest memory (`tune
    /// fill_skip`), whose bytes such images do not keep current anyway.
    pub fn fill_covered(&self, base: u64, len: u64) -> bool {
        if len == 0 || self.owned.is_empty() {
            return false;
        }
        let end = base.saturating_add(len);
        let mut spans: Vec<(u64, u64)> = Vec::new();
        for (k, img) in self.map.iter() {
            let (a, b) = (k.base, k.base.saturating_add(img.span.max(1)));
            if b <= base || a >= end {
                continue;
            }
            if !img.gpu_written || img.span == 0 || is_compressed(k.format) || !(base <= a && b <= end && len <= 2 * img.span + (64 << 10)) {
                return false;
            }
            spans.push((a, b));
        }
        spans.sort_unstable();
        let mut at = base;
        for (a, b) in spans {
            if a > at {
                return false;
            }
            at = at.max(b);
        }
        at >= end
    }

    /// Diagnosis: the images over [base, base+len) with what `fill_covered`
    /// checks (written by the GPU, span, compressed, 3D, layers).
    pub fn fill_cover_report(&self, base: u64, len: u64) -> String {
        let end = base.saturating_add(len);
        let parts: Vec<String> = self
            .map
            .iter()
            .filter(|(k, img)| {
                let b = k.base.saturating_add(img.span.max(1));
                k.base < end && b > base
            })
            .take(8)
            .map(|(k, img)| format!("{:#x} {}x{}x{} {:?}{} gpu_written {} span {:#x}", k.base, k.width, k.height, k.layers, k.format, if k.ty3d { " 3D" } else { "" }, img.gpu_written, img.span))
            .collect();
        if parts.is_empty() { "no image".into() } else { parts.join(", ") }
    }

    /// The guest bytes of the images among `keys` overwritten through a
    /// buffer (`reload_overwritten` reads them).
    pub fn overwritten_spans(&self, keys: &[ImageKey]) -> Vec<(u64, u64)> {
        keys.iter().filter_map(|k| self.map.get(k).filter(|i| i.overwritten && i.span > 0).map(|i| (k.base, i.span))).collect()
    }

    /// Whether any of `keys` was overwritten through a buffer.
    pub fn overwrites_pending(&self, keys: &[ImageKey]) -> bool {
        self.pending_overwrites > 0 && keys.iter().any(|k| self.map.get(k).is_some_and(|i| i.overwritten))
    }

    /// Gives the images among `keys` that `note_overwrites` marked their
    /// guest contents (the writers have completed): all-zero bytes clear the
    /// image, other contents are uploaded with the layout of its T#.
    pub fn reload_overwritten(&mut self, keys: &[ImageKey]) {
        for k in keys {
            let Some(img) = self.map.get_mut(k) else { continue };
            if !img.overwritten {
                continue;
            }
            img.overwritten = false;
            self.overwritten_keys.remove(k);
            self.pending_change += 1;
            self.pending_overwrites = self.pending_overwrites.saturating_sub(1);
            self.note_gen += 1;
            NOTE_GEN.fetch_add(1, std::sync::atomic::Ordering::Release);
            let (image, span, layout) = (img.image, img.span, img.layout);
            if !super::cp::guest_readable(k.base, span) {
                self.report(format!("image {:#x} overwritten through a buffer: guest memory not readable", k.base));
                continue;
            }
            let bytes = unsafe { std::slice::from_raw_parts(k.base as *const u8, span as usize) };
            let zero = bytes.chunks(4096).all(|c| c.iter().fold(0u8, |a, &x| a | x) == 0);
            let how = if zero {
                let format = k.format;
                self.one_shot(|cmd| unsafe {
                    let d = &ctx().device;
                    let range = vk::ImageSubresourceRange { aspect_mask: aspects(format), base_mip_level: 0, level_count: vk::REMAINING_MIP_LEVELS, base_array_layer: 0, layer_count: vk::REMAINING_ARRAY_LAYERS };
                    let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
                    d.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[mb], &[], &[]);
                    if is_depth(format) {
                        d.cmd_clear_depth_stencil_image(cmd, image, vk::ImageLayout::GENERAL, &vk::ClearDepthStencilValue { depth: 0.0, stencil: 0 }, &[range]);
                    } else {
                        d.cmd_clear_color_image(cmd, image, vk::ImageLayout::GENERAL, &vk::ClearColorValue::default(), &[range]);
                    }
                    let a = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
                    d.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::ALL_COMMANDS, vk::DependencyFlags::empty(), &[a], &[], &[]);
                });
                self.overwrite_clears += 1;
                "cleared (zeros)"
            } else if let (Some(t), false) = (layout, is_depth(k.format)) {
                if let Err(e) = self.upload(image, *k, &t) {
                    self.report(format!("image {:#x} overwritten through a buffer: upload failed: {e}", k.base));
                    continue;
                }
                self.overwrite_uploads += 1;
                "uploaded"
            } else {
                self.report(format!("image {:#x} ({:?}) overwritten through a buffer with non-zero data: not applied", k.base, k.format));
                continue;
            };
            static SEEN: Mutex<Option<std::collections::HashSet<u64>>> = Mutex::new(None);
            if SEEN.lock().get_or_insert_with(Default::default).insert(k.base) {
                crate::info!("images: {:#x} {}x{}x{} {:?} overwritten through a buffer view of its memory: {how}", k.base, k.width, k.height, k.layers, k.format);
            }
        }
    }

    /// Re-uploads a CPU-sourced texture whose guest bytes were written
    /// since its upload (write tracking: CPU faults, file reads, GPU writes).
    fn refresh(&mut self, key: ImageKey, t: &Tsharp, frame: u64) -> Result<(), String> {
        // Staging buffers of completed uploads go back at once (not only at
        // the next upload).
        if !self.inflight.is_empty() {
            self.reap(usize::MAX, u64::MAX);
        }
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
                REUPLOAD_CAUSE.with(|c| c.set(5));
                // (`DS_SKIP_REUPLOADS`: timing only, the image keeps its contents.)
                let r = if skip_reuploads() { SKIPPED_REUPLOADS.fetch_add(1, std::sync::atomic::Ordering::Relaxed); Ok((b, n)) } else { self.upload_parts(image, key, t, None, false) };
                REUPLOAD_CAUSE.with(|c| c.set(0));
                r?;
                self.map.get_mut(&key).unwrap().hashes = h;
            }
        }
        if !super::track::protected() && !scrubbing() && self.map[&key].checked_frame != frame {
            // CPU writes: one slice of the sampled hash per frame.
            let img = self.map.get_mut(&key).unwrap();
            img.checked_frame = frame;
            let s = (frame as usize) % SLICES;
            if slice_hash(b, n, s) != img.hashes[s] {
                self.reuploads += 1;
                self.report(format!("texture {:#x} {}x{} re-uploaded (CPU write)", t.base, t.width, t.height));
                REUPLOAD_CAUSE.with(|c| c.set(4));
                // (`DS_SKIP_REUPLOADS`: timing only, the image keeps its contents.)
                let r = if skip_reuploads() { SKIPPED_REUPLOADS.fetch_add(1, std::sync::atomic::Ordering::Relaxed); Ok((b, n)) } else { self.upload_parts(image, key, t, Some(&[]), false) };
                REUPLOAD_CAUSE.with(|c| c.set(0));
                r?;
                self.map.get_mut(&key).unwrap().hashes = all_hashes(b, n);
            }
        }
        if let Some((why, parts)) = super::track::take_dirty(region) {
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
                super::track::Why::Forced => "requested",
                super::track::Why::Sampled => "CPU write, sampled",
                super::track::Why::Written(_) => "CPU write",
                super::track::Why::None => "?",
            });
            self.report(msg);
            let t0 = std::time::Instant::now();
            REUPLOAD_CAUSE.with(|c| {
                c.set(match why {
                    super::track::Why::Remap(_, _, 0) => 1,
                    super::track::Why::Remap(..) => 2,
                    super::track::Why::HostWrite(_) => 3,
                    super::track::Why::Fault(_) | super::track::Why::Written(_) | super::track::Why::Sampled => 4,
                    super::track::Why::GpuWrite(..) => 5,
                    _ => 6,
                })
            });
            // (`DS_SKIP_REUPLOADS`: timing only, the image keeps its contents.)
            let r = if skip_reuploads() { SKIPPED_REUPLOADS.fetch_add(1, std::sync::atomic::Ordering::Relaxed); Ok((b, n)) } else { self.upload_parts(image, key, t, parts.as_deref(), matches!(why, super::track::Why::Forced)) };
            REUPLOAD_CAUSE.with(|c| c.set(0));
            r?;
            let h = all_hashes(b, n);
            if let super::track::Why::Remap(rb, rl, op) = why {
                use std::sync::atomic::Ordering::Relaxed;
                let s = &REMAP_UPLOADS[(op as usize).min(2)];
                s[0].fetch_add(1, Relaxed);
                if h == self.map[&key].hashes {
                    s[1].fetch_add(1, Relaxed);
                }
                if rb > b || rb.saturating_add(rl) < b + n {
                    s[2].fetch_add(1, Relaxed);
                }
                s[3].fetch_add(t0.elapsed().as_nanos() as u64, Relaxed);
            }
            self.map.get_mut(&key).unwrap().hashes = h;
        }
        Ok(())
    }

    /// The guest ranges of the next `max` uploaded textures not marked
    /// written yet (write-watch scrubbing).
    fn scrub_regions(&mut self, max: usize) -> Vec<(u64, u64)> {
        let g = LOOKUP_GEN.load(std::sync::atomic::Ordering::Acquire);
        if self.scrub_gen != g || self.scrub_keys.is_empty() {
            self.scrub_keys = self.map.iter().filter(|(_, i)| !i.gpu_written && i.source.is_some()).map(|(k, _)| *k).collect();
            self.scrub_gen = g;
        }
        let n = self.scrub_keys.len();
        let mut out = Vec::new();
        for _ in 0..max.min(n) {
            self.scrub_at = (self.scrub_at + 1) % n;
            let Some(img) = self.map.get(&self.scrub_keys[self.scrub_at]) else { continue };
            let (false, Some(region)) = (img.gpu_written, img.source) else { continue };
            if super::track::is_dirty(region) {
                continue;
            }
            if let Some(r) = super::track::range(region) {
                out.push(r);
            }
        }
        out
    }

    /// The next `max` uploaded textures to sample: (tracking region, guest
    /// range, slice, the slice's hash at upload); each visit samples the
    /// texture's next slice (`checked_frame` counts them while scrubbing).
    fn scrub_batch(&mut self, max: usize) -> Vec<(u32, u64, u64, usize, u64)> {
        let g = LOOKUP_GEN.load(std::sync::atomic::Ordering::Acquire);
        if self.scrub_gen != g || self.scrub_keys.is_empty() {
            self.scrub_keys = self.map.iter().filter(|(_, i)| !i.gpu_written && i.source.is_some()).map(|(k, _)| *k).collect();
            self.scrub_gen = g;
        }
        let n = self.scrub_keys.len();
        let mut out = Vec::new();
        for _ in 0..max.min(n) {
            self.scrub_at = (self.scrub_at + 1) % n;
            let k = self.scrub_keys[self.scrub_at];
            let Some(img) = self.map.get_mut(&k) else { continue };
            let (false, Some(region)) = (img.gpu_written, img.source) else { continue };
            let s = (img.checked_frame % SLICES as u64) as usize;
            img.checked_frame += 1;
            let stored = img.hashes[s];
            // Already marked: re-uploaded at its next use (textures whose
            // memory streaming reused are often never used again).
            if super::track::is_dirty(region) {
                continue;
            }
            if let Some((b, len)) = super::track::range(region) {
                out.push((region, b, len, s, stored));
            }
        }
        out
    }

    /// Adds an image (keeps the by-address index).
    fn map_insert(&mut self, key: ImageKey, img: Image) {
        if img.gpu_written || self.gpu_bases.contains(&key.base) {
            gpu_images_changed();
        }
        IMAGE_BYTES.fetch_add(img.bytes, std::sync::atomic::Ordering::Relaxed);
        let (w, bit) = present_bit(&key);
        PRESENT[w].fetch_or(bit, std::sync::atomic::Ordering::Relaxed);
        match self.map.insert(key, img) {
            None => self.by_base.entry(key.base).or_default().push(key),
            Some(old) => {
                // Replaced: destroyed once no batch can name it.
                IMAGE_BYTES.fetch_sub(old.bytes, std::sync::atomic::Ordering::Relaxed);
                super::exec::bury_images(vec![Evicted { image: old.image, mem: old.mem, views: old.views.values().copied().collect() }]);
            }
        }
        LOOKUP_GEN.fetch_add(1, std::sync::atomic::Ordering::Release);
    }

    fn map_remove(&mut self, key: &ImageKey) -> Option<Image> {
        let img = self.map.remove(key)?;
        let (w, bit) = present_bit(key);
        PRESENT[w].fetch_and(!bit, std::sync::atomic::Ordering::Relaxed);
        IMAGE_BYTES.fetch_sub(img.bytes, std::sync::atomic::Ordering::Relaxed);
        LOOKUP_GEN.fetch_add(1, std::sync::atomic::Ordering::Release);
        if img.gpu_written || self.gpu_bases.contains(&key.base) {
            gpu_images_changed();
        }
        if let Some(v) = self.by_base.get_mut(&key.base) {
            v.retain(|k| k != key);
            if v.is_empty() {
                self.by_base.remove(&key.base);
            }
        }
        Some(img)
    }

    /// An existing image at `key`'s address with the same size that can be
    /// viewed in `key`'s format (optionally only GPU-written ones).
    fn alias(&self, key: &ImageKey, gpu_written_only: bool) -> Option<ImageKey> {
        self.by_base
            .get(&key.base)?
            .iter()
            .filter(|k| {
                self.map.get(k).is_some_and(|i| !gpu_written_only || i.gpu_written)
                    && k.width == key.width
                    && k.height == key.height
                    && k.layers >= key.layers
                    && k.levels >= key.levels
                    && k.ty3d == key.ty3d
                    && view_compatible(k.format, key.format)
            })
            .copied()
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
            let (image, mem, bytes) = self.create(key)?;
            self.to_general(image, key.format);
            self.map_insert(key, Image { image, key, views: HashMap::default(), gpu_written: true, source: None, hashes: [0; SLICES], checked_frame: 0, write_gen: 0, block_gen: 0, span: 0, layout: None, overwritten: false, writer: 0, mem, bytes, used: UseStamp::new(super::FRAMES.load(std::sync::atomic::Ordering::Relaxed)) });
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
        if !img.gpu_written {
            LOOKUP_GEN.fetch_add(1, std::sync::atomic::Ordering::Release);
            gpu_images_changed();
        }
        img.gpu_written = true;
        self.gpu_bases.insert(k.base);
        // Generations move only on a change (every draw with a depth/stencil
        // target came here: lookups cached by generation never hit).
        if s.stencil_base != 0 && has_stencil(k.format) && self.stencil_of.insert(s.stencil_base, k) != Some(k) {
            LOOKUP_GEN.fetch_add(1, std::sync::atomic::Ordering::Release);
            gpu_images_changed();
            self.gpu_bases.insert(s.stencil_base);
        }
        if s.meta != 0 {
            mark_note_pages(s.meta, 4);
        }
        if s.meta != 0 && self.meta_of.insert(s.meta, (k, s.clear)) != Some((k, s.clear)) {
            self.note_gen += 1;
            NOTE_GEN.fetch_add(1, std::sync::atomic::Ordering::Release);
        }
        img.source = None;
        self.index_owned(k);
        Ok(k)
    }

    /// Attachment for a color or depth/stencil surface (one mip level, one layer).
    pub fn target(&mut self, s: &Surface) -> Result<Target, String> {
        let k = self.surface_image(s)?;
        let img = self.map.get_mut(&k).unwrap();
        img.used.stamp(super::FRAMES.load(std::sync::atomic::Ordering::Relaxed));
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
        Ok(Target { key: k, image: img.image, view, used: img.used, format: s.format, width: (img.key.width >> mip).max(1), height: (img.key.height >> mip).max(1) })
    }

    /// Every image at `base` (diagnosis of the image a display buffer
    /// presents): format, size, layers, GPU-written, frame of last use.
    pub fn at_base_report(&self, base: u64) -> String {
        let mut v: Vec<String> = self
            .map
            .values()
            .filter(|i| i.key.base == base)
            .map(|i| format!("{:?} {}x{}x{} gpu {} used {}", i.key.format, i.key.width, i.key.height, i.key.layers, i.gpu_written, i.used.get()))
            .collect();
        v.sort();
        v.join("; ")
    }

    /// The GPU image last rendered at `base` (display buffers).
    /// `size`: the size the buffer is registered with: an image of that
    /// size first (the title draws a display buffer at another size after
    /// it changes its display mode, and the larger image of the old size
    /// stays at the address, no longer drawn), else the largest.
    pub fn rendered_at(&self, base: u64, size: Option<(u32, u32)>) -> Option<(vk::Image, vk::Format, u32, u32)> {
        let at = || self.map.values().filter(|i| i.key.base == base && i.gpu_written && !is_depth(i.key.format));
        let area = |i: &&Image| i.key.width as u64 * i.key.height as u64;
        size.and_then(|(w, h)| at().filter(|i| (i.key.width, i.key.height) == (w, h)).max_by_key(area))
            .or_else(|| at().max_by_key(area))
            .map(|i| {
                // Presented: in use even when not rendered again (a static
                // screen flipped as it is), so never evicted meanwhile.
                i.used.stamp(super::FRAMES.load(std::sync::atomic::Ordering::Relaxed));
                (i.image, i.key.format, i.key.width, i.key.height)
            })
    }

    /// Reads mip 0 / layer 0 of a color image back (tightly packed texels).
    pub fn read_back(&mut self, image: vk::Image, format: vk::Format, width: u32, height: u32) -> Result<Vec<u8>, String> {
        if texel_bytes(format) == 0 || is_compressed(format) || is_depth(format) {
            return Err(format!("read back of {format:?} unsupported"));
        }
        self.read_back_aspect(image, format, vk::ImageAspectFlags::COLOR, width, height)
    }

    /// One texel at (x, y) of mip 0 / layer 0 of one aspect, in the layout
    /// of `read_back_aspect` (pixel watches of frame captures).
    pub fn read_back_texel(&mut self, image: vk::Image, format: vk::Format, aspect: vk::ImageAspectFlags, x: u32, y: u32) -> Result<Vec<u8>, String> {
        let bpp = if aspect == vk::ImageAspectFlags::STENCIL {
            1
        } else if aspect == vk::ImageAspectFlags::DEPTH {
            if matches!(format, vk::Format::D16_UNORM | vk::Format::D16_UNORM_S8_UINT) { 2 } else { 4 }
        } else {
            texel_bytes(format) as u64
        };
        if bpp == 0 || is_compressed(format) {
            return Err(format!("texel read back of {format:?} unsupported"));
        }
        let c = ctx();
        let (buf, mem, ptr) = c.host_buffer(bpp.max(4), vk::BufferUsageFlags::TRANSFER_DST)?;
        self.one_shot(|cmd| unsafe {
            let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::MEMORY_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_READ);
            c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[mb], &[], &[]);
            let region = vk::BufferImageCopy::default()
                .image_subresource(vk::ImageSubresourceLayers { aspect_mask: aspect, mip_level: 0, base_array_layer: 0, layer_count: 1 })
                .image_offset(vk::Offset3D { x: x as i32, y: y as i32, z: 0 })
                .image_extent(vk::Extent3D { width: 1, height: 1, depth: 1 });
            c.device.cmd_copy_image_to_buffer(cmd, image, vk::ImageLayout::GENERAL, buf, &[region]);
            let hb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::HOST_READ);
            c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::HOST, vk::DependencyFlags::empty(), &[hb], &[], &[]);
        });
        let out = unsafe { std::slice::from_raw_parts(ptr, bpp as usize).to_vec() };
        unsafe {
            c.device.destroy_buffer(buf, None);
            c.device.unmap_memory(mem);
            c.device.free_memory(mem, None);
        }
        Ok(out)
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

    /// Share of zero bytes in the cached image a T# resolves to, at the T#'s
    /// base mip (layer 0), as the GPU holds it (frame captures; compare with
    /// the guest data).
    pub fn gpu_zero_stats(&mut self, t: &Tsharp) -> Option<String> {
        let (k, _) = self.lookup(&Self::key_of(t)?, false)?;
        if k.ty3d || is_depth(k.format) {
            return None;
        }
        let image = self.map.get(&k)?.image;
        let mip = t.base_level.min(k.levels.saturating_sub(1));
        let (w, h) = ((k.width >> mip).max(1), (k.height >> mip).max(1));
        let bytes = texel_bytes(k.format) as u64;
        let size = if is_compressed(k.format) { w.div_ceil(4) as u64 * h.div_ceil(4) as u64 * bytes } else { w as u64 * h as u64 * bytes };
        if size == 0 || size > 64 << 20 {
            return None;
        }
        let c = ctx();
        let (buf, mem, ptr) = c.host_buffer(size, vk::BufferUsageFlags::TRANSFER_DST).ok()?;
        self.one_shot(|cmd| unsafe {
            let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::MEMORY_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_READ);
            c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[mb], &[], &[]);
            let region = vk::BufferImageCopy::default()
                .image_subresource(vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: mip, base_array_layer: 0, layer_count: 1 })
                .image_extent(vk::Extent3D { width: w, height: h, depth: 1 });
            c.device.cmd_copy_image_to_buffer(cmd, image, vk::ImageLayout::GENERAL, buf, &[region]);
            let hb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::HOST_READ);
            c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::HOST, vk::DependencyFlags::empty(), &[hb], &[], &[]);
        });
        let data = unsafe { std::slice::from_raw_parts(ptr, size as usize) };
        let z = data.iter().filter(|&&b| b == 0).count() * 100 / data.len();
        unsafe {
            c.device.destroy_buffer(buf, None);
            c.device.unmap_memory(mem);
            c.device.free_memory(mem, None);
        }
        Some(format!("gpu image mip {mip}: {z}% zero"))
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

    /// One texel of the cached image a T# resolves to (mip 0, layer 0; depth
    /// images: the aspect the T# reads), with the image's size and format
    /// (pixel watches of captures).
    pub fn texel_of(&mut self, t: &Tsharp, x: u32, y: u32) -> Result<(Vec<u8>, u32, u32, vk::Format), String> {
        let want = Self::key_of(t).ok_or("unsupported format")?;
        let key = match self.lookup(&want, false) {
            Some((k, _)) => k,
            None => self.alias(&want, false).ok_or("not cached")?,
        };
        let image = self.map[&key].image;
        let (x, y) = (x.min(key.width.saturating_sub(1)), y.min(key.height.saturating_sub(1)));
        let aspect = if !is_depth(key.format) {
            vk::ImageAspectFlags::COLOR
        } else if want.format == vk::Format::R8_UINT {
            vk::ImageAspectFlags::STENCIL
        } else {
            vk::ImageAspectFlags::DEPTH
        };
        Ok((self.read_back_texel(image, key.format, aspect, x, y)?, key.width, key.height, key.format))
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
        // Share of zero bytes in the first 64 KiB of the guest data (texture
        // data not loaded yet reads as zeros).
        let n = 64u64 << 10;
        let zeros = if super::cp::guest_readable(t.base, n) {
            let s = unsafe { std::slice::from_raw_parts(t.base as *const u8, n as usize) };
            format!("; guest data {}% zero", s.iter().filter(|&&b| b == 0).count() * 100 / s.len())
        } else {
            "; guest data unreadable".into()
        };
        // Per mip in the sampled range (upload layout): share of zero bytes
        // where each mip's elements are read from.
        let mut per_mip = String::new();
        if let (Some(el), false) = (elem_log2(t.format), key.ty3d) {
            if let Some(lay) = tiling::layout(t.sw_mode, el, key.width, key.height, key.levels, is_block(t.format)) {
                let e = 1u64 << el;
                for m in (t.base_level as usize)..lay.mips.len().min(t.base_level as usize + 4) {
                    let ml = &lay.mips[m];
                    let (mut z, mut n) = (0u64, 0u64);
                    for y in (0..ml.height).step_by((ml.height as usize / 16).max(1)) {
                        for x in (0..ml.width).step_by((ml.width as usize / 16).max(1)) {
                            let a = t.base + lay.offset(m, 0, x, y);
                            if super::cp::guest_readable(a, e) {
                                let s = unsafe { std::slice::from_raw_parts(a as *const u8, e as usize) };
                                z += s.iter().filter(|&&b| b == 0).count() as u64;
                                n += e;
                            }
                        }
                    }
                    if n > 0 {
                        per_mip.push_str(&format!(" m{m}:{}%", z * 100 / n));
                    }
                }
            }
        }
        if !per_mip.is_empty() {
            per_mip = format!("; zero bytes per mip{per_mip}");
        }
        // Uploaded textures: whether the sampled hashes of the source bytes
        // changed since the last upload (what the CPU-write check compares).
        if let Some(img) = self.map.get(&key).filter(|i| !i.gpu_written) {
            if let Some((b, n)) = img.source.and_then(super::track::range) {
                let now = all_hashes(b, n);
                let changed = (0..SLICES).filter(|&s| now[s] != img.hashes[s]).count();
                per_mip.push_str(&format!("; hashed slices changed since upload {changed}/{SLICES} (checked frame {})", img.checked_frame));
            }
        }
        format!("{pick}; images at base: [{}]{zeros}{per_mip}", all.join(", "))
    }

    /// View of a T# (format, type, levels/layers, swizzle) for sampling or storage.
    pub fn view(&mut self, t: &Tsharp, storage: bool, frame: u64) -> Result<(vk::ImageView, vk::Image), String> {
        if storage {
            return self.view_full(t, storage, frame).map(|r| (r.0, r.1));
        }
        // Generations read first: a change made by the full lookup itself
        // (a new image) makes the entry stale at its next use.
        let (lookup, track) = (LOOKUP_GEN.load(std::sync::atomic::Ordering::Acquire), super::track::GEN.load(std::sync::atomic::Ordering::Acquire));
        if let Some(&(f, g, tg, v, i, u)) = self.view_memo.get(&t.raw) {
            if (f == frame || scrubbing()) && g == lookup && tg == track {
                u.stamp(frame);
                // This thread's memo too: without it every thread but the one
                // that made the entry took this lock for each lookup (64% of
                // view lookups missed their memo, run 20261012-073715).
                tls(|m| {
                    if m.views.len() >= 1 << 16 {
                        m.views.clear();
                    }
                    m.views.insert(t.raw, (frame, lookup, track, v, i, u));
                });
                return Ok((v, i));
            }
        }
        let r = self.view_full(t, storage, frame)?;
        if self.view_memo.len() >= 1 << 16 {
            self.view_memo.clear();
        }
        self.view_memo.insert(t.raw, (frame, lookup, track, r.0, r.1, r.2));
        tls(|m| {
            if m.views.len() >= 1 << 16 {
                m.views.clear();
            }
            m.views.insert(t.raw, (frame, lookup, track, r.0, r.1, r.2));
        });
        Ok((r.0, r.1))
    }

    fn view_full(&mut self, t: &Tsharp, storage: bool, frame: u64) -> Result<(vk::ImageView, vk::Image, UseStamp), String> {
        if storage {
            // DS_TRACE_IMAGE_WRITERS=<hex base> or <w>x<h>x<depth>: each
            // shader that binds such an image for writing, once.
            static WATCH: std::sync::OnceLock<Option<(u64, [u32; 3])>> = std::sync::OnceLock::new();
            let w = *WATCH.get_or_init(|| {
                let v = std::env::var("DS_TRACE_IMAGE_WRITERS").ok()?;
                let d: Vec<u32> = v.split('x').filter_map(|x| x.parse().ok()).collect();
                if d.len() == 3 {
                    Some((0, [d[0], d[1], d[2]]))
                } else {
                    u64::from_str_radix(v.trim_start_matches("0x"), 16).ok().map(|b| (b, [0; 3]))
                }
            });
            if let Some((base, dims)) = w {
                static SEEN: Mutex<Option<std::collections::HashSet<(u64, u64)>>> = Mutex::new(None);
                let sh = BINDING_SHADER.with(|s| s.get());
                let hit = if base != 0 { t.base == base } else { [t.width, t.height, t.depth] == dims };
                if hit && SEEN.lock().get_or_insert_with(Default::default).insert((sh, t.base)) {
                    crate::info!("images: shader {sh:#x} writes {:#x} ({}x{}x{} format {}, frame {frame})", t.base, t.width, t.height, t.depth, t.format);
                }
            }
        }
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
        if storage {
            self.write_seq += 1;
        }
        let seq = self.write_seq;
        if storage && matches!(key.format, vk::Format::R32_SFLOAT | vk::Format::R16_UNORM) && self.depth_feeds.insert(key.base) {
            // A possible compute view of a depth plane: memos of
            // `depth_copies_possible` see it from now on.
            LOOKUP_GEN.fetch_add(1, std::sync::atomic::Ordering::Release);
        }
        let img = self.map.get_mut(&key).unwrap();
        if storage {
            img.write_gen = seq;
            img.writer = BINDING_SHADER.with(|s| s.get());
            if !img.gpu_written {
                crate::info!("images: {:#x} {}x{} {:?} becomes a storage image (was uploaded: {}; shader {:#x})", key.base, key.width, key.height, key.format, img.source.is_some(), BINDING_SHADER.with(|s| s.get()));
            }
            if !img.gpu_written {
                LOOKUP_GEN.fetch_add(1, std::sync::atomic::Ordering::Release);
                gpu_images_changed();
            }
            img.gpu_written = true;
            img.source = None;
            self.gpu_bases.insert(key.base);
            self.index_owned(key);
        }
        let img = self.map.get_mut(&key).unwrap();
        img.used.stamp(frame);
        if let Some(&v) = img.views.get(&vkey) {
            return Ok((v, img.image, img.used));
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
        Ok((v, img.image, img.used))
    }

    /// Sampler of an S#; `compare`: used by depth-compare instructions (the
    /// S# compare function applies only to those).
    pub fn sampler(&mut self, s: [u32; 4], compare_use: bool) -> Result<vk::Sampler, String> {
        if let Some(&v) = self.samplers.get(&(s, compare_use)) {
            tls(|m| {
                m.samplers.insert((s, compare_use), v);
            });
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
        tls(|m| {
            m.samplers.insert((s, compare_use), v);
        });
        Ok(v)
    }

    pub fn count(&self) -> usize {
        self.map.len()
    }

    /// One status line (hang reports).
    pub fn status(&self) -> String {
        use std::sync::atomic::Ordering::Relaxed;
        let remaps: Vec<String> = ["commit", "decommit", "protection"]
            .iter()
            .zip(&REMAP_UPLOADS)
            .map(|(what, s)| format!("{what} {} ({} unchanged, {} partial, {} ms)", s[0].load(Relaxed), s[1].load(Relaxed), s[2].load(Relaxed), s[3].load(Relaxed) / 1_000_000))
            .collect();
        let p = &PART_UPLOADS;
        let ww = &crate::mem::WRITE_WATCH_STATS;
        let check = if upload_check() {
            format!(
                " (checked: {} left out, {} changed; {} uploaded again, {} of them unchanged)",
                SUB_CHECKS[0].load(Relaxed),
                SUB_CHECKS[1].load(Relaxed),
                SUB_CHECKS[2].load(Relaxed),
                SUB_CHECKS[3].load(Relaxed)
            )
        } else {
            String::new()
        };
        let g: Vec<u64> = GPU_DETILE.iter().map(|c| c.load(Relaxed)).collect();
        let gpu = if g[0] + g[4] > 0 {
            let checked = if gpu_detile_check() { format!(", {} (mip, layer)s compared with the CPU's detiling of the bytes copied, {} differing, {} tiled ranges changed in guest memory during their upload", g[6], g[7], g[8]) } else { String::new() };
            format!("; detiled on the GPU: {} uploads ({} (mip, layer)s, {} MiB copied, {} MiB detiled, {} ms recording and submitting them; video-memory buffers made {}; transfers on the copy queue {}), {} left to the CPU{checked}", g[0], g[1], g[2] >> 20, g[3] >> 20, g[5] / 1_000_000, g[9], g[10], g[4])
        } else {
            String::new()
        };
        let gpu = if COPY_UPLOADS.load(Relaxed) > 0 { format!("{gpu}; CPU-detiled uploads through the copy queue {}", COPY_UPLOADS.load(Relaxed)) } else { gpu };
        format!(
            "    images: {} cached ({} MiB; GPU-written {} ({} MiB), kept for the session; {} textures evicted, {} MiB, in {} rounds), {} uploads ({} re-uploads; {} MiB, {} ms detiling and copying: 2D {} MiB, {} ms detiling ({} ms choosing (mapping queries, hashes)), {} ms staging, {} ms transfers; staging buffers made {}, waits for uploads in flight {}; batched (`tune upload_batch`) {} in {} command buffers, {} waits{}; pages checked in the mapping table {}, again with the operating system {}); partial re-uploads {} ({} (mip, layer)s uploaded, {} left out, {} of them unchanged written ones, {} MiB){check}; write-watch takes (all) {} ({:.1} us each, {} pages, {} found clean by a probe); GPU-owned images overwritten through buffers: {} cleared ({} right after a zero fill, no wait), {} uploaded; re-uploads after mapping changes: {}{gpu}\n",
            self.map.len(),
            IMAGE_BYTES.load(Relaxed) >> 20,
            self.map.values().filter(|i| i.gpu_written).count(),
            self.map.values().filter(|i| i.gpu_written).map(|i| i.bytes).sum::<u64>() >> 20,
            EVICTED[0].load(Relaxed),
            EVICTED[1].load(Relaxed) >> 20,
            EVICTED[2].load(Relaxed),
            self.uploads,
            self.reuploads,
            self.upload_bytes >> 20,
            self.upload_ns / 1_000_000,
            DETILED.load(Relaxed) >> 20,
            UPLOAD_SPLIT[0].load(Relaxed) / 1_000_000,
            UPLOAD_SPLIT[3].load(Relaxed) / 1_000_000,
            UPLOAD_SPLIT[1].load(Relaxed) / 1_000_000,
            UPLOAD_SPLIT[2].load(Relaxed) / 1_000_000,
            STAGING_MADE.load(Relaxed),
            UPLOAD_WAITS.load(Relaxed),
            BATCHED_STATS[0].load(Relaxed),
            BATCHED_STATS[1].load(Relaxed),
            BATCHED_STATS[2].load(Relaxed),
            predetile_line(),
            UPLOAD_TABLE[0].load(Relaxed),
            UPLOAD_TABLE[1].load(Relaxed),
            p[0].load(Relaxed),
            p[1].load(Relaxed),
            p[2].load(Relaxed),
            SUB_SAME.load(Relaxed),
            p[3].load(Relaxed) >> 20,
            ww[0].load(Relaxed),
            ww[1].load(Relaxed) as f64 / ww[0].load(Relaxed).max(1) as f64 / 1000.0,
            ww[2].load(Relaxed),
            crate::mem::PROBE_CLEAN.load(Relaxed),
            self.overwrite_clears,
            ZERO_FILL_CLEARS.load(std::sync::atomic::Ordering::Relaxed),
            self.overwrite_uploads,
            remaps.join(", ")
        ) + &zero_census_line()
    }
}
