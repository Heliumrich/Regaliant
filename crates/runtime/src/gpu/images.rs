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
    drop(g);
    if out.0.is_empty() {
        return;
    }
    let rounds = EVICTED[2].fetch_add(1, Relaxed) + 1;
    if rounds <= 20 || rounds % 100 == 0 {
        crate::info!(
            "images: video memory {} of budget {} MiB at flip {frame}: evicted {} textures ({} MiB), {} images cached ({} MiB)",
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

/// A host-visible buffer texture data is uploaded from (`Images::staging`).
struct Staging {
    buf: vk::Buffer,
    mem: vk::DeviceMemory,
    ptr: *mut u8,
    cap: u64,
}

/// Staging buffers kept for reuse: up to this many bytes, each of at most
/// STAGING_POOLED bytes (larger ones are made for their upload alone).
const STAGING_KEEP: u64 = 128 << 20;
const STAGING_POOLED: u64 = 32 << 20;
/// Uploads in flight, and their staging bytes, before the next one waits
/// for the oldest. Unbounded bytes (up to 32 uploads of ~85 MB arrays, freed
/// only at a later upload) made guest-memory imports fail by the ten
/// thousand and preceded device losses (runs 20261013-083710, -084645).
const MAX_INFLIGHT: usize = 32;
const MAX_INFLIGHT_BYTES: u64 = 128 << 20;
/// Uploads that waited for an earlier one (`MAX_INFLIGHT`), and staging
/// buffers made.
static UPLOAD_WAITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static STAGING_MADE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

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
    /// Staging buffers not in use (`staging`), uploads submitted without
    /// waiting (command buffer, fence, the staging buffer it reads; oldest
    /// first) and the command buffers and fences of completed ones.
    staging_free: Vec<Staging>,
    inflight: std::collections::VecDeque<(vk::CommandBuffer, vk::Fence, Staging)>,
    spare_cmds: Vec<(vk::CommandBuffer, vk::Fence)>,
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
        std::thread::sleep(std::time::Duration::from_millis(if marks { 10 } else { 2 }));
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

pub fn with<R>(f: impl FnOnce(&mut Images) -> R) -> R {
    static SCRUBBER: std::sync::Once = std::sync::Once::new();
    SCRUBBER.call_once(|| {
        if let Err(e) = crate::thread::spawn_host("gpu-texscrub", 1 << 20, Box::new(scrubber)) {
            crate::warn!("images: no texture scrubber thread ({e})");
        }
    });
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
            *g = Some(Images { map: HashMap::default(), by_base: HashMap::default(), gpu_bases: Default::default(), stencil_of: HashMap::default(), meta_of: Default::default(), last_note: None, note_gen: 0, view_memo: HashMap::default(), key_memo: HashMap::default(), pending_clears: HashMap::default(), samplers: HashMap::default(), pool, cmd, fence, staging_free: Vec::new(), inflight: Default::default(), spare_cmds: Vec::new(), reported: HashMap::default(), uploads: 0, reuploads: 0, upload_bytes: 0, upload_ns: 0, write_seq: 0, owned: Default::default(), max_span: 0, pending_overwrites: 0, overwritten_keys: Default::default(), pending_change: 0, pending_published: 0, scrub_keys: Vec::new(), scrub_gen: 0, scrub_at: 0, overwrite_clears: 0, overwrite_uploads: 0, partial_seen: Default::default(), dummies: HashMap::default(), depth_grows: 0, depth_feeds: Default::default() });
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
            let image = c.device.create_image(&ci, None).map_err(|e| format!("vkCreateImage {key:?}: {e}"))?;
            let req = c.device.get_image_memory_requirements(image);
            let mem = c.alloc(req, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
            c.device.bind_image_memory(image, mem, 0).map_err(|e| format!("bind image: {e}"))?;
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
        let (buf, mem, ptr) = ctx().host_buffer(cap, vk::BufferUsageFlags::TRANSFER_SRC)?;
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
        while let Some(&(_, fence, _)) = self.inflight.front() {
            let done = unsafe { c.device.get_fence_status(fence) }.unwrap_or(true);
            if !done {
                let bytes: u64 = self.inflight.iter().map(|x| x.2.cap).sum();
                if self.inflight.len() <= max && bytes <= max_bytes {
                    break;
                }
                UPLOAD_WAITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if let Err(e) = unsafe { c.device.wait_for_fences(&[fence], true, 10_000_000_000) } {
                    if !super::exec::DEVICE_LOST.swap(true, std::sync::atomic::Ordering::SeqCst) {
                        crate::error!("images: upload failed: {e}; GPU execution stops; checkpoints: {}", super::vk::checkpoint_report());
                    }
                }
            }
            let (cmd, fence, s) = self.inflight.pop_front().unwrap();
            unsafe {
                c.device.reset_fences(&[fence]).ok();
                c.device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty()).ok();
            }
            self.spare_cmds.push((cmd, fence));
            self.release_staging(s);
        }
    }

    /// Records and submits a transfer reading staging buffer `s` without
    /// waiting for it: on the one queue it runs after the work submitted
    /// before it and before batches submitted later, and its barriers order
    /// it against the image's other uses; `reap` recycles it.
    fn submit_upload(&mut self, s: Staging, f: impl FnOnce(vk::CommandBuffer)) {
        self.reap(MAX_INFLIGHT - 1, MAX_INFLIGHT_BYTES.saturating_sub(s.cap));
        let c = ctx();
        let (cmd, fence) = match self.spare_cmds.pop() {
            Some(x) => x,
            None => unsafe {
                let cmd = c.device.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(self.pool).command_buffer_count(1)).expect("upload cmd")[0];
                (cmd, c.device.create_fence(&vk::FenceCreateInfo::default(), None).expect("upload fence"))
            },
        };
        unsafe {
            c.device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).ok();
            f(cmd);
            c.device.end_command_buffer(cmd).ok();
            match c.submit(cmd, &[], &[], fence) {
                Ok(()) => self.inflight.push_back((cmd, fence, s)),
                Err(e) => {
                    if !super::exec::DEVICE_LOST.swap(true, std::sync::atomic::Ordering::SeqCst) {
                        crate::error!("images: submit failed: {e}; GPU execution stops; checkpoints: {}", super::vk::checkpoint_report());
                    }
                    c.device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty()).ok();
                    self.spare_cmds.push((cmd, fence));
                    self.release_staging(s);
                }
            }
        }
    }

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
        if let Ok((_, n)) = &r {
            self.upload_bytes += n;
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
        // Pages checked and read under the guest-reader lock: the title may
        // unmap layers of a streamed array meanwhile (`mem::GUEST_READERS`).
        let t_detile = std::time::Instant::now();
        let readers = crate::mem::GUEST_READERS.read();
        let whole = super::cp::valid(t.base, size) && crate::mem::os_readable(t.base, size);
        let mapped = |mip: usize, layer: u32| -> bool {
            let (s, n) = range(mip, layer);
            whole || (super::cp::valid(t.base + s, n) && crate::mem::os_readable(t.base + s, n))
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
                if sums_on {
                    let h = hash_bytes(t.base + s, n as usize);
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
        // Detile every (mip, layer) into one staging buffer; unmapped ones
        // of a first upload copy from one shared zero area at the end.
        let mut staging: Vec<u8> = Vec::new();
        let mut regions = Vec::new();
        let mut zeroed: Vec<(usize, u32, usize)> = Vec::new();
        let mut tmp = Vec::new();
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
        // Guest memory is no longer read.
        drop(readers);
        UPLOAD_SPLIT[0].fetch_add(t_detile.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        DETILED.fetch_add(staging.len() as u64, std::sync::atomic::Ordering::Relaxed);
        if sums_on {
            SUB_SUMS.lock().get_or_insert_with(HashMap::default).insert(key, sums);
        }
        if precise {
            use std::sync::atomic::Ordering::Relaxed;
            PART_UPLOADS[0].fetch_add(1, Relaxed);
            PART_UPLOADS[1].fetch_add(parts_done, Relaxed);
            PART_UPLOADS[2].fetch_add(parts_left, Relaxed);
            PART_UPLOADS[3].fetch_add(staging.len() as u64, Relaxed);
        }
        if regions.is_empty() && zeroed.is_empty() {
            return Err(format!("texture {:#x}+{size:#x}: no mip mapped", t.base));
        }
        if let Some(zero_len) = zeroed.iter().map(|z| z.2).max() {
            let off = staging.len() as u64;
            staging.resize(staging.len() + zero_len, 0);
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
        let c = ctx();
        let t_stage = std::time::Instant::now();
        let asynchronous = super::tune::upload_async();
        let st = if asynchronous {
            self.staging(staging.len().max(4) as u64)?
        } else {
            let (buf, mem, ptr) = c.host_buffer(staging.len().max(4) as u64, vk::BufferUsageFlags::TRANSFER_SRC)?;
            Staging { buf, mem, ptr, cap: staging.len().max(4) as u64 }
        };
        let (buf, mem) = (st.buf, st.mem);
        unsafe { std::ptr::copy_nonoverlapping(staging.as_ptr(), st.ptr, staging.len()) };
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
        if asynchronous {
            self.submit_upload(st, record);
            UPLOAD_SPLIT[2].fetch_add(t_copy.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
            self.uploads += 1;
            return Ok((t.base, size));
        }
        self.one_shot(record);
        let t_free = std::time::Instant::now();
        UPLOAD_SPLIT[2].fetch_add((t_free - t_copy).as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        unsafe {
            c.device.destroy_buffer(buf, None);
            c.device.unmap_memory(mem);
            c.device.free_memory(mem, None);
        }
        UPLOAD_SPLIT[1].fetch_add(t_free.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
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
        let readers = crate::mem::GUEST_READERS.read();
        if !(super::cp::valid(t.base, size) && crate::mem::os_readable(t.base, size)) {
            return Err(format!("3D texture {:#x}+{size:#x} not mapped", t.base));
        }
        let src = unsafe { std::slice::from_raw_parts(t.base as *const u8, size as usize) };
        let mut staging = Vec::new();
        lay.detile(src, &mut staging);
        drop(readers);
        let c = ctx();
        let (buf, mem, ptr) = c.host_buffer(staging.len().max(4) as u64, vk::BufferUsageFlags::TRANSFER_SRC)?;
        unsafe { std::ptr::copy_nonoverlapping(staging.as_ptr(), ptr, staging.len()) };
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
    fn image_for(&mut self, t: &Tsharp, frame: u64, storage: bool) -> Result<ImageKey, String> {
        let key = Self::key_of(t).ok_or_else(|| format!("image format {} unsupported", t.format))?;
        if let Some((k, how)) = self.lookup(&key, storage) {
            if how == "exact" && !self.map[&k].gpu_written {
                self.refresh(key, t, frame)?;
            }
            if how == "exact" {
                self.note_layout(k, t);
            }
            return Ok(k);
        }
        let (image, mem, bytes) = self.create(key)?;
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
                self.upload_parts(image, key, t, None, false)?;
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
                self.upload_parts(image, key, t, Some(&[]), false)?;
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
            self.upload_parts(image, key, t, parts.as_deref(), matches!(why, super::track::Why::Forced))?;
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

    /// The GPU image last rendered at `base` (display buffers).
    pub fn rendered_at(&self, base: u64) -> Option<(vk::Image, vk::Format, u32, u32)> {
        self.map
            .values()
            .filter(|i| i.key.base == base && i.gpu_written && !is_depth(i.key.format))
            .max_by_key(|i| i.key.width as u64 * i.key.height as u64)
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
        format!(
            "    images: {} cached ({} MiB; {} textures evicted, {} MiB, in {} rounds), {} uploads ({} re-uploads; {} MiB, {} ms detiling and copying: 2D {} MiB, {} ms detiling ({} ms choosing (mapping queries, hashes)), {} ms staging, {} ms transfers; staging buffers made {}, waits for uploads in flight {}); partial re-uploads {} ({} (mip, layer)s uploaded, {} left out, {} of them unchanged written ones, {} MiB){check}; write-watch takes (all) {} ({:.1} us each, {} pages, {} found clean by a probe); GPU-owned images overwritten through buffers: {} cleared ({} right after a zero fill, no wait), {} uploaded; re-uploads after mapping changes: {}\n",
            self.map.len(),
            IMAGE_BYTES.load(Relaxed) >> 20,
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
        )
    }
}
