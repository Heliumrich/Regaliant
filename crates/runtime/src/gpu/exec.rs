//! Vulkan execution of GPU work (docs/RENDERER.md milestone 1: compute).
//!
//! Each command-processor thread owns an `Exec` that records its queue's
//! dispatches into a Vulkan command buffer. Guest buffers are imported in
//! place (VK_EXT_external_memory_host windows over guest mappings), so shader
//! writes land directly in guest memory. Recorded work is submitted and
//! waited for (`flush`) before the command processor touches memory itself
//! (labels, DMA, waits, flips) and at the end of every submission, which keeps
//! CPU/GPU and cross-queue ordering as on hardware. Every command buffer
//! starts with a full memory barrier, which orders it after all earlier
//! submissions of every queue.
//!
//! Enabled with `DS_GPU=1` (dispatches are otherwise only decoded).

use super::cp::{Cp, Space};
use super::shaders::{self, Prepared, Resolved};
use super::verify;
use super::vk::ctx;
use ash::vk;
use parking_lot::Mutex;
use rdna::translate::{BindingKind, RES_INFO_DWORDS};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};

pub mod draw;
pub mod kept;
pub mod rgm;
pub(crate) use draw::{color_surface, depth_surface, set_equal_bias, toggle_skip_vs};

/// Dispatches of compute shaders on this list are skipped (`skipcs` control
/// command; diagnosis).
static SKIP_CS: parking_lot::Mutex<Option<std::collections::HashSet<u64>>> = parking_lot::Mutex::new(None);
/// SKIP_CS holds an address (every dispatch of every recording thread took
/// its lock to find it empty).
static SKIP_CS_ANY: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Adds or removes a compute shader address; returns whether it is listed now.
pub fn toggle_skip_cs(addr: u64) -> bool {
    let mut g = SKIP_CS.lock();
    let s = g.get_or_insert_with(Default::default);
    let listed = if s.remove(&addr) {
        false
    } else {
        s.insert(addr);
        true
    };
    SKIP_CS_ANY.store(!s.is_empty(), Ordering::Release);
    listed
}

/// Whether `cp`'s compute shader is on the skip list.
fn skip_cs(cp: &Cp) -> bool {
    SKIP_CS_ANY.load(Ordering::Acquire) && SKIP_CS.lock().as_ref().is_some_and(|s| s.contains(&cs_addr(cp)))
}

fn cs_addr(cp: &Cp) -> u64 {
    ((cp.reg(Space::Sh, 0x20C) as u64) | ((cp.reg(Space::Sh, 0x20D) as u64 & 0xFF) << 32)) << 8
}
pub use draw::{DrawArgs, GpuArgs, NEEDS_CPU_ARGS};

pub static EXECUTED: AtomicU64 = AtomicU64::new(0);
/// The Vulkan device was lost: GPU execution stops (reported once).
pub static DEVICE_LOST: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// `DS_GPU_SYNC=1`: complete every dispatch before the next (diagnosis of
/// device losses: the failing dispatch is named).
fn sync_mode() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_GPU_SYNC").ok().as_deref() == Some("1"))
}

/// Compute shaders whose dispatches are completed on the GPU right after
/// they are recorded (`tune sync_cs`), as the console runs them shortly
/// after its command processor reaches them: the title rewrites their
/// inputs soon after issuing them. By default two one-group kernels of
/// the level loads (in the base shader pack of 01.007.000) that turn
/// 52-byte records into a 1.5 MB buffer: run later, with inputs the title
/// had rewritten, they left World 8-3 in a different, dark state until the
/// next boot (6 of 10 boots with `forward_copies`, 0 of 4 completed at
/// once; FINDINGS.md 2026-10-14). `DS_SYNC_CS=<hex address>[,...]` replaces
/// the list (empty: none); `tune race_checks` finds such inputs (changes
/// with no CP or GPU writer).
fn sync_cs(addr: u64) -> bool {
    const DEFAULT: [u64; 2] = [0x10_089c_a600, 0x10_089d_0b00];
    static SET: std::sync::OnceLock<Vec<u64>> = std::sync::OnceLock::new();
    let s = SET.get_or_init(|| match std::env::var("DS_SYNC_CS") {
        Ok(v) => v.split(',').filter_map(|v| u64::from_str_radix(v.trim().trim_start_matches("0x"), 16).ok()).collect(),
        Err(_) => DEFAULT.to_vec(),
    });
    !s.is_empty() && super::tune::sync_cs() && s.contains(&addr)
}
pub static SKIPPED: AtomicU64 = AtomicU64::new(0);
pub static FLUSHES: AtomicU64 = AtomicU64::new(0);
/// Completions waited for so the CP could read memory, by reason: CP reads
/// (indirect arguments, labels, ...), dispatch shader reads (descriptors and
/// scalar loads), dispatch textures, draw shader reads, draw textures, draw
/// indices (`gpu exec` status line).
pub static SYNC_WHY: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];
/// SYNC_WHY at the last `profile reset` (per-window counts).
static SYNC_WHY_AT_RESET: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];
pub const SYNC_CP: usize = 0;
pub const SYNC_CS_READS: usize = 1;
pub const SYNC_CS_TEXTURES: usize = 2;
pub const SYNC_DRAW_READS: usize = 3;
pub const SYNC_DRAW_TEXTURES: usize = 4;
pub const SYNC_DRAW_INDICES: usize = 5;

/// Pipelines created (compute, graphics): new translations or states, and
/// the time their creation took (ns; cache hits are quick).
pub static PIPES_CREATED: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
pub static PIPES_NS: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
/// Dispatches whose compute pipeline was missing and the nanoseconds their
/// command processors spent creating it or waiting for it (first uses).
pub static FIRST_USES: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
/// Of those, pipelines whose descriptors are pushed (`tune::push_desc`).
pub static PIPES_PUSH: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
/// Commands recorded with pushed descriptors, with a descriptor set.
pub static PUSHED: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

thread_local! {
    /// This thread's PUSHED counts not yet added (every 256: one shared
    /// counter bumped by every draw and dispatch of every recording thread).
    static PUSHED_LOCAL: std::cell::Cell<[u32; 2]> = const { std::cell::Cell::new([0; 2]) };
}

fn note_pushed(k: usize) {
    PUSHED_LOCAL.with(|c| {
        let mut v = c.get();
        v[k] += 1;
        if v[k] == 256 {
            PUSHED[k].fetch_add(256, Ordering::Relaxed);
            v[k] = 0;
        }
        c.set(v);
    });
}

/// The Vulkan commands that shape the GPU's work, counted while profiling
/// (`profile report`): global barriers, render pass begins, graphics and
/// compute pipeline binds, descriptor pushes or set binds.
pub(super) static VK_OPS: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];
/// Global barriers by the source line that asked for them (profiling).
static BARRIER_SITES: Mutex<Option<HashMap<(&'static str, u32), u64>>> = Mutex::new(None);
/// The same by queue class (graphics CP, other compute queues, queue 0x50,
/// parallel workers): `Exec::barrier`'s callers (profiling).
static BARRIER_QUEUE_SITES: Mutex<Option<HashMap<(u8, &'static str, u32), u64>>> = Mutex::new(None);
pub(super) const VK_BARRIERS: usize = 0;
pub(super) const VK_RENDER_PASSES: usize = 1;
pub(super) const VK_GFX_PIPELINES: usize = 2;
pub(super) const VK_CS_PIPELINES: usize = 3;
pub(super) const VK_DESCRIPTORS: usize = 4;

#[inline]
pub(super) fn note_vk(k: usize) {
    if profiling() {
        VK_OPS[k].fetch_add(1, Ordering::Relaxed);
    }
}

/// Waits for a stage's shader reads (GPU work was writing what the
/// preparation read): [draws whose words were the same after the wait,
/// changed; dispatches same, changed]. Same: preparing with the earlier
/// values instead of waiting would have been right.
pub(super) static READ_WAIT_WORDS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

fn sync_why_status() -> String {
    let n: Vec<u64> = SYNC_WHY.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    let w: Vec<u64> = READ_WAIT_WORDS.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    format!(
        "    shader-read waits whose words were unchanged after the wait: draws {} of {}, dispatches {} of {}; commands that did not wait {} ({} data loads copied by batch prologues), stages used as prepared {} (checks held {}, failed {}, {} shaders wait again)\n",
        w[0],
        w[0] + w[1],
        w[2],
        w[2] + w[3],
        SPEC_STATS[3].load(Ordering::Relaxed),
        SPEC_STATS[4].load(Ordering::Relaxed),
        SPEC_STATS[0].load(Ordering::Relaxed),
        SPEC_STATS[1].load(Ordering::Relaxed),
        SPEC_STATS[2].load(Ordering::Relaxed),
        NO_SPEC.lock().as_ref().map_or(0, |n| n.len())
    ) + &format!(
        "    CP read waits: {} CP reads, {} dispatch shader reads, {} dispatch textures, {} draw shader reads, {} draw textures, {} draw indices; pipelines created: {} compute ({} ms, {} with pushed descriptors), {} graphics ({} ms, {} pushed); commands with pushed descriptors {}, with sets {}\n",
        n[0],
        n[1],
        n[2],
        n[3],
        n[4],
        n[5],
        PIPES_CREATED[0].load(Ordering::Relaxed),
        PIPES_NS[0].load(Ordering::Relaxed) / 1_000_000,
        PIPES_PUSH[0].load(Ordering::Relaxed),
        PIPES_CREATED[1].load(Ordering::Relaxed),
        PIPES_NS[1].load(Ordering::Relaxed) / 1_000_000,
        PIPES_PUSH[1].load(Ordering::Relaxed),
        PUSHED[0].load(Ordering::Relaxed),
        PUSHED[1].load(Ordering::Relaxed)
    ) + &format!(
        "    first uses on the command processors: {} compute pipelines missing at their dispatch ({} ms creating or waiting for them); {} graphics pipelines missing at their draw ({} ms; {} of them waited for one in flight, {} ms), {} created for a shader pair seen before with other render state ({} ms)\n",
        FIRST_USES[0].load(Ordering::Relaxed),
        FIRST_USES[1].load(Ordering::Relaxed) / 1_000_000,
        draw::GFX_FIRST_USES[0].load(Ordering::Relaxed),
        draw::GFX_FIRST_USES[1].load(Ordering::Relaxed) / 1_000_000,
        draw::GFX_FLIGHT_WAITS[0].load(Ordering::Relaxed),
        draw::GFX_FLIGHT_WAITS[1].load(Ordering::Relaxed) / 1_000_000,
        draw::GFX_REPEAT_PAIRS[0].load(Ordering::Relaxed),
        draw::GFX_REPEAT_PAIRS[1].load(Ordering::Relaxed) / 1_000_000
    )
}

/// Time the completion thread spent waiting for fences / running actions (ns).
pub static FENCE_NS: AtomicU64 = AtomicU64::new(0);
pub static ACTION_NS: AtomicU64 = AtomicU64::new(0);
/// Watchdog record of `Fixed` (host visible; null before GPU execution starts).
static WATCHDOG_PTR: std::sync::atomic::AtomicPtr<u32> = std::sync::atomic::AtomicPtr::new(std::ptr::null_mut());
/// Invocations stopped by the loop watchdog (`rdna::translate::LOOP_LIMIT`).
pub static WATCHDOG_STOPS: AtomicU64 = AtomicU64::new(0);
/// Fast clears performed (metadata writes, see `Exec::resolve_fast_clears`).
pub static FAST_CLEARS: AtomicU64 = AtomicU64::new(0);
/// Fill-kernel dispatches recorded as buffer fills (`Exec::fill_in_place`).
pub static FILLS_IN_PLACE: AtomicU64 = AtomicU64::new(0);
/// Of them, zero fills over GPU-written images only that cleared the
/// images without storing into guest memory (`tune fill_skip`), and their
/// bytes.
pub static FILL_SKIPS: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
/// Fills `tune fill_skip` considers (smaller ones store as before).
const FILL_SKIP_MIN: u64 = 256 << 10;
/// Logs why a big read-only binding stays in place, once per range and
/// reason (the first 48).
fn note_big_refusal(base: u64, size: u64, why: &'static str) {
    static SEEN: Mutex<Option<std::collections::HashSet<(u64, u64, &'static str)>>> = Mutex::new(None);
    let mut g = SEEN.lock();
    let seen = g.get_or_insert_with(Default::default);
    if seen.len() < 48 && seen.insert((base, size, why)) {
        crate::info!("gpu: read-only binding {base:#x}+{size:#x} ({} MiB) read in place: {why}", size >> 20);
    }
}

/// Dispatches whose global write notes were skipped (`Exec::last_notes`).
pub static NOTES_SKIPPED: AtomicU64 = AtomicU64::new(0);

/// The generations the global write notes depend on (`Exec::last_notes`).
fn note_gens() -> [u64; 4] {
    [
        super::images::LOOKUP_GEN.load(Ordering::Acquire),
        super::images::NOTE_GEN.load(Ordering::Acquire),
        super::track::RESET_GEN.load(Ordering::Acquire),
        super::precache::INSERT_GEN.load(Ordering::Acquire),
    ]
}
/// Fast clears resolved without waiting for the GPU (metadata value known
/// from the fill that wrote it).
pub static FAST_CLEARS_KNOWN: AtomicU64 = AtomicU64::new(0);
/// Block-compressed textures refreshed from GPU-written block views.
pub static BLOCK_COPIES: AtomicU64 = AtomicU64::new(0);
/// Of those, with the mip tail from a view of the tail block (`tune block_tails`).
pub static BLOCK_TAIL_COPIES: AtomicU64 = AtomicU64::new(0);
/// Depth planes refreshed from GPU-written color views (`sync_depth_views`).
pub static DEPTH_COPIES: AtomicU64 = AtomicU64::new(0);
/// Storage images bound one view per mip level (image_store_mip /
/// image_load_mip programs).
pub static MIP_VIEW_BINDINGS: AtomicU64 = AtomicU64::new(0);
/// Guest-memory imports: count, bytes, how many spanned several mappings,
/// and how many were dropped because their pages were remapped.
static IMPORTS: AtomicU64 = AtomicU64::new(0);
static IMPORT_BYTES: AtomicU64 = AtomicU64::new(0);
static IMPORTS_MULTI: AtomicU64 = AtomicU64::new(0);
static IMPORTS_STALE: AtomicU64 = AtomicU64::new(0);
/// The profile window's imports (`import_window_line`): [imports, bytes,
/// ns, longest ns, span-list lock acquisitions that waited, ns waited].
static IMPORT_WIN: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];
static CACHE_GEN_AT_RESET: AtomicU64 = AtomicU64::new(0);

/// Clears the window's import counts (`profile reset`).
pub(super) fn reset_import_window() {
    for c in &IMPORT_WIN {
        c.store(0, Ordering::Relaxed);
    }
    CACHE_GEN_AT_RESET.store(CACHE_GEN.load(Ordering::Relaxed), Ordering::Relaxed);
}

/// One profile line: the window's guest-memory imports, waits for the span
/// list's lock and the times imports left the list (each clears every
/// thread's span cache).
pub(super) fn import_window_line() -> String {
    let v: Vec<u64> = IMPORT_WIN.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    format!(
        "    guest-memory imports in the window: {} ({} MiB, {} ms, longest {} ms); span-list lock waits {} ({} ms); span caches cleared {} times; imports dropped for a mapping change while in progress {}\n",
        v[0],
        v[1] >> 20,
        v[2] / 1_000_000,
        v[3] / 1_000_000,
        v[4],
        v[5] / 1_000_000,
        CACHE_GEN.load(Ordering::Relaxed).saturating_sub(CACHE_GEN_AT_RESET.load(Ordering::Relaxed)),
        IMPORTS_RACED.load(Ordering::Relaxed)
    )
}

/// The span list's lock, counting the time waited when another thread
/// holds it for writing (an import in progress).
fn spans_read() -> parking_lot::RwLockReadGuard<'static, Vec<Span>> {
    if let Some(g) = SPANS.try_read() {
        return g;
    }
    let t = std::time::Instant::now();
    let g = SPANS.read();
    IMPORT_WIN[4].fetch_add(1, Ordering::Relaxed);
    IMPORT_WIN[5].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    super::hitch::note_since(super::hitch::Kind::Mirror, t, || "waiting for the import list (an import or a remapping)".into());
    g
}

fn spans_write() -> parking_lot::RwLockWriteGuard<'static, Vec<Span>> {
    if let Some(g) = SPANS.try_write() {
        return g;
    }
    let t = std::time::Instant::now();
    let g = SPANS.write();
    IMPORT_WIN[4].fetch_add(1, Ordering::Relaxed);
    IMPORT_WIN[5].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    super::hitch::note_since(super::hitch::Kind::Mirror, t, || "waiting for the import list to change it".into());
    g
}

/// GPU execution enabled (`DS_GPU=1`) and Vulkan available.
/// `DS_TRACE_WRITES=<hex bytes>`: logs once each shader (and CP DMA) that
/// writes a guest range at least that large (who fills a large buffer).
pub fn trace_writes(what: impl FnOnce() -> String, base: u64, size: u64) {
    static MIN: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    let min = *MIN.get_or_init(|| std::env::var("DS_TRACE_WRITES").ok().and_then(|v| u64::from_str_radix(v.trim_start_matches("0x"), 16).ok()).unwrap_or(0));
    if min == 0 || size < min {
        return;
    }
    let what = what();
    static SEEN: Mutex<Option<HashSet<(String, u64, u64)>>> = Mutex::new(None);
    if SEEN.lock().get_or_insert_with(HashSet::new).insert((what.clone(), base, size)) {
        crate::info!("writes: {what} writes {base:#x}+{size:#x} (flip {})", super::FRAMES.load(Ordering::Relaxed));
    }
}

pub fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        if std::env::var("DS_GPU").ok().as_deref() != Some("1") {
            return false;
        }
        match super::vk::init() {
            Ok(()) => {
                crate::info!("gpu: executing compute dispatches on Vulkan (DS_GPU=1)");
                super::resident::init();
                super::fe2::init();
                super::resident::section_test_maybe();
                super::resident::mixed_test_maybe();
                super::resident::protect_test_maybe();
                true
            }
            Err(e) => {
                crate::error!("gpu: DS_GPU=1 but Vulkan is unavailable: {e}");
                false
            }
        }
    })
}

/// Imported guest-memory window.
struct Span {
    start: u64,
    end: u64,
    buffer: vk::Buffer,
    mem: vk::DeviceMemory,
    /// Still backed by the pages it was imported with (cleared when guest
    /// pages in its range are committed or decommitted).
    valid: bool,
    /// The flip it was imported at.
    born: u64,
    /// Front end 2: the sequence of the first queued batch that uploaded
    /// from it, and the flip that batch was first seen completed (u64::MAX:
    /// none yet). Its pages are armed only SETTLE flips after that: the
    /// driver pins an import's pages after `vkAllocateMemory` returns
    /// (protecting a new import's pages lost the device, FINDINGS.md
    /// 2026-10-26), and an import first used long after its creation (a
    /// batch dropped past a label wait) was settled by age alone.
    first_use: AtomicU64,
    used_at: AtomicU64,
}

/// Imports whose guest pages were remapped (`revalidate_spans`), freed once
/// no batch can still use them (`reap_spans`): with SPAN_GEN after their
/// invalidation and, once no unsubmitted batch began before that, the last
/// batch sequence submitted then. They pin the pages they were imported
/// with: kept forever, they pinned more host memory at every level change
/// until imports failed (run 20261011-224123).
struct Dead {
    span: Span,
    generation: u64,
    seq: Option<u64>,
}

static GRAVEYARD: Mutex<Vec<Dead>> = Mutex::new(Vec::new());
static GRAVEYARD_LEN: AtomicU64 = AtomicU64::new(0);
/// Imports freed after remapping: [count, bytes].
static SPANS_FREED: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// Per executor, SPAN_GEN when its oldest batch not yet submitted began
/// (u64::MAX: none). Imports are looked up only while a batch records, so a
/// batch that began after an import was dropped cannot use it.
static EXEC_GENS: Mutex<Vec<std::sync::Arc<AtomicU64>>> = Mutex::new(Vec::new());

/// Whether an import, in use or dropped and not freed yet, may cover guest
/// [base, base+len): true also while the lists are busy (callers inside the
/// memory manager's lock only try them: `import_span` holds them while it
/// takes that lock).
pub(crate) fn imports_may_cover(base: u64, len: u64) -> bool {
    let end = base.saturating_add(len);
    // (Imports in progress too: the driver pins their pages before they are
    // listed.)
    {
        let Some(m) = IMPORTING.try_lock() else { return true };
        if m.iter().any(|x| x.start < end && base < x.end) {
            return true;
        }
    }
    let Some(g) = SPANS.try_read() else { return true };
    if g.iter().any(|s| s.start < end && base < s.end) {
        return true;
    }
    drop(g);
    let Some(gy) = GRAVEYARD.try_lock() else { return true };
    gy.iter().any(|d| d.span.start < end && base < d.span.end)
}

/// Frees dropped imports and images no batch can use any more (completion
/// thread), evicts images when video memory runs short, then decommits the
/// guest ranges that waited for imports (`mem::retry_deferred`).
fn reap_spans() {
    reap_dead_spans();
    reap_dead_images();
    super::images::maybe_evict();
    crate::mem::retry_deferred();
}

/// Images the cache dropped (`images::Evicted`), destroyed once no batch can
/// use them: as dead imports, SPAN_GEN when buried and, once every batch
/// begun before was submitted, the last batch sequence submitted then.
struct DeadImage {
    img: super::images::Evicted,
    generation: u64,
    seq: Option<u64>,
}

static IMAGE_GRAVE: Mutex<Vec<DeadImage>> = Mutex::new(Vec::new());
static IMAGE_GRAVE_LEN: AtomicU64 = AtomicU64::new(0);

/// Images the cache dropped, to destroy once no batch can name them (a
/// batch can only name images it looked up while it recorded).
pub(crate) fn bury_images(v: Vec<super::images::Evicted>) {
    if v.is_empty() {
        return;
    }
    let generation = SPAN_GEN.fetch_add(1, Ordering::AcqRel) + 1;
    let mut g = IMAGE_GRAVE.lock();
    g.extend(v.into_iter().map(|img| DeadImage { img, generation, seq: None }));
    IMAGE_GRAVE_LEN.store(g.len() as u64, Ordering::Release);
}

fn reap_dead_images() {
    if IMAGE_GRAVE_LEN.load(Ordering::Acquire) == 0 {
        return;
    }
    let oldest_open = EXEC_GENS.lock().iter().map(|g| g.load(Ordering::SeqCst)).min().unwrap_or(u64::MAX);
    let next = NEXT_SEQ.load(Ordering::SeqCst);
    let done = done_seq();
    let mut g = IMAGE_GRAVE.lock();
    g.retain_mut(|d| {
        if d.seq.is_none() && oldest_open >= d.generation {
            d.seq = Some(next - 1);
        }
        match d.seq {
            Some(s) if done >= s && !DEVICE_LOST.load(Ordering::Relaxed) => {
                unsafe {
                    for &v in &d.img.views {
                        ctx().device.destroy_image_view(v, None);
                    }
                    ctx().device.destroy_image(d.img.image, None);
                    ctx().device.free_memory(d.img.mem, None);
                }
                false
            }
            _ => true,
        }
    });
    IMAGE_GRAVE_LEN.store(g.len() as u64, Ordering::Release);
}

fn reap_dead_spans() {
    if GRAVEYARD_LEN.load(Ordering::Acquire) == 0 {
        return;
    }
    let oldest_open = EXEC_GENS.lock().iter().map(|g| g.load(Ordering::SeqCst)).min().unwrap_or(u64::MAX);
    let next = NEXT_SEQ.load(Ordering::SeqCst);
    let done = done_seq();
    let mut g = GRAVEYARD.lock();
    g.retain_mut(|d| {
        if d.seq.is_none() && oldest_open >= d.generation {
            // Every batch that may use it was submitted by now.
            d.seq = Some(next - 1);
        }
        match d.seq {
            Some(s) if done >= s && !DEVICE_LOST.load(Ordering::Relaxed) => {
                super::vk::note_gpu_freed(d.span.buffer);
                unsafe {
                    ctx().device.destroy_buffer(d.span.buffer, None);
                    ctx().device.free_memory(d.span.mem, None);
                }
                SPANS_FREED[0].fetch_add(1, Ordering::Relaxed);
                SPANS_FREED[1].fetch_add(d.span.end - d.span.start, Ordering::Relaxed);
                false
            }
            _ => true,
        }
    });
    GRAVEYARD_LEN.store(g.len() as u64, Ordering::Release);
}

#[derive(Clone, Copy)]
struct Pipe {
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    set_layout: vk::DescriptorSetLayout,
    /// The set layout is a push descriptor layout (`push_layout`): commands
    /// push their descriptors instead of allocating, writing and binding a
    /// set.
    push: bool,
}

/// Whether a set layout with these bindings is made a push descriptor
/// layout (`tune::push_desc`, all its descriptors within the device's
/// maxPushDescriptors; image tables alone are 64), and its create flags.
fn push_layout(bindings: &[vk::DescriptorSetLayoutBinding], push_mode: bool) -> (bool, vk::DescriptorSetLayoutCreateFlags) {
    let n: u32 = bindings.iter().map(|b| b.descriptor_count).sum();
    if push_mode && n <= ctx().max_push {
        (true, vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR)
    } else {
        (false, vk::DescriptorSetLayoutCreateFlags::empty())
    }
}

/// Whether a pipeline of these stages gets a push descriptor layout: the
/// switch on and all their descriptors within maxPushDescriptors.
fn push_fits(stages: &[&Prepared]) -> bool {
    super::tune::push_desc() && stages.iter().flat_map(|p| descriptor_bindings(&p.tr.out)).map(|b| binding_count(b.kind)).sum::<u32>() <= ctx().max_push
}

/// Pipelines with push descriptor layouts are cached under keys of their
/// own (others are the same pipeline in either setting).
fn push_key(key: u64, push: bool) -> u64 {
    if push { key ^ 0x9E37_79B9_7F4A_7C15 } else { key }
}

/// Buffers every queue binds, created once.
struct Fixed {
    gds: vk::Buffer,
    /// Register dumps (`DS_DUMP`), host visible.
    debug: vk::Buffer,
    debug_ptr: *mut u8,
    /// Loop watchdog record (`rdna::translate::LOOP_LIMIT`), host visible
    /// (mapped at WATCHDOG_PTR).
    watchdog: vk::Buffer,
    /// Zero-filled buffer bound for empty or unmapped resources.
    dummy: vk::Buffer,
    /// Zero-filled and never written (`Spec::addresses`: reads outside a
    /// resource's range load from it; writes there are skipped): its address.
    zero_va: u64,
    /// Hardware LOD statistics measured by the sampling instructions
    /// (`BindingKind::LodStats`; `lodstats`).
    lodstats: vk::Buffer,
    /// Copies of them made by GetLodStats packets (lodstats::RING slots),
    /// host visible.
    lod_readback: vk::Buffer,
    lod_readback_ptr: *mut u8,
    import_align: u64,
}

unsafe impl Send for Fixed {}
unsafe impl Sync for Fixed {}

static FIXED: std::sync::OnceLock<Fixed> = std::sync::OnceLock::new();

/// Imported guest-memory windows. Every buffer binding of every queue looks
/// one up, imports and remap invalidations are rare: a read-mostly lock, and
/// `CACHE_GEN` changes whenever an import leaves it so that threads can keep
/// their recent hits (`SPAN_CACHE`; a new import cannot make a hit stale,
/// and until 2026-10-28 every import, remapping anywhere and image burial
/// cleared them, ~1,800 times in a stream-in window). Imports are made
/// outside its lock (`import_span`). (One mutex used to guard these, the pipeline
/// maps and the fixed buffers, held across whole binding loops and pipeline
/// compiles: the queue threads spent as much time waiting for it as working.)
static SPANS: parking_lot::RwLock<Vec<Span>> = parking_lot::RwLock::new(Vec::new());
/// Advances when an import leaves the list or an image is buried: batches
/// begun before may still name them (`reap_spans`).
static SPAN_GEN: AtomicU64 = AtomicU64::new(1);
/// Advances when an import leaves the list (`SPAN_CACHE`).
static CACHE_GEN: AtomicU64 = AtomicU64::new(1);
const SPAN_CACHE_LEN: usize = 16;

thread_local! {
    /// Recently used imports of this thread (start, end, buffer), valid for
    /// one `CACHE_GEN`, most recent first.
    static SPAN_CACHE: std::cell::RefCell<(u64, Vec<(u64, u64, vk::Buffer)>)> = const { std::cell::RefCell::new((0, Vec::new())) };
}

/// Compute pipelines by translation key; graphics pipelines by state hash.
/// Pipelines are created outside these locks.
static PIPES: Mutex<Option<HashMap<u64, Pipe>>> = Mutex::new(None);
static GFX_PIPES: Mutex<Option<HashMap<u64, Pipe>>> = Mutex::new(None);

/// The LOD statistics copied into readback slot `slot` (call once the batch
/// with its `Exec::lod_stats_copy` completed).
pub fn lod_stats_words(slot: usize) -> Vec<u32> {
    let n = (super::lodstats::SLOT_BYTES / 4) as usize;
    let p = unsafe { fixed().lod_readback_ptr.add(slot * n * 4) } as *const u32;
    unsafe { std::slice::from_raw_parts(p, n).to_vec() }
}

/// Contents of the register-dump buffer (`DS_DUMP`); `clear` zeroes it
/// (call with no GPU work using it in flight).
pub fn debug_dump(clear: bool) -> Vec<u32> {
    let n = rdna::translate::DUMP_DWORDS as usize;
    let p = fixed().debug_ptr as *mut u32;
    let v = unsafe { std::slice::from_raw_parts(p, n).to_vec() };
    if clear {
        unsafe { std::ptr::write_bytes(p, 0, n) };
    }
    v
}

/// Import windows are aligned to this size within a guest mapping.
pub(super) const WINDOW: u64 = 64 << 20;
/// Bindings without a bounds check extend at most this far.
const UNBOUNDED_MAX: u64 = 256 << 20;
const DUMMY_BYTES: u64 = 64 << 10;
/// The zero page of address-mode translations (`Fixed::zero_va`): covers
/// the widest access (16 bytes).
const ZERO_PAGE_BYTES: u64 = 4 << 10;

/// The bindings of a translation that are descriptors (address-mode buffers,
/// `BindingKind::Address`, are not).
pub(super) fn descriptor_bindings(out: &rdna::translate::Output) -> impl Iterator<Item = &rdna::translate::Binding> {
    out.bindings.iter().filter(|b| !matches!(b.kind, BindingKind::Address(_)))
}

fn fixed() -> &'static Fixed {
    FIXED.get_or_init(|| {
        let c = ctx();
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC;
        let (gds, _) = c.device_buffer(rdna::translate::GDS_BYTES as u64, usage).expect("GDS buffer");
        let (dummy, _) = c.device_buffer(DUMMY_BYTES, usage).expect("dummy buffer");
        let (zero, _) = c.device_buffer(ZERO_PAGE_BYTES, usage).expect("zero page");
        let (debug, _, debug_ptr) = c.host_buffer(rdna::translate::DUMP_DWORDS as u64 * 4, usage).expect("debug buffer");
        let (lodstats, _) = c.device_buffer(super::lodstats::SLOT_BYTES, usage).expect("LOD statistics buffer");
        let (lod_readback, _, lod_readback_ptr) = c.host_buffer(super::lodstats::SLOT_BYTES * super::lodstats::RING as u64, usage).expect("LOD statistics readback");
        let (watchdog, _, watchdog_ptr) = c.host_buffer(rdna::translate::WATCHDOG_DWORDS as u64 * 4, usage).expect("watchdog buffer");
        unsafe { std::ptr::write_bytes(watchdog_ptr, 0, rdna::translate::WATCHDOG_DWORDS as usize * 4) };
        WATCHDOG_PTR.store(watchdog_ptr as *mut u32, Ordering::SeqCst);
        let mut host = vk::PhysicalDeviceExternalMemoryHostPropertiesEXT::default();
        let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut host);
        unsafe { c.instance.get_physical_device_properties2(c.pdev, &mut p2) };
        // Zero-initialize the GDS and the dummy buffer once.
        unsafe {
            let pool = c.device.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(c.queue_family), None).expect("pool");
            let cmd = c.device.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(pool).command_buffer_count(1)).expect("cmd")[0];
            let fence = c.device.create_fence(&vk::FenceCreateInfo::default(), None).expect("fence");
            c.device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).ok();
            c.device.cmd_fill_buffer(cmd, gds, 0, vk::WHOLE_SIZE, 0);
            c.device.cmd_fill_buffer(cmd, dummy, 0, vk::WHOLE_SIZE, 0);
            c.device.cmd_fill_buffer(cmd, zero, 0, vk::WHOLE_SIZE, 0);
            // Every bank: no mip sampled (15), count 0.
            c.device.cmd_fill_buffer(cmd, lodstats, 0, super::lodstats::MIN_BYTES, 15);
            c.device.cmd_fill_buffer(cmd, lodstats, super::lodstats::MIN_BYTES, super::lodstats::SLOT_BYTES - super::lodstats::MIN_BYTES, 0);
            c.device.end_command_buffer(cmd).ok();
            if c.submit(cmd, &[], &[], fence).is_ok() {
                c.device.wait_for_fences(&[fence], true, u64::MAX).ok();
            }
            c.device.destroy_fence(fence, None);
            c.device.destroy_command_pool(pool, None);
        }
        Fixed { gds, debug, debug_ptr, watchdog, dummy, zero_va: c.buffer_address(zero), lodstats, lod_readback, lod_readback_ptr, import_align: host.min_imported_host_pointer_alignment.max(4096) }
    })
}

/// Front end 2 (`tune fe2_precheck`): the placement checks of a stage
/// prepared ahead, made by the thread that prepared it (`gpu::ahead`): its
/// sized buffer bindings in the mirror's span checked in the epoch now
/// (`fe2::precheck`: takes and clean stamps; uploads stay the command
/// processor's). Bindings resolved by the shader and unsized pointers are
/// left to the command processor.
pub(super) fn fe2_precheck_stage(found: &super::shaders::Found) {
    let gpu_res = found.tr.spec.gpu_res;
    for b in &found.tr.out.bindings {
        let i = match b.kind {
            BindingKind::Address(i) if i < 64 && gpu_res & (1u64 << i) != 0 => continue,
            BindingKind::Buffer(i) | BindingKind::Address(i) => i,
            _ => continue,
        };
        let (base, size) = match found.resources.get(i) {
            Some(Resolved::Buffer(v)) => (v.base, v.byte_size()),
            Some(Resolved::Global(a, n)) => (*a, *n),
            _ => continue,
        };
        let Some(size) = size.filter(|&n| n > 0) else { continue };
        if super::fe2::in_span(base, size) {
            super::fe2::precheck(base, size, &fe2_armable);
        }
    }
}

/// Guest mapping containing `addr`: [start, end) of its VMA.
fn mapping(addr: u64) -> Option<(u64, u64)> {
    if super::tune::vma_cache() {
        return crate::mem::vma_bounds(addr);
    }
    let st = crate::mem::MEM.lock();
    let v = st.find(addr)?;
    if v.kind == crate::mem::VmaKind::Reserved {
        return None;
    }
    Some((v.start, v.end()))
}

/// Adjacent guest mappings around `addr` that one import may cover,
/// clipped to [lo, hi): the title maps buffers in pieces (e.g.
/// sceKernelBatchMap in 64 KiB steps), and a buffer may span several.
/// Returns [start, end) and the number of mappings.
fn mapping_run(addr: u64, lo: u64, hi: u64) -> Option<(u64, u64, usize)> {
    use crate::mem::VmaKind;
    // (The mixed range's view windows and private memory apart.)
    let (lo, hi) = crate::mem::same_backing(addr, lo, hi);
    let st = crate::mem::MEM.lock();
    let v = st.find(addr)?;
    if v.kind == VmaKind::Reserved {
        return None;
    }
    let usable = |p: &crate::mem::Vma| p.kind != VmaKind::Reserved && p.prot != 0;
    let (mut start, mut end, mut n) = (v.start, v.end(), 1);
    while start > lo {
        match st.vmas.range(..start).next_back().map(|(_, p)| p).filter(|p| p.end() == start && usable(p)) {
            Some(p) => {
                start = p.start;
                n += 1;
            }
            None => break,
        }
    }
    while end < hi {
        match st.vmas.get(&end).filter(|p| usable(p)) {
            Some(p) => {
                end = p.end();
                n += 1;
            }
            None => break,
        }
    }
    Some((start.max(lo), end.min(hi), n))
}

/// Why guest [base, base+size) cannot be imported (diagnostics).
fn unmapped_reason(base: u64, size: u64) -> String {
    let st = crate::mem::MEM.lock();
    let mut p = base;
    while p < base + size {
        match st.find(p) {
            None => return format!("nothing mapped at {p:#x}"),
            Some(v) if v.kind == crate::mem::VmaKind::Reserved => return format!("{p:#x} is only reserved ({:#x}+{:#x})", v.start, v.size),
            Some(v) if v.prot == 0 && p != base => return format!("{p:#x} is not accessible ({:?} {:#x}+{:#x})", v.kind, v.start, v.size),
            Some(v) => p = v.end(),
        }
    }
    "mapped: the import failed".into()
}

/// Mapping changes of guest memory inside a live import (diagnostics of
/// device losses at level loads): counts by kind (`track::REMAP_*`), and the
/// latest ones ((when, kind, base, len, import start, import end); no more
/// are kept once the device is lost, so they show what preceded it).
static REMAPS_OVER_IMPORTS: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
static REMAP_RING: Mutex<Option<std::collections::VecDeque<(std::time::Instant, u8, u64, u64, u64, u64)>>> = Mutex::new(None);

/// A commit, decommit or protection change of guest [base, base+len)
/// (`track::remapped`, inside the memory manager's lock: the import list is
/// only tried, never waited for).
pub(super) fn note_remap_over_imports(base: u64, len: u64, op: u8) {
    if DEVICE_LOST.load(Ordering::Relaxed) {
        return;
    }
    let Some(g) = SPANS.try_read() else { return };
    let end = base.saturating_add(len);
    let hit = g.iter().find(|s| s.valid && s.start < end && base < s.end).map(|s| (s.start, s.end));
    drop(g);
    let Some((s, e)) = hit else { return };
    REMAPS_OVER_IMPORTS[(op as usize).min(2)].fetch_add(1, Ordering::Relaxed);
    let mut r = REMAP_RING.lock();
    let r = r.get_or_insert_with(Default::default);
    if r.len() >= 64 {
        r.pop_front();
    }
    r.push_back((std::time::Instant::now(), op, base, len, s, e));
}

/// One status line: mapping changes inside live imports, and the latest.
fn remaps_over_imports_line() -> String {
    let n: Vec<u64> = REMAPS_OVER_IMPORTS.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    let last: Vec<String> = REMAP_RING
        .lock()
        .as_ref()
        .map(|r| {
            r.iter()
                .rev()
                .take(8)
                .map(|x| format!("{} {:#x}+{:#x} in {:#x}..{:#x} {:.1} s ago", ["commit", "decommit", "protection"][(x.1 as usize).min(2)], x.2, x.3, x.4, x.5, x.0.elapsed().as_secs_f64()))
                .collect()
        })
        .unwrap_or_default();
    format!(
        "    mapping changes inside live imports: commits {}, decommits {}, protection changes {}; latest (before a device loss, if any): {}\n",
        n[0],
        n[1],
        n[2],
        if last.is_empty() { "none".to_string() } else { last.join("; ") }
    )
}

/// Diagnosis: the valid import window over guest address `a` (start, end,
/// device address), without importing.
pub(super) fn window_of(a: u64) -> Option<(u64, u64, u64)> {
    window_buffer_of(a).map(|(s, e, va, _, _)| (s, e, va))
}

/// Whether `buf` is still the valid import window starting at guest
/// `start` and at least `bytes` long (not dropped, its handle not reused).
pub(super) fn window_valid(buf: vk::Buffer, start: u64, bytes: u64) -> bool {
    spans_read().iter().any(|s| s.valid && s.buffer == buf && s.start == start && s.end - s.start >= bytes)
}

/// The valid import window over guest address `a`: (start, end, device
/// address, buffer, the flip it was imported at), without importing.
pub(super) fn window_buffer_of(a: u64) -> Option<(u64, u64, u64, vk::Buffer, u64)> {
    spans_read().iter().find(|s| s.valid && s.start <= a && a < s.end).map(|s| (s.start, s.end, ctx().buffer_address(s.buffer), s.buffer, s.born))
}

/// Maps every import window into the guest page table (when it is made).
pub(super) fn map_imports_into_table() {
    let c = ctx();
    for s in spans_read().iter().filter(|s| s.valid) {
        super::pagetable::map(s.start, s.end - s.start, c.buffer_address(s.buffer));
    }
}

/// Drops imports whose guest pages were committed or decommitted since.
fn revalidate_spans() {
    let changed = crate::mem::take_repaged();
    if changed.is_empty() {
        return;
    }
    let ranges: Vec<(u64, u64)> = changed.iter().map(|&(a, b)| (a, b - a)).collect();
    super::mirror::invalidate(&ranges);
    for &(a, n) in &ranges {
        super::fe2::invalidate(a, n);
    }
    // Imports in progress over the changed pages are dropped when they end
    // (they may pin pages no longer mapped there): before the list is
    // looked at, so that one listed meanwhile is met below.
    for m in IMPORTING.lock().iter().filter(|m| changed.iter().any(|&(a, b)| a < m.end && m.start < b)) {
        m.stale.store(true, Ordering::SeqCst);
    }
    // The list is written only when an import in it is met: most mapping
    // changes are outside imports, and a writer waiting made every lookup
    // wait behind it.
    if !spans_read().iter().any(|s| s.valid && changed.iter().any(|&(a, b)| a < s.end && s.start < b)) {
        return;
    }
    let mut g = SPANS.write();
    let mut dead = Vec::new();
    let mut i = 0;
    while i < g.len() {
        let s = &g[i];
        if s.valid && changed.iter().any(|&(a, b)| a < s.end && s.start < b) {
            if IMPORTS_STALE.fetch_add(1, Ordering::Relaxed) < 20 {
                crate::info!("gpu: guest range {:#x}..{:#x} imported for the GPU was remapped (imported again on its next use)", s.start, s.end);
            }
            // Freed once no batch can use it (recorded work may).
            super::pagetable::unmap(s.start, s.end - s.start);
            dead.push(g.swap_remove(i));
        } else {
            i += 1;
        }
    }
    // Windows still valid over the pages just cleared map them again (two
    // windows may cover the same pages).
    for d in &dead {
        for v in g.iter().filter(|v| v.valid && v.start < d.end && d.start < v.end) {
            super::pagetable::map(v.start, v.end - v.start, ctx().buffer_address(v.buffer));
        }
    }
    if !dead.is_empty() {
        DROP_GEN.fetch_add(1, Ordering::AcqRel);
        CACHE_GEN.fetch_add(1, Ordering::AcqRel);
    }
    let generation = SPAN_GEN.fetch_add(1, Ordering::AcqRel) + 1;
    drop(g);
    let reimport: Vec<(u64, u64)> = if super::pagetable::made() { dead.iter().map(|d| (d.start, d.end)).collect() } else { Vec::new() };
    if !dead.is_empty() {
        let mut gy = GRAVEYARD.lock();
        gy.extend(dead.into_iter().map(|mut span| {
            span.valid = false;
            super::vk::note_gpu_dropped(span.buffer);
            Dead { span, generation, seq: None }
        }));
        GRAVEYARD_LEN.store(gy.len() as u64, Ordering::Release);
    }
    // GPU-resolved chains read through the guest page table: a dropped
    // window's pages still mapped are imported again at once, so that the
    // table goes from the old window to a new one over the same pages before
    // the next batch, without clearing them. Batches recorded before the
    // drop and submitted after it read them through the table (commands bind
    // windows themselves, and keep the dropped one until they complete).
    for (s, e) in reimport {
        let mut a = s;
        while a < e {
            // The next usable mapping from `a` on.
            let next = {
                let st = crate::mem::MEM.lock();
                let usable = |p: &crate::mem::Vma| p.kind != crate::mem::VmaKind::Reserved && p.prot != 0;
                match st.find(a) {
                    Some(v) if usable(v) => Some(a),
                    _ => st.vmas.range(a..e).map(|(_, p)| p).find(|p| usable(p)).map(|p| p.start),
                }
            };
            let Some(b) = next else { break };
            REIMPORTS.fetch_add(1, Ordering::Relaxed);
            a = match import_span(b, 4) {
                Some((_, _, end)) if end > b => end,
                _ => (b & !(WINDOW - 1)) + WINDOW,
            };
        }
    }
}

/// Import windows made again right after their window was dropped
/// (`revalidate_spans`, with the guest page table).
pub static REIMPORTS: AtomicU64 = AtomicU64::new(0);

/// In-place bindings with residency on (they import their pages): by
/// [written, unsized] and size bucket (<64K, <1M, <16M, more): [bindings,
/// bytes]; and the shaders behind the most bytes.
static IN_PLACE: Mutex<Option<(HashMap<(bool, bool, u8), [u64; 2]>, HashMap<u64, u64>)>> = Mutex::new(None);

fn note_in_place(written: bool, pointer: bool, size: u64, shader: u64) {
    let b = match size {
        0..65536 => 0,
        65536..1048576 => 1,
        1048576..16777216 => 2,
        _ => 3,
    };
    let mut g = IN_PLACE.lock();
    let (m, s) = g.get_or_insert_with(Default::default);
    let e = m.entry((written, pointer, b)).or_default();
    e[0] += 1;
    e[1] += size;
    *s.entry(shader).or_default() += size;
}

/// Status part: in-place bindings with residency on.
pub(super) fn in_place_line() -> String {
    let g = IN_PLACE.lock();
    let Some((m, s)) = g.as_ref() else { return String::new() };
    let mut v: Vec<(&(bool, bool, u8), &[u64; 2])> = m.iter().collect();
    v.sort_by(|a, b| b.1[1].cmp(&a.1[1]));
    let parts: Vec<String> = v
        .iter()
        .map(|((w, u, b), x)| format!("{}{} {} {} ({} GiB)", if *w { "written" } else { "read" }, if *u { " unsized" } else { "" }, ["<64K", "<1M", "<16M", ">=16M"][*b as usize], x[0], x[1] >> 30))
        .collect();
    let mut sh: Vec<(&u64, &u64)> = s.iter().collect();
    sh.sort_by(|a, b| b.1.cmp(a.1));
    let top: Vec<String> = sh.iter().take(6).map(|(a, n)| format!("{a:#x} {} GiB", *n >> 30)).collect();
    format!("    in-place bindings with residency on: {}; by shader: {}\n", parts.join(", "), top.join(", "))
}

/// New imports by the code that asked for them (residency on: their pages
/// turn host class), as (calls, bytes).
static IMPORT_SITES: Mutex<Option<HashMap<(&'static str, u32), (u64, u64)>>> = Mutex::new(None);

/// Status part: the sites that made imports (residency on).
pub(super) fn import_sites_line() -> String {
    let g = IMPORT_SITES.lock();
    let Some(m) = g.as_ref() else { return String::new() };
    let mut v: Vec<(&(&str, u32), &(u64, u64))> = m.iter().collect();
    v.sort_by(|a, b| b.1 .1.cmp(&a.1 .1));
    let list: Vec<String> = v.iter().take(10).map(|((f, l), (n, b))| format!("{}:{l} {n} ({} MiB)", f.rsplit(['/', '\\']).next().unwrap_or(f), b >> 20)).collect();
    let mut s = format!("    imports made with residency on, by site: {}\n", list.join(", "));
    if let Some(m) = MM_IMPORT_REFUSED.lock().as_ref() {
        let mut v: Vec<(&(&str, u32), &u64)> = m.iter().collect();
        v.sort_by(|a, b| b.1.cmp(a.1));
        let list: Vec<String> = v.iter().take(10).map(|((f, l), n)| format!("{}:{l} {n}", f.rsplit(['/', '\\']).next().unwrap_or(f))).collect();
        s += &format!("    imports refused in mirror mode, by site: {}\n", list.join(", "));
    }
    s
}

/// Bindings placed on resident GPU-written data (`gpu::rgm`; counted as
/// homes in the placement statistics), and indirect dispatches whose
/// arguments were read there.
static RGM_PLACED: AtomicU64 = AtomicU64::new(0);
pub(super) static RGM_ARGS: AtomicU64 = AtomicU64::new(0);

/// Mirror mode's placements by [written][unbounded pointer][pages: <16,
/// <256, <4096, more]: [placements, pages].
static MM_PLACE: [[AtomicU64; 2]; 16] = [const { [const { AtomicU64::new(0) }; 2] }; 16];

/// Status part: mirror mode's placements by kind and size.
pub(super) fn mm_place_line() -> String {
    if !super::resident::mirror_mode() {
        return String::new();
    }
    let mut parts = Vec::new();
    for (k, e) in MM_PLACE.iter().enumerate() {
        let (n, pages) = (e[0].load(Ordering::Relaxed), e[1].load(Ordering::Relaxed));
        if n == 0 {
            continue;
        }
        let what = format!("{}{} {}", if k >= 8 { "written" } else { "read" }, if k & 4 != 0 { " unbounded" } else { "" }, ["<64K", "<1M", "<16M", ">=16M"][k & 3]);
        parts.push(format!("{what} {n} ({pages} pages, {} GiB)", (pages * 4096) >> 30));
    }
    format!("    mirror placements: {}
", parts.join(", "))
}

/// Mirror mode: imports of the mirror's span refused (`span`), by the code
/// that asked for them.
static MM_IMPORT_REFUSED: Mutex<Option<HashMap<(&'static str, u32), u64>>> = Mutex::new(None);

/// Imported buffer covering guest [base, base+size): (buffer, window start, window end).
#[track_caller]
fn span(base: u64, size: u64) -> Option<(vk::Buffer, u64, u64)> {
    revalidate_spans();
    let generation = CACHE_GEN.load(Ordering::Acquire);
    let end = base.saturating_add(size);
    let hit = SPAN_CACHE.with(|c| {
        let mut c = c.borrow_mut();
        if c.0 != generation {
            c.0 = generation;
            c.1.clear();
            return None;
        }
        let i = c.1.iter().position(|&(s, e, _)| s <= base && end <= e)?;
        let x = c.1.remove(i);
        c.1.insert(0, x);
        Some((x.2, x.0, x.1))
    });
    if hit.is_some() {
        return hit;
    }
    let found = spans_read().iter().find(|s| s.valid && s.start <= base && end <= s.end).map(|s| (s.buffer, s.start, s.end));
    let site = std::panic::Location::caller();
    // Mirror mode: memory of the mirror's span is never imported (its pages
    // may be GPU-owned, no-access, which the driver cannot pin; an import
    // would make them host class for good): the caller's other path runs.
    if found.is_none() && super::resident::mirror_mode() && base < super::resident::MIRROR_END && end > super::resident::MIRROR_BASE && !super::resident::all_host(base, size) {
        let mut g = MM_IMPORT_REFUSED.lock();
        *g.get_or_insert_with(HashMap::new).entry((site.file(), site.line())).or_default() += 1;
        return None;
    }
    let r = found.or_else(|| {
        let r = import_span(base, size);
        if let (Some((_, st, e)), true) = (r, super::resident::enabled()) {
            let mut g = IMPORT_SITES.lock();
            let x = g.get_or_insert_with(HashMap::new).entry((site.file(), site.line())).or_default();
            x.0 += 1;
            x.1 += e - st;
        }
        r
    });
    if let Some((b, st, e)) = r {
        SPAN_CACHE.with(|c| {
            let mut c = c.borrow_mut();
            if CACHE_GEN.load(Ordering::Acquire) == generation {
                if c.0 != generation {
                    c.0 = generation;
                    c.1.clear();
                }
                c.1.insert(0, (st, e, b));
                c.1.truncate(SPAN_CACHE_LEN);
            }
        });
    }
    r
}

/// Front end 2 (`fe2::Armable`): calls `arm` with the parts of guest [a, b)
/// that live imports cover and over which every import, live or dropped
/// and not freed yet, is settled (`fe2::SETTLE` flips past its first
/// completed use; protecting a new import's pages lost the device:
/// FINDINGS.md 2026-10-26); false: the lists were busy. The list is not
/// held while arming (an arming's host calls held it, and every lookup
/// waited behind an import waiting for it): `fe2::arm_part` asks
/// `fe2_settled_over` again under the arm lock, and an import holds its
/// pages unarmed from before its driver call until it is listed
/// (`fe2::hold`).
fn fe2_armable(a: u64, b: u64, arm: &mut dyn FnMut(u64, u64)) -> bool {
    let Some(parts) = settled_parts(a, b) else { return false };
    for (p, q) in parts {
        arm(p, q);
    }
    true
}

/// The parts of guest [a, b) that live imports cover, less those over which
/// an import, live or dropped and not freed yet, is not settled (`fe2_armable`);
/// None: the lists were busy.
fn settled_parts(a: u64, b: u64) -> Option<Vec<(u64, u64)>> {
    let now = super::FRAMES.load(Ordering::Relaxed);
    let settled = |born: u64| born + super::fe2::SETTLE <= now;
    let done = done_seq();
    // A live import counts from the completion of the first batch that
    // uploaded from it (`Span::first_use`), first seen here.
    let used = |s: &Span| {
        let u = s.used_at.load(Ordering::Acquire);
        if u != u64::MAX {
            return u + super::fe2::SETTLE <= now;
        }
        if s.first_use.load(Ordering::Acquire) <= done {
            s.used_at.store(now, Ordering::Release);
        }
        false
    };
    let g = SPANS.try_read()?;
    let mut young: Vec<(u64, u64)> = g.iter().filter(|s| s.start < b && a < s.end && !(settled(s.born) && used(s))).map(|s| (s.start, s.end)).collect();
    {
        let gy = GRAVEYARD.try_lock()?;
        young.extend(gy.iter().filter(|d| d.span.start < b && a < d.span.end && !settled(d.span.born)).map(|d| (d.span.start, d.span.end)));
    }
    young.sort_unstable();
    let mut live: Vec<(u64, u64)> = g.iter().filter(|s| s.valid && s.start < b && a < s.end).map(|s| (s.start.max(a), s.end.min(b))).collect();
    drop(g);
    live.sort_unstable();
    let mut merged: Vec<(u64, u64)> = Vec::with_capacity(live.len());
    for (x, y) in live {
        match merged.last_mut() {
            Some(l) if x <= l.1 => l.1 = l.1.max(y),
            _ => merged.push((x, y)),
        }
    }
    // The covered parts less the young imports.
    let mut parts = Vec::with_capacity(merged.len());
    for (x, y) in merged {
        let mut p = x;
        for &(s, e) in young.iter().filter(|&&(s, e)| s < y && x < e) {
            if s > p {
                parts.push((p, s));
            }
            p = p.max(e);
            if p >= y {
                break;
            }
        }
        if p < y {
            parts.push((p, y));
        }
    }
    Some(parts)
}

/// Front end 2 (`fe2::arm_part`, under the arm lock): whether guest [a, b)
/// is still wholly inside settled live imports (an import listed since the
/// caller looked is young); None: the lists were busy.
pub(super) fn fe2_settled_over(a: u64, b: u64) -> Option<bool> {
    let parts = settled_parts(a, b)?;
    Some(parts.len() == 1 && parts[0] == (a, b))
}

/// Front end 2: batch `seq` (queued) uploads from these imports.
fn note_span_uses(used: &[(u64, vk::Buffer)], seq: u64) {
    let g = SPANS.read();
    for &(start, buf) in used {
        if let Some(s) = g.iter().find(|s| s.start == start && s.buffer == buf) {
            s.first_use.fetch_min(seq, Ordering::AcqRel);
        }
    }
}

/// Front end 2: live imports, those an upload has used, and those settled
/// for arming (used SETTLE flips ago).
pub(super) fn fe2_span_census() -> (usize, usize, usize) {
    let now = super::FRAMES.load(Ordering::Relaxed);
    let Some(g) = SPANS.try_read() else { return (0, 0, 0) };
    let live: Vec<&Span> = g.iter().filter(|s| s.valid).collect();
    let used = live.iter().filter(|s| s.first_use.load(Ordering::Relaxed) != u64::MAX).count();
    let settled = live.iter().filter(|s| s.used_at.load(Ordering::Relaxed).checked_add(super::fe2::SETTLE).is_some_and(|t| t <= now)).count();
    (live.len(), used, settled)
}

/// Mirror mode: whether the pages to be copied into the mirror (ascending
/// lists) are readable guest memory, one query per run of pages.
fn staged_readable(claim: &[u64], copy: &[u64]) -> bool {
    let page = super::resident::PAGE;
    for list in [claim, copy] {
        let mut i = 0;
        while i < list.len() {
            let mut j = i + 1;
            while j < list.len() && list[j] == list[j - 1] + page {
                j += 1;
            }
            if !crate::mem::os_readable(list[i], list[j - 1] + page - list[i]) {
                return false;
            }
            i = j;
        }
    }
    true
}

/// Guest [a, a+n) as pieces of imported buffers: (buffer, offset in it,
/// guest address, bytes); unmapped parts are left out.
fn guest_pieces(a: u64, n: u64) -> Vec<(vk::Buffer, u64, u64, u64)> {
    if let Some((buf, start, _)) = span(a, n) {
        return vec![(buf, a - start, a, n)];
    }
    let mut v = Vec::new();
    let end = a + n;
    let mut c = a;
    while c < end {
        let e = ((c & !(WINDOW - 1)) + WINDOW).min(end);
        if let Some((buf, start, _)) = span(c, e - c) {
            v.push((buf, c - start, c, e - c));
        }
        c = e;
    }
    v
}

/// Imports in progress (`import_span`, outside the import list's lock):
/// the windows they may take (registered before the mapping is looked at),
/// and whether a mapping change met one meanwhile (`revalidate_spans`: it
/// is dropped when it ends, and made again).
struct Importing {
    start: u64,
    end: u64,
    stale: std::sync::atomic::AtomicBool,
}

static IMPORTING: Mutex<Vec<std::sync::Arc<Importing>>> = Mutex::new(Vec::new());
/// An import in progress ended (`IMPORTING`).
static IMPORTING_DONE: parking_lot::Condvar = parking_lot::Condvar::new();
/// Imports dropped because a mapping change met them in progress.
static IMPORTS_RACED: AtomicU64 = AtomicU64::new(0);

/// Its import ended (dropped at every return of `import_once`, after the
/// import was listed): unregistered, and those waiting for it woken.
struct Registered(std::sync::Arc<Importing>);

impl Drop for Registered {
    fn drop(&mut self) {
        IMPORTING.lock().retain(|m| !std::sync::Arc::ptr_eq(m, &self.0));
        IMPORTING_DONE.notify_all();
    }
}

/// Imports the guest window around [base, base+size) (another queue may
/// have done so meanwhile). The driver's import (it pins the pages: up to
/// tens of milliseconds) runs outside the import list's lock: held for
/// writing meanwhile, every thread's lookups waited (282 waits, 959 ms,
/// against 296 ms of imports in a stream-in window, 2026-10-28 night); an
/// import over the same window waits for the one in progress instead.
fn import_span(base: u64, size: u64) -> Option<(vk::Buffer, u64, u64)> {
    // (A mapping change met the import: made again, three times at most.)
    for _ in 0..3 {
        if let Some(r) = import_once(base, size) {
            return r;
        }
    }
    None
}

/// One `import_span`: None when a mapping change met the import.
fn import_once(base: u64, size: u64) -> Option<Option<(vk::Buffer, u64, u64)>> {
    // Residency: smaller windows (their pages become host class).
    // Mirror mode: exact pages (imports are only for what the mirror cannot
    // hold: CP writes, host-class pages), so they cover no mirror page.
    let window = if super::resident::mirror_mode() { 4096 } else if super::resident::enabled() { RES_WINDOW } else { WINDOW };
    let (lo, hi) = (base & !(window - 1), (base + size).next_multiple_of(window));
    let me = loop {
        if let Some(s) = spans_read().iter().find(|s| s.valid && s.start <= base && base + size <= s.end) {
            return Some(Some((s.buffer, s.start, s.end)));
        }
        // Registered over the windows it may take, unless an import there is
        // in progress: then its end is waited for and the list looked at
        // again.
        let mut g = IMPORTING.lock();
        if g.iter().any(|m| m.start < hi && lo < m.end) {
            let t = std::time::Instant::now();
            IMPORTING_DONE.wait_for(&mut g, std::time::Duration::from_millis(50));
            drop(g);
            super::hitch::note_since(super::hitch::Kind::Mirror, t, || format!("waiting for an import in progress over {lo:#x}..{hi:#x}"));
            continue;
        }
        let me = std::sync::Arc::new(Importing { start: lo, end: hi, stale: std::sync::atomic::AtomicBool::new(false) });
        g.push(me.clone());
        break Registered(me);
    };
    let Some((vs, ve, pieces)) = mapping_run(base, lo, hi) else { return Some(None) };
    if base + size > ve {
        return Some(None);
    }
    // Imported pages must never be protected: host class from now on
    // (section-backed memory imports alias views instead).
    if super::resident::enabled() && !crate::sections::on() {
        super::resident::mark_host(vs, ve - vs);
    }
    {
        // Diagnostics: how the title mapped memory the GPU uses.
        let st = crate::mem::MEM.lock();
        if let Some(v) = st.find(base) {
            crate::debug!("gpu: importing for {base:#x}+{size:#x}: mapping {:#x}+{:#x} {:?} prot {:#x} '{}'", v.start, v.size, v.kind, v.prot, v.name);
        }
    }
    let a = fixed().import_align;
    let (mut start, mut end) = (vs & !(a - 1), ve.next_multiple_of(a).min(ve));
    if start < vs || end <= start {
        return Some(None);
    }
    // Section-backed memory: the import is of the alias views (contiguous in
    // one slot), the window narrowed to the binding when the run is not.
    let mut alias = 0u64;
    if crate::sections::on() || crate::mem::view_window(start) {
        let slot = crate::sections::alias_slot_for(start, end - start).or_else(|| {
            (start, end) = ((base & !(a - 1)).max(vs), (base + size).next_multiple_of(a).min(ve));
            crate::sections::alias_slot_for(start, end - start).or_else(|| if crate::sections::on() { crate::mem::bridge_aliases(start, end) } else { None })
        });
        let Some(slot) = slot else {
            ALIAS_MISSES.fetch_add(1, Ordering::Relaxed);
            return Some(None);
        };
        alias = crate::sections::alias_of(0, slot);
    }
    let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::INDIRECT_BUFFER;
    // The driver pins the pages for writing: write-tracked (read-only)
    // pages must be writable while importing (their regions turn dirty
    // and are re-armed on their next use).
    let _w = super::track::protected().then(|| super::track::host_write(start, end - start));
    // (A long one, its disarm included, is a hitch cause: `gpu::hitch`.)
    let tw = std::time::Instant::now();
    // Front end 2 arms none of its pages until it is listed (young then).
    let _hold = super::fe2::hold(start, end - start);
    // Pages armed through the host likewise (`hostcensus`, `precache`'s
    // write tracking by protection).
    crate::hostcensus::release(start, end - start);
    crate::hosted::disarm_if_armed(start, end - start);
    let t0 = std::time::Instant::now();
    let imported = ctx().import_host(start + alias, end - start, usage);
    super::hitch::note_since(super::hitch::Kind::Mirror, tw, || format!("an import of {start:#x}..{end:#x} ({} MiB; the driver's part {} ms)", (end - start) >> 20, t0.elapsed().as_millis()));
    match imported {
        Ok((buffer, mem, va)) => {
            let ns = t0.elapsed().as_nanos() as u64;
            let mut g = spans_write();
            if me.0.stale.load(Ordering::SeqCst) {
                // No batch saw it: freed now.
                drop(g);
                IMPORTS_RACED.fetch_add(1, Ordering::Relaxed);
                unsafe {
                    ctx().device.destroy_buffer(buffer, None);
                    ctx().device.free_memory(mem, None);
                }
                return None;
            }
            super::vk::note_gpu_range(buffer, va, end - start, || format!("guest import {start:#x}..{end:#x}"));
            let ms = ns / 1_000_000;
            if ms >= 50 {
                crate::info!("gpu: importing guest {start:#x}..{end:#x} ({} MiB) took {ms} ms", (end - start) >> 20);
            }
            IMPORT_WIN[0].fetch_add(1, Ordering::Relaxed);
            IMPORT_WIN[1].fetch_add(end - start, Ordering::Relaxed);
            IMPORT_WIN[2].fetch_add(ns, Ordering::Relaxed);
            IMPORT_WIN[3].fetch_max(ns, Ordering::Relaxed);
            g.push(Span { start, end, buffer, mem, valid: true, born: super::FRAMES.load(Ordering::Relaxed), first_use: AtomicU64::new(u64::MAX), used_at: AtomicU64::new(u64::MAX) });
            super::pagetable::map(start, end - start, va);
            drop(g);
            IMPORTS.fetch_add(1, Ordering::Relaxed);
            IMPORT_BYTES.fetch_add(end - start, Ordering::Relaxed);
            if pieces > 1 {
                IMPORTS_MULTI.fetch_add(1, Ordering::Relaxed);
            }
            Some(Some((buffer, start, end)))
        }
        Err(e) => {
            static REPORTED: AtomicU64 = AtomicU64::new(0);
            let k = REPORTED.fetch_add(1, Ordering::Relaxed);
            let heaps = if k < 4 { format!(" (memory: {}; {} imports dropped after remapping not freed yet)", super::vk::ctx_heaps(), GRAVEYARD_LEN.load(Ordering::Relaxed)) } else { String::new() };
            // What the operating system holds there (the first failures of
            // each window: uncommitted or protected pages fail an import).
            static SEEN: Mutex<Option<HashSet<u64>>> = Mutex::new(None);
            if SEEN.lock().get_or_insert_with(HashSet::new).insert(start) && k < 64 {
                crate::warn!("gpu: import of {start:#x}..{end:#x} failed over: {}", crate::mem::os_regions(start, end - start, 12));
            }
            crate::warn!("gpu: importing guest {start:#x}..{end:#x} failed: {e}{heaps}");
            Some(None)
        }
    }
}

/// A compute pipeline being created (`compute_pipe`): its result once done.
struct PipeFlight {
    done: Mutex<Option<Result<Pipe, String>>>,
    cv: parking_lot::Condvar,
    /// The creating thread's handle when it runs at the lowest priority
    /// (`gpu::precompile`): a thread that waits for the pipeline raises it
    /// (it would otherwise wait behind every normal thread).
    owner: Option<isize>,
}

impl PipeFlight {
    fn new() -> PipeFlight {
        PipeFlight { done: Mutex::new(None), cv: parking_lot::Condvar::new(), owner: super::precompile::boostable_thread() }
    }

    /// A thread is about to wait for this pipeline.
    fn waited(&self) {
        if let Some(h) = self.owner {
            super::precompile::boost(h);
        }
    }
}

/// Compute pipelines being created, by key: a thread that needs one waits
/// for it instead of compiling it again (compiles ahead of use,
/// `gpu::compile`, and several queues meeting a new program).
static PIPE_FLIGHTS: Mutex<Option<HashMap<u64, std::sync::Arc<PipeFlight>>>> = Mutex::new(None);
/// Waits for a pipeline in flight: count, nanoseconds; compiles avoided.
static PIPE_FLIGHT_WAITS: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// (waits for a pipeline in flight, their nanoseconds, compiles avoided).
pub fn inflight_stats() -> (u64, u64, u64) {
    (PIPE_FLIGHT_WAITS[0].load(Ordering::Relaxed), PIPE_FLIGHT_WAITS[1].load(Ordering::Relaxed), PIPE_FLIGHT_WAITS[2].load(Ordering::Relaxed))
}

/// The compute pipeline of a prepared shader if it exists.
fn cached_compute_pipe(p: &Prepared) -> Option<Pipe> {
    let key = push_key(p.tr.key, push_fits(&[p]));
    PIPES.lock().as_ref().and_then(|m| m.get(&key)).copied()
}

/// Background compiles of particle programs' pipelines (`tune
/// async_particles`): jobs (translation, program address, pipeline key),
/// the keys queued or failed, workers started, and [dispatches left out,
/// pipelines made, ns of compiles, failures].
static ASYNC_JOBS: Mutex<std::collections::VecDeque<(std::sync::Arc<super::shaders::Translated>, u64, u64)>> = Mutex::new(std::collections::VecDeque::new());
static ASYNC_CV: parking_lot::Condvar = parking_lot::Condvar::new();
static ASYNC_QUEUED: Mutex<Option<HashSet<u64>>> = Mutex::new(None);
static ASYNC_FAILED: Mutex<Option<HashSet<u64>>> = Mutex::new(None);
static ASYNC_WORKERS: AtomicU64 = AtomicU64::new(0);
pub static ASYNC_STATS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];
/// The error of a dispatch left out while its pipeline compiles.
pub(super) const ASYNC_PENDING: &str = "a particle program's pipeline compiles in the background (its dispatches are left out until then)";

/// Whether the background compile of a prepared stage's pipeline is queued
/// (its dispatches are left out until it is made; each would scan the
/// commands after it again, `Cp::prefetch_compiles`).
fn async_queued(p: &Prepared) -> bool {
    let key = push_key(p.tr.key, push_fits(&[p]));
    ASYNC_QUEUED.lock().as_ref().is_some_and(|s| s.contains(&key))
}

/// Starts the background compile of a prepared stage's pipeline unless it
/// is made, in flight or queued: false when the caller should make it where
/// it is (the store holds it, or a background compile of it failed and
/// that reports the error).
fn compile_in_background(p: &Prepared) -> bool {
    let key = push_key(p.tr.key, push_fits(&[p]));
    if ASYNC_FAILED.lock().as_ref().is_some_and(|s| s.contains(&key)) {
        return false;
    }
    // A pipeline the store holds is made in milliseconds: where it is first
    // used (or waited for when a session-start precompile has it in flight),
    // so that only a machine's first compiles leave dispatches out.
    if super::pcache::stored('c', key) {
        return false;
    }
    if compute_pipe_known(p) || !ASYNC_QUEUED.lock().get_or_insert_with(HashSet::new).insert(key) {
        return true;
    }
    ASYNC_JOBS.lock().push_back((p.found.tr.0.clone(), p.addr, key));
    if compile_trace() {
        let from = super::particles::origin(p.code_hash).map(|d| format!(" (from {d})")).unwrap_or_default();
        crate::info!("background compile: cs {:#x} key {key:#x} queued{from}", p.addr);
    }
    // As many workers as compile-ahead has (the largest programs take 5-12
    // s each, and an area's effects come together), below normal priority.
    let want = super::tune::compile_ahead().clamp(2, 16);
    while ASYNC_WORKERS.load(Ordering::Relaxed) < want {
        let i = ASYNC_WORKERS.fetch_add(1, Ordering::Relaxed);
        if i >= want {
            break;
        }
        let name = format!("gpu-async{i}");
        let _ = crate::thread::spawn_host(&name.clone(), 8 << 20, Box::new(move || async_worker(name)));
    }
    ASYNC_CV.notify_one();
    true
}

fn async_worker(name: String) {
    crate::log::set_thread_name(&name);
    unsafe {
        windows_sys::Win32::System::Threading::SetThreadPriority(windows_sys::Win32::System::Threading::GetCurrentThread(), windows_sys::Win32::System::Threading::THREAD_PRIORITY_BELOW_NORMAL);
    }
    loop {
        let (tr, addr, key) = {
            let mut g = ASYNC_JOBS.lock();
            loop {
                if let Some(j) = g.pop_front() {
                    break j;
                }
                ASYNC_CV.wait(&mut g);
            }
        };
        let t = std::time::Instant::now();
        let created = std::cell::Cell::new(false);
        match compute_pipe_with(&tr, addr, &created) {
            Ok(_) => {
                if created.get() {
                    ASYNC_STATS[1].fetch_add(1, Ordering::Relaxed);
                    ASYNC_STATS[2].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
                }
                if compile_trace() {
                    crate::info!("background compile: cs {addr:#x} key {key:#x} made after {} ms", t.elapsed().as_millis());
                }
            }
            Err(e) => {
                if ASYNC_STATS[3].fetch_add(1, Ordering::Relaxed) < 8 {
                    crate::warn!("gpu: background compile of cs {addr:#x} failed: {e}");
                }
                ASYNC_FAILED.lock().get_or_insert_with(HashSet::new).insert(key);
            }
        }
        if let Some(s) = ASYNC_QUEUED.lock().as_mut() {
            s.remove(&key);
        }
    }
}

/// Status line: particle programs and their background compiles.
pub fn async_line() -> String {
    let v: Vec<u64> = ASYNC_STATS.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    format!(
        "    particle programs (`async_particles`): {} known from the game files; {} dispatches left out while their pipeline compiled, {} pipelines made in the background ({:.1} s), {} failed; {} queued now\n",
        super::particles::known(),
        v[0],
        v[1],
        v[2] as f64 / 1e9,
        v[3],
        ASYNC_JOBS.lock().len()
    )
}

/// Whether the compute pipeline of a prepared shader exists or is being
/// created (a miss is a first use: `Cp::prefetch_compiles`).
pub(crate) fn compute_pipe_known(p: &Prepared) -> bool {
    let key = push_key(p.tr.key, push_fits(&[p]));
    PIPES.lock().as_ref().is_some_and(|m| m.contains_key(&key)) || PIPE_FLIGHTS.lock().as_ref().is_some_and(|m| m.contains_key(&key))
}

/// The graphics pipeline of a draw scanned ahead of its use (`draw`).
pub(crate) use draw::{precompile_gfx, precompile_gfx_translations, GfxRegs, GfxState};

/// Creates the compute pipeline of a stage prepared ahead of its use
/// (`gpu::compile` workers): Some(true) when it was created here,
/// Some(false) when it existed or another thread had it in flight, None
/// when creating it failed.
pub fn precompile(p: &Prepared) -> Option<bool> {
    if compute_pipe_known(p) {
        return Some(false);
    }
    let created = std::cell::Cell::new(false);
    let r = compute_pipe_with(&p.tr, p.addr, &created);
    if compile_trace() && created.get() {
        let key = push_key(p.tr.key, push_fits(&[p]));
        AHEAD_KEYS.lock().get_or_insert_with(HashSet::new).insert(key);
        AHEAD_SPECS.lock().get_or_insert_with(HashMap::new).entry(p.addr).or_default().push((key, p.code_hash, p.tr.spec.clone()));
        crate::info!("compile ahead: cs {:#x} key {:#x} created", p.addr, key);
    }
    r.ok().map(|_| created.get())
}

/// `DS_COMPILE_TRACE=1`: pipelines compiled ahead and first uses logged
/// with their keys (which preparations made ahead matched execution).
static AHEAD_KEYS: Mutex<Option<HashSet<u64>>> = Mutex::new(None);
/// And the keys, code hashes and specs of the compute pipelines compiled
/// ahead by program address: a first use under another key logs what
/// differs (`spec_diff`).
static AHEAD_SPECS: Mutex<Option<HashMap<u64, Vec<(u64, u64, rdna::translate::Spec)>>>> = Mutex::new(None);

/// What differs between a spec made ahead and the one of execution
/// (`DS_COMPILE_TRACE`).
fn spec_diff(a: &rdna::translate::Spec, b: &rdna::translate::Spec) -> String {
    let mut d: Vec<String> = Vec::new();
    macro_rules! fields {
        ($($f:ident),*) => {
            $(if a.$f != b.$f {
                d.push(format!("{} {:?} -> {:?}", stringify!($f), a.$f, b.$f));
            })*
        };
    }
    fields!(stage, user_sgprs, user_base, wave64, ps_input_addr, ps_inputs, barycentric, mrt, cs_local_size, cs_rsrc2, binding_base, probe, dump, force_dispatcher, dispatcher_for_loops, reg_vars, lane_guards, mip_views, lod_stats, early_tests, early_ok, per_lane, lane_votes, base_instance_sgpr, sgpr_pc, addresses, batched, write_marks, sgpr_ubo);
    if a.resources.len() != b.resources.len() {
        d.push(format!("resources {:?} -> {:?}", a.resources, b.resources));
    } else {
        for (i, (x, y)) in a.resources.iter().zip(&b.resources).enumerate() {
            if x != y {
                d.push(format!("resource {i} {x:?} -> {y:?}"));
            }
        }
    }
    if d.is_empty() {
        d.push(if a == b { "specs equal".to_string() } else { "code_base only".to_string() });
    }
    d.join(", ")
}

fn compile_trace() -> bool {
    static T: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *T.get_or_init(|| std::env::var("DS_COMPILE_TRACE").ok().as_deref() == Some("1"))
}

/// `compile_trace` for other modules (`gpu::compile`'s failed preparations).
pub(crate) fn compile_trace_on() -> bool {
    compile_trace()
}

/// The compute pipeline of a prepared shader (created on first use, outside
/// the map's lock; a pipeline another queue created meanwhile wins; one in
/// flight on another thread is waited for).
fn compute_pipe(p: &Prepared) -> Result<Pipe, String> {
    compute_pipe_with(&p.tr, p.addr, &std::cell::Cell::new(false))
}

/// Whether a compute pipeline of this translation gets a push descriptor
/// layout (`push_fits` of its stage).
fn push_fits_tr(tr: &super::shaders::Translated) -> bool {
    super::tune::push_desc() && descriptor_bindings(&tr.out).map(|b| binding_count(b.kind)).sum::<u32>() <= ctx().max_push
}

/// Creates the compute pipeline of a translation made from a shader object
/// (`gpu::precompile`) under the key its dispatches look up: Some(true)
/// when created here, Some(false) when it existed or was in flight, None
/// when creating it failed.
pub(crate) fn precompile_translation(tr: &super::shaders::Translated, addr: u64) -> Option<bool> {
    let key = push_key(tr.key, push_fits_tr(tr));
    if PIPES.lock().as_ref().is_some_and(|m| m.contains_key(&key)) || PIPE_FLIGHTS.lock().as_ref().is_some_and(|m| m.contains_key(&key)) {
        return Some(false);
    }
    let created = std::cell::Cell::new(false);
    compute_pipe_with(tr, addr, &created).ok().map(|_| created.get())
}

fn compute_pipe_with(tr: &super::shaders::Translated, addr: u64, created: &std::cell::Cell<bool>) -> Result<Pipe, String> {
    let push_mode = push_fits_tr(tr);
    let key = push_key(tr.key, push_mode);
    if let Some(x) = PIPES.lock().as_ref().and_then(|m| m.get(&key)) {
        return Ok(*x);
    }
    let (mine, flight) = {
        let mut g = PIPE_FLIGHTS.lock();
        let m = g.get_or_insert_with(HashMap::new);
        match m.get(&key) {
            Some(f) => (false, f.clone()),
            None => {
                let f = std::sync::Arc::new(PipeFlight::new());
                m.insert(key, f.clone());
                (true, f)
            }
        }
    };
    if !mine {
        let t = std::time::Instant::now();
        flight.waited();
        let r = {
            let mut d = flight.done.lock();
            while d.is_none() {
                flight.cv.wait(&mut d);
            }
            d.clone().unwrap()
        };
        PIPE_FLIGHT_WAITS[0].fetch_add(1, Ordering::Relaxed);
        PIPE_FLIGHT_WAITS[1].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        if r.is_ok() {
            PIPE_FLIGHT_WAITS[2].fetch_add(1, Ordering::Relaxed);
            return r;
        }
        // The other attempt failed: this one reports its own error.
        return compute_pipe_create(tr, addr, push_mode, key, created);
    }
    // Stored meanwhile by a thread that finished just before this one
    // registered.
    let stored = PIPES.lock().as_ref().and_then(|m| m.get(&key)).copied();
    // A panic must not leave the threads waiting for it blocked.
    let r = match stored {
        Some(x) => Ok(x),
        None => std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| compute_pipe_create(tr, addr, push_mode, key, created))).unwrap_or_else(|_| Err("pipeline creation panicked".to_string())),
    };
    *flight.done.lock() = Some(r.clone());
    flight.cv.notify_all();
    if let Some(m) = PIPE_FLIGHTS.lock().as_mut() {
        m.remove(&key);
    }
    r
}

fn compute_pipe_create(tr: &super::shaders::Translated, addr: u64, push_mode: bool, key: u64, created: &std::cell::Cell<bool>) -> Result<Pipe, String> {
    created.set(true);
    let t0 = std::time::Instant::now();
    let pipe = compute_pipeline(&tr.out, push_mode, Some(key))?;
    super::compile::note_program(addr);
    PIPES_CREATED[0].fetch_add(1, Ordering::Relaxed);
    PIPES_NS[0].fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
    if pipe.push {
        PIPES_PUSH[0].fetch_add(1, Ordering::Relaxed);
    }
    super::pstats::report(pipe.pipeline, &format!("cs {:#x}", addr));
    super::hitch::note_since(super::hitch::Kind::Pipeline, t0, || format!("cs {addr:#x}"));
    let ms = t0.elapsed().as_millis();
    if ms >= 200 {
        crate::info!("gpu: compute pipeline for cs {:#x} ({} SPIR-V words) took {ms} ms", addr, tr.out.spirv.len());
    }
    Ok(keep_pipe(&PIPES, key, pipe))
}

/// Graphics pipeline cached under a state hash, if any.
fn gfx_pipe(hkey: u64) -> Option<Pipe> {
    GFX_PIPES.lock().as_ref().and_then(|m| m.get(&hkey)).copied()
}

fn keep_gfx_pipe(hkey: u64, pipe: Pipe) -> Pipe {
    keep_pipe(&GFX_PIPES, hkey, pipe)
}

/// Stores a newly created pipeline unless another queue stored one under
/// the same key meanwhile (then the new one is destroyed unused).
fn keep_pipe(map: &Mutex<Option<HashMap<u64, Pipe>>>, key: u64, pipe: Pipe) -> Pipe {
    let mut g = map.lock();
    let m = g.get_or_insert_with(HashMap::new);
    if let Some(x) = m.get(&key) {
        unsafe {
            let d = &ctx().device;
            d.destroy_pipeline(pipe.pipeline, None);
            d.destroy_pipeline_layout(pipe.layout, None);
            d.destroy_descriptor_set_layout(pipe.set_layout, None);
        }
        return *x;
    }
    m.insert(key, pipe);
    pipe
}

/// A translation's SPIR-V as handed to the driver: with `DS_SPIRV_NONCE=1`
/// (tests of first-use compiles) the header's generator word is made unique
/// to this session, so the driver's shader cache serves nothing it compiled
/// in an earlier one.
pub(super) fn module_code(spirv: &[u32]) -> std::borrow::Cow<'_, [u32]> {
    static NONCE: std::sync::OnceLock<Option<u32>> = std::sync::OnceLock::new();
    let nonce = *NONCE.get_or_init(|| {
        (FORCE_NONCE.load(Ordering::Relaxed) || std::env::var("DS_SPIRV_NONCE").ok().as_deref() == Some("1")).then(|| {
            let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(1, |d| d.as_nanos() as u64);
            crate::warn!("gpu: DS_SPIRV_NONCE=1: shader modules carry a session nonce (the driver's shader cache misses)");
            (t ^ (t >> 32)) as u32 | 1
        })
    });
    match nonce {
        Some(n) if spirv.len() > 2 => {
            let mut v = spirv.to_vec();
            v[2] = n;
            std::borrow::Cow::Owned(v)
        }
        _ => std::borrow::Cow::Borrowed(spirv),
    }
}

/// Modules carry the session nonce whatever `DS_SPIRV_NONCE` says
/// (`--compile-bench`; set before the first module is made).
pub(super) static FORCE_NONCE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Creates and destroys the compute pipeline of a translation without a
/// pipeline cache (`--compile-bench`): how long its creation took.
pub(super) fn bench_compute_pipeline(tr: &super::shaders::Translated) -> Result<std::time::Duration, String> {
    let t = std::time::Instant::now();
    let pipe = compute_pipeline(&tr.out, push_fits_tr(tr), None)?;
    let d = t.elapsed();
    super::pstats::report(pipe.pipeline, &format!("compile bench {:#018x}", tr.key));
    unsafe {
        let dev = &ctx().device;
        dev.destroy_pipeline(pipe.pipeline, None);
        dev.destroy_pipeline_layout(pipe.layout, None);
        dev.destroy_descriptor_set_layout(pipe.set_layout, None);
    }
    Ok(d)
}

/// Pipeline creation flags `compute_pipeline` adds (`--pipeline-test` with
/// `DS_PT_NOOPT=1`: VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT).
static EXTRA_PIPE_FLAGS: AtomicU64 = AtomicU64::new(0);

/// Creates a compute pipeline (and its layouts) for a translated shader
/// (`push_mode`: a push descriptor layout if it fits, `push_layout`; with
/// `cache_key`, through that key's pipeline cache file).
fn compute_pipeline(out: &rdna::translate::Output, push_mode: bool, cache_key: Option<u64>) -> Result<Pipe, String> {
    let c = ctx();
    {
        unsafe {
            let bindings: Vec<vk::DescriptorSetLayoutBinding> = descriptor_bindings(out)
                .map(|b| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(b.binding)
                        .descriptor_type(descriptor_type(b.kind))
                        .descriptor_count(binding_count(b.kind))
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                })
                .collect();
            let (push, flags) = push_layout(&bindings, push_mode);
            let set_layout = c.device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().flags(flags).bindings(&bindings), None).map_err(|e| format!("set layout: {e}"))?;
            let layouts = [set_layout];
            let ranges = sgpr_pc_ranges(vk::ShaderStageFlags::COMPUTE, out.sgpr_pc.is_some());
            let layout = c.device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts).push_constant_ranges(ranges), None).map_err(|e| format!("pipeline layout: {e}"))?;
            let code = module_code(&out.spirv);
            let module = c.device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None).map_err(|e| format!("shader module: {e}"))?;
            let stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(c"main");
            let extra = vk::PipelineCreateFlags::from_raw(EXTRA_PIPE_FLAGS.load(Ordering::Relaxed) as u32);
            let ci = vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout).flags(vk::PipelineCreateFlags::DISPATCH_BASE | super::pstats::flags() | extra);
            let create = |cache| c.device.create_compute_pipelines(cache, &[ci], None).map_err(|(_, e)| format!("compute pipeline: {e}"));
            let pipeline = match cache_key {
                Some(k) => super::pcache::with_cache('c', k, create),
                None => create(vk::PipelineCache::null()),
            };
            c.device.destroy_shader_module(module, None);
            Ok(Pipe { pipeline: pipeline?[0], layout, set_layout, push })
        }
    }
}

/// `--pipeline-test <capture.txt>`: translates every compute shader of a
/// frame capture, creates its Vulkan pipeline and runs one thread group with
/// all buffers bound to zeros (images: skipped), reporting the time of each
/// step: driver acceptance and compile cost of the translator's output (the
/// driver may defer compilation to the first use). Each module gets a SPIR-V
/// generator word of its own (from the clock) so the driver's shader cache
/// does not serve it (`DS_PT_NONCE=0`: as translated); `DS_PT_NOOPT=1`
/// creates the pipelines with optimization disabled; `DS_PT_MIN_WORDS=<n>`
/// tests only translations of at least n words; `DS_PT_SPV_DIR=<dir>`
/// replaces a translation by `<dir>/<12-digit hex address>.spv` when that
/// exists (an optimizer's output of it: `spvval --opt`).
pub fn pipeline_test(capture: &str) -> i32 {
    let env_num = |k: &str, d: u64| std::env::var(k).ok().and_then(|v| v.parse::<u64>().ok()).unwrap_or(d);
    let nonce = env_num("DS_PT_NONCE", 1) != 0;
    let min_words = env_num("DS_PT_MIN_WORDS", 0) as usize;
    if env_num("DS_PT_NOOPT", 0) != 0 {
        EXTRA_PIPE_FLAGS.store(vk::PipelineCreateFlags::DISABLE_OPTIMIZATION.as_raw() as u64, Ordering::Relaxed);
    }
    let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(1, |d| d.as_nanos() as u64);
    if let Err(e) = super::vk::init() {
        crate::error!("pipeline test: Vulkan: {e}");
        return 2;
    }
    let items = match rdna::capture::translate_all(capture) {
        Ok(i) => i,
        Err(e) => {
            crate::error!("pipeline test: {e}");
            return 2;
        }
    };
    let c = ctx();
    let dummy = fixed().dummy;
    let (pool, cmd, fence, dpool) = unsafe {
        let pool = c.device.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(c.queue_family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER), None).unwrap();
        let cmd = c.device.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(pool).command_buffer_count(1)).unwrap()[0];
        let fence = c.device.create_fence(&vk::FenceCreateInfo::default(), None).unwrap();
        let sizes = [vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 256 }];
        let dpool = c.device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(4).pool_sizes(&sizes), None).unwrap();
        (pool, cmd, fence, dpool)
    };
    let (mut ok, mut failed, mut skipped) = (0, 0, 0);
    let (mut create_ms, mut run_ms) = (0u128, 0u128);
    let mut times: Vec<(u128, u128, u64, usize)> = Vec::new();
    for it in &items {
        if it.stage != rdna::translate::Stage::Compute {
            continue;
        }
        let out = match &it.out {
            Ok(o) => o,
            Err(e) => {
                failed += 1;
                crate::warn!("pipeline test: cs {:#x}: translation failed: {e}", it.addr);
                continue;
            }
        };
        if out.spirv.len() < min_words {
            continue;
        }
        let external = std::env::var("DS_PT_SPV_DIR").ok().and_then(|d| std::fs::read(std::path::Path::new(&d).join(format!("{:012x}.spv", it.addr))).ok());
        let replaced;
        let out = match external {
            Some(bytes) => {
                let mut o = out.clone();
                o.spirv = bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                replaced = o;
                &replaced
            }
            None => out,
        };
        let mut uniq;
        let out = if nonce {
            uniq = out.clone();
            // SPIR-V header word 2: the generator (not interpreted).
            uniq.spirv[2] = (stamp ^ it.addr ^ (ok as u64) << 48) as u32 | 1;
            &uniq
        } else {
            out
        };
        let t0 = std::time::Instant::now();
        // Sets of its own pool (no pushed descriptors).
        let p = match compute_pipeline(out, false, None) {
            Ok(p) => p,
            Err(e) => {
                failed += 1;
                crate::warn!("pipeline test: cs {:#x}: {e}", it.addr);
                continue;
            }
        };
        let tc = t0.elapsed().as_millis();
        create_ms += tc;
        let buffers_only = descriptor_bindings(out).all(|b| descriptor_type(b.kind) == vk::DescriptorType::STORAGE_BUFFER);
        let mut tr = 0u128;
        if buffers_only {
            unsafe {
                let layouts = [p.set_layout];
                let set = c.device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(dpool).set_layouts(&layouts)).unwrap()[0];
                let infos: Vec<vk::DescriptorBufferInfo> = out.bindings.iter().map(|_| vk::DescriptorBufferInfo { buffer: dummy, offset: 0, range: DUMMY_BYTES }).collect();
                let writes: Vec<vk::WriteDescriptorSet> = out
                    .bindings
                    .iter()
                    .zip(&infos)
                    .map(|(b, i)| vk::WriteDescriptorSet::default().dst_set(set).dst_binding(b.binding).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(std::slice::from_ref(i)))
                    .collect();
                c.device.update_descriptor_sets(&writes, &[]);
                c.device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).ok();
                c.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, p.pipeline);
                c.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, p.layout, 0, &[set], &[]);
                c.device.cmd_dispatch(cmd, 1, 1, 1);
                c.device.end_command_buffer(cmd).ok();
                let t1 = std::time::Instant::now();
                if c.submit(cmd, &[], &[], fence).is_ok() {
                    if let Err(e) = c.device.wait_for_fences(&[fence], true, 30_000_000_000) {
                        crate::error!("pipeline test: cs {:#x}: {e}", it.addr);
                    }
                }
                tr = t1.elapsed().as_millis();
                c.device.reset_fences(&[fence]).ok();
                c.device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty()).ok();
                c.device.reset_descriptor_pool(dpool, vk::DescriptorPoolResetFlags::empty()).ok();
            }
            run_ms += tr;
        } else {
            skipped += 1;
        }
        ok += 1;
        times.push((tc + tr, tr, it.addr, out.spirv.len()));
        unsafe {
            c.device.destroy_pipeline(p.pipeline, None);
            c.device.destroy_pipeline_layout(p.layout, None);
            c.device.destroy_descriptor_set_layout(p.set_layout, None);
        }
    }
    unsafe {
        c.device.destroy_fence(fence, None);
        c.device.destroy_descriptor_pool(dpool, None);
        c.device.destroy_command_pool(pool, None);
    }
    times.sort_by(|a, b| b.cmp(a));
    for (total, run, addr, words) in times.iter().take(15) {
        crate::info!("pipeline test: cs {addr:#x}: {total} ms (first dispatch {run} ms; {words} SPIR-V words)");
    }
    crate::info!(
        "pipeline test: {ok} compute pipelines created ({create_ms} ms{}{}), {} dispatched ({run_ms} ms), {skipped} with images not dispatched, {failed} failed",
        if nonce { ", driver cache bypassed" } else { "" },
        if EXTRA_PIPE_FLAGS.load(Ordering::Relaxed) != 0 { ", optimization disabled" } else { "" },
        ok - skipped
    );
    if failed > 0 { 1 } else { 0 }
}

/// `--record-bench <capture> [<cs address>]`: the CPU cost of recording a
/// captured compute shader's dispatches on this driver, per dispatch: with
/// all its descriptors pushed, a third of them pushed, none (pushed once
/// before), and one descriptor set bound; the dispatches are recorded, not
/// executed. The shader: the given one, else the buffers-only compute
/// shader with the most bindings.
pub fn record_bench(capture: &str, shader: Option<u64>) -> i32 {
    if let Err(e) = super::vk::init() {
        crate::error!("record bench: Vulkan: {e}");
        return 2;
    }
    let items = match rdna::capture::translate_all(capture) {
        Ok(i) => i,
        Err(e) => {
            crate::error!("record bench: {e}");
            return 2;
        }
    };
    let buffers_only = |o: &rdna::translate::Output| o.bindings.iter().all(|b| descriptor_type(b.kind) == vk::DescriptorType::STORAGE_BUFFER);
    let pick = items
        .iter()
        .filter(|it| it.stage == rdna::translate::Stage::Compute)
        .filter_map(|it| it.out.as_ref().ok().map(|o| (it.addr, o)))
        .filter(|(a, o)| buffers_only(o) && shader.is_none_or(|s| s == *a))
        .max_by_key(|(_, o)| o.bindings.len());
    let Some((addr, out)) = pick else {
        crate::error!("record bench: no buffers-only compute shader{} in {capture}", shader.map(|s| format!(" at {s:#x}")).unwrap_or_default());
        return 2;
    };
    let c = ctx();
    let dummy = fixed().dummy;
    let n = out.bindings.len();
    let (push, set_pipe) = match (compute_pipeline(out, true, None), compute_pipeline(out, false, None)) {
        (Ok(a), Ok(b)) => (a, b),
        (Err(e), _) | (_, Err(e)) => {
            crate::error!("record bench: cs {addr:#x}: {e}");
            return 2;
        }
    };
    const N: u32 = 20_000;
    let (pool, cmd, dpool, set) = unsafe {
        let pool = c.device.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(c.queue_family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER), None).unwrap();
        let cmd = c.device.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(pool).command_buffer_count(1)).unwrap()[0];
        let sizes = [vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 256 }];
        let dpool = c.device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(4).pool_sizes(&sizes), None).unwrap();
        let layouts = [set_pipe.set_layout];
        let set = c.device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(dpool).set_layouts(&layouts)).unwrap()[0];
        let infos: Vec<vk::DescriptorBufferInfo> = out.bindings.iter().map(|_| vk::DescriptorBufferInfo { buffer: dummy, offset: 0, range: DUMMY_BYTES }).collect();
        let writes: Vec<vk::WriteDescriptorSet> = out.bindings.iter().zip(&infos).map(|(b, i)| vk::WriteDescriptorSet::default().dst_set(set).dst_binding(b.binding).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(std::slice::from_ref(i))).collect();
        c.device.update_descriptor_sets(&writes, &[]);
        (pool, cmd, dpool, set)
    };
    // The pushed values change every dispatch (ranges vary), as a command's do.
    let info = |i: u32| vk::DescriptorBufferInfo { buffer: dummy, offset: 0, range: DUMMY_BYTES - 16 * (i % 8) as u64 };
    let mut lines = Vec::new();
    let mut ends: Vec<(f64, f64)> = Vec::new();
    for variant in ["all pushed", "a third pushed", "none pushed", "set bound"] {
        let mut per = Vec::new();
        for _rep in 0..5 {
            unsafe {
                c.device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).ok();
                let p = if variant == "set bound" { &set_pipe } else { &push };
                c.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, p.pipeline);
                let all: Vec<vk::DescriptorBufferInfo> = (0..n as u32).map(info).collect();
                if variant != "set bound" {
                    let w: Vec<vk::WriteDescriptorSet> = out.bindings.iter().zip(&all).map(|(b, i)| vk::WriteDescriptorSet::default().dst_binding(b.binding).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(std::slice::from_ref(i))).collect();
                    c.push_desc.cmd_push_descriptor_set(cmd, vk::PipelineBindPoint::COMPUTE, p.layout, 0, &w);
                }
                let k = match variant {
                    "all pushed" => n,
                    "a third pushed" => n.div_ceil(3),
                    _ => 0,
                };
                let t0 = std::time::Instant::now();
                for i in 0..N {
                    match variant {
                        "set bound" => c.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, p.layout, 0, &[set], &[]),
                        _ if k > 0 => {
                            let mut buf = [const { std::mem::MaybeUninit::<vk::WriteDescriptorSet>::uninit() }; 32];
                            let vals: [vk::DescriptorBufferInfo; 32] = std::array::from_fn(|j| info(i + j as u32));
                            for (j, b) in out.bindings.iter().take(k.min(32)).enumerate() {
                                buf[j].write(vk::WriteDescriptorSet::default().dst_binding(b.binding).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(std::slice::from_ref(&vals[j])));
                            }
                            let w = std::slice::from_raw_parts(buf.as_ptr() as *const vk::WriteDescriptorSet, k.min(32));
                            c.push_desc.cmd_push_descriptor_set(cmd, vk::PipelineBindPoint::COMPUTE, p.layout, 0, w);
                        }
                        _ => {}
                    }
                    c.device.cmd_dispatch(cmd, 1, 1, 1);
                }
                per.push(t0.elapsed().as_nanos() as f64 / N as f64);
                let t1 = std::time::Instant::now();
                c.device.end_command_buffer(cmd).ok();
                let t2 = std::time::Instant::now();
                c.device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty()).ok();
                let t3 = std::time::Instant::now();
                c.device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).ok();
                c.device.end_command_buffer(cmd).ok();
                c.device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty()).ok();
                ends.push(((t2 - t1).as_nanos() as f64 / 1000.0, (t3 - t2).as_nanos() as f64 / 1000.0));
            }
        }
        per.sort_by(|a, b| a.partial_cmp(b).unwrap());
        lines.push(format!("{variant}: {:.0} ns per dispatch (median of 5; min {:.0}, max {:.0})", per[2], per[0], per[4]));
    }
    // End and reset of a command buffer holding N dispatches (the batch
    // overheads each command processor pays per batch).
    ends.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    if let Some(&(e, r)) = ends.get(ends.len() / 2) {
        lines.push(format!("end of a command buffer of {N} dispatches {e:.0} us, its reset {r:.0} us (median)"));
    }
    for l in &lines {
        crate::info!("record bench: cs {addr:#x} ({n} bindings), {N} dispatches recorded: {l}");
    }
    unsafe {
        c.device.destroy_descriptor_pool(dpool, None);
        c.device.destroy_command_pool(pool, None);
        for p in [&push, &set_pipe] {
            c.device.destroy_pipeline(p.pipeline, None);
            c.device.destroy_pipeline_layout(p.layout, None);
            c.device.destroy_descriptor_set_layout(p.set_layout, None);
        }
    }
    0
}

fn descriptor_type(k: BindingKind) -> vk::DescriptorType {
    match k {
        BindingKind::SgprData | BindingKind::Buffer(_) | BindingKind::Address(_) | BindingKind::Gds | BindingKind::Debug | BindingKind::Watchdog | BindingKind::LodStats | BindingKind::Marks => vk::DescriptorType::STORAGE_BUFFER,
        BindingKind::SgprUniform => vk::DescriptorType::UNIFORM_BUFFER,
        BindingKind::Image(_) | BindingKind::ImageArray(_) => vk::DescriptorType::SAMPLED_IMAGE,
        BindingKind::StorageImage(_) | BindingKind::StorageImageMips(_) => vk::DescriptorType::STORAGE_IMAGE,
        BindingKind::Sampler(_) | BindingKind::CompareSampler(_) => vk::DescriptorType::SAMPLER,
    }
}

/// Descriptors in a binding (image tables and storage images bound per mip
/// level are arrays).
fn binding_count(k: BindingKind) -> u32 {
    match k {
        BindingKind::ImageArray(_) => rdna::translate::IMAGE_TABLE_LEN,
        BindingKind::StorageImageMips(_) => rdna::translate::MIP_VIEWS,
        _ => 1,
    }
}

/// The descriptor writes of a command's stages (`set`: ignored when pushed).
fn descriptor_writes(set: vk::DescriptorSet, bounds: &[Bound]) -> Vec<vk::WriteDescriptorSet<'_>> {
    let mut writes: Vec<vk::WriteDescriptorSet> = Vec::with_capacity(bounds.iter().map(|b| b.bufs.len() + b.imgs.len() + b.img_arrays.len()).sum());
    for b in bounds {
        for (binding, info) in &b.bufs {
            writes.push(vk::WriteDescriptorSet::default().dst_set(set).dst_binding(*binding).descriptor_type(b.buffer_type(*binding)).buffer_info(std::slice::from_ref(info)));
        }
        for (binding, ty, info) in &b.imgs {
            writes.push(vk::WriteDescriptorSet::default().dst_set(set).dst_binding(*binding).descriptor_type(*ty).image_info(std::slice::from_ref(info)));
        }
        for (binding, ty, infos) in &b.img_arrays {
            writes.push(vk::WriteDescriptorSet::default().dst_set(set).dst_binding(*binding).descriptor_type(*ty).image_info(infos));
        }
    }
    writes
}

/// A pushed descriptor's value (`PushedSet`).
#[derive(Clone, Copy)]
enum PushedDesc {
    Buffer(vk::Buffer, u64, u64),
    Image(vk::DescriptorType, vk::Sampler, vk::ImageView, vk::ImageLayout),
    /// Image arrays are pushed again every time.
    Array,
}

impl PushedDesc {
    fn same(&self, o: &PushedDesc) -> bool {
        match (self, o) {
            (PushedDesc::Buffer(a, b, c), PushedDesc::Buffer(x, y, z)) => a == x && b == y && c == z,
            (PushedDesc::Image(a, b, c, d), PushedDesc::Image(w, x, y, z)) => a == w && b == x && c == y && d == z,
            _ => false,
        }
    }
}

/// The push descriptors of one bind point in a command buffer: the layout
/// they were pushed with and each binding's value in push order. A push
/// with the same layout and no pipeline bound since writes only the
/// bindings that changed (Vulkan keeps the others): consecutive commands
/// of a shader mostly change only their SGPR data (the per-item culls also
/// two of their eight resources), and the driver's work per push and
/// draw or dispatch grows with the descriptors pushed.
#[derive(Default)]
pub(crate) struct PushedSet {
    cmd: vk::CommandBuffer,
    layout: vk::PipelineLayout,
    descs: Vec<(u32, PushedDesc)>,
}

/// Pushes or partial pushes (commands) and descriptors pushed and left out
/// (diagnostics, `push_line`).
pub(super) static PUSH_STATS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

thread_local! {
    /// This thread's PUSH_STATS not yet added.
    static PUSH_LOCAL: std::cell::Cell<[u32; 4]> = const { std::cell::Cell::new([0; 4]) };
}

fn note_push(partial: bool, pushed: usize, left: usize) {
    PUSH_LOCAL.with(|c| {
        let mut v = c.get();
        v[partial as usize] += 1;
        v[2] += pushed as u32;
        v[3] += left as u32;
        if v[0] + v[1] >= 256 {
            for (k, x) in v.iter().enumerate() {
                PUSH_STATS[k].fetch_add(*x as u64, Ordering::Relaxed);
            }
            v = [0; 4];
        }
        c.set(v);
    });
}

/// Status line: push descriptor writes left out as unchanged.
pub(super) fn push_line() -> String {
    let v: Vec<u64> = PUSH_STATS.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    format!(
        "    descriptor pushes: {} full, {} partial; {} descriptors pushed, {} left out as unchanged; buffer bindings by device address {}; batched dispatches {} holding {} more commands, {} commands of batched programs on their own\n",
        v[0],
        v[1],
        v[2],
        v[3],
        ADDRESSED.load(Ordering::Relaxed),
        BATCHED[1].load(Ordering::Relaxed),
        BATCHED[0].load(Ordering::Relaxed),
        BATCHED[2].load(Ordering::Relaxed)
    ) + &batch_miss_line()
}

/// Why batched programs' commands did not join the open batched dispatch
/// (`BATCH_MISS`).
fn batch_miss_line() -> String {
    let m: Vec<u64> = BATCH_MISS.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    const WHY: [&str; 8] = ["indirect", "not plain 1-D", "none open", "another batch", "commands since", "another pipeline", "table full", "other descriptors"];
    let mut progs: Vec<((u64, u8), u64)> = BATCH_MISS_PROGRAMS.lock().as_ref().map(|m| m.iter().map(|(k, v)| (*k, *v)).collect()).unwrap_or_default();
    progs.sort_by(|a, b| b.1.cmp(&a.1));
    let progs: Vec<String> = progs.iter().take(10).map(|((a, w), n)| format!("cs {a:#x} {} {n}", WHY[*w as usize])).collect();
    let mut sites: Vec<(u32, u64)> = BATCH_MISS_SITES.lock().as_ref().map(|m| m.iter().map(|(k, v)| (*k, *v)).collect()).unwrap_or_default();
    sites.sort_by(|a, b| b.1.cmp(&a.1));
    let sites: Vec<String> = sites.iter().take(6).map(|(l, n)| format!("exec.rs:{l} {n}")).collect();
    format!(
        "    batched programs' commands not joined to the open dispatch: indirect {}, not a plain 1-D direct dispatch {}, none open {}, another batch {}, commands recorded since {}, another pipeline {}, table full {}, other descriptors {}; by program: {}; recorded since by: {}; dispatches' new shadows filled in the prologue {}, before the dispatch {} (`shadow_pre`; off {}, deferred {}, shared copies {}, written earlier in the batch {}, by the main CP's open batch {}, by a closed batch {})\n",
        m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7],
        progs.join(", "),
        sites.join(", "),
        SHADOWS_PRE[0].load(Ordering::Relaxed),
        SHADOWS_PRE[1].load(Ordering::Relaxed),
        SHADOWS_PRE[2].load(Ordering::Relaxed),
        SHADOWS_PRE[3].load(Ordering::Relaxed),
        SHADOWS_PRE[4].load(Ordering::Relaxed),
        SHADOWS_PRE[5].load(Ordering::Relaxed),
        SHADOWS_PRE[6].load(Ordering::Relaxed),
        SHADOWS_PRE[7].load(Ordering::Relaxed)
    )
}

/// Records a command's descriptors: pushed into the command buffer (push
/// descriptor layouts, `set` None) or its written set bound. `state`: what
/// this bind point holds; `rebound`: a pipeline was bound for this command
/// (the push then writes every binding).
fn bind_descriptors(cmd: vk::CommandBuffer, at: vk::PipelineBindPoint, layout: vk::PipelineLayout, set: Option<vk::DescriptorSet>, bounds: &[Bound], state: &mut PushedSet, rebound: bool) {
    let c = ctx();
    match set {
        Some(set) => {
            note_pushed(1);
            note_vk(VK_DESCRIPTORS);
            state.layout = vk::PipelineLayout::null();
            state.descs.clear();
            unsafe { c.device.cmd_bind_descriptor_sets(cmd, at, layout, 0, &[set], &[]) };
        }
        None => {
            note_pushed(0);
            note_vk(VK_DESCRIPTORS);
            // Up to 32 writes from the stack (every command pushes; a vector
            // was one allocation per command).
            const STACK_WRITES: usize = 32;
            let n: usize = bounds.iter().map(|b| b.bufs.len() + b.imgs.len() + b.img_arrays.len()).sum();
            if n == 0 {
                return;
            }
            let partial = super::tune::push_partial() && !rebound && state.cmd == cmd && state.layout == layout && state.descs.len() == n;
            if !partial {
                state.cmd = cmd;
                state.layout = layout;
                state.descs.clear();
            }
            if n > STACK_WRITES {
                let writes = descriptor_writes(vk::DescriptorSet::null(), bounds);
                unsafe { c.push_desc.cmd_push_descriptor_set(cmd, at, layout, 0, &writes) };
                state.layout = vk::PipelineLayout::null();
                state.descs.clear();
                note_push(false, n, 0);
                return;
            }
            let mut buf = [const { std::mem::MaybeUninit::<vk::WriteDescriptorSet>::uninit() }; STACK_WRITES];
            let mut k = 0;
            let mut j = 0;
            // The binding at push position j: kept from the last push when
            // it holds the same value, else written (and recorded).
            let mut keep = |binding: u32, d: PushedDesc| -> bool {
                let same = partial && state.descs[j].0 == binding && state.descs[j].1.same(&d);
                if partial {
                    state.descs[j] = (binding, d);
                } else {
                    state.descs.push((binding, d));
                }
                j += 1;
                same
            };
            for b in bounds {
                for (binding, info) in &b.bufs {
                    if !keep(*binding, PushedDesc::Buffer(info.buffer, info.offset, info.range)) {
                        buf[k].write(vk::WriteDescriptorSet::default().dst_binding(*binding).descriptor_type(b.buffer_type(*binding)).buffer_info(std::slice::from_ref(info)));
                        k += 1;
                    }
                }
                for (binding, ty, info) in &b.imgs {
                    if !keep(*binding, PushedDesc::Image(*ty, info.sampler, info.image_view, info.image_layout)) {
                        buf[k].write(vk::WriteDescriptorSet::default().dst_binding(*binding).descriptor_type(*ty).image_info(std::slice::from_ref(info)));
                        k += 1;
                    }
                }
                for (binding, ty, infos) in &b.img_arrays {
                    keep(*binding, PushedDesc::Array);
                    buf[k].write(vk::WriteDescriptorSet::default().dst_binding(*binding).descriptor_type(*ty).image_info(infos));
                    k += 1;
                }
            }
            note_push(partial, k, n - k);
            if k == 0 {
                return;
            }
            // The first k entries are initialized.
            let writes = unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const vk::WriteDescriptorSet, k) };
            unsafe { c.push_desc.cmd_push_descriptor_set(cmd, at, layout, 0, writes) };
        }
    }
}

/// Dispatches are ordered by the title's own synchronization (partial
/// flushes, cache actions, end-of-pipe events) instead of a full barrier
/// after every dispatch: consecutive independent dispatches overlap on the
/// GPU as on the console (`DS_LAZY_BARRIERS=0`: a barrier after each).
fn lazy_barriers() -> bool {
    super::tune::lazy_barriers()
}

/// Records prologue copies grouped by (source, destination) buffer.
fn record_copies(cmd: vk::CommandBuffer, copies: &mut Vec<(vk::Buffer, vk::Buffer, vk::BufferCopy)>) {
    if copies.is_empty() {
        return;
    }
    if super::tune::skip_copies() & 4 != 0 {
        // Diagnosis: batch-start copies left out (timing only).
        copies.clear();
        return;
    }
    // The copies' destinations are distinct ranges of video memory and their
    // sources guest memory: any order is the same.
    copies.sort_unstable_by_key(|&(s, d, r)| (s, d, r.src_offset));
    let mut regions: Vec<vk::BufferCopy> = Vec::with_capacity(copies.len());
    let mut i = 0;
    while i < copies.len() {
        let (src, dst, _) = copies[i];
        regions.clear();
        while i < copies.len() && copies[i].0 == src && copies[i].1 == dst {
            regions.push(copies[i].2);
            i += 1;
        }
        PRE_CALLS.fetch_add(1, Ordering::Relaxed);
        unsafe { ctx().device.cmd_copy_buffer(cmd, src, dst, &regions) };
    }
    copies.clear();
}

/// Records a prologue's remaining copies and ends it (`Exec::end_pre`; on
/// the submission thread with `tune submit_finish`).
fn finish_pre(pre: vk::CommandBuffer, copies: &mut Vec<(vk::Buffer, vk::Buffer, vk::BufferCopy)>, after: &mut Vec<(vk::Buffer, vk::Buffer, vk::BufferCopy)>, bq: vk::QueryPool) {
    record_copies(pre, copies);
    if !after.is_empty() {
        global_barrier(pre);
        record_copies(pre, after);
    }
    unsafe {
        global_barrier(pre);
        if bq != vk::QueryPool::null() {
            ctx().device.cmd_write_timestamp(pre, vk::PipelineStageFlags::BOTTOM_OF_PIPE, bq, BQ_PRE + 1);
        }
        if super::vk::labels_on() {
            super::vk::label_end(pre);
        }
        ctx().device.end_command_buffer(pre).ok();
    }
}

/// Records a batch's copies for the copy queue into its `xpre` (`Exec::end_xpre`).
fn finish_xpre(x: vk::CommandBuffer, copies: &mut Vec<(vk::Buffer, vk::Buffer, vk::BufferCopy)>) {
    unsafe {
        ctx().device.begin_command_buffer(x, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).ok();
        record_copies(x, copies);
        ctx().device.end_command_buffer(x).ok();
    }
}

/// Ends a batch's command buffer: its end timestamp (`Slot::bq`; the
/// prologue's read as unavailable when it has none) and a barrier making
/// shader writes to imported memory visible to the host.
fn finish_main(cmd: vk::CommandBuffer, bq: vk::QueryPool, has_pre: bool) {
    let c = ctx();
    unsafe {
        if bq != vk::QueryPool::null() {
            c.device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, bq, BQ_BATCH + 1);
            if !has_pre {
                c.device.cmd_reset_query_pool(cmd, bq, BQ_PRE, 2);
            }
        }
        let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::MEMORY_WRITE).dst_access_mask(vk::AccessFlags::HOST_READ);
        c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::HOST, vk::DependencyFlags::empty(), &[mb], &[], &[]);
        c.device.end_command_buffer(cmd).ok();
    }
}

/// Counts a batch's copy-queue use (`XCOPY`).
fn note_copy_waits(own: bool, wait: u64) {
    if own {
        XCOPY[2].fetch_add(1, Ordering::Relaxed);
    } else if wait > 0 {
        XCOPY[3].fetch_add(1, Ordering::Relaxed);
    }
}

/// Submits batch `seq`: its copies on the copy queue (`xpre`; they signal
/// the copy timeline with `seq`), then its prologue and commands on the main
/// queue once its own copies there and those it binds that earlier batches
/// made there (`wait`) are done. Called in sequence order (`push_batch`):
/// the timeline's signals increase.
fn submit_batch(seq: u64, xpre: Option<vk::CommandBuffer>, pre: Option<vk::CommandBuffer>, cmd: vk::CommandBuffer, wait: u64, fence: vk::Fence, alt: bool) -> Result<(), String> {
    let c = ctx();
    // Guest page table updates queued before this batch, ahead of it.
    super::pagetable::submit_pending(c)?;
    SUBMIT_KIND.with(|k| k.set(if xpre.is_some() { 1 } else if wait > 0 { 2 } else { 0 }));
    let mut wait = wait;
    if let Some(x) = xpre {
        let q = c.copy.as_ref().ok_or("no copy queue")?;
        c.submit_timeline(true, &[x], &[], &[(q.copied, seq)], vk::Fence::null())?;
        wait = seq;
    }
    let both;
    let cmds: &[vk::CommandBuffer] = match pre {
        Some(p) => {
            both = [p, cmd];
            &both
        }
        None => std::slice::from_ref(&cmd),
    };
    // On the main queue every batch signals the batch timeline with its
    // sequence (the completion thread waits for that value).
    let tl = [(c.batches, seq)];
    let signals: &[(vk::Semaphore, u64)] = if alt { &[] } else { &tl };
    match (&c.copy, wait) {
        (Some(q), w) if w > 0 => c.submit_timeline_on(false, alt, cmds, &[(q.copied, w, vk::PipelineStageFlags2::ALL_COMMANDS)], signals, fence),
        _ => c.submit_timeline_on(false, alt, cmds, &[], signals, fence),
    }
}

/// A batch's submission as the submission thread hands it to Vulkan
/// (`submit_specs`): its command buffers (the copy queue's part, the
/// prologue, the batch), the copy queue's timeline value it waits for, its
/// fence, whether it runs on the second queue, its executor's name.
pub(super) struct SubmitSpec {
    xpre: Option<vk::CommandBuffer>,
    pre: Option<vk::CommandBuffer>,
    cmd: vk::CommandBuffer,
    wait: u64,
    fence: vk::Fence,
    alt: bool,
    name: String,
}

/// Batches submitted in one vkQueueSubmit2 with others (`tune
/// submit_coalesce`): [calls, batches in them].
pub static COALESCED: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];

/// Submits batches in sequence order: consecutive ones on the main queue
/// without a copy-queue part go to Vulkan in one call (`tune
/// submit_coalesce`: at most that many; the GPU then starts each right
/// after the one before instead of after a submission of its own), the
/// others one by one as `submit_batch`.
fn submit_specs(specs: Vec<(u64, SubmitSpec)>) {
    if specs.is_empty() || DEVICE_LOST.load(Ordering::Relaxed) {
        return;
    }
    let fail = |e: String, name: &str| {
        if !DEVICE_LOST.swap(true, Ordering::SeqCst) {
            crate::error!("gpu[{name}]: {e}; GPU execution stops; checkpoints: {}", super::vk::checkpoint_report());
        }
    };
    let plain = |s: &SubmitSpec| s.xpre.is_none() && !s.alt;
    let max = (super::tune::submit_coalesce() as usize).max(1);
    let mut i = 0;
    while i < specs.len() {
        if DEVICE_LOST.load(Ordering::Relaxed) {
            return;
        }
        let mut j = i + 1;
        if plain(&specs[i].1) {
            while j < specs.len() && j - i < max && plain(&specs[j].1) {
                j += 1;
            }
        }
        if j - i == 1 {
            let (seq, s) = &specs[i];
            if let Err(e) = submit_batch(*seq, s.xpre, s.pre, s.cmd, s.wait, s.fence, s.alt) {
                fail(e, &s.name);
            }
            i = j;
            continue;
        }
        let c = ctx();
        if let Err(e) = super::pagetable::submit_pending(c) {
            fail(e, &specs[i].1.name);
            return;
        }
        let cmds: Vec<Vec<vk::CommandBuffer>> = specs[i..j].iter().map(|(_, s)| s.pre.into_iter().chain([s.cmd]).collect()).collect();
        let list: Vec<(&[vk::CommandBuffer], u64, u64)> = specs[i..j].iter().zip(&cmds).map(|((seq, s), c)| (c.as_slice(), s.wait, *seq)).collect();
        COALESCED[0].fetch_add(1, Ordering::Relaxed);
        COALESCED[1].fetch_add((j - i) as u64, Ordering::Relaxed);
        if let Err(e) = c.submit_batches(&list, specs[j - 1].1.fence) {
            fail(e, &specs[i].1.name);
            return;
        }
        i = j;
    }
}

#[track_caller]
fn global_barrier(cmd: vk::CommandBuffer) {
    note_vk(VK_BARRIERS);
    if profiling() {
        let l = std::panic::Location::caller();
        *BARRIER_SITES.lock().get_or_insert_with(HashMap::new).entry((l.file(), l.line())).or_default() += 1;
    }
    let mb = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
        .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
    unsafe { ctx().device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::ALL_COMMANDS, vk::DependencyFlags::empty(), &[mb], &[], &[]) };
}

/// SGPR data ring space per slot.
const RING_SEG: u64 = 4 << 20;
/// Descriptor pool of a slot: sets, then descriptors of each POOL_TYPES type.
const POOL_LIMITS: [u32; 6] = [4096, 32768, 8192, 2048, 8192, 4096];
const POOL_TYPES: [vk::DescriptorType; 5] = [vk::DescriptorType::STORAGE_BUFFER, vk::DescriptorType::SAMPLED_IMAGE, vk::DescriptorType::STORAGE_IMAGE, vk::DescriptorType::SAMPLER, vk::DescriptorType::UNIFORM_BUFFER];

/// `DS_TRACE_CS=<hex address>[,...]`: logs those compute shaders' scalar-load
/// values and resources whenever the loads change (diagnosing data-dependent
/// hangs, inputs of passes that frame captures miss).
/// A traced wait for a read of `c`: which loads or descriptors of `stages`
/// read it (their programs are saved to the run directory once).
pub(super) fn explain_wait(queue: &str, c: (u64, u64), stages: &[&Prepared]) {
    static SAVED: Mutex<Option<HashSet<u64>>> = Mutex::new(None);
    for p in stages {
        let hits: Vec<String> = shaders::read_sources(p).into_iter().filter(|(_, a, n)| *a < c.0 + c.1 && c.0 < a + n).map(|(d, a, n)| format!("{d} @{a:#x}+{n:#x}")).collect();
        if hits.is_empty() {
            continue;
        }
        if SAVED.lock().get_or_insert_with(HashSet::new).insert(p.addr) {
            super::capture::save_program(p.addr);
        }
        // The colliding bytes as they are now (the wait completed the writer).
        let words: Vec<String> = (0..(c.1 / 4).min(16)).map(|k| c.0 + 4 * k).filter(|&a| super::cp::guest_readable(a, 4)).map(|a| format!("{:08x}", unsafe { std::ptr::read_volatile(a as *const u32) })).collect();
        crate::info!("gpu[{queue}]:   read by {:?} shader {:#x}: {} = [{}]", p.stage, p.addr, hits.join("; "), words.join(" "));
    }
}

fn trace_loads(queue: &str, p: &Prepared, groups: [u32; 3]) {
    static ADDRS: std::sync::OnceLock<Vec<u64>> = std::sync::OnceLock::new();
    let addrs = ADDRS.get_or_init(|| {
        std::env::var("DS_TRACE_CS").unwrap_or_default().split(',').filter_map(|v| u64::from_str_radix(v.trim().trim_start_matches("0x"), 16).ok()).collect()
    });
    if !addrs.contains(&p.addr) {
        return;
    }
    static LAST: Mutex<Option<HashMap<u64, Vec<u32>>>> = Mutex::new(None);
    let mut g = LAST.lock();
    let last = g.get_or_insert_with(HashMap::new).entry(p.addr).or_default();
    if *last != p.sgpr_data {
        *last = p.sgpr_data.clone();
        crate::info!("trace cs {:#x} [{queue}] {}x{}x{}: {}", p.addr, groups[0], groups[1], groups[2], shaders::describe_loads(p).join(" | "));
        crate::info!("trace cs {:#x} resources: {}", p.addr, shaders::describe_resources(p).join(" | "));
    }
}

/// `DS_CULL_CENSUS=<hex shader>` (diagnosis, the culls' experiment): what
/// that program's dispatches read through their first user-data pointer
/// (the item block and its first four buffer descriptors), how often an
/// item's block and descriptors change, the per-mesh buffers' extent; a
/// summary line every 300 flips.
fn cull_census(p: &Prepared) {
    static ADDR: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    let want = *ADDR.get_or_init(|| std::env::var("DS_CULL_CENSUS").ok().and_then(|v| u64::from_str_radix(v.trim().trim_start_matches("0x"), 16).ok()).unwrap_or(u64::MAX));
    if p.addr != want {
        return;
    }
    #[derive(Default)]
    struct Item {
        /// Hashes of the block without its pass pointer (+8), of the four
        /// descriptors, of the data at the block's +24 pointer, and the
        /// pointers at +0, +16, +24, +32.
        hash: [u64; 4],
        frames: u32,
        changes: [u32; 4],
        last: u64,
    }
    #[derive(Default)]
    struct Census {
        items: HashMap<u64, Item>,
        passes: HashMap<u64, u64>,
        buffers: HashSet<(u64, u64)>,
        pages: HashSet<u64>,
        desc_pages: HashSet<u64>,
        item_pages: HashSet<u64>,
        in_homes: u64,
        lo: u64,
        hi: u64,
        dispatches: u64,
        window_dispatches: u64,
        window_start: u64,
        window_items: u64,
        window_changed: [u64; 4],
        unreadable: u64,
    }
    static C: Mutex<Option<Census>> = Mutex::new(None);
    let frame = super::FRAMES.load(Ordering::Relaxed);
    let rd = |a: u64, n: usize| -> Option<Vec<u32>> {
        if !super::cp::guest_readable(a, 4 * n as u64) {
            return None;
        }
        Some((0..n).map(|k| unsafe { std::ptr::read_volatile((a + 4 * k as u64) as *const u32) }).collect())
    };
    let user = p.user;
    if user.len() < 2 {
        return;
    }
    let item = user[0] as u64 | (user[1] as u64) << 32;
    let mut g = C.lock();
    let c = g.get_or_insert_with(|| Census { lo: u64::MAX, window_start: frame, ..Default::default() });
    c.dispatches += 1;
    c.window_dispatches += 1;
    let Some(block) = rd(item, 10) else {
        c.unreadable += 1;
        return;
    };
    let fnv = |ws: &[u32]| ws.iter().fold(0xcbf29ce484222325u64, |h, &w| (h ^ w as u64).wrapping_mul(0x100000001b3));
    let mut desc = Vec::with_capacity(16);
    for i in 0..4 {
        if let Some(Some(a)) = p.res_addrs.get(i) {
            desc.extend(rd(*a, 4).unwrap_or_default());
            c.desc_pages.insert(a & !0xFFF);
        }
    }
    let p24 = block[6] as u64 | (block[7] as u64) << 32;
    let data24 = rd(p24, 6).unwrap_or_default();
    let ptrs = [block[0], block[1], block[4], block[5], block[6], block[7], block[8], block[9]];
    c.item_pages.insert(item & !0xFFF);
    let mut wo_pass = block.clone();
    wo_pass[2] = 0;
    wo_pass[3] = 0;
    let hash = [fnv(&wo_pass), fnv(&desc), fnv(&data24), fnv(&ptrs)];
    let e = c.items.entry(item).or_default();
    for k in 0..4 {
        if e.frames != 0 && e.hash[k] != hash[k] {
            e.changes[k] += 1;
            c.window_changed[k] += 1;
        }
    }
    e.hash = hash;
    if e.last != frame + 1 {
        e.frames += 1;
        e.last = frame + 1;
        c.window_items += 1;
    }
    let pass = block[2] as u64 | (block[3] as u64) << 32;
    *c.passes.entry(pass).or_default() += 1;
    for i in 0..4 {
        if let Some(Resolved::Buffer(v)) = p.resources.get(i) {
            let size = if v.stride != 0 { v.stride as u64 * v.num_records as u64 } else { v.num_records as u64 };
            if size != 0 && c.buffers.insert((v.base, size)) {
                c.lo = c.lo.min(v.base);
                c.hi = c.hi.max(v.base + size);
                if super::homes::overlaps(v.base, size) {
                    c.in_homes += 1;
                }
                let mut q = v.base & !0xFFF;
                while q < v.base + size {
                    c.pages.insert(q);
                    q += 0x1000;
                }
            }
        }
    }
    if frame >= c.window_start + 300 {
        let flips = (frame - c.window_start).max(1);
        let bytes: u64 = c.buffers.iter().map(|b| b.1).sum();
        let changed = |k: usize| c.items.values().filter(|i| i.changes[k] > 0).count();
        let mut passes: Vec<(u64, u64)> = c.passes.iter().map(|(&a, &n)| (a, n)).collect();
        passes.sort_by_key(|p| std::cmp::Reverse(p.1));
        let top: Vec<String> = passes.iter().take(4).map(|(a, n)| format!("{a:#x} x{n}")).collect();
        crate::info!(
            "cull census {:#x}: {:.0} dispatches/flip over {flips} flips; items {} in all ({:.0}/flip seen); items changed (block without +8 / descriptors / +24 data / pointers): {} / {} / {} / {} ever, {} / {} / {} / {} changes this window; pass blocks {} [{}]; per-mesh buffers {} ({:.1} MiB, pages {:.1} MiB in {:#x}..{:#x}, {} in homes); item pages {}, descriptor pages {}; unreadable {}",
            want,
            c.window_dispatches as f64 / flips as f64,
            c.items.len(),
            c.window_items as f64 / flips as f64,
            changed(0),
            changed(1),
            changed(2),
            changed(3),
            c.window_changed[0],
            c.window_changed[1],
            c.window_changed[2],
            c.window_changed[3],
            c.passes.len(),
            top.join(", "),
            c.buffers.len(),
            bytes as f64 / (1 << 20) as f64,
            (c.pages.len() as u64 * 4096) as f64 / (1 << 20) as f64,
            c.lo,
            c.hi,
            c.in_homes,
            c.item_pages.len(),
            c.desc_pages.len(),
            c.unreadable
        );
        c.window_start = frame;
        c.window_dispatches = 0;
        c.window_items = 0;
        c.window_changed = [0; 4];
    }
}

/// Descriptor writes of one shader stage.
#[derive(Default)]
struct Bound {
    bufs: Vec<(u32, vk::DescriptorBufferInfo)>,
    imgs: Vec<(u32, vk::DescriptorType, vk::DescriptorImageInfo)>,
    /// Image arrays: image tables (sampled), storage images bound per mip
    /// level.
    img_arrays: Vec<(u32, vk::DescriptorType, Vec<vk::DescriptorImageInfo>)>,
    /// Guest ranges the stage writes.
    writes: Vec<(u64, u64)>,
    /// Guest ranges its buffer bindings only read (unbounded pointers to
    /// their mapping's end; `TITLE_BARRIERS`).
    reads: Vec<(u64, u64)>,
    /// It binds the global data share.
    gds: bool,
    /// Those described by buffer descriptors (not unbounded pointers).
    sized_writes: Vec<(u64, u64)>,
    /// Images it samples or stores to.
    images: Vec<vk::Image>,
    /// Those it stores to, and those it samples through an R8_UINT view
    /// (the stencil aspect of a depth image; other views of one read depth).
    stored: Vec<vk::Image>,
    stencil_views: Vec<vk::Image>,
    /// Buffers bound through a VRAM shadow (copied in before, out after).
    shadows: Vec<Shadow>,
    /// Written buffer resources and where their result is after the stage:
    /// (guest base, size, buffer, offset of the base) - mirror refreshes.
    write_src: Vec<(u64, u64, vk::Buffer, u64)>,
    /// Where each buffer binding was placed (DS_PROFILE_GPU): (Placement,
    /// bound bytes, written).
    placements: Vec<(Placement, u64, bool)>,
    /// Buffer bindings read where guest memory (or its video-memory home)
    /// holds the data, with `tune race_checks`: (base, size, kind: 1
    /// written, 2 data-only loads read on the GPU, 4 an unbounded pointer).
    in_place: Vec<(u64, u64, u8)>,
    /// The stage's SGPR data offset as pushed (`Spec::sgpr_pc`): (byte
    /// offset in the push constants, dword offset in the ring).
    sgpr_base: Option<(u32, u32)>,
    /// The binding of its SGPR data when that is a uniform buffer
    /// (`BindingKind::SgprUniform`); its other buffers are storage buffers.
    ubo: Option<u32>,
    /// A batched translation's uniform window of the ring (`Spec::sgpr_ubo`
    /// 3): its offset (the table and SGPR data offsets are relative to it).
    window: Option<u64>,
}

impl Bound {
    /// The descriptor type of buffer binding `binding`.
    fn buffer_type(&self, binding: u32) -> vk::DescriptorType {
        if self.ubo == Some(binding) {
            vk::DescriptorType::UNIFORM_BUFFER
        } else {
            vk::DescriptorType::STORAGE_BUFFER
        }
    }
}

/// The push constant range of a pipeline layout whose stages read their
/// SGPR data offset from it (`Spec::sgpr_pc`): one dword per stage (a
/// draw's vertex and pixel stages, `sgpr_pc_offset`).
fn sgpr_pc_ranges(stages: vk::ShaderStageFlags, used: bool) -> &'static [vk::PushConstantRange] {
    const GFX: [vk::PushConstantRange; 1] = [vk::PushConstantRange { stage_flags: vk::ShaderStageFlags::from_raw(vk::ShaderStageFlags::VERTEX.as_raw() | vk::ShaderStageFlags::FRAGMENT.as_raw()), offset: 0, size: 8 }];
    const CS: [vk::PushConstantRange; 1] = [vk::PushConstantRange { stage_flags: vk::ShaderStageFlags::COMPUTE, offset: 0, size: 4 }];
    match (used, stages == vk::ShaderStageFlags::COMPUTE) {
        (false, _) => &[],
        (true, true) => &CS,
        (true, false) => &GFX,
    }
}

/// Pushes the SGPR data offsets of a command's stages (`Spec::sgpr_pc`;
/// `layout` has `sgpr_pc_ranges` when one of them reads it).
fn push_sgpr_bases(cmd: vk::CommandBuffer, layout: vk::PipelineLayout, compute: bool, bounds: &[Bound]) {
    let mut v = [0u32; 2];
    let mut any = false;
    for b in bounds {
        if let Some((at, base)) = b.sgpr_base {
            v[(at / 4) as usize & 1] = base;
            any = true;
        }
    }
    if !any {
        return;
    }
    let (stages, n) = if compute { (vk::ShaderStageFlags::COMPUTE, 4) } else { (sgpr_pc_ranges(vk::ShaderStageFlags::VERTEX, true)[0].stage_flags, 8) };
    let bytes = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, n) };
    unsafe { ctx().device.cmd_push_constants(cmd, layout, stages, 0, bytes) };
}

/// Commands about to be recorded with an in-place buffer binding that GPU
/// work of the same batch wrote since its last full barrier (`tune
/// race_checks`): [read-only by draws, by dispatches, written by draws, by
/// dispatches, draws in the same rendering instance after a title sync
/// (its barrier waits for the next instance), commands after more writes
/// than are tracked since a barrier, read-only data-only loads by
/// dispatches, read-only unbounded pointers by dispatches].
pub static UNORDERED: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];
/// Unordered read-only bindings of dispatches by (shader, kind): count and
/// an example (`tune race_checks`).
static UNORDERED_BY: Mutex<Option<HashMap<(u64, u8), (u64, String)>>> = Mutex::new(None);

/// How a binding uses guest memory that a video-memory home may hold.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum HomeUse {
    /// A buffer descriptor's range.
    Sized,
    /// An unbounded pointer (`Resolved::Global`): the window up to the
    /// mapping's end or UNBOUNDED_MAX.
    Unbounded,
    /// A draw's index buffer (may create the home).
    Index,
}

/// Where a buffer binding's data is read and written (profile).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Placement {
    /// Imported guest memory, across PCIe.
    Host,
    /// Read-only snapshot in video memory.
    Snapshot,
    /// Read-only copy made at the start of the batch (`Exec::pre_copy`).
    PreCopy,
    /// A resident shadow's working copy (read-only binding).
    Resident,
    /// A VRAM mirror of GPU-written data (read-only binding).
    Mirror,
    /// A VRAM shadow made for this dispatch.
    Shadow,
    /// A resident shadow reused.
    ShadowReused,
    /// Not mapped / empty (zeros).
    None,
    /// A video-memory scratch buffer, results lost (`tune scratch_writes`).
    Scratch,
    /// A video-memory home (`homes`).
    Home,
    /// The residency mirror (`gpu::resident`, read-only binding).
    Mirrored,
}

const PLACEMENTS: [Placement; 11] = [Placement::Host, Placement::Snapshot, Placement::PreCopy, Placement::Resident, Placement::Mirror, Placement::Shadow, Placement::ShadowReused, Placement::None, Placement::Scratch, Placement::Home, Placement::Mirrored];

/// Video memory written instead of large unshadowed buffers (diagnosis,
/// `tune scratch_writes`).
const SCRATCH_BYTES: u64 = 256 << 20;
/// Imports refused because no alias slot covered the range (`sections`).
static ALIAS_MISSES: AtomicU64 = AtomicU64::new(0);

pub fn alias_misses() -> u64 {
    ALIAS_MISSES.load(Ordering::Relaxed)
}

/// Largest read-only binding served from the residency mirror, and a
/// batch's staging for its uploads (`Exec::resident_bind`).
const RES_MAX: u64 = 256 << 20;
/// Import windows with residency on (`WINDOW` otherwise).
const RES_WINDOW: u64 = 4 << 20;
const STAGE_SEG: u64 = 16 << 20;

/// The part of guest [lo, hi) a live import covers, if any (residency:
/// such pages are host class).
fn imported_overlap(lo: u64, hi: u64) -> Option<(u64, u64)> {
    SPANS.read().iter().find(|s| s.valid && s.start < hi && lo < s.end).map(|s| (s.start.max(lo), s.end.min(hi)))
}
static SCRATCH: std::sync::OnceLock<Option<vk::Buffer>> = std::sync::OnceLock::new();

/// Size of the buffer depth-plane copies pass through (`sync_depth_views`;
/// a 2560x1440 D32 plane is 14 MiB).
const DEPTH_COPY_BYTES: u64 = 16 << 20;
static DEPTH_COPY_BUF: std::sync::OnceLock<Option<vk::Buffer>> = std::sync::OnceLock::new();

/// The graphics command processor's buffer for depth-plane copies (only it
/// records them: parallel workers stop for them).
fn depth_copy_buffer() -> Option<vk::Buffer> {
    *DEPTH_COPY_BUF.get_or_init(|| ctx().device_buffer(DEPTH_COPY_BYTES, vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST).ok().map(|(b, _)| b))
}

/// Host memory nothing reads, for `tune skip_copies` 2048 (diagnosis).
fn host_scratch() -> Option<vk::Buffer> {
    static B: std::sync::OnceLock<Option<vk::Buffer>> = std::sync::OnceLock::new();
    *B.get_or_init(|| ctx().host_buffer(1 << 20, vk::BufferUsageFlags::STORAGE_BUFFER).ok().map(|(b, _, _)| b))
}

/// Our own write-watched memory imported as guest memory is, nothing else
/// using it, for `tune skip_copies` 4096 (diagnosis).
fn imported_scratch() -> Option<vk::Buffer> {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc};
    const MEM_WRITE_WATCH: u32 = 0x0020_0000;
    static B: std::sync::OnceLock<Option<vk::Buffer>> = std::sync::OnceLock::new();
    *B.get_or_init(|| {
        let p = unsafe { VirtualAlloc(std::ptr::null(), 1 << 20, MEM_COMMIT | MEM_RESERVE | MEM_WRITE_WATCH, PAGE_READWRITE) } as u64;
        if p == 0 {
            return None;
        }
        ctx().import_host(p, 1 << 20, vk::BufferUsageFlags::STORAGE_BUFFER).ok().map(|(b, _, _)| b)
    })
}

/// `imported_scratch` over SPREAD_BYTES, for `tune skip_copies` 8192
/// (diagnosis): each window's merge stores at its guest address modulo that
/// size, over as many pages as into guest memory.
fn imported_spread() -> Option<vk::Buffer> {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc};
    const MEM_WRITE_WATCH: u32 = 0x0020_0000;
    static B: std::sync::OnceLock<Option<vk::Buffer>> = std::sync::OnceLock::new();
    *B.get_or_init(|| {
        let p = unsafe { VirtualAlloc(std::ptr::null(), SPREAD_BYTES as usize, MEM_COMMIT | MEM_RESERVE | MEM_WRITE_WATCH, PAGE_READWRITE) } as u64;
        if p == 0 {
            return None;
        }
        ctx().import_host(p, SPREAD_BYTES, vk::BufferUsageFlags::STORAGE_BUFFER).ok().map(|(b, _, _)| b)
    })
}

const SPREAD_BYTES: u64 = 64 << 20;

fn scratch_buffer() -> Option<vk::Buffer> {
    *SCRATCH.get_or_init(|| ctx().device_buffer(SCRATCH_BYTES, vk::BufferUsageFlags::STORAGE_BUFFER).ok().map(|(b, _)| b))
}

/// `DS_TRACE_BINDS=<hex address>[-<hex end>]`: each buffer binding over the
/// range, once per (shader, written, unbounded, placement): who reads and
/// writes a buffer, and where it was placed.
fn trace_bind(queue: &str, shader: u64, base: u64, size: u64, written: bool, unbounded: bool, placement: Placement) {
    static AT: std::sync::OnceLock<(u64, u64)> = std::sync::OnceLock::new();
    let (at, end) = *AT.get_or_init(|| {
        let v = std::env::var("DS_TRACE_BINDS").unwrap_or_default();
        let num = |x: &str| u64::from_str_radix(x.trim().trim_start_matches("0x"), 16).ok();
        match v.split_once('-') {
            Some((lo, hi)) => (num(lo).unwrap_or(0), num(hi).unwrap_or(0)),
            None => num(&v).map_or((0, 0), |x| (x, x + 1)),
        }
    });
    if at == 0 || base >= end || base.saturating_add(size) <= at {
        return;
    }
    static SEEN: Mutex<Option<HashSet<(u64, bool, bool, u8)>>> = Mutex::new(None);
    let key = (shader, written, unbounded, placement as u8);
    if SEEN.lock().get_or_insert_with(HashSet::new).insert(key) {
        crate::info!("gpu[{queue}]: binding {base:#x}+{size:#x} of shader {shader:#x}: {}{}, placed {placement:?} (flip {})", if written { "written" } else { "read" }, if unbounded { ", unbounded pointer" } else { "" }, super::FRAMES.load(Ordering::Relaxed));
    }
}

/// Buffer placements and workgroups per compute shader (DS_PROFILE_GPU=1).
#[derive(Default)]
struct ShaderPlacements {
    dispatches: u64,
    groups: u64,
    indirect: u64,
    /// Per binding: [count per Placement], total bytes, written count.
    bindings: Vec<([u64; 11], u64, u64)>,
}

static SHADER_PLACEMENTS: Mutex<Option<HashMap<u64, ShaderPlacements>>> = Mutex::new(None);

/// Per pixel program (`DS_PROFILE_GPU=1`): its draws' depth and stencil
/// tests before the shader as the title asks (DEPTH_BEFORE_SHADER), before
/// it as the draw's state allows (`Spec::early_ok`), after it (a kill,
/// alpha to coverage or an export with depth or stencil writes), after it
/// as the program writes memory.
static PS_TESTS: Mutex<Option<HashMap<u64, [u64; 4]>>> = Mutex::new(None);

/// Counts a draw's depth-test class for its pixel program (`PS_TESTS`).
pub(crate) fn note_ps_tests(addr: u64, class: usize) {
    PS_TESTS.lock().get_or_insert_with(HashMap::new).entry(addr).or_default()[class.min(3)] += 1;
}

/// Totals of `PS_TESTS` (the profile report).
fn ps_tests_line() -> Option<String> {
    let g = PS_TESTS.lock();
    let m = g.as_ref()?;
    let mut t = [0u64; 4];
    for x in m.values() {
        for k in 0..4 {
            t[k] += x[k];
        }
    }
    Some(format!(
        "    pixel programs' depth tests by draw: before the shader as asked {}, as allowed {}, after it (kill, coverage or exports with writes) {}, after it (memory writes) {}\n",
        t[0], t[1], t[2], t[3]
    ))
}

pub(crate) fn note_placements(addr: u64, groups: [u32; 3], indirect: bool, placements: &[(Placement, u64, bool)]) {
    let mut g = SHADER_PLACEMENTS.lock();
    let e = g.get_or_insert_with(HashMap::new).entry(addr).or_default();
    e.dispatches += 1;
    if indirect {
        e.indirect += 1;
    } else {
        e.groups += groups.iter().map(|&x| x as u64).product::<u64>();
    }
    if e.bindings.len() < placements.len() {
        e.bindings.resize(placements.len(), ([0; 11], 0, 0));
    }
    for (b, &(k, n, w)) in e.bindings.iter_mut().zip(placements) {
        b.0[PLACEMENTS.iter().position(|&x| x == k).unwrap_or(7)] += 1;
        b.1 += n;
        b.2 += w as u64;
    }
}

/// One line on a compute shader's workgroups and buffer placements.
fn placement_line(addr: u64) -> Option<String> {
    let g = SHADER_PLACEMENTS.lock();
    let e = g.as_ref()?.get(&addr)?;
    let direct = e.dispatches - e.indirect;
    let mut s = format!("        groups per dispatch {}", if direct > 0 { format!("{:.1}", e.groups as f64 / direct as f64) } else { "-".into() });
    if e.indirect > 0 {
        s.push_str(&format!(" ({} indirect)", e.indirect));
    }
    s.push_str("; buffers:");
    for (i, (counts, bytes, written)) in e.bindings.iter().enumerate() {
        let total: u64 = counts.iter().sum();
        if total == 0 {
            continue;
        }
        let kinds: Vec<String> = PLACEMENTS.iter().zip(counts.iter().copied()).filter(|&(_, c)| c > 0).map(|(k, c)| if c == total { format!("{k:?}") } else { format!("{k:?} {}%", c * 100 / total) }).collect();
        let avg = *bytes as f64 / total as f64;
        let size = if avg >= (1 << 20) as f64 { format!("{:.1} MB", avg / (1 << 20) as f64) } else if avg >= 1024.0 { format!("{:.1} KB", avg / 1024.0) } else { format!("{avg:.0} B") };
        s.push_str(&format!(" #{i} {} {size}{}", kinds.join("/"), if *written > 0 { " written" } else { "" }));
    }
    Some(s)
}

/// A guest buffer range shadowed in device memory for one dispatch: GPU
/// atomics on host memory cross PCIe one by one (the title's compaction
/// kernels spent 17-33 ms on ~60k atomics to a few counters) and small GPU
/// stores to host memory run at ~0.2 GB/s; copies in use the DMA path.
///
/// Write-back stores only the dwords the dispatch changed (the merge kernel
/// compares with a pristine copy taken with the working copy): a buffer's
/// range is often much larger than what the shader writes, and the CPU keeps
/// using the rest. The Bink decoder's DC pass binds 0x68300 bytes that end in
/// the decoder's own CPU state; copying the whole range back reverted the
/// title's slot index and command-buffer cursor (decodes appended to stale
/// command buffers until one overflowed) and its data pointers.
///
/// A shadow stays resident for the rest of its batch: later dispatches that
/// shadow the same window use its working copy (no copy in), read-only
/// bindings inside it read the working copy, and it is merged back once,
/// before anything else uses the guest range in place and when the batch is
/// submitted (`Exec::flush_resident`). The title's culling pass binds one
/// 300 KB counter buffer (atomics) in ~170 dispatches per frame; each copied
/// it in and merged it back.
#[derive(Clone, Copy)]
struct Shadow {
    host: vk::Buffer,
    /// Guest address of the bound window (`host_off`).
    guest: u64,
    /// Bound through a resident shadow of the same window (no copy in).
    reused: bool,
    /// Bound range within `host` (aligned start) and its offset in the shadow ring.
    host_off: u64,
    range: u64,
    shadow_off: u64,
    /// Offset of the pristine copy in the shadow ring.
    pristine_off: u64,
    /// The resource's own bytes within the range (copied back).
    bias: u64,
    size: u64,
    /// When its copy in was recorded, as `precache::watch` dates it (None:
    /// not followed; its working copy is not shared after the merge).
    since: Option<u64>,
    /// Guest memory in its window was written in place meanwhile (a CP write,
    /// an aliased binding): its working copy is not what the merge leaves
    /// there, so it is not shared (`tune shadow_share`).
    tainted: bool,
    /// The shader whose binding made it (profile: merges by window).
    writer: u64,
}

/// Command processors' waits for their own batches (`Exec::flush`,
/// DS_PROFILE_GPU) by call site and whether the graphics command processor
/// waited: [count, ns].
static FLUSH_SITES: Mutex<Option<HashMap<(&'static str, u32, bool), [u64; 2]>>> = Mutex::new(None);

fn flush_site_lines(flips: u64) -> String {
    let g = FLUSH_SITES.lock();
    let Some(m) = g.as_ref() else { return String::new() };
    let per = |x: u64| x as f64 / flips.max(1) as f64;
    let mut s = String::new();
    for gfx in [true, false] {
        let mut v: Vec<(&(&str, u32, bool), &[u64; 2])> = m.iter().filter(|(k, _)| k.2 == gfx).collect();
        v.sort_by(|a, b| b.1[1].cmp(&a.1[1]));
        let total: u64 = v.iter().map(|x| x.1[1]).sum();
        let parts: Vec<String> = v
            .iter()
            .take(12)
            .map(|((f, l, _), x)| {
                let file = f.rsplit(|c| c == '/' || c == '\\').next().unwrap_or(f);
                format!("{file}:{l} {:.2} ms ({:.1}, {:.0} us each)", per(x[1]) / 1e6, per(x[0]), x[1] as f64 / x[0].max(1) as f64 / 1000.0)
            })
            .collect();
        let who = if gfx { "graphics command processor" } else { "other command processors" };
        s.push_str(&format!("    waits for own batches per flip, {who} ({:.2} ms in all): {}\n", per(total) / 1e6, parts.join(", ")));
    }
    s
}

/// Shadow merges by window (DS_PROFILE_GPU): (guest, range) -> [merges,
/// bytes, the shader that made the last one merged, its queue
/// (`queue_tag`)].
static MERGE_BY: Mutex<Option<HashMap<(u64, u64), [u64; 4]>>> = Mutex::new(None);

/// An executor's queue for profile lines: 0 the graphics command processor,
/// 1 its parallel workers, else the compute queue's number.
fn queue_tag(name: &str) -> u64 {
    match name.strip_prefix("gpu-acb0x") {
        Some(h) => u64::from_str_radix(h, 16).unwrap_or(2),
        None if name.contains("-par") => 1,
        None => 0,
    }
}

fn queue_label(tag: u64) -> String {
    match tag {
        0 => "graphics".to_string(),
        1 => "workers".to_string(),
        q => format!("queue {q:#x}"),
    }
}
/// The flip MERGE_BY started counting at.
static MERGE_FROM: AtomicU64 = AtomicU64::new(0);

/// Buffer fills in place by range (DS_PROFILE_GPU): (base, len) -> [fills, value].
static FILL_BY: Mutex<Option<HashMap<(u64, u64), [u64; 2]>>> = Mutex::new(None);

fn fill_by_lines(flips: u64) -> String {
    let g = FILL_BY.lock();
    let Some(m) = g.as_ref() else { return String::new() };
    let mut v: Vec<(&(u64, u64), &[u64; 2])> = m.iter().collect();
    v.sort_by(|a, b| (b.1[0] * b.0 .1).cmp(&(a.1[0] * a.0 .1)));
    let total: u64 = v.iter().map(|x| x.1[0]).sum();
    let bytes: u64 = v.iter().map(|x| x.1[0] * x.0 .1).sum();
    let per = |n: u64| n as f64 / flips.max(1) as f64;
    let parts: Vec<String> = v.iter().take(12).map(|((b, n), x)| format!("{b:#x}+{n:#x} (value {:#x}) {:.1}", x[1], per(x[0]))).collect();
    format!("    buffer fills in place by range, per flip ({:.1} in all, {:.2} MiB, {} ranges): {}
", per(total), per(bytes) / 1048576.0, v.len(), parts.join(", "))
}

fn merge_by_lines(flips: u64) -> String {
    let g = MERGE_BY.lock();
    let Some(m) = g.as_ref() else { return String::new() };
    let mut v: Vec<(&(u64, u64), &[u64; 4])> = m.iter().collect();
    v.sort_by(|a, b| b.1[0].cmp(&a.1[0]));
    let total: u64 = v.iter().map(|x| x.1[0]).sum();
    let per = |n: u64| n as f64 / flips.max(1) as f64;
    let parts: Vec<String> = v.iter().take(16).map(|((g, r), x)| format!("{g:#x}+{r:#x} (cs {:#x}, {}) {:.1}", x[2], queue_label(x[3]), per(x[0]))).collect();
    // The windows with the most bytes merged, and the merges by queue.
    let mut by_bytes = v.clone();
    by_bytes.sort_by(|a, b| b.1[1].cmp(&a.1[1]));
    let mib = |n: u64| per(n) / (1u64 << 20) as f64;
    let big: Vec<String> = by_bytes.iter().take(12).map(|((g, r), x)| format!("{g:#x}+{r:#x} (cs {:#x}, {}) {:.1} MiB in {:.1}", x[2], queue_label(x[3]), mib(x[1]), per(x[0]))).collect();
    let mut by_queue: std::collections::BTreeMap<u64, (u64, u64)> = std::collections::BTreeMap::new();
    for (_, x) in &v {
        let e = by_queue.entry(x[3]).or_default();
        e.0 += x[0];
        e.1 += x[1];
    }
    let queues: Vec<String> = by_queue.iter().map(|(q, (n, b))| format!("{} {:.1} ({:.1} MiB)", queue_label(*q), per(*n), mib(*b))).collect();
    // By the memory type the title gave the window's mapping.
    let mut by_type: std::collections::BTreeMap<Option<i32>, (u64, u64)> = std::collections::BTreeMap::new();
    for ((g, r), x) in &v {
        let e = by_type.entry(crate::mem::mtype_at(*g)).or_default();
        e.0 += x[0];
        e.1 += *r;
    }
    let types: Vec<String> = by_type.iter().map(|(t, (n, b))| format!("type {} {:.1} ({} KiB of windows)", t.map_or("?".to_string(), |t| t.to_string()), per(*n), b >> 10)).collect();
    format!(
        "    shadow merges by window, per flip ({:.1} in all, {} windows; by memory type: {}; by queue: {}): {}\n    shadow merges by window's bytes, per flip (MiB of windows merged, merges): {}\n",
        per(total),
        v.len(),
        types.join(", "),
        queues.join(", "),
        parts.join(", "),
        big.join(", ")
    )
}

/// Device-local shadow space per slot (working and pristine copies).
const SHADOW_BYTES: u64 = 32 << 20;
/// A slot's shadow and batch-start copy segments are made at the size its
/// executor's batches need (`Exec::seg_want`: this small to start, raised
/// to what a batch needed, at most SHADOW_BYTES and PRE_SEG) and lowered
/// again when SEG_DECAY batches in a row used under a quarter of it: fixed
/// full-size segments held 6 GiB of video memory over 63 slots, growth
/// without the decay 3 GB after six level loads, and sizes per slot split
/// batches (-5% at the gate) (FINDINGS.md 2026-10-19).
/// `DS_SLOT_SEGS=full` (boot; A/B): full size from the start.
const SHADOW_INIT: u64 = 2 << 20;
const PRE_INIT: u64 = 4 << 20;
const SEG_DECAY: u32 = 256;
/// Largest buffer shadowed by default (larger atomic targets stay in host
/// memory; `tune shadow_max`).
#[allow(dead_code)]
const SHADOW_MAX: u64 = 4 << 20;

static SHADOWED: AtomicU64 = AtomicU64::new(0);
/// Indirect dispatches that read their arguments from a resident shadow's
/// working copy (`tune shadow_args`).
static SHADOW_ARGS: AtomicU64 = AtomicU64::new(0);
/// Indirect dispatches whose arguments the GPU read (`Exec::dispatch_gpu_indirect`).
static GPU_INDIRECT: AtomicU64 = AtomicU64::new(0);
/// Barriers between draws that read their arguments on the GPU and a later
/// dispatch rewriting them (`Exec::args_read`).
pub(super) static ARGS_WAR_BARRIERS: AtomicU64 = AtomicU64::new(0);
/// CP memory writes performed by the GPU (`Exec::gpu_cp_write`).
static GPU_CP_WRITES: AtomicU64 = AtomicU64::new(0);
static GPU_TIMESTAMPS: AtomicU64 = AtomicU64::new(0);
/// Batch begins (DS_PROFILE_GPU): [count, ns including slot waits, of them
/// draws' resident-shadow checks of their indices: count, ns].
static BEGIN_STATS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];
/// Batch begins in a slot the completion thread was to reset (`tune
/// offthread_reset`): [found reset, waited for the reset, taken back].
static OFFTHREAD_RESETS: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
/// Batch submissions (DS_PROFILE_GPU): [count, ns in `submit_for` without
/// its wait for deferred CP writes].
static SUBMIT_STATS: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
/// `submit_for`'s steps (DS_PROFILE_GPU): ns per step (`SUBMIT_STEP_NAMES`).
static SUBMIT_STEPS: [AtomicU64; 11] = [const { AtomicU64::new(0) }; 11];
/// The same for the submissions before a parallel run (`Cp::par_plan`), and
/// their count last.
static SUBMIT_STEPS_RUN: [AtomicU64; 12] = [const { AtomicU64::new(0) }; 12];
const SUBMIT_STEP_NAMES: [&str; 11] = [
    "shadow write-back",
    "end rendering",
    "timestamps",
    "prologue",
    "end",
    "hand-off",
    "copies submitted",
    "promises and labels",
    "in-flight write log",
    "copy invalidation",
    "own write log",
];

/// Times the steps of `submit_for` (profiling).
struct StepTimer(Option<std::time::Instant>, bool);

impl StepTimer {
    fn mark(&mut self, k: usize) {
        if let Some(t) = self.0 {
            let now = std::time::Instant::now();
            let ns = (now - t).as_nanos() as u64;
            SUBMIT_STEPS[k].fetch_add(ns, Ordering::Relaxed);
            if self.1 {
                SUBMIT_STEPS_RUN[k].fetch_add(ns, Ordering::Relaxed);
            }
            self.0 = Some(now);
        }
    }

    fn restart(&mut self) {
        if self.0.is_some() {
            self.0 = Some(std::time::Instant::now());
        }
    }
}

fn begin_stats_line() -> String {
    let s: Vec<u64> = BEGIN_STATS.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    let slot_wait = WAIT_NS[2].load(Ordering::Relaxed);
    let sub: Vec<u64> = SUBMIT_STATS.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    format!(
        "    batch begins: {} ({:.1} us each, slot waits included; {:.1} us without them; slots reset by the completion thread {:?}: found reset, waited for, taken back); submissions {} ({:.1} us each); draws' index checks against resident shadows: {} ({:.2} us each)\n",
        s[0],
        s[1] as f64 / s[0].max(1) as f64 / 1000.0,
        s[1].saturating_sub(slot_wait) as f64 / s[0].max(1) as f64 / 1000.0,
        OFFTHREAD_RESETS.iter().map(|c| c.load(Ordering::Relaxed)).collect::<Vec<u64>>(),
        sub[0],
        sub[1] as f64 / sub[0].max(1) as f64 / 1000.0,
        s[2],
        s[3] as f64 / s[2].max(1) as f64 / 1000.0
    ) + &{
        let steps: Vec<String> = SUBMIT_STEP_NAMES.iter().zip(SUBMIT_STEPS.iter()).map(|(n, v)| format!("{n} {:.1}", v.load(Ordering::Relaxed) as f64 / sub[0].max(1) as f64 / 1000.0)).collect();
        let runs = SUBMIT_STEPS_RUN[11].load(Ordering::Relaxed);
        let before: Vec<String> = SUBMIT_STEP_NAMES.iter().zip(SUBMIT_STEPS_RUN.iter()).map(|(n, v)| format!("{n} {:.1}", v.load(Ordering::Relaxed) as f64 / runs.max(1) as f64 / 1000.0)).collect();
        format!("    submission steps (us each): {}; of the {} before parallel runs: {}\n", steps.join(", "), runs, before.join(", "))
    }
}
/// The first GPU timestamp writes (batch, guest address), compared with the
/// title's clock once their batch completed (logged; a check of the clock
/// conversion).
static TS_CHECKS: Mutex<Vec<(u64, u64)>> = Mutex::new(Vec::new());
static TS_CHECKED: AtomicU64 = AtomicU64::new(0);

fn check_timestamps(done: u64) {
    if TS_CHECKED.load(Ordering::Relaxed) >= 8 {
        return;
    }
    let mut g = TS_CHECKS.lock();
    if g.is_empty() {
        return;
    }
    let now = super::cp::gpu_clock();
    g.retain(|&(seq, a)| {
        if seq > done {
            return true;
        }
        let v = unsafe { std::ptr::read_volatile(a as *const u64) };
        TS_CHECKED.fetch_add(1, Ordering::Relaxed);
        crate::info!("gpu: timestamp written by the GPU at {a:#x} (batch {seq}): {v:#x}, the title's clock {now:#x} at completion ({} us later)", (now as i64 - v as i64) / 100);
        false
    });
}
/// CP writes `tune gpu_cp_writes` left to the CPU path: [nothing to order
/// behind (performed now), (unused), memory the GPU cannot write here
/// (video-memory home, not importable, too large)].
pub static GPU_CP_REFUSED: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
/// Labels written GPU-side in batches not yet submitted (address & !7 ->
/// the owning queue's submit request): a command processor about to block on
/// one asks the owner to submit, which registers its promise.
static UNSUBMITTED_LABELS: Mutex<Option<HashMap<u64, std::sync::Arc<std::sync::atomic::AtomicBool>>>> = Mutex::new(None);
/// GPU-side label writes recorded while another queue blocked on the label
/// (`tune label_early`: their batches were submitted at the next packet).
pub static LABEL_EARLY: AtomicU64 = AtomicU64::new(0);

/// Asks the queue whose unsubmitted batch writes the label at `a` to submit
/// it (`Cp::wait`, before polling).
pub fn request_label_submit(a: u64) {
    if let Some(f) = UNSUBMITTED_LABELS.lock().as_ref().and_then(|m| m.get(&(a & !7))) {
        f.store(true, Ordering::SeqCst);
    }
}
/// Resident shadows: bindings that reused one, read-only bindings served
/// from one, merges, and merges forced before an in-place use.
static SHADOWS_REUSED: AtomicU64 = AtomicU64::new(0);
/// Written-buffer shadows and the shared copies (`tune shadow_share`):
/// [working copies shared after the merge, copies in from a shared copy, not
/// shared: written in place meanwhile, CPU-written meanwhile, no entry].
static SHADOWS_SHARED: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];
/// End-of-pipe writes without a full barrier (`tune eop_light`): [labels,
/// timestamps, waits that ordered the work after them, CP writes right after
/// front end 2's write-back with no barrier before them (`tune
/// fe2_cpw_light`)].
static EOP_LIGHT: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];
/// Dispatches storing deferred timestamps (`Exec::resolve_timestamps`), and
/// deferred timestamps a later one of the same resolve overwrote.
static TS_DISPATCHES: AtomicU64 = AtomicU64::new(0);
static TS_REPEATS: AtomicU64 = AtomicU64::new(0);
/// Copies in from shared copies counted for DS_SHADOW_SHARE_CHECK sampling.
static SHARE_CHECKED: AtomicU64 = AtomicU64::new(0);
static RESIDENT_READS: AtomicU64 = AtomicU64::new(0);
static MERGES: AtomicU64 = AtomicU64::new(0);
static EARLY_MERGES: AtomicU64 = AtomicU64::new(0);
/// The title's barriers before dispatches (a partial flush, cache action or
/// end-of-pipe event since the last command) by queue class (graphics CP,
/// other compute queues, queue 0x50, parallel workers) and what the
/// dispatch after it touches (`Exec::dispatch_hazard`: nothing the work
/// since the last full barrier wrote or read, not known, ranges written,
/// ranges read and now written, images or the global data share).
static TITLE_BARRIERS: [[AtomicU64; 5]; 4] = [const { [const { AtomicU64::new(0) }; 5] }; 4];
static TITLE_BARRIERS_LAST: Mutex<[[u64; 5]; 4]> = Mutex::new([[0; 5]; 4]);

/// The window's title barriers before dispatches per flip (frame windows;
/// counted while profiling).
pub(crate) fn title_barriers_clause(flips: u64) -> String {
    if !profiling() {
        return String::new();
    }
    let now: Vec<[u64; 5]> = TITLE_BARRIERS.iter().map(|c| std::array::from_fn(|k| c[k].load(Ordering::Relaxed))).collect();
    let mut last = TITLE_BARRIERS_LAST.lock();
    let f = flips.max(1) as f64;
    let mut parts = Vec::new();
    for (q, name) in ["graphics CP", "compute queues", "queue 0x50", "workers"].iter().enumerate() {
        let d: Vec<f64> = (0..5).map(|k| now[q][k].saturating_sub(last[q][k]) as f64 / f).collect();
        let all: f64 = d.iter().sum();
        if all > 0.0 {
            parts.push(format!("{name} {all:.0} (none {:.0}, unknown {:.0}, written {:.0}, read {:.0}, images {:.0})", d[0], d[1], d[2], d[3], d[4]));
        }
        last[q] = now[q];
    }
    if parts.is_empty() {
        return String::new();
    }
    format!("; title barriers before dispatches per flip, by what they guard: {}", parts.join(", "))
}
/// Shadow candidates bound in host memory because the slot's shadow ring was full.
static SHADOW_FULL: AtomicU64 = AtomicU64::new(0);
/// Read-only buffer snapshots (`Exec::snapshot`): taken, reused within a
/// batch, not taken for lack of space, bytes copied.
static SNAPS: AtomicU64 = AtomicU64::new(0);
static SNAPS_REUSED: AtomicU64 = AtomicU64::new(0);
static SNAPS_FULL: AtomicU64 = AtomicU64::new(0);
static SNAP_COPIED: AtomicU64 = AtomicU64::new(0);
/// Index buffers drawn from snapshots (draws, bytes; reuses included).
pub(super) static INDEX_SNAPS: AtomicU64 = AtomicU64::new(0);
pub(super) static INDEX_SNAP_BYTES: AtomicU64 = AtomicU64::new(0);
/// Video memory for snapshots per slot.
const SNAP_SEG: u64 = 16 << 20;
/// Device-local space per batch slot for copies made at the start of the
/// batch (`Exec::pre_copy`).
const PRE_SEG: u64 = 64 << 20;
static PRE_COPIES: AtomicU64 = AtomicU64::new(0);
static PRE_BYTES: AtomicU64 = AtomicU64::new(0);
/// vkCmdCopyBuffer calls recording batch-start copies (`tune pre_merge`).
static PRE_CALLS: AtomicU64 = AtomicU64::new(0);
/// PRE_COPIES, PRE_BYTES and PRE_CALLS at the last `profile reset`.
static PRE_SNAP: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// Starts a profile window of the batch-start copies (`profile reset`).
pub(super) fn reset_pre_window() {
    PRE_SNAP[0].store(PRE_COPIES.load(Ordering::Relaxed), Ordering::Relaxed);
    PRE_SNAP[1].store(PRE_BYTES.load(Ordering::Relaxed), Ordering::Relaxed);
    PRE_SNAP[2].store(PRE_CALLS.load(Ordering::Relaxed), Ordering::Relaxed);
    for (s, v) in XCOPY_SNAP.iter().zip(&XCOPY) {
        s.store(v.load(Ordering::Relaxed), Ordering::Relaxed);
    }
    XDRY_SNAP.store(XDRY.load(Ordering::Relaxed), Ordering::Relaxed);
    super::precache::reset_window();
}

/// The batch-start copies since the last `profile reset` (`profile report`).
pub(super) fn pre_window_line() -> String {
    let n = PRE_COPIES.load(Ordering::Relaxed).saturating_sub(PRE_SNAP[0].load(Ordering::Relaxed));
    let b = PRE_BYTES.load(Ordering::Relaxed).saturating_sub(PRE_SNAP[1].load(Ordering::Relaxed));
    let k = PRE_CALLS.load(Ordering::Relaxed).saturating_sub(PRE_SNAP[2].load(Ordering::Relaxed));
    let x: Vec<u64> = XCOPY.iter().zip(&XCOPY_SNAP).map(|(v, s)| v.load(Ordering::Relaxed).saturating_sub(s.load(Ordering::Relaxed))).collect();
    let xq = if ctx().copy.is_some() {
        format!(
            "    on the copy queue: {} copies ({} MiB) in {} batches; {} batches waited only for shared copies made there; {} batches put theirs into the prologue as the GPU had run dry (`xpre_dry`)\n",
            x[0],
            x[1] >> 20,
            x[2],
            x[3],
            XDRY.load(Ordering::Relaxed).saturating_sub(XDRY_SNAP.load(Ordering::Relaxed))
        )
    } else {
        String::new()
    };
    format!("    batch-start copies in the window: {n} ({} MiB, {k} copy calls); {}\n", b >> 20, super::precache::window_line()) + &xq
}
static PRE_REUSED: AtomicU64 = AtomicU64::new(0);
static PRE_FULL: AtomicU64 = AtomicU64::new(0);
/// Batch-start copies on the copy queue (`DS_COPY_QUEUE`): [copies, bytes,
/// batches with such copies, batches that waited only for such copies
/// earlier batches made], and their values at the last `profile reset`.
static XCOPY: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];
/// Batches whose copy-queue copies went into their prologue as the GPU had
/// run dry (`tune xpre_dry`), and the count at the last `profile reset`.
static XDRY: AtomicU64 = AtomicU64::new(0);
static XDRY_SNAP: AtomicU64 = AtomicU64::new(0);
/// Why `speculate` left a command to wait (diagnostics, since boot): stale
/// home, a data load this batch writes or not one mapping, a parallel
/// chunk's own writes, a stage that writes memory or images, a shader that
/// mispredicted, descriptor words GPU work writes (`tune spec_reads 2`).
static SPEC_REFUSED: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];
static XCOPY_SNAP: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];
/// GPU time of the batches' copy prologues (DS_PROFILE_GPU), ns.
static PRE_NS: AtomicU64 = AtomicU64::new(0);

/// `DS_GPU_SHADOW=0` binds written buffers in host memory directly (A/B
/// comparisons of the shadow path).
fn shadow_writes() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_GPU_SHADOW").ok().as_deref() != Some("0"))
}

/// Written buffers below this size are written in host memory directly
/// unless the shader uses atomics on them: their shadow copies and merge
/// (two DMA copies, a merge dispatch, barriers: ~25 us) cost more than the
/// few stores they receive (`DS_GPU_SHADOW_MIN=<bytes>`, default 64 KiB).
fn shadow_min() -> u64 {
    super::tune::shadow_min()
}

/// `DS_GPU_SHADOW_MERGE=0` copies whole shadows back (the earlier behaviour,
/// for A/B comparisons) instead of only the dwords a dispatch changed.
fn shadow_merge() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_GPU_SHADOW_MERGE").ok().as_deref() != Some("0"))
}

/// The shadow write-back kernel (`rdna::kernels::shadow_merge`).
struct MergePipe {
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    set_layout: vk::DescriptorSetLayout,
}

fn merge_pipe() -> Result<&'static MergePipe, String> {
    static PIPE: std::sync::OnceLock<Result<MergePipe, String>> = std::sync::OnceLock::new();
    PIPE.get_or_init(|| {
        let c = ctx();
        unsafe {
            let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..3)
                .map(|b| vk::DescriptorSetLayoutBinding::default().binding(b).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE))
                .collect();
            let set_layout = c.device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None).map_err(|e| format!("merge set layout: {e}"))?;
            let layouts = [set_layout];
            let ranges = [vk::PushConstantRange { stage_flags: vk::ShaderStageFlags::COMPUTE, offset: 0, size: 8 }];
            let layout = c.device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts).push_constant_ranges(&ranges), None).map_err(|e| format!("merge layout: {e}"))?;
            let code = rdna::kernels::shadow_merge();
            let module = c.device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None).map_err(|e| format!("merge module: {e}"))?;
            let stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(c"main");
            let ci = vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout);
            let pipeline = super::pcache::with_cache('i', 1, |cache| c.device.create_compute_pipelines(cache, &[ci], None)).map_err(|(_, e)| format!("merge pipeline: {e}"));
            c.device.destroy_shader_module(module, None);
            Ok(MergePipe { pipeline: pipeline?[0], layout, set_layout })
        }
    })
    .as_ref()
    .map_err(|e| e.clone())
}

/// The many-shadow write-back kernel (`rdna::kernels::shadow_merge_many`):
/// binding 0 the shadows, binding 1 its table; push constants {at, n}.
fn merge_many_pipe() -> Result<&'static MergePipe, String> {
    static PIPE: std::sync::OnceLock<Result<MergePipe, String>> = std::sync::OnceLock::new();
    PIPE.get_or_init(|| {
        let c = ctx();
        unsafe {
            let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..2)
                .map(|b| vk::DescriptorSetLayoutBinding::default().binding(b).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE))
                .collect();
            let set_layout = c.device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None).map_err(|e| format!("merge-many set layout: {e}"))?;
            let layouts = [set_layout];
            let ranges = [vk::PushConstantRange { stage_flags: vk::ShaderStageFlags::COMPUTE, offset: 0, size: 8 }];
            let layout = c.device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts).push_constant_ranges(&ranges), None).map_err(|e| format!("merge-many layout: {e}"))?;
            let code = rdna::kernels::shadow_merge_many();
            let module = c.device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None).map_err(|e| format!("merge-many module: {e}"))?;
            let stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(c"main");
            let ci = vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout);
            let pipeline = super::pcache::with_cache('i', 2, |cache| c.device.create_compute_pipelines(cache, &[ci], None)).map_err(|(_, e)| format!("merge-many pipeline: {e}"));
            c.device.destroy_shader_module(module, None);
            Ok(MergePipe { pipeline: pipeline?[0], layout, set_layout })
        }
    })
    .as_ref()
    .map_err(|e| e.clone())
}

/// Write-backs merged by one dispatch (`tune merge_many`): [dispatches, shadows].
static MERGE_MANY: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// Mirror mode's write-back kernel (`rdna::kernels::mirror_gather`).
fn gather_pipe() -> Result<&'static MergePipe, String> {
    static PIPE: std::sync::OnceLock<Result<MergePipe, String>> = std::sync::OnceLock::new();
    PIPE.get_or_init(|| {
        let c = ctx();
        unsafe {
            let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..2)
                .map(|b| vk::DescriptorSetLayoutBinding::default().binding(b).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE))
                .collect();
            let set_layout = c.device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None).map_err(|e| format!("gather set layout: {e}"))?;
            let layouts = [set_layout];
            let ranges = [vk::PushConstantRange { stage_flags: vk::ShaderStageFlags::COMPUTE, offset: 0, size: 16 }];
            let layout = c.device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts).push_constant_ranges(&ranges), None).map_err(|e| format!("gather layout: {e}"))?;
            let code = rdna::kernels::mirror_gather();
            let module = c.device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None).map_err(|e| format!("gather module: {e}"))?;
            let stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(c"main");
            let ci = vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout);
            let pipeline = super::pcache::with_cache('i', 6, |cache| c.device.create_compute_pipelines(cache, &[ci], None)).map_err(|(_, e)| format!("gather pipeline: {e}"));
            c.device.destroy_shader_module(module, None);
            Ok(MergePipe { pipeline: pipeline?[0], layout, set_layout })
        }
    })
    .as_ref()
    .map_err(|e| e.clone())
}

/// The textures a stage samples, with a coarse bound of their guest bytes
/// (16 per texel).
/// Appended to `out` (a reused vector: no allocation per command once it
/// has grown).
fn texture_ranges_into(p: &Prepared, out: &mut Vec<((u64, u64), super::desc::Tsharp)>) {
    let coarse = |t: &super::desc::Tsharp| ((t.base, texture_bytes(t)), *t);
    for r in p.resources.iter() {
        match r {
            Resolved::Image(t) => out.push(coarse(t)),
            Resolved::ImageTable(e) => out.extend(e.iter().flatten().map(coarse)),
            _ => {}
        }
    }
}
/// The guest bytes a texture covers for the command processors' texture
/// syncs (`Exec::sync_textures`): exact (`images::guest_size`, memoized per
/// thread by the T#'s layout fields; `tune tex_exact`, default), else at
/// most 16 bytes per texel. That bound reached 16-32 times past the end of
/// 8-bit and block-compressed textures: in boots where a GPU-written buffer
/// lay there, every particle dispatch reading such a texture took the slow
/// path (the image cache's lock, exact ranges) and queue 0x50 recorded its
/// indirect dispatches in 27.5 instead of 21.8 us (FINDINGS.md 2026-10-26).
/// The slow path's syncs use the exact size either way.
fn texture_bytes(t: &super::desc::Tsharp) -> u64 {
    type Key = (u16, u32, u32, u32, u32, u32, u32, u32, u32);
    thread_local! {
        static MEMO: std::cell::RefCell<HashMap<Key, u64>> = std::cell::RefCell::new(HashMap::new());
    }
    let rough = (t.width as u64 * t.height as u64 * t.depth.max(1) as u64 * 16).min(1 << 30);
    if !super::tune::tex_exact() {
        return rough;
    }
    let key = (t.format, t.width, t.height, t.depth, t.sw_mode, t.ty, t.max_mip, t.base_level, t.last_level);
    MEMO.with(|m| {
        if let Some(&n) = m.borrow().get(&key) {
            return n;
        }
        let n = super::images::guest_size(t).filter(|&n| n > 0).unwrap_or(rough);
        let mut m = m.borrow_mut();
        if m.len() >= 4096 {
            m.clear();
        }
        m.insert(key, n);
        n
    })
}

/// Most batch slots per queue (`tune slots`, default 4): one batch recording
/// while the batches of the others execute. A queue reuses a slot only after
/// its batch completed, so more slots let a command processor run further
/// ahead of the GPU. Slots are created when first used.
pub const MAX_SLOTS: usize = 8;

/// Largest read-only binding of memory the driver cannot import that is
/// copied by the CPU (`Exec::snapshot`) instead of bound as zeros.
const UNIMPORTABLE_SNAP_MAX: u64 = 64 << 10;
/// Bindings served that way.
static UNIMPORTABLE_SNAPS: AtomicU64 = AtomicU64::new(0);

/// Command-processor effect performed after GPU work completes.
pub type Action = Box<dyn FnOnce() + Send>;

/// A submitted batch (or action-only entry) awaiting completion.
struct Batch {
    queries: vk::QueryPool,
    /// Its batch and prologue timestamps (`Slot::bq`).
    bq: vk::QueryPool,
    seq: u64,
    /// Null for action-only entries.
    fence: vk::Fence,
    /// On the second queue (its fence is waited for, not the batch
    /// timeline).
    alt: bool,
    actions: Vec<Action>,
    checks: Vec<verify::Check>,
    shaders: Vec<u64>,
    submitted: std::time::Instant,
    name: String,
    /// Label writes the GPU performs within it: (address, value, bytes),
    /// made known to waits when it completes (`cp::gpu_label_written`).
    gpu_labels: Vec<(u64, u64, u8)>,
    /// Its other GPU operations (`ops_line`), named with a slow or lost batch.
    ops: [u64; OPS],
    /// Its barrier segments' marks (`seg_stamps`).
    segs: Vec<SegMark>,
    /// Its slot, reset here once it completed (`tune offthread_reset`).
    reset: Option<SlotReset>,
}

unsafe impl Send for Batch {}

/// Copies `n` timestamp queries from `first` into `buf` at `off` (8 bytes
/// each), once the work before them completed: by a full barrier (`tune
/// ts_nowait`; a copy that waits for availability is the only operation of
/// a batch that can stall the GPU indefinitely) or by the copy's own wait.
unsafe fn copy_timestamps(cmd: vk::CommandBuffer, pool: vk::QueryPool, first: u32, n: u32, buf: vk::Buffer, off: u64) {
    let c = ctx();
    unsafe {
        if super::tune::ts_nowait() {
            if super::tune::skip_barriers() & 8 == 0 {
                global_barrier(cmd);
            }
            c.device.cmd_copy_query_pool_results(cmd, pool, first, n, buf, off, 8, vk::QueryResultFlags::TYPE_64);
        } else {
            c.device.cmd_copy_query_pool_results(cmd, pool, first, n, buf, off, 8, vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT);
        }
    }
}

/// A batch's GPU operations besides its draws and dispatches, counted while
/// it is recorded (`Exec::ops`).
const OP_FILLS: usize = 0;
const OP_FILL_BYTES: usize = 1;
const OP_CP_WRITES: usize = 2;
const OP_TIMESTAMPS: usize = 3;
const OP_MERGES: usize = 4;
const OP_SHADOWS_IN: usize = 5;
const OP_SHADOW_BYTES: usize = 6;
const OP_PRE: usize = 7;
const OP_PRE_BYTES: usize = 8;
const OP_WRITE_BACK_BYTES: usize = 9;
const OP_FAST_CLEARS: usize = 10;
/// Rendering instances ended (`seg_stamps` names segments by it).
pub(super) const OP_PASS_ENDS: usize = 11;
/// Front end 2's write-backs by site (diagnostics): [before a CP write the
/// GPU performs, at a batch's close, of those with no label write promised
/// and no completion action, before CP writes of labels a CP of ours waits
/// on, left to the batch's close (`tune fe2_cpw_close`, `fe2_cpw_waited`)].
pub static SCATTER_SITES: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];
/// Front end 2's write-backs (`Exec::fe2_scatter`).
const OP_FE2_SCATTERS: usize = 12;
const OPS: usize = 13;

fn ops_line(ops: &[u64; OPS]) -> String {
    let kib = |b: u64| b.div_ceil(1024);
    let mut parts = Vec::new();
    if ops[OP_FILLS] > 0 {
        parts.push(format!("{} fills ({} KiB)", ops[OP_FILLS], kib(ops[OP_FILL_BYTES])));
    }
    if ops[OP_CP_WRITES] > 0 {
        parts.push(format!("{} CP writes", ops[OP_CP_WRITES]));
    }
    if ops[OP_TIMESTAMPS] > 0 {
        parts.push(format!("{} timestamps", ops[OP_TIMESTAMPS]));
    }
    if ops[OP_MERGES] > 0 {
        parts.push(format!("{} shadow merges", ops[OP_MERGES]));
    }
    if ops[OP_SHADOWS_IN] > 0 {
        parts.push(format!("{} shadow copies in ({} KiB)", ops[OP_SHADOWS_IN], kib(ops[OP_SHADOW_BYTES])));
    }
    if ops[OP_PRE] > 0 {
        parts.push(format!("{} batch-start copies ({} KiB)", ops[OP_PRE], kib(ops[OP_PRE_BYTES])));
    }
    if ops[OP_WRITE_BACK_BYTES] > 0 {
        parts.push(format!("home write-backs {} KiB", kib(ops[OP_WRITE_BACK_BYTES])));
    }
    if ops[OP_FAST_CLEARS] > 0 {
        parts.push(format!("{} fast clears", ops[OP_FAST_CLEARS]));
    }
    if parts.is_empty() {
        "no other operations".to_string()
    } else {
        parts.join(", ")
    }
}

/// Batches in submission order (sequence numbers increase).
static BATCHES: Mutex<std::collections::VecDeque<Batch>> = Mutex::new(std::collections::VecDeque::new());
static BATCHES_CV: parking_lot::Condvar = parking_lot::Condvar::new();
static NEXT_SEQ: AtomicU64 = AtomicU64::new(1);
/// Highest completed sequence number (all earlier ones completed too).
static DONE: Mutex<u64> = Mutex::new(0);
/// DONE without its lock (every CP read check asks for it).
static DONE_ATOMIC: AtomicU64 = AtomicU64::new(0);
/// When the completion thread finished the last batch (profiling: GPU
/// starvation, see `STARVED`).
static LAST_DONE_AT: Mutex<Option<std::time::Instant>> = Mutex::new(None);
/// Batches submitted per (queue, reason) while profiling.
static SUBMIT_WHY: Mutex<Option<HashMap<(String, &'static str), u64>>> = Mutex::new(None);
/// Submissions that found every earlier batch complete (the GPU idle), per
/// queue: [submissions, idle ns before them, ns spent recording them].
static STARVED: Mutex<Option<HashMap<String, [u64; 3]>>> = Mutex::new(None);
static DONE_CV: parking_lot::Condvar = parking_lot::Condvar::new();

/// Whether every batch handed over so far completed: the GPU ran out of our
/// work (`tune xpre_dry`).
fn gpu_ran_dry() -> bool {
    done_seq() + 1 >= NEXT_SEQ.load(Ordering::Acquire)
}

fn done_seq() -> u64 {
    DONE_ATOMIC.load(Ordering::Acquire)
}

/// Highest completed batch sequence (all earlier ones completed too).
pub fn completed_seq() -> u64 {
    done_seq()
}

/// The sequence the next batch handed over will get (earlier ones were).
pub fn next_seq() -> u64 {
    NEXT_SEQ.load(Ordering::Acquire)
}

/// Guest ranges written by unfinished batches, by sequence number
/// (`ranges::SeqRanges`: per batch a coalesced union, so the title's
/// thousands of writes per batch to the same buffers stay a few entries and a
/// lookup costs one ordered search per unfinished batch); `latest_seq` lets
/// lookups return at once when every noted batch has completed.
struct RangeLog {
    log: parking_lot::RwLock<super::ranges::SeqRanges>,
    latest_seq: AtomicU64,
    /// Bumped after every note (`Exec`'s cache of clean pages holds while it
    /// stays the same; a reader takes it before looking).
    notes: AtomicU64,
    /// Per bucket of 64 KiB guest pages (hashed), the latest batch that
    /// writes into one of them: a lookup whose pages' buckets all hold
    /// completed batches is answered without the log's lock (the command
    /// processors check every draw's descriptor reads; ~1.75 us per draw
    /// with the lock, run 20261011-133502). Never lowered: completion
    /// clears a bucket implicitly (its batch <= done).
    pages: [AtomicU64; PAGE_BUCKETS],
    /// The same per 4 MiB, for ranges of COARSE_FROM pages or more (marked
    /// here instead of page by page: the culls' 112 MB home took ~1,800
    /// atomic updates per batch that wrote it).
    coarse: [AtomicU64; COARSE_BUCKETS],
    /// The same for those large writes alone (`tune log_big`: what a small
    /// read needs beside its page buckets).
    big: [AtomicU64; COARSE_BUCKETS],
    /// Lookups (`latest`): [all, nothing unfinished, no bucket overlapping,
    /// answered under the lock, of those with a batch found].
    stats: [AtomicU64; 5],
}

const PAGE_BUCKETS: usize = 1 << 16;
const PAGE_SHIFT: u32 = 16;
const COARSE_BUCKETS: usize = 1 << 12;
const COARSE_SHIFT: u32 = 22;
/// Ranges of this many 64 KiB pages or more are marked per 4 MiB.
const COARSE_FROM: u64 = 64;

fn page_bucket(page: u64) -> usize {
    (page.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> (64 - 16)) as usize
}

fn coarse_bucket(page: u64) -> usize {
    (page.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> (64 - 12)) as usize
}

impl RangeLog {
    const fn new() -> RangeLog {
        RangeLog {
            log: parking_lot::RwLock::new(super::ranges::SeqRanges::new()),
            latest_seq: AtomicU64::new(0),
            notes: AtomicU64::new(0),
            pages: [const { AtomicU64::new(0) }; PAGE_BUCKETS],
            coarse: [const { AtomicU64::new(0) }; COARSE_BUCKETS],
            big: [const { AtomicU64::new(0) }; COARSE_BUCKETS],
            stats: [const { AtomicU64::new(0) }; 5],
        }
    }

    /// Marks the page buckets of [a, a+n) as written by batch `seq`.
    fn mark_pages(&self, seq: u64, a: u64, n: u64) {
        if n == 0 {
            return;
        }
        let (first, last) = (a >> PAGE_SHIFT, (a + n - 1) >> PAGE_SHIFT);
        if last - first >= COARSE_FROM {
            let (first, last) = (a >> COARSE_SHIFT, (a + n - 1) >> COARSE_SHIFT);
            if last - first >= COARSE_BUCKETS as u64 {
                for (b, g) in self.coarse.iter().zip(self.big.iter()) {
                    b.fetch_max(seq, Ordering::SeqCst);
                    g.fetch_max(seq, Ordering::SeqCst);
                }
                return;
            }
            for p in first..=last {
                self.coarse[coarse_bucket(p)].fetch_max(seq, Ordering::SeqCst);
                self.big[coarse_bucket(p)].fetch_max(seq, Ordering::SeqCst);
            }
            return;
        }
        for p in first..=last {
            self.pages[page_bucket(p)].fetch_max(seq, Ordering::SeqCst);
        }
        // Its 4 MiB blocks too (at most two): what large reads look at.
        for p in (a >> COARSE_SHIFT)..=((a + n - 1) >> COARSE_SHIFT) {
            self.coarse[coarse_bucket(p)].fetch_max(seq, Ordering::SeqCst);
        }
    }

    /// Whether some unfinished batch may write into one of `reads` (false:
    /// certainly none).
    fn may_overlap(&self, reads: &[(u64, u64)], done: u64) -> bool {
        let big = super::tune::log_big();
        reads.iter().any(|&(a, n)| {
            if n == 0 {
                return false;
            }
            let (first, last) = (a >> PAGE_SHIFT, (a + n - 1) >> PAGE_SHIFT);
            let coarse = |b: &[AtomicU64; COARSE_BUCKETS]| {
                let (c0, c1) = (a >> COARSE_SHIFT, (a + n - 1) >> COARSE_SHIFT);
                c1 - c0 >= COARSE_BUCKETS as u64 || (c0..=c1).any(|p| b[coarse_bucket(p)].load(Ordering::SeqCst) > done)
            };
            // Large reads go by the coarse buckets alone (every write marks
            // its own: until 2026-10-28 reads of 4 MiB or more always took
            // the log's lock, front end 2's ~113 MB pools at every binding).
            if last - first >= 64 {
                return coarse(&self.coarse);
            }
            // Small ones by their pages and the large writes' 4 MiB
            // buckets (a small write marks its pages: `tune log_big`).
            (first..=last).any(|p| self.pages[page_bucket(p)].load(Ordering::SeqCst) > done) || coarse(if big { &self.big } else { &self.coarse })
        })
    }

    fn note(&self, seq: u64, w: (u64, u64)) {
        if w.1 == 0 {
            return;
        }
        self.mark_pages(seq, w.0, w.1);
        self.latest_seq.fetch_max(seq, Ordering::SeqCst);
        let done = done_seq();
        let mut g = super::lockstat::WRITE_LOGS.write(&self.log);
        g.prune(done);
        g.insert(seq, w.0, w.1);
        drop(g);
        self.notes.fetch_add(1, Ordering::AcqRel);
    }

    /// Notes the ranges written by batch `seq` (one lock).
    fn note_ranges(&self, seq: u64, writes: &super::ranges::Ranges) {
        if writes.is_empty() {
            return;
        }
        for (a, n) in writes.iter() {
            self.mark_pages(seq, a, n);
        }
        self.latest_seq.fetch_max(seq, Ordering::SeqCst);
        let done = done_seq();
        let mut g = super::lockstat::WRITE_LOGS.write(&self.log);
        g.prune(done);
        g.insert_all(seq, writes);
        drop(g);
        self.notes.fetch_add(1, Ordering::AcqRel);
    }

    /// The latest unfinished batch that writes into `reads`.
    fn latest(&self, reads: &[(u64, u64)]) -> Option<u64> {
        let done = done_seq();
        // (Counted with `DS_LOG_STATS=1`: shared counters cost the lookups.)
        let count = |k: usize| {
            if log_stats_on() {
                self.stats[k].fetch_add(1, Ordering::Relaxed);
            }
        };
        count(0);
        if self.latest_seq.load(Ordering::SeqCst) <= done {
            count(1);
            return None;
        }
        if !self.may_overlap(reads, done) {
            count(2);
            return None;
        }
        count(3);
        let r = if super::tune::shared_log_reads() {
            super::lockstat::WRITE_LOGS.read(&self.log).latest(reads, done)
        } else {
            super::lockstat::WRITE_LOGS.write(&self.log).latest(reads, done)
        };
        if r.is_some() {
            count(4);
        }
        r
    }

    fn stats_text(&self) -> String {
        let v: Vec<u64> = self.stats.iter().map(|c| c.load(Ordering::Relaxed)).collect();
        format!("{} lookups ({} with nothing unfinished, {} with no bucket overlapping, {} under the lock, {} of them finding a batch)", v[0], v[1], v[2], v[3], v[4])
    }
}

#[cfg(test)]
mod range_log_tests {
    use super::*;

    static LOG: RangeLog = RangeLog::new();

    /// The write log's buckets never pass a read that overlaps a pending
    /// write, small or large (`tune log_big`: small reads go by their pages
    /// and the large writes' 4 MiB buckets), and large reads go by every
    /// write's 4 MiB buckets.
    #[test]
    fn buckets_find_every_overlapping_write() {
        let mut x = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let base = 0x10_0000_0000u64;
        for round in 0..300u64 {
            // This round's writes are the pending ones (earlier rounds' are
            // done).
            let done = round * 100 + 99;
            let mut writes = Vec::new();
            for k in 0..6 {
                let a = (base + next() % (1 << 31)) & !3;
                let n = if next() % 4 == 0 { next() % (96 << 20) + 1 } else { next() % (512 << 10) + 1 };
                LOG.mark_pages(done + 1 + k, a, n);
                writes.push((a, n));
            }
            for _ in 0..300 {
                // Reads near a write as often as not.
                let (wa, wn) = writes[(next() % writes.len() as u64) as usize];
                let a = if next() % 2 == 0 { (wa + next() % (wn + (8 << 20))).saturating_sub(4 << 20) & !3 } else { (base + next() % (1 << 31)) & !3 };
                let n = if next() % 8 == 0 { next() % (96 << 20) + 1 } else { next() % (128 << 10) + 1 };
                let overlaps = writes.iter().any(|&(wa, wn)| wa < a + n && a < wa + wn);
                if overlaps {
                    assert!(LOG.may_overlap(&[(a, n)], done), "round {round}: read {a:#x}+{n:#x} passed a pending write");
                }
            }
        }
    }
}

/// `DS_LOG_STATS=1` (diagnosis): the write logs' lookups counted by how
/// they were answered.
fn log_stats_on() -> bool {
    static W: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *W.get_or_init(|| std::env::var("DS_LOG_STATS").ok().as_deref() == Some("1"))
}

/// The write logs' lookups by how they were answered (`profile report`;
/// counted since the start with `DS_LOG_STATS=1`).
pub fn write_logs_line() -> String {
    if !log_stats_on() {
        return String::new();
    }
    format!("    write logs: in-flight batches' writes {}; deferred CP writes {}\n", INFLIGHT.stats_text(), CP_WRITES.stats_text())
}

/// What submitted, unfinished batches of every queue write (shader buffer
/// writes and deferred CP writes): a queue that passed a label wait on a
/// promise (`Cp::wait`) runs ahead of the batch writing the label and must
/// not read on the CPU what such batches write.
static INFLIGHT: RangeLog = RangeLog::new();
/// Deferred CP writes that later GPU work may read (performed by the
/// completion thread once their batch finished): a batch binding such a
/// range is submitted only after that.
static CP_WRITES: RangeLog = RangeLog::new();

/// Held executors' reads of their own held writes logged (diagnostics).
static HELD_OWN_LOGGED: AtomicU64 = AtomicU64::new(0);
/// Stretches past label waits stopped because a preparation read what the
/// stretch's own held batches write, by (the reading stages, the first range
/// read that overlaps, which held writes: 0 the recording batch's, 1 a closed
/// batch's, 2 the main command processor's): count (`held_own_line`).
static HELD_OWN: Mutex<Option<HashMap<(u64, u64, u64, u64, u8), u64>>> = Mutex::new(None);

/// One status line: the commonest reads of held work's own writes that
/// stopped a stretch past a label wait ("" before any).
fn held_own_line() -> String {
    let g = HELD_OWN.lock();
    let Some(m) = g.as_ref() else { return String::new() };
    let mut v: Vec<(&(u64, u64, u64, u64, u8), &u64)> = m.iter().collect();
    v.sort_by(|a, b| b.1.cmp(a.1));
    let total: u64 = m.values().sum();
    let top: Vec<String> = v
        .iter()
        .take(10)
        .map(|((s0, s1, a, n, by), c)| format!("{c}x stages {s0:#x} {s1:#x} read {a:#x}+{n:#x} ({})", ["the recording batch's writes", "a closed batch's", "the main CP's"][*by as usize]))
        .collect();
    format!("    stretches past label waits stopped by reads of their own held writes: {total} ({} kinds): {}
", m.len(), top.join("; "))
}

/// What the batches submitted and the CP writes recorded while the
/// graphics CP records past a compute queue's label wait write
/// (`Cp::past_wait_start`): its CPU reads past the wait are checked
/// against it when it settles. None: no such stretch.
static PAST_JOURNAL: Mutex<Option<super::ranges::Ranges>> = Mutex::new(None);
static PAST_JOURNAL_ON: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Starts the journal (a stretch past a label wait begins).
pub(super) fn journal_begin() {
    *PAST_JOURNAL.lock() = Some(super::ranges::Ranges::new());
    PAST_JOURNAL_WHO.lock().clear();
    PAST_JOURNAL_ON.store(true, Ordering::SeqCst);
}

/// Ends the journal: what was written since `journal_begin`.
pub(super) fn journal_end() -> super::ranges::Ranges {
    PAST_JOURNAL_ON.store(false, Ordering::SeqCst);
    PAST_JOURNAL.lock().take().unwrap_or_default()
}

fn journal_note(writes: &super::ranges::Ranges, who: &str) {
    if PAST_JOURNAL_ON.load(Ordering::Acquire) && !writes.is_empty() {
        if let Some(j) = PAST_JOURNAL.lock().as_mut() {
            j.extend(writes);
        }
        if trace_past_drops() {
            let mut g = PAST_JOURNAL_WHO.lock();
            for (a, n) in writes.iter().take(256) {
                if g.len() < 4096 {
                    g.push((a, n, who.to_string()));
                }
            }
        }
    }
}

fn journal_note_one(w: (u64, u64), who: &str) {
    if PAST_JOURNAL_ON.load(Ordering::Acquire) && w.1 > 0 {
        if let Some(j) = PAST_JOURNAL.lock().as_mut() {
            j.insert(w.0, w.1);
        }
        if trace_past_drops() {
            let mut g = PAST_JOURNAL_WHO.lock();
            if g.len() < 4096 {
                g.push((w.0, w.1, format!("{who} (CP write)")));
            }
        }
    }
}

/// Kinds of the CPU reads of held recording (`Exec::note_held_reads`).
pub const HELD_READ_KINDS: [&str; 6] = ["shared copy", "texture", "binding", "CP label poll", "CP read", "register pairs"];

/// `DS_TRACE_PAST_DROPS=<n>`: the first n (1: 60) stretches past a label
/// wait dropped because other queues wrote what they read on the CPU are
/// logged with that read's kind and range and who wrote it
/// (`Cp::past_wait_settle`). 0: none.
pub(super) fn trace_past_drops_max() -> u64 {
    static N: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *N.get_or_init(|| match std::env::var("DS_TRACE_PAST_DROPS").ok().and_then(|v| v.trim().parse::<u64>().ok()) {
        Some(1) => 60,
        Some(n) => n,
        None => 0,
    })
}

pub(super) fn trace_past_drops() -> bool {
    trace_past_drops_max() != 0
}

/// The journal's writes with their writers (`DS_TRACE_PAST_DROPS`).
static PAST_JOURNAL_WHO: Mutex<Vec<(u64, u64, String)>> = Mutex::new(Vec::new());

/// The writers of the journal's writes into [a, a+n) (`DS_TRACE_PAST_DROPS`).
pub(super) fn journal_writers(a: u64, n: u64) -> Vec<String> {
    let g = PAST_JOURNAL_WHO.lock();
    g.iter().filter(|w| w.0 < a + n && a < w.0 + w.1).take(4).map(|w| format!("{} [{:#x}+{:#x}]", w.2, w.0, w.1)).collect()
}

/// CP reads that waited for another queue's batch (diagnostics).
pub static FOREIGN_WAITS: AtomicU64 = AtomicU64::new(0);
/// Reads by parallel workers of what the main command processor's batch,
/// unsubmitted when the run was planned, writes (`tune par_main_writes`).
pub static MAIN_WRITE_READS: AtomicU64 = AtomicU64::new(0);
/// Of those (`tune par_main_writes 3`): [reads that found the batch not yet
/// submitted and waited for it, ns waited].
pub static MAIN_SUBMIT_WAITS: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// A deferred CP copy (performed with its batch's actions) reads `src`:
/// later GPU work binding it (it may overwrite it) is submitted after the
/// copy ran.
pub fn note_cp_source(seq: u64, src: (u64, u64)) {
    CP_WRITES.note(seq, src);
}

/// A stage used as prepared although submitted GPU work (batch `seq`) was
/// still writing what its preparation read (`Exec::speculate`): the words
/// it read are compared with memory once that batch completed.
struct SpecCheck {
    seq: u64,
    shader: u64,
    reads: Vec<(u64, u64)>,
    words: Vec<u32>,
}

static SPEC_CHECKS: Mutex<Vec<SpecCheck>> = Mutex::new(Vec::new());
static SPEC_PENDING: AtomicU64 = AtomicU64::new(0);
/// Shaders whose speculation was wrong, and the frame: they wait again for
/// NO_SPEC_FRAMES frames (mispredictions came at level loads, when the
/// words change; the same shaders' words held in every wait since).
static NO_SPEC: Mutex<Option<HashMap<u64, u64>>> = Mutex::new(None);
const NO_SPEC_FRAMES: u64 = 600;

/// Whether shader `addr` mispredicted lately (`NO_SPEC`).
fn no_spec(addr: u64) -> bool {
    let mut g = NO_SPEC.lock();
    let Some(m) = g.as_mut() else { return false };
    match m.get(&addr) {
        Some(&f) if f + NO_SPEC_FRAMES > super::FRAMES.load(Ordering::Relaxed) => true,
        Some(_) => {
            m.remove(&addr);
            false
        }
        None => false,
    }
}
/// [stages used as prepared (checked), checks that held, checks that
/// failed, commands that did not wait, data loads copied by the prologue].
static SPEC_STATS: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];

/// A data-only load the batch prologue copies from guest memory into a
/// stage's SGPR data, after the submitted work writing it (`Exec::
/// speculate`): stage index, dword index in the SGPR data, the imported
/// guest buffer and offset, bytes.
pub(super) struct SgprPatch {
    stage: usize,
    sgpr: u32,
    src: vk::Buffer,
    src_off: u64,
    bytes: u64,
}

/// Speculations on words the batch being recorded writes (`Exec::
/// speculate`; checked once it completed).
static SPEC_OWN: AtomicU64 = AtomicU64::new(0);

/// Whether every dword of `reads` lies in the descriptor (read from memory)
/// of a resource stage `p` only reads: a wrong one of those reads wrong data
/// at worst; pointer words and the descriptors of what it writes are not.
fn read_only_descriptors(p: &Prepared, reads: &[(u64, u64)]) -> bool {
    use rdna::analysis::ResKind;
    let res = &p.tr.plan.resources;
    let mut ro: Vec<(u64, u64)> = Vec::new();
    for (r, a) in res.iter().zip(p.res_addrs.iter()) {
        let Some(a) = *a else { continue };
        let n = 4 * r.dwords.len() as u64;
        if r.written || matches!(r.kind, ResKind::ImageTable | ResKind::Global) {
            if reads.iter().any(|&(b, m)| b < a + n && a < b + m) {
                return false;
            }
        } else {
            ro.push((a, n));
        }
    }
    reads.iter().all(|&(b, m)| {
        let mut d = b;
        while d < b + m {
            if !ro.iter().any(|&(a, n)| d >= a && d + 4 <= a + n) {
                return false;
            }
            d += 4;
        }
        true
    })
}

/// Read-only data a command's shaders read on the GPU, as guest memory held
/// it when the command was recorded (`tune race_checks`): compared once the
/// batch completed; a difference means the data changed while the batch was
/// in flight (the GPU reads it when the batch executes).
struct RaceCheck {
    seq: u64,
    shader: u64,
    base: u64,
    /// The words (ranges up to 4 KiB), else empty and `hash` of `len` bytes.
    words: Vec<u32>,
    len: u64,
    hash: u64,
    /// The executor that recorded the command.
    queue: std::sync::Arc<str>,
    flip: u64,
    /// When recorded: 1 a deferred CP write pending (the batch is submitted
    /// after it), 2 unfinished GPU work writing it, 4 a buffer binding (else
    /// a data-only load); 16 << placement (0 in place, 1 VRAM mirror or
    /// resident shadow, 2 batch-start copy, 3 video-memory home).
    why: u8,
    /// CPU-performed CP writes so far when recorded and when submitted
    /// (`cp::CPW_SEQ`).
    cpw: u64,
    cpw_submit: u64,
}

static RACE_CHECKS: Mutex<Vec<RaceCheck>> = Mutex::new(Vec::new());
/// [ranges checked, ranges changed].
static RACE_STATS: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];

/// What changed a checked range (`check_races`); the first five are not
/// explained by the order of the packet stream.
const RACE_CAUSES: [&str; 9] = [
    "a CP write the CPU performed after the command was recorded and before its batch was submitted",
    "a CP write the CPU performed after the batch was submitted",
    "a deferred CP write after an earlier batch, not pending when the command was recorded",
    "GPU work of an earlier batch not in flight when the command was recorded",
    "no CP or GPU write found (the title's CPU, or an untracked write)",
    "a deferred CP write pending when recorded, or after this batch (expected)",
    "GPU work of this or a later batch (expected)",
    "GPU work in flight when recorded (expected)",
    "unreadable",
];
static RACE_CAUSE_COUNTS: [AtomicU64; 9] = [const { AtomicU64::new(0) }; 9];
/// Per (cause, shader, why, queue): count, an example range and change.
static RACE_SUMMARY: Mutex<Option<HashMap<(usize, u64, u8, std::sync::Arc<str>), (u64, u64, u64, String)>>> = Mutex::new(None);

/// A hash of guest [base, base + len) (race checks of ranges over 4 KiB).
fn race_hash(base: u64, len: u64) -> u64 {
    let mut h = super::fx::FxHasher::default();
    let mut a = base;
    while a + 8 <= base + len {
        std::hash::Hasher::write_u64(&mut h, unsafe { std::ptr::read_unaligned(a as *const u64) });
        a += 8;
    }
    while a < base + len {
        std::hash::Hasher::write_u32(&mut h, unsafe { std::ptr::read_unaligned(a as *const u32) });
        a += 4;
    }
    std::hash::Hasher::finish(&h)
}

/// Writes of batches not yet submitted, by executor (`tune race_checks`):
/// all batches go to one device queue in submission order, so a binding
/// that reads another queue's such writes runs before them.
static OPEN_WRITES: Mutex<Option<HashMap<u64, (std::sync::Arc<str>, super::ranges::Ranges)>>> = Mutex::new(None);
/// Bindings that read another queue's unsubmitted writes, by (reading queue,
/// writing queue, shader): count and an example.
static OPEN_READS: Mutex<Option<HashMap<(std::sync::Arc<str>, std::sync::Arc<str>, u64), (u64, String)>>> = Mutex::new(None);
static OPEN_READ_COUNT: AtomicU64 = AtomicU64::new(0);
static OPEN_READS_LOGGED: AtomicU64 = AtomicU64::new(0);
static RACE_IDS: AtomicU64 = AtomicU64::new(1);

/// A queue's name without a parallel worker's suffix (the graphics queue's
/// workers record chunks of its runs).
fn queue_family(name: &str) -> &str {
    name.split("-par").next().unwrap_or(name)
}

/// GPU writes of submitted batches, by 4 KiB page (`tune race_checks`):
/// (batch, start, bytes); ranges over 1 MiB in a list of their own.
#[derive(Default)]
struct GpuWriteLog {
    pages: HashMap<u64, Vec<(u64, u64, u64)>>,
    big: std::collections::VecDeque<(u64, u64, u64)>,
}

static RACE_GPU_WRITES: Mutex<Option<GpuWriteLog>> = Mutex::new(None);

fn note_race_writes(seq: u64, writes: &super::ranges::Ranges) {
    if writes.is_empty() {
        return;
    }
    let mut g = RACE_GPU_WRITES.lock();
    let l = g.get_or_insert_with(Default::default);
    for (a, n) in writes.iter() {
        if n > 1 << 20 {
            if l.big.len() >= 4096 {
                l.big.pop_front();
            }
            l.big.push_back((seq, a, n));
            continue;
        }
        for p in a >> 12..=(a + n.max(1) - 1) >> 12 {
            let v = l.pages.entry(p).or_default();
            if v.len() >= 64 {
                v.remove(0);
            }
            v.push((seq, a, n));
        }
    }
}

/// The latest batch whose GPU writes overlap [a, a + n).
fn race_gpu_writer_range(a: u64, n: u64) -> Option<u64> {
    let g = RACE_GPU_WRITES.lock();
    let l = g.as_ref()?;
    let hit = |&&(_, b, m): &&(u64, u64, u64)| b < a + n && a < b + m;
    let mut best = l.big.iter().filter(hit).map(|e| e.0).max();
    for page in a >> 12..=(a + n.max(1) - 1) >> 12 {
        if let Some(s) = l.pages.get(&page).and_then(|v| v.iter().filter(hit).map(|e| e.0).max()) {
            best = best.max(Some(s));
        }
    }
    best
}

/// The latest batch whose GPU writes cover `addr`.
fn race_gpu_writer(addr: u64) -> Option<u64> {
    let g = RACE_GPU_WRITES.lock();
    let l = g.as_ref()?;
    let hit = |&&(_, a, n): &&(u64, u64, u64)| a <= addr && addr < a + n;
    let p = l.pages.get(&(addr >> 12)).and_then(|v| v.iter().filter(hit).map(|e| e.0).max());
    let b = l.big.iter().filter(hit).map(|e| e.0).max();
    p.max(b)
}

fn check_races(done: u64) {
    let due: Vec<RaceCheck> = {
        let mut g = RACE_CHECKS.lock();
        if g.is_empty() {
            return;
        }
        let (due, keep): (Vec<RaceCheck>, Vec<RaceCheck>) = std::mem::take(&mut *g).into_iter().partition(|c| c.seq <= done);
        *g = keep;
        due
    };
    for c in due {
        RACE_STATS[0].fetch_add(1, Ordering::Relaxed);
        let n = c.len;
        if !super::cp::guest_readable(c.base, n) {
            continue;
        }
        // Small ranges: the first word that changed; larger ones: by hash.
        let (j, was, now, cpw, gpu) = if c.words.is_empty() {
            if race_hash(c.base, n) == c.hash {
                continue;
            }
            (usize::MAX, 0, 0, super::cp::cpw_since_range(c.base, n, c.cpw), race_gpu_writer_range(c.base, n))
        } else {
            let changed = (0..c.words.len()).find(|&j| unsafe { std::ptr::read_volatile((c.base + 4 * j as u64) as *const u32) } != c.words[j]);
            let Some(j) = changed else { continue };
            let w = c.base + 4 * j as u64;
            (j, c.words[j], unsafe { std::ptr::read_volatile(w as *const u32) }, super::cp::cpw_since(w, c.cpw), race_gpu_writer(w))
        };
        RACE_STATS[1].fetch_add(1, Ordering::Relaxed);
        let what = if j == usize::MAX { "the hash".to_string() } else { format!("dword {j} {was:#x} -> {now:#x}") };
        let (cause, by) = match (&cpw, gpu) {
            (Some((s, 0, q, at, op)), _) => (if *s <= c.cpw_submit { 0 } else { 1 }, format!("{q} opcode {op:#x} packet {at:#x}")),
            (Some((_, after, q, at, op)), _) => (if *after >= c.seq || c.why & 1 != 0 { 5 } else { 2 }, format!("{q} opcode {op:#x} packet {at:#x} after batch {after}")),
            (None, Some(s)) if s >= c.seq => (6, format!("batch {s}")),
            (None, Some(s)) if c.why & 2 != 0 => (7, format!("batch {s}")),
            (None, Some(s)) => (3, format!("batch {s}")),
            (None, None) => (4, String::new()),
        };
        RACE_CAUSE_COUNTS[cause].fetch_add(1, Ordering::Relaxed);
        let first = {
            let mut g = RACE_SUMMARY.lock();
            let m = g.get_or_insert_with(HashMap::new);
            let e = m.entry((cause, c.shader, c.why, c.queue.clone())).or_insert_with(|| (0, c.base, n, format!("{what} by {by}")));
            e.0 += 1;
            e.0 == 1
        };
        if cause < 5 && first {
            crate::info!(
                "gpu: race check: {} shader {:#x} read {:#x}+{n:#x} on the GPU (why {:#x}); {what} between recording (flip {}) and the end of batch {} (flip {}): {} ({by})",
                c.queue,
                c.shader,
                c.base,
                c.why,
                c.flip,
                c.seq,
                super::FRAMES.load(Ordering::Relaxed),
                RACE_CAUSES[cause]
            );
        }
    }
}

/// The race checks' status line ("" when off).
pub fn race_line() -> String {
    let s: Vec<u64> = RACE_STATS.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    if s[0] == 0 {
        return String::new();
    }
    let mut out = format!("    race checks: {} read-only ranges read on the GPU checked after their batch, {} changed meanwhile:", s[0], s[1]);
    for (k, name) in RACE_CAUSES.iter().enumerate() {
        let n = RACE_CAUSE_COUNTS[k].load(Ordering::Relaxed);
        if n > 0 {
            out.push_str(&format!(" {n} {name};"));
        }
    }
    out.push('\n');
    let g = RACE_SUMMARY.lock();
    if let Some(m) = g.as_ref() {
        let mut v: Vec<_> = m.iter().filter(|(k, _)| k.0 < 5).collect();
        v.sort_by_key(|(_, e)| std::cmp::Reverse(e.0));
        for ((cause, shader, why, queue), (count, base, n, ex)) in v.into_iter().take(24) {
            out.push_str(&format!("      {count} x {queue} shader {shader:#x} (why {why:#x}) {base:#x}+{n:#x}: {ex}; {}\n", RACE_CAUSES[*cause]));
        }
    }
    drop(g);
    let opened = OPEN_READ_COUNT.load(Ordering::Relaxed);
    out.push_str(&format!("    bindings reading another queue's unsubmitted writes (run before them): {opened}\n"));
    let g = OPEN_READS.lock();
    if let Some(m) = g.as_ref() {
        let mut v: Vec<_> = m.iter().collect();
        v.sort_by_key(|(_, e)| std::cmp::Reverse(e.0));
        for ((reader, writer, shader), (count, ex)) in v.into_iter().take(16) {
            out.push_str(&format!("      {count} x {reader} shader {shader:#x} reads {writer}'s: {ex}\n"));
        }
    }
    let u: Vec<u64> = UNORDERED.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    out.push_str(&format!(
        "    unordered in-place bindings (written by this batch since its last barrier): read-only {} by draws, {} by dispatches ({} data-only loads, {} unbounded pointers); read and written {} by draws, {} by dispatches; draws after a title sync in the same rendering instance {}; commands after untracked writes {}\n",
        u[0], u[1], u[6], u[7], u[2], u[3], u[4], u[5]
    ));
    let g = UNORDERED_BY.lock();
    if let Some(m) = g.as_ref() {
        let mut v: Vec<_> = m.iter().collect();
        v.sort_by_key(|(_, e)| std::cmp::Reverse(e.0));
        for ((sh, kind), (count, ex)) in v.into_iter().take(16) {
            out.push_str(&format!("      {count} x cs {sh:#x} ({}): {ex}\n", match kind { 2 => "data-only loads", 4 => "unbounded pointer", _ => "buffer" }));
        }
    }
    out
}

/// Checks the speculations whose batch completed (completion thread).
fn check_speculations(done: u64) {
    if SPEC_PENDING.load(Ordering::Acquire) == 0 {
        return;
    }
    let due: Vec<SpecCheck> = {
        let mut g = SPEC_CHECKS.lock();
        let (due, keep): (Vec<SpecCheck>, Vec<SpecCheck>) = std::mem::take(&mut *g).into_iter().partition(|c| c.seq <= done);
        *g = keep;
        SPEC_PENDING.store(g.len() as u64, Ordering::Release);
        due
    };
    for c in due {
        let mut k = 0usize;
        let mut held = true;
        'reads: for &(a, n) in &c.reads {
            let m = (n / 4) as usize;
            if !super::cp::guest_readable(a, n) {
                held = false;
                break;
            }
            for j in 0..m {
                let now = unsafe { std::ptr::read_volatile((a + 4 * j as u64) as *const u32) };
                if c.words.get(k + j) != Some(&now) {
                    held = false;
                    break 'reads;
                }
            }
            k += m;
        }
        if held {
            SPEC_STATS[1].fetch_add(1, Ordering::Relaxed);
        } else {
            let n = SPEC_STATS[2].fetch_add(1, Ordering::Relaxed);
            NO_SPEC.lock().get_or_insert_with(HashMap::new).insert(c.shader, super::FRAMES.load(Ordering::Relaxed));
            if n < 20 {
                crate::info!("gpu: a stage used as prepared read words GPU work then changed (shader {:#x}, reads {:x?}); it waits for {NO_SPEC_FRAMES} frames", c.shader, c.reads);
            }
        }
    }
}

/// The latest unfinished batch that writes guest [base, base+size) (shader
/// writes and deferred CP writes), if any.
pub fn pending_writes(base: u64, size: u64) -> Option<u64> {
    let r = [(base, size)];
    INFLIGHT.latest(&r).max(CP_WRITES.latest(&r))
}

/// `wait_done` for at most `max`; whether batch `seq` completed.
pub fn wait_done_for(seq: u64, max: std::time::Duration) -> bool {
    let until = std::time::Instant::now() + max;
    let mut d = DONE.lock();
    while *d < seq {
        let now = std::time::Instant::now();
        if now >= until || super::shutting_down() {
            return false;
        }
        DONE_CV.wait_for(&mut d, (until - now).min(std::time::Duration::from_millis(100)));
    }
    true
}

/// Blocks until batch `seq` and all earlier ones completed (with their actions).
/// (A wait of more than 3 s is logged once, with its caller.)
#[track_caller]
pub fn wait_done(seq: u64) {
    let t = std::time::Instant::now();
    let mut logged = false;
    let mut d = DONE.lock();
    while *d < seq {
        if super::shutting_down() {
            return;
        }
        DONE_CV.wait_for(&mut d, std::time::Duration::from_millis(100));
        if !logged && *d < seq && t.elapsed() > std::time::Duration::from_secs(3) {
            logged = true;
            let l = std::panic::Location::caller();
            crate::warn!("gpu: waiting > 3 s for batch {seq} to complete (done {}) at {}:{}", *d, l.file(), l.line());
        }
    }
}

/// Queues a batch: `submit` runs (the vkQueueSubmit) under the queue lock so
/// sequence order is submission order.
#[allow(clippy::too_many_arguments)]
fn push_batch(fence: vk::Fence, queries: vk::QueryPool, bq: vk::QueryPool, actions: Vec<Action>, checks: Vec<verify::Check>, shaders: Vec<u64>, name: &str, gpu_labels: Vec<(u64, u64, u8)>, ops: [u64; OPS], segs: Vec<SegMark>, reset: Option<SlotReset>, alt: bool, submit: impl FnOnce(u64) -> Option<SubmitSpec> + Send + 'static) -> u64 {
    static STARTED: std::sync::Once = std::sync::Once::new();
    STARTED.call_once(|| {
        crate::thread::spawn_host("gpu-complete", 1 << 20, Box::new(completion_thread)).expect("gpu completion thread");
        crate::thread::spawn_host("gpu-submit", 1 << 20, Box::new(submitter_thread)).expect("gpu submission thread");
    });
    let mut q = BATCHES.lock();
    let seq = NEXT_SEQ.fetch_add(1, Ordering::SeqCst);
    if let Some(r) = &reset {
        r.state.store(seq, Ordering::Release);
    }
    // The submission thread submits in sequence order; inline only while
    // nothing is queued for it (the order holds when the switch changes).
    let mut sq = SUBMIT_Q.lock();
    if super::tune::submit_thread() || !sq.is_empty() {
        sq.push_back((seq, Box::new(submit)));
        drop(sq);
        SUBMIT_CV.notify_one();
    } else {
        drop(sq);
        if let Some(s) = submit(seq) {
            submit_specs(vec![(seq, s)]);
        }
        note_submitted(seq);
    }
    q.push_back(Batch { queries, bq, seq, fence, alt, actions, checks, shaders, submitted: std::time::Instant::now(), name: name.to_string(), gpu_labels, ops, segs, reset });
    BATCHES_CV.notify_one();
    seq
}

thread_local! {
    /// The batch just submitted (timeline names): 0 waits for no copy-queue
    /// copies, 1 its own copies ran there, 2 it waits only for copies earlier
    /// batches made there.
    static SUBMIT_KIND: std::cell::Cell<u8> = const { std::cell::Cell::new(0) };
}

/// Submissions for the submission thread (`tune submit_thread`): the
/// vkQueueSubmit of each batch in sequence order, off the command
/// processors (~7% of the graphics command processor's busy samples, ~220
/// batches per courtyard frame; profile run 20261012-001629).
type SubmitJob = Box<dyn FnOnce(u64) -> Option<SubmitSpec> + Send>;
static SUBMIT_Q: Mutex<std::collections::VecDeque<(u64, SubmitJob)>> = Mutex::new(std::collections::VecDeque::new());
static SUBMIT_CV: parking_lot::Condvar = parking_lot::Condvar::new();
/// Highest batch sequence submitted to Vulkan (the completion thread waits
/// for a fence only once its batch was submitted).
static SUBMITTED: Mutex<u64> = Mutex::new(0);
static SUBMITTED_CV: parking_lot::Condvar = parking_lot::Condvar::new();

fn note_submitted(seq: u64) {
    let mut g = SUBMITTED.lock();
    if *g < seq {
        *g = seq;
    }
    SUBMITTED_CV.notify_all();
}

fn submitter_thread() {
    crate::log::set_thread_name("gpu-submit");
    super::profile_this_thread("submit");
    loop {
        super::prio::apply();
        // The jobs queued (up to `tune submit_coalesce`): their batches are
        // finished first, then submitted together where they can be.
        let jobs: Vec<(u64, SubmitJob)> = {
            let mut q = SUBMIT_Q.lock();
            loop {
                if !q.is_empty() {
                    let n = q.len().min((super::tune::submit_coalesce() as usize).max(1));
                    break q.drain(..n).collect();
                }
                SUBMIT_CV.wait(&mut q);
            }
        };
        let tl = super::timeline::mark();
        SUBMIT_KIND.with(|k| k.set(0));
        let mut specs = Vec::with_capacity(jobs.len());
        let mut last = 0;
        let first = jobs[0].0;
        for (seq, job) in jobs {
            last = seq;
            if let Some(s) = job(seq) {
                specs.push((seq, s));
            }
        }
        submit_specs(specs);
        super::timeline::end("vk submit", tl, || if last == first { format!("#{first}{}", ["", " own copies", " earlier copies"][SUBMIT_KIND.with(|k| k.get()) as usize]) } else { format!("#{first}-#{last}") });
        note_submitted(last);
    }
}

/// Blocks until batch `seq` was submitted to Vulkan.
fn wait_submitted(seq: u64) {
    let mut g = SUBMITTED.lock();
    while *g < seq {
        if super::shutting_down() && DEVICE_LOST.load(Ordering::Relaxed) {
            return;
        }
        SUBMITTED_CV.wait_for(&mut g, std::time::Duration::from_millis(100));
    }
}

/// Waits for submitted batches in order, then performs their actions.
fn completion_thread() {
    crate::log::set_thread_name("gpu-complete");
    super::profile_this_thread("complete");
    loop {
        super::prio::apply();
        let mut b = {
            let mut q = BATCHES.lock();
            loop {
                if let Some(b) = q.pop_front() {
                    break b;
                }
                BATCHES_CV.wait(&mut q);
            }
        };
        if b.fence != vk::Fence::null() && !DEVICE_LOST.load(Ordering::Relaxed) {
            wait_submitted(b.seq);
            let tf = std::time::Instant::now();
            // (Batches submitted together carry one fence, the last's.)
            let r = if b.alt { unsafe { ctx().device.wait_for_fences(&[b.fence], true, 10_000_000_000) } } else if ctx().wait_batch(b.seq, 10_000_000_000) { Ok(()) } else { Err(vk::Result::TIMEOUT) };
            FENCE_NS.fetch_add(tf.elapsed().as_nanos() as u64, Ordering::Relaxed);
            if let Err(e) = r {
                if !DEVICE_LOST.swap(true, Ordering::SeqCst) {
                    let list: Vec<String> = b.shaders.iter().map(|a| format!("{a:#x}")).collect();
                    crate::error!("gpu[{}]: waiting for GPU work: {e}; GPU execution stops (batch #{}: shaders {}; {}); checkpoints: {}", b.name, b.seq, list.join(" "), ops_line(&b.ops), super::vk::checkpoint_report());
                }
            }
            FLUSHES.fetch_add(1, Ordering::Relaxed);
            collect_timestamps(b.bq, b.queries, &b.shaders, &b.name, b.seq, &b.segs, &b.ops);
            check_watchdog(&b.name, &b.shaders);
            let ms = b.submitted.elapsed().as_millis();
            if ms >= 250 {
                let list: Vec<String> = b.shaders.iter().map(|a| format!("{a:#x}")).collect();
                crate::info!("gpu[{}]: batch #{} of {} command(s) took {ms} ms (shaders {}; {})", b.name, b.seq, b.shaders.len(), list.join(" "), ops_line(&b.ops));
            }
        }
        // `tune host_census`: its written windows are armed before its CP
        // effects release anyone who reads them.
        crate::hostcensus::completed(b.seq);
        // `tune pre_protect`: shared copies due for arming.
        super::precache::arm_pending();
        // Label writes the GPU performed within the batch (before its CP
        // effects, which come after its work).
        for &(a, v, n) in &b.gpu_labels {
            super::cp::gpu_label_written(&b.name, a, v, n);
        }
        let ta = std::time::Instant::now();
        let tl = if b.actions.is_empty() { None } else { super::timeline::mark() };
        let n = b.actions.len();
        super::cp::set_after_batch(b.seq);
        for a in b.actions {
            a();
        }
        super::cp::set_after_batch(0);
        super::timeline::end("gpu-complete", tl, || format!("{} CP effect(s) of {} #{}", n, b.name, b.seq));
        ACTION_NS.fetch_add(ta.elapsed().as_nanos() as u64, Ordering::Relaxed);
        for ch in b.checks.iter_mut() {
            // A later batch (possibly already run) writes the destination again.
            if INFLIGHT.latest(&[verify::dest_range(ch)]).is_some_and(|s| s > b.seq) {
                ch.overwritten = true;
            }
            verify::run(&b.name, ch);
        }
        {
            let mut d = DONE.lock();
            *d = b.seq;
            DONE_ATOMIC.store(b.seq, Ordering::Release);
        }
        *LAST_DONE_AT.lock() = Some(std::time::Instant::now());
        DONE_CV.notify_all();
        // Its slot for the next batch there, unless the recording thread took
        // the reset back (`Exec::begin_batch`).
        if let Some(r) = b.reset.take() {
            if !DEVICE_LOST.load(Ordering::Relaxed) && r.state.compare_exchange(b.seq, u64::MAX, Ordering::AcqRel, Ordering::Acquire).is_ok() {
                let c = ctx();
                unsafe {
                    c.device.reset_fences(&[r.fence]).ok();
                    for p in r.pools {
                        if p != vk::CommandPool::null() {
                            c.device.reset_command_pool(p, vk::CommandPoolResetFlags::empty()).ok();
                        }
                    }
                    c.device.reset_descriptor_pool(r.desc_pool, vk::DescriptorPoolResetFlags::empty()).ok();
                }
                r.state.store(0, Ordering::Release);
            }
        }
        reap_spans();
        check_speculations(b.seq);
        check_races(b.seq);
        super::forward::completed(b.seq);
        check_timestamps(b.seq);
    }
}

/// A draw's image keys (`extra`: its attachments, then every texture of
/// `stages`) and whether one of them may have a fast clear, reload or block
/// copy pending, under one image-cache lock (a lock per texture and per
/// query took ~0.9 us per draw, run 20261011-135122).
pub(crate) fn image_hazards(stages: &[&Prepared], v: &mut Vec<super::images::ImageKey>) -> bool {
    // This thread's memos first (no lock); the cache's lock only on a miss
    // or when a fast clear or overwrite is pending somewhere. `v` holds the
    // attachments' keys (kept); the stages' images are added.
    let extra = v.len();
    let mut missed = false;
    'stages: for p in stages {
        for (i, r) in p.resources.iter().enumerate() {
            match r {
                Resolved::Image(t) => {
                    let storage = p.tr.plan.resources.get(i).is_some_and(|x| x.written);
                    match super::images::key_hit(t, storage) {
                        Some(k) => v.extend(k),
                        None => {
                            missed = true;
                            break 'stages;
                        }
                    }
                }
                Resolved::ImageTable(e) => {
                    for t in e.iter().flatten() {
                        match super::images::key_hit(t, false) {
                            Some(k) => v.extend(k),
                            None => {
                                missed = true;
                                break 'stages;
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }
    if !missed {
        let pending = if super::tune::pending_set() {
            super::images::pending_hit(&v)
        } else {
            super::images::any_pending() && super::images::with(|im| im.clears_pending(&v) || im.overwrites_pending(&v))
        } || super::images::block_copies_possible(v)
            || (super::tune::depth_views() && super::images::depth_copies_possible(v));
        return pending;
    }
    v.truncate(extra);
    super::images::with(|im| {
        for p in stages {
            for (i, r) in p.resources.iter().enumerate() {
                match r {
                    Resolved::Image(t) => {
                        let storage = p.tr.plan.resources.get(i).is_some_and(|x| x.written);
                        if let Some(k) = im.resolved_key(t, storage) {
                            v.push(k);
                        }
                    }
                    Resolved::ImageTable(e) => {
                        for t in e.iter().flatten() {
                            if let Some(k) = im.resolved_key(t, false) {
                                v.push(k);
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        im.clears_pending(v) || im.overwrites_pending(v) || im.block_copies_possible(v) || (super::tune::depth_views() && im.depth_feed_possible(v))
    })
}

/// Whether a fast clear or an overwrite reload may be pending for one of
/// `keys`, without the image cache's lock (its published pending set): a
/// command whose images only may need block copies took the lock twice for
/// nothing (`resolve_overwrites` was 3% of the graphics command
/// processor's busy samples, run 20261013-115322).
fn pending_for(keys: &[super::images::ImageKey]) -> bool {
    if super::tune::pending_set() {
        super::images::pending_hit(keys)
    } else {
        super::images::any_pending()
    }
}

/// Clear color of a DCC fast-clear code (gfx10: 00 -> 0000, 40 -> 0001,
/// 80 -> 1110, C0 -> 1111, RGBA; 20 -> the CLEAR_WORD registers; FF:
/// uncompressed, nothing to clear). Codes from Mesa radeonsi si_clear.c.
fn dcc_clear_color(code: u32, format: vk::Format, words: [u32; 2]) -> Result<Option<vk::ClearColorValue>, String> {
    let f = |v: [f32; 4]| Some(vk::ClearColorValue { float32: v });
    match code {
        0x00 => Ok(f([0.0, 0.0, 0.0, 0.0])),
        0x40 => Ok(f([0.0, 0.0, 0.0, 1.0])),
        0x80 => Ok(f([1.0, 1.0, 1.0, 0.0])),
        0xC0 => Ok(f([1.0, 1.0, 1.0, 1.0])),
        0xFF => Ok(None),
        // 0x10 (seen on the 64-bit lit-scene target with CLEAR_WORD 0): taken
        // as a clear to the registers like 0x20 (assumption, not verified).
        0x10 | 0x20 => {
            let h = |x: u32| super::images::half_to_f32(x as u16);
            match format {
                vk::Format::R16G16B16A16_SFLOAT => Ok(f([h(words[0]), h(words[0] >> 16), h(words[1]), h(words[1] >> 16)])),
                vk::Format::R32_SFLOAT => Ok(f([f32::from_bits(words[0]), 0.0, 0.0, 0.0])),
                vk::Format::R8G8B8A8_UNORM | vk::Format::R8G8B8A8_SRGB => {
                    let u = |s: u32| ((words[0] >> s) & 0xFF) as f32 / 255.0;
                    Ok(f([u(0), u(8), u(16), u(24)]))
                }
                _ => Err(format!("DCC clear-register code for {format:?} not supported")),
            }
        }
        c => Err(format!("DCC code {c:#x} not supported")),
    }
}

/// Reports invocations the loop watchdog stopped (after each completed
/// batch): once per program, which is saved for analysis.
fn check_watchdog(queue: &str, shaders: &[u64]) {
    let p = WATCHDOG_PTR.load(Ordering::Acquire);
    if p.is_null() {
        return;
    }
    // Shaders add to dword 0 with atomics while other batches may run.
    let n = unsafe { &*(p as *const std::sync::atomic::AtomicU32) }.swap(0, Ordering::AcqRel);
    if n == 0 {
        return;
    }
    let addr = unsafe { std::ptr::read_volatile(p.add(1)) as u64 | (std::ptr::read_volatile(p.add(2)) as u64) << 32 };
    WATCHDOG_STOPS.fetch_add(n as u64, Ordering::Relaxed);
    static SEEN: Mutex<Option<HashSet<u64>>> = Mutex::new(None);
    if !SEEN.lock().get_or_insert_with(HashSet::new).insert(addr) {
        return;
    }
    let list: Vec<String> = shaders.iter().map(|a| format!("{a:#x}")).collect();
    let saved = super::capture::save_program(addr).map(|f| format!("; program saved to {}", f.display())).unwrap_or_default();
    // Where it comes from, and the background compiles pending then (a
    // particle program's dispatches run while others of its effect are
    // left out).
    let from = super::shaders::program_hash(addr).ok().and_then(super::particles::origin).map(|d| format!(" from {d}")).unwrap_or_default();
    let pending = ASYNC_QUEUED.lock().as_ref().map_or(0, |s| s.len());
    crate::warn!(
        "gpu[{queue}]: loop watchdog stopped {n} invocation(s) of shader {addr:#x}{from} after {} blocks; its results are wrong (batch shaders {}; {pending} background compiles pending){saved}",
        rdna::translate::loop_limit(),
        list.join(" ")
    );
}

struct Slot {
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
    desc_pool: vk::DescriptorPool,
    /// Sequence number of the slot's last batch (0: none).
    seq: u64,
    /// Timestamp queries around each dispatch (DS_PROFILE_GPU=1).
    queries: vk::QueryPool,
    /// The batch's own timestamps (`batch_stamps`): its start and end
    /// (BQ_BATCH) and its prologue's (BQ_PRE).
    bq: vk::QueryPool,
    /// The SGPR data rings (RING_SEG each): system memory and video memory
    /// (allocated when `tune ring_vram` first chooses it).
    rings: [Option<(vk::Buffer, *mut u8)>; 2],
    /// Read-only buffer snapshots in video memory (SNAP_SEG, allocated on
    /// first use).
    snap: Option<(vk::Buffer, *mut u8)>,
    /// Device-local shadow space (working and pristine copies), `shadow_cap`
    /// bytes, made on first use (`Exec::shadow_room`; null before).
    shadow: vk::Buffer,
    shadow_mem: vk::DeviceMemory,
    shadow_cap: u64,
    /// The batch's prologue: copies of read-only buffers into `pre_buf`
    /// (PRE_SEG, allocated on first use), submitted before `cmd`.
    pre: vk::CommandBuffer,
    pre_buf: Option<vk::Buffer>,
    pre_mem: vk::DeviceMemory,
    pre_cap: u64,
    /// The prologue's copies on the copy queue (`DS_COPY_QUEUE`; null
    /// without one): copies of ranges no submitted batch writes.
    xpre: vk::CommandBuffer,
    /// Video-memory homes the slot's batch binds (kept until it completes).
    homes: Vec<std::sync::Arc<super::homes::HomeBuf>>,
    /// Shared copy chunks the slot's batch binds (kept until it completes).
    keep: Vec<std::sync::Arc<super::precache::Chunk>>,
    /// Kept windows' video memory the slot's batch binds (`gpu::keep`).
    kept: Vec<std::sync::Arc<super::keep::Slot>>,
    /// Timestamp queries and their copied results for CP timestamp writes
    /// the GPU performs (`Exec::gpu_timestamp`; made on first use).
    ts: Option<(vk::QueryPool, vk::Buffer)>,
    /// Staging for the batch's uploads into the residency mirror
    /// (STAGE_SEG of host memory, made on first use).
    stage: Option<(vk::Buffer, *mut u8)>,
    /// Mirror mode: more staging when the batch uploads more than STAGE_SEG
    /// (level loads; freed when the slot is reused).
    stage_extra: Vec<(vk::Buffer, vk::DeviceMemory, *mut u8)>,
    /// Mirror mode's write-backs: the gather kernel's outputs (host memory;
    /// records with the dwords, records of masks) and their sizes (grown
    /// when a batch needs more).
    mm_out: [(vk::Buffer, vk::DeviceMemory, *mut u8, u64); 2],
    /// The slot's own command pools (its queue's family; the copy queue's,
    /// null without one), so that a whole slot can be reset at once.
    cpool: vk::CommandPool,
    xcpool: vk::CommandPool,
    /// The batch whose completion the completion thread resets the slot
    /// after (`SlotReset`; 0: none or done, u64::MAX: resetting).
    reset_state: std::sync::Arc<AtomicU64>,
    /// The slot's last batch went to the completion thread for resetting.
    reset_pending: bool,
}

/// What the completion thread resets once a batch completed (`tune
/// offthread_reset`): its slot's command pools, descriptor pool and fence,
/// then `state` goes to 0. Resetting them took ~18 us of each batch begin
/// on the recording threads (profile, 2026-10-15 late night).
struct SlotReset {
    pools: [vk::CommandPool; 2],
    desc_pool: vk::DescriptorPool,
    fence: vk::Fence,
    state: std::sync::Arc<AtomicU64>,
}

/// Batch slots made and the buffers they hold (status line `batch slots`):
/// [slots, shadow bytes, batch-start copy segment bytes, snapshot bytes
/// (video memory the CPU writes), SGPR rings in video memory, host rings
/// and staging bytes]. Slots live as long as their executor.
static SLOT_MEM: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

/// The most a batch used of its slot's shadow and batch-start copy
/// segments, by executor role (graphics, compute queues, parallel
/// workers): [shadow bytes, copy bytes] (sizing the segments).
static SEG_MAX: [[AtomicU64; 2]; 3] = [const { [const { AtomicU64::new(0) }; 2] }; 3];

/// Notes a batch's segment use (`SEG_MAX`).
fn note_seg_use(deferred: bool, compute: bool, shadow: u64, pre: u64) {
    let role = if deferred { 2 } else if compute { 1 } else { 0 };
    SEG_MAX[role][0].fetch_max(shadow, Ordering::Relaxed);
    SEG_MAX[role][1].fetch_max(pre, Ordering::Relaxed);
}

/// The `batch slots` status line: what the executors' slots hold.
fn slot_mem_line() -> String {
    let v: Vec<u64> = SLOT_MEM.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    let device = v[1] + v[2] + v[3] + v[4];
    let m: Vec<String> = ["graphics", "compute queues", "workers"]
        .iter()
        .enumerate()
        .map(|(k, n)| format!("{n} {}/{} KiB", SEG_MAX[k][0].load(Ordering::Relaxed) >> 10, SEG_MAX[k][1].load(Ordering::Relaxed) >> 10))
        .collect();
    format!(
        "    batch slots: {} ({} MiB of video memory: shadows {}, batch-start copy segments {}, snapshots {}, SGPR rings {}; {} MiB of host rings and staging); most a batch used of its shadow/copy segment: {}
",
        v[0],
        device >> 20,
        v[1] >> 20,
        v[2] >> 20,
        v[3] >> 20,
        v[4] >> 20,
        v[5] >> 20,
        m.join(", ")
    )
}

impl Slot {
    fn new(family: u32, xfamily: Option<u32>) -> Result<Slot, String> {
        let c = ctx();
        let sizes: Vec<vk::DescriptorPoolSize> = POOL_TYPES.iter().zip(&POOL_LIMITS[1..]).map(|(&ty, &n)| vk::DescriptorPoolSize { ty, descriptor_count: n }).collect();
        let make_pool = |f: u32| unsafe { c.device.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(f).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER), None).map_err(|e| e.to_string()) };
        let pool = make_pool(family)?;
        let xpool = match xfamily {
            Some(f) => make_pool(f)?,
            None => vk::CommandPool::null(),
        };
        unsafe {
            let cmds = c.device.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(pool).command_buffer_count(2)).map_err(|e| e.to_string())?;
            let (cmd, pre) = (cmds[0], cmds[1]);
            let xpre = if xpool == vk::CommandPool::null() {
                vk::CommandBuffer::null()
            } else {
                c.device.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(xpool).command_buffer_count(1)).map_err(|e| e.to_string())?[0]
            };
            // SGPR data in system memory: a copy in video memory, read by
            // every invocation, made no measurable difference (`tune ring_vram`).
            let (ring, _mem, ring_ptr) = c.host_buffer(RING_SEG, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDIRECT_BUFFER)?;
            SLOT_MEM[0].fetch_add(1, Ordering::Relaxed);
            SLOT_MEM[5].fetch_add(RING_SEG, Ordering::Relaxed);
            Ok(Slot {
                cmd,
                fence: c.device.create_fence(&vk::FenceCreateInfo::default(), None).map_err(|e| e.to_string())?,
                desc_pool: c.device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(POOL_LIMITS[0]).pool_sizes(&sizes), None).map_err(|e| e.to_string())?,
                seq: 0,
                queries: if profiling() {
                    c.device.create_query_pool(&vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(STAMPS * MAX_TIMED), None).map_err(|e| e.to_string())?
                } else {
                    vk::QueryPool::null()
                },
                bq: if batch_stamps() {
                    c.device.create_query_pool(&vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(4), None).map_err(|e| e.to_string())?
                } else {
                    vk::QueryPool::null()
                },
                rings: [Some((ring, ring_ptr)), None],
                snap: None,
                shadow: vk::Buffer::null(),
                shadow_mem: vk::DeviceMemory::null(),
                shadow_cap: 0,
                pre,
                pre_buf: None,
                pre_mem: vk::DeviceMemory::null(),
                pre_cap: 0,
                stage: None,
                stage_extra: Vec::new(),
                mm_out: [(vk::Buffer::null(), vk::DeviceMemory::null(), std::ptr::null_mut(), 0); 2],
                xpre,
                homes: Vec::new(),
                keep: Vec::new(),
                kept: Vec::new(),
                ts: None,
                cpool: pool,
                xcpool: xpool,
                reset_state: std::sync::Arc::new(AtomicU64::new(0)),
                reset_pending: false,
            })
        }
    }
}

/// CP timestamp writes the GPU performs per batch (`Exec::gpu_timestamp`).
const TS_PER_BATCH: u32 = 512;

/// The timestamp store kernel (`rdna::kernels::timestamp_store`).
struct TsPipe {
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    set_layout: vk::DescriptorSetLayout,
}

/// A deferred end-of-pipe timestamp: its query, the guest memory buffer and
/// byte offset it is stored at, the clock offset (`title_clock_k`).
#[derive(Clone, Copy)]
struct TsEntry {
    q: u32,
    buffer: vk::Buffer,
    off: u64,
    k: u64,
}

/// The pipeline storing up to `rdna::kernels::TS_MANY` timestamps per
/// dispatch (`Exec::resolve_timestamps`).
fn ts_many_pipe() -> Result<&'static TsPipe, String> {
    static PIPE: std::sync::OnceLock<Result<TsPipe, String>> = std::sync::OnceLock::new();
    PIPE.get_or_init(|| {
        let c = ctx();
        unsafe {
            let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..2)
                .map(|b| vk::DescriptorSetLayoutBinding::default().binding(b).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE))
                .collect();
            let set_layout = c.device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None).map_err(|e| format!("timestamp set layout: {e}"))?;
            let layouts = [set_layout];
            let ranges = [vk::PushConstantRange { stage_flags: vk::ShaderStageFlags::COMPUTE, offset: 0, size: 4 * (4 + 2 * rdna::kernels::TS_MANY) }];
            let layout = c.device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts).push_constant_ranges(&ranges), None).map_err(|e| format!("timestamp layout: {e}"))?;
            let code = rdna::kernels::timestamp_store_many();
            let module = c.device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None).map_err(|e| format!("timestamp module: {e}"))?;
            let stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(c"main");
            let ci = vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout);
            let pipeline = super::pcache::with_cache('i', 3, |cache| c.device.create_compute_pipelines(cache, &[ci], None)).map_err(|(_, e)| format!("timestamp pipeline: {e}"));
            c.device.destroy_shader_module(module, None);
            Ok(TsPipe { pipeline: pipeline?[0], layout, set_layout })
        }
    })
    .as_ref()
    .map_err(|e| e.clone())
}

fn ts_pipe() -> Result<&'static TsPipe, String> {
    static PIPE: std::sync::OnceLock<Result<TsPipe, String>> = std::sync::OnceLock::new();
    PIPE.get_or_init(|| {
        let c = ctx();
        unsafe {
            let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..2)
                .map(|b| vk::DescriptorSetLayoutBinding::default().binding(b).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE))
                .collect();
            let set_layout = c.device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None).map_err(|e| format!("timestamp set layout: {e}"))?;
            let layouts = [set_layout];
            let ranges = [vk::PushConstantRange { stage_flags: vk::ShaderStageFlags::COMPUTE, offset: 0, size: 16 }];
            let layout = c.device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts).push_constant_ranges(&ranges), None).map_err(|e| format!("timestamp layout: {e}"))?;
            let code = rdna::kernels::timestamp_store();
            let module = c.device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None).map_err(|e| format!("timestamp module: {e}"))?;
            let stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(c"main");
            let ci = vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout);
            let pipeline = super::pcache::with_cache('i', 2, |cache| c.device.create_compute_pipelines(cache, &[ci], None)).map_err(|(_, e)| format!("timestamp pipeline: {e}"));
            c.device.destroy_shader_module(module, None);
            Ok(TsPipe { pipeline: pipeline?[0], layout, set_layout })
        }
    })
    .as_ref()
    .map_err(|e| e.clone())
}

/// The offset `k` that turns a GPU timestamp t (nanoseconds) into the
/// title's GPU clock as (t + k) / 10 (`cp::gpu_clock`, 100 MHz), from a
/// calibrated GPU timestamp (VK_EXT_calibrated_timestamps) renewed every
/// second. None: no calibration, or GPU ticks that are not nanoseconds.
fn title_clock_k() -> Option<u64> {
    static K: Mutex<Option<(std::time::Instant, u64)>> = Mutex::new(None);
    let mut g = K.lock();
    if let Some((at, k)) = *g {
        if at.elapsed() < std::time::Duration::from_secs(1) {
            return Some(k);
        }
    }
    let c = ctx();
    if (c.limits.timestamp_period - 1.0).abs() > 1e-6 {
        return None;
    }
    let (gpu, _) = c.calibrate()?;
    let title = super::cp::gpu_clock();
    let k = title.wrapping_mul(10).wrapping_sub(gpu);
    *g = Some((std::time::Instant::now(), k));
    Some(k)
}

const MAX_TIMED: u32 = 4096;
/// The batch timestamps' queries in `Slot::bq`: the batch's start and end,
/// its prologue's start and end.
const BQ_BATCH: u32 = 0;
const BQ_PRE: u32 = 2;

/// Timestamps at every batch's start and end and its prologue's (`Slot::bq`;
/// `DS_BATCH_STAMPS=0` at boot: none): each batch's span on the GPU, for the
/// hitch reports (`hitch::gpu_batch`: what the GPU was doing in a long
/// frame) and the frame windows' GPU busy time; with DS_PROFILE_GPU the
/// profile's batch times.
pub(super) fn batch_stamps() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| profiling() || std::env::var("DS_BATCH_STAMPS").ok().as_deref() != Some("0"))
}

/// Timed draws are keyed by their pixel shader (vertex shader without one)
/// with this bit set.
const DRAW_KEY: u64 = 1 << 63;

/// Whether DS_PROFILE_GPU is on (other modules' profile statistics).
pub fn profiling_on() -> bool {
    profiling()
}

fn profiling() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_PROFILE_GPU").ok().as_deref() == Some("1"))
}

/// Stretches past label waits checked again at their label
/// (`Exec::past_current_for`): [stretches, ranges, found stale, ns].
pub static PAST_CHECKS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];
/// Hot pages of small bindings a stretch past a label wait uploaded as
/// they are when it runs (`fe2::hot_pages`): [runs, pages].
pub static PAST_HOT: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
/// Bindings of at most this many pages get those uploads.
const STRETCH_HOT_PAGES: u64 = 64;
static PAST_STALE_LOGGED: AtomicU64 = AtomicU64::new(0);

/// The per-step timers of dispatches, their bindings' parts and their
/// batch-start copy lookups: with the profiler, or alone with
/// `DS_STEP_LAPS=1` (light enough not to change what it measures: the
/// profiler as a whole moved every boot to one speed, FINDINGS.md
/// 2026-10-26).
fn laps() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| profiling() || std::env::var("DS_STEP_LAPS").ok().as_deref() == Some("1"))
}

/// Per-command GPU timestamps (DS_PROFILE_GPU=1; `DS_PROFILE_GPU_STAMPS=0`
/// keeps only the batch ones): BOTTOM_OF_PIPE stamps around every draw and
/// dispatch serialize the GPU, which inflates the time of small commands.
pub(super) fn command_stamps() -> bool {
    matches!(stamp_mode(), 1 | 2)
}

/// `DS_PROFILE_GPU_STAMPS=seg`: no stamps around commands; a timestamp
/// after every full barrier the executors record (the GPU drained there
/// anyway), each segment's time split among the commands in it
/// (`collect_segments`): GPU time by command without serializing them.
pub(super) fn seg_stamps() -> bool {
    profiling() && stamp_mode() == 3
}

/// 0: batch stamps only, 1: every command, 2 (`DS_PROFILE_GPU_STAMPS=big`):
/// only commands large enough that the serialization does not distort
/// them (`stamp_worth`).
fn stamp_mode() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("DS_PROFILE_GPU_STAMPS").ok().as_deref() {
        Some("0") => 0,
        Some("big") => 2,
        Some("seg") => 3,
        _ => 1,
    })
}

/// Whether a command of `work` items (a dispatch's workgroups, a draw's
/// vertices times instances; None: indirect, unknown) gets stamps.
pub(super) fn stamp_worth(work: Option<u64>, big: u64) -> bool {
    match stamp_mode() {
        1 => true,
        2 => work.is_none_or(|w| w >= big),
        _ => false,
    }
}

/// A full barrier a batch recorded with a timestamp after it (`seg_stamps`):
/// its query, and the batch's command count and operation counts then.
#[derive(Clone, Copy)]
struct SegMark {
    query: u32,
    cmds: u32,
    ops: [u64; OPS],
    /// Where the barrier was recorded.
    site: &'static std::panic::Location<'static>,
}

/// GPU time by barrier segment (`seg_stamps`): per command key (a compute
/// shader, or DRAW_KEY | pixel shader) [ns, commands]; segments without
/// commands by what they recorded [ns, segments]; [ns in segments with
/// commands, without, segments].
static SEG_TIME: Mutex<Option<HashMap<u64, [u64; 2]>>> = Mutex::new(None);
static SEG_EMPTY: Mutex<Option<HashMap<&'static str, [u64; 2]>>> = Mutex::new(None);
static SEG_TOTAL: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
/// Segments between two barriers with nothing recorded between them, by the
/// two barriers' sites (the batch's start counts as a site): [ns, segments].
type SitePair = (&'static str, u32, &'static str, u32);
static SEG_PAIRS: Mutex<Option<HashMap<SitePair, [u64; 2]>>> = Mutex::new(None);
/// Segments with commands by queue and the barrier site that ended them:
/// [ns, segments, commands].
static SEG_ENDS: Mutex<Option<HashMap<(String, &'static str, u32), [u64; 3]>>> = Mutex::new(None);

/// What a segment without commands recorded (its operation counts' change).
fn seg_kind(a: &[u64; OPS], b: &[u64; OPS]) -> &'static str {
    const KINDS: [(usize, &str); 9] = [
        (OP_FE2_SCATTERS, "front end 2 write-backs"),
        (OP_MERGES, "shadow merges"),
        (OP_SHADOWS_IN, "shadow copies in"),
        (OP_TIMESTAMPS, "timestamps"),
        (OP_CP_WRITES, "CP writes"),
        (OP_FILLS, "buffer fills"),
        (OP_FAST_CLEARS, "fast clears"),
        (OP_WRITE_BACK_BYTES, "home write-backs"),
        (OP_PASS_ENDS, "rendering ends"),
    ];
    KINDS.iter().find(|&&(k, _)| b[k] > a[k]).map_or("barriers only", |&(_, n)| n)
}

/// Splits a completed batch's time among its barrier segments (`seg_stamps`).
fn collect_segments(pool: vk::QueryPool, shaders: &[u64], segs: &[SegMark], ops_end: &[u64; OPS], be: [u64; 2], queue: &str) {
    let c = ctx();
    let mut t = vec![0u64; segs.len()];
    if !segs.is_empty() && unsafe { c.device.get_query_pool_results(pool, 0, &mut t, vk::QueryResultFlags::TYPE_64) }.is_err() {
        return;
    }
    let period = c.limits.timestamp_period as f64;
    // Boundaries in recording order: the batch's start, each mark, its end
    // (kept monotonic: the start stamp follows the leading barrier's).
    let start: (&'static str, u32) = ("batch start", 0);
    let end: (&'static str, u32) = ("batch end", 0);
    let mut bounds: Vec<(u64, usize, [u64; OPS], (&'static str, u32))> = Vec::with_capacity(segs.len() + 2);
    bounds.push((be[0], 0, [0; OPS], start));
    for (m, &v) in segs.iter().zip(&t) {
        let last = bounds.last().map_or(0, |b| b.0);
        let file = m.site.file().rsplit(['/', '\\']).next().unwrap_or("");
        bounds.push((v.max(last), (m.cmds as usize).min(shaders.len()), m.ops, (file, m.site.line())));
    }
    let last = bounds.last().map_or(0, |b| b.0);
    bounds.push((be[1].max(last), shaders.len(), *ops_end, end));
    let mut gp = SEG_PAIRS.lock();
    let mp = gp.get_or_insert_with(HashMap::new);
    let mut g = SEG_TIME.lock();
    let m = g.get_or_insert_with(HashMap::new);
    let mut ge = SEG_EMPTY.lock();
    let me = ge.get_or_insert_with(HashMap::new);
    let mut gq = SEG_ENDS.lock();
    let mq = gq.get_or_insert_with(HashMap::new);
    for w in bounds.windows(2) {
        let d = ((w[1].0 - w[0].0) as f64 * period) as u64;
        let (from, to) = (w[0].1, w[1].1.max(w[0].1));
        let cmds = &shaders[from..to];
        // On the timeline: the queue and the segment's programs, or what it
        // held instead of commands.
        if super::timeline::on() {
            super::timeline::gpu_segment(w[0].0, w[1].0, || {
                if cmds.is_empty() {
                    return format!("{queue}: {}", seg_kind(&w[0].2, &w[1].2));
                }
                let mut names: Vec<String> = Vec::new();
                for &k in cmds {
                    let n = if k & DRAW_KEY != 0 { format!("draws {:#x}", k & !DRAW_KEY) } else { format!("cs {:#x}", k) };
                    if !names.contains(&n) {
                        names.push(n);
                    }
                }
                format!("{queue}: {} ({} commands)", names.join(" + "), cmds.len())
            });
        }
        if cmds.is_empty() {
            let kind = seg_kind(&w[0].2, &w[1].2);
            let e = me.entry(kind).or_insert([0; 2]);
            e[0] += d;
            e[1] += 1;
            if kind == "barriers only" {
                let e = mp.entry((w[0].3 .0, w[0].3 .1, w[1].3 .0, w[1].3 .1)).or_insert([0; 2]);
                e[0] += d;
                e[1] += 1;
            }
            SEG_TOTAL[1].fetch_add(d, Ordering::Relaxed);
        } else {
            let each = d / cmds.len() as u64;
            for &k in cmds {
                let e = m.entry(k).or_insert([0; 2]);
                e[0] += each;
                e[1] += 1;
            }
            let e = mq.entry((queue.to_string(), w[1].3 .0, w[1].3 .1)).or_insert([0; 3]);
            e[0] += d;
            e[1] += 1;
            e[2] += cmds.len() as u64;
            SEG_TOTAL[0].fetch_add(d, Ordering::Relaxed);
        }
        SEG_TOTAL[2].fetch_add(1, Ordering::Relaxed);
    }
}

/// The segment profile's lines (`seg_stamps`).
fn seg_lines() -> String {
    let t: Vec<u64> = SEG_TOTAL.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    let mut s = format!(
        "    gpu time by barrier segment (DS_PROFILE_GPU_STAMPS=seg): {} ms with commands, {} ms without, {} segments\n",
        t[0] / 1_000_000,
        t[1] / 1_000_000,
        t[2]
    );
    if let Some(m) = SEG_TIME.lock().as_ref() {
        let mut v: Vec<(&u64, &[u64; 2])> = m.iter().collect();
        v.sort_by(|a, b| b.1[0].cmp(&a.1[0]));
        for (k, x) in v.iter().take(30) {
            let what = if **k & DRAW_KEY != 0 { format!("draws {:#x}", **k & !DRAW_KEY) } else { format!("cs {:#x}", **k) };
            s.push_str(&format!("      {what}: {} ms, {} commands, {:.1} us each\n", x[0] / 1_000_000, x[1], x[0] as f64 / x[1].max(1) as f64 / 1000.0));
        }
    }
    if let Some(m) = SEG_EMPTY.lock().as_ref() {
        let mut v: Vec<(&&str, &[u64; 2])> = m.iter().collect();
        v.sort_by(|a, b| b.1[0].cmp(&a.1[0]));
        let parts: Vec<String> = v.iter().map(|(k, x)| format!("{k} {} ms ({} segments, {:.1} us each)", x[0] / 1_000_000, x[1], x[0] as f64 / x[1].max(1) as f64 / 1000.0)).collect();
        s.push_str(&format!("      segments without commands: {}\n", parts.join(", ")));
    }
    if let Some(m) = SEG_PAIRS.lock().as_ref() {
        let mut v: Vec<(&SitePair, &[u64; 2])> = m.iter().collect();
        v.sort_by(|a, b| b.1[0].cmp(&a.1[0]));
        let parts: Vec<String> = v.iter().take(16).map(|((f0, l0, f1, l1), x)| format!("{f0}:{l0} -> {f1}:{l1} {} ms ({})", x[0] / 1_000_000, x[1])).collect();
        s.push_str(&format!("      barrier-only segments by their barriers' sites: {}\n", parts.join(", ")));
    }
    if let Some(m) = SEG_ENDS.lock().as_ref() {
        // Per queue (the four with the most such time): the sites whose
        // barriers ended its segments with commands.
        let mut by_q: HashMap<&str, (u64, Vec<(&'static str, u32, [u64; 3])>)> = HashMap::new();
        for ((q, f, l), x) in m.iter() {
            let e = by_q.entry(q.as_str()).or_default();
            e.0 += x[0];
            e.1.push((f, *l, *x));
        }
        let mut qs: Vec<(&str, (u64, Vec<(&'static str, u32, [u64; 3])>))> = by_q.into_iter().collect();
        qs.sort_by(|a, b| b.1 .0.cmp(&a.1 .0));
        for (q, (ns, mut v)) in qs.into_iter().take(4) {
            v.sort_by(|a, b| b.2[0].cmp(&a.2[0]));
            let segs: u64 = v.iter().map(|x| x.2[1]).sum();
            let parts: Vec<String> = v.iter().take(8).map(|(f, l, x)| format!("{f}:{l} {} ms ({} segments, {:.1} commands, {:.1} us each)", x[0] / 1_000_000, x[1], x[2] as f64 / x[1].max(1) as f64, x[0] as f64 / x[1].max(1) as f64 / 1000.0)).collect();
            s.push_str(&format!("      {q}: segments with commands {} ms ({segs}), ended by: {}\n", ns / 1_000_000, parts.join(", ")));
        }
    }
    s
}

/// Timestamps per timed dispatch (all BOTTOM_OF_PIPE): before the shadow
/// copies in, before the dispatch, after it, after the copies out.
const STAMPS: u32 = 4;

/// GPU time per shader (DS_PROFILE_GPU=1): address -> [dispatches, dispatch
/// ns, copy-in ns, copy-out ns].
static SHADER_TIME: Mutex<Option<HashMap<u64, [u64; 4]>>> = Mutex::new(None);
/// GPU time between timed dispatches (draws, CP copies, barriers), ns.
static GAP_NS: AtomicU64 = AtomicU64::new(0);

/// Batch-start copies of a range already copied by another batch in the
/// same frame (DS_PROFILE_GPU): [copies, bytes] first in the frame, then
/// repeated; by size [<4 KiB, <64 KiB, <1 MiB, more] x [copies, bytes].
static PRE_REPEAT: Mutex<(u64, Option<HashSet<(u64, u64)>>, [u64; 4], [[u64; 2]; 4])> = Mutex::new((u64::MAX, None, [0; 4], [[0; 2]; 4]));

/// Batch-start copies across frames (DS_PROFILE_GPU): per range, the frame
/// of its last first-in-frame copy and a sampled hash of its bytes then
/// (`images::sampled_hash`); bytes of first-in-frame copies [of ranges not
/// copied in the two frames before, unchanged since, changed].
static PRE_STABLE: Mutex<(Option<HashMap<(u64, u64), (u64, u64)>>, [u64; 5])> = Mutex::new((None, [0; 5]));
/// Bytes of first-in-frame copies of ranges unchanged since an earlier
/// frame, by why the shared copies did not serve them
/// (`precache::GONE_NAMES`).
static PRE_UNCHANGED_WHY: Mutex<[u64; 10]> = Mutex::new([0; 10]);
/// Batch-start copies by content (DS_PROFILE_GPU with `DS_COPY_CONTENT=1`;
/// copies under 1 MiB): per (hash of the bytes, size), the frame they were
/// last copied in and where; per size class (<4 KiB, <64 KiB, <1 MiB),
/// [copies, bytes] of content copied in the last `CONTENT_FRAMES` frames at
/// the same address, at another address, and of content not seen so.
static PRE_CONTENT: Mutex<(Option<HashMap<(u64, u64), (u64, u64)>>, [[[u64; 2]; 3]; 3])> = Mutex::new((None, [[[0; 2]; 3]; 3]));
/// Frames a copy's content counts as seen (`PRE_CONTENT`).
const CONTENT_FRAMES: u64 = 8;

/// The content census is on (`PRE_CONTENT`).
fn copy_content() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_COPY_CONTENT").ok().as_deref() == Some("1"))
}

/// A batch-start copy of guest [base, base+size) made in `frame`
/// (`PRE_CONTENT`): its bytes hashed whole.
fn note_pre_content(base: u64, size: u64, frame: u64) {
    if size >= 1 << 20 {
        return;
    }
    let h = super::images::hash_bytes(base, size as usize);
    let class = match size {
        0..4096 => 0,
        4096..65536 => 1,
        _ => 2,
    };
    let mut g = PRE_CONTENT.lock();
    let m = g.0.get_or_insert_with(HashMap::new);
    if m.len() > 1 << 21 {
        m.clear();
    }
    let k = match m.insert((h, size), (frame, base)) {
        Some((f, b)) if f + CONTENT_FRAMES >= frame => {
            if b == base {
                0
            } else {
                1
            }
        }
        _ => 2,
    };
    g.1[class][k][0] += 1;
    g.1[class][k][1] += size;
}

/// The content census's profile line ("" when off).
fn pre_content_line() -> String {
    if !copy_content() {
        return String::new();
    }
    let g = PRE_CONTENT.lock();
    let mb = |b: u64| b as f64 / (1u64 << 20) as f64;
    let classes: Vec<String> = ["<4K", "<64K", "<1M"]
        .iter()
        .zip(g.1.iter())
        .map(|(n, c)| format!("{n}: same address {} ({:.0} MiB), another address {} ({:.0} MiB), new {} ({:.0} MiB)", c[0][0], mb(c[0][1]), c[1][0], mb(c[1][1]), c[2][0], mb(c[2][1])))
        .collect();
    format!("    batch-start copies under 1 MiB by content (bytes copied in the last {CONTENT_FRAMES} frames): {}\n", classes.join("; "))
}
/// Batch-start copies of 1 MiB or more by range (DS_PROFILE_GPU): [copies,
/// bytes, repeats within a frame, first-in-frame copies of a range unchanged
/// since an earlier frame, of a changed one], then copies by why no shared
/// copy served them (`precache::GONE_NAMES`); and the flip of the last
/// `profile reset`.
static PRE_BIG: Mutex<(u64, Option<HashMap<(u64, u64), [u64; 15]>>)> = Mutex::new((0, None));

fn note_pre_stable(base: u64, size: u64, n: u64, frame: u64, why: u8) {
    let h = super::images::sampled_hash(base, size);
    let mut g = PRE_STABLE.lock();
    let m = g.0.get_or_insert_with(HashMap::new);
    if m.len() > 1 << 20 {
        m.clear();
    }
    let k = match m.insert((base, size), (frame, h)) {
        Some((f, old)) if f + 2 >= frame => {
            if old == h {
                1
            } else {
                2
            }
        }
        _ => 0,
    };
    g.1[k] += n;
    if k == 1 {
        PRE_UNCHANGED_WHY.lock()[(why as usize).min(9)] += n;
    }
    if n >= 1 << 20 && k > 0 {
        if let Some(e) = PRE_BIG.lock().1.as_mut().and_then(|m| m.get_mut(&(base, size))) {
            e[2 + k] += 1;
        }
    }
}

/// A batch-start copy of `n` bytes (DS_PROFILE_GPU): ranges of 1 MiB or more
/// counted by range (`PRE_BIG`; `repeat`: copied before in this frame).
fn note_pre_big(base: u64, size: u64, n: u64, why: u8, repeat: bool) {
    if n < 1 << 20 {
        return;
    }
    let mut g = PRE_BIG.lock();
    let m = g.1.get_or_insert_with(HashMap::new);
    if m.len() > 1 << 14 {
        m.clear();
    }
    let e = m.entry((base, size)).or_insert([0; 15]);
    e[0] += 1;
    e[1] += n;
    e[2] += repeat as u64;
    e[5 + (why as usize).min(9)] += 1;
}

/// The profile line of the largest batch-start copies (`PRE_BIG`).
fn pre_big_line() -> String {
    let g = PRE_BIG.lock();
    let Some(m) = g.1.as_ref() else { return String::new() };
    let flips = super::FRAMES.load(Ordering::Relaxed).saturating_sub(g.0).max(1) as f64;
    let mut v: Vec<(&(u64, u64), &[u64; 15])> = m.iter().collect();
    v.sort_by(|a, b| b.1[1].cmp(&a.1[1]));
    let total: u64 = m.values().map(|e| e[1]).sum();
    let parts: Vec<String> = v
        .iter()
        .take(12)
        .map(|((b, s), e)| {
            let whys: Vec<String> = super::precache::GONE_NAMES.iter().zip(&e[5..15]).filter(|&(_, &c)| c > 0).map(|(n, c)| format!("{n} {c}")).collect();
            format!("{b:#x}+{s:#x} {:.1}/flip {:.1} MB/flip (repeated in the frame {}, first-in-frame unchanged {} / changed {}; why: {})", e[0] as f64 / flips, e[1] as f64 / flips / 1e6, e[2], e[3], e[4], whys.join(", "))
        })
        .collect();
    format!("    batch-start copies of 1 MiB or more ({:.1} MB/flip in {} ranges), the largest: {}\n", total as f64 / flips / 1e6, m.len(), parts.join("; "))
}

fn note_pre_repeat(base: u64, size: u64, n: u64, why: u8) {
    let frame = super::FRAMES.load(Ordering::Relaxed);
    let mut g = PRE_REPEAT.lock();
    if g.0 != frame {
        g.0 = frame;
        if let Some(s) = g.1.as_mut() {
            s.clear();
        }
    }
    let first = g.1.get_or_insert_with(HashSet::new).insert((base, size));
    drop(g);
    if copy_content() {
        note_pre_content(base, size, frame);
    }
    note_pre_big(base, size, n, why, !first);
    if first {
        note_pre_stable(base, size, n, frame, why);
    }
    let mut g = PRE_REPEAT.lock();
    let k = if first { 0 } else { 2 };
    g.2[k] += 1;
    g.2[k + 1] += n;
    let b = match n {
        0..4096 => 0,
        4096..65536 => 1,
        65536..1048576 => 2,
        _ => 3,
    };
    g.3[b][0] += 1;
    g.3[b][1] += n;
}

fn pre_repeat_line() -> String {
    let st = PRE_STABLE.lock().1;
    let g = PRE_REPEAT.lock();
    let mb = |b: u64| b as f64 / (1u64 << 20) as f64;
    let stable = format!(
        "    batch-start copies across frames (first in a frame): ranges not copied in the two frames before {:.0} MiB, unchanged since {:.0} MiB, changed {:.0} MiB (sampled hashes); write-watch queries {} ({:.1} us each, {} pages; {} found clean by a probe)\n",
        mb(st[0]),
        mb(st[1]),
        mb(st[2]),
        crate::mem::WRITE_WATCH_STATS[0].load(Ordering::Relaxed),
        crate::mem::WRITE_WATCH_STATS[1].load(Ordering::Relaxed) as f64 / crate::mem::WRITE_WATCH_STATS[0].load(Ordering::Relaxed).max(1) as f64 / 1000.0,
        crate::mem::WRITE_WATCH_STATS[2].load(Ordering::Relaxed),
        crate::mem::PROBE_CLEAN.load(Ordering::Relaxed)
    );
    let sizes: Vec<String> = ["<4K", "<64K", "<1M", ">=1M"].iter().zip(g.3.iter()).map(|(n, x)| format!("{n} {} ({:.0} MiB)", x[0], mb(x[1]))).collect();
    let why = *PRE_UNCHANGED_WHY.lock();
    let whys: Vec<String> = super::precache::GONE_NAMES.iter().zip(why).filter(|&(_, b)| b > 0).map(|(n, b)| format!("{n} {:.0} MiB", mb(b))).collect();
    format!(
        "    batch-start copies within a frame: first {} ({:.0} MiB), repeated {} ({:.0} MiB); by size {}\n",
        g.2[0],
        mb(g.2[1]),
        g.2[2],
        mb(g.2[3]),
        sizes.join(", ")
    ) + &stable
        + &format!("    first-in-frame copies of unchanged ranges, by why no shared copy served them: {}\n", whys.join(", "))
        + &pre_content_line()
        + &pre_big_line()
}

/// Command-processor time per recorded command by phase (DS_PROFILE_GPU),
/// ns: dispatches [prepare, waits + textures + pipeline, bindings, total,
/// count], draws [prepare + shader-read waits, textures + targets +
/// pipeline, bindings, total, count].
pub(super) static CS_PHASES: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];
pub(super) static DRAW_PHASES: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];

/// Command-processor time per compute shader and dispatch kind (`CS_KINDS`)
/// while profiling: [dispatches, then ns until prepared, until textures and
/// pipeline, until bound, in all].
static CS_CPU: Mutex<Option<HashMap<(u64, usize), [u64; 5]>>> = Mutex::new(None);

fn note_cs_cpu(addr: u64, kind: usize, t: [std::time::Duration; 4]) {
    let mut g = CS_CPU.lock();
    let e = g.get_or_insert_with(HashMap::new).entry((addr, kind)).or_insert([0; 5]);
    e[0] += 1;
    for k in 0..4 {
        e[k + 1] += t[k].as_nanos() as u64;
    }
}

/// `DS_PROFILE_CS_QUEUE=<text>`: the same per compute shader for the queues
/// whose name holds the text alone (`acb0x50`), in a table of its own.
static CS_CPU_Q: Mutex<Option<HashMap<u64, [u64; 5]>>> = Mutex::new(None);

fn cs_queue_filter() -> Option<&'static str> {
    static F: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    F.get_or_init(|| std::env::var("DS_PROFILE_CS_QUEUE").ok().filter(|v| !v.is_empty())).as_deref()
}

fn cs_queue_profiled(name: &str) -> bool {
    cs_queue_filter().is_some_and(|f| name.contains(f))
}

fn note_cs_cpu_queue(addr: u64, t: [std::time::Duration; 4]) {
    let mut g = CS_CPU_Q.lock();
    let e = g.get_or_insert_with(HashMap::new).entry(addr).or_insert([0; 5]);
    e[0] += 1;
    for k in 0..4 {
        e[k + 1] += t[k].as_nanos() as u64;
    }
}

/// The table of `DS_PROFILE_CS_QUEUE`'s queues (profile).
fn cs_cpu_queue_lines() -> String {
    let Some(f) = cs_queue_filter() else { return String::new() };
    let g = CS_CPU_Q.lock();
    let Some(m) = g.as_ref() else { return String::new() };
    let mut v: Vec<(&u64, &[u64; 5])> = m.iter().collect();
    v.sort_by(|a, b| b.1[4].cmp(&a.1[4]));
    let all: u64 = v.iter().map(|x| x.1[4]).sum();
    let mut s = format!("    command-processor time by compute shader on queues matching {f} ({} ms in all):\n", all / 1_000_000);
    for (i, (a, x)) in v.iter().take(25).enumerate() {
        let n = x[0].max(1) as f64;
        let us = |k: usize| x[k] as f64 / n / 1000.0;
        s.push_str(&format!(
            "      cs {a:#x}: {} dispatches, {} ms, us each: prepare and waits {:.1}, textures and pipeline {:.1}, bindings {:.1}, recording {:.1}\n",
            x[0],
            x[4] / 1_000_000,
            us(1),
            us(2) - us(1),
            us(3) - us(2),
            us(4) - us(3)
        ));
        if i < 12 {
            if let Some(l) = placement_line(**a) {
                s.push_str(&l);
                s.push('\n');
            }
        }
    }
    s
}

/// The compute shaders taking the most command-processor time (profile).
fn cs_cpu_lines() -> String {
    let g = CS_CPU.lock();
    let Some(m) = g.as_ref() else { return String::new() };
    let mut v: Vec<(&(u64, usize), &[u64; 5])> = m.iter().collect();
    v.sort_by(|a, b| b.1[4].cmp(&a.1[4]));
    let all: u64 = v.iter().map(|x| x.1[4]).sum();
    let mut s = format!("    command-processor time by compute shader ({} ms in all):\n", all / 1_000_000);
    // The 16 costliest, then the graphics command processor's own (its
    // serial runs and its chunks of parallel ones) and the compute queues'
    // costliest indirect ones, with their buffer placements.
    let main: Vec<&(&(u64, usize), &[u64; 5])> = v.iter().filter(|x| x.0 .1 == 0).take(10).collect();
    let n_main = main.len();
    let indirect: Vec<&(&(u64, usize), &[u64; 5])> = v.iter().filter(|x| x.0 .1 == 3).take(10).collect();
    for (i, ((a, kind), x)) in v.iter().take(16).chain(main.into_iter()).chain(indirect.into_iter()).enumerate() {
        if i == 16 {
            s.push_str("    the graphics command processor's own dispatches by its time:\n");
        }
        if i == 16 + n_main {
            s.push_str(&format!(
                "    compute queues' indirect dispatches by command-processor time (also those whose arguments the CPU read; dispatches with no groups since boot: {} direct, {} indirect):\n",
                ZERO_GROUPS[0].load(Ordering::Relaxed),
                ZERO_GROUPS[1].load(Ordering::Relaxed)
            ));
        }
        let n = x[0].max(1) as f64;
        let us = |k: usize| x[k] as f64 / n / 1000.0;
        s.push_str(&format!(
            "      cs {a:#x} ({}): {} dispatches, {} ms, us each: prepare and waits {:.1}, textures and pipeline {:.1}, bindings {:.1}, recording {:.1}\n",
            CS_KINDS[*kind],
            x[0],
            x[4] / 1_000_000,
            us(1),
            us(2) - us(1),
            us(3) - us(2),
            us(4) - us(3)
        ));
        if i >= 16 {
            if let Some(l) = placement_line(*a) {
                s.push_str(&l);
                s.push('\n');
            }
        }
    }
    s
}

/// Draws and dispatches by their place in their batch (DS_PROFILE_GPU):
/// [main command processor, parallel workers][place: 1st, 2nd, 3rd-4th,
/// 5th-8th, 9th-16th, 17th-32nd, 33rd-64th, 65th-128th, later] as
/// `note_phases` tables: what a fresh batch's first commands cost (a
/// parallel chunk is a batch of its own).
pub(super) static PLACE_PHASES: [[[AtomicU64; 5]; 9]; 2] = [const { [const { [const { AtomicU64::new(0) }; 5] }; 9] }; 2];

/// A command recorded as the `place`-th (from 0) of its batch.
pub(super) fn note_place(worker: bool, place: usize, t: [std::time::Duration; 4]) {
    let b = match place {
        0 => 0,
        1 => 1,
        2..=3 => 2,
        4..=7 => 3,
        8..=15 => 4,
        16..=31 => 5,
        32..=63 => 6,
        64..=127 => 7,
        _ => 8,
    };
    note_phases(&PLACE_PHASES[worker as usize][b], t);
}

fn place_lines() -> String {
    const NAMES: [&str; 9] = ["1st", "2nd", "3rd-4th", "5th-8th", "9th-16th", "17th-32nd", "33rd-64th", "65th-128th", "later"];
    let mut s = String::new();
    for (w, who) in ["main command processor", "parallel workers"].iter().enumerate() {
        let parts: Vec<String> = PLACE_PHASES[w]
            .iter()
            .zip(NAMES)
            .filter(|(t, _)| t[4].load(Ordering::Relaxed) > 0)
            .map(|(t, name)| {
                let n = t[4].load(Ordering::Relaxed) as f64;
                let us = |k: usize| t[k].load(Ordering::Relaxed) as f64 / n / 1000.0;
                format!("{name} {:.0}: {:.1} us ({:.1}/{:.1}/{:.1}/{:.1})", n, us(3), us(0), us(1) - us(0), us(2) - us(1), us(3) - us(2))
            })
            .collect();
        s.push_str(&format!("    {who}: draws and dispatches by place in their batch (count: us each (prepare/textures and pipeline/bindings/recording)): {}\n", parts.join(", ")));
    }
    s
}

pub(super) fn note_phases(table: &[AtomicU64; 5], t: [std::time::Duration; 4]) {
    for (k, d) in t.iter().enumerate() {
        table[k].fetch_add(d.as_nanos() as u64, Ordering::Relaxed);
    }
    table[4].fetch_add(1, Ordering::Relaxed);
}

fn phases_line(what: &str, table: &[AtomicU64; 5], names: [&str; 3]) -> String {
    let n = table[4].load(Ordering::Relaxed).max(1);
    let us = |k: usize| table[k].load(Ordering::Relaxed) as f64 / n as f64 / 1000.0;
    let (a, b, c, t) = (us(0), us(1) - us(0), us(2) - us(1), us(3));
    format!(
        "    command processor per {what}: {} recorded, {t:.1} us each ({} {a:.1}, {} {b:.1}, {} {c:.1}, recording {:.1})\n",
        table[4].load(Ordering::Relaxed),
        names[0],
        names[1],
        names[2],
        t - us(2)
    )
}

/// Finer command-processor time per draw (DS_PROFILE_GPU) by kind
/// (`DRAW_KINDS`), ns per step of `DRAW_STEP_NAMES`, then the count of draws
/// recorded to the end.
pub(super) static DRAW_STEPS: [[AtomicU64; 19]; 3] = [const { [const { AtomicU64::new(0) }; 19] }; 3];
/// The draws' image-hazard step by kind (DS_PROFILE_GPU): [draws, ns
/// growing sampled depth arrays, ns finding the images and pending work,
/// draws with pending work, ns resolving it].
pub(super) static HAZARD_STEPS: [[AtomicU64; 5]; 3] = [const { [const { AtomicU64::new(0) }; 5] }; 3];
/// Draw kinds of the profile: recorded by the graphics command processor
/// serially, in its own chunk of a parallel run, or by a parallel worker.
pub(super) const DRAW_KINDS: [&str; 3] = ["graphics serial", "graphics own chunks", "graphics workers"];
pub(super) const DRAW_STEP_NAMES: [&str; 18] = [
    "vs prepare", "ps prepare", "shader-read sync", "texture sync", "indices", "color targets", "depth target", "pipeline",
    "image hazards", "reserve", "bind vs", "bind ps", "set alloc", "set write", "rendering instance", "dynamic state",
    "index and draw", "write tracking",
];

/// The same per dispatch by kind (`CS_KINDS`), steps of `CS_STEP_NAMES`
/// and then the count of dispatches recorded to the end.
pub(super) static CS_STEPS: [[AtomicU64; CS_STEP_COUNT + 1]; 8] = [const { [const { AtomicU64::new(0) }; CS_STEP_COUNT + 1] }; 8];
/// Steps of draws and dispatches that were their batch's first command
/// (`place_lines`), by kind (`DRAW_KINDS`, `CS_KINDS`), the count last.
pub(super) static DRAW_STEPS_FIRST: [[AtomicU64; 19]; 3] = [const { [const { AtomicU64::new(0) }; 19] }; 3];
/// The same for draws that opened a GPU-resolved run (a GPU-resolved pair's
/// draw recorded in full, first commands of batches included), by kind:
/// they are in `DRAW_STEPS` and `DRAW_STEPS_FIRST` too.
pub(super) static DRAW_STEPS_GPU: [[AtomicU64; 19]; 3] = [const { [const { AtomicU64::new(0) }; 19] }; 3];
pub(super) static CS_STEPS_FIRST: [[AtomicU64; CS_STEP_COUNT + 1]; 8] = [const { [const { AtomicU64::new(0) }; CS_STEP_COUNT + 1] }; 8];
const CS_STEP_COUNT: usize = 20;
pub(super) const CS_STEP_NAMES: [&str; CS_STEP_COUNT] = [
    "prepare", "shader-read sync", "texture sync", "image hazards", "pipeline", "reserve", "bind", "sgpr data", "set alloc",
    "image notes", "set write", "shadow copies", "dispatch", "shadows after", "mirrors", "write tracking", "argument merge",
    "indirect checks", "indirect barrier", "forwarding",
];
/// Dispatch kinds of the profile: direct or indirect (arguments read by the
/// GPU), by the graphics command processor, its parallel workers or a
/// compute queue's command processor (`Exec::cs_kind`).
const CS_KINDS: [&str; 8] = ["graphics direct", "graphics indirect", "compute direct", "compute indirect", "graphics workers direct", "graphics workers indirect", "queue 0x50 direct", "queue 0x50 indirect"];

thread_local! {
    /// The dispatch being executed is an indirect one whose arguments the
    /// command processor read (`set_cpu_indirect`; profiling kinds).
    static CPU_INDIRECT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// The command processor executes an indirect dispatch with the arguments it
/// read (true) or is done with it (false).
pub fn set_cpu_indirect(on: bool) {
    CPU_INDIRECT.with(|c| c.set(on));
}

/// Buffer bindings made as device addresses (`tune addresses`).
pub static ADDRESSED: AtomicU64 = AtomicU64::new(0);

/// Dispatches with no groups (left out before their preparation): [direct,
/// indirect with the arguments read by the CPU].
pub static ZERO_GROUPS: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// The bind step (`Exec::bind_stage`) by part (DS_PROFILE_GPU; TSC cycles):
/// per buffer binding `BIND_STEP_NAMES[..7]`, then per stage the images and
/// samplers, then the counts of buffer bindings and of stages.
/// Per place class (`PLACE_CLASSES`: a fresh batch's first commands cost
/// several times a later one's bindings, `place_lines`).
pub(super) static BIND_STEPS_BY: [[AtomicU64; 13]; 12] = [const { [const { AtomicU64::new(0) }; 13] }; 12];
/// Parts of a buffer binding (`BindClock` marks 0-9); then images and
/// samplers per stage (10), buffer bindings (11) and stages (12).
const BIND_STEP_NAMES: [&str; 10] = ["unbounded size", "homes and mirrors", "resident shadows", "snapshot", "mirror", "batch-start copy", "mapping span", "host binding", "written-buffer shadows", "descriptor and rest"];

/// Place classes of the bind-step and batch-start copy tables (`place_class`).
const PLACE_CLASSES: [&str; 12] = [
    "graphics command processor, later commands",
    "graphics command processor, first command of a batch",
    "graphics command processor, 2nd-32nd commands",
    "compute queues but 0x50, later commands",
    "compute queues but 0x50, first command of a batch",
    "compute queues but 0x50, 2nd-32nd commands",
    "parallel workers, later commands",
    "parallel workers, first command of a batch",
    "parallel workers, 2nd-32nd commands",
    "queue 0x50, later commands",
    "queue 0x50, first command of a batch",
    "queue 0x50, 2nd-32nd commands",
];

/// The place class of a command recorded as the `place`-th (from 0) of its
/// batch by the graphics command processor (`kind` 0), a compute queue's
/// but 0x50's (1), a parallel worker (2) or queue 0x50's (3).
fn place_class(kind: u8, place: usize) -> u8 {
    3 * kind
        + match place {
            0 => 1,
            1..=31 => 2,
            _ => 0,
        }
}

/// Stages (calls) of a place class a thread adds to the shared tables at a
/// time: the rarer classes at once.
fn flush_every(class: usize) -> u64 {
    match class % 3 {
        1 => 1,
        2 => 32,
        _ => 256,
    }
}

thread_local! {
    /// This thread's BIND_STEPS_BY not yet added.
    static BIND_LOCAL: std::cell::Cell<[[u64; 13]; 12]> = const { std::cell::Cell::new([[0; 13]; 12]) };
    /// The place class of the command being bound (`place_class`).
    static BIND_PLACE: std::cell::Cell<u8> = const { std::cell::Cell::new(0) };
}

/// TSC lap clock of the bind step's parts (profiling only; ~10 ns a mark).
/// Dropped at the end of the stage: the time since its last mark counts as
/// images and samplers.
struct BindClock {
    last: u64,
    acc: [u64; 13],
}

impl BindClock {
    fn start() -> Option<BindClock> {
        laps().then(|| BindClock { last: unsafe { core::arch::x86_64::_rdtsc() }, acc: [0; 13] })
    }

    fn mark(c: &mut Option<BindClock>, k: usize) {
        if let Some(c) = c {
            let now = unsafe { core::arch::x86_64::_rdtsc() };
            c.acc[k] += now.wrapping_sub(c.last);
            c.last = now;
        }
    }

    fn count_buffer(c: &mut Option<BindClock>) {
        if let Some(c) = c {
            c.acc[11] += 1;
        }
    }
}

impl Drop for BindClock {
    fn drop(&mut self) {
        let now = unsafe { core::arch::x86_64::_rdtsc() };
        self.acc[10] += now.wrapping_sub(self.last);
        self.acc[12] += 1;
        let k = BIND_PLACE.with(|f| f.get()) as usize;
        BIND_LOCAL.with(|l| {
            let mut v = l.get();
            for (x, a) in v[k].iter_mut().zip(self.acc.iter()) {
                *x += a;
            }
            if v[k][12] >= flush_every(k) {
                for (c, x) in BIND_STEPS_BY[k].iter().zip(v[k].iter()) {
                    c.fetch_add(*x, Ordering::Relaxed);
                }
                v[k] = [0; 13];
            }
            l.set(v);
        });
    }
}

/// Profile lines: the bind step by part per place class (`BIND_STEPS_BY`).
fn bind_steps_line() -> String {
    BIND_STEPS_BY.iter().zip(PLACE_CLASSES).map(|(t, what)| bind_steps_of(t, &format!(" ({what})"))).collect()
}

fn bind_steps_of(steps: &[AtomicU64; 13], what: &str) -> String {
    let v: Vec<u64> = steps.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    if v[12] == 0 {
        return String::new();
    }
    let ns = |cycles: u64, n: u64| cycles as f64 * 1e9 / super::threadcpu::tsc_hz() / n.max(1) as f64;
    let parts: Vec<String> = BIND_STEP_NAMES.iter().enumerate().map(|(k, name)| format!("{name} {:.0}", ns(v[k], v[11]))).collect();
    format!(
        "    bind step by part{what}: {} stages, {} buffer bindings ({:.1} per stage), {:.0} ns per stage; ns per buffer binding: {}; images and samplers {:.0} ns per stage\n",
        v[12],
        v[11],
        v[11] as f64 / v[12] as f64,
        ns(v[..11].iter().sum(), v[12]),
        parts.join(", "),
        ns(v[10], v[12])
    )
}

/// `Exec::pre_copy` by outcome (DS_PROFILE_GPU) per place class
/// (`PLACE_CLASSES`): TSC cycles and calls per `PRE_STEP_NAMES`.
static PRE_STEPS_BY: [[AtomicU64; 12]; 12] = [const { [const { AtomicU64::new(0) }; 12] }; 12];
const PRE_STEP_NAMES: [&str; 6] = ["refused at once", "reused in the batch", "shared copy reused", "copied for the batch", "copied into the shared cache", "refused after looking"];

thread_local! {
    /// This thread's PRE_STEPS_BY not yet added.
    static PRE_LOCAL: std::cell::Cell<[[u64; 12]; 12]> = const { std::cell::Cell::new([[0; 12]; 12]) };
}

/// Times one `pre_copy` call; `k` is its outcome when it returns.
struct PreClock {
    t0: u64,
    k: usize,
}

impl PreClock {
    fn start() -> Option<PreClock> {
        laps().then(|| PreClock { t0: unsafe { core::arch::x86_64::_rdtsc() }, k: 0 })
    }

    fn set(c: &mut Option<PreClock>, k: usize) {
        if let Some(c) = c {
            c.k = k;
        }
    }
}

impl Drop for PreClock {
    fn drop(&mut self) {
        let dt = unsafe { core::arch::x86_64::_rdtsc() }.wrapping_sub(self.t0);
        // (A call outside a stage's binding counts in the class of the
        // stage bound last.)
        let c = BIND_PLACE.with(|f| f.get()) as usize;
        PRE_LOCAL.with(|l| {
            let mut v = l.get();
            v[c][2 * self.k] += dt;
            v[c][2 * self.k + 1] += 1;
            let calls: u64 = (0..6).map(|j| v[c][2 * j + 1]).sum();
            if calls >= 4 * flush_every(c) {
                for (s, x) in PRE_STEPS_BY[c].iter().zip(v[c].iter()) {
                    s.fetch_add(*x, Ordering::Relaxed);
                }
                v[c] = [0; 12];
            }
            l.set(v);
        });
    }
}

/// Profile lines: `pre_copy` by outcome per place class (`PRE_STEPS_BY`).
fn pre_steps_line() -> String {
    PRE_STEPS_BY.iter().zip(PLACE_CLASSES).map(|(t, what)| pre_steps_of(t, &format!(" ({what})"))).collect()
}

fn pre_steps_of(steps: &[AtomicU64; 12], what: &str) -> String {
    let v: Vec<u64> = steps.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    let calls: u64 = (0..6).map(|k| v[2 * k + 1]).sum();
    if calls == 0 {
        return String::new();
    }
    let hz = super::threadcpu::tsc_hz();
    let parts: Vec<String> = PRE_STEP_NAMES
        .iter()
        .enumerate()
        .map(|(k, name)| format!("{name} {} ({:.0} ns each, {:.0} ms)", v[2 * k + 1], v[2 * k] as f64 * 1e9 / hz / v[2 * k + 1].max(1) as f64, v[2 * k] as f64 * 1e3 / hz))
        .collect();
    format!("    batch-start copy lookups by outcome{what}: {calls} calls: {}\n", parts.join(", "))
}

/// Commands by program (`DS_STEP_LAPS=1`): per (executor class, program
/// key) the commands and their time from entry to return, collected per
/// thread and added every PROG_FLUSH commands; `profile reset` clears it.
/// Classes: 0 the graphics CP, 1 its parallel workers, 2 queue 0x50, 3 the
/// other compute queues. Keys: a draw's vertex and pixel programs, a
/// dispatch's program (and 1: indirect).
static PROG_CENSUS: Mutex<Option<HashMap<(u8, u64, u64), [u64; 2]>>> = Mutex::new(None);
const PROG_FLUSH: u32 = 256;
static PROG_FROM: AtomicU64 = AtomicU64::new(0);
/// Bumped by `profile reset`: a thread's batch counted before it is dropped.
static PROG_GEN: AtomicU64 = AtomicU64::new(0);

thread_local! {
    static PROG_LOCAL: std::cell::RefCell<(u64, u32, HashMap<(u8, u64, u64), [u64; 2]>)> = std::cell::RefCell::new((0, 0, HashMap::new()));
}

impl Exec {
    /// This executor's census class.
    fn prog_class(&self) -> u8 {
        if self.deferred {
            1
        } else if self.name == "gpu-acb0x50" {
            2
        } else if self.compute_cp {
            3
        } else {
            0
        }
    }
}

/// A command of program key (a, b) by class `class` took `ns`.
pub(super) fn prog_note(class: u8, a: u64, b: u64, ns: u64) {
    let epoch = PROG_GEN.load(Ordering::Relaxed);
    PROG_LOCAL.with(|l| {
        let mut l = l.borrow_mut();
        if l.0 != epoch {
            l.0 = epoch;
            l.1 = 0;
            l.2.clear();
        }
        let e = l.2.entry((class, a, b)).or_insert([0; 2]);
        e[0] += 1;
        e[1] += ns;
        l.1 += 1;
        if l.1 >= PROG_FLUSH {
            l.1 = 0;
            let mut g = PROG_CENSUS.lock();
            let m = g.get_or_insert_with(HashMap::new);
            for (k, v) in l.2.drain() {
                let e = m.entry(k).or_insert([0; 2]);
                e[0] += v[0];
                e[1] += v[1];
            }
        }
    });
}

/// The census's lines (`laps_report`): the costliest programs per class.
fn prog_census_lines() -> String {
    let g = PROG_CENSUS.lock();
    let Some(m) = g.as_ref() else { return String::new() };
    let flips = super::FRAMES.load(Ordering::Relaxed).saturating_sub(PROG_FROM.load(Ordering::Relaxed)).max(1) as f64;
    let mut out = String::new();
    for (c, name) in ["graphics CP", "parallel workers", "queue 0x50", "other compute queues"].iter().enumerate() {
        let mut v: Vec<(&(u8, u64, u64), &[u64; 2])> = m.iter().filter(|(k, _)| k.0 == c as u8).collect();
        if v.is_empty() {
            continue;
        }
        let all_n: u64 = v.iter().map(|x| x.1[0]).sum();
        let all_ns: u64 = v.iter().map(|x| x.1[1]).sum();
        v.sort_by(|a, b| b.1[1].cmp(&a.1[1]));
        let top: Vec<String> = v
            .iter()
            .take(16)
            .map(|((_, a, b), x)| {
                let what = if *b > 1 { format!("vs {a:#x} ps {b:#x}") } else if *b == 1 { format!("cs {a:#x} indirect") } else { format!("cs {a:#x}") };
                format!("{what} {:.0}/flip at {:.1} us ({:.2} ms)", x[0] as f64 / flips, x[1] as f64 / x[0].max(1) as f64 / 1e3, x[1] as f64 / flips / 1e6)
            })
            .collect();
        out.push_str(&format!(
            "    commands by program, {name}: {:.0} a flip, {:.2} ms a flip ({} programs; counted per thread every {PROG_FLUSH}): {}\n",
            all_n as f64 / flips,
            all_ns as f64 / flips / 1e6,
            v.len(),
            top.join("; ")
        ));
    }
    out
}

/// Lap timer of a command's steps (profiling only).
pub(super) struct Laps {
    last: Option<std::time::Instant>,
    steps: &'static [AtomicU64],
    /// This command's own step times (ns) and the steps' names, for the log
    /// of slow commands (`slow`).
    local: [u64; 24],
    names: &'static [&'static str],
}

/// Commands slower than this are logged with their steps (`Laps::slow`).
const SLOW_NS: u64 = 150_000;
/// Slow commands logged.
static SLOW_LOGGED: AtomicU64 = AtomicU64::new(0);

impl Laps {
    pub(super) fn for_draw(kind: usize, first: bool) -> Laps {
        Laps { last: laps().then(std::time::Instant::now), steps: if first { &DRAW_STEPS_FIRST[kind][..18] } else { &DRAW_STEPS[kind][..18] }, local: [0; 24], names: &DRAW_STEP_NAMES }
    }
    fn for_dispatch(kind: usize, first: bool) -> Laps {
        Laps { last: laps().then(std::time::Instant::now), steps: if first { &CS_STEPS_FIRST[kind][..CS_STEP_COUNT] } else { &CS_STEPS[kind][..CS_STEP_COUNT] }, local: [0; 24], names: &CS_STEP_NAMES }
    }
    /// Adds the time since the last mark to step `k`.
    pub(super) fn mark(&mut self, k: usize) {
        if let Some(t) = self.last {
            let now = std::time::Instant::now();
            let ns = (now - t).as_nanos() as u64;
            self.steps[k].fetch_add(ns, Ordering::Relaxed);
            if k < self.local.len() {
                self.local[k] += ns;
            }
            self.last = Some(now);
        }
    }
    /// A command that took SLOW_NS or more in its steps, after the first
    /// `profile reset` (`DS_STEP_LAPS`): logged with its steps of 5 us or
    /// more (the first 400).
    pub(super) fn slow(&self, queue: &str, what: impl FnOnce() -> String) {
        if self.last.is_none() || MERGE_FROM.load(Ordering::Relaxed) == 0 {
            return;
        }
        let total: u64 = self.local.iter().sum();
        if total < SLOW_NS || SLOW_LOGGED.fetch_add(1, Ordering::Relaxed) >= 400 {
            return;
        }
        let parts: Vec<String> = self.names.iter().zip(self.local.iter()).filter(|&(_, &ns)| ns >= 5_000).map(|(n, &ns)| format!("{n} {:.0}", ns as f64 / 1000.0)).collect();
        crate::info!("gpu[{queue}]: slow command {} at flip {}: {:.0} us ({})", what(), super::FRAMES.load(Ordering::Relaxed), total as f64 / 1000.0, parts.join(", "));
    }
}

/// Command-processor waits for GPU work (all queues): [ns, count] of read
/// checks that waited (CP reads of GPU-written memory), then of batch-slot
/// reuse (the slot's previous batch still running).
pub(super) static WAIT_NS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];
/// The same per queue: name -> [read-check ns, count, slot ns, count].
static QUEUE_WAITS: Mutex<Option<HashMap<String, [u64; 4]>>> = Mutex::new(None);

fn note_queue_wait(queue: &str, kind: usize, d: std::time::Duration) {
    let mut g = QUEUE_WAITS.lock();
    let e = g.get_or_insert_with(HashMap::new).entry(queue.to_string()).or_insert([0; 4]);
    e[2 * kind] += d.as_nanos() as u64;
    e[2 * kind + 1] += 1;
}

/// CP read checks (`sync_for_reads`, DS_PROFILE_GPU): ns in [home staleness,
/// this queue's writes, other queues' writes], then [_, checks, ranges].
pub(super) static SYNC_STEPS: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];
/// Pages of the clean-page cache (`Exec::clean_pages`): 64 KiB, 512 slots.
const CLEAN_SHIFT: u32 = 16;
const CLEAN_SLOTS: usize = 512;

fn sync_steps_line() -> String {
    let w: Vec<u64> = WAIT_NS.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    let waits = format!(
        "    command-processor waits: {} read checks waited {} ms ({:.2} ms each), {} batch-slot reuses waited {} ms ({:.2} ms each)\n",
        w[1],
        w[0] / 1_000_000,
        w[0] as f64 / w[1].max(1) as f64 / 1e6,
        w[3],
        w[2] / 1_000_000,
        w[2] as f64 / w[3].max(1) as f64 / 1e6
    );
    let per_queue = {
        let g = QUEUE_WAITS.lock();
        let mut v: Vec<(String, [u64; 4])> = g.as_ref().map(|m| m.iter().map(|(k, v)| (k.clone(), *v)).collect()).unwrap_or_default();
        v.sort_by(|a, b| (b.1[0] + b.1[2]).cmp(&(a.1[0] + a.1[2])));
        let parts: Vec<String> = v.iter().map(|(q, w)| format!("{q} reads {} ms ({}), slots {} ms ({})", w[0] / 1_000_000, w[1], w[2] / 1_000_000, w[3])).collect();
        format!("    command-processor waits by queue: {}\n", parts.join("; "))
    };
    waits + &per_queue + &sync_steps_line_inner()
}

fn sync_steps_line_inner() -> String {
    let n = SYNC_STEPS[4].load(Ordering::Relaxed).max(1);
    let us = |k: usize| SYNC_STEPS[k].load(Ordering::Relaxed) as f64 / n as f64 / 1000.0;
    format!(
        "    CP read checks: {} ({:.1} ranges each, {:.1} in pages known clean; us each: homes {:.2}, this queue's writes {:.2}, other queues' writes {:.2})\n",
        SYNC_STEPS[4].load(Ordering::Relaxed),
        SYNC_STEPS[5].load(Ordering::Relaxed) as f64 / n as f64,
        SYNC_STEPS[6].load(Ordering::Relaxed) as f64 / n as f64,
        us(0),
        us(1),
        us(2)
    )
}

fn draw_steps_line() -> String {
    let mut s = String::new();
    // Batches' first commands, by kind.
    for (kind, steps) in DRAW_KINDS.iter().zip(DRAW_STEPS_FIRST.iter()) {
        let n = steps[18].load(Ordering::Relaxed);
        if n == 0 {
            continue;
        }
        let total: u64 = steps[..18].iter().map(|c| c.load(Ordering::Relaxed)).sum();
        let parts: Vec<String> = DRAW_STEP_NAMES.iter().zip(steps.iter()).map(|(name, c)| format!("{name} {:.2}", c.load(Ordering::Relaxed) as f64 / n as f64 / 1000.0)).collect();
        s.push_str(&format!("    command processor per draw by step, first command of a batch, {kind} ({n} draws, {:.2} us each): {}\n", total as f64 / n as f64 / 1000.0, parts.join(", ")));
    }
    for (kind, steps) in CS_KINDS.iter().zip(CS_STEPS_FIRST.iter()) {
        let n = steps[CS_STEP_COUNT].load(Ordering::Relaxed);
        if n == 0 {
            continue;
        }
        let parts: Vec<String> = CS_STEP_NAMES.iter().zip(steps.iter()).map(|(name, c)| format!("{name} {:.2}", c.load(Ordering::Relaxed) as f64 / n as f64 / 1000.0)).collect();
        let total: u64 = steps[..CS_STEP_COUNT].iter().map(|c| c.load(Ordering::Relaxed)).sum();
        s.push_str(&format!("    command processor per dispatch by step, first command of a batch, {kind} ({n} dispatches, {:.2} us each): {}\n", total as f64 / n as f64 / 1000.0, parts.join(", ")));
    }
    for (kind, steps) in DRAW_KINDS.iter().zip(DRAW_STEPS.iter()) {
        let n = steps[18].load(Ordering::Relaxed);
        if n == 0 {
            continue;
        }
        let total: u64 = steps[..18].iter().map(|c| c.load(Ordering::Relaxed)).sum();
        let parts: Vec<String> = DRAW_STEP_NAMES.iter().zip(steps.iter()).map(|(name, c)| format!("{name} {:.2}", c.load(Ordering::Relaxed) as f64 / n as f64 / 1000.0)).collect();
        s.push_str(&format!("    command processor per draw by step, {kind} ({n} draws, {:.2} us each): {}\n", total as f64 / n as f64 / 1000.0, parts.join(", ")));
    }
    for (kind, steps) in DRAW_KINDS.iter().zip(DRAW_STEPS_GPU.iter()) {
        let n = steps[18].load(Ordering::Relaxed);
        if n == 0 {
            continue;
        }
        let total: u64 = steps[..18].iter().map(|c| c.load(Ordering::Relaxed)).sum();
        let parts: Vec<String> = DRAW_STEP_NAMES.iter().zip(steps.iter()).map(|(name, c)| format!("{name} {:.2}", c.load(Ordering::Relaxed) as f64 / n as f64 / 1000.0)).collect();
        s.push_str(&format!("    command processor per draw by step, {kind}, opening a GPU-resolved run ({n} draws, {:.2} us each; in the lines above too): {}\n", total as f64 / n as f64 / 1000.0, parts.join(", ")));
    }
    for (kind, h) in DRAW_KINDS.iter().zip(HAZARD_STEPS.iter()) {
        let v: Vec<u64> = h.iter().map(|c| c.load(Ordering::Relaxed)).collect();
        if v[0] == 0 {
            continue;
        }
        let us = |x: u64| x as f64 / v[0] as f64 / 1000.0;
        s.push_str(&format!(
            "    image-hazard step, {kind}: grow sampled depth {:.2} us, find images and pending work {:.2} us, pending at {:.1}% of draws, resolving {:.2} us (per draw)\n",
            us(v[1]),
            us(v[2]),
            v[3] as f64 * 100.0 / v[0] as f64,
            us(v[4])
        ));
    }
    for (kind, steps) in CS_KINDS.iter().zip(CS_STEPS.iter()) {
        let n = steps[CS_STEP_COUNT].load(Ordering::Relaxed);
        if n == 0 {
            continue;
        }
        let total: u64 = steps[..CS_STEP_COUNT].iter().map(|c| c.load(Ordering::Relaxed)).sum();
        let cs: Vec<String> = CS_STEP_NAMES.iter().zip(steps.iter()).map(|(name, c)| format!("{name} {:.2}", c.load(Ordering::Relaxed) as f64 / n as f64 / 1000.0)).collect();
        s.push_str(&format!("    command processor per dispatch by step ({kind}: {n}, {:.1} us each): {}\n", total as f64 / n as f64 / 1000.0, cs.join(", ")));
    }
    s
}

/// GPU time inside batches (first to last command), ns.
static BATCH_NS: AtomicU64 = AtomicU64::new(0);

/// GPU time inside batches so far, ns (batch stamps: `DS_PROFILE_GPU=1`).
pub fn batch_ns() -> u64 {
    BATCH_NS.load(Ordering::Relaxed)
}
/// GPU idle time between consecutive batches (one Vulkan queue runs them in
/// submission order): (end timestamp of the last batch, per queue of the
/// batch that ended the gap: [ns, gaps, gaps over 1 ms]).
static IDLE: Mutex<(u64, Option<HashMap<String, [u64; 3]>>)> = Mutex::new((0, None));

/// Batch GPU time by the batch's most frequent command (DS_PROFILE_GPU with
/// batch stamps only, which do not serialize the GPU): (queue, shader key)
/// -> [batches, ns, commands].
static BATCH_BY: Mutex<Option<HashMap<(String, u64), [u64; 3]>>> = Mutex::new(None);

/// A batch's most frequent command (draws by pixel shader with DRAW_KEY; 0:
/// none), the key its GPU time is counted under.
fn batch_key(shaders: &[u64]) -> u64 {
    let mut counts: HashMap<u64, u32> = HashMap::new();
    for &a in shaders {
        *counts.entry(a).or_default() += 1;
    }
    counts.iter().max_by_key(|(a, n)| (**n, std::cmp::Reverse(**a))).map_or(0, |(a, _)| *a)
}

/// The name of a batch key (`batch_key`) in reports.
pub(super) fn batch_key_name(a: u64) -> String {
    if a & DRAW_KEY != 0 {
        format!("draws {:#x}", a & !DRAW_KEY)
    } else if a == 0 {
        "(no commands)".to_string()
    } else {
        format!("cs {a:#x}")
    }
}

fn note_batch_by(queue: &str, shaders: &[u64], ns: u64) {
    let key = batch_key(shaders);
    let mut g = BATCH_BY.lock();
    let e = g.get_or_insert_with(HashMap::new).entry((queue.to_string(), key)).or_insert([0; 3]);
    e[0] += 1;
    e[1] += ns;
    e[2] += shaders.len() as u64;
}

fn batch_by_lines() -> String {
    let g = BATCH_BY.lock();
    let Some(m) = g.as_ref() else { return String::new() };
    let mut v: Vec<(&(String, u64), &[u64; 3])> = m.iter().collect();
    v.sort_by(|a, b| b.1[1].cmp(&a.1[1]));
    let total: u64 = v.iter().map(|x| x.1[1]).sum();
    let mut s = format!("    batch GPU time by the batch's most frequent command ({} ms in all):\n", total / 1_000_000);
    for ((q, a), x) in v.iter().take(25) {
        let what = if *a & DRAW_KEY != 0 { format!("draws {:#x}", a & !DRAW_KEY) } else if *a == 0 { "(no commands)".to_string() } else { format!("cs {a:#x}") };
        s.push_str(&format!("      {q} {what}: {} batches, {} ms, {:.1} commands and {:.0} us each\n", x[0], x[1] / 1_000_000, x[2] as f64 / x[0].max(1) as f64, x[1] as f64 / x[0].max(1) as f64 / 1000.0));
    }
    s
}

/// A batch's commands for the timeline: its most frequent shaders (draws
/// by pixel shader, "d"), with counts.
fn batch_contents(shaders: &[u64]) -> String {
    let mut counts: Vec<(u64, u32)> = Vec::new();
    for &a in shaders {
        match counts.iter_mut().find(|c| c.0 == a) {
            Some(c) => c.1 += 1,
            None => counts.push((a, 1)),
        }
    }
    counts.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let parts: Vec<String> = counts.iter().take(4).map(|&(a, n)| if a & DRAW_KEY != 0 { format!("d{:x}x{n}", a & !DRAW_KEY) } else { format!("{a:x}x{n}") }).collect();
    if parts.is_empty() {
        String::new()
    } else {
        format!(": {}{}", parts.join(" "), if counts.len() > 4 { " .." } else { "" })
    }
}

/// Reads a completed batch's timestamps: its span on the GPU (`bq`) for the
/// hitch reports, and with DS_PROFILE_GPU its batch, prologue and dispatch
/// times (`pool`) into the profile.
fn collect_timestamps(bq: vk::QueryPool, pool: vk::QueryPool, shaders: &[u64], queue: &str, seq: u64, segs: &[SegMark], ops: &[u64; OPS]) {
    if bq == vk::QueryPool::null() {
        return;
    }
    let c = ctx();
    let mut be = [0u64; 2];
    let mut pre = [0u64; 2];
    let got_be = unsafe { c.device.get_query_pool_results(bq, BQ_BATCH, &mut be, vk::QueryResultFlags::TYPE_64) }.is_ok();
    let got_pre = unsafe { c.device.get_query_pool_results(bq, BQ_PRE, &mut pre, vk::QueryResultFlags::TYPE_64) }.is_ok();
    if got_be {
        let start = if got_pre && pre[0] != 0 && pre[0] < be[0] { pre[0] } else { be[0] };
        super::hitch::gpu_batch(queue, seq, start, be[1], shaders.len() as u32, batch_key(shaders), || {
            // Its leading commands and its other operations (copies, fills,
            // merges and their sizes).
            let o = ops_line(ops);
            if o.is_empty() { batch_contents(shaders) } else { format!("{}; {o}", batch_contents(shaders)) }
        });
        // The timeline's GPU track, from the batch timestamps (with or
        // without the profiler).
        if super::timeline::on() {
            super::timeline::gpu(queue, seq, be[0], be[1], &format!(", {} commands{}", shaders.len(), batch_contents(shaders)));
            if got_pre && pre[0] != 0 {
                super::timeline::gpu(queue, seq, pre[0], pre[1], " prologue");
            }
        }
    }
    if pool == vk::QueryPool::null() {
        return;
    }
    if got_be {
        let period = c.limits.timestamp_period as f64;
        let batch_ns = (be[1].saturating_sub(be[0]) as f64 * period) as u64;
        BATCH_NS.fetch_add(batch_ns, Ordering::Relaxed);
        if !command_stamps() {
            note_batch_by(queue, shaders, batch_ns);
        }
        let mut g = IDLE.lock();
        if g.0 != 0 && be[0] > g.0 {
            let ns = ((be[0] - g.0) as f64 * period) as u64;
            // Gaps over 1 s: the profile was idle (menus, loading), not counted.
            if ns < 1_000_000_000 {
                let e = g.1.get_or_insert_with(HashMap::new).entry(queue.to_string()).or_insert([0; 3]);
                e[0] += ns;
                e[1] += 1;
                e[2] += (ns >= 1_000_000) as u64;
            }
        }
        g.0 = g.0.max(be[1]);
        drop(g);
        if seg_stamps() {
            collect_segments(pool, shaders, segs, ops, be, queue);
        }
    }
    if got_pre {
        PRE_NS.fetch_add((pre[1].saturating_sub(pre[0]) as f64 * c.limits.timestamp_period as f64) as u64, Ordering::Relaxed);
    }
    let n = (shaders.len() as u32).min(MAX_TIMED);
    if n == 0 || !command_stamps() {
        return;
    }
    // Only some commands are stamped (`DS_PROFILE_GPU_STAMPS=big`): each
    // result with its availability (ash takes the query count from the
    // slice and the stride from its element: one [value, available] pair
    // per query), the commands without stamps skipped.
    let some = stamp_mode() == 2;
    let mut data = vec![[0u64; 2]; (STAMPS * n) as usize];
    let r = if some {
        unsafe { c.device.get_query_pool_results(pool, 0, &mut data, vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WITH_AVAILABILITY) }
    } else {
        let mut plain = vec![0u64; (STAMPS * n) as usize];
        let r = unsafe { c.device.get_query_pool_results(pool, 0, &mut plain, vk::QueryResultFlags::TYPE_64) };
        for (d, v) in data.iter_mut().zip(plain) {
            *d = [v, 1];
        }
        r
    };
    if r.is_err() && !(some && r == Err(vk::Result::NOT_READY)) {
        return;
    }
    let period = c.limits.timestamp_period as f64;
    let ns = |a: u64, b: u64| (b.saturating_sub(a) as f64 * period) as u64;
    let stamp = |i: usize, k: usize| -> Option<u64> {
        let [v, available] = data[STAMPS as usize * i + k];
        (available != 0).then_some(v)
    };
    let mut g = SHADER_TIME.lock();
    let m = g.get_or_insert_with(HashMap::new);
    for (i, &a) in shaders.iter().take(n as usize).enumerate() {
        let (Some(q0), Some(q1), Some(q2), Some(q3)) = (stamp(i, 0), stamp(i, 1), stamp(i, 2), stamp(i, 3)) else { continue };
        let e = m.entry(a).or_insert([0; 4]);
        e[0] += 1;
        e[1] += ns(q1, q2);
        e[2] += ns(q0, q1);
        e[3] += ns(q2, q3);
        if i > 0 {
            if let Some(prev) = stamp(i - 1, STAMPS as usize - 1) {
                GAP_NS.fetch_add(ns(prev, q0), Ordering::Relaxed);
            }
        }
    }
}

/// Flips since the last `profile reset`.
pub fn profile_flips() -> u64 {
    super::FRAMES.load(Ordering::Relaxed).saturating_sub(MERGE_FROM.load(Ordering::Relaxed))
}

/// Clears the dispatch timings (`profile reset` control command).
pub fn reset_shader_profile() {
    draw::reset_draw_stats();
    super::drawcensus::reset();
    super::cp::reset_draw_state();
    if let Some(m) = SHADER_TIME.lock().as_mut() {
        m.clear();
    }
    for c in &VK_OPS {
        c.store(0, Ordering::Relaxed);
    }
    *BARRIER_SITES.lock() = None;
    *BARRIER_QUEUE_SITES.lock() = None;
    *CS_CPU.lock() = None;
    *SHADER_PLACEMENTS.lock() = None;
    *PS_TESTS.lock() = None;
    *SEG_TIME.lock() = None;
    *MERGE_BY.lock() = None;
    *FILL_BY.lock() = None;
    *FLUSH_SITES.lock() = None;
    MERGE_FROM.store(super::FRAMES.load(Ordering::Relaxed), Ordering::Relaxed);
    *SEG_EMPTY.lock() = None;
    *SEG_PAIRS.lock() = None;
    *SEG_ENDS.lock() = None;
    for c in &SEG_TOTAL {
        c.store(0, Ordering::Relaxed);
    }
    shaders::reset_repeats();
    *QUEUE_WAITS.lock() = None;
    {
        let mut g = PRE_REPEAT.lock();
        g.2 = [0; 4];
        g.3 = [[0; 2]; 4];
    }
    PRE_STABLE.lock().1 = [0; 5];
    *PRE_UNCHANGED_WHY.lock() = [0; 10];
    PRE_CONTENT.lock().1 = [[[0; 2]; 3]; 3];
    *PRE_BIG.lock() = (super::FRAMES.load(Ordering::Relaxed), None);
    for c in BEGIN_STATS.iter().chain(OFFTHREAD_RESETS.iter()).chain(SUBMIT_STATS.iter()).chain(SUBMIT_STEPS.iter()).chain(SUBMIT_STEPS_RUN.iter()) {
        c.store(0, Ordering::Relaxed);
    }
    *BATCH_BY.lock() = None;
    for c in READ_WAIT_WORDS.iter() {
        c.store(0, Ordering::Relaxed);
    }
    for c in crate::mem::WRITE_WATCH_STATS.iter() {
        c.store(0, Ordering::Relaxed);
    }
    crate::mem::ww_reset();
    *PROG_CENSUS.lock() = None;
    PROG_GEN.fetch_add(1, Ordering::Relaxed);
    PROG_FROM.store(super::FRAMES.load(Ordering::Relaxed), Ordering::Relaxed);
    for c in DRAW_STEPS.iter().flatten().chain(DRAW_STEPS_GPU.iter().flatten()).chain(HAZARD_STEPS.iter().flatten()).chain(CS_STEPS.iter().flatten()).chain(DRAW_STEPS_FIRST.iter().flatten()).chain(CS_STEPS_FIRST.iter().flatten()).chain(SYNC_STEPS.iter()).chain(WAIT_NS.iter()).chain(BIND_STEPS_BY.iter().flatten()).chain(PRE_STEPS_BY.iter().flatten()) {
        c.store(0, Ordering::Relaxed);
    }
    for c in CS_PHASES.iter().chain(DRAW_PHASES.iter()).chain(PLACE_PHASES.iter().flatten().flatten()) {
        c.store(0, Ordering::Relaxed);
    }
    for (r, c) in SYNC_WHY_AT_RESET.iter().zip(SYNC_WHY.iter()) {
        r.store(c.load(Ordering::Relaxed), Ordering::Relaxed);
    }
    GAP_NS.store(0, Ordering::Relaxed);
    PRE_NS.store(0, Ordering::Relaxed);
    BATCH_NS.store(0, Ordering::Relaxed);
    IDLE.lock().1 = None;
    *STARVED.lock() = None;
    *SUBMIT_WHY.lock() = None;
}

/// `DS_STEP_LAPS=1` without the profiler: the step lines alone (the
/// profiler's report has them otherwise).
pub fn laps_report() -> String {
    if !laps() || profiling() {
        return String::new();
    }
    draw_steps_line() + &bind_steps_line() + &pre_steps_line() + &prog_census_lines()
}

/// Shaders by total GPU time.
pub fn shader_profile() -> String {
    let g = SHADER_TIME.lock();
    // Without per-command stamps (`command_stamps`) the map stays empty;
    // the batch times, idle gaps and CP statistics are still reported.
    let empty = HashMap::new();
    let m = match g.as_ref() {
        Some(m) => m,
        None if profiling() => &empty,
        None => return String::new(),
    };
    let mut v: Vec<(&u64, &[u64; 4])> = m.iter().collect();
    v.sort_by(|a, b| (b.1[1] + b.1[2] + b.1[3]).cmp(&(a.1[1] + a.1[2] + a.1[3])));
    let tot = |k: usize, draws: bool| m.iter().filter(|(a, _)| (**a & DRAW_KEY != 0) == draws).map(|(_, x)| x[k]).sum::<u64>() / 1_000_000;
    let mut s = format!(
        "    gpu time: batches {} ms; dispatches {} ms, draws {} ms, shadow copies in {} ms / out {} ms, between timed commands {} ms; batch-start copies {} ms\n",
        BATCH_NS.load(Ordering::Relaxed) / 1_000_000,
        tot(1, false),
        tot(1, true),
        tot(2, false),
        tot(3, false),
        GAP_NS.load(Ordering::Relaxed) / 1_000_000,
        PRE_NS.load(Ordering::Relaxed) / 1_000_000
    );
    if seg_stamps() {
        s.push_str(&seg_lines());
    }
    s.push_str(&merge_by_lines(super::FRAMES.load(Ordering::Relaxed).saturating_sub(MERGE_FROM.load(Ordering::Relaxed))));
    s.push_str(&fill_by_lines(super::FRAMES.load(Ordering::Relaxed).saturating_sub(MERGE_FROM.load(Ordering::Relaxed))));
    s.push_str(&flush_site_lines(super::FRAMES.load(Ordering::Relaxed).saturating_sub(MERGE_FROM.load(Ordering::Relaxed))));
    s.push_str(&batch_by_lines());
    s.push_str(&begin_stats_line());
    {
        let v: Vec<u64> = VK_OPS.iter().map(|c| c.load(Ordering::Relaxed)).collect();
        s.push_str(&format!(
            "    vulkan commands recorded: {} global barriers, {} render pass begins, {} graphics and {} compute pipeline binds, {} descriptor pushes or set binds\n",
            v[VK_BARRIERS], v[VK_RENDER_PASSES], v[VK_GFX_PIPELINES], v[VK_CS_PIPELINES], v[VK_DESCRIPTORS]
        ));
        if let Some(m) = BARRIER_SITES.lock().as_ref() {
            let mut sites: Vec<(&(&str, u32), &u64)> = m.iter().collect();
            sites.sort_by(|a, b| b.1.cmp(a.1));
            let list: Vec<String> = sites.iter().take(12).map(|((f, l), n)| format!("{}:{l} {n}", f.rsplit(['/', '\\']).next().unwrap_or(f))).collect();
            s.push_str(&format!("    global barriers by site: {}\n", list.join(", ")));
        }
        if let Some(m) = BARRIER_QUEUE_SITES.lock().as_ref() {
            for (q, name) in ["graphics CP", "compute queues", "queue 0x50", "workers"].iter().enumerate() {
                let mut sites: Vec<(&(u8, &str, u32), &u64)> = m.iter().filter(|(k, _)| k.0 == q as u8).collect();
                sites.sort_by(|a, b| b.1.cmp(a.1));
                let all: u64 = sites.iter().map(|x| *x.1).sum();
                let list: Vec<String> = sites.iter().take(10).map(|((_, f, l), n)| format!("{}:{l} {n}", f.rsplit(['/', '\\']).next().unwrap_or(f))).collect();
                s.push_str(&format!("    batch barriers of the {name}: {all}; by site: {}\n", list.join(", ")));
            }
        }
    }
    if let Some(idle) = IDLE.lock().1.as_ref() {
        let mut v: Vec<(&String, &[u64; 3])> = idle.iter().collect();
        v.sort_by(|a, b| b.1[0].cmp(&a.1[0]));
        let total: u64 = v.iter().map(|x| x.1[0]).sum();
        let list: Vec<String> = v.iter().map(|(q, x)| format!("{q} {} ms ({} gaps, {} over 1 ms)", x[0] / 1_000_000, x[1], x[2])).collect();
        s.push_str(&format!("    gpu idle between batches {} ms, before batches of: {}\n", total / 1_000_000, list.join(", ")));
    }
    s.push_str(&phases_line("draw", &DRAW_PHASES, ["prepare and shader-read waits", "textures, targets, pipeline", "bindings"]));
    s.push_str(&draw_steps_line());
    s.push_str(&super::cp::draw_state_line());
    s.push_str(&draw::draw_diff_line());
    s.push_str(&draw::again_line());
    s.push_str(&draw::prep_repeat_line());
    s.push_str(&bind_steps_line());
    s.push_str(&pre_steps_line());
    s.push_str(&pre_repeat_line());
    s.push_str(&crate::mem::ww_lines());
    s.push_str(&sync_steps_line());
    s.push_str(&shaders::repeats_line());
    s.push_str(&phases_line("dispatch", &CS_PHASES, ["prepare and waits", "textures, pipeline", "bindings"]));
    s.push_str(&place_lines());
    s.push_str(&cs_cpu_lines());
    s.push_str(&cs_cpu_queue_lines());
    {
        let n: Vec<u64> = SYNC_WHY.iter().zip(SYNC_WHY_AT_RESET.iter()).map(|(c, r)| c.load(Ordering::Relaxed).saturating_sub(r.load(Ordering::Relaxed))).collect();
        s.push_str(&format!(
            "    command-processor waits for GPU work: {} CP reads, {} dispatch shader reads, {} dispatch textures, {} draw shader reads, {} draw textures, {} draw indices; speculation (since boot): on words the recording batch writes {}; refused: stale home {}, data this batch writes or not one mapping {}, a parallel chunk's own writes {}, a stage writing memory or images {}, mispredicting shader {}, descriptor words (`spec_reads 2`) {}\n",
            n[0],
            n[1],
            n[2],
            n[3],
            n[4],
            n[5],
            SPEC_OWN.load(Ordering::Relaxed),
            SPEC_REFUSED[0].load(Ordering::Relaxed),
            SPEC_REFUSED[1].load(Ordering::Relaxed),
            SPEC_REFUSED[2].load(Ordering::Relaxed),
            SPEC_REFUSED[3].load(Ordering::Relaxed),
            SPEC_REFUSED[4].load(Ordering::Relaxed),
            SPEC_REFUSED[5].load(Ordering::Relaxed)
        ));
        let w: Vec<u64> = READ_WAIT_WORDS.iter().map(|c| c.load(Ordering::Relaxed)).collect();
        s.push_str(&format!(
            "    shader-read waits whose words were unchanged after the wait: draws {} of {}, dispatches {} of {}; commands that did not wait {} ({} data loads copied by batch prologues), stages used as prepared {} (checks held {}, failed {})\n",
            w[0],
            w[0] + w[1],
            w[2],
            w[2] + w[3],
            SPEC_STATS[3].load(Ordering::Relaxed),
            SPEC_STATS[4].load(Ordering::Relaxed),
            SPEC_STATS[0].load(Ordering::Relaxed),
            SPEC_STATS[1].load(Ordering::Relaxed),
            SPEC_STATS[2].load(Ordering::Relaxed)
        ));
    }
    if let Some(w) = SUBMIT_WHY.lock().as_ref() {
        let mut v: Vec<(&(String, &'static str), &u64)> = w.iter().collect();
        v.sort_by(|a, b| b.1.cmp(a.1));
        let list: Vec<String> = v.iter().map(|((q, why), n)| format!("{q} {why} {n}")).collect();
        s.push_str(&format!("    batches by what ended them: {} (command-buffer ends left open: {})\n", list.join(", "), super::cp::CMDBUF_MERGED.load(Ordering::Relaxed)));
    }
    if let Some(st) = STARVED.lock().as_ref() {
        let mut v: Vec<(&String, &[u64; 3])> = st.iter().collect();
        v.sort_by(|a, b| b.1[1].cmp(&a.1[1]));
        let list: Vec<String> = v.iter().map(|(q, x)| format!("{q} {} submissions, idle {} ms, recorded in {} ms", x[0], x[1] / 1_000_000, x[2] / 1_000_000)).collect();
        s.push_str(&format!("    submissions to an idle GPU: {}\n", list.join(", ")));
    }
    if let Some(l) = ps_tests_line() {
        s.push_str(&l);
    }
    let top = std::env::var("DS_PROFILE_TOP").ok().and_then(|x| x.parse().ok()).unwrap_or(15);
    for (a, x) in v.iter().take(top) {
        let n = x[0].max(1);
        if **a & DRAW_KEY != 0 {
            let tests = PS_TESTS.lock().as_ref().and_then(|m| m.get(&(**a & !DRAW_KEY)).copied()).map(|t| format!(" (depth tests before as asked {}, as allowed {}, after {} / {})", t[0], t[1], t[2], t[3])).unwrap_or_default();
            s.push_str(&format!("      draws with shader {:#x}: {} draws, {} us each{tests}\n", **a & !DRAW_KEY, x[0], x[1] / 1000 / n));
            if let Some(l) = placement_line(**a) {
                // Buffers of the vertex stage, then of the pixel stage.
                let rest = l.split_once("; buffers:").map(|(_, r)| r).unwrap_or("");
                s.push_str(&format!("        buffers (vertex stage first):{rest}"));
                s.push('\n');
            }
        } else {
            s.push_str(&format!("      cs {a:#x}: {} dispatches, {} us each (+ copies {} / {} us)\n", x[0], x[1] / 1000 / n, x[2] / 1000 / n, x[3] / 1000 / n));
            if let Some(l) = placement_line(**a) {
                s.push_str(&l);
                s.push('\n');
            }
        }
    }
    s
}

pub struct Exec {
    /// The queue families of the slots' command pools (`Slot::new`): this
    /// queue's, the copy queue's (none without one).
    family: u32,
    xfamily: Option<u32>,
    slots: Vec<Slot>,
    cur: usize,
    /// Slots that could be created (a failed creation caps them).
    slot_limit: usize,
    /// Full-size slot segments from the start (`DS_SLOT_SEGS=full`).
    seg_full: bool,
    /// The sizes this executor's slots' shadow and batch-start copy
    /// segments are made with, the most its batches used since the last
    /// decay check, and the batches since (`SEG_DECAY`).
    seg_want: [u64; 2],
    seg_peak: [u64; 2],
    seg_batches: u32,
    /// The SGPR data ring in video memory could not be allocated.
    vram_ring_failed: bool,
    /// Offsets within the current slot's segment of each ring.
    ring_used: [u64; 2],
    /// Read-only buffer snapshots: space used in the current slot's buffer,
    /// failure to allocate, and the snapshots of the batch being recorded by
    /// range.
    snap_used: u64,
    snap_failed: bool,
    snaps: super::fx::Map<(u64, u64), u64>,
    /// Batch-start copies (`pre_copy`): space used in the current slot's
    /// buffer, whether the slot's prologue is being recorded, allocation
    /// failure, and the copies of the batch by guest range.
    pre_used: u64,
    pre_open: bool,
    pre_failed: bool,
    pres: super::fx::Map<(u64, u64), (vk::Buffer, u64)>,
    /// Pages this batch uploads into the residency mirror (`resident_bind`),
    /// the same as a set, scratch for the pages a binding needs, and the
    /// bytes of the slot's staging used.
    res_pages: Vec<u64>,
    res_mine: super::fx::Set<u64>,
    res_need: Vec<u64>,
    stage_used: u64,
    /// Front end 2 (`fe2`): the batch's uploads into the mirror (runs, and
    /// their pages as a set: current for the batch's work), whether GPU
    /// work recorded since the last write-back marks dwords, and the word
    /// generations taken when the batch is queued (scratch).
    fe2_up: Vec<(u64, u64)>,
    /// The imports (start, buffer) its uploads read (`Span::first_use`).
    fe2_spans: Vec<(u64, vk::Buffer)>,
    fe2_mine: super::fx::Set<u64>,
    fe2_marked: bool,
    fe2_gens: Vec<u32>,
    /// Runs the batch's bindings left to check at its close
    /// (`fe2::resolve`), and their pages (the batch's own meanwhile).
    fe2_pend: Vec<(u64, u64)>,
    fe2_pend_set: super::fx::Set<u64>,
    /// This queue's view (`fe2::view_cell`): what its checks need.
    fe2_view: std::sync::Arc<std::sync::atomic::AtomicU32>,
    /// GPU-resolved chains under front end 2 (`fe2_gpu_ranges`, scratch):
    /// the ranges a command's shader reads, the words read for them.
    f3_ranges: Vec<(u64, u64)>,
    f3_loads: Vec<(u64, u64)>,
    f3_reads: Vec<(u64, u64)>,
    /// Runs `fe2_gpu_ranges` placed in the batch being recorded, with the
    /// epoch their checks needed: current for its later work that needs no
    /// later one (as a run's first command's bindings are for its appends).
    f3_placed: super::fx::Set<(u64, u64, u32)>,
    /// The lowest epoch front end 2's placements needed since the command
    /// being recorded began (u32::MAX: none): what a GPU-resolved run its
    /// first command opens was checked for (`fe2_run_want`).
    fe2_min_want: u32,
    /// Resident GPU-written data (`gpu::rgm`, `exec/rgm.rs`): the open
    /// batch's claims and their granules, its W writes (all, and those not
    /// merged into guest memory yet), the merge table, a page list reused.
    rgm_claims: Vec<super::rgm::Claim>,
    rgm_mine: super::fx::Map<usize, (u16, u64)>,
    rgm_written: super::ranges::Ranges,
    rgm_unmerged: super::ranges::Ranges,
    rgm_table: Option<rgm::RgmTable>,
    rgm_pages: Vec<u64>,
    /// Prologue copies not recorded yet (`tune pre_merge`): (source,
    /// destination, region), recorded grouped by buffers by `end_pre`.
    pre_copies: Vec<(vk::Buffer, vk::Buffer, vk::BufferCopy)>,
    /// The batch's copies for the copy queue (`Slot::xpre`, recorded by
    /// `end_xpre`), and the copy queue's timeline value the batch waits for
    /// besides its own (shared copies earlier batches made there).
    xcopies: Vec<(vk::Buffer, vk::Buffer, vk::BufferCopy)>,
    copy_wait: u64,
    ring_align: u64,
    /// Offset alignment of SGPR data bound as a uniform buffer.
    ubo_align: u64,
    recording: bool,
    /// When the batch being recorded was begun.
    begun_at: std::time::Instant,
    /// Its start on the timeline (`timeline::mark`).
    begun_mark: Option<i64>,
    /// Label values the recording batch writes GPU-side (address, value,
    /// bytes): promised when it is submitted.
    pending_promises: Vec<(u64, u64, u8)>,
    /// Set by a command processor waiting for one of those labels.
    submit_request: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// Guest ranges written by the batch being recorded, and in write order
    /// for the self-checks (`verify`, only when enabled).
    pending: super::ranges::Ranges,
    pending_log: Vec<(u64, u64)>,
    /// Actions of the batch being recorded.
    actions: Vec<Action>,
    checks: Vec<verify::Check>,
    /// Guest ranges written by submitted batches or deferred CP writes, by sequence.
    inflight: super::ranges::SeqRanges,
    /// Last sequence number this queue queued, and the last carrying actions.
    last_seq: u64,
    last_action_seq: u64,
    /// The next submission waits for this sequence (deferred CP writes that
    /// later GPU work may read).
    submit_after: u64,
    /// CP timestamp writes the GPU performs in the recording batch.
    ts_used: u32,
    /// The recording batch reset its timestamp pool whole (`gpu_timestamp`).
    ts_pool_reset: bool,
    /// End-of-pipe timestamps of the batch whose readback and store wait for
    /// its end (`tune eop_light`).
    ts_deferred: Vec<TsEntry>,
    /// Kept windows (`gpu::keep`) the recording batch took in (ids), wrote
    /// and wrote back.
    kept_taken: Vec<u64>,
    kept_written: Vec<std::sync::Arc<super::keep::Kept>>,
    kept_given: Vec<std::sync::Arc<super::keep::Kept>>,
    /// An end-of-pipe label write in the batch that later work is not
    /// ordered after (`tune eop_light`): a wait passing on it orders it
    /// (`order_after_open_writes`).
    eop_unordered: bool,
    /// Of them, those to compare with the clock once the batch completed.
    ts_check: Vec<u64>,
    /// The recording batch holds CP writes performed by the GPU
    /// (`gpu_cp_write`); the last submitted batch that does.
    gpu_writes_open: bool,
    last_gpu_write_seq: u64,
    /// The last batch whose CP writes the CPU performs after it completed:
    /// this queue's own, and another queue's whose promised label a wait
    /// passed on. CP writes the GPU performs are submitted after those ran.
    cpu_action_seq: u64,
    cpu_dep_seq: u64,
    /// The title synchronized (flush, cache action, end-of-pipe event)
    /// since the last full barrier: the next dispatch or draw waits.
    barrier_wanted: bool,
    /// A draw read its arguments on the GPU since the last full barrier
    /// (`tune draw_args_gpu`): a dispatch writing an argument home waits for
    /// it (the console's command processor read them when it reached the
    /// draw, so the title rewrites them without synchronizing).
    args_read: bool,
    /// A dispatch ran since the last full barrier (draws wait for it).
    dispatched: bool,
    reported: HashMap<String, u64>,
    name: String,
    /// Shaders recorded in the current batch (slow-batch reports).
    batch_shaders: Vec<u64>,
    /// Active dynamic-rendering instance (draws).
    rendering: Option<draw::Rendering>,
    /// Graphics state recorded in the current command buffer (draws skip
    /// the calls that would set the same values again).
    dyncache: draw::DynCache,
    /// Cleared bindings for the next commands (`bind_stage`, `recycle`).
    bound_pool: Vec<Bound>,
    /// Image keys of the command being recorded (`image_hazards`; reused).
    keys_scratch: Vec<super::images::ImageKey>,
    /// Bumped whenever this executor's pending or in-flight writes grow.
    own_gen: u64,
    /// The recording batch's other GPU operations (`ops_line`).
    ops: [u64; OPS],
    /// 64 KiB pages that no pending or in-flight write and no stale home part
    /// overlapped when last checked (direct-mapped by page), valid while the
    /// write generations are `clean_tag` (`sync_for_reads`).
    clean_pages: Vec<u64>,
    clean_tag: [u64; 4],
    /// The reads `sync_for_reads` still checks (the vector reused).
    dirty_reads: Vec<(u64, u64)>,
    /// The written ranges (all, sized) this executor's last dispatch noted in
    /// the image cache, the write tracker and the copy cache, with their
    /// generations after it (`note_gens`): a dispatch writing the same ranges
    /// skips those notes while the generations stay the same. The title's
    /// culls write the same two buffers ~2,700 times per frame and each note
    /// took a global lock (5.3 us per dispatch profiled, run 20261013-021047).
    last_notes: (Vec<(u64, u64)>, Vec<(u64, u64)>, [u64; 4]),
    /// A draw's stage bindings (the vector reused).
    draw_bounds: Vec<Bound>,
    /// Descriptor pool usage of the recording slot (see POOL_LIMITS).
    pool_use: [u32; 6],
    /// Shadow space used in the recording slot.
    shadow_used: u64,
    /// Shadows of this batch whose working copies hold newer data than
    /// guest memory (see Shadow).
    resident: Vec<Shadow>,
    /// The shaders whose reads the next `sync_for` checks (wait traces).
    sync_readers: [u64; 2],
    /// The read a traced wait collided with (the caller explains it).
    pub(super) sync_culprit: Option<(u64, u64)>,
    /// Worker executor of a parallel run (`gpu::par`): batches are closed
    /// for the main command processor to submit in program order
    /// (`submit_closed`), never submitted or waited for here; anything that
    /// would need that stops the chunk (`stopped`) before the command
    /// records its own work.
    deferred: bool,
    /// The frame-ahead recorder's executor (`gpu::frameahead`, `set_held`):
    /// a deferred one whose closed batches keep their label writes (its
    /// later waits pass on them) and which stops before anything the
    /// current frame's work would see (CPU-side effects, image-cache notes,
    /// fast clears and storage images).
    held: bool,
    /// The graphics command processor's executor while it records past a
    /// parallel run (`past_begin`): held, its batches in its own slot
    /// rotation, until `past_commit` or `past_drop`.
    past: bool,
    /// Its closed batches' label writes (`open_promise`).
    held_promises: Vec<(u64, u64, u8)>,
    /// Guest ranges it read on the CPU while recording (stage preparations,
    /// command-processor reads, textures' guest memory), since `held_reset`.
    held_reads: Vec<(u64, u64)>,
    /// Each held read's kind (`HELD_READ_KINDS`), with `DS_TRACE_PAST_DROPS`.
    held_kinds: Vec<u8>,
    /// GPU-performed CP writes recorded while held (diagnostics).
    cp_written: Vec<(u64, u64)>,
    /// Held past a label wait (`past_begin_growing`): a slot still holding
    /// a held batch is passed over for a free or new one (up to
    /// HELD_SLOTS) instead of stopping.
    past_grow: bool,
    /// The oldest epoch front end 2's checks needed since the stretch past
    /// a label wait began (u32::MAX: none; `past_checks_want`).
    past_want_min: u32,
    /// The ranges its checks placed and the pages its batches upload
    /// (`past_current_for`).
    past_ranges: Vec<(u64, u64)>,
    past_up: HashSet<u64>,
    /// A worker records a chunk of a run planned past a label wait
    /// (`tune wait_runs`): its checks as the stretch's (the epoch now, hot
    /// pages uploaded when it runs, its placements kept for the check at
    /// its turn).
    stretch: bool,
    /// Its own image-cache notes (`Images::notes_for`), never in the image
    /// cache before a hand-over: fast clears pending (metadata address,
    /// value when known) and GPU-owned images overwritten by its work.
    held_clears: HashMap<super::images::ImageKey, (u64, Option<u32>)>,
    held_over: HashSet<super::images::ImageKey>,
    /// The images its commands used.
    held_keys: HashSet<super::images::ImageKey>,
    /// It records a long parallel run's chunk (`long_begin`; held too).
    long: bool,
    /// A long chunk's CPU reads served by forwarded copies the run did not
    /// make (`forward::split_long_reads`): checked against what the run's
    /// chunks write otherwise than by the planner's copies and what came
    /// before the run; and those its planner's copies served: against the
    /// chunks' only (such a copy rewrote its range after everything before
    /// the run).
    held_fwd_reads: Vec<(u64, u64)>,
    held_fwd_planned: Vec<(u64, u64)>,
    /// The copies its run's planner forwarded (packet -> words), whether the
    /// dispatch being recorded is one of them (`Cp::dispatch`), what its
    /// chunk writes otherwise, and whether it computed one of them otherwise
    /// than planned (the run's later chunks inherited the planned words).
    long_planned: std::sync::Arc<HashMap<u64, std::sync::Arc<[u32]>>>,
    pub noting_planned: bool,
    long_nonfwd: super::ranges::Ranges,
    long_mismatch: bool,
    /// What its batches write from `rec_begin` to `rec_take` (the graphics
    /// command processor's own part of a long parallel run: its workers'
    /// chunks are checked against it).
    rec_writes: Option<super::ranges::Ranges>,
    /// A worker's: what the main command processor's batch, not yet
    /// submitted when the run was planned, writes (`tune par_main_writes`),
    /// and (mode 3) whether it has been submitted since.
    main_writes: Option<std::sync::Arc<super::ranges::Ranges>>,
    main_submitted: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    pub(crate) stopped: bool,
    /// Why (statistics), and the reason a caller gives a flush that follows.
    pub(crate) stop_why: &'static str,
    flush_why: &'static str,
    closed: Vec<Closed>,
    /// Seals of the held work past a run, oldest first (`past_seal`).
    seals: std::collections::VecDeque<SealParts>,
    /// SPAN_GEN when its oldest batch not yet submitted began (u64::MAX:
    /// none), registered in EXEC_GENS (`reap_spans`).
    open_gen: std::sync::Arc<AtomicU64>,
    /// The graphics queue's family (command processor and parallel
    /// workers): batch-start copies are shared (`gpu::precache`).
    shares_pre: bool,
    /// Shared copies this batch's prologue makes (usable by later batches
    /// once it is submitted).
    shared_pending: Vec<(u64, u64, u64, bool)>,
    /// Ids of the shared copies this batch made of its shadows after writing
    /// them (`share_shadows`): its submission-time invalidation spares them.
    shared_kept: Vec<u64>,
    /// A draw's texture ranges (`texture_ranges_into`), reused per draw.
    tex_scratch: Vec<((u64, u64), super::desc::Tsharp)>,
    /// The last draw's programs and hashes of its buffer resources, image
    /// resources and user data (profiling: `exec::draw::DRAW_DIFFS`).
    last_draw_sig: [u64; 5],
    /// The draw being executed has the last draw's context registers,
    /// programs and primitive type (set by the command processor), and
    /// [batch, `cmd_uses`] right after the last full draw was recorded
    /// (profiling: `exec::draw::CONSEC_DIFFS`).
    pub(crate) draw_same: bool,
    last_draw_end: (u64, u64),
    /// The last draw's GPU-read arguments and index base (`DS_DRAW_STATS`).
    last_draw_where: (Option<u64>, Option<u64>),
    /// The last full draw, when a draw right after it can be recorded on the
    /// state it left (`draw_again`), and the stages of the full draw in
    /// progress as prepared (before binding filled their SGPR data).
    again: Option<draw::Again>,
    again_next: Option<[Option<draw::StageSig>; 2]>,
    /// The open batched draw run (`tune batch_draws`) and the program pair
    /// of the last draws with the same context state and how many in a row.
    draw_run: Option<draw::DrawRun>,
    draw_streak: ((u64, u64), u32),
    /// Speculations on words this batch writes (`speculate`): checked once
    /// it completed, its sequence set at submission.
    own_checks: Vec<SpecCheck>,
    /// Race checks of the batch being recorded (`tune race_checks`).
    race_pending: Vec<RaceCheck>,
    race_id: u64,
    race_name: std::sync::Arc<str>,
    /// Forwarded copies recorded into the batch being recorded (`gpu::forward`).
    forward_ids: Vec<u64>,
    /// The compute queue id in this executor's name (`gpu-acb<id>`; 0: none)
    /// and whether the batch being recorded goes to the main family's second
    /// Vulkan queue (`tune alt_queue`: a timing diagnosis, no ordering
    /// against the other queue; its batches share no batch-start copies
    /// unless bit 16 of the switch is set).
    alt_id: u64,
    /// Every batch goes to the compute-only family's queue (`vk::alt_compute_id`).
    on_compute: bool,
    /// A compute queue's command processor records into it (profile
    /// classes).
    compute_cp: bool,
    /// Queue 0x50's (its own place class in the bind-step profile).
    acb50: bool,
    /// The dispatch whose writes are being noted (`DS_TRACE_NOTE`).
    noting: u64,
    /// A command label is open in the batch's command buffer (`label`).
    label_open: bool,
    batch_alt: bool,
    batch_alt_share: bool,
    /// GPU writes this batch recorded since its last full barrier through
    /// `barrier` (None: more than `SINCE_MAX`): an indirect dispatch whose
    /// arguments none of them writes needs no barrier of its own.
    since_barrier: Option<Vec<(u64, u64)>>,
    /// Since that barrier: the guest ranges dispatches read (None: more
    /// than SINCE_MAX), the images they sampled and stored to, whether
    /// they used the global data share, whether a rendering instance began
    /// (`TITLE_BARRIERS`).
    since_reads: Option<Vec<(u64, u64)>>,
    since_img_r: Vec<vk::Image>,
    since_img_w: Vec<vk::Image>,
    since_gds: bool,
    since_draws: bool,
    /// The batch's barrier segments so far (`seg_stamps`).
    segs: Vec<SegMark>,
    /// The last draw, when another like it can be recorded on the state it
    /// left (`draw_repeat`, `tune draw_repeat`).
    repeat: Option<draw::Repeat>,
    /// Bumped by every `cmd()`: anything recorded into the batch's command
    /// buffer after a draw (the repeat checks it).
    cmd_uses: std::cell::Cell<u64>,
    /// The source line of the last `cmd()` (what was recorded after an open
    /// batched dispatch: `BATCH_MISS_SITES`).
    last_cmd_line: std::cell::Cell<u32>,
    /// Batches begun (the repeat checks it).
    batch_count: u64,
    /// The program of the last direct dispatches and how many in a row
    /// (`tune batch_cs`: a run of BATCH_STREAK makes it batchable).
    cs_streak: (u64, u32),
    /// Programs whose dispatches after a run's first were looked at
    /// (`Cp::streak_after`; once per program).
    cs_peeked: HashSet<u64>,
    /// The open batched dispatch (`tune batch_cs`).
    cs_run: Option<CsRun>,
    /// Mirror mode: the page runs its batch's bindings write (gathered at
    /// its end: `record_mirror_writeback`), bytes used in the slot's last
    /// extra staging buffer, and pages copied without a claim.
    owned: Vec<(u64, u64)>,
    /// Lazy mirror mode: the batch's ownership of its written ranges (made
    /// no-access before it is handed over: `resident::own_begin`).
    mm_own: Option<std::sync::Arc<super::resident::Owned>>,
    extra_used: u64,
    mm_copy: Vec<u64>,
    /// Mirror mode: a parallel worker's batch's write-back (carried by
    /// `Closed`: chunk batches take no other completion actions).
    eager_action: Option<Action>,
    /// Copies recorded into the prologue after its other copies (behind a
    /// barrier): mirror mode's copies of written ranges from the batch's
    /// start (`record_mirror_writeback`).
    pre_after: Vec<(vk::Buffer, vk::Buffer, vk::BufferCopy)>,
    /// Mirror mode: placements (page ranges) that deferred CP writes not yet
    /// performed write into: the batch is submitted after them, and their
    /// pages those writes changed are copied into the mirror at its close.
    mm_cp: Vec<(u64, u64)>,
    /// Mirror mode: the placement in progress binds an unbounded pointer
    /// (statistics, `mm_place_line`).
    mm_unsized: bool,
}

/// The most gather records a batch's output holds (mirror mode; 34 MB).
const MM_RECORDS_MAX: u64 = 1 << 18;

/// Extra staging per buffer (mirror mode, `Slot::stage_extra`).
const EXTRA_SEG: u64 = 64 << 20;

/// Consecutive direct dispatches of a program that make it batchable, each
/// of at most BATCH_SMALL workgroups (per-dispatch costs dominate there;
/// larger dispatches of buffer-heavy programs lose more to the GPU cost of
/// device addresses than batching saves).
pub(super) const BATCH_STREAK: u32 = 4;
pub(super) const BATCH_SMALL: u64 = 64;
/// Programs made batchable at the first dispatch of a run (`Cp::streak_after`)
/// or by a scan ahead that saw the run (`Cp::scan_commands`).
pub static BATCHABLE_AHEAD: AtomicU64 = AtomicU64::new(0);
/// Workgroups one batched dispatch may hold (its table in the SGPR ring:
/// `BATCH_ENTRY_DWORDS` each).
const BATCH_MAX_GROUPS: u32 = 4096;

/// A batched dispatch open for more commands of its program (`Spec::batched`):
/// recorded as an indirect dispatch whose group count, in the SGPR ring,
/// grows with every command appended while nothing else is recorded.
struct CsRun {
    batch: u64,
    /// `cmd_uses` right after it was recorded.
    uses: u64,
    pipeline: vk::Pipeline,
    /// The descriptors it was recorded with (buffers and images).
    descs: Vec<u64>,
    /// Its arguments (x, y, z) and table in the ring (host pointers).
    args: *mut u32,
    table: *mut u32,
    groups: u32,
    /// The ring and its uniform window (`Bound::window`; 0 without one).
    ring: vk::Buffer,
    window: u64,
    /// A GPU-resolved program's run (`gpures`): later commands are appended
    /// with their user data alone (`gpu_append`).
    gpu: Option<Box<GpuRun>>,
}

/// A GPU-resolved run (`CsRun::gpu`): the program, its run roots with the
/// run's values, its per-item roots, the run's first command's SGPR data
/// (the others' differ in their user dwords only), the ring it is in, and
/// the program registers the run was recorded with.
struct GpuRun {
    addr: u64,
    /// The run's first command's preparation (the plan of its translation).
    found: std::sync::Arc<shaders::Found>,
    roots: Vec<(super::gpures::Root, u64)>,
    items: Vec<super::gpures::Root>,
    template: Vec<u32>,
    users: usize,
    which: usize,
    regs: [u32; 6],
    /// Front end 2: the epoch its first command's checks needed
    /// (`Exec::fe2_run_want`); later commands append while no check needs a
    /// later one.
    fe2_want: u32,
}

/// GPU-resolved commands: [appended to a run, refused (run roots differ),
/// refused (other: no run, state, room), pages imported for them].
pub static GPU_APPENDS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];
/// GPU-resolved draws (`draw::draw_gpu_append`): [appended to a run, not
/// appended because run roots differ, indices elsewhere than the run's
/// buffer, other (no run, state, pending image work, room)].
pub static GPU_DRAW_APPENDS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];
/// GPU-resolved commands not appended because a formatted per-item buffer's
/// V# had another format or swizzle than their translation's.
pub static GPU_FORMAT_MISSES: AtomicU64 = AtomicU64::new(0);
/// GPU-resolved draws appended (`DS_STEP_LAPS`): [appends, ns in checks,
/// imports, image hazards, indices, SGPR data and arguments].
pub static GPU_APPEND_LAPS: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];
/// `DS_GPU_VERIFY=1`: [appended draws verified, of them with differences].
pub static GPU_VERIFY: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
/// Front end 2's GPU-resolved chains (stage F3, `Exec::fe2_gpu_ranges`):
/// commands whose per-item chains were not placed, by why (`F3_WHY`).
pub static F3_REFUSED: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];
pub(super) const F3_VIEW: usize = 0;
pub(super) const F3_PENDING: usize = 1;
pub(super) const F3_UNREADABLE: usize = 2;
pub(super) const F3_FORMAT: usize = 3;
pub(super) const F3_OUTSIDE: usize = 4;
pub(super) const F3_PLACE: usize = 5;
pub(super) const F3_WHY: [&str; 6] = ["the view moved since the run's first command", "words of the chains pending GPU or CP writes", "a chain or range unreadable", "a formatted buffer's format", "outside the mirror's span", "a placement failed"];
/// [ranges placed for them, their bytes; draws, dispatches recorded with
/// CPU-resolved chains after a refusal].
pub static F3_PLACED: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];
/// `DS_STEP_LAPS`: [stages placed, TSC cycles reading their chains, checking
/// the loads' imports and pending writes, placing the buffers; ranges found,
/// runs placed, runs the batch uploads or placed already].
pub static F3_LAPS: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];
/// Runs of at most this many pages pass `fe2_gpu_ranges` without a
/// placement when the batch uploads every page of them already.
const F3_MINE_PAGES: u64 = 4;
/// Runs a batch remembers as placed (`Exec::f3_placed`).
const F3_PLACED_MAX: usize = 4096;

/// `DS_F3_REGIONS=1` (diagnosis): per GPU-resolved stage program, its F3
/// calls (`Exec::fe2_gpu_ranges`) and the guest mappings its per-item chains
/// reach: the buffers placed in the mirror, the per-item blocks the shader
/// loads through the page table (FRONTEND2.md "F3": step two's regions).
fn f3_regions_on() -> bool {
    static W: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *W.get_or_init(|| std::env::var("DS_F3_REGIONS").ok().as_deref() == Some("1"))
}

#[derive(Default)]
struct F3Regions {
    calls: u64,
    /// The mappings [start, end) reached by buffers, by loads.
    bufs: HashSet<(u64, u64)>,
    loads: HashSet<(u64, u64)>,
    /// Ranges seen, by kind, and those outside any mapping.
    buf_ranges: u64,
    load_ranges: u64,
    unmapped: u64,
}

static F3_REGIONS: Mutex<Option<HashMap<u64, F3Regions>>> = Mutex::new(None);

/// A GPU-resolved stage of program `addr` with per-item `loads` and `bufs`
/// (`f3_regions_on`).
fn note_f3_regions(addr: u64, loads: &[(u64, u64)], bufs: &[(u64, u64)]) {
    let mut g = F3_REGIONS.lock();
    let e = g.get_or_insert_with(HashMap::new).entry(addr).or_default();
    e.calls += 1;
    for (list, is_buf) in [(bufs, true), (loads, false)] {
        for &(a, n) in list {
            if a == 0 || n == 0 {
                continue;
            }
            match crate::mem::vma_bounds(a) {
                Some(m) => {
                    if is_buf {
                        e.buf_ranges += 1;
                        e.bufs.insert(m);
                    } else {
                        e.load_ranges += 1;
                        e.loads.insert(m);
                    }
                }
                None => e.unmapped += 1,
            }
        }
    }
}

/// Clears the census (`profile reset`).
pub fn reset_f3_regions() {
    *F3_REGIONS.lock() = None;
}

/// The census (`DS_F3_REGIONS=1`; "" otherwise).
pub fn f3_regions_line() -> String {
    if !f3_regions_on() {
        return String::new();
    }
    let g = F3_REGIONS.lock();
    let Some(m) = g.as_ref() else { return String::new() };
    let mib = |s: &HashSet<(u64, u64)>| s.iter().map(|&(a, b)| b - a).sum::<u64>() as f64 / (1u64 << 20) as f64;
    let mut all_bufs: HashSet<(u64, u64)> = HashSet::new();
    let mut all_loads: HashSet<(u64, u64)> = HashSet::new();
    let calls: u64 = m.values().map(|e| e.calls).sum();
    // Calls by how many mappings their program's buffers reached: 1, 2, 3-4,
    // 5-8, more.
    let mut by_count = [0u64; 5];
    for e in m.values() {
        all_bufs.extend(e.bufs.iter().copied());
        all_loads.extend(e.loads.iter().copied());
        let k = match e.bufs.len() {
            0 | 1 => 0,
            2 => 1,
            3..=4 => 2,
            5..=8 => 3,
            _ => 4,
        };
        by_count[k] += e.calls;
    }
    let mut top: Vec<(&u64, &F3Regions)> = m.iter().collect();
    top.sort_by(|a, b| b.1.calls.cmp(&a.1.calls));
    let tops: Vec<String> = top
        .iter()
        .take(10)
        .map(|(a, e)| format!("{a:#x}: {} calls, buffers {:.1} ranges a call in {} mappings ({:.1} MiB), loads {:.1} in {} ({:.1} MiB), {} unmapped", e.calls, e.buf_ranges as f64 / e.calls.max(1) as f64, e.bufs.len(), mib(&e.bufs), e.load_ranges as f64 / e.calls.max(1) as f64, e.loads.len(), mib(&e.loads), e.unmapped))
        .collect();
    let pct = |n: u64| 100.0 * n as f64 / calls.max(1) as f64;
    format!(
        "    stage F3's regions (`DS_F3_REGIONS`): {} programs, {calls} calls; the buffers their chains reach in {} mappings ({:.1} MiB), the per-item blocks in {} ({:.1} MiB); calls by their program's buffer mappings: 1 {:.1}%, 2 {:.1}%, 3-4 {:.1}%, 5-8 {:.1}%, more {:.1}%; by calls: {}\n",
        m.len(),
        all_bufs.len(),
        mib(&all_bufs),
        all_loads.len(),
        mib(&all_loads),
        pct(by_count[0]),
        pct(by_count[1]),
        pct(by_count[2]),
        pct(by_count[3]),
        pct(by_count[4]),
        tops.join("; ")
    )
}

/// Status part of stage F3 ("" unless front end 2 is on).
pub fn f3_line() -> String {
    if !super::fe2::on() {
        return String::new();
    }
    let v: Vec<u64> = F3_PLACED.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    let why: Vec<String> = F3_REFUSED.iter().zip(F3_WHY).map(|(c, w)| format!("{w} {}", c.load(Ordering::Relaxed))).collect();
    let l: Vec<u64> = F3_LAPS.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    let laps = if l[0] > 0 {
        let us = |c: u64| c as f64 * 1e6 / super::threadcpu::tsc_hz() / l[0] as f64;
        format!(
            "    front end 2's GPU-resolved chains by part (`DS_STEP_LAPS`, {} stages): reading the chains {:.2} us, the loads' imports and pending writes {:.2} us, placing the buffers {:.2} us; {:.1} ranges, {:.1} placements and {:.1} runs the batch uploads or placed already a stage\n",
            l[0],
            us(l[1]),
            us(l[2]),
            us(l[3]),
            l[4] as f64 / l[0] as f64,
            l[5] as f64 / l[0] as f64,
            l[6] as f64 / l[0] as f64
        )
    } else {
        String::new()
    };
    format!(
        "    front end 2's GPU-resolved chains: {} ranges placed ({} MiB); not placed: {}; recorded with CPU-resolved chains after that: {} draws, {} dispatches\n",
        v[0],
        v[1] >> 20,
        why.join(", "),
        v[2],
        v[3]
    ) + &laps
}

/// Diagnosis of GPU-resolved commands recorded in full (`gpu_res_check`):
/// what their shader-resolved reads overlap that the CPU path would have
/// bound elsewhere: [checked, a home, pending GPU or CP writes, a resident
/// shadow, not imported].
pub static GPU_RES_CHECKS: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];

/// The compute registers a GPU-resolved run's commands must share with it
/// (program, resources, thread counts).
fn gpu_run_regs(cp: &Cp) -> [u32; 6] {
    [cp.reg(Space::Sh, 0x20C), cp.reg(Space::Sh, 0x20D), cp.reg(Space::Sh, 0x212), cp.reg(Space::Sh, 0x213), cp.reg(Space::Sh, 0x207), cp.reg(Space::Sh, 0x208) ^ (cp.reg(Space::Sh, 0x209) << 16)]
}

/// Bumped whenever import windows are dropped (`revalidate_spans`): the
/// ranges GPU-resolved resources reach are imported again after it.
static DROP_GEN: AtomicU64 = AtomicU64::new(0);

thread_local! {
    /// GPU-resolved resources whose ranges were imported: (translation key,
    /// resource, its root's pointer) -> `DROP_GEN` then.
    static GPU_IMPORTED: std::cell::RefCell<HashMap<(u64, u8, u64), u64>> = std::cell::RefCell::new(HashMap::new());
}

/// Makes sure the ranges a GPU-resolved command's shader-resolved buffers
/// reach are imported (in the guest page table): per root pointer met the
/// first time, the V# read from guest memory and its range imported.
/// False: one cannot be read or imported, or a formatted one's V# has
/// another format or swizzle than the translation was made for (its loads
/// decode with that one: the vertex formats of a draw run's meshes differ).
fn gpu_res_imported(found: &shaders::Found, addr: u64, mask: u64, user: &[u32]) -> bool {
    let plan = &found.tr.plan;
    let generation = DROP_GEN.load(Ordering::Acquire);
    for r in 0..plan.resources.len().min(64) {
        if mask & (1u64 << r) == 0 {
            continue;
        }
        let Some(root) = super::gpures::root_of_res(&plan.resources[r]) else { continue };
        // A V# in the user data itself (its root value is only its base), or
        // one with words of several origins: read again every time.
        let direct = plan.resources[r].dwords.iter().any(|p| matches!(p, rdna::analysis::Prov::User(_))) || super::gpures::res_roots(&plan.resources[r]).is_none_or(|v| v.len() > 1);
        let rv = super::gpures::root_value(root, user);
        // Per translation: a formatted buffer's check depends on its format.
        let key = (found.tr.key, r as u8, rv);
        if !direct && GPU_IMPORTED.with(|c| c.borrow().get(&key) == Some(&generation)) {
            continue;
        }
        let Some(v) = shaders::resource_vsharp(addr, plan, r, user) else { return false };
        if plan.resources[r].formatted {
            if let Some(rdna::translate::ResSpec::Buffer { format, dst_sel }) = found.tr.spec.resources.get(r) {
                if v.format != *format || v.dst_sel != *dst_sel {
                    GPU_FORMAT_MISSES.fetch_add(1, Ordering::Relaxed);
                    return false;
                }
            }
        }
        let n = v.extent();
        if n > 0 && v.base != 0 {
            // Within its mapping (unbounded descriptors reach far past it).
            let n = match mapping(v.base) {
                Some((_, e)) => n.min(e - v.base),
                None => return false,
            };
            if span(v.base, n).is_none() {
                return false;
            }
            // Static data the shader reads: into the mirror (`gmirror`).
            super::gmirror::want(v.base, n);
            GPU_APPENDS[3].fetch_add(1, Ordering::Relaxed);
        }
        GPU_IMPORTED.with(|c| {
            let mut c = c.borrow_mut();
            if c.len() > 1 << 16 {
                c.clear();
            }
            c.insert(key, generation);
        });
    }
    // The scalar loads the shader resolves: the range each reads (at an
    // offset from its pointer, perhaps through a pointer read from the
    // item), once per root pointer value.
    let loads = found.tr.spec.gpu_loads;
    for k in 0..plan.loads.len().min(64) {
        if loads & (1u64 << k) == 0 {
            continue;
        }
        let rv = super::gpures::root_value(super::gpures::root_of_ptr(&plan.loads[k].ptr).unwrap_or((0, None)), user);
        let key = (found.tr.key, 64 + k as u8, rv);
        if GPU_IMPORTED.with(|c| c.borrow().get(&key) == Some(&generation)) {
            continue;
        }
        let Some((a, n)) = shaders::load_range(addr, plan, k, user) else { return false };
        if a != 0 && super::cp::guest_readable(a, n) && span(a, n).is_none() {
            return false;
        }
        GPU_IMPORTED.with(|c| {
            let mut c = c.borrow_mut();
            if c.len() > 1 << 16 {
                c.clear();
            }
            c.insert(key, generation);
        });
    }
    true
}

/// Makes sure the pages a GPU-resolved command's per-item chains start from
/// are in the guest page table (imported): the user pointers and the
/// per-item roots' pointers. False: one cannot be imported.
fn gpu_pages_mapped(items: &[super::gpures::Root], user: &[u32]) -> bool {
    for &root in items {
        let ptr = super::gpures::root_value(root, user);
        if ptr == 0 {
            continue;
        }
        let k = root.0 as usize;
        let u = user.get(k).copied().unwrap_or(0) as u64 | ((user.get(k + 1).copied().unwrap_or(0) as u64 & 0xFFFF) << 32);
        for a in [u, ptr] {
            if a != 0 && super::cp::guest_readable(a, 4) && span(a, 4).is_none() {
                return false;
            }
        }
    }
    true
}

/// Bytes of a batched translation's uniform window of the ring.
const SGPR_WINDOW_BYTES: u64 = 4 * rdna::translate::SGPR_WINDOW_DWORDS as u64;

/// [commands appended to an open batched dispatch, batched dispatches
/// recorded, commands of batched programs run on their own].
pub static BATCHED: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
/// Dispatches' new VRAM shadows filled in the batch's prologue
/// (`tune shadow_pre`) and before the dispatch itself: [prologue, in place,
/// then why in place: off (or no merging), a deferred executor's, a shared
/// copy, a window written earlier in the batch, by the main command
/// processor's unsubmitted batch, by a closed batch not submitted yet].
pub static SHADOWS_PRE: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];

/// Why a batched program's command opened a new dispatch instead of joining
/// the open one: [indirect, not a plain 1-D direct dispatch (start, SGPR
/// patches, a descriptor set, groups), no open dispatch, another batch,
/// commands recorded since it, another pipeline, its table full, other
/// descriptors].
pub static BATCH_MISS: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];
/// Batched programs' commands not joined, by (program, reason), and the
/// exec.rs lines that recorded commands after an open batched dispatch.
static BATCH_MISS_PROGRAMS: Mutex<Option<HashMap<(u64, u8), u64>>> = Mutex::new(None);
static BATCH_MISS_SITES: Mutex<Option<HashMap<u32, u64>>> = Mutex::new(None);

/// A new window for the batched dispatches' statistics (`profile reset`).
pub(super) fn reset_batch_misses() {
    for c in BATCHED.iter().chain(BATCH_MISS.iter()).chain(SHADOWS_PRE.iter()) {
        c.store(0, Ordering::Relaxed);
    }
    *BATCH_MISS_PROGRAMS.lock() = None;
    *BATCH_MISS_SITES.lock() = None;
}

/// Whether a command's descriptors equal a run's (`desc_signature`), without
/// building its own signature.
fn same_descs(bound: &Bound, sig: &[u64]) -> bool {
    use ash::vk::Handle;
    if sig.len() != 4 * (bound.bufs.len() + bound.imgs.len()) {
        return false;
    }
    let mut k = 0;
    for (b, i) in &bound.bufs {
        if sig[k..k + 4] != [*b as u64, i.buffer.as_raw(), i.offset, i.range] {
            return false;
        }
        k += 4;
    }
    for (b, t, i) in &bound.imgs {
        if sig[k..k + 4] != [*b as u64 | (t.as_raw() as u64) << 32, i.sampler.as_raw(), i.image_view.as_raw(), i.image_layout.as_raw() as u64] {
            return false;
        }
        k += 4;
    }
    true
}

/// A command's descriptors as plain words (batched dispatches append only
/// commands whose descriptors equal the run's).
fn desc_signature(bound: &Bound) -> Vec<u64> {
    use ash::vk::Handle;
    let mut v = Vec::with_capacity(4 * (bound.bufs.len() + bound.imgs.len()));
    for (b, i) in &bound.bufs {
        v.extend([*b as u64, i.buffer.as_raw(), i.offset, i.range]);
    }
    for (b, t, i) in &bound.imgs {
        v.extend([*b as u64 | (t.as_raw() as u64) << 32, i.sampler.as_raw(), i.image_view.as_raw(), i.image_layout.as_raw() as u64]);
    }
    v
}

unsafe impl Send for Exec {}

/// A batch a deferred executor closed (see `Exec::deferred`).
struct Closed {
    slot: usize,
    pre: Option<vk::CommandBuffer>,
    shared: Vec<(u64, u64, u64, bool)>,
    /// `Exec::copy_wait` of the batch.
    copy_wait: u64,
    /// `Exec::shared_kept` of the batch.
    kept: Vec<u64>,
    /// `ops_line` of the batch.
    ops: [u64; OPS],
    writes: super::ranges::Ranges,
    checks: Vec<verify::Check>,
    shaders: Vec<u64>,
    races: Vec<RaceCheck>,
    /// Checks of the buffer updates recorded in it (`forward::add_check`).
    forward_ids: Vec<u64>,
    /// Its barrier segments' marks (`seg_stamps`).
    segs: Vec<SegMark>,
    /// Pages it uploads into the residency mirror (`Exec::res_pages`).
    res_pages: Vec<u64>,
    /// Granules whose copies in it records, and its W writes (`gpu::rgm`).
    rgm_claims: Vec<super::rgm::Claim>,
    rgm_written: super::ranges::Ranges,
    /// Mirror mode: its write-back (`Exec::eager_action`) and its ownership
    /// (`Exec::mm_own`).
    eager_action: Option<Action>,
    mm_own: Option<std::sync::Arc<super::resident::Owned>>,
    /// A held executor's label writes in it (address, value, bytes):
    /// promised when it is submitted.
    promises: Vec<(u64, u64, u8)>,
    /// Front end 2: its uploads into the mirror (`Exec::fe2_up`), valid for
    /// other batches once it is queued, and the imports they read
    /// (`Exec::fe2_spans`).
    fe2_up: Vec<(u64, u64)>,
    fe2_spans: Vec<(u64, vk::Buffer)>,
}

/// A seal of the graphics CP's held work past a run (`Exec::past_seal`):
/// what of the held lists comes before it, counted from the seal before.
struct SealParts {
    closed: usize,
    promises: usize,
    cp_written: usize,
}

/// What a sealed part of the held work is checked by before it commits
/// (`Cp::past_roll`): its CPU reads and used images, what its batches
/// write, and its forwarded copies as at the seal.
pub struct SealCheck {
    pub reads: Vec<(u64, u64)>,
    pub keys: HashSet<super::images::ImageKey>,
    pub writes: super::ranges::Ranges,
    pub fwd: Option<super::forward::HeldView>,
}

/// Slot sequence of a batch closed but not yet submitted.
const PENDING: u64 = u64::MAX;
/// Batch slots a deferred executor may create.
const DEFERRED_SLOTS: usize = 6;
/// The frame-ahead recorder's (`Exec::set_held`): it holds a frame's
/// prefix of batches.
const HELD_SLOTS: usize = 12;
/// Writes `Exec::since_barrier` lists (more: any range counts as written).
const SINCE_MAX: usize = 128;
/// Indirect dispatches with a barrier of their own, and without one (their
/// arguments not written since the batch's last barrier; `tune
/// exact_indirect`).
pub static INDIRECT_BARRIERS: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];
/// Draws that sampled an attachment of the current rendering instance and
/// kept it (neither written in it), and those that ended it.
pub static FEEDBACK: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];
/// Those draws by the attachment they sample and how they use it (see the
/// draw path), and their pixel shaders (profiling).
pub static FEEDBACK_KINDS: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];
pub static FEEDBACK_PS: Mutex<Option<HashMap<u64, u64>>> = Mutex::new(None);
/// Stages prepared again with GPU-side data loads because GPU work still
/// wrote their data (`tune gpu_loads 3`): [vertex, pixel, compute].
pub static HYBRID_LOADS: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
/// Snapshots of data-only loads read on the GPU (`tune data_snapshots`).
pub static DATA_SNAPS: AtomicU64 = AtomicU64::new(0);
/// Render passes ended without a barrier of their own because a full one
/// followed at once (`end_rendering_before_barrier`).
pub static MERGED_PASS_BARRIERS: AtomicU64 = AtomicU64::new(0);
/// GPU-performed CP writes with resident shadows in the batch: [shadows
/// merged first, written with shadows unmerged (`tune label_merge 0`)].
pub static LABEL_MERGES: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];
/// Batch slots deferred executors made beyond their first two (statistics).
pub static WORKER_SLOTS: AtomicU64 = AtomicU64::new(0);

/// Error of a command a deferred executor stopped at (see `Exec::stopped`).
pub(crate) const STOPPED: &str = "stopped (parallel run)";

impl Exec {
    pub fn new(name: &str) -> Result<Exec, String> {
        let c = ctx();
        let alt_id = name.strip_prefix("gpu-acb").and_then(|h| u64::from_str_radix(h.trim_start_matches("0x"), 16).ok()).unwrap_or(0);
        // This queue's batches all go to the compute-only family's queue
        // (`vk::alt_compute_id`): its command buffers come from that family.
        let on_compute = alt_id != 0 && super::vk::alt_compute_id() == Some(alt_id) && c.compute.is_some();
        let family = if on_compute { c.compute.as_ref().map_or(c.queue_family, |q| q.family) } else { c.queue_family };
        {
            let xfamily = c.copy.as_ref().map(|q| q.family);
            let slots = vec![Slot::new(family, xfamily)?, Slot::new(family, xfamily)?];
            Ok(Exec {
                family,
                xfamily,
                slots,
                cur: 0,
                slot_limit: MAX_SLOTS,
                seg_full: std::env::var("DS_SLOT_SEGS").ok().as_deref() == Some("full"),
                seg_want: if std::env::var("DS_SLOT_SEGS").ok().as_deref() == Some("full") { [SHADOW_BYTES, PRE_SEG] } else { [SHADOW_INIT, PRE_INIT] },
                seg_peak: [0; 2],
                seg_batches: 0,
                vram_ring_failed: false,
                ring_used: [0; 2],
                snap_used: 0,
                snap_failed: false,
                snaps: Default::default(),
                pre_used: 0,
                pre_open: false,
                pre_failed: false,
                pres: Default::default(),
                res_pages: Vec::new(),
                res_mine: Default::default(),
                res_need: Vec::new(),
                stage_used: 0,
                fe2_up: Vec::new(),
                fe2_spans: Vec::new(),
                fe2_mine: Default::default(),
                fe2_marked: false,
                fe2_gens: Vec::new(),
                fe2_pend: Vec::new(),
                fe2_pend_set: Default::default(),
                fe2_view: super::fe2::view_cell(name),
                f3_ranges: Vec::new(),
                f3_loads: Vec::new(),
                f3_reads: Vec::new(),
                f3_placed: Default::default(),
                fe2_min_want: u32::MAX,
                rgm_claims: Vec::new(),
                rgm_mine: Default::default(),
                rgm_written: Default::default(),
                rgm_unmerged: Default::default(),
                rgm_table: None,
                rgm_pages: Vec::new(),
                pre_copies: Vec::new(),
                xcopies: Vec::new(),
                copy_wait: 0,
                ring_align: c.limits.min_storage_buffer_offset_alignment.max(16),
                ubo_align: c.limits.min_uniform_buffer_offset_alignment.max(c.limits.min_storage_buffer_offset_alignment).max(16),
                recording: false,
                begun_at: std::time::Instant::now(),
                begun_mark: None,
                pending_promises: Vec::new(),
                submit_request: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
                pending: Default::default(),
                pending_log: Vec::new(),
                actions: Vec::new(),
                checks: Vec::new(),
                inflight: Default::default(),
                last_seq: 0,
                last_action_seq: 0,
                submit_after: 0,
                ts_used: 0,
                ts_pool_reset: false,
                ts_deferred: Vec::new(),
                kept_taken: Vec::new(),
                kept_written: Vec::new(),
                kept_given: Vec::new(),
                eop_unordered: false,
                ts_check: Vec::new(),
                gpu_writes_open: false,
                last_gpu_write_seq: 0,
                cpu_action_seq: 0,
                cpu_dep_seq: 0,
                barrier_wanted: false,
                args_read: false,
                dispatched: false,
                reported: HashMap::new(),
                name: name.to_string(),
                batch_shaders: Vec::new(),
                rendering: None,
                dyncache: Default::default(),
                bound_pool: Vec::new(),
                keys_scratch: Vec::new(),
                own_gen: 0,
                ops: [0; OPS],
                clean_pages: vec![u64::MAX; CLEAN_SLOTS],
                clean_tag: [u64::MAX; 4],
                dirty_reads: Vec::new(),
                last_notes: (Vec::new(), Vec::new(), [u64::MAX; 4]),
                draw_bounds: Vec::new(),
                pool_use: [0; 6],
                shadow_used: 0,
                resident: Vec::new(),
                sync_readers: [0; 2],
                sync_culprit: None,
                deferred: false,
                held: false,
                past: false,
                held_promises: Vec::new(),
                held_reads: Vec::new(),
                held_kinds: Vec::new(),
                cp_written: Vec::new(),
                past_grow: false,
                past_want_min: u32::MAX,
                past_ranges: Vec::new(),
                past_up: HashSet::new(),
                stretch: false,
                held_clears: HashMap::new(),
                held_over: HashSet::new(),
                held_keys: HashSet::new(),
                long: false,
                held_fwd_reads: Vec::new(),
                held_fwd_planned: Vec::new(),
                long_planned: Default::default(),
                noting_planned: false,
                long_nonfwd: Default::default(),
                long_mismatch: false,
                rec_writes: None,
                main_writes: None,
                main_submitted: None,
                stopped: false,
                stop_why: "",
                flush_why: "",
                closed: Vec::new(),
                seals: Default::default(),
                open_gen: {
                    let g = std::sync::Arc::new(AtomicU64::new(u64::MAX));
                    EXEC_GENS.lock().push(g.clone());
                    g
                },
                shares_pre: true,
                shared_pending: Vec::new(),
                shared_kept: Vec::new(),
                tex_scratch: Vec::new(),
                last_draw_sig: [0; 5],
                draw_same: false,
                last_draw_end: (u64::MAX, 0),
                last_draw_where: (None, None),
                again: None,
                again_next: None,
                draw_run: None,
                draw_streak: ((0, 0), 0),
                own_checks: Vec::new(),
                race_pending: Vec::new(),
                race_id: RACE_IDS.fetch_add(1, Ordering::Relaxed),
                race_name: name.into(),
                forward_ids: Vec::new(),
                alt_id,
                on_compute,
                compute_cp: name.starts_with("gpu-acb"),
                acb50: name == "gpu-acb0x50",
                noting: 0,
                label_open: false,
                batch_alt: false,
                batch_alt_share: false,
                since_barrier: Some(Vec::new()),
                since_reads: Some(Vec::new()),
                since_img_r: Vec::new(),
                since_img_w: Vec::new(),
                since_gds: false,
                since_draws: false,
                segs: Vec::new(),
                repeat: None,
                cmd_uses: std::cell::Cell::new(0),
                last_cmd_line: std::cell::Cell::new(0),
                cs_streak: (0, 0),
                cs_peeked: HashSet::new(),
                cs_run: None,
                owned: Vec::new(),
                mm_own: None,
                extra_used: 0,
                mm_copy: Vec::new(),
                pre_after: Vec::new(),
                mm_cp: Vec::new(),
                mm_unsized: false,
                eager_action: None,
                batch_count: 0,
            })
        }
    }

    fn report(&mut self, msg: String) {
        let n = self.reported.entry(msg.clone()).or_insert(0);
        *n += 1;
        if *n == 1 {
            crate::warn!("gpu[{}]: {msg}", self.name);
        }
    }

    /// `report` as an error line (front end 2's coherence failures), the
    /// first `example` given with it.
    fn report_error(&mut self, msg: &str, example: impl FnOnce() -> String) {
        let n = self.reported.entry(msg.to_string()).or_insert(0);
        *n += 1;
        if *n == 1 {
            crate::error!("gpu[{}]: {msg} (first: {})", self.name, example());
        }
    }

    #[track_caller]
    fn cmd(&self) -> vk::CommandBuffer {
        self.cmd_uses.set(self.cmd_uses.get().wrapping_add(1));
        self.last_cmd_line.set(std::panic::Location::caller().line());
        self.slots[self.cur].cmd
    }

    /// The recording slot's shadow buffer.
    fn shadow(&self) -> vk::Buffer {
        self.slots[self.cur].shadow
    }

    /// Makes the current slot's shadow space hold `need` bytes: made, or
    /// made again larger (at least double) while the batch has none of it
    /// in use. False: it is in use and too small (the caller ends the batch
    /// or does without), or no video memory.
    fn shadow_room(&mut self, need: u64) -> bool {
        if need > SHADOW_BYTES {
            return false;
        }
        let s = &self.slots[self.cur];
        if s.shadow != vk::Buffer::null() && s.shadow_cap >= need {
            return true;
        }
        if need > self.seg_want[0] {
            self.seg_want[0] = need.next_power_of_two().min(SHADOW_BYTES);
        }
        if self.shadow_used > 0 {
            return false;
        }
        let cap = if self.seg_full { SHADOW_BYTES } else { self.seg_want[0] };
        self.free_slot_seg(0);
        match ctx().device_buffer(cap, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDIRECT_BUFFER) {
            Ok((b, m)) => {
                let s = &mut self.slots[self.cur];
                (s.shadow, s.shadow_mem, s.shadow_cap) = (b, m, cap);
                SLOT_MEM[1].fetch_add(cap, Ordering::Relaxed);
                true
            }
            Err(e) => {
                crate::warn!("gpu[{}]: no video memory for shadows ({e})", self.name);
                false
            }
        }
    }

    /// The same for the batch-start copy segment (`pre_copy`).
    fn pre_room(&mut self, need: u64) -> bool {
        if need > PRE_SEG {
            return false;
        }
        let s = &self.slots[self.cur];
        if s.pre_buf.is_some() && s.pre_cap >= need {
            return true;
        }
        if need > self.seg_want[1] {
            self.seg_want[1] = need.next_power_of_two().min(PRE_SEG);
        }
        if self.pre_used > 0 {
            return false;
        }
        let cap = if self.seg_full { PRE_SEG } else { self.seg_want[1] };
        self.free_slot_seg(1);
        match ctx().device_buffer(cap, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER) {
            Ok((b, m)) => {
                let s = &mut self.slots[self.cur];
                (s.pre_buf, s.pre_mem, s.pre_cap) = (Some(b), m, cap);
                SLOT_MEM[2].fetch_add(cap, Ordering::Relaxed);
                true
            }
            Err(e) => {
                crate::warn!("gpu[{}]: no video memory for batch-start copies ({e})", self.name);
                self.pre_failed = true;
                false
            }
        }
    }

    /// Frees the current slot's shadow (0) or batch-start copy (1) segment:
    /// only while no batch can use it (the slot's last batch completed,
    /// nothing of this batch recorded into it).
    fn free_slot_seg(&mut self, which: usize) {
        let c = ctx();
        let s = &mut self.slots[self.cur];
        unsafe {
            if which == 0 && s.shadow != vk::Buffer::null() {
                super::vk::note_gpu_freed(s.shadow);
                c.device.destroy_buffer(s.shadow, None);
                c.device.free_memory(s.shadow_mem, None);
                SLOT_MEM[1].fetch_sub(s.shadow_cap, Ordering::Relaxed);
                (s.shadow, s.shadow_mem, s.shadow_cap) = (vk::Buffer::null(), vk::DeviceMemory::null(), 0);
            }
            if which == 1 {
                if let Some(b) = s.pre_buf.take() {
                    super::vk::note_gpu_freed(b);
                    c.device.destroy_buffer(b, None);
                    c.device.free_memory(s.pre_mem, None);
                    SLOT_MEM[2].fetch_sub(s.pre_cap, Ordering::Relaxed);
                    (s.pre_mem, s.pre_cap) = (vk::DeviceMemory::null(), 0);
                }
            }
        }
    }

    fn begin(&mut self) {
        if self.recording {
            return;
        }
        let t0 = profiling().then(std::time::Instant::now);
        self.begin_batch();
        if let Some(t) = t0 {
            BEGIN_STATS[0].fetch_add(1, Ordering::Relaxed);
            BEGIN_STATS[1].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
    }

    fn begin_batch(&mut self) {
        self.batch_count += 1;
        self.repeat = None;
        self.label_open = false;
        // Imports looked up from now on belong to this batch (`reap_spans`).
        if self.open_gen.load(Ordering::Relaxed) == u64::MAX {
            self.open_gen.store(SPAN_GEN.load(Ordering::SeqCst), Ordering::SeqCst);
        }
        if self.past && !self.past_grow && self.slots.get(self.cur).is_some_and(|s| s.seq == PENDING) {
            // Recording past a run: a slot holding one of its held batches
            // came round again.
            self.stop("no batch slot");
            return;
        }
        if self.deferred && (!self.past || self.past_grow) {
            // A slot whose batch is closed but not submitted cannot be reused,
            // and one whose batch the GPU still executes would make the chunk
            // wait for it (a worker records several chunks of a run): a free
            // one, or a new one, else one still executing, else the chunk
            // stops.
            let n = self.slots.len();
            let done = done_seq();
            let free = |s: &Slot| s.seq != PENDING && s.seq <= done;
            // (A slot not made yet is free: `ensure_slot` makes it. A queue
            // that goes past a wait before using all its slots meets one.)
            if self.cur < n && !free(&self.slots[self.cur]) {
                let next = |ok: &dyn Fn(&Slot) -> bool| (0..n).map(|k| (self.cur + k) % n).find(|&k| ok(&self.slots[k]));
                match next(&free) {
                    Some(k) => self.cur = k,
                    None => {
                        let made = if n < if self.held { HELD_SLOTS } else { DEFERRED_SLOTS } { Slot::new(self.family, self.xfamily).ok() } else { None };
                        match made {
                            Some(s) => {
                                self.slots.push(s);
                                self.cur = n;
                                WORKER_SLOTS.fetch_add(1, Ordering::Relaxed);
                            }
                            None => match next(&|s: &Slot| s.seq != PENDING) {
                                Some(k) => self.cur = k,
                                None => {
                                    self.stop("no batch slot");
                                    return;
                                }
                            },
                        }
                    }
                }
            }
        }
        self.ensure_slot();
        // The slot's previous batch must have completed before reuse.
        let slot = &self.slots[self.cur];
        if slot.seq != 0 {
            if slot.seq > done_seq() {
                let t = std::time::Instant::now();
                let tw = super::timeline::wait_start();
                if !wait_done_for(slot.seq, std::time::Duration::from_secs(3)) {
                    crate::warn!("gpu[{}]: waiting > 3 s for a free batch slot (batch {}, done {}; held {}, past {})", self.name, slot.seq, done_seq(), self.held, self.past);
                }
                wait_done(slot.seq);
                tw.end(&self.name, || format!("wait: batch slot (#{})", slot.seq));
                super::hitch::note_since(super::hitch::Kind::GpuWait, t, || format!("a free batch slot: the GPU behind (batch {})", slot.seq));
                WAIT_NS[2].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
                WAIT_NS[3].fetch_add(1, Ordering::Relaxed);
                note_queue_wait(&self.name, 1, t.elapsed());
            }
            wait_done(slot.seq);
            let c = ctx();
            // The completion thread resets the slot right after the batch
            // completed (`tune offthread_reset`): waited for, or taken back
            // when it cannot come (shutdown released the wait, device lost).
            let mut reset_done = false;
            if slot.reset_pending {
                let mut spins = 0u32;
                loop {
                    let v = slot.reset_state.load(Ordering::Acquire);
                    if v == 0 {
                        reset_done = true;
                        break;
                    }
                    if v != u64::MAX && (done_seq() < slot.seq || DEVICE_LOST.load(Ordering::Relaxed)) {
                        if slot.reset_state.compare_exchange(v, 0, Ordering::AcqRel, Ordering::Acquire).is_ok() {
                            break;
                        }
                        continue;
                    }
                    spins += 1;
                    if spins < 256 {
                        std::hint::spin_loop();
                    } else {
                        std::thread::yield_now();
                    }
                }
                if !reset_done {
                    OFFTHREAD_RESETS[2].fetch_add(1, Ordering::Relaxed);
                } else if spins > 0 {
                    OFFTHREAD_RESETS[1].fetch_add(1, Ordering::Relaxed);
                } else {
                    OFFTHREAD_RESETS[0].fetch_add(1, Ordering::Relaxed);
                }
            }
            if !reset_done {
                if done_seq() < slot.seq {
                    // Shutdown released the wait: the command buffer may still
                    // execute (resetting it would lose the device).
                    unsafe { c.device.wait_for_fences(&[slot.fence], true, 2_000_000_000).ok() };
                }
                unsafe {
                    c.device.reset_fences(&[slot.fence]).ok();
                    c.device.reset_command_buffer(slot.cmd, vk::CommandBufferResetFlags::empty()).ok();
                    c.device.reset_command_buffer(slot.pre, vk::CommandBufferResetFlags::empty()).ok();
                    c.device.reset_descriptor_pool(slot.desc_pool, vk::DescriptorPoolResetFlags::empty()).ok();
                }
            }
            self.slots[self.cur].reset_pending = false;
            self.slots[self.cur].seq = 0;
            self.slots[self.cur].homes.clear();
            self.slots[self.cur].keep.clear();
            self.slots[self.cur].kept.clear();
            for (b, m, _) in std::mem::take(&mut self.slots[self.cur].stage_extra) {
                unsafe {
                    ctx().device.destroy_buffer(b, None);
                    ctx().device.free_memory(m, None);
                }
            }
        }
        self.extra_used = 0;
        unsafe {
            ctx().device.begin_command_buffer(self.cmd(), &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).ok();
            let qp = self.slots[self.cur].queries;
            if qp != vk::QueryPool::null() {
                ctx().device.cmd_reset_query_pool(self.cmd(), qp, 0, STAMPS * MAX_TIMED);
            }
            let bq = self.slots[self.cur].bq;
            if bq != vk::QueryPool::null() {
                ctx().device.cmd_reset_query_pool(self.cmd(), bq, BQ_BATCH, 2);
            }
            // Mirror mode: the batch's touched-page list starts empty (the
            // full barrier below orders it before the batch's work). (Front
            // end 2's write-back that empties it itself leaves it so: every
            // batch whose work marks ends with one, and all batches run in
            // order; entries left over would only be written back again.)
            if let Some(m) = super::resident::marks().filter(|_| !super::fe2::scatter_resets()) {
                ctx().device.cmd_fill_buffer(self.cmd(), m, rdna::kernels::MARK_LIST_AT as u64 * 4, 4, 0);
            }
        }
        self.fe2_marked = false;
        // Every batch follows all earlier ones: also a parallel run's chunks,
        // which the title left unordered, as their shadows of a buffer they
        // all write (the culls' appends) are copied in and merged back per
        // batch (FINDINGS.md 2026-10-14 night). (`tune skip_barriers` 64:
        // left out, timing only.)
        self.ops = [0; OPS];
        self.segs.clear();
        if super::tune::skip_barriers() & 64 == 0 {
            self.barrier();
        }
        self.barrier_wanted = false;
        self.dispatched = false;
        let bq = self.slots[self.cur].bq;
        if bq != vk::QueryPool::null() {
            unsafe { ctx().device.cmd_write_timestamp(self.cmd(), vk::PipelineStageFlags::BOTTOM_OF_PIPE, bq, BQ_BATCH) };
        }
        self.recording = true;
        let aq = super::tune::alt_queue();
        self.batch_alt = self.on_compute || (self.alt_id != 0 && aq & 0xFFFF == self.alt_id && ctx().queue2.is_some());
        self.batch_alt_share = aq & 0x10000 != 0;
        self.dyncache = Default::default();
        self.begun_at = std::time::Instant::now();
        self.begun_mark = super::timeline::mark();
        self.pool_use = [0; 6];
        self.ts_used = 0;
        self.ts_pool_reset = false;
        self.ops = [0; OPS];
        self.ts_deferred.clear();
        self.eop_unordered = false;
        note_seg_use(self.deferred, self.compute_cp, self.shadow_used, 0);
        self.shadow_used = 0;
        self.resident.clear();
        // Segments smaller than the executor now wants, or more than twice
        // that, are made again at their first use in this batch (the slot
        // is free here).
        let (want, scap, pcap) = (self.seg_want, self.slots[self.cur].shadow_cap, self.slots[self.cur].pre_cap);
        if scap != 0 && (scap < want[0] || scap > 2 * want[0]) {
            self.free_slot_seg(0);
        }
        if pcap != 0 && (pcap < want[1] || pcap > 2 * want[1]) && self.pre_used == 0 {
            self.free_slot_seg(1);
        }
    }

    /// Commands in the batch being recorded by a command processor (not a
    /// parallel worker's chunk); 0 when none is open.
    pub fn serial_batch_len(&self) -> usize {
        if self.recording && !self.deferred {
            self.batch_shaders.len()
        } else {
            0
        }
    }

    /// Slot `i`'s batch is being submitted: the completion thread resets the
    /// slot once it completed (`tune offthread_reset`; `push_batch` notes
    /// the batch in `reset_state`).
    fn slot_reset(&mut self, i: usize) -> Option<SlotReset> {
        if !super::tune::offthread_reset() {
            return None;
        }
        let s = &mut self.slots[i];
        s.reset_pending = true;
        Some(SlotReset { pools: [s.cpool, s.xcpool], desc_pool: s.desc_pool, fence: s.fence, state: s.reset_state.clone() })
    }

    /// Creates the current slot when the rotation reached one not made yet
    /// (`tune slots` raised): it is used before the batch begins (homes of
    /// index buffers are kept by the slot their draw records into).
    fn ensure_slot(&mut self) {
        if self.cur == self.slots.len() {
            match Slot::new(self.family, self.xfamily) {
                Ok(slot) => self.slots.push(slot),
                Err(e) => {
                    crate::warn!("gpu[{}]: no batch slot {} ({e}); using {}", self.name, self.cur + 1, self.cur);
                    self.slot_limit = self.cur;
                    self.cur = 0;
                }
            }
        }
    }

    /// Submits the recorded batch without waiting for it (its actions run
    /// when it completes).
    /// Submits the recorded batch; `why` (profiling): what ended it.
    pub fn submit_for(&mut self, why: &'static str) {
        self.label_close();
        if !self.recording {
            return;
        }
        if super::runprobe::on() && self.name == "gpu-gfx" {
            super::runprobe::stop(why);
        }
        if super::shutting_down() {
            self.discard();
            return;
        }
        let t_submit = profiling().then(std::time::Instant::now);
        // (Chunk batches of parallel workers close here without a submission.)
        let mut st = StepTimer(t_submit.filter(|_| !self.deferred), why == "parallel run");
        if st.0.is_some() && st.1 {
            SUBMIT_STEPS_RUN[11].fetch_add(1, Ordering::Relaxed);
        }
        let mut waited = std::time::Duration::ZERO;
        if profiling() {
            *SUBMIT_WHY.lock().get_or_insert_with(HashMap::new).entry((self.name.clone(), why)).or_default() += 1;
        }
        if let Err(e) = self.flush_resident(None) {
            self.report(format!("shadow write-back: {e}"));
        }
        if let Err(e) = self.keep_end_batch() {
            self.report(format!("kept window write-back: {e}"));
        }
        if let Err(e) = self.rgm_merge(None) {
            self.report(format!("rgm merge: {e}"));
        }
        st.mark(0);
        if self.submit_after > done_seq() {
            // Deferred CP writes that this work may read come first.
            let tw = super::timeline::wait_start();
            let t = std::time::Instant::now();
            wait_done(self.submit_after);
            waited = tw.end(&self.name, || format!("wait: deferred CP writes (#{})", self.submit_after));
            super::hitch::note_since(super::hitch::Kind::GpuWait, t, || format!("deferred CP writes (batch {})", self.submit_after));
            st.restart();
        }
        self.end_rendering();
        st.mark(1);
        self.resolve_timestamps();
        st.mark(2);
        // Front end 2: what the batch's work wrote into the mirror goes to
        // guest memory before the batch completes; the checks its bindings
        // deferred are made now, their uploads into its prologue.
        if self.fe2_marked {
            SCATTER_SITES[1].fetch_add(1, Ordering::Relaxed);
            if self.pending_promises.is_empty() && self.actions.is_empty() {
                SCATTER_SITES[2].fetch_add(1, Ordering::Relaxed);
            }
            self.fe2_scatter();
        }
        if !self.fe2_pend.is_empty() {
            self.fe2_resolve();
        }
        // Mirror mode: pages deferred CP writes changed are copied into the
        // mirror again (those writes were waited for above), and the dwords
        // the batch's work changes in its written ranges go to guest memory
        // when it completes.
        if super::resident::mirror_mode() && !self.mm_cp.is_empty() {
            if let Err(e) = self.mirror_cp_copies() {
                self.report(format!("mirror copies after CP writes: {e}"));
            }
        }
        if super::resident::mirror_mode() && !self.owned.is_empty() {
            if let Err(e) = self.record_mirror_writeback() {
                self.report(format!("mirror write-back: {e}"));
            }
        }
        if !self.deferred && !self.xcopies.is_empty() && super::tune::xpre_dry() && gpu_ran_dry() {
            // The GPU waits for this batch: its copies go into its prologue
            // (`tune xpre_dry`).
            let pre = self.open_pre();
            record_copies(pre, &mut self.xcopies);
            for e in self.shared_pending.iter_mut() {
                e.3 = false;
            }
            XDRY.fetch_add(1, Ordering::Relaxed);
        }
        // Labels opened by the batch's last work (merges, timestamp stores)
        // end with its command buffer (`DS_GPU_LABELS`).
        self.label_close();
        let i = self.cur;
        // The executor's segment sizes come down after SEG_DECAY batches
        // that used under a quarter of them (after a level load's big
        // batches; `begin_batch` remakes slots more than twice as large).
        self.seg_peak = [self.seg_peak[0].max(self.shadow_used), self.seg_peak[1].max(self.pre_used)];
        self.seg_batches += 1;
        if self.seg_batches >= SEG_DECAY && !self.seg_full {
            for (k, init) in [SHADOW_INIT, PRE_INIT].into_iter().enumerate() {
                if self.seg_want[k] > init && self.seg_peak[k] * 4 <= self.seg_want[k] {
                    self.seg_want[k] = self.seg_peak[k].next_power_of_two().max(init);
                }
            }
            self.seg_peak = [0; 2];
            self.seg_batches = 0;
        }
        let (cmd, fence) = (self.slots[i].cmd, self.slots[i].fence);
        let bq = self.slots[i].bq;
        // Finished by the submission thread (`tune submit_finish`): the
        // prologue left open with its copies, the copy-queue copies, the
        // command buffer's end.
        let finish = !self.deferred && super::tune::submit_finish() && super::tune::submit_thread();
        let (pre, xpre, pre_job, xpre_job) = if finish {
            let pre_job = self.take_pre();
            let xpre_job = (!self.xcopies.is_empty()).then(|| (self.slots[i].xpre, std::mem::take(&mut self.xcopies)));
            (None, None, pre_job, xpre_job)
        } else {
            let pre = self.end_pre();
            let xpre = self.end_xpre();
            finish_main(cmd, bq, pre.is_some());
            (pre, xpre, None, None)
        };
        let copy_wait = std::mem::take(&mut self.copy_wait);
        st.mark(3);
        st.mark(4);
        let actions = std::mem::take(&mut self.actions);
        let has_actions = !actions.is_empty();
        let name = self.name.clone();
        if profiling() {
            // The GPU ran out of work before this batch: how long, and how
            // long the batch took to record.
            let idle = (done_seq() + 1 == NEXT_SEQ.load(Ordering::SeqCst)).then(|| LAST_DONE_AT.lock().map(|t| t.elapsed())).flatten();
            if let Some(idle) = idle {
                let mut g = STARVED.lock();
                let e = g.get_or_insert_with(HashMap::new).entry(self.name.clone()).or_insert([0; 3]);
                e[0] += 1;
                e[1] += idle.as_nanos() as u64;
                e[2] += self.begun_at.elapsed().as_nanos() as u64;
            }
        }
        let mut checks = std::mem::take(&mut self.checks);
        for ch in checks.iter_mut() {
            let (a, n) = verify::dest_range(ch);
            ch.overwritten = self.pending_log.get(ch.writes_mark..).unwrap_or(&[]).iter().any(|&(w, m)| w < a + n && a < w + m);
        }
        if self.deferred {
            let promises = if self.held && !has_actions {
                // Its later waits pass on them (`open_promise`); promised
                // when the batch is submitted (`submit_closed`).
                self.held_promises.extend_from_slice(&self.pending_promises);
                std::mem::take(&mut self.pending_promises)
            } else {
                if has_actions || !self.pending_promises.is_empty() {
                    crate::error!("gpu[{}]: CP effects in a parallel run (dropped)", self.name);
                }
                Vec::new()
            };
            super::timeline::end(&self.name, self.begun_mark.take(), || format!("chunk batch, {} commands ({why})", self.batch_shaders.len()));
            self.keep_submitted(0);
            self.closed.push(Closed { slot: i, pre, shared: std::mem::take(&mut self.shared_pending), copy_wait, kept: std::mem::take(&mut self.shared_kept), ops: self.ops, writes: std::mem::take(&mut self.pending), checks, shaders: std::mem::take(&mut self.batch_shaders), races: std::mem::take(&mut self.race_pending), forward_ids: std::mem::take(&mut self.forward_ids), segs: std::mem::take(&mut self.segs), res_pages: std::mem::take(&mut self.res_pages), rgm_claims: std::mem::take(&mut self.rgm_claims), rgm_written: std::mem::take(&mut self.rgm_written), eager_action: self.eager_action.take(), mm_own: self.mm_own.take(), promises, fe2_up: Vec::new(), fe2_spans: Vec::new() });
            self.res_mine.clear();
            if let Some(b) = self.closed.last_mut() {
                b.fe2_up = std::mem::take(&mut self.fe2_up);
                b.fe2_spans = std::mem::take(&mut self.fe2_spans);
            }
            self.fe2_mine.clear();
            self.f3_placed.clear();
            self.rgm_batch_done(true);
            self.stage_used = 0;
            self.slots[i].seq = PENDING;
            self.pending_promises.clear();
            self.pending_log.clear();
            self.recording = false;
            self.cur = (self.cur + 1) % self.slots.len();
            self.ring_used = [0; 2];
            self.snap_used = 0;
            self.snaps.clear();
            note_seg_use(self.deferred, self.compute_cp, 0, self.pre_used);
            self.pre_used = 0;
            self.pres.clear();
            return;
        }
        let commands = self.batch_shaders.len();
        let gpu_labels = self.pending_promises.clone();
        note_copy_waits(xpre.is_some() || xpre_job.is_some(), copy_wait);
        let alt = self.batch_alt;
        let reset = self.slot_reset(i);
        // Front end 2: the words its uploads touch, before it can run.
        let fe2_ep = super::precache::epoch() as u32;
        if !self.fe2_up.is_empty() {
            let mut gens = std::mem::take(&mut self.fe2_gens);
            super::fe2::generations(&self.fe2_up, &mut gens);
            self.fe2_gens = gens;
        }
        let seq = push_batch(fence, self.slots[i].queries, bq, actions, checks, std::mem::take(&mut self.batch_shaders), &self.name, gpu_labels, self.ops, std::mem::take(&mut self.segs), reset, alt, move |seq| {
            if DEVICE_LOST.load(Ordering::Relaxed) {
                return None;
            }
            let (mut pre, mut xpre) = (pre, xpre);
            if finish {
                pre = pre_job.map(|(p, mut copies, mut after)| {
                    finish_pre(p, &mut copies, &mut after, bq);
                    p
                });
                xpre = xpre_job.map(|(x, mut copies)| {
                    finish_xpre(x, &mut copies);
                    x
                });
                finish_main(cmd, bq, pre.is_some());
            }
            let _ = seq;
            Some(SubmitSpec { xpre, pre, cmd, wait: copy_wait, fence, alt, name })
        });
        super::timeline::end(&self.name, self.begun_mark.take(), || format!("batch #{seq}, {commands} commands ({why})"));
        st.mark(5);
        if std::mem::take(&mut self.gpu_writes_open) {
            self.last_gpu_write_seq = seq;
        }
        if !self.ts_check.is_empty() {
            TS_CHECKS.lock().extend(self.ts_check.drain(..).map(|a| (seq, a)));
        }
        self.slots[i].seq = seq;
        self.open_gen.store(u64::MAX, Ordering::SeqCst);
        if !self.fe2_up.is_empty() {
            super::fe2::uploaded(&self.fe2_up, &self.fe2_gens, fe2_ep);
            self.fe2_up.clear();
        }
        if !self.fe2_spans.is_empty() {
            note_span_uses(&std::mem::take(&mut self.fe2_spans), seq);
        }
        self.fe2_mine.clear();
        self.f3_placed.clear();
        super::precache::submitted(&std::mem::take(&mut self.shared_pending), seq);
        if !self.res_pages.is_empty() {
            super::resident::submitted(&std::mem::take(&mut self.res_pages), seq);
        }
        self.res_mine.clear();
        self.owned.clear();
        self.stage_used = 0;
        if !self.own_checks.is_empty() {
            let mut g = SPEC_CHECKS.lock();
            g.extend(self.own_checks.drain(..).map(|c| SpecCheck { seq, ..c }));
            SPEC_PENDING.store(g.len() as u64, Ordering::Release);
        }
        if !self.race_pending.is_empty() {
            let cpw_submit = super::cp::CPW_SEQ.load(Ordering::Acquire);
            RACE_CHECKS.lock().extend(self.race_pending.drain(..).map(|c| RaceCheck { seq, cpw_submit, ..c }));
        }
        if !self.forward_ids.is_empty() {
            super::forward::submitted(&std::mem::take(&mut self.forward_ids), seq);
        }
        if let Some(o) = self.mm_own.take() {
            super::resident::own_submitted(&o, seq);
        }
        st.mark(6);
        // The batch's writes are noted before its labels are promised: a
        // queue that passes a wait on one sees them (its copies of those
        // ranges stay on the main queue, after this batch).
        let writes = std::mem::take(&mut self.pending);
        self.pending_log.clear();
        INFLIGHT.note_ranges(seq, &writes);
        journal_note(&writes, &self.name);
        crate::hostcensus::gpu_wrote(seq, &writes);
        if self.name == "gpu-gfx" {
            super::frameahead::a_wrote_ranges(&writes);
        }
        if !super::resident::mirror_mode() {
            super::resident::gpu_wrote(&writes);
        }
        if super::tune::race_checks() {
            note_race_writes(seq, &writes);
            self.clear_open_writes(None);
        }
        st.mark(8);
        for (a, v, n) in std::mem::take(&mut self.pending_promises) {
            super::cp::promise(seq, a, v, n, false, self.promise_view());
        }
        if let Some(m) = UNSUBMITTED_LABELS.lock().as_mut() {
            m.retain(|_, f| !std::sync::Arc::ptr_eq(f, &self.submit_request));
        }
        self.submit_request.store(false, Ordering::SeqCst);
        st.mark(7);
        // Shared copies made by batches submitted before this one precede
        // its writes on the GPU (see `precache`).
        super::precache::invalidate_ranges_except(&writes, &std::mem::take(&mut self.shared_kept));
        st.mark(9);
        self.inflight.insert_all(seq, &writes);
        // Writes by other paths dirty resident pages again now that they
        // precede later batches (a copy in by a batch submitted before this
        // one ran before them).
        if super::rgm::get().is_some() {
            super::rgm::invalidate_except(&writes, &self.rgm_written);
        }
        self.rgm_batch_done(true);
        st.mark(10);
        self.keep_submitted(seq);
        self.own_gen += 1;
        self.last_seq = seq;
        if has_actions {
            self.last_action_seq = seq;
            self.cpu_action_seq = seq;
        }
        self.recording = false;
        // The graphics queue's batches wait for slots most (`tune slots_gfx`).
        let want = if self.name == "gpu-gfx" { super::tune::slots_gfx() } else { super::tune::slots() };
        let n = (want as usize).clamp(2, MAX_SLOTS).min(self.slot_limit);
        self.cur = (self.cur + 1) % n;
        self.ring_used = [0; 2];
        self.snap_used = 0;
        self.snaps.clear();
        note_seg_use(self.deferred, self.compute_cp, 0, self.pre_used);
        self.pre_used = 0;
        self.pres.clear();
        if let Some(t) = t_submit {
            SUBMIT_STATS[0].fetch_add(1, Ordering::Relaxed);
            SUBMIT_STATS[1].fetch_add(t.elapsed().saturating_sub(waited).as_nanos() as u64, Ordering::Relaxed);
        }
    }

    /// Drops the recorded batch (process shutdown: label waits were released,
    /// so the work may lack its inputs). The slot stays free.
    fn discard(&mut self) {
        self.label_close();
        if self.closed.is_empty() {
            self.open_gen.store(u64::MAX, Ordering::SeqCst);
        }
        self.keep_submitted(0);
        self.end_rendering();
        unsafe { ctx().device.end_command_buffer(self.cmd()).ok() };
        self.actions.clear();
        self.gpu_writes_open = false;
        self.checks.clear();
        self.race_pending.clear();
        if super::tune::race_checks() {
            self.clear_open_writes(None);
        }
        self.pending.clear();
        self.pending_log.clear();
        self.pending_promises.clear();
        self.shared_pending.clear();
        self.shared_kept.clear();
        self.batch_shaders.clear();
        self.resident.clear();
        super::resident::dropped(&std::mem::take(&mut self.res_pages));
        self.fe2_up.clear();
        self.fe2_spans.clear();
        self.fe2_mine.clear();
        self.f3_placed.clear();
        self.fe2_marked = false;
        self.fe2_pend.clear();
        self.fe2_pend_set.clear();
        self.rgm_batch_done(false);
        self.owned.clear();
        self.eager_action = None;
        self.pre_after.clear();
        self.mm_cp.clear();
        self.res_mine.clear();
        self.stage_used = 0;
        self.recording = false;
        self.ring_used = [0; 2];
        self.snap_used = 0;
        self.snaps.clear();
        let _ = self.end_pre();
        self.own_checks.clear();
        self.xcopies.clear();
        self.copy_wait = 0;
        note_seg_use(self.deferred, self.compute_cp, 0, self.pre_used);
        self.pre_used = 0;
        self.pres.clear();
    }

    /// The home of guest [base, base+size) for a binding (see `homes`):
    /// (buffer, offset of `base`, bytes from `base` to the home's end). The
    /// copies in the home needs are recorded first. A written binding too
    /// large for a VRAM shadow gets a home (`tune homes`; `create`: a sized
    /// buffer binding). A binding that only partly overlaps a home uses guest
    /// memory: the overlap's GPU-written parts are written back first.
    /// (buffer, offset of `base`, bytes available, the home when the caller
    /// must keep it for its work: returned only while no batch is being
    /// recorded, else the batch keeps it).
    pub(super) fn home_bind(&mut self, base: u64, size: u64, written: bool, kind: HomeUse) -> Option<(vk::Buffer, u64, u64, Option<std::sync::Arc<super::homes::HomeBuf>>)> {
        if size == 0 {
            return None;
        }
        // Mirror mode: the mirror is every range's home.
        if super::resident::mirror_mode() {
            self.mm_unsized = kind == HomeUse::Unbounded;
            return self.mirror_place(base, size, written).map(|(b, o, a)| (b, o, a, None));
        }
        // Front end 2: the mirror, inside its span.
        if super::fe2::on() && super::fe2::in_span(base, size) {
            return self.fe2_place(base, size, written).map(|(b, o, a)| (b, o, a, None));
        }
        if !super::tune::homes() && super::homes::any() {
            self.homes_evict_all();
        }
        if kind == HomeUse::Sized && written && size > super::tune::shadow_max() && size <= super::homes::MAX && super::tune::homes() {
            super::homes::note_large_write(base, size);
        }
        // A large read that may create a home (`tune home_reads`; only when
        // recording in order: a parallel chunk's later neighbours may have
        // bound the range in place already).
        let read_use = kind == HomeUse::Sized && !written && size >= super::tune::home_read_min() && !self.deferred && super::tune::homes() && super::tune::home_reads();
        if !super::homes::any() && !(kind == HomeUse::Index && super::tune::homes()) && !read_use {
            return None;
        }
        if self.deferred && super::homes::copy_in_pending(base, size) {
            self.stop("home copy-in");
            return None;
        }
        // The batch being recorded keeps each home once (its batch slot's
        // list): the home's shared reference count is touched for the first
        // binding only.
        let recording = self.recording;
        let mut taken: Option<std::sync::Arc<super::homes::HomeBuf>> = None;
        let b = {
            let held = self.slots.get(self.cur).map(|s| &s.homes);
            let mut keep = |h: &std::sync::Arc<super::homes::HomeBuf>| {
                if !(recording && held.is_some_and(|l| l.iter().any(|k| std::sync::Arc::ptr_eq(k, h)))) {
                    taken = Some(h.clone());
                }
            };
            let mut b = super::homes::bind(base, size, written, &mut keep);
            if b.is_none() && kind == HomeUse::Index && super::tune::homes() && super::homes::index_use(base, size) {
                b = super::homes::bind(base, size, written, &mut keep);
            }
            if b.is_none() && read_use && super::homes::read_use(base, size) {
                b = super::homes::bind(base, size, written, &mut keep);
            }
            b
        };
        let Some(b) = b else {
            if kind == HomeUse::Unbounded && !written {
                // An unbounded pointer's window (up to UNBOUNDED_MAX) that
                // passes over a home: assumed not to read it (writing the
                // home's GPU-written parts back for every such binding cost
                // more than the home saved), except within `tune
                // unbounded_near` bytes of its base, where those parts are
                // written back first (guest memory, which the pointer reads,
                // is stale there).
                if super::homes::overlaps(base, size) {
                    super::homes::UNBOUNDED_OVERLAPS.fetch_add(1, Ordering::Relaxed);
                    let inside = super::homes::overlaps(base, 4);
                    if inside {
                        super::homes::UNBOUNDED_STALE[0].fetch_add(1, Ordering::Relaxed);
                    }
                    if super::homes::any_stale(&[(base, size.min(4096))]) {
                        let n = super::homes::UNBOUNDED_STALE[1].fetch_add(1, Ordering::Relaxed);
                        if n < 20 {
                            crate::info!(
                                "gpu[{}]: unbounded pointer {base:#x} ({}) of shader {:#x} reads guest memory where home {} holds GPU-written data not written back{}",
                                self.name,
                                if inside { "inside the home" } else { "before it" },
                                super::images::BINDING_SHADER.with(|s| s.get()),
                                super::homes::describe(base, 4096),
                                if super::tune::unbounded_near() > 0 { " (written back first: `unbounded_near`)" } else { "" }
                            );
                        }
                    }
                    let near = super::tune::unbounded_near();
                    if near > 0 {
                        let w = [(base, size.min(near))];
                        if super::homes::any_stale(&w) {
                            super::homes::UNBOUNDED_STALE[2].fetch_add(1, Ordering::Relaxed);
                            self.home_write_back(&w);
                        }
                    }
                }
                return None;
            }
            if super::homes::overlaps(base, size) {
                if super::homes::PARTIAL.fetch_add(1, Ordering::Relaxed) < 40 {
                    crate::info!(
                        "gpu[{}]: binding {base:#x}+{size:#x} ({}) of shader {:#x} only partly overlaps a home: {}",
                        self.name,
                        if written { "written" } else { "read" },
                        super::images::BINDING_SHADER.with(|s| s.get()),
                        super::homes::describe(base, size)
                    );
                }
                self.home_write_back(&[(base, size)]);
                if written && !self.stopped {
                    // The binding writes guest memory in place over part of
                    // the home: that part is copied in again before the home's
                    // next GPU use, or the home's readers keep its old words
                    // (homes over GPU-written reads, `home_read_min` 1 MiB:
                    // queue 0x50 wrote 0x1475cdd1e0+0xef10, the last 3.6 KB of
                    // a 4.9 MB home and beyond, at the 5-3 load, and shaders
                    // looped on the stale words until the device was lost,
                    // 2026-10-24).
                    super::homes::written_in_place(base, size);
                }
            }
            return None;
        };
        if !b.copy_in.is_empty() {
            self.begin();
            self.end_rendering_then_barrier();
            let c = ctx();
            let cmd = self.cmd();
            for &(a, n, off) in &b.copy_in {
                for (src, src_off, ga, len) in guest_pieces(a, n) {
                    unsafe { c.device.cmd_copy_buffer(cmd, src, b.buf, &[vk::BufferCopy { src_offset: src_off, dst_offset: off + (ga - a), size: len }]) };
                    super::homes::COPIED_IN.fetch_add(len, Ordering::Relaxed);
                }
            }
            self.barrier();
            self.barrier_wanted = false;
            self.dispatched = false;
        }
        self.ensure_slot();
        if let Some(h) = &taken {
            let l = &mut self.slots[self.cur].homes;
            if !l.iter().any(|k| std::sync::Arc::ptr_eq(k, h)) {
                l.push(h.clone());
            }
        }
        // While recording the batch keeps it; else the caller keeps it for
        // the batch it then records into.
        Some((b.buf, b.offset, b.avail, if self.recording { None } else { taken }))
    }

    /// Keeps the home whose buffer is `buf` for the batch being recorded
    /// (once per batch; `held`: the home, if the caller has it); false if
    /// it no longer exists.
    pub(super) fn keep_home(&mut self, buf: vk::Buffer, held: Option<&std::sync::Arc<super::homes::HomeBuf>>) -> bool {
        // Front end 2's mirror lives as long as the process.
        if super::fe2::on() && super::resident::mirror_buffer().is_some_and(|(m, _)| m == buf) {
            return true;
        }
        self.ensure_slot();
        let l = &mut self.slots[self.cur].homes;
        if l.iter().any(|k| k.buf == buf) {
            return true;
        }
        match held.cloned().or_else(|| super::homes::handle(buf)) {
            Some(h) => {
                l.push(h);
                true
            }
            None => false,
        }
    }

    /// Writes every home's GPU-written parts back and drops the homes
    /// (`tune homes 0` while they exist).
    fn homes_evict_all(&mut self) {
        if self.deferred {
            self.stop("homes evicted");
            return;
        }
        let parts = super::homes::take_all();
        if parts.is_empty() {
            return;
        }
        self.begin();
        self.end_rendering_then_barrier();
        let c = ctx();
        let cmd = self.cmd();
        for (a, n, hb, off) in parts {
            for (dst, dst_off, ga, len) in guest_pieces(a, n) {
                unsafe { c.device.cmd_copy_buffer(cmd, hb.buf, dst, &[vk::BufferCopy { src_offset: off + (ga - a), dst_offset: dst_off, size: len }]) };
                super::homes::WRITE_BACK_BYTES.fetch_add(len, Ordering::Relaxed);
            }
            self.slots[self.cur].homes.push(hb);
        }
        self.barrier();
        self.barrier_wanted = false;
        self.dispatched = false;
        crate::info!("gpu[{}]: homes written back and dropped (tune homes 0)", self.name);
    }

    /// Clears images to zero in the batch being recorded (after the fill of
    /// their memory recorded just before: `Images::zero_filled`), between
    /// barriers.
    fn clear_images_zero(&mut self, images: &[(vk::Image, vk::Format)]) {
        self.begin();
        self.end_rendering();
        let d = &ctx().device;
        let cmd = self.cmd();
        unsafe {
            let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
            d.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[mb], &[], &[]);
            for &(image, format) in images {
                let range = vk::ImageSubresourceRange { aspect_mask: super::images::aspects(format), base_mip_level: 0, level_count: vk::REMAINING_MIP_LEVELS, base_array_layer: 0, layer_count: vk::REMAINING_ARRAY_LAYERS };
                if super::images::is_depth(format) {
                    d.cmd_clear_depth_stencil_image(cmd, image, vk::ImageLayout::GENERAL, &vk::ClearDepthStencilValue { depth: 0.0, stencil: 0 }, &[range]);
                } else {
                    d.cmd_clear_color_image(cmd, image, vk::ImageLayout::GENERAL, &vk::ClearColorValue::default(), &[range]);
                }
            }
            let a = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
            d.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::ALL_COMMANDS, vk::DependencyFlags::empty(), &[a], &[], &[]);
        }
    }

    /// Writes the write-watched (argument) homes' GPU-written parts back and
    /// drops the homes (`tune draw_args_gpu 0` while they exist; the main
    /// command processor when it plans a parallel run: no worker records).
    /// Later readers of those parts wait for this batch.
    pub fn evict_watched_homes(&mut self) {
        if self.deferred {
            return;
        }
        let parts = super::homes::take_watched();
        if parts.is_empty() {
            return;
        }
        self.begin();
        self.end_rendering_then_barrier();
        let c = ctx();
        let cmd = self.cmd();
        let mut written = Vec::with_capacity(parts.len());
        for (a, n, hb, off) in parts {
            for (dst, dst_off, ga, len) in guest_pieces(a, n) {
                unsafe { c.device.cmd_copy_buffer(cmd, hb.buf, dst, &[vk::BufferCopy { src_offset: off + (ga - a), dst_offset: dst_off, size: len }]) };
                super::homes::WRITE_BACK_BYTES.fetch_add(len, Ordering::Relaxed);
            }
            self.slots[self.cur].homes.push(hb);
            written.push((a, n));
        }
        self.barrier();
        self.barrier_wanted = false;
        self.dispatched = false;
        for (a, n) in written {
            self.note_write(a, n);
        }
    }

    /// Records copies of the GPU-written parts of the homes overlapping
    /// `ranges` back to guest memory, in order after all earlier work.
    fn home_write_back(&mut self, ranges: &[(u64, u64)]) {
        let mut parts = Vec::new();
        for &(a, n) in ranges {
            parts.extend(super::homes::stale_parts(a, n, true));
        }
        if parts.is_empty() {
            return;
        }
        if self.deferred {
            self.stop("home write-back");
            return;
        }
        self.begin();
        self.end_rendering_then_barrier();
        let c = ctx();
        let cmd = self.cmd();
        let total: u64 = parts.iter().map(|p| p.1).sum();
        self.ops[OP_WRITE_BACK_BYTES] += total;
        static LOGGED: AtomicU64 = AtomicU64::new(0);
        if total >= 1 << 20 && LOGGED.fetch_add(1, Ordering::Relaxed) < 40 {
            crate::info!("gpu[{}]: writing back {} KiB of homes for {:x?} (shader {:#x})", self.name, total >> 10, ranges, super::images::BINDING_SHADER.with(|s| s.get()));
        }
        for (a, n, hb, off) in parts {
            for (dst, dst_off, ga, len) in guest_pieces(a, n) {
                unsafe { c.device.cmd_copy_buffer(cmd, hb.buf, dst, &[vk::BufferCopy { src_offset: off + (ga - a), dst_offset: dst_off, size: len }]) };
                super::homes::WRITE_BACK_BYTES.fetch_add(len, Ordering::Relaxed);
            }
            self.slots[self.cur].homes.push(hb);
        }
        self.barrier();
        super::homes::WRITE_BACKS.fetch_add(1, Ordering::Relaxed);
        self.barrier_wanted = false;
        self.dispatched = false;
    }

    /// The batch's copy prologue, if it was begun, left open with the copies
    /// still to record into it (`tune submit_finish`: `finish_pre` on the
    /// submission thread).
    fn take_pre(&mut self) -> Option<(vk::CommandBuffer, Vec<(vk::Buffer, vk::Buffer, vk::BufferCopy)>, Vec<(vk::Buffer, vk::Buffer, vk::BufferCopy)>)> {
        if !std::mem::take(&mut self.pre_open) {
            self.pre_copies.clear();
            self.pre_after.clear();
            return None;
        }
        Some((self.slots[self.cur].pre, std::mem::take(&mut self.pre_copies), std::mem::take(&mut self.pre_after)))
    }

    /// Ends the batch's copy prologue, if it was begun: the copies complete
    /// before the batch's work (which begins with a full barrier).
    fn end_pre(&mut self) -> Option<vk::CommandBuffer> {
        if !std::mem::take(&mut self.pre_open) {
            self.pre_copies.clear();
            self.pre_after.clear();
            return None;
        }
        let (pre, bq) = (self.slots[self.cur].pre, self.slots[self.cur].bq);
        record_copies(pre, &mut self.pre_copies);
        if !self.pre_after.is_empty() {
            global_barrier(pre);
            record_copies(pre, &mut self.pre_after);
        }
        unsafe {
            global_barrier(pre);
            if bq != vk::QueryPool::null() {
                ctx().device.cmd_write_timestamp(pre, vk::PipelineStageFlags::BOTTOM_OF_PIPE, bq, BQ_PRE + 1);
            }
            if super::vk::labels_on() {
                super::vk::label_end(pre);
            }
            ctx().device.end_command_buffer(pre).ok();
        }
        Some(pre)
    }

    /// Records the batch's copies for the copy queue (`Slot::xpre`), if any.
    fn end_xpre(&mut self) -> Option<vk::CommandBuffer> {
        if self.xcopies.is_empty() {
            return None;
        }
        let x = self.slots[self.cur].xpre;
        unsafe {
            ctx().device.begin_command_buffer(x, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).ok();
            record_copies(x, &mut self.xcopies);
            ctx().device.end_command_buffer(x).ok();
        }
        Some(x)
    }

    /// A copy of guest [base, base+size) in video memory made by the GPU at
    /// the start of the batch, in its prologue (Slot::pre): (buffer, offset
    /// of the copy's 16-byte aligned start, bias of `base` in it). Bound
    /// instead of the imported system memory, which each invocation would
    /// read across PCIe. The prologue runs after all earlier submitted work
    /// and before this batch's, so the copy holds what an in-place read would
    /// see unless work recorded earlier in this batch, or a deferred CP
    /// write, writes the range (then None). None also: off or larger than
    /// `tune pre_max`, not one mapping, no space.
    /// A read-only binding of guest [base, base+size) served from the
    /// residency mirror (`gpu::resident`): (mirror buffer, offset of `base`
    /// in it, bytes available from `base`). Pages the mirror does not hold
    /// yet are uploaded by this batch (staged here, copied in its prologue),
    /// as many as its staging takes; the binding is refused until all are.
    fn resident_bind(&mut self, base: u64, size: u64) -> Option<(vk::Buffer, u64, u64)> {
        use super::resident::{self as res, Page};
        let (mbuf, _) = res::mirror()?;
        let end = base.checked_add(size)?;
        if base < res::MIRROR_BASE || end > res::MIRROR_END || size > RES_MAX {
            res::refused(0);
            return None;
        }
        let r = [(base, size)];
        if super::homes::overlaps(base, size) || self.overlaps_pending(&r) || CP_WRITES.latest(&r).is_some() || INFLIGHT.latest(&r).is_some() || self.resident.iter().any(|s| s.guest < end && base < s.guest + s.range) {
            res::refused(1);
            return None;
        }
        let (lo, hi) = (base & !(res::PAGE - 1), end.next_multiple_of(res::PAGE));
        let mut need = std::mem::take(&mut self.res_need);
        need.clear();
        let mut why = None;
        let mut p = lo;
        while p < hi {
            match res::page(p) {
                Page::Ready => {}
                Page::Pending if self.res_mine.contains(&p) => {}
                Page::Pending => {
                    why = Some(3);
                    break;
                }
                Page::Refuse => {
                    why = Some(2);
                    break;
                }
                Page::GpuWritten => {
                    why = Some(8);
                    break;
                }
                Page::Upload => need.push(p),
            }
            p += res::PAGE;
        }
        if why.is_none() && !need.is_empty() {
            why = self.resident_upload(lo, hi, mbuf, &mut need);
        }
        self.res_need = need;
        if let Some(w) = why {
            res::refused(w);
            return None;
        }
        res::served((hi - lo) / res::PAGE);
        Some((mbuf, base - res::MIRROR_BASE, hi - base))
    }

    /// Mirror mode (`DS_RESIDENT=2`; RENDERER.md "Mirror mode"): binds the
    /// mirror for any buffer binding inside its span. Pages the mirror lacks
    /// are copied into the batch's staging (claimed and armed, so the CPU's
    /// next write faults; volatile pages and others whose writes are not
    /// tracked are copied for every batch); a written binding's pages become
    /// GPU-owned (no-access once the batch is submitted). None: a page is
    /// host class (CP writes) or not readable; the old paths take it.
    /// Front end 2's placement (`fe2`): guest [base, base+size) bound from
    /// the mirror at its guest offset once the mirror holds it current
    /// (pages it lacks are copied from their import windows in the batch's
    /// prologue). Written ranges are imported (the write-back stores into
    /// them) and leave the batch with a write-back to record. A range
    /// reaching past the end of its mapping is placed up to there (the rest
    /// reads 0). None: nothing of it is mapped, or no video memory.
    fn fe2_place(&mut self, base: u64, size: u64, written: bool) -> Option<(vk::Buffer, u64, u64)> {
        if !super::fe2::callers_on() {
            if super::fe2census::on() || super::fe2::quiet_census_on() {
                super::fe2::set_caller(&self.name);
            }
            return self.fe2_place_inner(base, size, written);
        }
        super::fe2::set_caller(&self.name);
        let t0 = unsafe { core::arch::x86_64::_rdtsc() };
        let r = self.fe2_place_inner(base, size, written);
        super::fe2::note_place(unsafe { core::arch::x86_64::_rdtsc() }.wrapping_sub(t0));
        r
    }

    fn fe2_place_inner(&mut self, base: u64, size: u64, written: bool) -> Option<(vk::Buffer, u64, u64)> {
        use super::resident::{MIRROR_BASE, PAGE};
        // (Parts timed for `DS_FE2_CALLERS`.)
        let tsc = || if super::fe2::callers_on() { unsafe { core::arch::x86_64::_rdtsc() } } else { 0 };
        let t0 = tsc();
        let (mbuf, _) = super::resident::mirror_buffer()?;
        let end = base.checked_add(size)?;
        if size == 0 || !super::fe2::in_span(base, size) {
            return None;
        }
        let lo = base & !(PAGE - 1);
        // (The run of mappings around `base`, from this thread's cache.)
        let (_, mapped_end) = crate::mem::readable_span(base)?;
        let hi = end.next_multiple_of(PAGE).min(mapped_end.next_multiple_of(PAGE));
        if hi <= base {
            return None;
        }
        if let Err(e) = super::resident::back(lo, hi - lo) {
            self.report_error("front end 2: no video memory for the mirror (bound as zeros)", || format!("{lo:#x}..{hi:#x}: {e}"));
            return None;
        }
        let t1 = tsc();
        // Its uploads go into the prologue of the batch that will hold the
        // command: one is begun first (a binding before this one may have
        // closed the batch, and a deferred executor's next batch may take
        // another slot, whose prologue an earlier open would not have begun).
        if !self.recording {
            self.begin();
            if !self.recording || self.stopped {
                return None;
            }
        }
        if written {
            if span(lo, hi - lo).is_none() {
                super::fe2::note_unscatterable();
                self.report_error("front end 2: a written range cannot be imported (its GPU writes do not reach guest memory)", || format!("{lo:#x}..{hi:#x}"));
            }
            self.fe2_marked = true;
        }
        let t2 = tsc();
        // (Read before any take: what its pages are stamped with.)
        let ep = super::precache::epoch() as u32;
        let want = self.fe2_want(ep);
        self.fe2_min_want = self.fe2_min_want.min(want);
        if self.past_grow || self.stretch {
            self.past_want_min = self.past_want_min.min(want);
            self.past_ranges.push((lo, hi));
        }
        let mine = std::mem::take(&mut self.fe2_mine);
        let pend = std::mem::take(&mut self.fe2_pend_set);
        let mut up = std::mem::take(&mut self.fe2_up);
        let mut deferred = std::mem::take(&mut self.fe2_pend);
        let (first, first_pend) = (up.len(), deferred.len());
        // (Asked per page of the runs a check needs: empty sets are not
        // hashed into.)
        // (`tune fe2_defer_gfx`: the graphics queue's alone, not the compute
        // queues' many small batches with their label writes.)
        let defer = super::fe2::defer_checks() || (super::tune::fe2_defer_gfx() && self.name.starts_with("gpu-gfx"));
        let mine_any = !mine.is_empty() || !pend.is_empty();
        let r = super::fe2::ensure(lo, hi, want, ep, written, &|p| (!mine.is_empty() && mine.contains(&p)) || (!pend.is_empty() && pend.contains(&p)), mine_any, &mut |a, b| self.fe2_upload(mbuf, a, b, &mut up), &fe2_armable, defer.then_some(&mut deferred));
        // Past a label wait the CPU may write the pages it rewrites every
        // frame again before the label (FINDINGS.md 2026-10-29): a small
        // binding's such pages are uploaded as they are when the stretch
        // runs, whatever the check found.
        if (self.past_grow || self.stretch) && r.is_ok() && (hi - lo) / PAGE <= STRETCH_HOT_PAGES && super::tune::past_hot() {
            let mut hot = Vec::new();
            super::fe2::hot_pages(lo, hi, &mut hot);
            let own = |p: u64, up: &[(u64, u64)]| mine.contains(&p) || pend.contains(&p) || up[first..].iter().any(|&(s, e)| s <= p && p < e);
            'runs: for (a, b) in hot {
                let mut x = a;
                while x < b {
                    if own(x, &up) {
                        x += PAGE;
                        continue;
                    }
                    let mut y = x + PAGE;
                    while y < b && !own(y, &up) {
                        y += PAGE;
                    }
                    if !self.fe2_upload(mbuf, x, y, &mut up) {
                        break 'runs;
                    }
                    PAST_HOT[0].fetch_add(1, Ordering::Relaxed);
                    PAST_HOT[1].fetch_add((y - x) / PAGE, Ordering::Relaxed);
                    x = y;
                }
            }
        }
        let t3 = tsc();
        let (mut mine, mut pend) = (mine, pend);
        for &(a, b) in &up[first..] {
            let mut p = a;
            while p < b {
                mine.insert(p);
                if self.past_grow || self.stretch {
                    self.past_up.insert(p);
                }
                p += PAGE;
            }
        }
        for &(a, b) in &deferred[first_pend..] {
            let mut p = a;
            while p < b {
                pend.insert(p);
                p += PAGE;
            }
        }
        self.fe2_mine = mine;
        self.fe2_pend_set = pend;
        self.fe2_pend = deferred;
        self.fe2_up = up;
        if let Err(e) = r {
            self.report_error(&format!("front end 2: {e} (bound as the mirror holds it)"), || format!("{lo:#x}..{hi:#x}"));
        }
        // Deferred CP writes into the range not performed yet: the batch is
        // queued after them (its prologue's copies read what they wrote).
        if let Some(s) = CP_WRITES.latest(&[(base, size)]) {
            self.submit_after = self.submit_after.max(s);
        }
        if t0 != 0 {
            super::fe2::note_place_parts(t1.wrapping_sub(t0), t2.wrapping_sub(t1), tsc().wrapping_sub(t3));
        }
        Some((mbuf, base - MIRROR_BASE, hi - base))
    }

    /// The epoch front end 2's checks need now (`ep`: the epoch now): this
    /// queue's view, or the epoch now without one.
    fn fe2_want(&self, ep: u32) -> u32 {
        // Held past a label wait, the work runs after the label, whose writer
        // may synchronize again before writing it: the view now does not
        // cover the CPU writes it must see, and the epoch now does (as for
        // every check before views). With the view, a per-view copy kernel's
        // source passed stale in the mirror after a warp, its forwarded
        // words then differed from what the copy left (`forward`: "did not
        // land"), the block was no longer forwarded, and every later draw
        // reading it stopped the stretch and waited for the label (~9 ms
        // each, 732 times in one window; 2 of ~20 boots, 2026-10-28 night).
        match self.fe2_view.load(Ordering::Acquire) {
            _ if (self.past_grow || self.stretch) && super::fe2::stretch_epoch() => ep,
            super::fe2::NO_VIEW => ep,
            v => v.min(ep),
        }
    }

    /// The epoch a GPU-resolved run opened by the command being recorded was
    /// checked for: the lowest its placements needed (the epoch can advance
    /// while a command binds), else what a check would need now.
    pub(super) fn fe2_run_want(&self) -> u32 {
        match self.fe2_min_want {
            u32::MAX => self.fe2_want(super::precache::epoch() as u32),
            w => w,
        }
    }

    /// Front end 2 for a GPU-resolved command's per-item chains (stage F3;
    /// `gpures`, `Spec::gpu_res`, `gpu_loads`, `pt_linear`), their chains and
    /// V#s read from guest memory now: the ranges its shader-resolved loads
    /// read are read by the shader from guest memory through the page table
    /// (data the CPU path copies into the SGPR data when it records: per-item
    /// blocks the title rewrites every frame, which the mirror would take
    /// again in every view), so they must be imported and no GPU or CP work
    /// may still be pending on them; the ranges its shader-resolved buffers
    /// reach are placed in the mirror as of this executor's view as any
    /// binding is (once per batch and view: `f3_placed`). The words read for
    /// them must be current too (else `F3_PENDING`: the command is recorded
    /// with CPU-resolved chains, which wait). Err: why not (an index into
    /// `F3_REFUSED`).
    fn fe2_gpu_ranges(&mut self, found: &shaders::Found, addr: u64, user: &[u32]) -> Result<(), usize> {
        let mut ranges = std::mem::take(&mut self.f3_ranges);
        let mut loads = std::mem::take(&mut self.f3_loads);
        let mut reads = std::mem::take(&mut self.f3_reads);
        ranges.clear();
        loads.clear();
        reads.clear();
        let r = self.fe2_gpu_ranges_in(found, addr, user, &mut loads, &mut ranges, &mut reads);
        self.f3_ranges = ranges;
        self.f3_loads = loads;
        self.f3_reads = reads;
        if let Err(k) = r {
            F3_REFUSED[k].fetch_add(1, Ordering::Relaxed);
        }
        r
    }

    fn fe2_gpu_ranges_in(&mut self, found: &shaders::Found, addr: u64, user: &[u32], loads: &mut Vec<(u64, u64)>, ranges: &mut Vec<(u64, u64)>, reads: &mut Vec<(u64, u64)>) -> Result<(), usize> {
        use super::resident::PAGE;
        // (Parts timed with `DS_STEP_LAPS`: `F3_LAPS`.)
        let tsc = || if laps() { unsafe { core::arch::x86_64::_rdtsc() } } else { 0 };
        let t0 = tsc();
        let r = shaders::gpu_chain_ranges(addr, &found.tr.plan, &found.tr.spec, user, loads, ranges, reads);
        let t1 = tsc();
        match r {
            Ok(()) => {}
            Err(0) => return Err(F3_UNREADABLE),
            Err(_) => {
                GPU_FORMAT_MISSES.fetch_add(1, Ordering::Relaxed);
                return Err(F3_FORMAT);
            }
        }
        // The loads' ranges in import windows (the page table maps them for
        // the batches queued from now on); outside the span the shader reads
        // zeros, which the CPU path would not for readable memory there.
        loads.retain(|&(a, n)| a != 0 && n != 0);
        if f3_regions_on() {
            note_f3_regions(addr, loads, ranges);
        }
        for &(a, n) in loads.iter() {
            if !super::fe2::in_span(a, n) {
                if super::cp::guest_readable(a, n) {
                    return Err(F3_OUTSIDE);
                }
                continue;
            }
            if span(a, n).is_none() {
                return Err(F3_UNREADABLE);
            }
        }
        let pending = |s: &mut Self, r: &[(u64, u64)]| !r.is_empty() && (s.would_wait(r) || CP_WRITES.latest(r).is_some());
        if pending(self, reads) || pending(self, loads) {
            return Err(F3_PENDING);
        }
        let t2 = tsc();
        // (Read on the CPU now: checked when held work settles, as a
        // preparation's reads are.)
        if self.held {
            self.held_reads.extend_from_slice(reads);
            self.note_kinds(2, reads.len());
        }
        // Within their mappings (unbounded descriptors reach far past them),
        // runs on the same or adjacent pages placed as one.
        ranges.sort_unstable();
        let want = self.fe2_want(super::precache::epoch() as u32);
        if self.past_grow || self.stretch {
            self.past_want_min = self.past_want_min.min(want);
        }
        let mut run: Option<(u64, u64)> = None;
        let (mut placed, mut bytes, mut mine) = (0u64, 0u64, 0u64);
        for i in 0..=ranges.len() {
            let next = match ranges.get(i) {
                Some(&(a, n)) => {
                    if n == 0 || a == 0 {
                        continue;
                    }
                    let Some((_, e)) = mapping(a) else { return Err(F3_UNREADABLE) };
                    // (Unchecked buffers to the mapping's end, as bound in full.)
                    let n = if n == u64::MAX { (e - a).min(UNBOUNDED_MAX) } else { n.min(e - a) };
                    if !super::fe2::in_span(a, n) {
                        return Err(F3_OUTSIDE);
                    }
                    match run.as_mut() {
                        Some(l) if a <= l.1.next_multiple_of(PAGE) => {
                            l.1 = l.1.max(a + n);
                            continue;
                        }
                        _ => Some((a, a + n)),
                    }
                }
                None => None,
            };
            if let Some((a, b)) = std::mem::replace(&mut run, next) {
                // Small runs whose pages this batch uploads already (a ring of
                // per-draw blocks: the draws before in the batch placed them)
                // are current for all of its work: only deferred CP writes
                // into them still order the batch.
                if (b - a <= F3_MINE_PAGES * PAGE && (a & !(PAGE - 1)..b).step_by(PAGE as usize).all(|p| self.fe2_mine.contains(&p))) || self.f3_placed.contains(&(a, b, want)) {
                    if let Some(s) = CP_WRITES.latest(&[(a, b - a)]) {
                        self.submit_after = self.submit_after.max(s);
                    }
                    mine += 1;
                    continue;
                }
                if self.fe2_place(a, b - a, false).is_none() {
                    return Err(F3_PLACE);
                }
                if self.f3_placed.len() < F3_PLACED_MAX {
                    self.f3_placed.insert((a, b, want));
                }
                placed += 1;
                bytes += b - a;
            }
        }
        F3_PLACED[0].fetch_add(placed, Ordering::Relaxed);
        F3_PLACED[1].fetch_add(bytes, Ordering::Relaxed);
        if t0 != 0 {
            let t3 = tsc();
            for (k, v) in [1, t1.wrapping_sub(t0), t2.wrapping_sub(t1), t3.wrapping_sub(t2), (loads.len() + ranges.len()) as u64, placed, mine].into_iter().enumerate() {
                F3_LAPS[k].fetch_add(v, Ordering::Relaxed);
            }
        }
        Ok(())
    }

    /// Records the copy of guest [a, b) into the mirror at its offset, in
    /// the batch's prologue: from the import windows over it (those runs
    /// appended to `up`: current once the batch is queued), else, for
    /// memory the driver cannot import, from the CPU's copy now (current
    /// for this batch only: CPU writes after it are not in it).
    fn fe2_upload(&mut self, mbuf: vk::Buffer, a: u64, b: u64, up: &mut Vec<(u64, u64)>) -> bool {
        use super::resident::{MIRROR_BASE, PAGE};
        // (Only into a batch being recorded: `fe2_place` begins one.)
        if !self.recording {
            return false;
        }
        let _ = self.open_pre();
        let mut x = a;
        while x < b {
            let e = ((x & !(WINDOW - 1)) + WINDOW).min(b);
            match span(x, e - x) {
                Some((buf, start, wend)) => {
                    let e = e.min(wend);
                    if !self.fe2_spans.iter().any(|&(s, b)| s == start && b == buf) {
                        self.fe2_spans.push((start, buf));
                    }
                    self.pre_copies.push((buf, mbuf, vk::BufferCopy { src_offset: x - start, dst_offset: x - MIRROR_BASE, size: e - x }));
                    match up.last_mut() {
                        Some(l) if l.1 == x => l.1 = e,
                        _ => up.push((x, e)),
                    }
                    x = e;
                }
                None => {
                    if !crate::mem::os_readable(x, PAGE) || self.mirror_stage(mbuf, x).is_none() {
                        let readable = crate::mem::os_readable(x, PAGE);
                        self.report_error("front end 2: a page to copy into the mirror has no import window and cannot be copied by the CPU", || format!("{x:#x} (run {a:#x}..{b:#x}; readable {readable})"));
                        return false;
                    }
                    super::fe2::note_memcpy(1);
                    if self.held {
                        // (Read now: checked when held work settles.)
                        self.held_reads.push((x, PAGE));
                        self.note_kinds(2, 1);
                    }
                    x += PAGE;
                }
            }
        }
        true
    }

    /// Front end 2's checks the batch's bindings deferred, at its close
    /// (`fe2::resolve`): their uploads go into its prologue.
    fn fe2_resolve(&mut self) {
        let Some((mbuf, _)) = super::resident::mirror_buffer() else { return };
        let mut runs = std::mem::take(&mut self.fe2_pend);
        self.fe2_pend_set.clear();
        let mine = std::mem::take(&mut self.fe2_mine);
        let mut up = std::mem::take(&mut self.fe2_up);
        let first = up.len();
        let r = super::fe2::resolve(&mut runs, &|p| mine.contains(&p), &mut |a, b| self.fe2_upload(mbuf, a, b, &mut up), &fe2_armable);
        let mut mine = mine;
        for &(a, b) in &up[first..] {
            let mut p = a;
            while p < b {
                mine.insert(p);
                p += super::resident::PAGE;
            }
        }
        self.fe2_mine = mine;
        self.fe2_up = up;
        self.fe2_pend = runs;
        if let Err(e) = r {
            self.report_error(&format!("front end 2: {e} at a batch's close (its work reads the mirror as it is)"), String::new);
        }
    }

    /// Front end 2's write-back, recorded now (outside a render pass): the
    /// dwords GPU work marked go from the mirror to guest memory. Whether
    /// it was recorded (with its barriers).
    fn fe2_scatter(&mut self) -> bool {
        // (`DS_PROFILE_GPU_STAMPS=seg`: a segment of its own, `front end 2
        // write-backs`; its own barriers write no segment stamps, and its
        // time went to the commands before it.)
        let seg = seg_stamps();
        if seg {
            self.barrier();
        }
        let cmd = self.cmd();
        let recorded = if let Some(pipe) = super::fe2::record_scatter(cmd) {
            // The next dispatch binds its own pipeline and descriptors.
            self.dyncache.cs_pipeline = pipe;
            true
        } else {
            false
        };
        // (Its first barrier holds it after all earlier work and its last
        // all later work after it: a full one for what follows.)
        if recorded && super::tune::fe2_cpw_light() {
            self.barrier_noted();
        }
        self.ops[OP_FE2_SCATTERS] += 1;
        if seg {
            self.barrier();
        }
        self.fe2_marked = false;
        recorded
    }

    fn mirror_place(&mut self, base: u64, size: u64, written: bool) -> Option<(vk::Buffer, u64, u64)> {
        use super::resident as res;
        let (mbuf, _) = res::mirror()?;
        let end = base.checked_add(size)?;
        if size == 0 || base < res::MIRROR_BASE || end > res::MIRROR_END {
            return None;
        }
        let (lo, hi) = (base & !(res::PAGE - 1), end.next_multiple_of(res::PAGE));
        let mut claim = std::mem::take(&mut self.res_need);
        claim.clear();
        let mut copy = std::mem::take(&mut self.mm_copy);
        copy.clear();
        let host = !res::mm_scan(lo, hi, &self.res_mine, &mut claim, &mut copy);
        // A descriptor reaching past the end of its mappings (generic sizes
        // on ring allocations): the mapped part is placed, the rest reads 0.
        if !host && !staged_readable(&claim, &copy) {
            self.res_need = claim;
            self.mm_copy = copy;
            if let Some((_, me, _)) = mapping_run(base, lo, hi) {
                if me > base && me < end {
                    return self.mirror_place(base, me - base, written);
                }
            }
            res::refused(6);
            return None;
        }
        let ok = !host && self.mirror_fill(lo, hi, mbuf, &mut claim, &mut copy);
        self.res_need = claim;
        self.mm_copy = copy;
        if !ok {
            res::refused(if host { 2 } else { 6 });
            return None;
        }
        if written {
            self.owned.push((lo, hi));
        }
        // Deferred CP writes into the range not performed yet: the batch is
        // submitted after them and copies what they changed at its close.
        if let Some(s) = CP_WRITES.latest(&[(base, size)]) {
            self.submit_after = self.submit_after.max(s);
            self.mm_cp.push((lo, hi));
        }
        res::served((hi - lo) / res::PAGE);
        let pages = (hi - lo) / res::PAGE;
        let bucket = match pages {
            0..16 => 0,
            16..256 => 1,
            256..4096 => 2,
            _ => 3,
        };
        let k = (written as usize) * 8 + (self.mm_unsized as usize) * 4 + bucket;
        MM_PLACE[k][0].fetch_add(1, Ordering::Relaxed);
        MM_PLACE[k][1].fetch_add(pages, Ordering::Relaxed);
        Some((mbuf, base - res::MIRROR_BASE, hi - base))
    }

    /// Mirror mode's write-back: at the batch's end the gather kernel
    /// (`rdna::kernels::mirror_gather`) records the pages of the batch's
    /// touched-page list (the pages its shaders marked dwords in:
    /// `Spec::write_marks`), and the pages of ranges an earlier gather left
    /// marks in (`resident::take_regather`), with their mask words, in host
    /// memory; the batch's first completion action copies those pages from
    /// the mirror and stores the marked dwords to guest memory
    /// (`resident::apply_masks_copy`; lazily with `DS_MM_LAZY=1`) before its
    /// command-processor effects show the batch done. Dwords not written
    /// keep what the CPU wrote. When the list or the output overflows, the
    /// batch's written ranges (`owned`) are gathered again by range.
    fn record_mirror_writeback(&mut self) -> Result<(), String> {
        use super::resident as res;
        let Some(marks) = res::marks() else { return Ok(()) };
        let mut segs: Vec<(u64, u64)> = std::mem::take(&mut self.owned);
        segs.sort_unstable();
        let mut merged: Vec<(u64, u64)> = Vec::with_capacity(segs.len());
        for (a, e) in segs {
            match merged.last_mut() {
                Some(last) if a <= last.1 => last.1 = last.1.max(e),
                _ => merged.push((a, e)),
            }
        }
        // Lazy write-backs: the written ranges are no-access from now on
        // (before the batch is handed over), until a CPU access settles them.
        let own = res::own_begin(&merged);
        self.mm_own = own.clone();
        let mut leftover = res::take_regather();
        leftover.sort_unstable();
        let mut regather: Vec<(u64, u64)> = Vec::with_capacity(leftover.len());
        for (a, e) in leftover {
            match regather.last_mut() {
                Some(last) if a <= last.1 => last.1 = last.1.max(e),
                _ => regather.push((a, e)),
            }
        }
        let range_pages: u64 = regather.iter().map(|&(a, e)| (e - a) / res::PAGE).sum();
        let pages: u64 = merged.iter().map(|&(a, e)| (e - a) / res::PAGE).sum::<u64>().min(rdna::kernels::MARK_LIST_CAP as u64);
        // Room for every page of the written and leftover ranges (records of
        // 132 bytes) up to MM_RECORDS_MAX: a page left over is written back
        // with a later batch, after CPU writes it may then overwrite.
        let cap = (pages + range_pages).clamp(1, MM_RECORDS_MAX);
        let rec = 4 * rdna::kernels::GATHER_MASK_RECORD_DWORDS as u64;
        let (obuf, optr, room) = self.mm_buffers(1, 4 * rdna::kernels::GATHER_HEADER_DWORDS as u64 + cap * rec, rec)?;
        let cap = room.min(u32::MAX as u64) as u32;
        unsafe {
            std::ptr::write_volatile(optr as *mut u32, 0);
            std::ptr::write_volatile((optr as *mut u32).add(1), 0);
        }
        let pipe = gather_pipe()?;
        let c = ctx();
        let cmd = self.cmd();
        global_barrier(cmd);
        unsafe { c.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipe.pipeline) };
        self.dyncache.cs_pipeline = pipe.pipeline;
        let set = self.alloc_set(pipe.set_layout)?;
        let infos = [vk::DescriptorBufferInfo { buffer: marks, offset: 0, range: res::MARK_BYTES }, vk::DescriptorBufferInfo { buffer: obuf, offset: 0, range: vk::WHOLE_SIZE }];
        let writes: Vec<vk::WriteDescriptorSet> = infos
            .iter()
            .enumerate()
            .map(|(k, info)| vk::WriteDescriptorSet::default().dst_set(set).dst_binding(k as u32).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(std::slice::from_ref(info)))
            .collect();
        let push = |c: &super::vk::Ctx, v: [u32; 4]| {
            let pc: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
            unsafe { c.device.cmd_push_constants(cmd, pipe.layout, vk::ShaderStageFlags::COMPUTE, 0, &pc) };
        };
        unsafe {
            c.device.update_descriptor_sets(&writes, &[]);
            c.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, pipe.layout, 0, &[set], &[]);
        }
        // The touched-page list (its header is the dispatch's arguments).
        push(c, [0, cap, 0, 1]);
        unsafe { c.device.cmd_dispatch_indirect(cmd, marks, rdna::kernels::MARK_LIST_AT as u64 * 4) };
        // Ranges an earlier gather left marks in, by range (after the list:
        // its pages' records come first).
        if !regather.is_empty() {
            let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
            unsafe { c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &[mb], &[], &[]) };
            const PART: u64 = 1 << 20;
            for &(a, e) in &regather {
                let n_all = (e - a) / res::PAGE;
                let mut at = 0u64;
                while at < n_all {
                    let n = (n_all - at).min(PART);
                    let first = (a - res::MIRROR_BASE) / res::PAGE + at;
                    push(c, [n as u32, cap, first as u32, 0]);
                    unsafe { c.device.cmd_dispatch(cmd, n as u32, 1, 1) };
                    at += n;
                }
            }
        }
        unsafe {
            let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::HOST_READ);
            c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::HOST, vk::DependencyFlags::empty(), &[mb], &[], &[]);
        }
        let base = optr as usize;
        let action: Action = Box::new(move || {
            let out = base as *const u32;
            let listed = unsafe { std::ptr::read_volatile(out.add(1)) };
            let over = if res::lazy_writebacks() { res::apply_masks(out, cap) } else { res::apply_masks_copy(out, cap) };
            if let Some(o) = &own {
                res::own_done(o);
            }
            // A full list or output: the batch's written ranges keep marks
            // (the pages past them): gathered by range next time.
            if over > 0 || listed > rdna::kernels::MARK_LIST_CAP {
                res::regather(&merged, over.max(listed.saturating_sub(rdna::kernels::MARK_LIST_CAP)));
            }
        });
        if self.deferred {
            self.eager_action = Some(action);
        } else {
            self.actions.insert(0, action);
        }
        Ok(())
    }

    /// The current slot's host buffer `k` (0: records with dwords, 1: mask
    /// records) for mirror write-backs, grown to hold at least `bytes`:
    /// (buffer, its mapping, records it holds of `record` bytes after the
    /// header).
    fn mm_buffers(&mut self, k: usize, bytes: u64, record: u64) -> Result<(vk::Buffer, *mut u8, u64), String> {
        let c = ctx();
        let s = &mut self.slots[self.cur];
        let o = &mut s.mm_out[k];
        if o.3 < bytes {
            unsafe {
                if o.0 != vk::Buffer::null() {
                    c.device.destroy_buffer(o.0, None);
                    c.device.free_memory(o.1, None);
                }
            }
            let cap = bytes.max(1 << 20).next_power_of_two();
            let (b, m, ptr) = c.host_buffer(cap, vk::BufferUsageFlags::STORAGE_BUFFER)?;
            *o = (b, m, ptr, cap);
        }
        let header = 4 * rdna::kernels::GATHER_HEADER_DWORDS as u64;
        Ok((o.0, o.2, (o.3 - header) / record))
    }

    /// Mirror mode: the pages of placements that deferred CP writes wrote
    /// into (performed by now: the batch waited for them before its close)
    /// are copied into the mirror again (the prologue), those the CPU's CP
    /// writes changed (`resident::take_cp_pages`).
    fn mirror_cp_copies(&mut self) -> Result<(), String> {
        use super::resident as res;
        let Some((mbuf, _)) = res::mirror() else { return Ok(()) };
        let mut ranges = std::mem::take(&mut self.mm_cp);
        ranges.sort_unstable();
        let mut at = 0;
        for (lo, hi) in ranges {
            // Inside a range done already: nothing left.
            if lo.max(at) >= hi {
                continue;
            }
            for p in res::take_cp_pages(lo.max(at), hi) {
                if self.mirror_stage(mbuf, p).is_none() {
                    return Err(format!("no staging to copy {p:#x}"));
                }
            }
            at = at.max(hi);
        }
        Ok(())
    }

    /// Uploads the pages of `claim` (claimed and armed first) and `copy`
    /// into the mirror through the batch's staging. False: guest memory not
    /// readable there, or no staging.
    fn mirror_fill(&mut self, lo: u64, hi: u64, mbuf: vk::Buffer, claim: &mut Vec<u64>, copy: &mut Vec<u64>) -> bool {
        use super::resident as res;
        if claim.is_empty() && copy.is_empty() {
            return true;
        }
        let _readers = crate::mem::GUEST_READERS.read();
        // Readable where pages are copied (GPU-owned pages in between are
        // no-access: the mirror holds them).
        if !staged_readable(claim, copy) {
            return false;
        }
        if res::back(lo, hi - lo).is_err() {
            return false;
        }
        let claimed = if claim.is_empty() { 0 } else { res::claim_pages(claim) };
        // Pages another thread claimed meanwhile: copied without a claim.
        copy.extend_from_slice(&claim[claimed..]);
        let _ = self.open_pre();
        for (k, &pg) in claim[..claimed].iter().chain(copy.iter()).enumerate() {
            // Pages with a baseline: only the dwords the CPU changed (dwords
            // GPU work wrote meanwhile stay in the mirror).
            match res::baseline_diff(pg, pg as *const u32, k >= claimed) {
                Some(runs) => {
                    if !self.mirror_stage_runs(mbuf, pg, &runs) {
                        return false;
                    }
                }
                None => {
                    if self.mirror_stage(mbuf, pg).is_none() {
                        return false;
                    }
                }
            }
            if k < claimed {
                self.res_pages.push(pg);
            }
            self.res_mine.insert(pg);
        }
        true
    }

    /// Copies the dword runs `runs` (first dword, dwords) of guest page `pg`
    /// into the batch's staging and records their copies into the mirror.
    fn mirror_stage_runs(&mut self, mbuf: vk::Buffer, pg: u64, runs: &[(u32, u32)]) -> bool {
        use super::resident as res;
        let bytes: u64 = runs.iter().map(|r| 4 * r.1 as u64).sum();
        if bytes == 0 {
            return true;
        }
        let (sbuf, sptr, mut at) = if self.stage_used + bytes <= STAGE_SEG {
            let Some((sbuf, sptr)) = self.stage_buffer() else { return false };
            let at = self.stage_used;
            self.stage_used += bytes;
            (sbuf, sptr, at)
        } else {
            if self.slots[self.cur].stage_extra.is_empty() || self.extra_used + bytes > EXTRA_SEG {
                match ctx().host_buffer(EXTRA_SEG, vk::BufferUsageFlags::TRANSFER_SRC) {
                    Ok(x) => {
                        self.slots[self.cur].stage_extra.push(x);
                        self.extra_used = 0;
                    }
                    Err(e) => {
                        crate::warn!("gpu[{}]: no extra staging for mirror uploads ({e})", self.name);
                        return false;
                    }
                }
            }
            let (sbuf, _, sptr) = *self.slots[self.cur].stage_extra.last().expect("extra staging");
            let at = self.extra_used;
            self.extra_used += bytes;
            (sbuf, sptr, at)
        };
        for &(d, n) in runs {
            let len = 4 * n as u64;
            unsafe { std::ptr::copy_nonoverlapping((pg + 4 * d as u64) as *const u8, sptr.add(at as usize), len as usize) };
            self.pre_copies.push((sbuf, mbuf, vk::BufferCopy { src_offset: at, dst_offset: pg - res::MIRROR_BASE + 4 * d as u64, size: len }));
            at += len;
        }
        true
    }

    /// Copies guest page `pg` into the batch's staging and records its copy
    /// into the mirror (the batch's prologue); past STAGE_SEG, into extra
    /// staging buffers of the slot. Returns where the copy is.
    fn mirror_stage(&mut self, mbuf: vk::Buffer, pg: u64) -> Option<*const u8> {
        use super::resident as res;
        if self.stage_used + res::PAGE <= STAGE_SEG {
            let (sbuf, sptr) = self.stage_buffer()?;
            let at = self.stage_used;
            unsafe { std::ptr::copy_nonoverlapping(pg as *const u8, sptr.add(at as usize), res::PAGE as usize) };
            self.pre_copies.push((sbuf, mbuf, vk::BufferCopy { src_offset: at, dst_offset: pg - res::MIRROR_BASE, size: res::PAGE }));
            self.stage_used += res::PAGE;
            return Some(unsafe { sptr.add(at as usize) } as *const u8);
        }
        if self.slots[self.cur].stage_extra.is_empty() || self.extra_used + res::PAGE > EXTRA_SEG {
            match ctx().host_buffer(EXTRA_SEG, vk::BufferUsageFlags::TRANSFER_SRC) {
                Ok(x) => {
                    self.slots[self.cur].stage_extra.push(x);
                    self.extra_used = 0;
                }
                Err(e) => {
                    crate::warn!("gpu[{}]: no extra staging for mirror uploads ({e})", self.name);
                    return None;
                }
            }
        }
        let (sbuf, _, sptr) = *self.slots[self.cur].stage_extra.last().expect("extra staging");
        let at = self.extra_used;
        unsafe { std::ptr::copy_nonoverlapping(pg as *const u8, sptr.add(at as usize), res::PAGE as usize) };
        self.pre_copies.push((sbuf, mbuf, vk::BufferCopy { src_offset: at, dst_offset: pg - res::MIRROR_BASE, size: res::PAGE }));
        self.extra_used += res::PAGE;
        Some(unsafe { sptr.add(at as usize) } as *const u8)
    }

    /// Uploads `need` (pages of [lo, hi) the mirror lacks) as far as the
    /// batch's staging takes. None when all were uploaded, else why not
    /// (`resident::REFUSE_NAMES` index).
    fn resident_upload(&mut self, lo: u64, hi: u64, mbuf: vk::Buffer, need: &mut Vec<u64>) -> Option<usize> {
        use super::resident as res;
        // Imported pages cannot be armed: host class from now on.
        if let Some((a, e)) = imported_overlap(lo, hi).filter(|_| !crate::sections::on()) {
            res::mark_host(a, e - a);
            return Some(4);
        }
        let _readers = crate::mem::GUEST_READERS.read();
        if !crate::mem::os_readable(lo, hi - lo) {
            return Some(6);
        }
        let room = (STAGE_SEG - self.stage_used) / res::PAGE;
        let partial = (need.len() as u64) > room;
        need.truncate(room as usize);
        if need.is_empty() {
            return Some(5);
        }
        if res::back(need[0], need[need.len() - 1] + res::PAGE - need[0]).is_err() {
            return Some(7);
        }
        let Some((sbuf, sptr)) = self.stage_buffer() else { return Some(5) };
        let claimed = res::claim_pages(need);
        if claimed > 0 {
            let _ = self.open_pre();
        }
        for &pg in &need[..claimed] {
            let at = self.stage_used;
            unsafe { std::ptr::copy_nonoverlapping(pg as *const u8, sptr.add(at as usize), res::PAGE as usize) };
            self.pre_copies.push((sbuf, mbuf, vk::BufferCopy { src_offset: at, dst_offset: pg - res::MIRROR_BASE, size: res::PAGE }));
            self.stage_used += res::PAGE;
            self.res_pages.push(pg);
            self.res_mine.insert(pg);
        }
        if claimed < need.len() {
            Some(6)
        } else if partial {
            Some(5)
        } else {
            None
        }
    }

    /// The slot's staging ring for mirror uploads (made on first use).
    fn stage_buffer(&mut self) -> Option<(vk::Buffer, *mut u8)> {
        let s = &mut self.slots[self.cur];
        if s.stage.is_none() {
            match ctx().host_buffer(STAGE_SEG, vk::BufferUsageFlags::TRANSFER_SRC) {
                Ok((b, _m, ptr)) => {
                    SLOT_MEM[5].fetch_add(STAGE_SEG, Ordering::Relaxed);
                    s.stage = Some((b, ptr))
                }
                Err(e) => {
                    crate::warn!("gpu[{}]: no staging for mirror uploads ({e})", self.name);
                    return None;
                }
            }
        }
        s.stage
    }

    pub(super) fn pre_copy(&mut self, base: u64, size: u64) -> Option<(vk::Buffer, u64, u64)> {
        let mut clk = PreClock::start();
        // Over `pre_max`: only as a shared copy (`tune pre_big_max`).
        let big = size > super::tune::pre_max();
        let shares = self.shares_pre && (!self.batch_alt || self.batch_alt_share);
        if size == 0 || (big && !(size <= super::tune::pre_big_max() && shares && super::precache::enabled())) || self.pre_failed {
            return None;
        }
        if super::homes::overlaps(base, size) {
            if big {
                note_big_refusal(base, size, "a video-memory home overlaps it");
            }
            return None;
        }
        let r = [(base, size)];
        // Writes of this batch come after the prologue; those of submitted
        // batches (this queue's too) before it (`tune pre_inflight`; 0: also
        // refused while this queue's submitted batches still run).
        let pending = if super::tune::pre_inflight() { self.pending.overlaps_any(&r) || self.main_pending(&r) } else { self.overlaps_pending(&r) };
        if pending || CP_WRITES.latest(&r).is_some() {
            if big {
                note_big_refusal(base, size, if pending { "GPU work of this queue writes it" } else { "a deferred CP write" });
            }
            return None;
        }
        let al = base & !15;
        let bias = base - al;
        if let Some(&(buf, at)) = self.pres.get(&(base, size)) {
            PRE_REUSED.fetch_add(1, Ordering::Relaxed);
            PreClock::set(&mut clk, 1);
            return Some((buf, at, bias));
        }
        let n = (bias + size).next_multiple_of(4);
        let share = shares && super::precache::enabled() && super::precache::worth(size) && !(self.long && super::precache::gpu_written_lately(base, size));
        if share {
            // Copied by an earlier submitted batch (`gpu::precache`).
            if let Some((chunk, at, made)) = super::precache::lookup(base, size) {
                PRE_REUSED.fetch_add(1, Ordering::Relaxed);
                if self.held {
                    // (Taken before; checked when held work settles.)
                    self.held_reads.push((base, size));
                    self.note_kinds(0, 1);
                }
                self.copy_wait = self.copy_wait.max(made);
                let buf = chunk.buf;
                self.slots[self.cur].keep.push(chunk);
                self.pres.insert((base, size), (buf, at));
                PreClock::set(&mut clk, 2);
                return Some((buf, at, bias));
            }
        }
        PreClock::set(&mut clk, 5);
        let off = self.pre_used.next_multiple_of(self.ring_align);
        let Some((src, start, _)) = span(al, n) else {
            if big {
                note_big_refusal(base, size, "not one mapping");
            }
            return None;
        };
        // (Profiling: why no shared copy served it; left out of the clock.)
        let t_why = clk.as_ref().map(|_| unsafe { core::arch::x86_64::_rdtsc() });
        let why = if !profiling() {
            0
        } else if share {
            super::precache::miss_reason(base, size)
        } else {
            9
        };
        if let (Some(c), Some(t)) = (clk.as_mut(), t_why) {
            c.t0 = c.t0.wrapping_add(unsafe { core::arch::x86_64::_rdtsc() }.wrapping_sub(t));
        }
        let shared = if share { super::precache::insert(base, size, n) } else { None };
        // The batch's own segment for copies not shared (big ones are read
        // in place then).
        if shared.is_none() && big {
            note_big_refusal(base, size, "the shared copies declined it");
            return None;
        }
        if shared.is_none() && !self.pre_room(off + n) {
            PRE_FULL.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        let (buf, off) = match &shared {
            Some((chunk, at, _)) => (chunk.buf, *at),
            None => (self.slots[self.cur].pre_buf?, off),
        };
        let region = vk::BufferCopy { src_offset: al - start, dst_offset: off, size: n };
        // On the copy queue when no submitted batch writes the range: it
        // waits for none, and the batch waits for it (`submit_batch`). Else
        // in the prologue on the main queue, after all earlier work. (Not in
        // a parallel chunk: earlier chunks of its run, not submitted yet, may
        // write the range.)
        let on_copy = self.slots[self.cur].xpre != vk::CommandBuffer::null() && !self.deferred && super::tune::copy_queue() && INFLIGHT.latest(&r).is_none();
        if on_copy {
            self.xcopies.push((src, buf, region));
            XCOPY[0].fetch_add(1, Ordering::Relaxed);
            XCOPY[1].fetch_add(n, Ordering::Relaxed);
        } else {
            let pre = self.open_pre();
            if super::tune::pre_merge() {
                self.pre_copies.push((src, buf, region));
            } else {
                PRE_CALLS.fetch_add(1, Ordering::Relaxed);
                unsafe { ctx().device.cmd_copy_buffer(pre, src, buf, &[region]) };
            }
        }
        let was_shared = shared.is_some();
        match shared {
            Some((chunk, _, id)) => {
                self.slots[self.cur].keep.push(chunk);
                self.shared_pending.push((base, size, id, on_copy));
                PreClock::set(&mut clk, 4);
            }
            None => {
                self.pre_used = off + n;
                PreClock::set(&mut clk, 3);
            }
        }
        self.pres.insert((base, size), (buf, off));
        // DS_TRACE_PRE=<hex address>[-<hex end>]: the first 80 batch-start
        // copies over it (which commands read a range copied again and again).
        static AT: std::sync::OnceLock<(u64, u64)> = std::sync::OnceLock::new();
        let (at, at_end) = *AT.get_or_init(|| {
            let v = std::env::var("DS_TRACE_PRE").unwrap_or_default();
            let num = |x: &str| u64::from_str_radix(x.trim().trim_start_matches("0x"), 16).ok();
            match v.split_once('-') {
                Some((lo, hi)) => (num(lo).unwrap_or(0), num(hi).unwrap_or(0)),
                None => num(&v).map_or((0, 0), |x| (x, x + 1)),
            }
        });
        if at != 0 && base < at_end && at < base + size {
            static N: AtomicU64 = AtomicU64::new(0);
            if N.fetch_add(1, Ordering::Relaxed) < 80 {
                crate::info!("gpu[{}]: batch-start copy {base:#x}+{size:#x} ({}) for shaders {:#x} {:#x} (flip {}, batch command {})", self.name, if was_shared { "shared" } else { "this batch's" }, self.sync_readers[0], self.sync_readers[1], super::FRAMES.load(Ordering::Relaxed), self.batch_shaders.len());
            }
        }
        PRE_COPIES.fetch_add(1, Ordering::Relaxed);
        PRE_BYTES.fetch_add(n, Ordering::Relaxed);
        self.ops[OP_PRE] += 1;
        self.ops[OP_PRE_BYTES] += n;
        drop(clk);
        if profiling() {
            note_pre_repeat(base, size, n, why);
        }
        Some((buf, off, bias))
    }

    /// Returns a command's binding for reuse, cleared.
    fn recycle(&mut self, mut b: Bound) {
        if self.bound_pool.len() >= 8 {
            return;
        }
        b.bufs.clear();
        b.imgs.clear();
        b.img_arrays.clear();
        b.writes.clear();
        b.reads.clear();
        b.gds = false;
        b.sized_writes.clear();
        b.images.clear();
        b.stored.clear();
        b.stencil_views.clear();
        b.shadows.clear();
        b.write_src.clear();
        b.placements.clear();
        b.in_place.clear();
        b.sgpr_base = None;
        b.ubo = None;
        b.window = None;
        self.bound_pool.push(b);
    }

    /// Fills guest [base, base+len) with the word `v` in place (vkCmdFillBuffer
    /// on the guest range's buffer), ordered like a dispatch (lazy barriers).
    /// False: a video-memory home covers part of it or it is not one mapping
    /// (the kernel runs instead).
    fn fill_in_place(&mut self, base: u64, len: u64, v: u32) -> Result<bool, String> {
        if len == 0 || base % 4 != 0 || len % 4 != 0 || DEVICE_LOST.load(Ordering::Relaxed) || super::homes::overlaps(base, len) {
            return Ok(false);
        }
        if super::census::on() {
            super::census::note(base, len, super::census::FILL);
        }
        if super::smalltrack::on() {
            super::smalltrack::cp_write(base, len, true);
        }
        let Some((buffer, start, _)) = span(base, len) else { return Ok(false) };
        self.keep_cp_write(base, len);
        // Resident shadows holding newer data of the range go back first:
        // their merge would undo the fill (and so would this batch's W writes').
        self.flush_resident(Some(&[(base, len)]))?;
        self.rgm_before_in_place(&[(base, len)])?;
        self.begin();
        if !self.recording {
            return Ok(false);
        }
        if super::vk::labels_on() {
            let l = format!("{} fill {len:#x}", self.name);
            self.label(&l);
        }
        self.end_rendering();
        let cmd = self.cmd();
        if !lazy_barriers() || self.barrier_wanted {
            self.barrier();
            self.barrier_wanted = false;
            self.dispatched = false;
        }
        unsafe { ctx().device.cmd_fill_buffer(cmd, buffer, base - start, len, v) };
        // Front end 2: the mirror too (its later write-back of dwords marked
        // before the fill copies the filled values).
        if super::fe2::on() && super::fe2::in_span(base, len) {
            if let (Some((m, _)), Ok(())) = (super::resident::mirror_buffer(), super::resident::back(base & !4095, (base + len).next_multiple_of(4096) - (base & !4095))) {
                unsafe { ctx().device.cmd_fill_buffer(cmd, m, base - super::resident::MIRROR_BASE, len, v) };
            }
        }
        if profiling() {
            let mut g = FILL_BY.lock();
            let e = g.get_or_insert_with(HashMap::new).entry((base, len)).or_insert([0, 0]);
            e[0] += 1;
            e[1] = v as u64;
        }
        self.ops[OP_FILLS] += 1;
        self.ops[OP_FILL_BYTES] += len;
        if !lazy_barriers() {
            self.barrier();
        }
        Ok(true)
    }

    /// Performs the fast clears pending for `keys` before they are used: the
    /// title clears DCC and HTILE surfaces by writing only their metadata
    /// (e.g. zero-filling a HUD layer's DCC each frame); images here are
    /// never compressed, so the metadata is read back and the image cleared.
    /// Runs before a draw or dispatch reserves its slot (it flushes).
    pub(crate) fn resolve_fast_clears(&mut self, keys: &[super::images::ImageKey]) {
        if self.held && !keys.is_empty() {
            // The current frame's clears are not the frame-ahead recorder's
            // to take; its own are, once their values are known (else they
            // wait for its own unsubmitted work).
            if pending_for(keys) && super::images::with(|im| im.clears_pending(keys)) {
                self.stop("fast clear (the current frame's)");
                return;
            }
            let mine: Vec<(super::images::ImageKey, u64, Option<u32>)> = keys.iter().filter_map(|k| self.held_clears.get(k).map(|&(m, v)| (*k, m, v))).collect();
            if mine.is_empty() {
                return;
            }
            if mine.iter().any(|x| x.2.is_none()) {
                self.stop("fast clear (its own, value unknown)");
                return;
            }
            for x in &mine {
                self.held_clears.remove(&x.0);
            }
            let clears = super::images::with(|im| mine.iter().filter_map(|&(k, m, v)| im.fast_clear_of(k, m, v)).collect::<Vec<_>>());
            self.end_rendering();
            self.record_fast_clears(clears, false);
            return;
        }
        if keys.is_empty() || !pending_for(keys) || !super::images::with(|im| im.clears_pending(keys)) {
            return;
        }
        self.end_rendering();
        // The metadata write may still be in flight: read it back after the
        // GPU work, unless the fill that wrote it gave its value (then the
        // clear needs no wait; ~8 per courtyard frame drained the GPU).
        // DS_GPU_VERIFY=1: known values are compared with the read-back one.
        let check = verify::enabled();
        if check || !super::images::with(|im| im.clears_known(keys)) {
            self.flush_for("fast clear");
            if self.stopped {
                return;
            }
        }
        let clears = super::images::with(|im| im.take_clears(keys));
        self.record_fast_clears(clears, check);
    }

    /// Records `clears` (`resolve_fast_clears`): each image cleared to the
    /// value its metadata gives (known, else read now); `check`: known
    /// values compared with the metadata (`DS_GPU_VERIFY`).
    fn record_fast_clears(&mut self, clears: Vec<super::images::FastClear>, check: bool) {
        for c in clears {
            let v = match c.known {
                Some(v) => {
                    FAST_CLEARS_KNOWN.fetch_add(1, Ordering::Relaxed);
                    if check && super::cp::guest_readable(c.meta, 4) {
                        let read = unsafe { std::ptr::read_volatile(c.meta as *const u32) };
                        static MISMATCH: AtomicU64 = AtomicU64::new(0);
                        if read != v && MISMATCH.fetch_add(1, Ordering::Relaxed) < 20 {
                            crate::warn!("images: fast clear of {:#x}: metadata {:#x} reads {read:#x}, the fill wrote {v:#x}", c.key.base, c.meta);
                        }
                    }
                    v
                }
                None => {
                    if !super::cp::guest_readable(c.meta, 4) {
                        continue;
                    }
                    unsafe { std::ptr::read_volatile(c.meta as *const u32) }
                }
            };
            let k = c.key;
            let full = vk::ImageSubresourceRange { aspect_mask: super::images::aspects(k.format), base_mip_level: 0, level_count: vk::REMAINING_MIP_LEVELS, base_array_layer: 0, layer_count: vk::REMAINING_ARRAY_LAYERS };
            let what = if super::images::is_depth(k.format) {
                // HTILE: 0 = depth and stencil cleared; all ones = expanded.
                if v == u32::MAX {
                    continue;
                }
                if v != 0 {
                    self.report(format!("images: HTILE value {v:#x} of {:#x} not interpreted (no clear)", k.base));
                    continue;
                }
                let value = vk::ClearDepthStencilValue { depth: f32::from_bits(c.clear[0]), stencil: c.clear[1] & 0xFF };
                self.begin();
                if !self.recording {
                    return;
                }
                unsafe { ctx().device.cmd_clear_depth_stencil_image(self.cmd(), c.image, vk::ImageLayout::GENERAL, &value, &[full]) };
                format!("depth {} stencil {}", value.depth, value.stencil)
            } else {
                match dcc_clear_color(v & 0xFF, k.format, c.clear) {
                    Ok(Some(color)) => {
                        self.begin();
                        if !self.recording {
                            return;
                        }
                        unsafe { ctx().device.cmd_clear_color_image(self.cmd(), c.image, vk::ImageLayout::GENERAL, &color, &[full]) };
                        format!("DCC code {:#x}", v & 0xFF)
                    }
                    Ok(None) => continue,
                    Err(e) => {
                        self.report(format!("images: fast clear of {:#x}: {e}", k.base));
                        continue;
                    }
                }
            };
            self.barrier();
            FAST_CLEARS.fetch_add(1, Ordering::Relaxed);
            self.ops[OP_FAST_CLEARS] += 1;
            static SEEN: Mutex<Option<HashSet<u64>>> = Mutex::new(None);
            if SEEN.lock().get_or_insert_with(HashSet::new).insert(k.base) {
                crate::info!("images: fast clear of {:#x} {}x{} {:?} ({what}; metadata {:#x})", k.base, k.width, k.height, k.format, c.meta);
            }
        }
    }

    /// Grows depth images that the textures of `stages` sample with more
    /// layers than they have (shadow arrays rendered slice by slice; see
    /// `Images::depth_to_grow`), after the work that used them.
    pub(crate) fn grow_sampled_depth(&mut self, stages: &[&Prepared]) {
        for p in stages {
            for r in p.resources.iter() {
                let Resolved::Image(t) = r else { continue };
                let Some(k) = super::images::Images::key_of(t) else { continue };
                let fam = super::images::Images::depth_family(k.format);
                if fam.is_empty() || k.layers <= 1 || k.ty3d {
                    continue;
                }
                let Some(old) = super::images::with(|im| im.depth_to_grow(k.base, k.width, k.height, fam, k.layers)) else { continue };
                self.end_rendering();
                self.flush_for("depth growth");
                if self.stopped {
                    return;
                }
                if let Err(e) = super::images::with(|im| im.grow_depth(old, k.layers)) {
                    self.report(format!("images: growing depth image {:#x}: {e}", k.base));
                }
            }
        }
    }

    /// Reloads images among `keys` whose guest bytes GPU buffer writes or CP
    /// DMA covered since their last use (`Images::note_overwrites`), once the
    /// writers have completed. Runs before a draw or dispatch reserves its
    /// slot (it flushes).
    pub(crate) fn resolve_overwrites(&mut self, keys: &[super::images::ImageKey]) {
        if self.held && keys.iter().any(|k| self.held_over.contains(k)) {
            self.stop("overwritten image (its own)");
            return;
        }
        if keys.is_empty() || !pending_for(keys) || !super::images::with(|im| im.overwrites_pending(keys)) {
            return;
        }
        self.end_rendering();
        // Writers that bound a video-memory home left their data there: it
        // is written back before the flush the reload waits for (guest memory
        // is what the reload reads).
        if super::homes::any() && super::tune::image_homes() {
            let spans = super::images::with(|im| im.overwritten_spans(keys));
            if super::homes::any_stale(&spans) {
                super::homes::IMAGE_STALE[1].fetch_add(1, Ordering::Relaxed);
                self.home_write_back(&spans);
                if self.stopped {
                    return;
                }
            }
        }
        self.flush_for("overwritten image");
        if self.stopped {
            return;
        }
        super::images::with(|im| im.reload_overwritten(keys));
    }

    /// Copies blocks that GPU work stored through an uncompressed view into
    /// the block-compressed textures among `keys` (Images::take_block_copies),
    /// before the work that samples them.
    pub(crate) fn sync_block_textures(&mut self, keys: &[super::images::ImageKey]) {
        if keys.is_empty() {
            return;
        }
        if self.deferred {
            // The copies must precede every use in program order.
            if super::images::with(|im| !im.block_copies(keys).is_empty()) {
                self.stop("block copy");
            }
            return;
        }
        let copies = super::images::with(|im| im.take_block_copies(keys));
        if copies.is_empty() {
            return;
        }
        self.begin();
        self.end_rendering_then_barrier();
        let c = ctx();
        let cmd = self.cmd();
        for b in &copies {
            if b.levels > 0 {
                let regions: Vec<vk::ImageCopy> = (0..b.levels)
                    .map(|l| {
                        let sub = vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: l, base_array_layer: 0, layer_count: b.layers };
                        // Extent in texels of the (uncompressed) source.
                        vk::ImageCopy { src_subresource: sub, src_offset: vk::Offset3D::default(), dst_subresource: sub, dst_offset: vk::Offset3D::default(), extent: vk::Extent3D { width: (b.width >> l).max(1), height: (b.height >> l).max(1), depth: 1 } }
                    })
                    .collect();
                unsafe { c.device.cmd_copy_image(cmd, b.src, vk::ImageLayout::GENERAL, b.dst, vk::ImageLayout::GENERAL, &regions) };
            }
            // The mip tail from the view of the tail block: each tail mip
            // from where it sits inside that block.
            if let Some((tsrc, regs)) = &b.tail {
                let regions: Vec<vk::ImageCopy> = regs
                    .iter()
                    .map(|&(mip, x, y, w, h)| {
                        let src_sub = vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 };
                        let dst_sub = vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: mip, base_array_layer: 0, layer_count: 1 };
                        vk::ImageCopy { src_subresource: src_sub, src_offset: vk::Offset3D { x: x as i32, y: y as i32, z: 0 }, dst_subresource: dst_sub, dst_offset: vk::Offset3D::default(), extent: vk::Extent3D { width: w, height: h, depth: 1 } }
                    })
                    .collect();
                unsafe { c.device.cmd_copy_image(cmd, *tsrc, vk::ImageLayout::GENERAL, b.dst, vk::ImageLayout::GENERAL, &regions) };
                BLOCK_TAIL_COPIES.fetch_add(1, Ordering::Relaxed);
            }
            BLOCK_COPIES.fetch_add(1, Ordering::Relaxed);
            static SEEN: Mutex<Option<HashSet<u64>>> = Mutex::new(None);
            if SEEN.lock().get_or_insert_with(HashSet::new).insert(b.key.base) {
                crate::info!(
                    "images: {:#x} {}x{} {:?}: {} mip(s) of blocks copied in from its GPU-written {}x{} view{}",
                    b.key.base,
                    b.key.width,
                    b.key.height,
                    b.key.format,
                    b.levels,
                    b.width,
                    b.height,
                    b.tail.as_ref().map_or(String::new(), |(_, r)| format!(", mips {}..{} from its tail block's view", r.first().map_or(0, |x| x.0), r.last().map_or(0, |x| x.0)))
                );
            }
        }
        self.barrier();
    }

    /// Copies depth planes that GPU work stored through a color view into the
    /// depth attachments among `keys` (Images::take_depth_copies), before
    /// the draws that test against them: through a device-local buffer (a
    /// depth aspect cannot be the destination of an image copy from a color
    /// format), in bands of rows when the plane is larger than the buffer.
    pub(crate) fn sync_depth_views(&mut self, keys: &[super::images::ImageKey]) {
        if keys.is_empty() || !super::tune::depth_views() || !super::images::depth_copies_possible(keys) {
            return;
        }
        if self.deferred {
            // The copies must precede every use in program order.
            if super::images::with(|im| !im.depth_copies(keys).is_empty()) {
                self.stop("depth copy");
            }
            return;
        }
        let copies = super::images::with(|im| im.take_depth_copies(keys));
        if copies.is_empty() {
            return;
        }
        let Some(buf) = depth_copy_buffer() else {
            static WARNED: std::sync::Once = std::sync::Once::new();
            WARNED.call_once(|| crate::warn!("images: no buffer for depth-plane copies (depth surfaces written by compute keep stale depth)"));
            return;
        };
        self.begin();
        self.end_rendering_then_barrier();
        let c = ctx();
        let cmd = self.cmd();
        let transfer = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::TRANSFER_READ)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE);
        let mut first = true;
        for d in &copies {
            let row = d.width as u64 * d.texel as u64;
            let band = ((DEPTH_COPY_BYTES / row.max(1)) as u32).clamp(1, d.height.max(1));
            for layer in 0..d.layers {
                let mut y = 0;
                while y < d.height {
                    let h = band.min(d.height - y);
                    if !first {
                        // The buffer's previous band is read out before it is overwritten.
                        unsafe { c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[transfer], &[], &[]) };
                    }
                    first = false;
                    let region = |aspect| vk::BufferImageCopy {
                        buffer_offset: 0,
                        buffer_row_length: 0,
                        buffer_image_height: 0,
                        image_subresource: vk::ImageSubresourceLayers { aspect_mask: aspect, mip_level: 0, base_array_layer: layer, layer_count: 1 },
                        image_offset: vk::Offset3D { x: 0, y: y as i32, z: 0 },
                        image_extent: vk::Extent3D { width: d.width, height: h, depth: 1 },
                    };
                    unsafe {
                        c.device.cmd_copy_image_to_buffer(cmd, d.src, vk::ImageLayout::GENERAL, buf, &[region(vk::ImageAspectFlags::COLOR)]);
                        c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[transfer], &[], &[]);
                        c.device.cmd_copy_buffer_to_image(cmd, buf, d.dst, vk::ImageLayout::GENERAL, &[region(vk::ImageAspectFlags::DEPTH)]);
                    }
                    y += h;
                }
            }
            DEPTH_COPIES.fetch_add(1, Ordering::Relaxed);
            static SEEN: Mutex<Option<HashSet<u64>>> = Mutex::new(None);
            if SEEN.lock().get_or_insert_with(HashSet::new).insert(d.key.base) {
                crate::info!(
                    "images: {:#x} {}x{} {:?}: depth plane copied in from its GPU-written {}-byte color view (logged once; each write of that view)",
                    d.key.base,
                    d.width,
                    d.height,
                    d.key.format,
                    d.texel
                );
            }
        }
        self.barrier();
    }

    /// Submits recorded work and waits for all of this queue's GPU work and
    /// deferred effects.
    #[track_caller]
    pub fn flush(&mut self) {
        if self.deferred {
            let why = if self.flush_why.is_empty() { "flush" } else { self.flush_why };
            self.stop(why);
            return;
        }
        if super::frameprobe::on() && self.name == "gpu-gfx" {
            super::frameprobe::flushed(self.flush_why);
        }
        if super::runprobe::on() && self.name == "gpu-gfx" {
            super::runprobe::stop("flush");
        }
        let t0 = profiling().then(std::time::Instant::now);
        self.submit_for("sync");
        let tw = super::timeline::wait_start();
        let t = std::time::Instant::now();
        wait_done(self.last_seq);
        tw.end(&self.name, || format!("wait: own batches (#{})", self.last_seq));
        super::hitch::note_since(super::hitch::Kind::GpuWait, t, || format!("own batches for a CPU-side read ({})", self.flush_why));
        self.inflight.clear();
        if let Some(t0) = t0 {
            let l = std::panic::Location::caller();
            let mut g = FLUSH_SITES.lock();
            let e = g.get_or_insert_with(HashMap::new).entry((l.file(), l.line(), self.name == "gpu-gfx")).or_insert([0; 2]);
            e[0] += 1;
            e[1] += t0.elapsed().as_nanos() as u64;
        }
    }

    /// Completes this queue's recorded and in-flight work when any of it
    /// writes into `reads` (memory the CP reads now: indirect arguments,
    /// draw counts); returns whether it waited.
    pub fn flush_if_writes(&mut self, reads: &[(u64, u64)]) -> bool {
        self.sync_for(reads, SYNC_CP)
    }

    /// Whether GPU work is recorded or not yet completed.
    pub fn busy(&self) -> bool {
        self.recording || self.last_seq > done_seq()
    }

    /// GetLodStats with statistics measured by the GPU (`tune lod_stats 3`):
    /// after the work recorded so far, copies the counters into readback
    /// slot `slot` and, with `reset`, starts them again (no bank sampled).
    /// The caller reads the slot (`lod_stats_words`) in an action after it.
    pub fn lod_stats_copy(&mut self, slot: usize, reset: bool) {
        self.begin();
        self.end_rendering_then_barrier();
        let fx = fixed();
        let cmd = self.cmd();
        let d = &ctx().device;
        let bytes = super::lodstats::SLOT_BYTES;
        unsafe {
            d.cmd_copy_buffer(cmd, fx.lodstats, fx.lod_readback, &[vk::BufferCopy { src_offset: 0, dst_offset: slot as u64 * bytes, size: bytes }]);
            if reset {
                let m = super::lodstats::MIN_BYTES;
                d.cmd_fill_buffer(cmd, fx.lodstats, 0, m, 15);
                d.cmd_fill_buffer(cmd, fx.lodstats, m, bytes - m, 0);
            }
            let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::HOST_READ);
            d.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::HOST, vk::DependencyFlags::empty(), &[mb], &[], &[]);
        }
        self.barrier();
    }

    /// Whether deferred CP effects are still pending (later CP memory writes
    /// must queue behind them).
    pub fn deferred_pending(&self) -> bool {
        !self.actions.is_empty() || self.last_action_seq > done_seq() || self.gpu_writes_open || self.last_gpu_write_seq > done_seq()
    }

    /// Performs `f` after this queue's GPU work recorded so far and its
    /// earlier deferred effects, or now when nothing is outstanding. `write`
    /// is the guest range it writes; `ordered`: later GPU work binding that
    /// range is submitted after it (CP writes that following dispatches may
    /// read; `CP_WRITES`). Returns the batch
    /// sequence it runs after (None: performed now).
    pub fn after_gpu(&mut self, f: Action, write: Option<(u64, u64)>, ordered: bool) -> Option<u64> {
        if self.held {
            // Never before the current frame's work (`Cp::effect` stops
            // the frame-ahead recorder first).
            drop(f);
            self.stop("CP effect");
            return None;
        }
        if let (Some(w), true) = (write, self.name == "gpu-gfx") {
            super::frameahead::a_wrote(&[w]);
        }
        if super::runprobe::on() && self.name == "gpu-gfx" {
            super::runprobe::stop("CP effect");
        }
        if self.deferred {
            crate::error!("gpu[{}]: a CP effect in a parallel run (performed now)", self.name);
            self.stop("CP effect");
            f();
            return None;
        }
        if self.recording {
            self.actions.push(f);
            self.submit_for("effect");
            if let Some(w) = write {
                journal_note_one(w, &self.name);
                INFLIGHT.note(self.last_seq, w);
                if ordered {
                    CP_WRITES.note(self.last_seq, w);
                }
                self.inflight.insert(self.last_seq, w.0, w.1);
                self.own_gen += 1;
            }
            Some(self.last_seq)
        } else if self.last_seq > done_seq() {
            let seq = push_batch(vk::Fence::null(), vk::QueryPool::null(), vk::QueryPool::null(), vec![f], Vec::new(), Vec::new(), &self.name, Vec::new(), [0; OPS], Vec::new(), None, false, |_| None);
            self.last_seq = seq;
            self.last_action_seq = seq;
            self.cpu_action_seq = seq;
            if let Some(w) = write {
                journal_note_one(w, &self.name);
                INFLIGHT.note(seq, w);
                if ordered {
                    CP_WRITES.note(seq, w);
                }
                self.inflight.insert(seq, w.0, w.1);
                self.own_gen += 1;
            }
            Some(seq)
        } else {
            if let Some(w) = write {
                journal_note_one(w, &self.name);
            }
            f();
            None
        }
    }

    /// Submits the batch being recorded when the GPU has run out of work
    /// (every batch of every queue completed) and the batch has been
    /// recording for `tune idle_split` us: the main command processor's own
    /// chunk of a parallel run (~670-820 commands) took 7-12 ms to record
    /// while the GPU sat idle (timeline run 20261012-032934).
    pub(super) fn split_if_gpu_idle(&mut self) {
        if !self.recording || self.deferred {
            return;
        }
        let us = super::tune::idle_split();
        if us == 0 || done_seq() + 1 < NEXT_SEQ.load(Ordering::Relaxed) || self.begun_at.elapsed() < std::time::Duration::from_micros(us) {
            return;
        }
        self.submit_for("idle");
    }

    /// The title synchronized its GPU work (CS/PS partial flush, cache
    /// action, end-of-pipe event): the next dispatch or draw waits for
    /// earlier work.
    pub fn note_sync(&mut self) {
        self.barrier_wanted = true;
    }

    /// A full barrier when the title asked for one or (draws) after
    /// dispatches since the last.
    pub(super) fn barrier_if_wanted(&mut self, for_draw: bool) {
        if !self.recording {
            return;
        }
        if self.barrier_wanted || (for_draw && self.dispatched) {
            // Diagnosis: 16 leaves out the title's barriers, 32 the ones
            // between dispatches and the draws after them.
            let skip = super::tune::skip_barriers();
            if (self.barrier_wanted && skip & 16 == 0) || (!self.barrier_wanted && skip & 32 == 0) {
                self.barrier();
            }
            self.barrier_wanted = false;
            self.dispatched = false;
        }
    }

    /// Orders everything this queue does from now on after batch `seq` (a
    /// label wait passed on a value that batch will write): later batches
    /// follow it on the device queue, CP writes are deferred behind it.
    /// `by_cpu`: the CPU writes that value after the batch completed.
    pub fn depend_on(&mut self, seq: u64, by_cpu: bool) {
        self.last_seq = self.last_seq.max(seq);
        self.last_action_seq = self.last_action_seq.max(seq);
        if by_cpu {
            self.cpu_dep_seq = self.cpu_dep_seq.max(seq);
        }
    }

    /// Before the CP reads guest memory (shader inputs, uploads, indirect
    /// arguments): completes this queue's work writing into `reads` and
    /// waits for any queue's unfinished batch that does. Returns whether
    /// it waited (memory may have changed).
    /// Waits for GPU writes into the guest memory of textures a stage samples
    /// (they may be uploaded from it now). A coarse bound is checked first;
    /// only textures it finds written are looked at closely: images the GPU
    /// owns (render targets sampled by later passes) are read from their
    /// Vulkan image, and the others count with the exact span of their mips
    /// and layers (coarse bounds and render targets overlapped buffers GPU
    /// work was writing, e.g. the ranges of unbounded global-pointer
    /// bindings: ~2 needless waits of 5-7 ms per frame in the courtyard).
    pub fn sync_textures(&mut self, textures: &[((u64, u64), super::desc::Tsharp)], why: usize) -> bool {
        if self.held {
            // Textures are taken from guest memory as it is now.
            self.held_reads.extend(textures.iter().map(|x| x.0));
            self.note_kinds(1, textures.len());
        }
        if textures.is_empty() || super::tune::skip_steps() & 1 != 0 {
            return false;
        }
        if super::tune::tex_waits() == 0 {
            let coarse: Vec<(u64, u64)> = textures.iter().map(|x| x.0).collect();
            return self.sync_for(&coarse, why);
        }
        let done = done_seq();
        self.inflight.prune(done);
        let written = |s: &Self, r: (u64, u64)| s.pending.overlaps(r.0, r.1) || s.inflight.latest(&[r], done).is_some() || INFLIGHT.latest(&[r]).is_some();
        let hit: Vec<&super::desc::Tsharp> = textures.iter().filter(|(r, _)| written(self, *r)).map(|(_, t)| t).collect();
        if hit.is_empty() {
            return false;
        }
        let fine: Vec<(u64, u64)> = super::images::with(|im| hit.into_iter().filter(|t| im.reads_guest(t)).map(|t| (t.base, super::images::guest_size(t).unwrap_or((t.width as u64 * t.height as u64 * t.depth.max(1) as u64 * 16).min(1 << 30)))).collect());
        self.sync_for(&fine, why)
    }

    /// `sync_for_reads`, counted by the reason (`SYNC_WHY`) when it waited.
    /// `sync_for` of a stage's reads (`tune skip_read_waits`: none, a timing
    /// diagnosis with wrong results).
    pub(super) fn sync_for_stage(&mut self, reads: &[(u64, u64)], why: usize) -> bool {
        if super::tune::skip_read_waits() {
            return false;
        }
        self.sync_for(reads, why)
    }

    pub fn sync_for(&mut self, reads: &[(u64, u64)], why: usize) -> bool {
        if super::runprobe::on() && self.name == "gpu-gfx" {
            super::runprobe::reads(reads);
        }
        if super::xwait::on() {
            super::xwait::note_reads(&self.name, reads, why);
        }
        if super::smalltrack::on() {
            super::smalltrack::cpu_read(reads, why, &self.name);
        }
        // DS_TRACE_SYNC=1: the first 400 waits, with the read that collided
        // and the batch writing it; DS_TRACE_SYNC=<text>: only on queues
        // whose name contains it (e.g. acb0x50).
        static TRACE: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
        static TRACED: AtomicU64 = AtomicU64::new(0);
        let traced_here = TRACE
            .get_or_init(|| std::env::var("DS_TRACE_SYNC").ok().filter(|v| !v.is_empty() && v != "0"))
            .as_deref()
            .is_some_and(|v| v == "1" || self.name.contains(v));
        // `tune trace_sync <n>`: the next n waits (one is taken only when
        // the read waited: forwarded copies spare many that would).
        let live = super::tune::get_trace_sync() && self.would_wait(reads);
        let trace = (traced_here && TRACED.load(Ordering::Relaxed) < 400) || live;
        let culprit = if trace {
            let done = done_seq();
            // What forwarded copies leave to wait for (as `sync_for_reads`
            // sees it).
            let left = super::forward::uncovered(reads);
            left.as_deref().unwrap_or(reads).iter().find_map(|&r| {
                let who = if self.pending.overlaps(r.0, r.1) {
                    Some("this batch".to_string())
                } else if let Some(s) = self.inflight.latest(&[r], done) {
                    Some(format!("own batch {s}"))
                } else {
                    INFLIGHT.latest(&[r]).map(|s| format!("batch {s} (another queue)"))
                }?;
                let cp = match CP_WRITES.latest(&[r]) {
                    Some(s) => format!(", a deferred CP write after batch {s}"),
                    None => ", GPU work".to_string(),
                };
                let last = super::cp::last_write_info(r.0).map(|w| format!(" [last CP write there: {w}]")).unwrap_or_default();
                let fwd = format!(" [{}]", super::forward::describe(r.0, r.1));
                Some((r, format!("{who}{cp}{last}{fwd}")))
            })
        } else {
            None
        };
        let waited = self.sync_for_reads(reads);
        let w = waited.is_some();
        if let Some(t0) = waited {
            WAIT_NS[0].fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
            WAIT_NS[1].fetch_add(1, Ordering::Relaxed);
            note_queue_wait(&self.name, 0, t0.elapsed());
            SYNC_WHY[why.min(SYNC_WHY.len() - 1)].fetch_add(1, Ordering::Relaxed);
            if let Some(((a, n), src)) = culprit.filter(|_| !live || super::tune::take_trace_sync()) {
                self.sync_culprit = Some((a, n));
                {
                    TRACED.fetch_add(1, Ordering::Relaxed);
                    const WHY: [&str; 6] = ["CP read", "dispatch shader read", "dispatch texture", "draw shader read", "draw texture", "draw indices"];
                    crate::info!(
                        "gpu[{}]: waited {} us for {} {a:#x}+{n:#x} written by {src} (reading shaders {:#x} {:#x}; last recorded {:#x})",
                        self.name,
                        t0.elapsed().as_micros(),
                        WHY[why.min(5)],
                        self.sync_readers[0],
                        self.sync_readers[1],
                        self.batch_shaders.last().copied().unwrap_or(0)
                    );
                }
            }
        }
        w
    }

    /// Instead of waiting for GPU work that writes what the stages'
    /// preparations read (`tune spec_reads`), when that work is all
    /// submitted (not this batch's) and no home holds newer data: the
    /// data-only loads it writes (`Prepared::data_loads`) are copied from
    /// guest memory into the stages' SGPR data by the batch prologue, on the
    /// GPU after that work (`SgprPatch`); the other words it writes
    /// (descriptors, pointers) are used as read and checked once the work
    /// completed (`check_speculations`; a shader that mispredicted waits
    /// from then on). Some: no wait, with the copies to make once the
    /// stages' SGPR data is placed; None: wait.
    pub(super) fn speculate(&mut self, stages: &[&Prepared]) -> Option<Vec<SgprPatch>> {
        if !super::tune::spec_reads() {
            return None;
        }
        if stages.iter().any(|p| super::homes::any_stale(&p.reads) || super::keep::any_dirty(&p.reads)) {
            SPEC_REFUSED[0].fetch_add(1, Ordering::Relaxed);
            return None;
        }
        let done = done_seq();
        self.inflight.prune(done);
        let writer = |s: &Self, r: &[(u64, u64)]| s.inflight.latest(r, done).max(INFLIGHT.latest(r)).filter(|&x| x > done);
        let mut patches = Vec::new();
        let mut checks: Vec<SpecCheck> = Vec::new();
        let mut own: Vec<SpecCheck> = Vec::new();
        let mut any = false;
        for (k, p) in stages.iter().enumerate() {
            if p.reads.is_empty() {
                continue;
            }
            let pending = self.pending.overlaps_any(&p.reads);
            if !pending && writer(self, &p.reads).is_none() {
                continue;
            }
            // A parallel chunk's batch gets its sequence later (its main
            // command processor submits it): such reads wait there.
            if pending && self.deferred {
                SPEC_REFUSED[2].fetch_add(1, Ordering::Relaxed);
                return None;
            }
            any = true;
            let mut data = vec![false; p.words.len()];
            for d in p.data_loads.iter() {
                let bytes = 4 * d.dwords as u64;
                if self.pending.overlaps(d.addr, bytes) {
                    // Data this batch writes: no copy point before the stage
                    // outside rendering (`tune gpu_loads` reads it on the GPU).
                    SPEC_REFUSED[1].fetch_add(1, Ordering::Relaxed);
                    return None;
                }
                if writer(self, &[(d.addr, bytes)]).is_some() {
                    let Some((src, start, _)) = span(d.addr, bytes) else {
                        SPEC_REFUSED[1].fetch_add(1, Ordering::Relaxed);
                        return None;
                    };
                    patches.push(SgprPatch { stage: k, sgpr: d.sgpr, src, src_off: d.addr - start, bytes });
                }
                for w in d.word as usize..(d.word + d.dwords) as usize {
                    if let Some(x) = data.get_mut(w) {
                        *x = true;
                    }
                }
            }
            // The other words, where that work writes: used as read, checked
            // once it completed; not for a stage that writes memory or images
            // (a wrong descriptor there would corrupt them).
            let (mut reads, mut words): (Vec<(u64, u64)>, Vec<u32>) = (Vec::new(), Vec::new());
            let mut i = 0usize;
            for &(a, n) in p.reads.iter() {
                for j in 0..n / 4 {
                    if !data.get(i).copied().unwrap_or(true) {
                        let at = a + 4 * j;
                        match reads.last_mut() {
                            Some(r) if r.0 + r.1 == at => r.1 += 4,
                            _ => reads.push((at, 4)),
                        }
                        words.push(p.words[i]);
                    }
                    i += 1;
                }
            }
            let own_writes = self.pending.overlaps_any(&reads);
            let submitted = writer(self, &reads);
            if own_writes || submitted.is_some() {
                // Data-only patches alone (`tune spec_reads 2`): descriptor
                // words GPU work writes are waited for as before.
                if super::tune::spec_reads_mode() == 2 {
                    SPEC_REFUSED[5].fetch_add(1, Ordering::Relaxed);
                    return None;
                }
                if p.tr.plan.resources.iter().any(|r| r.written) && !read_only_descriptors(p, &reads) {
                    SPEC_REFUSED[3].fetch_add(1, Ordering::Relaxed);
                    return None;
                }
                if no_spec(p.addr) {
                    SPEC_REFUSED[4].fetch_add(1, Ordering::Relaxed);
                    return None;
                }
                match submitted.filter(|_| !own_writes) {
                    Some(s) => checks.push(SpecCheck { seq: s, shader: p.addr, reads, words }),
                    // This batch writes them: checked once it completed (its
                    // sequence comes with its submission).
                    None => own.push(SpecCheck { seq: 0, shader: p.addr, reads, words }),
                }
            }
        }
        if !any {
            return None;
        }
        SPEC_STATS[0].fetch_add((checks.len() + own.len()) as u64, Ordering::Relaxed);
        SPEC_STATS[3].fetch_add(1, Ordering::Relaxed);
        SPEC_STATS[4].fetch_add(patches.len() as u64, Ordering::Relaxed);
        if !own.is_empty() {
            SPEC_OWN.fetch_add(own.len() as u64, Ordering::Relaxed);
            self.own_checks.extend(own);
        }
        if !checks.is_empty() {
            let mut g = SPEC_CHECKS.lock();
            g.extend(checks);
            SPEC_PENDING.store(g.len() as u64, Ordering::Release);
        }
        Some(patches)
    }

    /// Records the prologue copies of `patches` for stage `stage`, whose SGPR
    /// data is at `ring` + `at` (`speculate`).
    pub(super) fn patch_sgpr(&mut self, patches: &[SgprPatch], stage: usize, ring: vk::Buffer, at: u64) {
        let mine: Vec<&SgprPatch> = patches.iter().filter(|p| p.stage == stage).collect();
        if mine.is_empty() {
            return;
        }
        let pre = self.open_pre();
        for p in mine {
            unsafe { ctx().device.cmd_copy_buffer(pre, p.src, ring, &[vk::BufferCopy { src_offset: p.src_off, dst_offset: at + 4 * p.sgpr as u64, size: p.bytes }]) };
        }
    }

    /// The batch prologue (`Slot::pre`), begun if it was not yet: it runs
    /// after all earlier submitted work and before this batch's.
    fn open_pre(&mut self) -> vk::CommandBuffer {
        let c = ctx();
        let (pre, bq) = (self.slots[self.cur].pre, self.slots[self.cur].bq);
        if !self.pre_open {
            unsafe {
                c.device.begin_command_buffer(pre, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).ok();
                if bq != vk::QueryPool::null() {
                    c.device.cmd_reset_query_pool(pre, bq, BQ_PRE, 2);
                }
            }
            // The whole prologue is one label range (`DS_GPU_LABELS`).
            if super::vk::labels_on() {
                super::vk::label_begin(pre, &format!("{} prologue", self.name));
            }
            // After all earlier submitted work (`tune skip_barriers` 128:
            // left out, timing only).
            if super::tune::skip_barriers() & 128 == 0 {
                global_barrier(pre);
            }
            if bq != vk::QueryPool::null() {
                unsafe { c.device.cmd_write_timestamp(pre, vk::PipelineStageFlags::BOTTOM_OF_PIPE, bq, BQ_PRE) };
            }
            self.pre_open = true;
        }
        pre
    }

    /// Whether `sync_for_reads` would wait.
    fn would_wait(&mut self, reads: &[(u64, u64)]) -> bool {
        self.overlaps_pending(reads) || INFLIGHT.latest(reads).is_some_and(|s| s > done_seq())
    }

    /// Waits until GPU work writing `reads` completed (homes copied back,
    /// this queue's batch submitted, other queues' batches done): when it
    /// waited, since when.
    pub fn sync_for_reads(&mut self, reads: &[(u64, u64)]) -> Option<std::time::Instant> {
        // Words a copy recorded earlier leaves there were read as forwarded
        // (`gpu::forward`): the copy need not complete first.
        let forwarded = super::forward::uncovered(reads);
        let reads: &[(u64, u64)] = forwarded.as_deref().unwrap_or(reads);
        if reads.is_empty() {
            return None;
        }
        if let Some(m) = &self.main_writes {
            if !m.is_empty() && m.overlaps_any(reads) {
                if MAIN_WRITE_READS.fetch_add(1, Ordering::Relaxed) % 20000 == 0 {
                    let hits: Vec<String> = reads.iter().flat_map(|&(a, n)| m.intersect(a, n).into_iter().take(3).map(move |(b, k)| format!("read {a:#x}+{n:#x} / written {b:#x}+{k:#x}"))).take(4).collect();
                    crate::info!("gpu[{}]: a worker reads the main command processor's unsubmitted writes ({} ranges): {} (shaders {:#x} {:#x})", self.name, m.len(), hits.join("; "), self.sync_readers[0], self.sync_readers[1]);
                }
                if super::tune::par_main_writes() == 1 {
                    self.stop("reads the main command processor's unsubmitted writes");
                    return Some(std::time::Instant::now());
                }
                // Mode 3: the main command processor submits that batch right
                // after handing the run over; once it has, its writes are
                // waited for like any submitted work (below).
                if let Some(f) = self.main_submitted.clone() {
                    if !f.load(Ordering::Acquire) {
                        let t = std::time::Instant::now();
                        let mut k = 0u32;
                        while !f.load(Ordering::Acquire) {
                            k += 1;
                            if k < 4096 {
                                std::hint::spin_loop();
                            } else {
                                std::thread::yield_now();
                            }
                        }
                        MAIN_SUBMIT_WAITS[0].fetch_add(1, Ordering::Relaxed);
                        MAIN_SUBMIT_WAITS[1].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    }
                    self.main_writes = None;
                    self.main_submitted = None;
                }
            }
        }
        let mut waited: Option<std::time::Instant> = None;
        let prof = profiling();
        let t0 = prof.then(std::time::Instant::now);
        if prof {
            SYNC_STEPS[4].fetch_add(1, Ordering::Relaxed);
            SYNC_STEPS[5].fetch_add(reads.len() as u64, Ordering::Relaxed);
        }
        let lap = |k: usize, t: &mut Option<std::time::Instant>| {
            if let Some(t0) = t {
                let now = std::time::Instant::now();
                SYNC_STEPS[k].fetch_add((now - *t0).as_nanos() as u64, Ordering::Relaxed);
                *t = Some(now);
            }
        };
        let mut t = t0;
        // Reads in pages known clean since the write generations last changed
        // skip the checks; the others are checked as before.
        let mut dirty = std::mem::take(&mut self.dirty_reads);
        dirty.clear();
        let reads: &[(u64, u64)] = if super::tune::clean_pages() {
            let tag = [self.own_gen, INFLIGHT.notes.load(Ordering::Acquire), super::homes::STALE_ADDED.load(Ordering::Acquire), super::keep::DIRTY_GEN.load(Ordering::Acquire)];
            if tag != self.clean_tag {
                self.clean_pages.fill(u64::MAX);
                self.clean_tag = tag;
            }
            for &(a, n) in reads {
                if n == 0 {
                    continue;
                }
                let (first, last) = (a >> CLEAN_SHIFT, (a + n - 1) >> CLEAN_SHIFT);
                let mut clean = true;
                for page in first..=last {
                    let slot = (page as usize) & (CLEAN_SLOTS - 1);
                    if self.clean_pages[slot] == page {
                        continue;
                    }
                    let whole = [(page << CLEAN_SHIFT, 1u64 << CLEAN_SHIFT)];
                    if super::homes::any_stale(&whole) || super::keep::any_dirty(&whole) || self.overlaps_pending(&whole) || INFLIGHT.latest(&whole).is_some() {
                        clean = false;
                        break;
                    }
                    self.clean_pages[slot] = page;
                }
                if !clean {
                    dirty.push((a, n));
                }
            }
            if prof {
                SYNC_STEPS[6].fetch_add((reads.len() - dirty.len()) as u64, Ordering::Relaxed);
            }
            &dirty
        } else {
            reads
        };
        if reads.is_empty() {
            lap(0, &mut t);
            self.dirty_reads = dirty;
            return None;
        }
        if super::homes::any_stale(reads) {
            // GPU work wrote these parts in video memory: copied back (after
            // all earlier submitted work) before the CPU reads them.
            waited.get_or_insert_with(std::time::Instant::now);
            self.home_write_back(reads);
            self.flush();
        }
        // So do kept windows GPU work wrote (`gpu::keep`).
        match self.keep_reads(reads) {
            Ok(true) => {
                waited.get_or_insert_with(std::time::Instant::now);
                self.flush();
            }
            Ok(false) => {}
            Err(e) => self.report(format!("kept window write-back: {e}")),
        }
        lap(0, &mut t);
        if self.overlaps_pending(reads) {
            waited.get_or_insert_with(std::time::Instant::now);
            if self.past_grow && !(self.pending.overlaps_any(reads) || self.closed_overlap(reads) || self.main_pending(reads)) {
                // Held past a label wait, and only batches of its own
                // submitted before write them: waited for as another queue's
                // would be.
                let done = done_seq();
                if let Some(s) = self.inflight.latest(reads, done) {
                    wait_done(s);
                }
            } else {
                if self.past_grow {
                    // (Diagnostics: by reader and range, `held_own_line`.)
                    let hit = reads.iter().find_map(|&r| {
                        if self.pending.overlaps_any(&[r]) {
                            Some((r, 0u8))
                        } else if self.closed_overlap(&[r]) {
                            Some((r, 1))
                        } else if self.main_pending(&[r]) {
                            Some((r, 2))
                        } else {
                            None
                        }
                    });
                    if let Some(((a, n), by)) = hit {
                        *HELD_OWN.lock().get_or_insert_with(HashMap::new).entry((self.sync_readers[0], self.sync_readers[1], a, n, by)).or_default() += 1;
                    }
                }
                if self.past_grow && HELD_OWN_LOGGED.fetch_add(1, Ordering::Relaxed) < 12 {
                    // Diagnostics: what a held executor read of its own held work.
                    let hits: Vec<String> = reads
                        .iter()
                        .filter(|&&r| self.pending.overlaps_any(&[r]) || self.closed_overlap(&[r]))
                        .take(4)
                        .map(|&(a, n)| {
                            let cp = self.cp_written.iter().any(|&(b, m)| b < a + n && a < b + m);
                            format!("{a:#x}+{n:#x}{}", if cp { " (a CP write's)" } else { "" })
                        })
                        .collect();
                    crate::info!("gpu[{}]: held work reads its own held writes: {} (stages {:#x} {:#x})", self.name, hits.join(", "), self.sync_readers[0], self.sync_readers[1]);
                }
                self.flush_for("reads own writes");
            }
        }
        lap(1, &mut t);
        if let Some(s) = INFLIGHT.latest(reads) {
            if s > done_seq() {
                waited.get_or_insert_with(std::time::Instant::now);
                FOREIGN_WAITS.fetch_add(1, Ordering::Relaxed);
                let tw = super::timeline::wait_start();
                let t = std::time::Instant::now();
                wait_done(s);
                tw.end(&self.name, || format!("wait: batch #{s} writing inputs"));
                super::hitch::note_since(super::hitch::Kind::GpuWait, t, || format!("another queue's batch {s} writing inputs"));
            }
        }
        lap(2, &mut t);
        self.dirty_reads = dirty;
        waited
    }

    /// Whether this queue's outstanding GPU work or deferred CP effects will
    /// write [a, a+n) (a label the CP waits for comes from the GPU itself).
    pub fn will_write(&mut self, a: u64, n: u64) -> bool {
        self.overlaps_pending(&[(a, n)])
    }

    /// A guest range the batch being recorded writes.
    #[track_caller]
    pub(super) fn note_write(&mut self, a: u64, n: u64) {
        // DS_TRACE_NOTE=<hex address>[-<hex end>]: the first 400 notes
        // covering it or overlapping the range (diagnosis: who writes a word
        // stage preparations wait for, or a region; `DS_TRACE_NOTE_FROM`).
        static AT: std::sync::OnceLock<(u64, u64)> = std::sync::OnceLock::new();
        let (at, at_end) = *AT.get_or_init(|| {
            let v = std::env::var("DS_TRACE_NOTE").unwrap_or_default();
            let num = |x: &str| u64::from_str_radix(x.trim().trim_start_matches("0x"), 16).ok();
            match v.split_once('-') {
                Some((lo, hi)) => (num(lo).unwrap_or(0), num(hi).unwrap_or(0)),
                None => num(&v).map_or((0, 0), |x| (x, x + 1)),
            }
        });
        if at != 0 && a < at_end && at < a.saturating_add(n) {
            // `DS_TRACE_NOTE_FROM=<flip>`: from that flip on.
            static FROM: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
            let from = *FROM.get_or_init(|| std::env::var("DS_TRACE_NOTE_FROM").ok().and_then(|v| v.parse().ok()).unwrap_or(0));
            static N: AtomicU64 = AtomicU64::new(0);
            if super::FRAMES.load(Ordering::Relaxed) >= from && N.fetch_add(1, Ordering::Relaxed) < 400 {
                let l = std::panic::Location::caller();
                crate::info!("gpu[{}]: note_write {a:#x}+{n:#x} overlaps {at:#x}..{at_end:#x} at {}:{} (writer {:#x}, last shader {:#x}, {} commands in the batch, flip {})", self.name, l.file(), l.line(), self.noting, self.batch_shaders.last().copied().unwrap_or(0), self.batch_shaders.len(), super::FRAMES.load(Ordering::Relaxed));
            }
        }
        super::precache::invalidate_gpu(a, n);
        if !self.rgm_written.contains(a, n) {
            // (By the caller's site: `note_write` is the common path.)
            if let Some(r) = super::rgm::get() {
                let d = r.invalidate(a, n);
                if d > 0 {
                    super::rgm::note_site_at(std::panic::Location::caller(), 1, d);
                    if d >= 64 {
                        super::rgm::note_big_writer(self.noting, &self.name, a, n, d);
                    }
                    // Large writes by other paths (unbounded pointers) into
                    // resident data recur: those granules take the old paths.
                    if d >= 16 {
                        r.demote(a, n);
                    }
                }
            }
        }
        super::gmirror::gpu_write(a, n, self.batch_shaders.last().copied().unwrap_or(0), &self.name);
        super::forward::written(a, n);
        if super::tune::race_checks() {
            self.note_open_write(a, n);
        }
        self.pending.insert(a, n);
        if let Some(r) = self.rec_writes.as_mut() {
            r.insert(a, n);
        }
        if self.long && !self.noting_planned {
            self.long_nonfwd.insert(a, n);
        }
        self.note_since(a, n);
        self.own_gen += 1;
        if verify::enabled() {
            self.pending_log.push((a, n));
        }
    }

    /// Starts noting what its batches write (`rec_writes`), with `first`.
    pub fn rec_begin(&mut self, first: &super::ranges::Ranges) {
        self.rec_writes = Some(first.clone());
    }

    /// What its batches wrote since `rec_begin` (None: not noting).
    pub fn rec_take(&mut self) -> Option<super::ranges::Ranges> {
        self.rec_writes.take()
    }

    /// A worker records a chunk of a long parallel run (`tune long_runs`):
    /// held as the frame-ahead recorder is (its label writes promised when
    /// its batches are submitted, its CPU reads, used images and image-cache
    /// notes kept for the check at its turn: `Cp::long_check`).
    pub fn long_begin(&mut self, planned: std::sync::Arc<HashMap<u64, std::sync::Arc<[u32]>>>) {
        self.long_planned = planned;
        self.long_nonfwd.clear();
        self.long_mismatch = false;
        self.held = true;
        self.long = true;
        self.held_reset();
        self.held_promises.clear();
    }

    /// The chunk passed its check: its image-cache notes go to the cache
    /// before its batches are submitted (later chunks check against them).
    pub fn long_commit(&mut self) {
        if !self.held_clears.is_empty() || !self.held_over.is_empty() {
            let (c, o) = (std::mem::take(&mut self.held_clears), std::mem::take(&mut self.held_over));
            super::images::with(|im| im.apply_held(&c, &o));
        }
        self.long_end();
    }

    /// The chunk ends without its notes going anywhere (dropped, cancelled
    /// or failed: its batches are discarded by the caller).
    pub fn long_end(&mut self) {
        self.held = false;
        self.long = false;
        self.stopped = false;
        self.stop_why = "";
        self.held_clears.clear();
        self.held_over.clear();
        self.held_promises.clear();
        self.held_reads.clear();
        self.held_kinds.clear();
        self.held_keys.clear();
        self.cp_written.clear();
        self.held_fwd_reads.clear();
        self.held_fwd_planned.clear();
        self.long_planned = Default::default();
        self.noting_planned = false;
    }

    /// What its long chunk wrote otherwise than by the planner's copies, and
    /// whether it computed one of those otherwise than planned.
    pub fn long_take(&mut self) -> (super::ranges::Ranges, bool) {
        (std::mem::take(&mut self.long_nonfwd), std::mem::take(&mut self.long_mismatch))
    }

    /// Whether the dispatch at packet `at` is a copy its long run's planner
    /// forwarded.
    pub fn long_planned_at(&self, at: u64) -> bool {
        self.long && self.long_planned.contains_key(&at)
    }

    /// Whether GPU work or a deferred CP write not done yet writes `r`.
    pub fn writes_pending(&self, r: &[(u64, u64)]) -> bool {
        self.overlaps_pending_ro(r) || INFLIGHT.latest(r).is_some() || CP_WRITES.latest(r).is_some()
    }

    /// Whether a long run's chunk recorded with it must be recorded again:
    /// it read on the CPU what the run's earlier part writes (`earlier`;
    /// reads served by forwarded copies: what its earlier chunks write,
    /// `chunks`), or it used an image that has a fast clear or an overwrite
    /// pending (noted by an earlier chunk, or by the work before the run).
    pub fn long_conflict(&self, earlier: &super::ranges::Ranges, chunks: &super::ranges::Ranges, before: &super::ranges::Ranges) -> Option<&'static str> {
        if chunks.overlaps_any(&self.held_fwd_reads) || chunks.overlaps_any(&self.held_fwd_planned) || before.overlaps_any(&self.held_fwd_reads) {
            return Some("read the run's earlier writes (forwarded copy)");
        }
        if let Some(i) = self.held_reads.iter().position(|r| earlier.overlaps(r.0, r.1)) {
            return Some(match self.held_kinds.get(i).copied() {
                Some(0) => "read the run's earlier writes (shared copy)",
                Some(1) => "read the run's earlier writes (texture)",
                Some(2) => "read the run's earlier writes (binding)",
                Some(3) => "read the run's earlier writes (CP label poll)",
                Some(4) => "read the run's earlier writes (CP read)",
                Some(5) => "read the run's earlier writes (register pairs)",
                _ => "read the run's earlier writes",
            });
        }
        if !self.held_keys.is_empty() && super::images::any_pending() {
            let keys: Vec<super::images::ImageKey> = self.held_keys.iter().copied().collect();
            if super::images::with(|im| im.clears_pending(&keys) || im.overwrites_pending(&keys)) {
                return Some("images with pending clears or reloads");
            }
        }
        None
    }

    /// The first CPU read that `earlier` writes (`DS_TRACE_LONG_DROPS`).
    pub fn long_conflict_read(&self, earlier: &super::ranges::Ranges) -> Option<(u64, u64)> {
        self.held_reads.iter().copied().find(|r| earlier.overlaps(r.0, r.1))
    }

    /// The first read served by a forwarded copy that `chunks` writes
    /// (`DS_TRACE_LONG_DROPS`).
    pub fn long_conflict_fwd_read(&self, chunks: &super::ranges::Ranges, before: &super::ranges::Ranges) -> Option<(u64, u64)> {
        self.held_fwd_reads.iter().copied().find(|r| chunks.overlaps(r.0, r.1) || before.overlaps(r.0, r.1)).or_else(|| self.held_fwd_planned.iter().copied().find(|r| chunks.overlaps(r.0, r.1)))
    }

    /// `overlaps_pending` without pruning (a shared borrow).
    fn overlaps_pending_ro(&self, reads: &[(u64, u64)]) -> bool {
        self.pending.overlaps_any(reads) || self.inflight.latest(reads, done_seq()).is_some() || self.main_pending(reads) || self.closed_overlap(reads)
    }

    fn overlaps_pending(&mut self, reads: &[(u64, u64)]) -> bool {
        let done = done_seq();
        self.inflight.prune(done);
        self.pending.overlaps_any(reads) || self.inflight.latest(reads, done).is_some() || self.main_pending(reads) || self.closed_overlap(reads)
    }

    /// Whether a closed batch not yet submitted (a deferred executor's)
    /// writes into `reads`.
    fn closed_overlap(&self, reads: &[(u64, u64)]) -> bool {
        !self.closed.is_empty() && self.closed.iter().any(|b| b.writes.overlaps_any(reads))
    }

    /// A worker's: whether `reads` overlap what the main command processor's
    /// batch writes while it is not submitted yet (`tune par_main_writes 3`;
    /// once it is, those writes are in-flight work like any other).
    fn main_pending(&self, reads: &[(u64, u64)]) -> bool {
        self.main_submitted.as_ref().is_some_and(|f| !f.load(Ordering::Acquire)) && self.main_writes.as_ref().is_some_and(|m| m.overlaps_any(reads))
    }

    /// A CP memory write that follows earlier GPU work, performed by the GPU
    /// in order with it (vkCmdUpdateBuffer between barriers) instead of by
    /// the CPU once the batch completed: the batch need not end there, and
    /// later work reading the value need not wait for that completion (the
    /// title issues ~170 such label and data writes per frame on the graphics
    /// queue). `label`: a value other queues may wait for (promised when the
    /// batch is submitted). False: not possible here.
    pub fn gpu_cp_write(&mut self, a: u64, data: &[u32], label: Option<(u64, u8)>, end_of_pipe: bool) -> bool {
        let bytes = 4 * data.len() as u64;
        if bytes == 0 || bytes > 65536 || a % 4 != 0 || DEVICE_LOST.load(Ordering::Relaxed) || super::homes::overlaps(a, bytes) {
            return false;
        }
        if super::census::on() {
            super::census::note(a, bytes, super::census::CP_WRITE);
        }
        if super::smalltrack::on() {
            super::smalltrack::cp_write(a, bytes, false);
        }
        // Mirror mode: the target pages become host class first (imported
        // exactly, written in place: `resident::host_for_cp`).
        super::resident::host_for_cp(a, bytes);
        let Some((buffer, start, _)) = span(a, bytes) else { return false };
        self.keep_cp_write(a, bytes);
        if self.recording && self.rgm_before_in_place(&[(a, bytes)]).is_err() {
            return false;
        }
        // CPU-side writes deferred into the recording batch run after it
        // completes: this one would pass them. Earlier batches' run before
        // the batch holding this one is submitted.
        if !self.actions.is_empty() {
            self.submit_for("effect order");
        }
        let cpu = self.cpu_action_seq.max(self.cpu_dep_seq);
        if cpu > done_seq() {
            self.submit_after = self.submit_after.max(cpu);
        }
        self.begin();
        if !self.recording {
            return false;
        }
        // What earlier work of this batch wrote into resident shadows reaches
        // guest memory before this write does: a title thread that sees the
        // label reads it there (the shadows were merged only when the batch
        // was submitted, after the label: `tune label_merge`).
        if !self.resident.is_empty() {
            if super::tune::label_merge() {
                LABEL_MERGES[0].fetch_add(1, Ordering::Relaxed);
                if self.flush_resident(None).is_err() {
                    return false;
                }
            } else {
                LABEL_MERGES[1].fetch_add(1, Ordering::Relaxed);
            }
        }
        self.gpu_writes_open = true;
        if super::vk::labels_on() {
            let l = format!("{} cp write", self.name);
            self.label(&l);
        }
        let full = !(end_of_pipe && super::tune::eop_light()) && super::tune::skip_barriers() & 4 == 0;
        if full && self.ts_deferred.is_empty() {
            self.end_rendering_before_barrier();
        } else {
            self.end_rendering();
        }
        // End-of-pipe events are written in order on the console: a title
        // that sees this value may read the timestamps taken before it. (A
        // label a CP of ours waits on: at the batch's close, `tune
        // ts_cpw_waited`.)
        let waited = super::cp::cp_waited(a);
        if !(waited && super::tune::ts_cpw_waited()) {
            self.resolve_timestamps();
        }
        // Front end 2: what earlier work wrote into the mirror reaches guest
        // memory before this write does (a title thread that sees it reads
        // that there), and the write goes into the mirror too (later work
        // reads it there).
        let mut after_scatter = false;
        if self.fe2_marked {
            // (A label our command processors wait on: GPU-to-GPU
            // synchronization, `tune fe2_cpw_waited`.)
            if waited {
                SCATTER_SITES[3].fetch_add(1, Ordering::Relaxed);
            }
            if super::tune::fe2_cpw_close() || (waited && super::tune::fe2_cpw_waited()) {
                SCATTER_SITES[4].fetch_add(1, Ordering::Relaxed);
            } else {
                SCATTER_SITES[0].fetch_add(1, Ordering::Relaxed);
                after_scatter = self.fe2_scatter();
            }
        }
        let fe2_mirror = if super::fe2::on() && super::fe2::in_span(a, bytes) && super::resident::back(a & !4095, (a + bytes).next_multiple_of(4096) - (a & !4095)).is_ok() {
            super::fe2::note_cp_mirror();
            super::resident::mirror_buffer().map(|(m, _)| (m, a - super::resident::MIRROR_BASE))
        } else {
            None
        };
        let cmd = self.cmd();
        let raw: Vec<u8> = data.iter().flat_map(|w| w.to_le_bytes()).collect();
        let to_mirror = |cmd: vk::CommandBuffer| {
            if let Some((m, off)) = fe2_mirror {
                unsafe { ctx().device.cmd_update_buffer(cmd, m, off, &raw) };
            }
        };
        self.taint_resident(a, bytes);
        if end_of_pipe && super::tune::eop_light() {
            self.ops[OP_CP_WRITES] += 1;
            // An end-of-pipe write (RELEASE_MEM): after the work before it,
            // which it does not hold later work back for (as on the console);
            // later transfers wait for it. A wait on it in this batch orders
            // the work after it (`order_after_open_writes`).
            let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::MEMORY_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
            unsafe {
                // (Right after front end 2's write-back none: its last
                // barrier holds all later work, its first all earlier work.)
                if super::tune::skip_barriers() & 256 == 0 && !(after_scatter && super::tune::fe2_cpw_light()) {
                    ctx().device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[mb], &[], &[]);
                } else if after_scatter {
                    EOP_LIGHT[3].fetch_add(1, Ordering::Relaxed);
                }
                ctx().device.cmd_update_buffer(cmd, buffer, a - start, &raw);
                to_mirror(cmd);
                // Diagnosis (`tune skip_copies` 1024, timing only): written
                // twice, with a barrier between (what one more costs).
                if super::tune::skip_copies() & 1024 != 0 {
                    ctx().device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[mb], &[], &[]);
                    ctx().device.cmd_update_buffer(cmd, buffer, a - start, &raw);
                }
            }
            self.eop_unordered = true;
            EOP_LIGHT[0].fetch_add(1, Ordering::Relaxed);
        } else if super::tune::skip_barriers() & 4 != 0 {
            self.ops[OP_CP_WRITES] += 1;
            unsafe { ctx().device.cmd_update_buffer(cmd, buffer, a - start, &raw) };
            to_mirror(cmd);
        } else {
            // (Right after front end 2's write-back none: see above.)
            if after_scatter && super::tune::fe2_cpw_light() && !seg_stamps() {
                EOP_LIGHT[3].fetch_add(1, Ordering::Relaxed);
            } else {
                self.barrier();
            }
            // Counted between its barriers (`seg_stamps` names the segment
            // by it).
            self.ops[OP_CP_WRITES] += 1;
            unsafe { ctx().device.cmd_update_buffer(cmd, buffer, a - start, &raw) };
            to_mirror(cmd);
            self.barrier();
            self.barrier_wanted = false;
            self.dispatched = false;
        }
        self.note_write(a, bytes);
        if self.held && self.cp_written.len() < 4096 {
            self.cp_written.push((a, bytes));
        }
        if full {
            // The barrier after it orders it before later work.
            self.ordered_since(a, bytes);
        }
        if let Some((v, n)) = label {
            self.pending_promises.push((a, v, n));
            // (The frame-ahead recorder's come after the current frame's
            // work: nothing it holds is submitted on request. Past a run, a
            // request ends the stretch: `Cp::run` settles it and submits.)
            if !self.held || self.past {
                UNSUBMITTED_LABELS.lock().get_or_insert_with(HashMap::new).insert(a & !7, self.submit_request.clone());
                // Another queue already blocks on it: submit at the next packet.
                if super::tune::label_early() && super::cp::waited_now(a) {
                    self.submit_request.store(true, Ordering::SeqCst);
                    LABEL_EARLY.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
        GPU_CP_WRITES.fetch_add(1, Ordering::Relaxed);
        true
    }

    /// The per-view copy kernel recorded as a buffer update of the words
    /// `forward::compute` found for it (`tune copy_update`): at the
    /// dispatch's place in the stream behind the same lazy barrier, later
    /// draws and dispatches ordered after it as after a dispatch, only the
    /// words it writes noted as written, and its forwarding entry started
    /// (checked against memory once the batch completed). None: recorded.
    /// Some: left to the dispatch (the computation given back) when the
    /// destination lies in a video-memory home, overlaps a resident shadow
    /// or the copy's inputs, or is not mapped for the GPU.
    /// `check_only`: a parallel worker's (its words checked after its batch
    /// completed, not forwarded).
    fn copy_as_update(&mut self, p: &Prepared, c: (u64, Vec<u32>, super::forward::Inputs), check_only: bool) -> Option<(u64, Vec<u32>, super::forward::Inputs)> {
        let (base, bytes) = (c.0, 4 * c.1.len() as u64);
        let refuse = |c| {
            super::forward::UPDATES[2].fetch_add(1, Ordering::Relaxed);
            Some(c)
        };
        if bytes == 0 || base % 4 != 0 || DEVICE_LOST.load(Ordering::Relaxed) || super::homes::overlaps(base, bytes) || c.2.overlaps(base, bytes) {
            return refuse(c);
        }
        if self.resident.iter().any(|r| r.guest < base + bytes && base < r.guest + r.range) {
            return refuse(c);
        }
        let Some((buffer, start, end)) = span(base, bytes) else { return refuse(c) };
        if base + bytes > end {
            return refuse(c);
        }
        // Front end 2: the words go into the mirror too (later work reads
        // them there); without its memory the copy stays a dispatch.
        let fe2_mirror = if super::fe2::on() && super::fe2::in_span(base, bytes) {
            let lo = base & !4095;
            match (super::resident::mirror_buffer(), super::resident::back(lo, (base + bytes).next_multiple_of(4096) - lo)) {
                (Some((m, _)), Ok(())) => Some(m),
                _ => return refuse(c),
            }
        } else {
            None
        };
        self.begin();
        if !self.recording {
            return refuse(c);
        }
        // A CP write still pending into the destination is performed before
        // this batch is submitted (as for a binding of it in place).
        if let Some(s) = CP_WRITES.latest(&[(base, bytes)]) {
            self.submit_after = self.submit_after.max(s);
        }
        self.end_rendering_lazy();
        if lazy_barriers() && self.barrier_wanted {
            if super::tune::skip_barriers() & 16 == 0 {
                self.barrier();
            }
            self.barrier_wanted = false;
            self.dispatched = false;
        }
        let (base, words, from) = c;
        let raw: &[u8] = unsafe { std::slice::from_raw_parts(words.as_ptr() as *const u8, bytes as usize) };
        let cmd = self.cmd();
        for (k, part) in raw.chunks(65536).enumerate() {
            unsafe { ctx().device.cmd_update_buffer(cmd, buffer, base - start + 65536 * k as u64, part) };
        }
        if let Some(m) = fe2_mirror {
            for (k, part) in raw.chunks(65536).enumerate() {
                unsafe { ctx().device.cmd_update_buffer(cmd, m, base - super::resident::MIRROR_BASE + 65536 * k as u64, part) };
            }
        }
        if lazy_barriers() {
            // Later draws and dispatches wait for it as for a dispatch.
            self.dispatched = true;
        } else {
            self.barrier();
        }
        let w = [(base, bytes)];
        if super::images::may_note(&w) {
            super::images::with(|im| im.note_writes(&w, &w));
        }
        super::mirror::invalidate(&w);
        super::track::gpu_write(base, bytes);
        self.note_write(base, bytes);
        // The next dispatch notes its writes again (these changed the record).
        self.last_notes.2 = [u64::MAX; 4];
        let id = if check_only {
            super::forward::add_check(base, words, "a worker's buffer update", &self.name)
        } else {
            super::forward::UPDATES[0].fetch_add(1, Ordering::Relaxed);
            super::forward::UPDATES[1].fetch_add(words.len() as u64, Ordering::Relaxed);
            super::forward::add(base, words, p.addr, p.code_hash, from, "as a buffer update", &self.name)
        };
        self.forward_ids.push(id);
        EXECUTED.fetch_add(1, Ordering::Relaxed);
        self.batch_shaders.push(p.addr);
        None
    }

    /// An end-of-pipe timestamp write (RELEASE_MEM DATA_SEL 3/4) performed by
    /// the GPU in order with this queue's work: the GPU clock when the work
    /// before it completed (vkCmdWriteTimestamp at the bottom of the pipe),
    /// stored in the title's clock by `rdna::kernels::timestamp_store`. The
    /// CPU took the time when the completion thread reached the batch, which
    /// ended every batch at such a write (~400 per courtyard frame). False:
    /// the CPU path (no calibration, memory the GPU cannot write here).
    pub fn gpu_timestamp(&mut self, a: u64) -> bool {
        if (self.deferred && !self.held) || a % 4 != 0 || DEVICE_LOST.load(Ordering::Relaxed) || super::homes::overlaps(a, 8) {
            return false;
        }
        let Some(k) = title_clock_k() else { return false };
        let Ok(tp) = ts_pipe() else { return false };
        let Some((buffer, start, _)) = span(a, 8) else { return false };
        if !self.actions.is_empty() {
            self.submit_for("effect order");
        }
        let cpu = self.cpu_action_seq.max(self.cpu_dep_seq);
        if cpu > done_seq() {
            self.submit_after = self.submit_after.max(cpu);
        }
        self.begin();
        // A descriptor set with two buffers per write (see `reserve`).
        if self.ts_used >= TS_PER_BATCH || self.pool_use[0] + 1 > POOL_LIMITS[0] || self.pool_use[1] + 2 > POOL_LIMITS[1] {
            self.submit_for("timestamps");
            self.begin();
        }
        if !self.recording {
            return false;
        }
        let c = ctx();
        if self.slots[self.cur].ts.is_none() {
            let pool = unsafe { c.device.create_query_pool(&vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(TS_PER_BATCH), None) };
            let buf = c.device_buffer(TS_PER_BATCH as u64 * 8, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST);
            match (pool, buf) {
                (Ok(p), Ok((b, _mem))) => self.slots[self.cur].ts = Some((p, b)),
                _ => return false,
            }
        }
        let Some((pool, tbuf)) = self.slots[self.cur].ts else { return false };
        // The deferred ones get their sets when resolved (`resolve_timestamps`;
        // the space stays reserved here).
        let eop = super::tune::eop_light();
        let set = if eop {
            vk::DescriptorSet::null()
        } else {
            let Ok(set) = self.alloc_set(tp.set_layout) else { return false };
            set
        };
        self.pool_use[0] += 1;
        self.pool_use[1] += 2;
        self.gpu_writes_open = true;
        // (A light one inside a rendering instance once the batch's pool is
        // reset: a timestamp write may come there, a query reset may not.)
        if !super::tune::eop_light() && super::tune::ts_nowait() && super::tune::skip_barriers() & 8 == 0 {
            self.end_rendering_before_barrier();
        } else if !(eop && self.ts_pool_reset) {
            self.end_rendering();
        }
        let q = self.ts_used;
        self.ts_used += 1;
        let align = c.limits.min_storage_buffer_offset_alignment.max(4);
        let off = a - start;
        let base = off / align * align;
        let cmd = self.cmd();
        if eop {
            // Taken when the work before it completed, without holding later
            // work back; read and stored to guest memory at the batch's end
            // (`resolve_timestamps`; the CPU reads it after completion, and a
            // command-processor read of it waits for the batch). The batch's
            // first resets every query of the pool, so the later ones need
            // not end the rendering instance they come in (~400 a frame at
            // the courtyard, each a restart of rendering for the next draw).
            unsafe {
                if !self.ts_pool_reset {
                    c.device.cmd_reset_query_pool(cmd, pool, 0, TS_PER_BATCH);
                    self.ts_pool_reset = true;
                }
                c.device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, pool, q);
            }
            self.ts_deferred.push(TsEntry { q, buffer, off, k });
            EOP_LIGHT[1].fetch_add(1, Ordering::Relaxed);
            self.taint_resident(a, 8);
            self.note_write(a, 8);
            self.ops[OP_TIMESTAMPS] += 1;
            if GPU_TIMESTAMPS.fetch_add(1, Ordering::Relaxed) < 8 {
                self.ts_check.push(a);
            }
            return true;
        }
        let infos = [vk::DescriptorBufferInfo { buffer: tbuf, offset: 0, range: TS_PER_BATCH as u64 * 8 }, vk::DescriptorBufferInfo { buffer, offset: base, range: off - base + 8 }];
        let writes: Vec<vk::WriteDescriptorSet> = infos
            .iter()
            .enumerate()
            .map(|(i, info)| vk::WriteDescriptorSet::default().dst_set(set).dst_binding(i as u32).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(std::slice::from_ref(info)))
            .collect();
        let pc: Vec<u8> = [q, ((off - base) / 4) as u32, k as u32, (k >> 32) as u32].iter().flat_map(|x| x.to_le_bytes()).collect();
        unsafe {
            c.device.cmd_reset_query_pool(cmd, pool, q, 1);
            c.device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, pool, q);
            copy_timestamps(cmd, pool, q, 1, tbuf, q as u64 * 8);
            let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ);
            c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &[mb], &[], &[]);
            c.device.update_descriptor_sets(&writes, &[]);
            c.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, tp.pipeline);
            self.dyncache.cs_pipeline = tp.pipeline;
            c.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, tp.layout, 0, &[set], &[]);
            c.device.cmd_push_constants(cmd, tp.layout, vk::ShaderStageFlags::COMPUTE, 0, &pc);
            c.device.cmd_dispatch(cmd, 1, 1, 1);
        }
        // Counted before its barrier (`seg_stamps` names the segment by it).
        self.ops[OP_TIMESTAMPS] += 1;
        let ordered = super::tune::skip_barriers() & 8 == 0;
        if ordered {
            self.barrier();
        }
        self.barrier_wanted = false;
        self.dispatched = false;
        self.taint_resident(a, 8);
        self.note_write(a, 8);
        if ordered {
            self.ordered_since(a, 8);
        }
        if GPU_TIMESTAMPS.fetch_add(1, Ordering::Relaxed) < 8 {
            self.ts_check.push(a);
        }
        true
    }

    /// The batch's deferred end-of-pipe timestamps (`tune eop_light`): their
    /// queries read at once and each stored to guest memory, at the batch's
    /// end or before a later CP write of the batch (each query holds when the
    /// work before it completed).
    fn resolve_timestamps(&mut self) {
        if self.ts_deferred.is_empty() {
            return;
        }
        if super::vk::labels_on() {
            let l = format!("{} timestamp stores", self.name);
            self.label(&l);
        }
        let mut deferred = std::mem::take(&mut self.ts_deferred);
        // A slot written twice in the batch keeps the later value (stored in
        // order, one dispatch each, the later store won; stores of one
        // dispatch run unordered).
        let before = deferred.len();
        let mut seen: Vec<(vk::Buffer, u64)> = Vec::with_capacity(before);
        for i in (0..deferred.len()).rev() {
            let key = (deferred[i].buffer, deferred[i].off);
            if seen.contains(&key) {
                deferred.remove(i);
            } else {
                seen.push(key);
            }
        }
        TS_REPEATS.fetch_add((before - deferred.len()) as u64, Ordering::Relaxed);
        let (Some((pool, tbuf)), Ok(tp)) = (self.slots[self.cur].ts, ts_many_pipe()) else { return };
        let c = ctx();
        let cmd = self.cmd();
        unsafe {
            copy_timestamps(cmd, pool, 0, self.ts_used, tbuf, 0);
            let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ);
            c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &[mb], &[], &[]);
            c.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, tp.pipeline);
        }
        self.dyncache.cs_pipeline = tp.pipeline;
        // Up to TS_MANY per dispatch (`tune ts_batch`), those of one buffer
        // and clock offset in order, the binding over the bytes they store.
        let per = (super::tune::ts_batch() as usize).clamp(1, rdna::kernels::TS_MANY as usize);
        let align = c.limits.min_storage_buffer_offset_alignment.max(4);
        let mut i = 0;
        while i < deferred.len() {
            let first = deferred[i];
            let mut j = i + 1;
            while j < deferred.len() && j - i < per && deferred[j].buffer == first.buffer && deferred[j].k == first.k {
                j += 1;
            }
            let group = &deferred[i..j];
            let lo = group.iter().map(|e| e.off).min().unwrap_or(0) / align * align;
            let hi = group.iter().map(|e| e.off + 8).max().unwrap_or(8);
            let Ok(set) = self.alloc_set(tp.set_layout) else { return };
            // Diagnosis (`tune skip_copies` 512, timing only): stored into
            // video memory instead of guest memory.
            let dst = match scratch_buffer() {
                Some(sb) if super::tune::skip_copies() & 512 != 0 => (sb, 0),
                _ => (first.buffer, lo),
            };
            let infos = [vk::DescriptorBufferInfo { buffer: tbuf, offset: 0, range: TS_PER_BATCH as u64 * 8 }, vk::DescriptorBufferInfo { buffer: dst.0, offset: dst.1, range: hi - lo }];
            let writes: Vec<vk::WriteDescriptorSet> = infos
                .iter()
                .enumerate()
                .map(|(b, info)| vk::WriteDescriptorSet::default().dst_set(set).dst_binding(b as u32).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(std::slice::from_ref(info)))
                .collect();
            let mut pc: Vec<u32> = vec![first.k as u32, (first.k >> 32) as u32, group.len() as u32, 0];
            for e in group {
                pc.push(e.q);
                pc.push(((e.off - lo) / 4) as u32);
            }
            pc.resize(4 + 2 * rdna::kernels::TS_MANY as usize, 0);
            let bytes: Vec<u8> = pc.iter().flat_map(|x| x.to_le_bytes()).collect();
            unsafe {
                c.device.update_descriptor_sets(&writes, &[]);
                c.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, tp.layout, 0, &[set], &[]);
                c.device.cmd_push_constants(cmd, tp.layout, vk::ShaderStageFlags::COMPUTE, 0, &bytes);
                c.device.cmd_dispatch(cmd, 1, 1, 1);
            }
            TS_DISPATCHES.fetch_add(1, Ordering::Relaxed);
            i = j;
        }
    }

    /// A wait passed on a value this batch writes at its end of pipe
    /// (`open_promise`): the work recorded from now on follows that write
    /// (and everything before it), as the wait makes it on the console.
    pub fn order_after_open_writes(&mut self) {
        if std::mem::take(&mut self.eop_unordered) {
            self.end_rendering_then_barrier();
            EOP_LIGHT[2].fetch_add(1, Ordering::Relaxed);
        }
    }

    /// The value the recording batch writes last to label `a` on the GPU
    /// (`gpu_cp_write`) and its size.
    pub fn open_promise(&self, a: u64) -> Option<(u64, u8)> {
        let find = |v: &[(u64, u64, u8)]| v.iter().rev().find(|p| p.0 == a).map(|p| (p.1, p.2));
        find(&self.pending_promises).or_else(|| find(&self.held_promises))
    }

    /// Whether the recording batch should be submitted now: another queue
    /// waits for one of its labels, or the batch has recorded for `tune
    /// label_latency`.
    pub fn submit_requested(&self) -> bool {
        self.recording && !self.pending_promises.is_empty() && (self.submit_request.load(Ordering::Relaxed) || self.begun_at.elapsed() > super::tune::label_latency())
    }

    /// Held past a run: whether another queue waits for a label of a batch
    /// it closed and holds (a stretch going on into later title submissions,
    /// `tune past_subs`, closes its batches at their ends); the stretch then
    /// settles, as for the recording batch's (`submit_requested`).
    pub fn held_label_requested(&self) -> bool {
        self.past && !self.held_promises.is_empty() && self.submit_request.load(Ordering::Relaxed)
    }

    /// Command labels (DS_GPU_LABELS=1, Nsight Graphics' GPU Trace): what
    /// is recorded from here to the next label is named `name` (a dispatch's
    /// or draw's program, or work of ours: merges, CP writes, timestamp
    /// stores, fills). Only while a batch is being recorded.
    pub(super) fn label(&mut self, name: &str) {
        if !self.recording || !super::vk::labels_on() {
            return;
        }
        let cmd = self.cmd();
        if self.label_open {
            super::vk::label_end(cmd);
        }
        super::vk::label_begin(cmd, name);
        self.label_open = true;
    }

    /// Ends the open label before the batch's command buffer ends.
    fn label_close(&mut self) {
        if std::mem::take(&mut self.label_open) && self.recording {
            super::vk::label_end(self.cmd());
        }
    }

    /// Whether this executor's indirect dispatches read their arguments from
    /// a resident shadow's working copy (`tune shadow_args`).
    fn shadow_args(&self) -> bool {
        match super::tune::shadow_args() {
            0 => false,
            2 => !self.compute_cp,
            _ => true,
        }
    }

    /// Indirect arguments at `a` (12 bytes) a resident shadow holds whole and
    /// nothing wrote in place since its copy in (not tainted): its working
    /// copy has the newest of every dword in its window (`tune shadow_args`).
    fn resident_args(&self, a: u64) -> Option<(vk::Buffer, u64)> {
        self.resident.iter().find(|r| !r.tainted && r.guest <= a && a + 12 <= r.guest + r.range).map(|r| (self.shadow(), r.shadow_off + (a - r.guest)))
    }

    /// A read-only binding of guest [base, base+size) inside a resident shadow
    /// reads its working copy: (buffer, offset of `base`, bytes from `base`
    /// to the copy's end). A binding that overlaps one only partly gets it
    /// merged back first and None.
    fn resident_read(&mut self, base: u64, size: u64) -> Result<Option<(vk::Buffer, u64, u64)>, String> {
        if let Some(r) = self.resident.iter().find(|r| r.guest <= base && base + size <= r.guest + r.range).copied() {
            RESIDENT_READS.fetch_add(1, Ordering::Relaxed);
            return Ok(Some((self.shadow(), r.shadow_off + (base - r.guest), r.guest + r.range - base)));
        }
        if self.resident.iter().any(|r| r.guest < base + size && base < r.guest + r.range) {
            EARLY_MERGES.fetch_add(1, Ordering::Relaxed);
            self.flush_resident(Some(&[(base, size)]))?;
        }
        Ok(None)
    }

    /// Writes the resident shadows overlapping `only` (all: None) back to
    /// guest memory: the dwords their dispatches changed within the
    /// resources' own bytes (see Shadow), between barriers.
    pub(super) fn flush_resident(&mut self, only: Option<&[(u64, u64)]>) -> Result<(), String> {
        let hit = |r: &Shadow| only.is_none_or(|rs| rs.iter().any(|&(a, n)| a < r.guest + r.range && r.guest < a + n));
        if !self.resident.iter().any(hit) {
            return Ok(());
        }
        let (flush, keep): (Vec<Shadow>, Vec<Shadow>) = std::mem::take(&mut self.resident).into_iter().partition(hit);
        self.resident = keep;
        let n = flush.len();
        if super::vk::labels_on() {
            let l = format!("{} merges ({n})", self.name);
            self.label(&l);
        }
        let c = ctx();
        // The dispatches' writes to the working copies come first.
        self.end_rendering_then_barrier();
        super::vk::checkpoint(self.cmd(), super::vk::mark::SHADOW_OUT, 0);
        let merge = if shadow_merge() { Some(merge_pipe()?) } else { None };
        // Diagnosis (`tune skip_copies 2`): the merges left out (timing only).
        // 16/32/64: only those of windows of at least 1 MiB / 256 KiB to 1 MiB
        // / under 256 KiB.
        let skip = super::tune::skip_copies();
        let by_class: Vec<Shadow>;
        let flushing: &[Shadow] = if skip & 2 != 0 {
            &[]
        } else if skip & 112 != 0 {
            let class = |r: u64| if r >= 1 << 20 { 16 } else if r >= 256 << 10 { 32 } else { 64 };
            by_class = flush.iter().filter(|s| skip & class(s.range) == 0).copied().collect();
            &by_class
        } else {
            &flush
        };
        let many = merge.is_some() && flushing.len() > 1 && super::tune::merge_many() && c.device_addresses && self.merge_many(flushing)?;
        for sh in flushing.iter().filter(|_| !many) {
            let own = sh.size.min(sh.range - sh.bias);
            // Diagnosis (`tune skip_copies`, timing only): windows under 256
            // KiB written back by a whole copy (128) or merged into video
            // memory instead of guest memory (256).
            let small = sh.range < 256 << 10;
            let merge = if small && skip & 128 != 0 { None } else { merge };
            let dst = match (scratch_buffer(), host_scratch()) {
                (Some(sb), _) if small && skip & 256 != 0 => (sb, 0),
                (_, Some(hb)) if small && skip & 2048 != 0 => (hb, 0),
                _ => match (imported_scratch(), imported_spread()) {
                    (Some(ib), _) if small && skip & 4096 != 0 => (ib, 0),
                    (_, Some(ib)) if small && skip & 8192 != 0 => (ib, (sh.guest % (SPREAD_BYTES - (256 << 10))) & !255),
                    _ => (sh.host, sh.host_off),
                },
            };
            // Diagnosis (`tune skip_copies`, guest memory merged as always):
            // windows under 256 KiB merged a second time into guest memory
            // (32768) or into the spread scratch (65536): what a merge's
            // stores cost without changing what later work reads.
            let again = match imported_spread() {
                Some(ib) if small && skip & 65536 != 0 => Some((ib, (sh.guest % (SPREAD_BYTES - (256 << 10))) & !255)),
                _ if small && skip & 32768 != 0 => Some(dst),
                _ => None,
            };
            for dst in std::iter::once(dst).chain(again) {
                match merge {
                    Some(m) => {
                        let set = self.alloc_set(m.set_layout)?;
                        let infos = [
                            vk::DescriptorBufferInfo { buffer: self.shadow(), offset: sh.shadow_off, range: sh.range },
                            vk::DescriptorBufferInfo { buffer: self.shadow(), offset: sh.pristine_off, range: sh.range },
                            vk::DescriptorBufferInfo { buffer: dst.0, offset: dst.1, range: sh.range },
                        ];
                        let writes: Vec<vk::WriteDescriptorSet> = infos
                            .iter()
                            .enumerate()
                            .map(|(k, info)| vk::WriteDescriptorSet::default().dst_set(set).dst_binding(k as u32).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(std::slice::from_ref(info)))
                            .collect();
                        let first = (sh.bias / 4) as u32;
                        let end = ((sh.bias + own).div_ceil(4)).min(sh.range / 4) as u32;
                        let pc: Vec<u8> = [first, end].iter().flat_map(|x| x.to_le_bytes()).collect();
                        unsafe {
                            let cmd = self.cmd();
                            c.device.update_descriptor_sets(&writes, &[]);
                            c.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, m.pipeline);
                            c.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, m.layout, 0, &[set], &[]);
                            c.device.cmd_push_constants(cmd, m.layout, vk::ShaderStageFlags::COMPUTE, 0, &pc);
                            c.device.cmd_dispatch(cmd, end.saturating_sub(first).div_ceil(rdna::kernels::MERGE_LOCAL_SIZE).max(1), 1, 1);
                        }
                    }
                    None => unsafe {
                        let r = vk::BufferCopy { src_offset: sh.shadow_off + sh.bias, dst_offset: sh.host_off + sh.bias, size: own };
                        c.device.cmd_copy_buffer(self.cmd(), self.shadow(), sh.host, &[r]);
                    },
                }
            }
        }
        if let Some(m) = merge {
            self.dyncache.cs_pipeline = m.pipeline;
        }
        if merge.is_some() && super::tune::shadow_share() && super::precache::enabled() {
            self.share_shadows(&flush);
        }
        if super::smalltrack::on() {
            for sh in &flush {
                super::smalltrack::merged(sh.guest, sh.range, &self.name);
            }
        }
        if super::tune::keep_small() {
            for sh in &flush {
                super::keep::note_merged(sh.guest, sh.range);
            }
        }
        if profiling() {
            let mut g = MERGE_BY.lock();
            let m = g.get_or_insert_with(HashMap::new);
            let tag = queue_tag(&self.name);
            for sh in &flush {
                let e = m.entry((sh.guest, sh.range)).or_insert([0; 4]);
                e[0] += 1;
                e[1] += sh.range;
                e[2] = sh.writer;
                e[3] = tag;
            }
        }
        // Later work reads guest memory after the write-back (counted
        // before that barrier: `seg_stamps` names the segment by it).
        self.ops[OP_MERGES] += flush.len() as u64;
        self.barrier();
        self.barrier_wanted = false;
        self.dispatched = false;
        MERGES.fetch_add(flush.len() as u64, Ordering::Relaxed);
        Ok(())
    }

    /// The write-back of `flushing` by one dispatch per MERGE_MANY_MAX
    /// shadows (`tune merge_many`): a table in the ring, destinations by
    /// device address. False when that is not possible here (no ring room,
    /// an import without an address): the per-shadow merges run instead.
    fn merge_many(&mut self, flushing: &[Shadow]) -> Result<bool, String> {
        use rdna::kernels::{MERGE_ENTRY_DWORDS, MERGE_LOCAL_SIZE, MERGE_MANY_MAX};
        let c = ctx();
        let Ok(m) = merge_many_pipe() else { return Ok(false) };
        let Some((ring, ring_ptr)) = self.slots[self.cur].rings[0] else { return Ok(false) };
        let mut entries: Vec<[u32; MERGE_ENTRY_DWORDS as usize]> = Vec::with_capacity(flushing.len());
        for sh in flushing {
            let va = c.buffer_address(sh.host);
            if va == 0 {
                return Ok(false);
            }
            let own = sh.size.min(sh.range - sh.bias);
            let first = (sh.bias / 4) as u32;
            let end = ((sh.bias + own).div_ceil(4)).min(sh.range / 4) as u32;
            let dst = va + sh.host_off;
            entries.push([0, (sh.shadow_off / 4) as u32, (sh.pristine_off / 4) as u32, first, end, dst as u32, (dst >> 32) as u32, 0]);
        }
        let bytes = 4 * MERGE_ENTRY_DWORDS as u64 * entries.len() as u64;
        let Some(off) = self.ring_alloc(0, bytes) else { return Ok(false) };
        // Chunks of MERGE_MANY_MAX entries, each a dispatch whose first
        // workgroup is 0: (first entry, entries, workgroups).
        let mut chunks: Vec<(u32, u32, u32)> = Vec::new();
        for (k, chunk) in entries.chunks_mut(MERGE_MANY_MAX as usize).enumerate() {
            let mut wg = 0u32;
            for e in chunk.iter_mut() {
                e[0] = wg;
                wg += e[4].saturating_sub(e[3]).div_ceil(MERGE_LOCAL_SIZE).max(1);
            }
            chunks.push((k as u32 * MERGE_MANY_MAX, chunk.len() as u32, wg));
        }
        unsafe { std::ptr::copy_nonoverlapping(entries.as_ptr() as *const u8, ring_ptr.add(off as usize), bytes as usize) };
        let set = self.alloc_set(m.set_layout)?;
        let infos = [vk::DescriptorBufferInfo { buffer: self.shadow(), offset: 0, range: vk::WHOLE_SIZE }, vk::DescriptorBufferInfo { buffer: ring, offset: off, range: bytes }];
        let writes: Vec<vk::WriteDescriptorSet> = infos
            .iter()
            .enumerate()
            .map(|(k, info)| vk::WriteDescriptorSet::default().dst_set(set).dst_binding(k as u32).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(std::slice::from_ref(info)))
            .collect();
        unsafe {
            let cmd = self.cmd();
            c.device.update_descriptor_sets(&writes, &[]);
            c.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, m.pipeline);
            c.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, m.layout, 0, &[set], &[]);
            for &(first, n, wg) in &chunks {
                let pc: Vec<u8> = [first * MERGE_ENTRY_DWORDS, n].iter().flat_map(|x| x.to_le_bytes()).collect();
                c.device.cmd_push_constants(cmd, m.layout, vk::ShaderStageFlags::COMPUTE, 0, &pc);
                c.device.cmd_dispatch(cmd, wg.max(1), 1, 1);
            }
        }
        self.dyncache.cs_pipeline = m.pipeline;
        MERGE_MANY[0].fetch_add(chunks.len() as u64, Ordering::Relaxed);
        MERGE_MANY[1].fetch_add(flushing.len() as u64, Ordering::Relaxed);
        Ok(true)
    }

    /// After the merge of `flush` (recorded): the working copies of the
    /// shadows nothing else wrote meanwhile become shared copies of their
    /// windows (`precache`; `tune shadow_share`), spared by this batch's own
    /// submission-time invalidation.
    fn share_shadows(&mut self, flush: &[Shadow]) {
        let c = ctx();
        for sh in flush {
            let Some(since) = sh.since else { continue };
            if sh.tainted {
                SHADOWS_SHARED[2].fetch_add(1, Ordering::Relaxed);
                continue;
            }
            if !super::precache::unwritten_since(sh.guest, sh.range, since) {
                SHADOWS_SHARED[3].fetch_add(1, Ordering::Relaxed);
                continue;
            }
            let Some((chunk, at, id)) = super::precache::insert(sh.guest, sh.range, sh.range) else {
                SHADOWS_SHARED[4].fetch_add(1, Ordering::Relaxed);
                continue;
            };
            // Reads the working copy, as the merge does (no hazard between them).
            let r = vk::BufferCopy { src_offset: sh.shadow_off, dst_offset: at, size: sh.range };
            unsafe { c.device.cmd_copy_buffer(self.cmd(), self.shadow(), chunk.buf, &[r]) };
            self.slots[self.cur].keep.push(chunk);
            self.shared_pending.push((sh.guest, sh.range, id, false));
            self.shared_kept.push(id);
            SHADOWS_SHARED[0].fetch_add(1, Ordering::Relaxed);
        }
    }

    /// A write into guest memory in place (not through a shadow) on this
    /// executor: resident shadows overlapping it no longer hold what their
    /// merge leaves there.
    fn taint_resident(&mut self, a: u64, n: u64) {
        for r in self.resident.iter_mut() {
            if r.guest < a.saturating_add(n) && a < r.guest + r.range {
                r.tainted = true;
            }
        }
    }

    /// A copy of guest [base, base+size) in video memory, taken now (when the
    /// command processor reaches the work that reads it): (buffer, offset of
    /// the copy's 16-byte aligned start, bias of `base` in it). Bound instead
    /// of the imported system memory, which every invocation would read across
    /// PCIe. None: the range is larger than `limit` (0: off), GPU work not yet
    /// complete (or a deferred CP write) writes it, or no space.
    pub(super) fn snapshot(&mut self, base: u64, size: u64, limit: u64) -> Option<(vk::Buffer, u64, u64)> {
        self.snapshot_as(base, size, limit, true)
    }

    /// A snapshot of [base, base + size) as memory holds it now, never one
    /// taken earlier in the batch (CP writes may have changed it since).
    pub(super) fn snapshot_exact(&mut self, base: u64, size: u64) -> Option<(vk::Buffer, u64, u64)> {
        DATA_SNAPS.fetch_add(1, Ordering::Relaxed);
        self.snapshot_as(base, size, 65536, false)
    }

    fn snapshot_as(&mut self, base: u64, size: u64, limit: u64, reuse: bool) -> Option<(vk::Buffer, u64, u64)> {
        // The frame-ahead recorder copies at its batch's start instead: on
        // the GPU, after the current frame's work (`set_held`).
        if size == 0 || size > limit || self.snap_failed || self.held || super::homes::overlaps(base, size) {
            return None;
        }
        let r = [(base, size)];
        if CP_WRITES.latest(&r).is_some() || INFLIGHT.latest(&r).is_some() || self.overlaps_pending(&r) {
            return None;
        }
        let al = base & !15;
        let bias = base - al;
        if let (true, Some(&at), Some((buf, _))) = (reuse, self.snaps.get(&(base, size)), self.slots[self.cur].snap) {
            SNAPS_REUSED.fetch_add(1, Ordering::Relaxed);
            return Some((buf, at, bias));
        }
        if self.slots[self.cur].snap.is_none() {
            match ctx().bar_buffer(SNAP_SEG, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDEX_BUFFER) {
                Ok((b, _m, p)) => {
                    SLOT_MEM[3].fetch_add(SNAP_SEG, Ordering::Relaxed);
                    self.slots[self.cur].snap = Some((b, p))
                }
                Err(e) => {
                    crate::warn!("gpu[{}]: no video memory for buffer snapshots ({e})", self.name);
                    self.snap_failed = true;
                    return None;
                }
            }
        }
        let (buf, ptr) = self.slots[self.cur].snap?;
        let seg = SNAP_SEG;
        let n = (bias + size).next_multiple_of(4);
        let off = self.snap_used.next_multiple_of(self.ring_align);
        if off + n > seg {
            SNAPS_FULL.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        if !super::cp::guest_readable(al, n) {
            return None;
        }
        self.snap_used = off + n;
        let at = off;
        unsafe { std::ptr::copy_nonoverlapping(al as *const u8, ptr.add(at as usize), n as usize) };
        SNAPS.fetch_add(1, Ordering::Relaxed);
        SNAP_COPIED.fetch_add(n, Ordering::Relaxed);
        self.snaps.insert((base, size), at);
        Some((buf, at, bias))
    }

    /// Allocates SGPR data space in the current slot's ring segment.
    fn ring_alloc(&mut self, which: usize, bytes: u64) -> Option<u64> {
        self.ring_alloc_at(which, bytes, self.ring_align)
    }

    /// `ring_alloc` at a multiple of `align`.
    fn ring_alloc_at(&mut self, which: usize, bytes: u64, align: u64) -> Option<u64> {
        let seg = RING_SEG;
        let off = self.ring_used[which].next_multiple_of(align);
        if off + bytes > seg {
            return None;
        }
        self.ring_used[which] = off + bytes;
        Some(off)
    }

    /// Makes room in the recording slot for the SGPR data and descriptors of
    /// `stages` (submits the batch and starts the next one otherwise).
    /// `push`: the command pushes its descriptors (no set from the pool).
    fn reserve(&mut self, stages: &[&Prepared], push: bool) {
        self.begin();
        if !self.recording {
            return;
        }
        let mut need = [if push { 0 } else { stages.len() as u32 }, 0, 0, 0, 0, 0];
        let mut ring = 0u64;
        for p in stages {
            // Room for the uniform buffer's alignment and 16-byte rounding.
            ring += 4 * p.sgpr_data.len() as u64 + self.ubo_align + 16;
            if push {
                continue;
            }
            for b in descriptor_bindings(&p.tr.out) {
                let ty = descriptor_type(b.kind);
                if let Some(k) = POOL_TYPES.iter().position(|&t| t == ty) {
                    need[1 + k] += binding_count(b.kind);
                }
            }
        }
        // Shadow space for written buffers (dispatches; see bind_stage): a
        // working and a pristine copy, and a descriptor set for the merge.
        let mut shadow = 0u64;
        for p in stages {
            for b in &p.tr.out.bindings {
                if let BindingKind::Buffer(i) | BindingKind::Address(i) = b.kind {
                    if !p.tr.plan.resources[i].written {
                        continue;
                    }
                    // Any written buffer may be shadowed (pointer-addressed
                    // ones when their mapping ends within `tune shadow_max`).
                    need[0] += 1;
                    need[1] += 3;
                    if let Some(Resolved::Buffer(v)) = p.resources.get(i) {
                        if let Some(n) = v.byte_size().filter(|&n| n + 16 <= super::tune::shadow_max()) {
                            shadow += 2 * (n + 16 + 256);
                        }
                    }
                }
            }
        }
        let seg = RING_SEG;
        let fits = self.ring_used.iter().all(|u| u.next_multiple_of(self.ring_align) + ring <= seg)
            && (0..6).all(|k| self.pool_use[k] + need[k] <= POOL_LIMITS[k])
            && (shadow == 0 || self.shadow_used + shadow <= self.slots[self.cur].shadow_cap || self.shadow_room(self.shadow_used + shadow));
        if !fits {
            self.submit_for("full");
            self.begin();
            if !self.recording {
                return;
            }
        }
        for k in 0..6 {
            self.pool_use[k] += need[k];
        }
    }

    /// A full barrier in the batch being recorded (see `global_barrier`).
    #[track_caller]
    fn barrier(&mut self) {
        if profiling() {
            let class = if self.deferred { 3 } else if self.name == "gpu-acb0x50" { 2 } else if self.compute_cp { 1 } else { 0 };
            let l = std::panic::Location::caller();
            *BARRIER_QUEUE_SITES.lock().get_or_insert_with(HashMap::new).entry((class, l.file(), l.line())).or_default() += 1;
        }
        global_barrier(self.cmd());
        if seg_stamps() {
            let qp = self.slots[self.cur].queries;
            let query = self.segs.len() as u32;
            if qp != vk::QueryPool::null() && query < STAMPS * MAX_TIMED {
                unsafe { ctx().device.cmd_write_timestamp(self.cmd(), vk::PipelineStageFlags::BOTTOM_OF_PIPE, qp, query) };
                self.segs.push(SegMark { query, cmds: self.batch_shaders.len() as u32, ops: self.ops, site: std::panic::Location::caller() });
            }
        }
        self.barrier_noted();
    }

    /// What a full barrier resets in this executor's tracking (the work
    /// since the last one), without recording one: for barriers recorded
    /// otherwise that order all earlier work before all later work (front
    /// end 2's write-back, `tune fe2_cpw_light`).
    fn barrier_noted(&mut self) {
        self.barrier_wanted = false;
        self.dispatched = false;
        self.args_read = false;
        match self.since_barrier.as_mut() {
            Some(v) => v.clear(),
            None => self.since_barrier = Some(Vec::new()),
        }
        match self.since_reads.as_mut() {
            Some(v) => v.clear(),
            None => self.since_reads = Some(Vec::new()),
        }
        self.since_img_r.clear();
        self.since_img_w.clear();
        self.since_gds = false;
        self.since_draws = false;
    }

    /// Whether a dispatch binding `bound` (and reading its arguments at
    /// `args`) touches what the work since the last full barrier wrote, or
    /// writes what it read: 0 no, 1 not known (a rendering instance, a list
    /// over SINCE_MAX), 2 it reads or writes a range written, 3 it writes a
    /// range read, 4 images or the global data share (`TITLE_BARRIERS`).
    fn dispatch_hazard(&self, bound: &Bound, args: Option<u64>) -> usize {
        let (Some(w), Some(r)) = (self.since_barrier.as_ref(), self.since_reads.as_ref()) else { return 1 };
        if self.since_draws {
            return 1;
        }
        let over = |v: &[(u64, u64)], a: u64, n: u64| v.iter().any(|&(x, m)| x < a.saturating_add(n) && a < x.saturating_add(m));
        if bound.reads.iter().chain(args.map(|a| (a, 12)).iter()).any(|&(a, n)| over(w, a, n)) || bound.writes.iter().any(|&(a, n)| over(w, a, n)) {
            return 2;
        }
        if bound.writes.iter().any(|&(a, n)| over(r, a, n)) {
            return 3;
        }
        if (bound.gds && self.since_gds) || bound.images.iter().any(|i| self.since_img_w.contains(i)) || bound.stored.iter().any(|i| self.since_img_r.contains(i)) {
            return 4;
        }
        0
    }

    /// A dispatch binding `bound` is recorded: what it reads and samples
    /// joins the lists since the last barrier (its writes come through
    /// `note_write`).
    fn note_since_reads(&mut self, bound: &Bound, args: Option<u64>) {
        if let Some(v) = self.since_reads.as_mut() {
            v.extend(bound.reads.iter().copied().chain(args.map(|a| (a, 12))));
            if v.len() > SINCE_MAX {
                self.since_reads = None;
            }
        }
        for &i in &bound.images {
            if !self.since_img_r.contains(&i) {
                self.since_img_r.push(i);
            }
        }
        for &i in &bound.stored {
            if !self.since_img_w.contains(&i) {
                self.since_img_w.push(i);
            }
        }
        self.since_gds |= bound.gds;
    }

    /// A GPU write of [a, a + n) recorded in this batch (`since_barrier`).
    fn note_since(&mut self, a: u64, n: u64) {
        if let Some(v) = self.since_barrier.as_mut() {
            if v.len() < if super::tune::race_checks() { 4096 } else { SINCE_MAX } {
                v.push((a, n));
            } else {
                self.since_barrier = None;
            }
        }
    }

    /// Diagnosis (`tune race_checks`): this executor's batch being recorded
    /// writes [a, a + n) (`OPEN_WRITES`).
    fn note_open_write(&self, a: u64, n: u64) {
        let mut g = OPEN_WRITES.lock();
        g.get_or_insert_with(HashMap::new).entry(self.race_id).or_insert_with(|| (self.race_name.clone(), Default::default())).1.insert(a, n);
    }

    /// Diagnosis: a binding of `shader` reads [a, a + n); counted when another
    /// queue's batch not yet submitted writes there (`OPEN_READS`).
    fn note_open_read(&self, shader: u64, a: u64, n: u64) {
        let g = OPEN_WRITES.lock();
        let Some(m) = g.as_ref() else { return };
        let mine = queue_family(&self.race_name);
        for (id, (q, r)) in m {
            if *id == self.race_id || queue_family(q) == mine || !r.overlaps(a, n) {
                continue;
            }
            OPEN_READ_COUNT.fetch_add(1, Ordering::Relaxed);
            let hit: Vec<String> = r.intersect(a, n).into_iter().take(3).map(|(b, k)| format!("{b:#x}+{k:#x}")).collect();
            let mut o = OPEN_READS.lock();
            let e = o.get_or_insert_with(HashMap::new).entry((self.race_name.clone(), q.clone(), shader)).or_insert_with(|| (0, format!("binds {a:#x}+{n:#x}, written {} (flip {})", hit.join(" "), super::FRAMES.load(Ordering::Relaxed))));
            e.0 += 1;
            if e.0 == 1 && OPEN_READS_LOGGED.fetch_add(1, Ordering::Relaxed) < 40 {
                crate::info!("gpu: race check: {} shader {shader:#x} binds {a:#x}+{n:#x}, which {q}'s batch not yet submitted writes ({})", self.race_name, hit.join(" "));
            }
        }
    }

    /// Diagnosis: this executor's writes went to the device (or were dropped).
    fn clear_open_writes(&self, which: Option<&super::ranges::Ranges>) {
        let mut g = OPEN_WRITES.lock();
        if let Some(e) = g.as_mut().and_then(|m| m.get_mut(&self.race_id)) {
            match which {
                Some(w) => {
                    for (a, n) in w.iter() {
                        e.1.remove(a, n);
                    }
                }
                None => e.1.clear(),
            }
        }
    }

    /// The write of [a, a + n) just noted precedes a full barrier.
    fn ordered_since(&mut self, a: u64, n: u64) {
        if let Some(v) = self.since_barrier.as_mut() {
            if v.last() == Some(&(a, n)) {
                v.pop();
            }
        }
    }

    /// Whether this batch wrote into [a, a + n) since its last full barrier.
    fn written_since_barrier(&self, a: u64, n: u64) -> bool {
        self.since_barrier.as_ref().is_none_or(|v| v.iter().any(|&(w, m)| w < a + n && a < w + m))
    }

    /// Diagnosis (`tune race_checks`): counts the in-place bindings of the
    /// command about to be recorded that this batch's GPU work wrote since
    /// its last full barrier (`UNORDERED`; the first 60 read-only ones are
    /// logged).
    fn note_unordered(&self, bounds: &[Bound], shaders: &[u64], draw: bool) {
        if self.since_barrier.is_none() {
            UNORDERED[5].fetch_add(1, Ordering::Relaxed);
            return;
        }
        for (b, &sh) in bounds.iter().zip(shaders) {
            for &(a, n, kind) in &b.in_place {
                if !self.written_since_barrier(a, n) {
                    continue;
                }
                let written = kind & 1 != 0;
                let k = if written { 2 } else { 0 } + if draw { 0 } else { 1 };
                UNORDERED[k].fetch_add(1, Ordering::Relaxed);
                if written || draw {
                    continue;
                }
                if kind & 2 != 0 {
                    UNORDERED[6].fetch_add(1, Ordering::Relaxed);
                } else if kind & 4 != 0 {
                    UNORDERED[7].fetch_add(1, Ordering::Relaxed);
                }
                let mut g = UNORDERED_BY.lock();
                let e = g.get_or_insert_with(HashMap::new).entry((sh, kind)).or_insert_with(|| {
                    let w: Vec<String> = self.since_barrier.iter().flatten().filter(|&&(w, m)| w < a + n && a < w + m).take(4).map(|(w, m)| format!("{w:#x}+{m:#x}")).collect();
                    (0, format!("binds {a:#x}+{n:#x}, written {} (last shaders {:x?}, {})", w.join(" "), &self.batch_shaders[self.batch_shaders.len().saturating_sub(3)..], self.name))
                });
                e.0 += 1;
            }
        }
    }

    /// What the batch being recorded writes (a parallel run's planning).
    pub fn unsubmitted_writes(&self) -> super::ranges::Ranges {
        if self.recording { self.pending.clone() } else { Default::default() }
    }

    /// A worker's chunk follows a main command processor's batch that
    /// writes `w` and was not yet submitted when the run was planned
    /// (`submitted`: set once it is, `tune par_main_writes 3`).
    pub fn set_main_writes(&mut self, w: Option<std::sync::Arc<super::ranges::Ranges>>, submitted: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>) {
        self.main_writes = w;
        self.main_submitted = submitted;
    }


    /// Makes this a deferred executor (`gpu::par` workers).
    pub fn set_deferred(&mut self) {
        self.deferred = true;
    }

    /// A deferred executor (`gpu::par` workers record with one).
    pub fn is_deferred(&self) -> bool {
        self.deferred
    }

    /// Makes this the frame-ahead recorder's executor (`held`).
    pub fn set_held(&mut self) {
        self.deferred = true;
        self.held = true;
    }

    /// The graphics command processor goes on past a parallel run before
    /// collecting its chunks (`Cp::past_start`): held from here (its own
    /// slots; its batch before the run was submitted).
    pub fn past_begin(&mut self) {
        self.deferred = true;
        self.held = true;
        self.past = true;
        self.held_reset();
        // (Its work so far was submitted: a request now is for a label it
        // holds, `held_label_requested`.)
        self.submit_request.store(false, Ordering::SeqCst);
        // Its forwarded copies stay its own until the stretch commits (the
        // run's workers record earlier commands meanwhile: `tune
        // forward_held`).
        if super::tune::forward_held() && self.name == "gpu-gfx" {
            super::forward::hold_begin();
        }
    }

    /// The graphics command processor goes on past a label wait
    /// (`Cp::past_wait_start`): held as past a run, with batch slots added
    /// as its held batches fill them.
    pub fn past_begin_growing(&mut self) {
        self.past_begin();
        // Past a label wait it records in program order, with no workers:
        // its copies forward globally as before.
        super::forward::hold_drop();
        self.past_grow = true;
        self.past_want_min = u32::MAX;
        self.past_ranges.clear();
        self.past_up.clear();
    }

    /// Whether what the stretch past a label wait placed is current as of
    /// epoch `need` (`tune past_check`: its label orders the CPU writes
    /// before that epoch ahead of its work, and its checks needed older
    /// ones): each range is checked again as of `need` (ranges taken since
    /// pass as they are, others are taken), the pages its own batches
    /// upload counting as its own; a page that would need an upload was
    /// written since the stretch checked it.
    pub fn past_current_for(&mut self, need: u32) -> bool {
        let mut ranges = std::mem::take(&mut self.past_ranges);
        ranges.sort_unstable();
        let up = std::mem::take(&mut self.past_up);
        // A label the CPU released needs every page taken after it: a new
        // epoch, so that the takes below are stamped with it (and no range
        // is noted fine for an epoch that has not begun).
        if need > super::precache::epoch() as u32 {
            super::precache::end_scope(1);
        }
        let ep = super::precache::epoch() as u32;
        let need = need.min(ep);
        let t = std::time::Instant::now();
        let (mut stale, mut checked) = (false, 0u64);
        let mut i = 0;
        while i < ranges.len() && !stale {
            // (Overlapping and adjacent ranges as one.)
            let (lo, mut hi) = ranges[i];
            i += 1;
            while i < ranges.len() && ranges[i].0 <= hi {
                hi = hi.max(ranges[i].1);
                i += 1;
            }
            checked += 1;
            let r = super::fe2::ensure(lo, hi, need, ep, false, &|p| up.contains(&p), !up.is_empty(), &mut |a, b| {
                stale = true;
                if PAST_STALE_LOGGED.fetch_add(1, Ordering::Relaxed) < 40 {
                    // (try_lock: the caller may hold the memory map.)
                    let map = crate::mem::MEM.try_lock().map_or_else(|| "busy".into(), |m| m.find(a).map_or_else(|| "none".into(), |v| format!("{:#x}+{:#x} {:?} '{}'", v.start, v.size, v.kind, v.name)));
                    crate::info!("gpu[{}]: a stretch past a label wait read {a:#x}..{b:#x} ({} pages) of {lo:#x}..{hi:#x}, written since its check (needs epoch {need}, its checks {}; flip {}; mapping {map})", self.name, (b - a) / super::resident::PAGE, self.past_want_min, super::FRAMES.load(Ordering::Relaxed));
                }
                true
            }, &fe2_armable, None);
            stale |= r.is_err();
        }
        PAST_CHECKS[0].fetch_add(1, Ordering::Relaxed);
        PAST_CHECKS[1].fetch_add(checked, Ordering::Relaxed);
        PAST_CHECKS[2].fetch_add(stale as u64, Ordering::Relaxed);
        PAST_CHECKS[3].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        !stale
    }

    /// A worker begins a chunk (`on`: of a run planned past a label wait,
    /// recorded as the stretch's work; `tune wait_runs`).
    pub fn stretch_set(&mut self, on: bool) {
        self.stretch = on;
        if on {
            self.past_want_min = u32::MAX;
            self.past_ranges.clear();
            self.past_up.clear();
        }
    }

    /// The view this queue's label writes submitted now are promised with
    /// (`cp::promise`): its view, the epoch now without one.
    fn promise_view(&self) -> u32 {
        match self.fe2_view.load(Ordering::Acquire) {
            super::fe2::NO_VIEW => super::precache::epoch() as u32,
            v => v,
        }
    }

    /// The oldest epoch the checks of the stretch past a label wait needed
    /// (u32::MAX: it made none): one older than its label needs may have
    /// missed a CPU write the title ordered before the label (`Cp::past_wait_settle`).
    pub fn past_checks_want(&self) -> u32 {
        self.past_want_min
    }

    /// What it recorded past the run holds (`Cp::past_settle`): its closed
    /// batches go to the GPU after the run's chunks', with their label
    /// writes; its image notes go to the image cache; the open batch goes
    /// on as an ordinary one.
    pub fn past_commit(&mut self) {
        // Its forwarded copies become everyone's before its batches are
        // submitted (`forward::submitted` dates them).
        super::forward::hold_commit();
        self.seals.clear();
        self.deferred = false;
        self.held = false;
        self.past = false;
        self.past_grow = false;
        self.stopped = false;
        self.stop_why = "";
        if !self.held_clears.is_empty() || !self.held_over.is_empty() {
            let (c, o) = (std::mem::take(&mut self.held_clears), std::mem::take(&mut self.held_over));
            super::images::with(|im| im.apply_held(&c, &o));
        }
        // The open batch's label writes can be asked for now (`gpu_cp_write`).
        if self.recording && !self.pending_promises.is_empty() {
            let mut g = UNSUBMITTED_LABELS.lock();
            let m = g.get_or_insert_with(HashMap::new);
            for p in &self.pending_promises {
                m.insert(p.0 & !7, self.submit_request.clone());
            }
        }
        self.held_promises.clear();
        self.held_reads.clear();
        self.held_kinds.clear();
        self.held_keys.clear();
        self.submit_closed();
    }

    /// The held work past a run is sealed here (`tune past_seal`: a long run
    /// is chained after it): its open batch is closed, and what it holds up
    /// to here commits before that run's chunks (`past_commit_sealed`). None
    /// (not sealed) when it holds image notes (fast clears or overwrites:
    /// they are not split) or its batch cannot be closed.
    pub fn past_seal(&mut self) -> Option<SealCheck> {
        if !self.past || self.stopped || !self.held_clears.is_empty() || !self.held_over.is_empty() {
            return None;
        }
        if self.recording {
            self.submit_for("seal");
        }
        if self.recording || self.stopped {
            return None;
        }
        let (c, p, w) = self.seals.iter().fold((0, 0, 0), |a, s| (a.0 + s.closed, a.1 + s.promises, a.2 + s.cp_written));
        let mut writes = super::ranges::Ranges::new();
        for x in &self.closed[c.min(self.closed.len())..] {
            writes.extend(&x.writes);
        }
        self.seals.push_back(SealParts { closed: self.closed.len() - c, promises: self.held_promises.len().saturating_sub(p), cp_written: self.cp_written.len().saturating_sub(w) });
        self.held_kinds.clear();
        Some(SealCheck { reads: std::mem::take(&mut self.held_reads), keys: std::mem::take(&mut self.held_keys), writes, fwd: super::forward::hold_snapshot() })
    }

    /// A seal given back unused (no run was chained after it): what it took
    /// out of the held lists goes back to them, and the work it closed stays
    /// held with what follows.
    pub fn past_unseal(&mut self, s: SealCheck) {
        self.seals.pop_back();
        self.held_reads.extend(s.reads);
        self.held_keys.extend(s.keys);
    }

    /// Its held label writes (address, value, bytes): a run chained after
    /// its held work sees them as written before it (`Cp::chain_next`).
    pub fn held_promise_labels(&self) -> Vec<(u64, u64, u8)> {
        self.held_promises.clone()
    }

    /// The oldest sealed part of the held work commits (`Cp::past_roll`):
    /// its forwarded copies become everyone's, its batches go to the GPU with
    /// their label writes; the rest stays held.
    pub fn past_commit_sealed(&mut self, fwd: Option<super::forward::HeldView>) {
        let Some(s) = self.seals.pop_front() else { return };
        if let Some(v) = fwd {
            super::forward::commit_snapshot(v);
        }
        self.held_promises.drain(..s.promises.min(self.held_promises.len()));
        self.cp_written.drain(..s.cp_written.min(self.cp_written.len()));
        let front: Vec<Closed> = self.closed.drain(..s.closed.min(self.closed.len())).collect();
        self.submit_closed_list(front);
    }

    /// What it recorded past the run is dropped (`Cp::past_settle`): its
    /// closed batches and its open one (begun past the run: the batch
    /// before it was submitted) never go to the GPU.
    pub fn past_drop(&mut self) {
        super::forward::hold_drop();
        self.seals.clear();
        super::forward::drop_entries(&std::mem::take(&mut self.forward_ids));
        self.discard_closed();
        if self.recording {
            self.discard();
            let c = ctx();
            let s = &mut self.slots[self.cur];
            unsafe {
                c.device.reset_command_buffer(s.cmd, vk::CommandBufferResetFlags::empty()).ok();
                c.device.reset_command_buffer(s.pre, vk::CommandBufferResetFlags::empty()).ok();
                c.device.reset_descriptor_pool(s.desc_pool, vk::DescriptorPoolResetFlags::empty()).ok();
            }
            s.homes.clear();
            s.keep.clear();
            s.kept.clear();
            s.seq = 0;
        }
        self.deferred = false;
        self.held = false;
        self.past = false;
        self.past_grow = false;
        self.stopped = false;
        self.stop_why = "";
        self.held_clears.clear();
        self.held_over.clear();
        self.held_promises.clear();
        self.held_reads.clear();
        self.held_kinds.clear();
        self.held_keys.clear();
    }

    /// The frame-ahead recorder starts a frame: not stopped, nothing read
    /// or noted.
    pub fn held_reset(&mut self) {
        self.stopped = false;
        self.stop_why = "";
        self.held_reads.clear();
        self.held_kinds.clear();
        self.cp_written.clear();
        self.held_clears.clear();
        self.held_over.clear();
        self.held_keys.clear();
    }

    /// The images the frame-ahead recorder's commands used since
    /// `held_reset`.
    pub fn held_keys(&self) -> &HashSet<super::images::ImageKey> {
        &self.held_keys
    }

    /// The frame-ahead recorder's own notes of writes `writes` (`sized`:
    /// the bounded ones) and of a fill (`Images::notes_for`).
    fn held_note(&mut self, writes: &[(u64, u64)], sized: &[(u64, u64)], fill: Option<((u64, u64), u32)>) {
        let (clears, over) = super::images::with(|im| im.notes_for(writes, sized, fill));
        for (k, meta, v) in clears {
            self.held_clears.insert(k, (meta, v));
        }
        self.held_over.extend(over);
    }

    /// Its notes of a buffer fill of `v` over [base, base+len) recorded in
    /// place; the GPU-owned images a zero fill clears (`zero_filled`).
    fn held_fill(&mut self, base: u64, len: u64, v: u32) -> Vec<(vk::Image, vk::Format)> {
        let zero = v == 0 && super::tune::zero_fill_clears();
        let r = [(base, len)];
        let (clears, over, zeroed) = super::images::with(|im| {
            let (c, o) = im.notes_for(&r, if zero { &[] } else { &r }, Some(((base, len), v)));
            (c, o, if zero { im.zero_fill_targets(base, len) } else { Vec::new() })
        });
        for (k, meta, v) in clears {
            self.held_clears.insert(k, (meta, v));
        }
        self.held_over.extend(over);
        for z in &zeroed {
            self.held_over.remove(&z.0);
        }
        zeroed.into_iter().map(|(_, i, f)| (i, f)).collect()
    }

    /// Notes the images a frame-ahead recorder's command uses; whether one
    /// has a fast clear or an overwrite of its own pending.
    pub(crate) fn held_hit(&mut self, keys: &[super::images::ImageKey]) -> bool {
        if !self.held {
            return false;
        }
        self.held_keys.extend(keys.iter().copied());
        keys.iter().any(|k| self.held_clears.contains_key(k) || self.held_over.contains(k))
    }

    /// Stops the frame-ahead recorder (`Cp::held_stop`).
    pub fn stop_held(&mut self, why: &'static str) {
        self.stop(why);
    }

    /// Guest ranges the frame-ahead recorder read on the CPU (`kind`: an
    /// index into `HELD_READ_KINDS`).
    pub fn note_held_reads(&mut self, reads: &[(u64, u64)], kind: u8) {
        if self.held {
            self.held_reads.extend_from_slice(reads);
            self.note_kinds(kind, reads.len());
        }
    }

    /// What it read on the CPU since `held_reset`.
    pub fn held_reads(&self) -> &[(u64, u64)] {
        &self.held_reads
    }

    /// The kinds of the last `n` held reads (`DS_TRACE_PAST_DROPS`).
    fn note_kinds(&mut self, kind: u8, n: usize) {
        if trace_past_drops() || self.long {
            self.held_kinds.resize(self.held_reads.len() - n, u8::MAX);
            self.held_kinds.resize(self.held_reads.len(), kind);
        }
    }

    /// The kind of held read `i` (`DS_TRACE_PAST_DROPS`).
    pub fn held_read_kind(&self, i: usize) -> &'static str {
        self.held_kinds.get(i).and_then(|&k| HELD_READ_KINDS.get(k as usize)).copied().unwrap_or("?")
    }

    /// Whether it holds recorded work (an open batch or closed ones).
    pub fn has_held_work(&self) -> bool {
        self.recording || !self.closed.is_empty()
    }

    /// Batches closed and held (not submitted).
    pub fn closed_count(&self) -> usize {
        self.closed.len()
    }

    /// What the closed batches write.
    pub fn closed_writes(&self) -> super::ranges::Ranges {
        let mut r = super::ranges::Ranges::new();
        for b in &self.closed {
            r.extend(&b.writes);
        }
        r
    }

    /// Stops a deferred executor's chunk (see `deferred`).
    fn stop(&mut self, why: &'static str) {
        if !self.stopped {
            self.stopped = true;
            self.stop_why = why;
        }
    }

    /// The reason the next flush has (a deferred executor's stop statistics).
    #[track_caller]
    fn flush_for(&mut self, why: &'static str) {
        self.flush_why = why;
        self.flush();
        self.flush_why = "";
    }

    /// Submits the batches this deferred executor closed, in order (the
    /// main command processor, after the work before them); returns the
    /// last one's sequence. At shutdown they are dropped.
    pub fn submit_closed(&mut self) -> Option<u64> {
        let all = std::mem::take(&mut self.closed);
        self.submit_closed_list(all)
    }

    /// Submits `list`, batches this deferred executor closed, in order; the
    /// last one's sequence.
    fn submit_closed_list(&mut self, list: Vec<Closed>) -> Option<u64> {
        let mut last = None;
        for b in list {
            if super::shutting_down() {
                // Never submitted: free for reuse (begin resets a command
                // buffer it records again).
                self.slots[b.slot].seq = 0;
                continue;
            }
            let (cmd, fence, queries, bq) = (self.slots[b.slot].cmd, self.slots[b.slot].fence, self.slots[b.slot].queries, self.slots[b.slot].bq);
            let name = self.name.clone();
            let (pre, copy_wait) = (b.pre, b.copy_wait);
            note_copy_waits(false, copy_wait);
            let reset = self.slot_reset(b.slot);
            let own = b.mm_own;
            // Front end 2: the words its uploads touch, before it can run.
            let fe2_ep = super::precache::epoch() as u32;
            if !b.fe2_up.is_empty() {
                let mut gens = std::mem::take(&mut self.fe2_gens);
                super::fe2::generations(&b.fe2_up, &mut gens);
                self.fe2_gens = gens;
            }
            let seq = push_batch(fence, queries, bq, b.eager_action.into_iter().collect(), b.checks, b.shaders, &self.name, Vec::new(), b.ops, b.segs, reset, false, move |_| {
                if DEVICE_LOST.load(Ordering::Relaxed) {
                    return None;
                }
                Some(SubmitSpec { xpre: None, pre, cmd, wait: copy_wait, fence, alt: false, name })
            });
            self.slots[b.slot].seq = seq;
            if !b.fe2_up.is_empty() {
                super::fe2::uploaded(&b.fe2_up, &self.fe2_gens, fe2_ep);
            }
            if !b.fe2_spans.is_empty() {
                note_span_uses(&b.fe2_spans, seq);
            }
            if let Some(o) = &own {
                super::resident::own_submitted(o, seq);
            }
            super::forward::submitted(&b.forward_ids, seq);
            if !b.races.is_empty() {
                let cpw_submit = super::cp::CPW_SEQ.load(Ordering::Acquire);
                RACE_CHECKS.lock().extend(b.races.into_iter().map(|c| RaceCheck { seq, cpw_submit, ..c }));
            }
            super::precache::submitted(&b.shared, seq);
            super::resident::submitted(&b.res_pages, seq);
            if !b.rgm_claims.is_empty() {
                if let Some(r) = super::rgm::get() {
                    r.submitted(&b.rgm_claims);
                }
            }
            INFLIGHT.note_ranges(seq, &b.writes);
            if super::rgm::get().is_some() {
                super::rgm::invalidate_except(&b.writes, &b.rgm_written);
            }
            journal_note(&b.writes, &self.name);
            crate::hostcensus::gpu_wrote(seq, &b.writes);
            if self.name.starts_with("gpu-gfx-par") {
                super::frameahead::a_wrote_ranges(&b.writes);
            }
            if !super::resident::mirror_mode() {
                super::resident::gpu_wrote(&b.writes);
            }
            if super::tune::race_checks() {
                note_race_writes(seq, &b.writes);
                self.clear_open_writes(Some(&b.writes));
            }
            super::precache::invalidate_ranges_except(&b.writes, &b.kept);
            self.inflight.insert_all(seq, &b.writes);
            // (After its writes are noted, as `submit_for` does.)
            for (a, v, n) in b.promises {
                super::cp::promise(seq, a, v, n, false, self.promise_view());
            }
            self.own_gen += 1;
            self.last_seq = seq;
            last = Some(seq);
        }
        if !self.recording {
            self.open_gen.store(u64::MAX, Ordering::SeqCst);
        }
        last
    }

    /// Closes the batch being recorded (a deferred executor at the end of
    /// its chunk).
    pub fn close(&mut self) {
        self.submit_for("chunk");
    }

    /// Drops the batches this deferred executor closed (never submitted):
    /// their slots are reset for reuse.
    pub fn discard_closed(&mut self) {
        let c = ctx();
        if super::tune::race_checks() {
            self.clear_open_writes(None);
        }
        for b in std::mem::take(&mut self.closed) {
            super::forward::drop_checks(&b.forward_ids);
            super::forward::drop_entries(&b.forward_ids);
            super::resident::dropped(&b.res_pages);
            if !b.rgm_claims.is_empty() {
                if let Some(r) = super::rgm::get() {
                    r.dropped(&b.rgm_claims);
                }
            }
            let s = &mut self.slots[b.slot];
            unsafe {
                c.device.reset_command_buffer(s.cmd, vk::CommandBufferResetFlags::empty()).ok();
                c.device.reset_command_buffer(s.pre, vk::CommandBufferResetFlags::empty()).ok();
                c.device.reset_descriptor_pool(s.desc_pool, vk::DescriptorPoolResetFlags::empty()).ok();
            }
            s.homes.clear();
            s.keep.clear();
            s.kept.clear();
            s.seq = 0;
        }
        self.held_promises.clear();
        if !self.recording {
            self.open_gen.store(u64::MAX, Ordering::SeqCst);
        }
    }

    /// Buffer, image and sampler bindings of a stage; fills the resource
    /// info blocks of its SGPR data.
    fn bind_stage(&mut self, p: &mut Prepared, allow_shadow: bool) -> Result<Bound, String> {
        // (Its placements counted by where the stage was made, for
        // `DS_FE2_CALLERS`.)
        let _source = super::fe2::StageSource::set(p.ahead, p.checked_ep);
        // What the preparation found (its own block, shared with its cache
        // entry), apart from the SGPR data written below.
        let found = p.found.clone();
        if self.held {
            if self.long && super::forward::active() {
                let (plain, fwd_planned, fwd) = super::forward::split_long_reads(&found.reads);
                self.held_reads.extend_from_slice(&plain);
                self.note_kinds(2, plain.len());
                self.held_fwd_reads.extend_from_slice(&fwd);
                self.held_fwd_planned.extend_from_slice(&fwd_planned);
            } else {
                self.held_reads.extend_from_slice(&found.reads);
                self.note_kinds(2, found.reads.len());
            }
        }
        super::images::BINDING_SHADER.with(|s| s.set(p.addr));
        let res_info = rdna::translate::res_info_offset(&found.tr.plan) as usize;
        let nb = found.tr.out.bindings.len();
        // A recycled binding's vectors keep their capacity (`recycle`).
        let mut bound = self.bound_pool.pop().unwrap_or_default();
        bound.bufs.reserve(nb);
        bound.imgs.reserve(nb);
        bound.images.reserve(nb);
        // The slot's shadow space (grown to what this executor's batches
        // need: `shadow_room` below).
        let mut seg = self.slots[self.cur].shadow_cap;
        let fx = fixed();
        BIND_PLACE.with(|f| f.set(place_class(if self.deferred { 2 } else if self.acb50 { 3 } else if self.compute_cp { 1 } else { 0 }, self.batch_shaders.len())));
        let mut clk = BindClock::start();
        {
            // Address-mode translations: the zero page in the address block's
            // header (`Spec::addresses`).
            let addr_at = rdna::translate::addr_info_offset(&found.tr.plan) as usize;
            let has_marks = found.tr.out.bindings.iter().any(|b| b.kind == BindingKind::Marks);
            if found.tr.out.bindings.iter().any(|b| matches!(b.kind, BindingKind::Address(_))) {
                p.sgpr_data[addr_at] = fx.zero_va as u32;
                p.sgpr_data[addr_at + 1] = (fx.zero_va >> 32) as u32;
            }
            let gpu_res = found.tr.spec.gpu_res;
            if gpu_res != 0 || found.tr.spec.gpu_loads != 0 {
                let pt = super::pagetable::address();
                p.sgpr_data[addr_at + 2] = pt as u32;
                p.sgpr_data[addr_at + 3] = (pt >> 32) as u32;
            }
            for b in &found.tr.out.bindings {
                match b.kind {
                    // Resolved by the shader through the page table (front end
                    // 2: in the mirror at guest offsets, its device address in
                    // the resource's address entry, `Spec::pt_linear`).
                    BindingKind::Address(i) if i < 64 && gpu_res & (1u64 << i) != 0 => {
                        if found.tr.spec.pt_linear {
                            let m = super::resident::mirror_address();
                            let ai = addr_at + (rdna::translate::ADDR_HEADER_DWORDS + rdna::translate::ADDR_INFO_DWORDS * i as u32) as usize;
                            p.sgpr_data[ai] = m as u32;
                            p.sgpr_data[ai + 1] = (m >> 32) as u32;
                        }
                    }
                    BindingKind::Buffer(i) | BindingKind::Address(i) => {
                        let (base, size, stride, limit) = match &found.resources[i] {
                            Resolved::Buffer(v) => (v.base, v.byte_size(), v.stride, v.index_limit()),
                            Resolved::Global(a, n) => (*a, *n, 0, u32::MAX),
                            _ => return Err("binding/resource mismatch".into()),
                        };
                        let mut size = match size {
                            Some(n) => n,
                            None => mapping(base).map(|(_, e)| (e - base).min(UNBOUNDED_MAX)).unwrap_or(0),
                        };
                        BindClock::mark(&mut clk, 0);
                        if size > 0 && !found.tr.plan.resources[i].written {
                            bound.reads.push((base, size));
                        }
                        // Written records of indirect draw arguments (16 or 20
                        // bytes): where draws may read them on the GPU (`tune
                        // draw_args_gpu`, `homes::vote_args`).
                        if matches!(stride, 16 | 20) && size <= super::tune::shadow_max() && found.tr.plan.resources[i].written && super::tune::draw_args_gpu() {
                            super::homes::note_shadow_write(base, size);
                        }
                        let mut info = vk::DescriptorBufferInfo { buffer: fx.dummy, offset: 0, range: DUMMY_BYTES };
                        let mut bias = 0u64;
                        let mut bound_size = 0u64;
                        let mut placement = Placement::None;
                        // Sized: buffers, and bases whose loads all read known
                        // bytes (data loads read by the shader).
                        let sized = matches!(found.resources[i], Resolved::Buffer(_) | Resolved::Global(_, Some(_)));
                        // Unbounded pointers whose mapping ends within `tune
                        // pre_unsized` bytes are read-only bindings of that much
                        // (a VRAM copy instead of reads across PCIe).
                        let read_only = size > 0 && !found.tr.plan.resources[i].written && (sized || size <= super::tune::pre_unsized());
                        if size > 0 && super::census::on() {
                            use super::census::{RO, RO_UNSIZED, WRITTEN};
                            super::census::note(base, size, if found.tr.plan.resources[i].written { WRITTEN } else if sized { RO } else { RO_UNSIZED });
                        }
                        // The residency mirror (read-only bindings, `gpu::resident`), or
                        // a video-memory home holding the range (see `homes`).
                        let from_mirror = if super::resident::mirror_mode() {
                            self.mm_unsized = matches!(found.resources[i], Resolved::Global(_, None));
                            // Diagnostics: large written bindings, the first of each shader.
                            if found.tr.plan.resources[i].written && size >= 16 << 20 {
                                static SEEN: Mutex<Option<HashSet<u64>>> = Mutex::new(None);
                                let mut g = SEEN.lock();
                                let seen = g.get_or_insert_with(HashSet::new);
                                if seen.len() < 40 && seen.insert(p.addr) {
                                    let what = match &found.resources[i] {
                                        Resolved::Buffer(v) => format!("V# stride {} records {} (index limit {})", v.stride, v.num_records, v.index_limit()),
                                        Resolved::Global(_, n) => format!("pointer ({n:?} bytes)"),
                                        _ => String::new(),
                                    };
                                    crate::info!("gpu[{}]: mirror mode: {:?} shader {:#x} writes {base:#x}+{size:#x}: {what}; atomic {}", self.name, p.stage, p.addr, found.tr.plan.resources[i].atomic);
                                }
                            }
                            self.mirror_place(base, size, found.tr.plan.resources[i].written)
                        } else if super::fe2::on() && super::fe2::in_span(base, size) {
                            // Diagnostics: large written bindings, the first of each shader
                            // (the global lock once per shader and thread, until 40 are
                            // logged: every thread's large written bindings took it).
                            static SEEN_FULL: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
                            thread_local! {
                                static SEEN_HERE: std::cell::RefCell<HashSet<u64>> = std::cell::RefCell::new(HashSet::new());
                            }
                            if found.tr.plan.resources[i].written && size >= 4 << 20 && !SEEN_FULL.load(Ordering::Relaxed) && SEEN_HERE.with(|s| s.borrow_mut().insert(p.addr)) {
                                static SEEN: Mutex<Option<HashSet<u64>>> = Mutex::new(None);
                                let mut g = SEEN.lock();
                                let seen = g.get_or_insert_with(HashSet::new);
                                if seen.len() >= 40 {
                                    SEEN_FULL.store(true, Ordering::Relaxed);
                                }
                                if seen.len() < 40 && seen.insert(p.addr) {
                                    let what = match &found.resources[i] {
                                        Resolved::Buffer(v) => format!("V# stride {} records {}", v.stride, v.num_records),
                                        Resolved::Global(_, n) => format!("pointer ({n:?} bytes)"),
                                        _ => String::new(),
                                    };
                                    crate::info!("gpu[{}]: front end 2: {:?} shader {:#x} writes {base:#x}+{size:#x}: {what}; atomic {}", self.name, p.stage, p.addr, found.tr.plan.resources[i].atomic);
                                }
                            }
                            self.fe2_place(base, size, found.tr.plan.resources[i].written)
                        } else if read_only && super::resident::enabled() {
                            self.resident_bind(base, size)
                        } else {
                            None
                        };
                        // Held or parallel recording stopped at this command
                        // (no batch could be begun for its uploads): it is
                        // recorded again from here.
                        if self.stopped && super::fe2::on() {
                            return Err(STOPPED.into());
                        }
                        let home = from_mirror.or_else(|| {
                            self.home_bind(base, size, found.tr.plan.resources[i].written, if sized { HomeUse::Sized } else { HomeUse::Unbounded })
                                .map(|h| (h.0, h.1, h.2))
                        });
                        // A small window GPU work writes, kept in video memory
                        // (`gpu::keep`): bound like a home.
                        let home = match home {
                            Some(h) => Some(h),
                            None if super::keep::any() || super::tune::keep_small() => {
                                let res = &found.tr.plan.resources[i];
                                let create = allow_shadow && shadow_writes() && res.written && (res.atomic || size >= shadow_min());
                                let k = self.keep_bind(base, size, res.written, !sized, create)?;
                                if self.stopped {
                                    return Err(STOPPED.into());
                                }
                                k
                            }
                            None => None,
                        };
                        // Resident GPU-written data (`gpu::rgm`): sized bindings
                        // on the working copy W.
                        let mut from_rgm = false;
                        let home = match home {
                            Some(h) => Some(h),
                            None if sized && from_mirror.is_none() && super::rgm::get().is_some() => {
                                let h = self.rgm_bind(base, size, found.tr.plan.resources[i].written);
                                from_rgm = h.is_some();
                                h
                            }
                            None => None,
                        };
                        // Otherwise guest memory is used where this batch's W
                        // writes may be: they go back to it first.
                        if home.is_none() && !self.rgm_unmerged.is_empty() {
                            if !sized {
                                self.rgm_before_in_place(&[(base, size)])?;
                            } else if found.tr.plan.resources[i].written {
                                self.rgm_before_in_place(&[(base, size)])?;
                            } else {
                                self.rgm_before_in_place(&[(base, size)])?;
                            }
                        }
                        BindClock::mark(&mut clk, 1);
                        // Inside a resident shadow: its working copy is newer.
                        let resident = if read_only && home.is_none() { self.resident_read(base, size)? } else { None };
                        BindClock::mark(&mut clk, 2);
                        // Small read-only buffers: a snapshot in video memory.
                        // Data-only loads read on the GPU (`gpu_loads`) of data
                        // no GPU work still writes: as memory holds it now, as
                        // the CPU-resolved loads read it (the shader would read
                        // it when the batch executes, after CP and title writes
                        // that came later: `tune race_checks`, FINDINGS.md
                        // 2026-10-14); data GPU work writes is read where it is.
                        let data_load = read_only && matches!(found.resources[i], Resolved::Global(_, Some(_)));
                        let snap = if !(read_only && resident.is_none() && home.is_none()) {
                            None
                        } else if data_load && super::tune::data_snapshots() {
                            self.snapshot_exact(base, size)
                        } else {
                            self.snapshot(base, size, super::tune::snap_max())
                        };
                        BindClock::mark(&mut clk, 3);
                        // Read-only bindings of buffers GPU work wrote: VRAM mirrors.
                        let mirrored = if read_only && snap.is_none() && home.is_none() { resident.or_else(|| super::mirror::lookup(base, size)) } else { None };
                        BindClock::mark(&mut clk, 4);
                        // Other read-only buffers: a copy made at the batch's start
                        // (dispatches only with `tune pre_cs`).
                        let pre = if read_only && snap.is_none() && mirrored.is_none() && home.is_none() && (!allow_shadow || super::tune::pre_cs()) { self.pre_copy(base, size) } else { None };
                        if pre.is_some() {
                            placement = Placement::PreCopy;
                        }
                        BindClock::mark(&mut clk, 5);
                        // Diagnosis: ranges read when the batch executes.
                        if read_only && super::tune::race_checks() && size <= 1 << 28 {
                            self.note_open_read(p.addr, base, size);
                        }
                        if read_only && (snap.is_none() || pre.is_some()) && size <= super::tune::race_max() && size % 4 == 0 && super::tune::race_checks() && !self.pending.overlaps(base, size) && super::cp::guest_readable(base, size) {
                            let r = [(base, size)];
                            let mut why = if CP_WRITES.latest(&r).is_some() { 1 } else { 0 } | if INFLIGHT.latest(&r).is_some() { 2 } else { 0 } | if data_load { 0 } else { 4 };
                            why |= 16 << if home.is_some() { 3 } else if pre.is_some() { 2 } else if mirrored.is_some() { 1 } else { 0 };
                            let (words, hash) = if size <= 4096 {
                                ((0..size / 4).map(|j| unsafe { std::ptr::read_volatile((base + 4 * j) as *const u32) }).collect(), 0)
                            } else {
                                (Vec::new(), race_hash(base, size))
                            };
                            let cpw = super::cp::CPW_SEQ.load(Ordering::Acquire);
                            self.race_pending.push(RaceCheck { seq: 0, shader: p.addr, base, words, len: size, hash, queue: self.race_name.clone(), flip: super::FRAMES.load(Ordering::Relaxed), why, cpw, cpw_submit: 0 });
                        }
                        let snap = snap.or(pre);
                        let mut span = if size > 0 && mirrored.is_none() && snap.is_none() && home.is_none() { span(base, size) } else { None };
                        if span.is_none() && size > 0 && mirrored.is_none() && snap.is_none() && home.is_none() {
                            // A descriptor reaching past the end of its mappings
                            // (generic sizes on ring allocations): the mapped part
                            // is bound, the rest reads 0.
                            if let Some((_, end, _)) = mapping_run(base, base & !(WINDOW - 1), (base + size).next_multiple_of(WINDOW)) {
                                if end > base && end < base + size {
                                    size = end - base;
                                    span = self::span(base, size);
                                }
                            }
                        }
                        BindClock::mark(&mut clk, 6);
                        if super::tune::race_checks() && size > 0 && snap.is_none() && mirrored.is_none() && (home.is_some() || span.is_some()) {
                            let unbounded = matches!(found.resources[i], Resolved::Global(_, None));
                            let kind = if found.tr.plan.resources[i].written { 1 } else { 0 } | if data_load { 2 } else { 0 } | if unbounded { 4 } else { 0 };
                            bound.in_place.push((base, size, kind));
                        }
                        if let Some((hbuf, hoff, avail)) = home {
                            placement = if from_mirror.is_some() { Placement::Mirrored } else { Placement::Home };
                            if from_rgm {
                                RGM_PLACED.fetch_add(1, Ordering::Relaxed);
                            }
                            // Mirror placements start at a mark word (`Spec::write_marks`).
                            let align = if from_mirror.is_some() { 128 } else { ctx().limits.min_storage_buffer_offset_alignment.max(4) };
                            let off = hoff & !(align - 1);
                            bias = hoff - off;
                            info = vk::DescriptorBufferInfo { buffer: hbuf, offset: off, range: (bias + size).next_multiple_of(4).min(bias + avail) };
                            bound_size = size;
                            if found.tr.plan.resources[i].written {
                                trace_writes(|| format!("{:?} shader {:#x}{}", p.stage, p.addr, super::shaders::program_hash(p.addr).ok().and_then(super::particles::origin).map(|d| format!(" (from {d})")).unwrap_or_default()), base, size);
                                bound.writes.push((base, size));
                                if super::xwait::on() {
                                    super::xwait::note_write(&self.name, base, size);
                                }
                                if matches!(found.resources[i], Resolved::Buffer(_)) {
                                    bound.sized_writes.push((base, size));
                                }
                            }
                        } else if let Some((sbuf, soff, sbias)) = snap {
                            if placement != Placement::PreCopy {
                                placement = Placement::Snapshot;
                            }
                            bias = sbias;
                            info = vk::DescriptorBufferInfo { buffer: sbuf, offset: soff, range: (sbias + size).next_multiple_of(4) };
                            bound_size = size;
                        } else if let Some((mbuf, moff, avail)) = mirrored {
                            placement = if resident.is_some() { Placement::Resident } else { Placement::Mirror };
                            let align = ctx().limits.min_storage_buffer_offset_alignment.max(4);
                            let off = moff & !(align - 1);
                            bias = moff - off;
                            let range = (bias + size).next_multiple_of(4).min(bias + avail);
                            info = vk::DescriptorBufferInfo { buffer: mbuf, offset: off, range };
                            bound_size = size;
                            if let Some(s) = CP_WRITES.latest(&[(base, size)]) {
                                self.submit_after = self.submit_after.max(s);
                            }
                        } else if size > 0 {
                            match span {
                                Some((buffer, start, end)) => {
                                    placement = Placement::Host;
                                    if super::resident::enabled() {
                                        note_in_place(found.tr.plan.resources[i].written, matches!(found.resources[i], Resolved::Global(_, None)), size, p.addr);
                                    }
                                    let rel = base - start;
                                    let align = ctx().limits.min_storage_buffer_offset_alignment.max(4);
                                    let off = rel & !(align - 1);
                                    bias = rel - off;
                                    let range = (bias + size).next_multiple_of(4).min(end - start - off);
                                    info = vk::DescriptorBufferInfo { buffer, offset: off, range };
                                    bound_size = size;
                                    // A pending CP write into this buffer is
                                    // performed before the batch is submitted.
                                    if let Some(s) = CP_WRITES.latest(&[(base, size)]) {
                                        self.submit_after = self.submit_after.max(s);
                                    }
                                    if found.tr.plan.resources[i].written {
                                        trace_writes(|| format!("{:?} shader {:#x}{}", p.stage, p.addr, super::shaders::program_hash(p.addr).ok().and_then(super::particles::origin).map(|d| format!(" (from {d})")).unwrap_or_default()), base, size);
                                        bound.writes.push((base, size));
                                        if super::xwait::on() {
                                            super::xwait::note_write(&self.name, base, size);
                                        }
                                        if matches!(found.resources[i], Resolved::Buffer(_)) {
                                            bound.sized_writes.push((base, size));
                                        }
                                    }
                                    BindClock::mark(&mut clk, 7);
                                    // Written buffers: a device-local copy for this dispatch
                                    // (GPU atomics and stores to host memory are slow).
                                    let s_off = self.shadow_used.next_multiple_of(256);
                                    let span = range.next_multiple_of(256);
                                    let res = &found.tr.plan.resources[i];
                                    let all_scratch = res.written && range <= SCRATCH_BYTES && super::tune::scratch_all();
                                    let want = !all_scratch && allow_shadow && shadow_writes() && res.written && range <= super::tune::shadow_max() && (res.atomic || range >= shadow_min());
                                    let window = (base - bias, range);
                                    let reuse = if want { self.resident.iter().position(|r| r.host == buffer && r.host_off == off && r.range == range) } else { None };
                                    if reuse.is_none() && self.resident.iter().any(|r| r.guest < window.0 + window.1 && window.0 < r.guest + r.range) {
                                        // Guest memory gets the resident shadows' data first.
                                        EARLY_MERGES.fetch_add(1, Ordering::Relaxed);
                                        self.flush_resident(Some(&[window]))?;
                                    }
                                    if want && reuse.is_none() && s_off + 2 * span > seg {
                                        if self.shadow_room(s_off + 2 * span) {
                                            seg = self.slots[self.cur].shadow_cap;
                                        } else {
                                            SHADOW_FULL.fetch_add(1, Ordering::Relaxed);
                                        }
                                    }
                                    if let Some(k) = reuse {
                                        // Its write-back covers this resource's bytes too.
                                        let r = &mut self.resident[k];
                                        let (lo, hi) = (r.bias.min(bias), (r.bias + r.size).max(bias + size));
                                        (r.bias, r.size) = (lo, hi - lo);
                                        let r = *r;
                                        bound.shadows.push(Shadow { reused: true, ..r });
                                        placement = Placement::ShadowReused;
                                        info = vk::DescriptorBufferInfo { buffer: self.shadow(), offset: r.shadow_off, range };
                                        SHADOWS_REUSED.fetch_add(1, Ordering::Relaxed);
                                    } else if want && s_off + 2 * span <= seg {
                                        // DS_TRACE_SHADOW=1: each shadowed range once per shader.
                                        static TRACE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                                        if *TRACE.get_or_init(|| std::env::var("DS_TRACE_SHADOW").ok().as_deref() == Some("1")) {
                                            static SEEN: std::sync::Mutex<Option<HashSet<(u64, u64)>>> = std::sync::Mutex::new(None);
                                            if SEEN.lock().unwrap().get_or_insert_with(HashSet::new).insert((p.addr, base)) {
                                                let global = matches!(found.resources[i], Resolved::Global(..));
                                                crate::info!("shadow: {:?} shader {:#x} writes {base:#x}+{size:#x} (bound {range:#x}{})", p.stage, p.addr, if global { ", global pointer" } else { "" });
                                            }
                                        }
                                        self.shadow_used = s_off + 2 * span;
                                        bound.shadows.push(Shadow { host: buffer, guest: base - bias, reused: false, host_off: off, range, shadow_off: s_off, pristine_off: s_off + span, bias, size, since: None, tainted: false, writer: p.addr });
                                        placement = Placement::Shadow;
                                        info = vk::DescriptorBufferInfo { buffer: self.shadow(), offset: s_off, range };
                                        SHADOWED.fetch_add(1, Ordering::Relaxed);
                                    } else if all_scratch || (res.written && range > super::tune::shadow_max() && range <= SCRATCH_BYTES && super::tune::scratch_writes()) {
                                        if let Some(sb) = scratch_buffer() {
                                            placement = Placement::Scratch;
                                            info = vk::DescriptorBufferInfo { buffer: sb, offset: 0, range };
                                        }
                                    }
                                    if res.written && matches!(found.resources[i], Resolved::Buffer(_)) {
                                        // Within the bound range (clamped to the mapping).
                                        bound.write_src.push((base, size.min(range - bias), info.buffer, info.offset + bias));
                                    }
                                    BindClock::mark(&mut clk, 8);
                                }
                                None => {
                                    // Memory the driver cannot import (e.g. the
                                    // executable's read-only pages, which hold some
                                    // shaders' constants): small read-only ranges
                                    // are copied by the CPU instead of read as zeros.
                                    let snap = if read_only { self.snapshot(base, size, UNIMPORTABLE_SNAP_MAX) } else { None };
                                    match snap {
                                        Some((sbuf, soff, sbias)) => {
                                            placement = Placement::Snapshot;
                                            bias = sbias;
                                            info = vk::DescriptorBufferInfo { buffer: sbuf, offset: soff, range: (sbias + size).next_multiple_of(4) };
                                            bound_size = size;
                                            UNIMPORTABLE_SNAPS.fetch_add(1, Ordering::Relaxed);
                                        }
                                        None => {
                                            let why = unmapped_reason(base, size);
                                            self.report(format!("{:?} shader {:#x}: buffer {base:#x}+{size:#x} is not mapped GPU memory (bound as zeros): {why}", p.stage, p.addr));
                                        }
                                    }
                                }
                            }
                        }
                        if profiling() {
                            bound.placements.push((placement, bound_size, found.tr.plan.resources[i].written));
                        }
                        trace_bind(&self.name, p.addr, base, size, found.tr.plan.resources[i].written, matches!(found.resources[i], Resolved::Global(_, None)), placement);
                        if super::smalltrack::on() {
                            let k = PLACEMENTS.iter().position(|&x| x == placement).unwrap_or(7);
                            super::smalltrack::bound(base, size, found.tr.plan.resources[i].written, k, &self.name, p.addr, matches!(found.resources[i], Resolved::Global(_, None)));
                        }
                        // DS_TRACE_PLACE=<hex shader>: its bindings' placements, the
                        // first 64 from flip DS_TRACE_PLACE_FROM on (diagnosis: what a
                        // program reads across PCIe).
                        static TRACE_PLACE: std::sync::OnceLock<(u64, u64)> = std::sync::OnceLock::new();
                        let (tp, tp_from) = *TRACE_PLACE.get_or_init(|| {
                            let a = std::env::var("DS_TRACE_PLACE").ok().and_then(|v| u64::from_str_radix(v.trim().trim_start_matches("0x"), 16).ok()).unwrap_or(0);
                            (a, std::env::var("DS_TRACE_PLACE_FROM").ok().and_then(|v| v.parse().ok()).unwrap_or(0))
                        });
                        if tp != 0 && tp == p.addr && super::FRAMES.load(Ordering::Relaxed) >= tp_from {
                            static N: AtomicU64 = AtomicU64::new(0);
                            if N.fetch_add(1, Ordering::Relaxed) < 64 {
                                let what = match &found.resources[i] {
                                    Resolved::Buffer(v) => format!("V# stride {} records {}", v.stride, v.num_records),
                                    Resolved::Global(_, n) => format!("pointer ({n:?} bytes)"),
                                    _ => String::new(),
                                };
                                crate::info!("place: {:?} shader {:#x} resource {i} {base:#x}+{size:#x} ({what}){}: {placement:?}", p.stage, p.addr, if found.tr.plan.resources[i].written { " written" } else { "" });
                            }
                        }
                        let ri = res_info + RES_INFO_DWORDS as usize * i;
                        p.sgpr_data[ri] = bias as u32;
                        p.sgpr_data[ri + 1] = stride;
                        p.sgpr_data[ri + 2] = bound_size.min(u32::MAX as u64) as u32;
                        p.sgpr_data[ri + 3] = limit;
                        // Mirror mode: the resource's first word in the written-bit
                        // map (its stores mark the dwords they write), or none.
                        if has_marks {
                            let ai = addr_at + (rdna::translate::ADDR_HEADER_DWORDS + rdna::translate::ADDR_INFO_DWORDS * i as u32) as usize;
                            p.sgpr_data[ai + 3] = if placement == Placement::Mirrored && found.tr.plan.resources[i].written { (info.offset / 128) as u32 } else { u32::MAX };
                        }
                        if matches!(b.kind, BindingKind::Address(_)) {
                            // The descriptor's range as an address and size.
                            let va = ctx().buffer_address(info.buffer) + info.offset;
                            let ai = addr_at + (rdna::translate::ADDR_HEADER_DWORDS + rdna::translate::ADDR_INFO_DWORDS * i as u32) as usize;
                            p.sgpr_data[ai] = va as u32;
                            p.sgpr_data[ai + 1] = (va >> 32) as u32;
                            p.sgpr_data[ai + 2] = info.range.min(u32::MAX as u64) as u32;
                            ADDRESSED.fetch_add(1, Ordering::Relaxed);
                        } else {
                            bound.bufs.push((b.binding, info));
                        }
                        BindClock::mark(&mut clk, 9);
                        BindClock::count_buffer(&mut clk);
                    }
                    BindingKind::Gds => {
                        bound.gds = true;
                        bound.bufs.push((b.binding, vk::DescriptorBufferInfo { buffer: fx.gds, offset: 0, range: rdna::translate::GDS_BYTES as u64 }))
                    }
                    BindingKind::Debug => bound.bufs.push((b.binding, vk::DescriptorBufferInfo { buffer: fx.debug, offset: 0, range: rdna::translate::DUMP_DWORDS as u64 * 4 })),
                    BindingKind::Watchdog => bound.bufs.push((b.binding, vk::DescriptorBufferInfo { buffer: fx.watchdog, offset: 0, range: rdna::translate::WATCHDOG_DWORDS as u64 * 4 })),
                    BindingKind::LodStats => bound.bufs.push((b.binding, vk::DescriptorBufferInfo { buffer: fx.lodstats, offset: 0, range: super::lodstats::SLOT_BYTES })),
                    BindingKind::Marks => bound.bufs.push((
                        b.binding,
                        match super::resident::marks() {
                            Some(m) => vk::DescriptorBufferInfo { buffer: m, offset: 0, range: super::resident::MARK_BYTES },
                            None => vk::DescriptorBufferInfo { buffer: fx.dummy, offset: 0, range: DUMMY_BYTES },
                        },
                    )),
                    _ => {}
                }
            }
        }
        let frame = super::FRAMES.load(Ordering::Relaxed);
        // Sampled images and samplers from this thread's memos when every
        // one of them is there (no lock); else one image-cache lock for all.
        // Into the recycled vectors (only images and samplers fill them).
        let memo = {
            let mut all = true;
            for b in &found.tr.out.bindings {
                match (b.kind, b.kind_resource().and_then(|i| found.resources.get(i))) {
                    (BindingKind::Image(_), Some(Resolved::Image(t))) => match super::images::view_hit(t, frame) {
                        Some((view, image)) => {
                            bound.imgs.push((b.binding, vk::DescriptorType::SAMPLED_IMAGE, vk::DescriptorImageInfo { sampler: vk::Sampler::null(), image_view: view, image_layout: vk::ImageLayout::GENERAL }));
                            bound.images.push(image);
                            if super::images::vk_format(t.format) == Some(vk::Format::R8_UINT) {
                                bound.stencil_views.push(image);
                            }
                        }
                        None => {
                            all = false;
                            break;
                        }
                    },
                    (BindingKind::Sampler(_) | BindingKind::CompareSampler(_), Some(Resolved::Sampler(sm))) => match super::images::sampler_hit(*sm, matches!(b.kind, BindingKind::CompareSampler(_))) {
                        Some(smp) => bound.imgs.push((b.binding, vk::DescriptorType::SAMPLER, vk::DescriptorImageInfo { sampler: smp, image_view: vk::ImageView::null(), image_layout: vk::ImageLayout::UNDEFINED })),
                        None => {
                            all = false;
                            break;
                        }
                    },
                    (BindingKind::StorageImage(_) | BindingKind::StorageImageMips(_) | BindingKind::ImageArray(_), _) => {
                        all = false;
                        break;
                    }
                    _ => {}
                }
            }
            all
        };
        if memo {
            return Ok(bound);
        }
        // Frame-ahead recording: a storage image's binding marks it written
        // in the image cache, which the current frame's work would see.
        // A long run's chunk binds images the GPU owns already (which binding
        // leaves as they are).
        if self.held && found.tr.out.bindings.iter().any(|b| matches!(b.kind, BindingKind::StorageImage(_) | BindingKind::StorageImageMips(_))) {
            let owned = self.long
                && super::images::with(|im| {
                    found.tr.out.bindings.iter().all(|b| match (b.kind, b.kind_resource().and_then(|i| found.resources.get(i))) {
                        (BindingKind::StorageImage(_) | BindingKind::StorageImageMips(_), Some(Resolved::Image(t))) => im.storage_owned(t),
                        (BindingKind::StorageImage(_) | BindingKind::StorageImageMips(_), _) => false,
                        _ => true,
                    })
                });
            if !owned {
                self.stop("storage image");
                return Err(STOPPED.into());
            }
        }
        bound.imgs.clear();
        bound.images.clear();
        bound.stored.clear();
        bound.stencil_views.clear();
        // One image-cache lock for all the stage's images and samplers.
        let mut failed: Vec<String> = Vec::new();
        let r: Result<(), String> = super::images::with(|im| {
            for b in &found.tr.out.bindings {
                match (b.kind, b.kind_resource().and_then(|i| found.resources.get(i))) {
                    (BindingKind::Image(_) | BindingKind::StorageImage(_), Some(Resolved::Image(t))) => {
                        let storage = matches!(b.kind, BindingKind::StorageImage(_));
                        let (view, image) = im.view(t, storage, frame)?;
                        let ty = if storage { vk::DescriptorType::STORAGE_IMAGE } else { vk::DescriptorType::SAMPLED_IMAGE };
                        bound.imgs.push((b.binding, ty, vk::DescriptorImageInfo { sampler: vk::Sampler::null(), image_view: view, image_layout: vk::ImageLayout::GENERAL }));
                        bound.images.push(image);
                        if storage {
                            bound.stored.push(image);
                        } else if super::images::vk_format(t.format) == Some(vk::Format::R8_UINT) {
                            bound.stencil_views.push(image);
                        }
                    }
                    (BindingKind::Sampler(_) | BindingKind::CompareSampler(_), Some(Resolved::Sampler(sm))) => {
                        let compare = matches!(b.kind, BindingKind::CompareSampler(_));
                        let smp = im.sampler(*sm, compare)?;
                        bound.imgs.push((b.binding, vk::DescriptorType::SAMPLER, vk::DescriptorImageInfo { sampler: smp, image_view: vk::ImageView::null(), image_layout: vk::ImageLayout::UNDEFINED }));
                    }
                    (BindingKind::ImageArray(_), Some(Resolved::ImageTable(entries))) => {
                        let dim = entries.iter().flatten().find_map(|t| rdna::translate::ImageDim::from_type(t.ty, t.depth)).ok_or("image table without entries")?;
                        let mut infos = Vec::with_capacity(rdna::translate::IMAGE_TABLE_LEN as usize);
                        for k in 0..rdna::translate::IMAGE_TABLE_LEN as usize {
                            let view = match entries.get(k).copied().flatten() {
                                Some(t) => match im.view(&t, false, frame) {
                                    Ok((view, image)) => {
                                        bound.images.push(image);
                                        Some(view)
                                    }
                                    Err(e) => {
                                        failed.push(format!("image table entry {:#x}: {e}", t.base));
                                        None
                                    }
                                },
                                None => None,
                            };
                            let view = match view {
                                Some(v) => v,
                                None => im.dummy_view(dim)?,
                            };
                            infos.push(vk::DescriptorImageInfo { sampler: vk::Sampler::null(), image_view: view, image_layout: vk::ImageLayout::GENERAL });
                        }
                        bound.img_arrays.push((b.binding, vk::DescriptorType::SAMPLED_IMAGE, infos));
                    }
                    (BindingKind::StorageImageMips(_), Some(Resolved::Image(t))) => {
                        // One view per mip level from the T#'s base level;
                        // those past its last level repeat the last.
                        let n = (t.last_level.saturating_sub(t.base_level) + 1).min(rdna::translate::MIP_VIEWS);
                        let mut infos = Vec::with_capacity(rdna::translate::MIP_VIEWS as usize);
                        let mut image = vk::Image::null();
                        for k in 0..n {
                            let mut tk = *t;
                            tk.base_level = t.base_level + k;
                            tk.raw[3] = (tk.raw[3] & !(0xf << 12)) | (tk.base_level << 12);
                            let (view, img) = im.view(&tk, true, frame)?;
                            image = img;
                            infos.push(vk::DescriptorImageInfo { sampler: vk::Sampler::null(), image_view: view, image_layout: vk::ImageLayout::GENERAL });
                        }
                        while infos.len() < rdna::translate::MIP_VIEWS as usize {
                            infos.push(infos[infos.len() - 1]);
                        }
                        MIP_VIEW_BINDINGS.fetch_add(1, Ordering::Relaxed);
                        // Each shader and image once: which programs write
                        // several levels per dispatch.
                        static SEEN: Mutex<Option<HashSet<(u64, u64)>>> = Mutex::new(None);
                        let sh = super::images::BINDING_SHADER.with(|s| s.get());
                        if SEEN.lock().get_or_insert_with(HashSet::new).insert((sh, t.base)) {
                            crate::info!("images: shader {sh:#x} binds {:#x} ({}x{} format {}) one view per mip level, levels {}..{}", t.base, t.width, t.height, t.format, t.base_level, t.last_level);
                        }
                        bound.images.push(image);
                        bound.stored.push(image);
                        bound.img_arrays.push((b.binding, vk::DescriptorType::STORAGE_IMAGE, infos));
                    }
                    (BindingKind::Image(_) | BindingKind::StorageImage(_) | BindingKind::StorageImageMips(_) | BindingKind::ImageArray(_) | BindingKind::Sampler(_) | BindingKind::CompareSampler(_), _) => return Err("binding/resource mismatch".into()),
                    _ => {}
                }
            }
            Ok(())
        });
        for f in failed {
            self.report(f);
        }
        r?;
        Ok(bound)
    }

    /// Copies a stage's SGPR data into the ring (space reserved) and binds it;
    /// where it went (ring buffer, offset).
    fn bind_sgpr(&mut self, p: &Prepared, bound: &mut Bound) -> Result<(vk::Buffer, u64), String> {
        // Little-endian dwords, as the shader reads them.
        let bytes: &[u8] = unsafe { std::slice::from_raw_parts(p.sgpr_data.as_ptr() as *const u8, 4 * p.sgpr_data.len()) };
        let vram = match super::tune::ring_vram() {
            1 => true,
            2 => self.name == "gfx",
            3 => self.name != "gfx",
            4 => super::tune::ring_range().contains(&p.addr),
            _ => false,
        };
        if vram && self.slots[self.cur].rings[1].is_none() && !self.vram_ring_failed {
            match ctx().bar_buffer(RING_SEG, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDIRECT_BUFFER) {
                Ok((b, _m, p)) => {
                    SLOT_MEM[4].fetch_add(RING_SEG, Ordering::Relaxed);
                    self.slots[self.cur].rings[1] = Some((b, p))
                }
                Err(e) => {
                    crate::warn!("gpu[{}]: no SGPR data ring in video memory ({e})", self.name);
                    self.vram_ring_failed = true;
                }
            }
        }
        let which = if vram && self.slots[self.cur].rings[1].is_some() { 1 } else { 0 };
        let Some((ring, ring_ptr)) = self.slots[self.cur].rings[which] else { return Err("no SGPR data ring".into()) };
        // A uniform buffer (`Spec::sgpr_ubo`) is bound at its alignment, with
        // the translation's whole uvec4 array (16-byte rounding).
        let ubo = p.tr.out.bindings.iter().any(|b| b.kind == BindingKind::SgprUniform);
        let pc = p.tr.out.sgpr_pc;
        // A batched translation's uniform window of the ring (`Spec::sgpr_ubo`
        // 3): its data placed as for a storage buffer.
        let window = ubo && pc.is_some();
        let len = bytes.len() as u64;
        let off = if ubo && !window { self.ring_alloc_at(which, len.next_multiple_of(16), self.ubo_align) } else { self.ring_alloc(which, len) }.ok_or("SGPR data ring exhausted")?;
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), ring_ptr.add(off as usize), bytes.len()) };
        // With `Spec::sgpr_pc` the whole ring, the stage adding its offset
        // (the binding then stays the same for the batch).
        let (at, range) = if pc.is_some() { (0, RING_SEG) } else { (off, len) };
        // The open batched run's window when this command's data lies inside
        // it (its descriptors then match the run's), else one from here on.
        let wb = match &self.cs_run {
            Some(r) if window && r.batch == self.batch_count && r.ring == ring && r.window <= off && off + len <= r.window + SGPR_WINDOW_BYTES => r.window,
            _ if window => self.window_at(off),
            _ => 0,
        };
        for b in &p.tr.out.bindings {
            if b.kind == BindingKind::SgprData {
                bound.bufs.push((b.binding, vk::DescriptorBufferInfo { buffer: ring, offset: at, range }));
            } else if b.kind == BindingKind::SgprUniform {
                let (o, r) = if window { (wb, SGPR_WINDOW_BYTES) } else { (off, range.next_multiple_of(16)) };
                bound.bufs.push((b.binding, vk::DescriptorBufferInfo { buffer: ring, offset: o, range: r }));
                bound.ubo = Some(b.binding);
            }
        }
        if window {
            bound.window = Some(wb);
        }
        bound.sgpr_base = pc.map(|k| (k, (off / 4) as u32));
        Ok((ring, off))
    }

    /// A uniform window of the ring that holds offset `off` (`Bound::window`):
    /// from `off` down to the uniform alignment, inside the ring segment.
    fn window_at(&self, off: u64) -> u64 {
        let a = self.ubo_align;
        (off / a * a).min((RING_SEG - SGPR_WINDOW_BYTES) / a * a)
    }

    /// Moves a command's uniform window (`Bound::window`) to `wb`.
    fn move_window(bound: &mut Bound, wb: u64) {
        if let (Some(_), Some(u)) = (bound.window, bound.ubo) {
            for (b, info) in bound.bufs.iter_mut() {
                if *b == u {
                    info.offset = wb;
                }
            }
            bound.window = Some(wb);
        }
    }

    fn alloc_set(&mut self, layout: vk::DescriptorSetLayout) -> Result<vk::DescriptorSet, String> {
        let layouts = [layout];
        let ai = vk::DescriptorSetAllocateInfo::default().descriptor_pool(self.slots[self.cur].desc_pool).set_layouts(&layouts);
        unsafe { ctx().device.allocate_descriptor_sets(&ai) }.map(|v| v[0]).map_err(|e| format!("descriptor set: {e}"))
    }

    fn write_set(&self, set: vk::DescriptorSet, bounds: &[Bound]) {
        let writes = descriptor_writes(set, bounds);
        unsafe { ctx().device.update_descriptor_sets(&writes, &[]) };
    }

    /// Executes a dispatch of `groups` thread groups with the compute state of `cp`.
    pub fn dispatch(&mut self, cp: &Cp, groups: [u32; 3], initiator: u32) {
        if DEVICE_LOST.load(Ordering::Relaxed) {
            SKIPPED.fetch_add(1, Ordering::Relaxed);
            return;
        }
        if skip_cs(cp) || super::tune::skip_cmds() & 2 != 0 {
            SKIPPED.fetch_add(1, Ordering::Relaxed);
            return;
        }
        let t_prog = laps().then(std::time::Instant::now);
        if super::tune::gpu_chains() != 0 && self.gpu_append(cp, groups, initiator) {
            if let Some(t) = t_prog {
                prog_note(self.prog_class(), cs_addr(cp), 0, t.elapsed().as_nanos() as u64);
            }
            return;
        }
        let r = self.dispatch_inner(cp, groups, initiator, None);
        if let Some(t) = t_prog {
            prog_note(self.prog_class(), cs_addr(cp), 0, t.elapsed().as_nanos() as u64);
        }
        if let Err(e) = r {
            if e != STOPPED {
                SKIPPED.fetch_add(1, Ordering::Relaxed);
                self.report(e);
            }
        } else if sync_mode() || (!self.deferred && sync_cs(cs_addr(cp))) {
            self.flush();
            if DEVICE_LOST.load(Ordering::Relaxed) {
                let cs = ((cp.reg(Space::Sh, 0x20C) as u64) | ((cp.reg(Space::Sh, 0x20D) as u64 & 0xFF) << 32)) << 8;
                crate::error!("gpu[{}]: device lost by the dispatch of cs {cs:#x} ({}x{}x{} groups)", self.name, groups[0], groups[1], groups[2]);
            }
        }
        self.split_if_gpu_idle();
    }

    /// Diagnosis: the reads of a GPU-resolved command's shader-resolved
    /// loads and buffers that the CPU path would have bound elsewhere than
    /// guest memory (`GPU_RES_CHECKS`; the first 40 logged).
    fn gpu_res_check(&mut self, p: &Prepared) {
        let mut ranges: Vec<(String, u64, u64)> = Vec::new();
        for (i, r) in p.resources.iter().enumerate().take(64) {
            if p.tr.spec.gpu_res & (1u64 << i) == 0 {
                continue;
            }
            if let Resolved::Buffer(v) = r {
                ranges.push((format!("buffer {i} (stride {} records {} format {})", v.stride, v.num_records, v.format), v.base, v.extent().max(4)));
            }
        }
        for (what, a, n) in shaders::read_sources(p) {
            if let Some(k) = what.strip_prefix("load ").and_then(|x| x.split(' ').next()).and_then(|x| x.parse::<usize>().ok()) {
                if k < 64 && p.tr.spec.gpu_loads & (1u64 << k) != 0 {
                    ranges.push((what, a, n));
                }
            }
        }
        for (what, a, n) in ranges {
            GPU_RES_CHECKS[0].fetch_add(1, Ordering::Relaxed);
            let r = [(a, n)];
            let class = if super::homes::overlaps(a, n) {
                1
            } else if self.overlaps_pending(&r) || INFLIGHT.latest(&r).is_some() || CP_WRITES.latest(&r).is_some() {
                2
            } else if self.resident.iter().any(|x| x.guest < a + n && a < x.guest + x.range) {
                3
            } else if span(a, n).is_none() {
                4
            } else {
                0
            };
            if class != 0 {
                let k = GPU_RES_CHECKS[class].fetch_add(1, Ordering::Relaxed);
                static LOGGED: AtomicU64 = AtomicU64::new(0);
                if k < 8 && LOGGED.fetch_add(1, Ordering::Relaxed) < 40 {
                    crate::info!("gpu[{}]: GPU-resolved cs {:#x}: {what} at {a:#x}+{n:#x} is {}", self.name, p.addr, ["", "in a video-memory home", "written by pending GPU or CP work", "in a resident shadow", "not imported"][class]);
                }
            }
        }
    }

    /// A direct dispatch of the open GPU-resolved run's program
    /// (`CsRun::gpu`) whose run roots hold the run's values: appended to the
    /// run's dispatch with its own user data, the rest of its SGPR data the
    /// run's first command's (no preparation, binding, syncs or notes: what
    /// it reads apart from the run's resources the shader resolves). False:
    /// the full path records it.
    fn gpu_append(&mut self, cp: &Cp, groups: [u32; 3], initiator: u32) -> bool {
        let Some(mut run) = self.cs_run.take() else { return false };
        let ok = self.gpu_append_to(&mut run, cp, groups, initiator);
        self.cs_run = Some(run);
        ok
    }

    fn gpu_append_to(&mut self, r: &mut CsRun, cp: &Cp, groups: [u32; 3], initiator: u32) -> bool {
        let Some(g) = r.gpu.as_ref() else { return false };
        if cs_addr(cp) != g.addr {
            return false;
        }
        if r.batch != self.batch_count || r.uses != self.cmd_uses.get() || !self.recording || self.held || initiator & (1 << 5) != 0 || groups[1] != 1 || groups[2] != 1 || groups[0] == 0 || r.groups + groups[0] > BATCH_MAX_GROUPS || (0x204..0x207).any(|x| cp.reg(Space::Sh, x) != 0) || gpu_run_regs(cp) != g.regs {
            GPU_APPENDS[2].fetch_add(1, Ordering::Relaxed);
            return false;
        }
        let user: [u32; 16] = std::array::from_fn(|i| cp.reg(Space::Sh, 0x240 + i as u32));
        if g.roots.iter().any(|&(root, v)| super::gpures::root_value(root, &user) != v) {
            GPU_APPENDS[1].fetch_add(1, Ordering::Relaxed);
            return false;
        }
        if super::fe2::on() {
            // Front end 2 (stage F3): the run's bindings were checked for the
            // view its first command had; this command's per-item chains are
            // placed now.
            if self.fe2_want(super::precache::epoch() as u32) > g.fe2_want {
                F3_REFUSED[F3_VIEW].fetch_add(1, Ordering::Relaxed);
                GPU_APPENDS[2].fetch_add(1, Ordering::Relaxed);
                return false;
            }
            let (found, addr) = (g.found.clone(), g.addr);
            if self.fe2_gpu_ranges(&found, addr, &user).is_err() {
                GPU_APPENDS[2].fetch_add(1, Ordering::Relaxed);
                return false;
            }
        } else if !gpu_pages_mapped(&g.items, &user) || !gpu_res_imported(&g.found, g.addr, g.found.tr.spec.gpu_res, &user) {
            GPU_APPENDS[2].fetch_add(1, Ordering::Relaxed);
            return false;
        }
        let bytes = 4 * g.template.len() as u64;
        let which = g.which;
        let Some(off) = self.ring_alloc(which, bytes) else {
            GPU_APPENDS[2].fetch_add(1, Ordering::Relaxed);
            return false;
        };
        // Inside the run's uniform window, when it has one.
        if r.window != 0 && off + bytes > r.window + SGPR_WINDOW_BYTES {
            GPU_APPENDS[2].fetch_add(1, Ordering::Relaxed);
            return false;
        }
        let Some((_, rp)) = self.slots[self.cur].rings[which] else { return false };
        let g = r.gpu.as_ref().unwrap();
        unsafe {
            let dst = rp.add(off as usize) as *mut u32;
            std::ptr::copy_nonoverlapping(g.template.as_ptr(), dst, g.template.len());
            std::ptr::copy_nonoverlapping(user.as_ptr(), dst, g.users.min(16));
            let item = ((off - r.window) / 4) as u32;
            for k in 0..groups[0] {
                let e = r.table.add((rdna::translate::BATCH_ENTRY_DWORDS * (r.groups + k)) as usize);
                e.write(item);
                e.add(1).write(k);
            }
            r.groups += groups[0];
            r.args.write(r.groups);
        }
        GPU_APPENDS[0].fetch_add(1, Ordering::Relaxed);
        BATCHED[0].fetch_add(1, Ordering::Relaxed);
        EXECUTED.fetch_add(1, Ordering::Relaxed);
        self.batch_shaders.push(g.addr);
        true
    }

    /// The profile's dispatch kind (`CS_KINDS`) of a dispatch recorded here.
    fn cs_kind(&self, indirect: bool) -> usize {
        // Indirect dispatches whose arguments the CPU read profile as such.
        let indirect = indirect || CPU_INDIRECT.with(|c| c.get());
        indirect as usize
            + if self.name == "gpu-acb0x50" {
                6
            } else if self.alt_id != 0 {
                2
            } else if self.deferred {
                4
            } else {
                0
            }
    }

    /// An indirect dispatch whose arguments (12 bytes at `args`) GPU work
    /// still writes, with the GPU reading them (vkCmdDispatchIndirect): the
    /// command processor would otherwise submit, wait for that work and read
    /// them itself while the GPU ran dry (~10 times per frame in the passage).
    /// False: not taken (switch off, thread-dimension sizes, a dispatch base,
    /// self-checks, or arguments readable now); the caller reads them.
    pub fn dispatch_gpu_indirect(&mut self, cp: &Cp, args: u64, initiator: u32) -> bool {
        if DEVICE_LOST.load(Ordering::Relaxed) || !super::tune::gpu_indirect() || initiator & (1 << 5) != 0 || verify::enabled() {
            return false;
        }
        if [0x204, 0x205, 0x206].iter().any(|&r| cp.reg(Space::Sh, r) != 0) {
            return false;
        }
        let mut lap = Laps::for_dispatch(self.cs_kind(true), false);
        let r = [(args, 12)];
        // A compute queue's parallel worker cannot see writes of the run's
        // earlier chunks (not submitted yet, recorded by other threads;
        // flushes inside a run, `tune par_events`, order them), and the
        // run's planning does not wait for the arguments: the GPU reads
        // them, in order.
        // (So for a long parallel run's chunk, `long`.)
        if !(self.deferred && (self.compute_cp || self.long)) && !(self.overlaps_pending(&r) || INFLIGHT.latest(&r).is_some() || CP_WRITES.latest(&r).is_some() || super::keep::holds(args, 12)) {
            return false;
        }
        // Arguments in a video-memory home: read by the CP after the home's
        // GPU-written parts are written back.
        if super::homes::overlaps(args, 12) {
            return false;
        }
        if skip_cs(cp) || super::tune::skip_cmds() & 2 != 0 {
            SKIPPED.fetch_add(1, Ordering::Relaxed);
            return true;
        }
        lap.mark(17);
        GPU_INDIRECT.fetch_add(1, Ordering::Relaxed);
        let t_prog = laps().then(std::time::Instant::now);
        let r = self.dispatch_inner(cp, [1, 1, 1], initiator, Some(args));
        if let Some(t) = t_prog {
            prog_note(self.prog_class(), cs_addr(cp), 1, t.elapsed().as_nanos() as u64);
        }
        if let Err(e) = r {
            if e != STOPPED {
                SKIPPED.fetch_add(1, Ordering::Relaxed);
                self.report(e);
            }
        } else if sync_mode() {
            self.flush();
        }
        true
    }

    /// `indirect`: the arguments' guest address (the GPU reads the group
    /// counts; `groups` is unused).
    fn dispatch_inner(&mut self, cp: &Cp, mut groups: [u32; 3], initiator: u32, indirect: Option<u64>) -> Result<(), String> {
        // Phase times only when profiling (a clock read costs ~20 ns); the
        // total always (slow dispatches are reported).
        let t_start = std::time::Instant::now();
        self.fe2_min_want = u32::MAX;
        let t0 = profiling().then_some(t_start);
        let since = |t0: Option<std::time::Instant>| t0.map_or(std::time::Duration::ZERO, |t| t.elapsed());
        // No groups (direct, or arguments the CPU read): nothing runs, so
        // nothing is prepared or waited for (~100,000 in a boot to the gate,
        // ~7 per flip there).
        if indirect.is_none() && groups.contains(&0) {
            ZERO_GROUPS[CPU_INDIRECT.with(|c| c.get()) as usize].fetch_add(1, Ordering::Relaxed);
            return Ok(());
        }
        let kind = self.cs_kind(indirect.is_some());
        let first = self.batch_shaders.is_empty();
        let mut lap = Laps::for_dispatch(kind, first);
        // Runs of direct dispatches of one program make it batchable.
        if super::tune::batch_cs() && indirect.is_none() && groups.iter().map(|&g| g as u64).product::<u64>() <= BATCH_SMALL {
            let a = cs_addr(cp);
            if self.cs_streak.0 == a {
                self.cs_streak.1 += 1;
                if self.cs_streak.1 == BATCH_STREAK {
                    super::shaders::mark_batchable(a);
                }
            } else {
                self.cs_streak = (a, 1);
                // A run starting here makes the program batchable now: before
                // its first pipeline is made (else made twice, the second
                // time batched when the run's fourth dispatch comes).
                if !super::shaders::batchable(a) && self.cs_peeked.insert(a) && cp.streak_after(a, BATCH_SMALL, BATCH_STREAK - 1) + 1 >= BATCH_STREAK {
                    super::shaders::mark_batchable(a);
                    BATCHABLE_AHEAD.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
        if let Some(a) = indirect {
            // A resident shadow holding newer arguments is merged back first,
            // unless the GPU reads them from its working copy (`tune
            // shadow_args`; checked again where they are bound).
            if !(self.shadow_args() && self.resident_args(a).is_some()) {
                self.flush_resident(Some(&[(a, 12)]))?;
            }
            lap.mark(16);
        }
        let wave64 = initiator & (1 << 15) == 0;
        // Prepared ahead by the workers (`gpu::ahead`) when it still holds.
        let [pre, _] = cp.take_ahead();
        let mut p = match pre.and_then(|s| super::ahead::take(&s, cp, wave64)) {
            Some(p) => p,
            None => shaders::prepare_compute(cp, wave64)?,
        };
        if super::drawcensus::on() {
            super::drawcensus::observe_cs(&self.name, p.addr, &p.tr.plan, indirect.is_some());
        }
        if super::tune::gpu_chains() != 0 && p.tr.spec.batched && indirect.is_none() {
            let user = &p.sgpr_data[..p.tr.spec.user_sgprs as usize];
            if p.tr.spec.gpu_res != 0 && super::fe2::on() {
                // Front end 2 (stage F3): what its per-item chains read placed
                // in the mirror; else this command resolves them on the CPU.
                if self.fe2_gpu_ranges(&p.found, p.addr, user).is_err() {
                    F3_PLACED[3].fetch_add(1, Ordering::Relaxed);
                    p = shaders::without_gpu_chains(|| shaders::prepare_compute(cp, wave64))?;
                }
            } else if p.tr.spec.gpu_res != 0 {
                // A GPU-resolved command recorded in full (a run's first):
                // what its chains start from must be in the page table.
                if let Some(d) = super::gpures::decided(p.addr) {
                    if !gpu_pages_mapped(&d.item_roots, user) || !gpu_res_imported(&p.found, p.addr, p.tr.spec.gpu_res, user) {
                        return Err(format!("cs {:#x}: GPU-resolved chains reach memory that cannot be imported", p.addr));
                    }
                }
                self.gpu_res_check(&p);
            } else {
                let continues = self.cs_streak.0 == p.addr && self.cs_streak.1 >= 2;
                super::gpures::observe(p.addr, &p.tr.plan, user, continues);
            }
        }
        lap.mark(0);
        // Data GPU work still writes: read by the shader itself (`tune
        // gpu_loads 3`; data the title writes stays captured when recorded).
        if super::tune::gpu_loads_mode() == 3 && !p.data_loads.is_empty() && self.would_wait(&shaders::data_load_ranges(&p)) {
            shaders::note_gpu_loads_program(p.addr);
            p = shaders::with_gpu_loads(|| shaders::prepare_compute(cp, wave64))?;
            HYBRID_LOADS[2].fetch_add(1, Ordering::Relaxed);
        }
        self.sync_readers = [p.addr, 0];
        let patches = self.speculate(&[&p]);
        if patches.is_none() && self.sync_for_stage(&p.reads, SYNC_CS_READS) {
            if self.stopped {
                return Err(STOPPED.into());
            }
            if let Some(c) = self.sync_culprit.take() {
                explain_wait(&self.name, c, &[&p]);
            }
            READ_WAIT_WORDS[if shaders::words_unchanged(&p) { 2 } else { 3 }].fetch_add(1, Ordering::Relaxed);
            p = shaders::prepare_compute(cp, wave64)?;
        }
        cull_census(&p);
        lap.mark(1);
        // Textures are uploaded from guest memory: earlier GPU writes first.
        let mut textures = std::mem::take(&mut self.tex_scratch);
        textures.clear();
        texture_ranges_into(&p, &mut textures);
        self.sync_textures(&textures, SYNC_CS_TEXTURES);
        self.tex_scratch = textures;
        if self.stopped {
            return Err(STOPPED.into());
        }
        lap.mark(2);
        let local = [cp.reg(Space::Sh, 0x207) & 0xFFFF, cp.reg(Space::Sh, 0x208) & 0xFFFF, cp.reg(Space::Sh, 0x209) & 0xFFFF];
        // Threads along X (USE_THREAD_DIMENSIONS gives threads, not groups).
        let threads_x = if initiator & (1 << 5) != 0 { groups[0] as u64 } else { groups[0] as u64 * local[0].max(1) as u64 };
        if initiator & (1 << 5) != 0 {
            // USE_THREAD_DIMENSIONS: sizes are in threads.
            for k in 0..3 {
                groups[k] = groups[k].div_ceil(local[k].max(1));
            }
        }
        if groups.contains(&0) {
            ZERO_GROUPS[CPU_INDIRECT.with(|c| c.get()) as usize].fetch_add(1, Ordering::Relaxed);
            return Ok(());
        }
        let t_prep = since(t0);
        trace_loads(&self.name, &p, groups);
        // The per-view copy kernel as a buffer update of the words it leaves
        // (`tune copy_update`); its computation is reused for forwarding when
        // it stays a dispatch (None: not tried, Some(None): refused).
        // (Held past a label wait the graphics command processor records in
        // program order still: its copies forward as before.)
        // (A long run's chunk forwards into its own view: `Cp::long_begin`.)
        let forward_here = super::tune::forward_copies() && (!self.deferred || self.past_grow || ((self.past || self.long) && super::forward::held())) && indirect.is_none() && (self.name == "gpu-gfx" || (self.long && self.name.starts_with("gpu-gfx-par"))) && !super::resident::mirror_mode();
        // The graphics queue's parallel workers (`copy_update 2`): updates
        // checked once their batches completed, never forwarded.
        let worker_here = super::tune::forward_copies() && self.deferred && !self.held && indirect.is_none() && self.name.starts_with("gpu-gfx") && super::tune::copy_update_workers() && !super::resident::mirror_mode();
        let mut forwarded: Option<Option<(u64, Vec<u32>, super::forward::Inputs)>> = None;
        if (forward_here || worker_here)
            && super::tune::copy_update()
            && patches.is_none()
            && initiator & (1 << 5) == 0
            && (0x204..0x207).all(|r| cp.reg(Space::Sh, r) == 0)
            && !verify::enabled()
            && !super::tune::race_checks()
            && !super::capture::capturing()
        {
            // A worker also cannot see what the main command processor's batch,
            // unsubmitted when the run was planned, writes.
            let busy = |r: &[(u64, u64)]| {
                self.overlaps_pending_ro(r) || INFLIGHT.latest(r).is_some() || CP_WRITES.latest(r).is_some() || self.main_writes.as_ref().is_some_and(|m| m.overlaps_any(r))
            };
            match super::forward::compute(&p, groups, local, busy) {
                Some(c) => match self.copy_as_update(&p, c, worker_here) {
                    None => return Ok(()),
                    Some(c) => forwarded = Some(Some(c)),
                },
                None => forwarded = Some(None),
            }
        }
        // Diagnosis (`DS_FE2_CHECK_COPIES=1`): a copy kernel's dispatch on a
        // queue whose copies are not forwarded (compute queues, the graphics
        // queue's workers) is checked once its batch completed against the
        // words its inputs give now: a stale source in the mirror shows as
        // words still as before (forwarding checks only the graphics CP's).
        if super::fe2::check_copies() && !forward_here && forwarded.is_none() && indirect.is_none() && patches.is_none() {
            let busy = |r: &[(u64, u64)]| self.overlaps_pending_ro(r) || INFLIGHT.latest(r).is_some() || CP_WRITES.latest(r).is_some() || self.main_writes.as_ref().is_some_and(|m| m.overlaps_any(r));
            if let Some((base, words, _)) = super::forward::compute(&p, groups, local, busy) {
                let id = super::forward::add_check(base, words, "a copy kernel's dispatch", &self.name);
                self.forward_ids.push(id);
            }
        }
        self.grow_sampled_depth(&[&p]);
        // Fast clears, overwrite reloads and block copies of the images, as
        // for draws: the keys from this thread's memos, the cache's lock only
        // when one may be pending.
        let mut keys = std::mem::take(&mut self.keys_scratch);
        keys.clear();
        let hz = image_hazards(&[&p], &mut keys);
        if self.held_hit(&keys) || hz {
            self.resolve_fast_clears(&keys);
            self.resolve_overwrites(&keys);
            self.sync_block_textures(&keys);
        }
        self.keys_scratch = keys;
        if self.stopped {
            return Err(STOPPED.into());
        }
        // The fill kernel over a whole range (`verify::full_fill`): one buffer
        // fill of guest memory in place, without the kernel's VRAM shadow
        // (a copy in, then a merge kernel storing every dword back across
        // PCIe: 1.3-2.3 ms per ~1 MiB HTILE clear, run 20261012-205403).
        let plain = indirect.is_none() && cp.reg(Space::Sh, 0x207) >> 16 == 0 && (0x204..0x207).all(|r| cp.reg(Space::Sh, r) == 0);
        if !plain && super::tune::fill_dma() && verify::fill_value(&p).is_some() {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| crate::info!("gpu[{}]: fill kernel cs {:#x} stays a dispatch: indirect {}, COMPUTE_NUM_THREAD_X {:#x}, start {:?}", self.name, p.addr, indirect.is_some(), cp.reg(Space::Sh, 0x207), [cp.reg(Space::Sh, 0x204), cp.reg(Space::Sh, 0x205), cp.reg(Space::Sh, 0x206)]));
        }
        if plain && (super::tune::fill_dma() || super::fe2::on()) && !super::resident::mirror_mode() {
            if let Some((base, len, v)) = verify::full_fill(&p, groups, local, threads_x) {
                // A zero fill over nothing but GPU-written images it clears
                // stores nothing into their guest memory (`tune fill_skip`).
                let skip = v == 0
                    && len >= FILL_SKIP_MIN
                    && super::tune::zero_fill_clears()
                    && !super::homes::overlaps(base, len)
                    && ((super::tune::fill_skip() && super::images::with(|im| im.fill_covered(base, len))) || super::tune::skip_copies() & 8 != 0);
                if skip {
                    self.flush_resident(Some(&[(base, len)]))?;
                    FILL_SKIPS[0].fetch_add(1, Ordering::Relaxed);
                    FILL_SKIPS[1].fetch_add(len, Ordering::Relaxed);
                } else if v == 0 && len >= FILL_SKIP_MIN && profiling() {
                    // Why a large zero fill stores into guest memory (a range's
                    // 1st and 300th time, the first 24 ranges; `DS_PROFILE_GPU=1`).
                    static SEEN: Mutex<Vec<(u64, u64, u64)>> = Mutex::new(Vec::new());
                    let mut g = SEEN.lock();
                    let n = match g.iter().position(|e| e.0 == base && e.1 == len) {
                        Some(i) => {
                            g[i].2 += 1;
                            g[i].2
                        }
                        None if g.len() < 24 => {
                            g.push((base, len, 1));
                            1
                        }
                        None => 0,
                    };
                    if n == 1 || n == 300 {
                        drop(g);
                        let homed = super::homes::overlaps(base, len);
                        let report = super::images::with(|im| im.fill_cover_report(base, len));
                        let access = crate::mem::prot_at(base).map_or("unmapped", crate::mem::prot_class);
                        crate::info!("gpu: zero fill {base:#x}+{len:#x} (#{n}) stores into guest memory (home {homed}, mapping {access}): {report}");
                    }
                }
                if skip || self.fill_in_place(base, len, v)? {
                    let zeroed = if self.held {
                        self.held_fill(base, len, v)
                    } else {
                        super::images::with(|im| {
                        let z = if v == 0 && super::tune::zero_fill_clears() {
                            im.note_meta_writes(&[(base, len)]);
                            im.zero_filled(base, len)
                        } else {
                            im.note_writes(&[(base, len)], &[(base, len)]);
                            Vec::new()
                        };
                        im.note_meta_fill(base, len, v);
                        z
                        })
                    };
                    if !zeroed.is_empty() {
                        self.clear_images_zero(&zeroed);
                    }
                    super::mirror::invalidate(&[(base, len)]);
                    super::track::gpu_write(base, len);
                    self.note_write(base, len);
                    if lazy_barriers() {
                        self.dispatched = true;
                    }
                    EXECUTED.fetch_add(1, Ordering::Relaxed);
                    FILLS_IN_PLACE.fetch_add(1, Ordering::Relaxed);
                    // `DS_GPU_VERIFY=1` compares the result as for the kernel.
                    if let Some(mut ch) = verify::check_for(&p, groups, local) {
                        ch.writes_mark = self.pending_log.len();
                        self.checks.push(ch);
                    }
                    return Ok(());
                }
            }
        }
        lap.mark(3);
        let Pipe { pipeline, layout, set_layout, push } = match cached_compute_pipe(&p) {
            Some(x) => {
                if compile_trace() {
                    let key = push_key(p.tr.key, push_fits(&[&p]));
                    if AHEAD_KEYS.lock().get_or_insert_with(HashSet::new).remove(&key) {
                        crate::info!("used ahead: cs {:#x} key {key:#x}", p.addr);
                    }
                }
                x
            }
            None => {
                // A first use: the programs of the commands after this one
                // are compiled meanwhile (`gpu::compile`).
                let t = std::time::Instant::now();
                if super::compile::at_first_uses() && !compute_pipe_known(&p) && !async_queued(&p) {
                    cp.prefetch_compiles();
                }
                // A first use of a per-frame effect's program (`tune
                // async_particles`; particle programs too with 1): its
                // pipeline compiles in the background and its dispatches are
                // left out until it is made, instead of this command
                // processor stopping for the compile (5-12 s for the largest
                // particle programs: first visits froze for up to 20.8 s,
                // FINDINGS.md 2026-10-23). The effect shows up that much
                // later; effects whose state persists break (FINDINGS.md
                // 2026-10-24).
                let mode = super::tune::async_particles();
                if mode != 0 && super::shaders::program_hash(p.addr).is_ok_and(|h| super::particles::in_background(h, mode == 1)) && compile_in_background(&p) {
                    ASYNC_STATS[0].fetch_add(1, Ordering::Relaxed);
                    return Err(ASYNC_PENDING.into());
                }
                let r = compute_pipe(&p);
                if compile_trace() {
                    let key = push_key(p.tr.key, push_fits(&[&p]));
                    let from = super::particles::origin(p.code_hash).map(|d| format!(" (from {d})")).unwrap_or_default();
                    crate::info!("first use: cs {:#x} key {key:#x} after {} ms{from}", p.addr, t.elapsed().as_millis());
                    if let Some(v) = AHEAD_SPECS.lock().as_ref().and_then(|m| m.get(&p.addr)) {
                        if let Some((k, code, s)) = v.last().filter(|_| !v.iter().any(|x| x.0 == key)) {
                            crate::info!("first use: cs {:#x} key {key:#x} differs from the pipeline compiled ahead ({k:#x}, {} of them; code hash {}): {}", p.addr, v.len(), if *code == p.code_hash { "same" } else { "differs" }, spec_diff(s, &p.tr.spec));
                        }
                    }
                }
                FIRST_USES[0].fetch_add(1, Ordering::Relaxed);
                FIRST_USES[1].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
                r?
            }
        };
        let t_pipe = since(t0);
        lap.mark(4);
        self.reserve(&[&p], push);
        if self.stopped {
            return Err(STOPPED.into());
        }
        if super::vk::labels_on() {
            let l = format!("{} cs {:#x}{}", self.name, p.addr, if indirect.is_some() { " indirect" } else { "" });
            self.label(&l);
        }
        lap.mark(5);
        let mut bound = self.bind_stage(&mut p, true)?;
        if self.stopped {
            return Err(STOPPED.into());
        }
        lap.mark(6);
        let (ring, at) = self.bind_sgpr(&p, &mut bound)?;
        if let Some(pt) = &patches {
            self.patch_sgpr(pt, 0, ring, at);
        }
        if profiling() {
            note_placements(p.addr, groups, indirect.is_some(), &bound.placements);
        }
        lap.mark(7);
        let t_bind = since(t0);
        let set = if push { None } else { Some(self.alloc_set(set_layout)?) };
        lap.mark(8);
        let writes_here = std::mem::take(&mut bound.writes);
        let sized_here = std::mem::take(&mut bound.sized_writes);
        let fill = verify::fill_value(&p);
        // The same writes as this executor's last dispatch, with nothing
        // changed since in the structures they were noted in. The notes
        // themselves move none of these generations: read before them, a
        // change by another thread while they run makes the next one note.
        let gens = note_gens();
        let repeat = super::tune::lean_notes() && fill.is_none() && !writes_here.is_empty() && self.last_notes.2 == gens && self.last_notes.0 == writes_here && self.last_notes.1 == sized_here;
        // Writes on no page of a GPU-owned image or fast-clear metadata have
        // nothing to note there (`images::may_note`). The frame-ahead
        // recorder notes them for itself (the image cache must not see
        // writes the current frame's work comes before).
        if self.held {
            if super::images::may_note(&writes_here) {
                self.held_note(&writes_here, &sized_here, fill);
            }
        } else if !repeat && (fill.is_some() || super::images::may_note(&writes_here)) {
            super::images::with(|im| {
                im.note_writes(&writes_here, &sized_here);
                if let Some(((base, len), v)) = fill {
                    im.note_meta_fill(base, len, v);
                }
            });
        }
        lap.mark(9);
        if let Some(set) = set {
            self.write_set(set, std::slice::from_ref(&bound));
        }
        lap.mark(10);
        self.end_rendering_lazy();
        let c = ctx();
        let qp = self.slots[self.cur].queries;
        let ti = self.batch_shaders.len() as u32;
        let batched = p.tr.out.batched;
        let stamped = !batched && stamp_worth(indirect.map_or(Some(groups[0] as u64 * groups[1] as u64 * groups[2] as u64), |_| None), 256);
        // Counted as a use only when stamps are recorded (an open batched
        // dispatch takes more commands while nothing was recorded since).
        let stamp_cmd = if qp != vk::QueryPool::null() && ti < MAX_TIMED && stamped { self.cmd() } else { self.slots[self.cur].cmd };
        let stamp = |k: u32| {
            if qp != vk::QueryPool::null() && ti < MAX_TIMED && stamped {
                unsafe { ctx().device.cmd_write_timestamp(stamp_cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, qp, STAMPS * ti + k) };
            }
        };
        if lazy_barriers() && self.barrier_wanted {
            if profiling() {
                let class = if self.deferred { 3 } else if self.name == "gpu-acb0x50" { 2 } else if self.compute_cp { 1 } else { 0 };
                TITLE_BARRIERS[class][self.dispatch_hazard(&bound, indirect)].fetch_add(1, Ordering::Relaxed);
            }
            if super::tune::skip_barriers() & 16 == 0 {
                self.barrier();
            }
            self.barrier_wanted = false;
            self.dispatched = false;
        }
        if profiling() {
            self.note_since_reads(&bound, indirect);
        }
        // Arguments draws read on the GPU since the last barrier, which this
        // dispatch rewrites: the draws read them first.
        if self.args_read && bound.writes.iter().any(|&(b, n)| super::homes::watched_overlaps(b, n)) {
            ARGS_WAR_BARRIERS.fetch_add(1, Ordering::Relaxed);
            self.barrier();
        }
        if super::tune::race_checks() {
            self.note_unordered(std::slice::from_ref(&bound), &[p.addr], false);
        }
        stamp(0);
        let fresh: Vec<Shadow> = bound.shadows.iter().filter(|s| !s.reused).copied().collect();
        self.ops[OP_SHADOWS_IN] += fresh.len() as u64;
        self.ops[OP_SHADOW_BYTES] += fresh.iter().map(|s| s.range).sum::<u64>();
        if !fresh.is_empty() && super::tune::skip_copies() & 1 == 0 {
            super::vk::checkpoint(self.cmd(), super::vk::mark::SHADOW_IN, p.addr);
            // Shadows get the guest data first (earlier writes are ordered by
            // the title's synchronization, or the barrier after every
            // dispatch without lazy barriers).
            // A window an earlier batch's merge left in the shared copies
            // (`tune shadow_share`): copied in from video memory. Its CPU
            // writes are followed from now on (its marks so far: the copy
            // runs after them).
            let share = shadow_merge() && super::tune::shadow_share() && super::precache::enabled();
            let mut kept: Vec<Option<(vk::Buffer, u64)>> = Vec::with_capacity(fresh.len());
            for sh in &fresh {
                if !share || sh.range < super::tune::shadow_share_min() {
                    kept.push(None);
                    continue;
                }
                let since = super::precache::watch(sh.guest, sh.range);
                if let Some(s) = bound.shadows.iter_mut().find(|s| s.shadow_off == sh.shadow_off) {
                    s.since = Some(since);
                }
                // Not in a parallel chunk: earlier chunks of the run, recorded
                // at the same time, may write the window (the culls' counters
                // accumulate across chunks), and their writes invalidate the
                // shared copy only when recorded; this chunk's copy in runs
                // after theirs and must see them (stale counters hung the GPU
                // at a World 5-1 load, runs 20261013-035334 .. -041906).
                let shared = if self.deferred || (self.batch_alt && !self.batch_alt_share) { None } else { super::precache::lookup(sh.guest, sh.range) };
                match shared {
                    Some((chunk, at, made)) => {
                        self.copy_wait = self.copy_wait.max(made);
                        kept.push(Some((chunk.buf, at)));
                        self.slots[self.cur].keep.push(chunk);
                        SHADOWS_SHARED[1].fetch_add(1, Ordering::Relaxed);
                    }
                    None => kept.push(None),
                }
            }
            // Into the prologue (`tune shadow_pre`) when no shadow comes from
            // a shared copy and nothing earlier in the batch (or a batch not
            // submitted yet that it follows) writes their windows: the
            // prologue runs before the batch's work, so the copies see guest
            // memory as they would here, and no barrier stops the GPU in the
            // middle of the batch.
            let why = if !super::tune::shadow_pre() || !shadow_merge() {
                2
            } else if self.deferred {
                3
            } else if kept.iter().any(|k| k.is_some()) {
                4
            } else if fresh.iter().any(|sh| self.pending.overlaps(sh.guest, sh.range)) {
                5
            } else if fresh.iter().any(|sh| self.main_pending(&[(sh.guest, sh.range)])) {
                6
            } else if fresh.iter().any(|sh| self.closed_overlap(&[(sh.guest, sh.range)])) {
                7
            } else {
                0
            };
            let to_pre = why == 0;
            SHADOWS_PRE[why].fetch_add(fresh.len() as u64, Ordering::Relaxed);
            if why != 0 {
                SHADOWS_PRE[1].fetch_add(fresh.len() as u64, Ordering::Relaxed);
            }
            if to_pre {
                let _ = self.open_pre();
                let shadow = self.shadow();
                for sh in &fresh {
                    self.pre_copies.push((sh.host, shadow, vk::BufferCopy { src_offset: sh.host_off, dst_offset: sh.pristine_off, size: sh.range }));
                    self.pre_after.push((shadow, shadow, vk::BufferCopy { src_offset: sh.pristine_off, dst_offset: sh.shadow_off, size: sh.range }));
                }
            }
            if !to_pre { unsafe {
                if shadow_merge() {
                    // One read of guest memory (the pristine copy), then the
                    // working copy from it: two reads could see different CPU
                    // writes, and the merge would store the older value.
                    for (sh, k) in fresh.iter().zip(&kept) {
                        match *k {
                            Some((buf, at)) => {
                                let r = vk::BufferCopy { src_offset: at, dst_offset: sh.pristine_off, size: sh.range };
                                c.device.cmd_copy_buffer(self.cmd(), buf, self.shadow(), &[r]);
                                // DS_SHADOW_SHARE_CHECK: guest memory and the
                                // shared copy, both as of this point.
                                let every = verify::share_check_every();
                                if every > 0 && SHARE_CHECKED.fetch_add(1, Ordering::Relaxed) % every == 0 {
                                    if let Some((check, rb)) = verify::shared_check((sh.guest, sh.range), sh.range) {
                                        c.device.cmd_copy_buffer(self.cmd(), sh.host, rb, &[vk::BufferCopy { src_offset: sh.host_off, dst_offset: 0, size: sh.range }]);
                                        c.device.cmd_copy_buffer(self.cmd(), buf, rb, &[vk::BufferCopy { src_offset: at, dst_offset: sh.range, size: sh.range }]);
                                        let hb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::HOST_READ);
                                        c.device.cmd_pipeline_barrier(self.cmd(), vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::HOST, vk::DependencyFlags::empty(), &[hb], &[], &[]);
                                        self.checks.push(check);
                                    }
                                }
                            }
                            None => {
                                let r = vk::BufferCopy { src_offset: sh.host_off, dst_offset: sh.pristine_off, size: sh.range };
                                c.device.cmd_copy_buffer(self.cmd(), sh.host, self.shadow(), &[r]);
                            }
                        }
                    }
                    let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_READ);
                    c.device.cmd_pipeline_barrier(self.cmd(), vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[mb], &[], &[]);
                    for sh in &fresh {
                        let r = vk::BufferCopy { src_offset: sh.pristine_off, dst_offset: sh.shadow_off, size: sh.range };
                        c.device.cmd_copy_buffer(self.cmd(), self.shadow(), self.shadow(), &[r]);
                    }
                } else {
                    for sh in &fresh {
                        let r = vk::BufferCopy { src_offset: sh.host_off, dst_offset: sh.shadow_off, size: sh.range };
                        c.device.cmd_copy_buffer(self.cmd(), sh.host, self.shadow(), &[r]);
                    }
                }
                let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
                c.device.cmd_pipeline_barrier(self.cmd(), vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &[mb], &[], &[]);
            } }
        }
        stamp(1);
        lap.mark(11);
        if super::vk::checkpoints_on() {
            super::vk::checkpoint(self.cmd(), super::vk::mark::DISPATCH, p.addr);
        }
        let start = [cp.reg(Space::Sh, 0x204), cp.reg(Space::Sh, 0x205), cp.reg(Space::Sh, 0x206)];
        // Arguments on resident GPU-written data (`gpu::rgm`) are read from
        // its working copy (the dispatch before typically wrote them there);
        // else this batch's W writes there go to guest memory first.
        let rgm_args = indirect.filter(|_| !self.rgm_unmerged.is_empty() || super::rgm::get().is_some()).and_then(|a| self.rgm_bind(a, 12, false)).map(|(b, off, _)| (b, off));
        if let (Some(a), None) = (indirect, rgm_args) {
            self.rgm_before_in_place(&[(a, 12)])?;
        }
        let args = match indirect {
            Some(_) if rgm_args.is_some() => {
                RGM_ARGS.fetch_add(1, Ordering::Relaxed);
                rgm_args
            }
            // Still held by a resident shadow (binding and batch changes since
            // the check above may have merged it): its working copy.
            Some(a) if self.shadow_args() && self.resident_args(a).is_some() => {
                SHADOW_ARGS.fetch_add(1, Ordering::Relaxed);
                self.resident_args(a)
            }
            Some(a) if super::fe2::on() && super::fe2::in_span(a, 12) => match self.fe2_place(a, 12, false) {
                Some((buffer, off, _)) => Some((buffer, off)),
                None => return Err(format!("cs {:#x}: indirect arguments {a:#x} are not mapped GPU memory (dispatch skipped)", p.addr)),
            },
            Some(a) if super::resident::mirror_mode() && {
                self.mm_unsized = false;
                self.mirror_place(a, 12, false).is_some()
            } => {
                let (buffer, off, _) = self.mirror_place(a, 12, false).expect("placed");
                if let Some(s) = CP_WRITES.latest(&[(a, 12)]) {
                    self.submit_after = self.submit_after.max(s);
                }
                Some((buffer, off))
            }
            Some(a) => match self.keep_args(a)? {
                Some(k) => Some(k),
                None => {
                    let (buffer, base, _) = span(a, 12).ok_or_else(|| format!("cs {:#x}: indirect arguments {a:#x} are not mapped GPU memory (dispatch skipped)", p.addr))?;
                    if let Some(s) = CP_WRITES.latest(&[(a, 12)]) {
                        self.submit_after = self.submit_after.max(s);
                    }
                    Some((buffer, a - base))
                }
            },
            None => None,
        };
        // A batched program (`Spec::batched`): appended to the open batched
        // dispatch when nothing was recorded since it and its descriptors
        // match, else recorded as a new one (a direct 1-D dispatch) or on
        // its own.
        let simple = batched && indirect.is_none() && start == [0, 0, 0] && patches.is_none() && set.is_none() && groups[1] == 1 && groups[2] == 1 && groups[0] <= BATCH_MAX_GROUPS;
        // SGPR data offsets are relative to the uniform window, if any.
        let mut item = ((at - bound.window.unwrap_or(0)) / 4) as u32;
        let appended = simple
            && match self.cs_run.as_mut() {
                Some(r) if r.batch == self.batch_count && r.uses == self.cmd_uses.get() && r.pipeline == pipeline && r.groups + groups[0] <= BATCH_MAX_GROUPS && same_descs(&bound, &r.descs) => {
                    unsafe {
                        for g in 0..groups[0] {
                            let e = r.table.add((rdna::translate::BATCH_ENTRY_DWORDS * (r.groups + g)) as usize);
                            e.write(item);
                            e.add(1).write(g);
                        }
                        r.groups += groups[0];
                        r.args.write(r.groups);
                    }
                    // A GPU-resolved run takes this command's run roots (it
                    // came here because they differ): the items after it
                    // that share them are appended alone again.
                    if let Some(g) = r.gpu.as_mut().filter(|g| g.addr == p.addr && p.tr.spec.gpu_res != 0) {
                        let user = &p.sgpr_data[..g.users.min(p.sgpr_data.len())];
                        for (root, v) in g.roots.iter_mut() {
                            *v = super::gpures::root_value(*root, user);
                        }
                        g.template.clear();
                        g.template.extend_from_slice(&p.sgpr_data);
                    }
                    true
                }
                _ => false,
            };
        if appended {
            BATCHED[0].fetch_add(1, Ordering::Relaxed);
        } else if batched {
            let miss = if indirect.is_some() {
                0
            } else if !simple {
                1
            } else {
                match self.cs_run.as_ref() {
                    None => 2,
                    Some(r) if r.batch != self.batch_count => 3,
                    Some(r) if r.uses != self.cmd_uses.get() => 4,
                    Some(r) if r.pipeline != pipeline => 5,
                    Some(r) if r.groups + groups[0] > BATCH_MAX_GROUPS => 6,
                    Some(_) => 7,
                }
            };
            BATCH_MISS[miss].fetch_add(1, Ordering::Relaxed);
            *BATCH_MISS_PROGRAMS.lock().get_or_insert_with(HashMap::new).entry((p.addr, miss as u8)).or_default() += 1;
            if miss == 4 {
                *BATCH_MISS_SITES.lock().get_or_insert_with(HashMap::new).entry(self.last_cmd_line.get()).or_default() += 1;
            }
            // A new batched dispatch: its arguments and table in the ring.
            let which = if self.slots[self.cur].rings[1].is_some_and(|r| r.0 == ring) { 1 } else { 0 };
            let table_bytes = 16 + 4 * (rdna::translate::BATCH_ENTRY_DWORDS * BATCH_MAX_GROUPS) as u64;
            let mut run_at = if simple { self.ring_alloc(which, table_bytes) } else { None };
            // The table inside the uniform window: else the window moves to
            // this command's data (the table follows it), or the dispatch
            // runs on its own.
            if let (Some(wb), Some(t)) = (bound.window, run_at) {
                if t + table_bytes > wb + SGPR_WINDOW_BYTES {
                    let nb = self.window_at(at);
                    if t + table_bytes <= nb + SGPR_WINDOW_BYTES {
                        Self::move_window(&mut bound, nb);
                        item = ((at - nb) / 4) as u32;
                    } else {
                        run_at = None;
                    }
                }
            }
            let wbase = bound.window.unwrap_or(0);
            let ring_ptr = self.slots[self.cur].rings[which].map(|r| r.1);
            unsafe {
                let cmd = self.cmd();
                let rebound = !super::tune::lean_cmd() || self.dyncache.cs_pipeline != pipeline;
                if rebound {
                    note_vk(VK_CS_PIPELINES);
                    c.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
                    self.dyncache.cs_pipeline = pipeline;
                }
                bind_descriptors(cmd, vk::PipelineBindPoint::COMPUTE, layout, set, std::slice::from_ref(&bound), &mut self.dyncache.pushed[1], rebound);
                match (run_at, ring_ptr) {
                    (Some(off), Some(rp)) => {
                        let args = rp.add(off as usize) as *mut u32;
                        let table = args.add(4);
                        for g in 0..groups[0] {
                            let e = table.add((rdna::translate::BATCH_ENTRY_DWORDS * g) as usize);
                            e.write(item);
                            e.add(1).write(g);
                        }
                        args.write(groups[0]);
                        args.add(1).write(1);
                        args.add(2).write(1);
                        let t = ((off + 16 - wbase) / 4) as u32;
                        c.device.cmd_push_constants(cmd, layout, vk::ShaderStageFlags::COMPUTE, 0, &t.to_le_bytes());
                        c.device.cmd_dispatch_indirect(cmd, ring, off);
                        let gpu = if p.tr.spec.gpu_res != 0 {
                            super::gpures::decided(p.addr).map(|d| {
                                let user = &p.sgpr_data[..p.tr.spec.user_sgprs as usize];
                                Box::new(GpuRun {
                                    addr: p.addr,
                                    found: p.found.clone(),
                                    roots: d.run_roots.iter().map(|&r| (r, super::gpures::root_value(r, user))).collect(),
                                    items: d.item_roots.clone(),
                                    template: p.sgpr_data.clone(),
                                    users: p.tr.spec.user_sgprs as usize,
                                    which,
                                    regs: gpu_run_regs(cp),
                                    fe2_want: self.fe2_run_want(),
                                })
                            })
                        } else {
                            None
                        };
                        self.cs_run = Some(CsRun { batch: self.batch_count, uses: self.cmd_uses.get(), pipeline, descs: desc_signature(&bound), args, table, groups: groups[0], ring, window: wbase, gpu });
                        BATCHED[1].fetch_add(1, Ordering::Relaxed);
                    }
                    _ => {
                        // On its own: the push constant names its SGPR data.
                        let pc = rdna::translate::BATCH_SINGLE | item;
                        c.device.cmd_push_constants(cmd, layout, vk::ShaderStageFlags::COMPUTE, 0, &pc.to_le_bytes());
                        BATCHED[2].fetch_add(1, Ordering::Relaxed);
                        if let Some((buffer, offset)) = args {
                            let a = indirect.unwrap_or(0);
                            if super::tune::skip_barriers() & 2 == 0 && (!super::tune::exact_indirect() || self.written_since_barrier(a, 12)) {
                                self.barrier();
                                INDIRECT_BARRIERS[0].fetch_add(1, Ordering::Relaxed);
                            } else {
                                INDIRECT_BARRIERS[1].fetch_add(1, Ordering::Relaxed);
                            }
                            c.device.cmd_dispatch_indirect(self.cmd(), buffer, offset);
                        } else if start != [0, 0, 0] {
                            c.device.cmd_dispatch_base(cmd, start[0], start[1], start[2], groups[0].saturating_sub(start[0]), groups[1].saturating_sub(start[1]), groups[2].saturating_sub(start[2]));
                        } else {
                            c.device.cmd_dispatch(cmd, groups[0], groups[1], groups[2]);
                        }
                    }
                }
            }
        }
        if !batched {
        unsafe {
            let cmd = self.cmd();
            let rebound = !super::tune::lean_cmd() || self.dyncache.cs_pipeline != pipeline;
            if rebound {
                note_vk(VK_CS_PIPELINES);
                c.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
                self.dyncache.cs_pipeline = pipeline;
            }
            bind_descriptors(cmd, vk::PipelineBindPoint::COMPUTE, layout, set, std::slice::from_ref(&bound), &mut self.dyncache.pushed[1], rebound);
            push_sgpr_bases(cmd, layout, true, std::slice::from_ref(&bound));
            if let Some((buffer, offset)) = args {
                lap.mark(12);
                // Work of this batch since its last barrier wrote the
                // arguments (`tune exact_indirect`; earlier batches complete
                // before this one starts).
                let a = indirect.unwrap_or(0);
                if super::tune::skip_barriers() & 2 == 0 && (!super::tune::exact_indirect() || self.written_since_barrier(a, 12)) {
                    self.barrier();
                    INDIRECT_BARRIERS[0].fetch_add(1, Ordering::Relaxed);
                } else {
                    INDIRECT_BARRIERS[1].fetch_add(1, Ordering::Relaxed);
                }
                lap.mark(18);
                c.device.cmd_dispatch_indirect(cmd, buffer, offset);
            } else if start != [0, 0, 0] {
                c.device.cmd_dispatch_base(cmd, start[0], start[1], start[2], groups[0].saturating_sub(start[0]), groups[1].saturating_sub(start[1]), groups[2].saturating_sub(start[2]));
            } else {
                c.device.cmd_dispatch(cmd, groups[0], groups[1], groups[2]);
            }
        }
        }
        stamp(2);
        lap.mark(12);
        if !lazy_barriers() || (!bound.shadows.is_empty() && super::tune::shadow_barrier()) {
            self.barrier();
        }
        // Written shadows stay resident for the rest of the batch (see Shadow);
        // a reused one merged back while this dispatch's bindings were made
        // joins again. Overlapping windows (aliased bindings) are merged now.
        for sh in &bound.shadows {
            if !self.resident.iter().any(|r| r.shadow_off == sh.shadow_off) {
                self.resident.push(Shadow { reused: false, ..*sh });
            }
        }
        // Writes of this dispatch in place (not through one of its shadows)
        // into a resident window.
        if !self.resident.is_empty() {
            for &(b, n) in &writes_here {
                if !bound.shadows.iter().any(|s| s.guest <= b && b + n <= s.guest + s.range) {
                    self.taint_resident(b, n);
                }
            }
        }
        let aliased = self.resident.iter().enumerate().any(|(i, a)| self.resident[..i].iter().any(|b| a.guest < b.guest + b.range && b.guest < a.guest + a.range));
        if aliased || !super::tune::shadow_keep() {
            self.flush_resident(None)?;
        }
        if lazy_barriers() {
            self.dispatched = true;
        }
        stamp(3);
        lap.mark(13);
        // VRAM mirrors of what this dispatch wrote (later read bindings use them).
        let mut mirrored: Vec<(u64, u64)> = Vec::new();
        let mut copies: Vec<(vk::Buffer, vk::Buffer, vk::BufferCopy)> = Vec::new();
        for &(base, size, src, src_off) in &bound.write_src {
            if let Some((mbuf, moff)) = super::mirror::for_write(base, size) {
                copies.push((src, mbuf, vk::BufferCopy { src_offset: src_off, dst_offset: moff, size }));
                mirrored.push((base, size));
            }
        }
        if mirrored.is_empty() {
            if !writes_here.is_empty() {
                super::mirror::invalidate(&writes_here);
            }
        } else {
            let unmirrored: Vec<(u64, u64)> = writes_here.iter().copied().filter(|w| !mirrored.contains(w)).collect();
            if !unmirrored.is_empty() {
                super::mirror::invalidate(&unmirrored);
            }
        }
        if !copies.is_empty() {
            unsafe {
                let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE);
                c.device.cmd_pipeline_barrier(self.cmd(), vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[mb], &[], &[]);
                for (src, dst, r) in &copies {
                    c.device.cmd_copy_buffer(self.cmd(), *src, *dst, &[*r]);
                }
            }
            // Readers later in this batch wait for the copies.
            self.barrier_wanted = true;
            for &(b, n) in &mirrored {
                super::mirror::refreshed(b, n);
            }
        }
        lap.mark(14);
        if repeat {
            // This batch's own record of them still: it may have begun since.
            self.own_gen += 1;
            for &(b, n) in &writes_here {
                super::forward::written(b, n);
                if super::tune::race_checks() {
                    self.note_open_write(b, n);
                }
                self.pending.insert(b, n);
                self.note_since(b, n);
                if verify::enabled() {
                    self.pending_log.push((b, n));
                }
            }
            NOTES_SKIPPED.fetch_add(1, Ordering::Relaxed);
        } else {
            for &(b, n) in &writes_here {
                super::track::gpu_write(b, n);
            }
            self.noting = p.addr;
            for &(b, n) in &writes_here {
                self.note_write(b, n);
            }
            self.noting = 0;
            if fill.is_some() {
                // Known values noted: the next write of the range is not a
                // repeat of this one.
                self.last_notes.2 = [u64::MAX; 4];
            } else if !writes_here.is_empty() {
                self.last_notes.0.clone_from(&writes_here);
                self.last_notes.1.clone_from(&sized_here);
                self.last_notes.2 = gens;
            }
        }
        lap.mark(15);
        if forward_here {
            let busy = |r: &[(u64, u64)]| self.overlaps_pending_ro(r) || INFLIGHT.latest(r).is_some() || CP_WRITES.latest(r).is_some();
            let computed = match forwarded {
                Some(c) => c,
                None => super::forward::compute(&p, groups, local, busy),
            };
            if let Some((base, words, from)) = computed {
                // A copy the run's planner forwarded: computed alike?
                if self.long {
                    if let Some(w) = self.long_planned.get(&cp.cur_packet()) {
                        if w[..] != words[..] {
                            self.long_mismatch = true;
                            super::forward::PLANNED_STATS[1].fetch_add(1, Ordering::Relaxed);
                        }
                    }
                }
                // (Diagnostics: how this dispatch bound the destination.)
                let how = match bound.shadows.iter().find(|s| s.guest <= base && base < s.guest + s.range) {
                    Some(s) if s.reused => "through a resident shadow",
                    Some(_) => "through a VRAM shadow",
                    None if super::homes::overlaps(base, 4) => "in a home",
                    None => "in place",
                };
                let id = super::forward::add(base, words, p.addr, p.code_hash, from, how, &self.name);
                self.forward_ids.push(id);
            }
        }
        lap.mark(19);
        EXECUTED.fetch_add(1, Ordering::Relaxed);
        self.batch_shaders.push(p.addr);
        let total = t_start.elapsed();
        if profiling() {
            note_phases(&CS_PHASES, [t_prep, t_pipe, t_bind, total]);
            note_place(self.deferred, self.batch_shaders.len().saturating_sub(1), [t_prep, t_pipe, t_bind, total]);
            note_cs_cpu(p.addr, kind, [t_prep, t_pipe, t_bind, total]);
            if cs_queue_profiled(&self.name) {
                note_cs_cpu_queue(p.addr, [t_prep, t_pipe, t_bind, total]);
            }
        }
        if laps() {
            if first {
                CS_STEPS_FIRST[kind][CS_STEP_COUNT].fetch_add(1, Ordering::Relaxed);
            } else {
                CS_STEPS[kind][CS_STEP_COUNT].fetch_add(1, Ordering::Relaxed);
            }
            lap.slow(&self.name, || format!("cs {:#x}{}{}", p.addr, if indirect.is_some() { " indirect" } else { "" }, if first { ", first of its batch" } else { "" }));
        }
        if total.as_millis() >= 200 {
            let phases = if profiling() { format!(" (prepare {} ms, pipeline {} ms, bindings {} ms)", t_prep.as_millis(), (t_pipe - t_prep).as_millis(), (t_bind - t_pipe).as_millis()) } else { String::new() };
            crate::info!("gpu[{}]: dispatch of cs {:#x} took {} ms{phases}", self.name, p.addr, total.as_millis());
        }
        // The write lists go back for the next binding (their capacity).
        bound.writes = writes_here;
        bound.sized_writes = sized_here;
        self.recycle(bound);
        if let Some(mut ch) = verify::check_for(&p, groups, local).filter(|_| indirect.is_none()) {
            // Expected values as recorded, when no GPU work still writes them.
            let src = verify::source_ranges(&ch);
            if !src.is_empty() && !self.overlaps_pending(&src) && INFLIGHT.latest(&src).is_none() && CP_WRITES.latest(&src).is_none() {
                verify::snapshot_source(&mut ch);
            }
            ch.writes_mark = self.pending_log.len();
            self.checks.push(ch);
        }
        Ok(())
    }
}

impl Drop for Exec {
    fn drop(&mut self) {
        self.flush();
        let c = ctx();
        unsafe {
            for s in &self.slots {
                // The completion thread must not reset what is destroyed.
                loop {
                    let v = s.reset_state.load(Ordering::Acquire);
                    if v == 0 || (v != u64::MAX && s.reset_state.compare_exchange(v, 0, Ordering::AcqRel, Ordering::Acquire).is_ok()) {
                        break;
                    }
                    std::thread::yield_now();
                }
                c.device.destroy_fence(s.fence, None);
                c.device.destroy_descriptor_pool(s.desc_pool, None);
                for qp in [s.queries, s.bq] {
                    if qp != vk::QueryPool::null() {
                        c.device.destroy_query_pool(qp, None);
                    }
                }
                c.device.destroy_command_pool(s.cpool, None);
                if s.xcpool != vk::CommandPool::null() {
                    c.device.destroy_command_pool(s.xcpool, None);
                }
            }
        }
    }
}

/// One status line for hang reports.
pub fn status() -> String {
    format!(
        "    gpu exec: {} dispatches executed ({} with arguments read by the GPU, {} of them from a resident shadow, {} of them behind a barrier of their own; stages prepared again with GPU-side data loads {:?} ({} programs since with them); {} render passes ended into the next full barrier; GPU-performed CP writes after merging resident shadows {} (written before the merge {}); draws sampling an attachment of their rendering instance: {} kept it, {} ended it (color written / not, depth z-written / s-written / cleared / not, stored {:?}; pixel shaders {}); {} CP writes by the GPU ({} timestamps), left to the CPU {:?}; {} buffers shadowed in VRAM, {} not for lack of space; resident shadows: {} reused, {} read-only bindings served, {} merges ({} of them by {} dispatches of `merge_many`), {} before an in-place use), {} skipped; {} draws executed ({} reading their arguments on the GPU, {} repeating the draw before (chains ended by opcode: {}), {} barriers before a dispatch rewrote them), {} without effect, {} skipped; {} invocations stopped by the loop watchdog; {} fast clears ({} with the fill's value, no wait), {} fill-kernel dispatches recorded as buffer fills ({} of them only clearing the GPU-written images they cover, {} MiB not stored), {} dispatches' write notes skipped as repeats; {} completions waited, {}; waits by opcode {}\n",
        EXECUTED.load(Ordering::Relaxed),
        GPU_INDIRECT.load(Ordering::Relaxed),
        SHADOW_ARGS.load(Ordering::Relaxed),
        INDIRECT_BARRIERS[0].load(Ordering::Relaxed),
        HYBRID_LOADS.iter().map(|c| c.load(Ordering::Relaxed)).collect::<Vec<u64>>(),
        shaders::gpu_loads_programs(),
        MERGED_PASS_BARRIERS.load(Ordering::Relaxed),
        LABEL_MERGES[0].load(Ordering::Relaxed),
        LABEL_MERGES[1].load(Ordering::Relaxed),
        FEEDBACK[0].load(Ordering::Relaxed),
        FEEDBACK[1].load(Ordering::Relaxed),
        FEEDBACK_KINDS.iter().map(|c| c.load(Ordering::Relaxed)).collect::<Vec<u64>>(),
        {
            let mut v: Vec<(u64, u64)> = FEEDBACK_PS.lock().as_ref().map(|m| m.iter().map(|(a, n)| (*a, *n)).collect()).unwrap_or_default();
            v.sort_by(|a, b| b.1.cmp(&a.1));
            v.iter().take(6).map(|(a, n)| format!("{a:#x} {n}")).collect::<Vec<String>>().join(", ")
        },
        GPU_CP_WRITES.load(Ordering::Relaxed),
        GPU_TIMESTAMPS.load(Ordering::Relaxed),
        GPU_CP_REFUSED.iter().map(|c| c.load(Ordering::Relaxed)).collect::<Vec<u64>>(),
        SHADOWED.load(Ordering::Relaxed),
        SHADOW_FULL.load(Ordering::Relaxed),
        SHADOWS_REUSED.load(Ordering::Relaxed),
        RESIDENT_READS.load(Ordering::Relaxed),
        MERGES.load(Ordering::Relaxed),
        MERGE_MANY[1].load(Ordering::Relaxed),
        MERGE_MANY[0].load(Ordering::Relaxed),
        EARLY_MERGES.load(Ordering::Relaxed),
        SKIPPED.load(Ordering::Relaxed),
        draw::DRAWN.load(Ordering::Relaxed),
        draw::GPU_ARG_DRAWS.load(Ordering::Relaxed),
        draw::REPEAT_DRAWS.load(Ordering::Relaxed),
        super::cp::repeat_ends_line(),
        ARGS_WAR_BARRIERS.load(Ordering::Relaxed),
        draw::DRAW_NOOP.load(Ordering::Relaxed),
        draw::DRAW_SKIPPED.load(Ordering::Relaxed),
        WATCHDOG_STOPS.load(Ordering::Relaxed),
        FAST_CLEARS.load(Ordering::Relaxed),
        FAST_CLEARS_KNOWN.load(Ordering::Relaxed),
        FILLS_IN_PLACE.load(Ordering::Relaxed),
        FILL_SKIPS[0].load(Ordering::Relaxed),
        FILL_SKIPS[1].load(Ordering::Relaxed) >> 20,
        NOTES_SKIPPED.load(Ordering::Relaxed),
        FLUSHES.load(Ordering::Relaxed),
        verify::status(),
        super::cp::sync_summary()
    ) + &super::images::IMAGES.try_lock().and_then(|g| g.as_ref().map(|i| i.status())).unwrap_or_default()
        + &slot_mem_line()
        + &super::track::status()
        + &crate::mem::remap_status()
        + &crate::mem::prot_census()
        + &crate::hosted::status()
        + &super::precache::protect_line()
        + &crate::placement::status()
        + &crate::titlemem::status()
        + &crate::titlefix::status()
        + &super::cp::draw_state_line()
        + &draw::draw_diff_line()
        + &draw::prep_repeat_line()
        + &draw::again_line()
        + &draw::batch_line()
        + &super::xwait::status()
        + &super::resident::owned_line()
        + &mm_place_line()
        + &super::lodstats::status()
        + &super::ahead::status()
        + &super::compile::status()
        + &async_line()
        + &super::cp::past_wait_status()
        + &held_own_line()
        + &super::forward::status_line()
        + &super::rgm::status_line(super::FRAMES.load(Ordering::Relaxed))
        + &super::par::status()
        + &super::keep::status()
        + &format!(
            "    batches submitted together (`tune submit_coalesce` {}): {} calls with {} batches
",
            super::tune::submit_coalesce(),
            COALESCED[0].load(Ordering::Relaxed),
            COALESCED[1].load(Ordering::Relaxed)
        )
        + &format!(
            "    end-of-pipe writes without a full barrier: {} labels, {} timestamps (stored by {} dispatches, {} overwritten in their batch); {} waits ordered the work after them; {} CP writes right after front end 2's write-back with no barrier before them\n",
            EOP_LIGHT[0].load(Ordering::Relaxed),
            EOP_LIGHT[1].load(Ordering::Relaxed),
            TS_DISPATCHES.load(Ordering::Relaxed),
            TS_REPEATS.load(Ordering::Relaxed),
            EOP_LIGHT[2].load(Ordering::Relaxed),
            EOP_LIGHT[3].load(Ordering::Relaxed)
        )
        + &format!(
            "    shadows shared after their merge: {}, copies in from them {}; not shared: {} written in place meanwhile, {} CPU-written, {} no entry\n",
            SHADOWS_SHARED[0].load(Ordering::Relaxed),
            SHADOWS_SHARED[1].load(Ordering::Relaxed),
            SHADOWS_SHARED[2].load(Ordering::Relaxed),
            SHADOWS_SHARED[3].load(Ordering::Relaxed),
            SHADOWS_SHARED[4].load(Ordering::Relaxed)
        )
        + &format!(
            "    images: {} block-compressed refreshes from GPU-written views ({} with the mip tail from the tail block's view), {} depth planes from GPU-written color views, {} storage bindings one view per mip level, {} draws without attachments\n",
            BLOCK_COPIES.load(Ordering::Relaxed),
            BLOCK_TAIL_COPIES.load(Ordering::Relaxed),
            DEPTH_COPIES.load(Ordering::Relaxed),
            MIP_VIEW_BINDINGS.load(Ordering::Relaxed),
            draw::ATTACHLESS.load(Ordering::Relaxed)
        )
        + &super::precache::status()
        + &super::precompile::status()
        + &super::lockstat::status()
        + &super::pcache::status()
        + &super::images::scrub_status()
        + &super::mirror::status()
        + &super::homes::status()
        + &super::resident::status()
        + &super::fe2::status()
        + &import_sites_line()
        + &in_place_line()
        + &sync_why_status()
        + &format!(
            "    buffer snapshots in VRAM: {} ({} MiB), {} reused, {} not for lack of space, {} for memory that cannot be imported; draws with snapshotted indices {} ({} MiB); batch-start copies {} ({} MiB), {} reused, {} not for lack of space; on the copy queue {} ({} MiB)\n",
            SNAPS.load(Ordering::Relaxed),
            SNAP_COPIED.load(Ordering::Relaxed) >> 20,
            SNAPS_REUSED.load(Ordering::Relaxed),
            SNAPS_FULL.load(Ordering::Relaxed),
            UNIMPORTABLE_SNAPS.load(Ordering::Relaxed),
            INDEX_SNAPS.load(Ordering::Relaxed),
            INDEX_SNAP_BYTES.load(Ordering::Relaxed) >> 20,
            PRE_COPIES.load(Ordering::Relaxed),
            PRE_BYTES.load(Ordering::Relaxed) >> 20,
            PRE_REUSED.load(Ordering::Relaxed),
            PRE_FULL.load(Ordering::Relaxed),
            XCOPY[0].load(Ordering::Relaxed),
            XCOPY[1].load(Ordering::Relaxed) >> 20
        )
        + &format!(
            "    guest memory imports: {} ({} MiB), {} over several mappings, {} dropped after remapping ({} freed, {} MiB; {} waiting); memory: {}\n",
            IMPORTS.load(Ordering::Relaxed),
            IMPORT_BYTES.load(Ordering::Relaxed) >> 20,
            IMPORTS_MULTI.load(Ordering::Relaxed),
            IMPORTS_STALE.load(Ordering::Relaxed),
            SPANS_FREED[0].load(Ordering::Relaxed),
            SPANS_FREED[1].load(Ordering::Relaxed) >> 20,
            GRAVEYARD_LEN.load(Ordering::Relaxed),
            super::vk::ctx_heaps()
        )
        + &remaps_over_imports_line()
        + &format!("    gpu completion: {} ms waiting for fences, {} ms in actions\n", FENCE_NS.load(Ordering::Relaxed) / 1_000_000, ACTION_NS.load(Ordering::Relaxed) / 1_000_000)
}
