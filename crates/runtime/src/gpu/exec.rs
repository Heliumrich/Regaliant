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
pub(crate) use draw::{color_surface, depth_surface, set_equal_bias, toggle_skip_vs};

/// Dispatches of compute shaders on this list are skipped (`skipcs` control
/// command; diagnosis).
static SKIP_CS: parking_lot::Mutex<Option<std::collections::HashSet<u64>>> = parking_lot::Mutex::new(None);

/// Adds or removes a compute shader address; returns whether it is listed now.
pub fn toggle_skip_cs(addr: u64) -> bool {
    let mut g = SKIP_CS.lock();
    let s = g.get_or_insert_with(Default::default);
    if s.remove(&addr) {
        false
    } else {
        s.insert(addr);
        true
    }
}

fn cs_addr(cp: &Cp) -> u64 {
    ((cp.reg(Space::Sh, 0x20C) as u64) | ((cp.reg(Space::Sh, 0x20D) as u64 & 0xFF) << 32)) << 8
}
pub use draw::DrawArgs;

pub static EXECUTED: AtomicU64 = AtomicU64::new(0);
/// The Vulkan device was lost: GPU execution stops (reported once).
pub static DEVICE_LOST: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// `DS_GPU_SYNC=1`: complete every dispatch before the next (diagnosis of
/// device losses: the failing dispatch is named).
fn sync_mode() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_GPU_SYNC").ok().as_deref() == Some("1"))
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

fn sync_why_status() -> String {
    let n: Vec<u64> = SYNC_WHY.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    format!(
        "    CP read waits: {} CP reads, {} dispatch shader reads, {} dispatch textures, {} draw shader reads, {} draw textures, {} draw indices; pipelines created: {} compute ({} ms), {} graphics ({} ms)\n",
        n[0],
        n[1],
        n[2],
        n[3],
        n[4],
        n[5],
        PIPES_CREATED[0].load(Ordering::Relaxed),
        PIPES_NS[0].load(Ordering::Relaxed) / 1_000_000,
        PIPES_CREATED[1].load(Ordering::Relaxed),
        PIPES_NS[1].load(Ordering::Relaxed) / 1_000_000
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
/// Block-compressed textures refreshed from GPU-written block views.
pub static BLOCK_COPIES: AtomicU64 = AtomicU64::new(0);
/// Guest-memory imports: count, bytes, how many spanned several mappings,
/// and how many were dropped because their pages were remapped.
static IMPORTS: AtomicU64 = AtomicU64::new(0);
static IMPORT_BYTES: AtomicU64 = AtomicU64::new(0);
static IMPORTS_MULTI: AtomicU64 = AtomicU64::new(0);
static IMPORTS_STALE: AtomicU64 = AtomicU64::new(0);

/// GPU execution enabled (`DS_GPU=1`) and Vulkan available.
/// `DS_TRACE_WRITES=<hex bytes>`: logs once each shader (and CP DMA) that
/// writes a guest range at least that large (who fills a large buffer).
pub fn trace_writes(what: &str, base: u64, size: u64) {
    static MIN: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    let min = *MIN.get_or_init(|| std::env::var("DS_TRACE_WRITES").ok().and_then(|v| u64::from_str_radix(v.trim_start_matches("0x"), 16).ok()).unwrap_or(0));
    if min == 0 || size < min {
        return;
    }
    static SEEN: Mutex<Option<HashSet<(String, u64, u64)>>> = Mutex::new(None);
    if SEEN.lock().get_or_insert_with(HashSet::new).insert((what.to_string(), base, size)) {
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
    /// Still backed by the pages it was imported with (cleared when guest
    /// pages in its range are committed or decommitted).
    valid: bool,
}

struct Pipe {
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    set_layout: vk::DescriptorSetLayout,
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
/// `SPAN_GEN` changes with every change so that threads can keep their
/// recent hits (`SPAN_CACHE`). (One mutex used to guard these, the pipeline
/// maps and the fixed buffers, held across whole binding loops and pipeline
/// compiles: the queue threads spent as much time waiting for it as working.)
static SPANS: parking_lot::RwLock<Vec<Span>> = parking_lot::RwLock::new(Vec::new());
static SPAN_GEN: AtomicU64 = AtomicU64::new(1);
const SPAN_CACHE_LEN: usize = 16;

thread_local! {
    /// Recently used imports of this thread (start, end, buffer), valid for
    /// one `SPAN_GEN`, most recent first.
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
const WINDOW: u64 = 64 << 20;
/// Bindings without a bounds check extend at most this far.
const UNBOUNDED_MAX: u64 = 256 << 20;
const DUMMY_BYTES: u64 = 64 << 10;

fn fixed() -> &'static Fixed {
    FIXED.get_or_init(|| {
        let c = ctx();
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC;
        let (gds, _) = c.device_buffer(rdna::translate::GDS_BYTES as u64, usage).expect("GDS buffer");
        let (dummy, _) = c.device_buffer(DUMMY_BYTES, usage).expect("dummy buffer");
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
            // Every bank: no mip sampled (15), count 0.
            c.device.cmd_fill_buffer(cmd, lodstats, 0, super::lodstats::SLOT_BYTES / 2, 15);
            c.device.cmd_fill_buffer(cmd, lodstats, super::lodstats::SLOT_BYTES / 2, super::lodstats::SLOT_BYTES / 2, 0);
            c.device.end_command_buffer(cmd).ok();
            if c.submit(cmd, &[], &[], fence).is_ok() {
                c.device.wait_for_fences(&[fence], true, u64::MAX).ok();
            }
            c.device.destroy_fence(fence, None);
            c.device.destroy_command_pool(pool, None);
        }
        Fixed { gds, debug, debug_ptr, watchdog, dummy, lodstats, lod_readback, lod_readback_ptr, import_align: host.min_imported_host_pointer_alignment.max(4096) }
    })
}

/// Guest mapping containing `addr`: [start, end) of its VMA.
fn mapping(addr: u64) -> Option<(u64, u64)> {
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

/// Drops imports whose guest pages were committed or decommitted since.
fn revalidate_spans() {
    let changed = crate::mem::take_repaged();
    if changed.is_empty() {
        return;
    }
    let ranges: Vec<(u64, u64)> = changed.iter().map(|&(a, b)| (a, b - a)).collect();
    super::mirror::invalidate(&ranges);
    let mut g = SPANS.write();
    for s in g.iter_mut().filter(|s| s.valid) {
        if changed.iter().any(|&(a, b)| a < s.end && s.start < b) {
            // Kept alive: recorded work may still reference it.
            s.valid = false;
            if IMPORTS_STALE.fetch_add(1, Ordering::Relaxed) < 20 {
                crate::info!("gpu: guest range {:#x}..{:#x} imported for the GPU was remapped (imported again on its next use)", s.start, s.end);
            }
        }
    }
    SPAN_GEN.fetch_add(1, Ordering::AcqRel);
}

/// Imported buffer covering guest [base, base+size): (buffer, window start, window end).
fn span(base: u64, size: u64) -> Option<(vk::Buffer, u64, u64)> {
    revalidate_spans();
    let generation = SPAN_GEN.load(Ordering::Acquire);
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
    let found = SPANS.read().iter().find(|s| s.valid && s.start <= base && end <= s.end).map(|s| (s.buffer, s.start, s.end));
    let r = found.or_else(|| import_span(base, size));
    if let Some((b, st, e)) = r {
        SPAN_CACHE.with(|c| {
            let mut c = c.borrow_mut();
            if SPAN_GEN.load(Ordering::Acquire) == generation {
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

/// Imports the guest window around [base, base+size) (another queue may
/// have done so meanwhile).
fn import_span(base: u64, size: u64) -> Option<(vk::Buffer, u64, u64)> {
    let mut g = SPANS.write();
    if let Some(s) = g.iter().find(|s| s.valid && s.start <= base && base + size <= s.end) {
        return Some((s.buffer, s.start, s.end));
    }
    let (vs, ve, pieces) = mapping_run(base, base & !(WINDOW - 1), (base + size).next_multiple_of(WINDOW))?;
    if base + size > ve {
        return None;
    }
    {
        // Diagnostics: how the title mapped memory the GPU uses.
        let st = crate::mem::MEM.lock();
        if let Some(v) = st.find(base) {
            crate::debug!("gpu: importing for {base:#x}+{size:#x}: mapping {:#x}+{:#x} {:?} prot {:#x} '{}'", v.start, v.size, v.kind, v.prot, v.name);
        }
    }
    let a = fixed().import_align;
    let start = vs & !(a - 1);
    let end = ve.next_multiple_of(a).min(ve);
    if start < vs || end <= start {
        return None;
    }
    let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::INDIRECT_BUFFER;
    // The driver pins the pages for writing: write-tracked (read-only)
    // pages must be writable while importing (their regions turn dirty
    // and are re-armed on their next use).
    let _w = super::track::protected().then(|| super::track::host_write(start, end - start));
    let t0 = std::time::Instant::now();
    match ctx().import_host(start, end - start, usage) {
        Ok((buffer, _mem)) => {
            let ms = t0.elapsed().as_millis();
            if ms >= 50 {
                crate::info!("gpu: importing guest {start:#x}..{end:#x} ({} MiB) took {ms} ms", (end - start) >> 20);
            }
            g.push(Span { start, end, buffer, valid: true });
            SPAN_GEN.fetch_add(1, Ordering::AcqRel);
            IMPORTS.fetch_add(1, Ordering::Relaxed);
            IMPORT_BYTES.fetch_add(end - start, Ordering::Relaxed);
            if pieces > 1 {
                IMPORTS_MULTI.fetch_add(1, Ordering::Relaxed);
            }
            Some((buffer, start, end))
        }
        Err(e) => {
            crate::warn!("gpu: importing guest {start:#x}..{end:#x} failed: {e}");
            None
        }
    }
}

/// The compute pipeline of a prepared shader (created on first use, outside
/// the map's lock; a pipeline another queue created meanwhile wins).
fn compute_pipe(p: &Prepared) -> Result<(vk::Pipeline, vk::PipelineLayout, vk::DescriptorSetLayout), String> {
    let key = p.tr.key;
    if let Some(x) = PIPES.lock().as_ref().and_then(|m| m.get(&key)) {
        return Ok((x.pipeline, x.layout, x.set_layout));
    }
    let t0 = std::time::Instant::now();
    let pipe = compute_pipeline(&p.tr.out)?;
    PIPES_CREATED[0].fetch_add(1, Ordering::Relaxed);
    PIPES_NS[0].fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
    super::pstats::report(pipe.pipeline, &format!("cs {:#x}", p.addr));
    let ms = t0.elapsed().as_millis();
    if ms >= 200 {
        crate::info!("gpu: compute pipeline for cs {:#x} ({} SPIR-V words) took {ms} ms", p.addr, p.tr.out.spirv.len());
    }
    Ok(keep_pipe(&PIPES, key, pipe))
}

/// Graphics pipeline cached under a state hash, if any.
fn gfx_pipe(hkey: u64) -> Option<(vk::Pipeline, vk::PipelineLayout, vk::DescriptorSetLayout)> {
    GFX_PIPES.lock().as_ref().and_then(|m| m.get(&hkey)).map(|x| (x.pipeline, x.layout, x.set_layout))
}

fn keep_gfx_pipe(hkey: u64, pipe: Pipe) -> (vk::Pipeline, vk::PipelineLayout, vk::DescriptorSetLayout) {
    keep_pipe(&GFX_PIPES, hkey, pipe)
}

/// Stores a newly created pipeline unless another queue stored one under
/// the same key meanwhile (then the new one is destroyed unused).
fn keep_pipe(map: &Mutex<Option<HashMap<u64, Pipe>>>, key: u64, pipe: Pipe) -> (vk::Pipeline, vk::PipelineLayout, vk::DescriptorSetLayout) {
    let mut g = map.lock();
    let m = g.get_or_insert_with(HashMap::new);
    if let Some(x) = m.get(&key) {
        unsafe {
            let d = &ctx().device;
            d.destroy_pipeline(pipe.pipeline, None);
            d.destroy_pipeline_layout(pipe.layout, None);
            d.destroy_descriptor_set_layout(pipe.set_layout, None);
        }
        return (x.pipeline, x.layout, x.set_layout);
    }
    let r = (pipe.pipeline, pipe.layout, pipe.set_layout);
    m.insert(key, pipe);
    r
}

/// Creates a compute pipeline (and its layouts) for a translated shader.
fn compute_pipeline(out: &rdna::translate::Output) -> Result<Pipe, String> {
    let c = ctx();
    {
        unsafe {
            let bindings: Vec<vk::DescriptorSetLayoutBinding> = out
                .bindings
                .iter()
                .map(|b| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(b.binding)
                        .descriptor_type(descriptor_type(b.kind))
                        .descriptor_count(binding_count(b.kind))
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                })
                .collect();
            let set_layout = c.device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None).map_err(|e| format!("set layout: {e}"))?;
            let layouts = [set_layout];
            let layout = c.device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts), None).map_err(|e| format!("pipeline layout: {e}"))?;
            let module = c.device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&out.spirv), None).map_err(|e| format!("shader module: {e}"))?;
            let stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(c"main");
            let ci = vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout).flags(vk::PipelineCreateFlags::DISPATCH_BASE | super::pstats::flags());
            let pipeline = c.device.create_compute_pipelines(super::pcache::get(), &[ci], None).map_err(|(_, e)| format!("compute pipeline: {e}"));
            c.device.destroy_shader_module(module, None);
            super::pcache::created();
            Ok(Pipe { pipeline: pipeline?[0], layout, set_layout })
        }
    }
}

/// `--pipeline-test <capture.txt>`: translates every compute shader of a
/// frame capture, creates its Vulkan pipeline and runs one thread group with
/// all buffers bound to zeros (images: skipped), reporting the time of each
/// step: driver acceptance and compile cost of the translator's output (the
/// driver may defer compilation to the first use).
pub fn pipeline_test(capture: &str) -> i32 {
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
        let t0 = std::time::Instant::now();
        let p = match compute_pipeline(out) {
            Ok(p) => p,
            Err(e) => {
                failed += 1;
                crate::warn!("pipeline test: cs {:#x}: {e}", it.addr);
                continue;
            }
        };
        let tc = t0.elapsed().as_millis();
        create_ms += tc;
        let buffers_only = out.bindings.iter().all(|b| descriptor_type(b.kind) == vk::DescriptorType::STORAGE_BUFFER);
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
    crate::info!("pipeline test: {ok} compute pipelines created ({create_ms} ms), {} dispatched ({run_ms} ms), {skipped} with images not dispatched, {failed} failed", ok - skipped);
    if failed > 0 { 1 } else { 0 }
}

fn descriptor_type(k: BindingKind) -> vk::DescriptorType {
    match k {
        BindingKind::SgprData | BindingKind::Buffer(_) | BindingKind::Gds | BindingKind::Debug | BindingKind::Watchdog | BindingKind::LodStats => vk::DescriptorType::STORAGE_BUFFER,
        BindingKind::Image(_) | BindingKind::ImageArray(_) => vk::DescriptorType::SAMPLED_IMAGE,
        BindingKind::StorageImage(_) => vk::DescriptorType::STORAGE_IMAGE,
        BindingKind::Sampler(_) => vk::DescriptorType::SAMPLER,
    }
}

/// Descriptors in a binding (image tables are arrays).
fn binding_count(k: BindingKind) -> u32 {
    if matches!(k, BindingKind::ImageArray(_)) { rdna::translate::IMAGE_TABLE_LEN } else { 1 }
}

/// Dispatches are ordered by the title's own synchronization (partial
/// flushes, cache actions, end-of-pipe events) instead of a full barrier
/// after every dispatch: consecutive independent dispatches overlap on the
/// GPU as on the console (`DS_LAZY_BARRIERS=0`: a barrier after each).
fn lazy_barriers() -> bool {
    super::tune::lazy_barriers()
}

fn global_barrier(cmd: vk::CommandBuffer) {
    let mb = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
        .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
    unsafe { ctx().device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::ALL_COMMANDS, vk::DependencyFlags::empty(), &[mb], &[], &[]) };
}

/// SGPR data ring space per slot.
const RING_SEG: u64 = 4 << 20;
/// Descriptor pool of a slot: sets, then descriptors of each POOL_TYPES type.
const POOL_LIMITS: [u32; 5] = [4096, 32768, 8192, 2048, 8192];
const POOL_TYPES: [vk::DescriptorType; 4] = [vk::DescriptorType::STORAGE_BUFFER, vk::DescriptorType::SAMPLED_IMAGE, vk::DescriptorType::STORAGE_IMAGE, vk::DescriptorType::SAMPLER];

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

/// Descriptor writes of one shader stage.
struct Bound {
    bufs: Vec<(u32, vk::DescriptorBufferInfo)>,
    imgs: Vec<(u32, vk::DescriptorType, vk::DescriptorImageInfo)>,
    /// Sampled image arrays (image tables).
    img_arrays: Vec<(u32, Vec<vk::DescriptorImageInfo>)>,
    /// Guest ranges the stage writes.
    writes: Vec<(u64, u64)>,
    /// Those described by buffer descriptors (not unbounded pointers).
    sized_writes: Vec<(u64, u64)>,
    /// Images it samples or stores to.
    images: Vec<vk::Image>,
    /// Buffers bound through a VRAM shadow (copied in before, out after).
    shadows: Vec<Shadow>,
    /// Written buffer resources and where their result is after the stage:
    /// (guest base, size, buffer, offset of the base) - mirror refreshes.
    write_src: Vec<(u64, u64, vk::Buffer, u64)>,
    /// Where each buffer binding was placed (DS_PROFILE_GPU): (Placement,
    /// bound bytes, written).
    placements: Vec<(Placement, u64, bool)>,
}

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
}

const PLACEMENTS: [Placement; 10] = [Placement::Host, Placement::Snapshot, Placement::PreCopy, Placement::Resident, Placement::Mirror, Placement::Shadow, Placement::ShadowReused, Placement::None, Placement::Scratch, Placement::Home];

/// Video memory written instead of large unshadowed buffers (diagnosis,
/// `tune scratch_writes`).
const SCRATCH_BYTES: u64 = 256 << 20;
static SCRATCH: std::sync::OnceLock<Option<vk::Buffer>> = std::sync::OnceLock::new();

fn scratch_buffer() -> Option<vk::Buffer> {
    *SCRATCH.get_or_init(|| ctx().device_buffer(SCRATCH_BYTES, vk::BufferUsageFlags::STORAGE_BUFFER).ok().map(|(b, _)| b))
}

/// Buffer placements and workgroups per compute shader (DS_PROFILE_GPU=1).
#[derive(Default)]
struct ShaderPlacements {
    dispatches: u64,
    groups: u64,
    indirect: u64,
    /// Per binding: [count per Placement], total bytes, written count.
    bindings: Vec<([u64; 10], u64, u64)>,
}

static SHADER_PLACEMENTS: Mutex<Option<HashMap<u64, ShaderPlacements>>> = Mutex::new(None);

fn note_placements(addr: u64, groups: [u32; 3], indirect: bool, placements: &[(Placement, u64, bool)]) {
    let mut g = SHADER_PLACEMENTS.lock();
    let e = g.get_or_insert_with(HashMap::new).entry(addr).or_default();
    e.dispatches += 1;
    if indirect {
        e.indirect += 1;
    } else {
        e.groups += groups.iter().map(|&x| x as u64).product::<u64>();
    }
    if e.bindings.len() < placements.len() {
        e.bindings.resize(placements.len(), ([0; 10], 0, 0));
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
}

/// Device-local shadow space per slot (working and pristine copies).
const SHADOW_BYTES: u64 = 32 << 20;
/// Largest buffer shadowed (larger atomic targets stay in host memory).
const SHADOW_MAX: u64 = 4 << 20;

static SHADOWED: AtomicU64 = AtomicU64::new(0);
/// Indirect dispatches whose arguments the GPU read (`Exec::dispatch_gpu_indirect`).
static GPU_INDIRECT: AtomicU64 = AtomicU64::new(0);
/// CP memory writes performed by the GPU (`Exec::gpu_cp_write`).
static GPU_CP_WRITES: AtomicU64 = AtomicU64::new(0);
/// Labels written GPU-side in batches not yet submitted (address & !7 ->
/// the owning queue's submit request): a command processor about to block on
/// one asks the owner to submit, which registers its promise.
static UNSUBMITTED_LABELS: Mutex<Option<HashMap<u64, std::sync::Arc<std::sync::atomic::AtomicBool>>>> = Mutex::new(None);
/// A batch holding unsubmitted labels is submitted after recording this long
/// (CPU threads of the title may poll them).
const LABEL_LATENCY: std::time::Duration = std::time::Duration::from_millis(2);

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
static RESIDENT_READS: AtomicU64 = AtomicU64::new(0);
static MERGES: AtomicU64 = AtomicU64::new(0);
static EARLY_MERGES: AtomicU64 = AtomicU64::new(0);
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
static PRE_REUSED: AtomicU64 = AtomicU64::new(0);
static PRE_FULL: AtomicU64 = AtomicU64::new(0);
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
            let pipeline = c.device.create_compute_pipelines(super::pcache::get(), &[ci], None).map_err(|(_, e)| format!("merge pipeline: {e}"));
            c.device.destroy_shader_module(module, None);
            Ok(MergePipe { pipeline: pipeline?[0], layout, set_layout })
        }
    })
    .as_ref()
    .map_err(|e| e.clone())
}

/// The textures a stage samples, with a coarse bound of their guest bytes
/// (16 per texel).
fn texture_ranges(p: &Prepared) -> Vec<((u64, u64), super::desc::Tsharp)> {
    let coarse = |t: &super::desc::Tsharp| ((t.base, (t.width as u64 * t.height as u64 * t.depth.max(1) as u64 * 16).min(1 << 30)), *t);
    p.resources
        .iter()
        .flat_map(|r| match r {
            Resolved::Image(t) => vec![coarse(t)],
            Resolved::ImageTable(e) => e.iter().flatten().map(coarse).collect(),
            _ => Vec::new(),
        })
        .collect()
}
/// Most batch slots per queue (`tune slots`, default 4): one batch recording
/// while the batches of the others execute. A queue reuses a slot only after
/// its batch completed, so more slots let a command processor run further
/// ahead of the GPU. Slots are created when first used.
pub const MAX_SLOTS: usize = 8;

/// Command-processor effect performed after GPU work completes.
pub type Action = Box<dyn FnOnce() + Send>;

/// A submitted batch (or action-only entry) awaiting completion.
struct Batch {
    queries: vk::QueryPool,
    seq: u64,
    /// Null for action-only entries.
    fence: vk::Fence,
    actions: Vec<Action>,
    checks: Vec<verify::Check>,
    shaders: Vec<u64>,
    submitted: std::time::Instant,
    name: String,
}

unsafe impl Send for Batch {}

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

fn done_seq() -> u64 {
    DONE_ATOMIC.load(Ordering::Acquire)
}

/// Highest completed batch sequence (all earlier ones completed too).
pub fn completed_seq() -> u64 {
    done_seq()
}

/// Guest ranges written by unfinished batches, by sequence number
/// (`ranges::SeqRanges`: per batch a coalesced union, so the title's
/// thousands of writes per batch to the same buffers stay a few entries and a
/// lookup costs one ordered search per unfinished batch); `latest_seq` lets
/// lookups return at once when every noted batch has completed.
struct RangeLog {
    log: Mutex<super::ranges::SeqRanges>,
    latest_seq: AtomicU64,
    /// Per bucket of 64 KiB guest pages (hashed), the latest batch that
    /// writes into one of them: a lookup whose pages' buckets all hold
    /// completed batches is answered without the log's lock (the command
    /// processors check every draw's descriptor reads; ~1.75 us per draw
    /// with the lock, run 20261011-133502). Never lowered: completion
    /// clears a bucket implicitly (its batch <= done).
    pages: [AtomicU64; PAGE_BUCKETS],
}

const PAGE_BUCKETS: usize = 1 << 16;
const PAGE_SHIFT: u32 = 16;

fn page_bucket(page: u64) -> usize {
    (page.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> (64 - 16)) as usize
}

impl RangeLog {
    const fn new() -> RangeLog {
        RangeLog { log: Mutex::new(super::ranges::SeqRanges::new()), latest_seq: AtomicU64::new(0), pages: [const { AtomicU64::new(0) }; PAGE_BUCKETS] }
    }

    /// Marks the page buckets of [a, a+n) as written by batch `seq`.
    fn mark_pages(&self, seq: u64, a: u64, n: u64) {
        if n == 0 {
            return;
        }
        let (first, last) = (a >> PAGE_SHIFT, (a + n - 1) >> PAGE_SHIFT);
        if last - first >= PAGE_BUCKETS as u64 {
            for b in self.pages.iter() {
                b.fetch_max(seq, Ordering::SeqCst);
            }
            return;
        }
        for p in first..=last {
            self.pages[page_bucket(p)].fetch_max(seq, Ordering::SeqCst);
        }
    }

    /// Whether some unfinished batch may write into one of `reads` (false:
    /// certainly none).
    fn may_overlap(&self, reads: &[(u64, u64)], done: u64) -> bool {
        reads.iter().any(|&(a, n)| {
            if n == 0 {
                return false;
            }
            let (first, last) = (a >> PAGE_SHIFT, (a + n - 1) >> PAGE_SHIFT);
            if last - first >= 64 {
                return true;
            }
            (first..=last).any(|p| self.pages[page_bucket(p)].load(Ordering::SeqCst) > done)
        })
    }

    fn note(&self, seq: u64, w: (u64, u64)) {
        if w.1 == 0 {
            return;
        }
        self.mark_pages(seq, w.0, w.1);
        self.latest_seq.fetch_max(seq, Ordering::SeqCst);
        let done = done_seq();
        let mut g = self.log.lock();
        g.prune(done);
        g.insert(seq, w.0, w.1);
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
        let mut g = self.log.lock();
        g.prune(done);
        g.insert_all(seq, writes);
    }

    /// The latest unfinished batch that writes into `reads`.
    fn latest(&self, reads: &[(u64, u64)]) -> Option<u64> {
        let done = done_seq();
        if self.latest_seq.load(Ordering::SeqCst) <= done || !self.may_overlap(reads, done) {
            return None;
        }
        self.log.lock().latest(reads, done)
    }
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
/// CP reads that waited for another queue's batch (diagnostics).
pub static FOREIGN_WAITS: AtomicU64 = AtomicU64::new(0);

/// A deferred CP copy (performed with its batch's actions) reads `src`:
/// later GPU work binding it (it may overwrite it) is submitted after the
/// copy ran.
pub fn note_cp_source(seq: u64, src: (u64, u64)) {
    CP_WRITES.note(seq, src);
}

/// Blocks until batch `seq` and all earlier ones completed (with their actions).
pub fn wait_done(seq: u64) {
    let mut d = DONE.lock();
    while *d < seq {
        if super::shutting_down() {
            return;
        }
        DONE_CV.wait_for(&mut d, std::time::Duration::from_millis(100));
    }
}

/// Queues a batch: `submit` runs (the vkQueueSubmit) under the queue lock so
/// sequence order is submission order.
fn push_batch(fence: vk::Fence, queries: vk::QueryPool, actions: Vec<Action>, checks: Vec<verify::Check>, shaders: Vec<u64>, name: &str, submit: impl FnOnce()) -> u64 {
    static STARTED: std::sync::Once = std::sync::Once::new();
    STARTED.call_once(|| {
        crate::thread::spawn_host("gpu-complete", 1 << 20, Box::new(completion_thread)).expect("gpu completion thread");
    });
    let mut q = BATCHES.lock();
    submit();
    let seq = NEXT_SEQ.fetch_add(1, Ordering::SeqCst);
    q.push_back(Batch { queries, seq, fence, actions, checks, shaders, submitted: std::time::Instant::now(), name: name.to_string() });
    BATCHES_CV.notify_one();
    seq
}

/// Waits for submitted batches in order, then performs their actions.
fn completion_thread() {
    crate::log::set_thread_name("gpu-complete");
    loop {
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
            let tf = std::time::Instant::now();
            let r = unsafe { ctx().device.wait_for_fences(&[b.fence], true, 10_000_000_000) };
            FENCE_NS.fetch_add(tf.elapsed().as_nanos() as u64, Ordering::Relaxed);
            if let Err(e) = r {
                if !DEVICE_LOST.swap(true, Ordering::SeqCst) {
                    let list: Vec<String> = b.shaders.iter().map(|a| format!("{a:#x}")).collect();
                    crate::error!("gpu[{}]: waiting for GPU work: {e}; GPU execution stops (batch shaders: {}); checkpoints: {}", b.name, list.join(" "), super::vk::checkpoint_report());
                }
            }
            FLUSHES.fetch_add(1, Ordering::Relaxed);
            collect_timestamps(b.queries, &b.shaders, &b.name);
            check_watchdog(&b.name, &b.shaders);
            let ms = b.submitted.elapsed().as_millis();
            if ms >= 250 {
                let list: Vec<String> = b.shaders.iter().map(|a| format!("{a:#x}")).collect();
                crate::info!("gpu[{}]: batch of {} dispatch(es) took {ms} ms (shaders {})", b.name, b.shaders.len(), list.join(" "));
            }
        }
        let ta = std::time::Instant::now();
        for a in b.actions {
            a();
        }
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
    }
}

/// A draw's image keys (`extra`: its attachments, then every texture of
/// `stages`) and whether one of them may have a fast clear, reload or block
/// copy pending, under one image-cache lock (a lock per texture and per
/// query took ~0.9 us per draw, run 20261011-135122).
pub(crate) fn image_hazards(stages: &[&Prepared], extra: &[super::images::ImageKey]) -> (Vec<super::images::ImageKey>, bool) {
    super::images::with(|im| {
        let mut v: Vec<super::images::ImageKey> = extra.to_vec();
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
        let pending = im.clears_pending(&v) || im.overwrites_pending(&v) || im.block_copies_possible(&v);
        (v, pending)
    })
}

/// Image cache keys of the images a prepared stage binds (fast-clear checks).
pub(crate) fn image_keys(p: &Prepared) -> Vec<super::images::ImageKey> {
    let mut v = Vec::new();
    for (i, r) in p.resources.iter().enumerate() {
        if let Resolved::Image(t) = r {
            let storage = p.tr.plan.resources.get(i).is_some_and(|x| x.written);
            if let Some(k) = super::images::with(|im| im.resolved_key(t, storage)) {
                v.push(k);
            }
        }
        if let Resolved::ImageTable(e) = r {
            for t in e.iter().flatten() {
                if let Some(k) = super::images::with(|im| im.resolved_key(t, false)) {
                    v.push(k);
                }
            }
        }
    }
    v
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
    crate::warn!(
        "gpu[{queue}]: loop watchdog stopped {n} invocation(s) of shader {addr:#x} after {} blocks; its results are wrong (batch shaders {}){saved}",
        rdna::translate::LOOP_LIMIT,
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
    /// The SGPR data rings (RING_SEG each): system memory and video memory
    /// (allocated when `tune ring_vram` first chooses it).
    rings: [Option<(vk::Buffer, *mut u8)>; 2],
    /// Read-only buffer snapshots in video memory (SNAP_SEG, allocated on
    /// first use).
    snap: Option<(vk::Buffer, *mut u8)>,
    /// Device-local shadow space (SHADOW_BYTES: working and pristine copies).
    shadow: vk::Buffer,
    /// The batch's prologue: copies of read-only buffers into `pre_buf`
    /// (PRE_SEG, allocated on first use), submitted before `cmd`.
    pre: vk::CommandBuffer,
    pre_buf: Option<vk::Buffer>,
    /// Video-memory homes the slot's batch binds (kept until it completes).
    homes: Vec<std::sync::Arc<super::homes::HomeBuf>>,
}

impl Slot {
    fn new(pool: vk::CommandPool) -> Result<Slot, String> {
        let c = ctx();
        let sizes: Vec<vk::DescriptorPoolSize> = POOL_TYPES.iter().zip(&POOL_LIMITS[1..]).map(|(&ty, &n)| vk::DescriptorPoolSize { ty, descriptor_count: n }).collect();
        unsafe {
            let cmds = c.device.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(pool).command_buffer_count(2)).map_err(|e| e.to_string())?;
            let (cmd, pre) = (cmds[0], cmds[1]);
            // SGPR data in system memory: a copy in video memory, read by
            // every invocation, made no measurable difference (`tune ring_vram`).
            let (ring, _mem, ring_ptr) = c.host_buffer(RING_SEG, vk::BufferUsageFlags::STORAGE_BUFFER)?;
            let (shadow, _smem) = c.device_buffer(SHADOW_BYTES, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST)?;
            Ok(Slot {
                cmd,
                fence: c.device.create_fence(&vk::FenceCreateInfo::default(), None).map_err(|e| e.to_string())?,
                desc_pool: c.device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(POOL_LIMITS[0]).pool_sizes(&sizes), None).map_err(|e| e.to_string())?,
                seq: 0,
                queries: if profiling() {
                    c.device.create_query_pool(&vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(STAMPS * MAX_TIMED + 4), None).map_err(|e| e.to_string())?
                } else {
                    vk::QueryPool::null()
                },
                rings: [Some((ring, ring_ptr)), None],
                snap: None,
                shadow,
                pre,
                pre_buf: None,
                homes: Vec::new(),
            })
        }
    }
}

const MAX_TIMED: u32 = 4096;
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

/// Per-command GPU timestamps (DS_PROFILE_GPU=1; `DS_PROFILE_GPU_STAMPS=0`
/// keeps only the batch ones): BOTTOM_OF_PIPE stamps around every draw and
/// dispatch serialize the GPU, which inflates the time of small commands.
pub(super) fn command_stamps() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_PROFILE_GPU_STAMPS").ok().as_deref() != Some("0"))
}

/// Timestamps per timed dispatch (all BOTTOM_OF_PIPE): before the shadow
/// copies in, before the dispatch, after it, after the copies out.
const STAMPS: u32 = 4;

/// GPU time per shader (DS_PROFILE_GPU=1): address -> [dispatches, dispatch
/// ns, copy-in ns, copy-out ns].
static SHADER_TIME: Mutex<Option<HashMap<u64, [u64; 4]>>> = Mutex::new(None);
/// GPU time between timed dispatches (draws, CP copies, barriers), ns.
static GAP_NS: AtomicU64 = AtomicU64::new(0);

/// Command-processor time per recorded command by phase (DS_PROFILE_GPU),
/// ns: dispatches [prepare, waits + textures + pipeline, bindings, total,
/// count], draws [prepare + shader-read waits, textures + targets +
/// pipeline, bindings, total, count].
pub(super) static CS_PHASES: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];
pub(super) static DRAW_PHASES: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];

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

/// Finer command-processor time per draw (DS_PROFILE_GPU), ns per step of
/// `DRAW_STEP_NAMES`; the count is DRAW_PHASES[4].
pub(super) static DRAW_STEPS: [AtomicU64; 18] = [const { AtomicU64::new(0) }; 18];
pub(super) const DRAW_STEP_NAMES: [&str; 18] = [
    "vs prepare", "ps prepare", "shader-read sync", "texture sync", "indices", "color targets", "depth target", "pipeline",
    "image hazards", "reserve", "bind vs", "bind ps", "set alloc", "set write", "rendering instance", "dynamic state",
    "index and draw", "write tracking",
];

/// The same per dispatch (all queues), steps of `CS_STEP_NAMES`; the count
/// is CS_PHASES[4].
pub(super) static CS_STEPS: [AtomicU64; 16] = [const { AtomicU64::new(0) }; 16];
pub(super) const CS_STEP_NAMES: [&str; 16] = [
    "prepare", "shader-read sync", "texture sync", "image hazards", "pipeline", "reserve", "bind", "sgpr data", "set alloc",
    "image notes", "set write", "shadow copies", "dispatch", "shadows after", "mirrors", "write tracking",
];

/// Lap timer of a command's steps (profiling only).
pub(super) struct Laps {
    last: Option<std::time::Instant>,
    steps: &'static [AtomicU64],
}

impl Laps {
    pub(super) fn new() -> Laps {
        Laps { last: profiling().then(std::time::Instant::now), steps: &DRAW_STEPS }
    }
    fn for_dispatch() -> Laps {
        Laps { last: profiling().then(std::time::Instant::now), steps: &CS_STEPS }
    }
    /// Adds the time since the last mark to step `k`.
    pub(super) fn mark(&mut self, k: usize) {
        if let Some(t) = self.last {
            let now = std::time::Instant::now();
            self.steps[k].fetch_add((now - t).as_nanos() as u64, Ordering::Relaxed);
            self.last = Some(now);
        }
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
pub(super) static SYNC_STEPS: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

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
        "    CP read checks: {} ({:.1} ranges each; us each: homes {:.2}, this queue's writes {:.2}, other queues' writes {:.2})\n",
        SYNC_STEPS[4].load(Ordering::Relaxed),
        SYNC_STEPS[5].load(Ordering::Relaxed) as f64 / n as f64,
        us(0),
        us(1),
        us(2)
    )
}

fn draw_steps_line() -> String {
    let n = DRAW_PHASES[4].load(Ordering::Relaxed).max(1);
    let parts: Vec<String> = DRAW_STEP_NAMES.iter().zip(DRAW_STEPS.iter()).map(|(name, c)| format!("{name} {:.2}", c.load(Ordering::Relaxed) as f64 / n as f64 / 1000.0)).collect();
    let n = CS_PHASES[4].load(Ordering::Relaxed).max(1);
    let cs: Vec<String> = CS_STEP_NAMES.iter().zip(CS_STEPS.iter()).map(|(name, c)| format!("{name} {:.2}", c.load(Ordering::Relaxed) as f64 / n as f64 / 1000.0)).collect();
    format!("    command processor per draw by step (us): {}\n    command processor per dispatch by step (us): {}\n", parts.join(", "), cs.join(", "))
}

/// GPU time inside batches (first to last command), ns.
static BATCH_NS: AtomicU64 = AtomicU64::new(0);
/// GPU idle time between consecutive batches (one Vulkan queue runs them in
/// submission order): (end timestamp of the last batch, per queue of the
/// batch that ended the gap: [ns, gaps, gaps over 1 ms]).
static IDLE: Mutex<(u64, Option<HashMap<String, [u64; 3]>>)> = Mutex::new((0, None));

/// Reads a completed batch's dispatch timestamps into SHADER_TIME.
fn collect_timestamps(pool: vk::QueryPool, shaders: &[u64], queue: &str) {
    if pool == vk::QueryPool::null() {
        return;
    }
    let c = ctx();
    let mut be = [0u64; 2];
    if unsafe { c.device.get_query_pool_results(pool, STAMPS * MAX_TIMED, &mut be, vk::QueryResultFlags::TYPE_64) }.is_ok() {
        let period = c.limits.timestamp_period as f64;
        BATCH_NS.fetch_add((be[1].saturating_sub(be[0]) as f64 * period) as u64, Ordering::Relaxed);
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
    }
    let mut pre = [0u64; 2];
    if unsafe { c.device.get_query_pool_results(pool, STAMPS * MAX_TIMED + 2, &mut pre, vk::QueryResultFlags::TYPE_64) }.is_ok() {
        PRE_NS.fetch_add((pre[1].saturating_sub(pre[0]) as f64 * c.limits.timestamp_period as f64) as u64, Ordering::Relaxed);
    }
    let n = (shaders.len() as u32).min(MAX_TIMED);
    if n == 0 || !command_stamps() {
        return;
    }
    let mut data = vec![0u64; (STAMPS * n) as usize];
    let r = unsafe { c.device.get_query_pool_results(pool, 0, &mut data, vk::QueryResultFlags::TYPE_64) };
    if r.is_err() {
        return;
    }
    let period = c.limits.timestamp_period as f64;
    let ns = |a: u64, b: u64| (b.saturating_sub(a) as f64 * period) as u64;
    let mut g = SHADER_TIME.lock();
    let m = g.get_or_insert_with(HashMap::new);
    for (i, &a) in shaders.iter().take(n as usize).enumerate() {
        let q = &data[STAMPS as usize * i..STAMPS as usize * (i + 1)];
        let e = m.entry(a).or_insert([0; 4]);
        e[0] += 1;
        e[1] += ns(q[1], q[2]);
        e[2] += ns(q[0], q[1]);
        e[3] += ns(q[2], q[3]);
        if i > 0 {
            GAP_NS.fetch_add(ns(data[STAMPS as usize * i - 1], q[0]), Ordering::Relaxed);
        }
    }
}

/// Clears the dispatch timings (`profile reset` control command).
pub fn reset_shader_profile() {
    if let Some(m) = SHADER_TIME.lock().as_mut() {
        m.clear();
    }
    *SHADER_PLACEMENTS.lock() = None;
    shaders::reset_repeats();
    *QUEUE_WAITS.lock() = None;
    for c in DRAW_STEPS.iter().chain(CS_STEPS.iter()).chain(SYNC_STEPS.iter()).chain(WAIT_NS.iter()) {
        c.store(0, Ordering::Relaxed);
    }
    for c in CS_PHASES.iter().chain(DRAW_PHASES.iter()) {
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
    if let Some(idle) = IDLE.lock().1.as_ref() {
        let mut v: Vec<(&String, &[u64; 3])> = idle.iter().collect();
        v.sort_by(|a, b| b.1[0].cmp(&a.1[0]));
        let total: u64 = v.iter().map(|x| x.1[0]).sum();
        let list: Vec<String> = v.iter().map(|(q, x)| format!("{q} {} ms ({} gaps, {} over 1 ms)", x[0] / 1_000_000, x[1], x[2])).collect();
        s.push_str(&format!("    gpu idle between batches {} ms, before batches of: {}\n", total / 1_000_000, list.join(", ")));
    }
    s.push_str(&phases_line("draw", &DRAW_PHASES, ["prepare and shader-read waits", "textures, targets, pipeline", "bindings"]));
    s.push_str(&draw_steps_line());
    s.push_str(&sync_steps_line());
    s.push_str(&shaders::repeats_line());
    s.push_str(&phases_line("dispatch", &CS_PHASES, ["prepare and waits", "textures, pipeline", "bindings"]));
    {
        let n: Vec<u64> = SYNC_WHY.iter().zip(SYNC_WHY_AT_RESET.iter()).map(|(c, r)| c.load(Ordering::Relaxed).saturating_sub(r.load(Ordering::Relaxed))).collect();
        s.push_str(&format!(
            "    command-processor waits for GPU work: {} CP reads, {} dispatch shader reads, {} dispatch textures, {} draw shader reads, {} draw textures, {} draw indices\n",
            n[0], n[1], n[2], n[3], n[4], n[5]
        ));
    }
    if let Some(w) = SUBMIT_WHY.lock().as_ref() {
        let mut v: Vec<(&(String, &'static str), &u64)> = w.iter().collect();
        v.sort_by(|a, b| b.1.cmp(a.1));
        let list: Vec<String> = v.iter().map(|((q, why), n)| format!("{q} {why} {n}")).collect();
        s.push_str(&format!("    batches by what ended them: {}\n", list.join(", ")));
    }
    if let Some(st) = STARVED.lock().as_ref() {
        let mut v: Vec<(&String, &[u64; 3])> = st.iter().collect();
        v.sort_by(|a, b| b.1[1].cmp(&a.1[1]));
        let list: Vec<String> = v.iter().map(|(q, x)| format!("{q} {} submissions, idle {} ms, recorded in {} ms", x[0], x[1] / 1_000_000, x[2] / 1_000_000)).collect();
        s.push_str(&format!("    submissions to an idle GPU: {}\n", list.join(", ")));
    }
    let top = std::env::var("DS_PROFILE_TOP").ok().and_then(|x| x.parse().ok()).unwrap_or(15);
    for (a, x) in v.iter().take(top) {
        let n = x[0].max(1);
        if **a & DRAW_KEY != 0 {
            s.push_str(&format!("      draws with shader {:#x}: {} draws, {} us each
", **a & !DRAW_KEY, x[0], x[1] / 1000 / n));
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
    pool: vk::CommandPool,
    slots: Vec<Slot>,
    cur: usize,
    /// Slots that could be created (a failed creation caps them).
    slot_limit: usize,
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
    pres: super::fx::Map<(u64, u64), u64>,
    ring_align: u64,
    recording: bool,
    /// When the batch being recorded was begun.
    begun_at: std::time::Instant,
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
    /// The title synchronized (flush, cache action, end-of-pipe event)
    /// since the last full barrier: the next dispatch or draw waits.
    barrier_wanted: bool,
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
    /// Descriptor pool usage of the recording slot (see POOL_LIMITS).
    pool_use: [u32; 5],
    /// Shadow space used in the recording slot.
    shadow_used: u64,
    /// Shadows of this batch whose working copies hold newer data than
    /// guest memory (see Shadow).
    resident: Vec<Shadow>,
    /// The shaders whose reads the next `sync_for` checks (wait traces).
    sync_readers: [u64; 2],
    /// The read a traced wait collided with (the caller explains it).
    pub(super) sync_culprit: Option<(u64, u64)>,
}

unsafe impl Send for Exec {}

impl Exec {
    pub fn new(name: &str) -> Result<Exec, String> {
        let c = ctx();
        unsafe {
            let pool = c
                .device
                .create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(c.queue_family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER), None)
                .map_err(|e| e.to_string())?;
            let slots = vec![Slot::new(pool)?, Slot::new(pool)?];
            Ok(Exec {
                pool,
                slots,
                cur: 0,
                slot_limit: MAX_SLOTS,
                vram_ring_failed: false,
                ring_used: [0; 2],
                snap_used: 0,
                snap_failed: false,
                snaps: Default::default(),
                pre_used: 0,
                pre_open: false,
                pre_failed: false,
                pres: Default::default(),
                ring_align: c.limits.min_storage_buffer_offset_alignment.max(16),
                recording: false,
                begun_at: std::time::Instant::now(),
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
                barrier_wanted: false,
                dispatched: false,
                reported: HashMap::new(),
                name: name.to_string(),
                batch_shaders: Vec::new(),
                rendering: None,
                dyncache: Default::default(),
                pool_use: [0; 5],
                shadow_used: 0,
                resident: Vec::new(),
                sync_readers: [0; 2],
                sync_culprit: None,
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

    fn cmd(&self) -> vk::CommandBuffer {
        self.slots[self.cur].cmd
    }

    /// The recording slot's shadow buffer.
    fn shadow(&self) -> vk::Buffer {
        self.slots[self.cur].shadow
    }

    fn begin(&mut self) {
        if self.recording {
            return;
        }
        if self.cur == self.slots.len() {
            match Slot::new(self.pool) {
                Ok(slot) => self.slots.push(slot),
                Err(e) => {
                    crate::warn!("gpu[{}]: no batch slot {} ({e}); using {}", self.name, self.cur + 1, self.cur);
                    self.slot_limit = self.cur;
                    self.cur = 0;
                }
            }
        }
        // The slot's previous batch must have completed before reuse.
        let slot = &self.slots[self.cur];
        if slot.seq != 0 {
            if slot.seq > done_seq() {
                let t = std::time::Instant::now();
                wait_done(slot.seq);
                WAIT_NS[2].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
                WAIT_NS[3].fetch_add(1, Ordering::Relaxed);
                note_queue_wait(&self.name, 1, t.elapsed());
            }
            wait_done(slot.seq);
            let c = ctx();
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
            self.slots[self.cur].seq = 0;
            self.slots[self.cur].homes.clear();
        }
        unsafe {
            ctx().device.begin_command_buffer(self.cmd(), &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).ok();
            let qp = self.slots[self.cur].queries;
            if qp != vk::QueryPool::null() {
                ctx().device.cmd_reset_query_pool(self.cmd(), qp, 0, STAMPS * MAX_TIMED + 2);
            }
        }
        global_barrier(self.cmd());
        self.barrier_wanted = false;
        self.dispatched = false;
        let qp = self.slots[self.cur].queries;
        if qp != vk::QueryPool::null() {
            unsafe { ctx().device.cmd_write_timestamp(self.cmd(), vk::PipelineStageFlags::BOTTOM_OF_PIPE, qp, STAMPS * MAX_TIMED) };
        }
        self.recording = true;
        self.dyncache = Default::default();
        self.begun_at = std::time::Instant::now();
        self.pool_use = [0; 5];
        self.shadow_used = 0;
        self.resident.clear();
    }

    /// Submits the recorded batch without waiting for it (its actions run
    /// when it completes).
    /// Submits the recorded batch; `why` (profiling): what ended it.
    pub fn submit_for(&mut self, why: &'static str) {
        if !self.recording {
            return;
        }
        if super::shutting_down() {
            self.discard();
            return;
        }
        if profiling() {
            *SUBMIT_WHY.lock().get_or_insert_with(HashMap::new).entry((self.name.clone(), why)).or_default() += 1;
        }
        if let Err(e) = self.flush_resident(None) {
            self.report(format!("shadow write-back: {e}"));
        }
        if self.submit_after > done_seq() {
            // Deferred CP writes that this work may read come first.
            wait_done(self.submit_after);
        }
        self.end_rendering();
        let c = ctx();
        let i = self.cur;
        let (cmd, fence) = (self.slots[i].cmd, self.slots[i].fence);
        if self.slots[i].queries != vk::QueryPool::null() {
            unsafe { c.device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, self.slots[i].queries, STAMPS * MAX_TIMED + 1) };
        }
        let pre = self.end_pre();
        if pre.is_none() && self.slots[i].queries != vk::QueryPool::null() {
            // No prologue: its timestamps read as unavailable.
            unsafe { c.device.cmd_reset_query_pool(cmd, self.slots[i].queries, STAMPS * MAX_TIMED + 2, 2) };
        }
        unsafe {
            // Make shader writes to imported memory visible to the host.
            let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::MEMORY_WRITE).dst_access_mask(vk::AccessFlags::HOST_READ);
            c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::HOST, vk::DependencyFlags::empty(), &[mb], &[], &[]);
            c.device.end_command_buffer(cmd).ok();
        }
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
        let seq = push_batch(fence, self.slots[i].queries, actions, checks, std::mem::take(&mut self.batch_shaders), &self.name, || {
            if DEVICE_LOST.load(Ordering::Relaxed) {
                return;
            }
            let r = match pre {
                Some(p) => c.submit_cmds(&[p, cmd], &[], &[], fence),
                None => c.submit(cmd, &[], &[], fence),
            };
            if let Err(e) = r {
                if !DEVICE_LOST.swap(true, Ordering::SeqCst) {
                    crate::error!("gpu[{name}]: {e}; GPU execution stops; checkpoints: {}", super::vk::checkpoint_report());
                }
            }
        });
        self.slots[i].seq = seq;
        for (a, v, n) in std::mem::take(&mut self.pending_promises) {
            super::cp::promise(seq, a, v, n);
        }
        if let Some(m) = UNSUBMITTED_LABELS.lock().as_mut() {
            m.retain(|_, f| !std::sync::Arc::ptr_eq(f, &self.submit_request));
        }
        self.submit_request.store(false, Ordering::SeqCst);
        let writes = std::mem::take(&mut self.pending);
        self.pending_log.clear();
        INFLIGHT.note_ranges(seq, &writes);
        self.inflight.insert_all(seq, &writes);
        self.last_seq = seq;
        if has_actions {
            self.last_action_seq = seq;
        }
        self.recording = false;
        let n = (super::tune::slots() as usize).clamp(2, MAX_SLOTS).min(self.slot_limit);
        self.cur = (self.cur + 1) % n;
        self.ring_used = [0; 2];
        self.snap_used = 0;
        self.snaps.clear();
        self.pre_used = 0;
        self.pres.clear();
    }

    /// Drops the recorded batch (process shutdown: label waits were released,
    /// so the work may lack its inputs). The slot stays free.
    fn discard(&mut self) {
        self.end_rendering();
        unsafe { ctx().device.end_command_buffer(self.cmd()).ok() };
        self.actions.clear();
        self.checks.clear();
        self.pending.clear();
        self.pending_log.clear();
        self.pending_promises.clear();
        self.batch_shaders.clear();
        self.resident.clear();
        self.recording = false;
        self.ring_used = [0; 2];
        self.snap_used = 0;
        self.snaps.clear();
        let _ = self.end_pre();
        self.pre_used = 0;
        self.pres.clear();
    }

    /// The home of guest [base, base+size) for a binding (see `homes`):
    /// (buffer, offset of `base`, bytes from `base` to the home's end). The
    /// copies in the home needs are recorded first. A written binding too
    /// large for a VRAM shadow gets a home (`tune homes`; `create`: a sized
    /// buffer binding). A binding that only partly overlaps a home uses guest
    /// memory: the overlap's GPU-written parts are written back first.
    pub(super) fn home_bind(&mut self, base: u64, size: u64, written: bool, kind: HomeUse) -> Option<(vk::Buffer, u64, u64, std::sync::Arc<super::homes::HomeBuf>)> {
        if size == 0 {
            return None;
        }
        if !super::tune::homes() && super::homes::any() {
            self.homes_evict_all();
        }
        if kind == HomeUse::Sized && written && size > SHADOW_MAX && size <= super::homes::MAX && super::tune::homes() {
            super::homes::note_large_write(base, size);
        }
        if !super::homes::any() && !(kind == HomeUse::Index && super::tune::homes()) {
            return None;
        }
        let mut b = super::homes::bind(base, size, written);
        if b.is_none() && kind == HomeUse::Index && super::tune::homes() && super::homes::index_use(base, size) {
            b = super::homes::bind(base, size, written);
        }
        let Some(b) = b else {
            if kind == HomeUse::Unbounded && !written {
                // An unbounded pointer's window (up to UNBOUNDED_MAX) that
                // passes over a home: assumed not to read it (writing the
                // home's GPU-written parts back for every such binding cost
                // more than the home saved).
                if super::homes::overlaps(base, size) {
                    super::homes::UNBOUNDED_OVERLAPS.fetch_add(1, Ordering::Relaxed);
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
            }
            return None;
        };
        if !b.copy_in.is_empty() {
            self.begin();
            self.end_rendering();
            let c = ctx();
            let cmd = self.cmd();
            global_barrier(cmd);
            for &(a, n, off) in &b.copy_in {
                for (src, src_off, ga, len) in guest_pieces(a, n) {
                    unsafe { c.device.cmd_copy_buffer(cmd, src, b.buf.buf, &[vk::BufferCopy { src_offset: src_off, dst_offset: off + (ga - a), size: len }]) };
                    super::homes::COPIED_IN.fetch_add(len, Ordering::Relaxed);
                }
            }
            global_barrier(cmd);
            self.barrier_wanted = false;
            self.dispatched = false;
        }
        let r = (b.buf.buf, b.offset, b.avail, b.buf.clone());
        self.slots[self.cur].homes.push(b.buf);
        Some(r)
    }

    /// Writes every home's GPU-written parts back and drops the homes
    /// (`tune homes 0` while they exist).
    fn homes_evict_all(&mut self) {
        let parts = super::homes::take_all();
        if parts.is_empty() {
            return;
        }
        self.begin();
        self.end_rendering();
        let c = ctx();
        let cmd = self.cmd();
        global_barrier(cmd);
        for (a, n, hb, off) in parts {
            for (dst, dst_off, ga, len) in guest_pieces(a, n) {
                unsafe { c.device.cmd_copy_buffer(cmd, hb.buf, dst, &[vk::BufferCopy { src_offset: off + (ga - a), dst_offset: dst_off, size: len }]) };
                super::homes::WRITE_BACK_BYTES.fetch_add(len, Ordering::Relaxed);
            }
            self.slots[self.cur].homes.push(hb);
        }
        global_barrier(cmd);
        self.barrier_wanted = false;
        self.dispatched = false;
        crate::info!("gpu[{}]: homes written back and dropped (tune homes 0)", self.name);
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
        self.begin();
        self.end_rendering();
        let c = ctx();
        let cmd = self.cmd();
        global_barrier(cmd);
        let total: u64 = parts.iter().map(|p| p.1).sum();
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
        global_barrier(cmd);
        super::homes::WRITE_BACKS.fetch_add(1, Ordering::Relaxed);
        self.barrier_wanted = false;
        self.dispatched = false;
    }

    /// Ends the batch's copy prologue, if it was begun: the copies complete
    /// before the batch's work (which begins with a full barrier).
    fn end_pre(&mut self) -> Option<vk::CommandBuffer> {
        if !std::mem::take(&mut self.pre_open) {
            return None;
        }
        let (pre, qp) = (self.slots[self.cur].pre, self.slots[self.cur].queries);
        unsafe {
            global_barrier(pre);
            if qp != vk::QueryPool::null() {
                ctx().device.cmd_write_timestamp(pre, vk::PipelineStageFlags::BOTTOM_OF_PIPE, qp, STAMPS * MAX_TIMED + 3);
            }
            ctx().device.end_command_buffer(pre).ok();
        }
        Some(pre)
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
    pub(super) fn pre_copy(&mut self, base: u64, size: u64) -> Option<(vk::Buffer, u64, u64)> {
        if size == 0 || size > super::tune::pre_max() || self.pre_failed || super::homes::overlaps(base, size) {
            return None;
        }
        let r = [(base, size)];
        if self.overlaps_pending(&r) || CP_WRITES.latest(&r).is_some() {
            return None;
        }
        let al = base & !15;
        let bias = base - al;
        if let (Some(&at), Some(buf)) = (self.pres.get(&(base, size)), self.slots[self.cur].pre_buf) {
            PRE_REUSED.fetch_add(1, Ordering::Relaxed);
            return Some((buf, at, bias));
        }
        let n = (bias + size).next_multiple_of(4);
        let off = self.pre_used.next_multiple_of(self.ring_align);
        if off + n > PRE_SEG {
            PRE_FULL.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        let (src, start, _) = span(al, n)?;
        if self.slots[self.cur].pre_buf.is_none() {
            match ctx().device_buffer(PRE_SEG, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER) {
                Ok((b, _m)) => self.slots[self.cur].pre_buf = Some(b),
                Err(e) => {
                    crate::warn!("gpu[{}]: no video memory for batch-start copies ({e})", self.name);
                    self.pre_failed = true;
                    return None;
                }
            }
        }
        let buf = self.slots[self.cur].pre_buf?;
        let c = ctx();
        let (pre, qp) = (self.slots[self.cur].pre, self.slots[self.cur].queries);
        if !self.pre_open {
            unsafe {
                c.device.begin_command_buffer(pre, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).ok();
                if qp != vk::QueryPool::null() {
                    c.device.cmd_reset_query_pool(pre, qp, STAMPS * MAX_TIMED + 2, 2);
                }
            }
            // After all earlier submitted work.
            global_barrier(pre);
            if qp != vk::QueryPool::null() {
                unsafe { c.device.cmd_write_timestamp(pre, vk::PipelineStageFlags::BOTTOM_OF_PIPE, qp, STAMPS * MAX_TIMED + 2) };
            }
            self.pre_open = true;
        }
        unsafe { c.device.cmd_copy_buffer(pre, src, buf, &[vk::BufferCopy { src_offset: al - start, dst_offset: off, size: n }]) };
        self.pre_used = off + n;
        self.pres.insert((base, size), off);
        PRE_COPIES.fetch_add(1, Ordering::Relaxed);
        PRE_BYTES.fetch_add(n, Ordering::Relaxed);
        Some((buf, off, bias))
    }

    /// Performs the fast clears pending for `keys` before they are used: the
    /// title clears DCC and HTILE surfaces by writing only their metadata
    /// (e.g. zero-filling a HUD layer's DCC each frame); images here are
    /// never compressed, so the metadata is read back and the image cleared.
    /// Runs before a draw or dispatch reserves its slot (it flushes).
    pub(crate) fn resolve_fast_clears(&mut self, keys: &[super::images::ImageKey]) {
        if keys.is_empty() || !super::images::with(|im| im.clears_pending(keys)) {
            return;
        }
        // The metadata write may still be in flight.
        self.end_rendering();
        self.flush();
        let clears = super::images::with(|im| im.take_clears(keys));
        for c in clears {
            if !super::cp::guest_readable(c.meta, 4) {
                continue;
            }
            let v = unsafe { std::ptr::read_volatile(c.meta as *const u32) };
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
                unsafe { ctx().device.cmd_clear_depth_stencil_image(self.cmd(), c.image, vk::ImageLayout::GENERAL, &value, &[full]) };
                format!("depth {} stencil {}", value.depth, value.stencil)
            } else {
                match dcc_clear_color(v & 0xFF, k.format, c.clear) {
                    Ok(Some(color)) => {
                        self.begin();
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
            global_barrier(self.cmd());
            FAST_CLEARS.fetch_add(1, Ordering::Relaxed);
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
            for r in &p.resources {
                let Resolved::Image(t) = r else { continue };
                let Some(k) = super::images::Images::key_of(t) else { continue };
                let fam = super::images::Images::depth_family(k.format);
                if fam.is_empty() || k.layers <= 1 || k.ty3d {
                    continue;
                }
                let Some(old) = super::images::with(|im| im.depth_to_grow(k.base, k.width, k.height, fam, k.layers)) else { continue };
                self.end_rendering();
                self.flush();
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
        if keys.is_empty() || !super::images::with(|im| im.overwrites_pending(keys)) {
            return;
        }
        self.end_rendering();
        self.flush();
        super::images::with(|im| im.reload_overwritten(keys));
    }

    /// Copies blocks that GPU work stored through an uncompressed view into
    /// the block-compressed textures among `keys` (Images::take_block_copies),
    /// before the work that samples them.
    pub(crate) fn sync_block_textures(&mut self, keys: &[super::images::ImageKey]) {
        if keys.is_empty() {
            return;
        }
        let copies = super::images::with(|im| im.take_block_copies(keys));
        if copies.is_empty() {
            return;
        }
        self.end_rendering();
        self.begin();
        let c = ctx();
        let cmd = self.cmd();
        global_barrier(cmd);
        for b in &copies {
            let regions: Vec<vk::ImageCopy> = (0..b.levels)
                .map(|l| {
                    let sub = vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: l, base_array_layer: 0, layer_count: b.layers };
                    // Extent in texels of the (uncompressed) source.
                    vk::ImageCopy { src_subresource: sub, src_offset: vk::Offset3D::default(), dst_subresource: sub, dst_offset: vk::Offset3D::default(), extent: vk::Extent3D { width: (b.width >> l).max(1), height: (b.height >> l).max(1), depth: 1 } }
                })
                .collect();
            unsafe { c.device.cmd_copy_image(cmd, b.src, vk::ImageLayout::GENERAL, b.dst, vk::ImageLayout::GENERAL, &regions) };
            BLOCK_COPIES.fetch_add(1, Ordering::Relaxed);
            static SEEN: Mutex<Option<HashSet<u64>>> = Mutex::new(None);
            if SEEN.lock().get_or_insert_with(HashSet::new).insert(b.key.base) {
                crate::info!("images: {:#x} {}x{} {:?}: {} mip(s) of blocks copied in from its GPU-written {}x{} view", b.key.base, b.key.width, b.key.height, b.key.format, b.levels, b.width, b.height);
            }
        }
        global_barrier(cmd);
    }

    /// Submits recorded work and waits for all of this queue's GPU work and
    /// deferred effects.
    pub fn flush(&mut self) {
        self.submit_for("sync");
        wait_done(self.last_seq);
        self.inflight.clear();
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
        self.end_rendering();
        let fx = fixed();
        let cmd = self.cmd();
        let d = &ctx().device;
        let bytes = super::lodstats::SLOT_BYTES;
        global_barrier(cmd);
        unsafe {
            d.cmd_copy_buffer(cmd, fx.lodstats, fx.lod_readback, &[vk::BufferCopy { src_offset: 0, dst_offset: slot as u64 * bytes, size: bytes }]);
            if reset {
                d.cmd_fill_buffer(cmd, fx.lodstats, 0, bytes / 2, 15);
                d.cmd_fill_buffer(cmd, fx.lodstats, bytes / 2, bytes / 2, 0);
            }
            let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::HOST_READ);
            d.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::HOST, vk::DependencyFlags::empty(), &[mb], &[], &[]);
        }
        global_barrier(cmd);
    }

    /// Whether deferred CP effects are still pending (later CP memory writes
    /// must queue behind them).
    pub fn deferred_pending(&self) -> bool {
        !self.actions.is_empty() || self.last_action_seq > done_seq()
    }

    /// Performs `f` after this queue's GPU work recorded so far and its
    /// earlier deferred effects, or now when nothing is outstanding. `write`
    /// is the guest range it writes; `ordered`: later GPU work binding that
    /// range is submitted after it (CP writes that following dispatches may
    /// read; `CP_WRITES`). Returns the batch
    /// sequence it runs after (None: performed now).
    pub fn after_gpu(&mut self, f: Action, write: Option<(u64, u64)>, ordered: bool) -> Option<u64> {
        if self.recording {
            self.actions.push(f);
            self.submit_for("effect");
            if let Some(w) = write {
                INFLIGHT.note(self.last_seq, w);
                if ordered {
                    CP_WRITES.note(self.last_seq, w);
                }
                self.inflight.insert(self.last_seq, w.0, w.1);
            }
            Some(self.last_seq)
        } else if self.last_seq > done_seq() {
            let seq = push_batch(vk::Fence::null(), vk::QueryPool::null(), vec![f], Vec::new(), Vec::new(), &self.name, || {});
            self.last_seq = seq;
            self.last_action_seq = seq;
            if let Some(w) = write {
                INFLIGHT.note(seq, w);
                if ordered {
                    CP_WRITES.note(seq, w);
                }
                self.inflight.insert(seq, w.0, w.1);
            }
            Some(seq)
        } else {
            f();
            None
        }
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
            global_barrier(self.cmd());
            self.barrier_wanted = false;
            self.dispatched = false;
        }
    }

    /// Orders everything this queue does from now on after batch `seq` (a
    /// label wait passed on a value that batch will write): later batches
    /// follow it on the device queue, CP writes are deferred behind it.
    pub fn depend_on(&mut self, seq: u64) {
        self.last_seq = self.last_seq.max(seq);
        self.last_action_seq = self.last_action_seq.max(seq);
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
        if textures.is_empty() {
            return false;
        }
        let coarse: Vec<(u64, u64)> = textures.iter().map(|x| x.0).collect();
        if super::tune::tex_waits() == 0 {
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
    pub fn sync_for(&mut self, reads: &[(u64, u64)], why: usize) -> bool {
        // DS_TRACE_SYNC=1: the first 400 waits, with the read that collided
        // and the batch writing it.
        static TRACE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        static TRACED: AtomicU64 = AtomicU64::new(0);
        let trace = (*TRACE.get_or_init(|| std::env::var("DS_TRACE_SYNC").ok().as_deref() == Some("1")) && TRACED.load(Ordering::Relaxed) < 400)
            || (super::tune::get_trace_sync() && self.would_wait(reads) && super::tune::take_trace_sync());
        let culprit = if trace {
            let done = done_seq();
            reads.iter().find_map(|&r| {
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
                Some((r, format!("{who}{cp}{last}")))
            })
        } else {
            None
        };
        let t0 = std::time::Instant::now();
        let w = self.sync_for_reads(reads);
        if w {
            WAIT_NS[0].fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
            WAIT_NS[1].fetch_add(1, Ordering::Relaxed);
            note_queue_wait(&self.name, 0, t0.elapsed());
            SYNC_WHY[why.min(SYNC_WHY.len() - 1)].fetch_add(1, Ordering::Relaxed);
            if let Some(((a, n), src)) = culprit {
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

    /// Whether `sync_for_reads` would wait.
    fn would_wait(&mut self, reads: &[(u64, u64)]) -> bool {
        self.overlaps_pending(reads) || INFLIGHT.latest(reads).is_some_and(|s| s > done_seq())
    }

    pub fn sync_for_reads(&mut self, reads: &[(u64, u64)]) -> bool {
        let mut waited = false;
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
        if super::homes::any_stale(reads) {
            // GPU work wrote these parts in video memory: copied back (after
            // all earlier submitted work) before the CPU reads them.
            self.home_write_back(reads);
            self.flush();
            waited = true;
        }
        lap(0, &mut t);
        if self.overlaps_pending(reads) {
            self.flush();
            waited = true;
        }
        lap(1, &mut t);
        if let Some(s) = INFLIGHT.latest(reads) {
            if s > done_seq() {
                FOREIGN_WAITS.fetch_add(1, Ordering::Relaxed);
                wait_done(s);
                waited = true;
            }
        }
        lap(2, &mut t);
        waited
    }

    /// Whether this queue's outstanding GPU work or deferred CP effects will
    /// write [a, a+n) (a label the CP waits for comes from the GPU itself).
    pub fn will_write(&mut self, a: u64, n: u64) -> bool {
        self.overlaps_pending(&[(a, n)])
    }

    /// A guest range the batch being recorded writes.
    pub(super) fn note_write(&mut self, a: u64, n: u64) {
        self.pending.insert(a, n);
        if verify::enabled() {
            self.pending_log.push((a, n));
        }
    }

    fn overlaps_pending(&mut self, reads: &[(u64, u64)]) -> bool {
        let done = done_seq();
        self.inflight.prune(done);
        self.pending.overlaps_any(reads) || self.inflight.latest(reads, done).is_some()
    }

    /// A CP memory write that follows earlier GPU work, performed by the GPU
    /// in order with it (vkCmdUpdateBuffer between barriers) instead of by
    /// the CPU once the batch completed: the batch need not end there, and
    /// later work reading the value need not wait for that completion (the
    /// title issues ~170 such label and data writes per frame on the graphics
    /// queue). `label`: a value other queues may wait for (promised when the
    /// batch is submitted). False: not possible here.
    pub fn gpu_cp_write(&mut self, a: u64, data: &[u32], label: Option<(u64, u8)>) -> bool {
        let bytes = 4 * data.len() as u64;
        if bytes == 0 || bytes > 65536 || a % 4 != 0 || DEVICE_LOST.load(Ordering::Relaxed) || super::homes::overlaps(a, bytes) {
            return false;
        }
        let Some((buffer, start, _)) = span(a, bytes) else { return false };
        self.begin();
        self.end_rendering();
        let cmd = self.cmd();
        let raw: Vec<u8> = data.iter().flat_map(|w| w.to_le_bytes()).collect();
        global_barrier(cmd);
        unsafe { ctx().device.cmd_update_buffer(cmd, buffer, a - start, &raw) };
        global_barrier(cmd);
        self.barrier_wanted = false;
        self.dispatched = false;
        self.note_write(a, bytes);
        if let Some((v, n)) = label {
            self.pending_promises.push((a, v, n));
            UNSUBMITTED_LABELS.lock().get_or_insert_with(HashMap::new).insert(a & !7, self.submit_request.clone());
        }
        GPU_CP_WRITES.fetch_add(1, Ordering::Relaxed);
        true
    }

    /// Whether the recording batch should be submitted now: another queue
    /// waits for one of its labels, or they have waited `LABEL_LATENCY`.
    pub fn submit_requested(&self) -> bool {
        self.recording && !self.pending_promises.is_empty() && (self.submit_request.load(Ordering::Relaxed) || self.begun_at.elapsed() > LABEL_LATENCY)
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
        self.end_rendering();
        let c = ctx();
        super::vk::checkpoint(self.cmd(), super::vk::mark::SHADOW_OUT, 0);
        // The dispatches' writes to the working copies come first.
        global_barrier(self.cmd());
        let merge = if shadow_merge() { Some(merge_pipe()?) } else { None };
        for sh in &flush {
            let own = sh.size.min(sh.range - sh.bias);
            match merge {
                Some(m) => {
                    let set = self.alloc_set(m.set_layout)?;
                    let infos = [
                        vk::DescriptorBufferInfo { buffer: self.shadow(), offset: sh.shadow_off, range: sh.range },
                        vk::DescriptorBufferInfo { buffer: self.shadow(), offset: sh.pristine_off, range: sh.range },
                        vk::DescriptorBufferInfo { buffer: sh.host, offset: sh.host_off, range: sh.range },
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
        // Later work reads guest memory after the write-back.
        global_barrier(self.cmd());
        self.barrier_wanted = false;
        self.dispatched = false;
        MERGES.fetch_add(flush.len() as u64, Ordering::Relaxed);
        Ok(())
    }

    /// A copy of guest [base, base+size) in video memory, taken now (when the
    /// command processor reaches the work that reads it): (buffer, offset of
    /// the copy's 16-byte aligned start, bias of `base` in it). Bound instead
    /// of the imported system memory, which every invocation would read across
    /// PCIe. None: the range is larger than `limit` (0: off), GPU work not yet
    /// complete (or a deferred CP write) writes it, or no space.
    pub(super) fn snapshot(&mut self, base: u64, size: u64, limit: u64) -> Option<(vk::Buffer, u64, u64)> {
        if size == 0 || size > limit || self.snap_failed || super::homes::overlaps(base, size) {
            return None;
        }
        let r = [(base, size)];
        if CP_WRITES.latest(&r).is_some() || INFLIGHT.latest(&r).is_some() || self.overlaps_pending(&r) {
            return None;
        }
        let al = base & !15;
        let bias = base - al;
        if let (Some(&at), Some((buf, _))) = (self.snaps.get(&(base, size)), self.slots[self.cur].snap) {
            SNAPS_REUSED.fetch_add(1, Ordering::Relaxed);
            return Some((buf, at, bias));
        }
        if self.slots[self.cur].snap.is_none() {
            match ctx().bar_buffer(SNAP_SEG, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDEX_BUFFER) {
                Ok((b, _m, p)) => self.slots[self.cur].snap = Some((b, p)),
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
        let seg = RING_SEG;
        let off = self.ring_used[which].next_multiple_of(self.ring_align);
        if off + bytes > seg {
            return None;
        }
        self.ring_used[which] = off + bytes;
        Some(off)
    }

    /// Makes room in the recording slot for the SGPR data and descriptors of
    /// `stages` (submits the batch and starts the next one otherwise).
    fn reserve(&mut self, stages: &[&Prepared]) {
        self.begin();
        let mut need = [stages.len() as u32, 0, 0, 0, 0];
        let mut ring = 0u64;
        for p in stages {
            ring += 4 * p.sgpr_data.len() as u64 + self.ring_align;
            for b in &p.tr.out.bindings {
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
                if let BindingKind::Buffer(i) = b.kind {
                    if !p.tr.plan.resources[i].written {
                        continue;
                    }
                    // Any written buffer may be shadowed (pointer-addressed
                    // ones when their mapping ends within SHADOW_MAX).
                    need[0] += 1;
                    need[1] += 3;
                    if let Some(Resolved::Buffer(v)) = p.resources.get(i) {
                        if let Some(n) = v.byte_size().filter(|&n| n + 16 <= SHADOW_MAX) {
                            shadow += 2 * (n + 16 + 256);
                        }
                    }
                }
            }
        }
        let seg = RING_SEG;
        let fits = self.ring_used.iter().all(|u| u.next_multiple_of(self.ring_align) + ring <= seg)
            && (0..5).all(|k| self.pool_use[k] + need[k] <= POOL_LIMITS[k])
            && (shadow == 0 || self.shadow_used + shadow <= SHADOW_BYTES);
        if !fits {
            self.submit_for("full");
            self.begin();
        }
        for k in 0..5 {
            self.pool_use[k] += need[k];
        }
    }

    /// Buffer, image and sampler bindings of a stage; fills the resource
    /// info blocks of its SGPR data.
    fn bind_stage(&mut self, p: &mut Prepared, allow_shadow: bool) -> Result<Bound, String> {
        super::images::BINDING_SHADER.with(|s| s.set(p.addr));
        let res_info = rdna::translate::res_info_offset(&p.tr.plan) as usize;
        let mut bound = Bound { bufs: Vec::new(), imgs: Vec::new(), img_arrays: Vec::new(), writes: Vec::new(), sized_writes: Vec::new(), images: Vec::new(), shadows: Vec::new(), write_src: Vec::new(), placements: Vec::new() };
        let seg = SHADOW_BYTES;
        let fx = fixed();
        {
            for b in &p.tr.out.bindings {
                match b.kind {
                    BindingKind::Buffer(i) => {
                        let (base, size, stride, limit) = match &p.resources[i] {
                            Resolved::Buffer(v) => (v.base, v.byte_size(), v.stride, v.index_limit()),
                            Resolved::Global(a) => (*a, None, 0, u32::MAX),
                            _ => return Err("binding/resource mismatch".into()),
                        };
                        let mut size = match size {
                            Some(n) => n,
                            None => mapping(base).map(|(_, e)| (e - base).min(UNBOUNDED_MAX)).unwrap_or(0),
                        };
                        let mut info = vk::DescriptorBufferInfo { buffer: fx.dummy, offset: 0, range: DUMMY_BYTES };
                        let mut bias = 0u64;
                        let mut bound_size = 0u64;
                        let mut placement = Placement::None;
                        let read_only = size > 0 && !p.tr.plan.resources[i].written && matches!(p.resources[i], Resolved::Buffer(_));
                        // A video-memory home holding the range (see `homes`).
                        let home = self
                            .home_bind(base, size, p.tr.plan.resources[i].written, if matches!(p.resources[i], Resolved::Buffer(_)) { HomeUse::Sized } else { HomeUse::Unbounded })
                            .map(|h| (h.0, h.1, h.2));
                        // Inside a resident shadow: its working copy is newer.
                        let resident = if read_only && home.is_none() { self.resident_read(base, size)? } else { None };
                        // Small read-only buffers: a snapshot in video memory.
                        let snap = if read_only && resident.is_none() && home.is_none() { self.snapshot(base, size, super::tune::snap_max()) } else { None };
                        // Read-only bindings of buffers GPU work wrote: VRAM mirrors.
                        let mirrored = if read_only && snap.is_none() && home.is_none() { resident.or_else(|| super::mirror::lookup(base, size)) } else { None };
                        // Other read-only buffers: a copy made at the batch's start
                        // (dispatches only with `tune pre_cs`).
                        let pre = if read_only && snap.is_none() && mirrored.is_none() && home.is_none() && (!allow_shadow || super::tune::pre_cs()) { self.pre_copy(base, size) } else { None };
                        if pre.is_some() {
                            placement = Placement::PreCopy;
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
                        if let Some((hbuf, hoff, avail)) = home {
                            placement = Placement::Home;
                            let align = ctx().limits.min_storage_buffer_offset_alignment.max(4);
                            let off = hoff & !(align - 1);
                            bias = hoff - off;
                            info = vk::DescriptorBufferInfo { buffer: hbuf, offset: off, range: (bias + size).next_multiple_of(4).min(bias + avail) };
                            bound_size = size;
                            if p.tr.plan.resources[i].written {
                                trace_writes(&format!("{:?} shader {:#x}", p.stage, p.addr), base, size);
                                bound.writes.push((base, size));
                                if matches!(p.resources[i], Resolved::Buffer(_)) {
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
                                    if p.tr.plan.resources[i].written {
                                        trace_writes(&format!("{:?} shader {:#x}", p.stage, p.addr), base, size);
                                        bound.writes.push((base, size));
                                        if matches!(p.resources[i], Resolved::Buffer(_)) {
                                            bound.sized_writes.push((base, size));
                                        }
                                    }
                                    // Written buffers: a device-local copy for this dispatch
                                    // (GPU atomics and stores to host memory are slow).
                                    let s_off = self.shadow_used.next_multiple_of(256);
                                    let span = range.next_multiple_of(256);
                                    let res = &p.tr.plan.resources[i];
                                    let want = allow_shadow && shadow_writes() && res.written && range <= SHADOW_MAX && (res.atomic || range >= shadow_min());
                                    let window = (base - bias, range);
                                    let reuse = if want { self.resident.iter().position(|r| r.host == buffer && r.host_off == off && r.range == range) } else { None };
                                    if reuse.is_none() && self.resident.iter().any(|r| r.guest < window.0 + window.1 && window.0 < r.guest + r.range) {
                                        // Guest memory gets the resident shadows' data first.
                                        EARLY_MERGES.fetch_add(1, Ordering::Relaxed);
                                        self.flush_resident(Some(&[window]))?;
                                    }
                                    if want && reuse.is_none() && s_off + 2 * span > seg {
                                        SHADOW_FULL.fetch_add(1, Ordering::Relaxed);
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
                                                let global = matches!(p.resources[i], Resolved::Global(_));
                                                crate::info!("shadow: {:?} shader {:#x} writes {base:#x}+{size:#x} (bound {range:#x}{})", p.stage, p.addr, if global { ", global pointer" } else { "" });
                                            }
                                        }
                                        self.shadow_used = s_off + 2 * span;
                                        bound.shadows.push(Shadow { host: buffer, guest: base - bias, reused: false, host_off: off, range, shadow_off: s_off, pristine_off: s_off + span, bias, size });
                                        placement = Placement::Shadow;
                                        info = vk::DescriptorBufferInfo { buffer: self.shadow(), offset: s_off, range };
                                        SHADOWED.fetch_add(1, Ordering::Relaxed);
                                    } else if res.written && range > SHADOW_MAX && range <= SCRATCH_BYTES && super::tune::scratch_writes() {
                                        if let Some(sb) = scratch_buffer() {
                                            placement = Placement::Scratch;
                                            info = vk::DescriptorBufferInfo { buffer: sb, offset: 0, range };
                                        }
                                    }
                                    if res.written && matches!(p.resources[i], Resolved::Buffer(_)) {
                                        // Within the bound range (clamped to the mapping).
                                        bound.write_src.push((base, size.min(range - bias), info.buffer, info.offset + bias));
                                    }
                                }
                                None => {
                                    let why = unmapped_reason(base, size);
                                    self.report(format!("{:?} shader {:#x}: buffer {base:#x}+{size:#x} is not mapped GPU memory (bound as zeros): {why}", p.stage, p.addr));
                                }
                            }
                        }
                        bound.placements.push((placement, bound_size, p.tr.plan.resources[i].written));
                        let ri = res_info + RES_INFO_DWORDS as usize * i;
                        p.sgpr_data[ri] = bias as u32;
                        p.sgpr_data[ri + 1] = stride;
                        p.sgpr_data[ri + 2] = bound_size.min(u32::MAX as u64) as u32;
                        p.sgpr_data[ri + 3] = limit;
                        bound.bufs.push((b.binding, info));
                    }
                    BindingKind::Gds => bound.bufs.push((b.binding, vk::DescriptorBufferInfo { buffer: fx.gds, offset: 0, range: rdna::translate::GDS_BYTES as u64 })),
                    BindingKind::Debug => bound.bufs.push((b.binding, vk::DescriptorBufferInfo { buffer: fx.debug, offset: 0, range: rdna::translate::DUMP_DWORDS as u64 * 4 })),
                    BindingKind::Watchdog => bound.bufs.push((b.binding, vk::DescriptorBufferInfo { buffer: fx.watchdog, offset: 0, range: rdna::translate::WATCHDOG_DWORDS as u64 * 4 })),
                    BindingKind::LodStats => bound.bufs.push((b.binding, vk::DescriptorBufferInfo { buffer: fx.lodstats, offset: 0, range: super::lodstats::SLOT_BYTES })),
                    _ => {}
                }
            }
        }
        let frame = super::FRAMES.load(Ordering::Relaxed);
        // One image-cache lock for all the stage's images and samplers.
        let mut failed: Vec<String> = Vec::new();
        let r: Result<(), String> = super::images::with(|im| {
            for b in &p.tr.out.bindings {
                match (b.kind, b.kind_resource().and_then(|i| p.resources.get(i))) {
                    (BindingKind::Image(_) | BindingKind::StorageImage(_), Some(Resolved::Image(t))) => {
                        let storage = matches!(b.kind, BindingKind::StorageImage(_));
                        let (view, image) = im.view(t, storage, frame)?;
                        let ty = if storage { vk::DescriptorType::STORAGE_IMAGE } else { vk::DescriptorType::SAMPLED_IMAGE };
                        bound.imgs.push((b.binding, ty, vk::DescriptorImageInfo { sampler: vk::Sampler::null(), image_view: view, image_layout: vk::ImageLayout::GENERAL }));
                        bound.images.push(image);
                    }
                    (BindingKind::Sampler(i), Some(Resolved::Sampler(sm))) => {
                        let compare = p.tr.out.compare_samplers.contains(&i);
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
                        bound.img_arrays.push((b.binding, infos));
                    }
                    (BindingKind::Image(_) | BindingKind::StorageImage(_) | BindingKind::ImageArray(_) | BindingKind::Sampler(_), _) => return Err("binding/resource mismatch".into()),
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

    /// Copies a stage's SGPR data into the ring (space reserved) and binds it.
    fn bind_sgpr(&mut self, p: &Prepared, bound: &mut Bound) -> Result<(), String> {
        let bytes: Vec<u8> = p.sgpr_data.iter().flat_map(|w| w.to_le_bytes()).collect();
        let vram = match super::tune::ring_vram() {
            1 => true,
            2 => self.name == "gfx",
            3 => self.name != "gfx",
            4 => super::tune::ring_range().contains(&p.addr),
            _ => false,
        };
        if vram && self.slots[self.cur].rings[1].is_none() && !self.vram_ring_failed {
            match ctx().bar_buffer(RING_SEG, vk::BufferUsageFlags::STORAGE_BUFFER) {
                Ok((b, _m, p)) => self.slots[self.cur].rings[1] = Some((b, p)),
                Err(e) => {
                    crate::warn!("gpu[{}]: no SGPR data ring in video memory ({e})", self.name);
                    self.vram_ring_failed = true;
                }
            }
        }
        let which = if vram && self.slots[self.cur].rings[1].is_some() { 1 } else { 0 };
        let Some((ring, ring_ptr)) = self.slots[self.cur].rings[which] else { return Err("no SGPR data ring".into()) };
        let off = self.ring_alloc(which, bytes.len() as u64).ok_or("SGPR data ring exhausted")?;
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), ring_ptr.add(off as usize), bytes.len()) };
        for b in &p.tr.out.bindings {
            if b.kind == BindingKind::SgprData {
                bound.bufs.push((b.binding, vk::DescriptorBufferInfo { buffer: ring, offset: off, range: bytes.len() as u64 }));
            }
        }
        Ok(())
    }

    fn alloc_set(&mut self, layout: vk::DescriptorSetLayout) -> Result<vk::DescriptorSet, String> {
        let layouts = [layout];
        let ai = vk::DescriptorSetAllocateInfo::default().descriptor_pool(self.slots[self.cur].desc_pool).set_layouts(&layouts);
        unsafe { ctx().device.allocate_descriptor_sets(&ai) }.map(|v| v[0]).map_err(|e| format!("descriptor set: {e}"))
    }

    fn write_set(&self, set: vk::DescriptorSet, bounds: &[Bound]) {
        let mut writes: Vec<vk::WriteDescriptorSet> = Vec::new();
        for b in bounds {
            for (binding, info) in &b.bufs {
                writes.push(vk::WriteDescriptorSet::default().dst_set(set).dst_binding(*binding).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(std::slice::from_ref(info)));
            }
            for (binding, ty, info) in &b.imgs {
                writes.push(vk::WriteDescriptorSet::default().dst_set(set).dst_binding(*binding).descriptor_type(*ty).image_info(std::slice::from_ref(info)));
            }
            for (binding, infos) in &b.img_arrays {
                writes.push(vk::WriteDescriptorSet::default().dst_set(set).dst_binding(*binding).descriptor_type(vk::DescriptorType::SAMPLED_IMAGE).image_info(infos));
            }
        }
        unsafe { ctx().device.update_descriptor_sets(&writes, &[]) };
    }

    /// Executes a dispatch of `groups` thread groups with the compute state of `cp`.
    pub fn dispatch(&mut self, cp: &Cp, groups: [u32; 3], initiator: u32) {
        if DEVICE_LOST.load(Ordering::Relaxed) {
            SKIPPED.fetch_add(1, Ordering::Relaxed);
            return;
        }
        if SKIP_CS.lock().as_ref().is_some_and(|s| s.contains(&cs_addr(cp))) {
            SKIPPED.fetch_add(1, Ordering::Relaxed);
            return;
        }
        if let Err(e) = self.dispatch_inner(cp, groups, initiator, None) {
            SKIPPED.fetch_add(1, Ordering::Relaxed);
            self.report(e);
        } else if sync_mode() {
            self.flush();
            if DEVICE_LOST.load(Ordering::Relaxed) {
                let cs = ((cp.reg(Space::Sh, 0x20C) as u64) | ((cp.reg(Space::Sh, 0x20D) as u64 & 0xFF) << 32)) << 8;
                crate::error!("gpu[{}]: device lost by the dispatch of cs {cs:#x} ({}x{}x{} groups)", self.name, groups[0], groups[1], groups[2]);
            }
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
        let r = [(args, 12)];
        if !(self.overlaps_pending(&r) || INFLIGHT.latest(&r).is_some() || CP_WRITES.latest(&r).is_some()) {
            return false;
        }
        if SKIP_CS.lock().as_ref().is_some_and(|s| s.contains(&cs_addr(cp))) {
            SKIPPED.fetch_add(1, Ordering::Relaxed);
            return true;
        }
        GPU_INDIRECT.fetch_add(1, Ordering::Relaxed);
        if let Err(e) = self.dispatch_inner(cp, [1, 1, 1], initiator, Some(args)) {
            SKIPPED.fetch_add(1, Ordering::Relaxed);
            self.report(e);
        } else if sync_mode() {
            self.flush();
        }
        true
    }

    /// `indirect`: the arguments' guest address (the GPU reads the group
    /// counts; `groups` is unused).
    fn dispatch_inner(&mut self, cp: &Cp, mut groups: [u32; 3], initiator: u32, indirect: Option<u64>) -> Result<(), String> {
        let t0 = std::time::Instant::now();
        let mut lap = Laps::for_dispatch();
        if let Some(a) = indirect {
            // A resident shadow holding newer arguments is merged back first.
            self.flush_resident(Some(&[(a, 12)]))?;
        }
        let wave64 = initiator & (1 << 15) == 0;
        // Prepared ahead by the workers (`gpu::ahead`) when it still holds.
        let [pre, _] = cp.take_ahead();
        let mut p = match pre.and_then(|s| super::ahead::take(&s, cp, wave64)) {
            Some(p) => p,
            None => shaders::prepare_compute(cp, wave64)?,
        };
        lap.mark(0);
        self.sync_readers = [p.addr, 0];
        if self.sync_for(&p.reads, SYNC_CS_READS) {
            if let Some(c) = self.sync_culprit.take() {
                explain_wait(&self.name, c, &[&p]);
            }
            p = shaders::prepare_compute(cp, wave64)?;
        }
        lap.mark(1);
        // Textures are uploaded from guest memory: earlier GPU writes first.
        self.sync_textures(&texture_ranges(&p), SYNC_CS_TEXTURES);
        lap.mark(2);
        let local = [cp.reg(Space::Sh, 0x207) & 0xFFFF, cp.reg(Space::Sh, 0x208) & 0xFFFF, cp.reg(Space::Sh, 0x209) & 0xFFFF];
        if initiator & (1 << 5) != 0 {
            // USE_THREAD_DIMENSIONS: sizes are in threads.
            for k in 0..3 {
                groups[k] = groups[k].div_ceil(local[k].max(1));
            }
        }
        if groups.contains(&0) {
            return Ok(());
        }
        let t_prep = t0.elapsed();
        trace_loads(&self.name, &p, groups);
        self.grow_sampled_depth(&[&p]);
        let keys = image_keys(&p);
        self.resolve_fast_clears(&keys);
        self.resolve_overwrites(&keys);
        self.sync_block_textures(&keys);
        lap.mark(3);
        let (pipeline, layout, set_layout) = compute_pipe(&p)?;
        let t_pipe = t0.elapsed();
        lap.mark(4);
        self.reserve(&[&p]);
        lap.mark(5);
        let mut bound = self.bind_stage(&mut p, true)?;
        lap.mark(6);
        self.bind_sgpr(&p, &mut bound)?;
        if profiling() {
            note_placements(p.addr, groups, indirect.is_some(), &bound.placements);
        }
        lap.mark(7);
        let t_bind = t0.elapsed();
        let set = self.alloc_set(set_layout)?;
        lap.mark(8);
        let writes_here = std::mem::take(&mut bound.writes);
        let sized_here = std::mem::take(&mut bound.sized_writes);
        super::images::with(|im| im.note_writes(&writes_here, &sized_here));
        lap.mark(9);
        self.write_set(set, std::slice::from_ref(&bound));
        lap.mark(10);
        self.end_rendering();
        let c = ctx();
        let qp = self.slots[self.cur].queries;
        let ti = self.batch_shaders.len() as u32;
        let stamp_cmd = self.cmd();
        let stamp = |k: u32| {
            if qp != vk::QueryPool::null() && ti < MAX_TIMED && command_stamps() {
                unsafe { ctx().device.cmd_write_timestamp(stamp_cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, qp, STAMPS * ti + k) };
            }
        };
        if lazy_barriers() && self.barrier_wanted {
            global_barrier(self.cmd());
            self.barrier_wanted = false;
            self.dispatched = false;
        }
        stamp(0);
        let fresh: Vec<Shadow> = bound.shadows.iter().filter(|s| !s.reused).copied().collect();
        if !fresh.is_empty() {
            super::vk::checkpoint(self.cmd(), super::vk::mark::SHADOW_IN, p.addr);
            // Shadows get the guest data first (earlier writes are ordered by
            // the title's synchronization, or the barrier after every
            // dispatch without lazy barriers).
            unsafe {
                if shadow_merge() {
                    // One read of guest memory (the pristine copy), then the
                    // working copy from it: two reads could see different CPU
                    // writes, and the merge would store the older value.
                    for sh in &fresh {
                        let r = vk::BufferCopy { src_offset: sh.host_off, dst_offset: sh.pristine_off, size: sh.range };
                        c.device.cmd_copy_buffer(self.cmd(), sh.host, self.shadow(), &[r]);
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
            }
        }
        stamp(1);
        lap.mark(11);
        super::vk::checkpoint(self.cmd(), super::vk::mark::DISPATCH, p.addr);
        let start = [cp.reg(Space::Sh, 0x204), cp.reg(Space::Sh, 0x205), cp.reg(Space::Sh, 0x206)];
        let args = match indirect {
            Some(a) => {
                let (buffer, base, _) = span(a, 12).ok_or_else(|| format!("cs {:#x}: indirect arguments {a:#x} are not mapped GPU memory (dispatch skipped)", p.addr))?;
                if let Some(s) = CP_WRITES.latest(&[(a, 12)]) {
                    self.submit_after = self.submit_after.max(s);
                }
                Some((buffer, a - base))
            }
            None => None,
        };
        unsafe {
            let cmd = self.cmd();
            c.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            c.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, layout, 0, &[set], &[]);
            if let Some((buffer, offset)) = args {
                // Earlier work of this batch wrote the arguments.
                global_barrier(cmd);
                c.device.cmd_dispatch_indirect(cmd, buffer, offset);
            } else if start != [0, 0, 0] {
                c.device.cmd_dispatch_base(cmd, start[0], start[1], start[2], groups[0].saturating_sub(start[0]), groups[1].saturating_sub(start[1]), groups[2].saturating_sub(start[2]));
            } else {
                c.device.cmd_dispatch(cmd, groups[0], groups[1], groups[2]);
            }
        }
        stamp(2);
        lap.mark(12);
        if !lazy_barriers() || (!bound.shadows.is_empty() && super::tune::shadow_barrier()) {
            global_barrier(self.cmd());
        }
        // Written shadows stay resident for the rest of the batch (see Shadow);
        // a reused one merged back while this dispatch's bindings were made
        // joins again. Overlapping windows (aliased bindings) are merged now.
        for sh in &bound.shadows {
            if !self.resident.iter().any(|r| r.shadow_off == sh.shadow_off) {
                self.resident.push(Shadow { reused: false, ..*sh });
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
        let unmirrored: Vec<(u64, u64)> = writes_here.iter().copied().filter(|w| !mirrored.contains(w)).collect();
        if !unmirrored.is_empty() {
            super::mirror::invalidate(&unmirrored);
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
        for &(b, n) in &writes_here {
            super::track::gpu_write(b, n);
        }
        for &(b, n) in &writes_here {
            self.note_write(b, n);
        }
        lap.mark(15);
        EXECUTED.fetch_add(1, Ordering::Relaxed);
        self.batch_shaders.push(p.addr);
        let total = t0.elapsed();
        if profiling() {
            note_phases(&CS_PHASES, [t_prep, t_pipe, t_bind, total]);
        }
        if total.as_millis() >= 200 {
            crate::info!(
                "gpu[{}]: dispatch of cs {:#x} took {} ms (prepare {} ms, pipeline {} ms, bindings {} ms)",
                self.name,
                p.addr,
                total.as_millis(),
                t_prep.as_millis(),
                (t_pipe - t_prep).as_millis(),
                (t_bind - t_pipe).as_millis()
            );
        }
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
                c.device.destroy_fence(s.fence, None);
                c.device.destroy_descriptor_pool(s.desc_pool, None);
            }
            c.device.destroy_command_pool(self.pool, None);
        }
    }
}

/// One status line for hang reports.
pub fn status() -> String {
    format!(
        "    gpu exec: {} dispatches executed ({} with arguments read by the GPU; {} CP writes by the GPU; {} buffers shadowed in VRAM, {} not for lack of space; resident shadows: {} reused, {} read-only bindings served, {} merges, {} before an in-place use), {} skipped; {} draws executed, {} without effect, {} skipped; {} invocations stopped by the loop watchdog; {} fast clears; {} completions waited, {}; waits by opcode {}\n",
        EXECUTED.load(Ordering::Relaxed),
        GPU_INDIRECT.load(Ordering::Relaxed),
        GPU_CP_WRITES.load(Ordering::Relaxed),
        SHADOWED.load(Ordering::Relaxed),
        SHADOW_FULL.load(Ordering::Relaxed),
        SHADOWS_REUSED.load(Ordering::Relaxed),
        RESIDENT_READS.load(Ordering::Relaxed),
        MERGES.load(Ordering::Relaxed),
        EARLY_MERGES.load(Ordering::Relaxed),
        SKIPPED.load(Ordering::Relaxed),
        draw::DRAWN.load(Ordering::Relaxed),
        draw::DRAW_NOOP.load(Ordering::Relaxed),
        draw::DRAW_SKIPPED.load(Ordering::Relaxed),
        WATCHDOG_STOPS.load(Ordering::Relaxed),
        FAST_CLEARS.load(Ordering::Relaxed),
        FLUSHES.load(Ordering::Relaxed),
        verify::status(),
        super::cp::sync_summary()
    ) + &super::images::IMAGES.try_lock().and_then(|g| g.as_ref().map(|i| i.status())).unwrap_or_default()
        + &super::track::status()
        + &crate::mem::remap_status()
        + &crate::titlemem::status()
        + &super::lodstats::status()
        + &super::ahead::status()
        + &super::mirror::status()
        + &super::homes::status()
        + &sync_why_status()
        + &format!(
            "    buffer snapshots in VRAM: {} ({} MiB), {} reused, {} not for lack of space; draws with snapshotted indices {} ({} MiB); batch-start copies {} ({} MiB), {} reused, {} not for lack of space\n",
            SNAPS.load(Ordering::Relaxed),
            SNAP_COPIED.load(Ordering::Relaxed) >> 20,
            SNAPS_REUSED.load(Ordering::Relaxed),
            SNAPS_FULL.load(Ordering::Relaxed),
            INDEX_SNAPS.load(Ordering::Relaxed),
            INDEX_SNAP_BYTES.load(Ordering::Relaxed) >> 20,
            PRE_COPIES.load(Ordering::Relaxed),
            PRE_BYTES.load(Ordering::Relaxed) >> 20,
            PRE_REUSED.load(Ordering::Relaxed),
            PRE_FULL.load(Ordering::Relaxed)
        )
        + &format!(
            "    guest memory imports: {} ({} MiB), {} over several mappings, {} dropped after remapping
",
            IMPORTS.load(Ordering::Relaxed),
            IMPORT_BYTES.load(Ordering::Relaxed) >> 20,
            IMPORTS_MULTI.load(Ordering::Relaxed),
            IMPORTS_STALE.load(Ordering::Relaxed)
        )
        + &format!("    gpu completion: {} ms waiting for fences, {} ms in actions
", FENCE_NS.load(Ordering::Relaxed) / 1_000_000, ACTION_NS.load(Ordering::Relaxed) / 1_000_000)
}
