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
/// Time the completion thread spent waiting for fences / running actions (ns).
pub static FENCE_NS: AtomicU64 = AtomicU64::new(0);
pub static ACTION_NS: AtomicU64 = AtomicU64::new(0);
/// Watchdog record of `Shared` (host visible; null before GPU execution starts).
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

struct Shared {
    spans: Vec<Span>,
    pipes: HashMap<u64, Pipe>,
    /// Graphics pipelines by state hash.
    gfx: HashMap<u64, Pipe>,
    gds: vk::Buffer,
    /// Register dumps (`DS_DUMP`), host visible.
    debug: vk::Buffer,
    debug_ptr: *mut u8,
    /// Loop watchdog record (`rdna::translate::LOOP_LIMIT`), host visible
    /// (mapped at WATCHDOG_PTR).
    watchdog: vk::Buffer,
    /// Zero-filled buffer bound for empty or unmapped resources.
    dummy: vk::Buffer,
    import_align: u64,
}

unsafe impl Send for Shared {}

static SHARED: Mutex<Option<Shared>> = Mutex::new(None);

/// Contents of the register-dump buffer (`DS_DUMP`); `clear` zeroes it
/// (call with no GPU work using it in flight).
pub fn debug_dump(clear: bool) -> Vec<u32> {
    let sh = shared();
    let n = rdna::translate::DUMP_DWORDS as usize;
    let p = sh.debug_ptr as *mut u32;
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

fn shared() -> parking_lot::MappedMutexGuard<'static, Shared> {
    let mut g = SHARED.lock();
    if g.is_none() {
        let c = ctx();
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC;
        let (gds, _) = c.device_buffer(rdna::translate::GDS_BYTES as u64, usage).expect("GDS buffer");
        let (dummy, _) = c.device_buffer(DUMMY_BYTES, usage).expect("dummy buffer");
        let (debug, _, debug_ptr) = c.host_buffer(rdna::translate::DUMP_DWORDS as u64 * 4, usage).expect("debug buffer");
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
            c.device.end_command_buffer(cmd).ok();
            if c.submit(cmd, &[], &[], fence).is_ok() {
                c.device.wait_for_fences(&[fence], true, u64::MAX).ok();
            }
            c.device.destroy_fence(fence, None);
            c.device.destroy_command_pool(pool, None);
        }
        *g = Some(Shared {
            spans: Vec::new(),
            pipes: HashMap::new(),
            gfx: HashMap::new(),
            gds,
            debug,
            debug_ptr,
            watchdog,
            dummy,
            import_align: host.min_imported_host_pointer_alignment.max(4096),
        });
    }
    parking_lot::MutexGuard::map(g, |s| s.as_mut().unwrap())
}

impl Shared {
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
    fn run(addr: u64, lo: u64, hi: u64) -> Option<(u64, u64, usize)> {
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
    fn revalidate(&mut self) {
        let changed = crate::mem::take_repaged();
        if changed.is_empty() {
            return;
        }
        let ranges: Vec<(u64, u64)> = changed.iter().map(|&(a, b)| (a, b - a)).collect();
        super::mirror::invalidate(&ranges);
        for s in self.spans.iter_mut().filter(|s| s.valid) {
            if changed.iter().any(|&(a, b)| a < s.end && s.start < b) {
                // Kept alive: recorded work may still reference it.
                s.valid = false;
                if IMPORTS_STALE.fetch_add(1, Ordering::Relaxed) < 20 {
                    crate::info!("gpu: guest range {:#x}..{:#x} imported for the GPU was remapped (imported again on its next use)", s.start, s.end);
                }
            }
        }
    }

    /// Imported buffer covering guest [base, base+size): (buffer, window start, window end).
    fn span(&mut self, base: u64, size: u64) -> Option<(vk::Buffer, u64, u64)> {
        self.revalidate();
        if let Some(s) = self.spans.iter().find(|s| s.valid && s.start <= base && base + size <= s.end) {
            return Some((s.buffer, s.start, s.end));
        }
        let (vs, ve, pieces) = Self::run(base, base & !(WINDOW - 1), (base + size).next_multiple_of(WINDOW))?;
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
        let a = self.import_align;
        let start = vs & !(a - 1);
        let end = ve.next_multiple_of(a).min(ve);
        if start < vs || end <= start {
            return None;
        }
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER;
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
                self.spans.push(Span { start, end, buffer, valid: true });
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

    fn pipeline(&mut self, p: &Prepared) -> Result<(vk::Pipeline, vk::PipelineLayout, vk::DescriptorSetLayout), String> {
        let key = p.tr.key;
        if let Some(x) = self.pipes.get(&key) {
            return Ok((x.pipeline, x.layout, x.set_layout));
        }
        let t0 = std::time::Instant::now();
        let pipe = compute_pipeline(&p.tr.out)?;
        super::pstats::report(pipe.pipeline, &format!("cs {:#x}", p.addr));
        let ms = t0.elapsed().as_millis();
        if ms >= 200 {
            crate::info!("gpu: compute pipeline for cs {:#x} ({} SPIR-V words) took {ms} ms", p.addr, p.tr.out.spirv.len());
        }
        let r = (pipe.pipeline, pipe.layout, pipe.set_layout);
        self.pipes.insert(key, pipe);
        Ok(r)
    }
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
    let dummy = shared().dummy;
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
        BindingKind::SgprData | BindingKind::Buffer(_) | BindingKind::Gds | BindingKind::Debug | BindingKind::Watchdog => vk::DescriptorType::STORAGE_BUFFER,
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

const RING_BYTES: u64 = 8 << 20;
/// Descriptor pool of a slot: sets, then descriptors of each POOL_TYPES type.
const POOL_LIMITS: [u32; 5] = [4096, 32768, 8192, 2048, 8192];
const POOL_TYPES: [vk::DescriptorType; 4] = [vk::DescriptorType::STORAGE_BUFFER, vk::DescriptorType::SAMPLED_IMAGE, vk::DescriptorType::STORAGE_IMAGE, vk::DescriptorType::SAMPLER];

/// `DS_TRACE_CS=<hex address>`: logs that compute shader's scalar-load
/// values whenever they change (diagnosing data-dependent hangs).
fn trace_loads(queue: &str, p: &Prepared, groups: [u32; 3]) {
    static ADDR: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    let a = *ADDR.get_or_init(|| std::env::var("DS_TRACE_CS").ok().and_then(|v| u64::from_str_radix(v.trim_start_matches("0x"), 16).ok()));
    if a != Some(p.addr) {
        return;
    }
    static LAST: Mutex<Vec<u32>> = Mutex::new(Vec::new());
    let mut last = LAST.lock();
    if *last != p.sgpr_data {
        *last = p.sgpr_data.clone();
        crate::info!("trace cs {:#x} [{queue}] {}x{}x{}: {}", p.addr, groups[0], groups[1], groups[2], shaders::describe_loads(p).join(" | "));
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
/// Video memory for snapshots per queue (half per slot).
const SNAP_BYTES: u64 = 32 << 20;

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

/// Guest ranges of a stage's textures (uploads read them).
fn texture_ranges(p: &Prepared) -> Vec<(u64, u64)> {
    p.resources
        .iter()
        .filter_map(|r| match r {
            Resolved::Image(t) => Some(vec![(t.base, (t.width as u64 * t.height as u64 * t.depth.max(1) as u64 * 16).min(1 << 30))]),
            Resolved::ImageTable(e) => Some(e.iter().flatten().map(|t| (t.base, (t.width as u64 * t.height as u64 * t.depth.max(1) as u64 * 16).min(1 << 30))).collect()),
            _ => None,
        })
        .flatten()
        .collect()
}
/// Batches in flight per queue: one recording while the other executes.
const SLOTS: usize = 2;

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
static DONE_CV: parking_lot::Condvar = parking_lot::Condvar::new();

fn done_seq() -> u64 {
    *DONE.lock()
}

/// Highest completed batch sequence (all earlier ones completed too).
pub fn completed_seq() -> u64 {
    done_seq()
}

/// Guest ranges written by unfinished batches (sequence, start, end),
/// bucketed by 1 MiB.
struct RangeLog(Mutex<Option<HashMap<u64, Vec<(u64, u64, u64)>>>>);

const RANGE_BUCKET: u32 = 20;

impl RangeLog {
    const fn new() -> RangeLog {
        RangeLog(Mutex::new(None))
    }

    fn note(&self, seq: u64, (a, n): (u64, u64)) {
        if n == 0 {
            return;
        }
        let done = done_seq();
        let mut g = self.0.lock();
        let m = g.get_or_insert_with(HashMap::new);
        if m.len() > 4096 {
            m.retain(|_, v| {
                v.retain(|e| e.0 > done);
                !v.is_empty()
            });
        }
        let end = a.saturating_add(n);
        for b in (a >> RANGE_BUCKET)..=((end - 1) >> RANGE_BUCKET) {
            let v = m.entry(b).or_default();
            v.retain(|e| e.0 > done);
            v.push((seq, a, end));
        }
    }

    /// The latest unfinished batch that writes into `reads`.
    fn latest(&self, reads: &[(u64, u64)]) -> Option<u64> {
        let done = done_seq();
        let mut g = self.0.lock();
        let m = g.as_mut()?;
        let mut best: Option<u64> = None;
        for &(r, n) in reads {
            if n == 0 {
                continue;
            }
            let end = r.saturating_add(n);
            for b in (r >> RANGE_BUCKET)..=((end - 1) >> RANGE_BUCKET) {
                if let Some(v) = m.get_mut(&b) {
                    v.retain(|e| e.0 > done);
                    for &(s, wa, we) in v.iter() {
                        if wa < end && r < we {
                            best = best.max(Some(s));
                        }
                    }
                }
            }
        }
        best
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
        let b = {
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
            collect_timestamps(b.queries, &b.shaders);
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
        for ch in &b.checks {
            verify::run(&b.name, ch);
        }
        *DONE.lock() = b.seq;
        DONE_CV.notify_all();
    }
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
}

const MAX_TIMED: u32 = 4096;
/// Timed draws are keyed by their pixel shader (vertex shader without one)
/// with this bit set.
const DRAW_KEY: u64 = 1 << 63;

fn profiling() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_PROFILE_GPU").ok().as_deref() == Some("1"))
}

/// Timestamps per timed dispatch (all BOTTOM_OF_PIPE): before the shadow
/// copies in, before the dispatch, after it, after the copies out.
const STAMPS: u32 = 4;

/// GPU time per shader (DS_PROFILE_GPU=1): address -> [dispatches, dispatch
/// ns, copy-in ns, copy-out ns].
static SHADER_TIME: Mutex<Option<HashMap<u64, [u64; 4]>>> = Mutex::new(None);
/// GPU time between timed dispatches (draws, CP copies, barriers), ns.
static GAP_NS: AtomicU64 = AtomicU64::new(0);

/// GPU time inside batches (first to last command), ns.
static BATCH_NS: AtomicU64 = AtomicU64::new(0);

/// Reads a completed batch's dispatch timestamps into SHADER_TIME.
fn collect_timestamps(pool: vk::QueryPool, shaders: &[u64]) {
    if pool == vk::QueryPool::null() {
        return;
    }
    let c = ctx();
    let mut be = [0u64; 2];
    if unsafe { c.device.get_query_pool_results(pool, STAMPS * MAX_TIMED, &mut be, vk::QueryResultFlags::TYPE_64) }.is_ok() {
        BATCH_NS.fetch_add((be[1].saturating_sub(be[0]) as f64 * c.limits.timestamp_period as f64) as u64, Ordering::Relaxed);
    }
    let n = (shaders.len() as u32).min(MAX_TIMED);
    if n == 0 {
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
    GAP_NS.store(0, Ordering::Relaxed);
    BATCH_NS.store(0, Ordering::Relaxed);
}

/// Shaders by total GPU time.
pub fn shader_profile() -> String {
    let g = SHADER_TIME.lock();
    let Some(m) = g.as_ref() else { return String::new() };
    let mut v: Vec<(&u64, &[u64; 4])> = m.iter().collect();
    v.sort_by(|a, b| (b.1[1] + b.1[2] + b.1[3]).cmp(&(a.1[1] + a.1[2] + a.1[3])));
    let tot = |k: usize, draws: bool| m.iter().filter(|(a, _)| (**a & DRAW_KEY != 0) == draws).map(|(_, x)| x[k]).sum::<u64>() / 1_000_000;
    let mut s = format!(
        "    gpu time: batches {} ms; dispatches {} ms, draws {} ms, shadow copies in {} ms / out {} ms, between timed commands {} ms\n",
        BATCH_NS.load(Ordering::Relaxed) / 1_000_000,
        tot(1, false),
        tot(1, true),
        tot(2, false),
        tot(3, false),
        GAP_NS.load(Ordering::Relaxed) / 1_000_000
    );
    let top = std::env::var("DS_PROFILE_TOP").ok().and_then(|x| x.parse().ok()).unwrap_or(15);
    for (a, x) in v.iter().take(top) {
        let n = x[0].max(1);
        if **a & DRAW_KEY != 0 {
            s.push_str(&format!("      draws with shader {:#x}: {} draws, {} us each
", **a & !DRAW_KEY, x[0], x[1] / 1000 / n));
        } else {
            s.push_str(&format!("      cs {a:#x}: {} dispatches, {} us each (+ copies {} / {} us)\n", x[0], x[1] / 1000 / n, x[2] / 1000 / n, x[3] / 1000 / n));
        }
    }
    s
}

pub struct Exec {
    pool: vk::CommandPool,
    slots: Vec<Slot>,
    cur: usize,
    /// The SGPR data rings: system memory and video memory (allocated when
    /// `tune ring_vram` first chooses it for a stage; failed: never).
    rings: [Option<(vk::Buffer, *mut u8)>; 2],
    vram_ring_failed: bool,
    /// Offsets within the current slot's segment of each ring.
    ring_used: [u64; 2],
    /// Read-only buffer snapshots in video memory (allocated on first use;
    /// a segment per slot), space used in the current segment, failure to
    /// allocate, and the snapshots of the batch being recorded by range.
    snap: Option<(vk::Buffer, *mut u8)>,
    snap_used: u64,
    snap_failed: bool,
    snaps: HashMap<(u64, u64), u64>,
    ring_align: u64,
    recording: bool,
    /// Guest ranges written by the batch being recorded.
    pending_writes: Vec<(u64, u64)>,
    /// Actions of the batch being recorded.
    actions: Vec<Action>,
    checks: Vec<verify::Check>,
    /// Guest ranges written by submitted batches or deferred CP writes, by sequence.
    inflight: Vec<(u64, (u64, u64))>,
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
    /// Descriptor pool usage of the recording slot (see POOL_LIMITS).
    pool_use: [u32; 5],
    /// Device-local shadow ring (SHADOW_BYTES per slot) and its use in the recording slot.
    shadow: vk::Buffer,
    shadow_used: u64,
    /// Shadows of this batch whose working copies hold newer data than
    /// guest memory (see Shadow).
    resident: Vec<Shadow>,
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
            let cmds = c.device.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(pool).command_buffer_count(SLOTS as u32)).map_err(|e| e.to_string())?;
            let sizes: Vec<vk::DescriptorPoolSize> = POOL_TYPES.iter().zip(&POOL_LIMITS[1..]).map(|(&ty, &n)| vk::DescriptorPoolSize { ty, descriptor_count: n }).collect();
            let mut slots = Vec::new();
            for cmd in cmds {
                slots.push(Slot {
                    cmd,
                    fence: c.device.create_fence(&vk::FenceCreateInfo::default(), None).map_err(|e| e.to_string())?,
                    desc_pool: c.device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(POOL_LIMITS[0]).pool_sizes(&sizes), None).map_err(|e| e.to_string())?,
                    seq: 0,
                    queries: if profiling() {
                        c.device.create_query_pool(&vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(STAMPS * MAX_TIMED + 2), None).map_err(|e| e.to_string())?
                    } else {
                        vk::QueryPool::null()
                    },
                });
            }
            // SGPR data in system memory: a copy in video memory, read by
            // every invocation, made no measurable difference (`tune ring_vram`).
            let (ring, _mem, ring_ptr) = c.host_buffer(RING_BYTES, vk::BufferUsageFlags::STORAGE_BUFFER)?;
            let (shadow, _smem) = c.device_buffer(SHADOW_BYTES * SLOTS as u64, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST)?;
            Ok(Exec {
                pool,
                slots,
                cur: 0,
                rings: [Some((ring, ring_ptr)), None],
                vram_ring_failed: false,
                ring_used: [0; 2],
                snap: None,
                snap_used: 0,
                snap_failed: false,
                snaps: HashMap::new(),
                ring_align: c.limits.min_storage_buffer_offset_alignment.max(16),
                recording: false,
                pending_writes: Vec::new(),
                actions: Vec::new(),
                checks: Vec::new(),
                inflight: Vec::new(),
                last_seq: 0,
                last_action_seq: 0,
                submit_after: 0,
                barrier_wanted: false,
                dispatched: false,
                reported: HashMap::new(),
                name: name.to_string(),
                batch_shaders: Vec::new(),
                rendering: None,
                pool_use: [0; 5],
                shadow,
                shadow_used: 0,
                resident: Vec::new(),
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

    fn begin(&mut self) {
        if self.recording {
            return;
        }
        // The slot's previous batch must have completed before reuse.
        let slot = &self.slots[self.cur];
        if slot.seq != 0 {
            wait_done(slot.seq);
            let c = ctx();
            unsafe {
                c.device.reset_fences(&[slot.fence]).ok();
                c.device.reset_command_buffer(slot.cmd, vk::CommandBufferResetFlags::empty()).ok();
                c.device.reset_descriptor_pool(slot.desc_pool, vk::DescriptorPoolResetFlags::empty()).ok();
            }
            self.slots[self.cur].seq = 0;
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
        self.pool_use = [0; 5];
        self.shadow_used = 0;
        self.resident.clear();
    }

    /// Submits the recorded batch without waiting for it (its actions run
    /// when it completes).
    pub fn submit(&mut self) {
        if !self.recording {
            return;
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
        unsafe {
            // Make shader writes to imported memory visible to the host.
            let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::MEMORY_WRITE).dst_access_mask(vk::AccessFlags::HOST_READ);
            c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::HOST, vk::DependencyFlags::empty(), &[mb], &[], &[]);
            c.device.end_command_buffer(cmd).ok();
        }
        let actions = std::mem::take(&mut self.actions);
        let has_actions = !actions.is_empty();
        let name = self.name.clone();
        let mut checks = std::mem::take(&mut self.checks);
        for ch in checks.iter_mut() {
            let (a, n) = verify::dest_range(ch);
            ch.overwritten = self.pending_writes.get(ch.writes_mark..).unwrap_or(&[]).iter().any(|&(w, m)| w < a + n && a < w + m);
        }
        let seq = push_batch(fence, self.slots[i].queries, actions, checks, std::mem::take(&mut self.batch_shaders), &self.name, || {
            if DEVICE_LOST.load(Ordering::Relaxed) {
                return;
            }
            if let Err(e) = c.submit(cmd, &[], &[], fence) {
                if !DEVICE_LOST.swap(true, Ordering::SeqCst) {
                    crate::error!("gpu[{name}]: {e}; GPU execution stops; checkpoints: {}", super::vk::checkpoint_report());
                }
            }
        });
        self.slots[i].seq = seq;
        for w in std::mem::take(&mut self.pending_writes) {
            INFLIGHT.note(seq, w);
            self.inflight.push((seq, w));
        }
        self.last_seq = seq;
        if has_actions {
            self.last_action_seq = seq;
        }
        self.recording = false;
        self.cur = (self.cur + 1) % SLOTS;
        self.ring_used = [0; 2];
        self.snap_used = 0;
        self.snaps.clear();
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
        self.submit();
        wait_done(self.last_seq);
        self.inflight.clear();
    }

    /// Completes this queue's recorded and in-flight work when any of it
    /// writes into `reads` (memory the CP reads now: indirect arguments,
    /// draw counts); returns whether it waited.
    pub fn flush_if_writes(&mut self, reads: &[(u64, u64)]) -> bool {
        self.sync_for_reads(reads)
    }

    /// Whether GPU work is recorded or not yet completed.
    pub fn busy(&self) -> bool {
        self.recording || self.last_seq > done_seq()
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
            self.submit();
            if let Some(w) = write {
                INFLIGHT.note(self.last_seq, w);
                if ordered {
                    CP_WRITES.note(self.last_seq, w);
                }
                self.inflight.push((self.last_seq, w));
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
                self.inflight.push((seq, w));
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
    pub fn sync_for_reads(&mut self, reads: &[(u64, u64)]) -> bool {
        let mut waited = false;
        if self.overlaps_pending(reads) {
            self.flush();
            waited = true;
        }
        if let Some(s) = INFLIGHT.latest(reads) {
            if s > done_seq() {
                FOREIGN_WAITS.fetch_add(1, Ordering::Relaxed);
                wait_done(s);
                waited = true;
            }
        }
        waited
    }

    /// Whether this queue's outstanding GPU work or deferred CP effects will
    /// write [a, a+n) (a label the CP waits for comes from the GPU itself).
    pub fn will_write(&mut self, a: u64, n: u64) -> bool {
        self.overlaps_pending(&[(a, n)])
    }

    fn overlaps_pending(&mut self, reads: &[(u64, u64)]) -> bool {
        let done = done_seq();
        self.inflight.retain(|(seq, _)| *seq > done);
        let ov = |&(a, n): &(u64, u64), &(w, m): &(u64, u64)| w < a + n && a < w + m;
        reads.iter().any(|r| self.pending_writes.iter().any(|w| ov(r, w)) || self.inflight.iter().any(|(_, w)| ov(r, w)))
    }

    /// A read-only binding of guest [base, base+size) inside a resident shadow
    /// reads its working copy: (buffer, offset of `base`, bytes from `base`
    /// to the copy's end). A binding that overlaps one only partly gets it
    /// merged back first and None.
    fn resident_read(&mut self, base: u64, size: u64) -> Result<Option<(vk::Buffer, u64, u64)>, String> {
        if let Some(r) = self.resident.iter().find(|r| r.guest <= base && base + size <= r.guest + r.range).copied() {
            RESIDENT_READS.fetch_add(1, Ordering::Relaxed);
            return Ok(Some((self.shadow, r.shadow_off + (base - r.guest), r.guest + r.range - base)));
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
                        vk::DescriptorBufferInfo { buffer: self.shadow, offset: sh.shadow_off, range: sh.range },
                        vk::DescriptorBufferInfo { buffer: self.shadow, offset: sh.pristine_off, range: sh.range },
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
                    c.device.cmd_copy_buffer(self.cmd(), self.shadow, sh.host, &[r]);
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
        if size == 0 || size > limit || self.snap_failed {
            return None;
        }
        let r = [(base, size)];
        if CP_WRITES.latest(&r).is_some() || INFLIGHT.latest(&r).is_some() || self.overlaps_pending(&r) {
            return None;
        }
        let al = base & !15;
        let bias = base - al;
        if let (Some(&at), Some((buf, _))) = (self.snaps.get(&(base, size)), self.snap) {
            SNAPS_REUSED.fetch_add(1, Ordering::Relaxed);
            return Some((buf, at, bias));
        }
        if self.snap.is_none() {
            match ctx().bar_buffer(SNAP_BYTES, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDEX_BUFFER) {
                Ok((b, _m, p)) => self.snap = Some((b, p)),
                Err(e) => {
                    crate::warn!("gpu[{}]: no video memory for buffer snapshots ({e})", self.name);
                    self.snap_failed = true;
                    return None;
                }
            }
        }
        let (buf, ptr) = self.snap?;
        let seg = SNAP_BYTES / SLOTS as u64;
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
        let at = seg * self.cur as u64 + off;
        unsafe { std::ptr::copy_nonoverlapping(al as *const u8, ptr.add(at as usize), n as usize) };
        SNAPS.fetch_add(1, Ordering::Relaxed);
        SNAP_COPIED.fetch_add(n, Ordering::Relaxed);
        self.snaps.insert((base, size), at);
        Some((buf, at, bias))
    }

    /// Allocates SGPR data space in the current slot's ring segment.
    fn ring_alloc(&mut self, which: usize, bytes: u64) -> Option<u64> {
        let seg = RING_BYTES / SLOTS as u64;
        let off = self.ring_used[which].next_multiple_of(self.ring_align);
        if off + bytes > seg {
            return None;
        }
        self.ring_used[which] = off + bytes;
        Some(seg * self.cur as u64 + off)
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
        let seg = RING_BYTES / SLOTS as u64;
        let fits = self.ring_used.iter().all(|u| u.next_multiple_of(self.ring_align) + ring <= seg)
            && (0..5).all(|k| self.pool_use[k] + need[k] <= POOL_LIMITS[k])
            && (shadow == 0 || self.shadow_used + shadow <= SHADOW_BYTES);
        if !fits {
            self.submit();
            self.begin();
        }
        for k in 0..5 {
            self.pool_use[k] += need[k];
        }
    }

    /// Buffer, image and sampler bindings of a stage; fills the resource
    /// info blocks of its SGPR data.
    fn bind_stage(&mut self, p: &mut Prepared, allow_shadow: bool) -> Result<Bound, String> {
        let res_info = rdna::translate::res_info_offset(&p.tr.plan) as usize;
        let mut bound = Bound { bufs: Vec::new(), imgs: Vec::new(), img_arrays: Vec::new(), writes: Vec::new(), sized_writes: Vec::new(), images: Vec::new(), shadows: Vec::new(), write_src: Vec::new() };
        let seg = SHADOW_BYTES;
        let shadow_base = seg * self.cur as u64;
        {
            let mut sh = shared();
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
                            None => Shared::mapping(base).map(|(_, e)| (e - base).min(UNBOUNDED_MAX)).unwrap_or(0),
                        };
                        let mut info = vk::DescriptorBufferInfo { buffer: sh.dummy, offset: 0, range: DUMMY_BYTES };
                        let mut bias = 0u64;
                        let mut bound_size = 0u64;
                        let read_only = size > 0 && !p.tr.plan.resources[i].written && matches!(p.resources[i], Resolved::Buffer(_));
                        // Inside a resident shadow: its working copy is newer.
                        let resident = if read_only { self.resident_read(base, size)? } else { None };
                        // Small read-only buffers: a snapshot in video memory.
                        let snap = if read_only && resident.is_none() { self.snapshot(base, size, super::tune::snap_max()) } else { None };
                        // Read-only bindings of buffers GPU work wrote: VRAM mirrors.
                        let mirrored = if read_only && snap.is_none() { resident.or_else(|| super::mirror::lookup(base, size)) } else { None };
                        let mut span = if size > 0 && mirrored.is_none() && snap.is_none() { sh.span(base, size) } else { None };
                        if span.is_none() && size > 0 && mirrored.is_none() && snap.is_none() {
                            // A descriptor reaching past the end of its mappings
                            // (generic sizes on ring allocations): the mapped part
                            // is bound, the rest reads 0.
                            if let Some((_, end, _)) = Shared::run(base, base & !(WINDOW - 1), (base + size).next_multiple_of(WINDOW)) {
                                if end > base && end < base + size {
                                    size = end - base;
                                    span = sh.span(base, size);
                                }
                            }
                        }
                        if let Some((sbuf, soff, sbias)) = snap {
                            bias = sbias;
                            info = vk::DescriptorBufferInfo { buffer: sbuf, offset: soff, range: (sbias + size).next_multiple_of(4) };
                            bound_size = size;
                        } else if let Some((mbuf, moff, avail)) = mirrored {
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
                                        info = vk::DescriptorBufferInfo { buffer: self.shadow, offset: r.shadow_off, range };
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
                                        bound.shadows.push(Shadow { host: buffer, guest: base - bias, reused: false, host_off: off, range, shadow_off: shadow_base + s_off, pristine_off: shadow_base + s_off + span, bias, size });
                                        info = vk::DescriptorBufferInfo { buffer: self.shadow, offset: shadow_base + s_off, range };
                                        SHADOWED.fetch_add(1, Ordering::Relaxed);
                                    }
                                    if res.written && matches!(p.resources[i], Resolved::Buffer(_)) {
                                        // Within the bound range (clamped to the mapping).
                                        bound.write_src.push((base, size.min(range - bias), info.buffer, info.offset + bias));
                                    }
                                }
                                None => {
                                    drop(sh);
                                    let why = Shared::unmapped_reason(base, size);
                                    self.report(format!("{:?} shader {:#x}: buffer {base:#x}+{size:#x} is not mapped GPU memory (bound as zeros): {why}", p.stage, p.addr));
                                    sh = shared();
                                }
                            }
                        }
                        let ri = res_info + RES_INFO_DWORDS as usize * i;
                        p.sgpr_data[ri] = bias as u32;
                        p.sgpr_data[ri + 1] = stride;
                        p.sgpr_data[ri + 2] = bound_size.min(u32::MAX as u64) as u32;
                        p.sgpr_data[ri + 3] = limit;
                        bound.bufs.push((b.binding, info));
                    }
                    BindingKind::Gds => bound.bufs.push((b.binding, vk::DescriptorBufferInfo { buffer: sh.gds, offset: 0, range: rdna::translate::GDS_BYTES as u64 })),
                    BindingKind::Debug => bound.bufs.push((b.binding, vk::DescriptorBufferInfo { buffer: sh.debug, offset: 0, range: rdna::translate::DUMP_DWORDS as u64 * 4 })),
                    BindingKind::Watchdog => bound.bufs.push((b.binding, vk::DescriptorBufferInfo { buffer: sh.watchdog, offset: 0, range: rdna::translate::WATCHDOG_DWORDS as u64 * 4 })),
                    _ => {}
                }
            }
        }
        let frame = super::FRAMES.load(Ordering::Relaxed);
        for b in &p.tr.out.bindings {
            match (b.kind, b.kind_resource().and_then(|i| p.resources.get(i))) {
                (BindingKind::Image(_) | BindingKind::StorageImage(_), Some(Resolved::Image(t))) => {
                    let storage = matches!(b.kind, BindingKind::StorageImage(_));
                    let (view, image) = super::images::with(|im| im.view(t, storage, frame))?;
                    let ty = if storage { vk::DescriptorType::STORAGE_IMAGE } else { vk::DescriptorType::SAMPLED_IMAGE };
                    bound.imgs.push((b.binding, ty, vk::DescriptorImageInfo { sampler: vk::Sampler::null(), image_view: view, image_layout: vk::ImageLayout::GENERAL }));
                    bound.images.push(image);
                }
                (BindingKind::Sampler(i), Some(Resolved::Sampler(sm))) => {
                    let compare = p.tr.out.compare_samplers.contains(&i);
                    let smp = super::images::with(|im| im.sampler(*sm, compare))?;
                    bound.imgs.push((b.binding, vk::DescriptorType::SAMPLER, vk::DescriptorImageInfo { sampler: smp, image_view: vk::ImageView::null(), image_layout: vk::ImageLayout::UNDEFINED }));
                }
                (BindingKind::ImageArray(_), Some(Resolved::ImageTable(entries))) => {
                    let dim = entries.iter().flatten().find_map(|t| rdna::translate::ImageDim::from_type(t.ty, t.depth)).ok_or("image table without entries")?;
                    let mut infos = Vec::with_capacity(rdna::translate::IMAGE_TABLE_LEN as usize);
                    for k in 0..rdna::translate::IMAGE_TABLE_LEN as usize {
                        let view = match entries.get(k).copied().flatten() {
                            Some(t) => match super::images::with(|im| im.view(&t, false, frame)) {
                                Ok((view, image)) => {
                                    bound.images.push(image);
                                    Some(view)
                                }
                                Err(e) => {
                                    self.report(format!("image table entry {:#x}: {e}", t.base));
                                    None
                                }
                            },
                            None => None,
                        };
                        let view = match view {
                            Some(v) => v,
                            None => super::images::with(|im| im.dummy_view(dim))?,
                        };
                        infos.push(vk::DescriptorImageInfo { sampler: vk::Sampler::null(), image_view: view, image_layout: vk::ImageLayout::GENERAL });
                    }
                    bound.img_arrays.push((b.binding, infos));
                }
                (BindingKind::Image(_) | BindingKind::StorageImage(_) | BindingKind::ImageArray(_) | BindingKind::Sampler(_), _) => return Err("binding/resource mismatch".into()),
                _ => {}
            }
        }
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
        if vram && self.rings[1].is_none() && !self.vram_ring_failed {
            match ctx().bar_buffer(RING_BYTES, vk::BufferUsageFlags::STORAGE_BUFFER) {
                Ok((b, _m, p)) => self.rings[1] = Some((b, p)),
                Err(e) => {
                    crate::warn!("gpu[{}]: no SGPR data ring in video memory ({e})", self.name);
                    self.vram_ring_failed = true;
                }
            }
        }
        let which = if vram && self.rings[1].is_some() { 1 } else { 0 };
        let Some((ring, ring_ptr)) = self.rings[which] else { return Err("no SGPR data ring".into()) };
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
        if let Err(e) = self.dispatch_inner(cp, groups, initiator) {
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

    fn dispatch_inner(&mut self, cp: &Cp, mut groups: [u32; 3], initiator: u32) -> Result<(), String> {
        let t0 = std::time::Instant::now();
        let wave64 = initiator & (1 << 15) == 0;
        let mut p = shaders::prepare_compute(cp, wave64)?;
        if self.sync_for_reads(&p.reads) {
            p = shaders::prepare_compute(cp, wave64)?;
        }
        // Textures are uploaded from guest memory: earlier GPU writes first.
        self.sync_for_reads(&texture_ranges(&p));
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
        let (pipeline, layout, set_layout) = shared().pipeline(&p)?;
        let t_pipe = t0.elapsed();
        self.reserve(&[&p]);
        let mut bound = self.bind_stage(&mut p, true)?;
        self.bind_sgpr(&p, &mut bound)?;
        let t_bind = t0.elapsed();
        let set = self.alloc_set(set_layout)?;
        let writes_here = std::mem::take(&mut bound.writes);
        let sized_here = std::mem::take(&mut bound.sized_writes);
        super::images::with(|im| {
            im.note_meta_writes(&writes_here);
            im.note_overwrites(&sized_here);
        });
        self.write_set(set, std::slice::from_ref(&bound));
        self.end_rendering();
        let c = ctx();
        let qp = self.slots[self.cur].queries;
        let ti = self.batch_shaders.len() as u32;
        let stamp_cmd = self.cmd();
        let stamp = |k: u32| {
            if qp != vk::QueryPool::null() && ti < MAX_TIMED {
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
                        c.device.cmd_copy_buffer(self.cmd(), sh.host, self.shadow, &[r]);
                    }
                    let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_READ);
                    c.device.cmd_pipeline_barrier(self.cmd(), vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[mb], &[], &[]);
                    for sh in &fresh {
                        let r = vk::BufferCopy { src_offset: sh.pristine_off, dst_offset: sh.shadow_off, size: sh.range };
                        c.device.cmd_copy_buffer(self.cmd(), self.shadow, self.shadow, &[r]);
                    }
                } else {
                    for sh in &fresh {
                        let r = vk::BufferCopy { src_offset: sh.host_off, dst_offset: sh.shadow_off, size: sh.range };
                        c.device.cmd_copy_buffer(self.cmd(), sh.host, self.shadow, &[r]);
                    }
                }
                let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
                c.device.cmd_pipeline_barrier(self.cmd(), vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &[mb], &[], &[]);
            }
        }
        stamp(1);
        super::vk::checkpoint(self.cmd(), super::vk::mark::DISPATCH, p.addr);
        let start = [cp.reg(Space::Sh, 0x204), cp.reg(Space::Sh, 0x205), cp.reg(Space::Sh, 0x206)];
        unsafe {
            let cmd = self.cmd();
            c.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            c.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, layout, 0, &[set], &[]);
            if start != [0, 0, 0] {
                c.device.cmd_dispatch_base(cmd, start[0], start[1], start[2], groups[0].saturating_sub(start[0]), groups[1].saturating_sub(start[1]), groups[2].saturating_sub(start[2]));
            } else {
                c.device.cmd_dispatch(cmd, groups[0], groups[1], groups[2]);
            }
        }
        stamp(2);
        if !lazy_barriers() || !bound.shadows.is_empty() {
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
        for &(b, n) in &writes_here {
            super::track::gpu_write(b, n);
        }
        self.pending_writes.extend(writes_here);
        EXECUTED.fetch_add(1, Ordering::Relaxed);
        self.batch_shaders.push(p.addr);
        let total = t0.elapsed();
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
        if let Some(mut ch) = verify::check_for(&p, groups, local) {
            // Expected values as recorded, when no GPU work still writes them.
            let src = verify::source_ranges(&ch);
            if !src.is_empty() && !self.overlaps_pending(&src) && INFLIGHT.latest(&src).is_none() && CP_WRITES.latest(&src).is_none() {
                verify::snapshot_source(&mut ch);
            }
            ch.writes_mark = self.pending_writes.len();
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
        "    gpu exec: {} dispatches executed ({} buffers shadowed in VRAM, {} not for lack of space; resident shadows: {} reused, {} read-only bindings served, {} merges, {} before an in-place use), {} skipped; {} draws executed, {} without effect, {} skipped; {} invocations stopped by the loop watchdog; {} fast clears; {} completions waited, {}; waits by opcode {}\n",
        EXECUTED.load(Ordering::Relaxed),
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
        + &super::mirror::status()
        + &format!(
            "    buffer snapshots in VRAM: {} ({} MiB), {} reused, {} not for lack of space; draws with snapshotted indices {} ({} MiB)\n",
            SNAPS.load(Ordering::Relaxed),
            SNAP_COPIED.load(Ordering::Relaxed) >> 20,
            SNAPS_REUSED.load(Ordering::Relaxed),
            SNAPS_FULL.load(Ordering::Relaxed),
            INDEX_SNAPS.load(Ordering::Relaxed),
            INDEX_SNAP_BYTES.load(Ordering::Relaxed) >> 20
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
