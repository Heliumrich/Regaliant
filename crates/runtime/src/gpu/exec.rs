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
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

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

/// GPU execution enabled (`DS_GPU=1`) and Vulkan available.
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
    /// Still backed by a guest mapping (re-checked when mappings change).
    valid: bool,
}

struct Pipe {
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    set_layout: vk::DescriptorSetLayout,
}

struct Shared {
    spans: Vec<Span>,
    map_gen: u64,
    pipes: HashMap<u64, Pipe>,
    gds: vk::Buffer,
    /// Zero-filled buffer bound for empty or unmapped resources.
    dummy: vk::Buffer,
    import_align: u64,
}

unsafe impl Send for Shared {}

static SHARED: Mutex<Option<Shared>> = Mutex::new(None);

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
            map_gen: crate::mem::MAP_GEN.load(Ordering::SeqCst),
            pipes: HashMap::new(),
            gds,
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

    /// Imported buffer covering guest [base, base+size): (buffer, window start, window end).
    fn span(&mut self, base: u64, size: u64) -> Option<(vk::Buffer, u64, u64)> {
        let map_gen = crate::mem::MAP_GEN.load(Ordering::SeqCst);
        if map_gen != self.map_gen {
            self.map_gen = map_gen;
            for s in self.spans.iter_mut().filter(|s| s.valid) {
                let ok = Self::mapping(s.start).is_some_and(|(a, b)| a <= s.start && s.end <= b);
                if !ok {
                    // Kept alive: recorded work may still reference it.
                    crate::warn!("gpu: guest range {:#x}..{:#x} imported for the GPU was unmapped", s.start, s.end);
                    s.valid = false;
                }
            }
        }
        if let Some(s) = self.spans.iter().find(|s| s.valid && s.start <= base && base + size <= s.end) {
            return Some((s.buffer, s.start, s.end));
        }
        let (vs, ve) = Self::mapping(base)?;
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
        let start = (base & !(WINDOW - 1)).max(vs) & !(a - 1);
        let end = ((base + size).next_multiple_of(WINDOW)).min(ve).next_multiple_of(a).min(ve);
        if start < vs || end <= start {
            return None;
        }
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER;
        let t0 = std::time::Instant::now();
        match ctx().import_host(start, end - start, usage) {
            Ok((buffer, _mem)) => {
                let ms = t0.elapsed().as_millis();
                if ms >= 50 {
                    crate::info!("gpu: importing guest {start:#x}..{end:#x} ({} MiB) took {ms} ms", (end - start) >> 20);
                }
                self.spans.push(Span { start, end, buffer, valid: true });
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
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                })
                .collect();
            let set_layout = c.device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None).map_err(|e| format!("set layout: {e}"))?;
            let layouts = [set_layout];
            let layout = c.device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts), None).map_err(|e| format!("pipeline layout: {e}"))?;
            let module = c.device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&out.spirv), None).map_err(|e| format!("shader module: {e}"))?;
            let stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(c"main");
            let ci = vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout).flags(vk::PipelineCreateFlags::DISPATCH_BASE);
            let pipeline = c.device.create_compute_pipelines(vk::PipelineCache::null(), &[ci], None).map_err(|(_, e)| format!("compute pipeline: {e}"));
            c.device.destroy_shader_module(module, None);
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
        BindingKind::SgprData | BindingKind::Buffer(_) | BindingKind::Gds => vk::DescriptorType::STORAGE_BUFFER,
        BindingKind::Image(_) => vk::DescriptorType::SAMPLED_IMAGE,
        BindingKind::StorageImage(_) => vk::DescriptorType::STORAGE_IMAGE,
        BindingKind::Sampler(_) => vk::DescriptorType::SAMPLER,
    }
}

fn global_barrier(cmd: vk::CommandBuffer) {
    let mb = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
        .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
    unsafe { ctx().device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::ALL_COMMANDS, vk::DependencyFlags::empty(), &[mb], &[], &[]) };
}

const RING_BYTES: u64 = 8 << 20;
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
                    crate::error!("gpu[{}]: waiting for GPU work: {e}; GPU execution stops (batch shaders: {})", b.name, list.join(" "));
                }
            }
            FLUSHES.fetch_add(1, Ordering::Relaxed);
            collect_timestamps(b.queries, &b.shaders);
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

struct Slot {
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
    desc_pool: vk::DescriptorPool,
    /// Sequence number of the slot's last batch (0: none).
    seq: u64,
    /// Timestamp queries around each dispatch (DS_PROFILE_GPU=1).
    queries: vk::QueryPool,
}

const MAX_TIMED: u32 = 1024;

fn profiling() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_PROFILE_GPU").ok().as_deref() == Some("1"))
}

/// GPU time per shader (DS_PROFILE_GPU=1): address -> (dispatches, ns).
static SHADER_TIME: Mutex<Option<HashMap<u64, (u64, u64)>>> = Mutex::new(None);

/// Reads a completed batch's dispatch timestamps into SHADER_TIME.
fn collect_timestamps(pool: vk::QueryPool, shaders: &[u64]) {
    let n = (shaders.len() as u32).min(MAX_TIMED);
    if pool == vk::QueryPool::null() || n == 0 {
        return;
    }
    let c = ctx();
    let mut data = vec![0u64; 2 * n as usize];
    let r = unsafe { c.device.get_query_pool_results(pool, 0, &mut data, vk::QueryResultFlags::TYPE_64) };
    if r.is_err() {
        return;
    }
    let period = c.limits.timestamp_period as f64;
    let mut g = SHADER_TIME.lock();
    let m = g.get_or_insert_with(HashMap::new);
    for (i, &a) in shaders.iter().take(n as usize).enumerate() {
        let ns = (data[2 * i + 1].saturating_sub(data[2 * i]) as f64 * period) as u64;
        let e = m.entry(a).or_insert((0, 0));
        e.0 += 1;
        e.1 += ns;
    }
}

/// Shaders by total GPU time.
pub fn shader_profile() -> String {
    let g = SHADER_TIME.lock();
    let Some(m) = g.as_ref() else { return String::new() };
    let mut v: Vec<(&u64, &(u64, u64))> = m.iter().collect();
    v.sort_by(|a, b| b.1 .1.cmp(&a.1 .1));
    let total: u64 = m.values().map(|x| x.1).sum();
    let mut s = format!("    gpu time by shader ({} ms in total):\n", total / 1_000_000);
    for (a, (n, ns)) in v.iter().take(15) {
        s.push_str(&format!("      cs {a:#x}: {} ms in {n} dispatches ({} us each)\n", ns / 1_000_000, ns / 1000 / (*n).max(1)));
    }
    s
}

pub struct Exec {
    pool: vk::CommandPool,
    slots: Vec<Slot>,
    cur: usize,
    ring: vk::Buffer,
    ring_ptr: *mut u8,
    /// Offset within the current slot's ring segment.
    ring_used: u64,
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
    reported: HashMap<String, u64>,
    name: String,
    /// Shaders recorded in the current batch (slow-batch reports).
    batch_shaders: Vec<u64>,
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
            let sizes = [
                vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 16384 },
                vk::DescriptorPoolSize { ty: vk::DescriptorType::SAMPLED_IMAGE, descriptor_count: 4096 },
                vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_IMAGE, descriptor_count: 1024 },
                vk::DescriptorPoolSize { ty: vk::DescriptorType::SAMPLER, descriptor_count: 4096 },
            ];
            let mut slots = Vec::new();
            for cmd in cmds {
                slots.push(Slot {
                    cmd,
                    fence: c.device.create_fence(&vk::FenceCreateInfo::default(), None).map_err(|e| e.to_string())?,
                    desc_pool: c.device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(2048).pool_sizes(&sizes), None).map_err(|e| e.to_string())?,
                    seq: 0,
                    queries: if profiling() {
                        c.device.create_query_pool(&vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(2 * MAX_TIMED), None).map_err(|e| e.to_string())?
                    } else {
                        vk::QueryPool::null()
                    },
                });
            }
            let (ring, _mem, ring_ptr) = c.host_buffer(RING_BYTES, vk::BufferUsageFlags::STORAGE_BUFFER)?;
            Ok(Exec {
                pool,
                slots,
                cur: 0,
                ring,
                ring_ptr,
                ring_used: 0,
                ring_align: c.limits.min_storage_buffer_offset_alignment.max(16),
                recording: false,
                pending_writes: Vec::new(),
                actions: Vec::new(),
                checks: Vec::new(),
                inflight: Vec::new(),
                last_seq: 0,
                last_action_seq: 0,
                submit_after: 0,
                reported: HashMap::new(),
                name: name.to_string(),
                batch_shaders: Vec::new(),
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
                ctx().device.cmd_reset_query_pool(self.cmd(), qp, 0, 2 * MAX_TIMED);
            }
        }
        global_barrier(self.cmd());
        self.recording = true;
    }

    /// Submits the recorded batch without waiting for it (its actions run
    /// when it completes).
    pub fn submit(&mut self) {
        if !self.recording {
            return;
        }
        if self.submit_after > done_seq() {
            // Deferred CP writes that this work may read come first.
            wait_done(self.submit_after);
        }
        let c = ctx();
        let i = self.cur;
        let (cmd, fence) = (self.slots[i].cmd, self.slots[i].fence);
        unsafe {
            // Make shader writes to imported memory visible to the host.
            let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::MEMORY_WRITE).dst_access_mask(vk::AccessFlags::HOST_READ);
            c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::HOST, vk::DependencyFlags::empty(), &[mb], &[], &[]);
            c.device.end_command_buffer(cmd).ok();
        }
        let actions = std::mem::take(&mut self.actions);
        let has_actions = !actions.is_empty();
        let name = self.name.clone();
        let seq = push_batch(fence, self.slots[i].queries, actions, std::mem::take(&mut self.checks), std::mem::take(&mut self.batch_shaders), &self.name, || {
            if DEVICE_LOST.load(Ordering::Relaxed) {
                return;
            }
            if let Err(e) = c.submit(cmd, &[], &[], fence) {
                if !DEVICE_LOST.swap(true, Ordering::SeqCst) {
                    crate::error!("gpu[{name}]: {e}; GPU execution stops");
                }
            }
        });
        self.slots[i].seq = seq;
        for w in std::mem::take(&mut self.pending_writes) {
            self.inflight.push((seq, w));
        }
        self.last_seq = seq;
        if has_actions {
            self.last_action_seq = seq;
        }
        self.recording = false;
        self.cur = (self.cur + 1) % SLOTS;
        self.ring_used = 0;
    }

    /// Submits recorded work and waits for all of this queue's GPU work and
    /// deferred effects.
    pub fn flush(&mut self) {
        self.submit();
        wait_done(self.last_seq);
        self.inflight.clear();
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
    /// is the guest range it writes; `ordered` makes later GPU work wait for
    /// it (CP writes that following dispatches may read).
    pub fn after_gpu(&mut self, f: Action, write: Option<(u64, u64)>, ordered: bool) {
        if self.recording {
            self.actions.push(f);
            self.submit();
            if let Some(w) = write {
                self.inflight.push((self.last_seq, w));
            }
            if ordered {
                self.submit_after = self.last_seq;
            }
        } else if self.last_seq > done_seq() {
            let seq = push_batch(vk::Fence::null(), vk::QueryPool::null(), vec![f], Vec::new(), Vec::new(), &self.name, || {});
            self.last_seq = seq;
            self.last_action_seq = seq;
            if let Some(w) = write {
                self.inflight.push((seq, w));
            }
            if ordered {
                self.submit_after = seq;
            }
        } else {
            f();
        }
    }

    fn overlaps_pending(&mut self, reads: &[(u64, u64)]) -> bool {
        let done = done_seq();
        self.inflight.retain(|(seq, _)| *seq > done);
        let ov = |&(a, n): &(u64, u64), &(w, m): &(u64, u64)| w < a + n && a < w + m;
        reads.iter().any(|r| self.pending_writes.iter().any(|w| ov(r, w)) || self.inflight.iter().any(|(_, w)| ov(r, w)))
    }

    /// Allocates SGPR data space in the current slot's ring segment.
    fn ring_alloc(&mut self, bytes: u64) -> Option<u64> {
        let seg = RING_BYTES / SLOTS as u64;
        let off = self.ring_used.next_multiple_of(self.ring_align);
        if off + bytes > seg {
            return None;
        }
        self.ring_used = off + bytes;
        Some(seg * self.cur as u64 + off)
    }

    /// Executes a dispatch of `groups` thread groups with the compute state of `cp`.
    pub fn dispatch(&mut self, cp: &Cp, groups: [u32; 3], initiator: u32) {
        if DEVICE_LOST.load(Ordering::Relaxed) {
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
        if self.overlaps_pending(&p.reads) {
            self.flush();
            p = shaders::prepare_compute(cp, wave64)?;
        }
        // Textures are uploaded from guest memory: earlier GPU writes first.
        let tex_ranges: Vec<(u64, u64)> = p
            .resources
            .iter()
            .filter_map(|r| match r {
                Resolved::Image(t) => Some((t.base, (t.width as u64 * t.height as u64 * t.depth.max(1) as u64 * 16).min(1 << 30))),
                _ => None,
            })
            .collect();
        if self.overlaps_pending(&tex_ranges) {
            self.flush();
        }
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
        let (pipeline, layout, set_layout) = shared().pipeline(&p)?;
        let t_pipe = t0.elapsed();
        // Buffer bindings (and the resource info blocks of the SGPR data).
        let res_info = rdna::translate::res_info_offset(&p.tr.plan) as usize;
        let mut buf_infos: Vec<(u32, vk::DescriptorBufferInfo)> = Vec::new();
        let mut writes_here: Vec<(u64, u64)> = Vec::new();
        {
            let mut sh = shared();
            for b in &p.tr.out.bindings {
                match b.kind {
                    BindingKind::Buffer(i) => {
                        let (base, size, stride) = match &p.resources[i] {
                            Resolved::Buffer(v) => (v.base, v.byte_size(), v.stride),
                            Resolved::Global(a) => (*a, None, 0),
                            _ => return Err("binding/resource mismatch".into()),
                        };
                        let size = match size {
                            Some(n) => n,
                            None => Shared::mapping(base).map(|(_, e)| (e - base).min(UNBOUNDED_MAX)).unwrap_or(0),
                        };
                        let mut info = vk::DescriptorBufferInfo { buffer: sh.dummy, offset: 0, range: DUMMY_BYTES };
                        let mut bias = 0u64;
                        let mut bound = 0u64;
                        if size > 0 {
                            match sh.span(base, size) {
                                Some((buffer, start, end)) => {
                                    let rel = base - start;
                                    let align = ctx().limits.min_storage_buffer_offset_alignment.max(4);
                                    let off = rel & !(align - 1);
                                    bias = rel - off;
                                    let range = (bias + size).next_multiple_of(4).min(end - start - off);
                                    info = vk::DescriptorBufferInfo { buffer, offset: off, range };
                                    bound = size;
                                    if p.tr.plan.resources[i].written {
                                        writes_here.push((base, size));
                                    }
                                }
                                None => self.report(format!("cs {:#x}: buffer {base:#x}+{size:#x} is not mapped GPU memory (bound as zeros)", p.addr)),
                            }
                        }
                        let ri = res_info + RES_INFO_DWORDS as usize * i;
                        p.sgpr_data[ri] = bias as u32;
                        p.sgpr_data[ri + 1] = stride;
                        p.sgpr_data[ri + 2] = bound.min(u32::MAX as u64) as u32;
                        buf_infos.push((b.binding, info));
                    }
                    BindingKind::Gds => buf_infos.push((b.binding, vk::DescriptorBufferInfo { buffer: sh.gds, offset: 0, range: rdna::translate::GDS_BYTES as u64 })),
                    _ => {}
                }
            }
        }
        // Images and samplers.
        let frame = super::FRAMES.load(Ordering::Relaxed);
        let mut img_infos: Vec<(u32, vk::DescriptorType, vk::DescriptorImageInfo)> = Vec::new();
        for b in &p.tr.out.bindings {
            match (b.kind, b.kind_resource().and_then(|i| p.resources.get(i))) {
                (BindingKind::Image(_) | BindingKind::StorageImage(_), Some(Resolved::Image(t))) => {
                    let storage = matches!(b.kind, BindingKind::StorageImage(_));
                    let view = super::images::with(|im| im.view(t, storage, frame))?;
                    let ty = if storage { vk::DescriptorType::STORAGE_IMAGE } else { vk::DescriptorType::SAMPLED_IMAGE };
                    img_infos.push((b.binding, ty, vk::DescriptorImageInfo { sampler: vk::Sampler::null(), image_view: view, image_layout: vk::ImageLayout::GENERAL }));
                }
                (BindingKind::Sampler(_), Some(Resolved::Sampler(sm))) => {
                    let smp = super::images::with(|im| im.sampler(*sm))?;
                    img_infos.push((b.binding, vk::DescriptorType::SAMPLER, vk::DescriptorImageInfo { sampler: smp, image_view: vk::ImageView::null(), image_layout: vk::ImageLayout::UNDEFINED }));
                }
                (BindingKind::Image(_) | BindingKind::StorageImage(_) | BindingKind::Sampler(_), _) => return Err("binding/resource mismatch".into()),
                _ => {}
            }
        }
        let t_bind = t0.elapsed();
        // The slot must be free before its ring segment is written.
        self.begin();
        // SGPR data.
        let bytes: Vec<u8> = p.sgpr_data.iter().flat_map(|w| w.to_le_bytes()).collect();
        let off = match self.ring_alloc(bytes.len() as u64) {
            Some(o) => o,
            None => {
                self.flush();
                self.begin();
                self.ring_alloc(bytes.len() as u64).ok_or("SGPR data larger than the ring")?
            }
        };
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.ring_ptr.add(off as usize), bytes.len()) };
        for b in &p.tr.out.bindings {
            if b.kind == BindingKind::SgprData {
                buf_infos.push((b.binding, vk::DescriptorBufferInfo { buffer: self.ring, offset: off, range: bytes.len() as u64 }));
            }
        }
        self.begin();
        let c = ctx();
        let layouts = [set_layout];
        let ai = vk::DescriptorSetAllocateInfo::default().set_layouts(&layouts);
        let set = match unsafe { c.device.allocate_descriptor_sets(&ai.descriptor_pool(self.slots[self.cur].desc_pool)) } {
            Ok(v) => v[0],
            Err(_) => {
                // Pool exhausted: complete the recorded work (resets the pool).
                // The SGPR data was written to the ring, which flush resets: rewrite it.
                self.flush();
                let off2 = self.ring_alloc(bytes.len() as u64).ok_or("ring")?;
                unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.ring_ptr.add(off2 as usize), bytes.len()) };
                for (binding, info) in buf_infos.iter_mut() {
                    if p.tr.out.bindings.iter().any(|b| b.binding == *binding && b.kind == BindingKind::SgprData) {
                        info.offset = off2;
                    }
                }
                self.begin();
                unsafe { c.device.allocate_descriptor_sets(&ai.descriptor_pool(self.slots[self.cur].desc_pool)) }.map_err(|e| format!("descriptor set: {e}"))?[0]
            }
        };
        let mut writes: Vec<vk::WriteDescriptorSet> = buf_infos
            .iter()
            .map(|(binding, info)| vk::WriteDescriptorSet::default().dst_set(set).dst_binding(*binding).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(std::slice::from_ref(info)))
            .collect();
        for (binding, ty, info) in &img_infos {
            writes.push(vk::WriteDescriptorSet::default().dst_set(set).dst_binding(*binding).descriptor_type(*ty).image_info(std::slice::from_ref(info)));
        }
        let start = [cp.reg(Space::Sh, 0x204), cp.reg(Space::Sh, 0x205), cp.reg(Space::Sh, 0x206)];
        unsafe {
            c.device.update_descriptor_sets(&writes, &[]);
            let cmd = self.cmd();
            let qp = self.slots[self.cur].queries;
            let ti = self.batch_shaders.len() as u32;
            if qp != vk::QueryPool::null() && ti < MAX_TIMED {
                c.device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, qp, 2 * ti);
            }
            c.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            c.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, layout, 0, &[set], &[]);
            if start != [0, 0, 0] {
                c.device.cmd_dispatch_base(cmd, start[0], start[1], start[2], groups[0].saturating_sub(start[0]), groups[1].saturating_sub(start[1]), groups[2].saturating_sub(start[2]));
            } else {
                c.device.cmd_dispatch(cmd, groups[0], groups[1], groups[2]);
            }
            if qp != vk::QueryPool::null() && ti < MAX_TIMED {
                c.device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, qp, 2 * ti + 1);
            }
        }
        global_barrier(self.cmd());
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
        if let Some(ch) = verify::check_for(&p, groups, local) {
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
        "    gpu exec: {} dispatches executed, {} skipped, {} completions waited, {}; waits by opcode {}\n",
        EXECUTED.load(Ordering::Relaxed),
        SKIPPED.load(Ordering::Relaxed),
        FLUSHES.load(Ordering::Relaxed),
        verify::status(),
        super::cp::sync_summary()
    ) + &format!("    gpu completion: {} ms waiting for fences, {} ms in actions
", FENCE_NS.load(Ordering::Relaxed) / 1_000_000, ACTION_NS.load(Ordering::Relaxed) / 1_000_000)
}
