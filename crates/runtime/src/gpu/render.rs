//! Vulkan backend: executes draws and dispatches recorded by the command
//! processor using translated shaders.

use super::cp::{CommandProcessor, Hooks, Regs};
use super::desc::{self, Tsharp};
use super::tiling;
use super::resources::{self, Image, ImageKey, Ring, Samplers};
use super::shaders::{self, Prepared, ResolvedRes, StageRegs};
use super::vk::ctx;
use ash::vk;
use gcn::translate::{BindingKind, Stage};
use parking_lot::Mutex;
use std::collections::HashMap;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct Attachment {
    key: ImageKey,
    view_format: vk::Format,
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
struct GfxKey {
    vs: u64,
    ps: Option<u64>,
    rect: bool,
    gs_params: u32,
    colors: Vec<vk::Format>,
    blend: Vec<u32>,
    masks: Vec<u32>,
    depth: vk::Format,
    stencil: bool,
    topology_class: u8,
    polygon: u8,
    depth_clamp: bool,
}

struct Pipe {
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    set_layout: vk::DescriptorSetLayout,
}

struct Rendering {
    colors: Vec<Option<Attachment>>,
    depth: Option<Attachment>,
    extent: (u32, u32),
}

/// Buffer bindings at least this large are accessed in place (zero-copy) in
/// guest memory imported into Vulkan instead of being copied through the
/// rings (titles bind multi-GB ranges as scatter destinations).
const ZC_MIN_BYTES: u64 = 16 << 20;
/// Largest written buffer binding treated as a write of whatever surfaces it
/// overlaps (render-target reload, HTILE/CMASK checks); larger bindings are
/// sparse scatter destinations.
const SURFACE_WRITE_MAX: u64 = 64 << 20;

/// Number of batches that can be in flight (each with its own command
/// buffer, fence, descriptor pool and ring segment).
const SLOTS: usize = 6;

struct Slot {
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
    desc_pool: vk::DescriptorPool,
    /// Sequence number of the batch last submitted from this slot (0 = none).
    seq: u64,
}

/// Work performed (in submission order) when a batch completes.
struct Completion {
    seq: u64,
    fence: vk::Fence,
    ring_ptr: *mut u8,
    writebacks: Vec<(u64, u64, u64)>,
    /// Guest ranges written in place by zero-copy bindings (ordering only).
    zc_writes: Vec<(u64, u64)>,
    actions: Vec<super::cp::EopAction>,
    submitted: std::time::Instant,
}

unsafe impl Send for Completion {}
unsafe impl Sync for Completion {}

/// Submitted batches not yet completed, oldest first. An entry stays queued
/// until its write-backs and actions have been performed, so scans for
/// pending guest-memory writes see it for its whole lifetime.
static COMPLETIONS: parking_lot::Mutex<std::collections::VecDeque<std::sync::Arc<Completion>>> = parking_lot::Mutex::new(std::collections::VecDeque::new());
static COMPLETIONS_CV: parking_lot::Condvar = parking_lot::Condvar::new();
static DONE: parking_lot::Mutex<u64> = parking_lot::Mutex::new(0);
static DONE_CV: parking_lot::Condvar = parking_lot::Condvar::new();
/// Diagnostics (BB_PERF): host-observed GPU busy time and CP waits for
/// pending guest-memory writes, in nanoseconds.
pub static GPU_BUSY_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static SYNC_WAIT_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Last completed batch: fence wait and write-back durations (ns), bytes written back.
static LAST_FENCE_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static LAST_WB_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static LAST_WB_BYTES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn push_completion(c: Completion) {
    COMPLETIONS.lock().push_back(std::sync::Arc::new(c));
    COMPLETIONS_CV.notify_one();
}

/// Waits for submitted batches, then performs their buffer write-backs and
/// end-of-pipe actions in order.
fn completion_thread() {
    crate::log::set_thread_name("GPU-complete");
    let mut last_done = std::time::Instant::now();
    loop {
        let c = {
            let mut q = COMPLETIONS.lock();
            while q.is_empty() {
                COMPLETIONS_CV.wait(&mut q);
            }
            q.front().unwrap().clone()
        };
        let tf = std::time::Instant::now();
        if c.fence != vk::Fence::null() {
            if let Err(e) = unsafe { ctx().device.wait_for_fences(&[c.fence], true, 10_000_000_000) } {
                crate::error!("render wait: {e} (GPU hang or device lost)");
            }
            let now = std::time::Instant::now();
            let start = c.submitted.max(last_done);
            GPU_BUSY_NS.fetch_add(now.saturating_duration_since(start).as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
            last_done = now;
        }
        let tw = std::time::Instant::now();
        LAST_FENCE_NS.store((tw - tf).as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        let mut wb_bytes = 0;
        for (addr, off, size) in &c.writebacks {
            let src = unsafe { std::slice::from_raw_parts(c.ring_ptr.add(*off as usize), *size as usize) };
            resources::copy_to_guest(src, *addr);
            wb_bytes += size;
        }
        LAST_WB_NS.store(tw.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        LAST_WB_BYTES.store(wb_bytes, std::sync::atomic::Ordering::Relaxed);
        for a in &c.actions {
            a.perform();
        }
        COMPLETIONS.lock().pop_front();
        *DONE.lock() = c.seq;
        DONE_CV.notify_all();
    }
}

/// Whether a submitted, not yet completed batch will write back GPU-written
/// buffer data overlapping guest memory [addr, addr+len).
fn writeback_in_flight(addr: u64, len: u64) -> bool {
    COMPLETIONS
        .lock()
        .iter()
        .any(|c| c.writebacks.iter().any(|&(w, _, n)| w < addr + len && addr < w + n) || c.zc_writes.iter().any(|&(w, n)| w < addr + len && addr < w + n))
}

/// Waits for in-flight batches whose completion writes guest memory in
/// [addr, addr+len) (end-of-pipe writes or buffer write-backs).
pub fn wait_overlapping(addr: u64, len: u64) {
    let seq = COMPLETIONS
        .lock()
        .iter()
        .filter(|c| {
            c.actions.iter().any(|a| a.data_sel != 0 && a.addr < addr + len && addr < a.addr + 8)
                || c.writebacks.iter().any(|&(w, _, n)| w < addr + len && addr < w + n)
                || c.zc_writes.iter().any(|&(w, n)| w < addr + len && addr < w + n)
        })
        .map(|c| c.seq)
        .max();
    if let Some(seq) = seq {
        let t0 = std::time::Instant::now();
        wait_done(seq);
        SYNC_WAIT_NS.fetch_add(t0.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Blocks until the batch with sequence number `seq` (and all before it) completed.
pub fn wait_done(seq: u64) {
    let mut d = DONE.lock();
    while *d < seq {
        DONE_CV.wait(&mut d);
    }
}

pub struct Renderer {
    pool: vk::CommandPool,
    slots: Vec<Slot>,
    cur: usize,
    /// Sequence number of the last submitted batch.
    last_seq: u64,
    /// Current slot's command buffer and descriptor pool.
    cmd: vk::CommandBuffer,
    recording: bool,
    ring: Ring,
    desc_pool: vk::DescriptorPool,
    batch: u64,
    /// Texture validation epoch: CPU-sourced textures are re-hashed at most
    /// once per epoch. It advances per title submission and whenever guest
    /// memory may have been modified by the GPU side (write-backs, DMA).
    epoch: u64,
    images: HashMap<ImageKey, Image>,
    by_base: HashMap<u64, Vec<ImageKey>>,
    samplers: Samplers,
    modules: HashMap<u64, vk::ShaderModule>,
    gfx_pipes: HashMap<GfxKey, Pipe>,
    cs_pipes: HashMap<u64, Pipe>,
    rendering: Option<Rendering>,
    barrier_needed: bool,
    /// Device-local ring segment (same slot rotation as `ring`).
    dring: Ring,
    /// Buffers written by shaders in the current batch, in program order.
    writebacks: Vec<PendingWrite>,
    /// Imported guest memory spans for zero-copy bindings: (start, end, buffer, memory).
    zc_spans: Vec<(u64, u64, vk::Buffer, vk::DeviceMemory)>,
    /// `mem::MAP_GEN` the spans were validated against.
    zc_gen: u64,
    /// Guest ranges written in place by zero-copy bindings in the current batch.
    zc_writes: Vec<(u64, u64)>,
    /// Accumulated GPU-thread time per activity (see PROF_NAMES).
    pub prof: [std::time::Duration; 6],
    /// Synchronizing flushes by reason (see FLUSH_REASONS).
    pub flushes: [u64; 6],
    /// Texture uploads performed and guest bytes hashed.
    pub uploads: u64,
    pub hashed_bytes: u64,
    /// Reported problems and how often they occurred.
    reported: HashMap<String, u64>,
    pub draws: u64,
    pub dispatches: u64,
    pub skipped: u64,
    /// Global data share (GDS_BYTES, device-local; zero-initialized).
    gds: vk::Buffer,
    /// RT image write counters for texture shadow copies.
    write_gen: HashMap<ImageKey, u64>,
    shadows: HashMap<(ImageKey, u32, u32, vk::Format), (ImageKey, u64)>,
}

unsafe impl Send for Renderer {}

/// A guest buffer range written by a shader in the current batch: the shader
/// works on `dev_off` in the device ring; at batch end the data is copied to
/// `host_off` in the host ring, from where the completion thread writes it
/// back to guest memory.
#[derive(Clone, Copy)]
struct PendingWrite {
    base: u64,
    size: u64,
    host_off: u64,
    dev_off: u64,
}

pub static RENDERER: Mutex<Option<Renderer>> = Mutex::new(None);

pub fn init() -> Result<(), String> {
    let c = ctx();
    unsafe {
        let pool = c
            .device
            .create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(c.queue_family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER), None)
            .map_err(|e| e.to_string())?;
        let cmds = c.device.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(pool).command_buffer_count(SLOTS as u32)).map_err(|e| e.to_string())?;
        let sizes = [
            vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 32768 },
            vk::DescriptorPoolSize { ty: vk::DescriptorType::SAMPLED_IMAGE, descriptor_count: 32768 },
            vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_IMAGE, descriptor_count: 8192 },
            vk::DescriptorPoolSize { ty: vk::DescriptorType::SAMPLER, descriptor_count: 32768 },
        ];
        let mut slots = Vec::with_capacity(SLOTS);
        for &cmd in &cmds {
            slots.push(Slot {
                cmd,
                fence: c.device.create_fence(&vk::FenceCreateInfo::default(), None).map_err(|e| e.to_string())?,
                desc_pool: c.device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(8192).pool_sizes(&sizes), None).map_err(|e| e.to_string())?,
                seq: 0,
            });
        }
        let mut ring = Ring::new(c, resources::RING_SIZE)?;
        ring.set_segment(0, SLOTS);
        let mut dring = Ring::new_device(c, resources::DEV_RING_SIZE)?;
        dring.set_segment(0, SLOTS);
        let (gds, _gds_mem) = c.device_buffer(
            gcn::translate::GDS_BYTES as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
        )?;
        {
            // Zero the GDS once.
            let cmd = slots[0].cmd;
            c.device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).map_err(|e| e.to_string())?;
            c.device.cmd_fill_buffer(cmd, gds, 0, vk::WHOLE_SIZE, 0);
            c.device.end_command_buffer(cmd).map_err(|e| e.to_string())?;
            c.submit(cmd, &[], &[], slots[0].fence)?;
            c.device.wait_for_fences(&[slots[0].fence], true, u64::MAX).map_err(|e| e.to_string())?;
            c.device.reset_fences(&[slots[0].fence]).map_err(|e| e.to_string())?;
            c.device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty()).map_err(|e| e.to_string())?;
        }
        crate::thread::spawn_host("GPU-complete", 1 << 20, Box::new(completion_thread)).map_err(|e| e.to_string())?;
        *RENDERER.lock() = Some(Renderer {
            pool,
            cmd: slots[0].cmd,
            desc_pool: slots[0].desc_pool,
            slots,
            cur: 0,
            last_seq: 0,
            recording: false,
            epoch: 0,
            ring,
            batch: 0,
            images: HashMap::new(),
            by_base: HashMap::new(),
            samplers: Samplers::new(),
            modules: HashMap::new(),
            gfx_pipes: HashMap::new(),
            cs_pipes: HashMap::new(),
            rendering: None,
            barrier_needed: false,
            dring,
            writebacks: Vec::new(),
            zc_spans: Vec::new(),
            zc_gen: 0,
            zc_writes: Vec::new(),
            prof: Default::default(),
            flushes: [0; 6],
            uploads: 0,
            hashed_bytes: 0,
            reported: HashMap::new(),
            draws: 0,
            dispatches: 0,
            skipped: 0,
            gds,
            write_gen: HashMap::new(),
            shadows: HashMap::new(),
        });
    }
    crate::info!("renderer initialized (ring {} MiB)", resources::RING_SIZE >> 20);
    Ok(())
}

fn global_barrier(cmd: vk::CommandBuffer) {
    let c = ctx();
    let mb = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
        .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
    unsafe {
        c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::ALL_COMMANDS, vk::DependencyFlags::empty(), &[mb], &[], &[]);
    }
}

fn f32r(v: u32) -> f32 {
    f32::from_bits(v)
}

/// CB color FORMAT/NUMBER_TYPE/COMP_SWAP -> Vulkan format.
fn cb_format(fmt: u32, ntype: u32, swap: u32) -> Option<vk::Format> {
    let nfmt = match ntype {
        6 => 9,
        n => n as u8,
    };
    let f = desc::vk_format(fmt as u8, nfmt)?;
    Some(match (f, swap) {
        (vk::Format::R8G8B8A8_UNORM, 1) => vk::Format::B8G8R8A8_UNORM,
        (vk::Format::R8G8B8A8_SRGB, 1) => vk::Format::B8G8R8A8_SRGB,
        (vk::Format::R8G8B8A8_SNORM, 1) => vk::Format::B8G8R8A8_SNORM,
        _ => f,
    })
}

fn blend_factor(f: u32) -> vk::BlendFactor {
    use vk::BlendFactor as B;
    match f {
        0 => B::ZERO,
        1 => B::ONE,
        2 => B::SRC_COLOR,
        3 => B::ONE_MINUS_SRC_COLOR,
        4 => B::SRC_ALPHA,
        5 => B::ONE_MINUS_SRC_ALPHA,
        6 => B::DST_ALPHA,
        7 => B::ONE_MINUS_DST_ALPHA,
        8 => B::DST_COLOR,
        9 => B::ONE_MINUS_DST_COLOR,
        10 => B::SRC_ALPHA_SATURATE,
        13 => B::CONSTANT_COLOR,
        14 => B::ONE_MINUS_CONSTANT_COLOR,
        15 => B::SRC1_COLOR,
        16 => B::ONE_MINUS_SRC1_COLOR,
        17 => B::SRC1_ALPHA,
        18 => B::ONE_MINUS_SRC1_ALPHA,
        19 => B::CONSTANT_ALPHA,
        20 => B::ONE_MINUS_CONSTANT_ALPHA,
        _ => B::ONE,
    }
}

fn blend_op(f: u32) -> vk::BlendOp {
    match f {
        1 => vk::BlendOp::SUBTRACT,
        2 => vk::BlendOp::MIN,
        3 => vk::BlendOp::MAX,
        4 => vk::BlendOp::REVERSE_SUBTRACT,
        _ => vk::BlendOp::ADD,
    }
}

fn stencil_op(op: u32) -> vk::StencilOp {
    match op {
        1 => vk::StencilOp::ZERO,
        3 | 4 => vk::StencilOp::REPLACE,
        5 => vk::StencilOp::INCREMENT_AND_CLAMP,
        6 => vk::StencilOp::DECREMENT_AND_CLAMP,
        7 => vk::StencilOp::INVERT,
        8 => vk::StencilOp::INCREMENT_AND_WRAP,
        9 => vk::StencilOp::DECREMENT_AND_WRAP,
        _ => vk::StencilOp::KEEP,
    }
}

fn scissor_rect(tl: u32, br: u32) -> (i32, i32, i32, i32) {
    ((tl & 0x7FFF) as i32, ((tl >> 16) & 0x7FFF) as i32, (br & 0x7FFF) as i32, ((br >> 16) & 0x7FFF) as i32)
}

impl Renderer {
    fn report(&mut self, msg: String) {
        let n = self.reported.entry(msg.clone()).or_insert(0);
        *n += 1;
        if *n == 1 {
            crate::warn!("render: {msg}");
        }
    }

    fn begin(&mut self) {
        if self.recording {
            return;
        }
        let c = ctx();
        unsafe {
            c.device.begin_command_buffer(self.cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).ok();
        }
        global_barrier(self.cmd);
        self.recording = true;
    }

    fn end_rendering(&mut self) {
        if self.rendering.take().is_some() {
            unsafe { ctx().device.cmd_end_rendering(self.cmd) };
            self.barrier_needed = true;
        }
    }

    /// `flush` with accounting of why the synchronization happened.
    pub fn flush_for(&mut self, why: usize) {
        if self.recording {
            self.flushes[why.min(5)] += 1;
        }
        self.flush();
    }

    /// Starts a new texture validation epoch.
    pub fn new_epoch(&mut self) {
        self.epoch += 1;
    }

    /// Submits recorded work and waits until it, its buffer write-backs and
    /// all earlier end-of-pipe actions have completed.
    pub fn flush(&mut self) {
        if !self.recording && *DONE.lock() >= self.last_seq {
            return;
        }
        if !self.writebacks.is_empty() {
            self.epoch += 1;
        }
        super::set_phase(super::phase::SUBMIT_WAIT, self.batch);
        let seq = self.submit_batch(Vec::new());
        let t0 = std::time::Instant::now();
        wait_done(seq);
        self.prof[0] += t0.elapsed();
    }

    /// Submits the recorded work (if any) without waiting; `actions` are
    /// performed after it and everything before it has completed. Returns the
    /// batch's sequence number.
    pub fn submit_batch(&mut self, actions: Vec<super::cp::EopAction>) -> u64 {
        let pending = std::mem::take(&mut self.writebacks);
        if self.recording && !pending.is_empty() {
            // Shader-written data: device ring -> host ring, visible to the host
            // once the batch's fence signals.
            self.end_rendering();
            global_barrier(self.cmd);
            let regions: Vec<vk::BufferCopy> = pending.iter().map(|p| vk::BufferCopy { src_offset: p.dev_off, dst_offset: p.host_off, size: p.size }).collect();
            let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::HOST_READ);
            unsafe {
                let c = ctx();
                c.device.cmd_copy_buffer(self.cmd, self.dring.buffer, self.ring.buffer, &regions);
                c.device.cmd_pipeline_barrier(self.cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::HOST, vk::DependencyFlags::empty(), &[mb], &[], &[]);
            }
        }
        let writebacks: Vec<(u64, u64, u64)> = pending.iter().map(|p| (p.base, p.host_off, p.size)).collect();
        let zc_writes = std::mem::take(&mut self.zc_writes);
        if self.recording && !zc_writes.is_empty() {
            // Shader writes to imported guest memory: visible to the host after the fence.
            self.end_rendering();
            let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::MEMORY_WRITE).dst_access_mask(vk::AccessFlags::HOST_READ);
            unsafe { ctx().device.cmd_pipeline_barrier(self.cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::HOST, vk::DependencyFlags::empty(), &[mb], &[], &[]) };
        }
        if !self.recording {
            if actions.is_empty() && writebacks.is_empty() {
                return self.last_seq;
            }
            // Ordering-only completion (nothing new to wait for on the GPU).
            self.last_seq += 1;
            push_completion(Completion { seq: self.last_seq, fence: vk::Fence::null(), ring_ptr: self.ring.ptr, writebacks, zc_writes, actions, submitted: std::time::Instant::now() });
            return self.last_seq;
        }
        self.end_rendering();
        let c = ctx();
        let slot = &mut self.slots[self.cur];
        unsafe {
            c.device.end_command_buffer(slot.cmd).ok();
            if let Err(e) = c.submit(slot.cmd, &[], &[], slot.fence) {
                crate::error!("render submit: {e}");
            }
        }
        self.last_seq += 1;
        slot.seq = self.last_seq;
        push_completion(Completion { seq: self.last_seq, fence: slot.fence, ring_ptr: self.ring.ptr, writebacks, zc_writes, actions, submitted: std::time::Instant::now() });
        self.recording = false;
        self.batch += 1;
        // Move to the next slot once its previous batch has completed.
        self.cur = (self.cur + 1) % SLOTS;
        let next = &self.slots[self.cur];
        if next.seq != 0 {
            let t0 = std::time::Instant::now();
            wait_done(next.seq);
            self.prof[0] += t0.elapsed();
            unsafe {
                c.device.reset_fences(&[next.fence]).ok();
                c.device.reset_command_buffer(next.cmd, vk::CommandBufferResetFlags::empty()).ok();
                c.device.reset_descriptor_pool(next.desc_pool, vk::DescriptorPoolResetFlags::empty()).ok();
            }
        }
        self.cmd = next.cmd;
        self.desc_pool = next.desc_pool;
        self.ring.set_segment(self.cur, SLOTS);
        self.dring.set_segment(self.cur, SLOTS);
        self.last_seq
    }

    fn module(&mut self, key: u64, spirv: &[u32]) -> Result<vk::ShaderModule, String> {
        if let Some(&m) = self.modules.get(&key) {
            return Ok(m);
        }
        let m = unsafe { ctx().device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(spirv), None) }.map_err(|e| format!("vkCreateShaderModule: {e}"))?;
        self.modules.insert(key, m);
        Ok(m)
    }

    fn set_layout(&mut self, stages: &[&Prepared]) -> Result<vk::DescriptorSetLayout, String> {
        let mut bindings = Vec::new();
        for p in stages {
            let flags = match p.stage {
                Stage::Vertex => vk::ShaderStageFlags::VERTEX,
                Stage::Fragment => vk::ShaderStageFlags::FRAGMENT,
                Stage::Compute => vk::ShaderStageFlags::COMPUTE,
            };
            for b in &p.translated.out.bindings {
                let ty = match b.kind {
                    BindingKind::SgprData | BindingKind::Buffer(_) | BindingKind::Gds => vk::DescriptorType::STORAGE_BUFFER,
                    BindingKind::Image(_) => vk::DescriptorType::SAMPLED_IMAGE,
                    BindingKind::StorageImage(_) => vk::DescriptorType::STORAGE_IMAGE,
                    BindingKind::Sampler(_) => vk::DescriptorType::SAMPLER,
                };
                bindings.push(vk::DescriptorSetLayoutBinding::default().binding(b.binding).descriptor_type(ty).descriptor_count(1).stage_flags(flags));
            }
        }
        unsafe { ctx().device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None) }.map_err(|e| format!("descriptor set layout: {e}"))
    }

    fn rect_gs(&mut self, params: u32) -> Result<vk::ShaderModule, String> {
        let key = 0x5245_4354_0000_0000 | params as u64;
        if let Some(&m) = self.modules.get(&key) {
            return Ok(m);
        }
        let spv = gcn::translate::gs::rect_list_gs(params);
        self.module(key, &spv)
    }

    // -------------------------------------------------------------------
    // Images

    fn register(&mut self, img: Image) {
        let k = img.key;
        self.by_base.entry(k.base).or_default().push(k);
        self.images.insert(k, img);
    }

    fn init_image(&mut self, key: ImageKey) -> Result<(), String> {
        if self.images.contains_key(&key) {
            return Ok(());
        }
        let img = resources::create_image(key)?;
        self.begin();
        self.end_rendering();
        let c = ctx();
        let b = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(img.image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: resources::aspect(key.format) | if key.format == vk::Format::D32_SFLOAT_S8_UINT { vk::ImageAspectFlags::STENCIL } else { vk::ImageAspectFlags::empty() },
                base_mip_level: 0,
                level_count: vk::REMAINING_MIP_LEVELS,
                base_array_layer: 0,
                layer_count: vk::REMAINING_ARRAY_LAYERS,
            });
        unsafe {
            c.device.cmd_pipeline_barrier(self.cmd, vk::PipelineStageFlags::TOP_OF_PIPE, vk::PipelineStageFlags::ALL_COMMANDS, vk::DependencyFlags::empty(), &[], &[], &[b]);
            let range = vk::ImageSubresourceRange {
                aspect_mask: resources::aspect(key.format),
                base_mip_level: 0,
                level_count: vk::REMAINING_MIP_LEVELS,
                base_array_layer: 0,
                layer_count: vk::REMAINING_ARRAY_LAYERS,
            };
            if resources::is_depth(key.format) {
                let mut r = range;
                if key.format == vk::Format::D32_SFLOAT_S8_UINT {
                    r.aspect_mask |= vk::ImageAspectFlags::STENCIL;
                }
                c.device.cmd_clear_depth_stencil_image(self.cmd, img.image, vk::ImageLayout::GENERAL, &vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 }, &[r]);
            } else if !matches!(key.format.as_raw(), 131..=146) {
                c.device.cmd_clear_color_image(self.cmd, img.image, vk::ImageLayout::GENERAL, &vk::ClearColorValue::default(), &[range]);
            }
        }
        self.barrier_needed = true;
        self.register(img);
        Ok(())
    }

    /// Render target image for a color buffer (padded dimensions from pitch/slice).
    fn color_target(&mut self, regs: &Regs, n: u32) -> Result<Option<Attachment>, String> {
        let r = 0xA318 + n * 0xF;
        let info = regs.ctx(r + 4);
        let fmt = (info >> 2) & 0x1F;
        if fmt == 0 {
            return Ok(None);
        }
        let base = (regs.ctx(r) as u64) << 8;
        let pitch = ((regs.ctx(r + 1) & 0x7FF) + 1) * 8;
        let slice = ((regs.ctx(r + 2) & 0x3F_FFFF) + 1) * 64;
        let height = slice / pitch;
        let view = regs.ctx(r + 3);
        let layers = ((view >> 13) & 0x7FF) + 1;
        let Some(format) = cb_format(fmt, (info >> 8) & 7, (info >> 11) & 3) else {
            self.report(format!("unsupported color format {fmt} type {} swap {}", (info >> 8) & 7, (info >> 11) & 3));
            return Ok(None);
        };
        // Storage class: use a canonical (non-sRGB) image format and an sRGB view where needed.
        let storage_fmt = canonical(format);
        let key = ImageKey { base, width: pitch, height, layers: layers.max(1), levels: 1, format: storage_fmt, depth: false, ty3d: false };
        let key = self.find_compatible(key).unwrap_or(key);
        self.init_image(key)?;
        if self.images[&key].guest_newer {
            self.refresh_from_guest(key, true)?;
        }
        // CB_COLOR*_ATTRIB.TILE_MODE_INDEX: guest surface layout for write-back.
        let img = self.images.get_mut(&key).unwrap();
        img.tile_index = Some((regs.ctx(r + 5) & 0x1F) as u8);
        // CB_COLOR*_INFO.FAST_CLEAR: CMASK (4 bits per 8x8 tile) tracks fast clears.
        let cmask = (regs.ctx(r + 7) as u64) << 8;
        img.meta = (info & (1 << 13) != 0 && cmask != 0).then(|| (cmask, (pitch.div_ceil(8) as u64 * height.div_ceil(8) as u64).div_ceil(2)));
        Ok(Some(Attachment { key, view_format: format }))
    }

    fn depth_target(&mut self, regs: &Regs) -> Result<Option<Attachment>, String> {
        let zinfo = regs.ctx(0xA010);
        let zfmt = zinfo & 3;
        let sinfo = regs.ctx(0xA011);
        let has_stencil = sinfo & 1 != 0;
        if zfmt == 0 && !has_stencil {
            return Ok(None);
        }
        let base = (regs.ctx(0xA014) as u64) << 8;
        let size = regs.ctx(0xA016);
        let pitch = ((size & 0x7FF) + 1) * 8;
        let height = (((size >> 11) & 0x7FF) + 1) * 8;
        let format = match (zfmt, has_stencil) {
            (1, false) => vk::Format::D16_UNORM,
            (_, true) => vk::Format::D32_SFLOAT_S8_UINT,
            _ => vk::Format::D32_SFLOAT,
        };
        let key = ImageKey { base, width: pitch, height, layers: 1, levels: 1, format, depth: true, ty3d: false };
        self.init_image(key)?;
        if self.images[&key].guest_newer {
            self.refresh_from_guest(key, true)?;
        }
        // DB_Z_INFO.TILE_MODE_INDEX
        let img = self.images.get_mut(&key).unwrap();
        img.tile_index = Some(((zinfo >> 20) & 7) as u8);
        // DB_Z_INFO.TILE_SURFACE_ENABLE: HTILE (one dword per 8x8 tile).
        let htile = (regs.ctx(0xA005) as u64) << 8;
        img.meta = (zinfo & (1 << 29) != 0 && htile != 0).then(|| (htile, pitch.div_ceil(8) as u64 * height.div_ceil(8) as u64 * 4));
        Ok(Some(Attachment { key, view_format: format }))
    }

    /// An existing image at the same base with the same size and format class.
    fn find_compatible(&self, key: ImageKey) -> Option<ImageKey> {
        self.by_base.get(&key.base)?.iter().copied().find(|k| {
            k.width == key.width && k.height == key.height && k.layers == key.layers && k.ty3d == key.ty3d && k.depth == key.depth && format_class(k.format) == format_class(key.format)
        })
    }

    /// Stand-in for a T# that cannot be honored (invalid descriptor, address
    /// outside guest memory, or a format we do not implement): a zeroed 1x1
    /// image of the expected view type and numeric class, so the rest of the
    /// draw still runs. GCN returns zeros when sampling an invalid
    /// descriptor; titles also bind such descriptors for unused slots.
    fn dummy_texture(&mut self, t: &Tsharp, storage: bool, why: &str) -> Result<vk::ImageView, String> {
        self.report(format!("texture stand-in (zero image): {why}: dfmt {} nfmt {} type {}", t.dfmt, t.nfmt, t.ty));
        let format = match desc::num_class(t.nfmt) {
            gcn::translate::NumClass::Uint => vk::Format::R8G8B8A8_UINT,
            gcn::translate::NumClass::Sint => vk::Format::R8G8B8A8_SINT,
            _ => vk::Format::R8G8B8A8_UNORM,
        };
        let (view_type, layers, ty3d) = match t.ty {
            desc::T_3D => (vk::ImageViewType::TYPE_3D, 1, true),
            desc::T_CUBE if t.last_array > t.base_array => (vk::ImageViewType::CUBE_ARRAY, 6, false),
            desc::T_CUBE => (vk::ImageViewType::CUBE, 6, false),
            desc::T_1D_ARRAY | desc::T_2D_ARRAY | desc::T_2D_MSAA_ARRAY => (vk::ImageViewType::TYPE_2D_ARRAY, 1, false),
            _ => (vk::ImageViewType::TYPE_2D, 1, false),
        };
        const DUMMY_BASE: u64 = 0xD000_0000_0000_0000;
        let key = ImageKey {
            base: DUMMY_BASE + ((format.as_raw() as u64) << 8) + view_type.as_raw() as u64,
            width: 1,
            height: 1,
            layers,
            levels: 1,
            format,
            depth: false,
            ty3d,
        };
        self.init_image(key)?;
        let img = self.images.get_mut(&key).unwrap();
        img.gpu_written = true; // never sourced from guest memory
        const IDENTITY_SEL: u16 = 4 | (5 << 3) | (6 << 6) | (7 << 9);
        img.view(format, IDENTITY_SEL, view_type, storage, 0, layers)
    }

    /// Image view for sampling/storing a T#.
    fn texture(&mut self, t: &Tsharp, storage: bool) -> Result<vk::ImageView, String> {
        let Some(vfmt) = desc::vk_format(t.dfmt, t.nfmt) else {
            return self.dummy_texture(t, storage, "unsupported format");
        };
        if t.base < crate::mem::GUEST_BASE || t.base >= crate::mem::GUEST_END {
            static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 12 {
                crate::info!("texture outside guest memory: base {:#x} {}x{} dfmt {} nfmt {} type {}", t.base, t.width, t.height, t.dfmt, t.nfmt, t.ty);
            }
            return self.dummy_texture(t, storage, "address outside guest memory");
        }
        let layers = t.layers().max(1);
        let total_layers = t.total_layers().max(1);
        let view_base = t.view_base_layer();
        let ty3d = t.ty == desc::T_3D;
        let levels = 1; // base level only for now
        let view_type = match t.ty {
            desc::T_1D => vk::ImageViewType::TYPE_1D,
            desc::T_3D => vk::ImageViewType::TYPE_3D,
            desc::T_CUBE => {
                if layers > 6 { vk::ImageViewType::CUBE_ARRAY } else { vk::ImageViewType::CUBE }
            }
            desc::T_1D_ARRAY => vk::ImageViewType::TYPE_1D_ARRAY,
            desc::T_2D_ARRAY | desc::T_2D_MSAA_ARRAY => vk::ImageViewType::TYPE_2D_ARRAY,
            _ => vk::ImageViewType::TYPE_2D,
        };
        // 1D textures are stored as 2D images with height 1; use 2D-array views for 1D arrays.
        let view_type = match view_type {
            vk::ImageViewType::TYPE_1D => vk::ImageViewType::TYPE_2D,
            vk::ImageViewType::TYPE_1D_ARRAY => vk::ImageViewType::TYPE_2D_ARRAY,
            v => v,
        };
        let store_fmt = canonical(vfmt);
        let depth_layers = if ty3d { t.depth } else { total_layers };
        // Render targets here whose guest memory was since written by buffer
        // stores: reload them before sampling (directly or via shadows).
        let newer: Vec<ImageKey> = self.by_base.get(&t.base).map(|ks| ks.iter().copied().filter(|k| self.images[k].guest_newer).collect()).unwrap_or_default();
        for k in newer {
            self.refresh_from_guest(k, false)?;
        }
        let key = ImageKey { base: t.base, width: t.width, height: t.height, layers: depth_layers, levels, format: store_fmt, depth: false, ty3d };
        // GPU-written image with the same size at this address?
        if let Some(k) = self.find_compatible(key) {
            let img = self.images.get_mut(&k).unwrap();
            if img.gpu_written || storage {
                img.gpu_written |= storage;
                let v = img.view(if storage { canonical(vfmt) } else { vfmt }, t.dst_sel, view_type, storage, view_base, layers)?;
                return Ok(v);
            }
        }
        // A depth render target at this address (sampled after a decompress):
        // copy its depth values into a color image of the sampled format.
        if matches!(vfmt, vk::Format::R32_SFLOAT | vk::Format::R16_UNORM) && !storage {
            if let Some(src) = self.by_base.get(&t.base).and_then(|ks| ks.iter().copied().find(|k| k.depth && self.images[k].gpu_written && (k.format == vk::Format::D16_UNORM) == (vfmt == vk::Format::R16_UNORM))) {
                return self.depth_shadow_view(src, t, vfmt, view_type);
            }
        }
        // A render target at this address with padded dimensions: copy into a
        // shadow image (from the most recently rendered one if several exist).
        if let Some(src) = self.by_base.get(&t.base).and_then(|ks| {
            ks.iter()
                .copied()
                .filter(|k| self.images[k].gpu_written && k.width >= t.width && k.height >= t.height && format_class(k.format) == format_class(store_fmt) && !k.depth)
                .max_by_key(|k| self.images[k].last_write)
        }) {
            if trace_frame() {
                crate::info!("FRAME texture {:#x} {}x{} {:?}: shadow of {}x{} {:?} (last write {})", t.base, t.width, t.height, vfmt, src.width, src.height, src.format, self.images[&src].last_write);
            }
            return self.shadow_view(src, t, vfmt, store_fmt, view_type);
        }
        if trace_frame() {
            let others: Vec<String> = self.by_base.get(&t.base).map(|ks| ks.iter().map(|k| format!("{}x{} {:?} gw {}", k.width, k.height, k.format, self.images[k].gpu_written)).collect()).unwrap_or_default();
            crate::info!("FRAME texture {:#x} {}x{} {:?}: guest memory (images here: [{}])", t.base, t.width, t.height, vfmt, others.join(", "));
        }
        self.init_image(key)?;
        let batch = self.epoch;
        let needs_upload = {
            let img = &self.images[&key];
            !img.gpu_written && img.upload_batch != batch
        };
        if needs_upload {
            let bytes = resources::texture_bytes(t, depth_layers.max(1));
            self.sync_guest(t.base, bytes);
            let t0 = std::time::Instant::now();
            let h = resources::hash_guest(t.base, bytes);
            self.hashed_bytes += bytes;
            self.prof[3] += t0.elapsed();
            let changed = self.images[&key].uploaded_hash != h;
            self.images.get_mut(&key).unwrap().upload_batch = batch;
            if changed {
                self.upload_texture(key, t, depth_layers.max(1))?;
                self.images.get_mut(&key).unwrap().uploaded_hash = h;
            }
        }
        let img = self.images.get_mut(&key).unwrap();
        if storage {
            img.gpu_written = true;
        }
        img.view(if storage { canonical(vfmt) } else { vfmt }, t.dst_sel, view_type, storage, view_base, layers)
    }

    /// Texture view of a depth render target's depth values: copied (depth
    /// aspect -> ring -> color image, cropped to the T# size) whenever the
    /// depth target was written since the last copy.
    fn depth_shadow_view(&mut self, src: ImageKey, t: &Tsharp, vfmt: vk::Format, view_type: vk::ImageViewType) -> Result<vk::ImageView, String> {
        let wgen = self.write_gen.get(&src).copied().unwrap_or(0);
        let skey = (src, t.width, t.height, vfmt);
        let dst_key = ImageKey { base: t.base ^ 0x4000_0000_0000_0000, width: t.width, height: t.height, layers: 1, levels: 1, format: vfmt, depth: false, ty3d: false };
        let fresh = self.shadows.get(&skey).is_some_and(|&(_, g)| g == wgen);
        if !fresh {
            self.init_image(dst_key)?;
            self.begin();
            self.end_rendering();
            let (w, h) = (t.width.min(src.width), t.height.min(src.height));
            let bpp = if vfmt == vk::Format::R16_UNORM { 2 } else { 4 };
            let off = self.ring.alloc(w as u64 * h as u64 * bpp).ok_or("ring full")?;
            global_barrier(self.cmd);
            let c = ctx();
            let to_buf = vk::BufferImageCopy {
                buffer_offset: off,
                buffer_row_length: 0,
                buffer_image_height: 0,
                image_subresource: vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::DEPTH, mip_level: 0, base_array_layer: 0, layer_count: 1 },
                image_offset: vk::Offset3D::default(),
                image_extent: vk::Extent3D { width: w, height: h, depth: 1 },
            };
            let to_img = vk::BufferImageCopy { image_subresource: vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 }, ..to_buf };
            unsafe {
                c.device.cmd_copy_image_to_buffer(self.cmd, self.images[&src].image, vk::ImageLayout::GENERAL, self.ring.buffer, &[to_buf]);
                global_barrier(self.cmd);
                c.device.cmd_copy_buffer_to_image(self.cmd, self.ring.buffer, self.images[&dst_key].image, vk::ImageLayout::GENERAL, &[to_img]);
            }
            self.barrier_needed = true;
            self.shadows.insert(skey, (dst_key, wgen));
            self.images.get_mut(&dst_key).unwrap().gpu_written = true;
        }
        let img = self.images.get_mut(&dst_key).unwrap();
        img.view(vfmt, t.dst_sel, view_type, false, 0, 1)
    }

    fn shadow_view(&mut self, src: ImageKey, t: &Tsharp, vfmt: vk::Format, store_fmt: vk::Format, view_type: vk::ImageViewType) -> Result<vk::ImageView, String> {
        // Array/cube textures may span several render targets laid out back
        // to back in guest memory (e.g. six cube shadow maps sampled as one
        // cube array): collect the consecutive ones that cover the layers.
        let want = if matches!(t.ty, desc::T_CUBE | desc::T_1D_ARRAY | desc::T_2D_ARRAY) { t.total_layers().max(src.layers) } else { src.layers };
        let mut parts = vec![(src, 0u32)];
        let mut covered = src.layers;
        let mut next = self.images.get(&src).and_then(|i| Self::image_guest_layout(&src, i)).map(|g| src.base + g.3);
        while covered < want {
            let Some(addr) = next else { break };
            let Some(k) = self.by_base.get(&addr).and_then(|ks| {
                ks.iter().copied().find(|k| self.images[k].gpu_written && !k.depth && k.width == src.width && k.height == src.height && format_class(k.format) == format_class(src.format))
            }) else {
                break;
            };
            parts.push((k, covered));
            covered += k.layers;
            next = Self::image_guest_layout(&k, &self.images[&k]).map(|g| k.base + g.3);
        }
        let layers = covered.min(want).max(1);
        let wgen: u64 = parts.iter().map(|(k, _)| self.write_gen.get(k).copied().unwrap_or(0)).sum();
        let skey = (src, t.width, t.height, store_fmt);
        let dst_key = ImageKey { base: t.base ^ 0x8000_0000_0000_0000, width: t.width, height: t.height, layers, levels: 1, format: store_fmt, depth: false, ty3d: false };
        let fresh = self.shadows.get(&skey).is_some_and(|&(k, g)| g == wgen && k == dst_key);
        if !fresh {
            self.init_image(dst_key)?;
            self.begin();
            self.end_rendering();
            if self.barrier_needed {
                global_barrier(self.cmd);
                self.barrier_needed = false;
            }
            for &(k, at) in &parts {
                let n = k.layers.min(layers.saturating_sub(at));
                if n == 0 {
                    continue;
                }
                let region = vk::ImageCopy {
                    src_subresource: vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: n },
                    src_offset: vk::Offset3D::default(),
                    dst_subresource: vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: at, layer_count: n },
                    dst_offset: vk::Offset3D::default(),
                    extent: vk::Extent3D { width: t.width, height: t.height, depth: 1 },
                };
                unsafe {
                    ctx().device.cmd_copy_image(self.cmd, self.images[&k].image, vk::ImageLayout::GENERAL, self.images[&dst_key].image, vk::ImageLayout::GENERAL, &[region]);
                }
            }
            self.barrier_needed = true;
            self.shadows.insert(skey, (dst_key, wgen));
            self.images.get_mut(&dst_key).unwrap().gpu_written = true;
        }
        let img = self.images.get_mut(&dst_key).unwrap();
        img.view(vfmt, t.dst_sel, view_type, false, t.view_base_layer(), t.layers().max(1))
    }

    fn upload_texture(&mut self, key: ImageKey, t: &Tsharp, layers: u32) -> Result<(), String> {
        self.uploads += 1;
        super::set_phase(super::phase::UPLOAD, t.base);
        let t0 = std::time::Instant::now();
        let data = resources::read_texture(t, layers);
        self.prof[4] += t0.elapsed();
        let Some(off) = self.ring.alloc(data.len() as u64) else {
            return Err("upload ring full".into());
        };
        self.ring.write(off, &data);
        self.begin();
        self.end_rendering();
        if self.barrier_needed {
            global_barrier(self.cmd);
            self.barrier_needed = false;
        }
        let (_, block) = resources::element_bits(t.dfmt);
        let region = vk::BufferImageCopy {
            buffer_offset: off,
            buffer_row_length: t.width.div_ceil(block) * block,
            buffer_image_height: t.height.div_ceil(block) * block,
            image_subresource: vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: if key.ty3d { 1 } else { layers } },
            image_offset: vk::Offset3D::default(),
            image_extent: vk::Extent3D { width: t.width, height: t.height, depth: if key.ty3d { layers } else { 1 } },
        };
        unsafe {
            ctx().device.cmd_copy_buffer_to_image(self.cmd, self.ring.buffer, self.images[&key].image, vk::ImageLayout::GENERAL, &[region]);
        }
        self.barrier_needed = true;
        if crate::log::enabled(crate::log::Level::Debug) {
            crate::debug!("uploaded texture {:#x} {}x{}x{} dfmt {} nfmt {} tiling {}", t.base, t.width, t.height, layers, t.dfmt, t.nfmt, t.tiling_index);
        }
        super::texdump::maybe_dump(t, &data);
        Ok(())
    }

    /// Whether GPU writes recorded in the current (unsubmitted) batch overlap
    /// guest memory [base, base+size).
    fn pending_write_overlaps(&self, base: u64, size: u64) -> bool {
        self.writebacks.iter().any(|p| p.base < base + size && base < p.base + p.size) || self.zc_overlaps(base, size)
    }

    /// Whether zero-copy writes recorded in this batch overlap [base, base+size).
    fn zc_overlaps(&self, base: u64, size: u64) -> bool {
        self.zc_writes.iter().any(|&(w, n)| w < base + size && base < w + n)
    }

    /// Whether a buffer binding uses zero-copy: large ranges (sparse scatter
    /// destinations, big tables) are bound in place instead of being copied.
    fn zc_candidate(&self, base: u64, size: u64) -> bool {
        size >= ZC_MIN_BYTES && base % 16 == 0 && ctx().ext_host.is_some()
    }

    /// Imported buffer covering guest [base, base+size): (buffer, offset, range).
    fn zc_binding(&mut self, base: u64, size: u64) -> Option<(vk::Buffer, u64, u64)> {
        let max_range = (ctx().limits.max_storage_buffer_range as u64) & !3;
        let span = match self.zc_spans.iter().find(|s| s.0 <= base && base < s.1) {
            Some(s) => (s.0, s.1, s.2),
            None => {
                let (start, end) = crate::mem::MEM.lock().committed_span(base)?;
                let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER;
                match ctx().import_host(start, end - start, usage) {
                    Ok((buf, mem)) => {
                        crate::info!("zero-copy: imported guest memory {start:#x}-{end:#x} ({} MiB)", (end - start) >> 20);
                        self.zc_spans.push((start, end, buf, mem));
                        (start, end, buf)
                    }
                    Err(e) => {
                        self.report(format!("zero-copy import of {start:#x}-{end:#x} failed: {e}"));
                        return None;
                    }
                }
            }
        };
        let range = size.min(span.1 - base).min(max_range) & !3;
        (range >= 4).then_some((span.2, base - span.0, range))
    }

    /// Drops imported spans when guest mappings changed (GPU idle first).
    fn zc_revalidate(&mut self) {
        let g = crate::mem::MAP_GEN.load(std::sync::atomic::Ordering::SeqCst);
        if g == self.zc_gen {
            return;
        }
        self.zc_gen = g;
        if self.zc_spans.is_empty() {
            return;
        }
        self.flush();
        let c = ctx();
        for (start, end, buf, mem) in self.zc_spans.drain(..) {
            unsafe {
                c.device.destroy_buffer(buf, None);
                c.device.free_memory(mem, None);
            }
            crate::debug!("zero-copy: released {start:#x}-{end:#x} (guest mappings changed)");
        }
    }

    /// Makes guest memory [base, base+size) reflect all GPU writes recorded so
    /// far (submits and waits as needed). Returns whether it had to wait.
    pub fn sync_guest(&mut self, base: u64, size: u64) -> bool {
        if self.pending_write_overlaps(base, size) {
            self.flush_for(1);
            return true;
        }
        if writeback_in_flight(base, size) {
            self.flushes[1] += 1;
            let t0 = std::time::Instant::now();
            wait_overlapping(base, size);
            self.prof[0] += t0.elapsed();
            return true;
        }
        false
    }

    /// Copies guest memory [base, base+size) into a new ring range for GPU
    /// reads and returns its offset. GPU writes that have not reached guest
    /// memory yet are applied: writes of submitted batches are waited for,
    /// writes recorded in this batch are forwarded with GPU copies (in
    /// program order) on top of the guest data.
    fn upload_guest(&mut self, base: u64, size: u64) -> Result<u64, String> {
        if writeback_in_flight(base, size) {
            self.flushes[1] += 1;
            let t0 = std::time::Instant::now();
            wait_overlapping(base, size);
            self.prof[0] += t0.elapsed();
        }
        let off = self.ring.alloc(size).ok_or("ring full")?;
        let t0 = std::time::Instant::now();
        self.ring.write_guest(off, base, size);
        self.prof[2] += t0.elapsed();
        let regions: Vec<vk::BufferCopy> = self
            .writebacks
            .iter()
            .filter_map(|p| {
                let (lo, hi) = (p.base.max(base), (p.base + p.size).min(base + size));
                (lo < hi).then(|| vk::BufferCopy { src_offset: p.dev_off + (lo - p.base), dst_offset: off + (lo - base), size: hi - lo })
            })
            .collect();
        if !regions.is_empty() {
            self.flushes[5] += 1;
            self.end_rendering();
            let c = ctx();
            for r in &regions {
                // Shader writes (and earlier forwards) before each copy.
                global_barrier(self.cmd);
                unsafe { c.device.cmd_copy_buffer(self.cmd, self.dring.buffer, self.ring.buffer, std::slice::from_ref(r)) };
            }
            self.barrier_needed = true;
        }
        Ok(off)
    }

    /// Submits the batch first if the ring segments may not fit `host` and
    /// `dev` more bytes (a draw/dispatch cannot be split across batches).
    fn reserve(&mut self, host: u64, dev: u64) {
        let need_host = host + self.ring.segment_size() / 8;
        let need_dev = dev + self.dring.segment_size() / 8;
        if self.ring.remaining() < need_host || self.dring.remaining() < need_dev {
            self.flushes[2] += 1;
            self.submit_batch(Vec::new());
        }
    }

    /// Upper bound of ring bytes texture uploads of a draw/dispatch may need
    /// (images not produced by the GPU; reloads of stale render targets).
    fn texture_upload_estimate(&self, stages: &[&Prepared]) -> u64 {
        let mut n = 0u64;
        for p in stages {
            for b in &p.translated.out.bindings {
                if let BindingKind::Image(i) | BindingKind::StorageImage(i) = b.kind {
                    if let ResolvedRes::Image(t) = &p.resources[i] {
                        // Images that are GPU-produced or already validated this epoch need no upload.
                        let ready = self.by_base.get(&t.base).is_some_and(|ks| {
                            ks.iter().any(|k| {
                                let img = &self.images[k];
                                (img.gpu_written && !img.guest_newer) || (!img.gpu_written && img.upload_batch == self.epoch)
                            })
                        });
                        if !ready {
                            let layers = if t.ty == desc::T_3D { t.depth } else { t.total_layers() }.max(1);
                            n += resources::texture_bytes(t, layers).min(64 << 20);
                        }
                    }
                }
            }
        }
        n
    }

    /// Ring bytes a draw/dispatch's buffer bindings need (host, device).
    fn binding_bytes(stages: &[&Prepared]) -> (u64, u64) {
        let (mut host, mut dev) = (0u64, 0u64);
        for p in stages {
            for b in &p.translated.out.bindings {
                if let BindingKind::Buffer(i) = b.kind {
                    if let ResolvedRes::Buffer(v) = &p.resources[i] {
                        if v.size() >= ZC_MIN_BYTES {
                            continue; // zero-copy candidate (or clamped below if import fails)
                        }
                        let size = v.size().clamp(4, 64 << 20).next_multiple_of(256);
                        host += size;
                        if buffer_written(p, i) {
                            dev += size;
                        }
                    }
                }
            }
        }
        (host, dev)
    }

    /// Guest surface of a render target: (bits per element, tile mode, padded
    /// height, total bytes), if its layout is known.
    fn image_guest_layout(key: &ImageKey, img: &Image) -> Option<(u32, tiling::TileMode, u32, u64)> {
        let idx = img.tile_index?;
        let bpp = match key.format {
            vk::Format::D32_SFLOAT | vk::Format::D32_SFLOAT_S8_UINT => 32,
            vk::Format::D16_UNORM => 16,
            f => match format_class(f) {
                c @ (1 | 2 | 4 | 8 | 16) => c * 8,
                _ => return None,
            },
        };
        let mode = tiling::tile_mode(idx);
        let (pitch, ph) = tiling::padded(key.width, key.height, bpp, mode);
        Some((bpp, mode, ph, tiling::slice_bytes(pitch, ph, bpp) * key.layers as u64))
    }

    /// Writes render targets whose GPU contents are newer than guest memory
    /// and overlap [base, base+size) back to guest memory in their tiled
    /// layout, so guest-memory readers (buffer loads, CPU-sourced textures)
    /// see them. Synchronous; must not run while a draw is being prepared.
    fn writeback_images(&mut self, base: u64, size: u64) {
        // Only targets rendered in the last few frames: older ones are dead
        // (their memory may have been reused by the title; writing their
        // contents back would corrupt it).
        let frame = super::FRAMES.load(std::sync::atomic::Ordering::Relaxed);
        for (_, i) in self.images.iter_mut() {
            if i.guest_stale && i.last_write_frame + 2 < frame {
                i.guest_stale = false;
            }
        }
        let keys: Vec<ImageKey> = self
            .images
            .iter()
            .filter(|(k, i)| i.guest_stale && !i.guest_newer && !k.ty3d && k.base < base + size && Self::image_guest_layout(k, i).is_some_and(|g| base < k.base + g.3))
            .map(|(k, _)| *k)
            .collect();
        for key in keys {
            let Some((bpp, mode, ph, total)) = Self::image_guest_layout(&key, &self.images[&key]) else { continue };
            let lin_slice = key.width as u64 * key.height as u64 * (bpp / 8) as u64;
            if self.ring.remaining() < lin_slice * key.layers as u64 + 4096 {
                self.submit_batch(Vec::new());
            }
            let Some(off) = self.ring.alloc(lin_slice * key.layers as u64) else {
                self.report(format!("render target write-back {:#x} too large for the ring", key.base));
                continue;
            };
            self.begin();
            self.end_rendering();
            global_barrier(self.cmd);
            let region = vk::BufferImageCopy {
                buffer_offset: off,
                buffer_row_length: 0,
                buffer_image_height: 0,
                image_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: if key.depth { vk::ImageAspectFlags::DEPTH } else { vk::ImageAspectFlags::COLOR },
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: key.layers,
                },
                image_offset: vk::Offset3D::default(),
                image_extent: vk::Extent3D { width: key.width, height: key.height, depth: 1 },
            };
            unsafe { ctx().device.cmd_copy_image_to_buffer(self.cmd, self.images[&key].image, vk::ImageLayout::GENERAL, self.ring.buffer, &[region]) };
            self.flush_for(1);
            let t0 = std::time::Instant::now();
            let lin = unsafe { std::slice::from_raw_parts(self.ring.ptr.add(off as usize), (lin_slice * key.layers as u64) as usize) };
            let slice = total / key.layers as u64;
            let mut tiled = vec![0u8; total as usize];
            resources::copy_from_guest(&mut tiled, key.base);
            for l in 0..key.layers as u64 {
                let src = &lin[(l * lin_slice) as usize..((l + 1) * lin_slice) as usize];
                let dst = &mut tiled[(l * slice) as usize..((l + 1) * slice) as usize];
                tiling::tile_slice(src, dst, key.width, key.height, key.width, ph, bpp, mode);
            }
            resources::copy_to_guest(&tiled, key.base);
            self.prof[1] += t0.elapsed();
            self.images.get_mut(&key).unwrap().guest_stale = false;
            self.epoch += 1;
            if crate::log::enabled(crate::log::Level::Debug) {
                crate::debug!("render target {:#x} {}x{} {:?} written back to guest memory", key.base, key.width, key.height, key.format);
            }
        }
    }

    /// Before a draw/dispatch allocates descriptors: makes guest memory read
    /// by its buffer bindings current with render targets, and lands pending
    /// buffer writes that its CPU-sourced textures may read (both may submit
    /// and wait, which is not allowed once preparation has started).
    fn prepare_guest_reads(&mut self, stages: &[&Prepared]) {
        self.zc_revalidate();
        for p in stages {
            for b in &p.translated.out.bindings {
                match b.kind {
                    BindingKind::Buffer(i) => {
                        let ResolvedRes::Buffer(v) = &p.resources[i] else { continue };
                        let full = v.size().max(4);
                        if self.zc_candidate(v.base, full) {
                            // Accessed in place at execution time: device-ring
                            // writes must have reached guest memory.
                            let range = full.min(ctx().limits.max_storage_buffer_range as u64);
                            if self.writebacks.iter().any(|w| w.base < v.base + range && v.base < w.base + w.size) {
                                self.flush_for(1);
                            } else if writeback_in_flight(v.base, range) {
                                wait_overlapping(v.base, range);
                            }
                            if p.read_buffers.contains(&i) && range <= SURFACE_WRITE_MAX {
                                self.writeback_images(v.base, range);
                            }
                        } else {
                            let size = full.min(64 << 20);
                            // Snapshot at record time: in-place writes of this
                            // batch must land first.
                            if self.zc_overlaps(v.base, size) {
                                self.flush_for(1);
                            }
                            if p.read_buffers.contains(&i) && v.size() <= SURFACE_WRITE_MAX {
                                self.writeback_images(v.base, size);
                            }
                        }
                    }
                    BindingKind::Image(i) | BindingKind::StorageImage(i) => {
                        if let ResolvedRes::Image(t) = &p.resources[i] {
                            let layers = if t.ty == desc::T_3D { t.depth } else { t.total_layers() }.max(1);
                            let bytes = resources::texture_bytes(t, layers);
                            if self.pending_write_overlaps(t.base, bytes) {
                                self.flush_for(1);
                            }
                            // Render targets here made stale by buffer stores.
                            let newer: Vec<ImageKey> = self.by_base.get(&t.base).map(|ks| ks.iter().copied().filter(|k| self.images[k].guest_newer).collect()).unwrap_or_default();
                            for k in newer {
                                if let Err(e) = self.refresh_from_guest(k, true) {
                                    self.report(format!("render target reload failed: {e}"));
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    // -------------------------------------------------------------------
    // Descriptors

    fn write_descriptors(&mut self, set: vk::DescriptorSet, prepared: &[&Prepared]) -> Result<(), String> {
        let c = ctx();
        enum Info {
            Buf(vk::DescriptorBufferInfo),
            Img(vk::DescriptorImageInfo),
        }
        let mut infos: Vec<(u32, vk::DescriptorType, Info)> = Vec::new();
        for p in prepared {
            for b in &p.translated.out.bindings {
                match b.kind {
                    BindingKind::SgprData => {
                        let bytes: Vec<u8> = p.sgpr_data.iter().flat_map(|w| w.to_le_bytes()).collect();
                        let off = self.ring.alloc(bytes.len() as u64).ok_or("ring full")?;
                        self.ring.write(off, &bytes);
                        infos.push((b.binding, vk::DescriptorType::STORAGE_BUFFER, Info::Buf(vk::DescriptorBufferInfo { buffer: self.ring.buffer, offset: off, range: bytes.len() as u64 })));
                    }
                    BindingKind::Buffer(i) => {
                        let ResolvedRes::Buffer(v) = &p.resources[i] else { return Err("binding/resource mismatch".into()) };
                        let base = v.base;
                        if self.zc_candidate(base, v.size().max(4)) {
                            if let Some((buf, off, range)) = self.zc_binding(base, v.size().max(4)) {
                                if buffer_written(p, i) {
                                    self.zc_writes.push((base, range));
                                    // Huge bindings are sparse writers (the
                                    // scatter kernel binds 2.2 GB to write a
                                    // few records): not a surface write.
                                    if range <= SURFACE_WRITE_MAX {
                                        self.note_meta_write(base, range);
                                        self.note_buffer_write(base, range);
                                    }
                                }
                                infos.push((b.binding, vk::DescriptorType::STORAGE_BUFFER, Info::Buf(vk::DescriptorBufferInfo { buffer: buf, offset: off, range })));
                                continue;
                            }
                        }
                        let size = v.size().clamp(4, 64 << 20);
                        if buffer_written(p, i) && v.size() <= SURFACE_WRITE_MAX {
                            self.note_meta_write(base, size);
                            self.note_buffer_write(base, size);
                        }
                        let off = self.upload_guest(base, size)?;
                        if buffer_written(p, i) {
                            // The shader works on a device-local copy; written
                            // back to guest memory when the batch completes.
                            let dev_off = self.dring.alloc(size).ok_or("device ring full")?;
                            self.end_rendering();
                            global_barrier(self.cmd);
                            let region = vk::BufferCopy { src_offset: off, dst_offset: dev_off, size };
                            unsafe { ctx().device.cmd_copy_buffer(self.cmd, self.ring.buffer, self.dring.buffer, &[region]) };
                            self.barrier_needed = true;
                            self.writebacks.push(PendingWrite { base, size, host_off: off, dev_off });
                            infos.push((b.binding, vk::DescriptorType::STORAGE_BUFFER, Info::Buf(vk::DescriptorBufferInfo { buffer: self.dring.buffer, offset: dev_off, range: size.next_multiple_of(4) })));
                            continue;
                        }
                        infos.push((b.binding, vk::DescriptorType::STORAGE_BUFFER, Info::Buf(vk::DescriptorBufferInfo { buffer: self.ring.buffer, offset: off, range: size.next_multiple_of(4) })));
                    }
                    BindingKind::Image(i) | BindingKind::StorageImage(i) => {
                        let ResolvedRes::Image(t) = &p.resources[i] else { return Err("binding/resource mismatch".into()) };
                        let storage = matches!(b.kind, BindingKind::StorageImage(_));
                        let view = self.texture(t, storage)?;
                        infos.push((
                            b.binding,
                            if storage { vk::DescriptorType::STORAGE_IMAGE } else { vk::DescriptorType::SAMPLED_IMAGE },
                            Info::Img(vk::DescriptorImageInfo { sampler: vk::Sampler::null(), image_view: view, image_layout: vk::ImageLayout::GENERAL }),
                        ));
                    }
                    BindingKind::Gds => {
                        infos.push((b.binding, vk::DescriptorType::STORAGE_BUFFER, Info::Buf(vk::DescriptorBufferInfo { buffer: self.gds, offset: 0, range: gcn::translate::GDS_BYTES as u64 })));
                    }
                    BindingKind::Sampler(i) => {
                        let ResolvedRes::Sampler(s) = &p.resources[i] else { return Err("binding/resource mismatch".into()) };
                        let smp = self.samplers.get(s)?;
                        infos.push((b.binding, vk::DescriptorType::SAMPLER, Info::Img(vk::DescriptorImageInfo { sampler: smp, image_view: vk::ImageView::null(), image_layout: vk::ImageLayout::UNDEFINED })));
                    }
                }
            }
        }
        let writes: Vec<vk::WriteDescriptorSet> = infos
            .iter()
            .map(|(binding, ty, info)| {
                let w = vk::WriteDescriptorSet::default().dst_set(set).dst_binding(*binding).descriptor_type(*ty);
                match info {
                    Info::Buf(b) => w.buffer_info(std::slice::from_ref(b)),
                    Info::Img(i) => w.image_info(std::slice::from_ref(i)),
                }
            })
            .collect();
        unsafe { c.device.update_descriptor_sets(&writes, &[]) };
        Ok(())
    }

    fn alloc_set(&mut self, layout: vk::DescriptorSetLayout) -> Result<vk::DescriptorSet, String> {
        let layouts = [layout];
        let ai = vk::DescriptorSetAllocateInfo::default().descriptor_pool(self.desc_pool).set_layouts(&layouts);
        match unsafe { ctx().device.allocate_descriptor_sets(&ai) } {
            Ok(v) => Ok(v[0]),
            Err(_) => {
                self.flushes[3] += 1;
                self.submit_batch(Vec::new());
                self.begin();
                unsafe { ctx().device.allocate_descriptor_sets(&ai) }.map(|v| v[0]).map_err(|e| format!("descriptor alloc: {e}"))
            }
        }
    }

    // -------------------------------------------------------------------
    // Draw

    fn draw_inner(&mut self, cp: &CommandProcessor, kind: &'static str, count: u32, index_addr: u64) -> Result<(), String> {
        super::set_phase(super::phase::DRAW, ((cp.regs.sh(0x2C48) as u64) | ((cp.regs.sh(0x2C49) as u64) << 32)) << 8);
        if trace_frame() {
            self.flush(); // frame trace: time each command on its own
        }
        let regs = &cp.regs;
        if self.ring.remaining() < self.ring.segment_size() / 4 {
            self.flushes[2] += 1;
            self.submit_batch(Vec::new());
        }
        self.begin();
        let tp = std::time::Instant::now();
        let vs = shaders::prepare(&StageRegs { stage: Stage::Vertex, regs })?;
        let ps_addr = ((regs.sh(0x2C08) as u64) | ((regs.sh(0x2C09) as u64) << 32)) << 8;
        let ps = if ps_addr != 0 { Some(shaders::prepare(&StageRegs { stage: Stage::Fragment, regs })?) } else { None };
        self.prof[5] += tp.elapsed();
        let prim = regs.uconfig[(0xC242 - 0xC000) as usize];
        let rect = prim == 0x11;
        if !matches!(prim, 1..=6 | 0x11) {
            return Err(format!("primitive type {prim:#x} unsupported"));
        }
        // Attachments.
        let target_mask = regs.ctx(0xA08E);
        let color_mode = (regs.ctx(0xA202) >> 4) & 7;
        let mut colors: Vec<Option<Attachment>> = Vec::new();
        let mut blends = Vec::new();
        let mut masks = Vec::new();
        if color_mode != 0 {
            for n in 0..8u32 {
                let mask = (target_mask >> (4 * n)) & 0xF;
                let written = ps.as_ref().is_some_and(|p| p.translated.out.mrt_mask & (1 << n) != 0);
                let att = if mask != 0 && written { self.color_target(regs, n)? } else { None };
                if att.is_some() {
                    colors.resize(n as usize, None);
                    colors.push(att);
                    while blends.len() < n as usize {
                        blends.push(0);
                        masks.push(0);
                    }
                    blends.push(regs.ctx(0xA1E0 + n));
                    masks.push(mask);
                }
            }
        }
        let dctl = regs.ctx(0xA200);
        let depth_used = dctl & 0x7 != 0 || regs.ctx(0xA000) & 3 != 0;
        let depth = if depth_used { self.depth_target(regs)? } else { None };
        self.apply_meta_clears(regs, &colors, depth);
        if trace_frame() {
            let res: Vec<String> = std::iter::once(&vs).chain(ps.iter()).flat_map(|p| stage_res_summary(p)).collect();
            crate::info!(
                "FRAME draw {} n={} prim {:#x} vs {:#x} ps {:#x} rt {:?} depth {:?} ds [dctl {:#x} sctl {:#x} ref {:#x}/{:#x} rc {:#x}] blend [{:#x} mask {:#x} cc {:#x}] vp [vte {:#x} x {}/{} y {}/{} z {}/{} clip {:#x} sc {:#x}/{:#x}] res [{}]",
                kind,
                count,
                prim,
                vs.addr,
                ps.as_ref().map(|p| p.addr).unwrap_or(0),
                colors.iter().map(|c| c.map(|a| format!("{:#x}:{}x{}:{:?}", a.key.base, a.key.width, a.key.height, a.view_format))).collect::<Vec<_>>(),
                depth.map(|d| format!("{:#x}", d.key.base)),
                regs.ctx(0xA200),
                regs.ctx(0xA10B),
                regs.ctx(0xA10C),
                regs.ctx(0xA10D),
                regs.ctx(0xA000),
                regs.ctx(0xA1E0),
                regs.ctx(0xA08E),
                regs.ctx(0xA202),
                regs.ctx(0xA206),
                f32r(regs.ctx(0xA10F)),
                f32r(regs.ctx(0xA110)),
                f32r(regs.ctx(0xA111)),
                f32r(regs.ctx(0xA112)),
                f32r(regs.ctx(0xA113)),
                f32r(regs.ctx(0xA114)),
                regs.ctx(0xA204),
                regs.ctx(0xA094),
                regs.ctx(0xA095),
                res.join(", ")
            );
        }
        if colors.iter().all(|c| c.is_none()) && depth.is_none() {
            if trace_frame() {
                let cb = |n: u32, o: u32| regs.ctx(0xA318 + n * 0xF + o);
                crate::info!(
                    "FRAME skipped draw: CB mode {} target mask {:#x} DB_RENDER_CONTROL {:#x} DB_RENDER_OVERRIDE {:#x} | CB0 base {:#x} info {:#x} attrib {:#x} cmask {:#x} fmask {:#x} clear {:#x}/{:#x} | CB1 base {:#x} info {:#x} | DB z {:#x} zinfo {:#x} htile {:#x}",
                    color_mode,
                    target_mask,
                    regs.ctx(0xA000),
                    regs.ctx(0xA003),
                    (cb(0, 0) as u64) << 8,
                    cb(0, 4),
                    cb(0, 5),
                    (cb(0, 7) as u64) << 8,
                    (cb(0, 9) as u64) << 8,
                    cb(0, 11),
                    cb(0, 12),
                    (cb(1, 0) as u64) << 8,
                    cb(1, 4),
                    (regs.ctx(0xA012) as u64) << 8,
                    regs.ctx(0xA010),
                    (regs.ctx(0xA005) as u64) << 8
                );
            }
            self.skipped += 1;
            return Ok(());
        }
        // Extent: smallest attachment.
        let mut extent = (u32::MAX, u32::MAX);
        for a in colors.iter().flatten().chain(depth.iter()) {
            extent = (extent.0.min(a.key.width), extent.1.min(a.key.height));
        }
        // Pipeline.
        let vs_params = vs.translated.out.params;
        let key = GfxKey {
            vs: vs.translated.key_hash ^ vs.addr,
            ps: ps.as_ref().map(|p| p.translated.key_hash ^ p.addr),
            rect,
            gs_params: if rect { vs_params } else { 0 },
            colors: colors.iter().map(|c| c.map(|a| a.view_format).unwrap_or(vk::Format::UNDEFINED)).collect(),
            blend: blends.clone(),
            masks: masks.clone(),
            depth: depth.map(|d| d.key.format).unwrap_or(vk::Format::UNDEFINED),
            stencil: depth.is_some_and(|d| d.key.format == vk::Format::D32_SFLOAT_S8_UINT),
            topology_class: match prim {
                1 => 0,
                2 | 3 => 1,
                _ => 2,
            },
            polygon: ((regs.ctx(0xA205) >> 3) & 3) as u8,
            depth_clamp: regs.ctx(0xA204) & (3 << 26) != 0,
        };
        if !self.gfx_pipes.contains_key(&key) {
            let t0 = std::time::Instant::now();
            let p = self.create_gfx_pipeline(&key, &vs, ps.as_ref())?;
            let ms = t0.elapsed().as_millis();
            if ms >= 100 {
                crate::info!("graphics pipeline for vs {:#x} ps {:#x} took {ms} ms to create", vs.addr, ps.as_ref().map(|p| p.addr).unwrap_or(0));
            }
            self.gfx_pipes.insert(key.clone(), p);
        }
        let (pipeline, layout, set_layout) = {
            let p = &self.gfx_pipes[&key];
            (p.pipeline, p.layout, p.set_layout)
        };
        // Descriptors (may record uploads: must happen before rendering starts).
        let mut stages = vec![&vs];
        if let Some(p) = &ps {
            stages.push(p);
        }
        let (host, dev) = Self::binding_bytes(&stages);
        let tex = self.texture_upload_estimate(&stages);
        self.reserve(host + tex + count as u64 * 4, dev);
        self.prepare_guest_reads(&stages);
        self.begin();
        let set = self.alloc_set(set_layout)?;
        // Uploads end any active rendering; record them first.
        let wb_before = self.writebacks.len() + self.zc_writes.len();
        self.write_descriptors(set, &stages)?;
        // Index data.
        let mut indexed = None;
        if kind == "index2" || kind == "index_offset2" {
            let isz = if cp.index_type == 1 { 4 } else { 2 };
            let n = count as u64 * isz;
            let off = self.upload_guest(index_addr, n)?;
            indexed = Some((off, if isz == 4 { vk::IndexType::UINT32 } else { vk::IndexType::UINT16 }));
        }
        // Begin rendering if attachments changed.
        let same = self.rendering.as_ref().is_some_and(|r| r.colors == colors && r.depth == depth);
        if !same {
            self.end_rendering();
            if self.barrier_needed {
                global_barrier(self.cmd);
                self.barrier_needed = false;
            }
            self.begin_rendering(&colors, depth, extent)?;
        }
        let tick = self.draws + self.dispatches + 1;
        for a in colors.iter().flatten() {
            if let Some(img) = self.images.get_mut(&a.key) {
                img.gpu_written = true;
                img.guest_stale = true;
                img.last_write = tick;
                img.last_write_frame = super::FRAMES.load(std::sync::atomic::Ordering::Relaxed);
            }
            *self.write_gen.entry(a.key).or_default() += 1;
        }
        if let Some(d) = depth {
            if let Some(img) = self.images.get_mut(&d.key) {
                img.gpu_written = true;
                img.guest_stale = true;
                img.last_write = tick;
                img.last_write_frame = super::FRAMES.load(std::sync::atomic::Ordering::Relaxed);
            }
            *self.write_gen.entry(d.key).or_default() += 1;
        }
        let c = ctx();
        let cmd = self.cmd;
        unsafe {
            c.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);
            c.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS, layout, 0, &[set], &[]);
            self.set_dynamic_state(regs, extent, prim, depth.is_some());
            // Depth/stencil clears requested through DB_RENDER_CONTROL.
            let rc = regs.ctx(0xA000);
            if rc & 3 != 0 && depth.is_some() {
                let mut aspect = vk::ImageAspectFlags::empty();
                if rc & 1 != 0 {
                    aspect |= vk::ImageAspectFlags::DEPTH;
                }
                if rc & 2 != 0 && key.stencil {
                    aspect |= vk::ImageAspectFlags::STENCIL;
                }
                if !aspect.is_empty() {
                    let ca = vk::ClearAttachment {
                        aspect_mask: aspect,
                        color_attachment: 0,
                        clear_value: vk::ClearValue { depth_stencil: vk::ClearDepthStencilValue { depth: f32r(regs.ctx(0xA00B)), stencil: regs.ctx(0xA00A) & 0xFF } },
                    };
                    let cr = vk::ClearRect { rect: vk::Rect2D { offset: vk::Offset2D::default(), extent: vk::Extent2D { width: extent.0, height: extent.1 } }, base_array_layer: 0, layer_count: 1 };
                    c.device.cmd_clear_attachments(cmd, &[ca], &[cr]);
                }
                if colors.iter().all(|c| c.is_none()) {
                    self.draws += 1;
                    return Ok(());
                }
            }
            let instances = cp.num_instances.max(1);
            match indexed {
                Some((off, ty)) => {
                    c.device.cmd_bind_index_buffer(cmd, self.ring.buffer, off, ty);
                    let base_vertex = regs.ctx(0xA102) as i32;
                    c.device.cmd_draw_indexed(cmd, count, instances, 0, base_vertex, 0);
                }
                None => c.device.cmd_draw(cmd, count, instances, 0, 0),
            }
        }
        self.draws += 1;
        if self.writebacks.len() + self.zc_writes.len() > wb_before {
            // Buffer writes: order later consumers; CPU-sourced textures may
            // alias the written memory.
            self.barrier_needed = true;
            self.epoch += 1;
        }
        if trace_frame() {
            self.trace_timing("draw");
            if let Some(p) = &ps {
                self.dump_after(p.addr, &colors, depth);
            }
        }
        Ok(())
    }

    /// Frame trace: submits the command just recorded (with its uploads) on
    /// its own and logs how long the GPU took.
    fn trace_timing(&mut self, what: &str) {
        let t0 = std::time::Instant::now();
        self.flush();
        let ld = |a: &std::sync::atomic::AtomicU64| a.load(std::sync::atomic::Ordering::Relaxed);
        crate::info!(
            "FRAME timing {what}: {:.3} ms (fence {:.3} ms, write-back {} KiB {:.3} ms)",
            t0.elapsed().as_secs_f64() * 1e3,
            ld(&LAST_FENCE_NS) as f64 / 1e6,
            ld(&LAST_WB_BYTES) >> 10,
            ld(&LAST_WB_NS) as f64 / 1e6
        );
    }

    fn begin_rendering(&mut self, colors: &[Option<Attachment>], depth: Option<Attachment>, extent: (u32, u32)) -> Result<(), String> {
        let mut color_infos = Vec::new();
        for a in colors {
            match a {
                Some(a) => {
                    let img = self.images.get_mut(&a.key).ok_or("attachment image missing")?;
                    let view = img.view(a.view_format, 0, vk::ImageViewType::TYPE_2D, false, 0, 1)?;
                    color_infos.push(
                        vk::RenderingAttachmentInfo::default()
                            .image_view(view)
                            .image_layout(vk::ImageLayout::GENERAL)
                            .load_op(vk::AttachmentLoadOp::LOAD)
                            .store_op(vk::AttachmentStoreOp::STORE),
                    );
                }
                None => color_infos.push(vk::RenderingAttachmentInfo::default().image_view(vk::ImageView::null())),
            }
        }
        let depth_info = match depth {
            Some(d) => {
                let img = self.images.get_mut(&d.key).ok_or("depth image missing")?;
                let view = img.view(d.view_format, 0, vk::ImageViewType::TYPE_2D, false, 0, 1)?;
                Some(vk::RenderingAttachmentInfo::default().image_view(view).image_layout(vk::ImageLayout::GENERAL).load_op(vk::AttachmentLoadOp::LOAD).store_op(vk::AttachmentStoreOp::STORE))
            }
            None => None,
        };
        let stencil = depth.is_some_and(|d| d.key.format == vk::Format::D32_SFLOAT_S8_UINT);
        let mut ri = vk::RenderingInfo::default()
            .render_area(vk::Rect2D { offset: vk::Offset2D::default(), extent: vk::Extent2D { width: extent.0, height: extent.1 } })
            .layer_count(1)
            .color_attachments(&color_infos);
        if let Some(d) = &depth_info {
            ri = ri.depth_attachment(d);
            if stencil {
                ri = ri.stencil_attachment(d);
            }
        }
        unsafe { ctx().device.cmd_begin_rendering(self.cmd, &ri) };
        self.rendering = Some(Rendering { colors: colors.to_vec(), depth, extent });
        Ok(())
    }

    unsafe fn set_dynamic_state(&mut self, regs: &Regs, extent: (u32, u32), prim: u32, has_depth: bool) {
        let c = ctx();
        let cmd = self.cmd;
        let vte = regs.ctx(0xA206);
        let (xs, xo, ys, yo, zs, zo) = (f32r(regs.ctx(0xA10F)), f32r(regs.ctx(0xA110)), f32r(regs.ctx(0xA111)), f32r(regs.ctx(0xA112)), f32r(regs.ctx(0xA113)), f32r(regs.ctx(0xA114)));
        let vp = if vte & 0x3F == 0x3F && xs != 0.0 {
            vk::Viewport { x: xo - xs, y: yo - ys, width: 2.0 * xs, height: 2.0 * ys, min_depth: zo.clamp(0.0, 1.0), max_depth: (zo + zs).clamp(0.0, 1.0) }
        } else {
            if vte & 0x3F != 0x3F {
                self.report(format!("PA_CL_VTE_CNTL {vte:#x} (viewport transform disabled) approximated"));
            }
            vk::Viewport { x: 0.0, y: 0.0, width: extent.0 as f32, height: extent.1 as f32, min_depth: 0.0, max_depth: 1.0 }
        };
        unsafe {
            c.device.cmd_set_viewport(cmd, 0, &[vp]);
            let mut s = (0i32, 0i32, extent.0 as i32, extent.1 as i32);
            // Screen, window and generic scissors always apply; the viewport
            // scissor only with PA_SC_MODE_CNTL_0.VPORT_SCISSOR_ENABLE.
            let vport_scissor = regs.ctx(0xA292) & 2 != 0;
            for (tl, br) in [(0xA00C, 0xA00D), (0xA081, 0xA082), (0xA090, 0xA091), (0xA094, 0xA095)] {
                if tl == 0xA094 && !vport_scissor {
                    continue;
                }
                let (x0, y0, x1, y1) = scissor_rect(regs.ctx(tl), regs.ctx(br));
                if x1 > x0 && y1 > y0 {
                    s = (s.0.max(x0), s.1.max(y0), s.2.min(x1), s.3.min(y1));
                }
            }
            let sc = vk::Rect2D { offset: vk::Offset2D { x: s.0.max(0), y: s.1.max(0) }, extent: vk::Extent2D { width: (s.2 - s.0).max(0) as u32, height: (s.3 - s.1).max(0) as u32 } };
            c.device.cmd_set_scissor(cmd, 0, &[sc]);
            let mode = regs.ctx(0xA205);
            let mut cull = vk::CullModeFlags::NONE;
            if mode & 1 != 0 {
                cull |= vk::CullModeFlags::FRONT;
            }
            if mode & 2 != 0 {
                cull |= vk::CullModeFlags::BACK;
            }
            if prim == 0x11 {
                cull = vk::CullModeFlags::NONE;
            }
            c.device.cmd_set_cull_mode(cmd, cull);
            c.device.cmd_set_front_face(cmd, if mode & 4 != 0 { vk::FrontFace::CLOCKWISE } else { vk::FrontFace::COUNTER_CLOCKWISE });
            let topo = match prim {
                1 => vk::PrimitiveTopology::POINT_LIST,
                2 => vk::PrimitiveTopology::LINE_LIST,
                3 => vk::PrimitiveTopology::LINE_STRIP,
                5 => vk::PrimitiveTopology::TRIANGLE_FAN,
                6 => vk::PrimitiveTopology::TRIANGLE_STRIP,
                _ => vk::PrimitiveTopology::TRIANGLE_LIST,
            };
            c.device.cmd_set_primitive_topology(cmd, topo);
            c.device.cmd_set_primitive_restart_enable(cmd, regs.ctx(0xA2A5) & 1 != 0 && matches!(prim, 3 | 5 | 6));
            c.device.cmd_set_rasterizer_discard_enable(cmd, false);
            let dctl = regs.ctx(0xA200);
            // BB_NO_DEPTH_TEST=1 (diagnostic): depth testing off for all draws.
            static NO_DEPTH: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            let no_depth = *NO_DEPTH.get_or_init(|| std::env::var_os("BB_NO_DEPTH_TEST").is_some());
            c.device.cmd_set_depth_test_enable(cmd, has_depth && dctl & 2 != 0 && !no_depth);
            c.device.cmd_set_depth_write_enable(cmd, has_depth && dctl & 4 != 0);
            c.device.cmd_set_depth_compare_op(cmd, desc::vk_compare((dctl >> 4) & 7));
            c.device.cmd_set_depth_bounds_test_enable(cmd, false);
            let stencil_on = has_depth && dctl & 1 != 0;
            c.device.cmd_set_stencil_test_enable(cmd, stencil_on);
            let sctl = regs.ctx(0xA10B);
            let refm = regs.ctx(0xA10C);
            let refm_bf = regs.ctx(0xA10D);
            let back_en = dctl & (1 << 7) != 0;
            let ffunc = desc::vk_compare((dctl >> 8) & 7);
            let bfunc = if back_en { desc::vk_compare((dctl >> 20) & 7) } else { ffunc };
            c.device.cmd_set_stencil_op(cmd, vk::StencilFaceFlags::FRONT, stencil_op(sctl & 0xF), stencil_op((sctl >> 4) & 0xF), stencil_op((sctl >> 8) & 0xF), ffunc);
            let bs = if back_en { sctl >> 12 } else { sctl };
            c.device.cmd_set_stencil_op(cmd, vk::StencilFaceFlags::BACK, stencil_op(bs & 0xF), stencil_op((bs >> 4) & 0xF), stencil_op((bs >> 8) & 0xF), bfunc);
            let rb = if back_en { refm_bf } else { refm };
            c.device.cmd_set_stencil_reference(cmd, vk::StencilFaceFlags::FRONT, refm & 0xFF);
            c.device.cmd_set_stencil_reference(cmd, vk::StencilFaceFlags::BACK, rb & 0xFF);
            c.device.cmd_set_stencil_compare_mask(cmd, vk::StencilFaceFlags::FRONT, (refm >> 8) & 0xFF);
            c.device.cmd_set_stencil_compare_mask(cmd, vk::StencilFaceFlags::BACK, (rb >> 8) & 0xFF);
            c.device.cmd_set_stencil_write_mask(cmd, vk::StencilFaceFlags::FRONT, (refm >> 16) & 0xFF);
            c.device.cmd_set_stencil_write_mask(cmd, vk::StencilFaceFlags::BACK, (rb >> 16) & 0xFF);
            // PA_SU_SC_MODE_CNTL.POLY_OFFSET_FRONT/BACK_ENABLE; registers
            // PA_SU_POLY_OFFSET_CLAMP (0xA2DF), FRONT_SCALE (0xA2E0, 16x the
            // slope factor), FRONT_OFFSET (0xA2E1), BACK_* (0xA2E2/3).
            let poly_offset = mode & (3 << 11) != 0;
            c.device.cmd_set_depth_bias_enable(cmd, poly_offset);
            if poly_offset {
                let (sr, or) = if mode & (1 << 11) != 0 { (0xA2E0, 0xA2E1) } else { (0xA2E2, 0xA2E3) };
                c.device.cmd_set_depth_bias(cmd, f32r(regs.ctx(or)), f32r(regs.ctx(0xA2DF)), f32r(regs.ctx(sr)) / 16.0);
            }
            c.device.cmd_set_line_width(cmd, 1.0);
            let bc = [f32r(regs.ctx(0xA105)), f32r(regs.ctx(0xA106)), f32r(regs.ctx(0xA107)), f32r(regs.ctx(0xA108))];
            c.device.cmd_set_blend_constants(cmd, &bc);
        }
    }

    fn create_gfx_pipeline(&mut self, key: &GfxKey, vs: &Prepared, ps: Option<&Prepared>) -> Result<Pipe, String> {
        super::set_phase(super::phase::PIPELINE, vs.addr);
        let c = ctx();
        let vsm = self.module(key.vs, &vs.translated.out.spirv)?;
        let psm = match ps {
            Some(p) => Some(self.module(key.ps.unwrap(), &p.translated.out.spirv)?),
            None => None,
        };
        let gsm = if key.rect { Some(self.rect_gs(key.gs_params)?) } else { None };
        let mut stages_p = vec![vs];
        if let Some(p) = ps {
            stages_p.push(p);
        }
        let set_layout = self.set_layout(&stages_p)?;
        let layouts = [set_layout];
        let layout = unsafe { c.device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts), None) }.map_err(|e| e.to_string())?;
        let main = c"main";
        let mut stages = vec![vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::VERTEX).module(vsm).name(main)];
        if let Some(g) = gsm {
            stages.push(vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::GEOMETRY).module(g).name(main));
        }
        if let Some(p) = psm {
            stages.push(vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::FRAGMENT).module(p).name(main));
        }
        let vi = vk::PipelineVertexInputStateCreateInfo::default();
        let topo = match key.topology_class {
            0 => vk::PrimitiveTopology::POINT_LIST,
            1 => vk::PrimitiveTopology::LINE_LIST,
            _ => vk::PrimitiveTopology::TRIANGLE_LIST,
        };
        let ia = vk::PipelineInputAssemblyStateCreateInfo::default().topology(topo);
        let vps = vk::PipelineViewportStateCreateInfo::default().viewport_count(1).scissor_count(1);
        let rs = vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(key.depth_clamp)
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0);
        let ms = vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let ds = vk::PipelineDepthStencilStateCreateInfo::default();
        let atts: Vec<vk::PipelineColorBlendAttachmentState> = key
            .colors
            .iter()
            .enumerate()
            .map(|(i, f)| {
                let b = key.blend.get(i).copied().unwrap_or(0);
                let m = key.masks.get(i).copied().unwrap_or(0);
                let enable = b & (1 << 30) != 0 && *f != vk::Format::UNDEFINED;
                let sep = b & (1 << 29) != 0;
                let (cs, cd, cf) = (b & 0x1F, (b >> 8) & 0x1F, (b >> 5) & 7);
                let (as_, ad, af) = if sep { ((b >> 16) & 0x1F, (b >> 24) & 0x1F, (b >> 21) & 7) } else { (cs, cd, cf) };
                vk::PipelineColorBlendAttachmentState::default()
                    .blend_enable(enable)
                    .src_color_blend_factor(blend_factor(cs))
                    .dst_color_blend_factor(blend_factor(cd))
                    .color_blend_op(blend_op(cf))
                    .src_alpha_blend_factor(blend_factor(as_))
                    .dst_alpha_blend_factor(blend_factor(ad))
                    .alpha_blend_op(blend_op(af))
                    .color_write_mask(vk::ColorComponentFlags::from_raw(m))
            })
            .collect();
        let cb = vk::PipelineColorBlendStateCreateInfo::default().attachments(&atts);
        let dyns = [
            vk::DynamicState::VIEWPORT,
            vk::DynamicState::SCISSOR,
            vk::DynamicState::LINE_WIDTH,
            vk::DynamicState::DEPTH_BIAS,
            vk::DynamicState::BLEND_CONSTANTS,
            vk::DynamicState::STENCIL_COMPARE_MASK,
            vk::DynamicState::STENCIL_WRITE_MASK,
            vk::DynamicState::STENCIL_REFERENCE,
            vk::DynamicState::CULL_MODE,
            vk::DynamicState::FRONT_FACE,
            vk::DynamicState::PRIMITIVE_TOPOLOGY,
            vk::DynamicState::DEPTH_TEST_ENABLE,
            vk::DynamicState::DEPTH_WRITE_ENABLE,
            vk::DynamicState::DEPTH_COMPARE_OP,
            vk::DynamicState::DEPTH_BOUNDS_TEST_ENABLE,
            vk::DynamicState::STENCIL_TEST_ENABLE,
            vk::DynamicState::STENCIL_OP,
            vk::DynamicState::DEPTH_BIAS_ENABLE,
            vk::DynamicState::PRIMITIVE_RESTART_ENABLE,
            vk::DynamicState::RASTERIZER_DISCARD_ENABLE,
        ];
        let dy = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dyns);
        let mut rinfo = vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&key.colors).depth_attachment_format(key.depth);
        if key.stencil {
            rinfo = rinfo.stencil_attachment_format(key.depth);
        }
        let gpci = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vi)
            .input_assembly_state(&ia)
            .viewport_state(&vps)
            .rasterization_state(&rs)
            .multisample_state(&ms)
            .depth_stencil_state(&ds)
            .color_blend_state(&cb)
            .dynamic_state(&dy)
            .layout(layout)
            .push_next(&mut rinfo);
        let pipeline = unsafe { c.device.create_graphics_pipelines(vk::PipelineCache::null(), &[gpci], None) }.map_err(|(_, e)| format!("vkCreateGraphicsPipelines: {e}"))?[0];
        crate::debug!("created graphics pipeline vs {:#x} ps {:?} colors {:?} depth {:?} rect {}", vs.addr, ps.map(|p| p.addr), key.colors, key.depth, key.rect);
        Ok(Pipe { pipeline, layout, set_layout })
    }

    // -------------------------------------------------------------------
    // Dispatch

    fn dispatch_inner(&mut self, cp: &CommandProcessor, x: u32, y: u32, z: u32) -> Result<(), String> {
        super::set_phase(super::phase::DISPATCH, ((cp.regs.sh(0x2E0C) as u64) | ((cp.regs.sh(0x2E0D) as u64) << 32)) << 8);
        if trace_frame() {
            self.flush();
        }
        if self.ring.remaining() < self.ring.segment_size() / 4 {
            self.flushes[2] += 1;
            self.submit_batch(Vec::new());
        }
        self.begin();
        let tp = std::time::Instant::now();
        let cs = shaders::prepare(&StageRegs { stage: Stage::Compute, regs: &cp.regs })?;
        self.prof[5] += tp.elapsed();
        if trace_frame() {
            let res: Vec<String> = stage_res_summary(&cs);
            crate::info!("FRAME {} dispatch {}x{}x{} cs {:#x} res [{}] written {:?}", cp.name, x, y, z, cs.addr, res.join(", "), cs.written_buffers);
        }
        let key = cs.translated.key_hash ^ cs.addr;
        if !self.cs_pipes.contains_key(&key) {
            let c = ctx();
            let m = self.module(key, &cs.translated.out.spirv)?;
            let set_layout = self.set_layout(&[&cs])?;
            let layouts = [set_layout];
            let layout = unsafe { c.device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts), None) }.map_err(|e| e.to_string())?;
            let stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(m).name(c"main");
            let ci = vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout).flags(vk::PipelineCreateFlags::DISPATCH_BASE);
            let t0 = std::time::Instant::now();
            let pipeline = unsafe { c.device.create_compute_pipelines(vk::PipelineCache::null(), &[ci], None) }.map_err(|(_, e)| format!("vkCreateComputePipelines: {e}"))?[0];
            let ms = t0.elapsed().as_millis();
            if ms >= 100 {
                crate::info!("compute pipeline for cs {:#x} ({} SPIR-V words) took {ms} ms to create", cs.addr, cs.translated.out.spirv.len());
            }
            self.cs_pipes.insert(key, Pipe { pipeline, layout, set_layout });
        }
        let (pipeline, layout, set_layout) = {
            let p = &self.cs_pipes[&key];
            (p.pipeline, p.layout, p.set_layout)
        };
        let (host, dev) = Self::binding_bytes(&[&cs]);
        let tex = self.texture_upload_estimate(&[&cs]);
        self.reserve(host + tex, dev);
        self.prepare_guest_reads(&[&cs]);
        self.begin();
        let set = self.alloc_set(set_layout)?;
        let wb_before = self.writebacks.len() + self.zc_writes.len();
        self.write_descriptors(set, &[&cs])?;
        self.end_rendering();
        if self.barrier_needed {
            global_barrier(self.cmd);
        }
        let c = ctx();
        let regs = &cp.regs;
        let (sx, sy, sz) = (regs.sh(0x2E04), regs.sh(0x2E05), regs.sh(0x2E06));
        unsafe {
            c.device.cmd_bind_pipeline(self.cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            c.device.cmd_bind_descriptor_sets(self.cmd, vk::PipelineBindPoint::COMPUTE, layout, 0, &[set], &[]);
            if sx | sy | sz != 0 {
                c.device.cmd_dispatch_base(self.cmd, sx, sy, sz, x.saturating_sub(sx), y.saturating_sub(sy), z.saturating_sub(sz));
            } else {
                c.device.cmd_dispatch(self.cmd, x, y, z);
            }
        }
        self.barrier_needed = true;
        self.dispatches += 1;
        if self.writebacks.len() + self.zc_writes.len() > wb_before {
            self.epoch += 1;
        }
        if trace_frame() {
            self.trace_timing("dispatch");
        }
        Ok(())
    }

    /// GPU buffer writes to [base, base+size) overlapping the guest surface
    /// of a rendered image: guest memory is now newer than the image.
    /// (Callers skip bindings larger than SURFACE_WRITE_MAX.)
    fn note_buffer_write(&mut self, base: u64, size: u64) {
        for (k, img) in self.images.iter_mut() {
            if !img.gpu_written || k.ty3d {
                continue;
            }
            let Some((_, _, _, total)) = Self::image_guest_layout(k, img) else { continue };
            if k.base < base + size && base < k.base + total {
                img.guest_newer = true;
                img.guest_stale = false;
            }
        }
    }

    /// Reloads a render target from its (tiled) guest surface after buffer
    /// writes made guest memory newer. Must run before descriptor
    /// allocation of the current draw/dispatch (it may wait for write-backs).
    fn refresh_from_guest(&mut self, key: ImageKey, may_submit: bool) -> Result<(), String> {
        let Some(img) = self.images.get_mut(&key) else { return Ok(()) };
        img.guest_newer = false;
        let Some((bpp, mode, ph, total)) = Self::image_guest_layout(&key, img) else { return Ok(()) };
        // (Depth+stencil targets: the depth aspect is reloaded; stencil lives
        // in a separate guest surface.)
        self.sync_guest(key.base, total);
        let t0 = std::time::Instant::now();
        let mut tiled = vec![0u8; total as usize];
        resources::copy_from_guest(&mut tiled, key.base);
        let lin_slice = key.width as usize * key.height as usize * (bpp / 8) as usize;
        let slice = total as usize / key.layers.max(1) as usize;
        let mut lin = vec![0u8; lin_slice * key.layers as usize];
        for l in 0..key.layers as usize {
            tiling::detile_slice(&tiled[l * slice..(l + 1) * slice], &mut lin[l * lin_slice..(l + 1) * lin_slice], key.width, key.height, key.width, ph, bpp, mode);
        }
        self.prof[4] += t0.elapsed();
        if may_submit && self.ring.remaining() < lin.len() as u64 + self.ring.segment_size() / 8 {
            self.flushes[2] += 1;
            self.submit_batch(Vec::new());
        }
        let off = self.ring.alloc(lin.len() as u64).ok_or("ring full (render target reload)")?;
        self.ring.write(off, &lin);
        self.begin();
        self.end_rendering();
        global_barrier(self.cmd);
        let region = vk::BufferImageCopy {
            buffer_offset: off,
            buffer_row_length: 0,
            buffer_image_height: 0,
            image_subresource: vk::ImageSubresourceLayers {
                aspect_mask: if key.depth { vk::ImageAspectFlags::DEPTH } else { vk::ImageAspectFlags::COLOR },
                mip_level: 0,
                base_array_layer: 0,
                layer_count: key.layers,
            },
            image_offset: vk::Offset3D::default(),
            image_extent: vk::Extent3D { width: key.width, height: key.height, depth: 1 },
        };
        unsafe { ctx().device.cmd_copy_buffer_to_image(self.cmd, self.ring.buffer, self.images[&key].image, vk::ImageLayout::GENERAL, &[region]) };
        self.barrier_needed = true;
        // Shadow copies of the old contents are out of date.
        *self.write_gen.entry(key).or_default() += 1;
        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 12 {
            crate::info!("render target {:#x} {}x{} {:?} reloaded from guest memory (written by buffer stores)", key.base, key.width, key.height, key.format);
        }
        Ok(())
    }

    /// GPU buffer writes to [base, base+size) that touch a render target's
    /// HTILE/CMASK surface: examined when the target is next bound.
    fn note_meta_write(&mut self, base: u64, size: u64) {
        for img in self.images.values_mut() {
            if let Some((m, n)) = img.meta {
                if m < base + size && base < m + n {
                    img.meta_dirty = true;
                }
            }
        }
    }

    /// Fast clears: titles clear depth by writing HTILE (ZMask 0 = tile holds
    /// the clear value DB_DEPTH_CLEAR) and color targets by writing CMASK
    /// (0 = tile holds CB_COLOR*_CLEAR_WORD*), typically with a compute fill.
    /// When the metadata of a target about to be rendered was written, the
    /// cleared state is applied to the image (fully cleared surfaces only;
    /// HTILE/CMASK are not otherwise emulated).
    fn apply_meta_clears(&mut self, regs: &Regs, colors: &[Option<Attachment>], depth: Option<Attachment>) {
        let mut todo: Vec<(ImageKey, Option<u32>)> = Vec::new();
        for (n, a) in colors.iter().enumerate() {
            if let Some(a) = a {
                if self.images.get(&a.key).is_some_and(|i| i.meta_dirty) {
                    todo.push((a.key, Some(n as u32)));
                }
            }
        }
        if let Some(d) = depth {
            if self.images.get(&d.key).is_some_and(|i| i.meta_dirty) {
                todo.push((d.key, None));
            }
        }
        for (key, slot) in todo {
            let Some((mbase, msize)) = self.images.get_mut(&key).and_then(|i| {
                i.meta_dirty = false;
                i.meta
            }) else {
                continue;
            };
            // The metadata must have landed in guest memory (waits if needed).
            self.sync_guest(mbase, msize);
            let probe = |off: u64| -> Option<u32> {
                let a = mbase + (off & !3);
                (a >= crate::mem::GUEST_BASE && a + 4 <= crate::mem::GUEST_END).then(|| unsafe { std::ptr::read_volatile(a as *const u32) })
            };
            let offs: Vec<u64> = (0..16).map(|k| k * 4).chain((0..16).map(|k| ((msize / 2) & !3) + k * 4)).chain((0..16).map(|k| msize.saturating_sub(64) + k * 4)).collect();
            let words: Vec<u32> = offs.iter().filter_map(|&o| probe(o)).collect();
            if words.is_empty() {
                continue;
            }
            static SEEN: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if SEEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 12 {
                crate::info!("metadata of {:#x} ({}) written: {:#x}+{:#x} sample {:08x?}", key.base, if slot.is_some() { "CMASK" } else { "HTILE" }, mbase, msize, &words[..4.min(words.len())]);
            }
            let c = ctx();
            let img = self.images[&key].image;
            match slot {
                None => {
                    // HTILE: ZMask [3:0] == 0 -> depth cleared; with stencil
                    // tiles enabled, SMem [9:8] == 0 -> stencil cleared.
                    let depth_clear = words.iter().all(|w| w & 0xF == 0);
                    let stencil_tiles = regs.ctx(0xA011) & (1 << 29) == 0 && key.format == vk::Format::D32_SFLOAT_S8_UINT;
                    let stencil_clear = stencil_tiles && words.iter().all(|w| (w >> 8) & 3 == 0);
                    let mut aspect = vk::ImageAspectFlags::empty();
                    if depth_clear {
                        aspect |= vk::ImageAspectFlags::DEPTH;
                    }
                    if stencil_clear {
                        aspect |= vk::ImageAspectFlags::STENCIL;
                    }
                    if aspect.is_empty() {
                        continue;
                    }
                    let value = vk::ClearDepthStencilValue { depth: f32r(regs.ctx(0xA00B)), stencil: regs.ctx(0xA00A) & 0xFF };
                    self.begin();
                    self.end_rendering();
                    global_barrier(self.cmd);
                    let range = vk::ImageSubresourceRange { aspect_mask: aspect, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: key.layers };
                    unsafe { c.device.cmd_clear_depth_stencil_image(self.cmd, img, vk::ImageLayout::GENERAL, &value, &[range]) };
                    self.barrier_needed = true;
                    static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                    if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 8 {
                        crate::info!("HTILE clear of depth {:#x} ({:?}, depth {} stencil {})", key.base, aspect, value.depth, value.stencil);
                    }
                }
                Some(n) => {
                    // CMASK: all tiles 0 -> fast cleared to the clear words.
                    if !words.iter().all(|&w| w == 0) {
                        continue;
                    }
                    let r = 0xA318 + n * 0xF;
                    let (w0, w1) = (regs.ctx(r + 11), regs.ctx(r + 12));
                    let format = colors[n as usize].map(|a| a.view_format).unwrap_or(key.format);
                    let Some(value) = clear_color(format, w0, w1) else {
                        self.report(format!("fast clear of {format:?} with clear words {w0:#x}/{w1:#x} not supported"));
                        continue;
                    };
                    self.begin();
                    self.end_rendering();
                    global_barrier(self.cmd);
                    let range = vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: key.layers };
                    unsafe { c.device.cmd_clear_color_image(self.cmd, img, vk::ImageLayout::GENERAL, &value, &[range]) };
                    self.barrier_needed = true;
                    static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                    if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 8 {
                        crate::info!("CMASK fast clear of {:#x} {:?} to {w0:#x}/{w1:#x}", key.base, format);
                    }
                }
            }
            if let Some(i) = self.images.get_mut(&key) {
                i.gpu_written = true;
                i.guest_stale = true;
            }
            // The hardware marks tiles expanded as it renders them (and an
            // eliminate pass expands the rest); we applied the whole clear, so
            // record the expanded state: a later unrelated write near the
            // metadata must not make the stale "cleared" state reapply.
            if mbase >= crate::mem::GUEST_BASE && mbase + msize <= crate::mem::GUEST_END {
                let expanded = vec![0xFFu8; msize as usize];
                resources::copy_to_guest(&expanded, mbase);
            }
        }
    }

    /// Records a GDS transfer after all work recorded so far.
    fn gds_op(&mut self, op: super::cp::GdsOp) -> Result<(), String> {
        use super::cp::GdsOp;
        let limit = gcn::translate::GDS_BYTES;
        let clamp = |offset: u32, bytes: u32| -> Option<(u64, u64)> {
            let (o, n) = (offset & !3, bytes & !3);
            (n > 0 && o < limit).then(|| (o as u64, n.min(limit - o) as u64))
        };
        if crate::log::enabled(crate::log::Level::Debug) {
            crate::debug!("gds: {op:?}");
        }
        match op {
            GdsOp::Fill { offset, bytes, value } => {
                let Some((o, n)) = clamp(offset, bytes) else { return Ok(()) };
                self.begin();
                self.end_rendering();
                global_barrier(self.cmd);
                unsafe { ctx().device.cmd_fill_buffer(self.cmd, self.gds, o, n, value) };
            }
            GdsOp::Load { offset, src, bytes } => {
                let Some((o, n)) = clamp(offset, bytes) else { return Ok(()) };
                self.reserve(n.next_multiple_of(256), 0);
                self.begin();
                let off = self.upload_guest(src, n)?;
                self.end_rendering();
                global_barrier(self.cmd);
                let region = vk::BufferCopy { src_offset: off, dst_offset: o, size: n };
                unsafe { ctx().device.cmd_copy_buffer(self.cmd, self.ring.buffer, self.gds, &[region]) };
            }
            GdsOp::Store { offset, dst, bytes } => {
                let Some((o, n)) = clamp(offset, bytes) else { return Ok(()) };
                self.reserve(n.next_multiple_of(256), n.next_multiple_of(256));
                self.begin();
                // Lands in guest memory like a shader write: device ring now,
                // host ring at batch end, guest memory on completion.
                let host_off = self.ring.alloc(n).ok_or("ring full")?;
                let dev_off = self.dring.alloc(n).ok_or("device ring full")?;
                self.end_rendering();
                global_barrier(self.cmd);
                let region = vk::BufferCopy { src_offset: o, dst_offset: dev_off, size: n };
                unsafe { ctx().device.cmd_copy_buffer(self.cmd, self.gds, self.dring.buffer, &[region]) };
                self.writebacks.push(PendingWrite { base: dst, size: n, host_off, dev_off });
                self.note_buffer_write(dst, n);
                self.epoch += 1;
            }
        }
        self.barrier_needed = true;
        Ok(())
    }

    /// The most frequent reported problems (message, count).
    pub fn top_reports(&self, n: usize) -> Vec<(String, u64)> {
        let mut v: Vec<(String, u64)> = self.reported.iter().map(|(m, c)| (m.clone(), *c)).collect();
        v.sort_by(|a, b| b.1.cmp(&a.1));
        v.truncate(n);
        v
    }

    pub fn image_count(&self) -> usize {
        self.images.len()
    }
    pub fn pipeline_count(&self) -> usize {
        self.gfx_pipes.len() + self.cs_pipes.len()
    }

    /// Image to present for a display buffer (after a flush).
    pub fn display_image(&mut self, base: u64, w: u32, h: u32) -> Option<(vk::Image, u32, u32)> {
        let k = self.display_key(base, w, h)?;
        Some((self.images[&k].image, w, h))
    }

    fn display_key(&self, base: u64, w: u32, h: u32) -> Option<ImageKey> {
        let ks = self.by_base.get(&base)?;
        ks.iter().copied().find(|k| !k.depth && k.width >= w && k.height >= h && self.images[k].gpu_written)
    }

    /// Reads back a display image as RGBA8 (for screenshots).
    pub fn capture(&mut self, base: u64, w: u32, h: u32) -> Option<Vec<u8>> {
        let k = self.display_key(base, w, h)?;
        if format_class(k.format) != 4 {
            return None;
        }
        self.begin();
        self.end_rendering();
        global_barrier(self.cmd);
        let off = self.ring.alloc((w * h * 4) as u64)?;
        let region = vk::BufferImageCopy {
            buffer_offset: off,
            buffer_row_length: 0,
            buffer_image_height: 0,
            image_subresource: vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 },
            image_offset: vk::Offset3D::default(),
            image_extent: vk::Extent3D { width: w, height: h, depth: 1 },
        };
        unsafe { ctx().device.cmd_copy_image_to_buffer(self.cmd, self.images[&k].image, vk::ImageLayout::GENERAL, self.ring.buffer, &[region]) };
        self.flush();
        let mut px = unsafe { std::slice::from_raw_parts(self.ring.ptr.add(off as usize), (w * h * 4) as usize) }.to_vec();
        if matches!(k.format, vk::Format::B8G8R8A8_UNORM | vk::Format::B8G8R8A8_SRGB) {
            for p in px.chunks_mut(4) {
                p.swap(0, 2);
            }
        }
        for p in px.chunks_mut(4) {
            p[3] = 255;
        }
        Some(px)
    }
}

impl Renderer {
    /// Debug aid (BB_DUMP_IMAGES=<flip>): reads back every GPU-written color
    /// image (first layer, base level), logs channel statistics and writes
    /// a viewable PNG next to the run log.
    pub fn dump_images(&mut self, flip: u64) {
        let Some(dir) = super::shader_dump::DIR.get().and_then(|d| d.parent()).map(|d| d.join(format!("images{flip:05}"))) else { return };
        let _ = std::fs::create_dir_all(&dir);
        let mut keys: Vec<ImageKey> = self.images.iter().filter(|(k, i)| i.gpu_written && !k.ty3d).map(|(k, _)| *k).collect();
        keys.sort_by_key(|k| (k.base, k.width, k.height));
        for k in keys {
            for layer in 0..k.layers.min(6) {
                self.dump_image(&dir, k, layer, "");
            }
        }
    }

    /// Debug: writes one layer of an image as PNG (and logs value ranges).
    fn dump_image(&mut self, dir: &std::path::Path, k: ImageKey, layer: u32, prefix: &str) {
        // Depth targets: the depth aspect as floats (D16 as 16-bit unorm).
        let (texel, aspect, view_fmt) = match k.format {
            vk::Format::D32_SFLOAT | vk::Format::D32_SFLOAT_S8_UINT => (4, vk::ImageAspectFlags::DEPTH, vk::Format::R32_SFLOAT),
            vk::Format::D16_UNORM => (2, vk::ImageAspectFlags::DEPTH, vk::Format::R16_UNORM),
            f => match format_class(f) {
                c @ (1 | 2 | 4 | 8 | 16) => (c, vk::ImageAspectFlags::COLOR, f),
                _ => return,
            },
        };
        let size = (k.width * k.height * texel) as u64;
        self.flush();
        self.begin();
        global_barrier(self.cmd);
        let Some(off) = self.ring.alloc(size) else { return };
        let region = vk::BufferImageCopy {
            buffer_offset: off,
            buffer_row_length: 0,
            buffer_image_height: 0,
            image_subresource: vk::ImageSubresourceLayers { aspect_mask: aspect, mip_level: 0, base_array_layer: layer, layer_count: 1 },
            image_offset: vk::Offset3D::default(),
            image_extent: vk::Extent3D { width: k.width, height: k.height, depth: 1 },
        };
        unsafe { ctx().device.cmd_copy_image_to_buffer(self.cmd, self.images[&k].image, vk::ImageLayout::GENERAL, self.ring.buffer, &[region]) };
        self.flush();
        let raw = unsafe { std::slice::from_raw_parts(self.ring.ptr.add(off as usize), size as usize) }.to_vec();
        let (rgba, stats) = to_viewable(view_fmt, &raw, k.width, k.height);
        crate::info!("dump {prefix}{:#x} {}x{} layer {} {:?}: {}", k.base, k.width, k.height, layer, k.format, stats);
        if let Some(px) = rgba {
            let name = format!("{prefix}img_{:x}_{}x{}_l{}_{:?}.png", k.base, k.width, k.height, layer, k.format);
            let _ = std::fs::write(dir.join(name), super::texdump::png(k.width, k.height, &px));
        }
    }

    /// Frame trace with `BB_DUMP_AFTER=<ps addrs>`: dumps the color targets of
    /// draws using one of the listed pixel shaders right after the draw.
    fn dump_after(&mut self, ps: u64, colors: &[Option<Attachment>], depth: Option<Attachment>) {
        static WANT: std::sync::OnceLock<Vec<u64>> = std::sync::OnceLock::new();
        let want = WANT.get_or_init(|| {
            std::env::var("BB_DUMP_AFTER").map(|v| v.split(',').filter_map(|s| u64::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok()).collect()).unwrap_or_default()
        });
        if !want.contains(&ps) {
            return;
        }
        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // At most BB_DUMP_AFTER_MAX (default 6) dumps per run.
        static MAX: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
        if n >= *MAX.get_or_init(|| std::env::var("BB_DUMP_AFTER_MAX").ok().and_then(|v| v.parse().ok()).unwrap_or(6)) {
            return;
        }
        let Some(dir) = super::shader_dump::DIR.get().and_then(|d| d.parent()).map(|d| d.join("after")) else { return };
        let _ = std::fs::create_dir_all(&dir);
        self.end_rendering();
        for a in colors.iter().flatten().chain(depth.iter()) {
            self.dump_image(&dir, a.key, 0, &format!("{n:03}_ps{ps:x}_"));
        }
    }
}

/// Clear value for a color image from CB_COLOR*_CLEAR_WORD0/1, which hold
/// the clear color in the surface's own memory layout.
fn clear_color(f: vk::Format, w0: u32, w1: u32) -> Option<vk::ClearColorValue> {
    use vk::Format as F;
    let b = |i: u32| ((w0 >> (8 * i)) & 0xFF) as f32 / 255.0;
    let lin = |c: f32| if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) };
    // Unsigned small floats (5-bit exponent, no sign).
    let uf = |v: u32, mbits: u32| -> f32 {
        let m = v & ((1 << mbits) - 1);
        let e = (v >> mbits) & 0x1F;
        match e {
            0 => m as f32 / (1 << mbits) as f32 * 2f32.powi(-14),
            31 => if m == 0 { f32::INFINITY } else { f32::NAN },
            _ => (1.0 + m as f32 / (1 << mbits) as f32) * 2f32.powi(e as i32 - 15),
        }
    };
    let h = |v: u32| half(v as u16);
    let float = |v: [f32; 4]| Some(vk::ClearColorValue { float32: v });
    match f {
        F::R8G8B8A8_UNORM => float([b(0), b(1), b(2), b(3)]),
        F::R8G8B8A8_SRGB => float([lin(b(0)), lin(b(1)), lin(b(2)), b(3)]),
        F::B8G8R8A8_UNORM => float([b(2), b(1), b(0), b(3)]),
        F::B8G8R8A8_SRGB => float([lin(b(2)), lin(b(1)), lin(b(0)), b(3)]),
        F::R8_UNORM => float([b(0), 0.0, 0.0, 1.0]),
        F::R8G8_UNORM => float([b(0), b(1), 0.0, 1.0]),
        F::B10G11R11_UFLOAT_PACK32 => float([uf(w0 & 0x7FF, 6), uf((w0 >> 11) & 0x7FF, 6), uf(w0 >> 22, 5), 1.0]),
        F::A2B10G10R10_UNORM_PACK32 => float([(w0 & 0x3FF) as f32 / 1023.0, ((w0 >> 10) & 0x3FF) as f32 / 1023.0, ((w0 >> 20) & 0x3FF) as f32 / 1023.0, (w0 >> 30) as f32 / 3.0]),
        F::R16_SFLOAT => float([h(w0), 0.0, 0.0, 1.0]),
        F::R16G16_SFLOAT => float([h(w0), h(w0 >> 16), 0.0, 1.0]),
        F::R16G16B16A16_SFLOAT => float([h(w0), h(w0 >> 16), h(w1), h(w1 >> 16)]),
        F::R16_UNORM => float([(w0 & 0xFFFF) as f32 / 65535.0, 0.0, 0.0, 1.0]),
        F::R16G16_UNORM => float([(w0 & 0xFFFF) as f32 / 65535.0, (w0 >> 16) as f32 / 65535.0, 0.0, 1.0]),
        F::R16G16B16A16_UNORM => float([(w0 & 0xFFFF) as f32 / 65535.0, (w0 >> 16) as f32 / 65535.0, (w1 & 0xFFFF) as f32 / 65535.0, (w1 >> 16) as f32 / 65535.0]),
        F::R32_SFLOAT => float([f32::from_bits(w0), 0.0, 0.0, 1.0]),
        F::R32G32_SFLOAT => float([f32::from_bits(w0), f32::from_bits(w1), 0.0, 1.0]),
        F::R32_UINT => Some(vk::ClearColorValue { uint32: [w0, 0, 0, 1] }),
        F::R32G32_UINT => Some(vk::ClearColorValue { uint32: [w0, w1, 0, 1] }),
        _ if w0 == 0 && w1 == 0 => Some(vk::ClearColorValue { uint32: [0; 4] }),
        _ => None,
    }
}

fn half(h: u16) -> f32 {
    let s = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let e = ((h >> 10) & 0x1F) as i32;
    let m = (h & 0x3FF) as f32;
    match e {
        0 => s * m * 2f32.powi(-24),
        31 => if m == 0.0 { s * f32::INFINITY } else { f32::NAN },
        _ => s * (1.0 + m / 1024.0) * 2f32.powi(e - 15),
    }
}

/// Converts raw image data to RGBA8 for viewing; returns per-channel stats.
fn to_viewable(f: vk::Format, raw: &[u8], w: u32, h: u32) -> (Option<Vec<u8>>, String) {
    use vk::Format as F;
    let n = (w * h) as usize;
    let mut chans: Vec<[f32; 4]> = Vec::with_capacity(n);
    match f {
        F::R8G8B8A8_UNORM | F::R8G8B8A8_SRGB => chans.extend(raw.chunks_exact(4).map(|p| [p[0] as f32 / 255.0, p[1] as f32 / 255.0, p[2] as f32 / 255.0, p[3] as f32 / 255.0])),
        F::B8G8R8A8_UNORM | F::B8G8R8A8_SRGB => chans.extend(raw.chunks_exact(4).map(|p| [p[2] as f32 / 255.0, p[1] as f32 / 255.0, p[0] as f32 / 255.0, p[3] as f32 / 255.0])),
        F::R32_SFLOAT => chans.extend(raw.chunks_exact(4).map(|p| {
            let v = f32::from_le_bytes([p[0], p[1], p[2], p[3]]);
            [v, v, v, 1.0]
        })),
        F::R16G16B16A16_SFLOAT => chans.extend(raw.chunks_exact(8).map(|p| {
            let g = |i: usize| half(u16::from_le_bytes([p[i], p[i + 1]]));
            [g(0), g(2), g(4), g(6)]
        })),
        F::R16G16_SFLOAT => chans.extend(raw.chunks_exact(4).map(|p| {
            let g = |i: usize| half(u16::from_le_bytes([p[i], p[i + 1]]));
            [g(0), g(2), 0.0, 1.0]
        })),
        F::R16_SFLOAT => chans.extend(raw.chunks_exact(2).map(|p| {
            let v = half(u16::from_le_bytes([p[0], p[1]]));
            [v, v, v, 1.0]
        })),
        F::R8_UNORM => chans.extend(raw.iter().map(|&v| [v as f32 / 255.0, v as f32 / 255.0, v as f32 / 255.0, 1.0])),
        F::B10G11R11_UFLOAT_PACK32 => chans.extend(raw.chunks_exact(4).map(|p| {
            let v = u32::from_le_bytes([p[0], p[1], p[2], p[3]]);
            let f11 = |x: u32| { let e = ((x >> 6) & 0x1F) as i32; let m = (x & 0x3F) as f32; if e == 0 { m * 2f32.powi(-20) } else { (1.0 + m / 64.0) * 2f32.powi(e - 15) } };
            let f10 = |x: u32| { let e = ((x >> 5) & 0x1F) as i32; let m = (x & 0x1F) as f32; if e == 0 { m * 2f32.powi(-19) } else { (1.0 + m / 32.0) * 2f32.powi(e - 15) } };
            [f11(v & 0x7FF), f11((v >> 11) & 0x7FF), f10(v >> 22), 1.0]
        })),
        _ => return (None, format!("format not viewable ({} bytes)", raw.len())),
    }
    let mut mn = [f32::INFINITY; 4];
    let mut mx = [f32::NEG_INFINITY; 4];
    let mut sum = [0f64; 4];
    for c in &chans {
        for i in 0..4 {
            mn[i] = mn[i].min(c[i]);
            mx[i] = mx[i].max(c[i]);
            sum[i] += c[i] as f64;
        }
    }
    let stats = format!(
        "min [{:.3} {:.3} {:.3} {:.3}] max [{:.3} {:.3} {:.3} {:.3}] mean [{:.3} {:.3} {:.3} {:.3}]",
        mn[0], mn[1], mn[2], mn[3], mx[0], mx[1], mx[2], mx[3],
        sum[0] / n as f64, sum[1] / n as f64, sum[2] / n as f64, sum[3] / n as f64
    );
    let px: Vec<u8> = chans.iter().flat_map(|c| [c[0], c[1], c[2]].map(|v| (v.clamp(0.0, 1.0) * 255.0) as u8).into_iter().chain([255])).collect();
    (Some(px), stats)
}

fn buffer_written(p: &Prepared, i: usize) -> bool {
    p.translated.out.bindings.iter().any(|b| b.kind == BindingKind::Buffer(i)) && p.written_buffers.contains(&i)
}

/// Storage format for an image: sRGB stored as UNORM (views reinterpret).
fn canonical(f: vk::Format) -> vk::Format {
    match f {
        vk::Format::R8G8B8A8_SRGB => vk::Format::R8G8B8A8_UNORM,
        vk::Format::B8G8R8A8_SRGB => vk::Format::B8G8R8A8_UNORM,
        vk::Format::R8_SRGB => vk::Format::R8_UNORM,
        f => f,
    }
}

/// Coarse format compatibility class (bytes per texel / block kind).
fn format_class(f: vk::Format) -> u32 {
    use vk::Format as F;
    match f {
        F::R8G8B8A8_UNORM | F::R8G8B8A8_SRGB | F::B8G8R8A8_UNORM | F::B8G8R8A8_SRGB | F::R8G8B8A8_SNORM | F::R8G8B8A8_UINT | F::R8G8B8A8_SINT | F::B8G8R8A8_SNORM | F::A2B10G10R10_UNORM_PACK32
        | F::R32_SFLOAT | F::R32_UINT | F::R32_SINT | F::R16G16_SFLOAT | F::R16G16_UNORM | F::R16G16_UINT | F::B10G11R11_UFLOAT_PACK32 => 4,
        F::R16G16B16A16_SFLOAT | F::R16G16B16A16_UNORM | F::R16G16B16A16_UINT | F::R32G32_SFLOAT | F::R32G32_UINT => 8,
        F::R32G32B32A32_SFLOAT | F::R32G32B32A32_UINT => 16,
        F::R8_UNORM | F::R8_UINT | F::R8_SRGB | F::R8_SNORM => 1,
        F::R16_UNORM | F::R16_SFLOAT | F::R16_UINT | F::R8G8_UNORM | F::R8G8_UINT => 2,
        other => 0x1000 + other.as_raw() as u32,
    }
}

/// Hooks implementation that drives the Vulkan renderer.
pub struct VkHooks {
    pub fallback: super::NullRenderer,
}

impl Hooks for VkHooks {
    fn draw(&mut self, cp: &CommandProcessor, kind: &'static str, count: u32, index_addr: u64) {
        let mut g = RENDERER.lock();
        let Some(r) = g.as_mut() else { return self.fallback.draw(cp, kind, count, index_addr) };
        if let Err(e) = r.draw_inner(cp, kind, count, index_addr) {
            r.skipped += 1;
            r.report(format!("draw skipped: {e}"));
        }
    }
    fn dispatch(&mut self, cp: &CommandProcessor, x: u32, y: u32, z: u32) {
        let mut g = RENDERER.lock();
        let Some(r) = g.as_mut() else { return self.fallback.dispatch(cp, x, y, z) };
        if let Err(e) = r.dispatch_inner(cp, x, y, z) {
            r.skipped += 1;
            r.report(format!("dispatch skipped: {e}"));
        }
    }
    fn gds(&mut self, cp: &CommandProcessor, op: super::cp::GdsOp) {
        let mut g = RENDERER.lock();
        let Some(r) = g.as_mut() else { return self.fallback.gds(cp, op) };
        if let Err(e) = r.gds_op(op) {
            r.report(format!("GDS transfer skipped: {e}"));
        }
    }
    fn flush(&mut self, _cp: &CommandProcessor) {
        if let Some(r) = RENDERER.lock().as_mut() {
            r.flush_for(0);
            // The caller is about to access guest memory directly (DMA etc.).
            r.new_epoch();
        }
    }
    fn sync_range(&mut self, _cp: &CommandProcessor, addr: u64, len: u64) {
        if let Some(r) = RENDERER.lock().as_mut() {
            if r.pending_write_overlaps(addr, len) {
                r.flushes[1] += 1;
                r.submit_batch(Vec::new());
            }
        }
        wait_overlapping(addr, len);
    }
    fn commit(&mut self, cp: &CommandProcessor, actions: Vec<super::cp::EopAction>) {
        let mut g = RENDERER.lock();
        match g.as_mut() {
            Some(r) => {
                r.submit_batch(actions);
            }
            None => {
                drop(g);
                self.fallback.commit(cp, actions);
            }
        }
    }
}

/// Flip to trace requested through the control channel (`trace` command).
pub static TRACE_REQUEST: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(u64::MAX);

/// True while the flip selected by BB_FRAME_TRACE (or a `trace` request) is being built.
fn trace_frame() -> bool {
    static WANT: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    // BB_TRACE_FROM=<seconds>: trace (and run synchronously) every command
    // processed after that much process time, e.g. to pin down a GPU hang.
    static FROM: std::sync::OnceLock<Option<f64>> = std::sync::OnceLock::new();
    if let Some(from) = *FROM.get_or_init(|| std::env::var("BB_TRACE_FROM").ok().and_then(|v| v.parse::<f64>().ok())) {
        if crate::log::elapsed_ms() >= from * 1000.0 {
            return true;
        }
    }
    let frame = super::FRAMES.load(std::sync::atomic::Ordering::Relaxed);
    if TRACE_REQUEST.load(std::sync::atomic::Ordering::Relaxed) == frame {
        return true;
    }
    match *WANT.get_or_init(|| std::env::var("BB_FRAME_TRACE").ok().and_then(|v| v.parse().ok())) {
        Some(w) => frame == w,
        None => false,
    }
}

/// Frame trace: a stage's resources with the pcs of the instructions using them.
fn stage_res_summary(p: &Prepared) -> Vec<String> {
    let tag = match p.stage {
        Stage::Vertex => "vs",
        Stage::Fragment => "ps",
        Stage::Compute => "cs",
    };
    p.resources
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let pcs: Vec<String> = p.res_pcs.get(i).map(|v| v.iter().map(|pc| format!("{pc:x}")).collect()).unwrap_or_default();
            format!("{tag}{i}@{} {}", pcs.join("/"), res_summary(r))
        })
        .collect()
}

fn res_summary(r: &ResolvedRes) -> String {
    match r {
        ResolvedRes::Buffer(v) => {
            // BB_TRACE_DWORDS: how many leading dwords of each buffer to show (default 4).
            static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
            let n = (*N.get_or_init(|| std::env::var("BB_TRACE_DWORDS").ok().and_then(|v| v.parse().ok()).unwrap_or(4))).min(256);
            let mut head = vec![0u8; n * 4];
            resources::copy_from_guest(&mut head, v.base);
            let w: Vec<String> = head.chunks(4).map(|c| format!("{:08x}", u32::from_le_bytes([c[0], c[1], c[2], c[3]]))).collect();
            format!("V#{:#x}+{:#x}(s{} f{}/{} [{}])", v.base, v.size(), v.stride, v.dfmt, v.nfmt, w.join(" "))
        }
        ResolvedRes::Image(t) => format!("T#{:#x} {}x{}x{} f{}/{} t{} ty{} arr{}..{} d{} lv{}..{}", t.base, t.width, t.height, t.layers(), t.dfmt, t.nfmt, t.tiling_index, t.ty, t.base_array, t.last_array, t.depth, t.base_level, t.last_level),
        ResolvedRes::Sampler(_) => "S#".into(),
    }
}

pub const PROF_NAMES: [&str; 6] = ["submit+wait", "writeback", "buffer upload", "texture hash", "texture read", "shader prep"];

pub const FLUSH_REASONS: [&str; 6] = ["eop/dma", "writeback sync", "ring full", "descriptors", "flip", "fwd copies"];
