//! Zero-known guest pages: a zero fill's stores into guest memory left out where guest memory
//! already holds zeros (FINDINGS.md 2026-11-15 evening).
//!
//! The title clears render-target pool memory with fill kernels every frame. Each such fill is
//! recorded as a fill of front end 2's mirror (video memory: ~6 us for 15 MiB) and a store of
//! the same zeros into guest memory across PCIe (~1.47 ms for 15 MiB, ~10.7 GB/s: ~14 MiB a
//! frame in the 1-1 walk, ~85 MiB at the title screen). In a walk all 62,235 large fills found
//! guest memory still zero from the fill before: nothing stores there between them (images
//! rendered there stay in video memory).
//!
//! A bitmap in video memory holds a bit per 4 KiB page of the span: set, guest memory's copy of
//! the page is all zeros. `rdna::kernels::zero_store` stores the pages of a fill whose bit is
//! clear and sets the bits of the pages it covers whole. Every other writer of guest memory
//! clears the bits of the pages it stores into, in GPU order: front end 2's write-back kernel
//! (every listed page, in the kernel), held pages' releases, other in-place fills, CP writes,
//! homes' write-backs (`clear`). CPU writes and mapping changes: the pages any write-watch take
//! finds written (`cpu_written`), the pages whose mapping changed (`cpu_dirty_range`) and a
//! query of the fill's pages when it is recorded have their bits cleared before the store
//! (`record_store`). The bits are set only for fills of the render-target pool's memory
//! (`eligible`), so the clears are recorded only where a bit may be set (`TRACKED`).
//!
//! `tune zk` (`DS_ZK`, default 1): 0 stores every fill whole as before. `DS_ZK_AUDIT=1`
//! (diagnosis): pages left out are read and their non-zero dwords counted (status line): a
//! writer that left a bit set.

use super::resident::{MIRROR_BASE, MIRROR_END};
use super::vk::ctx;
use ash::vk;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

const PAGE: u64 = 4096;
const PAGES: u64 = (MIRROR_END - MIRROR_BASE) / PAGE;
/// The bitmap's 32-bit words (a bit per page).
const WORDS: u64 = PAGES / 32;
const BITMAP_BYTES: u64 = WORDS * 4;
/// Zero stores between copies of the GPU's counters to the host.
const COUNTER_COPY_EVERY: u64 = 64;

struct Zk {
    bits: vk::Buffer,
    bits_addr: u64,
    counters: vk::Buffer,
    counters_addr: u64,
    readback: vk::Buffer,
    readback_ptr: usize,
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
}

static ZK: OnceLock<Option<Zk>> = OnceLock::new();

/// Bitmap words where a zero store may have set bits (a bit per word): only there do other
/// writers record clears.
static TRACKED: [AtomicU64; (WORDS / 64) as usize] = [const { AtomicU64::new(0) }; (WORDS / 64) as usize];

/// Pages the CPU wrote or whose mapping changed since a zero store last handled them (a bit
/// per page), made at first use.
static DIRTY: OnceLock<Box<[AtomicU64]>> = OnceLock::new();

/// [zero stores recorded, bytes they covered, pages cleared before them for CPU writes, clears
/// recorded for other writers, CPU writes found by the fills' queries (pages), fills left to
/// the plain store (not eligible)].
static STATS: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

fn dirty() -> &'static [AtomicU64] {
    DIRTY.get_or_init(|| (0..PAGES / 64).map(|_| AtomicU64::new(0)).collect::<Vec<_>>().into_boxed_slice())
}

fn audit() -> bool {
    static A: OnceLock<bool> = OnceLock::new();
    *A.get_or_init(|| std::env::var("DS_ZK_AUDIT").ok().as_deref() == Some("1"))
}

/// Makes the bitmap, the counters and the kernel's pipeline (front end 2's set-up, the mirror
/// made); without device addresses, none (every fill stored whole).
pub fn init() {
    ZK.get_or_init(|| match make() {
        Ok(z) => {
            crate::info!("zk: zero-known pages on (a {} KiB bitmap; `tune zk`, DS_ZK_AUDIT={})", BITMAP_BYTES >> 10, audit() as u8);
            Some(z)
        }
        Err(e) => {
            crate::warn!("zk: zero-known pages off: {e}");
            None
        }
    });
}

fn make() -> Result<Zk, String> {
    let c = ctx();
    let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC;
    let (bits, _m, bits_addr) = c.device_buffer_addressed(BITMAP_BYTES, usage)?;
    let cbytes = 4 * rdna::kernels::ZERO_STORE_COUNTERS as u64;
    let (counters, _cm, counters_addr) = c.device_buffer_addressed(cbytes, usage)?;
    if bits_addr == 0 || counters_addr == 0 {
        return Err("no buffer device addresses".into());
    }
    let (readback, _rm, rptr) = c.host_buffer(cbytes, vk::BufferUsageFlags::TRANSFER_DST)?;
    unsafe { std::ptr::write_bytes(rptr, 0, cbytes as usize) };
    super::resident::one_off_now(|cmd| unsafe {
        c.device.cmd_fill_buffer(cmd, bits, 0, vk::WHOLE_SIZE, 0);
        c.device.cmd_fill_buffer(cmd, counters, 0, vk::WHOLE_SIZE, 0);
    })?;
    unsafe {
        let ranges = [vk::PushConstantRange { stage_flags: vk::ShaderStageFlags::COMPUTE, offset: 0, size: 40 }];
        let layout = c.device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&ranges), None).map_err(|e| format!("zero store layout: {e}"))?;
        let code = rdna::kernels::zero_store();
        let module = c.device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None).map_err(|e| format!("zero store module: {e}"))?;
        let stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(c"main");
        let ci = vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout);
        let pipeline = super::pcache::with_cache('i', 47, |cache| c.device.create_compute_pipelines(cache, &[ci], None)).map_err(|(_, e)| format!("zero store pipeline: {e}"));
        c.device.destroy_shader_module(module, None);
        Ok(Zk { bits, bits_addr, counters, counters_addr, readback, readback_ptr: rptr as usize, pipeline: pipeline?[0], layout })
    }
}

fn get() -> Option<&'static Zk> {
    ZK.get().and_then(|z| z.as_ref())
}

/// Zero stores check the bitmap (`tune zk`, made).
pub fn on() -> bool {
    super::tune::zk() && get().is_some()
}

/// The bitmap's device address for front end 2's write-back kernels (0: none; they then clear
/// nothing, and no store leaves a page out: `on` is false too).
pub fn address() -> u64 {
    get().map_or(0, |z| z.bits_addr)
}

/// The render-target pool's engine allocator (`crate::engine`), looked up once it is known.
fn rt_owner() -> Option<u8> {
    static ID: AtomicU64 = AtomicU64::new(u64::MAX);
    match ID.load(Ordering::Relaxed) {
        u64::MAX => {
            let id = crate::engine::allocators().into_iter().find(|a| a.name == "RTSharedAllocator")?.id;
            ID.store(id as u64, Ordering::Relaxed);
            Some(id)
        }
        v => Some(v as u8),
    }
}

/// A zero fill of [base, base + len) whose guest stores may be left out where known zero: front
/// end 2's span, the render-target pool's memory (every MiB), at most 1 GiB.
pub fn eligible(base: u64, len: u64) -> bool {
    if len == 0 || len > 1 << 30 || base < MIRROR_BASE || base + len > MIRROR_END || !super::fe2::on() {
        return false;
    }
    let Some(rt) = rt_owner() else { return false };
    let mut m = base >> 20;
    while m <= (base + len - 1) >> 20 {
        if crate::engine::owner(m << 20) != rt {
            return false;
        }
        m += 1;
    }
    true
}

fn page_of(a: u64) -> u64 {
    (a - MIRROR_BASE) / PAGE
}

fn tracked(w0: u64, w1: u64) -> bool {
    (w0..w1).any(|w| TRACKED[(w / 64) as usize].load(Ordering::Relaxed) & (1 << (w % 64)) != 0)
}

/// Records into `cmd` the bitmap words over pages [p0, p1) cleared (whole words: a page beside
/// them is stored again at its next fill), then a barrier before compute work reads them.
fn record_clear_pages(cmd: vk::CommandBuffer, z: &Zk, p0: u64, p1: u64) {
    let (w0, w1) = (p0 / 32, p1.div_ceil(32));
    unsafe {
        ctx().device.cmd_fill_buffer(cmd, z.bits, w0 * 4, (w1 - w0) * 4, 0);
        let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
        ctx().device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &[mb], &[], &[]);
    }
}

/// Another writer stores into guest [base, base + len) at this point of `cmd` (an in-place
/// fill, a CP write, a home's write-back, a held page's release): the pages' bits cleared in
/// GPU order where a zero store may have set them.
pub fn clear(cmd: vk::CommandBuffer, base: u64, len: u64) {
    let Some(z) = get() else { return };
    if len == 0 || base >= MIRROR_END || base.saturating_add(len) <= MIRROR_BASE {
        return;
    }
    let (lo, hi) = (base.max(MIRROR_BASE), base.saturating_add(len).min(MIRROR_END));
    let (p0, p1) = (page_of(lo), page_of(hi - 1) + 1);
    if !tracked(p0 / 32, p1.div_ceil(32)) {
        return;
    }
    STATS[3].fetch_add(1, Ordering::Relaxed);
    record_clear_pages(cmd, z, p0, p1);
}

/// Pages a write-watch take found written by the CPU (`mem::take_written`): their bits cleared
/// before the next zero store over them.
pub fn cpu_written(pages: &[u64]) {
    if pages.is_empty() || get().is_none() {
        return;
    }
    let d = dirty();
    for &a in pages {
        if (MIRROR_BASE..MIRROR_END).contains(&a) {
            let p = page_of(a);
            if TRACKED[(p / 32 / 64) as usize].load(Ordering::Relaxed) & (1 << (p / 32 % 64)) != 0 {
                d[(p / 64) as usize].fetch_or(1 << (p % 64), Ordering::Relaxed);
            }
        }
    }
}

/// Guest [base, base + len)'s mapping changed (`fe2::hash_clear_range`): its pages' bits
/// cleared before the next zero store over them.
pub fn cpu_dirty_range(base: u64, len: u64) {
    if len == 0 || get().is_none() || base >= MIRROR_END || base.saturating_add(len) <= MIRROR_BASE {
        return;
    }
    let (lo, hi) = (base.max(MIRROR_BASE), base.saturating_add(len).min(MIRROR_END));
    let d = dirty();
    for p in page_of(lo)..page_of(hi - 1) + 1 {
        d[(p / 64) as usize].fetch_or(1 << (p % 64), Ordering::Relaxed);
    }
}

/// Records into `cmd` the zero store of guest [base, base + len) (eligible, 4-byte aligned),
/// guest memory's copy of `base` at device address `dst`: the bits of pages the CPU wrote
/// meanwhile cleared first, then `rdna::kernels::zero_store` between barriers. Returns the
/// pipeline it bound (the caller's bound compute state is gone); None: not recorded (the caller
/// stores the fill itself).
pub fn record_store(cmd: vk::CommandBuffer, dst: u64, base: u64, len: u64) -> Option<vk::Pipeline> {
    let z = get()?;
    if dst == 0 || base % 4 != 0 || len % 4 != 0 || len == 0 || len > 1 << 30 {
        return None;
    }
    let (p0, p1) = (page_of(base), page_of(base + len - 1) + 1);
    // (Marked tracked before the dirty pages are read: a take finding a page written from here
    // on marks it for the next store.)
    for w in p0 / 32..p1.div_ceil(32) {
        TRACKED[(w / 64) as usize].fetch_or(1 << (w % 64), Ordering::Relaxed);
    }
    // CPU writes: pages written since their last take (a query that resets nothing) and pages
    // takes found written since the last store here (`DIRTY`).
    let mut written = Vec::new();
    crate::mem::written_pages(p0 * PAGE + MIRROR_BASE, p1 * PAGE + MIRROR_BASE, &mut written)?;
    STATS[4].fetch_add(written.len() as u64, Ordering::Relaxed);
    let d = dirty();
    let mut clear_runs: Vec<(u64, u64)> = Vec::new();
    let mut add = |p: u64| match clear_runs.last_mut() {
        Some(r) if r.1 == p => r.1 = p + 1,
        Some(r) if r.1 > p => {}
        _ => clear_runs.push((p, p + 1)),
    };
    let mut wi = 0;
    for p in p0..p1 {
        let bit = 1u64 << (p % 64);
        let from_take = d[(p / 64) as usize].fetch_and(!bit, Ordering::Relaxed) & bit != 0;
        while wi < written.len() && page_of(written[wi]) < p {
            wi += 1;
        }
        let from_query = wi < written.len() && page_of(written[wi]) == p;
        if from_take || from_query {
            add(p);
        }
    }
    for &(a, b) in &clear_runs {
        STATS[2].fetch_add(b - a, Ordering::Relaxed);
        record_clear_pages(cmd, z, a, b);
    }
    let lead = base % PAGE;
    let groups = (lead + len).div_ceil(PAGE) as u32;
    let pc: Vec<u8> = [dst as u32, (dst >> 32) as u32, z.bits_addr as u32, (z.bits_addr >> 32) as u32, z.counters_addr as u32, (z.counters_addr >> 32) as u32, p0 as u32, lead as u32, len as u32, audit() as u32]
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    let c = ctx();
    unsafe {
        let before = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::MEMORY_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
        c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &[before], &[], &[]);
        c.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, z.pipeline);
        c.device.cmd_push_constants(cmd, z.layout, vk::ShaderStageFlags::COMPUTE, 0, &pc);
        c.device.cmd_dispatch(cmd, groups, 1, 1);
        let after = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE | vk::AccessFlags::HOST_READ);
        c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::ALL_COMMANDS | vk::PipelineStageFlags::HOST, vk::DependencyFlags::empty(), &[after], &[], &[]);
        if STATS[0].fetch_add(1, Ordering::Relaxed) % COUNTER_COPY_EVERY == COUNTER_COPY_EVERY - 1 {
            let bytes = 4 * rdna::kernels::ZERO_STORE_COUNTERS as u64;
            c.device.cmd_copy_buffer(cmd, z.counters, z.readback, &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size: bytes }]);
            let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::HOST_READ);
            c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::HOST, vk::DependencyFlags::empty(), &[mb], &[], &[]);
        }
    }
    STATS[1].fetch_add(len, Ordering::Relaxed);
    Some(z.pipeline)
}

/// A zero fill left to the plain store (not eligible).
pub fn note_plain() {
    STATS[5].fetch_add(1, Ordering::Relaxed);
}

/// Status: the zero stores, what the GPU stored and left out (as of the last counter copy),
/// the clears.
pub fn status_line() -> String {
    let Some(z) = get() else { return String::new() };
    let s: Vec<u64> = STATS.iter().map(|x| x.load(Ordering::Relaxed)).collect();
    if s[0] == 0 && s[5] == 0 {
        return String::new();
    }
    let g: Vec<u32> = (0..rdna::kernels::ZERO_STORE_COUNTERS as usize).map(|i| unsafe { std::ptr::read_volatile((z.readback_ptr as *const u32).add(i)) }).collect();
    let all = (g[0] as u64 + g[1] as u64).max(1);
    format!(
        "    zero-known pages (`tune zk` {}): {} zero stores of eligible fills ({} MiB), the GPU's count as of its last copy: pages stored {}, left out as zero {} ({:.1}%){}; bits cleared before stores for CPU writes or mapping changes {} pages (CPU writes the fills' queries found {}), clears recorded for other writers {}; fills stored whole as before (not eligible) {}\n",
        super::tune::zk() as u8,
        s[0],
        s[1] >> 20,
        g[0],
        g[1],
        100.0 * g[1] as f64 / all as f64,
        if audit() { format!(", audit: non-zero dwords in pages left out {}", g[2]) } else { String::new() },
        s[2],
        s[4],
        s[3],
        s[5]
    )
}
