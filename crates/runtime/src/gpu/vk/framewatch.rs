//! Frame watch (`DS_FRAME_WATCH=1`, control `framewatch 1|0`): each presented frame copied to a
//! 128x72 image (one linear blit of the frame as shown, before the window's scaling) and read
//! back when its presenter slot comes round again; its 16x9 grid of mean brightness (8x8 pixels
//! a cell) is checked against the frames before and after it. A frame whose cells jump against
//! both neighbours while the neighbours agree is a flash: a light or the whole picture wrong for
//! one frame (the user's videos, 2026-11-13: whole rooms lit as by daylight for one frame,
//! light pools at the player's feet blinking). Flashes are logged; every frame's grid goes to
//! `<run dir>/frames-luma.bin` (f32 log time in ms, then 144 cells, row by row) for analysis.
//! `DS_FRAME_WATCH=2` also keeps the last 8 frames at 480x270 and saves the frames before, at
//! and after a flash of 8 cells or more as `flash_<ms>_{a,b,c}.png` (at most 60, or
//! `DS_FRAME_WATCH_SAVES`; from the log time `DS_FRAME_WATCH_FROM` in ms on, when given).

use ash::vk;
use std::collections::VecDeque;
use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};

pub const W: u32 = 128;
pub const H: u32 = 72;
const CELLS_X: usize = 16;
const CELLS_Y: usize = 9;
pub const CELLS: usize = CELLS_X * CELLS_Y;
/// A cell flashes when it differs from both neighbours by this much in the same direction (0-255
/// display values) while they differ from each other by at most `AGREE`.
const JUMP: i32 = 20;
const AGREE: i32 = 10;
/// A frame with this many flashing cells is logged.
const MIN_CELLS: usize = 2;

pub static ON: AtomicBool = AtomicBool::new(false);
/// The flip count of the frame being presented (`videoout`'s, set before each present): each
/// recorded frame's goes to `<run dir>/frames-flip.bin` (u64 per frame, in frames-luma.bin's order).
pub static SHOWN_FLIP: AtomicU64 = AtomicU64::new(0);
/// Large flashes' frames saved (`DS_FRAME_WATCH=2`).
pub static SAVE: AtomicBool = AtomicBool::new(false);
const RING: usize = 8;
const RW: u32 = 480;
const RH: u32 = 270;
/// A flash this large saves its frames.
const SAVE_CELLS: usize = 8;
fn save_max() -> u32 {
    static N: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *N.get_or_init(|| std::env::var("DS_FRAME_WATCH_SAVES").ok().and_then(|v| v.parse().ok()).unwrap_or(60))
}

fn save_from() -> f64 {
    static T: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *T.get_or_init(|| std::env::var("DS_FRAME_WATCH_FROM").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0))
}
/// Frames watched, flash frames, flash frames with a quarter of the picture or more.
pub static STATS: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// `DS_FRAME_WATCH=1` turns it on at start.
pub fn init_from_env() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(init_env);
}

fn init_env() {
    match std::env::var("DS_FRAME_WATCH").as_deref() {
        Ok("1") => ON.store(true, Relaxed),
        Ok("2") => {
            ON.store(true, Relaxed);
            SAVE.store(true, Relaxed);
        }
        _ => {}
    }
}

pub struct Watch {
    image: vk::Image,
    memory: vk::DeviceMemory,
    /// One read-back buffer per presenter slot, and the log time of the frame recorded into it.
    bufs: Vec<(vk::Buffer, vk::DeviceMemory, *mut u8)>,
    pending: Vec<Option<f64>>,
    pending_flip: Vec<u64>,
    flip_file: Option<std::fs::File>,
    grids: VecDeque<(f64, [u8; CELLS])>,
    file: Option<std::fs::File>,
    logged: u32,
    /// The last frames at RWxRH (`SAVE`): images, their memory, the log time each holds.
    ring: Vec<(vk::Image, vk::DeviceMemory)>,
    ring_t: Vec<f64>,
    ring_next: usize,
    saved: u32,
    /// The saves' command buffer and fence (its own pool), and host buffer for three frames.
    pool: Option<vk::CommandPool>,
    one_shot: Option<(vk::CommandBuffer, vk::Fence)>,
    ring_buf: Option<(vk::Buffer, vk::DeviceMemory, *mut u8)>,
}

unsafe impl Send for Watch {}

impl Watch {
    pub fn new(slots: usize) -> Result<Watch, String> {
        let c = super::ctx();
        unsafe {
            let ici = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(vk::Format::R8G8B8A8_UNORM)
                .extent(vk::Extent3D { width: W, height: H, depth: 1 })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC);
            let image = c.device.create_image(&ici, None).map_err(|e| e.to_string())?;
            let memory = match c.alloc(c.device.get_image_memory_requirements(image), vk::MemoryPropertyFlags::DEVICE_LOCAL) {
                Ok(m) => m,
                Err(e) => {
                    c.device.destroy_image(image, None);
                    return Err(e);
                }
            };
            c.device.bind_image_memory(image, memory, 0).map_err(|e| e.to_string())?;
            let mut bufs = Vec::new();
            for _ in 0..slots {
                bufs.push(c.host_buffer((W * H * 4) as u64, vk::BufferUsageFlags::TRANSFER_DST)?);
            }
            let file = crate::run_dir().and_then(|d| std::fs::File::create(d.join("frames-luma.bin")).ok());
            let flip_file = crate::run_dir().and_then(|d| std::fs::File::create(d.join("frames-flip.bin")).ok());
            let mut ring = Vec::new();
            let (mut pool, mut one_shot, mut ring_buf) = (None, None, None);
            if SAVE.load(Relaxed) {
                let p = c.device.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(c.queue_family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER), None).map_err(|e| e.to_string())?;
                let cmd = c.device.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(p).command_buffer_count(1)).map_err(|e| e.to_string())?[0];
                let fence = c.device.create_fence(&vk::FenceCreateInfo::default(), None).map_err(|e| e.to_string())?;
                pool = Some(p);
                one_shot = Some((cmd, fence));
                ring_buf = Some(c.host_buffer(3 * (RW * RH * 4) as u64, vk::BufferUsageFlags::TRANSFER_DST)?);
                for _ in 0..RING {
                    let ici = ici.extent(vk::Extent3D { width: RW, height: RH, depth: 1 });
                    let img = c.device.create_image(&ici, None).map_err(|e| e.to_string())?;
                    let mem = c.alloc(c.device.get_image_memory_requirements(img), vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
                    c.device.bind_image_memory(img, mem, 0).map_err(|e| e.to_string())?;
                    ring.push((img, mem));
                }
            }
            crate::info!("frame watch: on (each presented frame's 16x9 brightness grid; one-frame flashes logged; frames-luma.bin{})", if ring.is_empty() { "" } else { "; large flashes' frames saved" });
            Ok(Watch { image, memory, bufs, pending: vec![None; slots], pending_flip: vec![0; slots], flip_file, grids: VecDeque::new(), file, logged: 0, ring_t: vec![-1.0; ring.len()], ring, ring_next: 0, saved: 0, pool, one_shot, ring_buf })
        }
    }

    /// Records the copy of `src` (in `layout`, `w`x`h`) for presenter slot `slot`.
    ///
    /// # Safety
    /// `cmd` is recording; `src` is valid in `layout` until the command buffer completes.
    pub unsafe fn record(&mut self, cmd: vk::CommandBuffer, slot: usize, src: vk::Image, layout: vk::ImageLayout, w: u32, h: u32) {
        let c = super::ctx();
        let range = vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 };
        let to_dst = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_READ)
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(self.image)
            .subresource_range(range);
        let to_src = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(self.image)
            .subresource_range(range);
        let layers = vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 };
        let region = vk::ImageBlit {
            src_subresource: layers,
            src_offsets: [vk::Offset3D { x: 0, y: 0, z: 0 }, vk::Offset3D { x: w as i32, y: h as i32, z: 1 }],
            dst_subresource: layers,
            dst_offsets: [vk::Offset3D { x: 0, y: 0, z: 0 }, vk::Offset3D { x: W as i32, y: H as i32, z: 1 }],
        };
        let copy = vk::BufferImageCopy::default().image_subresource(layers).image_extent(vk::Extent3D { width: W, height: H, depth: 1 });
        let host = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::HOST_READ);
        unsafe {
            c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &[to_dst]);
            c.device.cmd_blit_image(cmd, src, layout, self.image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[region], vk::Filter::LINEAR);
            c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &[to_src]);
            c.device.cmd_copy_image_to_buffer(cmd, self.image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, self.bufs[slot].0, &[copy]);
            c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::HOST, vk::DependencyFlags::empty(), &[host], &[], &[]);
        }
        let t = crate::log::elapsed_ms();
        if !self.ring.is_empty() {
            let i = self.ring_next % self.ring.len();
            self.ring_next += 1;
            let img = self.ring[i].0;
            let rd = vk::ImageMemoryBarrier::default().image(img).subresource_range(range);
            let to_dst = rd
                .src_access_mask(vk::AccessFlags::TRANSFER_READ)
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED);
            let to_src = rd
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED);
            let region = vk::ImageBlit { dst_offsets: [vk::Offset3D { x: 0, y: 0, z: 0 }, vk::Offset3D { x: RW as i32, y: RH as i32, z: 1 }], ..region };
            unsafe {
                c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &[to_dst]);
                c.device.cmd_blit_image(cmd, src, layout, img, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[region], vk::Filter::LINEAR);
                c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &[to_src]);
            }
            self.ring_t[i] = t;
        }
        self.pending[slot] = Some(t);
        self.pending_flip[slot] = SHOWN_FLIP.load(Relaxed);
    }

    /// Saves the kept frames at the log times `times` (`SAVE`) as `flash_<ms>_<tag>.png`: copied
    /// to a host buffer with the watch's own command buffer (their frames' command buffers are
    /// done: collected frames are), written on a thread.
    fn save(&mut self, at: f64, times: [(f64, char); 3]) {
        let c = super::ctx();
        let mut found = Vec::new();
        for (t, tag) in times {
            if let Some(i) = self.ring_t.iter().position(|&x| x == t) {
                found.push((self.ring[i].0, tag));
            }
        }
        let (Some((cmd, fence)), Some(&(buf, _, ptr)), Some(dir)) = (self.one_shot, self.ring_buf.as_ref(), crate::run_dir()) else { return };
        if found.len() < 3 {
            return;
        }
        let bytes = (RW * RH * 4) as u64;
        unsafe {
            if c.device.reset_fences(&[fence]).is_err() || c.device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty()).is_err() {
                return;
            }
            c.device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).ok();
            let layers = vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 };
            for (k, (img, _)) in found.iter().enumerate() {
                let copy = vk::BufferImageCopy::default().buffer_offset(k as u64 * bytes).image_subresource(layers).image_extent(vk::Extent3D { width: RW, height: RH, depth: 1 });
                c.device.cmd_copy_image_to_buffer(cmd, *img, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, buf, &[copy]);
            }
            let host = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::HOST_READ);
            c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::HOST, vk::DependencyFlags::empty(), &[host], &[], &[]);
            c.device.end_command_buffer(cmd).ok();
            if c.submit(cmd, &[], &[], fence).is_err() || c.device.wait_for_fences(&[fence], true, 2_000_000_000).is_err() {
                crate::warn!("frame watch: the kept frames could not be read back");
                return;
            }
            for (k, (_, tag)) in found.iter().enumerate() {
                let px = std::slice::from_raw_parts(ptr.add((k as u64 * bytes) as usize), bytes as usize).to_vec();
                let path = dir.join(format!("flash_{:.0}_{tag}.png", at));
                let _ = std::thread::Builder::new().name("flash-shot".into()).spawn(move || {
                    let _ = std::fs::write(&path, crate::gpu::png::encode(RW, RH, &px));
                });
            }
        }
        self.saved += 1;
    }

    /// Reads slot `slot`'s copy (its fence waited for) and checks the frame before it.
    pub fn collect(&mut self, slot: usize) {
        let Some(t) = self.pending[slot].take() else { return };
        let px = unsafe { std::slice::from_raw_parts(self.bufs[slot].2, (W * H * 4) as usize) };
        let mut grid = [0u8; CELLS];
        for cy in 0..CELLS_Y {
            for cx in 0..CELLS_X {
                let mut sum = 0u32;
                for y in cy * 8..cy * 8 + 8 {
                    for x in cx * 8..cx * 8 + 8 {
                        let i = (y * W as usize + x) * 4;
                        // Rec. 709 weights on the display values.
                        sum += (px[i] as u32 * 54 + px[i + 1] as u32 * 183 + px[i + 2] as u32 * 19) >> 8;
                    }
                }
                grid[cy * CELLS_X + cx] = (sum / 64) as u8;
            }
        }
        STATS[0].fetch_add(1, Relaxed);
        super::super::exec::outlier_note(t as f64, self.pending_flip[slot] as u64);
        if let Some(f) = self.file.as_mut() {
            let mut rec = Vec::with_capacity(4 + CELLS);
            rec.extend_from_slice(&(t as f32).to_le_bytes());
            rec.extend_from_slice(&grid);
            let _ = f.write_all(&rec);
        }
        if let Some(f) = self.flip_file.as_mut() {
            let _ = f.write_all(&self.pending_flip[slot].to_le_bytes());
        }
        self.grids.push_back((t, grid));
        if self.grids.len() > 3 {
            self.grids.pop_front();
        }
        if self.grids.len() == 3 {
            let (a, (tm, m), b) = (self.grids[0].1, self.grids[1], self.grids[2].1);
            let mut n = 0;
            let mut sign = 0i32;
            let mut worst = 0i32;
            for k in 0..CELLS {
                let (p, q, r) = (a[k] as i32, m[k] as i32, b[k] as i32);
                let (d1, d2) = (q - p, q - r);
                if d1.signum() == d2.signum() && d1.abs() >= JUMP && d2.abs() >= JUMP && (p - r).abs() <= AGREE {
                    n += 1;
                    sign += d1.signum();
                    worst = worst.max(d1.abs().min(d2.abs()));
                }
            }
            if n >= MIN_CELLS {
                STATS[1].fetch_add(1, Relaxed);
                if n >= CELLS / 4 {
                    STATS[2].fetch_add(1, Relaxed);
                }
                if n >= 20 {
                    super::super::drawring::dump(tm);
                    let l = super::super::exec::worker_progs_line();
                    if !l.is_empty() {
                        crate::info!("{}", l.trim_end());
                    }
                }
                if n >= SAVE_CELLS && !self.ring.is_empty() && self.saved < save_max() && tm >= save_from() {
                    let times = [(self.grids[0].0, 'a'), (tm, 'b'), (self.grids[2].0, 'c')];
                    self.save(tm, times);
                }
                if self.logged < 300 {
                    self.logged += 1;
                    crate::info!(
                        "frame watch: a {} flash at {:.0} ms: {n} of {CELLS} cells for one frame (up to {worst} levels against both neighbours)",
                        if sign >= 0 { "bright" } else { "dark" },
                        tm
                    );
                }
            }
        }
    }

    pub fn destroy(self) {
        let c = super::ctx();
        unsafe {
            for (b, m, _) in self.bufs {
                c.device.destroy_buffer(b, None);
                c.device.unmap_memory(m);
                c.device.free_memory(m, None);
            }
            c.device.destroy_image(self.image, None);
            c.device.free_memory(self.memory, None);
            for (i, m) in self.ring {
                c.device.destroy_image(i, None);
                c.device.free_memory(m, None);
            }
            if let Some((_, f)) = self.one_shot {
                c.device.destroy_fence(f, None);
            }
            if let Some(p) = self.pool {
                c.device.destroy_command_pool(p, None);
            }
            if let Some((b, m, _)) = self.ring_buf {
                c.device.destroy_buffer(b, None);
                c.device.unmap_memory(m);
                c.device.free_memory(m, None);
            }
        }
    }
}

/// The watch without a window (headless runs): its own command buffers and fences, one per
/// slot, on the main queue after the frame's work (the display buffer `image`, in `layout`).
struct Headless {
    watch: Watch,
    pool: vk::CommandPool,
    cmds: Vec<vk::CommandBuffer>,
    fences: Vec<vk::Fence>,
    n: usize,
}

unsafe impl Send for Headless {}

static HEADLESS: parking_lot::Mutex<Option<Headless>> = parking_lot::Mutex::new(None);
static HEADLESS_FAILED: AtomicBool = AtomicBool::new(false);

/// A flipped frame while no presenter runs (`present::present` without a window).
pub fn headless(image: vk::Image, layout: vk::ImageLayout, w: u32, h: u32) {
    if !ON.load(Relaxed) || HEADLESS_FAILED.load(Relaxed) {
        return;
    }
    let c = super::ctx();
    let mut g = HEADLESS.lock();
    if g.is_none() {
        let made = (|| -> Result<Headless, String> {
            unsafe {
                let pool = c
                    .device
                    .create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(c.queue_family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER), None)
                    .map_err(|e| e.to_string())?;
                let cmds = c.device.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(pool).command_buffer_count(2)).map_err(|e| e.to_string())?;
                let mut fences = Vec::new();
                for _ in 0..2 {
                    fences.push(c.device.create_fence(&vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED), None).map_err(|e| e.to_string())?);
                }
                Ok(Headless { watch: Watch::new(2)?, pool, cmds, fences, n: 0 })
            }
        })();
        match made {
            Ok(hd) => {
                crate::info!("frame watch: without a window (its own command buffers after each flipped frame)");
                *g = Some(hd);
            }
            Err(e) => {
                crate::warn!("frame watch: unavailable without a window ({e})");
                HEADLESS_FAILED.store(true, Relaxed);
                return;
            }
        }
    }
    let hd = g.as_mut().unwrap();
    let slot = hd.n % 2;
    hd.n += 1;
    unsafe {
        if c.device.wait_for_fences(&[hd.fences[slot]], true, 2_000_000_000).is_err() {
            return;
        }
        hd.watch.collect(slot);
        let cmd = hd.cmds[slot];
        if c.device.reset_fences(&[hd.fences[slot]]).is_err() || c.device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty()).is_err() {
            return;
        }
        c.device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).ok();
        // The frame's writes (earlier on this queue) before the copy reads it.
        let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::MEMORY_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_READ);
        c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[mb], &[], &[]);
        hd.watch.record(cmd, slot, image, layout, w, h);
        c.device.end_command_buffer(cmd).ok();
        if c.submit(cmd, &[], &[], hd.fences[slot]).is_err() {
            HEADLESS_FAILED.store(true, Relaxed);
        }
    }
    let _ = hd.pool;
}

/// Status line.
pub fn line() -> String {
    let n = STATS[0].load(Relaxed);
    if n == 0 {
        return String::new();
    }
    format!("    frame watch: {n} frames watched, {} one-frame flashes ({} over a quarter of the picture)\n", STATS[1].load(Relaxed), STATS[2].load(Relaxed))
}
