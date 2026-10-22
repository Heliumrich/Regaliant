//! Swapchain presentation: blits a rendered image (or uploaded pixels) to the window.
//!
//! The frame keeps its aspect (centred, black bars where the window's aspect
//! differs by 1% or more). A frame more than twice the window's size is first
//! halved by 2:1 linear blits (2x2 box filters) until the last blit reduces by
//! at most 2:1: a single linear blit samples 2x2 texels per pixel, so beyond
//! 2:1 it skips texels (a 3840x2160 frame in a 1280x720 window was point
//! sampled: aliased HUD text and edges).

use super::{ctx, window};
use ash::vk;
use parking_lot::Mutex;

const FRAMES: usize = 2;

/// An intermediate image of the halving chain.
struct Half {
    image: vk::Image,
    memory: vk::DeviceMemory,
    format: vk::Format,
    width: u32,
    height: u32,
}

struct Presenter {
    surface: vk::SurfaceKHR,
    swapchain: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    format: vk::Format,
    extent: vk::Extent2D,
    pool: vk::CommandPool,
    cmds: Vec<vk::CommandBuffer>,
    fences: Vec<vk::Fence>,
    acquire: Vec<vk::Semaphore>,
    done: Vec<vk::Semaphore>,
    frame: usize,
    staging: Option<(vk::Buffer, vk::DeviceMemory, *mut u8, u64)>,
    upload: Option<(vk::Image, vk::DeviceMemory, u32, u32)>,
    /// The halving chain's images, largest first.
    halves: Vec<Half>,
    /// The last (frame size, window size) logged.
    logged: (u32, u32, u32, u32),
}

unsafe impl Send for Presenter {}

static PRESENTER: Mutex<Option<Presenter>> = Mutex::new(None);

/// The next presented frame is read back (`shot_next`).
static SHOT_NEXT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Control channel `wshot`: the next presented frame to `window-<n>.png`.
pub fn shot_next() {
    SHOT_NEXT.store(true, std::sync::atomic::Ordering::SeqCst);
}

/// `DS_WINDOW_SHOT=<n,...>`: presented frame numbers whose swapchain image is
/// read back to `window-<n>.png` (what the window shows, after scaling).
fn window_shot(frame: usize) -> bool {
    static LIST: std::sync::OnceLock<Vec<usize>> = std::sync::OnceLock::new();
    SHOT_NEXT.swap(false, std::sync::atomic::Ordering::SeqCst)
        || LIST.get_or_init(|| std::env::var("DS_WINDOW_SHOT").map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect()).unwrap_or_default()).contains(&frame)
}

pub enum Source {
    /// An image in TRANSFER_SRC_OPTIMAL-compatible state; `layout` is its current layout.
    Image { image: vk::Image, format: vk::Format, layout: vk::ImageLayout, width: u32, height: u32 },
    /// Linear BGRA8 pixels.
    Pixels { data: Vec<u8>, width: u32, height: u32 },
}

unsafe fn create_swapchain(p: &mut Presenter) -> Result<(), String> {
    let c = ctx();
    unsafe {
        let caps = c.surface_i.get_physical_device_surface_capabilities(c.pdev, p.surface).map_err(|e| format!("surface caps: {e}"))?;
        let formats = c.surface_i.get_physical_device_surface_formats(c.pdev, p.surface).map_err(|e| format!("surface formats: {e}"))?;
        // Display buffers hold display-encoded values (the title applies its
        // own transfer function): UNORM swapchain formats show them as is,
        // 10-bit first (the title's display buffers are 10:10:10:2).
        let srgb_space = |f: &&vk::SurfaceFormatKHR| f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR;
        let fmt = formats
            .iter()
            .filter(srgb_space)
            .find(|f| f.format == vk::Format::A2B10G10R10_UNORM_PACK32)
            .or_else(|| formats.iter().filter(srgb_space).find(|f| f.format == vk::Format::B8G8R8A8_UNORM))
            .or_else(|| formats.first())
            .copied()
            .ok_or("no surface format")?;
        let mut extent = caps.current_extent;
        if extent.width == u32::MAX {
            extent = vk::Extent2D { width: window::CLIENT_W.load(std::sync::atomic::Ordering::Relaxed), height: window::CLIENT_H.load(std::sync::atomic::Ordering::Relaxed) };
        }
        if extent.width == 0 || extent.height == 0 {
            return Err("window minimized".into());
        }
        let count = (caps.min_image_count + 1).min(if caps.max_image_count == 0 { u32::MAX } else { caps.max_image_count });
        let old = p.swapchain;
        let sci = vk::SwapchainCreateInfoKHR::default()
            .surface(p.surface)
            .min_image_count(count)
            .image_format(fmt.format)
            .image_color_space(fmt.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::COLOR_ATTACHMENT | (caps.supported_usage_flags & vk::ImageUsageFlags::TRANSFER_SRC))
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(caps.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(vk::PresentModeKHR::FIFO)
            .clipped(true)
            .old_swapchain(old);
        let sc = c.swapchain_d.create_swapchain(&sci, None).map_err(|e| format!("create swapchain: {e}"))?;
        if old != vk::SwapchainKHR::null() {
            c.device.device_wait_idle().ok();
            c.swapchain_d.destroy_swapchain(old, None);
            for s in p.done.drain(..) {
                c.device.destroy_semaphore(s, None);
            }
        }
        p.swapchain = sc;
        p.images = c.swapchain_d.get_swapchain_images(sc).map_err(|e| format!("swapchain images: {e}"))?;
        p.format = fmt.format;
        p.extent = extent;
        for _ in 0..p.images.len() {
            p.done.push(c.device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None).map_err(|e| e.to_string())?);
        }
        crate::info!("swapchain {}x{} {:?} ({} images)", extent.width, extent.height, fmt.format, p.images.len());
    }
    Ok(())
}

pub fn init(title: &str) -> Result<(), String> {
    super::init()?;
    let w = window::create(title, 1280, 720)?;
    let c = ctx();
    unsafe {
        let sci = vk::Win32SurfaceCreateInfoKHR::default().hinstance(w.hinstance).hwnd(w.hwnd);
        let surface = c.win32_i.create_win32_surface(&sci, None).map_err(|e| format!("create surface: {e}"))?;
        let supported = c.surface_i.get_physical_device_surface_support(c.pdev, c.queue_family, surface).unwrap_or(false);
        if !supported {
            return Err("queue family cannot present".into());
        }
        let pool = c
            .device
            .create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(c.queue_family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER), None)
            .map_err(|e| e.to_string())?;
        let cmds = c
            .device
            .allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(pool).command_buffer_count(FRAMES as u32))
            .map_err(|e| e.to_string())?;
        let mut p = Presenter {
            surface,
            swapchain: vk::SwapchainKHR::null(),
            images: Vec::new(),
            format: vk::Format::UNDEFINED,
            extent: vk::Extent2D::default(),
            pool,
            cmds,
            fences: (0..FRAMES).map(|_| c.device.create_fence(&vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED), None).unwrap()).collect(),
            acquire: (0..FRAMES).map(|_| c.device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None).unwrap()).collect(),
            done: Vec::new(),
            frame: 0,
            staging: None,
            upload: None,
            halves: Vec::new(),
            logged: (0, 0, 0, 0),
        };
        create_swapchain(&mut p)?;
        *PRESENTER.lock() = Some(p);
    }
    Ok(())
}

fn barrier(c: &super::Ctx, cmd: vk::CommandBuffer, img: vk::Image, old: vk::ImageLayout, new: vk::ImageLayout) {
    let b = vk::ImageMemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
        .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)
        .old_layout(old)
        .new_layout(new)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(img)
        .subresource_range(vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 });
    unsafe {
        c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::ALL_COMMANDS, vk::DependencyFlags::empty(), &[], &[], &[b]);
    }
}

/// Presents one frame. Returns false if presentation is not possible (minimized).
pub fn present(src: Source) -> bool {
    let c = ctx();
    let mut g = PRESENTER.lock();
    let Some(p) = g.as_mut() else { return false };
    unsafe {
        if window::RESIZED.swap(false, std::sync::atomic::Ordering::SeqCst) {
            if let Err(e) = create_swapchain(p) {
                crate::debug!("swapchain recreate: {e}");
                return false;
            }
        }
        let f = p.frame % FRAMES;
        p.frame += 1;
        c.device.wait_for_fences(&[p.fences[f]], true, u64::MAX).ok();
        let (idx, _) = match c.swapchain_d.acquire_next_image(p.swapchain, u64::MAX, p.acquire[f], vk::Fence::null()) {
            Ok(r) => r,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                window::RESIZED.store(true, std::sync::atomic::Ordering::SeqCst);
                return false;
            }
            Err(e) => {
                crate::error!("acquire: {e}");
                return false;
            }
        };
        c.device.reset_fences(&[p.fences[f]]).ok();
        let cmd = p.cmds[f];
        c.device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty()).ok();
        c.device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).ok();
        let dst = p.images[idx as usize];
        barrier(c, cmd, dst, vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
        let (src_img, src_format, src_layout, sw, sh) = match src {
            Source::Image { image, format, layout, width, height } => {
                // GENERAL images are blitted in place (after a barrier that
                // makes earlier GPU writes visible).
                let to = if layout == vk::ImageLayout::GENERAL { layout } else { vk::ImageLayout::TRANSFER_SRC_OPTIMAL };
                barrier(c, cmd, image, layout, to);
                (image, format, layout, width, height)
            }
            Source::Pixels { data, width, height } => match upload_pixels(p, cmd, &data, width, height) {
                Ok(img) => (img, vk::Format::B8G8R8A8_SRGB, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, width, height),
                Err(e) => {
                    crate::error!("present upload: {e}");
                    c.device.end_command_buffer(cmd).ok();
                    return false;
                }
            },
        };
        let blit_layout = if src_layout == vk::ImageLayout::GENERAL { src_layout } else { vk::ImageLayout::TRANSFER_SRC_OPTIMAL };
        let fitted = crate::gpu::tune::present_fit();
        let (rx, ry, rw, rh) = if fitted { fit(sw, sh, p.extent.width, p.extent.height) } else { (0, 0, p.extent.width, p.extent.height) };
        if (rw, rh) != (p.extent.width, p.extent.height) {
            let black = vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 1.0] };
            let range = vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 };
            c.device.cmd_clear_color_image(cmd, dst, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &black, &[range]);
            barrier(c, cmd, dst, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
        }
        // Halvings while the frame is more than twice the destination's size.
        let (mut from, mut from_layout, mut fw, mut fh) = (src_img, blit_layout, sw, sh);
        let mut level = 0;
        while fitted && (fw > 2 * rw || fh > 2 * rh) {
            let (hw, hh) = (fw.div_ceil(2), fh.div_ceil(2));
            let half = match half_image(p, f, level, src_format, hw, hh) {
                Ok(h) => h,
                Err(e) => {
                    static ONCE: std::sync::Once = std::sync::Once::new();
                    ONCE.call_once(|| crate::warn!("present: no {hw}x{hh} halving image ({e}): one blit"));
                    break;
                }
            };
            barrier(c, cmd, half, vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
            blit(c, cmd, from, from_layout, (0, 0, fw, fh), half, (0, 0, hw, hh));
            barrier(c, cmd, half, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
            (from, from_layout, fw, fh) = (half, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, hw, hh);
            level += 1;
        }
        blit(c, cmd, from, from_layout, (0, 0, fw, fh), dst, (rx, ry, rw, rh));
        if p.logged != (sw, sh, p.extent.width, p.extent.height) {
            p.logged = (sw, sh, p.extent.width, p.extent.height);
            crate::info!("present: {sw}x{sh} frame into a {}x{} window at {rw}x{rh}+{rx}+{ry}: {level} halving(s), then {fw}x{fh} -> {rw}x{rh}", p.extent.width, p.extent.height);
        }
        if src_layout != vk::ImageLayout::TRANSFER_SRC_OPTIMAL && src_layout != vk::ImageLayout::GENERAL && src_img != vk::Image::null() {
            barrier(c, cmd, src_img, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, src_layout);
        }
        // Window shot: copy the swapchain image to a host buffer.
        let shot = if window_shot(p.frame) {
            let (w, h) = (p.extent.width, p.extent.height);
            match c.host_buffer(w as u64 * h as u64 * 4, vk::BufferUsageFlags::TRANSFER_DST) {
                Ok((buf, mem, ptr)) => {
                    barrier(c, cmd, dst, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
                    let region = vk::BufferImageCopy::default()
                        .image_subresource(vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 })
                        .image_extent(vk::Extent3D { width: w, height: h, depth: 1 });
                    c.device.cmd_copy_image_to_buffer(cmd, dst, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, buf, &[region]);
                    barrier(c, cmd, dst, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, vk::ImageLayout::PRESENT_SRC_KHR);
                    Some((buf, mem, ptr, w, h))
                }
                Err(e) => {
                    crate::warn!("window shot: {e}");
                    barrier(c, cmd, dst, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::PRESENT_SRC_KHR);
                    None
                }
            }
        } else {
            barrier(c, cmd, dst, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::PRESENT_SRC_KHR);
            None
        };
        c.device.end_command_buffer(cmd).ok();
        let done = p.done[idx as usize];
        if let Err(e) = c.submit(cmd, &[(p.acquire[f], vk::PipelineStageFlags::TRANSFER)], &[done], p.fences[f]) {
            crate::error!("{e}");
            return false;
        }
        if let Some((buf, mem, ptr, w, h)) = shot {
            c.device.wait_for_fences(&[p.fences[f]], true, 5_000_000_000).ok();
            let raw = std::slice::from_raw_parts(ptr, (w * h * 4) as usize).to_vec();
            if let Some(rgba) = crate::gpu::images::to_rgba8(p.format, &raw) {
                if let Some(dir) = crate::run_dir() {
                    let path = dir.join(format!("window-{}.png", p.frame));
                    if std::fs::write(&path, crate::gpu::png::encode(w, h, &rgba)).is_ok() {
                        crate::info!("window shot of presented frame {}: {}", p.frame, path.display());
                    }
                }
            }
            c.device.destroy_buffer(buf, None);
            c.device.unmap_memory(mem);
            c.device.free_memory(mem, None);
        }
        let scs = [p.swapchain];
        let idxs = [idx];
        let waits = [done];
        let pi = vk::PresentInfoKHR::default().wait_semaphores(&waits).swapchains(&scs).image_indices(&idxs);
        let q = c.queue.lock();
        match c.swapchain_d.queue_present(*q, &pi) {
            Ok(false) => {}
            Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => window::RESIZED.store(true, std::sync::atomic::Ordering::SeqCst),
            Err(e) => crate::error!("present: {e}"),
        }
    }
    true
}

/// Where a `sw`x`sh` frame goes in a `dw`x`dh` window (x, y, width, height):
/// as large as fits with its aspect kept, centred; the whole window when the
/// aspects differ by under 1%.
fn fit(sw: u32, sh: u32, dw: u32, dh: u32) -> (u32, u32, u32, u32) {
    let (a, b) = (sw as u64 * dh as u64, dw as u64 * sh as u64);
    if sw == 0 || sh == 0 || a.abs_diff(b) * 100 < a.max(b) {
        return (0, 0, dw, dh);
    }
    if a > b {
        // Wider than the window: bars above and below.
        let h = ((dw as u64 * sh as u64 + sw as u64 / 2) / sw as u64).clamp(1, dh as u64) as u32;
        (0, (dh - h) / 2, dw, h)
    } else {
        let w = ((dh as u64 * sw as u64 + sh as u64 / 2) / sh as u64).clamp(1, dw as u64) as u32;
        ((dw - w) / 2, 0, w, dh)
    }
}

/// A linear blit of rectangle `s` (x, y, width, height) of `src` to rectangle
/// `d` of `dst` (in TRANSFER_DST_OPTIMAL).
fn blit(c: &super::Ctx, cmd: vk::CommandBuffer, src: vk::Image, src_layout: vk::ImageLayout, s: (u32, u32, u32, u32), dst: vk::Image, d: (u32, u32, u32, u32)) {
    let layers = vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 };
    let corners = |(x, y, w, h): (u32, u32, u32, u32)| [vk::Offset3D { x: x as i32, y: y as i32, z: 0 }, vk::Offset3D { x: (x + w) as i32, y: (y + h) as i32, z: 1 }];
    let region = vk::ImageBlit { src_subresource: layers, src_offsets: corners(s), dst_subresource: layers, dst_offsets: corners(d) };
    unsafe { c.device.cmd_blit_image(cmd, src, src_layout, dst, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[region], vk::Filter::LINEAR) };
}

/// The halving chain's image `level` at `w`x`h`: the frame's format when it
/// can be blitted to and from, else 10:10:10:2. Made or remade on demand
/// (after the presenter's other frames in flight, the only other users, are
/// done; `current` is the frame being recorded).
unsafe fn half_image(p: &mut Presenter, current: usize, level: usize, format: vk::Format, w: u32, h: u32) -> Result<vk::Image, String> {
    let c = ctx();
    let blittable = |f: vk::Format| {
        let props = unsafe { c.instance.get_physical_device_format_properties(c.pdev, f) };
        props.optimal_tiling_features.contains(vk::FormatFeatureFlags::BLIT_SRC | vk::FormatFeatureFlags::BLIT_DST | vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR)
    };
    let format = if blittable(format) { format } else { vk::Format::A2B10G10R10_UNORM_PACK32 };
    if let Some(h_) = p.halves.get(level) {
        if (h_.format, h_.width, h_.height) == (format, w, h) {
            return Ok(h_.image);
        }
    }
    unsafe {
        let others: Vec<vk::Fence> = p.fences.iter().enumerate().filter(|&(i, _)| i != current).map(|(_, &f)| f).collect();
        c.device.wait_for_fences(&others, true, u64::MAX).ok();
        for old in p.halves.drain(level..) {
            c.device.destroy_image(old.image, None);
            c.device.free_memory(old.memory, None);
        }
        let ici = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D { width: w, height: h, depth: 1 })
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
        p.halves.push(Half { image, memory, format, width: w, height: h });
        Ok(image)
    }
}

unsafe fn upload_pixels(p: &mut Presenter, cmd: vk::CommandBuffer, data: &[u8], w: u32, h: u32) -> Result<vk::Image, String> {
    let c = ctx();
    let size = (w * h * 4) as u64;
    unsafe {
        if p.staging.as_ref().is_none_or(|s| s.3 < size) {
            let (b, m, ptr) = c.host_buffer(size, vk::BufferUsageFlags::TRANSFER_SRC)?;
            p.staging = Some((b, m, ptr, size));
        }
        if p.upload.as_ref().is_none_or(|u| u.2 != w || u.3 != h) {
            let ici = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(vk::Format::B8G8R8A8_SRGB)
                .extent(vk::Extent3D { width: w, height: h, depth: 1 })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC);
            let img = c.device.create_image(&ici, None).map_err(|e| e.to_string())?;
            let mem = c.alloc(c.device.get_image_memory_requirements(img), vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
            c.device.bind_image_memory(img, mem, 0).map_err(|e| e.to_string())?;
            p.upload = Some((img, mem, w, h));
        }
        let (sb, _, sp, _) = p.staging.unwrap();
        std::ptr::copy_nonoverlapping(data.as_ptr(), sp, size as usize);
        let img = p.upload.unwrap().0;
        barrier(c, cmd, img, vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
        let region = vk::BufferImageCopy {
            buffer_offset: 0,
            buffer_row_length: 0,
            buffer_image_height: 0,
            image_subresource: vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 },
            image_offset: vk::Offset3D::default(),
            image_extent: vk::Extent3D { width: w, height: h, depth: 1 },
        };
        c.device.cmd_copy_buffer_to_image(cmd, sb, img, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[region]);
        barrier(c, cmd, img, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
        Ok(img)
    }
}

#[cfg(test)]
mod tests {
    use super::fit;

    #[test]
    fn fit_keeps_the_frame_shape() {
        assert_eq!(fit(3840, 2160, 1280, 720), (0, 0, 1280, 720));
        assert_eq!(fit(3840, 2160, 1280, 1024), (0, 152, 1280, 720));
        assert_eq!(fit(3840, 2160, 1600, 600), (266, 0, 1067, 600));
        // Under 1% off: the whole window, no 1-pixel bars.
        assert_eq!(fit(3840, 2160, 1366, 768), (0, 0, 1366, 768));
        assert_eq!(fit(0, 0, 800, 600), (0, 0, 800, 600));
    }
}
