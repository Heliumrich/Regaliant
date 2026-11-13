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
    /// The PC settings' sharpening pass (made on first use; `sharpen_failed` after an error).
    sharpen: Option<super::sharpen::Sharpen>,
    sharpen_failed: bool,
    /// The swapchain holds exclusive fullscreen (VK_EXT_full_screen_exclusive, application
    /// controlled), on this monitor.
    exclusive: Option<isize>,
    /// The swapchain is HDR10 (ST 2084 PQ, BT.2020): the title's display buffers hold HDR.
    hdr: bool,
    /// The frames shown are HDR (the title's display buffers' format; the next swapchain's).
    want_hdr: bool,
    /// The frame watch (`framewatch`: one-frame flashes), made on first use.
    watch: Option<super::framewatch::Watch>,
    watch_failed: bool,
    /// The system text box drawn over the frame (`crate::textbox`), made on first use.
    text_box: Option<TextBox>,
}

/// The system text box's image (BGRA8 UNORM), its staging buffers (one per presenter slot) and
/// the content it holds (the dialog's generation, the frame height and HDR it was drawn for).
struct TextBox {
    image: vk::Image,
    memory: vk::DeviceMemory,
    w: u32,
    h: u32,
    staging: Vec<(vk::Buffer, vk::DeviceMemory, *mut u8)>,
    key: Option<(u64, u32, bool)>,
}

impl TextBox {
    unsafe fn destroy(self) {
        let c = ctx();
        unsafe {
            for (b, m, _) in self.staging {
                c.device.destroy_buffer(b, None);
                c.device.unmap_memory(m);
                c.device.free_memory(m, None);
            }
            c.device.destroy_image(self.image, None);
            c.device.free_memory(self.memory, None);
        }
    }
}

/// Draws the system text box (the character's name: `hle::libs::ime_dialog`) over the frame in
/// `dst` (TRANSFER_DST_OPTIMAL), centred across the picture at 38% of its height.
unsafe fn draw_text_box(p: &mut Presenter, cmd: vk::CommandBuffer, slot: usize, dst: vk::Image, picture: (u32, u32, u32, u32)) {
    let Some((generation, content)) = crate::hle::libs::ime_dialog::overlay() else { return };
    let c = ctx();
    let (rx, ry, rw, rh) = picture;
    let (w, h) = crate::textbox::size(rh);
    if w > rw || h > rh {
        return;
    }
    unsafe {
        if p.text_box.as_ref().is_none_or(|t| (t.w, t.h) != (w, h)) {
            // (The other slot may still read the old one.)
            let others: Vec<vk::Fence> = p.fences.iter().enumerate().filter(|&(i, _)| i != slot).map(|(_, &f)| f).collect();
            c.device.wait_for_fences(&others, true, u64::MAX).ok();
            if let Some(old) = p.text_box.take() {
                old.destroy();
            }
            let ici = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(vk::Format::B8G8R8A8_UNORM)
                .extent(vk::Extent3D { width: w, height: h, depth: 1 })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC);
            let Ok(image) = c.device.create_image(&ici, None) else { return };
            let Ok(memory) = c.alloc(c.device.get_image_memory_requirements(image), vk::MemoryPropertyFlags::DEVICE_LOCAL) else {
                c.device.destroy_image(image, None);
                return;
            };
            if c.device.bind_image_memory(image, memory, 0).is_err() {
                c.device.destroy_image(image, None);
                c.device.free_memory(memory, None);
                return;
            }
            let mut staging = Vec::new();
            for _ in 0..FRAMES {
                match c.host_buffer((w * h * 4) as u64, vk::BufferUsageFlags::TRANSFER_SRC) {
                    Ok(b) => staging.push(b),
                    Err(e) => {
                        crate::warn!("text box: no staging buffer ({e})");
                        TextBox { image, memory, w, h, staging, key: None }.destroy();
                        return;
                    }
                }
            }
            p.text_box = Some(TextBox { image, memory, w, h, staging, key: None });
        }
        let hdr = p.hdr;
        let t = p.text_box.as_mut().unwrap();
        let key = (generation, rh, hdr);
        if t.key != Some(key) {
            let (mut px, _, _) = crate::textbox::render(&content, rh);
            if hdr {
                crate::textbox::to_pq(&mut px, 203.0);
            }
            std::ptr::copy_nonoverlapping(px.as_ptr(), t.staging[slot].2, px.len());
            let range = vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 };
            let to_dst = vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_READ)
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(t.image)
                .subresource_range(range);
            c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &[to_dst]);
            let copy = vk::BufferImageCopy::default()
                .image_subresource(vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 })
                .image_extent(vk::Extent3D { width: w, height: h, depth: 1 });
            c.device.cmd_copy_buffer_to_image(cmd, t.staging[slot].0, t.image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[copy]);
            let to_src = vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(t.image)
                .subresource_range(range);
            c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &[to_src]);
            t.key = Some(key);
        }
        // Over the frame just blitted to `dst`.
        barrier(c, cmd, dst, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
        let x = rx + (rw - w) / 2;
        let y = (ry + rh * 38 / 100).saturating_sub(h / 2).max(ry);
        blit(c, cmd, t.image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, (0, 0, w, h), dst, (x, y, w, h));
    }
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
    /// An image in TRANSFER_SRC_OPTIMAL-compatible state; `layout` is its current layout; `hdr`:
    /// its values are HDR10 (PQ-encoded BT.2020, the title's HDR output).
    Image { image: vk::Image, format: vk::Format, layout: vk::ImageLayout, width: u32, height: u32, hdr: bool },
    /// Linear BGRA8 pixels.
    Pixels { data: Vec<u8>, width: u32, height: u32 },
}

unsafe fn create_swapchain(p: &mut Presenter) -> Result<(), String> {
    let c = ctx();
    unsafe {
        let caps = c.surface_i.get_physical_device_surface_capabilities(c.pdev, p.surface).map_err(|e| format!("surface caps: {e}"))?;
        let formats = c.surface_i.get_physical_device_surface_formats(c.pdev, p.surface).map_err(|e| format!("surface formats: {e}"))?;
        {
            // (Once: the formats and color spaces the display offers; HDR10's appear while
            // Windows' HDR is on for it.)
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                let list: Vec<String> = formats.iter().map(|f| format!("{:?}/{:?}", f.format, f.color_space)).collect();
                crate::info!("present: surface formats {}", list.join(", "));
            });
        }
        // Display buffers hold display-encoded values (the title applies its
        // own transfer function): UNORM swapchain formats show them as is. SDR goes to 8 bits
        // per channel as PC games present (until 2026-11-13 10-bit first, the title's buffers
        // being 10:10:10:2: the user saw a grey picture and NVIDIA overlay in exclusive
        // fullscreen); `DS_PRESENT_10BIT=1` keeps 10-bit SDR (diagnosis). HDR10 is 10-bit.
        let srgb_space = |f: &&vk::SurfaceFormatKHR| f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR;
        // The title's HDR output (PQ-encoded BT.2020 in 10:10:10:2 display buffers) goes to an
        // HDR10 swapchain as it is, where the display offers one.
        let hdr10 = p.want_hdr && c.colorspaces;
        let hdr_fmt = formats
            .iter()
            .find(|f| hdr10 && f.color_space == vk::ColorSpaceKHR::HDR10_ST2084_EXT && matches!(f.format, vk::Format::A2B10G10R10_UNORM_PACK32 | vk::Format::A2R10G10B10_UNORM_PACK32))
            .copied();
        if p.want_hdr && hdr_fmt.is_none() {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| crate::warn!("present: the frames are HDR but the display offers no HDR10 format: shown as SDR values"));
        }
        let ten = std::env::var("DS_PRESENT_10BIT").is_ok_and(|v| v == "1");
        let fmt = hdr_fmt
            .or_else(|| formats.iter().filter(srgb_space).find(|f| ten && f.format == vk::Format::A2B10G10R10_UNORM_PACK32).copied())
            .or_else(|| formats.iter().filter(srgb_space).find(|f| f.format == vk::Format::B8G8R8A8_UNORM).copied())
            .or_else(|| formats.iter().filter(srgb_space).find(|f| f.format == vk::Format::R8G8B8A8_UNORM).copied())
            .or_else(|| formats.iter().filter(srgb_space).find(|f| f.format == vk::Format::A2B10G10R10_UNORM_PACK32).copied())
            .or_else(|| formats.first().copied())
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
        // Exclusive fullscreen: held by the swapchain when the window asks for it (the PC tab's
        // Fullscreen), and never taken by the driver on its own for a window or borderless
        // fullscreen (a driver may change a display's mode for a window that covers it).
        let exclusive = if c.fse.is_some() { window::exclusive_wanted() } else { None };
        let mut fse_info = vk::SurfaceFullScreenExclusiveInfoEXT::default().full_screen_exclusive(if exclusive.is_some() {
            vk::FullScreenExclusiveEXT::APPLICATION_CONTROLLED
        } else {
            vk::FullScreenExclusiveEXT::DISALLOWED
        });
        let mut fse_win = vk::SurfaceFullScreenExclusiveWin32InfoEXT::default().hmonitor(exclusive.unwrap_or(0) as vk::HMONITOR);
        let mut sci = vk::SwapchainCreateInfoKHR::default()
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
        if c.fse.is_some() {
            sci = sci.push_next(&mut fse_info);
            if exclusive.is_some() {
                sci = sci.push_next(&mut fse_win);
            }
        }
        let sc = c.swapchain_d.create_swapchain(&sci, None).map_err(|e| format!("create swapchain: {e}"))?;
        if old != vk::SwapchainKHR::null() {
            // The old swapchain's images are only the presenter's: its frames done and the queue
            // idle under the queue's lock (a device-wide idle wait touches the queue while the
            // GPU threads submit to it).
            c.device.wait_for_fences(&p.fences, true, u64::MAX).ok();
            {
                let q = c.queue.lock();
                c.device.queue_wait_idle(*q).ok();
            }
            c.swapchain_d.destroy_swapchain(old, None);
            for s in p.done.drain(..) {
                c.device.destroy_semaphore(s, None);
            }
        }
        p.swapchain = sc;
        p.hdr = fmt.color_space == vk::ColorSpaceKHR::HDR10_ST2084_EXT;
        if p.hdr {
            if let Some(m) = c.hdr_metadata.as_ref() {
                // HDR10's usual mastering description (BT.2020 primaries, D65 white); the title's
                // own Display Adjustment sets its brightness.
                let xy = |x: f32, y: f32| vk::XYColorEXT { x, y };
                let md = vk::HdrMetadataEXT::default()
                    .display_primary_red(xy(0.708, 0.292))
                    .display_primary_green(xy(0.170, 0.797))
                    .display_primary_blue(xy(0.131, 0.046))
                    .white_point(xy(0.3127, 0.3290))
                    .max_luminance(1000.0)
                    .min_luminance(0.001)
                    .max_content_light_level(1000.0)
                    .max_frame_average_light_level(400.0);
                m.set_hdr_metadata(&[sc], &[md]);
            }
        }
        p.images = c.swapchain_d.get_swapchain_images(sc).map_err(|e| format!("swapchain images: {e}"))?;
        p.format = fmt.format;
        p.extent = extent;
        for _ in 0..p.images.len() {
            p.done.push(c.device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None).map_err(|e| e.to_string())?);
        }
        p.exclusive = None;
        if let (Some(f), Some(mon)) = (c.fse.as_ref(), exclusive) {
            match f.acquire_full_screen_exclusive_mode(sc) {
                Ok(()) => p.exclusive = Some(mon),
                Err(e) => crate::warn!("present: exclusive fullscreen not acquired ({e}): shown as borderless"),
            }
        }
        crate::info!(
            "swapchain {}x{} {:?} {:?} ({} images{})",
            extent.width,
            extent.height,
            fmt.format,
            fmt.color_space,
            p.images.len(),
            if p.exclusive.is_some() { ", exclusive fullscreen" } else { "" }
        );
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
            sharpen: None,
            sharpen_failed: false,
            exclusive: None,
            hdr: false,
            want_hdr: false,
            watch: None,
            watch_failed: false,
            text_box: None,
        };
        super::framewatch::init_from_env();
        create_swapchain(&mut p)?;
        *PRESENTER.lock() = Some(p);
    }
    ACTIVE.store(true, std::sync::atomic::Ordering::Relaxed);
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
    let Some(p) = g.as_mut() else {
        // Without a window: the frame watch still sees each frame (`framewatch::headless`).
        if let Source::Image { image, layout, width, height, .. } = src {
            drop(g);
            super::framewatch::init_from_env();
            super::framewatch::headless(image, layout, width, height);
        }
        return false;
    };
    unsafe {
        // HDR frames from SDR ones or back (the title's display buffers' format): a new swapchain.
        if let Source::Image { hdr, .. } = &src {
            if *hdr != p.want_hdr {
                p.want_hdr = *hdr;
                crate::info!("present: the title's frames are {}", if *hdr { "HDR10" } else { "SDR" });
                window::RESIZED.store(true, std::sync::atomic::Ordering::SeqCst);
            }
        }
        // The window's exclusive fullscreen changed (Alt+Tab, the mode): a new swapchain.
        if window::exclusive_wanted() != p.exclusive && c.fse.is_some() {
            window::RESIZED.store(true, std::sync::atomic::Ordering::SeqCst);
        }
        if window::RESIZED.swap(false, std::sync::atomic::Ordering::SeqCst) {
            if let Err(e) = create_swapchain(p) {
                crate::debug!("swapchain recreate: {e}");
                return false;
            }
        }
        let f = p.frame % FRAMES;
        p.frame += 1;
        c.device.wait_for_fences(&[p.fences[f]], true, u64::MAX).ok();
        if let Some(w) = p.watch.as_mut() {
            w.collect(f);
        }
        let (idx, _) = match c.swapchain_d.acquire_next_image(p.swapchain, u64::MAX, p.acquire[f], vk::Fence::null()) {
            Ok(r) => r,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR | vk::Result::ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT) => {
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
            Source::Image { image, format, layout, width, height, .. } => {
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
        // The PC settings' sharpening: the frame blitted at its shown size, sharpened, blitted
        // to the window.
        let amount = super::sharpen::amount();
        let mut sharpened = false;
        if amount > 0.0 && !p.sharpen_failed {
            if p.sharpen.is_none() {
                match super::sharpen::Sharpen::new() {
                    Ok(s) => p.sharpen = Some(s),
                    Err(e) => {
                        crate::warn!("present: no sharpening ({e})");
                        p.sharpen_failed = true;
                    }
                }
            }
            let fences: Vec<vk::Fence> = p.fences.iter().enumerate().filter(|&(i, _)| i != f).map(|(_, &x)| x).collect();
            if let Some(s) = p.sharpen.as_mut() {
                match s.ensure(rw, rh, || {
                    c.device.wait_for_fences(&fences, true, u64::MAX).ok();
                }) {
                    Ok((a, b, fresh)) => {
                        s.begin(cmd, fresh);
                        blit_to(c, cmd, from, from_layout, (0, 0, fw, fh), a, vk::ImageLayout::GENERAL, (0, 0, rw, rh));
                        s.record(cmd, amount);
                        blit(c, cmd, b, vk::ImageLayout::GENERAL, (0, 0, rw, rh), dst, (rx, ry, rw, rh));
                        sharpened = true;
                    }
                    Err(e) => {
                        crate::warn!("present: no sharpening images ({e})");
                        p.sharpen_failed = true;
                    }
                }
            }
        }
        if !sharpened {
            blit(c, cmd, from, from_layout, (0, 0, fw, fh), dst, (rx, ry, rw, rh));
        }
        // The frame watch's copy of the frame as shown (before the window's scaling).
        if super::framewatch::ON.load(std::sync::atomic::Ordering::Relaxed) && !p.watch_failed {
            if p.watch.is_none() {
                match super::framewatch::Watch::new(FRAMES) {
                    Ok(w) => p.watch = Some(w),
                    Err(e) => {
                        crate::warn!("frame watch: unavailable ({e})");
                        p.watch_failed = true;
                    }
                }
            }
            if let Some(w) = p.watch.as_mut() {
                w.record(cmd, f, from, from_layout, fw, fh);
            }
        }
        // The system text box (the character's name) over the frame.
        draw_text_box(p, cmd, f, dst, (rx, ry, rw, rh));
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
        // (Through Streamline's present once DLSS runs: its per-frame bookkeeping.)
        let presented = match crate::gpu::upscale::present_hook() {
            Some(f) => match f(*q, &pi) {
                vk::Result::SUCCESS => Ok(false),
                vk::Result::SUBOPTIMAL_KHR => Ok(true),
                e => Err(e),
            },
            None => c.swapchain_d.queue_present(*q, &pi),
        };
        match presented {
            Ok(false) => {}
            Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR | vk::Result::ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT) => window::RESIZED.store(true, std::sync::atomic::Ordering::SeqCst),
            Err(e) => crate::error!("present: {e}"),
        }
    }
    true
}

/// Where a `sw`x`sh` frame goes in a `dw`x`dh` window (x, y, width, height):
/// as large as fits with its aspect kept, centred; the whole window when the
/// aspects differ by under 1%.
/// A window shows the frames (`init` succeeded): the render scale follows its area.
static ACTIVE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// A window shows the frames.
pub fn active() -> bool {
    ACTIVE.load(std::sync::atomic::Ordering::Relaxed)
}

/// The window's area the title's frame fills (its output's shape; pixels; None without a window).
pub fn area() -> Option<(u32, u32)> {
    use std::sync::atomic::Ordering::Relaxed;
    if !ACTIVE.load(Relaxed) {
        return None;
    }
    let (w, h) = (window::CLIENT_W.load(Relaxed), window::CLIENT_H.load(Relaxed));
    if w < 64 || h < 64 {
        return None;
    }
    let (ow, oh) = crate::dispmode::output_size();
    let (_, _, fw, fh) = fit(ow, oh, w, h);
    Some((fw, fh))
}

pub fn fit(sw: u32, sh: u32, dw: u32, dh: u32) -> (u32, u32, u32, u32) {
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
    blit_to(c, cmd, src, src_layout, s, dst, vk::ImageLayout::TRANSFER_DST_OPTIMAL, d);
}

#[allow(clippy::too_many_arguments)]
fn blit_to(c: &super::Ctx, cmd: vk::CommandBuffer, src: vk::Image, src_layout: vk::ImageLayout, s: (u32, u32, u32, u32), dst: vk::Image, dst_layout: vk::ImageLayout, d: (u32, u32, u32, u32)) {
    let layers = vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 };
    let corners = |(x, y, w, h): (u32, u32, u32, u32)| [vk::Offset3D { x: x as i32, y: y as i32, z: 0 }, vk::Offset3D { x: (x + w) as i32, y: (y + h) as i32, z: 1 }];
    let region = vk::ImageBlit { src_subresource: layers, src_offsets: corners(s), dst_subresource: layers, dst_offsets: corners(d) };
    unsafe { c.device.cmd_blit_image(cmd, src, src_layout, dst, dst_layout, &[region], vk::Filter::LINEAR) };
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
