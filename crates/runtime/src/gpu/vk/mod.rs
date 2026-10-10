//! Vulkan device context shared by the renderer and the presenter.

pub mod present;
pub mod window;

use ash::vk;
use parking_lot::Mutex;
use std::ffi::{CStr, c_char};
use std::sync::OnceLock;

pub struct Ctx {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub pdev: vk::PhysicalDevice,
    pub device: ash::Device,
    pub queue: Mutex<vk::Queue>,
    pub queue_family: u32,
    pub mem_props: vk::PhysicalDeviceMemoryProperties,
    pub limits: vk::PhysicalDeviceLimits,
    pub surface_i: ash::khr::surface::Instance,
    pub win32_i: ash::khr::win32_surface::Instance,
    pub swapchain_d: ash::khr::swapchain::Device,
    pub push_desc: ash::khr::push_descriptor::Device,
    /// VK_EXT_external_memory_host (zero-copy access to guest memory).
    pub ext_host: Option<ash::ext::external_memory_host::Device>,
    /// VK_NV_device_diagnostic_checkpoints (DS_GPU_CHECKPOINTS=1): markers
    /// read back after a device loss.
    pub checkpoints: Option<ash::nv::device_diagnostic_checkpoints::Device>,
    /// VK_KHR_pipeline_executable_properties (DS_PIPELINE_STATS=1, `gpu::pstats`).
    pub exec_props: Option<ash::khr::pipeline_executable_properties::Device>,
    pub device_name: String,
    pub validation: bool,
    /// VK_KHR_fragment_shader_barycentric enabled.
    pub barycentric: bool,
}

static CTX: OnceLock<Ctx> = OnceLock::new();

pub fn ctx() -> &'static Ctx {
    CTX.get().expect("Vulkan not initialized")
}

pub fn try_ctx() -> Option<&'static Ctx> {
    CTX.get()
}

unsafe extern "system" fn debug_cb(
    sev: vk::DebugUtilsMessageSeverityFlagsEXT,
    _ty: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user: *mut std::ffi::c_void,
) -> vk::Bool32 {
    let msg = unsafe { CStr::from_ptr((*data).p_message) }.to_string_lossy();
    if sev.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
        crate::error!("vulkan: {msg}");
    } else {
        crate::warn!("vulkan: {msg}");
    }
    vk::FALSE
}

pub fn init() -> Result<(), String> {
    if CTX.get().is_some() {
        return Ok(());
    }
    unsafe {
        let entry = ash::Entry::load().map_err(|e| format!("loading Vulkan: {e}"))?;
        let app_name = c"dsport";
        let app = vk::ApplicationInfo::default()
            .application_name(app_name)
            .engine_name(app_name)
            .api_version(vk::API_VERSION_1_3);
        let layers_avail = entry.enumerate_instance_layer_properties().unwrap_or_default();
        let want_validation = std::env::var_os("DS_VK_VALIDATION").is_some();
        let validation_name = c"VK_LAYER_KHRONOS_validation";
        let has_validation = layers_avail.iter().any(|l| CStr::from_ptr(l.layer_name.as_ptr()) == validation_name);
        let validation = want_validation && has_validation;
        if want_validation && !has_validation {
            crate::warn!("DS_VK_VALIDATION set but the Khronos validation layer is not installed");
        }
        let layers: Vec<*const c_char> = if validation { vec![validation_name.as_ptr()] } else { vec![] };
        let mut exts = vec![ash::khr::surface::NAME.as_ptr(), ash::khr::win32_surface::NAME.as_ptr()];
        if validation {
            exts.push(ash::ext::debug_utils::NAME.as_ptr());
        }
        let ici = vk::InstanceCreateInfo::default().application_info(&app).enabled_layer_names(&layers).enabled_extension_names(&exts);
        let instance = entry.create_instance(&ici, None).map_err(|e| format!("vkCreateInstance: {e}"))?;
        if validation {
            let du = ash::ext::debug_utils::Instance::new(&entry, &instance);
            let mci = vk::DebugUtilsMessengerCreateInfoEXT::default()
                .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING)
                .message_type(vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION | vk::DebugUtilsMessageTypeFlagsEXT::GENERAL)
                .pfn_user_callback(Some(debug_cb));
            let _ = du.create_debug_utils_messenger(&mci, None);
        }
        // Pick a discrete GPU with a graphics+compute queue.
        let pdevs = instance.enumerate_physical_devices().map_err(|e| format!("enumerate devices: {e}"))?;
        let mut chosen = None;
        for &pd in &pdevs {
            let props = instance.get_physical_device_properties(pd);
            let qfs = instance.get_physical_device_queue_family_properties(pd);
            if let Some(qf) = qfs.iter().position(|q| q.queue_flags.contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)) {
                let discrete = props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU;
                if chosen.is_none() || discrete {
                    chosen = Some((pd, qf as u32, props));
                }
            }
        }
        let (pdev, qf, props) = chosen.ok_or("no Vulkan device with graphics+compute")?;
        let device_name = CStr::from_ptr(props.device_name.as_ptr()).to_string_lossy().into_owned();
        let prio = [1.0f32];
        let qci = [vk::DeviceQueueCreateInfo::default().queue_family_index(qf).queue_priorities(&prio)];
        let host_import = instance
            .enumerate_device_extension_properties(pdev)
            .unwrap_or_default()
            .iter()
            .any(|e| CStr::from_ptr(e.extension_name.as_ptr()) == ash::ext::external_memory_host::NAME);
        let mut dexts = vec![ash::khr::swapchain::NAME.as_ptr(), ash::khr::push_descriptor::NAME.as_ptr()];
        if host_import {
            dexts.push(ash::ext::external_memory_host::NAME.as_ptr());
        }
        let want_ckpt = std::env::var("DS_GPU_CHECKPOINTS").ok().as_deref() == Some("1");
        let has_ckpt = want_ckpt
            && instance
                .enumerate_device_extension_properties(pdev)
                .unwrap_or_default()
                .iter()
                .any(|e| CStr::from_ptr(e.extension_name.as_ptr()) == ash::nv::device_diagnostic_checkpoints::NAME);
        if has_ckpt {
            dexts.push(ash::nv::device_diagnostic_checkpoints::NAME.as_ptr());
        } else if want_ckpt {
            crate::warn!("DS_GPU_CHECKPOINTS=1 but VK_NV_device_diagnostic_checkpoints is unavailable");
        }
        let want_stats = std::env::var("DS_PIPELINE_STATS").ok().as_deref() == Some("1");
        let has_stats = want_stats
            && instance
                .enumerate_device_extension_properties(pdev)
                .unwrap_or_default()
                .iter()
                .any(|e| CStr::from_ptr(e.extension_name.as_ptr()) == ash::khr::pipeline_executable_properties::NAME);
        if has_stats {
            dexts.push(ash::khr::pipeline_executable_properties::NAME.as_ptr());
        } else if want_stats {
            crate::warn!("DS_PIPELINE_STATS=1 but VK_KHR_pipeline_executable_properties is unavailable");
        }
        let mut stats_f = vk::PhysicalDevicePipelineExecutablePropertiesFeaturesKHR::default().pipeline_executable_info(true);
        // Out-of-range buffer/image accesses must not fault: GCN bounds-checks
        // buffer accesses against the descriptor (loads return 0, stores are
        // dropped), and titles rely on it. robustness2 gives exactly that.
        let has_robust2 = instance
            .enumerate_device_extension_properties(pdev)
            .unwrap_or_default()
            .iter()
            .any(|e| CStr::from_ptr(e.extension_name.as_ptr()) == ash::ext::robustness2::NAME);
        let mut r2_supported = vk::PhysicalDeviceRobustness2FeaturesEXT::default();
        if has_robust2 {
            let mut f2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut r2_supported);
            instance.get_physical_device_features2(pdev, &mut f2);
        }
        let mut r2 = vk::PhysicalDeviceRobustness2FeaturesEXT::default()
            .robust_buffer_access2(r2_supported.robust_buffer_access2 != 0)
            .robust_image_access2(r2_supported.robust_image_access2 != 0)
            .null_descriptor(r2_supported.null_descriptor != 0);
        if has_robust2 {
            dexts.push(ash::ext::robustness2::NAME.as_ptr());
        }
        // Fragment barycentrics: the title interpolates some attributes itself
        // from each vertex's value (v_interp_mov of P10/P20).
        let has_bary_ext = instance
            .enumerate_device_extension_properties(pdev)
            .unwrap_or_default()
            .iter()
            .any(|e| CStr::from_ptr(e.extension_name.as_ptr()) == ash::khr::fragment_shader_barycentric::NAME);
        let mut bary_supported = vk::PhysicalDeviceFragmentShaderBarycentricFeaturesKHR::default();
        if has_bary_ext {
            let mut f2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut bary_supported);
            instance.get_physical_device_features2(pdev, &mut f2);
        }
        let barycentric = has_bary_ext && bary_supported.fragment_shader_barycentric != 0;
        let mut bary = vk::PhysicalDeviceFragmentShaderBarycentricFeaturesKHR::default().fragment_shader_barycentric(true);
        if barycentric {
            dexts.push(ash::khr::fragment_shader_barycentric::NAME.as_ptr());
        }
        let features = vk::PhysicalDeviceFeatures::default()
            .robust_buffer_access(true)
            .geometry_shader(true)
            .independent_blend(true)
            .dual_src_blend(true)
            .logic_op(true)
            .fill_mode_non_solid(true)
            .depth_clamp(true)
            .depth_bias_clamp(true)
            .sampler_anisotropy(true)
            .texture_compression_bc(true)
            .image_cube_array(true)
            .shader_int64(true)
            .shader_storage_image_write_without_format(true)
            .shader_storage_image_read_without_format(true)
            .vertex_pipeline_stores_and_atomics(true)
            .fragment_stores_and_atomics(true)
            .multi_draw_indirect(true)
            .sample_rate_shading(true);
        // Image tables are indexed per invocation (non-uniform indexing).
        let mut f12_supported = vk::PhysicalDeviceVulkan12Features::default();
        {
            let mut f2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut f12_supported);
            instance.get_physical_device_features2(pdev, &mut f2);
        }
        let nonuniform = f12_supported.shader_sampled_image_array_non_uniform_indexing != 0;
        if !nonuniform {
            crate::warn!("Vulkan: shaderSampledImageArrayNonUniformIndexing unsupported (image tables index uniformly)");
        }
        let mut f12 = vk::PhysicalDeviceVulkan12Features::default()
            .shader_sampled_image_array_non_uniform_indexing(nonuniform)
            .timeline_semaphore(true)
            .scalar_block_layout(true)
            .shader_float16(true)
            .storage_buffer8_bit_access(true)
            .uniform_and_storage_buffer8_bit_access(true)
            .shader_int8(true);
        let mut f13 = vk::PhysicalDeviceVulkan13Features::default().dynamic_rendering(true).synchronization2(true).maintenance4(true);
        let mut dci = vk::DeviceCreateInfo::default()
            .queue_create_infos(&qci)
            .enabled_extension_names(&dexts)
            .enabled_features(&features)
            .push_next(&mut f12)
            .push_next(&mut f13);
        if has_robust2 {
            dci = dci.push_next(&mut r2);
        }
        if barycentric {
            dci = dci.push_next(&mut bary);
        }
        if has_stats {
            dci = dci.push_next(&mut stats_f);
        }
        let device = instance.create_device(pdev, &dci, None).map_err(|e| format!("vkCreateDevice: {e}"))?;
        crate::info!(
            "Vulkan robustness: robustBufferAccess, robustness2 {} (buffer {}, image {}, null descriptors {})",
            has_robust2,
            r2_supported.robust_buffer_access2 != 0,
            r2_supported.robust_image_access2 != 0,
            r2_supported.null_descriptor != 0
        );
        let queue = device.get_device_queue(qf, 0);
        let mem_props = instance.get_physical_device_memory_properties(pdev);
        let ctx = Ctx {
            surface_i: ash::khr::surface::Instance::new(&entry, &instance),
            win32_i: ash::khr::win32_surface::Instance::new(&entry, &instance),
            swapchain_d: ash::khr::swapchain::Device::new(&instance, &device),
            push_desc: ash::khr::push_descriptor::Device::new(&instance, &device),
            ext_host: host_import.then(|| ash::ext::external_memory_host::Device::new(&instance, &device)),
            checkpoints: has_ckpt.then(|| ash::nv::device_diagnostic_checkpoints::Device::new(&instance, &device)),
            exec_props: has_stats.then(|| ash::khr::pipeline_executable_properties::Device::new(&instance, &device)),
            entry,
            instance,
            pdev,
            device,
            queue: Mutex::new(queue),
            queue_family: qf,
            mem_props,
            limits: props.limits,
            device_name,
            validation,
            barycentric,
        };
        crate::info!("Vulkan device: {} (queue family {qf}, validation {})", ctx.device_name, validation);
        {
            let exts = ctx.instance.enumerate_device_extension_properties(pdev).unwrap_or_default();
            let has = |n: &CStr| exts.iter().any(|e| CStr::from_ptr(e.extension_name.as_ptr()) == n);
            let mut host = vk::PhysicalDeviceExternalMemoryHostPropertiesEXT::default();
            let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut host);
            ctx.instance.get_physical_device_properties2(pdev, &mut p2);
            crate::info!(
                "Vulkan limits: maxStorageBufferRange {:#x}, minStorageBufferOffsetAlignment {}, external_memory_host {} (import alignment {:#x}), buffer_device_address {}",
                ctx.limits.max_storage_buffer_range,
                ctx.limits.min_storage_buffer_offset_alignment,
                has(ash::ext::external_memory_host::NAME),
                host.min_imported_host_pointer_alignment,
                has(ash::khr::buffer_device_address::NAME)
            );
        }
        CTX.set(ctx).ok();
        Ok(())
    }
}

/// Checkpoint marker kinds (high byte of the marker value).
pub mod mark {
    pub const DISPATCH: u64 = 1;
    pub const DRAW: u64 = 2;
    pub const SHADOW_IN: u64 = 3;
    pub const SHADOW_OUT: u64 = 4;
    pub const UPLOAD: u64 = 5;
    pub const BATCH: u64 = 6;
}

/// Records a diagnostic checkpoint (DS_GPU_CHECKPOINTS=1): `kind` and a
/// 56-bit value (shader or guest address, sequence number).
pub fn checkpoint(cmd: vk::CommandBuffer, kind: u64, value: u64) {
    if let Some(c) = try_ctx() {
        if let Some(cp) = &c.checkpoints {
            unsafe { cp.cmd_set_checkpoint(cmd, ((kind << 56) | (value & ((1 << 56) - 1))) as *const std::ffi::c_void) };
        }
    }
}

/// The queue's last checkpoints per pipeline stage, after a device loss.
pub fn checkpoint_report() -> String {
    let Some(c) = try_ctx() else { return String::new() };
    let Some(cp) = &c.checkpoints else { return "(DS_GPU_CHECKPOINTS=1 names the failing command)".into() };
    let q = c.queue.lock();
    unsafe {
        let n = cp.get_queue_checkpoint_data_len(*q);
        let mut v = vec![vk::CheckpointDataNV::default(); n];
        cp.get_queue_checkpoint_data(*q, &mut v);
        v.iter()
            .map(|d| {
                let m = d.p_checkpoint_marker as u64;
                let kind = match m >> 56 {
                    mark::DISPATCH => "dispatch cs",
                    mark::DRAW => "draw ps",
                    mark::SHADOW_IN => "shadow copy in",
                    mark::SHADOW_OUT => "shadow copy out",
                    mark::UPLOAD => "image upload",
                    mark::BATCH => "batch",
                    _ => "?",
                };
                format!("{:?}: {kind} {:#x}", d.stage, m & ((1 << 56) - 1))
            })
            .collect::<Vec<_>>()
            .join("; ")
    }
}

impl Ctx {
    pub fn memory_type(&self, bits: u32, want: vk::MemoryPropertyFlags) -> Option<u32> {
        (0..self.mem_props.memory_type_count)
            .find(|&i| bits & (1 << i) != 0 && self.mem_props.memory_types[i as usize].property_flags.contains(want))
    }

    pub fn alloc(&self, req: vk::MemoryRequirements, want: vk::MemoryPropertyFlags) -> Result<vk::DeviceMemory, String> {
        let ty = self.memory_type(req.memory_type_bits, want).ok_or("no suitable memory type")?;
        let ai = vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(ty);
        unsafe { self.device.allocate_memory(&ai, None) }.map_err(|e| format!("vkAllocateMemory({}): {e}", req.size))
    }

    /// Device-local buffer (not host-visible).
    pub fn device_buffer(&self, size: u64, usage: vk::BufferUsageFlags) -> Result<(vk::Buffer, vk::DeviceMemory), String> {
        unsafe {
            let bi = vk::BufferCreateInfo::default().size(size).usage(usage).sharing_mode(vk::SharingMode::EXCLUSIVE);
            let buf = self.device.create_buffer(&bi, None).map_err(|e| format!("vkCreateBuffer: {e}"))?;
            let req = self.device.get_buffer_memory_requirements(buf);
            let mem = self.alloc(req, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
            self.device.bind_buffer_memory(buf, mem, 0).map_err(|e| format!("bind buffer: {e}"))?;
            Ok((buf, mem))
        }
    }

    /// Imports existing host memory [ptr, ptr+size) (page aligned) as a GPU
    /// buffer (VK_EXT_external_memory_host): the GPU reads and writes it in place.
    pub fn import_host(&self, ptr: u64, size: u64, usage: vk::BufferUsageFlags) -> Result<(vk::Buffer, vk::DeviceMemory), String> {
        let ext = self.ext_host.as_ref().ok_or("VK_EXT_external_memory_host not available")?;
        let ht = vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT;
        unsafe {
            let mut props = vk::MemoryHostPointerPropertiesEXT::default();
            let r = (ext.fp().get_memory_host_pointer_properties_ext)(self.device.handle(), ht, ptr as *const std::ffi::c_void, &mut props);
            if r != vk::Result::SUCCESS {
                return Err(format!("vkGetMemoryHostPointerPropertiesEXT: {r}"));
            }
            let mut ebi = vk::ExternalMemoryBufferCreateInfo::default().handle_types(ht);
            let bi = vk::BufferCreateInfo::default().size(size).usage(usage).sharing_mode(vk::SharingMode::EXCLUSIVE).push_next(&mut ebi);
            let buf = self.device.create_buffer(&bi, None).map_err(|e| format!("vkCreateBuffer: {e}"))?;
            let req = self.device.get_buffer_memory_requirements(buf);
            let bits = req.memory_type_bits & props.memory_type_bits;
            let hv = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
            let ty = self
                .memory_type(bits, hv | vk::MemoryPropertyFlags::HOST_CACHED)
                .or_else(|| self.memory_type(bits, hv))
                .or_else(|| self.memory_type(bits, vk::MemoryPropertyFlags::empty()));
            let Some(ty) = ty else {
                self.device.destroy_buffer(buf, None);
                return Err(format!("no memory type for imported host memory (bits {bits:#x})"));
            };
            let mut imp = vk::ImportMemoryHostPointerInfoEXT::default().handle_type(ht).host_pointer(ptr as *mut std::ffi::c_void);
            let ai = vk::MemoryAllocateInfo::default().allocation_size(size).memory_type_index(ty).push_next(&mut imp);
            let mem = match self.device.allocate_memory(&ai, None) {
                Ok(m) => m,
                Err(e) => {
                    self.device.destroy_buffer(buf, None);
                    return Err(format!("vkAllocateMemory(import {size:#x}): {e}"));
                }
            };
            if let Err(e) = self.device.bind_buffer_memory(buf, mem, 0) {
                self.device.destroy_buffer(buf, None);
                self.device.free_memory(mem, None);
                return Err(format!("bind imported buffer: {e}"));
            }
            Ok((buf, mem))
        }
    }

    /// Device-local buffer the CPU writes through its mapping (the PCIe BAR
    /// window; writes are write-combined, reads by the CPU are very slow).
    /// GPU reads stay in video memory, unlike `host_buffer`.
    pub fn bar_buffer(&self, size: u64, usage: vk::BufferUsageFlags) -> Result<(vk::Buffer, vk::DeviceMemory, *mut u8), String> {
        unsafe {
            let bi = vk::BufferCreateInfo::default().size(size).usage(usage).sharing_mode(vk::SharingMode::EXCLUSIVE);
            let buf = self.device.create_buffer(&bi, None).map_err(|e| format!("vkCreateBuffer: {e}"))?;
            let req = self.device.get_buffer_memory_requirements(buf);
            let want = vk::MemoryPropertyFlags::DEVICE_LOCAL | vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
            let mem = match self.alloc(req, want) {
                Ok(m) => m,
                Err(e) => {
                    self.device.destroy_buffer(buf, None);
                    return Err(e);
                }
            };
            self.device.bind_buffer_memory(buf, mem, 0).map_err(|e| format!("bind buffer: {e}"))?;
            let ptr = self.device.map_memory(mem, 0, size, vk::MemoryMapFlags::empty()).map_err(|e| format!("map: {e}"))? as *mut u8;
            Ok((buf, mem, ptr))
        }
    }

    /// Host-visible, coherent buffer (staging / per-draw data).
    pub fn host_buffer(&self, size: u64, usage: vk::BufferUsageFlags) -> Result<(vk::Buffer, vk::DeviceMemory, *mut u8), String> {
        unsafe {
            let bi = vk::BufferCreateInfo::default().size(size).usage(usage).sharing_mode(vk::SharingMode::EXCLUSIVE);
            let buf = self.device.create_buffer(&bi, None).map_err(|e| format!("vkCreateBuffer: {e}"))?;
            let req = self.device.get_buffer_memory_requirements(buf);
            // Prefer cached host memory: results are read back by the CPU
            // (buffer write-back), which is very slow from write-combined memory.
            let hv = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
            let mem = self.alloc(req, hv | vk::MemoryPropertyFlags::HOST_CACHED).or_else(|_| self.alloc(req, hv))?;
            self.device.bind_buffer_memory(buf, mem, 0).map_err(|e| format!("bind buffer: {e}"))?;
            let ptr = self.device.map_memory(mem, 0, size, vk::MemoryMapFlags::empty()).map_err(|e| format!("map: {e}"))? as *mut u8;
            Ok((buf, mem, ptr))
        }
    }

    pub fn submit(&self, cmd: vk::CommandBuffer, wait: &[(vk::Semaphore, vk::PipelineStageFlags)], signal: &[vk::Semaphore], fence: vk::Fence) -> Result<(), String> {
        let cmds = [cmd];
        let ws: Vec<vk::Semaphore> = wait.iter().map(|w| w.0).collect();
        let wst: Vec<vk::PipelineStageFlags> = wait.iter().map(|w| w.1).collect();
        let si = vk::SubmitInfo::default().command_buffers(&cmds).wait_semaphores(&ws).wait_dst_stage_mask(&wst).signal_semaphores(signal);
        let q = self.queue.lock();
        unsafe { self.device.queue_submit(*q, &[si], fence) }.map_err(|e| format!("vkQueueSubmit: {e}"))
    }
}
