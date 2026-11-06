//! Vulkan device context shared by the renderer and the presenter.

pub mod defer;
pub mod present;
pub mod sharpen;
pub mod window;

use ash::vk;
use parking_lot::Mutex;
use std::ffi::{CStr, c_char};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

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
    /// VkPhysicalDevicePushDescriptorPropertiesKHR::maxPushDescriptors.
    pub max_push: u32,
    /// VK_EXT_external_memory_host (zero-copy access to guest memory).
    pub ext_host: Option<ash::ext::external_memory_host::Device>,
    /// VK_NV_device_diagnostic_checkpoints (DS_GPU_CHECKPOINTS=1): markers
    /// read back after a device loss.
    pub checkpoints: Option<ash::nv::device_diagnostic_checkpoints::Device>,
    /// VK_KHR_pipeline_executable_properties (DS_PIPELINE_STATS=1, `gpu::pstats`).
    pub exec_props: Option<ash::khr::pipeline_executable_properties::Device>,
    pub device_name: String,
    pub validation: bool,
    /// VK_EXT_debug_utils command labels (DS_GPU_LABELS=1): named ranges of
    /// the recorded work for GPU Trace (Nsight Graphics).
    pub labels: Option<ash::ext::debug_utils::Device>,
    /// VK_EXT_debug_utils object names under the validation layer
    /// (DS_VK_VALIDATION=1): its messages name the objects (`name_object`).
    pub names: Option<ash::ext::debug_utils::Device>,
    /// VK_KHR_fragment_shader_barycentric enabled.
    pub barycentric: bool,
    /// VK_NV_shader_subgroup_partitioned enabled (`subgroup_partitioned`).
    pub partitioned: bool,
    /// VK_EXT_memory_budget enabled (`heap_report`).
    pub memory_budget: bool,
    /// VK_EXT_calibrated_timestamps with the device and performance-counter
    /// time domains (`gpu::timeline`: GPU timestamps on the CPU clock).
    pub calibrated: Option<ash::ext::calibrated_timestamps::Device>,
    /// VK_EXT_device_fault: the device's own account of a loss (`fault_report`).
    pub fault: Option<ash::ext::device_fault::Device>,
    /// bufferDeviceAddress enabled: guest imports and video-memory homes get
    /// GPU addresses that fault reports name (`note_gpu_range`).
    pub device_addresses: bool,
    /// drawIndirectCount enabled: batched draws grow after recording
    /// (`exec::draw` draw runs).
    pub draw_indirect_count: bool,
    /// sparseBinding and sparseResidencyBuffer enabled (`DS_RESIDENT=1`:
    /// the residency mirror, `gpu::resident`).
    pub sparse_buffers: bool,
    /// shaderDrawParameters and drawIndirectFirstInstance enabled: vertex
    /// shaders of indirect draws take the start instance from
    /// gl_BaseInstance (`tune cpu_args_base`).
    pub draw_parameters: bool,
    /// The copy queue (a transfer-only queue family; `DS_COPY_QUEUE=0`:
    /// none): batch-start copies run on it beside the main queue (`exec`).
    pub copy: Option<CopyQueue>,
    /// Ordered async compute (RENDERER.md; `DS_ASYNC_COMPUTE` 2 or more at
    /// boot, `async_compute_boot`): a queue of a compute-only family that the
    /// compute queues' batches run on beside the main queue.
    pub compute: Option<ComputeQueue>,
    /// With `compute`: a timeline every submission to the main queue
    /// signals with the next value of `g_count` (`progress_signal`); a batch
    /// on the compute queue waits for the value current when it is
    /// submitted, after all earlier work of the main queue.
    pub g_progress: vk::Semaphore,
    g_count: AtomicU64,
    /// The distinct queue families buffers are used by (the main queue's,
    /// the copy queue's, the compute queue's).
    pub families: Vec<u32>,
    /// A timeline semaphore each batch on the main queue signals with its
    /// sequence (`exec`: batches submitted together, completions waited for
    /// by value).
    pub batches: vk::Semaphore,
    /// A timeline semaphore the host signals with a batch's sequence once
    /// its deferred effects (CP writes the CPU performs) ran: batches whose
    /// GPU-performed CP writes follow those wait for it
    /// (`exec::Exec::order_after_cpu`).
    pub cpu_done: vk::Semaphore,
}

/// A queue of a compute-only family (`Ctx::compute`) and the timeline its
/// batches signal with their sequence.
pub struct ComputeQueue {
    pub family: u32,
    pub queue: Mutex<vk::Queue>,
    pub batches: vk::Semaphore,
}

/// Ordered async compute is set up at boot (`DS_ASYNC_COMPUTE` 2 or more
/// when the device is made): the compute family's queue, the second
/// written-bit map, images shared between the families.
pub fn async_compute_boot() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| super::tune::async_compute() >= 2)
}

/// A transfer-only queue for batch-start copies, and its timeline.
pub struct CopyQueue {
    pub family: u32,
    pub queue: Mutex<vk::Queue>,
    /// Timeline semaphore: reaches a batch's sequence once the copies it
    /// made on this queue are done (signalled in sequence order).
    pub copied: vk::Semaphore,
}

static CTX: OnceLock<Ctx> = OnceLock::new();

thread_local! {
    /// What this thread is creating (`creating`), for the validation
    /// layer's messages.
    static CREATING: std::cell::RefCell<String> = const { std::cell::RefCell::new(String::new()) };
}

/// Notes what this thread creates until the guard drops, for the validation
/// layer's messages (only under it).
pub fn creating(what: impl FnOnce() -> String) -> Option<CreatingGuard> {
    try_ctx().filter(|c| c.validation)?;
    CREATING.with(|c| *c.borrow_mut() = what());
    Some(CreatingGuard)
}

pub struct CreatingGuard;

impl Drop for CreatingGuard {
    fn drop(&mut self) {
        CREATING.with(|c| c.borrow_mut().clear());
    }
}

/// Names an object for the validation layer's messages (only under it).
pub fn name_object<H: vk::Handle>(handle: H, name: &str) {
    if let Some(n) = ctx().names.as_ref() {
        if let Ok(name) = std::ffi::CString::new(name) {
            let _ = unsafe { n.set_debug_utils_object_name(&vk::DebugUtilsObjectNameInfoEXT::default().object_handle(handle).object_name(&name)) };
        }
    }
}

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
    // (The layer reports a module's SPIR-V while creating it, on the
    // creating thread: what that thread is creating, `creating`.)
    let what = CREATING.with(|c| c.borrow().clone());
    let msg = if what.is_empty() { msg.into_owned() } else { format!("{msg} [while creating {what}]") };
    if sev.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
        crate::error!("vulkan: {msg}");
    } else {
        crate::warn!("vulkan: {msg}");
    }
    vk::FALSE
}

/// Vulkan initialization step reached (an init watchdog names it: two boots
/// hung before initialization logged anything, the queues waiting for it).
static INIT_STAGE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);
const INIT_STAGES: [&str; 6] = ["loading the Vulkan loader", "creating the instance", "enumerating devices", "creating the device", "finishing", "done"];

fn init_stage(k: u8) {
    INIT_STAGE.store(k, std::sync::atomic::Ordering::SeqCst);
}

pub fn init() -> Result<(), String> {
    if let Some(n) = std::env::var("DS_LOOP_LIMIT").ok().and_then(|v| v.parse::<u32>().ok()).filter(|&n| n > 0) {
        rdna::translate::set_loop_limit(n);
        crate::info!("gpu: loop watchdog limit {n} blocks per invocation (DS_LOOP_LIMIT)");
    }
    if CTX.get().is_some() {
        return Ok(());
    }
    // Reports a driver call that does not return (every queue waits for init).
    let _ = std::thread::Builder::new().name("vk-init-watchdog".into()).spawn(|| {
        for s in 1..=12 {
            std::thread::sleep(std::time::Duration::from_secs(10));
            let k = INIT_STAGE.load(std::sync::atomic::Ordering::SeqCst) as usize;
            if k >= INIT_STAGES.len() - 1 {
                return;
            }
            crate::warn!("Vulkan initialization still {} after {} s", INIT_STAGES[k], 10 * s);
        }
    });
    let r = init_inner();
    init_stage(INIT_STAGES.len() as u8 - 1);
    r
}

/// The validation layer of a Vulkan SDK copied into the project
/// (local-data/research/vulkan-sdk/<version>/Bin: the LunarG installer's
/// `copy_only=1`, nothing registered with the system), the newest version.
fn local_layer_dir() -> Option<std::path::PathBuf> {
    let data = crate::run_dir()?.parent()?.parent()?.to_path_buf();
    let mut dirs: Vec<_> = std::fs::read_dir(data.join("research").join("vulkan-sdk")).ok()?.flatten().map(|e| e.path().join("Bin")).filter(|p| p.join("VkLayer_khronos_validation.json").is_file()).collect();
    dirs.sort();
    dirs.pop()
}

fn init_inner() -> Result<(), String> {
    unsafe {
        init_stage(0);
        // DS_VK_VALIDATION without a layer path of its own: the project's
        // SDK copy (`local_layer_dir`), for the loader to find (set before it
        // loads; this diagnosis only).
        if std::env::var_os("DS_VK_VALIDATION").is_some() && std::env::var_os("VK_ADD_LAYER_PATH").is_none() {
            if let Some(dir) = local_layer_dir() {
                crate::info!("Vulkan validation layer from {}", dir.display());
                std::env::set_var("VK_ADD_LAYER_PATH", &dir);
            }
        }
        let entry = ash::Entry::load().map_err(|e| format!("loading Vulkan: {e}"))?;
        init_stage(1);
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
        // Command labels for GPU Trace (DS_GPU_LABELS=1).
        let want_labels = std::env::var("DS_GPU_LABELS").ok().as_deref() == Some("1");
        let has_debug_utils = entry.enumerate_instance_extension_properties(None).unwrap_or_default().iter().any(|e| CStr::from_ptr(e.extension_name.as_ptr()) == ash::ext::debug_utils::NAME);
        let labels_on = want_labels && has_debug_utils;
        if want_labels && !has_debug_utils {
            crate::warn!("DS_GPU_LABELS=1 but VK_EXT_debug_utils is unavailable");
        }
        if validation || labels_on {
            exts.push(ash::ext::debug_utils::NAME.as_ptr());
        }
        let ici = vk::InstanceCreateInfo::default().application_info(&app).enabled_layer_names(&layers).enabled_extension_names(&exts);
        let instance = entry.create_instance(&ici, None).map_err(|e| format!("vkCreateInstance: {e}"))?;
        init_stage(2);
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
        // A transfer-only family for the copy queue (`DS_COPY_QUEUE=0`: none).
        let copy_family = if std::env::var("DS_COPY_QUEUE").ok().as_deref() != Some("0") {
            instance
                .get_physical_device_queue_family_properties(pdev)
                .iter()
                .position(|q| q.queue_count > 0 && q.queue_flags.contains(vk::QueueFlags::TRANSFER) && !q.queue_flags.intersects(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE))
                .map(|f| f as u32)
        } else {
            None
        };
        // A compute-only family (ordered async compute).
        let compute_family = if async_compute_boot() {
            instance
                .get_physical_device_queue_family_properties(pdev)
                .iter()
                .position(|q| q.queue_count > 0 && q.queue_flags.contains(vk::QueueFlags::COMPUTE) && !q.queue_flags.contains(vk::QueueFlags::GRAPHICS))
                .map(|f| f as u32)
        } else {
            None
        };
        let mut qci = vec![vk::DeviceQueueCreateInfo::default().queue_family_index(qf).queue_priorities(&prio)];
        if let Some(f) = copy_family {
            qci.push(vk::DeviceQueueCreateInfo::default().queue_family_index(f).queue_priorities(&prio));
        }
        if let Some(f) = compute_family {
            qci.push(vk::DeviceQueueCreateInfo::default().queue_family_index(f).queue_priorities(&prio));
        }
        let host_import = instance
            .enumerate_device_extension_properties(pdev)
            .unwrap_or_default()
            .iter()
            .any(|e| CStr::from_ptr(e.extension_name.as_ptr()) == ash::ext::external_memory_host::NAME);
        let mut dexts = vec![ash::khr::swapchain::NAME.as_ptr(), ash::khr::push_descriptor::NAME.as_ptr()];
        if host_import {
            dexts.push(ash::ext::external_memory_host::NAME.as_ptr());
        }
        let memory_budget = instance
            .enumerate_device_extension_properties(pdev)
            .unwrap_or_default()
            .iter()
            .any(|e| CStr::from_ptr(e.extension_name.as_ptr()) == ash::ext::memory_budget::NAME);
        if memory_budget {
            dexts.push(ash::ext::memory_budget::NAME.as_ptr());
        }
        let calibrated = instance
            .enumerate_device_extension_properties(pdev)
            .unwrap_or_default()
            .iter()
            .any(|e| CStr::from_ptr(e.extension_name.as_ptr()) == ash::ext::calibrated_timestamps::NAME)
            && ash::ext::calibrated_timestamps::Instance::new(&entry, &instance)
                .get_physical_device_calibrateable_time_domains(pdev)
                .is_ok_and(|d| d.contains(&vk::TimeDomainEXT::DEVICE) && d.contains(&vk::TimeDomainEXT::QUERY_PERFORMANCE_COUNTER));
        if calibrated {
            dexts.push(ash::ext::calibrated_timestamps::NAME.as_ptr());
        }
        // Device faults: the faulting addresses and the driver's description
        // after a device loss.
        let has_fault_ext = instance
            .enumerate_device_extension_properties(pdev)
            .unwrap_or_default()
            .iter()
            .any(|e| CStr::from_ptr(e.extension_name.as_ptr()) == ash::ext::device_fault::NAME);
        let mut fault_supported = vk::PhysicalDeviceFaultFeaturesEXT::default();
        if has_fault_ext {
            let mut f2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut fault_supported);
            instance.get_physical_device_features2(pdev, &mut f2);
        }
        let has_fault = has_fault_ext && fault_supported.device_fault != 0;
        let mut fault_f = vk::PhysicalDeviceFaultFeaturesEXT::default().device_fault(true).device_fault_vendor_binary(fault_supported.device_fault_vendor_binary != 0);
        if has_fault {
            dexts.push(ash::ext::device_fault::NAME.as_ptr());
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
        // Partitioned subgroup operations (write marks ORed per map word:
        // `rdna::translate::Spec::marks_part`).
        let partitioned = instance
            .enumerate_device_extension_properties(pdev)
            .unwrap_or_default()
            .iter()
            .any(|e| CStr::from_ptr(e.extension_name.as_ptr()) == ash::nv::shader_subgroup_partitioned::NAME);
        if partitioned {
            dexts.push(ash::nv::shader_subgroup_partitioned::NAME.as_ptr());
        }
        let mut bary = vk::PhysicalDeviceFragmentShaderBarycentricFeaturesKHR::default().fragment_shader_barycentric(true);
        if barycentric {
            dexts.push(ash::khr::fragment_shader_barycentric::NAME.as_ptr());
        }
        let first_instance = instance.get_physical_device_features(pdev).draw_indirect_first_instance != 0;
        // Sparse buffers for the residency mirror (`DS_RESIDENT=1`): the
        // device and the main queue's family must support them.
        let sparse_buffers = (super::resident::wanted() || super::fe2::wanted()) && {
            let f = instance.get_physical_device_features(pdev);
            let fam = instance.get_physical_device_queue_family_properties(pdev)[qf as usize].queue_flags;
            f.sparse_binding != 0 && f.sparse_residency_buffer != 0 && fam.contains(vk::QueueFlags::SPARSE_BINDING)
        };
        if (super::resident::wanted() || super::fe2::wanted()) && !sparse_buffers {
            crate::warn!("front end 2 (the default), DS_RESIDENT or DS_RGM wanted but the device or the main queue family has no sparse buffers");
        }
        // Storage images bound one view per mip level are indexed per
        // invocation (image_store_mip / image_load_mip).
        let core = instance.get_physical_device_features(pdev);
        let storage_dyn = core.shader_storage_image_array_dynamic_indexing != 0;
        // Gathers with offsets declare ImageGatherExtended (the validation
        // layer reported every such module without it).
        let features = vk::PhysicalDeviceFeatures::default()
            .shader_storage_image_array_dynamic_indexing(storage_dyn)
            .shader_image_gather_extended(core.shader_image_gather_extended != 0)
            .draw_indirect_first_instance(first_instance)
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
            .sample_rate_shading(true)
            .sparse_binding(sparse_buffers)
            .sparse_residency_buffer(sparse_buffers);
        // Image tables are indexed per invocation (non-uniform indexing).
        let mut f12_supported = vk::PhysicalDeviceVulkan12Features::default();
        let mut f11_supported = vk::PhysicalDeviceVulkan11Features::default();
        {
            let mut f2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut f12_supported).push_next(&mut f11_supported);
            instance.get_physical_device_features2(pdev, &mut f2);
        }
        let draw_parameters = f11_supported.shader_draw_parameters != 0;
        let mut f11 = vk::PhysicalDeviceVulkan11Features::default().shader_draw_parameters(draw_parameters);
        let nonuniform = f12_supported.shader_sampled_image_array_non_uniform_indexing != 0;
        if !nonuniform {
            crate::warn!("Vulkan: shaderSampledImageArrayNonUniformIndexing unsupported (image tables index uniformly)");
        }
        let storage_nonuniform = f12_supported.shader_storage_image_array_non_uniform_indexing != 0;
        if !storage_nonuniform || !storage_dyn {
            crate::warn!("Vulkan: storage image arrays without non-uniform indexing (programs that select mip levels per lane may fail or write wrong levels)");
        }
        let device_addresses = f12_supported.buffer_device_address != 0;
        let draw_indirect_count = f12_supported.draw_indirect_count != 0;
        let mut f12 = vk::PhysicalDeviceVulkan12Features::default()
            .buffer_device_address(device_addresses)
            .draw_indirect_count(draw_indirect_count)
            .shader_sampled_image_array_non_uniform_indexing(nonuniform)
            .shader_storage_image_array_non_uniform_indexing(storage_nonuniform)
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
            .push_next(&mut f11)
            .push_next(&mut f12)
            .push_next(&mut f13);
        if has_robust2 {
            dci = dci.push_next(&mut r2);
        }
        if barycentric {
            dci = dci.push_next(&mut bary);
        }
        if has_fault {
            dci = dci.push_next(&mut fault_f);
        }
        if has_stats {
            dci = dci.push_next(&mut stats_f);
        }
        init_stage(3);
        let device = instance.create_device(pdev, &dci, None).map_err(|e| format!("vkCreateDevice: {e}"))?;
        // Its command functions go through deferred recording's (`defer`).
        let device = defer::install(device);
        init_stage(4);
        crate::info!(
            "Vulkan robustness: robustBufferAccess, robustness2 {} (buffer {}, image {}, null descriptors {})",
            has_robust2,
            r2_supported.robust_buffer_access2 != 0,
            r2_supported.robust_image_access2 != 0,
            r2_supported.null_descriptor != 0
        );
        let queue = device.get_device_queue(qf, 0);
        let timeline = || {
            let mut tci = vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE).initial_value(0);
            device.create_semaphore(&vk::SemaphoreCreateInfo::default().push_next(&mut tci), None).map_err(|e| format!("timeline semaphore: {e}"))
        };
        let compute = match compute_family {
            Some(f) => Some(ComputeQueue { family: f, queue: Mutex::new(device.get_device_queue(f, 0)), batches: timeline()? }),
            None => None,
        };
        let g_progress = if compute.is_some() { timeline()? } else { vk::Semaphore::null() };
        if let Some(c) = &compute {
            crate::info!("Vulkan: compute queue in family {} (ordered async compute: the compute queues' batches beside the main queue)", c.family);
        } else if async_compute_boot() {
            crate::warn!("Vulkan: no compute-only queue family: ordered async compute off");
        }
        let mut families = vec![qf];
        for f in [copy_family, compute_family].into_iter().flatten() {
            if !families.contains(&f) {
                families.push(f);
            }
        }
        let copy = copy_family.and_then(|f| {
            let mut tci = vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE).initial_value(0);
            let copied = device.create_semaphore(&vk::SemaphoreCreateInfo::default().push_next(&mut tci), None).ok()?;
            Some(CopyQueue { family: f, queue: Mutex::new(device.get_device_queue(f, 0)), copied })
        });
        if let Some(c) = &copy {
            crate::info!("Vulkan: copy queue in family {} (batch-start copies beside the main queue)", c.family);
        }
        // Formats the title samples with depth comparison: which support it
        // (VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_DEPTH_COMPARISON_BIT).
        {
            let mut have = Vec::new();
            for f in [vk::Format::R16_UNORM, vk::Format::R32_SFLOAT, vk::Format::R16_SFLOAT, vk::Format::D16_UNORM, vk::Format::D32_SFLOAT] {
                let mut p3 = vk::FormatProperties3::default();
                let mut p2 = vk::FormatProperties2::default().push_next(&mut p3);
                instance.get_physical_device_format_properties2(pdev, f, &mut p2);
                let opt = p3.optimal_tiling_features.contains(vk::FormatFeatureFlags2::SAMPLED_IMAGE_DEPTH_COMPARISON);
                have.push(format!("{f:?} {opt}"));
            }
            crate::info!("Vulkan depth-comparison sampling support: {}", have.join(", "));
        }
        let mem_props = instance.get_physical_device_memory_properties(pdev);
        let max_push = {
            let mut pd = vk::PhysicalDevicePushDescriptorPropertiesKHR::default();
            let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut pd);
            instance.get_physical_device_properties2(pdev, &mut p2);
            pd.max_push_descriptors
        };
        let batches = {
            let mut tci = vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE).initial_value(0);
            device.create_semaphore(&vk::SemaphoreCreateInfo::default().push_next(&mut tci), None).map_err(|e| format!("batch timeline semaphore: {e}"))?
        };
        let cpu_done = {
            let mut tci = vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE).initial_value(0);
            device.create_semaphore(&vk::SemaphoreCreateInfo::default().push_next(&mut tci), None).map_err(|e| format!("effects timeline semaphore: {e}"))?
        };
        let ctx = Ctx {
            batches,
            cpu_done,
            surface_i: ash::khr::surface::Instance::new(&entry, &instance),
            win32_i: ash::khr::win32_surface::Instance::new(&entry, &instance),
            swapchain_d: ash::khr::swapchain::Device::new(&instance, &device),
            push_desc: ash::khr::push_descriptor::Device::new(&instance, &device),
            max_push,
            ext_host: host_import.then(|| ash::ext::external_memory_host::Device::new(&instance, &device)),
            checkpoints: has_ckpt.then(|| ash::nv::device_diagnostic_checkpoints::Device::new(&instance, &device)),
            exec_props: has_stats.then(|| ash::khr::pipeline_executable_properties::Device::new(&instance, &device)),
            calibrated: calibrated.then(|| ash::ext::calibrated_timestamps::Device::new(&instance, &device)),
            fault: has_fault.then(|| ash::ext::device_fault::Device::new(&instance, &device)),
            labels: labels_on.then(|| ash::ext::debug_utils::Device::new(&instance, &device)),
            names: validation.then(|| ash::ext::debug_utils::Device::new(&instance, &device)),
            device_addresses,
            draw_indirect_count,
            sparse_buffers,
            draw_parameters: draw_parameters && first_instance,
            families,
            copy,
            compute,
            g_progress,
            g_count: AtomicU64::new(0),
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
            partitioned,
            memory_budget,
        };
        crate::info!("Vulkan device: {} (queue family {qf}, validation {})", ctx.device_name, validation);
        {
            let exts = ctx.instance.enumerate_device_extension_properties(pdev).unwrap_or_default();
            let has = |n: &CStr| exts.iter().any(|e| CStr::from_ptr(e.extension_name.as_ptr()) == n);
            let mut host = vk::PhysicalDeviceExternalMemoryHostPropertiesEXT::default();
            let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut host);
            ctx.instance.get_physical_device_properties2(pdev, &mut p2);
            crate::info!(
                "Vulkan limits: maxStorageBufferRange {:#x}, minStorageBufferOffsetAlignment {}, external_memory_host {} (import alignment {:#x}), buffer_device_address {}, maxPushDescriptors {}, maxMemoryAllocationCount {}",
                ctx.limits.max_storage_buffer_range,
                ctx.limits.min_storage_buffer_offset_alignment,
                has(ash::ext::external_memory_host::NAME),
                host.min_imported_host_pointer_alignment,
                has(ash::khr::buffer_device_address::NAME),
                ctx.max_push,
                ctx.limits.max_memory_allocation_count
            );
        }
        CTX.set(ctx).ok();
        crate::info!("Vulkan memory heaps: {}", ctx_heaps());
        Ok(())
    }
}

impl Ctx {
    /// vkCmdPushDescriptorSetKHR, into `cmd`'s list when this thread
    /// claimed it (`defer`).
    pub unsafe fn push_descriptors(&self, cmd: vk::CommandBuffer, bp: vk::PipelineBindPoint, layout: vk::PipelineLayout, set: u32, writes: &[vk::WriteDescriptorSet]) {
        if !defer::push_descriptors(cmd, bp, layout, set, writes) {
            defer::direct(cmd);
            unsafe { self.push_desc.cmd_push_descriptor_set(cmd, bp, layout, set, writes) }
        }
    }

    /// Each memory heap: size, flags and (VK_EXT_memory_budget) this
    /// process's usage against its budget, MiB.
    pub fn heap_report(&self) -> String {
        let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
        let mut p2 = vk::PhysicalDeviceMemoryProperties2::default();
        if self.memory_budget {
            p2 = p2.push_next(&mut budget);
        }
        unsafe { self.instance.get_physical_device_memory_properties2(self.pdev, &mut p2) };
        let props = p2.memory_properties;
        (0..props.memory_heap_count as usize)
            .map(|i| {
                let h = props.memory_heaps[i];
                let local = if h.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL) { "device-local" } else { "host" };
                if self.memory_budget {
                    format!("heap {i} ({local}) {} MiB: used {} of budget {}", h.size >> 20, budget.heap_usage[i] >> 20, budget.heap_budget[i] >> 20)
                } else {
                    format!("heap {i} ({local}) {} MiB", h.size >> 20)
                }
            })
            .collect::<Vec<_>>()
            .join("; ")
    }
}

impl Ctx {
    /// This process's usage of the largest device-local heap and its budget,
    /// bytes (None without VK_EXT_memory_budget).
    pub fn device_budget(&self) -> Option<(u64, u64)> {
        if !self.memory_budget {
            return None;
        }
        let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
        let mut p2 = vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget);
        unsafe { self.instance.get_physical_device_memory_properties2(self.pdev, &mut p2) };
        let props = p2.memory_properties;
        let i = (0..props.memory_heap_count as usize).filter(|&i| props.memory_heaps[i].flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL)).max_by_key(|&i| props.memory_heaps[i].size)?;
        drop(p2);
        Some((budget.heap_usage[i], budget.heap_budget[i].min(vram_cap())))
    }

    /// The size of the largest device-local heap, bytes (at most
    /// `vram_cap`).
    pub fn device_heap_size(&self) -> u64 {
        let props = unsafe { self.instance.get_physical_device_memory_properties(self.pdev) };
        (0..props.memory_heap_count as usize).filter(|&i| props.memory_heaps[i].flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL)).map(|i| props.memory_heaps[i].size).max().unwrap_or(0).min(vram_cap())
    }
}

/// `DS_VRAM_CAP_MIB=<MiB>` (boot; simulation of a smaller GPU): the
/// device-local budget our decisions see (`Ctx::device_budget`: texture
/// eviction, and sizes derived from it) is at most this; allocations are
/// not limited by it.
pub fn vram_cap() -> u64 {
    static CAP: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *CAP.get_or_init(|| std::env::var("DS_VRAM_CAP_MIB").ok().and_then(|v| v.parse::<u64>().ok()).filter(|&m| m >= 1024).map_or(u64::MAX, |m| m << 20))
}

/// A share of the device-local budget seen at the first call (`1/div` of
/// it, at most `max`; `max` without VK_EXT_memory_budget): fixed pools
/// (the shared copy cache, video-memory homes) sized for the GPU at hand.
pub fn budget_share(div: u64, max: u64) -> u64 {
    static BUDGET: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    let b = *BUDGET.get_or_init(|| ctx().device_budget().map(|(_, b)| b));
    b.map_or(max, |b| (b / div).min(max))
}

/// `Ctx::heap_report` of the initialized context ("" before).
pub fn ctx_heaps() -> String {
    CTX.get().map(|c| c.heap_report()).unwrap_or_default()
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

/// The device offers partitioned subgroup operations
/// (VK_NV_shader_subgroup_partitioned, enabled when offered): write marks are
/// then ORed per map word (`rdna::translate::Spec::marks_part`; the 19
/// hottest compute programs' code -12.8%, GPU busy -6.2% at the vista and
/// -2.8% at Doran, frames +1.1-1.2%: FINDINGS.md 2026-11-06).
pub fn subgroup_partitioned() -> bool {
    try_ctx().is_some_and(|c| c.partitioned)
}

/// Command labels are recorded (DS_GPU_LABELS=1, `Exec::label`).
pub fn labels_on() -> bool {
    try_ctx().is_some_and(|c| c.labels.is_some())
}

/// Begins a named label range in `cmd` (DS_GPU_LABELS=1).
pub fn label_begin(cmd: vk::CommandBuffer, name: &str) {
    if let Some(l) = try_ctx().and_then(|c| c.labels.as_ref()) {
        if defer::label_begin(cmd, name) {
            return;
        }
        defer::direct(cmd);
        let s = std::ffi::CString::new(name).unwrap_or_default();
        unsafe { l.cmd_begin_debug_utils_label(cmd, &vk::DebugUtilsLabelEXT::default().label_name(&s)) };
    }
}

/// Ends the innermost label range begun in `cmd`.
pub fn label_end(cmd: vk::CommandBuffer) {
    if let Some(l) = try_ctx().and_then(|c| c.labels.as_ref()) {
        if defer::label_end(cmd) {
            return;
        }
        defer::direct(cmd);
        unsafe { l.cmd_end_debug_utils_label(cmd) };
    }
}

/// Records a diagnostic checkpoint (DS_GPU_CHECKPOINTS=1): `kind` and a
/// 56-bit value (shader or guest address, sequence number).
/// Diagnostic checkpoints are recorded (VK_NV_device_diagnostic_checkpoints).
pub fn checkpoints_on() -> bool {
    try_ctx().is_some_and(|c| c.checkpoints.is_some())
}

pub fn checkpoint(cmd: vk::CommandBuffer, kind: u64, value: u64) {
    if let Some(c) = try_ctx() {
        if let Some(cp) = &c.checkpoints {
            let marker = (kind << 56) | (value & ((1 << 56) - 1));
            if defer::checkpoint(cmd, marker) {
                return;
            }
            defer::direct(cmd);
            unsafe { cp.cmd_set_checkpoint(cmd, marker as *const std::ffi::c_void) };
        }
    }
}

/// GPU address ranges of guest imports and video-memory homes, for device
/// fault reports: live ones, dropped ones (no new work uses them) and the
/// last freed ones.
struct GpuRange {
    buf: u64,
    va: u64,
    size: u64,
    what: String,
    dropped_ms: Option<f64>,
    freed_ms: Option<f64>,
}

static GPU_RANGES: Mutex<Vec<GpuRange>> = Mutex::new(Vec::new());

/// Remembers buffer `buf`'s GPU address range (`va` 0: none) as `what`.
pub fn note_gpu_range(buf: vk::Buffer, va: u64, size: u64, what: impl FnOnce() -> String) {
    use ash::vk::Handle;
    if va == 0 {
        return;
    }
    GPU_RANGES.lock().push(GpuRange { buf: buf.as_raw(), va, size, what: what(), dropped_ms: None, freed_ms: None });
}

/// Buffer `buf` is no longer used for new work (freed once its batches end).
pub fn note_gpu_dropped(buf: vk::Buffer) {
    use ash::vk::Handle;
    let t = crate::log::elapsed_ms();
    if let Some(r) = GPU_RANGES.lock().iter_mut().rev().find(|r| r.buf == buf.as_raw() && r.freed_ms.is_none()) {
        r.dropped_ms.get_or_insert(t);
    }
}

/// Buffer `buf` was destroyed (its range is kept among the last freed).
pub fn note_gpu_freed(buf: vk::Buffer) {
    use ash::vk::Handle;
    let t = crate::log::elapsed_ms();
    let mut g = GPU_RANGES.lock();
    if let Some(r) = g.iter_mut().rev().find(|r| r.buf == buf.as_raw() && r.freed_ms.is_none()) {
        r.freed_ms = Some(t);
    }
    let freed = g.iter().filter(|r| r.freed_ms.is_some()).count();
    if freed > 4096 {
        let mut drop_n = freed - 4096;
        g.retain(|r| {
            if drop_n > 0 && r.freed_ms.is_some() {
                drop_n -= 1;
                false
            } else {
                true
            }
        });
    }
}

/// Where a GPU address lies among the noted ranges.
fn describe_address(a: u64) -> String {
    let g = GPU_RANGES.lock();
    let now = crate::log::elapsed_ms();
    let mut hits: Vec<String> = g
        .iter()
        .filter(|r| r.va <= a && a < r.va + r.size)
        .map(|r| {
            let state = match (r.dropped_ms, r.freed_ms) {
                (_, Some(f)) => format!("FREED {:.0} ms before the report", now - f),
                (Some(d), None) => format!("dropped {:.0} ms before the report, not freed", now - d),
                (None, None) => "live".into(),
            };
            format!("{} +{:#x} ({state})", r.what, a - r.va)
        })
        .collect();
    if hits.is_empty() {
        // The nearest noted ranges below and above.
        let below = g.iter().filter(|r| r.va + r.size <= a).max_by_key(|r| r.va + r.size).map(|r| format!("{:#x} past the end of {}", a - (r.va + r.size), r.what));
        let above = g.iter().filter(|r| r.va > a).min_by_key(|r| r.va).map(|r| format!("{:#x} before {}", r.va - a, r.what));
        hits.push(format!("in no noted range ({} ranges noted; {})", g.len(), [below, above].into_iter().flatten().collect::<Vec<_>>().join("; ")));
    }
    hits.join(" | ")
}

/// The device's account of a loss (VK_EXT_device_fault): description,
/// faulting addresses with the noted ranges they fall in, vendor codes; the
/// vendor binary goes to `gpu-fault.bin` in the run directory. Computed once.
pub fn fault_report() -> String {
    static REPORT: OnceLock<String> = OnceLock::new();
    REPORT
        .get_or_init(|| {
            let Some(c) = try_ctx() else { return String::new() };
            let Some(f) = &c.fault else { return "(no VK_EXT_device_fault)".into() };
            unsafe {
                let mut counts = vk::DeviceFaultCountsEXT::default();
                let r = (f.fp().get_device_fault_info_ext)(c.device.handle(), &mut counts, std::ptr::null_mut());
                if r != vk::Result::SUCCESS && r != vk::Result::INCOMPLETE {
                    return format!("vkGetDeviceFaultInfoEXT: {r}");
                }
                let mut addrs = vec![vk::DeviceFaultAddressInfoEXT::default(); counts.address_info_count as usize];
                let mut vendors = vec![vk::DeviceFaultVendorInfoEXT::default(); counts.vendor_info_count as usize];
                let mut bin = vec![0u8; counts.vendor_binary_size as usize];
                let mut info = vk::DeviceFaultInfoEXT::default();
                info.p_address_infos = if addrs.is_empty() { std::ptr::null_mut() } else { addrs.as_mut_ptr() };
                info.p_vendor_infos = if vendors.is_empty() { std::ptr::null_mut() } else { vendors.as_mut_ptr() };
                info.p_vendor_binary_data = if bin.is_empty() { std::ptr::null_mut() } else { bin.as_mut_ptr().cast() };
                let r = (f.fp().get_device_fault_info_ext)(c.device.handle(), &mut counts, &mut info);
                if r != vk::Result::SUCCESS && r != vk::Result::INCOMPLETE {
                    return format!("vkGetDeviceFaultInfoEXT: {r}");
                }
                let desc = CStr::from_ptr(info.description.as_ptr()).to_string_lossy().into_owned();
                let mut out = vec![format!("device fault: '{desc}'")];
                for a in addrs.iter().take(counts.address_info_count as usize) {
                    out.push(format!("{:?} at {:#x} (precision {:#x}): {}", a.address_type, a.reported_address, a.address_precision, describe_address(a.reported_address)));
                }
                for v in vendors.iter().take(counts.vendor_info_count as usize) {
                    let d = CStr::from_ptr(v.description.as_ptr()).to_string_lossy().into_owned();
                    out.push(format!("vendor '{d}' code {:#x} data {:#x}", v.vendor_fault_code, v.vendor_fault_data));
                }
                if !bin.is_empty() {
                    if let Some(dir) = crate::run_dir() {
                        let p = dir.join("gpu-fault.bin");
                        if std::fs::write(&p, &bin).is_ok() {
                            out.push(format!("vendor binary ({} bytes): {}", bin.len(), p.display()));
                        }
                    }
                }
                out.join("; ")
            }
        })
        .clone()
}

/// The queue's last checkpoints per pipeline stage, after a device loss
/// (and the device's fault report).
pub fn checkpoint_report() -> String {
    let fault = fault_report();
    let Some(c) = try_ctx() else { return String::new() };
    let Some(cp) = &c.checkpoints else { return format!("(DS_GPU_CHECKPOINTS=1 names the failing command); {fault}") };
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
            .chain(std::iter::once(fault))
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

    /// `alloc` for buffer memory: with bufferDeviceAddress the buffers bound
    /// to it may be addressed by shaders (`buffer_address`).
    pub fn alloc_buffer_mem(&self, req: vk::MemoryRequirements, want: vk::MemoryPropertyFlags) -> Result<vk::DeviceMemory, String> {
        if !self.device_addresses {
            return self.alloc(req, want);
        }
        let ty = self.memory_type(req.memory_type_bits, want).ok_or("no suitable memory type")?;
        let mut fl = vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        let ai = vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(ty).push_next(&mut fl);
        unsafe { self.device.allocate_memory(&ai, None) }.map_err(|e| format!("vkAllocateMemory({}): {e}", req.size))
    }

    /// Usage flags for a buffer that shaders may address (`Spec::addresses`).
    pub fn addressable(&self, usage: vk::BufferUsageFlags) -> vk::BufferUsageFlags {
        if self.device_addresses { usage | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS } else { usage }
    }

    /// The device address of `buf` (0 without bufferDeviceAddress). Every
    /// buffer our helpers create is addressable.
    #[inline]
    pub fn buffer_address(&self, buf: vk::Buffer) -> u64 {
        if !self.device_addresses {
            return 0;
        }
        unsafe { self.device.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buf)) }
    }

    /// Device-local buffer with a GPU address (0 without bufferDeviceAddress):
    /// video-memory homes, named in device fault reports.
    pub fn device_buffer_addressed(&self, size: u64, usage: vk::BufferUsageFlags) -> Result<(vk::Buffer, vk::DeviceMemory, u64), String> {
        if !self.device_addresses {
            let (b, m) = self.device_buffer(size, usage)?;
            return Ok((b, m, 0));
        }
        unsafe {
            let bi = self.shared(vk::BufferCreateInfo::default().size(size).usage(usage | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS));
            let buf = self.device.create_buffer(&bi, None).map_err(|e| format!("vkCreateBuffer: {e}"))?;
            let req = self.device.get_buffer_memory_requirements(buf);
            let Some(ty) = self.memory_type(req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL) else {
                self.device.destroy_buffer(buf, None);
                return Err("no suitable memory type".into());
            };
            let mut fl = vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
            let ai = vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(ty).push_next(&mut fl);
            let mem = match self.device.allocate_memory(&ai, None) {
                Ok(m) => m,
                Err(e) => {
                    self.device.destroy_buffer(buf, None);
                    return Err(format!("vkAllocateMemory({}): {e}", req.size));
                }
            };
            if let Err(e) = self.device.bind_buffer_memory(buf, mem, 0) {
                self.device.destroy_buffer(buf, None);
                self.device.free_memory(mem, None);
                return Err(format!("bind buffer: {e}"));
            }
            let va = self.device.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buf));
            Ok((buf, mem, va))
        }
    }

    /// `bi` with the buffer shared by the families of the main, copy and
    /// compute queues (concurrent; exclusive with the main queue's alone).
    fn shared<'a>(&'a self, bi: vk::BufferCreateInfo<'a>) -> vk::BufferCreateInfo<'a> {
        if self.families.len() > 1 {
            bi.sharing_mode(vk::SharingMode::CONCURRENT).queue_family_indices(&self.families)
        } else {
            bi.sharing_mode(vk::SharingMode::EXCLUSIVE)
        }
    }

    /// Device-local buffer (not host-visible).
    pub fn device_buffer(&self, size: u64, usage: vk::BufferUsageFlags) -> Result<(vk::Buffer, vk::DeviceMemory), String> {
        unsafe {
            let bi = self.shared(vk::BufferCreateInfo::default().size(size).usage(self.addressable(usage)));
            let buf = self.device.create_buffer(&bi, None).map_err(|e| format!("vkCreateBuffer: {e}"))?;
            let req = self.device.get_buffer_memory_requirements(buf);
            let mem = self.alloc_buffer_mem(req, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
            self.device.bind_buffer_memory(buf, mem, 0).map_err(|e| format!("bind buffer: {e}"))?;
            Ok((buf, mem))
        }
    }

    /// Imports existing host memory [ptr, ptr+size) (page aligned) as a GPU
    /// buffer (VK_EXT_external_memory_host): the GPU reads and writes it in place.
    /// The third value is the buffer's GPU address (0 without bufferDeviceAddress).
    pub fn import_host(&self, ptr: u64, size: u64, usage: vk::BufferUsageFlags) -> Result<(vk::Buffer, vk::DeviceMemory, u64), String> {
        let usage = if self.device_addresses { usage | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS } else { usage };
        let ext = self.ext_host.as_ref().ok_or("VK_EXT_external_memory_host not available")?;
        let ht = vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT;
        unsafe {
            let mut props = vk::MemoryHostPointerPropertiesEXT::default();
            let r = (ext.fp().get_memory_host_pointer_properties_ext)(self.device.handle(), ht, ptr as *const std::ffi::c_void, &mut props);
            if r != vk::Result::SUCCESS {
                return Err(format!("vkGetMemoryHostPointerPropertiesEXT: {r}"));
            }
            let mut ebi = vk::ExternalMemoryBufferCreateInfo::default().handle_types(ht);
            let bi = self.shared(vk::BufferCreateInfo::default().size(size).usage(usage)).push_next(&mut ebi);
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
            let mut fl = vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
            let ai = vk::MemoryAllocateInfo::default().allocation_size(size).memory_type_index(ty).push_next(&mut imp);
            let ai = if self.device_addresses { ai.push_next(&mut fl) } else { ai };
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
            let va = if self.device_addresses { self.device.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buf)) } else { 0 };
            Ok((buf, mem, va))
        }
    }

    /// Device-local buffer the CPU writes through its mapping (the PCIe BAR
    /// window; writes are write-combined, reads by the CPU are very slow).
    /// GPU reads stay in video memory, unlike `host_buffer`.
    pub fn bar_buffer(&self, size: u64, usage: vk::BufferUsageFlags) -> Result<(vk::Buffer, vk::DeviceMemory, *mut u8), String> {
        unsafe {
            let bi = self.shared(vk::BufferCreateInfo::default().size(size).usage(self.addressable(usage)));
            let buf = self.device.create_buffer(&bi, None).map_err(|e| format!("vkCreateBuffer: {e}"))?;
            let req = self.device.get_buffer_memory_requirements(buf);
            let want = vk::MemoryPropertyFlags::DEVICE_LOCAL | vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
            let mem = match self.alloc_buffer_mem(req, want) {
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
            let bi = self.shared(vk::BufferCreateInfo::default().size(size).usage(self.addressable(usage)));
            let buf = self.device.create_buffer(&bi, None).map_err(|e| format!("vkCreateBuffer: {e}"))?;
            let req = self.device.get_buffer_memory_requirements(buf);
            // Prefer cached host memory: results are read back by the CPU
            // (buffer write-back), which is very slow from write-combined memory.
            let hv = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
            let mem = self.alloc_buffer_mem(req, hv | vk::MemoryPropertyFlags::HOST_CACHED).or_else(|_| self.alloc_buffer_mem(req, hv))?;
            self.device.bind_buffer_memory(buf, mem, 0).map_err(|e| format!("bind buffer: {e}"))?;
            let ptr = self.device.map_memory(mem, 0, size, vk::MemoryMapFlags::empty()).map_err(|e| format!("map: {e}"))? as *mut u8;
            Ok((buf, mem, ptr))
        }
    }

    /// A GPU timestamp and the performance counter read together (None
    /// without VK_EXT_calibrated_timestamps).
    pub fn calibrate(&self) -> Option<(u64, i64)> {
        let cal = self.calibrated.as_ref()?;
        let info = [vk::CalibratedTimestampInfoEXT::default().time_domain(vk::TimeDomainEXT::DEVICE), vk::CalibratedTimestampInfoEXT::default().time_domain(vk::TimeDomainEXT::QUERY_PERFORMANCE_COUNTER)];
        let (t, _) = unsafe { cal.get_calibrated_timestamps(&info) }.ok()?;
        Some((t[0], t[1] as i64))
    }

    pub fn submit(&self, cmd: vk::CommandBuffer, wait: &[(vk::Semaphore, vk::PipelineStageFlags)], signal: &[vk::Semaphore], fence: vk::Fence) -> Result<(), String> {
        self.submit_cmds(&[cmd], wait, signal, fence)
    }

    /// Submits command buffers to the main queue, or the copy queue, with
    /// timeline semaphore waits (semaphore, value, stages that wait) and
    /// signals (semaphore, value).
    pub fn submit_timeline(&self, on_copy: bool, cmds: &[vk::CommandBuffer], waits: &[(vk::Semaphore, u64, vk::PipelineStageFlags2)], signals: &[(vk::Semaphore, u64)], fence: vk::Fence) -> Result<(), String> {
        self.submit_timeline_on(on_copy, false, cmds, waits, signals, fence)
    }

    /// `submit_timeline`, on the compute family's queue when `on_compute`
    /// (ordered async compute).
    pub fn submit_timeline_on(&self, on_copy: bool, on_compute: bool, cmds: &[vk::CommandBuffer], waits: &[(vk::Semaphore, u64, vk::PipelineStageFlags2)], signals: &[(vk::Semaphore, u64)], fence: vk::Fence) -> Result<(), String> {
        let cbs: Vec<vk::CommandBufferSubmitInfo> = cmds.iter().map(|&c| vk::CommandBufferSubmitInfo::default().command_buffer(c)).collect();
        let ws: Vec<vk::SemaphoreSubmitInfo> = waits.iter().map(|&(s, v, st)| vk::SemaphoreSubmitInfo::default().semaphore(s).value(v).stage_mask(st)).collect();
        let mut ss: Vec<vk::SemaphoreSubmitInfo> = signals.iter().map(|&(s, v)| vk::SemaphoreSubmitInfo::default().semaphore(s).value(v).stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)).collect();
        if on_copy {
            let q = self.copy.as_ref().ok_or("no copy queue")?.queue.lock();
            let si = vk::SubmitInfo2::default().command_buffer_infos(&cbs).wait_semaphore_infos(&ws).signal_semaphore_infos(&ss);
            return unsafe { self.device.queue_submit2(*q, &[si], fence) }.map_err(|e| format!("vkQueueSubmit2: {e}"));
        }
        if on_compute {
            let q = self.compute.as_ref().ok_or("no compute queue")?.queue.lock();
            let si = vk::SubmitInfo2::default().command_buffer_infos(&cbs).wait_semaphore_infos(&ws).signal_semaphore_infos(&ss);
            return unsafe { self.device.queue_submit2(*q, &[si], fence) }.map_err(|e| format!("vkQueueSubmit2 (compute queue): {e}"));
        }
        let q = self.queue.lock();
        ss.extend(self.progress_signal());
        let si = vk::SubmitInfo2::default().command_buffer_infos(&cbs).wait_semaphore_infos(&ws).signal_semaphore_infos(&ss);
        unsafe { self.device.queue_submit2(*q, &[si], fence) }.map_err(|e| format!("vkQueueSubmit2: {e}"))
    }

    /// With the compute family's queue: the main queue's progress signal
    /// for a submission to it, made under its lock (values in submission
    /// order).
    fn progress_signal(&self) -> Option<vk::SemaphoreSubmitInfo<'static>> {
        self.compute.as_ref()?;
        let v = self.g_count.fetch_add(1, Ordering::AcqRel) + 1;
        Some(vk::SemaphoreSubmitInfo::default().semaphore(self.g_progress).value(v).stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS))
    }

    /// The main queue's progress value of its latest submission (a batch
    /// on the compute family's queue waits for it).
    pub fn g_progress_value(&self) -> u64 {
        self.g_count.load(Ordering::Acquire)
    }

    /// Submits batches to the main queue in one call, in order: each its
    /// command buffers, an optional wait for the copy queue's timeline value,
    /// an optional wait for the compute family's queue's timeline value
    /// (ordered async compute: the compute batch it depends on), and a
    /// signal of `batches` with its sequence; `fence` on the call (the last
    /// batch's).
    pub fn submit_batches(&self, batches: &[(&[vk::CommandBuffer], u64, u64, u64)], fence: vk::Fence) -> Result<(), String> {
        let cbs: Vec<Vec<vk::CommandBufferSubmitInfo>> = batches.iter().map(|b| b.0.iter().map(|&c| vk::CommandBufferSubmitInfo::default().command_buffer(c)).collect()).collect();
        let copied = self.copy.as_ref().map(|q| q.copied);
        let cq = self.compute.as_ref().map(|q| q.batches);
        let ws: Vec<Vec<vk::SemaphoreSubmitInfo>> = batches
            .iter()
            .map(|b| {
                let mut w = Vec::new();
                if let (Some(s), true) = (copied, b.1 > 0) {
                    w.push(vk::SemaphoreSubmitInfo::default().semaphore(s).value(b.1).stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS));
                }
                if let (Some(s), true) = (cq, b.3 > 0) {
                    w.push(vk::SemaphoreSubmitInfo::default().semaphore(s).value(b.3).stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS));
                }
                w
            })
            .collect();
        let q = self.queue.lock();
        let ss: Vec<Vec<vk::SemaphoreSubmitInfo>> = (0..batches.len())
            .map(|k| {
                let mut v = vec![vk::SemaphoreSubmitInfo::default().semaphore(self.batches).value(batches[k].2).stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
                // (The progress signal on the last: its scope holds all
                // earlier submissions.)
                if k + 1 == batches.len() {
                    v.extend(self.progress_signal());
                }
                v
            })
            .collect();
        let infos: Vec<vk::SubmitInfo2> = (0..batches.len()).map(|k| vk::SubmitInfo2::default().command_buffer_infos(&cbs[k]).wait_semaphore_infos(&ws[k]).signal_semaphore_infos(&ss[k])).collect();
        unsafe { self.device.queue_submit2(*q, &infos, fence) }.map_err(|e| format!("vkQueueSubmit2 ({} batches): {e}", batches.len()))
    }

    /// Waits until the main queue's batch `seq` completed (`batches`);
    /// false on a timeout or error.
    /// The batch timeline's value now: the highest batch sequence the main
    /// queue has completed (every earlier one of its batches too).
    pub fn batch_value(&self) -> u64 {
        unsafe { self.device.get_semaphore_counter_value(self.batches) }.unwrap_or(0)
    }

    pub fn wait_batch(&self, seq: u64, timeout_ns: u64) -> bool {
        self.wait_batch_result(seq, timeout_ns).is_ok()
    }

    /// `wait_batch` telling a timeout (`vk::Result::TIMEOUT`) from an error.
    pub fn wait_batch_result(&self, seq: u64, timeout_ns: u64) -> Result<(), vk::Result> {
        let s = [self.batches];
        let v = [seq];
        unsafe { self.device.wait_semaphores(&vk::SemaphoreWaitInfo::default().semaphores(&s).values(&v), timeout_ns) }
    }

    /// The compute family's queue's batch timeline now (the highest batch
    /// sequence it completed, every earlier one of its batches too).
    pub fn batch_value_c(&self) -> u64 {
        self.compute.as_ref().map_or(0, |q| unsafe { self.device.get_semaphore_counter_value(q.batches) }.unwrap_or(0))
    }

    /// `wait_batch` for a batch on the compute family's queue.
    pub fn wait_batch_c(&self, seq: u64, timeout_ns: u64) -> bool {
        self.wait_batch_c_result(seq, timeout_ns).is_ok()
    }

    /// `wait_batch_c` telling a timeout (`vk::Result::TIMEOUT`) from an error.
    pub fn wait_batch_c_result(&self, seq: u64, timeout_ns: u64) -> Result<(), vk::Result> {
        let Some(q) = self.compute.as_ref() else { return Err(vk::Result::ERROR_UNKNOWN) };
        let s = [q.batches];
        let v = [seq];
        unsafe { self.device.wait_semaphores(&vk::SemaphoreWaitInfo::default().semaphores(&s).values(&v), timeout_ns) }
    }

    /// Submits command buffers that execute in order (binary semaphores).
    pub fn submit_cmds(&self, cmds: &[vk::CommandBuffer], wait: &[(vk::Semaphore, vk::PipelineStageFlags)], signal: &[vk::Semaphore], fence: vk::Fence) -> Result<(), String> {
        let cbs: Vec<vk::CommandBufferSubmitInfo> = cmds.iter().map(|&c| vk::CommandBufferSubmitInfo::default().command_buffer(c)).collect();
        let ws: Vec<vk::SemaphoreSubmitInfo> = wait.iter().map(|&(s, st)| vk::SemaphoreSubmitInfo::default().semaphore(s).stage_mask(vk::PipelineStageFlags2::from_raw(st.as_raw() as u64))).collect();
        let mut ss: Vec<vk::SemaphoreSubmitInfo> = signal.iter().map(|&s| vk::SemaphoreSubmitInfo::default().semaphore(s).stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)).collect();
        let q = self.queue.lock();
        ss.extend(self.progress_signal());
        let si = vk::SubmitInfo2::default().command_buffer_infos(&cbs).wait_semaphore_infos(&ws).signal_semaphore_infos(&ss);
        unsafe { self.device.queue_submit2(*q, &[si], fence) }.map_err(|e| format!("vkQueueSubmit2: {e}"))
    }
}
