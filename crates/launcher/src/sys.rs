//! What the launcher reads about the PC: the graphics card the game would use (the runtime's
//! own choice: the last discrete Vulkan device with graphics and compute, else the first), its
//! video memory and driver; the memory and processor threads; whether a process still runs.

use ash::vk;

#[derive(Clone, Debug, Default)]
pub struct Gpu {
    pub name: String,
    /// The driver's version as Vulkan reports it (compared with the prepared shaders').
    pub driver: u32,
    /// Shown to the player (NVIDIA's numbering on NVIDIA cards).
    pub driver_text: String,
    pub vendor: u32,
    /// Device-local memory in bytes.
    pub vram: u64,
    pub api: u32,
    /// VK_KHR_fragment_shader_barycentric with its feature (the game's shaders use it).
    pub barycentric: bool,
    /// Geometry shaders (the game's stand-in without barycentrics).
    pub geometry: bool,
}

pub const VENDOR_NVIDIA: u32 = 0x10DE;
pub const VENDOR_AMD: u32 = 0x1002;
pub const VENDOR_INTEL: u32 = 0x8086;

fn driver_text(vendor: u32, v: u32) -> String {
    if vendor == VENDOR_NVIDIA {
        format!("{}.{:02}", (v >> 22) & 0x3FF, (v >> 14) & 0xFF)
    } else if vendor == VENDOR_INTEL {
        format!("{}.{}", v >> 14, v & 0x3FFF)
    } else {
        format!("{}.{}.{}", vk::api_version_major(v), vk::api_version_minor(v), vk::api_version_patch(v))
    }
}

/// The game's graphics card, or why there is none.
pub fn gpu() -> Result<Gpu, String> {
    unsafe {
        let entry = ash::Entry::load().map_err(|_| "Vulkan is not installed. Install or update the graphics driver.".to_string())?;
        let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3);
        let instance = entry.create_instance(&vk::InstanceCreateInfo::default().application_info(&app), None).map_err(|e| format!("Vulkan could not start ({e}). Update the graphics driver."))?;
        let mut chosen: Option<Gpu> = None;
        for pd in instance.enumerate_physical_devices().unwrap_or_default() {
            let p = instance.get_physical_device_properties(pd);
            let qfs = instance.get_physical_device_queue_family_properties(pd);
            if !qfs.iter().any(|q| q.queue_flags.contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)) {
                continue;
            }
            let discrete = p.device_type == vk::PhysicalDeviceType::DISCRETE_GPU;
            if chosen.is_none() || discrete {
                let m = instance.get_physical_device_memory_properties(pd);
                let vram = m.memory_heaps[..m.memory_heap_count as usize].iter().filter(|h| h.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL)).map(|h| h.size).max().unwrap_or(0);
                let has_ext = instance.enumerate_device_extension_properties(pd).unwrap_or_default().iter().any(|e| std::ffi::CStr::from_ptr(e.extension_name.as_ptr()) == ash::khr::fragment_shader_barycentric::NAME);
                let barycentric = has_ext && {
                    let mut b = vk::PhysicalDeviceFragmentShaderBarycentricFeaturesKHR::default();
                    let mut f2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut b);
                    instance.get_physical_device_features2(pd, &mut f2);
                    b.fragment_shader_barycentric != 0
                };
                let geometry = instance.get_physical_device_features(pd).geometry_shader != 0;
                chosen = Some(Gpu {
                    name: std::ffi::CStr::from_ptr(p.device_name.as_ptr()).to_string_lossy().into_owned(),
                    driver: p.driver_version,
                    driver_text: driver_text(p.vendor_id, p.driver_version),
                    vendor: p.vendor_id,
                    vram,
                    api: p.api_version,
                    barycentric,
                    geometry,
                });
            }
        }
        instance.destroy_instance(None);
        chosen.ok_or_else(|| "No graphics card with Vulkan was found. Update the graphics driver.".to_string())
    }
}

/// The installed memory in bytes.
pub fn memory() -> u64 {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    let mut m: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    m.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
    if unsafe { GlobalMemoryStatusEx(&mut m) } == 0 { 0 } else { m.ullTotalPhys }
}

pub fn threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4)
}

/// Shader preparation threads: all logical processors but two, at most one per 1.5 GB of memory
/// beyond 6 GB (each compile holds the driver's memory), 2 to 12.
pub fn prepare_threads() -> usize {
    let gb = memory() as f64 / (1u64 << 30) as f64;
    let by_mem = ((gb - 6.0) / 1.5).floor().max(2.0) as usize;
    threads().saturating_sub(2).min(by_mem).clamp(2, 12)
}

pub fn process_alive(pid: u32) -> bool {
    use windows_sys::Win32::Foundation::{CloseHandle, STILL_ACTIVE};
    use windows_sys::Win32::System::Threading::{GetExitCodeProcess, OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION};
    unsafe {
        let h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
        if h.is_null() {
            return false;
        }
        let mut code = 0u32;
        let ok = GetExitCodeProcess(h, &mut code) != 0 && code == STILL_ACTIVE as u32;
        CloseHandle(h);
        ok
    }
}

/// Free space on the drive holding `p` (bytes), if known.
pub fn free_space(p: &std::path::Path) -> Option<u64> {
    use windows_sys::Win32::Storage::FileSystem::GetDiskFreeSpaceExW;
    let mut dir = p.to_path_buf();
    while !dir.exists() {
        dir = dir.parent()?.to_path_buf();
    }
    let w = crate::ui::wide(&dir.display().to_string());
    let mut free = 0u64;
    (unsafe { GetDiskFreeSpaceExW(w.as_ptr(), &mut free, std::ptr::null_mut(), std::ptr::null_mut()) } != 0).then_some(free)
}

/// Bytes as GB with one decimal.
pub fn gb(b: u64) -> String {
    format!("{:.1} GB", b as f64 / (1u64 << 30) as f64)
}
