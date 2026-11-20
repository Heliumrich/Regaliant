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

/// The installed memory in bytes (`DS_SIM_MEMORY_GB=<n>`, a test: that much).
pub fn memory() -> u64 {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    if let Some(gb) = std::env::var("DS_SIM_MEMORY_GB").ok().and_then(|v| v.parse::<f64>().ok()) {
        return (gb * (1u64 << 30) as f64) as u64;
    }
    let mut m: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    m.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
    if unsafe { GlobalMemoryStatusEx(&mut m) } == 0 { 0 } else { m.ullTotalPhys }
}

/// The logical processors (`DS_SIM_THREADS=<n>`, a test: that many).
pub fn threads() -> usize {
    if let Some(n) = std::env::var("DS_SIM_THREADS").ok().and_then(|v| v.parse::<usize>().ok()).filter(|&n| n > 0) {
        return n;
    }
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4)
}

/// Shader preparation threads: all logical processors but two, at most one per 1.5 GB of memory
/// beyond 6 GB (each compile holds the driver's memory), 2 to 12.
pub fn prepare_threads() -> usize {
    let gb = memory() as f64 / (1u64 << 30) as f64;
    let by_mem = ((gb - 6.0) / 1.5).floor().max(2.0) as usize;
    threads().saturating_sub(2).min(by_mem).clamp(2, 12)
}

/// The launcher's job for the shader preparation's steps (created at the first): it ends its
/// processes when the launcher ends, however it ends (a launcher that crashed or was ended had
/// left a hidden step running for hours, holding the game's data folder: the review of
/// 2026-11-20). The launcher itself and the game started by Play are not in it.
static JOB: std::sync::OnceLock<isize> = std::sync::OnceLock::new();

fn job() -> Option<windows_sys::Win32::Foundation::HANDLE> {
    use windows_sys::Win32::System::JobObjects::{CreateJobObjectW, JobObjectExtendedLimitInformation, SetInformationJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE};
    let h = *JOB.get_or_init(|| unsafe {
        let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
        if job.is_null() {
            return 0;
        }
        let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        if SetInformationJobObject(job, JobObjectExtendedLimitInformation, &info as *const _ as *const std::ffi::c_void, std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32) == 0 {
            return 0;
        }
        job as isize
    });
    (h != 0).then_some(h as windows_sys::Win32::Foundation::HANDLE)
}

/// Starts `cmd` (its creation flags given, `CREATE_SUSPENDED` added here) in the launcher's job,
/// then lets it run: its own processes (the hosted game) are in the job from their start.
pub fn spawn_in_job(cmd: &mut std::process::Command, flags: u32) -> std::io::Result<std::process::Child> {
    use std::os::windows::io::AsRawHandle;
    use std::os::windows::process::CommandExt;
    const CREATE_SUSPENDED: u32 = 0x0000_0004;
    let mut child = cmd.creation_flags(flags | CREATE_SUSPENDED).spawn()?;
    if let Some(j) = job() {
        unsafe { windows_sys::Win32::System::JobObjects::AssignProcessToJobObject(j, child.as_raw_handle() as windows_sys::Win32::Foundation::HANDLE) };
    }
    if resume_threads(child.id()) == 0 {
        let _ = child.kill();
        return Err(std::io::Error::other("its process could not be resumed"));
    }
    Ok(child)
}

/// Resumes the threads of process `pid` (a process created suspended has one); how many.
fn resume_threads(pid: u32) -> u32 {
    use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::System::Diagnostics::ToolHelp::{CreateToolhelp32Snapshot, Thread32First, Thread32Next, TH32CS_SNAPTHREAD, THREADENTRY32};
    use windows_sys::Win32::System::Threading::{OpenThread, ResumeThread, THREAD_SUSPEND_RESUME};
    let mut n = 0;
    unsafe {
        let snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
        if snap == INVALID_HANDLE_VALUE {
            return 0;
        }
        let mut te: THREADENTRY32 = std::mem::zeroed();
        te.dwSize = std::mem::size_of::<THREADENTRY32>() as u32;
        let mut ok = Thread32First(snap, &mut te) != 0;
        while ok {
            if te.th32OwnerProcessID == pid {
                let t = OpenThread(THREAD_SUSPEND_RESUME, 0, te.th32ThreadID);
                if !t.is_null() {
                    if ResumeThread(t) != u32::MAX {
                        n += 1;
                    }
                    CloseHandle(t);
                }
            }
            ok = Thread32Next(snap, &mut te) != 0;
        }
        CloseHandle(snap);
    }
    n
}

/// Keeps the PC from sleeping while the shader preparation runs (`on`), or lets it again (the
/// steps keep it awake themselves too; between steps only the launcher runs).
pub fn keep_awake(on: bool) {
    use windows_sys::Win32::System::Power::{SetThreadExecutionState, ES_CONTINUOUS, ES_SYSTEM_REQUIRED};
    unsafe { SetThreadExecutionState(if on { ES_CONTINUOUS | ES_SYSTEM_REQUIRED } else { ES_CONTINUOUS }) };
}

/// Waits on a thread of its own for process `pid` to end, then posts `msg` with its exit status
/// to `hwnd` (a game the launcher found running at its start: Play comes back when it ends).
pub fn watch_process(pid: u32, hwnd: usize, msg: u32) {
    std::thread::spawn(move || unsafe {
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::Threading::{GetExitCodeProcess, OpenProcess, WaitForSingleObject, INFINITE, PROCESS_QUERY_LIMITED_INFORMATION};
        const SYNCHRONIZE: u32 = 0x0010_0000;
        let h = OpenProcess(SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
        if h.is_null() {
            return;
        }
        WaitForSingleObject(h, INFINITE);
        let mut code = 0u32;
        GetExitCodeProcess(h, &mut code);
        CloseHandle(h);
        windows_sys::Win32::UI::WindowsAndMessaging::PostMessageW(hwnd as windows_sys::Win32::Foundation::HWND, msg, code as usize, 0);
    });
}

/// The compute step's threads (`--precompile-all`, no game running beside it): all logical
/// processors but one, at most one per 0.75 GB of memory beyond 4 GB, 2 to 32 (2026-11-20:
/// 17 compiles at once held 1.8 GB in all, ~110 MB each, among the largest programs, and left a
/// quarter of the CPUs idle).
pub fn compute_threads() -> usize {
    let gb = memory() as f64 / (1u64 << 30) as f64;
    let by_mem = ((gb - 4.0) / 0.75).floor().max(2.0) as usize;
    threads().saturating_sub(1).min(by_mem).clamp(2, 32)
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

/// Mains or battery, and the battery saver (the launcher's log: a laptop on battery prepares
/// slower).
pub fn power_text() -> String {
    use windows_sys::Win32::System::Power::{GetSystemPowerStatus, SYSTEM_POWER_STATUS};
    let mut s: SYSTEM_POWER_STATUS = unsafe { std::mem::zeroed() };
    if unsafe { GetSystemPowerStatus(&mut s) } == 0 {
        return "power unknown".into();
    }
    let source = match s.ACLineStatus {
        0 => "on battery",
        1 => "on mains power",
        _ => "power source unknown",
    };
    let battery = if s.BatteryFlag & 128 != 0 { String::new() } else if s.BatteryLifePercent <= 100 { format!(", battery {}%", s.BatteryLifePercent) } else { ", battery level unknown".into() };
    format!("{source}{battery}{}", if s.SystemStatusFlag == 1 { ", battery saver on" } else { "" })
}

/// Free space on the drive holding `p` (bytes), if known.
pub fn free_space(p: &std::path::Path) -> Option<u64> {
    use windows_sys::Win32::Storage::FileSystem::GetDiskFreeSpaceExW;
    // (`DS_SIM_FREE_GB=<n>`, a test: the drive's free space as given.)
    if let Some(gb) = std::env::var("DS_SIM_FREE_GB").ok().and_then(|v| v.parse::<f64>().ok()) {
        return Some((gb * (1u64 << 30) as f64) as u64);
    }
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
