//! Host scheduling of the GPU front end's threads: their priority (`tune
//! cp_priority`) and the logical CPUs they may run on (`tune cp_cpus`).
//! The console's command processors are hardware beside the CPU cores,
//! while here they are threads that queue behind the title's (13 job
//! workers that spin before they sleep) and other processes' threads
//! whenever the host's CPUs are oversubscribed.
//!
//! By default (`cp_cpus` AUTO) they run on the logical CPUs of one
//! last-level-cache group when the machine has several (a Ryzen CCD): the
//! front end's threads share most of their data, which then stays in one L3,
//! and the title's spinning job workers mostly run elsewhere (FINDINGS.md
//! 2026-10-16: -11% CPU per command, +6.7% flips/s on a 5900X).

use std::cell::Cell;

/// `tune cp_cpus` value for the automatic choice (`auto_cpus`).
pub const CPUS_AUTO: u64 = u64::MAX - 1;

thread_local! {
    /// The priority level and `cp_cpus` value this thread last applied.
    static APPLIED: Cell<(u64, u64)> = const { Cell::new((u64::MAX, u64::MAX)) };
    /// The graphics queue's command processor (`tune cp_priority 3`).
    static MAIN: Cell<bool> = const { Cell::new(false) };
}

/// The calling thread is the graphics queue's command processor.
pub fn set_main() {
    MAIN.with(|m| m.set(true));
}

/// Applies `tune cp_priority` and `tune cp_cpus` to the calling thread if
/// they changed since this thread last looked (two relaxed loads otherwise).
/// `cp_priority 3`: above normal for the graphics command processor only
/// (the critical path), normal for the other front-end threads.
#[inline]
pub fn apply() {
    let level = match super::tune::cp_priority() {
        3 => MAIN.with(|m| m.get()) as u64,
        p => p,
    };
    let mut cpus = match super::tune::cp_cpus() {
        CPUS_AUTO => auto_cpus(),
        m => m,
    };
    if super::tune::gfx_core() {
        cpus = own_core(cpus, MAIN.with(|m| m.get()));
    }
    let want = (level, cpus);
    let had = APPLIED.with(|a| a.replace(want));
    if had != want {
        set(had, want);
    }
}

#[cold]
fn set(had: (u64, u64), want: (u64, u64)) {
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, GetCurrentThread, GetProcessAffinityMask, SetThreadAffinityMask, SetThreadPriority, THREAD_PRIORITY_ABOVE_NORMAL, THREAD_PRIORITY_HIGHEST, THREAD_PRIORITY_NORMAL};
    if had.0 != want.0 {
        let p = match want.0 {
            0 => THREAD_PRIORITY_NORMAL,
            1 => THREAD_PRIORITY_ABOVE_NORMAL,
            _ => THREAD_PRIORITY_HIGHEST,
        };
        if unsafe { SetThreadPriority(GetCurrentThread(), p) } == 0 {
            crate::warn!("gpu: thread priority {p} not set ({})", std::io::Error::last_os_error());
        }
    }
    if had.1 != want.1 && !(had.1 == u64::MAX && want.1 == 0) {
        // 0: the process's own mask (all of its CPUs).
        let mut mask = want.1 as usize;
        if mask == 0 {
            let (mut process, mut system) = (0usize, 0usize);
            if unsafe { GetProcessAffinityMask(GetCurrentProcess(), &mut process, &mut system) } == 0 {
                return;
            }
            mask = process;
        }
        if unsafe { SetThreadAffinityMask(GetCurrentThread(), mask) } == 0 {
            crate::warn!("gpu: thread affinity {mask:#x} not set ({})", std::io::Error::last_os_error());
        }
    }
}

/// The logical CPUs `cp_cpus` AUTO stands for: the first last-level-cache
/// groups (in CPU order) holding at least min(12, half the process's CPUs),
/// when there are two or more such groups; 0 (all) otherwise. Logged once.
pub fn auto_cpus() -> u64 {
    static MASK: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *MASK.get_or_init(|| {
        let process = process_mask();
        let groups = l3_groups();
        let m = choose(&groups, process);
        let list: Vec<String> = groups.iter().map(|g| format!("{g:#x}")).collect();
        if m == 0 {
            crate::info!("gpu: front-end threads on all logical CPUs (last-level caches over CPUs {})", if list.is_empty() { "unknown".to_string() } else { list.join(", ") });
        } else {
            crate::info!("gpu: front-end threads on logical CPUs {m:#x} (last-level caches over CPUs {}; `tune cp_cpus 0` for all)", list.join(", "));
        }
        m
    })
}

/// `tune gfx_core`: the graphics command processor (`main`) on the first
/// physical core of the front end's CPUs (`cpus`, 0 = the process's), the
/// other front-end threads on the rest (the critical path's thread then
/// shares its core with no other front-end thread; another port's
/// graphics CP has a core of its own). Unchanged when the front end has a
/// single core.
fn own_core(cpus: u64, main: bool) -> u64 {
    static CORES: std::sync::OnceLock<(u64, Vec<u64>)> = std::sync::OnceLock::new();
    let (process, cores) = CORES.get_or_init(|| (process_mask(), core_groups()));
    split_core(if cpus == 0 { *process } else { cpus }, cores, main).unwrap_or(cpus)
}

/// `front` split into its first physical core (for `main`) and the rest;
/// None when it has a single core.
fn split_core(front: u64, cores: &[u64], main: bool) -> Option<u64> {
    if front == 0 {
        return None;
    }
    let first = 1u64 << front.trailing_zeros();
    let core = cores.iter().find(|&&c| c & first != 0).map_or(first, |&c| c & front);
    if front & !core == 0 {
        return None;
    }
    Some(if main { core } else { front & !core })
}

/// The CPU masks of the physical cores in processor group 0
/// (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX records of RelationProcessorCore:
/// PROCESSOR_RELATIONSHIP from 8, GroupMask[0].Mask at 32, .Group at 40).
fn core_groups() -> Vec<u64> {
    use windows_sys::Win32::System::SystemInformation::{GetLogicalProcessorInformationEx, RelationProcessorCore};
    let mut len = 0u32;
    unsafe { GetLogicalProcessorInformationEx(RelationProcessorCore, std::ptr::null_mut(), &mut len) };
    if len == 0 {
        return Vec::new();
    }
    let mut buf = vec![0u64; (len as usize).div_ceil(8)];
    if unsafe { GetLogicalProcessorInformationEx(RelationProcessorCore, buf.as_mut_ptr() as *mut _, &mut len) } == 0 {
        return Vec::new();
    }
    let bytes = unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const u8, len as usize) };
    let u32_at = |o: usize| u32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]);
    let mut out = Vec::new();
    let mut off = 0usize;
    while off + 8 <= bytes.len() {
        let (rel, size) = (u32_at(off), u32_at(off + 4) as usize);
        if size < 8 || off + size > bytes.len() {
            break;
        }
        if rel == RelationProcessorCore as u32 && size >= 42 {
            let mask = u64::from_le_bytes(bytes[off + 32..off + 40].try_into().unwrap());
            let group = u16::from_le_bytes([bytes[off + 40], bytes[off + 41]]);
            if group == 0 && mask != 0 {
                out.push(mask);
            }
        }
        off += size;
    }
    out
}

/// The process's affinity mask (processor group 0).
fn process_mask() -> u64 {
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, GetProcessAffinityMask};
    let (mut process, mut system) = (0usize, 0usize);
    if unsafe { GetProcessAffinityMask(GetCurrentProcess(), &mut process, &mut system) } == 0 {
        return 0;
    }
    process as u64
}

/// The CPU masks of the level-3 caches in processor group 0, sorted by
/// their lowest CPU (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX records:
/// Relationship at 0, Size at 4, CACHE_RELATIONSHIP from 8: Level at 8,
/// GroupMask.Mask at 40, GroupMask.Group at 48).
fn l3_groups() -> Vec<u64> {
    use windows_sys::Win32::System::SystemInformation::{GetLogicalProcessorInformationEx, RelationCache};
    let mut len = 0u32;
    unsafe { GetLogicalProcessorInformationEx(RelationCache, std::ptr::null_mut(), &mut len) };
    if len == 0 {
        return Vec::new();
    }
    let mut buf = vec![0u64; (len as usize).div_ceil(8)];
    if unsafe { GetLogicalProcessorInformationEx(RelationCache, buf.as_mut_ptr() as *mut _, &mut len) } == 0 {
        return Vec::new();
    }
    let bytes = unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const u8, len as usize) };
    let u32_at = |o: usize| u32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]);
    let mut out = Vec::new();
    let mut off = 0usize;
    while off + 8 <= bytes.len() {
        let (rel, size) = (u32_at(off), u32_at(off + 4) as usize);
        if size < 8 || off + size > bytes.len() {
            break;
        }
        if rel == RelationCache as u32 && size >= 50 && bytes[off + 8] == 3 {
            let mask = u64::from_le_bytes(bytes[off + 40..off + 48].try_into().unwrap());
            let group = u16::from_le_bytes([bytes[off + 48], bytes[off + 49]]);
            if group == 0 && mask != 0 {
                out.push(mask);
            }
        }
        off += size;
    }
    out.sort_by_key(|m| m.trailing_zeros());
    out.dedup();
    out
}

/// The front end's CPUs from the last-level-cache groups (see `auto_cpus`).
fn choose(groups: &[u64], process: u64) -> u64 {
    let groups: Vec<u64> = groups.iter().map(|g| g & process).filter(|&g| g != 0).collect();
    if groups.len() < 2 {
        return 0;
    }
    let want = 12.min(process.count_ones() / 2).max(1);
    let mut mask = 0u64;
    for g in &groups {
        mask |= g;
        if mask.count_ones() >= want {
            break;
        }
    }
    if mask == process {
        0
    } else {
        mask
    }
}

#[cfg(test)]
mod tests {
    use super::{choose, split_core};

    #[test]
    fn graphics_cp_core() {
        // 5900X: SMT pairs; the front end on the first CCD (CPUs 0-11).
        let cores: Vec<u64> = (0..12).map(|i| 0b11u64 << (2 * i)).collect();
        assert_eq!(split_core(0xfff, &cores, true), Some(0b11));
        assert_eq!(split_core(0xfff, &cores, false), Some(0xffc));
        // One core only: unchanged.
        assert_eq!(split_core(0b11, &cores, true), None);
        // No core information: the lowest CPU alone.
        assert_eq!(split_core(0xf0, &[], true), Some(0x10));
        assert_eq!(split_core(0xf0, &[], false), Some(0xe0));
    }

    #[test]
    fn front_end_cpus_from_cache_groups() {
        // 5900X: two CCDs of 12 logical CPUs.
        assert_eq!(choose(&[0xFFF, 0xFFF000], 0xFFFFFF), 0xFFF);
        // One last-level cache: all CPUs.
        assert_eq!(choose(&[0xFFFF], 0xFFFF), 0);
        assert_eq!(choose(&[], 0xFFFF), 0);
        // 3900X-like: four L3 groups of 6 logical CPUs: two of them.
        assert_eq!(choose(&[0x3F, 0xFC0, 0x3F000, 0xFC0000], 0xFFFFFF), 0xFFF);
        // 7950X: two CCDs of 16: one.
        assert_eq!(choose(&[0xFFFF, 0xFFFF_0000], 0xFFFF_FFFF), 0xFFFF);
        // Two small groups that together are every CPU: all.
        assert_eq!(choose(&[0xF, 0xF0], 0xFF), 0xF);
        assert_eq!(choose(&[0x3, 0xC], 0xF), 0x3);
        // A process mask limits the groups.
        assert_eq!(choose(&[0xFFF, 0xFFF000], 0xFFF), 0);
    }
}
