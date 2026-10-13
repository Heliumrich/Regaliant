//! Host scheduling of the GPU front end's threads: their priority (`tune
//! cp_priority`) and the logical CPUs they may run on (`tune cp_cpus`).
//! The console's command processors are hardware beside the CPU cores,
//! while here they are threads that queue behind the title's (13 job
//! workers that spin before they sleep) and other processes' threads
//! whenever the host's CPUs are oversubscribed.

use std::cell::Cell;

thread_local! {
    /// The `cp_priority` and `cp_cpus` values this thread last applied.
    static APPLIED: Cell<(u64, u64)> = const { Cell::new((u64::MAX, u64::MAX)) };
}

/// Applies `tune cp_priority` and `tune cp_cpus` to the calling thread if
/// they changed since this thread last looked (two relaxed loads otherwise).
#[inline]
pub fn apply() {
    let want = (super::tune::cp_priority(), super::tune::cp_cpus());
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
