//! Host scheduling priority of the GPU front end's threads (`tune
//! cp_priority`): the console's command processors are hardware beside the
//! CPU cores, while here they are threads that queue behind the title's
//! (13 job workers that spin before they sleep) and other processes' threads
//! whenever the host's CPUs are oversubscribed.

use std::cell::Cell;

thread_local! {
    /// The `cp_priority` value this thread last applied.
    static APPLIED: Cell<u64> = const { Cell::new(u64::MAX) };
}

/// Applies `tune cp_priority` to the calling thread if it changed since
/// this thread last looked (one relaxed load otherwise).
#[inline]
pub fn apply() {
    let want = super::tune::cp_priority();
    if APPLIED.with(|a| a.replace(want)) != want {
        set(want);
    }
}

#[cold]
fn set(v: u64) {
    use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_ABOVE_NORMAL, THREAD_PRIORITY_HIGHEST, THREAD_PRIORITY_NORMAL};
    let p = match v {
        0 => THREAD_PRIORITY_NORMAL,
        1 => THREAD_PRIORITY_ABOVE_NORMAL,
        _ => THREAD_PRIORITY_HIGHEST,
    };
    if unsafe { SetThreadPriority(GetCurrentThread(), p) } == 0 {
        crate::warn!("gpu: thread priority {p} not set ({})", std::io::Error::last_os_error());
    }
}
