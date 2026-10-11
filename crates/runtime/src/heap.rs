//! The host heap: mimalloc, or the Windows process heap with
//! `DS_ALLOC=system` (decided once, at the process's first allocation).
//!
//! The command processors allocate and free many small blocks per draw
//! (prepared stages, binding lists): with the Windows heap ~19% of the
//! graphics command processor's busy samples and ~23% of the parallel
//! workers' were in RtlAllocateHeap/RtlFreeHeap (profile run
//! 20261011-211613).
//!
//! mimalloc places its segments at hint addresses between 2 and 6 TiB on
//! 64-bit Windows, inside the guest address range (`mem::GUEST_BASE` ..
//! `GUEST_END`), which must be reserved in one piece: the first allocation
//! therefore reserves the guest range before anything else is allocated.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU8, Ordering};

pub struct HostAlloc;

const UNDECIDED: u8 = 0;
const MIMALLOC: u8 = 1;
const SYSTEM: u8 = 2;
const DECIDING: u8 = 3;

static MODE: AtomicU8 = AtomicU8::new(UNDECIDED);

#[inline]
fn mode() -> u8 {
    let m = MODE.load(Ordering::Acquire);
    if m == MIMALLOC || m == SYSTEM {
        return m;
    }
    decide()
}

#[cold]
fn decide() -> u8 {
    if MODE.compare_exchange(UNDECIDED, DECIDING, Ordering::AcqRel, Ordering::Acquire).is_ok() {
        crate::mem::reserve_guest_range_early();
        let m = if env_is_system() { SYSTEM } else { MIMALLOC };
        MODE.store(m, Ordering::Release);
        return m;
    }
    loop {
        let m = MODE.load(Ordering::Acquire);
        if m == MIMALLOC || m == SYSTEM {
            return m;
        }
        std::hint::spin_loop();
    }
}

/// `DS_ALLOC=system` (read without allocating).
fn env_is_system() -> bool {
    use windows_sys::Win32::System::Environment::GetEnvironmentVariableW;
    let name: [u16; 9] = [b'D' as u16, b'S' as u16, b'_' as u16, b'A' as u16, b'L' as u16, b'L' as u16, b'O' as u16, b'C' as u16, 0];
    let mut buf = [0u16; 16];
    let n = unsafe { GetEnvironmentVariableW(name.as_ptr(), buf.as_mut_ptr(), buf.len() as u32) } as usize;
    n == 6 && buf[..6].iter().zip("system".bytes()).all(|(&a, b)| a == b as u16)
}

/// Which heap is in use (status line).
pub fn name() -> &'static str {
    match MODE.load(Ordering::Acquire) {
        SYSTEM => "Windows heap (DS_ALLOC=system)",
        _ => "mimalloc",
    }
}

unsafe impl GlobalAlloc for HostAlloc {
    #[inline]
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        unsafe {
            if mode() == SYSTEM {
                System.alloc(l)
            } else {
                mimalloc::MiMalloc.alloc(l)
            }
        }
    }

    #[inline]
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        unsafe {
            if mode() == SYSTEM {
                System.alloc_zeroed(l)
            } else {
                mimalloc::MiMalloc.alloc_zeroed(l)
            }
        }
    }

    #[inline]
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe {
            if mode() == SYSTEM {
                System.dealloc(p, l)
            } else {
                mimalloc::MiMalloc.dealloc(p, l)
            }
        }
    }

    #[inline]
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        unsafe {
            if mode() == SYSTEM {
                System.realloc(p, l, n)
            } else {
                mimalloc::MiMalloc.realloc(p, l, n)
            }
        }
    }
}
