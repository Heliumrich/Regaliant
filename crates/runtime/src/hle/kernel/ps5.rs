//! PS5-specific libkernel services used by the title.

use super::errno::*;
use super::sce;
use crate::hle::Registry;
use crate::{reg, sysv};
use parking_lot::Mutex;

/// Partially-resident-texture apertures (`sceKernelSetPrtAperture`): GPU reads
/// of unmapped pages inside an aperture return zero instead of faulting. The
/// GPU front end consults this table.
pub static PRT_APERTURES: Mutex<[(u64, u64); 3]> = Mutex::new([(0, 0); 3]);

sysv! {
    /// Devkit "GPI" switches; a retail console reports none.
    fn sceKernelGetGPI() -> u64 { 0 }
    /// The title checks `a == 2 && b == 1` on the result. Retail meaning is
    /// unconfirmed; (2, 0) is reported, which makes that check false.
    fn sceKernelGetOperationMode(a: *mut i32, b: *mut i32) -> i32 {
        if !a.is_null() { *a = 2; }
        if !b.is_null() { *b = 0; }
        crate::hle::stub_log("sceKernelGetOperationMode (2, 0)", 0);
        0
    }
    fn sceKernelSetPrtAperture(index: i32, base: u64, size: u64) -> i32 {
        if !(0..3).contains(&index) || base % crate::mem::PAGE != 0 || size % crate::mem::PAGE != 0 { return sce(EINVAL); }
        PRT_APERTURES.lock()[index as usize] = (base, size);
        crate::info!("PRT aperture {index}: {base:#x}+{size:#x}");
        0
    }
    fn sceKernelGetPrtAperture(index: i32, base: *mut u64, size: *mut u64) -> i32 {
        if !(0..3).contains(&index) { return sce(EINVAL); }
        let (b, s) = PRT_APERTURES.lock()[index as usize];
        if !base.is_null() { *base = b; }
        if !size.is_null() { *size = s; }
        0
    }
    /// Random (version 4) UUID.
    fn sceKernelUuidCreate(out: *mut u8) -> i32 {
        if out.is_null() { return sce(EINVAL); }
        let mut b = [0u8; 16];
        let mut x = crate::hle::kernel::time::process_time_us() ^ unsafe { core::arch::x86_64::_rdtsc() } ^ 0x9E37_79B9_7F4A_7C15;
        for v in b.iter_mut() {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            *v = x as u8;
        }
        b[6] = (b[6] & 0x0F) | 0x40;
        b[8] = (b[8] & 0x3F) | 0x80;
        std::ptr::copy_nonoverlapping(b.as_ptr(), out, 16);
        0
    }
    /// The title registers a crash handler; this runtime reports faults itself
    /// and never invokes it (logged once).
    fn sceCoredumpRegisterCoredumpHandler(handler: u64, stack_size: u64, arg: u64) -> i32 {
        crate::info!("coredump handler {} registered (not invoked by this runtime)", crate::loader::describe_addr(handler));
        0
    }
    fn sceCoredumpUnregisterCoredumpHandler() -> i32 { 0 }
}

pub fn register(r: &mut Registry) {
    reg!(r, "libkernel", [
        sceKernelGetGPI, sceKernelGetOperationMode, sceKernelSetPrtAperture, sceKernelGetPrtAperture,
        sceKernelUuidCreate,
    ]);
    reg!(r, "libSceCoredump", [sceCoredumpRegisterCoredumpHandler, sceCoredumpUnregisterCoredumpHandler]);
}
