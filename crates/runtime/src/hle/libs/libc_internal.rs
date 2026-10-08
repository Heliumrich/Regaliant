//! libSceLibcInternalExt: services the title's libc.prx imports from the
//! system libc (heap tracing hooks).

use crate::hle::Registry;
use crate::{reg, sysv};

sysv! {
    /// libc passes a template {u64 size = 0x20, u32 1, u32 enabled = 0, u64, u64};
    /// leaving it unchanged reports heap tracing as disabled (no tracing tool
    /// is attached, which is the console's normal state).
    fn sceLibcHeapGetTraceInfo(info: *mut u8) -> i32 {
        crate::hle::stub_log("sceLibcHeapGetTraceInfo (heap tracing disabled)", 0);
        0
    }
}

pub fn register(r: &mut Registry) {
    reg!(r, "libSceLibcInternalExt", [sceLibcHeapGetTraceInfo]);
}
