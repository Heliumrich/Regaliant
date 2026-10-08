//! libSceAgcDriver: submission, GPU events and resource registration.
//!
//! Resource registration (owners/resources named by the title for GPU capture
//! tools) is recorded and used to name guest memory in GPU diagnostics.

use crate::hle::{Registry, cstr};
use crate::{reg, sysv};
use parking_lot::Mutex;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU32, Ordering};

const MAX_NAME: u32 = 127;

#[derive(Clone, Debug)]
pub struct Resource {
    pub owner: u32,
    pub addr: u64,
    pub size: u64,
    pub name: String,
    pub kind: u32,
}

static NEXT_HANDLE: AtomicU32 = AtomicU32::new(1);
static OWNERS: Mutex<BTreeMap<u32, String>> = Mutex::new(BTreeMap::new());
static RESOURCES: Mutex<BTreeMap<u32, Resource>> = Mutex::new(BTreeMap::new());

/// Name of the registered resource containing `addr`, if any (diagnostics).
pub fn resource_at(addr: u64) -> Option<Resource> {
    RESOURCES.lock().values().find(|r| addr >= r.addr && addr < r.addr + r.size.max(1)).cloned()
}

sysv! {
    fn sceAgcDriverGetResourceRegistrationMaxNameLength() -> u32 { MAX_NAME }
    fn sceAgcDriverQueryResourceRegistrationUserMemoryRequirements(size: *mut u64) -> i32 {
        // Bookkeeping lives on the host side; no title memory is needed.
        if !size.is_null() { *size = 0; }
        0
    }
    fn sceAgcDriverInitResourceRegistration(mem: u64, size: u64) -> i32 { 0 }
    fn sceAgcDriverGetDefaultOwner() -> u32 { 0 }
    fn sceAgcDriverRegisterOwner(out: *mut u32, name: *const u8) -> i32 {
        let h = NEXT_HANDLE.fetch_add(1, Ordering::Relaxed);
        OWNERS.lock().insert(h, cstr(name).to_string());
        crate::debug!("agc: owner {h} '{}'", cstr(name));
        if !out.is_null() { *out = h; }
        0
    }
    fn sceAgcDriverRegisterResource(out: *mut u32, owner: u32, addr: u64, size: u64, name: *const u8, kind: u32, user: u64) -> i32 {
        let h = NEXT_HANDLE.fetch_add(1, Ordering::Relaxed);
        let r = Resource { owner, addr, size, name: cstr(name).to_string(), kind };
        crate::debug!("agc: resource {h} '{}' type {kind:#x} {addr:#x}+{size:#x} owner {owner}", r.name);
        RESOURCES.lock().insert(h, r);
        if !out.is_null() { *out = h; }
        0
    }
    fn sceAgcDriverUnregisterResource(h: u32) -> i32 {
        RESOURCES.lock().remove(&h);
        0
    }
    fn sceAgcDriverUnregisterOwnerAndResources(owner: u32) -> i32 {
        OWNERS.lock().remove(&owner);
        RESOURCES.lock().retain(|_, r| r.owner != owner);
        0
    }
    fn sceAgcDriverGetResourceName(h: u32, buf: *mut u8, size: u32) -> i32 {
        match RESOURCES.lock().get(&h) {
            Some(r) => { crate::hle::write_cstr(buf, size as usize, &r.name); 0 }
            None => 0x8A6C_0035u32 as i32,
        }
    }
    fn sceAgcDriverGetResourceType(h: u32, kind: *mut u32) -> i32 {
        match RESOURCES.lock().get(&h) {
            Some(r) => { if !kind.is_null() { *kind = r.kind; } 0 }
            None => 0x8A6C_0035u32 as i32,
        }
    }
    /// No GPU capture tool is attached.
    fn sceAgcDriverIsCaptureInProgress() -> i32 { 0 }
}

pub fn register(r: &mut Registry) {
    reg!(r, "libSceAgcDriver", [
        sceAgcDriverGetResourceRegistrationMaxNameLength, sceAgcDriverQueryResourceRegistrationUserMemoryRequirements,
        sceAgcDriverInitResourceRegistration, sceAgcDriverGetDefaultOwner, sceAgcDriverRegisterOwner,
        sceAgcDriverRegisterResource, sceAgcDriverUnregisterResource, sceAgcDriverUnregisterOwnerAndResources,
        sceAgcDriverGetResourceName, sceAgcDriverGetResourceType, sceAgcDriverIsCaptureInProgress,
    ]);
}
