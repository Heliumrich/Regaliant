//! PS5 system services on the startup path: Json2 (C++ API), HTTP/2, NP Web
//! API 2, game-update check, game intents, system gestures, Share, NP state and
//! the Universal Data System (telemetry).
//!
//! This runtime behaves as an offline console: contexts are created so the
//! title's initialization proceeds, but no request ever reaches a network.
//! Each such answer is logged once as a fixed result.

use crate::hle::{Registry, stub_log};
use crate::{reg, sysv};
use std::sync::atomic::{AtomicI32, Ordering};

static NEXT: AtomicI32 = AtomicI32::new(1);
const WEBAPI2_ERROR_INVALID_ARGUMENT: i32 = 0x8055_3402u32 as i32;
/// NpWebApi2 "not signed in" (error codes per public sources, docs/PROVENANCE.md).
const WEBAPI2_ERROR_NOT_SIGNED_IN: i32 = 0x8055_3407u32 as i32;

fn cstr_or(p: *const u8) -> String {
    if p.is_null() { String::new() } else { unsafe { crate::hle::cstr(p) }.to_string() }
}
fn next_id() -> i32 {
    NEXT.fetch_add(1, Ordering::Relaxed)
}

sysv! {
    // sce::Json::MemAllocator::MemAllocator() — base-object constructor; the
    // title's derived allocator installs its own vtable afterwards.
    fn _ZN3sce4Json12MemAllocatorC2Ev(this: u64) -> u64 { this }
    fn _ZN3sce4Json12MemAllocatorD2Ev(this: u64) {}
    // sce::Json::Initializer
    fn _ZN3sce4Json11InitializerC1Ev(this: u64) -> u64 { this }
    fn _ZN3sce4Json11InitializerD1Ev(this: u64) {}
    fn _ZN3sce4Json11Initializer10initializeEPKNS0_13InitParameterE(this: u64, param: u64) -> i32 {
        stub_log("sce::Json::Initializer::initialize (no JSON work is requested offline)", 0);
        0
    }

    fn sceHttp2Init(net_id: i32, ssl_id: i32, pool_size: u64, max_requests: i32) -> i32 {
        let id = next_id();
        crate::info!("sceHttp2Init(pool {pool_size:#x}, max {max_requests}) -> context {id} (offline: no request will reach a network)");
        id
    }
    fn sceHttp2Term(ctx: i32) -> i32 { 0 }
    fn sceNpWebApi2Initialize(http_ctx: i32, pool_size: u64) -> i32 {
        let id = next_id();
        crate::info!("sceNpWebApi2Initialize(http {http_ctx}, pool {pool_size:#x}) -> context {id} (offline)");
        id
    }
    fn sceNpWebApi2Terminate(ctx: i32) -> i32 { 0 }
    /// (library context, user id) -> user context. Contexts exist offline;
    /// requests made through them fail.
    fn sceNpWebApi2CreateUserContext(lib_ctx: i32, user: i32) -> i32 {
        let id = next_id();
        crate::info!("sceNpWebApi2CreateUserContext(lib {lib_ctx}, user {user}) -> {id} (offline: requests will fail)");
        id
    }
    fn sceNpWebApi2DeleteUserContext(user_ctx: i32) -> i32 { 0 }
    /// (user context, api group, path, method, content parameter, *request id)
    fn sceNpWebApi2CreateRequest(user_ctx: i32, group: *const u8, path: *const u8, method: *const u8, content: u64, out: *mut i64) -> i32 {
        if out.is_null() { return WEBAPI2_ERROR_INVALID_ARGUMENT; }
        *out = next_id() as i64;
        crate::debug!("sceNpWebApi2CreateRequest {} {} {}", cstr_or(method), cstr_or(group), cstr_or(path));
        0
    }
    fn sceNpWebApi2AddHttpRequestHeader(req: i64, name: *const u8, value: *const u8) -> i32 { 0 }
    /// Offline console: the user is not signed in to PSN.
    fn sceNpWebApi2SendRequest(req: i64, data: u64, size: u64, info: u64) -> i32 {
        stub_log("sceNpWebApi2SendRequest (offline: not signed in)", WEBAPI2_ERROR_NOT_SIGNED_IN as i64);
        WEBAPI2_ERROR_NOT_SIGNED_IN
    }
    fn sceNpWebApi2ReadData(req: i64, data: u64, size: u64) -> i32 { WEBAPI2_ERROR_NOT_SIGNED_IN }
    fn sceNpWebApi2DeleteRequest(req: i64) -> i32 { 0 }
    // Push events: handles, filters and callbacks exist; no event is delivered.
    fn sceNpWebApi2PushEventCreateHandle(lib_ctx: i32) -> i32 { next_id() }
    fn sceNpWebApi2PushEventDeleteHandle(lib_ctx: i32, handle: i32) -> i32 { 0 }
    fn sceNpWebApi2PushEventCreateFilter(lib_ctx: i32, handle: i32, service: u64, label: u32, params: u64, count: u64) -> i32 { next_id() }
    fn sceNpWebApi2PushEventRegisterCallback(user_ctx: i32, filter: i32, cb: u64, arg: u64) -> i32 {
        stub_log("sceNpWebApi2PushEventRegisterCallback (offline: no push events)", 1);
        next_id()
    }
    fn sceNpWebApi2PushEventDeletePushContext(user_ctx: i32, push_ctx: u64) -> i32 { 0 }

    fn sceGameUpdateInitialize(param: u64) -> i32 {
        stub_log("sceGameUpdateInitialize (offline: update checks report no update)", 0);
        0
    }
    fn sceNpGameIntentInitialize(param: u64) -> i32 {
        stub_log("sceNpGameIntentInitialize (no game intents are delivered)", 0);
        0
    }

    // ---- System gestures (DualSense touch pad): no touch input is mapped yet,
    // so recognizers never report events.
    fn sceSystemGestureInitializePrimitiveTouchRecognizer(param: u64) -> i32 { 0 }
    fn sceSystemGestureFinalizePrimitiveTouchRecognizer() -> i32 { 0 }
    fn sceSystemGestureOpen(input_type: i32, param: u64) -> i32 {
        stub_log("sceSystemGestureOpen (touch pad input not mapped: no gesture events)", 1);
        1
    }
    fn sceSystemGestureClose(h: i32) -> i32 { 0 }
    fn sceSystemGestureCreateTouchRecognizer(h: i32, rec: *mut u8, kind: u32, rect: u64, param: u64) -> i32 {
        if !rec.is_null() { *(rec as *mut u32) = kind; }
        0
    }
    fn sceSystemGestureUpdatePrimitiveTouchRecognizer(data: u64) -> i32 { 0 }
    fn sceSystemGestureUpdateTouchRecognizer(h: i32, rec: u64) -> i32 { 0 }
    fn sceSystemGestureGetTouchEventsCount(h: i32, rec: u64) -> i32 { 0 }
    fn sceSystemGestureGetTouchEventByIndex(h: i32, rec: u64, index: u32, ev: u64) -> i32 { 0x8080_0001u32 as i32 }

    // ---- Share (screenshots/video sharing): accepted, nothing is captured.
    fn sceShareInitialize() -> i32 { stub_log("sceShareInitialize (sharing features inactive)", 0); 0 }
    fn sceShareSetContentParam(param: u64) -> i32 { 0 }

    // ---- NP (PSN): offline console.
    fn sceNpRegisterStateCallbackA(cb: u64, arg: u64) -> i32 { stub_log("sceNpRegisterStateCallbackA (offline: state stays signed out)", 1); 1 }
    fn sceNpUnregisterStateCallbackA(id: i32) -> i32 { 0 }
    fn sceNpSessionSignalingInitialize(param: u64) -> i32 { stub_log("sceNpSessionSignalingInitialize (offline)", 0); 0 }
    /// libSceUserServicePlatformPrivacyWs1 export D-CzAxQL0XI (unnamed): reported OK.
    fn ds_user_service_platform_privacy(a: u64, b: u64, c: u64) -> i32 {
        stub_log("libSceUserServicePlatformPrivacyWs1 D-CzAxQL0XI (unnamed; returns 0)", 0);
        0
    }

    // ---- NP entitlement access (add-on content). The supplied package has no
    // add-ons installed. Error code and structure sizes per public sources.
    fn sceNpEntitlementAccessInitialize(init: *const u8, boot: *mut u8) -> i32 {
        if init.is_null() || boot.is_null() { return 0x817D_0002u32 as i32; }
        std::ptr::write_bytes(boot, 0, 32);
        stub_log("sceNpEntitlementAccessInitialize (no add-on entitlements)", 0);
        0
    }
    fn sceNpEntitlementAccessGetAddcontEntitlementInfoList(label: u32, list: *mut u8, num: u32, hit: *mut u32) -> i32 {
        if hit.is_null() || (list.is_null() && num != 0) { return 0x817D_0002u32 as i32; }
        *hit = 0;
        0
    }

    // ---- Universal Data System (telemetry): contexts exist, events are dropped.
    fn sceNpUniversalDataSystemInitialize(param: u64) -> i32 { stub_log("sceNpUniversalDataSystemInitialize (offline: telemetry events are discarded)", 0); 0 }
    fn sceNpUniversalDataSystemCreateContext(out: *mut i32, user: i32, a: u32, b: u64) -> i32 {
        if !out.is_null() { *out = next_id(); }
        0
    }
    fn sceNpUniversalDataSystemCreateHandle(out: *mut i32) -> i32 {
        if !out.is_null() { *out = next_id(); }
        0
    }
    fn sceNpUniversalDataSystemRegisterContext(ctx: i32, h: i32, flags: u32) -> i32 { 0 }
    /// 24-byte memory statistics (the title zeroes it first, eboot+0x650132):
    /// nothing is allocated because events are discarded.
    fn sceNpUniversalDataSystemGetMemoryStat(stat: *mut u64) -> i32 {
        if stat.is_null() { return 0x8155_1002u32 as i32; }
        std::ptr::write_bytes(stat as *mut u8, 0, 24);
        0
    }
}

pub fn register(r: &mut Registry) {
    reg!(r, "libSceJson2", [
        _ZN3sce4Json12MemAllocatorC2Ev, _ZN3sce4Json12MemAllocatorD2Ev, _ZN3sce4Json11InitializerC1Ev,
        _ZN3sce4Json11InitializerD1Ev, _ZN3sce4Json11Initializer10initializeEPKNS0_13InitParameterE,
    ]);
    reg!(r, "libSceHttp2", [sceHttp2Init, sceHttp2Term]);
    reg!(r, "libSceNpWebApi2", [
        sceNpWebApi2Initialize, sceNpWebApi2Terminate, sceNpWebApi2CreateUserContext, sceNpWebApi2DeleteUserContext,
        sceNpWebApi2CreateRequest, sceNpWebApi2AddHttpRequestHeader, sceNpWebApi2SendRequest, sceNpWebApi2ReadData,
        sceNpWebApi2DeleteRequest, sceNpWebApi2PushEventCreateHandle, sceNpWebApi2PushEventDeleteHandle,
        sceNpWebApi2PushEventCreateFilter, sceNpWebApi2PushEventRegisterCallback, sceNpWebApi2PushEventDeletePushContext,
    ]);
    reg!(r, "libSceGameUpdate", [sceGameUpdateInitialize]);
    reg!(r, "libSceNpGameIntent", [sceNpGameIntentInitialize]);
    reg!(r, "libSceSystemGesture", [
        sceSystemGestureInitializePrimitiveTouchRecognizer, sceSystemGestureFinalizePrimitiveTouchRecognizer,
        sceSystemGestureOpen, sceSystemGestureClose, sceSystemGestureCreateTouchRecognizer,
        sceSystemGestureUpdatePrimitiveTouchRecognizer, sceSystemGestureUpdateTouchRecognizer,
        sceSystemGestureGetTouchEventsCount, sceSystemGestureGetTouchEventByIndex,
    ]);
    reg!(r, "libSceShare", [sceShareInitialize, sceShareSetContentParam]);
    reg!(r, "libSceNpManager", [sceNpRegisterStateCallbackA, sceNpUnregisterStateCallbackA]);
    reg!(r, "libSceNpSessionSignaling", [sceNpSessionSignalingInitialize]);
    r.func_nid("libSceUserServicePlatformPrivacyWs1", "D-CzAxQL0XI", "D-CzAxQL0XI (unnamed)", ds_user_service_platform_privacy as *const () as u64);
    reg!(r, "libSceNpEntitlementAccess", [sceNpEntitlementAccessInitialize, sceNpEntitlementAccessGetAddcontEntitlementInfoList]);
    reg!(r, "libSceNpUniversalDataSystem", [
        sceNpUniversalDataSystemInitialize, sceNpUniversalDataSystemCreateContext, sceNpUniversalDataSystemCreateHandle,
        sceNpUniversalDataSystemRegisterContext, sceNpUniversalDataSystemGetMemoryStat,
    ]);
}
