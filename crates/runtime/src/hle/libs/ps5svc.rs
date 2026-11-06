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

const UDS_ERROR_INVALID_ARGUMENT: i32 = 0x8055_3102u32 as i32;
const GAME_UPDATE_ERROR_INVALID_ARG: i32 = 0x8041_2803u32 as i32;
const GAME_UPDATE_ERROR_INVALID_SIZE: i32 = 0x8041_2804u32 as i32;
const HTTP2_ERROR_BEFORE_SEND: i32 = 0x817B_1065u32 as i32;
const HTTP2_ERROR_TIMEOUT: i32 = 0x817B_1068u32 as i32;
const HTTP2_ERROR_NULL_POINTER: i32 = 0x817B_1225u32 as i32;
const NP_ERROR_SIGNED_OUT: i32 = 0x8055_0006u32 as i32;
const SIGNIN_INITIALIZED: i32 = 1;
const SIGNIN_RUNNING: i32 = 2;
const SIGNIN_FINISHED: i32 = 3;
/// (status, polled while running)
static SIGNIN: parking_lot::Mutex<(i32, bool)> = parking_lot::Mutex::new((0, false));

/// Opaque UDS handles: 64-byte host blocks.
fn uds_alloc() -> u64 {
    let p = Box::into_raw(Box::new([0u64; 8])) as u64;
    UDS_LIVE.lock().insert(p);
    p
}
fn uds_free(p: u64) {
    if p != 0 && UDS_LIVE.lock().remove(&p) {
        drop(unsafe { Box::from_raw(p as *mut [u64; 8]) });
    }
    if p != 0 {
        let mut g = UDS_DATA.lock();
        g.0.remove(&p);
        g.1.remove(&p);
    }
}
/// What the port keeps of the events (`achievements`): event handle -> (name, its property
/// object), property object -> its top-level values.
type UdsData = (std::collections::BTreeMap<u64, (String, u64)>, std::collections::BTreeMap<u64, Vec<(String, crate::achievements::Value)>>);
static UDS_DATA: parking_lot::Mutex<UdsData> = parking_lot::Mutex::new((std::collections::BTreeMap::new(), std::collections::BTreeMap::new()));
fn uds_set(obj: u64, key: *const u8, v: crate::achievements::Value) -> i32 {
    if obj != 0 && !key.is_null() {
        let k = cstr_or(key);
        let mut g = UDS_DATA.lock();
        let props = g.1.entry(obj).or_default();
        props.retain(|(x, _)| *x != k);
        props.push((k, v));
    }
    uds_check(obj)
}
fn uds_check(p: u64) -> i32 {
    if p == 0 { UDS_ERROR_INVALID_ARGUMENT } else { 0 }
}
/// Handles allocated here (only these are freed; titles may pass their own).
static UDS_LIVE: parking_lot::Mutex<std::collections::BTreeSet<u64>> = parking_lot::Mutex::new(std::collections::BTreeSet::new());

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
    // ---- UDS events: property objects/arrays and events are opaque handles
    // (host allocations); posting discards the event.
    fn sceNpUniversalDataSystemCreateEvent(name: *const u8, props: u64, out: *mut u64, props_out: *mut u64) -> i32 {
        if name.is_null() || out.is_null() { return UDS_ERROR_INVALID_ARGUMENT; }
        *out = uds_alloc();
        let p = if props != 0 { props } else { uds_alloc() };
        if !props_out.is_null() { *props_out = p; }
        UDS_DATA.lock().0.insert(*out, (cstr_or(name), p));
        0
    }
    fn sceNpUniversalDataSystemDestroyEvent(ev: u64) -> i32 { uds_free(ev); 0 }
    fn sceNpUniversalDataSystemCreateEventPropertyObject(out: *mut u64) -> i32 {
        if out.is_null() { return UDS_ERROR_INVALID_ARGUMENT; }
        *out = uds_alloc();
        0
    }
    fn sceNpUniversalDataSystemDestroyEventPropertyObject(obj: u64) -> i32 { uds_free(obj); 0 }
    fn sceNpUniversalDataSystemCreateEventPropertyArray(out: *mut u64) -> i32 {
        if out.is_null() { return UDS_ERROR_INVALID_ARGUMENT; }
        *out = uds_alloc();
        0
    }
    fn sceNpUniversalDataSystemDestroyEventPropertyArray(arr: u64) -> i32 { uds_free(arr); 0 }
    fn sceNpUniversalDataSystemEventPropertyObjectSetString(obj: u64, key: *const u8, v: *const u8) -> i32 { uds_set(obj, key, crate::achievements::Value::Str(cstr_or(v))) }
    fn sceNpUniversalDataSystemEventPropertyObjectSetInt32(obj: u64, key: *const u8, v: i32) -> i32 { uds_set(obj, key, crate::achievements::Value::Int(v as i64)) }
    fn sceNpUniversalDataSystemEventPropertyObjectSetUInt32(obj: u64, key: *const u8, v: u32) -> i32 { uds_set(obj, key, crate::achievements::Value::Int(v as i64)) }
    fn sceNpUniversalDataSystemEventPropertyObjectSetUInt64(obj: u64, key: *const u8, v: u64) -> i32 { uds_set(obj, key, crate::achievements::Value::Int(v as i64)) }
    /// (object, key, array or NULL, *array used)
    fn sceNpUniversalDataSystemEventPropertyObjectSetArray(obj: u64, key: *const u8, v: u64, out: *mut u64) -> i32 {
        if !out.is_null() { *out = if v != 0 { v } else { uds_alloc() }; }
        uds_check(obj)
    }
    fn sceNpUniversalDataSystemEventPropertyArraySetString(arr: u64, v: *const u8) -> i32 { uds_check(arr) }
    fn sceNpUniversalDataSystemEventPropertyArraySetFloat32(arr: u64, v: f32) -> i32 { uds_check(arr) }
    /// (array, object or NULL, *object used)
    fn sceNpUniversalDataSystemEventPropertyArraySetObject(arr: u64, v: u64, out: *mut u64) -> i32 {
        if !out.is_null() { *out = if v != 0 { v } else { uds_alloc() }; }
        uds_check(arr)
    }
    fn sceNpUniversalDataSystemPostEvent(ctx: i32, h: i32, ev: u64, options: u64) -> i32 {
        // The trophies' events reach the port's achievements; nothing goes to a network.
        let e = {
            let g = UDS_DATA.lock();
            g.0.get(&ev).map(|(name, p)| (name.clone(), g.1.get(p).cloned().unwrap_or_default()))
        };
        if let Some((name, props)) = e {
            crate::achievements::event(&name, &props);
        }
        stub_log("sceNpUniversalDataSystemPostEvent (offline: events reach the port's achievements, nothing else)", 0);
        0
    }

    // ---- Game updates: no update is available (offline console).
    fn sceGameUpdateCreateRequest() -> i32 { next_id() }
    fn sceGameUpdateDeleteRequest(req: i32) -> i32 { 0 }
    fn sceGameUpdateAbortRequest(req: i32) -> i32 { 0 }
    /// (request, param {size_t size, ...}, result {size_t size, bool found,
    /// bool addcont_found, ..., char content_version[11], ...})
    fn sceGameUpdateCheck(req: i32, param: *const u64, result: *mut u64) -> i32 {
        if param.is_null() || result.is_null() { return GAME_UPDATE_ERROR_INVALID_ARG; }
        let size = *result;
        if size < 16 || size > 0x400 { return GAME_UPDATE_ERROR_INVALID_SIZE; }
        std::ptr::write_bytes(result as *mut u8, 0, size as usize);
        *result = size;
        stub_log("sceGameUpdateCheck (offline: no update found)", 0);
        0
    }

    // ---- HTTP/2: templates and requests exist; nothing reaches a network,
    // so sends complete with a timeout.
    fn sceHttp2CreateTemplate(ctx: i32, user_agent: *const u8, version: i32, auto_proxy: i32) -> i32 { next_id() }
    fn sceHttp2DeleteTemplate(id: i32) -> i32 { 0 }
    fn sceHttp2CreateRequestWithURL(tmpl: i32, method: *const u8, url: *const u8, length: u64) -> i32 {
        if method.is_null() || url.is_null() { return HTTP2_ERROR_NULL_POINTER; }
        crate::debug!("sceHttp2CreateRequestWithURL {} {}", cstr_or(method), cstr_or(url));
        next_id()
    }
    fn sceHttp2DeleteRequest(id: i32) -> i32 { 0 }
    fn sceHttp2AddRequestHeader(id: i32, name: *const u8, value: *const u8, mode: u32) -> i32 { 0 }
    fn sceHttp2SendRequestAsync(id: i32, data: u64, size: u64, kq: u64, opt: u64) -> i32 {
        stub_log("sceHttp2SendRequestAsync (offline: completes with a timeout)", 0);
        0
    }
    /// Async result {i32 event_type, i32 req_id, i32 result, pad, void* reserved}.
    fn sceHttp2WaitAsync(id: i32, result: *mut i32, timeout: *mut u32, opt: u64) -> i32 {
        if result.is_null() { return HTTP2_ERROR_NULL_POINTER; }
        std::ptr::write_bytes(result as *mut u8, 0, 24);
        *result.add(1) = id;
        *result.add(2) = HTTP2_ERROR_TIMEOUT;
        0
    }
    fn sceHttp2ReadDataAsync(id: i32, data: u64, size: u64, kq: u64, opt: u64) -> i32 { HTTP2_ERROR_BEFORE_SEND }
    fn sceHttp2GetStatusCode(id: i32, status: *mut i32) -> i32 { HTTP2_ERROR_BEFORE_SEND }
    fn sceHttp2GetResponseContentLength(id: i32, result: *mut i32, length: *mut u64) -> i32 { HTTP2_ERROR_BEFORE_SEND }
    fn sceHttp2GetAllResponseHeaders(id: i32, header: *mut u64, size: *mut u64) -> i32 { HTTP2_ERROR_BEFORE_SEND }

    // ---- NP account queries: signed out.
    fn sceNpGetAccountIdA(user: i32, id: *mut u64) -> i32 {
        if !id.is_null() { *id = 0; }
        NP_ERROR_SIGNED_OUT
    }
    fn sceNpGetAccountCountryA(user: i32, code: *mut u8) -> i32 {
        if !code.is_null() { std::ptr::write_bytes(code, 0, 4); }
        NP_ERROR_SIGNED_OUT
    }
    fn sceNpCreateRequest() -> i32 { next_id() }
    fn sceNpCheckPremium(req: i32, param: u64, result: *mut u8) -> i32 {
        if !result.is_null() { std::ptr::write_bytes(result, 0, 16); }
        NP_ERROR_SIGNED_OUT
    }

    // ---- Sign-in dialog: shown, and the user does not sign in.
    fn sceSigninDialogInitialize() -> i32 { *SIGNIN.lock() = (SIGNIN_INITIALIZED, false); 0 }
    fn sceSigninDialogOpen(param: u64) -> i32 {
        crate::info!("sign-in dialog opened (offline console: the user does not sign in)");
        *SIGNIN.lock() = (SIGNIN_RUNNING, false);
        0
    }
    fn sceSigninDialogUpdateStatus() -> i32 {
        let mut g = SIGNIN.lock();
        if g.0 == SIGNIN_RUNNING {
            if g.1 { g.0 = SIGNIN_FINISHED; }
            g.1 = true;
        }
        g.0
    }
    fn sceSigninDialogGetStatus() -> i32 { SIGNIN.lock().0 }
    /// {i32 result (1 = user canceled), i32 reserved[3]}
    fn sceSigninDialogGetResult(result: *mut i32) -> i32 {
        if result.is_null() { return 0x8135_0003u32 as i32; }
        std::ptr::write_bytes(result as *mut u8, 0, 16);
        *result = 1;
        0
    }
    fn sceSigninDialogClose() -> i32 { let mut g = SIGNIN.lock(); if g.0 == SIGNIN_RUNNING { g.0 = SIGNIN_FINISHED; } 0 }
    fn sceSigninDialogTerminate() -> i32 { *SIGNIN.lock() = (0, false); 0 }

    /// Whether a system dialog is on screen (the save-data dialog).
    fn sceCommonDialogIsUsed() -> u32 { super::savedata::dialog_running() as u32 }

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
        sceNpUniversalDataSystemCreateEvent, sceNpUniversalDataSystemDestroyEvent, sceNpUniversalDataSystemCreateEventPropertyObject,
        sceNpUniversalDataSystemDestroyEventPropertyObject, sceNpUniversalDataSystemCreateEventPropertyArray,
        sceNpUniversalDataSystemDestroyEventPropertyArray, sceNpUniversalDataSystemEventPropertyObjectSetString,
        sceNpUniversalDataSystemEventPropertyObjectSetInt32, sceNpUniversalDataSystemEventPropertyObjectSetUInt32,
        sceNpUniversalDataSystemEventPropertyObjectSetUInt64, sceNpUniversalDataSystemEventPropertyObjectSetArray,
        sceNpUniversalDataSystemEventPropertyArraySetString, sceNpUniversalDataSystemEventPropertyArraySetFloat32,
        sceNpUniversalDataSystemEventPropertyArraySetObject, sceNpUniversalDataSystemPostEvent,
    ]);
    reg!(r, "libSceGameUpdate", [sceGameUpdateCreateRequest, sceGameUpdateDeleteRequest, sceGameUpdateAbortRequest, sceGameUpdateCheck]);
    reg!(r, "libSceHttp2", [
        sceHttp2CreateTemplate, sceHttp2DeleteTemplate, sceHttp2CreateRequestWithURL, sceHttp2DeleteRequest, sceHttp2AddRequestHeader,
        sceHttp2SendRequestAsync, sceHttp2WaitAsync, sceHttp2ReadDataAsync, sceHttp2GetStatusCode, sceHttp2GetResponseContentLength,
        sceHttp2GetAllResponseHeaders,
    ]);
    reg!(r, "libSceNpManager", [sceNpGetAccountIdA, sceNpGetAccountCountryA, sceNpCreateRequest, sceNpCheckPremium]);
    reg!(r, "libSceSigninDialog", [
        sceSigninDialogInitialize, sceSigninDialogOpen, sceSigninDialogUpdateStatus, sceSigninDialogGetStatus, sceSigninDialogGetResult,
        sceSigninDialogClose, sceSigninDialogTerminate,
    ]);
    reg!(r, "libSceCommonDialog", [sceCommonDialogIsUsed]);
}
