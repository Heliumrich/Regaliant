//! PS5 system services on the startup path: Json2 (C++ API), HTTP/2, NP Web
//! API 2, game-update check and game intents.
//!
//! This runtime behaves as an offline console: contexts are created so the
//! title's initialization proceeds, but no request ever reaches a network.
//! Each such answer is logged once as a fixed result.

use crate::hle::{Registry, stub_log};
use crate::{reg, sysv};
use std::sync::atomic::{AtomicI32, Ordering};

static NEXT: AtomicI32 = AtomicI32::new(1);
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

    fn sceGameUpdateInitialize(param: u64) -> i32 {
        stub_log("sceGameUpdateInitialize (offline: update checks report no update)", 0);
        0
    }
    fn sceNpGameIntentInitialize(param: u64) -> i32 {
        stub_log("sceNpGameIntentInitialize (no game intents are delivered)", 0);
        0
    }
}

pub fn register(r: &mut Registry) {
    reg!(r, "libSceJson2", [
        _ZN3sce4Json12MemAllocatorC2Ev, _ZN3sce4Json12MemAllocatorD2Ev, _ZN3sce4Json11InitializerC1Ev,
        _ZN3sce4Json11InitializerD1Ev, _ZN3sce4Json11Initializer10initializeEPKNS0_13InitParameterE,
    ]);
    reg!(r, "libSceHttp2", [sceHttp2Init, sceHttp2Term]);
    reg!(r, "libSceNpWebApi2", [sceNpWebApi2Initialize, sceNpWebApi2Terminate]);
    reg!(r, "libSceGameUpdate", [sceGameUpdateInitialize]);
    reg!(r, "libSceNpGameIntent", [sceNpGameIntentInitialize]);
}
