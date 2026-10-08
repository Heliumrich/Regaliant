//! Application services: AppContent (+ param.sfo), UserService, SystemService, PlayGo.

use crate::hle::{Registry, cstr, write_cstr};
use crate::{fixed, reg, sysv};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Debug, Clone)]
pub enum SfoValue {
    Int(u32),
    Str(String),
}

/// Parses `/app0/sce_sys/param.sfo` once.
pub fn param_sfo() -> &'static HashMap<String, SfoValue> {
    static SFO: OnceLock<HashMap<String, SfoValue>> = OnceLock::new();
    SFO.get_or_init(|| {
        let mut m = HashMap::new();
        let Some(p) = crate::hle::kernel::fs::host_path_existing("/app0/sce_sys/param.sfo") else {
            crate::warn!("param.sfo not found");
            return m;
        };
        let Ok(b) = std::fs::read(&p) else { return m };
        let rd32 = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
        let rd16 = |o: usize| u16::from_le_bytes(b[o..o + 2].try_into().unwrap());
        if b.len() < 20 || rd32(0) != 0x4653_5000 {
            crate::warn!("param.sfo has a bad header");
            return m;
        }
        let (keys, data, n) = (rd32(8) as usize, rd32(12) as usize, rd32(16) as usize);
        for i in 0..n {
            let e = 20 + i * 16;
            if e + 16 > b.len() {
                break;
            }
            let ko = keys + rd16(e) as usize;
            let fmt = rd16(e + 2);
            let len = rd32(e + 4) as usize;
            let off = data + rd32(e + 12) as usize;
            let kend = b[ko..].iter().position(|&c| c == 0).map(|p| ko + p).unwrap_or(b.len());
            let key = String::from_utf8_lossy(&b[ko..kend]).into_owned();
            let v = if fmt == 0x0404 {
                SfoValue::Int(rd32(off))
            } else {
                let s = &b[off..off + len];
                let end = s.iter().position(|&c| c == 0).unwrap_or(s.len());
                SfoValue::Str(String::from_utf8_lossy(&s[..end]).into_owned())
            };
            m.insert(key, v);
        }
        crate::info!("param.sfo: TITLE_ID={:?} APP_VER={:?} CATEGORY={:?}", m.get("TITLE_ID"), m.get("APP_VER"), m.get("CATEGORY"));
        m
    })
}

fn sfo_int(key: &str) -> Option<u32> {
    match param_sfo().get(key) {
        Some(SfoValue::Int(v)) => Some(*v),
        _ => None,
    }
}

/// Parses `/app0/sce_sys/param.json` (PS5) once.
pub fn param_json() -> &'static serde_json::Value {
    static J: OnceLock<serde_json::Value> = OnceLock::new();
    J.get_or_init(|| {
        let Some(p) = crate::hle::kernel::fs::host_path_existing("/app0/sce_sys/param.json") else {
            crate::warn!("param.json not found");
            return serde_json::Value::Null;
        };
        match std::fs::read(&p).map_err(|e| e.to_string()).and_then(|b| serde_json::from_slice(&b).map_err(|e| e.to_string())) {
            Ok(v) => v,
            Err(e) => {
                crate::warn!("param.json unreadable: {e}");
                serde_json::Value::Null
            }
        }
    })
}

fn json_int(key: &str) -> Option<i64> {
    param_json().get(key).and_then(|v| v.as_i64())
}

pub const USER_ID: i32 = 1;
static USER_EVENT_SENT: AtomicBool = AtomicBool::new(false);
static PLAYGO_CHUNKS: Mutex<Option<u32>> = Mutex::new(None);

/// PS5 packages define chunks in `playgo-chunkdefs.xml` (Bluepoint packaging
/// file at the application root). Every enabled chunk is installed in the
/// supplied tree, so all report as local.
fn playgo_chunk_count() -> u32 {
    let mut g = PLAYGO_CHUNKS.lock();
    if let Some(n) = *g {
        return n;
    }
    let n = crate::hle::kernel::fs::host_path_existing("/app0/playgo-chunkdefs.xml")
        .and_then(|p| std::fs::read_to_string(p).ok())
        .map(|x| x.matches("<chunk id=").count() as u32)
        .filter(|&n| n > 0)
        .unwrap_or(1);
    crate::info!("PlayGo: {n} chunk definition(s), all reported installed (local)");
    *g = Some(n);
    n
}

static PLAYGO_SPEED: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(1); // TRICKLE

sysv! {
    // ---- AppContent
    fn sceAppContentInitialize(init: *const u8, boot: *mut u8) -> i32 {
        if !boot.is_null() { std::ptr::write_bytes(boot, 0, 40); }
        let j = param_json();
        crate::info!("param.json: {} {} {}", j.get("titleId").and_then(|v| v.as_str()).unwrap_or("?"),
            j.get("contentId").and_then(|v| v.as_str()).unwrap_or("?"), j.get("contentVersion").and_then(|v| v.as_str()).unwrap_or("?"));
        0
    }
    fn sceAppContentAppParamGetInt(id: i32, value: *mut i32) -> i32 {
        if value.is_null() { return 0x80D9_0002u32 as i32; }
        *value = match id {
            0 => 3, // SKU_FLAG_FULL
            1..=4 => json_int(&format!("userDefinedParam{id}")).unwrap_or(0) as i32,
            _ => return 0x80D9_0002u32 as i32,
        };
        crate::debug!("sceAppContentAppParamGetInt({id}) = {}", *value);
        0
    }
    fn sceAppContentGetAddcontInfoList(label: u32, list: *mut u8, num: u32, hit: *mut u32) -> i32 {
        if !hit.is_null() { *hit = 0; }
        crate::info!("sceAppContentGetAddcontInfoList: no additional content installed");
        0
    }

    // ---- UserService
    fn sceUserServiceInitialize(params: *const u8) -> i32 { 0 }
    fn sceUserServiceTerminate() -> i32 { 0 }
    fn sceUserServiceGetInitialUser(user: *mut i32) -> i32 {
        if user.is_null() { return 0x8096_0003u32 as i32; }
        *user = USER_ID;
        0
    }
    fn sceUserServiceGetLoginUserIdList(list: *mut i32) -> i32 {
        if list.is_null() { return 0x8096_0003u32 as i32; }
        *list = USER_ID;
        for i in 1..4 { *list.add(i) = -1; }
        0
    }
    fn sceUserServiceGetUserName(user: i32, name: *mut u8, size: u64) -> i32 {
        write_cstr(name, size as usize, "Hunter");
        0
    }
    fn sceUserServiceGetEvent(ev: *mut i32) -> i32 {
        if !USER_EVENT_SENT.swap(true, Ordering::SeqCst) {
            *ev = 0; // LOGIN
            *ev.add(1) = USER_ID;
            return 0;
        }
        0x8096_0007u32 as i32 // NO_EVENT
    }

    // ---- SystemService
    fn sceSystemServiceGetStatus(status: *mut u8) -> i32 {
        // SceSystemServiceStatus: int32 eventNum + 5 bools (+ reserved, left untouched).
        if !status.is_null() { std::ptr::write_bytes(status, 0, 9); }
        0
    }
    fn sceSystemServiceReceiveEvent(ev: *mut u8) -> i32 { 0x80A1_0003u32 as i32 }
    fn sceSystemServiceParamGetInt(id: i32, value: *mut i32) -> i32 {
        if value.is_null() { return 0x80A1_0002u32 as i32; }
        *value = match id {
            1 => 1,    // LANG: English (United States)
            2 => 0,    // DATE_FORMAT: YYYYMMDD
            3 => 1,    // TIME_FORMAT: 24 hour
            4 => 0,    // TIME_ZONE (minutes)
            5 => 0,    // SUMMERTIME
            7 => 0,    // GAME_PARENTAL_LEVEL: off
            1000 => 1, // ENTER_BUTTON_ASSIGN: cross
            _ => {
                crate::warn!("sceSystemServiceParamGetInt: unknown id {id}");
                return 0x80A1_0002u32 as i32;
            }
        };
        0
    }
    fn sceSystemServiceParamGetString(id: i32, buf: *mut u8, size: u64) -> i32 {
        write_cstr(buf, size as usize, "PS4");
        0
    }
    /// SceSystemServiceDisplaySafeAreaInfo {float ratio; reserved[128]}: the
    /// whole display is safe (the console default).
    fn sceSystemServiceGetDisplaySafeAreaInfo(info: *mut f32) -> i32 {
        if info.is_null() { return 0x80A1_0002u32 as i32; }
        std::ptr::write_bytes(info as *mut u8, 0, 4 + 128);
        *info = 1.0;
        0
    }
    fn sceSystemServiceHideSplashScreen() -> i32 {
        crate::info!("sceSystemServiceHideSplashScreen");
        0
    }

    // ---- PlayGo
    fn scePlayGoOpen(handle: *mut i32, param: *const u8) -> i32 {
        if !handle.is_null() { *handle = 1; }
        playgo_chunk_count();
        0
    }
    fn scePlayGoGetLocus(handle: i32, ids: *const u16, n: u32, loci: *mut i8) -> i32 {
        for i in 0..n as usize { *loci.add(i) = 3; } // LOCAL_FAST
        0
    }
    fn scePlayGoGetInstallSpeed(handle: i32, speed: *mut i32) -> i32 {
        if speed.is_null() { return 0x80B2_0002u32 as i32; }
        *speed = PLAYGO_SPEED.load(Ordering::Relaxed);
        0
    }
    fn scePlayGoSetInstallSpeed(handle: i32, speed: i32) -> i32 {
        PLAYGO_SPEED.store(speed, Ordering::Relaxed);
        0
    }
    fn scePlayGoGetEta(handle: i32, ids: *const u16, n: u32, eta: *mut i64) -> i32 {
        if !eta.is_null() { *eta = 0; } // installed
        0
    }
    fn scePlayGoGetToDoList(handle: i32, todo: *mut u8, n: u32, entries: *mut u32) -> i32 {
        if !entries.is_null() { *entries = 0; } // nothing left to install
        0
    }
    fn scePlayGoPrefetch(handle: i32, ids: *const u16, n: u32, min_locus: i32) -> i32 { 0 }
    fn scePlayGoGetChunkId(handle: i32, out: *mut u16, n: u32, entries: *mut u32) -> i32 {
        let count = playgo_chunk_count();
        if out.is_null() {
            if !entries.is_null() { *entries = count; }
            return 0;
        }
        let k = count.min(n);
        for i in 0..k { *out.add(i as usize) = i as u16; }
        if !entries.is_null() { *entries = k; }
        0
    }
}

fixed! {
    scePlayGoInitialize => 0,
    sceSystemServiceLaunchWebBrowser => 0x80A1_0002u32,
}

pub fn register(r: &mut Registry) {
    reg!(r, "libSceAppContent", [sceAppContentInitialize, sceAppContentAppParamGetInt, sceAppContentGetAddcontInfoList]);
    reg!(r, "libSceUserService", [
        sceUserServiceInitialize, sceUserServiceTerminate, sceUserServiceGetInitialUser,
        sceUserServiceGetLoginUserIdList, sceUserServiceGetUserName, sceUserServiceGetEvent,
    ]);
    reg!(r, "libSceSystemService", [
        sceSystemServiceGetStatus, sceSystemServiceReceiveEvent, sceSystemServiceParamGetInt,
        sceSystemServiceParamGetString, sceSystemServiceHideSplashScreen, sceSystemServiceGetDisplaySafeAreaInfo, sceSystemServiceLaunchWebBrowser,
    ]);
    reg!(r, "libScePlayGo", [
        scePlayGoInitialize, scePlayGoOpen, scePlayGoGetLocus, scePlayGoGetChunkId, scePlayGoSetInstallSpeed,
        scePlayGoGetInstallSpeed, scePlayGoGetEta, scePlayGoGetToDoList, scePlayGoPrefetch,
    ]);
    let _ = cstr; // shared helper import
}
