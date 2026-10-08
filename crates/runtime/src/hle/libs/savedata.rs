//! libSceSaveData: save directories and save-data memory backed by host
//! directories under local-data/saves/<user>/<title>/<dir>/ (metadata in
//! `sce_sys/`: param.bin, blocks, icon0.png).
//!
//! PS5 API (Mount3/Umount2/Prepare/Commit with transaction resources):
//! structure layouts and error codes per public sources (docs/PROVENANCE.md).
//! Writes go straight to the host directory; Prepare/Commit are accepted
//! (no staging copy yet). The save-data dialog (system UI) is answered
//! automatically and logged; prompts of the delete type are declined.

use crate::hle::{Registry, cstr};
use crate::{fixed, reg, sysv};
use parking_lot::Mutex;
use std::path::PathBuf;
use std::sync::OnceLock;

const ERR_PARAMETER: i32 = 0x809F_0000u32 as i32;
const ERR_BUSY: i32 = 0x809F_0003u32 as i32;
const ERR_INVALID_LOGIN_USER: i32 = 0x809F_0011u32 as i32;
/// Block count reported for saves created without one (32 KiB blocks).
const DEFAULT_BLOCKS: u64 = 1024;
const ERR_NOT_MOUNTED: i32 = 0x809F_0004u32 as i32;
const ERR_EXISTS: i32 = 0x809F_0007u32 as i32;
const ERR_NOT_FOUND: i32 = 0x809F_0008u32 as i32;
const ERR_INTERNAL: i32 = 0x809F_000Bu32 as i32;
const ERR_MOUNT_FULL: i32 = 0x809F_000Cu32 as i32;

const MOUNT_RDONLY: u32 = 1;
const MOUNT_CREATE: u32 = 4;
const MOUNT_CREATE2: u32 = 32;

static ROOT: OnceLock<PathBuf> = OnceLock::new();
static MOUNTS: Mutex<[Option<PathBuf>; 16]> = Mutex::new([const { None }; 16]);

pub fn init(root: PathBuf) {
    ROOT.set(root).ok();
}

fn title_id() -> String {
    if let Some(t) = super::app::param_json().get("titleId").and_then(|v| v.as_str()) {
        return t.to_string();
    }
    match super::app::param_sfo().get("TITLE_ID") {
        Some(super::app::SfoValue::Str(s)) => s.clone(),
        _ => "PPSA01341".into(),
    }
}

/// SQL-LIKE match used by dir-name searches: `%` any run, `_` one character
/// (case-insensitive).
fn dir_name_match(name: &[u8], pat: &[u8]) -> bool {
    match pat.split_first() {
        None => name.is_empty(),
        Some((b'%', rest)) => (0..=name.len()).any(|i| dir_name_match(&name[i..], rest)),
        Some((&c, rest)) => match name.split_first() {
            Some((&n, nrest)) => (c == b'_' || n.eq_ignore_ascii_case(&c)) && dir_name_match(nrest, rest),
            None => false,
        },
    }
}

/// Mounts `path` at the first free /savedataN point.
fn mount_at_free_slot(path: PathBuf, writable: bool) -> Result<String, i32> {
    let mut mounts = MOUNTS.lock();
    if mounts.iter().any(|m| m.as_ref() == Some(&path)) {
        return Err(ERR_BUSY);
    }
    let Some(slot) = mounts.iter().position(|s| s.is_none()) else { return Err(ERR_MOUNT_FULL) };
    mounts[slot] = Some(path.clone());
    let mp = format!("/savedata{slot}");
    crate::hle::kernel::fs::mount(&mp, vec![path], writable);
    Ok(mp)
}

fn read_blocks(dir: &std::path::Path) -> u64 {
    std::fs::read(dir.join("sce_sys").join("blocks"))
        .ok()
        .and_then(|b| b.get(..8).map(|x| u64::from_le_bytes(x.try_into().unwrap())))
        .unwrap_or(DEFAULT_BLOCKS)
}

fn user_root(user: i32, title: *const u8) -> PathBuf {
    let t = if title.is_null() { title_id() } else { unsafe { cstr(title) }.to_string() };
    let t = if t.is_empty() { title_id() } else { t };
    ROOT.get().expect("savedata root").join(format!("{user}")).join(t)
}

fn valid_dir_name(s: &str) -> bool {
    !s.is_empty() && s.len() < 32 && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

unsafe fn mount_point_index(mp: *const u8) -> Option<usize> {
    let s = unsafe { cstr(mp) };
    s.strip_prefix("/savedata").and_then(|n| n.parse::<usize>().ok()).filter(|&n| n < 16)
}

fn mounted_dir(mp: *const u8) -> Option<PathBuf> {
    let i = unsafe { mount_point_index(mp) }?;
    MOUNTS.lock()[i].clone()
}

#[repr(C)]
struct SaveParam {
    title: [u8; 128],
    sub_title: [u8; 128],
    detail: [u8; 1024],
    user_param: u32,
    pad: i32,
    mtime: i64,
    reserved: [u8; 32],
}

fn memory_file(user: i32) -> PathBuf {
    user_root(user, std::ptr::null()).join("sdmemory").join("memory.dat")
}

sysv! {
    fn sceSaveDataMount(m: *const u8, result: *mut u8) -> i32 {
        if m.is_null() || result.is_null() { return ERR_PARAMETER; }
        let user = *(m as *const i32);
        let title = *(m.add(8) as *const u64) as *const u8;
        let dir = *(m.add(16) as *const u64) as *const u8;
        let blocks = *(m.add(32) as *const u64);
        let mode = *(m.add(40) as *const u32);
        if dir.is_null() { return ERR_PARAMETER; }
        let name = cstr(dir).to_string();
        if !valid_dir_name(&name) { return ERR_PARAMETER; }
        let path = user_root(user, title).join(&name);
        let exists = path.join("sce_sys").exists() || path.exists();
        let mut created = false;
        if !exists {
            if mode & (MOUNT_CREATE | MOUNT_CREATE2) == 0 {
                crate::info!("sceSaveDataMount '{name}' mode {mode:#x}: not found");
                return ERR_NOT_FOUND;
            }
            if std::fs::create_dir_all(path.join("sce_sys")).is_err() { return ERR_INTERNAL; }
            created = true;
        } else if mode & MOUNT_CREATE != 0 && mode & MOUNT_CREATE2 == 0 {
            return ERR_EXISTS;
        }
        let mut mounts = MOUNTS.lock();
        let Some(slot) = mounts.iter().position(|s| s.is_none()) else { return ERR_MOUNT_FULL };
        mounts[slot] = Some(path.clone());
        let mp = format!("/savedata{slot}");
        crate::hle::kernel::fs::mount(&mp, vec![path], mode & MOUNT_RDONLY == 0);
        std::ptr::write_bytes(result, 0, 64);
        crate::hle::write_cstr(result, 16, &mp);
        *(result.add(16) as *mut u64) = blocks;
        *(result.add(28) as *mut u32) = if created { 1 } else { 0 };
        crate::info!("sceSaveDataMount '{name}' mode {mode:#x} blocks {blocks} -> {mp}{}", if created { " (created)" } else { "" });
        0
    }
    fn sceSaveDataUmount(mp: *const u8) -> i32 {
        let Some(i) = mount_point_index(mp) else { return ERR_PARAMETER };
        let mut mounts = MOUNTS.lock();
        if mounts[i].take().is_none() { return ERR_NOT_MOUNTED; }
        crate::hle::kernel::fs::unmount(cstr(mp));
        0
    }
    fn sceSaveDataDirNameSearch(cond: *const u8, result: *mut u8) -> i32 {
        if cond.is_null() || result.is_null() { return ERR_PARAMETER; }
        let user = *(cond as *const i32);
        let title = *(cond.add(8) as *const u64) as *const u8;
        let pattern = *(cond.add(16) as *const u64) as *const u8;
        let pat = if pattern.is_null() { String::new() } else { cstr(pattern).to_string() };
        let root = user_root(user, title);
        let mut names: Vec<String> = std::fs::read_dir(&root)
            .map(|rd| rd.flatten().filter(|e| e.path().is_dir()).map(|e| e.file_name().to_string_lossy().into_owned()).collect())
            .unwrap_or_default();
        names.retain(|n| n != "sdmemory");
        names.retain(|n| !n.starts_with("sce_") && valid_dir_name(n));
        if !pat.is_empty() {
            names.retain(|n| dir_name_match(n.as_bytes(), pat.as_bytes()));
        }
        names.sort();
        if *(cond.add(28) as *const u32) == 1 {
            names.reverse(); // descending order
        }
        let dir_names = *(result.add(8) as *const u64) as *mut u8;
        let cap = *(result.add(16) as *const u32) as usize;
        let params = *(result.add(24) as *const u64) as *mut SaveParam;
        let set = names.len().min(cap);
        for (i, n) in names.iter().take(set).enumerate() {
            if !dir_names.is_null() { crate::hle::write_cstr(dir_names.add(i * 32), 32, n); }
            if !params.is_null() {
                let mut p: SaveParam = std::mem::zeroed();
                if let Ok(b) = std::fs::read(root.join(n).join("sce_sys").join("param.bin")) {
                    if b.len() == std::mem::size_of::<SaveParam>() { std::ptr::copy_nonoverlapping(b.as_ptr(), &mut p as *mut _ as *mut u8, b.len()); }
                }
                std::ptr::write_unaligned(params.add(i), p);
            }
        }
        // SaveDataSearchInfo {u64 blocks, u64 free_blocks, reserved[32]} per hit.
        let infos = *(result.add(32) as *const u64) as *mut u64;
        if !infos.is_null() {
            for (i, n) in names.iter().take(set).enumerate() {
                let b = read_blocks(&root.join(n));
                std::ptr::write_bytes(infos.add(i * 6) as *mut u8, 0, 48);
                *infos.add(i * 6) = b;
                *infos.add(i * 6 + 1) = b;
            }
        }
        *(result as *mut u32) = names.len() as u32;
        *(result.add(20) as *mut u32) = set as u32;
        crate::info!("sceSaveDataDirNameSearch '{pat}': {} hit(s)", names.len());
        0
    }
    fn sceSaveDataSetParam(mp: *const u8, kind: u32, buf: *const u8, size: u64) -> i32 {
        let Some(dir) = mounted_dir(mp) else { return ERR_NOT_MOUNTED };
        let file = dir.join("sce_sys").join("param.bin");
        let mut p: SaveParam = std::mem::zeroed();
        if let Ok(b) = std::fs::read(&file) {
            if b.len() == std::mem::size_of::<SaveParam>() { std::ptr::copy_nonoverlapping(b.as_ptr(), &mut p as *mut _ as *mut u8, b.len()); }
        }
        match kind {
            0 if size as usize >= std::mem::size_of::<SaveParam>() => std::ptr::copy_nonoverlapping(buf, &mut p as *mut _ as *mut u8, std::mem::size_of::<SaveParam>()),
            1 => crate::hle::write_cstr(p.title.as_mut_ptr(), 128, cstr(buf)),
            2 => crate::hle::write_cstr(p.sub_title.as_mut_ptr(), 128, cstr(buf)),
            3 => crate::hle::write_cstr(p.detail.as_mut_ptr(), 1024, cstr(buf)),
            4 => p.user_param = *(buf as *const u32),
            _ => return ERR_PARAMETER,
        }
        p.mtime = (super::rtc::now_tick() / 1_000_000) as i64 - 62_135_596_800;
        let bytes = std::slice::from_raw_parts(&p as *const _ as *const u8, std::mem::size_of::<SaveParam>());
        let _ = std::fs::create_dir_all(dir.join("sce_sys"));
        if std::fs::write(&file, bytes).is_err() { return ERR_INTERNAL; }
        0
    }
    fn sceSaveDataSaveIcon(mp: *const u8, icon: *const u64) -> i32 {
        let Some(dir) = mounted_dir(mp) else { return ERR_NOT_MOUNTED };
        if icon.is_null() { return ERR_PARAMETER; }
        let buf = *icon as *const u8;
        let size = *icon.add(2) as usize;
        if buf.is_null() || size == 0 { return ERR_PARAMETER; }
        let _ = std::fs::create_dir_all(dir.join("sce_sys"));
        if std::fs::write(dir.join("sce_sys").join("icon0.png"), std::slice::from_raw_parts(buf, size)).is_err() { return ERR_INTERNAL; }
        0
    }
    fn sceSaveDataDelete(del: *const u8) -> i32 {
        if del.is_null() { return ERR_PARAMETER; }
        let user = *(del as *const i32);
        let title = *(del.add(8) as *const u64) as *const u8;
        let dir = *(del.add(16) as *const u64) as *const u8;
        if dir.is_null() { return ERR_PARAMETER; }
        let name = cstr(dir).to_string();
        if !valid_dir_name(&name) { return ERR_PARAMETER; }
        let path = user_root(user, title).join(&name);
        if !path.exists() { return ERR_NOT_FOUND; }
        crate::info!("sceSaveDataDelete '{name}'");
        if std::fs::remove_dir_all(&path).is_err() { return ERR_INTERNAL; }
        0
    }
    fn sceSaveDataSetupSaveDataMemory(user: i32, size: u64, param: *const u8) -> i32 {
        let f = memory_file(user);
        let _ = std::fs::create_dir_all(f.parent().unwrap());
        let cur = std::fs::metadata(&f).map(|m| m.len()).unwrap_or(0);
        if cur < size {
            let file = match std::fs::OpenOptions::new().create(true).write(true).truncate(false).open(&f) { Ok(x) => x, Err(_) => return ERR_INTERNAL };
            if file.set_len(size).is_err() { return ERR_INTERNAL; }
        }
        crate::info!("sceSaveDataSetupSaveDataMemory user {user} size {size:#x}");
        0
    }
    fn sceSaveDataGetSaveDataMemory(user: i32, buf: *mut u8, size: u64, offset: i64) -> i32 {
        use std::os::windows::fs::FileExt;
        let f = match std::fs::File::open(memory_file(user)) { Ok(f) => f, Err(_) => return ERR_NOT_FOUND };
        let out = std::slice::from_raw_parts_mut(buf, size as usize);
        out.fill(0);
        let _ = f.seek_read(out, offset as u64);
        0
    }
    // ---- PS5: Mount3 {i32 user, pad, dir_name* @8, u64 blocks @16,
    // u64 system_blocks @24, u32 mode @32, pad, i32 resource @40, reserved}
    // -> MountResult {mount_point[16], u64 required_blocks @16, u32 @24,
    // u32 mount_status @28 (1 = created), reserved} (64 bytes).
    fn sceSaveDataMount3(m: *const u8, result: *mut u8) -> i32 {
        if m.is_null() || result.is_null() { return ERR_PARAMETER; }
        let user = *(m as *const i32);
        let dir = *(m.add(8) as *const u64) as *const u8;
        let blocks = *(m.add(16) as *const u64);
        let mode = *(m.add(32) as *const u32);
        let resource = *(m.add(40) as *const i32);
        if dir.is_null() { return ERR_PARAMETER; }
        if user < 0 { return ERR_INVALID_LOGIN_USER; }
        let name = cstr(dir).to_string();
        if !valid_dir_name(&name) { return ERR_PARAMETER; }
        let path = user_root(user, std::ptr::null()).join(&name);
        let exists = path.join("sce_sys").exists();
        let mut created = false;
        if !exists {
            if mode & (MOUNT_CREATE | MOUNT_CREATE2) == 0 {
                crate::info!("sceSaveDataMount3 '{name}' mode {mode:#x}: not found");
                return ERR_NOT_FOUND;
            }
            if std::fs::create_dir_all(path.join("sce_sys")).is_err() { return ERR_INTERNAL; }
            let b = if blocks == 0 { DEFAULT_BLOCKS } else { blocks };
            if std::fs::write(path.join("sce_sys").join("blocks"), b.to_le_bytes()).is_err() { return ERR_INTERNAL; }
            created = true;
        } else if mode & MOUNT_CREATE != 0 && mode & MOUNT_CREATE2 == 0 {
            return ERR_EXISTS;
        }
        let mp = match mount_at_free_slot(path, mode & MOUNT_RDONLY == 0) { Ok(x) => x, Err(e) => return e };
        std::ptr::write_bytes(result, 0, 64);
        crate::hle::write_cstr(result, 16, &mp);
        *(result.add(28) as *mut u32) = if created { 1 } else { 0 };
        crate::info!("sceSaveDataMount3 '{name}' mode {mode:#x} blocks {blocks} resource {resource} -> {mp}{}", if created { " (created)" } else { "" });
        0
    }
    fn sceSaveDataUmount2(mode: u32, mp: *const u8) -> i32 {
        if mp.is_null() { return ERR_PARAMETER; }
        let Some(i) = mount_point_index(mp) else { return ERR_NOT_FOUND };
        let mut mounts = MOUNTS.lock();
        if mounts[i].take().is_none() { return ERR_NOT_FOUND; }
        crate::hle::kernel::fs::unmount(cstr(mp));
        crate::debug!("sceSaveDataUmount2 {} mode {mode:#x}", cstr(mp));
        0
    }
    fn sceSaveDataCreateTransactionResource(size: u32) -> i32 {
        static NEXT: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(1);
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    }
    fn sceSaveDataDeleteTransactionResource(resource: i32) -> i32 { 0 }
    /// (mount point, {i32 resource, u32 prepare_mode, reserved})
    fn sceSaveDataPrepare(mp: *const u8, param: *const u8) -> i32 {
        if mp.is_null() || param.is_null() { return ERR_PARAMETER; }
        if mounted_dir(mp).is_none() { return ERR_NOT_MOUNTED; }
        0
    }
    /// ({i32 resource, u32 commit_mode, reserved}): data is already on disk.
    fn sceSaveDataCommit(param: *const u8) -> i32 {
        if param.is_null() { return ERR_PARAMETER; }
        crate::debug!("sceSaveDataCommit resource {} mode {:#x}", *(param as *const i32), *(param.add(4) as *const u32));
        0
    }
    /// (mount point, {u64 blocks, u64 free_blocks, reserved[32]})
    fn sceSaveDataGetMountInfo(mp: *const u8, info: *mut u64) -> i32 {
        if mp.is_null() || info.is_null() { return ERR_PARAMETER; }
        let Some(dir) = mounted_dir(mp) else { return ERR_NOT_MOUNTED };
        let b = read_blocks(&dir);
        std::ptr::write_bytes(info as *mut u8, 0, 48);
        *info = b;
        *info.add(1) = b;
        0
    }

    fn sceSaveDataSetSaveDataMemory(user: i32, buf: *const u8, size: u64, offset: i64) -> i32 {
        use std::os::windows::fs::FileExt;
        let f = match std::fs::OpenOptions::new().write(true).open(memory_file(user)) { Ok(f) => f, Err(_) => return ERR_NOT_FOUND };
        if f.seek_write(std::slice::from_raw_parts(buf, size as usize), offset as u64).is_err() { return ERR_INTERNAL; }
        0
    }
}

fixed! {
    sceSaveDataInitialize => 0,
    sceSaveDataInitialize3 => 0,
    sceSaveDataTerminate => 0,
}

pub fn register(r: &mut Registry) {
    reg!(r, "libSceSaveData", [
        sceSaveDataMount, sceSaveDataUmount, sceSaveDataDirNameSearch, sceSaveDataSetParam, sceSaveDataSaveIcon,
        sceSaveDataDelete, sceSaveDataSetupSaveDataMemory, sceSaveDataGetSaveDataMemory, sceSaveDataSetSaveDataMemory,
        sceSaveDataInitialize, sceSaveDataInitialize3, sceSaveDataTerminate, sceSaveDataMount3, sceSaveDataUmount2,
        sceSaveDataCreateTransactionResource, sceSaveDataDeleteTransactionResource, sceSaveDataPrepare, sceSaveDataCommit,
        sceSaveDataGetMountInfo,
    ]);
    reg!(r, "libSceSaveDataDialog", [
        sceSaveDataDialogInitialize, sceSaveDataDialogTerminate, sceSaveDataDialogOpen, sceSaveDataDialogClose,
        sceSaveDataDialogGetStatus, sceSaveDataDialogUpdateStatus, sceSaveDataDialogGetResult,
        sceSaveDataDialogIsReadyToDisplay, sceSaveDataDialogProgressBarSetValue, sceSaveDataDialogProgressBarInc,
    ]);
}

// ---------------------------------------------------------------------------
// Save-data dialog (system UI). No overlay is drawn yet: each dialog is
// logged and answered. Dialogs with buttons finish after one status update
// (titles expect to observe RUNNING first); progress dialogs stay open until
// the title closes them. Delete-type prompts are declined so no save is
// removed without the user's decision.

const DLG_NONE: i32 = 0;
const DLG_INITIALIZED: i32 = 1;
const DLG_RUNNING: i32 = 2;
const DLG_FINISHED: i32 = 3;

const MODE_USER_MSG: i32 = 2;
const MODE_SYSTEM_MSG: i32 = 3;
const MODE_PROGRESS_BAR: i32 = 5;
const DISP_DELETE: i32 = 3;
const SYSMSG_PROGRESS: i32 = 5;
const USER_BUTTONS_NONE: i32 = 2;

const BUTTON_OK_YES: i32 = 1;
const BUTTON_NO: i32 = 2;
const RESULT_OK: i32 = 0;
const RESULT_CANCELED: i32 = 1;
const ERR_DIALOG_NOT_FINISHED: i32 = 0x80B8_0006u32 as i32;

struct Dialog {
    status: i32,
    polled: bool,
    holds: bool,
    mode: i32,
    result: i32,
    button: i32,
    user_data: u64,
    dir_name: String,
}

static DIALOG: Mutex<Dialog> =
    Mutex::new(Dialog { status: DLG_NONE, polled: false, holds: false, mode: 0, result: 0, button: 0, user_data: 0, dir_name: String::new() });

sysv! {
    fn sceSaveDataDialogInitialize() -> i32 {
        let mut d = DIALOG.lock();
        d.status = DLG_INITIALIZED;
        d.polled = false;
        0
    }
    fn sceSaveDataDialogTerminate() -> i32 {
        DIALOG.lock().status = DLG_NONE;
        0
    }
    /// SceSaveDataDialogParam: base[48], i32 size @48, i32 mode @52,
    /// i32 dispType @56, anim* @64, items* @72, userMsg* @80, sysMsg* @88,
    /// errorCode* @96, progBar* @104, userData @112.
    fn sceSaveDataDialogOpen(p: *const u8) -> i32 {
        if p.is_null() { return ERR_PARAMETER; }
        let mode = *(p.add(52) as *const i32);
        let disp = *(p.add(56) as *const i32);
        let items = *(p.add(72) as *const u64) as *const u8;
        let user_msg = *(p.add(80) as *const u64) as *const u8;
        let sys_msg = *(p.add(88) as *const u64) as *const u8;
        let prog = *(p.add(104) as *const u64) as *const u8;
        let user_data = *(p.add(112) as *const u64);
        let mut desc = format!("mode {mode} type {disp}");
        let mut has_buttons = true;
        if mode == MODE_USER_MSG && !user_msg.is_null() {
            let buttons = *(user_msg as *const i32);
            let msg = *(user_msg.add(8) as *const u64) as *const u8;
            desc.push_str(&format!(" user message (buttons {buttons}) {:?}", if msg.is_null() { "" } else { cstr(msg) }));
            has_buttons = buttons != USER_BUTTONS_NONE;
        }
        if mode == MODE_SYSTEM_MSG && !sys_msg.is_null() {
            let t = *(sys_msg as *const i32);
            desc.push_str(&format!(" system message {t}"));
            has_buttons = t != SYSMSG_PROGRESS;
        }
        if mode == MODE_PROGRESS_BAR {
            if !prog.is_null() {
                let msg = *(prog.add(8) as *const u64) as *const u8;
                if !msg.is_null() {
                    desc.push_str(&format!(" progress {:?}", cstr(msg)));
                }
            }
            has_buttons = false;
        }
        let mut dir_name = String::new();
        if !items.is_null() {
            let names = *(items.add(16) as *const u64) as *const u8;
            let n = *(items.add(24) as *const u32);
            if !names.is_null() && n > 0 { dir_name = cstr(names).to_string(); }
        }
        let decline = disp == DISP_DELETE;
        let outcome = if !has_buttons {
            "shown until closed"
        } else if decline {
            "declined (delete prompts are not confirmed automatically)"
        } else {
            "accepted"
        };
        let dir_desc = if dir_name.is_empty() { String::new() } else { format!(" dir '{dir_name}'") };
        crate::info!("save-data dialog: {desc}{dir_desc} -> {outcome}");
        let mut d = DIALOG.lock();
        d.status = DLG_RUNNING;
        d.polled = false;
        d.holds = !has_buttons;
        d.mode = mode;
        d.user_data = user_data;
        d.dir_name = dir_name;
        (d.result, d.button) = if decline { (RESULT_CANCELED, BUTTON_NO) } else { (RESULT_OK, BUTTON_OK_YES) };
        0
    }
    fn sceSaveDataDialogClose(param: *const u8) -> i32 {
        let mut d = DIALOG.lock();
        if d.status == DLG_RUNNING { d.status = DLG_FINISHED; }
        0
    }
    fn sceSaveDataDialogGetStatus() -> i32 { DIALOG.lock().status }
    fn sceSaveDataDialogUpdateStatus() -> i32 {
        let mut d = DIALOG.lock();
        if d.status == DLG_RUNNING && !d.holds {
            if d.polled { d.status = DLG_FINISHED; }
            d.polled = true;
        }
        d.status
    }
    /// SceSaveDataDialogResult: i32 mode, i32 result, i32 buttonId, pad,
    /// dirName* @16, param* @24, userData @32.
    fn sceSaveDataDialogGetResult(r: *mut u8) -> i32 {
        if r.is_null() { return ERR_PARAMETER; }
        let d = DIALOG.lock();
        if d.status != DLG_FINISHED { return ERR_DIALOG_NOT_FINISHED; }
        *(r as *mut i32) = d.mode;
        *(r.add(4) as *mut i32) = d.result;
        *(r.add(8) as *mut i32) = d.button;
        let dn = *(r.add(16) as *const u64) as *mut u8;
        if !dn.is_null() && !d.dir_name.is_empty() { crate::hle::write_cstr(dn, 32, &d.dir_name); }
        *(r.add(32) as *mut u64) = d.user_data;
        0
    }
    fn sceSaveDataDialogIsReadyToDisplay() -> i32 { 1 }
    fn sceSaveDataDialogProgressBarSetValue(target: i32, rate: u32) -> i32 { 0 }
    fn sceSaveDataDialogProgressBarInc(target: i32, delta: u32) -> i32 { 0 }
}
