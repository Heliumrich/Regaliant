//! libSceSaveData: save directories and save-data memory backed by host
//! directories under local-data/saves/<user>/<title>/.

use crate::hle::{Registry, cstr};
use crate::{fixed, reg, sysv};
use parking_lot::Mutex;
use std::path::PathBuf;
use std::sync::OnceLock;

const ERR_PARAMETER: i32 = 0x809F_0000u32 as i32;
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
    match super::app::param_sfo().get("TITLE_ID") {
        Some(super::app::SfoValue::Str(s)) => s.clone(),
        _ => "CUSA03173".into(),
    }
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
        if !pat.is_empty() {
            let prefix = pat.trim_end_matches('%').to_string();
            let wildcard = pat.ends_with('%');
            names.retain(|n| if wildcard { n.starts_with(&prefix) } else { *n == pat });
        }
        names.sort();
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
        sceSaveDataInitialize, sceSaveDataInitialize3, sceSaveDataTerminate,
    ]);
}
