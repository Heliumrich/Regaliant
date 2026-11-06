//! Guest file system: mount table, path resolution and file descriptors.
//!
//! `/app0` layers the update tree over the base tree and is read-only. Writable
//! mounts live under the project's local-data runtime directory. Paths are
//! normalized component by component and can never escape a mount root.

use super::errno::*;
use super::{posix_ret, sce};
use crate::hle::{Registry, cstr};
use crate::{reg, sysv};
use parking_lot::{Mutex, RwLock};
use std::fs::{File, OpenOptions};
use std::os::windows::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub const O_WRONLY: i32 = 0x1;
pub const O_RDWR: i32 = 0x2;
pub const O_ACCMODE: i32 = 0x3;
pub const O_APPEND: i32 = 0x8;
pub const O_CREAT: i32 = 0x200;
pub const O_TRUNC: i32 = 0x400;
pub const O_EXCL: i32 = 0x800;
pub const O_DIRECTORY: i32 = 0x20000;

#[derive(Clone, Debug)]
pub struct Mount {
    pub prefix: String,
    pub layers: Vec<PathBuf>,
    pub writable: bool,
}

static MOUNTS: RwLock<Vec<Mount>> = RwLock::new(Vec::new());

pub fn mount(prefix: &str, layers: Vec<PathBuf>, writable: bool) {
    let mut m = MOUNTS.write();
    m.retain(|x| x.prefix != prefix);
    crate::info!("mount {prefix} -> {:?} writable={writable}", layers);
    m.push(Mount { prefix: prefix.to_string(), layers, writable });
    m.sort_by(|a, b| b.prefix.len().cmp(&a.prefix.len()));
}

pub fn unmount(prefix: &str) {
    MOUNTS.write().retain(|x| x.prefix != prefix);
}

fn normalize(path: &str) -> Option<Vec<String>> {
    let mut out: Vec<String> = Vec::new();
    for c in path.split('/') {
        match c {
            "" | "." => {}
            ".." => {
                out.pop()?;
            }
            c => {
                if c.chars().any(|ch| matches!(ch, ':' | '\\' | '*' | '?' | '"' | '<' | '>' | '|') || (ch as u32) < 0x20) {
                    return None;
                }
                out.push(c.to_string());
            }
        }
    }
    Some(out)
}

pub struct Resolved {
    pub candidates: Vec<PathBuf>,
    pub writable: bool,
}

pub fn resolve(guest: &str) -> Option<Resolved> {
    let abs = if guest.starts_with('/') { guest.to_string() } else { format!("/app0/{guest}") };
    let comps = normalize(&abs)?;
    let mounts = MOUNTS.read();
    for m in mounts.iter() {
        let pre = normalize(&m.prefix)?;
        if comps.len() >= pre.len() && comps[..pre.len()] == pre[..] {
            let rel: PathBuf = comps[pre.len()..].iter().collect();
            return Some(Resolved { candidates: m.layers.iter().map(|l| l.join(&rel)).collect(), writable: m.writable });
        }
    }
    None
}

/// Host path for reading (first existing layer).
pub fn host_path_existing(guest: &str) -> Option<PathBuf> {
    let r = resolve(guest)?;
    if !r.writable {
        return r.candidates.into_iter().find(|p| ro_meta(p).is_some());
    }
    r.candidates.into_iter().find(|p| p.exists())
}

/// Metadata of a path on a read-only mount.
#[derive(Clone, Copy, Debug)]
pub struct CachedMeta {
    pub is_dir: bool,
    pub size: u64,
    pub mtime: [i64; 2],
}

/// Directory listings of read-only mounts (the protected game tree does not
/// change during a run): each directory is enumerated once, and existence,
/// size and stat queries become lookups. On Windows the listing carries
/// sizes and times, so no file is opened. Names are case-insensitive, as on
/// the host file system.
static RO_DIRS: Mutex<Option<std::collections::HashMap<PathBuf, Arc<std::collections::HashMap<String, CachedMeta>>>>> = Mutex::new(None);

fn ro_dir(dir: &Path) -> Option<Arc<std::collections::HashMap<String, CachedMeta>>> {
    if let Some(d) = RO_DIRS.lock().as_ref().and_then(|m| m.get(dir).cloned()) {
        return Some(d);
    }
    let mut listing = std::collections::HashMap::new();
    match std::fs::read_dir(dir) {
        Ok(rd) => {
            for e in rd.flatten() {
                let Ok(md) = e.metadata() else { continue };
                let mtime = md
                    .modified()
                    .ok()
                    .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                    .map(|d| [d.as_secs() as i64, d.subsec_nanos() as i64])
                    .unwrap_or([0, 0]);
                listing.insert(e.file_name().to_string_lossy().to_lowercase(), CachedMeta { is_dir: md.is_dir(), size: md.len(), mtime });
            }
        }
        Err(_) => {
            // Missing directory: cache the empty listing too.
        }
    }
    let d = Arc::new(listing);
    RO_DIRS.lock().get_or_insert_with(Default::default).insert(dir.to_path_buf(), d.clone());
    Some(d)
}

/// Cached metadata of a host path on a read-only mount (mount roots included).
pub fn ro_meta(host: &Path) -> Option<CachedMeta> {
    match (host.parent(), host.file_name()) {
        (Some(parent), Some(name)) => ro_dir(parent)?.get(&name.to_string_lossy().to_lowercase()).copied(),
        _ => std::fs::metadata(host).ok().map(|md| CachedMeta { is_dir: md.is_dir(), size: md.len(), mtime: [0, 0] }),
    }
}

/// Host path and metadata of a guest path (first existing layer).
pub fn guest_meta(guest: &str) -> Option<(PathBuf, CachedMeta)> {
    let r = resolve(guest)?;
    if !r.writable {
        let found = r.candidates.into_iter().find_map(|p| ro_meta(&p).map(|m| (p, m)));
        // A title file the port extends: its extended size.
        if let (Some((p, m)), Some(d)) = (&found, crate::pcsettings::generated(guest)) {
            return Some((p.clone(), CachedMeta { size: d.len() as u64, ..*m }));
        }
        return found;
    }
    let p = r.candidates.into_iter().find(|p| p.exists())?;
    let md = std::fs::metadata(&p).ok()?;
    let mtime = md
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| [d.as_secs() as i64, d.subsec_nanos() as i64])
        .unwrap_or([0, 0]);
    Some((p, CachedMeta { is_dir: md.is_dir(), size: md.len(), mtime }))
}

// ---------------------------------------------------------------------------
// File descriptors

pub enum Kind {
    File { file: File, writable: bool },
    /// Content the port serves from memory (`crate::pcsettings`: title files it extends).
    Mem(Arc<Vec<u8>>),
    Dir { entries: Vec<(String, bool)> },
    Console(i32),
}

pub struct Fd {
    pub kind: Kind,
    pub pos: Mutex<u64>,
    pub path: String,
}

static FDS: Mutex<Vec<Option<Arc<Fd>>>> = Mutex::new(Vec::new());

fn alloc_fd(fd: Fd) -> i32 {
    let mut t = FDS.lock();
    if t.is_empty() {
        t.push(Some(Arc::new(Fd { kind: Kind::Console(0), pos: Mutex::new(0), path: "stdin".into() })));
        t.push(Some(Arc::new(Fd { kind: Kind::Console(1), pos: Mutex::new(0), path: "stdout".into() })));
        t.push(Some(Arc::new(Fd { kind: Kind::Console(2), pos: Mutex::new(0), path: "stderr".into() })));
    }
    let a = Arc::new(fd);
    for (i, s) in t.iter_mut().enumerate().skip(3) {
        if s.is_none() {
            *s = Some(a);
            return i as i32;
        }
    }
    t.push(Some(a));
    (t.len() - 1) as i32
}

fn get_fd(fd: i32) -> Option<Arc<Fd>> {
    if (0..=2).contains(&fd) {
        return Some(Arc::new(Fd { kind: Kind::Console(fd), pos: Mutex::new(0), path: "console".into() }));
    }
    FDS.lock().get(fd as usize).and_then(|s| s.clone())
}

fn list_dir(cands: &[PathBuf]) -> Vec<(String, bool)> {
    let mut seen = std::collections::BTreeMap::new();
    for c in cands {
        if let Ok(rd) = std::fs::read_dir(c) {
            for e in rd.flatten() {
                let name = e.file_name().to_string_lossy().into_owned();
                let is_dir = e.file_type().map(|t| t.is_dir()).unwrap_or(false);
                seen.entry(name).or_insert(is_dir);
            }
        }
    }
    let mut v = vec![(".".to_string(), true), ("..".to_string(), true)];
    v.extend(seen);
    v
}

fn io_errno(e: &std::io::Error) -> i32 {
    use std::io::ErrorKind::*;
    match e.kind() {
        NotFound => ENOENT,
        PermissionDenied => EACCES,
        AlreadyExists => EEXIST,
        InvalidInput => EINVAL,
        DirectoryNotEmpty => ENOTEMPTY,
        NotADirectory => ENOTDIR,
        IsADirectory => EISDIR,
        _ => EIO,
    }
}

/// Opens a guest path. Returns fd or SCE error.
/// The developer command line the title reads at boot after its own
/// (/app0/misc/gamecommandlineargs.txt): `+convar=value` settings and
/// commands. The runtime adds its title settings to it (`title_args`).
const DEVLOG_ARGS: &str = "/devlog/app/commandlineargs.txt";

/// Title settings the runtime passes on the developer command line unless
/// `DS_GAME_ARGS` replaces them (`none`: nothing). Physics takes fixed
/// 1/59.94 s steps, at most `physicsMaxUpdateStepsPerFrame` per frame
/// (title default 1: below 60 fps physics ran at frame rate / 60, ~0.35x
/// real time at 21 fps); 6 covers the title's 0.1 s frame-step cap
/// (`gameMaxDeltaTime`) with the same step (FINDINGS.md 2026-10-15). The
/// texture streaming pool fills the title's memory up to
/// `texturePoolLowMemoryLimitMiBs` (title default 300) of free pages; our
/// stalls hold back what the title frees, and a first session ran out of
/// memory (its own stop, free pages 1 MiB): 600 keeps ~300 MiB more free
/// (FINDINGS.md 2026-10-25).
const TITLE_ARGS: &str = "+physicsMaxUpdateStepsPerFrame=6 +texturePoolLowMemoryLimitMiBs=600";

fn title_args() -> Option<String> {
    // (`;` separates settings too: run drivers pass variables without spaces.)
    let v = std::env::var("DS_GAME_ARGS").unwrap_or_else(|_| TITLE_ARGS.to_string()).replace(';', " ");
    let mut v = v.trim().to_string();
    if v == "none" {
        v.clear();
    }
    // The PC settings the title takes at boot (the SETTINGS screen's PC tab): the logos.
    if crate::pcsettings::store::value("Setting_PC_SkipIntro") == Some(1) {
        v.push_str(" +cp11_debug_skipSplashScreens=1");
    }
    let v = v.trim();
    (!v.is_empty()).then(|| v.to_string())
}

/// Opens the developer command line with the runtime's title settings
/// added: the file's own content (if any) and then `extra`, served from a
/// merged copy in the run directory (`title-commandline.txt`).
fn open_merged_args(r: &Resolved, path: &str, extra: &str) -> Option<i32> {
    let own = r.candidates.iter().find(|p| p.is_file()).and_then(|p| std::fs::read_to_string(p).ok()).unwrap_or_default();
    let own = own.trim_end();
    let merged = if own.is_empty() { format!("{extra}\n") } else { format!("{own} {extra}\n") };
    let host = crate::run_dir()?.join("title-commandline.txt");
    std::fs::write(&host, merged.as_bytes()).ok()?;
    let file = File::open(&host).ok()?;
    crate::info!("title command line {path}: runtime settings '{extra}' added after {} byte(s) of its own (DS_GAME_ARGS; copy in {})", own.len(), host.display());
    Some(alloc_fd(Fd { kind: Kind::File { file, writable: false }, pos: Mutex::new(0), path: path.to_string() }))
}

pub fn vfs_open(path: &str, flags: i32, _mode: u32) -> i32 {
    let Some(r) = resolve(path) else {
        crate::warn!("open '{path}': no mount");
        return sce(ENOENT);
    };
    let wants_write = flags & O_ACCMODE != 0 || flags & (O_CREAT | O_TRUNC | O_APPEND) != 0;
    if path == DEVLOG_ARGS && !wants_write {
        if let Some(fd) = title_args().and_then(|extra| open_merged_args(&r, path, &extra)) {
            return fd;
        }
    }
    if !wants_write && !r.writable {
        if let Some(d) = crate::pcsettings::generated(path) {
            crate::info!("open '{path}': the port's extended copy ({} bytes)", d.len());
            return alloc_fd(Fd { kind: Kind::Mem(d), pos: Mutex::new(0), path: path.to_string() });
        }
    }
    let existing = r.candidates.iter().find(|p| p.exists()).cloned();
    if wants_write && !r.writable {
        crate::warn!("open '{path}' for writing on read-only mount");
        return sce(EROFS);
    }
    if let Some(p) = &existing {
        if p.is_dir() {
            if flags & O_ACCMODE != 0 {
                return sce(EISDIR);
            }
            let entries = list_dir(&r.candidates);
            crate::debug!("opendir '{path}' ({} entries)", entries.len());
            return alloc_fd(Fd { kind: Kind::Dir { entries }, pos: Mutex::new(0), path: path.to_string() });
        }
        if flags & O_CREAT != 0 && flags & O_EXCL != 0 {
            return sce(EEXIST);
        }
    } else if flags & O_DIRECTORY != 0 || flags & O_CREAT == 0 {
        if crate::hle::iostat::note("kernel", path, crate::hle::iostat::What::Fail, 0) {
            crate::info!("open '{path}': not found");
        } else {
            crate::debug!("open '{path}': not found");
        }
        return sce(ENOENT);
    }
    let host = existing.unwrap_or_else(|| r.candidates[0].clone());
    let mut oo = OpenOptions::new();
    oo.read(true);
    if wants_write {
        oo.write(true);
    }
    if flags & O_CREAT != 0 {
        oo.create(true);
    }
    if flags & O_TRUNC != 0 {
        oo.truncate(true);
    }
    match oo.open(&host) {
        Ok(f) => {
            let pos = if flags & O_APPEND != 0 { f.metadata().map(|m| m.len()).unwrap_or(0) } else { 0 };
            let fd = alloc_fd(Fd { kind: Kind::File { file: f, writable: wants_write }, pos: Mutex::new(pos), path: path.to_string() });
            if crate::hle::iostat::note("kernel", path, crate::hle::iostat::What::Open, 0) {
                crate::info!("open '{path}' flags={flags:#x} -> fd {fd}");
            } else {
                crate::debug!("open '{path}' flags={flags:#x} -> fd {fd}");
            }
            fd
        }
        Err(e) => {
            crate::warn!("open '{path}' -> {}: {e}", host.display());
            sce(io_errno(&e))
        }
    }
}

pub fn vfs_close(fd: i32) -> i32 {
    let mut t = FDS.lock();
    match t.get_mut(fd as usize) {
        Some(s @ Some(_)) if fd > 2 => {
            *s = None;
            0
        }
        _ => sce(EBADF),
    }
}

fn console_write(fd: i32, data: &[u8]) {
    let s = String::from_utf8_lossy(data);
    for line in s.trim_end_matches('\n').split('\n') {
        crate::info!("[guest fd{fd}] {line}");
        // Developer automation (tools/tour.py) asks for a screenshot by
        // printing `dsport-shot: <name>` (taken at the next flip).
        if let Some(rest) = line.split("dsport-shot:").nth(1) {
            let name: String = rest.trim().chars().filter(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-').collect();
            if !name.is_empty() {
                *crate::hle::libs::videoout::SHOT_REQUEST.lock() = Some(name);
            }
        }
        // `dsport-frames: reset` starts a frame-time window, `dsport-frames:
        // <name>` logs it (tools/tour.py --frames: each location's frame
        // rate at the developer camera).
        if let Some(rest) = line.split("dsport-frames:").nth(1) {
            let rest = rest.trim();
            let report = crate::gpu::pacing::report_and_reset();
            if rest != "reset" {
                crate::info!("frames at {rest}: {report}");
            }
        }
        // `dsport-textreset` forgets the title's strings logged so far
        // (`text:` lines come once per string: tools/devscript.sh scripts
        // checking several bosses' banners in one run).
        if line.contains("dsport-textreset") {
            crate::exception::text_reset();
        }
        // `dsport-control: <line>` runs a control command (`press cross
        // 200ms`, `audio capture 30`, ...) at the script's point.
        if let Some(rest) = line.split("dsport-control:").nth(1) {
            let rest = rest.trim();
            if !rest.is_empty() {
                crate::INJECTED.lock().push(rest.to_string());
            }
        }
    }
}

pub fn pread_fd(fd: i32, buf: &mut [u8], off: u64) -> Result<usize, i32> {
    let f = get_fd(fd).ok_or(EBADF)?;
    match &f.kind {
        Kind::File { file, .. } => {
            let t0 = std::time::Instant::now();
            // The kernel writes the buffer: write-tracked pages cannot fault.
            let _w = crate::gpu::track::host_write(buf.as_ptr() as u64, buf.len() as u64);
            crate::gpu::exec::trace_writes(|| format!("file read (fd {fd}) [{off:#x}]"), buf.as_ptr() as u64, buf.len() as u64);
            let mut done = 0;
            let base = buf.as_ptr() as u64;
            while done < buf.len() {
                let what = || format!("read of '{}' [{:#x}] into {:#x}", f.path, off + done as u64, base + done as u64);
                match crate::mem::kernel_io(&what, base + done as u64, (buf.len() - done) as u64, true, || file.seek_read(&mut buf[done..], off + done as u64)) {
                    Ok(0) => break,
                    Ok(n) => done += n,
                    Err(e) => return Err(io_errno(&e)),
                }
            }
            if crate::hle::iostat::note("kernel", &f.path, crate::hle::iostat::What::Read, done as u64) {
                crate::info!("read '{}' [{off:#x}+{:#x}] -> {done:#x} bytes", f.path, buf.len());
            }
            crate::gpu::hitch::note_since(crate::gpu::hitch::Kind::File, t0, || format!("read '{}' {} KiB", f.path, done >> 10));
            Ok(done)
        }
        Kind::Mem(d) => {
            let start = (off as usize).min(d.len());
            let n = buf.len().min(d.len() - start);
            let what = || format!("read of '{}' [{off:#x}] (memory)", f.path);
            let _w = crate::gpu::track::host_write(buf.as_ptr() as u64, n as u64);
            crate::mem::kernel_io(&what, buf.as_ptr() as u64, n as u64, true, || {
                buf[..n].copy_from_slice(&d[start..start + n]);
                Ok(n)
            })
            .map_err(|e| io_errno(&e))
        }
        Kind::Dir { .. } => Err(EISDIR),
        Kind::Console(_) => Ok(0),
    }
}

pub fn vfs_read(fd: i32, buf: &mut [u8]) -> Result<usize, i32> {
    let f = get_fd(fd).ok_or(EBADF)?;
    match &f.kind {
        Kind::File { .. } | Kind::Mem(_) => {
            let mut pos = f.pos.lock();
            let n = pread_fd(fd, buf, *pos)?;
            *pos += n as u64;
            Ok(n)
        }
        Kind::Dir { .. } => Err(EISDIR),
        Kind::Console(_) => Ok(0),
    }
}

pub fn pwrite_fd(fd: i32, data: &[u8], off: Option<u64>) -> Result<usize, i32> {
    let f = get_fd(fd).ok_or(EBADF)?;
    match &f.kind {
        Kind::File { file, writable } => {
            if !writable {
                return Err(EBADF);
            }
            let t0 = std::time::Instant::now();
            let mut pos = f.pos.lock();
            let at = off.unwrap_or(*pos);
            let mut done = 0;
            while done < data.len() {
                let what = || format!("write of '{}' [{:#x}] from {:#x}", f.path, at + done as u64, data.as_ptr() as u64 + done as u64);
                match crate::mem::kernel_io(&what, data.as_ptr() as u64 + done as u64, (data.len() - done) as u64, false, || file.seek_write(&data[done..], at + done as u64)) {
                    Ok(n) => done += n,
                    Err(e) => return Err(io_errno(&e)),
                }
            }
            if off.is_none() {
                *pos = at + done as u64;
            }
            crate::gpu::hitch::note_since(crate::gpu::hitch::Kind::File, t0, || format!("write '{}' {} KiB", f.path, done >> 10));
            Ok(done)
        }
        Kind::Mem(_) => Err(EBADF),
        Kind::Dir { .. } => Err(EISDIR),
        Kind::Console(c) => {
            console_write(*c, data);
            Ok(data.len())
        }
    }
}

pub fn vfs_lseek(fd: i32, off: i64, whence: i32) -> i64 {
    let Some(f) = get_fd(fd) else { return sce(EBADF) as i64 };
    let size = match &f.kind {
        Kind::File { file, .. } => file.metadata().map(|m| m.len()).unwrap_or(0),
        Kind::Mem(d) => d.len() as u64,
        Kind::Dir { entries } => entries.len() as u64,
        Kind::Console(_) => return sce(ESPIPE) as i64,
    };
    let mut pos = f.pos.lock();
    let base = match whence {
        0 => 0i64,
        1 => *pos as i64,
        2 => size as i64,
        _ => return sce(EINVAL) as i64,
    };
    let n = base + off;
    if n < 0 {
        return sce(EINVAL) as i64;
    }
    *pos = n as u64;
    n
}

/// FreeBSD `struct stat` (120 bytes).
#[repr(C)]
#[derive(Default)]
pub struct Stat {
    pub dev: u32,
    pub ino: u32,
    pub mode: u16,
    pub nlink: u16,
    pub uid: u32,
    pub gid: u32,
    pub rdev: u32,
    pub atime: [i64; 2],
    pub mtime: [i64; 2],
    pub ctime: [i64; 2],
    pub size: i64,
    pub blocks: i64,
    pub blksize: u32,
    pub flags: u32,
    pub gen_: u32,
    pub lspare: i32,
    pub birthtime: [i64; 2],
}

const S_IFDIR: u16 = 0o040000;
const S_IFREG: u16 = 0o100000;

pub fn fill_stat(md: &std::fs::Metadata) -> Stat {
    let mut st = Stat::default();
    let t = md
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| [d.as_secs() as i64, d.subsec_nanos() as i64])
        .unwrap_or([0, 0]);
    st.atime = t;
    st.mtime = t;
    st.ctime = t;
    st.birthtime = t;
    if md.is_dir() {
        st.mode = S_IFDIR | 0o777;
        st.size = 0x10000;
        st.nlink = 2;
    } else {
        st.mode = S_IFREG | 0o777;
        st.size = md.len() as i64;
        st.nlink = 1;
    }
    st.blksize = 0x10000;
    st.blocks = (st.size + 511) / 512;
    st
}

fn stat_from_meta(m: &CachedMeta) -> Stat {
    let mut st = Stat { atime: m.mtime, mtime: m.mtime, ctime: m.mtime, birthtime: m.mtime, blksize: 0x10000, ..Default::default() };
    if m.is_dir {
        st.mode = S_IFDIR | 0o777;
        st.size = 0x10000;
        st.nlink = 2;
    } else {
        st.mode = S_IFREG | 0o777;
        st.size = m.size as i64;
        st.nlink = 1;
    }
    st.blocks = (st.size + 511) / 512;
    st
}

pub fn stat_path(path: &str, out: *mut Stat) -> i32 {
    match guest_meta(path) {
        Some((_, m)) => {
            unsafe { std::ptr::write_unaligned(out, stat_from_meta(&m)) };
            0
        }
        None => {
            crate::debug!("stat '{path}': not found");
            sce(ENOENT)
        }
    }
}

pub fn fstat_fd(fd: i32, out: *mut Stat) -> i32 {
    let Some(f) = get_fd(fd) else { return sce(EBADF) };
    let st = match &f.kind {
        Kind::File { file, .. } => match file.metadata() {
            Ok(md) => fill_stat(&md),
            Err(e) => return sce(io_errno(&e)),
        },
        Kind::Mem(d) => Stat { mode: S_IFREG | 0o777, nlink: 1, blksize: 0x10000, size: d.len() as i64, blocks: (d.len() as i64 + 511) / 512, ..Default::default() },
        Kind::Dir { .. } => Stat { mode: S_IFDIR | 0o777, nlink: 2, blksize: 0x10000, size: 0x10000, ..Default::default() },
        Kind::Console(_) => Stat { mode: 0o020000 | 0o666, ..Default::default() },
    };
    unsafe { std::ptr::write_unaligned(out, st) };
    0
}

/// FreeBSD getdirentries: packs `struct dirent` records.
pub fn getdirentries(fd: i32, buf: *mut u8, nbytes: usize, basep: *mut i64) -> i64 {
    let Some(f) = get_fd(fd) else { return sce(EBADF) as i64 };
    let Kind::Dir { entries } = &f.kind else { return sce(ENOTDIR) as i64 };
    let mut pos = f.pos.lock();
    if !basep.is_null() {
        unsafe { *basep = *pos as i64 };
    }
    let mut off = 0usize;
    while (*pos as usize) < entries.len() {
        let (name, is_dir) = &entries[*pos as usize];
        let nb = name.as_bytes();
        let namlen = nb.len().min(255);
        let reclen = 8 + ((namlen + 1 + 3) & !3);
        if off + reclen > nbytes {
            break;
        }
        unsafe {
            let p = buf.add(off);
            std::ptr::write_bytes(p, 0, reclen);
            std::ptr::write_unaligned(p as *mut u32, (*pos as u32) + 1);
            std::ptr::write_unaligned(p.add(4) as *mut u16, reclen as u16);
            *p.add(6) = if *is_dir { 4 } else { 8 };
            *p.add(7) = namlen as u8;
            std::ptr::copy_nonoverlapping(nb.as_ptr(), p.add(8), namlen);
        }
        off += reclen;
        *pos += 1;
    }
    off as i64
}

fn writable_target(path: &str) -> Result<PathBuf, i32> {
    let r = resolve(path).ok_or(ENOENT)?;
    if !r.writable {
        return Err(EROFS);
    }
    Ok(r.candidates[0].clone())
}

pub fn vfs_mkdir(path: &str) -> i32 {
    let p = match writable_target(path) {
        Ok(p) => p,
        Err(e) => return sce(e),
    };
    if p.exists() {
        return sce(EEXIST);
    }
    match std::fs::create_dir(&p) {
        Ok(()) => 0,
        Err(e) => sce(io_errno(&e)),
    }
}

pub fn vfs_rmdir(path: &str) -> i32 {
    match writable_target(path) {
        Ok(p) => match std::fs::remove_dir(&p) {
            Ok(()) => 0,
            Err(e) => sce(io_errno(&e)),
        },
        Err(e) => sce(e),
    }
}

pub fn vfs_unlink(path: &str) -> i32 {
    match writable_target(path) {
        Ok(p) => match std::fs::remove_file(&p) {
            Ok(()) => 0,
            Err(e) => sce(io_errno(&e)),
        },
        Err(e) => sce(e),
    }
}

pub fn vfs_rename(from: &str, to: &str) -> i32 {
    match (writable_target(from), writable_target(to)) {
        (Ok(a), Ok(b)) => match std::fs::rename(&a, &b) {
            Ok(()) => 0,
            Err(e) => sce(io_errno(&e)),
        },
        (Err(e), _) | (_, Err(e)) => sce(e),
    }
}

pub fn ftruncate(fd: i32, len: u64) -> i32 {
    let Some(f) = get_fd(fd) else { return sce(EBADF) };
    match &f.kind {
        Kind::File { file, writable: true } => match file.set_len(len) {
            Ok(()) => 0,
            Err(e) => sce(io_errno(&e)),
        },
        _ => sce(EINVAL),
    }
}

pub fn truncate(path: &str, len: u64) -> i32 {
    match writable_target(path) {
        Ok(p) => match OpenOptions::new().write(true).open(&p).and_then(|f| f.set_len(len)) {
            Ok(()) => 0,
            Err(e) => sce(io_errno(&e)),
        },
        Err(e) => sce(e),
    }
}

fn ret_len(r: Result<usize, i32>) -> i64 {
    match r {
        Ok(n) => n as i64,
        Err(e) => sce(e) as i64,
    }
}

fn posix_len(r: Result<usize, i32>) -> i64 {
    match r {
        Ok(n) => n as i64,
        Err(e) => {
            super::set_errno(e);
            -1
        }
    }
}

sysv! {
    fn sceKernelOpen(path: *const u8, flags: i32, mode: u32) -> i32 { vfs_open(cstr(path), flags, mode) }
    fn sceKernelClose(fd: i32) -> i32 { vfs_close(fd) }
    fn sceKernelRead(fd: i32, buf: *mut u8, n: u64) -> i64 {
        ret_len(vfs_read(fd, std::slice::from_raw_parts_mut(buf, n as usize)))
    }
    fn sceKernelWrite(fd: i32, buf: *const u8, n: u64) -> i64 {
        ret_len(pwrite_fd(fd, std::slice::from_raw_parts(buf, n as usize), None))
    }
    fn sceKernelPread(fd: i32, buf: *mut u8, n: u64, off: i64) -> i64 {
        ret_len(pread_fd(fd, std::slice::from_raw_parts_mut(buf, n as usize), off as u64))
    }
    fn sceKernelPwrite(fd: i32, buf: *const u8, n: u64, off: i64) -> i64 {
        ret_len(pwrite_fd(fd, std::slice::from_raw_parts(buf, n as usize), Some(off as u64)))
    }
    fn sceKernelLseek(fd: i32, off: i64, whence: i32) -> i64 { vfs_lseek(fd, off, whence) }
    fn sceKernelStat(path: *const u8, st: *mut Stat) -> i32 { stat_path(cstr(path), st) }
    fn sceKernelFstat(fd: i32, st: *mut Stat) -> i32 { fstat_fd(fd, st) }
    fn sceKernelMkdir(path: *const u8, mode: u32) -> i32 { vfs_mkdir(cstr(path)) }
    fn sceKernelRmdir(path: *const u8) -> i32 { vfs_rmdir(cstr(path)) }
    fn sceKernelUnlink(path: *const u8) -> i32 { vfs_unlink(cstr(path)) }
    fn sceKernelRename(from: *const u8, to: *const u8) -> i32 { vfs_rename(cstr(from), cstr(to)) }
    fn sceKernelGetdirentries(fd: i32, buf: *mut u8, n: i32, basep: *mut i64) -> i32 { getdirentries(fd, buf, n as usize, basep) as i32 }
    fn sceKernelGetdents(fd: i32, buf: *mut u8, n: i32) -> i32 { getdirentries(fd, buf, n as usize, std::ptr::null_mut()) as i32 }
    fn sceKernelFtruncate(fd: i32, len: i64) -> i32 { ftruncate(fd, len as u64) }
    fn sceKernelTruncate(path: *const u8, len: i64) -> i32 { truncate(cstr(path), len as u64) }
    fn sceKernelFsync(fd: i32) -> i32 { if get_fd(fd).is_some() { 0 } else { sce(EBADF) } }
    fn sceKernelCheckReachability(path: *const u8) -> i32 {
        if host_path_existing(cstr(path)).is_some() { 0 } else { sce(ENOENT) }
    }

    // POSIX
    fn open(path: *const u8, flags: i32, mode: u32) -> i32 { posix_ret(vfs_open(cstr(path), flags, mode)) }
    fn close(fd: i32) -> i32 { posix_ret(vfs_close(fd)) }
    fn read(fd: i32, buf: *mut u8, n: u64) -> i64 { posix_len(vfs_read(fd, std::slice::from_raw_parts_mut(buf, n as usize))) }
    fn _open(path: *const u8, flags: i32, mode: u32) -> i32 { posix_ret(vfs_open(cstr(path), flags, mode)) }
    fn _close(fd: i32) -> i32 { posix_ret(vfs_close(fd)) }
    fn _read(fd: i32, buf: *mut u8, n: u64) -> i64 { posix_len(vfs_read(fd, std::slice::from_raw_parts_mut(buf, n as usize))) }
    fn write(fd: i32, buf: *const u8, n: u64) -> i64 { posix_len(pwrite_fd(fd, std::slice::from_raw_parts(buf, n as usize), None)) }
    fn _write(fd: i32, buf: *const u8, n: u64) -> i64 { posix_len(pwrite_fd(fd, std::slice::from_raw_parts(buf, n as usize), None)) }
    fn pread(fd: i32, buf: *mut u8, n: u64, off: i64) -> i64 {
        posix_len(pread_fd(fd, std::slice::from_raw_parts_mut(buf, n as usize), off as u64))
    }
    fn pwrite(fd: i32, buf: *const u8, n: u64, off: i64) -> i64 {
        posix_len(pwrite_fd(fd, std::slice::from_raw_parts(buf, n as usize), Some(off as u64)))
    }
    fn lseek(fd: i32, off: i64, whence: i32) -> i64 {
        let r = vfs_lseek(fd, off, whence);
        if r < 0 { super::set_errno((r as i32) & 0xFFFF); -1 } else { r }
    }
    fn stat(path: *const u8, st: *mut Stat) -> i32 { posix_ret(stat_path(cstr(path), st)) }
    fn fstat(fd: i32, st: *mut Stat) -> i32 { posix_ret(fstat_fd(fd, st)) }
    fn mkdir(path: *const u8, mode: u32) -> i32 { posix_ret(vfs_mkdir(cstr(path))) }
    fn rmdir(path: *const u8) -> i32 { posix_ret(vfs_rmdir(cstr(path))) }
    fn unlink(path: *const u8) -> i32 { posix_ret(vfs_unlink(cstr(path))) }
    fn rename(from: *const u8, to: *const u8) -> i32 { posix_ret(vfs_rename(cstr(from), cstr(to))) }
    fn getdents(fd: i32, buf: *mut u8, n: i32) -> i32 {
        let r = getdirentries(fd, buf, n as usize, std::ptr::null_mut());
        if r < 0 { super::set_errno((r as i32) & 0xFFFF); -1 } else { r as i32 }
    }
    fn fsync(fd: i32) -> i32 { 0 }
}

pub fn register(r: &mut Registry) {
    reg!(r, "libkernel", [
        sceKernelOpen, sceKernelClose, sceKernelRead, sceKernelWrite, sceKernelPread, sceKernelPwrite, sceKernelLseek,
        sceKernelStat, sceKernelFstat, sceKernelMkdir, sceKernelRmdir, sceKernelUnlink, sceKernelRename,
        sceKernelGetdirentries, sceKernelGetdents, sceKernelFtruncate, sceKernelTruncate, sceKernelFsync,
        sceKernelCheckReachability, _open, _close, _read, _write,
    ]);
    reg!(r, "libScePosix", [open, close, read, write, pread, pwrite, lseek, stat, fstat, mkdir, rmdir, unlink, rename, getdents, fsync]);
}

pub fn ensure_dir(p: &Path) {
    let _ = std::fs::create_dir_all(p);
}
