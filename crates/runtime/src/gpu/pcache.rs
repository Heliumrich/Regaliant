//! Persistent Vulkan pipeline caches, one per pipeline
//! (`local-data/pipelines/<kind><key>.bin`).
//!
//! Creating the title's pipelines from scratch takes minutes per run (~700 in
//! 122 s at the 2026-10-11 courtyard), so compiled pipelines are kept across
//! runs. One cache file for all of them grew without bound (Vulkan caches
//! never drop entries: 1.5 GB by 2026-10-11, entries of earlier translations
//! and variants included, copied out of the driver and rewritten 26 times in
//! one run), and over its 1 GiB load limit it was dropped whole, so the next
//! run recompiled everything. Now each pipeline is created with a small cache
//! of its own, loaded from the file named by the pipeline's key (translation
//! and state: `exec::compute_pipe`, the graphics state hash); the file is
//! written when creation added data (a miss, or another translation under
//! the same key) and touched when it was used as is. At startup the files used
//! longest ago go beyond `BUDGET` bytes. Data of another driver or GPU is
//! rejected by the driver's header check and replaced.
//!
//! The data of every one-pipeline cache on the RTX 4090 driver (610.88)
//! carries the same ~2 MB block (2,097,212 bytes, at varying offsets) besides
//! the pipeline's own entries (tens of KB), so files store the pipeline's
//! bytes and a reference to the block, which is kept once (`<hash>.blk`;
//! found by comparing two pipelines' data, `discover`). Files written whole
//! are converted when read.
//!
//! Pipeline binaries (VK_KHR_pipeline_binary, `vk::pipeline_binaries`):
//! the driver's binaries of a pipeline kept in `binaries/<kind><key>.pb`
//! (`create_with`), which later sessions make the pipeline from without a
//! cache object or the driver's own cache. On this driver the cache data
//! alone did not spare a compile once the driver's cache was reset
//! (FINDINGS.md 2026-11-08 evening).

use super::vk::ctx;
use ash::vk;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use parking_lot::Mutex;

static DIR: OnceLock<PathBuf> = OnceLock::new();
/// The pipeline binaries' directory (`create_with`).
static BIN_DIR: OnceLock<PathBuf> = OnceLock::new();
/// The binaries' own budget (pruned as the cache files are).
const BIN_BUDGET: u64 = 4 << 30;
/// Pipeline binary files: magic, the global key, the fingerprint of the
/// pipeline's SPIR-V (`fingerprint`: the store's keys name translations, not
/// their code, and binaries of other code must never be used), the binaries
/// (key, data; data of SHARED_MIN bytes or more kept once, in `<key>.shared`,
/// and marked u64::MAX here), an FNV-1a hash of all before it.
const PB_MAGIC: &[u8; 8] = b"DSPBIN03";
const SHARED_MIN: usize = 1 << 20;
/// [pipelines made from binaries, their ns, binaries refused or failed,
/// pipelines whose binaries were captured, their bytes, capture ns, captures
/// failed, ns reading and decoding the files, making the binary objects,
/// making the pipelines, destroying the binary objects].
static PB: [AtomicU64; 11] = [const { AtomicU64::new(0) }; 11];
/// Binary files of other code under a pipeline's key (made again).
static PB_STALE: AtomicU64 = AtomicU64::new(0);

/// File header of data stored without its shared block: the magic, the
/// block's offset and length (u32 each) and its hash (u64), then the data
/// before and after it.
const MAGIC: &[u8; 8] = b"DSPCDDP1";
const HEADER: usize = 8 + 4 + 4 + 8;
/// Shortest block stored apart.
const MIN_BLOCK: usize = 1 << 20;
/// Known shared blocks by hash (loaded from `<hash>.blk` when first needed).
static BLOCKS: Mutex<Vec<(u64, Arc<Vec<u8>>)>> = Mutex::new(Vec::new());
/// The last data written whole: a shared block is found by comparing two.
static LAST_WHOLE: Mutex<Option<Vec<u8>>> = Mutex::new(None);

fn hash64(b: &[u8]) -> u64 {
    // FNV-1a.
    b.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &x| (h ^ x as u64).wrapping_mul(0x100_0000_01b3))
}

/// First position of `needle` in `hay`.
fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    find_from(hay, needle, 0)
}

/// First position of `needle` in `hay` from `start`.
fn find_from(hay: &[u8], needle: &[u8], start: usize) -> Option<usize> {
    let n = needle.len();
    if n == 0 || n > hay.len() {
        return None;
    }
    let sig = &needle[..n.min(64)];
    let last = hay.len() - n;
    let mut i = start;
    while i <= last {
        let p = hay[i..=last].iter().position(|&b| b == sig[0])?;
        let j = i + p;
        if &hay[j..j + sig.len()] == sig && &hay[j..j + n] == needle {
            return Some(j);
        }
        i = j + 1;
    }
    None
}

/// A run of `b` of at least MIN_BLOCK bytes that `a` also contains, grown
/// from 64-byte windows near the middle of `b` (each occurrence in `a` is
/// tried: the data may repeat short patterns).
fn discover(a: &[u8], b: &[u8]) -> Option<Vec<u8>> {
    if b.len() < MIN_BLOCK || a.len() < MIN_BLOCK {
        return None;
    }
    for k in 0..8 {
        let mid = (b.len() / 2 + k * 4099).min(b.len() - 64);
        let w = &b[mid..mid + 64];
        let mut from = 0;
        for _ in 0..256 {
            let Some(t) = find_from(a, w, from) else { break };
            let (mut s, mut u) = (mid, t);
            while s > 0 && u > 0 && b[s - 1] == a[u - 1] {
                s -= 1;
                u -= 1;
            }
            let (mut e, mut v) = (mid + 64, t + 64);
            while e < b.len() && v < a.len() && b[e] == a[v] {
                e += 1;
                v += 1;
            }
            if e - s >= MIN_BLOCK {
                return Some(b[s..e].to_vec());
            }
            from = t + 1;
        }
    }
    None
}

fn block_path(dir: &Path, h: u64) -> PathBuf {
    dir.join(format!("{h:016x}.blk"))
}

/// The shared block with hash `h` (from its file when first needed).
fn block(h: u64) -> Option<Arc<Vec<u8>>> {
    if let Some((_, b)) = BLOCKS.lock().iter().find(|(k, _)| *k == h) {
        return Some(b.clone());
    }
    let data = std::fs::read(block_path(DIR.get()?, h)).ok()?;
    if hash64(&data) != h {
        return None;
    }
    let b = Arc::new(data);
    BLOCKS.lock().push((h, b.clone()));
    Some(b)
}

/// The blocks stored so far (their files: `<hash>.blk`).
fn load_blocks(dir: &Path) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        if let Some(h) = name.strip_suffix(".blk").and_then(|x| u64::from_str_radix(x, 16).ok()) {
            let _ = block(h);
        }
    }
}

/// Pipeline cache data as stored: without a known shared block when it
/// contains one (a new block is found by comparing with the last data
/// written whole, and stored).
fn encode(data: &[u8]) -> Vec<u8> {
    let known: Vec<(u64, Arc<Vec<u8>>)> = BLOCKS.lock().clone();
    for (h, b) in &known {
        if let Some(o) = find(data, b) {
            let mut out = Vec::with_capacity(HEADER + data.len() - b.len());
            out.extend_from_slice(MAGIC);
            out.extend_from_slice(&(o as u32).to_le_bytes());
            out.extend_from_slice(&(b.len() as u32).to_le_bytes());
            out.extend_from_slice(&h.to_le_bytes());
            out.extend_from_slice(&data[..o]);
            out.extend_from_slice(&data[o + b.len()..]);
            return out;
        }
    }
    let prev = LAST_WHOLE.lock().replace(data.to_vec());
    if let (Some(prev), Some(dir)) = (prev, DIR.get()) {
        if let Some(b) = discover(&prev, data) {
            let h = hash64(&b);
            if std::fs::write(block_path(dir, h), &b).is_ok() {
                crate::info!("pipeline cache: a {} KiB block shared by the pipelines' data is stored once ({h:016x}.blk)", b.len() >> 10);
                BLOCKS.lock().push((h, Arc::new(b)));
                *LAST_WHOLE.lock() = None;
                return encode(data);
            }
        }
    }
    data.to_vec()
}

/// The pipeline cache data of a file (None: a block it names is missing).
#[cfg(test)]
fn decode(file: &[u8]) -> Option<Vec<u8>> {
    let mut out = Decoded::default();
    decode_into(file, &mut out).then_some(out.data)
}

/// A file's decoded data, and where its shared block lies in it (offset and
/// the block's hash; None: no block, or not known to be in place).
#[derive(Default)]
struct Decoded {
    data: Vec<u8>,
    block_at: Option<(usize, u64)>,
}

/// `decode` into `out` (its capacity kept): false when a block the file
/// names is missing. When `out` holds that block at the same offset already
/// (the last file of this thread put it there), only the pipeline's own
/// bytes around it are copied (53% of the files hold the block at offset
/// 36, after the cache's header).
fn decode_into(file: &[u8], out: &mut Decoded) -> bool {
    if file.len() < HEADER || &file[..8] != MAGIC {
        out.data.clear();
        out.data.extend_from_slice(file);
        out.block_at = None;
        return true;
    }
    let word = |r: std::ops::Range<usize>| file[r].iter().rev().fold(0u64, |a, &b| a << 8 | b as u64);
    let (o, n, h) = (word(8..12) as usize, word(12..16) as usize, word(16..24));
    let rest = &file[HEADER..];
    let Some(b) = block(h) else { return false };
    if b.len() != n || o > rest.len() {
        return false;
    }
    if out.block_at == Some((o, h)) && out.data.len() >= o + n {
        out.data[..o].copy_from_slice(&rest[..o]);
        out.data.truncate(o + n);
        out.data.extend_from_slice(&rest[o..]);
        return true;
    }
    out.data.clear();
    out.data.reserve(rest.len() + n);
    out.data.extend_from_slice(&rest[..o]);
    out.data.extend_from_slice(&b);
    out.data.extend_from_slice(&rest[o..]);
    out.block_at = Some((o, h));
    true
}

thread_local! {
    /// This thread's buffer for a file's decoded data (`with_cache`): ~2 MB
    /// a pipeline with the driver's shared block, which a fresh allocation
    /// committed page by page on every creation.
    static DATA: std::cell::RefCell<Decoded> = std::cell::RefCell::new(Decoded::default());
}

/// Pipeline files used as they were, whose last use the `pcache-touch`
/// thread records (for pruning; ~0.2 ms a file on the creating thread).
static TOUCH: Mutex<Vec<PathBuf>> = Mutex::new(Vec::new());

fn touch_later(p: &Path) {
    static STARTED: std::sync::Once = std::sync::Once::new();
    STARTED.call_once(|| {
        let _ = std::thread::Builder::new().name("pcache-touch".into()).spawn(|| loop {
            std::thread::sleep(std::time::Duration::from_secs(2));
            // (Taken first: a guard in the loop's head would be held through it.)
            let due = std::mem::take(&mut *TOUCH.lock());
            for p in due {
                if let Ok(f) = std::fs::File::options().write(true).open(&p) {
                    let _ = f.set_modified(std::time::SystemTime::now());
                }
            }
        });
    });
    TOUCH.lock().push(p.to_path_buf());
}

/// Writes a pipeline file (temporary name first: two threads may make the
/// same pipeline).
fn write_file(p: &Path, bytes: &[u8]) -> bool {
    let tmp = p.with_extension(format!("{:x}.tmp", std::thread::current().id().as_u64_compat()));
    if std::fs::write(&tmp, bytes).and_then(|_| std::fs::rename(&tmp, p)).is_ok() {
        return true;
    }
    let _ = std::fs::remove_file(&tmp);
    false
}

/// Stores the pipeline files written whole without their shared block
/// (before pruning, which goes by size).
fn convert(dir: &Path) {
    load_blocks(dir);
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    let mut n = 0;
    for e in rd.flatten() {
        let path = e.path();
        if path.extension().is_none_or(|x| x != "bin") {
            continue;
        }
        let Ok(file) = std::fs::read(&path) else { continue };
        if file.starts_with(MAGIC) {
            continue;
        }
        let enc = encode(&file);
        if enc.len() < file.len() && write_file(&path, &enc) {
            n += 1;
        }
    }
    if n > 0 {
        crate::info!("pipeline cache: {n} pipeline files stored without their shared block");
    }
}

/// Bytes of pipeline files kept (the least recently used beyond it go at
/// startup, down to three quarters of it). 4 GiB since 2026-11-04: the
/// files the 1-1 walk, the fights and the tours use passed 2 GiB with an
/// experiment's variants, and a prune to 1.5 GiB (4,068 files) left the
/// next suite compiling an arrival's particle programs anew (a 1.85 s frame).
const BUDGET: u64 = 4 << 30;

/// [pipelines created from their file unchanged, files written (misses),
/// bytes written, files pruned].
pub static STATS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// Files rewritten after a slow creation from them (`SLOW_CREATION`): the
/// driver refused their data, and its new data had the same size.
static REFRESHED: AtomicU64 = AtomicU64::new(0);

/// A creation from a file that took this long compiled (one from data the
/// driver accepts takes ~1-2 ms): its data is read back and compared even
/// when its size did not change. (On 2026-11-08, after a reboot, the
/// driver's ~2 MB shared block changed while its size did not: 6,021
/// creations took 776 ms each and only 37 files were rewritten, so every
/// boot compiled them again.)
const SLOW_CREATION: std::time::Duration = std::time::Duration::from_millis(20);

/// Nanoseconds of `with_cache`'s steps, summed: the file's read and its
/// decode, the cache object, the pipeline's creation, reading the cache's
/// size (its data when it grew), writing the file or queueing its last-use
/// mark, destroying the cache; then the calls.
static LAPS: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];

/// Shader modules made for the title's pipelines (`note_module`): count, ns.
static MODULES: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// A shader module for a pipeline of the title took `ns` (graphics and
/// compute pipelines make theirs before `with_cache`; a cached creation
/// still hands the driver the module's whole SPIR-V).
pub fn note_module(ns: u64) {
    MODULES[0].fetch_add(1, Ordering::Relaxed);
    MODULES[1].fetch_add(ns, Ordering::Relaxed);
}

/// Where the pipeline files live (set once at startup; none: no persistence).
/// Prunes the directory in the background.
pub fn set_dir(dir: PathBuf) {
    if std::fs::create_dir_all(&dir).is_err() || DIR.set(dir.clone()).is_err() {
        return;
    }
    let bin = dir.join("binaries");
    let bins = std::fs::create_dir_all(&bin).is_ok() && BIN_DIR.set(bin.clone()).is_ok();
    let _ = std::thread::Builder::new().name("pcache-prune".into()).spawn(move || {
        convert(&dir);
        prune(&dir, "bin", BUDGET);
        if bins {
            prune(&bin, "pb", BIN_BUDGET);
        }
    });
}

/// Removes the files (extension `ext`) used longest ago while the directory
/// is over `budget`.
fn prune(dir: &Path, ext: &str, budget: u64) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    let mut files: Vec<(std::time::SystemTime, u64, PathBuf)> = rd
        .flatten()
        .filter(|e| e.path().extension().is_some_and(|x| x == ext))
        .filter_map(|e| {
            let m = e.metadata().ok()?;
            Some((m.modified().ok()?, m.len(), e.path()))
        })
        .collect();
    let mut total: u64 = files.iter().map(|f| f.1).sum();
    crate::info!("pipeline cache: {} pipeline .{ext} files, {} MiB", files.len(), total >> 20);
    if total <= budget {
        return;
    }
    files.sort_by_key(|f| f.0);
    let mut removed = 0u64;
    for (_, len, path) in files {
        if total <= budget / 4 * 3 {
            break;
        }
        if std::fs::remove_file(&path).is_ok() {
            total -= len;
            removed += 1;
        }
    }
    STATS[3].fetch_add(removed, Ordering::Relaxed);
    crate::info!("pipeline cache: {removed} pipeline files used longest ago removed ({} MiB left)", total >> 20);
}

/// Creates a pipeline through the cache file of `key` (`kind`: 'c' compute,
/// 'g' graphics, 'i' the runtime's own): `create` gets the pipeline cache to
/// pass to vkCreate*Pipelines.
/// Pipeline creations running now (any thread; diagnosis of stalled submissions).
pub static CREATING: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

pub fn with_cache<T>(kind: char, key: u64, create: impl FnOnce(vk::PipelineCache) -> T) -> T {
    let d = &ctx().device;
    let t = std::time::Instant::now();
    let lap = |k: usize, t: &mut std::time::Instant| {
        let n = std::time::Instant::now();
        LAPS[k].fetch_add((n - *t).as_nanos() as u64, Ordering::Relaxed);
        *t = n;
    };
    let mut t = t;
    LAPS[7].fetch_add(1, Ordering::Relaxed);
    let path = DIR.get().map(|dir| dir.join(format!("{kind}{key:016x}.bin")));
    let file = path.as_ref().and_then(|p| std::fs::read(p).ok()).unwrap_or_default();
    lap(6, &mut t);
    let whole = !file.starts_with(MAGIC);
    // (Taken out of the thread's slot while in use: `create` may make
    // another pipeline.)
    let mut decoded = DATA.with(|b| std::mem::take(&mut *b.borrow_mut()));
    if !decode_into(&file, &mut decoded) {
        decoded.data.clear();
        decoded.block_at = None;
    }
    let data = &decoded.data;
    lap(0, &mut t);
    let cache = unsafe { d.create_pipeline_cache(&vk::PipelineCacheCreateInfo::default().initial_data(data), None) }
        .or_else(|_| unsafe { d.create_pipeline_cache(&vk::PipelineCacheCreateInfo::default(), None) })
        .unwrap_or(vk::PipelineCache::null());
    lap(1, &mut t);
    CREATING.fetch_add(1, Ordering::Relaxed);
    let tc = std::time::Instant::now();
    let r = create(cache);
    let slow = !file.is_empty() && tc.elapsed() >= SLOW_CREATION;
    CREATING.fetch_sub(1, Ordering::Relaxed);
    lap(2, &mut t);
    if cache == vk::PipelineCache::null() {
        DATA.with(|b| *b.borrow_mut() = decoded);
        return r;
    }
    if let Some(p) = &path {
        // The creation added nothing when the cache's size is unchanged
        // (the data itself, ~2 MB, is read back only when it grew).
        let mut size = 0usize;
        let sized = unsafe { (d.fp_v1_0().get_pipeline_cache_data)(d.handle(), cache, &mut size, std::ptr::null_mut()) } == vk::Result::SUCCESS;
        let got = if sized && size == data.len() && !(whole && !file.is_empty()) && !slow {
            Ok(None)
        } else {
            unsafe { d.get_pipeline_cache_data(cache) }.map(Some)
        };
        lap(3, &mut t);
        match got {
            Ok(None) => {
                STATS[0].fetch_add(1, Ordering::Relaxed);
                touch_later(p);
            }
            Ok(Some(new)) => match new {
                new if new.len() != data.len() => {
                    let enc = encode(&new);
                    if write_file(p, &enc) {
                        STATS[1].fetch_add(1, Ordering::Relaxed);
                        STATS[2].fetch_add(enc.len() as u64, Ordering::Relaxed);
                    }
                }
                new if whole && !file.is_empty() && !(slow && new != *data) => {
                    // A file written whole: store it without its shared block.
                    STATS[0].fetch_add(1, Ordering::Relaxed);
                    let enc = encode(&new);
                    if enc.len() < file.len() {
                        write_file(p, &enc);
                    }
                }
                new if slow && new != *data => {
                    // Refused data compiled again (`SLOW_CREATION`).
                    let enc = encode(&new);
                    if write_file(p, &enc) {
                        STATS[1].fetch_add(1, Ordering::Relaxed);
                        STATS[2].fetch_add(enc.len() as u64, Ordering::Relaxed);
                        if REFRESHED.fetch_add(1, Ordering::Relaxed) == 0 {
                            crate::info!("pipeline cache: the driver refused {kind}{key:016x}'s stored data (its creation compiled for {} ms): rewritten, as are later ones", tc.elapsed().as_millis());
                        }
                    }
                }
                _ => {
                    STATS[0].fetch_add(1, Ordering::Relaxed);
                    touch_later(p);
                }
            },
            Err(e) => crate::warn!("pipeline cache: reading the data of {kind}{key:016x} failed ({e})"),
        }
        lap(4, &mut t);
    }
    unsafe { d.destroy_pipeline_cache(cache, None) };
    lap(5, &mut t);
    DATA.with(|b| *b.borrow_mut() = decoded);
    r
}

/// Whether the store holds what makes the pipeline of `key` without a
/// compile in this session's mode: its binaries with pipeline binaries on,
/// else its cache file.
pub fn stored_cheap(kind: char, key: u64) -> bool {
    if ctx().pbin.is_some() {
        BIN_DIR.get().is_some_and(|dir| dir.join(format!("{kind}{key:016x}.pb")).exists())
    } else {
        DIR.get().is_some_and(|dir| dir.join(format!("{kind}{key:016x}.bin")).exists())
    }
}

/// Whether the store holds the pipeline of `key` (its creation then takes
/// milliseconds, not the seconds of a first compile).
pub fn stored(kind: char, key: u64) -> bool {
    DIR.get().is_some_and(|dir| dir.join(format!("{kind}{key:016x}.bin")).exists()) || BIN_DIR.get().is_some_and(|dir| dir.join(format!("{kind}{key:016x}.pb")).exists())
}

/// How `create_with` has a pipeline made: through `cache`, with its
/// binaries captured (VK_PIPELINE_CREATE_2_CAPTURE_DATA_BIT_KHR in a
/// VkPipelineCreateFlags2CreateInfoKHR with the create info's own flags),
/// or from `binaries` (a VkPipelineBinaryInfoKHR in its chain, no cache).
pub struct Via<'a> {
    pub cache: vk::PipelineCache,
    pub capture: bool,
    pub binaries: Option<&'a [super::vk::pbin::Binary]>,
}

/// Creates the pipeline of `key` (`kind` 'c' compute, 'g' graphics) through
/// the store: with pipeline binaries on (`vk::pipeline_binaries`), from its
/// stored binaries when it has them (and they were made from the same code,
/// `fp`), else made anew with its binaries captured into the store; without
/// them through its cache file (`with_cache`).
thread_local! {
    /// What this thread is having the driver compile (`Crumb`): named in the log when the
    /// driver faults inside it (`exception::veh_body`; the driver's compiler crashed so in
    /// mass rebuilds: FINDINGS.md 2026-11-15 evening), with its stages' SPIR-V saved into the
    /// run directory under `DS_PIPELINE_CRUMBS=1`.
    static CRUMB: std::cell::RefCell<Option<(String, Vec<(&'static str, Vec<u32>)>)>> = const { std::cell::RefCell::new(None) };
}

/// A pipeline compile's note (`CRUMB`) for as long as it lives.
pub struct Crumb;

impl Crumb {
    pub fn set(what: impl FnOnce() -> String, code: impl FnOnce() -> Vec<(&'static str, Vec<u32>)>) -> Crumb {
        static KEEP: OnceLock<bool> = OnceLock::new();
        let keep = *KEEP.get_or_init(|| std::env::var("DS_PIPELINE_CRUMBS").ok().as_deref() == Some("1"));
        let code = if keep { code() } else { Vec::new() };
        CRUMB.with(|c| {
            if let Ok(mut g) = c.try_borrow_mut() {
                *g = Some((what(), code));
            }
        });
        Crumb
    }
}

impl Drop for Crumb {
    fn drop(&mut self) {
        CRUMB.with(|c| {
            if let Ok(mut g) = c.try_borrow_mut() {
                *g = None;
            }
        });
    }
}

/// The driver faulted on this thread (`exception::veh_body`, at first chance): what it was
/// compiling, with the files its SPIR-V went to.
pub fn crumb_report() -> Option<String> {
    CRUMB.with(|c| {
        let g = c.try_borrow().ok()?;
        let (what, code) = g.as_ref()?;
        let mut files = Vec::new();
        if let Some(dir) = crate::run_dir() {
            for (stage, words) in code {
                let p = dir.join(format!("fault-{stage}-{:016x}.spv", fingerprint(&[words])));
                let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
                if std::fs::write(&p, bytes).is_ok() {
                    files.push(p.display().to_string());
                }
            }
        }
        Some(if files.is_empty() { what.clone() } else { format!("{what}; SPIR-V in {}", files.join(", ")) })
    })
}

pub fn create_with(kind: char, key: u64, fp: u64, create: impl Fn(Via) -> Result<vk::Pipeline, String>) -> Result<vk::Pipeline, String> {
    let Some(pb) = ctx().pbin.as_ref() else {
        return with_cache(kind, key, |cache| create(Via { cache, capture: false, binaries: None }));
    };
    let path = BIN_DIR.get().map(|dir| dir.join(format!("{kind}{key:016x}.pb")));
    // (Under the validation layer the driver refuses some graphics
    // pipelines' binaries that it takes without the layer (2026-11-08
    // night): such a boot keeps the files as they are.)
    let keep = ctx().validation;
    if let Some(p) = &path {
        let t = std::time::Instant::now();
        let stored = std::fs::read(p).ok().and_then(|b| decode_pb(&b, &pb.global, fp));
        PB[7].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        if let Some(stored) = stored {
            let tb = std::time::Instant::now();
            let shared_ok = stored.iter().all(|part| match part {
                super::vk::pbin::Part::Shared(k) => pb.has_shared(k) || load_shared(k).is_some_and(|d| pb.share(k, &d).is_ok()),
                super::vk::pbin::Part::Own(..) => true,
            });
            let r = match shared_ok.then(|| pb.from_parts(&stored)) {
                Some(Ok((handles, own))) => {
                    PB[8].fetch_add(tb.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    let tc = std::time::Instant::now();
                    CREATING.fetch_add(1, Ordering::Relaxed);
                    let r = create(Via { cache: vk::PipelineCache::null(), capture: false, binaries: Some(&handles) });
                    CREATING.fetch_sub(1, Ordering::Relaxed);
                    PB[9].fetch_add(tc.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    let td = std::time::Instant::now();
                    pb.destroy(&own);
                    PB[10].fetch_add(td.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    r
                }
                Some(Err(e)) => Err(format!("binaries from their data: {e}")),
                None => Err("a shared binary is missing".to_string()),
            };
            match r {
                Ok(pipe) => {
                    PB[0].fetch_add(1, Ordering::Relaxed);
                    PB[1].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    touch_later(p);
                    return Ok(pipe);
                }
                Err(e) => {
                    if PB[2].fetch_add(1, Ordering::Relaxed) < 8 {
                        crate::warn!("pipeline cache: the binaries of {kind}{key:016x} were not used ({e}); made anew");
                    }
                    if keep {
                        CREATING.fetch_add(1, Ordering::Relaxed);
                        let r = create(Via { cache: vk::PipelineCache::null(), capture: false, binaries: None });
                        CREATING.fetch_sub(1, Ordering::Relaxed);
                        return r;
                    }
                    let _ = std::fs::remove_file(p);
                }
            }
        }
    }
    // (Made with CAPTURE_DATA, without a cache object (VUID 09617), its
    // binaries then captured from it: what the driver's internal cache holds,
    // when it holds it, `pbin::Fns::probe`, gives the same.)
    CREATING.fetch_add(1, Ordering::Relaxed);
    let r = create(Via { cache: vk::PipelineCache::null(), capture: true, binaries: None });
    CREATING.fetch_sub(1, Ordering::Relaxed);
    if let (Ok(pipe), Some(p)) = (&r, &path) {
        let t = std::time::Instant::now();
        match pb.capture(*pipe, &|k| shared_path(k).is_some_and(|p| p.exists())) {
            Ok(bins) => {
                let enc = encode_pb(&pb.global, fp, &bins);
                if write_file(p, &enc) {
                    PB[3].fetch_add(1, Ordering::Relaxed);
                    PB[4].fetch_add(enc.len() as u64, Ordering::Relaxed);
                }
            }
            Err(e) => {
                if PB[6].fetch_add(1, Ordering::Relaxed) < 8 {
                    crate::warn!("pipeline cache: capturing the binaries of {kind}{key:016x} failed ({e})");
                }
            }
        }
        PB[5].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
    r
}

/// A pipeline's code fingerprint (`create_with`): its stages' SPIR-V words
/// and any other words that shape it.
pub fn fingerprint(parts: &[&[u32]]) -> u64 {
    let mut h = 0x9E37_79B9_7F4A_7C15u64;
    for p in parts {
        h = (h.rotate_left(5) ^ p.len() as u64).wrapping_mul(0x517C_C1B7_2722_0A95);
        for &w in p.iter() {
            h = (h.rotate_left(5) ^ w as u64).wrapping_mul(0x517C_C1B7_2722_0A95);
        }
    }
    h
}

fn fnv64(b: &[u8]) -> u64 {
    b.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &x| (h ^ x as u64).wrapping_mul(0x100_0000_01b3))
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// A shared binary's file (`<key>.shared`): its data and an FNV-1a hash.
fn shared_path(key: &[u8]) -> Option<PathBuf> {
    BIN_DIR.get().map(|d| d.join(format!("{}.shared", hex(key))))
}

/// Stores a shared binary's data once (`SHARED_MIN`).
fn store_shared(key: &[u8], data: &[u8]) {
    let Some(p) = shared_path(key) else { return };
    if p.exists() {
        return;
    }
    let mut w = data.to_vec();
    w.extend_from_slice(&fnv64(data).to_le_bytes());
    write_file(&p, &w);
}

fn load_shared(key: &[u8]) -> Option<Vec<u8>> {
    let mut b = std::fs::read(shared_path(key)?).ok()?;
    if b.len() < 8 {
        return None;
    }
    let tail = b.split_off(b.len() - 8);
    if fnv64(&b) != u64::from_le_bytes(tail.try_into().ok()?) {
        // (Damaged: removed, so the next capture stores it again.)
        let _ = std::fs::remove_file(shared_path(key)?);
        return None;
    }
    Some(b)
}

/// A pipeline binary file (`PB_MAGIC`); shared data stored once apart.
fn encode_pb(global: &[u8], fp: u64, bins: &[(Vec<u8>, Vec<u8>)]) -> Vec<u8> {
    let mut w = Vec::with_capacity(64 + bins.iter().map(|b| b.0.len() + b.1.len().min(SHARED_MIN) + 12).sum::<usize>());
    w.extend_from_slice(PB_MAGIC);
    w.extend_from_slice(&(global.len() as u32).to_le_bytes());
    w.extend_from_slice(global);
    w.extend_from_slice(&fp.to_le_bytes());
    w.extend_from_slice(&(bins.len() as u32).to_le_bytes());
    for (k, d) in bins {
        w.extend_from_slice(&(k.len() as u32).to_le_bytes());
        w.extend_from_slice(k);
        if d.is_empty() || d.len() >= SHARED_MIN {
            // (Empty: a shared binary stored before, `pbin::Fns::capture`.)
            if !d.is_empty() {
                store_shared(k, d);
            }
            w.extend_from_slice(&u64::MAX.to_le_bytes());
        } else {
            w.extend_from_slice(&(d.len() as u64).to_le_bytes());
            w.extend_from_slice(d);
        }
    }
    let h = fnv64(&w);
    w.extend_from_slice(&h.to_le_bytes());
    w
}

/// `encode_pb`'s inverse; None for another global key, other code (`fp`)
/// or a damaged file.
fn decode_pb(b: &[u8], global: &[u8], fp: u64) -> Option<Vec<super::vk::pbin::Part>> {
    if b.len() < 8 + 4 + 4 + 8 || &b[..8] != PB_MAGIC {
        return None;
    }
    let (body, tail) = b.split_at(b.len() - 8);
    if fnv64(body) != u64::from_le_bytes(tail.try_into().ok()?) {
        return None;
    }
    let mut at = 8usize;
    let mut take = |n: usize| -> Option<&[u8]> {
        let s = body.get(at..at.checked_add(n)?)?;
        at += n;
        Some(s)
    };
    let gl = u32::from_le_bytes(take(4)?.try_into().ok()?) as usize;
    if take(gl)? != global {
        return None;
    }
    if u64::from_le_bytes(take(8)?.try_into().ok()?) != fp {
        PB_STALE.fetch_add(1, Ordering::Relaxed);
        return None;
    }
    let n = u32::from_le_bytes(take(4)?.try_into().ok()?) as usize;
    if n == 0 || n > 64 {
        return None;
    }
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let kl = u32::from_le_bytes(take(4)?.try_into().ok()?) as usize;
        let k = take(kl)?.to_vec();
        let dl = u64::from_le_bytes(take(8)?.try_into().ok()?);
        if dl == u64::MAX {
            out.push(super::vk::pbin::Part::Shared(k));
        } else {
            let d = take(dl as usize)?.to_vec();
            out.push(super::vk::pbin::Part::Own(k, d));
        }
    }
    Some(out)
}

/// One status line.
pub fn status() -> String {
    let s: Vec<u64> = STATS.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    let l: Vec<u64> = LAPS.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    let n = l[7].max(1) as f64 * 1000.0;
    format!(
        "    pipeline cache files: {} pipelines from their file, {} written ({} MiB; {} of them after the driver refused their data), {} pruned; {} creations, us each: file read {:.0}, decode {:.0}, cache object {:.0}, creation {:.0}, its size or data {:.0}, write or last-use mark {:.0}, destroy {:.0}; shader modules before them: {} at {:.0} us\n",
        s[0],
        s[1],
        s[2] >> 20,
        REFRESHED.load(Ordering::Relaxed),
        s[3],
        l[7],
        l[6] as f64 / n,
        l[0] as f64 / n,
        l[1] as f64 / n,
        l[2] as f64 / n,
        l[3] as f64 / n,
        l[4] as f64 / n,
        l[5] as f64 / n,
        MODULES[0].load(Ordering::Relaxed),
        MODULES[1].load(Ordering::Relaxed) as f64 / MODULES[0].load(Ordering::Relaxed).max(1) as f64 / 1000.0
    ) + &{
        let b: Vec<u64> = PB.iter().map(|c| c.load(Ordering::Relaxed)).collect();
        if super::vk::try_ctx().is_none_or(|c| c.pbin.is_none()) {
            String::new()
        } else {
            format!(
                "    pipeline binaries: {} pipelines made from them ({:.0} us each: file {:.0}, binary objects {:.0}, pipeline {:.0}, destroy {:.0}), {} refused or failed, {} of other code; {} captured ({} MiB, {:.0} us each), {} captures failed\n",
                b[0],
                b[1] as f64 / b[0].max(1) as f64 / 1000.0,
                b[7] as f64 / b[0].max(1) as f64 / 1000.0,
                b[8] as f64 / b[0].max(1) as f64 / 1000.0,
                b[9] as f64 / b[0].max(1) as f64 / 1000.0,
                b[10] as f64 / b[0].max(1) as f64 / 1000.0,
                b[2],
                PB_STALE.load(Ordering::Relaxed),
                b[3],
                b[4] >> 20,
                b[5] as f64 / (b[3] + b[6]).max(1) as f64 / 1000.0,
                b[6]
            )
        }
    }
}

/// `ThreadId::as_u64` is unstable: a stable number per thread from its debug
/// form.
trait ThreadIdNumber {
    fn as_u64_compat(&self) -> u64;
}

impl ThreadIdNumber for std::thread::ThreadId {
    fn as_u64_compat(&self) -> u64 {
        format!("{self:?}").bytes().filter(u8::is_ascii_digit).fold(0u64, |a, b| a.wrapping_mul(10).wrapping_add((b - b'0') as u64))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Files decoded one after another into one buffer (the block kept in
    /// place when the offset repeats) come back as decoded fresh.
    #[test]
    fn decode_reuses_the_block() {
        let block: Vec<u8> = (0..(MIN_BLOCK + 3000)).map(|i| (i * 7 + i / 300) as u8).collect();
        let h = hash64(&block);
        BLOCKS.lock().push((h, std::sync::Arc::new(block.clone())));
        let file = |pre: &[u8], post: &[u8]| {
            let mut f = Vec::new();
            f.extend_from_slice(MAGIC);
            f.extend_from_slice(&(pre.len() as u32).to_le_bytes());
            f.extend_from_slice(&(block.len() as u32).to_le_bytes());
            f.extend_from_slice(&h.to_le_bytes());
            f.extend_from_slice(pre);
            f.extend_from_slice(post);
            f
        };
        let whole = |pre: &[u8], post: &[u8]| [pre, &block[..], post].concat();
        let mut d = Decoded::default();
        for (pre, post) in [(&[1u8; 36][..], &[2u8; 500][..]), (&[3u8; 36][..], &[4u8; 20][..]), (&[5u8; 40][..], &[6u8; 900][..]), (&[7u8; 36][..], &[8u8; 3000][..])] {
            assert!(decode_into(&file(pre, post), &mut d));
            assert_eq!(d.data, whole(pre, post));
        }
    }

    /// Two pipelines' data sharing a 1.5 MiB block at different offsets:
    /// `discover` finds it, and stored without it the data comes back whole.
    #[test]
    fn shared_block_round_trip() {
        // xorshift64 bytes, with a short repeated pattern at the window the
        // search starts from (as the driver's data may hold).
        let mut x = 0x9E37_79B9_7F4A_7C15u64;
        let mut block: Vec<u8> = (0..(3 << 19))
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                x as u8
            })
            .collect();
        let pat: Vec<u8> = block[700_000..700_200].to_vec();
        block[100..300].copy_from_slice(&pat);
        let a: Vec<u8> = [&b"header-a"[..], &block, &[7u8; 300]].concat();
        let b: Vec<u8> = [&[9u8; 5000][..], &block, &b"tail-b"[..]].concat();
        let found = discover(&a, &b).expect("block");
        assert_eq!(found, block);
        let h = hash64(&found);
        BLOCKS.lock().push((h, Arc::new(found)));
        for data in [&a, &b] {
            let enc = encode(data);
            assert!(enc.starts_with(MAGIC) && enc.len() < data.len() / 100);
            assert_eq!(decode(&enc).as_deref(), Some(&data[..]));
        }
        // Data without the block is stored whole.
        let c = b"short pipeline data".to_vec();
        assert_eq!(decode(&encode(&c)).as_deref(), Some(&c[..]));
    }
}

/// `dsport --pb-test` (diagnosis): pipeline binaries on one small compute
/// pipeline with no title running: made, its binaries probed from the
/// driver's internal cache, then made with CAPTURE_DATA and its binaries
/// captured, each step logged before its call (the driver faulted in both
/// in runs, never under the validation layer).
pub fn pb_test() -> i32 {
    if let Err(e) = super::vk::init() {
        crate::error!("pb-test: Vulkan: {e}");
        return 1;
    }
    let c = ctx();
    let Some(pb) = c.pbin.as_ref() else {
        crate::error!("pb-test: pipeline binaries are off (DS_PIPELINE_BINARIES=1)");
        return 1;
    };
    unsafe {
        let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..2)
            .map(|b| vk::DescriptorSetLayoutBinding::default().binding(b).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE))
            .collect();
        let Ok(set_layout) = c.device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None) else { return 1 };
        let layouts = [set_layout];
        let ranges = [vk::PushConstantRange { stage_flags: vk::ShaderStageFlags::COMPUTE, offset: 0, size: 4 * (4 + 2 * rdna::kernels::TS_MANY) }];
        let Ok(layout) = c.device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts).push_constant_ranges(&ranges), None) else { return 1 };
        let code = rdna::kernels::timestamp_store_many();
        let Ok(module) = c.device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None) else { return 1 };
        let stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(c"main");
        let ci = vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout);
        crate::info!("pb-test: making the pipeline");
        let p1 = c.device.create_compute_pipelines(vk::PipelineCache::null(), &[ci], None);
        crate::info!("pb-test: made: {:?}", p1.as_ref().map(|v| v[0]).map_err(|e| e.1));
        if std::env::var("DS_PB_TEST").ok().as_deref() != Some("capture") {
            crate::info!("pb-test: probing its binaries by its create info");
            let r = pb.probe(&ci as *const vk::ComputePipelineCreateInfo as *const std::ffi::c_void);
            crate::info!("pb-test: probed: {:?}", r.as_ref().map(|v| v.iter().map(|b| (b.0.len(), b.1.len())).collect::<Vec<_>>()));
        }
        let mut f2 = vk::PipelineCreateFlags2CreateInfoKHR::default().flags(vk::PipelineCreateFlags2KHR::from_raw(super::vk::pbin::CAPTURE_DATA));
        let ci2 = ci.push_next(&mut f2);
        crate::info!("pb-test: making it with CAPTURE_DATA");
        let p2 = c.device.create_compute_pipelines(vk::PipelineCache::null(), &[ci2], None);
        crate::info!("pb-test: made: {:?}", p2.as_ref().map(|v| v[0]).map_err(|e| e.1));
        if let Ok(v) = &p2 {
            crate::info!("pb-test: capturing its binaries");
            let r = pb.capture(v[0], &|_| false);
            crate::info!("pb-test: captured: {:?}", r.as_ref().map(|v| v.iter().map(|b| (b.0.len(), b.1.len())).collect::<Vec<_>>()));
        }
        c.device.destroy_shader_module(module, None);
    }
    crate::info!("pb-test: done");
    0
}
