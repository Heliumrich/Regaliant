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

use super::vk::ctx;
use ash::vk;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use parking_lot::Mutex;

static DIR: OnceLock<PathBuf> = OnceLock::new();

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
    let mut out = Vec::new();
    decode_into(file, &mut out).then_some(out)
}

/// `decode` into `out` (emptied first; its capacity kept): false when a
/// block the file names is missing.
fn decode_into(file: &[u8], out: &mut Vec<u8>) -> bool {
    out.clear();
    if file.len() < HEADER || &file[..8] != MAGIC {
        out.extend_from_slice(file);
        return true;
    }
    let word = |r: std::ops::Range<usize>| file[r].iter().rev().fold(0u64, |a, &b| a << 8 | b as u64);
    let (o, n, h) = (word(8..12) as usize, word(12..16) as usize, word(16..24));
    let rest = &file[HEADER..];
    let Some(b) = block(h) else { return false };
    if b.len() != n || o > rest.len() {
        return false;
    }
    out.reserve(rest.len() + n);
    out.extend_from_slice(&rest[..o]);
    out.extend_from_slice(&b);
    out.extend_from_slice(&rest[o..]);
    true
}

thread_local! {
    /// This thread's buffer for a file's decoded data (`with_cache`): ~2 MB
    /// a pipeline with the driver's shared block, which a fresh allocation
    /// committed page by page on every creation.
    static DATA: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
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
/// startup, down to three quarters of it).
const BUDGET: u64 = 2 << 30;

/// [pipelines created from their file unchanged, files written (misses),
/// bytes written, files pruned].
pub static STATS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// Nanoseconds of `with_cache`'s steps, summed: the file's read and its
/// decode, the cache object, the pipeline's creation, reading the cache's
/// size (its data when it grew), writing the file or queueing its last-use
/// mark, destroying the cache; then the calls.
static LAPS: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];

/// Where the pipeline files live (set once at startup; none: no persistence).
/// Prunes the directory in the background.
pub fn set_dir(dir: PathBuf) {
    if std::fs::create_dir_all(&dir).is_err() || DIR.set(dir.clone()).is_err() {
        return;
    }
    let _ = std::thread::Builder::new().name("pcache-prune".into()).spawn(move || {
        convert(&dir);
        prune(&dir);
    });
}

/// Removes the files used longest ago while the directory is over `BUDGET`.
fn prune(dir: &Path) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    let mut files: Vec<(std::time::SystemTime, u64, PathBuf)> = rd
        .flatten()
        .filter(|e| e.path().extension().is_some_and(|x| x == "bin"))
        .filter_map(|e| {
            let m = e.metadata().ok()?;
            Some((m.modified().ok()?, m.len(), e.path()))
        })
        .collect();
    let mut total: u64 = files.iter().map(|f| f.1).sum();
    crate::info!("pipeline cache: {} pipeline files, {} MiB", files.len(), total >> 20);
    if total <= BUDGET {
        return;
    }
    files.sort_by_key(|f| f.0);
    let mut removed = 0u64;
    for (_, len, path) in files {
        if total <= BUDGET / 4 * 3 {
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
    let mut data = DATA.with(|b| std::mem::take(&mut *b.borrow_mut()));
    if !decode_into(&file, &mut data) {
        data.clear();
    }
    lap(0, &mut t);
    let cache = unsafe { d.create_pipeline_cache(&vk::PipelineCacheCreateInfo::default().initial_data(&data), None) }
        .or_else(|_| unsafe { d.create_pipeline_cache(&vk::PipelineCacheCreateInfo::default(), None) })
        .unwrap_or(vk::PipelineCache::null());
    lap(1, &mut t);
    let r = create(cache);
    lap(2, &mut t);
    if cache == vk::PipelineCache::null() {
        DATA.with(|b| *b.borrow_mut() = data);
        return r;
    }
    if let Some(p) = &path {
        // The creation added nothing when the cache's size is unchanged
        // (the data itself, ~2 MB, is read back only when it grew).
        let mut size = 0usize;
        let sized = unsafe { (d.fp_v1_0().get_pipeline_cache_data)(d.handle(), cache, &mut size, std::ptr::null_mut()) } == vk::Result::SUCCESS;
        let got = if sized && size == data.len() && !(whole && !file.is_empty()) {
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
                new if whole && !file.is_empty() => {
                    // A file written whole: store it without its shared block.
                    STATS[0].fetch_add(1, Ordering::Relaxed);
                    let enc = encode(&new);
                    if enc.len() < file.len() {
                        write_file(p, &enc);
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
    DATA.with(|b| *b.borrow_mut() = data);
    r
}

/// Whether the store holds the pipeline of `key` (its creation then takes
/// milliseconds, not the seconds of a first compile).
pub fn stored(kind: char, key: u64) -> bool {
    DIR.get().is_some_and(|dir| dir.join(format!("{kind}{key:016x}.bin")).exists())
}

/// One status line.
pub fn status() -> String {
    let s: Vec<u64> = STATS.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    let l: Vec<u64> = LAPS.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    let n = l[7].max(1) as f64 * 1000.0;
    format!(
        "    pipeline cache files: {} pipelines from their file, {} written ({} MiB), {} pruned; {} creations, us each: file read {:.0}, decode {:.0}, cache object {:.0}, creation {:.0}, its size or data {:.0}, write or last-use mark {:.0}, destroy {:.0}\n",
        s[0],
        s[1],
        s[2] >> 20,
        s[3],
        l[7],
        l[6] as f64 / n,
        l[0] as f64 / n,
        l[1] as f64 / n,
        l[2] as f64 / n,
        l[3] as f64 / n,
        l[4] as f64 / n,
        l[5] as f64 / n
    )
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
