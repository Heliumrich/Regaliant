//! Persistent Vulkan pipeline cache (`local-data/pipeline-cache.bin`).
//!
//! Creating the title's ~150 pipelines takes ~150 s per run (62 s of it during
//! the first area load, on the graphics command processor). The translated
//! SPIR-V is deterministic, so a cache kept across runs lets the driver skip
//! compilation. The file is loaded once, the cache is shared by all pipeline
//! creations (Vulkan pipeline caches are internally synchronized), and it is
//! saved in the background after `SAVE_EVERY` pipeline creations at most every
//! `SAVE_INTERVAL`, and at shutdown (written to a temporary file, then
//! renamed). Incompatible data (another driver or GPU) is rejected by the
//! driver's header check.
//!
//! Vulkan caches never drop entries: pipelines of earlier translations stay in
//! the file. By 2026-10-11 it had grown to 1.5 GB, was copied out of the driver
//! and rewritten 26 times in one run, and pipeline creations waited for those
//! copies. Saved data over `MAX_LOAD` is therefore not loaded (the cache starts
//! empty and the file is replaced by the next save).

use super::vk::ctx;
use ash::vk;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::OnceLock;

static PATH: OnceLock<PathBuf> = OnceLock::new();
static CACHE: OnceLock<vk::PipelineCache> = OnceLock::new();
static NEW: AtomicU64 = AtomicU64::new(0);
static SAVING: AtomicBool = AtomicBool::new(false);
/// Size of the data last loaded or saved (unchanged data is not rewritten).
static SAVED_LEN: AtomicU64 = AtomicU64::new(0);
/// When the last save started (ms since `START`).
static LAST_SAVE_MS: AtomicU64 = AtomicU64::new(0);
static START: OnceLock<std::time::Instant> = OnceLock::new();
const SAVE_EVERY: u64 = 32;
const SAVE_INTERVAL_MS: u64 = 180_000;
const MAX_LOAD: usize = 1 << 30;

/// Where the cache lives (set once at startup; no file, no persistence).
pub fn set_path(p: PathBuf) {
    let _ = PATH.set(p);
}

/// The shared pipeline cache (created, from the saved data, on first use).
pub fn get() -> vk::PipelineCache {
    *CACHE.get_or_init(|| {
        let d = &ctx().device;
        START.get_or_init(std::time::Instant::now);
        let mut data = PATH.get().and_then(|p| std::fs::read(p).ok()).unwrap_or_default();
        if data.len() > MAX_LOAD {
            crate::warn!("pipeline cache: {} MiB of saved data (entries of earlier translations accumulate) is over {} MiB: starting empty", data.len() >> 20, MAX_LOAD >> 20);
            data = Vec::new();
        }
        if !data.is_empty() {
            let ci = vk::PipelineCacheCreateInfo::default().initial_data(&data);
            match unsafe { d.create_pipeline_cache(&ci, None) } {
                Ok(c) => {
                    crate::info!("pipeline cache: {} KiB loaded", data.len() >> 10);
                    SAVED_LEN.store(data.len() as u64, Ordering::Relaxed);
                    return c;
                }
                Err(e) => crate::warn!("pipeline cache: saved data rejected ({e}); starting empty"),
            }
        }
        unsafe { d.create_pipeline_cache(&vk::PipelineCacheCreateInfo::default(), None) }.unwrap_or_else(|e| {
            crate::warn!("pipeline cache: creation failed ({e}); pipelines are not cached");
            vk::PipelineCache::null()
        })
    })
}

/// Notes a pipeline creation; saves in the background now and then.
pub fn created() {
    let now = START.get_or_init(std::time::Instant::now).elapsed().as_millis() as u64;
    let last = LAST_SAVE_MS.load(Ordering::Relaxed);
    if NEW.fetch_add(1, Ordering::Relaxed) + 1 >= SAVE_EVERY && now >= last + SAVE_INTERVAL_MS && !SAVING.swap(true, Ordering::AcqRel) {
        NEW.store(0, Ordering::Relaxed);
        LAST_SAVE_MS.store(now, Ordering::Relaxed);
        let spawned = std::thread::Builder::new().name("pcache-save".into()).spawn(|| {
            save();
            SAVING.store(false, Ordering::Release);
        });
        if spawned.is_err() {
            SAVING.store(false, Ordering::Release);
        }
    }
}

/// Writes the cache to its file (temporary file, then rename).
pub fn save() {
    let (Some(path), Some(&cache)) = (PATH.get(), CACHE.get()) else { return };
    if cache == vk::PipelineCache::null() {
        return;
    }
    let data = match unsafe { ctx().device.get_pipeline_cache_data(cache) } {
        Ok(d) => d,
        Err(e) => {
            crate::warn!("pipeline cache: reading the data failed ({e})");
            return;
        }
    };
    if data.len() as u64 == SAVED_LEN.load(Ordering::Relaxed) {
        return;
    }
    let tmp = path.with_extension("tmp");
    let r = std::fs::write(&tmp, &data).and_then(|_| std::fs::rename(&tmp, path));
    match r {
        Ok(()) => {
            SAVED_LEN.store(data.len() as u64, Ordering::Relaxed);
            crate::info!("pipeline cache: {} KiB saved", data.len() >> 10)
        }
        Err(e) => crate::warn!("pipeline cache: saving to {} failed ({e})", path.display()),
    }
}
