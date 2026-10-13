//! VRAM mirrors of buffers that GPU passes write.
//!
//! Guest buffers are bound in place from system memory imported into Vulkan;
//! a draw's vertex stage then re-reads instance data across PCIe for every
//! vertex (the title's instanced foliage: 7616 instances of a 129-index mesh
//! per draw, ~4.5-8 ms each). When a dispatch writes a buffer, its result is
//! copied into a mirror in device memory right after it, and later read-only
//! bindings of that range use the mirror. A mirror is dropped when a CP write
//! or a GPU write that is not mirrored touches its range, or the range is
//! remapped, and is only used during the frame it was written in and the next
//! one: CPU writes to GPU-produced buffers are not seen (the title writes such
//! buffers on the GPU and reads them back, if at all, through host memory,
//! which keeps the data as before). `tune mirrors 0` turns it off.

use super::vk::ctx;
use ash::vk;
use parking_lot::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Device memory for mirrors (bump allocated; one mirror per written range).
const POOL_BYTES: u64 = 768 << 20;
/// Mirrored write sizes: smaller ranges are read cheaply in place.
pub const MIN: u64 = 16 << 10;
pub const MAX: u64 = 96 << 20;

struct Mirror {
    base: u64,
    size: u64,
    off: u64,
    /// Frame of the last refresh; None while invalid.
    frame: Option<u64>,
}

struct Pool {
    buffer: vk::Buffer,
    used: u64,
    list: Vec<Mirror>,
    full: bool,
}

static POOL: Mutex<Option<Pool>> = Mutex::new(None);
/// A mirror was made (until then `invalidate`, called for every command's
/// writes, has nothing to drop and takes no lock).
static LIVE: AtomicBool = AtomicBool::new(false);
/// Read bindings served from mirrors / refreshes (diagnostics).
pub static HITS: AtomicU64 = AtomicU64::new(0);
pub static REFRESHES: AtomicU64 = AtomicU64::new(0);

fn frame() -> u64 {
    super::FRAMES.load(Ordering::Relaxed)
}

fn with<R>(f: impl FnOnce(&mut Pool) -> R) -> Option<R> {
    let mut g = POOL.lock();
    if g.is_none() {
        match ctx().device_buffer(POOL_BYTES, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::INDEX_BUFFER) {
            Ok((buffer, _mem)) => *g = Some(Pool { buffer, used: 0, list: Vec::new(), full: false }),
            Err(e) => {
                crate::warn!("gpu: mirror pool: {e}");
                return None;
            }
        }
    }
    g.as_mut().map(f)
}

/// The mirror a write of [base, base+size) refreshes: (pool buffer, offset).
/// Other mirrors overlapping it are dropped. None: not mirrored (size, pool
/// full) — the caller then drops overlapping mirrors with `invalidate`.
pub fn for_write(base: u64, size: u64) -> Option<(vk::Buffer, u64)> {
    if !super::tune::mirrors() || !(MIN..=MAX).contains(&size) {
        return None;
    }
    with(|p| {
        for m in p.list.iter_mut() {
            if !(m.base == base && m.size == size) && m.base < base + size && base < m.base + m.size {
                m.frame = None;
            }
        }
        if let Some(m) = p.list.iter().find(|m| m.base == base && m.size == size) {
            return Some((p.buffer, m.off));
        }
        let off = p.used.next_multiple_of(256);
        if off + size > POOL_BYTES {
            if !p.full {
                p.full = true;
                crate::warn!("gpu: mirror pool full ({} mirrors): further written buffers are read from system memory", p.list.len());
            }
            return None;
        }
        p.used = off + size;
        p.list.push(Mirror { base, size, off, frame: None });
        LIVE.store(true, Ordering::Release);
        Some((p.buffer, off))
    })
    .flatten()
}

/// The refresh of [base, base+size) is recorded (valid for later work).
pub fn refreshed(base: u64, size: u64) {
    let f = frame();
    with(|p| {
        if let Some(m) = p.list.iter_mut().find(|m| m.base == base && m.size == size) {
            m.frame = Some(f);
            REFRESHES.fetch_add(1, Ordering::Relaxed);
        }
    });
}

/// A valid mirror covering [base, base+size): (buffer, offset of `base`,
/// bytes from `base` to the mirror's end).
pub fn lookup(base: u64, size: u64) -> Option<(vk::Buffer, u64, u64)> {
    if !super::tune::mirrors() {
        return None;
    }
    let f = frame();
    let g = POOL.lock();
    let p = g.as_ref()?;
    let m = p.list.iter().find(|m| m.frame.is_some_and(|w| f <= w + 1) && m.base <= base && base + size <= m.base + m.size)?;
    HITS.fetch_add(1, Ordering::Relaxed);
    Some((p.buffer, m.off + (base - m.base), m.base + m.size - base))
}

/// Drops the mirrors overlapping any of `ranges` (writes that do not
/// refresh them: CP writes, unmirrored GPU writes, remaps).
pub fn invalidate(ranges: &[(u64, u64)]) {
    if !LIVE.load(Ordering::Acquire) {
        return;
    }
    let mut g = POOL.lock();
    let Some(p) = g.as_mut() else { return };
    for m in p.list.iter_mut().filter(|m| m.frame.is_some()) {
        if ranges.iter().any(|&(a, n)| m.base < a + n && a < m.base + m.size) {
            m.frame = None;
        }
    }
}

/// One status line.
pub fn status() -> String {
    let g = POOL.lock();
    let (n, used) = g.as_ref().map(|p| (p.list.len(), p.used)).unwrap_or((0, 0));
    format!(
        "    buffer mirrors: {n} ({} MiB), {} refreshes, {} read bindings served\n",
        used >> 20,
        REFRESHES.load(Ordering::Relaxed),
        HITS.load(Ordering::Relaxed)
    )
}
