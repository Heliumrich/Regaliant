//! Video memory of images in blocks ("slabs"): an image's memory is a range of a slab, so making
//! one costs no `vkAllocateMemory` on the thread that records the frame. Each allocation of its
//! own cost 2-17 ms on the RTX 4090 at area arrivals, 149 ms at worst, whatever the image's size
//! (FINDINGS.md 2026-11-07). A helper thread allocates the next slab ahead; images over an eighth
//! of a slab keep an allocation of their own. Only images go into slabs (all optimally tiled,
//! so bufferImageGranularity does not apply). In the 23 arrivals of 1-1 the time over the median
//! went from 6.1 to 5.8 s and the worst frame from 267 to 217 ms, the GPU's time per frame level.
//!
//! Only with a device-local heap of 16 GiB or more (`on`): under a small budget the image cache
//! evicts, and an evicted image's range stays a hole in its slab until the slab empties (an 8 GB
//! budget left 29 slabs half used after evicting 1.9 GB), video memory that eviction exists to
//! give back. Smaller cards keep an allocation per image, as before.

use super::vk::ctx;
use ash::vk;
use parking_lot::{Condvar, Mutex};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// An image's memory: an allocation of its own or a range of a slab.
#[derive(Clone, Copy, Debug)]
pub struct Mem {
    pub memory: vk::DeviceMemory,
    pub offset: u64,
    size: u64,
    /// Its slab's index, or OWN.
    slab: u32,
}

const OWN: u32 = u32::MAX;

struct Slab {
    memory: vk::DeviceMemory,
    ty: u32,
    size: u64,
    used: u64,
    /// Free ranges (offset, length) by offset, neighbours merged.
    free: Vec<(u64, u64)>,
}

impl Slab {
    /// First fit of `size` bytes aligned to `align`.
    fn carve(&mut self, size: u64, align: u64) -> Option<u64> {
        let i = self.free.iter().position(|&(o, n)| o.next_multiple_of(align) + size <= o + n)?;
        let (o, n) = self.free[i];
        let at = o.next_multiple_of(align);
        let (head, tail) = (at - o, o + n - (at + size));
        match (head > 0, tail > 0) {
            (true, true) => {
                self.free[i] = (o, head);
                self.free.insert(i + 1, (at + size, tail));
            }
            (true, false) => self.free[i] = (o, head),
            (false, true) => self.free[i] = (at + size, tail),
            (false, false) => {
                self.free.remove(i);
            }
        }
        self.used += size;
        Some(at)
    }

    fn release(&mut self, at: u64, size: u64) {
        let i = self.free.partition_point(|&(o, _)| o < at);
        let joins_prev = i > 0 && self.free[i - 1].0 + self.free[i - 1].1 == at;
        let joins_next = i < self.free.len() && at + size == self.free[i].0;
        match (joins_prev, joins_next) {
            (true, true) => {
                self.free[i - 1].1 += size + self.free[i].1;
                self.free.remove(i);
            }
            (true, false) => self.free[i - 1].1 += size,
            (false, true) => self.free[i] = (at, size + self.free[i].1),
            (false, false) => self.free.insert(i, (at, size)),
        }
        self.used -= size;
    }
}

struct State {
    slabs: Vec<Option<Slab>>,
    /// Slabs allocated ahead, not in use: (memory type, memory).
    spare: Vec<(u32, vk::DeviceMemory)>,
    /// Memory types the helper is to allocate a spare slab of.
    wanted: Vec<u32>,
    helper: bool,
    failed: bool,
}

static STATE: Mutex<State> = Mutex::new(State { slabs: Vec::new(), spare: Vec::new(), wanted: Vec::new(), helper: false, failed: false });
static WAKE: Condvar = Condvar::new();

/// [images in slabs now, their bytes, images with memory of their own now, their bytes, slabs
/// made, slabs made by the thread that needed one (no spare), its ns, the helper's ns, slabs
/// freed, allocations of their own made, their ns].
static STATS: [AtomicU64; 11] = [const { AtomicU64::new(0) }; 11];
/// Images the driver required / preferred an allocation of their own for.
static DEDICATED: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// A slab's size: a 64th of the video memory, 64-256 MiB.
/// Slabs in use: a device-local heap of 16 GiB or more (at most `vram_cap`).
pub fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| ctx().device_heap_size() >= 16 << 30)
}

fn slab_size() -> u64 {
    static SIZE: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *SIZE.get_or_init(|| (ctx().device_heap_size() / 64).clamp(64 << 20, 256 << 20) & !((1 << 20) - 1))
}

fn allocate(ty: u32, size: u64) -> Result<vk::DeviceMemory, vk::Result> {
    let ai = vk::MemoryAllocateInfo::default().allocation_size(size).memory_type_index(ty);
    unsafe { ctx().device.allocate_memory(&ai, None) }
}

/// Memory of its own, as images had before slabs.
fn own(req: vk::MemoryRequirements) -> Result<Mem, String> {
    let t0 = std::time::Instant::now();
    let memory = ctx().alloc(req, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
    STATS[9].fetch_add(1, Relaxed);
    STATS[10].fetch_add(t0.elapsed().as_nanos() as u64, Relaxed);
    STATS[2].fetch_add(1, Relaxed);
    STATS[3].fetch_add(req.size, Relaxed);
    Ok(Mem { memory, offset: 0, size: req.size, slab: OWN })
}

/// Memory for an image with requirements `req` (bound at `offset` of `memory`).
pub fn alloc(req: vk::MemoryRequirements, requires_own: bool, prefers_own: bool) -> Result<Mem, String> {
    if requires_own {
        DEDICATED[0].fetch_add(1, Relaxed);
    }
    if prefers_own {
        DEDICATED[1].fetch_add(1, Relaxed);
    }
    if !on() || requires_own {
        return own(req);
    }
    let size = slab_size();
    if req.size > size / 8 {
        return own(req);
    }
    let Some(ty) = ctx().memory_type(req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL) else { return own(req) };
    let align = req.alignment.max(1);
    let mut g = STATE.lock();
    if g.failed {
        drop(g);
        return own(req);
    }
    // In the first slab with room (first fit).
    let found = g.slabs.iter_mut().enumerate().find_map(|(i, s)| {
        let s = s.as_mut().filter(|s| s.ty == ty)?;
        s.carve(req.size, align).map(|at| (i, s.memory, at))
    });
    if let Some((i, memory, at)) = found {
        drop(g);
        return Ok(placed(i, memory, at, req.size));
    }
    // A new slab: the spare, else one allocated here.
    let memory = match g.spare.iter().position(|s| s.0 == ty) {
        Some(i) => g.spare.swap_remove(i).1,
        None => {
            drop(g);
            let t0 = std::time::Instant::now();
            let r = allocate(ty, size);
            let ns = t0.elapsed().as_nanos() as u64;
            STATS[5].fetch_add(1, Relaxed);
            STATS[6].fetch_add(ns, Relaxed);
            super::hitch::note_since(super::hitch::Kind::Upload, t0, || format!("image slab allocation ({} MiB, no spare)", size >> 20));
            g = STATE.lock();
            match r {
                Ok(m) => m,
                Err(e) => {
                    // Out of video memory for a slab: images get memory of
                    // their own from now on (as before slabs).
                    if !g.failed {
                        g.failed = true;
                        crate::warn!("images: a {} MiB slab of video memory failed ({e}): images get memory of their own from now on", size >> 20);
                    }
                    drop(g);
                    return own(req);
                }
            }
        }
    };
    STATS[4].fetch_add(1, Relaxed);
    let mut s = Slab { memory, ty, size, used: 0, free: vec![(0, size)] };
    let at = s.carve(req.size, align).expect("an image an eighth of a slab fits an empty one");
    let i = match g.slabs.iter().position(Option::is_none) {
        Some(i) => {
            g.slabs[i] = Some(s);
            i
        }
        None => {
            g.slabs.push(Some(s));
            g.slabs.len() - 1
        }
    };
    want_spare(&mut g, ty);
    drop(g);
    Ok(placed(i, memory, at, req.size))
}

fn placed(i: usize, memory: vk::DeviceMemory, at: u64, size: u64) -> Mem {
    STATS[0].fetch_add(1, Relaxed);
    STATS[1].fetch_add(size, Relaxed);
    Mem { memory, offset: at, size, slab: i as u32 }
}

/// Asks the helper for a spare slab of memory type `ty` (none yet).
fn want_spare(g: &mut State, ty: u32) {
    if g.spare.iter().any(|s| s.0 == ty) || g.wanted.contains(&ty) {
        return;
    }
    g.wanted.push(ty);
    if !g.helper {
        g.helper = true;
        std::thread::Builder::new().name("image-slabs".into()).spawn(helper).ok();
    }
    WAKE.notify_one();
}

fn helper() {
    let mut g = STATE.lock();
    loop {
        while g.wanted.is_empty() {
            WAKE.wait(&mut g);
        }
        let ty = g.wanted.remove(0);
        if g.spare.iter().any(|s| s.0 == ty) {
            continue;
        }
        drop(g);
        let t0 = std::time::Instant::now();
        let r = allocate(ty, slab_size());
        STATS[7].fetch_add(t0.elapsed().as_nanos() as u64, Relaxed);
        g = STATE.lock();
        match r {
            Ok(m) => g.spare.push((ty, m)),
            Err(e) => {
                if !g.failed {
                    crate::warn!("images: a spare {} MiB slab of video memory failed ({e})", slab_size() >> 20);
                }
            }
        }
    }
}

/// Frees an image's memory (its batches are done).
pub fn free(m: Mem) {
    if m.slab == OWN {
        unsafe { ctx().device.free_memory(m.memory, None) };
        STATS[2].fetch_sub(1, Relaxed);
        STATS[3].fetch_sub(m.size, Relaxed);
        return;
    }
    STATS[0].fetch_sub(1, Relaxed);
    STATS[1].fetch_sub(m.size, Relaxed);
    let mut g = STATE.lock();
    let Some(s) = g.slabs.get_mut(m.slab as usize).and_then(|s| s.as_mut()) else { return };
    s.release(m.offset, m.size);
    if s.used > 0 {
        return;
    }
    // Empty: the spare of its type if there is none, else freed.
    let s = g.slabs[m.slab as usize].take().unwrap();
    if !g.spare.iter().any(|x| x.0 == s.ty) {
        g.spare.push((s.ty, s.memory));
        return;
    }
    drop(g);
    unsafe { ctx().device.free_memory(s.memory, None) };
    STATS[8].fetch_add(1, Relaxed);
}

/// Video memory the slabs hold but no image uses (free ranges and spares), bytes: memory that
/// new images take without an allocation (`images::maybe_evict` counts it as free).
pub fn idle_bytes() -> u64 {
    if !on() {
        return 0;
    }
    let g = STATE.lock();
    let holes: u64 = g.slabs.iter().flatten().map(|s| s.size - s.used).sum();
    holes + g.spare.len() as u64 * slab_size()
}

/// Status line part.
pub fn status() -> String {
    let st: Vec<u64> = STATS.iter().map(|c| c.load(Relaxed)).collect();
    if !on() {
        return format!("image memory: no slabs (a device-local heap under 16 GiB), {} images with their own ({} MiB; {} allocations, {} ms)", st[2], st[3] >> 20, st[9], st[10] / 1_000_000);
    }
    let g = STATE.lock();
    let slabs: Vec<&Slab> = g.slabs.iter().flatten().collect();
    let bytes: u64 = slabs.iter().map(|s| s.size).sum();
    let used: u64 = slabs.iter().map(|s| s.used).sum();
    let hole = slabs.iter().flat_map(|s| s.free.iter().map(|f| f.1)).max().unwrap_or(0);
    format!(
        "image memory: {} images in {} slabs of {} MiB ({} MiB, {} MiB of it used, largest hole {} MiB; {} spare), {} with their own ({} MiB); slabs made {} ({} on the thread that needed one, {} ms; the helper's {} ms), freed {}; allocations of their own {} ({} ms); the driver required one of their own for {} images, preferred it for {}",
        st[0],
        slabs.len(),
        slab_size() >> 20,
        bytes >> 20,
        used >> 20,
        hole >> 20,
        g.spare.len(),
        st[2],
        st[3] >> 20,
        st[4],
        st[5],
        st[6] / 1_000_000,
        st[7] / 1_000_000,
        st[8],
        st[9],
        st[10] / 1_000_000,
        DEDICATED[0].load(Relaxed),
        DEDICATED[1].load(Relaxed)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slab(size: u64) -> Slab {
        Slab { memory: vk::DeviceMemory::null(), ty: 0, size, used: 0, free: vec![(0, size)] }
    }

    #[test]
    fn carve_and_release_merge() {
        let mut s = slab(1 << 20);
        let a = s.carve(1000, 256).unwrap();
        let b = s.carve(5000, 4096).unwrap();
        let c = s.carve(100, 1).unwrap();
        assert_eq!(a, 0);
        assert_eq!(b, 4096);
        assert!(c >= 1000 && c + 100 <= 4096, "the gap before b is used: {c}");
        assert_eq!(s.used, 6100);
        s.release(b, 5000);
        s.release(a, 1000);
        s.release(c, 100);
        assert_eq!(s.used, 0);
        assert_eq!(s.free, vec![(0, 1 << 20)]);
    }

    #[test]
    fn full_slab_refuses() {
        let mut s = slab(8192);
        assert_eq!(s.carve(8192, 4096), Some(0));
        assert_eq!(s.carve(1, 1), None);
        s.release(0, 8192);
        assert_eq!(s.carve(4096, 4096), Some(0));
        assert_eq!(s.carve(4096, 4096), Some(4096));
        assert_eq!(s.free, vec![]);
    }
}
