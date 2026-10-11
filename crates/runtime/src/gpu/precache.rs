//! Batch-start copies kept in video memory across batches and frames
//! (`Exec::pre_copy`; `tune pre_share`).
//!
//! Read-only buffer bindings are copied into video memory in the prologue of
//! the batch that binds them. In the courtyard that was ~320 MB per frame
//! (~46% of the GPU's busy time); 54% repeated a range copied earlier in the
//! same frame and 38% was a range unchanged since its copy one or two frames
//! before (run 20261011-215002). Copies of ranges of at least
//! `tune pre_share_min` bytes now go to shared video-memory chunks and stay
//! there: a later batch of any queue binds the copy instead of copying again,
//! once the batch whose prologue makes it is submitted (so the copy precedes
//! it on the one Vulkan queue) and while the range is unchanged:
//!
//! - CPU writes: the guest range is write-watched (`mem::take_written`). An
//!   entry is checked once per synchronization epoch (`end_scope`: a title
//!   submission starts, a wait the CPU released, REWIND; any queue): writes
//!   the title orders before later commands are made before such an event.
//!   A written page drops every entry on it (taking its marks resets them).
//! - GPU and CP writes, host writes (file reads), mapping changes:
//!   `invalidate`, when the write is recorded and again when its batch is
//!   submitted (a copy recorded later by another queue's batch submitted
//!   earlier precedes the write on the GPU).
//!
//! Batches keep the chunks they bind alive until they complete. When all
//! chunks are in use, the entries of the chunk filled longest ago are
//! dropped, and the chunk is reused once no batch binds it any more. Without
//! write watching (`DS_WRITE_WATCH=0`, or the reservation failed) nothing is
//! shared.

use super::fx::Map;
use super::vk::ctx;
use ash::vk;
use parking_lot::Mutex;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// A video-memory buffer copies are placed in.
pub struct Chunk {
    pub buf: vk::Buffer,
    _mem: vk::DeviceMemory,
}

const CHUNK: u64 = 32 << 20;

struct Entry {
    chunk: Arc<Chunk>,
    /// Offset of the copy (its 16-byte aligned start) in the chunk.
    offset: u64,
    /// Sequence of the batch whose prologue copies it (0: not submitted).
    seq: u64,
    /// The synchronization epoch it was last checked for CPU writes in.
    checked: u64,
}

struct Slot {
    chunk: Arc<Chunk>,
    /// When it was last started (`Cache::fills`): eviction order.
    filled: u64,
}

#[derive(Default)]
struct Cache {
    entries: Map<(u64, u64), Entry>,
    /// Entry keys by base, and the largest size: overlap queries.
    by_base: BTreeMap<u64, Vec<u64>>,
    max_size: u64,
    /// Bounds of all entries (writes outside need no lookup).
    lo: u64,
    hi: u64,
    chunks: Vec<Slot>,
    /// The chunk being filled and its use.
    current: Option<(usize, u64)>,
    fills: u64,
    failed: bool,
}

static CACHE: Mutex<Option<Cache>> = Mutex::new(None);
/// Bounds of all entries, readable without the lock (`invalidate` skips
/// writes outside them; LO > HI: no entries).
static LO: AtomicU64 = AtomicU64::new(u64::MAX);
static HI: AtomicU64 = AtomicU64::new(0);
/// The synchronization epoch (`end_scope`).
static EPOCH: AtomicU64 = AtomicU64::new(1);

/// Statistics: [hits, copies, checks (epoch changed), dropped by CPU writes,
/// dropped by GPU/CP/host writes and mapping changes, dropped by eviction,
/// no space, failed checks].
static STATS: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];
/// Epochs by reason: [title submission, CPU-released wait, REWIND].
static ENDS: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// Shared copies on (`tune pre_share`, and the guest range write-watched).
pub fn enabled() -> bool {
    super::tune::pre_share() && crate::mem::write_watched()
}

/// Whether a binding of `size` bytes is worth sharing (`tune pre_share_min`;
/// smaller copies cost less than their checks).
pub fn worth(size: u64) -> bool {
    size >= super::tune::pre_share_min()
}

impl Cache {
    fn remove(&mut self, key: (u64, u64)) -> bool {
        if self.entries.remove(&key).is_none() {
            return false;
        }
        if let Some(v) = self.by_base.get_mut(&key.0) {
            v.retain(|&s| s != key.1);
            if v.is_empty() {
                self.by_base.remove(&key.0);
            }
        }
        true
    }

    /// Drops the entries overlapping [base, end); how many.
    fn drop_range(&mut self, base: u64, end: u64) -> u64 {
        if self.entries.is_empty() || end <= self.lo || self.hi <= base {
            return 0;
        }
        let mut hit: Vec<(u64, u64)> = Vec::new();
        for (&b, sizes) in self.by_base.range(base.saturating_sub(self.max_size)..end) {
            for &s in sizes {
                if b < end && base < b + s {
                    hit.push((b, s));
                }
            }
        }
        for &k in &hit {
            self.remove(k);
        }
        hit.len() as u64
    }

    /// Drops the entries on written pages (sorted page addresses).
    fn drop_pages(&mut self, pages: &[u64]) -> u64 {
        let mut n = 0;
        let mut i = 0;
        while i < pages.len() {
            let start = pages[i];
            let mut end = start + 4096;
            i += 1;
            while i < pages.len() && pages[i] == end {
                end += 4096;
                i += 1;
            }
            n += self.drop_range(start, end);
        }
        n
    }

    fn bounds_changed(&mut self) {
        if self.entries.is_empty() {
            LO.store(u64::MAX, Ordering::Release);
            HI.store(0, Ordering::Release);
        } else {
            LO.store(self.lo, Ordering::Release);
            HI.store(self.hi, Ordering::Release);
        }
    }

    /// Space for `n` bytes (256-byte aligned) in the chunk being filled, a
    /// free one, a new one, or (all in use) the chunk filled longest ago once
    /// nothing binds it (its entries go now).
    fn alloc(&mut self, n: u64) -> Option<(Arc<Chunk>, u64)> {
        let n = n.next_multiple_of(256);
        if n > CHUNK {
            return None;
        }
        if let Some((k, used)) = &mut self.current {
            if *used + n <= CHUNK {
                let off = *used;
                *used += n;
                return Some((self.chunks[*k].chunk.clone(), off));
            }
        }
        let max = (super::tune::pre_share_mib() as u64 * (1 << 20) / CHUNK).max(2) as usize;
        let idle = |s: &Slot| Arc::strong_count(&s.chunk) == 1;
        let k = match self.chunks.iter().position(idle) {
            Some(k) => k,
            None if self.chunks.len() < max && !self.failed => match ctx().device_buffer(CHUNK, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER) {
                Ok((buf, mem)) => {
                    self.chunks.push(Slot { chunk: Arc::new(Chunk { buf, _mem: mem }), filled: 0 });
                    self.chunks.len() - 1
                }
                Err(e) => {
                    crate::warn!("gpu: no video memory for shared batch-start copies ({e})");
                    self.failed = true;
                    return None;
                }
            },
            None => {
                // Evict the chunk filled longest ago (not the current one):
                // its entries go, the chunk follows once no batch binds it.
                let cur = self.current.map(|c| c.0);
                let old = (0..self.chunks.len()).filter(|&k| Some(k) != cur).min_by_key(|&k| self.chunks[k].filled)?;
                let victim = self.chunks[old].chunk.clone();
                let keys: Vec<(u64, u64)> = self.entries.iter().filter(|(_, e)| Arc::ptr_eq(&e.chunk, &victim)).map(|(k, _)| *k).collect();
                STATS[5].fetch_add(keys.len() as u64, Ordering::Relaxed);
                for k in keys {
                    self.remove(k);
                }
                drop(victim);
                if !idle(&self.chunks[old]) {
                    return None;
                }
                old
            }
        };
        self.fills += 1;
        self.chunks[k].filled = self.fills;
        self.current = Some((k, n));
        Some((self.chunks[k].chunk.clone(), 0))
    }
}

/// A submitted copy of guest [base, base+size) that is still current:
/// (chunk, offset). Checks it for CPU writes once per epoch.
pub fn lookup(base: u64, size: u64) -> Option<(Arc<Chunk>, u64)> {
    let mut g = CACHE.lock();
    let c = g.as_mut()?;
    let e = c.entries.get(&(base, size))?;
    if e.seq == 0 {
        return None;
    }
    let epoch = EPOCH.load(Ordering::Acquire);
    if e.checked != epoch {
        STATS[2].fetch_add(1, Ordering::Relaxed);
        let mut pages = Vec::new();
        match crate::mem::take_written(base, size, &mut pages) {
            None => {
                STATS[7].fetch_add(1, Ordering::Relaxed);
                c.remove((base, size));
                c.bounds_changed();
                return None;
            }
            Some(()) if !pages.is_empty() => {
                let n = c.drop_pages(&pages);
                STATS[3].fetch_add(n, Ordering::Relaxed);
                c.bounds_changed();
                return None;
            }
            Some(()) => {}
        }
        c.entries.get_mut(&(base, size))?.checked = epoch;
    }
    let e = c.entries.get(&(base, size))?;
    STATS[0].fetch_add(1, Ordering::Relaxed);
    Some((e.chunk.clone(), e.offset))
}

/// Space for a new copy of [base, base+size) (`n` bytes from its aligned
/// start); the entry is usable by other batches once `submitted` names it.
pub fn insert(base: u64, size: u64, n: u64) -> Option<(Arc<Chunk>, u64)> {
    let mut g = CACHE.lock();
    let c = g.get_or_insert_with(Cache::default);
    // The range's write marks so far: the copy sees those writes (it runs
    // later), other entries on those pages are stale.
    let mut pages = Vec::new();
    if crate::mem::take_written(base, size, &mut pages).is_none() {
        STATS[7].fetch_add(1, Ordering::Relaxed);
        return None;
    }
    if !pages.is_empty() {
        let n = c.drop_pages(&pages);
        STATS[3].fetch_add(n, Ordering::Relaxed);
    }
    let Some((chunk, offset)) = c.alloc(n) else {
        STATS[6].fetch_add(1, Ordering::Relaxed);
        c.bounds_changed();
        return None;
    };
    STATS[1].fetch_add(1, Ordering::Relaxed);
    c.remove((base, size));
    c.entries.insert((base, size), Entry { chunk: chunk.clone(), offset, seq: 0, checked: EPOCH.load(Ordering::Acquire) });
    c.by_base.entry(base).or_default().push(size);
    c.max_size = c.max_size.max(size);
    if c.entries.len() == 1 {
        (c.lo, c.hi) = (base, base + size);
    } else {
        c.lo = c.lo.min(base);
        c.hi = c.hi.max(base + size);
    }
    c.bounds_changed();
    Some((chunk, offset))
}

/// The batch `seq` whose prologue makes the copies of `keys` was submitted.
pub fn submitted(keys: &[(u64, u64)], seq: u64) {
    if keys.is_empty() {
        return;
    }
    let mut g = CACHE.lock();
    let Some(c) = g.as_mut() else { return };
    for k in keys {
        if let Some(e) = c.entries.get_mut(k) {
            if e.seq == 0 {
                e.seq = seq;
            }
        }
    }
}

/// Guest [base, base+len) may have changed: copies overlapping it go.
pub fn invalidate(base: u64, len: u64) {
    if len == 0 || base.saturating_add(len) <= LO.load(Ordering::Acquire) || HI.load(Ordering::Acquire) <= base {
        return;
    }
    let mut g = CACHE.lock();
    let Some(c) = g.as_mut() else { return };
    let n = c.drop_range(base, base.saturating_add(len));
    if n > 0 {
        STATS[4].fetch_add(n, Ordering::Relaxed);
        c.bounds_changed();
    }
}

/// The ranges a batch writes, at its submission (see the module notes).
pub fn invalidate_ranges(writes: &super::ranges::Ranges) {
    for (a, n) in writes.iter() {
        invalidate(a, n);
    }
}

/// A synchronization epoch ends (a title submission starts, a wait the CPU
/// released, REWIND; any queue): entries are checked for CPU writes again at
/// their next use.
pub fn end_scope(why: usize) {
    ENDS[why.min(2)].fetch_add(1, Ordering::Relaxed);
    EPOCH.fetch_add(1, Ordering::AcqRel);
}

/// One status line ("" before any use).
pub fn status() -> String {
    let s: Vec<u64> = STATS.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    if s[1] == 0 {
        return String::new();
    }
    let (chunks, entries) = CACHE.lock().as_ref().map_or((0, 0), |c| (c.chunks.len(), c.entries.len()));
    format!(
        "    shared batch-start copies: {} copies, {} reused; {} checks for CPU writes ({} failed), entries dropped by CPU writes {}, by GPU/CP/host writes and mapping changes {}, by eviction {}; {} without space; {} entries in {} chunks of {} MiB; epochs: {} submissions, {} waits the CPU released, {} REWIND\n",
        s[1],
        s[0],
        s[2],
        s[7],
        s[3],
        s[4],
        s[5],
        s[6],
        entries,
        chunks,
        CHUNK >> 20,
        ENDS[0].load(Ordering::Relaxed),
        ENDS[1].load(Ordering::Relaxed),
        ENDS[2].load(Ordering::Relaxed)
    )
}
