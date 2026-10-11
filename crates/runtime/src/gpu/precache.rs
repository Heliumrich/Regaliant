//! Batch-start copies shared across batches (`Exec::pre_copy`).
//!
//! Read-only buffer bindings are copied into video memory in the prologue of
//! the batch that binds them. With ~170 batches per frame, and the chunks of
//! parallel runs each a batch of its own, 54% of the ~317 MB copied per
//! courtyard frame were ranges another batch of the same frame had already
//! copied (run 20261011-190051). Copies of the graphics queue's family (its
//! command processor and parallel workers) go to shared video-memory chunks
//! instead; an entry can be bound by later batches once the batch whose
//! prologue makes the copy is submitted (so the copy precedes them on the
//! one Vulkan queue). Batches keep the chunks they bind alive until they
//! complete. Entries are dropped when the copied guest range may have
//! changed: at the queue's next title submission and at its label waits
//! (CPU writes come between submissions, or behind a wait the CPU releases),
//! and on GPU writes, CP writes, host writes and mapping changes that touch
//! them.

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
const MAX_CHUNKS: usize = 24;

struct Entry {
    chunk: Arc<Chunk>,
    /// Offset of the copy (its 16-byte aligned start) in the chunk.
    offset: u64,
    /// Sequence of the batch whose prologue copies it (0: not submitted).
    seq: u64,
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
    current: Option<(Arc<Chunk>, u64)>,
    chunks: Vec<Arc<Chunk>>,
    failed: bool,
}

static CACHE: Mutex<Option<Cache>> = Mutex::new(None);
/// Bounds of all entries, readable without the lock (`invalidate` skips
/// writes outside them; LO > HI: no entries).
static LO: AtomicU64 = AtomicU64::new(u64::MAX);
static HI: AtomicU64 = AtomicU64::new(0);
/// Statistics: [hits, copies, invalidated, scope ends, no space].
static STATS: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];

/// Shared copies on (`tune pre_share`).
pub fn enabled() -> bool {
    super::tune::pre_share()
}

impl Cache {
    fn remove(&mut self, key: (u64, u64)) {
        if self.entries.remove(&key).is_some() {
            if let Some(v) = self.by_base.get_mut(&key.0) {
                v.retain(|&s| s != key.1);
                if v.is_empty() {
                    self.by_base.remove(&key.0);
                }
            }
        }
    }

    /// Space for `n` bytes (16-byte aligned).
    fn alloc(&mut self, n: u64) -> Option<(Arc<Chunk>, u64)> {
        let n = n.next_multiple_of(256);
        if n > CHUNK {
            return None;
        }
        if let Some((c, used)) = &mut self.current {
            if *used + n <= CHUNK {
                let off = *used;
                *used += n;
                return Some((c.clone(), off));
            }
        }
        // A chunk nothing references any more (no entry, no batch), else a
        // new one.
        let free = self.chunks.iter().find(|c| Arc::strong_count(c) == 1).cloned();
        let chunk = match free {
            Some(c) => c,
            None if self.chunks.len() < MAX_CHUNKS && !self.failed => match ctx().device_buffer(CHUNK, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER) {
                Ok((buf, mem)) => {
                    let c = Arc::new(Chunk { buf, _mem: mem });
                    self.chunks.push(c.clone());
                    c
                }
                Err(e) => {
                    crate::warn!("gpu: no video memory for shared batch-start copies ({e})");
                    self.failed = true;
                    return None;
                }
            },
            None => return None,
        };
        self.current = Some((chunk.clone(), n));
        Some((chunk, 0))
    }
}

/// A submitted copy of guest [base, base+size): (chunk, offset).
pub fn lookup(base: u64, size: u64) -> Option<(Arc<Chunk>, u64)> {
    let g = CACHE.lock();
    let e = g.as_ref()?.entries.get(&(base, size))?;
    if e.seq == 0 {
        return None;
    }
    STATS[0].fetch_add(1, Ordering::Relaxed);
    Some((e.chunk.clone(), e.offset))
}

/// Space for a new copy of [base, base+size) (`n` bytes from its aligned
/// start); the entry is usable by other batches once `submitted` names it.
pub fn insert(base: u64, size: u64, n: u64) -> Option<(Arc<Chunk>, u64)> {
    let mut g = CACHE.lock();
    let c = g.get_or_insert_with(Cache::default);
    let Some((chunk, offset)) = c.alloc(n) else {
        STATS[4].fetch_add(1, Ordering::Relaxed);
        return None;
    };
    STATS[1].fetch_add(1, Ordering::Relaxed);
    c.remove((base, size));
    c.entries.insert((base, size), Entry { chunk: chunk.clone(), offset, seq: 0 });
    c.by_base.entry(base).or_default().push(size);
    c.max_size = c.max_size.max(size);
    if c.entries.len() == 1 {
        (c.lo, c.hi) = (base, base + size);
    } else {
        c.lo = c.lo.min(base);
        c.hi = c.hi.max(base + size);
    }
    LO.store(c.lo, Ordering::Release);
    HI.store(c.hi, Ordering::Release);
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
    let end = base.saturating_add(len);
    if c.entries.is_empty() || end <= c.lo || c.hi <= base {
        return;
    }
    let mut hit: Vec<(u64, u64)> = Vec::new();
    for (&b, sizes) in c.by_base.range(base.saturating_sub(c.max_size)..end) {
        for &s in sizes {
            if b < end && base < b + s {
                hit.push((b, s));
            }
        }
    }
    STATS[2].fetch_add(hit.len() as u64, Ordering::Relaxed);
    for k in hit {
        c.remove(k);
    }
}

/// The queue family's scope ends (a title submission, a label wait): all
/// entries go (their chunks are reused once no batch binds them).
pub fn end_scope() {
    let mut g = CACHE.lock();
    let Some(c) = g.as_mut() else { return };
    if c.entries.is_empty() {
        return;
    }
    STATS[3].fetch_add(1, Ordering::Relaxed);
    c.entries.clear();
    c.by_base.clear();
    c.max_size = 0;
    LO.store(u64::MAX, Ordering::Release);
    HI.store(0, Ordering::Release);
}

/// One status line ("" before any use).
pub fn status() -> String {
    let s: Vec<u64> = STATS.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    if s[1] == 0 {
        return String::new();
    }
    let chunks = CACHE.lock().as_ref().map_or(0, |c| c.chunks.len());
    format!(
        "    shared batch-start copies: {} copies, {} reused by later batches, {} dropped by writes, {} scope ends, {} without space; {} chunks of {} MiB\n",
        s[1],
        s[0],
        s[2],
        s[3],
        s[4],
        chunks,
        CHUNK >> 20
    )
}
