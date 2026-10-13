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
//! dropped, and the chunk is reused once no batch binds it any more.
//!
//! Written-buffer shadows (`Exec`: a dispatch's written binding in video
//! memory, merged back into guest memory when its batch ends) leave their
//! working copy here after the merge (`tune shadow_share`): the next batch
//! that shadows the same window copies it in from video memory instead of
//! across PCIe (~14 GB/s; ~4 ms of the courtyard frame's GPU time, mostly
//! 1-4 MB buffers GPU work rewrites every frame). Such a copy holds what the
//! merge left in guest memory only if nothing else wrote the window meanwhile:
//! CPU writes since its copy-in are dated by the write marks taken for the
//! window (`watch`, `unwritten_since`; every take dates the watched windows,
//! whichever entry it was for), CP writes into it in the same batch taint
//! the shadow (`Exec`), and the batch's own submission-time invalidation
//! spares the copies it made after its writes (`invalidate_ranges_except`).
//!
//! Copies over a chunk (read-only bindings of up to `tune pre_big_max`
//! bytes; off by default, no measured gain) get a video-memory slot of their
//! own within `tune pre_big_mib`. A big range that CPU or GPU writes changed
//! is not copied again for BIG_QUIET frames (a copy per change would cost
//! more than reading it in place). Without
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

impl Drop for Chunk {
    fn drop(&mut self) {
        // The last reference: no batch binds it (batches hold theirs until
        // they complete).
        super::vk::note_gpu_freed(self.buf);
        unsafe {
            ctx().device.destroy_buffer(self.buf, None);
            ctx().device.free_memory(self._mem, None);
        }
    }
}

/// Frames a big range that writes changed is not copied again for.
const BIG_QUIET: u64 = 120;

const CHUNK: u64 = 32 << 20;

struct Entry {
    chunk: Arc<Chunk>,
    /// Offset of the copy (its 16-byte aligned start) in the chunk.
    offset: u64,
    /// Unique per insertion: two batches may both copy a range they missed
    /// at the same time, the second entry replacing the first; only the
    /// batch that made an entry may mark it submitted (`submitted`), or a
    /// third batch could bind a copy whose batch is not submitted yet.
    id: u64,
    /// Sequence of the batch whose prologue copies it (0: not submitted).
    seq: u64,
    /// The synchronization epoch it was last checked for CPU writes in.
    checked: u64,
    /// Its chunk's slot (`Slot::live`).
    slot: usize,
    /// When it was made (a copy pending this long is given up: `insert`).
    made: std::time::Instant,
}

struct Slot {
    chunk: Arc<Chunk>,
    /// When it was last started (`Cache::fills`): eviction order.
    filled: u64,
    /// Keys of the entries made in it since (some may be gone or replaced):
    /// an eviction removes its entries without scanning all of them.
    keys: Vec<(u64, u64)>,
    /// Entries in it now: beyond them and the slot's own reference, the
    /// chunk's references are batches binding it (`Cache::unbound`).
    live: usize,
    /// Its buffer's size: CHUNK, or more for one big copy (`alloc_big`).
    cap: u64,
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
    /// Ranges GPU work wrote while they had an entry, and the frame: copied
    /// per batch for `tune pre_share_volatile` frames (their copies die
    /// within a frame and would fill the chunks).
    volatile: Map<(u64, u64), u64>,
    /// The chunk being filled and its use.
    current: Option<(usize, u64)>,
    fills: u64,
    failed: bool,
    /// Big ranges a write changed while they had an entry, and the frame
    /// (`BIG_QUIET`).
    big_changed: Map<(u64, u64), u64>,
    /// A big slot could not be allocated (warned once).
    big_failed: bool,
    /// Takes of write marks that reported pages, counted (`Watched`).
    marks: u64,
    /// Shadow windows whose CPU writes are followed (`watch`), oldest first.
    watched: Vec<Watched>,
    /// Entry ids `drop_range` leaves (`invalidate_ranges_except`).
    spare: Vec<u64>,
}

/// A written-buffer shadow's window [lo, hi): the count of mark takes
/// (`Cache::marks`) when one last reported a page in it.
struct Watched {
    lo: u64,
    hi: u64,
    written: u64,
}

/// Windows followed at most (the oldest go: unknown then, not kept).
const WATCHED_MAX: usize = 512;
/// Bounds of the watched windows, readable without the lock (`invalidate`
/// dates the windows a CPU-side write overlaps; LO > HI: none).
static WATCH_LO: AtomicU64 = AtomicU64::new(u64::MAX);
static WATCH_HI: AtomicU64 = AtomicU64::new(0);

static CACHE: Mutex<Option<Cache>> = Mutex::new(None);

/// Per bucket of 64 KiB guest pages (hashed), how many entries cover one of
/// its pages: writes whose buckets are all empty need not take the lock (GPU
/// and CP writes check every written range; with entries all over guest
/// memory the LO/HI bounds rarely exclude them: write tracking per dispatch
/// went from 0.57 to 1.59 us, profile run 20261012-001629). A write racing an
/// insert is caught again when its batch is submitted (`invalidate_ranges`).
const BUCKETS: usize = 1 << 16;
static COVER: [std::sync::atomic::AtomicU32; BUCKETS] = [const { std::sync::atomic::AtomicU32::new(0) }; BUCKETS];

fn bucket(page: u64) -> usize {
    (page.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 48) as usize
}

/// The same per 4 MiB (writes over 64 small buckets: the culls' 112 MB
/// index buffer, written by every one of their dispatches, took the cache's
/// lock each time).
const COARSE_BUCKETS: usize = 1 << 12;
static COVER_COARSE: [std::sync::atomic::AtomicU32; COARSE_BUCKETS] = [const { std::sync::atomic::AtomicU32::new(0) }; COARSE_BUCKETS];

fn coarse_bucket(page: u64) -> usize {
    (page.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 52) as usize
}

/// Counts an entry's pages in (or out of) the buckets.
fn cover(base: u64, size: u64, add: bool) {
    if size == 0 {
        return;
    }
    for page in (base >> 16)..=((base + size - 1) >> 16) {
        let c = &COVER[bucket(page)];
        if add {
            c.fetch_add(1, Ordering::AcqRel);
        } else {
            c.fetch_sub(1, Ordering::AcqRel);
        }
    }
    for page in (base >> 22)..=((base + size - 1) >> 22) {
        let c = &COVER_COARSE[coarse_bucket(page)];
        if add {
            c.fetch_add(1, Ordering::AcqRel);
        } else {
            c.fetch_sub(1, Ordering::AcqRel);
        }
    }
}

/// Whether an entry may overlap [base, base+len) (false: certainly none).
fn may_cover(base: u64, len: u64) -> bool {
    let (first, last) = (base >> 16, (base + len - 1) >> 16);
    if last - first >= 64 {
        let (first, last) = (base >> 22, (base + len - 1) >> 22);
        if last - first >= 64 {
            return true;
        }
        return (first..=last).any(|page| COVER_COARSE[coarse_bucket(page)].load(Ordering::Acquire) != 0);
    }
    (first..=last).any(|page| COVER[bucket(page)].load(Ordering::Acquire) != 0)
}
/// Bounds of all entries, readable without the lock (`invalidate` skips
/// writes outside them; LO > HI: no entries).
static LO: AtomicU64 = AtomicU64::new(u64::MAX);
static HI: AtomicU64 = AtomicU64::new(0);
/// The synchronization epoch (`end_scope`).
static EPOCH: AtomicU64 = AtomicU64::new(1);
/// Entry ids (`Entry::id`).
static NEXT_ID: AtomicU64 = AtomicU64::new(1);
/// Changes whenever an entry is inserted: while it stays the same, dropping
/// a range dropped before changes nothing (`Exec`'s write-note memo).
pub static INSERT_GEN: AtomicU64 = AtomicU64::new(0);

/// Statistics: [hits, copies, checks (epoch changed), dropped by CPU writes,
/// dropped by GPU/CP/host writes and mapping changes, dropped by eviction,
/// no space, failed checks, not kept (GPU-written lately), left to a batch
/// not submitted yet (`tune pre_keep`), big copies, big entries dropped by
/// writes, big copies declined (changed lately)].
static STATS: [AtomicU64; 13] = [const { AtomicU64::new(0) }; 13];
/// STATS at the last `profile reset` (`window_line`).
static SNAP: [AtomicU64; 13] = [const { AtomicU64::new(0) }; 13];

/// Starts a profile window (`profile reset`).
pub fn reset_window() {
    for (s, v) in SNAP.iter().zip(STATS.iter()) {
        s.store(v.load(Ordering::Relaxed), Ordering::Relaxed);
    }
}

/// The shared copies since the last `profile reset`.
pub fn window_line() -> String {
    let d: Vec<u64> = STATS.iter().zip(SNAP.iter()).map(|(v, s)| v.load(Ordering::Relaxed).saturating_sub(s.load(Ordering::Relaxed))).collect();
    format!(
        "shared: {} copies, {} reused, {} without space, {} left to a batch not submitted yet, entries dropped by eviction {}; big copies {}, dropped by writes {}, declined {}",
        d[1], d[0], d[6], d[9], d[5], d[10], d[11], d[12]
    )
}

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
        let Some(e) = self.entries.remove(&key) else { return false };
        if let Some(s) = self.chunks.get_mut(e.slot) {
            s.live = s.live.saturating_sub(1);
        }
        cover(key.0, key.1, false);
        if let Some(v) = self.by_base.get_mut(&key.0) {
            v.retain(|&s| s != key.1);
            if v.is_empty() {
                self.by_base.remove(&key.0);
            }
        }
        true
    }

    /// Drops the entries overlapping [base, end); how many. `gpu`: GPU work
    /// writes the range (the entries' keys turn volatile).
    fn drop_range(&mut self, base: u64, end: u64, gpu: bool) -> u64 {
        if self.entries.is_empty() || end <= self.lo || self.hi <= base {
            return 0;
        }
        let mut hit: Vec<(u64, u64)> = Vec::new();
        for (&b, sizes) in self.by_base.range(base.saturating_sub(self.max_size)..end) {
            for &s in sizes {
                if b < end && base < b + s && (self.spare.is_empty() || self.entries.get(&(b, s)).is_none_or(|e| !self.spare.contains(&e.id))) {
                    hit.push((b, s));
                }
            }
        }
        let frame = super::FRAMES.load(Ordering::Relaxed);
        for &k in &hit {
            self.remove(k);
            if gpu {
                self.volatile.insert(k, frame);
            }
            // (A copy's bytes run from the 16-byte aligned start.)
            if k.1 + 16 > CHUNK {
                self.big_changed.insert(k, frame);
                if STATS[11].fetch_add(1, Ordering::Relaxed) < 32 {
                    crate::info!("gpu: shared copy of {:#x}+{:#x} ({} MiB) dropped: {} wrote [{base:#x}, {end:#x})", k.0, k.1, k.1 >> 20, if gpu { "GPU work" } else { "the CPU, a CP or host write or a mapping change" });
                }
            }
        }
        if self.volatile.len() > 1 << 16 {
            let keep = super::tune::pre_share_volatile();
            self.volatile.retain(|_, f| f.saturating_add(keep) >= frame);
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
            n += self.drop_range(start, end, false);
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

    /// `mem::take_written` for the cache: the pages reported also date the
    /// watched windows they fall in (a failed query dates those overlapping
    /// the range), and texture regions on them other than `except` are told
    /// (`track::cpu_pages`: taking the marks resets them).
    fn take_marks(&mut self, base: u64, size: u64, out: &mut Vec<u64>, except: Option<u32>) -> Option<()> {
        let first = out.len();
        let r = crate::mem::take_written(base, size, out);
        match r {
            Some(()) if out.len() > first => super::track::cpu_pages(&out[first..], except),
            Some(()) => {}
            None => super::track::cpu_range(base, size, except),
        }
        if self.watched.is_empty() {
            return r;
        }
        match r {
            Some(()) if out.len() > first => {
                self.marks += 1;
                let g = self.marks;
                for &p in &out[first..] {
                    for w in self.watched.iter_mut() {
                        if w.lo <= p && p < w.hi {
                            w.written = g;
                        }
                    }
                }
            }
            Some(()) => {}
            None => {
                self.marks += 1;
                let (g, end) = (self.marks, base.saturating_add(size));
                for w in self.watched.iter_mut() {
                    if w.lo < end && base < w.hi {
                        w.written = g;
                    }
                }
            }
        }
        r
    }

    /// Takes the range's write marks: entries on written pages go (as in
    /// `lookup`), watched windows are dated.
    fn take_and_drop(&mut self, base: u64, size: u64) {
        let mut pages = Vec::new();
        match self.take_marks(base, size, &mut pages, None) {
            None => {
                STATS[7].fetch_add(1, Ordering::Relaxed);
                self.drop_range(base, base.saturating_add(size), false);
                self.bounds_changed();
            }
            Some(()) if !pages.is_empty() => {
                let n = self.drop_pages(&pages);
                STATS[3].fetch_add(n, Ordering::Relaxed);
                self.bounds_changed();
            }
            Some(()) => {}
        }
    }

    /// Whether no batch binds slot `k`'s chunk (only the slot and its
    /// entries hold it; batches take references under the lock only).
    fn unbound(&self, k: usize) -> bool {
        let s = &self.chunks[k];
        Arc::strong_count(&s.chunk) == 1 + s.live
    }

    /// Removes the entries in slot `k` (an eviction); whether its chunk is
    /// idle then.
    fn evict(&mut self, k: usize) -> bool {
        let victim = self.chunks[k].chunk.clone();
        for key in std::mem::take(&mut self.chunks[k].keys) {
            if self.entries.get(&key).is_some_and(|e| Arc::ptr_eq(&e.chunk, &victim)) {
                self.remove(key);
                STATS[5].fetch_add(1, Ordering::Relaxed);
            }
        }
        drop(victim);
        Arc::strong_count(&self.chunks[k].chunk) == 1
    }

    /// A slot of its own for a copy of `n` bytes over CHUNK: an idle big
    /// slot it fits in, a new one within `tune pre_big_mib`, or a big slot
    /// no batch binds (filled longest ago) evicted, and replaced by one of
    /// the size needed unless it fits.
    fn alloc_big(&mut self, n: u64) -> Option<(usize, Arc<Chunk>, u64)> {
        let size = n.next_multiple_of(1 << 20);
        let budget = super::tune::pre_big_mib() << 20;
        let big: u64 = self.chunks.iter().filter(|s| s.cap > CHUNK).map(|s| s.cap).sum();
        let idle_fit = (0..self.chunks.len()).filter(|&k| self.chunks[k].cap >= size && self.chunks[k].cap > CHUNK && Arc::strong_count(&self.chunks[k].chunk) == 1).min_by_key(|&k| self.chunks[k].cap);
        let k = match idle_fit {
            Some(k) => k,
            None => {
                // Room for a new one, or a big slot to evict: one it fits in,
                // or one whose size freed makes room.
                let k = if big + size <= budget {
                    None
                } else {
                    let old = (0..self.chunks.len())
                        .filter(|&k| self.chunks[k].cap > CHUNK && (self.chunks[k].cap >= size || big - self.chunks[k].cap + size <= budget) && self.unbound(k))
                        .min_by_key(|&k| self.chunks[k].filled)?;
                    if !self.evict(old) {
                        return None;
                    }
                    Some(old)
                };
                match k {
                    Some(k) if self.chunks[k].cap >= size => k,
                    _ => {
                        if self.big_failed {
                            return None;
                        }
                        let (buf, mem) = match ctx().device_buffer(size, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER) {
                            Ok(x) => x,
                            Err(e) => {
                                crate::warn!("gpu: no video memory for a {} MiB shared copy ({e}); big copies stop", size >> 20);
                                self.big_failed = true;
                                return None;
                            }
                        };
                        let chunk = Arc::new(Chunk { buf, _mem: mem });
                        match k {
                            // The evicted slot's buffer goes with its last reference.
                            Some(k) => {
                                self.chunks[k].chunk = chunk;
                                self.chunks[k].cap = size;
                                k
                            }
                            None => {
                                self.chunks.push(Slot { chunk, filled: 0, keys: Vec::new(), live: 0, cap: size });
                                self.chunks.len() - 1
                            }
                        }
                    }
                }
            }
        };
        self.fills += 1;
        let s = &mut self.chunks[k];
        s.filled = self.fills;
        s.keys.clear();
        s.live = 0;
        if STATS[10].fetch_add(1, Ordering::Relaxed) < 32 {
            crate::info!("gpu: big shared copy slot {k} of {} MiB for {} bytes", s.cap >> 20, n);
        }
        Some((k, s.chunk.clone(), 0))
    }

    /// Space for `n` bytes (256-byte aligned) in the chunk being filled, a
    /// free one, a new one, or (all in use) the chunk filled longest ago once
    /// nothing binds it (its entries go now). Over CHUNK: `alloc_big`.
    fn alloc(&mut self, n: u64) -> Option<(usize, Arc<Chunk>, u64)> {
        let n = n.next_multiple_of(256);
        if n > CHUNK {
            return self.alloc_big(n);
        }
        if let Some((k, used)) = &mut self.current {
            if *used + n <= CHUNK {
                let off = *used;
                *used += n;
                return Some((*k, self.chunks[*k].chunk.clone(), off));
            }
        }
        let max = (super::tune::pre_share_mib() as u64 * (1 << 20) / CHUNK).max(2) as usize;
        let idle = |s: &Slot| s.cap == CHUNK && Arc::strong_count(&s.chunk) == 1;
        let k = match self.chunks.iter().position(idle) {
            Some(k) => k,
            None if self.chunks.iter().filter(|s| s.cap == CHUNK).count() < max && !self.failed => match ctx().device_buffer(CHUNK, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER) {
                Ok((buf, mem)) => {
                    self.chunks.push(Slot { chunk: Arc::new(Chunk { buf, _mem: mem }), filled: 0, keys: Vec::new(), live: 0, cap: CHUNK });
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
                // `tune pre_keep`: of the chunks no batch binds now; entries
                // that every frame reuses keep theirs bound, and the chunk
                // filled longest ago is usually such a chunk, so the old
                // choice failed (no space) until the batches binding it
                // completed, with its entries gone (~45% of inserts failed,
                // profile 20261013-021047).
                let cur = self.current.map(|c| c.0);
                let keep = super::tune::pre_keep();
                let old = (0..self.chunks.len()).filter(|&k| Some(k) != cur && self.chunks[k].cap == CHUNK && (!keep || self.unbound(k))).min_by_key(|&k| self.chunks[k].filled)?;
                if !self.evict(old) {
                    return None;
                }
                old
            }
        };
        self.fills += 1;
        self.chunks[k].filled = self.fills;
        // An idle chunk has no entries left (each holds its chunk).
        self.chunks[k].keys.clear();
        self.chunks[k].live = 0;
        self.current = Some((k, n));
        Some((k, self.chunks[k].chunk.clone(), 0))
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
        match c.take_marks(base, size, &mut pages, None) {
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
/// start) and the new entry's id; the entry is usable by other batches once
/// `submitted` names it.
pub fn insert(base: u64, size: u64, n: u64) -> Option<(Arc<Chunk>, u64, u64)> {
    let mut g = CACHE.lock();
    let c = g.get_or_insert_with(Cache::default);
    if super::tune::pre_keep() && c.entries.get(&(base, size)).is_some_and(|e| e.seq == 0 && e.made.elapsed() < std::time::Duration::from_millis(500)) {
        // A batch not submitted yet copies it: this one copies it for itself
        // (replacing the entry wasted that copy and its space).
        STATS[9].fetch_add(1, Ordering::Relaxed);
        return None;
    }
    if n > CHUNK && c.big_changed.get(&(base, size)).is_some_and(|&f| f.saturating_add(BIG_QUIET) >= super::FRAMES.load(Ordering::Relaxed)) {
        STATS[12].fetch_add(1, Ordering::Relaxed);
        return None;
    }
    let keep = super::tune::pre_share_volatile();
    if keep > 0 && c.volatile.get(&(base, size)).is_some_and(|&f| f.saturating_add(keep) >= super::FRAMES.load(Ordering::Relaxed)) {
        STATS[8].fetch_add(1, Ordering::Relaxed);
        return None;
    }
    // The range's write marks so far: the copy sees those writes (it runs
    // later), other entries on those pages are stale.
    let mut pages = Vec::new();
    if c.take_marks(base, size, &mut pages, None).is_none() {
        STATS[7].fetch_add(1, Ordering::Relaxed);
        return None;
    }
    if !pages.is_empty() {
        let n = c.drop_pages(&pages);
        STATS[3].fetch_add(n, Ordering::Relaxed);
    }
    let Some((slot, chunk, offset)) = c.alloc(n) else {
        STATS[6].fetch_add(1, Ordering::Relaxed);
        c.bounds_changed();
        return None;
    };
    c.chunks[slot].keys.push((base, size));
    STATS[1].fetch_add(1, Ordering::Relaxed);
    c.remove((base, size));
    c.chunks[slot].live += 1;
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    c.entries.insert((base, size), Entry { chunk: chunk.clone(), offset, id, seq: 0, checked: EPOCH.load(Ordering::Acquire), slot, made: std::time::Instant::now() });
    INSERT_GEN.fetch_add(1, Ordering::Release);
    cover(base, size, true);
    c.by_base.entry(base).or_default().push(size);
    c.max_size = c.max_size.max(size);
    if c.entries.len() == 1 {
        (c.lo, c.hi) = (base, base + size);
    } else {
        c.lo = c.lo.min(base);
        c.hi = c.hi.max(base + size);
    }
    c.bounds_changed();
    Some((chunk, offset, id))
}

/// The batch `seq` whose prologue makes the copies of `made` ((base, size,
/// entry id)) was submitted: those entries, if still there, are usable.
pub fn submitted(made: &[(u64, u64, u64)], seq: u64) {
    if made.is_empty() {
        return;
    }
    let mut g = CACHE.lock();
    let Some(c) = g.as_mut() else { return };
    for &(base, size, id) in made {
        if let Some(e) = c.entries.get_mut(&(base, size)) {
            if e.id == id && e.seq == 0 {
                e.seq = seq;
            }
        }
    }
}

/// The pages of guest [base, base+size) the CPU wrote since their write
/// marks were last taken, into `out` (texture re-uploads; None: unknown,
/// treat it all as written). The marks are taken under the cache's lock like
/// its own takes: entries on those pages go, watched windows are dated, and
/// texture regions on them other than `except` (the caller's) are told.
pub fn take_written_for(base: u64, size: u64, out: &mut Vec<u64>, except: Option<u32>) -> Option<()> {
    let mut g = CACHE.lock();
    let c = g.get_or_insert_with(Cache::default);
    let first = out.len();
    let r = c.take_marks(base, size, out, except);
    match r {
        None => {
            STATS[7].fetch_add(1, Ordering::Relaxed);
            c.drop_range(base, base.saturating_add(size), false);
            c.bounds_changed();
        }
        Some(()) if out.len() > first => {
            let n = c.drop_pages(&out[first..]);
            STATS[3].fetch_add(n, Ordering::Relaxed);
            c.bounds_changed();
        }
        Some(()) => {}
    }
    r
}

/// Guest [base, base+len) may have changed (CP, host and mapping changes):
/// copies overlapping it go, and watched shadow windows it overlaps are dated
/// (file reads by DMA leave no write marks).
pub fn invalidate(base: u64, len: u64) {
    let end = base.saturating_add(len);
    if len != 0 && base < WATCH_HI.load(Ordering::Acquire) && WATCH_LO.load(Ordering::Acquire) < end {
        let mut g = CACHE.lock();
        if let Some(c) = g.as_mut() {
            if c.watched.iter().any(|w| w.lo < end && base < w.hi) {
                c.marks += 1;
                let m = c.marks;
                for w in c.watched.iter_mut().filter(|w| w.lo < end && base < w.hi) {
                    w.written = m;
                }
            }
        }
    }
    drop_overlapping(base, len, false);
}

/// GPU work writes guest [base, base+len): copies overlapping it go, and
/// their ranges are not kept for a while (`tune pre_share_volatile`).
pub fn invalidate_gpu(base: u64, len: u64) {
    drop_overlapping(base, len, true);
}

fn drop_overlapping(base: u64, len: u64, gpu: bool) {
    if len == 0 || base.saturating_add(len) <= LO.load(Ordering::Acquire) || HI.load(Ordering::Acquire) <= base || !may_cover(base, len) {
        return;
    }
    let mut g = CACHE.lock();
    let Some(c) = g.as_mut() else { return };
    let n = c.drop_range(base, base.saturating_add(len), gpu);
    if n > 0 {
        STATS[4].fetch_add(n, Ordering::Relaxed);
        c.bounds_changed();
    }
}

/// The ranges a batch writes, at its submission (see the module notes).
pub fn invalidate_ranges(writes: &super::ranges::Ranges) {
    invalidate_ranges_except(writes, &[]);
}

/// `invalidate_ranges`, sparing the entries `keep` (ids: the copies of
/// written-buffer shadows the batch made after its writes).
pub fn invalidate_ranges_except(writes: &super::ranges::Ranges, keep: &[u64]) {
    if keep.is_empty() {
        for (a, n) in writes.iter() {
            invalidate_gpu(a, n);
        }
        return;
    }
    let mut g = CACHE.lock();
    let Some(c) = g.as_mut() else { return };
    c.spare = keep.to_vec();
    for (a, n) in writes.iter() {
        c.drop_range(a, a.saturating_add(n), true);
    }
    c.spare.clear();
    c.bounds_changed();
}

/// Starts following CPU writes of a shadow's window [base, base+size) whose
/// copy in is recorded now: its marks so far are taken (the copy runs after
/// those writes) and the moment returned (`unwritten_since`).
pub fn watch(base: u64, size: u64) -> u64 {
    let mut g = CACHE.lock();
    let c = g.get_or_insert_with(Cache::default);
    c.take_and_drop(base, size);
    let (lo, hi) = (base & !4095, base.saturating_add(size).next_multiple_of(4096));
    if !c.watched.iter().any(|w| w.lo == lo && w.hi == hi) {
        if c.watched.len() >= WATCHED_MAX {
            c.watched.remove(0);
        }
        c.watched.push(Watched { lo, hi, written: 0 });
        // Bounds only grow (a dropped window only costs a lookup).
        WATCH_LO.fetch_min(lo, Ordering::AcqRel);
        WATCH_HI.fetch_max(hi, Ordering::AcqRel);
    }
    c.marks
}

/// Whether the CPU wrote no page of the window [base, base+size) since the
/// moment `since` (`watch`); false when unknown (the window's watch was
/// dropped).
pub fn unwritten_since(base: u64, size: u64, since: u64) -> bool {
    let mut g = CACHE.lock();
    let Some(c) = g.as_mut() else { return false };
    c.take_and_drop(base, size);
    let (lo, hi) = (base & !4095, base.saturating_add(size).next_multiple_of(4096));
    c.watched.iter().find(|w| w.lo == lo && w.hi == hi).is_some_and(|w| w.written <= since)
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
    let (chunks, entries, bigs, big_mib) = CACHE.lock().as_ref().map_or((0, 0, 0, 0), |c| {
        let big: Vec<u64> = c.chunks.iter().filter(|s| s.cap > CHUNK).map(|s| s.cap).collect();
        (c.chunks.len() - big.len(), c.entries.len(), big.len(), big.iter().sum::<u64>() >> 20)
    });
    format!(
        "    shared batch-start copies: {} copies, {} reused, {} not kept (GPU-written lately); {} checks for CPU writes ({} failed), entries dropped by CPU writes {}, by GPU/CP/host writes and mapping changes {}, by eviction {}; {} without space, {} left to a batch not submitted yet; {} entries in {} chunks of {} MiB and {} big slots ({} MiB; {} big copies, {} dropped by writes, {} declined as changed lately); epochs: {} submissions, {} waits the CPU released, {} REWIND\n",
        s[1],
        s[0],
        s[8],
        s[2],
        s[7],
        s[3],
        s[4],
        s[5],
        s[6],
        s[9],
        entries,
        chunks,
        CHUNK >> 20,
        bigs,
        big_mib,
        s[10],
        s[11],
        s[12],
        ENDS[0].load(Ordering::Relaxed),
        ENDS[1].load(Ordering::Relaxed),
        ENDS[2].load(Ordering::Relaxed)
    )
}
