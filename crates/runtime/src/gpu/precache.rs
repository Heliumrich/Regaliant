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
//! it on the main Vulkan queue, or the binding batch waits for it on the copy
//! queue's timeline: `lookup`) and while the range is unchanged:
//!
//! - CPU writes: the guest range is write-watched (`mem::take_written`). An
//!   entry is checked once per synchronization epoch (`end_scope`: a title
//!   submission starts, a wait the CPU released, REWIND; any queue): writes
//!   the title orders before later commands are made before such an event.
//!   A written page drops every entry on it (taking its marks resets them).
//!   Marks are taken without the cache's lock (`take_unlocked`: the query
//!   is a system call, and under the lock it made the recording threads wait
//!   ~10 ms per flip in all): a take is registered first and its pages are
//!   applied after under the lock; a check that would find a range unchanged
//!   first waits for the takes over it that began before it ended, since
//!   those may have reset its pages (`wait_takes`).
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
//! (Copies over a chunk in slots of their own, `tune pre_big_max`, no
//! measured gain, were removed on 2026-11-01.) Without
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
    /// Copied on the copy queue (batches binding it wait for its batch's
    /// copies there: `lookup`).
    on_copy: bool,
    /// The synchronization epoch it was last checked for CPU writes in.
    checked: u64,
    /// The flip it was made or last found clean by a take in (`tune
    /// pre_trust`).
    clean_flip: u64,
    /// Its chunk's slot (`Slot::live`).
    slot: usize,
    /// When it was made (a copy pending this long is given up: `insert`).
    made: std::time::Instant,
    /// The flip it was made in (`hostcensus`).
    born: u64,
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
    /// Its buffer's size (CHUNK).
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
    /// The chunk being filled and its use.
    current: Option<(usize, u64)>,
    fills: u64,
    failed: bool,
    /// Takes of write marks that reported pages, counted (`Watched`).
    marks: u64,
    /// Shadow windows whose CPU writes are followed (`watch`), oldest first.
    watched: Vec<Watched>,
    /// Pages of labels whose waits the CPU released (`label_watch`), oldest
    /// first: the page and the count of mark takes (`marks`) when one last
    /// reported it (or a CP, host or mapping change touched it).
    labels: Vec<(u64, u64)>,
    /// Entry ids `drop_range` leaves (`invalidate_ranges_except`).
    spare: Vec<u64>,
    /// Takes of write marks in flight (`take_unlocked`): their pages [lo, hi)
    /// and number, the last number given.
    takes: Vec<(u64, u64, u64)>,
    take_seq: u64,
    /// Why the entries being removed now go (`GONE_*`; diagnostics), and
    /// per key the reason its entry last went and the frame (DS_PROFILE_GPU:
    /// `miss_reason`).
    why: u8,
    gone: Map<(u64, u64), (u8, u64)>,
}

/// Why an entry went (`miss_reason`): unknown, CPU writes on its pages, a
/// failed write-mark query, GPU writes, CP or host writes or a mapping
/// change, eviction, replaced by a new copy; and the misses with an entry:
/// its batch not submitted yet, present; and never kept / too small.
pub const GONE_NAMES: [&str; 10] = ["never kept", "CPU writes", "failed write query", "GPU writes", "CP/host writes or remaps", "eviction", "replaced", "copy not submitted yet", "present", "too small to share"];

/// A written-buffer shadow's window [lo, hi): the count of mark takes
/// (`Cache::marks`) when one last reported a page in it.
struct Watched {
    lo: u64,
    hi: u64,
    written: u64,
}

/// Windows followed at most (the oldest go: unknown then, not kept).
const WATCHED_MAX: usize = 512;
/// Label pages followed at most (`Cache::labels`).
const LABELS_MAX: usize = 64;
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

/// `DS_TRACE_PRECHECK=<from flip>+<n>`: the first n write-mark takes of
/// lookups and inserts from that flip on, to prechecks.txt in the run
/// directory (kind L/I, epoch, base, size, pages found, thread; diagnostics).
static PRECHECK: Mutex<Option<(u64, u64, std::io::BufWriter<std::fs::File>)>> = Mutex::new(None);
static PRECHECK_ON: std::sync::OnceLock<Option<(u64, u64)>> = std::sync::OnceLock::new();

fn trace_take(kind: char, epoch: u64, base: u64, size: u64, found: Option<usize>) {
    let Some((from, n)) = *PRECHECK_ON.get_or_init(|| {
        let v = std::env::var("DS_TRACE_PRECHECK").ok()?;
        let (a, b) = v.split_once(['+', ','])?;
        Some((a.trim().parse().ok()?, b.trim().parse().ok()?))
    }) else {
        return;
    };
    if super::FRAMES.load(Ordering::Relaxed) < from {
        return;
    }
    use std::io::Write;
    let mut g = PRECHECK.lock();
    if g.is_none() {
        let Some(dir) = crate::run_dir() else { return };
        let Ok(f) = std::fs::File::create(dir.join("prechecks.txt")) else { return };
        *g = Some((0, n, std::io::BufWriter::new(f)));
    }
    let Some((done, n, w)) = g.as_mut() else { return };
    if *done >= *n {
        return;
    }
    *done += 1;
    let t = std::thread::current();
    let _ = writeln!(w, "{kind} {epoch} {base:#x} {size:#x} {} {}", found.map_or(-1, |k| k as i64), t.name().unwrap_or("?"));
    if *done == *n {
        let _ = w.flush();
    }
}

/// Statistics: [hits, copies, checks (epoch changed), dropped by CPU writes,
/// dropped by GPU/CP/host writes and mapping changes, dropped by eviction,
/// no space, failed checks, not kept (GPU-written lately), left to a batch
/// not submitted yet (`tune pre_keep`), three unused (the big copies'
/// until 2026-11-01), checks that waited for another thread's take].
static STATS: [AtomicU64; 14] = [const { AtomicU64::new(0) }; 14];
/// Bumped (under the lock) whenever a take made without it ends
/// (`wait_takes` waits for one).
static TAKES_ENDED: AtomicU64 = AtomicU64::new(0);
/// STATS at the last `profile reset` (`window_line`).
static SNAP: [AtomicU64; 14] = [const { AtomicU64::new(0) }; 14];

/// Starts a profile window (`profile reset`).
pub fn reset_window() {
    for (s, v) in SNAP.iter().zip(STATS.iter()).chain(ENDS_SNAP.iter().zip(ENDS.iter())) {
        s.store(v.load(Ordering::Relaxed), Ordering::Relaxed);
    }
}

/// The shared copies since the last `profile reset`.
pub fn window_line() -> String {
    let d: Vec<u64> = STATS.iter().zip(SNAP.iter()).map(|(v, s)| v.load(Ordering::Relaxed).saturating_sub(s.load(Ordering::Relaxed))).collect();
    let e: Vec<u64> = ENDS.iter().zip(ENDS_SNAP.iter()).map(|(v, s)| v.load(Ordering::Relaxed).saturating_sub(s.load(Ordering::Relaxed))).collect();
    format!(
        "shared: {} copies, {} reused ({} checked for CPU writes in a new epoch, {} of them waited for another thread's take; entries dropped by CPU writes {}), {} without space, {} left to a batch not submitted yet, entries dropped by eviction {}; big copies {}, dropped by writes {}, declined {}; epochs ended by {} title submissions, {} waits the CPU released, {} REWIND; {} used without a take since boot (`tune pre_trust`)",
        d[1], d[0], d[2], d[13], d[3], d[6], d[9], d[5], d[10], d[11], d[12], e[0], e[1], e[2], TRUSTED.load(Ordering::Relaxed)
    )
}

/// Epochs by reason: [title submission, CPU-released wait, REWIND].
static ENDS: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
/// ENDS at the last `profile reset` (`window_line`).
static ENDS_SNAP: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// Epochs ended since boot by reason (`ENDS`; front end 2's status).
pub fn epoch_ends() -> [u64; 3] {
    [0, 1, 2].map(|i| ENDS[i].load(Ordering::Relaxed))
}

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
        if super::exec::profiling_on() {
            if self.gone.len() > 1 << 16 {
                self.gone.clear();
            }
            self.gone.insert(key, (self.why, super::FRAMES.load(Ordering::Relaxed)));
        }
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

    /// Drops the entries overlapping [base, end); how many.
    fn drop_range(&mut self, base: u64, end: u64) -> u64 {
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

    /// What a take of [base, base+size)'s write marks found (`r`, the pages
    /// `new`), applied: the pages date the watched windows they fall in (a
    /// failed query dates those overlapping the range), texture regions on
    /// them other than `except` are told (`track::cpu_pages`: taking the
    /// marks resets them), and so are write-watched homes on them unless the
    /// take is a home's own check (`homes`).
    fn apply_marks(&mut self, base: u64, size: u64, r: Option<()>, new: &[u64], except: Option<u32>, homes: bool) {
        // Front end 2's pages the take found written leave the mirror.
        super::fe2::marks_taken(r, base, size, new);
        match r {
            Some(()) if !new.is_empty() => super::track::cpu_pages(new, except),
            Some(()) => {}
            None => super::track::cpu_range(base, size, except),
        }
        if homes {
            match r {
                Some(()) if !new.is_empty() => super::homes::marks_taken(Some(new), base, size),
                Some(()) => {}
                None => super::homes::marks_taken(None, base, size),
            }
        }
        if !self.labels.is_empty() && (r.is_none() || !new.is_empty()) {
            let end = base.saturating_add(size);
            for i in 0..self.labels.len() {
                let p = self.labels[i].0;
                if p + 4096 > base && p < end && (r.is_none() || new.contains(&p)) {
                    self.marks += 1;
                    self.labels[i].1 = self.marks;
                }
            }
        }
        if self.watched.is_empty() {
            return;
        }
        match r {
            Some(()) if !new.is_empty() => {
                self.marks += 1;
                let g = self.marks;
                for &p in new {
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
    }

    /// Registers a take of [base, base+size)'s marks about to be made
    /// without the lock; its number.
    fn begin_take(&mut self, base: u64, size: u64) -> u64 {
        self.take_seq += 1;
        self.takes.push((base & !4095, base.saturating_add(size).next_multiple_of(4096), self.take_seq));
        self.take_seq
    }

    /// Whether a take numbered up to `upto` is in flight over a page of
    /// [base, base+size).
    fn takes_over(&self, base: u64, size: u64, upto: u64) -> bool {
        let (lo, hi) = (base & !4095, base.saturating_add(size).next_multiple_of(4096));
        self.takes.iter().any(|&(a, b, t)| t <= upto && a < hi && lo < b)
    }

    /// Entries on the pages a take found written go (all of the range when
    /// the query failed).
    fn dropped_by_take(&mut self, base: u64, size: u64, r: Option<()>, pages: &[u64]) {
        match r {
            None => {
                STATS[7].fetch_add(1, Ordering::Relaxed);
                self.why = 2;
                self.drop_range(base, base.saturating_add(size));
                self.bounds_changed();
            }
            Some(()) if !pages.is_empty() => {
                self.why = 1;
                let n = self.drop_pages(pages);
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
        self.why = 5;
        for key in std::mem::take(&mut self.chunks[k].keys) {
            if self.entries.get(&key).is_some_and(|e| Arc::ptr_eq(&e.chunk, &victim)) {
                self.remove(key);
                STATS[5].fetch_add(1, Ordering::Relaxed);
            }
        }
        drop(victim);
        Arc::strong_count(&self.chunks[k].chunk) == 1
    }

    /// Space for `n` bytes (256-byte aligned) in the chunk being filled, a
    /// free one, a new one, or (all in use) the chunk filled longest ago once
    /// nothing binds it (its entries go now). None over CHUNK.
    fn alloc(&mut self, n: u64) -> Option<(usize, Arc<Chunk>, u64)> {
        let n = n.next_multiple_of(256);
        if n > CHUNK {
            return None;
        }
        if let Some((k, used)) = &mut self.current {
            if *used + n <= CHUNK {
                let off = *used;
                *used += n;
                return Some((*k, self.chunks[*k].chunk.clone(), off));
            }
        }
        // At most a sixteenth of the device's budget (`tune pre_share_mib`,
        // 4 GiB: ~1.44 GiB of a 24 GB GPU's 23 GB budget, ~448 MiB of an 8 GB
        // one; 1 GiB left a quarter of the vista's shareable copies without
        // space, FINDINGS.md 2026-10-27).
        let mib = super::vk::budget_share(16, super::tune::pre_share_mib() as u64 * (1 << 20));
        let max = (mib / CHUNK).max(2) as usize;
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

type Guard = parking_lot::MutexGuard<'static, Option<Cache>>;

fn lock() -> Guard {
    super::lockstat::COPY_CACHE.lock(&CACHE)
}

/// Takes [base, base+size)'s write marks into `out` without holding the
/// cache's lock during the query: registered as in flight first, applied
/// after (`Cache::apply_marks`, texture regions other than `except` told).
/// Returns the lock, held, the query's result and the last take number
/// given when it was applied (`wait_takes`).
fn take_unlocked(base: u64, size: u64, out: &mut Vec<u64>, except: Option<u32>, who: usize) -> (Guard, Option<()>, u64) {
    if !super::tune::marks_unlocked() {
        // Under the lock: no take is ever in flight.
        let mut g = lock();
        let c = g.get_or_insert_with(Cache::default);
        let first = out.len();
        let r = crate::mem::take_written(base, size, out, who);
        c.apply_marks(base, size, r, &out[first..], except, who != 2);
        let upto = c.take_seq;
        return (g, r, upto);
    }
    let token = lock().get_or_insert_with(Cache::default).begin_take(base, size);
    let first = out.len();
    let r = crate::mem::take_written(base, size, out, who);
    let mut g = lock();
    let c = g.get_or_insert_with(Cache::default);
    c.takes.retain(|t| t.2 != token);
    TAKES_ENDED.fetch_add(1, Ordering::AcqRel);
    c.apply_marks(base, size, r, &out[first..], except, who != 2);
    let upto = c.take_seq;
    (g, r, upto)
}

/// Waits, with the lock released, until no take numbered up to `upto` is
/// in flight over [base, base+size): such a take may have reset marks of
/// pages written before; its pages are applied once it ends.
fn wait_takes(mut g: Guard, base: u64, size: u64, upto: u64) -> Guard {
    let mut n = 0u32;
    while g.as_ref().is_some_and(|c| c.takes_over(base, size, upto)) {
        // The lock again only once a take ended (until 2026-10-28 every spin
        // took it: front end 2's threads taking one pool at once made ~1.2
        // million lock waits a run). (Read before the lock goes: a take
        // ends under it.)
        let seen = TAKES_ENDED.load(Ordering::Acquire);
        drop(g);
        let mut k = 0u32;
        while TAKES_ENDED.load(Ordering::Acquire) == seen {
            k += 1;
            if k < 64 {
                std::hint::spin_loop();
            } else {
                std::thread::yield_now();
            }
        }
        n += 1;
        g = lock();
    }
    if n > 0 {
        STATS[13].fetch_add(1, Ordering::Relaxed);
    }
    g
}

/// Lookups served without a take by `tune pre_trust` (diagnosis).
pub static TRUSTED: AtomicU64 = AtomicU64::new(0);

/// A submitted copy of guest [base, base+size) that is still current:
/// (chunk, offset, the copy queue's timeline value a batch binding it waits
/// for: the sequence of the batch that copied it there, else 0). Checks it
/// for CPU writes once per epoch.
pub fn lookup(base: u64, size: u64) -> Option<(Arc<Chunk>, u64, u64)> {
    let epoch = EPOCH.load(Ordering::Acquire);
    let id = {
        let mut g = lock();
        let c = g.as_mut()?;
        let e = c.entries.get_mut(&(base, size))?;
        if e.seq == 0 {
            return None;
        }
        if e.checked == epoch {
            STATS[0].fetch_add(1, Ordering::Relaxed);
            let r = (e.chunk.clone(), e.offset, if e.on_copy { e.seq } else { 0 });
            let born = e.born;
            drop(g);
            crate::hostcensus::copy_used(base, size, born);
            return Some(r);
        }
        let trust = super::tune::pre_trust();
        if trust > 0 && e.clean_flip.saturating_add(trust) >= super::FRAMES.load(Ordering::Relaxed) {
            STATS[0].fetch_add(1, Ordering::Relaxed);
            TRUSTED.fetch_add(1, Ordering::Relaxed);
            return Some((e.chunk.clone(), e.offset, if e.on_copy { e.seq } else { 0 }));
        }
        e.id
    };
    STATS[2].fetch_add(1, Ordering::Relaxed);
    let (tb, te) = (base, base + size);
    let mut pages = Vec::new();
    // `tune host_census`: takes of copies armed as static data are what
    // write tracking by protection would leave out.
    let armed = crate::hostcensus::static_live(base, size).then(std::time::Instant::now);
    let (mut g, r, upto) = take_unlocked(tb, te - tb, &mut pages, None, 0);
    if let Some(t) = armed {
        crate::hostcensus::note_armed_take(t.elapsed().as_nanos() as u64, !pages.is_empty());
    }
    trace_take('L', epoch, tb, te - tb, r.map(|_| pages.len()));
    {
        let c = g.as_mut()?;
        match r {
            None => {
                STATS[7].fetch_add(1, Ordering::Relaxed);
                c.why = 2;
                if c.entries.get(&(base, size)).is_some_and(|e| e.id == id) {
                    c.remove((base, size));
                }
                c.bounds_changed();
                return None;
            }
            Some(()) if !pages.is_empty() => {
                c.why = 1;
                let n = c.drop_pages(&pages);
                STATS[3].fetch_add(n, Ordering::Relaxed);
                c.bounds_changed();
                // (Written pages of a window may lie outside the range.)
                if !c.entries.get(&(base, size)).is_some_and(|e| e.id == id) {
                    return None;
                }
            }
            Some(()) => {}
        }
    }
    let mut g = wait_takes(g, tb, te - tb, upto);
    let flip = super::FRAMES.load(Ordering::Relaxed);
    let c = g.as_mut()?;
    let e = c.entries.get_mut(&(base, size)).filter(|e| e.id == id)?;
    e.checked = epoch;
    e.clean_flip = flip;
    STATS[0].fetch_add(1, Ordering::Relaxed);
    Some((e.chunk.clone(), e.offset, if e.on_copy { e.seq } else { 0 }))
}

/// Why the range has no usable copy (`GONE_NAMES` index; diagnostics).
pub fn miss_reason(base: u64, size: u64) -> u8 {
    let g = super::lockstat::COPY_CACHE.lock(&CACHE);
    let Some(c) = g.as_ref() else { return 0 };
    match c.entries.get(&(base, size)) {
        Some(e) if e.seq == 0 => 7,
        Some(_) => 8,
        None => c.gone.get(&(base, size)).map_or(0, |g| g.0),
    }
}

/// Space for a new copy of [base, base+size) (`n` bytes from its aligned
/// start) and the new entry's id; the entry is usable by other batches once
/// `submitted` names it.
pub fn insert(base: u64, size: u64, n: u64) -> Option<(Arc<Chunk>, u64, u64)> {
    // A batch not submitted yet copies it: this one copies it for itself
    // (replacing the entry wasted that copy and its space).
    let pending = |c: &Cache| super::tune::pre_keep() && c.entries.get(&(base, size)).is_some_and(|e| e.seq == 0 && e.made.elapsed() < std::time::Duration::from_millis(500));
    {
        let mut g = lock();
        let c = g.get_or_insert_with(Cache::default);
        if pending(c) {
            STATS[9].fetch_add(1, Ordering::Relaxed);
            return None;
        }
    }
    // The range's write marks so far: the copy sees those writes (it runs
    // later), other entries on those pages are stale. (A take in flight
    // over it from another thread can only drop the new entry later.)
    let mut pages = Vec::new();
    // Nothing to take when the epoch's block probes found the range clean.
    // The epoch the copy is checked for: read before the take, which covers
    // it (an epoch that ends meanwhile checks the copy again).
    let ep = EPOCH.load(Ordering::Acquire);
    let (mut g, r, _) = take_unlocked(base, size, &mut pages, None, 1);
    trace_take('I', EPOCH.load(Ordering::Relaxed), base, size, r.map(|_| pages.len()));
    let c = g.get_or_insert_with(Cache::default);
    if r.is_none() {
        STATS[7].fetch_add(1, Ordering::Relaxed);
        return None;
    }
    if !pages.is_empty() {
        c.why = 1;
        let k = c.drop_pages(&pages);
        STATS[3].fetch_add(k, Ordering::Relaxed);
    }
    if pending(c) {
        STATS[9].fetch_add(1, Ordering::Relaxed);
        c.bounds_changed();
        return None;
    }
    let Some((slot, chunk, offset)) = c.alloc(n) else {
        STATS[6].fetch_add(1, Ordering::Relaxed);
        c.bounds_changed();
        return None;
    };
    c.chunks[slot].keys.push((base, size));
    STATS[1].fetch_add(1, Ordering::Relaxed);
    c.why = 6;
    c.remove((base, size));
    c.chunks[slot].live += 1;
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    let flip = super::FRAMES.load(Ordering::Relaxed);
    c.entries.insert((base, size), Entry { chunk: chunk.clone(), offset, id, seq: 0, on_copy: false, checked: ep, clean_flip: flip, slot, made: std::time::Instant::now(), born: flip });
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
/// entry id, made on the copy queue)) was submitted: those entries, if still
/// there, are usable.
pub fn submitted(made: &[(u64, u64, u64, bool)], seq: u64) {
    if made.is_empty() {
        return;
    }
    let mut g = super::lockstat::COPY_CACHE.lock(&CACHE);
    let Some(c) = g.as_mut() else { return };
    for &(base, size, id, on_copy) in made {
        if let Some(e) = c.entries.get_mut(&(base, size)) {
            if e.id == id && e.seq == 0 {
                e.seq = seq;
                e.on_copy = on_copy;
            }
        }
    }
}

/// The pages of guest [base, base+size) the CPU wrote since their write
/// marks were last taken, into `out` (texture re-uploads; None: unknown,
/// treat it all as written). The marks are taken under the cache's lock like
/// its own takes: entries on those pages go, watched windows are dated, and
/// texture regions on them other than `except` (the caller's) are told.
/// `who`: the caller (`mem::WW_WHO_NAMES`; statistics).
pub fn take_written_for(base: u64, size: u64, out: &mut Vec<u64>, except: Option<u32>, who: usize) -> Option<()> {
    let first = out.len();
    let (mut g, r, upto) = take_unlocked(base, size, out, except, who);
    g.get_or_insert_with(Cache::default).dropped_by_take(base, size, r, &out[first..]);
    // Pages another take over the range reset are told to the texture
    // regions on them once it ends (as when it came first under one lock).
    drop(wait_takes(g, base, size, upto));
    r
}

/// Guest [base, base+len) may have changed (CP, host and mapping changes):
/// copies overlapping it go, and watched shadow windows it overlaps are dated
/// (file reads by DMA leave no write marks).
#[track_caller]
pub fn invalidate(base: u64, len: u64) {
    super::fe2::invalidate(base, len);
    let end = base.saturating_add(len);
    if len != 0 && base < LABEL_HI.load(Ordering::Acquire) && LABEL_LO.load(Ordering::Acquire) < end {
        let mut g = super::lockstat::COPY_CACHE.lock(&CACHE);
        if let Some(c) = g.as_mut() {
            for i in 0..c.labels.len() {
                let p = c.labels[i].0;
                if p + 4096 > base && p < end {
                    c.marks += 1;
                    c.labels[i].1 = c.marks;
                }
            }
        }
    }
    if len != 0 && base < WATCH_HI.load(Ordering::Acquire) && WATCH_LO.load(Ordering::Acquire) < end {
        let mut g = super::lockstat::COPY_CACHE.lock(&CACHE);
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

/// GPU work writes guest [base, base+len): copies overlapping it go.
pub fn invalidate_gpu(base: u64, len: u64) {
    drop_overlapping(base, len, true);
}

fn drop_overlapping(base: u64, len: u64, gpu: bool) {
    if len == 0 || base.saturating_add(len) <= LO.load(Ordering::Acquire) || HI.load(Ordering::Acquire) <= base || !may_cover(base, len) {
        return;
    }
    let mut g = super::lockstat::COPY_CACHE.lock(&CACHE);
    let Some(c) = g.as_mut() else { return };
    c.why = if gpu { 3 } else { 4 };
    let n = c.drop_range(base, base.saturating_add(len));
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
    let mut g = super::lockstat::COPY_CACHE.lock(&CACHE);
    let Some(c) = g.as_mut() else { return };
    c.spare = keep.to_vec();
    c.why = 3;
    for (a, n) in writes.iter() {
        c.drop_range(a, a.saturating_add(n));
    }
    c.spare.clear();
    c.bounds_changed();
}

/// Starts following CPU writes of a shadow's window [base, base+size) whose
/// copy in is recorded now: its marks so far are taken (the copy runs after
/// those writes) and the moment returned (`unwritten_since`).
pub fn watch(base: u64, size: u64) -> u64 {
    let mut pages = Vec::new();
    let (mut g, r, _) = take_unlocked(base, size, &mut pages, None, 6);
    let c = g.get_or_insert_with(Cache::default);
    c.dropped_by_take(base, size, r, &pages);
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

/// Bounds of the label pages followed, readable without the lock
/// (`invalidate`; LO > HI: none; they only grow).
static LABEL_LO: AtomicU64 = AtomicU64::new(u64::MAX);
static LABEL_HI: AtomicU64 = AtomicU64::new(0);

/// Front end 2 (lagged views): follows CPU writes of the page holding
/// the label at `a` from now on: its marks so far are taken (the writes
/// they report came before the epoch the caller ends next) and the moment
/// returned (`label_unwritten_since`).
pub fn label_watch(a: u64) -> u64 {
    let page = a & !4095;
    let mut pages = Vec::new();
    let (mut g, r, _) = take_unlocked(page, 4096, &mut pages, None, 7);
    let c = g.get_or_insert_with(Cache::default);
    c.dropped_by_take(page, 4096, r, &pages);
    if !c.labels.iter().any(|l| l.0 == page) {
        if c.labels.len() >= LABELS_MAX {
            c.labels.remove(0);
        }
        // (Dated now: a pass that watched it before it was dropped may have
        // missed writes since, and must not take it as unwritten.)
        c.marks += 1;
        let m = c.marks;
        c.labels.push((page, m));
        LABEL_LO.fetch_min(page, Ordering::AcqRel);
        LABEL_HI.fetch_max(page + 4096, Ordering::AcqRel);
    }
    c.marks
}

/// Whether the CPU wrote the page holding the label at `a` since the moment
/// `since` (`label_watch`): a take now, after any take over it in flight
/// ends; false when unknown (the page is no longer followed).
pub fn label_unwritten_since(a: u64, since: u64) -> bool {
    let page = a & !4095;
    let mut pages = Vec::new();
    let (mut g, r, upto) = take_unlocked(page, 4096, &mut pages, None, 7);
    g.get_or_insert_with(Cache::default).dropped_by_take(page, 4096, r, &pages);
    let g = wait_takes(g, page, 4096, upto);
    let Some(c) = g.as_ref() else { return false };
    c.labels.iter().find(|l| l.0 == page).is_some_and(|l| l.1 <= since)
}

/// Whether the CPU wrote no page of the window [base, base+size) since the
/// moment `since` (`watch`); false when unknown (the window's watch was
/// dropped).
pub fn unwritten_since(base: u64, size: u64, since: u64) -> bool {
    if lock().is_none() {
        return false;
    }
    let mut pages = Vec::new();
    let (mut g, r, upto) = take_unlocked(base, size, &mut pages, None, 6);
    g.get_or_insert_with(Cache::default).dropped_by_take(base, size, r, &pages);
    let g = wait_takes(g, base, size, upto);
    let Some(c) = g.as_ref() else { return false };
    let (lo, hi) = (base & !4095, base.saturating_add(size).next_multiple_of(4096));
    c.watched.iter().find(|w| w.lo == lo && w.hi == hi).is_some_and(|w| w.written <= since)
}

/// The current synchronization epoch (write-watched homes check their pages
/// once per epoch too).
pub fn epoch() -> u64 {
    EPOCH.load(Ordering::Acquire)
}

/// Epochs ended by title submissions and REWIND only (not by waits the CPU
/// released): what write-watched homes check their pages once per with
/// `tune home_submit_epochs`.
static SUBMIT_EPOCH: AtomicU64 = AtomicU64::new(0);

pub fn submit_epoch() -> u64 {
    SUBMIT_EPOCH.load(Ordering::Acquire)
}

/// A synchronization epoch ends (a title submission starts, a wait the CPU
/// released, REWIND; any queue): entries are checked for CPU writes again at
/// their next use.
pub fn end_scope(why: usize) {
    ENDS[why.min(2)].fetch_add(1, Ordering::Relaxed);
    if why != 1 {
        SUBMIT_EPOCH.fetch_add(1, Ordering::AcqRel);
    }
    EPOCH.fetch_add(1, Ordering::AcqRel);
}

/// One status line ("" before any use).
pub fn status() -> String {
    let s: Vec<u64> = STATS.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    if s[1] == 0 {
        return String::new();
    }
    let (chunks, entries) = super::lockstat::COPY_CACHE.lock(&CACHE).as_ref().map_or((0, 0), |c| (c.chunks.len(), c.entries.len()));
    format!(
        "    shared batch-start copies: {} copies, {} reused, {} not kept (GPU-written lately); {} checks for CPU writes ({} failed, {} waited for another thread's take), entries dropped by CPU writes {}, by GPU/CP/host writes and mapping changes {}, by eviction {}; {} without space, {} left to a batch not submitted yet; {} entries in {} chunks of {} MiB; epochs: {} submissions, {} waits the CPU released, {} REWIND ({} waits passed on GPU writes ended none)\n",
        s[1],
        s[0],
        s[8],
        s[2],
        s[7],
        s[13],
        s[3],
        s[4],
        s[5],
        s[6],
        s[9],
        entries,
        chunks,
        CHUNK >> 20,
        ENDS[0].load(Ordering::Relaxed),
        ENDS[1].load(Ordering::Relaxed),
        ENDS[2].load(Ordering::Relaxed),
        super::cp::EPOCH_GPU_WAITS.load(Ordering::Relaxed)
    )
}
