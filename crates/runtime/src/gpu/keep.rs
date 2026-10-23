//! Small GPU-written windows kept in video memory across batches (`tune
//! keep_small`; FINDINGS.md 2026-10-23).
//!
//! A dispatch's written buffer binding small enough for a VRAM shadow (the
//! compute queues' counter and per-object blocks: ~1,900 windows at the
//! gate, ~200 write-backs a frame) used to be copied in and merged back into
//! guest memory at the end of every batch that wrote it. Storing into those
//! guest pages cost ~13% of a gate frame's GPU time (~20 us each; the same
//! stores into memory nothing else uses cost next to nothing), and 91% of the
//! merges were of windows no CPU-side reader, CP write or fill ever touches.
//!
//! A kept window lives in a pool of video memory: a working copy, which every
//! binding inside the window binds (any queue, written or read-only, and GPU-
//! read indirect dispatch arguments), and a pristine copy, what guest memory
//! and the working copy last agreed on. Each batch's first use takes in what
//! the CPU wrote since (`rdna::kernels::keep_sync` with guest memory as the
//! source: dwords that differ from the pristine copy go to both copies); a
//! write-back (the same kernel the other way: dwords the GPU changed go to
//! guest memory and the pristine copy) happens only when something reads
//! guest memory there: a CPU-side read (`Exec::sync_for`: command-processor
//! reads, shader preparation, textures), which also makes the window eager
//! (written back at the end of every batch that writes it, as shadows were);
//! a binding that overlaps it without being inside (then it is dropped); a CP
//! write or fill into it; the switch turned off; eviction. Guest memory of a
//! window that is not written back is stale: every reader of guest memory
//! either binds the working copy or forces the write-back.
//!
//! Dirty: a window's last writing batch and its last write-back batch, by
//! sequence (submission order is sequence order, and every batch follows
//! the earlier ones on the one Vulkan queue): a write-back covers the writes
//! of batches before it and its own batch's. A reader on the CPU side that
//! depends on a write waits for it by the title's synchronization, which
//! orders its recording after the writer's (submitted or promised): it
//! finds the window dirty and writes it back. Recordings made before a
//! window was kept read guest memory; the title's synchronization orders any
//! that must see a kept write after it, so they find the window.

use super::vk::ctx;
use ash::vk;
use parking_lot::{Mutex, RwLock};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

/// Largest window kept.
pub const MAX: u64 = 256 << 10;
/// Windows start and end on multiples of this (storage buffer offsets: the
/// takes and write-backs bind guest memory at the window's start). Wider
/// granules merged neighbouring blocks into windows the next object's
/// bindings straddled (20261023-093530: 12,116 windows made, 34,438
/// write-backs for overlaps).
pub const GRAIN: u64 = 16;
/// Frames a window another queue wrote, or one dropped for overlaps
/// without a usable union, is not kept again.
const REFUSE_FLIPS: u64 = 600;
/// Video memory for all kept windows (working and pristine copies).
const BUDGET: u64 = 256 << 20;
const CHUNK: u64 = 16 << 20;
/// Bloom filter over 64 KiB blocks holding windows (bindings elsewhere skip
/// the registry).
const BLOOM_WORDS: usize = 4096;

pub struct Chunk {
    pub buf: vk::Buffer,
    mem: vk::DeviceMemory,
}

impl Drop for Chunk {
    fn drop(&mut self) {
        super::vk::note_gpu_freed(self.buf);
        unsafe {
            ctx().device.destroy_buffer(self.buf, None);
            ctx().device.free_memory(self.mem, None);
        }
    }
}

/// A window's video memory: its working copy at `off`, its pristine copy
/// `class / 2` bytes later. Returned to the pool when the registry and every
/// batch that bound it have let it go.
pub struct Slot {
    pub chunk: Arc<Chunk>,
    pub off: u64,
    index: usize,
    class: u64,
}

impl Drop for Slot {
    fn drop(&mut self) {
        POOL.lock().free.entry(self.class).or_default().push((self.index, self.off));
    }
}

#[derive(Default)]
struct Pool {
    chunks: Vec<Arc<Chunk>>,
    /// The chunk being filled and its next free offset.
    bump: Option<(usize, u64)>,
    /// Free slots by class (bytes for both copies).
    free: BTreeMap<u64, Vec<(usize, u64)>>,
}

static POOL: Mutex<Pool> = Mutex::new(Pool { chunks: Vec::new(), bump: None, free: BTreeMap::new() });

/// Bytes of each copy of a window of `size` bytes.
pub fn span(size: u64) -> u64 {
    size.next_multiple_of(GRAIN)
}

fn class(size: u64) -> u64 {
    (2 * span(size)).next_power_of_two().max(512)
}

fn alloc(size: u64) -> Option<Arc<Slot>> {
    let class = class(size);
    let mut p = POOL.lock();
    if let Some((index, off)) = p.free.get_mut(&class).and_then(|v| v.pop()) {
        let chunk = p.chunks[index].clone();
        return Some(Arc::new(Slot { chunk, off, index, class }));
    }
    if let Some((index, next)) = p.bump {
        let off = next.next_multiple_of(class);
        if off + class <= CHUNK {
            p.bump = Some((index, off + class));
            let chunk = p.chunks[index].clone();
            return Some(Arc::new(Slot { chunk, off, index, class }));
        }
    }
    if (p.chunks.len() as u64 + 1) * CHUNK > BUDGET {
        return None;
    }
    let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDIRECT_BUFFER | vk::BufferUsageFlags::INDEX_BUFFER;
    let (buf, mem) = ctx().device_buffer(CHUNK, usage).ok()?;
    p.chunks.push(Arc::new(Chunk { buf, mem }));
    let index = p.chunks.len() - 1;
    p.bump = Some((index, class));
    let chunk = p.chunks[index].clone();
    Some(Arc::new(Slot { chunk, off: 0, index, class }))
}

pub struct Kept {
    pub base: u64,
    pub size: u64,
    pub slot: Arc<Slot>,
    pub id: u64,
    /// The queue that made it (the only one that writes or drops it).
    pub owner: String,
    /// The last submitted batch that wrote it.
    write_seq: AtomicU64,
    /// The last submitted batch that wrote it back.
    given_seq: AtomicU64,
    /// Written back at the end of every batch that writes it (CPU-side
    /// readers seen).
    pub eager: AtomicBool,
    /// The flip it was last bound in (eviction order).
    last_flip: AtomicU64,
    /// Another queue wrote it: its owner writes it back and drops it at its
    /// next use.
    pub retire: AtomicBool,
}

impl Kept {
    pub fn buf(&self) -> vk::Buffer {
        self.slot.chunk.buf
    }
    /// Offset of the working copy (guest `base`) in `buf`.
    pub fn work(&self) -> u64 {
        self.slot.off
    }
    pub fn pristine(&self) -> u64 {
        self.slot.off + self.slot.class / 2
    }
    /// GPU writes of submitted batches not written back yet.
    pub fn dirty(&self) -> bool {
        self.write_seq.load(Ordering::Acquire) > self.given_seq.load(Ordering::Acquire)
    }
    pub fn contains(&self, a: u64, n: u64) -> bool {
        self.base <= a && a + n <= self.base + self.size
    }
    pub fn overlaps(&self, a: u64, n: u64) -> bool {
        a < self.base + self.size && self.base < a.saturating_add(n)
    }
}

static KEPT: RwLock<BTreeMap<u64, Arc<Kept>>> = RwLock::new(BTreeMap::new());
static ANY: AtomicBool = AtomicBool::new(false);
static BLOOM: [AtomicU64; BLOOM_WORDS] = [const { AtomicU64::new(0) }; BLOOM_WORDS];
static NEXT_ID: AtomicU64 = AtomicU64::new(1);
/// Bumped when a batch that wrote a kept window is submitted (the
/// command processors' clean-page memo holds while it stays the same).
pub static DIRTY_GEN: AtomicU64 = AtomicU64::new(0);
/// [created, dropped, takes, write-backs on CPU-side reads, eager write-backs,
/// write-backs for overlapping bindings, write-backs for CP writes and fills,
/// evicted, bindings served, indirect arguments served, creations refused
/// (no space), switch-off write-backs, ranges refused, write-backs for
/// other queues' writes]
pub static STATS: [AtomicU64; 14] = [const { AtomicU64::new(0) }; 14];

fn bloom_bit(block: u64) -> (usize, u64) {
    let h = block.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 46;
    ((h as usize / 64) % BLOOM_WORDS, 1u64 << (h % 64))
}

/// Whether a kept window may overlap [a, a+n) (false: certainly none).
fn maybe(a: u64, n: u64) -> bool {
    if !ANY.load(Ordering::Acquire) || n == 0 {
        return false;
    }
    let (first, last) = (a >> 16, (a + n - 1) >> 16);
    if last - first > 64 {
        return true;
    }
    (first..=last).any(|b| {
        let (w, bit) = bloom_bit(b);
        BLOOM[w].load(Ordering::Relaxed) & bit != 0
    })
}

pub fn any() -> bool {
    ANY.load(Ordering::Acquire)
}

pub enum Found {
    /// The window holding all of the range.
    Inside(Arc<Kept>),
    /// Windows the range overlaps without being inside one.
    Overlap(Vec<Arc<Kept>>),
}

/// The kept windows [a, a+n) touches.
pub fn find(a: u64, n: u64) -> Option<Found> {
    if !maybe(a, n) {
        return None;
    }
    let g = KEPT.read();
    let end = a.saturating_add(n);
    let mut hits: Vec<Arc<Kept>> = Vec::new();
    // Windows are at most MAX long and do not overlap each other.
    for (_, k) in g.range(a.saturating_sub(MAX)..end) {
        if k.overlaps(a, n) {
            hits.push(k.clone());
        }
    }
    match hits.len() {
        0 => None,
        1 if hits[0].contains(a, n) => Some(Found::Inside(hits.pop().unwrap())),
        _ => Some(Found::Overlap(hits)),
    }
}

/// The kept window holding address `a` (unbounded pointers: only the one
/// their base lies in counts).
pub fn at(a: u64) -> Option<Arc<Kept>> {
    if !maybe(a, 4) {
        return None;
    }
    let g = KEPT.read();
    g.range(..=a).next_back().map(|(_, k)| k).filter(|k| k.overlaps(a, 4)).cloned()
}

/// Shadow windows merged, by window: (the last frame, frames seen in).
/// Only windows that live across frames are kept (ring allocations, a new
/// window each frame, made and dropped windows every frame:
/// 20261023-094729).
static MERGED: Mutex<Option<std::collections::HashMap<(u64, u64), (u64, u32)>>> = Mutex::new(None);

/// A window's key: its start and end widened to GRAIN.
fn key(a: u64, n: u64) -> (u64, u64) {
    (a & !(GRAIN - 1), (a + n).next_multiple_of(GRAIN))
}

/// A small shadow of [a, a+n) was merged back (any switch setting).
pub fn note_merged(a: u64, n: u64) {
    if n > MAX {
        return;
    }
    let frame = super::FRAMES.load(Ordering::Relaxed);
    let mut g = MERGED.lock();
    let m = g.get_or_insert_with(Default::default);
    if m.len() > 1 << 16 {
        m.retain(|_, v| v.0 + 8 > frame);
    }
    let e = m.entry(key(a, n)).or_insert((frame, 0));
    if e.1 == 0 || e.0 != frame {
        e.1 = e.1.saturating_add(1);
    }
    e.0 = frame;
}

/// Whether a window for [a, a+n) lives across frames: merged in two
/// frames or more, the last within 8.
pub fn lasting(a: u64, n: u64) -> bool {
    let frame = super::FRAMES.load(Ordering::Relaxed);
    MERGED.lock().as_ref().and_then(|m| m.get(&key(a, n)).copied()).is_some_and(|(last, seen)| seen >= 2 && last + 8 > frame)
}

/// Whether a kept window holds all of [a, a+n).
pub fn holds(a: u64, n: u64) -> bool {
    matches!(find(a, n), Some(Found::Inside(_)))
}

/// The dirty kept windows overlapping any of `ranges`.
pub fn dirty_in(ranges: &[(u64, u64)]) -> Vec<Arc<Kept>> {
    let mut out: Vec<Arc<Kept>> = Vec::new();
    for &(a, n) in ranges {
        match find(a, n) {
            Some(Found::Inside(k)) if k.dirty() => {
                if !out.iter().any(|o| o.id == k.id) {
                    out.push(k);
                }
            }
            Some(Found::Overlap(ks)) => {
                for k in ks.into_iter().filter(|k| k.dirty()) {
                    if !out.iter().any(|o| o.id == k.id) {
                        out.push(k);
                    }
                }
            }
            _ => {}
        }
    }
    out
}

/// Whether any of `ranges` overlaps a dirty kept window.
pub fn any_dirty(ranges: &[(u64, u64)]) -> bool {
    ranges.iter().any(|&(a, n)| maybe(a, n)) && !dirty_in(ranges).is_empty()
}

/// Why a window was not made.
#[derive(Debug, PartialEq, Eq)]
pub enum Refused {
    Size,
    Overlap,
    Space,
}

/// Extents bindings of one object were seen with (unions of overlapping
/// bindings and windows): a window made inside one takes all of it.
static HINTS: Mutex<BTreeMap<u64, u64>> = Mutex::new(BTreeMap::new());
/// Ranges not kept (until a flip): written by two queues, or overlaps no
/// union settled.
static REFUSED: Mutex<BTreeMap<u64, (u64, u64)>> = Mutex::new(BTreeMap::new());

/// Bindings and windows over [base, end) belong together: later windows
/// there take all of it (merged with hints it overlaps while that stays
/// within MAX; else the range is not kept for a while).
pub fn hint(base: u64, end: u64) {
    let mut g = HINTS.lock();
    let (mut b, mut e) = (base, end);
    let over: Vec<u64> = g.range(..end).filter(|(_, he)| **he > base).map(|(hb, _)| *hb).collect();
    for hb in over {
        let he = g.remove(&hb).unwrap();
        b = b.min(hb);
        e = e.max(he);
    }
    if e - b <= MAX {
        g.insert(b, e);
    } else {
        drop(g);
        refuse(b, e);
    }
}

/// [base, end) is not kept for REFUSE_FLIPS frames.
pub fn refuse(base: u64, end: u64) {
    let now = super::FRAMES.load(Ordering::Relaxed);
    let mut g = REFUSED.lock();
    if g.len() > 4096 {
        g.retain(|_, v| v.1 > now);
    }
    g.insert(base, (end, now + REFUSE_FLIPS));
    STATS[12].fetch_add(1, Ordering::Relaxed);
}

fn refused(base: u64, end: u64) -> bool {
    let now = super::FRAMES.load(Ordering::Relaxed);
    REFUSED.lock().range(..end).any(|(rb, v)| v.0 > base && *rb < end && v.1 > now)
}

/// The window kept for `owner`'s written binding of [a, a+n): [a, a+n)
/// widened to GRAIN (or to the hint holding it), within guest memory
/// imported as one buffer from `mapped_start` to `mapped_end`.
pub fn create(a: u64, n: u64, mapped_start: u64, mapped_end: u64, owner: &str) -> Result<Arc<Kept>, Refused> {
    if ctx().limits.min_storage_buffer_offset_alignment > GRAIN {
        return Err(Refused::Size);
    }
    let (mut base, mut end) = (a & !(GRAIN - 1), (a + n).next_multiple_of(GRAIN));
    if let Some((hb, he)) = HINTS.lock().range(..=a).next_back().map(|(b, e)| (*b, *e)) {
        if hb <= a && a + n <= he {
            (base, end) = (hb & !(GRAIN - 1), he.next_multiple_of(GRAIN));
        }
    }
    let (base, end) = (base.max(mapped_start), end.min(mapped_end));
    if end <= base || end - base > MAX || a < base || a + n > end {
        return Err(Refused::Size);
    }
    if refused(base, end) {
        return Err(Refused::Overlap);
    }
    let size = end - base;
    let mut g = KEPT.write();
    if g.range(base.saturating_sub(MAX)..end).any(|(_, k)| k.overlaps(base, size)) {
        return Err(Refused::Overlap);
    }
    let Some(slot) = alloc(size) else {
        STATS[10].fetch_add(1, Ordering::Relaxed);
        return Err(Refused::Space);
    };
    let k = Arc::new(Kept {
        base,
        size,
        slot,
        id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
        owner: owner.to_string(),
        write_seq: AtomicU64::new(0),
        given_seq: AtomicU64::new(0),
        eager: AtomicBool::new(false),
        last_flip: AtomicU64::new(super::FRAMES.load(Ordering::Relaxed)),
        retire: AtomicBool::new(false),
    });
    for b in (base >> 16)..=((end - 1) >> 16) {
        let (w, bit) = bloom_bit(b);
        BLOOM[w].fetch_or(bit, Ordering::Relaxed);
    }
    g.insert(base, k.clone());
    ANY.store(true, Ordering::Release);
    STATS[0].fetch_add(1, Ordering::Relaxed);
    Ok(k)
}

/// Takes `k` out of the registry (the caller wrote it back if it must).
pub fn remove(k: &Kept) {
    let mut g = KEPT.write();
    if g.get(&k.base).is_some_and(|x| x.id == k.id) {
        g.remove(&k.base);
        STATS[1].fetch_add(1, Ordering::Relaxed);
    }
    ANY.store(!g.is_empty(), Ordering::Release);
}

/// `owner`'s window bound least recently (eviction).
pub fn oldest_of(owner: &str) -> Option<Arc<Kept>> {
    let g = KEPT.read();
    g.values().filter(|k| k.owner == owner).min_by_key(|k| k.last_flip.load(Ordering::Relaxed)).cloned()
}

/// `owner`'s windows.
pub fn owned_by(owner: &str) -> Vec<Arc<Kept>> {
    KEPT.read().values().filter(|k| k.owner == owner).cloned().collect()
}

/// A mapping change over [a, a+n): the windows there are dropped (their
/// GPU-written data with them, as the homes').
pub fn remapped(a: u64, n: u64) {
    if !any() {
        return;
    }
    let mut g = KEPT.write();
    let hit: Vec<u64> = g.range(a.saturating_sub(MAX)..a.saturating_add(n)).filter(|(_, k)| k.overlaps(a, n)).map(|(&b, _)| b).collect();
    for b in &hit {
        g.remove(b);
    }
    STATS[1].fetch_add(hit.len() as u64, Ordering::Relaxed);
    ANY.store(!g.is_empty(), Ordering::Release);
}

/// A binding of `k` (eviction order).
pub fn note_bound(k: &Kept) {
    k.last_flip.store(super::FRAMES.load(Ordering::Relaxed), Ordering::Relaxed);
    STATS[8].fetch_add(1, Ordering::Relaxed);
}

/// Batch `seq`, which wrote `written` and wrote back `given` (after its own
/// writes), was submitted.
pub fn submitted(seq: u64, written: &[Arc<Kept>], given: &[Arc<Kept>]) {
    for k in written {
        k.write_seq.fetch_max(seq, Ordering::AcqRel);
    }
    if !written.is_empty() {
        DIRTY_GEN.fetch_add(1, Ordering::AcqRel);
    }
    for k in given {
        k.given_seq.fetch_max(seq, Ordering::AcqRel);
    }
}

pub fn status() -> String {
    let n = |i: usize| STATS[i].load(Ordering::Relaxed);
    let (count, bytes) = {
        let g = KEPT.read();
        (g.len(), g.values().map(|k| k.size).sum::<u64>())
    };
    let chunks = POOL.lock().chunks.len();
    format!(
        "    kept windows (`keep_small`): {count} now ({} KiB, {chunks} chunks of {} MiB); created {}, dropped {}, refused for space {}, evicted {}, ranges refused {}; takes {}, bindings served {}, indirect arguments {}; write-backs: CPU-side reads {}, eager batch ends {}, overlapping bindings {}, other queues' writes {}, switch off {}; retakes after CP writes and fills {}\n",
        bytes >> 10,
        CHUNK >> 20,
        n(0),
        n(1),
        n(10),
        n(7),
        n(12),
        n(2),
        n(8),
        n(9),
        n(3),
        n(4),
        n(5),
        n(13),
        n(11),
        n(6)
    )
}
