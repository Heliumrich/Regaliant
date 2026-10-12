//! Video-memory homes of large buffers that GPU work writes.
//!
//! The title's per-item culls store the indices of the triangles they keep
//! into a 112 MB buffer that later draws read. Bound in place (imported
//! system memory), every store crossed PCIe: ~85% of the culls' GPU time
//! (66 against 8 us per dispatch when the stores went to scratch video
//! memory, run 20261011-065500). Such buffers are too large for the
//! per-batch VRAM shadows (`exec::SHADOW_MAX`). A buffer that GPU work
//! writes through a binding over that size (`note_large_write`) and that a
//! draw then uses as its index buffer gets a home: a device-local copy of
//! its guest range that every later GPU use inside the range binds
//! (dispatch and draw buffers, index buffers), across batches and queues
//! (one Vulkan queue runs all batches in submission order). (Homes for every
//! large written binding also caught buffers that draws sample as textures
//! and pools that unbounded pointers of other shaders span: their
//! write-backs made the courtyard slower, 1.94 against 2.27 flips/s.)
//!
//! Guest memory goes stale where GPU work writes a home (`stale`): readers
//! on the CPU side (command-processor reads, shader resolution) get those
//! parts written back first (`Exec::home_write_back`). Writes from the CPU
//! side that we see (CP writes, file reads) mark the range `dirty`; it is
//! copied in before the home's next GPU use. Writes by guest code are not
//! seen (imported pages cannot be write-protected safely, see `track`): a
//! home suits buffers only the GPU writes. A remap drops the home; batches
//! that bound it hold it until they complete.

use super::ranges::Ranges;
use super::vk::ctx;
use ash::vk;
use parking_lot::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

/// Largest buffer given a home.
pub const MAX: u64 = 256 << 20;
/// Video memory for all homes.
const BUDGET: u64 = 1 << 30;

/// A home's device buffer; destroyed when the last batch using it is done.
pub struct HomeBuf {
    pub buf: vk::Buffer,
    mem: vk::DeviceMemory,
    size: u64,
}

impl Drop for HomeBuf {
    fn drop(&mut self) {
        unsafe {
            ctx().device.destroy_buffer(self.buf, None);
            ctx().device.free_memory(self.mem, None);
        }
        USED.fetch_sub(self.size, Ordering::Relaxed);
    }
}

/// A home's state readable without the lock (`bind`'s fast path).
pub struct Shared {
    /// Unique per home (stale notes name it).
    id: u64,
    /// Parts written from the CPU side wait to be copied in.
    dirty: AtomicBool,
    /// Bumped when stale parts are removed (written back, overwritten from
    /// the CPU side): a thread's note that it marked a range stale holds
    /// while it stays the same.
    stale_gen: AtomicU64,
}

struct Home {
    base: u64,
    size: u64,
    buf: Arc<HomeBuf>,
    shared: Arc<Shared>,
    /// Guest ranges written from the CPU side since they were last copied
    /// in (all of it at creation).
    dirty: Ranges,
    /// Guest ranges GPU work wrote here: guest memory is stale there.
    stale: Ranges,
}

static HOMES: Mutex<Vec<Home>> = Mutex::new(Vec::new());
/// Bumped whenever homes are created or dropped: threads keep a copy of the
/// home ranges (`ranges`) and check most bindings without the lock.
static GEN: AtomicU64 = AtomicU64::new(1);

type Snap = (u64, Vec<(u64, u64)>, Vec<(Arc<HomeBuf>, Arc<Shared>)>);

thread_local! {
    static RANGES: std::cell::RefCell<Snap> = const { std::cell::RefCell::new((0, Vec::new(), Vec::new())) };
    /// Ranges this thread marked stale: (base, size, home, its stale_gen).
    static STALE_NOTES: std::cell::RefCell<Vec<(u64, u64, u64, u64)>> = const { std::cell::RefCell::new(Vec::new()) };
    /// The large written range this thread noted last (`note_large_write`).
    static LAST_LARGE: std::cell::Cell<(u64, u64)> = const { std::cell::Cell::new((0, 0)) };
}

/// Runs `f` on this thread's copy of the homes: ranges (base, size) and
/// their buffers and shared state, refreshed when homes come or go.
fn with_snap<R>(f: impl FnOnce(&[(u64, u64)], &[(Arc<HomeBuf>, Arc<Shared>)]) -> R) -> R {
    let g = GEN.load(Ordering::Acquire);
    RANGES.with(|c| {
        let mut c = c.borrow_mut();
        if c.0 != g {
            let homes = HOMES.lock();
            c.1 = homes.iter().map(|h| (h.base, h.size)).collect();
            c.2 = homes.iter().map(|h| (h.buf.clone(), h.shared.clone())).collect();
            c.0 = g;
        }
        f(&c.1, &c.2)
    })
}

/// Runs `f` on this thread's copy of the home ranges (base, size).
fn with_ranges<R>(f: impl FnOnce(&[(u64, u64)]) -> R) -> R {
    with_snap(|r, _| f(r))
}
/// Guest ranges GPU work wrote through bindings too large for a shadow
/// (home candidates).
static LARGE_WRITES: Mutex<Ranges> = Mutex::new(Ranges::new());
pub static UNBOUNDED_OVERLAPS: AtomicU64 = AtomicU64::new(0);
static ANY: AtomicBool = AtomicBool::new(false);
static USED: AtomicU64 = AtomicU64::new(0);
pub static CREATED: AtomicU64 = AtomicU64::new(0);
pub static DROPPED: AtomicU64 = AtomicU64::new(0);
pub static BINDINGS: AtomicU64 = AtomicU64::new(0);
pub static COPIED_IN: AtomicU64 = AtomicU64::new(0);
pub static WRITE_BACKS: AtomicU64 = AtomicU64::new(0);
pub static WRITE_BACK_BYTES: AtomicU64 = AtomicU64::new(0);
pub static PARTIAL: AtomicU64 = AtomicU64::new(0);

/// Whether any home exists (fast path for every binding).
pub fn any() -> bool {
    ANY.load(Ordering::Relaxed)
}

/// A binding of guest [base, base+size) in a home.
pub struct Bound {
    pub buf: Arc<HomeBuf>,
    /// Offset of `base` in the home's buffer.
    pub offset: u64,
    /// Bytes from `base` to the home's end.
    pub avail: u64,
    /// Copies in needed first: (guest address, bytes, offset in the home).
    pub copy_in: Vec<(u64, u64, u64)>,
}

/// The home holding all of [base, base+size), with the copies in its use
/// needs first (taken: the caller records them before the use). `written`:
/// the use writes the range (guest memory goes stale there).
pub fn bind(base: u64, size: u64, written: bool) -> Option<Bound> {
    if !any() || size == 0 {
        return None;
    }
    let found = with_snap(|r, s| r.iter().position(|&(b, n)| b <= base && base + size <= b + n).map(|i| (r[i], s[i].0.clone(), s[i].1.clone())));
    let ((hb, hn), buf, shared) = found?;
    // Without the lock when nothing waits to be copied in and a written
    // range is already marked stale by this thread (the culls and the draws
    // reading their indices bind the 112 MB home ~17k times per frame; the
    // lock was a main point of contention between the recording threads).
    if !shared.dirty.load(Ordering::Acquire) {
        let sgen = shared.stale_gen.load(Ordering::Acquire);
        let id = shared.id;
        if !written || STALE_NOTES.with(|n| n.borrow().iter().any(|&(b, m, h, sg)| h == id && sg == sgen && b <= base && base + size <= b + m)) {
            BINDINGS.fetch_add(1, Ordering::Relaxed);
            return Some(Bound { buf, offset: base - hb, avail: hb + hn - base, copy_in: Vec::new() });
        }
    }
    let mut g = HOMES.lock();
    let h = g.iter_mut().find(|h| h.base <= base && base + size <= h.base + h.size)?;
    let copy_in: Vec<(u64, u64, u64)> = if h.dirty.is_empty() {
        Vec::new()
    } else {
        // All dirty parts: other users of the home need them too, and the
        // copies are recorded before this use only.
        let v: Vec<(u64, u64, u64)> = h.dirty.iter().map(|(a, n)| (a, n, a - h.base)).collect();
        h.dirty.clear();
        h.shared.dirty.store(false, Ordering::Release);
        v
    };
    if written {
        h.stale.insert(base, size);
        let note = (base, size, h.shared.id, h.shared.stale_gen.load(Ordering::Acquire));
        STALE_NOTES.with(|n| {
            let mut n = n.borrow_mut();
            if n.len() >= 16 {
                n.remove(0);
            }
            n.push(note);
        });
    }
    BINDINGS.fetch_add(1, Ordering::Relaxed);
    Some(Bound { buf: h.buf.clone(), offset: base - h.base, avail: h.base + h.size - base, copy_in })
}

/// Whether binding [base, base+size) would copy guest data into its home
/// first (`bind`: parts the CPU wrote since; a parallel chunk stops there:
/// the copy must precede every earlier use in program order).
pub fn copy_in_pending(base: u64, size: u64) -> bool {
    if !any() {
        return false;
    }
    with_snap(|r, s| r.iter().position(|&(b, n)| b <= base && base + size <= b + n).is_some_and(|i| s[i].1.dirty.load(Ordering::Acquire)))
}

/// The homes overlapping [base, base+size) (diagnostics).
pub fn describe(base: u64, size: u64) -> String {
    let g = HOMES.lock();
    let v: Vec<String> = g.iter().filter(|h| h.base < base + size && base < h.base + h.size).map(|h| format!("{:#x}+{:#x}", h.base, h.size)).collect();
    v.join(", ")
}

/// GPU work wrote [base, base+size) through a binding too large for a
/// VRAM shadow.
pub fn note_large_write(base: u64, size: u64) {
    // The culls note the same range for every dispatch.
    if LAST_LARGE.with(|c| c.get()) == (base, size) {
        return;
    }
    LARGE_WRITES.lock().insert(base, size);
    LAST_LARGE.with(|c| c.set((base, size)));
}

/// An index buffer [base, base+size) inside a range noted by
/// `note_large_write`: that range gets a home (`create`).
pub fn index_use(base: u64, size: u64) -> bool {
    let found = LARGE_WRITES.lock().iter().find(|&(a, n)| a <= base && base + size <= a + n);
    match found {
        Some((a, n)) => create(a, n),
        None => false,
    }
}

/// Whether a home holds all of [base, base+size).
pub fn contains(base: u64, size: u64) -> bool {
    any() && with_ranges(|r| r.iter().any(|&(b, n)| b <= base && base + size <= b + n))
}

/// Whether [base, base+size) overlaps a home.
pub fn overlaps(base: u64, size: u64) -> bool {
    any() && with_ranges(|r| r.iter().any(|&(b, n)| b < base + size && base < b + n))
}

/// Creates a home for the pages of [base, base+size) unless one overlaps
/// them, the size is over `MAX` or the budget is spent; the whole range is
/// copied in at its first use.
pub fn create(base: u64, size: u64) -> bool {
    let (b, e) = (base & !0xFFF, (base + size).next_multiple_of(0x1000));
    let n = e - b;
    if n > MAX || USED.load(Ordering::Relaxed) + n > BUDGET {
        return false;
    }
    let mut g = HOMES.lock();
    if g.iter().any(|h| h.base < e && b < h.base + h.size) {
        return false;
    }
    let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::INDIRECT_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
    let (buf, mem) = match ctx().device_buffer(n, usage) {
        Ok(x) => x,
        Err(e) => {
            crate::warn!("gpu: no video memory for a home of {b:#x}+{n:#x} ({e})");
            return false;
        }
    };
    USED.fetch_add(n, Ordering::Relaxed);
    let mut dirty = Ranges::default();
    dirty.insert(b, n);
    static NEXT_ID: AtomicU64 = AtomicU64::new(1);
    let shared = Arc::new(Shared { id: NEXT_ID.fetch_add(1, Ordering::Relaxed), dirty: AtomicBool::new(true), stale_gen: AtomicU64::new(0) });
    g.push(Home { base: b, size: n, buf: Arc::new(HomeBuf { buf, mem, size: n }), shared, dirty, stale: Ranges::default() });
    ANY.store(true, Ordering::Relaxed);
    GEN.fetch_add(1, Ordering::AcqRel);
    CREATED.fetch_add(1, Ordering::Relaxed);
    crate::info!("gpu: video-memory home for {b:#x}+{n:#x} ({} MiB)", n >> 20);
    true
}

/// The stale parts (GPU-written, not yet in guest memory) of [base,
/// base+size): (guest address, bytes, home buffer, offset in the home).
/// `taken`: they will be written back (no longer stale).
pub fn stale_parts(base: u64, size: u64, taken: bool) -> Vec<(u64, u64, Arc<HomeBuf>, u64)> {
    if !any() {
        return Vec::new();
    }
    let mut g = HOMES.lock();
    let mut v = Vec::new();
    for h in g.iter_mut() {
        if !(h.base < base + size && base < h.base + h.size) {
            continue;
        }
        for (a, n) in h.stale.intersect(base, size) {
            v.push((a, n, h.buf.clone(), a - h.base));
        }
        if taken {
            h.stale.remove(base, size);
            h.shared.stale_gen.fetch_add(1, Ordering::AcqRel);
        }
    }
    v
}

/// Whether any of `reads` overlaps stale parts of a home.
pub fn any_stale(reads: &[(u64, u64)]) -> bool {
    any()
        && with_ranges(|r| reads.iter().any(|&(a, n)| r.iter().any(|&(b, m)| b < a + n && a < b + m)))
        && HOMES.lock().iter().any(|h| reads.iter().any(|&(a, n)| h.stale.overlaps(a, n)))
}

/// A write from the CPU side (CP write, file read) into guest memory: the
/// homes it overlaps copy it in before their next GPU use.
pub fn cpu_written(base: u64, size: u64) {
    if !any() || size == 0 {
        return;
    }
    let mut g = HOMES.lock();
    for h in g.iter_mut() {
        let (a, e) = (base.max(h.base), (base + size).min(h.base + h.size));
        if a < e {
            h.dirty.insert(a, e - a);
            h.shared.dirty.store(true, Ordering::Release);
            // The CPU-side write is newer than what GPU work left there.
            h.stale.remove(a, e - a);
            h.shared.stale_gen.fetch_add(1, Ordering::AcqRel);
        }
    }
}

/// The mapping of [base, base+size) changed: overlapping homes are dropped
/// (their contents with it).
pub fn remapped(base: u64, size: u64) {
    LARGE_WRITES.lock().remove(base, size);
    if !any() {
        return;
    }
    let mut g = HOMES.lock();
    let before = g.len();
    g.retain(|h| {
        let hit = h.base < base.saturating_add(size) && base < h.base + h.size;
        if hit {
            crate::info!("gpu: home {:#x}+{:#x} dropped (mapping change at {base:#x}+{size:#x})", h.base, h.size);
        }
        !hit
    });
    DROPPED.fetch_add((before - g.len()) as u64, Ordering::Relaxed);
    ANY.store(!g.is_empty(), Ordering::Relaxed);
    GEN.fetch_add(1, Ordering::AcqRel);
}

/// All homes with their GPU-written parts (taken), then dropped (`tune
/// homes 0` while homes exist: the caller writes those parts back first).
/// A queue still recording work that bound a home keeps writing it until
/// that batch is submitted; those writes are lost (A/B use only).
pub fn take_all() -> Vec<(u64, u64, Arc<HomeBuf>, u64)> {
    let mut g = HOMES.lock();
    let mut v = Vec::new();
    for h in g.iter_mut() {
        for (a, n) in h.stale.iter() {
            v.push((a, n, h.buf.clone(), a - h.base));
        }
    }
    DROPPED.fetch_add(g.len() as u64, Ordering::Relaxed);
    g.clear();
    ANY.store(false, Ordering::Relaxed);
    GEN.fetch_add(1, Ordering::AcqRel);
    v
}

pub fn status() -> String {
    format!(
        "    video-memory homes: {} created, {} dropped, {} MiB; {} bindings, {} bytes copied in, {} write-backs ({} KiB), {} partial overlaps, {} unbounded pointer bindings over homes\n",
        CREATED.load(Ordering::Relaxed),
        DROPPED.load(Ordering::Relaxed),
        USED.load(Ordering::Relaxed) >> 20,
        BINDINGS.load(Ordering::Relaxed),
        COPIED_IN.load(Ordering::Relaxed),
        WRITE_BACKS.load(Ordering::Relaxed),
        WRITE_BACK_BYTES.load(Ordering::Relaxed) >> 10,
        PARTIAL.load(Ordering::Relaxed),
        UNBOUNDED_OVERLAPS.load(Ordering::Relaxed)
    )
}
