//! Video-memory homes of large buffers that GPU work writes.
//!
//! The title's per-item culls store the indices of the triangles they keep
//! into a 112 MB buffer that later draws read. Bound in place (imported
//! system memory), every store crossed PCIe: ~85% of the culls' GPU time
//! (66 against 8 us per dispatch when the stores went to scratch video
//! memory, run 20261011-065500). Such buffers are too large for the
//! per-batch VRAM shadows (`tune shadow_max`). A buffer that GPU work
//! writes through a binding over that size (`note_large_write`) and that a
//! draw then uses as its index buffer gets a home: a device-local copy of
//! its guest range that every later GPU use inside the range binds
//! (dispatch and draw buffers, index buffers), across batches and queues
//! (one Vulkan queue runs all batches in submission order). With `tune
//! home_reads` (off by default: no measured gain) a large read-only buffer
//! binding inside such a range creates one too (`read_use`). (Homes for every
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
//!
//! Indirect draw arguments (`tune draw_args_gpu`): the culls write the
//! frame's DrawIndexIndirect records (a 300 KB buffer, 0x1008a9ea10 at the
//! courtyard) through VRAM shadows, which were copied in at each batch's
//! first use and merged back at its end (~5 ms of GPU time per frame) so
//! that the command processor could read them. Draws that read their
//! arguments on the GPU need neither: such a buffer (an exact written
//! binding of 16- or 20-byte records, `note_shadow_write`) named by draws
//! that could read their arguments on the GPU (`vote_args`) gets a home of
//! its own (`ready_args_candidates`, `create_args`) once the CPU was seen
//! writing none of its pages for OBSERVE_FRAMES frames (the CPU writes some
//! such buffers whole: dword merges kept both sides' writes, a home could
//! not). The home stays
//! write-watched: once per synchronization epoch, as for the batch-start
//! copies (`precache`), the pages the CPU wrote are compared word by word
//! with the guest memory the home last took in (`Home::snap`), and only the
//! words that changed are copied in before its next GPU use: the GPU-written
//! words around them stay, and a page the home covers in part may be written
//! next to it (the last page of the culls' records is, every frame). The
//! CPU rewrites those records at level loads (4-2 drew without boulders when
//! partly covered pages were left out).

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
        super::vk::note_gpu_freed(self.buf);
        unsafe {
            ctx().device.destroy_buffer(self.buf, None);
            ctx().device.free_memory(self.mem, None);
        }
        USED.fetch_sub(self.size, Ordering::Relaxed);
    }
}

/// The windows a write-watched home's checks for CPU writes are made per.
const CHECK_WIN: u64 = 2 << 20;

impl Shared {
    /// The windows of [a, a+n) (inside the home).
    fn win_range(&self, a: u64, n: u64) -> std::ops::Range<usize> {
        let first = (a.saturating_sub(self.base) / CHECK_WIN) as usize;
        let last = ((a + n.max(1) - 1).saturating_sub(self.base) / CHECK_WIN) as usize;
        first.min(self.wins.len())..(last + 1).min(self.wins.len())
    }

    /// Whether every window of [a, a+n) was checked in `epoch`.
    fn checked(&self, a: u64, n: u64, epoch: u64) -> bool {
        self.wins[self.win_range(a, n)].iter().all(|w| w.load(Ordering::Acquire) == epoch)
    }

    /// The span of the windows of [a, a+n) not checked in `epoch`: (start,
    /// bytes) inside a home of `size` bytes (None: all checked).
    fn unchecked_span(&self, a: u64, n: u64, epoch: u64, size: u64) -> Option<(u64, u64)> {
        // `tune home_check_win` 0: every window of the home at once.
        let r = if super::tune::home_check_win() { self.win_range(a, n) } else { 0..self.wins.len() };
        let first = r.clone().find(|&i| self.wins[i].load(Ordering::Acquire) != epoch)?;
        let last = r.rev().find(|&i| self.wins[i].load(Ordering::Acquire) != epoch)?;
        let start = first as u64 * CHECK_WIN;
        let end = ((last as u64 + 1) * CHECK_WIN).min(size);
        Some((self.base + start, end - start))
    }

    /// Marks the windows of [a, a+n) checked in `epoch`.
    fn mark_checked(&self, a: u64, n: u64, epoch: u64) {
        for w in &self.wins[self.win_range(a, n)] {
            if w.load(Ordering::Acquire) < epoch {
                w.store(epoch, Ordering::Release);
            }
        }
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
    /// Write-watched (`create_args`; every home with `tune homes_watch`).
    watched: bool,
    /// Per CHECK_WIN bytes of the home from `base` (one for a smaller
    /// home): the synchronization epoch the window's pages were last checked
    /// for CPU writes in (`precache::epoch`). A binding checks the windows it
    /// covers only (the culls' 112 MB index home was checked whole, ~126 us,
    /// by the first index binding of every epoch, which reads a few KiB).
    wins: Box<[AtomicU64]>,
    base: u64,
    /// The epoch a thread is checking it for now, its write-mark query made
    /// without the homes' lock (`check_unlocked`; 0: none).
    checking: AtomicU64,
    /// A home of indirect draw arguments (`create_args`).
    args: bool,
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
    /// Write-watched homes: guest memory's words as the home last took them
    /// in or saw them (pages the CPU wrote are compared with it).
    snap: Option<Vec<u32>>,
    /// Ranges written back since the last check: guest memory changed there
    /// without the CPU (taken into `snap` without comparing).
    resnap: Ranges,
}

/// Reads guest words [a, a+4n) (a mapped range).
fn guest_words(a: u64, n: usize, out: &mut [u32]) {
    for (i, w) in out.iter_mut().enumerate().take(n) {
        *w = unsafe { std::ptr::read_volatile((a + 4 * i as u64) as *const u32) };
    }
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
    /// The shadow-sized written binding this thread noted last, and the
    /// frame (noted again once per frame: kept among the recent ones).
    static LAST_SHADOW: std::cell::Cell<(u64, u64, u64)> = const { std::cell::Cell::new((0, 0, 0)) };
    /// The argument buffer this thread named last (`vote_args`), and the
    /// page (and frame) it last found none on.
    static LAST_VOTE: std::cell::Cell<(u64, u64, u64)> = const { std::cell::Cell::new((0, 0, 0)) };
    static LAST_MISS: std::cell::Cell<(u64, u64)> = const { std::cell::Cell::new((u64::MAX, 0)) };
}

/// Runs `f` on this thread's copy of the homes: ranges (base, size) and
/// their buffers and shared state, refreshed when homes come or go.
fn with_snap<R>(f: impl FnOnce(&[(u64, u64)], &[(Arc<HomeBuf>, Arc<Shared>)]) -> R) -> R {
    let g = GEN.load(Ordering::Acquire);
    RANGES.with(|c| {
        let mut c = c.borrow_mut();
        if c.0 != g {
            let homes = super::lockstat::HOMES.lock(&HOMES);
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
/// Exact written bindings small enough for a VRAM shadow whose records are
/// the size of indirect draw arguments (16 or 20 bytes), most recent last
/// (`tune draw_args_gpu`: where those arguments may live; all written
/// bindings were too many: the culls' buffer fell out within a frame).
static SHADOW_WRITES: Mutex<Vec<(u64, u64)>> = Mutex::new(Vec::new());
const SHADOW_WRITES_MAX: usize = 256;
/// Write-watched homes: bytes the CPU changed (copied in), written pages
/// the home covers only in part, checks.
pub static CPU_WRITTEN_PAGES: AtomicU64 = AtomicU64::new(0);
pub static EDGE_PAGES: AtomicU64 = AtomicU64::new(0);
pub static WATCH_CHECKS: AtomicU64 = AtomicU64::new(0);
/// Nanoseconds the checks took (~130 us each over whole homes, mostly the
/// write-watch query: a watcher thread starting them when epochs end saved
/// nothing, 2026-10-17).
pub static WATCH_NS: AtomicU64 = AtomicU64::new(0);
/// Frames an argument buffer is watched for CPU writes before its home.
const OBSERVE_FRAMES: u64 = 120;
/// Argument buffers watched before their home: (base, size, first frame);
/// buffers the CPU wrote (never given a home).
static OBSERVED: Mutex<Vec<(u64, u64, u64)>> = Mutex::new(Vec::new());
static CPU_SIDE: Mutex<Vec<(u64, u64)>> = Mutex::new(Vec::new());
/// Argument buffers named by draws that could read them on the GPU.
static VOTED: Mutex<Vec<(u64, u64)>> = Mutex::new(Vec::new());
/// Argument buffers whose home `tune draw_args_gpu 0` dropped: watched
/// before, made again without waiting OBSERVE_FRAMES.
static KNOWN: Mutex<Vec<(u64, u64)>> = Mutex::new(Vec::new());
/// Bumped when argument homes come or go (threads name a buffer again).
static VOTE_GEN: AtomicU64 = AtomicU64::new(1);
/// Whether any write-watched home exists.
static WATCHED_ANY: AtomicBool = AtomicBool::new(false);
pub static UNBOUNDED_OVERLAPS: AtomicU64 = AtomicU64::new(0);
/// Of those: [based inside a home, with GPU-written parts of a home not yet
/// in guest memory within 4 KiB of their base, written back first (`tune
/// unbounded_near`)].
pub static UNBOUNDED_STALE: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
/// Image uploads from guest memory (new textures, re-uploads, reloads of
/// overwritten images) over GPU-written parts of a home not yet in guest
/// memory: [uploads, of them reloads of overwritten images whose parts were
/// written back first (`tune image_homes`)].
pub static IMAGE_STALE: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
static ANY: AtomicBool = AtomicBool::new(false);
/// The write-watched homes (base, end, id), readable without the homes'
/// lock (`marks_taken` runs under the copy cache's, which a home's check
/// takes while holding the homes' lock); kept by `publish_watched`.
static WATCHED_RANGES: parking_lot::RwLock<Vec<(u64, u64, u64)>> = parking_lot::RwLock::new(Vec::new());
static WATCHED_ON: AtomicBool = AtomicBool::new(false);
/// Ranges inside watched homes whose write marks other takes reset (texture
/// re-uploads and scrubbing, copy-cache checks of ranges a home was made
/// over later): (home id, base, size); the home's next check compares them
/// too (its own query no longer sees those writes).
static TAKEN: Mutex<Vec<(u64, u64, u64)>> = Mutex::new(Vec::new());
/// Pages (or ranges of failed queries) handed over that way.
static TAKEN_PAGES: AtomicU64 = AtomicU64::new(0);
static USED: AtomicU64 = AtomicU64::new(0);
pub static CREATED: AtomicU64 = AtomicU64::new(0);
pub static DROPPED: AtomicU64 = AtomicU64::new(0);
pub static BINDINGS: AtomicU64 = AtomicU64::new(0);
pub static COPIED_IN: AtomicU64 = AtomicU64::new(0);
pub static WRITE_BACKS: AtomicU64 = AtomicU64::new(0);
/// Bytes guest code changed in homes that are write-watched only because of
/// `tune homes_watch` (unseen without it).
pub static GUEST_CHANGED: AtomicU64 = AtomicU64::new(0);
pub static WRITE_BACK_BYTES: AtomicU64 = AtomicU64::new(0);
pub static PARTIAL: AtomicU64 = AtomicU64::new(0);
/// Bumped after stale parts are added (GPU work bound a home for writing):
/// `Exec`'s cache of clean pages holds while it stays the same.
pub static STALE_ADDED: AtomicU64 = AtomicU64::new(0);

/// The epoch write-watched homes check their pages once per: every
/// synchronization epoch, or with `tune home_submit_epochs` those ended by
/// title submissions and REWIND only. Waits the CPU released end most
/// epochs (hundreds per frame), and the culls' 112 MB index home was
/// checked in ~19 of them per gate frame (~126 us each, under the homes'
/// lock that every cull's binding waits for); the CPU writes these homes
/// see come at level loads, before the submissions that use them.
fn check_epoch() -> u64 {
    if super::tune::home_submit_epochs() { super::precache::submit_epoch() } else { super::precache::epoch() }
}

/// Whether any home exists (fast path for every binding).
pub fn any() -> bool {
    ANY.load(Ordering::Relaxed)
}

/// Republishes the write-watched homes' ranges after homes came or went
/// (under the homes' lock), forgetting handed-over ranges of homes gone.
fn publish_watched(g: &[Home]) {
    let v: Vec<(u64, u64, u64)> = g.iter().filter(|h| h.shared.watched).map(|h| (h.base, h.base + h.size, h.shared.id)).collect();
    WATCHED_ON.store(!v.is_empty(), Ordering::Release);
    TAKEN.lock().retain(|t| v.iter().any(|w| w.2 == t.0));
    *WATCHED_RANGES.write() = v;
}

/// Another take reset the write marks of `pages` (`precache::apply_marks`;
/// None: a failed query over [base, base+size), all of it may be written):
/// those inside write-watched homes are handed to their next check.
pub fn marks_taken(pages: Option<&[u64]>, base: u64, size: u64) {
    if !WATCHED_ON.load(Ordering::Acquire) {
        return;
    }
    let w = WATCHED_RANGES.read();
    let mut add = Vec::new();
    match pages {
        Some(pages) => {
            for &p in pages {
                for &(b, e, id) in w.iter() {
                    if p < e && b < p + 4096 {
                        add.push((id, p, 4096));
                    }
                }
            }
        }
        None => {
            let end = base.saturating_add(size);
            for &(b, e, id) in w.iter() {
                if base < e && b < end {
                    add.push((id, base.max(b), end.min(e) - base.max(b)));
                }
            }
        }
    }
    drop(w);
    if !add.is_empty() {
        TAKEN_PAGES.fetch_add(add.len() as u64, Ordering::Relaxed);
        TAKEN.lock().extend(add);
    }
}

/// The ranges handed over for home `id` (`marks_taken`), removed.
fn taken_for(id: u64) -> Vec<(u64, u64)> {
    let mut g = TAKEN.lock();
    if g.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    g.retain(|t| {
        if t.0 == id {
            out.push((t.1, t.2));
            false
        } else {
            true
        }
    });
    out
}

/// A binding of guest [base, base+size) in a home.
pub struct Bound {
    /// The home's buffer (`bind` handed the home to its `keep`).
    pub buf: vk::Buffer,
    /// Offset of `base` in the home's buffer.
    pub offset: u64,
    /// Bytes from `base` to the home's end.
    pub avail: u64,
    /// Copies in needed first: (guest address, bytes, offset in the home).
    pub copy_in: Vec<(u64, u64, u64)>,
}

/// The home holding all of [base, base+size), with the copies in its use
/// needs first (taken: the caller records them before the use). `written`:
/// the use writes the range (guest memory goes stale there). `keep` gets
/// the home while it is certainly alive: the caller keeps it for the work
/// that uses it (and clones it only when it does not hold it yet).
pub fn bind(base: u64, size: u64, written: bool, keep: &mut dyn FnMut(&Arc<HomeBuf>)) -> Option<Bound> {
    if !any() || size == 0 {
        return None;
    }
    // Without the lock when nothing waits to be copied in and a written
    // range is already marked stale by this thread (the culls and the draws
    // reading their indices bind the 112 MB home ~17k times per frame; the
    // lock was a main point of contention between the recording threads),
    // and by reference: the homes' reference counts are shared by all
    // recording threads.
    enum Fast {
        Hit(vk::Buffer, u64, u64),
        Slow,
        Miss,
    }
    let f = with_snap(|r, s| {
        let Some(i) = r.iter().position(|&(b, n)| b <= base && base + size <= b + n) else { return Fast::Miss };
        let (hb, hn) = r[i];
        let shared = &s[i].1;
        if !shared.dirty.load(Ordering::Acquire) && (!shared.watched || shared.checked(base, size, check_epoch())) {
            let sgen = shared.stale_gen.load(Ordering::Acquire);
            let id = shared.id;
            if !written || STALE_NOTES.with(|n| n.borrow().iter().any(|&(b, m, h, sg)| h == id && sg == sgen && b <= base && base + size <= b + m)) {
                keep(&s[i].0);
                return Fast::Hit(s[i].0.buf, base - hb, hb + hn - base);
            }
        }
        Fast::Slow
    });
    match f {
        Fast::Miss => return None,
        Fast::Hit(buf, offset, avail) => {
            note_binding();
            return Some(Bound { buf, offset, avail, copy_in: Vec::new() });
        }
        Fast::Slow => {}
    }
    let epoch = check_epoch();
    if super::tune::home_check_unlocked() {
        check_before_lock(base, size, epoch);
    }
    let mut g = super::lockstat::HOMES.lock(&HOMES);
    let h = g.iter_mut().find(|h| h.base <= base && base + size <= h.base + h.size)?;
    if h.shared.watched && !h.shared.checked(base, size, epoch) {
        check_cpu_writes(h, base, size, epoch);
    }
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
        // Only new stale parts move the generation: every executor's cache of
        // pages clean of them (`Exec::clean_pages`) is dropped when it moves,
        // and homes written in every submission epoch moved it constantly.
        if !h.stale.contains(base, size) {
            h.stale.insert(base, size);
            STALE_ADDED.fetch_add(1, Ordering::AcqRel);
        }
        let note = (base, size, h.shared.id, h.shared.stale_gen.load(Ordering::Acquire));
        STALE_NOTES.with(|n| {
            let mut n = n.borrow_mut();
            if n.len() >= 16 {
                n.remove(0);
            }
            n.push(note);
        });
    }
    note_binding();
    keep(&h.buf);
    Some(Bound { buf: h.buf.buf, offset: base - h.base, avail: h.base + h.size - base, copy_in })
}

/// The home whose buffer is `buf` (to keep it for later work), if it
/// still exists.
pub fn handle(buf: vk::Buffer) -> Option<Arc<HomeBuf>> {
    with_snap(|_, s| s.iter().find(|(h, _)| h.buf == buf).map(|(h, _)| h.clone())).or_else(|| super::lockstat::HOMES.lock(&HOMES).iter().find(|h| h.buf.buf == buf).map(|h| h.buf.clone()))
}

/// Counts a binding (BINDINGS, in batches per thread: one shared counter
/// bumped by every binding of every recording thread).
fn note_binding() {
    thread_local! {
        static N: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
    }
    N.with(|c| {
        let v = c.get() + 1;
        if v == 256 {
            BINDINGS.fetch_add(256, Ordering::Relaxed);
            c.set(0);
        } else {
            c.set(v);
        }
    });
}

/// Whether binding [base, base+size) would copy guest data into its home
/// first (`bind`: parts the CPU wrote since; a parallel chunk stops there:
/// the copy must precede every earlier use in program order).
pub fn copy_in_pending(base: u64, size: u64) -> bool {
    if !any() {
        return false;
    }
    let epoch = check_epoch();
    let state = with_snap(|r, s| {
        r.iter()
            .position(|&(b, n)| b <= base && base + size <= b + n)
            .map(|i| (s[i].1.dirty.load(Ordering::Acquire), s[i].1.watched && !s[i].1.checked(base, size, epoch)))
    });
    match state {
        None | Some((false, false)) => false,
        Some((true, _)) => true,
        Some((false, true)) => {
            // A write-watched home not checked in this epoch: checked now
            // (pages the CPU wrote become dirty and wait for the caller's
            // in-order copy in).
            if super::tune::home_check_unlocked() {
                check_before_lock(base, size, epoch);
            }
            let mut g = super::lockstat::HOMES.lock(&HOMES);
            let Some(h) = g.iter_mut().find(|h| h.base <= base && base + size <= h.base + h.size) else { return false };
            if !h.shared.checked(base, size, epoch) {
                check_cpu_writes(h, base, size, epoch);
            }
            h.shared.dirty.load(Ordering::Acquire)
        }
    }
}

/// The homes overlapping [base, base+size) (diagnostics).
pub fn describe(base: u64, size: u64) -> String {
    let g = super::lockstat::HOMES.lock(&HOMES);
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

/// The check of the write-watched home holding [base, base+size) for
/// `epoch`, if it is due, with the homes' lock released for its write-mark
/// query (`tune home_check_unlocked`): one thread claims it
/// (`Shared::checking`) and applies what it found under the lock; others
/// binding that home wait for it, bindings of other homes not at all (the
/// query of the culls' 112 MB home took ~100 us under the lock that every
/// recording thread's home bindings waited for: 20,590 contended waits in a
/// 903-flip vista window, 2.3 s, run 20261026-091347).
fn check_before_lock(base: u64, size: u64, epoch: u64) {
    let Some((hn, shared)) = with_snap(|r, s| r.iter().position(|&(b, n)| b <= base && base + size <= b + n).map(|i| (r[i].1, s[i].1.clone()))) else { return };
    if !shared.watched {
        return;
    }
    let mut spins = 0u32;
    while !shared.checked(base, size, epoch) {
        match shared.checking.compare_exchange(0, epoch, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => {
                let Some((qa, qn)) = shared.unchecked_span(base, size, epoch, hn) else {
                    shared.checking.store(0, Ordering::Release);
                    return;
                };
                WATCH_CHECKS.fetch_add(1, Ordering::Relaxed);
                WATCH_BYTES.fetch_add(qn, Ordering::Relaxed);
                let t0 = std::time::Instant::now();
                let q = query_cpu_writes(qa, qn, shared.id);
                {
                    let mut g = super::lockstat::HOMES.lock(&HOMES);
                    if let Some(h) = g.iter_mut().find(|h| h.shared.id == shared.id) {
                        apply_cpu_writes(h, q);
                    }
                    shared.mark_checked(qa, qn, epoch);
                }
                shared.checking.store(0, Ordering::Release);
                WATCH_NS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                UNLOCKED_CHECKS.fetch_add(1, Ordering::Relaxed);
                return;
            }
            Err(_) => {
                // Another thread's check of this home (for this epoch or an
                // earlier one) is under way: its result first.
                spins += 1;
                if spins < 64 {
                    std::hint::spin_loop();
                } else {
                    std::thread::yield_now();
                }
                if spins == 64 {
                    CHECK_WAITS.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
    }
}

/// Checks made by `check_before_lock`, and bindings that waited for another
/// thread's check of their home.
static UNLOCKED_CHECKS: AtomicU64 = AtomicU64::new(0);
/// Bytes of homes the checks queried.
static WATCH_BYTES: AtomicU64 = AtomicU64::new(0);
static CHECK_WAITS: AtomicU64 = AtomicU64::new(0);

/// The CPU's writes into a write-watched home's pages since they were last
/// taken, in the windows of [a, a+n) not checked in `epoch` (then marked):
/// those pages are copied in before its next GPU use (what GPU work wrote
/// there and was not written back yet is lost: logged).
fn check_cpu_writes(h: &mut Home, a: u64, n: u64, epoch: u64) {
    let Some((qa, qn)) = h.shared.unchecked_span(a, n, epoch, h.size) else { return };
    WATCH_CHECKS.fetch_add(1, Ordering::Relaxed);
    WATCH_BYTES.fetch_add(qn, Ordering::Relaxed);
    let t0 = std::time::Instant::now();
    let q = query_cpu_writes(qa, qn, h.shared.id);
    apply_cpu_writes(h, q);
    h.shared.mark_checked(qa, qn, epoch);
    WATCH_NS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
}

/// What a home's check found: the query's result, the pages it found
/// written, and the ranges other takes handed over (`marks_taken`).
type Found = (Option<()>, Vec<u64>, Vec<(u64, u64)>);

/// The write-mark query of a home's check (no lock held).
fn query_cpu_writes(base: u64, size: u64, id: u64) -> Found {
    let mut pages = Vec::new();
    let taken = super::precache::take_written_for(base, size, &mut pages, None, 2);
    // Pages whose marks other takes reset meanwhile (the take above waited
    // for those in flight over the home to be applied).
    (taken, pages, taken_for(id))
}

/// The words of a home's pages the CPU wrote (`Found`) that changed are
/// marked to be copied in (under the homes' lock).
fn apply_cpu_writes(h: &mut Home, (taken, pages, handed): Found) {
    let (base, end) = (h.base, h.base + h.size);
    let resnap: Vec<(u64, u64)> = h.resnap.iter().collect();
    h.resnap.clear();
    let Some(snap) = h.snap.as_mut() else { return };
    // Words our write-backs changed: taken as they are.
    for (a, n) in resnap {
        let (a, e) = (a.max(base), (a + n).min(end));
        if a < e {
            let k = ((a - base) / 4) as usize;
            guest_words(a, ((e - a) / 4) as usize, &mut snap[k..]);
        }
    }
    let mut ranges: Vec<(u64, u64)> = match taken {
        Some(()) => pages
            .iter()
            .filter_map(|&p| {
                if p < base || p + 4096 > end {
                    EDGE_PAGES.fetch_add(1, Ordering::Relaxed);
                }
                let (a, e) = (p.max(base), (p + 4096).min(end));
                (a < e).then_some((a, e - a))
            })
            .collect(),
        None => vec![(base, h.size)],
    };
    for (a, n) in handed {
        let (a, e) = (a.max(base), (a + n).min(end));
        if a < e && taken.is_some() {
            ranges.push((a, e - a));
        }
    }
    // The words that changed, as runs.
    let mut changed: Vec<(u64, u64)> = Vec::new();
    let mut now = [0u32; 1024];
    for (a, n) in ranges {
        let k0 = ((a - base) / 4) as usize;
        let cnt = (n / 4) as usize;
        let mut i = 0;
        while i < cnt {
            let m = (cnt - i).min(1024);
            guest_words(a + 4 * i as u64, m, &mut now);
            for j in 0..m {
                let k = k0 + i + j;
                if snap[k] != now[j] {
                    snap[k] = now[j];
                    let w = base + 4 * k as u64;
                    match changed.last_mut() {
                        Some(r) if r.0 + r.1 == w => r.1 += 4,
                        _ => changed.push((w, 4)),
                    }
                }
            }
            i += m;
        }
    }
    if changed.is_empty() {
        return;
    }
    let bytes: u64 = changed.iter().map(|r| r.1).sum();
    CPU_WRITTEN_PAGES.fetch_add(bytes, Ordering::Relaxed);
    if !h.shared.args {
        GUEST_CHANGED.fetch_add(bytes, Ordering::Relaxed);
    }
    static LOGGED: AtomicU64 = AtomicU64::new(0);
    if LOGGED.fetch_add(1, Ordering::Relaxed) < if h.shared.args { 20 } else { 200 } {
        let lost: u64 = changed.iter().map(|&(a, n)| h.stale.intersect(a, n).iter().map(|r| r.1).sum::<u64>()).sum();
        crate::info!(
            "gpu: the CPU changed {bytes} bytes ({} runs, first {:#x}) of write-watched home {:#x}+{:#x}: copied in before its next GPU use ({lost} of them GPU-written and not written back)",
            changed.len(),
            changed[0].0,
            h.base,
            h.size
        );
    }
    for (a, n) in changed {
        h.dirty.insert(a, n);
        h.stale.remove(a, n);
    }
    h.shared.dirty.store(true, Ordering::Release);
    h.shared.stale_gen.fetch_add(1, Ordering::AcqRel);
}

/// GPU work writes [base, base+size) through a binding small enough for a
/// VRAM shadow, of 16- or 20-byte records (`tune draw_args_gpu`: noted for
/// `vote_args`).
pub fn note_shadow_write(base: u64, size: u64) {
    let frame = super::FRAMES.load(Ordering::Relaxed);
    if LAST_SHADOW.with(|c| c.get()) == (base, size, frame) {
        return;
    }
    LAST_SHADOW.with(|c| c.set((base, size, frame)));
    let mut g = SHADOW_WRITES.lock();
    if let Some(i) = g.iter().rposition(|&r| r == (base, size)) {
        let r = g.remove(i);
        g.push(r);
    } else {
        if g.len() >= SHADOW_WRITES_MAX {
            g.remove(0);
        }
        g.push((base, size));
    }
}

/// DS_TRACE_ARGS_HOMES=1: the argument homes' decisions (`vote_args`,
/// `ready_args_candidates`), the first 400.
fn trace_args(f: impl FnOnce() -> String) {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    static N: AtomicU64 = AtomicU64::new(0);
    if *ON.get_or_init(|| std::env::var("DS_TRACE_ARGS_HOMES").ok().as_deref() == Some("1")) && N.fetch_add(1, Ordering::Relaxed) < 400 {
        crate::info!("args homes: {}", f());
    }
}

/// An indexed indirect draw that could read its arguments [a, a+n) on the
/// GPU but for their home: the buffer GPU work writes them through (a
/// shadow-sized binding, `note_shadow_write`; the smallest that holds them)
/// becomes a candidate (`ready_args_candidates`). Cheap when repeated: each
/// thread remembers the last buffer it named and the last page it found
/// none on (for the frame).
pub fn vote_args(a: u64, n: u64) {
    let (lb, lm, lg) = LAST_VOTE.with(|c| c.get());
    let vgen = VOTE_GEN.load(Ordering::Acquire);
    if lg == vgen && lb <= a && a + n <= lb + lm {
        return;
    }
    let frame = super::FRAMES.load(Ordering::Relaxed);
    if LAST_MISS.with(|c| c.get()) == (a >> 12, frame) {
        return;
    }
    let found = SHADOW_WRITES.lock().iter().filter(|&&(b, m)| b <= a && a + n <= b + m).min_by_key(|r| r.1).copied();
    match found {
        Some(r) => {
            LAST_VOTE.with(|c| c.set((r.0, r.1, vgen)));
            let mut v = VOTED.lock();
            if !v.contains(&r) {
                if v.len() >= 32 {
                    v.remove(0);
                }
                v.push(r);
                trace_args(|| format!("{a:#x} names {:#x}+{:#x} (frame {frame})", r.0, r.1));
            }
        }
        None => {
            trace_args(|| format!("{a:#x}: no shadow-written binding holds it (frame {frame}, {} noted)", SHADOW_WRITES.lock().len()));
            LAST_MISS.with(|c| c.set((a >> 12, frame)));
        }
    }
}

/// The candidates (`vote_args`) ready for a home: watched for CPU writes
/// for OBSERVE_FRAMES frames first, given up for good once the CPU wrote one
/// of the pages a candidate covers whole. Called when parallel runs are
/// planned (a few times per frame).
pub fn ready_args_candidates() -> Vec<(u64, u64)> {
    let voted: Vec<(u64, u64)> = {
        let v = VOTED.lock();
        if v.is_empty() {
            return Vec::new();
        }
        v.clone()
    };
    let mut ready = Vec::new();
    for (b, m) in voted {
        let give_up = || {
            VOTED.lock().retain(|&r| r != (b, m));
            OBSERVED.lock().retain(|o| (o.0, o.1) != (b, m));
        };
        if !crate::mem::write_watched() || b % 4 != 0 || m % 4 != 0 || overlaps(b, m) || CPU_SIDE.lock().iter().any(|&r| r == (b, m)) {
            trace_args(|| format!("{b:#x}+{m:#x} given up (overlaps a home {})", overlaps(b, m)));
            give_up();
            continue;
        }
        let frame = super::FRAMES.load(Ordering::Relaxed);
        let mut pages = Vec::new();
        if super::precache::take_written_for(b, m, &mut pages, None, 5).is_none() {
            give_up();
            continue;
        }
        let interior = pages.iter().filter(|&&p| p >= b && p + 4096 <= b + m).count();
        trace_args(|| format!("{b:#x}+{m:#x}: {} pages written ({interior} inside), known {}", pages.len(), KNOWN.lock().contains(&(b, m))));
        if interior == 0 && KNOWN.lock().contains(&(b, m)) {
            VOTED.lock().retain(|&r| r != (b, m));
            ready.push((b, m));
            continue;
        }
        let mut obs = OBSERVED.lock();
        let Some(i) = obs.iter().position(|o| (o.0, o.1) == (b, m)) else {
            // The first take: writes from before the watch started.
            if obs.len() >= 32 {
                obs.remove(0);
            }
            obs.push((b, m, frame));
            continue;
        };
        if interior > 0 {
            std::mem::drop(obs);
            drop_cpu_side(b, m, interior);
            continue;
        }
        if frame >= obs[i].2 + OBSERVE_FRAMES {
            obs.remove(i);
            std::mem::drop(obs);
            VOTED.lock().retain(|&r| r != (b, m));
            ready.push((b, m));
        }
    }
    ready
}

/// A candidate the CPU writes: never given a home.
fn drop_cpu_side(b: u64, m: u64, interior: usize) {
    VOTED.lock().retain(|&r| r != (b, m));
    OBSERVED.lock().retain(|o| (o.0, o.1) != (b, m));
    CPU_SIDE.lock().push((b, m));
    crate::info!("gpu: indirect draw arguments in {b:#x}+{m:#x}: the CPU writes {interior} of its pages, no video-memory home (read on the CPU)");
}

/// A write-watched home of exactly [b, b+n) for indirect draw arguments
/// (`ready_args_candidates`). The caller makes sure no batch holds a shadow of the
/// range any more: all GPU work writing it completed (merged into guest
/// memory, which the home copies in at its first use) and no other thread
/// records work that writes it.
pub fn create_args(b: u64, n: u64) -> bool {
    let made = create_with(b, n, true, true, "indirect draw arguments, write-watched");
    if made {
        WATCHED_ANY.store(true, Ordering::Release);
        VOTE_GEN.fetch_add(1, Ordering::AcqRel);
    }
    made
}

/// Whether any write-watched (argument) home exists.
pub fn any_watched() -> bool {
    WATCHED_ANY.load(Ordering::Acquire)
}

/// Whether [base, base+size) overlaps a write-watched (argument) home.
pub fn watched_overlaps(base: u64, size: u64) -> bool {
    any_watched() && with_snap(|r, s| r.iter().zip(s).any(|(&(b, n), (_, sh))| sh.args && b < base + size && base < b + n))
}

/// The write-watched (argument) homes' GPU-written parts (taken), then the
/// homes are dropped (`tune draw_args_gpu 0` while they exist: the caller
/// writes those parts back first; no other thread may record work that
/// binds them).
pub fn take_watched() -> Vec<(u64, u64, Arc<HomeBuf>, u64)> {
    let mut g = super::lockstat::HOMES.lock(&HOMES);
    let mut v = Vec::new();
    let mut known = KNOWN.lock();
    for h in g.iter().filter(|h| h.shared.args) {
        for (a, n) in h.stale.iter() {
            v.push((a, n, h.buf.clone(), a - h.base));
        }
        if !known.contains(&(h.base, h.size)) {
            known.push((h.base, h.size));
        }
        crate::info!("gpu: home {:#x}+{:#x} written back and dropped (tune draw_args_gpu 0)", h.base, h.size);
    }
    let before = g.len();
    g.retain(|h| !h.shared.args);
    DROPPED.fetch_add((before - g.len()) as u64, Ordering::Relaxed);
    ANY.store(!g.is_empty(), Ordering::Relaxed);
    WATCHED_ANY.store(false, Ordering::Release);
    publish_watched(&g);
    GEN.fetch_add(1, Ordering::AcqRel);
    VOTE_GEN.fetch_add(1, Ordering::AcqRel);
    v
}

/// An index buffer [base, base+size) inside a range noted by
/// `note_large_write`: that range gets a home (`create`).
pub fn index_use(base: u64, size: u64) -> bool {
    use_inside(base, size, MAX)
}

/// A large read-only buffer binding [base, base+size) inside a range noted
/// by `note_large_write`: that range gets a home (`tune home_reads`) when it
/// is at most READ_HOME_MAX (the GPU-written buffers of 4-8 MiB copied back
/// for every reading batch; not the pools of 63-180 MiB the title's CPU
/// writes too, which homes for reads of 1 MiB caught as well, 2026-10-24).
pub fn read_use(base: u64, size: u64) -> bool {
    use_inside(base, size, READ_HOME_MAX)
}

/// The largest range given a home for reads (`read_use`).
const READ_HOME_MAX: u64 = 16 << 20;

fn use_inside(base: u64, size: u64, max: u64) -> bool {
    let found = LARGE_WRITES.lock().iter().find(|&(a, n)| a <= base && base + size <= a + n);
    match found {
        Some((a, n)) if n <= max && (max == MAX || !holds_cp_code(a, n)) => create(a, n),
        Some((_, n)) if n <= max => {
            CP_CODE_REFUSED.fetch_add(1, Ordering::Relaxed);
            false
        }
        _ => false,
    }
}

/// 64 KiB pages command processors read command buffers or waited labels
/// from (`note_cp_code`, kept while homes for reads are on).
static CP_CODE_PAGES: parking_lot::RwLock<Option<std::collections::HashSet<u64>>> = parking_lot::RwLock::new(None);
/// Homes for reads refused because their range held command buffers or
/// waited labels.
pub static CP_CODE_REFUSED: AtomicU64 = AtomicU64::new(0);

/// A command processor reads command buffer words or a label [a, a+n)
/// (`Cp::before_fetch`).
pub fn note_cp_code(a: u64, n: u64) {
    let (first, last) = (a >> 16, (a + n - 1) >> 16);
    {
        let g = CP_CODE_PAGES.read();
        if g.as_ref().is_some_and(|s| (first..=last).all(|p| s.contains(&p))) {
            return;
        }
    }
    let mut g = CP_CODE_PAGES.write();
    let s = g.get_or_insert_with(std::collections::HashSet::new);
    for p in first..=last {
        s.insert(p);
    }
}

/// Whether command buffers were read from any page of [a, a+n).
fn holds_cp_code(a: u64, n: u64) -> bool {
    let g = CP_CODE_PAGES.read();
    let Some(s) = g.as_ref() else { return false };
    (a >> 16..=(a + n - 1) >> 16).any(|p| s.contains(&p))
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
    if super::tune::home_cpu_check() && !cpu_clean(b, e - b) {
        return false;
    }
    create_with(b, e - b, super::tune::homes_watch(), false, "")
}

/// Frames a home candidate is watched for CPU writes before it is made
/// (`tune home_cpu_check`).
const CPU_OBSERVE: u64 = 30;
/// Frames a range the CPU wrote stays without a home.
const CPU_REFUSE: u64 = 600;
/// Candidates being watched: (base, size, the frame of their first take).
static CPU_WATCHING: Mutex<Vec<(u64, u64, u64)>> = Mutex::new(Vec::new());
/// Ranges refused: (base, size, until frame).
static CPU_REFUSED: Mutex<Vec<(u64, u64, u64)>> = Mutex::new(Vec::new());
/// Homes refused because the CPU wrote their range while it was watched.
pub static REFUSED_CPU: AtomicU64 = AtomicU64::new(0);

/// `tune home_cpu_check`: whether [b, b+n) may get a home: its write marks
/// are taken at the first request (writes from before), and again
/// CPU_OBSERVE frames later; the CPU writing an interior page meanwhile
/// refuses it for CPU_REFUSE frames (a home over memory the title writes is
/// checked and copied in every epoch: 2026-10-26, a 63 MiB pool made the
/// vista 3-10% slower with homes for reads).
fn cpu_clean(b: u64, n: u64) -> bool {
    let frame = super::FRAMES.load(Ordering::Relaxed);
    if CPU_REFUSED.lock().iter().any(|&(rb, rn, until)| rb < b + n && b < rb + rn && frame < until) {
        return false;
    }
    if overlaps(b, n) {
        return true;
    }
    let first = {
        let mut w = CPU_WATCHING.lock();
        match w.iter().find(|x| x.0 == b && x.1 == n) {
            Some(x) if frame < x.2 + CPU_OBSERVE => return false,
            Some(x) => Some(x.2),
            None => {
                if w.len() >= 32 {
                    w.remove(0);
                }
                w.push((b, n, frame));
                None
            }
        }
    };
    let mut pages = Vec::new();
    let taken = super::precache::take_written_for(b, n, &mut pages, None, 5);
    if first.is_none() {
        // Writes from before the watch started.
        return false;
    }
    CPU_WATCHING.lock().retain(|x| !(x.0 == b && x.1 == n));
    let interior = pages.iter().filter(|&&p| p >= b && p + 4096 <= b + n).count();
    if taken.is_some() && interior == 0 {
        return true;
    }
    let mut r = CPU_REFUSED.lock();
    r.retain(|x| x.2 > frame);
    r.push((b, n, frame + CPU_REFUSE));
    REFUSED_CPU.fetch_add(1, Ordering::Relaxed);
    crate::info!("gpu: no home for {b:#x}+{n:#x} for {CPU_REFUSE} frames: the CPU wrote {interior} of its pages while it was watched");
    false
}

/// A home of exactly [b, b+n) (`watched`: checked for CPU writes, see
/// `create_args`; `args`: of indirect draw arguments).
fn create_with(b: u64, n: u64, watched: bool, args: bool, what: &str) -> bool {
    let e = b + n;
    // An eighth of the device's budget at most (BUDGET on a 24 GB GPU).
    if n > MAX || USED.load(Ordering::Relaxed) + n > super::vk::budget_share(8, BUDGET) {
        return false;
    }
    let mut g = super::lockstat::HOMES.lock(&HOMES);
    if g.iter().any(|h| h.base < e && b < h.base + h.size) {
        return false;
    }
    let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::INDIRECT_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
    let (buf, mem) = match ctx().device_buffer_addressed(n, usage) {
        Ok((buf, mem, va)) => {
            super::vk::note_gpu_range(buf, va, n, || format!("home {b:#x}+{n:#x}"));
            (buf, mem)
        }
        Err(e) => {
            crate::warn!("gpu: no video memory for a home of {b:#x}+{n:#x} ({e})");
            return false;
        }
    };
    USED.fetch_add(n, Ordering::Relaxed);
    let mut dirty = Ranges::default();
    dirty.insert(b, n);
    static NEXT_ID: AtomicU64 = AtomicU64::new(1);
    // A watched home's first check takes the marks of writes made before it
    // existed: all of it is copied in at its first use anyway.
    let wins: Box<[AtomicU64]> = (0..n.div_ceil(CHECK_WIN).max(1)).map(|_| AtomicU64::new(0)).collect();
    let shared = Arc::new(Shared { id: NEXT_ID.fetch_add(1, Ordering::Relaxed), dirty: AtomicBool::new(true), stale_gen: AtomicU64::new(0), watched, wins, base: b, checking: AtomicU64::new(0), args });
    let snap = watched.then(|| {
        let mut v = vec![0u32; (n / 4) as usize];
        guest_words(b, v.len(), &mut v);
        v
    });
    g.push(Home { base: b, size: n, buf: Arc::new(HomeBuf { buf, mem, size: n }), shared, dirty, stale: Ranges::default(), snap, resnap: Ranges::default() });
    ANY.store(true, Ordering::Relaxed);
    publish_watched(&g);
    GEN.fetch_add(1, Ordering::AcqRel);
    CREATED.fetch_add(1, Ordering::Relaxed);
    if what.is_empty() {
        crate::info!("gpu: video-memory home for {b:#x}+{n:#x} ({} MiB)", n >> 20);
    } else {
        crate::info!("gpu: video-memory home for {b:#x}+{n:#x} ({} KiB; {what})", n >> 10);
    }
    true
}

/// The stale parts (GPU-written, not yet in guest memory) of [base,
/// base+size): (guest address, bytes, home buffer, offset in the home).
/// `taken`: they will be written back (no longer stale).
pub fn stale_parts(base: u64, size: u64, taken: bool) -> Vec<(u64, u64, Arc<HomeBuf>, u64)> {
    if !any() {
        return Vec::new();
    }
    let mut g = super::lockstat::HOMES.lock(&HOMES);
    let mut v = Vec::new();
    for h in g.iter_mut() {
        if !(h.base < base + size && base < h.base + h.size) {
            continue;
        }
        for (a, n) in h.stale.intersect(base, size) {
            v.push((a, n, h.buf.clone(), a - h.base));
            if taken && h.snap.is_some() {
                h.resnap.insert(a, n);
            }
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
        && super::lockstat::HOMES.lock(&HOMES).iter().any(|h| reads.iter().any(|&(a, n)| h.stale.overlaps(a, n)))
}

/// GPU work wrote guest memory in place over part of a home (a binding that
/// only partly overlaps it): as for a write from the CPU side, the home
/// copies that part in before its next GPU use.
pub fn written_in_place(base: u64, size: u64) {
    WRITTEN_IN_PLACE.fetch_add(1, Ordering::Relaxed);
    cpu_written(base, size);
}

/// Written partial overlaps (`written_in_place`).
pub static WRITTEN_IN_PLACE: AtomicU64 = AtomicU64::new(0);

/// A write from the CPU side (CP write, file read) into guest memory: the
/// homes it overlaps copy it in before their next GPU use.
pub fn cpu_written(base: u64, size: u64) {
    if !any() || size == 0 {
        return;
    }
    let mut g = super::lockstat::HOMES.lock(&HOMES);
    for h in g.iter_mut() {
        let (a, e) = (base.max(h.base), (base + size).min(h.base + h.size));
        if a < e {
            h.dirty.insert(a, e - a);
            h.shared.dirty.store(true, Ordering::Release);
            // The CPU-side write is newer than what GPU work left there.
            h.stale.remove(a, e - a);
            h.shared.stale_gen.fetch_add(1, Ordering::AcqRel);
            // A watched home's next check takes these words as they are (a
            // write we saw, not one of the guest code's).
            if h.shared.watched {
                h.resnap.insert(a, e - a);
            }
        }
    }
}

/// The mapping of [base, base+size) changed: overlapping homes are dropped
/// (their contents with it).
pub fn remapped(base: u64, size: u64) {
    LARGE_WRITES.lock().remove(base, size);
    SHADOW_WRITES.lock().retain(|&(b, m)| !(b < base.saturating_add(size) && base < b + m));
    KNOWN.lock().retain(|&(b, m)| !(b < base.saturating_add(size) && base < b + m));
    VOTE_GEN.fetch_add(1, Ordering::AcqRel);
    if !any() {
        return;
    }
    let mut g = super::lockstat::HOMES.lock(&HOMES);
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
    publish_watched(&g);
    GEN.fetch_add(1, Ordering::AcqRel);
}

/// All homes with their GPU-written parts (taken), then dropped (`tune
/// homes 0` while homes exist: the caller writes those parts back first).
/// A queue still recording work that bound a home keeps writing it until
/// that batch is submitted; those writes are lost (A/B use only).
pub fn take_all() -> Vec<(u64, u64, Arc<HomeBuf>, u64)> {
    let mut g = super::lockstat::HOMES.lock(&HOMES);
    let mut v = Vec::new();
    for h in g.iter_mut() {
        for (a, n) in h.stale.iter() {
            v.push((a, n, h.buf.clone(), a - h.base));
        }
    }
    DROPPED.fetch_add(g.len() as u64, Ordering::Relaxed);
    g.clear();
    ANY.store(false, Ordering::Relaxed);
    publish_watched(&g);
    GEN.fetch_add(1, Ordering::AcqRel);
    v
}

pub fn status() -> String {
    format!(
        "    video-memory homes: {} created, {} refused (the CPU wrote them), {} dropped, {} MiB; {} bindings, {} bytes copied in, {} write-backs ({} KiB), {} partial overlaps ({} of them written: copied in again), {} refused for reads over command buffers or labels, {} unbounded pointer bindings over homes ({} based inside one, {} with GPU-written parts not yet in guest memory within 4 KiB of their base, {} written back first), {} image uploads over GPU-written home data not in guest memory ({} reloads of overwritten images written back first); write-watched: {} checks ({} ms, {} MiB queried; {} of them with the query outside the lock, {} bindings waited for another thread's check), {} bytes the CPU changed ({} of them by guest code in homes watched for `homes_watch`), {} written pages covered in part, {} pages other takes found written handed over\n",
        CREATED.load(Ordering::Relaxed),
        REFUSED_CPU.load(Ordering::Relaxed),
        DROPPED.load(Ordering::Relaxed),
        USED.load(Ordering::Relaxed) >> 20,
        BINDINGS.load(Ordering::Relaxed),
        COPIED_IN.load(Ordering::Relaxed),
        WRITE_BACKS.load(Ordering::Relaxed),
        WRITE_BACK_BYTES.load(Ordering::Relaxed) >> 10,
        PARTIAL.load(Ordering::Relaxed),
        WRITTEN_IN_PLACE.load(Ordering::Relaxed),
        CP_CODE_REFUSED.load(Ordering::Relaxed),
        UNBOUNDED_OVERLAPS.load(Ordering::Relaxed),
        UNBOUNDED_STALE[0].load(Ordering::Relaxed),
        UNBOUNDED_STALE[1].load(Ordering::Relaxed),
        UNBOUNDED_STALE[2].load(Ordering::Relaxed),
        IMAGE_STALE[0].load(Ordering::Relaxed),
        IMAGE_STALE[1].load(Ordering::Relaxed),
        WATCH_CHECKS.load(Ordering::Relaxed),
        WATCH_NS.load(Ordering::Relaxed) / 1_000_000,
        WATCH_BYTES.load(Ordering::Relaxed) >> 20,
        UNLOCKED_CHECKS.load(Ordering::Relaxed),
        CHECK_WAITS.load(Ordering::Relaxed),
        CPU_WRITTEN_PAGES.load(Ordering::Relaxed),
        GUEST_CHANGED.load(Ordering::Relaxed),
        EDGE_PAGES.load(Ordering::Relaxed),
        TAKEN_PAGES.load(Ordering::Relaxed)
    )
}
