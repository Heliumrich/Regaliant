//! The video-memory mirror of static guest ranges for GPU-resolved chains
//! (RENDERER.md "GPU-resolved item passes", stage G3; `tune gpu_mirror`,
//! hosted runs: `--host`). Shader-resolved buffers are read through the
//! guest page table, at first from the import windows across PCIe (the culls'
//! per-mesh data cost ~1.3 ms of GPU a frame at the vista that way). A range
//! such a buffer reaches becomes a candidate when first met (`want`); one
//! that waited AGE flips with no GPU work recorded to write it, in an import
//! window that has existed as long, is mirrored (`promote`): its whole pages
//! armed read-only through the host, then copied from its import window into
//! the pool by the submission after (`pagetable::submit_pending`), and the
//! page table points at the copy from then on. A CPU write to a mirrored page
//! faults into the host, which makes the page writable and rings it; the next
//! submission drains the ring (`drain`) and points the page back at its import
//! window before the batch that may read it. Pages disarmed by anyone else
//! (`hosted::disarm`, `forget`: imports over them, mapping changes, the copy
//! cache's protection) leave the mirror the same way (`disarmed`).
//! Pool space is never reused (it is capped): pages left behind are few.

use super::vk::ctx;
use ash::vk;
use parking_lot::Mutex;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

const PAGE: u64 = 4096;
/// Pool chunks and the pool's cap.
const CHUNK: u64 = 64 << 20;
const MAX_BYTES: u64 = 1 << 30;

/// A mirrored range of guest pages [key, hi): at `dev` in pool chunk buffer
/// `buf`, offset `off`.
#[derive(Clone, Copy)]
struct Range {
    hi: u64,
    dev: u64,
    buf: vk::Buffer,
    off: u64,
}

/// A copy queued for the next submission: import window buffer and offset,
/// pool buffer and offset, bytes.
#[derive(Clone, Copy)]
pub(super) struct Copy {
    /// The guest pages copied.
    pub lo: u64,
    pub hi: u64,
    pub src: vk::Buffer,
    pub src_off: u64,
    pub dst: vk::Buffer,
    pub dst_off: u64,
    pub bytes: u64,
}

struct State {
    ranges: BTreeMap<u64, Range>,
    chunks: Vec<(vk::Buffer, u64, u64)>,
    copies: Vec<Copy>,
    /// Page table runs queued (guest start, end, device address of start;
    /// 0: back to the import window, resolved at the submission).
    runs: Vec<(u64, u64, u64)>,
    bytes: u64,
    /// Candidates: ranges met, [key, end) and the flip they were first met
    /// at, mirrored once they have waited AGE flips with no GPU work
    /// recorded to write them (`promote`).
    young: BTreeMap<u64, (u64, u64)>,
    /// The flip `promote` last ran at.
    promoted: u64,
}

static STATE: Mutex<State> = Mutex::new(State { ranges: BTreeMap::new(), chunks: Vec::new(), copies: Vec::new(), runs: Vec::new(), bytes: 0, young: BTreeMap::new(), promoted: 0 });
/// Mirrored ranges (without the lock: `disarmed` is called on every disarm).
static RANGES_N: AtomicU64 = AtomicU64::new(0);
/// Candidates waiting.
static YOUNG_N: AtomicU64 = AtomicU64::new(0);
/// Pages mirrored, pages that left it (written, disarmed), refusals (no
/// import window, arming failed, pool full), GPU writes over mirrored pages,
/// candidates dropped for GPU writes, candidates whose import window was
/// younger than AGE flips when they were due.
static STATS: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];

/// Flips a range waits as a candidate, and flips its import window must
/// have existed, before it is armed and copied.
/// - The window: arming pages as soon as their window was imported lost
///   the device during level loads in every run (with 4 write faults in
///   all, so not the writes), while the copy cache's protection, which arms
///   pages clean for 30 flips, did not over a tour (FINDINGS.md 2026-10-26):
///   a new import's pages are presumably pinned for the GPU after
///   `vkAllocateMemory` returns, and read-only pages fail that as they fail
///   an import.
/// - The candidate: GPU work recorded to write a range before it was met
///   may still be unsubmitted, and a copy made meanwhile would precede it
///   (`gpu_write` only sees work recorded later); such work is long done
///   AGE flips later, and work recorded while the range waits drops it.
const AGE: u64 = 60;

/// Whether the mirror is in use.
pub fn on() -> bool {
    super::tune::gpu_mirror() != 0 && crate::hosted::on() && super::pagetable::made()
}

/// Pool space for `bytes`: (buffer, offset, device address).
fn alloc(st: &mut State, bytes: u64) -> Option<(vk::Buffer, u64, u64)> {
    if bytes > CHUNK {
        return None;
    }
    if let Some(c) = st.chunks.iter_mut().find(|c| c.2 + bytes <= CHUNK) {
        let off = c.2;
        c.2 += bytes;
        return Some((c.0, off, c.1 + off));
    }
    if (st.chunks.len() as u64 + 1) * CHUNK > MAX_BYTES {
        return None;
    }
    let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;
    let (buf, _mem, va) = ctx().device_buffer_addressed(CHUNK, usage).ok()?;
    st.chunks.push((buf, va, bytes));
    Some((buf, 0, va))
}

/// Guest [base, base+len) is read by GPU-resolved shaders: its whole pages
/// not mirrored yet become a candidate (`promote` mirrors it AGE flips
/// later).
pub fn want(base: u64, len: u64) {
    if len == 0 || !on() {
        return;
    }
    let lo = (base & !(PAGE - 1)).max(rdna::translate::PT_BASE);
    let hi = base.saturating_add(len).next_multiple_of(PAGE).min(rdna::translate::PT_END);
    if hi <= lo {
        return;
    }
    // A range must lie in one import window.
    if hi - lo > super::exec::window_bytes() {
        STATS[2].fetch_add(1, Ordering::Relaxed);
        return;
    }
    let now = super::FRAMES.load(Ordering::Relaxed);
    let mut st = STATE.lock();
    if st.ranges.range(..hi).next_back().is_some_and(|(&k, r)| k <= lo && r.hi >= hi) {
        return;
    }
    let e = st.young.entry(lo).or_insert((hi, now));
    if e.0 < hi {
        e.0 = hi;
    }
    YOUNG_N.store(st.young.len() as u64, Ordering::Release);
}

/// Mirrors the whole pages of guest [lo, hi) not mirrored yet, a candidate
/// that waited AGE flips, when they lie in one import window that existed
/// AGE flips and can be armed (a younger window: a candidate again).
fn mirror(lo: u64, hi: u64) {
    // The import window first (without the mirror's lock: imports take the
    // import list's lock, then disarm through `disarmed`).
    let Some((ws, we, _, src, born)) = super::exec::window_buffer_of(lo) else {
        STATS[2].fetch_add(1, Ordering::Relaxed);
        return;
    };
    if hi > we {
        STATS[2].fetch_add(1, Ordering::Relaxed);
        return;
    }
    let now = super::FRAMES.load(Ordering::Relaxed);
    let mut st = STATE.lock();
    if now.saturating_sub(born) < AGE {
        STATS[7].fetch_add(1, Ordering::Relaxed);
        st.young.entry(lo).or_insert((hi, now));
        YOUNG_N.store(st.young.len() as u64, Ordering::Release);
        return;
    }
    // The gaps of [lo, hi) no range covers.
    let mut gaps = Vec::new();
    let mut a = lo;
    if let Some((&k, r)) = st.ranges.range(..=lo).next_back() {
        if k <= lo && r.hi > lo {
            a = r.hi;
        }
    }
    for (&k, r) in st.ranges.range(lo..hi) {
        if k > a {
            gaps.push((a, k));
        }
        a = a.max(r.hi);
    }
    if a < hi {
        gaps.push((a, hi));
    }
    for (a, b) in gaps {
        let Some((buf, off, dev)) = alloc(&mut st, b - a) else {
            STATS[4].fetch_add(1, Ordering::Relaxed);
            continue;
        };
        // Armed before the copy: a write after it faults and the page goes
        // back to its window before any batch reads the copy. (`tune
        // gpu_mirror` 3, diagnosis: not armed; 2: not read from the mirror.)
        // Diagnosis modes: 4 armed only, 5 copied only (neither read).
        let mode = super::tune::gpu_mirror();
        if !matches!(mode, 3 | 5) && crate::hosted::arm(a, b - a).is_err() {
            STATS[3].fetch_add(1, Ordering::Relaxed);
            continue;
        }
        if mode != 4 {
            st.copies.push(Copy { lo: a, hi: b, src, src_off: a - ws, dst: buf, dst_off: off, bytes: b - a });
        }
        if !matches!(mode, 2 | 4 | 5) {
            st.runs.push((a, b, dev));
        }
        st.ranges.insert(a, Range { hi: b, dev, buf, off });
        st.bytes += b - a;
        STATS[0].fetch_add((b - a) / PAGE, Ordering::Relaxed);
    }
    RANGES_N.store(st.ranges.len() as u64, Ordering::Release);
}

/// Pages [lo, hi) leave the mirror (written or disarmed): their ranges are
/// cut, and their page table entries go back to their import windows.
fn leave(st: &mut State, lo: u64, hi: u64) {
    let keys: Vec<u64> = st.ranges.range(..hi).rev().take_while(|(_, r)| r.hi > lo).map(|(&k, _)| k).collect();
    for k in keys {
        let r = st.ranges.remove(&k).unwrap();
        let (cut_lo, cut_hi) = (k.max(lo), r.hi.min(hi));
        if k < cut_lo {
            st.ranges.insert(k, Range { hi: cut_lo, ..r });
        }
        if cut_hi < r.hi {
            let d = cut_hi - k;
            st.ranges.insert(cut_hi, Range { hi: r.hi, dev: r.dev + d, buf: r.buf, off: r.off + d });
        }
        st.runs.push((cut_lo, cut_hi, 0));
        STATS[1].fetch_add((cut_hi - cut_lo) / PAGE, Ordering::Relaxed);
    }
    RANGES_N.store(st.ranges.len() as u64, Ordering::Release);
}

/// Once a flip (`take`): the candidates that waited AGE flips are mirrored.
fn promote() {
    let now = super::FRAMES.load(Ordering::Relaxed);
    let due: Vec<(u64, u64)> = {
        let mut st = STATE.lock();
        if st.promoted == now {
            return;
        }
        st.promoted = now;
        let keys: Vec<u64> = st.young.iter().filter(|&(_, &(_, since))| now.saturating_sub(since) >= AGE).map(|(&k, _)| k).collect();
        let due = keys.into_iter().map(|k| (k, st.young.remove(&k).unwrap().0)).collect();
        YOUNG_N.store(st.young.len() as u64, Ordering::Release);
        due
    };
    for (a, b) in due {
        mirror(a, b);
    }
}

/// Disarmed ranges not handled yet (`disarmed` runs where the import list's
/// lock may be held: the mirror's own lock is taken later, in `take`).
static DISARMED: Mutex<Vec<(u64, u64)>> = Mutex::new(Vec::new());

/// Guest [base, base+len) was disarmed (`hosted::disarm`, `forget`): its
/// mirrored pages leave the mirror at the next submission (before it).
pub fn disarmed(base: u64, len: u64) {
    if RANGES_N.load(Ordering::Acquire) == 0 {
        return;
    }
    let (lo, hi) = (base & !(PAGE - 1), base.saturating_add(len).next_multiple_of(PAGE));
    DISARMED.lock().push((lo, hi));
}

/// The host's write faults since the last call: mirrored pages written
/// leave the mirror (a lost ring: every range whose pages are not all
/// armed). Then the copies and page table runs queued, and the runs that
/// put back the mirrored ranges overlapping `over` (guest ranges whose
/// entries this submission rewrites for their import windows).
pub(super) fn take(over: &[(u64, u64)]) -> (Vec<Copy>, Vec<(u64, u64, u64)>) {
    if YOUNG_N.load(Ordering::Acquire) != 0 {
        promote();
    }
    if RANGES_N.load(Ordering::Acquire) == 0 && STATE.lock().runs.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let mut faults = Vec::new();
    let complete = crate::hosted::drain_faults(&mut faults);
    let disarmed = std::mem::take(&mut *DISARMED.lock());
    let mut st = STATE.lock();
    for &(page, _, _) in &faults {
        leave(&mut st, page, page + PAGE);
    }
    for (lo, hi) in disarmed {
        leave(&mut st, lo, hi);
    }
    if !complete {
        let lost: Vec<(u64, u64)> = st.ranges.iter().filter(|&(&k, r)| !crate::hosted::all_armed(k, r.hi - k)).map(|(&k, r)| (k, r.hi)).collect();
        for (a, b) in lost {
            leave(&mut st, a, b);
        }
    }
    // A copy whose import window was dropped since it was queued is not
    // made (the window may be freed before it would run): its pages leave
    // the mirror. A window still valid now outlives the batch about to be
    // submitted, and the copy runs before it.
    let mut copies = std::mem::take(&mut st.copies);
    // (A window is the same one by its buffer, start and size: handles of
    // freed windows are reused by later imports.)
    let ok = |k: &Copy| super::exec::window_valid(k.src, k.lo - k.src_off, k.src_off + k.bytes);
    let stale: Vec<(u64, u64)> = copies.iter().filter(|k| !ok(k)).map(|k| (k.lo, k.hi)).collect();
    if !stale.is_empty() {
        copies.retain(ok);
        STATS[2].fetch_add(stale.len() as u64, Ordering::Relaxed);
        for (a, b) in stale {
            leave(&mut st, a, b);
        }
    }
    let mut runs = std::mem::take(&mut st.runs);
    if !matches!(super::tune::gpu_mirror(), 2 | 4 | 5) {
        for &(a, b) in over {
            for (&k, r) in st.ranges.range(..b) {
                if r.hi > a {
                    runs.push((k, r.hi, r.dev));
                }
            }
        }
    }
    (copies, runs)
}

/// GPU work recorded to write guest [a, a+n) (`Exec::note_write`): the
/// candidates over it are dropped, and mirrored pages in it are disarmed and
/// leave the mirror before the work runs (its writes go to guest memory).
/// The first such writes over mirrored pages are logged with their shader.
pub fn gpu_write(a: u64, n: u64, shader: u64, queue: &str) {
    if (RANGES_N.load(Ordering::Acquire) == 0 && YOUNG_N.load(Ordering::Acquire) == 0) || n == 0 {
        return;
    }
    let (lo, hi) = (a & !(PAGE - 1), a.saturating_add(n).next_multiple_of(PAGE));
    let hit = {
        let mut st = STATE.lock();
        if !st.young.is_empty() {
            // Candidates are not larger than an import window.
            let from = lo.saturating_sub(super::exec::window_bytes());
            let drop: Vec<u64> = st.young.range(from..hi).filter(|&(_, &(e, _))| e > lo).map(|(&k, _)| k).collect();
            if !drop.is_empty() {
                STATS[6].fetch_add(drop.len() as u64, Ordering::Relaxed);
                for k in drop {
                    st.young.remove(&k);
                }
                YOUNG_N.store(st.young.len() as u64, Ordering::Release);
            }
        }
        st.ranges.range(..hi).next_back().is_some_and(|(_, r)| r.hi > lo)
    };
    if !hit {
        return;
    }
    let k = STATS[5].fetch_add(1, Ordering::Relaxed);
    if k < 20 {
        crate::info!("gpu[{queue}]: GPU-resolved mirror: GPU work (shader {shader:#x}) writes {a:#x}+{n:#x} over mirrored pages: disarmed");
    }
    crate::hosted::disarm(lo, hi - lo);
}

/// Status part ("" when off).
pub fn status() -> String {
    if !on() {
        return String::new();
    }
    let st = STATE.lock();
    format!(
        "    GPU-resolved mirror: {} ranges, {} MiB of pool ({} chunks); pages mirrored {}, left it {}; refused: no import window {}, arming failed {}, pool full {}; GPU writes over mirrored pages {}; candidates waiting {}, dropped for GPU writes {}, due in a young window {}\n",
        st.ranges.len(),
        st.bytes >> 20,
        st.chunks.len(),
        STATS[0].load(Ordering::Relaxed),
        STATS[1].load(Ordering::Relaxed),
        STATS[2].load(Ordering::Relaxed),
        STATS[3].load(Ordering::Relaxed),
        STATS[4].load(Ordering::Relaxed),
        STATS[5].load(Ordering::Relaxed),
        st.young.len(),
        STATS[6].load(Ordering::Relaxed),
        STATS[7].load(Ordering::Relaxed)
    )
}
