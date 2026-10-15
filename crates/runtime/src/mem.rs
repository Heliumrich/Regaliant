//! Guest virtual memory.
//!
//! The PS4 address range used by the guest is reserved once in the host process
//! and pages are committed/decommitted at 4 KiB granularity inside it, so guest
//! pointers are host pointers. A VMA map mirrors what the guest has mapped, and a
//! separate allocator models PS4 "direct memory" (physical) ranges.
//!
//! Limitation (reported, never silent): mapping the same direct-memory range at
//! two virtual addresses (aliasing) is not shared; it is detected and logged.
//! With `DS_SECTIONS=1` guest memory is section-backed instead (`sections`):
//! mappings are views, physical pages keep their contents across remaps and
//! aliases share them, and the GPU imports alias views.

use parking_lot::Mutex;
use std::collections::BTreeMap;
use windows_sys::Win32::System::Memory::*;

pub const GUEST_BASE: u64 = 0x4_0000_0000; // 16 GiB
pub const GUEST_END: u64 = 0x800_0000_0000; // 8 TiB (the title reserves ~1 TiB for virtual textures)
pub const MODULE_BASE: u64 = 0x8_0000_0000; // eboot image base (matches reference loaders)
pub const MODULE_END: u64 = 0x10_0000_0000;
pub const MMAP_BASE: u64 = 0x10_0000_0000; // default search start for guest mappings
pub const INTERNAL_BASE: u64 = 0x4_0000_0000; // runtime-owned guest-visible memory
pub const PAGE: u64 = 0x4000; // PS5 page size
/// Direct memory available to the title: PS5 game memory of 13.5 GiB minus the
/// default 1 GiB flexible budget (values from public reverse engineering, unconfirmed).
pub const DMEM_SIZE: u64 = 12800 << 20;

pub const PROT_CPU_READ: u32 = 0x1;
pub const PROT_CPU_WRITE: u32 = 0x2;
pub const PROT_CPU_EXEC: u32 = 0x4;
pub const PROT_GPU_READ: u32 = 0x10;
pub const PROT_GPU_WRITE: u32 = 0x20;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VmaKind {
    Reserved,
    Direct,
    Flexible,
    Module,
    Internal,
}

#[derive(Clone, Debug)]
pub struct Vma {
    pub start: u64,
    pub size: u64,
    pub kind: VmaKind,
    pub prot: u32,
    pub phys: u64,
    pub mtype: i32,
    pub name: String,
    /// Section-backed memory (`sections`): the alias slot of its pages
    /// (NO_SLOT: none) and, for memory other than direct, its offset in the
    /// anonymous section.
    pub slot: u8,
    pub off: u64,
}

/// `Vma::slot` without an alias.
pub const NO_SLOT: u8 = u8::MAX;
/// The anonymous section's size (`sections`): flexible memory, modules and
/// runtime-owned guest memory.
const ANON_SIZE: u64 = 4 << 30;

impl Vma {
    pub fn end(&self) -> u64 {
        self.start + self.size
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DmemBlock {
    pub size: u64,
    pub mtype: i32,
}

pub struct MemState {
    pub vmas: BTreeMap<u64, Vma>,
    pub dmem: BTreeMap<u64, DmemBlock>,
    pub flexible_used: u64,
    internal_cursor: u64,
}

/// Direct-memory mappings whose contents live with the physical pages on the
/// console but with the virtual pages here (diagnostics): `in_place` maps the
/// same physical pages again over their own mapping (we zero them), `moved`
/// maps physical pages last mapped at another address (their data stays
/// there). `released`: physical start -> (address, size) of unmapped ranges.
struct RemapStats {
    in_place: u64,
    in_place_bytes: u64,
    moved: u64,
    moved_bytes: u64,
    logged: u32,
    released: BTreeMap<u64, (u64, u64)>,
}

static REMAP: Mutex<RemapStats> = Mutex::new(RemapStats { in_place: 0, in_place_bytes: 0, moved: 0, moved_bytes: 0, logged: 0, released: BTreeMap::new() });

/// `readable_span` calls and cache misses (diagnostics). Calls are counted
/// per thread and added here every SPAN_CALLS_BATCH (one shared counter
/// bumped by every recording thread ~2.3 million times per second was
/// 0.5-1% of their busy samples, run 20261013-115322).
static SPAN_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
const SPAN_CALLS_BATCH: u32 = 1024;
static SPAN_MISSES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// One status line (diagnostics).
/// Contents of unmapped direct memory kept with its physical pages
/// (`DS_PHYS_CARRY=1`). On the console a physical page keeps its data while
/// unmapped and shows it wherever it is mapped next; here a mapping's pages
/// are private to its addresses, so the title's remaps of physical memory to
/// new addresses (~27k per tour, 8 GB, mostly `sceKernelBatchMap`) arrived
/// zeroed. Unmapping parks the non-zero 4 KiB pages by physical address
/// (`park`), the next mapping of them copies them in (`unpark`), releasing
/// the physical memory drops them (`drop_parked`).
static PARKED: Mutex<BTreeMap<u64, Box<[u8]>>> = Mutex::new(BTreeMap::new());
/// [pages parked, zero pages left out, pages copied into a new mapping,
/// pages dropped by releases, bytes held, pages not parked (cap)].
pub static PARK_STATS: [std::sync::atomic::AtomicU64; 6] = [const { std::sync::atomic::AtomicU64::new(0) }; 6];
/// Most bytes kept parked.
const PARK_CAP: u64 = 4 << 30;
/// Direct mappings whose address, size and physical offset are all 64 KiB
/// aligned, 16 KiB aligned, or neither.
static MAP_ALIGN: [std::sync::atomic::AtomicU64; 6] = [const { std::sync::atomic::AtomicU64::new(0) }; 6];
const PARK_PAGE: u64 = 4096;

/// `DS_PHYS_CARRY=1`: unmapped direct memory keeps its contents (`PARKED`).
pub fn phys_carry() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| env_is("DS_PHYS_CARRY", "1"))
}

/// Parks the non-zero pages of the direct mapping [va, va+size) of physical
/// `phys` (before its pages are decommitted).
fn park(va: u64, phys: u64, size: u64) {
    use std::sync::atomic::Ordering::Relaxed;
    if !os_readable(va, size) {
        return;
    }
    let mut g = PARKED.lock();
    let mut off = 0;
    while off < size {
        let page = unsafe { std::slice::from_raw_parts((va + off) as *const u64, (PARK_PAGE / 8) as usize) };
        let key = phys + off;
        if page.iter().all(|&w| w == 0) {
            if g.remove(&key).is_some() {
                PARK_STATS[4].fetch_sub(PARK_PAGE, Relaxed);
            }
            PARK_STATS[1].fetch_add(1, Relaxed);
        } else if PARK_STATS[4].load(Relaxed) + PARK_PAGE > PARK_CAP && !g.contains_key(&key) {
            PARK_STATS[5].fetch_add(1, Relaxed);
        } else {
            let bytes = unsafe { std::slice::from_raw_parts((va + off) as *const u8, PARK_PAGE as usize) };
            if g.insert(key, bytes.into()).is_none() {
                PARK_STATS[4].fetch_add(PARK_PAGE, Relaxed);
            }
            PARK_STATS[0].fetch_add(1, Relaxed);
        }
        off += PARK_PAGE;
    }
}

/// Copies the parked pages of physical [phys, phys+size) into the new
/// direct mapping at `va` (protection `prot`), and forgets them.
fn unpark(va: u64, phys: u64, size: u64, prot: u32) {
    use std::sync::atomic::Ordering::Relaxed;
    let mut g = PARKED.lock();
    let keys: Vec<u64> = g.range(phys..phys + size).map(|(k, _)| *k).collect();
    if keys.is_empty() {
        return;
    }
    let writable = matches!(win_prot(prot), PAGE_READWRITE | PAGE_EXECUTE_READWRITE);
    let mut old = 0u32;
    if !writable {
        unsafe { VirtualProtect(va as _, size as usize, PAGE_READWRITE, &mut old) };
    }
    for k in keys {
        if let Some(b) = g.remove(&k) {
            unsafe { std::ptr::copy_nonoverlapping(b.as_ptr(), (va + (k - phys)) as *mut u8, PARK_PAGE as usize) };
            PARK_STATS[4].fetch_sub(PARK_PAGE, Relaxed);
            PARK_STATS[2].fetch_add(1, Relaxed);
        }
    }
    if !writable {
        unsafe { VirtualProtect(va as _, size as usize, win_prot(prot), &mut old) };
    }
}

/// Forgets the parked pages of released physical [start, end).
fn drop_parked(start: u64, end: u64) {
    use std::sync::atomic::Ordering::Relaxed;
    let mut g = PARKED.lock();
    let keys: Vec<u64> = g.range(start..end).map(|(k, _)| *k).collect();
    for k in keys {
        g.remove(&k);
        PARK_STATS[4].fetch_sub(PARK_PAGE, Relaxed);
        PARK_STATS[3].fetch_add(1, Relaxed);
    }
}

pub fn remap_status() -> String {
    let r = REMAP.lock();
    let o = std::sync::atomic::Ordering::Relaxed;
    let carry = if phys_carry() {
        let v: Vec<u64> = PARK_STATS.iter().map(|c| c.load(o)).collect();
        format!(
            "; contents carried (DS_PHYS_CARRY): {} pages parked ({} zero left out), {} copied into later mappings, {} dropped by releases, {} MiB held, {} not parked (cap)",
            v[0],
            v[1],
            v[2],
            v[3],
            v[4] >> 20,
            v[5]
        )
    } else {
        String::new()
    };
    format!(
        "    direct memory remaps: {} in place ({} MiB, zeroed here), {} of pages last mapped elsewhere ({} MiB, contents not carried); mapping changes {}; readable-range checks {} ({} cache misses); decommits deferred under GPU imports {} ({} MiB; {} done later, {} mapped again first, {} waiting)
",
        r.in_place,
        r.in_place_bytes >> 20,
        r.moved,
        r.moved_bytes >> 20,
        MAP_GEN.load(o),
        SPAN_CALLS.load(o),
        SPAN_MISSES.load(o),
        DEFERRED_STATS[0].load(o),
        DEFERRED_STATS[3].load(o) >> 20,
        DEFERRED_STATS[1].load(o),
        DEFERRED_STATS[2].load(o),
        DEFERRED_LEN.load(o)
    )
    .replace(
        " waiting)\n",
        &format!(
            " waiting){carry}; direct mappings aligned to 64 KiB {}, to 16 KiB {}, less {}; other mappings {}, {}, {}\n",
            MAP_ALIGN[0].load(o),
            MAP_ALIGN[1].load(o),
            MAP_ALIGN[2].load(o),
            MAP_ALIGN[3].load(o),
            MAP_ALIGN[4].load(o),
            MAP_ALIGN[5].load(o)
        ),
    )
}

/// Incremented whenever guest mappings or their protections change (inside
/// the MEM lock): per-thread caches of readable ranges are valid for one
/// value (`readable_span`).
pub static MAP_GEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The latest mapping changes: (generation it made, start, end), oldest
/// first. A thread's cached readable runs that no change since its
/// generation touched stay valid: the title's texture streaming remaps
/// 64 KiB pieces many times per second, and every change used to empty
/// every command processor's cache (`readable_span` was ~12% of the
/// graphics thread's busy samples, run 20261011-111123).
static CHANGES: Mutex<std::collections::VecDeque<(u64, u64, u64)>> = Mutex::new(std::collections::VecDeque::new());
const CHANGES_KEEP: usize = 256;

/// Notes a mapping change of [start, end) (inside the MEM lock).
fn bump_gen(start: u64, end: u64) {
    let mut c = CHANGES.lock();
    let generation = MAP_GEN.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
    if c.len() >= CHANGES_KEEP {
        c.pop_front();
    }
    c.push_back((generation, start, end));
}

/// Brings a thread's cached runs from generation `from` to `to`: drops the
/// runs a change in between touched; false when the changes are not all
/// known any more (the caller empties the cache).
fn revalidate(runs: &mut Vec<(u64, u64, bool)>, from: u64, to: u64) -> bool {
    if from == u64::MAX {
        return false;
    }
    let c = CHANGES.lock();
    if c.front().is_none_or(|&(g, _, _)| g > from + 1) {
        return false;
    }
    for &(g, s, e) in c.iter() {
        if g > from && g <= to {
            runs.retain(|&(rs, re, _)| e <= rs || re <= s);
        }
    }
    true
}

/// Cached runs per thread.
const READABLE_RUNS: usize = 64;

thread_local! {
    /// This thread's recently confirmed runs: [start, end) and whether they
    /// are readable (unmapped gaps are kept too: descriptors of unbound
    /// resources point there again and again), sorted by start and
    /// disjoint; valid for one MAP_GEN value (`revalidate` carries them
    /// over changes elsewhere).
    static READABLE: std::cell::RefCell<(u64, Vec<(u64, u64, bool)>)> = const { std::cell::RefCell::new((u64::MAX, Vec::new())) };
    /// This thread's `readable_span` calls not yet added to SPAN_CALLS.
    static SPAN_CALLS_LOCAL: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

/// The run of adjacent accessible mappings containing `a` as [start, end)
/// (at most 64 mappings either side), from a per-thread cache when the
/// mappings have not changed since (the GPU command processors check every
/// descriptor and buffer they read, ~2.3 million times per second in the
/// courtyard; the mapping table lock is shared with the title's threads).
pub fn readable_span(a: u64) -> Option<(u64, u64)> {
    SPAN_CALLS_LOCAL.with(|n| {
        let k = n.get() + 1;
        if k == SPAN_CALLS_BATCH {
            SPAN_CALLS.fetch_add(k as u64, std::sync::atomic::Ordering::Relaxed);
            n.set(0);
        } else {
            n.set(k);
        }
    });
    let generation = MAP_GEN.load(std::sync::atomic::Ordering::Acquire);
    let hit = READABLE.with(|c| {
        let mut c = c.borrow_mut();
        if c.0 != generation {
            let from = c.0;
            if !crate::gpu::tune::map_ranges() || !revalidate(&mut c.1, from, generation) {
                c.1.clear();
            }
            c.0 = generation;
        }
        let i = c.1.partition_point(|r| r.0 <= a);
        i.checked_sub(1).map(|i| c.1[i]).filter(|r| a < r.1)
    });
    if let Some((s, e, ok)) = hit {
        return ok.then_some((s, e));
    }
    SPAN_MISSES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let (start, end, ok) = {
        let st = MEM.lock();
        let usable = |v: &Vma| v.kind != VmaKind::Reserved && v.prot != 0;
        match st.find(a) {
            Some(v) if usable(v) => {
                let (mut start, mut end) = (v.start, v.end());
                for _ in 0..64 {
                    match st.vmas.range(..start).next_back().map(|(_, p)| p).filter(|p| p.end() == start && usable(p)) {
                        Some(p) => start = p.start,
                        None => break,
                    }
                }
                for _ in 0..64 {
                    match st.vmas.get(&end).filter(|p| usable(p)) {
                        Some(p) => end = p.end(),
                        None => break,
                    }
                }
                (start, end, true)
            }
            // Reserved or inaccessible: that mapping.
            Some(v) => (v.start, v.end(), false),
            // Between mappings: the gap.
            None => {
                let lo = st.vmas.range(..=a).next_back().map(|(_, p)| p.end()).filter(|&e| e <= a).unwrap_or(a);
                let hi = st.vmas.range(a..).next().map(|(&k, _)| k).filter(|&k| k > a).unwrap_or(a + 1);
                (lo, hi, false)
            }
        }
    };
    READABLE.with(|c| {
        let mut c = c.borrow_mut();
        if c.0 != generation {
            c.0 = generation;
            c.1.clear();
        }
        // Newer runs replace the ones they overlap.
        c.1.retain(|r| r.1 <= start || end <= r.0);
        if c.1.len() >= READABLE_RUNS {
            c.1.clear();
        }
        let i = c.1.partition_point(|r| r.0 < start);
        c.1.insert(i, (start, end, ok));
    });
    ok.then_some((start, end))
}

/// Whether guest [a, a+len) is accessible (`readable_span`, else `mapped`).
pub fn readable(a: u64, len: u64) -> bool {
    let end = a.saturating_add(len.max(1));
    match readable_span(a) {
        Some((_, e)) if end <= e => true,
        Some(_) => mapped(a, len),
        None => false,
    }
}

/// Guest ranges whose host pages were committed or decommitted since the
/// GPU executor last looked: its imports of guest memory pin the pages they
/// were made with, so an import over a range that was remapped (even with
/// identical bounds, as the title's sceKernelBatchMap streaming does) is stale.
static REPAGED: Mutex<Vec<(u64, u64)>> = Mutex::new(Vec::new());
static REPAGED_PENDING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn repaged(addr: u64, size: u64) {
    let mut r = REPAGED.lock();
    if r.len() >= 1024 {
        // Nobody collected them (no GPU work in between): one covering range.
        let (lo, hi) = r.iter().fold((addr, addr + size), |(a, b), &(s, e)| (a.min(s), b.max(e)));
        r.clear();
        r.push((lo, hi));
    } else {
        r.push((addr, addr + size));
    }
    REPAGED_PENDING.store(true, std::sync::atomic::Ordering::Release);
}

/// Takes the ranges whose pages changed since the last call ([start, end)).
pub fn take_repaged() -> Vec<(u64, u64)> {
    if !REPAGED_PENDING.swap(false, std::sync::atomic::Ordering::Acquire) {
        return Vec::new();
    }
    std::mem::take(&mut *REPAGED.lock())
}

/// Whether guest [a, a+len) is covered by accessible mappings (from the
/// mapping table: VirtualQuery walks page tables and is very slow on the
/// title's multi-GiB mappings).
pub fn mapped(a: u64, len: u64) -> bool {
    let st = MEM.lock();
    let end = a.saturating_add(len.max(1));
    let mut p = a;
    while p < end {
        match st.find(p) {
            Some(v) if v.kind != VmaKind::Reserved && v.prot != 0 => p = v.end(),
            _ => return false,
        }
    }
    true
}

/// The mapped, readable mapping containing `a` as [start, end), if any.
pub fn mapped_span(a: u64) -> Option<(u64, u64)> {
    let st = MEM.lock();
    st.find(a).filter(|v| v.kind != VmaKind::Reserved && v.prot != 0).map(|v| (v.start, v.end()))
}

pub static MEM: Mutex<MemState> = Mutex::new(MemState {
    vmas: BTreeMap::new(),
    dmem: BTreeMap::new(),
    flexible_used: 0,
    internal_cursor: INTERNAL_BASE,
});

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemError {
    NoSpace,
    Invalid,
    Busy,
    HostFailure(u32),
}

pub fn align_up(v: u64, a: u64) -> u64 {
    if a <= 1 { v } else { (v + a - 1) & !(a - 1) }
}
pub fn align_down(v: u64, a: u64) -> u64 {
    if a <= 1 { v } else { v & !(a - 1) }
}

pub fn win_prot(prot: u32) -> u32 {
    let cpu = prot & 7;
    let gpu_rw = prot & (PROT_GPU_READ | PROT_GPU_WRITE) != 0;
    let exec = cpu & PROT_CPU_EXEC != 0;
    let write = cpu & PROT_CPU_WRITE != 0 || gpu_rw;
    let read = cpu != 0 || gpu_rw;
    match (read, write, exec) {
        (_, true, true) => PAGE_EXECUTE_READWRITE,
        (_, false, true) => PAGE_EXECUTE_READ,
        (_, true, false) => PAGE_READWRITE,
        (true, false, false) => PAGE_READONLY,
        (false, false, false) => PAGE_NOACCESS,
    }
}

fn last_error() -> u32 {
    unsafe { windows_sys::Win32::Foundation::GetLastError() }
}

static RESERVED_EARLY: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// The guest range was reserved with write watching (`take_written`).
static WRITE_WATCH: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// Why a requested write-watched reservation failed (Win32 error; 0: none).
static WRITE_WATCH_ERROR: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// Whether environment variable `name` (ASCII) equals `value`, read without
/// allocating (the heap may not be set up yet).
pub fn env_is(name: &str, value: &str) -> bool {
    use windows_sys::Win32::System::Environment::GetEnvironmentVariableW;
    let mut n = [0u16; 64];
    if name.len() >= n.len() {
        return false;
    }
    for (d, c) in n.iter_mut().zip(name.bytes()) {
        *d = c as u16;
    }
    let mut buf = [0u16; 64];
    let len = unsafe { GetEnvironmentVariableW(n.as_ptr(), buf.as_mut_ptr(), buf.len() as u32) } as usize;
    len == value.len() && buf[..len].iter().zip(value.bytes()).all(|(&a, b)| a == b as u16)
}

/// Reserves the whole guest range once, allocating nothing: called by
/// `reserve_guest_range` at the start of `main` and by the first heap
/// allocation should one come earlier (`heap`). With write watching (the
/// system notes which pages the CPU writes; `take_written`) unless
/// `DS_WRITE_WATCH=0`.
pub fn reserve_guest_range_early() {
    if RESERVED_EARLY.load(std::sync::atomic::Ordering::Acquire) {
        return;
    }
    let size = (GUEST_END - GUEST_BASE) as usize;
    // Section-backed guest memory (no write watching: views have none).
    if env_is("DS_SECTIONS", "1") && crate::sections::reserve(GUEST_BASE, size as u64, DMEM_SIZE, ANON_SIZE).is_ok() {
        RESERVED_EARLY.store(true, std::sync::atomic::Ordering::Release);
        return;
    }
    if !env_is("DS_WRITE_WATCH", "0") {
        let p = unsafe { VirtualAlloc(GUEST_BASE as _, size, MEM_RESERVE | MEM_WRITE_WATCH, PAGE_NOACCESS) };
        if !p.is_null() {
            WRITE_WATCH.store(true, std::sync::atomic::Ordering::Release);
            RESERVED_EARLY.store(true, std::sync::atomic::Ordering::Release);
            return;
        }
        WRITE_WATCH_ERROR.store(last_error(), std::sync::atomic::Ordering::Release);
    }
    let p = unsafe { VirtualAlloc(GUEST_BASE as _, size, MEM_RESERVE, PAGE_NOACCESS) };
    if !p.is_null() {
        RESERVED_EARLY.store(true, std::sync::atomic::Ordering::Release);
    }
}

/// VirtualAlloc's MEM_WRITE_WATCH and GetWriteWatch's WRITE_WATCH_FLAG_RESET
/// (not named by windows-sys).
const MEM_WRITE_WATCH: u32 = 0x0020_0000;
const WRITE_WATCH_FLAG_RESET: u32 = 1;

/// Whether CPU writes to the guest range are tracked (`take_written`):
/// write watching, or page protection for section-backed memory.
pub fn write_watched() -> bool {
    WRITE_WATCH.load(std::sync::atomic::Ordering::Acquire) || crate::sections::on()
}

/// Write watching of the guest range, for the start-up log.
pub fn write_watch_state() -> String {
    match (write_watched(), WRITE_WATCH_ERROR.load(std::sync::atomic::Ordering::Acquire)) {
        (true, _) => "on".into(),
        (false, 0) => "off".into(),
        (false, e) => format!("off (reserving with write watch failed: error {e})"),
    }
}

/// Statistics of `take_written`: [calls, ns, pages reported].
pub static WRITE_WATCH_STATS: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// The 4 KiB pages of guest [base, base+size) the CPU wrote since they were
/// last taken (their write marks are reset), into `out` (page addresses);
/// None without write watching or when the query fails (treat as written).
pub fn take_written(base: u64, size: u64, out: &mut Vec<u64>) -> Option<()> {
    use std::sync::atomic::Ordering::Relaxed;
    if crate::sections::on() {
        let t = std::time::Instant::now();
        let r = crate::gpu::resident::take_written(base, size, out);
        WRITE_WATCH_STATS[0].fetch_add(1, Relaxed);
        WRITE_WATCH_STATS[1].fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
        WRITE_WATCH_STATS[2].fetch_add(out.len() as u64, Relaxed);
        return r;
    }
    if !write_watched() || size == 0 {
        return None;
    }
    let (lo, hi) = (base & !4095, (base + size).next_multiple_of(4096));
    let t = std::time::Instant::now();
    let mut addrs = [std::ptr::null_mut::<std::ffi::c_void>(); 256];
    let mut at = lo;
    while at < hi {
        let mut count: usize = addrs.len();
        let mut gran = 0u32;
        let r = unsafe { GetWriteWatch(WRITE_WATCH_FLAG_RESET, at as _, (hi - at) as usize, addrs.as_mut_ptr(), &mut count, &mut gran) };
        if r != 0 {
            return None;
        }
        out.extend(addrs[..count].iter().map(|&a| a as u64));
        if count < addrs.len() {
            break;
        }
        // More than fit: continue after the last page reported.
        at = addrs[count - 1] as u64 + 4096;
    }
    WRITE_WATCH_STATS[0].fetch_add(1, Relaxed);
    WRITE_WATCH_STATS[1].fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
    WRITE_WATCH_STATS[2].fetch_add(out.len() as u64, Relaxed);
    Some(())
}

/// Reserves the whole guest range (`reserve_guest_range_early`). Must run
/// before anything else allocates widely.
pub fn reserve_guest_range() -> Result<(), String> {
    reserve_guest_range_early();
    if !RESERVED_EARLY.load(std::sync::atomic::Ordering::Acquire) {
        let err = last_error();
        // Find what occupies the range for a useful diagnostic.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        let mut addr = GUEST_BASE;
        let mut busy = Vec::new();
        while addr < GUEST_END {
            let n = unsafe { VirtualQuery(addr as _, &mut info, std::mem::size_of::<MEMORY_BASIC_INFORMATION>()) };
            if n == 0 {
                break;
            }
            if info.State != MEM_FREE {
                busy.push(format!("{:#x}+{:#x}", info.BaseAddress as u64, info.RegionSize));
            }
            addr = info.BaseAddress as u64 + info.RegionSize as u64;
        }
        return Err(format!("reserving guest range failed (error {err}); occupied: {busy:?}"));
    }
    Ok(())
}

fn host_commit(addr: u64, size: u64, prot: u32) -> Result<(), MemError> {
    crate::gpu::track::remapped(addr, size, crate::gpu::track::REMAP_COMMIT);
    repaged(addr, size);
    // Parts whose decommit was deferred are still committed with their old
    // contents: a new mapping starts zeroed, as after a decommit.
    let stale = cancel_deferred(addr, addr + size);
    if !stale.is_empty() {
        let _readers_out = GUEST_READERS.write();
        for &(a, e) in &stale {
            let p = unsafe { VirtualAlloc(a as _, (e - a) as usize, MEM_COMMIT, PAGE_READWRITE) };
            if !p.is_null() {
                unsafe { std::ptr::write_bytes(a as *mut u8, 0, (e - a) as usize) };
            }
        }
    }
    let p = unsafe { VirtualAlloc(addr as _, size as usize, MEM_COMMIT, win_prot(prot)) };
    if p.is_null() { Err(MemError::HostFailure(last_error())) } else { Ok(()) }
}

/// Guest ranges [start, end) unmapped while a GPU import covered them: their
/// host pages stay committed until no import does (`retry_deferred`). A
/// decommit under an import that batches in flight still used lost the
/// device at level loads (the title unmaps memory it is done with inside
/// the 64 MiB windows the GPU imports; FINDINGS.md 2026-10-13).
static DEFERRED: Mutex<Vec<(u64, u64)>> = Mutex::new(Vec::new());
static DEFERRED_LEN: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
/// [deferred, decommitted later, mapped again first, bytes deferred].
pub static DEFERRED_STATS: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// Decommits under GPU imports wait for them (`DS_DEFER_DECOMMIT=0`: at
/// once, as before; for A/B).
fn defer_decommits() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_DEFER_DECOMMIT").ok().as_deref() != Some("0"))
}

/// Removes [start, end) from the deferred decommits; the parts that were
/// deferred (still committed).
fn cancel_deferred(start: u64, end: u64) -> Vec<(u64, u64)> {
    use std::sync::atomic::Ordering::Relaxed;
    if DEFERRED_LEN.load(std::sync::atomic::Ordering::Acquire) == 0 {
        return Vec::new();
    }
    let mut d = DEFERRED.lock();
    let mut hit = Vec::new();
    let mut keep = Vec::with_capacity(d.len());
    for &(a, e) in d.iter() {
        if e <= start || end <= a {
            keep.push((a, e));
            continue;
        }
        hit.push((a.max(start), e.min(end)));
        if a < start {
            keep.push((a, start));
        }
        if end < e {
            keep.push((end, e));
        }
    }
    if !hit.is_empty() {
        DEFERRED_STATS[2].fetch_add(hit.len() as u64, Relaxed);
    }
    *d = keep;
    DEFERRED_LEN.store(d.len(), std::sync::atomic::Ordering::Release);
    hit
}

/// Decommits the deferred ranges no import covers any more (the GPU
/// completion thread, after freeing dropped imports).
pub fn retry_deferred() {
    if crate::sections::on() {
        crate::sections::retry_deferred(crate::gpu::imports_may_cover);
        return;
    }
    if DEFERRED_LEN.load(std::sync::atomic::Ordering::Acquire) == 0 {
        return;
    }
    // The memory manager's lock: no mapping changes meanwhile.
    let _st = MEM.lock();
    let mut d = DEFERRED.lock();
    let mut keep = Vec::new();
    for &(a, e) in d.iter() {
        if crate::gpu::imports_may_cover(a, e - a) {
            keep.push((a, e));
            continue;
        }
        let _readers_out = GUEST_READERS.write();
        unsafe {
            VirtualFree(a as _, (e - a) as usize, MEM_DECOMMIT);
        }
        DEFERRED_STATS[1].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    *d = keep;
    DEFERRED_LEN.store(d.len(), std::sync::atomic::Ordering::Release);
}

/// Bulk readers of guest memory on our threads (texture uploads) hold this
/// shared while they check that pages are mapped and read them; decommits and
/// protection changes take it exclusively. A title thread unmapped a layer of
/// a streamed texture array during a level load while a queue thread
/// detiled it (fatal access violation, run 20261012-082109). Holders must not
/// take `MEM` (decommits hold it while they wait here): they check pages with
/// `os_readable`.
pub static GUEST_READERS: parking_lot::RwLock<()> = parking_lot::RwLock::new(());

/// Whether every page of [a, a+len) is committed and readable now (operating
/// system query; takes none of our locks).
pub fn os_readable(a: u64, len: u64) -> bool {
    use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, MEM_COMMIT, PAGE_GUARD, PAGE_NOACCESS, VirtualQuery};
    let end = a.saturating_add(len.max(1));
    let mut p = a;
    while p < end {
        let mut mbi: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        if unsafe { VirtualQuery(p as *const _, &mut mbi, std::mem::size_of::<MEMORY_BASIC_INFORMATION>()) } == 0 {
            return false;
        }
        if mbi.State != MEM_COMMIT || mbi.Protect == 0 || mbi.Protect & (PAGE_NOACCESS | PAGE_GUARD) != 0 {
            return false;
        }
        let next = mbi.BaseAddress as u64 + mbi.RegionSize as u64;
        if next <= p {
            return false;
        }
        p = next;
    }
    true
}

fn host_decommit(addr: u64, size: u64) {
    crate::gpu::track::remapped(addr, size, crate::gpu::track::REMAP_DECOMMIT);
    repaged(addr, size);
    if defer_decommits() && crate::gpu::imports_may_cover(addr, size) {
        let mut d = DEFERRED.lock();
        d.push((addr, addr + size));
        DEFERRED_LEN.store(d.len(), std::sync::atomic::Ordering::Release);
        DEFERRED_STATS[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        DEFERRED_STATS[3].fetch_add(size, std::sync::atomic::Ordering::Relaxed);
        return;
    }
    let _readers_out = GUEST_READERS.write();
    unsafe {
        VirtualFree(addr as _, size as usize, MEM_DECOMMIT);
    }
}

pub fn host_protect(addr: u64, size: u64, prot: u32) -> Result<(), MemError> {
    crate::gpu::track::remapped(addr, size, crate::gpu::track::REMAP_PROTECT);
    let mut old = 0u32;
    let _readers_out = GUEST_READERS.write();
    let ok = unsafe { VirtualProtect(addr as _, size as usize, win_prot(prot), &mut old) };
    if ok == 0 { Err(MemError::HostFailure(last_error())) } else { Ok(()) }
}

/// `DS_TRACE_MAP=<hex lo>-<hex hi>`: logs every map, unmap and protection
/// change that touches [lo, hi), with the calling thread's guest frames.
fn trace_map(op: &str, start: u64, end: u64, detail: std::fmt::Arguments) {
    static RANGE: std::sync::OnceLock<Option<(u64, u64)>> = std::sync::OnceLock::new();
    let r = RANGE.get_or_init(|| {
        let v = std::env::var("DS_TRACE_MAP").ok()?;
        let (lo, hi) = v.split_once('-')?;
        let h = |x: &str| u64::from_str_radix(x.trim().trim_start_matches("0x"), 16).ok();
        Some((h(lo)?, h(hi)?))
    });
    if let Some((lo, hi)) = *r {
        if start < hi && lo < end {
            crate::info!("mem trace: {op} {start:#x}..{end:#x} {detail} [{}]: {}", crate::log::thread_name(), crate::exception::current_guest_frames(8).join(" <- "));
        }
    }
}

/// Maps the views of a new mapping (`sections`): the guest view and its
/// alias. Returns (alias slot, anonymous-section offset).
fn section_commit(addr: u64, size: u64, kind: VmaKind, prot: u32, phys: u64) -> Result<(u8, u64), MemError> {
    use crate::sections::{self as sec, Sec};
    crate::gpu::track::remapped(addr, size, crate::gpu::track::REMAP_COMMIT);
    repaged(addr, size);
    let (which, off) = if kind == VmaKind::Direct { (Sec::Dmem, phys) } else { (Sec::Anon, sec::anon_alloc(size).ok_or(MemError::NoSpace)?) };
    if let Err(e) = sec::map_view(addr, size, which, off, win_prot(prot)) {
        crate::warn!("mem: {e}");
        if which == Sec::Anon {
            sec::anon_free(off, size);
        }
        return Err(MemError::HostFailure(0));
    }
    let slot = match sec::map_alias(addr, size, which, off) {
        Ok(s) => s,
        Err(e) => {
            crate::warn!("mem: {e} (the GPU cannot import it)");
            NO_SLOT
        }
    };
    Ok((slot, if which == Sec::Anon { off } else { 0 }))
}

/// Unmaps the views of mapping `v` (`sections`): its guest view now, its
/// alias once nothing maps through it and no GPU import may use it.
fn section_release(v: &Vma) {
    crate::gpu::track::remapped(v.start, v.size, crate::gpu::track::REMAP_DECOMMIT);
    repaged(v.start, v.size);
    {
        let _readers_out = GUEST_READERS.write();
        if let Err(e) = crate::sections::unmap_view(v.start) {
            crate::warn!("mem: {e}");
        }
    }
    if v.slot != NO_SLOT {
        crate::sections::alias_unref(v.start, v.size, v.slot, crate::gpu::imports_may_cover);
    }
    if v.kind != VmaKind::Direct {
        crate::sections::anon_free(v.off, v.size);
    }
}

/// Splits the guest view of mapping `v` at `at` (`sections`): views cannot
/// be unmapped in part, so it is unmapped and mapped again as two views of
/// the same section pages (contents kept). Faults of other threads inside it
/// meanwhile wait (`sections::wait_split`); the residency's protections of its
/// pages are reset.
fn section_split(v: &Vma, at: u64) {
    use crate::sections::{self as sec, Sec};
    let (which, off) = if v.kind == VmaKind::Direct { (Sec::Dmem, v.phys) } else { (Sec::Anon, v.off) };
    crate::gpu::track::remapped(v.start, v.size, crate::gpu::track::REMAP_PROTECT);
    let _readers_out = GUEST_READERS.write();
    sec::split_begin(v.start, v.end());
    let lo = at - v.start;
    let r = sec::unmap_view(v.start)
        .and_then(|_| sec::map_view(v.start, lo, which, off, win_prot(v.prot)))
        .and_then(|_| sec::map_view(at, v.end() - at, which, off + lo, win_prot(v.prot)));
    sec::split_end();
    if let Err(e) = r {
        crate::error!("mem: splitting the view {:#x}+{:#x} at {at:#x}: {e}", v.start, v.size);
    }
}

impl MemState {
    /// Finds a free, aligned range of `size` bytes at or above `hint`.
    pub fn find_free(&self, hint: u64, size: u64, align: u64, limit: u64) -> Option<u64> {
        let align = align.max(PAGE);
        let mut cand = align_up(hint, align);
        for (_, v) in self.vmas.range(..) {
            if v.end() <= cand {
                continue;
            }
            if v.start >= cand + size {
                break;
            }
            cand = align_up(v.end(), align);
        }
        if cand + size <= limit { Some(cand) } else { None }
    }

    pub fn find(&self, addr: u64) -> Option<&Vma> {
        self.vmas.range(..=addr).next_back().map(|(_, v)| v).filter(|v| addr < v.end())
    }

    /// Ensures a VMA boundary exists at `addr`.
    fn split_at(&mut self, addr: u64) {
        let Some(v) = self.vmas.range(..addr).next_back().map(|(_, v)| v.clone()) else { return };
        if addr > v.start && addr < v.end() {
            if crate::sections::on() && v.kind != VmaKind::Reserved {
                section_split(&v, addr);
            }
            let mut lo = v.clone();
            lo.size = addr - v.start;
            let mut hi = v;
            hi.size = hi.end() - addr;
            if hi.kind == VmaKind::Direct {
                hi.phys += lo.size;
            } else {
                hi.off += lo.size;
            }
            hi.start = addr;
            self.vmas.insert(lo.start, lo);
            self.vmas.insert(hi.start, hi);
        }
    }

    fn overlapping(&self, start: u64, end: u64) -> Vec<Vma> {
        let mut out = Vec::new();
        if let Some((_, v)) = self.vmas.range(..start).next_back() {
            if v.end() > start {
                out.push(v.clone());
            }
        }
        for (_, v) in self.vmas.range(start..end) {
            out.push(v.clone());
        }
        out
    }

    /// Removes mappings in [start, end), decommitting host pages. Reserved ranges
    /// are left reserved only if `keep_reserved`.
    pub fn unmap_range(&mut self, start: u64, end: u64) {
        trace_map("unmap", start, end, format_args!(""));
        bump_gen(start, end);
        self.split_at(start);
        self.split_at(end);
        let keys: Vec<u64> = self.vmas.range(start..end).map(|(k, _)| *k).collect();
        for k in keys {
            let v = self.vmas.remove(&k).unwrap();
            if v.kind == VmaKind::Direct {
                let mut r = REMAP.lock();
                if r.released.len() > 1 << 16 {
                    r.released.clear();
                }
                r.released.insert(v.phys, (v.start, v.size));
            }
            if v.kind == VmaKind::Direct && phys_carry() && !crate::sections::on() {
                park(v.start, v.phys, v.size);
            }
            if v.kind != VmaKind::Reserved {
                if crate::sections::on() {
                    section_release(&v);
                } else {
                    host_decommit(v.start, v.size);
                }
            }
            if v.kind == VmaKind::Flexible {
                self.flexible_used = self.flexible_used.saturating_sub(v.size);
            }
        }
    }

    /// Maps [addr, addr+size). For fixed mappings any existing mapping there is replaced.
    #[allow(clippy::too_many_arguments)]
    pub fn map(
        &mut self,
        addr: u64,
        size: u64,
        kind: VmaKind,
        prot: u32,
        phys: u64,
        mtype: i32,
        name: &str,
    ) -> Result<u64, MemError> {
        if addr < GUEST_BASE || addr + size > GUEST_END || size == 0 {
            return Err(MemError::Invalid);
        }
        let existing = self.overlapping(addr, addr + size);
        if kind == VmaKind::Direct {
            // Diagnostics: contents the console would keep.
            let same = existing
                .iter()
                .filter(|v| v.kind == VmaKind::Direct && v.phys.wrapping_sub(phys) == v.start.wrapping_sub(addr))
                .map(|v| v.end().min(addr + size) - v.start.max(addr))
                .sum::<u64>();
            let mut r = REMAP.lock();
            if same > 0 {
                r.in_place += 1;
                r.in_place_bytes += same;
                if r.logged < 24 {
                    r.logged += 1;
                    crate::info!("mem: direct {addr:#x}+{size:#x} (phys {phys:#x}) mapped again over its own pages ({same:#x} bytes zeroed here) '{name}'");
                }
            }
            let prior: Vec<(u64, u64, u64)> = r
                .released
                .range(phys.saturating_sub(1 << 30)..phys + size)
                .filter(|(p, (_, n))| **p < phys + size && phys < **p + *n)
                .map(|(p, (va, n))| (*p, *va, *n))
                .collect();
            for (p, va, n) in prior {
                if va.wrapping_sub(p) != addr.wrapping_sub(phys) {
                    r.moved += 1;
                    r.moved_bytes += n.min(size);
                    if r.logged < 24 {
                        r.logged += 1;
                        crate::info!("mem: direct phys {p:#x}+{n:#x}, last mapped at {va:#x}, now mapped at {:#x} '{name}' (contents not carried over)", addr + p.saturating_sub(phys));
                    }
                }
            }
        }
        if !existing.is_empty() {
            self.unmap_range(addr, addr + size);
        }
        let (mut slot, mut off) = (NO_SLOT, 0);
        if kind != VmaKind::Reserved {
            if crate::sections::on() {
                (slot, off) = section_commit(addr, size, kind, prot, phys)?;
            } else {
                host_commit(addr, size, prot)?;
            }
        }
        if kind == VmaKind::Direct && phys_carry() && !crate::sections::on() {
            unpark(addr, phys, size, prot);
        }
        if kind != VmaKind::Reserved {
            // Alignment of mappings (64 KiB: what section views may need);
            // direct ones counted with their physical offset.
            let bits = if kind == VmaKind::Direct { addr | size | phys } else { addr | size };
            let k = if bits & 0xFFFF == 0 { 0 } else if bits & 0x3FFF == 0 { 1 } else { 2 };
            MAP_ALIGN[k + if kind == VmaKind::Direct { 0 } else { 3 }].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        if kind == VmaKind::Direct && !crate::sections::on() {
            for v in self.vmas.values() {
                if v.kind == VmaKind::Direct && phys < v.phys + v.size && v.phys < phys + size {
                    crate::warn!(
                        "direct memory aliasing: phys {:#x}+{:#x} already mapped at {:#x} (new va {:#x}); contents are NOT shared",
                        phys,
                        size,
                        v.start,
                        addr
                    );
                    break;
                }
            }
        }
        if kind == VmaKind::Flexible {
            self.flexible_used += size;
        }
        self.vmas.insert(addr, Vma { start: addr, size, kind, prot, phys, mtype, name: name.to_string(), slot, off });
        bump_gen(addr, addr + size);
        trace_map("map", addr, addr + size, format_args!("{kind:?} prot {prot:#x} phys {phys:#x} '{name}'"));
        Ok(addr)
    }

    /// Maximal run of adjacent committed direct/flexible mappings containing
    /// `addr`: [start, end).
    pub fn committed_span(&self, addr: u64) -> Option<(u64, u64)> {
        let usable = |v: &Vma| matches!(v.kind, VmaKind::Direct | VmaKind::Flexible);
        let v = self.find(addr).filter(|v| usable(v))?;
        let (mut start, mut end) = (v.start, v.end());
        while let Some(p) = self.vmas.range(..start).next_back().map(|(_, p)| p).filter(|p| p.end() == start && usable(p)) {
            start = p.start;
        }
        while let Some(n) = self.vmas.get(&end).filter(|n| usable(n)) {
            end = n.end();
        }
        Some((start, end))
    }

    pub fn protect(&mut self, start: u64, end: u64, prot: u32) -> Result<(), MemError> {
        trace_map("protect", start, end, format_args!("prot {prot:#x}"));
        self.split_at(start);
        self.split_at(end);
        let keys: Vec<u64> = self.vmas.range(start..end).map(|(k, _)| *k).collect();
        for k in keys {
            let v = self.vmas.get_mut(&k).unwrap();
            if v.kind == VmaKind::Reserved {
                continue;
            }
            v.prot = prot;
            host_protect(v.start, v.size, prot)?;
        }
        bump_gen(start, end);
        Ok(())
    }

    pub fn dmem_allocate(&mut self, search_start: u64, search_end: u64, len: u64, align: u64, mtype: i32) -> Option<u64> {
        let align = align.max(PAGE);
        let end_limit = search_end.min(DMEM_SIZE);
        let mut cand = align_up(search_start, align);
        for (&s, b) in self.dmem.iter() {
            if s + b.size <= cand {
                continue;
            }
            if s >= cand + len {
                break;
            }
            cand = align_up(s + b.size, align);
        }
        if cand + len > end_limit {
            return None;
        }
        self.dmem.insert(cand, DmemBlock { size: len, mtype });
        Some(cand)
    }

    pub fn dmem_release(&mut self, start: u64, len: u64) {
        let end = start + len;
        let keys: Vec<u64> = self.dmem.range(..end).filter(|(s, b)| **s + b.size > start).map(|(s, _)| *s).collect();
        for s in keys {
            let b = self.dmem.remove(&s).unwrap();
            if s < start {
                self.dmem.insert(s, DmemBlock { size: start - s, mtype: b.mtype });
            }
            if s + b.size > end {
                self.dmem.insert(end, DmemBlock { size: s + b.size - end, mtype: b.mtype });
            }
        }
        // Unmap any virtual mappings of the released physical range.
        let vas: Vec<(u64, u64)> = self
            .vmas
            .values()
            .filter(|v| v.kind == VmaKind::Direct && v.phys < end && start < v.phys + v.size)
            .map(|v| {
                let s = v.start + start.saturating_sub(v.phys);
                let e = v.start + (end.min(v.phys + v.size) - v.phys);
                (s, e)
            })
            .collect();
        for (s, e) in vas {
            self.unmap_range(s, e);
        }
        if phys_carry() {
            drop_parked(start, end);
        }
    }

    pub fn dmem_used(&self) -> u64 {
        self.dmem.values().map(|b| b.size).sum()
    }

    /// Largest free direct-memory block within [start, end).
    pub fn dmem_largest_free(&self, start: u64, end: u64) -> (u64, u64) {
        let end = end.min(DMEM_SIZE);
        let mut best = (0u64, 0u64);
        let mut cursor = start;
        for (&s, b) in self.dmem.iter() {
            if s + b.size <= cursor {
                continue;
            }
            if s > cursor && s.min(end) > cursor {
                let len = s.min(end) - cursor;
                if len > best.1 {
                    best = (cursor, len);
                }
            }
            cursor = cursor.max(s + b.size);
            if cursor >= end {
                break;
            }
        }
        if end > cursor && end - cursor > best.1 {
            best = (cursor, end - cursor);
        }
        best
    }

    /// Allocates runtime-owned guest-visible memory (stub code, HLE data objects).
    pub fn alloc_internal(&mut self, size: u64, prot: u32, name: &str) -> Result<u64, MemError> {
        let size = align_up(size, PAGE);
        let addr = self.find_free(self.internal_cursor, size, PAGE, MODULE_BASE).ok_or(MemError::NoSpace)?;
        self.map(addr, size, VmaKind::Internal, prot, 0, 0, name)?;
        self.internal_cursor = addr + size;
        Ok(addr)
    }
}

/// Simple bump allocator over internal guest memory for small HLE-owned objects.
pub struct GuestArena {
    cur: u64,
    end: u64,
}

pub static ARENA: Mutex<GuestArena> = Mutex::new(GuestArena { cur: 0, end: 0 });

pub fn arena_alloc(size: u64, align: u64) -> u64 {
    let mut a = ARENA.lock();
    let mut p = align_up(a.cur, align.max(8));
    if a.cur == 0 || p + size > a.end {
        let chunk = align_up(size.max(1 << 20), PAGE);
        let base = MEM
            .lock()
            .alloc_internal(chunk, PROT_CPU_READ | PROT_CPU_WRITE, "hle-arena")
            .expect("internal arena exhausted");
        a.cur = base;
        a.end = base + chunk;
        p = align_up(a.cur, align.max(8));
    }
    a.cur = p + size;
    p
}

/// Copies a NUL-terminated string into the guest arena and returns its address.
pub fn arena_cstr(s: &str) -> u64 {
    let p = arena_alloc(s.len() as u64 + 1, 1);
    unsafe {
        std::ptr::copy_nonoverlapping(s.as_ptr(), p as *mut u8, s.len());
        *((p + s.len() as u64) as *mut u8) = 0;
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Cached readable runs survive mapping changes that do not touch them.
    #[test]
    fn readable_runs_survive_unrelated_changes() {
        let g0 = MAP_GEN.load(std::sync::atomic::Ordering::SeqCst);
        let mut runs = vec![(0x1_0000, 0x2_0000, true), (0x5_0000, 0x6_0000, false)];
        bump_gen(0x2_0000, 0x3_0000); // adjacent to the first run
        bump_gen(0x5_8000, 0x5_9000); // inside the second
        let g1 = MAP_GEN.load(std::sync::atomic::Ordering::SeqCst);
        assert!(revalidate(&mut runs, g0, g1));
        assert_eq!(runs, vec![(0x1_0000, 0x2_0000, true)]);
        assert!(!revalidate(&mut runs, u64::MAX, g1), "a cache never filled");
        for _ in 0..CHANGES_KEEP + 1 {
            bump_gen(0x9_0000, 0x9_1000);
        }
        let g2 = MAP_GEN.load(std::sync::atomic::Ordering::SeqCst);
        assert!(!revalidate(&mut runs, g1, g2), "changes no longer all known");
    }
}
