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
    /// Section-backed memory (`sections`): the alias slots holding alias
    /// views of its pages that it references (bit per slot: the slot its
    /// mapping took, and slots `bridge_aliases` added) and, for memory other
    /// than direct, its offset in the anonymous section.
    pub slots: u8,
    pub off: u64,
}
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

/// A mapping's access class from the title's protection bits: "CPU+GPU",
/// "GPU only" (no CPU bit: the title's CPU code cannot touch it), "CPU only"
/// or "none".
pub fn prot_class(prot: u32) -> &'static str {
    match (prot & 7 != 0, prot & (PROT_GPU_READ | PROT_GPU_WRITE) != 0) {
        (true, true) => "CPU+GPU",
        (false, true) => "GPU only",
        (true, false) => "CPU only",
        (false, false) => "none",
    }
}

/// The title's protection bits of the mapping holding `a` (None: unmapped).
pub fn prot_at(a: u64) -> Option<u32> {
    MEM.lock().find(a).filter(|v| v.kind != VmaKind::Reserved).map(|v| v.prot)
}

/// The memory type the title allocated the direct memory under `a` with
/// (None: unmapped or not direct memory).
pub fn mtype_at(a: u64) -> Option<i32> {
    let g = MEM.lock();
    let v = g.find(a).filter(|v| v.kind == VmaKind::Direct)?;
    let phys = v.phys + (a - v.start);
    g.dmem.range(..=phys).next_back().filter(|(s, b)| phys < **s + b.size).map(|(_, b)| b.mtype)
}

/// Guest mappings by kind, access class and memory type (MiB, mappings;
/// status report). The memory type is the one the title asked for (direct
/// memory: `sceKernelAllocateDirectMemory`'s), which says how it means to
/// use the memory (CPU-cached for what the CPU reads back, GPU-cached or
/// write-combined for GPU work).
pub fn prot_census() -> String {
    let g = MEM.lock();
    let mut m: std::collections::BTreeMap<(String, &'static str, i32), (u64, u64)> = std::collections::BTreeMap::new();
    for v in g.vmas.values() {
        if v.kind == VmaKind::Reserved {
            continue;
        }
        let e = m.entry((format!("{:?}", v.kind), prot_class(v.prot), v.mtype)).or_insert((0, 0));
        e.0 += v.size;
        e.1 += 1;
    }
    // Direct memory by the type the title allocated it with.
    let mut types: std::collections::BTreeMap<i32, (u64, u64)> = std::collections::BTreeMap::new();
    for b in g.dmem.values() {
        let e = types.entry(b.mtype).or_insert((0, 0));
        e.0 += b.size;
        e.1 += 1;
    }
    drop(g);
    let dm: Vec<String> = types.iter().map(|(t, (b, n))| format!("type {t} {} MiB ({n})", b >> 20)).collect();
    let parts: Vec<String> = m.iter().map(|((k, c, t), (b, n))| format!("{k} {c} type {t} {} MiB ({n})", b >> 20)).collect();
    format!("    direct memory by the type the title allocated it with: {}\n", dm.join(", "))
        + &format!("    guest mappings by kind, the title's access bits and memory type: {}\n", parts.join(", "))
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
        "    direct memory remaps: {} in place ({} MiB, zeroed here), {} of pages last mapped elsewhere ({} MiB, contents not carried); mapping changes {}; readable-range checks {} ({} cache misses); decommits deferred under GPU imports {} ({} MiB; {} done later, {} mapped again first, {} waiting)\n",
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

/// Whether the mapping table holds [a, a+len) in mappings whose host pages
/// are committed and readable (`win_prot` not PAGE_NOACCESS): what
/// `os_readable` asks the operating system, from the table (bulk readers
/// under `GUEST_READERS` use it from before the lock: `table_view_held`).
pub fn host_readable(a: u64, len: u64) -> bool {
    use windows_sys::Win32::System::Memory::PAGE_NOACCESS;
    let st = MEM.lock();
    let end = a.saturating_add(len.max(1));
    let mut p = a;
    while p < end {
        match st.find(p) {
            Some(v) if v.kind != VmaKind::Reserved && win_prot(v.prot) != PAGE_NOACCESS => p = v.end(),
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
/// Takes that the probe query found clean (`tune ww_probe`).
pub static PROBE_CLEAN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// `take_written` by caller (`WW_WHO_NAMES`) and by size (log2 of the 4 KiB
/// pages, the last bucket 2^20 and more): [calls, ns, found clean by the
/// probe] (profile windows).
pub static WW_BY_WHO: [[std::sync::atomic::AtomicU64; 3]; 10] = [const { [const { std::sync::atomic::AtomicU64::new(0) }; 3] }; 10];
pub static WW_BY_SIZE: [[std::sync::atomic::AtomicU64; 3]; 21] = [const { [const { std::sync::atomic::AtomicU64::new(0) }; 3] }; 21];
pub const WW_WHO_NAMES: [&str; 10] = ["copy-cache checks", "copy-cache inserts", "homes watch", "texture scrubbing", "texture re-uploads", "homes candidates", "shadow windows", "other", "resident data", "front end 2"];
/// `take_written` for resident GPU-written data (`gpu::rgm`).
pub const WW_WHO_RGM: usize = 8;
/// `take_written` for front end 2's placements (`gpu::fe2`).
pub const WW_WHO_FE2: usize = 9;

/// Queries in flight (`InFlight`), and the calls and ns by how many other
/// queries were in flight when each began (0..7, 8 or more).
static WW_INFLIGHT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static WW_BY_CONC: [[std::sync::atomic::AtomicU64; 3]; 9] = [const { [const { std::sync::atomic::AtomicU64::new(0) }; 3] }; 9];

/// A query in flight (counted while it lives); `others`: those in flight
/// when it began.
struct InFlight {
    others: u64,
}

impl InFlight {
    fn new() -> InFlight {
        InFlight { others: WW_INFLIGHT.fetch_add(1, std::sync::atomic::Ordering::Relaxed) }
    }
}

impl Drop for InFlight {
    fn drop(&mut self) {
        WW_INFLIGHT.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
    }
}

fn ww_note(who: usize, pages: u64, ns: u64, clean: bool, others: u64) {
    use std::sync::atomic::Ordering::Relaxed;
    let b = (64 - pages.max(1).leading_zeros() as usize - 1).min(20);
    for s in [&WW_BY_WHO[who.min(9)], &WW_BY_SIZE[b], &WW_BY_CONC[(others as usize).min(8)]] {
        s[0].fetch_add(1, Relaxed);
        s[1].fetch_add(ns, Relaxed);
        s[2].fetch_add(clean as u64, Relaxed);
    }
}

/// The breakdown of `take_written` since the last profile reset.
pub fn ww_lines() -> String {
    use std::sync::atomic::Ordering::Relaxed;
    let f = |s: &[std::sync::atomic::AtomicU64; 3]| {
        let n = s[0].load(Relaxed);
        format!("{n} ({:.1} us, {} clean)", s[1].load(Relaxed) as f64 / n.max(1) as f64 / 1000.0, s[2].load(Relaxed))
    };
    let who: Vec<String> = WW_WHO_NAMES.iter().zip(WW_BY_WHO.iter()).filter(|(_, s)| s[0].load(Relaxed) > 0).map(|(n, s)| format!("{n} {}", f(s))).collect();
    let size: Vec<String> = WW_BY_SIZE.iter().enumerate().filter(|(_, s)| s[0].load(Relaxed) > 0).map(|(k, s)| format!("{}+ {}", 1u64 << k, f(s))).collect();
    let conc: Vec<String> = WW_BY_CONC.iter().enumerate().filter(|(_, s)| s[0].load(Relaxed) > 0).map(|(k, s)| format!("{k}{} {}", if k == 8 { "+" } else { "" }, f(s))).collect();
    format!("    write-watch queries by caller: {}\n    by pages: {}\n    by other queries in flight when one began: {}\n", who.join(", "), size.join(", "), conc.join(", "))
}

pub fn ww_reset() {
    use std::sync::atomic::Ordering::Relaxed;
    for c in WW_BY_WHO.iter().chain(WW_BY_SIZE.iter()).chain(WW_BY_CONC.iter()).flatten() {
        c.store(0, Relaxed);
    }
}

thread_local! {
    /// This thread's write-watch queries take turns with `tune ww_gate` 1
    /// (the parallel recording workers and the texture scrubber).
    static WW_GATED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Marks this thread's write-watch queries as taking turns (`tune ww_gate`).
pub fn set_ww_gated(on: bool) {
    WW_GATED.with(|g| g.set(on));
}

/// Held by the thread whose turn it is (`ww_turn`).
static WW_GATE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// A write-watch query's turn; released when dropped.
struct WwTurn(bool);

impl Drop for WwTurn {
    fn drop(&mut self) {
        if self.0 {
            WW_GATE.store(false, std::sync::atomic::Ordering::Release);
        }
    }
}

/// Write-watch queries wait on each other in the kernel (3.4 us alone,
/// 10-120 us with others in flight; FINDINGS.md 2026-10-17): with `tune
/// ww_gate` 1 the parallel recording workers and the texture scrubber query
/// one at a time (spinning for the turn), so that a command processor's
/// query, on the frame's critical path, meets at most one of theirs; with
/// 2 every thread takes turns.
fn ww_turn() -> WwTurn {
    use std::sync::atomic::Ordering;
    let gated = match crate::gpu::tune::ww_gate() {
        1 => WW_GATED.with(|g| g.get()),
        2 => true,
        _ => false,
    };
    if !gated {
        return WwTurn(false);
    }
    let mut spins = 0u32;
    while WW_GATE.compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed).is_err() {
        spins += 1;
        if spins < 4096 {
            std::hint::spin_loop();
        } else {
            std::thread::yield_now();
        }
    }
    WwTurn(true)
}

/// The 4 KiB pages of guest [base, base+size) the CPU wrote since they were
/// last taken (their write marks are reset), into `out` (page addresses);
/// None without write watching or when the query fails (treat as written).
/// `who`: the caller (`WW_WHO_NAMES`; statistics).
pub fn take_written(base: u64, size: u64, out: &mut Vec<u64>, who: usize) -> Option<()> {
    use std::sync::atomic::Ordering::Relaxed;
    if crate::sections::on() {
        let inflight = InFlight::new();
        let t = std::time::Instant::now();
        let r = crate::gpu::resident::take_written(base, size, out);
        let ns = t.elapsed().as_nanos() as u64;
        ww_note(who, size.div_ceil(4096), ns, out.is_empty(), inflight.others);
        WRITE_WATCH_STATS[0].fetch_add(1, Relaxed);
        WRITE_WATCH_STATS[1].fetch_add(ns, Relaxed);
        WRITE_WATCH_STATS[2].fetch_add(out.len() as u64, Relaxed);
        return r;
    }
    if !write_watched() || size == 0 {
        return None;
    }
    let (lo, hi) = (base & !4095, (base + size).next_multiple_of(4096));
    // Every take resets the marks it reads: resident GPU-written data
    // (`gpu::rgm`) learns of the pages written, whoever took them.
    let first = out.len();
    let r = take_written_inner(lo, hi, out, who);
    match r {
        Some(()) if out.len() > first => crate::gpu::rgm::cpu_written(&out[first..]),
        Some(()) => {}
        None => crate::gpu::rgm::invalidate(lo, hi - lo),
    }
    r
}

fn take_written_inner(lo: u64, hi: u64, out: &mut Vec<u64>, who: usize) -> Option<()> {
    use std::sync::atomic::Ordering::Relaxed;
    let _turn = ww_turn();
    let inflight = InFlight::new();
    let t = std::time::Instant::now();
    let mut addrs = [std::ptr::null_mut::<std::ffi::c_void>(); 256];
    let mut at = lo;
    if crate::gpu::tune::ww_probe() {
        // Clean ranges (most takes) end with one query that resets nothing;
        // otherwise the resetting queries start at the first written page.
        let mut count: usize = 1;
        let mut gran = 0u32;
        let r = unsafe { GetWriteWatch(0, lo as _, (hi - lo) as usize, addrs.as_mut_ptr(), &mut count, &mut gran) };
        if r == 0 && count == 0 {
            let ns = t.elapsed().as_nanos() as u64;
            ww_note(who, (hi - lo) / 4096, ns, true, inflight.others);
            WRITE_WATCH_STATS[0].fetch_add(1, Relaxed);
            WRITE_WATCH_STATS[1].fetch_add(ns, Relaxed);
            PROBE_CLEAN.fetch_add(1, Relaxed);
            return Some(());
        }
        if r == 0 && count == 1 {
            at = (addrs[0] as u64).clamp(lo, hi);
        }
    }
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
    let ns = t.elapsed().as_nanos() as u64;
    ww_note(who, (hi - lo) / 4096, ns, false, inflight.others);
    WRITE_WATCH_STATS[0].fetch_add(1, Relaxed);
    WRITE_WATCH_STATS[1].fetch_add(ns, Relaxed);
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

/// Host memory calls for guest mappings in the profile window
/// (`host_calls_line`): [commits, ns, decommits, ns, decommits deferred
/// under GPU imports, protection changes, ns, unmap notes, ns].
pub static HOST_CALLS: [std::sync::atomic::AtomicU64; 9] = [const { std::sync::atomic::AtomicU64::new(0) }; 9];
static FAULTS_AT_RESET: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Adds a call and its time to HOST_CALLS[k], [k + 1].
fn note_host_call(k: usize, t0: std::time::Instant) {
    use std::sync::atomic::Ordering::Relaxed;
    HOST_CALLS[k].fetch_add(1, Relaxed);
    HOST_CALLS[k + 1].fetch_add(t0.elapsed().as_nanos() as u64, Relaxed);
}

/// This process's page faults so far.
fn page_faults() -> u64 {
    use windows_sys::Win32::System::ProcessStatus::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
    let mut c: PROCESS_MEMORY_COUNTERS = unsafe { std::mem::zeroed() };
    c.cb = std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
    let ok = unsafe { GetProcessMemoryInfo(windows_sys::Win32::System::Threading::GetCurrentProcess(), &mut c, c.cb) };
    if ok != 0 { c.PageFaultCount as u64 } else { 0 }
}

/// Clears the window's host-call counts (`profile reset`).
pub fn reset_host_calls() {
    use std::sync::atomic::Ordering::Relaxed;
    for c in &HOST_CALLS {
        c.store(0, Relaxed);
    }
    FAULTS_AT_RESET.store(page_faults(), Relaxed);
}

/// One profile line: the window's host memory calls and page faults.
pub fn host_calls_line() -> String {
    use std::sync::atomic::Ordering::Relaxed;
    let v: Vec<u64> = HOST_CALLS.iter().map(|c| c.load(Relaxed)).collect();
    let faults = page_faults().saturating_sub(FAULTS_AT_RESET.load(Relaxed));
    format!(
        "    host memory calls in the window: commits {} ({} ms), decommits {} ({} ms; {} more deferred under GPU imports), protection changes {} ({} ms), unmap notes {} ({} ms); page faults {}\n",
        v[0],
        v[1] / 1_000_000,
        v[2],
        v[3] / 1_000_000,
        v[4],
        v[5],
        v[6] / 1_000_000,
        v[7],
        v[8] / 1_000_000,
        faults
    )
}

fn host_commit(addr: u64, size: u64, prot: u32) -> Result<(), MemError> {
    let t0 = std::time::Instant::now();
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
    let _arm = crate::gpu::resident::remap_guard();
    // Pages armed through the host read-write first: whether committing
    // already committed pages resets their protection is not ours to rely
    // on, and `settled` only clears their bits.
    crate::hosted::disarm(addr, size);
    let p = unsafe { VirtualAlloc(addr as _, size as usize, MEM_COMMIT, win_prot(prot)) };
    crate::gpu::resident::settled(addr, size);
    note_host_call(0, t0);
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
        let _arm = crate::gpu::resident::remap_guard();
        unsafe {
            VirtualFree(a as _, (e - a) as usize, MEM_DECOMMIT);
        }
        // Their page states go with them, armed bits too: decommitted pages
        // still marked armed made a disarm over them and their armed
        // neighbours fail (FINDINGS.md 2026-10-27 night).
        crate::gpu::resident::settled(a, e - a);
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
/// `os_readable`, or with the mapping table before taking this and
/// `table_view_held` after.
pub static GUEST_READERS: parking_lot::RwLock<()> = parking_lot::RwLock::new(());

/// Whether a view of the mapping table taken at generation `g0` (`MAP_GEN`
/// read before the table checks) still holds now that `GUEST_READERS` is
/// held: no mapping or protection change since. Unmaps and protection
/// changes move the generation before they change pages, and wait for the
/// readers, so ranges the table showed readable stay readable while the
/// lock is held. (`VirtualQuery` sizes its region by walking page tables:
/// thousands of checks per streamed texture array cost 0.5-0.6 s.)
pub fn table_view_held(g0: u64) -> bool {
    MAP_GEN.load(std::sync::atomic::Ordering::SeqCst) == g0
}

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

/// The operating system's regions over [a, a+len): one clause each
/// (state, protection, type), at most `max` (diagnostics of failed imports).
pub fn os_regions(a: u64, len: u64, max: usize) -> String {
    use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, VirtualQuery};
    let end = a.saturating_add(len.max(1));
    let mut p = a;
    let mut out = Vec::new();
    while p < end && out.len() < max {
        let mut mbi: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        if unsafe { VirtualQuery(p as *const _, &mut mbi, std::mem::size_of::<MEMORY_BASIC_INFORMATION>()) } == 0 {
            out.push(format!("{p:#x}: query failed"));
            break;
        }
        let next = (mbi.BaseAddress as u64 + mbi.RegionSize as u64).min(end);
        let state = match mbi.State {
            0x1000 => "commit",
            0x2000 => "reserve",
            0x10000 => "free",
            _ => "?",
        };
        let ty = match mbi.Type {
            0x20000 => "private",
            0x40000 => "mapped",
            0x1000000 => "image",
            _ => "-",
        };
        out.push(format!("{p:#x}+{:#x} {state} prot {:#x} {ty}", next - p, mbi.Protect));
        if next <= p {
            break;
        }
        p = next;
    }
    out.join(", ")
}

/// Kernel-mode accesses of guest memory that failed and were retried
/// (`kernel_io`): [calls, retries, calls that failed for good].
pub static KERNEL_IO: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// Runs `op`, one kernel-mode access of guest [base, base+len) (a file read
/// into it: `into_guest`, or a file write from it). Such accesses cannot
/// fault into our exception handler: a page inaccessible for a moment (a
/// section view being split, `sections::wait_split`; a page armed for write
/// tracking) fails them with os error 998 or 1784 instead, and the title
/// sees a failed read (the 8-1 load of tour 20261015-151613 never ended).
/// They are run again after the split (pages disarmed again), a few times,
/// with what the range looked like logged.
pub fn kernel_io<T>(what: &dyn Fn() -> String, base: u64, len: u64, into_guest: bool, mut op: impl FnMut() -> std::io::Result<T>) -> std::io::Result<T> {
    use std::sync::atomic::Ordering::Relaxed;
    // Mirror mode: no-access pages with GPU-written dwords first.
    crate::gpu::resident::release_range(base, len);
    const TRIES: u32 = 8;
    let mut tries = 0;
    loop {
        let r = op();
        match &r {
            Err(e) if matches!(e.raw_os_error(), Some(998 | 1784)) && tries < TRIES => {
                if tries == 0 && KERNEL_IO[0].fetch_add(1, Relaxed) < 32 {
                    crate::warn!("{}: {e}; retrying ({})", what(), describe_host(base, len));
                }
                tries += 1;
                KERNEL_IO[1].fetch_add(1, Relaxed);
                crate::sections::wait_split_range(base, base.saturating_add(len));
                if into_guest {
                    crate::gpu::resident::disarm_for_host(base, len);
                }
                // Pages armed through the host (`hosted`; reads from
                // no-access pages fail too).
                crate::hostcensus::release(base, len);
                crate::hosted::disarm(base, len);
                std::thread::sleep(std::time::Duration::from_millis(1 << tries.min(5)));
                continue;
            }
            Err(e) if tries > 0 => {
                KERNEL_IO[2].fetch_add(1, Relaxed);
                crate::warn!("{}: still failing after {tries} retries: {e} ({})", what(), describe_host(base, len));
            }
            Ok(_) if tries > 0 && KERNEL_IO[0].load(Relaxed) <= 32 => crate::info!("{}: succeeded on retry {tries}", what()),
            _ => {}
        }
        return r;
    }
}

/// What guest [base, base+len) looks like now (diagnostics): its mappings,
/// the operating system's regions, the write tracking's page states and the
/// last view split.
pub fn describe_host(base: u64, len: u64) -> String {
    let end = base.saturating_add(len.max(1));
    let mut out = format!("{base:#x}+{len:#x}:");
    match MEM.try_lock_for(std::time::Duration::from_millis(20)) {
        Some(st) => {
            for v in st.overlapping(base, end).iter().take(4) {
                out += &format!(" mapping {:#x}+{:#x} {:?} prot {:#x} '{}';", v.start, v.size, v.kind, v.prot, v.name);
            }
        }
        None => out += " (mappings busy);",
    }
    let mut p = base;
    let mut n = 0;
    while p < end && n < 6 {
        let mut m: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        if unsafe { VirtualQuery(p as _, &mut m, std::mem::size_of::<MEMORY_BASIC_INFORMATION>()) } == 0 {
            out += " (no region)";
            break;
        }
        out += &format!(" region {:#x}+{:#x} state {:#x} protect {:#x} type {:#x};", m.BaseAddress as u64, m.RegionSize, m.State, m.Protect, m.Type);
        p = m.BaseAddress as u64 + m.RegionSize as u64;
        n += 1;
    }
    out += &format!(" {}; {}", crate::gpu::resident::describe(base, len), crate::sections::split_line());
    out
}

fn host_decommit(addr: u64, size: u64) {
    let t0 = std::time::Instant::now();
    crate::gpu::track::remapped(addr, size, crate::gpu::track::REMAP_DECOMMIT);
    repaged(addr, size);
    if defer_decommits() && crate::gpu::imports_may_cover(addr, size) {
        let mut d = DEFERRED.lock();
        d.push((addr, addr + size));
        DEFERRED_LEN.store(d.len(), std::sync::atomic::Ordering::Release);
        DEFERRED_STATS[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        DEFERRED_STATS[3].fetch_add(size, std::sync::atomic::Ordering::Relaxed);
        HOST_CALLS[4].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        return;
    }
    let _readers_out = GUEST_READERS.write();
    let _arm = crate::gpu::resident::remap_guard();
    unsafe {
        VirtualFree(addr as _, size as usize, MEM_DECOMMIT);
    }
    crate::gpu::resident::settled(addr, size);
    note_host_call(2, t0);
}

pub fn host_protect(addr: u64, size: u64, prot: u32) -> Result<(), MemError> {
    let t0 = std::time::Instant::now();
    // Mirror mode: GPU-written dwords reach guest memory before its
    // protection changes (their pages' protections are restored first).
    crate::gpu::resident::release_range(addr, size);
    crate::gpu::track::remapped(addr, size, crate::gpu::track::REMAP_PROTECT);
    let mut old = 0u32;
    let _readers_out = GUEST_READERS.write();
    let _arm = crate::gpu::resident::remap_guard();
    let ok = unsafe { VirtualProtect(addr as _, size as usize, win_prot(prot), &mut old) };
    // After the change (a failed one left armed pages armed).
    if ok != 0 {
        crate::gpu::resident::settled(addr, size);
    }
    note_host_call(5, t0);
    if ok == 0 { Err(MemError::HostFailure(last_error())) } else { Ok(()) }
}

/// A recent unmap or protection removal (`note_unmap`): its guest range,
/// when (ms since the first note, flip), the thread and its guest frames.
struct UnmapNote {
    start: u64,
    end: u64,
    op: &'static str,
    ms: u64,
    flip: u64,
    thread: String,
    /// Guest return addresses (`exception::current_guest_returns`).
    frames: Vec<u64>,
}

/// The last UNMAP_NOTES unmaps and protection removals (a separate lock:
/// fault reports read it while the memory map's lock may be held).
static UNMAPS: Mutex<std::collections::VecDeque<UnmapNote>> = Mutex::new(std::collections::VecDeque::new());
const UNMAP_NOTES: usize = 8192;

/// `DS_UNMAP_FRAMES=0` (boot): unmap notes without the unmapping thread's
/// guest frames.
fn unmap_frames() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !env_is("DS_UNMAP_FRAMES", "0"))
}

/// Notes an unmap (or a protection change to no access) for fault reports
/// (`unmaps_covering`): the title's level changes free memory that jobs
/// still read in a share of loads (FINDINGS.md 2026-10-19).
fn note_unmap(st: &MemState, op: &'static str, start: u64, end: u64) {
    static T0: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    let t = std::time::Instant::now();
    let ms = T0.get_or_init(std::time::Instant::now).elapsed().as_millis() as u64;
    let n = UnmapNote {
        start,
        end,
        op,
        ms,
        flip: crate::gpu::FRAMES.load(std::sync::atomic::Ordering::Relaxed),
        thread: crate::log::thread_name(),
        // Guest frames checked against the memory map (the caller holds
        // it), not by an operating-system query per frame: that took ~10 ms
        // per unmap while a level streamed in (11.5 s of a 20 s window at
        // 1-1, the frame rate a third of normal; FINDINGS.md 2026-10-19).
        frames: if unmap_frames() {
            crate::exception::current_guest_returns(6, &|a| st.find(a).is_some_and(|v| v.kind != VmaKind::Reserved && v.prot != 0))
        } else {
            Vec::new()
        },
    };
    // The first notes with frames, once (the walk works where it matters).
    static SHOWN: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    if !n.frames.is_empty() && SHOWN.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 3 {
        let v: Vec<String> = n.frames.iter().map(|&a| crate::exception::describe_guest(a)).collect();
        crate::info!("mem: unmap note {op} {start:#x}..{end:#x} by {}: {}", n.thread, v.join(" <- "));
    }
    let mut g = UNMAPS.lock();
    if g.len() >= UNMAP_NOTES {
        g.pop_front();
    }
    g.push_back(n);
    drop(g);
    note_host_call(7, t);
}

/// The noted unmaps covering `addr`, newest first (at most four): fault
/// reports of reads from memory the title freed. Empty while the notes are
/// being written.
pub fn unmaps_covering(addr: u64) -> Vec<String> {
    let Some(g) = UNMAPS.try_lock() else { return vec!["(notes busy)".into()] };
    let now = g.back().map_or(0, |n| n.ms);
    g.iter()
        .rev()
        .filter(|n| n.start <= addr && addr < n.end)
        .take(4)
        .map(|n| {
            let frames = if n.frames.is_empty() {
                String::new()
            } else {
                let v: Vec<String> = n.frames.iter().map(|&a| crate::exception::describe_guest(a)).collect();
                format!(": {}", v.join(" <- "))
            };
            format!("{} {:#x}..{:#x} at flip {} ({} ms before the last note) by {}{frames}", n.op, n.start, n.end, n.flip, now.saturating_sub(n.ms), n.thread)
        })
        .collect()
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
/// alias. Returns (alias slots, as `Vma::slots`; anonymous-section offset).
fn section_commit(addr: u64, size: u64, kind: VmaKind, prot: u32, phys: u64) -> Result<(u8, u64), MemError> {
    use crate::sections::{self as sec, Sec};
    crate::gpu::track::remapped(addr, size, crate::gpu::track::REMAP_COMMIT);
    repaged(addr, size);
    let (which, off) = if kind == VmaKind::Direct { (Sec::Dmem, phys) } else { (Sec::Anon, sec::anon_alloc(size).ok_or(MemError::NoSpace)?) };
    mapped_over_gpu_writes(addr, size, which == Sec::Dmem, off);
    let _arm = crate::gpu::resident::remap_guard();
    if let Err(e) = sec::map_new_view(addr, size, which, off, win_prot(prot)) {
        crate::warn!("mem: {e}");
        if which == Sec::Anon {
            sec::anon_free(off, size);
        }
        return Err(MemError::HostFailure(0));
    }
    let slots = match sec::map_alias(addr, size, which, off) {
        Ok(s) => 1u8 << s,
        Err(e) => {
            crate::warn!("mem: {e} (the GPU cannot import it)");
            0
        }
    };
    crate::gpu::resident::settled(addr, size);
    Ok((slots, if which == Sec::Anon { off } else { 0 }))
}

/// Unmaps the views of mapping `v` (`sections`): its guest view now, its
/// alias once nothing maps through it and no GPU import may use it.
fn section_release(v: &Vma) {
    crate::gpu::track::remapped(v.start, v.size, crate::gpu::track::REMAP_DECOMMIT);
    repaged(v.start, v.size);
    {
        let _readers_out = GUEST_READERS.write();
        let _arm = crate::gpu::resident::remap_guard();
        if let Err(e) = crate::sections::unmap_view(v.start) {
            crate::warn!("mem: {e}");
        }
        crate::gpu::resident::settled(v.start, v.size);
    }
    for slot in 0..crate::sections::ALIAS_SLOTS as u8 {
        if v.slots & (1 << slot) != 0 {
            crate::sections::alias_unref(v.start, v.size, slot, crate::gpu::imports_may_cover);
        }
    }
    if v.kind != VmaKind::Direct {
        crate::sections::anon_free(v.off, v.size);
    }
}

/// Guest mappings unmapped while unfinished GPU work writes them: their
/// physical pages (section offset) and the batch: with section-backed memory
/// those writes land in the pages wherever they are mapped next
/// (`section_commit` waits for the batch with `DS_PHYS_WAIT=1`). Counts:
/// [unmapped with writes pending, of them direct, mapped again while
/// pending, waits that ended in time].
static PENDING_REUSE: Mutex<Vec<(bool, u64, u64, u64, u64)>> = Mutex::new(Vec::new());
pub static PENDING_STATS: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

fn phys_wait() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| env_is("DS_PHYS_WAIT", "1"))
}

fn unmapped_with_gpu_writes(v: &Vma) {
    use std::sync::atomic::Ordering::Relaxed;
    let Some(seq) = crate::gpu::exec::pending_writes(v.start, v.size) else { return };
    let direct = v.kind == VmaKind::Direct;
    let n = PENDING_STATS[0].fetch_add(1, Relaxed);
    if direct {
        PENDING_STATS[1].fetch_add(1, Relaxed);
    }
    if n < 16 {
        crate::info!(
            "mem: {:?} {:#x}+{:#x} '{}' unmapped while batch {seq} (completed up to {}) writes it",
            v.kind,
            v.start,
            v.size,
            v.name,
            crate::gpu::exec::completed_seq()
        );
    }
    if crate::sections::on() {
        let off = if direct { v.phys } else { v.off };
        let mut g = PENDING_REUSE.lock();
        let done = crate::gpu::exec::completed_seq();
        g.retain(|e| e.3 > done);
        if g.len() < 4096 {
            g.push((direct, off, v.size, seq, v.start));
        }
    }
}

/// A new mapping of section pages [off, off+size) (`direct`: direct memory,
/// else anonymous): GPU work still writing them through an earlier mapping
/// is counted, and waited for with `DS_PHYS_WAIT=1` (at most 250 ms).
fn mapped_over_gpu_writes(addr: u64, size: u64, direct: bool, off: u64) {
    use std::sync::atomic::Ordering::Relaxed;
    let hit = {
        let mut g = PENDING_REUSE.lock();
        if g.is_empty() {
            return;
        }
        let done = crate::gpu::exec::completed_seq();
        g.retain(|e| e.3 > done);
        g.iter().filter(|e| e.0 == direct && e.1 < off + size && off < e.1 + e.2).max_by_key(|e| e.3).copied()
    };
    let Some((_, o, n, seq, va)) = hit else { return };
    let k = PENDING_STATS[2].fetch_add(1, Relaxed);
    let waited = phys_wait() && crate::gpu::exec::wait_done_for(seq, std::time::Duration::from_millis(250));
    if waited {
        PENDING_STATS[3].fetch_add(1, Relaxed);
    }
    if k < 16 {
        crate::info!(
            "mem: {} {off:#x}+{size:#x} mapped at {addr:#x} while batch {seq} still writes {o:#x}+{n:#x} through its previous mapping at {va:#x}{}",
            if direct { "direct memory" } else { "anonymous memory" },
            if phys_wait() { if waited { " (waited for it)" } else { " (wait timed out)" } } else { "" }
        );
    }
}

/// Alias views mapped by `bridge_aliases`.
pub static BRIDGES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Makes one alias slot cover guest [start, end) without a gap (`sections`;
/// a GPU import is of one slot's alias views): a binding over mappings whose
/// aliases lie in different slots (a mapping takes another slot while an
/// older alias at its addresses is still referenced or imported) could not
/// be imported (24,448 such bindings in tour 20261015-163640, bound as zeros
/// or left out). Each mapping there without an alias in the first slot that
/// is free or covered over all of them gets one, referenced like its first.
/// Returns that slot.
pub fn bridge_aliases(start: u64, end: u64) -> Option<u8> {
    use crate::sections::{self as sec, Sec};
    let mut st = MEM.lock();
    let vmas = st.overlapping(start, end);
    let mut p = start;
    for v in &vmas {
        if v.start > p || v.kind == VmaKind::Reserved {
            return None;
        }
        p = v.end();
    }
    if p < end {
        return None;
    }
    let slot = (0..sec::ALIAS_SLOTS as u8).find(|&slot| vmas.iter().all(|v| sec::alias_covers(v.start, v.size, slot) || sec::alias_free(v.start, v.size, slot)))?;
    for v in &vmas {
        if sec::alias_covers(v.start, v.size, slot) {
            continue;
        }
        let (which, off) = if v.kind == VmaKind::Direct { (Sec::Dmem, v.phys) } else { (Sec::Anon, v.off) };
        if let Err(e) = sec::map_alias_in(v.start, v.size, which, off, slot) {
            crate::warn!("mem: bridging aliases for {start:#x}..{end:#x}: {e}");
            return None;
        }
        if let Some(m) = st.vmas.get_mut(&v.start) {
            m.slots |= 1 << slot;
        }
        BRIDGES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    Some(slot)
}

/// Splits the guest view of mapping `v` at `at` (`sections`): views cannot
/// be unmapped in part, so it is unmapped and mapped again as two views of
/// the same section pages (contents kept). Faults of other threads inside it
/// meanwhile wait (`sections::wait_split`); the residency's protections of its
/// pages are reset.
fn section_split(v: &Vma, at: u64, why: usize) {
    use crate::sections::{self as sec, Sec};
    let (which, off) = if v.kind == VmaKind::Direct { (Sec::Dmem, v.phys) } else { (Sec::Anon, v.off) };
    crate::gpu::track::remapped(v.start, v.size, crate::gpu::track::REMAP_PROTECT);
    let _readers_out = GUEST_READERS.write();
    let _arm = crate::gpu::resident::remap_guard();
    sec::split_begin(v.start, v.end(), why);
    let lo = at - v.start;
    let r = sec::unmap_view(v.start)
        .and_then(|_| sec::map_view(v.start, lo, which, off, win_prot(v.prot)))
        .and_then(|_| sec::map_view(at, v.end() - at, which, off + lo, win_prot(v.prot)));
    sec::split_end();
    crate::gpu::resident::settled(v.start, v.size);
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
    fn split_at(&mut self, addr: u64, why: usize) {
        let Some(v) = self.vmas.range(..addr).next_back().map(|(_, v)| v.clone()) else { return };
        if addr > v.start && addr < v.end() {
            if crate::sections::on() && v.kind != VmaKind::Reserved {
                section_split(&v, addr, why);
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
        self.unmap_range_for(start, end, crate::sections::SPLIT_UNMAP);
    }

    /// `unmap_range`; `why` names the cause of view splits (diagnostics).
    fn unmap_range_for(&mut self, start: u64, end: u64, why: usize) {
        trace_map("unmap", start, end, format_args!(""));
        note_unmap(self, "unmap", start, end);
        bump_gen(start, end);
        self.split_at(start, why);
        self.split_at(end, why);
        let keys: Vec<u64> = self.vmas.range(start..end).map(|(k, _)| *k).collect();
        for k in keys {
            let v = self.vmas.remove(&k).unwrap();
            if matches!(v.kind, VmaKind::Direct | VmaKind::Flexible) {
                unmapped_with_gpu_writes(&v);
            }
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
            self.unmap_range_for(addr, addr + size, crate::sections::SPLIT_MAP_OVER);
        }
        let (mut slots, mut off) = (0u8, 0);
        if kind != VmaKind::Reserved {
            if crate::sections::on() {
                (slots, off) = section_commit(addr, size, kind, prot, phys)?;
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
        if kind == VmaKind::Direct {
            // Diagnostics: physical memory mapped without an allocation
            // (with sections it shows whatever those pages hold).
            let covered: u64 = self.dmem.range(..phys + size).filter(|(s, b)| **s + b.size > phys).map(|(s, b)| (s + b.size).min(phys + size) - (*s).max(phys)).sum();
            if covered < size {
                static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
                if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 16 {
                    crate::warn!("mem: direct {addr:#x}+{size:#x} maps phys {phys:#x} of which {:#x} bytes are not allocated '{name}'", size - covered);
                }
            }
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
        self.vmas.insert(addr, Vma { start: addr, size, kind, prot, phys, mtype, name: name.to_string(), slots, off });
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
        if prot & 7 == 0 {
            note_unmap(self, "protect none", start, end);
        }
        self.split_at(start, crate::sections::SPLIT_PROTECT);
        self.split_at(end, crate::sections::SPLIT_PROTECT);
        // Before the pages change too: a bulk reader that checked the table
        // before taking `GUEST_READERS` sees the generation move
        // (`table_view_held`).
        bump_gen(start, end);
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
        if crate::sections::on() {
            crate::sections::dmem_allocated(cand, len);
        }
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
