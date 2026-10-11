//! Guest virtual memory.
//!
//! The PS4 address range used by the guest is reserved once in the host process
//! and pages are committed/decommitted at 4 KiB granularity inside it, so guest
//! pointers are host pointers. A VMA map mirrors what the guest has mapped, and a
//! separate allocator models PS4 "direct memory" (physical) ranges.
//!
//! Limitation (reported, never silent): mapping the same direct-memory range at
//! two virtual addresses (aliasing) is not shared; it is detected and logged.

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
}

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

/// `readable_span` calls and cache misses (diagnostics).
static SPAN_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static SPAN_MISSES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// One status line (diagnostics).
pub fn remap_status() -> String {
    let r = REMAP.lock();
    let o = std::sync::atomic::Ordering::Relaxed;
    format!(
        "    direct memory remaps: {} in place ({} MiB, zeroed here), {} of pages last mapped elsewhere ({} MiB, contents not carried); mapping changes {}; readable-range checks {} ({} cache misses)
",
        r.in_place,
        r.in_place_bytes >> 20,
        r.moved,
        r.moved_bytes >> 20,
        MAP_GEN.load(o),
        SPAN_CALLS.load(o),
        SPAN_MISSES.load(o)
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
}

/// The run of adjacent accessible mappings containing `a` as [start, end)
/// (at most 64 mappings either side), from a per-thread cache when the
/// mappings have not changed since (the GPU command processors check every
/// descriptor and buffer they read, ~2.3 million times per second in the
/// courtyard; the mapping table lock is shared with the title's threads).
pub fn readable_span(a: u64) -> Option<(u64, u64)> {
    SPAN_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
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

/// Reserves the whole guest range. Must run before anything else allocates widely.
pub fn reserve_guest_range() -> Result<(), String> {
    let p = unsafe { VirtualAlloc(GUEST_BASE as _, (GUEST_END - GUEST_BASE) as usize, MEM_RESERVE, PAGE_NOACCESS) };
    if p.is_null() {
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
    crate::gpu::track::remapped(addr, size);
    repaged(addr, size);
    let p = unsafe { VirtualAlloc(addr as _, size as usize, MEM_COMMIT, win_prot(prot)) };
    if p.is_null() { Err(MemError::HostFailure(last_error())) } else { Ok(()) }
}

fn host_decommit(addr: u64, size: u64) {
    crate::gpu::track::remapped(addr, size);
    repaged(addr, size);
    unsafe {
        VirtualFree(addr as _, size as usize, MEM_DECOMMIT);
    }
}

pub fn host_protect(addr: u64, size: u64, prot: u32) -> Result<(), MemError> {
    crate::gpu::track::remapped(addr, size);
    let mut old = 0u32;
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
            let mut lo = v.clone();
            lo.size = addr - v.start;
            let mut hi = v;
            hi.size = hi.end() - addr;
            if hi.kind == VmaKind::Direct {
                hi.phys += lo.size;
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
            if v.kind != VmaKind::Reserved {
                host_decommit(v.start, v.size);
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
        if kind != VmaKind::Reserved {
            host_commit(addr, size, prot)?;
        }
        if kind == VmaKind::Direct {
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
        self.vmas.insert(addr, Vma { start: addr, size, kind, prot, phys, mtype, name: name.to_string() });
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
