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

/// Incremented whenever guest mappings change (GPU zero-copy imports of guest
/// memory are revalidated against it).
pub static MAP_GEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

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
    let p = unsafe { VirtualAlloc(addr as _, size as usize, MEM_COMMIT, win_prot(prot)) };
    if p.is_null() { Err(MemError::HostFailure(last_error())) } else { Ok(()) }
}

fn host_decommit(addr: u64, size: u64) {
    crate::gpu::track::remapped(addr, size);
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
        MAP_GEN.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        self.split_at(start);
        self.split_at(end);
        let keys: Vec<u64> = self.vmas.range(start..end).map(|(k, _)| *k).collect();
        for k in keys {
            let v = self.vmas.remove(&k).unwrap();
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
        MAP_GEN.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
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
