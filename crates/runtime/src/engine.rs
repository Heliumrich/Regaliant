//! The engine's own map of its memory (the title's code, PPSA01341
//! 01.007.000, read with Ghidra on 2026-11-09; FINDINGS.md, ROUTE60.md).
//!
//! The engine's page allocator (`bpe::memory::PlatformPageAllocator`) takes
//! all direct memory at start and hands it out itself: an allocation of
//! `EMemoryType::CPUMemory` (1) comes from flexible memory in 64 KiB pages,
//! every other one from direct memory in 1 MiB units. Its tracking
//! structure (`gMemoryTracking`, a pointer at eboot+0x2ee4d60) keeps one byte
//! per MiB of the address space (+0x15fa68, indexed by address >> 20) naming
//! the allocator that owns that MiB (the byte at the allocator object's
//! +0x30; 0: none). This module reads that table: which engine allocator
//! owns a guest address (`owner`), and a census of the table (`status`).
//! Reading only: nothing in the title is changed.

use std::sync::atomic::{AtomicU64, Ordering};

/// The main executable's base (0: not checked or another title version).
static BASE: AtomicU64 = AtomicU64::new(0);

/// `RequestVirtualPages_NoThreadSafety` (eboot+0x80d9a0): its first bytes,
/// the 1 MiB table's displacement in it and a RIP-relative load of the
/// tracking pointer, checked before the offsets are trusted.
const RVP: usize = 0x80d9a0;
const RVP_ENTRY: [u8; 16] = [0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54, 0x53, 0x48, 0x83, 0xe4];
const MIB_TABLE_DISP_AT: usize = 0x80ddd9;
const TRACKING_REF_AT: usize = 0x80ddb5;
/// The tracking pointer (module offset) and the per-MiB table in it.
const TRACKING_PTR: u64 = 0x2ee4d60;
const MIB_TABLE: u64 = 0x15fa68;
/// Entries of the per-MiB table (up to 0xFC00000000).
const MIB_ENTRIES: u64 = 0xFC000;

/// Checks the allocator's code and keeps the base when it is the expected
/// title version (`titlefix::apply`).
pub fn init(base: u64, image: &[u8]) {
    let entry_ok = image.get(RVP..RVP + 16) == Some(&RVP_ENTRY[..]);
    let disp_ok = image.get(MIB_TABLE_DISP_AT..MIB_TABLE_DISP_AT + 4) == Some(&(MIB_TABLE as u32).to_le_bytes()[..]);
    let ref_ok = image.get(TRACKING_REF_AT..TRACKING_REF_AT + 4).is_some_and(|b| {
        let d = i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as i64;
        (TRACKING_REF_AT as i64 + 4 + d) as u64 == TRACKING_PTR
    });
    if entry_ok && disp_ok && ref_ok {
        BASE.store(base, Ordering::Relaxed);
        crate::info!("engine map: the page allocator's tracking pointer at eboot+{TRACKING_PTR:#x} (title version as expected)");
    } else {
        crate::warn!("engine map: the page allocator's code differs (entry {entry_ok}, table {disp_ok}, pointer {ref_ok}): off");
    }
}

/// The tracking structure's address once the engine set it up.
pub fn tracking() -> Option<u64> {
    let base = BASE.load(Ordering::Relaxed);
    if base == 0 {
        return None;
    }
    let p = base + TRACKING_PTR;
    if !crate::mem::readable(p, 8) {
        return None;
    }
    let t = unsafe { std::ptr::read_volatile(p as *const u64) };
    (t != 0 && crate::mem::readable(t + MIB_TABLE, MIB_ENTRIES)).then_some(t)
}

/// The per-MiB table's address once found (it does not move).
static TABLE: AtomicU64 = AtomicU64::new(0);

/// The engine allocator owning the MiB of `addr` (its ID; 0: none or not
/// known).
#[inline]
pub fn owner(addr: u64) -> u8 {
    let i = addr >> 20;
    if i >= MIB_ENTRIES {
        return 0;
    }
    let mut t = TABLE.load(Ordering::Relaxed);
    if t == 0 {
        match tracking() {
            Some(x) => {
                t = x + MIB_TABLE;
                TABLE.store(t, Ordering::Relaxed);
            }
            None => return 0,
        }
    }
    unsafe { std::ptr::read_volatile((t + i) as *const u8) }
}

/// The census of front end 2's events by owning allocator
/// (`DS_ENGINE_CENSUS=1`): ranges bound for GPU writes (`Exec::note_write`),
/// pages uploaded into the mirror (`fe2::upload_census`), in bytes; command
/// processors' waits for GPU work before reading guest memory (count).
pub const C_GPU_WRITE: usize = 0;
pub const C_UPLOAD: usize = 1;
pub const C_READ_WAIT: usize = 2;
/// Front end 2's placements past its memo (`Exec::fe2_place`): count and ns.
pub const C_PLACE: usize = 3;
pub const C_PLACE_NS: usize = 4;
/// The hold census (`DS_FE2_HOLD=1`): GPU-written pages held, and the
/// title's reads and writes that faulted on them.
pub const C_HELD: usize = 5;
pub const C_HOLD_TR: usize = 6;
pub const C_HOLD_TW: usize = 7;
/// CPU write faults on pages front end 2 armed (`fe2` fault loop), and those
/// of them on pages it holds as GPU-written.
pub const C_ARM_FAULT: usize = 8;
pub const C_ARM_FAULT_GPU: usize = 9;
const C_KINDS: usize = 10;
static CENSUS: [[AtomicU64; C_KINDS]; 256] = [const { [const { AtomicU64::new(0) }; C_KINDS] }; 256];
/// The flip the census started at (+1; 0: not started).
static CENSUS_FLIP0: AtomicU64 = AtomicU64::new(0);

pub fn census_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_ENGINE_CENSUS").ok().as_deref() == Some("1"))
}

/// Adds the bytes of [a, a + n) to their owners' `kind` counters.
pub fn note(kind: usize, a: u64, n: u64) {
    if !census_on() || n == 0 {
        return;
    }
    if CENSUS_FLIP0.load(Ordering::Relaxed) == 0 {
        CENSUS_FLIP0.store(crate::gpu::FRAMES.load(Ordering::Relaxed) + 1, Ordering::Relaxed);
    }
    let end = a.saturating_add(n);
    let mut x = a;
    while x < end {
        let e = ((x >> 20) + 1).checked_shl(20).unwrap_or(u64::MAX).min(end);
        CENSUS[owner(x) as usize][kind].fetch_add(e - x, Ordering::Relaxed);
        x = e;
    }
    if kind == C_UPLOAD {
        note_upload_record(a, n);
    }
}

/// The page allocator's live allocations (start, size, allocator ID), sorted,
/// rebuilt at most every 2 s for the census.
static INDEX: parking_lot::Mutex<(Option<std::time::Instant>, Vec<(u64, u64, u8)>)> = parking_lot::Mutex::new((None, Vec::new()));
/// Bytes uploaded into each allocation (by start): (ID, size, bytes).
static UP_BY_REC: parking_lot::Mutex<Option<std::collections::HashMap<u64, (u8, u64, u64)>>> = parking_lot::Mutex::new(None);

/// The live allocations of both pools (start, size, allocator ID).
fn records() -> Vec<(u64, u64, u8)> {
    let Some(t) = tracking() else { return Vec::new() };
    let mut v = Vec::new();
    for head in [t, t + 0xa0] {
        let Some(mut n) = rd64(head + 0x10) else { continue };
        let mut steps = 0;
        while n != head && n != 0 && steps < 20_000 {
            steps += 1;
            let rec = n - 0x20;
            let (Some(alloc), Some(size), Some(addr)) = (rd64(rec + 0x40), rd64(rec + 0x58), rd64(rec + 0x60)) else { break };
            let id = if alloc != 0 && crate::mem::readable(alloc + 0x30, 1) { unsafe { std::ptr::read_volatile((alloc + 0x30) as *const u8) } } else { 0 };
            v.push((addr, size, id));
            let Some(next) = rd64(n + 0x10) else { break };
            n = next;
        }
    }
    v.sort_unstable();
    v
}

/// Census: uploads of [a, a + n) by the allocation holding `a`.
fn note_upload_record(a: u64, n: u64) {
    let rec = {
        let mut g = INDEX.lock();
        if g.0.is_none_or(|t| t.elapsed().as_secs() >= 2) {
            g.1 = records();
            g.0 = Some(std::time::Instant::now());
        }
        let i = g.1.partition_point(|r| r.0 <= a);
        (i > 0).then(|| g.1[i - 1]).filter(|r| a < r.0 + r.1)
    };
    // (No record: the MiB, named by its owner in the per-MiB table, as id +
    // 100 in the report.)
    let (start, size, id) = rec.unwrap_or((a & !0xFFFFF, 0, owner(a).saturating_add(100)));
    let mut g = UP_BY_REC.lock();
    let e = g.get_or_insert_with(Default::default).entry(start).or_insert((id, size, 0));
    e.2 += n;
}

/// Counts one event of `kind` at `a`'s owner.
pub fn note_count(kind: usize, a: u64) {
    note_value(kind, a, 1);
}

/// Adds `v` to `a`'s owner's `kind` counter.
pub fn note_value(kind: usize, a: u64, v: u64) {
    if !census_on() {
        return;
    }
    CENSUS[owner(a) as usize][kind].fetch_add(v, Ordering::Relaxed);
}

/// Each allocator ID's name and memory type ("name (GPU)").
pub fn names() -> std::collections::HashMap<u8, String> {
    allocators().into_iter().map(|a| (a.id, format!("{} ({})", a.name, if a.mtype == 1 { "CPU" } else if a.mtype == 2 { "GPU" } else { "?" }))).collect()
}

/// The census line (per flip since it started).
fn census_line() -> String {
    if !census_on() {
        return String::new();
    }
    let f0 = CENSUS_FLIP0.load(Ordering::Relaxed);
    if f0 == 0 {
        return String::new();
    }
    let flips = (crate::gpu::FRAMES.load(Ordering::Relaxed) + 1).saturating_sub(f0).max(1) as f64;
    let names = names();
    let mut rows: Vec<(u64, String)> = Vec::new();
    for id in 0..256usize {
        let c: Vec<u64> = CENSUS[id].iter().map(|x| x.load(Ordering::Relaxed)).collect();
        if c.iter().all(|&x| x == 0) {
            continue;
        }
        let name = names.get(&(id as u8)).cloned().unwrap_or_else(|| if id == 0 { "none".into() } else { "?".into() });
        let armf = if c[C_ARM_FAULT] > 0 { format!(", write faults on armed pages {:.2} ({:.2} on GPU-written ones)", c[C_ARM_FAULT] as f64 / flips, c[C_ARM_FAULT_GPU] as f64 / flips) } else { String::new() };
        let held = if c[C_HELD] + c[C_HOLD_TR] + c[C_HOLD_TW] > 0 { format!(", held {:.0} pages, the title faulted on them {:.2} reads / {:.2} writes", c[C_HELD] as f64 / flips, c[C_HOLD_TR] as f64 / flips, c[C_HOLD_TW] as f64 / flips) } else { String::new() };
        rows.push((c[C_PLACE_NS], format!("{id} {name}: placements {:.0} ({:.2} ms), GPU-written {:.2} MiB, uploaded {:.2} MiB, read waits {:.2}{armf}{held}", c[C_PLACE] as f64 / flips, c[C_PLACE_NS] as f64 / flips / 1e6, c[C_GPU_WRITE] as f64 / flips / 1048576.0, c[C_UPLOAD] as f64 / flips / 1048576.0, c[C_READ_WAIT] as f64 / flips)));
    }
    rows.sort_by(|a, b| b.0.cmp(&a.0));
    // Uploads by allocation: how many allocations of each allocator take
    // them, and the largest.
    let mut recs: Vec<(u64, (u8, u64, u64))> = UP_BY_REC.lock().as_ref().map(|m| m.iter().map(|(k, v)| (*k, *v)).collect()).unwrap_or_default();
    recs.sort_by(|a, b| b.1 .2.cmp(&a.1 .2));
    let mut per_id: std::collections::BTreeMap<u8, (u64, u64)> = std::collections::BTreeMap::new();
    for (_, (id, _, bytes)) in &recs {
        let e = per_id.entry(*id).or_insert((0, 0));
        e.0 += 1;
        e.1 += bytes;
    }
    let ups: Vec<String> = per_id.iter().map(|(id, (k, b))| format!("{id}: {k} allocations, {:.2} MiB", *b as f64 / flips / 1048576.0)).collect();
    let top: Vec<String> = recs.iter().take(16).map(|(start, (id, size, b))| format!("{start:#x}+{size:#x} (id {id}) {:.3} MiB", *b as f64 / flips / 1048576.0)).collect();
    format!("    engine census per flip over {flips:.0} flips, by owning allocator: {}\n", rows.into_iter().map(|r| r.1).collect::<Vec<_>>().join("; "))
        + &format!("    engine census: uploads a flip by allocation: {}; largest: {}\n", ups.join(", "), top.join(", "))
}

/// MiB per owning allocator ID now, with each ID's lowest and highest
/// address (the census of the table).
pub fn census() -> Vec<(u8, u64, u64, u64)> {
    let Some(t) = tracking() else { return Vec::new() };
    let table = unsafe { std::slice::from_raw_parts((t + MIB_TABLE) as *const u8, MIB_ENTRIES as usize) };
    let mut per: [(u64, u64, u64); 256] = [(0, u64::MAX, 0); 256];
    for (i, &id) in table.iter().enumerate() {
        if id != 0 {
            let e = &mut per[id as usize];
            e.0 += 1;
            e.1 = e.1.min((i as u64) << 20);
            e.2 = e.2.max(((i as u64) + 1) << 20);
        }
    }
    per.iter().enumerate().filter(|(_, e)| e.0 > 0).map(|(id, e)| (id as u8, e.0, e.1, e.2)).collect()
}

/// A u64 of guest memory when readable.
fn rd64(a: u64) -> Option<u64> {
    (a != 0 && crate::mem::readable(a, 8)).then(|| unsafe { std::ptr::read_volatile(a as *const u64) })
}

/// A NUL-terminated printable string of guest memory (at most 120 bytes).
fn rdstr(a: u64) -> Option<String> {
    if a == 0 || !crate::mem::readable(a, 1) {
        return None;
    }
    let mut s = String::new();
    for k in 0..120u64 {
        if !crate::mem::readable(a + k, 1) {
            return None;
        }
        let c = unsafe { std::ptr::read_volatile((a + k) as *const u8) };
        if c == 0 {
            return (s.len() >= 2).then_some(s);
        }
        if !(0x20..0x7f).contains(&c) {
            return None;
        }
        s.push(c as char);
    }
    None
}

/// One engine allocator seen in the page allocator's records.
pub struct Allocator {
    pub ptr: u64,
    pub id: u8,
    /// Its name (the base constructor's argument, a pointer at +0x10).
    pub name: String,
    /// Its `EMemoryType` (+0x18): 1 CPU memory, 2 GPU memory.
    pub mtype: u32,
    pub allocs: u64,
    pub bytes: u64,
    pub lo: u64,
    pub hi: u64,
    /// The pool: 0 flexible (CPU) memory, 1 direct memory.
    pub pool: u8,
}

/// The allocators owning the page allocator's live allocations (the base
/// constructor `IAllocator(name, type)` keeps the name at +0x10, the memory
/// type at +0x18; the ID byte is at +0x30), by walking
/// each pool's record list (records of 0x78 bytes: list node at +0x20 with
/// its previous and next at +0x28 and +0x30, the allocator at +0x40, size
/// +0x58, address +0x60; the pools' list heads at the tracking structure's
/// +0x00 and +0xa0).
pub fn allocators() -> Vec<Allocator> {
    let Some(t) = tracking() else { return Vec::new() };
    let mut by: std::collections::BTreeMap<(u8, u64), Allocator> = std::collections::BTreeMap::new();
    for (pool, head) in [(0u8, t), (1u8, t + 0xa0)] {
        let Some(mut n) = rd64(head + 0x10) else { continue };
        let mut steps = 0;
        while n != head && n != 0 && steps < 20_000 {
            steps += 1;
            let rec = n - 0x20;
            let (Some(alloc), Some(size), Some(addr)) = (rd64(rec + 0x40), rd64(rec + 0x58), rd64(rec + 0x60)) else { break };
            let e = by.entry((pool, alloc)).or_insert_with(|| Allocator {
                ptr: alloc,
                id: if alloc != 0 && crate::mem::readable(alloc + 0x30, 1) { unsafe { std::ptr::read_volatile((alloc + 0x30) as *const u8) } } else { 0 },
                name: rd64(alloc + 0x10).and_then(rdstr).unwrap_or_default(),
                mtype: if alloc != 0 && crate::mem::readable(alloc + 0x18, 4) { unsafe { std::ptr::read_volatile((alloc + 0x18) as *const u32) } } else { 0 },
                allocs: 0,
                bytes: 0,
                lo: u64::MAX,
                hi: 0,
                pool,
            });
            e.allocs += 1;
            e.bytes += size;
            e.lo = e.lo.min(addr);
            e.hi = e.hi.max(addr + size);
            let Some(next) = rd64(n + 0x10) else { break };
            n = next;
        }
    }
    let mut v: Vec<Allocator> = by.into_values().collect();
    v.sort_by(|a, b| b.bytes.cmp(&a.bytes));
    v
}

/// The GPU per-frame arena's allocated units (start, length), merged into
/// runs: its object (the `Default TaggedArena` of GPU memory) keeps groups
/// of block descriptors in a list at +0x518 (each group's used blocks in a
/// list at +0x10), a descriptor the block's base at +0x10 and a 120-unit
/// occupancy bitmap at +0x18 and +0x20; the unit size at the arena's +0x88
/// (1 MiB). (eboot+0x7fe350 allocates a chunk, +0x7fea60 the units in a
/// block, +0x7fefc0 makes a block.)
pub fn arena_units() -> Vec<(u64, u64)> {
    let Some(arena) = allocators().into_iter().find(|a| a.pool == 1 && a.mtype == 2 && a.name == "Default TaggedArena").map(|a| a.ptr) else { return Vec::new() };
    let Some(unit) = rd64(arena + 0x88).filter(|&u| u.is_power_of_two() && u <= 1 << 26) else { return Vec::new() };
    let mut runs: Vec<(u64, u64)> = Vec::new();
    let head = arena + 0x518;
    let Some(mut g) = rd64(head) else { return Vec::new() };
    let mut groups = 0;
    while g != head && g != 0 && groups < 64 {
        groups += 1;
        let used = g + 0x10;
        let Some(mut d) = rd64(used) else { break };
        let mut blocks = 0;
        while d != used && d != 0 && blocks < 64 {
            blocks += 1;
            let (Some(base), Some(b0), Some(b1)) = (rd64(d + 0x10), rd64(d + 0x18), rd64(d + 0x20)) else { break };
            for u in 0..120u64 {
                let set = if u < 64 { b0 >> u & 1 } else { b1 >> (u - 64) & 1 } != 0;
                if set {
                    let s = base + u * unit;
                    match runs.last_mut() {
                        Some(r) if r.0 + r.1 == s => r.1 += unit,
                        _ => runs.push((s, unit)),
                    }
                }
            }
            let Some(next) = rd64(d) else { break };
            d = next;
        }
        let Some(next) = rd64(g) else { break };
        g = next;
    }
    runs
}

/// The GPU arena's live tags: (tag index, tag value, units, MiB runs).
/// The tag table holds 64 values at the arena's +0x528 with an occupancy
/// mask at +0x728; each block descriptor has a 128-bit unit bitmap per tag
/// at +0x28 + 16 x index (eboot+0x7fe540 frees a tag's chunks).
pub fn arena_tags() -> Vec<(u32, u64, u64, Vec<(u64, u64)>)> {
    let Some(arena) = allocators().into_iter().find(|a| a.pool == 1 && a.mtype == 2 && a.name == "Default TaggedArena").map(|a| a.ptr) else { return Vec::new() };
    let (Some(unit), Some(mask)) = (rd64(arena + 0x88), rd64(arena + 0x728)) else { return Vec::new() };
    let mut out = Vec::new();
    for i in 0..64u32 {
        if mask >> i & 1 == 0 {
            continue;
        }
        let tag = rd64(arena + 0x528 + 8 * i as u64).unwrap_or(0);
        let mut runs: Vec<(u64, u64)> = Vec::new();
        let mut units = 0u64;
        let head = arena + 0x518;
        let mut g = rd64(head).unwrap_or(0);
        let mut gn = 0;
        while g != head && g != 0 && gn < 64 {
            gn += 1;
            let used = g + 0x10;
            let mut d = rd64(used).unwrap_or(0);
            let mut dn = 0;
            while d != used && d != 0 && dn < 64 {
                dn += 1;
                let (Some(base), Some(b0), Some(b1)) = (rd64(d + 0x10), rd64(d + 0x28 + 16 * i as u64), rd64(d + 0x30 + 16 * i as u64)) else { break };
                for u in 0..120u64 {
                    if (if u < 64 { b0 >> u & 1 } else { b1 >> (u - 64) & 1 }) != 0 {
                        units += 1;
                        let st = base + u * unit;
                        match runs.last_mut() {
                            Some(r) if r.0 + r.1 == st => r.1 += unit,
                            _ => runs.push((st, unit)),
                        }
                    }
                }
                d = rd64(d).unwrap_or(0);
            }
            g = rd64(g).unwrap_or(0);
        }
        out.push((i, tag, units, runs));
    }
    out
}

/// For the status report.
pub fn status() -> String {
    if BASE.load(Ordering::Relaxed) == 0 {
        return String::new();
    }
    let c = census();
    if c.is_empty() {
        return "    engine map: no tracking structure yet\n".into();
    }
    let total: u64 = c.iter().map(|x| x.1).sum();
    let parts: Vec<String> = c.iter().map(|&(id, mib, lo, hi)| format!("{id}: {mib} MiB ({lo:#x}..{hi:#x})")).collect();
    let mut s = format!("    engine map: {total} MiB of direct memory owned by {} allocators: {}\n", c.len(), parts.join(", "));
    let a = allocators();
    let parts: Vec<String> = a
        .iter()
        .take(40)
        .map(|x| {
            // (The vtable module-relative when in the executable: the
            // allocator's class through the Ghidra project.)
            let base = BASE.load(Ordering::Relaxed);
            let vt = rd64(x.ptr).map(|v| if v >= base && v < base + 0x4000_0000 { format!("eboot+{:#x}", v - base) } else { format!("{v:#x}") }).unwrap_or_default();
            let obj = if x.ptr >= base && x.ptr < base + 0x4000_0000 { format!("eboot+{:#x}", x.ptr - base) } else { format!("{:#x}", x.ptr) };
            format!("{} id {} '{}' type {} {} allocations {} MiB ({:#x}..{:#x}) object {obj} vtable {vt}", if x.pool == 0 { "flexible" } else { "direct" }, x.id, x.name, x.mtype, x.allocs, x.bytes >> 20, x.lo, x.hi)
        })
        .collect();
    s += &format!("    engine allocators ({}): {}\n", a.len(), parts.join("; "));
    let units = arena_units();
    let total: u64 = units.iter().map(|r| r.1).sum();
    let parts: Vec<String> = units.iter().take(24).map(|r| format!("{:#x}+{:#x}", r.0, r.1)).collect();
    s += &format!("    engine GPU arena: {} MiB allocated in {} runs: {}
", total >> 20, units.len(), parts.join(", "));
    let tags: Vec<String> = arena_tags().iter().map(|(i, tag, n, runs)| format!("[{i}] tag {tag:#x}: {n} units in {} runs", runs.len())).collect();
    s += &format!("    engine GPU arena tags: {}
", tags.join(", "));
    s + &census_line()
}
