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
        alloc_table_init();
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
/// Of the placements: those of ranges bound for GPU writes, and those past
/// a label wait (count and ns).
pub const C_PLACE_W: usize = 10;
pub const C_PLACE_W_NS: usize = 11;
pub const C_PLACE_PAST: usize = 12;
pub const C_PLACE_PAST_NS: usize = 13;
/// Of the placements: unbounded bindings (base pointers whose loads the
/// translation cannot bound: up to 256 MiB), count and ns.
pub const C_PLACE_UNB: usize = 14;
pub const C_PLACE_UNB_NS: usize = 15;
const C_KINDS: usize = 16;
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
        let id = owner(x);
        CENSUS[id as usize][kind].fetch_add(e - x, Ordering::Relaxed);
        note_tag(kind, x, e - x, id);
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

/// Bytes uploaded by owner and the programs of the command whose placement
/// uploaded them (census).
static UP_BY_PROG: parking_lot::Mutex<Option<std::collections::HashMap<(u8, u64, u64), u64>>> = parking_lot::Mutex::new(None);

/// Census: an upload of [a, a + n) for a command of programs `p0`, `p1`.
pub fn note_upload_prog(a: u64, n: u64, p0: u64, p1: u64) {
    if !census_on() || n == 0 {
        return;
    }
    let id = owner(a);
    *UP_BY_PROG.lock().get_or_insert_with(Default::default).entry((id, p0, p1)).or_insert(0) += n;
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
    let id = owner(a);
    CENSUS[id as usize][kind].fetch_add(v, Ordering::Relaxed);
    note_tag(kind, a, v, id);
}

/// The GPU arena's allocator ID once known (0: not yet).
static ARENA_ID: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);
/// The GPU arena's tagged unit runs (start, length, tag), sorted, rebuilt at
/// most every 2 s for the census.
static TAG_INDEX: parking_lot::Mutex<(Option<std::time::Instant>, Vec<(u64, u64, u64)>)> = parking_lot::Mutex::new((None, Vec::new()));
/// Census events in the GPU arena by tag (frame allocator ID << 32 | slot):
/// (tag, kind) -> value.
static BY_TAG: parking_lot::Mutex<Option<std::collections::HashMap<(u64, usize), u64>>> = parking_lot::Mutex::new(None);

/// The tag owning arena address `a` (census).
fn arena_tag_of(a: u64) -> Option<u64> {
    let mut g = TAG_INDEX.lock();
    if g.0.is_none_or(|t| t.elapsed().as_secs() >= 2) {
        let mut v: Vec<(u64, u64, u64)> = arena_tags().into_iter().flat_map(|(_, tag, _, runs)| runs.into_iter().map(move |(s, n)| (s, n, tag))).collect();
        v.sort_unstable();
        g.1 = v;
        g.0 = Some(std::time::Instant::now());
    }
    let i = g.1.partition_point(|r| r.0 <= a);
    (i > 0).then(|| g.1[i - 1]).filter(|r| a < r.0 + r.1).map(|r| r.2)
}

/// Census: an event in the GPU arena counted by its tag too.
fn note_tag(kind: usize, a: u64, v: u64, id: u8) {
    let mut aid = ARENA_ID.load(Ordering::Relaxed);
    if aid == 0 {
        aid = allocators().into_iter().find(|x| x.pool == 1 && x.mtype == 2 && x.name == "Default TaggedArena").map_or(0, |x| x.id);
        if aid == 0 {
            return;
        }
        ARENA_ID.store(aid, Ordering::Relaxed);
    }
    if id != aid {
        return;
    }
    let tag = arena_tag_of(a).unwrap_or(u64::MAX);
    *BY_TAG.lock().get_or_insert_with(Default::default).entry((tag, kind)).or_insert(0) += v;
}

/// The arena-by-tag census line, by frame allocator (tag >> 32).
fn tag_line(flips: f64) -> String {
    let m: Vec<((u64, usize), u64)> = BY_TAG.lock().as_ref().map(|m| m.iter().map(|(k, v)| (*k, *v)).collect()).unwrap_or_default();
    if m.is_empty() {
        return String::new();
    }
    let mut by: std::collections::BTreeMap<u64, [u64; C_KINDS]> = std::collections::BTreeMap::new();
    for ((tag, kind), v) in m {
        let k = if tag == u64::MAX { u64::MAX } else { tag >> 32 };
        by.entry(k).or_insert([0; C_KINDS])[kind] += v;
    }
    let parts: Vec<String> = by
        .iter()
        .map(|(k, c)| {
            let who = if *k == u64::MAX { "no tag".to_string() } else { format!("frame allocator {k:#x}") };
            format!("{who}: uploaded {:.2} MiB, placements {:.0} ({:.2} ms), GPU-written {:.2} MiB, read waits {:.2}", c[C_UPLOAD] as f64 / flips / 1048576.0, c[C_PLACE] as f64 / flips, c[C_PLACE_NS] as f64 / flips / 1e6, c[C_GPU_WRITE] as f64 / flips / 1048576.0, c[C_READ_WAIT] as f64 / flips)
        })
        .collect();
    format!("    engine census: the GPU arena by frame allocator (tag): {}\n", parts.join("; "))
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
        let split = if c[C_PLACE_W] + c[C_PLACE_PAST] + c[C_PLACE_UNB] > 0 { format!(" (bound for GPU writes {:.0}, {:.2} ms; past label waits {:.0}, {:.2} ms; unbounded {:.0}, {:.2} ms)", c[C_PLACE_W] as f64 / flips, c[C_PLACE_W_NS] as f64 / flips / 1e6, c[C_PLACE_PAST] as f64 / flips, c[C_PLACE_PAST_NS] as f64 / flips / 1e6, c[C_PLACE_UNB] as f64 / flips, c[C_PLACE_UNB_NS] as f64 / flips / 1e6) } else { String::new() };
        rows.push((c[C_PLACE_NS], format!("{id} {name}: placements {:.0} ({:.2} ms){split}, GPU-written {:.2} MiB, uploaded {:.2} MiB, read waits {:.2}{armf}{held}", c[C_PLACE] as f64 / flips, c[C_PLACE_NS] as f64 / flips / 1e6, c[C_GPU_WRITE] as f64 / flips / 1048576.0, c[C_UPLOAD] as f64 / flips / 1048576.0, c[C_READ_WAIT] as f64 / flips)));
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
    // Uploads by the programs whose placements made them, for the CPU-type
    // heap, the arena and the GPU page allocator.
    let mut progs: Vec<((u8, u64, u64), u64)> = UP_BY_PROG.lock().as_ref().map(|m| m.iter().map(|(k, v)| (*k, *v)).collect()).unwrap_or_default();
    progs.sort_by(|a, b| b.1.cmp(&a.1));
    let mut by_prog = String::new();
    for id in [4u8, 11u8, 9u8] {
        let v: Vec<String> = progs.iter().filter(|x| x.0 .0 == id).take(10).map(|((_, p0, p1), b)| format!("{p0:#x}/{p1:#x} {:.3} MiB", *b as f64 / flips / 1048576.0)).collect();
        if !v.is_empty() {
            by_prog += &format!("    engine census: uploads a flip into {id} by the programs whose placement made them: {}\n", v.join(", "));
        }
    }
    format!("    engine census per flip over {flips:.0} flips, by owning allocator: {}\n", rows.into_iter().map(|r| r.1).collect::<Vec<_>>().join("; "))
        + &format!("    engine census: uploads a flip by allocation: {}; largest: {}\n", ups.join(", "), top.join(", "))
        + &by_prog
        + &tag_line(flips)
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

/// The engine's frame allocators (`TaggedFrameAllocator`) and their arenas,
/// registered by a title-fix hook at their init (eboot+0x8114a0: rdi the
/// allocator, rsi its arena).
static FRAME_ALLOCS: parking_lot::Mutex<Vec<(u64, u64)>> = parking_lot::Mutex::new(Vec::new());

/// The hook at a frame allocator's init (`titlefix`).
pub extern "sysv64" fn frame_alloc_hook(regs: *const u64, _arg: u32) {
    let (this, arena) = unsafe { (*regs.add(7), *regs.add(6)) };
    {
        let mut g = FRAME_ALLOCS.lock();
        if !g.iter().any(|x| x.0 == this) {
            g.push((this, arena));
        }
    }
    // The GPU arena (C2's allocation code compares allocators' arenas with
    // it): its memory type (+0x18) GPU and its name the default tagged
    // arena's.
    if GPU_ARENA.load(Ordering::Relaxed) == 0 && arena != 0 && crate::mem::readable(arena + 0x18, 4) && unsafe { std::ptr::read_volatile((arena + 0x18) as *const u32) } == 2 && rd64(arena + 0x10).and_then(rdstr).as_deref() == Some("Default TaggedArena") {
        GPU_ARENA.store(arena, Ordering::Release);
        crate::info!("engine map: the GPU arena at {arena:#x} (a frame allocator's init)");
    }
}

/// The GPU arena's object once found (0: not yet).
static GPU_ARENA: AtomicU64 = AtomicU64::new(0);
/// Each frame allocator slot's cursor and chunk end at the last submission.
static CURSORS: parking_lot::Mutex<Option<std::collections::HashMap<(u64, u32), (u64, u64)>>> = parking_lot::Mutex::new(None);
/// At submissions, the GPU arena's frame allocators' cursors: [bytes advanced
/// within a chunk, chunk switches, bytes left in the old chunks at a switch,
/// submissions sampled, cursors past their MiB's high-water mark (the
/// allocation table missed allocations), the largest such excess, cursors
/// behind it, cursors compared].
static ADV: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];

/// A title submission is about to be handed to the GPU (`gpu::submit`, on the
/// title's thread): the census samples the GPU arena's frame allocators'
/// cursors (C2's measure: what the title wrote there since the last one).
pub fn on_submit() {
    if !census_on() {
        return;
    }
    let mut arena = GPU_ARENA.load(Ordering::Relaxed);
    if arena == 0 {
        arena = allocators().into_iter().find(|a| a.pool == 1 && a.mtype == 2 && a.name == "Default TaggedArena").map_or(0, |a| a.ptr);
        if arena == 0 {
            return;
        }
        GPU_ARENA.store(arena, Ordering::Relaxed);
    }
    let fas: Vec<u64> = FRAME_ALLOCS.lock().iter().filter(|x| x.1 == arena).map(|x| x.0).collect();
    let mut g = CURSORS.lock();
    let m = g.get_or_insert_with(Default::default);
    for fa in fas {
        for slot in 0..21u32 {
            let base = if slot < 20 { fa + 0xb8 + slot as u64 * 0x30 } else { fa + 0x478 };
            let (Some(cur), Some(rem)) = (rd64(base), rd64(base + 8)) else { continue };
            if cur == 0 {
                continue;
            }
            let end = cur.saturating_add(rem);
            // The allocation table's completeness: the slot's cursor (the end
            // of its last allocation) against the high-water mark of the MiB
            // holding its last byte. The inserted code stores the mark before
            // the allocator moves the cursor, so a cursor past it means
            // allocations the code never saw.
            {
                let lm = (cur - 1) >> 20;
                let hwm = alloc_entry(lm) & 0x1F_FFFF;
                let off = cur - (lm << 20);
                if off > hwm {
                    ADV[4].fetch_add(1, Ordering::Relaxed);
                    ADV[5].fetch_max(off - hwm, Ordering::Relaxed);
                } else if off < hwm {
                    ADV[6].fetch_add(1, Ordering::Relaxed);
                }
                ADV[7].fetch_add(1, Ordering::Relaxed);
            }
            if let Some(&(pc, pe)) = m.get(&(fa, slot)) {
                if cur >= pc && cur <= pe {
                    ADV[0].fetch_add(cur - pc, Ordering::Relaxed);
                } else if cur != pc {
                    ADV[1].fetch_add(1, Ordering::Relaxed);
                    ADV[2].fetch_add(pe.saturating_sub(pc), Ordering::Relaxed);
                }
            }
            m.insert((fa, slot), (cur, end));
        }
    }
    drop(g);
    ADV[3].fetch_add(1, Ordering::Relaxed);
    arena_scan_census();
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
    if census_on() {
        let fas = FRAME_ALLOCS.lock().clone();
        let arena = GPU_ARENA.load(Ordering::Relaxed);
        let flips = crate::gpu::FRAMES.load(Ordering::Relaxed).max(1) as f64;
        let a: Vec<u64> = ADV.iter().map(|x| x.load(Ordering::Relaxed)).collect();
        s += &format!(
            "    engine frame allocators: {} registered ({} on the GPU arena); at {:.1} submissions a flip their cursors advanced {:.2} MiB a flip in place, {:.1} chunk switches a flip (old chunks' rests {:.2} MiB)
",
            fas.len(),
            fas.iter().filter(|x| x.1 == arena).count(),
            a[3] as f64 / flips,
            a[0] as f64 / flips / 1048576.0,
            a[1] as f64 / flips,
            a[2] as f64 / flips / 1048576.0
        );
        s += &format!("    engine allocation table against the cursors: {} compared, {} past their MiB's high-water mark (largest by {:#x}), {} behind it
", a[7], a[4], a[5], a[6]);
    }
    s + &census_line() + &bound_line()
}

// ---------------------------------------------------------------------------
// The GPU arena's allocation table (front end 4, docs/FRONTEND4.md). A frame
// allocator carves its allocations in order from chunks of whole arena units
// (1 MiB: the arena's granularity, its vtable +0xd0 returning the unit size
// at +0x88), one chunk per thread slot, and a unit is reused only after its
// tag was freed (eboot+0x7fe540: every release of the arena's units goes
// through it). So per MiB of the arena, the allocations of one tag's
// lifetime only grow, and the allocated bytes are known exactly (the
// census checks the cursors against it at title submissions). Two uses were
// priced on 2026-11-09 night and removed (TRIED.md): a memo of uploaded
// ranges across uses (not exact: a binding's range is not what its shader
// reads, and bytes inside a range were written after its first use), and
// unbounded pointers bounded by their frame allocations' extent (the vista
// level, the swamp x0.98: the graphics queue's uploads moved onto a compute
// queue's chain). The table stays for the census and front end 3's scope.
// ---------------------------------------------------------------------------

/// Per MiB of the address space (index address >> 20): the GPU arena's
/// frame allocations, (generation << 32) | high-water mark within the MiB
/// (0..=1 MiB). Written by the code inserted into the frame allocator
/// (`alloc_insert_code`) and the tag-free hook; 0: never allocated.
static ALLOC: std::sync::atomic::AtomicPtr<u64> = std::sync::atomic::AtomicPtr::new(std::ptr::null_mut());

/// Allocates the allocation table (`init`, before the title fixes).
fn alloc_table_init() {
    if !ALLOC.load(Ordering::Relaxed).is_null() {
        return;
    }
    let layout = std::alloc::Layout::array::<u64>(MIB_ENTRIES as usize).expect("layout");
    let p = unsafe { std::alloc::alloc_zeroed(layout) } as *mut u64;
    if !p.is_null() {
        ALLOC.store(p, Ordering::Release);
    }
}

/// The table entry of MiB `m` (`ALLOC`).
#[inline]
fn alloc_entry(m: u64) -> u64 {
    let t = ALLOC.load(Ordering::Relaxed);
    if t.is_null() || m >= MIB_ENTRIES {
        return 0;
    }
    unsafe { std::ptr::read_volatile(t.add(m as usize)) }
}

/// Code inserted before the frame allocator's `lea rcx,[rax+r12+7]`
/// (eboot+0x8121b2, where an allocation is placed: rax its start, r12 its
/// size, r15 the allocator; rcx, rdx and the flags are free there): for an
/// allocator on the GPU arena, each MiB the allocation [rax, (rax + r12 + 7)
/// & ~7) touches gets its high-water mark set to the allocation's end in
/// it, and its generation advanced when the allocation starts below the
/// mark (the MiB was reused). Plain stores: a MiB belongs to one chunk, one
/// thread slot (the shared slot is locked).
pub fn alloc_insert_code(_counter: u64) -> Vec<u8> {
    use iced_x86::code_asm::*;
    let table = ALLOC.load(Ordering::Relaxed) as u64;
    if table == 0 {
        crate::warn!("engine map: no allocation table; the arena's allocation code left out");
        return Vec::new();
    }
    let arena_cell = &GPU_ARENA as *const AtomicU64 as u64;
    let build = || -> Result<Vec<u8>, IcedError> {
        let mut a = CodeAssembler::new(64)?;
        let mut lp = a.create_label();
        let mut out = a.create_label();
        let mut same = a.create_label();
        a.lea(rsp, ptr(rsp - 0x80))?; // System V red zone
        for r in [rsi, rdi, r8, r9, r10, r11] {
            a.push(r)?;
        }
        a.mov(rcx, qword_ptr(r15 + 0x88))?;
        a.mov(rdx, arena_cell)?;
        a.cmp(rcx, qword_ptr(rdx))?;
        a.jne(out)?;
        a.lea(rsi, ptr(rax + r12 + 7))?;
        a.and(rsi, -8)?;
        a.mov(rdi, rax)?;
        a.mov(r8, table)?;
        a.set_label(&mut lp)?;
        a.cmp(rdi, rsi)?;
        a.jae(out)?;
        a.mov(rcx, rdi)?;
        a.shr(rcx, 20)?;
        a.cmp(rcx, MIB_ENTRIES as i32)?;
        a.jae(out)?;
        a.lea(r9, ptr(rcx + 1))?;
        a.shl(r9, 20)?;
        a.cmp(r9, rsi)?;
        a.cmova(r9, rsi)?;
        a.mov(r10, rdi)?;
        a.and(r10, 0xFFFFF)?;
        a.mov(rdx, qword_ptr(r8 + rcx * 8))?;
        a.mov(r11, rdx)?;
        a.and(r11, 0x1FFFFF)?;
        a.shr(rdx, 32)?;
        a.cmp(r10, r11)?;
        a.jae(same)?;
        a.inc(edx)?;
        a.set_label(&mut same)?;
        a.shl(rdx, 32)?;
        a.mov(r11, rcx)?;
        a.shl(r11, 20)?;
        a.mov(r10, r9)?;
        a.sub(r10, r11)?;
        a.or(rdx, r10)?;
        a.mov(qword_ptr(r8 + rcx * 8), rdx)?;
        a.mov(rdi, r9)?;
        a.jmp(lp)?;
        a.set_label(&mut out)?;
        for r in [r11, r10, r9, r8, rdi, rsi] {
            a.pop(r)?;
        }
        a.lea(rsp, ptr(rsp + 0x80))?;
        a.assemble(0)
    };
    match build() {
        Ok(v) => v,
        Err(e) => {
            crate::warn!("engine map: the arena's allocation code: {e}");
            Vec::new()
        }
    }
}

/// The GPU arena's tag being freed, its lock held (a title-fix hook at
/// eboot+0x7fe5d4: r8 the arena, r12 the tag): every MiB of its units gets
/// a new generation and no allocations.
pub extern "sysv64" fn arena_free_hook(regs: *const u64, _arg: u32) {
    let (arena, tag) = unsafe { (*regs.add(8), *regs.add(12)) };
    if arena == 0 || arena != GPU_ARENA.load(Ordering::Relaxed) {
        return;
    }
    let t = ALLOC.load(Ordering::Relaxed);
    let (Some(unit), Some(mask)) = (rd64(arena + 0x88), rd64(arena + 0x728)) else { return };
    if t.is_null() || !unit.is_power_of_two() || !(1 << 20..=1 << 26).contains(&unit) {
        return;
    }
    let Some(i) = (0..64u64).find(|&i| mask >> i & 1 != 0 && rd64(arena + 0x528 + 8 * i) == Some(tag)) else { return };
    ARENA_FREES[0].fetch_add(1, Ordering::Relaxed);
    let head = arena + 0x518;
    let mut g = rd64(head).unwrap_or(0);
    let mut gn = 0;
    let mut mibs = 0u64;
    while g != head && g != 0 && gn < 64 {
        gn += 1;
        let used = g + 0x10;
        let mut d = rd64(used).unwrap_or(0);
        let mut dn = 0;
        while d != used && d != 0 && dn < 64 {
            dn += 1;
            let (Some(base), Some(b0), Some(b1)) = (rd64(d + 0x10), rd64(d + 0x28 + 16 * i), rd64(d + 0x30 + 16 * i)) else { break };
            for u in 0..120u64 {
                if (if u < 64 { b0 >> u & 1 } else { b1 >> (u - 64) & 1 }) == 0 {
                    continue;
                }
                let s = base + u * unit;
                for m in (s >> 20)..((s + unit) >> 20) {
                    if m < MIB_ENTRIES {
                        let p = unsafe { t.add(m as usize) };
                        let e = unsafe { std::ptr::read_volatile(p) };
                        unsafe { std::ptr::write_volatile(p, ((e >> 32) + 1) << 32) };
                        mibs += 1;
                    }
                }
            }
            d = rd64(d).unwrap_or(0);
        }
        g = rd64(g).unwrap_or(0);
    }
    ARENA_FREES[1].fetch_add(mibs, Ordering::Relaxed);
}

/// The GPU arena's allocator ID (the byte at its object's +0x30; 0 until
/// the arena is known).
pub fn arena_owner() -> u8 {
    let a = GPU_ARENA.load(Ordering::Relaxed);
    if a == 0 || !crate::mem::readable(a + 0x30, 1) {
        return 0;
    }
    unsafe { std::ptr::read_volatile((a + 0x30) as *const u8) }
}

/// Whether `a` lies in the GPU arena's frame allocations of the current
/// generation (below its MiB's high-water mark in the allocation table).
pub fn arena_allocated(a: u64) -> bool {
    (a & 0xF_FFFF) < alloc_entry(a >> 20) & 0x1F_FFFF
}

/// The GPU arena's address range from the engine map (MiB-aligned lo, hi)
/// and when it was read.
static ARENA_SPAN: parking_lot::Mutex<(Option<std::time::Instant>, u64, u64)> = parking_lot::Mutex::new((None, 0, 0));
/// Front end 3's coherence scoped by the allocation table, priced
/// (`arena_scan_census`, at title submissions with the census on): [scans,
/// runs, queries, bytes scanned, pages found written, ns].
static SCAN: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

/// At a title submission (census): the GPU arena's live frame allocations
/// (each MiB's allocated bytes in the current generation, merged into runs)
/// queried for pages written since their last take, without a reset, timed:
/// what a coherence per submission confined to them would scan.
fn arena_scan_census() {
    let arena = arena_owner();
    if arena == 0 || ALLOC.load(Ordering::Relaxed).is_null() {
        return;
    }
    let (lo, hi) = {
        let mut g = ARENA_SPAN.lock();
        if g.0.is_none_or(|t| t.elapsed().as_secs() >= 2) {
            let (l, h) = census().into_iter().find(|x| x.0 == arena).map_or((0, 0), |x| (x.2, x.3));
            *g = (Some(std::time::Instant::now()), l, h);
        }
        (g.1, g.2)
    };
    if hi <= lo {
        return;
    }
    let t = std::time::Instant::now();
    let mut runs: Vec<(u64, u64)> = Vec::new();
    let mut cur: Option<(u64, u64)> = None;
    for m in (lo >> 20)..(hi >> 20) {
        let h = alloc_entry(m) & 0x1F_FFFF;
        if h == 0 || owner(m << 20) != arena {
            if let Some(r) = cur.take() {
                runs.push(r);
            }
            continue;
        }
        let (s, e) = (m << 20, (m << 20) + h);
        cur = match cur {
            Some((a, b)) if b == s => Some((a, e)),
            other => {
                if let Some(r) = other {
                    runs.push(r);
                }
                Some((s, e))
            }
        };
        if h < 1 << 20 {
            runs.extend(cur.take());
        }
    }
    runs.extend(cur.take());
    let (mut pages, mut calls, mut bytes) = (0u64, 0u64, 0u64);
    for &(a, b) in &runs {
        if let Some((p, c)) = crate::mem::query_written(a, b) {
            pages += p;
            calls += c;
        }
        bytes += b - a;
    }
    for (i, v) in [1, runs.len() as u64, calls, bytes, pages, t.elapsed().as_nanos() as u64].into_iter().enumerate() {
        SCAN[i].fetch_add(v, Ordering::Relaxed);
    }
}

/// The scan census's line.
fn scan_line() -> String {
    let s: Vec<u64> = SCAN.iter().map(|x| x.load(Ordering::Relaxed)).collect();
    if s[0] == 0 {
        return String::new();
    }
    let n = s[0] as f64;
    format!(
        "    engine census: the GPU arena's live frame allocations queried at title submissions (no reset): {} scans, each {:.1} runs, {:.1} queries, {:.2} MiB, {:.0} pages found written, {:.1} us\n",
        s[0],
        s[1] as f64 / n,
        s[2] as f64 / n,
        s[3] as f64 / n / 1048576.0,
        s[4] as f64 / n,
        s[5] as f64 / n / 1000.0
    )
}

/// The page allocator's records by address (both pools): (start, end,
/// record), sorted, and when they were read (`record_extent`).
static RECS: parking_lot::RwLock<(Option<std::time::Instant>, Vec<(u64, u64, u64)>)> = parking_lot::RwLock::new((None, Vec::new()));
/// Record lookups: [found and still linked, rebuilds, misses].
static REC_STATS: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// Reads the records anew (the pools' lists, as `allocators`).
fn recs_rebuild() {
    let Some(t) = tracking() else { return };
    let mut v = Vec::new();
    for head in [t, t + 0xa0] {
        let Some(mut n) = rd64(head + 0x10) else { continue };
        let mut steps = 0;
        while n != head && n != 0 && steps < 50_000 {
            steps += 1;
            let rec = n - 0x20;
            let (Some(size), Some(addr)) = (rd64(rec + 0x58), rd64(rec + 0x60)) else { break };
            if size != 0 {
                v.push((addr, addr.saturating_add(size), rec));
            }
            let Some(next) = rd64(n + 0x10) else { break };
            n = next;
        }
    }
    v.sort_unstable();
    REC_STATS[1].fetch_add(1, Ordering::Relaxed);
    *RECS.write() = (Some(std::time::Instant::now()), v);
}

/// Whether record `rec` still describes [start, end) and is linked in its
/// pool's list (a freed record is unlinked: its neighbours no longer point
/// at it).
fn rec_live(rec: u64, start: u64, end: u64) -> bool {
    let node = rec + 0x20;
    rd64(rec + 0x60) == Some(start)
        && rd64(rec + 0x58) == Some(end - start)
        && rd64(rec + 0x30).and_then(|next| rd64(next + 0x8)) == Some(node)
        && rd64(rec + 0x28).and_then(|prev| rd64(prev + 0x10)) == Some(node)
}

/// The end of the page allocator's allocation containing `base` (a record of
/// the engine's own tracking, checked live: its address, size and links);
/// None outside every record. The records are read again when the one found
/// no longer holds, or when an address an allocator owns (the MiB table) has
/// none (at most every 50 ms).
pub fn record_extent(base: u64) -> Option<u64> {
    for attempt in 0..2 {
        let found = {
            let g = RECS.read();
            let i = g.1.partition_point(|r| r.0 <= base);
            (i > 0).then(|| g.1[i - 1]).filter(|r| base < r.1)
        };
        match found {
            Some((s, e, rec)) if rec_live(rec, s, e) => {
                REC_STATS[0].fetch_add(1, Ordering::Relaxed);
                return Some(e);
            }
            _ if attempt == 0 && (found.is_some() || owner(base) != 0) && RECS.read().0.is_none_or(|t| t.elapsed().as_millis() >= 50) => recs_rebuild(),
            _ => break,
        }
    }
    REC_STATS[2].fetch_add(1, Ordering::Relaxed);
    None
}

/// The tag-free hook's counts: the GPU arena's tag frees and their MiBs.
static ARENA_FREES: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// Unbounded bindings (base pointers whose loads the translation cannot
/// bound) bounded by the page allocator's records (`Exec`'s stage bindings,
/// `tune unb_bound`): [bounded, their bytes before, after, left as they
/// were].
pub const UNB_REC_BOUNDED: usize = 0;
pub const UNB_REC_BEFORE: usize = 1;
pub const UNB_REC_AFTER: usize = 2;
pub const UNB_LEFT: usize = 3;
pub static UNB_STATS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

#[inline]
pub fn unb_count(i: usize, n: u64) {
    UNB_STATS[i].fetch_add(n, Ordering::Relaxed);
}

/// The flip the counters below count from (`reset`).
static STATS_FLIP0: AtomicU64 = AtomicU64::new(0);

/// `engine reset` (control): the census and the counters start again (an
/// in-run A/B's periods each get their own).
pub fn reset() {
    for row in CENSUS.iter() {
        for x in row.iter() {
            x.store(0, Ordering::Relaxed);
        }
    }
    for x in UNB_STATS.iter().chain(ARENA_FREES.iter()).chain(REC_STATS.iter()).chain(SCAN.iter()) {
        x.store(0, Ordering::Relaxed);
    }
    *BY_TAG.lock() = None;
    *UP_BY_REC.lock() = None;
    *UP_BY_PROG.lock() = None;
    alloc_census_reset();
    let f = crate::gpu::FRAMES.load(Ordering::Relaxed);
    CENSUS_FLIP0.store(if census_on() { f + 1 } else { 0 }, Ordering::Relaxed);
    STATS_FLIP0.store(f, Ordering::Relaxed);
}

/// `engine report` (control): the census and the bounds' line.
pub fn report() -> String {
    census_line() + &bound_line() + &scan_line() + &crate::gpu::fe2::hot_by_owner(2) + &crate::gpu::fe2::hot_by_owner(16) + &alloc_census_lines()
}

/// The allocation table's and the bounds' status line (per flip).
pub fn bound_line() -> String {
    let s: Vec<u64> = UNB_STATS.iter().chain(ARENA_FREES.iter()).map(|x| x.load(Ordering::Relaxed)).collect();
    if s.iter().all(|&x| x == 0) {
        return String::new();
    }
    let flips = crate::gpu::FRAMES.load(Ordering::Relaxed).saturating_sub(STATS_FLIP0.load(Ordering::Relaxed)).max(1) as f64;
    let nr = s[UNB_REC_BOUNDED].max(1) as f64;
    let r: Vec<u64> = REC_STATS.iter().map(|x| x.load(Ordering::Relaxed)).collect();
    format!(
        "    engine allocations: unbounded bindings a flip bounded by the page allocator's records {:.1} (on average {:.1} MiB -> {:.2} MiB), left {:.1}; record lookups {} found, {} rebuilds, {} misses; the arena's tag frees {:.1} a flip ({:.1} MiB)\n",
        s[UNB_REC_BOUNDED] as f64 / flips,
        s[UNB_REC_BEFORE] as f64 / nr / 1048576.0,
        s[UNB_REC_AFTER] as f64 / nr / 1048576.0,
        s[UNB_LEFT] as f64 / flips,
        r[0],
        r[1],
        r[2],
        s[4] as f64 / flips,
        s[5] as f64 / flips,
    )
}

// ---------------------------------------------------------------------------
// Census of the GPU arena's frame allocations by the code that makes them
// (`DS_ALLOC_CENSUS=1`, a title-fix hook at eboot+0x8121b7, in the frame
// allocator's placement of an allocation): which call sites allocate how
// much, and how many of their pages the CPU writes (front end 2's found
// writes). Whether the arena's CPU-written allocations can be told from its
// GPU-only scratch by their call site (FRONTEND3.md: a per-epoch scan
// confined to the CPU-written ones).
// ---------------------------------------------------------------------------

pub fn alloc_census_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_ALLOC_CENSUS").ok().as_deref() == Some("1"))
}

/// One allocation: [start, end), its call site (index into `SITES`), the
/// flip it was made in.
#[derive(Clone, Copy)]
struct AllocRec {
    start: u64,
    end: u64,
    site: u32,
    flip: u32,
}

/// The last allocations (`ALLOC_LOG_CAP`), call sites interned by their
/// two return addresses, and per site [allocations, bytes] since the last
/// `reset`.
struct AllocCensus {
    log: std::collections::VecDeque<AllocRec>,
    ids: std::collections::HashMap<[u64; 5], u32>,
    sites: Vec<[u64; 5]>,
    totals: Vec<[u64; 2]>,
}
const ALLOC_LOG_CAP: usize = 1 << 19;
static ALLOC_CENSUS: parking_lot::Mutex<Option<AllocCensus>> = parking_lot::Mutex::new(None);

/// The hook after the frame allocator placed an allocation (rax its start,
/// r12 its size, r15 the allocator, rbp its frame: [rbp+8] the return into
/// its caller, [[rbp]+8] the caller's).
pub extern "sysv64" fn alloc_census_hook(regs: *const u64, _arg: u32) {
    let r = |i: usize| unsafe { *regs.add(i) };
    let (start, size, this, rbp) = (r(0), r(12), r(15), r(5));
    let arena = GPU_ARENA.load(Ordering::Relaxed);
    if arena == 0 || size == 0 || rd64(this + 0x88) != Some(arena) {
        return;
    }
    // (Four return addresses up the frame chain: the engine reaches the
    // frame allocator through a generic wrapper and a proxy allocator.)
    // ([4]: the frame allocator's ID, the byte at its +0x30.)
    let mut key = [0u64; 5];
    key[4] = rd64(this + 0x30).map_or(0, |v| v & 0xff);
    let mut f = rbp;
    for k in key[..4].iter_mut() {
        *k = rd64(f + 8).unwrap_or(0);
        f = match rd64(f) {
            Some(n) if n > f => n,
            _ => break,
        };
    }
    let flip = crate::gpu::FRAMES.load(Ordering::Relaxed) as u32;
    let mut g = ALLOC_CENSUS.lock();
    let c = g.get_or_insert_with(|| AllocCensus { log: std::collections::VecDeque::new(), ids: Default::default(), sites: Vec::new(), totals: Vec::new() });
    let id = match c.ids.get(&key) {
        Some(&id) => id,
        None => {
            let id = c.sites.len() as u32;
            c.ids.insert(key, id);
            c.sites.push(key);
            c.totals.push([0; 2]);
            id
        }
    };
    c.totals[id as usize][0] += 1;
    c.totals[id as usize][1] += size;
    if c.log.len() >= ALLOC_LOG_CAP {
        c.log.pop_front();
    }
    c.log.push_back(AllocRec { start, end: start + size, site: id, flip });
}

/// An executable address as eboot+offset.
fn eboot_off(a: u64) -> String {
    let base = BASE.load(Ordering::Relaxed);
    if a >= base && a < base + 0x4000_0000 {
        format!("eboot+{:#x}", a - base)
    } else {
        format!("{a:#x}")
    }
}

/// The census's lines (`engine report`): call sites by bytes allocated since
/// the last reset, and the arena's pages with a CPU write found within the
/// last 2 flips attributed to the latest allocation (of the last 8 flips)
/// holding them.
fn alloc_census_lines() -> String {
    if !alloc_census_on() {
        return String::new();
    }
    let arena = arena_owner();
    let hot = crate::gpu::fe2::hot_pages_of(arena, 2);
    let g = ALLOC_CENSUS.lock();
    let Some(c) = g.as_ref() else { return String::new() };
    let flips = crate::gpu::FRAMES.load(Ordering::Relaxed).saturating_sub(STATS_FLIP0.load(Ordering::Relaxed)).max(1) as f64;
    let now = crate::gpu::FRAMES.load(Ordering::Relaxed) as u32;
    // page -> site of the latest allocation over it (oldest first, so later
    // ones overwrite).
    let mut by_page: std::collections::HashMap<u64, u32> = std::collections::HashMap::new();
    for a in c.log.iter().filter(|a| now.wrapping_sub(a.flip) <= 8) {
        let mut p = a.start & !4095;
        while p < a.end {
            by_page.insert(p, a.site);
            p += 4096;
        }
    }
    let mut written = vec![0u64; c.sites.len()];
    let mut unattributed = 0u64;
    for p in &hot {
        match by_page.get(p) {
            Some(&s) => written[s as usize] += 1,
            None => unattributed += 1,
        }
    }
    let mut order: Vec<usize> = (0..c.sites.len()).collect();
    order.sort_by(|&a, &b| c.totals[b][1].cmp(&c.totals[a][1]));
    let site = |i: usize| format!("[frame allocator {:#x}] {}", c.sites[i][4], c.sites[i][..4].iter().filter(|&&a| a != 0).map(|&a| eboot_off(a)).collect::<Vec<_>>().join(" <- "));
    let total_bytes: u64 = c.totals.iter().map(|t| t[1]).sum();
    let by_bytes: Vec<String> = order
        .iter()
        .take(24)
        .map(|&i| format!("{}: {:.0} allocations, {:.2} MiB a flip, {} pages written", site(i), c.totals[i][0] as f64 / flips, c.totals[i][1] as f64 / flips / 1048576.0, written[i]))
        .collect();
    let mut wo: Vec<usize> = (0..c.sites.len()).filter(|&i| written[i] > 0).collect();
    wo.sort_by(|&a, &b| written[b].cmp(&written[a]));
    let by_written: Vec<String> = wo.iter().take(16).map(|&i| format!("{}: {} pages ({:.2} MiB a flip allocated)", site(i), written[i], c.totals[i][1] as f64 / flips / 1048576.0)).collect();
    let writers = wo.len();
    let writer_bytes: u64 = wo.iter().map(|&i| c.totals[i][1]).sum();
    format!(
        "    engine allocation census: {} call sites, {:.2} MiB a flip allocated in the GPU arena; {} pages with a CPU write within 2 flips ({} not in an allocation of the last 8 flips); the {} sites with written pages allocate {:.2} MiB a flip\n      by bytes: {}\n      by written pages: {}\n",
        c.sites.len(),
        total_bytes as f64 / flips / 1048576.0,
        hot.len(),
        unattributed,
        writers,
        writer_bytes as f64 / flips / 1048576.0,
        by_bytes.join("; "),
        by_written.join("; ")
    )
}

/// `engine reset`: the census's totals start again (the log stays).
fn alloc_census_reset() {
    if let Some(c) = ALLOC_CENSUS.lock().as_mut() {
        for t in c.totals.iter_mut() {
            *t = [0; 2];
        }
    }
}
