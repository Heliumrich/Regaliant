//! Section-backed guest memory (`DS_SECTIONS=1`, boot; `mem` uses it).
//!
//! Direct memory is one pagefile section (offset = physical address) and the
//! other guest memory (flexible, modules, runtime-owned) a second one with an
//! offset allocator; every guest mapping is a view of its section, mapped into
//! a placeholder of the reserved guest range. So, as on the console, physical
//! pages keep their contents while unmapped and show them wherever they are
//! mapped next, and two mappings of the same pages share them. Each guest
//! mapping also gets an alias view at `va + alias_delta(slot)`: GPU imports
//! are made of the alias (never protected), so the guest view may be
//! protected for CPU-write tracking (protecting imported pages lost the
//! device on the RTX 4090). Alias slots: an alias view still referenced or
//! still imported keeps its range; a later mapping at the same guest address
//! takes the next free slot. Views and placeholders work at 16 KiB
//! granularity, and one import may span adjacent views (boot test,
//! `gpu::resident::section_test`).

use parking_lot::Mutex;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use windows_sys::Win32::Foundation::{GetLastError, HANDLE, INVALID_HANDLE_VALUE};
use windows_sys::Win32::System::Memory::{
    CreateFileMappingW, MapViewOfFile3, UnmapViewOfFile2, VirtualAlloc, VirtualAlloc2, VirtualFree, VirtualProtect, VirtualQuery, MEMORY_BASIC_INFORMATION, MEMORY_MAPPED_VIEW_ADDRESS,
    MEM_COMMIT, MEM_PRESERVE_PLACEHOLDER, MEM_RELEASE, MEM_REPLACE_PLACEHOLDER, MEM_RESERVE, MEM_RESERVE_PLACEHOLDER, PAGE_EXECUTE_READWRITE, PAGE_NOACCESS,
    PAGE_READWRITE, SEC_RESERVE,
};
use windows_sys::Win32::System::Threading::GetCurrentProcess;

/// VirtualFree's MEM_COALESCE_PLACEHOLDERS (not named by windows-sys).
const MEM_COALESCE_PLACEHOLDERS: u32 = 0x1;

/// Alias slots and their distance from the guest addresses they mirror.
pub const ALIAS_SLOTS: usize = 3;
const ALIAS_BASE_DELTA: u64 = 0x800_0000_0000;

/// The address an alias of guest address `va` has in `slot`.
pub fn alias_of(va: u64, slot: u8) -> u64 {
    va + ALIAS_BASE_DELTA * (slot as u64 + 1)
}

/// Which section a view maps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sec {
    /// Direct memory (offset = physical address).
    Dmem,
    /// Everything else (offsets from `anon_alloc`).
    Anon,
}

struct State {
    dmem: HANDLE,
    anon: HANDLE,
    anon_size: u64,
    /// Free anon offsets: start -> size (made on first use: `reserve` may
    /// not allocate).
    anon_free: BTreeMap<u64, u64>,
    anon_init: bool,
    /// Alias views by (slot, guest start): (size, section, offset, guest
    /// bytes still mapped through it, imported or deferred: kept).
    aliases: BTreeMap<(u8, u64), Alias>,
    /// Section ranges that may hold data, start -> end: direct memory mapped
    /// since it was last zeroed, anonymous offsets freed after use.
    dirty: [BTreeMap<u64, u64>; 2],
    /// Direct memory allocated again over dirty ranges: zeroed when it is
    /// next mapped (`map_new_view`).
    zero_next: BTreeMap<u64, u64>,
}

/// Adds [a, e) to the range set `m` (start -> end), coalescing.
fn rs_add(m: &mut BTreeMap<u64, u64>, mut a: u64, mut e: u64) {
    if a >= e {
        return;
    }
    if let Some((&s, &t)) = m.range(..=a).next_back() {
        if t >= a {
            m.remove(&s);
            a = s;
            e = e.max(t);
        }
    }
    while let Some((&s, &t)) = m.range(a..=e).next() {
        m.remove(&s);
        e = e.max(t);
    }
    m.insert(a, e);
}

/// Removes [a, e) from the range set `m`; returns the pieces removed.
fn rs_take(m: &mut BTreeMap<u64, u64>, a: u64, e: u64) -> Vec<(u64, u64)> {
    let mut out = Vec::new();
    let first = m.range(..a).next_back().filter(|(_, t)| **t > a).map(|(&s, _)| s).unwrap_or(a);
    let keys: Vec<u64> = m.range(first..e).map(|(&s, _)| s).collect();
    for s in keys {
        let t = m.remove(&s).unwrap();
        if s < a {
            m.insert(s, a);
        }
        if t > e {
            m.insert(e, t);
        }
        let (x, y) = (s.max(a), t.min(e));
        if x < y {
            out.push((x, y));
        }
    }
    out
}

/// Bytes zeroed by `map_new_view` (reused anonymous offsets, direct memory
/// allocated again).
static ZEROED: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// Direct memory [start, start+len) was allocated: what of it was mapped
/// before is zeroed at its next mapping, as private pages were before
/// sections (a mapping over the same pages without a new allocation keeps
/// their contents, as on the console).
pub fn dmem_allocated(start: u64, len: u64) {
    let mut g = STATE.lock();
    let Some(st) = g.as_mut() else { return };
    for (a, e) in rs_take(&mut st.dirty[0], start, start + len) {
        rs_add(&mut st.zero_next, a, e);
    }
}

#[derive(Clone, Copy, Debug)]
struct Alias {
    size: u64,
    refs: u64,
}

unsafe impl Send for State {}

static ON: AtomicBool = AtomicBool::new(false);
static STATE: Mutex<Option<State>> = Mutex::new(None);
/// Alias views no longer referenced but possibly imported: (slot, guest start).
static DEFERRED: Mutex<Vec<(u8, u64)>> = Mutex::new(Vec::new());

/// Section-backed guest memory is on (set by `reserve`).
pub fn on() -> bool {
    ON.load(Ordering::Acquire)
}

fn err<T>(what: &str) -> Result<T, String> {
    Err(format!("{what}: error {}", unsafe { GetLastError() }))
}

/// Creates the sections and reserves the guest range [base, base+size) and
/// its alias ranges as placeholders. Allocates nothing (it may run inside
/// the first heap allocation, `mem::reserve_guest_range_early`); the error is
/// the Win32 error.
pub fn reserve(base: u64, size: u64, dmem_size: u64, anon_size: u64) -> Result<(), u32> {
    unsafe {
        let dmem = CreateFileMappingW(INVALID_HANDLE_VALUE, std::ptr::null(), PAGE_EXECUTE_READWRITE | SEC_RESERVE, (dmem_size >> 32) as u32, dmem_size as u32, std::ptr::null());
        if dmem.is_null() {
            return Err(GetLastError());
        }
        let anon = CreateFileMappingW(INVALID_HANDLE_VALUE, std::ptr::null(), PAGE_EXECUTE_READWRITE | SEC_RESERVE, (anon_size >> 32) as u32, anon_size as u32, std::ptr::null());
        if anon.is_null() {
            return Err(GetLastError());
        }
        for slot in 0..=ALIAS_SLOTS as u64 {
            let at = if slot == 0 { base } else { alias_of(base, (slot - 1) as u8) };
            let p = VirtualAlloc2(std::ptr::null_mut(), at as _, size as usize, MEM_RESERVE | MEM_RESERVE_PLACEHOLDER, PAGE_NOACCESS, std::ptr::null_mut(), 0);
            if p.is_null() {
                return Err(GetLastError());
            }
        }
        *STATE.lock() = Some(State {
            dmem,
            anon,
            anon_size,
            anon_free: BTreeMap::new(),
            anon_init: false,
            aliases: BTreeMap::new(),
            dirty: [BTreeMap::new(), BTreeMap::new()],
            zero_next: BTreeMap::new(),
        });
    }
    ON.store(true, Ordering::Release);
    Ok(())
}

/// Guest range being split right now (`split_begin`): faults inside it wait
/// for the split (`wait_split`) instead of being taken for crashes.
static SPLIT_LO: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static SPLIT_HI: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Splits made, and the last one's range and time (`log::elapsed_ms`):
/// diagnostics of kernel-mode accesses that failed (`mem::kernel_io`).
static SPLITS: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
static LAST_SPLIT: Mutex<(u64, u64, f64)> = Mutex::new((0, 0, 0.0));
/// Causes of splits (`split_begin`): a mapping made over part of a view, a
/// part unmapped, a part's protection changed.
pub const SPLIT_MAP_OVER: usize = 0;
pub const SPLIT_UNMAP: usize = 1;
pub const SPLIT_PROTECT: usize = 2;

/// A view over [lo, hi) is about to be unmapped and mapped again in parts
/// (`why`: SPLIT_*).
pub fn split_begin(lo: u64, hi: u64, why: usize) {
    SPLITS[why.min(2)].fetch_add(1, Ordering::Relaxed);
    *LAST_SPLIT.lock() = (lo, hi, crate::log::elapsed_ms());
    SPLIT_LO.store(lo, Ordering::SeqCst);
    SPLIT_HI.store(hi, Ordering::SeqCst);
}

/// Waits while a split overlapping [lo, hi) is in progress (kernel-mode
/// accesses cannot fault into `wait_split`); whether one was.
pub fn wait_split_range(lo: u64, hi: u64) -> bool {
    if !on() {
        return false;
    }
    let mut waited = false;
    loop {
        let (a, b) = (SPLIT_LO.load(Ordering::SeqCst), SPLIT_HI.load(Ordering::SeqCst));
        if b == 0 || hi <= a || b <= lo {
            return waited;
        }
        waited = true;
        std::thread::yield_now();
    }
}

/// Bytes zeroed for new mappings (diagnostics): direct, anonymous.
pub fn zeroed() -> (u64, u64) {
    (ZEROED[0].load(Ordering::Relaxed), ZEROED[1].load(Ordering::Relaxed))
}

/// Splits so far and the last one (diagnostics).
pub fn split_line() -> String {
    let (lo, hi, at) = *LAST_SPLIT.lock();
    let n: Vec<u64> = SPLITS.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    format!("{} view splits (mappings over a part {}, parts unmapped {}, parts protected {}; last {lo:#x}..{hi:#x} at {at:.0} ms)", n.iter().sum::<u64>(), n[0], n[1], n[2])
}

/// The split is done.
pub fn split_end() {
    SPLIT_HI.store(0, Ordering::SeqCst);
    SPLIT_LO.store(0, Ordering::SeqCst);
}

/// A fault at `addr` inside a range being split: waits for the split to end
/// (the access is then retried). Asked by the exception handler.
pub fn wait_split(addr: u64) -> bool {
    if !on() {
        return false;
    }
    let (lo, hi) = (SPLIT_LO.load(Ordering::SeqCst), SPLIT_HI.load(Ordering::SeqCst));
    if hi == 0 || addr < lo || addr >= hi {
        return false;
    }
    while SPLIT_HI.load(Ordering::SeqCst) != 0 {
        std::hint::spin_loop();
    }
    true
}

/// Allocates `size` bytes of the anonymous section (16 KiB units).
pub fn anon_alloc(size: u64) -> Option<u64> {
    let mut g = STATE.lock();
    let st = g.as_mut()?;
    if !st.anon_init {
        st.anon_init = true;
        st.anon_free.insert(0, st.anon_size);
    }
    let (&off, &n) = st.anon_free.iter().find(|(_, n)| **n >= size)?;
    st.anon_free.remove(&off);
    if n > size {
        st.anon_free.insert(off + size, n - size);
    }
    Some(off)
}

/// Returns [off, off+size) of the anonymous section (coalescing); its pages
/// are zeroed when the offsets are mapped again (`map_new_view`).
pub fn anon_free(off: u64, size: u64) {
    let mut g = STATE.lock();
    let Some(st) = g.as_mut() else { return };
    rs_add(&mut st.dirty[1], off, off + size);
    let (mut s, mut e) = (off, off + size);
    if let Some((&p, &n)) = st.anon_free.range(..s).next_back() {
        if p + n == s {
            st.anon_free.remove(&p);
            s = p;
        }
    }
    if let Some(&n) = st.anon_free.get(&e) {
        st.anon_free.remove(&e);
        e += n;
    }
    st.anon_free.insert(s, e - s);
}

/// The placeholder or view containing `a`: (allocation start, end, is a
/// placeholder: reserved and not a mapped view).
fn region(a: u64) -> Option<(u64, u64, bool)> {
    use windows_sys::Win32::System::Memory::MEM_MAPPED;
    let mut m: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    if unsafe { VirtualQuery(a as _, &mut m, std::mem::size_of::<MEMORY_BASIC_INFORMATION>()) } == 0 {
        return None;
    }
    let start = m.AllocationBase as u64;
    // RegionSize runs from BaseAddress while the attributes stay the same:
    // for a placeholder, to its end.
    let end = m.BaseAddress as u64 + m.RegionSize as u64;
    Some((start, end, m.State == MEM_RESERVE && m.Type != MEM_MAPPED))
}

/// Makes [a, a+n) exactly one placeholder, splitting the placeholders at its
/// ends and coalescing the ones inside. Fails when part of it is mapped.
pub fn isolate(a: u64, n: u64) -> Result<(), String> {
    let end = a + n;
    unsafe {
        // Every region inside must be a placeholder.
        let mut p = a;
        while p < end {
            let Some((_, e, ph)) = region(p) else { return Err(format!("{p:#x}: no region")) };
            if !ph {
                return Err(format!("{p:#x} is mapped"));
            }
            p = e;
        }
        let (s0, e0, _) = region(a).ok_or("no region at the start")?;
        if s0 < a && VirtualFree(a as _, (e0.min(end) - a) as usize, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER) == 0 {
            return err(&format!("splitting at {a:#x}"));
        }
        let (s1, e1, _) = region(end - 1).ok_or("no region at the end")?;
        if e1 > end {
            let from = s1.max(a);
            if VirtualFree(from as _, (end - from) as usize, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER) == 0 {
                return err(&format!("splitting at {end:#x}"));
            }
        }
        let (_, e2, _) = region(a).ok_or("no region after splitting")?;
        if e2 < end && VirtualFree(a as _, n as usize, MEM_RELEASE | MEM_COALESCE_PLACEHOLDERS) == 0 {
            return err(&format!("coalescing {a:#x}+{n:#x}"));
        }
    }
    Ok(())
}

fn handle(sec: Sec) -> Option<HANDLE> {
    STATE.lock().as_ref().map(|s| match sec {
        Sec::Dmem => s.dmem,
        Sec::Anon => s.anon,
    })
}

/// Maps a view of `sec` at [va, va+n) (offset `off`) into placeholders,
/// commits its pages and gives it protection `prot`.
pub fn map_view(va: u64, n: u64, sec: Sec, off: u64, prot: u32) -> Result<(), String> {
    map_view_with(va, n, sec, off, prot, false)
}

/// `map_view` for a new guest mapping: its pages read as zeros where the
/// section's pages are stale (anonymous offsets used before, direct memory
/// allocated again: fresh flexible memory is zero-filled on the console).
pub fn map_new_view(va: u64, n: u64, sec: Sec, off: u64, prot: u32) -> Result<(), String> {
    map_view_with(va, n, sec, off, prot, true)
}

fn map_view_with(va: u64, n: u64, sec: Sec, off: u64, prot: u32, fresh: bool) -> Result<(), String> {
    let h = handle(sec).ok_or("sections not set up")?;
    isolate(va, n)?;
    unsafe {
        let v = MapViewOfFile3(h, GetCurrentProcess(), va as _, off, n as usize, MEM_REPLACE_PLACEHOLDER, PAGE_EXECUTE_READWRITE, std::ptr::null_mut(), 0);
        if v.Value.is_null() {
            return err(&format!("mapping a view at {va:#x}+{n:#x} (offset {off:#x})"));
        }
        if VirtualAlloc(va as _, n as usize, MEM_COMMIT, PAGE_EXECUTE_READWRITE).is_null() {
            let e = GetLastError();
            UnmapViewOfFile2(GetCurrentProcess(), v, MEM_PRESERVE_PLACEHOLDER);
            return Err(format!("committing {va:#x}+{n:#x}: error {e}"));
        }
        if fresh {
            let stale = match STATE.lock().as_mut() {
                Some(st) if sec == Sec::Dmem => {
                    let z = rs_take(&mut st.zero_next, off, off + n);
                    rs_add(&mut st.dirty[0], off, off + n);
                    z
                }
                Some(st) => rs_take(&mut st.dirty[1], off, off + n),
                None => Vec::new(),
            };
            for (a, e) in stale {
                std::ptr::write_bytes((va + (a - off)) as *mut u8, 0, (e - a) as usize);
                ZEROED[(sec == Sec::Anon) as usize].fetch_add(e - a, Ordering::Relaxed);
            }
        }
        let mut old = 0u32;
        if prot != PAGE_EXECUTE_READWRITE && VirtualProtect(va as _, n as usize, prot, &mut old) == 0 {
            return err(&format!("protecting the view at {va:#x}"));
        }
    }
    Ok(())
}

/// Unmaps the view starting at `va` (its range becomes a placeholder).
pub fn unmap_view(va: u64) -> Result<(), String> {
    unsafe {
        if UnmapViewOfFile2(GetCurrentProcess(), MEMORY_MAPPED_VIEW_ADDRESS { Value: va as _ }, MEM_PRESERVE_PLACEHOLDER) == 0 {
            return err(&format!("unmapping the view at {va:#x}"));
        }
    }
    Ok(())
}

/// Maps the alias of the guest mapping [va, va+n) (`sec` at `off`) in the
/// first slot whose range is free, preferring a slot whose alias views
/// adjoin it (one import may then span both: a binding over two mappings
/// whose aliases lie in different slots cannot be imported). Returns the
/// slot.
pub fn map_alias(va: u64, n: u64, sec: Sec, off: u64) -> Result<u8, String> {
    let adjoins = |slot: u8| {
        STATE.lock().as_ref().is_some_and(|s| {
            s.aliases.range(..(slot, va)).next_back().is_some_and(|(&(sl, st), a)| sl == slot && st + a.size == va) || s.aliases.contains_key(&(slot, va + n))
        })
    };
    let (mut order, rest): (Vec<u8>, Vec<u8>) = (0..ALIAS_SLOTS as u8).partition(|&sl| adjoins(sl));
    order.extend(rest);
    for slot in order {
        let busy = STATE.lock().as_ref().is_some_and(|s| {
            s.aliases.range(..(slot, va + n)).next_back().is_some_and(|(&(sl, st), a)| sl == slot && st + a.size > va)
        });
        if busy {
            continue;
        }
        let at = alias_of(va, slot);
        if map_view(at, n, sec, off, PAGE_READWRITE).is_err() {
            continue;
        }
        if let Some(s) = STATE.lock().as_mut() {
            s.aliases.insert((slot, va), Alias { size: n, refs: n });
        }
        return Ok(slot);
    }
    Err(format!("no free alias slot for {va:#x}+{n:#x}"))
}

/// The guest bytes [va, va+n) no longer map through their alias in `slot`:
/// once an alias view has none left it is unmapped, unless `imported(alias
/// guest range)` says a GPU import may still use it (`retry_deferred` later).
pub fn alias_unref(va: u64, n: u64, slot: u8, imported: impl Fn(u64, u64) -> bool) {
    let done = {
        let mut g = STATE.lock();
        let Some(s) = g.as_mut() else { return };
        let Some((&key, a)) = s.aliases.range_mut(..=(slot, va)).next_back().filter(|((sl, st), a)| *sl == slot && *st + a.size > va) else { return };
        a.refs = a.refs.saturating_sub(n);
        (a.refs == 0).then_some((key, a.size))
    };
    if let Some(((sl, st), size)) = done {
        if imported(st, size) {
            DEFERRED.lock().push((sl, st));
        } else {
            drop_alias(sl, st);
        }
    }
}

fn drop_alias(slot: u8, va: u64) {
    let _ = unmap_view(alias_of(va, slot));
    if let Some(s) = STATE.lock().as_mut() {
        s.aliases.remove(&(slot, va));
    }
}

/// Unmaps deferred alias views no import covers any more.
pub fn retry_deferred(imported: impl Fn(u64, u64) -> bool) {
    let list = std::mem::take(&mut *DEFERRED.lock());
    let mut keep = Vec::new();
    for (slot, va) in list {
        let size = STATE.lock().as_ref().and_then(|s| s.aliases.get(&(slot, va)).map(|a| a.size)).unwrap_or(0);
        if size > 0 && imported(va, size) {
            keep.push((slot, va));
        } else {
            drop_alias(slot, va);
        }
    }
    DEFERRED.lock().extend(keep);
}

/// The alias slot whose adjacent alias views cover guest [a, a+n) without
/// a gap (one import may span them), if any.
pub fn alias_slot_for(a: u64, n: u64) -> Option<u8> {
    let g = STATE.lock();
    let s = g.as_ref()?;
    (0..ALIAS_SLOTS as u8).find(|&slot| {
        let mut p = a;
        while p < a + n {
            match s.aliases.range(..=(slot, p)).next_back() {
                Some((&(sl, st), al)) if sl == slot && st <= p && p < st + al.size => p = st + al.size,
                _ => return false,
            }
        }
        true
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Views, splits, aliases and contents that follow the section pages,
    /// in a private test range.
    #[test]
    fn views_follow_section_pages() {
        let base = 0x6000_0000_0000u64;
        reserve(base, 1 << 30, 64 << 20, 64 << 20).expect("reserving the test range");
        let k = 16u64 << 10;
        // A view at offset 0, written; unmapped; mapped elsewhere: same data.
        map_view(base, 4 * k, Sec::Dmem, 0, PAGE_READWRITE).unwrap();
        unsafe { std::ptr::write_volatile((base + k) as *mut u32, 0xDEAD_BEEF) };
        unmap_view(base).unwrap();
        map_view(base + (1 << 20), 4 * k, Sec::Dmem, 0, PAGE_READWRITE).unwrap();
        assert_eq!(unsafe { std::ptr::read_volatile((base + (1 << 20) + k) as *const u32) }, 0xDEAD_BEEF);
        // Its alias shows the same pages; writes through either show in both.
        let slot = map_alias(base + (1 << 20), 4 * k, Sec::Dmem, 0).unwrap();
        let al = alias_of(base + (1 << 20), slot);
        assert_eq!(unsafe { std::ptr::read_volatile((al + k) as *const u32) }, 0xDEAD_BEEF);
        unsafe { std::ptr::write_volatile((al + 2 * k) as *mut u32, 7) };
        assert_eq!(unsafe { std::ptr::read_volatile((base + (1 << 20) + 2 * k) as *const u32) }, 7);
        // Splitting a view: unmap it, map two views over the same offsets.
        unmap_view(base + (1 << 20)).unwrap();
        map_view(base + (1 << 20), k, Sec::Dmem, 0, PAGE_READWRITE).unwrap();
        map_view(base + (1 << 20) + k, 3 * k, Sec::Dmem, k, PAGE_READWRITE).unwrap();
        assert_eq!(unsafe { std::ptr::read_volatile((base + (1 << 20) + k) as *const u32) }, 0xDEAD_BEEF);
        // A new mapping where an alias is still referenced takes another slot.
        let slot2 = map_alias(base + (1 << 20), k, Sec::Dmem, 0).unwrap();
        assert_ne!(slot, slot2);
        assert_eq!(alias_slot_for(base + (1 << 20), 4 * k), Some(slot));
        assert_eq!(alias_slot_for(base + (1 << 20), k), Some(slot.min(slot2)));
        assert_eq!(alias_slot_for(base + (1 << 20), 5 * k), None);
        // Unmapped placeholders coalesce for a larger view.
        unmap_view(base + (1 << 20)).unwrap();
        unmap_view(base + (1 << 20) + k).unwrap();
        map_view(base + (1 << 20), 4 * k, Sec::Anon, 0, PAGE_READWRITE).unwrap();
        unmap_view(base + (1 << 20)).unwrap();
        // The anonymous allocator coalesces.
        let a = anon_alloc(k).unwrap();
        let b = anon_alloc(2 * k).unwrap();
        anon_free(a, k);
        anon_free(b, 2 * k);
        assert_eq!(anon_alloc(64 << 20), Some(0));
        // A new view of anonymous offsets freed after use reads zeros.
        let at = base + (2 << 20);
        map_new_view(at, 4 * k, Sec::Anon, 0, PAGE_READWRITE).unwrap();
        unsafe { std::ptr::write_volatile((at + k) as *mut u32, 5) };
        unmap_view(at).unwrap();
        anon_free(0, 64 << 20);
        assert_eq!(anon_alloc(4 * k), Some(0));
        map_new_view(at, 4 * k, Sec::Anon, 0, PAGE_READWRITE).unwrap();
        assert_eq!(unsafe { std::ptr::read_volatile((at + k) as *const u32) }, 0);
        unmap_view(at).unwrap();
        // Direct memory keeps its contents across mappings until it is
        // allocated again; then its next mapping reads zeros.
        let at = base + (3 << 20);
        map_new_view(at, 4 * k, Sec::Dmem, 8 * k, PAGE_READWRITE).unwrap();
        unsafe { std::ptr::write_volatile(at as *mut u32, 9) };
        unmap_view(at).unwrap();
        map_new_view(at, 4 * k, Sec::Dmem, 8 * k, PAGE_READWRITE).unwrap();
        assert_eq!(unsafe { std::ptr::read_volatile(at as *const u32) }, 9);
        unmap_view(at).unwrap();
        dmem_allocated(8 * k, 4 * k);
        map_new_view(at, 4 * k, Sec::Dmem, 8 * k, PAGE_READWRITE).unwrap();
        assert_eq!(unsafe { std::ptr::read_volatile(at as *const u32) }, 0);
        unmap_view(at).unwrap();
    }

    #[test]
    fn range_sets() {
        let mut m = BTreeMap::new();
        rs_add(&mut m, 10, 20);
        rs_add(&mut m, 30, 40);
        rs_add(&mut m, 20, 25);
        assert_eq!(m.iter().map(|(&a, &b)| (a, b)).collect::<Vec<_>>(), vec![(10, 25), (30, 40)]);
        rs_add(&mut m, 5, 35);
        assert_eq!(m.iter().map(|(&a, &b)| (a, b)).collect::<Vec<_>>(), vec![(5, 40)]);
        assert_eq!(rs_take(&mut m, 10, 20), vec![(10, 20)]);
        assert_eq!(m.iter().map(|(&a, &b)| (a, b)).collect::<Vec<_>>(), vec![(5, 10), (20, 40)]);
        assert_eq!(rs_take(&mut m, 0, 30), vec![(5, 10), (20, 30)]);
        assert_eq!(m.iter().map(|(&a, &b)| (a, b)).collect::<Vec<_>>(), vec![(30, 40)]);
        assert_eq!(rs_take(&mut m, 50, 60), vec![]);
        assert_eq!(rs_take(&mut m, 35, 45), vec![(35, 40)]);
        assert_eq!(m.iter().map(|(&a, &b)| (a, b)).collect::<Vec<_>>(), vec![(30, 35)]);
    }
}
