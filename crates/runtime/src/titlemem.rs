//! Diagnostics of the title's own memory system (PPSA01341 01.007.000, the
//! BPE engine), read from guest memory: nothing is called or written. The
//! layout comes from disassembly of the executable (2026-10-11):
//! - heap table at eboot+0x2EA0DF0: heap pointers by index (a heap's index
//!   at +0x30, its name pointer at +0x10; allocation entry eboot+0x7F9CF0);
//! - heap vtables: +0x60 used bytes, +0x70 total bytes; free = total - used
//!   (+0x68, eboot+0x801A70) is what the out-of-memory report prints
//!   ("Free: x / y", eboot+0x7F9890);
//! - composite heaps (e.g. "CPUMemory" at eboot+0x2EE2900): sub-heap count at
//!   +0x88 (i32), {size limit, heap} pairs from +0x90; an allocation goes to
//!   the first sub-heap whose limit exceeds its size (eboot+0x8011E0); their
//!   used / total getters sum the sub-heaps' (eboot+0x800F00, +0x800F60);
//! - page manager at [eboot+0x2EE4D60]: 64 KiB pages of direct memory that
//!   heaps map on demand (sceKernelBatchMap); free pages = u32 +0x70 + +0x74
//!   + +0x110 + +0x114 (eboot+0x80D4D0); an allocation fails at once when
//!   +0x110 + +0x114 pages do not cover it (eboot+0x80DA7F); pool pages at
//!   +0x100; the lowest free bytes so far at eboot+0x2EE4D38.
//!
//! Each piece is checked against the code bytes it was read from first;
//! another executable reports nothing.

use std::sync::atomic::{AtomicU64, Ordering};

const HEAP_TABLE: u64 = 0x2EA0DF0;
const HEAP_SLOTS: u64 = 1024;
const PAGE_MANAGER: u64 = 0x2EE4D60;
const LOWEST_FREE: u64 = 0x2EE4D38;
/// eboot+0x80D4D0: `mov rcx,[rip+..]; mov edx,[rcx+74h]; mov eax,[rcx+114h];
/// add edx,[rcx+70h]; add eax,[rcx+110h]` (free bytes of the page manager).
const FREE_PAGES_FN: u64 = 0x80D4D0;
const FREE_PAGES_CODE: [u8; 22] = [0x48, 0x8b, 0x0d, 0x89, 0x78, 0x6d, 0x02, 0x8b, 0x51, 0x74, 0x8b, 0x81, 0x14, 0x01, 0x00, 0x00, 0x03, 0x51, 0x70, 0x03, 0x81, 0x10];

fn base() -> Option<u64> {
    crate::loader::main_module().map(|m| m.base)
}

fn rd<const N: usize>(a: u64) -> Option<[u8; N]> {
    if a < 0x1000 || !crate::mem::readable(a, N as u64) {
        return None;
    }
    let mut out = [0u8; N];
    for (i, b) in out.iter_mut().enumerate() {
        *b = unsafe { std::ptr::read_volatile((a + i as u64) as *const u8) };
    }
    Some(out)
}

fn rd64(a: u64) -> Option<u64> {
    rd::<8>(a).map(u64::from_le_bytes)
}

fn rd32(a: u64) -> Option<u32> {
    rd::<4>(a).map(u32::from_le_bytes)
}

fn cstr(a: u64) -> String {
    let mut s = String::new();
    for i in 0..48 {
        match rd::<1>(a + i) {
            Some([0]) | None => break,
            Some([c]) if c.is_ascii_graphic() || c == b' ' => s.push(c as char),
            Some(_) => s.push('?'),
        }
    }
    s
}

/// How a heap method computes its value, decoded from its code.
enum Getter {
    /// `mov rax,[rdi+d]; ret`
    Field64(u64),
    /// `mov eax,[rdi+d]; ret`
    Field32(u64),
    /// The sum of the sub-heaps' method at this vtable offset (composite).
    SubHeaps(u64),
}

fn getter(f: u64) -> Option<Getter> {
    let c = rd::<0x40>(f)?;
    match c {
        [0x48, 0x8b, 0x47, d, 0xc3, ..] => Some(Getter::Field64(d as u64)),
        [0x48, 0x8b, 0x87, a, b, cc, dd, 0xc3, ..] => Some(Getter::Field64(u32::from_le_bytes([a, b, cc, dd]) as u64)),
        [0x8b, 0x47, d, 0xc3, ..] => Some(Getter::Field32(d as u64)),
        [0x8b, 0x87, a, b, cc, dd, 0xc3, ..] => Some(Getter::Field32(u32::from_le_bytes([a, b, cc, dd]) as u64)),
        // eboot+0x800F00 / +0x800F60: `cmp dword [rdi+88h],0` ... `call [rax+X]`
        // over the {limit, heap} pairs at +0x90.
        _ if c[0xb..0x12] == [0x83, 0xbf, 0x88, 0, 0, 0, 0] && c[0x36..0x38] == [0xff, 0x50] => Some(Getter::SubHeaps(c[0x38] as u64)),
        _ => None,
    }
}

/// The value of the heap method at vtable offset `slot`.
fn heap_value(heap: u64, slot: u64, depth: u32) -> Option<u64> {
    let vt = rd64(heap)?;
    let f = rd64(vt + slot)?;
    match getter(f)? {
        Getter::Field64(d) => rd64(heap + d),
        Getter::Field32(d) => rd32(heap + d).map(u64::from),
        Getter::SubHeaps(s) if depth < 3 => {
            let mut sum = 0u64;
            for (_, sub) in sub_heaps(heap) {
                sum += heap_value(sub, s, depth + 1)?;
            }
            Some(sum)
        }
        Getter::SubHeaps(_) => None,
    }
}

fn is_composite(heap: u64) -> bool {
    rd64(heap).and_then(|vt| rd64(vt + 0x70)).and_then(getter).is_some_and(|g| matches!(g, Getter::SubHeaps(_)))
}

/// A composite heap's {size limit, heap} pairs.
fn sub_heaps(heap: u64) -> Vec<(u64, u64)> {
    let n = rd32(heap + 0x88).unwrap_or(0).min(16) as u64;
    (0..n).filter_map(|i| Some((rd64(heap + 0x90 + 16 * i)?, rd64(heap + 0x98 + 16 * i)?))).collect()
}

fn mib(v: u64) -> String {
    format!("{:.0}", v as f64 / 1048576.0)
}

/// The page manager's free and pool bytes and the lowest free so far.
fn pages() -> Option<(u64, u64, u64)> {
    let b = base()?;
    if rd::<22>(b + FREE_PAGES_FN)? != FREE_PAGES_CODE {
        return None;
    }
    let vm = rd64(b + PAGE_MANAGER)?;
    let free: u64 = [0x70u64, 0x74, 0x110, 0x114].iter().map(|&o| rd32(vm + o).unwrap_or(0) as u64).sum();
    let pool = rd64(vm + 0x100)?;
    Some((free << 16, pool << 16, rd64(b + LOWEST_FREE).unwrap_or(0)))
}

/// One line on the title's heaps and page pool ("" before they exist or for
/// another executable).
pub fn line() -> String {
    let Some(b) = base() else { return String::new() };
    let Some((free, pool, lowest)) = pages() else { return String::new() };
    let mut heaps = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for i in 0..HEAP_SLOTS {
        let Some(h) = rd64(b + HEAP_TABLE + 8 * i).filter(|&h| h != 0) else { continue };
        if !seen.insert(h) {
            continue;
        }
        let (Some(used), Some(total)) = (heap_value(h, 0x60, 0), heap_value(h, 0x70, 0)) else { continue };
        let mut s = format!("{} {}/{}", cstr(rd64(h + 0x10).unwrap_or(0)), mib(used), mib(total));
        if is_composite(h) {
            let parts: Vec<String> = sub_heaps(h)
                .into_iter()
                .map(|(limit, sub)| {
                    let lim = if limit == u64::MAX { "any".to_string() } else if limit >= 1 << 20 { format!("<{}M", limit >> 20) } else { format!("<{}K", (limit >> 10).max(1)) };
                    format!("{lim} {}/{}", heap_value(sub, 0x60, 1).map(mib).unwrap_or("?".into()), heap_value(sub, 0x70, 1).map(mib).unwrap_or("?".into()))
                })
                .collect();
            s.push_str(&format!(" [{}]", parts.join(", ")));
        }
        heaps.push((total, s));
    }
    heaps.sort_by(|a, b| b.0.cmp(&a.0));
    let list: Vec<String> = heaps.into_iter().map(|(_, s)| s).collect();
    format!("title memory (MiB used/total): pages free {} of {} (lowest {}); heaps: {}", mib(free), mib(pool), mib(lowest), list.join("; "))
}

/// For the status report.
pub fn status() -> String {
    let l = line();
    if l.is_empty() { l } else { format!("    {l}\n") }
}

static LAST_FREE: AtomicU64 = AtomicU64::new(u64::MAX);
static LAST_AT: AtomicU64 = AtomicU64::new(0);

/// Called about once a second: logs the line when the free pages moved by
/// 128 MiB or more since the last one, at most every `DS_HEAPS` seconds
/// (default 10; 0 = off).
pub fn tick(now_s: u64) {
    static EVERY: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    let every = *EVERY.get_or_init(|| std::env::var("DS_HEAPS").ok().and_then(|v| v.parse().ok()).unwrap_or(10));
    if every == 0 || now_s < LAST_AT.load(Ordering::Relaxed) + every {
        return;
    }
    let Some((free, _, _)) = pages() else { return };
    let last = LAST_FREE.load(Ordering::Relaxed);
    if last != u64::MAX && free.abs_diff(last) < 128 << 20 {
        return;
    }
    LAST_FREE.store(free, Ordering::Relaxed);
    LAST_AT.store(now_s, Ordering::Relaxed);
    crate::info!("{}", line());
}
