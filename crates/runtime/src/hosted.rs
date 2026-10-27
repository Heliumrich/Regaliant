//! Hosted mode: dsport under a host process that takes the write faults on
//! guest pages dsport armed (`--host`; RENDERER.md "Resident front end",
//! stage H1).
//!
//! A fault handled inside the process overwrites the 128 bytes below the
//! faulting thread's stack pointer with the state Windows saves for the
//! handler, and the title's leaf functions (System V) keep values there; a
//! debugger is told of a fault before Windows builds that frame, so faults
//! it answers leave them intact (FINDINGS.md 2026-10-26, `faultbench`). Each
//! one stops the whole process for 15-50 us, so this is for writes that are
//! rare.
//!
//! `dsport --host <args>` starts dsport again with the same arguments as its
//! debuggee and shares a mapping with it (named in `DS_HOSTED`): a bitmap of
//! armed guest pages (two levels: an index of 128 MiB leaves over the guest
//! range, leaves allocated by dsport) and a ring of the pages the host made
//! accessible again. A fault on an armed page (read-only or no-access) is
//! answered in the host: the page made read-write, its bit cleared, the page
//! with the access and the faulting instruction put in the ring, the access
//! retried. A fault on a guest page that allows the access by the time the
//! host sees it raced one the host answered for another thread (or a
//! disarm): the access is retried. Every other exception goes to dsport's
//! own handlers (first chance not handled). dsport arms a page by setting its bit before
//! protecting it, so an armed page's fault never reaches its own handlers,
//! and disarms by unprotecting before clearing the bit, one arm or disarm
//! at a time (`arm_lock`); after a mapping change the bits are only cleared
//! (`forget`).

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use windows_sys::Win32::Foundation::*;
use windows_sys::Win32::System::Diagnostics::Debug::*;
use windows_sys::Win32::System::Memory::*;
use windows_sys::Win32::System::Threading::*;

const PAGE: u64 = 4096;
/// Pages per leaf of the bitmap (128 MiB of guest space).
const LEAF_PAGES: u64 = 1 << 15;
/// Leaves over the guest range.
const L1: usize = ((crate::mem::GUEST_END - crate::mem::GUEST_BASE) / PAGE / LEAF_PAGES) as usize;
/// Leaves that can be allocated (512 GiB of armed guest space).
const MAX_LEAVES: usize = 4096;
/// Ring entries.
const RING: usize = 1 << 16;
const MAGIC: u64 = 0x6473_706f_7274_4831; // "dsportH1"

/// The shared mapping.
#[repr(C)]
struct Shared {
    magic: u64,
    /// Write faults the host answered, its time answering them (ns), the
    /// exceptions it passed on, ring entries lost to a full ring.
    faults: AtomicU64,
    host_ns: AtomicU64,
    passed: AtomicU64,
    overflow: AtomicU64,
    /// Faults retried because the page allowed the access by then.
    retried: AtomicU64,
    /// Debug events by code (1-9: exception, thread and process creation,
    /// thread and process exit, DLL load and unload, debug string, RIP), and
    /// the time from each event's report until the host continued it (ns).
    events: [AtomicU64; 10],
    event_ns: [AtomicU64; 10],
    /// Ring entries written (by the host).
    head: AtomicU64,
    /// Ring entries dsport took.
    tail: AtomicU64,
    leaves_used: AtomicU32,
    _pad: u32,
    /// Leaf index + 1 per 128 MiB of guest space (0: none).
    l1: [AtomicU32; L1],
    /// Per 128 MiB of guest space: armed bits cleared there so far (faults
    /// the host answered, disarms, mapping changes): a reader's view of
    /// the leaf is current while the count stays the same.
    leaf_events: [AtomicU64; L1],
    /// The same over the whole guest range.
    all_events: AtomicU64,
    /// (page | 1 for a write, the faulting instruction).
    ring: [[AtomicU64; 2]; RING],
    leaves: [[AtomicU64; (LEAF_PAGES / 64) as usize]; MAX_LEAVES],
}

/// dsport's view of the mapping (hosted), or null.
static SHARED: std::sync::atomic::AtomicPtr<Shared> = std::sync::atomic::AtomicPtr::new(std::ptr::null_mut());

fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

fn shared() -> Option<&'static Shared> {
    let p = SHARED.load(Ordering::Acquire);
    // SAFETY: a mapping of `Shared`'s size, kept for the process's life.
    (!p.is_null()).then(|| unsafe { &*p })
}

/// The bit of guest page `a` (leaf allocated when `create`).
fn bit(s: &Shared, a: u64, create: bool) -> Option<(&AtomicU64, u64)> {
    if !(crate::mem::GUEST_BASE..crate::mem::GUEST_END).contains(&a) {
        return None;
    }
    let page = (a - crate::mem::GUEST_BASE) / PAGE;
    let slot = &s.l1[(page / LEAF_PAGES) as usize];
    let mut leaf = slot.load(Ordering::Acquire);
    if leaf == 0 {
        if !create {
            return None;
        }
        let n = s.leaves_used.fetch_add(1, Ordering::AcqRel);
        if n as usize >= MAX_LEAVES {
            return None;
        }
        leaf = match slot.compare_exchange(0, n + 1, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => n + 1,
            // Another thread's leaf (this one stays unused).
            Err(other) => other,
        };
    }
    let i = page % LEAF_PAGES;
    Some((&s.leaves[(leaf - 1) as usize][(i / 64) as usize], 1u64 << (i % 64)))
}

/// Whether dsport runs under its host.
pub fn on() -> bool {
    !SHARED.load(Ordering::Acquire).is_null()
}

/// Maps the host's shared state when `DS_HOSTED` names it (at start-up).
pub fn init() {
    let Ok(name) = std::env::var("DS_HOSTED") else { return };
    let size = std::mem::size_of::<Shared>();
    unsafe {
        let h = OpenFileMappingW(FILE_MAP_ALL_ACCESS, 0, wide(&name).as_ptr());
        if h.is_null() {
            crate::error!("hosted: cannot open the host's mapping {name}: error {}", GetLastError());
            return;
        }
        let v = MapViewOfFile(h, FILE_MAP_ALL_ACCESS, 0, 0, size);
        if v.Value.is_null() {
            crate::error!("hosted: cannot map {name}: error {}", GetLastError());
            return;
        }
        let s = v.Value as *mut Shared;
        if (*s).magic != MAGIC {
            crate::error!("hosted: {name} is not the host's mapping");
            return;
        }
        SHARED.store(s, Ordering::Release);
    }
    crate::info!("hosted: under the host (write faults on armed guest pages answered there; {name})");
    // Every 5 s with debug events or faults since the last look: their
    // counts (each one stops the whole process while the host handles it).
    let _ = std::thread::Builder::new().name("host-events".into()).spawn(|| {
        let Some(s) = shared() else { return };
        let mut last = [0u64; 11];
        loop {
            std::thread::sleep(std::time::Duration::from_secs(5));
            let mut now = [0u64; 11];
            for (k, n) in now.iter_mut().enumerate().take(10) {
                *n = s.events[k].load(Ordering::Relaxed);
            }
            now[10] = s.faults.load(Ordering::Relaxed);
            if now == last {
                continue;
            }
            const NAMES: [&str; 10] = ["", "exceptions", "thread starts", "process start", "thread exits", "process exit", "DLL loads", "DLL unloads", "debug strings", "RIPs"];
            let parts: Vec<String> = (1..10).filter(|&k| now[k] != last[k]).map(|k| format!("{} {}", NAMES[k], now[k] - last[k])).collect();
            crate::info!("hosted: in 5 s: {}; armed-page faults {}", parts.join(", "), now[10] - last[10]);
            last = now;
        }
    });
}

/// Arming and disarming one at a time: a disarm between an arm's bits and
/// its protection would leave read-only pages without bits (a write there
/// would reach dsport's own handlers). Callers that check something before
/// arming (`gpu::fe2`) hold it across the check too.
static ARM_LOCK: parking_lot::ReentrantMutex<()> = parking_lot::ReentrantMutex::new(());

/// Holds off every other arm and disarm (re-entrant: `arm` and `disarm`
/// under it are fine).
pub fn arm_lock() -> parking_lot::ReentrantMutexGuard<'static, ()> {
    ARM_LOCK.lock()
}

/// Arms guest [base, base+len) (whole pages) against writes: its bits set,
/// then read-only.
pub fn arm(base: u64, len: u64) -> Result<(), String> {
    arm_as(base, len, PAGE_READONLY)
}

/// Arms guest [base, base+len) against any access: its bits set, then
/// no-access.
pub fn arm_noaccess(base: u64, len: u64) -> Result<(), String> {
    arm_as(base, len, PAGE_NOACCESS)
}

fn arm_as(base: u64, len: u64, prot: u32) -> Result<(), String> {
    let s = shared().ok_or("not hosted")?;
    let _lock = arm_lock();
    let (lo, hi) = (base & !(PAGE - 1), (base + len).next_multiple_of(PAGE));
    for a in (lo..hi).step_by(PAGE as usize) {
        let (w, m) = bit(s, a, true).ok_or_else(|| format!("{a:#x}: outside the guest range or out of leaves"))?;
        w.fetch_or(m, Ordering::AcqRel);
    }
    let mut old = 0u32;
    if unsafe { VirtualProtect(lo as _, (hi - lo) as usize, prot, &mut old) } == 0 {
        let e = unsafe { GetLastError() };
        disarm_bits(s, lo, hi);
        return Err(format!("VirtualProtect {lo:#x}+{:#x}: error {e}", hi - lo));
    }
    Ok(())
}

fn disarm_bits(s: &Shared, lo: u64, hi: u64) {
    let mut last_leaf = usize::MAX;
    for a in (lo..hi).step_by(PAGE as usize) {
        if let Some((w, m)) = bit(s, a, false) {
            w.fetch_and(!m, Ordering::AcqRel);
            let leaf = leaf_of(a);
            if leaf != last_leaf {
                s.leaf_events[leaf].fetch_add(1, Ordering::AcqRel);
                s.all_events.fetch_add(1, Ordering::AcqRel);
                last_leaf = leaf;
            }
        }
    }
}

/// The 128 MiB of guest space holding `a` (an index of `Shared::l1`).
fn leaf_of(a: u64) -> usize {
    ((a - crate::mem::GUEST_BASE) / PAGE / LEAF_PAGES) as usize
}

/// Armed bits cleared so far anywhere (0 when not hosted): unchanged, no
/// armed page lost its bit.
#[inline]
pub fn all_events() -> u64 {
    shared().map_or(0, |s| s.all_events.load(Ordering::Acquire))
}

/// Armed bits cleared so far in the 128 MiB of guest space holding `a`
/// (0 when not hosted): unchanged, no armed page there lost its bit.
#[inline]
pub fn leaf_events(a: u64) -> u64 {
    let Some(s) = shared() else { return 0 };
    if !(crate::mem::GUEST_BASE..crate::mem::GUEST_END).contains(&a) {
        return 0;
    }
    s.leaf_events[leaf_of(a)].load(Ordering::Acquire)
}

/// Disarm runs whose pages could not be made read-write (they keep their
/// bits: the host still answers their faults).
static DISARM_FAILED: AtomicU64 = AtomicU64::new(0);

/// Disarms guest [base, base+len): its armed pages read-write again, then
/// their bits cleared, run by run within the host's regions. One
/// VirtualProtect cannot span separately allocated or uncommitted regions,
/// and clearing the bits after such a failed call left read-only pages
/// without bits, whose writes reach dsport's own handlers (a title crash
/// on a page of a writable mapping: FINDINGS.md 2026-10-27 night); pages
/// not armed keep their protection.
pub fn disarm(base: u64, len: u64) {
    let Some(s) = shared() else { return };
    let _lock = arm_lock();
    let lo = (base & !(PAGE - 1)).max(crate::mem::GUEST_BASE);
    let hi = base.saturating_add(len).next_multiple_of(PAGE).min(crate::mem::GUEST_END);
    let armed = |a: u64| bit(s, a, false).is_some_and(|(w, m)| w.load(Ordering::Acquire) & m != 0);
    let mut a = lo;
    while a < hi {
        // The next armed page from a (unarmed words and leaves skipped).
        let Some((w, _)) = bit(s, a, false) else {
            a = crate::mem::GUEST_BASE + ((a - crate::mem::GUEST_BASE) / (PAGE * LEAF_PAGES) + 1) * PAGE * LEAF_PAGES;
            continue;
        };
        let i = (a - crate::mem::GUEST_BASE) / PAGE % 64;
        let rest = w.load(Ordering::Acquire) >> i;
        if rest == 0 {
            a += (64 - i) * PAGE;
            continue;
        }
        a += rest.trailing_zeros() as u64 * PAGE;
        if a >= hi {
            break;
        }
        // Its run: armed pages up to the end of the host region holding it.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        let region_end = if unsafe { VirtualQuery(a as _, &mut info, std::mem::size_of::<MEMORY_BASIC_INFORMATION>()) } != 0 { info.BaseAddress as u64 + info.RegionSize as u64 } else { a + PAGE };
        let mut e = a + PAGE;
        while e < hi && e < region_end && armed(e) {
            e += PAGE;
        }
        let mut old = 0u32;
        if unsafe { VirtualProtect(a as _, (e - a) as usize, PAGE_READWRITE, &mut old) } != 0 {
            disarm_bits(s, a, e);
        } else {
            let err = unsafe { GetLastError() };
            if DISARM_FAILED.fetch_add(1, Ordering::Relaxed) < 8 {
                crate::warn!("hosted: disarming {a:#x}+{:#x} failed (error {err}): its pages stay armed", e - a);
            }
        }
        a = e;
    }
    // Mirrored pages no longer protected leave the mirror.
    crate::gpu::gmirror::disarmed(lo, hi.saturating_sub(lo));
}

/// The armed bits of the 64 pages from `a` (a multiple of 64 pages from the
/// guest range's start): bit i for page a + 4096 * i (0 when not hosted or
/// none of them was ever armed).
#[inline]
pub fn armed_word(a: u64) -> u64 {
    let Some(s) = shared() else { return 0 };
    bit(s, a, false).map_or(0, |(w, _)| w.load(Ordering::Acquire))
}

/// Whether every page of guest [base, base+len) is armed (no fault on any
/// of them since it was armed, no disarm, no mapping change).
pub fn all_armed(base: u64, len: u64) -> bool {
    let Some(s) = shared() else { return false };
    let (lo, hi) = (base & !(PAGE - 1), (base + len.max(1)).next_multiple_of(PAGE));
    (lo..hi).step_by(PAGE as usize).all(|a| bit(s, a, false).is_some_and(|(w, m)| w.load(Ordering::Acquire) & m != 0))
}

/// Disarms guest [base, base+len) if a page of it is armed (imports need
/// writable pages: the driver pins them for writing).
pub fn disarm_if_armed(base: u64, len: u64) {
    let Some(s) = shared() else { return };
    let (lo, hi) = (base & !(PAGE - 1), base.saturating_add(len).next_multiple_of(PAGE).min(crate::mem::GUEST_END));
    // Leaf by leaf: a range without a leaf has nothing armed.
    let mut a = lo;
    while a < hi {
        let leaf_end = ((a - crate::mem::GUEST_BASE) / (PAGE * LEAF_PAGES) + 1) * PAGE * LEAF_PAGES + crate::mem::GUEST_BASE;
        let end = leaf_end.min(hi);
        if bit(s, a, false).is_some() && (a..end).step_by(PAGE as usize).any(|p| bit(s, p, false).is_some_and(|(w, m)| w.load(Ordering::Acquire) & m != 0)) {
            disarm(a, end - a);
        }
        a = end;
    }
}

/// Guest [base, base+len) changed its mapping or protection: its pages are
/// no longer armed (bits cleared, protection left as the change set it).
pub fn forget(base: u64, len: u64) {
    let Some(s) = shared() else { return };
    let (lo, hi) = (base & !(PAGE - 1), base.saturating_add(len).next_multiple_of(PAGE));
    disarm_bits(s, lo, hi.min(crate::mem::GUEST_END));
    crate::hostcensus::forget(base, len);
    crate::gpu::gmirror::disarmed(lo, hi.saturating_sub(lo));
}

/// The faults the host answered since the last call: (page, a write, the
/// faulting instruction); false when the ring overflowed meanwhile (faults
/// were lost: treat all armed pages as touched).
pub fn drain_faults(out: &mut Vec<(u64, bool, u64)>) -> bool {
    let Some(s) = shared() else { return true };
    let head = s.head.load(Ordering::Acquire);
    let mut tail = s.tail.load(Ordering::Acquire);
    let lost = head.saturating_sub(tail) > RING as u64;
    if lost {
        tail = head - RING as u64;
    }
    while tail < head {
        let e = &s.ring[(tail % RING as u64) as usize];
        let (a, rip) = (e[0].load(Ordering::Acquire), e[1].load(Ordering::Acquire));
        out.push((a & !(PAGE - 1), a & 1 != 0, rip));
        tail += 1;
    }
    s.tail.store(tail, Ordering::Release);
    !lost && s.overflow.load(Ordering::Acquire) == 0
}

/// The pages the host made accessible since the last call (`drain_faults`).
pub fn drain(out: &mut Vec<u64>) -> bool {
    let mut v = Vec::new();
    let ok = drain_faults(&mut v);
    out.extend(v.into_iter().map(|(p, _, _)| p));
    ok
}

/// Write faults on armed pages the host answered so far (0 when not hosted).
pub fn faults() -> u64 {
    shared().map_or(0, |s| s.faults.load(Ordering::Relaxed))
}

/// Status line ("" when not hosted).
pub fn status() -> String {
    let Some(s) = shared() else { return String::new() };
    let f = s.faults.load(Ordering::Relaxed);
    const NAMES: [&str; 10] = ["", "exceptions", "thread starts", "process start", "thread exits", "process exit", "DLL loads", "DLL unloads", "debug strings", "RIPs"];
    let events: Vec<String> = (1..10)
        .filter_map(|k| {
            let n = s.events[k].load(Ordering::Relaxed);
            (n > 0).then(|| format!("{} {n} ({:.0} ms in the host)", NAMES[k], s.event_ns[k].load(Ordering::Relaxed) as f64 / 1e6))
        })
        .collect();
    format!(
        "    hosted: {} faults on armed pages answered by the host ({:.1} us each there), {} retried (raced an answered one), {} exceptions passed on, {} leaves of the armed-page map, ring overflows {}, disarms that failed {}; debug events: {}\n{}",
        f,
        s.host_ns.load(Ordering::Relaxed) as f64 / 1000.0 / f.max(1) as f64,
        s.retried.load(Ordering::Relaxed),
        s.passed.load(Ordering::Relaxed),
        s.leaves_used.load(Ordering::Relaxed),
        s.overflow.load(Ordering::Relaxed),
        DISARM_FAILED.load(Ordering::Relaxed),
        events.join(", "),
        crate::hostcensus::status()
    )
}

/// `dsport --host ...`: runs dsport with the other arguments as its debuggee
/// and answers write faults on armed pages; exits with its exit code.
pub fn run_host() -> ! {
    let size = std::mem::size_of::<Shared>();
    let name = format!("Local\\dsport-host-{}", std::process::id());
    let s: &Shared = unsafe {
        let h = CreateFileMappingW(INVALID_HANDLE_VALUE, std::ptr::null(), PAGE_READWRITE, (size as u64 >> 32) as u32, size as u32, wide(&name).as_ptr());
        if h.is_null() {
            eprintln!("dsport --host: CreateFileMapping: error {}", GetLastError());
            std::process::exit(2);
        }
        let v = MapViewOfFile(h, FILE_MAP_ALL_ACCESS, 0, 0, size);
        if v.Value.is_null() {
            eprintln!("dsport --host: MapViewOfFile: error {}", GetLastError());
            std::process::exit(2);
        }
        let s = v.Value as *mut Shared;
        (*s).magic = MAGIC;
        &*s
    };
    let exe = std::env::current_exe().expect("own path");
    let args: Vec<String> = std::env::args().skip(1).filter(|a| a != "--host").collect();
    let cmdline = std::iter::once(format!("\"{}\"", exe.display())).chain(args.iter().map(|a| if a.contains(' ') { format!("\"{a}\"") } else { a.clone() })).collect::<Vec<_>>().join(" ");
    // SAFETY: before any other thread of the host exists.
    unsafe {
        std::env::set_var("DS_HOSTED", &name);
        // A process started by a debugger otherwise gets the checking heap.
        std::env::set_var("_NO_DEBUG_HEAP", "1");
    }
    let mut cmd = wide(&cmdline);
    let mut si: STARTUPINFOW = unsafe { std::mem::zeroed() };
    si.cb = std::mem::size_of::<STARTUPINFOW>() as u32;
    let mut pi: PROCESS_INFORMATION = unsafe { std::mem::zeroed() };
    let ok = unsafe { CreateProcessW(std::ptr::null(), cmd.as_mut_ptr(), std::ptr::null(), std::ptr::null(), 1, DEBUG_ONLY_THIS_PROCESS, std::ptr::null(), std::ptr::null(), &si, &mut pi) };
    if ok == 0 {
        eprintln!("dsport --host: CreateProcess: error {}", unsafe { GetLastError() });
        std::process::exit(2);
    }
    let mut first_bp = true;
    let mut code = 0u32;
    loop {
        let mut ev: DEBUG_EVENT = unsafe { std::mem::zeroed() };
        if unsafe { WaitForDebugEvent(&mut ev, INFINITE) } == 0 {
            break;
        }
        let t_event = std::time::Instant::now();
        let kind = (ev.dwDebugEventCode as usize).min(9);
        let mut status = DBG_CONTINUE;
        match ev.dwDebugEventCode {
            CREATE_PROCESS_DEBUG_EVENT => unsafe {
                let f = ev.u.CreateProcessInfo.hFile;
                if !f.is_null() {
                    CloseHandle(f);
                }
            },
            LOAD_DLL_DEBUG_EVENT => unsafe {
                let f = ev.u.LoadDll.hFile;
                if !f.is_null() {
                    CloseHandle(f);
                }
            },
            EXCEPTION_DEBUG_EVENT => unsafe {
                let t = std::time::Instant::now();
                let e = ev.u.Exception;
                let rec = e.ExceptionRecord;
                let exc = rec.ExceptionCode as u32;
                let (write, addr) = (rec.ExceptionInformation[0] == 1, rec.ExceptionInformation[1] as u64);
                let armed = exc == 0xC000_0005 && e.dwFirstChance != 0 && bit(s, addr, false).is_some_and(|(w, m)| w.load(Ordering::Acquire) & m != 0);
                if armed {
                    let page = addr & !(PAGE - 1);
                    let mut old = 0u32;
                    VirtualProtectEx(pi.hProcess, page as _, PAGE as usize, PAGE_READWRITE, &mut old);
                    if let Some((w, m)) = bit(s, page, false) {
                        w.fetch_and(!m, Ordering::AcqRel);
                    }
                    s.leaf_events[leaf_of(page)].fetch_add(1, Ordering::AcqRel);
                    s.all_events.fetch_add(1, Ordering::AcqRel);
                    let head = s.head.load(Ordering::Acquire);
                    if head.saturating_sub(s.tail.load(Ordering::Acquire)) >= RING as u64 {
                        s.overflow.fetch_add(1, Ordering::Relaxed);
                    }
                    let slot = &s.ring[(head % RING as u64) as usize];
                    slot[0].store(page | write as u64, Ordering::Release);
                    slot[1].store(rec.ExceptionAddress as u64, Ordering::Release);
                    s.head.store(head + 1, Ordering::Release);
                    s.faults.fetch_add(1, Ordering::Relaxed);
                    s.host_ns.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
                } else if exc == 0xC000_0005 && e.dwFirstChance != 0 && rec.ExceptionInformation[0] <= 1 && allows(pi.hProcess, addr, write) {
                    s.retried.fetch_add(1, Ordering::Relaxed);
                } else if exc == 0x8000_0003 && first_bp {
                    first_bp = false;
                } else {
                    status = DBG_EXCEPTION_NOT_HANDLED;
                    s.passed.fetch_add(1, Ordering::Relaxed);
                }
            },
            EXIT_PROCESS_DEBUG_EVENT => unsafe {
                code = ev.u.ExitProcess.dwExitCode;
                ContinueDebugEvent(ev.dwProcessId, ev.dwThreadId, DBG_CONTINUE);
                break;
            },
            _ => {}
        }
        s.events[kind].fetch_add(1, Ordering::Relaxed);
        s.event_ns[kind].fetch_add(t_event.elapsed().as_nanos() as u64, Ordering::Relaxed);
        unsafe { ContinueDebugEvent(ev.dwProcessId, ev.dwThreadId, status) };
    }
    std::process::exit(code as i32);
}

/// Whether the debuggee's guest page at `addr` is committed and allows the
/// access (a read, or a write when `write`).
fn allows(process: HANDLE, addr: u64, write: bool) -> bool {
    if !(crate::mem::GUEST_BASE..crate::mem::GUEST_END).contains(&addr) {
        return false;
    }
    let mut m: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    if unsafe { VirtualQueryEx(process, addr as _, &mut m, std::mem::size_of::<MEMORY_BASIC_INFORMATION>()) } == 0 || m.State != MEM_COMMIT {
        return false;
    }
    let p = m.Protect & 0xff;
    if write {
        matches!(p, PAGE_READWRITE | PAGE_EXECUTE_READWRITE)
    } else {
        matches!(p, PAGE_READONLY | PAGE_READWRITE | PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE)
    }
}

/// Writes `addr` with the 16 qwords below the stack pointer set to a pattern
/// and the AVX upper state live; the mask of slots changed after the write
/// (bit k: [rsp - 8 (k + 1)]), as `faultbench`'s probe.
#[inline(never)]
unsafe fn probe(addr: *mut u32) -> u64 {
    let mask: u64;
    unsafe {
        core::arch::asm!(
            "vpcmpeqd ymm1, ymm1, ymm1",
            "mov rax, 0x5a5a5a5a5a5a5a5a",
            "mov qword ptr [rsp-8], rax",
            "mov qword ptr [rsp-16], rax",
            "mov qword ptr [rsp-24], rax",
            "mov qword ptr [rsp-32], rax",
            "mov qword ptr [rsp-40], rax",
            "mov qword ptr [rsp-48], rax",
            "mov qword ptr [rsp-56], rax",
            "mov qword ptr [rsp-64], rax",
            "mov qword ptr [rsp-72], rax",
            "mov qword ptr [rsp-80], rax",
            "mov qword ptr [rsp-88], rax",
            "mov qword ptr [rsp-96], rax",
            "mov qword ptr [rsp-104], rax",
            "mov qword ptr [rsp-112], rax",
            "mov qword ptr [rsp-120], rax",
            "mov qword ptr [rsp-128], rax",
            "mov dword ptr [{a}], 1",
            "xor {m}, {m}",
            "cmp qword ptr [rsp-8], rax", "setne cl", "movzx ecx, cl", "or {m}, rcx",
            "cmp qword ptr [rsp-16], rax", "setne cl", "movzx ecx, cl", "shl ecx, 1", "or {m}, rcx",
            "cmp qword ptr [rsp-24], rax", "setne cl", "movzx ecx, cl", "shl ecx, 2", "or {m}, rcx",
            "cmp qword ptr [rsp-32], rax", "setne cl", "movzx ecx, cl", "shl ecx, 3", "or {m}, rcx",
            "cmp qword ptr [rsp-40], rax", "setne cl", "movzx ecx, cl", "shl ecx, 4", "or {m}, rcx",
            "cmp qword ptr [rsp-48], rax", "setne cl", "movzx ecx, cl", "shl ecx, 5", "or {m}, rcx",
            "cmp qword ptr [rsp-56], rax", "setne cl", "movzx ecx, cl", "shl ecx, 6", "or {m}, rcx",
            "cmp qword ptr [rsp-64], rax", "setne cl", "movzx ecx, cl", "shl ecx, 7", "or {m}, rcx",
            "cmp qword ptr [rsp-72], rax", "setne cl", "movzx ecx, cl", "shl ecx, 8", "or {m}, rcx",
            "cmp qword ptr [rsp-80], rax", "setne cl", "movzx ecx, cl", "shl ecx, 9", "or {m}, rcx",
            "cmp qword ptr [rsp-88], rax", "setne cl", "movzx ecx, cl", "shl ecx, 10", "or {m}, rcx",
            "cmp qword ptr [rsp-96], rax", "setne cl", "movzx ecx, cl", "shl ecx, 11", "or {m}, rcx",
            "cmp qword ptr [rsp-104], rax", "setne cl", "movzx ecx, cl", "shl ecx, 12", "or {m}, rcx",
            "cmp qword ptr [rsp-112], rax", "setne cl", "movzx ecx, cl", "shl ecx, 13", "or {m}, rcx",
            "cmp qword ptr [rsp-120], rax", "setne cl", "movzx ecx, cl", "shl ecx, 14", "or {m}, rcx",
            "cmp qword ptr [rsp-128], rax", "setne cl", "movzx ecx, cl", "shl ecx, 15", "or {m}, rcx",
            a = in(reg) addr,
            m = out(reg) mask,
            out("rax") _,
            out("rcx") _,
            out("ymm1") _,
            options(nostack),
        );
    }
    mask
}

/// `DS_HOST_TEST=1` (hosted, at start-up): arms pages of a 1 MiB guest
/// region one at a time and writes each from code that keeps values below
/// its stack pointer; the values, the ring and the pages' state are checked.
pub fn self_test_maybe() {
    if std::env::var("DS_HOST_TEST").ok().as_deref() != Some("1") {
        return;
    }
    match self_test() {
        Ok(s) => crate::info!("host test passed: {s}"),
        Err(e) => crate::error!("host test failed: {e}"),
    }
}

fn self_test() -> Result<String, String> {
    if !on() {
        return Err("not hosted (run with --host)".into());
    }
    const N: u64 = 4000;
    let base = crate::mem::MEM
        .lock()
        .alloc_internal(1 << 20, crate::mem::PROT_CPU_READ | crate::mem::PROT_CPU_WRITE, "host-test")
        .map_err(|e| format!("no guest memory for the test: {e:?}"))?;
    let mut drained = Vec::new();
    drain(&mut drained);
    drained.clear();
    let t = std::time::Instant::now();
    let mut changed = 0u64;
    for i in 0..N {
        let a = base + (i % 256) * PAGE;
        arm(a, PAGE)?;
        if unsafe { probe(a as *mut u32) } != 0 {
            changed += 1;
        }
    }
    let us = t.elapsed().as_secs_f64() * 1e6 / N as f64;
    let ok = drain(&mut drained);
    let wrong: Vec<&u64> = drained.iter().enumerate().filter(|(i, a)| **a != base + (*i as u64 % 256) * PAGE).map(|(_, a)| a).take(3).collect();
    if changed != 0 {
        return Err(format!("the values below the stack pointer changed in {changed} of {N} faults"));
    }
    if drained.len() as u64 != N || !wrong.is_empty() || !ok {
        return Err(format!("{} pages in the ring for {N} faults (wrong: {wrong:x?}, overflow: {})", drained.len(), !ok));
    }
    Ok(format!("{N} write faults on armed guest pages answered by the host, {us:.1} us per arm and fault, the values below the stack pointer intact, every page in the ring in order"))
}

/// The host process's command line flag.
pub fn wants_host() -> bool {
    std::env::args().skip(1).any(|a| a == "--host")
}
