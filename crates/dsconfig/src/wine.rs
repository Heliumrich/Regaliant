//! Whether the program runs under Wine (a Linux or macOS PC playing through Wine or Proton),
//! and what a write-watched page costs there (issue #14, 2026-11-25 night).
//!
//! The runtime write-watches the whole guest range (`mem::take_written` and the takes of its
//! front ends). Windows notes a page's first write after a reset in its kernel (about 0.5 us a
//! page here). Wine does so in the kernel only with Linux 6.7 and newer (in Proton 9 and newer
//! and Wine 10.10 and newer); otherwise it write-protects every watched page and handles a
//! signal at the first write after each reset in the process itself, under one lock for the
//! whole process (several microseconds a page and more). A player's first load into Boletaria read
//! the same files as here 7 to 27 times slower that way and never finished in six minutes.

use std::ffi::{c_char, c_void, CStr};

/// What Wine says about itself.
pub struct Wine {
    /// `wine_get_version` ("10.0", "9.0 (Staging)").
    pub version: String,
    /// `wine_get_build_id`, when exported (Proton's names its release).
    pub build: Option<String>,
    /// `wine_get_host_version`: the host system and its kernel release ("Linux 6.8.0-45-generic").
    pub host: Option<String>,
}

impl Wine {
    /// One line for logs: "Wine 10.0 (build wine-10.0) on Linux 6.8.0".
    pub fn describe(&self) -> String {
        let mut s = format!("Wine {}", self.version);
        if let Some(b) = self.build.as_deref().filter(|b| !b.is_empty()) {
            s.push_str(&format!(" (build {b})"));
        }
        if let Some(h) = &self.host {
            s.push_str(&format!(" on {h}"));
        }
        s
    }
}

/// Wine's own description when the process runs under it (its ntdll exports these names;
/// Windows' does not), else None.
pub fn detect() -> Option<Wine> {
    use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleA, GetProcAddress};
    // SAFETY: the exports are Wine's documented helpers: C functions without arguments
    // returning static strings, and one filling two string pointers.
    unsafe {
        let ntdll = GetModuleHandleA(c"ntdll.dll".as_ptr() as *const u8);
        if ntdll.is_null() {
            return None;
        }
        let text = |p: *const c_char| (!p.is_null()).then(|| CStr::from_ptr(p).to_string_lossy().trim().to_string());
        let version = GetProcAddress(ntdll, c"wine_get_version".as_ptr() as *const u8)?;
        let version: unsafe extern "C" fn() -> *const c_char = std::mem::transmute(version);
        let version = text(version())?;
        let build = GetProcAddress(ntdll, c"wine_get_build_id".as_ptr() as *const u8).and_then(|f| {
            let f: unsafe extern "C" fn() -> *const c_char = std::mem::transmute(f);
            text(f())
        });
        let host = GetProcAddress(ntdll, c"wine_get_host_version".as_ptr() as *const u8).and_then(|f| {
            let f: unsafe extern "C" fn(*mut *const c_char, *mut *const c_char) = std::mem::transmute(f);
            let (mut sys, mut rel) = (std::ptr::null(), std::ptr::null());
            f(&mut sys, &mut rel);
            match (text(sys), text(rel)) {
                (Some(s), Some(r)) => Some(format!("{s} {r}")),
                (s, _) => s,
            }
        });
        Some(Wine { version, build, host })
    }
}

/// Above this many nanoseconds a written page stopped its writer (Wine without the kernel's
/// help: a signal and a protection change each, a few microseconds and more under contention).
/// Windows and Wine with the kernel's help note writes in well under a microsecond.
pub const SLOW_WRITE_WATCH_NS: f64 = 1500.0;

/// What a write to a write-watched page costs right after its watch was reset, in nanoseconds a
/// page (the median of three rounds over 256 pages), or None when write watching is missing or
/// did not report every write.
pub fn write_watch_cost_ns() -> Option<f64> {
    use windows_sys::Win32::System::Memory::{GetWriteWatch, ResetWriteWatch, VirtualAlloc, VirtualFree, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE};
    const MEM_WRITE_WATCH: u32 = 0x0020_0000;
    const PAGES: usize = 256;
    const PAGE: usize = 4096;
    let size = PAGES * PAGE;
    // SAFETY: a private allocation of this function, written inside its bounds and released.
    unsafe {
        let p = VirtualAlloc(std::ptr::null(), size, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE) as *mut u8;
        if p.is_null() {
            return None;
        }
        for i in 0..PAGES {
            std::ptr::write_volatile(p.add(i * PAGE), 1);
        }
        let mut rounds = [0f64; 3];
        let mut reported = true;
        let mut addrs = vec![std::ptr::null_mut::<c_void>(); PAGES];
        for (k, r) in rounds.iter_mut().enumerate() {
            ResetWriteWatch(p as *const c_void, size);
            let t = std::time::Instant::now();
            for i in 0..PAGES {
                std::ptr::write_volatile(p.add(i * PAGE), k as u8 + 2);
            }
            *r = t.elapsed().as_nanos() as f64 / PAGES as f64;
            let mut count = PAGES;
            let mut gran = 0u32;
            reported &= GetWriteWatch(0, p as *const c_void, size, addrs.as_mut_ptr(), &mut count, &mut gran) == 0 && count == PAGES;
        }
        VirtualFree(p as *mut c_void, 0, MEM_RELEASE);
        rounds.sort_by(|a, b| a.total_cmp(b));
        reported.then_some(rounds[1])
    }
}

/// Linux's limit on one process's memory mappings, read through Wine's drive for the host's
/// root (Z:), when it can be: page protections split mappings, and Wine without the kernel's
/// write tracking changes them page by page.
pub fn max_map_count() -> Option<u64> {
    std::fs::read_to_string(r"Z:\proc\sys\vm\max_map_count").ok()?.trim().parse().ok()
}

/// The launcher's status line when writes are slow under Wine (None otherwise; the log says
/// why, `dsport`'s start).
pub fn slow_notice(wine: Option<&Wine>, cost_ns: Option<f64>) -> Option<&'static str> {
    (wine.is_some() && cost_ns.is_some_and(|c| c > SLOW_WRITE_WATCH_NS)).then_some("This Wine makes loading very slow. Proton 9+ or Wine 10.10+ on Linux 6.7+ fixes it.")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn windows_writes_are_cheap() {
        // On Windows (this test's host) there is no Wine and the cost is far below the mark.
        assert!(detect().is_none());
        let c = write_watch_cost_ns().expect("write watching reports every write");
        eprintln!("write watch: {c:.0} ns a page after a reset");
        assert!(c < SLOW_WRITE_WATCH_NS, "{c} ns");
        assert!(slow_notice(None, Some(c)).is_none());
        let w = Wine { version: "9.0".into(), build: Some("wine-9.0".into()), host: Some("Linux 6.5.0".into()) };
        assert_eq!(w.describe(), "Wine 9.0 (build wine-9.0) on Linux 6.5.0");
        assert!(slow_notice(Some(&w), Some(9000.0)).is_some());
        assert!(slow_notice(Some(&w), Some(300.0)).is_none());
    }
}
