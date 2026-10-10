//! High-level emulation registry and helpers (PS5 system libraries).
//!
//! HLE functions are Rust `extern "sysv64"` functions called directly by guest
//! code. They are registered under (library name, NID). Imports that resolve to
//! nothing get a trap stub that reports the call; by default that is fatal so
//! missing behavior is never hidden.

pub mod iostat;
pub mod kernel;
pub mod libs;

use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

/// Defines HLE functions with the System V ABI and optional entry tracing.
#[macro_export]
macro_rules! sysv {
    ( $( $(#[$m:meta])* fn $name:ident ( $( $arg:ident : $ty:ty ),* $(,)? ) $( -> $ret:ty )? $body:block )* ) => {
        $(
            $(#[$m])*
            #[allow(non_snake_case, unused_variables, unused_unsafe, clippy::missing_safety_doc)]
            pub unsafe extern "sysv64" fn $name( $( $arg : $ty ),* ) $( -> $ret )? {
                if $crate::log::TRACE_HLE.load(std::sync::atomic::Ordering::Relaxed) {
                    $crate::hle::trace_call(stringify!($name), &[ $( $crate::hle::ArgFmt::fmt_arg(&$arg) ),* ]);
                }
                unsafe { $body }
            }
        )*
    };
}

/// Registers functions by identifier under a library name.
#[macro_export]
macro_rules! reg {
    ($r:expr, $lib:expr, [ $( $f:ident ),* $(,)? ]) => {
        $( $r.func($lib, stringify!($f), $f as *const () as usize as u64); )*
    };
}

pub trait ArgFmt {
    fn fmt_arg(&self) -> String;
}
macro_rules! argfmt_int { ($($t:ty),*) => { $( impl ArgFmt for $t { fn fmt_arg(&self) -> String { format!("{:#x}", self) } } )* } }
argfmt_int!(u8, u16, u32, u64, usize, i8, i16, i32, i64, isize);
impl ArgFmt for bool {
    fn fmt_arg(&self) -> String {
        self.to_string()
    }
}
impl ArgFmt for f32 {
    fn fmt_arg(&self) -> String {
        self.to_string()
    }
}
impl ArgFmt for f64 {
    fn fmt_arg(&self) -> String {
        self.to_string()
    }
}
impl<T> ArgFmt for *const T {
    fn fmt_arg(&self) -> String {
        format!("{:p}", *self)
    }
}
impl<T> ArgFmt for *mut T {
    fn fmt_arg(&self) -> String {
        format!("{:p}", *self)
    }
}

pub fn trace_call(name: &str, args: &[String]) {
    crate::log::write(crate::log::Level::Debug, format_args!("hle {}({})", name, args.join(", ")));
}

#[derive(Clone, Debug)]
pub struct Export {
    pub addr: u64,
    pub name: String,
    pub is_data: bool,
}

#[derive(Default)]
pub struct Registry {
    pub by_lib: HashMap<String, HashMap<String, Export>>,
}

impl Registry {
    pub fn func(&mut self, lib: &str, name: &str, addr: u64) {
        let nid = sce_elf::nid::nid_for_name(name);
        self.by_lib
            .entry(lib.to_string())
            .or_default()
            .insert(nid, Export { addr, name: name.to_string(), is_data: false });
    }
    pub fn func_nid(&mut self, lib: &str, nid: &str, label: &str, addr: u64) {
        self.by_lib
            .entry(lib.to_string())
            .or_default()
            .insert(nid.to_string(), Export { addr, name: label.to_string(), is_data: false });
    }
    pub fn data(&mut self, lib: &str, name: &str, addr: u64) {
        let nid = sce_elf::nid::nid_for_name(name);
        self.by_lib
            .entry(lib.to_string())
            .or_default()
            .insert(nid, Export { addr, name: name.to_string(), is_data: true });
    }

    /// Libraries exported by the same system module share one HLE table; PS5
    /// "*.native"/"*_native" library names alias their plain names.
    fn canonical(lib: &str) -> &str {
        match lib {
            "libScePosix" | "libkernel_unity" | "libkernel_cpumode_platform" | "libkernel_pre250mmap"
            | "libkernel_ps2emu" | "libkernel_exception" | "libSceCoredump" | "libkernel_sysc_se" => "libkernel",
            "libSceSaveData_native" => "libSceSaveData",
            "libSceSaveDataDialog.native" => "libSceSaveDataDialog",
            "libSceMsgDialog.native" => "libSceMsgDialog",
            other => other,
        }
    }

    pub fn lookup(&self, lib: &str, nid: &str) -> Option<&Export> {
        if let Some(e) = self.by_lib.get(lib).and_then(|m| m.get(nid)) {
            return Some(e);
        }
        let c = Self::canonical(lib);
        if c != lib {
            if let Some(e) = self.by_lib.get(c).and_then(|m| m.get(nid)) {
                return Some(e);
            }
        }
        // Functions present in both the kernel and POSIX tables.
        if c == "libkernel" {
            return self.by_lib.get("libScePosix").and_then(|m| m.get(nid));
        }
        None
    }
}

static REGISTRY: OnceLock<Registry> = OnceLock::new();

pub fn registry() -> &'static Registry {
    REGISTRY.get_or_init(|| {
        let mut r = Registry::default();
        kernel::register(&mut r);
        libs::register(&mut r);
        r
    })
}

/// NID -> name table for diagnostics (data/ps4-names.txt + data/ps5-names.txt).
pub fn nid_names() -> &'static HashMap<String, String> {
    static NAMES: OnceLock<HashMap<String, String>> = OnceLock::new();
    NAMES.get_or_init(|| {
        let mut m = HashMap::new();
        let lists = [include_str!("../../../../data/ps4-names.txt"), include_str!("../../../../data/ps5-names.txt")];
        for line in lists.iter().flat_map(|l| l.lines()) {
            let n = line.trim();
            if n.is_empty() || n.starts_with('#') || n.contains(' ') || n.ends_with('?') {
                continue;
            }
            m.insert(sce_elf::nid::nid_for_name(n), n.to_string());
        }
        m
    })
}

pub fn describe_nid(nid: &str) -> String {
    match nid_names().get(nid) {
        Some(n) => format!("{n} ({nid})"),
        None => nid.to_string(),
    }
}

// ---------------------------------------------------------------------------
// Unresolved imports

#[derive(Clone, Debug)]
pub struct Unresolved {
    pub nid: String,
    pub lib: String,
    pub importer: String,
    pub is_data: bool,
}

pub static UNRESOLVED: RwLock<Vec<Unresolved>> = RwLock::new(Vec::new());
pub static LENIENT_IMPORTS: AtomicBool = AtomicBool::new(false);

std::arch::global_asm!(
    ".globl ds_unresolved_entry",
    "ds_unresolved_entry:",
    "push rbp",
    "mov rbp, rsp",
    "sub rsp, 64",
    "and rsp, -16",
    "mov [rsp], rdi",
    "mov [rsp+8], rsi",
    "mov [rsp+16], rdx",
    "mov [rsp+24], rcx",
    "mov [rsp+32], r8",
    "mov [rsp+40], r9",
    "mov rdi, rax",
    "mov rsi, rsp",
    "mov rdx, [rbp+8]",
    "call {handler}",
    "mov rsp, rbp",
    "pop rbp",
    "ret",
    handler = sym unresolved_handler,
);

unsafe extern "C" {
    fn ds_unresolved_entry();
}

extern "sysv64" fn unresolved_handler(index: u64, args: *const u64, ret: u64) -> u64 {
    let info = UNRESOLVED.read().get(index as usize).cloned();
    let a = unsafe { std::slice::from_raw_parts(args, 6) };
    let (desc, lib, importer) = match &info {
        Some(u) => (describe_nid(&u.nid), u.lib.clone(), u.importer.clone()),
        None => (format!("#{index}"), "?".into(), "?".into()),
    };
    let caller = crate::loader::describe_addr(ret);
    if LENIENT_IMPORTS.load(Ordering::Relaxed) {
        crate::warn!(
            "UNIMPLEMENTED {desc} [{lib}] imported by {importer}, called from {caller}, args {:#x} {:#x} {:#x} {:#x} -> returning 0 (lenient)",
            a[0],
            a[1],
            a[2],
            a[3]
        );
        return 0;
    }
    crate::error!(
        "UNIMPLEMENTED {desc} [{lib}] imported by {importer}, called from {caller}, args {:#x} {:#x} {:#x} {:#x} {:#x} {:#x}",
        a[0],
        a[1],
        a[2],
        a[3],
        a[4],
        a[5]
    );
    crate::fatal_exit(3, "unimplemented import called");
}

/// Builds a 24-byte trap stub: `mov rax, index; jmp [rip+0]; dq ds_unresolved_entry`.
pub fn unresolved_stub_code(index: u64) -> [u8; 24] {
    let mut c = [0u8; 24];
    c[0] = 0x48;
    c[1] = 0xB8;
    c[2..10].copy_from_slice(&index.to_le_bytes());
    c[10] = 0xFF;
    c[11] = 0x25;
    c[12..16].copy_from_slice(&0u32.to_le_bytes());
    c[16..24].copy_from_slice(&(ds_unresolved_entry as usize as u64).to_le_bytes());
    c
}

// ---------------------------------------------------------------------------
// Calling guest code from host code

/// Calls a guest function with up to 6 integer arguments (System V ABI).
pub unsafe fn call_guest(addr: u64, args: &[u64]) -> u64 {
    unsafe {
        type F6 = unsafe extern "sysv64" fn(u64, u64, u64, u64, u64, u64) -> u64;
        let f: F6 = std::mem::transmute(addr as usize);
        let g = |i: usize| args.get(i).copied().unwrap_or(0);
        f(g(0), g(1), g(2), g(3), g(4), g(5))
    }
}

// ---------------------------------------------------------------------------
// Guest memory helpers

pub unsafe fn cstr<'a>(p: *const u8) -> &'a str {
    if p.is_null() {
        return "";
    }
    unsafe {
        let s = std::ffi::CStr::from_ptr(p as *const std::ffi::c_char);
        s.to_str().unwrap_or("<non-utf8>")
    }
}

pub unsafe fn write_cstr(dst: *mut u8, cap: usize, s: &str) {
    if dst.is_null() || cap == 0 {
        return;
    }
    let n = s.len().min(cap - 1);
    unsafe {
        std::ptr::copy_nonoverlapping(s.as_ptr(), dst, n);
        *dst.add(n) = 0;
    }
}

// ---------------------------------------------------------------------------
// Explicit fixed-result functions (documented offline/no-device behavior).

static STUB_SEEN: parking_lot::Mutex<Option<std::collections::HashSet<&'static str>>> = parking_lot::Mutex::new(None);

pub fn stub_log(name: &'static str, ret: i64) {
    let mut g = STUB_SEEN.lock();
    let set = g.get_or_insert_with(Default::default);
    if set.insert(name) {
        crate::info!("{name} -> {ret:#x} (fixed HLE result)");
    }
}

/// Defines functions that ignore their arguments and return a fixed value.
#[macro_export]
macro_rules! fixed {
    ( $( $name:ident => $ret:expr ),* $(,)? ) => {
        $(
            #[allow(non_snake_case)]
            pub extern "sysv64" fn $name() -> i64 {
                let r = ($ret) as i64;
                $crate::hle::stub_log(stringify!($name), r);
                r
            }
        )*
    };
}
