//! Guest threads: each guest thread is a host thread running guest code natively
//! on its own stack, with its TLS slot pointing at the guest thread pointer.

use parking_lot::{Condvar, Mutex};
use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU64, Ordering};

std::arch::global_asm!(
    // u64 ds_guest_thread_call(entry, a0, a1, *mut saved_rsp)  [sysv64]
    ".globl ds_guest_thread_call",
    "ds_guest_thread_call:",
    "push rbp",
    "mov rbp, rsp",
    "push rbx",
    "push r12",
    "push r13",
    "push r14",
    "push r15",
    "sub rsp, 8",
    "mov [rcx], rsp",
    "mov rax, rdi",
    "mov rdi, rsi",
    "mov rsi, rdx",
    "call rax",
    "ds_guest_thread_resume:",
    "add rsp, 8",
    "pop r15",
    "pop r14",
    "pop r13",
    "pop r12",
    "pop rbx",
    "pop rbp",
    "ret",
    // ! ds_guest_thread_exit(saved_rsp, value)  [sysv64]
    ".globl ds_guest_thread_exit",
    "ds_guest_thread_exit:",
    "mov rsp, rdi",
    "mov rax, rsi",
    "jmp ds_guest_thread_resume",
);

unsafe extern "sysv64" {
    fn ds_guest_thread_call(entry: u64, a0: u64, a1: u64, saved_rsp: *mut u64) -> u64;
    fn ds_guest_thread_exit(saved_rsp: u64, value: u64) -> !;
}

pub const THREAD_MAGIC: u64 = 0x4453_5448_5245_4144; // "DSTHREAD"
pub const PTHREAD_KEYS_MAX: usize = 256;

pub struct GuestThread {
    pub magic: u64,
    pub id: u64,
    pub name: Mutex<String>,
    pub tp: AtomicU64,
    pub entry: u64,
    pub arg: u64,
    pub stack_size: u64,
    pub detached: AtomicBool,
    pub exited: Mutex<Option<u64>>,
    pub exit_cv: Condvar,
    pub saved_rsp: AtomicU64,
    pub errno: AtomicI32,
    pub priority: AtomicI32,
    pub affinity: AtomicU64,
    pub specific: Mutex<Vec<u64>>,
    pub host_tid: AtomicU64,
}

impl GuestThread {
    pub fn handle(&'static self) -> u64 {
        self as *const _ as u64
    }
    pub fn name(&self) -> String {
        self.name.lock().clone()
    }
}

static NEXT_ID: AtomicU64 = AtomicU64::new(1);
pub static THREADS: Mutex<Vec<&'static GuestThread>> = Mutex::new(Vec::new());

thread_local! {
    static CURRENT: Cell<Option<&'static GuestThread>> = const { Cell::new(None) };
}

pub fn current() -> Option<&'static GuestThread> {
    CURRENT.with(|c| c.get())
}

/// Validates a guest-supplied thread handle.
pub fn from_handle(h: u64) -> Option<&'static GuestThread> {
    if h == 0 {
        return None;
    }
    let threads = THREADS.lock();
    threads.iter().copied().find(|t| t.handle() == h && t.magic == THREAD_MAGIC)
}

pub fn new_thread(name: &str, entry: u64, arg: u64, stack_size: u64, priority: i32) -> &'static GuestThread {
    let t: &'static GuestThread = Box::leak(Box::new(GuestThread {
        magic: THREAD_MAGIC,
        id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
        name: Mutex::new(name.to_string()),
        tp: AtomicU64::new(0),
        entry,
        arg,
        stack_size,
        detached: AtomicBool::new(false),
        exited: Mutex::new(None),
        exit_cv: Condvar::new(),
        saved_rsp: AtomicU64::new(0),
        errno: AtomicI32::new(0),
        priority: AtomicI32::new(priority),
        affinity: AtomicU64::new(0x7F),
        specific: Mutex::new(vec![0; PTHREAD_KEYS_MAX]),
        host_tid: AtomicU64::new(0),
    }));
    let tp = crate::tls::alloc_thread_area(t.handle());
    t.tp.store(tp, Ordering::SeqCst);
    THREADS.lock().push(t);
    t
}

/// Binds the calling host thread to a guest thread (TLS slot + current pointer).
pub fn enter(t: &'static GuestThread) {
    CURRENT.with(|c| c.set(Some(t)));
    crate::tls::set_current_tp(t.tp.load(Ordering::SeqCst));
    crate::log::set_thread_name(&t.name());
    t.host_tid.store(unsafe { windows_sys::Win32::System::Threading::GetCurrentThreadId() } as u64, Ordering::SeqCst);
}

/// Runs guest code `entry(a0, a1)` on the current (already entered) thread.
/// Returns when the function returns or the thread calls pthread_exit.
pub fn run_entry(t: &'static GuestThread, entry: u64, a0: u64, a1: u64) -> u64 {
    unsafe { ds_guest_thread_call(entry, a0, a1, t.saved_rsp.as_ptr()) }
}

/// Terminates the current guest thread with `value` (pthread_exit semantics).
pub fn exit_current(value: u64) -> ! {
    let t = current().expect("pthread_exit on a non-guest thread");
    run_key_destructors(t);
    let rsp = t.saved_rsp.load(Ordering::SeqCst);
    unsafe { ds_guest_thread_exit(rsp, value) }
}

fn finish(t: &'static GuestThread, value: u64) {
    *t.exited.lock() = Some(value);
    t.exit_cv.notify_all();
    crate::debug!("thread '{}' exited with {:#x}", t.name(), value);
}

pub fn run_key_destructors(t: &'static GuestThread) {
    // POSIX: up to PTHREAD_DESTRUCTOR_ITERATIONS rounds.
    for _ in 0..4 {
        let mut any = false;
        for key in 0..PTHREAD_KEYS_MAX {
            let val = {
                let mut s = t.specific.lock();
                std::mem::replace(&mut s[key], 0)
            };
            if val == 0 {
                continue;
            }
            if let Some(dtor) = crate::hle::kernel::pthread::key_destructor(key) {
                any = true;
                unsafe {
                    crate::hle::call_guest(dtor, &[val]);
                }
            }
        }
        if !any {
            break;
        }
    }
}

/// Creates a host thread whose whole stack is committed up front. PS4 code does
/// not probe the stack, so lazily grown (guard-page) Windows stacks would fault
/// when a large frame is touched below the guard page.
pub fn spawn_host(name: &str, stack_size: usize, f: Box<dyn FnOnce() + Send>) -> std::io::Result<()> {
    use windows_sys::Win32::System::Threading::{CreateThread, SetThreadDescription};
    unsafe extern "system" fn proc_(param: *mut std::ffi::c_void) -> u32 {
        let f = unsafe { Box::from_raw(param as *mut Box<dyn FnOnce() + Send>) };
        f();
        0
    }
    let stack = (stack_size + 0xFFFF) & !0xFFFF;
    let param = Box::into_raw(Box::new(f)) as *mut std::ffi::c_void;
    let mut tid = 0u32;
    let h = unsafe { CreateThread(std::ptr::null(), stack, Some(proc_), param, 0, &mut tid) };
    if h.is_null() {
        drop(unsafe { Box::from_raw(param as *mut Box<dyn FnOnce() + Send>) });
        return Err(std::io::Error::last_os_error());
    }
    let wname: Vec<u16> = name.encode_utf16().chain(std::iter::once(0)).collect();
    unsafe {
        SetThreadDescription(h, wname.as_ptr());
        windows_sys::Win32::Foundation::CloseHandle(h);
    }
    Ok(())
}

/// Starts a host thread that runs a guest thread function.
pub fn spawn(t: &'static GuestThread) -> std::io::Result<()> {
    let host_stack = (t.stack_size as usize).max(256 << 10) + (1 << 20);
    spawn_host(
        &t.name(),
        host_stack,
        Box::new(move || {
            enter(t);
            crate::debug!("thread '{}' start entry={:#x} arg={:#x}", t.name(), t.entry, t.arg);
            let v = run_entry(t, t.entry, t.arg, 0);
            run_key_destructors(t);
            finish(t, v);
        }),
    )
}

pub fn join(t: &'static GuestThread) -> u64 {
    let mut g = t.exited.lock();
    while g.is_none() {
        t.exit_cv.wait(&mut g);
    }
    g.unwrap()
}
