//! libSceFiber: user-mode fibers on title-provided stacks.
//!
//! A fiber runs on the host thread that runs or switches to it, sharing that
//! thread's guest TLS (as on the console). Context switches save the System V
//! callee-saved registers, MXCSR and the x87 control word on the outgoing
//! stack and swap stack pointers.
//!
//! Signatures confirmed at the title's call sites: `_sceFiberInitializeImpl(
//! fiber, name, entry, argOnInitialize, addrContext, sizeContext, optParam,
//! sdkVersion)`, `sceFiberSwitch(fiber, argOnRunTo, &argOnRun)`,
//! `sceFiberReturnToThread(argOnReturn, &argOnRun)`; `SceFiber` is 0x100 bytes.

use crate::hle::{Registry, cstr};
use crate::{reg, sysv};
use parking_lot::Mutex;
use std::cell::Cell;
use std::collections::HashMap;

const FIBER_MAGIC: u64 = 0x5245_4249_4653_4400; // "\0DSFIBER"
const ERR_NULL: i32 = 0x8059_0001u32 as i32;
const ERR_ALIGN: i32 = 0x8059_0002u32 as i32;
const ERR_RANGE: i32 = 0x8059_0003u32 as i32;
const ERR_INVALID: i32 = 0x8059_0004u32 as i32;
const ERR_PERMISSION: i32 = 0x8059_0005u32 as i32;
const ERR_STATE: i32 = 0x8059_0006u32 as i32;

std::arch::global_asm!(
    // void ds_fiber_switch(u64* save_rsp [rdi], u64 new_rsp [rsi])  [sysv64]
    ".globl ds_fiber_switch",
    "ds_fiber_switch:",
    "push rbp",
    "push rbx",
    "push r12",
    "push r13",
    "push r14",
    "push r15",
    "sub rsp, 8",
    "stmxcsr [rsp]",
    "fnstcw [rsp+4]",
    "mov [rdi], rsp",
    "mov rsp, rsi",
    "ldmxcsr [rsp]",
    "fldcw [rsp+4]",
    "add rsp, 8",
    "pop r15",
    "pop r14",
    "pop r13",
    "pop r12",
    "pop rbx",
    "pop rbp",
    "ret",
    // First activation of a fiber: r12 = entry, r13 = argOnInitialize, r14 = argOnRun.
    ".globl ds_fiber_start",
    "ds_fiber_start:",
    "mov rdi, r13",
    "mov rsi, r14",
    "call r12",
    "call {returned}",
    "ud2",
    returned = sym fiber_entry_returned,
);

unsafe extern "sysv64" {
    fn ds_fiber_switch(save_rsp: *mut u64, new_rsp: u64);
    fn ds_fiber_start();
}

extern "sysv64" fn fiber_entry_returned() -> ! {
    let f = CURRENT.with(|c| c.get());
    crate::error!("fiber {f:#x} returned from its entry function (not allowed)");
    crate::fatal_exit(0xD520, "fiber entry returned");
}

/// Saved stack pointer of a suspended fiber, kept in the title's SceFiber
/// object (stable address even while other threads create fibers).
const OBJ_RSP: u64 = 0x08;

#[derive(Clone)]
struct Fiber {
    name: String,
    entry: u64,
    arg_init: u64,
    ctx: u64,
    ctx_size: u64,
    started: bool,
    running: bool,
    /// Where the value passed by the next Run/Switch to this fiber goes.
    arg_on_run_ptr: u64,
}

static FIBERS: Mutex<Option<HashMap<u64, Fiber>>> = Mutex::new(None);

thread_local! {
    /// Fiber currently running on this host thread (0 = the thread itself).
    static CURRENT: Cell<u64> = const { Cell::new(0) };
    /// Thread stack pointer saved by sceFiberRun while its fibers run.
    static RUN_RSP: Cell<u64> = const { Cell::new(0) };
    /// sceFiberRun's argOnReturn pointer.
    static RUN_RESULT_PTR: Cell<u64> = const { Cell::new(0) };
}

fn with<R>(f: impl FnOnce(&mut HashMap<u64, Fiber>) -> R) -> R {
    f(FIBERS.lock().get_or_insert_with(HashMap::new))
}

fn valid(fiber: u64) -> bool {
    fiber != 0 && unsafe { *(fiber as *const u64) } == FIBER_MAGIC && with(|m| m.contains_key(&fiber))
}

/// Prepares the initial stack of a fiber: ds_fiber_switch "returns" into
/// ds_fiber_start with the entry and its first argument in callee-saved regs.
/// Returns the initial saved stack pointer.
unsafe fn prepare_stack(f: &Fiber) -> u64 {
    let top = (f.ctx + f.ctx_size) & !15;
    let x = top - 8; // return slot; rsp == top (16-aligned) after `ret`
    unsafe {
        let w = |a: u64, v: u64| std::ptr::write(a as *mut u64, v);
        w(x, ds_fiber_start as *const () as u64);
        w(x - 8, 0); // rbp
        w(x - 16, 0); // rbx
        w(x - 24, f.entry); // r12
        w(x - 32, f.arg_init); // r13
        w(x - 40, 0); // r14 = argOnRun (filled on first activation)
        w(x - 48, 0); // r15
        w(x - 56, 0x1F80 | (0x037F << 32)); // MXCSR | x87 control word
    }
    x - 56
}

/// Delivers `arg` to a fiber about to run: as its entry argument the first time,
/// otherwise through the pointer of the call that suspended it.
unsafe fn deliver(fiber: u64, f: &mut Fiber, arg: u64) {
    unsafe {
        if !f.started {
            let rsp = *((fiber + OBJ_RSP) as *const u64);
            std::ptr::write((rsp + 16) as *mut u64, arg); // r14 slot (x - 40 == rsp + 16)
            f.started = true;
        } else if f.arg_on_run_ptr != 0 {
            std::ptr::write(f.arg_on_run_ptr as *mut u64, arg);
        }
    }
}

sysv! {
    fn sceFiberOptParamInitialize(p: *mut u8) -> i32 {
        if p.is_null() { return ERR_NULL; }
        std::ptr::write_bytes(p, 0, 0x80);
        0
    }
    fn _sceFiberInitializeImpl(fiber: u64, name: *const u8, entry: u64, arg_init: u64, ctx: u64, ctx_size: u64, opt: u64, sdk: u32) -> i32 {
        if fiber == 0 || entry == 0 { return ERR_NULL; }
        if fiber % 8 != 0 || ctx % 16 != 0 { return ERR_ALIGN; }
        if ctx == 0 || ctx_size < 512 { return ERR_RANGE; }
        let f = Fiber {
            name: cstr(name).chars().take(31).collect(),
            entry, arg_init, ctx, ctx_size, started: false, running: false, arg_on_run_ptr: 0,
        };
        let rsp = prepare_stack(&f);
        std::ptr::write_bytes(fiber as *mut u8, 0, 0x100);
        *(fiber as *mut u64) = FIBER_MAGIC;
        *((fiber + OBJ_RSP) as *mut u64) = rsp;
        crate::debug!("fiber {fiber:#x} '{}' entry {} ctx {ctx:#x}+{ctx_size:#x}", f.name, crate::loader::describe_addr(entry));
        with(|m| m.insert(fiber, f));
        0
    }
    fn sceFiberFinalize(fiber: u64) -> i32 {
        if fiber == 0 { return ERR_NULL; }
        if !valid(fiber) { return ERR_INVALID; }
        if with(|m| m.get(&fiber).is_some_and(|f| f.running)) { return ERR_STATE; }
        with(|m| m.remove(&fiber));
        *(fiber as *mut u64) = 0;
        0
    }
    fn sceFiberRun(fiber: u64, arg: u64, ret_out: *mut u64) -> i32 {
        if fiber == 0 { return ERR_NULL; }
        if !valid(fiber) { return ERR_INVALID; }
        if CURRENT.with(|c| c.get()) != 0 { return ERR_PERMISSION; }
        let ok = with(|m| {
            let f = m.get_mut(&fiber).unwrap();
            if f.running { return false; }
            f.running = true;
            unsafe { deliver(fiber, f, arg) };
            true
        });
        if !ok { return ERR_STATE; }
        CURRENT.with(|c| c.set(fiber));
        RUN_RESULT_PTR.with(|c| c.set(ret_out as u64));
        ds_fiber_switch(RUN_RSP.with(|c| c.as_ptr()), *((fiber + OBJ_RSP) as *const u64));
        // Back on the thread: a fiber called sceFiberReturnToThread.
        0
    }
    fn sceFiberSwitch(fiber: u64, arg: u64, arg_on_run: *mut u64) -> i32 {
        if fiber == 0 { return ERR_NULL; }
        if !valid(fiber) { return ERR_INVALID; }
        let cur = CURRENT.with(|c| c.get());
        if cur == 0 { return ERR_PERMISSION; }
        let ok = with(|m| {
            if m.get(&fiber).is_none_or(|f| f.running) { return false; }
            let f = m.get_mut(&fiber).unwrap();
            f.running = true;
            unsafe { deliver(fiber, f, arg) };
            let c = m.get_mut(&cur).unwrap();
            c.running = false;
            c.arg_on_run_ptr = arg_on_run as u64;
            true
        });
        if !ok { return ERR_STATE; }
        CURRENT.with(|c| c.set(fiber));
        ds_fiber_switch((cur + OBJ_RSP) as *mut u64, *((fiber + OBJ_RSP) as *const u64));
        0
    }
    fn sceFiberReturnToThread(ret: u64, arg_on_run: *mut u64) -> i32 {
        let cur = CURRENT.with(|c| c.get());
        if cur == 0 { return ERR_PERMISSION; }
        let p = RUN_RESULT_PTR.with(|c| c.get());
        if p != 0 { *(p as *mut u64) = ret; }
        with(|m| {
            let c = m.get_mut(&cur).unwrap();
            c.running = false;
            c.arg_on_run_ptr = arg_on_run as u64;
        });
        CURRENT.with(|c| c.set(0));
        ds_fiber_switch((cur + OBJ_RSP) as *mut u64, RUN_RSP.with(|c| c.get()));
        0
    }
    fn sceFiberGetSelf(out: *mut u64) -> i32 {
        if out.is_null() { return ERR_NULL; }
        let cur = CURRENT.with(|c| c.get());
        if cur == 0 { return ERR_PERMISSION; }
        *out = cur;
        0
    }
    fn sceFiberGetInfo(fiber: u64, info: *mut u64) -> i32 {
        if fiber == 0 || info.is_null() { return ERR_NULL; }
        let Some(f) = with(|m| m.get(&fiber).cloned()) else { return ERR_INVALID };
        // SceFiberInfo: size, entry, argOnInitialize, addrContext, sizeContext,
        // name[32], sizeContextMargin.
        *info.add(1) = f.entry;
        *info.add(2) = f.arg_init;
        *info.add(3) = f.ctx;
        *info.add(4) = f.ctx_size;
        crate::hle::write_cstr(info.add(5) as *mut u8, 32, &f.name);
        // Margin (untouched stack) is not tracked: report the whole context.
        *info.add(9) = f.ctx_size;
        0
    }
    fn sceFiberRename(fiber: u64, name: *const u8) -> i32 {
        if fiber == 0 || name.is_null() { return ERR_NULL; }
        let n: String = cstr(name).chars().take(31).collect();
        if with(|m| m.get_mut(&fiber).map(|f| f.name = n)).is_none() { return ERR_INVALID; }
        0
    }
}

pub fn register(r: &mut Registry) {
    reg!(r, "libSceFiber", [
        sceFiberOptParamInitialize, _sceFiberInitializeImpl, sceFiberFinalize, sceFiberRun, sceFiberSwitch,
        sceFiberReturnToThread, sceFiberGetSelf, sceFiberGetInfo, sceFiberRename,
    ]);
}
