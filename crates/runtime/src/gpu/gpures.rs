//! GPU-resolved programs (RENDERER.md "GPU-resolved item passes"; `tune
//! gpu_chains`): batched compute programs whose commands in a run share some
//! descriptor chains and differ in the others. A chain's root is a pointer
//! the first level of the user data holds: user dword pair k itself (its own
//! loads), or the pointer read from it at byte offset `off`. Roots that hold
//! the same value in every command of a run are the run's roots: what they
//! reach is resolved by the CPU once per run. The other chains (per item)
//! are resolved by the shader through the guest page table
//! (`Spec::gpu_loads`, `gpu_res`), and a program qualifies only when every
//! resource a per-item chain reaches is a read-only buffer. Learned from the
//! commands the CPU path resolves.

use parking_lot::Mutex;
use rdna::analysis::{Plan, Prov, Ptr, ResKind, Resource};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

/// A chain's root: user dword pair `k` and the byte offset of the pointer
/// read from it (None: the user pointer's own loads).
pub type Root = (u16, Option<u32>);

/// Commands observed in runs before a program is decided.
const LEARN_CMDS: u32 = 256;
const LEARN_RUNS: u32 = 4;

fn root_of_ptr(p: &Ptr) -> Option<Root> {
    match p {
        Ptr::User(k) => Some((*k, None)),
        Ptr::Mem(inner, off) => match &**inner {
            Ptr::User(k) => Some((*k, Some(*off))),
            other => root_of_ptr(other),
        },
        Ptr::Pc(_) => None,
    }
}

pub fn root_of_res(r: &Resource) -> Option<Root> {
    match r.dwords.first()? {
        Prov::Mem(p, _) => root_of_ptr(p),
        Prov::User(k) => Some((*k, None)),
        _ => None,
    }
}

/// A decided program: the loads and resources resolved by the shader, and
/// the run roots the CPU compares per command.
#[derive(Clone, Debug)]
pub struct Decided {
    pub gpu_loads: u64,
    pub gpu_res: u64,
    pub run_roots: Vec<Root>,
    /// The per-item roots with an offset (their pointers' pages must be
    /// mapped: `ensure_mapped`).
    pub item_roots: Vec<Root>,
}

enum State {
    Learning { roots: Vec<Root>, varies: u64, cmds: u32, runs: u32 },
    Eligible(Decided),
    Refused,
}

static PROGRAMS: Mutex<Option<HashMap<u64, State>>> = Mutex::new(None);

thread_local! {
    /// Per program, the root values of this thread's last observed command.
    static LAST: std::cell::RefCell<HashMap<u64, Vec<u64>>> = std::cell::RefCell::new(HashMap::new());
}

/// Decided programs (fast lookup without the lock), and counts: programs
/// decided eligible, refused.
static DECIDED_N: AtomicU64 = AtomicU64::new(0);
static STATS: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// The pointer value of `root` for user data `user` (0 when unreadable).
pub fn root_value(root: Root, user: &[u32]) -> u64 {
    let k = root.0 as usize;
    let (Some(&lo), Some(&hi)) = (user.get(k), user.get(k + 1)) else { return 0 };
    let u = lo as u64 | ((hi as u64 & 0xFFFF) << 32);
    match root.1 {
        None => u,
        Some(off) => {
            let a = u.wrapping_add(off as u64);
            if !super::cp::guest_readable(a, 8) {
                return 0;
            }
            // SAFETY: readable guest memory (checked above).
            unsafe { std::ptr::read_unaligned(a as *const u64) & 0xFFFF_FFFF_FFFF }
        }
    }
}

/// The program's decision, if it is GPU-resolved (eligible).
pub fn decided(addr: u64) -> Option<Decided> {
    if DECIDED_N.load(Ordering::Relaxed) == 0 {
        return None;
    }
    match PROGRAMS.lock().as_ref()?.get(&addr)? {
        State::Eligible(d) => Some(d.clone()),
        _ => None,
    }
}

/// A command of batchable program `addr` resolved by the CPU path: its root
/// values against this thread's last command of the program (`continues`:
/// the same run).
pub fn observe(addr: u64, plan: &Plan, user: &[u32], continues: bool) {
    if super::tune::gpu_chains() == 0 {
        return;
    }
    let mut g = PROGRAMS.lock();
    let m = g.get_or_insert_with(HashMap::new);
    let st = m.entry(addr).or_insert_with(|| {
        let mut roots: Vec<Root> = Vec::new();
        for r in plan.loads.iter().filter_map(|l| root_of_ptr(&l.ptr)).chain(plan.resources.iter().filter_map(root_of_res)) {
            if r.1.is_some() && !roots.contains(&r) {
                roots.push(r);
            }
        }
        if plan.loads.len() > 64 || plan.resources.len() > 64 || roots.len() > 64 {
            return State::Refused;
        }
        State::Learning { roots, varies: 0, cmds: 0, runs: 0 }
    });
    let State::Learning { roots, varies, cmds, runs } = st else { return };
    let vals: Vec<u64> = roots.iter().map(|&r| root_value(r, user)).collect();
    let prev = LAST.with(|l| l.borrow_mut().insert(addr, vals.clone()));
    match prev {
        Some(p) if continues && p.len() == vals.len() => {
            for (i, (a, b)) in p.iter().zip(&vals).enumerate() {
                if a != b {
                    *varies |= 1 << i;
                }
            }
            *cmds += 1;
        }
        _ => *runs += 1,
    }
    if *cmds < LEARN_CMDS || *runs < LEARN_RUNS {
        return;
    }
    let run_roots: Vec<Root> = roots.iter().enumerate().filter(|(i, _)| *varies & (1 << i) == 0).map(|(_, &r)| r).collect();
    let item_roots: Vec<Root> = roots.iter().enumerate().filter(|(i, _)| *varies & (1 << i) != 0).map(|(_, &r)| r).collect();
    let per_item = |r: Option<Root>| match r {
        Some((_, None)) => true,
        Some(x) => !run_roots.contains(&x),
        None => false,
    };
    let mut gpu_loads = 0u64;
    for (i, l) in plan.loads.iter().enumerate() {
        if per_item(root_of_ptr(&l.ptr)) {
            gpu_loads |= 1 << i;
        }
    }
    let mut gpu_res = 0u64;
    let mut ok = true;
    for (i, r) in plan.resources.iter().enumerate() {
        if per_item(root_of_res(r)) {
            if r.kind != ResKind::Buffer || r.written {
                ok = false;
            }
            gpu_res |= 1 << i;
        }
    }
    *st = if ok && gpu_res != 0 {
        let d = Decided { gpu_loads, gpu_res, run_roots, item_roots };
        crate::info!("gpu: cs {addr:#x} resolves its per-item chains on the GPU from now on (loads {gpu_loads:#x}, buffers {gpu_res:#x}; run roots {:?}, per-item roots {:?})", d.run_roots, d.item_roots);
        STATS[0].fetch_add(1, Ordering::Relaxed);
        DECIDED_N.fetch_add(1, Ordering::Relaxed);
        State::Eligible(d)
    } else {
        crate::info!("gpu: cs {addr:#x} keeps CPU-resolved chains (a per-item chain reaches {}; buffers {gpu_res:#x})", if ok { "no buffer" } else { "an image, sampler or written buffer" });
        STATS[1].fetch_add(1, Ordering::Relaxed);
        State::Refused
    };
}

/// Status part ("" when off).
pub fn status() -> String {
    if super::tune::gpu_chains() == 0 {
        return String::new();
    }
    let a = &super::exec::GPU_APPENDS;
    let c = &super::exec::GPU_RES_CHECKS;
    format!(
        "    GPU-resolved programs: {} eligible, {} refused; commands appended to their runs {}, recorded in full: run roots differ {}, no open run or other state {}; ranges imported for them {}; reads of those in full checked {}: in a home {}, pending writes {}, in a resident shadow {}, not imported {}\n",
        STATS[0].load(Ordering::Relaxed),
        STATS[1].load(Ordering::Relaxed),
        a[0].load(Ordering::Relaxed),
        a[1].load(Ordering::Relaxed),
        a[2].load(Ordering::Relaxed),
        a[3].load(Ordering::Relaxed),
        c[0].load(Ordering::Relaxed),
        c[1].load(Ordering::Relaxed),
        c[2].load(Ordering::Relaxed),
        c[3].load(Ordering::Relaxed),
        c[4].load(Ordering::Relaxed)
    )
}
