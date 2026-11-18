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

/// A pointer's root (None: the program's own data).
pub fn root_of_ptr(p: &Ptr) -> Option<Root> {
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
        Prov::Mem(p, _) | Prov::PtrLo(p, _) | Prov::PtrHi(p, _) => root_of_ptr(p),
        Prov::User(k) => Some((*k, None)),
        _ => None,
    }
}

/// The roots of all of a resource's descriptor dwords (a descriptor may mix
/// words read from memory, words of the user data and pointer arithmetic on
/// either); words from constants or the program's own data add none. None:
/// a word comes from a table entry or from nothing the analysis followed.
pub fn res_roots(r: &Resource) -> Option<Vec<Root>> {
    let mut v = Vec::new();
    for p in &r.dwords {
        let root = match p {
            Prov::Const(_) => continue,
            Prov::User(k) => (*k, None),
            Prov::Mem(ptr, _) | Prov::PtrLo(ptr, _) | Prov::PtrHi(ptr, _) => match root_of_ptr(ptr) {
                Some(x) => x,
                None => continue,
            },
            Prov::Table(..) | Prov::Unknown => return None,
        };
        if !v.contains(&root) {
            v.push(root);
        }
    }
    Some(v)
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
    /// User-data dwords that descriptors the CPU resolves for the whole run
    /// take as they are: compared word by word per command (draws).
    pub run_user: u32,
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
    if !matches!(super::tune::gpu_chains(), 1 | 2) {
        return;
    }
    let mut g = PROGRAMS.lock();
    let m = g.get_or_insert_with(HashMap::new);
    let st = m.entry(addr).or_insert_with(|| {
        let mut roots: Vec<Root> = Vec::new();
        let mut res = Vec::new();
        for r in plan.resources.iter() {
            match res_roots(r) {
                Some(v) => res.extend(v),
                None => return State::Refused,
            }
        }
        for r in plan.loads.iter().filter_map(|l| root_of_ptr(&l.ptr)).chain(res) {
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
        // Per item when any of its words comes through a per-item chain.
        if res_roots(r).unwrap_or_default().into_iter().any(|x| per_item(Some(x))) {
            if r.kind != ResKind::Buffer || r.written {
                ok = false;
            }
            gpu_res |= 1 << i;
        }
    }
    *st = if ok && gpu_res != 0 {
        let d = Decided { gpu_loads, gpu_res, run_roots, item_roots, run_user: 0 };
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

/// GPU-resolved draws (stage G4): program pairs (vertex, pixel) whose draws
/// in a run (right after a draw of the pair with nothing recorded between
/// and the same context state) share some chains of each stage and differ
/// in others, learned as batched programs are. Unlike a batched program's
/// single item pointer, a draw's user data holds several pointers, some
/// shared by the run: a user pointer is a root of its own (k, None), and
/// loads from it are per-item only when it varies. A pair qualifies when the
/// per-item chains of both stages reach only read-only buffers.
#[derive(Clone, Debug)]
pub struct DrawDecided {
    /// Vertex stage, pixel stage.
    pub stages: [Decided; 2],
}

enum DrawState {
    Learning { roots: [Vec<Root>; 2], varies: [u64; 2], cmds: u32, runs: u32 },
    Eligible(std::sync::Arc<DrawDecided>),
    Refused,
}

/// Read by every draw while any pair is decided (`draw_decided`).
static PAIRS: parking_lot::RwLock<Option<HashMap<(u64, u64), DrawState>>> = parking_lot::RwLock::new(None);
/// Pairs decided eligible (fast lookup without the lock), and counts: pairs
/// decided eligible, refused when learned, refused later (a run's first
/// draw found per-item data elsewhere than guest memory).
static DRAW_DECIDED_N: AtomicU64 = AtomicU64::new(0);
pub static DRAW_STATS: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

thread_local! {
    /// Per pair, the root values of this thread's last observed draw.
    static LAST_DRAW: std::cell::RefCell<HashMap<(u64, u64), Vec<u64>>> = std::cell::RefCell::new(HashMap::new());
    /// Pairs this thread found decided (learning over).
    static SETTLED: std::cell::RefCell<std::collections::HashSet<(u64, u64)>> = std::cell::RefCell::new(std::collections::HashSet::new());
}

/// A draw stage's roots: user pointers and the pointers read from them
/// (None: a descriptor word whose origin the analysis did not follow).
fn draw_roots(plan: &Plan) -> Option<Vec<Root>> {
    let mut res = Vec::new();
    for r in plan.resources.iter() {
        res.extend(res_roots(r)?);
    }
    let mut roots: Vec<Root> = Vec::new();
    for r in plan.loads.iter().filter_map(|l| root_of_ptr(&l.ptr)).chain(res) {
        for x in [r, (r.0, None)] {
            if !roots.contains(&x) {
                roots.push(x);
            }
        }
    }
    (plan.loads.len() <= 64 && plan.resources.len() <= 64 && roots.len() <= 64).then_some(roots)
}

/// A stage's decision from the roots that vary within runs: None when a
/// per-item chain reaches anything but a read-only buffer. A resource is per
/// item when any of its words comes through a varying root; the run's
/// resources that take user-data words as they are name those words
/// (`Decided::run_user`).
fn draw_stage(plan: &Plan, roots: &[Root], varies: u64) -> Option<Decided> {
    let run_roots: Vec<Root> = roots.iter().enumerate().filter(|(i, _)| varies & (1 << i) == 0).map(|(_, &r)| r).collect();
    let item_roots: Vec<Root> = roots.iter().enumerate().filter(|(i, _)| varies & (1 << i) != 0).map(|(_, &r)| r).collect();
    let per_item = |r: Option<Root>| r.is_some_and(|x| item_roots.contains(&x));
    let mut gpu_loads = 0u64;
    for (i, l) in plan.loads.iter().enumerate() {
        if per_item(root_of_ptr(&l.ptr)) {
            gpu_loads |= 1 << i;
        }
    }
    let mut gpu_res = 0u64;
    let mut run_user = 0u32;
    for (i, r) in plan.resources.iter().enumerate() {
        if res_roots(r)?.into_iter().any(|x| per_item(Some(x))) {
            if r.kind != ResKind::Buffer || r.written {
                return None;
            }
            gpu_res |= 1 << i;
        } else {
            for p in &r.dwords {
                if let Prov::User(k) = p {
                    run_user |= 1u32 << (*k).min(31);
                }
            }
        }
    }
    Some(Decided { gpu_loads, gpu_res, run_roots, item_roots, run_user })
}

/// `DS_RUN_CENSUS=1` (a census, ROUTE60.md 2026-11-18 morning): the pairs learned as with
/// `tune gpu_chains` 1 but never drawn GPU-resolved; each prepared draw priced as an append to
/// its pair's run (`Exec::run_census_note`).
pub fn census_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_RUN_CENSUS").ok().as_deref() == Some("1"))
}

/// `DS_RUN_CENSUS`: [prepared draws with a pixel stage, of a pair not (yet) eligible, of an
/// eligible pair; of those: would append (the run of the pair goes on with the run roots and
/// run user words of its first draw, the per-item buffers' formats too), would append with
/// run-time formats (a per-item buffer's format alone differs), opens a run; chain walks an
/// append makes (timed), their cycles, the ranges they reach, of those current in the mirror
/// as its state stands (front end 2's quick check over at most 16 pages), loads among the
/// ranges, walks that failed; ranges not current by place: inside the GPU arena's live frame
/// allocations, in an arena MiB above its high-water mark, elsewhere; current ranges inside
/// the live frame allocations].
pub static RUN_CENSUS: [AtomicU64; 16] = [const { AtomicU64::new(0) }; 16];
/// `DS_RUN_CENSUS`: the not-current ranges outside the GPU arena by the engine allocator owning
/// them: [ranges, of them on pages front end 2 found CPU-written within the last 4 flips].
pub static RUN_CENSUS_OWNER: [[AtomicU64; 2]; 256] = [const { [const { AtomicU64::new(0) }; 2] }; 256];
/// `DS_RUN_CENSUS`: whether the not-current ranges' contents change between flips (the first
/// 64 bytes hashed per address): [outside the arena: checked, seen in an earlier flip, changed
/// since; inside the arena's live frame allocations: the same three].
pub static RUN_STABLE: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];
static RUN_SEEN: [parking_lot::Mutex<Option<HashMap<u64, (u64, u64)>>>; 64] = [const { parking_lot::Mutex::new(None) }; 64];

/// `DS_RUN_CENSUS`: notes the contents of a not-current range an append reaches (`arena`: inside
/// the GPU arena's live frame allocations) against the last flip that reached it.
pub fn census_stable(a: u64, n: u64, arena: bool) {
    let len = n.min(64) as usize;
    if len == 0 || !crate::mem::readable(a, len as u64) {
        return;
    }
    let bytes = unsafe { std::slice::from_raw_parts(a as *const u8, len) };
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        h = (h ^ b as u64).wrapping_mul(0x100_0000_01b3);
    }
    let flip = super::FRAMES.load(Ordering::Relaxed);
    let k = if arena { 3 } else { 0 };
    RUN_STABLE[k].fetch_add(1, Ordering::Relaxed);
    let mut g = RUN_SEEN[((a >> 6) as usize) & 63].lock();
    if let Some((h0, f0)) = g.get_or_insert_with(HashMap::new).insert(a, (h, flip)) {
        if f0 != flip {
            RUN_STABLE[k + 1].fetch_add(1, Ordering::Relaxed);
            if h0 != h {
                RUN_STABLE[k + 2].fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

/// The census line ("" when off).
pub fn census_line() -> String {
    if !census_on() {
        return String::new();
    }
    let c: Vec<u64> = RUN_CENSUS.iter().map(|x| x.load(Ordering::Relaxed)).collect();
    if c[0] == 0 {
        return String::new();
    }
    let hz = super::threadcpu::tsc_hz();
    let walk = if c[6] == 0 { 0.0 } else { c[7] as f64 * 1e9 / hz / c[6] as f64 };
    let per = |x: u64| if c[6] == 0 { 0.0 } else { x as f64 / c[6] as f64 };
    format!(
        "    GPU-resolved runs, a census (`DS_RUN_CENSUS`): prepared draws {}, of pairs not eligible {}, eligible {}: would append {} ({:.1}% of the draws), with run-time formats {} more, opening a run {}; an append's chain walk {walk:.0} ns ({} timed, {} failed), reaching {:.2} ranges ({:.2} loads), {:.2} current in the mirror as it stands ({:.2} of them in the GPU arena's live frame allocations); not current: {:.2} in those allocations, {:.2} in an arena MiB above its high-water mark, {:.2} elsewhere\n",
        c[0], c[1], c[2], c[3], 100.0 * c[3] as f64 / c[0] as f64, c[4], c[5], c[6], c[11], per(c[8]), per(c[10]), per(c[9]), per(c[15]), per(c[12]), per(c[13]), per(c[14])
    ) + &{
        let names: std::collections::HashMap<u8, String> = crate::engine::allocators().into_iter().map(|a| (a.id, a.name)).collect();
        let mut v: Vec<(usize, u64, u64)> = RUN_CENSUS_OWNER.iter().enumerate().map(|(i, x)| (i, x[0].load(Ordering::Relaxed), x[1].load(Ordering::Relaxed))).filter(|x| x.1 > 0).collect();
        v.sort_by(|a, b| b.1.cmp(&a.1));
        let parts: Vec<String> = v.iter().take(8).map(|&(i, n, h)| format!("{} ({i}) {:.2} ({:.2} written within 4 flips)", names.get(&(i as u8)).map_or("?", |s| s.as_str()), per(n), per(h))).collect();
        let s: Vec<u64> = RUN_STABLE.iter().map(|x| x.load(Ordering::Relaxed)).collect();
        format!(
            "      not current elsewhere, by owner (an append's): {}\n      their contents (the first 64 bytes) against the last flip that reached the same address: outside the arena {} checked, {} seen in an earlier flip, {} of those changed; in the arena's live frame allocations {} checked, {} seen in an earlier flip, {} changed\n",
            parts.join(", "),
            s[0], s[1], s[2], s[3], s[4], s[5]
        )
    }
}

/// The pair's decision, if its draws are GPU-resolved (eligible).
pub fn draw_decided(vs: u64, ps: u64) -> Option<std::sync::Arc<DrawDecided>> {
    if DRAW_DECIDED_N.load(Ordering::Relaxed) == 0 {
        return None;
    }
    match PAIRS.read().as_ref()?.get(&(vs, ps))? {
        DrawState::Eligible(d) => Some(d.clone()),
        _ => None,
    }
}

/// A draw of pair (vs, ps) resolved by the CPU path, its stages' plans and
/// user data (vertex first): its root values against this thread's last
/// draw of the pair (`continues`: the same run).
pub fn observe_draw(pair: (u64, u64), stages: [(&Plan, &[u32]); 2], continues: bool) {
    if (super::tune::gpu_chains() == 0 && !census_on()) || SETTLED.with(|s| s.borrow().contains(&pair)) {
        return;
    }
    let mut g = PAIRS.write();
    let m = g.get_or_insert_with(HashMap::new);
    let st = m.entry(pair).or_insert_with(|| match (draw_roots(stages[0].0), draw_roots(stages[1].0)) {
        (Some(a), Some(b)) => DrawState::Learning { roots: [a, b], varies: [0, 0], cmds: 0, runs: 0 },
        _ => DrawState::Refused,
    });
    let DrawState::Learning { roots, varies, cmds, runs } = st else {
        SETTLED.with(|s| s.borrow_mut().insert(pair));
        return;
    };
    let vals: Vec<u64> = (0..2).flat_map(|k| roots[k].iter().map(move |&r| root_value(r, stages[k].1))).collect();
    let prev = LAST_DRAW.with(|l| l.borrow_mut().insert(pair, vals.clone()));
    match prev {
        Some(p) if continues && p.len() == vals.len() => {
            let n0 = roots[0].len();
            for (i, (a, b)) in p.iter().zip(&vals).enumerate() {
                if a != b {
                    if i < n0 {
                        varies[0] |= 1 << i;
                    } else {
                        varies[1] |= 1 << (i - n0);
                    }
                }
            }
            *cmds += 1;
        }
        _ => *runs += 1,
    }
    if *cmds < LEARN_CMDS || *runs < LEARN_RUNS {
        return;
    }
    let (vs, ps) = pair;
    *st = match (draw_stage(stages[0].0, &roots[0], varies[0]), draw_stage(stages[1].0, &roots[1], varies[1])) {
        (Some(a), Some(b)) if a.gpu_res | a.gpu_loads | b.gpu_res | b.gpu_loads != 0 => {
            crate::info!(
                "gpu: draws of vs {vs:#x} ps {ps:#x} resolve their per-item chains on the GPU from now on (vs loads {:#x} buffers {:#x}, ps loads {:#x} buffers {:#x}; per-item roots vs {:?} ps {:?})",
                a.gpu_loads, a.gpu_res, b.gpu_loads, b.gpu_res, a.item_roots, b.item_roots
            );
            DRAW_STATS[0].fetch_add(1, Ordering::Relaxed);
            DRAW_DECIDED_N.fetch_add(1, Ordering::Relaxed);
            DrawState::Eligible(std::sync::Arc::new(DrawDecided { stages: [a, b] }))
        }
        (Some(_), Some(_)) => {
            crate::info!("gpu: draws of vs {vs:#x} ps {ps:#x} keep CPU-resolved chains (nothing varies per draw but user data)");
            DRAW_STATS[1].fetch_add(1, Ordering::Relaxed);
            DrawState::Refused
        }
        _ => {
            crate::info!("gpu: draws of vs {vs:#x} ps {ps:#x} keep CPU-resolved chains (a per-item chain reaches an image, sampler or written buffer)");
            DRAW_STATS[1].fetch_add(1, Ordering::Relaxed);
            DrawState::Refused
        }
    };
}

/// A GPU-resolved pair refused from now on (`why`: what a run's first draw
/// found).
pub fn refuse_draw(pair: (u64, u64), why: &str) {
    let mut g = PAIRS.write();
    let Some(st) = g.as_mut().and_then(|m| m.get_mut(&pair)) else { return };
    if matches!(st, DrawState::Eligible(_)) {
        *st = DrawState::Refused;
        DRAW_STATS[2].fetch_add(1, Ordering::Relaxed);
        crate::info!("gpu: draws of vs {:#x} ps {:#x} keep CPU-resolved chains from now on ({why})", pair.0, pair.1);
    }
}

/// Status part ("" when off).
pub fn status() -> String {
    if super::tune::gpu_chains() == 0 {
        return String::new();
    }
    let a = &super::exec::GPU_APPENDS;
    let c = &super::exec::GPU_RES_CHECKS;
    let d = &super::exec::GPU_DRAW_APPENDS;
    let l: Vec<u64> = super::exec::GPU_APPEND_LAPS.iter().map(|x| x.load(Ordering::Relaxed)).collect();
    let laps = if l[0] > 0 {
        let us = |k: usize| l[k] as f64 / l[0] as f64 / 1000.0;
        format!(
            "    GPU-resolved draws appended, us each by step (`DS_STEP_LAPS`, {} timed): checks {:.2}, imports {:.2}, image hazards {:.2}, indices {:.2}, SGPR data and arguments {:.2} (all {:.2})\n",
            l[0],
            us(1),
            us(2),
            us(3),
            us(4),
            us(5),
            (1..6).map(us).sum::<f64>()
        )
    } else {
        String::new()
    };
    format!(
        "    GPU-resolved programs: {} eligible, {} refused; commands appended to their runs {}, recorded in full: run roots differ {}, no open run or other state {}; ranges imported for them {}; reads of those in full checked {}: in a home {}, pending writes {}, in a resident shadow {}, not imported {}\n    GPU-resolved draws: pairs {} eligible, {} refused, {} refused later; draws appended to their runs {}, not appended: run roots, words or data differ {}, indices elsewhere {}, other state {}; commands not appended for a per-item vertex format other than their translation's {}; verified (`DS_GPU_VERIFY`) {}, with differences {}\n",
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
        c[4].load(Ordering::Relaxed),
        DRAW_STATS[0].load(Ordering::Relaxed),
        DRAW_STATS[1].load(Ordering::Relaxed),
        DRAW_STATS[2].load(Ordering::Relaxed),
        d[0].load(Ordering::Relaxed),
        d[1].load(Ordering::Relaxed),
        d[2].load(Ordering::Relaxed),
        d[3].load(Ordering::Relaxed),
        super::exec::GPU_FORMAT_MISSES.load(Ordering::Relaxed),
        super::exec::GPU_VERIFY[0].load(Ordering::Relaxed),
        super::exec::GPU_VERIFY[1].load(Ordering::Relaxed)
    ) + &laps
        + &super::exec::f3_line()
}
