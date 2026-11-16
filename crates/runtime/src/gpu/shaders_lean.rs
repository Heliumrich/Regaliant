//! The lean lane's stage preparation (ROUTE60.md "2026-11-16 midday"; `tune lean`).
//!
//! `super::prepare` interprets a program's analysis plan for every command: it walks the plan's
//! pointer trees (`Ptr`) through the reader's pointer list and chain memo, looks its per-plan
//! facts up in thread-local maps, builds a whole specialization and finds the translation by
//! comparing or hashing it. Here a plan is compiled once per thread into flat operations (pointer
//! chain nodes, per-resource reads and decode rules, scalar loads) with those facts folded in,
//! and the translation is found by the stage template's number and the resource specializations
//! in a small table. The memory reads go through the same reader primitives in the same order,
//! so the read ranges, words, unread ranges and data loads are the full path's; anything this
//! path does not handle (image tables, unreadable chains, words without a provenance, diagnoses,
//! a translation not seen yet on this thread) returns None and the full path runs.
//! `DS_LEAN_CHECK=1` prepares every stage both ways and compares the results.

use super::super::desc::{Tsharp, Vsharp};
use super::{pool, DataLoad, Found, Local, Prepared, Reader, Resolved, Translated, UserData};
use rdna::analysis::{Plan, Prov, Ptr, ResKind};
use rdna::translate::{self, ImageDim, ResSpec, Spec, Stage};
use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Pointer chain nodes a compiled plan holds at most (plans with more take the full path).
const MAX_NODES: usize = 48;

/// Where a pointer or word comes from: a user-data pair, a chain node's value, or a
/// program-relative address.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Root {
    User(u16),
    Node(u16),
    Pc(u64),
}

/// A pointer read at `off` from `from` (one structurally distinct `Ptr::Mem` of the plan).
#[derive(Clone, Copy, Debug)]
struct Node {
    from: Root,
    off: u32,
}

/// How a descriptor dword is made (`Reader::eval`).
#[derive(Clone, Copy, Debug)]
enum Word {
    User(u16),
    Const(u32),
    Mem(Root, u32),
    PtrLo(Root, u32),
    PtrHi(Root, u32),
    /// No provenance the lean path evaluates (the full path reports it).
    Bad,
}

/// A resource: its descriptor's first dword's pointer and offset when read from memory, whether
/// its dwords are one block behind it, and each dword's rule.
struct Res {
    kind: ResKind,
    first: Option<(Root, u32)>,
    block: bool,
    n: u8,
    words: [Word; 8],
    written: bool,
    /// Globals: the bytes their constant-offset loads read (`Resource::extent`).
    extent: Option<u64>,
}

/// A scalar load: `dwords` at `off` from `from`; `data`: `Plan::load_data`.
#[derive(Clone, Copy)]
struct Load {
    from: Root,
    off: u32,
    dwords: u32,
    data: bool,
}

/// A translation this thread found for a template (by its content key) and resource
/// specializations, with its own plan's scalar loads compiled against the plan's nodes.
struct TrEntry {
    tkey: u64,
    resources: Vec<ResSpec>,
    tr: Arc<Local<Translated>>,
    loads: Vec<Load>,
    sgpr_dwords: usize,
    code_base: usize,
    res_info: usize,
    /// The translation's plan's resource count (the info blocks it initializes).
    nres: usize,
}

/// A plan compiled for one thread.
pub(super) struct LeanPlan {
    key: u64,
    nodes: RefCell<Vec<(Ptr, Node)>>,
    res: Vec<Res>,
    fmt_mask: u64,
    store_fmt: u64,
    dim3: u64,
    dim2: (u64, u64),
    cmp: u64,
    /// The plan supports the lean path (no image tables, no errors).
    ok: bool,
    hints: Cell<[u32; 3]>,
    trs: RefCell<Vec<TrEntry>>,
    /// The next entry `learn` replaces when `trs` is full.
    next: Cell<usize>,
    /// Per buffer resource: the format and swizzle `single_format` last answered for, with
    /// the thread's format-memo generation and the answer.
    single: RefCell<Vec<Option<(u32, u64, bool)>>>,
}

/// Translations kept per plan (round robin).
const TRS_PER_PLAN: usize = 16;

/// Direct-mapped slots of compiled plans per thread.
const PLAN_SLOTS: usize = 1024;

thread_local! {
    static PLANS: RefCell<Vec<Option<Rc<LeanPlan>>>> = const { RefCell::new(Vec::new()) };
    /// The resource specializations of the preparation in progress (the translation's key).
    static RES_SPECS: RefCell<Vec<ResSpec>> = const { RefCell::new(Vec::new()) };
}

/// [preparations served, fell back (plan unsupported), fell back (translation not in the
/// table), fell back (unreadable or unevaluable), fell back (diagnosis), checked, differing,
/// of those equal at a second lean read, still differing then].
pub static STATS: [AtomicU64; 9] = [const { AtomicU64::new(0) }; 9];

fn slot(key: u64) -> usize {
    ((key ^ (key >> 21) ^ (key >> 43)) as usize) & (PLAN_SLOTS - 1)
}

/// `DS_LEAN_CHECK=1`: every stage prepared both ways and compared (`compare`).
pub(super) fn check_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_LEAN_CHECK").ok().as_deref() == Some("1"))
}

/// The root a pointer comes from, adding its chain's nodes (inner first).
fn root_of(p: &Ptr, nodes: &mut Vec<(Ptr, Node)>) -> Option<Root> {
    match p {
        Ptr::User(k) => Some(Root::User(*k)),
        Ptr::Pc(o) => Some(Root::Pc(*o)),
        Ptr::Mem(inner, off) => {
            if let Some(i) = nodes.iter().position(|(q, _)| q == p) {
                return Some(Root::Node(i as u16));
            }
            let from = root_of(inner, nodes)?;
            if nodes.len() >= MAX_NODES {
                return None;
            }
            nodes.push((p.clone(), Node { from, off: *off }));
            Some(Root::Node((nodes.len() - 1) as u16))
        }
    }
}

fn word_of(p: &Prov, nodes: &mut Vec<(Ptr, Node)>) -> Word {
    match p {
        Prov::User(k) => Word::User(*k),
        Prov::Const(c) => Word::Const(*c),
        Prov::Mem(ptr, off) => root_of(ptr, nodes).map_or(Word::Bad, |r| Word::Mem(r, *off)),
        Prov::PtrLo(ptr, b) => root_of(ptr, nodes).map_or(Word::Bad, |r| Word::PtrLo(r, *b)),
        Prov::PtrHi(ptr, b) => root_of(ptr, nodes).map_or(Word::Bad, |r| Word::PtrHi(r, *b)),
        Prov::Unknown | Prov::Table(..) => Word::Bad,
    }
}

fn compile_loads(plan: &Plan, nodes: &mut Vec<(Ptr, Node)>) -> Option<Vec<Load>> {
    let mut v = Vec::with_capacity(plan.loads.len());
    for (j, l) in plan.loads.iter().enumerate() {
        v.push(Load { from: root_of(&l.ptr, nodes)?, off: l.offset, dwords: l.dwords, data: plan.load_data.get(j).copied().unwrap_or(false) });
    }
    Some(v)
}

/// This thread's compiled plan for `key` (compiled from the full path's per-plan facts on
/// first use; None while the full path has not made the plan).
fn plan_for(key: u64, insts: &[rdna::Inst], stage: Stage) -> Option<Rc<LeanPlan>> {
    let s = slot(key);
    let hit = PLANS.with(|t| t.borrow().get(s).and_then(|e| e.as_ref().filter(|p| p.key == key).cloned()));
    if hit.is_some() {
        return hit;
    }
    let plan: Arc<Plan> = super::LOCAL_PLANS.with(|c| c.borrow().get(&key).map(|p| (**p).clone()))?;
    let mut nodes: Vec<(Ptr, Node)> = Vec::new();
    let mut res = Vec::with_capacity(plan.resources.len());
    let mut ok = plan.errors.is_empty();
    for (i, r) in plan.resources.iter().enumerate() {
        if r.kind == ResKind::ImageTable {
            ok = false;
        }
        let first = match r.dwords.first() {
            Some(Prov::Mem(p, off)) => match root_of(p, &mut nodes) {
                Some(rt) => Some((rt, *off)),
                None => {
                    ok = false;
                    None
                }
            },
            _ => None,
        };
        let mut words = [Word::Bad; 8];
        for (k, p) in r.dwords.iter().take(8).enumerate() {
            words[k] = word_of(p, &mut nodes);
        }
        res.push(Res { kind: r.kind, first, block: plan.blocks.get(i).copied().unwrap_or(false), n: r.dwords.len().min(8) as u8, words, written: r.written, extent: r.extent.map(|e| e as u64) });
    }
    if compile_loads(&plan, &mut nodes).is_none() {
        ok = false;
    }
    let has_buffers = plan.resources.iter().any(|r| r.kind == ResKind::Buffer);
    let fmt_mask = if has_buffers { super::LOCAL_FORMATS.with(|c| *c.borrow_mut().entry(key).or_insert_with(|| super::shared_scan(1, key, || super::format_resources(insts, &plan)))) } else { 0 };
    let compute = stage == Stage::Compute;
    let store_fmt = if compute && has_buffers { super::LOCAL_STORE_FORMATS.with(|c| *c.borrow_mut().entry(key).or_insert_with(|| super::format_store_resources(insts, &plan))) } else { 0 };
    let any_image = plan.resources.iter().any(|r| r.kind == ResKind::Image);
    let dim3 = if compute && any_image { super::LOCAL_DIM3.with(|c| *c.borrow_mut().entry(key).or_insert_with(|| super::d3_resources(insts, &plan))) } else { 0 };
    let dim2 = if compute && any_image { super::LOCAL_DIM2.with(|c| *c.borrow_mut().entry(key).or_insert_with(|| super::d2_resources(insts, &plan))) } else { (0, 0) };
    let cmp = if any_image { super::compare_images(key, insts, &plan) } else { 0 };
    let n = plan.resources.len();
    let lp = Rc::new(LeanPlan {
        key,
        nodes: RefCell::new(nodes),
        res,
        fmt_mask,
        store_fmt,
        dim3,
        dim2,
        cmp,
        ok,
        hints: Cell::new(super::READ_HINTS.with(|h| h.borrow().get(&key).copied().unwrap_or([0; 3]))),
        trs: RefCell::new(Vec::new()),
        next: Cell::new(0),
        single: RefCell::new(vec![None; n]),
    });
    PLANS.with(|t| {
        let mut t = t.borrow_mut();
        if t.is_empty() {
            t.resize(PLAN_SLOTS, None);
        }
        t[s] = Some(lp.clone());
    });
    Some(lp)
}

/// A node's value, read once per preparation (`Reader::ptr`'s order: the chain it comes from
/// first, then its two words).
fn node_val(i: usize, lp: &LeanPlan, nodes: &[(Ptr, Node)], vals: &mut [Option<u64>; MAX_NODES], rd: &mut Reader, user: &[u32], code_base: u64) -> Option<u64> {
    if let Some(v) = vals[i] {
        return Some(v);
    }
    let n = nodes[i].1;
    let base = root_val(n.from, lp, nodes, vals, rd, user, code_base)? + n.off as u64;
    let lo = rd.u32(base)? as u64;
    let hi = rd.u32(base + 4)? as u64;
    let v = lo | (hi << 32);
    vals[i] = Some(v);
    Some(v)
}

fn root_val(r: Root, lp: &LeanPlan, nodes: &[(Ptr, Node)], vals: &mut [Option<u64>; MAX_NODES], rd: &mut Reader, user: &[u32], code_base: u64) -> Option<u64> {
    match r {
        Root::User(k) => Some((*user.get(k as usize)? as u64) | ((*user.get(k as usize + 1)? as u64) << 32)),
        Root::Pc(o) => Some(code_base + o),
        Root::Node(i) => node_val(i as usize, lp, nodes, vals, rd, user, code_base),
    }
}

fn eval_word(w: Word, lp: &LeanPlan, nodes: &[(Ptr, Node)], vals: &mut [Option<u64>; MAX_NODES], rd: &mut Reader, user: &[u32], code_base: u64) -> Option<u32> {
    match w {
        Word::User(k) => user.get(k as usize).copied(),
        Word::Const(c) => Some(c),
        Word::Mem(r, off) => {
            let b = root_val(r, lp, nodes, vals, rd, user, code_base)?;
            rd.u32(b + off as u64)
        }
        Word::PtrLo(r, b) => Some((root_val(r, lp, nodes, vals, rd, user, code_base)? + b as u64) as u32),
        Word::PtrHi(r, b) => Some(((root_val(r, lp, nodes, vals, rd, user, code_base)? + b as u64) >> 32) as u32),
        Word::Bad => None,
    }
}

/// The lean preparation of a stage: its program, user data, stage kind and user-data layout,
/// and its template's content key (`super::tmpl_key`): Some when this path made it, None for
/// the full path.
#[allow(clippy::too_many_arguments)]
pub(super) fn prepare(addr: u64, insts: &Arc<Local<Vec<rdna::Inst>>>, code_hash: u64, user: &[u32], stage: Stage, user_sgprs: u16, user_base: u16, tkey: u64) -> Option<Result<Prepared, String>> {
    if tkey == 0 || super::fmt_census_on() || super::super::slotcensus::on() || super::armed_census_on() {
        STATS[4].fetch_add(1, Ordering::Relaxed);
        return None;
    }
    let t0 = unsafe { core::arch::x86_64::_rdtsc() };
    let r = RES_SPECS.with(|r| {
        let mut specs = r.borrow_mut();
        prepare_with(addr, insts, code_hash, user, stage, user_sgprs, user_base, tkey, &mut specs)
    });
    if r.is_some() {
        time(0, unsafe { core::arch::x86_64::_rdtsc() }.wrapping_sub(t0));
        if twice_on() {
            // The same preparation again on warm caches: what it costs without its misses.
            PHASE_FIRST.with(|c| c.set(false));
            let t1 = unsafe { core::arch::x86_64::_rdtsc() };
            let r2 = RES_SPECS.with(|r| {
                let mut specs = r.borrow_mut();
                prepare_with(addr, insts, code_hash, user, stage, user_sgprs, user_base, tkey, &mut specs)
            });
            PHASE_FIRST.with(|c| c.set(true));
            if r2.is_some() {
                time(3, unsafe { core::arch::x86_64::_rdtsc() }.wrapping_sub(t1));
            }
        }
    }
    r
}

/// With `DS_LEAN_TWICE`: cycles of a lean preparation's phases [plan and reader, resources,
/// depth views and translation, scalar loads and SGPR data, result], and preparations timed.
static PHASES: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

thread_local! {
    static PHASE_LOCAL: Cell<[u64; 6]> = const { Cell::new([0; 6]) };
    /// The preparation in progress is the first of the two (`DS_LEAN_TWICE`): its phases are timed.
    static PHASE_FIRST: Cell<bool> = const { Cell::new(true) };
}

/// `DS_LEAN_TWICE=1` (a diagnosis): each lean preparation made twice, the second on warm caches
/// timed apart (`time` kind 3).
fn twice_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_LEAN_TWICE").ok().as_deref() == Some("1"))
}

/// Cycles and preparations: [lean, full path with the lane off, full path for what the lane
/// left] (`time`).
static CYCLES: [[AtomicU64; 2]; 4] = [const { [const { AtomicU64::new(0) }; 2] }; 4];

thread_local! {
    static CYC_LOCAL: Cell<[(u64, u64); 4]> = const { Cell::new([(0, 0); 4]) };
}

/// Adds one preparation's cycles to kind `k` (0 lean, 1 the full path with the lane off, 2 the
/// full path for what the lane left).
pub(super) fn time(k: usize, cycles: u64) {
    CYC_LOCAL.with(|c| {
        let mut v = c.get();
        v[k].0 += cycles;
        v[k].1 += 1;
        if v[k].1 >= 4096 {
            CYCLES[k][0].fetch_add(v[k].0, Ordering::Relaxed);
            CYCLES[k][1].fetch_add(v[k].1, Ordering::Relaxed);
            v[k] = (0, 0);
        }
        c.set(v);
    });
}

/// `profile reset`: the cycle sums emptied.
pub fn reset_cycles() {
    for k in &CYCLES {
        for c in k {
            c.store(0, Ordering::Relaxed);
        }
    }
    for c in &PHASES {
        c.store(0, Ordering::Relaxed);
    }
}

#[allow(clippy::too_many_arguments)]
fn prepare_with(addr: u64, insts: &Arc<Local<Vec<rdna::Inst>>>, code_hash: u64, user: &[u32], stage: Stage, user_sgprs: u16, user_base: u16, tkey: u64, specs: &mut Vec<ResSpec>) -> Option<Result<Prepared, String>> {
    let phases = twice_on() && PHASE_FIRST.with(|c| c.get());
    let mut tp = if phases { unsafe { core::arch::x86_64::_rdtsc() } } else { 0 };
    let mut mark = |k: usize| {
        if phases {
            let t = unsafe { core::arch::x86_64::_rdtsc() };
            PHASE_LOCAL.with(|c| {
                let mut v = c.get();
                v[k] += t.wrapping_sub(tp);
                c.set(v);
            });
            tp = t;
        }
    };
    let plan_key = code_hash ^ (user_sgprs as u64) << 48 ^ (user_base as u64) << 56;
    let lp = plan_for(plan_key, insts, stage)?;
    if !lp.ok {
        STATS[1].fetch_add(1, Ordering::Relaxed);
        return None;
    }
    let map_gen = crate::mem::MAP_GEN.load(Ordering::Acquire);
    let images_gen = super::super::images::GPU_IMAGES_GEN.load(Ordering::Acquire);
    let mut rd = Reader::with_sizes(addr, lp.hints.get());
    rd.load_spans();
    let nodes_ref = lp.nodes.borrow();
    let nodes: &[(Ptr, Node)] = &nodes_ref;
    let mut vals = [None::<u64>; MAX_NODES];
    let code_base = addr;
    let nres = lp.res.len();
    let mut resources = pool::resolved(nres);
    let mut res_addrs = pool::addrs(nres);
    specs.clear();
    let compute = stage == Stage::Compute;
    let mut specialized = 0u64;
    mark(0);
    let fail = |resources: Vec<Resolved>, res_addrs: Vec<Option<u64>>| {
        pool::put_resolved(resources);
        pool::put_addrs(res_addrs);
        STATS[3].fetch_add(1, Ordering::Relaxed);
    };
    for (i, r) in lp.res.iter().enumerate() {
        let mut w = [0u32; 8];
        let src = match r.first {
            Some((root, off)) => match root_val(root, &lp, nodes, &mut vals, &mut rd, user, code_base) {
                Some(b) => Some(b + off as u64),
                None => {
                    fail(resources, res_addrs);
                    return None;
                }
            },
            None => None,
        };
        res_addrs.push(src);
        let n = r.n as usize;
        let block = r.block && src.is_some_and(|b| rd.read_into(b, &mut w[..n]));
        if !block {
            for k in 0..n {
                match eval_word(r.words[k], &lp, nodes, &mut vals, &mut rd, user, code_base) {
                    Some(x) => w[k] = x,
                    None => {
                        fail(resources, res_addrs);
                        return None;
                    }
                }
            }
        }
        match r.kind {
            ResKind::Buffer => {
                let v = Vsharp::decode(&w);
                let (format, dst_sel) = if lp.fmt_mask & (1 << i.min(63)) == 0 {
                    (0, 0)
                } else if compute && translate::rt_format_entry(v.format as u16) != 0 && lp.store_fmt & (1 << i.min(63)) == 0 {
                    let want = (v.format as u32) << 16 | v.dst_sel as u32;
                    let fgen = super::FMT_GEN.with(|g| g.get());
                    let memo = lp.single.borrow()[i];
                    let single = match memo {
                        Some((f, g, ans)) if f == want && g == fgen => ans,
                        _ => {
                            let ans = super::single_format(code_hash, i, v.format, v.dst_sel);
                            let g2 = super::FMT_GEN.with(|g| g.get());
                            lp.single.borrow_mut()[i] = Some((want, g2, ans));
                            ans
                        }
                    };
                    if single {
                        if i < 64 {
                            specialized |= 1 << i;
                        }
                        (v.format, v.dst_sel)
                    } else {
                        (translate::FORMAT_RT, 0)
                    }
                } else {
                    (v.format, v.dst_sel)
                };
                specs.push(ResSpec::Buffer { format, dst_sel });
                resources.push(Resolved::Buffer(v));
            }
            ResKind::Global => {
                specs.push(ResSpec::Global);
                resources.push(Resolved::Global(w[0] as u64 | ((w[1] as u64) << 32), r.extent));
            }
            ResKind::Image => {
                let t = Tsharp::decode(&w);
                let Some(dim) = ImageDim::from_type(t.ty, t.depth) else {
                    fail(resources, res_addrs);
                    return None;
                };
                let dim = if compute && matches!(t.ty, 9 | 10) && !r.written && lp.dim3 & (1 << i.min(63)) != 0 { ImageDim::D3OrD2 } else { dim };
                let dim = if compute && matches!(dim, ImageDim::D2 | ImageDim::D2Array) && !r.written && super::dim_by_instructions() {
                    let bit = 1u64 << i.min(63);
                    if lp.dim2.1 & bit != 0 {
                        ImageDim::D2Array
                    } else if lp.dim2.0 & bit != 0 {
                        ImageDim::D2
                    } else {
                        dim
                    }
                } else {
                    dim
                };
                specs.push(ResSpec::Image { dim, class: super::num_class(t.format), depth: true });
                resources.push(Resolved::Image(t));
            }
            ResKind::Sampler => {
                specs.push(ResSpec::Sampler);
                resources.push(Resolved::Sampler([w[0], w[1], w[2], w[3]]));
            }
            ResKind::ImageTable => {
                fail(resources, res_addrs);
                return None;
            }
        }
    }
    mark(1);
    // Depth views for the images comparison instructions sample.
    let mut cmp_images = 0u64;
    let mut cmp_depth = 0u64;
    if super::super::tune::cmp_emulate() && resources.iter().any(|r| matches!(r, Resolved::Image(_))) {
        cmp_images = lp.cmp;
        for (r, s) in resources.iter().zip(specs.iter_mut()) {
            if let (Resolved::Image(_), ResSpec::Image { depth, .. }) = (r, s) {
                *depth = false;
            }
        }
        if cmp_images != 0 {
            super::super::images::with(|im| {
                for (i, (r, s)) in resources.iter().zip(specs.iter_mut()).enumerate() {
                    if let (Resolved::Image(t), ResSpec::Image { depth, .. }) = (r, s) {
                        if i < 64 && cmp_images & (1 << i) != 0 {
                            *depth = im.depth_view(t);
                            if *depth {
                                cmp_depth |= 1 << i;
                            }
                        }
                    }
                }
            });
        }
    }
    // The translation by template and resource specializations.
    let found_tr = {
        let trs = lp.trs.borrow();
        trs.iter().position(|e| e.tkey == tkey && e.resources == *specs)
    };
    let Some(ti) = found_tr else {
        fail(resources, res_addrs);
        STATS[3].fetch_sub(1, Ordering::Relaxed);
        STATS[2].fetch_add(1, Ordering::Relaxed);
        return None;
    };
    let _ = specialized;
    mark(2);
    let trs = lp.trs.borrow();
    let te = &trs[ti];
    let tr = te.tr.clone();
    // Scalar-load data.
    let mut sgpr_data = pool::words(te.sgpr_dwords.max(translate::SGPR_DATA_USER as usize));
    sgpr_data.extend_from_slice(user);
    sgpr_data.resize(translate::SGPR_DATA_USER as usize, 0);
    let mut data_loads = pool::loads(te.loads.iter().filter(|l| l.data).count());
    for l in &te.loads {
        let start = sgpr_data.len();
        sgpr_data.resize(start + l.dwords as usize, 0);
        let Some(b) = root_val(l.from, &lp, nodes, &mut vals, &mut rd, user, code_base) else {
            // (The full path leaves the words zero and goes on; its unread ranges differ.)
            drop(trs);
            pool::put_words(sgpr_data);
            pool::put_loads(data_loads);
            fail(resources, res_addrs);
            return None;
        };
        let word = rd.words.len();
        if !rd.read_into(b + l.off as u64, &mut sgpr_data[start..]) {
            for k in 0..l.dwords as usize {
                sgpr_data[start + k] = rd.u32(b + l.off as u64 + 4 * k as u64).unwrap_or(0);
            }
        }
        if l.data && rd.words.len() == word + l.dwords as usize {
            data_loads.push(DataLoad { sgpr: start as u32, addr: b + l.off as u64, dwords: l.dwords, word: word as u32 });
        }
    }
    sgpr_data.resize(te.sgpr_dwords, 0);
    sgpr_data[te.code_base] = addr as u32;
    sgpr_data[te.code_base + 1] = (addr >> 32) as u32;
    for k in 0..te.nres {
        sgpr_data[te.res_info + translate::RES_INFO_DWORDS as usize * k + 3] = u32::MAX;
    }
    for (k, r) in resources.iter().enumerate() {
        if let Resolved::Image(t) = r {
            sgpr_data[te.res_info + translate::RES_INFO_DWORDS as usize * k] = super::super::lodstats::info(t);
            super::super::lodstats::note(&t.raw);
        }
    }
    drop(trs);
    mark(3);
    let used = [rd.reads.len() as u32, rd.words.len() as u32, 0];
    lp.hints.set(used);
    rd.save_spans();
    drop(nodes_ref);
    let Reader { reads, words, unread, chain_vals, .. } = rd;
    pool::put_addrs(chain_vals);
    STATS[0].fetch_add(1, Ordering::Relaxed);
    let found = Arc::new(Found { insts: insts.clone(), tr, resources, res_addrs, reads, words, unread, data_loads });
    mark(4);
    if phases {
        PHASE_LOCAL.with(|c| {
            let mut v = c.get();
            v[5] += 1;
            if v[5] >= 4096 {
                for (a, b) in PHASES.iter().zip(v.iter()) {
                    a.fetch_add(*b, Ordering::Relaxed);
                }
                v = [0; 6];
            }
            c.set(v);
        });
    }
    Some(Ok(Prepared {
        stage,
        addr,
        found,
        sgpr_data,
        user: UserData::new(user),
        map_gen,
        images_gen,
        cmp_images,
        cmp_depth,
        code_hash,
        ahead: 1,
    }))
}

/// After the full path prepared a stage that this path left to it: its translation recorded
/// for the template and resource specializations, when the translation is the one the full
/// path finds for that specialization (not a run-time variant standing in for one).
pub(super) fn learn(p: &Prepared, spec: &Spec, tkey: u64) {
    if tkey == 0 || p.tr.spec.resources != spec.resources {
        return;
    }
    let plan_key = p.code_hash ^ (spec.user_sgprs as u64) << 48 ^ (spec.user_base as u64) << 56;
    let s = slot(plan_key);
    let Some(lp) = PLANS.with(|t| t.borrow().get(s).and_then(|e| e.as_ref().filter(|q| q.key == plan_key).cloned())) else { return };
    if !lp.ok {
        return;
    }
    let mut trs = lp.trs.borrow_mut();
    if trs.iter().any(|e| e.tkey == tkey && e.resources == spec.resources) {
        return;
    }
    let loads = {
        let mut nodes = lp.nodes.borrow_mut();
        match compile_loads(&p.tr.plan, &mut nodes) {
            Some(l) => l,
            None => return,
        }
    };
    let e = TrEntry {
        tkey,
        resources: spec.resources.clone(),
        tr: p.found.tr.clone(),
        loads,
        sgpr_dwords: translate::sgpr_data_dwords(&p.tr.plan) as usize,
        code_base: translate::code_base_offset(&p.tr.plan) as usize,
        res_info: translate::res_info_offset(&p.tr.plan) as usize,
        nres: p.tr.plan.resources.len(),
    };
    if trs.len() < TRS_PER_PLAN {
        trs.push(e);
    } else {
        let k = lp.next.get() % TRS_PER_PLAN;
        lp.next.set(k + 1);
        trs[k] = e;
    }
}

/// `DS_LEAN_CHECK=1`: the lean preparation against the full path's; false (logged, the first
/// 40) when they differ. `second`: a second lean preparation after a difference (one in the
/// words alone that this one does not show is the title's data read twice moments apart).
pub(super) fn compare(a: &Prepared, b: &Prepared, second: bool) -> bool {
    if !second {
        STATS[5].fetch_add(1, Ordering::Relaxed);
    }
    let same_tr = Arc::ptr_eq(&a.found.tr.0, &b.found.tr.0) || a.tr.spec == b.tr.spec;
    let diffs: Vec<&str> = [
        (!same_tr, "translation"),
        (a.resources != b.resources, "resources"),
        (a.res_addrs != b.res_addrs, "resource addresses"),
        (a.reads != b.reads, "reads"),
        (a.words != b.words, "words"),
        (a.unread != b.unread, "unread"),
        (a.data_loads != b.data_loads, "data loads"),
        (a.sgpr_data != b.sgpr_data, "sgpr data"),
        (a.cmp_images != b.cmp_images || a.cmp_depth != b.cmp_depth, "compared images"),
        (a.user != b.user, "user data"),
    ]
    .into_iter()
    .filter(|x| x.0)
    .map(|x| x.1)
    .collect();
    if diffs.is_empty() {
        if second {
            STATS[7].fetch_add(1, Ordering::Relaxed);
        }
        return true;
    }
    if STATS[6 + 2 * second as usize].fetch_add(1, Ordering::Relaxed) < 40 {
        crate::info!("lean check: {:?} {:#x}: the {} lean preparation differs in {}", a.stage, a.addr, if second { "second" } else { "first" }, diffs.join(", "));
    }
    false
}

/// Status line ("" before any lean preparation).
pub fn status() -> String {
    let s: Vec<u64> = STATS.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    let hz = super::super::threadcpu::tsc_hz();
    let per = |k: usize| {
        let (c, n) = (CYCLES[k][0].load(Ordering::Relaxed), CYCLES[k][1].load(Ordering::Relaxed));
        if n == 0 { "-".to_string() } else { format!("{:.0} ns ({n})", c as f64 * 1e9 / hz / n as f64) }
    };
    let ph: Vec<u64> = PHASES.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    let phase_line = if ph[5] == 0 {
        String::new()
    } else {
        let ns = |c: u64| c as f64 * 1e9 / hz / ph[5] as f64;
        format!(
            "    lean preparation by phase (`DS_LEAN_TWICE`, {} timed): plan and reader {:.0} ns, resources {:.0}, depth views and translation {:.0}, scalar loads and SGPR data {:.0}, result {:.0}
",
            ph[5], ns(ph[0]), ns(ph[1]), ns(ph[2]), ns(ph[3]), ns(ph[4])
        )
    };
    let timing = phase_line + &format!("    stage preparation cycles (since the last profile reset): lean {}, full path with the lane off {}, full path for what the lane left {}, lean again on warm caches {}
", per(0), per(1), per(2), per(3));
    if s.iter().all(|&x| x == 0) {
        return timing;
    }
    format!(
        "    lean stage preparations (`tune lean`): {} made here, left to the full path: {} unsupported plans, {} translations not in the table, {} unreadable or unevaluable, {} diagnoses; checked {}, differing {} (equal at a second lean read {}, still differing {})\n",
        s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8]
    ) + &timing
}
