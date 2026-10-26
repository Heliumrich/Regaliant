//! Draw census for GPU-resolved item draws (RENDERER.md "GPU-resolved item
//! passes", stage G4; `DS_DRAW_CENSUS=1`): per program pair, the draws, those
//! right after a draw of the pair with nothing recorded between and the same
//! context state (in runs), the roots of each stage's chains (`gpures::Root`:
//! user dword pairs and the pointers read from them) that vary within runs,
//! and what the varying (per-item) chains reach: read-only buffers, written
//! buffers, images, samplers. A sample of draws checks whether their per-item
//! buffers hold data GPU work wrote and the CPU path would not read from guest
//! memory (a home, pending GPU or CP writes). `profile reset` clears it; the
//! profile report prints the totals and the pairs with the most draws.

use super::gpures::{root_of_res, root_value, Root};
use parking_lot::Mutex;
use rdna::analysis::{Plan, Prov, Ptr, ResKind};
use std::collections::HashMap;
use std::sync::Arc;

pub fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_DRAW_CENSUS").ok().as_deref() == Some("1"))
}

/// A stage's chain roots: every pointer the first level of its user data
/// holds (k, None) and every pointer read from one (k, Some(off)).
fn roots_of(plan: &Plan) -> Vec<Root> {
    fn of_ptr(p: &Ptr) -> Option<Root> {
        match p {
            Ptr::User(k) => Some((*k, None)),
            Ptr::Mem(inner, off) => match &**inner {
                Ptr::User(k) => Some((*k, Some(*off))),
                other => of_ptr(other),
            },
            Ptr::Pc(_) => None,
        }
    }
    let mut roots: Vec<Root> = Vec::new();
    for r in plan.loads.iter().filter_map(|l| of_ptr(&l.ptr)).chain(plan.resources.iter().filter_map(root_of_res)) {
        if !roots.contains(&r) {
            roots.push(r);
        }
        // A pointer read from user pointer k: k itself is a root too.
        if r.1.is_some() && !roots.contains(&(r.0, None)) {
            roots.push((r.0, None));
        }
    }
    roots.truncate(64);
    roots
}

struct Stage {
    plan: Arc<Plan>,
    roots: Vec<Root>,
    /// Roots seen to differ between draws of a run.
    varies: u64,
}

#[derive(Default)]
struct Pair {
    stages: Vec<Stage>,
    draws: u64,
    in_runs: u64,
    gpu_args: u64,
    /// Sampled draws, and of them those whose per-item buffers hold data GPU
    /// work wrote (a home, pending writes).
    checked: u64,
    gpu_written: u64,
}

static PAIRS: Mutex<Option<HashMap<(u64, u64), Pair>>> = Mutex::new(None);

thread_local! {
    /// Per pair, this thread's last draw's root values.
    static LAST: std::cell::RefCell<HashMap<(u64, u64), Vec<u64>>> = std::cell::RefCell::new(HashMap::new());
}

/// Whether a stage's resource is reached by a per-item chain: its root
/// varies within runs (a pointer read from a user pointer that does not vary
/// is a run root, compared per item as `gpu_append` compares them).
fn per_item(st: &Stage, r: Option<Root>) -> bool {
    r.and_then(|x| st.roots.iter().position(|&y| y == x)).is_some_and(|i| st.varies & (1 << i) != 0)
}

/// A draw of `pair` with its stages' plans and user data (vertex stage
/// first); `consecutive`: right after a draw of the pair, nothing recorded
/// between, the same context state. Returns the per-item buffers' resource
/// indices per stage on a sampled draw (the caller checks their ranges and
/// reports through `checked`).
pub fn observe(pair: (u64, u64), stages: &[(&Arc<Plan>, &[u32])], consecutive: bool, gpu_args: bool) -> Option<Vec<Vec<usize>>> {
    let vals: Vec<u64> = stages.iter().flat_map(|(plan, user)| roots_of(plan).into_iter().map(|r| root_value(r, user)).collect::<Vec<_>>()).collect();
    let prev = LAST.with(|l| l.borrow_mut().insert(pair, vals.clone()));
    let mut g = PAIRS.lock();
    let m = g.get_or_insert_with(HashMap::new);
    let p = m.entry(pair).or_default();
    if p.stages.is_empty() {
        p.stages = stages.iter().map(|(plan, _)| Stage { plan: (*plan).clone(), roots: roots_of(plan), varies: 0 }).collect();
    }
    p.draws += 1;
    if gpu_args {
        p.gpu_args += 1;
    }
    if consecutive {
        p.in_runs += 1;
        if let Some(prev) = prev.filter(|v| v.len() == vals.len()) {
            let mut i = 0;
            for st in p.stages.iter_mut() {
                for k in 0..st.roots.len() {
                    if prev[i + k] != vals[i + k] {
                        st.varies |= 1 << k;
                    }
                }
                i += st.roots.len();
            }
        }
    }
    // Every 16th draw in a run: its per-item buffers checked.
    if !consecutive || p.in_runs % 16 != 0 {
        return None;
    }
    Some(
        p.stages
            .iter()
            .map(|st| (0..st.plan.resources.len()).filter(|&i| st.plan.resources[i].kind == ResKind::Buffer && per_item(st, root_of_res(&st.plan.resources[i]))).collect())
            .collect(),
    )
}

/// A sampled draw's per-item buffers checked: whether one holds data GPU
/// work wrote.
pub fn checked(pair: (u64, u64), gpu_written: bool) {
    let mut g = PAIRS.lock();
    if let Some(p) = g.as_mut().and_then(|m| m.get_mut(&pair)) {
        p.checked += 1;
        if gpu_written {
            p.gpu_written += 1;
        }
    }
}

pub fn reset() {
    *PAIRS.lock() = None;
}

/// What a pair's per-item chains reach: [read-only buffers, written buffers,
/// images, samplers, image tables, global bases].
fn reach(p: &Pair) -> [u32; 6] {
    let mut n = [0u32; 6];
    for st in &p.stages {
        for r in st.plan.resources.iter() {
            if !per_item(st, root_of_res(r)) {
                continue;
            }
            let k = match r.kind {
                ResKind::Buffer if r.written => 1,
                ResKind::Buffer => 0,
                ResKind::Image => 2,
                ResKind::Sampler => 3,
                ResKind::ImageTable => 4,
                ResKind::Global => 5,
            };
            n[k] += 1;
        }
    }
    n
}

/// Whether a resource's descriptor comes from user data directly (no chain).
fn direct(r: &rdna::analysis::Resource) -> bool {
    matches!(r.dwords.first(), Some(Prov::User(_)))
}

/// The profile report's lines over `flips` flips ("" when off or empty).
pub fn lines(flips: u64) -> String {
    if !on() {
        return String::new();
    }
    let g = PAIRS.lock();
    let Some(m) = g.as_ref() else { return String::new() };
    let flips = flips.max(1);
    let (mut all, mut runs, mut elig, mut imgs, mut written, mut none) = (0u64, 0u64, 0u64, 0u64, 0u64, 0u64);
    let mut v: Vec<(&(u64, u64), &Pair)> = m.iter().collect();
    v.sort_by(|a, b| b.1.draws.cmp(&a.1.draws));
    for (_, p) in &v {
        all += p.draws;
        runs += p.in_runs;
        let n = reach(p);
        if n[1] > 0 {
            written += p.in_runs;
        } else if n[2] + n[3] + n[4] > 0 {
            imgs += p.in_runs;
        } else if n[0] + n[5] > 0 {
            elig += p.in_runs;
        } else {
            none += p.in_runs;
        }
    }
    let mut s = format!(
        "    draw census ({} pairs, {:.0} draws a flip): in runs {:.0} a flip; of those, per-item chains reach only read-only buffers {:.0}, images or samplers {:.0}, written buffers {:.0}, nothing (user data only) {:.0}\n",
        m.len(),
        all as f64 / flips as f64,
        runs as f64 / flips as f64,
        elig as f64 / flips as f64,
        imgs as f64 / flips as f64,
        written as f64 / flips as f64,
        none as f64 / flips as f64
    );
    let top: Vec<String> = v
        .iter()
        .take(30)
        .map(|((vs, ps), p)| {
            let n = reach(p);
            let varies: Vec<String> = p.stages.iter().map(|st| st.roots.iter().enumerate().filter(|(i, _)| st.varies & (1 << i) != 0).map(|(_, r)| format!("{}{}", r.0, r.1.map_or(String::new(), |o| format!("+{o}")))).collect::<Vec<_>>().join(" ")).collect();
            let direct_n: usize = p.stages.iter().map(|st| st.plan.resources.iter().filter(|r| direct(r)).count()).sum();
            format!(
                "vs {vs:#x} ps {ps:#x}: {:.1} a flip, {:.0}% in runs, GPU-read arguments {:.0}%; varying roots vs [{}] ps [{}]; per-item reach: ro buffers {}, written {}, images {}, samplers {}, tables {}, globals {}; direct descriptors {direct_n}; sampled {} (GPU-written per-item data {})",
                p.draws as f64 / flips as f64,
                100.0 * p.in_runs as f64 / p.draws.max(1) as f64,
                100.0 * p.gpu_args as f64 / p.draws.max(1) as f64,
                varies.first().map_or("", |x| x.as_str()),
                varies.get(1).map_or("", |x| x.as_str()),
                n[0],
                n[1],
                n[2],
                n[3],
                n[4],
                n[5],
                p.checked,
                p.gpu_written
            )
        })
        .collect();
    for t in top {
        s += &format!("      {t}\n");
    }
    s
}
