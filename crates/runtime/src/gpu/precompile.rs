//! Compute pipelines made when the title creates their shader objects
//! (`tune precompile`).
//!
//! The title creates a shader object (`sceAgcCreateShader`) for each program
//! of a shader file it loads (.csdr: the engine's at boot, a level's materials
//! and particle effects as they load), well before the programs first run,
//! and many more than run (5,646 compute objects against ~340 programs
//! dispatched in a first session to the courtyard). A first dispatch
//! otherwise compiles its pipeline on the command processor (0.2-2.9 s for
//! the large programs; `gpu::compile` scans ahead from there), in every
//! first session and again whenever the translations change (a new build).
//!
//! A program's translation spec comes from its dispatch: the registers the
//! object carries (RSRC2, the thread-group size; wave64 follows from the
//! code) and the descriptors bound then (image dimensions and number
//! classes, typed buffers' formats), which no object holds. So every first
//! translation of a compute program records its spec by code hash
//! (`learn`), in a file that outlives the pipeline store (`DS_SPECS`, else
//! local-data/shader-specs.txt). With `precompile` 2 (the default) a created object
//! whose program was seen before has its pipelines made from those specs on
//! `tune precompile_workers` threads at the lowest priority, under the keys
//! its dispatches look up (`shaders::translate_for_spec`,
//! `exec::precompile_translation`); 3 also makes the guess from the object
//! alone (`shaders::guess_compute_spec`; complete for ~40% of the programs a
//! session dispatches, and most objects never run). With 1, and above, each
//! first translation is also compared with that guess (status line).
//!
//! Graphics pipelines the same way: each one created records both stages'
//! specs and its state (`learn_gfx`; target and depth formats, blend, write
//! masks, primitive class, depth clamp), and once the title has created
//! objects for both of a learned pipeline's programs it is made
//! (`exec::precompile_gfx_translations`).
//!
//! The learned programs' code is kept too (`<store>.code`: code hash, word
//! count, words; local data like the pipeline store), so every learned
//! pipeline whose programs' code is there is made from the start of a
//! session (`early`), before the title loads them: a first session's ~500
//! compute pipelines take ~330 s of compiles, more than the boot leaves if
//! they start only when the objects are created.

use parking_lot::{Condvar, Mutex};
use rdna::translate::{ImageDim, MrtSpec, NumClass, PsInput, ResSpec, Spec};
use std::collections::HashMap;
use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};

/// A compute shader object's registers: RSRC2, the thread-group size and
/// the dispatch modifier (`sceAgcCreateShader`).
#[derive(Clone, Copy)]
struct Created {
    rsrc2: u32,
    local: [u32; 3],
    modifier: u32,
}

/// Compute programs by address, as their shader objects describe them.
static CREATED: Mutex<Option<HashMap<u64, Created>>> = Mutex::new(None);
/// Programs whose first translation was compared (addresses).
static COMPARED: Mutex<Option<std::collections::HashSet<u64>>> = Mutex::new(None);

/// What a dispatch adds to a program's spec: RSRC2, the thread-group size,
/// wave64, the resources' specs and whether it ran batched (runs of
/// consecutive dispatches, `shaders::batchable`: its translation differs).
#[derive(Clone, PartialEq, Eq, Hash)]
struct Seen {
    rsrc2: u32,
    local: [u32; 3],
    wave64: bool,
    resources: Vec<ResSpec>,
    batched: bool,
}

/// The specs dispatches used, by code hash, with the program's rank (the
/// order in which programs were first dispatched, loaded from the store,
/// then appended to it), and the store's path.
static LEARNED: Mutex<Option<HashMap<u64, (u32, Vec<Seen>)>>> = Mutex::new(None);
static STORE: std::sync::OnceLock<std::path::PathBuf> = std::sync::OnceLock::new();

/// A stage of a learned graphics pipeline: what its draws' registers and
/// descriptors put in its spec (`shaders::vertex_spec`, `pixel_spec`).
#[derive(Clone, PartialEq, Eq, Hash)]
struct StageSeen {
    user_sgprs: u16,
    wave64: bool,
    base_instance: Option<u16>,
    ps_input_addr: u32,
    ps_inputs: Vec<PsInput>,
    mrt: [MrtSpec; 8],
    early_tests: bool,
    early_ok: bool,
    /// A pixel stage of a draw measuring LOD statistics where they are
    /// decided per draw (`shaders::lod_per_draw`: its depth tests stay after
    /// the shader and it keeps them; the other draws' variant has none).
    lod_measure: bool,
    resources: Vec<ResSpec>,
}

impl StageSeen {
    fn of(s: &Spec) -> StageSeen {
        let lod_measure = s.stage == rdna::translate::Stage::Fragment && s.lod_stats && !s.batched && !s.early_tests && !s.early_ok && super::shaders::lod_per_draw();
        StageSeen { user_sgprs: s.user_sgprs, wave64: s.wave64, base_instance: s.base_instance_sgpr, ps_input_addr: s.ps_input_addr, ps_inputs: s.ps_inputs.clone(), mrt: s.mrt, early_tests: s.early_tests, early_ok: s.early_ok, lod_measure, resources: s.resources.clone() }
    }
}

/// A learned graphics pipeline: its programs' code hashes (pixel 0: none),
/// their stages and the state.
#[derive(Clone, PartialEq, Eq, Hash)]
struct GfxRecipe {
    vs: u64,
    vs_seen: StageSeen,
    ps: u64,
    ps_seen: Option<StageSeen>,
    state: super::exec::GfxState,
}

/// Learned graphics pipelines with their ranks, by program hash, and those
/// queued already.
#[derive(Default)]
struct GfxLearned {
    recipes: Vec<(u32, GfxRecipe)>,
    set: std::collections::HashSet<GfxRecipe>,
    by_hash: HashMap<u64, Vec<usize>>,
    queued: std::collections::HashSet<usize>,
}

static GFX: Mutex<Option<GfxLearned>> = Mutex::new(None);
/// Learned programs' code by code hash (`<store>.code`).
static CODES: Mutex<Option<HashMap<u64, std::sync::Arc<Vec<u32>>>>> = Mutex::new(None);
/// Compute programs whose learned specs were queued (by code hash): made
/// once per session, early or when their object was created.
static QUEUED_HASHES: Mutex<Option<std::collections::HashSet<u64>>> = Mutex::new(None);
/// Store lines read or written so far (ranks of new entries).
static LINES: AtomicU64 = AtomicU64::new(0);
/// Programs of created objects by code hash (their addresses).
static HASH_ADDR: Mutex<Option<HashMap<u64, u64>>> = Mutex::new(None);

/// [shader objects, first translations, with an object, guesses equal,
/// equal but for buffers of unknown format, then the first field that
/// differed: wave64, thread-group size, RSRC2, user data, resource count,
/// a buffer, an image's dimension, class or depth view, another resource,
/// another field; translations after the first; specs learned, specs
/// loaded].
static STATS: [AtomicU64; 21] = [const { AtomicU64::new(0) }; 21];
const OBJECTS: usize = 0;
const FIRSTS: usize = 1;
const WITH_OBJECT: usize = 2;
const EQUAL: usize = 3;
const UNKNOWN_FORMAT: usize = 4;
const DIFF: usize = 5;
const DIFF_NAMES: [&str; 10] = ["wave64", "thread-group size", "RSRC2", "user data", "resource count", "a buffer", "an image's dimension", "an image's class", "an image's depth view", "another field"];
const LATER: usize = 15;
const LEARNT: usize = 16;
const LOADED: usize = 17;
/// Programs' code loaded, kept this session, pipelines queued early.
const CODES_LOADED: usize = 18;
const CODES_KEPT: usize = 19;
const EARLY: usize = 20;
/// Dispatch modifier bit 15 against the wave64 of first translations:
/// [set and wave64, set and wave32] (the rest: clear).
static MODIFIER: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// Translations the dispatch path made itself for programs with learned
/// specs (`first_translation`): [any, one of whose learned specs gives the
/// same key spec (precompile not done with it yet, or it failed), none
/// does (the key spec differs)].
static MISSES: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// The fields in which two specs' Debug forms differ (diagnostics).
fn spec_diff(a: &Spec, b: &Spec) -> String {
    let (fa, fb) = (format!("{a:?}"), format!("{b:?}"));
    // Fields at top level: split on ", " outside brackets.
    let split = |s: &str| -> Vec<String> {
        let (mut out, mut cur, mut depth) = (Vec::new(), String::new(), 0i32);
        let body = s.trim_start_matches("Spec {").trim_end_matches('}');
        let chars: Vec<char> = body.chars().collect();
        let mut i = 0;
        while i < chars.len() {
            let c = chars[i];
            match c {
                '(' | '[' | '{' => depth += 1,
                ')' | ']' | '}' => depth -= 1,
                _ => {}
            }
            if c == ',' && depth == 0 {
                out.push(cur.trim().to_string());
                cur.clear();
            } else {
                cur.push(c);
            }
            i += 1;
        }
        if !cur.trim().is_empty() {
            out.push(cur.trim().to_string());
        }
        out
    };
    let (xa, xb) = (split(&fa), split(&fb));
    let diffs: Vec<String> = xa.iter().zip(xb.iter()).filter(|(x, y)| x != y).map(|(x, y)| format!("{x} | {}", y.split_once(':').map_or(y.as_str(), |(_, v)| v.trim()))).collect();
    if xa.len() != xb.len() {
        return format!("field counts {} and {}", xa.len(), xb.len());
    }
    diffs.join("; ")
}

/// Where learned specs are kept (`main`: DS_SPECS, else under local-data).
pub fn set_store(path: std::path::PathBuf) {
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let _ = STORE.set(path);
}

/// Reads the store once (compute specs and graphics pipelines; a line's
/// index is its rank).
fn load() {
    static DONE_LOADING: std::sync::Once = std::sync::Once::new();
    DONE_LOADING.call_once(|| {
        let mut m: HashMap<u64, (u32, Vec<Seen>)> = HashMap::new();
        let mut g = GfxLearned::default();
        if let Some(text) = STORE.get().and_then(|p| std::fs::read_to_string(p).ok()) {
            for (k, line) in text.lines().enumerate() {
                let rank = k as u32;
                if let Some(r) = line.strip_prefix("G ").and_then(parse_gfx) {
                    if g.set.insert(r.clone()) {
                        add_recipe(&mut g, rank, r);
                        STATS[LOADED].fetch_add(1, Ordering::Relaxed);
                    }
                } else if let Some((h, s)) = parse(line) {
                    let v = &mut m.entry(h).or_insert_with(|| (rank, Vec::new())).1;
                    if !v.contains(&s) {
                        v.push(s);
                        STATS[LOADED].fetch_add(1, Ordering::Relaxed);
                    }
                }
                LINES.store(k as u64 + 1, Ordering::Relaxed);
            }
        }
        let mut codes: HashMap<u64, std::sync::Arc<Vec<u32>>> = HashMap::new();
        if let Some(b) = code_path().and_then(|p| std::fs::read(p).ok()) {
            let mut at = 0usize;
            while at + 12 <= b.len() {
                let h = u64::from_le_bytes(b[at..at + 8].try_into().unwrap());
                let n = u32::from_le_bytes(b[at + 8..at + 12].try_into().unwrap()) as usize;
                let end = at + 12 + 4 * n;
                if n == 0 || end > b.len() {
                    break;
                }
                let words: Vec<u32> = b[at + 12..end].chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
                codes.insert(h, std::sync::Arc::new(words));
                at = end;
            }
        }
        STATS[CODES_LOADED].fetch_add(codes.len() as u64, Ordering::Relaxed);
        *LEARNED.lock() = Some(m);
        *GFX.lock() = Some(g);
        *CODES.lock() = Some(codes);
    });
}

/// The code file beside the store.
fn code_path() -> Option<std::path::PathBuf> {
    STORE.get().map(|p| p.with_extension("code"))
}

/// Keeps the code of the program at `addr` (code hash `h`) unless kept.
fn keep_code(addr: u64, h: u64) {
    load();
    if CODES.lock().as_ref().is_some_and(|m| m.contains_key(&h)) {
        return;
    }
    let Ok((words, hash)) = super::shaders::program_code(addr) else { return };
    if hash != h {
        return;
    }
    let mut rec = Vec::with_capacity(12 + 4 * words.len());
    rec.extend_from_slice(&h.to_le_bytes());
    rec.extend_from_slice(&(words.len() as u32).to_le_bytes());
    for w in &words {
        rec.extend_from_slice(&w.to_le_bytes());
    }
    {
        let mut g = CODES.lock();
        let m = g.get_or_insert_with(HashMap::new);
        if m.contains_key(&h) {
            return;
        }
        m.insert(h, std::sync::Arc::new(words));
    }
    STATS[CODES_KEPT].fetch_add(1, Ordering::Relaxed);
    if let Some(p) = code_path() {
        let _ = std::fs::OpenOptions::new().create(true).append(true).open(p).and_then(|mut f| f.write_all(&rec));
    }
}

/// The learned compute programs whose code is kept, with their specs
/// (`--compile-bench`).
pub(crate) fn learned_compute() -> Vec<(u64, std::sync::Arc<Vec<u32>>, Vec<Spec>)> {
    load();
    let codes = CODES.lock().clone().unwrap_or_default();
    learned(|m| {
        m.iter()
            .filter_map(|(h, (_, seen))| {
                let code = codes.get(h)?.clone();
                let specs = seen.iter().map(|s| super::shaders::compute_spec(0, *h, s.rsrc2, s.local, s.wave64, s.resources.clone(), s.batched)).collect();
                Some((*h, code, specs))
            })
            .collect()
    })
}

fn code_of(h: u64) -> Option<std::sync::Arc<Vec<u32>>> {
    load();
    CODES.lock().as_ref().and_then(|m| m.get(&h).cloned())
}

/// Most pipelines queued at the start of a session (`early`).
const EARLY_MAX: usize = 2500;

/// Queues the learned pipelines whose programs' code is kept, by rank, up
/// to EARLY_MAX (once, when the workers start).
fn early() {
    load();
    let codes: std::collections::HashSet<u64> = CODES.lock().as_ref().map(|m| m.keys().copied().collect()).unwrap_or_default();
    let mut jobs: Vec<(u32, Job)> = Vec::new();
    learned(|m| {
        for (h, (rank, _)) in m.iter() {
            if codes.contains(h) {
                jobs.push((*rank, Job::Code(*h)));
            }
        }
    });
    gfx_learned(|g| {
        for (i, (rank, r)) in g.recipes.iter().enumerate() {
            if codes.contains(&r.vs) && (r.ps == 0 || codes.contains(&r.ps)) {
                jobs.push((*rank, Job::Gfx(i)));
            }
        }
    });
    // The earliest used first; the rest (later areas, a growing store) when
    // their objects are created, so a session holds at most EARLY_MAX
    // pipelines it may not need.
    jobs.sort_unstable();
    jobs.truncate(EARLY_MAX);
    {
        let mut q = QUEUED_HASHES.lock();
        let q = q.get_or_insert_with(Default::default);
        jobs.retain(|(_, j)| match j {
            Job::Code(h) => q.insert(*h),
            _ => true,
        });
    }
    gfx_learned(|g| {
        jobs.retain(|(_, j)| match j {
            Job::Gfx(i) => g.queued.insert(*i),
            _ => true,
        })
    });
    STATS[EARLY].fetch_add(jobs.len() as u64, Ordering::Relaxed);
    let mut q = QUEUE.lock();
    for (r, j) in jobs {
        q.ready.push(std::cmp::Reverse((r, j)));
    }
    CV.notify_all();
}

fn add_recipe(g: &mut GfxLearned, rank: u32, r: GfxRecipe) {
    let i = g.recipes.len();
    g.by_hash.entry(r.vs).or_default().push(i);
    if r.ps != 0 && r.ps != r.vs {
        g.by_hash.entry(r.ps).or_default().push(i);
    }
    g.recipes.push((rank, r));
}

/// The learned compute specs.
fn learned<R>(f: impl FnOnce(&mut HashMap<u64, (u32, Vec<Seen>)>) -> R) -> R {
    load();
    let mut g = LEARNED.lock();
    f(g.get_or_insert_with(HashMap::new))
}

/// The learned graphics pipelines.
fn gfx_learned<R>(f: impl FnOnce(&mut GfxLearned) -> R) -> R {
    load();
    let mut g = GFX.lock();
    f(g.get_or_insert_with(GfxLearned::default))
}

/// Appends a line to the store.
fn store_line(line: &str) {
    LINES.fetch_add(1, Ordering::Relaxed);
    if let Some(p) = STORE.get() {
        let r = std::fs::OpenOptions::new().create(true).append(true).open(p).and_then(|mut f| f.write_all(line.as_bytes()));
        if let Err(e) = r {
            static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            if !WARNED.swap(true, Ordering::Relaxed) {
                crate::warn!("gpu: learned specs not stored in {} ({e})", p.display());
            }
        }
    }
}

/// The title created a compute shader object for the program at `addr`.
pub fn created(addr: u64, rsrc2: u32, local: [u32; 3], modifier: u32) {
    STATS[OBJECTS].fetch_add(1, Ordering::Relaxed);
    CREATED.lock().get_or_insert_with(HashMap::new).insert(addr, Created { rsrc2, local, modifier });
    if super::tune::precompile() >= 2 {
        QUEUE.lock().fresh.push_back(addr);
        CV.notify_one();
        ensure_workers();
    }
}

/// The title created a vertex (NGG) or pixel shader object for the program
/// at `addr`: learned graphics pipelines of its program are made once the
/// other stage's program has an object too.
pub fn created_graphics(addr: u64) {
    if super::tune::precompile() >= 2 {
        QUEUE.lock().fresh.push_back(addr);
        CV.notify_one();
        ensure_workers();
    }
}

/// Objects created and not looked at yet (in creation order), and programs
/// with specs to make, the earliest first dispatched first (by rank; the
/// guesses of `precompile` 3 after all learned ones).
struct Queues {
    fresh: std::collections::VecDeque<u64>,
    ready: std::collections::BinaryHeap<std::cmp::Reverse<(u32, Job)>>,
}

/// A pipeline job: a compute program by its object's address or by its
/// kept code (code hash), a learned graphics pipeline (index).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Job {
    Addr(u64),
    Code(u64),
    Gfx(usize),
}

static QUEUE: Mutex<Queues> = Mutex::new(Queues { fresh: std::collections::VecDeque::new(), ready: std::collections::BinaryHeap::new() });
static CV: Condvar = Condvar::new();
static WORKERS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
/// [pipelines created, their ns (translation included), present already,
/// objects with no spec to make, failed, objects done].
static DONE: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

thread_local! {
    /// This precompile thread's own handle (`boostable_thread`).
    static HANDLE: std::cell::Cell<Option<isize>> = const { std::cell::Cell::new(None) };
}

/// Raised priorities (`boost`), for the status line.
static BOOSTS: AtomicU64 = AtomicU64::new(0);

/// The calling thread's handle if it is a precompile thread (`exec`'s
/// pipeline flights record it).
pub(crate) fn boostable_thread() -> Option<isize> {
    HANDLE.with(|h| h.get())
}

/// A command processor waits for a pipeline a precompile thread is making:
/// that thread runs above normal until its job ends (`worker`).
pub(crate) fn boost(h: isize) {
    use windows_sys::Win32::System::Threading::{SetThreadPriority, THREAD_PRIORITY_ABOVE_NORMAL};
    unsafe { SetThreadPriority(h as _, THREAD_PRIORITY_ABOVE_NORMAL) };
    BOOSTS.fetch_add(1, Ordering::Relaxed);
}

/// The lowest priority again (after each job).
fn lower() {
    use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_LOWEST};
    unsafe { SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_LOWEST) };
}

fn ensure_workers() {
    let want = super::tune::precompile_workers().min(16) as usize;
    while WORKERS.load(Ordering::Acquire) < want {
        let n = WORKERS.fetch_add(1, Ordering::AcqRel);
        if n >= want {
            WORKERS.fetch_sub(1, Ordering::AcqRel);
            break;
        }
        let name = format!("gpu-precompile{n}");
        let r = crate::thread::spawn_host(
            &name.clone(),
            8 << 20,
            Box::new(move || {
                crate::log::set_thread_name(&name);
                crate::exception::arm_watch_current_thread(false);
                // Idle time only: the command processors and the title first.
                // A real handle (the pseudo-handle names the caller) lets a
                // waiting thread raise this one (`boost`).
                unsafe {
                    use windows_sys::Win32::System::Threading::{GetCurrentThreadId, OpenThread, THREAD_QUERY_LIMITED_INFORMATION, THREAD_SET_LIMITED_INFORMATION};
                    let h = OpenThread(THREAD_SET_LIMITED_INFORMATION | THREAD_QUERY_LIMITED_INFORMATION, 0, GetCurrentThreadId());
                    if !h.is_null() {
                        HANDLE.with(|c| c.set(Some(h as isize)));
                    }
                }
                lower();
                worker();
            }),
        );
        if let Err(e) = r {
            crate::warn!("gpu: no precompile worker ({e})");
            WORKERS.fetch_sub(1, Ordering::AcqRel);
            break;
        }
    }
}

fn worker() {
    // The title creates its first shader objects before the device exists.
    while super::vk::try_ctx().is_none() {
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    static EARLY_ONCE: std::sync::Once = std::sync::Once::new();
    EARLY_ONCE.call_once(early);
    loop {
        // Looking at a created object is cheap (its code hash), making a
        // pipeline is not: every object is looked at first.
        let (fresh, job) = {
            let mut q = QUEUE.lock();
            loop {
                if let Some(a) = q.fresh.pop_front() {
                    break (Some(a), None);
                } else if let Some(std::cmp::Reverse((_, j))) = q.ready.pop() {
                    break (None, Some(j));
                }
                CV.wait(&mut q);
            }
        };
        if let Some(a) = fresh {
            classify(a);
            continue;
        }
        match job {
            None => {}
            Some(Job::Addr(a)) => {
                run(a);
                DONE[5].fetch_add(1, Ordering::Relaxed);
            }
            Some(Job::Code(h)) => run_code(h),
            Some(Job::Gfx(i)) => run_gfx(i),
        }
        lower();
    }
}

/// Looks at a created object: its program's code hash, a compute program
/// with learned specs queued by its rank, graphics pipelines it completes
/// queued by theirs.
fn classify(addr: u64) {
    let Ok(code_hash) = super::shaders::program_hash_uncached(addr) else {
        DONE[4].fetch_add(1, Ordering::Relaxed);
        return;
    };
    HASH_ADDR.lock().get_or_insert_with(HashMap::new).insert(code_hash, addr);
    let compute = CREATED.lock().as_ref().is_some_and(|m| m.contains_key(&addr));
    if compute {
        match rank(code_hash) {
            Some(r) => {
                // Learned: its code kept for the next session's start.
                keep_code(addr, code_hash);
                if QUEUED_HASHES.lock().get_or_insert_with(Default::default).insert(code_hash) || super::tune::precompile() >= 3 {
                    QUEUE.lock().ready.push(std::cmp::Reverse((r, Job::Addr(addr))));
                    CV.notify_one();
                } else {
                    DONE[5].fetch_add(1, Ordering::Relaxed);
                }
            }
            None => {
                DONE[3].fetch_add(1, Ordering::Relaxed);
                DONE[5].fetch_add(1, Ordering::Relaxed);
            }
        }
        return;
    }
    if gfx_learned(|g| g.by_hash.contains_key(&code_hash)) {
        keep_code(addr, code_hash);
    }
    let jobs: Vec<(u32, usize)> = gfx_learned(|g| {
        let mut v = Vec::new();
        let Some(ids) = g.by_hash.get(&code_hash) else { return v };
        // (Lock order: the learned pipelines, then the addresses.)
        let present = HASH_ADDR.lock();
        let has = |h: u64| present.as_ref().is_some_and(|m| m.contains_key(&h));
        for &i in ids {
            let (rank, r) = &g.recipes[i];
            if has(r.vs) && (r.ps == 0 || has(r.ps)) && !g.queued.contains(&i) {
                v.push((*rank, i));
            }
        }
        drop(present);
        for &(_, i) in &v {
            g.queued.insert(i);
        }
        v
    });
    if !jobs.is_empty() {
        let mut q = QUEUE.lock();
        for (r, i) in jobs {
            q.ready.push(std::cmp::Reverse((r, Job::Gfx(i))));
        }
        CV.notify_all();
    }
}

/// Makes the learned graphics pipeline `i` from the programs of the
/// objects created for its stages.
fn run_gfx(i: usize) {
    let Some(r) = gfx_learned(|g| g.recipes.get(i).map(|e| e.1.clone())) else { return };
    let (vs_addr, ps_addr) = {
        let m = HASH_ADDR.lock();
        let m = m.as_ref();
        (m.and_then(|m| m.get(&r.vs).copied()).unwrap_or(0), m.and_then(|m| m.get(&r.ps).copied()).unwrap_or(0))
    };
    // A stage's translation: from the kept code, else the object's program.
    let stage = |h: u64, addr: u64, spec: Spec| -> Result<std::sync::Arc<super::shaders::Translated>, String> {
        match code_of(h) {
            Some(code) => super::shaders::translate_code_for_spec(&code, h, spec),
            None if addr != 0 => super::shaders::translate_for_spec(addr, spec),
            None => Err("no code".into()),
        }
    };
    let t = std::time::Instant::now();
    let made = (|| -> Result<Option<bool>, String> {
        let v = &r.vs_seen;
        let Some(vs_spec) = super::shaders::vertex_spec(vs_addr, v.user_sgprs, v.wave64, v.base_instance, v.resources.clone()) else { return Ok(None) };
        let vs_tr = stage(r.vs, vs_addr, vs_spec)?;
        let ps_spec = |p: &StageSeen, measure: bool| super::shaders::pixel_spec(ps_addr, p.user_sgprs, p.wave64, p.ps_input_addr, p.ps_inputs.clone(), p.mrt, p.early_tests, p.early_ok, p.resources.clone(), measure);
        let ps_tr = match &r.ps_seen {
            Some(p) if r.ps != 0 => Some((stage(r.ps, ps_addr, ps_spec(p, p.lod_measure))?, ps_addr)),
            _ => None,
        };
        let made = super::exec::precompile_gfx_translations((&vs_tr, vs_addr), ps_tr.as_ref().map(|(t, a)| (&**t, *a)), &r.state);
        if made == Some(true) && r.ps_seen.as_ref().is_some_and(|p| p.lod_measure) {
            GFX_DONE[4].fetch_add(1, Ordering::Relaxed);
        }
        Ok(made)
    })();
    match made {
        Ok(Some(true)) => {
            GFX_DONE[0].fetch_add(1, Ordering::Relaxed);
            GFX_DONE[1].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
        Ok(Some(false)) => {
            GFX_DONE[2].fetch_add(1, Ordering::Relaxed);
        }
        Ok(None) | Err(_) => {
            if GFX_DONE[3].fetch_add(1, Ordering::Relaxed) < 8 {
                crate::info!("gpu: precompile of graphics pipeline vs {:#x} ps {:#x} failed: {}", r.vs, r.ps, made.err().unwrap_or_default());
            }
        }
    }
}

/// Graphics pipelines made from objects: [created, their ns, present
/// already, failed, of the created those of draws measuring LOD statistics
/// in rotation].
static GFX_DONE: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];

/// A graphics pipeline was created on a draw's path (`exec::draw`): its
/// stages' specs and state are learned (once).
pub(crate) fn learn_gfx(vs: (&super::shaders::Translated, u64), ps: Option<(&super::shaders::Translated, u64)>, state: &super::exec::GfxState) {
    let Ok(vs_hash) = super::shaders::program_hash(vs.1) else { return };
    let ps_hash = match ps {
        Some(p) => match super::shaders::program_hash(p.1) {
            Ok(h) => h,
            Err(_) => return,
        },
        None => 0,
    };
    let r = GfxRecipe { vs: vs_hash, vs_seen: StageSeen::of(&vs.0.spec), ps: ps_hash, ps_seen: ps.map(|p| StageSeen::of(&p.0.spec)), state: *state };
    let new = gfx_learned(|g| {
        if g.set.contains(&r) {
            return false;
        }
        g.set.insert(r.clone());
        let rank = LINES.load(Ordering::Relaxed) as u32;
        add_recipe(g, rank, r.clone());
        // Made (or being made) now: not queued again this session.
        let i = g.recipes.len() - 1;
        g.queued.insert(i);
        true
    });
    if new {
        STATS[LEARNT].fetch_add(1, Ordering::Relaxed);
        store_line(&format_gfx(&r));
        keep_code(vs.1, vs_hash);
        if let Some(p) = ps {
            keep_code(p.1, ps_hash);
        }
    }
}

/// The rank of the compute program `code_hash` among those to make: its
/// learned rank, after all of them with `precompile` 3; None: nothing to
/// make.
fn rank(code_hash: u64) -> Option<u32> {
    match learned(|m| m.get(&code_hash).map(|e| e.0)) {
        Some(r) => Some(r),
        None if super::tune::precompile() >= 3 => Some(u32::MAX),
        None => None,
    }
}

/// Makes the pipelines of the program at `addr`: for the specs its
/// dispatches used before, and (`precompile` 3) for its object's guess.
fn run(addr: u64) {
    let Some(c) = CREATED.lock().as_ref().and_then(|m| m.get(&addr).copied()) else { return };
    let Ok(code_hash) = super::shaders::program_hash(addr) else {
        DONE[4].fetch_add(1, Ordering::Relaxed);
        return;
    };
    let seen = learned(|m| m.get(&code_hash).map(|e| e.1.clone())).unwrap_or_default();
    let mut specs: Vec<Spec> = seen.into_iter().map(|s| super::shaders::compute_spec(addr, code_hash, s.rsrc2, s.local, s.wave64, s.resources, s.batched)).collect();
    if super::tune::precompile() >= 3 {
        let user_sgprs = ((c.rsrc2 >> 1) & 0x1F).min(16) as u16;
        if let Ok((insts, plan, _)) = super::shaders::stage_plan(addr, user_sgprs, 0) {
            let (g, known) = super::shaders::guess_compute_spec(addr, &insts, &plan, code_hash, c.rsrc2, c.local);
            if known && !specs.contains(&g) {
                specs.push(g);
            }
        }
    }
    if specs.is_empty() {
        DONE[3].fetch_add(1, Ordering::Relaxed);
        return;
    }
    for spec in specs {
        let t = std::time::Instant::now();
        let r = super::shaders::translate_for_spec(addr, spec).and_then(|tr| super::exec::precompile_translation(&tr, addr).ok_or_else(|| "pipeline creation failed".to_string()));
        match r {
            Ok(true) => {
                DONE[0].fetch_add(1, Ordering::Relaxed);
                DONE[1].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
            }
            Ok(false) => {
                DONE[2].fetch_add(1, Ordering::Relaxed);
            }
            Err(e) => {
                if DONE[4].fetch_add(1, Ordering::Relaxed) < 8 {
                    crate::info!("gpu: precompile of cs {addr:#x} failed: {e}");
                }
            }
        }
    }
}

/// Makes the learned compute pipelines of a program from its kept code.
fn run_code(h: u64) {
    let Some(code) = code_of(h) else { return };
    let seen = learned(|m| m.get(&h).map(|e| e.1.clone())).unwrap_or_default();
    for s in seen {
        let t = std::time::Instant::now();
        let spec = super::shaders::compute_spec(0, h, s.rsrc2, s.local, s.wave64, s.resources, s.batched);
        let r = super::shaders::translate_code_for_spec(&code, h, spec).and_then(|tr| super::exec::precompile_translation(&tr, 0).ok_or_else(|| "pipeline creation failed".to_string()));
        match r {
            Ok(true) => {
                DONE[0].fetch_add(1, Ordering::Relaxed);
                DONE[1].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
            }
            Ok(false) => {
                DONE[2].fetch_add(1, Ordering::Relaxed);
            }
            Err(e) => {
                if DONE[4].fetch_add(1, Ordering::Relaxed) < 8 {
                    crate::info!("gpu: precompile of cs {h:#018x} from its kept code failed: {e}");
                }
            }
        }
    }
}

/// Format-accessed buffers of first translations by (V# format, dst_sel).
static FORMATS: Mutex<Option<HashMap<(u8, u16), u64>>> = Mutex::new(None);

/// A compute program's translation is missing (`shaders::prepare`, before
/// translating with `actual`, the code base left out): its spec is learned,
/// and the first time per program it is compared with the guess from its
/// shader object.
pub fn first_translation(addr: u64, insts: &[rdna::Inst], plan: &rdna::analysis::Plan, code_hash: u64, actual: &Spec) {
    // A program whose specs were learned: why its translation was not made
    // ahead (the same key spec from a learned one, or none).
    if let Some(seen) = learned(|m| m.get(&code_hash).map(|e| e.1.clone())) {
        MISSES[0].fetch_add(1, Ordering::Relaxed);
        let mut key = actual.clone();
        key.code_base = 0;
        let made: Vec<Spec> = seen.into_iter().map(|s| super::shaders::compute_spec(0, code_hash, s.rsrc2, s.local, s.wave64, s.resources, s.batched)).collect();
        if made.iter().any(|m| *m == key) {
            MISSES[1].fetch_add(1, Ordering::Relaxed);
        } else {
            let n = MISSES[2].fetch_add(1, Ordering::Relaxed);
            if n < 12 {
                let best = made.iter().map(|m| spec_diff(m, &key)).min_by_key(|d| d.len()).unwrap_or_default();
                crate::info!("gpu: cs {addr:#x} (code hash {code_hash:#018x}) translated at its dispatch: none of its {} learned specs gives its key spec (closest differs in: {best})", made.len());
            }
        }
    }
    LEARN_ADDR.with(|c| c.set(Some(addr)));
    learn(code_hash, actual);
    LEARN_ADDR.with(|c| c.set(None));
    if !COMPARED.lock().get_or_insert_with(Default::default).insert(addr) {
        STATS[LATER].fetch_add(1, Ordering::Relaxed);
        return;
    }
    STATS[FIRSTS].fetch_add(1, Ordering::Relaxed);
    let Some(c) = CREATED.lock().as_ref().and_then(|m| m.get(&addr).copied()) else { return };
    STATS[WITH_OBJECT].fetch_add(1, Ordering::Relaxed);
    let k = if c.modifier & (1 << 15) != 0 { 0 } else { 2 } + if actual.wave64 { 0 } else { 1 };
    MODIFIER[k].fetch_add(1, Ordering::Relaxed);
    let (guess, _) = super::shaders::guess_compute_spec(addr, insts, plan, code_hash, c.rsrc2, c.local);
    if guess == *actual {
        STATS[EQUAL].fetch_add(1, Ordering::Relaxed);
        return;
    }
    for (g, a) in guess.resources.iter().zip(actual.resources.iter()) {
        if let (ResSpec::Buffer { format: 0, dst_sel: 0xFAC }, ResSpec::Buffer { format, dst_sel }) = (g, a) {
            *FORMATS.lock().get_or_insert_with(HashMap::new).entry((*format, *dst_sel)).or_insert(0) += 1;
        }
    }
    let why = differs(&guess, actual);
    if why == usize::MAX {
        STATS[UNKNOWN_FORMAT].fetch_add(1, Ordering::Relaxed);
        return;
    }
    let n = STATS[DIFF + why].fetch_add(1, Ordering::Relaxed);
    if n < 2 {
        crate::info!("gpu: precompile guess for cs {addr:#x} differs in {}: guess resources {:?}; actual {:?}", DIFF_NAMES[why], guess.resources, actual.resources);
    }
}

/// Records the spec a dispatch of the program `code_hash` used (once per
/// distinct spec; appended to the store).
fn learn(code_hash: u64, spec: &Spec) {
    let s = Seen { rsrc2: spec.cs_rsrc2, local: spec.cs_local_size, wave64: spec.wave64, resources: spec.resources.clone(), batched: spec.batched };
    let new = learned(|m| {
        let rank = LINES.load(Ordering::Relaxed) as u32;
        let v = &mut m.entry(code_hash).or_insert_with(|| (rank, Vec::new())).1;
        if v.contains(&s) {
            false
        } else {
            v.push(s.clone());
            true
        }
    });
    if !new {
        return;
    }
    STATS[LEARNT].fetch_add(1, Ordering::Relaxed);
    store_line(&format_line(code_hash, &s));
    if let Some(a) = LEARN_ADDR.with(|c| c.get()) {
        keep_code(a, code_hash);
    }
}

thread_local! {
    /// The program being learned (`first_translation`): its code is kept.
    static LEARN_ADDR: std::cell::Cell<Option<u64>> = const { std::cell::Cell::new(None) };
}

const DIMS: [ImageDim; 9] = [ImageDim::D1, ImageDim::D2, ImageDim::D3, ImageDim::Cube, ImageDim::D1Array, ImageDim::D2Array, ImageDim::D2Msaa, ImageDim::D2MsaaArray, ImageDim::CubeArray];
const CLASSES: [NumClass; 3] = [NumClass::Float, NumClass::Sint, NumClass::Uint];

/// One store line: code hash, RSRC2, thread-group size, wave64, then the
/// resources (`b<format>/<dst_sel>`, `g`, `s`, `i<dim>/<class>/<depth>`).
fn format_line(code_hash: u64, s: &Seen) -> String {
    let res: Vec<String> = s
        .resources
        .iter()
        .map(|r| match r {
            ResSpec::Buffer { format, dst_sel } => format!("b{format}/{dst_sel}"),
            ResSpec::Global => "g".into(),
            ResSpec::Sampler => "s".into(),
            ResSpec::Image { dim, class, depth } => format!("i{}/{}/{}", DIMS.iter().position(|d| d == dim).unwrap_or(1), CLASSES.iter().position(|c| c == class).unwrap_or(0), *depth as u8),
        })
        .collect();
    // (A final `B`: batched.)
    format!("{code_hash:016x} {:x} {} {} {} {} {}{}\n", s.rsrc2, s.local[0], s.local[1], s.local[2], s.wave64 as u8, res.join(" "), if s.batched { " B" } else { "" })
}

fn parse(line: &str) -> Option<(u64, Seen)> {
    let mut it = line.split_ascii_whitespace();
    let h = u64::from_str_radix(it.next()?, 16).ok()?;
    let rsrc2 = u32::from_str_radix(it.next()?, 16).ok()?;
    let local = [it.next()?.parse().ok()?, it.next()?.parse().ok()?, it.next()?.parse().ok()?];
    let wave64 = match it.next()? {
        "1" => true,
        "0" => false,
        _ => return None,
    };
    let mut resources = Vec::new();
    let mut batched = false;
    for t in it {
        if t == "B" {
            batched = true;
            continue;
        }
        let (k, rest) = t.split_at(1);
        let nums = rest.split('/').filter(|x| !x.is_empty()).map(|x| x.parse::<u32>().ok()).collect::<Option<Vec<u32>>>()?;
        resources.push(match (k, nums.as_slice()) {
            ("b", [f, d]) => ResSpec::Buffer { format: u8::try_from(*f).ok()?, dst_sel: u16::try_from(*d).ok()? },
            ("g", []) => ResSpec::Global,
            ("s", []) => ResSpec::Sampler,
            ("i", [d, c, z]) => ResSpec::Image { dim: *DIMS.get(*d as usize)?, class: *CLASSES.get(*c as usize)?, depth: *z != 0 },
            _ => return None,
        });
    }
    Some((h, Seen { rsrc2, local, wave64, resources, batched }))
}

fn res_tokens(res: &[ResSpec]) -> Vec<String> {
    res.iter()
        .map(|r| match r {
            ResSpec::Buffer { format, dst_sel } => format!("b{format}/{dst_sel}"),
            ResSpec::Global => "g".into(),
            ResSpec::Sampler => "s".into(),
            ResSpec::Image { dim, class, depth } => format!("i{}/{}/{}", DIMS.iter().position(|d| d == dim).unwrap_or(1), CLASSES.iter().position(|c| c == class).unwrap_or(0), *depth as u8),
        })
        .collect()
}

fn res_token(t: &str) -> Option<ResSpec> {
    let (k, rest) = t.split_at(1);
    let nums = rest.split('/').filter(|x| !x.is_empty()).map(|x| x.parse::<u32>().ok()).collect::<Option<Vec<u32>>>()?;
    Some(match (k, nums.as_slice()) {
        ("b", [f, d]) => ResSpec::Buffer { format: u8::try_from(*f).ok()?, dst_sel: u16::try_from(*d).ok()? },
        ("g", []) => ResSpec::Global,
        ("s", []) => ResSpec::Sampler,
        ("i", [d, c, z]) => ResSpec::Image { dim: *DIMS.get(*d as usize)?, class: *CLASSES.get(*c as usize)?, depth: *z != 0 },
        _ => return None,
    })
}

/// A stage: user data, wave64, base-instance SGPR (- none), PS_INPUT_ADDR,
/// early tests (bit 0 asked for, bit 1 allowed: `Spec::early_ok`; bit 2 a
/// draw measuring LOD statistics in rotation), `m` the targets' (export format, class) pairs, then `p`
/// pixel inputs (param/flat/default, 9 none) and the resources.
fn format_stage(s: &StageSeen) -> String {
    let mrt: Vec<String> = s.mrt.iter().flat_map(|m| [m.export_format.to_string(), m.class.to_string()]).collect();
    let inputs = s.ps_inputs.iter().map(|i| format!("p{}/{}/{}", i.param, i.flat as u8, i.default.unwrap_or(9)));
    let mut t = vec![s.user_sgprs.to_string(), (s.wave64 as u8).to_string(), s.base_instance.map_or("-".to_string(), |b| b.to_string()), format!("{:x}", s.ps_input_addr), (s.early_tests as u8 | (s.early_ok as u8) << 1 | (s.lod_measure as u8) << 2).to_string(), format!("m{}", mrt.join("/"))];
    t.extend(inputs);
    t.extend(res_tokens(&s.resources));
    t.join(" ")
}

fn parse_stage(text: &str) -> Option<StageSeen> {
    let mut it = text.split_ascii_whitespace();
    let user_sgprs = it.next()?.parse().ok()?;
    let wave64 = it.next()? == "1";
    let base_instance = match it.next()? {
        "-" => None,
        b => Some(b.parse().ok()?),
    };
    let ps_input_addr = u32::from_str_radix(it.next()?, 16).ok()?;
    let early: u8 = it.next()?.parse().ok()?;
    let (early_tests, early_ok, lod_measure) = (early & 1 != 0, early & 2 != 0, early & 4 != 0);
    let m: Vec<u8> = it.next()?.strip_prefix('m')?.split('/').map(|x| x.parse().ok()).collect::<Option<Vec<u8>>>()?;
    if m.len() != 16 {
        return None;
    }
    let mrt: [MrtSpec; 8] = std::array::from_fn(|k| MrtSpec { export_format: m[2 * k], class: m[2 * k + 1] });
    let (mut ps_inputs, mut resources) = (Vec::new(), Vec::new());
    for t in it {
        if let Some(p) = t.strip_prefix('p') {
            let v: Vec<u8> = p.split('/').map(|x| x.parse().ok()).collect::<Option<Vec<u8>>>()?;
            let [param, flat, default] = v.as_slice() else { return None };
            ps_inputs.push(PsInput { param: *param, flat: *flat != 0, default: (*default != 9).then_some(*default) });
        } else {
            resources.push(res_token(t)?);
        }
    }
    Some(StageSeen { user_sgprs, wave64, base_instance, ps_input_addr, ps_inputs, mrt, early_tests, early_ok, lod_measure, resources })
}

/// `G <vs hash> <ps hash> ; <vertex stage> ; <pixel stage | -> ; <state>`;
/// the state: rect, colors, blend (hex), write masks, depth format,
/// primitive class, depth clamp.
fn format_gfx(r: &GfxRecipe) -> String {
    let st = &r.state;
    let join = |v: Vec<String>| v.join("/");
    format!(
        "G {:016x} {:016x} ; {} ; {} ; {} {} {} {} {} {} {}\n",
        r.vs,
        r.ps,
        format_stage(&r.vs_seen),
        r.ps_seen.as_ref().map_or("-".to_string(), format_stage),
        st.rect as u8,
        join(st.colors.iter().map(|c| c.to_string()).collect()),
        join(st.blend.iter().map(|b| format!("{b:x}")).collect()),
        join(st.masks.iter().map(|m| m.to_string()).collect()),
        st.depth,
        st.topology_class,
        st.depth_clamp as u8
    )
}

fn parse_gfx(text: &str) -> Option<GfxRecipe> {
    let parts: Vec<&str> = text.split(" ; ").collect();
    let [head, vs, ps, state] = parts.as_slice() else { return None };
    let mut h = head.split_ascii_whitespace();
    let vs_hash = u64::from_str_radix(h.next()?, 16).ok()?;
    let ps_hash = u64::from_str_radix(h.next()?, 16).ok()?;
    let vs_seen = parse_stage(vs)?;
    let ps_seen = if ps.trim() == "-" { None } else { Some(parse_stage(ps)?) };
    let mut it = state.split_ascii_whitespace();
    let rect = it.next()? == "1";
    let colors: Vec<i32> = it.next()?.split('/').map(|x| x.parse().ok()).collect::<Option<Vec<i32>>>()?;
    let blend: Vec<u32> = it.next()?.split('/').map(|x| u32::from_str_radix(x, 16).ok()).collect::<Option<Vec<u32>>>()?;
    let masks: Vec<u8> = it.next()?.split('/').map(|x| x.parse().ok()).collect::<Option<Vec<u8>>>()?;
    let depth = it.next()?.parse().ok()?;
    let topology_class = it.next()?.parse().ok()?;
    let depth_clamp = it.next()? == "1";
    let state = super::exec::GfxState { rect, colors: colors.try_into().ok()?, blend: blend.try_into().ok()?, masks: masks.try_into().ok()?, depth, topology_class, depth_clamp };
    Some(GfxRecipe { vs: vs_hash, vs_seen, ps: ps_hash, ps_seen, state })
}

/// The first field (`DIFF_NAMES` index) in which `g` and `a` differ;
/// usize::MAX when only buffers the guess left unknown differ.
fn differs(g: &Spec, a: &Spec) -> usize {
    if g.wave64 != a.wave64 {
        return 0;
    }
    if g.cs_local_size != a.cs_local_size {
        return 1;
    }
    if g.cs_rsrc2 != a.cs_rsrc2 {
        return 2;
    }
    if g.user_sgprs != a.user_sgprs {
        return 3;
    }
    if g.resources.len() != a.resources.len() {
        return 4;
    }
    let mut unknown_only = false;
    for (x, y) in g.resources.iter().zip(a.resources.iter()) {
        if x == y {
            continue;
        }
        match (x, y) {
            // The guess's marker for a format-accessed buffer.
            (ResSpec::Buffer { format: 0, dst_sel: 0xFAC }, ResSpec::Buffer { .. }) => unknown_only = true,
            (ResSpec::Buffer { .. }, _) => return 5,
            (ResSpec::Image { dim: d1, class: c1, .. }, ResSpec::Image { dim: d2, class: c2, .. }) => {
                return if d1 != d2 {
                    6
                } else if c1 != c2 {
                    7
                } else {
                    8
                };
            }
            _ => return 9,
        }
    }
    let mut g2 = g.clone();
    let mut a2 = a.clone();
    g2.resources.clear();
    a2.resources.clear();
    if g2 != a2 {
        return 9;
    }
    if unknown_only {
        usize::MAX
    } else {
        9
    }
}

/// One status line ("" when off).
pub fn status() -> String {
    let s: Vec<u64> = STATS.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    if s[OBJECTS] == 0 || super::tune::precompile() == 0 {
        return String::new();
    }
    let diffs: Vec<String> = DIFF_NAMES.iter().enumerate().filter(|(k, _)| s[DIFF + k] > 0).map(|(k, n)| format!("{n} {}", s[DIFF + k])).collect();
    let m: Vec<u64> = MODIFIER.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    let d: Vec<u64> = DONE.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    let made = if super::tune::precompile() >= 2 {
        format!(
            "; made from objects ({} of them done, {} waiting): {} pipelines ({:.1} s), {} present already, {} objects with no spec to make, {} failed, {} raised for a waiting thread",
            d[5],
            {
                let q = QUEUE.lock();
                q.fresh.len() + q.ready.len()
            },
            d[0],
            d[1] as f64 / 1e9,
            d[2],
            d[3],
            d[4],
            BOOSTS.load(Ordering::Relaxed)
        )
    } else {
        String::new()
    };
    let made = if super::tune::precompile() >= 2 {
        let g: Vec<u64> = GFX_DONE.iter().map(|c| c.load(Ordering::Relaxed)).collect();
        format!("{made}; graphics pipelines made from objects: {} ({:.1} s), {} present already, {} failed ({} created for draws measuring LOD statistics in rotation)", g[0], g[1] as f64 / 1e9, g[2], g[3], g[4])
    } else {
        made
    };
    let mut f: Vec<((u8, u16), u64)> = FORMATS.lock().as_ref().map(|m| m.iter().map(|(k, v)| (*k, *v)).collect()).unwrap_or_default();
    f.sort_by(|a, b| b.1.cmp(&a.1));
    let f: Vec<String> = f.iter().take(12).map(|((fm, ds), n)| format!("{fm}/{ds:#x} {n}")).collect();
    format!(
        "    compute shader objects (`precompile`): {} created; specs {} loaded, {} learned; translated at their dispatch with specs learned {} ({} of them with a learned spec giving the same key spec, {} with none); programs' code {} loaded, {} kept; {} jobs queued at the start; {} programs translated first ({} later translations with another spec), {} of them from an object: guess equal {}, equal but for format-accessed buffers {} ({}), different {} ({}); dispatch modifier bit 15 set: wave64 {}, wave32 {}; clear: wave64 {}, wave32 {}{made}\n",
        s[OBJECTS],
        s[LOADED],
        s[LEARNT],
        MISSES[0].load(Ordering::Relaxed),
        MISSES[1].load(Ordering::Relaxed),
        MISSES[2].load(Ordering::Relaxed),
        s[CODES_LOADED],
        s[CODES_KEPT],
        s[EARLY],
        s[FIRSTS],
        s[LATER],
        s[WITH_OBJECT],
        s[EQUAL],
        s[UNKNOWN_FORMAT],
        f.join(", "),
        (DIFF..DIFF + 10).map(|k| s[k]).sum::<u64>(),
        diffs.join(", "),
        m[0],
        m[1],
        m[2],
        m[3]
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn store_lines_round_trip() {
        let s = Seen {
            rsrc2: 0x8984,
            local: [8, 8, 1],
            wave64: true,
            resources: vec![
                ResSpec::Image { dim: ImageDim::D2Array, class: NumClass::Uint, depth: false },
                ResSpec::Buffer { format: 20, dst_sel: 4 },
                ResSpec::Global,
                ResSpec::Sampler,
                ResSpec::Image { dim: ImageDim::CubeArray, class: NumClass::Float, depth: true },
            ],
            batched: true,
        };
        let line = format_line(0x1234_5678_9abc_def0, &s);
        let (h, t) = parse(line.trim_end()).unwrap();
        assert_eq!(h, 0x1234_5678_9abc_def0);
        assert!(t == s);
        assert!(parse("zz 1 2 3 4 1").is_none());
        assert!(parse("12 84 64 1 1 1 q7").is_none());
    }

    #[test]
    fn graphics_lines_round_trip() {
        let stage = |ps: bool| StageSeen {
            user_sgprs: 12,
            wave64: !ps,
            base_instance: (!ps).then_some(11),
            ps_input_addr: if ps { 0x2 } else { 0 },
            ps_inputs: if ps { vec![PsInput { param: 0, flat: false, default: None }, PsInput { param: 3, flat: true, default: Some(2) }] } else { Vec::new() },
            mrt: std::array::from_fn(|k| MrtSpec { export_format: if ps && k < 2 { 9 } else { 0 }, class: if k == 1 { 2 } else { 0 } }),
            early_tests: ps,
            early_ok: ps,
            lod_measure: ps,
            resources: vec![ResSpec::Buffer { format: 0, dst_sel: 0 }, ResSpec::Image { dim: ImageDim::D2, class: NumClass::Float, depth: false }, ResSpec::Sampler],
        };
        let r = GfxRecipe {
            vs: 0x1111,
            vs_seen: stage(false),
            ps: 0x2222,
            ps_seen: Some(stage(true)),
            state: super::super::exec::GfxState { rect: true, colors: [44, 37, 0, 0, 0, 0, 0, 0], blend: [0x4000_0123, 0, 0, 0, 0, 0, 0, 0], masks: [15, 7, 0, 0, 0, 0, 0, 0], depth: 130, topology_class: 2, depth_clamp: true },
        };
        let line = format_gfx(&r);
        let back = parse_gfx(line.trim_end().strip_prefix("G ").unwrap()).unwrap();
        assert!(back == r);
        let mut nops = r.clone();
        nops.ps = 0;
        nops.ps_seen = None;
        let line = format_gfx(&nops);
        assert!(parse_gfx(line.trim_end().strip_prefix("G ").unwrap()).unwrap() == nops);
    }
}
