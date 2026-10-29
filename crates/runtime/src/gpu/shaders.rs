//! Shader front end: program decoding (cached), resolution of descriptors and
//! scalar loads against guest state, specialization and translation to SPIR-V
//! (cached by program contents and specialization).

use super::cp::{Cp, Space};
use super::desc::{Tsharp, Vsharp};
use parking_lot::Mutex;
use rdna::analysis::{Plan, Prov, Ptr, ResKind};
use rdna::translate::{self, ImageDim, MrtSpec, NumClass, Output, PsInput, ResSpec, Spec, Stage};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

struct Program {
    words: usize,
    hash: u64,
    insts: Arc<Vec<rdna::Inst>>,
}

static PROGRAMS: Mutex<Option<HashMap<u64, Program>>> = Mutex::new(None);
static PLANS: Mutex<Option<HashMap<u64, Arc<Plan>>>> = Mutex::new(None);
static TRANSLATIONS: Mutex<Option<HashMap<(u64, u64), Arc<Translated>>>> = Mutex::new(None);
/// Translations at least this large are logged with their program saved
/// (`prepare`), once per program address.
const BIG_SPIRV_WORDS: usize = 40_000;
static BIG_SPIRV_SEEN: Mutex<Option<std::collections::HashSet<u64>>> = Mutex::new(None);
thread_local! {
    /// Preparations made inside `with_gpu_loads` read their data-only loads
    /// on the GPU whatever `tune gpu_loads` says.
    static FORCE_GPU_LOADS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Vertex stages prepared inside `with_base_instance`: the user-data
    /// word that receives the draw's start instance on the GPU.
    static BASE_INSTANCE: std::cell::Cell<Option<u16>> = const { std::cell::Cell::new(None) };
    /// Draw stages prepared while a `BatchedDraw` guard is held: translated
    /// batched (`Spec::batched`, buffers by device address).
    static BATCHED_DRAW: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Batched draw stages prepared while a `GpuDrawMasks` guard is held:
    /// their GPU-resolved loads and buffers (`Spec::gpu_loads`, `gpu_res`;
    /// vertex stage, pixel stage).
    static GPU_DRAW_MASKS: std::cell::Cell<Option<[(u64, u64); 2]>> = const { std::cell::Cell::new(None) };
}

/// While held, the batched draw stages this thread prepares resolve the
/// given loads and buffers on the GPU (a GPU-resolved pair, `gpures`).
pub struct GpuDrawMasks(bool);

pub fn gpu_draw_masks(m: Option<[(u64, u64); 2]>) -> GpuDrawMasks {
    if m.is_some() {
        GPU_DRAW_MASKS.with(|c| c.set(m));
    }
    GpuDrawMasks(m.is_some())
}

impl Drop for GpuDrawMasks {
    fn drop(&mut self) {
        if self.0 {
            GPU_DRAW_MASKS.with(|c| c.set(None));
        }
    }
}

/// The GPU-resolved masks of a batched draw stage being prepared (0:
/// vertex, 1: pixel), when the page table exists.
fn gpu_draw_masks_now(stage: usize) -> Option<(u64, u64)> {
    let m = GPU_DRAW_MASKS.with(|c| c.get())?;
    (super::pagetable::address() != 0).then_some(m[stage])
}

/// While held, the draw stages this thread prepares are translated batched
/// (a batched draw run, `exec::draw`); cleared when dropped.
pub struct BatchedDraw(bool);

pub fn batched_draw(on: bool) -> BatchedDraw {
    if on {
        BATCHED_DRAW.with(|c| c.set(true));
    }
    BatchedDraw(on)
}

impl Drop for BatchedDraw {
    fn drop(&mut self) {
        if self.0 {
            BATCHED_DRAW.with(|c| c.set(false));
        }
    }
}

fn batched_draw_now() -> bool {
    BATCHED_DRAW.with(|c| c.get())
}

/// Program pairs (vertex, pixel) whose consecutive draws with the same
/// context state run batched (`mark_draw_batchable`), and pairs refused
/// (their stages write memory or use the varying's location).
static DRAW_PAIRS: parking_lot::RwLock<Option<(std::collections::HashSet<(u64, u64)>, std::collections::HashSet<(u64, u64)>)>> = parking_lot::RwLock::new(None);
static DRAW_PAIRS_ANY: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn draw_batchable(vs: u64, ps: u64) -> bool {
    DRAW_PAIRS_ANY.load(std::sync::atomic::Ordering::Relaxed) && DRAW_PAIRS.read().as_ref().is_some_and(|(on, _)| on.contains(&(vs, ps)))
}

/// Pair (vs, ps) is batched from now on (unless refused before).
pub fn mark_draw_batchable(vs: u64, ps: u64) {
    let mut g = DRAW_PAIRS.write();
    let (on, refused) = g.get_or_insert_with(Default::default);
    if refused.contains(&(vs, ps)) || !on.insert((vs, ps)) {
        return;
    }
    DRAW_PAIRS_ANY.store(true, std::sync::atomic::Ordering::Relaxed);
    if on.len() <= 64 || on.len() % 64 == 0 {
        crate::info!("gpu: draws of vs {vs:#x} ps {ps:#x} batched from now on ({} pairs)", on.len());
    }
}

/// Pair (vs, ps) cannot run batched: never again.
pub fn refuse_draw_batchable(vs: u64, ps: u64, why: &str) {
    let mut g = DRAW_PAIRS.write();
    let (on, refused) = g.get_or_insert_with(Default::default);
    on.remove(&(vs, ps));
    if refused.insert((vs, ps)) && refused.len() <= 40 {
        crate::info!("gpu: draws of vs {vs:#x} ps {ps:#x} not batched: {why}");
    }
}

/// Runs `f` with the vertex stages it prepares taking user-data word `slot`
/// from the draw's start instance on the GPU (`Spec::base_instance_sgpr`:
/// indexed indirect draws whose arguments the GPU reads).
pub fn with_base_instance<T>(slot: u16, f: impl FnOnce() -> T) -> T {
    BASE_INSTANCE.with(|c| c.set(Some(slot)));
    let r = f();
    BASE_INSTANCE.with(|c| c.set(None));
    r
}

/// Whether anything of the stage resolved on the CPU (descriptors, scalar
/// loads and their pointers) depends on user-data word `slot`: then its
/// value must be known when the draw is recorded.
pub fn uses_user_word(p: &Prepared, slot: u16) -> bool {
    user_deps(&p.tr.plan) & (1u32 << slot.min(31)) != 0
}

/// NGG programs whose CPU-resolved inputs depend on the start-instance
/// word: their indirect draws read their arguments on the CPU.
static CPU_ARGS_PROGRAMS: Mutex<Option<std::collections::HashSet<u64>>> = Mutex::new(None);
static CPU_ARGS_ANY: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn note_cpu_args_program(addr: u64) {
    if CPU_ARGS_PROGRAMS.lock().get_or_insert_with(Default::default).insert(addr) {
        crate::info!("gpu: vertex program {addr:#x} resolves inputs from its start-instance word on the CPU: its indirect draws read their arguments there");
    }
    CPU_ARGS_ANY.store(true, std::sync::atomic::Ordering::Release);
}

pub fn cpu_args_program(addr: u64) -> bool {
    CPU_ARGS_ANY.load(std::sync::atomic::Ordering::Acquire) && CPU_ARGS_PROGRAMS.lock().as_ref().is_some_and(|s| s.contains(&addr))
}

/// Runs `f` with the stage preparations it makes reading their data-only
/// scalar loads on the GPU (`tune gpu_loads 3`: for stages whose data the
/// command processor would otherwise wait for).
pub fn with_gpu_loads<T>(f: impl FnOnce() -> T) -> T {
    FORCE_GPU_LOADS.with(|c| c.set(true));
    let r = f();
    FORCE_GPU_LOADS.with(|c| c.set(false));
    r
}

thread_local! {
    /// Set while `without_gpu_chains` runs.
    static NO_GPU_CHAINS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Runs `f` with the compute preparations it makes resolving every chain on
/// the CPU, a GPU-resolved program's included (front end 2: a command whose
/// per-item chains it cannot take on the GPU, `exec::fe2_gpu_ranges`).
pub fn without_gpu_chains<T>(f: impl FnOnce() -> T) -> T {
    NO_GPU_CHAINS.with(|c| c.set(true));
    let r = f();
    NO_GPU_CHAINS.with(|c| c.set(false));
    r
}

/// Programs whose data-only loads GPU work was found writing when they were
/// prepared (`tune gpu_loads 3`): prepared with GPU-side loads from then on
/// (no second preparation per command); a generation for the threads'
/// copies.
static GPU_LOADS_PROGRAMS: Mutex<Option<std::collections::HashSet<u64>>> = Mutex::new(None);
static GPU_LOADS_GEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

thread_local! {
    static LOCAL_GPU_LOADS: std::cell::RefCell<(u64, std::collections::HashSet<u64>)> = std::cell::RefCell::new((0, std::collections::HashSet::new()));
}

/// Program `addr` reads data GPU work writes: GPU-side loads from now on.
pub fn note_gpu_loads_program(addr: u64) {
    if GPU_LOADS_PROGRAMS.lock().get_or_insert_with(std::collections::HashSet::new).insert(addr) {
        GPU_LOADS_GEN.fetch_add(1, std::sync::atomic::Ordering::Release);
    }
}

/// Whether program `addr` was noted (this thread's copy of the set).
fn gpu_loads_program(addr: u64) -> bool {
    let generation = GPU_LOADS_GEN.load(std::sync::atomic::Ordering::Acquire);
    if generation == 0 {
        return false;
    }
    LOCAL_GPU_LOADS.with(|l| {
        let mut l = l.borrow_mut();
        if l.0 != generation {
            l.1 = GPU_LOADS_PROGRAMS.lock().clone().unwrap_or_default();
            l.0 = generation;
        }
        l.1.contains(&addr)
    })
}

/// Programs prepared with GPU-side loads because they were noted.
pub fn gpu_loads_programs() -> usize {
    GPU_LOADS_PROGRAMS.lock().as_ref().map_or(0, |s| s.len())
}

/// The data-only loads' ranges of a preparation made with CPU-resolved loads.
pub fn data_load_ranges(p: &Prepared) -> Vec<(u64, u64)> {
    p.data_loads.iter().map(|d| (d.addr, 4 * d.dwords as u64)).collect()
}

/// Programs logged under a `gpu_loads_sel` selection (bisection).
static GPU_LOADS_SELECTED: Mutex<Option<std::collections::HashSet<(u64, u64)>>> = Mutex::new(None);

thread_local! {
    /// This thread's view of the shared caches (entries never change once
    /// made), so that every command does not take their locks: programs by
    /// address with the shader registrations count at which they were last
    /// confirmed and the address's registration number (validating them by
    /// hashing their code took a few microseconds per command), plans, and
    /// translations.
    /// Per-thread handles (`Rc`) on the shared programs and plans: a
    /// preparation clones the handle, not the shared reference count the
    /// recording threads all touch (only a new preparation keeps a shared
    /// clone).
    static LOCAL_PROGRAMS: std::cell::RefCell<super::fx::Map<u64, (u64, u64, Arc<Local<Vec<rdna::Inst>>>, u64)>> = std::cell::RefCell::new(super::fx::Map::default());
    static LOCAL_PLANS: std::cell::RefCell<super::fx::Map<u64, std::rc::Rc<Arc<Plan>>>> = std::cell::RefCell::new(super::fx::Map::default());
    /// The reader's sizes in the last full preparation of each plan (ranges
    /// read, words read, pointers loaded): its vectors start at that
    /// capacity instead of doubling from empty (their growth was ~4% of the
    /// graphics command processor's busy samples, run 20261013-143720).
    static READ_HINTS: std::cell::RefCell<super::fx::Map<u64, [u32; 3]>> = std::cell::RefCell::new(super::fx::Map::default());
    static LOCAL_TRANSLATIONS: std::cell::RefCell<HashMap<(u64, u64), Arc<Translated>>> = std::cell::RefCell::new(HashMap::new());
    /// Translations by (plan key, fast hash of the specialization), checked
    /// by comparing the specialization: SipHashing the whole Spec for every
    /// command took about a microsecond per stage.
    static FAST_TRANSLATIONS: std::cell::RefCell<super::fx::Map<(u64, u64), Arc<Local<Translated>>>> = std::cell::RefCell::new(super::fx::Map::default());
    /// Each plan's latest translation on this thread: a preparation whose
    /// specialization equals it skips hashing the specialization (runs of
    /// commands repeat their shaders' specializations).
    static LAST_TRANSLATIONS: std::cell::RefCell<super::fx::Map<u64, Arc<Local<Translated>>>> = std::cell::RefCell::new(super::fx::Map::default());
    /// The vectors of the specializations being built (`spec_vecs`), kept
    /// for the next preparation.
    static SPEC_VECS: std::cell::RefCell<(Vec<PsInput>, Vec<ResSpec>)> = const { std::cell::RefCell::new((Vec::new(), Vec::new())) };
}

/// Empty vectors for a specialization's pixel inputs and resources, with
/// the capacity of earlier ones (`spec_vecs_back`).
fn spec_vecs() -> (Vec<PsInput>, Vec<ResSpec>) {
    SPEC_VECS.with(|v| {
        let (mut a, mut b) = std::mem::take(&mut *v.borrow_mut());
        a.clear();
        b.clear();
        (a, b)
    })
}

fn spec_vecs_back(spec: Spec) {
    SPEC_VECS.with(|v| *v.borrow_mut() = (spec.ps_inputs, spec.resources));
}

/// A recording thread's own handle on a shared program or translation:
/// preparations and their copies clone the handle, whose reference count
/// only its thread touches. (The shared counts are contended: a clone and
/// drop took up to ~100 ns on the development machine's Ryzen 9 5900X with
/// several recording threads, 3 ns uncontended.)
pub struct Local<T>(pub Arc<T>);

impl<T> std::ops::Deref for Local<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.0
    }
}

#[derive(Clone)]
pub struct Translated {
    pub out: Output,
    pub plan: Arc<Plan>,
    /// Cache key (program hash ^ specialization hash).
    pub key: u64,
    /// Buffer resources targeted by atomic instructions.
    pub atomic_res: Vec<usize>,
    /// The specialization it was made for (program address zero unless
    /// DS_WAVEMASK keys translations by it): the per-thread lookup by a fast
    /// hash compares it.
    pub spec: Spec,
}

pub(crate) fn hash_words(w: &[u32]) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    w.hash(&mut h);
    h.finish()
}

/// Program words at `addr`: the Agc shader object's code size when known,
/// else up to the first s_endpgm (at most 256 KiB).
fn code_slice(addr: u64) -> Option<&'static [u32]> {
    let size = match crate::hle::libs::agc::shader_code_size(addr) {
        Some(n) => n,
        None => {
            let mut n = 0usize;
            while n < 0x40000 && super::cp::guest_readable(addr + n as u64, 4) {
                let w = unsafe { std::ptr::read_volatile((addr + n as u64) as *const u32) };
                n += 4;
                if w == 0xBF81_0000 {
                    break;
                }
            }
            n
        }
    };
    if size < 4 || !super::cp::guest_readable(addr, size as u64) {
        return None;
    }
    Some(unsafe { std::slice::from_raw_parts(addr as *const u32, size / 4) })
}

/// Decodes (and caches) the program at `addr`; returns it and its hash.
pub(crate) fn program(addr: u64) -> Result<(Arc<Vec<rdna::Inst>>, u64), String> {
    program_local(addr).map(|(p, h)| (Arc::clone(&p.0), h))
}

/// `program` as this thread's handle on it.
fn program_local(addr: u64) -> Result<(Arc<Local<Vec<rdna::Inst>>>, u64), String> {
    use crate::hle::libs::agc;
    // A program from a shader object stays as decoded until a shader object
    // is created again at its address (or anywhere: then it is checked).
    let regs = agc::shader_registrations();
    let hit = LOCAL_PROGRAMS.with(|c| {
        let mut c = c.borrow_mut();
        let e = c.get_mut(&addr)?;
        if e.0 != regs {
            if agc::shader_registration(addr) != Some(e.1) {
                return None;
            }
            e.0 = regs;
        }
        Some((e.2.clone(), e.3))
    });
    if let Some(h) = hit {
        return Ok(h);
    }
    let reg = agc::shader_registration(addr);
    let (insts, hash) = program_checked(addr)?;
    let insts = Arc::new(Local(insts));
    if let Some(r) = reg {
        LOCAL_PROGRAMS.with(|c| c.borrow_mut().insert(addr, (regs, r, insts.clone(), hash)));
    }
    Ok((insts, hash))
}

/// The program at `addr`, decoded (cached and checked by hashing its code).
fn program_checked(addr: u64) -> Result<(Arc<Vec<rdna::Inst>>, u64), String> {
    let code = code_slice(addr).ok_or_else(|| format!("shader {addr:#x} not mapped"))?;
    let mut g = PROGRAMS.lock();
    let map = g.get_or_insert_with(HashMap::new);
    if let Some(p) = map.get(&addr) {
        if p.words <= code.len() && hash_words(&code[..p.words]) == p.hash {
            return Ok((p.insts.clone(), p.hash));
        }
    }
    let insts = rdna::decode_program(code).map_err(|e| format!("decode {addr:#x}: {}", e.0))?;
    let last = insts.last().ok_or("empty program")?;
    let words = (last.pc / 4 + last.len as u32) as usize;
    let hash = hash_words(&code[..words.min(code.len())]);
    note_per_lane(addr, &code[..words.min(code.len())], hash);
    let insts = Arc::new(insts);
    map.insert(addr, Program { words, hash, insts: insts.clone() });
    Ok((insts, hash))
}

/// A pointer chain's structural hash (`Reader::ptrs`).
fn ptr_hash(p: &Ptr) -> u64 {
    match p {
        Ptr::User(i) => 0x9E37_79B9_7F4A_7C15 ^ *i as u64,
        Ptr::Mem(inner, off) => ptr_hash(inner).rotate_left(17) ^ (*off as u64 + 1).wrapping_mul(0xBF58_476D_1CE4_E5B9),
        Ptr::Pc(o) => (*o ^ 0x94D0_49BB_1331_11EB).wrapping_mul(0x2545_F491_4F6C_DD1D),
    }
}

/// Guest memory reader that records what it read (hazard checks; adjacent
/// reads merged). It lives within one preparation of a stage whose plan
/// (`'a`) holds the pointer chains it resolves.
pub struct Reader<'a> {
    pub reads: Vec<(u64, u64)>,
    /// The words read, in the order of `reads` (a preparation made ahead of
    /// execution is checked against them, see `words_unchanged`).
    pub words: Vec<u32>,
    /// Ranges it could not read (not mapped or not readable): what was
    /// resolved without them holds while they stay so.
    pub unread: Vec<(u64, u64)>,
    /// The mappings of recent reads (no memory-map lookup inside them; a
    /// stage reads its user data, descriptor tables and constants from a few
    /// mappings in turn), replaced round robin.
    spans: [(u64, u64); 4],
    next_span: usize,
    /// Pointers loaded from memory so far, by the plan's chain (each resource
    /// dword names its pointer chain again; kept by reference, not cloned:
    /// the clones' allocations and deep comparisons were ~6% of the graphics
    /// command processor's time, run 20261012-202512), with the chain's
    /// structural hash (equal chains elsewhere in the plan are found by it
    /// before one deep comparison: comparing every entry deeply was up to 7%
    /// of the compute queues' busy samples, run 20261013-115322).
    ptrs: Vec<(&'a Ptr, u64, u64)>,
    /// Address of the program (program-relative pointers).
    code_base: u64,
    /// Pointers resolved so far by the plan's chain number
    /// (`Plan::chain_ids`): loads and resources naming equal chains find
    /// them without hashing or comparing chains (`ptr_hash` was 7.8% of the
    /// compute queues' busy samples, run 20261014-145828).
    chain_vals: Vec<Option<u64>>,
}

impl<'a> Reader<'a> {
    pub fn new(code_base: u64) -> Reader<'a> {
        Self::with_sizes(code_base, [0; 3])
    }

    /// A reader whose vectors start with room for `sizes` (ranges read,
    /// words read, pointers loaded).
    fn with_sizes(code_base: u64, sizes: [u32; 3]) -> Reader<'a> {
        Reader {
            reads: pool::pairs(sizes[0] as usize),
            words: pool::words(sizes[1] as usize),
            unread: pool::pairs(0),
            spans: [(0, 0); 4],
            next_span: 0,
            ptrs: Vec::with_capacity(sizes[2] as usize),
            code_base,
            chain_vals: pool::addrs(0),
        }
    }

    /// `ptr` of the chain numbered `id` in its plan (u32::MAX: unnumbered).
    fn chain(&mut self, id: u32, p: &'a Ptr, user: &[u32]) -> Option<u64> {
        if id == u32::MAX || !super::tune::chain_memo() {
            return self.ptr(p, user);
        }
        let i = id as usize;
        if let Some(Some(v)) = self.chain_vals.get(i) {
            return Some(*v);
        }
        let v = self.ptr(p, user)?;
        if self.chain_vals.len() <= i {
            self.chain_vals.resize(i + 1, None);
        }
        self.chain_vals[i] = Some(v);
        Some(v)
    }

    /// Whether guest memory [addr, addr + n) is readable: within a recent
    /// read's mapping, else one memory-map lookup (`readable_span` was ~7% of
    /// the recording threads' time, about 50 lookups per command, mostly
    /// from here: run 20261012-202512).
    fn readable(&mut self, addr: u64, n: u64) -> bool {
        let end = addr.saturating_add(n);
        if !super::tune::span_cache() {
            // The earlier behavior: the last read's mapping, two lookups.
            if addr >= self.spans[0].0 && end <= self.spans[0].1 {
                return true;
            }
            if !super::cp::guest_readable(addr, n) {
                return false;
            }
            self.spans[0] = crate::mem::readable_span(addr).unwrap_or((addr, end));
            return true;
        }
        if self.spans.iter().any(|&(s, e)| addr >= s && end <= e) {
            return true;
        }
        if !super::cp::valid(addr, n) {
            return false;
        }
        match crate::mem::readable_span(addr) {
            Some((s, e)) if end <= e => {
                self.spans[self.next_span] = (s, e);
                self.next_span = (self.next_span + 1) % self.spans.len();
                true
            }
            // Across mappings with different protections: the slow check.
            Some(_) => crate::mem::readable(addr, n),
            None => false,
        }
    }

    fn u32(&mut self, addr: u64) -> Option<u32> {
        if addr % 4 != 0 {
            return None;
        }
        if !self.readable(addr, 4) {
            self.unread.push((addr, 4));
            return None;
        }
        match self.reads.last_mut() {
            Some(r) if r.0 + r.1 == addr => r.1 += 4,
            _ => self.reads.push((addr, 4)),
        }
        // A copy recorded earlier leaves other words there (`gpu::forward`).
        let w = super::forward::word_or(addr, unsafe { std::ptr::read_volatile(addr as *const u32) });
        self.words.push(w);
        Some(w)
    }

    /// Reads `out.len()` consecutive dwords at `addr` with one readability
    /// check; false (nothing read) if any of them is not readable.
    fn read_into(&mut self, addr: u64, out: &mut [u32]) -> bool {
        let n = 4 * out.len() as u64;
        if addr % 4 != 0 || n == 0 {
            return false;
        }
        if !self.readable(addr, n) {
            self.unread.push((addr, n));
            return false;
        }
        match self.reads.last_mut() {
            Some(r) if r.0 + r.1 == addr => r.1 += n,
            _ => self.reads.push((addr, n)),
        }
        for (k, o) in out.iter_mut().enumerate() {
            *o = unsafe { std::ptr::read_volatile((addr + 4 * k as u64) as *const u32) };
        }
        super::forward::overlay(addr, out);
        self.words.extend_from_slice(out);
        true
    }

    fn ptr(&mut self, p: &'a Ptr, user: &[u32]) -> Option<u64> {
        match p {
            Ptr::User(i) => Some((*user.get(*i as usize)? as u64) | ((*user.get(*i as usize + 1)? as u64) << 32)),
            Ptr::Mem(inner, off) => {
                // The same chain object first, then an equal one elsewhere in
                // the plan.
                if let Some(&(_, _, v)) = self.ptrs.iter().find(|e| std::ptr::eq(e.0, p)) {
                    return Some(v);
                }
                let h = ptr_hash(p);
                if let Some(&(_, _, v)) = self.ptrs.iter().find(|e| e.1 == h && e.0 == p) {
                    return Some(v);
                }
                let base = self.ptr(inner, user)? + *off as u64;
                let v = (self.u32(base)? as u64) | ((self.u32(base + 4)? as u64) << 32);
                self.ptrs.push((p, h, v));
                Some(v)
            }
            Ptr::Pc(o) => Some(self.code_base + *o),
        }
    }

    fn eval(&mut self, p: &'a Prov, user: &[u32]) -> Option<u32> {
        match p {
            Prov::User(i) => user.get(*i as usize).copied(),
            Prov::Const(c) => Some(*c),
            Prov::Mem(ptr, off) => {
                let b = self.ptr(ptr, user)?;
                self.u32(b + *off as u64)
            }
            Prov::PtrLo(ptr, b) => Some((self.ptr(ptr, user)? + *b as u64) as u32),
            Prov::PtrHi(ptr, b) => Some(((self.ptr(ptr, user)? + *b as u64) >> 32) as u32),
            Prov::Unknown | Prov::Table(..) => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Resolved {
    Buffer(Vsharp),
    /// A base address and, for data loads at constant offsets, the bytes they
    /// read from it.
    Global(u64, Option<u64>),
    Image(Tsharp),
    /// The entries of an image table (None: unusable entry, bound as a
    /// dummy image).
    ImageTable(Vec<Option<Tsharp>>),
    Sampler([u32; 4]),
}

/// What preparing a stage found in guest memory, shared by the
/// preparation's copies (one allocation; a repeated preparation from the
/// cache, `prep_cache`, costs no deep copy). `Prepared` derefs to it.
/// The program and its translation are held here too: the recording
/// threads share their reference counts, so a copy of a preparation
/// touches only this block's (contended clones cost up to ~100 ns each on
/// the development machine's Ryzen 9 5900X).
pub struct Found {
    pub insts: Arc<Local<Vec<rdna::Inst>>>,
    pub tr: Arc<Local<Translated>>,
    pub resources: Vec<Resolved>,
    /// Guest address of each descriptor's first dword when read from memory.
    pub res_addrs: Vec<Option<u64>>,
    /// Guest memory read while resolving (for write hazards).
    pub reads: Vec<(u64, u64)>,
    /// The words read, in the order of `reads`.
    pub words: Vec<u32>,
    /// Guest ranges it could not read (`Reader::unread`).
    pub unread: Vec<(u64, u64)>,
    /// The data-only scalar loads (`Plan::load_data`) as read: where their
    /// dwords sit in `sgpr_data`, the guest address, and where their words
    /// start in `words`.
    pub data_loads: Vec<DataLoad>,
}

impl Drop for Found {
    fn drop(&mut self) {
        pool::put_resolved(std::mem::take(&mut self.resources));
        pool::put_addrs(std::mem::take(&mut self.res_addrs));
        pool::put_pairs(std::mem::take(&mut self.reads));
        pool::put_words(std::mem::take(&mut self.words));
        pool::put_pairs(std::mem::take(&mut self.unread));
        pool::put_loads(std::mem::take(&mut self.data_loads));
    }
}

/// The vectors of finished preparations, kept per thread for the next ones
/// (`Found` and `Prepared` give theirs back when dropped): a preparation
/// made ~10 allocations, ~5% of the recording workers' busy samples in
/// `mi_malloc_aligned` (2026-10-15).
mod pool {
    use super::{DataLoad, Resolved};
    use std::cell::RefCell;

    /// Vectors kept per kind at most, and the largest capacity kept.
    const MAX: usize = 64;
    const CAP_MAX: usize = 1 << 16;

    #[derive(Default)]
    struct Pool {
        pairs: Vec<Vec<(u64, u64)>>,
        words: Vec<Vec<u32>>,
        resolved: Vec<Vec<Resolved>>,
        addrs: Vec<Vec<Option<u64>>>,
        loads: Vec<Vec<DataLoad>>,
    }

    thread_local! {
        static POOL: RefCell<Pool> = RefCell::new(Pool::default());
    }

    macro_rules! kind {
        ($take:ident, $put:ident, $field:ident, $t:ty) => {
            /// An empty vector with room for `cap` (from the pool if one is left).
            pub fn $take(cap: usize) -> Vec<$t> {
                if !crate::gpu::tune::prep_pool() {
                    return Vec::with_capacity(cap);
                }
                let v = POOL.try_with(|p| p.try_borrow_mut().ok().and_then(|mut p| p.$field.pop())).ok().flatten();
                match v {
                    Some(mut v) => {
                        v.reserve(cap);
                        v
                    }
                    None => Vec::with_capacity(cap),
                }
            }

            /// Gives `v` back (emptied) for a later preparation.
            pub fn $put(mut v: Vec<$t>) {
                if v.capacity() == 0 || v.capacity() > CAP_MAX || !crate::gpu::tune::prep_pool() {
                    return;
                }
                v.clear();
                let _ = POOL.try_with(|p| {
                    if let Ok(mut p) = p.try_borrow_mut() {
                        if p.$field.len() < MAX {
                            p.$field.push(v);
                        }
                    }
                });
            }
        };
    }

    kind!(pairs, put_pairs, pairs, (u64, u64));
    kind!(words, put_words, words, u32);
    kind!(resolved, put_resolved, resolved, Resolved);
    kind!(addrs, put_addrs, addrs, Option<u64>);
    kind!(loads, put_loads, loads, DataLoad);
}

/// Everything needed to bind one shader stage for a draw/dispatch: what was
/// found (`Found`, shared by copies; fields reached through `Deref`), and
/// the SGPR data and user data, a copy's own.
pub struct Prepared {
    pub stage: Stage,
    pub addr: u64,
    pub found: Arc<Found>,
    /// SGPR data buffer contents (resource info blocks filled at binding).
    pub sgpr_data: Vec<u32>,
    pub user: UserData,
    /// `mem::MAP_GEN` and `images::GPU_IMAGES_GEN` when it was made (a
    /// preparation made ahead of execution is checked against them).
    pub map_gen: u64,
    pub images_gen: u64,
    /// Image resources that depth-comparison instructions sample (bit per
    /// resource index) and whether each was a depth view (the only part of a
    /// preparation that depends on the image cache).
    pub cmp_images: u64,
    pub cmp_depth: u64,
    /// The program code's hash (`program_checked`).
    pub code_hash: u64,
    /// Where it was made, for `DS_FE2_CALLERS` (`fe2::StageSource`): 1 by
    /// the thread binding it, 2 ahead by a worker (`gpu::ahead`), 3 that
    /// and its placement checks made there in epoch `checked_ep`.
    pub ahead: u8,
    pub checked_ep: u32,
}

impl Clone for Prepared {
    fn clone(&self) -> Prepared {
        let mut sgpr_data = pool::words(self.sgpr_data.len());
        sgpr_data.extend_from_slice(&self.sgpr_data);
        Prepared {
            stage: self.stage,
            addr: self.addr,
            found: self.found.clone(),
            sgpr_data,
            user: self.user,
            map_gen: self.map_gen,
            images_gen: self.images_gen,
            cmp_images: self.cmp_images,
            cmp_depth: self.cmp_depth,
            code_hash: self.code_hash,
            ahead: self.ahead,
            checked_ep: self.checked_ep,
        }
    }
}

impl Drop for Prepared {
    fn drop(&mut self) {
        pool::put_words(std::mem::take(&mut self.sgpr_data));
    }
}

/// A stage's user-data dwords (at most 32), held inline: a preparation
/// and each copy of one would otherwise allocate them.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct UserData {
    n: u8,
    w: [u32; 32],
}

impl UserData {
    pub fn new(s: &[u32]) -> UserData {
        let n = s.len().min(32);
        let mut w = [0u32; 32];
        w[..n].copy_from_slice(&s[..n]);
        UserData { n: n as u8, w }
    }
}

impl std::ops::Deref for UserData {
    type Target = [u32];
    fn deref(&self) -> &[u32] {
        &self.w[..self.n as usize]
    }
}

impl std::fmt::Debug for UserData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        (**self).fmt(f)
    }
}

impl std::ops::Deref for Prepared {
    type Target = Found;
    fn deref(&self) -> &Found {
        &self.found
    }
}

/// A data-only scalar load of a prepared stage (`Prepared::data_loads`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DataLoad {
    pub sgpr: u32,
    pub addr: u64,
    pub dwords: u32,
    pub word: u32,
}

/// Image resources (bit per index) that depth-comparison sampling
/// instructions (`image_sample_c*`, `image_gather4_c*`) use: only for them
/// does the translation depend on whether the image is a depth view.
fn compare_images(plan_key: u64, insts: &[rdna::Inst], plan: &Plan) -> u64 {
    thread_local! {
        static MASKS: std::cell::RefCell<super::fx::Map<u64, u64>> = std::cell::RefCell::new(super::fx::Map::default());
    }
    MASKS.with(|m| {
        *m.borrow_mut().entry(plan_key).or_insert_with(|| {
            let mut mask = 0u64;
            for i in insts {
                let Some(name) = i.name() else { continue };
                if name == "image_get_lod" || !(name.starts_with("image_sample") || name.starts_with("image_gather4")) {
                    continue;
                }
                let has_c = name.trim_start_matches("image_sample").trim_start_matches("image_gather4").split('_').any(|m| m == "c");
                if let (true, Some(&(ri, _))) = (has_c, plan.inst_res.get(&i.pc)) {
                    if ri < 64 {
                        mask |= 1 << ri;
                    }
                }
            }
            mask
        })
    })
}

/// Whether the depth-view decisions a preparation made for its compared
/// images still hold (`images::Images::depth_view`).
pub fn depth_views_hold(p: &Prepared) -> bool {
    if p.cmp_images == 0 || p.images_gen == super::images::GPU_IMAGES_GEN.load(std::sync::atomic::Ordering::Acquire) {
        return true;
    }
    super::images::with(|im| {
        p.resources.iter().enumerate().all(|(i, r)| match r {
            Resolved::Image(t) if i < 64 && p.cmp_images & (1 << i) != 0 => im.depth_view(t) == (p.cmp_depth & (1 << i) != 0),
            _ => true,
        })
    })
}

/// The registers a stage preparation reads: the live command processor, or
/// a snapshot taken ahead of execution (`StageRegs`).
pub trait RegRead {
    fn sh(&self, i: u32) -> u32;
    /// Context register by byte address (0x28000..).
    fn ctx(&self, a: u32) -> u32;
}

impl RegRead for Cp {
    fn sh(&self, i: u32) -> u32 {
        self.reg(Space::Sh, i)
    }
    fn ctx(&self, a: u32) -> u32 {
        self.reg(Space::Ctx, (a - 0x28000) / 4)
    }
}

/// The kind of stage a preparation is for.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StageKind {
    Vertex,
    Pixel,
    Compute,
}

/// SH registers (dword indices) and context registers (byte addresses) the
/// preparation of a stage kind reads (`prepare_ngg`, `prepare_ps`,
/// `prepare_compute`; keep in step with them).
fn reg_lists(kind: StageKind) -> (&'static [u32], &'static [u32]) {
    static LISTS: std::sync::OnceLock<[(Vec<u32>, Vec<u32>); 3]> = std::sync::OnceLock::new();
    let l = LISTS.get_or_init(|| {
        let vs_sh: Vec<u32> = [ES_PGM_LO, ES_PGM_LO + 1, GS_RSRC2].into_iter().chain(GS_USER_DATA..GS_USER_DATA + 32).collect();
        let ps_sh: Vec<u32> = [PS_PGM_LO, PS_PGM_LO + 1, PS_RSRC2].into_iter().chain(PS_USER_DATA..PS_USER_DATA + 32).collect();
        let mut ps_ctx: Vec<u32> = vec![SPI_PS_IN_CONTROL, SPI_PS_INPUT_ADDR, SPI_SHADER_COL_FORMAT, DB_SHADER_CONTROL];
        ps_ctx.extend((0..32).map(|n| SPI_PS_INPUT_CNTL_0 + 4 * n));
        ps_ctx.extend((0..8).map(|m| CB_COLOR0_INFO + 0x3C * m));
        let cs_sh: Vec<u32> = [CS_PGM_LO, CS_PGM_LO + 1, CS_RSRC2, CS_NUM_THREAD, CS_NUM_THREAD + 1, CS_NUM_THREAD + 2].into_iter().chain(CS_USER_DATA..CS_USER_DATA + 16).collect();
        let mut v = [(vs_sh, vec![VGT_SHADER_STAGES_EN]), (ps_sh, ps_ctx), (cs_sh, Vec::new())];
        for (sh, ctx) in v.iter_mut() {
            sh.sort_unstable();
            ctx.sort_unstable();
        }
        v
    });
    let e = &l[kind as usize];
    (&e.0, &e.1)
}

/// The registers a stage preparation reads, captured ahead of execution
/// (values in the order of `reg_lists`).
#[derive(Debug)]
pub struct StageRegs {
    pub kind: StageKind,
    pub sh: Vec<u32>,
    pub ctx: Vec<u32>,
    /// A register outside the lists was read (the preparation is not used).
    pub missed: std::sync::atomic::AtomicBool,
    /// The first such register (SH: dword index; context: byte address).
    pub missed_reg: std::sync::atomic::AtomicU32,
}

impl StageRegs {
    pub fn capture<R: RegRead + ?Sized>(kind: StageKind, r: &R) -> StageRegs {
        let (sh, ctx) = reg_lists(kind);
        StageRegs {
            kind,
            sh: sh.iter().map(|&i| r.sh(i)).collect(),
            ctx: ctx.iter().map(|&a| r.ctx(a)).collect(),
            missed: std::sync::atomic::AtomicBool::new(false),
            missed_reg: std::sync::atomic::AtomicU32::new(0),
        }
    }

    /// Whether `r` holds the same values.
    pub fn matches<R: RegRead + ?Sized>(&self, r: &R) -> bool {
        let (sh, ctx) = reg_lists(self.kind);
        sh.iter().zip(&self.sh).all(|(&i, &v)| r.sh(i) == v) && ctx.iter().zip(&self.ctx).all(|(&a, &v)| r.ctx(a) == v)
    }
}

impl RegRead for StageRegs {
    fn sh(&self, i: u32) -> u32 {
        let (sh, _) = reg_lists(self.kind);
        match sh.binary_search(&i) {
            Ok(k) => self.sh[k],
            Err(_) => {
                if !self.missed.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    self.missed_reg.store(i, std::sync::atomic::Ordering::Relaxed);
                }
                0
            }
        }
    }
    fn ctx(&self, a: u32) -> u32 {
        let (_, ctx) = reg_lists(self.kind);
        match ctx.binary_search(&a) {
            Ok(k) => self.ctx[k],
            Err(_) => {
                if !self.missed.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    self.missed_reg.store(a, std::sync::atomic::Ordering::Relaxed);
                }
                0
            }
        }
    }
}

/// Whether the guest words a preparation read are still in memory and what
/// it could not read still cannot be read (a preparation made ahead of
/// execution, or a reused one, is used only then): a data load from pages
/// the title's streaming had not mapped yet resolved to zeros, and a cached
/// preparation kept them after the pages were mapped (DS_PREP_CHECK, run
/// 20261013-064118).
pub fn words_unchanged(p: &Prepared) -> bool {
    let checked = crate::mem::MAP_GEN.load(std::sync::atomic::Ordering::Acquire) != p.map_gen;
    if checked && p.unread.iter().any(|&(a, n)| super::cp::guest_readable(a, n)) {
        return false;
    }
    let mut k = 0usize;
    for &(a, n) in p.reads.iter() {
        let m = (n / 4) as usize;
        if checked && !super::cp::guest_readable(a, n) {
            return false;
        }
        let Some(want) = p.words.get(k..k + m) else { return false };
        if super::forward::active() {
            // As the reader sees them now (`gpu::forward`).
            let mut now = vec![0u32; m];
            for (j, o) in now.iter_mut().enumerate() {
                *o = unsafe { std::ptr::read_volatile((a + 4 * j as u64) as *const u32) };
            }
            super::forward::overlay(a, &mut now);
            if now != want {
                return false;
            }
            k += m;
            continue;
        }
        for (j, &w) in want.iter().enumerate() {
            if unsafe { std::ptr::read_volatile((a + 4 * j as u64) as *const u32) } != w {
                return false;
            }
        }
        k += m;
    }
    k == p.words.len()
}

fn num_class(format: u16) -> NumClass {
    match rdna::format::gfx10(format).map(|f| f.num) {
        Some(rdna::format::Num::Uint) => NumClass::Uint,
        Some(rdna::format::Num::Sint) => NumClass::Sint,
        _ => NumClass::Float,
    }
}

/// Compute-stage registers (SH dword indices).
const CS_PGM_LO: u32 = 0x20C;
const CS_RSRC2: u32 = 0x213;
const CS_NUM_THREAD: u32 = 0x207;
const CS_USER_DATA: u32 = 0x240;

/// Resolves and translates the compute shader bound on `cp`. `wave64` comes
/// from the dispatch initiator (CS_W32_EN clear).
/// Programs dispatched in runs of consecutive direct dispatches (`tune
/// batch_cs`, `Exec::note_cs_streak`): translated batched (`Spec::batched`).
static BATCHABLE: [std::sync::atomic::AtomicU64; 64] = [const { std::sync::atomic::AtomicU64::new(0) }; 64];

pub fn batchable(addr: u64) -> bool {
    BATCHABLE.iter().any(|a| a.load(std::sync::atomic::Ordering::Relaxed) == addr)
}

/// Marks program `addr` batchable (at most 64 programs).
pub fn mark_batchable(addr: u64) {
    use std::sync::atomic::Ordering;
    if batchable(addr) {
        return;
    }
    for a in &BATCHABLE {
        if a.compare_exchange(0, addr, Ordering::AcqRel, Ordering::Acquire).is_ok() {
            crate::info!("gpu: cs {addr:#x} batched from now on (runs of consecutive dispatches)");
            return;
        }
        if a.load(Ordering::Acquire) == addr {
            return;
        }
    }
}

pub fn prepare_compute<R: RegRead + ?Sized>(cp: &R, wave64: bool) -> Result<Prepared, String> {
    let sh = |i: u32| cp.sh(i);
    let addr = ((sh(CS_PGM_LO) as u64) | ((sh(CS_PGM_LO + 1) as u64 & 0xFF) << 32)) << 8;
    let rsrc2 = sh(CS_RSRC2);
    let user_sgprs = ((rsrc2 >> 1) & 0x1F).min(16) as u16;
    let user: [u32; 16] = std::array::from_fn(|i| sh(CS_USER_DATA + i as u32));
    let local = [sh(CS_NUM_THREAD) & 0xFFFF, sh(CS_NUM_THREAD + 1) & 0xFFFF, sh(CS_NUM_THREAD + 2) & 0xFFFF];
    let (insts, code_hash) = program_local(addr)?;
    let (ps_inputs, resources) = spec_vecs();
    let mut spec = Spec {
        stage: Stage::Compute,
        user_sgprs,
        user_base: 0,
        code_base: addr,
        wave64,
        ps_input_addr: 0,
        ps_inputs,
        barycentric: false,
        mrt: [MrtSpec::default(); 8],
        cs_local_size: local,
        cs_rsrc2: rsrc2,
        resources,
        binding_base: 0,
        probe: probe_for(addr),
        dump: dump_for(addr),
        force_dispatcher: super::tune::block_chain() == 0,
        dispatcher_for_loops: super::tune::block_chain() == 2,
        reg_vars: super::tune::reg_vars(),
        lane_guards: super::tune::lane_guards(),
        mip_views: super::tune::mip_views(),
        lod_stats: super::tune::lod_stats() == 3,
        early_tests: false,
        per_lane: false,
        lane_votes: false,
        base_instance_sgpr: None,
        sgpr_pc: super::tune::sgpr_pc().then_some(translate::sgpr_pc_offset(Stage::Compute)),
        addresses: super::tune::addresses() || (super::tune::batch_cs() && batchable(addr)),
        batched: super::tune::batch_cs() && batchable(addr),
        write_marks: super::resident::mirror_wanted() || super::fe2::wanted(),
        sgpr_ubo: super::tune::sgpr_ubo(),
        gpu_loads: 0,
        gpu_res: 0,
        pt_linear: false,
    };
    if super::tune::per_lane() != 0 && per_lane_program(code_hash) {
        let votes = lane_votes_program(code_hash);
        spec.per_lane = super::tune::per_lane() == 1 || !votes;
        spec.lane_votes = spec.per_lane && votes;
    }
    // GPU-resolved per-item chains (`gpures`, `tune gpu_chains`; batched
    // programs, whose translations use device addresses).
    if spec.batched && super::tune::gpu_chains() != 0 && super::pagetable::address() != 0 && !NO_GPU_CHAINS.with(|c| c.get()) {
        if let Some(d) = super::gpures::decided(addr) {
            spec.gpu_loads = d.gpu_loads;
            spec.gpu_res = d.gpu_res;
            // Front end 2: its buffers from the mirror at guest offsets.
            spec.pt_linear = super::fe2::on();
        }
    }
    let r = prepare(addr, insts, code_hash, &user, &mut spec);
    spec_vecs_back(spec);
    r
}

/// Compute programs translated per lane (`Spec::per_lane`), by FNV-1a of
/// their words up to the first s_endpgm, each checked by reading its code:
/// its cross-lane operations only scalarize per-lane work (FINDINGS.md
/// 2026-10-13: their translations then take 128 registers instead of 168
/// and run 35-40% faster); and whether its EXEC branches are decided per
/// subgroup (`Spec::lane_votes`).
const PER_LANE_PROGRAMS: &[(u64, &str, bool)] = &[
    // Tiled lighting passes (~40k workgroups of one wave64): per-cluster
    // light masks OR-reduced across the wave (DPP, v_permlanex16,
    // v_readlane of lanes 31 and 63) and walked with s_ff1; light data read
    // with scalar loads, per-lane tests inside.
    (0xb65e_e39a_31c6_fa92, "tiled pass cs 0x10088ab600", false),
    (0xbf04_60d4_7979_f8d3, "tiled pass cs 0x10088b8200", false),
    // The froxel fog (the same light-mask walks, two waterfall loops, LDS
    // slots per lane): per lane in half the time. Its light-mask unions
    // enable all lanes and read the ones EXEC had masked off as zero; with
    // EXEC branches decided per invocation such lanes had left the region
    // and the reads found no value: a haze band along the right screen edge
    // (the volume is 214 froxels wide in 27 groups of 8; lanes past it read
    // no depth). Its EXEC branches are decided per 32-lane subgroup
    // (`Spec::lane_votes`; the tiled passes' would cost them ~20%).
    (0xcb83_739f_ea04_1817, "froxel fog cs 0x100892b200", true),
];

/// Program hashes (`hash_words`) found in PER_LANE_PROGRAMS (read without a
/// lock by every compute preparation).
static PER_LANE_HASHES: [std::sync::atomic::AtomicU64; 8] = [const { std::sync::atomic::AtomicU64::new(0) }; 8];

fn per_lane_program(code_hash: u64) -> bool {
    code_hash != 0 && PER_LANE_HASHES.iter().any(|h| h.load(std::sync::atomic::Ordering::Relaxed) == code_hash)
}

/// Program hashes of PER_LANE_PROGRAMS entries with subgroup-decided EXEC
/// branches.
static LANE_VOTE_HASHES: [std::sync::atomic::AtomicU64; 8] = [const { std::sync::atomic::AtomicU64::new(0) }; 8];

fn lane_votes_program(code_hash: u64) -> bool {
    code_hash != 0 && LANE_VOTE_HASHES.iter().any(|h| h.load(std::sync::atomic::Ordering::Relaxed) == code_hash)
}

/// FNV-1a of a program's bytes up to and including its first s_endpgm.
fn program_id(words: &[u32]) -> u64 {
    let n = words.iter().position(|&w| w == 0xBF81_0000).map_or(words.len(), |i| i + 1);
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &w in &words[..n] {
        for b in w.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

/// Notes a decoded program found in PER_LANE_PROGRAMS (by its hash).
fn note_per_lane(addr: u64, words: &[u32], hash: u64) {
    let id = program_id(words);
    let Some(&(_, what, votes)) = PER_LANE_PROGRAMS.iter().find(|p| p.0 == id) else { return };
    if per_lane_program(hash) {
        return;
    }
    // (The vote list first: a preparation that finds the program per lane
    // finds its votes too.)
    if votes {
        for slot in &LANE_VOTE_HASHES {
            if slot.compare_exchange(0, hash, std::sync::atomic::Ordering::Relaxed, std::sync::atomic::Ordering::Relaxed).is_ok() {
                break;
            }
        }
    }
    for slot in &PER_LANE_HASHES {
        if slot.compare_exchange(0, hash, std::sync::atomic::Ordering::Relaxed, std::sync::atomic::Ordering::Relaxed).is_ok() {
            crate::info!("shader {addr:#x}: translated per lane ({what}{})", if votes { "; EXEC branches per subgroup" } else { "" });
            return;
        }
    }
}

/// User-data dword count of a graphics stage: RSRC2 USER_SGPR[5:1] and USER_SGPR_MSB[27].
fn user_count(rsrc2: u32) -> u16 {
    (((rsrc2 >> 1) & 0x1F) | (((rsrc2 >> 27) & 1) << 5)).min(32) as u16
}

/// Pixel-shader registers (SH dword indices) and context registers (byte addresses).
const PS_PGM_LO: u32 = 0x08;
const PS_RSRC2: u32 = 0x0B;
const PS_USER_DATA: u32 = 0x0C;
const SPI_PS_INPUT_CNTL_0: u32 = 0x28644;
const SPI_PS_INPUT_ADDR: u32 = 0x286D0;
const SPI_PS_IN_CONTROL: u32 = 0x286D8;
/// DB_SHADER_CONTROL (bit 12 DEPTH_BEFORE_SHADER: `Spec::early_tests`).
const DB_SHADER_CONTROL: u32 = 0x2880C;
const SPI_SHADER_COL_FORMAT: u32 = 0x28714;
const CB_COLOR0_INFO: u32 = 0x28C70;

/// Resolves and translates the pixel shader bound on `cp`.
pub fn prepare_ps<R: RegRead + ?Sized>(cp: &R) -> Result<Prepared, String> {
    let sh = |i: u32| cp.sh(i);
    let ctx = |a: u32| cp.ctx(a);
    let addr = ((sh(PS_PGM_LO) as u64) | ((sh(PS_PGM_LO + 1) as u64 & 0xFF) << 32)) << 8;
    let rsrc2 = sh(PS_RSRC2);
    let user_sgprs = user_count(rsrc2);
    let user: [u32; 32] = std::array::from_fn(|i| sh(PS_USER_DATA + i as u32));
    let (insts, code_hash) = program_local(addr)?;
    let in_control = ctx(SPI_PS_IN_CONTROL);
    let (mut ps_inputs, resources) = spec_vecs();
    for n in 0..(in_control & 0x3F).min(32) {
        let c = ctx(SPI_PS_INPUT_CNTL_0 + 4 * n);
        let off = c & 0x3F;
        ps_inputs.push(PsInput { param: (off & 0x1F) as u8, flat: (c >> 10) & 1 != 0, default: (off & 0x20 != 0).then_some(((c >> 8) & 3) as u8) });
    }
    let col = ctx(SPI_SHADER_COL_FORMAT);
    let mut mrt = [MrtSpec::default(); 8];
    for (m, slot) in mrt.iter_mut().enumerate() {
        let info = ctx(CB_COLOR0_INFO + 0x3C * m as u32);
        let class = match (info >> 8) & 7 {
            4 => 2,
            5 => 1,
            _ => 0,
        };
        *slot = MrtSpec { export_format: ((col >> (4 * m)) & 0xF) as u8, class };
    }
    let mut spec = Spec {
        stage: Stage::Fragment,
        user_sgprs,
        user_base: 0,
        code_base: addr,
        wave64: (in_control >> 15) & 1 == 0,
        ps_input_addr: ctx(SPI_PS_INPUT_ADDR),
        barycentric: super::vk::ctx().barycentric,
        ps_inputs,
        mrt,
        cs_local_size: [1, 1, 1],
        cs_rsrc2: 0,
        resources,
        binding_base: PS_BINDING_BASE,
        probe: probe_for(addr),
        dump: None,
        force_dispatcher: super::tune::block_chain() == 0,
        dispatcher_for_loops: super::tune::block_chain() == 2,
        reg_vars: super::tune::reg_vars(),
        lane_guards: super::tune::lane_guards(),
        mip_views: super::tune::mip_views(),
        lod_stats: super::tune::lod_stats() == 3,
        early_tests: ctx(DB_SHADER_CONTROL) & (1 << 12) != 0,
        per_lane: false,
        lane_votes: false,
        base_instance_sgpr: None,
        sgpr_pc: super::tune::sgpr_pc().then_some(translate::sgpr_pc_offset(Stage::Fragment)),
        addresses: super::tune::addresses() || batched_draw_now(),
        batched: batched_draw_now(),
        write_marks: super::resident::mirror_wanted() || super::fe2::wanted(),
        sgpr_ubo: super::tune::sgpr_ubo(),
        gpu_loads: 0,
        gpu_res: 0,
        pt_linear: false,
    };
    if spec.batched {
        if let Some((l, r)) = gpu_draw_masks_now(1) {
            spec.gpu_loads = l;
            spec.gpu_res = r;
            spec.pt_linear = super::fe2::on();
        }
    }
    let r = prepare(addr, insts, code_hash, &user, &mut spec);
    spec_vecs_back(spec);
    r
}

/// `DS_PROBE=<shader address>:<pc>:<vgpr>` (hex address and pc, decimal
/// VGPR) or the `probe` control command: that pixel shader exports the
/// VGPR's value after the instruction at pc as its color (float targets) or
/// its raw bits (integer targets), or that compute shader stores it as the
/// data of its image stores (debugging). An optional fourth field `:<n>`
/// starts probing in the n-th frame (0-based) that uses the shader, e.g. to
/// keep a reference frame intact.
pub(crate) fn probe_for(addr: u64) -> Option<(u32, u32)> {
    static USES: std::sync::Mutex<(u64, u32)> = std::sync::Mutex::new((u64::MAX, 0));
    // Every stage preparation asks: the shared lock only while a probe is
    // set (it was ~1% of the recording threads' busy samples).
    PROBE_FROM_ENV.call_once(|| {
        let mut g = PROBE.lock();
        let p = *g.get_or_insert_with(|| std::env::var("DS_PROBE").ok().and_then(|v| parse_probe(&v)));
        PROBE_ON.store(p.is_some(), std::sync::atomic::Ordering::Relaxed);
    });
    if !PROBE_ON.load(std::sync::atomic::Ordering::Relaxed) {
        return None;
    }
    let p = *PROBE.lock().get_or_insert_with(|| std::env::var("DS_PROBE").ok().and_then(|v| parse_probe(&v)));
    let (a, pc, vg, from) = p?;
    if a != addr {
        return None;
    }
    let mut u = USES.lock().unwrap();
    let flip = super::capture::flips();
    if u.0 != flip {
        *u = (flip, u.1 + 1);
    }
    (u.1 > from).then_some((pc, vg))
}

/// Probe setting (see `probe_for`): None until read from `DS_PROBE`.
static PROBE: parking_lot::Mutex<Option<Option<(u64, u32, u32, u32)>>> = parking_lot::Mutex::new(None);
/// A probe is set (PROBE holds one); `DS_PROBE` read once.
static PROBE_ON: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static PROBE_FROM_ENV: std::sync::Once = std::sync::Once::new();

/// `<shader>:<pc>:<register>[:<n>]`; the register is `v<k>` (or `<k>`) or
/// `s<k>`, with a suffix `u` / `i` to show an unsigned / signed integer as
/// a float.
fn parse_probe(v: &str) -> Option<(u64, u32, u32, u32)> {
    use rdna::translate::{PROBE_SGPR, PROBE_SINT, PROBE_UINT};
    let mut it = v.split(':');
    let a = u64::from_str_radix(it.next()?.trim().trim_start_matches("0x"), 16).ok()?;
    let pc = u32::from_str_radix(it.next()?.trim().trim_start_matches("0x"), 16).ok()?;
    let r = it.next()?.trim();
    let (r, conv) = match r.strip_suffix('u') {
        Some(x) => (x, PROBE_UINT),
        None => match r.strip_suffix('i') {
            Some(x) => (x, PROBE_SINT),
            None => (r, 0),
        },
    };
    let (r, file) = match r.strip_prefix('s') {
        Some(x) => (x, PROBE_SGPR),
        None => (r.strip_prefix('v').unwrap_or(r), 0),
    };
    let n: u32 = r.parse().ok()?;
    if n >= if file != 0 { 128 } else { 256 } {
        return None;
    }
    let vg = n | file | conv;
    let from = it.next().and_then(|x| x.trim().parse().ok()).unwrap_or(0);
    Some((a, pc, vg, from))
}

/// `probe <shader>:<pc>:<vgpr>[:<n>]` / `probe off` (control command).
pub fn set_probe(v: &str) -> Result<(), String> {
    let p = if v.trim() == "off" { None } else { Some(parse_probe(v).ok_or("usage: probe <hex shader>:<hex pc>:<v<k>|s<k>>[u|i][:<frames>] | off")?) };
    // The environment's setting is replaced (read first so that it cannot
    // come back).
    PROBE_FROM_ENV.call_once(|| {});
    *PROBE.lock() = Some(p);
    PROBE_ON.store(p.is_some(), std::sync::atomic::Ordering::Relaxed);
    Ok(())
}

/// `DS_DUMP=<shader>:<pc>:<x>,<y>,<z>` (hex shader address and pc, decimal
/// workgroup id): that compute shader's workgroup dumps its registers after
/// the instruction at pc (`DS_EMU_CHECK` compares them with the interpreter).
pub(crate) fn dump_for(addr: u64) -> Option<(u32, [u32; 3])> {
    static D: std::sync::OnceLock<Option<(u64, u32, [u32; 3])>> = std::sync::OnceLock::new();
    let d = D.get_or_init(|| {
        let v = std::env::var("DS_DUMP").ok()?;
        let mut it = v.split(':');
        let a = u64::from_str_radix(it.next()?.trim_start_matches("0x"), 16).ok()?;
        let pc = u32::from_str_radix(it.next()?.trim_start_matches("0x"), 16).ok()?;
        let g: Vec<u32> = it.next()?.split(',').filter_map(|x| x.parse().ok()).collect();
        Some((a, pc, [g.first().copied().unwrap_or(0), g.get(1).copied().unwrap_or(0), g.get(2).copied().unwrap_or(0)]))
    });
    d.and_then(|(a, pc, g)| (a == addr).then_some((pc, g)))
}

/// Binding numbers of the pixel stage start here (vertex stage from 0).
pub const PS_BINDING_BASE: u32 = 32;

/// NGG primitive shader: program in the ES registers, resources and user
/// data in the GS ones (gfx10.3 register database).
const ES_PGM_LO: u32 = 0xC8;
const GS_RSRC2: u32 = 0x8B;
const GS_USER_DATA: u32 = 0x8C;
const VGT_SHADER_STAGES_EN: u32 = 0x28B54;

/// Resolves and translates the NGG (vertex) shader bound on `cp`.
pub fn prepare_ngg<R: RegRead + ?Sized>(cp: &R) -> Result<Prepared, String> {
    let sh = |i: u32| cp.sh(i);
    let stages = cp.ctx(VGT_SHADER_STAGES_EN);
    let addr = ((sh(ES_PGM_LO) as u64) | ((sh(ES_PGM_LO + 1) as u64 & 0xFF) << 32)) << 8;
    let rsrc2 = sh(GS_RSRC2);
    let user_sgprs = user_count(rsrc2);
    let user: [u32; 32] = std::array::from_fn(|i| sh(GS_USER_DATA + i as u32));
    let (insts, code_hash) = program_local(addr)?;
    let (ps_inputs, resources) = spec_vecs();
    let mut spec = Spec {
        stage: Stage::Vertex,
        user_sgprs,
        user_base: 8,
        code_base: addr,
        wave64: (stages >> 22) & 1 == 0,
        ps_input_addr: 0,
        ps_inputs,
        barycentric: false,
        mrt: [MrtSpec::default(); 8],
        cs_local_size: [1, 1, 1],
        cs_rsrc2: 0,
        resources,
        binding_base: 0,
        probe: None,
        dump: super::capture::vs_dump_for(addr).and_then(|(first, inst, count)| {
            let at = super::capture::vs_dump_pc().filter(|pc| insts.iter().any(|i| i.pc == *pc));
            at.or_else(|| pos_export(&insts).map(|i| i.pc)).map(|pc| (pc, [first, inst, count]))
        }),
        force_dispatcher: super::tune::block_chain() == 0,
        dispatcher_for_loops: super::tune::block_chain() == 2,
        reg_vars: super::tune::reg_vars(),
        lane_guards: super::tune::lane_guards(),
        mip_views: super::tune::mip_views(),
        lod_stats: super::tune::lod_stats() == 3,
        early_tests: false,
        per_lane: false,
        lane_votes: false,
        base_instance_sgpr: BASE_INSTANCE.with(|c| c.get()).map(|k| 8 + k),
        sgpr_pc: super::tune::sgpr_pc().then_some(translate::sgpr_pc_offset(Stage::Vertex)),
        addresses: super::tune::addresses() || batched_draw_now(),
        batched: batched_draw_now(),
        write_marks: super::resident::mirror_wanted() || super::fe2::wanted(),
        sgpr_ubo: super::tune::sgpr_ubo(),
        gpu_loads: 0,
        gpu_res: 0,
        pt_linear: false,
    };
    if spec.batched {
        if let Some((l, r)) = gpu_draw_masks_now(0) {
            spec.gpu_loads = l;
            spec.gpu_res = r;
            spec.pt_linear = super::fe2::on();
        }
    }
    let r = prepare(addr, insts, code_hash, &user, &mut spec);
    spec_vecs_back(spec);
    r
}

/// Program address of the pixel shader in a register view.
pub fn ps_addr_of<R: RegRead + ?Sized>(r: &R) -> u64 {
    ((r.sh(PS_PGM_LO) as u64) | ((r.sh(PS_PGM_LO + 1) as u64 & 0xFF) << 32)) << 8
}

/// Program address of the NGG (vertex) shader in a register view.
pub fn ngg_addr_of<R: RegRead + ?Sized>(r: &R) -> u64 {
    ((r.sh(ES_PGM_LO) as u64) | ((r.sh(ES_PGM_LO + 1) as u64 & 0xFF) << 32)) << 8
}

/// Program address of the pixel shader bound on `cp`.
pub fn ps_addr(cp: &Cp) -> u64 {
    let sh = |i: u32| cp.reg(Space::Sh, i);
    ((sh(PS_PGM_LO) as u64) | ((sh(PS_PGM_LO + 1) as u64 & 0xFF) << 32)) << 8
}

/// Program address of the NGG (vertex) shader bound on `cp`.
pub fn ngg_addr(cp: &Cp) -> u64 {
    let sh = |i: u32| cp.reg(Space::Sh, i);
    ((sh(ES_PGM_LO) as u64) | ((sh(ES_PGM_LO + 1) as u64 & 0xFF) << 32)) << 8
}

/// The first position export (`exp pos0`) of a vertex program.
pub fn pos_export(insts: &[rdna::Inst]) -> Option<&rdna::Inst> {
    insts.iter().find(|i| i.enc == rdna::Enc::Exp && i.exp_target() == 12)
}

/// One line per resolved resource (frame captures).
/// Where a prepared stage's CPU-side reads come from (wait traces): each
/// scalar load and each descriptor read from memory, with its guest range.
/// The V# of resource `r` of `plan` for user data `user`, read from guest
/// memory now (GPU-resolved chains, `gpures`: the range such a resource
/// reaches, imported before the shader reads it). None: unreadable.
pub fn resource_vsharp(code_base: u64, plan: &Plan, r: usize, user: &[u32]) -> Option<Vsharp> {
    let res = plan.resources.get(r)?;
    let mut rd = Reader::new(code_base);
    let mut w = [0u32; 4];
    for (k, p) in res.dwords.iter().take(4).enumerate() {
        w[k] = rd.eval(p, user)?;
    }
    Some(Vsharp::decode(&w))
}

/// The guest range scalar load `k` of `plan` reads for user data `user`,
/// its pointer chain read from guest memory now (GPU-resolved loads,
/// `gpures`: imported before the shader reads it). None: unreadable.
pub fn load_range(code_base: u64, plan: &Plan, k: usize, user: &[u32]) -> Option<(u64, u64)> {
    let l = plan.loads.get(k)?;
    let mut rd = Reader::new(code_base);
    let base = rd.ptr(&l.ptr, user)?;
    Some((base.wrapping_add(l.offset as u64), 4 * l.dwords as u64))
}

/// Front end 2's GPU-resolved chains (stage F3, `exec::fe2_gpu_ranges`):
/// into `loads` the guest ranges the shader-resolved loads of `spec`
/// (`gpu_loads`) read, into `bufs` those its shader-resolved buffers
/// (`gpu_res`) reach (u64::MAX bytes: to the end of the mapping), for user
/// data `user`, their pointer chains and V#s read from guest memory now
/// (forwarded words as `Reader` reads them); into `reads` the words read for
/// that. Err(0): a chain is unreadable; Err(1): a formatted buffer's V# has
/// another format or swizzle than the translation's (its loads decode with
/// that one).
pub fn gpu_chain_ranges(code_base: u64, plan: &Plan, spec: &Spec, user: &[u32], loads: &mut Vec<(u64, u64)>, bufs: &mut Vec<(u64, u64)>, reads: &mut Vec<(u64, u64)>) -> Result<(), u8> {
    let mut rd = Reader::new(code_base);
    for (k, l) in plan.loads.iter().enumerate().take(64) {
        if spec.gpu_loads & (1u64 << k) == 0 {
            continue;
        }
        let base = rd.ptr(&l.ptr, user).ok_or(0u8)?;
        loads.push((base.wrapping_add(l.offset as u64), 4 * l.dwords as u64));
    }
    for (r, res) in plan.resources.iter().enumerate().take(64) {
        if spec.gpu_res & (1u64 << r) == 0 {
            continue;
        }
        let mut w = [0u32; 4];
        for (k, p) in res.dwords.iter().take(4).enumerate() {
            w[k] = rd.eval(p, user).ok_or(0u8)?;
        }
        let v = Vsharp::decode(&w);
        if res.formatted {
            if let Some(translate::ResSpec::Buffer { format, dst_sel }) = spec.resources.get(r) {
                if v.format != *format || v.dst_sel != *dst_sel {
                    return Err(1);
                }
            }
        }
        // Without range checks (OOB_SELECT 2) the shader reaches past the
        // records: to the end of the mapping, as the CPU path binds it
        // (u64::MAX here; `exec::fe2_gpu_ranges` clips it).
        let n = v.byte_size().unwrap_or(u64::MAX);
        if n > 0 && v.base != 0 {
            bufs.push((v.base, n));
        }
    }
    reads.extend_from_slice(&rd.reads);
    Ok(())
}

pub fn read_sources(p: &Prepared) -> Vec<(String, u64, u64)> {
    let mut rd = Reader::new(p.addr);
    let mut v = Vec::new();
    for (k, l) in p.tr.plan.loads.iter().enumerate() {
        if let Some(b) = rd.ptr(&l.ptr, &p.user) {
            v.push((format!("load {k} {:?}+{:#x} x{}", l.ptr, l.offset, l.dwords), b + l.offset as u64, 4 * l.dwords as u64));
        }
    }
    for (i, a) in p.res_addrs.iter().enumerate() {
        if let Some(a) = a {
            let res = &p.tr.plan.resources[i];
            v.push((format!("resource {i} {:?} {:?}", res.kind, res.dwords.first()), *a, 4 * res.dwords.len() as u64));
        }
    }
    v
}

pub fn describe_resources(p: &Prepared) -> Vec<String> {
    p.resources
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let used = &p.tr.plan.resources[i];
            let from = match p.res_addrs.get(i).copied().flatten() {
                Some(a) => format!(" @{a:#x}"),
                None => String::new(),
            };
            let rw = format!("{}{}{from}", if used.read { "r" } else { "" }, if used.written { "w" } else { "" });
            match r {
                Resolved::Buffer(v) => format!("{i}: V# {:#x} stride {} records {} format {} dst_sel {:#x} oob {} {rw}", v.base, v.stride, v.num_records, v.format, v.dst_sel, v.oob_select),
                Resolved::Global(a, n) => format!("{i}: global {a:#x}{} {rw}", n.map(|n| format!("+{n:#x}")).unwrap_or_default()),
                Resolved::ImageTable(e) => {
                    let list: Vec<String> = e.iter().enumerate().filter_map(|(k, t)| t.map(|t| format!("{k}:{:#x} {}x{}x{} f{}", t.base, t.width, t.height, t.depth, t.format))).collect();
                    format!("{i}: image table of {} entries {rw} [{}]", e.len(), list.join(", "))
                }
                Resolved::Image(t) => format!(
                    "{i}: T# {:#x} format {} {}x{}x{} type {} sw_mode {} levels {}..{} base_array {} dst_sel {:#x} {rw} [{}]",
                    t.base,
                    t.format,
                    t.width,
                    t.height,
                    t.depth,
                    t.ty,
                    t.sw_mode,
                    t.base_level,
                    t.last_level,
                    t.base_array,
                    t.dst_sel,
                    t.raw.iter().map(|w| format!("{w:08x}")).collect::<Vec<_>>().join(" ")
                ),
                Resolved::Sampler(w) => format!("{i}: S# {:08x} {:08x} {:08x} {:08x} {rw}", w[0], w[1], w[2], w[3]),
            }
        })
        .collect()
}

/// One line per resolved scalar load: pointer provenance, offset and the
/// loaded dwords (hex, and as floats) (frame captures).
pub fn describe_loads(p: &Prepared) -> Vec<String> {
    let offs = translate::load_offsets(&p.tr.plan);
    p.tr
        .plan
        .loads
        .iter()
        .zip(offs)
        .map(|(l, o)| {
            let vals: Vec<String> = (0..l.dwords as usize)
                .map(|k| {
                    let w = p.sgpr_data.get(o as usize + k).copied().unwrap_or(0);
                    let f = f32::from_bits(w);
                    if f.is_finite() && f != 0.0 && (f.abs() < 1e6 && f.abs() > 1e-6) { format!("{w:08x}({f})") } else { format!("{w:08x}") }
                })
                .collect();
            format!("load {:?} + {:#x}: {}", l.ptr, l.offset, vals.join(" "))
        })
        .collect()
}

/// How often a stage's preparation repeats the previous one on its thread
/// (profile): [calls, same program / user data / register state, and the
/// same bytes read] per stage (vertex, pixel, compute).
pub static REPEATS: [[std::sync::atomic::AtomicU64; 3]; 4] = [const { [const { std::sync::atomic::AtomicU64::new(0) }; 3] }; 4];
/// The same with only the user-data dwords that descriptors and scalar
/// loads depend on in the key: [same deps, also same bytes] per stage.
pub static REPEATS_DEPS: [[std::sync::atomic::AtomicU64; 2]; 4] = [const { [const { std::sync::atomic::AtomicU64::new(0) }; 2] }; 4];

thread_local! {
    static LAST_PREP: std::cell::RefCell<[(u64, u64); 4]> = const { std::cell::RefCell::new([(0, 0); 4]) };
    static LAST_DEPS: std::cell::RefCell<[(u64, u64); 4]> = const { std::cell::RefCell::new([(0, 0); 4]) };
    /// This thread is a compute queue's command processor (statistics).
    static COMPUTE_QUEUE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// User-data dwords that a plan's descriptors and scalar loads depend on.
fn user_deps(plan: &rdna::analysis::Plan) -> u32 {
    fn walk_ptr(p: &Ptr, m: &mut u32) {
        match p {
            Ptr::User(i) => *m |= 3 << (*i).min(30),
            Ptr::Mem(inner, _) => walk_ptr(inner, m),
            Ptr::Pc(_) => {}
        }
    }
    fn walk(p: &Prov, m: &mut u32) {
        match p {
            Prov::User(i) => *m |= 1 << (*i).min(31),
            Prov::Mem(ptr, _) | Prov::PtrLo(ptr, _) | Prov::PtrHi(ptr, _) => walk_ptr(ptr, m),
            _ => {}
        }
    }
    let mut m = 0u32;
    for r in &plan.resources {
        for d in &r.dwords {
            walk(d, &mut m);
        }
    }
    for l in &plan.loads {
        walk_ptr(&l.ptr, &mut m);
    }
    m
}

/// Preparations whose inputs and bytes read match one of the previous frame
/// (profile; how much a cache across frames could serve): per stage
/// [calls, same full inputs, same inputs counting only the user data the
/// descriptors depend on].
pub static CROSS_FRAME: [[std::sync::atomic::AtomicU64; 3]; 4] = [const { [const { std::sync::atomic::AtomicU64::new(0) }; 3] }; 4];

/// Marks the calling thread as a compute queue's command processor: its
/// compute preparations are counted apart from the graphics queue's
/// (`repeats_line`).
pub fn set_compute_queue(on: bool) {
    COMPUTE_QUEUE.with(|c| c.set(on));
}

/// The repeat statistics' class of a stage: vertex, pixel, compute on the
/// graphics queue (and its workers), compute on a compute queue.
fn stat_slot(stage: Stage) -> usize {
    match stage {
        Stage::Vertex => 0,
        Stage::Fragment => 1,
        _ if COMPUTE_QUEUE.with(|c| c.get()) => 3,
        _ => 2,
    }
}

/// (frame, this frame's (kind, stage, key, bytes) set, the previous frame's).
type FrameSets = (u64, super::fx::Set<(u8, u8, u64, u64)>, super::fx::Set<(u8, u8, u64, u64)>);
static FRAME_SETS: parking_lot::Mutex<Option<FrameSets>> = parking_lot::Mutex::new(None);

/// Notes a preparation's (kind, key, bytes) for the cross-frame statistic;
/// true when the previous frame had the same.
fn seen_last_frame(kind: u8, stage: usize, key: u64, bytes: u64) -> bool {
    let frame = super::FRAMES.load(std::sync::atomic::Ordering::Relaxed);
    let mut g = FRAME_SETS.lock();
    let st = g.get_or_insert_with(|| (frame, Default::default(), Default::default()));
    if st.0 != frame {
        st.2 = std::mem::take(&mut st.1);
        st.0 = frame;
    }
    let k = (kind, stage as u8, key, bytes);
    let hit = st.2.contains(&k);
    st.1.insert(k);
    hit
}

fn note_repeat_deps(stage: Stage, key: u64, bytes: u64) {
    use std::sync::atomic::Ordering;
    let s = stat_slot(stage);
    if seen_last_frame(1, s, key, bytes) {
        CROSS_FRAME[s][2].fetch_add(1, Ordering::Relaxed);
    }
    LAST_DEPS.with(|c| {
        let mut c = c.borrow_mut();
        if c[s].0 == key {
            REPEATS_DEPS[s][0].fetch_add(1, Ordering::Relaxed);
            if c[s].1 == bytes {
                REPEATS_DEPS[s][1].fetch_add(1, Ordering::Relaxed);
            }
        }
        c[s] = (key, bytes);
    });
}

fn note_repeat(stage: Stage, key: u64, reads: &[(u64, u64)]) -> u64 {
    use std::sync::atomic::Ordering;
    let s = stat_slot(stage);
    let mut h = super::fx::FxHasher::default();
    for &(a, n) in reads {
        h.write_u64(a);
        // One check per range: per-word checks made this profile-only
        // statistic most of the readable-range checks (~7.6 million per
        // second, run 20261011-115142) and inflated their profile share.
        if !super::cp::guest_readable(a, n) {
            continue;
        }
        let mut k = 0;
        while k + 4 <= n {
            h.write_u32(unsafe { std::ptr::read_volatile((a + k) as *const u32) });
            k += 4;
        }
    }
    let bytes = h.finish();
    REPEATS[s][0].fetch_add(1, Ordering::Relaxed);
    CROSS_FRAME[s][0].fetch_add(1, Ordering::Relaxed);
    if seen_last_frame(0, s, key, bytes) {
        CROSS_FRAME[s][1].fetch_add(1, Ordering::Relaxed);
    }
    LAST_PREP.with(|c| {
        let mut c = c.borrow_mut();
        if c[s].0 == key {
            REPEATS[s][1].fetch_add(1, Ordering::Relaxed);
            if c[s].1 == bytes {
                REPEATS[s][2].fetch_add(1, Ordering::Relaxed);
            }
        }
        c[s] = (key, bytes);
    });
    bytes
}

/// One line on how often preparations repeat (profile report).
pub fn repeats_line() -> String {
    use std::sync::atomic::Ordering;
    let names = ["vertex", "pixel", "compute (graphics queue)", "compute (compute queues)"];
    let parts: Vec<String> = (0..4)
        .map(|s| {
            let r = &REPEATS[s];
            let n = r[0].load(Ordering::Relaxed).max(1);
            let d = &REPEATS_DEPS[s];
            format!(
                "{} {} ({}% same inputs, {}% also same bytes; with only the user data descriptors depend on {}%, {}%)",
                names[s],
                r[0].load(Ordering::Relaxed),
                r[1].load(Ordering::Relaxed) * 100 / n,
                r[2].load(Ordering::Relaxed) * 100 / n,
                d[0].load(Ordering::Relaxed) * 100 / n,
                d[1].load(Ordering::Relaxed) * 100 / n
            )
        })
        .collect();
    let (h, m) = (PREP_HITS.load(Ordering::Relaxed), PREP_MISSES.load(Ordering::Relaxed));
    let cross: Vec<String> = (0..4)
        .map(|s| {
            let c = &CROSS_FRAME[s];
            let n = c[0].load(Ordering::Relaxed).max(1);
            format!("{} {}% ({}% with only the user data descriptors depend on)", names[s], c[1].load(Ordering::Relaxed) * 100 / n, c[2].load(Ordering::Relaxed) * 100 / n)
        })
        .collect();
    format!(
        "    preparations repeating the previous one: {}; served from the cache {h} of {}\n    preparations identical to one of the previous frame (inputs and bytes read): {}\n",
        parts.join(", "),
        h + m,
        cross.join(", ")
    )
}

/// Stage preparations served from the cache since `profile reset` (the
/// sampler-only report, which has no `repeats_line`).
pub fn prep_cache_line() -> String {
    use std::sync::atomic::Ordering::Relaxed;
    let (h, m) = (PREP_HITS.load(Relaxed), PREP_MISSES.load(Relaxed));
    format!("    stage preparations: served from the cache {h} of {} (prep_cache {})\n", h + m, super::tune::prep_cache_mode())
}

pub fn reset_repeats() {
    for r in CROSS_FRAME.iter() {
        for c in r.iter() {
            c.store(0, std::sync::atomic::Ordering::Relaxed);
        }
    }
    for r in REPEATS.iter() {
        for c in r.iter() {
            c.store(0, std::sync::atomic::Ordering::Relaxed);
        }
    }
    for r in REPEATS_DEPS.iter() {
        for c in r.iter() {
            c.store(0, std::sync::atomic::Ordering::Relaxed);
        }
    }
    PREP_HITS.store(0, std::sync::atomic::Ordering::Relaxed);
    PREP_MISSES.store(0, std::sync::atomic::Ordering::Relaxed);
}

/// A stage's last preparation on this thread: reused for the next one with
/// the same program, register state and the user-data dwords its descriptors
/// and loads depend on, when the guest words it read are unchanged. More than
/// half of the draws' vertex and pixel preparations repeat that way (run
/// 20261011-091944); dispatches do not.
struct PrepEntry {
    /// Its program and plan, besides the key (a hash).
    addr: u64,
    plan_key: u64,
    /// Validated by the words it read (`words_unchanged`).
    prepared: Prepared,
}

/// Entries kept per stage and thread with `tune prep_cache 2` (cleared when
/// full): a frame's distinct preparations on one recording thread.
const PREP_CACHE_MAX: usize = 2048;

thread_local! {
    /// Vertex and pixel preparations by key (`prepare`): the latest only
    /// (`tune prep_cache 1`) or up to PREP_CACHE_MAX (2): most preparations
    /// that miss the latest repeat one of the previous frame (pixel 57%,
    /// vertex 24%; run 20261013-104455). Live, 2 serves 59% instead of 43%
    /// yet costs ~6% more CPU per command (inserts, re-validations after
    /// mapping changes, a colder working set; run 20261013-115322).
    static PREP_CACHE: std::cell::RefCell<[super::fx::Map<u64, PrepEntry>; 2]> = std::cell::RefCell::new([super::fx::Map::default(), super::fx::Map::default()]);
    /// The latest preparation per stage (`tune prep_cache 1`; kept apart so
    /// that alternating the modes in a replay A/B leaves the map as it was).
    static PREP_LATEST: std::cell::RefCell<[Option<(u64, PrepEntry)>; 2]> = const { std::cell::RefCell::new([None, None]) };
    static LOCAL_DEPS: std::cell::RefCell<super::fx::Map<u64, u32>> = std::cell::RefCell::new(super::fx::Map::default());
    /// `format_resources` per plan.
    static LOCAL_FORMATS: std::cell::RefCell<super::fx::Map<u64, u64>> = std::cell::RefCell::new(super::fx::Map::default());
}

/// The resources of `plan` whose V# format and swizzle the translation
/// reads (format loads and stores, typed buffer ones included; a bit per
/// resource, all past 64). Other buffers keep them out of their spec: one
/// translation whatever the title's descriptors hold there (and a spec a
/// shader object alone can give, `guess_compute_spec`).
fn format_resources(insts: &[rdna::Inst], plan: &Plan) -> u64 {
    if plan.resources.len() > 64 {
        return u64::MAX;
    }
    let mut m = 0u64;
    for i in insts {
        let Some(n) = i.name() else { continue };
        let n = n.strip_prefix('t').unwrap_or(n);
        if n.starts_with("buffer_load_format_") || n.starts_with("buffer_store_format_") {
            if let Some(&(r, _)) = plan.inst_res.get(&i.pc) {
                m |= 1 << r.min(63);
            }
        }
    }
    m
}

/// Translates `insts` (planned as `plan`) for `spec` and keeps the
/// translation under (plan_key, spec_hash) for `key_spec` (the spec it is
/// kept for: the code base left out).
fn translate_kept(addr: u64, insts: &[rdna::Inst], plan: &Arc<Plan>, plan_key: u64, spec_hash: u64, spec: &Spec, key_spec: Spec) -> Result<Arc<Translated>, String> {
    let t0 = std::time::Instant::now();
    let out = translate::translate(insts, plan, spec).map_err(|e| format!("{:?} shader {addr:#x}: {e}", spec.stage))?;
    super::hitch::note_since(super::hitch::Kind::Translate, t0, || format!("{:?} {addr:#x}", spec.stage));
    let ms = t0.elapsed().as_millis();
    if ms >= 100 {
        crate::info!("{:?} shader {addr:#x}: translated in {ms} ms ({} instructions, {} SPIR-V words)", spec.stage, insts.len(), out.spirv.len());
    }
    // Large translations compile for seconds on first use (play stalled
    // 0.5-3.4 s per effect shader, FINDINGS.md 2026-10-14): each logged
    // once, its program saved.
    if out.spirv.len() >= BIG_SPIRV_WORDS && BIG_SPIRV_SEEN.lock().get_or_insert_with(std::collections::HashSet::new).insert(addr) {
        let saved = super::capture::save_program(addr).map(|f| format!("; program saved to {}", f.display())).unwrap_or_default();
        let from = program_hash(addr).ok().and_then(super::particles::origin).map(|d| format!("; from {d}")).unwrap_or_default();
        crate::info!(
            "{:?} shader {addr:#x}: {} instructions -> {} SPIR-V words ({:.0} per instruction; wave64 {}, per lane {}, lane votes {}, local size {:?}){from}{saved}",
            spec.stage,
            insts.len(),
            out.spirv.len(),
            out.spirv.len() as f64 / insts.len().max(1) as f64,
            spec.wave64,
            spec.per_lane,
            spec.lane_votes,
            spec.cs_local_size
        );
    }
    for w in &out.warnings {
        crate::warn!("{:?} shader {addr:#x}: {w}", spec.stage);
    }
    let mut atomic_res: Vec<usize> = insts
        .iter()
        .filter(|i| i.name().is_some_and(|n| n.starts_with("buffer_atomic_") || n.starts_with("global_atomic_")))
        .filter_map(|i| plan.inst_res.get(&i.pc).map(|r| r.0))
        .collect();
    atomic_res.sort_unstable();
    atomic_res.dedup();
    let t = Arc::new(Translated { out, plan: Arc::clone(plan), key: plan_key ^ spec_hash.rotate_left(1), atomic_res, spec: key_spec });
    TRANSLATIONS.lock().get_or_insert_with(HashMap::new).insert((plan_key, spec_hash), t.clone());
    Ok(t)
}

/// The code hash of the program at `addr` (as `prepare` keys it).
pub(crate) fn program_hash(addr: u64) -> Result<u64, String> {
    program_local(addr).map(|(_, h)| h)
}

/// The words of the program at `addr` its code hash covers (up to its last
/// instruction), and that hash (`gpu::precompile` keeps them).
pub(crate) fn program_code(addr: u64) -> Result<(Vec<u32>, u64), String> {
    let code = code_slice(addr).ok_or_else(|| format!("shader {addr:#x} not mapped"))?;
    let insts = rdna::decode_program(code).map_err(|e| format!("decode {addr:#x}: {}", e.0))?;
    let last = insts.last().ok_or("empty program")?;
    let words = ((last.pc / 4 + last.len as u32) as usize).min(code.len());
    Ok((code[..words].to_vec(), hash_words(&code[..words])))
}

/// The translation of stored program words (`gpu::precompile`, before the
/// title has loaded the program) for `key_spec`, kept where `prepare`
/// looks for it (translations do not depend on the program's address).
pub(crate) fn translate_code_for_spec(code: &[u32], code_hash: u64, key_spec: Spec) -> Result<Arc<Translated>, String> {
    if super::tune::gpu_loads_mode() != 0 {
        return Err("gpu_loads".into());
    }
    if hash_words(code) != code_hash {
        return Err("stored code does not match its hash".into());
    }
    let plan_key = code_hash ^ (key_spec.user_sgprs as u64) << 48 ^ (key_spec.user_base as u64) << 56;
    let spec_hash = {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        key_spec.hash(&mut h);
        h.finish()
    };
    if let Some(t) = TRANSLATIONS.lock().as_ref().and_then(|m| m.get(&(plan_key, spec_hash)).cloned()) {
        return Ok(t);
    }
    let insts = rdna::decode_program(code).map_err(|e| format!("decode stored code: {}", e.0))?;
    let cached = PLANS.lock().as_ref().and_then(|m| m.get(&plan_key).cloned());
    let plan = match cached {
        Some(p) => p,
        None => {
            let p = Arc::new(rdna::analysis::analyze_opts(&insts, key_spec.user_sgprs, key_spec.user_base, false));
            PLANS.lock().get_or_insert_with(HashMap::new).insert(plan_key, p.clone());
            p
        }
    };
    if !plan.errors.is_empty() {
        return Err(plan.errors.join("; "));
    }
    translate_kept(0, &insts, &plan, plan_key, spec_hash, &key_spec.clone(), key_spec)
}

/// `program_hash` without keeping the decoded program (`gpu::precompile`
/// looks at every shader object the title creates, ~40,000 in a session,
/// and makes pipelines for few).
pub(crate) fn program_hash_uncached(addr: u64) -> Result<u64, String> {
    if let Some(p) = PROGRAMS.lock().as_ref().and_then(|m| m.get(&addr).map(|p| (p.words, p.hash))) {
        if let Some(code) = code_slice(addr) {
            if p.0 <= code.len() && hash_words(&code[..p.0]) == p.1 {
                return Ok(p.1);
            }
        }
    }
    let code = code_slice(addr).ok_or_else(|| format!("shader {addr:#x} not mapped"))?;
    let insts = rdna::decode_program(code).map_err(|e| format!("decode {addr:#x}: {}", e.0))?;
    let last = insts.last().ok_or("empty program")?;
    let words = (last.pc / 4 + last.len as u32) as usize;
    Ok(hash_words(&code[..words.min(code.len())]))
}

/// The decoded program at `addr` and its plan for `user_sgprs` user-data
/// dwords from SGPR `user_base` (as `prepare` makes them without GPU-side
/// data loads).
pub(crate) fn stage_plan(addr: u64, user_sgprs: u16, user_base: u16) -> Result<(Arc<Local<Vec<rdna::Inst>>>, Arc<Plan>, u64), String> {
    let (insts, code_hash) = program_local(addr)?;
    let plan_key = code_hash ^ (user_sgprs as u64) << 48 ^ (user_base as u64) << 56;
    let cached = PLANS.lock().as_ref().and_then(|m| m.get(&plan_key).cloned());
    let plan = match cached {
        Some(p) => p,
        None => {
            let p = Arc::new(rdna::analysis::analyze_opts(&insts, user_sgprs, user_base, false));
            PLANS.lock().get_or_insert_with(HashMap::new).insert(plan_key, p.clone());
            p
        }
    };
    if !plan.errors.is_empty() {
        return Err(plan.errors.join("; "));
    }
    Ok((insts, plan, code_hash))
}

/// The translation of the program at `addr` for `key_spec` (a key spec as
/// `compute_spec`, `vertex_spec` or `pixel_spec` make it; `gpu::precompile`),
/// kept where `prepare` looks for it.
pub(crate) fn translate_for_spec(addr: u64, key_spec: Spec) -> Result<Arc<Translated>, String> {
    // Only the default form (data loads resolved by the CPU) is made ahead.
    if super::tune::gpu_loads_mode() != 0 {
        return Err("gpu_loads".into());
    }
    let (insts, plan, code_hash) = stage_plan(addr, key_spec.user_sgprs, key_spec.user_base)?;
    let plan_key = code_hash ^ (key_spec.user_sgprs as u64) << 48 ^ (key_spec.user_base as u64) << 56;
    let spec_hash = {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        key_spec.hash(&mut h);
        h.finish()
    };
    if let Some(t) = TRANSLATIONS.lock().as_ref().and_then(|m| m.get(&(plan_key, spec_hash)).cloned()) {
        return Ok(t);
    }
    let mut spec = key_spec.clone();
    spec.code_base = addr;
    translate_kept(addr, &insts, &plan, plan_key, spec_hash, &spec, key_spec)
}

/// The translation spec of a compute program at `addr` guessed from its
/// shader object (`gpu::precompile`): the user-data count, RSRC2 and thread
/// group size it carries; wave64 unless the code saves EXEC in 32 bits;
/// images by their instructions' dimension as float views, those sampled
/// with a depth comparison as depth views; buffers as `prepare` keys them
/// (unknown for format accesses).
pub(crate) fn guess_compute_spec(addr: u64, insts: &[rdna::Inst], plan: &Plan, code_hash: u64, rsrc2: u32, local: [u32; 3]) -> (Spec, bool) {
    let wave64 = !insts.iter().any(|i| i.name().is_some_and(|n| n.ends_with("_saveexec_b32") || n.ends_with("_wrexec_b32")));
    let fmt = format_resources(insts, plan);
    let user_sgprs = ((rsrc2 >> 1) & 0x1F).min(16) as u16;
    let plan_key = code_hash ^ (user_sgprs as u64) << 48;
    let cmp = super::tune::cmp_emulate().then(|| compare_images(plan_key, insts, plan)).unwrap_or(0);
    let mut known = true;
    let resources = plan
        .resources
        .iter()
        .enumerate()
        .map(|(r, res)| match res.kind {
            ResKind::Buffer if fmt & (1 << r.min(63)) != 0 => {
                known = false;
                ResSpec::Buffer { format: 0, dst_sel: 0xFAC }
            }
            ResKind::Buffer => ResSpec::Buffer { format: 0, dst_sel: 0 },
            ResKind::Global => ResSpec::Global,
            ResKind::Sampler => ResSpec::Sampler,
            ResKind::Image | ResKind::ImageTable => {
                let dim = insts
                    .iter()
                    .find(|i| i.enc == rdna::Enc::Mimg && plan.inst_res.get(&i.pc).is_some_and(|x| x.0 == r))
                    .map(|i| match i.mimg_dim() {
                        0 => ImageDim::D1,
                        1 => ImageDim::D2,
                        2 => ImageDim::D3,
                        3 => ImageDim::Cube,
                        4 => ImageDim::D1Array,
                        5 => ImageDim::D2Array,
                        6 => ImageDim::D2Msaa,
                        _ => ImageDim::D2MsaaArray,
                    })
                    .unwrap_or(ImageDim::D2);
                // As `prepare` decides: images keep depth views without
                // comparison emulation, else only the compared ones.
                let depth = match res.kind {
                    ResKind::ImageTable => false,
                    _ if !super::tune::cmp_emulate() => true,
                    _ => r < 64 && cmp & (1 << r) != 0,
                };
                ResSpec::Image { dim, class: NumClass::Float, depth }
            }
        })
        .collect();
    (compute_spec(addr, code_hash, rsrc2, local, wave64, resources), known)
}

/// The key spec (code base left out) `prepare_ngg` makes for the vertex
/// program at `addr` drawn with these registers and descriptors, under the
/// current switches (None while vertex dumps are on: they depend on the
/// code).
pub(crate) fn vertex_spec(addr: u64, user_sgprs: u16, wave64: bool, base_instance_sgpr: Option<u16>, resources: Vec<ResSpec>) -> Option<Spec> {
    if super::capture::vs_dump().is_some() || super::capture::vs_dump_for(addr).is_some() {
        return None;
    }
    let _ = addr;
    Some(Spec {
        stage: Stage::Vertex,
        user_sgprs,
        user_base: 8,
        code_base: 0,
        wave64,
        ps_input_addr: 0,
        ps_inputs: Vec::new(),
        barycentric: false,
        mrt: [MrtSpec::default(); 8],
        cs_local_size: [1, 1, 1],
        cs_rsrc2: 0,
        resources,
        binding_base: 0,
        probe: None,
        dump: None,
        force_dispatcher: super::tune::block_chain() == 0,
        dispatcher_for_loops: super::tune::block_chain() == 2,
        reg_vars: super::tune::reg_vars(),
        lane_guards: super::tune::lane_guards(),
        mip_views: super::tune::mip_views(),
        lod_stats: super::tune::lod_stats() == 3,
        early_tests: false,
        per_lane: false,
        lane_votes: false,
        base_instance_sgpr,
        sgpr_pc: super::tune::sgpr_pc().then_some(translate::sgpr_pc_offset(Stage::Vertex)),
        addresses: super::tune::addresses(),
        batched: false,
        write_marks: super::resident::mirror_wanted() || super::fe2::wanted(),
        sgpr_ubo: super::tune::sgpr_ubo(),
        gpu_loads: 0,
        gpu_res: 0,
        pt_linear: false,
    })
}

/// The key spec `prepare_ps` makes for the pixel program at `addr` drawn
/// with these registers and descriptors, under the current switches.
#[allow(clippy::too_many_arguments)]
pub(crate) fn pixel_spec(addr: u64, user_sgprs: u16, wave64: bool, ps_input_addr: u32, ps_inputs: Vec<PsInput>, mrt: [MrtSpec; 8], early_tests: bool, resources: Vec<ResSpec>) -> Spec {
    Spec {
        stage: Stage::Fragment,
        user_sgprs,
        user_base: 0,
        code_base: 0,
        wave64,
        ps_input_addr,
        ps_inputs,
        barycentric: super::vk::ctx().barycentric,
        mrt,
        cs_local_size: [1, 1, 1],
        cs_rsrc2: 0,
        resources,
        binding_base: PS_BINDING_BASE,
        probe: probe_for(addr),
        dump: None,
        force_dispatcher: super::tune::block_chain() == 0,
        dispatcher_for_loops: super::tune::block_chain() == 2,
        reg_vars: super::tune::reg_vars(),
        lane_guards: super::tune::lane_guards(),
        mip_views: super::tune::mip_views(),
        lod_stats: super::tune::lod_stats() == 3,
        early_tests,
        per_lane: false,
        lane_votes: false,
        base_instance_sgpr: None,
        sgpr_pc: super::tune::sgpr_pc().then_some(translate::sgpr_pc_offset(Stage::Fragment)),
        addresses: super::tune::addresses(),
        batched: false,
        write_marks: super::resident::mirror_wanted() || super::fe2::wanted(),
        sgpr_ubo: super::tune::sgpr_ubo(),
        gpu_loads: 0,
        gpu_res: 0,
        pt_linear: false,
    }
}

/// The key spec (code base left out) `prepare_compute` makes for the
/// compute program at `addr` (code hash `code_hash`) dispatched with these
/// registers and descriptors, under the current switches.
pub(crate) fn compute_spec(addr: u64, code_hash: u64, rsrc2: u32, local: [u32; 3], wave64: bool, resources: Vec<ResSpec>) -> Spec {
    let mut spec = Spec {
        stage: Stage::Compute,
        user_sgprs: ((rsrc2 >> 1) & 0x1F).min(16) as u16,
        user_base: 0,
        code_base: 0,
        wave64,
        ps_input_addr: 0,
        ps_inputs: Vec::new(),
        barycentric: false,
        mrt: [MrtSpec::default(); 8],
        cs_local_size: local,
        cs_rsrc2: rsrc2,
        resources,
        binding_base: 0,
        probe: probe_for(addr),
        dump: dump_for(addr),
        force_dispatcher: super::tune::block_chain() == 0,
        dispatcher_for_loops: super::tune::block_chain() == 2,
        reg_vars: super::tune::reg_vars(),
        lane_guards: super::tune::lane_guards(),
        mip_views: super::tune::mip_views(),
        lod_stats: super::tune::lod_stats() == 3,
        early_tests: false,
        per_lane: false,
        lane_votes: false,
        base_instance_sgpr: None,
        sgpr_pc: super::tune::sgpr_pc().then_some(translate::sgpr_pc_offset(Stage::Compute)),
        addresses: super::tune::addresses() || (super::tune::batch_cs() && batchable(addr)),
        batched: super::tune::batch_cs() && batchable(addr),
        write_marks: super::resident::mirror_wanted() || super::fe2::wanted(),
        sgpr_ubo: super::tune::sgpr_ubo(),
        gpu_loads: 0,
        gpu_res: 0,
        pt_linear: false,
    };
    if super::tune::per_lane() != 0 && per_lane_program(code_hash) {
        let votes = lane_votes_program(code_hash);
        spec.per_lane = super::tune::per_lane() == 1 || !votes;
        spec.lane_votes = spec.per_lane && votes;
    }
    spec
}
/// Preparations served from / missing PREP_CACHE (diagnostics).
pub static PREP_HITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static PREP_MISSES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Cache hits compared with a full preparation (DS_PREP_CHECK=1) and those
/// that differed.
static PREP_CHECKED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static PREP_DIFFERED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Differences explained by guest writes during the check.
static PREP_RACED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// DS_PREP_CHECK=1: every PREP_CACHE hit is prepared in full as well and
/// the two are compared (a correctness check of the cache's key and
/// validation; the full preparation is used).
fn prep_check() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_PREP_CHECK").ok().as_deref() == Some("1"))
}

/// Compares a cache hit with the full preparation (`prep_check`).
fn prep_compare(hit: &Prepared, full: &Prepared) {
    use std::sync::atomic::Ordering::Relaxed;
    let mut diff = Vec::new();
    if hit.addr != full.addr {
        diff.push("address");
    }
    if !Arc::ptr_eq(&hit.tr.0, &full.tr.0) && hit.tr.spec != full.tr.spec {
        diff.push("translation");
    }
    if hit.resources != full.resources {
        diff.push("resources");
    }
    if hit.res_addrs != full.res_addrs {
        diff.push("descriptor addresses");
    }
    if hit.sgpr_data != full.sgpr_data {
        diff.push("SGPR data");
    }
    if hit.user != full.user {
        diff.push("user data");
    }
    if hit.reads != full.reads || hit.words != full.words || hit.unread != full.unread {
        diff.push("reads");
    }
    if hit.data_loads != full.data_loads {
        diff.push("data loads");
    }
    if (hit.cmp_images, hit.cmp_depth) != (full.cmp_images, full.cmp_depth) {
        diff.push("depth views");
    }
    let n = PREP_CHECKED.fetch_add(1, Relaxed) + 1;
    if !diff.is_empty() {
        if !words_unchanged(hit) {
            // The title wrote memory the cached preparation read between the
            // cache's check and the full preparation: a full preparation a
            // moment earlier would have matched (not a cache error).
            let r = PREP_RACED.fetch_add(1, Relaxed) + 1;
            if r <= 5 {
                crate::info!("prep check: {:?} shader {:#x}: memory it read changed during the check (differs in {}; not counted)", full.stage, full.addr, diff.join(", "));
            }
        } else {
            let d = PREP_DIFFERED.fetch_add(1, Relaxed) + 1;
            if d <= 20 {
                let mut detail = Vec::new();
                if let Some(i) = hit.reads.iter().zip(full.reads.iter()).position(|(a, b)| a != b) {
                    detail.push(format!("read {i}: cached {:x?}, full {:x?}", hit.reads[i], full.reads[i]));
                } else if hit.reads.len() != full.reads.len() {
                    detail.push(format!("reads: cached {}, full {} ranges (full's next {:x?})", hit.reads.len(), full.reads.len(), full.reads.get(hit.reads.len())));
                }
                if let Some(i) = hit.words.iter().zip(full.words.iter()).position(|(a, b)| a != b) {
                    detail.push(format!("word {i}: cached {:#x}, full {:#x}", hit.words[i], full.words[i]));
                }
                if hit.unread != full.unread {
                    detail.push(format!("unread: cached {:x?}, full {:x?}", &hit.unread[..hit.unread.len().min(4)], &full.unread[..full.unread.len().min(4)]));
                }
                if let Some(i) = hit.sgpr_data.iter().zip(full.sgpr_data.iter()).position(|(a, b)| a != b) {
                    detail.push(format!("SGPR data {i}: cached {:#x}, full {:#x}", hit.sgpr_data[i], full.sgpr_data[i]));
                }
                crate::warn!("prep check: {:?} shader {:#x}: the cached preparation differs from a full one in {} ({})", full.stage, full.addr, diff.join(", "), detail.join("; "));
            }
        }
    }
    if n.is_power_of_two() && n >= 1024 {
        crate::info!("prep check: {n} cache hits compared with full preparations, {} differed, {} changed during the check", PREP_DIFFERED.load(Relaxed), PREP_RACED.load(Relaxed));
    }
}

/// A cached preparation for a call with user data `user`, if the guest words
/// it read and its depth-view decisions (part of its translation) still hold.
fn reuse(e: &mut PrepEntry, user: &[u32]) -> Option<Prepared> {
    let map_gen = crate::mem::MAP_GEN.load(std::sync::atomic::Ordering::Acquire);
    let images_gen = super::images::GPU_IMAGES_GEN.load(std::sync::atomic::Ordering::Acquire);
    if !words_unchanged(&e.prepared) || !depth_views_hold(&e.prepared) {
        return None;
    }
    // Readable at this mapping, the same depth views at these images: later
    // reuses skip those checks.
    e.prepared.map_gen = map_gen;
    e.prepared.images_gen = images_gen;
    let mut p = e.prepared.clone();
    for (i, w) in p.sgpr_data.iter_mut().enumerate().take(translate::SGPR_DATA_USER as usize) {
        *w = user.get(i).copied().unwrap_or(0);
    }
    p.user = UserData::new(user);
    Some(p)
}

fn stage_slot(s: Stage) -> usize {
    match s {
        Stage::Vertex => 0,
        Stage::Fragment => 1,
        _ => 2,
    }
}

fn prepare(addr: u64, insts: Arc<Local<Vec<rdna::Inst>>>, code_hash: u64, user: &[u32], spec: &mut Spec) -> Result<Prepared, String> {
    // Diagnostics (DS_PROFILE_GPU with DS_PROFILE_REPEATS=1: hashing every
    // word read and a global set per preparation inflate the prepare times
    // the profile reports): the inputs that would key a cached preparation.
    static REPEAT_STATS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let repeat_stats = *REPEAT_STATS.get_or_init(|| std::env::var("DS_PROFILE_REPEATS").ok().as_deref() == Some("1"));
    let repeat_key = (repeat_stats && super::exec::profiling_on()).then(|| {
        let mut h = super::fx::FxHasher::default();
        h.write_u64(addr);
        h.write_u64(code_hash);
        for &w in &user[..(spec.user_sgprs as usize).min(user.len())] {
            h.write_u32(w);
        }
        spec.hash(&mut h);
        h.finish()
    });
    // The plan depends only on the program, the user-data layout and where
    // data loads are read.
    let sel = super::tune::gpu_loads_sel();
    let selected = sel == 0 || code_hash & ((1u64 << (sel >> 32).min(63)) - 1) == sel & 0xFFFF_FFFF;
    let gpu_loads = FORCE_GPU_LOADS.with(|c| c.get())
        || match super::tune::gpu_loads_mode() {
            0 => false,
            2 => !spec.per_lane && selected,
            3 => gpu_loads_program(addr),
            _ => selected,
        };
    if gpu_loads && sel >> 32 >= 8 && GPU_LOADS_SELECTED.lock().get_or_insert_with(std::collections::HashSet::new).insert((sel, addr)) {
        let saved = super::capture::save_program(addr).map(|f| format!("; program saved to {}", f.display())).unwrap_or_default();
        crate::info!("gpu_loads selection {sel:#x}: {:?} shader {addr:#x} (code hash {code_hash:#018x}){saved}", spec.stage);
    }
    let plan_key = code_hash ^ (spec.user_sgprs as u64) << 48 ^ (spec.user_base as u64) << 56 ^ (gpu_loads as u64) << 40;
    let cached = LOCAL_PLANS.with(|c| c.borrow().get(&plan_key).cloned()).or_else(|| {
        let p = std::rc::Rc::new(PLANS.lock().as_ref().and_then(|m| m.get(&plan_key).cloned())?);
        LOCAL_PLANS.with(|c| c.borrow_mut().insert(plan_key, p.clone()));
        Some(p)
    });
    let plan = match cached {
        Some(p) => p,
        None => {
            let t0 = std::time::Instant::now();
            let p = Arc::new(rdna::analysis::analyze_opts(&insts, spec.user_sgprs, spec.user_base, gpu_loads));
            let ms = t0.elapsed().as_millis();
            if ms >= 100 {
                crate::info!("{:?} shader {addr:#x}: analysis took {ms} ms ({} instructions)", spec.stage, insts.len());
            }
            PLANS.lock().get_or_insert_with(HashMap::new).insert(plan_key, p.clone());
            let p = std::rc::Rc::new(p);
            LOCAL_PLANS.with(|c| c.borrow_mut().insert(plan_key, p.clone()));
            p
        }
    };
    if !plan.errors.is_empty() {
        return Err(format!("{:?} shader {addr:#x}: {}", spec.stage, plan.errors.join("; ")));
    }
    // The cached preparation of this stage, if its inputs and the words it
    // read are the same.
    let slot = stage_slot(spec.stage);
    // Compute preparations do not repeat (the per-item culls differ in the
    // user data their descriptors depend on).
    let cacheable = slot != 2 && super::tune::prep_cache();
    let prep_key = if !cacheable {
        0
    } else {
        let deps = LOCAL_DEPS.with(|c| *c.borrow_mut().entry(plan_key).or_insert_with(|| user_deps(&plan)));
        let mut h = super::fx::FxHasher::default();
        h.write_u64(addr);
        h.write_u64(plan_key);
        for (i, &w) in user.iter().enumerate().take(32) {
            if deps & (1 << i) != 0 {
                h.write_u32(w);
            }
        }
        spec.hash(&mut h);
        // The depth-view decisions, the one image-cache input, are checked
        // on reuse (`depth_views_hold`): entries outlive image lookups
        // changing as textures stream.
        h.write_u8(super::tune::cmp_emulate() as u8);
        h.finish()
    };
    let same = |e: &PrepEntry| e.addr == addr && e.plan_key == plan_key;
    let hit = if !cacheable {
        None
    } else if super::tune::prep_cache_mode() >= 2 {
        PREP_CACHE.with(|c| c.borrow_mut()[slot].get_mut(&prep_key).filter(|e| same(e)).and_then(|e| reuse(e, user)))
    } else {
        PREP_LATEST.with(|l| l.borrow_mut()[slot].as_mut().filter(|(k, e)| *k == prep_key && same(e)).and_then(|(_, e)| reuse(e, user)))
    };
    let mut checked_hit = None;
    let hit = hit.and_then(|p| {
        PREP_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if prep_check() {
            checked_hit = Some(p);
            return None;
        }
        Some(p)
    });
    if let Some(p) = hit {
        // Texture-streaming statistics as a full preparation notes them.
        for r in p.resources.iter() {
            match r {
                Resolved::Image(t) => super::lodstats::note(&t.raw),
                Resolved::ImageTable(e) => {
                    for t in e.iter().flatten() {
                        super::lodstats::note(&t.raw);
                    }
                }
                _ => {}
            }
        }
        return Ok(p);
    }
    if cacheable && checked_hit.is_none() {
        PREP_MISSES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    let map_gen = crate::mem::MAP_GEN.load(std::sync::atomic::Ordering::Acquire);
    let images_gen = super::images::GPU_IMAGES_GEN.load(std::sync::atomic::Ordering::Acquire);
    let sizes = READ_HINTS.with(|h| h.borrow().get(&plan_key).copied().unwrap_or([0; 3]));
    let mut rd = Reader::with_sizes(addr, sizes);
    let mut resources = pool::resolved(plan.resources.len());
    let mut res_addrs = pool::addrs(plan.resources.len());
    spec.resources.clear();
    spec.resources.reserve(plan.resources.len());
    let fmt_mask = if plan.resources.iter().any(|r| r.kind == ResKind::Buffer) { LOCAL_FORMATS.with(|c| *c.borrow_mut().entry(plan_key).or_insert_with(|| format_resources(&insts, &plan))) } else { 0 };
    for (i, res) in plan.resources.iter().enumerate() {
        let mut w = [0u32; 8];
        let src = match res.dwords.first() {
            Some(Prov::Mem(ptr, off)) => rd.chain(plan.chain_ids.get(plan.loads.len() + i).copied().unwrap_or(u32::MAX), ptr, user).map(|b| b + *off as u64),
            _ => None,
        };
        res_addrs.push(src);
        // A descriptor in consecutive words: one block read.
        let n = res.dwords.len().min(8);
        let block = plan.blocks.get(i).copied().unwrap_or(false) && src.is_some_and(|b| rd.read_into(b, &mut w[..n]));
        if !block {
            for (k, p) in res.dwords.iter().enumerate() {
                w[k] = rd.eval(p, user).ok_or_else(|| format!("{:?} shader {addr:#x}: resource {i} dword {k} unresolvable ({p:?})", spec.stage))?;
            }
        }
        match res.kind {
            ResKind::Buffer => {
                let v = Vsharp::decode(&w);
                // Raw accesses do not read the format and swizzle (see
                // `format_resources`).
                let (format, dst_sel) = if fmt_mask & (1 << i.min(63)) != 0 { (v.format, v.dst_sel) } else { (0, 0) };
                spec.resources.push(ResSpec::Buffer { format, dst_sel });
                resources.push(Resolved::Buffer(v));
            }
            ResKind::Global => {
                spec.resources.push(ResSpec::Global);
                resources.push(Resolved::Global(w[0] as u64 | ((w[1] as u64) << 32), res.extent.map(|e| e as u64)));
            }
            ResKind::Image => {
                let t = Tsharp::decode(&w);
                let dim = ImageDim::from_type(t.ty, t.depth).ok_or_else(|| format!("{:?} shader {addr:#x}: image resource {i} has type {}", spec.stage, t.ty))?;
                super::lodstats::note(&t.raw);
                // `depth` is decided below for all images at once.
                spec.resources.push(ResSpec::Image { dim, class: num_class(t.format), depth: true });
                resources.push(Resolved::Image(t));
            }
            ResKind::Sampler => {
                spec.resources.push(ResSpec::Sampler);
                resources.push(Resolved::Sampler([w[0], w[1], w[2], w[3]]));
            }
            ResKind::ImageTable => {
                // Entries of 32 bytes (T#s) from the table's base: a V#'s
                // range, or from a pointer (a T#-sized s_load at a run-time
                // offset) as many as the analysis saw loaded at constant
                // offsets, else up to the first entry that is no usable T#
                // (at most 32: the title's waterfall loops select entries by
                // a 32-bit mask).
                let len = rdna::translate::IMAGE_TABLE_LEN as u64;
                let (tbase, n, stop) = if res.dwords.len() == 2 {
                    let b = w[0] as u64 | ((w[1] as u64) << 32);
                    match res.extent {
                        Some(e) => (b, (e as u64 / 32).min(len), false),
                        None => (b, 32.min(len), true),
                    }
                } else {
                    let v = Vsharp::decode(&w);
                    (v.base, (v.extent() / 32).min(len), false)
                };
                let mut entries = Vec::with_capacity(n as usize);
                let mut first: Option<(ImageDim, u16)> = None;
                for k in 0..n {
                    let a = tbase + 32 * k;
                    let mut d = [0u32; 8];
                    let ok = rd.read_into(a, &mut d);
                    let t = Tsharp::decode(&d);
                    if ok {
                        super::lodstats::note(&t.raw);
                    }
                    let dim = ImageDim::from_type(t.ty, t.depth);
                    let usable = ok && t.base != 0 && super::images::vk_format(t.format).is_some() && dim.is_some() && first.is_none_or(|(fd, _)| Some(fd) == dim);
                    if !usable && stop {
                        break;
                    }
                    if usable && first.is_none() {
                        first = Some((dim.unwrap(), t.format));
                    }
                    entries.push(usable.then_some(t));
                }
                let (dim, format) = first.ok_or_else(|| format!("{:?} shader {addr:#x}: image table {i} ({tbase:#x}, {n} entries) has no usable T#", spec.stage))?;
                spec.resources.push(ResSpec::Image { dim, class: num_class(format), depth: false });
                resources.push(Resolved::ImageTable(entries));
            }
        }
    }
    // Depth views keep the hardware comparison, color views emulate it: the
    // image cache decides, for the images comparison instructions sample
    // (the others' translation does not depend on it).
    let mut cmp_images = 0u64;
    let mut cmp_depth = 0u64;
    if super::tune::cmp_emulate() && resources.iter().any(|r| matches!(r, Resolved::Image(_))) {
        cmp_images = compare_images(plan_key, &insts, &plan);
        for (r, s) in resources.iter().zip(spec.resources.iter_mut()) {
            if let (Resolved::Image(_), ResSpec::Image { depth, .. }) = (r, s) {
                *depth = false;
            }
        }
        if cmp_images != 0 {
            super::images::with(|im| {
                for (i, (r, s)) in resources.iter().zip(spec.resources.iter_mut()).enumerate() {
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
    // Translations do not depend on the program's address (see
    // translate::code_base_offset): copies of a program share one, and its
    // pipeline (the Vulkan pipeline cache finds it again in the next run).
    static BY_ADDRESS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let by_address = *BY_ADDRESS.get_or_init(|| std::env::var_os("DS_WAVEMASK").is_some() || std::env::var_os("DS_LOCKSTEP_OFF").is_some());
    let base = spec.code_base;
    if !by_address {
        spec.code_base = 0;
    }
    let last_hit = LAST_TRANSLATIONS.with(|c| c.borrow().get(&plan_key).filter(|t| t.spec == *spec).cloned());
    let from_last = last_hit.is_some();
    let (fast, fast_hit) = match last_hit {
        Some(t) => (0, Some(t)),
        None => {
            let fast = {
                let mut h = super::fx::FxHasher::default();
                spec.hash(&mut h);
                h.finish()
            };
            (fast, FAST_TRANSLATIONS.with(|c| c.borrow().get(&(plan_key, fast)).filter(|t| t.spec == *spec).cloned()))
        }
    };
    let key_spec = if fast_hit.is_none() { Some(spec.clone()) } else { None };
    let spec_hash = match &key_spec {
        Some(ks) => {
            let mut h = std::collections::hash_map::DefaultHasher::new();
            ks.hash(&mut h);
            h.finish()
        }
        None => 0,
    };
    spec.code_base = base;
    let tr = if let Some(t) = fast_hit {
        t
    } else {
        let cached = LOCAL_TRANSLATIONS.with(|c| c.borrow().get(&(plan_key, spec_hash)).cloned()).or_else(|| {
            let t = TRANSLATIONS.lock().as_ref().and_then(|m| m.get(&(plan_key, spec_hash)).cloned())?;
            LOCAL_TRANSLATIONS.with(|c| c.borrow_mut().insert((plan_key, spec_hash), t.clone()));
            Some(t)
        });
        let t = match cached {
            Some(t) => t,
            None => {
                if spec.stage == Stage::Compute && super::tune::precompile() > 0 {
                    if let Some(ks) = &key_spec {
                        super::precompile::first_translation(addr, &insts, &plan, code_hash, ks);
                    }
                }
                let t = translate_kept(addr, &insts, &plan, plan_key, spec_hash, spec, key_spec.clone().unwrap_or_else(|| spec.clone()))?;
                LOCAL_TRANSLATIONS.with(|c| c.borrow_mut().insert((plan_key, spec_hash), t.clone()));
                t
            }
        };
        // This thread's handle on it.
        let t = Arc::new(Local(t));
        if key_spec.as_ref().is_some_and(|ks| *ks == t.spec) {
            FAST_TRANSLATIONS.with(|c| c.borrow_mut().insert((plan_key, fast), t.clone()));
        }
        t
    };
    if !from_last {
        LAST_TRANSLATIONS.with(|c| c.borrow_mut().insert(plan_key, tr.clone()));
    }
    // Scalar-load data, resolved now.
    let mut sgpr_data = pool::words((translate::sgpr_data_dwords(&tr.plan) as usize).max(translate::SGPR_DATA_USER as usize));
    sgpr_data.extend_from_slice(user);
    sgpr_data.resize(translate::SGPR_DATA_USER as usize, 0);
    let mut data_loads = pool::loads(tr.plan.load_data.iter().filter(|&&d| d).count());
    for (j, l) in tr.plan.loads.iter().enumerate() {
        let base = rd.chain(tr.plan.chain_ids.get(j).copied().unwrap_or(u32::MAX), &l.ptr, user);
        let start = sgpr_data.len();
        sgpr_data.resize(start + l.dwords as usize, 0);
        let Some(b) = base else { continue };
        let word = rd.words.len();
        if !rd.read_into(b + l.offset as u64, &mut sgpr_data[start..]) {
            for k in 0..l.dwords as usize {
                sgpr_data[start + k] = rd.u32(b + l.offset as u64 + 4 * k as u64).unwrap_or(0);
            }
        }
        if tr.plan.load_data.get(j).copied().unwrap_or(false) && rd.words.len() == word + l.dwords as usize {
            data_loads.push(DataLoad { sgpr: start as u32, addr: b + l.offset as u64, dwords: l.dwords, word: word as u32 });
        }
    }
    sgpr_data.resize(translate::sgpr_data_dwords(&tr.plan) as usize, 0);
    let cb = translate::code_base_offset(&tr.plan) as usize;
    sgpr_data[cb] = addr as u32;
    sgpr_data[cb + 1] = (addr >> 32) as u32;
    // Resource info blocks are filled when binding; until then no index
    // range check applies.
    let ri = translate::res_info_offset(&tr.plan) as usize;
    for k in 0..tr.plan.resources.len() {
        sgpr_data[ri + translate::RES_INFO_DWORDS as usize * k + 3] = u32::MAX;
    }
    // LOD information of the sampled images (hardware LOD statistics).
    for (k, r) in resources.iter().enumerate() {
        match r {
            Resolved::Image(t) => sgpr_data[ri + translate::RES_INFO_DWORDS as usize * k] = super::lodstats::info(t),
            Resolved::ImageTable(entries) => {
                let at = (translate::lod_info_offset(&tr.plan) + translate::IMAGE_TABLE_LEN * translate::table_slot(&tr.plan, k)) as usize;
                for (j, t) in entries.iter().enumerate().take(translate::IMAGE_TABLE_LEN as usize) {
                    sgpr_data[at + j] = t.as_ref().map(super::lodstats::info).unwrap_or(0);
                }
            }
            _ => {}
        }
    }
    // The reader's references into the translation's plan end here.
    let used = [rd.reads.len() as u32, rd.words.len() as u32, rd.ptrs.len() as u32];
    if used != sizes {
        READ_HINTS.with(|h| h.borrow_mut().insert(plan_key, used));
    }
    let Reader { reads, words, unread, chain_vals, .. } = rd;
    pool::put_addrs(chain_vals);
    let prepared = Prepared {
        stage: spec.stage,
        addr,
        found: Arc::new(Found { insts, tr, resources, res_addrs, reads, words, unread, data_loads }),
        sgpr_data,
        user: UserData::new(user),
        map_gen,
        images_gen,
        cmp_images,
        cmp_depth,
        code_hash,
        ahead: 1,
        checked_ep: 0,
    };
    if let Some(h) = &checked_hit {
        prep_compare(h, &prepared);
    }
    if cacheable {
        // The words it read are the cache's check (no second read).
        let e = PrepEntry { addr, plan_key, prepared: prepared.clone() };
        if super::tune::prep_cache_mode() >= 2 {
            PREP_CACHE.with(|c| {
                let m = &mut c.borrow_mut()[slot];
                if m.len() >= PREP_CACHE_MAX {
                    m.clear();
                }
                m.insert(prep_key, e);
            });
        } else {
            PREP_LATEST.with(|l| l.borrow_mut()[slot] = Some((prep_key, e)));
        }
    }
    if let Some(k) = repeat_key {
        let tr = &prepared.tr;
        let bytes = note_repeat(spec.stage, k, &prepared.reads);
        let deps = user_deps(&tr.plan);
        let mut h = super::fx::FxHasher::default();
        h.write_u64(addr);
        h.write_u64(code_hash);
        for (i, &w) in user.iter().enumerate().take(32) {
            if deps & (1 << i) != 0 {
                h.write_u32(w);
            }
        }
        spec.hash(&mut h);
        note_repeat_deps(spec.stage, h.finish(), bytes);
    }
    Ok(prepared)
}
