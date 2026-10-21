//! gfx10 PM4 command processor.
//!
//! Executes command buffers submitted through libSceAgcDriver on a GPU queue
//! thread: register state (context, SH, uconfig), control flow (indirect
//! buffers, COND_EXEC, REWIND, the Agc branch), synchronisation (WAIT_REG_MEM,
//! RELEASE_MEM labels and end-of-pipe interrupts), memory packets (WRITE_DATA,
//! COPY_DATA, DMA_DATA), flips and the private Agc packets
//! (`hle::libs::agc_cmd`).
//!
//! With `DS_GPU=1`, dispatches run on Vulkan (`exec`); the recorded GPU work
//! is completed before any packet that touches memory or synchronizes, so
//! labels and CPU/GPU synchronisation stay ordered as on hardware. Draws are
//! decoded and counted but not rendered yet.

use crate::hle::libs::agc_cmd::{DS_MAGIC, op, pop};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

pub const CTX_BASE: u32 = 0x28000;
pub const SH_BASE: u32 = 0xB000;
pub const UC_BASE: u32 = 0x30000;
const CTX_DWORDS: usize = 0x2000;
const SH_DWORDS: usize = 0x400;
const UC_DWORDS: usize = 0x4000;

pub static DRAWS: AtomicU64 = AtomicU64::new(0);
pub static DISPATCHES: AtomicU64 = AtomicU64::new(0);
pub static PACKETS: AtomicU64 = AtomicU64::new(0);
pub static EOP_INTERRUPTS: AtomicU64 = AtomicU64::new(0);
/// GPU completions waited for, by the PM4 opcode that required them.
pub static SYNC_BY_OPCODE: [AtomicU64; 256] = [const { AtomicU64::new(0) }; 256];
/// CP effects deferred until earlier GPU work completed, by opcode and why:
/// [end of pipe, after a sync event, behind an earlier deferred effect].
pub static DEFERRED_BY_OPCODE: [[AtomicU64; 3]; 256] = [const { [const { AtomicU64::new(0) }; 3] }; 256];
/// RELEASE_MEM packets by DATA_SEL and INT_SEL (8 x 8).
pub static RELEASE_KINDS: [AtomicU64; 64] = [const { AtomicU64::new(0) }; 64];
/// Waits passed on a value the waiting queue's own recording batch writes
/// on the GPU (`tune gpu_cp_writes`).
pub static OPEN_PROMISED: AtomicU64 = AtomicU64::new(0);

/// Most frequent synchronizing opcodes (diagnostics).
pub fn sync_summary() -> String {
    let mut v: Vec<(u64, usize)> = SYNC_BY_OPCODE.iter().enumerate().map(|(i, c)| (c.load(Ordering::Relaxed), i)).filter(|x| x.0 > 0).collect();
    v.sort_by(|a, b| b.cmp(a));
    let waits = v.iter().take(6).map(|(n, op)| format!("{op:#x}: {n}")).collect::<Vec<_>>().join(", ");
    let mut d: Vec<(u64, usize, [u64; 3])> = DEFERRED_BY_OPCODE
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let x = [c[0].load(Ordering::Relaxed), c[1].load(Ordering::Relaxed), c[2].load(Ordering::Relaxed)];
            (x.iter().sum(), i, x)
        })
        .filter(|x| x.0 > 0)
        .collect();
    d.sort_by(|a, b| b.cmp(a));
    let deferred = d.iter().take(6).map(|(_, op, x)| format!("{op:#x}: {} end of pipe / {} after sync / {} behind deferred", x[0], x[1], x[2])).collect::<Vec<_>>().join(", ");
    let kinds: Vec<String> = RELEASE_KINDS.iter().enumerate().filter(|(_, c)| c.load(Ordering::Relaxed) > 0).map(|(i, c)| format!("data {} int {}: {}", i / 8, i % 8, c.load(Ordering::Relaxed))).collect();
    format!(
        "{waits}; deferred CP effects {deferred}; RELEASE_MEM by kind {}; waits passed on the own batch's GPU writes {}; label writes recorded while another queue waited (submitted at once) {}; indirect draws with arguments in homes read on the CPU {:?} (base vertex, no start-instance SGPR, 8-bit indices, indices outside homes, CPU-side start instance), {} of them after trying the GPU",
        kinds.join(", "),
        OPEN_PROMISED.load(Ordering::Relaxed),
        super::exec::LABEL_EARLY.load(Ordering::Relaxed),
        CPU_ARG_REFUSED.iter().map(|c| c.load(Ordering::Relaxed)).collect::<Vec<u64>>(),
        CPU_ARG_FALLBACKS.load(Ordering::Relaxed)
    )
}

/// Register file of one queue.
#[derive(Clone)]
pub struct Regs {
    pub ctx: Vec<u32>,
    pub sh: Vec<u32>,
    pub uc: Vec<u32>,
}

impl Regs {
    fn new() -> Regs {
        Regs { ctx: vec![0; CTX_DWORDS], sh: vec![0; SH_DWORDS], uc: vec![0; UC_DWORDS] }
    }
}

/// The highest register index written so far plus one, per space (context,
/// SH, UCONFIG), over all command processors and chunk states: every register
/// at or above it is zero everywhere, so state snapshots (`ChunkState`) hold
/// only what lies below (the UCONFIG file alone is 64 KiB; a parallel run
/// copied ~20 whole register files).
static HIGH: [std::sync::atomic::AtomicUsize; 3] = [const { std::sync::atomic::AtomicUsize::new(0) }; 3];

fn note_high(space: Space, index: usize) {
    let h = &HIGH[space as usize];
    if index >= h.load(Ordering::Relaxed) {
        h.fetch_max(index + 1, Ordering::AcqRel);
    }
}

fn high(space: Space, cap: usize) -> usize {
    HIGH[space as usize].load(Ordering::Acquire).min(cap)
}

/// Copies a snapshot's registers into `dst` and zeroes the rest up to `hi`
/// (registers the snapshot does not hold were zero when it was taken).
fn copy_prefix(dst: &mut [u32], src: &[u32], hi: usize) {
    let n = src.len().min(dst.len());
    dst[..n].copy_from_slice(&src[..n]);
    let end = hi.min(dst.len());
    if end > n {
        dst[n..end].fill(0);
    }
}

/// `tune plan_check`: logs the first registers where a chunk start state made
/// with `put_regs` differs from the one made a register at a time.
fn plan_check_states(a: &ChunkState, b: &ChunkState, at: u64) {
    static CHECKED: AtomicU64 = AtomicU64::new(0);
    static BAD: AtomicU64 = AtomicU64::new(0);
    let n = CHECKED.fetch_add(1, Ordering::Relaxed) + 1;
    let mut diff = Vec::new();
    for (name, x, y) in [("ctx", &a.regs.ctx, &b.regs.ctx), ("sh", &a.regs.sh, &b.regs.sh), ("uc", &a.regs.uc, &b.regs.uc)] {
        for i in 0..x.len().max(y.len()) {
            let (u, v) = (x.get(i).copied().unwrap_or(0), y.get(i).copied().unwrap_or(0));
            if u != v && diff.len() < 6 {
                diff.push(format!("{name}[{i:#x}] {u:#x} against {v:#x}"));
            }
        }
    }
    let other = (a.index_base, a.index_count, a.num_instances, a.draw_indirect_base, a.dispatch_indirect_base) != (b.index_base, b.index_count, b.num_instances, b.draw_indirect_base, b.dispatch_indirect_base);
    if !diff.is_empty() || other {
        if BAD.fetch_add(1, Ordering::Relaxed) < 20 {
            crate::warn!("gpu: plan_check: chunk start at {at:#x}: {} (other fields differ: {other})", diff.join(", "));
        }
    } else if n % 10000 == 0 {
        crate::info!("gpu: plan_check: {n} chunk start states checked, {} differed", BAD.load(Ordering::Relaxed));
    }
}

/// Writes `count` registers of a snapshot from `first` (growing it) with the
/// dwords at `a` (`put_reg` for each, in one copy).
fn put_regs(v: &mut Vec<u32>, space: Space, cap: usize, first: usize, f: &Fetched, a: u64, count: usize) {
    if count == 0 || first >= cap {
        return;
    }
    let end = (first + count).min(cap);
    if end > v.len() {
        v.resize(end, 0);
    }
    let dst = &mut v[first..end];
    match f.slice(a, dst.len()) {
        Some(src) => dst.copy_from_slice(src),
        None => {
            for (j, d) in dst.iter_mut().enumerate() {
                *d = f.rd(a + 4 * j as u64);
            }
        }
    }
    note_high(space, end - 1);
}

/// Writes register `i` of a snapshot (growing it), notes the index.
fn put_reg(v: &mut Vec<u32>, space: Space, cap: usize, i: usize, val: u32) {
    if i >= cap {
        return;
    }
    if i >= v.len() {
        v.resize(i + 1, 0);
    }
    v[i] = val;
    note_high(space, i);
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Space {
    Ctx,
    Sh,
    Uc,
}

/// Draws, those with the last draw's context registers on the same command
/// processor, and of those the ones with its programs and primitive type too
/// (`Cp::draw`; `draw_state_line`).
pub static DRAW_STATE: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// Runs of draws with the last draw's context registers, programs and
/// primitive type, by length (1, 2, 3-4, 5-8, 9-16, 17-32, more): [runs,
/// draws].
static DRAW_RUNS: [[AtomicU64; 2]; 7] = [const { [const { AtomicU64::new(0) }; 2] }; 7];

fn note_draw_run(n: u32) {
    if n == 0 {
        return;
    }
    let k = match n {
        1 => 0,
        2 => 1,
        3..=4 => 2,
        5..=8 => 3,
        9..=16 => 4,
        17..=32 => 5,
        _ => 6,
    };
    DRAW_RUNS[k][0].fetch_add(1, Ordering::Relaxed);
    DRAW_RUNS[k][1].fetch_add(n as u64, Ordering::Relaxed);
}

/// For the status report (`DRAW_STATE`, `DRAW_RUNS`).
pub fn draw_state_line() -> String {
    let d: Vec<u64> = DRAW_STATE.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    if d[0] == 0 {
        return String::new();
    }
    let runs: Vec<String> = ["1", "2", "3-4", "5-8", "9-16", "17-32", ">32"]
        .iter()
        .zip(DRAW_RUNS.iter())
        .map(|(l, r)| format!("{l}: {} runs, {} draws", r[0].load(Ordering::Relaxed), r[1].load(Ordering::Relaxed)))
        .collect();
    format!(
        "    draws with the last draw's context registers: {} of {} ({:.1}%), {} with its programs and primitive type too ({:.1}%); runs of such draws by length: {}\n",
        d[1],
        d[0],
        100.0 * d[1] as f64 / d[0] as f64,
        d[2],
        100.0 * d[2] as f64 / d[0] as f64,
        runs.join(", ")
    )
}

/// Chains of repeatable draws ended, by the opcode of the packet that ended
/// them (`keeps_repeat`; the status line's top ones).
pub static REPEAT_ENDS: [AtomicU64; 256] = [const { AtomicU64::new(0) }; 256];

/// The opcodes that ended most chains of repeatable draws (`REPEAT_ENDS`).
pub fn repeat_ends_line() -> String {
    let mut v: Vec<(u64, usize)> = REPEAT_ENDS.iter().enumerate().map(|(k, c)| (c.load(Ordering::Relaxed), k)).filter(|x| x.0 > 0).collect();
    v.sort_unstable_by(|a, b| b.cmp(a));
    v.iter().take(6).map(|(c, k)| format!("{k:#x} {c}")).collect::<Vec<_>>().join(", ")
}

pub struct Cp {
    pub name: String,
    /// Outcome of the last draw (frame captures; no allocation per draw).
    pub last_draw: std::borrow::Cow<'static, str>,
    /// The last packet was an indexed indirect draw with GPU-read arguments
    /// that the executor can repeat (`tune draw_repeat`): no other packet
    /// (but NOPs) since, so the registers are the same.
    repeat_ok: bool,
    /// A context register changed value since the last draw (measurement:
    /// `DRAW_STATE`), and the last draw's vertex and pixel programs and
    /// primitive type.
    ctx_dirty: bool,
    last_draw_progs: (u64, u64, u32),
    /// Draws in the current run of draws with the last draw's context
    /// registers, programs and primitive type (`note_draw_run`).
    same_run: u32,
    pub compute: bool,
    pub regs: Regs,
    index_base: u64,
    index_count: u32,
    num_instances: u32,
    draw_indirect_base: u64,
    dispatch_indirect_base: u64,
    markers: Vec<String>,
    ctx_stack: Vec<Vec<u32>>,
    /// Address of the packet being executed / address being polled.
    cur_packet: u64,
    /// Opcode of the packet being executed (private packets: 0x100 + op).
    cur_opcode: u32,
    waiting_on: u64,
    /// Set while the CP polls memory (WAIT_REG_MEM): submitters must not
    /// wait for this queue then (the label may come from them).
    pub wait_flag: Option<&'static std::sync::atomic::AtomicBool>,
    unknown_seen: HashSet<u32>,
    notes_seen: HashSet<&'static str>,
    depth: u32,
    /// Vulkan executor (created on the first dispatch when enabled).
    /// Boxed: commands take it out of the processor and put it back (one
    /// pointer moved per draw and dispatch, not the whole executor).
    exec: Option<Box<super::exec::Exec>>,
    exec_failed: bool,
    /// A partial flush / acquire was requested: the next CP memory write
    /// must wait for the GPU work before it.
    fence_pending: bool,
    /// SET_PREDICATION state for packets with the predicate bit.
    pred: Option<Pred>,
    /// When the title submitted the command buffer being executed.
    pub submitted_at: Option<Instant>,
    /// Another command buffer follows this one at once (the title's
    /// submission holds more, or its next submission is queued): its end
    /// need not submit the batch (`tune cmdbuf_merge`).
    pub more_follows: Option<Box<dyn Fn() -> bool + Send>>,
    pub next_in_submission: bool,
    /// Stage preparations started ahead for the next commands of the
    /// current synchronization-free run (`scan_ahead`), in packet order.
    ahead: std::collections::VecDeque<AheadCmd>,
    /// The executing command's preparations made ahead (taken by the
    /// executor through `take_ahead`; [vertex, pixel] or [compute, -]).
    cur_ahead: std::cell::RefCell<[Option<Arc<super::ahead::Slot>>; 2]>,
    /// The scan's copies of the SH and context register files (kept to
    /// avoid allocations).
    scan_regs: std::cell::RefCell<(Vec<u32>, Vec<u32>)>,
    /// The fetched buffer's base, the packet after the dispatch about to
    /// execute and the buffer's end (`tune compile_ahead`): scanned when
    /// the dispatch's pipeline is missing (`prefetch_compiles`).
    compile_ctx: std::cell::Cell<Option<(u64, u64, u64)>>,
    /// The buffer, the packet after it and the buffer's end of the label
    /// wait about to execute (`tune wait_ahead`): scanned if it blocks.
    wait_ctx: Option<(Fetched, u64, u64)>,
    /// A parallel run in progress (`gpu::par`).
    par: Option<ParRun>,
    /// The last WAIT_REG_MEM passed on a GPU queue's write (a promise, an
    /// open promise of this queue, a latched CP write): no CPU write
    /// released it, so it ends no copy-cache epoch (`tune epoch_gpu_waits`).
    last_wait_gpu: bool,
    /// How the last WAIT_REG_MEM passed (its `label_event` kind; diagnostics).
    last_wait_kind: u8,
    /// The end of the last run the planner found too short (no new plan
    /// until execution passes it: each command would scan the rest again).
    par_skip: (u64, u64),
    /// Its commands (`tune cp_classes`: serial commands by run length).
    par_skip_n: usize,
    /// This command processor's chunk workers.
    par_pool: super::par::Pool,
    /// A worker's per-command recording cost in its chunk (`tune par_cost`).
    pub chunk_costs: Option<Vec<u32>>,
    /// Per-command recording costs of the parallel runs seen, by run
    /// signature (`tune par_cost`).
    run_costs: HashMap<u64, RunCost>,
}

/// The recording cost of a parallel run's commands as its workers measured
/// it (TSC cycles, smoothed over the times the run was seen; NaN: not
/// measured yet), and the frame it was last planned in.
struct RunCost {
    costs: Vec<f32>,
    seen: u64,
}

/// Splits commands [main_n, n) into `m` chunks of about equal summed cost
/// (`costs` per command), each of at least `min` commands: the first command
/// of each chunk. None without costs for all of them.
fn cost_firsts(costs: &[f32], main_n: usize, n: usize, m: usize, min: usize) -> Option<Vec<usize>> {
    let region = costs.get(main_n..n)?;
    if m < 2 || region.iter().any(|c| !c.is_finite()) || n - main_n < m * min {
        return None;
    }
    let total: f64 = region.iter().map(|&c| c as f64).sum();
    if total <= 0.0 {
        return None;
    }
    let mut firsts = Vec::with_capacity(m);
    firsts.push(main_n);
    let mut acc = 0.0;
    for (j, &c) in region.iter().enumerate() {
        let k = firsts.len();
        if k == m {
            break;
        }
        acc += c as f64;
        if acc >= total * k as f64 / m as f64 {
            let lo = firsts[k - 1] + min;
            let hi = n - (m - k) * min;
            firsts.push((main_n + j + 1).clamp(lo, hi));
        }
    }
    while firsts.len() < m {
        let k = firsts.len();
        firsts.push((firsts[k - 1] + min).min(n - (m - k) * min));
    }
    firsts.windows(2).all(|w| w[1] >= w[0] + min).then_some(firsts)
}

/// The command-processor state a chunk of a parallel run starts from and
/// ends with (`gpu::par`).
#[derive(Clone)]
pub struct ChunkState {
    regs: Regs,
    index_base: u64,
    index_count: u32,
    num_instances: u32,
    draw_indirect_base: u64,
    dispatch_indirect_base: u64,
}

/// A parallel run in progress: chunk i + 1 starts at `starts[i]` and is
/// chunk i of `run` (the workers'); it ends where the next starts (the last
/// at `end`). `next`: the chunk to collect next.
struct ParRun {
    starts: Vec<u64>,
    end: u64,
    next: usize,
    run: Arc<super::par::Run>,
    /// How long the main command processor waited for the workers so far.
    waited_ns: u64,
    /// The run holds flushes or cache actions (`tune par_events`): later CP
    /// writes follow the GPU work before them.
    sync: bool,
    /// Its signature and command count, and each worker chunk's first
    /// command (`tune par_cost`; signature 0: not timed).
    sig: u64,
    n: usize,
    firsts: Vec<usize>,
}

/// Commands per chunk at least (a third of `tune par_min`, the commands a
/// run needs to be split, when that is below 192), and at most per
/// parallel run.
const PAR_CHUNK_MIN: usize = 64;
const PAR_MAX: usize = 8192;

/// Command-buffer ends that left the batch open (`tune cmdbuf_merge`).
pub static CMDBUF_MERGED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn is_command(op: u32) -> bool {
    matches!(op, op::DRAW_INDEX_2 | op::DRAW_INDEX_AUTO | op::DRAW_INDEX_OFFSET_2 | op::DRAW_INDIRECT | op::DRAW_INDEX_INDIRECT | op::DISPATCH_DIRECT | op::DISPATCH_INDIRECT)
}

/// Packets that leave a run open (state only; see `Cp::scan_ahead`).
/// Flushes and cache actions that parallel runs of this kind of queue keep
/// inside them (`tune par_events`).
fn par_event(op: u32, compute: bool) -> bool {
    matches!(op, op::EVENT_WRITE | op::ACQUIRE_MEM) && super::tune::par_events() & if compute { 1 } else { 2 } != 0
}

fn is_state(op: u32) -> bool {
    matches!(
        op,
        op::SET_CONTEXT_REG
            | op::SET_CONTEXT_REG_INDEX
            | op::SET_SH_REG
            | op::SET_SH_REG_INDEX
            | op::SET_UCONFIG_REG
            | op::SET_UCONFIG_REG_INDEX
            | op::LOAD_CONTEXT_REG_INDEX
            | op::LOAD_SH_REG_INDEX
            | op::LOAD_UCONFIG_REG
            | op::INDEX_TYPE
            | op::INDEX_BASE
            | op::INDEX_BUFFER_SIZE
            | op::NUM_INSTANCES
            | op::CONTEXT_CONTROL
            | op::SET_BASE
    )
}

/// A command the scan ahead saw: its packet address and the preparations
/// handed to the workers.
struct AheadCmd {
    at: u64,
    stages: [Option<Arc<super::ahead::Slot>>; 2],
    /// A draw's baked pipeline state registers (scans for compiles ahead).
    gfx: Option<Box<super::exec::GfxRegs>>,
}

/// Commands per scan ahead (a bound on work wasted when execution takes
/// another path).
const AHEAD_MAX: usize = 256;

/// Register view of a scan ahead: the command processor's register files
/// with the writes of the packets scanned so far.
struct ScanRegs<'a> {
    sh: &'a [u32],
    ctx: &'a [u32],
}

impl super::shaders::RegRead for ScanRegs<'_> {
    fn sh(&self, i: u32) -> u32 {
        self.sh.get(i as usize).copied().unwrap_or(0)
    }
    fn ctx(&self, a: u32) -> u32 {
        self.ctx.get(((a - 0x28000) / 4) as usize).copied().unwrap_or(0)
    }
}

/// SET_PREDICATION (as Agc encodes it: control = operation << 16 |
/// wait << 12 | condition << 8, address, dwords covered).
#[derive(Clone, Copy, Debug)]
struct Pred {
    op: u32,
    cond: u32,
    addr: u64,
    count: u32,
}

unsafe fn rd32(a: u64) -> u32 {
    unsafe { std::ptr::read_volatile(a as *const u32) }
}
unsafe fn rd64(a: u64) -> u64 {
    unsafe { std::ptr::read_volatile(a as *const u64) }
}
unsafe fn wr32(a: u64, v: u32) {
    super::precache::invalidate(a, 4);
    unsafe { std::ptr::write_volatile(a as *mut u32, v) }
}
unsafe fn wr64(a: u64, v: u64) {
    super::precache::invalidate(a, 8);
    unsafe { std::ptr::write_volatile(a as *mut u64, v) }
}

fn addr64(lo: u32, hi: u32) -> u64 {
    lo as u64 | ((hi as u64 & 0xFFFF_FFFF) << 32)
}

/// Guest addresses the command processor may touch.
pub(crate) fn valid(a: u64, len: u64) -> bool {
    a >= crate::mem::GUEST_BASE && a.checked_add(len).is_some_and(|e| e <= crate::mem::GUEST_END)
}

/// Whether guest memory [a, a+len) is mapped and readable.
pub fn guest_readable(a: u64, len: u64) -> bool {
    valid(a, len) && crate::mem::readable(a, len)
}

/// Compute shaders whose dispatches `DS_TRACE_CS` traces.
fn trace_cs_list() -> &'static [u64] {
    static L: std::sync::OnceLock<Vec<u64>> = std::sync::OnceLock::new();
    L.get_or_init(|| std::env::var("DS_TRACE_CS").map(|v| v.split(',').filter_map(|a| u64::from_str_radix(a.trim().trim_start_matches("0x"), 16).ok()).collect()).unwrap_or_default())
}

/// GPU timestamp (100 MHz, like the console's GPU clock counter).
pub fn gpu_clock() -> u64 {
    static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    let s = START.get_or_init(Instant::now);
    (s.elapsed().as_nanos() / 10) as u64
}

/// Command buffers up to this size are copied when execution starts.
const FETCH_DWORDS: u32 = 512;
/// Payload dwords of each packet of a larger buffer copied when execution
/// reaches the packet (`Cp::run`).
const PACKET_FETCH: usize = 32;

/// A command buffer as the CP fetched it (see `Cp::run`): small buffers are
/// copied, larger ones read from guest memory.
#[derive(Clone)]
struct Fetched {
    base: u64,
    words: Vec<u32>,
    /// Packets the CPU streams behind a REWIND (not scanned ahead).
    live: bool,
}

impl Fetched {
    fn new(base: u64, size: u32) -> Fetched {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let on = *ON.get_or_init(|| std::env::var("DS_CP_FETCH").ok().as_deref() != Some("0"));
        let n = size as usize;
        let words = if on && size > 0 && size <= FETCH_DWORDS && valid(base, 4 * n as u64) { unsafe { std::slice::from_raw_parts(base as *const u32, n) }.to_vec() } else { Vec::new() };
        Fetched { base, words, live: false }
    }

    /// The `len` dwords at `a` when they are all fetched, or all live (none
    /// fetched); None when they straddle the fetched range.
    fn slice(&self, a: u64, len: usize) -> Option<&[u32]> {
        if self.words.is_empty() {
            return valid(a, 4 * len as u64).then(|| unsafe { std::slice::from_raw_parts(a as *const u32, len) });
        }
        let i = (a.checked_sub(self.base)? / 4) as usize;
        self.words.get(i..i + len)
    }

    /// The dword at `a`: fetched, or live memory past the fetched range.
    fn rd(&self, a: u64) -> u32 {
        if a >= self.base {
            if let Some(&w) = self.words.get(((a - self.base) / 4) as usize) {
                return w;
            }
        }
        unsafe { rd32(a) }
    }
}

fn compare(func: u32, value: u64, reference: u64) -> bool {
    match func & 7 {
        0 => true,
        1 => value < reference,
        2 => value <= reference,
        3 => value == reference,
        4 => value != reference,
        5 => value >= reference,
        6 => value > reference,
        _ => false,
    }
}

/// Last GPU write to each label-sized address: (queue, value, packet address)
/// for wait diagnostics.
/// (and the exact address written: two 4-byte labels share a slot).
static LAST_WRITE: parking_lot::Mutex<Option<std::collections::HashMap<u64, (String, u64, u64, u64)>>> = parking_lot::Mutex::new(None);

/// Recent CP writes to each polled label (time, value), newest last: a
/// queue that reaches a WAIT_REG_MEM late still sees the writes made after
/// its command buffer was submitted (`missed_write`).
static LABEL_WRITES: parking_lot::Mutex<Option<HashMap<u64, std::collections::VecDeque<(Instant, u64)>>>> = parking_lot::Mutex::new(None);
const LABEL_WRITES_KEEP: usize = 64;
/// Waits passed on a write made after their submission (diagnostics).
pub static MISSED: AtomicU64 = AtomicU64::new(0);

/// Whether a CP write to `addr` made at or after `since` satisfies the
/// comparison.
fn missed_write(addr: u64, wide: bool, func: u32, mask: u64, reference: u64, since: Instant) -> bool {
    let g = LABEL_WRITES.lock();
    let Some(h) = g.as_ref().and_then(|m| m.get(&addr)) else { return false };
    h.iter().rev().take_while(|(t, _)| *t >= since).any(|&(_, v)| {
        let v = if wide { v } else { v & 0xFFFF_FFFF };
        compare(func, v & mask, reference & mask)
    })
}

/// Whether an equality wait for `reference` waits on a counter that has
/// moved past it: every recorded write from `reference`'s last one on (or,
/// when the counter passed it before the recorded window, every recorded
/// write; at least three, starting a whole number of steps and less than 64
/// steps above it) kept the value or added one step (1..4, the same
/// throughout; the title's frame counters count by 1 and by 2), and it rose
/// by two steps or more (a frame counter; a flag set from 0 to 1 is not
/// one). Run 20261011-103835: acb0x48 waited for 0x801 at 0x805 after cold
/// pipeline compiles and every queue stopped.
fn counter_passed(addr: u64, wide: bool, func: u32, mask: u64, reference: u64) -> bool {
    if func & 7 != 3 {
        return false;
    }
    let g = LABEL_WRITES.lock();
    let Some(h) = g.as_ref().and_then(|m| m.get(&addr)) else { return false };
    let val = |v: u64| (if wide { v } else { v & 0xFFFF_FFFF }) & mask;
    let r = reference & mask;
    let vals: Vec<u64> = h.iter().map(|&(_, v)| val(v)).collect();
    let Some(step) = vals.windows(2).map(|w| w[1].wrapping_sub(w[0])).find(|&d| d != 0) else { return false };
    if !(1..=4).contains(&step) {
        return false;
    }
    let later = match vals.iter().rposition(|&v| v == r) {
        Some(i) => &vals[i..],
        None if vals.len() >= 3 && vals[0] > r && (vals[0] - r) % step == 0 && vals[0] - r < 64 * step => &vals[..],
        None => return false,
    };
    later.windows(2).all(|w| w[1] == w[0] || w[1] == w[0].wrapping_add(step)) && later.last().is_some_and(|&l| l >= r.wrapping_add(2 * step))
}

/// The recorded writes of a waited-on label (wait reports).
fn label_writes(addr: u64) -> String {
    let g = LABEL_WRITES.lock();
    let Some(h) = g.as_ref().and_then(|m| m.get(&addr)) else { return "none".into() };
    let v: Vec<String> = h.iter().rev().take(12).rev().map(|&(_, v)| format!("{v:#x}")).collect();
    v.join(" ")
}
/// Equality waits passed because their counter had moved on (diagnostics).
pub static COUNTER_PASSED: AtomicU64 = AtomicU64::new(0);

/// CP memory writes the CPU performed, for the race checks (`tune
/// race_checks`): a global order number and, per 8-byte granule, the last
/// such write's (number, batch it ran after (0: performed on the CP's
/// thread when the CP reached it), queue, packet, opcode).
pub static CPW_SEQ: AtomicU64 = AtomicU64::new(0);
type CpwEntry = (u64, u64, Arc<str>, u64, u32);
static CPW_LOG: parking_lot::Mutex<Option<HashMap<u64, CpwEntry>>> = parking_lot::Mutex::new(None);

thread_local! {
    /// The batch whose deferred CP effects this thread is performing (the
    /// completion thread; 0 elsewhere).
    static AFTER_BATCH: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// The completion thread performs batch `seq`'s CP effects (0: done).
pub fn set_after_batch(seq: u64) {
    AFTER_BATCH.with(|c| c.set(seq));
}

/// The batch whose effects this thread performs (the completion thread's
/// actions; 0 elsewhere).
pub fn after_batch() -> u64 {
    AFTER_BATCH.with(|c| c.get())
}

fn cpw_note(queue: &str, a: u64, n: u64, at: u64, opcode: u32) {
    if !super::tune::race_checks() {
        return;
    }
    let s = CPW_SEQ.fetch_add(1, Ordering::AcqRel) + 1;
    let after = AFTER_BATCH.with(|c| c.get());
    let q: Arc<str> = queue.into();
    let mut g = CPW_LOG.lock();
    let m = g.get_or_insert_with(HashMap::new);
    if m.len() > 1 << 21 {
        m.clear();
    }
    let mut g8 = a & !7;
    while g8 < a + n.min(1 << 16) {
        m.insert(g8, (s, after, q.clone(), at, opcode));
        g8 += 8;
    }
}

/// The latest CPU-performed CP write into [a, a + n) with an order number
/// above `after` (as `cpw_since`).
pub fn cpw_since_range(a: u64, n: u64, after: u64) -> Option<(u64, u64, String, u64, u32)> {
    let g = CPW_LOG.lock();
    let m = g.as_ref()?;
    let mut best: Option<&CpwEntry> = None;
    let mut g8 = a & !7;
    while g8 < a + n {
        if let Some(e) = m.get(&g8).filter(|e| e.0 > after) {
            if best.is_none_or(|b| e.0 > b.0) {
                best = Some(e);
            }
        }
        g8 += 8;
    }
    best.map(|e| (e.0, e.1, e.2.to_string(), e.3, e.4))
}

/// The last CPU-performed CP write to the granule of `addr` with an order
/// number above `after`: (number, batch it ran after, queue, packet, opcode).
pub fn cpw_since(addr: u64, after: u64) -> Option<(u64, u64, String, u64, u32)> {
    let g = CPW_LOG.lock();
    let e = g.as_ref()?.get(&(addr & !7))?;
    (e.0 > after).then(|| (e.0, e.1, e.2.to_string(), e.3, e.4))
}

fn note_write(queue: &str, addr: u64, value: u64, at: u64) {
    super::homes::cpu_written(addr, 8);
    if WAITED.lock().as_ref().is_some_and(|w| w.contains(&(addr & !7))) {
        let mut g = LABEL_WRITES.lock();
        let h = g.get_or_insert_with(HashMap::new).entry(addr).or_default();
        // A repeated value refreshes its time: counters several queues
        // write each frame keep their recent frames in the window.
        match h.back_mut() {
            Some(last) if last.1 == value => last.0 = Instant::now(),
            _ => {
                if h.len() >= LABEL_WRITES_KEEP {
                    h.pop_front();
                }
                h.push_back((Instant::now(), value));
            }
        }
    }
    {
        let mut g = LAST_WRITE.lock();
        let m = g.get_or_insert_with(Default::default);
        if m.len() > 1 << 16 {
            m.clear();
        }
        m.insert(addr & !7, (queue.to_string(), value, at, addr));
    }
    latch_waits(addr & !7, 8);
    label_event(queue, b'W', addr, value, 0, at);
}

/// A WAIT_REG_MEM in progress. Hardware polls continuously, so a label one
/// queue sets is seen by a queue already waiting on it even if the CPU
/// resets it soon after (the title resets per-frame labels while building
/// the next frame). Our pollers sleep between checks and missed such
/// short-lived values, leaving every queue waiting on another (runs
/// 20261008-185032, -204128, -210312). CP writes therefore evaluate the
/// waits on the written word at once and latch the ones they satisfy.
struct ActiveWait {
    addr: u64,
    wide: bool,
    func: u32,
    mask: u64,
    reference: u64,
    hit: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

static ACTIVE_WAITS: parking_lot::Mutex<Vec<ActiveWait>> = parking_lot::Mutex::new(Vec::new());
/// Blocking waits in progress (cheap check before taking the lock).
static ACTIVE_COUNT: AtomicU64 = AtomicU64::new(0);

/// A wait on a label a GPU queue writes sleeps on the label's own address
/// (`WaitOnAddress`, `tune label_wake`) instead of polling: a promise for
/// the label, a latched CP write or a GPU-performed write to it wakes the
/// waits on that address (and a change of its value in memory before they
/// sleep). The compute queues wait on the graphics queue's labels 12-23 ms
/// at a time, ~4 times per frame each, and their polls (spins, yields,
/// 100 us sleeps) took the front end's CPUs (FINDINGS.md 2026-10-16).
fn wake_label(addr: u64, bytes: u8) {
    if ACTIVE_COUNT.load(Ordering::Acquire) == 0 {
        return;
    }
    unsafe { windows_sys::Win32::System::Threading::WakeByAddressAll(addr as *const core::ffi::c_void) };
    if bytes >= 8 {
        // A 32-bit wait on its upper half.
        unsafe { windows_sys::Win32::System::Threading::WakeByAddressAll((addr + 4) as *const core::ffi::c_void) };
    }
}

/// Whether a queue blocks on the label-sized word at `a` now.
pub(super) fn waited_now(a: u64) -> bool {
    if ACTIVE_COUNT.load(Ordering::Acquire) == 0 {
        return false;
    }
    let a = a & !7;
    ACTIVE_WAITS.lock().iter().any(|w| w.addr & !7 == a)
}
/// Waits satisfied by a latched CP write whose value was gone by the time
/// the waiter looked again (diagnostics).
pub static LATCHED: AtomicU64 = AtomicU64::new(0);
/// Parallel runs split at equal recording cost (`tune par_cost`).
pub static PAR_COSTED: AtomicU64 = AtomicU64::new(0);
/// Batches the graphics queue ended for `tune split_every` (diagnostics).
pub static SPLITS: AtomicU64 = AtomicU64::new(0);
/// Sleeps of label waits on their label's address (diagnostics).
pub static WAKE_SLEEPS: AtomicU64 = AtomicU64::new(0);

/// Latches the active waits overlapping [addr, addr+bytes) that the
/// current memory contents satisfy (called right after a CP write).
fn latch_waits(addr: u64, bytes: u64) {
    let g = ACTIVE_WAITS.lock();
    let mut hits: Vec<u64> = Vec::new();
    for w in g.iter() {
        let n = if w.wide { 8 } else { 4 };
        if w.addr < addr + bytes && addr < w.addr + n {
            let v = if w.wide { unsafe { rd64(w.addr) } } else { (unsafe { rd32(w.addr) }) as u64 };
            if compare(w.func, v & w.mask, w.reference & w.mask) {
                w.hit.store(true, Ordering::SeqCst);
                hits.push(w.addr);
            }
        }
    }
    drop(g);
    for a in hits {
        wake_label(a, 4);
    }
}

/// A label write the GPU performed in order within a batch
/// (`Exec::gpu_cp_write`), seen once the batch completed: recorded like a
/// CPU-performed CP write (the history late waiters pass on, `missed_write`)
/// and latched for the active waits its value satisfies (by the value: the
/// title may have reset the label since). Without this, a queue reaching its
/// wait after the batch completed and the label was reset blocked forever:
/// run 20261012-141114 stopped at flip 737 with graphics and acb0x50 waiting
/// for 1 at 0x11d37ff7e0, which acb0x20's GPU-side write had set ("recorded
/// writes none").
pub(super) fn gpu_label_written(queue: &str, addr: u64, value: u64, bytes: u8) {
    if WAITED.lock().as_ref().is_some_and(|w| w.contains(&(addr & !7))) {
        let mut g = LABEL_WRITES.lock();
        let h = g.get_or_insert_with(HashMap::new).entry(addr).or_default();
        match h.back_mut() {
            Some(last) if last.1 == value => last.0 = Instant::now(),
            _ => {
                if h.len() >= LABEL_WRITES_KEEP {
                    h.pop_front();
                }
                h.push_back((Instant::now(), value));
            }
        }
    }
    let mut woke: Vec<u64> = Vec::new();
    {
        let g = ACTIVE_WAITS.lock();
        for w in g.iter() {
            // The waited word within the written bytes.
            let v = if w.addr == addr {
                if w.wide {
                    if bytes < 8 {
                        continue;
                    }
                    value
                } else {
                    value & 0xFFFF_FFFF
                }
            } else if !w.wide && bytes >= 8 && w.addr == addr + 4 {
                value >> 32
            } else {
                continue;
            };
            if compare(w.func, v & w.mask, w.reference & w.mask) {
                w.hit.store(true, Ordering::SeqCst);
                woke.push(w.addr);
            }
        }
    }
    for a in woke {
        wake_label(a, 4);
    }
    label_event(queue, b'W', addr, value, 0, 0);
}

/// Label activity of all queues (CP writes, waits) on addresses some
/// WAIT_REG_MEM polls: a flight recorder dumped for the label when a wait
/// stalls. A value that changes between CP events was written by the CPU
/// (or by shader work).
struct LabelEvent {
    us: u64,
    queue: String,
    /// W = CP write, I = wait satisfied at once, S = wait blocks, E = wait
    /// satisfied, L = wait satisfied by a latched CP write.
    kind: u8,
    addr: u64,
    value: u64,
    reference: u64,
    packet: u64,
}

static LABEL_LOG: parking_lot::Mutex<std::collections::VecDeque<LabelEvent>> = parking_lot::Mutex::new(std::collections::VecDeque::new());
static WAITED: parking_lot::Mutex<Option<HashSet<u64>>> = parking_lot::Mutex::new(None);

/// Label values deferred CP writes will store: address -> (batch sequence,
/// value, bytes, whether the CPU stores it after the batch completed rather
/// than the GPU within it). A WAIT_REG_MEM that a promised value satisfies passes at
/// once and orders its queue after that batch (`Exec::depend_on`): its later
/// GPU work follows the batch on the device queue, its CP writes wait for
/// it, its CP reads wait for unfinished writers (`Exec::sync_for_reads`), so
/// the command processors record ahead of the GPU instead of idling until
/// each batch completes. `DS_PROMISES=0` turns this off.
static PROMISES: parking_lot::Mutex<Option<HashMap<u64, (u64, u64, u8, bool, Instant)>>> = parking_lot::Mutex::new(None);
/// Moves with every promise made: a polling wait looks at the promises
/// again only then (one that did not satisfy it cannot start to; it can
/// only go stale). Every poll took PROMISES' lock before: ~6% of the
/// compute queues' busy samples, run 20261013-115322.
static PROMISE_GEN: AtomicU64 = AtomicU64::new(0);
/// Waits passed on a promise (diagnostics).
pub static PROMISED: AtomicU64 = AtomicU64::new(0);
/// Multi-draw packets (DRAW_INDIRECT_MULTI / DRAW_INDEX_INDIRECT_MULTI), the
/// draws they expanded to, and those packets whose count GPU work wrote
/// (diagnostics: how much of the per-draw work they cause).
pub static MDI_PACKETS: AtomicU64 = AtomicU64::new(0);
/// Indexed indirect draws tried with the GPU reading their arguments whose
/// vertex program needed them on the CPU (`draw_gpu_args`).
pub static CPU_ARG_FALLBACKS: AtomicU64 = AtomicU64::new(0);
/// Waits passed on a GPU queue's write that ended no copy-cache epoch
/// (`Cp::last_wait_gpu`).
pub static EPOCH_GPU_WAITS: AtomicU64 = AtomicU64::new(0);
/// Indexed indirect draws with arguments in a video-memory home read on the
/// CPU, by why: a base-vertex SGPR, no start-instance SGPR of the vertex
/// stage, 8-bit indices, indices outside homes, a vertex program needing the
/// start instance on the CPU.
pub static CPU_ARG_REFUSED: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];
pub static MDI_DRAWS: AtomicU64 = AtomicU64::new(0);
pub static MDI_COUNT_FROM_MEMORY: AtomicU64 = AtomicU64::new(0);

fn promises_on() -> bool {
    super::tune::promises()
}

pub(super) fn promise(seq: u64, addr: u64, value: u64, bytes: u8, by_cpu: bool) {
    if !promises_on() {
        return;
    }
    let mut g = PROMISES.lock();
    let m = g.get_or_insert_with(HashMap::new);
    if m.len() > 1 << 14 {
        let done = super::exec::completed_seq();
        m.retain(|_, p| p.0 > done);
    }
    m.insert(addr, (seq, value, bytes, by_cpu, Instant::now()));
    drop(g);
    PROMISE_GEN.fetch_add(1, Ordering::Release);
    wake_label(addr, bytes);
}

/// The batch whose promised write to `addr` satisfies the comparison, and
/// whether the CPU performs that write.
fn promised(addr: u64, wide: bool, func: u32, mask: u64, reference: u64) -> Option<(u64, bool)> {
    let g = PROMISES.lock();
    let &(seq, value, bytes, by_cpu, _) = g.as_ref()?.get(&addr)?;
    if seq <= super::exec::completed_seq() || (wide && bytes < 8) {
        return None;
    }
    let v = if wide { value } else { value & 0xFFFF_FFFF };
    compare(func, v & mask, reference & mask).then_some((seq, by_cpu))
}
const LABEL_LOG_CAP: usize = 1 << 15;

fn label_event(queue: &str, kind: u8, addr: u64, value: u64, reference: u64, packet: u64) {
    let a = addr & !7;
    if kind == b'W' && !WAITED.lock().as_ref().is_some_and(|w| w.contains(&a)) {
        return;
    }
    static T0: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    let us = T0.get_or_init(Instant::now).elapsed().as_micros() as u64;
    let mut g = LABEL_LOG.lock();
    if g.len() >= LABEL_LOG_CAP {
        g.pop_front();
    }
    g.push_back(LabelEvent { us, queue: queue.to_string(), kind, addr: a, value, reference, packet });
}

/// The recorded history of one label (newest last).
fn label_history(addr: u64, max: usize) -> String {
    let a = addr & !7;
    let g = LABEL_LOG.lock();
    let ev: Vec<&LabelEvent> = g.iter().filter(|e| e.addr == a).collect();
    let mut out = String::new();
    for e in &ev[ev.len().saturating_sub(max)..] {
        let what = match e.kind {
            b'W' => format!("wrote {:#x}", e.value),
            b'I' => format!("wait ref {:#x}: satisfied at once (value {:#x})", e.reference, e.value),
            b'S' => format!("wait ref {:#x}: blocks (value {:#x})", e.reference, e.value),
            b'L' => format!("wait ref {:#x}: satisfied by a CP write it latched (value now {:#x})", e.reference, e.value),
            b'P' => format!("wait ref {:#x}: passed on a promised write (value now {:#x})", e.reference, e.value),
            b'M' => format!("wait ref {:#x}: passed on a write made after its submission (value now {:#x})", e.reference, e.value),
            _ => format!("wait ref {:#x}: satisfied (value {:#x})", e.reference, e.value),
        };
        out.push_str(&format!("\n    {:>10.3} ms {} {what} (packet {:#x})", e.us as f64 / 1000.0, e.queue, e.packet));
    }
    out
}

/// Records the label-sized words of a block the CP wrote (DMA fills and
/// copies up to 16 KiB; larger blocks are data, not labels).
fn note_block(queue: &str, addr: u64, bytes: u64, at: u64) {
    super::homes::cpu_written(addr, bytes);
    if bytes > 16 << 10 {
        return;
    }
    let mut a = addr & !7;
    while a < addr + bytes {
        let v = if valid(a, 8) { unsafe { std::ptr::read_volatile(a as *const u64) } } else { 0 };
        note_write(queue, a, v, at);
        a += 8;
    }
}

/// Blocking label waits per (waiting queue, queue that last wrote the
/// label): count and nanoseconds (`profile report`).
static WAIT_STATS: parking_lot::Mutex<Option<HashMap<(String, String), (u64, u64)>>> = parking_lot::Mutex::new(None);

pub fn reset_wait_stats() {
    *WAIT_STATS.lock() = None;
    *EPOCH_WAITS.lock() = None;
    *EPOCH_CLASSES.lock() = None;
    *LONG_WAITS.lock() = None;
    LONG_WAITS_LOGGED.store(0, Ordering::Relaxed);
}

/// Waits the CPU released (new copy-cache epochs, DS_PROFILE_GPU): (queue,
/// label, last CP write there) -> count.
static EPOCH_WAITS: parking_lot::Mutex<Option<HashMap<(String, u64, String), u64>>> = parking_lot::Mutex::new(None);
/// The same by (queue, how the wait passed (`Cp::last_wait_kind`), what the
/// label held) -> (count, labels).
static EPOCH_CLASSES: parking_lot::Mutex<Option<HashMap<(String, u8, &'static str), (u64, std::collections::HashSet<u64>)>>> = parking_lot::Mutex::new(None);

fn epoch_wait_lines() -> String {
    let g = EPOCH_WAITS.lock();
    let Some(m) = g.as_ref() else { return String::new() };
    let mut v: Vec<(&(String, u64, String), &u64)> = m.iter().collect();
    v.sort_by(|a, b| b.1.cmp(a.1));
    let total: u64 = v.iter().map(|x| *x.1).sum();
    let mut s = format!("    waits the CPU released (copy-cache epochs), {total} in all:\n");
    for ((q, a, by), n) in v.iter().take(12) {
        s.push_str(&format!("      {q} label {a:#x} ({by}): {n}\n"));
    }
    drop(g);
    let g = EPOCH_CLASSES.lock();
    if let Some(m) = g.as_ref() {
        let mut v: Vec<_> = m.iter().collect();
        v.sort_by(|a, b| b.1 .0.cmp(&a.1 .0));
        s.push_str("    the same by queue, how the wait passed (I at once, M a write seen since or a counter past it, E after polling, P/L on a GPU write) and what the label held:\n");
        for ((q, k, what), (n, labels)) in v.iter().take(24) {
            s.push_str(&format!("      {q} {} {what}: {n} ({} labels)\n", *k as char, labels.len()));
        }
    }
    s
}

/// The graphics command processor's time by packet class (`tune
/// cp_classes`): [ns without waits, packets, ns of waits] for draws and
/// dispatches of own chunks of parallel runs, draws and dispatches of serial
/// runs, CP effects (other packets), state packets, NOPs and private
/// operations, planning parallel runs, collecting their chunks, submissions
/// at buffer ends.
static CP_CLASSES: [[AtomicU64; 3]; 10] = [const { [const { AtomicU64::new(0) }; 3] }; 10];
/// The graphics command processor's serial commands by why they were not in
/// a parallel run (`tune cp_classes`): `SERIAL_WHY_NAMES`.
static SERIAL_WHY: [AtomicU64; 9] = [const { AtomicU64::new(0) }; 9];
const SERIAL_WHY_NAMES: [&str; 9] = ["small fetched buffers", "streamed", "runs of 1", "runs of 2-7", "runs of 8-31", "runs of 32-127", "runs of 128 or more", "other", "small fetched buffers' count"];
/// FRAMES when the classes were last cleared.
static CP_CLASSES_FROM: AtomicU64 = AtomicU64::new(0);
/// The same per PM4 opcode for the compute queue `tune cp_classes_acb`
/// names: [ns without waits, packets, ns of waits].
static ACB_OPS: [[AtomicU64; 3]; 256] = [const { [const { AtomicU64::new(0) }; 3] }; 256];

fn acb_op_end(opcode: u32, t: Option<(Instant, u64)>) {
    let Some((t0, w0)) = t else { return };
    let waited = super::timeline::waited_ns().saturating_sub(w0);
    let ns = (t0.elapsed().as_nanos() as u64).saturating_sub(waited);
    let c = &ACB_OPS[(opcode & 0xFF) as usize];
    c[0].fetch_add(ns, Ordering::Relaxed);
    c[1].fetch_add(1, Ordering::Relaxed);
    c[2].fetch_add(waited, Ordering::Relaxed);
}

fn op_name(opcode: u32) -> String {
    let n = match opcode {
        op::NOP => "NOP",
        op::SET_BASE => "SET_BASE",
        op::DISPATCH_DIRECT => "DISPATCH_DIRECT",
        op::DISPATCH_INDIRECT => "DISPATCH_INDIRECT",
        op::COND_EXEC => "COND_EXEC",
        op::WRITE_DATA => "WRITE_DATA",
        op::WAIT_REG_MEM => "WAIT_REG_MEM",
        op::INDIRECT_BUFFER => "INDIRECT_BUFFER",
        op::COPY_DATA => "COPY_DATA",
        op::EVENT_WRITE => "EVENT_WRITE",
        op::RELEASE_MEM => "RELEASE_MEM",
        op::DMA_DATA => "DMA_DATA",
        op::ACQUIRE_MEM => "ACQUIRE_MEM",
        op::SET_SH_REG => "SET_SH_REG",
        op::SET_SH_REG_INDEX => "SET_SH_REG_INDEX",
        op::SET_UCONFIG_REG => "SET_UCONFIG_REG",
        op::LOAD_SH_REG_INDEX => "LOAD_SH_REG_INDEX",
        op::WAIT_REG_MEM64 => "WAIT_REG_MEM64",
        1 => "label submissions after packets",
        _ => return format!("op {opcode:#04x}"),
    };
    n.to_string()
}
/// The planner's time (`tune cp_classes`; ns): [runs left serial, their
/// count, then for runs split: the scan, setup (fast clears, homes,
/// indirect-argument reads), the chunks' start states, the rest of starting
/// the workers, their count; then of starting: getting the workers and the
/// unsubmitted writes, submitting the batch before the run, making the run,
/// waking the workers].
static PLAN_PHASES: [AtomicU64; 11] = [const { AtomicU64::new(0) }; 11];
/// Runs planned while the batch being recorded had writes (`tune
/// cp_classes`): [such runs, runs whose writes the forwarded copies all
/// cover, the written ranges, the ranges' pieces not covered].
static PLAN_MAIN_WRITES: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// Lap clock of `par_plan` (`PLAN_PHASES`).
struct PlanLaps(Option<Instant>);

impl PlanLaps {
    fn mark(&mut self, k: usize) {
        if let Some(t) = self.0 {
            let now = Instant::now();
            PLAN_PHASES[k].fetch_add((now - t).as_nanos() as u64, Ordering::Relaxed);
            self.0 = Some(now);
        }
    }
}

const CP_CLASS_NAMES: [&str; 10] = ["own parallel chunks: draws", "own parallel chunks: dispatches", "serial draws", "serial dispatches", "CP effects", "state", "NOPs and private operations", "planning", "collecting", "buffer-end submissions"];

/// A packet class's start (`cp_classes`): the clock and the thread's waits.
fn class_start(on: bool) -> Option<(Instant, u64)> {
    on.then(|| (Instant::now(), super::timeline::waited_ns()))
}

fn class_end(k: usize, t: Option<(Instant, u64)>) {
    let Some((t0, w0)) = t else { return };
    let waited = super::timeline::waited_ns().saturating_sub(w0);
    let ns = (t0.elapsed().as_nanos() as u64).saturating_sub(waited);
    let c = &CP_CLASSES[k];
    c[0].fetch_add(ns, Ordering::Relaxed);
    c[1].fetch_add(1, Ordering::Relaxed);
    c[2].fetch_add(waited, Ordering::Relaxed);
}

/// Clears the packet classes (`profile reset`).
pub fn reset_classes() {
    for c in CP_CLASSES.iter().flatten().chain(ACB_OPS.iter().flatten()).chain(SERIAL_WHY.iter()).chain(PLAN_PHASES.iter()).chain(PLAN_MAIN_WRITES.iter()) {
        c.store(0, Ordering::Relaxed);
    }
    CP_CLASSES_FROM.store(super::FRAMES.load(Ordering::Relaxed), Ordering::Relaxed);
}

/// The packet classes per flip since they were cleared ("" when none).
pub fn classes_line() -> String {
    let flips = super::FRAMES.load(Ordering::Relaxed).saturating_sub(CP_CLASSES_FROM.load(Ordering::Relaxed)).max(1) as f64;
    let total: u64 = CP_CLASSES.iter().map(|c| c[0].load(Ordering::Relaxed)).sum();
    if total == 0 {
        return String::new();
    }
    let parts: Vec<String> = CP_CLASSES
        .iter()
        .zip(CP_CLASS_NAMES)
        .map(|(c, name)| {
            let (ns, n, w) = (c[0].load(Ordering::Relaxed), c[1].load(Ordering::Relaxed), c[2].load(Ordering::Relaxed));
            format!("{name} {:.2} ms ({:.0} packets, {:.1} us each; waits {:.2} ms)", ns as f64 / flips / 1e6, n as f64 / flips, ns as f64 / n.max(1) as f64 / 1e3, w as f64 / flips / 1e6)
        })
        .collect();
    let mut s = format!("    graphics command processor per flip by packet class (waits apart; {:.2} ms in all): {}\n", total as f64 / flips / 1e6, parts.join("; "));
    let why: Vec<String> = SERIAL_WHY.iter().zip(SERIAL_WHY_NAMES).map(|(c, name)| format!("{name} {:.1}", c.load(Ordering::Relaxed) as f64 / flips)).collect();
    s.push_str(&format!("    its serial commands per flip by why they were not in a parallel run: {}\n", why.join(", ")));
    let ph: Vec<u64> = PLAN_PHASES.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    let us = |ns: u64, n: u64| ns as f64 / n.max(1) as f64 / 1e3;
    s.push_str(&format!(
        "    planning per flip: {:.1} runs left serial at {:.1} us each; {:.1} runs split at {:.1} us each (scan {:.1}, setup {:.1}, chunk start states {:.1}, starting workers {:.1}: getting them and the unsubmitted writes {:.1}, submitting the batch before the run {:.1}, making the run {:.1}, waking them {:.1}, the rest {:.1})\n",
        ph[1] as f64 / flips,
        us(ph[0], ph[1]),
        ph[6] as f64 / flips,
        us(ph[2] + ph[3] + ph[4] + ph[5] + ph[7] + ph[8] + ph[9] + ph[10], ph[6]),
        us(ph[2], ph[6]),
        us(ph[3], ph[6]),
        us(ph[4], ph[6]),
        us(ph[5] + ph[7] + ph[8] + ph[9] + ph[10], ph[6]),
        us(ph[7], ph[6]),
        us(ph[8], ph[6]),
        us(ph[9], ph[6]),
        us(ph[10], ph[6]),
        us(ph[5], ph[6])
    ));
    let mw: Vec<u64> = PLAN_MAIN_WRITES.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    s.push_str(&format!(
        "    runs planned with the batch's writes unsubmitted: {:.1} per flip, {:.1} of them all covered by forwarded copies ({} written ranges, {} pieces not covered)\n",
        mw[0] as f64 / flips,
        mw[1] as f64 / flips,
        mw[2],
        mw[3]
    ));
    let mut ops: Vec<(usize, u64, u64, u64)> = ACB_OPS
        .iter()
        .enumerate()
        .map(|(k, c)| (k, c[0].load(Ordering::Relaxed), c[1].load(Ordering::Relaxed), c[2].load(Ordering::Relaxed)))
        .filter(|&(_, ns, _, w)| ns + w > 0)
        .collect();
    if !ops.is_empty() {
        let all: u64 = ops.iter().map(|o| o.1).sum();
        ops.sort_by_key(|o| std::cmp::Reverse(o.1));
        let parts: Vec<String> = ops
            .iter()
            .map(|&(k, ns, n, w)| format!("{} {:.2} ms ({:.0} packets, {:.1} us each; waits {:.2} ms)", op_name(k as u32), ns as f64 / flips / 1e6, n as f64 / flips, ns as f64 / n.max(1) as f64 / 1e3, w as f64 / flips / 1e6))
            .collect();
        s.push_str(&format!("    compute queue {:#x} per flip by packet (waits apart; {:.2} ms in all): {}\n", super::tune::cp_classes_acb(), all as f64 / flips / 1e6, parts.join("; ")));
    }
    s
}

pub fn wait_stats() -> String {
    let mut s = epoch_wait_lines();
    s.push_str(&wait_stats_inner());
    s
}

/// Blocking label waits by waiter and writer, longest total first.
fn wait_stats_inner() -> String {
    let g = WAIT_STATS.lock();
    let Some(m) = g.as_ref() else { return String::new() };
    let mut v: Vec<(&(String, String), &(u64, u64))> = m.iter().collect();
    v.sort_by(|a, b| b.1 .1.cmp(&a.1 .1));
    let mut s = format!("    label waits (waiter <- last writer: count, total ms, mean us; {} sleeps on wakes, `label_wake`):\n", WAKE_SLEEPS.load(Ordering::Relaxed));
    for ((w, by), (n, ns)) in v.iter().take(24) {
        s.push_str(&format!("      {w} <- {by}: {n}, {} ms, {} us\n", ns / 1_000_000, ns / 1000 / (*n).max(1)));
    }
    s
}

/// Blocking label waits of 1 ms or more by (waiter, last writer, label):
/// [count, ns, ended in memory with no promise for the label, ended in
/// memory with a promise for it (of a value that did not satisfy the wait or
/// made after the wait ended), passed on a promise, latched]; the first 40
/// are logged in detail (`long_wait`).
static LONG_WAITS: parking_lot::Mutex<Option<HashMap<(String, String, u64), [u64; 6]>>> = parking_lot::Mutex::new(None);
static LONG_WAITS_LOGGED: AtomicU64 = AtomicU64::new(0);

/// A blocking wait on `a` that lasted `ns` and ended as `kind` (b'E': the
/// value in memory, b'P': a promise, b'L': latched).
#[allow(clippy::too_many_arguments)]
fn long_wait(waiter: &str, a: u64, wide: bool, func: u32, mask: u64, reference: u64, at: u64, t0: Instant, kind: u8) {
    let ns = t0.elapsed().as_nanos() as u64;
    if ns < 1_000_000 {
        return;
    }
    let (by, wv, wpk) = last_write(a).unwrap_or_else(|| ("cpu".into(), 0, 0));
    let p = PROMISES.lock().as_ref().and_then(|m| m.get(&a).copied());
    let slot = match (kind, p) {
        (b'P', _) => 4,
        (b'L', _) => 5,
        (_, None) => 2,
        _ => 3,
    };
    {
        let mut g = LONG_WAITS.lock();
        let e = g.get_or_insert_with(HashMap::new).entry((waiter.to_string(), by.clone(), a)).or_default();
        e[0] += 1;
        e[1] += ns;
        e[slot] += 1;
    }
    // DS_TRACE_WAITS=<queue>: that queue's long waits (the first 200), with
    // the same detail, whenever they happen.
    static TRACE_Q: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    static TRACED_Q: AtomicU64 = AtomicU64::new(0);
    let traced = TRACE_Q.get_or_init(|| std::env::var("DS_TRACE_WAITS").ok()).as_deref() == Some(waiter) && TRACED_Q.fetch_add(1, Ordering::Relaxed) < 200;
    if traced || LONG_WAITS_LOGGED.fetch_add(1, Ordering::Relaxed) < 40 {
        let promise = match p {
            None => "none".to_string(),
            Some((seq, v, bytes, by_cpu, t)) => {
                let since = t.checked_duration_since(t0).map(|d| format!("{:.2} ms into the wait", d.as_secs_f64() * 1e3)).unwrap_or_else(|| format!("{:.2} ms before the wait", t0.duration_since(t).as_secs_f64() * 1e3));
                let vv = if wide { v } else { v & 0xFFFF_FFFF };
                format!("batch {seq} ({}; completed through {}), value {v:#x} ({bytes} bytes{}), made {since}, satisfies: {}", if by_cpu { "CPU after it" } else { "GPU in it" }, super::exec::completed_seq(), "", !(wide && bytes < 8) && compare(func, vv & mask, reference & mask))
            }
        };
        let header = if wpk != 0 && valid(wpk, 4) { unsafe { rd32(wpk) } } else { 0 };
        let op = if header >> 30 == 3 { format!("{:#04x}", (header >> 8) & 0xFF) } else { "?".into() };
        crate::info!(
            "gpu[{waiter}]: label wait {:.2} ms at {at:#x} on [{a:#x}] func {func} ref {reference:#x} mask {mask:#x} ended {} (value now {:#x}); last write {wv:#x} by {by} (packet {wpk:#x}, opcode {op}); promise: {promise}; label history:{}",
            ns as f64 / 1e6,
            match kind {
                b'P' => "on a promise",
                b'L' => "latched",
                _ => "in memory",
            },
            if wide { unsafe { rd64(a) } } else { (unsafe { rd32(a) }) as u64 },
            label_history(a, 10)
        );
    }
}

/// Long label waits by waiter, writer and label (profile report).
pub fn long_waits_line() -> String {
    let g = LONG_WAITS.lock();
    let Some(m) = g.as_ref() else { return String::new() };
    let mut v: Vec<_> = m.iter().collect();
    v.sort_by_key(|(_, e)| std::cmp::Reverse(e[1]));
    let parts: Vec<String> = v
        .iter()
        .take(10)
        .map(|((w, by, a), e)| format!("{w} <- {by} [{a:#x}]: {} waits {:.1} ms (in memory without a promise {}, with one {}, on a promise {}, latched {})", e[0], e[1] as f64 / 1e6, e[2], e[3], e[4], e[5]))
        .collect();
    format!("    label waits of 1 ms or more: {}\n", parts.join("; "))
}

/// The last CP write to the label-sized word at `addr` (wait traces).
pub fn last_write_info(addr: u64) -> Option<String> {
    last_write(addr).map(|(q, v, pk)| format!("{q} wrote {v:#x} (packet {pk:#x})"))
}

fn last_write(addr: u64) -> Option<(String, u64, u64)> {
    LAST_WRITE.lock().as_ref()?.get(&(addr & !7)).map(|(q, v, pk, _)| (q.clone(), *v, *pk))
}

/// `last_write` with the exact address written.
fn last_write_at(addr: u64) -> Option<(String, u64, u64, u64)> {
    LAST_WRITE.lock().as_ref()?.get(&(addr & !7)).cloned()
}

/// Whether a CP write to exactly `addr` recorded for the label (`LABEL_WRITES`)
/// had value `v` (the low half for 4-byte labels).
fn label_wrote(addr: u64, wide: bool, v: u64) -> bool {
    let g = LABEL_WRITES.lock();
    let Some(h) = g.as_ref().and_then(|m| m.get(&addr)) else { return false };
    h.iter().any(|&(_, w)| if wide { w == v } else { w as u32 == v as u32 })
}

/// What the queue thread is doing (diagnostics).
pub static STATE: parking_lot::Mutex<Vec<(String, String)>> = parking_lot::Mutex::new(Vec::new());

fn set_state(name: &str, s: String) {
    let mut g = STATE.lock();
    match g.iter_mut().find(|(n, _)| n == name) {
        Some(e) => e.1 = s,
        None => g.push((name.to_string(), s)),
    }
}

impl Cp {
    pub fn new(name: &str, compute: bool) -> Cp {
        Cp {
            name: name.to_string(),
            last_draw: std::borrow::Cow::Borrowed(""),
            repeat_ok: false,
            ctx_dirty: true,
            last_draw_progs: (0, 0, 0),
            same_run: 0,
            compute,
            regs: Regs::new(),
            index_base: 0,
            index_count: 0,
            num_instances: 1,
            draw_indirect_base: 0,
            dispatch_indirect_base: 0,
            markers: Vec::new(),
            ctx_stack: Vec::new(),
            cur_packet: 0,
            cur_opcode: 0,
            waiting_on: 0,
            wait_flag: None,
            unknown_seen: HashSet::new(),
            notes_seen: HashSet::new(),
            depth: 0,
            exec: None,
            exec_failed: false,
            fence_pending: false,
            pred: None,
            submitted_at: None,
            more_follows: None,
            next_in_submission: false,
            ahead: std::collections::VecDeque::new(),
            cur_ahead: std::cell::RefCell::new([None, None]),
            scan_regs: std::cell::RefCell::new((Vec::new(), Vec::new())),
            compile_ctx: std::cell::Cell::new(None),
            wait_ctx: None,
            par: None,
            par_skip: (0, 0),
            par_skip_n: 0,
            par_pool: super::par::Pool::new(),
            chunk_costs: None,
            run_costs: HashMap::new(),
            last_wait_gpu: false,
            last_wait_kind: b'-',
        }
    }

    pub fn chunk_state(&self) -> ChunkState {
        let (c, s, u) = if super::tune::lean_regs() { (high(Space::Ctx, CTX_DWORDS), high(Space::Sh, SH_DWORDS), high(Space::Uc, UC_DWORDS)) } else { (CTX_DWORDS, SH_DWORDS, UC_DWORDS) };
        ChunkState {
            regs: Regs { ctx: self.regs.ctx[..c].to_vec(), sh: self.regs.sh[..s].to_vec(), uc: self.regs.uc[..u].to_vec() },
            index_base: self.index_base,
            index_count: self.index_count,
            num_instances: self.num_instances,
            draw_indirect_base: self.draw_indirect_base,
            dispatch_indirect_base: self.dispatch_indirect_base,
        }
    }

    pub fn set_chunk_state(&mut self, s: &ChunkState) {
        self.repeat_ok = false;
        self.ctx_dirty = true;
        copy_prefix(&mut self.regs.ctx, &s.regs.ctx, high(Space::Ctx, CTX_DWORDS));
        copy_prefix(&mut self.regs.sh, &s.regs.sh, high(Space::Sh, SH_DWORDS));
        copy_prefix(&mut self.regs.uc, &s.regs.uc, high(Space::Uc, UC_DWORDS));
        self.index_base = s.index_base;
        self.index_count = s.index_count;
        self.num_instances = s.num_instances;
        self.draw_indirect_base = s.draw_indirect_base;
        self.dispatch_indirect_base = s.dispatch_indirect_base;
    }

    /// Gives this (worker) command processor a deferred executor.
    pub fn make_worker_exec(&mut self) -> Result<(), String> {
        let mut e = super::exec::Exec::new(&self.name)?;
        e.set_deferred();
        self.exec = Some(Box::new(e));
        Ok(())
    }

    pub fn chunk_start(&mut self, main_writes: &Arc<super::ranges::Ranges>, submitted: &Arc<std::sync::atomic::AtomicBool>) {
        if let Some(e) = self.exec.as_mut() {
            e.stopped = false;
            e.stop_why = "";
            let pending = !main_writes.is_empty() && !submitted.load(Ordering::Acquire);
            e.set_main_writes(Some(main_writes.clone()), pending.then(|| submitted.clone()));
        }
    }

    /// Closes the batch a chunk recorded into.
    pub fn chunk_finish(&mut self) {
        if let Some(e) = self.exec.as_mut() {
            e.close();
        }
    }

    /// Executes the packets in [from, to) of a run (a parallel worker):
    /// returns the address of the packet where the executor stopped (it
    /// and the rest are left to the main command processor), or None.
    pub fn run_chunk(&mut self, from: u64, to: u64) -> Option<u64> {
        let f = Fetched::new(from, 0);
        let mut cur = from;
        let mut commands = 0u64;
        let mut last = unsafe { core::arch::x86_64::_rdtsc() };
        while cur < to {
            if super::shutting_down() || !valid(cur, 4) {
                return Some(cur);
            }
            let h = f.rd(cur);
            match h >> 30 {
                3 => {}
                2 => {
                    cur += 4;
                    continue;
                }
                _ => return Some(cur),
            }
            let count = (h >> 16) & 0x3FFF;
            let opcode = (h >> 8) & 0xFF;
            if count == 0x3FFF {
                cur += 4;
                continue;
            }
            let n = count as u64 + 1;
            let next = cur + 4 * (n + 1);
            if next > to || !valid(cur, 4 * (n + 1)) || !(is_command(opcode) || is_state(opcode) || opcode == op::NOP || par_event(opcode, self.compute)) {
                return Some(cur);
            }
            PACKETS.fetch_add(1, Ordering::Relaxed);
            self.cur_packet = cur;
            self.cur_opcode = opcode;
            let p = |i: u64| f.rd(cur + 4 * (1 + i));
            // Stages prepared ahead within the chunk, as in `run`.
            let command = is_command(opcode);
            if command && !self.ahead.is_empty() {
                if self.ahead.front().is_some_and(|c| c.at == cur) {
                    let c = self.ahead.pop_front().unwrap();
                    *self.cur_ahead.borrow_mut() = c.stages;
                } else {
                    self.drop_ahead();
                }
            }
            let flow = self.packet(opcode, h, cur, n, &p);
            if command {
                for s in self.take_ahead().iter().flatten() {
                    s.cancel();
                }
            }
            if self.exec.as_ref().is_some_and(|e| e.stopped) {
                super::par::COMMANDS.fetch_add(commands, Ordering::Relaxed);
                self.drop_ahead();
                return Some(cur);
            }
            if !matches!(flow, Flow::Next) {
                self.drop_ahead();
                return Some(next);
            }
            commands += command as u64;
            if command {
                if let Some(v) = self.chunk_costs.as_mut() {
                    let now = unsafe { core::arch::x86_64::_rdtsc() };
                    v.push(now.wrapping_sub(last).min(u32::MAX as u64) as u32);
                    last = now;
                }
            }
            if command && self.ahead.is_empty() && super::ahead::enabled_for(self.compute) {
                self.scan_ahead(&f, next, to);
            }
            cur = next;
        }
        super::par::COMMANDS.fetch_add(commands, Ordering::Relaxed);
        self.drop_ahead();
        None
    }

    /// Why this worker's chunk stopped.
    pub fn worker_stop_why(&self) -> &'static str {
        self.exec.as_ref().map_or("", |e| e.stop_why)
    }

    /// After a label wait of the graphics queue: the CPU may have written
    /// data before releasing it, so shared batch-start copies are dropped
    /// (`gpu::precache`), unless the label holds what a GPU queue's CP wrote
    /// last (the queues synchronize this way ~250 times per frame).
    fn end_copy_scope_after_wait(&self, a: u64, wide: bool) {
        if !valid(a, if wide { 8 } else { 4 }) {
            return;
        }
        if self.last_wait_gpu && super::tune::epoch_gpu_waits() {
            EPOCH_GPU_WAITS.fetch_add(1, Ordering::Relaxed);
            return;
        }
        let now = unsafe { if wide { rd64(a) } else { rd32(a) as u64 } };
        let last = last_write(a);
        let by_gpu = last.as_ref().is_some_and(|(q, v, _)| q.starts_with("gpu-") && if wide { *v == now } else { *v as u32 == now as u32 });
        if !by_gpu {
            super::precache::end_scope(1);
            if super::exec::profiling_on() {
                let what = match last_write_at(a) {
                    None => "no CP write recorded",
                    Some((q, _, _, at)) if !q.starts_with("gpu-") => if at == a { "last written by a non-GPU queue" } else { "last written by a non-GPU queue (other half)" },
                    Some((_, _, _, at)) if !wide && at != a => if label_wrote(a, wide, now) { "a GPU queue's earlier value (slot's last write to the other half)" } else { "no GPU queue's value (slot's last write to the other half)" },
                    Some(_) => if label_wrote(a, wide, now) { "a GPU queue's earlier value" } else { "no GPU queue's value" },
                };
                let mut g = EPOCH_CLASSES.lock();
                let e = g.get_or_insert_with(HashMap::new).entry((self.name.clone(), self.last_wait_kind, what)).or_default();
                e.0 += 1;
                e.1.insert(a);
                drop(g);
                let by = match last {
                    Some((q, v, _)) if (if wide { v == now } else { v as u32 == now as u32 }) => format!("written by {q}"),
                    Some((q, _, _)) => format!("{q} wrote another value"),
                    None => "no CP write recorded".into(),
                };
                *EPOCH_WAITS.lock().get_or_insert_with(HashMap::new).entry((self.name.clone(), a, by)).or_default() += 1;
            }
        }
    }

    /// A title submission starts (its command buffers follow): CPU writes
    /// come between submissions, so shared batch-start copies of the
    /// graphics queue are dropped (`gpu::precache`).
    pub fn new_submission(&mut self) {
        super::precache::end_scope(0);
    }

    /// Submits the batches this worker closed (the main command processor,
    /// in program order); the last one's sequence.
    pub fn worker_submit(&mut self) -> Option<u64> {
        self.exec.as_mut().and_then(|e| e.submit_closed())
    }

    /// Drops the batches this worker closed (a cancelled parallel run).
    pub fn worker_discard(&mut self) {
        if let Some(e) = self.exec.as_mut() {
            e.discard_closed();
        }
    }

    /// Applies a state packet of a run to `st` as `packet` would.
    fn apply_state(st: &mut ChunkState, opcode: u32, n: u64, p: &dyn Fn(u64) -> u32) {
        let set = |v: &mut Vec<u32>, space: Space, cap: usize, first: u32| {
            for i in 1..n {
                put_reg(v, space, cap, (first + (i - 1) as u32) as usize, p(i));
            }
        };
        let load = |v: &mut Vec<u32>, space: Space, cap: usize, a: u64, count: u32| {
            if count == 0 || !valid(a, 8 * count as u64) {
                return;
            }
            for i in 0..count as u64 {
                let (o, val) = unsafe { (rd32(a + 8 * i), rd32(a + 8 * i + 4)) };
                put_reg(v, space, cap, (o & 0xFFFF) as usize, val);
            }
        };
        match opcode {
            op::SET_CONTEXT_REG | op::SET_CONTEXT_REG_INDEX => set(&mut st.regs.ctx, Space::Ctx, CTX_DWORDS, p(0) & 0xFFFF),
            op::SET_SH_REG | op::SET_SH_REG_INDEX => set(&mut st.regs.sh, Space::Sh, SH_DWORDS, p(0) & 0xFFFF),
            op::SET_UCONFIG_REG | op::SET_UCONFIG_REG_INDEX => set(&mut st.regs.uc, Space::Uc, UC_DWORDS, p(0) & 0xFFFF),
            op::LOAD_CONTEXT_REG_INDEX => load(&mut st.regs.ctx, Space::Ctx, CTX_DWORDS, addr64(p(0), p(1)), p(2)),
            op::LOAD_SH_REG_INDEX => load(&mut st.regs.sh, Space::Sh, SH_DWORDS, addr64(p(0), p(1)), p(2)),
            op::LOAD_UCONFIG_REG => load(&mut st.regs.uc, Space::Uc, UC_DWORDS, addr64(p(0), p(1)), p(2)),
            op::INDEX_TYPE => put_reg(&mut st.regs.uc, Space::Uc, UC_DWORDS, crate::hle::libs::agc_cmd::REG_VGT_INDEX_TYPE as usize, p(0)),
            op::INDEX_BASE => st.index_base = addr64(p(0), p(1)),
            op::INDEX_BUFFER_SIZE => st.index_count = p(0),
            op::NUM_INSTANCES => st.num_instances = p(0).max(1),
            op::SET_BASE => {
                let base = addr64(p(1), p(2));
                if p(0) == 2 {
                    st.dispatch_indirect_base = base
                } else {
                    st.draw_indirect_base = base
                }
            }
            op::DRAW_INDIRECT | op::DRAW_INDEX_INDIRECT => {
                // The base vertex and start instance the packet hands to user SGPRs.
                let a = st.draw_indirect_base + p(0) as u64;
                let (bytes, k) = if opcode == op::DRAW_INDEX_INDIRECT { (20, 3) } else { (16, 2) };
                if !valid(a, bytes) {
                    return;
                }
                let index_ok = opcode == op::DRAW_INDIRECT || matches!(st.regs.uc.get(crate::hle::libs::agc_cmd::REG_VGT_INDEX_TYPE as usize).map(|t| t & 3), Some(0 | 1));
                if !index_ok {
                    return;
                }
                let (bv, si) = unsafe { (rd32(a + 4 * k), rd32(a + 4 * (k + 1))) };
                let (vtx_loc, inst_loc) = (p(1) & 0xFFFF, p(1) >> 16);
                let inst_loc = if inst_loc != 0 {
                    inst_loc
                } else {
                    let r = ScanRegs { sh: &st.regs.sh, ctx: &st.regs.ctx };
                    crate::hle::libs::agc::instance_offset_slot(super::shaders::ngg_addr_of(&r)).map_or(0, |s| 0x8C + s)
                };
                if vtx_loc != 0 {
                    put_reg(&mut st.regs.sh, Space::Sh, SH_DWORDS, vtx_loc as usize, bv);
                }
                if inst_loc != 0 {
                    put_reg(&mut st.regs.sh, Space::Sh, SH_DWORDS, inst_loc as usize, si);
                }
            }
            _ => {}
        }
    }

    /// Splits the synchronization-free run that starts with the command at
    /// `from` into chunks when it is long enough: the first stays here, the
    /// others go to the workers' queue now (`gpu::par`).
    fn par_plan(&mut self, f: &Fetched, from: u64, end: u64) {
        // (Profile: `tune cp_classes`, `PLAN_PHASES`.)
        let mut lap = PlanLaps(super::tune::cp_classes().then(Instant::now));
        let k = if self.compute { (super::tune::par_compute() as usize).min(8) } else { super::par::workers() };
        // The run's commands, its end and the indirect arguments read in it.
        let mut ends: Vec<u64> = Vec::new();
        let mut args: Vec<(u64, u64)> = Vec::new();
        let (mut draw_base, mut disp_base) = (self.draw_indirect_base, self.dispatch_indirect_base);
        let mut c = from;
        let mut why = 0x130usize;
        let mut sync = false;
        // Whether the chunks' indirect dispatches read their arguments on
        // the GPU (`Exec::dispatch_gpu_indirect`; compute queues' runs,
        // `tune par_compute`; captures and interpreter checks read them
        // here).
        let gpu_dispatch_args = self.compute && super::tune::gpu_indirect() && !super::capture::capturing() && !super::verify::enabled() && !(super::exec::enabled() && super::emucheck::wanted(self));
        // The run's signature: its packet headers (opcode and size), the
        // same when the title records the same pass again.
        let mut sig = 0xcbf2_9ce4_8422_2325u64;
        let run_end = loop {
            if c >= end || !valid(c, 4) {
                break c;
            }
            let h = f.rd(c);
            sig = (sig ^ h as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            match h >> 30 {
                3 => {}
                2 => {
                    c += 4;
                    continue;
                }
                _ => break c,
            }
            let count = (h >> 16) & 0x3FFF;
            if count == 0x3FFF {
                c += 4;
                continue;
            }
            if h & 1 != 0 {
                why = 0x133;
                break c;
            }
            let opc = (h >> 8) & 0xFF;
            let n = count as u64 + 1;
            let next = c + 4 * (n + 1);
            if next > end || !valid(c, 4 * (n + 1)) {
                break c;
            }
            let at = c;
            let p = |i: u64| f.rd(at + 4 * (1 + i));
            if is_command(opc) {
                ends.push(next);
                match opc {
                    op::DRAW_INDIRECT => args.push((draw_base + p(0) as u64, 16)),
                    op::DRAW_INDEX_INDIRECT => args.push((draw_base + p(0) as u64, 20)),
                    // Dispatch arguments the chunks let the GPU read
                    // (`Exec::dispatch_gpu_indirect`) need no wait here
                    // (one per run start cost queue 0x50's runs a GPU round
                    // trip of ~2.5 ms, twice per flip with `par_compute`).
                    op::DISPATCH_INDIRECT if gpu_dispatch_args => {}
                    op::DISPATCH_INDIRECT => args.push((if n >= 3 { addr64(p(0), p(1)) } else { disp_base + p(0) as u64 }, 12)),
                    _ => {}
                }
                if ends.len() >= PAR_MAX {
                    why = 0x131;
                    break next;
                }
            } else if opc == op::NOP {
                if n >= 2 && p(0) == DS_MAGIC && !matches!(p(1), pop::PUSH_MARKER | pop::SET_MARKER | pop::POP_MARKER) {
                    why = 0x100 + (p(1) as usize).min(0x2F);
                    break c;
                }
            } else if opc == op::SET_BASE {
                if p(0) == 2 {
                    disp_base = addr64(p(1), p(2));
                } else {
                    draw_base = addr64(p(1), p(2));
                }
            } else if par_event(opc, self.compute) {
                sync = true;
            } else if !is_state(opc) {
                // EVENT_WRITE by event type (statistics: partial flushes
                // and cache events order only GPU work).
                why = if opc == op::EVENT_WRITE { 0x200 + (p(0) & 0x3F) as usize } else { opc as usize };
                break c;
            }
            c = next;
        };
        let n = ends.len();
        super::par::note_run(n, why, self.compute);
        let par_min = (if self.compute { super::tune::par_min_compute() as usize } else { super::par::min_run() }).max(48);
        let chunk_min = (par_min / 3).clamp(16, PAR_CHUNK_MIN);
        let chunks = ((k + 1) * super::par::split()).min(n / chunk_min);
        if n < par_min || chunks < 2 {
            // Not again before execution passes this run's end.
            self.par_skip = (end, run_end);
            self.par_skip_n = n;
            lap.mark(0);
            PLAN_PHASES[1].fetch_add(1, Ordering::Relaxed);
            return;
        }
        lap.mark(2);
        // Pending fast clears are applied now (a worker cannot apply one at
        // the image's first use in program order, so each stopped its chunk:
        // ~4 times per frame in the courtyard). The run's start is a
        // synchronization point; the clear lands before every use in it.
        let clears = super::images::with(|im| im.pending_clear_keys());
        if !clears.is_empty() {
            if let Some(e) = self.exec.as_mut() {
                e.resolve_fast_clears(&clears);
            }
        }
        // Indirect arguments are read when the chunks run: GPU work still
        // writing them completes first. Draw arguments (20-byte records)
        // that GPU work writes through VRAM shadows get a video-memory home
        // now, before any worker records (`tune draw_args_gpu`): read by the
        // GPU, they need no wait (a draw that cannot read them there reads
        // them after a write-back: its chunk stops).
        if !super::tune::draw_args_gpu() && super::homes::any_watched() {
            if let Some(e) = self.exec.as_mut() {
                e.evict_watched_homes();
            }
        }
        if !args.is_empty() && super::tune::draw_args_gpu() {
            let cands = super::homes::ready_args_candidates();
            if !cands.is_empty() {
                // Every writer completes first (merged into guest memory,
                // which a home copies in at its first use); no worker
                // records yet.
                self.sync_reads(&cands);
                for &(b, m) in &cands {
                    super::homes::create_args(b, m);
                }
            }
            args.retain(|&(a, n)| !(n == 20 && super::homes::contains(a, n)));
        }
        if !args.is_empty() {
            args.sort_unstable();
            let mut merged: Vec<(u64, u64)> = Vec::new();
            for (a, n) in args {
                match merged.last_mut() {
                    Some(m) if a <= m.0 + m.1 + 4096 => m.1 = m.1.max(a + n - m.0),
                    _ => merged.push((a, n)),
                }
            }
            self.sync_reads(&merged);
        }
        lap.mark(3);
        // The main command processor's chunk first (`par::main_weight`
        // times a worker's), then equal ones for the workers' queue.
        let m = chunks - 1;
        let w = super::par::main_weight();
        let main_n = ((n as f64 * w / (w + m as f64)).round() as usize).clamp(chunk_min, n - m * chunk_min);
        let per = (n - main_n) / m;
        // At equal recording cost when the run was measured before (`tune
        // par_cost`), else at equal command counts.
        let timed = super::tune::par_cost();
        let sig = if timed { (sig ^ n as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1 } else { 0 };
        let frame = super::FRAMES.load(Ordering::Relaxed);
        let costed = if timed {
            if self.run_costs.len() > 512 {
                self.run_costs.retain(|_, r| r.seen + 120 >= frame);
            }
            let rc = self.run_costs.entry(sig).or_insert_with(|| RunCost { costs: vec![f32::NAN; n], seen: frame });
            rc.seen = frame;
            if rc.costs.len() == n {
                cost_firsts(&rc.costs, main_n, n, m, chunk_min)
            } else {
                None
            }
        } else {
            None
        };
        if costed.is_some() {
            PAR_COSTED.fetch_add(1, Ordering::Relaxed);
        }
        let firsts: Vec<usize> = costed.unwrap_or_else(|| (0..m).map(|i| main_n + i * per).collect());
        let starts: Vec<u64> = firsts.iter().map(|&i| ends[i - 1]).collect();
        // The workers are woken now to spin for the run (`tune
        // par_prewake`): the chunk start states and the submission before
        // the run take ~120-150 us more.
        if super::tune::par_prewake() > 0 {
            self.par_pool.prewake();
        }
        // The state at each chunk's start.
        let mut st = self.chunk_state();
        let mut states: Vec<Box<ChunkState>> = Vec::new();
        let mut c = from;
        // (`tune plan_check`: the same a register at a time.)
        let mut check = super::tune::plan_check().then(|| self.chunk_state());
        for &s in &starts {
            while c < s {
                let h = f.rd(c);
                if h >> 30 == 2 {
                    c += 4;
                    continue;
                }
                let count = (h >> 16) & 0x3FFF;
                if count == 0x3FFF {
                    c += 4;
                    continue;
                }
                let n = count as u64 + 1;
                let at = c;
                let opc = (h >> 8) & 0xFF;
                // Register ranges in one copy (most of a run's packets).
                let regs = match opc {
                    op::SET_CONTEXT_REG | op::SET_CONTEXT_REG_INDEX => Some((Space::Ctx, CTX_DWORDS)),
                    op::SET_SH_REG | op::SET_SH_REG_INDEX => Some((Space::Sh, SH_DWORDS)),
                    op::SET_UCONFIG_REG | op::SET_UCONFIG_REG_INDEX => Some((Space::Uc, UC_DWORDS)),
                    _ => None,
                };
                match regs {
                    Some((space, cap)) if n >= 2 => {
                        let v = match space {
                            Space::Ctx => &mut st.regs.ctx,
                            Space::Sh => &mut st.regs.sh,
                            Space::Uc => &mut st.regs.uc,
                        };
                        put_regs(v, space, cap, (f.rd(at + 4) & 0xFFFF) as usize, f, at + 8, (n - 1) as usize);
                    }
                    _ => Self::apply_state(&mut st, opc, n, &|i: u64| f.rd(at + 4 * (1 + i))),
                }
                if let Some(st2) = check.as_mut() {
                    Self::apply_state(st2, opc, n, &|i: u64| f.rd(at + 4 * (1 + i)));
                }
                c += 4 * (n + 1);
            }
            if let Some(st2) = check.as_ref() {
                plan_check_states(&st, st2, s);
            }
            states.push(Box::new(st.clone()));
        }
        lap.mark(4);
        let mut workers = Vec::new();
        for i in 0..k.min(m) {
            match self.par_pool.worker(i, &self.name, self.compute) {
                Some(w) => workers.push(w),
                None => break,
            }
        }
        if workers.is_empty() {
            return;
        }
        self.drop_ahead();
        // What this queue's batch being recorded writes: the workers cannot
        // wait for it (`tune par_main_writes`).
        let mut main_writes = self.exec.as_ref().map(|e| e.unsubmitted_writes()).unwrap_or_default();
        if lap.0.is_some() && !main_writes.is_empty() {
            // (Profile: how much of them the forwarded copies cover.)
            let ranges: Vec<(u64, u64)> = main_writes.iter().collect();
            let left = super::forward::uncovered(&ranges);
            PLAN_MAIN_WRITES[0].fetch_add(1, Ordering::Relaxed);
            PLAN_MAIN_WRITES[1].fetch_add(left.as_ref().is_some_and(|v| v.is_empty()) as u64, Ordering::Relaxed);
            PLAN_MAIN_WRITES[2].fetch_add(ranges.len() as u64, Ordering::Relaxed);
            PLAN_MAIN_WRITES[3].fetch_add(left.map_or(ranges.len(), |v| v.len()) as u64, Ordering::Relaxed);
        }
        lap.mark(7);
        // Mode 3: the batch is submitted right after the hand-over (a
        // worker that reads its writes waits for that submission).
        let submit_after = super::tune::par_main_writes() == 3 && !main_writes.is_empty();
        if super::tune::par_main_writes() == 2 && !main_writes.is_empty() {
            self.submit_gpu("parallel run");
            main_writes = Default::default();
        }
        lap.mark(8);
        let queue = states.into_iter().enumerate().map(|(i, st)| (st, starts[i], if i + 1 < starts.len() { starts[i + 1] } else { run_end })).collect();
        let submitted = Arc::new(std::sync::atomic::AtomicBool::new(!submit_after));
        let mut run = super::par::Run::new(queue, Arc::new(main_writes), submitted.clone());
        run.timed = timed;
        let run = Arc::new(run);
        lap.mark(9);
        for w in &workers {
            w.give(run.clone());
        }
        self.par_pool.wake();
        if submit_after {
            self.submit_gpu("parallel run");
            submitted.store(true, Ordering::Release);
        }
        lap.mark(10);
        super::par::RUNS.fetch_add(1, Ordering::Relaxed);
        super::par::CHUNKS.fetch_add(starts.len() as u64, Ordering::Relaxed);
        self.par = Some(ParRun { starts, end: run_end, next: 0, run, waited_ns: 0, sync, sig, n, firsts });
        lap.mark(5);
        PLAN_PHASES[6].fetch_add(1, Ordering::Relaxed);
    }

    /// The main command processor reached the next chunk of the parallel
    /// run: its own work so far goes to the GPU, then the chunk's batches
    /// (its worker submits them once given the turn, or did when the chunk
    /// before ran to its end); execution continues with the chunk's end
    /// state from its end, or from the packet where the worker stopped.
    /// Returns where to continue.
    fn par_collect(&mut self) -> u64 {
        let (run, i, to, last) = {
            let pr = self.par.as_mut().unwrap();
            let i = pr.next;
            pr.next += 1;
            if pr.sync {
                self.fence_pending = true;
            }
            (pr.run.clone(), i, if i + 1 < pr.starts.len() { pr.starts[i + 1] } else { pr.end }, pr.next == pr.starts.len())
        };
        self.submit_gpu("par");
        run.grant(i);
        let tw = super::timeline::wait_start();
        let (done, waited) = run.collect(i);
        tw.end(&self.name, || "wait: parallel chunk".into());
        // (Before the run ends below.)
        let cost_key = self.par.as_ref().filter(|p| p.sig != 0 && !done.costs.is_empty()).and_then(|p| Some((p.sig, p.n, *p.firsts.get(i)?)));
        match self.par.as_mut() {
            Some(pr) if last => {
                super::par::note_main_wait(pr.waited_ns + waited);
                self.par = None;
            }
            Some(pr) => pr.waited_ns += waited,
            None => {}
        }
        if done.stopped.is_some() {
            super::par::note_stop(done.why);
        }
        if let Some((sig, n, first)) = cost_key {
            self.note_chunk_costs(sig, n, first, &done.costs);
        }
        // This queue's later work and CP effects follow the chunk's batches
        // (a label written after the run must wait for them).
        if let (Some(seq), Some(e)) = (done.last, self.exec.as_mut()) {
            e.depend_on(seq, false);
        }
        self.set_chunk_state(&done.state);
        match done.stopped {
            Some(s) => {
                super::par::STOPS.fetch_add(1, Ordering::Relaxed);
                s
            }
            None => to,
        }
    }

    /// Folds the per-command costs a chunk measured (its first command:
    /// `first`) into its run's costs (`tune par_cost`): smoothed; a chunk's
    /// first command (it carries the chunk's setup) only when unknown, as
    /// the mean of the others.
    fn note_chunk_costs(&mut self, sig: u64, n: usize, first: usize, costs: &[u32]) {
        let Some(rc) = self.run_costs.get_mut(&sig) else { return };
        if rc.costs.len() != n || costs.len() < 2 {
            return;
        }
        let mut sum = 0.0f64;
        for (j, &c) in costs.iter().enumerate().skip(1) {
            let Some(slot) = rc.costs.get_mut(first + j) else { break };
            let v = c as f32;
            *slot = if slot.is_finite() { 0.5 * *slot + 0.5 * v } else { v };
            sum += c as f64;
        }
        if let Some(slot) = rc.costs.get_mut(first) {
            if !slot.is_finite() {
                *slot = (sum / (costs.len() - 1) as f64) as f32;
            }
        }
    }

    /// Abandons the parallel run in progress (execution left it): the
    /// workers drop the batches of chunks whose turn did not come (none had:
    /// execution leaves a run only from the main command processor's own
    /// chunk or the rest of a stopped one); their chunks are executed here.
    fn par_cancel(&mut self) {
        let Some(pr) = self.par.take() else { return };
        pr.run.cancel();
        super::par::CANCELLED.fetch_add(1, Ordering::Relaxed);
    }

    /// The executing command's stage preparations made ahead ([vertex,
    /// pixel] for draws, [compute, -] for dispatches), once.
    pub fn take_ahead(&self) -> [Option<Arc<super::ahead::Slot>>; 2] {
        std::mem::take(&mut *self.cur_ahead.borrow_mut())
    }

    /// Drops the preparations made ahead (execution took another path).
    fn drop_ahead(&mut self) {
        for c in self.ahead.drain(..) {
            for s in c.stages.iter().flatten() {
                s.cancel();
            }
        }
    }

    /// The commands after a label wait that blocked (`tune wait_ahead`):
    /// from `from` with this command processor's registers, through
    /// register writes, flushes, cache actions, end-of-pipe events, CP
    /// writes and further label waits (what a stage reads is checked again
    /// when it is used, `ahead::take`), up to any other control packet, the
    /// buffer's end or WAIT_SCAN_MAX commands. A synchronization-free run
    /// long enough for the parallel workers ends it before its first
    /// command (they prepare their own). The stages are prepared by this
    /// command processor while it waits (`wait`), not handed to workers.
    fn scan_through(&mut self, f: &Fetched, from: u64, end: u64) -> Vec<AheadCmd> {
        let (cmds, why, ns) = self.scan_commands(f, from, end, false);
        super::ahead::note_wait_scan(cmds.len(), why, ns);
        cmds
    }

    /// A dispatch's pipeline is missing (its program's first use with this
    /// state): the commands after it in its buffer are scanned as at a label
    /// wait (`scan_commands`, through synchronization and long runs) and
    /// their compute stages handed to the compile workers (`gpu::compile`).
    pub fn prefetch_compiles(&self) {
        let Some((base, from, end)) = self.compile_ctx.take() else { return };
        // Read in place (the scan does not copy the buffer).
        let f = Fetched { base, words: Vec::new(), live: false };
        let (cmds, why, ns) = self.scan_commands(&f, from, end, true);
        super::compile::note_scan_end(why, cmds.len());
        let mut slots: Vec<Arc<super::ahead::Slot>> = Vec::new();
        let mut draws = Vec::new();
        for c in cmds {
            let [a, b] = c.stages;
            match (c.gfx, a) {
                (Some(g), Some(vs)) => draws.push((vs, b, *g)),
                (None, Some(s)) => slots.push(s),
                _ => {}
            }
        }
        super::compile::submit(slots, draws, ns);
    }

    /// The scan of `scan_through` (`whole`: also through runs the parallel
    /// workers would record): the commands, why it ended, its nanoseconds.
    fn scan_commands(&self, f: &Fetched, from: u64, end: u64, whole: bool) -> (Vec<AheadCmd>, usize, u64) {
        use super::ahead::{END_ARGS, END_BUFFER, END_HEADER, END_INVALID, END_MAX, END_PREDICATED};
        const WAIT_SCAN_MAX: usize = 1024;
        let t0 = Instant::now();
        // Runs this command processor would split among parallel workers.
        let par_min = if !whole && f.words.is_empty() && !super::capture::capturing() && (if self.compute { super::tune::par_compute() > 0 } else { super::par::workers() > 0 }) {
            (if self.compute { super::tune::par_min_compute() as usize } else { super::par::min_run() }).max(48)
        } else {
            usize::MAX
        };
        let mut why = END_BUFFER;
        let (mut sh, mut ctx) = std::mem::take(&mut *self.scan_regs.borrow_mut());
        sh.clear();
        sh.extend_from_slice(&self.regs.sh);
        ctx.clear();
        ctx.extend_from_slice(&self.regs.ctx);
        let mut draw_base = self.draw_indirect_base;
        let mut cmds: Vec<AheadCmd> = Vec::new();
        // Draws' baked pipeline state for compiles ahead (`whole` scans), and
        // VGT_PRIMITIVE_TYPE as the scanned writes leave it.
        let gfx_state = whole && super::compile::gfx_at_first_uses();
        let mut prim = self.reg(Space::Uc, 0x242);
        // The synchronization-free run being scanned: its first command's
        // index and its commands.
        let (mut run_first, mut run_len) = (0usize, 0usize);
        let mut cur = from;
        while cur < end {
            if cmds.len() >= WAIT_SCAN_MAX {
                why = END_MAX;
                break;
            }
            if !valid(cur, 4) {
                why = END_INVALID;
                break;
            }
            let h = f.rd(cur);
            match h >> 30 {
                3 => {}
                2 => {
                    cur += 4;
                    continue;
                }
                _ => {
                    why = END_HEADER;
                    break;
                }
            }
            let count = (h >> 16) & 0x3FFF;
            if count == 0x3FFF {
                cur += 4;
                continue;
            }
            // Predicated packets may be skipped at execution.
            if h & 1 != 0 {
                why = END_PREDICATED;
                break;
            }
            let opcode = (h >> 8) & 0xFF;
            let n = count as u64 + 1;
            let next = cur + 4 * (n + 1);
            if next > end || !valid(cur, 4 * (n + 1)) {
                why = END_BUFFER;
                break;
            }
            let at = cur;
            let p = |i: u64| f.rd(at + 4 * (1 + i));
            let set = |v: &mut Vec<u32>, first: u32| {
                for i in 1..n {
                    if let Some(r) = v.get_mut((first + (i - 1) as u32) as usize) {
                        *r = p(i);
                    }
                }
            };
            let mut command: Option<[Option<super::shaders::StageRegs>; 2]> = None;
            let mut gfx: Option<Box<super::exec::GfxRegs>> = None;
            let mut wave64 = true;
            let mut sync = false;
            match opcode {
                op::NOP => {
                    // Private packets (flips, branches, draws, state) end
                    // the scan; debug markers do not.
                    if n >= 2 && p(0) == DS_MAGIC && !matches!(p(1), pop::PUSH_MARKER | pop::SET_MARKER | pop::POP_MARKER) {
                        why = 0x100 + (p(1) as usize).min(0x2F);
                        break;
                    }
                }
                // GPU-side synchronization, CP writes and label waits:
                // passed (they end synchronization-free runs).
                op::EVENT_WRITE | op::ACQUIRE_MEM | op::RELEASE_MEM | op::WRITE_DATA | op::WAIT_REG_MEM | op::WAIT_REG_MEM64 => sync = true,
                op::SET_UCONFIG_REG | op::SET_UCONFIG_REG_INDEX => {
                    let first = p(0) & 0xFFFF;
                    if first <= 0x242 && 0x242 < first + (n - 1) as u32 {
                        prim = p(1 + (0x242 - first) as u64);
                    }
                }
                op::LOAD_UCONFIG_REG | op::INDEX_TYPE | op::INDEX_BASE | op::INDEX_BUFFER_SIZE | op::NUM_INSTANCES | op::CONTEXT_CONTROL => {}
                op::SET_CONTEXT_REG | op::SET_CONTEXT_REG_INDEX => set(&mut ctx, p(0) & 0xFFFF),
                op::SET_SH_REG | op::SET_SH_REG_INDEX => set(&mut sh, p(0) & 0xFFFF),
                op::LOAD_CONTEXT_REG_INDEX | op::LOAD_SH_REG_INDEX => {
                    // As `load_pairs`: memory holds {offset, value} pairs.
                    let (a, count) = (addr64(p(0), p(1)), p(2) as u64);
                    if count > 0 && !valid(a, 8 * count) {
                        why = END_ARGS;
                        break;
                    }
                    let v = if opcode == op::LOAD_CONTEXT_REG_INDEX { &mut ctx } else { &mut sh };
                    for i in 0..count {
                        let (o, val) = unsafe { (rd32(a + 8 * i), rd32(a + 8 * i + 4)) };
                        if let Some(r) = v.get_mut((o & 0xFFFF) as usize) {
                            *r = val;
                        }
                    }
                }
                op::SET_BASE => {
                    if p(0) != 2 {
                        draw_base = addr64(p(1), p(2));
                    }
                }
                op::DRAW_INDEX_2 | op::DRAW_INDEX_AUTO | op::DRAW_INDEX_OFFSET_2 | op::DRAW_INDIRECT | op::DRAW_INDEX_INDIRECT => {
                    if matches!(opcode, op::DRAW_INDIRECT | op::DRAW_INDEX_INDIRECT) {
                        // As `packet` does: the base vertex and start instance
                        // go to the user SGPRs the packet names.
                        let a = draw_base + p(0) as u64;
                        let (bytes, k) = if opcode == op::DRAW_INDEX_INDIRECT { (20, 3) } else { (16, 2) };
                        if !valid(a, bytes) {
                            why = END_ARGS;
                            break;
                        }
                        let (bv, si) = unsafe { (rd32(a + 4 * k), rd32(a + 4 * (k + 1))) };
                        let (vtx_loc, inst_loc) = (p(1) & 0xFFFF, p(1) >> 16);
                        let inst_loc = if inst_loc != 0 {
                            inst_loc
                        } else {
                            let ngg = super::shaders::ngg_addr_of(&ScanRegs { sh: &sh, ctx: &ctx });
                            crate::hle::libs::agc::instance_offset_slot(ngg).map_or(0, |s| 0x8C + s)
                        };
                        if vtx_loc != 0 {
                            if let Some(r) = sh.get_mut(vtx_loc as usize) {
                                *r = bv;
                            }
                        }
                        if inst_loc != 0 {
                            if let Some(r) = sh.get_mut(inst_loc as usize) {
                                *r = si;
                            }
                        }
                    }
                    let view = ScanRegs { sh: &sh, ctx: &ctx };
                    let ps = super::shaders::ps_addr_of(&view);
                    if gfx_state {
                        let c = &ctx;
                        gfx = Some(Box::new(super::exec::GfxRegs::capture(&|r: u32| c.get(((r - 0x28000) / 4) as usize).copied().unwrap_or(0), prim)));
                    }
                    command = Some([
                        Some(super::shaders::StageRegs::capture(super::shaders::StageKind::Vertex, &view)),
                        (ps != 0).then(|| super::shaders::StageRegs::capture(super::shaders::StageKind::Pixel, &view)),
                    ]);
                }
                op::DISPATCH_DIRECT | op::DISPATCH_INDIRECT => {
                    let initiator = if opcode == op::DISPATCH_DIRECT { p(3) } else if n >= 3 { p(2) } else { p(1) };
                    wave64 = initiator & (1 << 15) == 0;
                    let view = ScanRegs { sh: &sh, ctx: &ctx };
                    command = Some([Some(super::shaders::StageRegs::capture(super::shaders::StageKind::Compute, &view)), None]);
                }
                _ => {
                    why = opcode as usize;
                    break;
                }
            }
            if sync {
                if run_len >= par_min {
                    break;
                }
                run_len = 0;
            }
            if let Some(stages) = command {
                if run_len == 0 {
                    run_first = cmds.len();
                }
                run_len += 1;
                let stages = stages.map(|r| r.map(|r| super::ahead::Slot::new(r, wave64)));
                cmds.push(AheadCmd { at, stages, gfx });
            }
            cur = next;
        }
        // A run the workers will record: its commands are theirs.
        if run_len >= par_min {
            cmds.truncate(run_first);
        }
        *self.scan_regs.borrow_mut() = (sh, ctx);
        (cmds, why, t0.elapsed().as_nanos() as u64)
    }

    /// Scans the packets from `from` to the end of the synchronization-free
    /// run (register writes, draws and dispatches only; any other packet
    /// ends it) and hands the stages of its draws and dispatches to the
    /// preparation workers with the registers they will read. Registers
    /// are taken from a copy of this command processor's files with the
    /// scanned writes applied, indirect draw arguments as memory holds them
    /// now: whatever differs at execution is caught there (`ahead::take`).
    fn scan_ahead(&mut self, f: &Fetched, from: u64, end: u64) {
        use super::ahead::{END_ARGS, END_BUFFER, END_HEADER, END_INVALID, END_MAX, END_PREDICATED};
        let t0 = Instant::now();
        // Most scans find a synchronization packet before any command: a
        // look at the packet headers first spares them the copies of the
        // register files below (~3 us each).
        {
            let mut c = from;
            let found = loop {
                if c >= end || !valid(c, 4) {
                    break false;
                }
                let h = f.rd(c);
                match h >> 30 {
                    3 => {}
                    2 => {
                        c += 4;
                        continue;
                    }
                    _ => break false,
                }
                let count = (h >> 16) & 0x3FFF;
                if count == 0x3FFF {
                    c += 4;
                    continue;
                }
                let op = (h >> 8) & 0xFF;
                if h & 1 != 0 {
                    break false;
                }
                match op {
                    op::DRAW_INDEX_2 | op::DRAW_INDEX_AUTO | op::DRAW_INDEX_OFFSET_2 | op::DRAW_INDIRECT | op::DRAW_INDEX_INDIRECT | op::DISPATCH_DIRECT | op::DISPATCH_INDIRECT => break true,
                    op::NOP => {
                        if count >= 1 && valid(c + 4, 8) && f.rd(c + 4) == DS_MAGIC && !matches!(f.rd(c + 8), pop::PUSH_MARKER | pop::SET_MARKER | pop::POP_MARKER) {
                            super::ahead::note_scan(0x100 + (f.rd(c + 8) as usize).min(0x2F), t0.elapsed().as_nanos() as u64);
                            return;
                        }
                    }
                    op::SET_CONTEXT_REG
                    | op::SET_CONTEXT_REG_INDEX
                    | op::SET_SH_REG
                    | op::SET_SH_REG_INDEX
                    | op::SET_UCONFIG_REG
                    | op::SET_UCONFIG_REG_INDEX
                    | op::LOAD_CONTEXT_REG_INDEX
                    | op::LOAD_SH_REG_INDEX
                    | op::LOAD_UCONFIG_REG
                    | op::INDEX_TYPE
                    | op::INDEX_BASE
                    | op::INDEX_BUFFER_SIZE
                    | op::NUM_INSTANCES
                    | op::CONTEXT_CONTROL
                    | op::SET_BASE => {}
                    _ => {
                        super::ahead::note_scan(op as usize, t0.elapsed().as_nanos() as u64);
                        return;
                    }
                }
                c += 4 * (count as u64 + 2);
            };
            if !found {
                super::ahead::note_scan(END_BUFFER, t0.elapsed().as_nanos() as u64);
                return;
            }
        }
        let mut why = END_BUFFER;
        let (mut sh, mut ctx) = std::mem::take(&mut *self.scan_regs.borrow_mut());
        sh.clear();
        sh.extend_from_slice(&self.regs.sh);
        ctx.clear();
        ctx.extend_from_slice(&self.regs.ctx);
        let mut draw_base = self.draw_indirect_base;
        let mut cur = from;
        let mut slots: Vec<Arc<super::ahead::Slot>> = Vec::new();
        while cur < end {
            if self.ahead.len() >= AHEAD_MAX {
                why = END_MAX;
                break;
            }
            if !valid(cur, 4) {
                why = END_INVALID;
                break;
            }
            let h = f.rd(cur);
            match h >> 30 {
                3 => {}
                2 => {
                    cur += 4;
                    continue;
                }
                _ => {
                    why = END_HEADER;
                    break;
                }
            }
            let count = (h >> 16) & 0x3FFF;
            if count == 0x3FFF {
                cur += 4;
                continue;
            }
            // Predicated packets may be skipped at execution.
            if h & 1 != 0 {
                why = END_PREDICATED;
                break;
            }
            let opcode = (h >> 8) & 0xFF;
            let n = count as u64 + 1;
            let next = cur + 4 * (n + 1);
            if next > end || !valid(cur, 4 * (n + 1)) {
                why = END_BUFFER;
                break;
            }
            let at = cur;
            let p = |i: u64| f.rd(at + 4 * (1 + i));
            let set = |v: &mut Vec<u32>, first: u32| {
                for i in 1..n {
                    if let Some(r) = v.get_mut((first + (i - 1) as u32) as usize) {
                        *r = p(i);
                    }
                }
            };
            let mut command: Option<[Option<super::shaders::StageRegs>; 2]> = None;
            let mut wave64 = true;
            match opcode {
                op::NOP => {
                    // Private packets (flips, branches, draws, state) end
                    // the run; debug markers do not.
                    if n >= 2 && p(0) == DS_MAGIC && !matches!(p(1), pop::PUSH_MARKER | pop::SET_MARKER | pop::POP_MARKER) {
                        why = 0x100 + (p(1) as usize).min(0x2F);
                        break;
                    }
                }
                // Uconfig registers (primitive and index type, ...) are not
                // read by stage preparations.
                op::SET_UCONFIG_REG | op::SET_UCONFIG_REG_INDEX | op::LOAD_UCONFIG_REG | op::INDEX_TYPE | op::INDEX_BASE | op::INDEX_BUFFER_SIZE | op::NUM_INSTANCES | op::CONTEXT_CONTROL => {}
                op::SET_CONTEXT_REG | op::SET_CONTEXT_REG_INDEX => set(&mut ctx, p(0) & 0xFFFF),
                op::SET_SH_REG | op::SET_SH_REG_INDEX => set(&mut sh, p(0) & 0xFFFF),
                op::LOAD_CONTEXT_REG_INDEX | op::LOAD_SH_REG_INDEX => {
                    // As `load_pairs`: memory holds {offset, value} pairs.
                    let (a, count) = (addr64(p(0), p(1)), p(2) as u64);
                    if count > 0 && !valid(a, 8 * count) {
                        why = END_ARGS;
                        break;
                    }
                    let v = if opcode == op::LOAD_CONTEXT_REG_INDEX { &mut ctx } else { &mut sh };
                    for i in 0..count {
                        let (o, val) = unsafe { (rd32(a + 8 * i), rd32(a + 8 * i + 4)) };
                        if let Some(r) = v.get_mut((o & 0xFFFF) as usize) {
                            *r = val;
                        }
                    }
                }
                op::SET_BASE => {
                    if p(0) != 2 {
                        draw_base = addr64(p(1), p(2));
                    }
                }
                op::DRAW_INDEX_2 | op::DRAW_INDEX_AUTO | op::DRAW_INDEX_OFFSET_2 | op::DRAW_INDIRECT | op::DRAW_INDEX_INDIRECT => {
                    if matches!(opcode, op::DRAW_INDIRECT | op::DRAW_INDEX_INDIRECT) {
                        // As `packet` does: the base vertex and start instance
                        // go to the user SGPRs the packet names.
                        let a = draw_base + p(0) as u64;
                        let (bytes, k) = if opcode == op::DRAW_INDEX_INDIRECT { (20, 3) } else { (16, 2) };
                        if !valid(a, bytes) {
                            why = END_ARGS;
                            break;
                        }
                        let (bv, si) = unsafe { (rd32(a + 4 * k), rd32(a + 4 * (k + 1))) };
                        let (vtx_loc, inst_loc) = (p(1) & 0xFFFF, p(1) >> 16);
                        let inst_loc = if inst_loc != 0 {
                            inst_loc
                        } else {
                            let ngg = super::shaders::ngg_addr_of(&ScanRegs { sh: &sh, ctx: &ctx });
                            crate::hle::libs::agc::instance_offset_slot(ngg).map_or(0, |s| 0x8C + s)
                        };
                        if vtx_loc != 0 {
                            if let Some(r) = sh.get_mut(vtx_loc as usize) {
                                *r = bv;
                            }
                        }
                        if inst_loc != 0 {
                            if let Some(r) = sh.get_mut(inst_loc as usize) {
                                *r = si;
                            }
                        }
                    }
                    let view = ScanRegs { sh: &sh, ctx: &ctx };
                    let ps = super::shaders::ps_addr_of(&view);
                    command = Some([
                        Some(super::shaders::StageRegs::capture(super::shaders::StageKind::Vertex, &view)),
                        (ps != 0).then(|| super::shaders::StageRegs::capture(super::shaders::StageKind::Pixel, &view)),
                    ]);
                }
                op::DISPATCH_DIRECT | op::DISPATCH_INDIRECT => {
                    let initiator = if opcode == op::DISPATCH_DIRECT { p(3) } else if n >= 3 { p(2) } else { p(1) };
                    wave64 = initiator & (1 << 15) == 0;
                    let view = ScanRegs { sh: &sh, ctx: &ctx };
                    command = Some([Some(super::shaders::StageRegs::capture(super::shaders::StageKind::Compute, &view)), None]);
                }
                _ => {
                    why = opcode as usize;
                    break;
                }
            }
            if let Some(stages) = command {
                let stages = stages.map(|r| r.map(|r| super::ahead::Slot::new(r, wave64)));
                slots.extend(stages.iter().flatten().cloned());
                self.ahead.push_back(AheadCmd { at, stages, gfx: None });
            }
            cur = next;
        }
        *self.scan_regs.borrow_mut() = (sh, ctx);
        super::ahead::submit(&slots);
        super::ahead::note_scan(why, t0.elapsed().as_nanos() as u64);
    }

    /// Completes recorded GPU work (at synchronization points: the CP waits
    /// for shader work only where the packet stream asks it to).
    fn sync(&mut self) {
        self.fence_pending = false;
        if let Some(e) = self.exec.as_mut() {
            if e.busy() {
                let k = (self.cur_opcode as usize).min(255);
                SYNC_BY_OPCODE[k].fetch_add(1, Ordering::Relaxed);
            }
            e.flush();
        }
    }

    /// Runs a CP memory effect now, or after this queue's outstanding GPU
    /// work and earlier deferred effects. `end_of_pipe` effects (RELEASE_MEM)
    /// always follow the GPU work; other CP writes follow it only after a
    /// partial flush or acquire (otherwise the CP does not wait for shaders),
    /// and later GPU work then waits for them (it may read what they write).
    /// Returns the batch sequence a deferred effect runs after.
    /// A memory write that would be deferred behind earlier GPU work, done by
    /// the GPU in order with it (`Exec::gpu_cp_write`); false: take the CPU
    /// path (switch off, nothing outstanding, or earlier CPU-side effects
    /// still pending, which it must not overtake).
    fn gpu_cp_write(&mut self, a: u64, data: &[u32], label: Option<(u64, u8)>, end_of_pipe: bool) -> bool {
        if !super::tune::gpu_cp_writes() {
            return false;
        }
        super::replay::note_cp_write(a, 4 * data.len() as u64);
        let needs_prior = end_of_pipe || self.fence_pending;
        let Some(e) = self.exec.as_mut() else { return false };
        // Nothing outstanding to follow: the CPU performs the write now.
        let refused = if !(e.busy() && (needs_prior || e.deferred_pending())) {
            Some(0)
        } else if !e.gpu_cp_write(a, data, label, end_of_pipe) {
            Some(2)
        } else {
            None
        };
        if let Some(k) = refused {
            super::exec::GPU_CP_REFUSED[k].fetch_add(1, Ordering::Relaxed);
            return false;
        }
        self.fence_pending = false;
        // Diagnostics (no latching: the value is not in memory yet).
        let mut g = LAST_WRITE.lock();
        let m = g.get_or_insert_with(Default::default);
        if m.len() > 1 << 16 {
            m.clear();
        }
        m.insert(a & !7, (self.name.clone(), label.map(|l| l.0).unwrap_or(data[0] as u64), self.cur_packet, a));
        true
    }

    fn effect(&mut self, end_of_pipe: bool, write: Option<(u64, u64)>, f: Box<dyn FnOnce() + Send>) -> Option<u64> {
        if let Some((a, n)) = write {
            super::replay::note_cp_write(a, n);
            super::forward::written(a, n);
        }
        // Mirror mode: the pages it changes are copied into the mirror at
        // the next placement over them (`resident::cp_wrote`).
        let f = match write {
            Some((a, n)) if super::resident::mirror_mode() => Box::new(move || {
                f();
                super::resident::cp_wrote(a, n);
            }) as Box<dyn FnOnce() + Send>,
            _ => f,
        };
        let needs_prior = end_of_pipe || self.fence_pending;
        match self.exec.as_mut() {
            Some(e) if e.busy() && (needs_prior || e.deferred_pending()) => {
                let k = (self.cur_opcode as usize).min(255);
                let why = if end_of_pipe { 0 } else if self.fence_pending { 1 } else { 2 };
                DEFERRED_BY_OPCODE[k][why].fetch_add(1, Ordering::Relaxed);
                self.fence_pending = false;
                let f = match write {
                    Some((a, n)) if super::tune::race_checks() => {
                        let (q, at, op) = (self.name.clone(), self.cur_packet, self.cur_opcode);
                        Box::new(move || {
                            f();
                            cpw_note(&q, a, n, at, op);
                        }) as Box<dyn FnOnce() + Send>
                    }
                    _ => f,
                };
                e.after_gpu(f, write, !end_of_pipe)
            }
            _ => {
                if needs_prior {
                    self.fence_pending = false;
                }
                f();
                if let Some((a, n)) = write {
                    cpw_note(&self.name, a, n, self.cur_packet, self.cur_opcode);
                }
                None
            }
        }
    }

    /// Makes memory the CP reads now current: completes this queue's GPU
    /// work (and deferred CP writes) that writes into `reads`. Indirect
    /// arguments are typically written by a culling dispatch shortly before;
    /// the draws after the first then find nothing pending.
    fn sync_reads(&mut self, reads: &[(u64, u64)]) {
        if let Some(e) = self.exec.as_mut() {
            if e.flush_if_writes(reads) {
                let k = (self.cur_opcode as usize).min(255);
                SYNC_BY_OPCODE[k].fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Completes all GPU work recorded so far and its deferred effects.
    pub fn finish(&mut self) {
        self.sync();
    }

    /// Hands recorded GPU work to the device without waiting.
    fn submit_gpu(&mut self, why: &'static str) {
        if let Some(e) = self.exec.as_mut() {
            e.submit_for(why);
        }
    }

    /// Executes a dispatch (when GPU execution is enabled).
    /// An indirect dispatch with the GPU reading its arguments (see
    /// `Exec::dispatch_gpu_indirect`); false: not taken.
    fn dispatch_gpu_indirect(&mut self, args: u64, initiator: u32) -> bool {
        if !super::exec::enabled() || self.exec_failed {
            return false;
        }
        let Some(mut e) = self.exec.take() else { return false };
        let taken = e.dispatch_gpu_indirect(self, args, initiator);
        self.exec = Some(e);
        taken
    }

    fn dispatch(&mut self, groups: [u32; 3], initiator: u32) {
        if !super::exec::enabled() || self.exec_failed {
            self.note("dispatches are decoded but not executed (set DS_GPU=1 to run them)");
            return;
        }
        if self.exec.is_none() {
            match super::exec::Exec::new(&self.name) {
                Ok(e) => self.exec = Some(Box::new(e)),
                Err(e) => {
                    crate::error!("gpu[{}]: no executor: {e}", self.name);
                    self.exec_failed = true;
                    return;
                }
            }
        }
        let mut e = self.exec.take().unwrap();
        super::xwait::note_command(&self.name);
        e.dispatch(self, groups, initiator);
        self.exec = Some(e);
        self.split_if_due();
    }

    /// Resets the context registers to their clear-state values: zero, except
    /// the registers with a non-zero Agc register default (assumed to be the
    /// hardware clear state; e.g. CB_COLOR_CONTROL = 0x00CC0010, scissor
    /// bottom-right corners 0x40004000: the title leaves these unprogrammed
    /// for ordinary draws).
    fn clear_context(&mut self) {
        static DEFAULTS: std::sync::OnceLock<Vec<(u32, u32)>> = std::sync::OnceLock::new();
        let d = DEFAULTS.get_or_init(|| {
            let mut v = Vec::new();
            for def in crate::hle::libs::agc_defaults::PUBLIC.iter().chain(crate::hle::libs::agc_defaults::INTERNAL) {
                if matches!(def.table(), 0 | 3) {
                    v.extend(def.regs.iter().filter(|r| r.1 != 0).copied());
                }
            }
            v
        });
        self.regs.ctx.fill(0);
        self.ctx_dirty = true;
        for &(i, val) in d {
            self.set_reg(Space::Ctx, i, val);
        }
    }

    /// Diagnostics (`tune split_every`): ends the graphics queue's batch
    /// after every N commands recorded serially.
    fn split_if_due(&mut self) {
        let n = super::tune::split_every();
        if n == 0 || self.compute || self.par.is_some() {
            return;
        }
        if let Some(e) = self.exec.as_mut() {
            if e.serial_batch_len() as u64 >= n {
                SPLITS.fetch_add(1, Ordering::Relaxed);
                e.submit_for("split (diagnostics)");
            }
        }
    }

    /// Executes a draw (when GPU execution is enabled).
    fn draw(&mut self, args: super::exec::DrawArgs) {
        if !super::exec::enabled() || self.exec_failed {
            self.note("draws are decoded but not rendered (set DS_GPU=1 to render them)");
            return;
        }
        if self.exec.is_none() {
            match super::exec::Exec::new(&self.name) {
                Ok(e) => self.exec = Some(Box::new(e)),
                Err(e) => {
                    crate::error!("gpu[{}]: no executor: {e}", self.name);
                    self.exec_failed = true;
                    return;
                }
            }
        }
        // Measurement: draws with the last draw's context registers (and
        // programs and primitive type), as a draw that took its targets,
        // dynamic state and pipeline from the last one could.
        let progs = (super::shaders::ngg_addr(self), super::shaders::ps_addr(self), self.reg(Space::Uc, 0x242));
        DRAW_STATE[0].fetch_add(1, Ordering::Relaxed);
        let same = !self.ctx_dirty && progs == self.last_draw_progs;
        if !self.ctx_dirty {
            DRAW_STATE[1].fetch_add(1, Ordering::Relaxed);
            if same {
                DRAW_STATE[2].fetch_add(1, Ordering::Relaxed);
            }
        }
        // Runs of such draws (batching candidates), by length.
        if same {
            self.same_run += 1;
        } else {
            note_draw_run(self.same_run);
            self.same_run = 1;
        }
        self.ctx_dirty = false;
        self.last_draw_progs = progs;
        super::xwait::note_command(&self.name);
        let mut e = self.exec.take().unwrap();
        self.last_draw = e.draw(self, args);
        self.exec = Some(e);
        self.split_if_due();
    }

    /// An indexed indirect draw with the GPU reading its arguments (`tune
    /// draw_args_gpu`): arguments and indices in video-memory homes, the
    /// start instance handed to a user SGPR (none for the base vertex), a
    /// vertex program that needs neither on the CPU. False: not taken (the
    /// arguments are read here). Homes for arguments that GPU work writes
    /// through VRAM shadows are made when parallel runs are planned
    /// (`par_plan`).
    fn draw_gpu_args(&mut self, opcode: u32, p: &dyn Fn(u64) -> u32) -> bool {
        const GS_USER_DATA: u32 = 0x8C;
        if opcode != op::DRAW_INDEX_INDIRECT || !super::tune::draw_args_gpu() || !super::exec::enabled() || self.exec_failed || super::capture::capturing() || !super::vk::try_ctx().is_some_and(|c| c.draw_parameters) {
            return false;
        }
        let a = self.draw_indirect_base + p(0) as u64;
        let in_home = super::homes::contains(a, 20);
        // Arguments in a home read on the CPU instead (after a write-back of
        // their record), counted by why.
        let refuse = |k: usize| {
            if in_home {
                CPU_ARG_REFUSED[k].fetch_add(1, Ordering::Relaxed);
            }
            false
        };
        let (vtx_loc, inst_loc) = (p(1) & 0xFFFF, p(1) >> 16);
        let inst_loc = if inst_loc != 0 { inst_loc } else { self.instance_offset_loc() };
        if vtx_loc != 0 {
            return refuse(0);
        }
        if !(GS_USER_DATA..GS_USER_DATA + 32).contains(&inst_loc) {
            return refuse(1);
        }
        let Some(sz) = self.index_size() else { return refuse(2) };
        let index_bytes = self.index_count as u64 * sz as u64;
        if index_bytes == 0 || !super::homes::contains(self.index_base, index_bytes) {
            return refuse(3);
        }
        if super::shaders::cpu_args_program(super::shaders::ngg_addr(self)) {
            return refuse(4);
        }
        if !in_home {
            // Read here this time; their buffer may get a home (`par_plan`).
            if valid(a, 20) {
                super::homes::vote_args(a, 20);
            }
            return false;
        }
        // The draw before was like this one, nothing came between them: one
        // more indirect draw on the state it left (`tune draw_repeat`).
        if std::mem::take(&mut self.repeat_ok) && super::tune::draw_repeat() {
            let index_base = self.index_base;
            if let Some(e) = self.exec.as_mut() {
                if e.draw_repeat(a, (index_base, sz), index_bytes) {
                    // (Stage preparations made ahead for it are not used.)
                    let _ = self.take_ahead();
                    self.last_draw = std::borrow::Cow::Borrowed("drawn (repeating the draw before)");
                    self.repeat_ok = true;
                    self.split_if_due();
                    return true;
                }
            }
        }
        let g = super::exec::GpuArgs { addr: a, inst_slot: (inst_loc - GS_USER_DATA) as u16, index_bytes };
        self.draw(super::exec::DrawArgs { count: 0, instances: 0, index: Some((self.index_base, sz)), base_vertex: None, first_instance: 0, gpu_args: Some(g), base_inst: None });
        if self.last_draw == super::exec::NEEDS_CPU_ARGS {
            CPU_ARG_FALLBACKS.fetch_add(1, Ordering::Relaxed);
            return false;
        }
        self.repeat_ok = super::tune::draw_repeat() && self.exec.as_ref().is_some_and(|e| e.repeat_ready());
        true
    }

    /// Whether a packet leaves the draw state as the last draw had it (a
    /// repeat of that draw stays possible, `tune draw_repeat`): indexed
    /// indirect draws, plain NOPs, and register writes and loads and index
    /// or base settings that only write the values already there.
    fn keeps_repeat(&self, opcode: u32, n: u64, p: &dyn Fn(u64) -> u32) -> bool {
        let same = |space: Space| {
            let first = p(0) & 0xFFFF;
            (1..n).all(|i| self.reg(space, first + (i - 1) as u32) == p(i))
        };
        // {offset, value} pairs in memory, as `load_pairs` reads them.
        let same_pairs = |space: Space| {
            let (a, k) = (addr64(p(0), p(1)), p(2) as u64);
            k <= 4096 && valid(a, 8 * k.max(1)) && (0..k).all(|i| unsafe { self.reg(space, rd32(a + 8 * i) & 0xFFFF) == rd32(a + 8 * i + 4) })
        };
        match opcode {
            op::DRAW_INDEX_INDIRECT => true,
            op::NOP => !(n >= 2 && p(0) == DS_MAGIC),
            op::SET_CONTEXT_REG => same(Space::Ctx),
            op::SET_SH_REG => same(Space::Sh),
            op::SET_UCONFIG_REG => same(Space::Uc),
            op::LOAD_CONTEXT_REG_INDEX => same_pairs(Space::Ctx),
            op::LOAD_SH_REG_INDEX => same_pairs(Space::Sh),
            op::LOAD_UCONFIG_REG => same_pairs(Space::Uc),
            op::INDEX_BASE => self.index_base == addr64(p(0), p(1)),
            op::INDEX_BUFFER_SIZE => self.index_count == p(0),
            op::NUM_INSTANCES => self.num_instances == p(0).max(1),
            op::SET_BASE => (if p(0) == 2 { self.dispatch_indirect_base } else { self.draw_indirect_base }) == addr64(p(1), p(2)),
            _ => false,
        }
    }

    /// Whether an indexed indirect draw whose arguments the CPU reads hands
    /// the start instance over as Vulkan's first instance (`tune
    /// cpu_args_base`): no base-vertex word, a start-instance word among the
    /// vertex stage's user data, draw parameters, a vertex program not known
    /// to need the word on the CPU; not while a frame is captured.
    fn cpu_args_base_ok(&self, vtx_loc: u32, inst_loc: u32) -> bool {
        const GS_USER_DATA: u32 = 0x8C;
        super::tune::cpu_args_base()
            && vtx_loc == 0
            && (GS_USER_DATA..GS_USER_DATA + 32).contains(&inst_loc)
            && !self.exec_failed
            && !super::capture::capturing()
            && super::exec::enabled()
            && super::vk::try_ctx().is_some_and(|c| c.draw_parameters)
            && !super::shaders::cpu_args_program(super::shaders::ngg_addr(self))
    }

    /// An indexed indirect draw whose arguments the CPU read (`w`): the
    /// start instance as Vulkan's first instance, the vertex stage taking
    /// its user-data word from gl_BaseInstance (`tune cpu_args_base`); a
    /// repeat of the draw before when it can be (`tune draw_repeat`), else
    /// recorded in full; a vertex stage that needs the word on the CPU gets
    /// it in its register as before. The outcome.
    fn draw_cpu_args_base(&mut self, w: [u32; 5], sz: u32, inst_loc: u32) -> String {
        const GS_USER_DATA: u32 = 0x8C;
        let index = (self.index_base + w[2] as u64 * sz as u64, sz);
        if std::mem::take(&mut self.repeat_ok) && super::tune::draw_repeat() {
            if let Some(e) = self.exec.as_mut() {
                if e.draw_repeat_cpu(index, w[0], w[1], w[3] as i32, w[4]) {
                    // (Stage preparations made ahead for it are not used.)
                    let _ = self.take_ahead();
                    self.repeat_ok = true;
                    self.split_if_due();
                    return "drawn (repeating the draw before)".into();
                }
            }
        }
        self.draw(super::exec::DrawArgs { count: w[0], instances: w[1], index: Some(index), base_vertex: Some(w[3] as i32), first_instance: w[4], gpu_args: None, base_inst: Some((inst_loc - GS_USER_DATA) as u16) });
        if self.last_draw == super::exec::NEEDS_CPU_ARGS {
            // The start instance in its register, as before.
            self.set_reg(Space::Sh, inst_loc, w[4]);
            self.draw(super::exec::DrawArgs { count: w[0], instances: w[1], index: Some(index), base_vertex: Some(w[3] as i32), first_instance: 0, gpu_args: None, base_inst: None });
            return std::mem::take(&mut self.last_draw).into_owned();
        }
        self.repeat_ok = super::tune::draw_repeat() && self.exec.as_ref().is_some_and(|e| e.repeat_ready());
        std::mem::take(&mut self.last_draw).into_owned()
    }

    /// Whether this (main) command processor is recording its own chunk of a
    /// parallel run (profile: draw kinds).
    pub(super) fn in_parallel_run(&self) -> bool {
        self.par.is_some()
    }

    /// START_INST_LOC for an indirect draw packet that names none (our Agc
    /// builders): the SH register of the bound primitive shader's
    /// instance-offset user SGPR (its shader object declares it, see
    /// `agc::instance_offset_slot`), else 0 (none: the start instance then
    /// stays part of the instance id, as in Vulkan).
    fn instance_offset_loc(&self) -> u32 {
        const GS_USER_DATA: u32 = 0x8C;
        crate::hle::libs::agc::instance_offset_slot(super::shaders::ngg_addr(self)).map_or(0, |s| GS_USER_DATA + s)
    }

    /// DRAW_INDIRECT_MULTI / DRAW_INDEX_INDIRECT_MULTI: draws described by
    /// consecutive argument records. gfx9+ layout (Mesa radeonsi and the Agc
    /// builders emit it): data offset; BASE_VTX_LOC; START_INST_LOC;
    /// DRAW_INDEX_LOC [15:0] | bit 30 count from memory | bit 31 draw index
    /// enable; count; count address lo/hi; stride; initiator. Each draw gets
    /// its base vertex, start instance and draw index in the user SGPRs the
    /// packet names. Returns the outcome summary.
    fn draw_indirect_multi(&mut self, indexed: bool, p: &dyn Fn(u64) -> u32) -> String {
        let (vtx_loc, inst_loc, id_loc) = (p(1) & 0xFFFF, p(2) & 0xFFFF, p(3) & 0xFFFF);
        let inst_loc = if inst_loc != 0 { inst_loc } else { self.instance_offset_loc() };
        let id_enable = p(3) & (1 << 31) != 0;
        let mut count = p(4);
        let mut count_src = format!("max {count}");
        MDI_PACKETS.fetch_add(1, Ordering::Relaxed);
        if p(3) & (1 << 30) != 0 {
            MDI_COUNT_FROM_MEMORY.fetch_add(1, Ordering::Relaxed);
            // The count may have been written by earlier GPU work.
            let ca = addr64(p(5), p(6));
            self.sync_reads(&[(ca, 4)]);
            let mem = if valid(ca, 4) { Some(unsafe { rd32(ca) }) } else { None };
            count_src = format!("max {count}, memory {ca:#x} = {mem:?}");
            count = mem.map_or(0, |m| count.min(m));
        }
        let capturing = super::capture::capturing();
        let mut subs: Vec<String> = Vec::new();
        let rec = if indexed { 20 } else { 16 };
        let stride = if p(7) != 0 { p(7) as u64 } else { rec };
        // So may the arguments.
        if count > 0 {
            let first = self.draw_indirect_base + p(0) as u64;
            self.sync_reads(&[(first, (count.min(1 << 16) as u64 - 1) * stride + rec)]);
        }
        let size = if indexed { self.index_size() } else { Some(0) };
        let Some(sz) = size else {
            self.note("draws with 8-bit indices are not rendered yet");
            return "not executed (8-bit indices)".into();
        };
        let (mut drawn, mut outcome) = (0u32, String::new());
        for i in 0..count.min(1 << 16) {
            let a = self.draw_indirect_base + p(0) as u64 + i as u64 * stride;
            if !valid(a, rec) {
                break;
            }
            let w = unsafe { [rd32(a), rd32(a + 4), rd32(a + 8), rd32(a + 12), if indexed { rd32(a + 16) } else { 0 }] };
            if capturing && subs.len() < 8 {
                subs.push(format!("{:?}", &w[..rec as usize / 4]));
            }
            let (args, bv, si) = if indexed {
                (super::exec::DrawArgs { count: w[0], instances: w[1], index: Some((self.index_base + w[2] as u64 * sz as u64, sz)), base_vertex: Some(w[3] as i32), first_instance: w[4], gpu_args: None, base_inst: None }, w[3], w[4])
            } else {
                (super::exec::DrawArgs { count: w[0], instances: w[1], index: None, base_vertex: Some(w[2] as i32), first_instance: w[3], gpu_args: None, base_inst: None }, w[2], w[3])
            };
            if vtx_loc != 0 {
                self.set_reg(Space::Sh, vtx_loc, bv);
            }
            let mut args = args;
            if inst_loc != 0 {
                self.set_reg(Space::Sh, inst_loc, si);
                args.first_instance = 0;
            }
            if id_enable && id_loc != 0 {
                self.set_reg(Space::Sh, id_loc, i);
            }
            if args.count == 0 || args.instances == 0 {
                continue;
            }
            self.draw(args);
            drawn += 1;
            MDI_DRAWS.fetch_add(1, Ordering::Relaxed);
            outcome = std::mem::take(&mut self.last_draw).into_owned();
        }
        let detail = if capturing { format!(" [count {count_src}, stride {stride}, args {}]", subs.join(" ")) } else { String::new() };
        format!("{drawn} of {count} draws ({}){detail}", if drawn > 0 { outcome } else { "none".into() })
    }

    /// Smallest and largest index of `count` indices from `start` of the
    /// bound index buffer, and the first ones (frame captures).
    fn index_range(&self, start: u32, count: u32) -> String {
        let Some(sz) = self.index_size() else { return String::new() };
        let a = self.index_base + start as u64 * sz as u64;
        let n = count.min(1 << 20) as u64;
        if n == 0 || !guest_readable(a, n * sz as u64) {
            return " indices unreadable".into();
        }
        let get = |i: u64| unsafe { if sz == 2 { (rd32((a + 2 * i) & !3) >> (((a + 2 * i) & 2) * 8)) & 0xFFFF } else { rd32(a + 4 * i) } };
        let (mut lo, mut hi) = (u32::MAX, 0);
        for i in 0..n {
            let v = get(i);
            lo = lo.min(v);
            hi = hi.max(v);
        }
        let head: Vec<String> = (0..n.min(6)).map(|i| get(i).to_string()).collect();
        format!(" indices {lo}..{hi} [{}]", head.join(" "))
    }

    /// Whether a packet with the predicate bit executes under the current
    /// SET_PREDICATION state, and what the predicate read (None: no
    /// predication). ZPASS / PRIMCOUNT: begin/end counter pairs (16 bytes per
    /// render backend, bit 63 valid); BOOL64 / BOOL32: a value in memory.
    fn pred_value(&self) -> Option<(bool, String)> {
        let pr = self.pred?;
        let (visible, what) = match pr.op {
            1 | 2 => {
                if !guest_readable(pr.addr, 16 * 16) {
                    return Some((true, format!("zpass {:#x} unreadable", pr.addr)));
                }
                let (mut sum, mut valid) = (0u64, 0);
                for rb in 0..16u64 {
                    let (b, e) = unsafe { (rd64(pr.addr + 16 * rb), rd64(pr.addr + 16 * rb + 8)) };
                    if b >> 63 != 0 && e >> 63 != 0 {
                        valid += 1;
                        sum += (e & !(1 << 63)).wrapping_sub(b & !(1 << 63));
                    }
                }
                (sum > 0, format!("zpass {sum} ({valid} valid pairs)"))
            }
            3 => {
                let v = if guest_readable(pr.addr, 8) { unsafe { rd64(pr.addr) } } else { 0 };
                (v != 0, format!("bool64 {v:#x}"))
            }
            4 => {
                let v = if guest_readable(pr.addr, 4) { unsafe { rd32(pr.addr) } } else { 0 };
                (v != 0, format!("bool32 {v:#x}"))
            }
            o => return Some((true, format!("op {o}"))),
        };
        Some((if pr.cond == 1 { visible } else { !visible }, what))
    }

    /// Index size in bytes (VGT_INDEX_TYPE), None for 8-bit indices.
    fn index_size(&self) -> Option<u32> {
        match self.reg(Space::Uc, crate::hle::libs::agc_cmd::REG_VGT_INDEX_TYPE) & 3 {
            0 => Some(2),
            1 => Some(4),
            _ => None,
        }
    }

    fn note(&mut self, what: &'static str) {
        if self.notes_seen.insert(what) {
            crate::info!("gpu[{}]: {what}", self.name);
        }
    }

    fn set_reg(&mut self, space: Space, index: u32, value: u32) {
        let (v, n) = match space {
            Space::Ctx => (&mut self.regs.ctx, CTX_DWORDS),
            Space::Sh => (&mut self.regs.sh, SH_DWORDS),
            Space::Uc => (&mut self.regs.uc, UC_DWORDS),
        };
        if (index as usize) < n {
            if space == Space::Ctx && v[index as usize] != value {
                self.ctx_dirty = true;
            }
            v[index as usize] = value;
            note_high(space, index as usize);
        }
    }

    pub fn num_instances(&self) -> u32 {
        self.num_instances
    }

    pub fn reg(&self, space: Space, index: u32) -> u32 {
        let v = match space {
            Space::Ctx => &self.regs.ctx,
            Space::Sh => &self.regs.sh,
            Space::Uc => &self.regs.uc,
        };
        v.get(index as usize).copied().unwrap_or(0)
    }

    /// Register read by absolute byte address (COPY_DATA/WRITE_DATA register forms).
    fn reg_by_addr(&self, a: u32) -> u32 {
        match a {
            0x28000..=0x2FFFF => self.reg(Space::Ctx, (a - CTX_BASE) / 4),
            0xB000..=0xBFFF => self.reg(Space::Sh, (a - SH_BASE) / 4),
            0x30000..=0x3FFFF => self.reg(Space::Uc, (a - UC_BASE) / 4),
            _ => 0,
        }
    }

    /// Executes a command buffer (`size` dwords at `addr`). Small buffers (up
    /// to `FETCH_DWORDS`) are copied when execution starts, as the hardware's
    /// prefetch would have fetched them (a CPU rewrite of a started buffer is
    /// not seen; `DS_CP_FETCH=0` reads them lazily). Larger buffers are read
    /// as they execute: titles patch submitted frame command buffers until
    /// the GPU reaches the patched packets (copying them at submission broke
    /// rendering).
    pub fn run(&mut self, addr: u64, size: u32) {
        if self.depth > 8 {
            crate::error!("gpu[{}]: indirect buffer nesting too deep at {addr:#x}", self.name);
            return;
        }
        self.depth += 1;
        let mut cur = addr;
        let mut end = addr + 4 * size as u64;
        let mut f = Fetched::new(addr, size);
        // Time by packet class (the graphics queue; `tune cp_classes`).
        let classes = !self.compute && super::tune::cp_classes();
        if classes && !f.words.is_empty() {
            SERIAL_WHY[8].fetch_add(1, Ordering::Relaxed);
        }
        let acb_ops = self.compute && super::tune::cp_classes() && self.name.strip_prefix("gpu-acb").and_then(|h| u64::from_str_radix(h.trim_start_matches("0x"), 16).ok()) == Some(super::tune::cp_classes_acb());
        while cur < end {
            // The next chunk of a parallel run: collected here.
            if let Some(pr) = &self.par {
                if pr.starts[pr.next] == cur {
                    let t = class_start(classes);
                    cur = self.par_collect();
                    class_end(8, t);
                    continue;
                }
                if cur > pr.starts[pr.next] {
                    // Execution left the run's packets.
                    self.par_cancel();
                }
            }
            if super::shutting_down() {
                // Process shutdown releases label waits: the work behind
                // them would run without its inputs. Nothing more runs.
                break;
            }
            if !valid(cur, 4) {
                crate::error!("gpu[{}]: command pointer {cur:#x} outside guest memory", self.name);
                break;
            }
            let h = f.rd(cur);
            match h >> 30 {
                3 => {}
                2 => {
                    cur += 4;
                    continue;
                }
                0 => {
                    // Type-0: base register + (count + 1) values.
                    let n = ((h >> 16) & 0x3FFF) as u64 + 1;
                    cur += 4 * (n + 1);
                    self.note("type-0 packet skipped");
                    continue;
                }
                _ => {
                    crate::error!("gpu[{}]: bad packet header {h:#010x} at {cur:#x}", self.name);
                    break;
                }
            }
            let count = (h >> 16) & 0x3FFF;
            let opcode = (h >> 8) & 0xFF;
            if count == 0x3FFF {
                cur += 4; // header-only NOP
                continue;
            }
            let n = count as u64 + 1; // payload dwords
            let next = cur + 4 * (n + 1);
            if next > end + 4 * 0x4000 || !valid(cur, 4 * (n + 1)) {
                crate::error!("gpu[{}]: packet {opcode:#x} at {cur:#x} overruns its buffer", self.name);
                break;
            }
            let skip = self.par_skip.0 == end && cur < self.par_skip.1;
            if self.par.is_none() && !skip && is_command(opcode) && h & 1 == 0 && f.words.is_empty() && !f.live && self.exec.is_some() && (if self.compute { super::tune::par_compute() > 0 } else { super::par::workers() > 0 }) && !super::capture::capturing() {
                let t = class_start(classes);
                self.par_plan(&f, cur, end);
                class_end(7, t);
            }
            PACKETS.fetch_add(1, Ordering::Relaxed);
            self.cur_packet = cur;
            self.cur_opcode = opcode;
            // The packet as the CP fetches it on reaching it, before it
            // executes: handlers that wait (for GPU work, a label, a display
            // buffer) read their arguments as fetched. Read after the wait,
            // they came from what the title had written into the buffer
            // meanwhile: a FLIP (which waits for the work before it) took its
            // arguments from the rewritten buffer while its first dwords had
            // been read before, the flip failed, and the next frame waited for
            // its display buffer forever (runs 20261012-195508, -191738).
            let mut pk = [0u32; PACKET_FETCH];
            let fetched = if f.words.is_empty() { (n as usize).min(PACKET_FETCH) } else { 0 };
            for (i, w) in pk.iter_mut().enumerate().take(fetched) {
                *w = f.rd(cur + 4 * (1 + i as u64));
            }
            let p = |i: u64| if (i as usize) < fetched { pk[i as usize] } else { f.rd(cur + 4 * (1 + i)) };
            let command = matches!(
                opcode,
                op::DRAW_INDEX_2 | op::DRAW_INDEX_AUTO | op::DRAW_INDEX_OFFSET_2 | op::DRAW_INDIRECT | op::DRAW_INDEX_INDIRECT | op::DISPATCH_DIRECT | op::DISPATCH_INDIRECT
            );
            if command && !self.ahead.is_empty() {
                // The preparations made ahead for this command (another one
                // first: execution took another path than the scan).
                if self.ahead.front().is_some_and(|c| c.at == cur) {
                    let c = self.ahead.pop_front().unwrap();
                    *self.cur_ahead.borrow_mut() = c.stages;
                } else {
                    self.drop_ahead();
                }
            } else if !command
                && !self.ahead.is_empty()
                && !matches!(
                    opcode,
                    op::NOP
                        | op::SET_CONTEXT_REG
                        | op::SET_CONTEXT_REG_INDEX
                        | op::SET_SH_REG
                        | op::SET_SH_REG_INDEX
                        | op::SET_UCONFIG_REG
                        | op::SET_UCONFIG_REG_INDEX
                        | op::LOAD_UCONFIG_REG
                        | op::LOAD_CONTEXT_REG_INDEX
                        | op::LOAD_SH_REG_INDEX
                        | op::INDEX_TYPE
                        | op::INDEX_BASE
                        | op::INDEX_BUFFER_SIZE
                        | op::NUM_INSTANCES
                        | op::CONTEXT_CONTROL
                        | op::SET_BASE
                )
                // Preparations made past GPU-side synchronization, CP writes
                // and label waits (`scan_through`) are checked when used.
                && !matches!(opcode, op::EVENT_WRITE | op::ACQUIRE_MEM | op::RELEASE_MEM | op::WRITE_DATA | op::WAIT_REG_MEM | op::WAIT_REG_MEM64)
            {
                self.drop_ahead();
            }
            if super::capture::capturing() {
                // Synchronization and memory packets in frame captures.
                let name = match opcode {
                    op::WAIT_REG_MEM => Some("WAIT_REG_MEM"),
                    op::WAIT_REG_MEM64 => Some("WAIT_REG_MEM64"),
                    op::WRITE_DATA => Some("WRITE_DATA"),
                    op::RELEASE_MEM => Some("RELEASE_MEM"),
                    op::EVENT_WRITE => Some("EVENT_WRITE"),
                    op::ACQUIRE_MEM => Some("ACQUIRE_MEM"),
                    op::DMA_DATA => Some("DMA_DATA"),
                    op::COPY_DATA => Some("COPY_DATA"),
                    op::COND_EXEC => Some("COND_EXEC"),
                    op::INDIRECT_BUFFER => Some("INDIRECT_BUFFER"),
                    op::PFP_SYNC_ME => Some("PFP_SYNC_ME"),
                    op::DISPATCH_DIRECT | op::DISPATCH_INDIRECT | op::SET_SH_REG | op::SET_CONTEXT_REG | op::SET_UCONFIG_REG => None,
                    _ if std::env::var("DS_CAPTURE_ALL_PACKETS").ok().as_deref() == Some("1") => Some("PACKET"),
                    _ => None,
                };
                if let Some(name) = name {
                    let args: Vec<String> = (0..n.min(8)).map(|i| format!("{:08x}", p(i))).collect();
                    super::capture::packet(self, &format!("{name} {opcode:#x} @{cur:#x} [{}]", args.join(" ")));
                }
            }
            if classes && command && self.par.is_none() {
                let k = if !f.words.is_empty() {
                    0
                } else if f.live {
                    1
                } else if self.par_skip.0 == end && cur < self.par_skip.1 {
                    match self.par_skip_n {
                        0 | 1 => 2,
                        2..=7 => 3,
                        8..=31 => 4,
                        32..=127 => 5,
                        _ => 6,
                    }
                } else {
                    7
                };
                SERIAL_WHY[k].fetch_add(1, Ordering::Relaxed);
            }
            let dispatch = matches!(opcode, op::DISPATCH_DIRECT | op::DISPATCH_INDIRECT);
            let class = if command {
                (if self.par.is_some() { 0 } else { 2 }) + dispatch as usize
            } else if is_state(opcode) {
                5
            } else if opcode == op::NOP {
                6
            } else {
                4
            };
            let t = class_start(classes && opcode != op::INDIRECT_BUFFER);
            let ta = class_start(acb_ops && opcode != op::INDIRECT_BUFFER);
            // A label wait that blocks prepares the stages after it meanwhile
            // (`tune wait_ahead`).
            if matches!(opcode, op::WAIT_REG_MEM | op::WAIT_REG_MEM64) && !f.live && self.exec.is_some() && super::tune::wait_ahead_for(self.compute) {
                self.wait_ctx = Some((f.clone(), next, end));
            }
            // A dispatch or draw whose pipeline turns out missing scans the
            // commands after it for more first uses (`tune compile_ahead`,
            // `compile_gfx`).
            if command && !f.live && (if dispatch { super::compile::at_first_uses() } else { super::compile::gfx_at_first_uses() }) {
                self.compile_ctx.set(Some((f.base, next, end)));
            }
            let flow = self.packet(opcode, h, cur, n, &p);
            self.wait_ctx = None;
            self.compile_ctx.set(None);
            class_end(class, t);
            acb_op_end(opcode, ta);
            if command {
                // Preparations of this command that it did not take.
                for s in self.take_ahead().iter().flatten() {
                    s.cancel();
                }
                if self.ahead.is_empty() && matches!(flow, Flow::Next) && !f.live && self.exec.is_some() && super::ahead::enabled_for(self.compute) {
                    // Within a parallel run's first chunk, only to its end.
                    let until = self.par.as_ref().map_or(end, |p| p.starts[p.next].min(end));
                    self.scan_ahead(&f, next, until);
                }
            }
            if self.exec.as_ref().is_some_and(|e| e.submit_requested()) {
                let ta = class_start(acb_ops);
                self.submit_gpu("label");
                acb_op_end(1, ta);
            }
            if opcode == op::REWIND {
                // The CPU streams packets behind a REWIND: read them live.
                f = Fetched::new(next, 0);
                f.live = true;
            }
            match flow {
                Flow::Next => cur = next,
                Flow::Skip(d) => cur = next + 4 * d as u64,
                Flow::Chain(a, s) => {
                    cur = a;
                    end = a + 4 * s as u64;
                    f = Fetched::new(a, s);
                }
                Flow::Stop => break,
            }
        }
        // A parallel run planned in this buffer ends in it.
        if self.par.is_some() {
            self.par_cancel();
        }
        self.depth -= 1;
        if self.depth == 0 {
            // The batch stays open into the next command buffer when that one
            // is already here (`tune cmdbuf_merge`): a batch less to begin.
            let mode = super::tune::cmdbuf_merge();
            let merge = (mode >= 2 || (mode == 1 && !self.compute)) && (self.next_in_submission || self.more_follows.as_ref().is_some_and(|f| f()));
            if merge {
                CMDBUF_MERGED.fetch_add(1, Ordering::Relaxed);
            } else {
                let t = class_start(classes);
                self.submit_gpu("cmdbuf");
                class_end(9, t);
            }
        }
    }

    fn packet(&mut self, opcode: u32, h: u32, at: u64, n: u64, p: &dyn Fn(u64) -> u32) -> Flow {
        if self.repeat_ok && !self.keeps_repeat(opcode, n, p) {
            self.repeat_ok = false;
            REPEAT_ENDS[(opcode & 0xFF) as usize].fetch_add(1, Ordering::Relaxed);
        }
        match opcode {
            op::NOP => {
                if n >= 2 && p(0) == DS_MAGIC {
                    return self.private(p(1), at, n - 2, &|i| p(2 + i));
                }
            }
            op::SET_CONTEXT_REG | op::SET_CONTEXT_REG_INDEX => self.set_range(Space::Ctx, p(0) & 0xFFFF, n, p),
            op::SET_SH_REG | op::SET_SH_REG_INDEX => self.set_range(Space::Sh, p(0) & 0xFFFF, n, p),
            op::SET_UCONFIG_REG | op::SET_UCONFIG_REG_INDEX => self.set_range(Space::Uc, p(0) & 0xFFFF, n, p),
            op::SET_CONFIG_REG => self.note("SET_CONFIG_REG ignored"),
            op::SET_SH_REG_OFFSET => self.note("SET_SH_REG_OFFSET ignored"),
            op::LOAD_CONTEXT_REG_INDEX => self.load_pairs(Space::Ctx, addr64(p(0), p(1)), p(2)),
            op::LOAD_SH_REG_INDEX => self.load_pairs(Space::Sh, addr64(p(0), p(1)), p(2)),
            op::LOAD_UCONFIG_REG => self.load_pairs(Space::Uc, addr64(p(0), p(1)), p(2)),
            op::CLEAR_STATE => self.clear_context(),
            op::CONTEXT_CONTROL | op::PFP_SYNC_ME => {}
            // Partial flushes, cache actions and acquires order later work
            // after earlier work: GPU work is already ordered by barriers, so
            // only a following CP memory write has to wait (see `fence`).
            op::EVENT_WRITE | op::ACQUIRE_MEM => {
                self.fence_pending = true;
                if let Some(e) = self.exec.as_mut() {
                    e.note_sync();
                }
            }
            op::INDEX_TYPE => self.set_reg(Space::Uc, crate::hle::libs::agc_cmd::REG_VGT_INDEX_TYPE, p(0)),
            op::INDEX_BASE => self.index_base = addr64(p(0), p(1)),
            op::INDEX_BUFFER_SIZE => self.index_count = p(0),
            op::NUM_INSTANCES => self.num_instances = p(0).max(1),
            op::SET_BASE => {
                let base = addr64(p(1), p(2));
                if p(0) == 2 { self.dispatch_indirect_base = base } else { self.draw_indirect_base = base }
            }
            op::SET_PREDICATION => {
                let ctl = p(0);
                let pop = (ctl >> 16) & 7;
                let addr = addr64(p(1), p(2));
                self.pred = (pop != 0).then_some(Pred { op: pop, cond: (ctl >> 8) & 1, addr, count: if n >= 4 { p(3) } else { 0 } });
                self.note("SET_PREDICATION recorded but not applied");
                if super::capture::capturing() {
                    let what = self.pred_value().map(|(d, w)| format!("{w} -> {}", if d { "draw" } else { "skip" })).unwrap_or_else(|| "clear".into());
                    super::capture::packet(self, &format!("SET_PREDICATION op {pop} cond {} wait {} addr {addr:#x} count {}: {what}", (ctl >> 8) & 1, (ctl >> 12) & 1, if n >= 4 { p(3) } else { 0 }));
                }
            }
            op::DRAW_INDEX_2 | op::DRAW_INDEX_AUTO | op::DRAW_INDEX_OFFSET_2 | op::DRAW_INDIRECT | op::DRAW_INDEX_INDIRECT
            | op::DRAW_INDIRECT_MULTI | op::DRAW_INDEX_INDIRECT_MULTI | op::DRAW_INDEX_MULTI_AUTO => {
                DRAWS.fetch_add(1, Ordering::Relaxed);
                let instances = self.num_instances;
                let mut multi: Option<String> = None;
                let mut indirect: Option<String> = None;
                let args = match opcode {
                    op::DRAW_INDEX_AUTO => Some(super::exec::DrawArgs { count: p(0), instances, index: None, base_vertex: None, first_instance: 0, gpu_args: None, base_inst: None }),
                    op::DRAW_INDEX_2 => self.index_size().map(|sz| super::exec::DrawArgs { count: p(3), instances, index: Some((addr64(p(1), p(2)), sz)), base_vertex: None, first_instance: 0, gpu_args: None, base_inst: None }),
                    op::DRAW_INDEX_OFFSET_2 => self.index_size().map(|sz| super::exec::DrawArgs { count: p(2), instances, index: Some((self.index_base + p(1) as u64 * sz as u64, sz)), base_vertex: None, first_instance: 0, gpu_args: None, base_inst: None }),
                    op::DRAW_INDIRECT | op::DRAW_INDEX_INDIRECT if self.draw_gpu_args(opcode, p) => {
                        multi = Some(std::mem::take(&mut self.last_draw).into_owned());
                        if super::capture::capturing() {
                            indirect = Some("args read by the GPU".into());
                        }
                        None
                    }
                    op::DRAW_INDIRECT | op::DRAW_INDEX_INDIRECT => {
                        // Arguments may have been written by earlier GPU work.
                        let a = self.draw_indirect_base + p(0) as u64;
                        self.sync_reads(&[(a, if opcode == op::DRAW_INDEX_INDIRECT { 20 } else { 16 })]);
                        // The CP hands the base vertex and start instance to the
                        // shader through the user SGPRs the packet names
                        // (BASE_VTX_LOC [15:0], START_INST_LOC [31:16]: SH
                        // register offsets; as Mesa radeonsi emits them), and
                        // the instance id VGPR then starts at 0. Our Agc
                        // packets name no locations: the start instance goes
                        // to the bound shader's instance-offset slot.
                        let (vtx_loc, inst_loc) = (p(1) & 0xFFFF, p(1) >> 16);
                        let inst_loc = if inst_loc != 0 { inst_loc } else { self.instance_offset_loc() };
                        let args = if opcode == op::DRAW_INDIRECT && valid(a, 16) {
                            let w = unsafe { [rd32(a), rd32(a + 4), rd32(a + 8), rd32(a + 12)] };
                            if super::capture::capturing() {
                                indirect = Some(format!("args {w:?} locs {vtx_loc:#x}/{inst_loc:#x}"));
                            }
                            Some((super::exec::DrawArgs { count: w[0], instances: w[1], index: None, base_vertex: Some(w[2] as i32), first_instance: w[3], gpu_args: None, base_inst: None }, w[2], w[3]))
                        } else if opcode == op::DRAW_INDEX_INDIRECT && valid(a, 20) {
                            let w = unsafe { [rd32(a), rd32(a + 4), rd32(a + 8), rd32(a + 12), rd32(a + 16)] };
                            if super::capture::capturing() {
                                let ty = self.reg(Space::Uc, crate::hle::libs::agc_cmd::REG_VGT_INDEX_TYPE) & 3;
                                indirect = Some(format!("args {w:?} locs {vtx_loc:#x}/{inst_loc:#x} index {:#x} type {ty} size {}{}", self.index_base, self.index_count, self.index_range(w[2], w[0])));
                            }
                            match self.index_size() {
                                // The start instance as Vulkan's first instance
                                // (`tune cpu_args_base`): drawn here.
                                Some(sz) if self.cpu_args_base_ok(vtx_loc, inst_loc) => {
                                    multi = Some(self.draw_cpu_args_base(w, sz, inst_loc));
                                    None
                                }
                                sz => sz.map(|sz| (super::exec::DrawArgs { count: w[0], instances: w[1], index: Some((self.index_base + w[2] as u64 * sz as u64, sz)), base_vertex: Some(w[3] as i32), first_instance: w[4], gpu_args: None, base_inst: None }, w[3], w[4])),
                            }
                        } else {
                            None
                        };
                        if let Some((_, bv, si)) = args {
                            if vtx_loc != 0 {
                                self.set_reg(Space::Sh, vtx_loc, bv);
                            }
                            if inst_loc != 0 {
                                self.set_reg(Space::Sh, inst_loc, si);
                            }
                        }
                        args.map(|(mut a, _, _)| {
                            if inst_loc != 0 {
                                a.first_instance = 0;
                            }
                            a
                        })
                    }
                    op::DRAW_INDIRECT_MULTI | op::DRAW_INDEX_INDIRECT_MULTI => {
                        multi = Some(self.draw_indirect_multi(opcode == op::DRAW_INDEX_INDIRECT_MULTI, p));
                        None
                    }
                    _ => {
                        self.note("multi-draw packets are not rendered yet");
                        None
                    }
                };
                self.last_draw = multi.take().map_or(std::borrow::Cow::Borrowed("not executed"), std::borrow::Cow::Owned);
                match args {
                    Some(a) => {
                        self.draw(a);
                        if super::capture::draw_dumps() || super::capture::watching() || super::capture::listed_draw(self) {
                            // Per-draw target dumps: complete the draw first.
                            if let Some(e) = self.exec.as_mut() {
                                e.flush();
                            }
                        }
                    }
                    None if matches!(opcode, op::DRAW_INDEX_2 | op::DRAW_INDEX_OFFSET_2 | op::DRAW_INDEX_INDIRECT) && self.index_size().is_none() => {
                        self.note("draws with 8-bit indices are not rendered yet")
                    }
                    None => {}
                }
                if super::capture::capturing() {
                    let what = match opcode {
                        op::DRAW_INDEX_2 => format!(
                            "DRAW_INDEX_2 count {} index {:#x} type {} instances {}",
                            p(3),
                            addr64(p(1), p(2)),
                            self.reg(Space::Uc, crate::hle::libs::agc_cmd::REG_VGT_INDEX_TYPE) & 3,
                            self.num_instances
                        ),
                        op::DRAW_INDEX_AUTO => format!("DRAW_INDEX_AUTO count {} instances {}", p(0), self.num_instances),
                        op::DRAW_INDEX_OFFSET_2 => format!("DRAW_INDEX_OFFSET_2 count {} offset {} base {:#x}", p(2), p(1), self.index_base),
                        _ => format!("indirect draw opcode {opcode:#x} offset {:#x} base {:#x}{}", p(0), self.draw_indirect_base, indirect.take().map(|s| format!(" {s}")).unwrap_or_default()),
                    };
                    let what = match (h & 1 != 0).then(|| self.pred_value()) {
                        Some(Some((d, w))) => format!("{what} predicated ({w}: {})", if d { "draw" } else { "skip" }),
                        Some(None) => format!("{what} predicated (no predication set)"),
                        None => what,
                    };
                    super::capture::draw(self, &what);
                }
            }
            op::DISPATCH_DIRECT | op::DISPATCH_INDIRECT => {
                DISPATCHES.fetch_add(1, Ordering::Relaxed);
                let mut indirect_groups = None;
                let mut indirect_extra: Option<[u32; 2]> = None;
                let mut gpu_read = false;
                if opcode == op::DISPATCH_DIRECT {
                    // DS_DELAY_CS=<shader>:<ms>: wait before that shader's
                    // dispatches (diagnosis of CPU/GPU data races).
                    static DELAY: std::sync::OnceLock<Option<(Vec<u64>, u64)>> = std::sync::OnceLock::new();
                    if let Some((list, ms)) = DELAY.get_or_init(|| {
                        let v = std::env::var("DS_DELAY_CS").ok()?;
                        let (a, ms) = v.split_once(':')?;
                        Some((a.split(',').filter_map(|x| u64::from_str_radix(x.trim_start_matches("0x"), 16).ok()).collect(), ms.parse().ok()?))
                    }) {
                        let cs = ((self.reg(Space::Sh, 0x20C) as u64) | ((self.reg(Space::Sh, 0x20D) as u64 & 0xFF) << 32)) << 8;
                        if list.contains(&cs) {
                            std::thread::sleep(std::time::Duration::from_millis(*ms));
                        }
                    }
                    super::emucheck::race_check(self, p(3));
                    super::emucheck::race_watch(self, p(3));
                    // DS_EMU_CHECK: interpret a sample of the groups first.
                    let check = if super::exec::enabled() && super::emucheck::wanted(self) {
                        self.sync();
                        super::emucheck::prepare(self, [p(0), p(1), p(2)], p(3))
                    } else {
                        None
                    };
                    self.dispatch([p(0), p(1), p(2)], p(3));
                    if let Some(c) = check {
                        self.sync();
                        super::emucheck::compare(c);
                    }
                } else {
                    // Graphics ring: [offset from the SET_BASE base, initiator];
                    // compute queues: [address lo, address hi, initiator] (the
                    // form sceAgcAcbDispatchIndirect emits).
                    let (a, initiator) = if n >= 3 { (addr64(p(0), p(1)), p(2)) } else { (self.dispatch_indirect_base + p(0) as u64, p(1)) };
                    // Arguments that GPU work still writes: the GPU reads them
                    // (captures and interpreter checks need the group counts).
                    let gpu = valid(a, 12) && !super::capture::capturing() && !(super::exec::enabled() && super::emucheck::wanted(self)) && self.dispatch_gpu_indirect(a, initiator);
                    gpu_read = gpu;
                    if !gpu {
                        // Arguments may have been written by earlier GPU work.
                        self.sync_reads(&[(a, 12)]);
                        if !valid(a, 12) {
                            crate::warn!("gpu[{}]: DISPATCH_INDIRECT arguments at {a:#x} outside guest memory (dispatch skipped)", self.name);
                        }
                        if valid(a, 12) {
                            let g = unsafe { [rd32(a), rd32(a + 4), rd32(a + 8)] };
                            indirect_groups = Some(g);
                            // Captures: the two dwords after the group counts
                            // (titles keep item counts there).
                            if super::capture::capturing() && valid(a + 12, 8) {
                                indirect_extra = Some(unsafe { [rd32(a + 12), rd32(a + 16)] });
                            }
                            // DS_EMU_CHECK: interpret a sample of the groups first
                            // (after the earlier work has completed).
                            let check = if super::exec::enabled() && super::emucheck::wanted(self) {
                                self.sync();
                                super::emucheck::prepare(self, g, initiator)
                            } else {
                                None
                            };
                            super::exec::set_cpu_indirect(true);
                            self.dispatch(g, initiator);
                            super::exec::set_cpu_indirect(false);
                            if let Some(c) = check {
                                self.sync();
                                super::emucheck::compare(c);
                            }
                        }
                    }
                }
                // DS_TRACE_CS=<shader>[,<shader>..]: the first 8 dispatches of
                // each of those compute shaders log their resources and what
                // the image cache holds for each image (diagnostics).
                if !trace_cs_list().is_empty() {
                    let cs = ((self.reg(Space::Sh, 0x20C) as u64) | ((self.reg(Space::Sh, 0x20D) as u64 & 0xFF) << 32)) << 8;
                    if trace_cs_list().contains(&cs) {
                        static SEEN: parking_lot::Mutex<Vec<(u64, u32)>> = parking_lot::Mutex::new(Vec::new());
                        let n = {
                            let mut g = SEEN.lock();
                            match g.iter_mut().find(|e| e.0 == cs) {
                                Some(e) => {
                                    e.1 += 1;
                                    e.1
                                }
                                None => {
                                    g.push((cs, 1));
                                    1
                                }
                            }
                        };
                        if n <= 8 {
                            let mut lines = Vec::new();
                            if let Ok(p) = super::shaders::prepare_compute(self, true) {
                                lines.extend(super::shaders::describe_resources(&p));
                                for r in &p.resources {
                                    if let super::shaders::Resolved::Image(t) = r {
                                        lines.push(format!("image {:#x}: {}", t.base, super::images::with(|im| im.describe(t))));
                                    }
                                }
                            }
                            crate::info!(
                                "gpu[{}]: trace cs {cs:#x} dispatch #{n} {}x{}x{} at flip {}:\n    {}",
                                self.name,
                                p(0),
                                p(1),
                                p(2),
                                super::FRAMES.load(Ordering::Relaxed),
                                lines.join("\n    ")
                            );
                        }
                    }
                }
                if super::capture::dispatch_dumps() || super::capture::watching() {
                    // Per-dispatch image dumps and pixel watches: complete
                    // the dispatch first.
                    if let Some(e) = self.exec.as_mut() {
                        e.flush();
                    }
                }
                if super::capture::capturing() {
                    let what = if opcode == op::DISPATCH_DIRECT {
                        format!("DISPATCH_DIRECT {}x{}x{} initiator {:#x}", p(0), p(1), p(2), p(3))
                    } else {
                        let g = indirect_groups.map(|g| format!(" groups {}x{}x{}", g[0], g[1], g[2])).unwrap_or_else(|| if gpu_read { " (groups read by the GPU)".into() } else { " (skipped)".into() });
                        let g = match indirect_extra {
                            Some(x) => format!("{g} (then {:#x} {:#x})", x[0], x[1]),
                            None => g,
                        };
                        if n >= 3 {
                            format!("DISPATCH_INDIRECT args {:#x}{g}", addr64(p(0), p(1)))
                        } else {
                            format!("DISPATCH_INDIRECT {:#x} base {:#x}{g}", p(0), self.dispatch_indirect_base)
                        }
                    };
                    super::capture::dispatch(self, &what);
                    // DS_REPEAT_CS=<shader>: run that captured direct dispatch
                    // again on the same inputs (determinism checks with
                    // DS_CAPTURE_DISPATCHES).
                    let cs = ((self.reg(Space::Sh, 0x20C) as u64) | ((self.reg(Space::Sh, 0x20D) as u64 & 0xFF) << 32)) << 8;
                    if opcode == op::DISPATCH_DIRECT && std::env::var("DS_REPEAT_CS").ok().and_then(|v| u64::from_str_radix(v.trim_start_matches("0x"), 16).ok()) == Some(cs) {
                        self.dispatch([p(0), p(1), p(2)], p(3));
                        if let Some(e) = self.exec.as_mut() {
                            e.flush();
                        }
                        super::capture::dispatch(self, &format!("{what} (repeated)"));
                    }
                }
            }
            // CP memory writes wait for shaders only after a partial flush or
            // acquire (titles synchronize with those or RELEASE_MEM first).
            op::WRITE_DATA => self.write_data(n, p),
            op::COPY_DATA => self.copy_data(p),
            op::DMA_DATA => self.dma_data(p),
            op::RELEASE_MEM => self.release_mem(p),
            op::WAIT_REG_MEM => {
                if !super::tune::lazy_wait_submit() {
                    self.submit_gpu("wait");
                }
                let c = p(0);
                let a = addr64(p(1), p(2));
                let flow = self.wait(c, a, false, p(3) as u64, p(4) as u64, at);
                self.end_copy_scope_after_wait(a, false);
                return flow;
            }
            op::WAIT_REG_MEM64 => {
                if !super::tune::lazy_wait_submit() {
                    self.submit_gpu("wait");
                }
                let c = p(0);
                let a = addr64(p(1), p(2));
                let flow = self.wait(c, a, true, addr64(p(3), p(4)), addr64(p(5), p(6)), at);
                self.end_copy_scope_after_wait(a, true);
                return flow;
            }
            op::COND_EXEC => {
                self.sync();
                let a = addr64(p(0), p(1));
                // Another queue's unfinished batch may write it (queues run
                // ahead on promises).
                self.sync_reads(&[(a, 4)]);
                let skip = p(3) & 0x3FFF;
                if valid(a, 4) && unsafe { rd32(a) } == 0 {
                    return Flow::Skip(skip);
                }
            }
            op::INDIRECT_BUFFER => {
                let a = addr64(p(0) & !3, p(1));
                let c = p(2);
                let size = c & 0xF_FFFF;
                if c & (1 << 20) != 0 {
                    return Flow::Chain(a, size);
                }
                self.run(a, size);
            }
            op::REWIND => {
                if !self.compute {
                    super::precache::end_scope(2);
                }
                // The parser stalls until the CPU marks the packet valid.
                if p(0) & (1 << 31) == 0 {
                    return self.poll_until(at, &format!("REWIND at {at:#x} (waiting for the valid bit)"), || unsafe { rd32(at + 4) } & (1 << 31) != 0);
                }
            }
            _ => {
                if self.unknown_seen.insert(opcode) {
                    let words: Vec<String> = (0..n.min(8)).map(|i| format!("{:08x}", p(i))).collect();
                    crate::warn!("gpu[{}]: unhandled PM4 opcode {opcode:#04x} (header {h:#010x}, {n} dwords: {}) at {at:#x}", self.name, words.join(" "));
                }
            }
        }
        Flow::Next
    }

    fn set_range(&mut self, space: Space, first: u32, n: u64, p: &dyn Fn(u64) -> u32) {
        for i in 1..n {
            self.set_reg(space, first + (i - 1) as u32, p(i));
        }
    }

    /// Our indirect register packets: memory holds `n` {offset, value} pairs.
    fn load_pairs(&mut self, space: Space, a: u64, n: u32) {
        if n == 0 {
            return;
        }
        if !valid(a, 8 * n as u64) {
            crate::error!("gpu[{}]: register pairs at {a:#x} ({n}) outside guest memory", self.name);
            return;
        }
        for i in 0..n as u64 {
            let (o, v) = unsafe { (rd32(a + 8 * i), rd32(a + 8 * i + 4)) };
            self.set_reg(space, o & 0xFFFF, v);
        }
    }

    fn write_data(&mut self, n: u64, p: &dyn Fn(u64) -> u32) {
        let c = p(0);
        let dst = (c >> 8) & 0xF;
        let no_incr = (c >> 16) & 1 != 0;
        let a = addr64(p(1), p(2));
        let count = n.saturating_sub(3);
        match dst {
            1 | 2 | 5 => {
                if !valid(a, 4 * count.max(1)) {
                    crate::error!("gpu[{}]: WRITE_DATA to {a:#x} outside guest memory", self.name);
                    return;
                }
                if count == 0 {
                    return;
                }
                let vals: Vec<u32> = (0..count).map(|i| p(3 + i)).collect();
                let (name, at) = (self.name.clone(), self.cur_packet);
                let len = if no_incr { 4 } else { 4 * count };
                super::mirror::invalidate(&[(a, len)]);
                let label = match (count, no_incr) {
                    (1, _) => Some((vals[0] as u64, 4u8)),
                    (2, false) => Some((vals[0] as u64 | (vals[1] as u64) << 32, 8u8)),
                    _ => None,
                };
                // Without address increment only the last value stays.
                let gpu_vals = if no_incr { vec![*vals.last().unwrap()] } else { vals.clone() };
                if self.gpu_cp_write(a, &gpu_vals, label, false) {
                    return;
                }
                let seq = self.effect(
                    false,
                    Some((a, len)),
                    Box::new(move || {
                        for (i, v) in vals.iter().enumerate() {
                            let d = if no_incr { a } else { a + 4 * i as u64 };
                            unsafe { wr32(d, *v) };
                        }
                        if vals.len() <= 4 {
                            note_write(&name, a, vals[0] as u64 | (vals.get(1).copied().unwrap_or(0) as u64) << 32, at);
                        }
                    }),
                );
                if let (Some(s), Some((v, bytes))) = (seq, label) {
                    promise(s, a, v, bytes, true);
                }
            }
            0 => {
                for i in 0..count {
                    let r = (a as u32).wrapping_add(if no_incr { 0 } else { 4 * i as u32 });
                    self.write_reg_addr(r, p(3 + i));
                }
            }
            _ => {
                if self.unknown_seen.insert(0x1_0000 | dst) {
                    let words: Vec<String> = (0..n.min(6)).map(|i| format!("{:08x}", p(i))).collect();
                    crate::warn!("gpu[{}]: WRITE_DATA with dst_sel {dst} ignored (addr {a:#x}; {})", self.name, words.join(" "));
                }
            }
        }
    }

    fn write_reg_addr(&mut self, r: u32, v: u32) {
        match r {
            0x28000..=0x2FFFF => self.set_reg(Space::Ctx, (r - CTX_BASE) / 4, v),
            0xB000..=0xBFFF => self.set_reg(Space::Sh, (r - SH_BASE) / 4, v),
            0x30000..=0x3FFFF => self.set_reg(Space::Uc, (r - UC_BASE) / 4, v),
            // Dword-index form (MMIO offsets).
            _ if r < 0x10000 => self.write_reg_addr(r * 4, v),
            _ => {}
        }
    }

    fn copy_data(&mut self, p: &dyn Fn(u64) -> u32) {
        let c = p(0);
        let src = c & 0xF;
        let dst = (c >> 8) & 0xF;
        let wide = (c >> 16) & 1 != 0;
        let src_v = addr64(p(1), p(2));
        let dst_a = addr64(p(3), p(4));
        if matches!(src, 1 | 2) && !valid(src_v, 8) {
            crate::error!("gpu[{}]: COPY_DATA from {src_v:#x} outside guest memory", self.name);
            return;
        }
        if !matches!(src, 0 | 1 | 2 | 5 | 9) {
            self.note("COPY_DATA from GDS/perf counters reads 0");
        }
        let reg = if src == 0 { self.reg_by_addr(src_v as u32) as u64 } else { 0 };
        let read = move || -> u64 {
            match src {
                0 => reg,
                1 | 2 => {
                    if wide { unsafe { rd64(src_v) } } else { (unsafe { rd32(src_v) }) as u64 }
                }
                5 => src_v,
                9 => gpu_clock(),
                _ => 0,
            }
        };
        match dst {
            1 | 2 | 5 => {
                if !valid(dst_a, 8) {
                    crate::error!("gpu[{}]: COPY_DATA to {dst_a:#x} outside guest memory", self.name);
                    return;
                }
                let (name, at) = (self.name.clone(), self.cur_packet);
                super::mirror::invalidate(&[(dst_a, if wide { 8 } else { 4 })]);
                let seq = self.effect(
                    false,
                    Some((dst_a, if wide { 8 } else { 4 })),
                    Box::new(move || {
                        let value = read();
                        if wide { unsafe { wr64(dst_a, value) } } else { unsafe { wr32(dst_a, value as u32) } }
                        note_write(&name, dst_a, value, at);
                    }),
                );
                if let (Some(s), true) = (seq, matches!(src, 1 | 2)) {
                    super::exec::note_cp_source(s, (src_v, if wide { 8 } else { 4 }));
                }
            }
            0 => {
                // A register needs the value now: memory sources must be current.
                if matches!(src, 1 | 2) && self.exec.as_ref().is_some_and(|e| e.busy()) {
                    self.sync();
                }
                let v = read();
                self.write_reg_addr(dst_a as u32, v as u32);
            }
            _ => self.note("COPY_DATA to GDS/other ignored"),
        }
    }

    fn dma_data(&mut self, p: &dyn Fn(u64) -> u32) {
        let c1 = p(0);
        let c6 = p(5);
        let src_sel = (c1 >> 29) & 3;
        let dst_sel = (c1 >> 20) & 3;
        let bytes = (c6 & 0x03FF_FFFF) as u64;
        let sas = (c6 >> 26) & 1 != 0;
        let das = (c6 >> 27) & 1 != 0;
        let saic = (c6 >> 28) & 1 != 0;
        let daic = (c6 >> 29) & 1 != 0;
        let src = addr64(p(1), p(2));
        let dst = addr64(p(3), p(4));
        if bytes == 0 {
            return;
        }
        if dst_sel == 1 || das {
            self.note("DMA_DATA to GDS/registers ignored");
            return;
        }
        if !valid(dst, if daic { 4 } else { bytes }) {
            crate::error!("gpu[{}]: DMA_DATA to {dst:#x}+{bytes:#x} outside guest memory", self.name);
            return;
        }
        let (name, at) = (self.name.clone(), self.cur_packet);
        let range = Some((dst, if daic { 4 } else { bytes }));
        super::mirror::invalidate(&[(dst, if daic { 4 } else { bytes })]);
        super::precache::invalidate(dst, if daic { 4 } else { bytes });
        super::exec::trace_writes(|| format!("{} DMA_DATA (src_sel {src_sel}, from {src:#x})", self.name), dst, bytes);
        if super::exec::enabled() {
            super::images::with(|im| {
                im.note_meta_writes(&[(dst, if daic { 4 } else { bytes as u64 })]);
                im.note_overwrites(&[(dst, if daic { 4 } else { bytes as u64 })]);
            });
        }
        match src_sel {
            2 => {
                // Fill with the immediate dword.
                let v = src as u32;
                self.effect(
                    false,
                    range,
                    Box::new(move || {
                        if daic {
                            unsafe { wr32(dst, v) };
                            note_write(&name, dst, v as u64, at);
                        } else {
                            for i in 0..bytes / 4 {
                                unsafe { wr32(dst + 4 * i, v) };
                            }
                            note_block(&name, dst, bytes, at);
                        }
                    }),
                );
            }
            0 | 3 if !sas => {
                if !valid(src, if saic { 4 } else { bytes }) {
                    crate::error!("gpu[{}]: DMA_DATA from {src:#x}+{bytes:#x} outside guest memory", self.name);
                    return;
                }
                let seq = self.effect(
                    false,
                    range,
                    Box::new(move || {
                        if saic || daic {
                            for i in 0..bytes / 4 {
                                let s = if saic { src } else { src + 4 * i };
                                let d = if daic { dst } else { dst + 4 * i };
                                unsafe { wr32(d, rd32(s)) };
                            }
                        } else {
                            unsafe { std::ptr::copy(src as *const u8, dst as *mut u8, bytes as usize) };
                            // Shared copies checked in this epoch are stale now.
                            super::precache::invalidate(dst, bytes);
                        }
                        note_block(&name, dst, if daic { 4 } else { bytes }, at);
                    }),
                );
                if let Some(s) = seq {
                    super::exec::note_cp_source(s, (src, if saic { 4 } else { bytes }));
                }
            }
            _ => {
                // Register or GDS source: register values / counters.
                let v = if sas { self.reg_by_addr(src as u32) } else { 0 };
                self.note("DMA_DATA from registers/GDS: register value or 0");
                super::replay::note_cp_write(dst, if daic { 4 } else { bytes });
                super::forward::written(dst, if daic { 4 } else { bytes });
                for i in 0..(bytes / 4).max(1) {
                    unsafe { wr32(if daic { dst } else { dst + 4 * i }, v) };
                }
                cpw_note(&self.name, dst, if daic { 4 } else { bytes }, self.cur_packet, self.cur_opcode);
            }
        }
    }

    fn release_mem(&mut self, p: &dyn Fn(u64) -> u32) {
        if let Some(e) = self.exec.as_mut() {
            e.note_sync();
        }
        let data_cntl = p(1);
        let int_sel = (data_cntl >> 24) & 7;
        let data_sel = (data_cntl >> 29) & 7;
        RELEASE_KINDS[(data_sel * 8 + int_sel) as usize].fetch_add(1, Ordering::Relaxed);
        let a = addr64(p(2) & !3, p(3));
        let data = addr64(p(4), p(5));
        let ctx = p(6);
        if data_sel > 4 {
            self.note("RELEASE_MEM with GDS data ignored");
        }
        let write = matches!(data_sel, 1..=4);
        if write && !valid(a, 8) {
            crate::error!("gpu[{}]: RELEASE_MEM to {a:#x} outside guest memory", self.name);
            return;
        }
        let (name, at) = (self.name.clone(), self.cur_packet);
        if matches!(data_sel, 1 | 2) && (int_sel == 0 || int_sel == 3) {
            let (vals, label) = if data_sel == 1 { (vec![data as u32], (data & 0xFFFF_FFFF, 4u8)) } else { (vec![data as u32, (data >> 32) as u32], (data, 8u8)) };
            if self.gpu_cp_write(a, &vals, Some(label), true) {
                return;
            }
        }
        // Neither a write nor an interrupt: nothing the CPU would do after the
        // GPU work (deferring it ended the batch for an empty action).
        if !write && (int_sel == 0 || int_sel == 3) {
            return;
        }
        if matches!(data_sel, 3 | 4) && (int_sel == 0 || int_sel == 3) && super::tune::gpu_cp_writes() {
            if let Some(e) = self.exec.as_mut() {
                if e.busy() && e.gpu_timestamp(a) {
                    self.fence_pending = false;
                    return;
                }
            }
        }
        let seq = self.effect(
            true,
            write.then_some((a, 8)),
            Box::new(move || {
                if write {
                    match data_sel {
                        1 => unsafe { wr32(a, data as u32) },
                        2 => unsafe { wr64(a, data) },
                        _ => unsafe { wr64(a, gpu_clock()) },
                    }
                    note_write(&name, a, data, at);
                }
                // Not for a replayed frame (`gpu::replay`): the title is held.
                if int_sel != 0 && int_sel != 3 && !super::replay::replaying() {
                    EOP_INTERRUPTS.fetch_add(1, Ordering::Relaxed);
                    crate::hle::libs::agcdriver::raise_eop(ctx);
                }
            }),
        );
        if let Some(s) = seq {
            match data_sel {
                1 => promise(s, a, data & 0xFFFF_FFFF, 4, true),
                2 => promise(s, a, data, 8, true),
                _ => {}
            }
        }
    }

    /// Polls `addr` until the comparison holds (WAIT_REG_MEM).
    fn wait(&mut self, control: u32, a: u64, wide: bool, reference: u64, mask: u64, at: u64) -> Flow {
        let func = control & 7;
        let mem = (control >> 4) & 3 == 1;
        if !mem {
            let v = self.reg_by_addr(a as u32) as u64;
            if !compare(func, v & mask, reference & mask) {
                self.note("WAIT_REG_MEM on a register that does not match: continuing");
            }
            return Flow::Next;
        }
        if !valid(a, 8) {
            crate::error!("gpu[{}]: WAIT_REG_MEM on {a:#x} outside guest memory", self.name);
            return Flow::Next;
        }
        let read = move || if wide { unsafe { rd64(a) } } else { (unsafe { rd32(a) }) as u64 };
        WAITED.lock().get_or_insert_with(HashSet::new).insert(a & !7);
        self.last_wait_gpu = false;
        self.last_wait_kind = b'-';
        let v0 = read();
        if compare(func, v0 & mask, reference & mask) {
            self.last_wait_kind = b'I';
            label_event(&self.name, b'I', a, v0, reference, at);
            return Flow::Next;
        }
        if let Some((seq, by_cpu)) = promised(a, wide, func, mask, reference) {
            if let Some(e) = self.exec.as_mut() {
                e.depend_on(seq, by_cpu);
                PROMISED.fetch_add(1, Ordering::Relaxed);
                self.last_wait_kind = b'P';
                label_event(&self.name, b'P', a, v0, reference, at);
                self.last_wait_gpu = true;
                return Flow::Next;
            }
        }
        // A matching write made after this command buffer was submitted:
        // hardware queues reach their waits at once and would have seen it.
        // Ours fall behind at times (pipeline compilation) and then waited
        // for a value already gone, e.g. a frame counter compared for
        // equality (run 20261011-062446: acb0x48 waited for 0x7d8 at 0x7dc,
        // and every queue stopped).
        if self.submitted_at.is_some_and(|since| missed_write(a, wide, func, mask, reference, since)) {
            MISSED.fetch_add(1, Ordering::Relaxed);
            self.last_wait_kind = b'M';
            label_event(&self.name, b'M', a, v0, reference, at);
            return Flow::Next;
        }
        // A value this queue's recording batch writes on the GPU: its later
        // work follows that write in the batch (`tune gpu_cp_writes`).
        if let Some((v, bytes)) = self.exec.as_ref().and_then(|e| e.open_promise(a)) {
            let v = if wide { v } else { v & 0xFFFF_FFFF };
            if !(wide && bytes < 8) && compare(func, v & mask, reference & mask) {
                OPEN_PROMISED.fetch_add(1, Ordering::Relaxed);
                // Later work follows that write (end-of-pipe writes do not
                // order it by themselves: `tune eop_light`).
                if let Some(e) = self.exec.as_mut() {
                    e.order_after_open_writes();
                }
                self.last_wait_kind = b'P';
                label_event(&self.name, b'P', a, v0, reference, at);
                self.last_wait_gpu = true;
                return Flow::Next;
            }
        }
        // An equality wait on a counter that has moved past the value (its
        // command buffer came after the write: run 20261011-082829, acb0x48
        // waited for 0x706 at 0x70a while pipelines compiled).
        if counter_passed(a, wide, func, mask, reference) {
            if COUNTER_PASSED.fetch_add(1, Ordering::Relaxed) < 20 {
                crate::info!("gpu[{}]: WAIT_REG_MEM at {at:#x} for [{a:#x}] == {reference:#x} passed: the counter is at {v0:#x}", self.name);
            }
            self.last_wait_kind = b'M';
            label_event(&self.name, b'M', a, v0, reference, at);
            return Flow::Next;
        }
        // In a replayed frame, a wait the title's CPU released when the frame
        // ran passes (`gpu::replay`).
        if super::replay::pass_wait(at, a) {
            self.last_wait_kind = b'M';
            label_event(&self.name, b'M', a, v0, reference, at);
            return Flow::Next;
        }
        label_event(&self.name, b'S', a, v0, reference, at);
        // Let recorded work and deferred effects progress while polling; a
        // queue holding this label in an unsubmitted batch submits it.
        self.submit_gpu("wait");
        super::exec::request_label_submit(a);
        let what = format!("WAIT_REG_MEM at {at:#x}: [{a:#x}] & {mask:#x} func {func} ref {reference:#x}");
        self.waiting_on = a;
        // Submitters stop waiting for this queue only when the label is not
        // written by its own outstanding GPU work (it may come from them).
        let own = self.exec.as_mut().is_some_and(|e| e.will_write(a, if wide { 8 } else { 4 }));
        if let (Some(f), false) = (self.wait_flag, own) {
            f.store(true, Ordering::SeqCst);
        }
        let hit = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        ACTIVE_WAITS.lock().push(ActiveWait { addr: a, wide, func, mask, reference, hit: hit.clone() });
        ACTIVE_COUNT.fetch_add(1, Ordering::AcqRel);
        let t0 = Instant::now();
        let tw = super::timeline::wait_start();
        let use_promise = self.exec.is_some();
        let promises_seen = std::cell::Cell::new(u64::MAX);
        let promises_moved = || {
            let g = PROMISE_GEN.load(Ordering::Acquire);
            promises_seen.replace(g) != g
        };
        // The stages of the commands after this wait are prepared meanwhile
        // (`tune wait_ahead`; checked when used as any made ahead), those
        // of an earlier wait's scan first.
        let mut pending: Vec<Arc<super::ahead::Slot>> = Vec::new();
        if let Some((f, from, end)) = self.wait_ctx.take() {
            if self.ahead.is_empty() {
                let cmds = self.scan_through(&f, from, end);
                self.ahead.extend(cmds);
            }
            pending = self.ahead.iter().flat_map(|c| c.stages.iter().flatten().cloned()).collect();
        }
        let mut next_slot = 0usize;
        let compile_ahead = super::compile::at_waits();
        let mut work = || {
            while next_slot < pending.len() {
                next_slot += 1;
                let slot = &pending[next_slot - 1];
                if super::ahead::prepare_here(slot) {
                    // A compute stage whose pipeline is missing: its compile
                    // starts now, on a compile worker (`tune compile_ahead`).
                    if compile_ahead {
                        if let Some(p) = super::ahead::prepared_compute(slot) {
                            if !super::exec::compute_pipe_known(&p) {
                                super::compile::submit_prepared(p);
                            }
                        }
                    }
                    return true;
                }
            }
            false
        };
        // A label a GPU queue wrote last: its next value comes with a
        // promise, a latched CP write or a GPU-performed write seen
        // (`wake_label`).
        let gpu_label = last_write(a).is_some_and(|(q, _, _)| q.starts_with("gpu-"));
        // Measurement only (`tune xwait_skip`, unsafe): the graphics CP does
        // not wait for labels compute queue 0x50 wrote last (an upper bound
        // for recording past them; what follows may read stale data).
        if super::tune::xwait_skip() && self.name == "gpu-gfx" && last_write(a).is_some_and(|(q, _, _)| q == "gpu-acb0x50") {
            self.note("`tune xwait_skip`: label waits for gpu-acb0x50 skipped (measurement; pictures may be wrong)");
            return Flow::Next;
        }
        let probe = super::xwait::on() && self.name == "gpu-gfx";
        if probe {
            if let Some((q, _, _)) = last_write(a) {
                super::xwait::wait_start(&q);
            }
        }
        let r = self.poll_until_wake(
            at,
            &what,
            || hit.load(Ordering::SeqCst) || compare(func, read() & mask, reference & mask) || (use_promise && promises_moved() && promised(a, wide, func, mask, reference).is_some()),
            &mut work,
            (gpu_label && super::tune::label_wake() && a % if wide { 8 } else { 4 } == 0).then_some((a, wide)),
        );
        if probe {
            super::xwait::wait_end();
        }
        {
            let by = last_write(a).map(|w| w.0).unwrap_or_else(|| "cpu".into());
            tw.end(&self.name, || format!("wait: label {a:#x} (written by {by})"));
            let mut g = WAIT_STATS.lock();
            let e = g.get_or_insert_with(HashMap::new).entry((self.name.clone(), by)).or_default();
            e.0 += 1;
            e.1 += t0.elapsed().as_nanos() as u64;
        }
        ACTIVE_WAITS.lock().retain(|w| !std::sync::Arc::ptr_eq(&w.hit, &hit));
        ACTIVE_COUNT.fetch_sub(1, Ordering::AcqRel);
        if let Some(f) = self.wait_flag {
            f.store(false, Ordering::SeqCst);
        }
        let in_memory = compare(func, read() & mask, reference & mask);
        let latched = hit.load(Ordering::SeqCst) && !in_memory;
        if latched {
            LATCHED.fetch_add(1, Ordering::Relaxed);
        }
        // Passed on a promise (its batch may have completed meanwhile).
        let mut kind = if latched { b'L' } else { b'E' };
        if !in_memory && !latched {
            if let (Some((seq, by_cpu)), Some(e)) = (promised(a, wide, func, mask, reference), self.exec.as_mut()) {
                e.depend_on(seq, by_cpu);
                PROMISED.fetch_add(1, Ordering::Relaxed);
                kind = b'P';
            }
        }
        self.last_wait_kind = kind;
        label_event(&self.name, kind, a, read(), reference, at);
        self.last_wait_gpu = kind == b'P' || kind == b'L';
        long_wait(&self.name, a, wide, func, mask, reference, at, t0, kind);
        match r {
            Flow::Stop => Flow::Stop,
            _ => Flow::Next,
        }
    }

    /// Spins (then sleeps) until `cond`; reports long waits once.
    fn poll_until(&mut self, at: u64, what: &str, cond: impl Fn() -> bool) -> Flow {
        self.poll_until_wake(at, what, cond, &mut || false, None)
    }

    /// `poll_until`, calling `work` between the checks while it finds
    /// something to do (it returns whether it did).
    /// `wake` (a label's address and width): after its spins the wait sleeps
    /// on the label's address (`wake_label`; at most 1 ms at a time) instead
    /// of yielding and sleeping 100 us between checks.
    fn poll_until_wake(&mut self, at: u64, what: &str, cond: impl Fn() -> bool, work: &mut dyn FnMut() -> bool, wake: Option<(u64, bool)>) -> Flow {
        let start = Instant::now();
        let mut reported = false;
        set_state(&self.name, what.to_string());
        let mut spins = 0u32;
        let yields = 200 + super::tune::wait_yields().min(100_000) as u32;
        while !cond() {
            if crate::gpu::shutting_down() {
                return Flow::Stop;
            }
            if super::replay::force_wait(at, start, what) {
                break;
            }
            if work() {
                continue;
            }
            spins += 1;
            if spins < 200 {
                std::hint::spin_loop();
            } else if let Some((a, wide)) = wake {
                // The value now: a write after this wakes the sleep too.
                let seen: u64 = unsafe { if wide { rd64(a) } else { rd32(a) as u64 } };
                if cond() {
                    break;
                }
                WAKE_SLEEPS.fetch_add(1, Ordering::Relaxed);
                unsafe { windows_sys::Win32::System::Threading::WaitOnAddress(a as *const core::ffi::c_void, &seen as *const u64 as *const core::ffi::c_void, if wide { 8 } else { 4 }, 1) };
            } else if spins < yields {
                std::thread::yield_now();
            } else {
                crate::thread::short_sleep(Duration::from_micros(100));
            }
            if !reported && start.elapsed() > Duration::from_secs(3) {
                reported = true;
                let marker = self.markers.last().cloned().unwrap_or_default();
                let a = self.waiting_on;
                let detail = if a != 0 {
                    let cur = unsafe { rd64(a) };
                    match last_write(a) {
                        Some((q, v, pk)) => format!("; value now {cur:#x}, last GPU write {v:#x} by {q} (packet {pk:#x})"),
                        None => format!("; value now {cur:#x}, never written by the GPU"),
                    }
                } else {
                    String::new()
                };
                let history = if a != 0 { format!("; recorded writes {}; label history:{}", label_writes(a), label_history(a, 16)) } else { String::new() };
                crate::warn!("gpu[{}]: waiting > 3 s: {what}{}{detail}{history}", self.name, if marker.is_empty() { String::new() } else { format!(" (marker '{marker}')") });
            }
        }
        set_state(&self.name, "running".into());
        self.waiting_on = 0;
        let _ = at;
        Flow::Next
    }

    fn private(&mut self, operation: u32, at: u64, n: u64, a: &dyn Fn(u64) -> u32) -> Flow {
        match operation {
            // A replayed frame (`gpu::replay`) is not presented (its work
            // completes here as for a flip).
            pop::FLIP if super::replay::replaying() => self.sync(),
            pop::FLIP if super::tune::flip_async() => {
                // Queued now, shown once this queue's GPU work before it
                // completed (as the hardware's CP queues it) instead of the
                // command processor waiting for that work here.
                let handle = a(0) as i32;
                let index = a(1) as i32;
                let arg = addr64(a(3), a(4)) as i64;
                let mut r = crate::hle::libs::videoout::reserve_flip(handle, index, arg);
                if r == Err(crate::hle::libs::videoout::ERR_FLIP_QUEUE_FULL) {
                    set_state(&self.name, format!("flip of buffer {index}: waiting for a free flip slot"));
                    while r == Err(crate::hle::libs::videoout::ERR_FLIP_QUEUE_FULL) && !crate::gpu::shutting_down() {
                        std::thread::sleep(Duration::from_micros(500));
                        r = crate::hle::libs::videoout::reserve_flip(handle, index, arg);
                    }
                    set_state(&self.name, "running".into());
                }
                match r {
                    Ok(ticket) => {
                        self.effect(true, None, Box::new(move || crate::hle::libs::videoout::commit_flip(ticket)));
                    }
                    Err(e) => crate::warn!("gpu[{}]: flip of buffer {index} (handle {handle:#x}) failed ({e:#x})", self.name),
                }
                let n = crate::gpu::FRAMES.fetch_add(1, Ordering::Relaxed) + 1;
                super::capture::on_flip(n);
            }
            pop::FLIP => {
                self.sync();
                let handle = a(0) as i32;
                let index = a(1) as i32;
                let arg = addr64(a(3), a(4)) as i64;
                // A full flip queue stalls the queue (as on hardware), which
                // paces the title through its label waits.
                let mut r = crate::hle::libs::videoout::submit_flip(handle, index, arg);
                if r == crate::hle::libs::videoout::ERR_FLIP_QUEUE_FULL {
                    set_state(&self.name, format!("flip of buffer {index}: waiting for a free flip slot"));
                    while r == crate::hle::libs::videoout::ERR_FLIP_QUEUE_FULL && !crate::gpu::shutting_down() {
                        std::thread::sleep(Duration::from_micros(500));
                        r = crate::hle::libs::videoout::submit_flip(handle, index, arg);
                    }
                    set_state(&self.name, "running".into());
                }
                if r != 0 {
                    // The packet as executed and as it is in memory now (a
                    // command buffer reused before execution shows here).
                    let words: Vec<String> = (0..n.min(9)).map(|k| format!("{:08x}", a(k))).collect();
                    let now: Vec<String> = (0..n.min(9) + 3).map(|k| at + 4 * k).filter(|&w| guest_readable(w, 4)).map(|w| format!("{:08x}", unsafe { rd32(w) })).collect();
                    crate::warn!("gpu[{}]: flip of buffer {index} (handle {handle:#x}) failed ({r:#x}); packet at {at:#x}: args {} (memory now: {}), submitted {} ms ago", self.name, words.join(" "), now.join(" "), self.submitted_at.map_or(0, |t| t.elapsed().as_millis()));
                }
                let n = crate::gpu::FRAMES.fetch_add(1, Ordering::Relaxed) + 1;
                super::capture::on_flip(n);
            }
            pop::WAIT_SAFE_FOR_RENDERING => {
                let (handle, index) = (a(0) as i32, a(1) as i32);
                if crate::hle::libs::videoout::buffer_busy(handle, index) && !super::replay::replaying() {
                    let what = format!("WaitUntilSafeForRendering: display buffer {index} is displayed or queued");
                    return self.poll_until(at, &what, || !crate::hle::libs::videoout::buffer_busy(handle, index));
                }
            }
            pop::LOD_STATS => {
                // Hardware LOD statistics (texture streaming feedback): the
                // title reads, per counter bank (T# word 6), the 8-byte entry
                // at 0x40 + 8 * bank: bits 56..59 the finest mip sampled (15:
                // none), bits 32..55 a count (eboot+0xac90b0).
                let (buf, size, flags) = (addr64(a(0), a(1)), a(2) as u64, a(3));
                let mode = super::tune::lod_stats();
                if buf != 0 {
                    static SEEN: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                    let n = SEEN.fetch_add(1, Ordering::Relaxed);
                    let writable = size >= super::lodstats::BUFFER_BYTES && guest_readable(buf, super::lodstats::BUFFER_BYTES);
                    if mode == 3 && writable && self.exec.is_some() {
                        // Copied after the work recorded so far, written to the
                        // title's buffer once that completed (before the label
                        // writes behind it).
                        let e = self.exec.as_mut().unwrap();
                        let slot = super::lodstats::take_slot();
                        e.lod_stats_copy(slot, (flags >> 19) & 1 != 0);
                        let name = self.name.clone();
                        let log = n < 2 || n.is_power_of_two() || n % 1000 == 0;
                        let f: Box<dyn FnOnce() + Send> = Box::new(move || {
                            let words = super::exec::lod_stats_words(slot);
                            let (sampled, by_mip) = unsafe { super::lodstats::write_measured(buf, &words, slot) };
                            if log {
                                let mips: Vec<String> = by_mip.iter().enumerate().filter(|&(_, &c)| c > 0).map(|(m, c)| format!("{m}:{c}")).collect();
                                crate::info!("gpu[{name}]: GetLodStats #{n} (buffer {buf:#x}, flags {flags:#x}): {sampled} of 256 counter banks sampled (finest mip:banks {}; measured on the GPU)", mips.join(" "));
                            }
                        });
                        e.after_gpu(f, Some((buf, super::lodstats::BUFFER_BYTES)), false);
                    } else if mode != 0 && writable {
                        // Before the label write behind it (a deferred effect).
                        let sampled = unsafe { super::lodstats::report(buf, mode, (flags >> 19) & 1 != 0) };
                        if n < 2 || n.is_power_of_two() {
                            crate::info!("gpu[{}]: GetLodStats #{n} (buffer {buf:#x}, flags {flags:#x}): {sampled} of 256 counter banks reported as sampled at mip 0 (mode {mode}; stand-in for the hardware counters)", self.name);
                        }
                    } else if n < 2 || n.is_power_of_two() {
                        crate::warn!("gpu[{}]: GetLodStats #{n} (buffer {buf:#x} size {size:#x}, flags {flags:#x}) not written (mode {mode}): the title streams no high mips", self.name);
                    }
                }
            }
            pop::WORKLOADS_ACTIVE | pop::WORKLOAD_COMPLETE => {}
            pop::CONTEXT_STATE => match a(0) {
                0 => self.clear_context(),
                1 => self.ctx_stack.push(self.regs.ctx.clone()),
                2 => match self.ctx_stack.pop() {
                    Some(c) => {
                        self.regs.ctx = c;
                        self.ctx_dirty = true;
                    }
                    None => crate::warn!("gpu[{}]: context state pop without push at {at:#x}", self.name),
                },
                3 => {
                    self.ctx_stack.push(self.regs.ctx.clone());
                    self.clear_context();
                }
                other => crate::warn!("gpu[{}]: unknown context state operation {other}", self.name),
            },
            pop::ACB_RESET => {
                self.regs.sh.fill(0);
            }
            pop::PUSH_MARKER | pop::SET_MARKER => {
                let mut bytes = Vec::new();
                for i in 1..n {
                    bytes.extend_from_slice(&a(i).to_le_bytes());
                }
                let s = String::from_utf8_lossy(&bytes[..bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len())]).into_owned();
                if operation == pop::SET_MARKER {
                    self.markers.pop();
                }
                self.markers.push(s);
                if self.markers.len() > 64 {
                    self.markers.remove(0);
                }
            }
            pop::POP_MARKER => {
                self.markers.pop();
            }
            pop::BRANCH => {
                self.sync();
                let ctl = a(0);
                let cmp = addr64(a(1), a(2));
                self.sync_reads(&[(cmp, 8)]);
                let mask = addr64(a(3), a(4));
                let reference = addr64(a(5), a(6));
                let pass = valid(cmp, 8) && compare((ctl >> 8) & 7, unsafe { rd64(cmp) } & mask, reference & mask);
                let (lo, hi, s) = if pass { (a(7), a(8), a(9)) } else { (a(10), a(11), a(12)) };
                let target = addr64(lo, hi);
                let size = s & 0xF_FFFF;
                if target != 0 && size != 0 {
                    self.run(target, size);
                }
            }
            pop::DRAW_INDEX_MULTI_INSTANCED => {
                DRAWS.fetch_add(1, Ordering::Relaxed);
                self.note("multi-instanced draws are not rendered yet");
            }
            _ => {
                crate::warn!("gpu[{}]: unknown private packet op {operation} at {at:#x}", self.name);
            }
        }
        Flow::Next
    }
}

enum Flow {
    Next,
    Skip(u32),
    Chain(u64, u32),
    Stop,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn history(addr: u64, values: &[u64]) {
        let mut g = LABEL_WRITES.lock();
        let h = g.get_or_insert_with(HashMap::new).entry(addr).or_default();
        h.clear();
        h.extend(values.iter().map(|&v| (Instant::now(), v)));
    }

    /// Equality waits on frame counters that moved on pass; waits on flags,
    /// on values not yet reached and on counters that jump do not.
    #[test]
    fn counter_passed_rule() {
        let m = 0xFFFF_FFFF;
        history(0x1000, &[0x7fe, 0x7ff, 0x800, 0x801, 0x802, 0x803, 0x804, 0x805]);
        assert!(counter_passed(0x1000, false, 3, m, 0x801));
        assert!(!counter_passed(0x1000, false, 3, m, 0x804), "one step behind may still come");
        assert!(!counter_passed(0x1000, false, 3, m, 0x806), "not reached yet");
        assert!(!counter_passed(0x1000, false, 2, m, 0x801), "only equality waits");
        // Counting by 2 (run 20261011-105452: 0x82c 0x82e 0x830 0x832).
        history(0x1008, &[0x82c, 0x82e, 0x830, 0x832, 0x834]);
        assert!(counter_passed(0x1008, false, 3, m, 0x82e));
        assert!(!counter_passed(0x1008, false, 3, m, 0x832));
        // The waited value before the recorded window.
        history(0x1010, &[0x802, 0x803, 0x803, 0x804, 0x805]);
        assert!(counter_passed(0x1010, false, 3, m, 0x801));
        assert!(!counter_passed(0x1010, false, 3, m, 0x700), "too far behind the window");
        // A flag (0 -> 1 -> 0) is not a counter.
        history(0x1018, &[0, 1, 0, 1]);
        assert!(!counter_passed(0x1018, false, 3, m, 0));
        // A counter that was reset is not trusted.
        history(0x1020, &[0x10, 0x11, 0x0, 0x1, 0x2, 0x3]);
        assert!(!counter_passed(0x1020, false, 3, m, 0x11));
    }

    #[test]
    fn register_snapshots_hold_the_written_part() {
        // A snapshot's registers, then zeros up to the current high mark
        // (written after the snapshot), the rest untouched (never written).
        let mut dst = vec![7u32; 8];
        copy_prefix(&mut dst, &[1, 2, 3], 6);
        assert_eq!(dst, [1, 2, 3, 0, 0, 0, 7, 7]);
        // Writes to a snapshot grow it and raise the mark; beyond the file
        // they are ignored.
        let mut v = vec![1u32, 2];
        put_reg(&mut v, Space::Sh, 4, 3, 9);
        assert_eq!(v, [1, 2, 0, 9]);
        assert!(high(Space::Sh, SH_DWORDS) >= 4);
        put_reg(&mut v, Space::Sh, 4, 4, 5);
        assert_eq!(v.len(), 4);
    }

    #[test]
    fn register_ranges_copy_like_single_writes() {
        // A SET_*_REG payload at 0x1000 (fetched): registers 3.. get 10, 11,
        // 12; a range past the file's end stops at it.
        let f = Fetched { base: 0x1000, words: vec![10, 11, 12, 13, 14], live: false };
        let mut a = vec![7u32; 2];
        put_regs(&mut a, Space::Ctx, 64, 3, &f, 0x1000, 3);
        let mut b = vec![7u32; 2];
        for j in 0..3 {
            put_reg(&mut b, Space::Ctx, 64, 3 + j, 10 + j as u32);
        }
        assert_eq!(a, b);
        assert_eq!(a, [7, 7, 0, 10, 11, 12]);
        let mut c = Vec::new();
        put_regs(&mut c, Space::Ctx, 5, 3, &f, 0x1004, 4);
        assert_eq!(c, [0, 0, 0, 11, 12]);
        assert_eq!(f.slice(0x1008, 3), Some(&[12u32, 13, 14][..]));
        assert_eq!(f.slice(0x1010, 2), None, "straddles the fetched words");
    }

    #[test]
    fn chunks_split_at_equal_cost() {
        // Commands 0..4 are the main command processor's; 4..16 go to three
        // workers. The last four cost three times the others.
        let mut c = vec![1.0f32; 16];
        for x in c.iter_mut().skip(12) {
            *x = 3.0;
        }
        // Total 8 + 12 = 20 over three chunks: cuts where ~6.7 and ~13.3
        // are reached.
        assert_eq!(cost_firsts(&c, 4, 16, 3, 2), Some(vec![4, 11, 14]));
        // Equal costs: equal counts.
        assert_eq!(cost_firsts(&[1.0; 16], 4, 16, 3, 2), Some(vec![4, 8, 12]));
        // Chunks keep their minimum size.
        let mut d = vec![1.0f32; 16];
        d[4] = 100.0;
        assert_eq!(cost_firsts(&d, 4, 16, 3, 3), Some(vec![4, 7, 10]));
        // Unmeasured commands: no split by cost.
        let mut e = vec![1.0f32; 16];
        e[9] = f32::NAN;
        assert_eq!(cost_firsts(&e, 4, 16, 3, 2), None);
        // Main's part may stay unmeasured.
        let mut g = vec![f32::NAN; 16];
        for x in g.iter_mut().skip(4) {
            *x = 1.0;
        }
        assert_eq!(cost_firsts(&g, 4, 16, 3, 2), Some(vec![4, 8, 12]));
    }
}
