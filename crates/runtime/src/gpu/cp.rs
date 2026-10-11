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
    format!("{waits}; deferred CP effects {deferred}")
}

/// Register file of one queue.
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

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Space {
    Ctx,
    Sh,
    Uc,
}

pub struct Cp {
    pub name: String,
    /// Outcome of the last draw (frame captures).
    pub last_draw: String,
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
    exec: Option<super::exec::Exec>,
    exec_failed: bool,
    /// A partial flush / acquire was requested: the next CP memory write
    /// must wait for the GPU work before it.
    fence_pending: bool,
    /// SET_PREDICATION state for packets with the predicate bit.
    pred: Option<Pred>,
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
    unsafe { std::ptr::write_volatile(a as *mut u32, v) }
}
unsafe fn wr64(a: u64, v: u64) {
    unsafe { std::ptr::write_volatile(a as *mut u64, v) }
}

fn addr64(lo: u32, hi: u32) -> u64 {
    lo as u64 | ((hi as u64 & 0xFFFF_FFFF) << 32)
}

/// Guest addresses the command processor may touch.
fn valid(a: u64, len: u64) -> bool {
    a >= crate::mem::GUEST_BASE && a.checked_add(len).is_some_and(|e| e <= crate::mem::GUEST_END)
}

/// Whether guest memory [a, a+len) is mapped and readable.
pub fn guest_readable(a: u64, len: u64) -> bool {
    valid(a, len) && crate::mem::mapped(a, len)
}

/// GPU timestamp (100 MHz, like the console's GPU clock counter).
pub fn gpu_clock() -> u64 {
    static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    let s = START.get_or_init(Instant::now);
    (s.elapsed().as_nanos() / 10) as u64
}

/// Command buffers up to this size are copied when execution starts.
const FETCH_DWORDS: u32 = 512;

/// A command buffer as the CP fetched it (see `Cp::run`): small buffers are
/// copied, larger ones read from guest memory.
struct Fetched {
    base: u64,
    words: Vec<u32>,
}

impl Fetched {
    fn new(base: u64, size: u32) -> Fetched {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let on = *ON.get_or_init(|| std::env::var("DS_CP_FETCH").ok().as_deref() != Some("0"));
        let n = size as usize;
        let words = if on && size > 0 && size <= FETCH_DWORDS && valid(base, 4 * n as u64) { unsafe { std::slice::from_raw_parts(base as *const u32, n) }.to_vec() } else { Vec::new() };
        Fetched { base, words }
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
static LAST_WRITE: parking_lot::Mutex<Option<std::collections::HashMap<u64, (String, u64, u64)>>> = parking_lot::Mutex::new(None);

fn note_write(queue: &str, addr: u64, value: u64, at: u64) {
    {
        let mut g = LAST_WRITE.lock();
        let m = g.get_or_insert_with(Default::default);
        if m.len() > 1 << 16 {
            m.clear();
        }
        m.insert(addr & !7, (queue.to_string(), value, at));
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
/// Waits satisfied by a latched CP write whose value was gone by the time
/// the waiter looked again (diagnostics).
pub static LATCHED: AtomicU64 = AtomicU64::new(0);

/// Latches the active waits overlapping [addr, addr+bytes) that the
/// current memory contents satisfy (called right after a CP write).
fn latch_waits(addr: u64, bytes: u64) {
    let g = ACTIVE_WAITS.lock();
    for w in g.iter() {
        let n = if w.wide { 8 } else { 4 };
        if w.addr < addr + bytes && addr < w.addr + n {
            let v = if w.wide { unsafe { rd64(w.addr) } } else { (unsafe { rd32(w.addr) }) as u64 };
            if compare(w.func, v & w.mask, w.reference & w.mask) {
                w.hit.store(true, Ordering::SeqCst);
            }
        }
    }
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
/// value, bytes). A WAIT_REG_MEM that a promised value satisfies passes at
/// once and orders its queue after that batch (`Exec::depend_on`): its later
/// GPU work follows the batch on the device queue, its CP writes wait for
/// it, its CP reads wait for unfinished writers (`Exec::sync_for_reads`), so
/// the command processors record ahead of the GPU instead of idling until
/// each batch completes. `DS_PROMISES=0` turns this off.
static PROMISES: parking_lot::Mutex<Option<HashMap<u64, (u64, u64, u8)>>> = parking_lot::Mutex::new(None);
/// Waits passed on a promise (diagnostics).
pub static PROMISED: AtomicU64 = AtomicU64::new(0);

fn promises_on() -> bool {
    super::tune::promises()
}

pub(super) fn promise(seq: u64, addr: u64, value: u64, bytes: u8) {
    if !promises_on() {
        return;
    }
    let mut g = PROMISES.lock();
    let m = g.get_or_insert_with(HashMap::new);
    if m.len() > 1 << 14 {
        let done = super::exec::completed_seq();
        m.retain(|_, p| p.0 > done);
    }
    m.insert(addr, (seq, value, bytes));
}

/// The batch whose promised write to `addr` satisfies the comparison.
fn promised(addr: u64, wide: bool, func: u32, mask: u64, reference: u64) -> Option<u64> {
    let g = PROMISES.lock();
    let &(seq, value, bytes) = g.as_ref()?.get(&addr)?;
    if seq <= super::exec::completed_seq() || (wide && bytes < 8) {
        return None;
    }
    let v = if wide { value } else { value & 0xFFFF_FFFF };
    compare(func, v & mask, reference & mask).then_some(seq)
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
            _ => format!("wait ref {:#x}: satisfied (value {:#x})", e.reference, e.value),
        };
        out.push_str(&format!("\n    {:>10.3} ms {} {what} (packet {:#x})", e.us as f64 / 1000.0, e.queue, e.packet));
    }
    out
}

/// Records the label-sized words of a block the CP wrote (DMA fills and
/// copies up to 16 KiB; larger blocks are data, not labels).
fn note_block(queue: &str, addr: u64, bytes: u64, at: u64) {
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
}

/// Blocking label waits by waiter and writer, longest total first.
pub fn wait_stats() -> String {
    let g = WAIT_STATS.lock();
    let Some(m) = g.as_ref() else { return String::new() };
    let mut v: Vec<(&(String, String), &(u64, u64))> = m.iter().collect();
    v.sort_by(|a, b| b.1 .1.cmp(&a.1 .1));
    let mut s = String::from("    label waits (waiter <- last writer: count, total ms, mean us):\n");
    for ((w, by), (n, ns)) in v.iter().take(24) {
        s.push_str(&format!("      {w} <- {by}: {n}, {} ms, {} us\n", ns / 1_000_000, ns / 1000 / (*n).max(1)));
    }
    s
}

fn last_write(addr: u64) -> Option<(String, u64, u64)> {
    LAST_WRITE.lock().as_ref()?.get(&(addr & !7)).cloned()
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
            last_draw: String::new(),
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
        }
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
        let needs_prior = end_of_pipe || self.fence_pending;
        let Some(e) = self.exec.as_mut() else { return false };
        if !(e.busy() && needs_prior) || e.deferred_pending() || !e.gpu_cp_write(a, data, label) {
            return false;
        }
        self.fence_pending = false;
        // Diagnostics (no latching: the value is not in memory yet).
        let mut g = LAST_WRITE.lock();
        let m = g.get_or_insert_with(Default::default);
        if m.len() > 1 << 16 {
            m.clear();
        }
        m.insert(a & !7, (self.name.clone(), label.map(|l| l.0).unwrap_or(data[0] as u64), self.cur_packet));
        true
    }

    fn effect(&mut self, end_of_pipe: bool, write: Option<(u64, u64)>, f: Box<dyn FnOnce() + Send>) -> Option<u64> {
        let needs_prior = end_of_pipe || self.fence_pending;
        match self.exec.as_mut() {
            Some(e) if e.busy() && (needs_prior || e.deferred_pending()) => {
                let k = (self.cur_opcode as usize).min(255);
                let why = if end_of_pipe { 0 } else if self.fence_pending { 1 } else { 2 };
                DEFERRED_BY_OPCODE[k][why].fetch_add(1, Ordering::Relaxed);
                self.fence_pending = false;
                e.after_gpu(f, write, !end_of_pipe)
            }
            _ => {
                if needs_prior {
                    self.fence_pending = false;
                }
                f();
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
                Ok(e) => self.exec = Some(e),
                Err(e) => {
                    crate::error!("gpu[{}]: no executor: {e}", self.name);
                    self.exec_failed = true;
                    return;
                }
            }
        }
        let mut e = self.exec.take().unwrap();
        e.dispatch(self, groups, initiator);
        self.exec = Some(e);
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
        for &(i, val) in d {
            self.set_reg(Space::Ctx, i, val);
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
                Ok(e) => self.exec = Some(e),
                Err(e) => {
                    crate::error!("gpu[{}]: no executor: {e}", self.name);
                    self.exec_failed = true;
                    return;
                }
            }
        }
        let mut e = self.exec.take().unwrap();
        self.last_draw = e.draw(self, args);
        self.exec = Some(e);
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
        if p(3) & (1 << 30) != 0 {
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
                (super::exec::DrawArgs { count: w[0], instances: w[1], index: Some((self.index_base + w[2] as u64 * sz as u64, sz)), base_vertex: Some(w[3] as i32), first_instance: w[4] }, w[3], w[4])
            } else {
                (super::exec::DrawArgs { count: w[0], instances: w[1], index: None, base_vertex: Some(w[2] as i32), first_instance: w[3] }, w[2], w[3])
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
            outcome = std::mem::take(&mut self.last_draw);
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
            v[index as usize] = value;
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
        while cur < end {
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
            PACKETS.fetch_add(1, Ordering::Relaxed);
            self.cur_packet = cur;
            self.cur_opcode = opcode;
            let p = |i: u64| f.rd(cur + 4 * (1 + i));
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
            let flow = self.packet(opcode, h, cur, n, &p);
            if self.exec.as_ref().is_some_and(|e| e.submit_requested()) {
                self.submit_gpu("label");
            }
            if opcode == op::REWIND {
                // The CPU streams packets behind a REWIND: read them live.
                f = Fetched::new(next, 0);
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
        self.depth -= 1;
        if self.depth == 0 {
            self.submit_gpu("cmdbuf");
        }
    }

    fn packet(&mut self, opcode: u32, h: u32, at: u64, n: u64, p: &dyn Fn(u64) -> u32) -> Flow {
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
                    op::DRAW_INDEX_AUTO => Some(super::exec::DrawArgs { count: p(0), instances, index: None, base_vertex: None, first_instance: 0 }),
                    op::DRAW_INDEX_2 => self.index_size().map(|sz| super::exec::DrawArgs { count: p(3), instances, index: Some((addr64(p(1), p(2)), sz)), base_vertex: None, first_instance: 0 }),
                    op::DRAW_INDEX_OFFSET_2 => self.index_size().map(|sz| super::exec::DrawArgs { count: p(2), instances, index: Some((self.index_base + p(1) as u64 * sz as u64, sz)), base_vertex: None, first_instance: 0 }),
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
                            Some((super::exec::DrawArgs { count: w[0], instances: w[1], index: None, base_vertex: Some(w[2] as i32), first_instance: w[3] }, w[2], w[3]))
                        } else if opcode == op::DRAW_INDEX_INDIRECT && valid(a, 20) {
                            let w = unsafe { [rd32(a), rd32(a + 4), rd32(a + 8), rd32(a + 12), rd32(a + 16)] };
                            if super::capture::capturing() {
                                let ty = self.reg(Space::Uc, crate::hle::libs::agc_cmd::REG_VGT_INDEX_TYPE) & 3;
                                indirect = Some(format!("args {w:?} locs {vtx_loc:#x}/{inst_loc:#x} index {:#x} type {ty} size {}{}", self.index_base, self.index_count, self.index_range(w[2], w[0])));
                            }
                            self.index_size().map(|sz| (super::exec::DrawArgs { count: w[0], instances: w[1], index: Some((self.index_base + w[2] as u64 * sz as u64, sz)), base_vertex: Some(w[3] as i32), first_instance: w[4] }, w[3], w[4]))
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
                self.last_draw = multi.take().unwrap_or_else(|| "not executed".into());
                match args {
                    Some(a) => {
                        self.draw(a);
                        if super::capture::draw_dumps() {
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
                            self.dispatch(g, initiator);
                            if let Some(c) = check {
                                self.sync();
                                super::emucheck::compare(c);
                            }
                        }
                    }
                }
                if super::capture::dispatch_dumps() {
                    // Per-dispatch image dumps: complete the dispatch first.
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
                return self.wait(c, addr64(p(1), p(2)), false, p(3) as u64, p(4) as u64, at);
            }
            op::WAIT_REG_MEM64 => {
                if !super::tune::lazy_wait_submit() {
                    self.submit_gpu("wait");
                }
                let c = p(0);
                return self.wait(c, addr64(p(1), p(2)), true, addr64(p(3), p(4)), addr64(p(5), p(6)), at);
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
                    promise(s, a, v, bytes);
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
        super::exec::trace_writes(&format!("{} DMA_DATA (src_sel {src_sel}, from {src:#x})", self.name), dst, bytes);
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
                for i in 0..(bytes / 4).max(1) {
                    unsafe { wr32(if daic { dst } else { dst + 4 * i }, v) };
                }
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
                if int_sel != 0 && int_sel != 3 {
                    EOP_INTERRUPTS.fetch_add(1, Ordering::Relaxed);
                    crate::hle::libs::agcdriver::raise_eop(ctx);
                }
            }),
        );
        if let Some(s) = seq {
            match data_sel {
                1 => promise(s, a, data & 0xFFFF_FFFF, 4),
                2 => promise(s, a, data, 8),
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
        let v0 = read();
        if compare(func, v0 & mask, reference & mask) {
            label_event(&self.name, b'I', a, v0, reference, at);
            return Flow::Next;
        }
        if let Some(seq) = promised(a, wide, func, mask, reference) {
            if let Some(e) = self.exec.as_mut() {
                e.depend_on(seq);
                PROMISED.fetch_add(1, Ordering::Relaxed);
                label_event(&self.name, b'P', a, v0, reference, at);
                return Flow::Next;
            }
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
        let t0 = Instant::now();
        let use_promise = self.exec.is_some();
        let r = self.poll_until(at, &what, || {
            hit.load(Ordering::SeqCst) || compare(func, read() & mask, reference & mask) || (use_promise && promised(a, wide, func, mask, reference).is_some())
        });
        {
            let by = last_write(a).map(|w| w.0).unwrap_or_else(|| "cpu".into());
            let mut g = WAIT_STATS.lock();
            let e = g.get_or_insert_with(HashMap::new).entry((self.name.clone(), by)).or_default();
            e.0 += 1;
            e.1 += t0.elapsed().as_nanos() as u64;
        }
        ACTIVE_WAITS.lock().retain(|w| !std::sync::Arc::ptr_eq(&w.hit, &hit));
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
            if let (Some(seq), Some(e)) = (promised(a, wide, func, mask, reference), self.exec.as_mut()) {
                e.depend_on(seq);
                PROMISED.fetch_add(1, Ordering::Relaxed);
                kind = b'P';
            }
        }
        label_event(&self.name, kind, a, read(), reference, at);
        match r {
            Flow::Stop => Flow::Stop,
            _ => Flow::Next,
        }
    }

    /// Spins (then sleeps) until `cond`; reports long waits once.
    fn poll_until(&mut self, at: u64, what: &str, cond: impl Fn() -> bool) -> Flow {
        let start = Instant::now();
        let mut reported = false;
        set_state(&self.name, what.to_string());
        let mut spins = 0u32;
        while !cond() {
            if crate::gpu::shutting_down() {
                return Flow::Stop;
            }
            spins += 1;
            if spins < 200 {
                std::hint::spin_loop();
            } else if spins < 1000 {
                std::thread::yield_now();
            } else {
                std::thread::sleep(Duration::from_micros(100));
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
                let history = if a != 0 { format!("; label history:{}", label_history(a, 16)) } else { String::new() };
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
                    crate::warn!("gpu[{}]: flip of buffer {index} failed ({r:#x})", self.name);
                }
                let n = crate::gpu::FRAMES.fetch_add(1, Ordering::Relaxed) + 1;
                super::capture::on_flip(n);
            }
            pop::WAIT_SAFE_FOR_RENDERING => {
                let (handle, index) = (a(0) as i32, a(1) as i32);
                if crate::hle::libs::videoout::buffer_busy(handle, index) {
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
                    if mode != 0 && writable {
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
                    Some(c) => self.regs.ctx = c,
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
