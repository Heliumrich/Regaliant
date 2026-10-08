//! gfx10 PM4 command processor.
//!
//! Executes command buffers submitted through libSceAgcDriver on a GPU queue
//! thread: register state (context, SH, uconfig), control flow (indirect
//! buffers, COND_EXEC, REWIND, the Agc branch), synchronisation (WAIT_REG_MEM,
//! RELEASE_MEM labels and end-of-pipe interrupts), memory packets (WRITE_DATA,
//! COPY_DATA, DMA_DATA), flips and the private Agc packets
//! (`hle::libs::agc_cmd`).
//!
//! Draws and dispatches are decoded and counted; they are not rendered yet
//! (no shader translation or Vulkan pipeline exists). Memory and labels are
//! written exactly as the packets request, so CPU/GPU synchronisation runs
//! as on hardware.

use crate::hle::libs::agc_cmd::{DS_MAGIC, op, pop};
use std::collections::HashSet;
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
    pub compute: bool,
    pub regs: Regs,
    index_base: u64,
    index_count: u32,
    num_instances: u32,
    draw_indirect_base: u64,
    dispatch_indirect_base: u64,
    markers: Vec<String>,
    ctx_stack: Vec<Vec<u32>>,
    unknown_seen: HashSet<u32>,
    notes_seen: HashSet<&'static str>,
    depth: u32,
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

/// GPU timestamp (100 MHz, like the console's GPU clock counter).
pub fn gpu_clock() -> u64 {
    static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    let s = START.get_or_init(Instant::now);
    (s.elapsed().as_nanos() / 10) as u64
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
            compute,
            regs: Regs::new(),
            index_base: 0,
            index_count: 0,
            num_instances: 1,
            draw_indirect_base: 0,
            dispatch_indirect_base: 0,
            markers: Vec::new(),
            ctx_stack: Vec::new(),
            unknown_seen: HashSet::new(),
            notes_seen: HashSet::new(),
            depth: 0,
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

    fn reg(&self, space: Space, index: u32) -> u32 {
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

    /// Executes `size` dwords at `addr`.
    pub fn run(&mut self, addr: u64, size: u32) {
        if self.depth > 8 {
            crate::error!("gpu[{}]: indirect buffer nesting too deep at {addr:#x}", self.name);
            return;
        }
        self.depth += 1;
        let mut cur = addr;
        let mut end = addr + 4 * size as u64;
        while cur < end {
            if !valid(cur, 4) {
                crate::error!("gpu[{}]: command pointer {cur:#x} outside guest memory", self.name);
                break;
            }
            let h = unsafe { rd32(cur) };
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
            let p = |i: u64| unsafe { rd32(cur + 4 * (1 + i)) };
            match self.packet(opcode, h, cur, n, &p) {
                Flow::Next => cur = next,
                Flow::Skip(d) => cur = next + 4 * d as u64,
                Flow::Chain(a, s) => {
                    cur = a;
                    end = a + 4 * s as u64;
                }
                Flow::Stop => break,
            }
        }
        self.depth -= 1;
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
            op::CLEAR_STATE => {
                self.regs.ctx.fill(0);
            }
            op::CONTEXT_CONTROL | op::PFP_SYNC_ME | op::EVENT_WRITE | op::ACQUIRE_MEM => {}
            op::INDEX_TYPE => self.set_reg(Space::Uc, crate::hle::libs::agc_cmd::REG_VGT_INDEX_TYPE, p(0)),
            op::INDEX_BASE => self.index_base = addr64(p(0), p(1)),
            op::INDEX_BUFFER_SIZE => self.index_count = p(0),
            op::NUM_INSTANCES => self.num_instances = p(0).max(1),
            op::SET_BASE => {
                let base = addr64(p(1), p(2));
                if p(0) == 2 { self.dispatch_indirect_base = base } else { self.draw_indirect_base = base }
            }
            op::SET_PREDICATION => self.note("SET_PREDICATION recorded but not applied"),
            op::DRAW_INDEX_2 | op::DRAW_INDEX_AUTO | op::DRAW_INDEX_OFFSET_2 | op::DRAW_INDIRECT | op::DRAW_INDEX_INDIRECT
            | op::DRAW_INDIRECT_MULTI | op::DRAW_INDEX_INDIRECT_MULTI | op::DRAW_INDEX_MULTI_AUTO => {
                DRAWS.fetch_add(1, Ordering::Relaxed);
                self.note("draws are decoded but not rendered yet (no shader translation)");
            }
            op::DISPATCH_DIRECT | op::DISPATCH_INDIRECT => {
                DISPATCHES.fetch_add(1, Ordering::Relaxed);
                self.note("dispatches are decoded but not executed yet (no shader translation)");
            }
            op::WRITE_DATA => self.write_data(n, p),
            op::COPY_DATA => self.copy_data(p),
            op::DMA_DATA => self.dma_data(p),
            op::RELEASE_MEM => self.release_mem(p),
            op::WAIT_REG_MEM => {
                let c = p(0);
                return self.wait(c, addr64(p(1), p(2)), false, p(3) as u64, p(4) as u64, at);
            }
            op::WAIT_REG_MEM64 => {
                let c = p(0);
                return self.wait(c, addr64(p(1), p(2)), true, addr64(p(3), p(4)), addr64(p(5), p(6)), at);
            }
            op::COND_EXEC => {
                let a = addr64(p(0), p(1));
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
                for i in 0..count {
                    let d = if no_incr { a } else { a + 4 * i };
                    unsafe { wr32(d, p(3 + i)) };
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
        let value: u64 = match src {
            0 => self.reg_by_addr(src_v as u32) as u64,
            1 | 2 => {
                if !valid(src_v, 8) {
                    crate::error!("gpu[{}]: COPY_DATA from {src_v:#x} outside guest memory", self.name);
                    return;
                }
                if wide { unsafe { rd64(src_v) } } else { (unsafe { rd32(src_v) }) as u64 }
            }
            5 => src_v,
            9 => gpu_clock(),
            _ => {
                self.note("COPY_DATA from GDS/perf counters reads 0");
                0
            }
        };
        match dst {
            1 | 2 | 5 => {
                if !valid(dst_a, 8) {
                    crate::error!("gpu[{}]: COPY_DATA to {dst_a:#x} outside guest memory", self.name);
                    return;
                }
                if wide { unsafe { wr64(dst_a, value) } } else { unsafe { wr32(dst_a, value as u32) } }
            }
            0 => self.write_reg_addr(dst_a as u32, value as u32),
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
        match src_sel {
            2 => {
                // Fill with the immediate dword.
                let v = src as u32;
                if daic {
                    unsafe { wr32(dst, v) };
                } else {
                    for i in 0..bytes / 4 {
                        unsafe { wr32(dst + 4 * i, v) };
                    }
                }
            }
            0 | 3 if !sas => {
                if !valid(src, if saic { 4 } else { bytes }) {
                    crate::error!("gpu[{}]: DMA_DATA from {src:#x}+{bytes:#x} outside guest memory", self.name);
                    return;
                }
                if saic || daic {
                    for i in 0..bytes / 4 {
                        let s = if saic { src } else { src + 4 * i };
                        let d = if daic { dst } else { dst + 4 * i };
                        unsafe { wr32(d, rd32(s)) };
                    }
                } else {
                    unsafe { std::ptr::copy(src as *const u8, dst as *mut u8, bytes as usize) };
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
        let data_cntl = p(1);
        let int_sel = (data_cntl >> 24) & 7;
        let data_sel = (data_cntl >> 29) & 7;
        let a = addr64(p(2) & !3, p(3));
        let data = addr64(p(4), p(5));
        let ctx = p(6);
        match data_sel {
            0 => {}
            1 | 2 | 3 | 4 => {
                if !valid(a, 8) {
                    crate::error!("gpu[{}]: RELEASE_MEM to {a:#x} outside guest memory", self.name);
                } else {
                    match data_sel {
                        1 => unsafe { wr32(a, data as u32) },
                        2 => unsafe { wr64(a, data) },
                        _ => unsafe { wr64(a, gpu_clock()) },
                    }
                }
            }
            _ => self.note("RELEASE_MEM with GDS data ignored"),
        }
        if int_sel != 0 && int_sel != 3 {
            EOP_INTERRUPTS.fetch_add(1, Ordering::Relaxed);
            crate::hle::libs::agcdriver::raise_eop(ctx);
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
        if compare(func, read() & mask, reference & mask) {
            return Flow::Next;
        }
        let what = format!("WAIT_REG_MEM at {at:#x}: [{a:#x}] & {mask:#x} func {func} ref {reference:#x}");
        match self.poll_until(at, &what, || compare(func, read() & mask, reference & mask)) {
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
                crate::warn!("gpu[{}]: waiting > 3 s: {what}{}", self.name, if marker.is_empty() { String::new() } else { format!(" (marker '{marker}')") });
            }
        }
        set_state(&self.name, "running".into());
        let _ = at;
        Flow::Next
    }

    fn private(&mut self, operation: u32, at: u64, n: u64, a: &dyn Fn(u64) -> u32) -> Flow {
        match operation {
            pop::FLIP => {
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
                crate::gpu::FRAMES.fetch_add(1, Ordering::Relaxed);
            }
            pop::WAIT_SAFE_FOR_RENDERING => {
                let (handle, index) = (a(0) as i32, a(1) as i32);
                if crate::hle::libs::videoout::buffer_busy(handle, index) {
                    let what = format!("WaitUntilSafeForRendering: display buffer {index} is displayed or queued");
                    return self.poll_until(at, &what, || !crate::hle::libs::videoout::buffer_busy(handle, index));
                }
            }
            pop::WORKLOADS_ACTIVE | pop::WORKLOAD_COMPLETE | pop::LOD_STATS => {}
            pop::CONTEXT_STATE => match a(0) {
                0 => self.regs.ctx.fill(0),
                1 => self.ctx_stack.push(self.regs.ctx.clone()),
                2 => match self.ctx_stack.pop() {
                    Some(c) => self.regs.ctx = c,
                    None => crate::warn!("gpu[{}]: context state pop without push at {at:#x}", self.name),
                },
                3 => {
                    self.ctx_stack.push(self.regs.ctx.clone());
                    self.regs.ctx.fill(0);
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
                let ctl = a(0);
                let cmp = addr64(a(1), a(2));
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
