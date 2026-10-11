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

thread_local! {
    /// This thread's view of the shared caches (entries never change once
    /// made), so that every command does not take their locks: programs by
    /// address with the shader registrations count at which they were last
    /// confirmed and the address's registration number (validating them by
    /// hashing their code took a few microseconds per command), plans, and
    /// translations.
    static LOCAL_PROGRAMS: std::cell::RefCell<HashMap<u64, (u64, u64, Arc<Vec<rdna::Inst>>, u64)>> = std::cell::RefCell::new(HashMap::new());
    static LOCAL_PLANS: std::cell::RefCell<HashMap<u64, Arc<Plan>>> = std::cell::RefCell::new(HashMap::new());
    static LOCAL_TRANSLATIONS: std::cell::RefCell<HashMap<(u64, u64), Arc<Translated>>> = std::cell::RefCell::new(HashMap::new());
}

pub struct Translated {
    pub out: Output,
    pub plan: Arc<Plan>,
    /// Cache key (program hash ^ specialization hash).
    pub key: u64,
    /// Buffer resources targeted by atomic instructions.
    pub atomic_res: Vec<usize>,
}

fn hash_words(w: &[u32]) -> u64 {
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
    let insts = Arc::new(insts);
    map.insert(addr, Program { words, hash, insts: insts.clone() });
    Ok((insts, hash))
}

/// Guest memory reader that records what it read (hazard checks; adjacent
/// reads merged).
pub struct Reader {
    pub reads: Vec<(u64, u64)>,
    /// The mapping of the last read (no memory-map lookup inside it).
    span: (u64, u64),
    /// Pointers loaded from memory so far (each resource dword names its
    /// pointer chain again).
    ptrs: Vec<(Ptr, u64)>,
    /// Address of the program (program-relative pointers).
    code_base: u64,
}

impl Reader {
    pub fn new(code_base: u64) -> Reader {
        Reader { reads: Vec::new(), span: (0, 0), ptrs: Vec::new(), code_base }
    }

    fn u32(&mut self, addr: u64) -> Option<u32> {
        if addr % 4 != 0 {
            return None;
        }
        if !(addr >= self.span.0 && addr + 4 <= self.span.1) {
            if !super::cp::guest_readable(addr, 4) {
                return None;
            }
            self.span = crate::mem::readable_span(addr).unwrap_or((addr, addr + 4));
        }
        match self.reads.last_mut() {
            Some(r) if r.0 + r.1 == addr => r.1 += 4,
            _ => self.reads.push((addr, 4)),
        }
        Some(unsafe { std::ptr::read_volatile(addr as *const u32) })
    }

    fn ptr(&mut self, p: &Ptr, user: &[u32]) -> Option<u64> {
        match p {
            Ptr::User(i) => Some((*user.get(*i as usize)? as u64) | ((*user.get(*i as usize + 1)? as u64) << 32)),
            Ptr::Mem(inner, off) => {
                if let Some(&(_, v)) = self.ptrs.iter().find(|(q, _)| q == p) {
                    return Some(v);
                }
                let base = self.ptr(inner, user)? + *off as u64;
                let v = (self.u32(base)? as u64) | ((self.u32(base + 4)? as u64) << 32);
                self.ptrs.push((p.clone(), v));
                Some(v)
            }
            Ptr::Pc(o) => Some(self.code_base + *o),
        }
    }

    fn eval(&mut self, p: &Prov, user: &[u32]) -> Option<u32> {
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

#[derive(Clone, Debug)]
pub enum Resolved {
    Buffer(Vsharp),
    Global(u64),
    Image(Tsharp),
    /// The entries of an image table (None: unusable entry, bound as a
    /// dummy image).
    ImageTable(Vec<Option<Tsharp>>),
    Sampler([u32; 4]),
}

/// Everything needed to bind one shader stage for a draw/dispatch.
pub struct Prepared {
    pub stage: Stage,
    pub addr: u64,
    pub insts: Arc<Vec<rdna::Inst>>,
    pub tr: Arc<Translated>,
    pub resources: Vec<Resolved>,
    /// Guest address of each descriptor's first dword when read from memory.
    pub res_addrs: Vec<Option<u64>>,
    /// SGPR data buffer contents (resource info blocks filled at binding).
    pub sgpr_data: Vec<u32>,
    pub user: Vec<u32>,
    /// Guest memory read while resolving (for write hazards).
    pub reads: Vec<(u64, u64)>,
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
pub fn prepare_compute(cp: &Cp, wave64: bool) -> Result<Prepared, String> {
    let sh = |i: u32| cp.reg(Space::Sh, i);
    let addr = ((sh(CS_PGM_LO) as u64) | ((sh(CS_PGM_LO + 1) as u64 & 0xFF) << 32)) << 8;
    let rsrc2 = sh(CS_RSRC2);
    let user_sgprs = ((rsrc2 >> 1) & 0x1F).min(16) as u16;
    let user: Vec<u32> = (0..16).map(|i| sh(CS_USER_DATA + i)).collect();
    let local = [sh(CS_NUM_THREAD) & 0xFFFF, sh(CS_NUM_THREAD + 1) & 0xFFFF, sh(CS_NUM_THREAD + 2) & 0xFFFF];
    let (insts, code_hash) = program(addr)?;
    let mut spec = Spec {
        stage: Stage::Compute,
        user_sgprs,
        user_base: 0,
        code_base: addr,
        wave64,
        ps_input_addr: 0,
        ps_inputs: Vec::new(),
        barycentric: false,
        mrt: [MrtSpec::default(); 8],
        cs_local_size: local,
        cs_rsrc2: rsrc2,
        resources: Vec::new(),
        binding_base: 0,
        probe: probe_for(addr),
        dump: dump_for(addr),
        force_dispatcher: super::tune::block_chain() == 0,
        dispatcher_for_loops: super::tune::block_chain() == 2,
    };
    prepare(addr, insts, code_hash, &user, &mut spec)
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
const SPI_SHADER_COL_FORMAT: u32 = 0x28714;
const CB_COLOR0_INFO: u32 = 0x28C70;

/// Resolves and translates the pixel shader bound on `cp`.
pub fn prepare_ps(cp: &Cp) -> Result<Prepared, String> {
    let sh = |i: u32| cp.reg(Space::Sh, i);
    let ctx = |a: u32| cp.reg(Space::Ctx, (a - 0x28000) / 4);
    let addr = ((sh(PS_PGM_LO) as u64) | ((sh(PS_PGM_LO + 1) as u64 & 0xFF) << 32)) << 8;
    let rsrc2 = sh(PS_RSRC2);
    let user_sgprs = user_count(rsrc2);
    let user: Vec<u32> = (0..32).map(|i| sh(PS_USER_DATA + i)).collect();
    let (insts, code_hash) = program(addr)?;
    let in_control = ctx(SPI_PS_IN_CONTROL);
    let mut ps_inputs = Vec::new();
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
        resources: Vec::new(),
        binding_base: PS_BINDING_BASE,
        probe: probe_for(addr),
        dump: None,
        force_dispatcher: super::tune::block_chain() == 0,
        dispatcher_for_loops: super::tune::block_chain() == 2,
    };
    prepare(addr, insts, code_hash, &user, &mut spec)
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
    *PROBE.lock() = Some(p);
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
pub fn prepare_ngg(cp: &Cp) -> Result<Prepared, String> {
    let sh = |i: u32| cp.reg(Space::Sh, i);
    let stages = cp.reg(Space::Ctx, (VGT_SHADER_STAGES_EN - 0x28000) / 4);
    let addr = ((sh(ES_PGM_LO) as u64) | ((sh(ES_PGM_LO + 1) as u64 & 0xFF) << 32)) << 8;
    let rsrc2 = sh(GS_RSRC2);
    let user_sgprs = user_count(rsrc2);
    let user: Vec<u32> = (0..32).map(|i| sh(GS_USER_DATA + i)).collect();
    let (insts, code_hash) = program(addr)?;
    let mut spec = Spec {
        stage: Stage::Vertex,
        user_sgprs,
        user_base: 8,
        code_base: addr,
        wave64: (stages >> 22) & 1 == 0,
        ps_input_addr: 0,
        ps_inputs: Vec::new(),
        barycentric: false,
        mrt: [MrtSpec::default(); 8],
        cs_local_size: [1, 1, 1],
        cs_rsrc2: 0,
        resources: Vec::new(),
        binding_base: 0,
        probe: None,
        dump: super::capture::vs_dump().and_then(|(first, inst)| pos_export(&insts).map(|i| (i.pc, [first, inst, 0]))),
        force_dispatcher: super::tune::block_chain() == 0,
        dispatcher_for_loops: super::tune::block_chain() == 2,
    };
    prepare(addr, insts, code_hash, &user, &mut spec)
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
                Resolved::Global(a) => format!("{i}: global {a:#x} {rw}"),
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

fn prepare(addr: u64, insts: Arc<Vec<rdna::Inst>>, code_hash: u64, user: &[u32], spec: &mut Spec) -> Result<Prepared, String> {
    // The plan depends only on the program and the user-data layout.
    let plan_key = code_hash ^ (spec.user_sgprs as u64) << 48 ^ (spec.user_base as u64) << 56;
    let cached = LOCAL_PLANS.with(|c| c.borrow().get(&plan_key).cloned()).or_else(|| {
        let p = PLANS.lock().as_ref().and_then(|m| m.get(&plan_key).cloned())?;
        LOCAL_PLANS.with(|c| c.borrow_mut().insert(plan_key, p.clone()));
        Some(p)
    });
    let plan = match cached {
        Some(p) => p,
        None => {
            let t0 = std::time::Instant::now();
            let p = Arc::new(rdna::analysis::analyze(&insts, spec.user_sgprs, spec.user_base));
            let ms = t0.elapsed().as_millis();
            if ms >= 100 {
                crate::info!("{:?} shader {addr:#x}: analysis took {ms} ms ({} instructions)", spec.stage, insts.len());
            }
            PLANS.lock().get_or_insert_with(HashMap::new).insert(plan_key, p.clone());
            LOCAL_PLANS.with(|c| c.borrow_mut().insert(plan_key, p.clone()));
            p
        }
    };
    if !plan.errors.is_empty() {
        return Err(format!("{:?} shader {addr:#x}: {}", spec.stage, plan.errors.join("; ")));
    }
    let mut rd = Reader::new(addr);
    let mut resources = Vec::with_capacity(plan.resources.len());
    let mut res_addrs = Vec::with_capacity(plan.resources.len());
    spec.resources.clear();
    for (i, res) in plan.resources.iter().enumerate() {
        let mut w = [0u32; 8];
        let src = match res.dwords.first() {
            Some(Prov::Mem(ptr, off)) => rd.ptr(ptr, user).map(|b| b + *off as u64),
            _ => None,
        };
        res_addrs.push(src);
        for (k, p) in res.dwords.iter().enumerate() {
            w[k] = rd.eval(p, user).ok_or_else(|| format!("{:?} shader {addr:#x}: resource {i} dword {k} unresolvable ({p:?})", spec.stage))?;
        }
        match res.kind {
            ResKind::Buffer => {
                let v = Vsharp::decode(&w);
                spec.resources.push(ResSpec::Buffer { format: v.format, dst_sel: v.dst_sel });
                resources.push(Resolved::Buffer(v));
            }
            ResKind::Global => {
                spec.resources.push(ResSpec::Global);
                resources.push(Resolved::Global(w[0] as u64 | ((w[1] as u64) << 32)));
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
                // The table's V#: entries of 32 bytes (T#s) from its base.
                let v = Vsharp::decode(&w);
                let n = (v.extent() / 32).min(rdna::translate::IMAGE_TABLE_LEN as u64);
                let mut entries = Vec::with_capacity(n as usize);
                let mut first: Option<(ImageDim, u16)> = None;
                for k in 0..n {
                    let a = v.base + 32 * k;
                    let mut d = [0u32; 8];
                    let mut ok = true;
                    for (j, dw) in d.iter_mut().enumerate() {
                        match rd.u32(a + 4 * j as u64) {
                            Some(x) => *dw = x,
                            None => ok = false,
                        }
                    }
                    let t = Tsharp::decode(&d);
                    if ok {
                        super::lodstats::note(&t.raw);
                    }
                    let dim = ImageDim::from_type(t.ty, t.depth);
                    let usable = ok && t.base != 0 && super::images::vk_format(t.format).is_some() && dim.is_some() && first.is_none_or(|(fd, _)| Some(fd) == dim);
                    if usable && first.is_none() {
                        first = Some((dim.unwrap(), t.format));
                    }
                    entries.push(usable.then_some(t));
                }
                let (dim, format) = first.ok_or_else(|| format!("{:?} shader {addr:#x}: image table {i} ({:#x}, {n} entries) has no usable T#", spec.stage, v.base))?;
                spec.resources.push(ResSpec::Image { dim, class: num_class(format), depth: false });
                resources.push(Resolved::ImageTable(entries));
            }
        }
    }
    if super::tune::cmp_emulate() && resources.iter().any(|r| matches!(r, Resolved::Image(_))) {
        super::images::with(|im| {
            for (r, s) in resources.iter().zip(spec.resources.iter_mut()) {
                if let (Resolved::Image(t), ResSpec::Image { depth, .. }) = (r, s) {
                    *depth = im.depth_view(t);
                }
            }
        });
    }
    // Translations do not depend on the program's address (see
    // translate::code_base_offset): copies of a program share one, and its
    // pipeline (the Vulkan pipeline cache finds it again in the next run).
    let spec_hash = {
        static BY_ADDRESS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let by_address = *BY_ADDRESS.get_or_init(|| std::env::var_os("DS_WAVEMASK").is_some());
        let base = spec.code_base;
        if !by_address {
            spec.code_base = 0;
        }
        let mut h = std::collections::hash_map::DefaultHasher::new();
        spec.hash(&mut h);
        spec.code_base = base;
        h.finish()
    };
    let tr = {
        let cached = LOCAL_TRANSLATIONS.with(|c| c.borrow().get(&(plan_key, spec_hash)).cloned()).or_else(|| {
            let t = TRANSLATIONS.lock().as_ref().and_then(|m| m.get(&(plan_key, spec_hash)).cloned())?;
            LOCAL_TRANSLATIONS.with(|c| c.borrow_mut().insert((plan_key, spec_hash), t.clone()));
            Some(t)
        });
        match cached {
            Some(t) => t,
            None => {
                let t0 = std::time::Instant::now();
                let out = translate::translate(&insts, &plan, spec).map_err(|e| format!("{:?} shader {addr:#x}: {e}", spec.stage))?;
                let ms = t0.elapsed().as_millis();
                if ms >= 100 {
                    crate::info!("{:?} shader {addr:#x}: translated in {ms} ms ({} instructions, {} SPIR-V words)", spec.stage, insts.len(), out.spirv.len());
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
                let t = Arc::new(Translated { out, plan: plan.clone(), key: plan_key ^ spec_hash.rotate_left(1), atomic_res });
                TRANSLATIONS.lock().get_or_insert_with(HashMap::new).insert((plan_key, spec_hash), t.clone());
                LOCAL_TRANSLATIONS.with(|c| c.borrow_mut().insert((plan_key, spec_hash), t.clone()));
                t
            }
        }
    };
    // Scalar-load data, resolved now.
    let mut sgpr_data = user.to_vec();
    sgpr_data.resize(translate::SGPR_DATA_USER as usize, 0);
    for l in &tr.plan.loads {
        let base = rd.ptr(&l.ptr, user);
        for k in 0..l.dwords {
            sgpr_data.push(base.and_then(|b| rd.u32(b + l.offset as u64 + 4 * k as u64)).unwrap_or(0));
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
    Ok(Prepared { stage: spec.stage, addr, insts, tr, resources, res_addrs, sgpr_data, user: user.to_vec(), reads: rd.reads })
}
