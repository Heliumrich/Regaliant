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

pub struct Translated {
    pub out: Output,
    pub plan: Arc<Plan>,
    /// Cache key (program hash ^ specialization hash).
    pub key: u64,
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
fn program(addr: u64) -> Result<(Arc<Vec<rdna::Inst>>, u64), String> {
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

/// Guest memory reader that records what it read (hazard checks).
pub struct Reader {
    pub reads: Vec<(u64, u64)>,
}

impl Reader {
    fn u32(&mut self, addr: u64) -> Option<u32> {
        if addr % 4 != 0 || !super::cp::guest_readable(addr, 4) {
            return None;
        }
        self.reads.push((addr, 4));
        Some(unsafe { std::ptr::read_volatile(addr as *const u32) })
    }

    fn ptr(&mut self, p: &Ptr, user: &[u32]) -> Option<u64> {
        match p {
            Ptr::User(i) => Some((*user.get(*i as usize)? as u64) | ((*user.get(*i as usize + 1)? as u64) << 32)),
            Ptr::Mem(inner, off) => {
                let base = self.ptr(inner, user)? + *off as u64;
                Some((self.u32(base)? as u64) | ((self.u32(base + 4)? as u64) << 32))
            }
            Ptr::Abs(a) => Some(*a),
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
            Prov::Unknown => None,
        }
    }
}

#[derive(Clone, Debug)]
pub enum Resolved {
    Buffer(Vsharp),
    Global(u64),
    Image(Tsharp),
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
        mrt: [MrtSpec::default(); 8],
        cs_local_size: local,
        cs_rsrc2: rsrc2,
        resources: Vec::new(),
        binding_base: 0,
        probe: None,
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
        ps_inputs,
        mrt,
        cs_local_size: [1, 1, 1],
        cs_rsrc2: 0,
        resources: Vec::new(),
        binding_base: PS_BINDING_BASE,
        probe: probe_for(addr),
    };
    prepare(addr, insts, code_hash, &user, &mut spec)
}

/// `DS_PROBE=<shader address>:<pc>:<vgpr>` (hex address and pc, decimal
/// VGPR): that pixel shader exports the VGPR's value after the instruction
/// at pc as its color (debugging).
fn probe_for(addr: u64) -> Option<(u32, u32)> {
    static P: std::sync::OnceLock<Option<(u64, u32, u32)>> = std::sync::OnceLock::new();
    let p = P.get_or_init(|| {
        let v = std::env::var("DS_PROBE").ok()?;
        let mut it = v.split(':');
        let a = u64::from_str_radix(it.next()?.trim_start_matches("0x"), 16).ok()?;
        let pc = u32::from_str_radix(it.next()?.trim_start_matches("0x"), 16).ok()?;
        let vg = it.next()?.parse().ok()?;
        Some((a, pc, vg))
    });
    p.and_then(|(a, pc, vg)| (a == addr).then_some((pc, vg)))
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
        mrt: [MrtSpec::default(); 8],
        cs_local_size: [1, 1, 1],
        cs_rsrc2: 0,
        resources: Vec::new(),
        binding_base: 0,
        probe: None,
    };
    prepare(addr, insts, code_hash, &user, &mut spec)
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
    let cached = PLANS.lock().as_ref().and_then(|m| m.get(&plan_key).cloned());
    let plan = match cached {
        Some(p) => p,
        None => {
            let t0 = std::time::Instant::now();
            let p = Arc::new(rdna::analysis::analyze(&insts, spec.user_sgprs, spec.user_base, addr));
            let ms = t0.elapsed().as_millis();
            if ms >= 100 {
                crate::info!("{:?} shader {addr:#x}: analysis took {ms} ms ({} instructions)", spec.stage, insts.len());
            }
            PLANS.lock().get_or_insert_with(HashMap::new).insert(plan_key, p.clone());
            p
        }
    };
    if !plan.errors.is_empty() {
        return Err(format!("{:?} shader {addr:#x}: {}", spec.stage, plan.errors.join("; ")));
    }
    let mut rd = Reader { reads: Vec::new() };
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
                spec.resources.push(ResSpec::Image { dim, class: num_class(t.format) });
                resources.push(Resolved::Image(t));
            }
            ResKind::Sampler => {
                spec.resources.push(ResSpec::Sampler);
                resources.push(Resolved::Sampler([w[0], w[1], w[2], w[3]]));
            }
        }
    }
    let spec_hash = {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        spec.hash(&mut h);
        h.finish()
    };
    let tr = {
        let cached = TRANSLATIONS.lock().as_ref().and_then(|m| m.get(&(plan_key, spec_hash)).cloned());
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
                let t = Arc::new(Translated { out, plan: plan.clone(), key: plan_key ^ spec_hash.rotate_left(1) });
                TRANSLATIONS.lock().get_or_insert_with(HashMap::new).insert((plan_key, spec_hash), t.clone());
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
    Ok(Prepared { stage: spec.stage, addr, insts, tr, resources, res_addrs, sgpr_data, user: user.to_vec(), reads: rd.reads })
}
