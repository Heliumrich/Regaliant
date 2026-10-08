//! Shader front end: program decoding/caching, fetch-shader inlining,
//! descriptor resolution against live guest state, specialization and
//! translation to SPIR-V.

use super::cp::Regs;
use super::desc::{self, Ssharp, Tsharp, Vsharp};
use gcn::analysis::{self, Plan, Prov, Ptr, ResKind};
use gcn::translate::{self, ImageDim, MrtSpec, Output, PsInput, ResSpec, Spec, Stage};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

pub const PS_BINDING_BASE: u32 = 32;

struct Program {
    len_words: usize,
    hash: u64,
    insts: Arc<Vec<gcn::Inst>>,
}

static PROGRAMS: Mutex<Option<HashMap<u64, Program>>> = Mutex::new(None);

#[derive(Clone)]
pub struct Translated {
    pub out: Arc<Output>,
    pub key_hash: u64,
}

static CACHE: Mutex<Option<HashMap<(u64, u64), Translated>>> = Mutex::new(None);

fn hash_words(w: &[u32]) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    w.hash(&mut h);
    h.finish()
}

fn guest_u32(addr: u64) -> Option<u32> {
    if addr < crate::mem::GUEST_BASE || addr + 4 > crate::mem::GUEST_END || addr % 4 != 0 {
        return None;
    }
    let st = crate::mem::MEM.lock();
    let v = st.find(addr)?;
    if v.kind == crate::mem::VmaKind::Reserved {
        return None;
    }
    drop(st);
    Some(unsafe { std::ptr::read_volatile(addr as *const u32) })
}

/// Decodes (and caches) the program at `addr`.
fn program(addr: u64) -> Result<(Arc<Vec<gcn::Inst>>, u64), String> {
    let code = super::shader_dump::code_slice(addr).ok_or_else(|| format!("shader {addr:#x} not mapped"))?;
    let mut g = PROGRAMS.lock();
    let map = g.get_or_insert_with(HashMap::new);
    if let Some(p) = map.get(&addr) {
        if p.len_words <= code.len() && hash_words(&code[..p.len_words]) == p.hash {
            return Ok((p.insts.clone(), p.hash));
        }
    }
    let insts = gcn::decode_program(code).map_err(|e| format!("decode {addr:#x}: {}", e.0))?;
    let last = insts.last().unwrap();
    let len = (last.pc / 4 + last.len as u32) as usize;
    let hash = hash_words(&code[..len]);
    let insts = Arc::new(insts);
    map.insert(addr, Program { len_words: len, hash, insts: insts.clone() });
    Ok((insts, hash))
}

fn eval_ptr(p: &Ptr, user: &[u32]) -> Option<u64> {
    match p {
        Ptr::User(i) => Some((*user.get(*i as usize)? as u64) | ((*user.get(*i as usize + 1)? as u64) << 32)),
        Ptr::Mem(inner, off) => {
            let base = eval_ptr(inner, user)? + *off as u64;
            Some((guest_u32(base)? as u64) | ((guest_u32(base + 4)? as u64) << 32))
        }
        Ptr::Abs(a) => Some(*a),
    }
}

fn eval(p: &Prov, user: &[u32]) -> Option<u32> {
    match p {
        Prov::User(i) => user.get(*i as usize).copied(),
        Prov::Const(c) => Some(*c),
        Prov::Mem(ptr, off) => guest_u32(eval_ptr(ptr, user)? + *off as u64),
        Prov::PtrLo(ptr, b) => Some((eval_ptr(ptr, user)? + *b as u64) as u32),
        Prov::PtrHi(ptr, b) => Some(((eval_ptr(ptr, user)? + *b as u64) >> 32) as u32),
        Prov::Unknown => None,
    }
}

#[derive(Clone, Debug)]
pub enum ResolvedRes {
    Buffer(Vsharp),
    Image(Tsharp),
    Sampler(Ssharp),
}

/// Everything needed to bind one shader stage for a draw/dispatch.
pub struct Prepared {
    pub stage: Stage,
    pub addr: u64,
    pub translated: Translated,
    pub resources: Vec<ResolvedRes>,
    pub sgpr_data: Vec<u32>,
    /// Buffer resources written by the shader (need write-back to guest memory).
    pub written_buffers: Vec<usize>,
    /// Buffer resources read by the shader (need current guest contents).
    pub read_buffers: Vec<usize>,
    /// Debug: instruction pcs using each resource (frame traces).
    pub res_pcs: Vec<Vec<u32>>,
}

/// Register state for a stage, captured at draw time.
pub struct StageRegs<'a> {
    pub stage: Stage,
    pub regs: &'a Regs,
}

const VS_PGM_LO: u32 = 0x2C48;
const PS_PGM_LO: u32 = 0x2C08;
const CS_PGM_LO: u32 = 0x2E0C;

fn image_dim(t: &Tsharp) -> ImageDim {
    match t.ty {
        desc::T_1D => ImageDim::D1,
        desc::T_3D => ImageDim::D3,
        desc::T_CUBE => {
            if t.last_array > t.base_array { ImageDim::CubeArray } else { ImageDim::Cube }
        }
        desc::T_1D_ARRAY => ImageDim::D1Array,
        desc::T_2D_ARRAY | desc::T_2D_MSAA_ARRAY => ImageDim::D2Array,
        _ => ImageDim::D2,
    }
}

/// Debug (`BB_PS_DEBUG=<shader addr>:<pc>:v<N>[:<pc2>:v<M>][;<addr>:...]`, `u<N>` converts
/// unsigned integers to floats divided by 255):
/// export v[N..N+2] captured after the instruction at `pc` instead of the
/// shader's outputs (first capture to MRT0, second to MRT1 and beyond).
fn debug_capture(addr: u64) -> Vec<(u32, u32, bool)> {
    static SPEC: std::sync::OnceLock<Vec<(u64, Vec<(u32, u32, bool)>)>> = std::sync::OnceLock::new();
    let specs = SPEC.get_or_init(|| {
        let Ok(v) = std::env::var("BB_PS_DEBUG") else { return Vec::new() };
        v.split(';')
            .filter_map(|e| {
                let mut it = e.trim().split(':');
                let a = u64::from_str_radix(it.next()?.trim_start_matches("0x"), 16).ok()?;
                let mut caps = Vec::new();
                while let (Some(pc), Some(r)) = (it.next(), it.next()) {
                    // vN: raw bits; uN: unsigned integer converted to float.
                    let int = r.starts_with('u');
                    caps.push((u32::from_str_radix(pc.trim_start_matches("0x"), 16).ok()?, r.trim_start_matches(['v', 'u']).parse().ok()?, int));
                }
                Some((a, caps))
            })
            .collect()
    });
    specs.iter().find(|(a, _)| *a == addr).map(|(_, c)| c.clone()).unwrap_or_default()
}

pub fn prepare(sr: &StageRegs) -> Result<Prepared, String> {
    let r = sr.regs;
    let (pgm_lo, user_base, rsrc1, rsrc2) = match sr.stage {
        Stage::Vertex => (VS_PGM_LO, 0x2C4C, r.sh(0x2C4A), r.sh(0x2C4B)),
        Stage::Fragment => (PS_PGM_LO, 0x2C0C, r.sh(0x2C0A), r.sh(0x2C0B)),
        Stage::Compute => (CS_PGM_LO, 0x2E40, r.sh(0x2E12), r.sh(0x2E13)),
    };
    let addr = ((r.sh(pgm_lo) as u64) | ((r.sh(pgm_lo + 1) as u64) << 32)) << 8;
    let ub = (user_base - 0x2C00) as usize;
    let user: Vec<u32> = r.sh[ub..ub + 16].to_vec();
    let user_sgprs = ((rsrc2 >> 1) & 0x1F) as u16;
    let (main, main_hash) = program(addr)?;
    // Inline the fetch subroutine (VS).
    let (code, code_hash) = match main.iter().find(|i| i.enc == gcn::Enc::Sop1 && i.op == 33) {
        Some(sw) if sr.stage == Stage::Vertex => {
            let s = sw.ssrc0() as usize;
            let faddr = (*user.get(s).ok_or("fetch pointer outside user data")? as u64) | ((*user.get(s + 1).ok_or("fetch pointer outside user data")? as u64) << 32);
            let (fetch, fh) = program(faddr)?;
            (Arc::new(translate::inline_fetch(&main, &fetch)), main_hash ^ fh.rotate_left(17))
        }
        _ => (main.clone(), main_hash),
    };
    let plan: Plan = analysis::analyze(&code, user_sgprs, addr);
    let stage_name = match sr.stage {
        Stage::Vertex => "vs",
        Stage::Fragment => "ps",
        Stage::Compute => "cs",
    };
    if !plan.errors.is_empty() {
        super::shader_dump::dump(stage_name, addr, &user);
        return Err(format!("{:?} shader {addr:#x}: {}", sr.stage, plan.errors.join("; ")));
    }
    // Resolve descriptors.
    let mut resources = Vec::with_capacity(plan.resources.len());
    let mut specs = Vec::with_capacity(plan.resources.len());
    let mut strides = Vec::with_capacity(plan.resources.len());
    for (i, res) in plan.resources.iter().enumerate() {
        let mut w = [0u32; 8];
        for (k, p) in res.dwords.iter().enumerate() {
            w[k] = match eval(p, &user) {
                Some(v) => v,
                None => {
                    super::shader_dump::dump(stage_name, addr, &user);
                    static LOGGED: Mutex<Option<std::collections::HashSet<u64>>> = Mutex::new(None);
                    if LOGGED.lock().get_or_insert_with(Default::default).insert(addr) {
                        crate::info!("{:?} shader {addr:#x}: resource {i} dword {k} ({p:?}) unresolvable with user data {:x?}", sr.stage, &user[..(user_sgprs as usize).min(16)]);
                    }
                    return Err(format!("{:?} shader {addr:#x}: resource {i} dword {k} unresolvable ({p:?})", sr.stage));
                }
            };
        }
        match res.kind {
            ResKind::Buffer => {
                let v = Vsharp::decode(&w);
                specs.push(ResSpec::Buffer { dfmt: v.dfmt, nfmt: v.nfmt, dst_sel: v.dst_sel });
                strides.push(v.stride);
                resources.push(ResolvedRes::Buffer(v));
            }
            ResKind::Image => {
                let t = Tsharp::decode(&w);
                specs.push(ResSpec::Image { dim: image_dim(&t), class: desc::num_class(t.nfmt) });
                strides.push(0);
                resources.push(ResolvedRes::Image(t));
            }
            ResKind::Sampler => {
                specs.push(ResSpec::Sampler);
                strides.push(0);
                resources.push(ResolvedRes::Sampler(Ssharp([w[0], w[1], w[2], w[3]])));
            }
        }
    }
    // Stage specialization.
    let mut spec = Spec {
        stage: sr.stage,
        user_sgprs,
        code_base: addr,
        vs_vgpr_comp_cnt: (rsrc1 >> 24) & 3,
        vs_step_rates: if sr.stage == Stage::Vertex { [r.ctx(0xA2A8).max(1), r.ctx(0xA2A9).max(1)] } else { [1, 1] },
        ps_input_addr: 0,
        ps_inputs: Vec::new(),
        mrt: [MrtSpec::default(); 8],
        cs_local_size: [1, 1, 1],
        cs_rsrc2: 0,
        resources: specs,
        buffer_strides: strides,
        binding_base: if sr.stage == Stage::Fragment { PS_BINDING_BASE } else { 0 },
        debug_capture: debug_capture(addr),
    };
    match sr.stage {
        Stage::Fragment => {
            spec.ps_input_addr = r.ctx(0xA1B4);
            let num_interp = r.ctx(0xA1B6) & 0x3F;
            for n in 0..num_interp.min(32) {
                let c = r.ctx(0xA191 + n);
                let off = c & 0x3F;
                spec.ps_inputs.push(PsInput {
                    param: (off & 0x1F) as u8,
                    flat: (c >> 10) & 1 != 0,
                    default: if off & 0x20 != 0 { Some(((c >> 8) & 3) as u8) } else { None },
                });
            }
            let col = r.ctx(0xA1C5);
            for m in 0..8 {
                let info = r.ctx(0xA31C + m * 0xF);
                let nt = (info >> 8) & 7;
                spec.mrt[m as usize] = MrtSpec { export_format: ((col >> (m * 4)) & 0xF) as u8, class: match nt { 4 => 2, 5 => 1, _ => 0 } };
            }
        }
        Stage::Compute => {
            spec.cs_local_size = [r.sh(0x2E07) & 0xFFFF, r.sh(0x2E08) & 0xFFFF, r.sh(0x2E09) & 0xFFFF];
            spec.cs_rsrc2 = rsrc2;
        }
        Stage::Vertex => {}
    }
    let key_hash = {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        spec.hash(&mut h);
        h.finish()
    };
    let translated = {
        let mut g = CACHE.lock();
        let map = g.get_or_insert_with(HashMap::new);
        match map.get(&(code_hash, key_hash)) {
            Some(t) => t.clone(),
            None => {
                super::shader_dump::dump(stage_name, addr, &user);
                let out = translate::translate(&code, &plan, &spec).map_err(|e| format!("{:?} shader {addr:#x}: {e}", sr.stage))?;
                for w in &out.warnings {
                    crate::warn!("{:?} shader {addr:#x}: {w}", sr.stage);
                }
                if let Some(dir) = super::shader_dump::DIR.get() {
                    let _ = std::fs::create_dir_all(dir);
                    let name = format!("{}_{addr:x}_{key_hash:016x}.spv", match sr.stage { Stage::Vertex => "vs", Stage::Fragment => "ps", Stage::Compute => "cs" });
                    let bytes: Vec<u8> = out.spirv.iter().flat_map(|w| w.to_le_bytes()).collect();
                    let _ = std::fs::write(dir.join(name), bytes);
                }
                let t = Translated { out: Arc::new(out), key_hash };
                map.insert((code_hash, key_hash), t.clone());
                t
            }
        }
    };
    // Scalar-load data (resolved now, per draw).
    let mut sgpr_data = user.clone();
    sgpr_data.resize(translate::SGPR_DATA_USER as usize, 0);
    for l in &plan.loads {
        let base = eval_ptr(&l.ptr, &user);
        for k in 0..l.dwords {
            sgpr_data.push(base.and_then(|b| guest_u32(b + l.offset as u64 + 4 * k as u64)).unwrap_or(0));
        }
    }
    let written_buffers = plan.resources.iter().enumerate().filter(|(_, r)| r.kind == ResKind::Buffer && r.written).map(|(i, _)| i).collect();
    let read_buffers = plan.resources.iter().enumerate().filter(|(_, r)| r.kind == ResKind::Buffer && r.read).map(|(i, _)| i).collect();
    let mut res_pcs = vec![Vec::new(); plan.resources.len()];
    for (&pc, &(r, s)) in &plan.inst_res {
        if let Some(v) = res_pcs.get_mut(r) {
            v.push(pc);
        }
        if let Some(v) = s.and_then(|s| res_pcs.get_mut(s)) {
            v.push(pc);
        }
    }
    for v in &mut res_pcs {
        v.sort_unstable();
    }
    Ok(Prepared { stage: sr.stage, addr, translated, resources, sgpr_data, written_buffers, read_buffers, res_pcs })
}
