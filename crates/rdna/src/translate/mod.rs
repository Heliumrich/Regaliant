//! RDNA (gfx10) -> SPIR-V translation.
//!
//! Ported from the earlier native port's GCN translator and re-keyed by gfx10
//! instruction names. Execution model: each SPIR-V invocation emulates one
//! hardware lane as a wave of one. Scalar registers are per-invocation copies
//! (scalar inputs are uniform, so the copies agree across lanes); EXEC/VCC and
//! compare masks keep this lane's bit in bit 0. VALU results are written only
//! when the lane is active. Branchy programs run in a loop + switch
//! dispatcher over basic blocks. Cross-lane operations within 32 lanes (DPP
//! rows, permlane, swizzles) use subgroup operations; wave64 operations that
//! cross 32-lane halves are approximated and reported as warnings.

mod alu;
pub mod gs;
mod mem;

use crate::analysis::{Plan, ResKind};
use crate::{Enc, Inst};
use rspirv::binary::Assemble;
use rspirv::dr::{Builder, Operand};
use rspirv::spirv::{self, BuiltIn, Capability, Decoration, ExecutionMode, ExecutionModel, StorageClass, Word};
use std::collections::{BTreeSet, HashMap};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Stage {
    /// NGG primitive shader (merged ES/GS) run as a Vulkan vertex shader:
    /// per-vertex work only; primitive exports are dropped.
    Vertex,
    Fragment,
    Compute,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ImageDim {
    D1,
    D2,
    D3,
    Cube,
    D1Array,
    D2Array,
    D2Msaa,
    D2MsaaArray,
    CubeArray,
}

impl ImageDim {
    /// From a T# TYPE field (word3[31:28]) and its layer count (cube
    /// descriptors count faces: more than 6 make a cube array).
    pub fn from_type(t: u32, layers: u32) -> Option<ImageDim> {
        Some(match t {
            8 => ImageDim::D1,
            9 => ImageDim::D2,
            10 => ImageDim::D3,
            11 if layers > 6 => ImageDim::CubeArray,
            11 => ImageDim::Cube,
            12 => ImageDim::D1Array,
            13 => ImageDim::D2Array,
            14 => ImageDim::D2Msaa,
            15 => ImageDim::D2MsaaArray,
            _ => return None,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NumClass {
    Float,
    Sint,
    Uint,
}

/// Fragment output / export conversion for one MRT.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct MrtSpec {
    /// SPI_SHADER_COL_FORMAT value for this target (0 = none).
    pub export_format: u8,
    /// 0 float, 1 sint, 2 uint (render target number type).
    pub class: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PsInput {
    /// Vertex parameter index feeding this attribute (SPI_PS_INPUT_CNTL.OFFSET).
    pub param: u8,
    pub flat: bool,
    /// Use a constant default instead of a vertex output (0..3 per DEFAULT_VAL).
    pub default: Option<u8>,
}

/// Per-resource specialization resolved from descriptors at draw time.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ResSpec {
    /// V#: unified FORMAT (word3[18:12]) and DST_SEL_XYZW (word3[11:0]).
    Buffer { format: u8, dst_sel: u16 },
    Image { dim: ImageDim, class: NumClass },
    Sampler,
    /// Base address of GLOBAL instructions (bound like a raw buffer).
    Global,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Spec {
    pub stage: Stage,
    /// Number of user-data dwords.
    pub user_sgprs: u16,
    /// First SGPR receiving user data (8 for NGG shaders, 0 otherwise).
    pub user_base: u16,
    pub code_base: u64,
    /// Wave64 (else wave32): lane numbering for cross-lane operations.
    pub wave64: bool,
    pub ps_input_addr: u32,
    pub ps_inputs: Vec<PsInput>,
    pub mrt: [MrtSpec; 8],
    pub cs_local_size: [u32; 3],
    /// COMPUTE_PGM_RSRC2 (TGID enables, TG size, TIDIG count, LDS size).
    pub cs_rsrc2: u32,
    pub resources: Vec<ResSpec>,
    /// Binding index of the first binding (stages share one descriptor set).
    pub binding_base: u32,
    /// Debugging: (pc, vgpr): the value of `vgpr` after the instruction at
    /// `pc` replaces the color of every MRT export ((v, v, v, 1) as float).
    pub probe: Option<(u32, u32)>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BindingKind {
    /// u32 array: user data, resolved scalar loads, resource info blocks
    /// (see [`SGPR_DATA_USER`], [`load_offsets`], [`res_info_offset`]).
    SgprData,
    /// Raw u32 storage buffer: a V# range or a global base address.
    Buffer(usize),
    Image(usize),
    StorageImage(usize),
    Sampler(usize),
    /// u32 array: the global data share (GDS_BYTES), shared by all shaders.
    Gds,
}

/// Size of the global data share.
pub const GDS_BYTES: u32 = 64 << 10;

#[derive(Clone, Debug)]
pub struct Binding {
    pub binding: u32,
    pub kind: BindingKind,
}

impl Binding {
    /// Plan resource index of a resource binding.
    pub fn kind_resource(&self) -> Option<usize> {
        match self.kind {
            BindingKind::Buffer(i) | BindingKind::Image(i) | BindingKind::StorageImage(i) | BindingKind::Sampler(i) => Some(i),
            _ => None,
        }
    }
}

pub struct Output {
    pub spirv: Vec<u32>,
    pub bindings: Vec<Binding>,
    /// Number of vertex parameters exported (vertex) / highest attribute read (fragment).
    pub params: u32,
    /// Bit mask of MRTs written (fragment).
    pub mrt_mask: u32,
    /// Bit mask of parameter locations exported (vertex).
    pub param_mask: u32,
    /// Sampler resources used by depth-compare instructions (their Vulkan
    /// samplers need compareEnable; others must not have it).
    pub compare_samplers: Vec<usize>,
    pub writes_depth: bool,
    pub warnings: Vec<String>,
}

/// Dwords of user data at the start of the SGPR data buffer.
pub const SGPR_DATA_USER: u32 = 32;
/// Dwords per resource info block: [byte bias of the resource base within
/// the bound range, stride, num_records, 0].
pub const RES_INFO_DWORDS: u32 = 4;

/// Offsets (in dwords) of each scalar load in the SGPR data buffer.
pub fn load_offsets(plan: &Plan) -> Vec<u32> {
    let mut v = Vec::with_capacity(plan.loads.len());
    let mut off = SGPR_DATA_USER;
    for l in &plan.loads {
        v.push(off);
        off += l.dwords;
    }
    v
}

/// Dword offset of the first resource info block in the SGPR data buffer.
pub fn res_info_offset(plan: &Plan) -> u32 {
    SGPR_DATA_USER + plan.loads.iter().map(|l| l.dwords).sum::<u32>()
}

/// Total dwords of the SGPR data buffer.
pub fn sgpr_data_dwords(plan: &Plan) -> u32 {
    res_info_offset(plan) + RES_INFO_DWORDS * plan.resources.len() as u32
}

pub(crate) struct Types {
    pub bool_: Word,
    pub u32: Word,
    pub i32: Word,
    pub f32: Word,
    pub u64: Word,
    pub v2f: Word,
    pub v3f: Word,
    pub v4f: Word,
    pub v2u: Word,
    pub v3u: Word,
    pub v4u: Word,
    pub v4i: Word,
    pub v2i: Word,
    pub v3i: Word,
    pub fn_u32: Word,
    pub fn_bool: Word,
    pub sb_u32: Word,
}

pub(crate) struct ImageRes {
    pub var: Word,
    pub img_ty: Word,
    pub dim: ImageDim,
    pub class: NumClass,
    pub storage: bool,
}

pub(crate) struct T<'a> {
    pub b: Builder,
    pub t: Types,
    pub glsl: Word,
    pub spec: &'a Spec,
    pub plan: &'a Plan,
    pub sgpr: Word,
    pub vgpr: Word,
    pub scc: Word,
    pub consts: HashMap<(Word, u32), Word>,
    pub interface: Vec<Word>,
    pub sgpr_data: Word,
    pub load_offsets: Vec<u32>,
    pub res_info: u32,
    pub buffers: HashMap<usize, Word>,
    pub images: HashMap<usize, ImageRes>,
    pub samplers: HashMap<usize, Word>,
    pub sampler_ty: Word,
    pub bindings: Vec<Binding>,
    pub next_binding: u32,
    // stage interface
    pub out_pos: Option<Word>,
    pub out_params: HashMap<u32, Word>,
    /// Sampler resources used with / without depth comparison.
    pub compare_samplers: BTreeSet<usize>,
    pub plain_samplers: BTreeSet<usize>,
    pub out_mrt: HashMap<u32, (Word, Word)>,
    pub out_depth: Option<Word>,
    pub in_attrs: HashMap<u32, Word>,
    pub frag_coord: Option<Word>,
    pub lds: Option<Word>,
    /// GDS storage buffer (when the program uses GDS instructions).
    pub gds: Option<Word>,
    /// Hardware lane index within the wave (compute: from the local index).
    pub lane_id: Option<Word>,
    /// LocalInvocationIndex (compute).
    pub local_index: Option<Word>,
    /// Workgroup array for wave64 lane exchanges (created on first use).
    pub xchg: Option<Word>,
    /// Blocks every invocation of the workgroup executes alike (barriers allowed).
    pub uniformity: crate::uniform::Uniformity,
    /// SubgroupLocalInvocationId input (created on first use).
    pub sg_lane_var: Option<Word>,
    /// (vgpr, lane) slots written by v_writelane with constant lanes (SGPR
    /// spills): per-invocation copies of those lanes' values.
    pub lane_slots: HashMap<(u32, u32), Word>,
    pub warnings: Vec<String>,
    pub params: u32,
    pub mrt_mask: u32,
    pub pc_var: Word,
    /// Probe value (see `Spec::probe`).
    pub probe_var: Option<Word>,
    pub sb_struct: Option<Word>,
    /// Binding of each raw buffer variable, and its uvec2/uvec4 views
    /// (aliases of the same binding for aligned 8/16-byte accesses).
    pub buffer_binding: HashMap<Word, u32>,
    pub buffer_views: HashMap<(Word, u32), Word>,
    /// Block struct types of the vector views (created once: types are deduplicated).
    pub view_structs: HashMap<u32, Word>,
}

pub type R<T> = Result<T, String>;

pub(crate) fn e<E: std::fmt::Debug>(r: Result<Word, E>) -> R<Word> {
    r.map_err(|x| format!("spirv builder: {x:?}"))
}

pub(crate) fn ok<E: std::fmt::Debug>(r: Result<(), E>) -> R<()> {
    r.map_err(|x| format!("spirv builder: {x:?}"))
}

impl<'a> T<'a> {
    pub fn u(&mut self, v: u32) -> Word {
        let ty = self.t.u32;
        *self.consts.entry((ty, v)).or_insert_with(|| self.b.constant_bit32(ty, v))
    }
    pub fn i(&mut self, v: i32) -> Word {
        let ty = self.t.i32;
        *self.consts.entry((ty, v as u32)).or_insert_with(|| self.b.constant_bit32(ty, v as u32))
    }
    pub fn f(&mut self, v: f32) -> Word {
        let ty = self.t.f32;
        *self.consts.entry((ty, v.to_bits())).or_insert_with(|| self.b.constant_bit32(ty, v.to_bits()))
    }
    pub fn bool_c(&mut self, v: bool) -> Word {
        let ty = self.t.bool_;
        *self.consts.entry((ty, v as u32)).or_insert_with(|| if v { self.b.constant_true(ty) } else { self.b.constant_false(ty) })
    }

    fn reg_ptr(&mut self, file: Word, idx: u32) -> R<Word> {
        let i = self.u(idx);
        let pt = self.t.fn_u32;
        e(self.b.access_chain(pt, None, file, [i]))
    }

    /// Raw 32-bit value of a scalar register slot (0..127; NULL reads 0).
    pub fn sreg(&mut self, idx: u32) -> R<Word> {
        if idx == crate::OP_NULL as u32 || idx >= 128 {
            return Ok(self.u(0));
        }
        let p = self.reg_ptr(self.sgpr, idx)?;
        e(self.b.load(self.t.u32, None, p, None, []))
    }
    pub fn set_sreg(&mut self, idx: u32, v: Word) -> R<()> {
        if idx >= 128 || idx == crate::OP_NULL as u32 {
            return Ok(()); // writes to constants / null are ignored
        }
        let p = self.reg_ptr(self.sgpr, idx)?;
        ok(self.b.store(p, v, None, []))
    }
    pub fn vreg(&mut self, idx: u32) -> R<Word> {
        let p = self.reg_ptr(self.vgpr, idx & 255)?;
        e(self.b.load(self.t.u32, None, p, None, []))
    }
    /// Unconditional VGPR write (initialization, lane-exact writes).
    pub fn set_vreg_raw(&mut self, idx: u32, v: Word) -> R<()> {
        let p = self.reg_ptr(self.vgpr, idx & 255)?;
        ok(self.b.store(p, v, None, []))
    }
    /// VGPR write predicated on this lane being active.
    pub fn set_vreg(&mut self, idx: u32, v: Word) -> R<()> {
        let act = self.active()?;
        let old = self.vreg(idx)?;
        let nv = e(self.b.select(self.t.u32, None, act, v, old))?;
        self.set_vreg_raw(idx, nv)
    }
    pub fn active(&mut self) -> R<Word> {
        let ex = self.sreg(crate::OP_EXEC_LO as u32)?;
        let one = self.u(1);
        let zero = self.u(0);
        let b = e(self.b.bitwise_and(self.t.u32, None, ex, one))?;
        e(self.b.i_not_equal(self.t.bool_, None, b, zero))
    }
    pub fn get_scc(&mut self) -> R<Word> {
        e(self.b.load(self.t.bool_, None, self.scc, None, []))
    }
    pub fn set_scc(&mut self, v: Word) -> R<()> {
        ok(self.b.store(self.scc, v, None, []))
    }

    /// 32-bit value of an operand (SGPR, VGPR, inline constant, literal, special).
    pub fn operand(&mut self, ins: &Inst, op: u16) -> R<Word> {
        Ok(match op {
            0..=127 => self.sreg(op as u32)?,
            128..=192 => self.u((op - 128) as u32),
            193..=208 => self.u((-((op - 192) as i32)) as u32),
            240 => self.u(0.5f32.to_bits()),
            241 => self.u((-0.5f32).to_bits()),
            242 => self.u(1.0f32.to_bits()),
            243 => self.u((-1.0f32).to_bits()),
            244 => self.u(2.0f32.to_bits()),
            245 => self.u((-2.0f32).to_bits()),
            246 => self.u(4.0f32.to_bits()),
            247 => self.u((-4.0f32).to_bits()),
            248 => self.u(0x3E22_F983), // 1 / (2 * pi)
            crate::OP_VCCZ | crate::OP_EXECZ => {
                let r = if op == crate::OP_VCCZ { crate::OP_VCC_LO } else { crate::OP_EXEC_LO } as u32;
                let lo = self.sreg(r)?;
                let hi = self.sreg(r + 1)?;
                let o = e(self.b.bitwise_or(self.t.u32, None, lo, hi))?;
                let z = self.u(0);
                let c = e(self.b.i_equal(self.t.bool_, None, o, z))?;
                self.bool_to_u(c)?
            }
            crate::OP_SCC => {
                let s = self.get_scc()?;
                self.bool_to_u(s)?
            }
            crate::OP_LITERAL => self.u(ins.lit.unwrap_or(0)),
            256..=511 => self.vreg((op - 256) as u32)?,
            _ => {
                self.warn(format!("{:#x}: unsupported operand {}", ins.pc, crate::operand_name(op)));
                self.u(0)
            }
        })
    }

    pub fn to_f(&mut self, v: Word) -> R<Word> {
        e(self.b.bitcast(self.t.f32, None, v))
    }
    pub fn to_u(&mut self, v: Word) -> R<Word> {
        e(self.b.bitcast(self.t.u32, None, v))
    }
    pub fn to_i(&mut self, v: Word) -> R<Word> {
        e(self.b.bitcast(self.t.i32, None, v))
    }
    pub fn glsl1(&mut self, ty: Word, op: spirv::GlslStd450Op, a: Word) -> R<Word> {
        e(self.b.ext_inst(ty, None, self.glsl, op as u32, [Operand::IdRef(a)]))
    }
    pub fn glsl2(&mut self, ty: Word, op: spirv::GlslStd450Op, a: Word, c: Word) -> R<Word> {
        e(self.b.ext_inst(ty, None, self.glsl, op as u32, [Operand::IdRef(a), Operand::IdRef(c)]))
    }
    pub fn glsl3(&mut self, ty: Word, op: spirv::GlslStd450Op, a: Word, c: Word, d: Word) -> R<Word> {
        e(self.b.ext_inst(ty, None, self.glsl, op as u32, [Operand::IdRef(a), Operand::IdRef(c), Operand::IdRef(d)]))
    }
    pub fn bool_to_u(&mut self, c: Word) -> R<Word> {
        let one = self.u(1);
        let z = self.u(0);
        e(self.b.select(self.t.u32, None, c, one, z))
    }
    pub fn warn(&mut self, s: String) {
        if !self.warnings.contains(&s) {
            self.warnings.push(s);
        }
    }

    /// Reads dword `idx` of the SGPR data buffer.
    pub fn sgpr_data(&mut self, idx: u32) -> R<Word> {
        let z = self.u(0);
        let i = self.u(idx);
        let p = e(self.b.access_chain(self.t.sb_u32, None, self.sgpr_data, [z, i]))?;
        e(self.b.load(self.t.u32, None, p, None, []))
    }

    /// Dword `k` of resource `r`'s info block (0 bias, 1 stride, 2 num_records).
    pub fn res_info(&mut self, r: usize, k: u32) -> R<Word> {
        let idx = self.res_info + RES_INFO_DWORDS * r as u32 + k;
        self.sgpr_data(idx)
    }

    /// Creates a module-scope OpVariable (never inside the current block).
    pub fn global_var(&mut self, ptr_ty: Word, sc: StorageClass) -> Word {
        let id = self.b.id();
        let inst = rspirv::dr::Instruction::new(spirv::Op::Variable, Some(ptr_ty), Some(id), vec![Operand::StorageClass(sc)]);
        self.b.module_mut().types_global_values.push(inst);
        id
    }

    fn alloc_binding(&mut self, kind: BindingKind) -> u32 {
        let n = self.spec.binding_base + self.next_binding;
        self.next_binding += 1;
        self.bindings.push(Binding { binding: n, kind });
        n
    }

    fn decorate_binding(&mut self, var: Word, binding: u32) {
        self.b.decorate(var, Decoration::DescriptorSet, [Operand::LiteralBit32(0)]);
        self.b.decorate(var, Decoration::Binding, [Operand::LiteralBit32(binding)]);
    }

    /// Storage buffer `struct { uint data[]; }` variable.
    fn sb_var(&mut self) -> Word {
        // rspirv deduplicates identical types: create and decorate the block type once.
        let st = match self.sb_struct {
            Some(s) => s,
            None => {
                let rta = self.b.type_runtime_array(self.t.u32);
                self.b.decorate(rta, Decoration::ArrayStride, [Operand::LiteralBit32(4)]);
                let st = self.b.type_struct([rta]);
                self.b.decorate(st, Decoration::Block, []);
                self.b.member_decorate(st, 0, Decoration::Offset, [Operand::LiteralBit32(0)]);
                self.sb_struct = Some(st);
                st
            }
        };
        let pt = self.b.type_pointer(None, StorageClass::StorageBuffer, st);
        self.global_var(pt, StorageClass::StorageBuffer)
    }

    /// `struct { uvecN data[]; }` view (N = 2 or 4) of the binding of raw
    /// buffer variable `var`, for aligned 8/16-byte accesses.
    pub fn vec_view(&mut self, var: Word, n: u32) -> Option<Word> {
        if let Some(&v) = self.buffer_views.get(&(var, n)) {
            return Some(v);
        }
        let binding = *self.buffer_binding.get(&var)?;
        let st = match self.view_structs.get(&n) {
            Some(&s) => s,
            None => {
                let vt = if n == 2 { self.t.v2u } else { self.t.v4u };
                let rta = self.b.type_runtime_array(vt);
                self.b.decorate(rta, Decoration::ArrayStride, [Operand::LiteralBit32(4 * n)]);
                let st = self.b.type_struct([rta]);
                self.b.decorate(st, Decoration::Block, []);
                self.b.member_decorate(st, 0, Decoration::Offset, [Operand::LiteralBit32(0)]);
                self.view_structs.insert(n, st);
                st
            }
        };
        let pt = self.b.type_pointer(None, StorageClass::StorageBuffer, st);
        let v = self.global_var(pt, StorageClass::StorageBuffer);
        self.decorate_binding(v, binding);
        self.b.decorate(v, Decoration::Aliased, []);
        self.buffer_views.insert((var, n), v);
        Some(v)
    }

    /// This invocation's index in its subgroup (for cross-lane operations).
    pub fn subgroup_lane(&mut self) -> R<Word> {
        self.b.capability(Capability::GroupNonUniform);
        let var = match self.sg_lane_var {
            Some(v) => v,
            None => {
                let v = self.input(self.t.u32, Some(BuiltIn::SubgroupLocalInvocationId), None);
                if self.spec.stage == Stage::Fragment {
                    self.b.decorate(v, Decoration::Flat, []);
                }
                self.sg_lane_var = Some(v);
                v
            }
        };
        e(self.b.load(self.t.u32, None, var, None, []))
    }

    /// Hardware lane index of this invocation within its wave.
    pub fn lane(&mut self) -> R<Word> {
        match self.lane_id {
            Some(l) => Ok(l),
            None => self.subgroup_lane(),
        }
    }

    /// Value of `v` in subgroup lane `src` (subgroup shuffle).
    /// Value of `v` in subgroup lane `src` and whether that invocation is
    /// executing here. Invocations that diverged away read as 0 (shuffles
    /// from them are undefined), like hardware lanes outside EXEC under
    /// whole-wave mode.
    pub fn lane_fetch(&mut self, v: Word, src: Word) -> R<(Word, Word)> {
        self.b.capability(Capability::GroupNonUniformShuffle);
        self.b.capability(Capability::GroupNonUniformBallot);
        let sg = self.u(spirv::Scope::Subgroup as u32);
        let t = self.bool_c(true);
        let ballot = e(self.b.group_non_uniform_ballot(self.t.v4u, None, sg, t))?;
        let live = e(self.b.group_non_uniform_ballot_bit_extract(self.t.bool_, None, sg, ballot, src))?;
        let x = e(self.b.group_non_uniform_shuffle(self.t.u32, None, sg, v, src))?;
        let z = self.u(0);
        let x = e(self.b.select(self.t.u32, None, live, x, z))?;
        Ok((x, live))
    }

    /// Lane fetch that also requires the source lane's EXEC bit.
    pub fn lane_fetch_active(&mut self, v: Word, src: Word) -> R<(Word, Word)> {
        let u = self.t.u32;
        let (x, live) = self.lane_fetch(v, src)?;
        let ex = self.sreg(crate::OP_EXEC_LO as u32)?;
        let one = self.u(1);
        let exb = e(self.b.bitwise_and(u, None, ex, one))?;
        let (sex, _) = self.lane_fetch(exb, src)?;
        let z = self.u(0);
        let act = e(self.b.i_not_equal(self.t.bool_, None, sex, z))?;
        let ok_ = e(self.b.logical_and(self.t.bool_, None, live, act))?;
        let x = e(self.b.select(u, None, ok_, x, z))?;
        Ok((x, ok_))
    }

    /// Value of `v` in hardware lane `lane` of this invocation's wave. Wave32
    /// waves are subgroups. Wave64 compute code in uniform blocks (see
    /// [`crate::uniform`]) exchanges values through workgroup memory between
    /// barriers; elsewhere wave64 reads stay within the 32-lane subgroup
    /// (reported).
    pub fn wave_read(&mut self, ins: &Inst, v: Word, lane: Word) -> R<Word> {
        let u = self.t.u32;
        let exchange = self.spec.wave64 && self.spec.stage == Stage::Compute && self.uniformity.is_uniform(ins.pc) && self.local_index.is_some();
        if !exchange {
            if self.spec.wave64 {
                self.warn(format!("{:#x}: wave64 lane read within the 32-lane subgroup", ins.pc));
            }
            let k31 = self.u(31);
            let l = e(self.b.bitwise_and(u, None, lane, k31))?;
            return Ok(self.lane_fetch(v, l)?.0);
        }
        let li = self.local_index.unwrap();
        let xchg = match self.xchg {
            Some(x) => x,
            None => {
                let n = self.spec.cs_local_size.iter().map(|&x| x.max(1)).product::<u32>().max(64);
                let len = self.u(n);
                let arr = self.b.type_array(u, len);
                let pt = self.b.type_pointer(None, StorageClass::Workgroup, arr);
                let x = self.global_var(pt, StorageClass::Workgroup);
                self.xchg = Some(x);
                x
            }
        };
        let p_wg = self.b.type_pointer(None, StorageClass::Workgroup, u);
        let wg = self.u(spirv::Scope::Workgroup as u32);
        let sem = self.u((spirv::MemorySemantics::ACQUIRE_RELEASE | spirv::MemorySemantics::WORKGROUP_MEMORY).bits());
        ok(self.b.control_barrier(wg, wg, sem))?;
        let mine = e(self.b.access_chain(p_wg, None, xchg, [li]))?;
        ok(self.b.store(mine, v, None, []))?;
        ok(self.b.control_barrier(wg, wg, sem))?;
        let m = self.u(!63);
        let base = e(self.b.bitwise_and(u, None, li, m))?;
        let k63 = self.u(63);
        let l = e(self.b.bitwise_and(u, None, lane, k63))?;
        let idx = e(self.b.i_add(u, None, base, l))?;
        let src = e(self.b.access_chain(p_wg, None, xchg, [idx]))?;
        e(self.b.load(u, None, src, None, []))
    }

    fn input(&mut self, ty: Word, builtin: Option<BuiltIn>, location: Option<u32>) -> Word {
        let pt = self.b.type_pointer(None, StorageClass::Input, ty);
        let v = self.global_var(pt, StorageClass::Input);
        if let Some(bi) = builtin {
            self.b.decorate(v, Decoration::BuiltIn, [Operand::BuiltIn(bi)]);
        }
        if let Some(l) = location {
            self.b.decorate(v, Decoration::Location, [Operand::LiteralBit32(l)]);
        }
        self.interface.push(v);
        v
    }

    fn output(&mut self, ty: Word, builtin: Option<BuiltIn>, location: Option<u32>) -> Word {
        let pt = self.b.type_pointer(None, StorageClass::Output, ty);
        let v = self.global_var(pt, StorageClass::Output);
        if let Some(bi) = builtin {
            self.b.decorate(v, Decoration::BuiltIn, [Operand::BuiltIn(bi)]);
        }
        if let Some(l) = location {
            self.b.decorate(v, Decoration::Location, [Operand::LiteralBit32(l)]);
        }
        self.interface.push(v);
        v
    }

    /// Emits `f` in a block executed only when `cond` holds.
    pub fn guarded(&mut self, cond: Word, f: &mut dyn FnMut(&mut T) -> R<()>) -> R<()> {
        let merge = self.b.id();
        let body = self.b.id();
        ok(self.b.selection_merge(merge, spirv::SelectionControl::NONE))?;
        ok(self.b.branch_conditional(cond, body, merge, []))?;
        e(self.b.begin_block(Some(body)))?;
        f(self)?;
        ok(self.b.branch(merge))?;
        e(self.b.begin_block(Some(merge)))?;
        Ok(())
    }
}

/// Splits the program into basic blocks: returns block start pcs.
fn block_starts(code: &[Inst]) -> BTreeSet<u32> {
    let mut s = BTreeSet::new();
    s.insert(code[0].pc);
    for (i, ins) in code.iter().enumerate() {
        if let Some(t) = ins.branch_target() {
            s.insert(t);
            if let Some(n) = code.get(i + 1) {
                s.insert(n.pc);
            }
        }
        if ins.is_endpgm() {
            if let Some(n) = code.get(i + 1) {
                s.insert(n.pc);
            }
        }
    }
    s
}

pub fn translate(code: &[Inst], plan: &Plan, spec: &Spec) -> R<Output> {
    if code.is_empty() {
        return Err("empty program".into());
    }
    let mut b = Builder::new();
    b.set_version(1, 3);
    b.capability(Capability::Shader);
    b.capability(Capability::ImageQuery);
    b.capability(Capability::StorageImageWriteWithoutFormat);
    b.capability(Capability::StorageImageReadWithoutFormat);
    b.capability(Capability::Int64);
    b.capability(Capability::Sampled1D);
    b.capability(Capability::Image1D);
    b.capability(Capability::SampledCubeArray);
    b.capability(Capability::ImageCubeArray);
    b.capability(Capability::ImageGatherExtended);
    b.extension("SPV_KHR_storage_buffer_storage_class");
    let glsl = b.ext_inst_import("GLSL.std.450");
    b.memory_model(spirv::AddressingModel::Logical, spirv::MemoryModel::GLSL450);
    let void = b.type_void();
    let bool_ = b.type_bool();
    let u32t = b.type_int(32, 0);
    let i32t = b.type_int(32, 1);
    let f32t = b.type_float(32, None);
    let u64t = b.type_int(64, 0);
    let v2f = b.type_vector(f32t, 2);
    let v3f = b.type_vector(f32t, 3);
    let v4f = b.type_vector(f32t, 4);
    let v2u = b.type_vector(u32t, 2);
    let v3u = b.type_vector(u32t, 3);
    let v4u = b.type_vector(u32t, 4);
    let v2i = b.type_vector(i32t, 2);
    let v3i = b.type_vector(i32t, 3);
    let v4i = b.type_vector(i32t, 4);
    let t = Types {
        bool_,
        u32: u32t,
        i32: i32t,
        f32: f32t,
        u64: u64t,
        v2f,
        v3f,
        v4f,
        v2u,
        v3u,
        v4u,
        v2i,
        v3i,
        v4i,
        fn_u32: b.type_pointer(None, StorageClass::Function, u32t),
        fn_bool: b.type_pointer(None, StorageClass::Function, bool_),
        sb_u32: b.type_pointer(None, StorageClass::StorageBuffer, u32t),
    };
    let sampler_ty = b.type_sampler();
    let mut tr = T {
        b,
        t,
        glsl,
        spec,
        plan,
        sgpr: 0,
        vgpr: 0,
        scc: 0,
        consts: HashMap::new(),
        interface: Vec::new(),
        sgpr_data: 0,
        load_offsets: load_offsets(plan),
        res_info: res_info_offset(plan),
        buffers: HashMap::new(),
        images: HashMap::new(),
        samplers: HashMap::new(),
        sampler_ty,
        bindings: Vec::new(),
        next_binding: 0,
        out_pos: None,
        out_params: HashMap::new(),
        compare_samplers: BTreeSet::new(),
        plain_samplers: BTreeSet::new(),
        out_mrt: HashMap::new(),
        out_depth: None,
        in_attrs: HashMap::new(),
        frag_coord: None,
        lds: None,
        gds: None,
        lane_id: None,
        local_index: None,
        xchg: None,
        uniformity: crate::uniform::analyze(code),
        sg_lane_var: None,
        lane_slots: HashMap::new(),
        warnings: Vec::new(),
        params: 0,
        mrt_mask: 0,
        pc_var: 0,
        probe_var: None,
        sb_struct: None,
        buffer_binding: HashMap::new(),
        buffer_views: HashMap::new(),
        view_structs: HashMap::new(),
    };
    // SGPR data buffer (binding base + 0).
    let sd = tr.sb_var();
    let bnd = tr.alloc_binding(BindingKind::SgprData);
    tr.decorate_binding(sd, bnd);
    tr.sgpr_data = sd;
    if code.iter().any(|i| i.enc == Enc::Ds && i.ds_gds()) {
        let v = tr.sb_var();
        let bnd = tr.alloc_binding(BindingKind::Gds);
        tr.decorate_binding(v, bnd);
        tr.gds = Some(v);
    }
    // Resource bindings in plan order.
    for (i, r) in plan.resources.iter().enumerate() {
        match (r.kind, spec.resources.get(i)) {
            (ResKind::Buffer | ResKind::Global, _) => {
                let v = tr.sb_var();
                let bnd = tr.alloc_binding(BindingKind::Buffer(i));
                tr.decorate_binding(v, bnd);
                // Vector views of the same binding alias it.
                tr.b.decorate(v, Decoration::Aliased, []);
                tr.buffer_binding.insert(v, bnd);
                tr.buffers.insert(i, v);
            }
            (ResKind::Image, Some(ResSpec::Image { dim, class })) => {
                let (dim, class) = (*dim, *class);
                let sampled_ty = match class {
                    NumClass::Float => tr.t.f32,
                    NumClass::Sint => tr.t.i32,
                    NumClass::Uint => tr.t.u32,
                };
                // 1D images are 2D images of height 1 on the host (so they can
                // alias other images); coordinates get a zero y component.
                let (sd, arrayed, ms) = match dim {
                    ImageDim::D1 | ImageDim::D2 => (spirv::Dim::Dim2D, 0, 0),
                    ImageDim::D3 => (spirv::Dim::Dim3D, 0, 0),
                    ImageDim::Cube => (spirv::Dim::DimCube, 0, 0),
                    ImageDim::CubeArray => (spirv::Dim::DimCube, 1, 0),
                    ImageDim::D1Array | ImageDim::D2Array => (spirv::Dim::Dim2D, 1, 0),
                    ImageDim::D2Msaa => (spirv::Dim::Dim2D, 0, 1),
                    ImageDim::D2MsaaArray => (spirv::Dim::Dim2D, 1, 1),
                };
                let storage = r.written;
                let img_ty = tr.b.type_image(sampled_ty, sd, 0, arrayed, ms, if storage { 2 } else { 1 }, spirv::ImageFormat::Unknown, None);
                let pt = tr.b.type_pointer(None, StorageClass::UniformConstant, img_ty);
                let v = tr.global_var(pt, StorageClass::UniformConstant);
                let bnd = tr.alloc_binding(if storage { BindingKind::StorageImage(i) } else { BindingKind::Image(i) });
                tr.decorate_binding(v, bnd);
                tr.images.insert(i, ImageRes { var: v, img_ty, dim, class, storage });
            }
            (ResKind::Sampler, _) => {
                let pt = tr.b.type_pointer(None, StorageClass::UniformConstant, tr.sampler_ty);
                let v = tr.global_var(pt, StorageClass::UniformConstant);
                let bnd = tr.alloc_binding(BindingKind::Sampler(i));
                tr.decorate_binding(v, bnd);
                tr.samplers.insert(i, v);
            }
            (ResKind::Image, _) => return Err(format!("image resource {i} has no specialization")),
        }
    }
    // Function.
    let fn_ty = tr.b.type_function(void, []);
    let func = e(tr.b.begin_function(void, None, spirv::FunctionControl::NONE, fn_ty))?;
    e(tr.b.begin_block(None))?;
    let arr_s = {
        let n = tr.u(128);
        tr.b.type_array(tr.t.u32, n)
    };
    let arr_v = {
        let n = tr.u(256);
        tr.b.type_array(tr.t.u32, n)
    };
    let p_arr_s = tr.b.type_pointer(None, StorageClass::Function, arr_s);
    let p_arr_v = tr.b.type_pointer(None, StorageClass::Function, arr_v);
    tr.sgpr = tr.b.variable(p_arr_s, None, StorageClass::Function, None);
    tr.vgpr = tr.b.variable(p_arr_v, None, StorageClass::Function, None);
    tr.scc = tr.b.variable(tr.t.fn_bool, None, StorageClass::Function, None);
    tr.pc_var = tr.b.variable(tr.t.fn_u32, None, StorageClass::Function, None);
    if tr.spec.probe.is_some() {
        let z = tr.u(0);
        tr.probe_var = Some(tr.b.variable(tr.t.fn_u32, None, StorageClass::Function, Some(z)));
    }
    for ins in code {
        if let Some((true, vgpr, lane)) = ins.const_lane_op() {
            if !tr.lane_slots.contains_key(&(vgpr as u32, lane)) {
                let v = tr.b.variable(tr.t.fn_u32, None, StorageClass::Function, None);
                tr.lane_slots.insert((vgpr as u32, lane), v);
            }
        }
    }
    // Initial state: all registers defined (zero), this lane active.
    let f = tr.bool_c(false);
    tr.set_scc(f)?;
    let zero = tr.u(0);
    let zs = tr.b.constant_null(arr_s);
    ok(tr.b.store(tr.sgpr, zs, None, []))?;
    let zv = tr.b.constant_null(arr_v);
    ok(tr.b.store(tr.vgpr, zv, None, []))?;
    // EXEC holds this lane's bit (bit 0) only.
    let one = tr.u(1);
    tr.set_sreg(crate::OP_EXEC_LO as u32, one)?;
    tr.set_sreg(crate::OP_EXEC_HI as u32, zero)?;
    for i in 0..spec.user_sgprs.min(SGPR_DATA_USER as u16) as u32 {
        let v = tr.sgpr_data(i)?;
        tr.set_sreg(spec.user_base as u32 + i, v)?;
    }
    mem::stage_inputs(&mut tr)?;

    // Body.
    let starts = block_starts(code);
    if starts.len() == 1 {
        for ins in code {
            if ins.is_endpgm() {
                break;
            }
            translate_inst(&mut tr, ins)?;
        }
        ok(tr.b.ret())?;
    } else {
        emit_dispatch_loop(&mut tr, code, &starts)?;
    }
    ok(tr.b.end_function())?;

    let model = match spec.stage {
        Stage::Vertex => ExecutionModel::Vertex,
        Stage::Fragment => ExecutionModel::Fragment,
        Stage::Compute => ExecutionModel::GLCompute,
    };
    let iface = tr.interface.clone();
    tr.b.entry_point(model, func, "main", iface);
    match spec.stage {
        Stage::Fragment => {
            tr.b.execution_mode(func, ExecutionMode::OriginUpperLeft, []);
            if tr.out_depth.is_some() {
                tr.b.execution_mode(func, ExecutionMode::DepthReplacing, []);
            }
        }
        Stage::Compute => {
            let l = spec.cs_local_size;
            tr.b.execution_mode(func, ExecutionMode::LocalSize, [l[0].max(1), l[1].max(1), l[2].max(1)]);
        }
        Stage::Vertex => {}
    }
    let spirv = tr.b.module().assemble();
    let param_mask = tr.out_params.keys().filter(|&&p| p < 32).fold(0u32, |m, &p| m | 1 << p);
    let compare_samplers = tr.compare_samplers.iter().copied().collect();
    Ok(Output { spirv, bindings: tr.bindings, params: tr.params, mrt_mask: tr.mrt_mask, param_mask, compare_samplers, writes_depth: tr.out_depth.is_some(), warnings: tr.warnings })
}

const PC_EXIT: u32 = 0xFFFF_FFFF;

/// Branch condition of a conditional SOPP branch (None = always taken).
fn branch_cond(tr: &mut T, ins: &Inst) -> R<Option<Word>> {
    let name = ins.name().unwrap_or("");
    let bt = tr.t.bool_;
    Ok(match name {
        "s_branch" => None,
        "s_cbranch_scc0" => {
            let s = tr.get_scc()?;
            Some(e(tr.b.logical_not(bt, None, s))?)
        }
        "s_cbranch_scc1" => Some(tr.get_scc()?),
        "s_cbranch_vccz" | "s_cbranch_vccnz" | "s_cbranch_execz" | "s_cbranch_execnz" => {
            let src = if name.contains("vcc") { crate::OP_VCC_LO } else { crate::OP_EXEC_LO } as u32;
            let lo = tr.sreg(src)?;
            let hi = if tr.spec.wave64 { tr.sreg(src + 1)? } else { tr.u(0) };
            let o = e(tr.b.bitwise_or(tr.t.u32, None, lo, hi))?;
            let z = tr.u(0);
            let zero = e(tr.b.i_equal(bt, None, o, z))?;
            Some(if name.ends_with('z') && !name.ends_with("nz") { zero } else { e(tr.b.logical_not(bt, None, zero))? })
        }
        _ => {
            tr.warn(format!("{:#x}: {name} never taken", ins.pc));
            Some(tr.bool_c(false))
        }
    })
}

fn emit_dispatch_loop(tr: &mut T, code: &[Inst], starts: &BTreeSet<u32>) -> R<()> {
    let first = code[0].pc;
    let s0 = tr.u(first);
    ok(tr.b.store(tr.pc_var, s0, None, []))?;
    let header = tr.b.id();
    let dispatch = tr.b.id();
    let sw_merge = tr.b.id();
    let cont = tr.b.id();
    let exit = tr.b.id();
    ok(tr.b.branch(header))?;
    e(tr.b.begin_block(Some(header)))?;
    ok(tr.b.loop_merge(exit, cont, spirv::LoopControl::NONE, []))?;
    ok(tr.b.branch(dispatch))?;
    e(tr.b.begin_block(Some(dispatch)))?;
    let pc = e(tr.b.load(tr.t.u32, None, tr.pc_var, None, []))?;
    let block_ids: Vec<(u32, Word)> = starts.iter().map(|&s| (s, tr.b.id())).collect();
    ok(tr.b.selection_merge(sw_merge, spirv::SelectionControl::NONE))?;
    let targets: Vec<(Operand, Word)> = block_ids.iter().map(|&(s, id)| (Operand::LiteralBit32(s), id)).collect();
    ok(tr.b.switch(pc, sw_merge, targets))?;
    let starts_vec: Vec<u32> = starts.iter().copied().collect();
    for (bi, &(start, id)) in block_ids.iter().enumerate() {
        e(tr.b.begin_block(Some(id)))?;
        let end = starts_vec.get(bi + 1).copied().unwrap_or(u32::MAX);
        let mut next_pc = PC_EXIT;
        let mut terminated = false;
        for (k, ins) in code.iter().enumerate() {
            if ins.pc < start || ins.pc >= end {
                continue;
            }
            next_pc = code.get(k + 1).map(|n| n.pc).unwrap_or(PC_EXIT);
            if ins.is_endpgm() {
                let x = tr.u(PC_EXIT);
                ok(tr.b.store(tr.pc_var, x, None, []))?;
                terminated = true;
                break;
            }
            if let Some(target) = ins.branch_target() {
                let cond = branch_cond(tr, ins)?;
                let tv = tr.u(target);
                let val = match cond {
                    None => tv,
                    Some(c) => {
                        let fv = tr.u(next_pc);
                        e(tr.b.select(tr.t.u32, None, c, tv, fv))?
                    }
                };
                ok(tr.b.store(tr.pc_var, val, None, []))?;
                terminated = true;
                break;
            }
            translate_inst(tr, ins)?;
        }
        if !terminated {
            let x = tr.u(if end == u32::MAX { PC_EXIT } else { next_pc });
            ok(tr.b.store(tr.pc_var, x, None, []))?;
        }
        ok(tr.b.branch(sw_merge))?;
    }
    e(tr.b.begin_block(Some(sw_merge)))?;
    ok(tr.b.branch(cont))?;
    e(tr.b.begin_block(Some(cont)))?;
    let pc2 = e(tr.b.load(tr.t.u32, None, tr.pc_var, None, []))?;
    let ex = tr.u(PC_EXIT);
    let done = e(tr.b.i_equal(tr.t.bool_, None, pc2, ex))?;
    ok(tr.b.branch_conditional(done, exit, header, []))?;
    e(tr.b.begin_block(Some(exit)))?;
    ok(tr.b.ret())?;
    Ok(())
}

fn translate_inst(tr: &mut T, ins: &Inst) -> R<()> {
    translate_one(tr, ins)?;
    if let (Some((pc, vg)), Some(var)) = (tr.spec.probe, tr.probe_var) {
        if ins.pc == pc {
            let v = tr.vreg(vg)?;
            ok(tr.b.store(var, v, None, []))?;
        }
    }
    Ok(())
}

fn translate_one(tr: &mut T, ins: &Inst) -> R<()> {
    match ins.enc {
        Enc::Sop2 | Enc::Sop1 | Enc::Sopc | Enc::Sopk | Enc::Sopp => alu::scalar(tr, ins),
        Enc::Vop1 | Enc::Vop2 | Enc::Vopc | Enc::Vop3 | Enc::Vop3p => alu::vector(tr, ins),
        Enc::Smem => mem::smem(tr, ins),
        Enc::Vintrp => mem::interp(tr, ins),
        Enc::Mubuf | Enc::Mtbuf => mem::buffer(tr, ins),
        Enc::Mimg => mem::image(tr, ins),
        Enc::Exp => mem::export(tr, ins),
        Enc::Ds => mem::ds(tr, ins),
        Enc::Global => mem::global(tr, ins),
        Enc::Flat | Enc::Scratch => {
            tr.warn(format!("{:#x}: {} unsupported", ins.pc, ins.name().unwrap_or("flat/scratch")));
            Ok(())
        }
    }
}
