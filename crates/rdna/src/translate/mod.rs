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
//!
//! Hardware lanes of a wave execute in lockstep, and shaders rely on it:
//! lanes exchange data through LDS without barriers. Invocations do not run
//! in lockstep, so LDS accesses get workgroup barriers wherever another
//! lane's earlier access could conflict (in blocks every invocation executes
//! alike).
//!
//! Programs whose loops nest (each loop a contiguous address range from its
//! header to its last back edge, entered only at the header) need no
//! dispatcher: their blocks are emitted in address order, each run when the
//! next address is its start, and each loop range as a structured loop that
//! repeats while the next address is its header (an invocation's path visits
//! the blocks of one iteration in increasing order; exits, early ends and
//! jumps to an outer header fall through the remaining guards). The driver
//! then sees structured control flow instead of one loop over a switch.
//! Others (overlapping or irreducible loops) keep the dispatcher. Compute programs that use LDS run in *lockstep mode* when their
//! workgroup is a single wave, or when no scalar value can differ between
//! its waves (no v_readlane / v_readfirstlane, no wave id): lane-dependent
//! branches are decided for the whole workgroup by a vote (as the hardware
//! decides them for each wave), so every block is uniform and all LDS
//! accesses are ordered. With several waves a wave whose lanes are all
//! inactive runs a region with EXEC empty instead of skipping it.
//! Single-wave compute programs that read compare masks as values (ballots
//! they store or count: [`crate::wavemask::reads_mask_values`]) run in
//! lockstep mode as well, with or without LDS.
//!
//! Outside lockstep mode invocations that took different paths reach a
//! uniform block after different numbers of dispatcher steps, so the
//! workgroup barriers placed there (LDS ordering, wave64 lane exchanges,
//! s_barrier) would be different dynamic instances: invalid in Vulkan, and on
//! NVIDIA a warp whose threads arrive at different steps passes the barrier
//! early (a 64-lane luminance reduction raced and the exposure came out
//! wrong, depending on timing). Such blocks are *sync blocks*: the dispatcher
//! runs in rounds, and an invocation that reaches a sync block after other
//! blocks of its round ends the round there; every invocation then starts
//! the next round at that block together (see `emit_dispatch_loop`).
//!
//! Single-wave lockstep programs run in *wave-mask mode*: lane masks (EXEC,
//! VCC, compare results) have the hardware layout, bit i for lane i, so
//! masks used as values (ballots kept in LDS, counted, scanned with s_ff1)
//! and the scalar state derived from them agree across invocations, and
//! branches need no votes. A compare computes its own lane's bit; a ballot
//! through workgroup memory completes the mask wherever it may be read as a
//! whole (see [`crate::wavemask`]); elsewhere an invocation keeps a partial
//! mask whose own bit is right.

mod alu;
pub mod gs;
mod mem;

use crate::analysis::{Plan, ResKind};
use crate::{Enc, Inst};
use rspirv::binary::Assemble;
use rspirv::dr::{Builder, Operand};
use rspirv::spirv::{self, BuiltIn, Capability, Decoration, ExecutionMode, ExecutionModel, StorageClass, Word};
use std::collections::BTreeSet;
use crate::HashMap;

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
    /// `depth`: bound as a depth/stencil view (depth-comparison sampling
    /// works; on color views it is emulated, see `mem::compare_texels`).
    Image { dim: ImageDim, class: NumClass, depth: bool },
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
    /// Fragment barycentrics and per-vertex inputs are available
    /// (VK_KHR_fragment_shader_barycentric): the barycentric VGPRs hold the
    /// real (i, j) and v_interp_mov reads each vertex's value.
    pub barycentric: bool,
    pub mrt: [MrtSpec; 8],
    pub cs_local_size: [u32; 3],
    /// COMPUTE_PGM_RSRC2 (TGID enables, TG size, TIDIG count, LDS size).
    pub cs_rsrc2: u32,
    pub resources: Vec<ResSpec>,
    /// Binding index of the first binding (stages share one descriptor set).
    pub binding_base: u32,
    /// Debugging: (pc, register): the value of the register after the
    /// instruction at `pc` replaces the color of every MRT export ((v, v, v,
    /// 1) as float), or (compute) the data of every image and buffer store.
    /// The register is a VGPR number, or an SGPR with PROBE_SGPR; with
    /// PROBE_UINT / PROBE_SINT its integer value is converted to float.
    pub probe: Option<(u32, u32)>,
    /// Debugging (compute): (pc, workgroup id): after each execution of the
    /// instruction at pc, the invocations of that workgroup write their
    /// registers to the debug binding (see [`DUMP_STRIDE`]). Vertex stage:
    /// (pc, [first vertex index, instance index, 0]): the invocations of that
    /// instance with vertex indices from the first one (1024 slots) do.
    pub dump: Option<(u32, [u32; 3])>,
    /// The loop-and-switch dispatcher instead of structured control flow:
    /// for every program, or for programs with loops only (A/B comparisons).
    pub force_dispatcher: bool,
    pub dispatcher_for_loops: bool,
}

/// Entries of an image table bound as one array of sampled images
/// (`ResKind::ImageTable`); later entries are not reachable.
pub const IMAGE_TABLE_LEN: u32 = 64;

/// Probe register flags (see `Spec::probe`).
pub const PROBE_SGPR: u32 = 1 << 12;
pub const PROBE_UINT: u32 = 1 << 13;
pub const PROBE_SINT: u32 = 1 << 14;

/// Dwords per invocation in the debug binding: VGPRs 0..256, SGPRs at 256
/// (128), SCC at 384, execution count at 385.
pub const DUMP_STRIDE: u32 = 512;
/// Size of the debug binding in dwords (up to 1024 invocations).
pub const DUMP_DWORDS: u32 = 1024 * DUMP_STRIDE;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BindingKind {
    /// u32 array: user data, resolved scalar loads, resource info blocks
    /// (see [`SGPR_DATA_USER`], [`load_offsets`], [`res_info_offset`]).
    SgprData,
    /// Raw u32 storage buffer: a V# range or a global base address.
    Buffer(usize),
    Image(usize),
    StorageImage(usize),
    /// IMAGE_TABLE_LEN sampled images: the entries of an image table.
    ImageArray(usize),
    Sampler(usize),
    /// u32 array: the global data share (GDS_BYTES), shared by all shaders.
    Gds,
    /// u32 array of DUMP_DWORDS: register dumps (`Spec::dump`).
    Debug,
    /// u32 array of WATCHDOG_DWORDS: the loop watchdog's record (see [`LOOP_LIMIT`]).
    Watchdog,
}

/// Size of the global data share.
pub const GDS_BYTES: u32 = 64 << 10;

/// Loop watchdog: an invocation of a program with backward branches leaves
/// after this many blocks. A lane that never leaves a loop would otherwise
/// run until the driver declares the device lost (seen in Outpost Passage),
/// which also stalls other applications' GPU work. A stopped invocation adds
/// 1 to dword 0 of the watchdog binding and stores its program address
/// (`Spec::code_base`) in dwords 1 (low) and 2 (high).
pub const LOOP_LIMIT: u32 = 1 << 20;
/// Size of the watchdog binding in dwords.
pub const WATCHDOG_DWORDS: u32 = 4;

#[derive(Clone, Debug)]
pub struct Binding {
    pub binding: u32,
    pub kind: BindingKind,
}

impl Binding {
    /// Plan resource index of a resource binding.
    pub fn kind_resource(&self) -> Option<usize> {
        match self.kind {
            BindingKind::Buffer(i) | BindingKind::Image(i) | BindingKind::StorageImage(i) | BindingKind::ImageArray(i) | BindingKind::Sampler(i) => Some(i),
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
/// the bound range, stride, bound bytes, records an index may address
/// (structured range check; u32::MAX: none)].
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
    /// Bound as a depth/stencil view (`ResSpec::Image::depth`).
    pub depth: bool,
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
    /// Image tables: resource -> pc of the load that reads its T#, and the
    /// byte offset each such load used (Function variables).
    pub image_tables: HashMap<usize, u32>,
    pub table_offsets: HashMap<u32, Word>,
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
    /// Attributes read per vertex (v_interp_mov of P10/P20, with
    /// `Spec::barycentric`), and their per-vertex input arrays.
    pub per_vertex_attrs: BTreeSet<u32>,
    pub per_vertex_in: HashMap<u32, Word>,
    /// BaryCoordKHR / BaryCoordNoPerspKHR inputs (loaded at entry).
    pub bary: Option<(Word, Word)>,
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
    /// Workgroup array for wave64 first-lane reads: values, then flags.
    pub xchg_first: Option<Word>,
    /// Blocks every invocation of the workgroup executes alike (barriers allowed).
    pub uniformity: crate::uniform::Uniformity,
    /// Lockstep mode (see the module documentation).
    pub lockstep: bool,
    /// Lockstep ballots: workgroup u32[6] (three rotating (lo, hi) slots)
    /// and this invocation's ballot counter (mod 3).
    pub vote: Option<Word>,
    pub vote_ctr: Option<Word>,
    /// Sync blocks (see the module documentation) and the dispatcher steps
    /// of the current round.
    pub sync_starts: Vec<u32>,
    pub sync_steps: Option<Word>,
    /// LDS accesses since the last workgroup barrier: bit 0 reads, bit 1
    /// writes (3 at block entry: unknown).
    pub lds_state: u8,
    /// SubgroupLocalInvocationId input (created on first use).
    pub sg_lane_var: Option<Word>,
    /// (vgpr, lane) slots written by v_writelane with constant lanes (SGPR
    /// spills): per-invocation copies of those lanes' values.
    pub lane_slots: HashMap<(u32, u32), Word>,
    /// Lane reads that take the invocation's own value (see
    /// [`atomic_result_reads`]).
    pub own_lane_reads: BTreeSet<u32>,
    /// Lane-0 elections: compare pc -> source operand (0/1) holding the lane
    /// index, compared as lane 0 (see [`lane0_elections`]).
    pub elections: HashMap<u32, usize>,
    /// Wave-mask mode (see the module documentation).
    pub wave_masks: bool,
    /// Wave-mask mode: pcs of the mask writes that produce complete masks.
    pub complete: crate::HashSet<u32>,
    /// Wave-mask mode: whether this lane's bit lies in the high dword of a
    /// mask, and its position in that dword.
    pub own_bit: Option<(Word, Word)>,
    pub warnings: Vec<String>,
    pub params: u32,
    pub mrt_mask: u32,
    pub pc_var: Word,
    /// Probe value (see `Spec::probe`).
    pub probe_var: Option<Word>,
    /// Loop watchdog binding and this invocation's block counter (programs
    /// with backward branches; see [`LOOP_LIMIT`]).
    pub watchdog: Option<Word>,
    pub loop_ctr: Option<Word>,
    /// Debug binding (see `Spec::dump`) and WorkgroupId (compute).
    pub dump_buf: Option<Word>,
    pub wg_id: Option<Word>,
    /// VertexIndex and InstanceIndex (vertex stage; register dumps).
    pub vertex_ids: Option<(Word, Word)>,
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
        let lo = self.sreg(crate::OP_EXEC_LO as u32)?;
        let hi = if self.own_bit.is_some() { self.sreg(crate::OP_EXEC_HI as u32)? } else { self.u(0) };
        self.mask_bit(lo, hi)
    }

    /// This lane's bit of a lane mask (lo, hi): bit 0, or in wave-mask mode
    /// the bit of the lane's index.
    pub fn mask_bit(&mut self, lo: Word, hi: Word) -> R<Word> {
        let u = self.t.u32;
        let w = match self.own_bit {
            Some((in_hi, sh)) => {
                let w = e(self.b.select(u, None, in_hi, hi, lo))?;
                e(self.b.shift_right_logical(u, None, w, sh))?
            }
            None => lo,
        };
        let one = self.u(1);
        let zero = self.u(0);
        let b = e(self.b.bitwise_and(u, None, w, one))?;
        e(self.b.i_not_equal(self.t.bool_, None, b, zero))
    }

    /// Lane mask (lo, hi) of a per-lane condition written at `pc` (lanes
    /// outside EXEC 0): this lane's bit alone (bit 0; in wave-mask mode the
    /// bit of its index), or in wave-mask mode the complete mask where the
    /// value may be read as a whole (a ballot).
    pub fn lane_mask(&mut self, pc: u32, c: Word) -> R<(Word, Word)> {
        let u = self.t.u32;
        let z = self.u(0);
        if c == self.bool_c(false) {
            return Ok((z, z));
        }
        let act = self.active()?;
        let c = e(self.b.logical_and(self.t.bool_, None, c, act))?;
        match self.own_bit {
            Some(_) if self.complete.contains(&pc) => self.wave_ballot(c),
            Some((in_hi, sh)) => {
                let one = self.u(1);
                let bit = e(self.b.shift_left_logical(u, None, one, sh))?;
                let m = e(self.b.select(u, None, c, bit, z))?;
                let lo = e(self.b.select(u, None, in_hi, z, m))?;
                let hi = e(self.b.select(u, None, in_hi, m, z))?;
                Ok((lo, hi))
            }
            None => Ok((self.bool_to_u(c)?, z)),
        }
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
                let hi = if self.spec.wave64 { self.sreg(r + 1)? } else { self.u(0) };
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
        let act = self.active()?;
        let one = self.u(1);
        let z = self.u(0);
        let exb = e(self.b.select(u, None, act, one, z))?;
        let (sex, _) = self.lane_fetch(exb, src)?;
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
        if exchange {
            self.lds_state = 0;
        }
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

    /// v_readfirstlane: the value of `v` in the first active lane of this
    /// invocation's wave (lane 0 when EXEC is empty), the same in every lane.
    /// Exact within a subgroup; wave64 compute code in uniform blocks
    /// combines the two 32-lane halves through workgroup memory; elsewhere a
    /// wave64 read stays within the 32-lane subgroup (reported).
    pub fn wave_read_first(&mut self, ins: &Inst, v: Word) -> R<Word> {
        let u = self.t.u32;
        let bt = self.t.bool_;
        self.b.capability(Capability::GroupNonUniformBallot);
        let sg = self.u(spirv::Scope::Subgroup as u32);
        let one = self.u(1);
        let z = self.u(0);
        let act = self.active()?;
        let ballot = e(self.b.group_non_uniform_ballot(self.t.v4u, None, sg, act))?;
        let bits = e(self.b.composite_extract(u, None, ballot, [0]))?;
        let any = e(self.b.i_not_equal(bt, None, bits, z))?;
        let lsb = self.glsl1(self.t.i32, spirv::GlslStd450Op::FindILsb, bits)?;
        let lsb = self.to_u(lsb)?;
        let first = e(self.b.select(u, None, any, lsb, z))?;
        let (val, _) = self.lane_fetch(v, first)?;
        let threads: u32 = self.spec.cs_local_size.iter().map(|&x| x.max(1)).product();
        let compute = self.spec.stage == Stage::Compute;
        if !self.spec.wave64 || (compute && threads <= 32) {
            return Ok(val);
        }
        if !(compute && self.uniformity.is_uniform(ins.pc) && self.local_index.is_some()) {
            self.warn(format!("{:#x}: wave64 first-lane read within the 32-lane subgroup", ins.pc));
            return Ok(val);
        }
        // Each lane publishes its half's candidate and whether that half has
        // an active lane; the wave takes the lower half's unless it has none.
        self.lds_state = 0;
        let li = self.local_index.unwrap();
        let n = threads.next_multiple_of(64);
        let xf = match self.xchg_first {
            Some(x) => x,
            None => {
                let len = self.u(2 * n);
                let arr = self.b.type_array(u, len);
                let pt = self.b.type_pointer(None, StorageClass::Workgroup, arr);
                let x = self.global_var(pt, StorageClass::Workgroup);
                self.xchg_first = Some(x);
                x
            }
        };
        let p_wg = self.b.type_pointer(None, StorageClass::Workgroup, u);
        let wg = self.u(spirv::Scope::Workgroup as u32);
        let sem = self.u((spirv::MemorySemantics::ACQUIRE_RELEASE | spirv::MemorySemantics::WORKGROUP_MEMORY).bits());
        let kn = self.u(n);
        ok(self.b.control_barrier(wg, wg, sem))?;
        let p = e(self.b.access_chain(p_wg, None, xf, [li]))?;
        ok(self.b.store(p, val, None, []))?;
        let anyu = e(self.b.select(u, None, any, one, z))?;
        let i = e(self.b.i_add(u, None, kn, li))?;
        let p = e(self.b.access_chain(p_wg, None, xf, [i]))?;
        ok(self.b.store(p, anyu, None, []))?;
        ok(self.b.control_barrier(wg, wg, sem))?;
        let m = self.u(!63);
        let base = e(self.b.bitwise_and(u, None, li, m))?;
        let k32 = self.u(32);
        let upper = e(self.b.i_add(u, None, base, k32))?;
        let load = |tr: &mut Self, idx: Word| -> R<Word> {
            let p = e(tr.b.access_chain(p_wg, None, xf, [idx]))?;
            e(tr.b.load(u, None, p, None, []))
        };
        let lower_val = load(self, base)?;
        let upper_val = load(self, upper)?;
        let i = e(self.b.i_add(u, None, kn, base))?;
        let lower_any = load(self, i)?;
        let i = e(self.b.i_add(u, None, kn, upper))?;
        let upper_any = load(self, i)?;
        // Lanes past the end of a partial last wave do not exist.
        let kt = self.u(threads);
        let exists = e(self.b.u_less_than(bt, None, upper, kt))?;
        let lower_none = e(self.b.i_equal(bt, None, lower_any, z))?;
        let upper_some = e(self.b.i_not_equal(bt, None, upper_any, z))?;
        let use_upper = e(self.b.logical_and(bt, None, lower_none, upper_some))?;
        let use_upper = e(self.b.logical_and(bt, None, use_upper, exists))?;
        e(self.b.select(u, None, use_upper, upper_val, lower_val))
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

    /// Data VGPR of a store; a compute probe replaces it (see `Spec::probe`).
    pub fn store_vreg(&mut self, idx: u32) -> R<Word> {
        match (self.spec.stage, self.probe_var) {
            (Stage::Compute, Some(pv)) => e(self.b.load(self.t.u32, None, pv, None, [])),
            _ => self.vreg(idx),
        }
    }

    /// Workgroup barrier ordering workgroup memory (uniform blocks only).
    pub fn workgroup_barrier(&mut self) -> R<()> {
        let wg = self.u(spirv::Scope::Workgroup as u32);
        let sem = self.u((spirv::MemorySemantics::ACQUIRE_RELEASE | spirv::MemorySemantics::WORKGROUP_MEMORY).bits());
        ok(self.b.control_barrier(wg, wg, sem))?;
        self.lds_state = 0;
        Ok(())
    }

    /// Orders an LDS access after other lanes' earlier accesses, as lockstep
    /// hardware lanes are: a barrier before a read that follows writes, or a
    /// write that follows any access, since the last barrier.
    pub fn lds_order(&mut self, pc: u32, write: bool) -> R<()> {
        if self.spec.stage != Stage::Compute {
            return Ok(());
        }
        if !self.uniformity.is_uniform(pc) {
            self.warn(format!("{pc:#x}: LDS access in divergent code is not ordered across lanes"));
            return Ok(());
        }
        let need = if write { self.lds_state != 0 } else { self.lds_state & 2 != 0 };
        if need {
            self.workgroup_barrier()?;
        }
        self.lds_state |= if write { 2 } else { 1 };
        Ok(())
    }

    /// Ballot of a per-lane condition over the workgroup (lockstep mode: the
    /// workgroup is one wave): (lo, hi) lane masks. Each subgroup ORs its
    /// lanes' bits of either half and one of its invocations adds them to a
    /// slot. Three rotating slots of workgroup memory need one barrier per
    /// ballot: a slot is cleared (by lane 0) after the barrier of the ballot
    /// following its use, when every lane has read it, and reused after the
    /// next barrier.
    pub fn wave_ballot(&mut self, c: Word) -> R<(Word, Word)> {
        let (Some(vote), Some(ctr), Some(li)) = (self.vote, self.vote_ctr, self.local_index) else {
            return Err("wave ballot outside lockstep mode".into());
        };
        let u = self.t.u32;
        let k = e(self.b.load(u, None, ctr, None, []))?;
        let one = self.u(1);
        let two = self.u(2);
        let three = self.u(3);
        let kn = e(self.b.i_add(u, None, k, one))?;
        let kn = e(self.b.u_mod(u, None, kn, three))?;
        ok(self.b.store(ctr, kn, None, []))?;
        let base = e(self.b.i_mul(u, None, k, two))?;
        let k31 = self.u(31);
        let pos = e(self.b.bitwise_and(u, None, li, k31))?;
        let bit = e(self.b.shift_left_logical(u, None, one, pos))?;
        let z = self.u(0);
        let mine = e(self.b.select(u, None, c, bit, z))?;
        let k32 = self.u(32);
        let lower = e(self.b.u_less_than(self.t.bool_, None, li, k32))?;
        let lo_part = e(self.b.select(u, None, lower, mine, z))?;
        let hi_part = e(self.b.select(u, None, lower, z, mine))?;
        self.b.capability(Capability::GroupNonUniform);
        self.b.capability(Capability::GroupNonUniformArithmetic);
        let sg = self.u(spirv::Scope::Subgroup as u32);
        let lo_sum = e(self.b.group_non_uniform_bitwise_or(u, None, sg, spirv::GroupOperation::Reduce, lo_part, None))?;
        let hi_sum = e(self.b.group_non_uniform_bitwise_or(u, None, sg, spirv::GroupOperation::Reduce, hi_part, None))?;
        let elected = e(self.b.group_non_uniform_elect(self.t.bool_, None, sg))?;
        let p_wg = self.b.type_pointer(None, StorageClass::Workgroup, u);
        self.guarded(elected, &mut |tr| {
            let scope = tr.u(spirv::Scope::Workgroup as u32);
            let sem = tr.u(0);
            let p = e(tr.b.access_chain(p_wg, None, vote, [base]))?;
            e(tr.b.atomic_or(u, None, p, scope, sem, lo_sum))?;
            let base1 = e(tr.b.i_add(u, None, base, one))?;
            let p = e(tr.b.access_chain(p_wg, None, vote, [base1]))?;
            e(tr.b.atomic_or(u, None, p, scope, sem, hi_sum))?;
            Ok(())
        })?;
        self.workgroup_barrier()?;
        let p = e(self.b.access_chain(p_wg, None, vote, [base]))?;
        let lo = e(self.b.load(u, None, p, None, []))?;
        let base1 = e(self.b.i_add(u, None, base, one))?;
        let p = e(self.b.access_chain(p_wg, None, vote, [base1]))?;
        let hi = e(self.b.load(u, None, p, None, []))?;
        let old = e(self.b.i_add(u, None, k, two))?;
        let old = e(self.b.u_mod(u, None, old, three))?;
        let old = e(self.b.i_mul(u, None, old, two))?;
        let first = e(self.b.i_equal(self.t.bool_, None, li, z))?;
        self.guarded(first, &mut |tr| {
            let p = e(tr.b.access_chain(p_wg, None, vote, [old]))?;
            ok(tr.b.store(p, z, None, []))?;
            let old1 = e(tr.b.i_add(u, None, old, one))?;
            let p = e(tr.b.access_chain(p_wg, None, vote, [old1]))?;
            ok(tr.b.store(p, z, None, []))
        })?;
        Ok((lo, hi))
    }

    /// Whether any invocation of the workgroup has `c` (lockstep mode; the
    /// slot protocol of `wave_ballot` with one dword).
    pub fn wave_any(&mut self, c: Word) -> R<Word> {
        let (Some(vote), Some(ctr), Some(li)) = (self.vote, self.vote_ctr, self.local_index) else {
            return Err("vote outside lockstep mode".into());
        };
        let u = self.t.u32;
        let k = e(self.b.load(u, None, ctr, None, []))?;
        let one = self.u(1);
        let two = self.u(2);
        let three = self.u(3);
        let kn = e(self.b.i_add(u, None, k, one))?;
        let kn = e(self.b.u_mod(u, None, kn, three))?;
        ok(self.b.store(ctr, kn, None, []))?;
        let idx = e(self.b.i_mul(u, None, k, two))?;
        let p_wg = self.b.type_pointer(None, StorageClass::Workgroup, u);
        self.guarded(c, &mut |tr| {
            let p = e(tr.b.access_chain(p_wg, None, vote, [idx]))?;
            let scope = tr.u(spirv::Scope::Workgroup as u32);
            let sem = tr.u(0);
            e(tr.b.atomic_or(u, None, p, scope, sem, one))?;
            Ok(())
        })?;
        self.workgroup_barrier()?;
        let p = e(self.b.access_chain(p_wg, None, vote, [idx]))?;
        let x = e(self.b.load(u, None, p, None, []))?;
        let old = e(self.b.i_add(u, None, k, two))?;
        let old = e(self.b.u_mod(u, None, old, three))?;
        let old = e(self.b.i_mul(u, None, old, two))?;
        let z = self.u(0);
        let first = e(self.b.i_equal(self.t.bool_, None, li, z))?;
        self.guarded(first, &mut |tr| {
            let p = e(tr.b.access_chain(p_wg, None, vote, [old]))?;
            ok(tr.b.store(p, z, None, []))
        })?;
        e(self.b.i_not_equal(self.t.bool_, None, x, z))
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

/// Lane reads (v_readlane / v_readfirstlane) of a value an atomic has just
/// returned: the wave-aggregated atomic (a ballot counts the lanes, one lane
/// adds the count, every lane takes the old value from that lane and adds
/// its v_mbcnt prefix). Each invocation is a wave of one here (its ballot
/// holds only its own bit, its prefix is 0), so it ran the atomic for itself
/// and the result to take is its own; another lane's would make all lanes
/// share one slot.
fn atomic_result_reads(code: &[Inst]) -> BTreeSet<u32> {
    let mut out = BTreeSet::new();
    for (i, ins) in code.iter().enumerate() {
        if !matches!(ins.base_name(), Some("v_readlane_b32" | "v_readfirstlane_b32")) {
            continue;
        }
        let s = ins.src(0);
        if s < crate::OP_VGPR0 {
            continue;
        }
        let v = s - crate::OP_VGPR0;
        for p in code[..i].iter().rev().take(32) {
            match vgpr_def(p, v) {
                Some(atomic) => {
                    if atomic {
                        out.insert(ins.pc);
                    }
                    break;
                }
                None => {}
            }
        }
    }
    out
}

/// Lane-0 elections of wave-aggregated atomics that count with popcounts of
/// lane masks. Pattern (the deferred lighting's tile classification appends
/// each 8x8 block of tiles this way):
///
///   v_cmp_lg_u32 vcc, 0, vL            ; vL: the lane's index in the group
///   s_not_b64 vcc, vcc                 ; (or v_cmp_eq_u32 directly)
///   s_mov_b64 exec, vcc                ; lane 0 only
///   v_bcnt_u32_b32 vN, sMaskHi, 0      ; lanes taking part
///   v_bcnt_u32_b32 vN, sMaskLo, vN
///   buffer_atomic_add vN, ... glc      ; lane 0 reserves their slots
///   v_readlane_b32 sBase, vN, 0        ; every lane takes the base ...
///   ... v_mbcnt (mask) + sBase         ; ... plus its prefix
///
/// The workgroup is a single wave, so vL is its lane index. Here every
/// invocation is a wave of one: its masks hold only its own bit and its
/// prefix is 0, so it must reserve its own slot (with its own count) and read
/// its own result (`atomic_result_reads`). The compare therefore takes lane
/// 0's result in every lane. Lane-0 work that adds a count gathered across
/// lanes (DPP sums read with v_readlane) is left alone.
/// Returns compare pc -> index of the source operand holding vL.
fn lane0_elections(code: &[Inst], spec: &Spec) -> HashMap<u32, usize> {
    use crate::{OP_EXEC_LO, OP_VCC_LO, OP_VGPR0};
    let mut out = HashMap::default();
    let wave = if spec.wave64 { 64 } else { 32 };
    let [sx, sy, sz] = spec.cs_local_size.map(|x| x.max(1));
    if spec.stage != Stage::Compute || sx * sy * sz != wave {
        return out;
    }
    // Writes of VGPR v anywhere in the program (pcs).
    let writes = |v: u16| -> Vec<usize> { code.iter().enumerate().filter(|(_, p)| vgpr_def(p, v).is_some()).map(|(i, _)| i).collect() };
    // Whether VGPR v holds the lane index (the linear local invocation index
    // of a single-wave group) at instruction `at`.
    let lane_index = |v: u16, at: usize| -> bool {
        let w = writes(v);
        if w.iter().any(|&i| i >= at) && w.len() != 1 {
            return false;
        }
        match w.as_slice() {
            [] => v == 0 && sy == 1 && sz == 1,
            [i] if *i < at => {
                let p = &code[*i];
                // v_lshl_add_u32 vL, v1, log2(sx), v0 with v0/v1 still the
                // local invocation id.
                let k = p.src(1);
                p.base_name() == Some("v_lshl_add_u32")
                    && p.src(0) == OP_VGPR0 + 1
                    && p.src(2) == OP_VGPR0
                    && (129..=192).contains(&k)
                    && 1u32 << (k - 128) == sx
                    && sz == 1
                    && writes(0).iter().chain(writes(1).iter()).all(|&j| j >= *i)
            }
            _ => false,
        }
    };
    // Compare operands and destination: (cmp, src0, src1, sdst).
    let cmp_parts = |p: &Inst| -> Option<(&'static str, u16, u16, u16)> {
        let name = p.base_name()?;
        let rest = name.strip_prefix("v_cmp_")?;
        let cmp = match rest {
            "eq_u32" | "eq_i32" => "eq",
            "lg_u32" | "lg_i32" | "ne_u32" | "ne_i32" => "ne",
            _ => return None,
        };
        match (p.enc, p.ext) {
            (crate::Enc::Vopc, crate::Ext::Sdwa(d)) => {
                let s0 = if d >> 23 & 1 != 0 { (d & 0xFF) as u16 } else { OP_VGPR0 + (d & 0xFF) as u16 };
                let s1 = if d >> 31 & 1 != 0 { (p.w0 >> 9 & 0xFF) as u16 } else { OP_VGPR0 + (p.w0 >> 9 & 0xFF) as u16 };
                Some((cmp, s0, s1, p.sdwa_sdst().unwrap_or(OP_VCC_LO)))
            }
            (crate::Enc::Vopc, _) => Some((cmp, p.src(0), p.src(1), OP_VCC_LO)),
            (crate::Enc::Vop3, _) => Some((cmp, p.src(0), p.src(1), p.vdst())),
            _ => None,
        }
    };
    // Whether `p` may write SGPR `s` (pairs count both halves).
    let writes_sgpr = |p: &Inst, s: u16| -> bool {
        let name = p.base_name().unwrap_or("");
        let pair = |d: u16| s == d || s == d + 1;
        match p.enc {
            crate::Enc::Sop1 | crate::Enc::Sop2 | crate::Enc::Sopk => {
                let d = p.sdst();
                d == s || (name.ends_with("_b64") || name.ends_with("_u64") || name.ends_with("_i64")) && d + 1 == s || name.contains("saveexec") && pair(OP_EXEC_LO)
            }
            crate::Enc::Smem => p.sdst() <= s && s < p.sdst() + 16,
            crate::Enc::Vopc => {
                if name.starts_with("v_cmpx") {
                    pair(OP_EXEC_LO)
                } else {
                    pair(p.sdwa_sdst().unwrap_or(OP_VCC_LO))
                }
            }
            crate::Enc::Vop3 if name.starts_with("v_cmpx") => pair(OP_EXEC_LO),
            crate::Enc::Vop3 if name.starts_with("v_cmp") => pair(p.vdst()),
            crate::Enc::Vop3 if p.is_vop3b() => pair(p.vop3_sdst()),
            crate::Enc::Vop3 if matches!(name, "v_readlane_b32" | "v_readfirstlane_b32") => p.vdst() == s,
            crate::Enc::Vop1 if name == "v_readfirstlane_b32" => p.vdst() == s,
            crate::Enc::Vop2 => (name.contains("_co_") || name.contains("_ci_")) && pair(OP_VCC_LO),
            _ => false,
        }
    };
    for (ia, a) in code.iter().enumerate() {
        let name = a.base_name().unwrap_or("");
        if !(name.starts_with("buffer_atomic_add") || name.starts_with("global_atomic_add")) || !a.glc() {
            continue;
        }
        let d = a.vdst();
        // The data: a popcount of SGPR lane masks.
        let Some(id) = (0..ia).rev().take(24).find(|&i| vgpr_def(&code[i], d).is_some()) else { continue };
        let def = &code[id];
        if def.base_name() != Some("v_bcnt_u32_b32") || def.src(0) >= 106 {
            continue;
        }
        // The result is read with v_readlane / v_readfirstlane.
        let read = code[ia + 1..].iter().take(8).any(|p| matches!(p.base_name(), Some("v_readlane_b32" | "v_readfirstlane_b32")) && p.src(0) == OP_VGPR0 + d);
        if !read {
            continue;
        }
        // EXEC at the atomic: s_mov_b64 exec, X.
        let Some(ie) = (0..ia).rev().take(24).find(|&i| writes_sgpr(&code[i], OP_EXEC_LO)) else { continue };
        let e = &code[ie];
        if e.base_name() != Some("s_mov_b64") || e.sdst() != OP_EXEC_LO {
            continue;
        }
        // X from a compare, possibly inverted by s_not_b64.
        let mut want = e.ssrc0();
        let mut from = ie;
        let mut negated = false;
        let mut found = None;
        for _ in 0..3 {
            let Some(ix) = (0..from).rev().take(24).find(|&i| writes_sgpr(&code[i], want)) else { break };
            let x = &code[ix];
            if x.base_name() == Some("s_not_b64") && x.sdst() == want {
                negated = !negated;
                want = x.ssrc0();
                from = ix;
                continue;
            }
            if let Some((cmp, s0, s1, sd)) = cmp_parts(x) {
                if sd == want {
                    found = Some((ix, cmp, s0, s1));
                }
            }
            break;
        }
        let Some((ix, cmp, s0, s1)) = found else { continue };
        // Lane 0 must end up active: (0 == vL) or not(0 != vL).
        if (cmp == "eq") == negated {
            continue;
        }
        let zero = 128u16;
        let lane_op = if s0 == zero && s1 >= OP_VGPR0 && lane_index(s1 - OP_VGPR0, ix) {
            1
        } else if s1 == zero && s0 >= OP_VGPR0 && lane_index(s0 - OP_VGPR0, ix) {
            0
        } else {
            continue;
        };
        out.insert(code[ix].pc, lane_op);
    }
    out
}

/// Attributes that v_interp_mov_f32 reads as P10 or P20: the title
/// interpolates packed 16-bit attributes itself (flat-shaded inputs whose
/// parameter cache holds each vertex's value, combined with the barycentric
/// VGPRs through v_fma_mix_f32), e.g. the UVs of alpha-tested foliage.
fn per_vertex_attrs(code: &[Inst]) -> BTreeSet<u32> {
    code.iter().filter(|i| i.enc == Enc::Vintrp && i.base_name() == Some("v_interp_mov_f32") && i.intrp_vsrc() & 3 != 2).map(|i| i.intrp_attr()).collect()
}

/// Whether `p` may write VGPR `v`: Some(true) for an atomic returning the
/// previous value into it, Some(false) for any other (possible) write.
fn vgpr_def(p: &Inst, v: u16) -> Option<bool> {
    let name = p.base_name().unwrap_or("");
    let d = p.vdst();
    match p.enc {
        Enc::Mubuf | Enc::Mtbuf | Enc::Mimg | Enc::Global | Enc::Flat | Enc::Scratch | Enc::Ds => {
            let atomic = name.contains("atomic") && (p.enc == Enc::Ds || p.glc()) || name.contains("_rtn");
            let returns = atomic && (p.enc != Enc::Ds || name.contains("_rtn"));
            let store = name.contains("store") || name.contains("write") || name.contains("atomic") && !returns;
            if store || !(d..d + 4).contains(&v) {
                return None;
            }
            Some(returns && d == v)
        }
        Enc::Vop1 | Enc::Vop2 | Enc::Vop3 | Enc::Vop3p | Enc::Vintrp => {
            // Compares and lane reads write SGPRs.
            if name.starts_with("v_cmp") || matches!(name, "v_readlane_b32" | "v_readfirstlane_b32") {
                return None;
            }
            (d == v || d + 1 == v).then_some(false)
        }
        _ => None,
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
    let (lockstep, one_wave) = lockstep_mode(spec, code);
    let wave_masks = lockstep && one_wave && wave_masks_wanted(spec.code_base);
    let mut tr = T {
        b,
        t,
        glsl,
        spec,
        plan,
        sgpr: 0,
        vgpr: 0,
        scc: 0,
        consts: HashMap::default(),
        interface: Vec::new(),
        sgpr_data: 0,
        load_offsets: load_offsets(plan),
        res_info: res_info_offset(plan),
        buffers: HashMap::default(),
        images: HashMap::default(),
        samplers: HashMap::default(),
        sampler_ty,
        bindings: Vec::new(),
        next_binding: 0,
        out_pos: None,
        out_params: HashMap::default(),
        compare_samplers: BTreeSet::new(),
        plain_samplers: BTreeSet::new(),
        out_mrt: HashMap::default(),
        out_depth: None,
        in_attrs: HashMap::default(),
        per_vertex_attrs: if spec.barycentric && spec.stage == Stage::Fragment { per_vertex_attrs(code) } else { BTreeSet::new() },
        per_vertex_in: HashMap::default(),
        bary: None,
        frag_coord: None,
        lds: None,
        gds: None,
        lane_id: None,
        local_index: None,
        xchg: None,
        xchg_first: None,
        uniformity: crate::uniform::analyze(code),
        lockstep: false,
        vote: None,
        vote_ctr: None,
        sync_starts: Vec::new(),
        sync_steps: None,
        lds_state: 0,
        sg_lane_var: None,
        lane_slots: HashMap::default(),
        image_tables: HashMap::default(),
        table_offsets: HashMap::default(),
        // In wave-mask mode lane reads and elections are exact.
        own_lane_reads: if wave_masks { BTreeSet::new() } else { atomic_result_reads(code) },
        elections: if wave_masks { HashMap::default() } else { lane0_elections(code, spec) },
        wave_masks,
        complete: if wave_masks { crate::wavemask::complete_masks(code, spec.wave64) } else { Default::default() },
        own_bit: None,
        warnings: Vec::new(),
        params: 0,
        mrt_mask: 0,
        pc_var: 0,
        probe_var: None,
        watchdog: None,
        loop_ctr: None,
        dump_buf: None,
        wg_id: None,
        vertex_ids: None,
        sb_struct: None,
        buffer_binding: HashMap::default(),
        buffer_views: HashMap::default(),
        view_structs: HashMap::default(),
    };
    let mut elected: Vec<u32> = tr.elections.keys().copied().collect();
    elected.sort();
    for pc in elected {
        tr.warn(format!("{pc:#x}: lane-0 election of a wave-aggregated atomic taken by every lane (each appends itself)"));
    }
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
    if code.iter().any(|i| i.branch_target().is_some_and(|t| t <= i.pc)) {
        let v = tr.sb_var();
        let bnd = tr.alloc_binding(BindingKind::Watchdog);
        tr.decorate_binding(v, bnd);
        tr.watchdog = Some(v);
    }
    if spec.dump.is_some() && matches!(spec.stage, Stage::Compute | Stage::Vertex) {
        let v = tr.sb_var();
        let bnd = tr.alloc_binding(BindingKind::Debug);
        tr.decorate_binding(v, bnd);
        tr.dump_buf = Some(v);
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
            (ResKind::Image, Some(ResSpec::Image { dim, class, depth })) => {
                let (dim, class, depth) = (*dim, *class, *depth);
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
                tr.images.insert(i, ImageRes { var: v, img_ty, dim, class, storage, depth });
            }
            (ResKind::Sampler, _) => {
                let pt = tr.b.type_pointer(None, StorageClass::UniformConstant, tr.sampler_ty);
                let v = tr.global_var(pt, StorageClass::UniformConstant);
                let bnd = tr.alloc_binding(BindingKind::Sampler(i));
                tr.decorate_binding(v, bnd);
                tr.samplers.insert(i, v);
            }
            (ResKind::ImageTable, Some(ResSpec::Image { dim, class, .. })) => {
                if r.written {
                    return Err(format!("image table {i} is written (storage tables unsupported)"));
                }
                let (dim, class) = (*dim, *class);
                let sampled_ty = match class {
                    NumClass::Float => tr.t.f32,
                    NumClass::Sint => tr.t.i32,
                    NumClass::Uint => tr.t.u32,
                };
                let (sd, arrayed) = match dim {
                    ImageDim::D1 | ImageDim::D2 | ImageDim::D2Msaa => (spirv::Dim::Dim2D, 0),
                    ImageDim::D3 => (spirv::Dim::Dim3D, 0),
                    ImageDim::Cube => (spirv::Dim::DimCube, 0),
                    ImageDim::CubeArray => (spirv::Dim::DimCube, 1),
                    ImageDim::D1Array | ImageDim::D2Array | ImageDim::D2MsaaArray => (spirv::Dim::Dim2D, 1),
                };
                tr.b.extension("SPV_EXT_descriptor_indexing");
                tr.b.capability(Capability::ShaderNonUniform);
                tr.b.capability(Capability::SampledImageArrayNonUniformIndexing);
                let img_ty = tr.b.type_image(sampled_ty, sd, 0, arrayed, 0, 1, spirv::ImageFormat::Unknown, None);
                let n = tr.u(IMAGE_TABLE_LEN);
                let arr = tr.b.type_array(img_ty, n);
                let pt = tr.b.type_pointer(None, StorageClass::UniformConstant, arr);
                let v = tr.global_var(pt, StorageClass::UniformConstant);
                let bnd = tr.alloc_binding(BindingKind::ImageArray(i));
                tr.decorate_binding(v, bnd);
                tr.images.insert(i, ImageRes { var: v, img_ty, dim, class, storage: false, depth: false });
                tr.image_tables.insert(i, r.table.ok_or("image table without its load")?);
            }
            (ResKind::Image | ResKind::ImageTable, _) => return Err(format!("image resource {i} has no specialization")),
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
    if tr.watchdog.is_some() {
        let z = tr.u(0);
        tr.loop_ctr = Some(tr.b.variable(tr.t.fn_u32, None, StorageClass::Function, Some(z)));
    }
    if tr.spec.probe.is_some() {
        let z = tr.u(0);
        tr.probe_var = Some(tr.b.variable(tr.t.fn_u32, None, StorageClass::Function, Some(z)));
    }
    let mut table_pcs: Vec<u32> = tr.image_tables.values().copied().collect();
    table_pcs.sort();
    table_pcs.dedup();
    for pc in table_pcs {
        let z = tr.u(0);
        let v = tr.b.variable(tr.t.fn_u32, None, StorageClass::Function, Some(z));
        tr.table_offsets.insert(pc, v);
    }
    // Lockstep mode (see the module documentation).
    tr.lockstep = lockstep;
    if tr.lockstep {
        let z = tr.u(0);
        tr.vote_ctr = Some(tr.b.variable(tr.t.fn_u32, None, StorageClass::Function, Some(z)));
        let len = tr.u(6);
        let arr = tr.b.type_array(tr.t.u32, len);
        let pt = tr.b.type_pointer(None, StorageClass::Workgroup, arr);
        tr.vote = Some(tr.global_var(pt, StorageClass::Workgroup));
        tr.uniformity.set_lockstep();
    }
    for ins in code {
        if let Some((true, vgpr, lane)) = ins.const_lane_op() {
            if !tr.lane_slots.contains_key(&(vgpr as u32, lane)) {
                let v = tr.b.variable(tr.t.fn_u32, None, StorageClass::Function, None);
                tr.lane_slots.insert((vgpr as u32, lane), v);
            }
        }
    }
    tr.sync_starts = sync_starts(&tr, code);
    if !tr.sync_starts.is_empty() {
        let z = tr.u(0);
        tr.sync_steps = Some(tr.b.variable(tr.t.fn_u32, None, StorageClass::Function, Some(z)));
    }
    // Initial state: all registers defined (zero), this lane active.
    let f = tr.bool_c(false);
    tr.set_scc(f)?;
    let zero = tr.u(0);
    let zs = tr.b.constant_null(arr_s);
    ok(tr.b.store(tr.sgpr, zs, None, []))?;
    let zv = tr.b.constant_null(arr_v);
    ok(tr.b.store(tr.vgpr, zv, None, []))?;
    // EXEC holds this lane's bit (bit 0) only; in wave-mask mode the lanes
    // of the wave.
    let (exec_lo, exec_hi) = if tr.wave_masks {
        let threads: u32 = spec.cs_local_size.iter().map(|&x| x.max(1)).product();
        let lo = if threads >= 32 { u32::MAX } else { (1u32 << threads) - 1 };
        let hi = if !spec.wave64 || threads <= 32 { 0 } else if threads >= 64 { u32::MAX } else { (1u32 << (threads - 32)) - 1 };
        (tr.u(lo), tr.u(hi))
    } else {
        (tr.u(1), zero)
    };
    tr.set_sreg(crate::OP_EXEC_LO as u32, exec_lo)?;
    tr.set_sreg(crate::OP_EXEC_HI as u32, exec_hi)?;
    for i in 0..spec.user_sgprs.min(SGPR_DATA_USER as u16) as u32 {
        let v = tr.sgpr_data(i)?;
        tr.set_sreg(spec.user_base as u32 + i, v)?;
    }
    mem::stage_inputs(&mut tr)?;
    if tr.wave_masks {
        let lane = tr.lane()?;
        let k32 = tr.u(32);
        let k31 = tr.u(31);
        let in_hi = e(tr.b.u_greater_than_equal(tr.t.bool_, None, lane, k32))?;
        let sh = e(tr.b.bitwise_and(tr.t.u32, None, lane, k31))?;
        tr.own_bit = Some((in_hi, sh));
    }
    if let (Some(vote), Some(li)) = (tr.vote, tr.local_index) {
        // Ballot slots start cleared.
        let z = tr.u(0);
        let first = e(tr.b.i_equal(tr.t.bool_, None, li, z))?;
        let p_wg = tr.b.type_pointer(None, StorageClass::Workgroup, tr.t.u32);
        tr.guarded(first, &mut |tr| {
            for k in 0..6 {
                let i = tr.u(k);
                let p = e(tr.b.access_chain(p_wg, None, vote, [i]))?;
                ok(tr.b.store(p, z, None, []))?;
            }
            Ok(())
        })?;
        tr.workgroup_barrier()?;
    }

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
        match loop_ranges(code, &starts) {
            Some(loops) if !spec.force_dispatcher && (loops.is_empty() || !spec.dispatcher_for_loops) => emit_structured(&mut tr, code, &starts, &loops)?,
            _ => emit_dispatch_loop(&mut tr, code, &starts)?,
        }
    }
    ok(tr.b.end_function())?;

    let model = match spec.stage {
        Stage::Vertex => ExecutionModel::Vertex,
        Stage::Fragment => ExecutionModel::Fragment,
        Stage::Compute => ExecutionModel::GLCompute,
    };
    let iface = tr.interface.clone();
    tr.b.entry_point(model, func, "main", iface);
    // Hardware rounds every float multiply and add (v_mad_f32 / v_mac_f32
    // are not fused): keep the driver from contracting them into FMAs, which
    // changes results in the last bit (texture coordinates on texel
    // boundaries select other texels).
    let ids: Vec<Word> = tr
        .b
        .module_ref()
        .functions
        .iter()
        .flat_map(|f| f.blocks.iter())
        .flat_map(|b| b.instructions.iter())
        .filter(|i| matches!(i.class.opcode, spirv::Op::FAdd | spirv::Op::FSub | spirv::Op::FMul))
        .filter_map(|i| i.result_id)
        .collect();
    for id in ids {
        tr.b.decorate(id, Decoration::NoContraction, []);
    }
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

/// Wave-mask mode for the shader at `addr` (diagnosis: `DS_WAVEMASK=0`
/// disables it, `DS_WAVEMASK=<hex address>[,...]` keeps it for those
/// shaders only).
fn wave_masks_wanted(addr: u64) -> bool {
    match std::env::var("DS_WAVEMASK") {
        Err(_) => true,
        Ok(v) => v.split(',').any(|a| u64::from_str_radix(a.trim().trim_start_matches("0x"), 16).is_ok_and(|x| x == addr && x != 0)),
    }
}

/// Lockstep mode, and whether the workgroup is a single wave (see the
/// module documentation).
fn lockstep_mode(spec: &Spec, code: &[Inst]) -> (bool, bool) {
    let threads: u32 = spec.cs_local_size.iter().map(|&x| x.max(1)).product();
    let uses_lds = code.iter().any(|i| i.enc == Enc::Ds && !i.ds_gds() && !matches!(i.name(), Some("ds_swizzle_b32" | "ds_bpermute_b32" | "ds_permute_b32")));
    let one_wave = threads <= if spec.wave64 { 64 } else { 32 };
    let wave_scalars = spec.cs_rsrc2 & (1 << 10) != 0 || code.iter().any(|i| matches!(i.base_name(), Some("v_readlane_b32" | "v_readfirstlane_b32")));
    // A single wave that reads lane masks as values (ballots it stores or
    // counts) needs wave-mask mode, which runs in lockstep, with or without
    // LDS (the title's batch culling stores one compare ballot per 32 items).
    // (`DS_MASK_VALUES=0`: not for such programs, diagnosis.)
    let mask_values = spec.stage == Stage::Compute
        && one_wave
        && !uses_lds
        && std::env::var("DS_MASK_VALUES").ok().as_deref() != Some("0")
        && crate::wavemask::reads_mask_values(code, spec.wave64);
    let lockstep = spec.stage == Stage::Compute && (uses_lds || mask_values) && (one_wave || !wave_scalars) && std::env::var("DS_LOCKSTEP").ok().as_deref() != Some("0");
    (lockstep, one_wave)
}

const PC_EXIT: u32 = 0xFFFF_FFFF;

/// Branch condition of a conditional SOPP branch (None = always taken).
/// `before` holds the block's instructions before the branch.
fn branch_cond(tr: &mut T, ins: &Inst, before: &[Inst]) -> R<Option<Word>> {
    let name = ins.name().unwrap_or("");
    let bt = tr.t.bool_;
    if tr.lockstep && !tr.wave_masks && tr.uniformity.divergent_branches.contains(&ins.pc) {
        // The wave's decision: EXEC/VCC zero when no lane has its bit; SCC
        // from lane masks is the OR of the lanes' SCC (results != 0), or the
        // AND for equality compares.
        let (lane, all) = match name {
            "s_cbranch_vccz" | "s_cbranch_vccnz" | "s_cbranch_execz" | "s_cbranch_execnz" => {
                let src = if name.contains("vcc") { crate::OP_VCC_LO } else { crate::OP_EXEC_LO } as u32;
                let lo = tr.sreg(src)?;
                let hi = tr.sreg(src + 1)?;
                let o = e(tr.b.bitwise_or(tr.t.u32, None, lo, hi))?;
                let z = tr.u(0);
                (e(tr.b.i_not_equal(bt, None, o, z))?, false)
            }
            "s_cbranch_scc0" | "s_cbranch_scc1" => {
                let all = match before.iter().rev().find(|i| crate::uniform::writes_scc(i)) {
                    Some(w) => {
                        let n = w.name().unwrap_or("");
                        n.starts_with("s_cmp_eq_") || n.starts_with("s_cmpk_eq_")
                    }
                    None => {
                        tr.warn(format!("{:#x}: lane-dependent SCC from another block decided as any-lane", ins.pc));
                        false
                    }
                };
                (tr.get_scc()?, all)
            }
            _ => return Ok(Some(tr.bool_c(false))),
        };
        let wave = if all {
            let not = e(tr.b.logical_not(bt, None, lane))?;
            let any_not = tr.wave_any(not)?;
            e(tr.b.logical_not(bt, None, any_not))?
        } else {
            tr.wave_any(lane)?
        };
        return Ok(Some(match name {
            "s_cbranch_scc1" | "s_cbranch_vccnz" | "s_cbranch_execnz" => wave,
            _ => e(tr.b.logical_not(bt, None, wave))?,
        }));
    }
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

/// Starts of the sync blocks of a compute program outside lockstep mode:
/// uniform blocks with an instruction that may place a workgroup barrier
/// (s_barrier, an LDS access, a wave64 v_readlane / v_readfirstlane).
fn sync_starts(tr: &T, code: &[Inst]) -> Vec<u32> {
    if tr.spec.stage != Stage::Compute || tr.lockstep {
        return Vec::new();
    }
    let starts: Vec<u32> = block_starts(code).into_iter().collect();
    if starts.len() <= 1 {
        return Vec::new();
    }
    let barrier = |i: &Inst| match i.base_name() {
        Some("s_barrier") => true,
        Some("v_readlane_b32" | "v_readfirstlane_b32") => tr.spec.wave64,
        _ => i.enc == Enc::Ds && !i.ds_gds() && !matches!(i.name(), Some("ds_swizzle_b32" | "ds_bpermute_b32" | "ds_permute_b32")),
    };
    let mut out = Vec::new();
    for (bi, &s) in starts.iter().enumerate() {
        let end = starts.get(bi + 1).copied().unwrap_or(u32::MAX);
        if code.iter().any(|i| i.pc >= s && i.pc < end && tr.uniformity.is_uniform(i.pc) && barrier(i)) {
            out.push(s);
        }
    }
    out
}

/// Translates block [start, end) and stores the address of the block to
/// run next (PC_EXIT after s_endpgm) in `pc_var`.
fn emit_block_body(tr: &mut T, code: &[Inst], start: u32, end: u32) -> R<()> {
    // LDS accesses of other blocks are unknown here.
    tr.lds_state = 3;
    let mut next_pc = PC_EXIT;
    let mut terminated = false;
    let mut first = None;
    for (k, ins) in code.iter().enumerate() {
        if ins.pc < start || ins.pc >= end {
            continue;
        }
        let first = *first.get_or_insert(k);
        next_pc = code.get(k + 1).map(|n| n.pc).unwrap_or(PC_EXIT);
        if ins.is_endpgm() {
            let x = tr.u(PC_EXIT);
            ok(tr.b.store(tr.pc_var, x, None, []))?;
            terminated = true;
            break;
        }
        if let Some(target) = ins.branch_target() {
            let cond = branch_cond(tr, ins, &code[first..k])?;
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
    Ok(())
}

/// Next-address value after the loop watchdog stopped an invocation (no
/// block starts there).
const PC_STOPPED: u32 = 0xFFFF_FFFE;

/// The loops of a program as address ranges [header, end) (end: the end of
/// the block of its last back edge), outermost first for equal headers; None
/// when two ranges overlap without nesting or a branch enters a range
/// elsewhere than at its header (the dispatcher handles those).
fn loop_ranges(code: &[Inst], starts: &BTreeSet<u32>) -> Option<Vec<(u32, u32)>> {
    let block_end = |pc: u32| starts.range(pc + 1..).next().copied().unwrap_or(u32::MAX);
    let mut loops: std::collections::BTreeMap<u32, u32> = std::collections::BTreeMap::new();
    for ins in code {
        if let Some(t) = ins.branch_target() {
            if t <= ins.pc {
                let e = block_end(ins.pc);
                let x = loops.entry(t).or_insert(e);
                *x = (*x).max(e);
            }
        }
    }
    let list: Vec<(u32, u32)> = loops.into_iter().collect();
    for (i, &(_, e1)) in list.iter().enumerate() {
        for &(h2, e2) in &list[i + 1..] {
            // h1 < h2: nested (e2 <= e1) or disjoint (h2 >= e1).
            if h2 < e1 && e2 > e1 {
                return None;
            }
        }
    }
    for ins in code {
        if let Some(t) = ins.branch_target() {
            for &(h, e) in &list {
                let inside = |a: u32| a >= h && a < e;
                if inside(t) && t != h && !inside(ins.pc) {
                    return None;
                }
            }
        }
    }
    Some(list)
}

/// Structured control flow (see the module documentation): the blocks of
/// [lo, hi) in address order, loops as structured loops.
fn emit_structured(tr: &mut T, code: &[Inst], starts: &BTreeSet<u32>, loops: &[(u32, u32)]) -> R<()> {
    let first = code[0].pc;
    let s0 = tr.u(first);
    ok(tr.b.store(tr.pc_var, s0, None, []))?;
    let starts_vec: Vec<u32> = starts.iter().copied().collect();
    emit_range(tr, code, &starts_vec, loops, 0, starts_vec.len(), None)?;
    emit_watchdog_record(tr)?;
    ok(tr.b.ret())?;
    Ok(())
}

/// Blocks `lo..hi` of `starts`; `inside`: the header of the loop being
/// emitted (its own range is not a nested loop).
fn emit_range(tr: &mut T, code: &[Inst], starts: &[u32], loops: &[(u32, u32)], lo: usize, hi: usize, inside: Option<u32>) -> R<()> {
    let mut i = lo;
    while i < hi {
        let s = starts[i];
        if let Some(&(h, end)) = loops.iter().find(|&&(h, _)| h == s && Some(h) != inside) {
            let j = starts.iter().position(|&x| x >= end).unwrap_or(starts.len()).min(hi);
            emit_loop(tr, code, starts, loops, i, j, h)?;
            i = j;
            continue;
        }
        let end = starts.get(i + 1).copied().unwrap_or(u32::MAX);
        let pc = e(tr.b.load(tr.t.u32, None, tr.pc_var, None, []))?;
        let k = tr.u(s);
        let here = e(tr.b.i_equal(tr.t.bool_, None, pc, k))?;
        let (body, merge) = (tr.b.id(), tr.b.id());
        ok(tr.b.selection_merge(merge, spirv::SelectionControl::NONE))?;
        ok(tr.b.branch_conditional(here, body, merge, []))?;
        e(tr.b.begin_block(Some(body)))?;
        emit_block_body(tr, code, s, end)?;
        ok(tr.b.branch(merge))?;
        e(tr.b.begin_block(Some(merge)))?;
        i += 1;
    }
    Ok(())
}

/// The loop with header `h` over blocks `lo..hi`: repeats while the next
/// address is `h` (counted by the loop watchdog).
fn emit_loop(tr: &mut T, code: &[Inst], starts: &[u32], loops: &[(u32, u32)], lo: usize, hi: usize, h: u32) -> R<()> {
    let (head, body, cont, merge) = (tr.b.id(), tr.b.id(), tr.b.id(), tr.b.id());
    ok(tr.b.branch(head))?;
    e(tr.b.begin_block(Some(head)))?;
    ok(tr.b.loop_merge(merge, cont, spirv::LoopControl::NONE, []))?;
    ok(tr.b.branch(body))?;
    e(tr.b.begin_block(Some(body)))?;
    emit_range(tr, code, starts, loops, lo, hi, Some(h))?;
    ok(tr.b.branch(cont))?;
    e(tr.b.begin_block(Some(cont)))?;
    let bt = tr.t.bool_;
    let pc = e(tr.b.load(tr.t.u32, None, tr.pc_var, None, []))?;
    let kh = tr.u(h);
    let mut again = e(tr.b.i_equal(bt, None, pc, kh))?;
    if let Some(ctr) = tr.loop_ctr {
        let n = e(tr.b.load(tr.t.u32, None, ctr, None, []))?;
        let one = tr.u(1);
        let n = e(tr.b.i_add(tr.t.u32, None, n, one))?;
        ok(tr.b.store(ctr, n, None, []))?;
        let lim = tr.u(LOOP_LIMIT);
        let over = e(tr.b.u_greater_than_equal(bt, None, n, lim))?;
        // Stopped: no block runs any more (the record follows at the end).
        let stop = e(tr.b.logical_and(bt, None, again, over))?;
        let ks = tr.u(PC_STOPPED);
        let npc = e(tr.b.select(tr.t.u32, None, stop, ks, pc))?;
        ok(tr.b.store(tr.pc_var, npc, None, []))?;
        let go = e(tr.b.logical_not(bt, None, over))?;
        again = e(tr.b.logical_and(bt, None, again, go))?;
    }
    ok(tr.b.branch_conditional(again, head, merge, []))?;
    e(tr.b.begin_block(Some(merge)))?;
    Ok(())
}

/// After the program: an invocation left with a block still to run was
/// stopped by the loop watchdog; it records that and the program address.
fn emit_watchdog_record(tr: &mut T) -> R<()> {
    let Some(wd) = tr.watchdog else { return Ok(()) };
    let ex = tr.u(PC_EXIT);
    let pc3 = e(tr.b.load(tr.t.u32, None, tr.pc_var, None, []))?;
    let stopped = e(tr.b.i_not_equal(tr.t.bool_, None, pc3, ex))?;
    let rec = tr.b.id();
    let end = tr.b.id();
    ok(tr.b.selection_merge(end, spirv::SelectionControl::NONE))?;
    ok(tr.b.branch_conditional(stopped, rec, end, []))?;
    e(tr.b.begin_block(Some(rec)))?;
    let z = tr.u(0);
    let p = e(tr.b.access_chain(tr.t.sb_u32, None, wd, [z, z]))?;
    let scope = tr.u(spirv::Scope::Device as u32);
    let one = tr.u(1);
    e(tr.b.atomic_i_add(tr.t.u32, None, p, scope, z, one))?;
    let base = tr.spec.code_base;
    for (k, v) in [(1, base as u32), (2, (base >> 32) as u32)] {
        let i = tr.u(k);
        let p = e(tr.b.access_chain(tr.t.sb_u32, None, wd, [z, i]))?;
        let c = tr.u(v);
        ok(tr.b.store(p, c, None, []))?;
    }
    ok(tr.b.branch(end))?;
    e(tr.b.begin_block(Some(end)))?;
    Ok(())
}

/// The dispatcher: a loop over a switch on the next block's address. With
/// sync blocks it runs in rounds (an outer loop): an invocation reaching a
/// sync block after other blocks of the round leaves the inner loop, whose
/// merge reconverges each subgroup, and the next round starts at that block
/// for every invocation, so its barriers are the same dynamic instances.
fn emit_dispatch_loop(tr: &mut T, code: &[Inst], starts: &BTreeSet<u32>) -> R<()> {
    let first = code[0].pc;
    let s0 = tr.u(first);
    ok(tr.b.store(tr.pc_var, s0, None, []))?;
    let rounds = match tr.sync_steps {
        Some(steps) if !tr.sync_starts.is_empty() => {
            let (oh, ob, oc, ox) = (tr.b.id(), tr.b.id(), tr.b.id(), tr.b.id());
            ok(tr.b.branch(oh))?;
            e(tr.b.begin_block(Some(oh)))?;
            ok(tr.b.loop_merge(ox, oc, spirv::LoopControl::NONE, []))?;
            ok(tr.b.branch(ob))?;
            e(tr.b.begin_block(Some(ob)))?;
            let z = tr.u(0);
            ok(tr.b.store(steps, z, None, []))?;
            Some((steps, oh, oc, ox))
        }
        _ => None,
    };
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
    if let Some((steps, ..)) = rounds {
        let bt = tr.t.bool_;
        let n = e(tr.b.load(tr.t.u32, None, steps, None, []))?;
        let mut at_sync = tr.bool_c(false);
        for s in tr.sync_starts.clone() {
            let k = tr.u(s);
            let eq = e(tr.b.i_equal(bt, None, pc, k))?;
            at_sync = e(tr.b.logical_or(bt, None, at_sync, eq))?;
        }
        let z = tr.u(0);
        let started = e(tr.b.i_not_equal(bt, None, n, z))?;
        let leave = e(tr.b.logical_and(bt, None, at_sync, started))?;
        let (brk, go) = (tr.b.id(), tr.b.id());
        ok(tr.b.selection_merge(go, spirv::SelectionControl::NONE))?;
        ok(tr.b.branch_conditional(leave, brk, go, []))?;
        e(tr.b.begin_block(Some(brk)))?;
        ok(tr.b.branch(exit))?;
        e(tr.b.begin_block(Some(go)))?;
        let one = tr.u(1);
        let n1 = e(tr.b.i_add(tr.t.u32, None, n, one))?;
        ok(tr.b.store(steps, n1, None, []))?;
    }
    let block_ids: Vec<(u32, Word)> = starts.iter().map(|&s| (s, tr.b.id())).collect();
    ok(tr.b.selection_merge(sw_merge, spirv::SelectionControl::NONE))?;
    let targets: Vec<(Operand, Word)> = block_ids.iter().map(|&(s, id)| (Operand::LiteralBit32(s), id)).collect();
    ok(tr.b.switch(pc, sw_merge, targets))?;
    let starts_vec: Vec<u32> = starts.iter().copied().collect();
    for (bi, &(start, id)) in block_ids.iter().enumerate() {
        e(tr.b.begin_block(Some(id)))?;
        let end = starts_vec.get(bi + 1).copied().unwrap_or(u32::MAX);
        emit_block_body(tr, code, start, end)?;
        ok(tr.b.branch(sw_merge))?;
    }
    e(tr.b.begin_block(Some(sw_merge)))?;
    ok(tr.b.branch(cont))?;
    e(tr.b.begin_block(Some(cont)))?;
    let pc2 = e(tr.b.load(tr.t.u32, None, tr.pc_var, None, []))?;
    let ex = tr.u(PC_EXIT);
    let mut done = e(tr.b.i_equal(tr.t.bool_, None, pc2, ex))?;
    if let Some(ctr) = tr.loop_ctr {
        let n = e(tr.b.load(tr.t.u32, None, ctr, None, []))?;
        let one = tr.u(1);
        let n = e(tr.b.i_add(tr.t.u32, None, n, one))?;
        ok(tr.b.store(ctr, n, None, []))?;
        let lim = tr.u(LOOP_LIMIT);
        let over = e(tr.b.u_greater_than_equal(tr.t.bool_, None, n, lim))?;
        done = e(tr.b.logical_or(tr.t.bool_, None, done, over))?;
    }
    ok(tr.b.branch_conditional(done, exit, header, []))?;
    e(tr.b.begin_block(Some(exit)))?;
    if let Some((_, oh, oc, ox)) = rounds {
        // Next round unless finished (or stopped by the watchdog).
        ok(tr.b.branch(oc))?;
        e(tr.b.begin_block(Some(oc)))?;
        let pc3 = e(tr.b.load(tr.t.u32, None, tr.pc_var, None, []))?;
        let mut fin = e(tr.b.i_equal(tr.t.bool_, None, pc3, ex))?;
        if let Some(ctr) = tr.loop_ctr {
            let n = e(tr.b.load(tr.t.u32, None, ctr, None, []))?;
            let lim = tr.u(LOOP_LIMIT);
            let over = e(tr.b.u_greater_than_equal(tr.t.bool_, None, n, lim))?;
            fin = e(tr.b.logical_or(tr.t.bool_, None, fin, over))?;
        }
        ok(tr.b.branch_conditional(fin, ox, oh, []))?;
        e(tr.b.begin_block(Some(ox)))?;
    }
    emit_watchdog_record(tr)?;
    ok(tr.b.ret())?;
    Ok(())
}

fn translate_inst(tr: &mut T, ins: &Inst) -> R<()> {
    translate_one(tr, ins)?;
    if let (Some((pc, reg)), Some(var)) = (tr.spec.probe, tr.probe_var) {
        if ins.pc == pc {
            let n = reg & 0xFFF;
            let v = if reg & PROBE_SGPR != 0 { tr.sreg(n)? } else { tr.vreg(n)? };
            let v = if reg & (PROBE_UINT | PROBE_SINT) != 0 {
                let (u, f) = (tr.t.u32, tr.t.f32);
                let x = if reg & PROBE_SINT != 0 {
                    let i = e(tr.b.bitcast(tr.t.i32, None, v))?;
                    e(tr.b.convert_s_to_f(f, None, i))?
                } else {
                    e(tr.b.convert_u_to_f(f, None, v))?
                };
                e(tr.b.bitcast(u, None, x))?
            } else {
                v
            };
            ok(tr.b.store(var, v, None, []))?;
        }
    }
    if let Some((pc, g)) = tr.spec.dump {
        if ins.pc == pc {
            emit_dump(tr, g)?;
        }
    }
    Ok(())
}

/// Register dump of the invocations of workgroup `g`, or of a range of
/// vertices (see `Spec::dump`).
fn emit_dump(tr: &mut T, g: [u32; 3]) -> R<()> {
    let u = tr.t.u32;
    let bt = tr.t.bool_;
    let (buf, li, cond) = match (tr.dump_buf, tr.wg_id, tr.local_index, tr.vertex_ids) {
        (Some(buf), Some(wg), Some(li), _) => {
            let mut cond = tr.bool_c(true);
            for (k, gk) in g.iter().enumerate() {
                let c = e(tr.b.composite_extract(u, None, wg, [k as u32]))?;
                let kk = tr.u(*gk);
                let eq = e(tr.b.i_equal(bt, None, c, kk))?;
                cond = e(tr.b.logical_and(bt, None, cond, eq))?;
            }
            (buf, li, cond)
        }
        (Some(buf), _, _, Some((vi, ii))) => {
            let first = tr.u(g[0]);
            let slot = e(tr.b.i_sub(u, None, vi, first))?;
            let n = tr.u(DUMP_DWORDS / DUMP_STRIDE);
            let inside = e(tr.b.u_less_than(bt, None, slot, n))?;
            let inst = tr.u(g[1]);
            let same = e(tr.b.i_equal(bt, None, ii, inst))?;
            let cond = e(tr.b.logical_and(bt, None, inside, same))?;
            (buf, slot, cond)
        }
        _ => return Ok(()),
    };
    tr.guarded(cond, &mut |tr| {
        let stride = tr.u(DUMP_STRIDE);
        let base = e(tr.b.i_mul(u, None, li, stride))?;
        let z = tr.u(0);
        let put = |tr: &mut T, k: u32, v: Word| -> R<()> {
            let kk = tr.u(k);
            let i = e(tr.b.i_add(u, None, base, kk))?;
            let p = e(tr.b.access_chain(tr.t.sb_u32, None, buf, [z, i]))?;
            ok(tr.b.store(p, v, None, []))
        };
        for k in 0..256 {
            let v = tr.vreg(k)?;
            put(tr, k, v)?;
        }
        for k in 0..128 {
            let v = tr.sreg(k)?;
            put(tr, 256 + k, v)?;
        }
        let s = tr.get_scc()?;
        let s = tr.bool_to_u(s)?;
        put(tr, 384, s)?;
        let k385 = tr.u(385);
        let i = e(tr.b.i_add(u, None, base, k385))?;
        let p = e(tr.b.access_chain(tr.t.sb_u32, None, buf, [z, i]))?;
        let n = e(tr.b.load(u, None, p, None, []))?;
        let one = tr.u(1);
        let n1 = e(tr.b.i_add(u, None, n, one))?;
        ok(tr.b.store(p, n1, None, []))
    })
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

#[cfg(test)]
mod tests {
    use super::*;

    fn cs_spec(local: [u32; 3]) -> Spec {
        Spec {
            stage: Stage::Compute,
            user_sgprs: 2,
            user_base: 0,
            code_base: 0x1000,
            wave64: true,
            ps_input_addr: 0,
            ps_inputs: Vec::new(),
            barycentric: false,
            mrt: [MrtSpec::default(); 8],
            cs_local_size: local,
            cs_rsrc2: 0,
            resources: Vec::new(),
            binding_base: 0,
            probe: None,
            dump: None,
            force_dispatcher: false,
            dispatcher_for_loops: false,
        }
    }

    /// The tile classification's append kernel (cs 0x10088ab000 of PPSA01341
    /// 01.007.000; one invocation per tile, 8x8 per wave): lane 0 of each
    /// wave reserves the slots counted with v_bcnt of the class masks.
    #[test]
    fn lane0_election_of_mask_counted_append() {
        let words = [0u32; 83];
        let code = crate::decode_program(&words).unwrap();
        let e = lane0_elections(&code, &cs_spec([8, 8, 1]));
        // v_cmp_lg_u32 vcc, 0, v0 (v0 = ly * 8 + lx): operand 1 is the lane.
        assert_eq!(e.into_iter().collect::<Vec<_>>(), vec![(0x9c, 1)]);
        // Not for groups of several waves.
        assert!(lane0_elections(&code, &cs_spec([16, 8, 1])).is_empty());
    }

    /// v_interp_mov_f32 of P10 / P20 (cs 0x100cc3d100's foliage alpha test:
    /// P10, P0, P20 of attribute 0): that attribute is read per vertex; a
    /// lone P0 read (a flat value) is not.
    #[test]
    fn per_vertex_attributes() {
        let code = crate::decode_program(&[0xc80e0000, 0xc80a0002, 0xc8120001, 0xbf810000]).unwrap();
        assert_eq!(per_vertex_attrs(&code).into_iter().collect::<Vec<_>>(), vec![0]);
        let code = crate::decode_program(&[0xc80a0002, 0xbf810000]).unwrap();
        assert!(per_vertex_attrs(&code).is_empty());
    }

    /// Wave-mask mode (one wave, LDS): the compare whose mask is counted
    /// gets a ballot (two subgroup ORs), the one only selected from does
    /// not; the module parses.
    #[test]
    fn wave_mask_ballots_only_whole_reads() {
        use crate::tables::*;
        let op = |t: Table, n: &str| t.iter().find(|(_, x)| x.split('/').next() == Some(n)).unwrap().0 as u32;
        let vopc = |o: u32, s0: u32, v1: u32| (0x3E << 25) | (o << 17) | (v1 << 9) | s0;
        let words = [
            vopc(op(VOPC, "v_cmp_gt_u32"), 0xA0, 0),                              // vcc = 32 > v0
            (0x17D << 23) | (4 << 16) | (op(SOP1, "s_bcnt1_i32_b64") << 8) | 106, // s4 = popcount(vcc)
            vopc(op(VOPC, "v_cmp_lt_f32"), 256, 1),                               // vcc = v0 < v1
            (op(VOP2, "v_cndmask_b32") << 25) | (2 << 17) | (1 << 9) | 256,       // v2 = vcc ? v1 : v0
            (0x3F << 25) | (3 << 17) | (op(VOP1, "v_mov_b32") << 9) | 4,          // v3 = s4
            (0x36 << 26) | (op(DS, "ds_write_b32") << 18),                        // ds_write_b32 v0, v3
            3 << 8,
            0xBF81_0000, // s_endpgm
        ];
        let code = crate::decode_program(&words).unwrap();
        assert_eq!(crate::wavemask::complete_masks(&code, true).into_iter().collect::<Vec<_>>(), vec![code[0].pc]);
        let plan = crate::analysis::analyze(&code, 2, 0, 0);
        let mut spec = cs_spec([64, 1, 1]);
        spec.cs_rsrc2 = 1 << 15;
        let out = translate(&code, &plan, &spec).unwrap();
        let mut loader = rspirv::dr::Loader::new();
        rspirv::binary::parse_words(&out.spirv, &mut loader).unwrap();
        let ors = loader.module().functions.iter().flat_map(|f| f.blocks.iter()).flat_map(|b| b.instructions.iter()).filter(|i| i.class.opcode == spirv::Op::GroupNonUniformBitwiseOr).count();
        assert_eq!(ors, 2);
    }

    /// The counting kernel (cs 0x10088aa500) sums the class masks with DPP
    /// and v_readlane; its lane 0 adds the wave's total: no election.
    #[test]
    fn lane0_election_not_for_wave_sums() {
        let words = [0u32; 77];
        let code = crate::decode_program(&words).unwrap();
        assert!(lane0_elections(&code, &cs_spec([8, 8, 1])).is_empty());
    }
}
