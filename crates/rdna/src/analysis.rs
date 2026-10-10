//! Scalar provenance analysis (ported from the earlier native port's GCN
//! analysis to gfx10).
//!
//! Tracks where each SGPR dword comes from (user data, a scalar memory load
//! through a pointer derived from user data, or a constant) so that resource
//! descriptors and scalar-loaded data can be resolved on the CPU at draw time.
//! Flow-sensitive: states are propagated along the basic-block graph and
//! merged at joins (values that differ between paths become unknown), so a
//! register clobbered on one branch does not affect the others.

use crate::{Enc, Inst};
use std::collections::{BTreeSet, VecDeque};
use crate::HashMap;

/// Source of a 64-bit pointer used by scalar loads.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Ptr {
    /// User-data SGPR pair (lo index).
    User(u16),
    /// Pointer loaded from memory: dword pair at `offset` from another pointer.
    Mem(Box<Ptr>, u32),
    /// Absolute address (e.g. from s_getpc_b64).
    Abs(u64),
}

/// Provenance of one SGPR dword.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Prov {
    Unknown,
    User(u16),
    Const(u32),
    /// dword at byte `offset` from pointer `ptr` (plus a constant pointer bias).
    Mem(Ptr, u32),
    /// Low/high half of `ptr + bias` (pointer arithmetic).
    PtrLo(Ptr, u32),
    PtrHi(Ptr, u32),
    /// Dword `.1` of what the s_buffer_load at pc `.0` read: an entry of a
    /// table read at a run-time offset (e.g. a T# picked per light probe).
    Table(u32, u8),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ResKind {
    /// V# used by buffer instructions (MUBUF/MTBUF/s_buffer_load).
    Buffer,
    /// T# used by image instructions.
    Image,
    /// S# used by sampling instructions.
    Sampler,
    /// 64-bit base address (SGPR pair `saddr`) of GLOBAL instructions.
    Global,
    /// T#s read from a buffer at a run-time offset (a descriptor table):
    /// `dwords` is the table's V#, `Resource::table` the pc of the
    /// s_buffer_load_dwordx8 that reads the T#.
    ImageTable,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Resource {
    pub kind: ResKind,
    /// Provenance of each descriptor dword (4 for V#/S#, 8 for T#, 2 for a
    /// global base address).
    pub dwords: Vec<Prov>,
    /// Buffer is written by the shader / image is a storage image.
    pub written: bool,
    /// Buffer is read by the shader (loads, atomics, scalar buffer loads).
    pub read: bool,
    /// Buffer is read with format conversion (buffer_load_format/tbuffer).
    pub formatted: bool,
    /// Buffer is accessed with atomics.
    pub atomic: bool,
    /// Image tables: pc of the load that reads the T# from the table.
    pub table: Option<u32>,
}

/// A scalar memory load whose dwords are resolved on the CPU at draw time.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ScalarLoad {
    pub ptr: Ptr,
    pub offset: u32,
    pub dwords: u32,
}

#[derive(Default, Clone, Debug)]
pub struct Plan {
    pub resources: Vec<Resource>,
    /// Instruction pc -> resource indices (descriptor, sampler) it uses.
    pub inst_res: HashMap<u32, (usize, Option<usize>)>,
    pub loads: Vec<ScalarLoad>,
    /// Instruction pc -> index into `loads` for s_load_dword* instructions.
    pub inst_load: HashMap<u32, usize>,
    /// Problems found (unresolvable descriptors etc.).
    pub errors: Vec<String>,
    /// s_buffer_load pc -> provenance of the V# it reads through.
    pub table_loads: HashMap<u32, Vec<Prov>>,
}

impl Plan {
    fn resource(&mut self, r: Resource) -> usize {
        if let Some(i) = self.resources.iter().position(|x| x.kind == r.kind && x.dwords == r.dwords && x.table == r.table) {
            let x = &mut self.resources[i];
            x.written |= r.written;
            x.read |= r.read;
            x.formatted |= r.formatted;
            x.atomic |= r.atomic;
            return i;
        }
        self.resources.push(r);
        self.resources.len() - 1
    }
}

#[derive(Clone)]
struct State {
    s: Vec<Prov>,
    /// SGPR values spilled into VGPR lanes (v_writelane with constant lanes).
    lanes: Vec<((u16, u32), Prov)>,
}

impl State {
    /// Merges another path's state into this one; returns whether it changed.
    fn merge(&mut self, o: &State) -> bool {
        let mut changed = false;
        for (a, b) in self.s.iter_mut().zip(o.s.iter()) {
            if a != b && *a != Prov::Unknown {
                *a = Prov::Unknown;
                changed = true;
            }
        }
        let before = self.lanes.len();
        self.lanes.retain(|(k, v)| o.lanes.iter().any(|(k2, v2)| k2 == k && v2 == v));
        changed || self.lanes.len() != before
    }
    fn get(&self, r: u16) -> Prov {
        match r {
            0..=127 => self.s[r as usize].clone(),
            128 => Prov::Const(0),
            129..=192 => Prov::Const((r - 128) as u32),
            193..=208 => Prov::Const((-((r - 192) as i32)) as u32),
            _ => Prov::Unknown,
        }
    }
    fn set(&mut self, r: u16, p: Prov) {
        if (r as usize) < self.s.len() && r != crate::OP_NULL {
            self.s[r as usize] = p;
        }
    }
    /// Interprets SGPR pair `r` as a pointer.
    fn ptr(&self, r: u16) -> Option<(Ptr, u32)> {
        match (self.get(r), self.get(r + 1)) {
            (Prov::User(a), Prov::User(b)) if b == a + 1 => Some((Ptr::User(a), 0)),
            (Prov::Mem(p, o), Prov::Mem(q, o2)) if p == q && o2 == o + 4 => Some((Ptr::Mem(Box::new(p), o), 0)),
            (Prov::PtrLo(p, b), Prov::PtrHi(q, b2)) if p == q && b == b2 => Some((p, b)),
            _ => None,
        }
    }
}

fn src_operand(st: &State, ins: &Inst, op: u16) -> Prov {
    if op == crate::OP_LITERAL {
        return ins.lit.map(Prov::Const).unwrap_or(Prov::Unknown);
    }
    st.get(op)
}

/// SALU ops whose destination is a 64-bit register pair.
fn writes_pair(name: &str) -> bool {
    name.ends_with("_b64") || name.ends_with("_u64") || name.ends_with("_i64") || name == "s_getpc_b64" || name == "s_swappc_b64"
}

/// Runs the analysis. User-data dword `i` (of `user_sgprs`) is loaded into
/// SGPR `user_base + i` (8 for merged/NGG shaders, 0 otherwise).
pub fn analyze(code: &[Inst], user_sgprs: u16, user_base: u16, code_base: u64) -> Plan {
    let mut init = State { s: vec![Prov::Unknown; 128], lanes: Vec::new() };
    for i in 0..user_sgprs.min(32) {
        if let Some(s) = init.s.get_mut((user_base + i) as usize) {
            *s = Prov::User(i);
        }
    }
    let n = code.len();
    if n == 0 {
        return Plan::default();
    }
    // Basic blocks.
    let index: HashMap<u32, usize> = code.iter().enumerate().map(|(i, ins)| (ins.pc, i)).collect();
    let mut starts = BTreeSet::new();
    starts.insert(0usize);
    for (i, ins) in code.iter().enumerate() {
        if let Some(t) = ins.branch_target() {
            if let Some(&j) = index.get(&t) {
                starts.insert(j);
            }
            if i + 1 < n {
                starts.insert(i + 1);
            }
        }
        if ins.is_endpgm() && i + 1 < n {
            starts.insert(i + 1);
        }
    }
    let starts: Vec<usize> = starts.into_iter().collect();
    let block_at: HashMap<usize, usize> = starts.iter().enumerate().map(|(b, &s)| (s, b)).collect();
    let end = |b: usize| starts.get(b + 1).copied().unwrap_or(n);
    let succ: Vec<Vec<usize>> = (0..starts.len())
        .map(|b| {
            let e = end(b);
            let last = &code[e - 1];
            let mut v = Vec::new();
            if last.is_endpgm() {
                return v;
            }
            match last.branch_target() {
                Some(t) => {
                    if let Some(j) = index.get(&t) {
                        v.push(block_at[j]);
                    }
                    if last.name() != Some("s_branch") && e < n {
                        v.push(block_at[&e]);
                    }
                }
                None if e < n => v.push(b + 1),
                None => {}
            }
            v
        })
        .collect();
    // Fixpoint over block entry states (records are discarded here).
    let mut entry: Vec<Option<State>> = vec![None; starts.len()];
    entry[0] = Some(init);
    let mut work: VecDeque<usize> = VecDeque::from([0]);
    let mut queued = vec![false; starts.len()];
    queued[0] = true;
    let mut scratch = Plan::default();
    while let Some(b) = work.pop_front() {
        queued[b] = false;
        let mut st = entry[b].clone().unwrap();
        for ins in &code[starts[b]..end(b)] {
            step(&mut st, ins, code_base, &mut scratch);
        }
        scratch = Plan::default();
        for &sb in &succ[b] {
            let changed = match &mut entry[sb] {
                None => {
                    entry[sb] = Some(st.clone());
                    true
                }
                Some(old) => old.merge(&st),
            };
            if changed && !queued[sb] {
                queued[sb] = true;
                work.push_back(sb);
            }
        }
    }
    // Final pass over reachable blocks with their converged entry states.
    let mut plan = Plan::default();
    for b in 0..starts.len() {
        let Some(mut st) = entry[b].clone() else { continue };
        for ins in &code[starts[b]..end(b)] {
            step(&mut st, ins, code_base, &mut plan);
        }
    }
    for (i, r) in plan.resources.iter().enumerate() {
        if r.dwords.iter().any(|p| matches!(p, Prov::Unknown | Prov::Table(..))) {
            plan.errors.push(format!("resource {i} ({:?}) has unresolvable descriptor dwords {:?}", r.kind, r.dwords));
        }
    }
    plan
}

/// Transfer function of one instruction (records resources/loads in `plan`).
fn step(st: &mut State, ins: &Inst, code_base: u64, plan: &mut Plan) {
    // SGPR spills through VGPR lanes.
    if let Some((write, vgpr, lane)) = ins.const_lane_op() {
        if write {
            let v = src_operand(st, ins, ins.src(0));
            st.lanes.retain(|(k, _)| *k != (vgpr, lane));
            st.lanes.push(((vgpr, lane), v));
        } else {
            let v = st.lanes.iter().find(|(k, _)| *k == (vgpr, lane)).map(|(_, p)| p.clone()).unwrap_or(Prov::Unknown);
            st.set(ins.vdst(), v);
        }
        return;
    }
    let name = ins.base_name().unwrap_or("");
    match ins.enc {
        Enc::Sop1 => {
            let d = ins.sdst();
            let s = ins.ssrc0();
            match name {
                "s_getpc_b64" => {
                    let pc = code_base + ins.pc as u64 + 4;
                    st.set(d, Prov::PtrLo(Ptr::Abs(pc), 0));
                    st.set(d + 1, Prov::PtrHi(Ptr::Abs(pc), 0));
                }
                "s_mov_b32" => st.set(d, src_operand(st, ins, s)),
                "s_not_b32" => {
                    let v = match src_operand(st, ins, s) {
                        Prov::Const(x) => Prov::Const(!x),
                        _ => Prov::Unknown,
                    };
                    st.set(d, v);
                }
                "s_brev_b32" => {
                    let v = match src_operand(st, ins, s) {
                        Prov::Const(x) => Prov::Const(x.reverse_bits()),
                        _ => Prov::Unknown,
                    };
                    st.set(d, v);
                }
                "s_mov_b64" => {
                    st.set(d, src_operand(st, ins, s));
                    let hi = if s < 128 { st.get(s + 1) } else { Prov::Const(0) };
                    st.set(d + 1, hi);
                }
                _ => {
                    st.set(d, Prov::Unknown);
                    if writes_pair(name) {
                        st.set(d + 1, Prov::Unknown);
                    }
                    // saveexec forms also write EXEC (no provenance kept for it).
                }
            }
        }
        Enc::Sop2 => {
            let d = ins.sdst();
            let a = src_operand(st, ins, ins.ssrc0());
            let b = src_operand(st, ins, ins.ssrc1());
            let result = match (name, &a, &b) {
                // s_add_u32 lo, ptr_lo, const (pointer arithmetic, paired with s_addc_u32)
                ("s_add_u32", _, Prov::Const(c)) => match st.ptr(ins.ssrc0()) {
                    Some((p, bias)) => Prov::PtrLo(p, bias.wrapping_add(*c)),
                    None => match a {
                        Prov::Const(x) => Prov::Const(x.wrapping_add(*c)),
                        _ => Prov::Unknown,
                    },
                },
                // The same with the constant first: s_getpc_b64 s[4:5];
                // s_add_u32 s4, lit, s4; s_addc_u32 s5, 0, s5 addresses a
                // table in the shader's code (descriptors built in the shader).
                ("s_add_u32", Prov::Const(c), _) => match st.ptr(ins.ssrc1()) {
                    Some((p, bias)) => Prov::PtrLo(p, bias.wrapping_add(*c)),
                    None => Prov::Unknown,
                },
                ("s_addc_u32", _, Prov::Const(0)) | ("s_addc_u32", Prov::Const(0), _) => {
                    // s_addc_u32 hi, ptr_hi, 0 following the s_add_u32 above
                    match st.get(d.saturating_sub(1)) {
                        Prov::PtrLo(p, bias) => Prov::PtrHi(p, bias),
                        _ => Prov::Unknown,
                    }
                }
                ("s_add_i32", Prov::Const(x), Prov::Const(y)) => Prov::Const(x.wrapping_add(*y)),
                ("s_sub_u32" | "s_sub_i32", Prov::Const(x), Prov::Const(y)) => Prov::Const(x.wrapping_sub(*y)),
                ("s_and_b32", Prov::Const(x), Prov::Const(y)) => Prov::Const(x & y),
                ("s_or_b32", Prov::Const(x), Prov::Const(y)) => Prov::Const(x | y),
                ("s_xor_b32", Prov::Const(x), Prov::Const(y)) => Prov::Const(x ^ y),
                ("s_andn2_b32", Prov::Const(x), Prov::Const(y)) => Prov::Const(x & !y),
                ("s_orn2_b32", Prov::Const(x), Prov::Const(y)) => Prov::Const(x | !y),
                ("s_mul_i32", Prov::Const(x), Prov::Const(y)) => Prov::Const(x.wrapping_mul(*y)),
                ("s_lshl_b32", Prov::Const(x), Prov::Const(y)) => Prov::Const(x << (y & 31)),
                ("s_lshr_b32", Prov::Const(x), Prov::Const(y)) => Prov::Const(x >> (y & 31)),
                ("s_ashr_i32", Prov::Const(x), Prov::Const(y)) => Prov::Const(((*x as i32) >> (y & 31)) as u32),
                // Bitfield mask: ((1 << size) - 1) << offset.
                ("s_bfm_b32", Prov::Const(x), Prov::Const(y)) => Prov::Const((((1u64 << (x & 31)) - 1) as u32) << (y & 31)),
                ("s_bfe_u32", Prov::Const(x), Prov::Const(y)) => {
                    let (off, w) = (y & 31, (y >> 16) & 0x7F);
                    Prov::Const(if w == 0 { 0 } else { (((*x as u64) >> off) & ((1u64 << w.min(32)) - 1)) as u32 })
                }
                ("s_lshl1_add_u32", Prov::Const(x), Prov::Const(y)) => Prov::Const((x << 1).wrapping_add(*y)),
                ("s_lshl2_add_u32", Prov::Const(x), Prov::Const(y)) => Prov::Const((x << 2).wrapping_add(*y)),
                ("s_lshl3_add_u32", Prov::Const(x), Prov::Const(y)) => Prov::Const((x << 3).wrapping_add(*y)),
                ("s_lshl4_add_u32", Prov::Const(x), Prov::Const(y)) => Prov::Const((x << 4).wrapping_add(*y)),
                ("s_pack_ll_b32_b16", Prov::Const(x), Prov::Const(y)) => Prov::Const((x & 0xFFFF) | (y << 16)),
                // s_and_b32 hi, ptr_hi, 0xffff (address masking keeps the pointer)
                ("s_and_b32", Prov::PtrHi(p, bias), Prov::Const(0xFFFF)) => Prov::PtrHi(p.clone(), *bias),
                _ => Prov::Unknown,
            };
            st.set(d, result);
            if writes_pair(name) {
                st.set(d + 1, Prov::Unknown);
            }
        }
        Enc::Sopk => {
            let d = ins.sdst();
            match name {
                "s_movk_i32" => st.set(d, Prov::Const(ins.simm16() as i16 as i32 as u32)),
                n if n.starts_with("s_cmpk_") => {}
                "s_setreg_b32" | "s_setreg_imm32_b32" => {}
                _ => st.set(d, Prov::Unknown),
            }
        }
        Enc::Smem => {
            let d = ins.sdst();
            let n: u32 = if name.ends_with("dwordx16") {
                16
            } else if name.ends_with("dwordx8") {
                8
            } else if name.ends_with("dwordx4") {
                4
            } else if name.ends_with("dwordx2") {
                2
            } else if name.ends_with("dword") {
                1
            } else {
                return; // cache control, memtime, ...
            };
            if name.starts_with("s_store") || name.starts_with("s_buffer_store") {
                return;
            }
            // gfx10: address = base + imm offset + soffset (unless null).
            let soff = ins.smem_soffset();
            let off = if soff == crate::OP_NULL {
                Some(ins.smem_offset() as u32)
            } else {
                match st.get(soff) {
                    Prov::Const(c) => Some((ins.smem_offset() as u32).wrapping_add(c)),
                    _ => None,
                }
            };
            if name.starts_with("s_load") {
                match (st.ptr(ins.smem_sbase()), off) {
                    (Some((p, bias)), Some(o)) => {
                        let o = o.wrapping_add(bias);
                        let idx = plan.loads.iter().position(|l| l.ptr == p && l.offset == o && l.dwords == n).unwrap_or_else(|| {
                            plan.loads.push(ScalarLoad { ptr: p.clone(), offset: o, dwords: n });
                            plan.loads.len() - 1
                        });
                        plan.inst_load.insert(ins.pc, idx);
                        for k in 0..n {
                            st.set(d + k as u16, Prov::Mem(p.clone(), o + 4 * k));
                        }
                    }
                    (Some((p, bias)), None) => {
                        // Known base, dynamic offset (e.g. a loop over a constant
                        // table): the base is bound as a buffer and read at run time.
                        let dwords = vec![Prov::PtrLo(p.clone(), bias), Prov::PtrHi(p, bias)];
                        let r = plan.resource(Resource { kind: ResKind::Global, dwords, written: false, read: true, formatted: false, atomic: false, table: None });
                        plan.inst_res.insert(ins.pc, (r, None));
                        for k in 0..n {
                            st.set(d + k as u16, Prov::Unknown);
                        }
                    }
                    _ => {
                        plan.errors.push(format!("{:#x}: s_load with unresolvable address", ins.pc));
                        for k in 0..n {
                            st.set(d + k as u16, Prov::Unknown);
                        }
                    }
                }
            } else if name.starts_with("s_buffer_load") {
                // V# in s[sbase:sbase+3]. What it reads is known only at run
                // time: a T# loaded this way is a table entry (ImageTable).
                let dwords: Vec<Prov> = (0..4).map(|k| st.get(ins.smem_sbase() + k)).collect();
                plan.table_loads.insert(ins.pc, dwords.clone());
                let r = plan.resource(Resource { kind: ResKind::Buffer, dwords, written: false, read: true, formatted: false, atomic: false, table: None });
                plan.inst_res.insert(ins.pc, (r, None));
                for k in 0..n {
                    st.set(d + k as u16, Prov::Table(ins.pc, k as u8));
                }
            } else {
                for k in 0..n {
                    st.set(d + k as u16, Prov::Unknown);
                }
            }
        }
        Enc::Mubuf | Enc::Mtbuf => {
            let base = ins.mem_srsrc();
            let dwords = (0..4).map(|k| st.get(base + k)).collect();
            let store = name.contains("_store") || name.contains("_atomic");
            let atomic = name.contains("_atomic");
            let formatted = name.contains("_format") || ins.enc == Enc::Mtbuf;
            let r = plan.resource(Resource { kind: ResKind::Buffer, dwords, written: store, read: !store || atomic, formatted, atomic, table: None });
            plan.inst_res.insert(ins.pc, (r, None));
        }
        Enc::Mimg => {
            let base = ins.mem_srsrc();
            let dwords: Vec<Prov> = (0..8).map(|k| st.get(base + k)).collect();
            let store = name.starts_with("image_store") || name.starts_with("image_atomic");
            // A whole T# from one table load: an image table.
            let table = match dwords.first() {
                Some(&Prov::Table(pc, 0)) if dwords.iter().enumerate().all(|(k, p)| *p == Prov::Table(pc, k as u8)) => plan.table_loads.get(&pc).cloned().map(|v| (pc, v)),
                _ => None,
            };
            let img = match table {
                Some((pc, v)) => plan.resource(Resource { kind: ResKind::ImageTable, dwords: v, written: store, read: !store, formatted: false, atomic: false, table: Some(pc) }),
                None => plan.resource(Resource { kind: ResKind::Image, dwords, written: store, read: !store, formatted: false, atomic: false, table: None }),
            };
            let samp = if name.starts_with("image_sample") || name.starts_with("image_gather4") || name == "image_get_lod" {
                let sb = ins.mimg_ssamp();
                let sd = (0..4).map(|k| st.get(sb + k)).collect();
                Some(plan.resource(Resource { kind: ResKind::Sampler, dwords: sd, written: false, read: false, formatted: false, atomic: false, table: None }))
            } else {
                None
            };
            plan.inst_res.insert(ins.pc, (img, samp));
        }
        Enc::Vop1 if name == "v_readfirstlane_b32" => st.set(ins.vdst(), Prov::Unknown),
        Enc::Vop3 => {
            if crate::vop3_kind(ins.op) == crate::Vop3Kind::FromVopc {
                // Compare result: SGPR pair in the VDST field (gfx10 v_cmpx:
                // EXEC only).
                let d = if name.starts_with("v_cmpx_") { crate::OP_EXEC_LO } else { ins.vdst() };
                st.set(d, Prov::Unknown);
                st.set(d + 1, Prov::Unknown);
            } else if ins.is_vop3b() {
                let d = ins.vop3_sdst();
                st.set(d, Prov::Unknown);
                st.set(d + 1, Prov::Unknown);
            } else if matches!(name, "v_readfirstlane_b32" | "v_readlane_b32") {
                st.set(ins.vdst(), Prov::Unknown);
            }
        }
        Enc::Vopc => {
            // gfx10 v_cmpx writes EXEC only (VCC keeps e.g. a descriptor
            // table pointer across it).
            let d = if name.starts_with("v_cmpx_") { crate::OP_EXEC_LO } else { ins.sdwa_sdst().unwrap_or(crate::OP_VCC_LO) };
            st.set(d, Prov::Unknown);
            st.set(d + 1, Prov::Unknown);
        }
        Enc::Global if ins.flat_saddr() != crate::OP_NULL => {
            let sa = ins.flat_saddr();
            let dwords = vec![st.get(sa), st.get(sa + 1)];
            let store = name.starts_with("global_store") || name.starts_with("global_atomic");
            let atomic = name.starts_with("global_atomic");
            let r = plan.resource(Resource { kind: ResKind::Global, dwords, written: store, read: !store || atomic, formatted: false, atomic, table: None });
            plan.inst_res.insert(ins.pc, (r, None));
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A V# built in the shader: s_getpc_b64 plus a constant given as the
    /// first operand of s_add_u32 addresses a table in the code (cs
    /// 0x1008898d00 of PPSA01341 01.007.000).
    #[test]
    fn getpc_descriptor_with_constant_first() {
        let words = [0u32; 13];
        let code = crate::decode_program(&words).unwrap();
        let plan = analyze(&code, 0, 0, 0x1000);
        assert!(plan.errors.is_empty(), "{:?}", plan.errors);
        let r = &plan.resources[0];
        assert_eq!(r.kind, ResKind::Buffer);
        // s_getpc_b64 is at pc 0xc: the next instruction's address.
        let pc = Ptr::Abs(0x1000 + 0xc + 4);
        assert_eq!(r.dwords, vec![Prov::PtrLo(pc.clone(), 0x474), Prov::PtrHi(pc, 0x474), Prov::Const(16), Prov::Const(0x1000_5004)]);
    }
}
