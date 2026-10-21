//! Compute kernels the runtime runs on its own behalf (not translated from a
//! title's shaders), built here because this crate owns the SPIR-V builder.

use rspirv::binary::Assemble;
use rspirv::dr::{Builder, Instruction, Operand};
use rspirv::spirv::{self, BuiltIn, Capability, Decoration, ExecutionMode, ExecutionModel, FunctionControl, Op, SelectionControl, StorageClass, Word};

/// Workgroup size of `shadow_merge`.
pub const MERGE_LOCAL_SIZE: u32 = 64;

fn global(b: &mut Builder, ptr_ty: Word, sc: StorageClass) -> Word {
    let id = b.id();
    b.module_mut().types_global_values.push(Instruction::new(Op::Variable, Some(ptr_ty), Some(id), vec![Operand::StorageClass(sc)]));
    id
}

/// Write-back of a dispatch's device-local buffer copy (a "shadow"): every
/// dword of binding 0 (the copy after the dispatch) that differs from
/// binding 1 (the copy taken before it) is stored to binding 2 (guest
/// memory), for dword indices `first..end` (push constants `{first, end}`).
/// Dwords the dispatch did not change are left alone: the CPU may have
/// written them meanwhile, and a whole-range copy would revert that.
pub fn shadow_merge() -> Vec<u32> {
    build_merge().expect("shadow merge kernel")
}

fn build_merge() -> Result<Vec<u32>, rspirv::dr::Error> {
    let lit = Operand::LiteralBit32;
    let mut b = Builder::new();
    b.set_version(1, 3);
    b.capability(Capability::Shader);
    b.extension("SPV_KHR_storage_buffer_storage_class");
    b.memory_model(spirv::AddressingModel::Logical, spirv::MemoryModel::GLSL450);
    let void = b.type_void();
    let bool_ = b.type_bool();
    let u32t = b.type_int(32, 0);
    let v3 = b.type_vector(u32t, 3);
    let rta = b.type_runtime_array(u32t);
    b.decorate(rta, Decoration::ArrayStride, [lit(4)]);
    let st = b.type_struct([rta]);
    b.decorate(st, Decoration::Block, []);
    b.member_decorate(st, 0, Decoration::Offset, [lit(0)]);
    let p_st = b.type_pointer(None, StorageClass::StorageBuffer, st);
    let p_u = b.type_pointer(None, StorageClass::StorageBuffer, u32t);
    let pc = b.type_struct([u32t, u32t]);
    b.decorate(pc, Decoration::Block, []);
    b.member_decorate(pc, 0, Decoration::Offset, [lit(0)]);
    b.member_decorate(pc, 1, Decoration::Offset, [lit(4)]);
    let p_pc = b.type_pointer(None, StorageClass::PushConstant, pc);
    let p_pcu = b.type_pointer(None, StorageClass::PushConstant, u32t);
    let p_in = b.type_pointer(None, StorageClass::Input, v3);
    let gid = global(&mut b, p_in, StorageClass::Input);
    b.decorate(gid, Decoration::BuiltIn, [Operand::BuiltIn(BuiltIn::GlobalInvocationId)]);
    let mut bufs = [0; 3];
    for (i, v) in bufs.iter_mut().enumerate() {
        *v = global(&mut b, p_st, StorageClass::StorageBuffer);
        b.decorate(*v, Decoration::DescriptorSet, [lit(0)]);
        b.decorate(*v, Decoration::Binding, [lit(i as u32)]);
    }
    b.decorate(bufs[0], Decoration::NonWritable, []);
    b.decorate(bufs[1], Decoration::NonWritable, []);
    let pcv = global(&mut b, p_pc, StorageClass::PushConstant);
    let c0 = b.constant_bit32(u32t, 0);
    let c1 = b.constant_bit32(u32t, 1);
    let fty = b.type_function(void, []);
    let f = b.begin_function(void, None, FunctionControl::NONE, fty)?;
    b.begin_block(None)?;
    let g = b.load(v3, None, gid, None, [])?;
    let x = b.composite_extract(u32t, None, g, [0])?;
    let pf = b.access_chain(p_pcu, None, pcv, [c0])?;
    let first = b.load(u32t, None, pf, None, [])?;
    let pe = b.access_chain(p_pcu, None, pcv, [c1])?;
    let end = b.load(u32t, None, pe, None, [])?;
    let i = b.i_add(u32t, None, x, first)?;
    let inside = b.u_less_than(bool_, None, i, end)?;
    let (l_body, l_store, l_join, l_end) = (b.id(), b.id(), b.id(), b.id());
    b.selection_merge(l_end, SelectionControl::NONE)?;
    b.branch_conditional(inside, l_body, l_end, [])?;
    b.begin_block(Some(l_body))?;
    let pw = b.access_chain(p_u, None, bufs[0], [c0, i])?;
    let w = b.load(u32t, None, pw, None, [])?;
    let pp = b.access_chain(p_u, None, bufs[1], [c0, i])?;
    let old = b.load(u32t, None, pp, None, [])?;
    let changed = b.i_not_equal(bool_, None, w, old)?;
    b.selection_merge(l_join, SelectionControl::NONE)?;
    b.branch_conditional(changed, l_store, l_join, [])?;
    b.begin_block(Some(l_store))?;
    let ph = b.access_chain(p_u, None, bufs[2], [c0, i])?;
    b.store(ph, w, None, [])?;
    b.branch(l_join)?;
    b.begin_block(Some(l_join))?;
    b.branch(l_end)?;
    b.begin_block(Some(l_end))?;
    b.ret()?;
    b.end_function()?;
    b.entry_point(ExecutionModel::GLCompute, f, "main", [gid]);
    b.execution_mode(f, ExecutionMode::LocalSize, [MERGE_LOCAL_SIZE, 1, 1]);
    Ok(b.module().assemble())
}

/// Workgroup size of `mirror_gather` (one workgroup per 4 KiB page, four
/// dwords per invocation).
pub const GATHER_LOCAL_SIZE: u32 = 256;
/// Dwords per record of `mirror_gather`: the page's index, its 32 mask
/// words, its 1024 dwords.
pub const GATHER_RECORD_DWORDS: u32 = 1 + 32 + 1024;
/// Dwords before the first record (dword 0: the record count).
pub const GATHER_HEADER_DWORDS: u32 = 4;
/// Dwords per record without the page's dwords (`mirror_gather` flags bit
/// 0 clear): its index and its 32 mask words.
pub const GATHER_MASK_RECORD_DWORDS: u32 = 1 + 32;

/// Mirror mode's write-back gather: workgroup g handles page g of a written
/// range. Binding 0 is the written-bit map from the range's first word (32
/// words per page: bit k of word j marks dword 32j+k), binding 1 the
/// mirror from the range's first byte, binding 2 the host output: dword 0
/// counts records, records start at dword GATHER_HEADER_DWORDS
/// (GATHER_RECORD_DWORDS each). A page with marked dwords takes a record
/// (while fewer than `cap` were taken): `first_page + g`, its mask words,
/// its marked dwords at their indices (the others are not written), and
/// clears the bits it recorded (bits set meanwhile stay for a later
/// gather). Pages over the capacity keep their bits. Push constants
/// `{pages, cap, first_page, flags}`: flags bit 0 clear leaves the dwords
/// out (records of GATHER_MASK_RECORD_DWORDS).
pub fn mirror_gather() -> Vec<u32> {
    build_gather().expect("mirror gather kernel")
}

fn build_gather() -> Result<Vec<u32>, rspirv::dr::Error> {
    let lit = Operand::LiteralBit32;
    let mut b = Builder::new();
    b.set_version(1, 3);
    b.capability(Capability::Shader);
    b.extension("SPV_KHR_storage_buffer_storage_class");
    b.memory_model(spirv::AddressingModel::Logical, spirv::MemoryModel::GLSL450);
    let void = b.type_void();
    let bool_ = b.type_bool();
    let u32t = b.type_int(32, 0);
    let v3 = b.type_vector(u32t, 3);
    let rta = b.type_runtime_array(u32t);
    b.decorate(rta, Decoration::ArrayStride, [lit(4)]);
    let st = b.type_struct([rta]);
    b.decorate(st, Decoration::Block, []);
    b.member_decorate(st, 0, Decoration::Offset, [lit(0)]);
    let p_st = b.type_pointer(None, StorageClass::StorageBuffer, st);
    let p_u = b.type_pointer(None, StorageClass::StorageBuffer, u32t);
    let pc = b.type_struct([u32t, u32t, u32t, u32t]);
    b.decorate(pc, Decoration::Block, []);
    for (k, off) in [0u32, 4, 8, 12].into_iter().enumerate() {
        b.member_decorate(pc, k as u32, Decoration::Offset, [lit(off)]);
    }
    let p_pc = b.type_pointer(None, StorageClass::PushConstant, pc);
    let p_pcu = b.type_pointer(None, StorageClass::PushConstant, u32t);
    let p_in = b.type_pointer(None, StorageClass::Input, v3);
    let lid_v = global(&mut b, p_in, StorageClass::Input);
    b.decorate(lid_v, Decoration::BuiltIn, [Operand::BuiltIn(BuiltIn::LocalInvocationId)]);
    let wid_v = global(&mut b, p_in, StorageClass::Input);
    b.decorate(wid_v, Decoration::BuiltIn, [Operand::BuiltIn(BuiltIn::WorkgroupId)]);
    let mut bufs = [0; 3];
    for (i, v) in bufs.iter_mut().enumerate() {
        *v = global(&mut b, p_st, StorageClass::StorageBuffer);
        b.decorate(*v, Decoration::DescriptorSet, [lit(0)]);
        b.decorate(*v, Decoration::Binding, [lit(i as u32)]);
    }
    let (marks, mirror, out) = (bufs[0], bufs[1], bufs[2]);
    b.decorate(mirror, Decoration::NonWritable, []);
    let pcv = global(&mut b, p_pc, StorageClass::PushConstant);
    let c = |b: &mut Builder, v: u32| b.constant_bit32(u32t, v);
    let (c0, c1, c2, c3, c31, c32) = (c(&mut b, 0), c(&mut b, 1), c(&mut b, 2), c(&mut b, 3), c(&mut b, 31), c(&mut b, 32));
    let c1024 = c(&mut b, 1024);
    let c33 = c(&mut b, 33);
    let rec_full = c(&mut b, GATHER_RECORD_DWORDS);
    let rec_mask = c(&mut b, GATHER_MASK_RECORD_DWORDS);
    let hdr = c(&mut b, GATHER_HEADER_DWORDS);
    let arr32 = b.type_array(u32t, c32);
    let p_arr = b.type_pointer(None, StorageClass::Workgroup, arr32);
    let p_wu = b.type_pointer(None, StorageClass::Workgroup, u32t);
    let m_sh = global(&mut b, p_arr, StorageClass::Workgroup);
    let any_sh = global(&mut b, p_wu, StorageClass::Workgroup);
    let slot_sh = global(&mut b, p_wu, StorageClass::Workgroup);
    // Workgroup scope (2), device scope (1); acquire-release (0x8) of
    // workgroup memory (0x100).
    let scope_wg = c(&mut b, 2);
    let scope_dev = c(&mut b, 1);
    let sem_bar = c(&mut b, 0x108);
    let sem_none = c0;
    let fty = b.type_function(void, []);
    let f = b.begin_function(void, None, FunctionControl::NONE, fty)?;
    b.begin_block(None)?;
    let l = b.load(v3, None, lid_v, None, [])?;
    let lid = b.composite_extract(u32t, None, l, [0])?;
    let w = b.load(v3, None, wid_v, None, [])?;
    let pg = b.composite_extract(u32t, None, w, [0])?;
    let pc_at = |b: &mut Builder, k: Word| -> Result<Word, rspirv::dr::Error> {
        let p = b.access_chain(p_pcu, None, pcv, [k])?;
        b.load(u32t, None, p, None, [])
    };
    let cap = pc_at(&mut b, c1)?;
    let first_page = pc_at(&mut b, c2)?;
    let flags = pc_at(&mut b, c3)?;
    let data_bit = b.bitwise_and(u32t, None, flags, c1)?;
    let with_data = b.i_not_equal(bool_, None, data_bit, c0)?;
    let rec_dw = b.select(u32t, None, with_data, rec_full, rec_mask)?;
    let is0 = b.i_equal(bool_, None, lid, c0)?;
    let lt32 = b.u_less_than(bool_, None, lid, c32)?;
    // any = 0
    let (l_z, l_zd) = (b.id(), b.id());
    b.selection_merge(l_zd, SelectionControl::NONE)?;
    b.branch_conditional(is0, l_z, l_zd, [])?;
    b.begin_block(Some(l_z))?;
    b.store(any_sh, c0, None, [])?;
    b.branch(l_zd)?;
    b.begin_block(Some(l_zd))?;
    b.control_barrier(scope_wg, scope_wg, sem_bar)?;
    // The page's mask words.
    let pg32 = b.i_mul(u32t, None, pg, c32)?;
    let wi = b.i_add(u32t, None, pg32, lid)?;
    let (l_m, l_md, l_any, l_anyd) = (b.id(), b.id(), b.id(), b.id());
    b.selection_merge(l_md, SelectionControl::NONE)?;
    b.branch_conditional(lt32, l_m, l_md, [])?;
    b.begin_block(Some(l_m))?;
    let pm = b.access_chain(p_u, None, marks, [c0, wi])?;
    let mw = b.load(u32t, None, pm, None, [])?;
    let psm = b.access_chain(p_wu, None, m_sh, [lid])?;
    b.store(psm, mw, None, [])?;
    let nz = b.i_not_equal(bool_, None, mw, c0)?;
    b.selection_merge(l_anyd, SelectionControl::NONE)?;
    b.branch_conditional(nz, l_any, l_anyd, [])?;
    b.begin_block(Some(l_any))?;
    b.atomic_or(u32t, None, any_sh, scope_wg, sem_none, c1)?;
    b.branch(l_anyd)?;
    b.begin_block(Some(l_anyd))?;
    b.branch(l_md)?;
    b.begin_block(Some(l_md))?;
    b.control_barrier(scope_wg, scope_wg, sem_bar)?;
    let any = b.load(u32t, None, any_sh, None, [])?;
    let touched = b.i_not_equal(bool_, None, any, c0)?;
    // Invocation 0 of a touched page takes a record.
    let take = b.logical_and(bool_, None, touched, is0)?;
    let (l_t, l_td) = (b.id(), b.id());
    b.selection_merge(l_td, SelectionControl::NONE)?;
    b.branch_conditional(take, l_t, l_td, [])?;
    b.begin_block(Some(l_t))?;
    let pcount = b.access_chain(p_u, None, out, [c0, c0])?;
    let s = b.atomic_i_add(u32t, None, pcount, scope_dev, sem_none, c1)?;
    b.store(slot_sh, s, None, [])?;
    b.branch(l_td)?;
    b.begin_block(Some(l_td))?;
    b.control_barrier(scope_wg, scope_wg, sem_bar)?;
    let slot = b.load(u32t, None, slot_sh, None, [])?;
    let room = b.u_less_than(bool_, None, slot, cap)?;
    let go = b.logical_and(bool_, None, touched, room)?;
    let (l_go, l_end) = (b.id(), b.id());
    b.selection_merge(l_end, SelectionControl::NONE)?;
    b.branch_conditional(go, l_go, l_end, [])?;
    b.begin_block(Some(l_go))?;
    let sr = b.i_mul(u32t, None, slot, rec_dw)?;
    let rec = b.i_add(u32t, None, hdr, sr)?;
    // Header: the page's index.
    let (l_h, l_hd) = (b.id(), b.id());
    b.selection_merge(l_hd, SelectionControl::NONE)?;
    b.branch_conditional(is0, l_h, l_hd, [])?;
    b.begin_block(Some(l_h))?;
    let page_abs = b.i_add(u32t, None, first_page, pg)?;
    let ph = b.access_chain(p_u, None, out, [c0, rec])?;
    b.store(ph, page_abs, None, [])?;
    b.branch(l_hd)?;
    b.begin_block(Some(l_hd))?;
    // Mask words: recorded, and their bits cleared in the map.
    let (l_k, l_kd) = (b.id(), b.id());
    b.selection_merge(l_kd, SelectionControl::NONE)?;
    b.branch_conditional(lt32, l_k, l_kd, [])?;
    b.begin_block(Some(l_k))?;
    let psm2 = b.access_chain(p_wu, None, m_sh, [lid])?;
    let mv = b.load(u32t, None, psm2, None, [])?;
    let r1 = b.i_add(u32t, None, rec, c1)?;
    let ri = b.i_add(u32t, None, r1, lid)?;
    let pr = b.access_chain(p_u, None, out, [c0, ri])?;
    b.store(pr, mv, None, [])?;
    let inv = b.not(u32t, None, mv)?;
    let pm2 = b.access_chain(p_u, None, marks, [c0, wi])?;
    b.atomic_and(u32t, None, pm2, scope_dev, sem_none, inv)?;
    b.branch(l_kd)?;
    b.begin_block(Some(l_kd))?;
    // Four dwords per invocation (dwords 4*lid.. share mask word lid / 8),
    // with data.
    let (l_dd, l_ddd) = (b.id(), b.id());
    b.selection_merge(l_ddd, SelectionControl::NONE)?;
    b.branch_conditional(with_data, l_dd, l_ddd, [])?;
    b.begin_block(Some(l_dd))?;
    let wsel = b.shift_right_logical(u32t, None, lid, c3)?;
    let psm3 = b.access_chain(p_wu, None, m_sh, [wsel])?;
    let word = b.load(u32t, None, psm3, None, [])?;
    let d_base = b.shift_left_logical(u32t, None, lid, c2)?;
    let pg1024 = b.i_mul(u32t, None, pg, c1024)?;
    let rdata = b.i_add(u32t, None, rec, c33)?;
    for k in 0..4u32 {
        let kk = c(&mut b, k);
        let d = b.i_add(u32t, None, d_base, kk)?;
        let bit = b.bitwise_and(u32t, None, d, c31)?;
        let sh = b.shift_right_logical(u32t, None, word, bit)?;
        let one = b.bitwise_and(u32t, None, sh, c1)?;
        let set = b.i_not_equal(bool_, None, one, c0)?;
        let (l_s, l_sd) = (b.id(), b.id());
        b.selection_merge(l_sd, SelectionControl::NONE)?;
        b.branch_conditional(set, l_s, l_sd, [])?;
        b.begin_block(Some(l_s))?;
        let md = b.i_add(u32t, None, pg1024, d)?;
        let pmd = b.access_chain(p_u, None, mirror, [c0, md])?;
        let v = b.load(u32t, None, pmd, None, [])?;
        let od = b.i_add(u32t, None, rdata, d)?;
        let pod = b.access_chain(p_u, None, out, [c0, od])?;
        b.store(pod, v, None, [])?;
        b.branch(l_sd)?;
        b.begin_block(Some(l_sd))?;
    }
    b.branch(l_ddd)?;
    b.begin_block(Some(l_ddd))?;
    b.branch(l_end)?;
    b.begin_block(Some(l_end))?;
    b.ret()?;
    b.end_function()?;
    b.entry_point(ExecutionModel::GLCompute, f, "main", [lid_v, wid_v]);
    b.execution_mode(f, ExecutionMode::LocalSize, [GATHER_LOCAL_SIZE, 1, 1]);
    Ok(b.module().assemble())
}

/// A GPU timestamp stored in the title's GPU clock (an end-of-pipe
/// RELEASE_MEM timestamp performed by the GPU): one invocation reads the
/// 64-bit timestamp at index `query` of binding 0 (copied from a query pool,
/// nanoseconds) and stores (t + k) / 10 (100 MHz ticks) as two dwords at
/// dword `dst` of binding 1 (push constants `{query, dst, k_lo, k_hi}`; `k`
/// wraps, so it may stand for a negative offset).
pub fn timestamp_store() -> Vec<u32> {
    build_timestamp().expect("timestamp kernel")
}

fn build_timestamp() -> Result<Vec<u32>, rspirv::dr::Error> {
    let lit = Operand::LiteralBit32;
    let mut b = Builder::new();
    b.set_version(1, 3);
    b.capability(Capability::Shader);
    b.capability(Capability::Int64);
    b.extension("SPV_KHR_storage_buffer_storage_class");
    b.memory_model(spirv::AddressingModel::Logical, spirv::MemoryModel::GLSL450);
    let void = b.type_void();
    let u32t = b.type_int(32, 0);
    let u64t = b.type_int(64, 0);
    let rta64 = b.type_runtime_array(u64t);
    b.decorate(rta64, Decoration::ArrayStride, [lit(8)]);
    let st64 = b.type_struct([rta64]);
    b.decorate(st64, Decoration::Block, []);
    b.member_decorate(st64, 0, Decoration::Offset, [lit(0)]);
    let p_st64 = b.type_pointer(None, StorageClass::StorageBuffer, st64);
    let p_u64 = b.type_pointer(None, StorageClass::StorageBuffer, u64t);
    let rta = b.type_runtime_array(u32t);
    b.decorate(rta, Decoration::ArrayStride, [lit(4)]);
    let st = b.type_struct([rta]);
    b.decorate(st, Decoration::Block, []);
    b.member_decorate(st, 0, Decoration::Offset, [lit(0)]);
    let p_st = b.type_pointer(None, StorageClass::StorageBuffer, st);
    let p_u = b.type_pointer(None, StorageClass::StorageBuffer, u32t);
    let pc = b.type_struct([u32t, u32t, u32t, u32t]);
    b.decorate(pc, Decoration::Block, []);
    for i in 0..4 {
        b.member_decorate(pc, i, Decoration::Offset, [lit(4 * i)]);
    }
    let p_pc = b.type_pointer(None, StorageClass::PushConstant, pc);
    let p_pcu = b.type_pointer(None, StorageClass::PushConstant, u32t);
    let src = global(&mut b, p_st64, StorageClass::StorageBuffer);
    b.decorate(src, Decoration::DescriptorSet, [lit(0)]);
    b.decorate(src, Decoration::Binding, [lit(0)]);
    b.decorate(src, Decoration::NonWritable, []);
    let dst = global(&mut b, p_st, StorageClass::StorageBuffer);
    b.decorate(dst, Decoration::DescriptorSet, [lit(0)]);
    b.decorate(dst, Decoration::Binding, [lit(1)]);
    let pcv = global(&mut b, p_pc, StorageClass::PushConstant);
    let c: Vec<Word> = (0..4).map(|i| b.constant_bit32(u32t, i)).collect();
    let c32 = b.constant_bit64(u64t, 32);
    let c10 = b.constant_bit64(u64t, 10);
    let fty = b.type_function(void, []);
    let f = b.begin_function(void, None, FunctionControl::NONE, fty)?;
    b.begin_block(None)?;
    let mut pcs = [0; 4];
    for (i, v) in pcs.iter_mut().enumerate() {
        let p = b.access_chain(p_pcu, None, pcv, [c[i]])?;
        *v = b.load(u32t, None, p, None, [])?;
    }
    let pt = b.access_chain(p_u64, None, src, [c[0], pcs[0]])?;
    let t = b.load(u64t, None, pt, None, [])?;
    let klo = b.u_convert(u64t, None, pcs[2])?;
    let khi = b.u_convert(u64t, None, pcs[3])?;
    let khi = b.shift_left_logical(u64t, None, khi, c32)?;
    let k = b.bitwise_or(u64t, None, khi, klo)?;
    let sum = b.i_add(u64t, None, t, k)?;
    let v = b.u_div(u64t, None, sum, c10)?;
    let lo = b.u_convert(u32t, None, v)?;
    let vh = b.shift_right_logical(u64t, None, v, c32)?;
    let hi = b.u_convert(u32t, None, vh)?;
    let p0 = b.access_chain(p_u, None, dst, [c[0], pcs[1]])?;
    b.store(p0, lo, None, [])?;
    let d1 = b.i_add(u32t, None, pcs[1], c[1])?;
    let p1 = b.access_chain(p_u, None, dst, [c[0], d1])?;
    b.store(p1, hi, None, [])?;
    b.ret()?;
    b.end_function()?;
    b.entry_point(ExecutionModel::GLCompute, f, "main", []);
    b.execution_mode(f, ExecutionMode::LocalSize, [1, 1, 1]);
    Ok(b.module().assemble())
}

/// Timestamps `timestamp_store_many` stores per dispatch (its push
/// constants: 128 bytes, the guaranteed minimum).
pub const TS_MANY: u32 = 14;

/// Up to `TS_MANY` GPU timestamps stored in the title's GPU clock by one
/// dispatch (`timestamp_store` stores one): invocation i < count reads the
/// 64-bit timestamp at index q_i of binding 0 and stores (t + k) / 10 as two
/// dwords at dword d_i of binding 1 (push constants
/// `{k_lo, k_hi, count, 0, q_0, d_0, ..., q_13, d_13}`).
pub fn timestamp_store_many() -> Vec<u32> {
    build_timestamp_many().expect("timestamp kernel (many)")
}

fn build_timestamp_many() -> Result<Vec<u32>, rspirv::dr::Error> {
    let lit = Operand::LiteralBit32;
    let mut b = Builder::new();
    b.set_version(1, 3);
    b.capability(Capability::Shader);
    b.capability(Capability::Int64);
    b.extension("SPV_KHR_storage_buffer_storage_class");
    b.memory_model(spirv::AddressingModel::Logical, spirv::MemoryModel::GLSL450);
    let void = b.type_void();
    let bool_ = b.type_bool();
    let u32t = b.type_int(32, 0);
    let u64t = b.type_int(64, 0);
    let v3 = b.type_vector(u32t, 3);
    let rta64 = b.type_runtime_array(u64t);
    b.decorate(rta64, Decoration::ArrayStride, [lit(8)]);
    let st64 = b.type_struct([rta64]);
    b.decorate(st64, Decoration::Block, []);
    b.member_decorate(st64, 0, Decoration::Offset, [lit(0)]);
    let p_st64 = b.type_pointer(None, StorageClass::StorageBuffer, st64);
    let p_u64 = b.type_pointer(None, StorageClass::StorageBuffer, u64t);
    let rta = b.type_runtime_array(u32t);
    b.decorate(rta, Decoration::ArrayStride, [lit(4)]);
    let st = b.type_struct([rta]);
    b.decorate(st, Decoration::Block, []);
    b.member_decorate(st, 0, Decoration::Offset, [lit(0)]);
    let p_st = b.type_pointer(None, StorageClass::StorageBuffer, st);
    let p_u = b.type_pointer(None, StorageClass::StorageBuffer, u32t);
    let n_words = b.constant_bit32(u32t, 4 + 2 * TS_MANY);
    let arr = b.type_array(u32t, n_words);
    b.decorate(arr, Decoration::ArrayStride, [lit(4)]);
    let pc = b.type_struct([arr]);
    b.decorate(pc, Decoration::Block, []);
    b.member_decorate(pc, 0, Decoration::Offset, [lit(0)]);
    let p_pc = b.type_pointer(None, StorageClass::PushConstant, pc);
    let p_pcu = b.type_pointer(None, StorageClass::PushConstant, u32t);
    let p_in = b.type_pointer(None, StorageClass::Input, v3);
    let lid = global(&mut b, p_in, StorageClass::Input);
    b.decorate(lid, Decoration::BuiltIn, [Operand::BuiltIn(BuiltIn::LocalInvocationId)]);
    let src = global(&mut b, p_st64, StorageClass::StorageBuffer);
    b.decorate(src, Decoration::DescriptorSet, [lit(0)]);
    b.decorate(src, Decoration::Binding, [lit(0)]);
    b.decorate(src, Decoration::NonWritable, []);
    let dst = global(&mut b, p_st, StorageClass::StorageBuffer);
    b.decorate(dst, Decoration::DescriptorSet, [lit(0)]);
    b.decorate(dst, Decoration::Binding, [lit(1)]);
    let pcv = global(&mut b, p_pc, StorageClass::PushConstant);
    let c: Vec<Word> = (0..6).map(|i| b.constant_bit32(u32t, i)).collect();
    let c32 = b.constant_bit64(u64t, 32);
    let c10 = b.constant_bit64(u64t, 10);
    let fty = b.type_function(void, []);
    let f = b.begin_function(void, None, FunctionControl::NONE, fty)?;
    b.begin_block(None)?;
    let l = b.load(v3, None, lid, None, [])?;
    let x = b.composite_extract(u32t, None, l, [0])?;
    let pn = b.access_chain(p_pcu, None, pcv, [c[0], c[2]])?;
    let count = b.load(u32t, None, pn, None, [])?;
    let inside = b.u_less_than(bool_, None, x, count)?;
    let (l_body, l_end) = (b.id(), b.id());
    b.selection_merge(l_end, SelectionControl::NONE)?;
    b.branch_conditional(inside, l_body, l_end, [])?;
    b.begin_block(Some(l_body))?;
    let x2 = b.i_mul(u32t, None, x, c[2])?;
    let iq = b.i_add(u32t, None, x2, c[4])?;
    let id_ = b.i_add(u32t, None, x2, c[5])?;
    let pq = b.access_chain(p_pcu, None, pcv, [c[0], iq])?;
    let q = b.load(u32t, None, pq, None, [])?;
    let pd = b.access_chain(p_pcu, None, pcv, [c[0], id_])?;
    let d = b.load(u32t, None, pd, None, [])?;
    let pklo = b.access_chain(p_pcu, None, pcv, [c[0], c[0]])?;
    let klo = b.load(u32t, None, pklo, None, [])?;
    let pkhi = b.access_chain(p_pcu, None, pcv, [c[0], c[1]])?;
    let khi = b.load(u32t, None, pkhi, None, [])?;
    let klo = b.u_convert(u64t, None, klo)?;
    let khi = b.u_convert(u64t, None, khi)?;
    let khi = b.shift_left_logical(u64t, None, khi, c32)?;
    let k = b.bitwise_or(u64t, None, khi, klo)?;
    let pt = b.access_chain(p_u64, None, src, [c[0], q])?;
    let t = b.load(u64t, None, pt, None, [])?;
    let sum = b.i_add(u64t, None, t, k)?;
    let v = b.u_div(u64t, None, sum, c10)?;
    let lo = b.u_convert(u32t, None, v)?;
    let vh = b.shift_right_logical(u64t, None, v, c32)?;
    let hi = b.u_convert(u32t, None, vh)?;
    let p0 = b.access_chain(p_u, None, dst, [c[0], d])?;
    b.store(p0, lo, None, [])?;
    let d1 = b.i_add(u32t, None, d, c[1])?;
    let p1 = b.access_chain(p_u, None, dst, [c[0], d1])?;
    b.store(p1, hi, None, [])?;
    b.branch(l_end)?;
    b.begin_block(Some(l_end))?;
    b.ret()?;
    b.end_function()?;
    b.entry_point(ExecutionModel::GLCompute, f, "main", [lid]);
    b.execution_mode(f, ExecutionMode::LocalSize, [16, 1, 1]);
    Ok(b.module().assemble())
}

#[cfg(test)]
mod tests {
    #[test]
    fn timestamp_many_kernel_assembles() {
        let w = super::timestamp_store_many();
        assert_eq!(w[0], 0x0723_0203);
        if let Ok(dir) = std::env::var("RDNA_KERNEL_DUMP") {
            let bytes: Vec<u8> = w.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(std::path::Path::new(&dir).join("timestamp_store_many.spv"), bytes).unwrap();
        }
    }

    #[test]
    fn timestamp_kernel_assembles() {
        let w = super::timestamp_store();
        assert_eq!(w[0], 0x0723_0203);
        if let Ok(dir) = std::env::var("RDNA_KERNEL_DUMP") {
            let bytes: Vec<u8> = w.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(std::path::Path::new(&dir).join("timestamp_store.spv"), bytes).unwrap();
        }
    }

    #[test]
    fn gather_kernel_assembles() {
        let w = super::mirror_gather();
        assert_eq!(w[0], 0x0723_0203);
        if let Ok(dir) = std::env::var("RDNA_KERNEL_DUMP") {
            let bytes: Vec<u8> = w.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(std::path::Path::new(&dir).join("mirror_gather.spv"), bytes).unwrap();
        }
    }

    #[test]
    fn merge_kernel_assembles() {
        let w = super::shadow_merge();
        assert_eq!(w[0], 0x0723_0203);
        if let Ok(dir) = std::env::var("RDNA_KERNEL_DUMP") {
            let bytes: Vec<u8> = w.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(std::path::Path::new(&dir).join("shadow_merge.spv"), bytes).unwrap();
        }
    }
}
