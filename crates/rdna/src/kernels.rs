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

/// The sync of a kept window (`gpu::keep`), for dword indices `first..end`
/// (push constants `{first, end, give}`), with binding 0 guest memory,
/// binding 1 the pristine copy (what guest memory and the video-memory copy
/// last agreed on) and binding 2 the video-memory copy: a dword the CPU
/// changed (guest memory differs from the pristine copy) goes to the copy and
/// the pristine copy, whatever the GPU did (the title rewrites memory after
/// the GPU is done with it: ring allocations); with `give` set, a dword only
/// the GPU changed goes to guest memory and the pristine copy. Dwords neither
/// side changed are left alone (MERGE_LOCAL_SIZE per workgroup).
pub fn keep_sync() -> Vec<u32> {
    build_sync().expect("keep sync kernel")
}

fn build_sync() -> Result<Vec<u32>, rspirv::dr::Error> {
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
    let pc = b.type_struct([u32t, u32t, u32t]);
    b.decorate(pc, Decoration::Block, []);
    for k in 0..3 {
        b.member_decorate(pc, k, Decoration::Offset, [lit(4 * k)]);
    }
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
    let pcv = global(&mut b, p_pc, StorageClass::PushConstant);
    let c0 = b.constant_bit32(u32t, 0);
    let c1 = b.constant_bit32(u32t, 1);
    let c2 = b.constant_bit32(u32t, 2);
    let fty = b.type_function(void, []);
    let f = b.begin_function(void, None, FunctionControl::NONE, fty)?;
    b.begin_block(None)?;
    let g = b.load(v3, None, gid, None, [])?;
    let x = b.composite_extract(u32t, None, g, [0])?;
    let pf = b.access_chain(p_pcu, None, pcv, [c0])?;
    let first = b.load(u32t, None, pf, None, [])?;
    let pe = b.access_chain(p_pcu, None, pcv, [c1])?;
    let end = b.load(u32t, None, pe, None, [])?;
    let pg = b.access_chain(p_pcu, None, pcv, [c2])?;
    let give = b.load(u32t, None, pg, None, [])?;
    let i = b.i_add(u32t, None, x, first)?;
    let inside = b.u_less_than(bool_, None, i, end)?;
    let (l_body, l_cpu, l_else, l_gpu, l_join2, l_join, l_end) = (b.id(), b.id(), b.id(), b.id(), b.id(), b.id(), b.id());
    b.selection_merge(l_end, SelectionControl::NONE)?;
    b.branch_conditional(inside, l_body, l_end, [])?;
    b.begin_block(Some(l_body))?;
    let ph = b.access_chain(p_u, None, bufs[0], [c0, i])?;
    let h = b.load(u32t, None, ph, None, [])?;
    let pp = b.access_chain(p_u, None, bufs[1], [c0, i])?;
    let p = b.load(u32t, None, pp, None, [])?;
    let pw = b.access_chain(p_u, None, bufs[2], [c0, i])?;
    let w = b.load(u32t, None, pw, None, [])?;
    let cpu = b.i_not_equal(bool_, None, h, p)?;
    b.selection_merge(l_join, SelectionControl::NONE)?;
    b.branch_conditional(cpu, l_cpu, l_else, [])?;
    // The CPU changed it: the copy follows.
    b.begin_block(Some(l_cpu))?;
    b.store(pw, h, None, [])?;
    b.store(pp, h, None, [])?;
    b.branch(l_join)?;
    // Only the GPU changed it: back to guest memory when giving.
    b.begin_block(Some(l_else))?;
    let giving = b.i_not_equal(bool_, None, give, c0)?;
    let gpu = b.i_not_equal(bool_, None, w, p)?;
    let both = b.logical_and(bool_, None, giving, gpu)?;
    b.selection_merge(l_join2, SelectionControl::NONE)?;
    b.branch_conditional(both, l_gpu, l_join2, [])?;
    b.begin_block(Some(l_gpu))?;
    b.store(ph, w, None, [])?;
    b.store(pp, w, None, [])?;
    b.branch(l_join2)?;
    b.begin_block(Some(l_join2))?;
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

/// Dwords per entry of `shadow_merge_many`'s table.
pub const MERGE_ENTRY_DWORDS: u32 = 8;
/// Entries one `shadow_merge_many` dispatch takes (its search's reach).
pub const MERGE_MANY_MAX: u32 = 256;

/// The write-back of several shadows in one dispatch (`shadow_merge` for
/// each): binding 0 holds their working and pristine copies, binding 1 the
/// table: `n` entries (push constant 1, at most MERGE_MANY_MAX) from dword
/// `at` (push constant 0), MERGE_ENTRY_DWORDS each: [first workgroup, the
/// working copy's dword offset in binding 0, the pristine copy's, first
/// dword, end dword, the destination's device address (low, high: guest
/// memory at the copy's start), 0], in increasing first-workgroup order from
/// 0. Workgroup g takes the last entry whose first workgroup is at most g
/// (an unrolled binary search) and its dwords first + (g - first workgroup)
/// * MERGE_LOCAL_SIZE + local x below end: one that differs from its
/// pristine copy is stored to the destination + 4 * index.
pub fn shadow_merge_many() -> Vec<u32> {
    build_merge_many().expect("shadow merge (many) kernel")
}

fn build_merge_many() -> Result<Vec<u32>, rspirv::dr::Error> {
    let lit = Operand::LiteralBit32;
    let mut b = Builder::new();
    b.set_version(1, 3);
    b.capability(Capability::Shader);
    b.capability(Capability::Int64);
    b.capability(Capability::PhysicalStorageBufferAddresses);
    b.extension("SPV_KHR_storage_buffer_storage_class");
    b.extension("SPV_KHR_physical_storage_buffer");
    b.memory_model(spirv::AddressingModel::PhysicalStorageBuffer64, spirv::MemoryModel::GLSL450);
    let void = b.type_void();
    let bool_ = b.type_bool();
    let u32t = b.type_int(32, 0);
    let u64t = b.type_int(64, 0);
    let v3 = b.type_vector(u32t, 3);
    let rta = b.type_runtime_array(u32t);
    b.decorate(rta, Decoration::ArrayStride, [lit(4)]);
    let st = b.type_struct([rta]);
    b.decorate(st, Decoration::Block, []);
    b.member_decorate(st, 0, Decoration::Offset, [lit(0)]);
    let p_st = b.type_pointer(None, StorageClass::StorageBuffer, st);
    let p_u = b.type_pointer(None, StorageClass::StorageBuffer, u32t);
    let p_dev = b.type_pointer(None, StorageClass::PhysicalStorageBuffer, u32t);
    let pc = b.type_struct([u32t, u32t]);
    b.decorate(pc, Decoration::Block, []);
    b.member_decorate(pc, 0, Decoration::Offset, [lit(0)]);
    b.member_decorate(pc, 1, Decoration::Offset, [lit(4)]);
    let p_pc = b.type_pointer(None, StorageClass::PushConstant, pc);
    let p_pcu = b.type_pointer(None, StorageClass::PushConstant, u32t);
    let p_in = b.type_pointer(None, StorageClass::Input, v3);
    let lid_v = global(&mut b, p_in, StorageClass::Input);
    b.decorate(lid_v, Decoration::BuiltIn, [Operand::BuiltIn(BuiltIn::LocalInvocationId)]);
    let wid_v = global(&mut b, p_in, StorageClass::Input);
    b.decorate(wid_v, Decoration::BuiltIn, [Operand::BuiltIn(BuiltIn::WorkgroupId)]);
    let mut bufs = [0; 2];
    for (i, v) in bufs.iter_mut().enumerate() {
        *v = global(&mut b, p_st, StorageClass::StorageBuffer);
        b.decorate(*v, Decoration::DescriptorSet, [lit(0)]);
        b.decorate(*v, Decoration::Binding, [lit(i as u32)]);
        b.decorate(*v, Decoration::NonWritable, []);
    }
    let (copies, table) = (bufs[0], bufs[1]);
    let pcv = global(&mut b, p_pc, StorageClass::PushConstant);
    let c = |b: &mut Builder, v: u32| b.constant_bit32(u32t, v);
    let (c0, c1, c4) = (c(&mut b, 0), c(&mut b, 1), c(&mut b, 4));
    let c32_64 = b.constant_bit64(u64t, 32);
    let ced = c(&mut b, MERGE_ENTRY_DWORDS);
    let cls = c(&mut b, MERGE_LOCAL_SIZE);
    let fty = b.type_function(void, []);
    let f = b.begin_function(void, None, FunctionControl::NONE, fty)?;
    b.begin_block(None)?;
    let l = b.load(v3, None, lid_v, None, [])?;
    let lx = b.composite_extract(u32t, None, l, [0])?;
    let w = b.load(v3, None, wid_v, None, [])?;
    let g = b.composite_extract(u32t, None, w, [0])?;
    let pa = b.access_chain(p_pcu, None, pcv, [c0])?;
    let at = b.load(u32t, None, pa, None, [])?;
    let pn = b.access_chain(p_pcu, None, pcv, [c1])?;
    let n = b.load(u32t, None, pn, None, [])?;
    let word = |b: &mut Builder, e: Word, k: u32| -> Result<Word, rspirv::dr::Error> {
        let ck = b.constant_bit32(u32t, k);
        let base = b.i_mul(u32t, None, e, ced)?;
        let i = b.i_add(u32t, None, at, base)?;
        let i = b.i_add(u32t, None, i, ck)?;
        let p = b.access_chain(p_u, None, table, [c0, i])?;
        b.load(u32t, None, p, None, [])
    };
    // The last entry whose first workgroup is at most g.
    let mut lo = c0;
    let mut step = MERGE_MANY_MAX / 2;
    while step > 0 {
        let cs = b.constant_bit32(u32t, step);
        let cand = b.i_add(u32t, None, lo, cs)?;
        let inr = b.u_less_than(bool_, None, cand, n)?;
        // Entries past n are not read (their index is clamped to 0).
        let safe = b.select(u32t, None, inr, cand, c0)?;
        let wg = word(&mut b, safe, 0)?;
        let le = b.u_less_than_equal(bool_, None, wg, g)?;
        let take = b.logical_and(bool_, None, inr, le)?;
        lo = b.select(u32t, None, take, cand, lo)?;
        step /= 2;
    }
    let wg0 = word(&mut b, lo, 0)?;
    let wdw = word(&mut b, lo, 1)?;
    let pdw = word(&mut b, lo, 2)?;
    let first = word(&mut b, lo, 3)?;
    let end = word(&mut b, lo, 4)?;
    let dlo = word(&mut b, lo, 5)?;
    let dhi = word(&mut b, lo, 6)?;
    let rel = b.i_sub(u32t, None, g, wg0)?;
    let rel = b.i_mul(u32t, None, rel, cls)?;
    let i = b.i_add(u32t, None, first, rel)?;
    let i = b.i_add(u32t, None, i, lx)?;
    let inside = b.u_less_than(bool_, None, i, end)?;
    let (l_body, l_store, l_join, l_end) = (b.id(), b.id(), b.id(), b.id());
    b.selection_merge(l_end, SelectionControl::NONE)?;
    b.branch_conditional(inside, l_body, l_end, [])?;
    b.begin_block(Some(l_body))?;
    let wi = b.i_add(u32t, None, wdw, i)?;
    let pw = b.access_chain(p_u, None, copies, [c0, wi])?;
    let wv = b.load(u32t, None, pw, None, [])?;
    let pi = b.i_add(u32t, None, pdw, i)?;
    let pp = b.access_chain(p_u, None, copies, [c0, pi])?;
    let old = b.load(u32t, None, pp, None, [])?;
    let changed = b.i_not_equal(bool_, None, wv, old)?;
    b.selection_merge(l_join, SelectionControl::NONE)?;
    b.branch_conditional(changed, l_store, l_join, [])?;
    b.begin_block(Some(l_store))?;
    let lo64 = b.u_convert(u64t, None, dlo)?;
    let hi64 = b.u_convert(u64t, None, dhi)?;
    let hi64 = b.shift_left_logical(u64t, None, hi64, c32_64)?;
    let base = b.bitwise_or(u64t, None, lo64, hi64)?;
    let i4 = b.i_mul(u32t, None, i, c4)?;
    let i64_ = b.u_convert(u64t, None, i4)?;
    let a = b.i_add(u64t, None, base, i64_)?;
    let ptr = b.convert_u_to_ptr(p_dev, None, a)?;
    b.store(ptr, wv, Some(spirv::MemoryAccess::ALIGNED), [lit(4)])?;
    b.branch(l_join)?;
    b.begin_block(Some(l_join))?;
    b.branch(l_end)?;
    b.begin_block(Some(l_end))?;
    b.ret()?;
    b.end_function()?;
    b.entry_point(ExecutionModel::GLCompute, f, "main", [lid_v, wid_v]);
    b.execution_mode(f, ExecutionMode::LocalSize, [MERGE_LOCAL_SIZE, 1, 1]);
    Ok(b.module().assemble())
}

/// Dwords per entry of `resident_merge`'s table.
pub const RESIDENT_ENTRY_DWORDS: u32 = 8;

/// The write-back of resident GPU-written data (the runtime's `gpu::rgm`):
/// the working copy W and the reference R are two buffers over the same
/// guest span, by device address (push constants: table start `at` in
/// dwords, entries `n` (at most MERGE_MANY_MAX), W's address (low, high),
/// R's (low, high)); binding 0 holds the table, RESIDENT_ENTRY_DWORDS per
/// entry: [first workgroup, byte offset in the span (low, high), the
/// destination's device address (low, high: guest memory at that offset),
/// dwords, 0, 0], in increasing first-workgroup order from 0. Workgroup g
/// takes the last entry whose first workgroup is at most g, and its dword i
/// = (g - first workgroup) * MERGE_LOCAL_SIZE + local x below the entry's
/// dwords: one where W differs from R is stored to the destination + 4 * i
/// and to R (what guest memory now holds there).
pub fn resident_merge() -> Vec<u32> {
    build_resident_merge().expect("resident merge kernel")
}

fn build_resident_merge() -> Result<Vec<u32>, rspirv::dr::Error> {
    let lit = Operand::LiteralBit32;
    let mut b = Builder::new();
    b.set_version(1, 3);
    b.capability(Capability::Shader);
    b.capability(Capability::Int64);
    b.capability(Capability::PhysicalStorageBufferAddresses);
    b.extension("SPV_KHR_storage_buffer_storage_class");
    b.extension("SPV_KHR_physical_storage_buffer");
    b.memory_model(spirv::AddressingModel::PhysicalStorageBuffer64, spirv::MemoryModel::GLSL450);
    let void = b.type_void();
    let bool_ = b.type_bool();
    let u32t = b.type_int(32, 0);
    let u64t = b.type_int(64, 0);
    let v3 = b.type_vector(u32t, 3);
    let rta = b.type_runtime_array(u32t);
    b.decorate(rta, Decoration::ArrayStride, [lit(4)]);
    let st = b.type_struct([rta]);
    b.decorate(st, Decoration::Block, []);
    b.member_decorate(st, 0, Decoration::Offset, [lit(0)]);
    let p_st = b.type_pointer(None, StorageClass::StorageBuffer, st);
    let p_u = b.type_pointer(None, StorageClass::StorageBuffer, u32t);
    let p_dev = b.type_pointer(None, StorageClass::PhysicalStorageBuffer, u32t);
    let pc = b.type_struct([u32t, u32t, u32t, u32t, u32t, u32t]);
    b.decorate(pc, Decoration::Block, []);
    for k in 0..6 {
        b.member_decorate(pc, k, Decoration::Offset, [lit(4 * k)]);
    }
    let p_pc = b.type_pointer(None, StorageClass::PushConstant, pc);
    let p_pcu = b.type_pointer(None, StorageClass::PushConstant, u32t);
    let p_in = b.type_pointer(None, StorageClass::Input, v3);
    let lid_v = global(&mut b, p_in, StorageClass::Input);
    b.decorate(lid_v, Decoration::BuiltIn, [Operand::BuiltIn(BuiltIn::LocalInvocationId)]);
    let wid_v = global(&mut b, p_in, StorageClass::Input);
    b.decorate(wid_v, Decoration::BuiltIn, [Operand::BuiltIn(BuiltIn::WorkgroupId)]);
    let table = global(&mut b, p_st, StorageClass::StorageBuffer);
    b.decorate(table, Decoration::DescriptorSet, [lit(0)]);
    b.decorate(table, Decoration::Binding, [lit(0)]);
    b.decorate(table, Decoration::NonWritable, []);
    let pcv = global(&mut b, p_pc, StorageClass::PushConstant);
    let c = |b: &mut Builder, v: u32| b.constant_bit32(u32t, v);
    let (c0, c1, c2, c3, c4, c5) = (c(&mut b, 0), c(&mut b, 1), c(&mut b, 2), c(&mut b, 3), c(&mut b, 4), c(&mut b, 5));
    let c32_64 = b.constant_bit64(u64t, 32);
    let ced = c(&mut b, RESIDENT_ENTRY_DWORDS);
    let cls = c(&mut b, MERGE_LOCAL_SIZE);
    let fty = b.type_function(void, []);
    let f = b.begin_function(void, None, FunctionControl::NONE, fty)?;
    b.begin_block(None)?;
    let l = b.load(v3, None, lid_v, None, [])?;
    let lx = b.composite_extract(u32t, None, l, [0])?;
    let w = b.load(v3, None, wid_v, None, [])?;
    let g = b.composite_extract(u32t, None, w, [0])?;
    let pcw = |b: &mut Builder, k: Word| -> Result<Word, rspirv::dr::Error> {
        let p = b.access_chain(p_pcu, None, pcv, [k])?;
        b.load(u32t, None, p, None, [])
    };
    let at = pcw(&mut b, c0)?;
    let n = pcw(&mut b, c1)?;
    let wlo = pcw(&mut b, c2)?;
    let whi = pcw(&mut b, c3)?;
    let rlo = pcw(&mut b, c4)?;
    let rhi = pcw(&mut b, c5)?;
    let addr64 = |b: &mut Builder, lo: Word, hi: Word| -> Result<Word, rspirv::dr::Error> {
        let lo64 = b.u_convert(u64t, None, lo)?;
        let hi64 = b.u_convert(u64t, None, hi)?;
        let hi64 = b.shift_left_logical(u64t, None, hi64, c32_64)?;
        b.bitwise_or(u64t, None, lo64, hi64)
    };
    let word = |b: &mut Builder, e: Word, k: u32| -> Result<Word, rspirv::dr::Error> {
        let ck = b.constant_bit32(u32t, k);
        let base = b.i_mul(u32t, None, e, ced)?;
        let i = b.i_add(u32t, None, at, base)?;
        let i = b.i_add(u32t, None, i, ck)?;
        let p = b.access_chain(p_u, None, table, [c0, i])?;
        b.load(u32t, None, p, None, [])
    };
    // The last entry whose first workgroup is at most g.
    let mut lo = c0;
    let mut step = MERGE_MANY_MAX / 2;
    while step > 0 {
        let cs = b.constant_bit32(u32t, step);
        let cand = b.i_add(u32t, None, lo, cs)?;
        let inr = b.u_less_than(bool_, None, cand, n)?;
        // Entries past n are not read (their index is clamped to 0).
        let safe = b.select(u32t, None, inr, cand, c0)?;
        let wg = word(&mut b, safe, 0)?;
        let le = b.u_less_than_equal(bool_, None, wg, g)?;
        let take = b.logical_and(bool_, None, inr, le)?;
        lo = b.select(u32t, None, take, cand, lo)?;
        step /= 2;
    }
    let wg0 = word(&mut b, lo, 0)?;
    let olo = word(&mut b, lo, 1)?;
    let ohi = word(&mut b, lo, 2)?;
    let dlo = word(&mut b, lo, 3)?;
    let dhi = word(&mut b, lo, 4)?;
    let dwords = word(&mut b, lo, 5)?;
    let rel = b.i_sub(u32t, None, g, wg0)?;
    let rel = b.i_mul(u32t, None, rel, cls)?;
    let i = b.i_add(u32t, None, rel, lx)?;
    let inside = b.u_less_than(bool_, None, i, dwords)?;
    let (l_body, l_store, l_join, l_end) = (b.id(), b.id(), b.id(), b.id());
    b.selection_merge(l_end, SelectionControl::NONE)?;
    b.branch_conditional(inside, l_body, l_end, [])?;
    b.begin_block(Some(l_body))?;
    let i4 = b.i_mul(u32t, None, i, c4)?;
    let i64_ = b.u_convert(u64t, None, i4)?;
    let off = addr64(&mut b, olo, ohi)?;
    let off = b.i_add(u64t, None, off, i64_)?;
    let wbase = addr64(&mut b, wlo, whi)?;
    let rbase = addr64(&mut b, rlo, rhi)?;
    let wa = b.i_add(u64t, None, wbase, off)?;
    let ra = b.i_add(u64t, None, rbase, off)?;
    let wp = b.convert_u_to_ptr(p_dev, None, wa)?;
    let rp = b.convert_u_to_ptr(p_dev, None, ra)?;
    let wv = b.load(u32t, None, wp, Some(spirv::MemoryAccess::ALIGNED), [lit(4)])?;
    let rv = b.load(u32t, None, rp, Some(spirv::MemoryAccess::ALIGNED), [lit(4)])?;
    let changed = b.i_not_equal(bool_, None, wv, rv)?;
    b.selection_merge(l_join, SelectionControl::NONE)?;
    b.branch_conditional(changed, l_store, l_join, [])?;
    b.begin_block(Some(l_store))?;
    let dbase = addr64(&mut b, dlo, dhi)?;
    let da = b.i_add(u64t, None, dbase, i64_)?;
    let dp = b.convert_u_to_ptr(p_dev, None, da)?;
    b.store(dp, wv, Some(spirv::MemoryAccess::ALIGNED), [lit(4)])?;
    b.store(rp, wv, Some(spirv::MemoryAccess::ALIGNED), [lit(4)])?;
    b.branch(l_join)?;
    b.begin_block(Some(l_join))?;
    b.branch(l_end)?;
    b.begin_block(Some(l_end))?;
    b.ret()?;
    b.end_function()?;
    b.entry_point(ExecutionModel::GLCompute, f, "main", [lid_v, wid_v]);
    b.execution_mode(f, ExecutionMode::LocalSize, [MERGE_LOCAL_SIZE, 1, 1]);
    Ok(b.module().assemble())
}

/// The written-bit map's layout (`Spec::write_marks`), in u32 words of its
/// buffer: one bit per dword of the mirror's span (MARK_SPAN_BYTES), then
/// one bit per 4 KiB page of it (pages with marks in the batch running),
/// then the batch's touched-page list: [count, 1, 1, 0] (an indirect
/// dispatch's arguments) and up to MARK_LIST_CAP page indices.
pub const MARK_SPAN_BYTES: u64 = 32 << 30;
pub const MARK_WORDS: u32 = (MARK_SPAN_BYTES / 128) as u32;
pub const MARK_PAGE_BITS_AT: u32 = MARK_WORDS;
pub const MARK_LIST_AT: u32 = MARK_PAGE_BITS_AT + (MARK_SPAN_BYTES / 4096 / 32) as u32;
pub const MARK_LIST_CAP: u32 = 1 << 20;
pub const MARK_TOTAL_WORDS: u32 = MARK_LIST_AT + 4 + MARK_LIST_CAP;

/// Workgroup size of `mirror_gather`: one invocation per mark word of a
/// page.
pub const GATHER_LOCAL_SIZE: u32 = 32;
/// Dwords per record of `mirror_gather`: the page's index and its 32 mask
/// words.
pub const GATHER_MASK_RECORD_DWORDS: u32 = 1 + 32;
/// Dwords before the first record: the record count, the touched-page
/// list's count (list mode), two unused.
pub const GATHER_HEADER_DWORDS: u32 = 4;

/// Mirror mode's write-back gather: workgroup g takes a page of the mirror,
/// with flags bit 0 the g-th of the touched-page list (MARK_LIST_AT; past
/// its count or MARK_LIST_CAP it does nothing; workgroup 0 copies the
/// list's count to output dword 1), else page `first_page + g` for g below
/// `pages`. Binding 0 is the whole written-bit map, binding 1 the host
/// output: dword 0 counts records, records start at dword
/// GATHER_HEADER_DWORDS. A page with marked dwords takes a record (while
/// fewer than `cap` were taken): its index and its 32 mask words, whose
/// bits it clears; a page over the capacity keeps its bits (a later range
/// gather takes them). Every page taken gets its touched bit cleared. Push
/// constants `{pages, cap, first_page, flags}`.
pub fn mirror_gather() -> Vec<u32> {
    build_gather().expect("mirror gather kernel")
}

fn build_gather() -> Result<Vec<u32>, rspirv::dr::Error> {
    let lit = Operand::LiteralBit32;
    let mut b = Builder::new();
    b.set_version(1, 3);
    b.capability(Capability::Shader);
    b.capability(Capability::GroupNonUniform);
    b.capability(Capability::GroupNonUniformVote);
    b.capability(Capability::GroupNonUniformBallot);
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
    let mut bufs = [0; 2];
    for (i, v) in bufs.iter_mut().enumerate() {
        *v = global(&mut b, p_st, StorageClass::StorageBuffer);
        b.decorate(*v, Decoration::DescriptorSet, [lit(0)]);
        b.decorate(*v, Decoration::Binding, [lit(i as u32)]);
    }
    let (marks, out) = (bufs[0], bufs[1]);
    let pcv = global(&mut b, p_pc, StorageClass::PushConstant);
    let c = |b: &mut Builder, v: u32| b.constant_bit32(u32t, v);
    let (c0, c1, c2, c3, c5, c31, c32) = (c(&mut b, 0), c(&mut b, 1), c(&mut b, 2), c(&mut b, 3), c(&mut b, 5), c(&mut b, 31), c(&mut b, 32));
    let list_at = c(&mut b, MARK_LIST_AT);
    let list_first = c(&mut b, MARK_LIST_AT + 4);
    let list_cap = c(&mut b, MARK_LIST_CAP);
    let bits_at = c(&mut b, MARK_PAGE_BITS_AT);
    let rec_dw = c(&mut b, GATHER_MASK_RECORD_DWORDS);
    let hdr = c(&mut b, GATHER_HEADER_DWORDS);
    let scope_sg = c(&mut b, 3);
    let scope_dev = c(&mut b, 1);
    let sem_none = c0;
    let fty = b.type_function(void, []);
    let f = b.begin_function(void, None, FunctionControl::NONE, fty)?;
    b.begin_block(None)?;
    let l = b.load(v3, None, lid_v, None, [])?;
    let lid = b.composite_extract(u32t, None, l, [0])?;
    let w = b.load(v3, None, wid_v, None, [])?;
    let g = b.composite_extract(u32t, None, w, [0])?;
    let pc_at = |b: &mut Builder, k: Word| -> Result<Word, rspirv::dr::Error> {
        let p = b.access_chain(p_pcu, None, pcv, [k])?;
        b.load(u32t, None, p, None, [])
    };
    let pages = pc_at(&mut b, c0)?;
    let cap = pc_at(&mut b, c1)?;
    let first_page = pc_at(&mut b, c2)?;
    let flags = pc_at(&mut b, c3)?;
    let lb = b.bitwise_and(u32t, None, flags, c1)?;
    let list = b.i_not_equal(bool_, None, lb, c0)?;
    let is0 = b.i_equal(bool_, None, lid, c0)?;
    // List mode: the list's count (workgroup 0 copies it out).
    let pcount = b.access_chain(p_u, None, marks, [c0, list_at])?;
    let lcount = b.load(u32t, None, pcount, None, [])?;
    let g0 = b.i_equal(bool_, None, g, c0)?;
    let copy_count = b.logical_and(bool_, None, list, g0)?;
    let copy_count = b.logical_and(bool_, None, copy_count, is0)?;
    let (l_cc, l_ccd) = (b.id(), b.id());
    b.selection_merge(l_ccd, SelectionControl::NONE)?;
    b.branch_conditional(copy_count, l_cc, l_ccd, [])?;
    b.begin_block(Some(l_cc))?;
    let po1 = b.access_chain(p_u, None, out, [c0, c1])?;
    b.store(po1, lcount, None, [])?;
    b.branch(l_ccd)?;
    b.begin_block(Some(l_ccd))?;
    let under = b.u_less_than(bool_, None, lcount, list_cap)?;
    let lim_list = b.select(u32t, None, under, lcount, list_cap)?;
    let limit = b.select(u32t, None, list, lim_list, pages)?;
    let inside = b.u_less_than(bool_, None, g, limit)?;
    let (l_in, l_end) = (b.id(), b.id());
    b.selection_merge(l_end, SelectionControl::NONE)?;
    b.branch_conditional(inside, l_in, l_end, [])?;
    b.begin_block(Some(l_in))?;
    let li = b.i_add(u32t, None, list_first, g)?;
    let pli = b.access_chain(p_u, None, marks, [c0, li])?;
    let from_list = b.load(u32t, None, pli, None, [])?;
    let from_range = b.i_add(u32t, None, first_page, g)?;
    let page = b.select(u32t, None, list, from_list, from_range)?;
    let p32 = b.i_mul(u32t, None, page, c32)?;
    let wi = b.i_add(u32t, None, p32, lid)?;
    let pm = b.access_chain(p_u, None, marks, [c0, wi])?;
    let mw = b.load(u32t, None, pm, None, [])?;
    let nz = b.i_not_equal(bool_, None, mw, c0)?;
    let any = b.group_non_uniform_any(bool_, None, scope_sg, nz)?;
    // The page's touched bit, cleared.
    let (l_pb, l_pbd) = (b.id(), b.id());
    b.selection_merge(l_pbd, SelectionControl::NONE)?;
    b.branch_conditional(is0, l_pb, l_pbd, [])?;
    b.begin_block(Some(l_pb))?;
    let pw = b.shift_right_logical(u32t, None, page, c5)?;
    let pwi = b.i_add(u32t, None, bits_at, pw)?;
    let pbit = b.bitwise_and(u32t, None, page, c31)?;
    let pmask = b.shift_left_logical(u32t, None, c1, pbit)?;
    let pinv = b.not(u32t, None, pmask)?;
    let ppb = b.access_chain(p_u, None, marks, [c0, pwi])?;
    b.atomic_and(u32t, None, ppb, scope_dev, sem_none, pinv)?;
    b.branch(l_pbd)?;
    b.begin_block(Some(l_pbd))?;
    let (l_rec, l_recd) = (b.id(), b.id());
    b.selection_merge(l_recd, SelectionControl::NONE)?;
    b.branch_conditional(any, l_rec, l_recd, [])?;
    b.begin_block(Some(l_rec))?;
    // Lane 0 takes a record; the others learn its slot.
    let (l_t, l_td) = (b.id(), b.id());
    b.selection_merge(l_td, SelectionControl::NONE)?;
    b.branch_conditional(is0, l_t, l_td, [])?;
    b.begin_block(Some(l_t))?;
    let pcnt = b.access_chain(p_u, None, out, [c0, c0])?;
    let s_taken = b.atomic_i_add(u32t, None, pcnt, scope_dev, sem_none, c1)?;
    b.branch(l_td)?;
    b.begin_block(Some(l_td))?;
    let s_phi = b.phi(u32t, None, [(s_taken, l_t), (c0, l_rec)])?;
    let slot = b.group_non_uniform_broadcast_first(u32t, None, scope_sg, s_phi)?;
    let room = b.u_less_than(bool_, None, slot, cap)?;
    let (l_r, l_rd) = (b.id(), b.id());
    b.selection_merge(l_rd, SelectionControl::NONE)?;
    b.branch_conditional(room, l_r, l_rd, [])?;
    b.begin_block(Some(l_r))?;
    let sr = b.i_mul(u32t, None, slot, rec_dw)?;
    let rec = b.i_add(u32t, None, hdr, sr)?;
    let (l_h, l_hd) = (b.id(), b.id());
    b.selection_merge(l_hd, SelectionControl::NONE)?;
    b.branch_conditional(is0, l_h, l_hd, [])?;
    b.begin_block(Some(l_h))?;
    let ph = b.access_chain(p_u, None, out, [c0, rec])?;
    b.store(ph, page, None, [])?;
    b.branch(l_hd)?;
    b.begin_block(Some(l_hd))?;
    let r1 = b.i_add(u32t, None, rec, c1)?;
    let ri = b.i_add(u32t, None, r1, lid)?;
    let pr = b.access_chain(p_u, None, out, [c0, ri])?;
    b.store(pr, mw, None, [])?;
    let inv = b.not(u32t, None, mw)?;
    let pm2 = b.access_chain(p_u, None, marks, [c0, wi])?;
    b.atomic_and(u32t, None, pm2, scope_dev, sem_none, inv)?;
    b.branch(l_rd)?;
    b.begin_block(Some(l_rd))?;
    b.branch(l_recd)?;
    b.begin_block(Some(l_recd))?;
    b.branch(l_end)?;
    b.begin_block(Some(l_end))?;
    b.ret()?;
    b.end_function()?;
    b.entry_point(ExecutionModel::GLCompute, f, "main", [lid_v, wid_v]);
    b.execution_mode(f, ExecutionMode::LocalSize, [GATHER_LOCAL_SIZE, 1, 1]);
    Ok(b.module().assemble())
}

/// Counters of `mirror_scatter` (binding 1): pages written back, dwords,
/// pages with marks but no guest page-table entry (their dwords are lost to
/// guest memory), scatters that found the touched-page list past
/// MARK_LIST_CAP.
pub const SCATTER_COUNTERS: u32 = 4;

/// Workgroup size of `mirror_scatter`: invocation t copies dwords t, t +
/// 256, t + 512 and t + 768 of its page, so that a subgroup's stores are
/// consecutive dwords (one transfer across PCIe rather than one per dword).
pub const SCATTER_LOCAL_SIZE: u32 = 256;

/// Front end 2's write-back (docs/FRONTEND2.md "Coherence"): workgroup g
/// takes the g-th page of the touched-page list (MARK_LIST_AT; past its
/// count or MARK_LIST_CAP it does nothing) and stores every dword its
/// written-bit map marks from the mirror into guest memory, then clears the
/// page's mask words and its touched bit. The mirror's span starts at
/// device address `mirror`; guest memory is where the guest page table
/// (`table`: one u64 device address per 4 KiB page of the span, 0 where
/// nothing is mapped) puts the page. Binding 0 is the whole written-bit
/// map, binding 1 the SCATTER_COUNTERS counters. Push constants `{mirror
/// (low, high), table (low, high)}`.
pub fn mirror_scatter() -> Vec<u32> {
    build_scatter(false, true).expect("mirror scatter kernel")
}

/// `mirror_scatter` emptying the touched-page list itself: workgroup 0's
/// first invocation takes the list's count and stores 0 there (an atomic
/// exchange; the indirect dispatch read the count already), and the other
/// workgroups never read it (they are bounded by MARK_LIST_CAP alone: the
/// dispatch made as many as the count), so the write-back needs no fill of
/// the count and no barrier before one.
pub fn mirror_scatter_reset() -> Vec<u32> {
    build_scatter(true, true).expect("mirror scatter kernel (emptying the list)")
}

/// `mirror_scatter_reset` storing nothing into guest memory (its marks
/// cleared and counted as before): timing only, the upper bound of a lazy
/// write-back.
pub fn mirror_scatter_nostore() -> Vec<u32> {
    build_scatter(true, false).expect("mirror scatter kernel (no stores)")
}

fn build_scatter(reset: bool, stores: bool) -> Result<Vec<u32>, rspirv::dr::Error> {
    let lit = Operand::LiteralBit32;
    let mut b = Builder::new();
    b.set_version(1, 3);
    b.capability(Capability::Shader);
    b.capability(Capability::Int64);
    b.capability(Capability::PhysicalStorageBufferAddresses);
    b.extension("SPV_KHR_storage_buffer_storage_class");
    b.extension("SPV_KHR_physical_storage_buffer");
    b.memory_model(spirv::AddressingModel::PhysicalStorageBuffer64, spirv::MemoryModel::GLSL450);
    let void = b.type_void();
    let bool_ = b.type_bool();
    let u32t = b.type_int(32, 0);
    let u64t = b.type_int(64, 0);
    let v3 = b.type_vector(u32t, 3);
    let rta = b.type_runtime_array(u32t);
    b.decorate(rta, Decoration::ArrayStride, [lit(4)]);
    let st = b.type_struct([rta]);
    b.decorate(st, Decoration::Block, []);
    b.member_decorate(st, 0, Decoration::Offset, [lit(0)]);
    let p_st = b.type_pointer(None, StorageClass::StorageBuffer, st);
    let p_u = b.type_pointer(None, StorageClass::StorageBuffer, u32t);
    let p_dev = b.type_pointer(None, StorageClass::PhysicalStorageBuffer, u32t);
    let p_dev64 = b.type_pointer(None, StorageClass::PhysicalStorageBuffer, u64t);
    let pc = b.type_struct([u32t, u32t, u32t, u32t]);
    b.decorate(pc, Decoration::Block, []);
    for k in 0..4 {
        b.member_decorate(pc, k, Decoration::Offset, [lit(4 * k)]);
    }
    let p_pc = b.type_pointer(None, StorageClass::PushConstant, pc);
    let p_pcu = b.type_pointer(None, StorageClass::PushConstant, u32t);
    let p_in = b.type_pointer(None, StorageClass::Input, v3);
    let lid_v = global(&mut b, p_in, StorageClass::Input);
    b.decorate(lid_v, Decoration::BuiltIn, [Operand::BuiltIn(BuiltIn::LocalInvocationId)]);
    let wid_v = global(&mut b, p_in, StorageClass::Input);
    b.decorate(wid_v, Decoration::BuiltIn, [Operand::BuiltIn(BuiltIn::WorkgroupId)]);
    let mut bufs = [0; 2];
    for (i, v) in bufs.iter_mut().enumerate() {
        *v = global(&mut b, p_st, StorageClass::StorageBuffer);
        b.decorate(*v, Decoration::DescriptorSet, [lit(0)]);
        b.decorate(*v, Decoration::Binding, [lit(i as u32)]);
    }
    let (marks, counters) = (bufs[0], bufs[1]);
    let pcv = global(&mut b, p_pc, StorageClass::PushConstant);
    let c = |b: &mut Builder, v: u32| b.constant_bit32(u32t, v);
    let (c0, c1, c2, c3, c5, c31, c32) = (c(&mut b, 0), c(&mut b, 1), c(&mut b, 2), c(&mut b, 3), c(&mut b, 5), c(&mut b, 31), c(&mut b, 32));
    let c32_64 = b.constant_bit64(u64t, 32);
    let c0_64 = b.constant_bit64(u64t, 0);
    let c8_64 = b.constant_bit64(u64t, 8);
    let c4096_64 = b.constant_bit64(u64t, 4096);
    let list_at = c(&mut b, MARK_LIST_AT);
    let list_first = c(&mut b, MARK_LIST_AT + 4);
    let list_cap = c(&mut b, MARK_LIST_CAP);
    let bits_at = c(&mut b, MARK_PAGE_BITS_AT);
    let scope_dev = c(&mut b, 1);
    let scope_wg = c(&mut b, 2);
    // AcquireRelease | UniformMemory: the mask words read before any is cleared.
    let sem_wg = c(&mut b, 0x8 | 0x40);
    let sem_none = c0;
    let fty = b.type_function(void, []);
    let f = b.begin_function(void, None, FunctionControl::NONE, fty)?;
    b.begin_block(None)?;
    let l = b.load(v3, None, lid_v, None, [])?;
    let lid = b.composite_extract(u32t, None, l, [0])?;
    let w = b.load(v3, None, wid_v, None, [])?;
    let g = b.composite_extract(u32t, None, w, [0])?;
    let pc_at = |b: &mut Builder, k: Word| -> Result<Word, rspirv::dr::Error> {
        let p = b.access_chain(p_pcu, None, pcv, [k])?;
        b.load(u32t, None, p, None, [])
    };
    let addr64 = |b: &mut Builder, lo: Word, hi: Word| -> Result<Word, rspirv::dr::Error> {
        let lo64 = b.u_convert(u64t, None, lo)?;
        let hi64 = b.u_convert(u64t, None, hi)?;
        let hi64 = b.shift_left_logical(u64t, None, hi64, c32_64)?;
        b.bitwise_or(u64t, None, lo64, hi64)
    };
    let mlo = pc_at(&mut b, c0)?;
    let mhi = pc_at(&mut b, c1)?;
    let tlo = pc_at(&mut b, c2)?;
    let thi = pc_at(&mut b, c3)?;
    let mirror = addr64(&mut b, mlo, mhi)?;
    let table = addr64(&mut b, tlo, thi)?;
    let is0 = b.i_equal(bool_, None, lid, c0)?;
    let pcount = b.access_chain(p_u, None, marks, [c0, list_at])?;
    let inside = if reset {
        // Workgroup 0's first invocation takes the count, empties the list
        // and counts a list past its capacity.
        let g0 = b.i_equal(bool_, None, g, c0)?;
        let first = b.logical_and(bool_, None, g0, is0)?;
        let (l_rs, l_rsd) = (b.id(), b.id());
        b.selection_merge(l_rsd, SelectionControl::NONE)?;
        b.branch_conditional(first, l_rs, l_rsd, [])?;
        b.begin_block(Some(l_rs))?;
        let lcount = b.atomic_exchange(u32t, None, pcount, scope_dev, sem_none, c0)?;
        let over = b.u_greater_than(bool_, None, lcount, list_cap)?;
        let (l_ov, l_ovd) = (b.id(), b.id());
        b.selection_merge(l_ovd, SelectionControl::NONE)?;
        b.branch_conditional(over, l_ov, l_ovd, [])?;
        b.begin_block(Some(l_ov))?;
        let pov = b.access_chain(p_u, None, counters, [c0, c3])?;
        b.atomic_i_add(u32t, None, pov, scope_dev, sem_none, c1)?;
        b.branch(l_ovd)?;
        b.begin_block(Some(l_ovd))?;
        b.branch(l_rsd)?;
        b.begin_block(Some(l_rsd))?;
        // (Uniform in the workgroup: the barrier below is inside it.)
        b.u_less_than(bool_, None, g, list_cap)?
    } else {
        let lcount = b.load(u32t, None, pcount, None, [])?;
        // Workgroup 0 counts a list past its capacity.
        let g0 = b.i_equal(bool_, None, g, c0)?;
        let over = b.u_greater_than(bool_, None, lcount, list_cap)?;
        let note_over = b.logical_and(bool_, None, g0, is0)?;
        let note_over = b.logical_and(bool_, None, note_over, over)?;
        let (l_ov, l_ovd) = (b.id(), b.id());
        b.selection_merge(l_ovd, SelectionControl::NONE)?;
        b.branch_conditional(note_over, l_ov, l_ovd, [])?;
        b.begin_block(Some(l_ov))?;
        let pov = b.access_chain(p_u, None, counters, [c0, c3])?;
        b.atomic_i_add(u32t, None, pov, scope_dev, sem_none, c1)?;
        b.branch(l_ovd)?;
        b.begin_block(Some(l_ovd))?;
        let under = b.u_less_than(bool_, None, lcount, list_cap)?;
        let limit = b.select(u32t, None, under, lcount, list_cap)?;
        // (Uniform in the workgroup: the barrier below is inside it.)
        b.u_less_than(bool_, None, g, limit)?
    };
    let (l_in, l_end) = (b.id(), b.id());
    b.selection_merge(l_end, SelectionControl::NONE)?;
    b.branch_conditional(inside, l_in, l_end, [])?;
    b.begin_block(Some(l_in))?;
    let li = b.i_add(u32t, None, list_first, g)?;
    let pli = b.access_chain(p_u, None, marks, [c0, li])?;
    let page = b.load(u32t, None, pli, None, [])?;
    let first_word = b.i_mul(u32t, None, page, c32)?;
    // The page's entry in the guest page table, its first byte in the mirror.
    let page64 = b.u_convert(u64t, None, page)?;
    let toff = b.i_mul(u64t, None, page64, c8_64)?;
    let ta = b.i_add(u64t, None, table, toff)?;
    let tp = b.convert_u_to_ptr(p_dev64, None, ta)?;
    let dest = b.load(u64t, None, tp, Some(spirv::MemoryAccess::ALIGNED), [lit(8)])?;
    let mapped = b.i_not_equal(bool_, None, dest, c0_64)?;
    let moff = b.i_mul(u64t, None, page64, c4096_64)?;
    let msrc = b.i_add(u64t, None, mirror, moff)?;
    let bit_of = b.bitwise_and(u32t, None, lid, c31)?;
    // Dwords lid + 256 r, r = 0..3: marked ones copied.
    for r in 0..4u32 {
        let cr = b.constant_bit32(u32t, r * SCATTER_LOCAL_SIZE);
        let d = b.i_add(u32t, None, lid, cr)?;
        let wd = b.shift_right_logical(u32t, None, d, c5)?;
        let wi = b.i_add(u32t, None, first_word, wd)?;
        let pm = b.access_chain(p_u, None, marks, [c0, wi])?;
        let mw = b.load(u32t, None, pm, None, [])?;
        let sh = b.shift_right_logical(u32t, None, mw, bit_of)?;
        let bit = b.bitwise_and(u32t, None, sh, c1)?;
        let set = b.i_not_equal(bool_, None, bit, c0)?;
        let go = if stores { b.logical_and(bool_, None, set, mapped)? } else { b.constant_false(bool_) };
        let (l_cp, l_cpd) = (b.id(), b.id());
        b.selection_merge(l_cpd, SelectionControl::NONE)?;
        b.branch_conditional(go, l_cp, l_cpd, [])?;
        b.begin_block(Some(l_cp))?;
        let d4 = b.shift_left_logical(u32t, None, d, c2)?;
        let d64 = b.u_convert(u64t, None, d4)?;
        let sa = b.i_add(u64t, None, msrc, d64)?;
        let sp = b.convert_u_to_ptr(p_dev, None, sa)?;
        let v = b.load(u32t, None, sp, Some(spirv::MemoryAccess::ALIGNED), [lit(4)])?;
        let da = b.i_add(u64t, None, dest, d64)?;
        let dp = b.convert_u_to_ptr(p_dev, None, da)?;
        b.store(dp, v, Some(spirv::MemoryAccess::ALIGNED), [lit(4)])?;
        b.branch(l_cpd)?;
        b.begin_block(Some(l_cpd))?;
    }
    // Invocation 0 counts the page's marked dwords (before they are cleared).
    let (l_cnt, l_cntd) = (b.id(), b.id());
    b.selection_merge(l_cntd, SelectionControl::NONE)?;
    b.branch_conditional(is0, l_cnt, l_cntd, [])?;
    b.begin_block(Some(l_cnt))?;
    let mut total = c0;
    for k in 0..32u32 {
        let ck = b.constant_bit32(u32t, k);
        let wi = b.i_add(u32t, None, first_word, ck)?;
        let pm = b.access_chain(p_u, None, marks, [c0, wi])?;
        let mw = b.load(u32t, None, pm, None, [])?;
        let n = b.bit_count(u32t, None, mw)?;
        total = b.i_add(u32t, None, total, n)?;
    }
    let which_dw = b.select(u32t, None, mapped, c1, c0)?;
    let pdw = b.access_chain(p_u, None, counters, [c0, c1])?;
    let add = b.i_mul(u32t, None, total, which_dw)?;
    b.atomic_i_add(u32t, None, pdw, scope_dev, sem_none, add)?;
    b.branch(l_cntd)?;
    b.begin_block(Some(l_cntd))?;
    // Every mask word read before any is cleared.
    b.control_barrier(scope_wg, scope_wg, sem_wg)?;
    let first32 = b.u_less_than(bool_, None, lid, c32)?;
    let (l_clr, l_clrd) = (b.id(), b.id());
    b.selection_merge(l_clrd, SelectionControl::NONE)?;
    b.branch_conditional(first32, l_clr, l_clrd, [])?;
    b.begin_block(Some(l_clr))?;
    let wi = b.i_add(u32t, None, first_word, lid)?;
    let pm = b.access_chain(p_u, None, marks, [c0, wi])?;
    b.store(pm, c0, None, [])?;
    b.branch(l_clrd)?;
    b.begin_block(Some(l_clrd))?;
    // Invocation 0: the page's touched bit cleared, the page counted.
    let (l_pb, l_pbd) = (b.id(), b.id());
    b.selection_merge(l_pbd, SelectionControl::NONE)?;
    b.branch_conditional(is0, l_pb, l_pbd, [])?;
    b.begin_block(Some(l_pb))?;
    let pw = b.shift_right_logical(u32t, None, page, c5)?;
    let pwi = b.i_add(u32t, None, bits_at, pw)?;
    let pbit = b.bitwise_and(u32t, None, page, c31)?;
    let pmask = b.shift_left_logical(u32t, None, c1, pbit)?;
    let pinv = b.not(u32t, None, pmask)?;
    let ppb = b.access_chain(p_u, None, marks, [c0, pwi])?;
    b.atomic_and(u32t, None, ppb, scope_dev, sem_none, pinv)?;
    let which = b.select(u32t, None, mapped, c0, c2)?;
    let pcnt = b.access_chain(p_u, None, counters, [c0, which])?;
    b.atomic_i_add(u32t, None, pcnt, scope_dev, sem_none, c1)?;
    b.branch(l_pbd)?;
    b.begin_block(Some(l_pbd))?;
    b.branch(l_end)?;
    b.begin_block(Some(l_end))?;
    b.ret()?;
    b.end_function()?;
    b.entry_point(ExecutionModel::GLCompute, f, "main", [lid_v, wid_v]);
    b.execution_mode(f, ExecutionMode::LocalSize, [SCATTER_LOCAL_SIZE, 1, 1]);
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
    fn scatter_kernel_assembles() {
        let w = super::mirror_scatter();
        assert_eq!(w[0], 0x0723_0203);
        if let Ok(dir) = std::env::var("RDNA_KERNEL_DUMP") {
            let bytes: Vec<u8> = w.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(std::path::Path::new(&dir).join("mirror_scatter.spv"), bytes).unwrap();
        }
    }

    #[test]
    fn scatter_reset_kernel_assembles() {
        let w = super::mirror_scatter_reset();
        assert_eq!(w[0], 0x0723_0203);
        assert_ne!(w, super::mirror_scatter());
        if let Ok(dir) = std::env::var("RDNA_KERNEL_DUMP") {
            let bytes: Vec<u8> = w.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(std::path::Path::new(&dir).join("mirror_scatter_reset.spv"), bytes).unwrap();
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
    #[test]
    fn merge_many_kernel_assembles() {
        let w = super::shadow_merge_many();
        assert_eq!(w[0], 0x0723_0203);
        if let Ok(dir) = std::env::var("RDNA_KERNEL_DUMP") {
            let bytes: Vec<u8> = w.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(std::path::Path::new(&dir).join("shadow_merge_many.spv"), bytes).unwrap();
        }
    }
    #[test]
    fn resident_merge_kernel_assembles() {
        let w = super::resident_merge();
        assert_eq!(w[0], 0x0723_0203);
        if let Ok(dir) = std::env::var("RDNA_KERNEL_DUMP") {
            let bytes: Vec<u8> = w.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(std::path::Path::new(&dir).join("resident_merge.spv"), bytes).unwrap();
        }
    }
}

/// Workgroup size of `latency_test`.
pub const LATENCY_LOCAL_SIZE: u32 = 64;

/// A test kernel (`dsport --dispatch-bench`): each invocation runs `iters`
/// (push constant 0) steps of a dependent integer chain seeded by its index
/// and stores the result at dword `base + index` of binding 0 (push constant
/// 1). `salt` is baked into the code: kernels with different salts are
/// different programs (and pipelines).
pub fn latency_test(salt: u32) -> Vec<u32> {
    build_latency(salt).expect("latency test kernel")
}

fn build_latency(salt: u32) -> Result<Vec<u32>, rspirv::dr::Error> {
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
    let buf = global(&mut b, p_st, StorageClass::StorageBuffer);
    b.decorate(buf, Decoration::DescriptorSet, [lit(0)]);
    b.decorate(buf, Decoration::Binding, [lit(0)]);
    let pcv = global(&mut b, p_pc, StorageClass::PushConstant);
    let c0 = b.constant_bit32(u32t, 0);
    let c1 = b.constant_bit32(u32t, 1);
    let cm = b.constant_bit32(u32t, 1664525);
    let ca = b.constant_bit32(u32t, 1013904223u32.wrapping_add(salt.wrapping_mul(2)));
    let fty = b.type_function(void, []);
    let f = b.begin_function(void, None, FunctionControl::NONE, fty)?;
    let l_entry = b.begin_block(None)?;
    let g = b.load(v3, None, gid, None, [])?;
    let x0 = b.composite_extract(u32t, None, g, [0])?;
    let p_it = b.access_chain(p_pcu, None, pcv, [c0])?;
    let iters = b.load(u32t, None, p_it, None, [])?;
    let p_base = b.access_chain(p_pcu, None, pcv, [c1])?;
    let base = b.load(u32t, None, p_base, None, [])?;
    let (l_head, l_body, l_cont, l_merge) = (b.id(), b.id(), b.id(), b.id());
    let (x_next, i_next) = (b.id(), b.id());
    b.branch(l_head)?;
    b.begin_block(Some(l_head))?;
    let i = b.phi(u32t, None, [(c0, l_entry), (i_next, l_cont)])?;
    let x = b.phi(u32t, None, [(x0, l_entry), (x_next, l_cont)])?;
    b.loop_merge(l_merge, l_cont, spirv::LoopControl::NONE, [])?;
    let more = b.u_less_than(bool_, None, i, iters)?;
    b.branch_conditional(more, l_body, l_merge, [])?;
    b.begin_block(Some(l_body))?;
    let m = b.i_mul(u32t, None, x, cm)?;
    b.i_add(u32t, Some(x_next), m, ca)?;
    b.branch(l_cont)?;
    b.begin_block(Some(l_cont))?;
    b.i_add(u32t, Some(i_next), i, c1)?;
    b.branch(l_head)?;
    b.begin_block(Some(l_merge))?;
    let idx = b.i_add(u32t, None, base, x0)?;
    let p = b.access_chain(p_u, None, buf, [c0, idx])?;
    b.store(p, x, None, [])?;
    b.ret()?;
    b.end_function()?;
    b.entry_point(ExecutionModel::GLCompute, f, "main", [gid]);
    b.execution_mode(f, ExecutionMode::LocalSize, [LATENCY_LOCAL_SIZE, 1, 1]);
    Ok(b.module().assemble())
}
