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
/// Then one bit per page set by the CPU for diagnosis (the runtime's
/// `DS_INDEX_CENSUS`: pages draws read indices from), read by
/// `mirror_scatter_census`.
pub const MARK_AUX_AT: u32 = MARK_LIST_AT + 4 + MARK_LIST_CAP;
pub const MARK_AUX_WORDS: u32 = (MARK_SPAN_BYTES / 4096 / 32) as u32;
pub const MARK_TOTAL_WORDS: u32 = MARK_AUX_AT + MARK_AUX_WORDS;

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

/// Workgroup size of `mirror_compact`: one invocation per word of touched
/// bits (32 pages).
pub const COMPACT_LOCAL_SIZE: u32 = 256;

/// Front end 2's touched-page list made from the touched bits
/// (`Spec::touch_bits`: stores set their pages' bits without reading them
/// back, so no store waits on an atomic's result): invocation i of the
/// dispatch takes touched-bit word `w0 + i` (below `w1`; push constants
/// `{w0, w1}`), and when any of its 32 pages is touched it reserves room in
/// the list with one atomic and enters them (past MARK_LIST_CAP they are
/// counted only: the scatter counts the overflow, and their bits stay set
/// for the next write-back). The bits are left for the scatter, which
/// clears each page's as it writes the page back. Binding 0 is the whole
/// written-bit map (binding 1 of the scatter's set is not used).
pub fn mirror_compact() -> Vec<u32> {
    build_compact().expect("mirror compact kernel")
}

fn build_compact() -> Result<Vec<u32>, rspirv::dr::Error> {
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
    for k in 0..4 {
        b.member_decorate(pc, k, Decoration::Offset, [lit(4 * k)]);
    }
    let p_pc = b.type_pointer(None, StorageClass::PushConstant, pc);
    let p_pcu = b.type_pointer(None, StorageClass::PushConstant, u32t);
    let p_in = b.type_pointer(None, StorageClass::Input, v3);
    let gid_v = global(&mut b, p_in, StorageClass::Input);
    b.decorate(gid_v, Decoration::BuiltIn, [Operand::BuiltIn(BuiltIn::GlobalInvocationId)]);
    let mut bufs = [0; 2];
    for (i, v) in bufs.iter_mut().enumerate() {
        *v = global(&mut b, p_st, StorageClass::StorageBuffer);
        b.decorate(*v, Decoration::DescriptorSet, [lit(0)]);
        b.decorate(*v, Decoration::Binding, [lit(i as u32)]);
    }
    let marks = bufs[0];
    let pcv = global(&mut b, p_pc, StorageClass::PushConstant);
    let c = |b: &mut Builder, v: u32| b.constant_bit32(u32t, v);
    let (c0, c1, c32) = (c(&mut b, 0), c(&mut b, 1), c(&mut b, 32));
    let list_at = c(&mut b, MARK_LIST_AT);
    let list_first = c(&mut b, MARK_LIST_AT + 4);
    let list_cap = c(&mut b, MARK_LIST_CAP);
    let bits_at = c(&mut b, MARK_PAGE_BITS_AT);
    let scope_dev = c(&mut b, 1);
    let sem_none = c0;
    let fty = b.type_function(void, []);
    let f = b.begin_function(void, None, FunctionControl::NONE, fty)?;
    b.begin_block(None)?;
    let gv = b.load(v3, None, gid_v, None, [])?;
    let gid = b.composite_extract(u32t, None, gv, [0])?;
    let pc_at = |b: &mut Builder, k: Word| -> Result<Word, rspirv::dr::Error> {
        let p = b.access_chain(p_pcu, None, pcv, [k])?;
        b.load(u32t, None, p, None, [])
    };
    let w0 = pc_at(&mut b, c0)?;
    let w1 = pc_at(&mut b, c1)?;
    let wi = b.i_add(u32t, None, w0, gid)?;
    let inside = b.u_less_than(bool_, None, wi, w1)?;
    let (l_in, l_end) = (b.id(), b.id());
    b.selection_merge(l_end, SelectionControl::NONE)?;
    b.branch_conditional(inside, l_in, l_end, [])?;
    b.begin_block(Some(l_in))?;
    let bw = b.i_add(u32t, None, bits_at, wi)?;
    let pb = b.access_chain(p_u, None, marks, [c0, bw])?;
    let bits = b.load(u32t, None, pb, None, [])?;
    let any = b.i_not_equal(bool_, None, bits, c0)?;
    let (l_any, l_anyd) = (b.id(), b.id());
    b.selection_merge(l_anyd, SelectionControl::NONE)?;
    b.branch_conditional(any, l_any, l_anyd, [])?;
    b.begin_block(Some(l_any))?;
    let n = b.bit_count(u32t, None, bits)?;
    let pcount = b.access_chain(p_u, None, marks, [c0, list_at])?;
    let base = b.atomic_i_add(u32t, None, pcount, scope_dev, sem_none, n)?;
    let page0 = b.i_mul(u32t, None, wi, c32)?;
    for k in 0..32u32 {
        let ck = c(&mut b, k);
        let sh = b.shift_right_logical(u32t, None, bits, ck)?;
        let bk = b.bitwise_and(u32t, None, sh, c1)?;
        let set = b.i_not_equal(bool_, None, bk, c0)?;
        let below_mask = c(&mut b, if k == 0 { 0 } else { u32::MAX >> (32 - k) });
        let below_bits = b.bitwise_and(u32t, None, bits, below_mask)?;
        let below = b.bit_count(u32t, None, below_bits)?;
        let idx = b.i_add(u32t, None, base, below)?;
        let room = b.u_less_than(bool_, None, idx, list_cap)?;
        let go = b.logical_and(bool_, None, set, room)?;
        let (l_st, l_std) = (b.id(), b.id());
        b.selection_merge(l_std, SelectionControl::NONE)?;
        b.branch_conditional(go, l_st, l_std, [])?;
        b.begin_block(Some(l_st))?;
        let li = b.i_add(u32t, None, list_first, idx)?;
        let pl = b.access_chain(p_u, None, marks, [c0, li])?;
        let page = b.i_add(u32t, None, page0, ck)?;
        b.store(pl, page, None, [])?;
        b.branch(l_std)?;
        b.begin_block(Some(l_std))?;
    }
    b.branch(l_anyd)?;
    b.begin_block(Some(l_anyd))?;
    b.branch(l_end)?;
    b.begin_block(Some(l_end))?;
    b.ret()?;
    b.end_function()?;
    b.entry_point(ExecutionModel::GLCompute, f, "main", [gid_v]);
    b.execution_mode(f, ExecutionMode::LocalSize, [COMPACT_LOCAL_SIZE, 1, 1]);
    Ok(b.module().assemble())
}

/// Counters of `mirror_scatter` (binding 1): pages written back, dwords,
/// pages with marks but no guest page-table entry (their dwords are lost to
/// guest memory), scatters that found the touched-page list past
/// MARK_LIST_CAP; then `mirror_scatter_census`'s dwords written back and
/// those of them guest memory held already, and the dwords and pages it
/// wrote back in pages whose bit is set in the map's MARK_AUX_AT region.
pub const SCATTER_COUNTERS: u32 = 8;

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

/// `mirror_scatter_reset` that reads each marked dword's guest-memory value
/// before storing it (across PCIe: diagnosis) and counts the dwords and
/// those guest memory held already (counters 4 and 5): what a write-back
/// of changed dwords alone would store.
pub fn mirror_scatter_census() -> Vec<u32> {
    build_scatter_ex(true, true, 1, false, false, 0, false).expect("mirror scatter kernel (census)")
}

/// `mirror_scatter_census` storing only the dwords whose guest-memory value
/// differs from the mirror's (exact: a store left out would have written
/// the value guest memory held when it was read; a CPU write racing it
/// lands as if after the GPU's).
pub fn mirror_scatter_changed() -> Vec<u32> {
    build_scatter_ex(true, true, 2, false, false, 0, false).expect("mirror scatter kernel (changed dwords)")
}

fn build_scatter(reset: bool, stores: bool) -> Result<Vec<u32>, rspirv::dr::Error> {
    build_scatter_ex(reset, stores, 0, false, false, 0, false)
}

/// `mirror_scatter_reset` that also sets, for each page it writes back
/// dwords of, the page's bit in the map's aux region (diagnosis, the
/// runtime's `DS_WB_PAGES`: which pages GPU work really wrote).
pub fn mirror_scatter_note() -> Vec<u32> {
    build_scatter_ex(true, true, 0, false, true, 0, false).expect("mirror scatter kernel (noting pages)")
}

/// `mirror_scatter_reset` storing only the marked dwords of sectors (8
/// dwords) that differ from a video-memory image of guest memory (the
/// runtime's `DS_WB_IMAGE`): the image is a second buffer over the mirror's
/// span at device address `image` (push constants 4 and 5), valid for a
/// page while its bit in the map's aux region is set. A valid page's marked
/// dwords are compared with the image, and the marked dwords of every
/// sector holding one that differs go to guest memory and the image, those
/// of the other sectors nowhere (whole sectors: stores into host memory
/// keep their runs); an invalid page's go to guest memory, then the whole
/// page is copied from the mirror into the image and its bit set. Counters
/// 6 and 7: dwords left out as unchanged, pages copied into the image.
/// Guest memory holds what the image holds as long as every other writer of
/// the page keeps the image current (uploads into the mirror, CP writes,
/// fills: the runtime).
pub fn mirror_scatter_image() -> Vec<u32> {
    build_scatter_ex(true, true, 0, false, false, 1, false).expect("mirror scatter kernel (image)")
}

/// `mirror_scatter_image` that stores every marked dword as
/// `mirror_scatter_reset` does, and for each it would leave out reads guest
/// memory first (across PCIe: diagnosis) and counts them and those that
/// differ there (counters 4 and 5; the latter must stay 0): the image's
/// exactness checked without leaving anything out.
pub fn mirror_scatter_image_check() -> Vec<u32> {
    build_scatter_ex(true, true, 0, false, false, 2, false).expect("mirror scatter kernel (image, checked)")
}

/// `mirror_scatter_reset` leaving pages whose bit is set in the map's aux
/// region out of guest memory (diagnosis, the runtime's `DS_FE2_LAZY2`:
/// their marks and touched bits are cleared as after a write-back, their
/// dwords and pages counted in counters 6 and 7 instead of 1 and 0).
pub fn mirror_scatter_lazy() -> Vec<u32> {
    build_scatter_ex(true, true, 0, true, false, 0, false).expect("mirror scatter kernel (lazy pages)")
}

/// Dwords of a handoff region (`mirror_handoff`, `mirror_scatter_region`):
/// a header `[count, 1, 1, 0]` (the region scatter's indirect dispatch), then
/// MARK_LIST_CAP page indices from REGION_PAGES_AT, then 32 mask words for
/// each from REGION_MASKS_AT.
pub const REGION_PAGES_AT: u32 = 4;
pub const REGION_MASKS_AT: u32 = REGION_PAGES_AT + MARK_LIST_CAP;
pub const REGION_WORDS: u64 = REGION_MASKS_AT as u64 + 32 * MARK_LIST_CAP as u64;

/// Workgroup size of `mirror_handoff`: an invocation a mask word of its page.
pub const HANDOFF_LOCAL_SIZE: u32 = 32;

/// Front end 2's write-back handed over (the runtime's `tune wb_region`):
/// workgroup g takes the g-th page of the touched-page list as
/// `mirror_scatter_reset` does, copies its 32 mask words and the page's
/// index into entry g of the handoff region at device address `region`
/// (push constants 6 and 7; bindings and the other push constants as the
/// write-back's) and clears them and the page's touched bit; workgroup 0's
/// first invocation empties the list and stores its count (at most
/// MARK_LIST_CAP; a list past it counted in counter 3) into the region's
/// header. `mirror_scatter_region` then stores the marked dwords.
pub fn mirror_handoff() -> Vec<u32> {
    build_handoff().expect("mirror handoff kernel")
}

/// `mirror_scatter_reset` taking its pages and mask words from a handoff
/// region (`mirror_handoff`; indirect dispatch on the region's header) and
/// clearing nothing: the mirror's marked dwords as they are when it runs go
/// to guest memory.
pub fn mirror_scatter_region() -> Vec<u32> {
    build_scatter_ex(true, true, 0, false, false, 0, true).expect("mirror scatter kernel (handoff region)")
}

fn build_handoff() -> Result<Vec<u32>, rspirv::dr::Error> {
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
    let pc = b.type_struct(vec![u32t; 8]);
    b.decorate(pc, Decoration::Block, []);
    for k in 0..8u32 {
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
    let (c0, c1, c3, c5, c6, c7, c31, c32) = (c(&mut b, 0), c(&mut b, 1), c(&mut b, 3), c(&mut b, 5), c(&mut b, 6), c(&mut b, 7), c(&mut b, 31), c(&mut b, 32));
    let c2_64 = b.constant_bit64(u64t, 2);
    let c32_64 = b.constant_bit64(u64t, 32);
    let list_at = c(&mut b, MARK_LIST_AT);
    let list_first = c(&mut b, MARK_LIST_AT + 4);
    let list_cap = c(&mut b, MARK_LIST_CAP);
    let bits_at = c(&mut b, MARK_PAGE_BITS_AT);
    let pages_at = c(&mut b, REGION_PAGES_AT);
    let masks_at = c(&mut b, REGION_MASKS_AT);
    let scope_dev = c1;
    let sem_none = c0;
    let fty = b.type_function(void, []);
    let f = b.begin_function(void, None, FunctionControl::NONE, fty)?;
    b.begin_block(None)?;
    let l = b.load(v3, None, lid_v, None, [])?;
    let lid = b.composite_extract(u32t, None, l, [0])?;
    let w = b.load(v3, None, wid_v, None, [])?;
    let g = b.composite_extract(u32t, None, w, [0])?;
    let p6 = b.access_chain(p_pcu, None, pcv, [c6])?;
    let rlo = b.load(u32t, None, p6, None, [])?;
    let p7 = b.access_chain(p_pcu, None, pcv, [c7])?;
    let rhi = b.load(u32t, None, p7, None, [])?;
    let rlo64 = b.u_convert(u64t, None, rlo)?;
    let rhi64 = b.u_convert(u64t, None, rhi)?;
    let rhi64 = b.shift_left_logical(u64t, None, rhi64, c32_64)?;
    let region = b.bitwise_or(u64t, None, rlo64, rhi64)?;
    let region_word = |b: &mut Builder, at: Word| -> Result<Word, rspirv::dr::Error> {
        let at64 = b.u_convert(u64t, None, at)?;
        let off = b.shift_left_logical(u64t, None, at64, c2_64)?;
        let a = b.i_add(u64t, None, region, off)?;
        b.convert_u_to_ptr(p_dev, None, a)
    };
    let is0 = b.i_equal(bool_, None, lid, c0)?;
    // Workgroup 0's first invocation: the list's count taken (the indirect
    // dispatch read it already), the list emptied, the region's header.
    let g0 = b.i_equal(bool_, None, g, c0)?;
    let first = b.logical_and(bool_, None, g0, is0)?;
    let (l_h, l_hd) = (b.id(), b.id());
    b.selection_merge(l_hd, SelectionControl::NONE)?;
    b.branch_conditional(first, l_h, l_hd, [])?;
    b.begin_block(Some(l_h))?;
    let pcount = b.access_chain(p_u, None, marks, [c0, list_at])?;
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
    let n = b.select(u32t, None, over, list_cap, lcount)?;
    for (k, v) in [n, c1, c1, c0].into_iter().enumerate() {
        let ck = c(&mut b, k as u32);
        let hp = region_word(&mut b, ck)?;
        b.store(hp, v, Some(spirv::MemoryAccess::ALIGNED), [lit(4)])?;
    }
    b.branch(l_hd)?;
    b.begin_block(Some(l_hd))?;
    let inside = b.u_less_than(bool_, None, g, list_cap)?;
    let (l_in, l_end) = (b.id(), b.id());
    b.selection_merge(l_end, SelectionControl::NONE)?;
    b.branch_conditional(inside, l_in, l_end, [])?;
    b.begin_block(Some(l_in))?;
    let li = b.i_add(u32t, None, list_first, g)?;
    let pli = b.access_chain(p_u, None, marks, [c0, li])?;
    let page = b.load(u32t, None, pli, None, [])?;
    let first_word = b.i_mul(u32t, None, page, c32)?;
    let wi = b.i_add(u32t, None, first_word, lid)?;
    let pm = b.access_chain(p_u, None, marks, [c0, wi])?;
    let mw = b.load(u32t, None, pm, None, [])?;
    let g32 = b.i_mul(u32t, None, g, c32)?;
    let ri = b.i_add(u32t, None, masks_at, g32)?;
    let ri = b.i_add(u32t, None, ri, lid)?;
    let rp = region_word(&mut b, ri)?;
    b.store(rp, mw, Some(spirv::MemoryAccess::ALIGNED), [lit(4)])?;
    b.store(pm, c0, None, [])?;
    let (l_p, l_pd) = (b.id(), b.id());
    b.selection_merge(l_pd, SelectionControl::NONE)?;
    b.branch_conditional(is0, l_p, l_pd, [])?;
    b.begin_block(Some(l_p))?;
    let pi = b.i_add(u32t, None, pages_at, g)?;
    let pp = region_word(&mut b, pi)?;
    b.store(pp, page, Some(spirv::MemoryAccess::ALIGNED), [lit(4)])?;
    let pw = b.shift_right_logical(u32t, None, page, c5)?;
    let pwi = b.i_add(u32t, None, bits_at, pw)?;
    let pbit = b.bitwise_and(u32t, None, page, c31)?;
    let pmask = b.shift_left_logical(u32t, None, c1, pbit)?;
    let pinv = b.not(u32t, None, pmask)?;
    let ppb = b.access_chain(p_u, None, marks, [c0, pwi])?;
    b.atomic_and(u32t, None, ppb, scope_dev, sem_none, pinv)?;
    b.branch(l_pd)?;
    b.begin_block(Some(l_pd))?;
    b.branch(l_end)?;
    b.begin_block(Some(l_end))?;
    b.ret()?;
    b.end_function()?;
    b.entry_point(ExecutionModel::GLCompute, f, "main", [lid_v, wid_v]);
    b.execution_mode(f, ExecutionMode::LocalSize, [HANDOFF_LOCAL_SIZE, 1, 1]);
    Ok(b.module().assemble())
}

/// `read`: 0 stores every marked dword; 1 reads its guest-memory value first
/// and counts (counters 4 and 5); 2 as 1, storing only changed dwords.
/// `lazy`: pages with their aux bit set store nothing (`mirror_scatter_lazy`).
/// `note`: pages written back get their aux bit set (`mirror_scatter_note`).
/// `image`: 1 `mirror_scatter_image`, 2 `mirror_scatter_image_check`.
/// `region`: pages and mask words from a handoff region, nothing cleared
/// (`mirror_scatter_region`; with `reset` and none of the others).
fn build_scatter_ex(reset: bool, stores: bool, read: u8, lazy: bool, note: bool, image: u8, region: bool) -> Result<Vec<u32>, rspirv::dr::Error> {
    let census = read != 0;
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
    let pc_members = if region { 8 } else if image != 0 { 6 } else { 4 };
    let pc = b.type_struct(vec![u32t; pc_members]);
    b.decorate(pc, Decoration::Block, []);
    for k in 0..pc_members as u32 {
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
    // AcquireRelease | UniformMemory: the mask words read before any is cleared
    // (`image`: | WorkgroupMemory, the sums below added before they are read).
    let sem_wg = c(&mut b, 0x8 | 0x40 | if image != 0 { 0x100 } else { 0 });
    let sem_none = c0;
    // (`image`: the workgroup's dwords left out and, checked, those that
    // differ in guest memory, summed in workgroup memory and added to the
    // counters once a page: an atomic a page instead of one an invocation;
    // and the page's 128 sectors of 8 dwords holding a dword that differs
    // from the image, a bit each.)
    let wg_sums = if image != 0 {
        let p_wg = b.type_pointer(None, StorageClass::Workgroup, u32t);
        let n4 = c(&mut b, 4);
        let arr = b.type_array(u32t, n4);
        let p_arr = b.type_pointer(None, StorageClass::Workgroup, arr);
        Some((global(&mut b, p_wg, StorageClass::Workgroup), global(&mut b, p_wg, StorageClass::Workgroup), global(&mut b, p_arr, StorageClass::Workgroup), p_wg))
    } else {
        None
    };
    let p_fn = b.type_pointer(None, StorageClass::Function, u32t);
    let fty = b.type_function(void, []);
    let f = b.begin_function(void, None, FunctionControl::NONE, fty)?;
    b.begin_block(None)?;
    // (`census`: this invocation's dwords written back and those guest
    // memory held already.)
    let n_var = b.variable(p_fn, None, StorageClass::Function, Some(c0));
    let same_var = b.variable(p_fn, None, StorageClass::Function, Some(c0));
    // (`image`: each round's dword and whether it is copied, kept from the
    // comparison to the stores.)
    let (mut vvars, mut govars) = (Vec::new(), Vec::new());
    if image != 0 {
        for _ in 0..4 {
            vvars.push(b.variable(p_fn, None, StorageClass::Function, Some(c0)));
            govars.push(b.variable(p_fn, None, StorageClass::Function, Some(c0)));
        }
    }
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
    let image_base = if image != 0 {
        let c4 = c(&mut b, 4);
        let ilo = pc_at(&mut b, c4)?;
        let ihi = pc_at(&mut b, c5)?;
        Some(addr64(&mut b, ilo, ihi)?)
    } else {
        None
    };
    // (`region`: the handoff region's device address, push constants 6 and 7.)
    let region_base = if region {
        let c6 = c(&mut b, 6);
        let c7 = c(&mut b, 7);
        let rlo = pc_at(&mut b, c6)?;
        let rhi = pc_at(&mut b, c7)?;
        Some(addr64(&mut b, rlo, rhi)?)
    } else {
        None
    };
    // (Only in `region` mode: the other kernels' code stays as it was.)
    let c2_64 = if region { b.constant_bit64(u64t, 2) } else { 0 };
    // A region dword's pointer (`region`).
    let region_word = |b: &mut Builder, rb: Word, at: Word| -> Result<Word, rspirv::dr::Error> {
        let at64 = b.u_convert(u64t, None, at)?;
        let off = b.shift_left_logical(u64t, None, at64, c2_64)?;
        let a = b.i_add(u64t, None, rb, off)?;
        b.convert_u_to_ptr(p_dev, None, a)
    };
    let is0 = b.i_equal(bool_, None, lid, c0)?;
    let pcount = b.access_chain(p_u, None, marks, [c0, list_at])?;
    let inside = if region {
        // (The handoff emptied the list; the dispatch is the region's count.)
        b.u_less_than(bool_, None, g, list_cap)?
    } else if reset {
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
    if let Some((wl, wb, wsec, p_wg)) = wg_sums {
        // Invocation 0 zeroes the sums and the sector bits before any
        // invocation adds to them.
        let (l_z, l_zd) = (b.id(), b.id());
        b.selection_merge(l_zd, SelectionControl::NONE)?;
        b.branch_conditional(is0, l_z, l_zd, [])?;
        b.begin_block(Some(l_z))?;
        b.store(wl, c0, None, [])?;
        b.store(wb, c0, None, [])?;
        for k in 0..4u32 {
            let ck = c(&mut b, k);
            let pk = b.access_chain(p_wg, None, wsec, [ck])?;
            b.store(pk, c0, None, [])?;
        }
        b.branch(l_zd)?;
        b.begin_block(Some(l_zd))?;
        b.control_barrier(scope_wg, scope_wg, sem_wg)?;
    }
    let page = match region_base {
        Some(rb) => {
            let pages_at = c(&mut b, REGION_PAGES_AT);
            let at = b.i_add(u32t, None, pages_at, g)?;
            let pp = region_word(&mut b, rb, at)?;
            b.load(u32t, None, pp, Some(spirv::MemoryAccess::ALIGNED), [lit(4)])?
        }
        None => {
            let li = b.i_add(u32t, None, list_first, g)?;
            let pli = b.access_chain(p_u, None, marks, [c0, li])?;
            b.load(u32t, None, pli, None, [])?
        }
    };
    let first_word = b.i_mul(u32t, None, page, c32)?;
    // The page's k-th mask word: the map's, or (`region`) entry g's.
    let region_first = match region_base {
        Some(_) => {
            let masks_at = c(&mut b, REGION_MASKS_AT);
            let g32 = b.i_mul(u32t, None, g, c32)?;
            Some(b.i_add(u32t, None, masks_at, g32)?)
        }
        None => None,
    };
    let mask_word = |b: &mut Builder, k: Word| -> Result<Word, rspirv::dr::Error> {
        match (region_base, region_first) {
            (Some(rb), Some(rf)) => {
                let at = b.i_add(u32t, None, rf, k)?;
                let pp = region_word(b, rb, at)?;
                b.load(u32t, None, pp, Some(spirv::MemoryAccess::ALIGNED), [lit(4)])
            }
            _ => {
                let wi = b.i_add(u32t, None, first_word, k)?;
                let pm = b.access_chain(p_u, None, marks, [c0, wi])?;
                b.load(u32t, None, pm, None, [])
            }
        }
    };
    // The page's entry in the guest page table, its first byte in the mirror.
    let page64 = b.u_convert(u64t, None, page)?;
    let toff = b.i_mul(u64t, None, page64, c8_64)?;
    let ta = b.i_add(u64t, None, table, toff)?;
    let tp = b.convert_u_to_ptr(p_dev64, None, ta)?;
    let dest = b.load(u64t, None, tp, Some(spirv::MemoryAccess::ALIGNED), [lit(8)])?;
    let mapped = b.i_not_equal(bool_, None, dest, c0_64)?;
    // (`lazy`: the page's aux bit; such a page stores nothing.)
    let lazy_page = if lazy {
        let aux_at = c(&mut b, MARK_AUX_AT);
        let aw = b.shift_right_logical(u32t, None, page, c5)?;
        let awi = b.i_add(u32t, None, aux_at, aw)?;
        let pa = b.access_chain(p_u, None, marks, [c0, awi])?;
        let av = b.load(u32t, None, pa, None, [])?;
        let abit = b.bitwise_and(u32t, None, page, c31)?;
        let ash = b.shift_right_logical(u32t, None, av, abit)?;
        let aone = b.bitwise_and(u32t, None, ash, c1)?;
        Some(b.i_not_equal(bool_, None, aone, c0)?)
    } else {
        None
    };
    let stored = match lazy_page {
        Some(l) => {
            let eager = b.logical_not(bool_, None, l)?;
            b.logical_and(bool_, None, mapped, eager)?
        }
        None => mapped,
    };
    let moff = b.i_mul(u64t, None, page64, c4096_64)?;
    // (`image`: the page's validity bit in the aux region and its address
    // in the image.)
    let image_page = match image_base {
        Some(base) => {
            let aux_at = c(&mut b, MARK_AUX_AT);
            let aw = b.shift_right_logical(u32t, None, page, c5)?;
            let awi = b.i_add(u32t, None, aux_at, aw)?;
            let pa = b.access_chain(p_u, None, marks, [c0, awi])?;
            let av = b.load(u32t, None, pa, None, [])?;
            let abit = b.bitwise_and(u32t, None, page, c31)?;
            let ash = b.shift_right_logical(u32t, None, av, abit)?;
            let aone = b.bitwise_and(u32t, None, ash, c1)?;
            let valid = b.i_not_equal(bool_, None, aone, c0)?;
            let isrc = b.i_add(u64t, None, base, moff)?;
            Some((valid, isrc))
        }
        None => None,
    };
    let msrc = b.i_add(u64t, None, mirror, moff)?;
    let bit_of = b.bitwise_and(u32t, None, lid, c31)?;
    if let Some((valid, isrc)) = image_page {
        // Compared with the image (a valid page's) by sectors of 8 dwords
        // (32 bytes): a sector any of whose marked dwords differs has all
        // of them stored to guest memory and the image, the others none,
        // so the stores to guest memory keep the plain kernel's runs.
        // (Dwords left out one by one cut them into PCIe writes of a few
        // dwords: ~13.7 ms of a vista frame's GPU time, FINDINGS.md
        // 2026-10-31 night.) First the comparison, each round's dword and
        // whether it is copied kept, the sectors' bits set.
        let (_, _, wsec, p_wg) = wg_sums.expect("image sums");
        let c8 = c(&mut b, 8);
        let invalid = b.logical_not(bool_, None, valid)?;
        // The sector bit word of dword d (d >> 8) and its bit ((d >> 3) & 31).
        let sector_at = |b: &mut Builder, d: Word| -> Result<(Word, Word), rspirv::dr::Error> {
            let sw = b.shift_right_logical(u32t, None, d, c8)?;
            let sp = b.access_chain(p_wg, None, wsec, [sw])?;
            let sd = b.shift_right_logical(u32t, None, d, c3)?;
            let sb = b.bitwise_and(u32t, None, sd, c31)?;
            Ok((sp, sb))
        };
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
            let go = b.logical_and(bool_, None, set, stored)?;
            let go_u = b.select(u32t, None, go, c1, c0)?;
            b.store(govars[r as usize], go_u, None, [])?;
            let (l_a, l_ad) = (b.id(), b.id());
            b.selection_merge(l_ad, SelectionControl::NONE)?;
            b.branch_conditional(go, l_a, l_ad, [])?;
            b.begin_block(Some(l_a))?;
            let d4 = b.shift_left_logical(u32t, None, d, c2)?;
            let d64 = b.u_convert(u64t, None, d4)?;
            let sa = b.i_add(u64t, None, msrc, d64)?;
            let sp = b.convert_u_to_ptr(p_dev, None, sa)?;
            let v = b.load(u32t, None, sp, Some(spirv::MemoryAccess::ALIGNED), [lit(4)])?;
            b.store(vvars[r as usize], v, None, [])?;
            let ia = b.i_add(u64t, None, isrc, d64)?;
            let ip = b.convert_u_to_ptr(p_dev, None, ia)?;
            let gv = b.load(u32t, None, ip, Some(spirv::MemoryAccess::ALIGNED), [lit(4)])?;
            let differs = b.i_not_equal(bool_, None, gv, v)?;
            let flag = b.logical_and(bool_, None, differs, valid)?;
            let (l_f, l_fd) = (b.id(), b.id());
            b.selection_merge(l_fd, SelectionControl::NONE)?;
            b.branch_conditional(flag, l_f, l_fd, [])?;
            b.begin_block(Some(l_f))?;
            let (ps, sb) = sector_at(&mut b, d)?;
            let sm = b.shift_left_logical(u32t, None, c1, sb)?;
            b.atomic_or(u32t, None, ps, scope_wg, sem_none, sm)?;
            b.branch(l_fd)?;
            b.begin_block(Some(l_fd))?;
            b.branch(l_ad)?;
            b.begin_block(Some(l_ad))?;
        }
        // Every sector's bit set before any is read; then the stores.
        b.control_barrier(scope_wg, scope_wg, sem_wg)?;
        for r in 0..4u32 {
            let cr = b.constant_bit32(u32t, r * SCATTER_LOCAL_SIZE);
            let d = b.i_add(u32t, None, lid, cr)?;
            let go_u = b.load(u32t, None, govars[r as usize], None, [])?;
            let go = b.i_not_equal(bool_, None, go_u, c0)?;
            let (l_b, l_bd) = (b.id(), b.id());
            b.selection_merge(l_bd, SelectionControl::NONE)?;
            b.branch_conditional(go, l_b, l_bd, [])?;
            b.begin_block(Some(l_b))?;
            let v = b.load(u32t, None, vvars[r as usize], None, [])?;
            let d4 = b.shift_left_logical(u32t, None, d, c2)?;
            let d64 = b.u_convert(u64t, None, d4)?;
            let da = b.i_add(u64t, None, dest, d64)?;
            let dp = b.convert_u_to_ptr(p_dev, None, da)?;
            let ia = b.i_add(u64t, None, isrc, d64)?;
            let ip = b.convert_u_to_ptr(p_dev, None, ia)?;
            let (ps, sb) = sector_at(&mut b, d)?;
            let sw = b.load(u32t, None, ps, None, [])?;
            let ssh = b.shift_right_logical(u32t, None, sw, sb)?;
            let sone = b.bitwise_and(u32t, None, ssh, c1)?;
            let hit = b.i_not_equal(bool_, None, sone, c0)?;
            let store_it = b.logical_or(bool_, None, invalid, hit)?;
            let (l_st, l_skip, l_std) = (b.id(), b.id(), b.id());
            b.selection_merge(l_std, SelectionControl::NONE)?;
            b.branch_conditional(store_it, l_st, l_skip, [])?;
            b.begin_block(Some(l_st))?;
            b.store(dp, v, Some(spirv::MemoryAccess::ALIGNED), [lit(4)])?;
            b.store(ip, v, Some(spirv::MemoryAccess::ALIGNED), [lit(4)])?;
            b.branch(l_std)?;
            b.begin_block(Some(l_skip))?;
            let s0 = b.load(u32t, None, n_var, None, [])?;
            let s1 = b.i_add(u32t, None, s0, c1)?;
            b.store(n_var, s1, None, [])?;
            if image == 2 {
                // (Checking: guest memory read before the store it would
                // leave out, which is made anyway.)
                let old = b.load(u32t, None, dp, Some(spirv::MemoryAccess::ALIGNED), [lit(4)])?;
                let bad = b.i_not_equal(bool_, None, old, v)?;
                let one_if = b.select(u32t, None, bad, c1, c0)?;
                let m0 = b.load(u32t, None, same_var, None, [])?;
                let m1 = b.i_add(u32t, None, m0, one_if)?;
                b.store(same_var, m1, None, [])?;
                b.store(dp, v, Some(spirv::MemoryAccess::ALIGNED), [lit(4)])?;
            }
            b.branch(l_std)?;
            b.begin_block(Some(l_std))?;
            b.branch(l_bd)?;
            b.begin_block(Some(l_bd))?;
        }
    } else {
        // Dwords lid + 256 r, r = 0..3: marked ones copied.
        for r in 0..4u32 {
            let cr = b.constant_bit32(u32t, r * SCATTER_LOCAL_SIZE);
            let d = b.i_add(u32t, None, lid, cr)?;
            let wd = b.shift_right_logical(u32t, None, d, c5)?;
            let mw = mask_word(&mut b, wd)?;
            let sh = b.shift_right_logical(u32t, None, mw, bit_of)?;
            let bit = b.bitwise_and(u32t, None, sh, c1)?;
            let set = b.i_not_equal(bool_, None, bit, c0)?;
            let go = if stores { b.logical_and(bool_, None, set, stored)? } else { b.constant_false(bool_) };
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
            let mut changed = None;
            if census {
                let old = b.load(u32t, None, dp, Some(spirv::MemoryAccess::ALIGNED), [lit(4)])?;
                let eq = b.i_equal(bool_, None, old, v)?;
                let one_if = b.select(u32t, None, eq, c1, c0)?;
                let s0 = b.load(u32t, None, same_var, None, [])?;
                let s1 = b.i_add(u32t, None, s0, one_if)?;
                b.store(same_var, s1, None, [])?;
                let n0 = b.load(u32t, None, n_var, None, [])?;
                let n1 = b.i_add(u32t, None, n0, c1)?;
                b.store(n_var, n1, None, [])?;
                if read == 2 {
                    changed = Some(b.logical_not(bool_, None, eq)?);
                }
            }
            match changed {
                Some(ch) => {
                    let (l_st, l_std) = (b.id(), b.id());
                    b.selection_merge(l_std, SelectionControl::NONE)?;
                    b.branch_conditional(ch, l_st, l_std, [])?;
                    b.begin_block(Some(l_st))?;
                    b.store(dp, v, Some(spirv::MemoryAccess::ALIGNED), [lit(4)])?;
                    b.branch(l_std)?;
                    b.begin_block(Some(l_std))?;
                }
                None => b.store(dp, v, Some(spirv::MemoryAccess::ALIGNED), [lit(4)])?,
            }
            b.branch(l_cpd)?;
            b.begin_block(Some(l_cpd))?;
        }
    }
    if let Some((valid, isrc)) = image_page {
        // An invalid page: copied whole from the mirror into the image.
        let invalid = b.logical_not(bool_, None, valid)?;
        let (l_cp2, l_cp2d) = (b.id(), b.id());
        b.selection_merge(l_cp2d, SelectionControl::NONE)?;
        b.branch_conditional(invalid, l_cp2, l_cp2d, [])?;
        b.begin_block(Some(l_cp2))?;
        for r in 0..4u32 {
            let cr = b.constant_bit32(u32t, r * SCATTER_LOCAL_SIZE);
            let d = b.i_add(u32t, None, lid, cr)?;
            let d4 = b.shift_left_logical(u32t, None, d, c2)?;
            let d64 = b.u_convert(u64t, None, d4)?;
            let sa = b.i_add(u64t, None, msrc, d64)?;
            let sp = b.convert_u_to_ptr(p_dev, None, sa)?;
            let v = b.load(u32t, None, sp, Some(spirv::MemoryAccess::ALIGNED), [lit(4)])?;
            let ia = b.i_add(u64t, None, isrc, d64)?;
            let ip = b.convert_u_to_ptr(p_dev, None, ia)?;
            b.store(ip, v, Some(spirv::MemoryAccess::ALIGNED), [lit(4)])?;
        }
        b.branch(l_cp2d)?;
        b.begin_block(Some(l_cp2d))?;
        // The dwords left out and, checked, those that differ in guest
        // memory, summed for the workgroup (counters 6, 4 and 5: below).
        let n = b.load(u32t, None, n_var, None, [])?;
        let any = b.i_not_equal(bool_, None, n, c0)?;
        let (l_ic, l_icd) = (b.id(), b.id());
        b.selection_merge(l_icd, SelectionControl::NONE)?;
        b.branch_conditional(any, l_ic, l_icd, [])?;
        b.begin_block(Some(l_ic))?;
        let (wl, wb, _, _) = wg_sums.expect("image sums");
        b.atomic_i_add(u32t, None, wl, scope_wg, sem_none, n)?;
        if image == 2 {
            let bad = b.load(u32t, None, same_var, None, [])?;
            b.atomic_i_add(u32t, None, wb, scope_wg, sem_none, bad)?;
        }
        b.branch(l_icd)?;
        b.begin_block(Some(l_icd))?;
    }
    if census {
        let n = b.load(u32t, None, n_var, None, [])?;
        let any = b.i_not_equal(bool_, None, n, c0)?;
        let (l_cn, l_cnd) = (b.id(), b.id());
        b.selection_merge(l_cnd, SelectionControl::NONE)?;
        b.branch_conditional(any, l_cn, l_cnd, [])?;
        b.begin_block(Some(l_cn))?;
        let c4 = b.constant_bit32(u32t, 4);
        let pn = b.access_chain(p_u, None, counters, [c0, c4])?;
        b.atomic_i_add(u32t, None, pn, scope_dev, sem_none, n)?;
        let same = b.load(u32t, None, same_var, None, [])?;
        let pse = b.access_chain(p_u, None, counters, [c0, c5])?;
        b.atomic_i_add(u32t, None, pse, scope_dev, sem_none, same)?;
        b.branch(l_cnd)?;
        b.begin_block(Some(l_cnd))?;
    }
    // Invocation 0 counts the page's marked dwords (before they are cleared).
    let (l_cnt, l_cntd) = (b.id(), b.id());
    b.selection_merge(l_cntd, SelectionControl::NONE)?;
    b.branch_conditional(is0, l_cnt, l_cntd, [])?;
    b.begin_block(Some(l_cnt))?;
    let mut total = c0;
    for k in 0..32u32 {
        let ck = b.constant_bit32(u32t, k);
        let mw = mask_word(&mut b, ck)?;
        let n = b.bit_count(u32t, None, mw)?;
        total = b.i_add(u32t, None, total, n)?;
    }
    let which_dw = b.select(u32t, None, stored, c1, c0)?;
    let pdw = b.access_chain(p_u, None, counters, [c0, c1])?;
    let add = b.i_mul(u32t, None, total, which_dw)?;
    b.atomic_i_add(u32t, None, pdw, scope_dev, sem_none, add)?;
    if note {
        // (A page with dwords written back: its aux bit.)
        let any = b.u_greater_than(bool_, None, add, c0)?;
        let (l_nt, l_ntd) = (b.id(), b.id());
        b.selection_merge(l_ntd, SelectionControl::NONE)?;
        b.branch_conditional(any, l_nt, l_ntd, [])?;
        b.begin_block(Some(l_nt))?;
        let aux_at = c(&mut b, MARK_AUX_AT);
        let aw = b.shift_right_logical(u32t, None, page, c5)?;
        let awi = b.i_add(u32t, None, aux_at, aw)?;
        let pa = b.access_chain(p_u, None, marks, [c0, awi])?;
        let abit = b.bitwise_and(u32t, None, page, c31)?;
        let amask = b.shift_left_logical(u32t, None, c1, abit)?;
        b.atomic_or(u32t, None, pa, scope_dev, sem_none, amask)?;
        b.branch(l_ntd)?;
        b.begin_block(Some(l_ntd))?;
    }
    if let Some(l) = lazy_page {
        // (A lazy page's dwords and the page in counters 6 and 7.)
        let (l_lz, l_lzd) = (b.id(), b.id());
        b.selection_merge(l_lzd, SelectionControl::NONE)?;
        b.branch_conditional(l, l_lz, l_lzd, [])?;
        b.begin_block(Some(l_lz))?;
        let c6 = c(&mut b, 6);
        let c7 = c(&mut b, 7);
        let p6 = b.access_chain(p_u, None, counters, [c0, c6])?;
        b.atomic_i_add(u32t, None, p6, scope_dev, sem_none, total)?;
        let p7 = b.access_chain(p_u, None, counters, [c0, c7])?;
        b.atomic_i_add(u32t, None, p7, scope_dev, sem_none, c1)?;
        b.branch(l_lzd)?;
        b.begin_block(Some(l_lzd))?;
    }
    if census {
        // (The page's bit in the aux region: dwords and pages counted.)
        let aux_at = c(&mut b, MARK_AUX_AT);
        let aw = b.shift_right_logical(u32t, None, page, c5)?;
        let awi = b.i_add(u32t, None, aux_at, aw)?;
        let pa = b.access_chain(p_u, None, marks, [c0, awi])?;
        let av = b.load(u32t, None, pa, None, [])?;
        let abit = b.bitwise_and(u32t, None, page, c31)?;
        let ash = b.shift_right_logical(u32t, None, av, abit)?;
        let aone = b.bitwise_and(u32t, None, ash, c1)?;
        let aset = b.i_not_equal(bool_, None, aone, c0)?;
        let (l_ax, l_axd) = (b.id(), b.id());
        b.selection_merge(l_axd, SelectionControl::NONE)?;
        b.branch_conditional(aset, l_ax, l_axd, [])?;
        b.begin_block(Some(l_ax))?;
        let c6 = c(&mut b, 6);
        let c7 = c(&mut b, 7);
        let p6 = b.access_chain(p_u, None, counters, [c0, c6])?;
        b.atomic_i_add(u32t, None, p6, scope_dev, sem_none, add)?;
        let p7 = b.access_chain(p_u, None, counters, [c0, c7])?;
        b.atomic_i_add(u32t, None, p7, scope_dev, sem_none, c1)?;
        b.branch(l_axd)?;
        b.begin_block(Some(l_axd))?;
    }
    b.branch(l_cntd)?;
    b.begin_block(Some(l_cntd))?;
    // Every mask word read before any is cleared (`region`: the handoff
    // cleared them).
    b.control_barrier(scope_wg, scope_wg, sem_wg)?;
    let first32 = if region { b.constant_false(bool_) } else { b.u_less_than(bool_, None, lid, c32)? };
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
    if !region {
        let pw = b.shift_right_logical(u32t, None, page, c5)?;
        let pwi = b.i_add(u32t, None, bits_at, pw)?;
        let pbit = b.bitwise_and(u32t, None, page, c31)?;
        let pmask = b.shift_left_logical(u32t, None, c1, pbit)?;
        let pinv = b.not(u32t, None, pmask)?;
        let ppb = b.access_chain(p_u, None, marks, [c0, pwi])?;
        b.atomic_and(u32t, None, ppb, scope_dev, sem_none, pinv)?;
    }
    let which = b.select(u32t, None, mapped, c0, c2)?;
    let pcnt = b.access_chain(p_u, None, counters, [c0, which])?;
    let one_unless_lazy = match lazy_page {
        Some(l) => b.select(u32t, None, l, c0, c1)?,
        None => c1,
    };
    b.atomic_i_add(u32t, None, pcnt, scope_dev, sem_none, one_unless_lazy)?;
    if let Some((valid, _)) = image_page {
        let invalid = b.logical_not(bool_, None, valid)?;
        let (l_iv, l_ivd) = (b.id(), b.id());
        b.selection_merge(l_ivd, SelectionControl::NONE)?;
        b.branch_conditional(invalid, l_iv, l_ivd, [])?;
        b.begin_block(Some(l_iv))?;
        let aux_at = c(&mut b, MARK_AUX_AT);
        let aw = b.shift_right_logical(u32t, None, page, c5)?;
        let awi = b.i_add(u32t, None, aux_at, aw)?;
        let pa = b.access_chain(p_u, None, marks, [c0, awi])?;
        let abit = b.bitwise_and(u32t, None, page, c31)?;
        let amask = b.shift_left_logical(u32t, None, c1, abit)?;
        b.atomic_or(u32t, None, pa, scope_dev, sem_none, amask)?;
        let c7 = c(&mut b, 7);
        let p7 = b.access_chain(p_u, None, counters, [c0, c7])?;
        b.atomic_i_add(u32t, None, p7, scope_dev, sem_none, c1)?;
        b.branch(l_ivd)?;
        b.begin_block(Some(l_ivd))?;
        // The workgroup's sums (after the barrier): dwords left out into
        // counter 6 and, checked, into 4, those differing in guest memory
        // into 5.
        let (wl, wb, _, _) = wg_sums.expect("image sums");
        let n = b.load(u32t, None, wl, None, [])?;
        let c6 = c(&mut b, 6);
        let p6 = b.access_chain(p_u, None, counters, [c0, c6])?;
        b.atomic_i_add(u32t, None, p6, scope_dev, sem_none, n)?;
        if image == 2 {
            let c4 = c(&mut b, 4);
            let p4 = b.access_chain(p_u, None, counters, [c0, c4])?;
            b.atomic_i_add(u32t, None, p4, scope_dev, sem_none, n)?;
            let bad = b.load(u32t, None, wb, None, [])?;
            let p5 = b.access_chain(p_u, None, counters, [c0, c5])?;
            b.atomic_i_add(u32t, None, p5, scope_dev, sem_none, bad)?;
        }
    }
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
    fn detile_kernel_assembles() {
        let w = super::detile();
        assert_eq!(w[0], 0x0723_0203);
        if let Ok(dir) = std::env::var("RDNA_KERNEL_DUMP") {
            let bytes: Vec<u8> = w.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(std::path::Path::new(&dir).join("detile.spv"), bytes).unwrap();
        }
    }

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
    fn scatter_image_kernels_assemble() {
        for (w, name) in [(super::mirror_scatter_image(), "mirror_scatter_image.spv"), (super::mirror_scatter_image_check(), "mirror_scatter_image_check.spv")] {
            assert_eq!(w[0], 0x0723_0203);
            if let Ok(dir) = std::env::var("RDNA_KERNEL_DUMP") {
                let bytes: Vec<u8> = w.iter().flat_map(|x| x.to_le_bytes()).collect();
                std::fs::write(std::path::Path::new(&dir).join(name), bytes).unwrap();
            }
        }
    }

    #[test]
    fn handoff_kernels_assemble() {
        let reset = super::mirror_scatter_reset();
        for (w, name) in [(super::mirror_handoff(), "mirror_handoff.spv"), (super::mirror_scatter_region(), "mirror_scatter_region.spv"), (reset.clone(), "mirror_scatter_reset.spv")] {
            assert_eq!(w[0], 0x0723_0203);
            if let Ok(dir) = std::env::var("RDNA_KERNEL_DUMP") {
                let bytes: Vec<u8> = w.iter().flat_map(|x| x.to_le_bytes()).collect();
                std::fs::write(std::path::Path::new(&dir).join(name), bytes).unwrap();
            }
        }
        assert_ne!(super::mirror_scatter_region(), reset);
    }

    #[test]
    fn scatter_note_kernel_assembles() {
        let w = super::mirror_scatter_note();
        assert_eq!(w[0], 0x0723_0203);
        assert_ne!(w, super::mirror_scatter_reset());
        if let Ok(dir) = std::env::var("RDNA_KERNEL_DUMP") {
            let bytes: Vec<u8> = w.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(std::path::Path::new(&dir).join("mirror_scatter_note.spv"), bytes).unwrap();
        }
    }

    #[test]
    fn scatter_lazy_kernel_assembles() {
        let w = super::mirror_scatter_lazy();
        assert_eq!(w[0], 0x0723_0203);
        assert_ne!(w, super::mirror_scatter_reset());
        if let Ok(dir) = std::env::var("RDNA_KERNEL_DUMP") {
            let bytes: Vec<u8> = w.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(std::path::Path::new(&dir).join("mirror_scatter_lazy.spv"), bytes).unwrap();
        }
    }

    #[test]
    fn compact_kernel_assembles() {
        let w = super::mirror_compact();
        assert_eq!(w[0], 0x0723_0203);
        if let Ok(dir) = std::env::var("RDNA_KERNEL_DUMP") {
            let bytes: Vec<u8> = w.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(std::path::Path::new(&dir).join("mirror_compact.spv"), bytes).unwrap();
        }
    }

    #[test]
    fn scatter_census_kernel_assembles() {
        for (w, name) in [(super::mirror_scatter_census(), "mirror_scatter_census.spv"), (super::mirror_scatter_changed(), "mirror_scatter_changed.spv")] {
            assert_eq!(w[0], 0x0723_0203);
            assert_ne!(w, super::mirror_scatter_reset());
            if let Ok(dir) = std::env::var("RDNA_KERNEL_DUMP") {
                let bytes: Vec<u8> = w.iter().flat_map(|x| x.to_le_bytes()).collect();
                std::fs::write(std::path::Path::new(&dir).join(name), bytes).unwrap();
            }
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

/// Workgroup size of `detile`.
pub const DETILE_LOCAL_SIZE: u32 = 64;
/// Push-constant words of `detile` (112 bytes, under the guaranteed 128).
pub const DETILE_PC_WORDS: usize = 28;

/// Texture detiling on the GPU (front end 2 stage F4): one (mip, layer) of a
/// gfx10 thin surface, tiled in binding 0 (u32 words from the part's start
/// at byte `src_off`), written to binding 1 as tightly packed rows of
/// elements from dword `dst_off`. Invocation o (row `gid.y` of `stride`
/// invocations) writes output dword o < `count`. An element's byte offset
/// is `src_off` + its block's ((y >> lh) * pitch_blocks + (x >> lw)) <<
/// block_log2 + the XOR over address bits i of the parities of x mask_i &
/// (x % bw) and y mask_i & (y % bh) (the CPU detiler's `xs[x] ^ ys[y]`),
/// with (x, y) the element's coordinates plus the mip-tail offsets (tx,
/// ty); bytes from `src_len` (a multiple of 4) on read as zero. Push constants: {src_off,
/// dst_off, w, h, tx, ty, pitch_blocks, block_log2 | lw << 8 | lh << 16 |
/// elem_log2 << 24, count, stride, src_len, 0, then per address bit i < 16
/// x mask | y mask << 16}.
pub fn detile() -> Vec<u32> {
    build_detile().expect("detile kernel")
}

fn build_detile() -> Result<Vec<u32>, rspirv::dr::Error> {
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
    let pc = b.type_struct(vec![u32t; DETILE_PC_WORDS]);
    b.decorate(pc, Decoration::Block, []);
    for i in 0..DETILE_PC_WORDS as u32 {
        b.member_decorate(pc, i, Decoration::Offset, [lit(4 * i)]);
    }
    let p_pc = b.type_pointer(None, StorageClass::PushConstant, pc);
    let p_pcu = b.type_pointer(None, StorageClass::PushConstant, u32t);
    let p_in = b.type_pointer(None, StorageClass::Input, v3);
    let gid = global(&mut b, p_in, StorageClass::Input);
    b.decorate(gid, Decoration::BuiltIn, [Operand::BuiltIn(BuiltIn::GlobalInvocationId)]);
    let src = global(&mut b, p_st, StorageClass::StorageBuffer);
    b.decorate(src, Decoration::DescriptorSet, [lit(0)]);
    b.decorate(src, Decoration::Binding, [lit(0)]);
    b.decorate(src, Decoration::NonWritable, []);
    let dst = global(&mut b, p_st, StorageClass::StorageBuffer);
    b.decorate(dst, Decoration::DescriptorSet, [lit(0)]);
    b.decorate(dst, Decoration::Binding, [lit(1)]);
    let pcv = global(&mut b, p_pc, StorageClass::PushConstant);
    let k: Vec<Word> = (0..=32).map(|i| b.constant_bit32(u32t, i)).collect();
    let kff = b.constant_bit32(u32t, 0xFF);
    let kffff = b.constant_bit32(u32t, 0xFFFF);
    let fty = b.type_function(void, []);
    let f = b.begin_function(void, None, FunctionControl::NONE, fty)?;
    b.begin_block(None)?;
    let mut pcw = Vec::with_capacity(DETILE_PC_WORDS);
    for i in 0..DETILE_PC_WORDS {
        let p = b.access_chain(p_pcu, None, pcv, [k[i]])?;
        pcw.push(b.load(u32t, None, p, None, [])?);
    }
    let (src_off, dst_off, w, h, tx, ty, pitch, shifts, count, stride, src_len) = (pcw[0], pcw[1], pcw[2], pcw[3], pcw[4], pcw[5], pcw[6], pcw[7], pcw[8], pcw[9], pcw[10]);
    let g = b.load(v3, None, gid, None, [])?;
    let gx = b.composite_extract(u32t, None, g, [0])?;
    let gy = b.composite_extract(u32t, None, g, [1])?;
    let row = b.i_mul(u32t, None, gy, stride)?;
    let o = b.i_add(u32t, None, row, gx)?;
    let inside = b.u_less_than(bool_, None, o, count)?;
    let (l_body, l_end) = (b.id(), b.id());
    b.selection_merge(l_end, SelectionControl::NONE)?;
    b.branch_conditional(inside, l_body, l_end, [])?;
    b.begin_block(Some(l_body))?;
    let bl = b.bitwise_and(u32t, None, shifts, kff)?;
    let s8 = b.shift_right_logical(u32t, None, shifts, k[8])?;
    let lw = b.bitwise_and(u32t, None, s8, kff)?;
    let s16 = b.shift_right_logical(u32t, None, shifts, k[16])?;
    let lh = b.bitwise_and(u32t, None, s16, kff)?;
    let e = b.shift_right_logical(u32t, None, shifts, k[24])?;
    let bw = b.shift_left_logical(u32t, None, k[1], lw)?;
    let mw = b.i_sub(u32t, None, bw, k[1])?;
    let bh = b.shift_left_logical(u32t, None, k[1], lh)?;
    let mh = b.i_sub(u32t, None, bh, k[1])?;
    let total = b.i_mul(u32t, None, w, h)?;
    // Byte offset of element `el` in binding 0.
    let offset = |b: &mut Builder, el: Word| -> Result<Word, rspirv::dr::Error> {
        let x = b.u_mod(u32t, None, el, w)?;
        let y = b.u_div(u32t, None, el, w)?;
        let x = b.i_add(u32t, None, x, tx)?;
        let y = b.i_add(u32t, None, y, ty)?;
        let by = b.shift_right_logical(u32t, None, y, lh)?;
        let bx = b.shift_right_logical(u32t, None, x, lw)?;
        let r = b.i_mul(u32t, None, by, pitch)?;
        let blk = b.i_add(u32t, None, r, bx)?;
        let blk = b.shift_left_logical(u32t, None, blk, bl)?;
        let xm = b.bitwise_and(u32t, None, x, mw)?;
        let ym = b.bitwise_and(u32t, None, y, mh)?;
        let mut inner = k[0];
        for i in 0..16 {
            let m = pcw[12 + i];
            let mx = b.bitwise_and(u32t, None, m, kffff)?;
            let my = b.shift_right_logical(u32t, None, m, k[16])?;
            let ax = b.bitwise_and(u32t, None, mx, xm)?;
            let ay = b.bitwise_and(u32t, None, my, ym)?;
            let cx = b.bit_count(u32t, None, ax)?;
            let cy = b.bit_count(u32t, None, ay)?;
            let c = b.i_add(u32t, None, cx, cy)?;
            let p = b.bitwise_and(u32t, None, c, k[1])?;
            let p = b.shift_left_logical(u32t, None, p, k[i])?;
            inner = b.bitwise_or(u32t, None, inner, p)?;
        }
        let off = b.i_add(u32t, None, src_off, blk)?;
        b.i_add(u32t, None, off, inner)
    };
    // The dword holding byte `off` (0 when that dword is not below
    // `src_len`, a multiple of 4, or when `ok` is false).
    let src_words = b.shift_right_logical(u32t, None, src_len, k[2])?;
    let word_at = |b: &mut Builder, off: Word, ok: Word| -> Result<Word, rspirv::dr::Error> {
        let idx = b.shift_right_logical(u32t, None, off, k[2])?;
        let inb = b.u_less_than(bool_, None, idx, src_words)?;
        let inb = b.logical_and(bool_, None, inb, ok)?;
        let idx = b.select(u32t, None, inb, idx, k[0])?;
        let p = b.access_chain(p_u, None, src, [k[0], idx])?;
        let v = b.load(u32t, None, p, None, [])?;
        b.select(u32t, None, inb, v, k[0])
    };
    let wide = b.u_greater_than_equal(bool_, None, e, k[2])?;
    let (l_wide, l_narrow, l_join) = (b.id(), b.id(), b.id());
    b.selection_merge(l_join, SelectionControl::NONE)?;
    b.branch_conditional(wide, l_wide, l_narrow, [])?;
    // Elements of 4, 8 or 16 bytes: dword (o mod n) of element o / n, n
    // their dwords.
    b.begin_block(Some(l_wide))?;
    let q = b.i_sub(u32t, None, e, k[2])?;
    let el = b.shift_right_logical(u32t, None, o, q)?;
    let n = b.shift_left_logical(u32t, None, k[1], q)?;
    let nm = b.i_sub(u32t, None, n, k[1])?;
    let kd = b.bitwise_and(u32t, None, o, nm)?;
    let kb = b.shift_left_logical(u32t, None, kd, k[2])?;
    let base = offset(&mut b, el)?;
    let off = b.i_add(u32t, None, base, kb)?;
    let ok = b.u_less_than(bool_, None, el, total)?;
    let v_wide = word_at(&mut b, off, ok)?;
    b.branch(l_join)?;
    // Elements of 1 or 2 bytes: the 4 >> e elements from o << (2 - e).
    b.begin_block(Some(l_narrow))?;
    let per = b.shift_right_logical(u32t, None, k[4], e)?;
    let sh = b.i_sub(u32t, None, k[2], e)?;
    let first = b.shift_left_logical(u32t, None, o, sh)?;
    let bits = b.shift_left_logical(u32t, None, k[8], e)?;
    let one_bits = b.shift_left_logical(u32t, None, k[1], bits)?;
    let emask = b.i_sub(u32t, None, one_bits, k[1])?;
    let e3 = b.i_add(u32t, None, e, k[3])?;
    let mut acc = k[0];
    for j in 0..4 {
        let el = b.i_add(u32t, None, first, k[j])?;
        let in_word = b.u_less_than(bool_, None, k[j], per)?;
        let in_part = b.u_less_than(bool_, None, el, total)?;
        let ok = b.logical_and(bool_, None, in_word, in_part)?;
        let off = offset(&mut b, el)?;
        let wd = word_at(&mut b, off, ok)?;
        let lo = b.bitwise_and(u32t, None, off, k[3])?;
        let lo = b.shift_left_logical(u32t, None, lo, k[3])?;
        let v = b.shift_right_logical(u32t, None, wd, lo)?;
        let v = b.bitwise_and(u32t, None, v, emask)?;
        let at = b.shift_left_logical(u32t, None, k[j], e3)?;
        let v = b.shift_left_logical(u32t, None, v, at)?;
        acc = b.bitwise_or(u32t, None, acc, v)?;
    }
    let v_narrow = acc;
    b.branch(l_join)?;
    b.begin_block(Some(l_join))?;
    let v = b.phi(u32t, None, [(v_wide, l_wide), (v_narrow, l_narrow)])?;
    let di = b.i_add(u32t, None, dst_off, o)?;
    let pd = b.access_chain(p_u, None, dst, [k[0], di])?;
    b.store(pd, v, None, [])?;
    b.branch(l_end)?;
    b.begin_block(Some(l_end))?;
    b.ret()?;
    b.end_function()?;
    b.entry_point(ExecutionModel::GLCompute, f, "main", [gid]);
    b.execution_mode(f, ExecutionMode::LocalSize, [DETILE_LOCAL_SIZE, 1, 1]);
    Ok(b.module().assemble())
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
