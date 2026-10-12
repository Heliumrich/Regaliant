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

#[cfg(test)]
mod tests {
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
    fn merge_kernel_assembles() {
        let w = super::shadow_merge();
        assert_eq!(w[0], 0x0723_0203);
        if let Ok(dir) = std::env::var("RDNA_KERNEL_DUMP") {
            let bytes: Vec<u8> = w.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(std::path::Path::new(&dir).join("shadow_merge.spv"), bytes).unwrap();
        }
    }
}
