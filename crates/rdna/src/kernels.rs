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
/// Then one bit per page: the counting write-back's buckets
/// (`mirror_scatter_counting`: WB_BUCKETS dword counts, then WB_BUCKETS
/// page counts), else unused.
pub const MARK_AUX_AT: u32 = MARK_LIST_AT + 4 + MARK_LIST_CAP;
pub const MARK_AUX_WORDS: u32 = (MARK_SPAN_BYTES / 4096 / 32) as u32;
pub const MARK_TOTAL_WORDS: u32 = MARK_AUX_AT + MARK_AUX_WORDS;
/// `mirror_scatter_counting`'s buckets: 2^WB_BUCKET_SHIFT pages (1 MiB).
pub const WB_BUCKET_SHIFT: u32 = 8;
pub const WB_BUCKETS: u32 = (MARK_SPAN_BYTES / 4096 >> WB_BUCKET_SHIFT) as u32;
/// Then a bit per bucket: its pages' dwords are left out of the counting
/// write-back's stores (diagnosis, the runtime's `wbb lazy`).
pub const WB_LAZY_AT: u32 = MARK_AUX_AT + 2 * WB_BUCKETS;
const _: () = assert!(2 * WB_BUCKETS + WB_BUCKETS / 32 <= MARK_AUX_WORDS);

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
/// MARK_LIST_CAP; four more unused (the lazy write-back study's census,
/// FINDINGS.md 2026-10-31 night).
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
    build_scatter(false, false, false).expect("mirror scatter kernel")
}

/// `mirror_scatter` emptying the touched-page list itself: workgroup 0's
/// first invocation takes the list's count and stores 0 there (an atomic
/// exchange; the indirect dispatch read the count already), and the other
/// workgroups never read it (they are bounded by MARK_LIST_CAP alone: the
/// dispatch made as many as the count), so the write-back needs no fill of
/// the count and no barrier before one.
pub fn mirror_scatter_reset() -> Vec<u32> {
    build_scatter(true, false, false).expect("mirror scatter kernel (emptying the list)")
}

/// `mirror_scatter_reset` that also adds, per bucket of 2^WB_BUCKET_SHIFT
/// pages of the span, the dwords it writes back and the pages with any into
/// the map's aux region ([dwords by bucket, then pages by bucket];
/// diagnosis, the runtime's `DS_WB_BUCKETS`), and stores nothing for pages
/// of buckets whose bit at WB_LAZY_AT is set (their marks cleared, their
/// dwords counted as if stored).
pub fn mirror_scatter_counting() -> Vec<u32> {
    build_scatter(true, true, false).expect("mirror scatter kernel (counting buckets)")
}

/// `mirror_scatter_reset` that also counts what holding pages would leave
/// out (diagnosis, the runtime's `DS_FE2_HOLD=1`; FRONTEND2.md "GPU-held
/// pages"): a bitmap of held pages (a bit per page of the span, bit i of
/// word i / 32) at the device address in push constants 4-5 (non-zero), and
/// for each listed page whose bit is set its marked dwords are added to
/// counter 4 and the page to counter 5 (as stored or not); stores unchanged.
/// A page with marked dwords also gets its bit in the map's aux region (bit
/// i of word MARK_AUX_AT + i / 32: written back since the runtime last read
/// and cleared them).
pub fn mirror_scatter_hold_census() -> Vec<u32> {
    build_scatter(true, false, true).expect("mirror scatter kernel (hold census)")
}

fn build_scatter(reset: bool, count: bool, hold: bool) -> Result<Vec<u32>, rspirv::dr::Error> {
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
    let pc_members = if hold { 6 } else { 4 };
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
    let page = {
        let li = b.i_add(u32t, None, list_first, g)?;
        let pli = b.access_chain(p_u, None, marks, [c0, li])?;
        b.load(u32t, None, pli, None, [])?
    };
    let first_word = b.i_mul(u32t, None, page, c32)?;
    // The page's k-th mask word.
    let mask_word = |b: &mut Builder, k: Word| -> Result<Word, rspirv::dr::Error> {
        let wi = b.i_add(u32t, None, first_word, k)?;
        let pm = b.access_chain(p_u, None, marks, [c0, wi])?;
        b.load(u32t, None, pm, None, [])
    };
    // The page's entry in the guest page table, its first byte in the mirror.
    let page64 = b.u_convert(u64t, None, page)?;
    let toff = b.i_mul(u64t, None, page64, c8_64)?;
    let ta = b.i_add(u64t, None, table, toff)?;
    let tp = b.convert_u_to_ptr(p_dev64, None, ta)?;
    let dest = b.load(u64t, None, tp, Some(spirv::MemoryAccess::ALIGNED), [lit(8)])?;
    let mapped = b.i_not_equal(bool_, None, dest, c0_64)?;
    let moff = b.i_mul(u64t, None, page64, c4096_64)?;
    let msrc = b.i_add(u64t, None, mirror, moff)?;
    // (`count`: a page of a lazy bucket stores nothing; its dwords are
    // counted as stored.)
    let store_ok = if count {
        let shift = c(&mut b, WB_BUCKET_SHIFT);
        let bucket = b.shift_right_logical(u32t, None, page, shift)?;
        let lw = b.shift_right_logical(u32t, None, bucket, c5)?;
        let lazy_at = c(&mut b, WB_LAZY_AT);
        let lwi = b.i_add(u32t, None, lazy_at, lw)?;
        let plw = b.access_chain(p_u, None, marks, [c0, lwi])?;
        let lword = b.load(u32t, None, plw, None, [])?;
        let lbit = b.bitwise_and(u32t, None, bucket, c31)?;
        let lsh = b.shift_right_logical(u32t, None, lword, lbit)?;
        let lone = b.bitwise_and(u32t, None, lsh, c1)?;
        let eager = b.i_equal(bool_, None, lone, c0)?;
        b.logical_and(bool_, None, mapped, eager)?
    } else {
        mapped
    };
    let bit_of = b.bitwise_and(u32t, None, lid, c31)?;
    // Dwords lid + 256 r, r = 0..3: marked ones copied.
    for r in 0..4u32 {
        let cr = b.constant_bit32(u32t, r * SCATTER_LOCAL_SIZE);
        let d = b.i_add(u32t, None, lid, cr)?;
        let wd = b.shift_right_logical(u32t, None, d, c5)?;
        let mw = mask_word(&mut b, wd)?;
        let sh = b.shift_right_logical(u32t, None, mw, bit_of)?;
        let bit = b.bitwise_and(u32t, None, sh, c1)?;
        let set = b.i_not_equal(bool_, None, bit, c0)?;
        let go = b.logical_and(bool_, None, set, store_ok)?;
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
        let mw = mask_word(&mut b, ck)?;
        let n = b.bit_count(u32t, None, mw)?;
        total = b.i_add(u32t, None, total, n)?;
    }
    let which_dw = b.select(u32t, None, mapped, c1, c0)?;
    let pdw = b.access_chain(p_u, None, counters, [c0, c1])?;
    let add = b.i_mul(u32t, None, total, which_dw)?;
    b.atomic_i_add(u32t, None, pdw, scope_dev, sem_none, add)?;
    if hold {
        // (The page's bit in the held-page bitmap: its dwords and the page
        // counted when set.)
        let c4 = c(&mut b, 4);
        let hlo = pc_at(&mut b, c4)?;
        let hhi = pc_at(&mut b, c5)?;
        let hmap = addr64(&mut b, hlo, hhi)?;
        let hw = b.shift_right_logical(u32t, None, page, c5)?;
        let hw64 = b.u_convert(u64t, None, hw)?;
        let c4_64 = b.constant_bit64(u64t, 4);
        let hoff = b.i_mul(u64t, None, hw64, c4_64)?;
        let ha = b.i_add(u64t, None, hmap, hoff)?;
        let hp = b.convert_u_to_ptr(p_dev, None, ha)?;
        let hword = b.load(u32t, None, hp, Some(spirv::MemoryAccess::ALIGNED), [lit(4)])?;
        let hbit = b.bitwise_and(u32t, None, page, c31)?;
        let hsh = b.shift_right_logical(u32t, None, hword, hbit)?;
        let hone = b.bitwise_and(u32t, None, hsh, c1)?;
        let held_dw = b.i_mul(u32t, None, total, hone)?;
        let phd = b.access_chain(p_u, None, counters, [c0, c4])?;
        b.atomic_i_add(u32t, None, phd, scope_dev, sem_none, held_dw)?;
        let php = b.access_chain(p_u, None, counters, [c0, c5])?;
        b.atomic_i_add(u32t, None, php, scope_dev, sem_none, hone)?;
        // (Written back: the page's bit in the aux region.)
        let any = b.u_greater_than(bool_, None, total, c0)?;
        let one_at = b.shift_left_logical(u32t, None, c1, hbit)?;
        let wbit = b.select(u32t, None, any, one_at, c0)?;
        let aux_at = c(&mut b, MARK_AUX_AT);
        let awi = b.i_add(u32t, None, aux_at, hw)?;
        let paw = b.access_chain(p_u, None, marks, [c0, awi])?;
        b.atomic_or(u32t, None, paw, scope_dev, sem_none, wbit)?;
    }
    if count {
        // (The page's bucket: its dwords, and the page once when any.)
        let shift = c(&mut b, WB_BUCKET_SHIFT);
        let bucket = b.shift_right_logical(u32t, None, page, shift)?;
        let dw_at = c(&mut b, MARK_AUX_AT);
        let bi = b.i_add(u32t, None, dw_at, bucket)?;
        let pb = b.access_chain(p_u, None, marks, [c0, bi])?;
        b.atomic_i_add(u32t, None, pb, scope_dev, sem_none, add)?;
        let any = b.u_greater_than(bool_, None, add, c0)?;
        let one = b.select(u32t, None, any, c1, c0)?;
        let pg_at = c(&mut b, MARK_AUX_AT + WB_BUCKETS);
        let pi = b.i_add(u32t, None, pg_at, bucket)?;
        let pp = b.access_chain(p_u, None, marks, [c0, pi])?;
        b.atomic_i_add(u32t, None, pp, scope_dev, sem_none, one)?;
    }
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
    {
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
    fn compact_kernel_assembles() {
        let w = super::mirror_compact();
        assert_eq!(w[0], 0x0723_0203);
        if let Ok(dir) = std::env::var("RDNA_KERNEL_DUMP") {
            let bytes: Vec<u8> = w.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(std::path::Path::new(&dir).join("mirror_compact.spv"), bytes).unwrap();
        }
    }

    #[test]
    fn scatter_hold_census_kernel_assembles() {
        let w = super::mirror_scatter_hold_census();
        assert_eq!(w[0], 0x0723_0203);
        assert_ne!(w, super::mirror_scatter_reset());
        if let Ok(dir) = std::env::var("RDNA_KERNEL_DUMP") {
            let bytes: Vec<u8> = w.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(std::path::Path::new(&dir).join("mirror_scatter_hold_census.spv"), bytes).unwrap();
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
    fn scatter_counting_kernel_assembles() {
        let w = super::mirror_scatter_counting();
        assert_eq!(w[0], 0x0723_0203);
        assert_ne!(w, super::mirror_scatter_reset());
        if let Ok(dir) = std::env::var("RDNA_KERNEL_DUMP") {
            let bytes: Vec<u8> = w.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(std::path::Path::new(&dir).join("mirror_scatter_counting.spv"), bytes).unwrap();
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
