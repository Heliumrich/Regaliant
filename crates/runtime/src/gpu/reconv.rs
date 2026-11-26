//! The subgroup reconvergence check (`check`, once at start): whether a subgroup's invocations
//! are back together at a structured merge on this device, without maximal reconvergence declared
//! and (where the device offers it) with it (`exec::declare_max_reconvergence`,
//! `vk::max_reconvergence`). The translation's cross-lane reads assume they are (RENDERER.md "Wave
//! semantics"); NVIDIA's RTX 40 series brings them together without the declaration. The log line
//! says how many invocations of 32 each subgroup held after a loop that only odd invocations run
//! (1 to 8 iterations by index) and after a branch every third one takes: 32 and 32 when they are
//! back together. A report from a design that draws wrongly, beside one from a design that draws
//! right, says whether the declaration's guarantee is what differs (issue #4, 2026-11-24).

use super::vk::ctx;
use ash::vk;
use rspirv::binary::Assemble;
use rspirv::dr::{Builder, Instruction, Operand};
use rspirv::spirv::{self, BuiltIn, Capability, Decoration, FunctionControl, GroupOperation, LoopControl, Op, Scope, SelectionControl, StorageClass, Word};

/// Invocations of the check's one workgroup: two subgroups of 32.
const LANES: u32 = 64;
/// Words each invocation writes: its subgroup's invocations at the loop's merge, at the branch's
/// merge, whether its neighbour (subgroup lane ^ 1) was at the loop's merge, and its value.
const PER: u32 = 4;

fn global(b: &mut Builder, ptr_ty: Word, sc: StorageClass) -> Word {
    let id = b.id();
    b.module_mut().types_global_values.push(Instruction::new(Op::Variable, Some(ptr_ty), Some(id), vec![Operand::StorageClass(sc)]));
    id
}

fn kernel() -> Result<Vec<u32>, rspirv::dr::Error> {
    let lit = Operand::LiteralBit32;
    let mut b = Builder::new();
    b.set_version(1, 3);
    b.capability(Capability::Shader);
    b.capability(Capability::GroupNonUniform);
    b.capability(Capability::GroupNonUniformBallot);
    b.capability(Capability::GroupNonUniformShuffle);
    b.memory_model(spirv::AddressingModel::Logical, spirv::MemoryModel::GLSL450);
    let void = b.type_void();
    let bool_ = b.type_bool();
    let u32t = b.type_int(32, 0);
    let v4u = b.type_vector(u32t, 4);
    let rta = b.type_runtime_array(u32t);
    b.decorate(rta, Decoration::ArrayStride, [lit(4)]);
    let st = b.type_struct([rta]);
    b.decorate(st, Decoration::Block, []);
    b.member_decorate(st, 0, Decoration::Offset, [lit(0)]);
    let p_st = b.type_pointer(None, StorageClass::StorageBuffer, st);
    let p_u = b.type_pointer(None, StorageClass::StorageBuffer, u32t);
    let p_in = b.type_pointer(None, StorageClass::Input, u32t);
    let p_fn = b.type_pointer(None, StorageClass::Function, u32t);
    let li_var = global(&mut b, p_in, StorageClass::Input);
    b.decorate(li_var, Decoration::BuiltIn, [Operand::BuiltIn(BuiltIn::LocalInvocationIndex)]);
    let sl_var = global(&mut b, p_in, StorageClass::Input);
    b.decorate(sl_var, Decoration::BuiltIn, [Operand::BuiltIn(BuiltIn::SubgroupLocalInvocationId)]);
    let out = global(&mut b, p_st, StorageClass::StorageBuffer);
    b.decorate(out, Decoration::DescriptorSet, [lit(0)]);
    b.decorate(out, Decoration::Binding, [lit(0)]);
    let k = |b: &mut Builder, v: u32| b.constant_bit32(u32t, v);
    let (c0, c1, c3, c7) = (k(&mut b, 0), k(&mut b, 1), k(&mut b, 3), k(&mut b, 7));
    let (ca, cb, cc) = (k(&mut b, 1_664_525), k(&mut b, 1_013_904_223), k(&mut b, 0x9e37_79b9));
    let cper = k(&mut b, PER);
    let sg = k(&mut b, Scope::Subgroup as u32);
    let t = b.constant_true(bool_);
    let fty = b.type_function(void, []);
    let f = b.begin_function(void, None, FunctionControl::NONE, fty)?;
    b.begin_block(None)?;
    let x_var = b.variable(p_fn, None, StorageClass::Function, None);
    let i_var = b.variable(p_fn, None, StorageClass::Function, None);
    let li = b.load(u32t, None, li_var, None, [])?;
    let sl = b.load(u32t, None, sl_var, None, [])?;
    b.store(x_var, li, None, [])?;
    b.store(i_var, c0, None, [])?;
    // Odd invocations run a loop of 1 to 8 iterations (by index); even ones skip it.
    let odd_bit = b.bitwise_and(u32t, None, li, c1)?;
    let odd = b.i_not_equal(bool_, None, odd_bit, c0)?;
    let (body, merge1) = (b.id(), b.id());
    b.selection_merge(merge1, SelectionControl::NONE)?;
    b.branch_conditional(odd, body, merge1, [])?;
    b.begin_block(Some(body))?;
    let sh = b.shift_right_logical(u32t, None, li, c1)?;
    let m = b.bitwise_and(u32t, None, sh, c7)?;
    let trip = b.i_add(u32t, None, m, c1)?;
    let (header, test, lbody, cont, lmerge) = (b.id(), b.id(), b.id(), b.id(), b.id());
    b.branch(header)?;
    b.begin_block(Some(header))?;
    b.loop_merge(lmerge, cont, LoopControl::NONE, [])?;
    b.branch(test)?;
    b.begin_block(Some(test))?;
    let i = b.load(u32t, None, i_var, None, [])?;
    let more = b.u_less_than(bool_, None, i, trip)?;
    b.branch_conditional(more, lbody, lmerge, [])?;
    b.begin_block(Some(lbody))?;
    let x = b.load(u32t, None, x_var, None, [])?;
    let x = b.i_mul(u32t, None, x, ca)?;
    let x = b.i_add(u32t, None, x, cb)?;
    b.store(x_var, x, None, [])?;
    b.branch(cont)?;
    b.begin_block(Some(cont))?;
    let i = b.load(u32t, None, i_var, None, [])?;
    let i1 = b.i_add(u32t, None, i, c1)?;
    b.store(i_var, i1, None, [])?;
    b.branch(header)?;
    b.begin_block(Some(lmerge))?;
    b.branch(merge1)?;
    b.begin_block(Some(merge1))?;
    // At the merge: the subgroup's invocations here, and the neighbour's value as a cross-lane
    // read takes it (`T::lane_fetch`).
    let ballot = b.group_non_uniform_ballot(v4u, None, sg, t)?;
    let n1 = b.group_non_uniform_ballot_bit_count(u32t, None, sg, GroupOperation::Reduce, ballot)?;
    let nb = b.bitwise_xor(u32t, None, sl, c1)?;
    let live = b.group_non_uniform_ballot_bit_extract(bool_, None, sg, ballot, nb)?;
    let x = b.load(u32t, None, x_var, None, [])?;
    let y = b.group_non_uniform_shuffle(u32t, None, sg, x, nb)?;
    let y = b.select(u32t, None, live, y, c0)?;
    let livew = b.select(u32t, None, live, c1, c0)?;
    // Every third invocation takes a branch of straight-line work (a lane-guarded region).
    let r3 = b.u_mod(u32t, None, li, c3)?;
    let third = b.i_equal(bool_, None, r3, c0)?;
    let (body2, merge2) = (b.id(), b.id());
    b.selection_merge(merge2, SelectionControl::NONE)?;
    b.branch_conditional(third, body2, merge2, [])?;
    b.begin_block(Some(body2))?;
    let z = b.bitwise_xor(u32t, None, y, cc)?;
    let z = b.i_mul(u32t, None, z, c7)?;
    let z = b.shift_right_logical(u32t, None, z, c3)?;
    b.store(x_var, z, None, [])?;
    b.branch(merge2)?;
    b.begin_block(Some(merge2))?;
    let ballot2 = b.group_non_uniform_ballot(v4u, None, sg, t)?;
    let n2 = b.group_non_uniform_ballot_bit_count(u32t, None, sg, GroupOperation::Reduce, ballot2)?;
    let xv = b.load(u32t, None, x_var, None, [])?;
    let base = b.i_mul(u32t, None, li, cper)?;
    for (j, v) in [n1, n2, livew, xv].into_iter().enumerate() {
        let kj = k(&mut b, j as u32);
        let at = b.i_add(u32t, None, base, kj)?;
        let p = b.access_chain(p_u, None, out, [c0, at])?;
        b.store(p, v, None, [])?;
    }
    b.ret()?;
    b.end_function()?;
    b.entry_point(spirv::ExecutionModel::GLCompute, f, "main", [li_var, sl_var]);
    b.execution_mode(f, spirv::ExecutionMode::LocalSize, [LANES, 1, 1]);
    Ok(b.module().assemble())
}

/// The Vulkan objects of one run of a check kernel, destroyed when it ends (`dispatch`): those made so
/// far, also when a step fails; left alone when the GPU may still use them (a wait that timed
/// out: `std::mem::forget`).
#[derive(Default)]
struct Objects {
    buf: vk::Buffer,
    mem: vk::DeviceMemory,
    module: vk::ShaderModule,
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    pool: vk::DescriptorPool,
    cpool: vk::CommandPool,
    fence: vk::Fence,
}

impl Drop for Objects {
    fn drop(&mut self) {
        let d = &ctx().device;
        unsafe {
            if self.fence != vk::Fence::null() {
                d.destroy_fence(self.fence, None);
            }
            if self.cpool != vk::CommandPool::null() {
                d.destroy_command_pool(self.cpool, None);
            }
            if self.pool != vk::DescriptorPool::null() {
                d.destroy_descriptor_pool(self.pool, None);
            }
            if self.pipeline != vk::Pipeline::null() {
                d.destroy_pipeline(self.pipeline, None);
            }
            if self.layout != vk::PipelineLayout::null() {
                d.destroy_pipeline_layout(self.layout, None);
            }
            if self.set_layout != vk::DescriptorSetLayout::null() {
                d.destroy_descriptor_set_layout(self.set_layout, None);
            }
            if self.module != vk::ShaderModule::null() {
                d.destroy_shader_module(self.module, None);
            }
            if self.buf != vk::Buffer::null() {
                d.destroy_buffer(self.buf, None);
            }
            if self.mem != vk::DeviceMemory::null() {
                d.free_memory(self.mem, None);
            }
        }
    }
}

/// One run of the check kernel `code`: per invocation its [n1, n2, neighbour, value].
fn run(code: &[u32]) -> Result<Vec<[u32; 4]>, String> {
    let words = dispatch(code, &vec![u32::MAX; (LANES * PER) as usize])?;
    Ok(words.chunks(PER as usize).map(|w| [w[0], w[1], w[2], w[3]]).collect())
}

/// One workgroup of a check kernel over a host-visible storage buffer (binding 0) that holds
/// `init` before the dispatch: the buffer's words after it.
fn dispatch(code: &[u32], init: &[u32]) -> Result<Vec<u32>, String> {
    let c = ctx();
    let d = &c.device;
    let bytes = (init.len() * 4) as u64;
    let mut o = Objects::default();
    unsafe {
        let (buf, mem, ptr) = c.host_buffer(bytes, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST)?;
        (o.buf, o.mem) = (buf, mem);
        std::ptr::copy_nonoverlapping(init.as_ptr(), ptr as *mut u32, init.len());
        o.module = d.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(code), None).map_err(|e| format!("module: {e}"))?;
        let binding = [vk::DescriptorSetLayoutBinding::default().binding(0).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE)];
        o.set_layout = d.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&binding), None).map_err(|e| format!("set layout: {e}"))?;
        let layouts = [o.set_layout];
        o.layout = d.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts), None).map_err(|e| format!("pipeline layout: {e}"))?;
        // (32 lanes as the game's compute pipelines require them, `exec::compute_pipeline`.)
        let mut req32 = vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo::default().required_subgroup_size(32);
        let mut stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(o.module).name(c"main");
        if c.wants_subgroup32(vk::ShaderStageFlags::COMPUTE) {
            stage = stage.push_next(&mut req32);
        }
        let ci = [vk::ComputePipelineCreateInfo::default().stage(stage).layout(o.layout)];
        o.pipeline = d.create_compute_pipelines(vk::PipelineCache::null(), &ci, None).map_err(|(_, e)| format!("pipeline: {e}"))?[0];
        let sizes = [vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 1 }];
        o.pool = d.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&sizes), None).map_err(|e| format!("descriptor pool: {e}"))?;
        let set = d.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(o.pool).set_layouts(&layouts)).map_err(|e| format!("descriptor set: {e}"))?[0];
        let bi = [vk::DescriptorBufferInfo { buffer: o.buf, offset: 0, range: bytes }];
        let w = [vk::WriteDescriptorSet::default().dst_set(set).dst_binding(0).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&bi)];
        d.update_descriptor_sets(&w, &[]);
        o.cpool = d.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(c.queue_family), None).map_err(|e| format!("command pool: {e}"))?;
        let cmd = d.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(o.cpool).command_buffer_count(1)).map_err(|e| format!("command buffer: {e}"))?[0];
        o.fence = d.create_fence(&vk::FenceCreateInfo::default(), None).map_err(|e| format!("fence: {e}"))?;
        d.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).map_err(|e| format!("begin: {e}"))?;
        d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, o.pipeline);
        d.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, o.layout, 0, &[set], &[]);
        d.cmd_dispatch(cmd, 1, 1, 1);
        let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::HOST_READ);
        d.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::HOST, vk::DependencyFlags::empty(), &[mb], &[], &[]);
        d.end_command_buffer(cmd).map_err(|e| format!("end: {e}"))?;
        c.submit(cmd, &[], &[], o.fence)?;
        if let Err(e) = d.wait_for_fences(&[o.fence], true, 5_000_000_000) {
            // (The GPU may still run it: its objects are left alone.)
            std::mem::forget(o);
            return Err(format!("wait: {e}"));
        }
        Ok(std::slice::from_raw_parts(ptr as *const u32, init.len()).to_vec())
    }
}

/// "a of 32 after the loop, b after the branch, neighbours read c of 64" for one run.
fn summary(r: &[[u32; 4]]) -> String {
    let n1 = r.iter().map(|v| v[0]).min().unwrap_or(0);
    let n2 = r.iter().map(|v| v[1]).min().unwrap_or(0);
    let live = r.iter().filter(|v| v[2] == 1).count();
    format!("at least {n1} of 32 at the loop's merge, {n2} at the branch's, {live} of {LANES} neighbours read")
}

/// Runs the check (once, at start, on the device's queue before any game work) and logs it.
pub fn check() {
    if std::env::var("DS_RECONV_CHECK").ok().as_deref() == Some("0") {
        return;
    }
    let code = match kernel() {
        Ok(c) => c,
        Err(e) => {
            crate::warn!("subgroup reconvergence check: kernel: {e}");
            return;
        }
    };
    let plain = run(&code).map(|r| summary(&r)).unwrap_or_else(|e| format!("failed ({e})"));
    let declared = if super::vk::max_reconvergence() {
        match super::exec::declare_max_reconvergence(&code) {
            Some(d) => run(&d).map(|r| summary(&r)).unwrap_or_else(|e| format!("failed ({e})")),
            None => "not made".into(),
        }
    } else {
        "not offered or not on".into()
    };
    crate::info!("subgroup reconvergence check: without the declaration {plain}; declaring maximal reconvergence {declared} (32, 32 and 64: back together; `DS_RECONV_CHECK=0` skips it)");
}

/// The NaN rules check's inputs (words 0-3, written by the CPU so that no compiler can fold
/// them): a NaN, 71.0, 0.0 and 1.0.
const NAN_IN: [u32; 4] = [0x7fc0_0000, 0x428e_0000, 0, 0x3f80_0000];
/// Where the kernel's results start.
const NAN_OUT: usize = 8;
/// The kernel's results in order, and what the translation's float rules (RENDERER.md) expect.
/// (The translation tests NaNs on the bits since 2026-11-26; `IsNan` stays as a driver fact. The
/// compares are the translation's own: ordered for v_cmp_lt and the like, unordered for
/// v_cmp_nge and the like.)
const NAN_EXPECT: [(&str, u32); 11] = [
    ("NMin(NaN, 1)", 0x3f80_0000),
    ("NMax(NaN, 1)", 0x3f80_0000),
    ("the old med3 form (71, NaN, 0)", 0x428e_0000),
    ("med3(71, NaN, 0)", 0),
    ("NClamp(NaN, 0, 1)", 0),
    ("IsNan(NaN)", 1),
    ("NaN == NaN", 0),
    ("NaN != NaN", 1),
    ("cvt_i32(NaN)", 0),
    ("NaN < 1 (ordered)", 0),
    ("NaN < 1 (unordered)", 1),
];

/// The NaN rules check's kernel (one invocation): the operations the translation's float rules
/// rest on (RENDERER.md "float rules"), each on the NaN it reads, written from `NAN_OUT` on.
fn nan_kernel() -> Result<Vec<u32>, rspirv::dr::Error> {
    use rspirv::spirv::GlslStd450Op as GLOp;
    let lit = Operand::LiteralBit32;
    let mut b = Builder::new();
    b.set_version(1, 3);
    b.capability(Capability::Shader);
    let glsl = b.ext_inst_import("GLSL.std.450");
    b.memory_model(spirv::AddressingModel::Logical, spirv::MemoryModel::GLSL450);
    let void = b.type_void();
    let bool_ = b.type_bool();
    let u32t = b.type_int(32, 0);
    let i32t = b.type_int(32, 1);
    let f32t = b.type_float(32, None);
    let rta = b.type_runtime_array(u32t);
    b.decorate(rta, Decoration::ArrayStride, [lit(4)]);
    let st = b.type_struct([rta]);
    b.decorate(st, Decoration::Block, []);
    b.member_decorate(st, 0, Decoration::Offset, [lit(0)]);
    let p_st = b.type_pointer(None, StorageClass::StorageBuffer, st);
    let p_u = b.type_pointer(None, StorageClass::StorageBuffer, u32t);
    let buf = global(&mut b, p_st, StorageClass::StorageBuffer);
    b.decorate(buf, Decoration::DescriptorSet, [lit(0)]);
    b.decorate(buf, Decoration::Binding, [lit(0)]);
    let c0 = b.constant_bit32(u32t, 0);
    let c1 = b.constant_bit32(u32t, 1);
    let kabs = b.constant_bit32(u32t, 0x7fff_ffff);
    let kinf = b.constant_bit32(u32t, 0x7f80_0000);
    let lo = b.constant_bit32(f32t, (-2147483648.0f32).to_bits());
    let hi = b.constant_bit32(f32t, 2147483520.0f32.to_bits());
    let idx: Vec<Word> = (0..(NAN_OUT + NAN_EXPECT.len()) as u32).map(|i| b.constant_bit32(u32t, i)).collect();
    let fty = b.type_function(void, []);
    let f = b.begin_function(void, None, FunctionControl::NONE, fty)?;
    b.begin_block(None)?;
    let mut input = Vec::new();
    for &at in &idx[..NAN_IN.len()] {
        let p = b.access_chain(p_u, None, buf, [c0, at])?;
        let u = b.load(u32t, None, p, None, [])?;
        input.push(b.bitcast(f32t, None, u)?);
    }
    let (nan, k71, zero, one) = (input[0], input[1], input[2], input[3]);
    let gl = |b: &mut Builder, op: GLOp, args: &[Word]| b.ext_inst(f32t, None, glsl, op as u32, args.iter().map(|&a| Operand::IdRef(a)));
    let nan_bits = |b: &mut Builder, x: Word| -> Result<Word, rspirv::dr::Error> {
        let u = b.bitcast(u32t, None, x)?;
        let a = b.bitwise_and(u32t, None, u, kabs)?;
        b.u_greater_than(bool_, None, a, kinf)
    };
    let nmin = gl(&mut b, GLOp::NMin, &[nan, one])?;
    let nmax = gl(&mut b, GLOp::NMax, &[nan, one])?;
    // med3(71.0, NaN, 0.0) as the translation makes it (`translate::alu`): the median of
    // NMin/NMax, and min3 when a NaN is among the inputs.
    let mn = gl(&mut b, GLOp::NMin, &[k71, nan])?;
    let mx = gl(&mut b, GLOp::NMax, &[k71, nan])?;
    let m2 = gl(&mut b, GLOp::NMin, &[mx, zero])?;
    let med = gl(&mut b, GLOp::NMax, &[mn, m2])?;
    let min3 = gl(&mut b, GLOp::NMin, &[mn, zero])?;
    let na = nan_bits(&mut b, k71)?;
    let nb = nan_bits(&mut b, nan)?;
    let nc = nan_bits(&mut b, zero)?;
    let any = b.logical_or(bool_, None, na, nb)?;
    let any = b.logical_or(bool_, None, any, nc)?;
    let med3 = b.select(f32t, None, any, min3, med)?;
    let nclamp = gl(&mut b, GLOp::NClamp, &[nan, zero, one])?;
    let isnan = b.is_nan(bool_, None, nan)?;
    let eq = b.f_ord_equal(bool_, None, nan, nan)?;
    let ne = b.f_unord_not_equal(bool_, None, nan, nan)?;
    let olt = b.f_ord_less_than(bool_, None, nan, one)?;
    let ult = b.f_unord_less_than(bool_, None, nan, one)?;
    let cl = gl(&mut b, GLOp::NClamp, &[nan, lo, hi])?;
    let cv = b.convert_f_to_s(i32t, None, cl)?;
    let cv = b.bitcast(u32t, None, cv)?;
    let cvt = b.select(u32t, None, nb, c0, cv)?;
    let mut outs = Vec::new();
    for v in [nmin, nmax, med, med3, nclamp] {
        outs.push(b.bitcast(u32t, None, v)?);
    }
    for t in [isnan, eq, ne] {
        outs.push(b.select(u32t, None, t, c1, c0)?);
    }
    outs.push(cvt);
    for t in [olt, ult] {
        outs.push(b.select(u32t, None, t, c1, c0)?);
    }
    for (j, v) in outs.into_iter().enumerate() {
        let p = b.access_chain(p_u, None, buf, [c0, idx[NAN_OUT + j]])?;
        b.store(p, v, None, [])?;
    }
    b.ret()?;
    b.end_function()?;
    b.entry_point(spirv::ExecutionModel::GLCompute, f, "main", []);
    b.execution_mode(f, spirv::ExecutionMode::LocalSize, [1, 1, 1]);
    Ok(b.module().assemble())
}

/// The NaN rules check (once at start, after the subgroup check): the operations the
/// translation's float rules rest on, on a NaN read from memory, without and with
/// SignedZeroInfNanPreserve declared (where the device offers it for 32 bits). The log line
/// names any operation whose result differs from what the rules expect (NMin/NMax giving the
/// NaN, `OpIsNan` folded, a NaN equal to itself): cards differ in what their drivers do with
/// NaNs, and every reporter of stars, speckles and glowing lines ran the RTX 50 path, which
/// declares the preservation (2026-11-26).
pub fn nan_check() {
    if std::env::var("DS_NAN_CHECK").ok().as_deref() == Some("0") {
        return;
    }
    let code = match nan_kernel() {
        Ok(c) => c,
        Err(e) => {
            crate::warn!("NaN rules check: kernel: {e}");
            return;
        }
    };
    let mut init = vec![0xdead_beefu32; NAN_OUT + NAN_EXPECT.len()];
    init[..NAN_IN.len()].copy_from_slice(&NAN_IN);
    let summary = |r: Result<Vec<u32>, String>| match r {
        Ok(w) => {
            let mut off = Vec::new();
            for (j, &(name, want)) in NAN_EXPECT.iter().enumerate() {
                let got = w[NAN_OUT + j];
                if got != want {
                    off.push(format!("{name} gave {got:#x}, not {want:#x}"));
                }
            }
            if off.is_empty() { "as the rules expect".to_string() } else { off.join(", ") }
        }
        Err(e) => format!("failed ({e})"),
    };
    let plain = summary(dispatch(&code, &init));
    let offered = super::vk::nan_preserve_offered();
    let declared = if offered & 2 != 0 {
        match super::exec::declare_float_modes(&code, super::exec::NAN_PRESERVE_MODES, offered) {
            Some(d) => summary(dispatch(&d, &init)),
            None => "not made".into(),
        }
    } else {
        "not offered".into()
    };
    let used = if super::vk::nan_preserve() { "declaring it" } else { "plain" };
    crate::info!("NaN rules check: plain {plain}; declaring NaN preservation {declared} (the translated shaders run {used}; `DS_NAN_CHECK=0` skips it)");
}

#[cfg(test)]
mod tests {
    /// The kernel and its declared form assemble; `RECONV_SPV_DUMP=<dir>` writes both for spirv-val.
    #[test]
    fn kernel_assembles() {
        let code = super::kernel().expect("kernel");
        let declared = crate::gpu::exec::declare_max_reconvergence(&code).expect("declared form");
        assert_eq!(declared.len(), code.len() + 1 + 8 + 3);
        if let Some(dir) = std::env::var_os("RECONV_SPV_DUMP") {
            let dir = std::path::PathBuf::from(dir);
            let bytes = |w: &[u32]| w.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
            std::fs::write(dir.join("reconv_plain.spv"), bytes(&code)).unwrap();
            std::fs::write(dir.join("reconv_declared.spv"), bytes(&declared)).unwrap();
        }
    }

    /// The NaN rules kernel assembles, also with NaN preservation declared for 32 bits;
    /// `RECONV_SPV_DUMP=<dir>` writes both for spirv-val.
    #[test]
    fn nan_kernel_assembles() {
        let code = super::nan_kernel().expect("kernel");
        assert_eq!(code[0], 0x0723_0203);
        let declared = crate::gpu::exec::declare_float_modes(&code, crate::gpu::exec::NAN_PRESERVE_MODES, 2).expect("declared form");
        assert!(declared.len() > code.len());
        if let Some(dir) = std::env::var_os("RECONV_SPV_DUMP") {
            let dir = std::path::PathBuf::from(dir);
            let bytes = |w: &[u32]| w.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
            std::fs::write(dir.join("nan_plain.spv"), bytes(&code)).unwrap();
            std::fs::write(dir.join("nan_declared.spv"), bytes(&declared)).unwrap();
        }
    }
}
