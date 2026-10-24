//! `dsport --dispatch-bench`: whether the GPU overlaps consecutive small
//! dispatches. The title's particle chain records ~1,000 dispatches a frame
//! at a vista, one or a few workgroups of a long program each, at ~35 us of
//! GPU time each (FINDINGS.md 2026-10-24). Here one workgroup per dispatch
//! of a latency-bound test kernel (`rdna::kernels::latency_test`) is
//! recorded N times back to back: one pipeline or N distinct ones, with or
//! without a full barrier between dispatches, with the descriptor set bound
//! once or pushed per dispatch; each case timed by timestamps (median of a
//! few submissions), one dispatch alone for reference.

use super::vk::ctx;
use ash::vk;

pub fn run() -> i32 {
    if let Err(e) = super::vk::init() {
        crate::error!("dispatch bench: Vulkan: {e}");
        return 2;
    }
    match bench() {
        Ok(()) => 0,
        Err(e) => {
            crate::error!("dispatch bench: {e}");
            2
        }
    }
}

const N: usize = 256;

fn bench() -> Result<(), String> {
    let c = ctx();
    let d = &c.device;
    let env = |k: &str, v: u32| std::env::var(k).ok().and_then(|x| x.parse().ok()).unwrap_or(v);
    let iters = env("DS_DB_ITERS", 8000);
    let groups = env("DS_DB_GROUPS", 1);
    let local = rdna::kernels::LATENCY_LOCAL_SIZE;
    let words = (N as u64 + 1) * local as u64 * groups as u64;
    let (buf, _mem) = c.device_buffer(4 * words, vk::BufferUsageFlags::STORAGE_BUFFER)?;
    unsafe {
        let binding = [vk::DescriptorSetLayoutBinding::default().binding(0).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE)];
        let set_layout = d.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&binding), None).map_err(|e| format!("set layout: {e}"))?;
        let push_layout = d
            .create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().flags(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR).bindings(&binding), None)
            .map_err(|e| format!("push set layout: {e}"))?;
        let range = [vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::COMPUTE).offset(0).size(8)];
        let layout = d.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&[set_layout]).push_constant_ranges(&range), None).map_err(|e| format!("layout: {e}"))?;
        let playout = d.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&[push_layout]).push_constant_ranges(&range), None).map_err(|e| format!("push layout: {e}"))?;
        let make = |salt: u32, layout: vk::PipelineLayout| -> Result<vk::Pipeline, String> {
            let code = rdna::kernels::latency_test(salt);
            let module = d.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None).map_err(|e| format!("module: {e}"))?;
            let stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(c"main");
            let ci = vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout);
            let p = d.create_compute_pipelines(vk::PipelineCache::null(), &[ci], None).map_err(|(_, e)| format!("pipeline: {e}"))?[0];
            d.destroy_shader_module(module, None);
            Ok(p)
        };
        let t0 = std::time::Instant::now();
        let mut pipes = Vec::with_capacity(N);
        let mut ppipes = Vec::with_capacity(N);
        for s in 0..N as u32 {
            pipes.push(make(s, layout)?);
            ppipes.push(make(s, playout)?);
        }
        crate::info!("dispatch bench: {} pipelines made in {} ms; {iters} steps a thread, {groups} workgroup(s) of {local} a dispatch", 2 * N, t0.elapsed().as_millis());
        let sizes = [vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 1 }];
        let pool = d.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&sizes), None).map_err(|e| format!("pool: {e}"))?;
        let set = d.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&[set_layout])).map_err(|e| format!("set: {e}"))?[0];
        let bi = [vk::DescriptorBufferInfo::default().buffer(buf).offset(0).range(vk::WHOLE_SIZE)];
        d.update_descriptor_sets(&[vk::WriteDescriptorSet::default().dst_set(set).dst_binding(0).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&bi)], &[]);
        let cpool = d.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(c.queue_family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER), None).map_err(|e| format!("command pool: {e}"))?;
        let cmd = d.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(cpool).command_buffer_count(1)).map_err(|e| format!("cmd: {e}"))?[0];
        let fence = d.create_fence(&vk::FenceCreateInfo::default(), None).map_err(|e| format!("fence: {e}"))?;
        let qp = d.create_query_pool(&vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(2), None).map_err(|e| format!("query pool: {e}"))?;
        let period = {
            let props = c.instance.get_physical_device_properties(c.pdev);
            props.limits.timestamp_period as f64
        };
        // A case: n dispatches; distinct pipelines; a barrier between; pushed descriptors.
        let mut time_case = |n: usize, distinct: bool, barriers: bool, push: bool| -> Result<f64, String> {
            let mut ts = Vec::new();
            for _ in 0..7 {
                d.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty()).ok();
                d.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).map_err(|e| format!("begin: {e}"))?;
                d.cmd_reset_query_pool(cmd, qp, 0, 2);
                let full = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::MEMORY_WRITE).dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
                d.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::ALL_COMMANDS, vk::DependencyFlags::empty(), &[full], &[], &[]);
                d.cmd_write_timestamp(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, qp, 0);
                let lay = if push { playout } else { layout };
                if !push {
                    d.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, layout, 0, &[set], &[]);
                }
                let mut bound = vk::Pipeline::null();
                for k in 0..n {
                    let p = if push { ppipes[if distinct { k } else { 0 }] } else { pipes[if distinct { k } else { 0 }] };
                    if p != bound {
                        d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, p);
                        bound = p;
                    }
                    if push {
                        let w = [vk::WriteDescriptorSet::default().dst_binding(0).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&bi)];
                        c.push_desc.cmd_push_descriptor_set(cmd, vk::PipelineBindPoint::COMPUTE, playout, 0, &w);
                    }
                    let pcs = [iters, (k as u32) * local * groups];
                    let bytes: Vec<u8> = pcs.iter().flat_map(|x| x.to_le_bytes()).collect();
                    d.cmd_push_constants(cmd, lay, vk::ShaderStageFlags::COMPUTE, 0, &bytes);
                    d.cmd_dispatch(cmd, groups, 1, 1);
                    if barriers && k + 1 < n {
                        d.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::ALL_COMMANDS, vk::DependencyFlags::empty(), &[full], &[], &[]);
                    }
                }
                d.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, qp, 1);
                d.end_command_buffer(cmd).map_err(|e| format!("end: {e}"))?;
                c.submit(cmd, &[], &[], fence)?;
                d.wait_for_fences(&[fence], true, 10_000_000_000).map_err(|e| format!("wait: {e}"))?;
                d.reset_fences(&[fence]).ok();
                let mut t = [0u64; 2];
                d.get_query_pool_results(qp, 0, &mut t, vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT).map_err(|e| format!("query: {e}"))?;
                ts.push((t[1].wrapping_sub(t[0])) as f64 * period / 1000.0);
            }
            ts.sort_by(|a, b| a.total_cmp(b));
            Ok(ts[ts.len() / 2])
        };
        let one = time_case(1, false, false, false)?;
        crate::info!("dispatch bench: one dispatch alone {one:.1} us");
        for (distinct, barriers, push) in [(false, false, false), (true, false, false), (false, true, false), (true, true, false), (false, false, true), (true, false, true)] {
            let t = time_case(N, distinct, barriers, push)?;
            crate::info!(
                "dispatch bench: {N} dispatches, {} pipeline(s), {}, descriptors {}: {:.0} us ({:.1} us each; {:.1}x one alone; {:.0} us if serial)",
                if distinct { "distinct" } else { "one" },
                if barriers { "a full barrier between" } else { "no barriers" },
                if push { "pushed per dispatch" } else { "bound once" },
                t,
                t / N as f64,
                t / one,
                one * N as f64
            );
        }
    }
    Ok(())
}
