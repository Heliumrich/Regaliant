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
        let time_case = |n: usize, distinct: bool, barriers: bool, push: bool| -> Result<f64, String> {
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
                        c.push_descriptors(cmd, vk::PipelineBindPoint::COMPUTE, playout, 0, &w);
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
        // Two queues (with `DS_ASYNC_COMPUTE=2` at boot the device has a
        // compute-only family's queue, `Ctx::compute`): the chain with a
        // full barrier between dispatches split in halves, one on the main
        // queue and one there, submitted together, against the whole chain
        // on the main queue; wall time until both fences signal. Whether the
        // GPU overlaps two barrier-bound streams (front end 2's frames are
        // chains of small dispatches and drains on one queue: the schedule
        // model's two-queue bound assumes it does, FINDINGS.md 2026-11-01).
        if let Some(cq) = c.compute.as_ref() {
            let cpool2 = d.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(cq.family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER), None).map_err(|e| format!("command pool 2: {e}"))?;
            let cmd2 = d.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(cpool2).command_buffer_count(1)).map_err(|e| format!("cmd 2: {e}"))?[0];
            let fence2 = d.create_fence(&vk::FenceCreateInfo::default(), None).map_err(|e| format!("fence 2: {e}"))?;
            let full = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::MEMORY_WRITE).dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
            let record = |cmd: vk::CommandBuffer, from: usize, n: usize| -> Result<(), String> {
                d.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty()).ok();
                d.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).map_err(|e| format!("begin: {e}"))?;
                d.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, layout, 0, &[set], &[]);
                d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipes[0]);
                for k in from..from + n {
                    let pcs = [iters, (k as u32) * local * groups];
                    let bytes: Vec<u8> = pcs.iter().flat_map(|x| x.to_le_bytes()).collect();
                    d.cmd_push_constants(cmd, layout, vk::ShaderStageFlags::COMPUTE, 0, &bytes);
                    d.cmd_dispatch(cmd, groups, 1, 1);
                    if k + 1 < from + n {
                        d.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::ALL_COMMANDS, vk::DependencyFlags::empty(), &[full], &[], &[]);
                    }
                }
                d.end_command_buffer(cmd).map_err(|e| format!("end: {e}"))
            };
            // On the main queue (`main`), the compute family's (`alt`) or both.
            let wall = |main: usize, alt: usize| -> Result<f64, String> {
                let mut ts = Vec::new();
                for _ in 0..7 {
                    if main > 0 {
                        record(cmd, 0, main)?;
                    }
                    if alt > 0 {
                        record(cmd2, main, alt)?;
                    }
                    let t = std::time::Instant::now();
                    if main > 0 {
                        c.submit(cmd, &[], &[], fence)?;
                    }
                    if alt > 0 {
                        let q = cq.queue.lock();
                        d.queue_submit(*q, &[vk::SubmitInfo::default().command_buffers(&[cmd2])], fence2).map_err(|e| format!("submit 2: {e}"))?;
                    }
                    if main > 0 {
                        d.wait_for_fences(&[fence], true, 10_000_000_000).map_err(|e| format!("wait: {e}"))?;
                    }
                    if alt > 0 {
                        d.wait_for_fences(&[fence2], true, 10_000_000_000).map_err(|e| format!("wait 2: {e}"))?;
                    }
                    ts.push(t.elapsed().as_secs_f64() * 1e6);
                    d.reset_fences(&[fence, fence2]).ok();
                }
                ts.sort_by(|a, b| a.total_cmp(b));
                Ok(ts[ts.len() / 2])
            };
            let one_q = wall(N, 0)?;
            let alt_q = wall(0, N)?;
            let two_q = wall(N / 2, N / 2)?;
            crate::info!(
                "dispatch bench: {N} dispatches with a full barrier between, wall time: main queue {one_q:.0} us, compute family {}'s queue {alt_q:.0} us, halves on both {two_q:.0} us ({:.2}x the main queue's)",
                cq.family,
                two_q / one_q
            );
        }
        // Submission boundaries: K command buffers of one dispatch each
        // after a full barrier (front end 2's batches end in a write-back's
        // barriers), submitted as K calls each signaling a timeline value
        // (the batches' way, through `submit_timeline_on`: with the compute
        // family's queue each also signals the main queue's progress), as
        // one call of K submissions each signaling, and as one submission of
        // the K command buffers with one signal; the GPU time from the first
        // one's start to the last one's end against one command buffer with
        // the K dispatches and the barriers between: what a boundary costs.
        {
            const K: usize = 64;
            let mut tci = vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE).initial_value(0);
            let sem = d.create_semaphore(&vk::SemaphoreCreateInfo::default().push_next(&mut tci), None).map_err(|e| format!("timeline semaphore: {e}"))?;
            let cmds = d.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(cpool).command_buffer_count(K as u32)).map_err(|e| format!("cmds: {e}"))?;
            let full = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::MEMORY_WRITE).dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
            for (k, &cb) in cmds.iter().enumerate() {
                d.begin_command_buffer(cb, &vk::CommandBufferBeginInfo::default()).map_err(|e| format!("begin: {e}"))?;
                if k == 0 {
                    d.cmd_reset_query_pool(cb, qp, 0, 2);
                    d.cmd_write_timestamp(cb, vk::PipelineStageFlags::TOP_OF_PIPE, qp, 0);
                }
                d.cmd_pipeline_barrier(cb, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::ALL_COMMANDS, vk::DependencyFlags::empty(), &[full], &[], &[]);
                d.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::COMPUTE, layout, 0, &[set], &[]);
                d.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, pipes[0]);
                let pcs = [iters, (k as u32) * local * groups];
                let bytes: Vec<u8> = pcs.iter().flat_map(|x| x.to_le_bytes()).collect();
                d.cmd_push_constants(cb, layout, vk::ShaderStageFlags::COMPUTE, 0, &bytes);
                d.cmd_dispatch(cb, groups, 1, 1);
                if k == K - 1 {
                    d.cmd_write_timestamp(cb, vk::PipelineStageFlags::BOTTOM_OF_PIPE, qp, 1);
                }
                d.end_command_buffer(cb).map_err(|e| format!("end: {e}"))?;
            }
            let value = std::cell::Cell::new(0u64);
            // 0: K calls; 1: one call of K submissions; 2: one submission.
            let span = |how: u32| -> Result<(f64, f64), String> {
                let mut ts = Vec::new();
                let mut walls = Vec::new();
                for _ in 0..7 {
                    let t = std::time::Instant::now();
                    match how {
                        0 => {
                            for (k, &cb) in cmds.iter().enumerate() {
                                value.set(value.get() + 1);
                                c.submit_timeline_on(false, false, &[cb], &[], &[(sem, value.get())], if k == K - 1 { fence } else { vk::Fence::null() })?;
                            }
                        }
                        1 => {
                            let infos: Vec<(vk::CommandBufferSubmitInfo, vk::SemaphoreSubmitInfo)> = cmds
                                .iter()
                                .map(|&cb| {
                                    value.set(value.get() + 1);
                                    (vk::CommandBufferSubmitInfo::default().command_buffer(cb), vk::SemaphoreSubmitInfo::default().semaphore(sem).value(value.get()).stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS))
                                })
                                .collect();
                            let sis: Vec<vk::SubmitInfo2> = infos.iter().map(|(cbi, ssi)| vk::SubmitInfo2::default().command_buffer_infos(std::slice::from_ref(cbi)).signal_semaphore_infos(std::slice::from_ref(ssi))).collect();
                            let q = c.queue.lock();
                            d.queue_submit2(*q, &sis, fence).map_err(|e| format!("vkQueueSubmit2: {e}"))?;
                        }
                        _ => {
                            value.set(value.get() + 1);
                            c.submit_timeline_on(false, false, &cmds, &[], &[(sem, value.get())], fence)?;
                        }
                    }
                    d.wait_for_fences(&[fence], true, 10_000_000_000).map_err(|e| format!("wait: {e}"))?;
                    walls.push(t.elapsed().as_secs_f64() * 1e6);
                    d.reset_fences(&[fence]).ok();
                    let mut q = [0u64; 2];
                    d.get_query_pool_results(qp, 0, &mut q, vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT).map_err(|e| format!("query: {e}"))?;
                    ts.push((q[1].wrapping_sub(q[0])) as f64 * period / 1000.0);
                }
                ts.sort_by(|a, b| a.total_cmp(b));
                walls.sort_by(|a, b| a.total_cmp(b));
                Ok((ts[ts.len() / 2], walls[walls.len() / 2]))
            };
            let (calls, calls_w) = span(0)?;
            let (subs, subs_w) = span(1)?;
            let (one_sub, one_w) = span(2)?;
            let reference = time_case(K, false, true, false)?;
            let per = |t: f64| (t - reference) / (K - 1) as f64;
            crate::info!(
                "dispatch bench: {K} one-dispatch command buffers behind full barriers, GPU time first to last (wall): {K} calls each signaling {calls:.0} us ({calls_w:.0}; {:.1} us a boundary), one call of {K} signaling submissions {subs:.0} us ({subs_w:.0}; {:.1}), one submission {one_sub:.0} us ({one_w:.0}; {:.1}); one command buffer with barriers between {reference:.0} us",
                per(calls),
                per(subs),
                per(one_sub)
            );
        }
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
