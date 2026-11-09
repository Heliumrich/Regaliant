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
    if let Some(secs) = std::env::var("DS_SUBMIT_BENCH").ok().and_then(|v| v.parse::<u64>().ok()) {
        return match submit_bench(secs) {
            Ok(()) => 0,
            Err(e) => {
                crate::error!("submit bench: {e}");
                2
            }
        };
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

/// `DS_SUBMIT_BENCH=<seconds>` (with `--dispatch-bench`): ~1,000 one-command submissions a second
/// to the main queue for that long, no title running, each call timed: whether the driver's
/// submissions of 2 ms or more (in game, ~5 a second ~170 ms apart, FINDINGS.md 2026-11-07) come
/// without the title and front end 2 around them, and their spacing.
fn submit_bench(secs: u64) -> Result<(), String> {
    let c = ctx();
    let d = &c.device;
    let (buf, _mem) = c.device_buffer(4096, vk::BufferUsageFlags::TRANSFER_DST)?;
    const K: usize = 64;
    let (cmds, fences) = unsafe {
        let pool = d.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(c.queue_family), None).map_err(|e| format!("pool: {e}"))?;
        let cmds = d.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(pool).command_buffer_count(K as u32)).map_err(|e| format!("cmds: {e}"))?;
        for &cb in &cmds {
            d.begin_command_buffer(cb, &vk::CommandBufferBeginInfo::default()).map_err(|e| format!("begin: {e}"))?;
            d.cmd_fill_buffer(cb, buf, 0, 4, 0);
            d.end_command_buffer(cb).map_err(|e| format!("end: {e}"))?;
        }
        let fences: Vec<vk::Fence> = (0..K).map(|_| d.create_fence(&vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED), None)).collect::<Result<_, _>>().map_err(|e| format!("fence: {e}"))?;
        (cmds, fences)
    };
    // `DS_SUBMIT_BENCH_ALLOCS=<n>`: n allocations of 1 MiB held meanwhile, half device-local and
    // half host-visible (the game holds thousands: imports, slabs, the mirror's blocks): whether
    // the driver's work per submission or its periodic passes grow with them.
    let n_allocs = std::env::var("DS_SUBMIT_BENCH_ALLOCS").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(0);
    let mut held = Vec::with_capacity(n_allocs);
    for i in 0..n_allocs {
        let r = if i % 2 == 0 { c.device_buffer(1 << 20, vk::BufferUsageFlags::TRANSFER_DST).map(|(b, m)| (b, m)) } else { c.host_buffer(1 << 20, vk::BufferUsageFlags::TRANSFER_SRC).map(|(b, m, _)| (b, m)) };
        match r {
            Ok(x) => held.push(x),
            Err(e) => {
                crate::warn!("submit bench: allocation {i} failed: {e}");
                break;
            }
        }
    }
    if n_allocs > 0 {
        crate::info!("submit bench: {} allocations of 1 MiB held", held.len());
    }
    let start = std::time::Instant::now();
    let end = start + std::time::Duration::from_secs(secs);
    // `DS_SUBMIT_BENCH_LOAD=1`: a second thread keeps the GPU busy in bursts of varying length
    // (fills of a 512 MiB buffer, 0-30 ms apart), as a game's load varies: whether the driver's
    // clock and power changes under it stall submissions.
    let load = std::env::var("DS_SUBMIT_BENCH_LOAD").ok().as_deref() == Some("1");
    let loader = load.then(|| {
        std::thread::spawn(move || -> Result<u64, String> {
            let c = ctx();
            let d = &c.device;
            let size = 512u64 << 20;
            let (big, _bmem) = c.device_buffer(size, vk::BufferUsageFlags::TRANSFER_DST)?;
            let (cmd, fence) = unsafe {
                let pool = d.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(c.queue_family), None).map_err(|e| format!("pool: {e}"))?;
                let cmd = d.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(pool).command_buffer_count(1)).map_err(|e| format!("cmd: {e}"))?[0];
                d.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default()).map_err(|e| format!("begin: {e}"))?;
                for _ in 0..8 {
                    d.cmd_fill_buffer(cmd, big, 0, size, 0x5a5a_5a5a);
                }
                d.end_command_buffer(cmd).map_err(|e| format!("end: {e}"))?;
                (cmd, d.create_fence(&vk::FenceCreateInfo::default(), None).map_err(|e| format!("fence: {e}"))?)
            };
            let mut bursts = 0u64;
            let mut x = 0x9E37_79B9u32;
            while std::time::Instant::now() < end {
                unsafe { d.reset_fences(&[fence]).map_err(|e| format!("reset: {e}"))? };
                c.submit(cmd, &[], &[], fence)?;
                unsafe { d.wait_for_fences(&[fence], true, u64::MAX).map_err(|e| format!("wait: {e}"))? };
                bursts += 1;
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                std::thread::sleep(std::time::Duration::from_millis((x % 31) as u64));
            }
            Ok(bursts)
        })
    });
    // `DS_SUBMIT_BENCH_US=<us>`: the pause between submissions (default 900; 0: none, the 64
    // fences pace them): whether the driver's slow submissions come per time or per number.
    let pause = std::env::var("DS_SUBMIT_BENCH_US").ok().and_then(|v| v.parse::<u64>().ok()).unwrap_or(900);
    // `DS_SUBMIT_BENCH_TL=1`: each submission signals a timeline semaphore with its number and
    // waits for the one before (as front end 2's batches do).
    let tl = (std::env::var("DS_SUBMIT_BENCH_TL").ok().as_deref() == Some("1"))
        .then(|| {
            let mut tci = vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE).initial_value(0);
            unsafe { d.create_semaphore(&vk::SemaphoreCreateInfo::default().push_next(&mut tci), None) }.map_err(|e| format!("timeline: {e}"))
        })
        .transpose()?;
    // `DS_SUBMIT_BENCH_SHAPE=<shape>`: what one submission holds: `cmds8` (eight command buffers
    // in one submit info), `infos8` (eight submit infos of one command buffer in one call),
    // `nofence` (one command buffer, no fence: paced by a timeline semaphore it signals): which
    // count the driver's slow submissions follow.
    let shape = std::env::var("DS_SUBMIT_BENCH_SHAPE").unwrap_or_default();
    let pace = (shape == "nofence")
        .then(|| {
            let mut tci = vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE).initial_value(0);
            unsafe { d.create_semaphore(&vk::SemaphoreCreateInfo::default().push_next(&mut tci), None) }.map_err(|e| format!("timeline: {e}"))
        })
        .transpose()?;
    // `DS_SUBMIT_BENCH_SLOW_US=<us>`: the duration counted as slow (default 2000).
    let slow_us = std::env::var("DS_SUBMIT_BENCH_SLOW_US").ok().and_then(|v| v.parse::<u128>().ok()).unwrap_or(2000);
    let alt = if shape == "alt_queue" {
        let cq = c.compute.as_ref().ok_or("no compute queue")?;
        let (cbuf, _cmem) = c.device_buffer(4096, vk::BufferUsageFlags::TRANSFER_DST)?;
        std::mem::forget(_cmem);
        unsafe {
            let pool = d.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(cq.family), None).map_err(|e| format!("pool: {e}"))?;
            let ccmds = d.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(pool).command_buffer_count(K as u32)).map_err(|e| format!("cmds: {e}"))?;
            for &cb in &ccmds {
                d.begin_command_buffer(cb, &vk::CommandBufferBeginInfo::default()).map_err(|e| format!("begin: {e}"))?;
                d.cmd_fill_buffer(cb, cbuf, 0, 4, 0);
                d.end_command_buffer(cb).map_err(|e| format!("end: {e}"))?;
            }
            let cfences: Vec<vk::Fence> = (0..K).map(|_| d.create_fence(&vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED), None)).collect::<Result<_, _>>().map_err(|e| format!("fence: {e}"))?;
            Some((&cq.queue, ccmds, cfences))
        }
    } else {
        None
    };
    let mut slow_calls: Vec<(u64, u8)> = Vec::new();
    let mut durs: Vec<u32> = Vec::with_capacity(1 << 20);
    let (mut n, mut slow) = (0u64, Vec::new());
    let mut worst = 0u128;
    while std::time::Instant::now() < end {
        let k = (n as usize) % K;
        if let Some(sem) = pace {
            // Command buffer k is free once submission n - K completed.
            if n >= K as u64 {
                let v = [n - K as u64 + 1];
                let sems = [sem];
                unsafe { d.wait_semaphores(&vk::SemaphoreWaitInfo::default().semaphores(&sems).values(&v), u64::MAX) }.map_err(|e| format!("wait: {e}"))?;
            }
            let t0 = std::time::Instant::now();
            // (Raw: `submit_timeline` notes slow calls as hitches, which needs the title's clock.)
            let cbs = [vk::CommandBufferSubmitInfo::default().command_buffer(cmds[k])];
            let sig = [vk::SemaphoreSubmitInfo::default().semaphore(sem).value(n + 1).stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
            let info = [vk::SubmitInfo2::default().command_buffer_infos(&cbs).signal_semaphore_infos(&sig)];
            let q = c.queue.lock();
            unsafe { d.queue_submit2(*q, &info, vk::Fence::null()) }.map_err(|e| format!("submit: {e}"))?;
            drop(q);
            let us = t0.elapsed().as_micros();
            worst = worst.max(us);
                durs.push(us as u32);
            if us >= slow_us {
                slow.push((start.elapsed().as_secs_f64() * 1000.0, us));
            }
            n += 1;
            if pause > 0 {
                std::thread::sleep(std::time::Duration::from_micros(pause));
            }
            continue;
        }
        if shape == "empty_half" || shape == "alt_queue" {
            // Every other call: an empty submission to the main queue (`empty_half`), or a fill on
            // the compute family's queue (`alt_queue`): whether those calls count toward the
            // driver's slow one.
            let t0 = std::time::Instant::now();
            if n % 2 == 0 {
                unsafe {
                    d.wait_for_fences(&[fences[k]], true, u64::MAX).map_err(|e| format!("wait: {e}"))?;
                    d.reset_fences(&[fences[k]]).map_err(|e| format!("reset: {e}"))?;
                }
                let t0b = std::time::Instant::now();
                c.submit(cmds[k], &[], &[], fences[k])?;
                let us = t0b.elapsed().as_micros();
                worst = worst.max(us);
                durs.push(us as u32);
                if us >= slow_us {
                    slow.push((start.elapsed().as_secs_f64() * 1000.0, us));
                    slow_calls.push((n, 0u8));
                }
            } else if shape == "empty_half" {
                let info = [vk::SubmitInfo2::default()];
                let q = c.queue.lock();
                unsafe { d.queue_submit2(*q, &info, vk::Fence::null()) }.map_err(|e| format!("submit: {e}"))?;
                drop(q);
                let us = t0.elapsed().as_micros();
                worst = worst.max(us);
                durs.push(us as u32);
                if us >= slow_us {
                    slow.push((start.elapsed().as_secs_f64() * 1000.0, us));
                    slow_calls.push((n, 1u8));
                }
            } else {
                let (cq, ccmds, cfences) = alt.as_ref().ok_or("no compute queue")?;
                unsafe {
                    d.wait_for_fences(&[cfences[k]], true, u64::MAX).map_err(|e| format!("wait: {e}"))?;
                    d.reset_fences(&[cfences[k]]).map_err(|e| format!("reset: {e}"))?;
                }
                let t0b = std::time::Instant::now();
                let cbs = [vk::CommandBufferSubmitInfo::default().command_buffer(ccmds[k])];
                let info = [vk::SubmitInfo2::default().command_buffer_infos(&cbs)];
                let q = cq.lock();
                unsafe { d.queue_submit2(*q, &info, cfences[k]) }.map_err(|e| format!("submit: {e}"))?;
                drop(q);
                let us = t0b.elapsed().as_micros();
                worst = worst.max(us);
                durs.push(us as u32);
                if us >= slow_us {
                    slow.push((start.elapsed().as_secs_f64() * 1000.0, us));
                    slow_calls.push((n, 2u8));
                }
            }
            n += 1;
            if pause > 0 {
                std::thread::sleep(std::time::Duration::from_micros(pause));
            }
            continue;
        }
        if shape == "cmds8" || shape == "infos8" {
            // Eight command buffers a submission: the fence of group k / 8 guards them.
            if k % 8 != 0 {
                n += 1;
                continue;
            }
            unsafe {
                d.wait_for_fences(&[fences[k]], true, u64::MAX).map_err(|e| format!("wait: {e}"))?;
                d.reset_fences(&[fences[k]]).map_err(|e| format!("reset: {e}"))?;
            }
            let group = &cmds[k..k + 8];
            let t0 = std::time::Instant::now();
            if shape == "cmds8" {
                c.submit_cmds(group, &[], &[], fences[k])?;
            } else {
                let cbs: Vec<[vk::CommandBufferSubmitInfo; 1]> = group.iter().map(|&cb| [vk::CommandBufferSubmitInfo::default().command_buffer(cb)]).collect();
                let infos: Vec<vk::SubmitInfo2> = cbs.iter().map(|cb| vk::SubmitInfo2::default().command_buffer_infos(cb)).collect();
                let q = c.queue.lock();
                unsafe { d.queue_submit2(*q, &infos, fences[k]) }.map_err(|e| format!("submit: {e}"))?;
            }
            let us = t0.elapsed().as_micros();
            worst = worst.max(us);
                durs.push(us as u32);
            if us >= slow_us {
                slow.push((start.elapsed().as_secs_f64() * 1000.0, us));
            }
            n += 1;
            if pause > 0 {
                std::thread::sleep(std::time::Duration::from_micros(pause));
            }
            continue;
        }
        unsafe {
            d.wait_for_fences(&[fences[k]], true, u64::MAX).map_err(|e| format!("wait: {e}"))?;
            d.reset_fences(&[fences[k]]).map_err(|e| format!("reset: {e}"))?;
        }
        let t0 = std::time::Instant::now();
        match tl {
            Some(sem) => {
                let waits: Vec<(vk::Semaphore, u64, vk::PipelineStageFlags2)> = if n > 0 { vec![(sem, n, vk::PipelineStageFlags2::ALL_COMMANDS)] } else { Vec::new() };
                let cbs = [vk::CommandBufferSubmitInfo::default().command_buffer(cmds[k])];
                let ws: Vec<vk::SemaphoreSubmitInfo> = waits.iter().map(|&(s2, v, st)| vk::SemaphoreSubmitInfo::default().semaphore(s2).value(v).stage_mask(st)).collect();
                let sig = [vk::SemaphoreSubmitInfo::default().semaphore(sem).value(n + 1).stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
                let info = [vk::SubmitInfo2::default().command_buffer_infos(&cbs).wait_semaphore_infos(&ws).signal_semaphore_infos(&sig)];
                let q = c.queue.lock();
                unsafe { d.queue_submit2(*q, &info, fences[k]) }.map_err(|e| format!("submit: {e}"))?;
            }
            None => c.submit(cmds[k], &[], &[], fences[k])?,
        }
        let us = t0.elapsed().as_micros();
        worst = worst.max(us);
                durs.push(us as u32);
        if us >= slow_us {
            slow.push((start.elapsed().as_secs_f64() * 1000.0, us));
        }
        n += 1;
        if pause > 0 {
            std::thread::sleep(std::time::Duration::from_micros(pause));
        }
    }
    if let Some(h) = loader {
        match h.join() {
            Ok(Ok(b)) => crate::info!("submit bench: the load thread ran {b} bursts"),
            Ok(Err(e)) => crate::warn!("submit bench: the load thread failed: {e}"),
            Err(_) => crate::warn!("submit bench: the load thread panicked"),
        }
    }
    unsafe { d.queue_wait_idle(*c.queue.lock()).ok() };
    let gaps: Vec<f64> = slow.windows(2).map(|w| w[1].0 - w[0].0).collect();
    let mut sorted = gaps.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = |f: f64| sorted.get(((sorted.len() as f64 - 1.0) * f) as usize).copied().unwrap_or(0.0);
    crate::info!(
        "submit bench: {n} submissions in {:.1} s ({:.0} a second), {} took {slow_us} us or more (worst {} us); gaps between them ms: p10 {:.0}, p50 {:.0}, p90 {:.0}",
        start.elapsed().as_secs_f64(),
        n as f64 / start.elapsed().as_secs_f64(),
        slow.len(),
        worst,
        q(0.1),
        q(0.5),
        q(0.9)
    );
    if !durs.is_empty() {
        let mut v = durs.clone();
        v.sort_unstable();
        let mean = v.iter().map(|&x| x as f64).sum::<f64>() / v.len() as f64;
        crate::info!("submit bench: {} timed calls: mean {:.1} us, p10 {} us, p50 {} us, p90 {} us, p99 {} us", v.len(), mean, v[v.len() / 10], v[v.len() / 2], v[v.len() * 9 / 10], v[v.len() * 99 / 100]);
    }
    if !slow_calls.is_empty() {
        let v: Vec<String> = slow_calls.iter().take(24).map(|(i, w)| format!("{i}{}", ["m", "e", "c"][*w as usize])).collect();
        crate::info!("submit bench: slow calls by index (m main, e empty, c compute): {}", v.join(" "));
    }
    for (t, us) in slow.iter().take(40) {
        crate::info!("submit bench: slow submission at {t:.0} ms: {us} us");
    }
    Ok(())
}

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
