//! Driver statistics of compiled pipelines (`DS_PIPELINE_STATS=1`).
//!
//! With VK_KHR_pipeline_executable_properties enabled, pipelines are created
//! with CAPTURE_STATISTICS and each new one logs what the driver reports per
//! executable (register use, spills, instruction counts; names and meaning
//! are the driver's), labelled with the guest shader addresses. The flag
//! changes the pipeline cache key, so such runs compile their pipelines anew.

use super::vk::ctx;
use ash::vk;
use std::ffi::CStr;

/// Whether statistics are captured (extension enabled at device creation).
pub fn enabled() -> bool {
    ctx().exec_props.is_some()
}

/// Creation flags for captured statistics.
pub fn flags() -> vk::PipelineCreateFlags {
    if enabled() {
        vk::PipelineCreateFlags::CAPTURE_STATISTICS_KHR
    } else {
        vk::PipelineCreateFlags::empty()
    }
}

/// Logs the statistics of each executable of `pipeline`.
pub fn report(pipeline: vk::Pipeline, label: &str) {
    let Some(ep) = ctx().exec_props.as_ref() else { return };
    let info = vk::PipelineInfoKHR::default().pipeline(pipeline);
    let execs = match unsafe { ep.get_pipeline_executable_properties(&info) } {
        Ok(v) => v,
        Err(e) => {
            crate::warn!("pipeline stats {label}: {e}");
            return;
        }
    };
    for (i, x) in execs.iter().enumerate() {
        let name = unsafe { CStr::from_ptr(x.name.as_ptr()) }.to_string_lossy().into_owned();
        let ei = vk::PipelineExecutableInfoKHR::default().pipeline(pipeline).executable_index(i as u32);
        let stats = unsafe { ep.get_pipeline_executable_statistics(&ei) }.unwrap_or_default();
        let list: Vec<String> = stats
            .iter()
            .map(|s| {
                let n = unsafe { CStr::from_ptr(s.name.as_ptr()) }.to_string_lossy();
                let v = unsafe {
                    match s.format {
                        vk::PipelineExecutableStatisticFormatKHR::BOOL32 => (s.value.b32 != 0).to_string(),
                        vk::PipelineExecutableStatisticFormatKHR::INT64 => s.value.i64.to_string(),
                        vk::PipelineExecutableStatisticFormatKHR::UINT64 => s.value.u64.to_string(),
                        _ => format!("{:.3}", s.value.f64),
                    }
                };
                format!("{n}={v}")
            })
            .collect();
        crate::info!("pipeline stats {label} [{name}, subgroup {}]: {}", x.subgroup_size, list.join(", "));
    }
}
