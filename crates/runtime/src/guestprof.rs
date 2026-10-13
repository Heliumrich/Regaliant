//! Sampling profile of the title's threads (`DS_PROFILE_GUEST=1`): every
//! 5 ms each guest thread that ran since the last look is sampled once and
//! the sample weighted by the CPU time it used since (thread cycle counts):
//! in guest code by address (64-byte buckets), in our code by the outermost
//! host function of ours (the HLE call it is in). `profile reset` /
//! `profile report` delimit windows. It answers where the title's CPU time
//! goes on the host (its job workers keep ~12 logical CPUs busy at the
//! gate).

use parking_lot::Mutex;
use std::collections::HashMap;

/// CPU cycles by (thread kind, label) and by thread kind since `since`.
struct Profile {
    labels: HashMap<(String, String), u64>,
    kinds: HashMap<String, u64>,
    since: std::time::Instant,
}

static PROFILE: Mutex<Option<Profile>> = Mutex::new(None);

/// Starts the sampler if `DS_PROFILE_GUEST=1`.
pub fn start() {
    if std::env::var("DS_PROFILE_GUEST").ok().as_deref() != Some("1") {
        return;
    }
    *PROFILE.lock() = Some(Profile { labels: HashMap::new(), kinds: HashMap::new(), since: std::time::Instant::now() });
    let _ = crate::thread::spawn_host("guest-profiler", 1 << 20, Box::new(sampler));
    crate::info!("guest profiler: sampling the title's threads every 5 ms");
}

/// A thread's kind: its name without trailing digits or a CPU suffix.
fn kind_of(name: &str) -> String {
    let n = name.trim_end_matches(|c: char| c.is_ascii_digit());
    n.strip_suffix(" CPU").unwrap_or(n).trim_end().to_string()
}

fn sampler() {
    use windows_sys::Win32::System::Threading::{OpenThread, THREAD_QUERY_LIMITED_INFORMATION};
    use windows_sys::Win32::System::WindowsProgramming::QueryThreadCycleTime;
    crate::log::set_thread_name("guest-profiler");
    let (ours_lo, ours_hi) = crate::exception::host_image();
    // host tid -> (handle, last cycle count)
    let mut handles: HashMap<u32, (windows_sys::Win32::Foundation::HANDLE, u64)> = HashMap::new();
    let mut guest_labels: HashMap<u64, String> = HashMap::new();
    loop {
        std::thread::sleep(std::time::Duration::from_millis(5));
        let threads: Vec<(u32, String)> = match crate::thread::THREADS.try_lock() {
            Some(t) => t
                .iter()
                .filter(|g| g.exited.try_lock().is_some_and(|e| e.is_none()))
                .map(|g| (g.host_tid.load(std::sync::atomic::Ordering::Relaxed) as u32, g.name()))
                .filter(|(tid, _)| *tid != 0)
                .collect(),
            None => continue,
        };
        let mut samples: Vec<(String, String, u64)> = Vec::with_capacity(threads.len());
        for (tid, name) in threads {
            let e = handles.entry(tid).or_insert_with(|| (unsafe { OpenThread(THREAD_QUERY_LIMITED_INFORMATION, 0, tid) }, 0));
            if e.0.is_null() {
                continue;
            }
            let mut cycles = 0u64;
            if unsafe { QueryThreadCycleTime(e.0 as _, &mut cycles) } == 0 {
                continue;
            }
            let used = if e.1 == 0 { 0 } else { cycles.saturating_sub(e.1) };
            e.1 = cycles;
            if used == 0 {
                continue;
            }
            let Some((host, guest)) = crate::exception::sample_frames(tid, 1) else { continue };
            let label = match host.iter().rev().find(|&&a| a >= ours_lo && a < ours_hi) {
                // In our code: the outermost function of ours (the HLE call).
                Some(&a) => format!("host {}", crate::exception::frame_label(a)),
                None => match guest.first() {
                    Some(&g) => {
                        let b = g & !0x3F;
                        guest_labels.entry(b).or_insert_with(|| crate::loader::describe_addr(b)).clone()
                    }
                    None => "(no frames)".to_string(),
                },
            };
            samples.push((kind_of(&name), label, used));
        }
        let mut g = PROFILE.lock();
        let Some(p) = g.as_mut() else { continue };
        for (kind, label, used) in samples {
            *p.kinds.entry(kind.clone()).or_default() += used;
            *p.labels.entry((kind, label)).or_default() += used;
        }
    }
}

/// Clears the samples (`profile reset`).
pub fn reset() {
    if let Some(p) = PROFILE.lock().as_mut() {
        p.labels.clear();
        p.kinds.clear();
        p.since = std::time::Instant::now();
    }
}

/// The window's profile (`profile report`): CPU per thread kind (logical
/// CPUs' worth over the window), then the CPU time by label overall and for
/// the busiest kinds.
pub fn report() -> String {
    let g = PROFILE.lock();
    let Some(p) = g.as_ref() else { return String::new() };
    let secs = p.since.elapsed().as_secs_f64().max(1e-3);
    let cpus = |c: u64| crate::gpu::threadcpu::ms(c) / 1000.0 / secs;
    let total: u64 = p.kinds.values().sum::<u64>().max(1);
    let mut s = format!("    guest profile ({:.1} s; the title's threads' CPU, sampled every 5 ms and weighted by the CPU used): {:.2} logical CPUs in all\n", secs, cpus(total));
    let mut kinds: Vec<(&String, &u64)> = p.kinds.iter().collect();
    kinds.sort_by(|a, b| b.1.cmp(a.1));
    for (k, c) in kinds.iter().take(16) {
        s.push_str(&format!("      {k}: {:.2} logical CPUs\n", cpus(**c)));
    }
    let mut all: HashMap<&String, u64> = HashMap::new();
    for ((_, l), c) in &p.labels {
        *all.entry(l).or_default() += c;
    }
    let mut v: Vec<(&String, u64)> = all.into_iter().collect();
    v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
    s.push_str("    guest profile by label (share of the title's CPU):\n");
    for (l, c) in v.iter().take(40) {
        s.push_str(&format!("      {:5.1}%  {l}\n", *c as f64 * 100.0 / total as f64));
    }
    for (k, kc) in kinds.iter().take(3) {
        let mut v: Vec<(&String, u64)> = p.labels.iter().filter(|((kk, _), _)| kk == *k).map(|((_, l), c)| (l, *c)).collect();
        v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
        s.push_str(&format!("    guest profile {k} by label (share of its CPU):\n"));
        for (l, c) in v.iter().take(16) {
            s.push_str(&format!("      {:5.1}%  {l}\n", *c as f64 * 100.0 / (**kc).max(1) as f64));
        }
    }
    s
}
