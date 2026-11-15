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
    crate::thread::sampler_priority();
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

/// The hitch flight recorder (`DS_HITCH_TRACE=1`): every 2 ms the title's main thread
/// (`GAME_MainThread`) is sampled whether it runs or waits (its stack's label as in the profile:
/// the HLE call of ours it is in, or its guest code address) with the CPU it used since, and the
/// job workers' CPU summed; the last 4 s are kept. Each hitch line (`gpu::hitch`) is followed by
/// where the main thread was meanwhile (`explain`): what the frame waited for on the title's side,
/// which the hitch report's own causes (our work) leave unexplained.
struct Trace {
    /// (time us, label index, the main thread's CPU cycles since its last sample)
    main: std::collections::VecDeque<(u64, u32, u64)>,
    /// (time us, the job workers' CPU cycles since the last sample)
    workers: std::collections::VecDeque<(u64, u64)>,
    /// The graphics command processor: (time us, its host frames' labels innermost first, its CPU
    /// cycles since its last sample).
    cp: std::collections::VecDeque<(u64, Vec<u32>, u64)>,
    /// Its parallel recording workers (gpu-gfx-par*) that ran since the last tick, alike.
    par: std::collections::VecDeque<(u64, Vec<u32>, u64)>,
    /// The submission thread (`gpu-submit`), alike: where its calls into the driver wait
    /// (FINDINGS.md 2026-11-07: vkQueueSubmit2 blocking 2-34 ms ~6 times a second).
    sub: std::collections::VecDeque<(u64, Vec<u32>, u64)>,
    labels: Vec<String>,
    index: HashMap<String, u32>,
}

impl Trace {
    fn label(&mut self, label: String) -> u32 {
        match self.index.get(&label) {
            Some(&k) => k,
            None => {
                self.labels.push(label.clone());
                self.index.insert(label, (self.labels.len() - 1) as u32);
                (self.labels.len() - 1) as u32
            }
        }
    }
}

static TRACE: Mutex<Option<Trace>> = Mutex::new(None);

/// Starts the flight recorder if `DS_HITCH_TRACE=1`.
pub fn start_trace() {
    if std::env::var("DS_HITCH_TRACE").ok().as_deref() != Some("1") {
        return;
    }
    *TRACE.lock() = Some(Trace { main: Default::default(), workers: Default::default(), cp: Default::default(), par: Default::default(), sub: Default::default(), labels: Vec::new(), index: HashMap::new() });
    let _ = crate::thread::spawn_host("hitch-trace", 1 << 20, Box::new(tracer));
    crate::info!("hitch trace: the title's main thread, the graphics command processor and its parallel workers sampled every 2 ms, explained after each hitch");
}

fn tracer() {
    use windows_sys::Win32::System::Threading::{OpenThread, THREAD_QUERY_LIMITED_INFORMATION};
    use windows_sys::Win32::System::WindowsProgramming::QueryThreadCycleTime;
    crate::log::set_thread_name("hitch-trace");
    crate::thread::sampler_priority();
    let (ours_lo, ours_hi) = crate::exception::host_image();
    let mut handles: HashMap<u32, (windows_sys::Win32::Foundation::HANDLE, u64)> = HashMap::new();
    let mut guest_labels: HashMap<u64, String> = HashMap::new();
    let mut looked = std::time::Instant::now() - std::time::Duration::from_secs(10);
    let mut main_tid = 0u32;
    let mut cp_tid = 0u32;
    let mut par_tids: Vec<u32> = Vec::new();
    let mut sub_tid = 0u32;
    let mut worker_tids: Vec<u32> = Vec::new();
    loop {
        std::thread::sleep(std::time::Duration::from_millis(2));
        // The threads, looked up again every second (they are made at boot).
        if looked.elapsed() > std::time::Duration::from_secs(1) {
            looked = std::time::Instant::now();
            if let Some(t) = crate::thread::THREADS.try_lock() {
                let live: Vec<(u32, String)> = t
                    .iter()
                    .filter(|g| g.exited.try_lock().is_some_and(|e| e.is_none()))
                    .map(|g| (g.host_tid.load(std::sync::atomic::Ordering::Relaxed) as u32, g.name()))
                    .filter(|(tid, _)| *tid != 0)
                    .collect();
                main_tid = live.iter().find(|(_, n)| n == "GAME_MainThread").map_or(0, |x| x.0);
                worker_tids = live.iter().filter(|(_, n)| n.starts_with("BPE JobWorkerThread")).map(|x| x.0).collect();
            }
            let fe = crate::gpu::cpu_threads();
            cp_tid = fe.iter().find(|t| t.0 == "gfx").map_or(0, |t| t.1);
            par_tids = fe.iter().filter(|t| t.0 == "par" && thread_description(t.1).starts_with("gpu-gfx-par")).map(|t| t.1).collect();
            sub_tid = fe.iter().find(|t| t.0 == "submit").map_or(0, |t| t.1);
        }
        let now = crate::hle::kernel::time::process_time_us();
        let mut cycles_of = |tid: u32| -> Option<u64> {
            let e = handles.entry(tid).or_insert_with(|| (unsafe { OpenThread(THREAD_QUERY_LIMITED_INFORMATION, 0, tid) }, 0));
            if e.0.is_null() {
                return None;
            }
            let mut c = 0u64;
            if unsafe { QueryThreadCycleTime(e.0 as _, &mut c) } == 0 {
                return None;
            }
            let used = if e.1 == 0 { 0 } else { c.saturating_sub(e.1) };
            e.1 = c;
            Some(used)
        };
        let workers: u64 = worker_tids.iter().filter_map(|&t| cycles_of(t)).sum();
        let main = if main_tid != 0 { cycles_of(main_tid).map(|used| (used, crate::exception::sample_frames(main_tid, 1))) } else { None };
        let label = main.as_ref().map(|(_, f)| match f {
            Some((host, guest)) => match host.iter().rev().find(|&&a| a >= ours_lo && a < ours_hi) {
                Some(&a) => format!("host {}", crate::exception::frame_label(a)),
                None => match guest.first() {
                    Some(&g) => {
                        let b = g & !0x3F;
                        guest_labels.entry(b).or_insert_with(|| crate::loader::describe_addr(b)).clone()
                    }
                    None => "(no frames)".to_string(),
                },
            },
            None => "(not sampled)".to_string(),
        });
        let cp = if cp_tid != 0 { cycles_of(cp_tid).and_then(|used| crate::exception::sample_frames(cp_tid, 0).map(|(host, _)| (used, host))) } else { None };
        // (Other modules' frames by their exports: the system call a frame is in.)
        let name = |a: u64| if a >= ours_lo && a < ours_hi { crate::exception::frame_label(a) } else { crate::exception::export_label(a) };
        let cp_labels: Option<(u64, Vec<String>)> = cp.map(|(used, host)| (used, host.iter().map(|&a| name(a)).collect()));
        let sub = if sub_tid != 0 { cycles_of(sub_tid).and_then(|used| crate::exception::sample_frames(sub_tid, 0).map(|(host, _)| (used, host))) } else { None };
        let sub_labels: Option<(u64, Vec<String>)> = sub.map(|(used, host)| (used, host.iter().map(|&a| name(a)).collect()));
        // (The workers that ran since the last tick only: idle ones cost no suspension.)
        let par_labels: Vec<(u64, Vec<String>)> = par_tids
            .iter()
            .filter_map(|&tid| {
                let used = cycles_of(tid)?;
                if used == 0 {
                    return None;
                }
                let (host, _) = crate::exception::sample_frames(tid, 0)?;
                Some((used, host.iter().map(|&a| name(a)).collect()))
            })
            .collect();
        let mut g = TRACE.lock();
        let Some(t) = g.as_mut() else { continue };
        if let (Some((used, _)), Some(label)) = (main, label) {
            let k = t.label(label);
            t.main.push_back((now, k, used));
        }
        if let Some((used, labels)) = cp_labels {
            let ks: Vec<u32> = labels.into_iter().map(|l| t.label(l)).collect();
            t.cp.push_back((now, ks, used));
        }
        for (used, labels) in par_labels {
            let ks: Vec<u32> = labels.into_iter().map(|l| t.label(l)).collect();
            t.par.push_back((now, ks, used));
        }
        if let Some((used, labels)) = sub_labels {
            let ks: Vec<u32> = labels.into_iter().map(|l| t.label(l)).collect();
            t.sub.push_back((now, ks, used));
        }
        t.workers.push_back((now, workers));
        let old = now.saturating_sub(4_000_000);
        while t.main.front().is_some_and(|x| x.0 < old) {
            t.main.pop_front();
        }
        while t.cp.front().is_some_and(|x| x.0 < old) {
            t.cp.pop_front();
        }
        while t.par.front().is_some_and(|x| x.0 < old) {
            t.par.pop_front();
        }
        while t.sub.front().is_some_and(|x| x.0 < old) {
            t.sub.pop_front();
        }
        while t.workers.front().is_some_and(|x| x.0 < old) {
            t.workers.pop_front();
        }
    }
}

/// Where the title's main thread was from `lo` to `hi` (us, `gpu::hitch::now_us`): its labels by
/// samples (2 ms each), each split into running and waiting by the CPU it used, and the job
/// workers' CPU meanwhile.
pub fn explain(lo: u64, hi: u64) -> Option<String> {
    let g = TRACE.lock();
    let t = g.as_ref()?;
    let mut by: HashMap<u32, (u32, u32)> = HashMap::new();
    let mut n = 0u32;
    // A sample counts as running when the thread used at least a quarter of the period.
    let busy_cycles = (crate::gpu::threadcpu::tsc_hz() / 2000.0) as u64;
    for &(at, k, used) in t.main.iter().filter(|x| x.0 > lo && x.0 <= hi) {
        n += 1;
        let e = by.entry(k).or_default();
        if used >= busy_cycles {
            e.0 += 1;
        } else {
            e.1 += 1;
        }
        let _ = at;
    }
    if n == 0 {
        return None;
    }
    let mut v: Vec<(u32, (u32, u32))> = by.into_iter().collect();
    v.sort_by(|a, b| (b.1 .0 + b.1 .1).cmp(&(a.1 .0 + a.1 .1)));
    let running: u32 = v.iter().map(|x| x.1 .0).sum();
    let parts: Vec<String> = v
        .iter()
        .take(6)
        .map(|(k, (r, w))| format!("{} {}% ({}{})", t.labels[*k as usize], (r + w) * 100 / n, if *r > 0 { format!("{r} running") } else { String::new() }, if *w > 0 { format!("{}{w} waiting", if *r > 0 { ", " } else { "" }) } else { String::new() }))
        .collect();
    let wcycles: u64 = t.workers.iter().filter(|x| x.0 > lo && x.0 <= hi).map(|x| x.1).sum();
    let secs = (hi - lo) as f64 / 1e6;
    Some(format!(
        "main thread {} samples, running in {}%: {}; job workers {:.1} logical CPUs",
        n,
        running * 100 / n,
        parts.join("; "),
        crate::gpu::threadcpu::ms(wcycles) / 1000.0 / secs.max(1e-3)
    ))
}

/// Where the graphics command processor was from `lo` to `hi` (us): the functions of ours on its
/// stack by the share of samples they were in (those in nearly every sample left out: its loop),
/// and its innermost frames while it ran.
pub fn explain_cp(lo: u64, hi: u64) -> Option<String> {
    let g = TRACE.lock();
    let t = g.as_ref()?;
    explain_samples(t, &t.cp, lo, hi, "graphics CP")
}

/// The graphics CP's parallel workers from `lo` to `hi` (us), as `explain_cp` (their samples
/// together: those that ran).
pub fn explain_par(lo: u64, hi: u64) -> Option<String> {
    let g = TRACE.lock();
    let t = g.as_ref()?;
    explain_samples(t, &t.par, lo, hi, "parallel workers")
}

/// The submission thread from `lo` to `hi` (us), as `explain_cp`.
pub fn explain_submit(lo: u64, hi: u64) -> Option<String> {
    let g = TRACE.lock();
    let t = g.as_ref()?;
    explain_samples(t, &t.sub, lo, hi, "submission thread")
}

/// A thread's description (its name as `thread::spawn_host` set it), or "".
pub(crate) fn thread_description(tid: u32) -> String {
    use windows_sys::Win32::System::Threading::{GetThreadDescription, OpenThread, THREAD_QUERY_LIMITED_INFORMATION};
    unsafe {
        let h = OpenThread(THREAD_QUERY_LIMITED_INFORMATION, 0, tid);
        if h.is_null() {
            return String::new();
        }
        let mut p: windows_sys::core::PWSTR = std::ptr::null_mut();
        let ok = GetThreadDescription(h, &mut p) >= 0 && !p.is_null();
        windows_sys::Win32::Foundation::CloseHandle(h);
        if !ok {
            return String::new();
        }
        let mut n = 0;
        while *p.add(n) != 0 {
            n += 1;
        }
        let s = String::from_utf16_lossy(std::slice::from_raw_parts(p, n));
        windows_sys::Win32::Foundation::LocalFree(p as _);
        s
    }
}

fn explain_samples(t: &Trace, all: &std::collections::VecDeque<(u64, Vec<u32>, u64)>, lo: u64, hi: u64, who: &str) -> Option<String> {
    let busy_cycles = (crate::gpu::threadcpu::tsc_hz() / 2000.0) as u64;
    let samples: Vec<&(u64, Vec<u32>, u64)> = all.iter().filter(|x| x.0 > lo && x.0 <= hi).collect();
    let n = samples.len() as u32;
    if n == 0 {
        return None;
    }
    let ours = |k: u32| {
        let l = &t.labels[k as usize];
        (l.starts_with("dsport::") || l.starts_with("rdna::")) && !l.contains("parking_lot")
    };
    let mut incl: HashMap<u32, u32> = HashMap::new();
    let mut leaf: HashMap<u64, u32> = HashMap::new();
    let mut waits: HashMap<u32, u32> = HashMap::new();
    let mut running = 0u32;
    for (_, ks, used) in &samples {
        let mut seen: Vec<u32> = ks.iter().copied().filter(|&k| ours(k)).collect();
        seen.sort_unstable();
        seen.dedup();
        for k in seen {
            *incl.entry(k).or_default() += 1;
        }
        if *used >= busy_cycles {
            running += 1;
            // The innermost function of ours and what it was calling (the
            // module or function; itself when in its own code).
            if let Some(i) = ks.iter().position(|&k| ours(k)) {
                *leaf.entry(((ks[i] as u64) << 32) | ks[i.saturating_sub(1)] as u64).or_default() += 1;
            } else if let Some(&k) = ks.first() {
                *leaf.entry(((k as u64) << 32) | k as u64).or_default() += 1;
            }
        } else if let Some(&k) = ks.iter().find(|&&k| ours(k)) {
            // Waiting: in the innermost function of ours.
            *waits.entry(k).or_default() += 1;
        }
    }
    let mut v: Vec<(u32, u32)> = incl.into_iter().filter(|&(_, c)| c * 100 < n * 97).collect();
    v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let short = |k: u32| {
        let l = &t.labels[k as usize];
        l.strip_prefix("dsport::").unwrap_or(l).to_string()
    };
    let parts: Vec<String> = v.iter().take(14).map(|&(k, c)| format!("{} {}%", short(k), c * 100 / n)).collect();
    let mut lv: Vec<(u64, u32)> = leaf.into_iter().collect();
    lv.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let leaves: Vec<String> = lv
        .iter()
        .take(8)
        .map(|&(k, c)| {
            let (f, callee) = ((k >> 32) as u32, k as u32);
            if f == callee { format!("{} {c}", short(f)) } else { format!("{} > {} {c}", short(f), short(callee)) }
        })
        .collect();
    let mut wv: Vec<(u32, u32)> = waits.into_iter().collect();
    wv.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let waiting: Vec<String> = wv.iter().take(5).map(|&(k, c)| format!("{} {}", short(k), c)).collect();
    Some(format!("{who} {} samples, running in {}%; in: {}; innermost while running: {}; waiting in: {}", n, running * 100 / n, parts.join(", "), leaves.join(", "), waiting.join(", ")))
}
