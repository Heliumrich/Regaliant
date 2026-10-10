//! GPU emulation front end for the PS5 (AGC): submission queues, the gfx10
//! PM4 command processor (`cp`) and flips.
//!
//! Command buffers submitted through libSceAgcDriver are queued per hardware
//! queue (graphics, or a compute queue id) and executed in submission order
//! by one command-processor thread per queue, concurrently with the CPU, as
//! on the console. Dispatches run on Vulkan with `DS_GPU=1` (`exec`); draws
//! are decoded but not rendered yet (docs/STATUS.md).

pub mod capture;
pub mod cp;
pub mod desc;
pub mod emucheck;
/// Run-time switches for A/B measurements within one run (`tune <name>
/// <value>`; slowly varying load from other GPU users cancels out), each
/// seeded from its environment variable.
pub mod tune {
    use std::sync::atomic::{AtomicU64, Ordering};

    struct Tunable {
        name: &'static str,
        env: &'static str,
        default: u64,
        value: AtomicU64,
    }

    const UNSET: u64 = u64::MAX;

    static ALL: [Tunable; 9] = [
        Tunable { name: "promises", env: "DS_PROMISES", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "lazy", env: "DS_LAZY_BARRIERS", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "shadow_min", env: "DS_GPU_SHADOW_MIN", default: 64 << 10, value: AtomicU64::new(UNSET) },
        Tunable { name: "mirrors", env: "DS_MIRRORS", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "ring_vram", env: "DS_RING_VRAM", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "snap_max", env: "DS_SNAP_MAX", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "snap_index", env: "DS_SNAP_INDEX", default: 4 << 20, value: AtomicU64::new(UNSET) },
        Tunable { name: "ring_lo", env: "DS_RING_LO", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "ring_hi", env: "DS_RING_HI", default: u64::MAX >> 1, value: AtomicU64::new(UNSET) },
    ];

    fn get(i: usize) -> u64 {
        let t = &ALL[i];
        let v = t.value.load(Ordering::Relaxed);
        if v != UNSET {
            return v;
        }
        let v = std::env::var(t.env).ok().and_then(|x| x.trim().parse().ok()).unwrap_or(t.default);
        t.value.store(v, Ordering::Relaxed);
        v
    }

    /// Label promises (`Cp::wait`).
    pub fn promises() -> bool {
        get(0) != 0
    }

    /// Dispatches ordered by the title's synchronization only.
    pub fn lazy_barriers() -> bool {
        get(1) != 0
    }

    /// Written buffers below this size skip VRAM shadows (unless atomic).
    pub fn shadow_min() -> u64 {
        get(2)
    }

    /// VRAM mirrors of GPU-written buffers (`gpu::mirror`).
    pub fn mirrors() -> bool {
        get(3) != 0
    }

    /// Which stages read their SGPR data from video memory instead of system
    /// memory (diagnostic: it changes timing, not results): 0 none, 1 all,
    /// 2 the graphics queue's, 3 the compute queues', 4 shaders at addresses
    /// in [`ring_lo`, `ring_hi`).
    pub fn ring_vram() -> u64 {
        get(4)
    }

    pub fn ring_range() -> std::ops::Range<u64> {
        get(7)..get(8)
    }

    /// Read-only buffer bindings up to this size are snapshotted into video
    /// memory when recorded (0: off).
    pub fn snap_max() -> u64 {
        get(5)
    }

    /// Index buffers up to this size are snapshotted into video memory when
    /// recorded (0: off).
    pub fn snap_index() -> u64 {
        get(6)
    }

    /// Sets a switch by name; false if unknown.
    pub fn set(name: &str, v: u64) -> bool {
        match ALL.iter().find(|t| t.name == name) {
            Some(t) => {
                t.value.store(v, Ordering::Relaxed);
                true
            }
            None => false,
        }
    }
}

pub mod exec;
pub mod pcache;
pub mod pstats;
pub mod mirror;
pub mod images;
pub mod png;
pub mod track;
pub mod shaders;
pub mod swizzle_tables;
pub mod tiling;
pub mod verify;
pub mod vk;

use parking_lot::{Condvar, Mutex};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Flips presented so far (VideoOut).
pub static FRAMES: AtomicU64 = AtomicU64::new(0);
/// Command buffer submissions received (DCB/ACB).
pub static SUBMITS: AtomicU64 = AtomicU64::new(0);
/// Debug: dump GPU-visible images at the next flip (control `dump`).
pub static DUMP_REQUEST: AtomicBool = AtomicBool::new(false);
static SHUTDOWN: AtomicBool = AtomicBool::new(false);

pub fn shutting_down() -> bool {
    SHUTDOWN.load(Ordering::Relaxed)
}

/// Stops command-processor waits (process shutdown).
pub fn shutdown() {
    SHUTDOWN.store(true, Ordering::SeqCst);
}

/// Queue id of the graphics (DCB) queue; compute queues use their Agc ids.
pub const GFX: u32 = u32::MAX;

/// Submitted command buffers; with `DS_CHECK_SUBMIT=1` also their content
/// hashes and the submission time (see `worker`).
struct Submission {
    bufs: Vec<(u64, u32)>,
    hashes: Vec<u64>,
    /// Copies of the buffers at submission (DS_CHECK_SUBMIT).
    copies: Vec<Vec<u32>>,
    at: std::time::Instant,
}

fn buffer_copy(addr: u64, size: u32) -> Vec<u32> {
    if !cp::guest_readable(addr, 4 * size as u64) {
        return Vec::new();
    }
    unsafe { std::slice::from_raw_parts(addr as *const u32, size as usize) }.to_vec()
}

/// `DS_TRACE_SUBMIT=1`: logs submissions and their completion.
fn trace_submit() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_TRACE_SUBMIT").ok().as_deref() == Some("1"))
}

fn check_submit() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_CHECK_SUBMIT").ok().as_deref() == Some("1"))
}

fn buffer_hash(addr: u64, size: u32) -> u64 {
    if !cp::guest_readable(addr, 4 * size as u64) {
        return 0;
    }
    let w = unsafe { std::slice::from_raw_parts(addr as *const u32, size as usize) };
    w.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &x| (h ^ x as u64).wrapping_mul(0x100_0000_01b3))
}

/// Submission backpressure (`DS_SUBMIT_THROTTLE=1`, diagnostic): a submit
/// first waits (up to `THROTTLE_MAX`) until the queue completed all earlier
/// submissions, GPU work included, to test whether a problem depends on how
/// far the command processor lags behind the submitting threads (it can lag
/// by much more than hardware, e.g. during pipeline compilation). Not while
/// the CP polls a label (it may be waiting for the submitting thread).
fn throttle() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_SUBMIT_THROTTLE").ok().as_deref() == Some("1"))
}
const THROTTLE_MAX: std::time::Duration = std::time::Duration::from_millis(250);
/// Submissions up to this many dwords wait for their completion (see `submit`).
const SMALL_SUBMIT: u64 = 512;

/// The packets of a command buffer copy, one token each (diagnostics).
fn pm4_summary(w: &[u32]) -> String {
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < w.len() {
        let h = w[i];
        if h >> 30 != 3 {
            out.push(format!("{i}:t{}", h >> 30));
            i += 1;
            continue;
        }
        let n = ((h >> 16) & 0x3FFF) as usize + 1;
        let p = |k: usize| w.get(i + 1 + k).copied().unwrap_or(0);
        let a64 = |k: usize| p(k) as u64 | (p(k + 1) as u64) << 32;
        let s = match (h >> 8) & 0xFF {
            0x10 if p(0) == crate::hle::libs::agc_cmd::DS_MAGIC => format!("mk{}", p(1)),
            0x93 => format!("wait64[{:#x}]>={:#x}", a64(1), a64(3)),
            0x3C => format!("wait[{:#x}]={:#x}", a64(1), p(3)),
            0x37 => format!("wd[{:#x}]={:#x}", a64(1), p(3)),
            0x49 => format!("rel[{:#x}]={:#x}", a64(2), a64(4)),
            0x15 => format!("disp{}x{}x{}", p(0), p(1), p(2)),
            0x76 => format!("sh{:#x}+{}", p(0), n - 1),
            0x46 => format!("ev{:#x}", p(0) & 0x3F),
            0x58 => "acq".to_string(),
            o => format!("op{o:#x}/{n}"),
        };
        out.push(format!("{i}:{s}"));
        i += n + 1;
    }
    out.join(" ")
}

/// (dispatches, draws, other packets) of a command buffer's top level.
fn packet_mix(addr: u64, size: u32) -> (u32, u32, u32) {
    let (mut d, mut w, mut o) = (0, 0, 0);
    let mut i = 0u64;
    while i < size as u64 {
        let a = addr + 4 * i;
        if !cp::guest_readable(a, 4) {
            break;
        }
        let h = unsafe { *(a as *const u32) };
        if h >> 30 != 3 {
            i += 1;
            continue;
        }
        let n = ((h >> 16) & 0x3FFF) as u64 + 1;
        match (h >> 8) & 0xFF {
            0x15 | 0x16 => d += 1,
            0x24 | 0x25 | 0x27 | 0x2C | 0x2D | 0x30 | 0x35 | 0x38 => w += 1,
            0x76 | 0x69 | 0x79 | 0x10 => {}
            _ => o += 1,
        }
        i += n + 1;
    }
    (d, w, o)
}

/// Whether a submission returns only once its work completed (see `submit`).
fn waits_for_completion(bufs: &[(u64, u32)]) -> bool {
    static SYNC: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    let mode = *SYNC.get_or_init(|| match std::env::var("DS_SUBMIT_SYNC").ok().as_deref() {
        Some("1") => 2,
        Some("small") => 1,
        _ => 0,
    });
    mode == 2 || (mode == 1 && bufs.iter().map(|x| x.1 as u64).sum::<u64>() <= SMALL_SUBMIT)
}
static THROTTLE_WAITS: AtomicU64 = AtomicU64::new(0);
static THROTTLE_NS: AtomicU64 = AtomicU64::new(0);
static THROTTLE_TIMEOUTS: AtomicU64 = AtomicU64::new(0);

/// Submission backpressure statistics.
pub fn throttle_status() -> String {
    format!(
        "submit throttle: {} waits, {} ms total, {} timeouts",
        THROTTLE_WAITS.load(Ordering::Relaxed),
        THROTTLE_NS.load(Ordering::Relaxed) / 1_000_000,
        THROTTLE_TIMEOUTS.load(Ordering::Relaxed)
    )
}

struct Queue {
    id: u32,
    pending: Mutex<VecDeque<Submission>>,
    cv: Condvar,
    /// Submissions completed (diagnostics, backpressure).
    done: AtomicU64,
    /// Submissions queued so far.
    submitted: AtomicU64,
    /// Signalled when a submission completes.
    done_cv: Condvar,
    /// The command processor is polling a label.
    cp_waiting: std::sync::atomic::AtomicBool,
    /// Host thread id of the queue's command processor (stack samples).
    tid: std::sync::atomic::AtomicU32,
}

static QUEUES: Mutex<Vec<&'static Queue>> = Mutex::new(Vec::new());

fn queue(id: u32) -> &'static Queue {
    let mut qs = QUEUES.lock();
    if let Some(q) = qs.iter().find(|q| q.id == id) {
        return q;
    }
    let q: &'static Queue = Box::leak(Box::new(Queue {
        id,
        pending: Mutex::new(VecDeque::new()),
        cv: Condvar::new(),
        done: AtomicU64::new(0),
        submitted: AtomicU64::new(0),
        done_cv: Condvar::new(),
        cp_waiting: std::sync::atomic::AtomicBool::new(false),
        tid: std::sync::atomic::AtomicU32::new(0),
    }));
    qs.push(q);
    let name = if id == GFX { "gpu-gfx".to_string() } else { format!("gpu-acb{id:#x}") };
    crate::info!("gpu: {name} command processor started");
    crate::thread::spawn_host(&name.clone(), 8 << 20, Box::new(move || worker(q, &name))).expect("gpu queue thread");
    q
}

fn worker(q: &'static Queue, name: &str) {
    crate::log::set_thread_name(name);
    q.tid.store(unsafe { windows_sys::Win32::System::Threading::GetCurrentThreadId() }, Ordering::Relaxed);
    crate::exception::arm_watch_current_thread();
    let mut cp = cp::Cp::new(name, q.id != GFX);
    cp.wait_flag = Some(&q.cp_waiting);
    loop {
        let batch = {
            let mut g = q.pending.lock();
            loop {
                if let Some(b) = g.pop_front() {
                    break b;
                }
                q.cv.wait(&mut g);
            }
        };
        for (i, &(addr, size)) in batch.bufs.iter().enumerate() {
            if trace_submit() && size == 277 {
                // Bink decode command buffers: their counter (dword 272) and
                // dword 159 at submission and now, and the label the first
                // packet waits for.
                let rd = |k: u64| if cp::guest_readable(addr + 4 * k, 4) { unsafe { *((addr + 4 * k) as *const u32) } } else { 0 };
                let sub = batch.copies.get(i).map(|w| (w.get(272).copied().unwrap_or(0), w.get(159).copied().unwrap_or(0))).unwrap_or((0, 0));
                let label = (rd(8) as u64) | ((rd(9) as u64) << 32);
                let lv = if cp::guest_readable(label, 8) { unsafe { *(label as *const u64) } } else { 0 };
                crate::info!("bink trace: {addr:#x} start: [272] {:#x} (submitted {:#x}) [159] {:#x} (submitted {:#x}), waits [{label:#x}] >= {:#x} (now {lv:#x}), {} ms after submission", rd(272), sub.0, rd(159), sub.1, rd(10), batch.at.elapsed().as_millis());
            }
            if let Some(&h) = batch.hashes.get(i) {
                if buffer_hash(addr, size) != h {
                    crate::warn!("gpu[{name}]: command buffer {addr:#x} ({size} dwords) changed between its submission and its execution ({} ms later)", batch.at.elapsed().as_millis());
                }
            }
            cp.run(addr, size);
            if let Some(&h) = batch.hashes.get(i) {
                if buffer_hash(addr, size) != h {
                    let now = buffer_copy(addr, size);
                    let diffs: Vec<String> = batch.copies.get(i).map(|old| old.iter().zip(&now).enumerate().filter(|(_, (a, b))| a != b).take(6).map(|(k, (a, b))| format!("[{k}] {a:#x}->{b:#x}")).collect()).unwrap_or_default();
                    crate::warn!("gpu[{name}]: command buffer {addr:#x} ({size} dwords) changed during its execution ({} ms after submission): {}", batch.at.elapsed().as_millis(), diffs.join(" "));
                }
            }
        }
        if throttle() || waits_for_completion(&batch.bufs) {
            cp.finish();
        }
        let _g = q.pending.lock();
        let d = q.done.fetch_add(1, Ordering::SeqCst) + 1;
        q.done_cv.notify_all();
        if trace_submit() {
            crate::info!("submit trace: queue {:#x} #{d} completed ({} ms after submission)", q.id, batch.at.elapsed().as_millis());
        }
    }
}

/// Queues command buffers (address, size in dwords) for execution on `queue_id`.
pub fn submit(queue_id: u32, buffers: Vec<(u64, u32)>) {
    if buffers.is_empty() {
        return;
    }
    SUBMITS.fetch_add(1, Ordering::Relaxed);
    let q = queue(queue_id);
    let hashes = if check_submit() { buffers.iter().map(|&(a, n)| buffer_hash(a, n)).collect() } else { Vec::new() };
    let copies = if check_submit() || trace_submit() { buffers.iter().map(|&(a, n)| buffer_copy(a, n)).collect() } else { Vec::new() };
    let mut g = q.pending.lock();
    if throttle() {
        let t0 = std::time::Instant::now();
        let mut waited = false;
        while q.done.load(Ordering::SeqCst) < q.submitted.load(Ordering::SeqCst) && !q.cp_waiting.load(Ordering::SeqCst) {
            waited = true;
            let left = THROTTLE_MAX.saturating_sub(t0.elapsed());
            if left.is_zero() {
                THROTTLE_TIMEOUTS.fetch_add(1, Ordering::Relaxed);
                break;
            }
            q.done_cv.wait_for(&mut g, left.min(std::time::Duration::from_millis(5)));
        }
        if waited {
            THROTTLE_WAITS.fetch_add(1, Ordering::Relaxed);
            THROTTLE_NS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
    }
    let index = q.submitted.fetch_add(1, Ordering::SeqCst) + 1;
    if trace_submit() {
        let (d, w, o) = buffers.iter().map(|&(a, n)| packet_mix(a, n)).fold((0, 0, 0), |x, y| (x.0 + y.0, x.1 + y.1, x.2 + y.2));
        crate::info!("submit trace: queue {queue_id:#x} #{index} {:#x}+{} (done {}) dispatches {d} draws {w} other {o}", buffers[0].0, buffers[0].1, q.done.load(Ordering::SeqCst));
        // Bink decodes (they start with a "BinkGPU" marker): list the
        // packets of the first one and of ones with other sizes.
        static LISTED: AtomicU64 = AtomicU64::new(0);
        if let Some(w) = copies.first() {
            let bink = w.len() > 3 && w[1] == crate::hle::libs::agc_cmd::DS_MAGIC && w[2] == crate::hle::libs::agc_cmd::pop::PUSH_MARKER && w.get(4) == Some(&u32::from_le_bytes(*b"Bink"));
            if bink && (w.len() != 277 || LISTED.load(Ordering::Relaxed) == 0) && LISTED.fetch_add(1, Ordering::Relaxed) < 10 {
                crate::info!("submit trace: #{index} packets: {}", pm4_summary(w));
            }
        }
    }
    g.push_back(Submission { bufs: buffers, hashes, copies, at: std::time::Instant::now() });
    q.cv.notify_one();
    // DS_SUBMIT_SYNC=small (up to SMALL_SUBMIT dwords) or =1 (all),
    // diagnostic: the submit returns once its work completed (no CPU/GPU
    // overlap for those submissions; about halves the frame rate). Bounded,
    // and not while the CP polls a label that may come from the submitter.
    if waits_for_completion(&g.back().map(|b| b.bufs.clone()).unwrap_or_default()) {
        let t0 = std::time::Instant::now();
        while q.done.load(Ordering::SeqCst) < index && !q.cp_waiting.load(Ordering::SeqCst) && t0.elapsed() < THROTTLE_MAX {
            q.done_cv.wait_for(&mut g, std::time::Duration::from_millis(5));
        }
    }
    drop(g);
    q.cv.notify_one();
}

/// `DS_PROFILE_GPU=1`: samples the GPU queue threads every 10 ms; the most
/// frequent host stacks (top 3 frames) are listed in hang reports.
static PROFILE: Mutex<Option<std::collections::HashMap<String, u64>>> = Mutex::new(None);

/// Inclusive samples per (queue kind, host function) and samples per queue
/// kind ("gfx" / "acb"): the share of time a function was on the stack.
static PROFILE_FN: Mutex<Option<(std::collections::HashMap<(&'static str, String), u64>, std::collections::HashMap<&'static str, u64>)>> = Mutex::new(None);

pub fn start_profiler() {
    if std::env::var("DS_PROFILE_GPU").ok().as_deref() != Some("1") {
        return;
    }
    *PROFILE.lock() = Some(std::collections::HashMap::new());
    *PROFILE_FN.lock() = Some(Default::default());
    let _ = crate::thread::spawn_host(
        "gpu-profiler",
        1 << 20,
        Box::new(|| {
            crate::log::set_thread_name("gpu-profiler");
            loop {
                std::thread::sleep(std::time::Duration::from_millis(10));
                let qs: Vec<&'static Queue> = QUEUES.lock().clone();
                for q in qs {
                    let Some(st) = crate::exception::sample_thread(q.tid.load(Ordering::Relaxed), 0) else { continue };
                    let host = st.lines().next().unwrap_or("").trim().trim_start_matches("host: ").to_string();
                    let key: Vec<&str> = host.split(" <- ").take(4).collect();
                    let name = if q.id == GFX { "gfx".to_string() } else { format!("acb{:#x}", q.id) };
                    let k = format!("{name}: {}", key.join(" <- "));
                    if let Some(m) = PROFILE.lock().as_mut() {
                        *m.entry(k).or_default() += 1;
                    }
                    let kind = if q.id == GFX { "gfx" } else { "acb" };
                    let mut fns: Vec<&str> = host.split(" <- ").map(|f| f.split("+0x").next().unwrap_or(f)).collect();
                    fns.sort_unstable();
                    fns.dedup();
                    if let Some((m, n)) = PROFILE_FN.lock().as_mut() {
                        *n.entry(kind).or_default() += 1;
                        for f in fns {
                            *m.entry((kind, f.to_string())).or_default() += 1;
                        }
                    }
                }
            }
        }),
    );
}

/// Clears the samples and dispatch timings (`profile reset` control command).
pub fn reset_profile() {
    if let Some(m) = PROFILE.lock().as_mut() {
        m.clear();
    }
    if let Some((m, n)) = PROFILE_FN.lock().as_mut() {
        m.clear();
        n.clear();
    }
    exec::reset_shader_profile();
    cp::reset_wait_stats();
}

/// The profile so far (`profile report` control command).
pub fn profile_report() -> String {
    let mut s = profile_summary();
    s.push_str(&exec::shader_profile());
    s.push_str(&cp::wait_stats());
    s
}

fn profile_summary() -> String {
    let g = PROFILE.lock();
    let Some(m) = g.as_ref() else { return String::new() };
    let mut v: Vec<(&String, &u64)> = m.iter().collect();
    v.sort_by(|a, b| b.1.cmp(a.1));
    let mut s = String::from("    gpu profile (10 ms samples):\n");
    for (k, n) in v.iter().take(20) {
        s.push_str(&format!("      {n:6}  {k}\n"));
    }
    drop(g);
    // Inclusive time per function of our crate (and the waits), per queue kind.
    if let Some((m, totals)) = PROFILE_FN.lock().as_ref() {
        let top = std::env::var("DS_PROFILE_FN_TOP").ok().and_then(|x| x.parse().ok()).unwrap_or(40);
        for kind in ["gfx", "acb"] {
            let total = totals.get(kind).copied().unwrap_or(0).max(1);
            let mut v: Vec<(&String, &u64)> = m.iter().filter(|((k, f), _)| *k == kind && (f.starts_with("dsport::") || f.contains("Wait") || f.contains("Sleep") || f.contains("Submit"))).map(|((_, f), n)| (f, n)).collect();
            v.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
            s.push_str(&format!("    gpu profile {kind}: {total} samples, inclusive share per function:\n"));
            for (f, n) in v.iter().take(top) {
                s.push_str(&format!("      {:5.1}%  {f}\n", **n as f64 * 100.0 / total as f64));
            }
        }
    }
    s
}

/// One line per queue for hang reports.
pub fn status() -> String {
    status_ex(false)
}

/// Queue lines, with host stack samples of the command processors.
pub fn status_ex(with_stacks: bool) -> String {
    let mut s = String::new();
    let qs: Vec<&'static Queue> = match QUEUES.try_lock() {
        Some(g) => g.clone(),
        None => return s,
    };
    for q in qs {
        let pending = q.pending.try_lock().map(|g| g.len().to_string()).unwrap_or_else(|| "?".into());
        let name = if q.id == GFX { "gpu-gfx".to_string() } else { format!("gpu-acb{:#x}", q.id) };
        let state = cp::STATE.try_lock().and_then(|g| g.iter().find(|(n, _)| *n == name).map(|(_, s)| s.clone())).unwrap_or_else(|| "idle".into());
        s.push_str(&format!("    {name}: {} submission(s) done, {pending} pending; {state}\n", q.done.load(Ordering::Relaxed)));
        if with_stacks {
            if let Some(st) = crate::exception::sample_thread(q.tid.load(Ordering::Relaxed), 16) {
                s.push_str(&format!("        {st}\n"));
            }
        }
    }
    s.push_str(&format!(
        "    gpu totals: {} submits, {} packets, {} draws, {} dispatches, {} EOP interrupts, {} flips, {} label waits latched, {} passed on promises, {} CP reads waited for another queue\n",
        SUBMITS.load(Ordering::Relaxed),
        cp::PACKETS.load(Ordering::Relaxed),
        cp::DRAWS.load(Ordering::Relaxed),
        cp::DISPATCHES.load(Ordering::Relaxed),
        cp::EOP_INTERRUPTS.load(Ordering::Relaxed),
        FRAMES.load(Ordering::Relaxed),
        cp::LATCHED.load(Ordering::Relaxed),
        cp::PROMISED.load(Ordering::Relaxed),
        exec::FOREIGN_WAITS.load(Ordering::Relaxed)
    ));
    s.push_str(&format!("    {}
", throttle_status()));
    s.push_str(&exec::status());
    s.push_str(&profile_summary());
    s.push_str(&exec::shader_profile());
    s
}
