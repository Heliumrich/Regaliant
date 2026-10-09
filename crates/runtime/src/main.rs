//! dsport: runs the PS5 Demon's Souls executable (PPSA01341 01.007.000) natively
//! on Windows x86-64 with high-level emulation of the PS5 system libraries.

#[macro_use]
mod log;
mod ehframe;
mod audio;
mod exception;
mod gpu;
mod hle;
mod input;
mod loader;
mod mem;
mod patch;
mod thread;
mod tls;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

pub struct Config {
    pub root: PathBuf,
    /// Application root (`PPSA01341-app0`): read-only assets mounted at /app0.
    pub app: PathBuf,
    /// Directory holding the decrypted executable and `sce_module/`.
    pub code: PathBuf,
    pub run_dir: PathBuf,
    pub seconds: Option<u64>,
    /// Development control channel (see `poll_control`).
    pub control: Option<PathBuf>,
    /// `--pipeline-test <capture.txt>`: create pipelines for a captured frame and exit.
    pub pipeline_test: Option<String>,
    pub trace: bool,
    pub lenient: bool,
    pub debug: bool,
    pub headless: bool,
}

static RUN_DIR: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();

/// This run's output directory (logs, captures).
pub fn run_dir() -> Option<&'static Path> {
    RUN_DIR.get().map(|p| p.as_path())
}

/// System allocator that reports implausibly large requests (almost always a
/// guest value misread as a size) with the requesting host frames, before the
/// request fails and aborts the process.
struct GuardedAlloc;

const HUGE_ALLOC: usize = 4 << 30;

#[cold]
fn report_huge_alloc(size: usize) {
    use std::io::Write;
    let mut frames = [std::ptr::null_mut::<std::ffi::c_void>(); 24];
    let n = unsafe { windows_sys::Win32::System::Diagnostics::Debug::RtlCaptureStackBackTrace(1, 24, frames.as_mut_ptr(), std::ptr::null_mut()) } as usize;
    let mut err = std::io::stderr().lock();
    let _ = writeln!(err, "huge host allocation of {size} bytes ({size:#x}) requested; frames:");
    for f in &frames[..n] {
        let _ = writeln!(err, "    {}", exception::host_symbol(*f as u64));
    }
}

unsafe impl std::alloc::GlobalAlloc for GuardedAlloc {
    unsafe fn alloc(&self, l: std::alloc::Layout) -> *mut u8 {
        if l.size() > HUGE_ALLOC {
            report_huge_alloc(l.size());
        }
        unsafe { std::alloc::System.alloc(l) }
    }
    unsafe fn alloc_zeroed(&self, l: std::alloc::Layout) -> *mut u8 {
        if l.size() > HUGE_ALLOC {
            report_huge_alloc(l.size());
        }
        unsafe { std::alloc::System.alloc_zeroed(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: std::alloc::Layout) {
        unsafe { std::alloc::System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: std::alloc::Layout, new_size: usize) -> *mut u8 {
        if new_size > HUGE_ALLOC {
            report_huge_alloc(new_size);
        }
        unsafe { std::alloc::System.realloc(p, l, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: GuardedAlloc = GuardedAlloc;

fn utc_stamp() -> String {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs() as i64;
    let days = secs.div_euclid(86400);
    let rem = secs.rem_euclid(86400);
    // Civil-from-days (Howard Hinnant).
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{:04}{:02}{:02}-{:02}{:02}{:02}", y, m, d, rem / 3600, (rem / 60) % 60, rem % 60)
}

fn parse_args() -> Config {
    let exe = std::env::current_exe().unwrap();
    // target/<profile>/dsport.exe -> project root
    let mut root = exe.parent().and_then(|p| p.parent()).and_then(|p| p.parent()).map(Path::to_path_buf).unwrap_or_default();
    let mut seconds = None;
    let mut trace = false;
    let mut lenient = false;
    let mut debug = false;
    let mut headless = false;
    let mut control = None;
    let mut app = None;
    let mut pipeline_test = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--seconds" => seconds = args.next().and_then(|s| s.parse().ok()),
            "--trace" => trace = true,
            "--lenient" => lenient = true,
            "--debug" => debug = true,
            "--headless" => headless = true,
            "--root" => root = PathBuf::from(args.next().expect("--root needs a path")),
            "--control" => control = Some(PathBuf::from(args.next().expect("--control needs a path"))),
            "--app" => app = Some(PathBuf::from(args.next().expect("--app needs a path"))),
            "--pipeline-test" => pipeline_test = Some(args.next().expect("--pipeline-test needs a capture file")),
            other => {
                eprintln!("unknown argument {other}");
                std::process::exit(2);
            }
        }
    }
    let data = root.join("local-data");
    let app = app.unwrap_or_else(|| root.join("Game").join("Demon-Souls-Remake-PPSA01341-1.07").join("PPSA01341-app0"));
    Config {
        code: app.join("decrypted"),
        app,
        run_dir: data.join("runs").join(utc_stamp()),
        root,
        seconds,
        trace,
        lenient,
        debug,
        headless,
        control,
        pipeline_test,
    }
}

/// Keeps the newest run directories (and any containing a KEEP marker).
fn prune_runs(runs: &Path, keep: usize) {
    let Ok(rd) = std::fs::read_dir(runs) else { return };
    let mut dirs: Vec<PathBuf> = rd.flatten().map(|e| e.path()).filter(|p| p.is_dir()).collect();
    dirs.sort();
    if dirs.len() <= keep {
        return;
    }
    let n = dirs.len() - keep;
    for d in dirs.into_iter().take(n) {
        if !d.join("KEEP").exists() {
            let _ = std::fs::remove_dir_all(&d);
        }
    }
}

/// Development control channel (`--control <file>`): executes complete lines
/// appended to the file since the last poll.
///   press <buttons> [hold]   e.g. `press down`, `press cross 10`, `press ls_up+r1 30`; the hold
///                            is in flips (default 6), or wall-clock with a suffix: `4s`, `300ms`
///   shot <name>              screenshot of the next presented flip -> shot_<name>.png
///   dump                     dump all GPU images at the next flip
///   trace                    log every draw/dispatch of an upcoming flip (FRAME lines)
///   loglevel trace|debug|info  change logging at run time (trace = every HLE call)
///   textreset                log text-hook strings again (`DS_TEXT_HOOK`)
///   quit                     orderly shutdown
fn poll_control(path: &Path, offset: &mut u64) {
    use std::io::{Read, Seek, SeekFrom};
    let Ok(mut f) = std::fs::File::open(path) else { return };
    let len = f.metadata().map(|m| m.len()).unwrap_or(0);
    if len < *offset {
        *offset = 0; // truncated: start over
    }
    if len == *offset || f.seek(SeekFrom::Start(*offset)).is_err() {
        return;
    }
    let mut buf = String::new();
    if f.read_to_string(&mut buf).is_err() {
        return;
    }
    let Some(end) = buf.rfind('\n') else { return };
    *offset += end as u64 + 1;
    for line in buf[..end].lines().map(str::trim).filter(|l| !l.is_empty() && !l.starts_with('#')) {
        let mut parts = line.split_whitespace();
        let cmd = parts.next().unwrap_or("");
        let flip = gpu::FRAMES.load(std::sync::atomic::Ordering::Relaxed);
        let result = match cmd {
            "press" => {
                let names = parts.next().unwrap_or("");
                match parts.next() {
                    Some(h) if h.ends_with("ms") => match h.trim_end_matches("ms").parse::<u64>() {
                        Ok(ms) => input::press_for(names, Duration::from_millis(ms)),
                        Err(_) => Err(format!("bad hold '{h}'")),
                    },
                    Some(h) if h.ends_with('s') => match h.trim_end_matches('s').parse::<f64>() {
                        Ok(s) if s > 0.0 && s < 3600.0 => input::press_for(names, Duration::from_secs_f64(s)),
                        _ => Err(format!("bad hold '{h}'")),
                    },
                    h => input::press_now(names, h.and_then(|h| h.parse().ok()).unwrap_or(6)),
                }
            }
            "shot" => {
                let name: String = parts.next().unwrap_or("x").chars().filter(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-').collect();
                *hle::libs::videoout::SHOT_REQUEST.lock() = Some(name);
                Ok(())
            }
            "dump" => {
                gpu::DUMP_REQUEST.store(true, std::sync::atomic::Ordering::Relaxed);
                Ok(())
            }
            "breaklog" => {
                let n = parts.next().and_then(|v| v.parse().ok()).unwrap_or(1000);
                exception::BREAK_VERBOSE.store(n, std::sync::atomic::Ordering::Relaxed);
                Ok(())
            }
            "loglevel" => {
                let (lvl, hle) = match parts.next().unwrap_or("") {
                    "trace" => (log::Level::Debug, true),
                    "debug" => (log::Level::Debug, false),
                    "info" => (log::Level::Info, false),
                    other => {
                        warn!("control: unknown log level '{other}'");
                        (log::Level::Info, false)
                    }
                };
                log::LEVEL.store(lvl as u8, std::sync::atomic::Ordering::Relaxed);
                log::TRACE_HLE.store(hle, std::sync::atomic::Ordering::Relaxed);
                Ok(())
            }
            "ime" => {
                // Text for a pending system IME dialog (e.g. the character name).
                let text = line.trim_start().strip_prefix("ime").unwrap_or("").trim().to_string();
                *hle::libs::ime_dialog::CONTROL_TEXT.lock() = Some(text);
                Ok(())
            }
            "eqbias" => {
                // Diagnosis of EQUAL depth tests: `eqbias <units>|always|off`.
                let v = match parts.next().unwrap_or("off") {
                    "off" => None,
                    "always" => Some(f32::INFINITY),
                    x => x.parse().ok(),
                };
                gpu::exec::set_equal_bias(v);
                Ok(())
            }
            "probe" => {
                // Diagnosis: `probe <shader>:<pc>:<vgpr>[:<frames>]` / `probe off`.
                let rest = line.trim_start().strip_prefix("probe").unwrap_or("").trim();
                gpu::shaders::set_probe(rest)
            }
            "skipvs" => {
                // Diagnosis: `skipvs <hex vertex shader address>` toggles
                // skipping the draws that use it.
                match parts.next().and_then(|a| u64::from_str_radix(a.trim_start_matches("0x"), 16).ok()) {
                    Some(a) => {
                        let on = gpu::exec::toggle_skip_vs(a);
                        info!("control: draws with vertex shader {a:#x} {}", if on { "skipped" } else { "drawn again" });
                        Ok(())
                    }
                    None => Err("usage: skipvs <hex address>".into()),
                }
            }
            "textreset" => {
                exception::text_reset();
                Ok(())
            }
            "stacks" => {
                info!("guest thread stacks:\n{}", thread_summary_ex(true));
                Ok(())
            }
            "capture" => {
                // `capture [<from>-<to>]`: the next frame, optionally with
                // per-draw reports for that range of its events.
                if let Some((a, b)) = parts.next().and_then(|r| r.split_once('-')) {
                    match (a.parse(), b.parse()) {
                        (Ok(a), Ok(b)) => gpu::capture::set_draw_range((a, b)),
                        _ => warn!("control: bad draw range '{a}-{b}'"),
                    }
                }
                gpu::capture::arm_next();
                Ok(())
            }
            "vmas" => {
                // Debug: guest address space layout.
                let m = crate::mem::MEM.lock();
                let mut s = format!("{} guest mappings:", m.vmas.len());
                for v in m.vmas.values() {
                    s.push_str(&format!("\n  {:#014x}-{:#014x} {:>10} KiB {:?} prot {:#x} phys {:#x} '{}'", v.start, v.end(), v.size >> 10, v.kind, v.prot, v.phys, v.name));
                }
                drop(m);
                info!("{s}");
                Ok(())
            }
            "quit" => {
                info!("control: quit at flip {flip}");
                shutdown(0);
            }
            _ => Err(format!("unknown command '{cmd}'")),
        };
        match result {
            Ok(()) => info!("control: '{line}' at flip {flip}"),
            Err(e) => warn!("control: '{line}': {e}"),
        }
    }
}

pub fn fatal_report(text: &str) {
    crate::log::write(crate::log::Level::Error, format_args!("{text}"));
    if let Some(dir) = RUN_DIR.get() {
        let _ = std::fs::write(dir.join("crash.txt"), text);
    }
}

fn thread_summary() -> String {
    thread_summary_ex(false)
}

/// Guest threads and their states; with `with_stacks`, each running thread is
/// sampled (rip, frame chain, guest return addresses) for hang diagnosis.
fn thread_summary_ex(with_stacks: bool) -> String {
    let mut s = String::new();
    let list: Vec<&'static thread::GuestThread> = match thread::THREADS.try_lock() {
        Some(threads) => threads.iter().copied().collect(),
        None => return s,
    };
    for t in list {
        let state = match t.exited.try_lock() {
            Some(g) => match *g {
                Some(v) => format!("exited({v:#x}) at {:.3} s", t.exit_ms.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1000.0),
                None => "running".into(),
            },
            None => "?".into(),
        };
        let tid = t.host_tid.load(std::sync::atomic::Ordering::Relaxed);
        s.push_str(&format!("    thread {} '{}' host-tid {} {}\n", t.id, t.name(), tid, state));
        if state == "running" {
            if let Some(w) = thread::describe_wait(t) {
                s.push_str(&format!("        {w}\n"));
            }
        }
        if with_stacks && state == "running" {
            if let Some(st) = exception::sample_thread(tid as u32, 16) {
                s.push_str(&format!("        {st}\n"));
            }
        }
    }
    s.push_str(&gpu::status_ex(with_stacks));
    s.push_str(&audio::status());
    s.push_str(&hle::libs::ajm::status());
    s
}

pub fn fatal_exit(code: u32, reason: &str) -> ! {
    crate::log::write(crate::log::Level::Error, format_args!("terminating: {reason}\n{}", thread_summary()));
    crate::log::try_flush();
    unsafe {
        windows_sys::Win32::System::Threading::TerminateProcess(windows_sys::Win32::System::Threading::GetCurrentProcess(), code);
    }
    loop {
        std::thread::sleep(Duration::from_secs(1));
    }
}

pub fn shutdown(code: i32) -> ! {
    gpu::shutdown();
    info!("shutdown with status {code}\n{}", thread_summary());
    crate::log::flush();
    unsafe {
        windows_sys::Win32::System::Threading::TerminateProcess(windows_sys::Win32::System::Threading::GetCurrentProcess(), code as u32);
    }
    loop {
        std::thread::sleep(Duration::from_secs(1));
    }
}

fn read_proc_param(m: &loader::LoadedModule) -> (u64, String, i32) {
    let mut stack = 0u64;
    let mut name = "GAME_MainThread".to_string();
    let mut prio = hle::kernel::pthread::PRIO_DEFAULT;
    if let Some(pp) = m.proc_param {
        unsafe {
            let size = *(pp as *const u64);
            let magic = *((pp + 8) as *const u32);
            let sdk = *((pp + 0x10) as *const u64);
            hle::kernel::misc::SDK_VERSION.store(sdk, std::sync::atomic::Ordering::Relaxed);
            info!("proc param at {pp:#x}: size {size:#x} magic {magic:#x} sdk {sdk:#x}");
            let field = |off: u64| if off + 8 <= size { *((pp + off) as *const u64) } else { 0 };
            let name_pp = field(0x20);
            if name_pp != 0 && *(name_pp as *const u64) != 0 {
                name = hle::cstr(*(name_pp as *const u64) as *const u8).to_string();
            }
            let prio_p = field(0x28);
            if prio_p != 0 {
                prio = *(prio_p as *const i32);
            }
            let stack_p = field(0x30);
            if stack_p != 0 {
                stack = *(stack_p as *const u32) as u64;
            }
        }
    }
    (stack, name, prio)
}

fn main() {
    // The guest address range must be reserved before the process grows.
    if let Err(e) = mem::reserve_guest_range() {
        eprintln!("{e}");
        std::process::exit(2);
    }
    let slot = match tls::init_slot() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    };
    let cfg = parse_args();
    let runs = cfg.root.join("local-data").join("runs");
    prune_runs(&runs, 30);
    std::fs::create_dir_all(&cfg.run_dir).expect("create run dir");
    RUN_DIR.set(cfg.run_dir.clone()).ok();
    log::init(&cfg.run_dir.join("dsport.log")).expect("open log");
    log::set_thread_name("host");
    if cfg.debug || cfg.trace {
        log::LEVEL.store(log::Level::Debug as u8, std::sync::atomic::Ordering::Relaxed);
    }
    log::TRACE_HLE.store(cfg.trace, std::sync::atomic::Ordering::Relaxed);
    hle::LENIENT_IMPORTS.store(cfg.lenient, std::sync::atomic::Ordering::Relaxed);
    info!("dsport {} | run dir {} | TLS slot {}", env!("CARGO_PKG_VERSION"), cfg.run_dir.display(), slot);
    exception::install();
    if let Some(c) = &cfg.pipeline_test {
        let code = gpu::exec::pipeline_test(c);
        log::flush();
        std::process::exit(code);
    }
    gpu::capture::init_from_env();
    gpu::start_profiler();
    if std::env::var_os("DS_TEST_FAULT").is_some() {
        info!("DS_TEST_FAULT: writing to an invalid address from host code");
        unsafe { std::ptr::write_volatile(0x32c3be8 as *mut u64, 1) };
    }
    hle::kernel::time::init();
    if !cfg.headless {
        if let Err(e) = gpu::vk::present::init("dsport - Demon's Souls (PPSA01341)") {
            warn!("no presentation window: {e}");
        }
    }

    // Guest file system. The supplied game tree is mounted read-only; all
    // writable state lives under local-data/.
    let rt = cfg.root.join("local-data").join("runtime");
    for d in ["temp0", "download0", "data", "devlog"] {
        hle::kernel::fs::ensure_dir(&rt.join(d));
    }
    info!("application root {} (read-only), code {}", cfg.app.display(), cfg.code.display());
    hle::kernel::fs::mount("/app0", vec![cfg.app.clone()], false);
    hle::kernel::fs::mount("/hostapp", vec![cfg.app.clone()], false);
    hle::kernel::fs::mount("/temp0", vec![rt.join("temp0")], true);
    hle::kernel::fs::mount("/temp", vec![rt.join("temp0")], true);
    hle::kernel::fs::mount("/download0", vec![rt.join("download0")], true);
    hle::kernel::fs::mount("/data", vec![rt.join("data")], true);
    // The title writes its own log files to /devlog/app when it can (useful diagnostics).
    hle::kernel::fs::ensure_dir(&rt.join("devlog").join("logs"));
    hle::kernel::fs::mount("/devlog/app", vec![rt.join("devlog")], true);
    hle::libs::savedata::init(cfg.root.join("local-data").join("saves"));

    // Modules: the decrypted executable and the title's own libraries.
    let t0 = Instant::now();
    let eboot = loader::load(&cfg.code.join("eboot.bin"), true).unwrap_or_else(|e| {
        error!("loading eboot failed: {e}");
        fatal_exit(1, "load failure");
    });
    let mut libs: Vec<Arc<loader::LoadedModule>> = Vec::new();
    for name in ["libc.prx", "libSceNpCppWebApi.prx"] {
        let p = cfg.code.join("sce_module").join(name);
        if !p.exists() {
            error!("{name} not found in {}", p.display());
            fatal_exit(1, "missing title module");
        }
        match loader::load(&p, false) {
            Ok(m) => libs.push(m),
            Err(e) => {
                error!("loading {name} failed: {e}");
                fatal_exit(1, "load failure");
            }
        }
    }
    let _ = hle::registry();
    let modules: Vec<Arc<loader::LoadedModule>> = loader::MODULES.read().clone();
    for m in &modules {
        match loader::relocate(m) {
            Ok(s) => info!("{}: linked (lle {}, hle {}, unresolved {}, weak-null {})", m.name, s.lle, s.hle, s.unresolved, s.weak_null),
            Err(e) => {
                error!("relocating {} failed: {e}", m.name);
                fatal_exit(1, "link failure");
            }
        }
    }
    for m in &modules {
        match loader::finalize(m) {
            Ok(s) => loader::report_patch(m, &s),
            Err(e) => {
                error!("finalizing {} failed: {e}", m.name);
                fatal_exit(1, "patch failure");
            }
        }
    }
    let unresolved = hle::UNRESOLVED.read().len();
    {
        // Planning aid: every unresolved function import, by library.
        let mut lines: Vec<String> = hle::UNRESOLVED.read().iter().map(|u| format!("{}	{}	{}", u.lib, u.nid, hle::describe_nid(&u.nid))).collect();
        lines.sort();
        let _ = std::fs::write(cfg.run_dir.join("unresolved.txt"), lines.join("
") + "
");
    }
    info!("load+link+patch took {:.1} ms; {} distinct unresolved function imports", t0.elapsed().as_secs_f64() * 1000.0, unresolved);

    // Guest main thread.
    let (stack, name, prio) = read_proc_param(&eboot);
    let stack = stack.max(8 << 20);
    let main_t = thread::new_thread(&name, eboot.entry, 0, stack, prio);
    let eboot2 = eboot.clone();
    thread::spawn_host(
        &name,
        stack as usize + (1 << 20),
        Box::new(move || {
            thread::enter(main_t);
            for m in &libs {
                if let Some(init) = m.init {
                    info!("starting module {} at {}", m.name, loader::describe_addr(init));
                    let r = thread::run_entry(main_t, init, 0, 0);
                    info!("module {} start returned {:#x}", m.name, r);
                }
            }
            let params = mem::arena_alloc(40, 8);
            unsafe {
                *(params as *mut u32) = 1;
                *((params + 8) as *mut u64) = mem::arena_cstr("eboot.bin");
            }
            if let Ok(list) = std::env::var("DS_BREAK") {
                for off in list.split(',').filter_map(|v| u64::from_str_radix(v.trim().trim_start_matches("0x"), 16).ok()) {
                    match exception::install_breakpoint(eboot2.base + off) {
                        Ok(()) => info!("breakpoint at {}", loader::describe_addr(eboot2.base + off)),
                        Err(e) => warn!("breakpoint {off:#x}: {e}"),
                    }
                }
            }
            info!("entering {} at {}", eboot2.name, loader::describe_addr(eboot2.entry));
            let r = thread::run_entry(main_t, eboot2.entry, params, hle::kernel::misc::rtld_exit_fn());
            info!("eboot entry returned {r:#x}");
            shutdown(r as i32);
        }),
    )
    .expect("spawn main thread");

    let start = Instant::now();
    let mut control = cfg.control.clone().map(|p| (p, 0u64));
    let mut ticks = 0u64;
    loop {
        std::thread::sleep(Duration::from_millis(250));
        ticks += 1;
        if ticks % 8 == 0 && exception::break_stats_enabled() {
            exception::dump_break_stats();
        }
        crate::log::try_flush();
        if let Some((path, offset)) = control.as_mut() {
            poll_control(path, offset);
        }
        if gpu::vk::window::CLOSED.load(std::sync::atomic::Ordering::SeqCst) {
            info!("window closed by user");
            shutdown(0);
        }
        if let Some(s) = cfg.seconds {
            if start.elapsed() >= Duration::from_secs(s) {
                info!("time limit of {s}s reached; guest thread stacks:\n{}", thread_summary_ex(true));
                shutdown(0);
            }
        }
    }
}
