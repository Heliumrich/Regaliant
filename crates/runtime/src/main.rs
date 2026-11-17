//! dsport: runs the PS5 Demon's Souls executable (PPSA01341 01.007.000) natively
//! on Windows x86-64 with high-level emulation of the PS5 system libraries.

// No console window: players start it from the launcher or Explorer (redirected output still
// reaches development tools).
#![windows_subsystem = "windows"]

#[macro_use]
mod log;
mod ehframe;
mod achievements;
mod audio;
mod cpucheck;
mod exception;
mod gpu;
mod guestprof;
mod heap;
mod heapsites;
mod hosted;
mod hostcensus;
mod placement;
mod hostload;
mod hle;
mod input;
mod loader;
mod mem;
mod sections;
mod spatial;
mod patch;
mod pcsettings;
mod jobidle;
mod camhook;
mod dispmode;
mod display;
mod displayclock;
mod jitter;
mod engine;
mod thread;
mod titlefix;
mod vgdiag;
mod cullread;
mod cpudraw;
mod titlemem;
mod textbox;
mod uistate;
mod tls;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

pub struct Config {
    /// Where the port keeps its files (`dsconfig::paths`: the checkout's local-data, a portable
    /// folder, or the player's AppData folders).
    pub paths: dsconfig::paths::Paths,
    /// Why the game folder cannot be used (told to the player at start), if so.
    pub game_problem: Option<String>,
    /// Application root (`PPSA01341-app0`): read-only assets mounted at /app0.
    pub app: PathBuf,
    /// Directory holding the decrypted executable and `sce_module/`.
    pub code: PathBuf,
    pub run_dir: PathBuf,
    pub seconds: Option<u64>,
    /// Development control channel (see `poll_control`).
    pub control: Option<PathBuf>,
    /// `--cache <dir>`: where compiled pipelines (`pipelines/`) and the
    /// learned shader specs (`shader-specs.txt`, `.code`) are kept, instead
    /// of local-data (play-test packages extracted side by side share one).
    pub cache: Option<PathBuf>,
    /// `--pipeline-test <capture.txt>`: create pipelines for a captured frame and exit.
    pub pipeline_test: Option<String>,
    /// `--record-bench <capture> [<cs address>]` (`gpu::exec::record_bench`).
    pub record_bench: Option<(String, Option<u64>)>,
    /// `--compile-bench [min instructions] [max programs]` (`gpu::compbench`).
    pub compile_bench: Option<Vec<String>>,
    /// `--precompile-all [threads]` (`gpu::compbench::precompile_all`).
    pub precompile_all: Option<usize>,
    /// `--translation-fingerprint`: prints `translation = <fingerprint>` (the launcher's check
    /// that the prepared shaders match this build: `gpu::compbench::translation_fingerprint`).
    pub translation_fingerprint: bool,
    /// `--progress <file>`: the precompile's progress for the launcher.
    pub progress: Option<PathBuf>,
    /// `--safe-mode` (the launcher's, for one start): the PC settings' defaults for this session
    /// only, in a window, without an upscaler, with the GPU front end's cautious fallbacks.
    pub safe_mode: bool,
    /// `--dispatch-bench` (`gpu::dispbench`).
    pub dispatch_bench: bool,
    pub trace: bool,
    pub lenient: bool,
    pub debug: bool,
    pub headless: bool,
}

static RUN_DIR: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();
static PATHS: std::sync::OnceLock<dsconfig::paths::Paths> = std::sync::OnceLock::new();

/// Where the port keeps its files (set at start).
pub fn paths() -> &'static dsconfig::paths::Paths {
    PATHS.get_or_init(dsconfig::paths::Paths::current)
}

/// This run's output directory (logs, captures).
pub fn run_dir() -> Option<&'static Path> {
    RUN_DIR.get().map(|p| p.as_path())
}

/// The host heap (`heap`: mimalloc or the Windows heap) reporting implausibly
/// large requests (almost always a guest value misread as a size) with the
/// requesting host frames, before the request fails and aborts the process.
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
        let p = unsafe { heap::HostAlloc.alloc(l) };
        if heapsites::tracked(l.size()) {
            heapsites::note_alloc(p, l.size());
        }
        p
    }
    unsafe fn alloc_zeroed(&self, l: std::alloc::Layout) -> *mut u8 {
        if l.size() > HUGE_ALLOC {
            report_huge_alloc(l.size());
        }
        let p = unsafe { heap::HostAlloc.alloc_zeroed(l) };
        if heapsites::tracked(l.size()) {
            heapsites::note_alloc(p, l.size());
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: std::alloc::Layout) {
        if heapsites::tracked(l.size()) {
            heapsites::note_free(p);
        }
        unsafe { heap::HostAlloc.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: std::alloc::Layout, new_size: usize) -> *mut u8 {
        if new_size > HUGE_ALLOC {
            report_huge_alloc(new_size);
        }
        if heapsites::tracked(l.size()) {
            heapsites::note_free(p);
        }
        let q = unsafe { heap::HostAlloc.realloc(p, l, new_size) };
        if heapsites::tracked(new_size) {
            heapsites::note_alloc(q, new_size);
        }
        q
    }
}

#[global_allocator]
static ALLOCATOR: GuardedAlloc = GuardedAlloc;

pub(crate) fn utc_stamp() -> String {
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
    let mut paths = dsconfig::paths::Paths::current();
    let mut seconds = None;
    let mut trace = false;
    let mut lenient = false;
    let mut debug = false;
    let mut headless = false;
    let mut control = None;
    let mut cache = None;
    let mut app = None;
    let mut pipeline_test = None;
    let mut compile_bench: Option<Vec<String>> = None;
    let mut precompile_all: Option<usize> = None;
    let mut translation_fingerprint = false;
    let mut progress = None;
    let mut safe_mode = false;
    let mut dispatch_bench = false;
    let mut record_bench = None;
    let mut args = std::env::args().skip(1).peekable();
    while let Some(a) = args.next() {
        match a.as_str() {
            "--seconds" => seconds = args.next().and_then(|s| s.parse().ok()),
            "--trace" => trace = true,
            "--lenient" => lenient = true,
            "--debug" => debug = true,
            "--headless" => headless = true,
            // (Read by `hosted::wants_host`: front end 2 unhosted, its arming off.)
            "--no-host" => {}
            "--root" => {
                // A development checkout's root (its local-data holds everything).
                let root = PathBuf::from(args.next().expect("--root needs a path"));
                let data = root.join("local-data");
                paths = dsconfig::paths::Paths { layout: dsconfig::paths::Layout::Dev, exe_dir: paths.exe_dir.clone(), user: data.clone(), cache: data, root: Some(root) };
            }
            "--control" => control = Some(PathBuf::from(args.next().expect("--control needs a path"))),
            "--cache" => cache = Some(PathBuf::from(args.next().expect("--cache needs a directory"))),
            "--app" => app = Some(PathBuf::from(args.next().expect("--app needs a path"))),
            "--pipeline-test" => pipeline_test = Some(args.next().expect("--pipeline-test needs a capture file")),
            "--dispatch-bench" => dispatch_bench = true,
            "--progress" => progress = Some(PathBuf::from(args.next().expect("--progress needs a file"))),
            "--safe-mode" => safe_mode = true,
            "--pb-test" => std::process::exit(gpu::pcache::pb_test()),
            "--compile-bench" => {
                let min = args.next().expect("--compile-bench needs <min instructions> <max programs>");
                let max = args.next().expect("--compile-bench needs <min instructions> <max programs>");
                compile_bench = Some(vec![min, max]);
            }
            "--precompile-all" => {
                // (An optional thread count.)
                let n = args.peek().and_then(|a| a.parse::<usize>().ok());
                if n.is_some() {
                    args.next();
                }
                precompile_all = Some(n.unwrap_or(8));
            }
            "--translation-fingerprint" => translation_fingerprint = true,
            "--record-bench" => {
                let file = args.next().expect("--record-bench needs a capture file");
                let cs = args.next().and_then(|a| u64::from_str_radix(a.trim_start_matches("0x"), 16).ok());
                record_bench = Some((file, cs));
            }
            other => {
                eprintln!("unknown argument {other}");
                std::process::exit(2);
            }
        }
    }
    if safe_mode {
        // The GPU front end's cautious fallbacks (no arming of GPU-written pages, no held
        // pages) and no upscaler library loaded: set before any thread starts or reads them.
        for (k, v) in [("DS_FE2_ARM_GPU", "0"), ("DS_C4", "0"), ("DS_UPSCALER", "off")] {
            if std::env::var_os(k).is_none() {
                // SAFETY: still the only thread (parse_args runs before any is started).
                unsafe { std::env::set_var(k, v) };
            }
        }
    }
    // The game: `--app`, else the folder chosen in the launcher (or the checkout's own copy),
    // checked as the launcher checks it (`dsconfig::game`).
    let chosen = app.or_else(|| paths.chosen_game());
    let (app, code, game_problem) = match chosen.as_deref().map(dsconfig::game::check) {
        Some(Ok(g)) => (g.app, g.code, None),
        Some(Err(e)) => {
            let dir = chosen.clone().unwrap_or_default();
            (dir.clone(), dir.join("decrypted"), Some(format!("{e}\n\nThe game folder: {}\nStart {} to choose another.", dir.display(), dsconfig::paths::LAUNCHER_EXE)))
        }
        None => (PathBuf::new(), PathBuf::new(), Some(format!("The game folder is not set. Start {} and choose it.", dsconfig::paths::LAUNCHER_EXE))),
    };
    let run_dir = paths.logs().join(utc_stamp());
    Config {
        code,
        app,
        run_dir,
        paths,
        game_problem,
        seconds,
        trace,
        lenient,
        debug,
        // The launcher's translation check and shader preparation (and the compile bench) run
        // no game: no window (each showed an empty one until 2026-11-14) and no message boxes.
        headless: headless || translation_fingerprint || precompile_all.is_some() || compile_bench.is_some(),
        control,
        cache,
        pipeline_test,
        record_bench,
        compile_bench,
        precompile_all,
        translation_fingerprint,
        progress,
        safe_mode,
        dispatch_bench,
    }
}

/// Keeps the newest `keep` run directories without a KEEP marker
/// (`DS_KEEP_RUNS=<n>` sets the number); those with one are never removed
/// and take no place among them (until 2026-11-05 they did, and with ~170
/// marked the unmarked ones lasted ~30 boots). A run marks itself when its
/// GPU device is lost or it crashes (`keep_run`).
fn prune_runs(runs: &Path, keep: usize) {
    let keep = std::env::var("DS_KEEP_RUNS").ok().and_then(|v| v.parse().ok()).unwrap_or(keep);
    let Ok(rd) = std::fs::read_dir(runs) else { return };
    let mut dirs: Vec<PathBuf> = rd.flatten().map(|e| e.path()).filter(|p| p.is_dir() && !p.join("KEEP").exists()).collect();
    dirs.sort();
    if dirs.len() <= keep {
        return;
    }
    let n = dirs.len() - keep;
    for d in dirs.into_iter().take(n) {
        let _ = std::fs::remove_dir_all(&d);
    }
}

/// Keeps this run's directory from pruning (`prune_runs`): a line naming
/// why, appended to its KEEP marker.
pub fn keep_run(reason: &str) {
    if let Some(dir) = RUN_DIR.get() {
        use std::io::Write;
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(dir.join("KEEP")) {
            let _ = writeln!(f, "{reason}");
        }
    }
}

/// Development control channel (`--control <file>`): executes complete lines
/// appended to the file since the last poll.
///   press <buttons> [hold]   e.g. `press down`, `press cross 10`, `press ls_up+r1 30`; the hold
///                            is in flips (default 6), or wall-clock with a suffix: `4s`, `300ms`
///   key <input> [ms]         hold a key or mouse button of the keyboard and mouse layout
///                            (controls.txt names: `key W 1000`, `key Mouse1`; default 200 ms)
///   mouse <dx> <dy>          mouse motion in counts; `wheel <n>` wheel notches (up positive)
///   capture on|off           mouse input as if the window had captured the cursor (headless)
///   stick l|r <x> <y> [ms]   hold a stick at a tilt (-128..127 from the centre; default 1000 ms)
///   find32 <hex u32>|<lo>-<hi>  guest addresses (4-byte aligned) holding a 32-bit value (first 64)
///   fscan start|changed|same|floats [+|-|=|*]  search the title's flexible memory for a value by
///                            how it changes (pages, then f32 values; see `fscan`)
///   fmat                     yaw-only rotation matrices in the title's flexible memory (see `fmat`)
///   cam                      the gameplay camera's controllers (raw mouse look), with
///                            DS_CAMDEBUG=1 also its rigs
///   pcset <setting> <value>  a PC setting (`Setting_PC_...`) as if changed in the SETTINGS screen
///   convars                  the PC settings' title console variables and their values
///   convar <name> <value>    one of them set (`title`: the title's own again; tests)
///   shot <name>              screenshot of the next presented flip -> shot_<name>.png
///   burst <name> <n> [x y w h] [half]  shots of the next n flips, each one (cropped, at half size) -> <name>_NNN.png
///   dump                     dump all GPU images at the next flip
///   trace                    log every draw/dispatch of an upcoming flip (FRAME lines)
///   loglevel trace|debug|info  change logging at run time (trace = every HLE call)
///   textreset                log text-hook strings again (`DS_TEXT_HOOK`)
///   mem <addr> <len>         write guest memory (hex) to mem-<addr>.bin in the run directory
///   emucheck <cs,..>[:<n>]   interpreter check of the next dispatches of those shaders (`off`)
///   find <hex u64>|<lo>-<hi> guest addresses (8-byte aligned) holding that value (or a value
///                            in [lo, hi), listed) (first 64)
///   watch <slot> <addr>|off [w|rw] [len] [all|guest]  hardware watchpoint (DR0-3) at run time
///   findstr <text>           guest addresses holding that ASCII text (first 64)
///   poke <addr> <u64> [<mask>]  write guest memory (experiments; logged)
///   note <text>              log the text (a marker for scripted runs)
///   quit                     orderly shutdown
fn poll_control(path: &Path, offset: &mut u64) {
    use std::io::{Read, Seek, SeekFrom};
    // Lines the title's automation asked for first (`INJECTED`).
    let mut buf: String = std::mem::take(&mut *INJECTED.lock()).into_iter().map(|l| l + "\n").collect();
    if let Ok(mut f) = std::fs::File::open(path) {
        let len = f.metadata().map(|m| m.len()).unwrap_or(0);
        if len < *offset {
            *offset = 0; // truncated: start over
        }
        let mut file = String::new();
        if len > *offset && f.seek(SeekFrom::Start(*offset)).is_ok() && f.read_to_string(&mut file).is_ok() {
            if let Some(end) = file.rfind('\n') {
                *offset += end as u64 + 1;
                buf.push_str(&file[..=end]);
            }
        }
    }
    if buf.is_empty() {
        return;
    }
    for line in buf.lines().map(str::trim).filter(|l| !l.is_empty() && !l.starts_with('#')) {
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
            "pcset" => {
                // `pcset <Setting_PC_...> <value>`: a PC setting as if changed in the menu.
                let name = parts.next().unwrap_or("");
                let value = parts.next().unwrap_or("");
                pcsettings::store::set_by_name(name, value)
            }
            "rsize" => {
                // Diagnosis: `rsize <w>x<h>` (0x0: the title's own) in a run without a window.
                let v = parts.next().unwrap_or("");
                match v.split_once(['x', 'X']).and_then(|(w, h)| Some((w.parse::<u32>().ok()?, h.parse::<u32>().ok()?))) {
                    Some((w, h)) => dispmode::set_headless_size(w, h),
                    None => Err(format!("rsize: '{v}' is not <w>x<h>")),
                }
            }
            "uielems" => {
                // Diagnosis: `uielems <tag>` writes every UIElement component's first bytes.
                let tag = parts.next().unwrap_or("x");
                match RUN_DIR.get() {
                    Some(dir) => uistate::dump_elements(dir, tag).map(|m| info!("{m}")),
                    None => Err("no run directory".into()),
                }
            }
            "bindinput" => {
                // Tests: `bindinput <input>` (as in controls.txt) for a key binding waiting for its key.
                pcsettings::bindings::test_input(parts.next().unwrap_or(""))
            }
            "pointer" => {
                // Tests: `pointer <x> <y>` moves the menu pointer to canvas coordinates (1920x1080);
                // `pointer screen <x> <y>` to a point of the picture as shown, through the window;
                // `pointer click|right|wheel <n>`.
                use pcsettings::pointer::{Event, event, test_move, test_screen};
                match (parts.next(), parts.next()) {
                    (Some("screen"), Some(x)) => match (x.parse::<f32>(), parts.next().and_then(|y| y.parse::<f32>().ok())) {
                        (Ok(x), Some(y)) => test_screen(x, y),
                        _ => Err("usage: pointer screen <x> <y>".into()),
                    },
                    (Some("click"), _) => {
                        event(Event::Left);
                        Ok(())
                    }
                    (Some("right"), _) => {
                        event(Event::Right);
                        Ok(())
                    }
                    (Some("wheel"), Some(n)) => n.parse::<i32>().map(|n| event(Event::Wheel(n))).map_err(|e| e.to_string()),
                    (Some(x), Some(y)) => match (x.parse::<f32>(), y.parse::<f32>()) {
                        (Ok(x), Ok(y)) => {
                            test_move(x, y);
                            Ok(())
                        }
                        _ => Err("usage: pointer <x> <y> | click | right | wheel <n>".into()),
                    },
                    _ => Err("usage: pointer <x> <y> | click | right | wheel <n>".into()),
                }
            }
            "uinodes" => {
                // Diagnosis: `uinodes <tag>` writes every runtime UI node's first bytes.
                let tag = parts.next().unwrap_or("x");
                match RUN_DIR.get() {
                    Some(dir) => uistate::dump_nodes(dir, tag).map(|m| info!("{m}")),
                    None => Err("no run directory".into()),
                }
            }
            "uistack" => {
                // Diagnosis: the title's UI manager: the active menu stack and HUD scenes.
                info!("{}", uistate::report());
                Ok(())
            }
            "convars" => {
                crate::pcsettings::convar::report();
                Ok(())
            }
            "convar" => {
                // `convar <name> <value|title>`: one of the PC settings' title console variables.
                let name = parts.next().unwrap_or("");
                let value = parts.next().unwrap_or("");
                crate::pcsettings::convar::set_for_test(name, value)
            }
            "shot" => {
                let name: String = parts.next().unwrap_or("x").chars().filter(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-').collect();
                *hle::libs::videoout::SHOT_REQUEST.lock() = Some(name);
                Ok(())
            }
            "framewatch" => {
                // `framewatch 1|0`: the presenter's one-frame flash watch (`gpu::vk::framewatch`).
                let on = parts.next() != Some("0");
                gpu::vk::framewatch::ON.store(on, std::sync::atomic::Ordering::Relaxed);
                info!("control: frame watch {}", if on { "on" } else { "off" });
                Ok(())
            }
            "burst" => {
                let name: String = parts.next().unwrap_or("burst").chars().filter(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-').collect();
                let count = parts.next().and_then(|v| v.parse::<u32>().ok()).unwrap_or(30).clamp(1, 600);
                let rest: Vec<&str> = parts.collect();
                let half = rest.contains(&"half");
                let nums: Vec<u32> = rest.iter().filter_map(|v| v.parse().ok()).collect();
                let crop = (nums.len() == 4).then(|| (nums[0], nums[1], nums[2], nums[3]));
                *hle::libs::videoout::BURST.lock() = Some(hle::libs::videoout::Burst { name, left: count, index: 0, crop, half });
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
            "find" => {
                // Debug: `find <hex u64>` or `find <hex lo>-<hex hi>`: readable
                // guest memory holding the value (or a value in [lo, hi)) at an
                // 8-byte aligned address (first 64 hits; ranges list the values).
                let hex = |s: &str| u64::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok();
                let arg = parts.next().unwrap_or("");
                let range = match arg.split_once('-') {
                    Some((lo, hi)) => hex(lo).zip(hex(hi)),
                    None => hex(arg).map(|v| (v, v + 1)),
                };
                match range {
                    Some((vlo, vhi)) => {
                        let spans: Vec<(u64, u64, String)> = mem::MEM.lock().vmas.values().filter(|m| m.kind != mem::VmaKind::Reserved && m.prot & 1 != 0).map(|m| (m.start, m.end(), m.name.clone())).collect();
                        let mut hits = Vec::new();
                        let t0 = Instant::now();
                        for (lo, hi, name) in &spans {
                            let mut a = (*lo + 7) & !7;
                            while a + 8 <= *hi && hits.len() < 64 {
                                let v = unsafe { *(a as *const u64) };
                                if (vlo..vhi).contains(&v) {
                                    hits.push(if vhi - vlo > 1 { format!("{a:#x} ({name}) = {v:#x}") } else { format!("{a:#x} ({name})") });
                                }
                                a += 8;
                            }
                        }
                        info!("find {arg}: {} hit(s) in {} mappings ({} ms): {}", hits.len(), spans.len(), t0.elapsed().as_millis(), hits.join(", "));
                        Ok(())
                    }
                    None => Err("usage: find <hex u64> | <hex lo>-<hex hi>".into()),
                }
            }
            "fmat" => fmat(),
            "note" => {
                // A marker in the log for scripted runs.
                info!("control: note {}", parts.by_ref().collect::<Vec<_>>().join(" "));
                Ok(())
            }
            "camdump" => {
                // Debug (`DS_CAMDEBUG=1`): the camera rigs' controllers' fields.
                info!("camdump:
{}", camhook::dump());
                Ok(())
            }
            "cam" => {
                // Debug: the gameplay camera's controllers (`camhook`) and,
                // with DS_CAMDEBUG=1, its rigs.
                info!("cam: {}", camhook::describe());
                Ok(())
            }
            "wshot" => {
                // The next presented frame as the window shows it
                // (`window-<n>.png` in the run directory).
                gpu::vk::present::shot_next();
                Ok(())
            }
            "window" => {
                // `window fullscreen|windowed|toggle`: what Alt+Enter does;
                // `window size <w>x<h>`: the windowed window's client size.
                let size = |v: &str| v.split_once('x').and_then(|(w, h)| Some((w.parse::<u32>().ok()?, h.parse::<u32>().ok()?))).filter(|&(w, h)| (160..=16384).contains(&w) && (90..=16384).contains(&h));
                match (parts.next(), parts.next()) {
                    (Some("fullscreen"), _) => Ok(gpu::vk::window::set_fullscreen(Some(true))),
                    (Some("windowed"), _) => Ok(gpu::vk::window::set_fullscreen(Some(false))),
                    (Some("toggle"), _) => Ok(gpu::vk::window::set_fullscreen(None)),
                    (Some("exclusive"), _) => Ok(gpu::vk::window::set_window_mode(2)),
                    // `window background` / `window foreground`: what Alt+Tab away and back does
                    // (the window's own activation messages).
                    (Some("background"), _) => Ok(gpu::vk::window::simulate_activation(false)),
                    (Some("foreground"), _) => Ok(gpu::vk::window::simulate_activation(true)),
                    // `window displaychange`: what a monitor's mode change sends the window.
                    (Some("displaychange"), _) => Ok(gpu::vk::window::simulate_display_change()),
                    (Some("size"), Some(v)) if size(v).is_some() => {
                        let (w, h) = size(v).unwrap();
                        gpu::vk::window::set_client_size(w, h);
                        Ok(())
                    }
                    _ => Err("usage: window fullscreen|exclusive|windowed|toggle|background|foreground | window size <w>x<h>".into()),
                }
            }
            "fscan" => {
                // Debug (finding a value the title keeps, e.g. the camera's
                // angles): `fscan start` hashes the pages of the title's
                // flexible memory; `fscan changed` keeps the pages that changed
                // since (intersected with earlier ones), `fscan same` drops the
                // pages that changed and copies the rest once few remain;
                // `fscan floats [+|-|=|*]` lists the copied pages' f32 values
                // that rose, fell, stayed or changed since the copy (keeping
                // those as the candidates, first 48 listed) and copies again.
                fscan(parts.next().unwrap_or(""), parts.next().unwrap_or("*"))
            }
            "find32" => {
                // Debug: `find32 <hex u32>` or `find32 <hex lo>-<hex hi>`: readable
                // guest memory holding the 32-bit value (or one in [lo, hi)) at a
                // 4-byte aligned address (first 64 hits, with the value as f32).
                let hex = |s: &str| u32::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok();
                let arg = parts.next().unwrap_or("");
                let range = match arg.split_once('-') {
                    Some((lo, hi)) => hex(lo).zip(hex(hi)),
                    None => hex(arg).map(|v| (v, v.saturating_add(1))),
                };
                match range {
                    Some((vlo, vhi)) => {
                        let spans: Vec<(u64, u64, String)> = mem::MEM.lock().vmas.values().filter(|m| m.kind != mem::VmaKind::Reserved && m.prot & 1 != 0).map(|m| (m.start, m.end(), m.name.clone())).collect();
                        let mut hits = Vec::new();
                        let t0 = Instant::now();
                        for (lo, hi, name) in &spans {
                            let mut a = (*lo + 3) & !3;
                            while a + 4 <= *hi && hits.len() < 64 {
                                let v = unsafe { *(a as *const u32) };
                                if (vlo..vhi).contains(&v) {
                                    hits.push(format!("{a:#x} ({name}) = {v:#x} ({})", f32::from_bits(v)));
                                }
                                a += 4;
                            }
                        }
                        info!("find32 {arg}: {} hit(s) in {} mappings ({} ms): {}", hits.len(), spans.len(), t0.elapsed().as_millis(), hits.join(", "));
                        Ok(())
                    }
                    None => Err("usage: find32 <hex u32> | <hex lo>-<hex hi>".into()),
                }
            }
            "stick" => {
                // Diagnosis: `stick l|r <x> <y> [ms]`: holds a stick at a tilt
                // (-128..127 from the centre, y down; default 1000 ms).
                let side = parts.next();
                let x = parts.next().and_then(|v| v.parse::<i32>().ok());
                let y = parts.next().and_then(|v| v.parse::<i32>().ok());
                let ms = parts.next().and_then(|v| v.parse::<u64>().ok()).unwrap_or(1000);
                match (side, x, y) {
                    (Some(sd @ ("l" | "r")), Some(x), Some(y)) => {
                        input::hold_stick(sd == "r", x, y, Duration::from_millis(ms));
                        Ok(())
                    }
                    _ => Err("usage: stick l|r <x> <y> [ms]".into()),
                }
            }
            "poke" => {
                // Experiment: `poke <hex address> <hex u64> [<hex mask>]`: writes
                // (old & !mask) | (value & mask) to the 8-byte aligned guest
                // address (mask: all bits by default); logs old and new values.
                let hex = |s: Option<&str>| s.and_then(|v| u64::from_str_radix(v.trim_start_matches("0x"), 16).ok());
                match (hex(parts.next()), hex(parts.next())) {
                    (Some(a), Some(v)) if a % 8 == 0 && gpu::cp::guest_readable(a, 8) => {
                        let mask = hex(parts.next()).unwrap_or(u64::MAX);
                        let p = a as *mut u64;
                        let old = unsafe { p.read_volatile() };
                        let new = (old & !mask) | (v & mask);
                        unsafe { p.write_volatile(new) };
                        info!("control: poke {a:#x}: {old:#018x} -> {new:#018x}");
                        Ok(())
                    }
                    _ => Err("usage: poke <hex 8-byte aligned address> <hex u64> [<hex mask>]".into()),
                }
            }
            "findstr" => {
                // Debug: `findstr <text>`: readable guest memory holding the
                // ASCII text (first 64 hits; the rest of the line is the text).
                let text = line.trim_start().strip_prefix("findstr").unwrap_or("").trim();
                if text.is_empty() {
                    Err("usage: findstr <text>".into())
                } else {
                    let needle = text.as_bytes();
                    let spans: Vec<(u64, u64, String)> = mem::MEM.lock().vmas.values().filter(|m| m.kind != mem::VmaKind::Reserved && m.prot & 1 != 0).map(|m| (m.start, m.end(), m.name.clone())).collect();
                    let mut hits = Vec::new();
                    let t0 = Instant::now();
                    for (lo, hi, name) in &spans {
                        if hits.len() >= 64 || hi - lo < needle.len() as u64 {
                            continue;
                        }
                        let hay = unsafe { std::slice::from_raw_parts(*lo as *const u8, (hi - lo) as usize) };
                        let mut i = 0;
                        while let Some(k) = hay[i..].windows(needle.len()).position(|w| w == needle) {
                            hits.push(format!("{:#x} ({name})", lo + (i + k) as u64));
                            i += k + 1;
                            if hits.len() >= 64 {
                                break;
                            }
                        }
                    }
                    info!("findstr {text}: {} hit(s) in {} mappings ({} ms): {}", hits.len(), spans.len(), t0.elapsed().as_millis(), hits.join(", "));
                    Ok(())
                }
            }
            "key" => {
                // Diagnosis: `key <input> [ms]` holds a key or mouse button of
                // the keyboard and mouse layout (default 200 ms).
                let name = parts.next().unwrap_or("");
                let ms = parts.next().and_then(|v| v.parse::<u64>().ok()).unwrap_or(200);
                input::inject_hold(name, Duration::from_millis(ms))
            }
            "mousemove" => {
                // Diagnosis: `mousemove <dx/s> <dy/s> <ms>` steady mouse motion (counts a second)
                // fed at every vblank, as a polled mouse delivers it.
                let dx = parts.next().and_then(|v| v.parse::<f64>().ok());
                let dy = parts.next().and_then(|v| v.parse::<f64>().ok());
                let ms = parts.next().and_then(|v| v.parse::<u64>().ok());
                match (dx, dy, ms) {
                    (Some(dx), Some(dy), Some(ms)) => {
                        input::inject_motion_for(dx, dy, std::time::Duration::from_millis(ms));
                        Ok(())
                    }
                    _ => Err("usage: mousemove <dx/s> <dy/s> <ms>".into()),
                }
            }
            "mouse" => {
                // Diagnosis: `mouse <dx> <dy>` mouse motion in counts.
                let dx = parts.next().and_then(|v| v.parse::<i64>().ok());
                let dy = parts.next().and_then(|v| v.parse::<i64>().ok());
                match (dx, dy) {
                    (Some(dx), Some(dy)) => {
                        input::inject_motion(dx, dy, 0);
                        Ok(())
                    }
                    _ => Err("usage: mouse <dx> <dy>".into()),
                }
            }
            "wheel" => {
                // Diagnosis: `wheel <notches>` (up positive).
                match parts.next().and_then(|v| v.parse::<i32>().ok()) {
                    Some(n) => {
                        input::inject_motion(0, 0, n);
                        Ok(())
                    }
                    None => Err("usage: wheel <notches>".into()),
                }
            }
            "capture" if matches!(parts.clone().next(), Some("on" | "off")) => {
                // Diagnosis: `capture on|off`: mouse input counts as if the
                // window had captured the cursor (headless runs). Any other
                // `capture` is a frame capture (below).
                let on = parts.next() == Some("on");
                gpu::vk::window::CAPTURE_TEST.store(on, std::sync::atomic::Ordering::Relaxed);
                Ok(())
            }
            "watch" => {
                // Debug: `watch <slot 0-3> <hex address> [w|rw] [1|2|4|8] [all|guest]`
                // or `watch <slot> off`: hardware watchpoint on every
                // registered thread (writes: all threads by default; rw:
                // guest threads by default). Hits are logged (`watch N ...`).
                let slot = parts.next().and_then(|v| v.parse::<usize>().ok()).filter(|&s| s < 4);
                let arg = parts.next().unwrap_or("");
                let addr = if arg == "off" { Some(0) } else { u64::from_str_radix(arg.trim_start_matches("0x"), 16).ok() };
                let (mut rw, mut len, mut guest) = (1u32, 8u32, None::<bool>);
                for o in parts.by_ref() {
                    match o {
                        "w" => rw = 1,
                        "rw" => rw = 3,
                        "1" | "2" | "4" | "8" => len = o.parse().unwrap(),
                        "all" => guest = Some(false),
                        "guest" => guest = Some(true),
                        _ => {}
                    }
                }
                match (slot, addr) {
                    (Some(s), Some(a)) => {
                        let guest_only = guest.unwrap_or(rw == 3);
                        let n = exception::set_watch(s, a, rw, len, guest_only);
                        info!("control: watch {s} {a:#x} {} len {len}{}: {n} thread(s) armed", if rw == 3 { "rw" } else { "w" }, if guest_only { " (guest threads)" } else { "" });
                        Ok(())
                    }
                    _ => Err("usage: watch <slot 0-3> <hex address>|off [w|rw] [1|2|4|8] [all|guest]".into()),
                }
            }
            "emucheck" => {
                // Diagnosis: `emucheck <hex shader>[,...][:<groups>]` / `emucheck off`.
                let rest = line.trim_start().strip_prefix("emucheck").unwrap_or("").trim();
                gpu::emucheck::arm(rest)
            }
            "probe" => {
                // Diagnosis: `probe <shader>:<pc>:<v<k>|s<k>>[u|i][:<frames>]` / `probe off`.
                let rest = line.trim_start().strip_prefix("probe").unwrap_or("").trim();
                gpu::shaders::set_probe(rest)
            }
            "csdump" => {
                // Diagnosis: `csdump <hex compute shader address>`: its next dispatch writes
                // its buffer resources' guest memory to the run directory.
                match parts.next().and_then(|a| u64::from_str_radix(a.trim_start_matches("0x"), 16).ok()) {
                    Some(a) => {
                        gpu::exec::request_cs_dump(a);
                        Ok(())
                    }
                    None => Err("usage: csdump <hex address>".into()),
                }
            }
            "skipcs" => {
                // Diagnosis: `skipcs <hex compute shader address>` toggles
                // skipping its dispatches.
                match parts.next().and_then(|a| u64::from_str_radix(a.trim_start_matches("0x"), 16).ok()) {
                    Some(a) => {
                        let on = gpu::exec::toggle_skip_cs(a);
                        info!("control: dispatches of compute shader {a:#x} {}", if on { "skipped" } else { "run again" });
                        Ok(())
                    }
                    None => Err("usage: skipcs <hex address>".into()),
                }
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
            "engine" => {
                // `engine reset` restarts the engine census and front end 4's
                // counters; `engine report` logs them (`engine`).
                match parts.next() {
                    Some("reset") => {
                        engine::reset();
                        Ok(())
                    }
                    Some("report") => {
                        info!("engine report at flip {flip}:\n{}", engine::report());
                        Ok(())
                    }
                    _ => Err("usage: engine reset|report".into()),
                }
            }
            "profile" => {
                // `profile reset` restarts the DS_PROFILE_GPU sampling and
                // dispatch timings; `profile report` logs them.
                match parts.next() {
                    Some("reset") => {
                        gpu::reset_profile();
                        Ok(())
                    }
                    Some("report") => {
                        info!("profile at flip {flip}:\n{}", gpu::profile_report());
                        Ok(())
                    }
                    _ => Err("usage: profile reset|report".into()),
                }
            }
            "replay" => {
                // `replay <iterations> [<switch> <a> <b>] [timeline=<k>]`: one
                // frame of the title replayed through the GPU front end with
                // the title held, optionally alternating a `tune` switch, and
                // the timeline of the last k iterations (`gpu::replay`).
                let n = parts.next().and_then(|v| v.parse::<u32>().ok()).filter(|n| (1..=200).contains(n));
                let mut rest: Vec<&str> = parts.by_ref().collect();
                let mut timeline = 0u32;
                if let Some(k) = rest.last().and_then(|t| t.strip_prefix("timeline=")).and_then(|k| k.parse().ok()) {
                    timeline = k;
                    rest.pop();
                }
                let ab = match rest.as_slice() {
                    [] => Ok(None),
                    [name, a, b] => match (a.parse::<u64>(), b.parse::<u64>()) {
                        (Ok(a), Ok(b)) => Ok(Some((name.to_string(), a, b))),
                        _ => Err(()),
                    },
                    _ => Err(()),
                };
                match (n, ab) {
                    (Some(n), Ok(ab)) => gpu::replay::start(n, ab, timeline),
                    _ => Err("usage: replay <iterations 1-200> [<switch> <a> <b>] [timeline=<k>]".into()),
                }
            }
            "timeline" => {
                // `timeline <flips>`: spans of command processing and GPU
                // batches over the next flips (`gpu::timeline`).
                match parts.next().and_then(|n| n.parse::<u32>().ok()) {
                    Some(n) if n > 0 => gpu::timeline::start(n),
                    _ => Err("usage: timeline <flips>".into()),
                }
            }
            "labels" => {
                // `labels <n>`: the last n events of the queues' label log
                // (waits, CP writes, flips) to labels-<flip>.txt in the run
                // directory (`gpu::cp::label_dump`, `tools/labelflow.py`).
                match (parts.next().and_then(|n| n.parse::<usize>().ok()), run_dir()) {
                    (Some(n), Some(dir)) if n > 0 => std::fs::write(dir.join(format!("labels-{flip}.txt")), gpu::cp::label_dump(n)).map_err(|e| e.to_string()),
                    _ => Err("usage: labels <events>".into()),
                }
            }
            "stacks" => {
                info!("guest thread stacks:\n{}", thread_summary_ex(true));
                Ok(())
            }
            "status" => {
                // The status report without stacks (GPU, image cache with its
                // upload breakdown, audio, file traffic).
                info!("status at flip {flip}:\n{}", thread_summary());
                Ok(())
            }
            "iostat" => {
                info!("file traffic at flip {flip}:\n{}", hle::iostat::status());
                Ok(())
            }
            "wbs" => {
                // `wbs on|off` (`DS_WB_SAME=2`): write-backs leave out the
                // pages they would store unchanged.
                match gpu::fe2::wbs_control(parts.next().unwrap_or("")) {
                    Ok(m) => {
                        info!("control: wbs: {m} at flip {flip}");
                        Ok(())
                    }
                    Err(e) => Err(e),
                }
            }
            "ptaudit" => {
                // `ptaudit` (diagnosis): the guest page table read back,
                // each entry checked against the live import windows.
                info!("control: ptaudit at flip {flip}: {}", gpu::exec::pt_audit());
                Ok(())
            }
            "maskwait" => {
                // `maskwait off|on` (A/B, `cullread`): the title's walks of
                // its GPU frustum culling masks wait for the GPU's writes.
                match parts.next().unwrap_or("") {
                    v @ ("on" | "1" | "off" | "0") => {
                        let on = matches!(v, "on" | "1");
                        cullread::MASK_WAIT_ON.store(on, std::sync::atomic::Ordering::Relaxed);
                        info!("control: maskwait {} at flip {flip}", if on { "on" } else { "off" });
                        Ok(())
                    }
                    other => Err(format!("maskwait: on or off, not {other:?}")),
                }
            }
            "c4" => {
                // `c4 off|on` (`DS_C4=1`): front end 4's held pages (off
                // releases them all).
                match gpu::fe2::c4_control(parts.next().unwrap_or("")) {
                    Ok(m) => {
                        info!("control: c4: {m} at flip {flip}");
                        Ok(())
                    }
                    Err(e) => Err(e),
                }
            }
            "armbehind" => {
                // `armbehind off|on` (front end 2's arming behind the
                // graphics CP and queue 0x50): off disarms what it armed.
                match gpu::fe2::behind_control(parts.next().unwrap_or("")) {
                    Ok(m) => {
                        info!("control: armbehind: {m} at flip {flip}");
                        Ok(())
                    }
                    Err(e) => Err(e),
                }
            }
            "c4arm" => {
                // `c4arm off|on` (front end 4's own arming): off releases
                // the held pages its rounds armed and holds none of them.
                match gpu::fe2::c4_arm_control(parts.next().unwrap_or("")) {
                    Ok(m) => {
                        info!("control: c4arm: {m} at flip {flip}");
                        Ok(())
                    }
                    Err(e) => Err(e),
                }
            }
            "c5" => {
                // `c5 off|on` (front end 4's C5): stores into held pages
                // marked again, or not.
                match gpu::fe2::c5_control(parts.next().unwrap_or("")) {
                    Ok(m) => {
                        info!("control: c5: {m} at flip {flip}");
                        Ok(())
                    }
                    Err(e) => Err(e),
                }
            }
            "wbb" => {
                // `wbb lazy|eager` (`DS_WB_BUCKETS=1`, diagnosis): the
                // write-back leaves out the last census's untouched buckets.
                match gpu::fe2::wbb_control(parts.next().unwrap_or("")) {
                    Ok(m) => {
                        info!("control: wbb: {m} at flip {flip}");
                        Ok(())
                    }
                    Err(e) => Err(e),
                }
            }
            "audio" => {
                // `audio capture <seconds>`: the mix as played, to
                // `<run dir>/audio-<n>.wav`.
                match (parts.next(), parts.next().and_then(|s| s.parse::<f64>().ok())) {
                    (Some("capture"), Some(s)) if s > 0.0 && s <= 600.0 => audio::start_capture(s),
                    // `audio split <seconds>`: each AudioOut2 port group's mix to its own file.
                    (Some("split"), Some(s)) if s > 0.0 && s <= 120.0 => hle::libs::audio2::start_split(s),
                    // `audio ports <seconds>`: each AudioOut port's buffers as the title wrote them.
                    (Some("ports"), Some(s)) if s > 0.0 && s <= 300.0 => hle::libs::devices::start_raw_capture(s),
                    _ => Err("usage: audio capture <seconds 0-600> | audio split <seconds 0-120> | audio ports <seconds 0-300>".into()),
                }
            }
            "frametimes" => {
                // Intervals between presented flips since the last
                // `frametimes` (`gpu::pacing`).
                info!("frame times at flip {flip}: {}", gpu::pacing::report_and_reset());
                Ok(())
            }
            "capture" => {
                // `capture [<from>-<to>] [dispatches] [images=<base>[,..]]
                // [buffers=<shader>[,..]] [pixel=<x>,<y>]`:
                // the next frame, optionally with per-draw reports for that
                // range of its events, dumps of the images its dispatches
                // write and of the buffers those compute shaders bind.
                for arg in parts.by_ref() {
                    if arg == "dispatches" {
                        gpu::capture::set_dispatch_dumps(true);
                    } else if let Some(p) = arg.strip_prefix("pixel=") {
                        // ... reporting the cb0 value of that pixel per draw.
                        let xy: Vec<u32> = p.split(',').filter_map(|v| v.parse().ok()).collect();
                        gpu::capture::set_watch_pixel((xy.len() == 2).then(|| (xy[0], xy[1])));
                    } else if let Some(list) = arg.strip_prefix("ps=") {
                        // ... reports of the targets of draws with these pixel shaders.
                        gpu::capture::set_draw_ps(list.split(',').filter_map(|a| u64::from_str_radix(a.trim_start_matches("0x"), 16).ok()).collect());
                    } else if let Some(list) = arg.strip_prefix("vs=") {
                        // ... and of draws with these vertex programs (with
                        // DS_VS_DUMP, their vertices' registers or positions).
                        gpu::capture::set_draw_vs(list.split(',').filter_map(|a| u64::from_str_radix(a.trim_start_matches("0x"), 16).ok()).collect());
                    } else if let Some(list) = arg.strip_prefix("images=") {
                        // ... only these images (PNG and raw texels) after
                        // each dispatch that writes them.
                        gpu::capture::set_dispatch_dumps(true);
                        gpu::capture::set_image_dumps(list.split(',').filter_map(|a| u64::from_str_radix(a.trim().trim_start_matches("0x"), 16).ok()).collect());
                    } else if let Some(cs) = arg.strip_prefix("buffers=") {
                        // ... and the buffers that compute shader binds.
                        gpu::capture::set_dispatch_dumps(true);
                        gpu::capture::set_buffer_dumps(cs.split(',').filter_map(|a| u64::from_str_radix(a.trim().trim_start_matches("0x"), 16).ok()).collect());
                    } else if let Some((a, b)) = arg.split_once('-') {
                        match (a.parse(), b.parse()) {
                            (Ok(a), Ok(b)) => gpu::capture::set_draw_range((a, b)),
                            _ => warn!("control: bad draw range '{a}-{b}'"),
                        }
                    }
                }
                gpu::capture::arm_next();
                Ok(())
            }
            "mem" => {
                // Debug: `mem <hex address> <hex length>` writes guest memory
                // to mem-<address>.bin in the run directory.
                let hex = |s: Option<&str>| s.and_then(|v| u64::from_str_radix(v.trim_start_matches("0x"), 16).ok());
                match (hex(parts.next()), hex(parts.next())) {
                    (Some(a), Some(n)) if n > 0 && n <= 256 << 20 => {
                        // Guest memory GPU work may read, or any readable process
                        // memory (the title's heap objects live outside those ranges).
                        if !gpu::cp::guest_readable(a, n) && !mem::os_readable(a, n) {
                            Err(format!("{a:#x}+{n:#x} is not readable memory"))
                        } else if let Some(dir) = RUN_DIR.get() {
                            let bytes = unsafe { std::slice::from_raw_parts(a as *const u8, n as usize) };
                            std::fs::write(dir.join(format!("mem-{a:x}.bin")), bytes).map_err(|e| e.to_string())
                        } else {
                            Err("no run directory".into())
                        }
                    }
                    _ => Err("usage: mem <hex address> <hex length up to 0x10000000>".into()),
                }
            }
            "fillimg" => {
                // Experiment: `fillimg <hex address> <value>` fills the cached
                // images at that guest address with one value (all channels,
                // layers and mips); `fillimg cs=<hex shader> <value>` those
                // the compute shader last bound as storage images.
                let arg = parts.next().unwrap_or("");
                let v = parts.next().and_then(|v| v.parse::<f32>().ok());
                let hex = |a: &str| u64::from_str_radix(a.trim_start_matches("0x"), 16).ok();
                match (arg.strip_prefix("cs="), v) {
                    (Some(cs), Some(v)) => match hex(cs) {
                        Some(cs) => {
                            let (bases, n) = gpu::images::with(|im| {
                                let b = im.written_by(cs);
                                let n: usize = b.iter().map(|&a| im.fill_at(a, v)).sum();
                                (b, n)
                            });
                            info!("control: fillimg cs={cs:#x} {v}: {n} image(s) filled at {bases:x?}");
                            Ok(())
                        }
                        None => Err("usage: fillimg cs=<hex shader> <value>".into()),
                    },
                    (None, Some(v)) => match hex(arg) {
                        Some(a) => {
                            let n = gpu::images::with(|im| im.fill_at(a, v));
                            info!("control: fillimg {a:#x} {v}: {n} image(s) filled");
                            Ok(())
                        }
                        None => Err("usage: fillimg <hex address>|cs=<hex shader> <value>".into()),
                    },
                    _ => Err("usage: fillimg <hex address>|cs=<hex shader> <value>".into()),
                }
            }
            "upscaler" => {
                // `upscaler off|dlss|fsr|xess`: the Anti-Aliasing's upscaler, live (tests).
                let v = parts.next().unwrap_or("off");
                gpu::upscale::set_choice(v);
                Ok(())
            }
            "upjitter" => {
                // `upjitter <x> <y>`: the upscalers' jitter scale from now on (tests).
                let x = parts.next().and_then(|v| v.parse::<f32>().ok()).unwrap_or(0.5);
                let y = parts.next().and_then(|v| v.parse::<f32>().ok()).unwrap_or(-0.5);
                gpu::upscale::set_jitter_scale(x, y);
                Ok(())
            }
            "wsinfo" | "wstrim" => {
                // Diagnosis: `wsinfo` logs the process's working set, private bytes and page
                // faults; `wstrim` empties the working set first (what play touches again in a
                // while is its hot set: the RAM it needs to run without paging).
                use windows_sys::Win32::System::ProcessStatus::{EmptyWorkingSet, GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS, PROCESS_MEMORY_COUNTERS_EX};
                use windows_sys::Win32::System::Threading::GetCurrentProcess;
                let info = || {
                    let mut c: PROCESS_MEMORY_COUNTERS_EX = unsafe { std::mem::zeroed() };
                    c.cb = std::mem::size_of::<PROCESS_MEMORY_COUNTERS_EX>() as u32;
                    unsafe { GetProcessMemoryInfo(GetCurrentProcess(), &mut c as *mut PROCESS_MEMORY_COUNTERS_EX as *mut PROCESS_MEMORY_COUNTERS, c.cb) };
                    format!("working set {} MiB (peak {}), private {} MiB, page faults {}", c.WorkingSetSize >> 20, c.PeakWorkingSetSize >> 20, c.PrivateUsage >> 20, c.PageFaultCount)
                };
                if cmd == "wstrim" {
                    info!("wstrim: before: {}", info());
                    let t = Instant::now();
                    unsafe { EmptyWorkingSet(GetCurrentProcess()) };
                    info!("wstrim: emptied in {:.0} ms: {}", t.elapsed().as_secs_f64() * 1000.0, info());
                } else {
                    info!("wsinfo: {}", info());
                }
                Ok(())
            }
            "pcstate" => {
                // Tests: every PC setting's stored value beside what is in effect, and the
                // title's console variables (settings tests: an effect, and the same after a
                // restart).
                pcsettings::state_report();
                Ok(())
            }
            "devicelost" => {
                // Tests: the GPU device taken as lost (the game ends as on a real loss).
                warn!("devicelost: the GPU device taken as lost (test)");
                gpu::exec::lose_device();
                Ok(())
            }
            "taadump" => {
                // `taadump <name>`: the TAA's input colour at its next dispatch (diagnosis).
                let name: String = parts.next().unwrap_or("taa").chars().filter(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-' || *c == ':').collect();
                *gpu::exec::TAA_DUMP.lock() = Some(name);
                Ok(())
            }
            "imgpng" => {
                // Diagnosis: `imgpng <hex address> <name>`: the GPU image at that guest address as
                // <run dir>/<name>.png.
                let a = parts.next().and_then(|a| u64::from_str_radix(a.trim_start_matches("0x"), 16).ok());
                let name: String = parts.next().unwrap_or("img").chars().filter(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-').collect();
                match (a, RUN_DIR.get()) {
                    (Some(a), Some(dir)) => {
                        let r = gpu::images::with(|im| im.dump_one(a, dir, &name));
                        info!("control: imgpng {a:#x} {name}: {}", r.as_deref().unwrap_or_else(|e| e));
                        r.map(|_| ())
                    }
                    _ => Err("usage: imgpng <hex address> <name>".into()),
                }
            }
            "imgstat" => {
                // Diagnosis: `imgstat <hex address>` logs value statistics of
                // the cached images at that guest address.
                match parts.next().and_then(|a| u64::from_str_radix(a.trim_start_matches("0x"), 16).ok()) {
                    Some(a) => {
                        let lines = gpu::images::with(|im| im.stats_at(a));
                        info!("control: imgstat {a:#x}: {} image(s)\n    {}", lines.len(), lines.join("\n    "));
                        Ok(())
                    }
                    None => Err("usage: imgstat <hex address>".into()),
                }
            }
            "tune" => {
                // Measurements: `tune <switch> <value>` (`gpu::tune`)
                // changes a GPU switch for the rest of the run.
                match (parts.next(), parts.next().and_then(|v| v.parse::<u64>().ok())) {
                    (Some(name), Some(v)) if gpu::tune::set(name, v) => {
                        info!("control: tune {name} = {v} at flip {}", gpu::FRAMES.load(std::sync::atomic::Ordering::Relaxed));
                        if name == "guest_cpus" {
                            thread::apply_guest_cpus(gpu::tune::guest_cpus());
                        }
                        if name == "power_optout" {
                            thread::apply_power_optout(v != 0);
                        }
                        Ok(())
                    }
                    _ => Err("usage: tune <switch> <value> (see gpu::tune)".into()),
                }
            }
            "tiny" => {
                // Diagnosis: `tiny <hex ps>` toggles limiting that pixel
                // shader's draws to one pixel (vertex cost vs pixel cost).
                match parts.next().and_then(|a| u64::from_str_radix(a.trim_start_matches("0x"), 16).ok()) {
                    Some(a) => {
                        let mut g = gpu::exec::draw::TINY_SCISSOR.lock();
                        let on = if let Some(i) = g.iter().position(|&x| x == a) { g.remove(i); false } else { g.push(a); true };
                        gpu::exec::draw::TINY_ANY.store(!g.is_empty(), std::sync::atomic::Ordering::Relaxed);
                        info!("control: draws with ps {a:#x} {}", if on { "limited to one pixel" } else { "full again" });
                        Ok(())
                    }
                    None => Err("usage: tiny <hex ps address>".into()),
                }
            }
            "texcheck" => {
                // Diagnosis: `texcheck` lists the textures whose guest bytes
                // changed since their last upload with no write marked.
                for l in gpu::images::Images::check_sources() {
                    info!("{l}");
                }
                Ok(())
            }
            "reupload" => {
                // Diagnosis: `reupload [<hex address>]` uploads the textures
                // sourced from guest memory (at that address) again at their
                // next use.
                let base = parts.next().and_then(|a| u64::from_str_radix(a.trim_start_matches("0x"), 16).ok());
                let n = gpu::images::with(|im| im.force_reupload(base));
                info!("control: {n} texture(s) marked for upload");
                Ok(())
            }
            "clearimg" => {
                // Diagnosis: `clearimg <hex address>` zero-fills the cached
                // images at that guest address.
                match parts.next().and_then(|a| u64::from_str_radix(a.trim_start_matches("0x"), 16).ok()) {
                    Some(a) => {
                        let n = gpu::images::with(|im| im.clear_at(a));
                        info!("control: cleared {n} image(s) at {a:#x}");
                        Ok(())
                    }
                    None => Err("usage: clearimg <hex address>".into()),
                }
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

/// Control lines asked for by the title's developer automation (a guest
/// print of `dsport-control: <line>`, `hle::kernel::fs`), run before the
/// control file's next lines (tools/devscript.sh scripts pressing buttons at
/// their own points, e.g. at a boss's fog door).
pub static INJECTED: parking_lot::Mutex<Vec<String>> = parking_lot::Mutex::new(Vec::new());

/// `fscan` state: page hashes, candidate pages, their copies, candidate
/// float addresses.
#[derive(Default)]
struct FScan {
    pages: Vec<(u64, u64)>,
    cand: Option<Vec<u64>>,
    copy: std::collections::BTreeMap<u64, Vec<u8>>,
    floats: Option<Vec<u64>>,
}

static FSCAN: parking_lot::Mutex<Option<FScan>> = parking_lot::Mutex::new(None);
/// Candidate pages copied for `fscan floats` (64 MiB) at most.
const FSCAN_COPY_PAGES: usize = 16384;

fn page_hash(a: u64) -> u64 {
    let p = a as *const u64;
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for i in 0..512 {
        h = (h ^ unsafe { p.add(i).read_volatile() }).wrapping_mul(0x100_0000_01b3);
    }
    h
}

/// Debug (finding the gameplay camera's frame): 4x4 f32 matrices in the
/// title's flexible memory whose upper 3x3 is a rotation about the vertical
/// axis only (rows [c 0 s] [0 1 0] [-s 0 c], 16 bytes apart, c*c + s*s = 1,
/// not the identity); logs each with c, s and its fourth row (translation),
/// at most 64.
fn fmat() -> Result<(), String> {
    let t0 = Instant::now();
    let spans: Vec<(u64, u64)> = mem::MEM.lock().vmas.values().filter(|m| m.kind == mem::VmaKind::Flexible && m.prot & 3 == 3).map(|m| (m.start, m.end())).collect();
    let rd = |a: u64| f32::from_bits(unsafe { (a as *const u32).read_volatile() });
    let near = |a: f32, b: f32| (a - b).abs() < 1e-4;
    let mut found: Vec<String> = Vec::new();
    let mut n = 0usize;
    for (lo, hi) in spans {
        let mut a = lo.next_multiple_of(16);
        while a + 64 <= hi {
            let (c, s) = (rd(a), rd(a + 8));
            if c.is_finite() && s.is_finite() && c.abs() <= 1.0 && s.abs() > 1e-3 && near(c * c + s * s, 1.0) && near(rd(a + 0x28), c) && near(rd(a + 0x20), -s) && near(rd(a + 0x14), 1.0) && near(rd(a + 4), 0.0) && near(rd(a + 0x10), 0.0) && near(rd(a + 0x18), 0.0) && near(rd(a + 0x24), 0.0) {
                n += 1;
                if found.len() < 64 {
                    found.push(format!("{a:#x} c {c:.5} s {s:.5} t ({:.3}, {:.3}, {:.3})", rd(a + 0x30), rd(a + 0x34), rd(a + 0x38)));
                }
            }
            a += 16;
        }
    }
    info!("fmat: {n} yaw matrices ({} ms): {}", t0.elapsed().as_millis(), found.join("; "));
    Ok(())
}

fn fscan(cmd: &str, arg: &str) -> Result<(), String> {
    let t0 = Instant::now();
    let mut g = FSCAN.lock();
    if cmd == "start" {
        let spans: Vec<(u64, u64)> = mem::MEM.lock().vmas.values().filter(|m| m.kind == mem::VmaKind::Flexible && m.prot & 3 == 3).map(|m| (m.start, m.end())).collect();
        let mut st = FScan::default();
        for (lo, hi) in spans {
            let mut a = lo;
            while a + 4096 <= hi {
                st.pages.push((a, page_hash(a)));
                a += 4096;
            }
        }
        info!("fscan start: {} pages ({} ms)", st.pages.len(), t0.elapsed().as_millis());
        *g = Some(st);
        return Ok(());
    }
    let st = g.as_mut().ok_or("fscan start first")?;
    match cmd {
        "changed" | "same" => {
            let mut changed = std::collections::HashSet::new();
            for (a, h) in st.pages.iter_mut() {
                let n = page_hash(*a);
                if n != *h {
                    changed.insert(*a);
                    *h = n;
                }
            }
            let prev = st.cand.take().unwrap_or_else(|| st.pages.iter().map(|p| p.0).collect());
            let next: Vec<u64> = prev.into_iter().filter(|a| changed.contains(a) == (cmd == "changed")).collect();
            if next.len() <= FSCAN_COPY_PAGES {
                st.copy = next.iter().map(|&a| (a, unsafe { std::slice::from_raw_parts(a as *const u8, 4096) }.to_vec())).collect();
            }
            let listed = if next.len() <= 16 { format!(": {}", next.iter().map(|a| format!("{a:#x}")).collect::<Vec<_>>().join(" ")) } else { String::new() };
            info!("fscan {cmd}: {} pages changed, {} candidate pages{} ({} ms){listed}", changed.len(), next.len(), if next.len() <= FSCAN_COPY_PAGES { " (copied)" } else { "" }, t0.elapsed().as_millis());
            st.cand = Some(next);
            Ok(())
        }
        "floats" => {
            if st.copy.is_empty() {
                return Err("no copied pages (fscan same first)".into());
            }
            let mut found: Vec<(u64, f32, f32)> = Vec::new();
            for (&a, old) in st.copy.iter_mut() {
                let now = unsafe { std::slice::from_raw_parts(a as *const u8, 4096) };
                for i in (0..4096).step_by(4) {
                    let (o, n) = (f32::from_le_bytes(old[i..i + 4].try_into().unwrap()), f32::from_le_bytes(now[i..i + 4].try_into().unwrap()));
                    let addr = a + i as u64;
                    if let Some(f) = &st.floats {
                        if f.binary_search(&addr).is_err() {
                            continue;
                        }
                    }
                    // Not plausible values: non-finite, huge, or tiny (integers
                    // and pointers read as floats).
                    let odd = |v: f32| !v.is_finite() || v.abs() > 1e5 || (v != 0.0 && v.abs() < 1e-6);
                    if odd(o) || odd(n) {
                        continue;
                    }
                    let keep = match arg {
                        "+" => n > o,
                        "-" => n < o,
                        "=" => n == o,
                        _ => n != o,
                    };
                    if keep {
                        found.push((addr, o, n));
                    }
                }
                old.copy_from_slice(now);
            }
            let list: Vec<String> = found.iter().take(48).map(|(a, o, n)| format!("{a:#x} {o} -> {n}")).collect();
            info!("fscan floats {arg}: {} candidates ({} ms): {}", found.len(), t0.elapsed().as_millis(), list.join(", "));
            st.floats = Some(found.iter().map(|f| f.0).collect());
            Ok(())
        }
        "reset" => {
            st.floats = None;
            info!("fscan reset: float candidates cleared");
            Ok(())
        }
        _ => Err("usage: fscan start|changed|same|floats [+|-|=|*]|reset".into()),
    }
}

pub fn fatal_report(text: &str) {
    crate::log::write(crate::log::Level::Error, format_args!("{text}"));
    if let Some(dir) = RUN_DIR.get() {
        let _ = std::fs::write(dir.join("crash.txt"), crate::log::redact(text).as_bytes());
    }
    keep_run("crashed (crash.txt)");
}

fn thread_summary() -> String {
    thread_summary_ex(false)
}

/// Guest threads and their states; with `with_stacks`, each running thread is
/// sampled (rip, frame chain, guest return addresses) for hang diagnosis.
/// The stacks of this process's own threads by name (the GPU front end's, front end 2's, the
/// host's event loop's), for the stall watch (`DS_STALL_WATCH`): the queue threads' workers,
/// the completion and release threads are in no other dump.
fn host_thread_stacks() -> String {
    use windows_sys::Win32::System::Diagnostics::ToolHelp::{CreateToolhelp32Snapshot, Thread32First, Thread32Next, TH32CS_SNAPTHREAD, THREADENTRY32};
    let pid = unsafe { windows_sys::Win32::System::Threading::GetCurrentProcessId() };
    let snap = unsafe { CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0) };
    if snap == windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE {
        return String::new();
    }
    let mut tids: Vec<u32> = Vec::new();
    let mut te: THREADENTRY32 = unsafe { std::mem::zeroed() };
    te.dwSize = std::mem::size_of::<THREADENTRY32>() as u32;
    let mut ok = unsafe { Thread32First(snap, &mut te) } != 0;
    while ok {
        if te.th32OwnerProcessID == pid {
            tids.push(te.th32ThreadID);
        }
        ok = unsafe { Thread32Next(snap, &mut te) } != 0;
    }
    unsafe { windows_sys::Win32::Foundation::CloseHandle(snap) };
    let mut named: Vec<(String, u32)> = tids.into_iter().map(|t| (guestprof::thread_description(t), t)).filter(|(n, _)| (n.starts_with("gpu") || n.starts_with("fe2") || n.starts_with("host") || n.starts_with("forward")) && !n.starts_with("gpu-precompile")).collect();
    named.sort();
    let mut s = String::new();
    for (name, tid) in named {
        if let Some(st) = exception::sample_thread(tid, 0) {
            s.push_str(&format!("    '{name}' ({tid}): {st}\n"));
        }
    }
    s
}

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
    s.push_str(&displayclock::status_line());
    s.push_str(&gpu::vk::present::stats_line());
    s.push_str(&audio::status());
    s.push_str(&hle::libs::audio2::status());
    s.push_str(&hle::libs::ajm::status());
    s.push_str(&hle::iostat::status());
    s
}

/// Exit codes the launcher knows (`crates/launcher`): the game files cannot be used (told at
/// start), the processor lacks what the game's code needs, the GPU device was lost.
pub const EXIT_GAME_FILES: u32 = 3;
pub const EXIT_CPU: u32 = 8;
pub const EXIT_DEVICE_LOST: u32 = 5;
/// The game's modules could not be loaded, linked or patched (told by `fatal_exit`).
pub const EXIT_START: u32 = 0xD502;
/// Another session runs with the same data folder (told at start).
pub const EXIT_RUNNING: u32 = 9;
/// The process could not be set up before its log (told at start).
pub const EXIT_EARLY: u32 = 10;
/// Windows ran out of memory for the game (told by `fatal_exit`).
pub const EXIT_MEMORY: u32 = 7;
/// No usable Vulkan device at start (told by `fatal_exit`): a lost GPU that
/// needs Windows restarted, a missing or old driver.
pub const EXIT_NO_GPU: u32 = 11;

/// Whether this run has no window (`--headless`): no message boxes then.
pub static HEADLESS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Tells the player about a failure in a message box (the log alone reaches no one), unless
/// this is a development run (`DS_MESSAGES=1` shows them there too) or a headless one.
pub fn player_error(text: &str) {
    let dev = paths().layout == dsconfig::paths::Layout::Dev;
    let show = std::env::var("DS_MESSAGES").map(|v| v == "1").unwrap_or(!dev);
    if !show || HEADLESS.load(std::sync::atomic::Ordering::Relaxed) {
        return;
    }
    use windows_sys::Win32::UI::WindowsAndMessaging::{MB_ICONERROR, MB_OK, MB_SETFOREGROUND, MB_TOPMOST, MessageBoxW};
    let w = |s: &str| s.encode_utf16().chain(std::iter::once(0)).collect::<Vec<u16>>();
    let (t, m) = (w("Regaliant"), w(text));
    unsafe { MessageBoxW(std::ptr::null_mut(), m.as_ptr(), t.as_ptr(), MB_OK | MB_ICONERROR | MB_SETFOREGROUND | MB_TOPMOST) };
}

/// What the player is told when the game has to end (`fatal_exit`), with its log folder.
fn fatal_text(code: u32, reason: &str) -> String {
    let log = RUN_DIR.get().map(|d| format!("\n\nThe log is in:\n{}", d.display())).unwrap_or_default();
    match code {
        EXIT_DEVICE_LOST => format!("The graphics card stopped responding, so the game has to close.\n\nUpdating the graphics driver may help. If it happens again, please report it and attach the log.{log}"),
        EXIT_MEMORY => format!("Windows ran out of memory for the game, so it has to close.\n\nClose other programs and start the game again. If it happens again, let Windows manage the page file size (Settings > System > About > Advanced system settings > Performance > Advanced > Virtual memory).{log}"),
        EXIT_START => format!("The game could not start ({reason}).\n\nPlease report it and attach the log.{log}"),
        EXIT_NO_GPU => format!("The game could not start the graphics card ({reason}).\n\nIf Windows reported a graphics driver problem, restart Windows. Otherwise update the graphics driver: the game needs a Vulkan 1.3 graphics card.{log}"),
        _ => format!("The game met an error it cannot recover from and has to close.\n\nPlease report it and attach the log.{log}"),
    }
}

/// One game session at a time for a data folder (two would share saves, settings and the
/// shader store): a second start is told and ends. Modes that run no game are not counted.
fn single_instance(cfg: &Config) {
    // (The launcher's translation check runs no game either and ends before any part of a
    // session's start-up: holding the lock, it turned away a Play pressed while it ran,
    // 2026-11-14.)
    if cfg.precompile_all.is_some() || cfg.compile_bench.is_some() || cfg.translation_fingerprint {
        return;
    }
    use windows_sys::Win32::Foundation::{ERROR_ALREADY_EXISTS, GetLastError};
    use windows_sys::Win32::System::Threading::CreateMutexW;
    // (Named by the data folder: a checkout and a player's install do not exclude each other.)
    let key: String = cfg.paths.user.display().to_string().chars().map(|c| if c.is_ascii_alphanumeric() { c.to_ascii_lowercase() } else { '_' }).collect();
    let name: Vec<u16> = format!("Local\\Regaliant_{key}").encode_utf16().chain(std::iter::once(0)).collect();
    // (Kept for the process's life: never closed.)
    let h = unsafe { CreateMutexW(std::ptr::null(), 0, name.as_ptr()) };
    if !h.is_null() && unsafe { GetLastError() } == ERROR_ALREADY_EXISTS {
        error!("another game session runs with the same data folder: this start ends");
        player_error("The game is already running.");
        log::flush();
        std::process::exit(EXIT_RUNNING as i32);
    }
}

/// Only the first thread that ends the game tells the player (others wait for the end).
static ENDING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// No usable Vulkan device at start (`gpu::exec::enabled`): the player is told and the game
/// ends with EXIT_NO_GPU. Not through `fatal_exit`: its thread summary reads the GPU front
/// end's status, which needs the Vulkan context that is missing (that exit panicked inside
/// itself and the process hung, 2026-11-11, with the GPU lost until a restart).
pub fn no_gpu_exit(reason: &str) -> ! {
    if ENDING.swap(true, std::sync::atomic::Ordering::SeqCst) {
        loop {
            std::thread::sleep(Duration::from_secs(1));
        }
    }
    display::restore_on_exit();
    session_ended();
    crate::log::write(crate::log::Level::Error, format_args!("terminating: {reason}"));
    crate::log::try_flush();
    player_error(&fatal_text(EXIT_NO_GPU, reason));
    unsafe {
        windows_sys::Win32::System::Threading::TerminateProcess(windows_sys::Win32::System::Threading::GetCurrentProcess(), EXIT_NO_GPU);
    }
    loop {
        std::thread::sleep(Duration::from_secs(1));
    }
}

pub fn fatal_exit(code: u32, reason: &str) -> ! {
    if ENDING.swap(true, std::sync::atomic::Ordering::SeqCst) {
        loop {
            std::thread::sleep(Duration::from_secs(1));
        }
    }
    // Exclusive fullscreen's display mode back first (the desktop must not stay changed).
    display::restore_on_exit();
    session_ended();
    // After a fault, every thread's frames (what the others were doing:
    // the title's teardown when a job read freed memory).
    let summary = if reason == "fatal exception" { thread_summary_ex(true) } else { thread_summary() };
    crate::log::write(crate::log::Level::Error, format_args!("terminating: {reason}\n{summary}"));
    crate::log::try_flush();
    // The player is told (the game's window out of the way of the message).
    if let Some(w) = gpu::vk::window::get() {
        unsafe { windows_sys::Win32::UI::WindowsAndMessaging::ShowWindow(w.hwnd as _, windows_sys::Win32::UI::WindowsAndMessaging::SW_MINIMIZE) };
    }
    player_error(&fatal_text(code, reason));
    unsafe {
        windows_sys::Win32::System::Threading::TerminateProcess(windows_sys::Win32::System::Threading::GetCurrentProcess(), code);
    }
    loop {
        std::thread::sleep(Duration::from_secs(1));
    }
}

/// A profile-instrumented build (`-Cprofile-generate` with `--cfg
/// dsport_pgo_gen`) writes its counters here: TerminateProcess skips the
/// profile runtime's exit hook.
fn pgo_write() {
    #[cfg(dsport_pgo_gen)]
    unsafe {
        unsafe extern "C" {
            fn __llvm_profile_write_file() -> i32;
        }
        let r = __llvm_profile_write_file();
        info!("profile counters written (status {r})");
        crate::log::flush();
    }
}

/// The session ended in an orderly way (`shutdown`, `fatal_exit`): its marker goes.
fn session_ended() {
    let m = paths().running_marker();
    if std::fs::read_to_string(&m).is_ok_and(|t| t.starts_with(&format!("pid {}\n", std::process::id()))) {
        let _ = std::fs::remove_file(m);
    }
}

pub fn shutdown(code: i32) -> ! {
    display::restore_on_exit();
    session_ended();
    hle::libs::videoout::wait_shots();
    gpu::shutdown();
    info!("shutdown with status {code}\n{}", thread_summary());
    crate::log::flush();
    pgo_write();
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
    // Under front end 2 (the default) and with `--host` this process only
    // hosts dsport (its debugger; `hosted::wants_host`).
    if hosted::wants_host() {
        hosted::run_host();
    }
    // The guest address range must be reserved before the process grows
    // (the first heap allocation reserved it already, see `heap`).
    // (Before the log exists: the player is told, and the launcher shows the code.)
    if let Err(e) = mem::reserve_guest_range() {
        eprintln!("{e}");
        player_error(&format!("The game could not reserve the memory range its program needs ({e}).

This happens when other software takes that range. Restart the PC and try again."));
        std::process::exit(EXIT_EARLY as i32);
    }
    let slot = match tls::init_slot() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("{e}");
            player_error(&format!("The game could not set up its threads ({e})."));
            std::process::exit(EXIT_EARLY as i32);
        }
    };
    let cfg = parse_args();
    PATHS.set(cfg.paths.clone()).ok();
    HEADLESS.store(cfg.headless, std::sync::atomic::Ordering::Relaxed);
    prune_runs(&cfg.paths.logs(), cfg.paths.logs_kept());
    if let Err(e) = std::fs::create_dir_all(&cfg.run_dir).and_then(|_| log::init(&cfg.run_dir.join("dsport.log"))) {
        player_error(&format!("The game could not write its log in {}.\n\n{e}", cfg.run_dir.display()));
        std::process::exit(1);
    }
    RUN_DIR.set(cfg.run_dir.clone()).ok();
    // A session in progress (the launcher's: one left behind did not end normally). Runs of no
    // game write none (the launcher's check left its marker behind, then read as a session that
    // did not end normally, or replaced a running game's).
    if cfg.precompile_all.is_none() && cfg.compile_bench.is_none() && !cfg.translation_fingerprint {
        let _ = std::fs::write(cfg.paths.running_marker(), format!("pid {}\nlog {}\n", std::process::id(), cfg.run_dir.display()));
    }
    log::set_thread_name("host");
    // Panics (host threads abort the process) are logged with the thread
    // that panicked: the run log is what post-mortems read.
    // A panic ends the game as a crash does (the player told, the display mode back).
    std::panic::set_hook(Box::new(move |info| {
        crate::error!("panic: {info}");
        crate::fatal_exit(0xD540, "panic");
    }));
    if cfg.debug || cfg.trace {
        log::LEVEL.store(log::Level::Debug as u8, std::sync::atomic::Ordering::Relaxed);
    }
    log::TRACE_HLE.store(cfg.trace, std::sync::atomic::Ordering::Relaxed);
    hle::LENIENT_IMPORTS.store(cfg.lenient, std::sync::atomic::Ordering::Relaxed);
    info!("Regaliant {} | run dir {} | TLS slot {} | host heap: {} | guest write watch: {}", env!("CARGO_PKG_VERSION"), cfg.run_dir.display(), slot, heap::name(), mem::write_watch_state());
    info!("files: {:?} layout, settings and saves in {}, logs and caches in {}", cfg.paths.layout, cfg.paths.user.display(), cfg.paths.cache.display());
    if let Some(p) = &cfg.game_problem {
        error!("game files: {}", p.replace('\n', " "));
        player_error(p);
        log::flush();
        std::process::exit(EXIT_GAME_FILES as i32);
    }
    cpucheck::check();
    single_instance(&cfg);
    exception::install();
    heapsites::start();
    hosted::init();
    hosted::self_test_maybe();
    if cfg.dispatch_bench {
        let code = gpu::dispbench::run();
        log::flush();
        std::process::exit(code);
    }
    if let Some(c) = &cfg.pipeline_test {
        let code = gpu::exec::pipeline_test(c);
        log::flush();
        std::process::exit(code);
    }
    if let Some((c, cs)) = &cfg.record_bench {
        let code = gpu::exec::record_bench(c, *cs);
        log::flush();
        std::process::exit(code);
    }
    // The launcher's translation check: the translator's answer for a sample of the game's
    // compute programs (it depends on the device and the translator only), before any part of a
    // session: no window, settings, saves or stores. It takes no one-session lock and may run
    // beside a game session (it ran their start-up until 2026-11-14: a pending Save Folder move
    // or PS5 import applied under a running game, the stores' unfinished files deleted).
    if cfg.translation_fingerprint {
        // (Translations made here are for the game's hosted sessions: their specs as those make
        // them, `fe2::specs_as_hosted`.)
        gpu::fe2::specs_as_hosted();
        let code = match gpu::compbench::translation_fingerprint(&cfg.app) {
            Ok(fp) => {
                println!("translation = {fp}");
                info!("translation fingerprint: {fp}");
                0
            }
            Err(e) => {
                error!("translation fingerprint: {e}");
                2
            }
        };
        log::flush();
        std::process::exit(code);
    }
    gpu::particles::start(cfg.app.clone());
    gpu::capture::init_from_env();
    gpu::start_profiler();
    guestprof::start();
    hostload::start();
    if std::env::var_os("DS_TEST_FAULT").is_some() {
        info!("DS_TEST_FAULT: writing to an invalid address from host code");
        unsafe { std::ptr::write_volatile(0x32c3be8 as *mut u64, 1) };
    }
    thread::apply_power_optout(gpu::tune::power_optout());
    hle::kernel::time::init();
    // (Its samples are timed by the process clock.)
    guestprof::start_trace();
    if !cfg.headless {
        if let Err(e) = gpu::vk::present::init("Regaliant - Demon's Souls") {
            warn!("no presentation window: {e}");
        }
    }

    // Guest file system. The supplied game tree is mounted read-only; all
    // writable state lives in the port's own folders (`dsconfig::paths`).
    let rt = cfg.paths.runtime();
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
    // Keyboard and mouse layout (written with the defaults when missing).
    input::init(&cfg.paths.controls_file());
    // `DS_ACHIEVEMENTS_FILE`: another file (test runs: theirs, apart from a player's).
    let achievements_file = std::env::var_os("DS_ACHIEVEMENTS_FILE").map(PathBuf::from).unwrap_or_else(|| cfg.paths.achievements_file());
    achievements::init(&cfg.app, achievements_file, "en-GB");
    // Monitors and the PC tab's display choices (before its menu lists are made).
    // Safe mode: the display choices and PC settings at their defaults for this session (a
    // scratch file for the display's, the player's files left as they are).
    if cfg.safe_mode {
        info!("safe mode (the launcher's, for this start): default settings, a window, no upscaler, HDR off, arming and held pages off");
        let scratch = cfg.paths.cache.join("safe-mode-display.txt");
        let _ = std::fs::remove_file(&scratch);
        display::init(scratch);
    } else {
        display::init(cfg.paths.display_file());
    }
    pcsettings::init(cfg.paths.settings_file(), cfg.safe_mode);
    // The saves: the PC tab's Save Folder (the port's own by default), a PS5 import asked for
    // (a game session's: the shader preparation leaves them as they are).
    if cfg.precompile_all.is_none() && cfg.compile_bench.is_none() {
        hle::libs::savedata::init(pcsettings::saves::start(&cfg.paths, &hle::libs::savedata::title_id()));
    }
    // Compiled pipelines and learned shader specs: local-data, or the
    // `--cache` directory (Play.bat: one next to the package folder, so a
    // new package starts with what earlier ones learned on this machine).
    let cache_root = cfg.cache.clone().unwrap_or_else(|| cfg.paths.shaders());
    if cfg.cache.is_some() {
        hle::kernel::fs::ensure_dir(&cache_root);
        info!("pipeline and shader-spec cache: {}", cache_root.display());
    }
    // DS_PIPELINE_DIR=<directory name under the cache>: another pipeline
    // store (experiments that start without compiled pipelines).
    let pipe_dir = std::env::var("DS_PIPELINE_DIR").ok().filter(|d| !d.is_empty() && d.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_'));
    // A translator switch makes other SPIR-V under the same pipeline keys:
    // no pipeline store then unless one is named for the experiment.
    match (pipe_dir, gpu::trcache::translator_switch()) {
        (Some(d), _) => gpu::pcache::set_dir(cache_root.join(d)),
        (None, Some(v)) => info!("pipeline store: off while {v} is set (DS_PIPELINE_DIR=<name> keeps one)"),
        (None, None) => gpu::pcache::set_dir(cache_root.join("pipelines")),
    }
    // Translations kept across runs (`gpu::trcache`).
    gpu::trcache::set_dir(cache_root.join("translations"));
    // DS_SPECS=<file name under the cache, or a path>: where the shader
    // programs' learned specs and code are kept (`gpu::precompile`).
    let specs = std::env::var("DS_SPECS").ok().filter(|v| !v.is_empty()).map(std::path::PathBuf::from);
    gpu::precompile::set_store(match specs {
        Some(p) if p.components().count() > 1 => p,
        Some(p) => cache_root.join(p),
        None => cache_root.join("shader-specs.txt"),
    });
    // (Translations made here are for the game's hosted sessions: their specs as those make
    // them, `fe2::specs_as_hosted`.)
    if cfg.compile_bench.is_some() || cfg.precompile_all.is_some() {
        gpu::fe2::specs_as_hosted();
    }
    if let Some(a) = &cfg.compile_bench {
        let code = gpu::compbench::run(a, &cfg.app);
        log::flush();
        std::process::exit(code);
    }
    if let Some(n) = cfg.precompile_all {
        let code = gpu::compbench::precompile_all(&cfg.app, n, cfg.progress.as_deref(), Some(&cache_root.join("prepared.txt")));
        log::flush();
        std::process::exit(code);
    }
    // A player's install executes the GPU work unless DS_GPU says otherwise (a development run
    // sets DS_GPU=1 itself; the launcher's preparation and fingerprint runs above are as before).
    if cfg.paths.layout != dsconfig::paths::Layout::Dev {
        gpu::exec::default_on();
    }

    // Modules: the decrypted executable and the title's own libraries.
    let t0 = Instant::now();
    let eboot = loader::load(&cfg.code.join("eboot.bin"), true).unwrap_or_else(|e| {
        error!("loading eboot failed: {e}");
        fatal_exit(EXIT_START, "load failure");
    });
    let mut libs: Vec<Arc<loader::LoadedModule>> = Vec::new();
    for name in ["libc.prx", "libSceNpCppWebApi.prx"] {
        let p = cfg.code.join("sce_module").join(name);
        if !p.exists() {
            error!("{name} not found in {}", p.display());
            fatal_exit(EXIT_START, "missing title module");
        }
        match loader::load(&p, false) {
            Ok(m) => libs.push(m),
            Err(e) => {
                error!("loading {name} failed: {e}");
                fatal_exit(EXIT_START, "load failure");
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
                fatal_exit(EXIT_START, "link failure");
            }
        }
    }
    for m in &modules {
        match loader::finalize(m) {
            Ok(s) => loader::report_patch(m, &s),
            Err(e) => {
                error!("finalizing {} failed: {e}", m.name);
                fatal_exit(EXIT_START, "patch failure");
            }
        }
    }
    let unresolved = hle::UNRESOLVED.read().len();
    {
        // Planning aid: every unresolved function import, by library.
        let mut lines: Vec<String> = hle::UNRESOLVED.read().iter().map(|u| format!("{}\t{}\t{}", u.lib, u.nid, hle::describe_nid(&u.nid))).collect();
        lines.sort();
        let _ = std::fs::write(cfg.run_dir.join("unresolved.txt"), lines.join("\n") + "\n");
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
    // Boot watch (`DS_BOOT_WATCH=<s>`, default 15, 0 off): a boot that has
    // flipped no frame by then dumps the guest thread stacks once (some
    // boots stalled ~40 s early in the title's start-up, FINDINGS.md
    // 2026-10-18 night; a normal boot flips within ~5 s).
    let mut boot_watch = std::env::var("DS_BOOT_WATCH").ok().and_then(|v| v.parse::<u64>().ok()).unwrap_or(15);
    // Stall watch (`DS_STALL_WATCH=<ms>`, diagnosis; 0, the default: off): once frames flip, a
    // gap of that long without a flip logs every thread's stack and wait record while it lasts
    // (at most `DS_STALL_WATCH_MAX` times a run, default 4; once a gap).
    let stall_watch = std::env::var("DS_STALL_WATCH").ok().and_then(|v| v.parse::<u64>().ok()).unwrap_or(0);
    let mut stall_left = std::env::var("DS_STALL_WATCH_MAX").ok().and_then(|v| v.parse::<u32>().ok()).unwrap_or(4);
    let mut stall_at = (0u64, Instant::now(), false);
    loop {
        std::thread::sleep(Duration::from_millis(250));
        ticks += 1;
        if stall_watch > 0 && stall_left > 0 {
            let f = gpu::FRAMES.load(std::sync::atomic::Ordering::Relaxed);
            if f != stall_at.0 {
                stall_at = (f, Instant::now(), false);
            } else if f > 30 && gpu::vk::try_ctx().is_some() && !stall_at.2 && stall_at.1.elapsed() >= Duration::from_millis(stall_watch) {
                stall_at.2 = true;
                stall_left -= 1;
                warn!("stall watch: no flip for {} ms after flip {f} (DS_STALL_WATCH); our threads:\n{}the title's threads and the queues:\n{}", stall_at.1.elapsed().as_millis(), host_thread_stacks(), thread_summary_ex(true));
            }
        }
        if boot_watch > 0 && start.elapsed() >= Duration::from_secs(boot_watch) {
            if gpu::FRAMES.load(std::sync::atomic::Ordering::Relaxed) == 0 && gpu::exec::enabled() {
                warn!("boot watch: no frame flipped {boot_watch} s after start (DS_BOOT_WATCH); guest thread stacks:\n{}", thread_summary_ex(true));
            }
            boot_watch = 0;
        }
        if ticks % 8 == 0 && exception::break_stats_enabled() {
            exception::dump_break_stats();
        }
        if ticks % 4 == 0 {
            titlemem::tick(start.elapsed().as_secs());
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
