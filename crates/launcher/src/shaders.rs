//! The Shaders page: the game's shaders prepared ahead (`dsport --precompile-all`, every compute
//! program of the game's shader files compiled into the player's own pipeline store), with its
//! progress, a stop that lets the current shaders finish, and the option to prepare them when
//! Play is pressed the first time. Prepared shaders hold for one graphics card and driver
//! (prepared.txt records them) and for the game's translation of them: a new driver, or a
//! version of the port that translates the shaders differently (`dsport
//! --translation-fingerprint`, checked once at start on a thread of its own), asks for another
//! preparation.

use crate::sys;
use crate::ui::{self, Builder};
use crate::App;
use dsconfig::paths::read_kv;
use std::path::PathBuf;
use std::process::Child;
use std::time::Instant;
use windows_sys::Win32::Foundation::HWND;

const ID_PREPARE: i32 = 201;
const ID_STOP: i32 = 202;
const ID_AUTO: i32 = 203;
/// The timer that follows a preparation.
pub const TIMER: usize = 3;
/// The timer that waits for the translation check (`start_check`).
pub const CHECK_TIMER: usize = 4;

/// This build's translation fingerprint (`dsport --translation-fingerprint`): None while the
/// check runs, then its answer.
static CHECK: std::sync::Mutex<Option<Result<String, String>>> = std::sync::Mutex::new(None);
static CHECK_STARTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Asks the game for its translation fingerprint on a thread of its own (a few seconds).
pub fn start_check(app: &App) {
    if CHECK_STARTED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        return;
    }
    let exe = dsconfig::paths::game_exe_in(&app.paths.exe_dir);
    let dir = app.paths.exe_dir.clone();
    std::thread::spawn(move || {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        const BELOW_NORMAL_PRIORITY_CLASS: u32 = 0x0000_4000;
        let r = std::process::Command::new(&exe)
            .arg("--translation-fingerprint")
            .current_dir(&dir)
            .creation_flags(CREATE_NO_WINDOW | BELOW_NORMAL_PRIORITY_CLASS)
            .stdin(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .output()
            .map_err(|e| e.to_string())
            .and_then(|o| {
                String::from_utf8_lossy(&o.stdout).lines().find_map(|l| l.trim().strip_prefix("translation = ").map(|v| v.trim().to_string())).filter(|v| !v.is_empty()).ok_or_else(|| format!("no answer (exit code {:?})", o.status.code()))
            });
        *CHECK.lock().unwrap_or_else(|e| e.into_inner()) = Some(r);
    });
}

/// The check's answer once it has one.
fn checked_translation() -> Option<Result<String, String>> {
    CHECK.lock().unwrap_or_else(|e| e.into_inner()).clone()
}

/// The check timer: the page shows the check's outcome once it has one.
pub fn check_tick(app: &mut App) {
    if checked_translation().is_some() {
        unsafe { windows_sys::Win32::UI::WindowsAndMessaging::KillTimer(app.hwnd, CHECK_TIMER) };
        refresh(app);
        app.update_play();
    }
}

#[derive(Default)]
pub struct Page {
    status: usize,
    bar: usize,
    prepare: usize,
    stop: usize,
    auto: usize,
    child: Option<Child>,
    started: Option<Instant>,
    /// The game starts when the preparation is done (it began from Play).
    pub play_after: bool,
    /// A stop was asked for (the child finishes its current shaders).
    stopping: Option<Instant>,
    /// The game starts once the stop is done.
    then_play: bool,
}

fn h(v: usize) -> HWND {
    v as HWND
}

fn progress_file(app: &App) -> PathBuf {
    app.paths.shaders().join("prepare-progress.txt")
}

fn marker(app: &App) -> PathBuf {
    app.paths.shaders().join("prepared.txt")
}

/// Whether the shaders are prepared for this graphics card and driver; the status line.
pub fn state(app: &App) -> (bool, String) {
    let kv = read_kv(&marker(app));
    if kv.is_empty() {
        return (false, "The shaders are not prepared on this PC yet.".into());
    }
    let get = |k: &str| kv.iter().find(|(x, _)| x == k).map(|(_, v)| v.clone()).unwrap_or_default();
    let date = get("date");
    let date = if date.len() >= 8 { format!("{}-{}-{}", &date[..4], &date[4..6], &date[6..8]) } else { date };
    // (A marker from before the translation fingerprint, or from a version of the port that
    // translates the shaders differently: its pipelines no longer match.)
    let updated = (false, "The game was updated since the shaders were prepared. Prepare them again.".to_string());
    let translation = get("translation");
    if translation.is_empty() {
        return updated;
    }
    if let Some(Ok(current)) = checked_translation() {
        if current != translation {
            return updated;
        }
    }
    match sys::gpu() {
        Ok(g) if g.name == get("device") && g.driver.to_string() == get("driver") => (true, format!("The shaders are prepared (on {date}, for this graphics card and driver).")),
        Ok(g) if g.name == get("device") => (false, "The graphics driver changed since the shaders were prepared. Prepare them again.".into()),
        Ok(_) => (false, "The graphics card changed since the shaders were prepared. Prepare them again.".into()),
        Err(e) => (false, e),
    }
}

pub fn prepared(app: &App) -> bool {
    // (Play waits for the translation check, a few seconds at most.)
    let t = Instant::now();
    while CHECK_STARTED.load(std::sync::atomic::Ordering::Relaxed) && checked_translation().is_none() && t.elapsed().as_secs() < 20 {
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    state(app).0
}

pub fn build(app: &mut App) {
    let b: Builder = app.builder();
    unsafe {
        let r = (16, 62, 748, 62);
        app.put(1, b.label("The game compiles each shader the first time it needs it, which makes play stutter. Preparing them here once takes about 20 to 60 minutes and makes the longest stutters much shorter. Some stutter remains on first visits. Nothing of it shows while you play.", r, -1), r);
        let r = (16, 134, 748, 40);
        app.shaders.status = app.put(1, b.label("", r, -1), r) as usize;
        let r = (16, 178, 748, 20);
        app.shaders.bar = app.put(1, b.progress(r, -1), r) as usize;
        let r = (16, 210, 140, 30);
        app.shaders.prepare = app.put(1, b.button("Prepare now", r, ID_PREPARE), r) as usize;
        let r = (164, 210, 100, 30);
        app.shaders.stop = app.put(1, b.button("Stop", r, ID_STOP), r) as usize;
        let r = (16, 256, 748, 40);
        app.shaders.auto = app.put(1, b.check("Prepare the shaders when Play is pressed and they are not prepared yet.", r, ID_AUTO), r) as usize;
        ui::set_check(h(app.shaders.auto), app.launch.prepare_shaders);
        let where_ = format!(
            "Prepared shaders are kept in {}. They take up to 4 GB. Preparing uses {} of this PC's {} processor threads.",
            ui::short(&app.paths.shaders(), 70),
            sys::prepare_threads(),
            sys::threads()
        );
        let r = (16, 306, 748, 60);
        app.put(1, b.label(&where_, r, -1), r);
    }
    refresh(app);
}

/// The page's state when nothing runs.
pub fn refresh(app: &mut App) {
    if app.shaders.child.is_some() {
        return;
    }
    let (_, text) = state(app);
    unsafe {
        ui::set_text(h(app.shaders.status), &text);
        ui::show(h(app.shaders.bar), false);
        ui::enable(h(app.shaders.prepare), true);
        ui::enable(h(app.shaders.stop), false);
    }
}

pub fn running(app: &App) -> bool {
    app.shaders.child.is_some()
}

/// Whether a stop was asked for (the preparation ends on its own).
pub fn stopping(app: &App) -> bool {
    app.shaders.stopping.is_some()
}

/// Starts the preparation (`play_after`: the game starts when it is done).
pub fn start(app: &mut App, play_after: bool) {
    if app.shaders.child.is_some() || app.running.is_some() {
        return;
    }
    let exe = dsconfig::paths::game_exe_in(&app.paths.exe_dir);
    if !exe.is_file() {
        unsafe { ui::message(app.hwnd, &format!("{} was not found next to the launcher.", dsconfig::paths::GAME_EXE), crate::TITLE, windows_sys::Win32::UI::WindowsAndMessaging::MB_ICONERROR) };
        return;
    }
    let dir = app.paths.shaders();
    if let Some(free) = sys::free_space(&dir) {
        if free < 3 << 30 {
            let t = format!("Only {} is free on the drive of {}. Preparing the shaders needs up to 4 GB. Prepare them anyway?", sys::gb(free), dir.display());
            if unsafe { ui::message(app.hwnd, &t, crate::TITLE, windows_sys::Win32::UI::WindowsAndMessaging::MB_ICONWARNING | windows_sys::Win32::UI::WindowsAndMessaging::MB_YESNO) } != windows_sys::Win32::UI::WindowsAndMessaging::IDYES {
                return;
            }
        }
    }
    let _ = std::fs::create_dir_all(&dir);
    let progress = progress_file(app);
    let _ = std::fs::remove_file(&progress);
    let _ = std::fs::remove_file(progress.with_extension("stop"));
    use std::os::windows::process::CommandExt;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    const BELOW_NORMAL_PRIORITY_CLASS: u32 = 0x0000_4000;
    let mut cmd = std::process::Command::new(&exe);
    cmd.arg("--precompile-all").arg(sys::prepare_threads().to_string()).arg("--progress").arg(&progress).current_dir(&app.paths.exe_dir).creation_flags(CREATE_NO_WINDOW | BELOW_NORMAL_PRIORITY_CLASS);
    match cmd.spawn() {
        Ok(c) => {
            app.shaders.child = Some(c);
            app.shaders.started = Some(Instant::now());
            app.shaders.play_after = play_after;
            app.shaders.stopping = None;
            app.shaders.then_play = false;
            unsafe {
                ui::set_text(h(app.shaders.status), "Preparing the shaders. Reading the game's shader files...");
                ui::set_progress(h(app.shaders.bar), 0);
                ui::show(h(app.shaders.bar), true);
                ui::enable(h(app.shaders.prepare), false);
                ui::enable(h(app.shaders.stop), true);
                windows_sys::Win32::UI::WindowsAndMessaging::SetTimer(app.hwnd, TIMER, 500, None);
            }
            app.update_play();
            app.set_status(if play_after { "Preparing the shaders first. The game starts when they are done." } else { "" });
        }
        Err(e) => app.set_status(&format!("Could not start the preparation: {e}")),
    }
}

/// Stops the preparation, then starts the game.
pub fn stop_then_play(app: &mut App) {
    stop(app);
    app.shaders.then_play = true;
}

/// Asks the preparation to stop (its current shaders finish first).
pub fn stop(app: &mut App) {
    if app.shaders.child.is_none() || app.shaders.stopping.is_some() {
        return;
    }
    app.shaders.play_after = false;
    app.shaders.stopping = Some(Instant::now());
    let _ = std::fs::write(progress_file(app).with_extension("stop"), "stop\n");
    unsafe {
        ui::set_text(h(app.shaders.status), "Stopping after the shaders being compiled now...");
        ui::enable(h(app.shaders.stop), false);
    }
}

/// The timer: progress, and the end of the preparation.
pub fn tick(app: &mut App) {
    let Some(child) = app.shaders.child.as_mut() else { return };
    let exited = match child.try_wait() {
        Ok(Some(s)) => Some(s.code().unwrap_or(-1)),
        Ok(None) => None,
        Err(_) => Some(-1),
    };
    // (A stop that takes too long: the child is ended.)
    if exited.is_none() && app.shaders.stopping.is_some_and(|t| t.elapsed().as_secs() > 90) {
        let _ = child.kill();
    }
    if let Some(code) = exited {
        app.shaders.child = None;
        unsafe { windows_sys::Win32::UI::WindowsAndMessaging::KillTimer(app.hwnd, TIMER) };
        let _ = std::fs::remove_file(progress_file(app).with_extension("stop"));
        let stopped = app.shaders.stopping.take().is_some();
        refresh(app);
        app.update_play();
        if stopped {
            unsafe { ui::set_text(h(app.shaders.status), "Preparation stopped. The next one goes on where it left off.") };
            if std::mem::take(&mut app.shaders.then_play) {
                crate::start_game(app);
            }
        } else if code != 0 || !prepared(app) {
            app.shaders.play_after = false;
            unsafe { ui::set_text(h(app.shaders.status), &format!("Preparation ended with an error (code {code}). Open logs shows its log.")) };
        } else if std::mem::take(&mut app.shaders.play_after) {
            crate::start_game(app);
        }
        return;
    }
    if app.shaders.stopping.is_some() {
        return;
    }
    // Progress: "<done> <shaders>" and "<code done> <code>".
    let Ok(t) = std::fs::read_to_string(progress_file(app)) else { return };
    let nums: Vec<u64> = t.split_whitespace().filter_map(|w| w.parse().ok()).collect();
    if let [done, total, wdone, wtotal] = nums[..] {
        let frac = if wtotal > 0 { wdone as f64 / wtotal as f64 } else { 0.0 };
        let el = app.shaders.started.map_or(0.0, |s| s.elapsed().as_secs_f64());
        let left = if frac > 0.02 { el * (1.0 - frac) / frac } else { 0.0 };
        let eta = if frac > 0.02 { format!(" About {} minutes left.", (left / 60.0).ceil().max(1.0) as u64) } else { String::new() };
        unsafe {
            ui::set_text(h(app.shaders.status), &format!("Preparing the shaders: {done} of {total} ({:.0}%).{eta}", frac * 100.0));
            ui::set_progress(h(app.shaders.bar), (frac * 1000.0) as u32);
        }
    }
}

pub fn command(app: &mut App, id: i32, code: u32) {
    match (id, code) {
        (ID_PREPARE, 0) => start(app, false),
        (ID_STOP, 0) => stop(app),
        (ID_AUTO, 0) => {
            app.launch.prepare_shaders = unsafe { ui::checked(h(app.shaders.auto)) };
            app.save_launch();
        }
        _ => {}
    }
}
