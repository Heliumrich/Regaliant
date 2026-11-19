//! The Shaders page: the game's shaders prepared ahead, in two steps: the game plays by itself in
//! the background to learn them (`dsport --prepare-play newgame`, then `tour`: a new game, then
//! every level with its walks, fights and boss; no window, no sound, the player's controllers,
//! keyboard and mouse left alone; its saves and settings its own), and beside the play, on the
//! CPU it leaves idle, every compute program of the game's shader files is compiled into the
//! player's own pipeline store (`--precompile-all` on its own when the new game cannot be
//! played). Its progress, a stop that lets the current step end, and the option to prepare them
//! when Play is pressed the first time. Prepared shaders hold for one graphics card and driver (prepared.txt records them,
//! played.txt the self-play's stages) and for the game's translation of them: a new driver, or a
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

/// The self-play's exit status when the stop file ended it (`prepplay::EXIT_STOPPED`).
const STAGE_STOPPED: i32 = 12;
/// The game's exit status when another session holds the player's data folder
/// (`main::EXIT_RUNNING`): the step did not run.
const GAME_RUNNING: i32 = 9;

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

/// The preparation's steps, in order (the compute step only when the new game failed).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Step {
    /// The compute programs compiled on their own (`--precompile-all`).
    Compute,
    /// The game plays a new game by itself (`--prepare-play newgame`).
    NewGame,
    /// The game visits the i-th of `LEVEL_PARTS` parts of its levels by itself (`--prepare-play
    /// tour --prepare-chunk i/n`, the save `NewGame` made): each part its own session, so that
    /// the memory a session's level changes take stays bounded.
    Levels(u8),
}

/// The levels' parts (sessions).
const LEVEL_PARTS: u8 = 3;

/// The whole preparation's time: the levels end at the next level after it (the latest levels
/// are left to the player's first visits), so that it stays within an hour on slower PCs.
const BUDGET: std::time::Duration = std::time::Duration::from_secs(55 * 60);

impl Step {
    /// The self-play stage (None: the compute step).
    fn stage(self) -> Option<&'static str> {
        match self {
            Step::Compute => None,
            Step::NewGame => Some("newgame"),
            Step::Levels(_) => Some("tour"),
        }
    }

    /// The session's time limit (a stage that hangs ends).
    fn seconds(self) -> u64 {
        match self {
            Step::Compute => 0,
            Step::NewGame => 2400,
            Step::Levels(_) => 3600,
        }
    }

    /// Its number among the self-play's steps (1 the new game).
    fn number(self) -> u8 {
        match self {
            Step::Levels(i) => i + 1,
            _ => 1,
        }
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
    /// The step the child runs.
    step: Option<Step>,
    /// When the step began.
    started: Option<Instant>,
    /// The game starts when the preparation is done (it began from Play).
    pub play_after: bool,
    /// A stop was asked for (the child finishes its current shaders).
    stopping: Option<Instant>,
    /// The game starts once the stop is done.
    then_play: bool,
    /// When the preparation's time is up (`BUDGET` from its start).
    deadline: Option<std::time::SystemTime>,
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

/// The self-play's stages and how they ended (`record`).
fn played_file(app: &App) -> PathBuf {
    app.paths.shaders().join("played.txt")
}

/// Whether the compute shaders are prepared for this graphics card, driver and translation; the
/// status line.
fn compute_state(app: &App) -> (bool, String) {
    // (They are prepared from the game's files: nothing to do before a folder is chosen.)
    if !app.game.ok {
        return (false, "Choose the game's folder on the Game page first.".into());
    }
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

/// How the self-play went (played.txt): the new game's end, the levels' parts played, the last
/// part's end (0: played to its end or to the preparation's time, 3: stopped, other: it could
/// not go on); for this game version's translation, this card and driver (else nothing).
#[derive(Default, Clone, Copy, PartialEq, Eq, Debug)]
struct Played {
    newgame: Option<i32>,
    parts: u8,
    tour: Option<i32>,
    /// The preparation's time ran out before the last part ended (the latest levels left).
    cut: bool,
}

/// The translation, card and driver the self-play's record holds for.
fn identity() -> (String, String, String) {
    let translation = match checked_translation() {
        Some(Ok(t)) => t,
        _ => String::new(),
    };
    match sys::gpu() {
        Ok(g) => (translation, g.name, g.driver.to_string()),
        Err(_) => (translation, String::new(), String::new()),
    }
}

fn played(app: &App) -> Played {
    let kv = read_kv(&played_file(app));
    let get = |k: &str| kv.iter().find(|(x, _)| x == k).map(|(_, v)| v.clone()).unwrap_or_default();
    let (translation, device, driver) = identity();
    if kv.is_empty() || get("device") != device || get("driver") != driver || (!translation.is_empty() && get("translation") != translation) {
        return Played::default();
    }
    let code = |k: &str| get(k).parse::<i32>().ok();
    Played { newgame: code("newgame"), parts: get("parts").parse().unwrap_or(0), tour: code("tour"), cut: get("cut") == "1" }
}

/// The self-play's record changed by `f` (with the translation, card and driver now).
fn record(app: &App, f: impl FnOnce(&mut Played)) {
    let mut p = played(app);
    f(&mut p);
    let (translation, device, driver) = identity();
    let line = |v: Option<i32>| v.map_or(String::new(), |c| c.to_string());
    let t = format!(
        "# The game's self-play that learned the shaders (dsport --prepare-play); written by the launcher\n# (0: played to its end, {STAGE_STOPPED}: stopped, other: it could not go on; parts: the levels' parts played; cut: the time ran out first)\nnewgame = {}\nparts = {}\ntour = {}\ncut = {}\ntranslation = {translation}\ndriver = {driver}\ndevice = {device}\n",
        line(p.newgame),
        p.parts,
        line(p.tour),
        p.cut as u8,
    );
    let _ = std::fs::write(played_file(app), t);
}

/// The prepared marker as the launcher writes it when the self-play's last part ended without
/// one: the date, this card and driver and this version's translation (`compute_state`'s keys).
fn write_marker(app: &App) {
    let (translation, device, driver) = identity();
    if translation.is_empty() || device.is_empty() {
        return;
    }
    let t = format!(
        "# The game's shaders prepared ahead; written by the launcher (the self-play's last part ended without marking them)\ndate = {}\nprograms = unknown\ndevice = {device}\ndriver = {driver}\ntranslation = {translation}\n",
        utc_stamp()
    );
    let _ = std::fs::write(marker(app), t);
}

/// Now as `YYYYMMDD-HHMMSS` (UTC).
fn utc_stamp() -> String {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs()) as i64;
    let (days, rem) = (secs.div_euclid(86400), secs.rem_euclid(86400));
    // Civil from days (Howard Hinnant).
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + if m <= 2 { 1 } else { 0 };
    format!("{y:04}{m:02}{d:02}-{:02}{:02}{:02}", rem / 3600, (rem / 60) % 60, rem % 60)
}

/// Whether the shaders are prepared for this graphics card and driver (the compute shaders, and
/// the self-play played or tried to its end); the status line.
pub fn state(app: &App) -> (bool, String) {
    let (ok, text) = compute_state(app);
    if !ok {
        return (ok, text);
    }
    let ended = |c: Option<i32>| c.is_some_and(|c| c != STAGE_STOPPED);
    let p = played(app);
    match (p.newgame, p.tour) {
        (Some(0), Some(0)) if !p.cut => (true, text),
        (Some(0), Some(0)) => (true, format!("{text} The latest levels are left to their first visits.")),
        // (A stage that could not go on: what it learned is kept, and play learns the rest.)
        (Some(c), _) if c != 0 && ended(Some(c)) => (true, format!("{text} Some graphics shaders could not be learned ahead.")),
        (Some(0), t) if ended(t) => (true, format!("{text} Some graphics shaders could not be learned ahead.")),
        _ => (false, "The shaders are partly prepared. Prepare now finishes them.".into()),
    }
}

pub fn prepared(app: &App) -> bool {
    wait_check();
    state(app).0
}

/// Waits for the translation check to end (a few seconds at most): Play waits for it, also to
/// start the game, as the check's run of the game holds the game's one-session lock meanwhile
/// (a start then is told the game already runs: seen 2026-11-14 with Play pressed 2 s after the
/// launcher opened and the preparation on Play turned off).
pub fn wait_check() {
    let t = Instant::now();
    while CHECK_STARTED.load(std::sync::atomic::Ordering::Relaxed) && checked_translation().is_none() && t.elapsed().as_secs() < 20 {
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
}

pub fn build(app: &mut App) {
    let b: Builder = app.builder();
    unsafe {
        let r = (16, 62, 748, 62);
        app.put(2, b.label("The game compiles each shader the first time it needs it, which makes play stutter. Preparing them here lets the game play by itself in the background: a new game, then every level with its enemies and bosses. Nothing of it shows or sounds, but the graphics card is busy. It takes about 30 to 60 minutes, once.", r, -1), r);
        let r = (16, 134, 748, 40);
        app.shaders.status = app.put(2, b.label("", r, -1), r) as usize;
        let r = (16, 178, 748, 20);
        app.shaders.bar = app.put(2, b.progress(r, -1), r) as usize;
        let r = (16, 210, 140, 30);
        app.shaders.prepare = app.put(2, b.button("Prepare now", r, ID_PREPARE), r) as usize;
        let r = (164, 210, 100, 30);
        app.shaders.stop = app.put(2, b.button("Stop", r, ID_STOP), r) as usize;
        let r = (16, 256, 748, 40);
        app.shaders.auto = app.put(2, b.check("Prepare the shaders when Play is pressed and they are not prepared yet.", r, ID_AUTO), r) as usize;
        ui::set_check(h(app.shaders.auto), app.launch.prepare_shaders);
        let where_ = format!(
            "Prepared shaders are kept in {}. They take up to 5 GB. Preparing uses {} of this PC's {} processor threads.",
            ui::short(&app.paths.shaders(), 70),
            sys::prepare_threads(),
            sys::threads()
        );
        let r = (16, 306, 748, 60);
        app.put(2, b.label(&where_, r, -1), r);
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
        ui::enable(h(app.shaders.prepare), app.game.ok);
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

/// Starts the preparation (`play_after`: the game starts when it is done) at its first step not
/// done: the compute shaders unless they are prepared, then the self-play's stages not played
/// (all three again when everything is done: Prepare now).
pub fn start(app: &mut App, play_after: bool) {
    // (Not without the game's files: it ended with "an error (code 3)", 2026-11-19.)
    if app.shaders.child.is_some() || app.running.is_some() || !app.game.ok {
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
            let t = format!("Only {} is free on the drive of {}. Preparing the shaders needs up to 5 GB. Prepare them anyway?", sys::gb(free), dir.display());
            if unsafe { ui::message(app.hwnd, &t, crate::TITLE, windows_sys::Win32::UI::WindowsAndMessaging::MB_ICONWARNING | windows_sys::Win32::UI::WindowsAndMessaging::MB_YESNO) } != windows_sys::Win32::UI::WindowsAndMessaging::IDYES {
                return;
            }
        }
    }
    let _ = std::fs::create_dir_all(&dir);
    wait_check();
    // (The next part of the levels when the new game was played for this version, card and
    // driver and the levels are not done; else, and when everything is done, the new game.)
    let p = played(app);
    let save = app.paths.cache.join("prepare").join("user").join("saves");
    let first = match p {
        Played { newgame: Some(0), parts, tour, .. } if parts < LEVEL_PARTS && tour != Some(0) && save.is_dir() => Step::Levels(parts + 1),
        _ => Step::NewGame,
    };
    app.shaders.deadline = Some(std::time::SystemTime::now() + BUDGET);
    app.shaders.play_after = play_after;
    app.shaders.stopping = None;
    app.shaders.then_play = false;
    if begin(app, first) {
        unsafe {
            ui::set_progress(h(app.shaders.bar), 0);
            ui::show(h(app.shaders.bar), true);
            ui::enable(h(app.shaders.prepare), false);
            ui::enable(h(app.shaders.stop), true);
            windows_sys::Win32::UI::WindowsAndMessaging::SetTimer(app.hwnd, TIMER, 500, None);
        }
        app.update_play();
        app.set_status(if play_after { "Preparing the shaders first. The game starts when they are done." } else { "" });
    }
}

/// Starts `step`'s child (false when it could not be started: the status says why).
fn begin(app: &mut App, step: Step) -> bool {
    let exe = dsconfig::paths::game_exe_in(&app.paths.exe_dir);
    let progress = progress_file(app);
    let _ = std::fs::remove_file(&progress);
    let _ = std::fs::remove_file(progress.with_extension("stop"));
    use std::os::windows::process::CommandExt;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    const BELOW_NORMAL_PRIORITY_CLASS: u32 = 0x0000_4000;
    let mut cmd = std::process::Command::new(&exe);
    match step.stage() {
        None => cmd.arg("--precompile-all").arg(sys::prepare_threads().to_string()),
        Some(s) => cmd.arg("--prepare-play").arg(s).arg("--seconds").arg(step.seconds().to_string()).arg("--prepare-threads").arg(sys::prepare_threads().to_string()),
    };
    if let Step::Levels(i) = step {
        cmd.arg("--prepare-chunk").arg(format!("{i}/{LEVEL_PARTS}"));
        if let Some(d) = app.shaders.deadline.and_then(|d| d.duration_since(std::time::UNIX_EPOCH).ok()) {
            cmd.arg("--prepare-deadline").arg(d.as_secs().to_string());
        }
    }
    cmd.arg("--progress").arg(&progress).current_dir(&app.paths.exe_dir).creation_flags(CREATE_NO_WINDOW | BELOW_NORMAL_PRIORITY_CLASS);
    match cmd.spawn() {
        Ok(c) => {
            app.shaders.child = Some(c);
            app.shaders.step = Some(step);
            app.shaders.started = Some(Instant::now());
            let text = match step {
                Step::Compute => "Preparing the compute shaders: reading the game's shader files...".to_string(),
                Step::NewGame => format!("Preparing the shaders, step 1 of {}: the game starts a new game in the background...", LEVEL_PARTS + 1),
                Step::Levels(_) => format!("Preparing the shaders, step {} of {}: the game loads its save in the background...", step.number(), LEVEL_PARTS + 1),
            };
            unsafe { ui::set_text(h(app.shaders.status), &text) };
            true
        }
        Err(e) => {
            app.shaders.child = None;
            app.shaders.step = None;
            app.set_status(&format!("Could not start the preparation: {e}"));
            false
        }
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

/// The preparation ended (`note`: what the page says when it is not the usual state).
fn finish(app: &mut App, note: Option<String>) {
    app.shaders.step = None;
    unsafe { windows_sys::Win32::UI::WindowsAndMessaging::KillTimer(app.hwnd, TIMER) };
    refresh(app);
    app.update_play();
    if let Some(n) = note {
        unsafe { ui::set_text(h(app.shaders.status), &n) };
    }
}

/// The timer: progress, and the end of each step.
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
    let step = app.shaders.step.unwrap_or(Step::Compute);
    if let Some(code) = exited {
        app.shaders.child = None;
        let _ = std::fs::remove_file(progress_file(app).with_extension("stop"));
        let stopped = app.shaders.stopping.take().is_some();
        let code = if stopped && code != 0 { STAGE_STOPPED } else { code };
        // (Another game session held the player's folder: the step did not run.)
        if code == GAME_RUNNING {
            app.shaders.play_after = false;
            finish(app, Some("The game is running, so the shaders cannot be prepared now. Close the game and press Prepare now.".into()));
            return;
        }
        let cut = app.shaders.deadline.is_some_and(|d| std::time::SystemTime::now() >= d);
        match step {
            // (A new game played again makes a new save: the levels are played again too. One
            // stopped leaves the record as it was: a store prepared before stays prepared.)
            Step::NewGame if code != STAGE_STOPPED => record(app, |p| *p = Played { newgame: Some(code), ..Played::default() }),
            Step::NewGame => {}
            // (A part that could not go on is passed: the next part plays its own levels.)
            Step::Levels(i) if code != STAGE_STOPPED => record(app, |p| {
                p.parts = p.parts.max(i);
                if i >= LEVEL_PARTS {
                    p.tour = Some(code);
                    p.cut = cut;
                }
            }),
            Step::Levels(_) => record(app, |p| p.tour = Some(STAGE_STOPPED)),
            Step::Compute => {}
        }
        // The last part ended without marking the store prepared (it could not go on, or the
        // session ended unasked): the launcher marks it, so that Play does not prepare again
        // (the release review of 2026-11-19).
        if step == Step::Levels(LEVEL_PARTS) && code != STAGE_STOPPED && !compute_state(app).0 {
            write_marker(app);
        }
        if stopped {
            finish(app, Some("Preparation stopped. The next one goes on where it left off.".into()));
            if std::mem::take(&mut app.shaders.then_play) {
                crate::start_game(app);
            }
            return;
        }
        let next = match step {
            Step::Compute if code != 0 || !compute_state(app).0 => {
                app.shaders.play_after = false;
                finish(app, Some(format!("Preparation ended with an error (code {code}). Open logs shows its log.")));
                return;
            }
            Step::NewGame if code == 0 => Some(Step::Levels(1)),
            // (The new game could not be played: the compute shaders on their own, as before.)
            Step::NewGame if !compute_state(app).0 => Some(Step::Compute),
            // (The next part even past the preparation's time: it marks the shaders prepared at
            // once then.)
            Step::Levels(i) if i < LEVEL_PARTS => Some(Step::Levels(i + 1)),
            _ => None,
        };
        if let Some(n) = next {
            if begin(app, n) {
                return;
            }
        }
        // (A stage that could not go on: what it learned is kept, and the game can be played.)
        let note = (code != 0 && step != Step::Compute).then(|| format!("The game's self-play could not finish (code {code}): some graphics shaders are learned only while you play. Open logs shows its log."));
        finish(app, note);
        if std::mem::take(&mut app.shaders.play_after) && prepared(app) {
            crate::start_game(app);
        }
        return;
    }
    if app.shaders.stopping.is_some() {
        return;
    }
    // Progress: "<done> <of>" and "<code done> <code>" (the self-play: its steps twice, then the
    // background compute shaders made and listed).
    let Ok(t) = std::fs::read_to_string(progress_file(app)) else { return };
    let nums: Vec<u64> = t.split_whitespace().filter_map(|w| w.parse().ok()).collect();
    if nums.len() >= 4 {
        let (done, total, wdone, wtotal) = (nums[0], nums[1], nums[2], nums[3]);
        let frac = if wtotal > 0 { wdone as f64 / wtotal as f64 } else { 0.0 };
        let el = app.shaders.started.map_or(0.0, |s| s.elapsed().as_secs_f64());
        let left = if frac > 0.02 { el * (1.0 - frac) / frac } else { 0.0 };
        let eta = if frac > 0.02 && step != Step::NewGame { format!(" About {} minutes left in this step.", (left / 60.0).ceil().max(1.0) as u64) } else { String::new() };
        let compute = match nums.get(4..6) {
            Some(&[c, all]) if all > 0 => format!(" Compute shaders: {c} of {all}."),
            _ => String::new(),
        };
        let steps = LEVEL_PARTS + 1;
        let budget = app.shaders.deadline.and_then(|d| d.duration_since(std::time::SystemTime::now()).ok()).map_or(String::new(), |l| format!(" At most {} minutes left.", (l.as_secs() / 60).max(1)));
        let text = match step {
            Step::Compute => format!("Preparing the compute shaders: {done} of {total} ({:.0}%).{eta}", frac * 100.0),
            Step::NewGame => format!("Preparing the shaders, step 1 of {steps}: the game plays a new game in the background, {done} of {total}.{budget}{compute}"),
            Step::Levels(_) => format!("Preparing the shaders, step {} of {steps}: the game visits its levels in the background, {done} of {total}.{budget}{compute}", step.number()),
        };
        unsafe {
            ui::set_text(h(app.shaders.status), &text);
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
