//! The shader preparation's self-play (`--prepare-play <stage>`, run by the launcher after the
//! compute programs are prepared): the game plays itself before the player's first session, so
//! the graphics pipelines a playthrough meets, and the compute programs' real specs, are made
//! and kept in the player's shader store ahead of it. (A store prepared for compute alone left a
//! first session ~380 graphics pipelines to make while it played and ~180 compute programs whose
//! guessed variants did not match, 2026-11-19.)
//!
//! Nothing of it shows or sounds: no window (`--headless`: a swapchain that is never shown) and
//! no message boxes, no audio device (`audio::SILENT`); the system's controllers, keyboard and
//! mouse are neither read nor rumbled (`input::SYSTEM_OFF`). It plays through its own presses,
//! reading its log as the development drivers do (`log::tap_read`). Its settings (the player's
//! PC settings and display choices, copied, its saves always its own), saves, trophies and the
//! title's own folders are its own (`<cache>/prepare`); the shader store is the player's, and the
//! player's one-session lock is taken (the two never run together).
//!
//! Stages:
//! - `newgame`: from no saves, the first-run screens, a new game (character creation with its
//!   menus opened), the opening cinematic played, the tutorial's first steps, the in-game menu
//!   with its character preview.
//! - `tour`: the save `newgame` made, continued; the title's own automation (its developers'
//!   screenshot positions, `misc/automation/demonssoulsscreenshots.txt`) visits the levels'
//!   positions and looks four ways at each (`DS_PREPARE_TOUR` changes the choice).
//!
//! Progress in `--progress <file>` (`<steps done> <steps>` on two lines, as the precompile writes
//! it); the stage stops when the launcher's stop file appears (`<file>` with the extension
//! `.stop`, as the precompile reads it). Exit status 0 when the stage was played to its
//! end, `EXIT_STOPPED` when the stop file ended it, `EXIT_FAILED` when it could not go on (its log
//! says where).

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::sync::atomic::Ordering::Relaxed;
use std::time::{Duration, Instant};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Stage {
    NewGame,
    Tour,
}

impl Stage {
    pub fn parse(s: &str) -> Option<Stage> {
        match s {
            "newgame" => Some(Stage::NewGame),
            "tour" => Some(Stage::Tour),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Stage::NewGame => "newgame",
            Stage::Tour => "tour",
        }
    }
}

/// A stage stopped by the launcher's stop file.
pub const EXIT_STOPPED: i32 = 3;
/// A stage that could not go on (a screen it waited for did not come).
pub const EXIT_FAILED: i32 = 4;

/// The character's name in the self-play's own save.
const NAME: &str = "Prepare";

/// The text-database lookup of PPSA01341 01.007.000 (`exception::text_hook`): the stages read the
/// screens by their texts, as the development drivers do.
const TEXT_HOOK: &str = "86d2ee:r13+8*:r13+10*";

/// PC settings never taken from the player's: where saves are kept (the Save Folder may be the
/// player's Documents or Saved Games) and a PS5 import asked for.
const PRIVATE_SETTINGS: [&str; 2] = ["Setting_PC_SaveFolder", "Setting_PC_ImportSaves"];

struct Setup {
    stage: Stage,
    /// The player's settings folder: its one-session lock is taken (`main::single_instance`).
    player_user: PathBuf,
    progress: Option<PathBuf>,
    /// The self-play's own title folders (the tour's automation file goes to its devlog).
    runtime: PathBuf,
}

static SETUP: OnceLock<Setup> = OnceLock::new();

/// The player's settings folder, whose one-session lock a self-play run takes.
pub fn player_user() -> Option<&'static Path> {
    SETUP.get().map(|s| s.player_user.as_path())
}

/// Sets a self-play run up (`main`, before any thread starts): its own folders under the
/// player's cache (`prepare`), the player's shader store, no window, no sound, no system input,
/// the text hook its stages read the screens by. Its log stays in the player's logs folder
/// (`<time>-prepare-<stage>`).
pub fn setup(stage: Stage, cfg: &mut crate::Config) {
    let player = cfg.paths.clone();
    let store = cfg.cache.clone().unwrap_or_else(|| player.shaders());
    let dir = player.cache.join("prepare");
    let user = dir.join("user");
    let _ = std::fs::create_dir_all(&user);
    // The player's PC settings (some choose shader variants: the upscaler, the Visual Mode),
    // display choices and game folder; never their saves, key bindings or trophies.
    match std::fs::read_to_string(player.settings_file()) {
        Ok(t) => {
            let kept: String = t.lines().filter(|l| !PRIVATE_SETTINGS.iter().any(|k| l.trim_start().starts_with(k))).map(|l| format!("{l}\n")).collect();
            let _ = std::fs::write(user.join("pcsettings.txt"), kept);
        }
        Err(_) => {
            let _ = std::fs::remove_file(user.join("pcsettings.txt"));
        }
    }
    for f in ["display.txt", "game.txt"] {
        let src = player.user.join(f);
        if src.is_file() {
            let _ = std::fs::copy(&src, user.join(f));
        } else {
            let _ = std::fs::remove_file(user.join(f));
        }
    }
    let paths = dsconfig::paths::Paths { layout: player.layout, exe_dir: player.exe_dir.clone(), user, cache: dir, root: player.root.clone() };
    if stage == Stage::NewGame {
        // A new game from nothing: no saves, trophies or title folders of an earlier one.
        for d in [paths.saves(), paths.save_backups(), paths.runtime()] {
            let _ = std::fs::remove_dir_all(d);
        }
        for f in [paths.save_folder_record(), paths.achievements_file()] {
            let _ = std::fs::remove_file(f);
        }
    }
    if let Some(name) = cfg.run_dir.file_name().map(|n| n.to_string_lossy().into_owned()) {
        cfg.run_dir = cfg.run_dir.with_file_name(format!("{name}-prepare-{}", stage.name()));
    }
    // SAFETY: still the only thread (`main` calls this right after `parse_args`).
    unsafe {
        if std::env::var_os("DS_TEXT_HOOK").is_none() {
            std::env::set_var("DS_TEXT_HOOK", TEXT_HOOK);
        }
        std::env::set_var("DS_IME_TEXT", NAME);
    }
    crate::audio::SILENT.store(true, Relaxed);
    crate::input::SYSTEM_OFF.store(true, Relaxed);
    let runtime = paths.runtime();
    cfg.paths = paths;
    cfg.cache = Some(store);
    cfg.headless = true;
    let _ = SETUP.set(Setup { stage, player_user: player.user, progress: cfg.progress.clone(), runtime });
}

/// Starts the stage (`main`, once the log is open; before the title starts: the tour's
/// automation is read at its boot): its driver thread plays it and ends the session.
pub fn start(app: &Path) {
    let Some(s) = SETUP.get() else { return };
    crate::log::tap_on();
    crate::info!(
        "prepare-play: stage {} (the shader preparation's self-play: no window, no sound, the system's controllers, keyboard and mouse left alone; its settings, saves and title folders in {})",
        s.stage.name(),
        crate::paths().cache.display()
    );
    let devlog = s.runtime.join("devlog");
    let _ = std::fs::create_dir_all(&devlog);
    let args = devlog.join("commandlineargs.txt");
    let total = match s.stage {
        Stage::NewGame => {
            let _ = std::fs::remove_file(&args);
            NEWGAME_STEPS
        }
        Stage::Tour => match tour_script(app) {
            Ok((text, views)) => {
                // (No '-' in the name: `automate_loadfile=#/prepare-tour.txt` failed to execute,
                // its usage printed, 2026-11-19.)
                let written = std::fs::write(devlog.join("preparetour.txt"), text).and_then(|_| std::fs::write(&args, "+automate_loadfile=#/preparetour.txt +automate_go\n"));
                if let Err(e) = written {
                    crate::error!("prepare-play: writing the tour into {}: {e}", devlog.display());
                    end(EXIT_FAILED);
                }
                crate::info!("prepare-play: the tour has {views} views");
                views + TOUR_STEPS
            }
            Err(e) => {
                crate::error!("prepare-play: the tour: {e}");
                end(EXIT_FAILED);
            }
        },
    };
    let (stage, progress) = (s.stage, s.progress.clone());
    let spawned = std::thread::Builder::new().name("prepare-play".into()).spawn(move || {
        crate::log::set_thread_name("prepare-play");
        let mut p = Play::new(progress, total);
        let r = match stage {
            Stage::NewGame => p.newgame(),
            Stage::Tour => p.tour(),
        };
        let code = match r {
            Ok(()) => {
                p.drain(180);
                p.done();
                crate::info!("prepare-play: stage {} played to its end in {:.0} s", stage.name(), p.t0.elapsed().as_secs_f64());
                0
            }
            Err(End::Stopped) => {
                crate::info!("prepare-play: stopped (the launcher's stop file) after {:.0} s", p.t0.elapsed().as_secs_f64());
                EXIT_STOPPED
            }
            Err(End::Failed(why)) => {
                crate::error!("prepare-play: stage {} could not go on: {why} (after {:.0} s)", stage.name(), p.t0.elapsed().as_secs_f64());
                p.drain(60);
                EXIT_FAILED
            }
        };
        end(code);
    });
    if let Err(e) = spawned {
        crate::error!("prepare-play: no driver thread ({e})");
        end(EXIT_FAILED);
    }
}

fn end(code: i32) -> ! {
    crate::shutdown(code)
}

enum End {
    Stopped,
    Failed(String),
}

/// The log's lines as the stages read them (`log::tap_read`), the last `KEEP`.
struct Watch {
    lines: VecDeque<String>,
    /// The index of `lines[0]` among all lines read.
    first: usize,
    cursor: u64,
}

const KEEP: usize = 200_000;

impl Watch {
    fn poll(&mut self) {
        let (new, c) = crate::log::tap_read(self.cursor);
        self.cursor = c;
        self.lines.extend(new);
        while self.lines.len() > KEEP {
            self.lines.pop_front();
            self.first += 1;
        }
    }

    /// The index of the next line.
    fn mark(&mut self) -> usize {
        self.poll();
        self.first + self.lines.len()
    }

    fn since(&self, at: usize) -> impl Iterator<Item = &String> {
        self.lines.iter().skip(at.saturating_sub(self.first))
    }

    fn found(&mut self, pat: &str, at: usize) -> bool {
        self.poll();
        self.since(at).any(|l| matches(l, pat))
    }

    fn count(&mut self, pat: &str, at: usize) -> usize {
        self.poll();
        self.since(at).filter(|l| matches(l, pat)).count()
    }
}

/// `pat`'s parts (split at `.*`) in `line`, in order.
fn matches(line: &str, pat: &str) -> bool {
    let mut at = 0;
    for part in pat.split(".*") {
        match line[at..].find(part) {
            Some(i) => at += i + part.len(),
            None => return false,
        }
    }
    true
}

struct Play {
    w: Watch,
    progress: Option<PathBuf>,
    stop: Option<PathBuf>,
    done_steps: u32,
    total: u32,
    t0: Instant,
}

/// The new game's steps (progress): the title, character creation, its menus, the character
/// made, the cinematic, in the game, the walk, the menu.
const NEWGAME_STEPS: u32 = 8;
/// The tour's steps besides its views: the title, in the game, the tour started.
const TOUR_STEPS: u32 = 3;

impl Play {
    fn new(progress: Option<PathBuf>, total: u32) -> Play {
        // (The launcher's stop file, as the precompile reads it.)
        let stop = progress.as_ref().map(|p| p.with_extension("stop"));
        let p = Play { w: Watch { lines: VecDeque::new(), first: 0, cursor: 0 }, progress, stop, done_steps: 0, total: total.max(1), t0: Instant::now() };
        p.write_progress();
        p
    }

    fn write_progress(&self) {
        if let Some(p) = &self.progress {
            let t = format!("{} {}\n{} {}\n", self.done_steps, self.total, self.done_steps, self.total);
            let tmp = p.with_extension("tmp");
            if std::fs::write(&tmp, t).is_ok() {
                let _ = std::fs::rename(&tmp, p);
            }
        }
    }

    /// A step done (`what`: what comes next).
    fn step(&mut self, what: &str) {
        self.done_steps = (self.done_steps + 1).min(self.total);
        self.write_progress();
        crate::info!("prepare-play: {} of {} ({what}) at {:.0} s", self.done_steps, self.total, self.t0.elapsed().as_secs_f64());
    }

    fn done(&mut self) {
        self.done_steps = self.total;
        self.write_progress();
    }

    fn check_stop(&self) -> Result<(), End> {
        if self.stop.as_ref().is_some_and(|f| f.exists()) {
            return Err(End::Stopped);
        }
        Ok(())
    }

    fn sleep(&mut self, secs: f64) -> Result<(), End> {
        let until = Instant::now() + Duration::from_secs_f64(secs);
        loop {
            self.check_stop()?;
            let now = Instant::now();
            if now >= until {
                return Ok(());
            }
            std::thread::sleep((until - now).min(Duration::from_millis(250)));
        }
    }

    /// Waits up to `secs` for a line matching `pat` from line `at` on.
    fn wait_for(&mut self, pat: &str, at: usize, secs: f64) -> Result<bool, End> {
        let until = Instant::now() + Duration::from_secs_f64(secs);
        loop {
            if self.w.found(pat, at) {
                return Ok(true);
            }
            self.check_stop()?;
            if Instant::now() >= until {
                return Ok(false);
            }
            std::thread::sleep(Duration::from_millis(250));
        }
    }

    fn press(&self, names: &str, ms: u64) {
        if let Err(e) = crate::input::press_for(names, Duration::from_millis(ms)) {
            crate::warn!("prepare-play: press {names}: {e}");
        }
    }

    /// Cross until a line matching `pat` comes (at most `tries` presses, 6 s apart).
    fn cross_until(&mut self, pat: &str, at: usize, tries: u32) -> Result<bool, End> {
        for _ in 0..tries {
            if self.w.found(pat, at) {
                return Ok(true);
            }
            self.press("cross", 300);
            if self.wait_for(pat, at, 6.0)? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// In the game: HUD scenes up and no menu open (`uistate`), for two seconds running, or
    /// `hud_text` logged since `at`.
    fn wait_in_game(&mut self, at: usize, hud_text: &str, secs: f64) -> Result<bool, End> {
        let until = Instant::now() + Duration::from_secs_f64(secs);
        let mut since: Option<Instant> = None;
        loop {
            if self.w.found(hud_text, at) {
                return Ok(true);
            }
            let up = crate::uistate::hud_count().is_some_and(|n| n > 0) && crate::uistate::menu_depth() == Some(0);
            match (up, since) {
                (true, None) => since = Some(Instant::now()),
                (true, Some(t)) if t.elapsed() >= Duration::from_secs(2) => return Ok(true),
                (false, _) => since = None,
                _ => {}
            }
            self.check_stop()?;
            if Instant::now() >= until {
                return Ok(false);
            }
            std::thread::sleep(Duration::from_millis(250));
        }
    }

    /// The pipelines queued for the background graphics workers made (at most `secs`): they are
    /// kept as they are made, and this session ends after them.
    fn drain(&mut self, secs: u64) {
        let s = &crate::gpu::exec::draw::ASYNC_GFX_STATS;
        let pending = || s[1].load(Relaxed).saturating_sub(s[3].load(Relaxed) + s[7].load(Relaxed));
        let t = Instant::now();
        while pending() > 0 && t.elapsed() < Duration::from_secs(secs) {
            std::thread::sleep(Duration::from_millis(250));
        }
        crate::info!("prepare-play: {} graphics pipelines queued, {} made, {} failed, {} still queued", s[1].load(Relaxed), s[3].load(Relaxed), s[7].load(Relaxed), pending());
    }

    /// Waits up to `secs` for a line matching any of `pats` from line `at` on.
    fn wait_any(&mut self, pats: &[&str], at: usize, secs: f64) -> Result<bool, End> {
        let until = Instant::now() + Duration::from_secs_f64(secs);
        loop {
            if pats.iter().any(|p| self.w.found(p, at)) {
                return Ok(true);
            }
            self.check_stop()?;
            if Instant::now() >= until {
                return Ok(false);
            }
            std::thread::sleep(Duration::from_millis(250));
        }
    }

    /// The title's boot to its title screen ("PRESS ANY BUTTON"), through its attract movie and
    /// first-run screens (Cross held on each, as the development drivers do), then to the main
    /// menu.
    fn title(&mut self) -> Result<(), End> {
        if !self.wait_any(&["attract_movie.bk2", "text: "], 0, 300.0)? {
            return Err(End::Failed("the title did not start (no attract movie, no text)".into()));
        }
        self.sleep(2.0)?;
        let m = self.w.mark();
        let mut ok = false;
        for _ in 0..10 {
            if self.w.found("PRESS ANY BUTTON", m) || self.w.found("main_menu.bk2", m) {
                ok = true;
                break;
            }
            self.press("cross", 4000);
            if self.wait_any(&["PRESS ANY BUTTON", "main_menu.bk2"], m, 8.0)? {
                ok = true;
                break;
            }
        }
        if !ok {
            return Err(End::Failed("no title screen".into()));
        }
        self.sleep(2.0)?;
        if !self.cross_until("main_menu.bk2", m, 6)? {
            return Err(End::Failed("no main menu".into()));
        }
        self.sleep(2.0)
    }

    fn newgame(&mut self) -> Result<(), End> {
        self.title()?;
        self.step("the main menu: a new game");
        // NEW GAME is the first entry without saves; the offline notice; character creation.
        let m = self.w.mark();
        if !self.cross_until("Enter your player's name", m, 8)? {
            return Err(End::Failed("no character creation".into()));
        }
        self.step("character creation");
        self.sleep(3.0)?;
        // The entries below the name: class, gift, saved creations, foundation, appearance,
        // Finalise. Each but the saved creations is opened (Cross), changed (Right) and left
        // (Circle): their previews. (Cross on an empty saved-creation slot opens nothing, and
        // Circle there would open the quit prompt: that row is only passed.)
        for e in ["class", "gift", "saveload", "foundation", "appearance"] {
            self.press("down", 300);
            self.sleep(2.0)?;
            if e == "saveload" {
                continue;
            }
            for b in ["cross", "right", "circle"] {
                self.press(b, 300);
                self.sleep(3.0)?;
            }
        }
        self.press("down", 300);
        self.sleep(2.0)?;
        if !self.w.found("Finalise your character.", m) {
            crate::warn!("prepare-play: the cursor may not be on Finalise");
        }
        self.step("the character made");
        let m = self.w.mark();
        self.press("cross", 300);
        if !self.wait_for("committed [\"SAVEDATA0PlayerProfile0\"]", m, 60.0)? {
            return Err(End::Failed("the character was not saved".into()));
        }
        // The opening cinematic, played to its end (its own graphics).
        if self.wait_for("create thread 'Bink IO'", m, 60.0)? {
            self.step("the opening cinematic");
        }
        if !self.wait_in_game(m, "Crescent Moon Grass", 420.0)? {
            return Err(End::Failed("not in the game after the cinematic".into()));
        }
        self.step("in the game: the tutorial's first steps");
        self.sleep(8.0)?;
        for (what, secs) in [("ls_up", 6.0), ("rs_right", 3.0), ("ls_up", 6.0), ("rs_left", 5.0), ("ls_up", 6.0), ("rs_right", 6.0), ("ls_up", 4.0)] {
            self.press(what, (secs * 1000.0) as u64);
            self.sleep(secs + 0.7)?;
        }
        self.step("the in-game menu");
        self.menu()?;
        self.sleep(4.0)?;
        Ok(())
    }

    /// The in-game menu opened (Options, again until its equipment screen's character preview is
    /// rendered: its 640x360 depth target) and closed (Circle).
    fn menu(&mut self) -> Result<(), End> {
        let all = self.w.mark();
        for _ in 0..3 {
            let m = self.w.mark();
            self.press("options", 300);
            if self.wait_for("new render target .*640x360 D32", m, 8.0)? {
                break;
            }
            self.sleep(10.0)?;
        }
        if !self.w.found("new render target .*640x360 D32", all) && !self.w.found("new render target .*640x360 D32", 0) {
            crate::warn!("prepare-play: the menu's character preview was not seen");
        }
        self.sleep(3.0)?;
        // Its other pages (R1), then closed.
        for _ in 0..4 {
            self.press("r1", 300);
            self.sleep(2.5)?;
        }
        self.press("circle", 300);
        self.sleep(3.0)
    }

    fn tour(&mut self) -> Result<(), End> {
        self.title()?;
        self.step("the main menu: the save continued");
        let m = self.w.mark();
        if !self.cross_until("Continue Offline", m, 6)? {
            return Err(End::Failed("no offline notice".into()));
        }
        self.sleep(2.0)?;
        let m = self.w.mark();
        self.press("cross", 300);
        if !self.wait_in_game(m, "Crescent Moon Grass", 180.0)? {
            return Err(End::Failed("not in the game after the load".into()));
        }
        self.step("in the game: the tour's start");
        // The automation's own waits, then its views (`dsport-tour:` before each). One that
        // started before the save was loaded had its level loads ignored.
        if self.w.found("dsport-tour: start", 0) {
            return Err(End::Failed("the tour started before the game was loaded".into()));
        }
        let m = self.w.mark();
        if !self.wait_for("dsport-tour: start", m, 300.0)? {
            return Err(End::Failed("the tour did not start".into()));
        }
        self.step("the tour");
        let mut seen = 0usize;
        let mut last = Instant::now();
        loop {
            if self.w.found("dsport-tour: done", m) {
                return Ok(());
            }
            let n = self.w.count("dsport-tour: view", m);
            if n > seen {
                for _ in seen..n {
                    self.done_steps = (self.done_steps + 1).min(self.total);
                }
                self.write_progress();
                seen = n;
                last = Instant::now();
            }
            // (A level load and its resources take ~10-60 s; a view that never ends is a stall.)
            if last.elapsed() > Duration::from_secs(600) {
                return Err(End::Failed(format!("the tour stopped after {seen} views")));
            }
            self.sleep(1.0)?;
        }
    }
}

/// The tour's automation script (the title's developer automation: `tools/tour.py` writes the
/// development tours the same way) and its number of views: the screenshot positions of
/// `misc/automation/demonssoulsscreenshots.txt`, each level loaded once, the camera at each
/// position looking four ways (the position's yaw and three quarter turns from it).
/// `DS_PREPARE_TOUR=<positions a level>[:<directions>]` changes the choice (0 positions: all).
fn tour_script(app: &Path) -> Result<(String, u32), String> {
    let src = app.join("misc").join("automation").join("demonssoulsscreenshots.txt");
    let text = std::fs::read_to_string(&src).map_err(|e| format!("{}: {e}", src.display()))?;
    let (per_level, dirs) = std::env::var("DS_PREPARE_TOUR")
        .ok()
        .and_then(|v| {
            let mut it = v.split(':');
            let p = it.next()?.trim().parse::<usize>().ok()?;
            let d = it.next().map_or(Some(4), |d| d.trim().parse::<u32>().ok())?;
            Some((p, d.clamp(1, 8)))
        })
        .unwrap_or((0, 4));
    tour_lines(&text, per_level, dirs).ok_or_else(|| format!("no positions in {}", src.display()))
}

/// The tour's script from the automation file's text (None: no positions in it).
fn tour_lines(text: &str, per_level: usize, dirs: u32) -> Option<(String, u32)> {
    // (level, camera words) in the file's order.
    let mut locs: Vec<((u32, u32), Vec<(String, f64)>)> = Vec::new();
    for line in text.lines() {
        let l = line.trim();
        let Some(rest) = l.strip_prefix("*callmacro doArea") else { continue };
        let rest = rest.trim_start_matches("Stats").trim_start_matches("Internal");
        // Two quoted strings (the area, the image), the world and level, then name=value words.
        let mut q = rest.split('"');
        let (_, _area, _, _png) = (q.next(), q.next(), q.next(), q.next());
        let Some(tail) = q.next() else { continue };
        let mut words = tail.split_whitespace();
        let (Some(w), Some(lv)) = (words.next().and_then(|v| v.parse().ok()), words.next().and_then(|v| v.parse().ok())) else { continue };
        let cam: Vec<(String, f64)> = words.filter_map(|kv| kv.split_once('=')).filter_map(|(k, v)| Some((k.to_string(), v.parse::<f64>().ok()?))).collect();
        if cam.len() == 6 {
            locs.push(((w, lv), cam));
        }
    }
    if locs.is_empty() {
        return None;
    }
    let mut levels: Vec<(u32, u32)> = Vec::new();
    for (lv, _) in &locs {
        if !levels.contains(lv) {
            levels.push(*lv);
        }
    }
    // (The automation counts game time from the title's boot, and the title ignores level loads
    // until it is in the game: its first wait covers the self-play's way through the title, the
    // main menu and the save's load, ~50-80 s.)
    let mut out = vec![
        "// The shader preparation's tour (written by the game's self-play from misc/automation/demonssoulsscreenshots.txt)".to_string(),
        "wait 120.0".to_string(),
        "waitForLevelLoading".to_string(),
        "waitForResourceLoading".to_string(),
        "print \"dsport-tour: start\"".to_string(),
    ];
    let mut views = 0u32;
    for lv in &levels {
        let here: Vec<&Vec<(String, f64)>> = locs.iter().filter(|(l, _)| l == lv).map(|(_, c)| c).collect();
        // `per_level` positions spread over the level's list (0: all).
        let picked: Vec<&Vec<(String, f64)>> = if per_level == 0 || here.len() <= per_level {
            here
        } else {
            (0..per_level).map(|i| here[i * here.len() / per_level]).collect()
        };
        out.push(format!("+cp11_loadLevel {} {}", lv.0, lv.1));
        let mut first = true;
        for cam in picked {
            for d in 0..dirs {
                let words: Vec<String> = cam
                    .iter()
                    .map(|(k, v)| {
                        let v = if k == "eulerY" { (v + 360.0 * d as f64 / dirs as f64).rem_euclid(360.0) } else { *v };
                        format!("{k}={v:.2}")
                    })
                    .collect();
                views += 1;
                out.push(format!("print \"dsport-tour: view {views} (level {}-{})\"", lv.0, lv.1));
                out.push(format!("+SetCameraPos {}", words.join(" ")));
                if first {
                    // The level's load (the camera moves the area loader).
                    out.extend(["wait 5.0", "waitForLevelLoading", "waitForResourceLoading", "wait 3.0", "waitForResourceLoading"].map(String::from));
                    first = false;
                } else if d == 0 {
                    // Another position: its areas streamed in.
                    out.extend(["wait 2.0", "waitForResourceLoading", "wait 1.5"].map(String::from));
                } else {
                    out.push("wait 1.5".to_string());
                }
            }
        }
    }
    out.push("print \"dsport-tour: done\"".to_string());
    out.push(String::new());
    Some((out.join("\n"), views))
}

#[cfg(test)]
mod tests {
    #[test]
    fn patterns() {
        assert!(super::matches("I: new render target 0x10 640x360 D32 (x)", "new render target .*640x360 D32"));
        assert!(!super::matches("I: new render target 0x10 640x360 R8", "new render target .*640x360 D32"));
        assert!(super::matches("committed [\"SAVEDATA0PlayerProfile0\"] ok", "committed [\"SAVEDATA0PlayerProfile0\"]"));
    }

    /// The tour from an automation text of the title's shape (made-up positions): each level
    /// loaded once, its positions spread, four yaws a position, the macro's own lines skipped.
    #[test]
    fn tour() {
        let text = "*macro doAreaInternal areaName screenshotName world level camX camY camZ camRotX camRotY camRotZ\n\
            \x20   *callmacro doAreaInternal areaName screenshotName world level camX camY camZ camRotX camRotY camRotZ\n\
            *callmacro doArea \"Area A\" \"#/a/A_0.png\" 1 1 posX=1.00 posY=2.00 posZ=3.00 eulerX=-5.00 eulerY=300.00 eulerZ=0.00\n\
            *callmacro doAreaStats \"Area A\" \"#/a/A_1.png\" 1 1 posX=4.00 posY=5.00 posZ=6.00 eulerX=0.00 eulerY=10.00 eulerZ=0.00\n\
            *callmacro doArea \"Area B\" \"#/b/B_0.png\" 2 1 posX=7.00 posY=8.00 posZ=9.00 eulerX=0.00 eulerY=90.00 eulerZ=0.00\n";
        let (s, views) = super::tour_lines(text, 0, 4).unwrap();
        assert_eq!(views, 12);
        assert_eq!(s.matches("+cp11_loadLevel").count(), 2);
        assert!(s.contains("+cp11_loadLevel 1 1") && s.contains("+cp11_loadLevel 2 1"));
        // 300 + 90 wraps to 30.
        assert!(s.contains("posX=1.00 posY=2.00 posZ=3.00 eulerX=-5.00 eulerY=30.00 eulerZ=0.00"));
        assert!(s.contains("dsport-tour: view 12 (level 2-1)") && s.ends_with("print \"dsport-tour: done\"\n"));
        let (_, one) = super::tour_lines(text, 1, 2).unwrap();
        assert_eq!(one, 4);
        assert!(super::tour_lines("nothing here", 0, 4).is_none());
    }
}
