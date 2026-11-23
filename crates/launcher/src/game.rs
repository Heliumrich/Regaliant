//! The Game page: the game folder (checked when chosen, kept in game.txt once it passes), what
//! this PC brings (graphics card, memory, processor) and safe mode for the next start.

use crate::sys;
use crate::ui::{self, Builder};
use crate::App;
use std::path::Path;
use windows_sys::Win32::Foundation::HWND;

const ID_DIR: i32 = 101;
const ID_BROWSE: i32 = 102;
const ID_CHECK: i32 = 103;
const ID_SAFE: i32 = 106;

#[derive(Default)]
pub struct Page {
    dir: usize,
    status: usize,
    pc: usize,
    safe: usize,
    /// Whether the game folder passed the check.
    pub ok: bool,
}

fn h(v: usize) -> HWND {
    v as HWND
}

pub fn build(app: &mut App) {
    let b: Builder = app.builder();
    let dir = app.paths.chosen_game().filter(|d| d.exists()).map(|d| d.display().to_string()).unwrap_or_default();
    unsafe {
        let r = (16, 67, 100, 20);
        app.put(0, b.label("Game folder:", r, -1), r);
        let r = (116, 64, 444, 24);
        app.game.dir = app.put(0, b.edit(&dir, r, ID_DIR), r) as usize;
        let r = (568, 63, 96, 26);
        app.put(0, b.button("Browse...", r, ID_BROWSE), r);
        let r = (672, 63, 92, 26);
        app.put(0, b.button("Check", r, ID_CHECK), r);
        let r = (16, 102, 748, 90);
        app.game.status = app.put(0, b.label("", r, -1), r) as usize;
        let r = (12, 196, 756, 150);
        app.put(0, b.group("This PC", r), r);
        let r = (26, 218, 730, 120);
        app.game.pc = app.put(0, b.label("", r, -1), r) as usize;
        let r = (16, 360, 748, 40);
        app.game.safe = app.put(0, b.check("Start in safe mode once: default settings for that session, in a window, without an upscaler. Use it when the game does not start.", r, ID_SAFE), r) as usize;
        ui::set_check(h(app.game.safe), app.launch.safe_mode);
        // (Also in the launcher's log: what the player was told about this PC, in bug reports and
        // the small-card tests.)
        let pc = pc_text();
        ui::set_text(h(app.game.pc), &pc);
        crate::shaders::log(app, &format!("this PC: {}", pc.replace('\n', " | ")));
    }
    if dir.is_empty() {
        unsafe { ui::set_text(h(app.game.status), "Choose the game's folder: the PS5 game Demon's Souls (PPSA01341), version 01.007.000, with its executable decrypted.") };
        app.game.ok = false;
    } else {
        check(app, Path::new(&dir), false);
    }
}

/// What this PC brings, in short lines.
fn pc_text() -> String {
    let mut lines = Vec::new();
    match sys::gpu() {
        Ok(g) => {
            lines.push(format!("Graphics card: {}, {} of video memory, driver {}.", g.name, sys::gb(g.vram), g.driver_text));
            if g.api < ash::vk::API_VERSION_1_3 {
                lines.push("It offers an older Vulkan than the game needs (1.3). Update the graphics driver.".into());
            }
            if !g.barycentric && !g.geometry {
                lines.push("This card lacks shader features the game needs. Surfaces will look faceted.".into());
            } else if !g.barycentric {
                lines.push("This card lacks fragment shader barycentrics. The game draws some surfaces another way, which costs some speed.".into());
            }
            if g.vendor == sys::VENDOR_AMD || g.vendor == sys::VENDOR_INTEL {
                lines.push("AMD and Intel graphics cards are untested with this port. Please report how it runs.".into());
            }
            // (Measured in smaller rooms simulated on a 24 GB card, 2026-11-22: from 7 GB of room
            // the picture stays right with textures in system memory in busy places; 4 GB of room
            // and 3.5 GB (what a 4 GB card leaves a game) draw right in the 8-level tour, slowly; what a
            // 3 GB card leaves drew parts of some levels wrong.)
            if g.vram < 3584 << 20 {
                lines.push("Cards with less than 4 GB of video memory are not supported. Parts of the game can draw wrong.".into());
            } else if g.vram < 5632 << 20 {
                lines.push("This card has little video memory for this game. It runs, but slowly. The Low preset in the game's PC tab helps.".into());
            } else if g.vram < 11 << 30 {
                lines.push("On this card the game keeps some textures in system memory. Busy places can run slower.".into());
            }
        }
        Err(e) => lines.push(e),
    }
    let mem = sys::memory();
    lines.push(format!("Memory: {}. Processor: {} threads.", sys::gb(mem), sys::threads()));
    // (The card reads the game's memory through Windows' graphics share of it, about half the
    // installed memory. A 16 GB PC simulated on the tested one: on 2026-11-22 afternoon the game
    // froze at the first level load and lost its graphics device; since that night's eviction
    // and write-back fallback it played every tested level, a first session as fast as with
    // room.)
    if mem < 15 << 30 {
        lines.push("The game needs 16 GB of memory. It may stop when memory runs out.".into());
    } else if mem < 24 << 30 {
        lines.push("16 GB of memory has not been tested on a real PC. 32 GB is recommended.".into());
        lines.push("Keep Windows' page file on automatic and about 30 GB free on the system drive.".into());
    } else if mem < 30 << 30 {
        lines.push("Keep Windows' page file on automatic and about 30 GB free on the system drive.".into());
    }
    if sys::threads() < 8 {
        lines.push("The game was tested with at least 6 processor cores. It may run slowly here.".into());
    }
    lines.join("\n")
}

/// Checks the folder; once it passes (and `keep`), game.txt takes it.
pub fn check(app: &mut App, dir: &Path, keep: bool) {
    let (ok, text) = match dsconfig::game::check(dir) {
        Ok(g) => {
            if keep {
                if let Err(e) = app.paths.set_game(&g.app) {
                    app.set_status(&format!("Could not save the game folder: {e}"));
                }
            }
            let mut t = format!("Demon's Souls {} is ready.\nFound in {}.", g.version, ui::short(&g.app, 90));
            if g.id != dsconfig::game::TITLE_ID {
                t.push_str(&format!("\nThis is the {} release. The port was tested with {}.", g.id, dsconfig::game::TITLE_ID));
            }
            match dsconfig::game::tested_executable(&g) {
                Ok(true) => {}
                Ok(false) => t.push_str("\nIts executable differs from the one this port was tested with. It may not work."),
                Err(e) => t.push_str(&format!("\nIts executable could not be read ({e}).")),
            }
            (true, t)
        }
        Err(e) => (false, e.to_string()),
    };
    app.game.ok = ok;
    unsafe { ui::set_text(h(app.game.status), &text) };
    // (A folder chosen now: the prepared shaders' check, which needs the game's files, begins;
    // the Shaders page says what it can do now.)
    if keep {
        if ok {
            crate::shaders::start_check(app);
            unsafe { windows_sys::Win32::UI::WindowsAndMessaging::SetTimer(app.hwnd, crate::shaders::CHECK_TIMER, 500, None) };
        }
        crate::shaders::refresh(app);
    }
    app.update_play();
    crate::saves::refresh(app);
    unsafe { ui::redraw(app.hwnd) };
}

/// Safe mode for the next start (the box), cleared once used.
pub fn safe_mode(app: &App) -> bool {
    unsafe { ui::checked(h(app.game.safe)) }
}

pub fn clear_safe_mode(app: &mut App) {
    unsafe { ui::set_check(h(app.game.safe), false) };
    app.launch.safe_mode = false;
    app.save_launch();
}

fn check_field(app: &mut App) {
    let t = unsafe { ui::get_text(h(app.game.dir)) }.trim().trim_matches('"').to_string();
    if t.is_empty() {
        app.game.ok = false;
        unsafe { ui::set_text(h(app.game.status), "Choose the game's folder.") };
        crate::shaders::refresh(app);
        app.update_play();
        return;
    }
    check(app, Path::new(&t), true);
}

pub fn command(app: &mut App, id: i32, code: u32) {
    match (id, code) {
        (ID_CHECK, 0) => check_field(app),
        (ID_BROWSE, 0) => {
            if let Some(p) = ui::pick_folder(app.hwnd, "The Demon's Souls game folder (PPSA01341, 01.007.000)") {
                unsafe { ui::set_text(h(app.game.dir), &p.display().to_string()) };
                check_field(app);
            }
        }
        (ID_SAFE, 0) => {
            app.launch.safe_mode = safe_mode(app);
            app.save_launch();
            app.set_status(if app.launch.safe_mode { "The next start uses safe mode." } else { "" });
        }
        _ => {}
    }
}
