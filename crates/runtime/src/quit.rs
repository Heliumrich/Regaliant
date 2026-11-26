//! Quitting from the game's own menus (the user's request of 2026-11-25 evening: a Quit Game
//! where a normal PC game has one, in the game's own style, no overlay, never losing progress).
//!
//! The title menu gets an Exit Game item after Gallery (`pcsettings::front_end`): a copy of its
//! Settings item showing the game's own `SETTINGS_HEADING_EXITGAME` text ("EXIT GAME" in every
//! language), its choice `QuitGame`. The title's handler of a menu choice (eboot+0x7c9ae0)
//! reads the choice's hash into r12d and compares it with its items' from eboot+0x7c9d70; a
//! choice it does not know leaves the menu as it was. `menu_hook` sees `QuitGame` there and
//! asks the main loop to end the session as closing the window does, and the main loop waits
//! for the save being written, if any, before it ends (`wait_for_saves`; saves are committed by
//! renames, `hle::libs::savedata`, so an end never breaks one, but a write still open would be
//! lost).
//!
//! In game, the SETTINGS screen's PC page ends with Save and Quit to Desktop
//! (`pcsettings::quit_button`): a copy of the game's own Save and Exit Game button from its EXIT
//! GAME page (the game saves and returns to the title), 51 units high where the original is 50,
//! which is how `tick` knows Cross was pressed on it (the UI nodes, `uistate`, as the key
//! binding rows are found); hidden on the title screen, where the title menu has EXIT GAME. The
//! session then ends when the title's own movies open again
//! (`note_open`): the game has left the game world, its save written (within a minute of the
//! press; a press the game did not act on lapses then).

use parking_lot::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::time::{Duration, Instant};

/// Set when the player chose Exit Game: the main loop ends the session.
pub static REQUESTED: AtomicBool = AtomicBool::new(false);

/// The title menu's choice that quits (`SetUIToGameValue 'FrontEnd_MainMenu_Selection',
/// 'QuitGame'`): the hash the title compares (FNV-1a of the upper-cased name).
fn quit_choice() -> u32 {
    static H: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *H.get_or_init(|| crate::pcsettings::setting_hash("QuitGame"))
}

/// The title menu's dispatch on a choice (`titlefix`, eboot+0x7c9d70, r12d the choice's hash).
pub extern "sysv64" fn menu_hook(regs: *const u64, _arg: u32) {
    // SAFETY: the hook passes the 16 general registers in encoding order (r12 the 13th).
    let choice = unsafe { *regs.add(12) } as u32;
    if choice == quit_choice() && !REQUESTED.swap(true, Ordering::SeqCst) {
        crate::info!("quit: Exit Game chosen on the title menu");
    }
}

/// The design height of Save and Quit to Desktop (the game's Save and Exit Game: 50).
pub const BUTTON_HEIGHT: f32 = 51.0;

/// When Save and Quit to Desktop was pressed: the session ends at the title within a minute (the
/// game's save and exit takes about a second here; a press the game did not act on lapses).
static PENDING: Mutex<Option<Instant>> = Mutex::new(None);
static PENDING_ON: AtomicBool = AtomicBool::new(false);
const PENDING_FOR: Duration = Duration::from_secs(60);

/// Each vblank while a menu is open (`pointer::tick`, under the UI nodes' guard): Cross on Save
/// and Quit to Desktop.
pub fn tick(focused: &[u64]) {
    static LAST_CROSS: AtomicU32 = AtomicU32::new(0);
    static LAST_TICK: Mutex<Option<Instant>> = Mutex::new(None);
    let presses = crate::input::cross_presses();
    let pressed = LAST_CROSS.swap(presses, Ordering::Relaxed) != presses;
    // (A press counted while no menu was open, the one that opened this one too, is not this
    // menu's: the first tick after a pause only takes the count.)
    let fresh = LAST_TICK.lock().replace(Instant::now()).is_some_and(|t| t.elapsed() < Duration::from_millis(200));
    if !pressed || !fresh {
        return;
    }
    if focused.iter().any(|&n| crate::uistate::definition_size(n).is_some_and(|(_, h)| h == BUTTON_HEIGHT)) {
        *PENDING.lock() = Some(Instant::now());
        PENDING_ON.store(true, Ordering::SeqCst);
        crate::info!("quit: Save and Quit to Desktop chosen: the game saves and leaves for the title, then the session ends");
    }
}

/// A file the title opened (`hle::kernel::fs`): with Save and Quit to Desktop pending, the
/// title's own movies mean the game left the game world, its save written: the session ends.
pub fn note_open(path: &str) {
    if !PENDING_ON.load(Ordering::Relaxed) {
        return;
    }
    if !["/movies/attract_movie.bk2", "/movies/logo_intro.bk2", "/movies/main_menu.bk2"].iter().any(|m| path.ends_with(m)) {
        return;
    }
    let since = PENDING.lock().take();
    PENDING_ON.store(false, Ordering::SeqCst);
    match since {
        Some(t) if t.elapsed() <= PENDING_FOR => {
            if !REQUESTED.swap(true, Ordering::SeqCst) {
                crate::info!("quit: the title is back {:.1} s after Save and Quit to Desktop ({path}): the session ends", t.elapsed().as_secs_f64());
            }
        }
        Some(t) => crate::info!("quit: Save and Quit to Desktop had lapsed (the title came back {:.0} s after it; the game goes on)", t.elapsed().as_secs_f64()),
        None => {}
    }
}

/// Before the session ends at the player's word (the title's Exit Game, or the window closed):
/// waits while the game has a save open for writing, and until none has been for half a second
/// (a save the game starts right after a settings screen closes), at most ten seconds.
pub fn wait_for_saves() {
    let start = Instant::now();
    let mut quiet_since: Option<Instant> = None;
    let mut waited = false;
    loop {
        let open = crate::hle::libs::savedata::writes_open();
        if open > 0 {
            quiet_since = None;
            waited = true;
        } else {
            let q = *quiet_since.get_or_insert_with(Instant::now);
            if q.elapsed() >= Duration::from_millis(500) {
                break;
            }
        }
        if start.elapsed() >= Duration::from_secs(10) {
            if open > 0 {
                crate::warn!("quit: {open} save(s) still open for writing after 10 s; the session ends anyway");
            }
            return;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    if waited {
        crate::info!("quit: waited {} ms for the game's save to be written", start.elapsed().as_millis());
    }
}
