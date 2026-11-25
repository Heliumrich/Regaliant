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

use std::sync::atomic::{AtomicBool, Ordering};
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
            crate::warn!("quit: a save was still open for writing after 10 s ({open}); ending anyway (the save committed last stays)");
            return;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    if waited {
        crate::info!("quit: waited {} ms for the game's save to be written", start.elapsed().as_millis());
    }
}
