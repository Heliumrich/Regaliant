//! The achievement pop-up, in the title's own HUD: a copy of its area-name caption (the frame
//! that names an area as the player enters it: its font, colours and layout) added to the
//! gauges' HUD scene (`hud`). It shows while `Setting_PC_TrophyPopup` is on (a copy of the
//! subtitles' condition on their setting, answered at the title's `UISetting_bool::Evaluate`)
//! and names the trophy through `Setting_PC_TrophyPopupName`, an int the title takes as a text
//! key's hash (as the caption takes the area's name), answered with the hash of the trophy's
//! name key (`PC_TROPHY_NAME_<id>`, in the title's text with the PC settings' lines). Each new
//! unlock shows for SHOW, one after another.

use parking_lot::Mutex;
use std::time::{Duration, Instant};

pub const SETTING: &str = "Setting_PC_TrophyPopup";
pub const NAME_SETTING: &str = "Setting_PC_TrophyPopupName";
/// The caption's second line takes its text hash from an int setting too (its static key is
/// not used while a binding is set): answered with the hash of TITLE_KEY.
pub const TITLE_SETTING: &str = "Setting_PC_TrophyPopupTitle";
pub const TITLE_KEY: &str = "PC_TROPHY_POPUP_TITLE";
const SHOW: Duration = Duration::from_millis(6500);
const GAP: Duration = Duration::from_millis(800);

/// The text key naming trophy `id` in the pop-up.
pub fn name_key(id: u32) -> String {
    format!("PC_TROPHY_NAME_{id:02}")
}

struct State {
    queue: Vec<u32>,
    /// The trophy shown and since when.
    current: Option<(u32, Instant)>,
    /// The last one shown (its name stays while the caption fades out).
    last: Option<u32>,
    ended: Option<Instant>,
}

static STATE: Mutex<State> = Mutex::new(State { queue: Vec::new(), current: None, last: None, ended: None });

/// Whether the pop-up shows now.
pub fn visible() -> bool {
    STATE.lock().current.is_some()
}

/// The hash of the shown trophy's name key (0: none yet).
pub fn name_hash() -> u32 {
    STATE.lock().last.map(|id| super::cgpr::fnv1a(name_key(id).as_bytes())).unwrap_or(0)
}

/// Takes new unlocks and steps the pop-up (the vblank thread, each tick).
pub fn tick() {
    let news = crate::achievements::take_news();
    let mut s = STATE.lock();
    s.queue.extend(news);
    let now = Instant::now();
    if let Some((_, since)) = s.current {
        if now.duration_since(since) >= SHOW {
            s.current = None;
            s.ended = Some(now);
        }
    }
    if s.current.is_none() && !s.queue.is_empty() && s.ended.is_none_or(|e| now.duration_since(e) >= GAP) {
        let id = s.queue.remove(0);
        crate::info!("achievements: pop-up for trophy {id}");
        s.current = Some((id, now));
        s.last = Some(id);
    }
}
