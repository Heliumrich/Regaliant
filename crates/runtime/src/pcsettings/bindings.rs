//! Key bindings in the title's menus: a row per action of the keyboard and mouse layout
//! (`input`), its value the input of the action's first binding, chosen among every input
//! (Left/Right, the pointer's clicks on the value) or captured: Cross on a row (Enter, E, a
//! click) shows "Press a key" and takes the next key, mouse button or wheel turn (with Shift,
//! Ctrl or Alt held when the layout has that combination; Esc cancels, 8 s too). A change
//! rebinds the action at once and writes local-data/controls.txt (an input another action had
//! swaps over). The rows sit in a copy of the title's HELP screen (a list that scrolls), opened
//! from the PC tab (`mod.rs`); the focused row is found in the title's UI nodes (`uistate`): the
//! list of ACTIONS.len() rows whose definitions are ROW_WIDTH wide.

use crate::input::{self, PromptInput};
use parking_lot::Mutex;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

/// The binding rows' design width (`mod.rs` gives their Element this width).
pub const ROW_WIDTH: f32 = 520.0;

/// The actions (layout targets) in the order of the list, with their names.
pub const ACTIONS: &[(&str, &str)] = &[
    ("ls_up", "Move Forward"),
    ("ls_down", "Move Back"),
    ("ls_left", "Move Left"),
    ("ls_right", "Move Right"),
    ("walk", "Walk"),
    ("circle", "Roll, Dash"),
    ("r1", "Attack"),
    ("r2", "Strong Attack"),
    ("l1", "Guard"),
    ("l2", "Parry"),
    ("triangle", "Two-Handed"),
    ("square", "Use Item"),
    ("cross", "Interact"),
    ("r3", "Lock On"),
    ("touchpad", "Tool Belt"),
    ("options", "Menu"),
    ("up", "Next Spell"),
    ("down", "Next Item"),
    ("right", "Right Weapon"),
    ("left", "Left Weapon"),
    ("rs_up", "Camera Up"),
    ("rs_down", "Camera Down"),
    ("rs_left", "Camera Left"),
    ("rs_right", "Camera Right"),
];

/// The inputs a binding can take, the choice's options: none, every key, the mouse buttons, the
/// wheel, and the layout's modifier combinations when the game started.
pub fn inputs() -> &'static [Option<(PromptInput, u8)>] {
    static V: OnceLock<Vec<Option<(PromptInput, u8)>>> = OnceLock::new();
    V.get_or_init(|| {
        let mut v = vec![None];
        v.extend(input::bindable_keys().into_iter().map(|k| Some((PromptInput::Key(k), 0))));
        v.extend((0..5).map(|m| Some((PromptInput::Mouse(m), 0))));
        v.push(Some((PromptInput::WheelUp, 0)));
        v.push(Some((PromptInput::WheelDown, 0)));
        for (i, mods) in input::bound_inputs() {
            if mods != 0 && !v.contains(&Some((i, mods))) {
                v.push(Some((i, mods)));
            }
        }
        v
    })
}

fn key_label(k: u8) -> String {
    match k {
        b'A'..=b'Z' | b'0'..=b'9' => (k as char).to_string(),
        0x70..=0x7B => format!("F{}", k - 0x6F),
        0x20 => "Space".into(),
        0x0D => "Enter".into(),
        0x1B => "Esc".into(),
        0x09 => "Tab".into(),
        0x08 => "Backspace".into(),
        0x26 => "Up".into(),
        0x28 => "Down".into(),
        0x25 => "Left".into(),
        0x27 => "Right".into(),
        0x10 => "Shift".into(),
        0xA0 => "L Shift".into(),
        0xA1 => "R Shift".into(),
        0x11 => "Ctrl".into(),
        0xA2 => "L Ctrl".into(),
        0xA3 => "R Ctrl".into(),
        0x12 => "Alt".into(),
        0xA4 => "L Alt".into(),
        0xA5 => "R Alt".into(),
        0x14 => "Caps Lock".into(),
        0x2D => "Insert".into(),
        0x2E => "Delete".into(),
        0x24 => "Home".into(),
        0x23 => "End".into(),
        0x21 => "Page Up".into(),
        0x22 => "Page Down".into(),
        _ => format!("Key {k:#04x}"),
    }
}

/// An input's name in the menu.
pub fn label(i: Option<(PromptInput, u8)>) -> String {
    let Some((i, mods)) = i else { return "-".into() };
    let mut s = String::new();
    for (bit, name) in [(input::PROMPT_CTRL, "Ctrl+"), (input::PROMPT_ALT, "Alt+"), (input::PROMPT_SHIFT, "Shift+")] {
        if mods & bit != 0 {
            s += name;
        }
    }
    s += &match i {
        PromptInput::Key(k) => key_label(k),
        PromptInput::Mouse(0) => "Left Mouse".into(),
        PromptInput::Mouse(1) => "Right Mouse".into(),
        PromptInput::Mouse(2) => "Middle Mouse".into(),
        PromptInput::Mouse(m) => format!("Mouse {}", m + 1),
        PromptInput::WheelUp => "Wheel Up".into(),
        PromptInput::WheelDown => "Wheel Down".into(),
    };
    s
}

/// The option shown while a row waits for its key ("Press a key"): one past `inputs`.
pub fn capture_option() -> u32 {
    inputs().len() as u32
}

/// The option shown for action `a`: its first binding's place among `inputs`.
pub fn index_of(a: usize) -> u32 {
    if CAPTURE.lock().is_some_and(|(c, _)| c == a) {
        return capture_option();
    }
    let Some(&(target, _)) = ACTIONS.get(a) else { return 0 };
    let cur = input::binding_of(target);
    inputs().iter().position(|i| *i == cur).unwrap_or(0) as u32
}

/// Action `a` set to option `idx` (from the menu: Left/Right; "Press a key" is not chosen so).
pub fn set(a: usize, idx: u32) {
    if idx >= capture_option() {
        return;
    }
    let (Some(&(target, _)), Some(&i)) = (ACTIONS.get(a), inputs().get(idx as usize)) else { return };
    if input::binding_of(target) == i {
        return;
    }
    if let Err(e) = input::rebind(target, i) {
        crate::warn!("bindings: {e}");
    }
}

// ---------------------------------------------------------------------------
// Capture: the next key, mouse button or wheel turn for a row

/// The action waiting for its input, and since when.
static CAPTURE: Mutex<Option<(usize, Instant)>> = Mutex::new(None);
const CAPTURE_TIMEOUT: Duration = Duration::from_secs(8);

/// A binding is being captured (the keyboard and mouse do not reach the game meanwhile).
pub fn capturing() -> bool {
    let mut g = CAPTURE.lock();
    if g.is_some_and(|(_, t)| t.elapsed() > CAPTURE_TIMEOUT) {
        crate::info!("bindings: no key pressed: the binding is kept");
        *g = None;
    }
    g.is_some()
}

fn finish(input: Option<(PromptInput, u8)>) {
    let Some((a, _)) = CAPTURE.lock().take() else { return };
    let Some(input) = input else {
        crate::info!("bindings: capture cancelled");
        return;
    };
    // The combination when the layout has it as an option, else the input alone.
    let chosen = if inputs().contains(&Some(input)) { input } else { (input.0, 0) };
    if let Some(&(target, _)) = ACTIONS.get(a) {
        if let Err(e) = input::rebind(target, Some(chosen)) {
            crate::warn!("bindings: {e}");
        }
    }
}

fn held_mods() -> u8 {
    use crate::gpu::vk::window::key_down;
    (if key_down(0x10) { input::PROMPT_SHIFT } else { 0 }) | (if key_down(0x11) { input::PROMPT_CTRL } else { 0 }) | (if key_down(0x12) { input::PROMPT_ALT } else { 0 })
}

/// The window's key press (`vk`, its sided code for Shift, Ctrl and Alt): taken when a binding
/// waits (Esc cancels; a modifier alone waits for its key, unless released alone: `key_up`).
pub fn key_down(vk: u8, sided: Option<u8>) -> bool {
    if !capturing() {
        return false;
    }
    match vk {
        0x1B => finish(None),
        0x10 | 0x11 | 0x12 => {}
        _ => finish(Some((PromptInput::Key(vk), held_mods()))),
    }
    let _ = sided;
    true
}

/// A modifier released without another key: it is the binding (Shift alone, sided).
pub fn key_up(vk: u8, sided: Option<u8>) -> bool {
    if !capturing() {
        return false;
    }
    if matches!(vk, 0x10 | 0x11 | 0x12) {
        let k = sided.unwrap_or(vk);
        finish(Some((PromptInput::Key(k), 0)));
    }
    true
}

/// The window's mouse button (0 left .. 4) or wheel turn (`Wheel(+1/-1)`): taken when a binding
/// waits.
pub fn mouse_down(button: Option<u8>, wheel: i32) -> bool {
    if !capturing() {
        return false;
    }
    let mods = held_mods();
    match (button, wheel) {
        (Some(b), _) => finish(Some((PromptInput::Mouse(b), mods))),
        (None, w) if w > 0 => finish(Some((PromptInput::WheelUp, mods))),
        (None, w) if w < 0 => finish(Some((PromptInput::WheelDown, mods))),
        _ => {}
    }
    true
}

/// Tests (`bindinput <name>`): an input for a waiting binding, as if pressed in the window.
pub fn test_input(name: &str) -> Result<(), String> {
    let (i, mods) = input::parse_prompt_input(name).ok_or_else(|| format!("unknown input {name}"))?;
    if !capturing() {
        return Err("no binding waits for a key".into());
    }
    // Esc cancels, as in the window.
    finish(if i == PromptInput::Key(0x1B) && mods == 0 { None } else { Some((i, mods)) });
    Ok(())
}

/// The binding row focused now, if the bindings list has the focus (its index in ACTIONS).
pub fn focused_row(tree: &crate::uistate::Tree, focused: &[u64]) -> Option<usize> {
    use crate::uistate;
    for &n in focused {
        let p = uistate::parent(n);
        let Some(rows) = tree.children.get(&p) else { continue };
        if rows.len() != ACTIONS.len() || !rows.iter().all(|&r| uistate::definition_size(r).is_some_and(|(w, _)| w == ROW_WIDTH)) {
            continue;
        }
        let mut order: Vec<(f32, u64)> = rows.iter().map(|&r| (uistate::local(r).1, r)).collect();
        order.sort_by(|a, b| a.0.total_cmp(&b.0));
        return order.iter().position(|&(_, r)| r == n);
    }
    None
}

/// Each vblank while a menu is open (`pointer::tick`): Cross on a focused binding row starts the
/// capture.
pub fn tick(tree: &crate::uistate::Tree, focused: &[u64]) {
    static LAST_CROSS: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let presses = input::cross_presses();
    let pressed = LAST_CROSS.swap(presses, std::sync::atomic::Ordering::Relaxed) != presses;
    if !pressed || capturing() {
        return;
    }
    if let Some(a) = focused_row(tree, focused) {
        crate::info!("bindings: {} waits for a key", ACTIONS[a].1);
        *CAPTURE.lock() = Some((a, Instant::now()));
    }
}
