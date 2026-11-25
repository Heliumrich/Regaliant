//! Key and controller bindings in the title's menus: a row per action of the keyboard and mouse
//! layout (`input`), its value the input of the action's first binding, chosen among every input
//! (Left/Right, the pointer's clicks on the value) or captured: Cross on a row (Enter, E, a
//! click) shows "Press a key" and takes the next key, mouse button or wheel turn (with Shift,
//! Ctrl or Alt held when the layout has that combination; Esc cancels, 8 s too). A change
//! rebinds the action at once and writes local-data/controls.txt (an input another action had
//! swaps over). The controller's list has a row per pad button target (`PAD_ACTIONS`), its
//! value the pad button on it; Cross on a row shows "Press a button" and takes the next pad
//! button (Esc cancels; a pad button cancels a key's capture); the targets of two buttons swap,
//! so every target keeps a button. The rows sit in copies of the title's HELP screen (a list
//! that scrolls), opened from the PC tab (`mod.rs`); the focused row is found in the title's UI
//! nodes (`uistate`): the list of ACTIONS.len() (PAD_ACTIONS.len()) rows whose definitions are
//! ROW_WIDTH wide.

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
    // (The touch pad's left half, then its right: the game's Left and Right TouchPad Mode set
    // what each opens, the Tool Belt for both by default; a key reaches the right half only here.
    // The controller's list keeps one row: its 16 targets sit on 16 buttons, and a PlayStation
    // controller's finger picks the half itself.)
    ("touchpad", "Tool Belt"),
    ("touchpad_r", "Touch Pad Right"),
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

/// The controller's actions (pad button targets) in the order of its list, with their names.
pub const PAD_ACTIONS: &[(&str, &str)] = &[
    ("cross", "Interact"),
    ("circle", "Roll, Dash"),
    ("square", "Use Item"),
    ("triangle", "Two-Handed"),
    ("r1", "Attack"),
    ("r2", "Strong Attack"),
    ("l1", "Guard"),
    ("l2", "Parry"),
    ("r3", "Lock On"),
    ("l3", "Left Stick Press"),
    ("touchpad", "Tool Belt"),
    ("options", "Menu"),
    ("up", "Next Spell"),
    ("down", "Next Item"),
    ("right", "Right Weapon"),
    ("left", "Left Weapon"),
];

/// Which list a row is in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum List {
    Keys,
    Pad,
}

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

/// The controller rows' options: the pad buttons (`input::PAD_BUTTONS`), then "-" (a target no
/// button has: a file edited so) and "Press a button".
pub fn pad_options() -> Vec<String> {
    // (A PlayStation controller in use: its buttons' names, `input::pad_label`.)
    (0..input::PAD_BUTTONS.len()).map(|i| input::pad_label(i).to_string()).chain(["-".to_string(), "Press a button".to_string()]).collect()
}

/// The controller rows' "-" and "Press a button" options.
pub fn pad_none_option() -> u32 {
    input::PAD_BUTTONS.len() as u32
}
pub fn pad_capture_option() -> u32 {
    input::PAD_BUTTONS.len() as u32 + 1
}

fn target_bit(target: &str) -> u32 {
    input::button_bit(target).unwrap_or(0)
}

/// The option shown for controller action `a`: the pad button on its target.
pub fn pad_index_of(a: usize) -> u32 {
    if CAPTURE.lock().as_ref().is_some_and(|c| c.list == List::Pad && c.row == a) {
        return pad_capture_option();
    }
    let Some(&(target, _)) = PAD_ACTIONS.get(a) else { return 0 };
    input::pad_button_of(target_bit(target)).map(|b| b as u32).unwrap_or(pad_none_option())
}

/// Controller action `a` set to option `idx` (Left/Right: a pad button; "-" and "Press a
/// button" are not chosen so).
pub fn pad_set(a: usize, idx: u32) {
    if idx >= pad_none_option() {
        return;
    }
    let Some(&(target, _)) = PAD_ACTIONS.get(a) else { return };
    if input::pad_target(idx as usize) == target_bit(target) {
        return;
    }
    if let Err(e) = input::rebind_pad(target, idx as usize) {
        crate::warn!("bindings: {e}");
    }
}

/// The option shown for action `a`: its first binding's place among `inputs`.
pub fn index_of(a: usize) -> u32 {
    if CAPTURE.lock().as_ref().is_some_and(|c| c.list == List::Keys && c.row == a) {
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
// Capture: the next key, mouse button or wheel turn (a pad button) for a row

/// A row waiting for its input.
struct Capture {
    list: List,
    row: usize,
    since: Instant,
    /// Pad buttons held since the capture started, not taken until released (None: not polled yet).
    held: Option<u32>,
}

static CAPTURE: Mutex<Option<Capture>> = Mutex::new(None);
const CAPTURE_TIMEOUT: Duration = Duration::from_secs(8);

/// A binding is being captured (the keyboard, the mouse and the pad do not reach the game meanwhile).
pub fn capturing() -> bool {
    capturing_list().is_some()
}

fn capturing_list() -> Option<List> {
    let mut g = CAPTURE.lock();
    if g.as_ref().is_some_and(|c| c.since.elapsed() > CAPTURE_TIMEOUT) {
        crate::info!("bindings: nothing pressed: the binding is kept");
        *g = None;
    }
    g.as_ref().map(|c| c.list)
}

/// The pad's buttons held now (bit i: `input::PAD_BUTTONS[i]`), each pad poll: true while a
/// binding is captured (the pad is held back from the game). A controller row takes the first
/// button pressed after the capture started; a key's row is cancelled by one.
pub fn pad_input(phys: u32) -> bool {
    let mut g = CAPTURE.lock();
    let Some(c) = g.as_mut() else { return false };
    let held = *c.held.get_or_insert(phys);
    // Buttons held at the start count once released.
    let held = held & phys;
    c.held = Some(held);
    let new = phys & !held;
    if new == 0 {
        return true;
    }
    let (list, row) = (c.list, c.row);
    *g = None;
    drop(g);
    // What is held now reaches the game only once released.
    input::swallow_pad(phys);
    match list {
        List::Keys => crate::info!("bindings: a controller button cancelled the key's capture"),
        List::Pad => {
            let button = new.trailing_zeros() as usize;
            if let Some(&(target, _)) = PAD_ACTIONS.get(row) {
                if let Err(e) = input::rebind_pad(target, button) {
                    crate::warn!("bindings: {e}");
                }
            }
        }
    }
    true
}

fn finish(input: Option<(PromptInput, u8)>) {
    let Some(c) = CAPTURE.lock().take() else { return };
    let a = c.row;
    let Some(input) = input else {
        crate::info!("bindings: capture cancelled");
        return;
    };
    if c.list == List::Pad {
        return;
    }
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
    let Some(list) = capturing_list() else { return false };
    match vk {
        0x1B => finish(None),
        _ if list == List::Pad => {}
        0x10 | 0x11 | 0x12 => {}
        _ => finish(Some((PromptInput::Key(vk), held_mods()))),
    }
    let _ = sided;
    true
}

/// A modifier released without another key: it is the binding (Shift alone, sided).
pub fn key_up(vk: u8, sided: Option<u8>) -> bool {
    let Some(list) = capturing_list() else { return false };
    if list == List::Keys && matches!(vk, 0x10 | 0x11 | 0x12) {
        let k = sided.unwrap_or(vk);
        finish(Some((PromptInput::Key(k), 0)));
    }
    true
}

/// The window's mouse button (0 left .. 4) or wheel turn (`Wheel(+1/-1)`): taken when a binding
/// waits.
pub fn mouse_down(button: Option<u8>, wheel: i32) -> bool {
    let Some(list) = capturing_list() else { return false };
    if list == List::Pad {
        return true;
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

/// Tests (`bindinput <name>`): an input for a waiting binding, as if pressed in the window (a
/// pad button by its layout file name, `PadY`: as if pressed on the pad).
pub fn test_input(name: &str) -> Result<(), String> {
    if let Some(b) = input::PAD_BUTTONS.iter().position(|(n, _, _)| n.eq_ignore_ascii_case(name)) {
        if !capturing() {
            return Err("no binding waits".into());
        }
        // A poll with nothing held (the start's buttons released), then the button.
        pad_input(0);
        pad_input(1 << b);
        return Ok(());
    }
    let (i, mods) = input::parse_prompt_input(name).ok_or_else(|| format!("unknown input {name}"))?;
    if !capturing() {
        return Err("no binding waits for a key".into());
    }
    // Esc cancels, as in the window.
    finish(if i == PromptInput::Key(0x1B) && mods == 0 { None } else { Some((i, mods)) });
    Ok(())
}

/// The binding row focused now, if a bindings list has the focus (its list and its index in
/// ACTIONS or PAD_ACTIONS).
pub fn focused_row(tree: &crate::uistate::Tree, focused: &[u64]) -> Option<(List, usize)> {
    use crate::uistate;
    for &n in focused {
        let p = uistate::parent(n);
        let Some(rows) = tree.children.get(&p) else { continue };
        let list = match rows.len() {
            l if l == ACTIONS.len() => List::Keys,
            l if l == PAD_ACTIONS.len() => List::Pad,
            _ => continue,
        };
        if !rows.iter().all(|&r| uistate::definition_size(r).is_some_and(|(w, _)| w == ROW_WIDTH)) {
            continue;
        }
        let mut order: Vec<(f32, u64)> = rows.iter().map(|&r| (uistate::local(r).1, r)).collect();
        order.sort_by(|a, b| a.0.total_cmp(&b.0));
        return order.iter().position(|&(_, r)| r == n).map(|i| (list, i));
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
    if let Some((list, a)) = focused_row(tree, focused) {
        match list {
            List::Keys => crate::info!("bindings: {} waits for a key", ACTIONS[a].1),
            List::Pad => crate::info!("bindings: {} waits for a controller button", PAD_ACTIONS[a].1),
        }
        *CAPTURE.lock() = Some(Capture { list, row: a, since: Instant::now(), held: None });
    }
}
