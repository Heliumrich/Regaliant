//! Host input aggregation for the PS5 pad: an XInput controller (its own
//! fixed mapping), the output window's keyboard and mouse in a Dark Souls
//! III-style PC layout, and a scripted timeline for reproducible runs.
//!
//! Keyboard and mouse (`Layout`, window focused): the layout is read from
//! local-data/controls.txt, written with the defaults (`DEFAULT_LAYOUT`) on
//! first start; each line binds a key, mouse button or wheel direction
//! (optionally with Shift+, Ctrl+ or Alt+) to a pad button, a stick
//! direction or `walk`. A binding with modifiers wins over the plain one of
//! the same input while its modifiers are held (Shift+Mouse1 = R2 instead of
//! Mouse1 = R1). Wheel notches are d-pad taps. Mouse motion (raw counts,
//! `window`) turns the camera through the right stick: its speed over the
//! interval since the last pad read sets the deflection (`mouse_look`).
//!
//! Scripted input (`DS_INPUT`): `flip:buttons[:hold]` entries separated by
//! `;`, e.g. `DS_INPUT=700:cross;760:down:3;800:cross`. `buttons` joins names
//! with `+` (cross, circle, square, triangle, up, down, left, right, l1, r1,
//! l2, r2, l3, r3, options, touchpad, ls_up/down/left/right,
//! rs_up/down/left/right). Each press lasts `hold` GPU flips (default 6).

use parking_lot::Mutex;
use std::collections::VecDeque;
use std::sync::OnceLock;
use std::sync::atomic::Ordering::Relaxed;
use std::time::{Duration, Instant};
use windows_sys::Win32::UI::Input::XboxController::{XINPUT_STATE, XInputGetState};

pub const BTN_L3: u32 = 0x2;
pub const BTN_R3: u32 = 0x4;
pub const BTN_OPTIONS: u32 = 0x8;
pub const BTN_UP: u32 = 0x10;
pub const BTN_RIGHT: u32 = 0x20;
pub const BTN_DOWN: u32 = 0x40;
pub const BTN_LEFT: u32 = 0x80;
pub const BTN_L2: u32 = 0x100;
pub const BTN_R2: u32 = 0x200;
pub const BTN_L1: u32 = 0x400;
pub const BTN_R1: u32 = 0x800;
pub const BTN_TRIANGLE: u32 = 0x1000;
pub const BTN_CIRCLE: u32 = 0x2000;
pub const BTN_CROSS: u32 = 0x4000;
pub const BTN_SQUARE: u32 = 0x8000;
pub const BTN_TOUCHPAD: u32 = 0x100000;

/// Pad state in DS4 terms (sticks 0..255, centre 0x80; y grows downwards).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PadState {
    pub buttons: u32,
    pub lx: u8,
    pub ly: u8,
    pub rx: u8,
    pub ry: u8,
    pub l2: u8,
    pub r2: u8,
}

impl Default for PadState {
    fn default() -> Self {
        PadState { buttons: 0, lx: 0x80, ly: 0x80, rx: 0x80, ry: 0x80, l2: 0, r2: 0 }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stick {
    LUp,
    LDown,
    LLeft,
    LRight,
    RUp,
    RDown,
    RLeft,
    RRight,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Step {
    flip: u64,
    hold: u64,
    buttons: u32,
    sticks: Vec<Stick>,
}

fn parse_name(n: &str) -> Option<Result<u32, Stick>> {
    Some(match n {
        "cross" | "x" => Ok(BTN_CROSS),
        "circle" | "o" => Ok(BTN_CIRCLE),
        "square" => Ok(BTN_SQUARE),
        "triangle" => Ok(BTN_TRIANGLE),
        "up" => Ok(BTN_UP),
        "down" => Ok(BTN_DOWN),
        "left" => Ok(BTN_LEFT),
        "right" => Ok(BTN_RIGHT),
        "l1" => Ok(BTN_L1),
        "r1" => Ok(BTN_R1),
        "l2" => Ok(BTN_L2),
        "r2" => Ok(BTN_R2),
        "l3" => Ok(BTN_L3),
        "r3" => Ok(BTN_R3),
        "options" | "start" => Ok(BTN_OPTIONS),
        "touchpad" => Ok(BTN_TOUCHPAD),
        "ls_up" => Err(Stick::LUp),
        "ls_down" => Err(Stick::LDown),
        "ls_left" => Err(Stick::LLeft),
        "ls_right" => Err(Stick::LRight),
        "rs_up" => Err(Stick::RUp),
        "rs_down" => Err(Stick::RDown),
        "rs_left" => Err(Stick::RLeft),
        "rs_right" => Err(Stick::RRight),
        _ => return None,
    })
}

fn parse_script(s: &str) -> Result<Vec<Step>, String> {
    let mut steps = Vec::new();
    for entry in s.split(';').map(str::trim).filter(|e| !e.is_empty()) {
        let parts: Vec<&str> = entry.split(':').collect();
        if parts.len() < 2 || parts.len() > 3 {
            return Err(format!("bad entry '{entry}' (want flip:buttons[:hold])"));
        }
        let flip = parts[0].trim().parse::<u64>().map_err(|_| format!("bad flip in '{entry}'"))?;
        let hold = match parts.get(2) {
            Some(h) => h.trim().parse::<u64>().map_err(|_| format!("bad hold in '{entry}'"))?.max(1),
            None => 6,
        };
        let mut step = Step { flip, hold, buttons: 0, sticks: Vec::new() };
        for name in parts[1].split('+').map(|n| n.trim().to_ascii_lowercase()) {
            match parse_name(&name) {
                Some(Ok(b)) => step.buttons |= b,
                Some(Err(st)) => step.sticks.push(st),
                None => return Err(format!("unknown button '{name}' in '{entry}'")),
            }
        }
        steps.push(step);
    }
    Ok(steps)
}

fn script() -> &'static [Step] {
    static SCRIPT: OnceLock<Vec<Step>> = OnceLock::new();
    SCRIPT.get_or_init(|| match std::env::var("DS_INPUT") {
        Ok(s) => match parse_script(&s) {
            Ok(v) => {
                crate::info!("input script: {} steps", v.len());
                v
            }
            Err(e) => {
                crate::error!("DS_INPUT ignored: {e}");
                Vec::new()
            }
        },
        Err(_) => Vec::new(),
    })
}

fn apply_stick(st: &mut PadState, s: Stick) {
    match s {
        Stick::LUp => st.ly = 0,
        Stick::LDown => st.ly = 255,
        Stick::LLeft => st.lx = 0,
        Stick::LRight => st.lx = 255,
        Stick::RUp => st.ry = 0,
        Stick::RDown => st.ry = 255,
        Stick::RLeft => st.rx = 0,
        Stick::RRight => st.rx = 255,
    }
}

/// Presses scheduled at run time (control channel), relative to the flip at
/// which they were requested.
static LIVE: parking_lot::Mutex<Vec<Step>> = parking_lot::Mutex::new(Vec::new());

/// Schedules a press of `names` (joined with '+') for `hold` flips starting
/// at the next flip.
pub fn press_now(names: &str, hold: u64) -> Result<(), String> {
    let flip = crate::gpu::FRAMES.load(std::sync::atomic::Ordering::Relaxed) + 1;
    let mut steps = parse_script(&format!("{flip}:{names}:{}", hold.max(1)))?;
    let mut live = LIVE.lock();
    live.retain(|s| s.flip + s.hold > flip);
    live.append(&mut steps);
    Ok(())
}

/// Presses held for a wall-clock duration (control channel `press <b> 4s`):
/// frame rates differ about sixfold between runs with and without GPU
/// execution, so holds counted in flips do not transfer between them.
static LIVE_TIMED: parking_lot::Mutex<Vec<(std::time::Instant, Step)>> = parking_lot::Mutex::new(Vec::new());

/// Presses `names` from now for `dur`.
pub fn press_for(names: &str, dur: std::time::Duration) -> Result<(), String> {
    let steps = parse_script(&format!("0:{names}:1"))?;
    let now = std::time::Instant::now();
    let mut live = LIVE_TIMED.lock();
    live.retain(|x| x.0 > now);
    live.extend(steps.into_iter().map(|s| (now + dur, s)));
    Ok(())
}

/// Sticks held at a tilt by the control channel (`hold_stick`): until when,
/// right stick, x, y.
static STICKS: parking_lot::Mutex<Vec<(std::time::Instant, bool, i32, i32)>> = parking_lot::Mutex::new(Vec::new());

/// Diagnosis (control channel `stick`): holds the left or right stick at
/// (x, y) from the centre (-128..127, y down) for `dur`.
pub fn hold_stick(right: bool, x: i32, y: i32, dur: std::time::Duration) {
    let now = std::time::Instant::now();
    let mut g = STICKS.lock();
    g.retain(|s| s.0 > now);
    g.push((now + dur, right, x.clamp(-128, 127), y.clamp(-128, 127)));
}

fn scripted(st: &mut PadState, flip: u64) {
    let now = std::time::Instant::now();
    for &(until, right, x, y) in STICKS.lock().iter() {
        if now < until {
            let (ax, ay) = ((128 + x) as u8, (128 + y) as u8);
            if right {
                (st.rx, st.ry) = (ax, ay);
            } else {
                (st.lx, st.ly) = (ax, ay);
            }
        }
    }
    for (until, step) in LIVE_TIMED.lock().iter() {
        if now < *until {
            st.buttons |= step.buttons;
            for &s in &step.sticks {
                apply_stick(st, s);
            }
        }
    }
    for step in LIVE.lock().iter() {
        if flip >= step.flip && flip < step.flip + step.hold {
            st.buttons |= step.buttons;
            for &s in &step.sticks {
                apply_stick(st, s);
            }
        }
    }
    static LOGGED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    for (i, step) in script().iter().enumerate() {
        if flip >= step.flip && flip < step.flip + step.hold {
            st.buttons |= step.buttons;
            for &s in &step.sticks {
                apply_stick(st, s);
            }
            if LOGGED.fetch_max(i + 1, std::sync::atomic::Ordering::Relaxed) <= i {
                crate::info!("input script: flip {flip} step {i} ({:#x} {:?}) for {} flips", step.buttons, step.sticks, step.hold);
            }
        }
    }
}

pub fn xinput_state() -> Option<XINPUT_STATE> {
    for i in 0..4 {
        let mut s: XINPUT_STATE = unsafe { std::mem::zeroed() };
        if unsafe { XInputGetState(i, &mut s) } == 0 {
            return Some(s);
        }
    }
    None
}

fn axis(v: i16, invert: bool) -> u8 {
    let f = (v as f32 + 32768.0) / 65535.0;
    let f = if invert { 1.0 - f } else { f };
    (f * 255.0).round().clamp(0.0, 255.0) as u8
}

fn xinput(st: &mut PadState) {
    let Some(s) = xinput_state() else { return };
    let g = s.Gamepad;
    let map = [
        (0x0001, BTN_UP),
        (0x0002, BTN_DOWN),
        (0x0004, BTN_LEFT),
        (0x0008, BTN_RIGHT),
        (0x0010, BTN_OPTIONS),  // Start
        (0x0020, BTN_TOUCHPAD), // Back
        (0x0040, BTN_L3),
        (0x0080, BTN_R3),
        (0x0100, BTN_L1),
        (0x0200, BTN_R1),
        (0x1000, BTN_CROSS),
        (0x2000, BTN_CIRCLE),
        (0x4000, BTN_SQUARE),
        (0x8000, BTN_TRIANGLE),
    ];
    for (x, p) in map {
        if g.wButtons & x != 0 {
            st.buttons |= p;
        }
    }
    st.l2 = st.l2.max(g.bLeftTrigger);
    st.r2 = st.r2.max(g.bRightTrigger);
    st.lx = axis(g.sThumbLX, false);
    st.ly = axis(g.sThumbLY, true);
    st.rx = axis(g.sThumbRX, false);
    st.ry = axis(g.sThumbRY, true);
}

// ---------------------------------------------------------------------------
// Keyboard and mouse

/// The default keyboard and mouse layout, also written to controls.txt on
/// first start: Dark Souls III's PC defaults on the PS5 pad's buttons.
pub const DEFAULT_LAYOUT: &str = "\
# Keyboard and mouse layout of the Demon's Souls port (Dark Souls III-style PC
# controls). An Xbox-style controller keeps its own mapping. Edit and restart;
# delete this file to get the defaults back.
#
# Lines: <input> = <pad target>. Inputs: key names (A..Z, 0..9, Space, Enter,
# Esc, Tab, Backspace, Up, Down, Left, Right, Shift, LShift, RShift, Ctrl,
# LCtrl, RCtrl, Alt, LAlt, RAlt, F2..F12, Insert, Delete, Home, End, PageUp,
# PageDown, CapsLock), mouse buttons (Mouse1 left, Mouse2 right, Mouse3
# middle, Mouse4 and Mouse5 the side buttons), WheelUp, WheelDown; each may
# have Shift+, Ctrl+ or Alt+ in front (that binding then wins while the
# modifier is held). Targets: cross circle square triangle l1 r1 l2 r2 l3 r3
# options touchpad, up down left right (d-pad), ls_up ls_down ls_left
# ls_right (left stick), rs_up rs_down rs_left rs_right (right stick, the
# camera), walk (the left stick at walking tilt).
# F1 releases the mouse cursor or takes it back.

# Mouse look: 1 on, 0 off; raw (1: the mouse turns the camera by its counts,
# without acceleration, smoothing or a speed cap; 0: through the right stick,
# with the game's response curve and top speed); sensitivity (1.0: 2000 mouse
# counts turn the camera as far as a second of full right stick at the game's
# base camera speed); the vertical axis's own sensitivity (optional, the same
# as mouse_sensitivity when left out); invert the vertical axis (1).
mouse_look = 1
mouse_look_raw = 1
mouse_sensitivity = 1.0
# mouse_sensitivity_y = 1.0
mouse_invert_y = 0

W = ls_up
S = ls_down
A = ls_left
D = ls_right
LAlt = walk
Space = circle          # dash (hold), backstep, roll; back in menus
Mouse1 = r1             # attack (right hand)
Shift+Mouse1 = r2       # strong attack
Mouse2 = l1             # guard (left hand)
LCtrl = l2              # parry / left-hand strong attack
F = triangle            # two-hand
R = square              # use item
E = cross               # interact; confirm in menus
Q = r3                  # lock on / reset camera
Mouse3 = r3
G = touchpad            # tool belt
Esc = options           # menu
Up = up                 # switch spell
Down = down             # switch item
Right = right           # switch right-hand weapon
Left = left             # switch left-hand weapon
WheelUp = up
WheelDown = down
Shift+WheelUp = right
Shift+WheelDown = left
Shift+Left = l1         # menus: previous page
Shift+Right = r1        # menus: next page
Enter = cross
Backspace = circle
I = rs_up               # camera without a mouse
K = rs_down
J = rs_left
L = rs_right
";

/// Modifier bits of a binding.
const MOD_SHIFT: u8 = 1;
const MOD_CTRL: u8 = 2;
const MOD_ALT: u8 = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Src {
    Key(u8),
    /// Mouse button by bit (`window::MOUSE_BUTTONS`).
    Mouse(u8),
    WheelUp,
    WheelDown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Dst {
    Button(u32),
    Stick(Stick),
    Walk,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Bind {
    src: Src,
    mods: u8,
    dst: Dst,
}

#[derive(Clone, Debug)]
struct Layout {
    binds: Vec<Bind>,
    mouse_look: bool,
    sensitivity: f32,
    /// The vertical axis's own sensitivity (`mouse_sensitivity_y`; None: the
    /// same as `sensitivity`).
    sensitivity_y: Option<f32>,
    invert_y: bool,
    /// Mouse counts turn the gameplay camera directly (`camhook`), not
    /// through the stick (`mouse_look_raw`, default 1 since 2026-10-21, also
    /// for files written before; `DS_MOUSE_RAW` overrides).
    mouse_raw: bool,
}

/// A key name's virtual-key code.
fn key_code(name: &str) -> Option<u8> {
    let n = name.trim().to_ascii_lowercase();
    if n.len() == 1 {
        let c = n.as_bytes()[0];
        if c.is_ascii_lowercase() {
            return Some(c.to_ascii_uppercase());
        }
        if c.is_ascii_digit() {
            return Some(c);
        }
    }
    if let Some(f) = n.strip_prefix('f').and_then(|d| d.parse::<u8>().ok()) {
        return (1..=24).contains(&f).then_some(0x6F + f);
    }
    Some(match n.as_str() {
        "space" => 0x20,
        "enter" | "return" => 0x0D,
        "esc" | "escape" => 0x1B,
        "tab" => 0x09,
        "backspace" => 0x08,
        "up" => 0x26,
        "down" => 0x28,
        "left" => 0x25,
        "right" => 0x27,
        "shift" => 0x10,
        "lshift" => 0xA0,
        "rshift" => 0xA1,
        "ctrl" | "control" => 0x11,
        "lctrl" => 0xA2,
        "rctrl" => 0xA3,
        "alt" => 0x12,
        "lalt" => 0xA4,
        "ralt" => 0xA5,
        "capslock" => 0x14,
        "insert" => 0x2D,
        "delete" => 0x2E,
        "home" => 0x24,
        "end" => 0x23,
        "pageup" => 0x21,
        "pagedown" => 0x22,
        _ => return None,
    })
}

fn parse_src(s: &str) -> Option<(Src, u8)> {
    let mut mods = 0u8;
    let mut rest = s.trim();
    loop {
        let lower = rest.to_ascii_lowercase();
        let (m, len) = if lower.starts_with("shift+") {
            (MOD_SHIFT, 6)
        } else if lower.starts_with("ctrl+") {
            (MOD_CTRL, 5)
        } else if lower.starts_with("alt+") {
            (MOD_ALT, 4)
        } else {
            break;
        };
        mods |= m;
        rest = rest[len..].trim_start();
    }
    let src = match rest.to_ascii_lowercase().as_str() {
        "mouse1" => Src::Mouse(0),
        "mouse2" => Src::Mouse(1),
        "mouse3" => Src::Mouse(2),
        "mouse4" => Src::Mouse(3),
        "mouse5" => Src::Mouse(4),
        "wheelup" => Src::WheelUp,
        "wheeldown" => Src::WheelDown,
        _ => Src::Key(key_code(rest)?),
    };
    Some((src, mods))
}

fn parse_dst(s: &str) -> Option<Dst> {
    let n = s.trim().to_ascii_lowercase();
    if n == "walk" {
        return Some(Dst::Walk);
    }
    match parse_name(&n)? {
        Ok(b) => Some(Dst::Button(b)),
        Err(st) => Some(Dst::Stick(st)),
    }
}

fn parse_layout(text: &str) -> Result<Layout, String> {
    let mut l = Layout { binds: Vec::new(), mouse_look: true, sensitivity: 1.0, sensitivity_y: None, invert_y: false, mouse_raw: true };
    for (i, line) in text.lines().enumerate() {
        let line = line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let Some((a, b)) = line.split_once('=') else { return Err(format!("line {}: expected <input> = <target>", i + 1)) };
        let (a, b) = (a.trim(), b.trim());
        match a.to_ascii_lowercase().as_str() {
            "mouse_look" => l.mouse_look = b != "0",
            "mouse_invert_y" => l.invert_y = b != "0",
            "mouse_look_raw" => l.mouse_raw = b != "0",
            "mouse_sensitivity" => {
                l.sensitivity = b.parse::<f32>().ok().filter(|v| v.is_finite() && *v > 0.0).ok_or_else(|| format!("line {}: bad sensitivity '{b}'", i + 1))?;
            }
            "mouse_sensitivity_y" => {
                l.sensitivity_y = Some(b.parse::<f32>().ok().filter(|v| v.is_finite() && *v > 0.0).ok_or_else(|| format!("line {}: bad sensitivity '{b}'", i + 1))?);
            }
            _ => {
                let (src, mods) = parse_src(a).ok_or_else(|| format!("line {}: unknown input '{a}'", i + 1))?;
                let dst = parse_dst(b).ok_or_else(|| format!("line {}: unknown target '{b}'", i + 1))?;
                l.binds.push(Bind { src, mods, dst });
            }
        }
    }
    Ok(l)
}

static LAYOUT: OnceLock<Layout> = OnceLock::new();

fn layout() -> &'static Layout {
    LAYOUT.get_or_init(|| parse_layout(DEFAULT_LAYOUT).expect("default layout"))
}

/// Reads the keyboard and mouse layout from `path` (written with the
/// defaults when missing; the defaults on errors, logged).
pub fn init(path: &std::path::Path) {
    if !path.exists() {
        if let Err(e) = std::fs::write(path, DEFAULT_LAYOUT) {
            crate::warn!("input: {} not written ({e})", path.display());
        }
    }
    let mut l = match std::fs::read_to_string(path) {
        Ok(text) => match parse_layout(&text) {
            Ok(l) => {
                crate::info!("input: keyboard and mouse layout from {} ({} bindings; mouse look {}{}, sensitivity {}{}, invert y {})", path.display(), l.binds.len(), l.mouse_look, if l.mouse_raw { " (raw)" } else { "" }, l.sensitivity, l.sensitivity_y.map(|v| format!(" (vertical {v})")).unwrap_or_default(), l.invert_y);
                l
            }
            Err(e) => {
                crate::warn!("input: {}: {e}; the default layout is used", path.display());
                parse_layout(DEFAULT_LAYOUT).expect("default layout")
            }
        },
        Err(_) => parse_layout(DEFAULT_LAYOUT).expect("default layout"),
    };
    if let Ok(v) = std::env::var("DS_MOUSE_RAW") {
        l.mouse_raw = v != "0";
        crate::info!("input: raw mouse look {} (DS_MOUSE_RAW)", if l.mouse_raw { "on" } else { "off" });
    }
    let _ = LAYOUT.set(l);
}

/// Keys and mouse buttons held by the control channel (`inject_hold`):
/// until when.
static INJECTED: Mutex<Vec<(Instant, Src)>> = Mutex::new(Vec::new());

/// Diagnosis (control channel `key`): holds a key or mouse button (a layout
/// input name without modifiers) for `dur`, as if pressed in the window.
pub fn inject_hold(name: &str, dur: Duration) -> Result<(), String> {
    let (src, mods) = parse_src(name).ok_or_else(|| format!("unknown input '{name}'"))?;
    if mods != 0 || matches!(src, Src::WheelUp | Src::WheelDown) {
        return Err(format!("'{name}': a key or mouse button without modifiers"));
    }
    let now = Instant::now();
    let mut g = INJECTED.lock();
    g.retain(|x| x.0 > now);
    g.push((now + dur, src));
    Ok(())
}

/// Diagnosis (control channel `mouse`, `wheel`): mouse motion in counts and
/// wheel notches (up positive), as if reported by the window.
pub fn inject_motion(dx: i64, dy: i64, wheel: i32) {
    use crate::gpu::vk::window::{MOUSE_DX, MOUSE_DY, MOUSE_WHEEL};
    MOUSE_DX.fetch_add(dx, Relaxed);
    MOUSE_DY.fetch_add(dy, Relaxed);
    MOUSE_WHEEL.fetch_add(wheel * 120, Relaxed);
}

/// Whether `src` is held: in the window or by the control channel (a sided
/// modifier held there also holds the common one).
fn held(src: Src) -> bool {
    use crate::gpu::vk::window::{MOUSE_BUTTONS, key_down};
    let window = match src {
        Src::Key(k) => key_down(k),
        Src::Mouse(m) => MOUSE_BUTTONS.load(Relaxed) & (1 << m) != 0,
        Src::WheelUp | Src::WheelDown => false,
    };
    if window {
        return true;
    }
    let now = Instant::now();
    let g = INJECTED.lock();
    g.iter().any(|&(until, s)| {
        now < until
            && (s == src
                || matches!((s, src), (Src::Key(0xA0 | 0xA1), Src::Key(0x10)) | (Src::Key(0xA2 | 0xA3), Src::Key(0x11)) | (Src::Key(0xA4 | 0xA5), Src::Key(0x12))))
    })
}

/// The modifier bits held now.
fn held_mods() -> u8 {
    (if held(Src::Key(0x10)) { MOD_SHIFT } else { 0 }) | (if held(Src::Key(0x11)) { MOD_CTRL } else { 0 }) | (if held(Src::Key(0x12)) { MOD_ALT } else { 0 })
}

/// Of the bindings of `src`, those that apply with `held` modifiers: the
/// ones with the most modifiers, all held.
fn applying(l: &Layout, src: Src, held: u8) -> impl Iterator<Item = &Bind> {
    let best = l.binds.iter().filter(|b| b.src == src && b.mods & !held == 0).map(|b| b.mods.count_ones()).max();
    l.binds.iter().filter(move |b| b.src == src && b.mods & !held == 0 && Some(b.mods.count_ones()) == best)
}

/// Wheel taps not yet given to the title, and the one being given: its
/// button, whether it is down, pad reads and the flip of its phase's start.
/// A phase lasts at least two reads and one flip (the title may read the pad
/// more than once per frame).
static TAPS: Mutex<(VecDeque<u32>, Option<(u32, bool, u32, u64)>)> = Mutex::new((VecDeque::new(), None));

fn wheel_taps(st: &mut PadState, l: &Layout, held: u8) {
    use crate::gpu::vk::window::MOUSE_WHEEL;
    let mut g = TAPS.lock();
    let w = MOUSE_WHEEL.load(Relaxed);
    let notches = w / 120;
    if notches != 0 {
        MOUSE_WHEEL.fetch_sub(notches * 120, Relaxed);
        let src = if notches > 0 { Src::WheelUp } else { Src::WheelDown };
        let buttons: u32 = applying(l, src, held).filter_map(|b| if let Dst::Button(x) = b.dst { Some(x) } else { None }).fold(0, |a, x| a | x);
        for _ in 0..notches.unsigned_abs() {
            if buttons != 0 && g.0.len() < 8 {
                g.0.push_back(buttons);
            }
        }
    }
    let flip = crate::gpu::FRAMES.load(Relaxed);
    let (queue, cur) = &mut *g;
    loop {
        match cur {
            None => match queue.pop_front() {
                Some(b) => *cur = Some((b, true, 0, flip)),
                None => break,
            },
            Some((_, down, reads, since)) if *reads >= 2 && flip > *since => {
                if *down {
                    (*down, *reads, *since) = (false, 0, flip);
                    break;
                }
                *cur = None;
            }
            Some(_) => break,
        }
    }
    if let Some((b, down, reads, _)) = cur {
        if *down {
            st.buttons |= *b;
        }
        *reads += 1;
    }
}

/// Mouse look: the last deflection given and when the motion was last taken.
static LOOK: Mutex<(Option<Instant>, f32, f32)> = Mutex::new((None, 0.0, 0.0));

/// Mouse counts per second for a full right-stick deflection at sensitivity 1.
const FULL_SPEED: f32 = 2000.0;
/// Deflection added before mouse motion's own (a stick dead zone to step
/// over). None: the title's camera takes the raw stick, (rx - 128) / 127,
/// even at 1/127 (its pad layer applies the radial dead zone our
/// scePadGetControllerInformation reports, 13 of 127, only to values the
/// camera does not use; FINDINGS.md 2026-10-17).
const STICK_DEAD: f32 = 0.0;

/// Raw mouse look is chosen (`mouse_look_raw`, `DS_MOUSE_RAW`).
pub fn raw_mouse_look() -> bool {
    let l = layout();
    l.mouse_look && l.mouse_raw
}

/// Raw mouse look with the mouse captured.
fn raw_active() -> bool {
    use crate::gpu::vk::window::{CAPTURED, CAPTURE_TEST};
    raw_mouse_look() && (CAPTURED.load(Relaxed) || CAPTURE_TEST.load(Relaxed))
}

/// Mouse motion waits for the gameplay camera (`camhook`).
pub fn camera_motion_pending() -> bool {
    use crate::gpu::vk::window::{MOUSE_DX, MOUSE_DY};
    raw_active() && (MOUSE_DX.load(Relaxed) != 0 || MOUSE_DY.load(Relaxed) != 0)
}

/// Mouse motion for the gameplay camera (`camhook`), taken: None when raw
/// mouse look is off or the mouse is not captured (then nothing is taken).
pub fn take_camera_motion() -> Option<(i64, i64)> {
    use crate::gpu::vk::window::{MOUSE_DX, MOUSE_DY};
    if !raw_active() {
        return None;
    }
    let (dx, dy) = (MOUSE_DX.swap(0, Relaxed), MOUSE_DY.swap(0, Relaxed));
    Some((dx, if invert_y(layout()) { -dy } else { dy }))
}

/// The PC settings' mouse sensitivity (a factor on controls.txt's, f32 bits) and vertical
/// inversion (flipping controls.txt's), from the SETTINGS screen's PC tab: read at every use.
pub static PC_SENSITIVITY: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0x3f80_0000);
pub static PC_INVERT_Y: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn pc_sensitivity() -> f32 {
    let v = f32::from_bits(PC_SENSITIVITY.load(Relaxed));
    if v.is_finite() && v > 0.0 { v } else { 1.0 }
}

fn invert_y(l: &Layout) -> bool {
    l.invert_y ^ PC_INVERT_Y.load(Relaxed)
}

/// The camera's base yaw speed when the title's is unusable (radians per
/// second at full tilt).
const RAW_FALLBACK_YAW_SPEED: f32 = 2.2;

/// Radians of yaw and of pitch per mouse count (raw mouse look), the same on
/// both axes unless `mouse_sensitivity_y` sets the vertical one: FULL_SPEED
/// counts at sensitivity 1 turn as far as a second of
/// full stick at the camera's base yaw speed (`yaw_speed`, radians per
/// second), as mouse look through the stick does below full tilt. Signs as
/// the camera's angles turn for right and downward motion.
pub fn camera_radians_per_count(yaw_speed: f32) -> (f32, f32) {
    let speed = if yaw_speed.is_finite() && yaw_speed > 0.0 && yaw_speed < 100.0 { yaw_speed } else { RAW_FALLBACK_YAW_SPEED };
    let l = layout();
    let k = speed / FULL_SPEED;
    let pc = pc_sensitivity();
    (k * l.sensitivity * pc * RAW_YAW_SIGN, k * l.sensitivity_y.unwrap_or(l.sensitivity) * pc * RAW_PITCH_SIGN)
}

/// The camera's yaw and pitch rise for rightward and downward motion, as for
/// the stick tilted right and down (FINDINGS.md 2026-10-21 evening).
const RAW_YAW_SIGN: f32 = 1.0;
const RAW_PITCH_SIGN: f32 = 1.0;

fn mouse_look(st: &mut PadState, l: &Layout) {
    use crate::gpu::vk::window::{CAPTURED, CAPTURE_TEST, MOUSE_DX, MOUSE_DY};
    let mut g = LOOK.lock();
    // Raw mouse look while the gameplay camera runs: the hook takes the
    // motion (`take_camera_motion`), the stick stays.
    if !l.mouse_look || !(CAPTURED.load(Relaxed) || CAPTURE_TEST.load(Relaxed)) || (l.mouse_raw && crate::camhook::live()) {
        *g = (None, 0.0, 0.0);
        return;
    }
    let now = Instant::now();
    // The title may read the pad several times a frame: motion is taken
    // over at least 8 ms, reads in between give the last deflection.
    let due = g.0.is_none_or(|t| now.duration_since(t) >= Duration::from_millis(8));
    if due {
        let secs = g.0.map_or(1.0 / 60.0, |t| now.duration_since(t).as_secs_f32()).clamp(1.0 / 240.0, 0.25);
        let (dx, dy) = (MOUSE_DX.swap(0, Relaxed) as f32, MOUSE_DY.swap(0, Relaxed) as f32);
        let pc = pc_sensitivity();
        let full = FULL_SPEED / (l.sensitivity * pc);
        let full_y = FULL_SPEED / (l.sensitivity_y.unwrap_or(l.sensitivity) * pc);
        let dy = if invert_y(l) { -dy } else { dy };
        *g = (Some(now), (dx / secs / full).clamp(-1.0, 1.0), (dy / secs / full_y).clamp(-1.0, 1.0));
    }
    // Radially, as the title removes the dead zone.
    let m = (g.1 * g.1 + g.2 * g.2).sqrt();
    if m > 1e-3 {
        let r = STICK_DEAD + (1.0 - STICK_DEAD) * m.min(1.0);
        let to_axis = |d: f32| (128.0 + d / m * r * 127.0).round().clamp(0.0, 255.0) as u8;
        st.rx = to_axis(g.1);
        st.ry = to_axis(g.2);
    }
}

/// Left-stick tilt while `walk` is held.
const WALK_TILT: f32 = 0.4;

fn keyboard_mouse(st: &mut PadState) {
    let l = layout();
    let mods = held_mods();
    // Stick directions held: left x-, x+, y-, y+, right x-, x+, y-, y+.
    let mut dirs = [false; 8];
    let mut walk = false;
    for b in &l.binds {
        if b.mods & !mods != 0 || !held(b.src) || !applying(l, b.src, mods).any(|x| x == b) {
            continue;
        }
        match b.dst {
            Dst::Button(x) => st.buttons |= x,
            Dst::Walk => walk = true,
            Dst::Stick(s) => {
                dirs[match s {
                    Stick::LLeft => 0,
                    Stick::LRight => 1,
                    Stick::LUp => 2,
                    Stick::LDown => 3,
                    Stick::RLeft => 4,
                    Stick::RRight => 5,
                    Stick::RUp => 6,
                    Stick::RDown => 7,
                }] = true
            }
        }
    }
    let stick = |neg: bool, pos: bool, cur: u8| -> u8 {
        match (neg, pos) {
            (true, false) => 0,
            (false, true) => 255,
            _ => cur,
        }
    };
    st.lx = stick(dirs[0], dirs[1], st.lx);
    st.ly = stick(dirs[2], dirs[3], st.ly);
    st.rx = stick(dirs[4], dirs[5], st.rx);
    st.ry = stick(dirs[6], dirs[7], st.ry);
    if walk {
        let tilt = |v: u8| (128.0 + (v as f32 - 128.0) * WALK_TILT).round() as u8;
        st.lx = tilt(st.lx);
        st.ly = tilt(st.ly);
    }
    wheel_taps(st, l, mods);
    mouse_look(st, l);
}

/// Current combined pad state.
pub fn pad_state() -> PadState {
    let mut st = PadState::default();
    xinput(&mut st);
    keyboard_mouse(&mut st);
    scripted(&mut st, crate::gpu::FRAMES.load(std::sync::atomic::Ordering::Relaxed));
    if st.l2 > 30 {
        st.buttons |= BTN_L2;
    }
    if st.r2 > 30 {
        st.buttons |= BTN_R2;
    }
    if st.buttons & BTN_L2 != 0 {
        st.l2 = st.l2.max(255);
    }
    if st.buttons & BTN_R2 != 0 {
        st.r2 = st.r2.max(255);
    }
    st
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn script_parses_and_applies() {
        let s = parse_script("700:cross; 760:down+ls_left:3 ;800:Cross").unwrap();
        assert_eq!(s.len(), 3);
        assert_eq!(s[0], Step { flip: 700, hold: 6, buttons: BTN_CROSS, sticks: vec![] });
        assert_eq!(s[1].buttons, BTN_DOWN);
        assert_eq!(s[1].sticks, vec![Stick::LLeft]);
        assert_eq!(s[1].hold, 3);
        assert!(parse_script("700:jump").is_err());
        assert!(parse_script("x:cross").is_err());
    }

    #[test]
    fn default_layout_parses() {
        let l = parse_layout(DEFAULT_LAYOUT).unwrap();
        assert!(l.mouse_look && !l.invert_y && l.sensitivity == 1.0 && l.mouse_raw);
        // Files written before the key existed get raw mouse look too.
        assert!(parse_layout("mouse_look = 1\n").unwrap().mouse_raw);
        assert!(!parse_layout("mouse_look_raw = 0\n").unwrap().mouse_raw);
        let find = |src: Src, mods: u8| l.binds.iter().find(|b| b.src == src && b.mods == mods).map(|b| b.dst);
        assert_eq!(find(Src::Mouse(0), 0), Some(Dst::Button(BTN_R1)));
        assert_eq!(find(Src::Mouse(0), MOD_SHIFT), Some(Dst::Button(BTN_R2)));
        assert_eq!(find(Src::Key(0x20), 0), Some(Dst::Button(BTN_CIRCLE)));
        assert_eq!(find(Src::Key(0xA4), 0), Some(Dst::Walk));
        assert_eq!(find(Src::Key(0x1B), 0), Some(Dst::Button(BTN_OPTIONS)));
        assert_eq!(find(Src::Key(b'W'), 0), Some(Dst::Stick(Stick::LUp)));
        assert_eq!(find(Src::WheelDown, MOD_SHIFT), Some(Dst::Button(BTN_LEFT)));
    }

    #[test]
    fn modified_binding_wins() {
        let l = parse_layout("Mouse1 = r1\nShift+Mouse1 = r2\nCtrl+Shift+Mouse1 = l3\n").unwrap();
        let got = |held: u8| applying(&l, Src::Mouse(0), held).map(|b| b.dst).collect::<Vec<_>>();
        assert_eq!(got(0), vec![Dst::Button(BTN_R1)]);
        assert_eq!(got(MOD_SHIFT), vec![Dst::Button(BTN_R2)]);
        assert_eq!(got(MOD_SHIFT | MOD_CTRL), vec![Dst::Button(BTN_L3)]);
        assert_eq!(got(MOD_CTRL), vec![Dst::Button(BTN_R1)]);
    }

    #[test]
    fn layout_errors_name_the_line() {
        assert!(parse_layout("W = jump").unwrap_err().contains("line 1"));
        assert!(parse_layout("\nHyper+W = ls_up").unwrap_err().contains("line 2"));
        assert!(parse_layout("mouse_sensitivity = -1").is_err());
        assert!(parse_layout("mouse_sensitivity_y = 0").is_err());
        let l = parse_layout("mouse_sensitivity = 2\nmouse_sensitivity_y = 0.5").unwrap();
        assert!(l.sensitivity == 2.0 && l.sensitivity_y == Some(0.5));
        assert_eq!(parse_layout(DEFAULT_LAYOUT).unwrap().sensitivity_y, None);
        assert_eq!(key_code("f12"), Some(0x7B));
        assert_eq!(key_code("LAlt"), Some(0xA4));
        assert_eq!(parse_src("shift+wheelup"), Some((Src::WheelUp, MOD_SHIFT)));
    }
}
