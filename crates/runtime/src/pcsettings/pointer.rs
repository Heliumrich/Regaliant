//! The menu pointer: while a menu of the title is open (its UI manager's menu stack, `uistate`)
//! and the keyboard and mouse are the device in use, the window frees the cursor and shows the
//! port's pointer; moving it over an item of the focused list or grid moves the title's focus
//! there with the title's own navigation (d-pad presses toward it, so its sounds, scrolling and
//! rules apply), a left click on an item selects it (Cross, after the focus reached it), a right
//! click goes back (Circle), the wheel moves the focus (Up/Down).
//!
//! The items: the focused node and its siblings (the children of its parent: a list's rows, a
//! grid's slots, the SETTINGS tabs), each covering its subtree's placed rectangles on the
//! title's canvas (1080 units high; 1920 wide at 16:9, `uistate::bounds`); the pointer is mapped from the window to the
//! canvas as the presenter fits the frame. Every list keeps a selected item (flag 0x1); the
//! focused one is the selected item of the list on the input focus's path (0x40).

use crate::input;
use crate::uistate;
use parking_lot::Mutex;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU32, Ordering::Relaxed};
use std::time::{Duration, Instant};

/// Pointer mode now (the window frees the cursor and draws the pointer).
pub static ACTIVE: AtomicBool = AtomicBool::new(false);
/// The pointer in client pixels, and a count of moves.
pub static CLIENT_X: AtomicI32 = AtomicI32::new(-1);
pub static CLIENT_Y: AtomicI32 = AtomicI32::new(-1);
pub static MOVES: AtomicU32 = AtomicU32::new(0);

#[derive(Clone, Copy, Debug)]
pub enum Event {
    Left,
    Right,
    Wheel(i32),
}

static EVENTS: Mutex<Vec<Event>> = Mutex::new(Vec::new());

/// A pointer event from the window (or the control channel).
pub fn event(e: Event) {
    let mut g = EVENTS.lock();
    if g.len() < 32 {
        g.push(e);
    }
}

/// The pointer at canvas coordinates for tests (`pointer <x> <y>`): as if moved there.
static TEST_AT: Mutex<Option<(f32, f32)>> = Mutex::new(None);

pub fn test_move(x: f32, y: f32) {
    *TEST_AT.lock() = Some((x, y));
    MOVES.fetch_add(1, Relaxed);
    input::note_device(input::DEVICE_KBM);
}

struct State {
    /// The item the pointer last moved onto (the focus is moved there) and its list.
    target: Option<u64>,
    target_group: u64,
    /// A click waits for the focus to reach the target.
    click: bool,
    /// The focused items last seen.
    focused: Vec<u64>,
    /// Lists whose horizontal moves take L1/R1 (a tab bar: Left/Right did not move it), and the
    /// last horizontal press (its list, its selected item then, when).
    shoulder: std::collections::HashSet<u64>,
    tried: Option<(u64, u64, Instant)>,
    next_press: Instant,
    /// Presses waiting their turn (spaced so the title sees each).
    queue: std::collections::VecDeque<&'static str>,
    moves_seen: u32,
    logged: u32,
}

static STATE: Mutex<Option<State>> = Mutex::new(None);

/// The pointer on the canvas (None: outside the frame, or unknown).
fn canvas_pos() -> Option<(f32, f32)> {
    if let Some(p) = *TEST_AT.lock() {
        return Some(p);
    }
    let (x, y) = (CLIENT_X.load(Relaxed), CLIENT_Y.load(Relaxed));
    if x < 0 || y < 0 {
        return None;
    }
    let cw = crate::gpu::vk::window::CLIENT_W.load(Relaxed);
    let ch = crate::gpu::vk::window::CLIENT_H.load(Relaxed);
    // The frame as the presenter fits it; the canvas 1080 units high, as wide as the output's
    // shape (wider than 1920 with another Aspect Ratio).
    let (ow, oh) = crate::dispmode::output_size();
    let (rx, ry, rw, rh) = crate::gpu::vk::present::fit(ow, oh, cw, ch);
    if rw == 0 || rh == 0 {
        return None;
    }
    let width = 1080.0 * ow as f32 / oh as f32;
    let u = (x as f32 - rx as f32) / rw as f32 * width;
    let v = (y as f32 - ry as f32) / rh as f32 * 1080.0;
    ((0.0..width).contains(&u) && (0.0..1080.0).contains(&v)).then_some((u, v))
}

/// The focused items: the selected item (0x1) of each list on the input focus's path (its
/// container marked 0x40: the path runs from the screen down to the lists the player moves in,
/// e.g. the start menu's tab bar (L1/R1) and the SETTINGS tab list (Up/Down); it moves to a
/// page's list as the page is entered and back as it is left).
fn focused(t: &uistate::Tree) -> Vec<u64> {
    t.nodes
        .iter()
        .copied()
        .filter(|&n| uistate::flags(n) & uistate::SELECTED != 0)
        .filter(|&n| {
            let p = uistate::parent(n);
            p != 0 && uistate::flags(p) & uistate::FOCUS_PATH != 0
        })
        .filter(|&n| uistate::absolute(n).is_some())
        .collect()
}

fn center(r: (f32, f32, f32, f32)) -> (f32, f32) {
    ((r.0 + r.2) / 2.0, (r.1 + r.3) / 2.0)
}

/// A click on an item's value (a choice's or slider's widget: a descendant in the right part of
/// the item under the pointer): Left or Right by the side of the widget's centre it lands on.
fn value_side(t: &uistate::Tree, item: u64, (x, y): (f32, f32)) -> Option<&'static str> {
    let ib = uistate::bounds(t, item)?;
    let split = ib.0 + (ib.2 - ib.0) * 0.55;
    let mut best: Option<(f32, (f32, f32, f32, f32))> = None;
    let mut stack: Vec<(u64, u32)> = t.children.get(&item).map(|v| v.iter().map(|&k| (k, 0)).collect()).unwrap_or_default();
    while let Some((n, depth)) = stack.pop() {
        if let Some(r) = uistate::bounds(t, n) {
            let area = (r.2 - r.0) * (r.3 - r.1);
            if x >= r.0 && x < r.2 && y >= r.1 && y < r.3 && center(r).0 > split && best.is_none_or(|b| area < b.0) {
                best = Some((area, r));
            }
        }
        if depth < 8 {
            if let Some(k) = t.children.get(&n) {
                stack.extend(k.iter().map(|&c| (c, depth + 1)));
            }
        }
    }
    best.map(|(_, r)| if x < center(r).0 { "left" } else { "right" })
}

/// The vblank thread, each vblank.
pub fn tick() {
    // The node index is built once the title shows its first menu (the UI's node pools exist
    // by then; a scan of some seconds, usually done before the player uses the pointer).
    static STARTED: AtomicBool = AtomicBool::new(false);
    if !STARTED.load(Relaxed) && uistate::menu_depth().unwrap_or(0) > 0 && uistate::tree().is_some() {
        STARTED.store(true, Relaxed);
    }
    let menu = uistate::menu_depth().unwrap_or(0) > 0;
    let want = menu && input::last_device() == input::DEVICE_KBM;
    if ACTIVE.swap(want, Relaxed) != want {
        crate::info!("pointer: {}", if want { "on (a menu is open, keyboard and mouse in use)" } else { "off" });
        crate::gpu::vk::window::pointer_mode_changed();
        EVENTS.lock().clear();
        if !want {
            *TEST_AT.lock() = None;
        }
    }
    // The focus is followed while any menu is open (the pad's moves too: the pointer then
    // starts from the item the player left), forgotten when the menus close.
    if !menu {
        *STATE.lock() = None;
        return;
    }
    let Some(tree) = uistate::tree() else { return };
    let mut g = STATE.lock();
    let s = g.get_or_insert_with(|| State {
        target: None,
        target_group: 0,
        click: false,
        focused: Vec::new(),
        shoulder: Default::default(),
        tried: None,
        next_press: Instant::now(),
        queue: Default::default(),
        moves_seen: MOVES.load(Relaxed),
        logged: 0,
    });
    let found = focused(&tree);
    if found != s.focused {
        for &n in found.iter().filter(|n| !s.focused.contains(n)) {
            if s.logged < 60 {
                s.logged += 1;
                crate::info!("pointer: focus on node {n:#x} at {:?}{}", uistate::bounds(&tree, n), if want { "" } else { " (followed, the pointer off)" });
            }
        }
        s.focused = found.clone();
    }
    super::bindings::tick(&tree, &found);
    // A horizontal press that did not move its list: that list takes L1/R1.
    if let Some((group, before, when)) = s.tried {
        let now_sel = found.iter().copied().find(|&n| uistate::parent(n) == group);
        if now_sel != Some(before) {
            s.tried = None;
        } else if when.elapsed() > Duration::from_millis(400) {
            s.tried = None;
            if s.shoulder.insert(group) {
                crate::info!("pointer: the list {group:#x} moves with L1/R1");
            }
        }
    }
    if !want {
        s.target = None;
        s.click = false;
        s.queue.clear();
        s.moves_seen = MOVES.load(Relaxed);
        return;
    }
    let events: Vec<Event> = std::mem::take(&mut *EVENTS.lock());
    let moves = MOVES.load(Relaxed);
    let moved = moves != s.moves_seen;
    s.moves_seen = moves;
    if found.is_empty() {
        // No focused item (a screen without a list, or one the index misses: scanned again).
        uistate::rescan();
        for e in events {
            match e {
                Event::Right => s.queue.push_back("circle"),
                Event::Wheel(n) => (0..n.unsigned_abs().min(4)).for_each(|_| s.queue.push_back(if n > 0 { "up" } else { "down" })),
                Event::Left => s.queue.push_back("cross"),
            }
        }
        pump(s);
        return;
    }
    // The items: each focused item's list (its parent's children), where they cover the pointer:
    // the one whose centre is nearest (an item's glow or frame may reach into its neighbours').
    let pos = canvas_pos();
    let hovered: Option<(u64, u64)> = pos.and_then(|(x, y)| {
        let dist = |r: (f32, f32, f32, f32)| {
            let (cx, cy) = center(r);
            (cx - x).powi(2) + (cy - y).powi(2)
        };
        found
            .iter()
            .flat_map(|&f| {
                let group = uistate::parent(f);
                tree.children.get(&group).map(|v| v.as_slice()).unwrap_or(&[]).iter().map(move |&n| (group, n))
            })
            .filter_map(|(group, n)| uistate::bounds(&tree, n).map(|r| (group, n, r)))
            .filter(|(_, _, r)| x >= r.0 && x < r.2 && y >= r.1 && y < r.3)
            .min_by(|a, b| dist(a.2).total_cmp(&dist(b.2)))
            .map(|(group, n, _)| (group, n))
    });
    let focused_in = |group: u64| found.iter().copied().find(|&n| uistate::parent(n) == group);
    if moved {
        if let Some((group, h)) = hovered {
            s.target = Some(h);
            s.target_group = group;
        }
    }
    for e in events {
        match e {
            Event::Left => match hovered {
                Some((group, h)) if focused_in(group) == Some(h) => s.queue.push_back(pos.and_then(|p| value_side(&tree, h, p)).unwrap_or("cross")),
                Some((group, h)) => {
                    s.target = Some(h);
                    s.target_group = group;
                    s.click = true;
                }
                None => {}
            },
            Event::Right => {
                s.target = None;
                s.queue.push_back("circle");
            }
            Event::Wheel(n) => {
                s.target = None;
                (0..n.unsigned_abs().min(4)).for_each(|_| s.queue.push_back(if n > 0 { "up" } else { "down" }));
            }
        }
    }
    if !s.queue.is_empty() {
        pump(s);
        return;
    }
    // Toward the target, one press at a time, from its list's selected item.
    if let Some(t) = s.target {
        let group = s.target_group;
        match focused_in(group) {
            Some(f) if f == t => {
                s.target = None;
                if std::mem::take(&mut s.click) {
                    s.queue.push_back(pos.and_then(|p| value_side(&tree, t, p)).unwrap_or("cross"));
                    pump(s);
                }
            }
            Some(f) if Instant::now() >= s.next_press && s.tried.is_none() => {
                if let (Some(a), Some(b)) = (uistate::bounds(&tree, f), uistate::bounds(&tree, t)) {
                    let ((ax, ay), (bx, by)) = (center(a), center(b));
                    let dir = if (by - ay).abs() >= (bx - ax).abs() {
                        if by > ay { "down" } else { "up" }
                    } else {
                        let shoulder = s.shoulder.contains(&group);
                        if !shoulder {
                            s.tried = Some((group, f, Instant::now()));
                        }
                        match (bx > ax, shoulder) {
                            (true, false) => "right",
                            (false, false) => "left",
                            (true, true) => "r1",
                            (false, true) => "l1",
                        }
                    };
                    press(dir);
                    s.next_press = Instant::now() + Duration::from_millis(110);
                } else {
                    s.target = None;
                }
            }
            Some(_) => {}
            // The target's list lost the focus (another screen): forget it.
            None => {
                s.target = None;
                s.click = false;
            }
        }
    }
}

/// The next queued press, when its turn has come.
fn pump(s: &mut State) {
    if Instant::now() >= s.next_press {
        if let Some(name) = s.queue.pop_front() {
            press(name);
            s.next_press = Instant::now() + Duration::from_millis(110);
        }
    }
}

fn press(name: &str) {
    // A pad press that does not make the pad the device in use.
    let _ = input::press_quiet(name, Duration::from_millis(60));
}
