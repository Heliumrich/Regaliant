//! The menu pointer: while a menu of the title is open (its UI manager's menu stack, `uistate`)
//! and the keyboard and mouse are the device in use, the window frees the cursor and shows the
//! port's pointer; moving it over an item of the focused list moves the title's selection there
//! at once: one d-pad press toward it makes the title's list navigation run, and a hook at that
//! navigation's entry (`select_hook`) turns the move into the list's own jump to the item (its
//! highlight, scrolling and one navigation sound). Lists the hook does not serve (grids) step
//! toward the item with d-pad presses as before. A left click on an item selects it (Cross,
//! after the selection reached it), a right click goes back (Circle), the wheel moves the
//! selection (Up/Down).
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

/// Tests: the pointer at `(u, v)` of the picture as shown (1080 units high), placed in the
/// window's client area as a real mouse there would be (through Menu & HUD Size's mapping).
pub fn test_screen(u: f32, v: f32) -> Result<(), String> {
    let cw = crate::gpu::vk::window::CLIENT_W.load(Relaxed);
    let ch = crate::gpu::vk::window::CLIENT_H.load(Relaxed);
    let (ow, oh) = crate::dispmode::output_size();
    let (rx, ry, rw, rh) = crate::gpu::vk::present::fit(ow, oh, cw, ch);
    if rw == 0 || rh == 0 {
        return Err("no picture in the window".into());
    }
    let width = 1080.0 * ow as f32 / oh as f32;
    let x = rx as f32 + u / width * rw as f32;
    let y = ry as f32 + v / 1080.0 * rh as f32;
    *TEST_AT.lock() = None;
    CLIENT_X.store(x.round() as i32, Relaxed);
    CLIENT_Y.store(y.round() as i32, Relaxed);
    MOVES.fetch_add(1, Relaxed);
    input::note_device(input::DEVICE_KBM);
    crate::info!("pointer: test at client ({:.0}, {:.0}) of {cw}x{ch} (picture {u:.0},{v:.0}; canvas {:?})", x, y, canvas_pos());
    Ok(())
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
    /// The target a direct jump was asked for (once per target).
    direct_tried: Option<u64>,
}

static STATE: Mutex<Option<State>> = Mutex::new(None);

/// The item the pointer asks the title's next list move to jump to (`select_hook`), and since
/// when (a request the title did not take within `DIRECT_WAIT` falls back to stepping).
static DIRECT: Mutex<Option<(u64, Instant)>> = Mutex::new(None);
const DIRECT_WAIT: Duration = Duration::from_millis(250);
/// Direct jumps made, and lists the hook did not find the item in.
static DIRECT_STATS: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// The title's menu list selection change (eboot+0xe2a8d0, `titlefix`), at its entry with the
/// call's registers (`patch::Patcher::hook`: rax, rcx, rdx, rbx, rsp, rbp, rsi, rdi, r8 .. r15,
/// written back after the handler; the vector registers in the FXSAVE area below them, restored
/// from there). When the pointer waits for a jump to an item of this list (rdi) the move becomes
/// the list's own jump to that item: no direction (xmm0 zero), its index in esi, the other
/// arguments as the list's focus handler passes them (0).
pub extern "sysv64" fn select_hook(regs: *const u64, _arg: u32) {
    let mut g = DIRECT.lock();
    let Some((target, since)) = *g else { return };
    if since.elapsed() > DIRECT_WAIT {
        *g = None;
        return;
    }
    let regs = regs as *mut u64;
    unsafe {
        let list = *regs.add(7);
        if list < 0x10000 {
            return;
        }
        // The list's items: entries of three words, the item's node first (`items` at +0x158,
        // their end at +0x160).
        let (begin, end) = (*((list + 0x158) as *const u64), *((list + 0x160) as *const u64));
        if begin == 0 || end < begin || (end - begin) % 0x18 != 0 || end - begin > 0x18 * 4096 {
            return;
        }
        let n = (end - begin) / 0x18;
        let Some(i) = (0..n).find(|&i| *((begin + i * 0x18) as *const u64) == target) else {
            DIRECT_STATS[1].fetch_add(1, Relaxed);
            return;
        };
        *g = None;
        *regs.add(6) = i;
        *regs.add(2) = 0;
        *regs.add(1) = 0;
        *regs.add(8) = 0;
        // xmm0: the trampoline saved the flags at regs - 8, aligned the stack down to 16 and put
        // the FXSAVE area 512 bytes below; XMM0 is at byte 160 of it.
        let fx = ((regs as u64 - 8) & !15) - 512;
        std::ptr::write_bytes((fx + 160) as *mut u8, 0, 16);
    }
    if DIRECT_STATS[0].fetch_add(1, Relaxed) < 4 {
        crate::info!("pointer: the title's list jumps to the item under the pointer");
    }
}

/// The title's input grid cell selection (eboot+0xf648b0, `titlefix`), at its entry: when the
/// pointer waits for a jump to a cell of this grid (rdi), the move selects that cell: its column
/// (esi) and row (edx) found with the grid's own lookups, called here on the title's thread
/// (the row count of a column, eboot+0xf640c0; the cell at a column and row, eboot+0xf63db0;
/// the columns are the definition's entries of 40 bytes at +0xa0..+0xa8).
pub extern "sysv64" fn grid_hook(regs: *const u64, _arg: u32) {
    let mut g = DIRECT.lock();
    let Some((target, since)) = *g else { return };
    if since.elapsed() > DIRECT_WAIT {
        *g = None;
        return;
    }
    let Some(base) = crate::loader::main_module().map(|m| m.base) else { return };
    let regs = regs as *mut u64;
    unsafe {
        let grid = *regs.add(7);
        if grid < 0x10000 {
            return;
        }
        let def = *((grid + 0x110) as *const u64);
        if def < 0x10000 {
            return;
        }
        let (cb, ce) = (*((def + 0xa0) as *const u64), *((def + 0xa8) as *const u64));
        if ce < cb || (ce - cb) % 40 != 0 || ce - cb > 40 * 64 {
            return;
        }
        let rows: extern "sysv64" fn(u64, u32) -> u64 = std::mem::transmute(base + 0xf640c0);
        let cell: extern "sysv64" fn(u64, u32, u32) -> u64 = std::mem::transmute(base + 0xf63db0);
        for c in 0..((ce - cb) / 40) as u32 {
            let nr = rows(grid, c) as u32;
            if nr == u32::MAX || nr > 512 {
                continue;
            }
            for r in 0..nr {
                if cell(grid, c, r) == target {
                    *g = None;
                    *regs.add(6) = c as u64;
                    *regs.add(2) = r as u64;
                    if DIRECT_STATS[0].fetch_add(1, Relaxed) < 4 {
                        crate::info!("pointer: the title's grid jumps to the cell under the pointer (column {c}, row {r})");
                    }
                    return;
                }
            }
        }
        DIRECT_STATS[1].fetch_add(1, Relaxed);
    }
}

/// Status: direct jumps made and moves in lists the hook did not serve.
pub fn direct_line() -> String {
    let (a, b) = (DIRECT_STATS[0].load(Relaxed), DIRECT_STATS[1].load(Relaxed));
    if a == 0 && b == 0 {
        return String::new();
    }
    format!("    menu pointer: {a} direct jumps to the item under the pointer, {b} list moves without the item\n")
}

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
    // Menu & HUD Size: the interface is drawn scaled around the screen's centre.
    let s = super::ui_size();
    let (u, v) = if s < 1.0 { ((u - width / 2.0) / s + width / 2.0, (v - 540.0) / s + 540.0) } else { (u, v) };
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
    // The nodes are read with their pools held mapped (`uistate::guarded`; a skipped vblank
    // when the title's mappings change meanwhile); a new scan is asked for after it.
    if uistate::guarded(&tree, || tick_nodes(&tree, want)) == Some(true) {
        uistate::rescan();
    }
}

/// `tick`'s part that reads the UI nodes (under `uistate::guarded`): true when the index
/// misses the focused item (a new scan wanted).
fn tick_nodes(tree: &std::sync::Arc<uistate::Tree>, want: bool) -> bool {
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
        direct_tried: None,
    });
    let found = focused(tree);
    if found != s.focused {
        for &n in found.iter().filter(|n| !s.focused.contains(n)) {
            if s.logged < 60 {
                s.logged += 1;
                crate::info!("pointer: focus on node {n:#x} at {:?}{}", uistate::bounds(tree, n), if want { "" } else { " (followed, the pointer off)" });
            }
        }
        s.focused = found.clone();
    }
    super::bindings::tick(tree, &found);
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
        return false;
    }
    let events: Vec<Event> = std::mem::take(&mut *EVENTS.lock());
    let moves = MOVES.load(Relaxed);
    let moved = moves != s.moves_seen;
    s.moves_seen = moves;
    if found.is_empty() {
        // No focused item (a screen without a list, or one the index misses: scanned again).
        for e in events {
            match e {
                Event::Right => s.queue.push_back("circle"),
                Event::Wheel(n) => (0..n.unsigned_abs().min(4)).for_each(|_| s.queue.push_back(if n > 0 { "up" } else { "down" })),
                Event::Left => s.queue.push_back("cross"),
            }
        }
        pump(s);
        return true;
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
                let siblings = tree.children.get(&group).map(|v| v.as_slice()).unwrap_or(&[]).iter().map(move |&n| (group, n));
                // A grid's other columns: the cells under the group's siblings of the focused
                // cell's size (as placed), taken as the focused group's (the move toward one runs
                // the grid's own selection, `grid_hook`).
                let (_, _, fw, fh) = uistate::local(f);
                let same = move |n: u64| {
                    let (_, _, w, h) = uistate::local(n);
                    fw > 1.0 && fh > 1.0 && (w - fw).abs() < 0.5 && (h - fh).abs() < 0.5
                };
                let cousins: Vec<(u64, u64)> = match fw.is_finite() {
                    true => tree
                        .children
                        .get(&uistate::parent(group))
                        .map(|v| v.as_slice())
                        .unwrap_or(&[])
                        .iter()
                        .filter(|&&col| col != group)
                        .flat_map(|col| tree.children.get(col).map(|v| v.as_slice()).unwrap_or(&[]).iter())
                        .filter(|&&n| same(n))
                        .map(|&n| (group, n))
                        .collect(),
                    false => Vec::new(),
                };
                siblings.chain(cousins)
            })
            .filter_map(|(group, n)| uistate::bounds(tree, n).map(|r| (group, n, r)))
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
                Some((group, h)) if focused_in(group) == Some(h) => s.queue.push_back(pos.and_then(|p| value_side(tree, h, p)).unwrap_or("cross")),
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
        return false;
    }
    // Toward the target, one press at a time, from its list's selected item.
    if let Some(t) = s.target {
        let group = s.target_group;
        match focused_in(group) {
            Some(f) if f == t => {
                s.target = None;
                s.direct_tried = None;
                *DIRECT.lock() = None;
                s.next_press = s.next_press.min(Instant::now() + Duration::from_millis(40));
                if std::mem::take(&mut s.click) {
                    s.queue.push_back(pos.and_then(|p| value_side(tree, t, p)).unwrap_or("cross"));
                    pump(s);
                }
            }
            Some(f) if Instant::now() >= s.next_press && s.tried.is_none() => {
                // A jump the title's list takes in one move (`select_hook`): asked once per
                // target; a request it did not take within DIRECT_WAIT steps as before.
                let direct = crate::titlefix::applied("menu pointer selects directly") && s.direct_tried != Some(t);
                if direct {
                    s.direct_tried = Some(t);
                    *DIRECT.lock() = Some((t, Instant::now()));
                }
                if let (Some(a), Some(b)) = (uistate::bounds(tree, f), uistate::bounds(tree, t)) {
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
                    s.next_press = Instant::now() + if direct { DIRECT_WAIT } else { Duration::from_millis(110) };
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
    false
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
