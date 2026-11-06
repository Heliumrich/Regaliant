//! The title's UI manager, read from guest memory (PPSA01341 01.007.000; nothing is called or
//! written). From the disassembly of `UIManager::PrintSceneState` (eboot+0xde5040..0xde5990)
//! and the console commands that queue work on the manager (eboot+0xdeba60):
//! - the manager at [eboot+0x329B050];
//! - its active menu stack (top first when read from the end): a vector at +0x1260 / +0x1268
//!   of 0x18-byte entries, guarded by a spin lock at +0x1220 (a reader count at +0x1228);
//! - its active HUD scenes: a vector at +0x1288 / +0x1290;
//! - scenes in transition: menu stack +0x910 / +0x918, HUD +0x11F0 / +0x11F8.
//!
//! The menu stack tells the port whether a menu is open (the menu pointer, `pcsettings`).
//! The offsets are checked against the code bytes they were read from first; another
//! executable reports nothing.

const MANAGER: u64 = 0x329B050;
/// eboot+0xde55bd: `mov r14,[rbx+1260h]; mov r15,[rbx+1268h]` (the active menu stack).
const STACK_CODE_AT: u64 = 0xde55bd;
const STACK_CODE: [u8; 14] = [0x4c, 0x8b, 0xb3, 0x60, 0x12, 0x00, 0x00, 0x4c, 0x8b, 0xbb, 0x68, 0x12, 0x00, 0x00];
/// eboot+0xde5696: `mov rax,[rbx+1290h]; cmp rax,[rbx+1288h]` (the HUD scenes).
const HUD_CODE_AT: u64 = 0xde5696;
const HUD_CODE: [u8; 14] = [0x48, 0x8b, 0x83, 0x90, 0x12, 0x00, 0x00, 0x48, 0x3b, 0x83, 0x88, 0x12, 0x00, 0x00];
const ENTRY: u64 = 0x18;

fn base() -> Option<u64> {
    crate::loader::main_module().map(|m| m.base)
}

fn rd64(a: u64) -> Option<u64> {
    if a < 0x1000 || !crate::mem::readable(a, 8) {
        return None;
    }
    Some(unsafe { std::ptr::read_volatile(a as *const u64) })
}

/// The layout holds for this executable (its code bytes as read).
fn checked() -> Option<u64> {
    static OK: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    *OK.get_or_init(|| {
        let b = base()?;
        let same = |at: u64, code: &[u8]| crate::mem::readable(b + at, code.len() as u64) && unsafe { std::slice::from_raw_parts((b + at) as *const u8, code.len()) } == code;
        if same(STACK_CODE_AT, &STACK_CODE) && same(HUD_CODE_AT, &HUD_CODE) {
            Some(b)
        } else {
            crate::warn!("uistate: the UI manager's layout does not match this executable");
            None
        }
    })
}

/// The UI manager (None before the title made it).
fn manager() -> Option<u64> {
    let b = checked()?;
    rd64(b + MANAGER).filter(|&m| m >= 0x1000)
}

fn vec_len(m: u64, at: u64) -> Option<u64> {
    let (begin, end) = (rd64(m + at)?, rd64(m + at + 8)?);
    (end >= begin && (end - begin) % ENTRY == 0 && end - begin < 0x10000).then(|| (end - begin) / ENTRY)
}

/// Scenes on the active menu stack (None: no UI manager yet, or another executable).
pub fn menu_depth() -> Option<u64> {
    vec_len(manager()?, 0x1260)
}

/// Active HUD scenes.
pub fn hud_count() -> Option<u64> {
    let m = manager()?;
    let (begin, end) = (rd64(m + 0x1288)?, rd64(m + 0x1290)?);
    (end >= begin && end - begin < 0x10000).then(|| end - begin)
}

/// Diagnosis (`uistack`): the manager, the stack's entries (raw words) and the HUD vector.
pub fn report() -> String {
    let Some(m) = manager() else { return "uistack: no UI manager (not made yet, or another executable)".into() };
    let mut s = format!("uistack: manager {m:#x}, menu stack {:?} scene(s), HUD bytes {:?}", menu_depth(), hud_count());
    if let (Some(begin), Some(n)) = (rd64(m + 0x1260), menu_depth()) {
        for i in 0..n.min(8) {
            let e = begin + i * ENTRY;
            let w: Vec<String> = (0..3).map(|k| rd64(e + k * 8).map(|v| format!("{v:#x}")).unwrap_or("?".into())).collect();
            s += &format!("; [{i}] {}", w.join(" "));
        }
    }
    s
}

/// The UIElement component's vtable (eboot+0x25d83a8; its loader, eboot+0xe09210, reads X +0xa4,
/// Y +0xa8, Width +0xb0, Height +0xb4).
const ELEMENT_VTABLE: u64 = 0x25d83a8;

/// Diagnosis (`uielems <tag>`): every UIElement component in guest memory (found by its vtable)
/// with its first 0x100 bytes, one line each, in `<run dir>/uielems-<tag>.txt`.
pub fn dump_elements(dir: &std::path::Path, tag: &str) -> Result<String, String> {
    dump_by_vtable(dir, &format!("uielems-{tag}"), ELEMENT_VTABLE, 0x40)
}

/// The runtime UI node's vtable (eboot+0x25d8430: its definition, a UIElement, at +0x110).
const NODE_VTABLE: u64 = 0x25d8430;

/// Diagnosis (`uinodes <tag>`): every runtime UI node with its first 0x300 bytes.
pub fn dump_nodes(dir: &std::path::Path, tag: &str) -> Result<String, String> {
    dump_by_vtable(dir, &format!("uinodes-{tag}"), NODE_VTABLE, 0xD0)
}

fn dump_by_vtable(dir: &std::path::Path, name: &str, vtable: u64, words_n: u64) -> Result<String, String> {
    let b = checked().ok_or("no UI manager layout for this executable")?;
    let vt = b + vtable;
    let spans: Vec<(u64, u64)> = crate::mem::MEM.lock().vmas.values().filter(|m| m.kind != crate::mem::VmaKind::Reserved && m.prot & 1 != 0).map(|m| (m.start, m.end())).collect();
    let t0 = std::time::Instant::now();
    let mut out = String::new();
    let mut n = 0;
    for (lo, hi) in spans {
        let mut a = (lo + 7) & !7;
        while a + words_n * 4 <= hi {
            if unsafe { *(a as *const u64) } == vt {
                let words: Vec<u32> = (0..words_n).map(|i| unsafe { *((a + i * 4) as *const u32) }).collect();
                out += &format!("{a:#x}");
                for w in &words {
                    out += &format!(" {w:08x}");
                }
                out += "\n";
                n += 1;
            }
            a += 8;
        }
    }
    let path = dir.join(format!("{name}.txt"));
    std::fs::write(&path, out).map_err(|e| e.to_string())?;
    Ok(format!("{name}: {n} objects in {} ms -> {}", t0.elapsed().as_millis(), path.display()))
}

// ---------------------------------------------------------------------------
// The runtime UI node tree (the menu pointer, `pcsettings::pointer`). From the disassembly of
// the node's size and position update (eboot+0xe0b8f0) and dumps of every node in a boot
// (2026-11-06): a node's definition (a UIElement: X +0xa4, Y +0xa8, W +0xb0, H +0xb4) at +0x110,
// its parent at +0x2c8, flags at +0x2dc (0x41 set on the focused item), its position relative
// to the parent at +0x2fc/+0x300 and its size at +0x304/+0x308, in units of a 1920x1080 canvas
// (FLT_MAX: not placed; a scene's root is drawn at the origin then). Flag 0x1 marks the selected
// item of each list (68 at once in a SETTINGS page), 0x40 the input focus's path (the containers
// from the screen down to the list the player moves in: entering a page moves it from the tab
// list to the page's list, leaving moves it back). The node pools
// do not change while the game runs (36458 nodes in every state of a boot), so one scan finds
// them all.

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

pub struct Tree {
    pub nodes: Vec<u64>,
    pub children: HashMap<u64, Vec<u64>>,
}

static TREE: Mutex<Option<Arc<Tree>>> = Mutex::new(None);
static SCANNING: AtomicBool = AtomicBool::new(false);
static LAST_SCAN: Mutex<Option<std::time::Instant>> = Mutex::new(None);

/// The node tree (None while the first scan runs: it starts on the first call, on a thread of
/// its own).
pub fn tree() -> Option<Arc<Tree>> {
    let t = TREE.lock().clone();
    match t {
        // Scanned before the title made its UI's pools: again (at most every 20 s).
        Some(t) if t.nodes.is_empty() => {
            rescan();
            None
        }
        Some(t) => Some(t),
        None => {
            start_scan();
            None
        }
    }
}

fn start_scan() {
    if checked().is_some() && manager().is_some() && !SCANNING.swap(true, Ordering::SeqCst) {
        *LAST_SCAN.lock() = Some(std::time::Instant::now());
        std::thread::Builder::new()
            .name("ui-tree".into())
            .spawn(|| {
                let t0 = std::time::Instant::now();
                let t = scan();
                crate::info!("uistate: {} UI nodes ({} with children) in {} ms", t.nodes.len(), t.children.len(), t0.elapsed().as_millis());
                *TREE.lock() = Some(Arc::new(t));
                SCANNING.store(false, Ordering::SeqCst);
            })
            .ok();
    }
}

/// A new scan (the old tree serves meanwhile), at most every 20 s: nodes the index misses.
pub fn rescan() {
    if LAST_SCAN.lock().is_some_and(|t| t.elapsed() < std::time::Duration::from_secs(20)) {
        return;
    }
    crate::info!("uistate: the UI nodes are scanned again (a focused item outside the index)");
    start_scan();
}

fn scan() -> Tree {
    let Some(b) = checked() else { return Tree { nodes: Vec::new(), children: HashMap::new() } };
    let vt = b + NODE_VTABLE;
    let spans: Vec<(u64, u64)> = crate::mem::MEM.lock().vmas.values().filter(|m| m.kind != crate::mem::VmaKind::Reserved && m.prot & 1 != 0).map(|m| (m.start, m.end())).collect();
    let mut nodes = Vec::new();
    for (lo, hi) in spans {
        let mut a = (lo + 7) & !7;
        while a + 0x310 <= hi {
            if unsafe { std::ptr::read_volatile(a as *const u64) } == vt {
                nodes.push(a);
            }
            a += 8;
        }
    }
    let mut children: HashMap<u64, Vec<u64>> = HashMap::new();
    for &n in &nodes {
        let p = parent(n);
        if p != 0 {
            children.entry(p).or_default().push(n);
        }
    }
    Tree { nodes, children }
}

fn rdf(a: u64) -> f32 {
    unsafe { std::ptr::read_volatile(a as *const f32) }
}

/// A node's parent (0: none). Nodes are pool objects: readable for the whole run.
pub fn parent(n: u64) -> u64 {
    unsafe { std::ptr::read_volatile((n + 0x2C8) as *const u64) }
}

pub fn flags(n: u64) -> u32 {
    unsafe { std::ptr::read_volatile((n + 0x2DC) as *const u32) }
}

/// A list's selected item.
pub const SELECTED: u32 = 0x1;
/// On the input focus's path: the screen's containers down to the list the player moves in, and
/// that list's item when the focus arrived with the screen or the list.
pub const FOCUS_PATH: u32 = 0x40;

/// A node's definition (its UIElement) and the definition's design width and height.
pub fn definition_size(n: u64) -> Option<(f32, f32)> {
    let d = unsafe { std::ptr::read_volatile((n + 0x110) as *const u64) };
    if d < 0x1000 || !crate::mem::readable(d + 0xB0, 8) {
        return None;
    }
    Some((rdf(d + 0xB0), rdf(d + 0xB4)))
}

/// A node's rectangle relative to its parent (x, y, w, h).
pub fn local(n: u64) -> (f32, f32, f32, f32) {
    (rdf(n + 0x2FC), rdf(n + 0x300), rdf(n + 0x304), rdf(n + 0x308))
}

fn placed(v: f32) -> bool {
    v.is_finite() && v.abs() < 1.0e6
}

/// A node's position on the canvas (its and its ancestors' positions summed); None when one of
/// them is not placed.
pub fn absolute(n: u64) -> Option<(f32, f32)> {
    let (mut x, mut y) = (0.0f32, 0.0f32);
    let mut a = n;
    for _ in 0..64 {
        if a == 0 {
            return Some((x, y));
        }
        let (lx, ly, _, _) = local(a);
        let p = parent(a);
        if !placed(lx) || !placed(ly) {
            // A scene's root without a placement is at the origin; elsewhere: not shown.
            if p == 0 && a != n {
                return Some((x, y));
            }
            return None;
        }
        x += lx;
        y += ly;
        a = p;
    }
    None
}

/// The canvas rectangle (x0, y0, x1, y1) covering a node and its placed descendants of some size.
pub fn bounds(t: &Tree, n: u64) -> Option<(f32, f32, f32, f32)> {
    let (ax, ay) = absolute(n)?;
    let mut acc: Option<(f32, f32, f32, f32)> = None;
    fn walk(t: &Tree, n: u64, ox: f32, oy: f32, depth: u32, acc: &mut Option<(f32, f32, f32, f32)>) {
        let (_, _, w, h) = local(n);
        if placed(w) && placed(h) && w > 1.0 && h > 1.0 {
            let r = (ox, oy, ox + w, oy + h);
            *acc = Some(match *acc {
                None => r,
                Some(a) => (a.0.min(r.0), a.1.min(r.1), a.2.max(r.2), a.3.max(r.3)),
            });
        }
        if depth > 12 {
            return;
        }
        for &k in t.children.get(&n).map(|v| v.as_slice()).unwrap_or(&[]) {
            let (kx, ky, _, _) = local(k);
            if placed(kx) && placed(ky) {
                walk(t, k, ox + kx, oy + ky, depth + 1, acc);
            }
        }
    }
    walk(t, n, ax, ay, 0, &mut acc);
    acc
}
