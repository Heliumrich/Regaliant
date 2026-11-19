//! The title's virtual geometry scene (`VirtualGeometryViewContext::CreateScene`,
//! eboot+0xa655c0) and the hold of its spawning after a renumbering (FINDINGS.md 2026-11-14).
//!
//! Each frame the title hashes the ordered list of active VG entities into the scene's
//! generation; a new generation resets the scene's model numbering, which is then assigned
//! again by first appearance in that list, and stores the scene's spawn enable (the context's
//! +1, checked by the spawn passes; the per-view culling and the draws run regardless). Only a
//! rebuild stores it: between resets it keeps its value. Plants are spawned with the
//! numbering current when they spawn (the model number is part of each instance's key, by
//! which the per-view pass buckets instances into the models' draws) and keep it. In the
//! port's first VG frames after the intro's camera cut, the scene saw another entity set for
//! two frames (a first numbering: weeds 0-3), spawned the tutorial tunnel's weeds with it, and
//! reset to the gameplay set's numbering (grass 0-7, weeds 8-11) right after: those weeds were
//! drawn as the tall thorned grass models, sticking out of the walls and ceiling (the
//! "starburst"), beside the weeds spawned again with the new numbering. The console shows none.
//!
//! The fix holds spawning for a few frames after a scene's first reset (its generation was 0:
//! the scene was just made) and after any reset while that hold is on: the rebuild's enable
//! store (eboot+0xa66b93) is kept at 0 and the title's own value saved; the per-frame update's
//! hook (eboot+0xa553c7, before the active list is hashed, the context in [rbp-0xE8]) puts the
//! saved value back once the scene has held its generation that long, so plants spawn only
//! with a numbering that held. Resets in play do not hold (`reset_hook`): the active entities
//! change as the camera turns, every few frames while it moves, and a hold after each kept new
//! plants out the whole time (2026-11-15). `DS_VG_HOLD=<frames>` sets the hold (default 4; 0
//! off).
//! `DS_VG_DIAG=<lines>` logs the active entity lists, resets and enable values.
//!
//! The active list itself (`keep_hook`, 2026-11-19): an entity is active when its bounds lie
//! within a distance and touch the view frustum (eboot+0xa60da0). One at the view's edge left
//! and re-entered the list as the camera moved by a pixel (the player's tilts in the tutorial
//! courtyard: four entities in and out over consecutive frames), and every change renumbers the
//! models of the entities after it, whose plants then vanish and grow back over ~50 frames (the
//! courtyard's dry leaves). An entity that passed the test within the last `DS_VG_KEEP` frames
//! (default 120; 0 off) stays in the list, in its place, while its own models are present.

use parking_lot::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

static LISTS: AtomicU64 = AtomicU64::new(0);
static RESETS: AtomicU64 = AtomicU64::new(0);
static HELD: AtomicU64 = AtomicU64::new(0);
static RELEASED: AtomicU64 = AtomicU64::new(0);
static LAST: Mutex<Vec<u32>> = Mutex::new(Vec::new());

/// A view context's hold: frames since its last reset, the generation that reset made, and
/// the enable value the rebuild computed (to put back), if held.
struct Hold {
    ctx: u64,
    since: u32,
    generation: u32,
    saved: Option<u8>,
}

static HOLDS: Mutex<Vec<Hold>> = Mutex::new(Vec::new());

fn hold_frames() -> u32 {
    static H: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *H.get_or_init(|| std::env::var("DS_VG_HOLD").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(4))
}

fn diag_limit() -> u64 {
    static LIMIT: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *LIMIT.get_or_init(|| std::env::var("DS_VG_DIAG").ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(0))
}

fn flip() -> u64 {
    crate::gpu::FRAMES.load(Ordering::Relaxed)
}

/// The per-frame update before the active entity list is hashed (every frame VG is active):
/// a held enable goes back once its scene has kept the generation `hold_frames()` frames; with
/// `DS_VG_DIAG`, the list is logged when it changes.
pub extern "sysv64" fn list_hook(regs: *const u64, _arg: u32) {
    let r = |i: usize| unsafe { *regs.add(i) };
    let (rsi, rbp) = (r(6), r(5));
    let ctx = unsafe { *(rbp.wrapping_sub(0xe8) as *const u64) };
    {
        let mut holds = HOLDS.lock();
        if let Some(h) = holds.iter_mut().find(|h| h.ctx == ctx) {
            h.since = h.since.saturating_add(1);
            if h.since >= hold_frames() {
                if let Some(v) = h.saved.take() {
                    // (Only the generation the hold began with: a reset since, or the scene's
                    // deactivation, stored its own value.)
                    let generation = unsafe { *((ctx + 8) as *const u32) };
                    if generation == h.generation {
                        unsafe { *((ctx + 1) as *mut u8) = v };
                        let n = RELEASED.fetch_add(1, Ordering::Relaxed) + 1;
                        if n <= 3 || n <= diag_limit() {
                            crate::info!("vg scene: spawning resumed at flip {} ({} frame(s) after the reset)", flip(), h.since);
                        }
                    }
                }
            }
        }
    }
    if diag_limit() == 0 {
        return;
    }
    let end = unsafe { *((rbp.wrapping_sub(0xd8)) as *const u64) };
    if end < rsi || end - rsi > 4 * 4096 || rsi == 0 {
        return;
    }
    let n = ((end - rsi) / 4) as usize;
    let ids: Vec<u32> = (0..n).map(|k| unsafe { *((rsi + 4 * k as u64) as *const u32) }).collect();
    let mut last = LAST.lock();
    if *last == ids {
        return;
    }
    let same_set = {
        let (mut a, mut b) = (last.clone(), ids.clone());
        a.sort_unstable();
        b.sort_unstable();
        a == b
    };
    *last = ids.clone();
    drop(last);
    if LISTS.fetch_add(1, Ordering::Relaxed) < diag_limit() {
        let shown: Vec<String> = ids.iter().take(48).map(|x| format!("{x:x}")).collect();
        crate::info!(
            "vg diag: active entities ({n}{}) at flip {}: {}{}",
            if same_set { ", same set reordered" } else { "" },
            flip(),
            shown.join(" "),
            if n > 48 { " ..." } else { "" }
        );
    }
}

/// The scene's model numbering reset (a new generation): the hold starts when the scene was
/// just made (generation 0 before) or a hold is still on (a reset again within it: the
/// starburst's two numberings two frames apart). Resets in play (the active entities change
/// as the camera turns: every few frames while it moves, 2026-11-15) do not hold: spawning held
/// 4 frames after each kept new plants out while the camera moved (the user's foliage "in and
/// out of existence" in 0.1.0's videos), and the console spawns at once.
pub extern "sysv64" fn reset_hook(regs: *const u64, _arg: u32) {
    let r = |i: usize| unsafe { *regs.add(i) };
    let ctx = r(14);
    let new = r(3) as u32;
    let old = unsafe { *((ctx + 8) as *const u32) };
    let held = {
        let mut holds = HOLDS.lock();
        let on = holds.iter().any(|h| h.ctx == ctx && h.since < hold_frames());
        let hold = old == 0 || on;
        // (A reset in play ends a hold's wait for its generation: no value to put back.)
        let h = Hold { ctx, since: if hold { 0 } else { u32::MAX }, generation: new, saved: None };
        match holds.iter_mut().find(|h| h.ctx == ctx) {
            Some(e) => *e = h,
            None => holds.push(h),
        }
        hold
    };
    let n = RESETS.fetch_add(1, Ordering::Relaxed) + 1;
    if n <= 3 || n <= diag_limit() {
        crate::info!("vg scene: numbering reset {n} at flip {} (context {ctx:#x}, generation {old:#x} -> {new:#x}{})", flip(), if held { "; spawning held" } else { "" });
    }
}

fn keep_frames() -> u64 {
    static K: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *K.get_or_init(|| std::env::var("DS_VG_KEEP").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(120))
}

/// Per (update, entity): the flip it last passed the title's view test.
static LAST_PASS: Mutex<Option<std::collections::HashMap<(u64, u64), u64>>> = Mutex::new(None);
/// [entities that passed, kept in the list after failing, not kept for want of room in the
/// lists, not kept as their models were gone].
static KEEP: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

fn rd64(a: u64) -> u64 {
    unsafe { (a as *const u64).read_volatile() }
}

fn rd32(a: u64) -> u32 {
    unsafe { (a as *const u32).read_volatile() }
}

/// The per-frame update's step to the next entity (eboot+0xa55170, `add r14,8`; reached after
/// every entity, passed or not: r14 the entity's slot, r15 the update, whose active entities
/// are [+0x48, +0x50) with room to +0x58; the hashed id list ends at [rbp-0xD8], room to
/// [rbp-0xD0]). A passed entity was appended last; one that failed but passed within
/// `keep_frames()` flips is appended to both lists as the title appends a passed one (its id
/// at +0x220), when its own check holds (models present: +0x50 bit 4, +0xB0 != +0xB8) and both
/// lists have room (they are not grown here: none is kept then).
pub extern "sysv64" fn keep_hook(regs: *const u64, _arg: u32) {
    let keep = keep_frames();
    if keep == 0 {
        return;
    }
    let r = |i: usize| unsafe { *regs.add(i) };
    let (rbp, r14, r15) = (r(5), r(14), r(15));
    let ent = rd64(r14);
    if ent == 0 {
        return;
    }
    let (begin, end, cap) = (rd64(r15 + 0x48), rd64(r15 + 0x50), rd64(r15 + 0x58));
    let passed = end > begin && rd64(end - 8) == ent;
    let now = flip();
    let mut g = LAST_PASS.lock();
    let m = g.get_or_insert_with(std::collections::HashMap::new);
    if passed {
        KEEP[0].fetch_add(1, Ordering::Relaxed);
        if m.len() >= 4096 {
            m.retain(|_, &mut t| now.saturating_sub(t) <= keep);
        }
        m.insert((r15, ent), now);
        return;
    }
    let Some(&last) = m.get(&(r15, ent)) else { return };
    if now.saturating_sub(last) > keep {
        m.remove(&(r15, ent));
        return;
    }
    drop(g);
    if rd32(ent + 0x50) & 0x10 == 0 || rd64(ent + 0xb0) == rd64(ent + 0xb8) {
        KEEP[3].fetch_add(1, Ordering::Relaxed);
        return;
    }
    let (ids_end, ids_cap) = (rd64(rbp.wrapping_sub(0xd8)), rd64(rbp.wrapping_sub(0xd0)));
    if end + 8 > cap || ids_end + 4 > ids_cap || ids_end == 0 {
        KEEP[2].fetch_add(1, Ordering::Relaxed);
        return;
    }
    unsafe {
        (ids_end as *mut u32).write_volatile(rd32(ent + 0x220));
        (rbp.wrapping_sub(0xd8) as *mut u64).write_volatile(ids_end + 4);
        (end as *mut u64).write_volatile(ent);
        ((r15 + 0x50) as *mut u64).write_volatile(end + 8);
    }
    let n = KEEP[1].fetch_add(1, Ordering::Relaxed) + 1;
    if n <= 3 || n <= diag_limit() {
        crate::info!("vg scene: entity {:x} kept in the active list at flip {now} ({} flip(s) after it last passed the view test)", rd32(ent + 0x220), now - last);
    }
}

/// Diagnosis (`DS_VG_DIAG`): the spawn grid's centre (eboot+0xa5c37d, the branch of the
/// per-frame update that stores a new centre position and hexagonal cell and marks the frame's
/// spawn data for a rebuild; rdi its state, the cell stored last at [rdi+0x2C] (two ints) and
/// the position at [rdi+0x20], rcx the double buffer's index times 9). The branch also runs
/// every frame when the grid follows the camera; a line when the stored cell differs from the
/// last one seen, and the count of branches taken.
static CELL_BRANCHES: AtomicU64 = AtomicU64::new(0);
static LAST_CELL: AtomicU64 = AtomicU64::new(u64::MAX);
pub extern "sysv64" fn cell_hook(regs: *const u64, _arg: u32) {
    let r = |i: usize| unsafe { *regs.add(i) };
    let (rcx, rdi) = (r(1), r(7));
    CELL_BRANCHES.fetch_add(1, Ordering::Relaxed);
    let cell = rd64(rdi + 0x2c);
    if LAST_CELL.swap(cell, Ordering::Relaxed) == cell {
        return;
    }
    static LOGGED: AtomicU64 = AtomicU64::new(0);
    if LOGGED.fetch_add(1, Ordering::Relaxed) < diag_limit() {
        let f = |o: u64| f32::from_bits(rd32(rdi + o));
        crate::info!(
            "vg spawn grid: rebuild marked at flip {} (the cell stored before: {},{}; buffer {}; the position stored before {:.2} {:.2} {:.2}; {} branches so far)",
            flip(),
            cell as u32 as i32,
            (cell >> 32) as u32 as i32,
            rcx / 9,
            f(0x20),
            f(0x24),
            f(0x28),
            CELL_BRANCHES.load(Ordering::Relaxed)
        );
    }
}

/// The status report's line.
pub fn status() -> String {
    let v = |i: usize| KEEP[i].load(Ordering::Relaxed);
    format!(
        "    VG active entities (`DS_VG_KEEP` {} flips): {} passes of the view test, {} kept after failing it, {} not kept for want of room, {} not kept without models; {} numbering resets\n",
        keep_frames(),
        v(0),
        v(1),
        v(2),
        v(3),
        RESETS.load(Ordering::Relaxed)
    )
}

/// The rebuild's spawn enable store (`mov [r15+1],al`): during a hold, al = 0 and the
/// title's value kept for `list_hook` to put back. (The hook trampoline restores the
/// registers from the block the handler sees, so the change reaches the instruction.)
pub extern "sysv64" fn enable_hook(regs: *const u64, _arg: u32) {
    let regs = regs as *mut u64;
    let ctx = unsafe { *regs.add(15) };
    let rax = unsafe { *regs };
    let al = (rax & 0xff) as u8;
    static SEEN: AtomicU64 = AtomicU64::new(0);
    if SEEN.fetch_add(1, Ordering::Relaxed) < diag_limit() {
        crate::info!("vg diag: spawn enable {al} at flip {}", flip());
    }
    if hold_frames() == 0 {
        return;
    }
    let mut holds = HOLDS.lock();
    let Some(h) = holds.iter_mut().find(|h| h.ctx == ctx) else { return };
    if h.since >= hold_frames() {
        return;
    }
    h.saved = Some(al);
    if al != 0 {
        unsafe { *regs = rax & !0xff };
        let n = HELD.fetch_add(1, Ordering::Relaxed) + 1;
        if n <= 3 || n <= diag_limit() {
            crate::info!("vg scene: spawning held at flip {} for {} frame(s)", flip(), hold_frames());
        }
    }
}
