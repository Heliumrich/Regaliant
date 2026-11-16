//! The title's static models on its two draw paths: a census (`DS_CPUDRAW_CENSUS=1`) and
//! switches for its investigation (`DS_CPU_DRAW_PATH`; diagnosis: modes 1 and 4 lose geometry).
//! Its static model sections hand each item of a kind-4 program to the GPU-driven batches when
//! they are built (eboot+0xa21e10: bit 0 of the item's +0x30, which the CPU draw path then
//! skips, and a 0x60-byte record per item, its pass mask at +0x38, from which `BuildXPR` makes
//! the section's instance arrays); its CPU draw path walks the models of each pass's list per
//! frame (eboot+0xab4df0: eboot+0xab4210 a model's chosen LOD, eboot+0xab4040 LOD 0 whole;
//! eboot+0xabac10 the per-item key of what it adds).
//!
//! Why: each of the batches' items costs the port a dispatch to record (the per-item cull cs
//! 0x1008994d00, ~1,050 a flip at Castlewall, beside their copies), where the CPU path would
//! draw a visible item with one draw: with the batches' items left out (modes 1 and 4) the
//! front end records ~5,000 commands a flip at Castlewall against ~9,400 and turns at 56-60
//! flips/s against 47-49, but the CPU path then draws none of the trebuchets, the brazier's
//! bowl, the burning tower's frame or the buildings behind (FINDINGS.md 2026-11-16 morning):
//! not because it skips their items by bit 0 (cleared, mode 4), their pass flags (set, measured
//! and removed), a missing record (present, mode 4) or stale item boxes (none outside their
//! model's: measured and removed), but because the cell builder that builds a section
//! (eboot+0xe97880) then removes the entities it absorbed from the scene (its +0x1c8 list,
//! eboot+0xaaafc0): at Castlewall's boot 59 sections absorbed 35,658 entities, ~94.5% of the
//! handed-off items. The batches are their only path: skipping that removal as well (mode 10,
//! measured and removed) loses the same geometry, so the absorbed models are not in the CPU
//! path's lists either (their LOD count reads zero after the build).

use parking_lot::Mutex;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};

/// Models whose items were handed off: (item count, an item's address, the model's flags
/// (+0x198) and +0x158 at its first hand-off).
static HANDED: Mutex<Option<HashMap<u64, (u32, u64, u64, u64)>>> = Mutex::new(None);
/// Each handed-off model's first 0x200 bytes at its first hand-off (64 words).
static AT_BAKE: Mutex<Option<HashMap<u64, Box<[u64; 64]>>>> = Mutex::new(None);

/// Every item handed off (census) and every item the CPU path put in a draw list
/// (eboot+0xabac10, the per-item key both adders take before pushing an item).
static HANDED_ITEMS: Mutex<Option<HashSet<u64>>> = Mutex::new(None);
static ADDED: Mutex<Option<HashSet<u64>>> = Mutex::new(None);

/// At eboot+0xabac10's entry: the item in rdi.
pub extern "sysv64" fn added_hook(regs: *const u64, _arg: u32) {
    let item = unsafe { *regs.add(7) };
    ADDED.lock().get_or_insert_with(HashSet::new).insert(item);
}

/// Models the CPU path walked: [eboot+0xab4210 (a chosen LOD), eboot+0xab4040 (whole),
/// eboot+0xab4e7d (every model of a pass's list, before its culling)].
static WALKED: [Mutex<Option<HashSet<u64>>>; 3] = [const { Mutex::new(None) }; 3];
/// [hand-offs, eboot+0xab4210 calls, eboot+0xab4040 calls].
static CALLS: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

fn rd64(a: u64) -> Option<u64> {
    (a >= 0x10000 && crate::mem::readable(a, 8)).then(|| unsafe { std::ptr::read_volatile(a as *const u64) })
}

fn rd16(a: u64) -> Option<u16> {
    (a >= 0x10000 && crate::mem::readable(a, 2)).then(|| unsafe { std::ptr::read_volatile(a as *const u16) })
}

fn rdf(a: u64) -> f32 {
    if a >= 0x10000 && crate::mem::readable(a, 4) {
        unsafe { std::ptr::read_volatile(a as *const f32) }
    } else {
        f32::NAN
    }
}

/// `DS_CPU_DRAW_PATH`: 0 the title's hand-off as it is, 1 no item handed off (the stand-in of
/// 2026-10-10), 3 every item handed off and given to the CPU path too (its bit 0 cleared after
/// the hand-off: both paths draw it), 4 as 3 with the item's hand-off records given no pass
/// (`epilogue_hook`). (2, the items of models linked into the scene at their section's build
/// kept on the CPU path, and 5, 4 with the models' CPU pass bits set, were measured on
/// 2026-11-16 and removed: TRIED.md; so were 6 and 7 (item boxes), 8 (models with a LOD count:
/// all have one at the build) and 9 (models the section does not absorb: 5.5% of the items).)
fn mode() -> u8 {
    static M: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *M.get_or_init(|| std::env::var("DS_CPU_DRAW_PATH").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(0))
}

/// `DS_CPU_DRAW_REGION=x/y/z/r` (with mode 1): only models whose box center (+0x128) lies within
/// r of (x, y, z) are kept off the batches.
fn in_region(model: u64) -> bool {
    static R: std::sync::OnceLock<Option<[f32; 4]>> = std::sync::OnceLock::new();
    let Some(r) = *R.get_or_init(|| {
        std::env::var("DS_CPU_DRAW_REGION").ok().and_then(|v| {
            let f: Vec<f32> = v.replace([';', ':', ','], "/").split('/').filter_map(|x| x.trim().parse().ok()).collect();
            (f.len() >= 4).then(|| [f[0], f[1], f[2], f[3]])
        })
    }) else {
        return true;
    };
    let Some(b) = rd64(model + 0x128) else { return false };
    let c = [(rdf(b) + rdf(b + 12)) / 2.0, (rdf(b + 4) + rdf(b + 16)) / 2.0, (rdf(b + 8) + rdf(b + 20)) / 2.0];
    ((c[0] - r[0]).powi(2) + (c[1] - r[1]).powi(2) + (c[2] - r[2]).powi(2)).sqrt() <= r[3]
}

fn census() -> bool {
    static C: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *C.get_or_init(|| std::env::var("DS_CPUDRAW_CENSUS").ok().as_deref() == Some("1"))
}

/// A mesh part whose program (+0x20) is not of kind 4 (+0x3d8 zero): the hand-off's check, made
/// on it instead of the item's part, skips the hand-off (`handoff_hook`).
static STAND_PROG: [u64; 0x80] = [0; 0x80];
fn stand_part() -> u64 {
    static P: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *P.get_or_init(|| {
        let mut b = Box::new([0u64; 8]);
        b[4] = STAND_PROG.as_ptr() as u64;
        Box::leak(b).as_ptr() as u64
    })
}

/// Items kept on the CPU path and models they belong to (`DS_CPU_DRAW_PATH`), items handed off;
/// modes 3 and 4: items handed off whose bit 0 was cleared, records given no pass.
static KEPT: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// At eboot+0xa222a8 (FUN_00a21e10's epilogue, `add rsp,68h; pop rbx`; the item at [rbp-0x70],
/// the collector at [rbp-0x30] with its records' vector at +0x58..+0x60, a record's item at
/// +0x18 and its pass mask at +0x38): modes 3 and 4 clear bit 0 of an item this call handed
/// off (the CPU draw path then draws it), mode 4 also zeroes the pass masks of its records.
pub extern "sysv64" fn epilogue_hook(regs: *const u64, _arg: u32) {
    let m = mode();
    if m != 3 && m != 4 {
        return;
    }
    let rbp = unsafe { *regs.add(5) };
    let item = unsafe { std::ptr::read_volatile(rbp.wrapping_sub(0x70) as *const u64) };
    let coll = unsafe { std::ptr::read_volatile(rbp.wrapping_sub(0x30) as *const u64) };
    let (Some(begin), Some(end)) = (rd64(coll + 0x58), rd64(coll + 0x60)) else { return };
    let mut found = false;
    let mut at = end;
    for _ in 0..2 {
        if at < begin + 8 {
            break;
        }
        at -= 8;
        let Some(rec) = rd64(at) else { break };
        if rd64(rec + 0x18) != Some(item) {
            break;
        }
        found = true;
        if m == 4 {
            unsafe { std::ptr::write_volatile((rec + 0x38) as *mut u32, 0) };
            KEPT[3].fetch_add(1, Ordering::Relaxed);
        }
    }
    if found && crate::mem::readable(item + 0x30, 1) {
        unsafe {
            let b = (item + 0x30) as *mut u8;
            std::ptr::write_volatile(b, std::ptr::read_volatile(b) & !1);
        }
        KEPT[2].fetch_add(1, Ordering::Relaxed);
    }
}

/// At eboot+0xa21e6c (the hand-off's program check, `mov rax,[rbx+20h]; cmp dword ptr
/// [rax+3D8h],4`, then `jne` to the epilogue; rbx the item's mesh part, popped there): an item
/// the hand-off takes (its program of kind 4 at +0x3d8, nothing at +0x60, no flag 0x10 at
/// +0x3f0, nothing at +0x3e4), with the item at [rbp-0x70] and its model at [[[rbp-0x68]]].
/// One the CPU path keeps gets rbx pointed at `stand_part` for the check alone.
pub extern "sysv64" fn handoff_hook(regs: *const u64, _arg: u32) {
    let (rbx, rbp) = unsafe { (*regs.add(3), *regs.add(5)) };
    let Some(prog) = rd64(rbx + 0x20) else { return };
    let rd32 = |a: u64| (a >= 0x10000 && crate::mem::readable(a, 4)).then(|| unsafe { std::ptr::read_volatile(a as *const u32) });
    let takes = rd32(prog + 0x3d8) == Some(4) && rd64(prog + 0x60) == Some(0) && rd32(prog + 0x3f0).is_some_and(|f| f & 0x10 == 0) && rd32(prog + 0x3e4) == Some(0);
    if !takes {
        return;
    }
    let item = unsafe { std::ptr::read_volatile(rbp.wrapping_sub(0x70) as *const u64) };
    let p2 = unsafe { std::ptr::read_volatile(rbp.wrapping_sub(0x68) as *const u64) };
    let model = rd64(p2).and_then(rd64);
    let keep = mode() == 1 && model.is_some_and(in_region);
    if keep {
        unsafe { *(regs as *mut u64).add(3) = stand_part() };
        KEPT[0].fetch_add(1, Ordering::Relaxed);
    } else {
        KEPT[1].fetch_add(1, Ordering::Relaxed);
    }
    if !census() {
        return;
    }
    CALLS[0].fetch_add(1, Ordering::Relaxed);
    HANDED_ITEMS.lock().get_or_insert_with(HashSet::new).insert(item);
    let Some(model) = model else { return };
    let (flags, p158) = (rd64(model + 0x198).unwrap_or(0), rd64(model + 0x158).unwrap_or(0));
    let mut g = HANDED.lock();
    let e = g.get_or_insert_with(HashMap::new).entry(model).or_insert((0, item, flags, p158));
    e.0 += 1;
    if e.0 == 1 && crate::mem::readable(model, 0x200) {
        let mut w = Box::new([0u64; 64]);
        for (k, x) in w.iter_mut().enumerate() {
            *x = unsafe { std::ptr::read_volatile((model + 8 * k as u64) as *const u64) };
        }
        AT_BAKE.lock().get_or_insert_with(HashMap::new).insert(model, w);
    }
}

/// Per model: the pass kinds whose list held it (bit = kind, `listed_hook`) and the LODs
/// eboot+0xab4210 was asked for (bit = LOD index, `lod_hook`).
static KINDS: Mutex<Option<HashMap<u64, (u32, u32)>>> = Mutex::new(None);

/// At eboot+0xab4210's entry: the model in rsi, the LOD index in edx. (Its per-item test
/// is eboot+0x99c770 on the item's world box at +0x18, which eboot+0xabaca0 writes from its
/// local box and a transform; then bit 0.)
pub extern "sysv64" fn lod_hook(regs: *const u64, _arg: u32) {
    let (model, lod) = unsafe { (*regs.add(6), *regs.add(2) as u32) };
    CALLS[1].fetch_add(1, Ordering::Relaxed);
    WALKED[0].lock().get_or_insert_with(HashSet::new).insert(model);
    KINDS.lock().get_or_insert_with(HashMap::new).entry(model).or_insert((0, 0)).1 |= 1 << lod.min(31);
}

/// At eboot+0xab4e7d (FUN_00ab4df0's loop over a pass's model list, before its pass checks of
/// the model's flags at +0x198: pass kind 4 bit 7, kind 3 bit 6, kinds 1-2 bit 1, others bit
/// 0, while the title's settings bytes there are zero): the model at [r13].
pub extern "sysv64" fn listed_hook(regs: *const u64, _arg: u32) {
    let (r13, r15) = unsafe { (*regs.add(13), *regs.add(15)) };
    let Some(model) = rd64(r13) else { return };
    WALKED[2].lock().get_or_insert_with(HashSet::new).insert(model);
    // (The pass's kind at [[r15+0x70]+0x30], eboot+0xab4df0's iVar3.)
    let kind = rd64(r15 + 0x70).and_then(|v| rd16(v + 0x30)).unwrap_or(31) as u32;
    KINDS.lock().get_or_insert_with(HashMap::new).entry(model).or_insert((0, 0)).0 |= 1 << kind.min(31);
}

/// At eboot+0xab4040's entry: the model in rdi.
pub extern "sysv64" fn whole_hook(regs: *const u64, _arg: u32) {
    let model = unsafe { *regs.add(7) };
    CALLS[2].fetch_add(1, Ordering::Relaxed);
    WALKED[1].lock().get_or_insert_with(HashSet::new).insert(model);
}

/// Whether `model` still holds `item` in one of its LOD entries (table at +0x130, 0x48 bytes
/// each, their count at +0x14a; an entry's items at +8, 0x38 bytes each, their count at +0x12):
/// a model freed since its hand-off (a level reloaded) does not.
fn live(model: u64, item: u64) -> bool {
    let (Some(t), Some(n)) = (rd64(model + 0x130), rd16(model + 0x14a)) else { return false };
    (0..(n & 0xff) as u64).any(|k| {
        let e = t + k * 0x48;
        match (rd64(e + 8), rd16(e + 0x12)) {
            (Some(items), Some(c)) => item >= items && item < items + c as u64 * 0x38,
            _ => false,
        }
    })
}

fn describe(m: u64) -> String {
    let b = rd64(m + 0x128).unwrap_or(0);
    let c = [(rdf(b) + rdf(b + 12)) / 2.0, (rdf(b + 4) + rdf(b + 16)) / 2.0, (rdf(b + 8) + rdf(b + 20)) / 2.0];
    format!("{m:#x} flags {:#x} +158 {:#x} lods {} center ({:.1}, {:.1}, {:.1})", rd64(m + 0x198).unwrap_or(0), rd64(m + 0x158).unwrap_or(0), rd16(m + 0x14a).unwrap_or(0) & 0xff, c[0], c[1], c[2])
}

/// The status lines: the items kept on the CPU path, the census ("" without either).
pub fn status() -> String {
    let k: Vec<u64> = KEPT.iter().map(|x| x.load(Ordering::Relaxed)).collect();
    let kept = if mode() != 0 { format!("    static models on the CPU draw path (`DS_CPU_DRAW_PATH` {}): {} items kept off the GPU-driven batches, {} handed to them; {} handed off and given to the CPU path too, {} records left with no pass
", mode(), k[0], k[1], k[2], k[3]) } else { String::new() };
    kept + &census_status()
}

fn census_status() -> String {
    let c: Vec<u64> = CALLS.iter().map(|x| x.load(Ordering::Relaxed)).collect();
    if c[0] == 0 && c[1] == 0 {
        return String::new();
    }
    let handed = HANDED.lock().clone().unwrap_or_default();
    let w0 = WALKED[0].lock().clone().unwrap_or_default();
    let w1 = WALKED[1].lock().clone().unwrap_or_default();
    let w2 = WALKED[2].lock().clone().unwrap_or_default();
    let (mut listed, mut listed_items, mut unlisted_near) = (0u64, 0u64, Vec::new());
    let mut listed_models: Vec<u64> = Vec::new();
    let (mut alive, mut items, mut lod, mut whole, mut neither, mut neither_items) = (0u64, 0u64, 0u64, 0u64, 0u64, 0u64);
    let mut samples: Vec<String> = Vec::new();
    let mut walked_samples: Vec<String> = Vec::new();
    let mut keys: Vec<(&u64, &(u32, u64, u64, u64))> = handed.iter().collect();
    keys.sort_by(|a, b| b.1 .0.cmp(&a.1 .0));
    // `DS_CPUDRAW_NEAR=x,y,z`: the never-walked models listed by their bounds' distance from there.
    let near_at: Option<[f32; 3]> = std::env::var("DS_CPUDRAW_NEAR").ok().and_then(|v| {
        let f: Vec<f32> = v.replace([';', ':', '/'], ",").split(',').filter_map(|x| x.trim().parse().ok()).collect();
        (f.len() >= 3).then(|| [f[0], f[1], f[2]])
    });
    let dist = |m: u64| -> f32 {
        let Some(p) = near_at else { return 0.0 };
        let b = rd64(m + 0x128).unwrap_or(0);
        let c = [(rdf(b) + rdf(b + 12)) / 2.0, (rdf(b + 4) + rdf(b + 16)) / 2.0, (rdf(b + 8) + rdf(b + 20)) / 2.0];
        ((c[0] - p[0]).powi(2) + (c[1] - p[1]).powi(2) + (c[2] - p[2]).powi(2)).sqrt()
    };
    let mut near: Vec<(f32, u64, u32)> = Vec::new();
    // (flags at the hand-off, +0x158 at the hand-off zero) -> [walked models, items, never walked, items]
    let mut by_flags: HashMap<(u64, u64), [u64; 4]> = HashMap::new();
    for (&m, &(n, item, flags0, p158)) in keys {
        // (A model freed since: its item no longer in its LOD table, or flags not of a model.)
        let flags = rd64(m + 0x198).unwrap_or(0);
        let lods = rd16(m + 0x14a).unwrap_or(0) & 0xff;
        if !live(m, item) || flags >> 48 != 0x17 || flags0 >> 48 != 0x17 || !(1..=12).contains(&lods) {
            continue;
        }
        alive += 1;
        items += n as u64;
        let (a, b) = (w0.contains(&m), w1.contains(&m));
        if w2.contains(&m) {
            listed += 1;
            listed_items += n as u64;
            listed_models.push(m);
        } else {
            unlisted_near.push((dist(m), m, n));
        }
        lod += a as u64;
        whole += b as u64;
        let _ = p158;
        let e = by_flags.entry((flags0, flags)).or_insert([0; 4]);
        if a || b {
            e[0] += 1;
            e[1] += n as u64;
        } else {
            e[2] += 1;
            e[3] += n as u64;
        }
        if !a && !b {
            neither += 1;
            neither_items += n as u64;
            near.push((dist(m), m, n));
            if near_at.is_none() && samples.len() < 16 {
                samples.push(format!("{} ({n} items)", describe(m)));
            }
        } else if walked_samples.len() < 6 {
            walked_samples.push(format!("{} ({n} items)", describe(m)));
        }
    }
    if near_at.is_some() {
        near.sort_by(|a, b| a.0.total_cmp(&b.0));
        samples = near.iter().take(40).map(|&(d, m, n)| format!("{} ({n} items, {d:.0} m)", describe(m))).collect();
    }
    // Words of the models at their bake that separate the listed ones from the never-listed:
    // per offset [listed zero, listed non-zero, never zero, never non-zero], and the most common
    // value in each group with its share.
    let bake = AT_BAKE.lock().clone().unwrap_or_default();
    let mut sep: Vec<String> = Vec::new();
    {
        let unl: HashSet<u64> = unlisted_near.iter().map(|x| x.1).collect();
        let lis: Vec<u64> = listed_models.iter().copied().filter(|m| bake.contains_key(m)).collect();
        let nev: Vec<u64> = unl.iter().copied().filter(|m| bake.contains_key(m)).collect();
        if !lis.is_empty() && !nev.is_empty() {
            let mut rows: Vec<(f64, String)> = Vec::new();
            for k in 0..64usize {
                let top = |ms: &[u64]| -> (u64, f64) {
                    let mut h: HashMap<u64, u64> = HashMap::new();
                    for m in ms {
                        *h.entry(bake[m][k]).or_insert(0) += 1;
                    }
                    let (v, n) = h.into_iter().max_by_key(|x| x.1).unwrap_or((0, 0));
                    (v, n as f64 / ms.len().max(1) as f64)
                };
                let zl = lis.iter().filter(|m| bake[*m][k] == 0).count() as f64 / lis.len() as f64;
                let zn = nev.iter().filter(|m| bake[*m][k] == 0).count() as f64 / nev.len() as f64;
                let (vl, sl) = top(&lis);
                let (vn, sn) = top(&nev);
                // How well "zero" or "the never-listed group's common value" splits them.
                let score = (zl - zn).abs().max(if vl != vn { sn * (1.0 - lis.iter().filter(|m| bake[*m][k] == vn).count() as f64 / lis.len() as f64) } else { 0.0 });
                rows.push((score, format!("+{:#x}: zero listed {:.2} never {:.2}; common listed {vl:#x} ({sl:.2}) never {vn:#x} ({sn:.2})", 8 * k, zl, zn)));
            }
            rows.sort_by(|a, b| b.0.total_cmp(&a.0));
            sep = rows.into_iter().take(12).map(|r| format!("{:.2} {}", r.0, r.1)).collect();
            sep.insert(0, format!("{} listed and {} never-listed models at their bake", lis.len(), nev.len()));
        }
    }
    // Live models whose handed-off items the CPU path never put in a draw list: their LOD and
    // fade fields (+0x14b and +0x17c/+0x17d LODs, +0x174 and +0x180.. fades).
    let hi = HANDED_ITEMS.lock().clone().unwrap_or_default();
    let kinds = KINDS.lock().clone().unwrap_or_default();
    let ad = ADDED.lock().clone().unwrap_or_default();
    let (mut items_h, mut items_a, mut none_models, mut none_items) = (0u64, 0u64, 0u64, 0u64);
    // Handed-off items of live models with bit 0 of +0x30 set now / clear now, and by LOD index.
    let (mut bit_set, mut bit_clear) = (0u64, 0u64);
    let mut by_lod: [[u64; 2]; 8] = [[0; 2]; 8];
    // Items by whether they have their own bounds (+0x18, which the CPU path's per-item frustum
    // test uses) and were added: [no bounds not added, no bounds added, bounds not added,
    // bounds added]; samples of never-added items' bounds.
    let mut by_bounds = [0u64; 4];
    let mut bound_samples: Vec<String> = Vec::new();
    let mut none: Vec<(f32, String)> = Vec::new();
    let mut some: Vec<(f32, String)> = Vec::new();
    for &m in listed_models.iter().chain(unlisted_near.iter().map(|x| &x.1)) {
        let (Some(t), Some(n)) = (rd64(m + 0x130), rd16(m + 0x14a)) else { continue };
        let (mut h, mut a) = (0u64, 0u64);
        for k in 0..(n & 0xff) as u64 {
            let e = t + k * 0x48;
            let (Some(items), Some(c)) = (rd64(e + 8), rd16(e + 0x12)) else { continue };
            for j in 0..c as u64 {
                let it = items + j * 0x38;
                if hi.contains(&it) {
                    h += 1;
                    let added = ad.contains(&it);
                    a += added as u64;
                    if rd16(it + 0x30).is_some_and(|v| v & 1 != 0) {
                        bit_set += 1;
                    } else {
                        bit_clear += 1;
                    }
                    by_lod[(k as usize).min(7)][added as usize] += 1;
                    let b = rd64(it + 0x18).unwrap_or(0);
                    by_bounds[(b != 0) as usize * 2 + added as usize] += 1;
                    if b != 0 && !added && bound_samples.len() < 12 && dist(m) < 40.0 {
                        bound_samples.push(format!("item {it:#x} lod {k} bounds {b:#x} [{:.1} {:.1} {:.1} | {:.1} {:.1} {:.1}] model center {:.0} m", rdf(b), rdf(b + 4), rdf(b + 8), rdf(b + 12), rdf(b + 16), rdf(b + 20), dist(m)));
                    }
                }
            }
        }
        items_h += h;
        items_a += a;
        let rd8 = |x: u64| rd16(x).map_or(-99, |v| (v & 0xff) as u8 as i8 as i32);
        let (kl, ll) = kinds.get(&m).copied().unwrap_or((0, 0));
        let fields = format!("{} kinds {kl:#x} lods asked {ll:#x} lod {} {} {} fade {:.2} [{:.2} {:.2} {:.2} {:.2}] +1a0 {:#x}", describe(m), rd8(m + 0x14b), rd8(m + 0x17c), rd8(m + 0x17d), rdf(m + 0x174), rdf(m + 0x180), rdf(m + 0x184), rdf(m + 0x188), rdf(m + 0x18c), rd64(m + 0x1a0).unwrap_or(0));
        if h > 0 && a == 0 {
            none_models += 1;
            none_items += h;
            none.push((dist(m), format!("{fields} ({h} items, {:.0} m)", dist(m))));
        } else if h > 0 {
            some.push((dist(m), format!("{fields} ({a}/{h} items added, {:.0} m)", dist(m))));
        }
    }
    none.sort_by(|a, b| a.0.total_cmp(&b.0));
    some.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut kh: HashMap<u32, u64> = HashMap::new();
    for (_, &(kl, _)) in kinds.iter() {
        *kh.entry(kl).or_insert(0) += 1;
    }
    let mut kh: Vec<(u32, u64)> = kh.into_iter().collect();
    kh.sort_by(|a, b| b.1.cmp(&a.1));
    let kinds_line = format!("    cpu draw census: models by the pass kinds listing them (mask: models): {}\n", kh.iter().map(|(k, n)| format!("{k:#x}: {n}")).collect::<Vec<_>>().join(", "));
    // Every recorded model within 15 m of the sort point, unfiltered.
    let mut raw: Vec<(f32, String)> = Vec::new();
    for (&m, &(n, item, flags0, _)) in handed.iter() {
        let d = dist(m);
        if !(d <= 15.0) {
            continue;
        }
        let flags = rd64(m + 0x198).unwrap_or(0);
        let lods = rd16(m + 0x14a).unwrap_or(0) & 0xff;
        let (kl, ll) = kinds.get(&m).copied().unwrap_or((0, 0));
        let item_added = ad.contains(&item);
        raw.push((d, format!("{} live {} flags0 {flags0:#x} lods {lods} listed {} walked {} kinds {kl:#x} asked {ll:#x} first item {item:#x} added {item_added} ({n} hand-offs, {d:.0} m)", describe(m), live(m, item), w2.contains(&m), w0.contains(&m) || w1.contains(&m))));
        let _ = flags;
    }
    raw.sort_by(|a, b| a.0.total_cmp(&b.0));
    let raw_line = format!("    cpu draw census: recorded models within 15 m ({}): {}\n", raw.len(), raw.iter().take(60).map(|x| x.1.clone()).collect::<Vec<_>>().join("; "));
    let added_line = raw_line + &kinds_line + &format!(
        "    cpu draw census: of the live models' handed-off items {items_h} (bit 0 set now {bit_set}, clear {bit_clear}; by LOD [not added, added] {:?}; by own bounds [none: not added, added; bounds: not added, added] {:?}: {}), {items_a} in a CPU draw list; models with none there {none_models} ({none_items} items): {}; models with some: {}
",
        by_lod,
        by_bounds,
        bound_samples.join("; "),
        none.iter().take(24).map(|x| x.1.clone()).collect::<Vec<_>>().join("; "),
        some.iter().take(8).map(|x| x.1.clone()).collect::<Vec<_>>().join("; ")
    );
    unlisted_near.sort_by(|a, b| a.0.total_cmp(&b.0));
    let unlisted: Vec<String> = unlisted_near.iter().take(30).map(|&(d, m, n)| format!("{} ({n} items, {d:.0} m)", describe(m))).collect();
    let mut fl: Vec<((u64, u64), [u64; 4])> = by_flags.into_iter().collect();
    fl.sort_by(|a, b| (b.1[1] + b.1[3]).cmp(&(a.1[1] + a.1[3])));
    let fl: Vec<String> = fl.iter().take(32).map(|((f0, f), v)| format!("{f0:#x} -> {f:#x} walked {} ({} items) never {} ({} items)", v[0], v[1], v[2], v[3])).collect();
    added_line + &format!(
        "    cpu draw census (`DS_CPUDRAW_CENSUS`): by flags at the hand-off: {}
    cpu draw census: words separating them: {}
    cpu draw census: handed-off models in a pass's list {listed} ({listed_items} items), never listed {} ({} items): {}
    cpu draw census: {} hand-offs, {} models handed to the GPU-driven batches, {alive} of them alive with {items} items; the CPU path walked {} models by LOD ({} calls) and {} whole ({} calls); of the live handed-off models walked by LOD {lod}, whole {whole}, by neither {neither} ({neither_items} items): {}; walked ones: {}\n",
        fl.join("; "),
        sep.join("; "),
        unlisted_near.len(),
        unlisted_near.iter().map(|x| x.2 as u64).sum::<u64>(),
        unlisted.join("; "),
        c[0],
        handed.len(),
        w0.len(),
        c[1],
        w1.len(),
        c[2],
        samples.join("; "),
        walked_samples.join("; ")
    )
}
