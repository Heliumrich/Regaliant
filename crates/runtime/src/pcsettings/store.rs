//! The PC settings' values, in the title's own settings paths (PPSA01341 01.007.000;
//! FINDINGS.md 2026-11-06 afternoon). The title's settings store (`[[ctx+0x10]+0xA155F0]`, built
//! at eboot+0xbd0120) keeps vectors of 12-byte {key hash, sub-key, value} entries: bools at
//! +0x18, ints at +0x40, floats at +0x90 (begin, end); a key is FNV-1a of the setting's name in
//! upper case. The menu's bindings (`UISetting_<type>::Evaluate`: int eboot+0xc241f0, float
//! +0xc24cc0) scan them and fall back to the item's default; the menu's setters (int
//! eboot+0xfa5cb3, float +0xfa5ede) post {key, sub-key, value} to the store, which keeps only
//! the settings it registers. For the PC keys the port answers the scans (the hooks hand the
//! scan a one-entry store with the port's value, so it takes the title's own "found" path) and
//! takes the setters' values, keeps them in local-data/pcsettings.txt and applies them at once.
//! PC toggles are int choices (Off, On): a bool item also scans the store itself
//! (eboot+0xfa5990) and through further paths, and shows no value without an entry.

use parking_lot::Mutex;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Kind {
    Int,
    Float,
}

/// A PC setting: its key name, kind and default (the value's 32 bits).
#[derive(Clone, Debug)]
pub struct Def {
    pub name: String,
    pub kind: Kind,
    pub default: u32,
    /// A row that shows the achievements (answered live from them, never stored).
    pub live: Option<Live>,
}

/// What a live row shows.
#[derive(Clone, Copy, Debug)]
pub enum Live {
    /// A trophy unlocked (1) or not.
    Trophy(u32),
    /// The unlocked trophies of a grade (P, G, S, B).
    Count(char),
}

struct Store {
    path: Option<PathBuf>,
    defs: HashMap<u32, Def>,
    values: HashMap<u32, u32>,
}

static STORE: Mutex<Option<Store>> = Mutex::new(None);
/// What the hooks saw: [-, int scans answered, float, PC values taken, title values seen].
static STATS: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];

fn fmt(kind: Kind, bits: u32) -> String {
    match kind {
        Kind::Int => (bits as i32).to_string(),
        Kind::Float => {
            let t = format!("{:.4}", f32::from_bits(bits));
            t.trim_end_matches('0').trim_end_matches('.').to_string()
        }
    }
}

fn parse(kind: Kind, s: &str) -> Option<u32> {
    match kind {
        Kind::Int => match s {
            "true" | "on" => Some(1),
            "false" | "off" => Some(0),
            _ => s.parse::<i32>().ok().map(|v| v as u32),
        },
        Kind::Float => s.parse::<f32>().ok().filter(|v| v.is_finite()).map(f32::to_bits),
    }
}

/// Loads the values (`name = value` lines; unknown names and bad values are logged and left
/// out) for these definitions.
pub fn init(path: PathBuf, defs: Vec<Def>) {
    let mut values = HashMap::new();
    let by_name: HashMap<String, (u32, Kind)> = defs.iter().map(|d| (d.name.clone(), (super::setting_hash(&d.name), d.kind))).collect();
    if let Ok(text) = std::fs::read_to_string(&path) {
        for line in text.lines().map(str::trim).filter(|l| !l.is_empty() && !l.starts_with('#')) {
            let Some((k, v)) = line.split_once('=') else { continue };
            match by_name.get(k.trim()) {
                Some(&(h, kind)) => match parse(kind, v.trim()) {
                    Some(bits) => {
                        values.insert(h, bits);
                    }
                    None => crate::warn!("pc settings: {}: bad value '{}' for {k}", path.display(), v.trim()),
                },
                None => crate::warn!("pc settings: {}: unknown setting '{}'", path.display(), k.trim()),
            }
        }
    }
    let n = values.len();
    let defs: HashMap<u32, Def> = defs.into_iter().map(|d| (super::setting_hash(&d.name), d)).collect();
    crate::info!("pc settings: {} settings, {n} set in {}", defs.len(), path.display());
    *STORE.lock() = Some(Store { path: Some(path), defs, values });
}

/// The value of a PC key (its default when unset); `None` for other keys.
pub fn get(hash: u32) -> Option<u32> {
    let g = STORE.lock();
    let s = g.as_ref()?;
    let d = s.defs.get(&hash)?;
    match d.live {
        Some(Live::Trophy(id)) => return Some(crate::achievements::unlocked(id) as u32),
        Some(Live::Count(g)) => return Some(crate::achievements::count(g).0),
        None => {}
    }
    Some(s.values.get(&hash).copied().unwrap_or(d.default))
}

/// The value of a PC setting by name.
pub fn value(name: &str) -> Option<u32> {
    get(super::setting_hash(name))
}

fn set(hash: u32, bits: u32) -> bool {
    let mut g = STORE.lock();
    let Some(s) = g.as_mut() else { return false };
    let Some(d) = s.defs.get(&hash).cloned() else { return false };
    if d.live.is_some() {
        // A row showing the achievements; the menu cannot change it.
        return true;
    }
    if s.values.get(&hash) == Some(&bits) {
        return true;
    }
    s.values.insert(hash, bits);
    crate::info!("pc settings: {} = {}", d.name, fmt(d.kind, bits));
    if let Some(p) = &s.path {
        let mut names: Vec<(&String, Kind, u32)> = s.values.iter().filter_map(|(h, v)| s.defs.get(h).map(|d| (&d.name, d.kind, *v))).collect();
        names.sort();
        let text: String = std::iter::once("# PC settings (the game's SETTINGS screen, PC tab); written by the port\n".to_string())
            .chain(names.iter().map(|(n, k, v)| format!("{n} = {}\n", fmt(*k, *v))))
            .collect();
        if let Err(e) = std::fs::write(p, text) {
            crate::warn!("pc settings: write {}: {e}", p.display());
        }
    }
    true
}

fn reg(regs: *const u64, i: usize) -> u64 {
    unsafe { *regs.add(i) }
}

fn set_reg(regs: *const u64, i: usize, v: u64) {
    unsafe { *(regs as *mut u64).add(i) = v };
}

/// A store of one entry for a scan the port answers: each vector (bools +0x18, ints +0x40,
/// floats +0x90) holds the same {key, sub-key, value}, so the scan takes the title's own
/// "found" path with the PC value. One a thread (the scan follows its hook at once).
#[repr(C, align(16))]
struct OneEntry {
    store: [u64; 0x28],
    entry: [u32; 4],
}

thread_local! {
    static ONE: std::cell::UnsafeCell<Box<OneEntry>> = std::cell::UnsafeCell::new(Box::new(OneEntry { store: [0; 0x28], entry: [0; 4] }));
}

/// The address of this thread's one-entry store holding {key, sub-key, value}.
fn one_entry(key: u32, sub: u32, value: u32) -> u64 {
    ONE.with(|c| {
        let o = unsafe { &mut **c.get() };
        o.entry = [key, sub, value, 0];
        let e = o.entry.as_ptr() as u64;
        for (begin, end) in [(0x18, 0x20), (0x40, 0x48), (0x90, 0x98)] {
            o.store[begin / 8] = e;
            o.store[end / 8] = e + 12;
        }
        o.store.as_ptr() as u64
    })
}

/// A scan of the store for `key` (with `sub` its sub-key) in register `store_reg`: for a PC key
/// the register is pointed at a one-entry store with the port's value.
fn answer(regs: *const u64, key: u32, sub: u32, store_reg: usize, stat: usize) {
    if let Some(v) = get(key) {
        STATS[stat].fetch_add(1, Ordering::Relaxed);
        set_reg(regs, store_reg, one_entry(key, sub, v));
    }
}

/// `UISetting_int::Evaluate` before its scan (eboot+0xc24245; the same registers).
pub extern "sysv64" fn eval_int_hook(regs: *const u64, _arg: u32) {
    answer(regs, reg(regs, 12) as u32, reg(regs, 3) as u32, 13, 1);
}

/// `UISetting_float::Evaluate` before its scan (eboot+0xc24d15; the same registers).
pub extern "sysv64" fn eval_float_hook(regs: *const u64, _arg: u32) {
    answer(regs, reg(regs, 12) as u32, reg(regs, 3) as u32, 13, 2);
}

/// A PC setting by name (`pcset`; the value as in pcsettings.txt).
pub fn set_by_name(name: &str, value: &str) -> Result<(), String> {
    let h = super::setting_hash(name);
    let kind = STORE.lock().as_ref().and_then(|s| s.defs.get(&h).map(|d| d.kind)).ok_or_else(|| format!("no PC setting {name}"))?;
    let bits = parse(kind, value).ok_or_else(|| format!("bad value '{value}' for {name}"))?;
    set(h, bits);
    super::apply();
    Ok(())
}

fn taken(key: u32, bits: u32, kind: &str) {
    if set(key, bits) {
        super::apply();
        STATS[3].fetch_add(1, Ordering::Relaxed);
    } else if STATS[4].fetch_add(1, Ordering::Relaxed) < 64 {
        let name = super::setting_name(key).unwrap_or_else(|| format!("#{key:08x}"));
        crate::info!("pc settings: the menu set {name} ({kind}) = {bits:#x}");
    }
}

/// The int setter (eboot+0xfa5cba: r12d the key, r14d the value).
pub extern "sysv64" fn set_int_hook(regs: *const u64, _arg: u32) {
    taken(reg(regs, 12) as u32, reg(regs, 14) as u32, "int");
}

/// The float setter (eboot+0xfa5ee5: r13d the key, the value at [r15+0x10]).
pub extern "sysv64" fn set_float_hook(regs: *const u64, _arg: u32) {
    let v = unsafe { ((reg(regs, 15) + 0x10) as *const u32).read_volatile() };
    taken(reg(regs, 13) as u32, v, "float");
}

pub fn status() -> String {
    let s: Vec<u64> = STATS.iter().map(|a| a.load(Ordering::Relaxed)).collect();
    if s.iter().all(|&v| v == 0) {
        return String::new();
    }
    format!("    pc settings: scans answered (int/float) {}/{}, PC values taken {}, title values seen {}\n", s[1], s[2], s[3], s[4])
}
