//! The title's console variables the PC settings set while the game runs (PPSA01341
//! 01.007.000; FINDINGS.md 2026-11-06 afternoon). Each is an object in the main executable's
//! data, built the first time code uses it (`ConVar` constructors: bool eboot+0x450a20, int
//! +0x431500, float +0x9260b0; each stores its vtable at +0 and its value at +0x68), so the port
//! writes the value once the object exists (its vtable set) and keeps the title's own value to
//! put back. Each object is checked by its name string first (another title version is left
//! alone and logged). The renderer reads the values each frame.

use parking_lot::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Kind {
    Bool,
    Int,
    Float,
}

pub struct Var {
    pub name: &'static str,
    /// Module-relative addresses of the name string and the object.
    string: u64,
    obj: u64,
    pub kind: Kind,
}

const fn v(name: &'static str, string: u64, obj: u64, kind: Kind) -> Var {
    Var { name, string, obj, kind }
}

pub const VARS: &[Var] = &[
    v("r_bloomEnable", 0x24ebf06, 0x2f0ba58, Kind::Bool),
    v("r_bloomOverrideEnable", 0x2479775, 0x2f0bc48, Kind::Bool),
    v("r_dofOverrideEnable", 0x2440f19, 0x2f0c8f8, Kind::Bool),
    v("r_dofFarAmount", 0x241a954, 0x2f0cce8, Kind::Float),
    v("r_dofNearAmount", 0x2423fb6, 0x2f0c9e8, Kind::Float),
    v("r_vignetteOverrideEnable", 0x23c6d08, 0x2f0c228, Kind::Bool),
    v("r_vignetteScale", 0x23fe798, 0x2f0c518, Kind::Float),
    v("r_enableCascadeShadows", 0x23eb9ca, 0x2f1cce0, Kind::Bool),
    v("r_enableSpotShadows", 0x23cfe9e, 0x2f1a190, Kind::Bool),
    v("r_contactShadowsEnabled", 0x24a95a9, 0x2f1c300, Kind::Bool),
    v("r_ssrMaxIterations", 0x2407e0b, 0x2f20008, Kind::Int),
    v("r_virtualGeometryScaleMultiplier", 0x2470e6c, 0x2f2deb8, Kind::Float),
    v("r_virtualGeometryNumShadowCascades", 0x23c6f14, 0x2f2dfb8, Kind::Int),
    v("r_aaHistorySharpeningStrength", 0x2424018, 0x2f0e0b8, Kind::Float),
    v("r_motionBlurAmount", 0x2451804, 0x2f0d0d8, Kind::Float),
    v("r_forceLod", 0x23ebbb8, 0x2f2fb30, Kind::Int),
];

const VALUE: u64 = 0x68;

struct State {
    /// Wanted value (bits) per var; `None`: the title's own.
    want: Vec<Option<u32>>,
    /// The title's own value, read before the first write.
    orig: Vec<Option<u32>>,
    /// What the port wrote last.
    wrote: Vec<Option<u32>>,
}

static BASE: AtomicU64 = AtomicU64::new(0);
static OK: [AtomicBool; 32] = [const { AtomicBool::new(false) }; 32];
static STATE: Mutex<Option<State>> = Mutex::new(None);
static DIRTY: AtomicBool = AtomicBool::new(false);

/// Checks each object's name in the main executable's image (`base` its load address).
pub fn init(base: u64, image: &[u8]) {
    BASE.store(base, Ordering::Relaxed);
    let mut bad = Vec::new();
    for (i, var) in VARS.iter().enumerate() {
        let s = var.string as usize;
        let ok = image.get(s..s + var.name.len() + 1).is_some_and(|b| &b[..var.name.len()] == var.name.as_bytes() && b[var.name.len()] == 0);
        OK[i].store(ok, Ordering::Relaxed);
        if !ok {
            bad.push(var.name);
        }
    }
    if bad.is_empty() {
        crate::info!("pc settings: {} title console variables found", VARS.len());
    } else {
        crate::warn!("pc settings: console variables not where expected (another title version?), left alone: {}", bad.join(", "));
    }
    *STATE.lock() = Some(State { want: vec![None; VARS.len()], orig: vec![None; VARS.len()], wrote: vec![None; VARS.len()] });
    // The stored PC settings (read before the executable loaded).
    super::apply();
}

fn index(name: &str) -> Option<usize> {
    VARS.iter().position(|v| v.name == name)
}

pub fn float(v: f32) -> u32 {
    v.to_bits()
}

/// Sets the wanted value of a variable (`None`: the title's own); applied at the next `tick`.
pub fn want(name: &str, bits: Option<u32>) {
    let Some(i) = index(name) else {
        crate::warn!("pc settings: no console variable {name}");
        return;
    };
    if let Some(s) = STATE.lock().as_mut() {
        if s.want[i] != bits {
            s.want[i] = bits;
            DIRTY.store(true, Ordering::Relaxed);
        }
    }
}

fn fmt(kind: Kind, bits: u32) -> String {
    match kind {
        Kind::Bool => ((bits & 0xff) != 0).to_string(),
        Kind::Int => (bits as i32).to_string(),
        Kind::Float => f32::from_bits(bits).to_string(),
    }
}

/// Writes the wanted values into the objects that exist (each flip, cheap when nothing waits).
pub fn tick() {
    if !DIRTY.load(Ordering::Relaxed) {
        return;
    }
    let base = BASE.load(Ordering::Relaxed);
    if base == 0 {
        return;
    }
    let mut g = STATE.lock();
    let Some(s) = g.as_mut() else { return };
    let mut waiting = false;
    for (i, var) in VARS.iter().enumerate() {
        if !OK[i].load(Ordering::Relaxed) {
            continue;
        }
        let obj = base + var.obj;
        let built = unsafe { (obj as *const u64).read_volatile() } != 0;
        let target = s.want[i].or(if s.wrote[i].is_some() { s.orig[i] } else { None });
        let Some(target) = target else { continue };
        if !built {
            waiting = true;
            continue;
        }
        let at = (obj + VALUE) as *mut u8;
        let cur = unsafe {
            match var.kind {
                Kind::Bool => at.read_volatile() as u32,
                _ => (at as *const u32).read_volatile(),
            }
        };
        if s.orig[i].is_none() {
            s.orig[i] = Some(cur);
        }
        if cur != target || s.wrote[i] != Some(target) {
            unsafe {
                match var.kind {
                    Kind::Bool => at.write_volatile((target & 0xff) as u8),
                    _ => (at as *mut u32).write_volatile(target),
                }
            }
            crate::info!(
                "pc settings: {} = {} (the title's own {})",
                var.name,
                fmt(var.kind, target),
                s.orig[i].map(|o| fmt(var.kind, o)).unwrap_or_default()
            );
            s.wrote[i] = if s.want[i].is_some() { Some(target) } else { None };
        }
    }
    if !waiting {
        DIRTY.store(false, Ordering::Relaxed);
    }
}

/// The title's own value of a variable, once read (for the log and the presets).
pub fn title_value(name: &str) -> Option<u32> {
    let i = index(name)?;
    STATE.lock().as_ref()?.orig[i]
}

/// Logs every variable: whether the title built it, its value, what the port wants (`convars`).
pub fn report() {
    let base = BASE.load(Ordering::Relaxed);
    let g = STATE.lock();
    let Some(s) = g.as_ref() else { return };
    let lines: Vec<String> = VARS
        .iter()
        .enumerate()
        .map(|(i, var)| {
            let obj = base + var.obj;
            let ok = OK[i].load(Ordering::Relaxed);
            let built = ok && base != 0 && unsafe { (obj as *const u64).read_volatile() } != 0;
            let cur = built.then(|| unsafe {
                let at = (obj + VALUE) as *const u8;
                match var.kind {
                    Kind::Bool => at.read_volatile() as u32,
                    _ => (at as *const u32).read_volatile(),
                }
            });
            format!(
                "{} {}",
                var.name,
                match (ok, cur) {
                    (false, _) => "not found".to_string(),
                    (true, None) => "not built yet".to_string(),
                    (true, Some(c)) => format!("= {}{}", fmt(var.kind, c), s.want[i].map(|w| format!(" (wanted {})", fmt(var.kind, w))).unwrap_or_default()),
                }
            )
        })
        .collect();
    crate::info!("pc settings: console variables: {}", lines.join("; "));
}

/// A variable set by name for a test (`convar <name> <value|title>`): the value parsed by the
/// variable's kind; `title` puts the title's own back.
pub fn set_for_test(name: &str, value: &str) -> Result<(), String> {
    let i = index(name).ok_or_else(|| format!("no console variable {name} (known: {})", VARS.iter().map(|v| v.name).collect::<Vec<_>>().join(", ")))?;
    let bits = if value == "title" {
        None
    } else {
        Some(match VARS[i].kind {
            Kind::Bool => match value {
                "1" | "true" | "on" => 1,
                "0" | "false" | "off" => 0,
                _ => return Err(format!("bad bool '{value}'")),
            },
            Kind::Int => value.parse::<i32>().map_err(|_| format!("bad int '{value}'"))? as u32,
            Kind::Float => value.parse::<f32>().map_err(|_| format!("bad float '{value}'"))?.to_bits(),
        })
    };
    want(name, bits);
    Ok(())
}
