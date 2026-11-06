//! The PC settings, in the title's own SETTINGS screen: the port extends the title's menu data
//! (`ui/scripts/_cmn/uistartmenu.cgpr`) and its text (`text/common/_cmn/ui.<lang>.ctxt`) in
//! memory as the title loads them, with entries built from the title's own (its widgets, its
//! layout, its data sources), so the title's UI draws, navigates and stores them like its own.
//! Nothing is drawn over the game, and no game file is written: the title's files are read
//! from the game tree and the extended copies live in memory only. `DS_PC_MENU=0` serves the
//! title's files as they are.

pub mod cgpr;
pub mod value;

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// The title's hash of a setting name: FNV-1a over the name in upper case (eboot+0xdd45d0).
pub fn setting_hash(name: &str) -> u32 {
    cgpr::fnv1a(name.to_ascii_uppercase().as_bytes())
}

/// The settings the title reported changed (hash -> the value's 32 bits), latest first seen.
static CHANGED: Mutex<Option<HashMap<u32, u32>>> = Mutex::new(None);
static CHANGES: AtomicU64 = AtomicU64::new(0);

/// Names for the log: the title's settings and the PC menu's.
fn setting_name(hash: u32) -> Option<String> {
    const TITLE: &[&str] = &[
        "Setting_Accessibility_AdaptiveTriggers", "Setting_Accessibility_ColorBlindness", "Setting_Accessibility_DialogueBoost",
        "Setting_Accessibility_HUDBacking", "Setting_Accessibility_HapticMode", "Setting_Accessibility_HapticVibration",
        "Setting_Accessibility_MonoSound", "Setting_Accessibility_SoundBalance", "Setting_Accessibility_SubtitlesBacking",
        "Setting_AutoEquip_Items", "Setting_Bow_Aim_Toggle", "Setting_Brightness", "Setting_CameraEnableReset", "Setting_CameraHorizAxes",
        "Setting_CameraMode", "Setting_CameraShake", "Setting_CameraSpeed", "Setting_CameraVerticalAxes", "Setting_Camera_VerticalAxes",
        "Setting_Contrast", "Setting_ControllerAudio", "Setting_ControllerAudioMode", "Setting_EnableHDR", "Setting_GameplayFilmGrain",
        "Setting_HelmetArmor_Visibility", "Setting_MidnightMode", "Setting_MotionBlur", "Setting_MusicVolume", "Setting_NetworkPassword",
        "Setting_NetworkPasswordEnabled", "Setting_PlayOnline", "Setting_PlayerOffset", "Setting_SFXVolume", "Setting_Saturation",
        "Setting_ShowBlood", "Setting_ShowHUD", "Setting_ShowHUD_EnemyStatus", "Setting_ShowHUD_Equipment",
        "Setting_ShowHUD_ItemNotifications", "Setting_ShowHUD_SoulCounter", "Setting_ShowHUD_StatusBars", "Setting_ShowSubtitles",
        "Setting_SpeakerMode", "Setting_TotalPlayTime", "Setting_TouchPadLeft", "Setting_TouchPadRight", "Setting_VibrationStrength",
        "Setting_VisualMode", "Setting_VoiceVolume",
    ];
    if let Some(n) = TITLE.iter().find(|n| setting_hash(n) == hash) {
        return Some(n.to_string());
    }
    let t = pad(TEST.setting, "Setting_GameplayFilmGrain".len()).ok()?;
    (setting_hash(&t) == hash).then_some(t)
}

/// Hook at the entry of the title's handler of a changed setting (eboot+0xb4700, an observer
/// its settings registry calls: rsi the setting, its name hash at +8, its value at +0x10):
/// the port learns every change as the title makes it (logged; read-only).
pub extern "sysv64" fn changed_hook(regs: *const u64, _arg: u32) {
    let rec = unsafe { *regs.add(6) };
    if rec < 0x10000 {
        return;
    }
    let hash = unsafe { ((rec + 8) as *const u32).read_volatile() };
    let value = unsafe { ((rec + 0x10) as *const u32).read_volatile() };
    let n = CHANGES.fetch_add(1, Ordering::Relaxed);
    let prev = CHANGED.lock().get_or_insert_with(Default::default).insert(hash, value);
    if n < 400 && prev != Some(value) {
        let name = setting_name(hash).unwrap_or_else(|| format!("#{hash:08x}"));
        crate::info!("setting changed: {name} = {value:#x} ({}) [record {rec:#x}]", f32::from_bits(value));
    }
}

const START_MENU: &str = "/app0/ui/scripts/_cmn/uistartmenu.cgpr";
const TEXT_DIR: &str = "/app0/text/common/_cmn/ui.";

fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_PC_MENU").ok().as_deref() != Some("0"))
}

/// Whether the port extends this guest path (a cheap test: every file lookup asks).
fn extends(guest: &str) -> bool {
    guest == START_MENU || (guest.starts_with(TEXT_DIR) && guest.ends_with(".ctxt"))
}

static GENERATED: Mutex<Option<HashMap<String, Option<Arc<Vec<u8>>>>>> = Mutex::new(None);

/// The content the port serves for a game file it extends, built from the game's own file the
/// first time the title asks for it; `None`: the game's file as it is (not extended, or the
/// build failed, which is logged).
pub fn generated(guest: &str) -> Option<Arc<Vec<u8>>> {
    if !extends(guest) || !enabled() {
        return None;
    }
    if let Some(v) = GENERATED.lock().as_ref().and_then(|m| m.get(guest).cloned()) {
        return v;
    }
    let built = (|| {
        let host = crate::hle::kernel::fs::host_path_existing(guest).ok_or("not in the game tree")?;
        let orig = std::fs::read(&host).map_err(|_| "unreadable")?;
        if guest == START_MENU {
            start_menu(&orig)
        } else {
            text(&orig)
        }
    })();
    let v = match built {
        Ok(d) => {
            crate::info!("pc settings: {guest} extended in memory ({} bytes)", d.len());
            Some(Arc::new(d))
        }
        Err(e) => {
            crate::warn!("pc settings: {guest} served as it is: {e}");
            None
        }
    };
    GENERATED.lock().get_or_insert_with(Default::default).insert(guest.to_string(), v.clone());
    v
}

/// A setting of the PC menu: its key (`Setting_PC...`, padded to its template's length), its
/// label and tooltip text keys and English text.
struct Item {
    entity: &'static str,
    setting: &'static str,
    label: (&'static str, &'static str),
    tooltip: (&'static str, &'static str),
}

const TEST: Item = Item {
    entity: "PCTest",
    setting: "Setting_PCTest",
    label: ("PC_ENTRY_TEST", "PC Test Toggle"),
    tooltip: ("PC_TOOLTIP_TEST", "A setting of the PC version (test)."),
};

/// `key` padded with underscores to `n` bytes (equal-length replacement).
fn pad(key: &str, n: usize) -> Result<String, String> {
    if key.len() > n {
        return Err(format!("{key} is longer than its template's {n} bytes"));
    }
    Ok(format!("{key}{}", "_".repeat(n - key.len())))
}

/// The padded keys of an item cloned from a template with these keys.
fn keys(item: &Item, t_setting: &str, t_label: &str, t_tip: &str) -> Result<(String, String, String), String> {
    Ok((pad(item.setting, t_setting.len())?, pad(item.label.0, t_label.len())?, pad(item.tooltip.0, t_tip.len())?))
}

fn start_menu(orig: &[u8]) -> Result<Vec<u8>, String> {
    let mut f = cgpr::parse(orig).ok_or("unexpected layout")?;
    if cgpr::serialize(&f) != orig {
        return Err("does not serialize back to the same bytes".into());
    }
    let readable = f.comps.iter().filter(|c| value::Payload::parse(&c.payload).is_some()).count();
    crate::info!("pc settings: {readable} of {} menu components read as values and link lists (writing back the same bytes)", f.comps.len());
    // The DISPLAY page: the SettingsPages child whose Scene is DisplaySettings.
    let scene = cgpr::lstr("DisplaySettings");
    let (ri, mut path) = f.roots.iter().enumerate().find_map(|(i, r)| r.path_to("SettingsPages").map(|p| (i, p))).ok_or("no SettingsPages")?;
    let pages = f.roots[ri].at_mut(&path);
    let pi = pages
        .kids
        .iter()
        .position(|p| p.comps.iter().any(|(_, idx)| f.comps.get(*idx as usize).is_some_and(|c| cgpr::contains(&c.payload, &scene))))
        .ok_or("no DisplaySettings page")?;
    path.push(pi);
    let page = f.roots[ri].at_mut(&path);
    let mp = page.path_to("MenuContents").ok_or("no MenuContents")?;
    path.extend(mp);
    let (t_setting, t_label, t_tip) = ("Setting_GameplayFilmGrain", "UI_PHOTO_MODE_FILTER_FILM_GRAIN", "SETTINGS_TOOLTIP_DISPLAY_FILM_GRAIN");
    let at = f.roots[ri].at(&path).kids.iter().position(|k| k.name == b"FilmGrain").ok_or("no FilmGrain item")?;
    let (s, l, t) = keys(&TEST, t_setting, t_label, t_tip)?;
    let repl = vec![
        (t_setting.as_bytes().to_vec(), s.into_bytes()),
        (t_label.as_bytes().to_vec(), l.into_bytes()),
        (t_tip.as_bytes().to_vec(), t.into_bytes()),
    ];
    let mut item = cgpr::clone_entity(&mut f.comps, &f.roots[ri].at(&path).kids[at], &repl, &cgpr::salt(TEST.entity));
    item.name = TEST.entity.as_bytes().to_vec();
    f.roots[ri].at_mut(&path).kids.insert(at + 1, item);
    let out = cgpr::serialize(&f);
    cgpr::parse(&out).ok_or("the extended file does not parse")?;
    Ok(out)
}

/// The title's text file with the PC settings' lines (`KEY:HASH=value`, the hash FNV-1a of
/// the key in upper-case hex; English for every language so far).
fn text(orig: &[u8]) -> Result<Vec<u8>, String> {
    let mut out = orig.to_vec();
    if !out.ends_with(b"\n") {
        out.push(b'\n');
    }
    let (_, l, t) = keys(&TEST, "Setting_GameplayFilmGrain", "UI_PHOTO_MODE_FILTER_FILM_GRAIN", "SETTINGS_TOOLTIP_DISPLAY_FILM_GRAIN")?;
    for (k, v) in [(l, TEST.label.1), (t, TEST.tooltip.1)] {
        out.extend_from_slice(format!("{k}:{:X}={v}\n", cgpr::fnv1a(k.as_bytes())).as_bytes());
    }
    Ok(out)
}
