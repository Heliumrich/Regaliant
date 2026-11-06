//! The PC settings, in the title's own SETTINGS screen: the port extends the title's menu data
//! (`ui/scripts/_cmn/uistartmenu.cgpr`) and its text (`text/common/_cmn/ui.<lang>.ctxt`) in
//! memory as the title loads them, with a PC tab and page built from the title's own (the
//! DISPLAY tab and page copied with their links, items copied from the title's own toggle,
//! choice and slider), so the title's UI draws, navigates and resets them like its own; the
//! values live in the port (`store`), answered at the title's own lookups. Nothing is drawn
//! over the game, and no game file is written: the title's files are read from the game tree
//! and the extended copies live in memory only. `DS_PC_MENU=0` serves the title's files as
//! they are.

pub mod cgpr;
pub mod convar;
mod hud;
pub mod popup;
pub mod store;
pub mod value;

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use value::{Body, Node, Payload, h};

/// The title's hash of a setting name: FNV-1a over the name in upper case (eboot+0xdd45d0).
pub fn setting_hash(name: &str) -> u32 {
    cgpr::fnv1a(name.to_ascii_uppercase().as_bytes())
}

/// A PC setting in the menu.
pub struct Item {
    /// `Setting_PC_<id>` is its key; `PC_<ID>`, `PC_<ID>_TIP`, `PC_<ID>_<n>` its text keys.
    pub id: String,
    pub label: String,
    pub tooltip: String,
    pub kind: ItemKind,
}

pub enum ItemKind {
    /// Off and On (the title's own `BOOL_OFF` and `BOOL_ON` text, in every language), an int
    /// setting of 0 or 1: the title's choice item, whose value shows from the first frame.
    Toggle(bool),
    /// Options and the default's index.
    Choice(Vec<String>, u32),
    Slider { min: f32, max: f32, default: f32 },
    /// A trophy's state (Locked, Unlocked: the port answers it live, changes are ignored).
    Status(u32),
    /// The trophies of a grade unlocked: "k / all" (live, read-only).
    Count(char, u32),
}

impl Item {
    pub fn key(&self) -> String {
        format!("Setting_PC_{}", self.id)
    }
    fn text_key(&self, suffix: &str) -> String {
        format!("PC_{}{suffix}", self.id.to_ascii_uppercase())
    }
    /// The options' text keys, with their text when the port adds it.
    fn options(&self) -> Vec<(String, Option<String>)> {
        match &self.kind {
            ItemKind::Toggle(_) => vec![("BOOL_OFF".into(), None), ("BOOL_ON".into(), None)],
            ItemKind::Choice(opts, _) => opts.iter().enumerate().map(|(i, o)| (self.text_key(&format!("_{i}")), Some(o.clone()))).collect(),
            ItemKind::Slider { .. } => Vec::new(),
            ItemKind::Status(_) => vec![("PC_TROPHY_LOCKED".into(), Some("Locked".into())), ("PC_TROPHY_UNLOCKED".into(), Some("Unlocked".into()))],
            ItemKind::Count(g, all) => (0..=*all).map(|k| (format!("PC_TROPHY_COUNT_{g}_{k}"), Some(format!("{k} / {all}")))).collect(),
        }
    }
    fn def(&self) -> store::Def {
        let (kind, default, live) = match &self.kind {
            ItemKind::Toggle(on) => (store::Kind::Int, *on as u32, None),
            ItemKind::Choice(_, d) => (store::Kind::Int, *d, None),
            ItemKind::Slider { default, .. } => (store::Kind::Float, default.to_bits(), None),
            ItemKind::Status(id) => (store::Kind::Int, 0, Some(store::Live::Trophy(*id))),
            ItemKind::Count(g, _) => (store::Kind::Int, 0, Some(store::Live::Count(*g))),
        };
        store::Def { name: self.key(), kind, default, live }
    }
}

/// The PC tab's items, in order (each one works: what it changes is in `apply`).
pub fn items() -> Vec<Item> {
    let s = |v: &str| v.to_string();
    let toggle = |id: &str, label: &str, tooltip: &str, on| Item { id: s(id), label: s(label), tooltip: s(tooltip), kind: ItemKind::Toggle(on) };
    vec![
        Item {
            id: s("WindowMode"),
            label: s("Window Mode"),
            tooltip: s("Shows the game in a window or fullscreen without borders (Alt+Enter switches too)."),
            kind: ItemKind::Choice(vec![s("Windowed"), s("Borderless Fullscreen")], 0),
        },
        toggle("SkipIntro", "Skip Intro Logos", "Skips the logos shown when the game starts (from the next start).", false),
        Item {
            id: s("Sharpening"),
            label: s("Sharpening"),
            tooltip: s("Sharpens the image shown (0.00: off, as in the original)."),
            kind: ItemKind::Slider { min: 0.0, max: 1.0, default: 0.0 },
        },
        Item {
            id: s("TextureFiltering"),
            label: s("Texture Filtering"),
            tooltip: s("Sharper textures at an angle: anisotropic filtering above the original's."),
            kind: ItemKind::Choice(vec![s("As Original"), s("8x Anisotropic"), s("16x Anisotropic")], 0),
        },
        Item {
            id: s("MouseSensitivity"),
            label: s("Mouse Sensitivity"),
            tooltip: s("How far the camera turns per mouse movement (1.00: the controls file's own)."),
            kind: ItemKind::Slider { min: 0.1, max: 3.0, default: 1.0 },
        },
        toggle("InvertMouseY", "Invert Mouse Vertical", "Moving the mouse forward looks down.", false),
    ]
}

/// Applies the PC settings to the title (its console variables, while the game runs) and to
/// the port; at start and after every change. Settings at their defaults leave the title's own.
pub fn apply() {
    let get = |id: &str| store::value(&format!("Setting_PC_{id}"));
    let aniso = match get("TextureFiltering") {
        Some(1) => 8,
        Some(2) => 16,
        _ => 0,
    };
    crate::gpu::images::ANISO.store(aniso, std::sync::atomic::Ordering::Relaxed);
    // Sharpening: the presenter's pass (amount x2: 1.00 moves a pixel twice its difference
    // from its neighbours' mean, within their range).
    let sharpen = get("Sharpening").map(f32::from_bits).filter(|v| v.is_finite() && *v > 0.001).unwrap_or(0.0);
    crate::gpu::vk::sharpen::AMOUNT.store((sharpen * 2.0).to_bits(), std::sync::atomic::Ordering::Relaxed);
    let sens = get("MouseSensitivity").map(f32::from_bits).filter(|v| v.is_finite() && *v > 0.0).unwrap_or(1.0);
    crate::input::PC_SENSITIVITY.store(sens.to_bits(), std::sync::atomic::Ordering::Relaxed);
    crate::input::PC_INVERT_Y.store(get("InvertMouseY") == Some(1), std::sync::atomic::Ordering::Relaxed);
    // The window: applied when it changes (the window opens with it, `window_mode`).
    static WINDOW_MODE: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(u32::MAX);
    if let Some(m) = get("WindowMode") {
        if WINDOW_MODE.swap(m, std::sync::atomic::Ordering::Relaxed) != m {
            crate::gpu::vk::window::set_fullscreen(Some(m == 1));
        }
    }
}

/// The window mode the window opens with (the PC tab's; `None`: the window's own default).
pub fn window_fullscreen() -> Option<bool> {
    store::value("Setting_PC_WindowMode").map(|m| m == 1)
}

/// A tab the port adds to the SETTINGS screen (a copy of DISPLAY's tab and page).
pub struct Tab {
    /// Its copies' salt and the tab entity's name.
    pub id: &'static str,
    /// The page's scene name (DISPLAY's `DisplaySettings` replaced: unique).
    pub scene: &'static str,
    pub label: (&'static str, &'static str),
    pub tooltip: (&'static str, &'static str),
    pub items: Vec<Item>,
}

/// The tabs, in order before EXIT GAME: PC, then the trophies.
pub fn tabs() -> Vec<Tab> {
    let mut v = vec![Tab {
        id: "PCTab",
        scene: "PCSettings",
        label: ("PC_SETTINGS_HEADING", "PC"),
        tooltip: ("PC_SETTINGS_TOOLTIP_TAB", "Settings of the PC version: display, graphics, controls and sound."),
        items: items(),
    }];
    let trophies = trophy_items();
    if !trophies.is_empty() {
        v.push(Tab {
            id: "PCTrophies",
            scene: "PCTrophies",
            label: ("PC_TROPHIES_HEADING", "TROPHIES"),
            tooltip: ("PC_TROPHIES_TOOLTIP_TAB", "The game's trophies: those unlocked and those still to earn."),
            items: trophies,
        });
    }
    v
}

fn grade_name(g: char) -> &'static str {
    match g {
        'P' => "Platinum",
        'G' => "Gold",
        'S' => "Silver",
        _ => "Bronze",
    }
}

/// The TROPHIES tab: how many of each grade are unlocked (live); the opener of the full list
/// (`trophy_browser`) is placed above them.
fn trophy_items() -> Vec<Item> {
    ['P', 'G', 'S', 'B']
        .into_iter()
        .filter_map(|g| {
            let (_, all) = crate::achievements::count(g);
            (all > 0).then(|| Item {
                id: format!("TrophyCount{g}"),
                label: format!("{} Trophies", grade_name(g)),
                tooltip: format!("The {} trophies unlocked. All Trophies lists every trophy.", grade_name(g).to_lowercase()),
                kind: ItemKind::Count(g, all),
            })
        })
        .collect()
}

/// The full list's rows: (text key prefix, name, heading, detail) per trophy, hidden ones masked
/// until unlocked (as at this start).
fn trophy_rows() -> Vec<(String, String, String, String)> {
    crate::achievements::list()
        .into_iter()
        .map(|(t, at)| {
            let shown = !t.hidden || at.is_some();
            let name = if shown { t.name.clone() } else { "Hidden Trophy".into() };
            let heading = format!("{} - {}", grade_name(t.grade), if at.is_some() { "Unlocked" } else { "Locked" });
            let detail = if shown { t.detail.replace(['\n', '\r'], " ") } else { "A hidden trophy: its name shows once it is unlocked.".into() };
            (format!("PC_TROPHY_{:02}", t.id), name, heading, detail)
        })
        .collect()
}

/// The PC settings' values (local-data/pcsettings.txt); the trophies' rows read their state.
pub fn init(path: std::path::PathBuf) {
    let mut defs: Vec<store::Def> = tabs().iter().flat_map(|t| t.items.iter().map(Item::def)).collect();
    // The achievement pop-up's (`popup`, `hud`).
    defs.push(store::Def { name: popup::SETTING.into(), kind: store::Kind::Int, default: 0, live: Some(store::Live::Popup) });
    defs.push(store::Def { name: popup::NAME_SETTING.into(), kind: store::Kind::Int, default: 0, live: Some(store::Live::PopupName) });
    defs.push(store::Def { name: popup::TITLE_SETTING.into(), kind: store::Kind::Int, default: 0, live: Some(store::Live::PopupTitle) });
    store::init(path, defs);
    apply();
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
    tabs().iter().flat_map(|t| t.items.iter().map(Item::key)).find(|k| setting_hash(k) == hash)
}

/// Hook at the entry of the title's handler of a changed setting (eboot+0xb4700, an observer
/// its settings registry calls: rsi the setting, its name hash at +8, its value at +0x10):
/// logged (read-only).
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
const HUD: &str = "/app0/ui/scripts/_cmn/uihud.cgpr";
const TEXT_DIR: &str = "/app0/text/common/_cmn/ui.";

fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_PC_MENU").ok().as_deref() != Some("0"))
}

/// Whether the port extends this guest path (a cheap test: every file lookup asks).
fn extends(guest: &str) -> bool {
    guest == START_MENU || guest == HUD || (guest.starts_with(TEXT_DIR) && guest.ends_with(".ctxt"))
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
        } else if guest == HUD {
            hud::build(&orig)
        } else {
            Ok(text(&orig))
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

/// The template item of each kind in the DISPLAY page, with its setting key and text keys.
struct Template {
    entity: &'static str,
    setting: &'static str,
    label: &'static str,
    tooltip: &'static str,
}

const CHOICE: Template = Template { entity: "VisualMode", setting: "Setting_VisualMode", label: "SETTINGS_ENTRY_VISUAL_MODE", tooltip: "SETTINGS_TOOLTIP_DISPLAY_VISUAL_MODE" };
const SLIDER: Template = Template { entity: "MotionBlur", setting: "Setting_MotionBlur", label: "SETTINGS_ENTRY_MOTION_BLUR", tooltip: "SETTINGS_TOOLTIP_DISPLAY_MOTION_BLUR" };

/// Data source fields (FNV-1a of their names where known).
const DS_ENTRIES: [u32; 2] = [0x3b8e_9522, 0xc5ad_cd62]; // int, float (`Floats`)
const F_DEFAULT: u32 = 0x8d03_b720;
const F_BINDING: u32 = 0xc127_7787;
const F_BINDING_DEFAULT: u32 = 0xcd09_3f9f;
const F_OPTIONS: u32 = 0xbe5e_3fad;

fn raw_u32(v: u32) -> Node {
    Node::Raw(v.to_le_bytes().to_vec())
}

/// Sets `Value` inside an object field (`{Value: ...}`), or the field itself when it is raw.
fn set_value(n: &mut Node, v: Node) {
    match n {
        Node::Obj { fields, .. } => {
            if let Some((_, x)) = fields.iter_mut().find(|(k, _)| *k == h("Value")) {
                *x = v;
            }
        }
        other => *other = v,
    }
}

/// The item's data source entry: defaults, range and options for the item's kind.
fn edit_data_source(p: &mut Payload, item: &Item) -> Result<(), String> {
    let Body::Fields(fields) = &mut p.value.body else { return Err("data source without fields".into()) };
    let (_, entries) = fields.iter_mut().find(|(k, _)| DS_ENTRIES.contains(k)).ok_or("no data source entries")?;
    let Node::Structs(entries) = entries else { return Err("data source entries not an array".into()) };
    let e = entries.first_mut().ok_or("empty data source")?;
    fn fld(e: &mut [(u32, Node)], k: u32) -> Result<&mut Node, String> {
        e.iter_mut().find(|(x, _)| *x == k).map(|(_, v)| v).ok_or_else(|| format!("no field {k:#x}"))
    }
    let (default, min, max) = match &item.kind {
        ItemKind::Toggle(on) => (raw_u32(*on as u32), Some(raw_u32(0)), Some(raw_u32(1))),
        ItemKind::Status(_) => (raw_u32(0), Some(raw_u32(0)), Some(raw_u32(1))),
        ItemKind::Count(_, all) => (raw_u32(0), Some(raw_u32(0)), Some(raw_u32(*all))),
        ItemKind::Choice(opts, d) => {
            let n = opts.len().max(1) as u32;
            (raw_u32(*d), Some(raw_u32(0)), Some(raw_u32(n - 1)))
        }
        ItemKind::Slider { min, max, default } => (raw_u32(default.to_bits()), Some(raw_u32(min.to_bits())), Some(raw_u32(max.to_bits()))),
    };
    *fld(e, F_DEFAULT)? = default.clone();
    if let Some(b) = fld(e, F_BINDING)?.field_mut(F_BINDING_DEFAULT) {
        set_value(b, default);
    }
    if let Some(m) = min {
        set_value(fld(e, h("MinValue"))?, m);
    }
    if let Some(m) = max {
        set_value(fld(e, h("MaxValue"))?, m);
    }
    let opts = item.options();
    if !opts.is_empty() {
        let (_, o) = e.iter_mut().find(|(k, _)| *k == F_OPTIONS).ok_or("no options")?;
        let Node::Structs(list) = o else { return Err("options not an array".into()) };
        let first = list.first().cloned().ok_or("no option to copy")?;
        let mut out = Vec::new();
        for (i, (key, _)) in opts.iter().enumerate() {
            let mut s = first.clone();
            for (k, v) in s.iter_mut() {
                if *k == h("Value") {
                    set_value(v, raw_u32(i as u32));
                } else if *k == h("Description") {
                    if let Some(kn) = v.field_mut(h("Key")) {
                        *kn = Node::Str(key.clone().into_bytes());
                    }
                }
            }
            out.push(s);
        }
        *list = out;
    }
    Ok(())
}

/// An item copied from its kind's template, with its own key, text, defaults and options.
fn build_item(f: &mut cgpr::File, template: &cgpr::Entity, t: &Template, item: &Item) -> Result<cgpr::Entity, String> {
    let repl = vec![
        (t.setting.as_bytes().to_vec(), item.key().into_bytes()),
        (t.label.as_bytes().to_vec(), item.text_key("").into_bytes()),
        (t.tooltip.as_bytes().to_vec(), item.text_key("_TIP").into_bytes()),
    ];
    let first_new = f.comps.len();
    let mut e = cgpr::clone_entity(&mut f.comps, template, &repl, &cgpr::salt(&format!("PCItem_{}", item.id)));
    e.name = format!("PC_{}", item.id).into_bytes();
    // The copy's own data source (copied: it holds the key).
    let mut used = Vec::new();
    cgpr::comp_indices(&e, &mut used);
    let mut edited = false;
    for &i in &used {
        let i = i as usize;
        if i < first_new {
            continue;
        }
        let Some(mut p) = Payload::parse(&f.comps[i].payload) else { continue };
        let is_ds = matches!(&p.value.body, Body::Fields(fl) if fl.iter().any(|(k, _)| DS_ENTRIES.contains(k)));
        if is_ds {
            edit_data_source(&mut p, item)?;
            f.comps[i].payload = p.ser();
            edited = true;
        }
    }
    if !edited {
        return Err(format!("{}: no data source in the copy", item.id));
    }
    // Nothing of the template's own key or text may remain in the copy.
    for &i in &used {
        let pl = &f.comps[i as usize].payload;
        for s in [t.setting, t.label, t.tooltip] {
            if cgpr::contains(pl, &cgpr::lstr(s)) {
                return Err(format!("{}: the copy still names {s}", item.id));
            }
        }
    }
    Ok(e)
}

/// The full trophy list: the title's HELP screen (`HelpMenu`, a scrolling list with a heading
/// and a description beside it) copied with the GAME page's item that opens it (their one link
/// remapped), a row per trophy copied from a plain topic row (name, grade and state as the
/// heading, the detail as the description, no movie). Returns the opener; the screen is added
/// as a root.
fn trophy_browser(f: &mut cgpr::File, rows: &[(String, String, String, String)]) -> Result<cgpr::Entity, String> {
    let help_i = f.roots.iter().position(|r| r.name == b"HelpMenu").ok_or("no HELP screen")?;
    let pages_ri = f.roots.iter().position(|r| r.path_to("SettingsPages").is_some()).ok_or("no SettingsPages")?;
    let pages_path = f.roots[pages_ri].path_to("SettingsPages").unwrap();
    let game_scene = cgpr::lstr("GameSettings");
    let game_pi = {
        let pages = f.roots[pages_ri].at(&pages_path);
        pages
            .kids
            .iter()
            .position(|p| p.comps.iter().any(|(_, idx)| f.comps.get(*idx as usize).is_some_and(|c| cgpr::contains(&c.payload, &game_scene))))
            .ok_or("no GAME page")?
    };
    let mut game_path = pages_path.clone();
    game_path.push(game_pi);
    let opener_rel = f.roots[pages_ri].at(&game_path).path_to("HelpMenu").ok_or("no HELP item")?;
    let mut opener_path = game_path.clone();
    opener_path.extend(opener_rel);
    let repl = vec![
        (b"SETTINGS_ENTRY_HELP_MENU".to_vec(), b"PC_TROPHIES_OPEN".to_vec()),
        (b"SETTINGS_TOOLTIP_ACCESSIBILITY_HELP_MENU".to_vec(), b"PC_TROPHIES_OPEN_TIP".to_vec()),
        (b"HelpMenu".to_vec(), b"PCTrophyList".to_vec()),
        (b"HelpMenuRoot".to_vec(), b"PCTrophyListRoot".to_vec()),
        (b"HELP_MENU_HEADING_MAIN".to_vec(), b"PC_TROPHIES_HEADING".to_vec()),
    ];
    let salt = cgpr::salt("PCTrophyList");
    let mut copies = cgpr::clone_group(&mut f.comps, &[f.roots[pages_ri].at(&opener_path), &f.roots[help_i]], &repl, &salt);
    let mut screen = copies.pop().unwrap();
    let mut opener = copies.pop().unwrap();
    opener.name = b"PC_TrophiesOpen".to_vec();
    screen.name = b"PCTrophyList".to_vec();
    // The rows: copies of a plain topic (a Frame without a section heading in its SettingsValue).
    let mc_rel = screen.path_to("MenuContents").ok_or("no list")?;
    let (frame, value) = {
        let mc = screen.at(&mc_rel);
        let k = (0..mc.kids.len().saturating_sub(1))
            .step_by(2)
            .find(|&k| mc.kids[k].name == b"Frame" && mc.kids[k + 1].name == b"SettingsValue" && mc.kids[k + 1].kids.is_empty())
            .ok_or("no plain topic row")?;
        (mc.kids[k].clone(), mc.kids[k + 1].clone())
    };
    // The template row's own keys and movie.
    let mut strings = Vec::new();
    let mut idx = Vec::new();
    cgpr::comp_indices(&frame, &mut idx);
    for i in idx {
        if let Some(p) = Payload::parse(&f.comps[i as usize].payload) {
            fn walk(n: &Node, out: &mut Vec<String>) {
                match n {
                    Node::Str(b) => out.push(String::from_utf8_lossy(b).into_owned()),
                    Node::Obj { fields, .. } => fields.iter().for_each(|(_, v)| walk(v, out)),
                    Node::List(l) => l.iter().for_each(|t| {
                        if let Body::Fields(fl) = &t.body {
                            fl.iter().for_each(|(_, v)| walk(v, out))
                        }
                    }),
                    Node::Structs(st) => st.iter().flatten().for_each(|(_, v)| walk(v, out)),
                    Node::Raw(_) => {}
                }
            }
            if let Body::Fields(fl) = &p.value.body {
                fl.iter().for_each(|(_, v)| walk(v, &mut strings));
            }
        }
    }
    let find = |pre: &str| strings.iter().find(|s| s.starts_with(pre)).cloned();
    let entry = find("HELP_ENTRY_").ok_or("row without its entry key")?;
    let desc = find("HELP_DESCRIPTION_").ok_or("row without its description key")?;
    let sub = find("HELP_SUBHEADING_").filter(|s| s.len() > "HELP_SUBHEADING_".len()).ok_or("row without its heading key")?;
    let movie = strings.iter().find(|s| s.contains("Movies")).cloned();
    let mut kids = Vec::new();
    for (key, ..) in rows {
        let mut r = vec![
            (entry.as_bytes().to_vec(), key.as_bytes().to_vec()),
            (desc.as_bytes().to_vec(), format!("{key}_TIP").into_bytes()),
            (sub.as_bytes().to_vec(), format!("{key}_SUB").into_bytes()),
        ];
        if let Some(m) = &movie {
            r.push((m.as_bytes().to_vec(), Vec::new()));
        }
        let mut pair = cgpr::clone_group(&mut f.comps, &[&frame, &value], &r, &cgpr::salt(key));
        let v = pair.pop().unwrap();
        let fr = pair.pop().unwrap();
        kids.push(fr);
        kids.push(v);
    }
    screen.at_mut(&mc_rel).kids = kids;
    // No movies: the movie's frame goes.
    if let Some(path) = screen.path_to("HelpItemMovie") {
        let (last, parent) = path.split_last().unwrap();
        screen.at_mut(parent).kids.remove(*last);
    }
    crate::info!("pc settings: trophy list with {} rows (the HELP screen's layout)", rows.len());
    f.roots.push(screen);
    Ok(opener)
}

fn start_menu(orig: &[u8]) -> Result<Vec<u8>, String> {
    let mut f = cgpr::parse(orig).ok_or("unexpected layout")?;
    if cgpr::serialize(&f) != orig {
        return Err("does not serialize back to the same bytes".into());
    }
    let readable = f.comps.iter().filter(|c| Payload::parse(&c.payload).is_some()).count();
    crate::info!("pc settings: {readable} of {} menu components read as values and link lists (writing back the same bytes)", f.comps.len());
    // The tabs (the DISPLAY button's parent) and the pages (DISPLAY's: the SettingsPages child
    // whose Scene is DisplaySettings).
    let ri = f.roots.iter().position(|r| r.path_to("SettingsPages").is_some()).ok_or("no SettingsPages")?;
    let pages_path = f.roots[ri].path_to("SettingsPages").unwrap();
    let button_path = f.roots.iter().position(|r| r.path_to("DisplayButton").is_some()).map(|bi| (bi, f.roots[bi].path_to("DisplayButton").unwrap())).ok_or("no DisplayButton")?;
    let scene = cgpr::lstr("DisplaySettings");
    let pages = f.roots[ri].at(&pages_path);
    let pi = pages
        .kids
        .iter()
        .position(|p| p.comps.iter().any(|(_, idx)| f.comps.get(*idx as usize).is_some_and(|c| cgpr::contains(&c.payload, &scene))))
        .ok_or("no DisplaySettings page")?;
    let exit_pi = pages
        .kids
        .iter()
        .position(|p| p.comps.iter().any(|(_, idx)| f.comps.get(*idx as usize).is_some_and(|c| cgpr::contains(&c.payload, &cgpr::lstr("ExitGame")))))
        .unwrap_or(pages.kids.len());
    let mut page_path = pages_path.clone();
    page_path.push(pi);
    let (bi, bp) = button_path.clone();
    let contents_rel = f.roots[ri].at(&page_path).path_to("MenuContents").ok_or("no MenuContents")?;
    let (ti_choice, ti_slider) = {
        let templates = f.roots[ri].at(&page_path).at(&contents_rel);
        let template = |name: &str| templates.kids.iter().position(|k| k.name == name.as_bytes()).ok_or(format!("no {name} item"));
        (template(CHOICE.entity)?, template(SLIDER.entity)?)
    };
    // Links from outside into the DISPLAY page (the Settings scene hides every page on close)
    // get copies for each new page; the DISPLAY tab's own are copied with it.
    let mut page_guids = std::collections::HashSet::new();
    cgpr::collect_guids(f.roots[ri].at(&page_path), &mut page_guids);
    let mut skip = Vec::new();
    cgpr::comp_indices(f.roots[bi].at(&bp), &mut skip);
    cgpr::comp_indices(f.roots[ri].at(&page_path), &mut skip);
    let skip: std::collections::HashSet<u32> = skip.into_iter().collect();
    let tabs = tabs();
    let salts: Vec<[u8; 16]> = tabs.iter().map(|t| cgpr::salt(t.id)).collect();
    let mut made = Vec::new();
    for (tab, salt) in tabs.iter().zip(&salts) {
        // The tab and page, copied together (the tab's links show and hide its page).
        let repl = vec![
            (b"SETTINGS_HEADING_DISPLAY".to_vec(), tab.label.0.as_bytes().to_vec()),
            (b"SETTINGS_TOOLTIP_DISPLAY_TAB".to_vec(), tab.tooltip.0.as_bytes().to_vec()),
            (b"DisplaySettings".to_vec(), tab.scene.as_bytes().to_vec()),
        ];
        let mut copies = cgpr::clone_group(&mut f.comps, &[f.roots[bi].at(&bp), f.roots[ri].at(&page_path)], &repl, salt);
        let mut page = copies.pop().unwrap();
        let mut button = copies.pop().unwrap();
        button.name = format!("{}Button", tab.id).into_bytes();
        // The page's items, copied from the DISPLAY page's own.
        let mut kids = Vec::new();
        for item in &tab.items {
            let (ti, t) = match item.kind {
                ItemKind::Toggle(_) | ItemKind::Choice(..) | ItemKind::Status(_) | ItemKind::Count(..) => (ti_choice, &CHOICE),
                ItemKind::Slider { .. } => (ti_slider, &SLIDER),
            };
            let tpl = f.roots[ri].at(&page_path).at(&contents_rel).kids[ti].clone();
            kids.push(build_item(&mut f, &tpl, t, item)?);
        }
        page.at_mut(&contents_rel).kids = kids;
        if tab.id == "PCTrophies" {
            match trophy_browser(&mut f, &trophy_rows()) {
                Ok(opener) => page.at_mut(&contents_rel).kids.insert(0, opener),
                Err(e) => crate::warn!("pc settings: no trophy list ({e})"),
            }
        }
        made.push((button, page));
    }
    let mut added = 0;
    for (i, c) in f.comps.iter_mut().enumerate() {
        if skip.contains(&(i as u32)) {
            continue;
        }
        let Some(mut p) = Payload::parse(&c.payload) else { continue };
        let mut n = 0;
        for list in p.links.iter_mut() {
            let mut extra = Vec::new();
            for salt in &salts {
                extra.extend(list.iter().filter(|l| page_guids.contains(&l.target)).map(|l| value::Link { target: cgpr::salted(&l.target, salt), ..l.clone() }));
            }
            n += extra.len();
            list.extend(extra);
        }
        if n > 0 {
            added += n;
            c.payload = p.ser();
        }
    }
    crate::info!(
        "pc settings: tabs {}; {added} links into the DISPLAY page copied for them",
        tabs.iter().map(|t| format!("{} ({} items)", t.label.1, t.items.len())).collect::<Vec<_>>().join(", ")
    );
    // The tabs before EXIT GAME, the pages beside the others, in order.
    let mut exit_pi = exit_pi;
    for (button, page) in made {
        let tabs_e = f.roots[bi].at_mut(&bp[..bp.len() - 1]);
        let at = tabs_e.kids.iter().position(|k| k.name == b"ExitGameButton").unwrap_or(tabs_e.kids.len());
        tabs_e.kids.insert(at, button);
        f.roots[ri].at_mut(&pages_path).kids.insert(exit_pi, page);
        exit_pi += 1;
    }
    let out = cgpr::serialize(&f);
    cgpr::parse(&out).ok_or("the extended file does not parse")?;
    Ok(out)
}

/// The title's text file with the PC settings' lines (`KEY:HASH=value`, the hash FNV-1a of
/// the key in upper-case hex; English for every language so far).
fn text(orig: &[u8]) -> Vec<u8> {
    let mut out = orig.to_vec();
    if !out.is_empty() && !out.ends_with(b"\n") {
        out.push(b'\n');
    }
    let mut lines: Vec<(String, String)> = Vec::new();
    for tab in tabs() {
        lines.push((tab.label.0.into(), tab.label.1.into()));
        lines.push((tab.tooltip.0.into(), tab.tooltip.1.into()));
        for item in &tab.items {
            lines.push((item.text_key(""), item.label.clone()));
            lines.push((item.text_key("_TIP"), item.tooltip.clone()));
            lines.extend(item.options().into_iter().filter_map(|(k, t)| t.map(|t| (k, t))));
        }
    }
    lines.push(("PC_TROPHIES_OPEN".into(), "All Trophies".into()));
    lines.push(("PC_TROPHIES_OPEN_TIP".into(), "Every trophy of the game, with what it asks for.".into()));
    lines.push(("PC_TROPHIES_HEADING".into(), "TROPHIES".into()));
    lines.push((popup::TITLE_KEY.into(), "TROPHY UNLOCKED".into()));
    for (t, _) in crate::achievements::list() {
        lines.push((popup::name_key(t.id), t.name.clone()));
    }
    for (key, name, heading, detail) in trophy_rows() {
        lines.push((key.clone(), name));
        lines.push((format!("{key}_SUB"), heading));
        lines.push((format!("{key}_TIP"), detail));
    }
    let mut seen = std::collections::HashSet::new();
    lines.retain(|(k, _)| seen.insert(k.clone()));
    for (k, v) in lines {
        out.extend_from_slice(format!("{k}:{:X}={v}\n", cgpr::fnv1a(k.as_bytes())).as_bytes());
    }
    out
}
