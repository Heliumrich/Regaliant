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
    pub id: &'static str,
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
        }
    }
    fn def(&self) -> store::Def {
        let (kind, default) = match &self.kind {
            ItemKind::Toggle(on) => (store::Kind::Int, *on as u32),
            ItemKind::Choice(_, d) => (store::Kind::Int, *d),
            ItemKind::Slider { default, .. } => (store::Kind::Float, default.to_bits()),
        };
        store::Def { name: self.key(), kind, default }
    }
}

/// The PC tab's items, in order (each one works: what it changes is in `apply`).
pub fn items() -> Vec<Item> {
    let s = |v: &str| v.to_string();
    let toggle = |id, label: &str, tooltip: &str, on| Item { id, label: s(label), tooltip: s(tooltip), kind: ItemKind::Toggle(on) };
    vec![
        Item {
            id: "WindowMode",
            label: s("Window Mode"),
            tooltip: s("Shows the game in a window or fullscreen without borders (Alt+Enter switches too)."),
            kind: ItemKind::Choice(vec![s("Windowed"), s("Borderless Fullscreen")], 0),
        },
        toggle("SkipIntro", "Skip Intro Logos", "Skips the logos shown when the game starts (from the next start).", false),
        toggle("DepthOfField", "Depth of Field", "Blurs what is far from the focus, as in the original.", true),
        toggle("Bloom", "Bloom", "The glow around bright light.", true),
        toggle("Vignette", "Vignette", "Darkens the corners of the image.", true),
        Item {
            id: "Sharpening",
            label: s("Sharpening"),
            tooltip: s("Sharpens the image shown (0.00: off, as in the original)."),
            kind: ItemKind::Slider { min: 0.0, max: 1.0, default: 0.0 },
        },
        Item {
            id: "TextureFiltering",
            label: s("Texture Filtering"),
            tooltip: s("Sharper textures at an angle: anisotropic filtering above the original's."),
            kind: ItemKind::Choice(vec![s("As Original"), s("8x Anisotropic"), s("16x Anisotropic")], 0),
        },
        Item {
            id: "MouseSensitivity",
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
    let off = |id: &str| get(id) == Some(0);
    let dof = off("DepthOfField");
    convar::want("r_dofOverrideEnable", dof.then_some(1));
    convar::want("r_dofFarAmount", dof.then_some(convar::float(0.0)));
    convar::want("r_dofNearAmount", dof.then_some(convar::float(0.0)));
    let bloom = off("Bloom");
    convar::want("r_bloomOverrideEnable", bloom.then_some(1));
    convar::want("r_bloomEnable", bloom.then_some(0));
    let vignette = off("Vignette");
    convar::want("r_vignetteOverrideEnable", vignette.then_some(1));
    convar::want("r_vignetteScale", vignette.then_some(convar::float(0.0)));
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

const TAB_LABEL: (&str, &str) = ("PC_SETTINGS_HEADING", "PC");
const TAB_TOOLTIP: (&str, &str) = ("PC_SETTINGS_TOOLTIP_TAB", "Settings of the PC version: display, graphics, controls and sound.");

/// The PC settings' values (local-data/pcsettings.txt).
pub fn init(path: std::path::PathBuf) {
    store::init(path, items().iter().map(Item::def).collect());
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
    items().iter().map(Item::key).find(|k| setting_hash(k) == hash)
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
    // The tab and page, copied together (the tab's links show and hide the copied page).
    let tab_salt = cgpr::salt("PCTab");
    let repl = vec![
        (b"SETTINGS_HEADING_DISPLAY".to_vec(), TAB_LABEL.0.as_bytes().to_vec()),
        (b"SETTINGS_TOOLTIP_DISPLAY_TAB".to_vec(), TAB_TOOLTIP.0.as_bytes().to_vec()),
        (b"DisplaySettings".to_vec(), b"PCSettings".to_vec()),
    ];
    let (bi, bp) = button_path.clone();
    let mut copies = cgpr::clone_group(&mut f.comps, &[f.roots[bi].at(&bp), f.roots[ri].at(&page_path)], &repl, &tab_salt);
    let mut pc_page = copies.pop().unwrap();
    let mut pc_button = copies.pop().unwrap();
    pc_button.name = b"PCButton".to_vec();
    // The page's items: the PC settings, copied from the DISPLAY page's own.
    let contents_rel = f.roots[ri].at(&page_path).path_to("MenuContents").ok_or("no MenuContents")?;
    let templates = f.roots[ri].at(&page_path).at(&contents_rel);
    let template = |name: &str| templates.kids.iter().position(|k| k.name == name.as_bytes()).ok_or(format!("no {name} item"));
    let (ti_choice, ti_slider) = (template(CHOICE.entity)?, template(SLIDER.entity)?);
    let mut kids = Vec::new();
    for item in items() {
        let (ti, t) = match item.kind {
            ItemKind::Toggle(_) | ItemKind::Choice(..) => (ti_choice, &CHOICE),
            ItemKind::Slider { .. } => (ti_slider, &SLIDER),
        };
        let tpl = f.roots[ri].at(&page_path).at(&contents_rel).kids[ti].clone();
        kids.push(build_item(&mut f, &tpl, t, &item)?);
    }
    pc_page.at_mut(&contents_rel).kids = kids;
    // Links from outside into the DISPLAY page (the Settings scene hides every page on close)
    // get their copies for the PC page; the DISPLAY tab's own were copied with it.
    let mut page_guids = std::collections::HashSet::new();
    cgpr::collect_guids(f.roots[ri].at(&page_path), &mut page_guids);
    let mut skip = Vec::new();
    cgpr::comp_indices(f.roots[bi].at(&bp), &mut skip);
    cgpr::comp_indices(f.roots[ri].at(&page_path), &mut skip);
    let skip: std::collections::HashSet<u32> = skip.into_iter().collect();
    let mut added = 0;
    for (i, c) in f.comps.iter_mut().enumerate() {
        if skip.contains(&(i as u32)) {
            continue;
        }
        let Some(mut p) = Payload::parse(&c.payload) else { continue };
        let mut n = 0;
        for list in p.links.iter_mut() {
            let extra: Vec<value::Link> =
                list.iter().filter(|l| page_guids.contains(&l.target)).map(|l| value::Link { target: cgpr::salted(&l.target, &tab_salt), ..l.clone() }).collect();
            n += extra.len();
            list.extend(extra);
        }
        if n > 0 {
            added += n;
            c.payload = p.ser();
        }
    }
    crate::info!("pc settings: PC tab with {} items; {added} links into the DISPLAY page copied for it", items().len());
    // The tab before EXIT GAME, the page beside the others.
    let tabs = f.roots[bi].at_mut(&bp[..bp.len() - 1]);
    let at = tabs.kids.iter().position(|k| k.name == b"ExitGameButton").unwrap_or(tabs.kids.len());
    tabs.kids.insert(at, pc_button);
    f.roots[ri].at_mut(&pages_path).kids.insert(exit_pi, pc_page);
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
    let mut lines = vec![(TAB_LABEL.0.to_string(), TAB_LABEL.1.to_string()), (TAB_TOOLTIP.0.to_string(), TAB_TOOLTIP.1.to_string())];
    for item in items() {
        lines.push((item.text_key(""), item.label.clone()));
        lines.push((item.text_key("_TIP"), item.tooltip.clone()));
        lines.extend(item.options().into_iter().filter_map(|(k, t)| t.map(|t| (k, t))));
    }
    for (k, v) in lines {
        out.extend_from_slice(format!("{k}:{:X}={v}\n", cgpr::fnv1a(k.as_bytes())).as_bytes());
    }
    out
}
