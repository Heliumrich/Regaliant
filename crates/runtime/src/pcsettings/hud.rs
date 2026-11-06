//! The title's HUD data (`ui/scripts/_cmn/uihud.cgpr`) with the achievement pop-up (`popup`):
//! the area-name caption's frame (`HUD_Tendency`'s `WorldTendency_AreaName_Frame`: the area's
//! name over the world's) copied into the gauges' scene (`HUD_Gauges`, present while playing),
//! without the tendency emblem, the event listener that shows it on entering an area or its
//! sound; its condition replaced by a copy of the subtitles' (`ConversationSubtitle`: shown
//! while a setting is on) on `Setting_PC_TrophyPopup`; its name bound to the int setting the
//! pop-up answers with the trophy name's text hash, its second line the static "TROPHY
//! UNLOCKED".

use super::cgpr::{self, Component, Entity};
use super::value::{Node, Payload, h};

fn comp_class(f: &cgpr::File, idx: u32) -> Option<u32> {
    Payload::parse(&f.comps[idx as usize].payload).map(|p| p.value.hash)
}

/// The entity's component of `class`, copied for it (its other users keep theirs) and edited.
fn edit(f: &mut cgpr::File, e: &mut Entity, class: u32, change: impl FnOnce(&mut Payload) -> Result<(), String>) -> Result<(), String> {
    let slot = e.comps.iter().position(|(_, i)| comp_class(f, *i) == Some(class)).ok_or_else(|| format!("no component {class:#x} on {}", String::from_utf8_lossy(&e.name)))?;
    let idx = e.comps[slot].1 as usize;
    let mut p = Payload::parse(&f.comps[idx].payload).ok_or("component unreadable")?;
    change(&mut p)?;
    let cat = f.comps[idx].cat;
    f.comps.push(Component { cat, payload: p.ser() });
    e.comps[slot].1 = (f.comps.len() - 1) as u32;
    Ok(())
}

/// A `Text` component's static key and the key of the int setting it may take a text hash from.
fn set_text(p: &mut Payload, key: &str, setting: &str) -> Result<(), String> {
    let text = p.value.field_mut(h("Text")).ok_or("Text without its text")?;
    *text.field_mut(h("Key")).ok_or("text without Key")? = Node::Str(key.as_bytes().to_vec());
    let n = text.replace_str_first_setting(setting);
    if n == 0 {
        return Err("text without its setting".into());
    }
    Ok(())
}

impl Node {
    /// The (one) setting key inside a text's int binding replaced by `setting`.
    fn replace_str_first_setting(&mut self, setting: &str) -> usize {
        let Some(fields) = self.fields_mut() else { return 0 };
        for (_, v) in fields.iter_mut() {
            if let Node::Obj { fields: inner, .. } = v {
                if let Some((_, Node::Obj { fields: key, .. })) = inner.iter_mut().find(|(k, _)| *k == h("Key")) {
                    if let Some((_, s @ Node::Str(_))) = key.iter_mut().next() {
                        *s = Node::Str(setting.as_bytes().to_vec());
                        return 1;
                    }
                }
            }
        }
        0
    }
}

pub fn build(orig: &[u8]) -> Result<Vec<u8>, String> {
    let mut f = cgpr::parse(orig).ok_or("unexpected layout")?;
    if cgpr::serialize(&f) != orig {
        return Err("does not serialize back to the same bytes".into());
    }
    let tend = f.roots.iter().position(|r| r.name == b"HUD_Tendency").ok_or("no HUD_Tendency")?;
    let gauges = f.roots.iter().position(|r| r.name == b"HUD_Gauges").ok_or("no HUD_Gauges")?;
    let frame = f.roots[tend].find("WorldTendency_AreaName_Frame").ok_or("no area-name frame")?.clone();
    let sub = f.roots.iter().position(|r| r.name.starts_with(b"ConversationSubtitle")).ok_or("no subtitles")?;
    let hide = h("HideCondition");
    let sub_hide = f.roots[sub].comps.iter().map(|(_, i)| *i).find(|&i| comp_class(&f, i) == Some(hide)).ok_or("no subtitle condition")?;
    let mut pop = cgpr::clone_entity(&mut f.comps, &frame, &[], &cgpr::salt("PCTrophyPopup"));
    pop.name = b"PC_TrophyPopup".to_vec();
    // The caption's texts only; its listener and sound go.
    pop.kids.retain(|k| k.name == b"AreaName");
    let (listener, sound) = (h("DSEventMan"), h("SoundEventPlayer"));
    let keep: Vec<(([u8; 16], u32), bool)> = pop.comps.iter().map(|&(g, i)| ((g, i), matches!(comp_class(&f, i), Some(c) if c == listener || c == sound))).collect();
    pop.comps = keep.into_iter().filter(|(_, drop)| !drop).map(|(c, _)| c).collect();
    // Shown while the setting is on: the subtitles' condition, on the pop-up's setting.
    let slot = pop.comps.iter().position(|(_, i)| comp_class(&f, *i) == Some(hide)).ok_or("frame without its condition")?;
    let mut cond = Payload::parse(&f.comps[sub_hide as usize].payload).ok_or("subtitle condition unreadable")?;
    if cond.value.replace_str(b"Setting_ShowSubtitles", super::popup::SETTING.as_bytes()) != 1 {
        return Err("subtitle condition without its setting".into());
    }
    let cat = f.comps[sub_hide as usize].cat;
    f.comps.push(Component { cat, payload: cond.ser() });
    pop.comps[slot].1 = (f.comps.len() - 1) as u32;
    // The texts: the trophy's name (its text hash from the pop-up's int setting), then the title.
    let text = h("Text");
    let area = pop.kids.first_mut().ok_or("no area name")?;
    edit(&mut f, area, text, |p| set_text(p, "", super::popup::NAME_SETTING))?;
    let world = area.kids.iter_mut().find(|k| k.name == b"WorldName").ok_or("no world name")?;
    world.kids.clear();
    edit(&mut f, world, text, |p| set_text(p, "", super::popup::TITLE_SETTING))?;
    f.roots[gauges].kids.push(pop);
    let out = cgpr::serialize(&f);
    cgpr::parse(&out).ok_or("the extended file does not parse")?;
    Ok(out)
}
