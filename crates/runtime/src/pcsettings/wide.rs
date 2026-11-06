//! Wide screens (the PC tab's Aspect Ratio): the title's UI files adapted as they load when the
//! output is wider than 16:9. The UI lays itself out on a canvas of the output's size (2160
//! high, so its 4K sprites keep their size; 1080 units high, 2 pixels a unit): elements anchored
//! to an edge of a screen-sized parent (the HUD, the start menu's panels) reach the screen's
//! edges, but elements of a fixed 1920x1080 (the 16:9 frame: movies, logos, the title screen,
//! dialogs and messages, the crosshair) sit at the canvas's left. The outermost such element of
//! each branch is moved right by half the extra width (its `X`: the frame centred; what is inside
//! moves with it), and a fill of that size is made as wide as the canvas and moved back to its
//! left edge, so it covers the screen: a `Quad` without a texture (black backings, dims, fades)
//! or showing a render target (a texture named without a path: the loading screen's fog,
//! drawn by particles into 'loading_fog'). The render target's own element stays where it is.
//! (The title's `AlignRelativeToParent` cannot centre them: their parents are scene roots of no
//! size at the canvas's origin.)

use super::cgpr;
use super::value::{Body, Node, Payload, h};

/// The Element's position field X (FNV-1a of "X"), relative to the parent.
const X: u32 = 0xdd0c_1e27;

fn raw_f32(n: &Node) -> Option<f32> {
    match n {
        Node::Raw(b) if b.len() == 4 => Some(f32::from_le_bytes(b[..4].try_into().ok()?)),
        _ => None,
    }
}

fn get_f32(p: &Payload, field: u32) -> Option<f32> {
    let Body::Fields(fields) = &p.value.body else { return None };
    fields.iter().find(|(k, _)| *k == field).and_then(|(_, v)| raw_f32(v))
}

/// Sets (or adds) a raw float field.
fn set_f32(p: &mut Payload, field: u32, v: f32) -> bool {
    let Body::Fields(fields) = &mut p.value.body else { return false };
    match fields.iter_mut().find(|(k, _)| *k == field) {
        Some((_, n)) => *n = Node::Raw(v.to_le_bytes().to_vec()),
        None => fields.push((field, Node::Raw(v.to_le_bytes().to_vec()))),
    }
    true
}

/// The non-empty strings of a node (a texture's path, a render target's or movie's name).
fn strings(n: &Node, out: &mut Vec<Vec<u8>>) {
    match n {
        Node::Str(s) if !s.is_empty() => out.push(s.clone()),
        Node::Obj { fields, .. } => fields.iter().for_each(|(_, v)| strings(v, out)),
        Node::List(items) => items.iter().for_each(|t| {
            if let Body::Fields(f) = &t.body {
                f.iter().for_each(|(_, v)| strings(v, out));
            }
        }),
        Node::Structs(items) => items.iter().flatten().for_each(|(_, v)| strings(v, out)),
        _ => {}
    }
}

fn payload_strings(p: &Payload) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    if let Body::Fields(f) = &p.value.body {
        f.iter().for_each(|(_, v)| strings(v, &mut out));
    }
    out
}

/// A texture's path (an image) rather than a render target's name.
fn is_path(s: &[u8]) -> bool {
    s.starts_with(b"$/") || s.ends_with(b".ctxr")
}

struct Pass<'a> {
    f: &'a cgpr::File,
    classes: Vec<Option<u32>>,
    /// Half the extra width, in UI units.
    dx: f32,
    /// The canvas's width in UI units.
    width: f32,
    element: u32,
    quad: u32,
    own_layout: [u32; 4],
    /// Render targets the file's quads show (their names).
    targets: Vec<Vec<u8>>,
    /// Changed Element payloads to append (their indices follow the file's components).
    added: Vec<cgpr::Component>,
    moved: usize,
    widened: usize,
}

impl Pass<'_> {
    fn cls(&self, i: u32) -> Option<u32> {
        self.classes.get(i as usize).copied().flatten()
    }

    /// `shift`: how far an ancestor was moved right (UI units).
    fn visit(&mut self, e: &mut cgpr::Entity, shift: f32) {
        let mut shift_kids = shift;
        let own = e.comps.iter().any(|(_, i)| self.cls(*i).is_some_and(|c| self.own_layout.contains(&c)));
        let slot = e.comps.iter().position(|(_, i)| self.cls(*i) == Some(self.element));
        // The render target's own element (what it draws into) stays.
        let target = e.comps.iter().any(|(_, i)| {
            self.cls(*i) != Some(self.quad) && Payload::parse(&self.f.comps[*i as usize].payload).is_some_and(|p| payload_strings(&p).iter().any(|s| self.targets.contains(s)))
        });
        if target {
            return;
        }
        if let (false, Some(slot)) = (own, slot) {
            let idx = e.comps[slot].1 as usize;
            if let Some(mut el) = Payload::parse(&self.f.comps[idx].payload) {
                // The 16:9 frame (within a few units: the loading screen's fog quad is 1921 wide).
                let full = matches!((get_f32(&el, h("Width")), get_f32(&el, h("Height"))), (Some(w), Some(hh)) if (w - 1920.0).abs() <= 8.0 && (hh - 1080.0).abs() <= 8.0);
                if full {
                    let x = get_f32(&el, X).unwrap_or(0.0);
                    // A fill: a Quad without a texture, or showing a render target.
                    let solid = e.comps.iter().any(|(_, i)| {
                        self.cls(*i) == Some(self.quad) && Payload::parse(&self.f.comps[*i as usize].payload).is_some_and(|p| payload_strings(&p).iter().all(|s| !is_path(s)))
                    });
                    let changed = if solid {
                        // The whole canvas: from its left edge (the ancestors' shift undone).
                        self.widened += 1;
                        set_f32(&mut el, h("Width"), self.width) && set_f32(&mut el, X, x - shift)
                    } else if shift == 0.0 {
                        // The outermost 16:9 frame of its branch: centred.
                        self.moved += 1;
                        shift_kids = self.dx;
                        set_f32(&mut el, X, x + self.dx)
                    } else {
                        false
                    };
                    if changed {
                        self.added.push(cgpr::Component { cat: self.f.comps[idx].cat, payload: el.ser() });
                        e.comps[slot].1 = (self.f.comps.len() + self.added.len() - 1) as u32;
                    }
                }
            }
        }
        for k in e.kids.iter_mut() {
            self.visit(k, shift_kids);
        }
    }
}

/// Adapts a UI file for a canvas `factor` times as wide as 16:9 (below 1 for 16:10: the frames
/// then overhang both sides equally): the file, the count of 16:9 frames centred and of fills
/// fitted.
pub fn adapt(d: &[u8], factor: f32) -> Result<(Vec<u8>, usize, usize), String> {
    let mut f = cgpr::parse(d).ok_or("not a UI file")?;
    let classes: Vec<Option<u32>> = f.comps.iter().map(|c| Payload::parse(&c.payload).map(|p| p.value.hash)).collect();
    let quad = h("Quad");
    let mut targets: Vec<Vec<u8>> = Vec::new();
    for (c, cls) in f.comps.iter().zip(&classes) {
        if *cls == Some(quad) {
            if let Some(p) = Payload::parse(&c.payload) {
                targets.extend(payload_strings(&p).into_iter().filter(|s| !is_path(s)));
            }
        }
    }
    let width = 1920.0 * factor;
    let mut roots = std::mem::take(&mut f.roots);
    let mut pass = Pass {
        f: &f,
        classes,
        dx: (width - 1920.0) / 2.0,
        width,
        element: h("Element"),
        quad,
        own_layout: [h("AlignRelativeToParent"), h("FitRelativeToParent"), h("FitSizeToParent"), h("UpdateElementPosAndSize")],
        targets,
        added: Vec::new(),
        moved: 0,
        widened: 0,
    };
    for r in roots.iter_mut() {
        pass.visit(r, 0.0);
    }
    let (added, moved, widened) = (std::mem::take(&mut pass.added), pass.moved, pass.widened);
    f.roots = roots;
    f.comps.extend(added);
    Ok((cgpr::serialize(&f), moved, widened))
}
