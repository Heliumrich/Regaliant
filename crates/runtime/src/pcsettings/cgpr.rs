//! The title's UI files (`ui/scripts/**/*.cgpr`, Bluepoint's serialized entity graphs), as far
//! as the PC settings need them: a list of components (a category hash, a LEB128 size and a
//! payload: a typed value and a trailer) that entities share by index, then the entity
//! hierarchy (GUID, name, a word, components as GUID and index, children) and a footer ("PRPG",
//! a version, the length of what precedes it). Payloads stay bytes; edits replace strings at
//! equal length, so no size inside a payload changes (`tools/cgpr.py` reads the payloads too).

pub struct Component {
    pub cat: u32,
    pub payload: Vec<u8>,
}

#[derive(Clone)]
pub struct Entity {
    pub guid: [u8; 16],
    pub name: Vec<u8>,
    pub flags: u32,
    pub comps: Vec<([u8; 16], u32)>,
    pub kids: Vec<Entity>,
}

pub struct File {
    pub comps: Vec<Component>,
    pub roots: Vec<Entity>,
    pub footer: [u8; 12],
}

fn u32_at(d: &[u8], i: usize) -> Option<u32> {
    Some(u32::from_le_bytes(d.get(i..i + 4)?.try_into().ok()?))
}

fn leb(d: &[u8], i: &mut usize) -> Option<u64> {
    let mut v = 0u64;
    let mut s = 0;
    loop {
        let b = *d.get(*i)?;
        *i += 1;
        v |= ((b & 0x7f) as u64) << s;
        s += 7;
        if b < 0x80 {
            return Some(v);
        }
        if s > 56 {
            return None;
        }
    }
}

fn put_leb(out: &mut Vec<u8>, mut v: u64) {
    loop {
        let b = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            out.push(b);
            return;
        }
        out.push(b | 0x80);
    }
}

fn entity(d: &[u8], i: &mut usize, depth: u32) -> Option<Entity> {
    if depth > 256 {
        return None;
    }
    let guid: [u8; 16] = d.get(*i..*i + 16)?.try_into().ok()?;
    *i += 16;
    let n = *d.get(*i)? as usize;
    let name = d.get(*i + 1..*i + 1 + n)?.to_vec();
    *i += 1 + n;
    let flags = u32_at(d, *i)?;
    let cc = u32_at(d, *i + 4)? as usize;
    *i += 8;
    let mut comps = Vec::with_capacity(cc.min(4096));
    for _ in 0..cc {
        let g: [u8; 16] = d.get(*i..*i + 16)?.try_into().ok()?;
        comps.push((g, u32_at(d, *i + 16)?));
        *i += 20;
    }
    let kc = u32_at(d, *i)? as usize;
    *i += 4;
    let mut kids = Vec::with_capacity(kc.min(4096));
    for _ in 0..kc {
        kids.push(entity(d, i, depth + 1)?);
    }
    Some(Entity { guid, name, flags, comps, kids })
}

pub fn parse(d: &[u8]) -> Option<File> {
    let n = u32_at(d, 0)? as usize;
    let mut i = 4;
    let mut comps = Vec::with_capacity(n.min(1 << 20));
    for _ in 0..n {
        let cat = u32_at(d, i)?;
        let mut j = i + 4;
        let size = leb(d, &mut j)? as usize;
        comps.push(Component { cat, payload: d.get(j..j + size)?.to_vec() });
        i = j + size;
    }
    let rn = u32_at(d, i)? as usize;
    i += 4;
    let mut roots = Vec::with_capacity(rn.min(4096));
    for _ in 0..rn {
        roots.push(entity(d, &mut i, 0)?);
    }
    let footer: [u8; 12] = d.get(i..)?.try_into().ok()?;
    (&footer[..4] == b"PRPG").then_some(File { comps, roots, footer })
}

fn put_entity(out: &mut Vec<u8>, e: &Entity) {
    out.extend_from_slice(&e.guid);
    out.push(e.name.len() as u8);
    out.extend_from_slice(&e.name);
    out.extend_from_slice(&e.flags.to_le_bytes());
    out.extend_from_slice(&(e.comps.len() as u32).to_le_bytes());
    for (g, idx) in &e.comps {
        out.extend_from_slice(g);
        out.extend_from_slice(&idx.to_le_bytes());
    }
    out.extend_from_slice(&(e.kids.len() as u32).to_le_bytes());
    for k in &e.kids {
        put_entity(out, k);
    }
}

pub fn serialize(f: &File) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&(f.comps.len() as u32).to_le_bytes());
    for c in &f.comps {
        out.extend_from_slice(&c.cat.to_le_bytes());
        put_leb(&mut out, c.payload.len() as u64);
        out.extend_from_slice(&c.payload);
    }
    out.extend_from_slice(&(f.roots.len() as u32).to_le_bytes());
    for e in &f.roots {
        put_entity(&mut out, e);
    }
    // The footer: magic, version, and the length of what precedes the footer.
    let len = out.len() as u32;
    out.extend_from_slice(&f.footer[..8]);
    out.extend_from_slice(&len.to_le_bytes());
    out
}

/// A u8-length-prefixed string as the files store it.
pub fn lstr(s: &str) -> Vec<u8> {
    let mut v = vec![s.len() as u8];
    v.extend_from_slice(s.as_bytes());
    v
}

pub fn contains(hay: &[u8], needle: &[u8]) -> bool {
    !needle.is_empty() && hay.windows(needle.len()).any(|w| w == needle)
}

/// `hay` with every `from` replaced by `to` (equal lengths).
pub fn replace(hay: &[u8], from: &[u8], to: &[u8]) -> Vec<u8> {
    debug_assert_eq!(from.len(), to.len());
    let mut out = hay.to_vec();
    let mut i = 0;
    while i + from.len() <= out.len() {
        if &out[i..i + from.len()] == from {
            out[i..i + from.len()].copy_from_slice(to);
            i += from.len();
        } else {
            i += 1;
        }
    }
    out
}

impl Entity {
    /// The first entity named `name` in this subtree (depth first).
    pub fn find(&self, name: &str) -> Option<&Entity> {
        if self.name == name.as_bytes() {
            return Some(self);
        }
        self.kids.iter().find_map(|k| k.find(name))
    }

    pub fn find_mut(&mut self, name: &str) -> Option<&mut Entity> {
        if self.name == name.as_bytes() {
            return Some(self);
        }
        self.kids.iter_mut().find_map(|k| k.find_mut(name))
    }

    /// The path of child indices to the first entity named `name`.
    pub fn path_to(&self, name: &str) -> Option<Vec<usize>> {
        if self.name == name.as_bytes() {
            return Some(Vec::new());
        }
        for (i, k) in self.kids.iter().enumerate() {
            if let Some(mut p) = k.path_to(name) {
                p.insert(0, i);
                return Some(p);
            }
        }
        None
    }

    pub fn at(&self, path: &[usize]) -> &Entity {
        match path.split_first() {
            None => self,
            Some((&i, rest)) => self.kids[i].at(rest),
        }
    }

    pub fn at_mut(&mut self, path: &[usize]) -> &mut Entity {
        match path.split_first() {
            None => self,
            Some((&i, rest)) => self.kids[i].at_mut(rest),
        }
    }

    pub fn count(&self) -> usize {
        1 + self.kids.iter().map(Entity::count).sum::<usize>()
    }
}

pub fn salted(g: &[u8; 16], salt: &[u8; 16]) -> [u8; 16] {
    let mut n = *g;
    for (a, b) in n.iter_mut().zip(salt) {
        *a ^= b;
    }
    n
}

pub fn collect_guids(e: &Entity, out: &mut std::collections::HashSet<[u8; 16]>) {
    out.insert(e.guid);
    out.extend(e.comps.iter().map(|(g, _)| *g));
    for k in &e.kids {
        collect_guids(k, out);
    }
}

type Guids = std::collections::HashSet<[u8; 16]>;

/// A deep copy of `e` with GUIDs derived from the originals and `salt` (stable from boot to
/// boot, distinct per copy). Components that link to entities or component instances inside
/// the copied subtree (their trailers hold the GUIDs: a frame's `UseLinked` names its value
/// widget, a `ValueList` its arrows) are copied with the links pointing at the copy's own;
/// links outside it stay. Components holding one of `repl`'s strings (whole strings, any
/// length) are copied with it replaced; the others are shared.
pub fn clone_entity(comps: &mut Vec<Component>, e: &Entity, repl: &[(Vec<u8>, Vec<u8>)], salt: &[u8; 16]) -> Entity {
    clone_group(comps, &[e], repl, salt).pop().unwrap()
}

/// Copies of several subtrees with one salt: links between them point at each other's copies.
pub fn clone_group(comps: &mut Vec<Component>, roots: &[&Entity], repl: &[(Vec<u8>, Vec<u8>)], salt: &[u8; 16]) -> Vec<Entity> {
    let mut inside = Guids::new();
    for r in roots {
        collect_guids(r, &mut inside);
    }
    roots.iter().map(|r| clone_inner(comps, r, repl, salt, &inside)).collect()
}

/// A component's payload with `repl`'s strings replaced and the links into `inside` moved to
/// the copies; the title's bytes when nothing changes. Payloads the value model does not read
/// (one of the start menu's 4731) take equal-length replacements and GUID windows only.
fn transform(p: &[u8], repl: &[(Vec<u8>, Vec<u8>)], salt: &[u8; 16], inside: &Guids) -> Vec<u8> {
    if let Some(mut pl) = super::value::Payload::parse(p) {
        let mut changed = false;
        for (from, to) in repl {
            changed |= pl.value.replace_str(from, to) > 0;
        }
        for l in pl.links.iter_mut().flatten() {
            if inside.contains(&l.target) {
                l.target = salted(&l.target, salt);
                changed = true;
            }
        }
        return if changed { pl.ser() } else { p.to_vec() };
    }
    let mut p = p.to_vec();
    for (from, to) in repl {
        if from.len() == to.len() && contains(&p, from) {
            p = replace(&p, from, to);
        }
    }
    let mut i = 0;
    while i + 16 <= p.len() {
        let w: [u8; 16] = p[i..i + 16].try_into().unwrap();
        if inside.contains(&w) {
            p[i..i + 16].copy_from_slice(&salted(&w, salt));
            i += 16;
        } else {
            i += 1;
        }
    }
    p
}

fn clone_inner(comps: &mut Vec<Component>, e: &Entity, repl: &[(Vec<u8>, Vec<u8>)], salt: &[u8; 16], inside: &Guids) -> Entity {
    let mut out_comps = Vec::with_capacity(e.comps.len());
    for (g, idx) in &e.comps {
        let c = &comps[*idx as usize];
        let p = transform(&c.payload, repl, salt, inside);
        let idx = if p != c.payload {
            let cat = c.cat;
            comps.push(Component { cat, payload: p });
            (comps.len() - 1) as u32
        } else {
            *idx
        };
        out_comps.push((salted(g, salt), idx));
    }
    let kids = e.kids.iter().map(|k| clone_inner(comps, k, repl, salt, inside)).collect();
    Entity { guid: salted(&e.guid, salt), name: e.name.clone(), flags: e.flags, comps: out_comps, kids }
}

/// Every component index an entity subtree uses.
pub fn comp_indices(e: &Entity, out: &mut Vec<u32>) {
    out.extend(e.comps.iter().map(|(_, i)| *i));
    for k in &e.kids {
        comp_indices(k, out);
    }
}

pub fn fnv1a(b: &[u8]) -> u32 {
    let mut h = 0x811c_9dc5u32;
    for &c in b {
        h = (h ^ c as u32).wrapping_mul(0x0100_0193);
    }
    h
}

/// 16 salt bytes from a name.
pub fn salt(name: &str) -> [u8; 16] {
    let mut s = [0u8; 16];
    for (k, chunk) in s.chunks_mut(4).enumerate() {
        let h = fnv1a(format!("{name}#{k}").as_bytes()) | 1;
        chunk.copy_from_slice(&h.to_le_bytes());
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small file of the layout (two components, a root with a child) parses, serializes back
    /// to the same bytes, and a clone with a replaced string copies only that component.
    #[test]
    fn round_trip_and_clone() {
        let mut d = Vec::new();
        d.extend_from_slice(&2u32.to_le_bytes());
        for (cat, payload) in [(0x7bf3_b2fbu32, b"\x07Setting".to_vec()), (0xf4ee_124d, vec![0u8; 200])] {
            d.extend_from_slice(&cat.to_le_bytes());
            put_leb(&mut d, payload.len() as u64);
            d.extend_from_slice(&payload);
        }
        d.extend_from_slice(&1u32.to_le_bytes());
        let kid = Entity { guid: [2; 16], name: b"Kid".to_vec(), flags: 0, comps: vec![([4; 16], 1)], kids: vec![] };
        let root = Entity { guid: [1; 16], name: b"Root".to_vec(), flags: 7, comps: vec![([3; 16], 0), ([5; 16], 1)], kids: vec![kid] };
        put_entity(&mut d, &root);
        let len = d.len() as u32;
        d.extend_from_slice(b"PRPG");
        d.extend_from_slice(&0x86u32.to_le_bytes());
        d.extend_from_slice(&len.to_le_bytes());
        let mut f = parse(&d).expect("parses");
        assert_eq!(serialize(&f), d);
        assert_eq!(f.roots[0].count(), 2);
        let c = clone_entity(&mut f.comps, &f.roots[0], &[(b"Setting".to_vec(), b"PCthing".to_vec())], &salt("x"));
        assert_eq!(f.comps.len(), 3);
        assert_eq!(c.comps[0].1, 2);
        assert_eq!(c.comps[1].1, 1);
        assert_eq!(f.comps[2].payload, b"\x07PCthing");
        assert_ne!(c.guid, f.roots[0].guid);
        f.roots.push(c);
        let out = serialize(&f);
        assert!(parse(&out).is_some());
    }
}
