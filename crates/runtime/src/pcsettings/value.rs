//! A component's payload in the title's UI files, read into a tree so the PC settings can edit
//! it: a typed value (class hash, LEB128 size, a body of a u16 field count and fields, or raw
//! bytes) and a trailer (LEB128 size, two link lists). A field is a name hash, a size and a
//! payload: an inline object (class hash, u16 count, fields), a list of typed values (u16
//! count), an array of structs (u16 count, each a u16 field count and fields), a u8-length
//! string, or raw bytes; the first reading that consumes the payload exactly is taken (as
//! `tools/cgpr.py` does), and every reading writes the same bytes back (checked before an edit).
//! A link is an event hash, a target (a component instance's GUID), two words, a byte, an
//! action hash and a typed parameter (class 0xFFFFFFFF and size 0: none).

#[derive(Clone, Debug, PartialEq)]
pub enum Node {
    Obj { hash: u32, fields: Vec<(u32, Node)> },
    List(Vec<Typed>),
    Structs(Vec<Vec<(u32, Node)>>),
    Str(Vec<u8>),
    Raw(Vec<u8>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Typed {
    pub hash: u32,
    pub body: Body,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Body {
    Fields(Vec<(u32, Node)>),
    Raw(Vec<u8>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Link {
    pub event: u32,
    pub target: [u8; 16],
    pub a: u32,
    pub b: u32,
    pub c: u8,
    pub action: u32,
    /// The parameter as bytes (a typed value).
    pub param: Vec<u8>,
}

/// A component's payload: its value and its two link lists.
#[derive(Clone, Debug, PartialEq)]
pub struct Payload {
    pub value: Typed,
    pub links: [Vec<Link>; 2],
}

type R<T> = Result<T, ()>;

fn u16_at(d: &[u8], i: usize) -> R<u16> {
    Ok(u16::from_le_bytes(d.get(i..i + 2).ok_or(())?.try_into().unwrap()))
}

fn u32_at(d: &[u8], i: usize) -> R<u32> {
    Ok(u32::from_le_bytes(d.get(i..i + 4).ok_or(())?.try_into().unwrap()))
}

fn leb(d: &[u8], i: usize) -> R<(usize, usize)> {
    let mut v = 0usize;
    let mut s = 0;
    let mut i = i;
    loop {
        let b = *d.get(i).ok_or(())?;
        i += 1;
        v |= ((b & 0x7f) as usize) << s;
        s += 7;
        if b < 0x80 {
            return Ok((v, i));
        }
        if s > 35 {
            return Err(());
        }
    }
}

fn put_leb(out: &mut Vec<u8>, mut v: usize) {
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

fn fields(d: &[u8], mut i: usize, end: usize, count: usize) -> R<Vec<(u32, Node)>> {
    let mut out = Vec::with_capacity(count.min(256));
    for _ in 0..count {
        if i + 5 > end {
            return Err(());
        }
        let name = u32_at(d, i)?;
        let (size, j) = leb(d, i + 4)?;
        if j + size > end {
            return Err(());
        }
        out.push((name, payload(d, j, j + size)));
        i = j + size;
    }
    if i != end {
        return Err(());
    }
    Ok(out)
}

fn payload(d: &[u8], i: usize, end: usize) -> Node {
    let n = end - i;
    if n >= 6 {
        if let (Ok(cls), Ok(cnt)) = (u32_at(d, i), u16_at(d, i + 4)) {
            if cnt != 0 && cnt < 4096 {
                if let Ok(f) = fields(d, i + 6, end, cnt as usize) {
                    return Node::Obj { hash: cls, fields: f };
                }
            }
        }
    }
    if n >= 2 {
        if let Ok(cnt) = u16_at(d, i) {
            if cnt != 0 {
                let mut items = Vec::new();
                let mut k = i + 2;
                let mut ok = true;
                for _ in 0..cnt {
                    match typed(d, k, end) {
                        Ok((v, e)) => {
                            items.push(v);
                            k = e;
                        }
                        Err(()) => {
                            ok = false;
                            break;
                        }
                    }
                }
                if ok && k == end {
                    return Node::List(items);
                }
            }
        }
    }
    if n >= 4 {
        if let Ok(cnt) = u16_at(d, i) {
            if cnt != 0 && cnt < 4096 {
                if let Ok(s) = structs(d, i + 2, end, cnt as usize) {
                    return Node::Structs(s);
                }
            }
        }
    }
    if n >= 1 && d[i] as usize == n - 1 && d[i + 1..end].iter().all(|&c| (32..127).contains(&c)) {
        return Node::Str(d[i + 1..end].to_vec());
    }
    Node::Raw(d[i..end].to_vec())
}

fn structs(d: &[u8], mut k: usize, end: usize, cnt: usize) -> R<Vec<Vec<(u32, Node)>>> {
    let mut out = Vec::with_capacity(cnt);
    for _ in 0..cnt {
        let fc = u16_at(d, k)? as usize;
        k += 2;
        let mut f = Vec::with_capacity(fc.min(256));
        for _ in 0..fc {
            if k + 5 > end {
                return Err(());
            }
            let name = u32_at(d, k)?;
            let (size, j) = leb(d, k + 4)?;
            if j + size > end {
                return Err(());
            }
            f.push((name, payload(d, j, j + size)));
            k = j + size;
        }
        out.push(f);
    }
    if k != end {
        return Err(());
    }
    Ok(out)
}

fn typed(d: &[u8], i: usize, end: usize) -> R<(Typed, usize)> {
    if i + 5 > end {
        return Err(());
    }
    let cls = u32_at(d, i)?;
    let (size, j) = leb(d, i + 4)?;
    if j + size > end {
        return Err(());
    }
    let body_end = j + size;
    if size >= 2 {
        let cnt = u16_at(d, j)? as usize;
        if let Ok(f) = fields(d, j + 2, body_end, cnt) {
            return Ok((Typed { hash: cls, body: Body::Fields(f) }, body_end));
        }
    }
    Ok((Typed { hash: cls, body: Body::Raw(d[j..body_end].to_vec()) }, body_end))
}

fn put_fields(out: &mut Vec<u8>, f: &[(u32, Node)]) {
    for (h, v) in f {
        let p = ser_node(v);
        out.extend_from_slice(&h.to_le_bytes());
        put_leb(out, p.len());
        out.extend_from_slice(&p);
    }
}

pub fn ser_node(n: &Node) -> Vec<u8> {
    let mut out = Vec::new();
    match n {
        Node::Obj { hash, fields } => {
            out.extend_from_slice(&hash.to_le_bytes());
            out.extend_from_slice(&(fields.len() as u16).to_le_bytes());
            put_fields(&mut out, fields);
        }
        Node::List(items) => {
            out.extend_from_slice(&(items.len() as u16).to_le_bytes());
            for t in items {
                out.extend_from_slice(&ser_typed(t));
            }
        }
        Node::Structs(items) => {
            out.extend_from_slice(&(items.len() as u16).to_le_bytes());
            for f in items {
                out.extend_from_slice(&(f.len() as u16).to_le_bytes());
                put_fields(&mut out, f);
            }
        }
        Node::Str(s) => {
            out.push(s.len() as u8);
            out.extend_from_slice(s);
        }
        Node::Raw(r) => out.extend_from_slice(r),
    }
    out
}

pub fn ser_typed(t: &Typed) -> Vec<u8> {
    let body = match &t.body {
        Body::Raw(r) => r.clone(),
        Body::Fields(f) => {
            let mut b = (f.len() as u16).to_le_bytes().to_vec();
            put_fields(&mut b, f);
            b
        }
    };
    let mut out = t.hash.to_le_bytes().to_vec();
    put_leb(&mut out, body.len());
    out.extend_from_slice(&body);
    out
}

fn links(d: &[u8], mut j: usize) -> R<(Vec<Link>, usize)> {
    let n = u32_at(d, j)? as usize;
    j += 4;
    let mut out = Vec::with_capacity(n.min(1024));
    for _ in 0..n {
        let event = u32_at(d, j)?;
        let target: [u8; 16] = d.get(j + 4..j + 20).ok_or(())?.try_into().unwrap();
        let a = u32_at(d, j + 20)?;
        let b = u32_at(d, j + 24)?;
        let c = *d.get(j + 28).ok_or(())?;
        let action = u32_at(d, j + 29)?;
        let k = j + 33;
        let (psize, k2) = leb(d, k + 4)?;
        let pend = k2 + psize;
        let param = d.get(k..pend).ok_or(())?.to_vec();
        out.push(Link { event, target, a, b, c, action, param });
        j = pend;
    }
    Ok((out, j))
}

fn put_links(out: &mut Vec<u8>, l: &[Link]) {
    out.extend_from_slice(&(l.len() as u32).to_le_bytes());
    for k in l {
        out.extend_from_slice(&k.event.to_le_bytes());
        out.extend_from_slice(&k.target);
        out.extend_from_slice(&k.a.to_le_bytes());
        out.extend_from_slice(&k.b.to_le_bytes());
        out.push(k.c);
        out.extend_from_slice(&k.action.to_le_bytes());
        out.extend_from_slice(&k.param);
    }
}

impl Payload {
    /// The payload read, if it writes back to the same bytes.
    pub fn parse(p: &[u8]) -> Option<Payload> {
        let (value, end) = typed(p, 0, p.len()).ok()?;
        let (size, j) = leb(p, end).ok()?;
        if j + size != p.len() {
            return None;
        }
        let (l0, k) = links(p, j).ok()?;
        let (l1, k) = links(p, k).ok()?;
        if k != p.len() {
            return None;
        }
        let out = Payload { value, links: [l0, l1] };
        (out.ser() == p).then_some(out)
    }

    pub fn ser(&self) -> Vec<u8> {
        let mut out = ser_typed(&self.value);
        let mut t = Vec::new();
        put_links(&mut t, &self.links[0]);
        put_links(&mut t, &self.links[1]);
        put_leb(&mut out, t.len());
        out.extend_from_slice(&t);
        out
    }
}

/// FNV-1a of a field or class name (`cgpr::fnv1a`).
pub fn h(name: &str) -> u32 {
    super::cgpr::fnv1a(name.as_bytes())
}

impl Node {
    pub fn fields_mut(&mut self) -> Option<&mut Vec<(u32, Node)>> {
        match self {
            Node::Obj { fields, .. } => Some(fields),
            _ => None,
        }
    }

    /// The field `hash` of an object.
    pub fn field_mut(&mut self, hash: u32) -> Option<&mut Node> {
        self.fields_mut()?.iter_mut().find(|(k, _)| *k == hash).map(|(_, v)| v)
    }

    /// Every string in this subtree equal to `from` replaced by `to` (any length); the count.
    pub fn replace_str(&mut self, from: &[u8], to: &[u8]) -> usize {
        match self {
            Node::Str(s) if s.as_slice() == from => {
                *s = to.to_vec();
                1
            }
            Node::Obj { fields, .. } => fields.iter_mut().map(|(_, v)| v.replace_str(from, to)).sum(),
            Node::List(items) => items.iter_mut().map(|t| t.replace_str(from, to)).sum(),
            Node::Structs(items) => items.iter_mut().flat_map(|f| f.iter_mut()).map(|(_, v)| v.replace_str(from, to)).sum(),
            _ => 0,
        }
    }
}

impl Typed {
    pub fn fields_mut(&mut self) -> Option<&mut Vec<(u32, Node)>> {
        match &mut self.body {
            Body::Fields(f) => Some(f),
            Body::Raw(_) => None,
        }
    }

    pub fn field_mut(&mut self, hash: u32) -> Option<&mut Node> {
        self.fields_mut()?.iter_mut().find(|(k, _)| *k == hash).map(|(_, v)| v)
    }

    pub fn replace_str(&mut self, from: &[u8], to: &[u8]) -> usize {
        match &mut self.body {
            Body::Fields(f) => f.iter_mut().map(|(_, v)| v.replace_str(from, to)).sum(),
            Body::Raw(_) => 0,
        }
    }
}

/// The field `hash` of a struct.
pub fn struct_field(f: &mut [(u32, Node)], hash: u32) -> Option<&mut Node> {
    f.iter_mut().find(|(k, _)| *k == hash).map(|(_, v)| v)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A payload with every kind of node and two link lists writes back the same bytes, and an
    /// edited string changes the sizes around it.
    #[test]
    fn payload_round_trip_and_edit() {
        let inner = Node::Obj { hash: 0x82ac_54e8, fields: vec![(0xeb93_826d, Node::Str(b"Setting_X".to_vec()))] };
        let opts = Node::Structs(vec![
            vec![(1, Node::Raw(vec![0, 0, 0, 0])), (2, Node::Obj { hash: 7, fields: vec![(3, Node::Str(b"KEY_A".to_vec()))] })],
            vec![(1, Node::Raw(vec![2, 0, 0, 0])), (2, Node::Obj { hash: 7, fields: vec![(3, Node::Str(b"KEY_B".to_vec()))] })],
        ]);
        let value = Typed {
            hash: 0x1234_5678,
            body: Body::Fields(vec![
                (10, inner),
                (11, opts),
                (12, Node::List(vec![Typed { hash: 9, body: Body::Raw(vec![1, 2, 3]) }])),
                (13, Node::Raw(vec![0x3f, 0, 0, 0])),
            ]),
        };
        let link = Link { event: 0xd25a_0297, target: [5; 16], a: 0, b: 0, c: 1, action: 0x3ff1_b799, param: vec![0xff, 0xff, 0xff, 0xff, 0] };
        let p = Payload { value, links: [vec![link.clone(), link], vec![]] };
        let bytes = p.ser();
        let q = Payload::parse(&bytes).expect("parses");
        assert_eq!(q.ser(), bytes);
        let mut r = q.clone();
        assert_eq!(r.value.replace_str(b"Setting_X", b"Setting_PC_RenderScale"), 1);
        let b2 = r.ser();
        assert!(b2.len() >= bytes.len() + 13);
        let mut s = Payload::parse(&b2).expect("the edited payload parses");
        assert_eq!(s.value.replace_str(b"Setting_PC_RenderScale", b"x"), 1);
    }
}
