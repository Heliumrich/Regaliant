//! `DS_ITEM_CENSUS=1` (diagnosis; FINDINGS.md 2026-11-18): a frame's prepared draws against
//! the frame before, by what they draw. A draw's stages are prepared from the title's user
//! data (pointers into memory the title writes anew every frame, so by program and user data
//! no draw repeats: FINDINGS-ARCHIVE.md 2026-10-21); this census asks whether they repeat by
//! content: the words their preparations read from memory, and the buffers, images and
//! samplers they resolved to. A draw kept across frames by its content is only worth building
//! if most of a heavy frame's draws find themselves in the frame before.
//!
//! Per pass (the depth target and the colour slots, `exec::pass_key`) and program pair, a draw
//! of this frame is looked up in the frame before by: everything (user data, words read,
//! resources); the content (words read and resources); the words alone; the resources alone;
//! the mesh (the first index's address and the count). Where the resources match and the
//! words do not, the words that differ are counted by position, and likewise the user data
//! and, for a mesh match, the resources: what a key would have to leave out.
//!
//! Repeats (a draw that differs from the one before only in its arguments: more instances of
//! one mesh) are recorded without a preparation and are not in it. Control `itemcensus`
//! prints the totals since the last report.

use parking_lot::Mutex;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

pub fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_ITEM_CENSUS").ok().as_deref() == Some("1"))
}

/// One prepared draw.
struct Rec {
    /// The pass: the depth target's address and the packed sizes and slot count.
    pass: (u64, u64),
    pair: (u64, u64),
    user: [Vec<u32>; 2],
    words: [Vec<u32>; 2],
    /// The guest ranges the words were read from (address and bytes), in their order.
    reads: [Vec<(u64, u64)>; 2],
    /// One hash per resolved resource, the vertex stage's then the pixel stage's.
    res: Vec<u64>,
    /// The first index's address and the count.
    mesh: (u64, u32),
}

fn h(parts: &[&dyn Fn(&mut std::collections::hash_map::DefaultHasher)]) -> u64 {
    let mut s = std::collections::hash_map::DefaultHasher::new();
    for p in parts {
        p(&mut s);
    }
    s.finish()
}

impl Rec {
    fn base(&self, s: &mut std::collections::hash_map::DefaultHasher) {
        self.pass.hash(s);
        self.pair.hash(s);
    }
    fn k_exact(&self) -> u64 {
        h(&[&|s| self.base(s), &|s| self.user.hash(s), &|s| self.words.hash(s), &|s| self.res.hash(s)])
    }
    fn k_content(&self) -> u64 {
        h(&[&|s| self.base(s), &|s| self.words.hash(s), &|s| self.res.hash(s)])
    }
    fn k_words(&self) -> u64 {
        h(&[&|s| self.base(s), &|s| self.words.hash(s)])
    }
    fn k_res(&self) -> u64 {
        h(&[&|s| self.base(s), &|s| self.res.hash(s)])
    }
    fn k_mesh(&self) -> u64 {
        h(&[&|s| self.base(s), &|s| self.mesh.hash(s)])
    }
}

/// A pass's or a pair's counts: draws, those found in the frame before by everything, by
/// content, by the words, by the resources, by the mesh.
#[derive(Default, Clone, Copy)]
struct Hits {
    draws: u64,
    exact: u64,
    content: u64,
    words: u64,
    res: u64,
    mesh: u64,
}

/// What differs where a pair's draw matches the frame before's only in part: the positions
/// of differing words (the two stages' words in a row, 96 at most) among draws whose
/// resources match, of differing user-data words (64) among the same, and of differing
/// resources (48) among draws whose mesh matches; with the draws compared for each.
struct Diffs {
    words: [u32; 96],
    words_n: u32,
    /// Compared draws whose word counts differ (not counted by position).
    words_len: u32,
    user: [u32; 64],
    user_n: u32,
    res: [u32; 48],
    res_n: u32,
    /// Among draws whose resources match the frame before's: the ranges and words read, those
    /// read from the same address as then, and of the words read from another address those
    /// that hold the same value and those that do not.
    ranges: u64,
    ranges_same: u64,
    words_read: u64,
    words_same_addr: u64,
    words_moved_same: u64,
    words_moved_diff: u64,
}

impl Default for Diffs {
    fn default() -> Diffs {
        Diffs { words: [0; 96], words_n: 0, words_len: 0, user: [0; 64], user_n: 0, res: [0; 48], res_n: 0, ranges: 0, ranges_same: 0, words_read: 0, words_same_addr: 0, words_moved_same: 0, words_moved_diff: 0 }
    }
}

#[derive(Default)]
struct State {
    cur: Vec<Rec>,
    prev: Vec<Rec>,
    frames: u64,
    by_pass: HashMap<(u64, u64), Hits>,
    by_pair: HashMap<((u64, u64), (u64, u64)), (Hits, Diffs)>,
}

static STATE: Mutex<Option<State>> = Mutex::new(None);

/// A draw whose stages were prepared (`Exec::draw_inner`).
pub fn note(pass: (u64, u64), vs: &super::shaders::Prepared, ps: Option<&super::shaders::Prepared>, index: Option<u64>, count: u32) {
    let res_hash = |r: &super::shaders::Resolved| {
        let mut s = std::collections::hash_map::DefaultHasher::new();
        match r {
            super::shaders::Resolved::Buffer(v) => (1u8, v).hash(&mut s),
            super::shaders::Resolved::Global(a, n) => (2u8, a, n).hash(&mut s),
            super::shaders::Resolved::Image(t) => (3u8, t).hash(&mut s),
            super::shaders::Resolved::ImageTable(t) => (4u8, t).hash(&mut s),
            super::shaders::Resolved::Sampler(w) => (5u8, w).hash(&mut s),
        }
        s.finish()
    };
    let mut res: Vec<u64> = vs.resources.iter().map(res_hash).collect();
    // (A mark between the stages: a resource moved from one to the other is not a match.)
    res.push(u64::MAX);
    let empty: &[u32] = &[];
    let (pu, pw, pr) = match ps {
        Some(p) => {
            res.extend(p.resources.iter().map(res_hash));
            (p.user.words(), &p.words[..], p.reads.to_vec())
        }
        None => (empty, empty, Vec::new()),
    };
    let rec = Rec { pass, pair: (vs.addr, ps.map_or(0, |p| p.addr)), user: [vs.user.words().to_vec(), pu.to_vec()], words: [vs.words.to_vec(), pw.to_vec()], reads: [vs.reads.to_vec(), pr], res, mesh: (index.unwrap_or(0), count) };
    STATE.lock().get_or_insert_with(State::default).cur.push(rec);
}

/// The graphics queue's flip: the frame's draws are looked up in the frame before's.
pub fn flip() {
    let mut g = STATE.lock();
    let s = g.get_or_insert_with(State::default);
    let cur = std::mem::take(&mut s.cur);
    let prev = std::mem::take(&mut s.prev);
    if !prev.is_empty() && !cur.is_empty() {
        s.frames += 1;
        let set = |f: &dyn Fn(&Rec) -> u64| -> HashMap<u64, usize> { prev.iter().enumerate().map(|(i, r)| (f(r), i)).collect() };
        let (exact, content, words, res, mesh) = (set(&Rec::k_exact), set(&Rec::k_content), set(&Rec::k_words), set(&Rec::k_res), set(&Rec::k_mesh));
        for r in &cur {
            let hit = [exact.contains_key(&r.k_exact()), content.contains_key(&r.k_content()), words.contains_key(&r.k_words()), res.contains_key(&r.k_res()), mesh.contains_key(&r.k_mesh())];
            let add = |t: &mut Hits| {
                t.draws += 1;
                t.exact += hit[0] as u64;
                t.content += hit[1] as u64;
                t.words += hit[2] as u64;
                t.res += hit[3] as u64;
                t.mesh += hit[4] as u64;
            };
            add(s.by_pass.entry(r.pass).or_default());
            let e = s.by_pair.entry((r.pass, r.pair)).or_default();
            add(&mut e.0);
            let d = &mut e.1;
            // The resources match, the content does not: which words differ; and the user
            // data against the same draw.
            if let (false, Some(&i)) = (hit[1], res.get(&r.k_res())) {
                let p = &prev[i];
                let (a, b): (Vec<u32>, Vec<u32>) = (r.words.concat(), p.words.concat());
                if a.len() != b.len() {
                    d.words_len += 1;
                } else {
                    d.words_n += 1;
                    for (k, (x, y)) in a.iter().zip(&b).enumerate().take(96) {
                        d.words[k] += (x != y) as u32;
                    }
                }
            }
            if let (false, Some(&i)) = (hit[0], content.get(&r.k_content())) {
                let p = &prev[i];
                let (a, b): (Vec<u32>, Vec<u32>) = (r.user.concat(), p.user.concat());
                d.user_n += 1;
                for (k, (x, y)) in a.iter().zip(&b).enumerate().take(64) {
                    d.user[k] += (x != y) as u32;
                }
            }
            // The resources match: where the words were read from against that draw's.
            if let Some(&i) = res.get(&r.k_res()) {
                let p = &prev[i];
                for st in 0..2 {
                    let (mut at, mut pat) = (0usize, 0usize);
                    for (k, &(a, len)) in r.reads[st].iter().enumerate() {
                        let n = (len / 4) as usize;
                        d.ranges += 1;
                        d.words_read += n as u64;
                        let old = p.reads[st].get(k).copied();
                        let same_addr = old == Some((a, len));
                        if same_addr {
                            d.ranges_same += 1;
                            d.words_same_addr += n as u64;
                        } else if let Some((_, olen)) = old.filter(|o| o.1 == len) {
                            let (x, y) = (r.words[st].get(at..at + n), p.words[st].get(pat..pat + (olen / 4) as usize));
                            if let (Some(x), Some(y)) = (x, y) {
                                let same = x.iter().zip(y).filter(|(u, v)| u == v).count() as u64;
                                d.words_moved_same += same;
                                d.words_moved_diff += n as u64 - same;
                            }
                        } else {
                            d.words_moved_diff += n as u64;
                        }
                        at += n;
                        pat += old.map_or(0, |o| (o.1 / 4) as usize);
                    }
                }
            }
            // The mesh matches, the resources do not: which resources differ.
            if let (false, Some(&i)) = (hit[3], mesh.get(&r.k_mesh())) {
                let p = &prev[i];
                d.res_n += 1;
                for (k, (x, y)) in r.res.iter().zip(&p.res).enumerate().take(48) {
                    d.res[k] += (x != y) as u32;
                }
            }
        }
    }
    s.prev = cur;
}

/// The totals since the last report (control `itemcensus`), counted afresh.
pub fn report() -> String {
    let mut g = STATE.lock();
    let Some(s) = g.as_mut() else { return "item census: off (DS_ITEM_CENSUS=1)".into() };
    let frames = s.frames.max(1) as f64;
    let pct = |n: u64, d: u64| if d == 0 { 0.0 } else { 100.0 * n as f64 / d as f64 };
    let line = |t: &Hits| format!("{:.0} draws a frame; found in the frame before: by everything {:.1}%, by content (words and resources) {:.1}%, by the words {:.1}%, by the resources {:.1}%, by the mesh {:.1}%", t.draws as f64 / frames, pct(t.exact, t.draws), pct(t.content, t.draws), pct(t.words, t.draws), pct(t.res, t.draws), pct(t.mesh, t.draws));
    let mut out = format!("item census: {} frames compared with the frame before\n", s.frames);
    let mut passes: Vec<_> = s.by_pass.iter().collect();
    passes.sort_by(|a, b| b.1.draws.cmp(&a.1.draws));
    let name = |p: &(u64, u64)| format!("depth {:#x} ({}x{}), {} colour slot(s) of {}x{}", p.0, p.1 & 0x3FFF, (p.1 >> 14) & 0x3FFF, p.1 >> 56, (p.1 >> 28) & 0x3FFF, (p.1 >> 42) & 0x3FFF);
    for (p, t) in passes.iter().take(10) {
        out += &format!("  pass {}: {}\n", name(p), line(t));
    }
    let mut pairs: Vec<_> = s.by_pair.iter().collect();
    pairs.sort_by(|a, b| b.1.0.draws.cmp(&a.1.0.draws));
    let top = |v: &[u32], n: u32| -> String {
        let mut idx: Vec<(usize, u32)> = v.iter().copied().enumerate().filter(|e| e.1 > 0).collect();
        idx.sort_by(|a, b| b.1.cmp(&a.1));
        idx.iter().take(10).map(|(i, c)| format!("{i}: {:.0}%", 100.0 * *c as f64 / n.max(1) as f64)).collect::<Vec<_>>().join(", ")
    };
    for ((p, pair), (t, d)) in pairs.iter().take(14) {
        out += &format!("  vs {:#x} ps {:#x} in depth {:#x}/{} slots: {}\n", pair.0, pair.1, p.0, p.1 >> 56, line(t));
        if d.words_n + d.words_len > 0 {
            out += &format!("    resources matched, content not ({} draws; {} with another word count): words differing at {}\n", d.words_n, d.words_len, top(&d.words, d.words_n));
        }
        if d.user_n > 0 {
            out += &format!("    content matched, user data not ({} draws): user words differing at {}\n", d.user_n, top(&d.user, d.user_n));
        }
        if d.res_n > 0 {
            out += &format!("    mesh matched, resources not ({} draws): resources differing at {}\n", d.res_n, top(&d.res, d.res_n));
        }
        if d.ranges > 0 {
            let n = (t.res.max(1)) as f64;
            out += &format!(
                "    where the resources matched: {:.1} ranges and {:.1} words read a draw; {:.1} ranges and {:.1} words at the frame before's addresses; of the words read elsewhere {:.1} hold the same value and {:.1} another\n",
                d.ranges as f64 / n,
                d.words_read as f64 / n,
                d.ranges_same as f64 / n,
                d.words_same_addr as f64 / n,
                d.words_moved_same as f64 / n,
                d.words_moved_diff as f64 / n
            );
        }
    }
    s.frames = 0;
    s.by_pass.clear();
    s.by_pair.clear();
    out
}
