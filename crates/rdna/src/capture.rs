//! Frame captures (`capture-<flip>.txt` + `shaders/` written by the runtime):
//! parsing and translation of every captured shader with the register state
//! recorded for its first use. Used by `rdnadis --translate` and the
//! runtime's pipeline test.

use crate::analysis::ResKind;
use crate::translate::{ImageDim, MrtSpec, NumClass, Output, ResSpec, Spec, Stage};
use std::collections::BTreeMap;

/// Shader use recorded in a frame capture.
#[derive(Clone, Debug)]
pub struct Use {
    pub stage: Stage,
    pub rsrc2: u32,
    pub users: u16,
    pub threads: [u32; 3],
    pub col_format: u32,
    pub ps_input_addr: u32,
}

fn hex(s: &str) -> u32 {
    u32::from_str_radix(s.trim_start_matches("0x"), 16).unwrap_or(0)
}

/// Parses `capture-<flip>.txt`: shader address -> first use.
pub fn parse(text: &str) -> BTreeMap<u64, Use> {
    let mut uses = BTreeMap::new();
    let mut threads = [1u32; 3];
    let mut pending: Vec<(u64, Use)> = Vec::new();
    let flush = |pending: &mut Vec<(u64, Use)>, uses: &mut BTreeMap<u64, Use>| {
        for (a, u) in pending.drain(..) {
            uses.entry(a).or_insert(u);
        }
    };
    for line in text.lines() {
        let t = line.trim_start();
        let w: Vec<&str> = t.split_whitespace().collect();
        if t.starts_with("dispatch") || t.starts_with("draw") {
            flush(&mut pending, &mut uses);
            if let Some(i) = w.iter().position(|x| *x == "threads") {
                let v: Vec<u32> = w[i + 1].split('x').map(|x| x.parse().unwrap_or(1)).collect();
                threads = [v[0], *v.get(1).unwrap_or(&1), *v.get(2).unwrap_or(&1)];
            }
        } else if matches!(w.first(), Some(&"cs" | &"ps" | &"es")) && w.len() >= 7 {
            let stage = match w[0] {
                "cs" => Stage::Compute,
                "ps" => Stage::Fragment,
                _ => Stage::Vertex,
            };
            let addr = u64::from_str_radix(w[1].trim_start_matches("0x"), 16).unwrap_or(0);
            let rsrc2 = hex(w[5]);
            let users = w[6].trim_start_matches("user[").trim_end_matches(']').parse().unwrap_or(0);
            pending.push((addr, Use { stage, rsrc2, users, threads, col_format: 0, ps_input_addr: 0 }));
        } else if w.first() == Some(&"viewport") {
            let col = w.iter().position(|x| *x == "col_format").map(|i| hex(w[i + 1])).unwrap_or(0);
            let psi = w.iter().position(|x| *x == "ps_input").map(|i| hex(w[i + 1].split('/').nth(1).unwrap_or("0"))).unwrap_or(0);
            for (_, u) in pending.iter_mut() {
                u.col_format = col;
                u.ps_input_addr = psi;
            }
        }
    }
    flush(&mut pending, &mut uses);
    uses
}

/// One captured shader and its translation.
pub struct Item {
    pub addr: u64,
    pub stage: Stage,
    pub local: [u32; 3],
    pub out: Result<Output, String>,
}

fn image_dim(d: u32) -> ImageDim {
    match d {
        0 => ImageDim::D1,
        1 => ImageDim::D2,
        2 => ImageDim::D3,
        3 => ImageDim::Cube,
        4 => ImageDim::D1Array,
        5 => ImageDim::D2Array,
        6 => ImageDim::D2Msaa,
        _ => ImageDim::D2MsaaArray,
    }
}

fn translate_one(shader_dir: &std::path::Path, addr: u64, u: &Use) -> Result<Output, String> {
    let bytes = std::fs::read(shader_dir.join(format!("{addr:012x}.bin"))).map_err(|_| "binary missing".to_string())?;
    let code: Vec<u32> = bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let insts = crate::decode_program(&code).map_err(|e| format!("decode error {}", e.0))?;
    let user_base = if u.stage == Stage::Vertex { 8 } else { 0 };
    let users = if u.stage == Stage::Vertex && u.users == 0 { 24 } else { u.users };
    let plan = crate::analysis::analyze(&insts, users, user_base, addr);
    let wave64 = !insts.iter().any(|i| i.name().is_some_and(|n| n.ends_with("_saveexec_b32") || n.ends_with("_wrexec_b32")));
    let resources: Vec<ResSpec> = plan
        .resources
        .iter()
        .enumerate()
        .map(|(r, res)| match res.kind {
            ResKind::Buffer => ResSpec::Buffer { format: 77, dst_sel: 0xFAC },
            ResKind::Global => ResSpec::Global,
            ResKind::Sampler => ResSpec::Sampler,
            ResKind::Image => {
                let dim = insts
                    .iter()
                    .find(|i| i.enc == crate::Enc::Mimg && plan.inst_res.get(&i.pc).is_some_and(|x| x.0 == r))
                    .map(|i| image_dim(i.mimg_dim()))
                    .unwrap_or(ImageDim::D2);
                ResSpec::Image { dim, class: NumClass::Float }
            }
        })
        .collect();
    let mut mrt = [MrtSpec::default(); 8];
    for (i, m) in mrt.iter_mut().enumerate() {
        m.export_format = ((u.col_format >> (4 * i)) & 0xF) as u8;
    }
    let spec = Spec {
        probe: None,
        dump: None,
        stage: u.stage,
        user_sgprs: users,
        user_base,
        code_base: addr,
        wave64,
        ps_input_addr: u.ps_input_addr,
        ps_inputs: Vec::new(),
        mrt,
        cs_local_size: u.threads,
        cs_rsrc2: u.rsrc2,
        resources,
        binding_base: 0,
    };
    crate::translate::translate(&insts, &plan, &spec)
}

/// Translates every shader of a capture (stage and register state from the
/// capture; buffers specialized as 32_32_32_32_FLOAT, images as float).
pub fn translate_all(capture: &str) -> Result<Vec<Item>, String> {
    let text = std::fs::read_to_string(capture).map_err(|e| format!("{capture}: {e}"))?;
    let shader_dir = std::path::Path::new(capture).parent().unwrap_or(std::path::Path::new(".")).join("shaders");
    Ok(parse(&text).into_iter().map(|(addr, u)| Item { addr, stage: u.stage, local: u.threads, out: translate_one(&shader_dir, addr, &u) }).collect())
}
