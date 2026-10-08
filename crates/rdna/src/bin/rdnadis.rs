//! rdnadis <file.bin>...   disassembles RDNA (gfx10) shader binaries; with
//! several files, prints only a summary (unknown opcodes, instruction mix).

use std::collections::BTreeMap;

fn main() {
    let mut files: Vec<String> = std::env::args().skip(1).collect();
    // --translate <capture.txt> <out dir>: translates every shader of a frame
    // capture (stage and register state from the capture) to SPIR-V.
    if files.first().is_some_and(|f| f == "--translate") {
        if files.len() != 3 {
            eprintln!("usage: rdnadis --translate <capture.txt> <out dir>");
            std::process::exit(2);
        }
        std::process::exit(translate_capture(&files[1], &files[2]));
    }
    // --analyze: descriptor/scalar-load provenance per file (32 user SGPRs assumed).
    let analyze = files.first().is_some_and(|f| f == "--analyze");
    if analyze {
        files.remove(0);
        let (mut ok, mut bad) = (0, 0);
        for f in &files {
            let bytes = std::fs::read(f).expect("read shader");
            let code: Vec<u32> = bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            let Ok(insts) = rdna::decode_program(&code) else { continue };
            let plan = rdna::analysis::analyze(&insts, 32, 0, 0);
            if plan.errors.is_empty() {
                ok += 1;
            } else {
                bad += 1;
                println!("{f}: {} resource(s), {} load(s), errors:", plan.resources.len(), plan.loads.len());
                for e in plan.errors.iter().take(4) {
                    println!("    {e}");
                }
            }
        }
        println!("{ok} shader(s) fully resolved, {bad} with unresolved descriptors or loads");
        return;
    }
    if files.is_empty() {
        eprintln!("usage: rdnadis <file.bin>...");
        std::process::exit(2);
    }
    let single = files.len() == 1;
    let mut mix: BTreeMap<String, u64> = BTreeMap::new();
    let mut unknown: BTreeMap<String, u64> = BTreeMap::new();
    let mut failed = 0;
    for f in &files {
        let bytes = std::fs::read(f).expect("read shader");
        let code: Vec<u32> = bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        match rdna::decode_program(&code) {
            Ok(insts) => {
                for i in &insts {
                    match i.name() {
                        Some(n) => *mix.entry(n.to_string()).or_default() += 1,
                        None => *unknown.entry(format!("{:?} op {:#x} ({:#010x})", i.enc, i.op, i.w0)).or_default() += 1,
                    }
                    if single {
                        println!("{}", rdna::disasm(i));
                    }
                }
            }
            Err(e) => {
                failed += 1;
                eprintln!("{f}: {}", e.0);
            }
        }
    }
    if !single {
        println!("{} file(s), {failed} failed to decode", files.len());
        println!("unknown opcodes:");
        for (k, v) in &unknown {
            println!("  {v:6}  {k}");
        }
        println!("instruction mix ({} distinct):", mix.len());
        let mut v: Vec<_> = mix.into_iter().collect();
        v.sort_by(|a, b| b.1.cmp(&a.1));
        for (k, n) in v {
            println!("  {n:6}  {k}");
        }
    }
}

/// Shader use recorded in a frame capture.
#[derive(Clone, Debug)]
struct Use {
    stage: rdna::translate::Stage,
    rsrc2: u32,
    users: u16,
    threads: [u32; 3],
    col_format: u32,
    ps_input_addr: u32,
}

fn hex(s: &str) -> u32 {
    u32::from_str_radix(s.trim_start_matches("0x"), 16).unwrap_or(0)
}

/// Parses `capture-<flip>.txt`: shader address -> first use.
fn parse_capture(text: &str) -> BTreeMap<u64, Use> {
    use rdna::translate::Stage;
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

/// Translates every shader of a capture; returns the process exit code.
fn translate_capture(capture: &str, out_dir: &str) -> i32 {
    use rdna::analysis::ResKind;
    use rdna::translate::{ImageDim, MrtSpec, NumClass, ResSpec, Spec, Stage};
    let Ok(text) = std::fs::read_to_string(capture) else {
        eprintln!("cannot read {capture}");
        return 2;
    };
    let shader_dir = std::path::Path::new(capture).parent().unwrap_or(std::path::Path::new(".")).join("shaders");
    let _ = std::fs::create_dir_all(out_dir);
    let uses = parse_capture(&text);
    let (mut ok, mut failed) = (0, 0);
    let mut warn_kinds: BTreeMap<String, u32> = BTreeMap::new();
    for (addr, u) in &uses {
        let path = shader_dir.join(format!("{addr:012x}.bin"));
        let Ok(bytes) = std::fs::read(&path) else {
            println!("{addr:#x}: binary missing");
            failed += 1;
            continue;
        };
        let code: Vec<u32> = bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        let insts = match rdna::decode_program(&code) {
            Ok(i) => i,
            Err(e) => {
                println!("{addr:#x}: decode error {}", e.0);
                failed += 1;
                continue;
            }
        };
        let user_base = if u.stage == Stage::Vertex { 8 } else { 0 };
        let users = if u.stage == Stage::Vertex && u.users == 0 { 24 } else { u.users };
        let plan = rdna::analysis::analyze(&insts, users, user_base, *addr);
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
                        .find(|i| i.enc == rdna::Enc::Mimg && plan.inst_res.get(&i.pc).is_some_and(|x| x.0 == r))
                        .map(|i| match i.mimg_dim() {
                            0 => ImageDim::D1,
                            1 => ImageDim::D2,
                            2 => ImageDim::D3,
                            3 => ImageDim::Cube,
                            4 => ImageDim::D1Array,
                            5 => ImageDim::D2Array,
                            6 => ImageDim::D2Msaa,
                            _ => ImageDim::D2MsaaArray,
                        })
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
            stage: u.stage,
            user_sgprs: users,
            user_base,
            code_base: *addr,
            wave64,
            ps_input_addr: u.ps_input_addr,
            ps_inputs: Vec::new(),
            mrt,
            cs_local_size: u.threads,
            cs_rsrc2: u.rsrc2,
            resources,
            binding_base: 0,
        };
        match rdna::translate::translate(&insts, &plan, &spec) {
            Ok(out) => {
                ok += 1;
                let bytes: Vec<u8> = out.spirv.iter().flat_map(|w| w.to_le_bytes()).collect();
                let _ = std::fs::write(std::path::Path::new(out_dir).join(format!("{addr:012x}.spv")), bytes);
                for w in &out.warnings {
                    // Group warnings by message (without the pc prefix).
                    let k = match w.split_once(": ") {
                        Some((a, b)) if a.starts_with("0x") => b.to_string(),
                        _ => w.clone(),
                    };
                    *warn_kinds.entry(k).or_default() += 1;
                }
                println!("{addr:#x} {:?}: {} words, {} binding(s), {} warning(s)", u.stage, out.spirv.len(), out.bindings.len(), out.warnings.len());
            }
            Err(e) => {
                failed += 1;
                println!("{addr:#x} {:?}: FAILED {e}", u.stage);
            }
        }
    }
    println!("{ok} translated, {failed} failed");
    if !warn_kinds.is_empty() {
        println!("warnings:");
        for (k, n) in &warn_kinds {
            println!("  {n:4}  {k}");
        }
    }
    if failed > 0 { 1 } else { 0 }
}
