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
    // --spvdis <file.spv>: SPIR-V disassembly (rspirv).
    if files.first().is_some_and(|f| f == "--spvdis") && files.len() == 2 {
        use rspirv::binary::Disassemble;
        let bytes = std::fs::read(&files[1]).expect("read spv");
        let words: Vec<u32> = bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        let mut loader = rspirv::dr::Loader::new();
        rspirv::binary::parse_words(&words, &mut loader).expect("parse spv");
        println!("{}", loader.module().disassemble());
        return;
    }
    // --plan <file> [user sgprs] [user base]: resources and scalar loads with provenance
    // (--plan-gpu: with data-only loads read on the GPU, see analysis::analyze_opts).
    if files.first().is_some_and(|f| f == "--plan" || f == "--plan-gpu") && files.len() >= 2 {
        let gpu = files[0] == "--plan-gpu";
        let bytes = std::fs::read(&files[1]).expect("read shader");
        let code: Vec<u32> = bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        let insts = rdna::decode_program(&code).expect("decode");
        let users: u16 = files.get(2).and_then(|x| x.parse().ok()).unwrap_or(32);
        let base: u16 = files.get(3).and_then(|x| x.parse().ok()).unwrap_or(0);
        let plan = rdna::analysis::analyze_opts(&insts, users, base, gpu);
        for (i, r) in plan.resources.iter().enumerate() {
            println!("resource {i}: {:?} read {} written {}", r.kind, r.read, r.written);
            for (k, d) in r.dwords.iter().enumerate() {
                println!("    dword {k}: {d:?}");
            }
        }
        for (i, l) in plan.loads.iter().enumerate() {
            println!("load {i}: {:?} + {:#x} x{}", l.ptr, l.offset, l.dwords);
        }
        let mut uses: Vec<(u32, usize, Option<usize>)> = plan.inst_res.iter().map(|(&pc, &(r, s))| (pc, r, s)).collect();
        uses.sort();
        for (pc, r, s) in uses {
            println!("use {pc:#x}: resource {r}{}", s.map(|x| format!(", sampler {x}")).unwrap_or_default());
        }
        for e in &plan.errors {
            println!("error: {e}");
        }
        return;
    }
    // --masks <file.bin>...: per program, whether it uses LDS and whether a
    // compare mask is read as a value (wave64 / wave32), i.e. whether a
    // single-wave compute dispatch of it runs in wave-mask mode.
    if files.first().is_some_and(|f| f == "--masks") {
        for f in &files[1..] {
            let bytes = std::fs::read(f).expect("read shader");
            let code: Vec<u32> = bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            let Ok(insts) = rdna::decode_program(&code) else {
                println!("{f}: decode failed");
                continue;
            };
            let lds = insts.iter().any(|i| i.enc == rdna::Enc::Ds && !i.ds_gds());
            let mut pcs: Vec<u32> = rdna::wavemask::mask_value_writes(&insts, true).into_iter().collect();
            pcs.sort();
            let pcs: Vec<String> = pcs.iter().map(|p| format!("{p:#x}")).collect();
            println!("{f}: lds {lds} mask values {} / {} [{}]", rdna::wavemask::reads_mask_values(&insts, true), rdna::wavemask::reads_mask_values(&insts, false), pcs.join(" "));
            // Lane reads (v_readlane, v_readfirstlane) in blocks whose
            // invocations may diverge: outside lockstep a wave64 reads only its
            // 32-lane subgroup there.
            let u = rdna::uniform::analyze(&insts);
            let div: Vec<String> = insts.iter().filter(|i| matches!(i.base_name(), Some("v_readlane_b32" | "v_readfirstlane_b32")) && !u.is_uniform(i.pc)).map(|i| format!("{:#x}", i.pc)).collect();
            println!("    divergent lane reads {} [{}]", div.len(), div.join(" "));
            let xl: Vec<String> = insts.iter().filter(|i| rdna::cross_lane(i) && !u.is_uniform(i.pc)).map(|i| format!("{:#x}", i.pc)).collect();
            println!("    divergent cross-lane operations {} [{}]", xl.len(), xl.iter().take(8).cloned().collect::<Vec<_>>().join(" "));
            let kept: Vec<String> = rdna::translate::divergent_lane_ops(&insts).iter().map(|p| format!("{p:#x}")).collect();
            println!("    of them keeping a single wave in lockstep {} [{}]", kept.len(), kept.iter().take(8).cloned().collect::<Vec<_>>().join(" "));
            // Where each such mask is read whole (the first read in address order).
            for (w, r) in rdna::wavemask::mask_value_reads(&insts, true) {
                let show = |pc: u32| insts.iter().find(|i| i.pc == pc).map(|i| format!("{pc:#x} {}", rdna::disasm(i))).unwrap_or_default();
                println!("    {} -> {}", show(w), r.map(show).unwrap_or_else(|| "(not found in address order)".into()));
            }
        }
        return;
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
            let plan = rdna::analysis::analyze(&insts, 32, 0);
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

/// Translates every shader of a capture; returns the process exit code.
fn translate_capture(capture: &str, out_dir: &str) -> i32 {
    let items = match rdna::capture::translate_all(capture) {
        Ok(i) => i,
        Err(e) => {
            eprintln!("{e}");
            return 2;
        }
    };
    let _ = std::fs::create_dir_all(out_dir);
    let (mut ok, mut failed) = (0, 0);
    let mut warn_kinds: BTreeMap<String, u32> = BTreeMap::new();
    for it in &items {
        match &it.out {
            Ok(out) => {
                ok += 1;
                let bytes: Vec<u8> = out.spirv.iter().flat_map(|w| w.to_le_bytes()).collect();
                let _ = std::fs::write(std::path::Path::new(out_dir).join(format!("{:012x}.spv", it.addr)), bytes);
                if it.stage == rdna::translate::Stage::Vertex {
                    // The rect-list expansion stage for this vertex shader's outputs.
                    let gs: Vec<u8> = rdna::translate::gs::rect_list_gs(out.param_mask).iter().flat_map(|w| w.to_le_bytes()).collect();
                    let _ = std::fs::write(std::path::Path::new(out_dir).join(format!("{:012x}-rect-gs.spv", it.addr)), gs);
                }
                for w in &out.warnings {
                    // Group warnings by message (without the pc prefix).
                    let k = match w.split_once(": ") {
                        Some((a, b)) if a.starts_with("0x") => b.to_string(),
                        _ => w.clone(),
                    };
                    *warn_kinds.entry(k).or_default() += 1;
                }
                let model = if out.sg_waves { ", wave-mask mode per subgroup" } else if out.wave_masks { ", wave-mask mode" } else if out.lockstep { ", lockstep" } else { "" };
                let sg = if out.sg_appends > 0 { format!(", {} append(s) per subgroup", out.sg_appends) } else { String::new() };
                let ka = if out.known_active > 0 { format!(", {} instruction(s) known active", out.known_active) } else { String::new() };
                println!("{:#x} {:?}: {} words, {} binding(s), {} warning(s){model}{sg}{ka}", it.addr, it.stage, out.spirv.len(), out.bindings.len(), out.warnings.len());
                for w in &out.warnings {
                    println!("    {w}");
                }
            }
            Err(e) => {
                failed += 1;
                println!("{:#x} {:?}: FAILED {e}", it.addr, it.stage);
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
