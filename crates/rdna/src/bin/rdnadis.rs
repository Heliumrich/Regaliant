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
                for w in &out.warnings {
                    // Group warnings by message (without the pc prefix).
                    let k = match w.split_once(": ") {
                        Some((a, b)) if a.starts_with("0x") => b.to_string(),
                        _ => w.clone(),
                    };
                    *warn_kinds.entry(k).or_default() += 1;
                }
                println!("{:#x} {:?}: {} words, {} binding(s), {} warning(s)", it.addr, it.stage, out.spirv.len(), out.bindings.len(), out.warnings.len());
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
