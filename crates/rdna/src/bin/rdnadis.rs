//! rdnadis <file.bin>...   disassembles RDNA (gfx10) shader binaries; with
//! several files, prints only a summary (unknown opcodes, instruction mix).

use std::collections::BTreeMap;

fn main() {
    let files: Vec<String> = std::env::args().skip(1).collect();
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
