//! The translator's source hash (`SOURCE_HASH`): the runtime keeps
//! translations across runs under it, so a change to the code that makes them
//! starts a new store. Left out: `src/bin/` (the disassembler) and, of
//! `src/kernels.rs` (the runtime's own kernels), all but its constants (the
//! written-bit map's layout, which translations use): until 2026-11-15 a kernel's
//! edit made every stored translation cold (translated again at its first use). The
//! pipeline store does not depend on this hash (a binary is taken by its key and the
//! SPIR-V it was made from).

use std::path::{Path, PathBuf};

fn collect(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            collect(&p, out);
        } else {
            out.push(p);
        }
    }
}

fn main() {
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=build.rs");
    let mut files = Vec::new();
    collect(Path::new("src"), &mut files);
    files.sort();
    // FNV-1a over each file's path and bytes.
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for f in &files {
        let name = f.to_string_lossy().replace('\\', "/");
        if name.starts_with("src/bin/") {
            continue;
        }
        let bytes = std::fs::read(f).unwrap_or_default();
        let bytes = if name == "src/kernels.rs" {
            String::from_utf8_lossy(&bytes)
                .lines()
                .filter(|l| {
                    let t = l.trim_start();
                    t.starts_with("pub const ") || t.starts_with("const ")
                })
                .collect::<Vec<_>>()
                .join("\n")
                .into_bytes()
        } else {
            bytes
        };
        for b in name.bytes().chain(bytes) {
            h = (h ^ b as u64).wrapping_mul(0x100_0000_01b3);
        }
    }
    println!("cargo:rustc-env=RDNA_SOURCE_HASH={h:016x}");
}
