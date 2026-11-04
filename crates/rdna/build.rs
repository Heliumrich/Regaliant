//! The translator's source hash (`SOURCE_HASH`): the runtime keeps
//! translations across runs under it, so any change to this crate's code
//! starts a new store.

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
        for b in name.bytes().chain(std::fs::read(f).unwrap_or_default()) {
            h = (h ^ b as u64).wrapping_mul(0x100_0000_01b3);
        }
    }
    println!("cargo:rustc-env=RDNA_SOURCE_HASH={h:016x}");
}
