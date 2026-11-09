//! sce-ghidra: prepares a PS4/PS5 module for a disassembler that reads plain
//! ELF files (Ghidra): writes a copy with the dynamic relocations applied
//! at image base 0 (RELATIVE: the addend; to an imported symbol: a stand-in
//! address `EXT_BASE + 16 * i` of its own; to a defined one: its value plus
//! the addend) and the ELF type made ET_DYN, and a symbol list naming the
//! stand-ins, the import slots and the module's own exported symbols
//! (`<hex address> <kind> <name>`, kinds: ext, slot, export). Proprietary
//! inputs and what this writes stay out of Git (local-data/).
//!
//! usage: sce-ghidra <module> <out.elf> <out-symbols.txt> [--names <file>]...

use sce_elf::{Module, nid};
use std::collections::HashMap;
use std::io::Write;

/// Where imported symbols' stand-in addresses start (past any module image).
const EXT_BASE: u64 = 0x4000_0000;

fn main() {
    let mut args = std::env::args().skip(1);
    let mut paths = Vec::new();
    let mut name_files = Vec::new();
    while let Some(a) = args.next() {
        match a.as_str() {
            "--names" => name_files.push(args.next().expect("--names needs a file")),
            _ => paths.push(a),
        }
    }
    let [module, out_elf, out_syms] = paths.as_slice() else {
        eprintln!("usage: sce-ghidra <module> <out.elf> <out-symbols.txt> [--names <file>]...");
        std::process::exit(2);
    };
    let mut names: HashMap<String, String> = HashMap::new();
    for f in &name_files {
        let text = std::fs::read_to_string(f).expect("read names file");
        for line in text.lines() {
            let n = line.trim();
            if !n.is_empty() && !n.starts_with('#') {
                names.insert(nid::nid_for_name(n), n.to_string());
            }
        }
    }
    let bytes = std::fs::read(module).unwrap_or_else(|e| panic!("{module}: {e}"));
    let m = Module::parse(bytes.clone()).unwrap_or_else(|e| panic!("{module}: {e}"));
    let d = m.dynamic().unwrap_or_else(|e| panic!("{module}: dynamic: {e}"));
    // The file as it is (a plain ELF: decrypted executables are), patched.
    let mut out = m.file_bytes().to_vec();
    // e_type at offset 16: ET_DYN.
    out[16..18].copy_from_slice(&3u16.to_le_bytes());
    // Virtual address -> file offset through the loadable segments.
    let loads: Vec<(u64, u64, u64)> = m.phdrs.iter().filter(|p| p.p_type == 1 && p.p_filesz > 0).map(|p| (p.p_vaddr, p.p_filesz, p.p_offset)).collect();
    let file_off = |va: u64| loads.iter().find(|&&(v, n, _)| va >= v && va + 8 <= v + n).map(|&(v, _, o)| o + (va - v));
    let sym_name = |i: usize| -> String {
        let s = &d.symbols[i];
        let nid_part = s.name.split('#').next().unwrap_or(&s.name).to_string();
        names.get(&nid_part).cloned().unwrap_or_else(|| format!("nid_{}", nid_part.replace(['+', '-'], "_")))
    };
    let mut syms = std::fs::File::create(out_syms).expect("create symbols file");
    let (mut applied, mut outside) = (0u64, 0u64);
    let mut ext_named = vec![false; d.symbols.len()];
    for r in d.relas.iter().chain(d.plt_relas.iter()) {
        let value = match r.r_type() {
            // R_X86_64_RELATIVE
            8 => r.addend as u64,
            // R_X86_64_64, GLOB_DAT, JUMP_SLOT
            1 | 6 | 7 => {
                let i = r.sym();
                let Some(s) = d.symbols.get(i) else { continue };
                if s.is_defined() {
                    s.value.wrapping_add(if r.r_type() == 1 { r.addend as u64 } else { 0 })
                } else {
                    let ext = EXT_BASE + 16 * i as u64;
                    if !ext_named[i] {
                        ext_named[i] = true;
                        writeln!(syms, "{ext:#x} ext {}", sym_name(i)).unwrap();
                    }
                    if r.r_type() != 1 {
                        writeln!(syms, "{:#x} slot {}", r.offset, sym_name(i)).unwrap();
                    }
                    ext.wrapping_add(if r.r_type() == 1 { r.addend as u64 } else { 0 })
                }
            }
            _ => continue,
        };
        match file_off(r.offset) {
            Some(o) => {
                out[o as usize..o as usize + 8].copy_from_slice(&value.to_le_bytes());
                applied += 1;
            }
            None => outside += 1,
        }
    }
    for s in d.symbols.iter().filter(|s| s.is_defined() && s.value != 0) {
        let nid_part = s.name.split('#').next().unwrap_or(&s.name);
        let n = names.get(nid_part).cloned().unwrap_or_else(|| format!("nid_{}", nid_part.replace(['+', '-'], "_")));
        writeln!(syms, "{:#x} export {n}", s.value).unwrap();
    }
    std::fs::write(out_elf, &out).expect("write output ELF");
    eprintln!("{module}: {applied} relocations applied ({outside} outside the file's segments), {} symbols; stand-ins at {EXT_BASE:#x}", d.symbols.len());
}
