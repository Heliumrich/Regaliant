//! sce-inspect: dump program headers and dynamic-link metadata of PS4 modules.
//!
//! usage: sce-inspect [--imports] [--exports] [--names <file>] <module>...

use sce_elf::{Module, nid};
use std::collections::{BTreeMap, HashMap};

fn main() {
    let mut show_imports = false;
    let mut show_exports = false;
    let mut name_files = Vec::new();
    let mut paths = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--imports" => show_imports = true,
            "--exports" => show_exports = true,
            "--names" => name_files.push(args.next().expect("--names needs a file")),
            _ => paths.push(a),
        }
    }
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
    for p in paths {
        let bytes = std::fs::read(&p).unwrap_or_else(|e| panic!("{p}: {e}"));
        let m = match Module::parse(bytes) {
            Ok(m) => m,
            Err(e) => {
                println!("{p}: ERROR {e}");
                continue;
            }
        };
        println!("== {p}");
        println!(
            "container={:?} type={:#06x} osabi={} entry={:#x} phnum={}",
            m.container, m.ehdr.e_type, m.ehdr.osabi, m.ehdr.e_entry, m.ehdr.e_phnum
        );
        for (i, ph) in m.phdrs.iter().enumerate() {
            println!(
                "  ph{i:<2} type={:#010x} flags={} off={:#x} vaddr={:#x} filesz={:#x} memsz={:#x} align={:#x} data={}",
                ph.p_type,
                ph.p_flags,
                ph.p_offset,
                ph.p_vaddr,
                ph.p_filesz,
                ph.p_memsz,
                ph.p_align,
                m.segment_data(i).is_some()
            );
        }
        let d = match m.dynamic() {
            Ok(d) => d,
            Err(e) => {
                println!("  dynamic: ERROR {e}");
                continue;
            }
        };
        if let Some(mi) = &d.module_info {
            println!("  module: {} id={} v{}.{}", mi.name, mi.id, mi.version_major, mi.version_minor);
        }
        println!(
            "  init={:?} fini={:?} init_array={:?} fini_array={:?} preinit={:?} flags={:#x}",
            d.init.map(|v| format!("{v:#x}")),
            d.fini.map(|v| format!("{v:#x}")),
            d.init_array.map(|(a, s)| format!("{a:#x}+{s:#x}")),
            d.fini_array.map(|(a, s)| format!("{a:#x}+{s:#x}")),
            d.preinit_array.map(|(a, s)| format!("{a:#x}+{s:#x}")),
            d.flags
        );
        println!("  needed: {:?}", d.needed);
        for nm in &d.needed_modules {
            println!("  needed-module id={} {} v{}.{}", nm.id, nm.name, nm.version_major, nm.version_minor);
        }
        for l in &d.export_libs {
            println!("  export-lib id={} {} v{} attr={:#x}", l.id, l.name, l.version, l.attr);
        }
        // Import statistics per library.
        let mut per_lib: BTreeMap<String, Vec<String>> = BTreeMap::new();
        let mut exports: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for s in &d.symbols {
            let Some((n, lib, _module)) = nid::split_symbol(&s.name) else { continue };
            let label = match names.get(n) {
                Some(name) => format!("{n} {name}"),
                None => n.to_string(),
            };
            if s.is_defined() {
                let ln = d.export_lib_by_id(lib).map(|l| l.name.clone()).unwrap_or(format!("?{lib}"));
                exports.entry(ln).or_default().push(label);
            } else {
                let ln = d.import_lib_by_id(lib).map(|l| l.name.clone()).unwrap_or(format!("?{lib}"));
                per_lib.entry(ln).or_default().push(label);
            }
        }
        for l in &d.import_libs {
            let n = per_lib.get(&l.name).map(|v| v.len()).unwrap_or(0);
            println!("  import-lib id={} {} v{} attr={:#x} symbols={n}", l.id, l.name, l.version, l.attr);
        }
        let mut rtypes: BTreeMap<u32, usize> = BTreeMap::new();
        for r in d.relas.iter().chain(d.plt_relas.iter()) {
            *rtypes.entry(r.r_type()).or_default() += 1;
        }
        println!(
            "  symbols={} relas={} plt={} types={:?}",
            d.symbols.len(),
            d.relas.len(),
            d.plt_relas.len(),
            rtypes
        );
        if show_imports {
            for (lib, syms) in &per_lib {
                let known = syms.iter().filter(|s| s.contains(' ')).count();
                println!("  [{lib}] {} imports ({known} named)", syms.len());
                for s in syms {
                    println!("    {s}");
                }
            }
        }
        if show_exports {
            for (lib, syms) in &exports {
                println!("  <{lib}> {} exports", syms.len());
                for s in syms {
                    println!("    {s}");
                }
            }
        }
    }
}
