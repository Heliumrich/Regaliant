//! `dsport --compile-bench <min instructions> <max programs>`: how long the
//! driver takes to compile the learned compute programs' pipelines (the
//! first-visit freezes of 2026-10-23: the title's particle programs took
//! 6-12.8 s each). Each program kept in the shader-spec store
//! (`gpu::precompile`) with at least that many instructions, the largest
//! first, is translated for its first learned spec and its pipeline created
//! with the driver's shader cache bypassed (a session nonce in the module,
//! as `DS_SPIRV_NONCE=1`), one at a time on this thread; per program its
//! code hash, instructions, SPIR-V words and the translation and compile
//! times, then the slowest. `DS_CB_HASHES=<hex,...>` takes only those
//! programs; `DS_CB_SPV=<dir>` writes each module as `<hash>.spv` (as
//! translated, without the nonce). Tunables (`DS_*`) change translations as
//! in a run.

use std::time::Instant;

/// `DS_CB_GUESS=1`: no compiles; every learned compute program's spec
/// guessed as from its shader object (`shaders::guess_compute_spec`, with
/// its learned RSRC2 and thread-group size) and compared with what its
/// dispatches used, overall and for the engine's own shaders
/// (coredata/enginesupport/shaders); typed buffers' formats by access.
fn guess_report(app: &std::path::Path) -> i32 {
    use rdna::translate::ResSpec;
    use std::collections::HashMap;
    // (Specs depend on the device's features.)
    if let Err(e) = super::vk::init() {
        crate::error!("guess report: Vulkan: {e}");
        return 2;
    }
    // The engine's programs by code hash: their directory.
    let mut engine: HashMap<u64, String> = HashMap::new();
    let root = app.join("coredata").join("enginesupport").join("shaders").join("_ps5");
    let mut stack: Vec<(std::path::PathBuf, Option<String>)> = vec![(root, None)];
    let mut files = 0;
    while let Some((d, top)) = stack.pop() {
        for e in std::fs::read_dir(&d).into_iter().flatten().flatten() {
            let p = e.path();
            if p.is_dir() {
                let name = top.clone().unwrap_or_else(|| e.file_name().to_string_lossy().into_owned());
                stack.push((p, Some(name)));
            } else if p.extension().is_some_and(|x| x == "csdr") {
                files += 1;
                if let Some(v) = std::fs::read(&p).ok().and_then(|b| super::particles::bundle_programs(&b)) {
                    let dir = top.clone().unwrap_or_default();
                    engine.extend(v.iter().filter(|x| x.stage == 0).map(|x| (x.hash, dir.clone())));
                }
            }
        }
    }
    // Per engine directory: [specs, equal, format-only, different], and
    // the same for programs binding no image and no typed buffer.
    let mut by_dir: HashMap<String, [u64; 4]> = HashMap::new();
    let mut plain = [0u64; 4];
    crate::info!("guess report: {} engine compute programs in {files} engine shader files", engine.len());
    // [specs, equal, format-only, other] overall and for engine programs.
    let mut n = [[0u64; 4]; 2];
    let mut other_why: HashMap<&'static str, u64> = HashMap::new();
    // Format-accessed buffers: access pattern -> (format, dst_sel) -> count.
    let mut by_access: HashMap<String, HashMap<(u8, u16), u64>> = HashMap::new();
    let mut engine_learned = 0;
    for (h, code, specs) in super::precompile::learned_compute() {
        let Ok(insts) = rdna::decode_program(&code) else { continue };
        let is_engine = engine.contains_key(&h);
        engine_learned += is_engine as u64;
        for spec in &specs {
            let plan = rdna::analysis::analyze_opts(&insts, spec.user_sgprs, spec.user_base, false);
            let (guess, _) = super::shaders::guess_compute_spec(0, &insts, &plan, h, spec.cs_rsrc2, spec.cs_local_size);
            let mut actual = spec.clone();
            actual.code_base = guess.code_base;
            let class = if guess == actual {
                1
            } else {
                let mut g2 = guess.clone();
                let mut fmt_only = g2.resources.len() == actual.resources.len();
                if fmt_only {
                    for (r, (a, b)) in g2.resources.iter_mut().zip(&actual.resources).enumerate() {
                        match (&*a, b) {
                            (ResSpec::Buffer { .. }, ResSpec::Buffer { format, dst_sel }) => {
                                if a != b {
                                    // The buffer's format accesses, as a key.
                                    let mut names: Vec<&str> = insts.iter().filter(|i| plan.inst_res.get(&i.pc).is_some_and(|x| x.0 == r)).filter_map(|i| i.name()).filter(|n| n.contains("_format_")).collect();
                                    names.sort();
                                    names.dedup();
                                    *by_access.entry(names.join("+")).or_default().entry((*format, *dst_sel)).or_default() += 1;
                                }
                                *a = b.clone();
                            }
                            (ResSpec::Image { dim: d1, class: c1, depth: z1 }, ResSpec::Image { dim: d2, class: c2, depth: z2 }) => {
                                if d1 != d2 {
                                    *other_why.entry("an image's dimension").or_default() += 1;
                                } else if c1 != c2 {
                                    *other_why.entry("an image's class").or_default() += 1;
                                } else if z1 != z2 {
                                    *other_why.entry("an image's depth view").or_default() += 1;
                                }
                            }
                            _ => {}
                        }
                    }
                    fmt_only = g2 == actual;
                }
                if fmt_only {
                    2
                } else {
                    if g2.resources.len() != actual.resources.len() {
                        *other_why.entry("resource count").or_default() += 1;
                    } else if g2.wave64 != actual.wave64 {
                        *other_why.entry("wave64").or_default() += 1;
                    }
                    3
                }
            };
            for (k, on) in [(0, true), (1, is_engine)] {
                if on {
                    n[k][0] += 1;
                    n[k][class] += 1;
                }
            }
            if let Some(dir) = engine.get(&h) {
                let e = by_dir.entry(dir.clone()).or_default();
                e[0] += 1;
                e[class] += 1;
                // No image and no typed (format) buffer access: the spec is
                // the code's own (formats and image setups come from the
                // title's descriptors at run time).
                let typed = insts.iter().any(|i| i.name().is_some_and(|n| n.contains("_format_")));
                let images = actual.resources.iter().any(|r| matches!(r, ResSpec::Image { .. }));
                if !typed && !images {
                    plain[0] += 1;
                    plain[class] += 1;
                }
            }
        }
    }
    for (k, what) in [(0, "all learned"), (1, "engine")] {
        crate::info!("guess report ({what}): {} specs: guess equal {}, equal but for typed buffers' formats {}, different {}", n[k][0], n[k][1], n[k][2], n[k][3]);
    }
    crate::info!("guess report: {engine_learned} of the learned programs are the engine's; other differences: {other_why:?}");
    crate::info!("guess report (engine programs binding no image and no typed buffer): {} specs: equal {}, format-only {}, different {}", plain[0], plain[1], plain[2], plain[3]);
    let mut dirs: Vec<(&String, &[u64; 4])> = by_dir.iter().collect();
    dirs.sort_by(|a, b| b.1[0].cmp(&a.1[0]).then(a.0.cmp(b.0)));
    let parts: Vec<String> = dirs.iter().map(|(d, x)| format!("{d} {}/{}/{}/{}", x[0], x[1], x[2], x[3])).collect();
    crate::info!("guess report by engine directory (specs/equal/format-only/different): {}", parts.join(", "));
    let mut keys: Vec<(&String, &HashMap<(u8, u16), u64>)> = by_access.iter().collect();
    keys.sort_by_key(|(_, m)| std::cmp::Reverse(m.values().sum::<u64>()));
    let (mut all, mut top) = (0u64, 0u64);
    for (k, m) in keys {
        let mut v: Vec<(&(u8, u16), &u64)> = m.iter().collect();
        v.sort_by_key(|(_, c)| std::cmp::Reverse(**c));
        let total: u64 = m.values().sum();
        all += total;
        top += *v[0].1;
        let list: Vec<String> = v.iter().take(4).map(|((f, d), c)| format!("{f}/{d:#x} {c}")).collect();
        crate::info!("    {k}: {total} ({})", list.join(", "));
    }
    crate::info!("guess report: the most frequent format of each access pattern would be right for {top} of {all} typed buffers");
    0
}

pub fn run(args: &[String], app: &std::path::Path) -> i32 {
    if std::env::var("DS_CB_GUESS").ok().as_deref() == Some("1") {
        return guess_report(app);
    }
    let min_insts: usize = args.first().and_then(|s| s.parse().ok()).unwrap_or(2000);
    let max: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(40);
    let mut only: Vec<u64> = std::env::var("DS_CB_HASHES").map(|v| v.split(',').filter_map(|h| u64::from_str_radix(h.trim().trim_start_matches("0x"), 16).ok()).collect()).unwrap_or_default();
    // DS_CB_BINS=<file,...>: programs saved by a run (`shaders/<address>.bin`),
    // found in the store by their code hash.
    for f in std::env::var("DS_CB_BINS").map(|v| v.split(',').map(str::to_string).collect::<Vec<_>>()).unwrap_or_default() {
        let Ok(bytes) = std::fs::read(&f) else {
            crate::warn!("compile bench: cannot read {f}");
            continue;
        };
        let words: Vec<u32> = bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        let Some(last) = rdna::decode_program(&words).ok().and_then(|i| i.last().map(|l| (l.pc / 4 + l.len as u32) as usize)) else { continue };
        let h = super::shaders::hash_words(&words[..last.min(words.len())]);
        crate::info!("compile bench: {f}: code hash {h:#018x}");
        only.push(h);
    }
    let spv_dir = std::env::var("DS_CB_SPV").ok().filter(|d| !d.is_empty()).map(std::path::PathBuf::from);
    let spv_in = std::env::var("DS_CB_SPV_IN").ok().filter(|d| !d.is_empty()).map(std::path::PathBuf::from);
    if let Some(d) = &spv_dir {
        let _ = std::fs::create_dir_all(d);
    }
    super::exec::FORCE_NONCE.store(true, std::sync::atomic::Ordering::Relaxed);
    if let Err(e) = super::vk::init() {
        crate::error!("compile bench: Vulkan: {e}");
        return 2;
    }
    let mut progs: Vec<(u64, std::sync::Arc<Vec<u32>>, Vec<rdna::translate::Spec>, usize)> = super::precompile::learned_compute()
        .into_iter()
        .filter(|(h, _, specs)| !specs.is_empty() && (only.is_empty() || only.contains(h)))
        .filter_map(|(h, code, specs)| {
            let n = rdna::decode_program(&code).ok()?.len();
            (n >= min_insts || !only.is_empty()).then_some((h, code, specs, n))
        })
        .collect();
    progs.sort_by(|a, b| b.3.cmp(&a.3));
    progs.truncate(max);
    crate::info!("compile bench: {} programs of {min_insts} instructions or more (the largest first)", progs.len());
    let mut rows: Vec<(u64, usize, usize, f64, f64)> = Vec::new();
    let t_all = Instant::now();
    for (h, code, specs, n) in &progs {
        let t0 = Instant::now();
        let tr = match super::shaders::translate_code_for_spec(code, *h, specs[0].clone()) {
            Ok(t) => t,
            Err(e) => {
                crate::warn!("compile bench: {h:#018x}: translation failed: {e}");
                continue;
            }
        };
        let translate_ms = t0.elapsed().as_secs_f64() * 1e3;
        if let Some(d) = &spv_dir {
            let bytes: Vec<u8> = tr.out.spirv.iter().flat_map(|w| w.to_le_bytes()).collect();
            let _ = std::fs::write(d.join(format!("{h:016x}.spv")), bytes);
        }
        // DS_CB_SPV_IN=<dir>: `<hash>.spv` there replaces the translation's
        // module (the same interface, e.g. `spvval --opt` output).
        let replaced = spv_in.as_ref().and_then(|d| std::fs::read(d.join(format!("{h:016x}.spv"))).ok()).map(|b| {
            let mut t = (*tr).clone();
            t.out.spirv = b.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            t
        });
        let tr = replaced.as_ref().unwrap_or(&tr);
        match super::exec::bench_compute_pipeline(tr) {
            Ok(d) => {
                let ms = d.as_secs_f64() * 1e3;
                crate::info!("compile bench: {h:#018x}: {n} instructions, {} SPIR-V words{}, translated in {translate_ms:.0} ms, compiled in {ms:.0} ms", tr.out.spirv.len(), if replaced.is_some() { " (replaced)" } else { "" });
                rows.push((*h, *n, tr.out.spirv.len(), translate_ms, ms));
            }
            Err(e) => crate::warn!("compile bench: {h:#018x}: pipeline creation failed: {e}"),
        }
    }
    rows.sort_by(|a, b| b.4.total_cmp(&a.4));
    let total: f64 = rows.iter().map(|r| r.4).sum();
    let words: usize = rows.iter().map(|r| r.2).sum();
    crate::info!("compile bench: {} pipelines in {:.1} s ({:.1} s of compiles, {:.1} us per SPIR-V word); slowest:", rows.len(), t_all.elapsed().as_secs_f64(), total / 1e3, total * 1e3 / words.max(1) as f64);
    for (h, n, w, t, c) in rows.iter().take(15) {
        crate::info!("    {h:#018x}: {c:.0} ms ({n} instructions, {w} words, {:.1} us per word; translated in {t:.0} ms)", c * 1e3 / *w as f64);
    }
    0
}
