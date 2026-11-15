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
/// A `Debug` string's top-level fields (`Name { a: .., b: .. }`: each `a: ..`), nested braces
/// and brackets kept whole.
fn top_fields(d: &str) -> Vec<String> {
    let Some(start) = d.find('{') else { return vec![d.to_string()] };
    let body = &d[start + 1..d.len().saturating_sub(1)];
    let (mut out, mut cur, mut depth) = (Vec::new(), String::new(), 0i32);
    for ch in body.chars() {
        match ch {
            '{' | '[' | '(' => depth += 1,
            '}' | ']' | ')' => depth -= 1,
            ',' if depth == 0 => {
                out.push(cur.trim().to_string());
                cur.clear();
                continue;
            }
            _ => {}
        }
        cur.push(ch);
    }
    if !cur.trim().is_empty() {
        out.push(cur.trim().to_string());
    }
    out
}

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
    // The particle bundles' programs (compiled at their first dispatch: a
    // guess at level load would have them ready).
    let mut particle: std::collections::HashSet<u64> = std::collections::HashSet::new();
    let pdir = app.join("particles").join("!cookedparticleshaders").join("version01").join("_ps5");
    for e in std::fs::read_dir(&pdir).into_iter().flatten().flatten() {
        if let Some(v) = std::fs::read(e.path()).ok().and_then(|b| super::particles::bundle_programs(&b)) {
            particle.extend(v.iter().map(|x| x.hash));
        }
    }
    // Per engine directory: [specs, equal, format-only, different], and
    // the same for programs binding no image and no typed buffer.
    let mut by_dir: HashMap<String, [u64; 4]> = HashMap::new();
    let mut plain = [0u64; 4];
    crate::info!("guess report: {} engine compute programs in {files} engine shader files", engine.len());
    // [specs, equal, format-only, other] overall, for engine programs and
    // for particle programs.
    let mut n = [[0u64; 4]; 3];
    let mut other_why: HashMap<&'static str, u64> = HashMap::new();
    // Field by field (`Spec`'s top-level fields that differ, guess -> learned), with the
    // preparation's variant rule applied to the guess first (`precompile_all`: LOD statistics
    // off where a dispatch prepares without them).
    let mut fields: HashMap<String, u64> = HashMap::new();
    let mut examples: Vec<String> = Vec::new();
    // The differences in detail: (guess -> learned) image dimensions and
    // classes, and the resource kinds when their counts differ.
    let mut pairs: HashMap<String, u64> = HashMap::new();
    // The same by the programs' origin (particle bundles, engine, other).
    let mut pairs_by: HashMap<String, u64> = HashMap::new();
    // Format-accessed buffers: access pattern -> (format, dst_sel) -> count.
    let mut by_access: HashMap<String, HashMap<(u8, u16), u64>> = HashMap::new();
    let mut engine_learned = 0;
    // Engine specs not equal to the guess: [equal to one of its image-dimension variants, equal
    // to the guess with the LOD rule applied].
    let mut var_equal = [0u64; 5];
    for (h, code, specs) in super::precompile::learned_compute() {
        let Ok(insts) = rdna::decode_program(&code) else { continue };
        let is_engine = engine.contains_key(&h);
        engine_learned += is_engine as u64;
        for spec in &specs {
            let plan = rdna::analysis::analyze_opts(&insts, spec.user_sgprs, spec.user_base, false);
            let guess = super::shaders::guess_compute_spec(0, &insts, &plan, h, spec.cs_rsrc2, spec.cs_local_size);
            let mut actual = spec.clone();
            actual.code_base = guess.code_base;
            // (Keyed as a preparation keys it now.)
            super::shaders::key_learned(h, &insts, &plan, &mut actual.resources);
            {
                let mut g = guess.clone();
                if g.lod_stats && !g.batched && super::shaders::cs_lod_rotation() && !actual.lod_stats {
                    g.lod_stats = false;
                    g.lod_lean = false;
                }
                if g != actual {
                    let (a, b) = (format!("{g:?}"), format!("{actual:?}"));
                    for (fa, fb) in top_fields(&a).iter().zip(top_fields(&b).iter()) {
                        if fa != fb {
                            let name = fa.split(':').next().unwrap_or("?").trim().to_string();
                            *fields.entry(name.clone()).or_default() += 1;
                            if examples.len() < 12 {
                                examples.push(format!("{h:#018x} {name}: {} -> {}", fa.chars().take(160).collect::<String>(), fb.chars().take(160).collect::<String>()));
                            }
                        }
                    }
                }
            }
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
                                let tag = if particle.contains(&h) { "particle" } else if engine.contains_key(&h) { "engine" } else { "other" };
                                if d1 != d2 {
                                    *other_why.entry("an image's dimension").or_default() += 1;
                                    *pairs.entry(format!("dim {d1:?} -> {d2:?}")).or_default() += 1;
                                    *pairs_by.entry(format!("{tag}: dim {d1:?} -> {d2:?}")).or_default() += 1;
                                } else if c1 != c2 {
                                    *other_why.entry("an image's class").or_default() += 1;
                                    *pairs.entry(format!("class {c1:?} -> {c2:?}")).or_default() += 1;
                                    *pairs_by.entry(format!("{tag}: class {c1:?} -> {c2:?}")).or_default() += 1;
                                    // (The image instructions on that resource.)
                                    let mut ops: Vec<&str> = insts.iter().filter(|i| i.enc == rdna::Enc::Mimg && plan.inst_res.get(&i.pc).is_some_and(|x| x.0 == r)).filter_map(|i| i.name()).collect();
                                    ops.sort();
                                    ops.dedup();
                                    *pairs_by.entry(format!("{tag}: class {c1:?} -> {c2:?} by [{}]", ops.join(" "))).or_default() += 1;
                                    // And for the ones guessed right, the same key with their class.
                                }
                                if c1 == c2 && d1 == d2 && matches!(c2, rdna::translate::NumClass::Float) && tag == "engine" {
                                    let mut ops: Vec<&str> = insts.iter().filter(|i| i.enc == rdna::Enc::Mimg && plan.inst_res.get(&i.pc).is_some_and(|x| x.0 == r)).filter_map(|i| i.name()).collect();
                                    ops.sort();
                                    ops.dedup();
                                    *pairs_by.entry(format!("{tag}: Float right by [{}]", ops.join(" "))).or_default() += 1;
                                }
                                if d1 != d2 || c1 != c2 {
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
                        // (Another learned spec of the program with the
                        // plan's count: this one an older build's.)
                        let stale = specs.iter().any(|o| o.resources.len() == g2.resources.len());
                        *other_why.entry(if stale { "resource count (another spec of the program has the plan's)" } else { "resource count" }).or_default() += 1;
                        let kinds = |v: &[ResSpec]| v.iter().map(|r| match r { ResSpec::Buffer { .. } => 'b', ResSpec::Image { .. } => 'i', ResSpec::Sampler => 's', ResSpec::Global => 'g' }).collect::<String>();
                        *pairs.entry(format!("resources {} -> {}", kinds(&g2.resources), kinds(&actual.resources))).or_default() += 1;
                    } else if g2.wave64 != actual.wave64 {
                        *other_why.entry("wave64").or_default() += 1;
                    }
                    3
                }
            };
            for (k, on) in [(0, true), (1, is_engine), (2, particle.contains(&h))] {
                if on {
                    n[k][0] += 1;
                    n[k][class] += 1;
                }
            }
            if class != 1 && is_engine {
                let mut g = guess.clone();
                if g.lod_stats && !g.batched && super::shaders::cs_lod_rotation() && !actual.lod_stats {
                    g.lod_stats = false;
                    g.lod_lean = false;
                }
                // (Typed buffers' formats are read at run time: those differences match.)
                let norm = |s: &rdna::translate::Spec| {
                    let mut s = s.clone();
                    if s.resources.len() == actual.resources.len() {
                        for (a, b) in s.resources.iter_mut().zip(&actual.resources) {
                            if matches!((&*a, b), (ResSpec::Buffer { .. }, ResSpec::Buffer { .. })) {
                                *a = b.clone();
                            }
                        }
                    }
                    s
                };
                if norm(&g) == actual {
                    var_equal[1] += 1;
                } else if super::shaders::guess_dim_variants(&g, if super::shaders::dim_by_instructions() { let (a, b) = super::shaders::d2_resources(&insts, &plan); a | b } else { 0 }).iter().any(|v| norm(v) == actual) {
                    var_equal[0] += 1;
                } else if g.resources.len() == actual.resources.len() {
                    // Upper bounds: the guess with every image's dimension as learned, then also
                    // its depth view, then also its class.
                    let mut u = norm(&g);
                    let set = |u: &mut rdna::translate::Spec, k: u8| {
                        for (a, b) in u.resources.iter_mut().zip(&actual.resources) {
                            if let (ResSpec::Image { dim, depth, class }, ResSpec::Image { dim: d2, depth: z2, class: c2 }) = (a, b) {
                                match k {
                                    0 => *dim = *d2,
                                    1 => *depth = *z2,
                                    _ => *class = *c2,
                                }
                            }
                        }
                    };
                    for k in 0..3u8 {
                        set(&mut u, k);
                        if u == actual {
                            var_equal[2 + k as usize] += 1;
                            break;
                        }
                    }
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
    for (k, what) in [(0, "all learned"), (1, "engine"), (2, "particle bundles")] {
        crate::info!("guess report ({what}): {} specs: guess equal {}, equal but for typed buffers' formats {}, different {}", n[k][0], n[k][1], n[k][2], n[k][3]);
    }
    crate::info!("guess report: {engine_learned} of the learned programs are the engine's; other differences: {other_why:?}");
    {
        let progs = bundle_compute(app);
        let engine_variants: usize = progs.iter().map(|p| p.2.len() - 1).sum();
        crate::info!("guess report: the preparation would make {} compute pipelines' specs ({} of them variants of the guess)", progs.iter().map(|p| p.2.len()).sum::<usize>(), engine_variants);
    }
    crate::info!("guess report (engine): of the specs not equal to the guess, {} equal one of its image-dimension variants (`shaders::guess_dim_variants`), {} the guess with the LOD rule and typed buffers' formats; of the rest, equal with the learned image dimensions {}, also their depth views {}, also their classes {}", var_equal[0], var_equal[1], var_equal[2], var_equal[3], var_equal[4]);
    let mut fv: Vec<_> = fields.into_iter().collect();
    fv.sort_by(|a, b| b.1.cmp(&a.1));
    crate::info!("guess report: fields that differ after the preparation's variant rule (count): {fv:?}");
    for e in &examples {
        crate::info!("guess report: example {e}");
    }
    let mut pv: Vec<(&String, &u64)> = pairs.iter().collect();
    pv.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
    crate::info!("guess report: differences in detail (guess -> learned): {}", pv.iter().take(24).map(|(k, c)| format!("{k} {c}")).collect::<Vec<_>>().join(", "));
    let mut pb: Vec<(&String, &u64)> = pairs_by.iter().collect();
    pb.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
    crate::info!("guess report: image differences by origin (guess -> learned): {}", pb.iter().take(30).map(|(k, c)| format!("{k} {c}")).collect::<Vec<_>>().join(", "));
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

/// RSRC2 and the thread-group size a compute shader object holds, as a
/// shader file stores it (its SH register pairs; pointers self-relative
/// until `sceAgcCreateShader` relocates them: `hle::libs::agc::shader`).
fn object_regs(o: &[u8]) -> Option<(u32, [u32; 3])> {
    let rd32 = |at: usize| o.get(at..at + 4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]));
    let rd64 = |at: usize| o.get(at..at + 8).map(|b| u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]));
    // (Magic "1234", version 0x18; SH registers at 0x20, their count at 0x5C.)
    if rd32(0)? != 0x3433_3231 || rd32(4)? != 0x18 {
        return None;
    }
    let rel = rd64(0x20)?;
    if rel == 0 {
        return None;
    }
    let regs = 0x20usize.checked_add(usize::try_from(rel).ok()?)?;
    let n = *o.get(0x5C)? as usize;
    let (mut rsrc2, mut t) = (None, [1u32; 3]);
    for k in 0..n {
        let e = regs + 8 * k;
        let (r, v) = (rd32(e)?, rd32(e + 4)?);
        match r {
            0x213 => rsrc2 = Some(v),
            0x207..=0x209 => t[(r - 0x207) as usize] = (v & 0xFFFF).max(1),
            _ => {}
        }
    }
    Some((rsrc2?, t))
}

/// The compute programs of the game's shader files (`DS_CB_BUNDLES`), each
/// with the spec guessed from its shader object, once per code hash.
fn bundle_compute(app: &std::path::Path) -> Vec<(u64, std::sync::Arc<Vec<u32>>, Vec<rdna::translate::Spec>)> {
    let sample: u64 = std::env::var("DS_CB_SAMPLE").ok().and_then(|v| v.parse().ok()).filter(|&n| n > 0).unwrap_or(1);
    bundle_compute_sampled(app, sample)
}

/// The programs a preparation's fingerprint translates: one in this many by code hash
/// (~16 of the game's 8,233 compute programs).
const FINGERPRINT_SAMPLE: u64 = 509;

/// The preparation's rules (the guessed specs, `shaders::guess_compute_spec`, and the variants
/// made): raised whenever they change, since the fingerprint's sample may not show it (2: an
/// image only loaded from guessed Uint by its words' first uses, 2026-11-13; 3: the engine
/// programs' image-dimension variants, `shaders::guess_dim_variants`, 2026-11-13; 4: their
/// depth-view variants too, 2026-11-13; 5: specs as a hosted session makes them,
/// `fe2::specs_as_hosted`, 2026-11-13; 6: the materials' shader bundles too, 2026-11-13; 7:
/// written 3D images as 2D, `shaders::guess_d3_variant`, 2026-11-13; 8: compute images bound
/// 2D or as 2D arrays keyed by their instructions' dimension, `shaders::d2_resources`, and every
/// combination of up to three compared images' depth views, 2026-11-15).
const PREPARATION_RULES: u32 = 8;

/// The translation fingerprint of the prepared shaders: a sample of the game's compute
/// programs (`FINGERPRINT_SAMPLE`) translated as `precompile_all` translates them, their
/// SPIR-V hashed. A port update that changes translations changes it; the prepared pipelines
/// then no longer match (the store takes a binary only for the SPIR-V it was made from), so
/// the launcher asks for another preparation (`prepared.txt`'s `translation`).
pub fn translation_fingerprint(app: &std::path::Path) -> Result<String, String> {
    // (Specs depend on the device's features.)
    super::vk::init().map_err(|e| format!("Vulkan: {e}"))?;
    let progs = bundle_compute_sampled(app, FINGERPRINT_SAMPLE);
    if progs.is_empty() {
        return Err("no compute programs found in the game's shader files".into());
    }
    let mut items: Vec<(u64, u64)> = progs
        .into_iter()
        .filter_map(|(h, code, specs)| {
            let mut spec = specs.into_iter().next()?;
            if spec.lod_stats && !spec.batched && super::shaders::cs_lod_rotation() {
                spec.lod_stats = false;
                spec.lod_lean = false;
            }
            let tr = super::shaders::translate_code_unkept(&code, h, spec).ok()?;
            Some((h, super::shaders::hash_words(&tr.out.spirv)))
        })
        .collect();
    items.sort_unstable();
    let fp = items.iter().fold(0xcbf2_9ce4_8422_2325u64, |a, &(h, s)| ((a ^ h).wrapping_mul(0x100_0000_01b3) ^ s).wrapping_mul(0x100_0000_01b3));
    Ok(format!("{fp:016x}-{}-r{PREPARATION_RULES}", items.len()))
}

/// `bundle_compute`, one program in `sample` by code hash.
fn bundle_compute_sampled(app: &std::path::Path, sample: u64) -> Vec<(u64, std::sync::Arc<Vec<u32>>, Vec<rdna::translate::Spec>)> {
    let mut files: Vec<std::path::PathBuf> = Vec::new();
    let pdir = app.join("particles").join("!cookedparticleshaders").join("version01").join("_ps5");
    files.extend(std::fs::read_dir(&pdir).into_iter().flatten().flatten().map(|e| e.path()));
    // The engine's shaders, and the materials' bundles (their compute programs: 58% of a prepared
    // first session's remaining misses, 2026-11-13; a material's `_trinity` bundle repeats its
    // other one's programs, left out).
    // (A stack: the engine's tree is walked before the materials', so a program in both is
    // the engine's and gets its image variants.)
    let materials = app.join("materials").join("!cookedshaders");
    let mut stack = vec![materials.clone(), app.join("coredata").join("enginesupport").join("shaders").join("_ps5")];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).into_iter().flatten().flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if !(p.starts_with(&materials) && p.file_stem().is_some_and(|s| s.to_string_lossy().ends_with("_trinity"))) {
                files.push(p);
            }
        }
    }
    let mut seen = std::collections::HashSet::new();
    let (mut programs, mut no_object) = (0u64, 0u64);
    let mut out = Vec::new();
    let engine_dir = app.join("coredata");
    let mut variants = 0u64;
    let mut d3_variants = 0u64;
    for f in files.iter().filter(|p| p.extension().is_some_and(|x| x == "csdr")) {
        let Some(v) = std::fs::read(f).ok().and_then(|b| super::particles::bundle_programs(&b)) else { continue };
        let engine = f.starts_with(&engine_dir);
        for p in v.into_iter().filter(|p| p.stage == 0) {
            if !seen.insert(p.hash) || p.hash % sample != 0 {
                continue;
            }
            programs += 1;
            let Some((rsrc2, local)) = object_regs(&p.object) else {
                no_object += 1;
                continue;
            };
            let Ok(insts) = rdna::decode_program(&p.code) else { continue };
            let user_sgprs = ((rsrc2 >> 1) & 0x1F).min(16) as u16;
            let plan = rdna::analysis::analyze_opts(&insts, user_sgprs, 0, false);
            let spec = super::shaders::guess_compute_spec(0, &insts, &plan, p.hash, rsrc2, local);
            // (Diagnosis, DS_CB_OBJ=<hex code hash,...>: the preparation's registers and spec.)
            if let Ok(list) = std::env::var("DS_CB_OBJ") {
                if list.split(',').any(|x| u64::from_str_radix(x.trim().trim_start_matches("0x"), 16).ok() == Some(p.hash)) {
                    crate::info!("prep obj: code {:#x} rsrc2 {rsrc2:#x} local {local:?} user_sgprs {user_sgprs} user_base {}: {spec:?}", p.hash, spec.user_base);
                }
            }
            // (The guess first; the engine's image-dimension variants after it.)
            let mut specs = vec![spec];
            if engine {
                // (2D arrays the dispatches key by their instructions are not flipped to 2D.)
                let keyed2d = if super::shaders::dim_by_instructions() {
                    let (a, b) = super::shaders::d2_resources(&insts, &plan);
                    a | b
                } else {
                    0
                };
                specs.extend(super::shaders::guess_dim_variants(&specs[0], keyed2d));
                variants += specs.len() as u64 - 1;
            }
            if let Some(v) = super::shaders::guess_d3_variant(&specs[0]) {
                specs.push(v);
                d3_variants += 1;
            }
            out.push((p.hash, std::sync::Arc::new(p.code), specs));
        }
    }
    crate::info!("compile bench: {programs} compute programs in the game's shader files (every {sample}th by code hash): {} guessed, {variants} engine image-dimension variants, {d3_variants} written 3D images as 2D, {no_object} without an object read", out.len());
    out
}

/// A spec's hash (the preparation's duplicates).
fn spec_hash(s: &rdna::translate::Spec) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    s.hash(&mut h);
    h.finish()
}

/// `--precompile-all [threads]`: every compute program of the game's shader
/// files with a shader object (`bundle_compute`) is translated (not kept)
/// and its pipeline made through the pipeline store under the key its
/// dispatches look up, the largest first, on `threads` threads (default 8):
/// later sessions then find it instead of compiling it at the program's
/// first dispatch, wherever the dispatch binds what the guess says (86% of
/// the particle programs' learned specs, FINDINGS.md 2026-11-02 evening).
/// Then the player's own learned specs and graphics recipes whose code their
/// store keeps (`precompile::learned_work`): after an update that translates
/// differently, what their sessions used is made again before they play.
/// Pipelines already in the store take moments. Progress every 10 s in the log, and every
/// second in `progress` (the launcher's: `<programs done> <programs>` and `<code words done>
/// <code words>` lines); when all are done, `marker` records it (the launcher's "prepared": the
/// date, the counts, the device and its driver, for which the pipelines hold).
pub fn precompile_all(app: &std::path::Path, threads: usize, progress: Option<&std::path::Path>, marker: Option<&std::path::Path>) -> i32 {
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::Relaxed};
    if let Err(e) = super::vk::init() {
        crate::error!("precompile all: Vulkan: {e}");
        return 2;
    }
    let t_all = Instant::now();
    // Each program's variants as its dispatches prepare them: with LOD statistics in rotation
    // (`shaders::cs_lod_rotation`) a dispatch is prepared without them first (every first use),
    // and a measuring one (one in `tune lod_cs`, binding counted textures: programs that sample
    // images) with them; the variants without statistics come first, all of them, the largest
    // first (they set the end).
    let mut first = Vec::new();
    let mut second = Vec::new();
    // The engine's image variants: [pipelines, code words].
    let mut variants = [0usize; 2];
    for (h, code, specs) in bundle_compute(app) {
        let n = code.len();
        // (The guess and, for the engine's programs, its image-dimension variants.)
        for (k, spec) in specs.into_iter().enumerate() {
            let (f0, s0) = (first.len(), second.len());
            if spec.lod_stats && !spec.batched && super::shaders::cs_lod_rotation() {
                let mut plain = spec.clone();
                plain.lod_stats = false;
                plain.lod_lean = false;
                first.push((h, code.clone(), plain, n));
                if spec.resources.iter().any(|r| matches!(r, rdna::translate::ResSpec::Image { .. })) {
                    second.push((h, code.clone(), spec, n));
                }
            } else {
                first.push((h, code.clone(), spec, n));
            }
            if k > 0 {
                let added = first.len() - f0 + second.len() - s0;
                variants[0] += added;
                variants[1] += added * n;
            }
        }
    }
    first.sort_by(|a, b| b.3.cmp(&a.3));
    second.sort_by(|a, b| b.3.cmp(&a.3));
    crate::info!("precompile all: {} variants dispatches use first, then {} measuring LOD statistics", first.len(), second.len());
    let mut progs: Vec<(u64, std::sync::Arc<Vec<u32>>, rdna::translate::Spec, usize)> = first;
    progs.extend(second);
    // The player's own learned specs and graphics recipes (their store's, kept with their
    // code): what their sessions used, prepared again after an update (`precompile::learned_work`;
    // a spec equal to a guess is made once, the store then holds it).
    let (learned, recipes) = super::precompile::learned_work();
    let guessed = progs.len();
    {
        let have: std::collections::HashSet<(u64, u64)> = progs.iter().map(|p| (p.0, spec_hash(&p.2))).collect();
        let mut extra: Vec<(u64, std::sync::Arc<Vec<u32>>, rdna::translate::Spec, usize)> = Vec::new();
        for (h, code, specs) in learned {
            for spec in specs {
                if !have.contains(&(h, spec_hash(&spec))) {
                    extra.push((h, code.clone(), spec, code.len()));
                }
            }
        }
        extra.sort_by(|a, b| b.3.cmp(&a.3));
        progs.extend(extra);
    }
    let learned_n = progs.len() - guessed;
    let gfx_words: usize = recipes.iter().map(|r| r.1).sum();
    let total_words: usize = progs.iter().map(|p| p.3).sum::<usize>() + gfx_words;
    let all = progs.len() + recipes.len();
    crate::info!(
        "precompile all: {} compute pipelines ({} code words; the engine's image variants {} of them, {} words; the player's learned specs {learned_n}) and {} learned graphics pipelines on {threads} threads",
        progs.len(),
        total_words - gfx_words,
        variants[0],
        variants[1],
        recipes.len()
    );
    let next = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
    let code_done = AtomicU64::new(0);
    let failed = AtomicUsize::new(0);
    let compile_ns = AtomicU64::new(0);
    // A stop asked for by the launcher (`<progress>.stop`): the programs being compiled finish,
    // no other starts.
    let stop = std::sync::atomic::AtomicBool::new(false);
    let stop_file = progress.map(|p| p.with_extension("stop"));
    std::thread::scope(|s| {
        for _ in 0..threads.max(1) {
            s.spawn(|| loop {
                if stop.load(Relaxed) {
                    break;
                }
                let i = next.fetch_add(1, Relaxed);
                if i >= all {
                    break;
                }
                // (The compute pipelines first, then the learned graphics ones.)
                if let Some((h, code, spec, n)) = progs.get(i) {
                    let r = super::shaders::translate_code_unkept(code, *h, spec.clone()).and_then(|tr| super::exec::store_compute_pipeline(&tr));
                    match r {
                        Ok(d) => {
                            compile_ns.fetch_add(d.as_nanos() as u64, Relaxed);
                        }
                        Err(e) => {
                            if failed.fetch_add(1, Relaxed) < 8 {
                                crate::warn!("precompile all: {h:#018x}: {e}");
                            }
                        }
                    }
                    code_done.fetch_add(*n as u64, Relaxed);
                } else {
                    let (r, n) = recipes[i - progs.len()];
                    let t = Instant::now();
                    if !super::precompile::prepare_gfx(r) {
                        failed.fetch_add(1, Relaxed);
                    }
                    compile_ns.fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
                    code_done.fetch_add(n as u64, Relaxed);
                }
                done.fetch_add(1, Relaxed);
            });
        }
        // Progress: programs and code words done, and the time left at
        // the rate of the words so far.
        let mut last = Instant::now();
        let mut last_file = Instant::now();
        let write_progress = || {
            if let Some(p) = progress {
                let t = format!("{} {}\n{} {}\n", done.load(Relaxed), all, code_done.load(Relaxed), total_words);
                let tmp = p.with_extension("tmp");
                if std::fs::write(&tmp, t).is_ok() {
                    let _ = std::fs::rename(&tmp, p);
                }
            }
        };
        write_progress();
        while done.load(Relaxed) < all {
            std::thread::sleep(std::time::Duration::from_millis(200));
            if !stop.load(Relaxed) && stop_file.as_ref().is_some_and(|f| f.exists()) {
                crate::info!("precompile all: stop asked for at {} of {} programs", done.load(Relaxed), all);
                stop.store(true, Relaxed);
            }
            if stop.load(Relaxed) {
                // (Done once the threads in a compile come back.)
                if next.load(Relaxed) >= all || done.load(Relaxed) >= next.load(Relaxed).min(all) {
                    break;
                }
                continue;
            }
            if last_file.elapsed().as_millis() >= 1000 {
                last_file = Instant::now();
                write_progress();
            }
            if last.elapsed().as_secs() >= 10 {
                last = Instant::now();
                let w = code_done.load(Relaxed) as f64;
                let el = t_all.elapsed().as_secs_f64();
                let left = if w > 0.0 { el * (total_words as f64 - w) / w } else { 0.0 };
                crate::info!("precompile all: {} of {} programs, {:.0}% of the code, {:.0} s, about {:.0} s left", done.load(Relaxed), all, 100.0 * w / total_words.max(1) as f64, el, left);
            }
        }
    });
    if stop.load(Relaxed) {
        crate::info!("precompile all: stopped at {} of {} programs ({:.1} s)", done.load(Relaxed), all, t_all.elapsed().as_secs_f64());
        return 4;
    }
    crate::info!(
        "precompile all: {} programs in {:.1} s ({:.1} s of pipeline creation over the threads; {} failed)",
        all,
        t_all.elapsed().as_secs_f64(),
        compile_ns.load(Relaxed) as f64 / 1e9,
        failed.load(Relaxed)
    );
    if let Some(p) = progress {
        let _ = std::fs::write(p, format!("{} {}\n{} {}\n", all, all, total_words, total_words));
    }
    if let Some(m) = marker {
        let c = super::vk::ctx();
        let props = unsafe { c.instance.get_physical_device_properties(c.pdev) };
        let uuid: String = props.pipeline_cache_uuid.iter().map(|b| format!("{b:02x}")).collect();
        // (What the pipelines were translated as: another build that translates differently
        // makes them stale, `translation_fingerprint`.)
        let translation = translation_fingerprint(app).unwrap_or_else(|e| {
            crate::warn!("precompile all: no translation fingerprint ({e})");
            String::new()
        });
        let t = format!(
            "# The game's shaders prepared ahead (dsport --precompile-all); written when all were done\ndate = {}\nprograms = {}\nfailed = {}\nseconds = {:.0}\ndevice = {}\ndriver = {}\ncache_uuid = {uuid}\ntranslation = {translation}\n",
            crate::utc_stamp(),
            all,
            failed.load(Relaxed),
            t_all.elapsed().as_secs_f64(),
            c.device_name,
            props.driver_version,
        );
        if let Err(e) = std::fs::write(m, t) {
            crate::warn!("precompile all: cannot write {}: {e}", m.display());
        }
    }
    0
}

/// The process's committed private bytes.
fn private_bytes() -> u64 {
    use windows_sys::Win32::System::ProcessStatus::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS, PROCESS_MEMORY_COUNTERS_EX};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut c: PROCESS_MEMORY_COUNTERS_EX = unsafe { std::mem::zeroed() };
    c.cb = std::mem::size_of::<PROCESS_MEMORY_COUNTERS_EX>() as u32;
    let ok = unsafe { GetProcessMemoryInfo(GetCurrentProcess(), &mut c as *mut PROCESS_MEMORY_COUNTERS_EX as *mut PROCESS_MEMORY_COUNTERS, c.cb) };
    if ok == 0 { 0 } else { c.PrivateUsage as u64 }
}

/// `DS_CB_MEMTEST=<n>` (with `--compile-bench`): the host memory a pipeline
/// made through the pipeline store holds while it lives and after it is
/// destroyed: up to n learned compute pipelines (keyed as the precompile
/// keys them) made and kept, all destroyed, then made and destroyed one at
/// a time; the process's committed bytes after each phase.
fn memtest(n: usize) -> i32 {
    if let Err(e) = super::vk::init() {
        crate::error!("memtest: Vulkan: {e}");
        return 2;
    }
    // The translations first (their memory apart from the pipelines').
    let mut trs = Vec::new();
    'progs: for (h, code, _) in super::precompile::learned_compute() {
        for spec in super::precompile::keyed_learned_specs(h, &code) {
            match super::shaders::translate_code_for_spec(&code, h, spec) {
                Ok(tr) => trs.push(tr),
                Err(_) => continue,
            }
            if trs.len() >= n {
                break 'progs;
            }
        }
    }
    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
    let c0 = private_bytes();
    crate::info!("memtest: {} translations ready; committed {:.3} GiB", trs.len(), gib(c0));
    let t = Instant::now();
    let mut kept = Vec::new();
    for tr in &trs {
        match super::exec::store_compute_pipeline_parts(tr) {
            Ok(p) => kept.push(p),
            Err(e) => crate::warn!("memtest: {e}"),
        }
    }
    let c1 = private_bytes();
    crate::info!("memtest: {} pipelines made and kept in {:.1} s: committed {:.3} GiB (+{:.2} MiB a pipeline)", kept.len(), t.elapsed().as_secs_f64(), gib(c1), (c1 as f64 - c0 as f64) / (1u64 << 20) as f64 / kept.len().max(1) as f64);
    let made = kept.len();
    for p in kept.drain(..) {
        super::exec::destroy_pipeline_parts(p);
    }
    let c2 = private_bytes();
    crate::info!("memtest: all destroyed: committed {:.3} GiB ({:+.2} MiB a pipeline against before)", gib(c2), (c2 as f64 - c0 as f64) / (1u64 << 20) as f64 / made.max(1) as f64);
    let t = Instant::now();
    for tr in &trs {
        if let Ok(p) = super::exec::store_compute_pipeline_parts(tr) {
            super::exec::destroy_pipeline_parts(p);
        }
    }
    let c3 = private_bytes();
    crate::info!("memtest: made and destroyed one at a time again in {:.1} s: committed {:.3} GiB ({:+.2} MiB a pipeline against before)", t.elapsed().as_secs_f64(), gib(c3), (c3 as f64 - c2 as f64) / (1u64 << 20) as f64 / made.max(1) as f64);
    0
}

/// `DS_CB_SHAPES=1`: no compiles; the game's compute programs (particle
/// bundles and engine shader files) grouped by their code with constants
/// left out: L1 without literal dwords, L2 also without inline constants and
/// SOPK immediates. How many pipelines would serve them if constants were
/// read at run time, by count and by instructions (compile time follows
/// program size), and the largest groups.
fn shapes_report(app: &std::path::Path) -> i32 {
    use std::collections::HashMap;
    let mut files: Vec<(std::path::PathBuf, bool)> = Vec::new();
    let pdir = app.join("particles").join("!cookedparticleshaders").join("version01").join("_ps5");
    files.extend(std::fs::read_dir(&pdir).into_iter().flatten().flatten().map(|e| (e.path(), true)));
    let mut stack = vec![app.join("coredata").join("enginesupport").join("shaders").join("_ps5")];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).into_iter().flatten().flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else {
                files.push((p, false));
            }
        }
    }
    let fnv = |h: &mut u64, v: u64| {
        *h ^= v;
        *h = h.wrapping_mul(0x100_0000_01b3);
    };
    // Per class (particle, engine): programs, instructions, and per level the
    // shapes with their member count and instructions.
    let mut seen = std::collections::HashSet::new();
    let mut tot = [(0u64, 0u64); 2];
    let mut shapes: [[HashMap<u64, (u64, u64)>; 2]; 2] = Default::default();
    let mut lits = [0u64; 2];
    for (f, particle) in files.iter().filter(|(p, _)| p.extension().is_some_and(|x| x == "csdr")) {
        let Some(v) = std::fs::read(f).ok().and_then(|b| super::particles::bundle_programs(&b)) else { continue };
        for p in v.into_iter().filter(|p| p.stage == 0) {
            if !seen.insert(p.hash) {
                continue;
            }
            let Ok(insts) = rdna::decode_program(&p.code) else { continue };
            let c = (!*particle) as usize;
            let n = insts.len() as u64;
            tot[c].0 += 1;
            tot[c].1 += n;
            let (mut h1, mut h2) = (0xcbf2_9ce4_8422_2325u64, 0xcbf2_9ce4_8422_2325u64);
            for i in &insts {
                let ext = match i.ext {
                    rdna::Ext::None => 0u64,
                    rdna::Ext::Dpp16(d) => 1 << 40 | d as u64,
                    rdna::Ext::Dpp8(d, fi) => 2 << 40 | (fi as u64) << 32 | d as u64,
                    rdna::Ext::Sdwa(d) => 3 << 40 | d as u64,
                };
                if i.lit.is_some() {
                    lits[c] += 1;
                }
                let base = (i.enc as u64) << 48 | (i.op as u64) << 32;
                fnv(&mut h1, base | i.w0 as u64);
                fnv(&mut h1, i.w1 as u64);
                fnv(&mut h1, ext);
                // L2: constant operands (inline constants and the literal,
                // 128..=255) as one marker; SOPK immediates of the moves,
                // adds and multiplies left out.
                let is_const = |o: u32| (128..=255).contains(&o);
                let (mut w0, mut w1) = (i.w0, i.w1);
                match i.enc {
                    rdna::Enc::Vop3 | rdna::Enc::Vop3p => {
                        for k in 0..3 {
                            if is_const((w1 >> (9 * k)) & 0x1FF) {
                                w1 = (w1 & !(0x1FF << (9 * k))) | (128 << (9 * k));
                            }
                        }
                    }
                    rdna::Enc::Vop1 | rdna::Enc::Vop2 | rdna::Enc::Vopc => {
                        if is_const(w0 & 0x1FF) && !matches!(w0 & 0x1FF, 233 | 234 | 249 | 250) {
                            w0 = (w0 & !0x1FF) | 128;
                        }
                    }
                    rdna::Enc::Sop2 | rdna::Enc::Sopc => {
                        for k in 0..2 {
                            if is_const((w0 >> (8 * k)) & 0xFF) {
                                w0 = (w0 & !(0xFF << (8 * k))) | (128 << (8 * k));
                            }
                        }
                    }
                    rdna::Enc::Sop1 => {
                        if is_const(w0 & 0xFF) {
                            w0 = (w0 & !0xFF) | 128;
                        }
                    }
                    rdna::Enc::Sopk => {
                        // s_movk_i32, s_cmovk_i32, s_addk_i32, s_mulk_i32.
                        if matches!(i.op, 0 | 2 | 15 | 16) {
                            w0 &= !0xFFFF;
                        }
                    }
                    _ => {}
                }
                fnv(&mut h2, base | w0 as u64);
                fnv(&mut h2, w1 as u64);
                fnv(&mut h2, ext);
            }
            for (lvl, h) in [h1, h2].into_iter().enumerate() {
                let e = shapes[lvl][c].entry(h).or_insert((0, 0));
                e.0 += 1;
                e.1 = n;
            }
        }
    }
    for (c, name) in ["particle", "engine"].iter().enumerate() {
        crate::info!("shapes: {name} programs {} ({} instructions, {} with a literal)", tot[c].0, tot[c].1, lits[c]);
        for lvl in 0..2 {
            let m = &shapes[lvl][c];
            let insts: u64 = m.values().map(|v| v.1).sum();
            let singles = m.values().filter(|v| v.0 == 1).count();
            crate::info!(
                "shapes: {name} L{}: {} shapes ({:.1}% of programs), {} instructions to compile ({:.1}%), {singles} programs alone in theirs",
                lvl + 1,
                m.len(),
                100.0 * m.len() as f64 / tot[c].0.max(1) as f64,
                insts,
                100.0 * insts as f64 / tot[c].1.max(1) as f64
            );
            let mut big: Vec<(u64, u64)> = m.values().copied().collect();
            big.sort_by(|a, b| (b.0 * b.1).cmp(&(a.0 * a.1)));
            let top: Vec<String> = big.iter().take(12).map(|(k, n)| format!("{k}x{n}")).collect();
            crate::info!("shapes: {name} L{} largest groups (members x instructions): {}", lvl + 1, top.join(" "));
        }
    }
    0
}

pub fn run(args: &[String], app: &std::path::Path) -> i32 {
    if std::env::var("DS_CB_GUESS").ok().as_deref() == Some("1") {
        return guess_report(app);
    }
    if std::env::var("DS_CB_SHAPES").ok().as_deref() == Some("1") {
        return shapes_report(app);
    }
    if let Some(n) = std::env::var("DS_CB_MEMTEST").ok().and_then(|v| v.parse::<usize>().ok()).filter(|&n| n > 0) {
        return memtest(n);
    }
    let min_insts: usize = args.first().and_then(|s| s.parse().ok()).unwrap_or(2000);
    let max: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(40);
    let mut only: Vec<u64> = std::env::var("DS_CB_HASHES").map(|v| v.split(',').filter_map(|h| u64::from_str_radix(h.trim().trim_start_matches("0x"), 16).ok()).collect()).unwrap_or_default();
    // DS_CB_BINS=<file,...>: programs saved by a run (`shaders/<address>.bin`),
    // found in the store by their code hash (';' separates them too: the
    // run scripts split their variables at commas).
    for f in std::env::var("DS_CB_BINS").map(|v| v.split([',', ';']).filter(|f| !f.is_empty()).map(str::to_string).collect::<Vec<_>>()).unwrap_or_default() {
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
    // (DS_CB_NONCE=0: modules as translated, which the driver's shader
    // cache may serve.)
    super::exec::FORCE_NONCE.store(std::env::var("DS_CB_NONCE").ok().as_deref() != Some("0"), std::sync::atomic::Ordering::Relaxed);
    if let Err(e) = super::vk::init() {
        crate::error!("compile bench: Vulkan: {e}");
        return 2;
    }
    // DS_CB_BUNDLES=1: the game's own compute programs (the particle
    // bundles' and the engine's shader files) instead of the learned ones,
    // each with the spec guessed from its shader object: what compiling
    // them all ahead would take (DS_CB_SAMPLE=<n>: every n-th by code
    // hash).
    let source = if std::env::var("DS_CB_BUNDLES").ok().as_deref() == Some("1") { bundle_compute(app) } else { super::precompile::learned_compute() };
    // DS_CB_MAX_INSTS=<n>: programs of at most n instructions.
    let max_insts = std::env::var("DS_CB_MAX_INSTS").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(usize::MAX);
    let mut progs: Vec<(u64, std::sync::Arc<Vec<u32>>, Vec<rdna::translate::Spec>, usize)> = source
        .into_iter()
        .filter(|(h, _, specs)| !specs.is_empty() && (only.is_empty() || only.contains(h)))
        .filter_map(|(h, code, specs)| {
            let n = rdna::decode_program(&code).ok()?.len();
            ((n >= min_insts || !only.is_empty()) && n <= max_insts).then_some((h, code, specs, n))
        })
        .collect();
    progs.sort_by(|a, b| b.3.cmp(&a.3));
    progs.truncate(max);
    crate::info!("compile bench: {} programs of {min_insts} instructions or more (the largest first)", progs.len());
    let mut rows: Vec<(u64, usize, usize, f64, f64)> = Vec::new();
    let t_all = Instant::now();
    // DS_CB_PER_LANE=1: translated per lane (`Spec::per_lane`) whatever the
    // learned spec says; 2 with subgroup-decided EXEC branches too (timing:
    // whether the program is right that way is not checked).
    let per_lane = std::env::var("DS_CB_PER_LANE").ok().and_then(|v| v.parse::<u32>().ok()).unwrap_or(0);
    // DS_CB_SPEC=<n>: the program's n-th learned spec (in store order; the
    // last if it has fewer) instead of its first.
    let spec_ix = std::env::var("DS_CB_SPEC").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(0);
    for (h, code, specs, n) in &progs {
        let t0 = Instant::now();
        let mut spec = specs[spec_ix.min(specs.len() - 1)].clone();
        if per_lane != 0 {
            spec.per_lane = true;
            spec.lane_votes = per_lane == 2;
        }
        // DS_CB_LOCKSTEP_OFF=1: without lockstep (`Spec::lockstep_off`).
        if std::env::var("DS_CB_LOCKSTEP_OFF").ok().as_deref() == Some("1") {
            spec.lockstep_off = true;
        }
        // DS_CB_SG_WAVES=1: as subgroup waves (`Spec::sg_waves`).
        if std::env::var("DS_CB_SG_WAVES").ok().as_deref() == Some("1") {
            spec.sg_waves = true;
        }
        // DS_CB_NO_GUARDS=1: VGPR writes without their lane guards
        // (`Spec::lane_guards`; sizing only: wrong under a partial EXEC).
        if std::env::var("DS_CB_NO_GUARDS").ok().as_deref() == Some("1") {
            spec.lane_guards = false;
        }
        // (Learned specs as a preparation keys them now; `DS_CB_FORMATS=1`
        // keeps the formats the learned dispatches had instead of the
        // run-time decode, `FORMAT_RT`: what specializing them would save.)
        if let Ok(insts) = rdna::decode_program(code) {
            let plan = rdna::analysis::analyze_opts(&insts, spec.user_sgprs, spec.user_base, false);
            let learned = spec.resources.clone();
            super::shaders::key_learned(*h, &insts, &plan, &mut spec.resources);
            if std::env::var("DS_CB_FORMATS").ok().as_deref() == Some("1") {
                for (r, l) in spec.resources.iter_mut().zip(learned.iter()) {
                    if matches!(r, rdna::translate::ResSpec::Buffer { format, .. } if *format == rdna::translate::FORMAT_RT) {
                        *r = l.clone();
                    }
                }
            }
        }
        let tr = match super::shaders::translate_code_for_spec(code, *h, spec) {
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
        // DS_CB_NOCOMPILE=1: translations only (with DS_CB_SPV: the modules
        // for `spvval`).
        if std::env::var("DS_CB_NOCOMPILE").ok().as_deref() == Some("1") {
            rows.push((*h, *n, tr.out.spirv.len(), translate_ms, 0.0));
            continue;
        }
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
    // DS_CB_ROWS=<file>: every program's row (code hash, instructions,
    // SPIR-V words, translation and compile ms) as CSV.
    if let Some(f) = std::env::var("DS_CB_ROWS").ok().filter(|f| !f.is_empty()) {
        let csv: String = rows.iter().map(|(h, n, w, t, c)| format!("{h:016x},{n},{w},{t:.1},{c:.1}\n")).collect();
        if let Err(e) = std::fs::write(&f, csv) {
            crate::warn!("compile bench: cannot write {f}: {e}");
        }
    }
    0
}
