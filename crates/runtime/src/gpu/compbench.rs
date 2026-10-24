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

pub fn run(args: &[String]) -> i32 {
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
