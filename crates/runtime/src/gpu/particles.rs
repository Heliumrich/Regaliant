//! The title's GPU particle programs (particles/!cookedparticleshaders):
//! their code hashes, read from the game files once on a background thread
//! at start. A bundle (`.csdr`) holds a header (a word, the program count), five
//! words per program (stage, flag, code bytes, metadata bytes, the code's
//! offset from that word), then each program's code followed by its
//! 0x130-byte shader object; most hold an effect's three compute programs.
//! Their pipelines are compiled in the background at first use and their
//! dispatches left out until then (`exec::compile_in_background`, `tune
//! async_particles`): the largest take 5-12 s each to compile, and a first
//! visit froze while they did (FINDINGS.md 2026-10-23/24).

use parking_lot::RwLock;
use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};

static HASHES: RwLock<Option<HashSet<u64>>> = RwLock::new(None);
/// [bundles read, programs hashed, bundles not understood, ms taken,
/// programs whose last instruction ends before their code size].
static STATS: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];

/// Reads the bundles under `app` (the PPSA01341-app0 folder) on a
/// background thread; until it is done no program counts as a particle one.
pub fn start(app: std::path::PathBuf) {
    let _ = crate::thread::spawn_host("particle-scan", 1 << 20, Box::new(move || scan(&app)));
}

fn scan(app: &std::path::Path) {
    crate::log::set_thread_name("particle-scan");
    let t0 = std::time::Instant::now();
    let dir = app.join("particles").join("!cookedparticleshaders").join("version01").join("_ps5");
    let Ok(rd) = std::fs::read_dir(&dir) else {
        crate::warn!("particles: no shader bundles at {}", dir.display());
        return;
    };
    let mut set = HashSet::new();
    for e in rd.flatten() {
        let p = e.path();
        if p.extension().is_none_or(|x| x != "csdr") {
            continue;
        }
        let Ok(b) = std::fs::read(&p) else { continue };
        STATS[0].fetch_add(1, Ordering::Relaxed);
        if !programs(&b, &mut set) {
            STATS[2].fetch_add(1, Ordering::Relaxed);
        }
    }
    STATS[1].store(set.len() as u64, Ordering::Relaxed);
    STATS[3].store(t0.elapsed().as_millis() as u64, Ordering::Relaxed);
    crate::info!("particles: {} programs in {} shader bundles ({} not understood; {} with code past their last instruction) read in {} ms", set.len(), STATS[0].load(Ordering::Relaxed), STATS[2].load(Ordering::Relaxed), STATS[4].load(Ordering::Relaxed), t0.elapsed().as_millis());
    *HASHES.write() = Some(set);
}

/// A program of a shader bundle: its stage, code hash, code (up to its last
/// instruction, as the run time hashes it) and the shader object after it.
pub(super) struct BundleProgram {
    pub stage: u32,
    pub hash: u64,
    pub code: Vec<u32>,
    pub object: Vec<u8>,
}

/// The programs of a shader bundle (`.csdr`), None when its layout is not
/// the expected one.
pub(super) fn bundle_programs(b: &[u8]) -> Option<Vec<BundleProgram>> {
    let rd = |o: usize| b.get(o..o + 4).map(|x| u32::from_le_bytes([x[0], x[1], x[2], x[3]]));
    // (The first word varies by bundle, 0x1d00-0x3c00 seen.)
    let (Some(_), Some(n)) = (rd(0), rd(4)) else { return None };
    if n == 0 || n > 64 {
        return None;
    }
    let mut out = Vec::new();
    for i in 0..n as usize {
        let e = 8 + 20 * i;
        let (stage, size, meta, off) = (rd(e)?, rd(e + 8)?, rd(e + 12)?, rd(e + 16)?);
        let at = e + 16 + off as usize;
        let code = b.get(at..at + size as usize)?;
        let object = b.get(at + size as usize..at + size as usize + meta as usize).unwrap_or(&[]).to_vec();
        let words: Vec<u32> = code.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        let end = rdna::decode_program(&words).ok().and_then(|v| v.last().map(|l| (l.pc / 4 + l.len as u32) as usize))?;
        if end != words.len() {
            STATS[4].fetch_add(1, Ordering::Relaxed);
        }
        let code = words[..end.min(words.len())].to_vec();
        out.push(BundleProgram { stage, hash: super::shaders::hash_words(&code), code, object });
    }
    Some(out)
}

/// Adds the code hashes of a bundle's programs: false when its layout is
/// not the expected one.
fn programs(b: &[u8], set: &mut HashSet<u64>) -> bool {
    match bundle_programs(b) {
        Some(v) => {
            set.extend(v.iter().map(|p| p.hash));
            true
        }
        None => false,
    }
}

/// Whether the program with this code hash is one of the title's particle
/// programs.
pub fn is_particle(code_hash: u64) -> bool {
    HASHES.read().as_ref().is_some_and(|s| s.contains(&code_hash))
}

/// Programs known (status lines).
pub fn known() -> u64 {
    STATS[1].load(Ordering::Relaxed)
}
