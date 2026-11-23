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
//! The engine's own programs (coredata/enginesupport/shaders, one directory
//! per effect) are read too: each program's directory names it in the logs,
//! and those of per-frame effects whose results only look late when left
//! out (`DS_ASYNC_ENGINE`, directory names; default the particles' lighting
//! atlas passes, particle-lightmap-update) compile in the background the
//! same way.

use parking_lot::RwLock;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// The game's shader folders (paths in its files, `dsconfig::zar::Source`): the particles'
/// bundles and the engine's own programs.
pub const PARTICLE_SHADERS: &str = "particles/!cookedparticleshaders/version01/_ps5";
pub const ENGINE_SHADERS: &str = "coredata/enginesupport/shaders/_ps5";

/// Whether a file name or path has the extension (as `Path::extension` reads it).
pub fn has_ext(name: &str, ext: &str) -> bool {
    std::path::Path::new(name).extension().is_some_and(|x| x == ext)
}

static HASHES: RwLock<Option<HashSet<u64>>> = RwLock::new(None);
/// The engine's programs by code hash: their directory.
static ENGINE: RwLock<Option<HashMap<u64, Arc<str>>>> = RwLock::new(None);
/// Engine programs compiled in the background (`DS_ASYNC_ENGINE`).
static ENGINE_ASYNC: RwLock<Option<HashSet<u64>>> = RwLock::new(None);
/// [bundles read, programs hashed, bundles not understood, ms taken,
/// programs whose last instruction ends before their code size].
static STATS: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];

/// Reads the bundles in the game's files (its folder or archive) on a
/// background thread; until it is done no program counts as a particle one.
pub fn start(src: dsconfig::zar::Source) {
    let _ = crate::thread::spawn_host("particle-scan", 1 << 20, Box::new(move || scan(&src)));
}

fn scan(src: &dsconfig::zar::Source) {
    crate::log::set_thread_name("particle-scan");
    let t0 = std::time::Instant::now();
    let Ok(rd) = src.list(PARTICLE_SHADERS) else {
        crate::warn!("particles: no shader bundles at {PARTICLE_SHADERS} in {}", src.display());
        return;
    };
    let mut set = HashSet::new();
    for e in rd {
        if !has_ext(&e.name, "csdr") {
            continue;
        }
        let Ok(b) = src.read(&format!("{PARTICLE_SHADERS}/{}", e.name)) else { continue };
        STATS[0].fetch_add(1, Ordering::Relaxed);
        if !programs(&b, &mut set) {
            STATS[2].fetch_add(1, Ordering::Relaxed);
        }
    }
    STATS[1].store(set.len() as u64, Ordering::Relaxed);
    STATS[3].store(t0.elapsed().as_millis() as u64, Ordering::Relaxed);
    crate::info!("particles: {} programs in {} shader bundles ({} not understood; {} with code past their last instruction) read in {} ms", set.len(), STATS[0].load(Ordering::Relaxed), STATS[2].load(Ordering::Relaxed), STATS[4].load(Ordering::Relaxed), t0.elapsed().as_millis());
    *HASHES.write() = Some(set);
    scan_engine(src);
}

/// The engine's shader files: each program's code hash and its directory.
fn scan_engine(src: &dsconfig::zar::Source) {
    let t0 = std::time::Instant::now();
    let dirs: Vec<String> = std::env::var("DS_ASYNC_ENGINE").unwrap_or_else(|_| "particle-lightmap-update".into()).split(',').map(|d| d.trim().to_string()).filter(|d| !d.is_empty()).collect();
    let mut map: HashMap<u64, Arc<str>> = HashMap::new();
    let mut slow = HashSet::new();
    let (mut files, mut odd) = (0u64, 0u64);
    let mut stack: Vec<(String, Option<Arc<str>>)> = vec![(ENGINE_SHADERS.to_string(), None)];
    while let Some((d, top)) = stack.pop() {
        for e in src.list(&d).unwrap_or_default() {
            let p = format!("{d}/{}", e.name);
            if e.dir {
                let name = top.clone().unwrap_or_else(|| Arc::from(e.name.as_str()));
                stack.push((p, Some(name)));
            } else if has_ext(&e.name, "csdr") {
                let Some(dir) = top.clone() else { continue };
                files += 1;
                match src.read(&p).ok().and_then(|b| bundle_programs(&b)) {
                    Some(v) => {
                        for prog in v {
                            if dirs.iter().any(|x| **x == *dir) {
                                slow.insert(prog.hash);
                            }
                            map.insert(prog.hash, dir.clone());
                        }
                    }
                    None => odd += 1,
                }
            }
        }
    }
    crate::info!("particles: the engine's shaders: {} programs in {files} files ({odd} not understood) read in {} ms; {} of them compiled in the background ({})", map.len(), t0.elapsed().as_millis(), slow.len(), dirs.join(", "));
    *ENGINE.write() = Some(map);
    *ENGINE_ASYNC.write() = Some(slow);
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

/// Whether the program with this code hash compiles in the background at
/// first use: an engine program of a directory in `DS_ASYNC_ENGINE`, or
/// (`particles`) a particle program.
pub fn in_background(code_hash: u64, particles: bool) -> bool {
    (particles && is_particle(code_hash)) || ENGINE_ASYNC.read().as_ref().is_some_and(|s| s.contains(&code_hash))
}

/// Where the program with this code hash comes from: an engine shader
/// directory or the particle bundles (logs).
pub fn origin(code_hash: u64) -> Option<Arc<str>> {
    if let Some(d) = ENGINE.read().as_ref().and_then(|m| m.get(&code_hash).cloned()) {
        return Some(d);
    }
    is_particle(code_hash).then(|| Arc::from("particle bundles"))
}

/// Programs known (status lines).
pub fn known() -> u64 {
    STATS[1].load(Ordering::Relaxed)
}
