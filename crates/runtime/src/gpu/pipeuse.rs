//! Which pipelines recent sessions used (`<shader-spec store>.used`), so the
//! precompile (`gpu::precompile`) keeps alive only those it will likely
//! need. Every pipeline costs the driver host memory while it lives: ~2.6
//! MiB made from pipeline binaries, ~0.5 MiB through cache files (`dsport
//! --compile-bench` with `DS_CB_MEMTEST`, FINDINGS.md 2026-11-09); a
//! precompile of every learned spec and recipe made ~15,000 pipelines a
//! boot, ~2,000 of them used, and the process committed 73 GB.
//!
//! The file holds lines `<session> <key>` (a pipeline map key: compute
//! `push_key(translation key, push mode)`, graphics the pipeline-state
//! hash), one per pipeline a session used, appended every 10 s by a
//! background thread. A session's number is the file's highest plus one.
//! "Recent": used in one of the last `DS_PIPE_RECENT` sessions (default 16).

use parking_lot::Mutex;
use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

static PATH: OnceLock<std::path::PathBuf> = OnceLock::new();

struct State {
    session: u64,
    recent: HashSet<u64>,
}

static STATE: OnceLock<State> = OnceLock::new();
/// Keys used this session, noted once each, and those not written yet.
static NOTED: Mutex<Option<HashSet<u64>>> = Mutex::new(None);
static PENDING: Mutex<Vec<u64>> = Mutex::new(Vec::new());
/// [keys used this session, keys written, recent keys loaded, lines read].
static STATS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

thread_local! {
    /// Keys this thread noted already (no global lock after a key's first
    /// use on a thread).
    static HERE: std::cell::RefCell<HashSet<u64>> = std::cell::RefCell::new(HashSet::new());
}

/// The usage file beside the shader-spec store (`precompile::set_store`).
pub fn set_path(store: &std::path::Path) {
    let mut name = store.file_name().map(|n| n.to_os_string()).unwrap_or_default();
    name.push(".used");
    let _ = PATH.set(store.with_file_name(name));
}

fn keep_sessions() -> u64 {
    std::env::var("DS_PIPE_RECENT").ok().and_then(|v| v.parse().ok()).unwrap_or(16)
}

fn state() -> &'static State {
    STATE.get_or_init(|| {
        let mut last: HashMap<u64, u64> = HashMap::new();
        let mut top = 0u64;
        let mut lines = 0u64;
        if let Some(text) = PATH.get().and_then(|p| std::fs::read_to_string(p).ok()) {
            for l in text.lines() {
                let mut it = l.split_whitespace();
                let (Some(s), Some(k)) = (it.next().and_then(|s| s.parse::<u64>().ok()), it.next().and_then(|k| u64::from_str_radix(k, 16).ok())) else { continue };
                lines += 1;
                top = top.max(s);
                let e = last.entry(k).or_insert(0);
                *e = (*e).max(s);
            }
        }
        let session = top + 1;
        let keep = keep_sessions();
        let recent: HashSet<u64> = last.iter().filter(|&(_, &s)| s + keep >= session).map(|(&k, _)| k).collect();
        // A file of many repeated or old lines is written again with each
        // key's last session (keys older than four times the window go).
        if lines > 2 * last.len() as u64 + 4096 {
            if let Some(p) = PATH.get() {
                let mut out = String::new();
                for (k, s) in &last {
                    if s + 4 * keep >= session {
                        out.push_str(&format!("{s} {k:016x}\n"));
                    }
                }
                let tmp = p.with_extension("used.tmp");
                if std::fs::write(&tmp, out).is_ok() {
                    let _ = std::fs::rename(&tmp, p);
                }
            }
        }
        STATS[2].store(recent.len() as u64, Ordering::Relaxed);
        STATS[3].store(lines, Ordering::Relaxed);
        crate::info!("gpu: pipeline uses: session {session}, {} pipelines used in the last {keep} sessions ({lines} lines read)", recent.len());
        let _ = std::thread::Builder::new().name("pipe-uses".into()).spawn(|| loop {
            std::thread::sleep(std::time::Duration::from_secs(10));
            flush();
        });
        State { session, recent }
    })
}

/// Whether a recent session used the pipeline of `key`.
pub fn recent(key: u64) -> bool {
    state().recent.contains(&key)
}

/// A draw or dispatch used the pipeline of `key`.
#[inline]
pub fn note(key: u64) {
    if HERE.with(|h| h.borrow().contains(&key)) {
        return;
    }
    note_slow(key);
}

#[cold]
fn note_slow(key: u64) {
    HERE.with(|h| h.borrow_mut().insert(key));
    let first = NOTED.lock().get_or_insert_with(HashSet::new).insert(key);
    if first {
        STATS[0].fetch_add(1, Ordering::Relaxed);
        PENDING.lock().push(key);
    }
}

/// Appends the keys used since the last flush.
fn flush() {
    let keys = std::mem::take(&mut *PENDING.lock());
    if keys.is_empty() {
        return;
    }
    let Some(p) = PATH.get() else { return };
    let s = state().session;
    let mut out = String::with_capacity(keys.len() * 24);
    for k in &keys {
        out.push_str(&format!("{s} {k:016x}\n"));
    }
    match std::fs::OpenOptions::new().create(true).append(true).open(p).and_then(|mut f| f.write_all(out.as_bytes())) {
        Ok(()) => {
            STATS[1].fetch_add(keys.len() as u64, Ordering::Relaxed);
        }
        Err(e) => {
            static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            if !WARNED.swap(true, Ordering::Relaxed) {
                crate::warn!("gpu: pipeline uses not stored in {} ({e})", p.display());
            }
        }
    }
}

/// Status line part.
pub fn line() -> String {
    if STATE.get().is_none() {
        return String::new();
    }
    format!(
        "pipeline uses (session {}): {} pipelines used this session, {} written; {} used in recent sessions",
        state().session,
        STATS[0].load(Ordering::Relaxed),
        STATS[1].load(Ordering::Relaxed),
        STATS[2].load(Ordering::Relaxed)
    )
}
