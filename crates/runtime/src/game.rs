//! The game's files: its folder, or its `.zar` archive read in place (`dsconfig::zar`, since
//! 2026-11-23: issue #3). Set once at start; the guest's `/app0` mount (`hle::kernel::fs`), the
//! loader, the shader files' readers and the trophies read the game through it.

use dsconfig::zar::Source;
use std::sync::OnceLock;

static SOURCE: OnceLock<Source> = OnceLock::new();

/// The game's files from now on (the first call's).
pub fn set(s: Source) {
    let _ = SOURCE.set(s);
}

/// The game's files (an empty folder until `set`).
pub fn source() -> &'static Source {
    static NONE: OnceLock<Source> = OnceLock::new();
    SOURCE.get().unwrap_or_else(|| NONE.get_or_init(|| Source::Dir(Default::default())))
}

/// `rel` under `dir` (`/` between; `dir` "" for the root).
pub fn join(dir: &str, rel: &str) -> String {
    if dir.is_empty() {
        rel.to_string()
    } else if rel.is_empty() {
        dir.to_string()
    } else {
        format!("{dir}/{rel}")
    }
}
