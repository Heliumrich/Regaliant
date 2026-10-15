//! Contention of the GPU front end's global locks (status line): every
//! recording thread takes them per buffer binding. An acquisition that has
//! to wait is counted with the time it waited; one that does not costs a
//! `try_lock`.

use parking_lot::{Mutex, MutexGuard};
use std::sync::atomic::{AtomicU64, Ordering};

pub struct LockStat {
    name: &'static str,
    waits: AtomicU64,
    ns: AtomicU64,
}

impl LockStat {
    pub const fn new(name: &'static str) -> LockStat {
        LockStat { name, waits: AtomicU64::new(0), ns: AtomicU64::new(0) }
    }

    /// `m.lock()`, counting the wait when another thread holds it.
    #[inline]
    pub fn lock<'a, T>(&self, m: &'a Mutex<T>) -> MutexGuard<'a, T> {
        if let Some(g) = m.try_lock() {
            return g;
        }
        let t = std::time::Instant::now();
        let g = m.lock();
        self.waits.fetch_add(1, Ordering::Relaxed);
        self.ns.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        g
    }

    fn part(&self) -> String {
        format!("{} {} ({} ms)", self.name, self.waits.load(Ordering::Relaxed), self.ns.load(Ordering::Relaxed) / 1_000_000)
    }

    fn reset(&self) {
        self.waits.store(0, Ordering::Relaxed);
        self.ns.store(0, Ordering::Relaxed);
    }
}

/// The batch-start copy cache (`precache`).
pub static COPY_CACHE: LockStat = LockStat::new("copy cache");
/// VRAM mirrors of GPU-written buffers (`mirror`).
pub static MIRRORS: LockStat = LockStat::new("mirrors");
/// Video-memory homes (`homes`).
pub static HOMES: LockStat = LockStat::new("homes");

const ALL: [&LockStat; 3] = [&COPY_CACHE, &MIRRORS, &HOMES];

/// Clears the counts (`profile reset`).
pub fn reset_stats() {
    for s in ALL {
        s.reset();
    }
}

/// One status line.
pub fn status() -> String {
    let parts: Vec<String> = ALL.iter().map(|s| s.part()).collect();
    format!("    contended global locks (waits, time waited): {}\n", parts.join(", "))
}
