//! `DS_WAIT_CENSUS=1` (diagnosis): the title's sleeps, event-queue waits and
//! flip-status polls by thread and kind (control command `waits`: counts, the
//! requested and the measured time), to find what a thread waits on between
//! frames.

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::OnceLock;

pub fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_WAIT_CENSUS").ok().as_deref() == Some("1"))
}

/// (thread, kind, argument bucket) -> [calls, measured ns].
static M: Mutex<Option<HashMap<(String, String, u64), [u64; 2]>>> = Mutex::new(None);

/// A wait of `kind` with its argument (a sleep's or a timeout's microseconds;
/// 0: none) that took `ns`.
pub fn note(kind: &str, arg: u64, ns: u64) {
    // Threads that differ in a number are one row (the job workers).
    let thread: String = crate::log::thread_name().chars().map(|c| if c.is_ascii_digit() { '#' } else { c }).collect();
    // Arguments by power of two above 16 us.
    let bucket = if arg <= 16 { arg } else { arg.next_power_of_two() };
    let mut g = M.lock();
    let e = g.get_or_insert_with(HashMap::new).entry((thread, kind.to_string(), bucket)).or_default();
    e[0] += 1;
    e[1] += ns;
}

/// The census since the last report, largest measured time first.
pub fn report() -> String {
    let m = M.lock().take().unwrap_or_default();
    let mut v: Vec<_> = m.into_iter().collect();
    v.sort_by(|a, b| b.1[1].cmp(&a.1[1]).then(b.1[0].cmp(&a.1[0])));
    let mut s = String::from("    waits by thread and kind (`DS_WAIT_CENSUS`; argument in us, by power of two):\n");
    for ((thread, kind, arg), [n, ns]) in v.into_iter().take(40) {
        s.push_str(&format!("      {thread}: {kind} (up to {arg} us): {n} calls, {:.1} ms in all, {:.0} us each\n", ns as f64 / 1e6, ns as f64 / n.max(1) as f64 / 1e3));
    }
    s
}

/// Times a wait from here to its drop (nothing when the census is off).
pub struct Scope(Option<(std::time::Instant, String, u64)>);

pub fn scope(kind: impl FnOnce() -> String, arg: u64) -> Scope {
    Scope(on().then(|| (std::time::Instant::now(), kind(), arg)))
}

impl Drop for Scope {
    fn drop(&mut self) {
        if let Some((t, kind, arg)) = self.0.take() {
            note(&kind, arg, t.elapsed().as_nanos() as u64);
        }
    }
}
