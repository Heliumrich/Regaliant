//! File traffic of the title by API and file extension (diagnostics: what it
//! loads and streams, e.g. texture chunks). The first few events of each
//! (API, extension, kind) are logged by the callers; `status()` joins the
//! status report.

use parking_lot::Mutex;
use std::collections::HashMap;

#[derive(Clone, Copy)]
pub enum What {
    /// A file opened (kernel) or a path resolved to an id (APR).
    Open,
    Read,
    /// An open or resolve that failed.
    Fail,
}

/// (API, extension) -> [opened, reads, bytes read, failed].
static STATS: Mutex<Option<HashMap<(&'static str, String), [u64; 4]>>> = Mutex::new(None);

/// Events of one (API, extension, kind) reported to the caller for logging.
const LOG_FIRST: u64 = 8;

fn extension(path: &str) -> String {
    let name = path.rsplit('/').next().unwrap_or(path);
    match name.rfind('.') {
        Some(i) if i > 0 => name[i..].to_ascii_lowercase(),
        _ => "(none)".into(),
    }
}

/// `DS_TRACE_IO=<substring>`: every event on a path containing it is
/// reported for logging (e.g. `.ctxc` for texture chunks).
fn traced(path: &str) -> bool {
    static PAT: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    PAT.get_or_init(|| std::env::var("DS_TRACE_IO").ok().filter(|p| !p.is_empty())).as_deref().is_some_and(|p| path.contains(p))
}

/// Counts an event; true for the first `LOG_FIRST` of its API, extension
/// and kind and for traced paths (the caller logs those).
pub fn note(api: &'static str, path: &str, what: What, bytes: u64) -> bool {
    let trace = traced(path);
    let mut g = STATS.lock();
    let e = g.get_or_insert_with(HashMap::new).entry((api, extension(path))).or_insert([0; 4]);
    let k = match what {
        What::Open => 0,
        What::Read => {
            e[2] += bytes;
            1
        }
        What::Fail => 3,
    };
    e[k] += 1;
    e[k] <= LOG_FIRST || trace
}

/// One line per API with its busiest extensions (by bytes, then opens).
pub fn status() -> String {
    let g = STATS.lock();
    let Some(m) = g.as_ref() else { return String::new() };
    let mut s = String::new();
    for api in ["apr", "kernel"] {
        let mut v: Vec<(&String, &[u64; 4])> = m.iter().filter(|((a, _), _)| *a == api).map(|((_, e), x)| (e, x)).collect();
        if v.is_empty() {
            continue;
        }
        v.sort_by(|a, b| b.1[2].cmp(&a.1[2]).then(b.1[0].cmp(&a.1[0])).then(a.0.cmp(b.0)));
        let list: Vec<String> = v
            .iter()
            .take(16)
            .map(|(e, x)| {
                let fail = if x[3] > 0 { format!(", {} failed", x[3]) } else { String::new() };
                format!("{e} {} opened, {} reads {:.1} MiB{fail}", x[0], x[1], x[2] as f64 / (1u64 << 20) as f64)
            })
            .collect();
        s.push_str(&format!("    file i/o ({api}): {}\n", list.join("; ")));
    }
    s
}
