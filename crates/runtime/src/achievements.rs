//! The title's trophies as PC achievements. The definitions are the title's own, read at start
//! from the game tree (never shipped): the trophy set (`sce_sys/trophy2/trophy00.ucp`: names,
//! details, grades, hidden flags, unlock conditions) and the Universal Data System definitions
//! (`sce_sys/uds/uds00.ucp`: statistics and the rules that turn the title's events into them).
//! The title posts its events (`sceNpUniversalDataSystemPostEvent`; trophies mostly as
//! `_UnlockTrophy` with `_trophy_id`, the collection ones as `DsCollection` counts): the rules
//! whose condition matches update their statistic (count, max, latest), and a trophy whose
//! condition holds is unlocked: logged, kept in local-data/achievements.txt (id and time) and
//! announced. The platinum follows when every other trophy is unlocked.
//!
//! Archive layout (`.ucp`, big-endian): a header (u32 entry count at 0x10), then from byte 96
//! 64-byte entries: a 32-byte name, u64 offset, u64 size.

use parking_lot::Mutex;
use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Clone, Debug)]
pub struct Trophy {
    pub id: u32,
    pub name: String,
    pub detail: String,
    /// P, G, S or B.
    pub grade: char,
    pub hidden: bool,
    /// (statistic, comparator, target).
    cond: Option<(u32, String, i64)>,
}

#[derive(Clone, Debug)]
struct Rule {
    event: String,
    /// (property, comparator, value) the event must satisfy.
    filter: Option<(String, String, String)>,
    /// The property whose value the statistic takes (count rules have none).
    input: Option<String>,
    stat: u32,
}

/// A property of a posted event.
#[derive(Clone, Debug)]
pub enum Value {
    Int(i64),
    Str(String),
}

struct State {
    trophies: Vec<Trophy>,
    rules: Vec<Rule>,
    /// Aggregation per statistic ("count", "max", ...).
    aggregation: HashMap<u32, String>,
    stats: HashMap<u32, i64>,
    unlocked: HashMap<u32, u64>,
    path: PathBuf,
    /// Unlocks to announce (the popup takes them).
    news: Vec<u32>,
}

static STATE: Mutex<Option<State>> = Mutex::new(None);

fn archive(d: &[u8]) -> Option<HashMap<String, Vec<u8>>> {
    let n = u32::from_be_bytes(d.get(0x10..0x14)?.try_into().ok()?) as usize;
    let mut out = HashMap::new();
    for j in 0..n.min(4096) {
        let e = d.get(96 + 64 * j..96 + 64 * (j + 1))?;
        let name: Vec<u8> = e[..32].iter().copied().take_while(|&c| c != 0).collect();
        let off = u64::from_be_bytes(e[32..40].try_into().ok()?) as usize;
        let size = u64::from_be_bytes(e[40..48].try_into().ok()?) as usize;
        if let Some(b) = d.get(off..off + size) {
            out.insert(String::from_utf8_lossy(&name).into_owned(), b.to_vec());
        }
    }
    Some(out)
}

fn json(files: &HashMap<String, Vec<u8>>, name: &str) -> Option<serde_json::Value> {
    serde_json::from_slice(files.get(name)?).ok()
}

fn num(v: &serde_json::Value) -> Option<i64> {
    v.as_i64().or_else(|| v.as_str()?.parse().ok())
}

/// Reads the definitions from the game's files (its folder or archive) and the unlocks kept so
/// far; `lang` the trophy text's language (e.g. "en-GB"; en-US when the set has none).
pub fn init(src: &dsconfig::zar::Source, keep: PathBuf, lang: &str) {
    let built = (|| -> Result<State, String> {
        let t = src.read("sce_sys/trophy2/trophy00.ucp").ok().and_then(|d| archive(&d)).ok_or("no trophy set")?;
        let conf = json(&t, "tropconf.json").ok_or("no tropconf.json")?;
        let meta = json(&t, &format!("tropmeta_{lang}.json")).or_else(|| json(&t, "tropmeta_en-US.json")).ok_or("no trophy text")?;
        let mut text = HashMap::new();
        for m in meta["metadata"]["trophyMetadata"].as_array().into_iter().flatten() {
            let id = num(&m["id"]).unwrap_or(-1) as u32;
            text.insert(id, (m["name"].as_str().unwrap_or("").to_string(), m["detail"].as_str().unwrap_or("").to_string()));
        }
        let mut trophies = Vec::new();
        for tr in conf["trophies"].as_array().ok_or("no trophies")? {
            let id = num(&tr["id"]).ok_or("trophy without id")? as u32;
            let cond = tr.get("unlockCondition").and_then(|c| Some((num(&c["udsStatId"])? as u32, c["comparator"].as_str()?.to_string(), num(&c["targetValue"])?)));
            let (name, detail) = text.get(&id).cloned().unwrap_or_default();
            trophies.push(Trophy { id, name, detail, grade: tr["grade"].as_str().and_then(|g| g.chars().next()).unwrap_or('B'), hidden: tr["hidden"].as_bool().unwrap_or(false), cond });
        }
        let u = src.read("sce_sys/uds/uds00.ucp").ok().and_then(|d| archive(&d)).ok_or("no UDS definitions")?;
        let stats = json(&u, "stats_definition.json").ok_or("no stats_definition.json")?;
        let ex = json(&u, "stats_extraction.json").ok_or("no stats_extraction.json")?;
        let mut aggregation = HashMap::new();
        for s in stats["statDefinitionArray"].as_array().into_iter().flatten() {
            if let (Some(id), Some(a)) = (num(&s["statId"]), s["aggregation"].as_str()) {
                aggregation.insert(id as u32, a.to_string());
            }
        }
        // The rules trophies depend on (the rest feed the console's activity statistics).
        let wanted: std::collections::HashSet<u32> = trophies.iter().filter_map(|t| t.cond.as_ref().map(|c| c.0)).collect();
        let mut rules = Vec::new();
        for r in ex["statsExtractionRuleArray"].as_array().into_iter().flatten() {
            let Some(stat) = num(&r["action"]["output"]["statId"]) else { continue };
            if !wanted.contains(&(stat as u32)) {
                continue;
            }
            let Some(event) = r["condition"]["eventName"].as_str() else { continue };
            let p = &r["condition"]["property"];
            let filter = p["path"].as_str().map(|path| (path.to_string(), p["comparator"].as_str().unwrap_or("==").to_string(), p["value"].as_str().map(str::to_string).unwrap_or_else(|| p["value"].to_string())));
            rules.push(Rule { event: event.to_string(), filter, input: r["action"]["input"].as_str().map(str::to_string), stat: stat as u32 });
        }
        let mut unlocked = HashMap::new();
        let mut stats_now = HashMap::new();
        if let Ok(s) = std::fs::read_to_string(&keep) {
            for l in s.lines().map(str::trim).filter(|l| !l.starts_with('#')) {
                let w: Vec<&str> = l.split_whitespace().collect();
                match w.as_slice() {
                    ["trophy", id, t] => {
                        if let (Ok(id), Ok(t)) = (id.parse(), t.parse()) {
                            unlocked.insert(id, t);
                        }
                    }
                    ["stat", id, v] => {
                        if let (Ok(id), Ok(v)) = (id.parse(), v.parse()) {
                            stats_now.insert(id, v);
                        }
                    }
                    _ => {}
                }
            }
        }
        Ok(State { trophies, rules, aggregation, stats: stats_now, unlocked, path: keep, news: Vec::new() })
    })();
    match built {
        Ok(s) => {
            crate::info!("achievements: {} trophies ({} unlocked), {} event rules ({lang})", s.trophies.len(), s.unlocked.len(), s.rules.len());
            *STATE.lock() = Some(s);
        }
        Err(e) => crate::warn!("achievements: none ({e})"),
    }
}

fn compare(a: i64, op: &str, b: i64) -> bool {
    match op {
        "==" | "eq" => a == b,
        "!=" | "ne" => a != b,
        ">" | "gt" => a > b,
        ">=" | "ge" => a >= b,
        "<" | "lt" => a < b,
        "<=" | "le" => a <= b,
        _ => false,
    }
}

fn prop<'a>(props: &'a [(String, Value)], path: &str) -> Option<&'a Value> {
    let key = path.strip_prefix("$.").unwrap_or(path);
    props.iter().find(|(k, _)| k == key).map(|(_, v)| v)
}

/// An event the title posted (its name and top-level properties).
pub fn event(name: &str, props: &[(String, Value)]) {
    let mut g = STATE.lock();
    let Some(s) = g.as_mut() else { return };
    let mut changed = false;
    for r in s.rules.iter().filter(|r| r.event == name) {
        if let Some((path, op, want)) = &r.filter {
            let ok = match prop(props, path) {
                Some(Value::Int(v)) => want.parse::<i64>().is_ok_and(|w| compare(*v, op, w)),
                Some(Value::Str(v)) => match op.as_str() {
                    "==" => v == want,
                    "!=" => v != want,
                    _ => false,
                },
                None => false,
            };
            if !ok {
                continue;
            }
        }
        let cur = s.stats.get(&r.stat).copied().unwrap_or(0);
        let next = match (s.aggregation.get(&r.stat).map(String::as_str), r.input.as_deref().and_then(|p| prop(props, p))) {
            (Some("count"), _) | (_, None) => cur + 1,
            (Some("max"), Some(Value::Int(v))) => cur.max(*v),
            (_, Some(Value::Int(v))) => *v,
            _ => cur,
        };
        if next != cur {
            s.stats.insert(r.stat, next);
            changed = true;
        }
    }
    if !changed {
        return;
    }
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let mut new = Vec::new();
    for t in &s.trophies {
        if s.unlocked.contains_key(&t.id) {
            continue;
        }
        if let Some((stat, op, target)) = &t.cond {
            if compare(s.stats.get(stat).copied().unwrap_or(0), op, *target) {
                new.push(t.id);
            }
        }
    }
    for id in &new {
        s.unlocked.insert(*id, now);
    }
    // The platinum: every other trophy unlocked.
    if let Some(p) = s.trophies.iter().find(|t| t.grade == 'P') {
        if !s.unlocked.contains_key(&p.id) && s.trophies.iter().filter(|t| t.grade != 'P').all(|t| s.unlocked.contains_key(&t.id)) {
            s.unlocked.insert(p.id, now);
            new.push(p.id);
        }
    }
    for id in &new {
        if let Some(t) = s.trophies.iter().find(|t| t.id == *id) {
            crate::info!("achievements: unlocked \"{}\" ({}): {}", t.name, t.grade, t.detail);
        }
        s.news.push(*id);
    }
    let mut text = String::from("# Achievements (the title's trophies) and the statistics they depend on; written by the port\n");
    let mut u: Vec<_> = s.unlocked.iter().collect();
    u.sort();
    for (id, t) in u {
        text.push_str(&format!("trophy {id} {t}\n"));
    }
    let mut st: Vec<_> = s.stats.iter().collect();
    st.sort();
    for (id, v) in st {
        text.push_str(&format!("stat {id} {v}\n"));
    }
    if let Err(e) = std::fs::write(&s.path, text) {
        crate::warn!("achievements: write {}: {e}", s.path.display());
    }
}

/// The trophies with their unlock times (for the menu's list).
pub fn list() -> Vec<(Trophy, Option<u64>)> {
    let g = STATE.lock();
    let Some(s) = g.as_ref() else { return Vec::new() };
    s.trophies.iter().map(|t| (t.clone(), s.unlocked.get(&t.id).copied())).collect()
}

/// Whether a trophy is unlocked.
pub fn unlocked(id: u32) -> bool {
    STATE.lock().as_ref().is_some_and(|s| s.unlocked.contains_key(&id))
}

/// Unlocks not announced yet (taken).
pub fn take_news() -> Vec<u32> {
    STATE.lock().as_mut().map(|s| std::mem::take(&mut s.news)).unwrap_or_default()
}

/// The trophies of a grade (P, G, S, B): (unlocked, all).
pub fn count(grade: char) -> (u32, u32) {
    let g = STATE.lock();
    let Some(s) = g.as_ref() else { return (0, 0) };
    let of: Vec<&Trophy> = s.trophies.iter().filter(|t| t.grade == grade).collect();
    (of.iter().filter(|t| s.unlocked.contains_key(&t.id)).count() as u32, of.len() as u32)
}
