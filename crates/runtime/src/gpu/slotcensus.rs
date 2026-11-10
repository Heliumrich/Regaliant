//! Diagnosis (`DS_SLOT_CENSUS=1`): how a stage's buffer bindings repeat.
//! Per recording thread and (program, resource) the last bound range is
//! kept; each binding is counted as the same range as that program's last
//! binding of the resource on this thread, or a changed one, and a changed
//! one by where it lies against the thread's previous changed range (right
//! after it: the title's frame allocator handing out consecutive per-draw
//! data) and whether it lies in the GPU arena. Sizes the CPU side's design:
//! slots that repeat could skip their placement, consecutive changed ones
//! could share one.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

pub fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_SLOT_CENSUS").ok().as_deref() == Some("1"))
}

/// Per executor class (0 the graphics CP, 1 its workers, 2 compute queues):
/// [bindings, same range, changed, changed right after the thread's previous
/// changed range (within 4 KiB), changed within 64 KiB after it, changed in
/// the GPU arena, same in the arena, changed bytes, first sight of the slot].
static N: [[AtomicU64; 9]; 3] = [const { [const { AtomicU64::new(0) }; 9] }; 3];

thread_local! {
    static LAST: RefCell<(HashMap<(u64, u32), (u64, u64)>, (u64, u64))> = RefCell::new((HashMap::new(), (0, 0)));
}

/// A buffer binding of resource `i` of program `prog`: [base, base+size).
pub fn note(class: usize, prog: u64, i: u32, base: u64, size: u64) {
    let c = &N[class.min(2)];
    c[0].fetch_add(1, Ordering::Relaxed);
    let arena = crate::engine::arena_owner();
    let in_arena = arena != 0 && crate::engine::owner(base) == arena;
    LAST.with(|l| {
        let mut l = l.borrow_mut();
        let (map, prev) = &mut *l;
        if map.len() > 1 << 16 {
            map.clear();
        }
        match map.insert((prog, i), (base, size)) {
            Some(old) if old == (base, size) => {
                c[1].fetch_add(1, Ordering::Relaxed);
                if in_arena {
                    c[6].fetch_add(1, Ordering::Relaxed);
                }
                return;
            }
            Some(_) => {}
            None => {
                c[8].fetch_add(1, Ordering::Relaxed);
            }
        }
        c[2].fetch_add(1, Ordering::Relaxed);
        c[7].fetch_add(size, Ordering::Relaxed);
        if in_arena {
            c[5].fetch_add(1, Ordering::Relaxed);
        }
        let end = prev.0 + prev.1;
        if prev.1 != 0 && base >= end {
            if base - end <= 4096 {
                c[3].fetch_add(1, Ordering::Relaxed);
            }
            if base - end <= 65536 {
                c[4].fetch_add(1, Ordering::Relaxed);
            }
        }
        *prev = (base, size);
    });
}

pub fn line() -> String {
    if !on() {
        return String::new();
    }
    let names = ["graphics CP", "its workers", "compute queues"];
    let parts: Vec<String> = N
        .iter()
        .zip(names)
        .map(|(c, n)| {
            let v: Vec<u64> = c.iter().map(|x| x.load(Ordering::Relaxed)).collect();
            let pc = |x: u64| 100.0 * x as f64 / v[0].max(1) as f64;
            format!(
                "{n}: {} bindings, the same range as the program's last {:.1}% ({:.1}% in the arena), changed {:.1}% ({:.1}% in the arena; first sight {:.1}%; right after the thread's previous changed range {:.1}%, within 64 KiB {:.1}%; {:.0} bytes each)",
                v[0],
                pc(v[1]),
                pc(v[6]),
                pc(v[2]),
                pc(v[5]),
                pc(v[8]),
                pc(v[3]),
                pc(v[4]),
                v[7] as f64 / v[2].max(1) as f64
            )
        })
        .collect();
    format!("    buffer bindings by repeat (`DS_SLOT_CENSUS`): {}\n", parts.join("; "))
}

pub fn reset() {
    for c in N.iter() {
        for x in c.iter() {
            x.store(0, Ordering::Relaxed);
        }
    }
}
