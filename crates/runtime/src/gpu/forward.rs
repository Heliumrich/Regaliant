//! Forwarded copies (`tune forward_copies`). The title fills its per-view
//! constant blocks (the graphics queue's 0x1008880010 at the courtyard gate:
//! a 4x4 matrix at +0xc0, image descriptors at +0xa00..+0xc00) with a small
//! compute kernel that copies CPU-staged words, `if (i < cb[0]) dst[i] =
//! src[i % cb[1]]` over 32-bit elements (cs 0x1008881200 in 01.007.000).
//! Draws then resolve descriptors and constants from the block, and the
//! command processor waited for the copy to complete on the GPU before each
//! draw that read a freshly copied block (~1 ms each, ~12 ms per frame at the
//! gate: FINDINGS.md 2026-10-14). The copy's result is known when it is
//! recorded (its control block and source are written by the CPU): later
//! reads of the destination by the graphics queue's command processor and
//! its parallel workers (stage preparations and their checks) take these
//! words instead of waiting. The copy itself still runs on the GPU in order.
//! An entry ends when other GPU work or a CP write touches its destination
//! (later in the stream: the words would be stale), when a reader finds a
//! word that is neither the destination's old word (the copy has not run)
//! nor the copied one (it has): the title's CPU rewrote it, or once its
//! batch completed: memory then holds the words, which are compared
//! (`STATS`); a destination whose copy did not land is not forwarded again.

use parking_lot::{Mutex, RwLock};
use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use super::shaders::{Prepared, Resolved};

/// The words a copy leaves at `base` as the command processor computed them.
struct Entry {
    id: u64,
    base: u64,
    words: Arc<[u32]>,
    /// The batch that performs the copy (0 while it is being recorded).
    seq: u64,
    program: u64,
    hash: u64,
    /// What it was computed from: the source's words and the control block.
    from: Inputs,
    /// The destination's words when the copy was recorded.
    old: Arc<[u32]>,
}

/// A copy's inputs as read when it was recorded.
#[derive(Clone)]
pub struct Inputs {
    src: u64,
    stride: u64,
    src_words: Vec<u32>,
    cb: u64,
    count: u64,
    modulus: u64,
}

static ENTRIES: RwLock<Vec<Entry>> = RwLock::new(Vec::new());
static COUNT: AtomicUsize = AtomicUsize::new(0);
static GEN: AtomicU64 = AtomicU64::new(1);
static NEXT_ID: AtomicU64 = AtomicU64::new(1);
/// At most this many entries live (the oldest goes first).
const MAX_ENTRIES: usize = 64;

/// [copies forwarded, entries ended by a later write, entries retired after
/// their batch, words served, stage reads that needed no wait thanks to
/// them, words compared equal after the copy ran, words still as before the
/// copy then (it did not land), copies not forwarded (inputs GPU work still
/// writes, odd formats or sizes), entries ended because a reader found the
/// title had rewritten the destination, words the title rewrote after the
/// copy ran].
pub static STATS: [AtomicU64; 10] = [const { AtomicU64::new(0) }; 10];

/// Resource indices of a recognized copy kernel.
#[derive(Clone, Copy)]
struct Roles {
    src: usize,
    dst: usize,
    cb: usize,
}

/// Recognition per program code hash (Some: the copy kernel).
static PROGRAMS: Mutex<Option<HashMap<u64, Option<Roles>>>> = Mutex::new(None);
/// (program hash, destination) pairs whose forwarded words once differed
/// from memory: not forwarded again.
static DISTRUSTED: Mutex<Option<HashSet<(u64, u64)>>> = Mutex::new(None);

thread_local! {
    /// This thread reads guest memory for the graphics queue.
    static HERE: Cell<bool> = const { Cell::new(false) };
    /// This thread's copy of the entries: (generation, [(base, bytes,
    /// copied words, old words)]).
    static LOCAL: RefCell<(u64, Vec<LocalEntry>)> = const { RefCell::new((0, Vec::new())) };
}

/// The calling thread records for the graphics queue (its command processor
/// or one of its parallel workers): its reads may take forwarded words.
pub fn enable_here() {
    HERE.with(|h| h.set(true));
}

/// Whether reads on this thread may take forwarded words now (cheap).
#[inline]
pub fn active() -> bool {
    COUNT.load(Ordering::Acquire) != 0 && HERE.with(|h| h.get())
}

type LocalEntry = (u64, u64, Arc<[u32]>, Arc<[u32]>);

fn with_local<R>(f: impl FnOnce(&[LocalEntry]) -> R) -> R {
    let generation = GEN.load(Ordering::Acquire);
    LOCAL.with(|l| {
        let mut l = l.borrow_mut();
        if l.0 != generation {
            let g = ENTRIES.read();
            l.1 = g.iter().map(|e| (e.base, 4 * e.words.len() as u64, e.words.clone(), e.old.clone())).collect();
            l.0 = generation;
        }
        f(&l.1)
    })
}

/// The entry at `base` is stale: a reader found a destination word that is
/// neither the old nor the copied one (the title's CPU rewrote it).
fn drop_stale(base: u64) {
    let mut g = ENTRIES.write();
    let before = g.len();
    g.retain(|e| e.base != base);
    if g.len() != before {
        STATS[8].fetch_add(1, Ordering::Relaxed);
        COUNT.store(g.len(), Ordering::Release);
        GEN.fetch_add(1, Ordering::AcqRel);
    }
}

/// The word a reader takes at `addr` (4-byte aligned) given `mem`, what
/// guest memory holds there: the copied word while memory still holds the
/// destination's old word (the copy has not run) or that word itself.
#[inline]
pub fn word_or(addr: u64, mem: u32) -> u32 {
    if !active() {
        return mem;
    }
    let hit = with_local(|v| {
        v.iter().find(|e| e.0 <= addr && addr < e.0 + e.1).map(|e| {
            let k = ((addr - e.0) / 4) as usize;
            (e.0, e.2[k], e.3.get(k).copied())
        })
    });
    match hit {
        Some((_, w, old)) if mem == w || Some(mem) == old => {
            STATS[3].fetch_add(1, Ordering::Relaxed);
            w
        }
        Some((base, ..)) => {
            drop_stale(base);
            mem
        }
        None => mem,
    }
}

/// Replaces the words of `out` (read at `addr` from guest memory) that
/// entries cover, entry by entry: all of an entry's words in range, or none
/// when one of them shows the title rewrote the destination (the entry ends).
pub fn overlay(addr: u64, out: &mut [u32]) {
    if !active() {
        return;
    }
    let end = addr + 4 * out.len() as u64;
    let mut stale = Vec::new();
    let n = with_local(|v| {
        let mut n = 0u64;
        for (base, bytes, words, old) in v {
            let (lo, hi) = (addr.max(*base), end.min(base + bytes));
            if lo >= hi {
                continue;
            }
            let ok = (lo..hi).step_by(4).all(|a| {
                let (i, k) = (((a - addr) / 4) as usize, ((a - base) / 4) as usize);
                out[i] == words[k] || old.get(k) == Some(&out[i])
            });
            if !ok {
                stale.push(*base);
                continue;
            }
            for a in (lo..hi).step_by(4) {
                out[((a - addr) / 4) as usize] = words[((a - base) / 4) as usize];
                n += 1;
            }
        }
        n
    });
    for base in stale {
        drop_stale(base);
    }
    if n > 0 {
        STATS[3].fetch_add(n, Ordering::Relaxed);
    }
}

/// `reads` without the parts entries cover (the reader took the forwarded
/// words there, so GPU work writing them need not complete first). None:
/// nothing covered (use `reads` as they are).
pub fn uncovered(reads: &[(u64, u64)]) -> Option<Vec<(u64, u64)>> {
    if !active() {
        return None;
    }
    with_local(|v| {
        if !reads.iter().any(|&(a, n)| v.iter().any(|e| e.0 < a + n && a < e.0 + e.1)) {
            return None;
        }
        let mut out = Vec::with_capacity(reads.len());
        let mut covered = 0u64;
        for &(a, n) in reads {
            // Pieces of [a, a + n) outside every entry.
            let mut pieces = vec![(a, a + n)];
            for (base, bytes, _, _) in v {
                let (lo, hi) = (*base, base + bytes);
                let mut next = Vec::with_capacity(pieces.len() + 1);
                for (s, e) in pieces {
                    if e <= lo || hi <= s {
                        next.push((s, e));
                        continue;
                    }
                    if s < lo {
                        next.push((s, lo));
                    }
                    if hi < e {
                        next.push((hi, e));
                    }
                }
                pieces = next;
            }
            if pieces.is_empty() {
                covered += 1;
            }
            out.extend(pieces.into_iter().map(|(s, e)| (s, e - s)));
        }
        STATS[4].fetch_add(covered, Ordering::Relaxed);
        Some(out)
    })
}

/// GPU work or a CP write recorded later in the stream writes [a, a + n):
/// entries it overlaps end.
pub fn written(a: u64, n: u64) {
    if COUNT.load(Ordering::Acquire) == 0 || n == 0 {
        return;
    }
    let mut g = ENTRIES.write();
    let before = g.len();
    g.retain(|e| !(e.base < a + n && a < e.base + 4 * e.words.len() as u64));
    let ended = before - g.len();
    if ended > 0 {
        STATS[1].fetch_add(ended as u64, Ordering::Relaxed);
        COUNT.store(g.len(), Ordering::Release);
        GEN.fetch_add(1, Ordering::AcqRel);
    }
}

/// The copy kernel's roles, if the program is it (cached by code hash).
fn roles(p: &Prepared) -> Option<Roles> {
    let insts = &p.found.insts;
    let key = p.code_hash;
    if let Some(r) = PROGRAMS.lock().get_or_insert_with(HashMap::new).get(&key) {
        return *r;
    }
    let r = recognize(insts, &p.found.tr.plan);
    if r.is_some() {
        crate::info!("gpu: cs {:#x} (code hash {key:#018x}) is a guarded modulo copy: its results are forwarded to later reads (`tune forward_copies`)", p.addr);
    }
    PROGRAMS.lock().get_or_insert_with(HashMap::new).insert(key, r);
    r
}

/// `if (i < cb[0]) dst[i] = src[i % cb[1]]`, as the title's compiler emits
/// it for 64-lane groups (the unsigned remainder through a reciprocal),
/// with nothing else touching memory.
fn recognize(insts: &[rdna::Inst], plan: &rdna::analysis::Plan) -> Option<Roles> {
    const SEQUENCE: [&str; 26] = [
        "v_lshl_add_u32",
        "s_buffer_load_dword",
        "v_cmpx_gt_u32",
        "s_cbranch_execz",
        "s_buffer_load_dword",
        "v_cvt_f32_u32",
        "s_sub_i32",
        "v_rcp_iflag_f32",
        "v_mul_f32",
        "v_cvt_u32_f32",
        "v_readfirstlane_b32",
        "s_mul_i32",
        "s_mul_hi_u32",
        "v_add_u32",
        "v_mul_hi_u32",
        "v_mul_lo_u32",
        "v_sub_u32",
        "v_cmp_le_u32",
        "v_subrev_u32",
        "v_cndmask_b32",
        "v_cmp_le_u32",
        "v_subrev_u32",
        "v_cndmask_b32",
        "buffer_load_format_x",
        "buffer_store_format_x",
        "s_endpgm",
    ];
    let body: Vec<&rdna::Inst> = insts.iter().filter(|i| !matches!(i.name(), Some("s_inst_prefetch" | "s_waitcnt" | "s_nop"))).collect();
    if body.len() != SEQUENCE.len() || body.iter().zip(SEQUENCE).any(|(i, s)| i.name() != Some(s)) {
        return None;
    }
    // The control block's two words at offsets 0 and 4 of one descriptor.
    let (l0, l1) = (body[1], body[4]);
    if l0.smem_sbase() != l1.smem_sbase() || l0.smem_offset() != 0 || l1.smem_offset() != 4 {
        return None;
    }
    let cb = plan.inst_res.get(&l0.pc)?.0;
    let src = plan.inst_res.get(&body[23].pc)?.0;
    let dst = plan.inst_res.get(&body[24].pc)?.0;
    (cb != src && cb != dst && src != dst && plan.resources.get(dst)?.written && !plan.resources.get(src)?.written).then_some(Roles { src, dst, cb })
}

/// Formats a copy moves unchanged: 32_UINT, 32_SINT, 32_FLOAT.
fn plain32(format: u8) -> bool {
    matches!(format, 20..=22)
}

/// The words a dispatch of `p` with `groups` (64-lane groups along X) will
/// leave at its destination, when `p` is the copy kernel and its control
/// block and source can be read now (`busy`: GPU work or a deferred CP write
/// still writes one of those ranges). (destination base, words).
pub fn compute(p: &Prepared, groups: [u32; 3], local: [u32; 3], busy: impl Fn(&[(u64, u64)]) -> bool) -> Option<(u64, Vec<u32>, Inputs)> {
    let r = roles(p)?;
    let buf = |i: usize| match p.found.resources.get(i) {
        Some(Resolved::Buffer(v)) => Some(*v),
        _ => None,
    };
    let (src, dst, cb) = (buf(r.src)?, buf(r.dst)?, buf(r.cb)?);
    if DISTRUSTED.lock().as_ref().is_some_and(|d| d.contains(&(p.code_hash, dst.base))) {
        return None;
    }
    let refuse = || {
        STATS[7].fetch_add(1, Ordering::Relaxed);
        None
    };
    if local != [64, 1, 1] || groups[1] != 1 || groups[2] != 1 || !plain32(src.format) || !plain32(dst.format) || dst.stride != 4 || src.stride < 4 || src.add_tid || dst.add_tid {
        return refuse();
    }
    if !super::cp::guest_readable(cb.base, 8) {
        return refuse();
    }
    let n = unsafe { std::ptr::read_volatile(cb.base as *const u32) } as u64;
    let m = unsafe { std::ptr::read_volatile((cb.base + 4) as *const u32) } as u64;
    let count = n.min(groups[0] as u64 * 64).min(dst.num_records as u64);
    let in_src = m.min(src.num_records as u64);
    if m == 0 || count == 0 || count > 1 << 16 || in_src > 1 << 16 {
        return refuse();
    }
    let src_bytes = (in_src - 1) * src.stride as u64 + 4;
    let inputs = [(cb.base, 8), (src.base, src_bytes)];
    if busy(&inputs) || super::homes::any_stale(&inputs) || !super::cp::guest_readable(src.base, src_bytes) {
        return refuse();
    }
    let src_words: Vec<u32> = (0..in_src).map(|j| unsafe { std::ptr::read_volatile((src.base + j * src.stride as u64) as *const u32) }).collect();
    let words = (0..count)
        .map(|i| {
            let j = i % m;
            if j < src.num_records as u64 { src_words[j as usize] } else { 0 }
        })
        .collect();
    Some((dst.base, words, Inputs { src: src.base, stride: src.stride as u64, src_words, cb: cb.base, count: n, modulus: m }))
}

/// Starts forwarding `words` at `base` (written by the dispatch of `program`
/// just recorded, after its own writes were noted); returns the entry's id.
pub fn add(base: u64, words: Vec<u32>, program: u64, hash: u64, from: Inputs) -> u64 {
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    let n = words.len() as u64;
    let old: Arc<[u32]> = if super::cp::guest_readable(base, 4 * n) {
        (0..n).map(|k| unsafe { std::ptr::read_volatile((base + 4 * k) as *const u32) }).collect()
    } else {
        Arc::from(Vec::new())
    };
    let mut g = ENTRIES.write();
    let end = base + 4 * words.len() as u64;
    g.retain(|e| !(e.base < end && base < e.base + 4 * e.words.len() as u64));
    if g.len() >= MAX_ENTRIES {
        g.remove(0);
    }
    g.push(Entry { id, base, words: words.into(), seq: 0, program, hash, from, old });
    COUNT.store(g.len(), Ordering::Release);
    GEN.fetch_add(1, Ordering::AcqRel);
    STATS[0].fetch_add(1, Ordering::Relaxed);
    id
}

/// The batch holding these entries' copies was submitted as `seq`.
pub fn submitted(ids: &[u64], seq: u64) {
    if ids.is_empty() {
        return;
    }
    let mut g = ENTRIES.write();
    for e in g.iter_mut() {
        if ids.contains(&e.id) {
            e.seq = seq;
        }
    }
}

/// Batches up to `done` completed: their entries' words are compared with
/// memory (the GPU copied them by now) and the entries end.
pub fn completed(done: u64) {
    if COUNT.load(Ordering::Acquire) == 0 {
        return;
    }
    let due: Vec<Entry> = {
        let mut g = ENTRIES.write();
        if !g.iter().any(|e| e.seq != 0 && e.seq <= done) {
            return;
        }
        let (due, keep): (Vec<Entry>, Vec<Entry>) = std::mem::take(&mut *g).into_iter().partition(|e| e.seq != 0 && e.seq <= done);
        *g = keep;
        COUNT.store(g.len(), Ordering::Release);
        GEN.fetch_add(1, Ordering::AcqRel);
        due
    };
    for e in due {
        STATS[2].fetch_add(1, Ordering::Relaxed);
        if !super::cp::guest_readable(e.base, 4 * e.words.len() as u64) {
            continue;
        }
        let mut differ = 0u64;
        let mut rewritten = 0u64;
        let mut first = None;
        for (k, &w) in e.words.iter().enumerate() {
            let now = unsafe { std::ptr::read_volatile((e.base + 4 * k as u64) as *const u32) };
            if now == w {
                continue;
            }
            if e.old.get(k) == Some(&now) && e.old[k] != w {
                differ += 1;
                first.get_or_insert((k, w, now));
            } else {
                rewritten += 1;
            }
        }
        STATS[9].fetch_add(rewritten, Ordering::Relaxed);
        STATS[5].fetch_add(e.words.len() as u64 - differ - rewritten, Ordering::Relaxed);
        if let Some((k, w, now)) = first {
            STATS[6].fetch_add(differ, Ordering::Relaxed);
            static LOGGED: AtomicU64 = AtomicU64::new(0);
            if LOGGED.fetch_add(1, Ordering::Relaxed) < 20 {
                // What changed since the copy was recorded: its source, its
                // control block, or neither (then something else wrote the
                // destination, or the GPU copied other words).
                let f = &e.from;
                let src_now: Vec<u32> = if super::cp::guest_readable(f.src, (f.src_words.len() as u64).saturating_sub(1) * f.stride + 4) {
                    (0..f.src_words.len() as u64).map(|j| unsafe { std::ptr::read_volatile((f.src + j * f.stride) as *const u32) }).collect()
                } else {
                    Vec::new()
                };
                let src_changed = src_now.iter().zip(&f.src_words).filter(|(a, b)| a != b).count();
                let cb_now = if super::cp::guest_readable(f.cb, 8) { unsafe { [std::ptr::read_volatile(f.cb as *const u32), std::ptr::read_volatile((f.cb + 4) as *const u32)] } } else { [0, 0] };
                crate::warn!(
                    "gpu: forwarded copy to {:#x} (cs {:#x}, batch {}): {differ} of {} words still as before the copy after it ran, first dword {k}: forwarded {w:#x}, memory {now:#x}; source {:#x} ({} words, stride {}): {src_changed} words changed since recorded{}; control block {:#x}: count {} modulus {} when recorded, {} {} now; last CP write to the source: {}; this destination is no longer forwarded",
                    e.base,
                    e.program,
                    e.seq,
                    e.words.len(),
                    f.src,
                    f.src_words.len(),
                    f.stride,
                    if src_now.is_empty() { " (unreadable now)" } else { "" },
                    f.cb,
                    f.count,
                    f.modulus,
                    cb_now[0],
                    cb_now[1],
                    super::cp::last_write_info(f.src).unwrap_or_else(|| "none".into())
                );
            }
            DISTRUSTED.lock().get_or_insert_with(HashSet::new).insert((e.hash, e.base));
        }
    }
}

/// The status line ("" when nothing was forwarded).
pub fn status_line() -> String {
    let s: Vec<u64> = STATS.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    if s[0] == 0 && s[7] == 0 {
        return String::new();
    }
    format!(
        "    forwarded copies: {} made ({} not: inputs still written or odd shapes), {} ended by later writes, {} by the title's rewrites, {} retired after their batch; {} words served, {} stage reads with no wait for the copy; after the copies ran {} words equal, {} as before the copy (it did not land), {} rewritten since\n",
        s[0], s[7], s[1], s[8], s[2], s[3], s[4], s[5], s[6], s[9]
    )
}
