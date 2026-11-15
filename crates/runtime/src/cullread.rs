//! The title's CPU reads of its GPU culling results. Its job threads read the index counts the
//! index-culling compute (`xprindexculling`) writes into the indirect draw arguments (20-byte
//! records) and drop the objects that read 0 from the depth pre-pass's visibility mask
//! (`cmp dword ptr [r10+rbx*4], 0` at eboot+0xad91de: r10 the records, rbx the record's first
//! dword, both overwritten right after).
//!
//! Each frame the GPU clears the records (`cs_memset32`) and the culls (237 dispatches a frame
//! on the graphics queue at Forlorn Outpost, recorded in parallel chunks) add the counts. The
//! title submits them earlier in the frame and reads them without a fence: the console's GPU has
//! run them by then. Ours runs behind and its write-backs land batch by batch, so a read could
//! find the clear's zeros or part of the counts, and a visible object read 0 for one frame: in
//! the intro cutscene the enclosing box missed the depth pre-pass, its colour pass (depth
//! EQUAL) drew nothing and the sky flashed (FINDINGS 2026-11-14); at Forlorn Outpost wall pieces
//! went missing for a frame, showing black or the beams behind them (the user's videos,
//! 2026-11-15). The wait (`read_wait`, 2026-11-15): a read waits until the GPU work submitted
//! so far that writes the array has completed (an inline check of two counters; the call only
//! while a writer is in flight). The hold: a blank record (the clear's state; a culled object's
//! record keeps its other words: instances, first index and instance, with a count of 0) read
//! non-zero within the last `HOLD_FLIPS` flips, in the same array, reads as non-zero (a static
//! 1; the title only tests it) instead of 0, for the culls not yet submitted at the read. Real
//! culls pass as they are (a hold of every recent zero kept culled objects visible for two
//! more flips: in the four-cutscene script ~270 a flip and the title's GPU memory ~300 MB
//! higher).
//!
//! One array is read (0x1008a9ea10) where this was measured; the table still keys entries by
//! array (a hash of r10 picks a slot of `HOLD_LEN` entries, r10's low half tags the entry), so
//! a view with arrays of its own would not hold another's objects.
//!
//! The hold is an inline insertion (`hold_code`: the read runs ~3,400 times a frame in the
//! tunnel; a hook's register save would cost more). Diagnosis (`DS_CULLREAD_DIAG`, a hook in
//! its place, DIAGNOSTICS.md): 1 the same hold, with reads, zeros and holds logged per flip and
//! array; 0 no hold; 2 the hold with held reads sampled for a refill; 3 the hold with record 0's
//! words logged at its 0 reads; 4 the hold with each read waiting for the GPU work in flight
//! that writes the array (at most 100 ms); 5 that wait without the hold; 7 as 4, and first a
//! wait until the graphics command processor has run every title submission queued before the
//! read (at most 100 ms; not while it polls a label); 8 as 4, with one in 4 reads of a set
//! record at 0 that read non-zero lately read again for up to 3 ms; 9 as 4, counting records
//! whose first-instance field changed since their last read (it is a per-frame offset: ~40%
//! change every flip); 10 as 4, with the graphics queue's backlog of title submissions not yet
//! run at the flip's reads and at its set zero reads of records seen lately.

use std::sync::atomic::{AtomicU64, Ordering};

/// Flips a non-zero read holds a record.
pub const HOLD_FLIPS: u32 = 2;
/// Records a slot covers (dword index of the first word; the title's arrays have 75,000).
const HOLD_BITS: u32 = 17;
pub const HOLD_LEN: u32 = 1 << HOLD_BITS;
/// Slots (one per array, by hash).
const SLOT_BITS: u32 = 2;
const HASH: u64 = 0x9E37_79B9_7F4A_7C15;

/// Per array slot and record (its first dword's index): the flip of its last non-zero read
/// (high half) and the array's address (low half); 0: none.
fn table() -> &'static [AtomicU64] {
    static T: std::sync::OnceLock<Box<[AtomicU64]>> = std::sync::OnceLock::new();
    T.get_or_init(|| (0..(HOLD_LEN as usize) << SLOT_BITS).map(|_| AtomicU64::new(0)).collect())
}

fn entry(r10: u64, rbx: u64) -> &'static AtomicU64 {
    let slot = r10.wrapping_mul(HASH) >> (64 - SLOT_BITS);
    &table()[((slot << HOLD_BITS) + rbx) as usize]
}

static ONE: u32 = 1;

/// The inline hold's addresses: the table, the current flip's counter and the word a held read
/// is pointed at.
fn hold_addrs() -> (u64, u64, u64) {
    (table().as_ptr() as u64, crate::hle::libs::videoout::QUEUED_FLIP.as_ptr() as u64, &ONE as *const u32 as u64)
}

/// The array the title reads (its first byte; 0 until its first read: `read_wait`).
static ARRAY: AtomicU64 = AtomicU64::new(0);
/// The title's arrays: 75,000 dwords.
const ARRAY_BYTES: u64 = 75_000 * 4;
/// The latest batch submitted that writes the array (u64::MAX until the array is known, so the
/// first reads take `read_wait`).
static ARRAY_SEQ: AtomicU64 = AtomicU64::new(u64::MAX);
/// Reads that waited for the GPU's writes to the array, their total wait (us), waits that timed
/// out (100 ms).
static READ_WAITS: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// A batch submitted with its guest writes noted (`exec`): one that writes the array is the
/// latest a read of it waits for.
#[inline]
pub fn note_submitted(seq: u64, writes: &crate::gpu::ranges::Ranges) {
    let a = ARRAY.load(Ordering::Relaxed);
    if a != 0 && writes.overlaps(a, ARRAY_BYTES) {
        ARRAY_SEQ.fetch_max(seq, Ordering::AcqRel);
        ev(b'A', seq, 0);
    }
}

/// The title's read at `r10` when the inline check found the array unknown, another array, or
/// a batch writing it not completed: waits until the GPU work submitted so far that writes the
/// array has completed (its write-backs in guest memory), at most 100 ms. The console's GPU has
/// run the culls the title submitted earlier in the frame by the time it reads them (it reads
/// without a fence); ours runs behind and its write-backs land batch by batch, so a read could
/// find the clear's zeros or part of the culls' counts: a visible object then missed the depth
/// pre-pass for a frame (a wall missing, what is behind it drawn instead: the user's 0.1.0
/// videos of 2026-11-14).
pub extern "sysv64" fn read_wait(r10: u64) {
    if ARRAY.load(Ordering::Acquire) == 0 && ARRAY.compare_exchange(0, r10, Ordering::AcqRel, Ordering::Acquire).is_ok() {
        // (Batches submitted from now on are noted; earlier ones are found by their ranges.)
        let seq = crate::gpu::exec::pending_writes(r10, ARRAY_BYTES).unwrap_or(0);
        let _ = ARRAY_SEQ.compare_exchange(u64::MAX, seq, Ordering::AcqRel, Ordering::Acquire);
    }
    let Some(seq) = crate::gpu::exec::pending_writes(r10, ARRAY_BYTES) else { return };
    if seq <= crate::gpu::exec::done_now() {
        return;
    }
    let t = std::time::Instant::now();
    let ok = crate::gpu::exec::wait_done_for(seq, std::time::Duration::from_millis(100));
    READ_WAITS[0].fetch_add(1, Ordering::Relaxed);
    READ_WAITS[1].fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);
    if !ok {
        READ_WAITS[2].fetch_add(1, Ordering::Relaxed);
    }
}

/// The title's index culls (`gpu::exec::CULL_CS`) as COMPUTE_PGM_LO values (address >> 8).
const CULL_PGM_LO: [u32; 4] = [0x1008_9937, 0x1008_994d, 0x1008_9976, 0x1008_998c];
/// The latest graphics submission of the title found to set an index cull's program (its
/// queue index; 0: none yet), and the scan's totals: [submissions, of them with culls, dwords,
/// ns].
static LATEST_CULL_SUB: AtomicU64 = AtomicU64::new(0);
static SCANS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

fn scan_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| matches!(std::env::var("DS_CULLREAD_DIAG").ok().as_deref(), Some("12")))
}

/// A graphics submission queued by the title (`gpu::submit`, its queue index): whether its
/// command buffers (and those they call, two levels) set an index cull's program is noted.
pub fn note_title_submission(index: u64, bufs: &[(u64, u32)]) {
    if !scan_on() {
        return;
    }
    let t = std::time::Instant::now();
    let mut dwords = 0u64;
    let found = bufs.iter().any(|&(a, n)| sets_cull_program(a, n, 2, &mut dwords));
    ev(b's', index, found as u64);
    if found {
        LATEST_CULL_SUB.fetch_max(index, Ordering::AcqRel);
        SCANS[1].fetch_add(1, Ordering::Relaxed);
        // The culls' per-object descriptors (their user data): the range and its hash now.
        let mut range = (u64::MAX, 0u64, 0u64);
        for &(a, n) in bufs {
            cull_user_data(a, n, &mut range);
        }
        if range.1 > range.0 && range.1 - range.0 <= 1 << 22 {
            let h = hash_range(range.0, range.1 - range.0);
            *DESC.lock() = Some(Desc { sub: index, base: range.0, end: range.1, dispatches: range.2, at_submit: h, at_run: None });
            ev(b'd', range.0, (range.1 - range.0) << 32 | range.2);
        }
    }
    SCANS[0].fetch_add(1, Ordering::Relaxed);
    SCANS[2].fetch_add(dwords, Ordering::Relaxed);
    SCANS[3].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
}

/// Mode 12: the latest culls' submission's per-object descriptors (the USER_DATA_0 pointers of its
/// dispatches): their range, dispatch count and hashes when the title queued the submission and
/// when the graphics command processor finished running it.
struct Desc {
    sub: u64,
    base: u64,
    end: u64,
    dispatches: u64,
    at_submit: u64,
    at_run: Option<u64>,
}
static DESC: parking_lot::Mutex<Option<Desc>> = parking_lot::Mutex::new(None);
/// Mode 12, at passes' first reads: [checks, descriptors changed since the submission was queued,
/// since the CP ran it, the CP had not run it yet].
static DESC_STATS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

fn hash_range(a: u64, n: u64) -> u64 {
    if !crate::gpu::cp::guest_readable(a, n) {
        return 0;
    }
    let w = unsafe { std::slice::from_raw_parts(a as *const u64, (n / 8) as usize) };
    w.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &x| (h ^ x).wrapping_mul(0x100_0000_01b3).rotate_left(5))
}

/// The graphics command processor finished running title submission `index` (`gpu::worker`): for
/// the culls' submission, its descriptors' hash then (did the title write them after queueing it?).
pub fn note_run_end(index: u64) {
    if !scan_on() {
        return;
    }
    let mut g = DESC.lock();
    if let Some(d) = g.as_mut().filter(|d| d.sub == index) {
        let h = hash_range(d.base, d.end - d.base);
        d.at_run = Some(h);
        ev(b'h', (h != d.at_submit) as u64, 0);
    }
}

/// At a pass's first read: the descriptors' hash against the one when the CP ran the submission
/// (changed: the title wrote them after the GPU may have read them).
fn check_desc_at_read() {
    let g = DESC.lock();
    if let Some(d) = g.as_ref() {
        let h = hash_range(d.base, d.end - d.base);
        let (since_submit, since_run, not_run) = (h != d.at_submit, d.at_run.is_some_and(|r| r != h), d.at_run.is_none());
        ev(b'H', d.sub, since_submit as u64 | (since_run as u64) << 1 | (not_run as u64) << 2);
        DESC_STATS[0].fetch_add(1, Ordering::Relaxed);
        DESC_STATS[1].fetch_add(since_submit as u64, Ordering::Relaxed);
        DESC_STATS[2].fetch_add(since_run as u64, Ordering::Relaxed);
        DESC_STATS[3].fetch_add(not_run as u64, Ordering::Relaxed);
    }
}

/// The culls' dispatches in the command buffer at `addr` (`size` dwords; not following calls):
/// the lowest and highest USER_DATA_0 value set before a dispatch after a cull's program was
/// loaded, and the dispatch count, into `range` (low, high + 40, count).
fn cull_user_data(addr: u64, size: u32, range: &mut (u64, u64, u64)) {
    use crate::hle::libs::agc_cmd::op;
    if size == 0 || !crate::gpu::cp::guest_readable(addr, 4 * size as u64) {
        return;
    }
    let w = unsafe { std::slice::from_raw_parts(addr as *const u32, size as usize) };
    let (mut culling, mut ud) = (false, 0u64);
    let mut i = 0usize;
    while i < w.len() {
        let h = w[i];
        if h >> 30 != 3 {
            i += 1;
            continue;
        }
        let count = ((h >> 16) & 0x3FFF) as usize;
        if count == 0x3FFF {
            i += 1;
            continue;
        }
        let n = count + 1;
        let p = |k: usize| w.get(i + 1 + k).copied().unwrap_or(0);
        match (h >> 8) & 0xFF {
            op::LOAD_SH_REG_INDEX => {
                let (a, cnt) = (p(0) as u64 | (p(1) as u64 & 0xFFFF) << 32, p(2) as u64);
                if cnt > 0 && cnt < 4096 && crate::gpu::cp::guest_readable(a & !3, 8 * cnt) {
                    for k in 0..cnt {
                        let (o, v) = unsafe { (*((a & !3) as *const u32).add(2 * k as usize), *((a & !3) as *const u32).add(2 * k as usize + 1)) };
                        if o & 0xFFFF == 0x20C {
                            culling = CULL_PGM_LO.contains(&v);
                        }
                    }
                }
            }
            op::SET_SH_REG if n >= 3 && p(0) & 0xFFFF == 0x240 => {
                ud = p(1) as u64 | (p(2) as u64 & 0xFFFF) << 32;
            }
            op::DISPATCH_DIRECT if culling && ud != 0 => {
                range.0 = range.0.min(ud);
                range.1 = range.1.max(ud + 40);
                range.2 += 1;
            }
            _ => {}
        }
        i += n + 1;
    }
}

/// Whether the command buffer at `addr` (`size` dwords) sets COMPUTE_PGM_LO to an index cull's
/// program (SET_SH_REG or a register load's pairs), following INDIRECT_BUFFER packets and the
/// HLE's private branches `depth` levels deep.
fn sets_cull_program(addr: u64, size: u32, depth: u32, dwords: &mut u64) -> bool {
    use crate::hle::libs::agc_cmd::{op, pop, DS_MAGIC};
    if size == 0 || !crate::gpu::cp::guest_readable(addr, 4 * size as u64) {
        return false;
    }
    *dwords += size as u64;
    let w = unsafe { std::slice::from_raw_parts(addr as *const u32, size as usize) };
    let mut i = 0usize;
    while i < w.len() {
        let h = w[i];
        if h >> 30 != 3 {
            i += 1;
            continue;
        }
        let count = ((h >> 16) & 0x3FFF) as usize;
        if count == 0x3FFF {
            i += 1;
            continue;
        }
        let n = count + 1;
        let p = |k: usize| w.get(i + 1 + k).copied().unwrap_or(0);
        match (h >> 8) & 0xFF {
            op::SET_SH_REG | op::SET_SH_REG_INDEX => {
                let first = p(0) & 0xFFFF;
                for k in 1..n {
                    if first + (k - 1) as u32 == 0x20C && CULL_PGM_LO.contains(&p(k)) {
                        static DUMPED: AtomicU64 = AtomicU64::new(0);
                        if DUMPED.fetch_add(1, Ordering::Relaxed) < 3 {
                            let from = i.saturating_sub(24);
                            let words: Vec<String> = w[from..(i + 60).min(w.len())].iter().map(|x| format!("{x:08x}")).collect();
                            crate::info!("cull program set in a submission at {:#x} (dword {i}): dwords from {from}: {}", addr + 4 * i as u64, words.join(" "));
                        }
                        return true;
                    }
                }
            }
            op::LOAD_SH_REG_INDEX => {
                let (a, cnt) = (p(0) as u64 | (p(1) as u64 & 0xFFFF) << 32, p(2) as u64);
                if cnt > 0 && cnt < 4096 && crate::gpu::cp::guest_readable(a & !3, 8 * cnt) {
                    for k in 0..cnt {
                        let (o, v) = unsafe { (*((a & !3) as *const u32).add(2 * k as usize), *((a & !3) as *const u32).add(2 * k as usize + 1)) };
                        if o & 0xFFFF == 0x20C && CULL_PGM_LO.contains(&v) {
                            static DUMPED: AtomicU64 = AtomicU64::new(0);
                            if DUMPED.fetch_add(1, Ordering::Relaxed) < 3 {
                                let from = i.saturating_sub(16);
                                let words: Vec<String> = w[from..(i + 64).min(w.len())].iter().map(|x| format!("{x:08x}")).collect();
                                let pairs: Vec<String> = (0..cnt.min(24)).map(|j| unsafe { format!("{:x}={:08x}", *((a & !3) as *const u32).add(2 * j as usize), *((a & !3) as *const u32).add(2 * j as usize + 1)) }).collect();
                                crate::info!("cull program loaded in a submission at {:#x} (dword {i} of {size}; pairs at {a:#x}, {cnt}: {}): dwords from {from}: {}", addr + 4 * i as u64, pairs.join(" "), words.join(" "));
                            }
                            return true;
                        }
                    }
                }
            }
            op::INDIRECT_BUFFER if depth > 0 => {
                let target = (p(0) & !3) as u64 | ((p(1) as u64 & 0xFFFF) << 32);
                let dw = p(2) & 0xF_FFFF;
                if target != 0 && sets_cull_program(target, dw, depth - 1, dwords) {
                    return true;
                }
            }
            op::NOP if depth > 0 && n >= 15 && p(0) == DS_MAGIC && p(1) == pop::BRANCH => {
                for (lo, hi, sz) in [(p(9), p(10), p(11)), (p(12), p(13), p(14))] {
                    let target = lo as u64 | (hi as u64) << 32;
                    if target != 0 && sets_cull_program(target, sz & 0x0FFF_FFFF, depth - 1, dwords) {
                        return true;
                    }
                }
            }
            _ => {}
        }
        i += n + 1;
    }
    false
}

/// Mode 12, per flip: [reads while the latest submission setting a cull's program was not yet
/// run by the graphics command processor, reads after, the most it was behind (submissions),
/// reads while index culls recorded were not yet submitted to the GPU, the most such culls].
static BEHIND: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];

/// Index culls recorded by the command processors (a ticket each, in recording order) and the
/// latest ticket of a batch submitted to the GPU (`gpu::exec`: a batch carries its latest cull's
/// ticket from recording, through a deferred executor's closed list, to its submission).
static CULLS_RECORDED: AtomicU64 = AtomicU64::new(0);
static CULLS_SUBMITTED: AtomicU64 = AtomicU64::new(0);

/// The latest ticket of a recorded index cull, and of a recorded clear of the array (the same
/// counter: the later one is the array's latest recorded operation).
static LAST_CULL_T: AtomicU64 = AtomicU64::new(0);
static LAST_CLEAR_T: AtomicU64 = AtomicU64::new(0);
/// The fill kernel the title clears the array with (`cs_memset32`).
const CLEAR_CS: u64 = 0x10_0888_1200;
/// The index culls' programs (`gpu::exec::CULL_CS`).
const CULL_ADDRS: [u64; 4] = [0x10_0899_3700, 0x10_0899_4d00, 0x10_0899_7600, 0x10_0899_8c00];
/// Mode 12: the culls' noted writes over the array (`Exec::note_write`).
static CULL_NOTES: AtomicU64 = AtomicU64::new(0);
/// Mode 12: other writers of the array: (program, address, bytes) -> count.
static OTHER_WRITERS: parking_lot::Mutex<std::collections::BTreeMap<(u64, u64, u64), u64>> = parking_lot::Mutex::new(std::collections::BTreeMap::new());

/// Index culls with no groups left out by the command processor: [direct, indirect with the
/// arguments read on the CPU].
static CULLS_SKIPPED: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// Mode 12: a title call setting COMPUTE_PGM_LO (`hle::libs::agc_cmd`): for an index cull's
/// program, the guest frames of the first few distinct callers are logged.
pub fn note_program_set(lo: u32, via: &str) {
    if !scan_on() || !CULL_PGM_LO.contains(&lo) {
        return;
    }
    static SEEN: parking_lot::Mutex<Vec<String>> = parking_lot::Mutex::new(Vec::new());
    let frames = crate::exception::current_guest_frames(10);
    let key = frames.iter().take(3).cloned().collect::<Vec<_>>().join(" < ");
    let mut g = SEEN.lock();
    if g.len() < 6 && !g.contains(&key) {
        g.push(key);
        crate::info!("cull program {lo:#x} set by the title ({via}) at flip {}: {}", crate::hle::libs::videoout::QUEUED_FLIP.load(Ordering::Relaxed), frames.join(" < "));
    }
}

/// Mode 12: a DISPATCH_DIRECT the title emitted through the HLE (`sceAgcCbDispatch`, at `p`): the
/// 96 dwords before it are searched for a SET_SH_REG of an index cull's program (its callers
/// logged by `note_program_set`).
pub fn note_dispatch_emitted(p: u64) {
    if !scan_on() || p < 4 * 96 || !crate::gpu::cp::guest_readable(p - 4 * 96, 4 * 96) {
        return;
    }
    let w = unsafe { std::slice::from_raw_parts((p - 4 * 96) as *const u32, 96) };
    for i in 0..w.len().saturating_sub(2) {
        if w[i] >> 30 == 3 && (w[i] >> 8) & 0xFF == 0x76 {
            let n = ((w[i] >> 16) & 0x3FFF) as usize;
            let first = w[i + 1] & 0xFFFF;
            for k in 0..n {
                if first + k as u32 == 0x20C && i + 2 + k < w.len() {
                    note_program_set(w[i + 2 + k], "a dispatch after SET_SH_REG");
                }
            }
        }
    }
}

/// Index culls recorded by kind: [direct, indirect read on the CPU, indirect read by the GPU].
static CULL_KINDS: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// An index cull recorded: indirect with the GPU reading its arguments, or the CPU.
pub fn note_cull_kind(gpu_indirect: bool, cpu_indirect: bool) {
    CULL_KINDS[if gpu_indirect { 2 } else { cpu_indirect as usize }].fetch_add(1, Ordering::Relaxed);
}

/// An index cull dispatch left out for its zero groups (`indirect`: arguments the CPU read).
pub fn note_cull_skipped(indirect: bool) {
    CULLS_SKIPPED[indirect as usize].fetch_add(1, Ordering::Relaxed);
    ev(b'S', indirect as u64, 0);
}

/// An index cull recorded into a batch: its ticket.
pub fn note_cull_recorded() -> u64 {
    let t = CULLS_RECORDED.fetch_add(1, Ordering::AcqRel) + 1;
    let prev = LAST_CULL_T.fetch_max(t, Ordering::AcqRel);
    if scan_on() && LAST_CLEAR_T.load(Ordering::Acquire) > prev {
        // The first cull recorded after a clear.
        ev(b'K', t, 0);
    }
    t
}

/// A write the batch being recorded makes (`Exec::note_write`, by the program `writer`): a clear
/// of the array takes a ticket (returned, for the batch to carry).
#[inline]
pub fn note_array_write(a: u64, n: u64, writer: u64) -> Option<u64> {
    let arr = ARRAY.load(Ordering::Relaxed);
    if arr == 0 || a >= arr + ARRAY_BYTES || arr >= a.saturating_add(n) {
        return None;
    }
    if writer != CLEAR_CS {
        if scan_on() && CULL_ADDRS.contains(&writer) {
            CULL_NOTES.fetch_add(1, Ordering::Relaxed);
        }
        // Mode 12: writers of the array other than the culls and the clear (logged once each).
        if scan_on() && !CULL_ADDRS.contains(&writer) {
            let mut g = OTHER_WRITERS.lock();
            let e = g.entry((writer, a, n)).or_insert(0);
            *e += 1;
            if *e == 1 && g.len() <= 64 {
                crate::info!("cull array written by {writer:#x}: {a:#x}+{n:#x} at flip {}", crate::hle::libs::videoout::QUEUED_FLIP.load(Ordering::Relaxed));
            }
            ev(b'X', writer, a.wrapping_sub(arr) << 32 | n.min(0xFFFF_FFFF));
        }
        return None;
    }
    let t = CULLS_RECORDED.fetch_add(1, Ordering::AcqRel) + 1;
    LAST_CLEAR_T.fetch_max(t, Ordering::AcqRel);
    ev(b'C', t, n);
    Some(t)
}

/// A fill of guest memory in place the batch being recorded makes (`Exec::fill_in_place_parts`,
/// the fill kernel's dispatch as a buffer fill): over the array, a clear's ticket (returned).
#[inline]
pub fn note_array_fill(a: u64, n: u64, v: u32, writer: u64) -> Option<u64> {
    let arr = ARRAY.load(Ordering::Relaxed);
    if arr == 0 || a >= arr + ARRAY_BYTES || arr >= a.saturating_add(n) {
        return None;
    }
    let t = CULLS_RECORDED.fetch_add(1, Ordering::AcqRel) + 1;
    LAST_CLEAR_T.fetch_max(t, Ordering::AcqRel);
    ev(b'C', t, (v as u64) << 32 | n.min(0xFFFF_FFFF));
    static LOGGED: AtomicU64 = AtomicU64::new(0);
    if scan_on() && LOGGED.fetch_add(1, Ordering::Relaxed) < 8 {
        crate::info!("cull array filled in place: {a:#x}+{n:#x} with {v:#x} (writer {writer:#x}) at flip {}", crate::hle::libs::videoout::QUEUED_FLIP.load(Ordering::Relaxed));
    }
    Some(t)
}

/// A batch submitted to the GPU with its latest cull's or clear's ticket (0: none), its earliest
/// cull's and its clear's (0: none).
#[inline]
pub fn note_culls_submitted(seq: u64, ticket: u64, first: u64, clear: u64) {
    if ticket != 0 {
        CULLS_SUBMITTED.fetch_max(ticket, Ordering::AcqRel);
        ev(b'B', seq, first << 32 | (ticket & 0xFFFF_FFFF));
        if clear != 0 {
            ev(b'Z', seq, clear);
        }
    }
}

/// Mode 12's event log, a ring: (microseconds since the first event, kind, two values). Kinds:
/// `s` the title queued a graphics submission (index, 1 if it sets a cull's program), `R`/`r` the
/// graphics command processor begins/ends running one, `C` a clear of the array recorded (ticket,
/// bytes), `K` the first cull recorded after a clear (ticket), `B` a batch with culls or a clear
/// submitted to the GPU (sequence, latest ticket), `W` a pass's reads begin (flip, done batch),
/// `w` they ended (reads, blank zero reads << 32 | set zero reads).
struct Ev {
    us: u64,
    kind: u8,
    a: u64,
    b: u64,
}
static EVENTS: parking_lot::Mutex<std::collections::VecDeque<Ev>> = parking_lot::Mutex::new(std::collections::VecDeque::new());
static DUMPS: AtomicU64 = AtomicU64::new(0);

fn t0() -> &'static std::time::Instant {
    static T: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    T.get_or_init(std::time::Instant::now)
}

/// Notes an event in mode 12's ring.
pub fn ev(kind: u8, a: u64, b: u64) {
    if !scan_on() {
        return;
    }
    let us = t0().elapsed().as_micros() as u64;
    let mut g = EVENTS.lock();
    if g.len() >= 4096 {
        g.pop_front();
    }
    g.push_back(Ev { us, kind, a, b });
}

/// Logs the ring's last `n` events (a pass's reads found many blank records).
fn dump_events(n: usize, why: &str) {
    if DUMPS.fetch_add(1, Ordering::Relaxed) >= 40 {
        return;
    }
    let g = EVENTS.lock();
    let end = g.back().map_or(0, |e| e.us);
    let from = g.len().saturating_sub(n);
    let mut line = String::new();
    crate::info!("cull events ({why}): the last {} before now (us before now, kind, values):", g.len() - from);
    for (k, e) in g.iter().skip(from).enumerate() {
        line += &format!(" [-{} {} {:#x} {:#x}]", end - e.us, e.kind as char, e.a, e.b);
        if k % 12 == 11 {
            crate::info!("cull events:{line}");
            line.clear();
        }
    }
    if !line.is_empty() {
        crate::info!("cull events:{line}");
    }
}

/// Mode 12, the current pass's reads: [reads, blank zero reads, set zero reads].
static SESSION: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
/// Mode 12: per pass (its cull info, rcx at the read), the object sets its reads went through in
/// order (the set's table, rdi, with its first and last record and its reads), for the current
/// and the previous session of each pass.
#[derive(Clone, Default)]
struct SetRun {
    table: u64,
    first: u64,
    last: u64,
    reads: u64,
    blank: u64,
}
static SETS: parking_lot::Mutex<Option<std::collections::HashMap<u64, (Vec<SetRun>, Vec<SetRun>)>>> = parking_lot::Mutex::new(None);

fn note_set(info: u64, table: u64, record: u64, blank: bool, start: bool) {
    let mut g = SETS.lock();
    let m = g.get_or_insert_with(std::collections::HashMap::new);
    let e = m.entry(info).or_default();
    if start {
        e.1 = std::mem::take(&mut e.0);
    }
    match e.0.last_mut() {
        Some(r) if r.table == table => {
            r.last = r.last.max(record);
            r.reads += 1;
            r.blank += blank as u64;
        }
        _ => e.0.push(SetRun { table, first: record, last: record, reads: 1, blank: blank as u64 }),
    }
}

fn sets_line(info: u64) -> String {
    let g = SETS.lock();
    let Some((cur, prev)) = g.as_ref().and_then(|m| m.get(&info)) else { return String::new() };
    let f = |v: &Vec<SetRun>| v.iter().map(|r| format!("{:#x} {}..{} ({} reads, {} blank)", r.table, r.first, r.last, r.reads, r.blank)).collect::<Vec<_>>().join("; ");
    format!("pass {info:#x}: this time [{}]; before [{}]", f(cur), f(prev))
}
static LAST_INFO: AtomicU64 = AtomicU64::new(0);

/// Status line: the reads' waits.
pub fn status() -> String {
    let w: Vec<u64> = READ_WAITS.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    let s: Vec<u64> = SCANS.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    let scans = if s[0] > 0 { format!("    culls' submissions scanned: {} of {} title submissions set a cull's program ({} dwords, {} ms); the culls' noted writes over the array: {}, culls recorded: {}, left out with no groups: {} direct, {} indirect read on the CPU\n", s[1], s[0], s[2], s[3] / 1_000_000, CULL_NOTES.load(Ordering::Relaxed), CULLS_RECORDED.load(Ordering::Relaxed), CULLS_SKIPPED[0].load(Ordering::Relaxed), CULLS_SKIPPED[1].load(Ordering::Relaxed)) + &format!("    culls' descriptors at passes' first reads: {} checks, changed since the submission was queued {}, since the CP ran it {}, not yet run {}\n", DESC_STATS[0].load(Ordering::Relaxed), DESC_STATS[1].load(Ordering::Relaxed), DESC_STATS[2].load(Ordering::Relaxed), DESC_STATS[3].load(Ordering::Relaxed)) + &format!("    culls recorded by kind: {} direct, {} indirect read on the CPU, {} indirect read by the GPU
", CULL_KINDS[0].load(Ordering::Relaxed), CULL_KINDS[1].load(Ordering::Relaxed), CULL_KINDS[2].load(Ordering::Relaxed)) } else { String::new() };
    if w[0] == 0 {
        return scans;
    }
    format!("    culling results read by the title: {} reads waited for the GPU's writes ({} ms in all, {} after 100 ms)\n{scans}", w[0], w[1] / 1000, w[2])
}

/// The code inserted before the read (registers kept, flags clobbered: the displaced `cmp`
/// sets them; the red zone stepped over): first the wait (`read_wait`, called with every
/// caller-saved register and the SSE state kept, only when r10 is not the known array or a
/// batch writing it has not completed), then the hold. `counter` counts the held reads.
pub fn hold_code(counter: u64) -> Vec<u8> {
    use iced_x86::code_asm::*;
    let (tab, flip, one) = hold_addrs();
    let (array, array_seq, done) = (&ARRAY as *const AtomicU64 as u64, &ARRAY_SEQ as *const AtomicU64 as u64, crate::gpu::exec::done_atomic_addr());
    let wait = read_wait as *const () as usize as u64;
    let asm = || -> Result<Vec<u8>, IcedError> {
        let mut a = CodeAssembler::new(64)?;
        let mut zero = a.create_label();
        let mut done_l = a.create_label();
        let mut slow = a.create_label();
        let mut hold = a.create_label();
        a.lea(rsp, ptr(rsp - 0x80))?;
        a.push(rax)?;
        a.push(rcx)?;
        a.push(rdx)?;
        // The wait: nothing to do while r10 is the known array and its latest writer completed.
        a.mov(rax, array)?;
        a.cmp(r10, qword_ptr(rax))?;
        a.jne(slow)?;
        a.mov(rax, array_seq)?;
        a.mov(rax, qword_ptr(rax))?;
        a.mov(rcx, done)?;
        a.cmp(rax, qword_ptr(rcx))?;
        a.jbe(hold)?;
        a.set_label(&mut slow)?;
        // `read_wait(r10)`: the other caller-saved registers, rbx and rbp (the stack's
        // alignment), the SSE state.
        for r in [rsi, rdi, r8, r9, r10, r11, rbx, rbp] {
            a.push(r)?;
        }
        a.mov(rbp, rsp)?;
        a.and(rsp, -16)?;
        a.sub(rsp, 512)?;
        a.fxsave64(ptr(rsp))?;
        a.mov(rdi, r10)?;
        a.mov(rax, wait)?;
        a.call(rax)?;
        a.fxrstor64(ptr(rsp))?;
        a.mov(rsp, rbp)?;
        for r in [rbp, rbx, r11, r10, r9, r8, rdi, rsi] {
            a.pop(r)?;
        }
        a.set_label(&mut hold)?;
        a.cmp(rbx, HOLD_LEN as i32)?;
        a.jae(done_l)?;
        // The entry: slot (r10 * HASH) >> (64 - SLOT_BITS), then the record.
        a.mov(rax, HASH)?;
        a.imul_2(rax, r10)?;
        a.shr(rax, 64 - SLOT_BITS)?;
        a.shl(rax, HOLD_BITS)?;
        a.add(rax, rbx)?;
        a.mov(rdx, tab)?;
        a.lea(rdx, ptr(rdx + rax * 8))?;
        a.mov(rcx, flip)?;
        a.mov(ecx, dword_ptr(rcx))?;
        a.mov(eax, dword_ptr(r10 + rbx * 4))?;
        a.test(eax, eax)?;
        a.jz(zero)?;
        // Seen: (flip << 32) | r10's low half.
        a.shl(rcx, 32)?;
        a.mov(eax, r10d)?;
        a.or(rax, rcx)?;
        a.mov(qword_ptr(rdx), rax)?;
        a.jmp(done_l)?;
        a.set_label(&mut zero)?;
        // Only a blank record (the clear's state; a culled object's record keeps its other
        // words) ...
        a.mov(eax, dword_ptr(r10 + rbx * 4 + 4))?;
        a.or(eax, dword_ptr(r10 + rbx * 4 + 8))?;
        a.or(eax, dword_ptr(r10 + rbx * 4 + 12))?;
        a.or(eax, dword_ptr(r10 + rbx * 4 + 16))?;
        a.jnz(done_l)?;
        // ... read non-zero lately in the same array.
        a.mov(rax, qword_ptr(rdx))?;
        a.test(rax, rax)?;
        a.jz(done_l)?;
        a.cmp(eax, r10d)?;
        a.jne(done_l)?;
        a.shr(rax, 32)?;
        a.sub(ecx, eax)?;
        a.cmp(ecx, HOLD_FLIPS as i32)?;
        a.ja(done_l)?;
        a.mov(rax, counter)?;
        a.lock().inc(qword_ptr(rax))?;
        a.mov(r10, one)?;
        a.lea(rax, ptr(rbx * 4))?;
        a.sub(r10, rax)?;
        a.set_label(&mut done_l)?;
        a.pop(rdx)?;
        a.pop(rcx)?;
        a.pop(rax)?;
        a.lea(rsp, ptr(rsp + 0x80))?;
        a.assemble(0)
    };
    asm().expect("culling read hold: assembly")
}

/// Diagnosis: per flip, each array's reads, zeros and holds; mode 2's sampled holds (still 0,
/// refilled) and the refills' total wait.
static FLIP: AtomicU64 = AtomicU64::new(0);
static SAMPLED: AtomicU64 = AtomicU64::new(0);
static REFILL: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];
static REFILL_US: AtomicU64 = AtomicU64::new(0);
/// Mode 9: per record (its index), the first-instance field last read, and per flip [reads whose
/// record now holds another first instance, of those reading 0, zero reads whose record kept it].
fn last_first() -> &'static [std::sync::atomic::AtomicU32] {
    static T: std::sync::OnceLock<Box<[std::sync::atomic::AtomicU32]>> = std::sync::OnceLock::new();
    T.get_or_init(|| (0..(HOLD_LEN as usize) / 5 + 1).map(|_| std::sync::atomic::AtomicU32::new(0)).collect())
}
static MOVED: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
/// Mode 10, per flip: [the graphics queue's backlog at the first read (u64::MAX: none yet), the
/// largest, set zero reads seen lately with a backlog, without].
static BACKLOG: [AtomicU64; 4] = [AtomicU64::new(u64::MAX), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
/// Mode 8: the lowest and highest first-dword index of the flip's set zero reads seen lately.
static SET_SEEN_IDX: [AtomicU64; 2] = [AtomicU64::new(u64::MAX), AtomicU64::new(0)];
/// Zero reads by (record blank, read non-zero within `HOLD_FLIPS`): index blank << 1 | seen.
static ZERO_KIND: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];
/// Mode 11: per object (the set's table in rdi, the object's index in rax at the read), the words
/// 1-4 of the last non-blank record it read, behind a tag: [(tag << 32) | word 1, word 2, word 3,
/// word 4].
fn ident_table() -> &'static [[AtomicU64; 4]] {
    static T: std::sync::OnceLock<Box<[[AtomicU64; 4]]>> = std::sync::OnceLock::new();
    T.get_or_init(|| (0..1usize << 20).map(|_| [const { AtomicU64::new(0) }; 4]).collect())
}
/// Mode 11, per flip: [non-blank reads whose record holds another base vertex than the object's
/// last, of those reading 0, zero reads of a record with the object's own base vertex, non-blank
/// reads with it, blank reads, objects without a last, reads whose word 1 differs, word 2, word 4,
/// of the word-2 ones reading 0].
static IDENT: [AtomicU64; 10] = [const { AtomicU64::new(0) }; 10];
/// Mode 11: a few records' words logged (the first reads of the run).
static IDENT_SAMPLES: AtomicU64 = AtomicU64::new(0);
static STATS: parking_lot::Mutex<Vec<(u64, u64, u64, u64)>> = parking_lot::Mutex::new(Vec::new());

fn mode() -> u32 {
    static M: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *M.get_or_init(|| std::env::var("DS_CULLREAD_DIAG").ok().and_then(|v| v.parse().ok()).unwrap_or(1))
}

/// Mode 4 and 5's waits: reads that found GPU writes to the array in flight, their total wait
/// (us), waits that timed out.
static WAITS: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// Mode 7's waits for the graphics command processor to take up the title's submissions made
/// before the read: reads that waited, total us, waits given up (a label poll or 100 ms).
static SUB_WAITS: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// Mode 7: the read waits until the graphics command processor has run every title submission
/// queued before it (the culls the title submitted earlier in the frame recorded and handed to
/// the GPU), unless it polls a label (at most 100 ms).
fn wait_submitted() {
    let t = std::time::Instant::now();
    match crate::gpu::wait_gfx_caught_up(std::time::Duration::from_millis(100)) {
        None => {}
        Some(ok) => {
            SUB_WAITS[0].fetch_add(1, Ordering::Relaxed);
            SUB_WAITS[1].fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);
            if !ok {
                SUB_WAITS[2].fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

/// Modes 4, 5 and 7: the read waits until the GPU work in flight that writes the array (its
/// 75,000 dwords from r10) has completed, at most 100 ms (the title's reads then see the
/// culling's whole result for the work submitted so far, as the console's timing gives them).
fn wait_pending(r10: u64) {
    let Some(seq) = crate::gpu::exec::pending_writes(r10, 75_000 * 4) else { return };
    if seq <= crate::gpu::exec::done_now() {
        return;
    }
    let t = std::time::Instant::now();
    let ok = crate::gpu::exec::wait_done_for(seq, std::time::Duration::from_millis(100));
    WAITS[0].fetch_add(1, Ordering::Relaxed);
    WAITS[1].fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);
    if !ok {
        WAITS[2].fetch_add(1, Ordering::Relaxed);
    }
}

/// Diagnosis: at the read (before it), the hold of `hold_code` (unless mode 0 or 5) and the log.
pub extern "sysv64" fn read_hook(regs: *const u64, _arg: u32) {
    let r = |i: usize| unsafe { *regs.add(i) };
    let (rbx, r10) = (r(3), r(10));
    // (The array known as in `read_wait`: the clear and writers' notes need it.)
    if ARRAY.load(Ordering::Relaxed) == 0 {
        let _ = ARRAY.compare_exchange(0, r10, Ordering::AcqRel, Ordering::Acquire);
    }
    if mode() == 7 {
        wait_submitted();
    }
    if mode() >= 4 {
        wait_pending(r10);
    }
    let v = unsafe { std::ptr::read_volatile(r10.wrapping_add(rbx.wrapping_mul(4)) as *const u32) };
    let flip = crate::hle::libs::videoout::QUEUED_FLIP.load(Ordering::Relaxed);
    let last = FLIP.swap(flip, Ordering::Relaxed);
    let mut held = false;
    // Zero reads: is the whole 20-byte record zero (the clear's state) or are its other words
    // set (instances, first index or instance: written for a culled object)?
    let w = |k: u64| unsafe { std::ptr::read_volatile(r10.wrapping_add(rbx.wrapping_mul(4) + 4 * k) as *const u32) };
    let blank = v == 0 && (1..5).all(|k| w(k) == 0);
    if v == 0 {
        let seen = rbx < HOLD_LEN as u64 && {
            let e = entry(r10, rbx).load(Ordering::Relaxed);
            e != 0 && e & 0xFFFF_FFFF == r10 & 0xFFFF_FFFF && (flip as u32).wrapping_sub((e >> 32) as u32) <= HOLD_FLIPS
        };
        ZERO_KIND[(blank as usize) << 1 | seen as usize].fetch_add(1, Ordering::Relaxed);
        if mode() == 10 && !blank && seen {
            let b = crate::gpu::gfx_backlog();
            BACKLOG[if b > 0 { 2 } else { 3 }].fetch_add(1, Ordering::Relaxed);
        }
        if mode() == 9 && !blank && rbx < HOLD_LEN as u64 && rbx % 5 == 0 {
            let first = w(4);
            let prev = last_first()[(rbx / 5) as usize].load(Ordering::Relaxed);
            MOVED[if prev != 0 && prev != first { 1 } else { 2 }].fetch_add(1, Ordering::Relaxed);
        }
        // Mode 8: a set record read 0 though non-zero lately: one in 4 is read again for up
        // to 3 ms (a count that lands later, or one the culls computed as 0?), and the record
        // indices of these reads are kept for the flip's line.
        if mode() == 8 && !blank && seen {
            SET_SEEN_IDX[0].fetch_min(rbx, Ordering::Relaxed);
            SET_SEEN_IDX[1].fetch_max(rbx, Ordering::Relaxed);
            if SAMPLED.fetch_add(1, Ordering::Relaxed) % 4 == 0 {
                let t = std::time::Instant::now();
                let a = r10.wrapping_add(rbx.wrapping_mul(4)) as *const u32;
                while unsafe { std::ptr::read_volatile(a) } == 0 && t.elapsed() < std::time::Duration::from_millis(3) {
                    std::hint::spin_loop();
                }
                let filled = unsafe { std::ptr::read_volatile(a) } != 0;
                REFILL[filled as usize].fetch_add(1, Ordering::Relaxed);
                if filled {
                    REFILL_US.fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);
                }
            }
        }
        if rbx == 0 && mode() == 3 {
            crate::info!("cull read: flip {flip}: record 0 reads 0: [{} {} {} {}] after it", w(1), w(2), w(3), w(4));
        }
    }
    if mode() == 10 {
        let b = crate::gpu::gfx_backlog();
        let _ = BACKLOG[0].compare_exchange(u64::MAX, b, Ordering::Relaxed, Ordering::Relaxed);
        BACKLOG[1].fetch_max(b, Ordering::Relaxed);
    }
    // Mode 12: has the graphics command processor run the latest title submission that sets a
    // cull's program? (And the event log: a pass's reads begin at record 0.)
    if mode() == 12 {
        let info = r(1);
        if rbx == 0 {
            let s: Vec<u64> = SESSION.iter().map(|c| c.swap(0, Ordering::Relaxed)).collect();
            if s[0] > 0 {
                ev(b'w', s[0], s[1] << 32 | s[2]);
                if s[1] >= 50 {
                    let last = LAST_INFO.load(Ordering::Relaxed);
                    if DUMPS.load(Ordering::Relaxed) < 40 {
                        crate::info!("cull sets: {}", sets_line(last));
                    }
                    dump_events(260, &format!("a pass's {} reads found {} blank records", s[0], s[1]));
                }
            }
            ev(b'W', flip, crate::gpu::exec::done_now());
            check_desc_at_read();
        }
        LAST_INFO.store(info, Ordering::Relaxed);
        note_set(info, r(7), rbx / 5, blank, rbx == 0);
        SESSION[0].fetch_add(1, Ordering::Relaxed);
        if v == 0 {
            if blank && SESSION[1].load(Ordering::Relaxed) == 0 {
                // The pass's first blank read: its record, the latest batch writing the array
                // and the latest completed batch.
                ev(b'b', rbx / 5, ARRAY_SEQ.load(Ordering::Acquire) << 32 | (crate::gpu::exec::done_now() & 0xFFFF_FFFF));
            }
            SESSION[if blank { 1 } else { 2 }].fetch_add(1, Ordering::Relaxed);
        }
        let c = LATEST_CULL_SUB.load(Ordering::Acquire);
        let (_, d) = crate::gpu::gfx_counts();
        if c > d {
            BEHIND[0].fetch_add(1, Ordering::Relaxed);
            BEHIND[2].fetch_max(c - d, Ordering::Relaxed);
        } else {
            BEHIND[1].fetch_add(1, Ordering::Relaxed);
        }
        let (rec, sub) = (CULLS_RECORDED.load(Ordering::Acquire), CULLS_SUBMITTED.load(Ordering::Acquire));
        if rec > sub {
            BEHIND[3].fetch_add(1, Ordering::Relaxed);
            BEHIND[4].fetch_max(rec - sub, Ordering::Relaxed);
        }
    }
    // Mode 11: does the record belong to the object reading it (its base vertex as at the
    // object's last non-blank read)?
    if mode() == 11 {
        let (rax, rdi) = (r(0), r(7));
        let (w1, w2, w3, w4) = (w(1), w(2), w(3), w(4));
        if IDENT_SAMPLES.fetch_add(1, Ordering::Relaxed) % 50_000 < 8 {
            crate::info!("cull read sample: set {rdi:#x} object {rax} record {} at flip {flip}: [{v} {w1} {w2} {w3} {w4}]", rbx / 5);
        }
        let tag = (rdi as u32) ^ (rax as u32).rotate_left(16) ^ 0x8000_0001;
        let slot = ((rdi.wrapping_mul(HASH) ^ rax.wrapping_mul(0xC2B2_AE3D_27D4_EB4F)) >> 44) as usize;
        let e = &ident_table()[slot];
        if blank {
            IDENT[4].fetch_add(1, Ordering::Relaxed);
        } else {
            let last = e[0].load(Ordering::Relaxed);
            if last >> 32 != tag as u64 {
                IDENT[5].fetch_add(1, Ordering::Relaxed);
            } else {
                if e[2].load(Ordering::Relaxed) as u32 != w3 {
                    IDENT[0].fetch_add(1, Ordering::Relaxed);
                    IDENT[1].fetch_add((v == 0) as u64, Ordering::Relaxed);
                } else {
                    IDENT[3].fetch_add(1, Ordering::Relaxed);
                    IDENT[2].fetch_add((v == 0) as u64, Ordering::Relaxed);
                }
                IDENT[6].fetch_add((last as u32 != w1) as u64, Ordering::Relaxed);
                let d2 = e[1].load(Ordering::Relaxed) as u32 != w2;
                IDENT[7].fetch_add(d2 as u64, Ordering::Relaxed);
                IDENT[9].fetch_add((d2 && v == 0) as u64, Ordering::Relaxed);
                IDENT[8].fetch_add((e[3].load(Ordering::Relaxed) as u32 != w4) as u64, Ordering::Relaxed);
            }
            e[0].store(((tag as u64) << 32) | w1 as u64, Ordering::Relaxed);
            e[1].store(w2 as u64, Ordering::Relaxed);
            e[2].store(w3 as u64, Ordering::Relaxed);
            e[3].store(w4 as u64, Ordering::Relaxed);
        }
    }
    // Mode 9: the record's first instance against the one it held at its last read.
    if mode() == 9 && rbx < HOLD_LEN as u64 && rbx % 5 == 0 {
        let first = w(4);
        if first != 0 || v != 0 {
            let prev = last_first()[(rbx / 5) as usize].swap(first, Ordering::Relaxed);
            if prev != 0 && prev != first {
                MOVED[0].fetch_add(1, Ordering::Relaxed);
            }
        }
    }
    if mode() != 0 && mode() != 5 && rbx < HOLD_LEN as u64 {
        let e = entry(r10, rbx);
        if v != 0 {
            e.store((flip << 32) | (r10 & 0xFFFF_FFFF), Ordering::Relaxed);
        } else if blank {
            let seen = e.load(Ordering::Relaxed);
            if seen != 0 && seen & 0xFFFF_FFFF == r10 & 0xFFFF_FFFF && (flip as u32).wrapping_sub((seen >> 32) as u32) <= HOLD_FLIPS {
                held = true;
                // Mode 2: one held read in 16 is read again for up to 2 ms: does the GPU's
                // refill reach it (a read inside the clear's window) or does it stay 0 (a cull)?
                if mode() == 2 && SAMPLED.fetch_add(1, Ordering::Relaxed) % 16 == 0 {
                    let t = std::time::Instant::now();
                    let a = r10.wrapping_add(rbx.wrapping_mul(4)) as *const u32;
                    while unsafe { std::ptr::read_volatile(a) } == 0 && t.elapsed() < std::time::Duration::from_millis(2) {
                        std::hint::spin_loop();
                    }
                    let filled = unsafe { std::ptr::read_volatile(a) } != 0;
                    REFILL[filled as usize].fetch_add(1, Ordering::Relaxed);
                    if filled {
                        REFILL_US.fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);
                    }
                }
                // The displaced `cmp dword ptr [r10+rbx*4], 0` then reads ONE.
                unsafe { *(regs.add(10) as *mut u64) = (&ONE as *const u32 as u64).wrapping_sub(rbx.wrapping_mul(4)) };
            }
        }
    }
    let mut s = STATS.lock();
    if last != flip && last != 0 && !s.is_empty() {
        let line: Vec<String> = s.iter().map(|(a, n, z, h)| format!("{a:#x}: {n} reads, {z} zero, {h} held")).collect();
        let (stay, fill) = (REFILL[0].swap(0, Ordering::Relaxed), REFILL[1].swap(0, Ordering::Relaxed));
        let refill = if stay + fill > 0 {
            if mode() == 8 {
                let (lo, hi) = (SET_SEEN_IDX[0].swap(u64::MAX, Ordering::Relaxed), SET_SEEN_IDX[1].swap(0, Ordering::Relaxed));
                format!("; sampled set zeros seen lately (records {}..{}): {fill} refilled (mean {} us), {stay} still 0 after 3 ms", lo / 5, hi / 5, REFILL_US.swap(0, Ordering::Relaxed) / fill.max(1))
            } else {
                format!("; sampled holds: {fill} refilled (mean {} us), {stay} still 0 after 2 ms", REFILL_US.swap(0, Ordering::Relaxed) / fill.max(1))
            }
        } else {
            String::new()
        };
        let k: Vec<u64> = ZERO_KIND.iter().map(|c| c.swap(0, Ordering::Relaxed)).collect();
        let w: Vec<u64> = WAITS.iter().map(|c| c.swap(0, Ordering::Relaxed)).collect();
        let sw: Vec<u64> = SUB_WAITS.iter().map(|c| c.swap(0, Ordering::Relaxed)).collect();
        let mut waits = if mode() >= 4 { format!("; waits for the GPU's writes: {} ({} us, {} timed out)", w[0], w[1], w[2]) } else { String::new() };
        if mode() == 7 {
            waits += &format!("; waits for the graphics CP to take up earlier submissions: {} ({} us, {} given up)", sw[0], sw[1], sw[2]);
        }
        if mode() == 9 {
            let m: Vec<u64> = MOVED.iter().map(|c| c.swap(0, Ordering::Relaxed)).collect();
            waits += &format!("; records holding another first instance than at their last read: {} (set zero reads: {} moved, {} kept)", m[0], m[1], m[2]);
        }
        if mode() == 12 {
            let b: Vec<u64> = BEHIND.iter().map(|c| c.swap(0, Ordering::Relaxed)).collect();
            waits += &format!("; reads before the command processor ran the latest culls' submission: {} (after: {}, most behind {}); reads while recorded culls were not yet submitted: {} (most {})", b[0], b[1], b[2], b[3], b[4]);
        }
        if mode() == 11 {
            let i: Vec<u64> = IDENT.iter().map(|c| c.swap(0, Ordering::Relaxed)).collect();
            waits += &format!("; records of another base vertex than the object's last: {} ({} read 0); own: {} ({} read 0); blank {}; new objects {}; word 1 changed {}, word 2 {} ({} read 0), word 4 {}", i[0], i[1], i[3], i[2], i[4], i[5], i[6], i[7], i[9], i[8]);
        }
        if mode() == 10 {
            let b: Vec<u64> = BACKLOG.iter().map(|c| c.swap(0, Ordering::Relaxed)).collect();
            waits += &format!("; graphics submissions not yet run at the reads: first {}, most {}; set zero reads seen lately with a backlog {}, without {}", b[0], b[1], b[2], b[3]);
            BACKLOG[0].store(u64::MAX, Ordering::Relaxed);
        }
        crate::info!(
            "cull read: flip {last}: {}{refill}; zero records: blank {} ({} seen lately), set {} ({} seen lately){waits}",
            line.join("; "),
            k[2] + k[3],
            k[3],
            k[0] + k[1],
            k[1]
        );
        s.clear();
    }
    let i = match s.iter().position(|e| e.0 == r10) {
        Some(i) => i,
        None => {
            s.push((r10, 0, 0, 0));
            s.len() - 1
        }
    };
    s[i].1 += 1;
    s[i].2 += (v == 0) as u64;
    s[i].3 += held as u64;
}
