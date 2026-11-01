//! Hardware LOD statistics (`sceAgcDcbGetLodStats`), the title's texture
//! streaming feedback. Each frame its streamer lends up to 256 counter banks
//! to textures (T# word 5 bit 25, LOD_HDW_CNT_EN; word 6 bits 0..7, the
//! bank), then reads per bank the 8-byte entry at 0x40 + 8 * bank of the
//! statistics buffer: bits 56..59 the finest mip sampled (15: not sampled),
//! bits 32..55 a count (eboot+0xac90b0, eboot+0xac5b70). With the buffer left
//! as written (zero) the streamer never loaded the high mips (chunk files) of
//! most textures: they stayed at a quarter or an eighth of their resolution.
//!
//! The GPU here has no such counters. Mode 3 (`tune lod_stats`; mode 4
//! the same with `Spec::lod_lean`: a bank's
//! mip stored only when finer than the one it holds, the counts spread over
//! `rdna::translate::LOD_COUNT_SLOTS` slots per bank and summed here)
//! measures them: the translated sampling instructions of textures whose T#
//! enables counting add to the T#'s bank the finest mip they sample (their
//! level of detail relative to the texture's mip 0, from the implicit
//! derivatives, the LOD operand or the gradients) and the number of lanes
//! (`rdna::translate::Spec::lod_stats`); a GetLodStats packet copies the
//! banks after the work before it and the copy is written to the title's
//! buffer once that work completed. Mode 2, the earlier stand-in, reports a
//! bank as sampled at mip 0 when a draw or dispatch binds its texture
//! (every texture of a bound image table counts): the streamer then loaded
//! the high mips of every texture it rotated banks through, the texture pool
//! grew by ~13 MiB/s until the title's memory ran out at a level load (run
//! 20261011-101255: "Out of Memory ... CPUMemory", VirtualMemoryGPU 4.2 GiB).
//!
//! Pixel programs whose depth tests stay after the shader (a kill, coverage
//! or export with depth or stencil writes) decide their statistics per draw
//! (`shaders::lod_per_draw`): the statistics' atomics are side effects, so
//! the driver shades every fragment of their draws before testing depth,
//! hidden ones too (5 ms of GPU time a frame at 1-3's Penetrator position,
//! runs 20261030-093305 .. -094629). Their stages are prepared without them;
//! a draw binding no texture that enables counting (~96% of such draws at
//! 1-3) needs none (its atomics never ran), one that binds such a texture is
//! prepared again with them (`rotate`): every one under `tune lod_late` 1,
//! one in N under N >= 2, by its place among them in the report. The others
//! carry their textures: a bank lent to a carried texture reports the
//! finest mip and the largest count measured for that texture (by its T#
//! base address: banks are lent anew every frame) in the last 2N reports,
//! this one's included.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};

/// Bytes of the measured statistics (mip per bank, then count per bank).
pub const SLOT_BYTES: u64 = rdna::translate::LOD_STATS_DWORDS as u64 * 4;
/// The finest mip per bank, before the count slots.
pub const MIN_BYTES: u64 = 256 * 4;
/// Readback slots for measured statistics (GetLodStats packets in flight).
pub const RING: usize = 16;
static NEXT_SLOT: AtomicUsize = AtomicUsize::new(0);
static SLOT_BUSY: [AtomicBool; RING] = [const { AtomicBool::new(false) }; RING];

/// The LOD information of a T# for the translated sampling instructions
/// (`rdna::translate::LOD_INFO_ENABLED`): its counter bank and base level
/// when it enables counting, else 0.
pub fn info(t: &super::desc::Tsharp) -> u32 {
    if (t.raw[5] >> 25) & 1 == 0 {
        return 0;
    }
    rdna::translate::LOD_INFO_ENABLED | ((t.base_level & 0xF) << 8) | (t.raw[6] & 0xFF)
}

/// Measured reports: [reports, reports with a sampled bank, banks sampled,
/// then banks per finest mip 0..15, then banks carried: from earlier
/// reports' measurements (no sample of their own in this one), with this
/// one's, without any measurement to report] (status line).
static TOTALS: [AtomicU64; 22] = [const { AtomicU64::new(0) }; 22];

/// Draws of pixel programs deciding their statistics per draw: [without a
/// counted texture (never measure), measuring, carrying].
static LATE_DRAWS: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
/// Measuring draws that measured in a pass of their own (`tune lod_pass`),
/// and those whose pass was left out (the draw kept no depth test and
/// write, its pipeline did not push its descriptors, or its rendering
/// instance ended).
pub static PASSES: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// One status line on the measured statistics ("" before any).
pub fn status() -> String {
    let t: Vec<u64> = TOTALS.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    if t[0] == 0 {
        return String::new();
    }
    let mips: Vec<String> = (0..16).filter(|&m| t[3 + m] > 0).map(|m| format!("{m}:{}", t[3 + m])).collect();
    let l: Vec<u64> = LATE_DRAWS.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    let rotated = if l.iter().sum::<u64>() > 0 {
        let pass = PASSES[0].load(Ordering::Relaxed) + PASSES[1].load(Ordering::Relaxed);
        let pass = if pass > 0 { format!(" ({} of them in a pass of their own, {} left without)", PASSES[0].load(Ordering::Relaxed), PASSES[1].load(Ordering::Relaxed)) } else { String::new() };
        format!("; late-test draws deciding per draw: {} without counted textures, {} measuring{pass}, {} carrying; banks carried: {} from earlier reports, {} with this one's, {} with nothing measured", l[0], l[1], l[2], t[19], t[20], t[21])
    } else {
        String::new()
    };
    format!("    LOD statistics measured on the GPU: {} reports, {} with sampled banks, {} banks sampled (finest mip:banks {}){rotated}\n", t[0], t[1], t[2], mips.join(" "))
}

/// Reports with reset so far (the streamer's frames), and the draws of
/// rotated programs with counted textures in this one.
static REPORTS: AtomicU64 = AtomicU64::new(0);
static LATE_SEQ: AtomicU64 = AtomicU64::new(0);
/// Per bank in this report: the texture (T# base address) a rotated draw
/// bound with it, and the one a draw without statistics bound (0: none).
/// Set by the graphics CP's draws only (preparations made ahead belong to
/// later reports).
static BANK_TEX: [AtomicU64; 256] = [const { AtomicU64::new(0) }; 256];
static CARRIED: [AtomicU64; 256] = [const { AtomicU64::new(0) }; 256];

fn counts(raw: &[u32; 8]) -> bool {
    (raw[5] >> 25) & 1 != 0
}

fn texture(raw: &[u32; 8]) -> u64 {
    ((raw[0] as u64) << 8) | (((raw[1] & 0xFF) as u64) << 40)
}

/// The counted textures among `res`.
fn each_counted(res: &[super::shaders::Resolved], mut f: impl FnMut(&[u32; 8])) {
    for r in res {
        match r {
            super::shaders::Resolved::Image(t) if counts(&t.raw) => f(&t.raw),
            super::shaders::Resolved::ImageTable(e) => e.iter().flatten().filter(|t| counts(&t.raw)).for_each(|t| f(&t.raw)),
            _ => {}
        }
    }
}

/// Whether `res` binds a texture with counting enabled.
pub fn counted(res: &[super::shaders::Resolved]) -> bool {
    let mut any = false;
    each_counted(res, |_| any = true);
    any
}

/// A draw of a pixel program prepared without its statistics
/// (`shaders::lod_rotated`) with resources `res`: whether it measures (to
/// be prepared again with them, its textures then given to `bound`): with a
/// counted texture, every time at `tune lod_late` 1, else in rotation, its
/// counted textures carried when it does not.
pub fn rotate(res: &[super::shaders::Resolved]) -> bool {
    if !counted(res) {
        LATE_DRAWS[0].fetch_add(1, Ordering::Relaxed);
        return false;
    }
    let n = super::tune::lod_late().max(1);
    let k = LATE_SEQ.fetch_add(1, Ordering::Relaxed);
    if n == 1 || (k + REPORTS.load(Ordering::Relaxed)) % n == 0 {
        LATE_DRAWS[1].fetch_add(1, Ordering::Relaxed);
        return true;
    }
    LATE_DRAWS[2].fetch_add(1, Ordering::Relaxed);
    carry(res);
    false
}

/// A rotated draw without statistics bound `res`: its counted textures'
/// banks report their recent measurements.
pub fn carry(res: &[super::shaders::Resolved]) {
    each_counted(res, |raw| {
        let (b, t) = ((raw[6] & 0xFF) as usize, texture(raw));
        BANK_TEX[b].store(t, Ordering::Relaxed);
        CARRIED[b].store(t, Ordering::Relaxed);
    });
}

/// A rotated draw measuring with resources `res`: its banks' measurements
/// are kept for the textures.
pub fn bound(res: &[super::shaders::Resolved]) {
    each_counted(res, |raw| BANK_TEX[(raw[6] & 0xFF) as usize].store(texture(raw), Ordering::Relaxed));
}

/// What a GetLodStats packet found of the rotation (`end_report`).
pub struct Carry {
    report: u64,
    window: u64,
    banks: Box<[u64; 256]>,
    carried: Box<[u64; 256]>,
}

/// The rotation's state at a GetLodStats packet (graphics CP, in order with
/// the draws), started again with `reset`; None while nothing rotates.
pub fn end_report(reset: bool) -> Option<Carry> {
    let n = super::tune::lod_late();
    let take = |a: &[AtomicU64; 256]| -> Box<[u64; 256]> { Box::new(std::array::from_fn(|b| if reset { a[b].swap(0, Ordering::Relaxed) } else { a[b].load(Ordering::Relaxed) })) };
    let c = Carry { report: REPORTS.load(Ordering::Relaxed), window: (2 * n).min(HIST as u64), banks: take(&BANK_TEX), carried: take(&CARRIED) };
    if reset {
        REPORTS.fetch_add(1, Ordering::Relaxed);
        LATE_SEQ.store(0, Ordering::Relaxed);
    }
    (n >= 2).then_some(c)
}

/// Per texture the measurements of the last HIST reports: (report, finest
/// mip, count) by report % HIST.
const HIST: usize = 32;
struct Hist {
    at: [u64; HIST],
    mip: [u8; HIST],
    count: [u32; HIST],
    last: u64,
}
static MEASURED: parking_lot::Mutex<Option<HashMap<u64, Hist>>> = parking_lot::Mutex::new(None);

/// Applies the rotation to report `c` (GPU values `mip`/`count` per bank):
/// the measured banks of rotated textures are kept, the carried ones get the
/// window's finest mip and largest count. Returns [banks reported from
/// earlier reports only, with this one's, carried with nothing measured].
fn apply(c: &Carry, mips: &mut [u32; 256], cnts: &mut [u32; 256]) -> [u64; 3] {
    let mut out = [0u64; 3];
    let mut g = MEASURED.lock();
    let m = g.get_or_insert_with(HashMap::new);
    let r = c.report;
    for b in 0..256 {
        let t = c.banks[b];
        if t != 0 && cnts[b] > 0 && mips[b] < 15 {
            let h = m.entry(t).or_insert_with(|| Hist { at: [u64::MAX; HIST], mip: [15; HIST], count: [0; HIST], last: r });
            let s = (r % HIST as u64) as usize;
            if h.at[s] != r {
                (h.at[s], h.mip[s], h.count[s]) = (r, 15, 0);
            }
            h.mip[s] = h.mip[s].min(mips[b] as u8);
            h.count[s] = h.count[s].max(cnts[b]);
            h.last = r;
        }
    }
    for b in 0..256 {
        let t = c.carried[b];
        if t == 0 {
            continue;
        }
        let w = m.get(&t).map(|h| {
            (0..HIST).filter(|&s| h.at[s] <= r && r - h.at[s] < c.window).fold((15u32, 0u32), |(mm, nn), s| (mm.min(h.mip[s] as u32), nn.max(h.count[s])))
        });
        match w {
            Some((wm, wn)) if wn > 0 && wm < 15 => {
                out[if cnts[b] > 0 && mips[b] < 15 { 1 } else { 0 }] += 1;
                mips[b] = mips[b].min(wm);
                cnts[b] = cnts[b].max(wn);
            }
            _ => out[2] += 1,
        }
    }
    // Textures not measured for a while are forgotten.
    if r % 256 == 0 {
        m.retain(|_, h| h.last + 4 * HIST as u64 > r);
    }
    out
}

/// A readback slot for one GetLodStats copy (waits while all are in use).
pub fn take_slot() -> usize {
    let s = NEXT_SLOT.fetch_add(1, Ordering::Relaxed) % RING;
    let mut waited = 0u32;
    while SLOT_BUSY[s].swap(true, Ordering::AcqRel) {
        if waited == 0 {
            crate::warn!("gpu: LOD statistics readback slot {s} still in use (more than {RING} reports in flight)");
        }
        waited += 1;
        if waited > 5000 || super::shutting_down() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    s
}

/// Writes measured statistics (`words`: mip per bank, then count per bank)
/// into the title's buffer, with the rotation's carried banks (`carry`),
/// and frees readback slot `slot`. Returns the banks reported as sampled
/// and how many of them per finest mip.
///
/// # Safety
/// `buf` must be writable guest memory of `BUFFER_BYTES`.
pub unsafe fn write_measured(buf: u64, words: &[u32], slot: usize, carry: Option<&Carry>) -> (u32, [u32; 16]) {
    let mut sampled = 0;
    let mut by_mip = [0u32; 16];
    // (Counts in LOD_COUNT_SLOTS slots per bank under mode 4; slot 0 alone
    // under mode 3.)
    let mut mips: [u32; 256] = std::array::from_fn(|bank| words.get(bank).copied().unwrap_or(15));
    let mut cnts: [u32; 256] = std::array::from_fn(|bank| (0..rdna::translate::LOD_COUNT_SLOTS as usize).map(|k| words.get(256 + 256 * k + bank).copied().unwrap_or(0) as u64).sum::<u64>().min(u32::MAX as u64) as u32);
    if let Some(c) = carry {
        let n = apply(c, &mut mips, &mut cnts);
        for (k, v) in n.iter().enumerate() {
            TOTALS[19 + k].fetch_add(*v, Ordering::Relaxed);
        }
    }
    for bank in 0..256 {
        let (mip, count) = (mips[bank], cnts[bank]);
        let entry = if count > 0 && mip < 15 {
            sampled += 1;
            by_mip[mip as usize] += 1;
            ((mip as u64) << 56) | ((count.min(0xFF_FFFF) as u64) << 32)
        } else {
            15u64 << 56
        };
        unsafe { ((buf + ENTRIES + 8 * bank as u64) as *mut u64).write_unaligned(entry) };
    }
    SLOT_BUSY[slot % RING].store(false, Ordering::Release);
    TOTALS[0].fetch_add(1, Ordering::Relaxed);
    if sampled > 0 {
        TOTALS[1].fetch_add(1, Ordering::Relaxed);
        TOTALS[2].fetch_add(sampled as u64, Ordering::Relaxed);
        for (m, &c) in by_mip.iter().enumerate() {
            TOTALS[3 + m].fetch_add(c as u64, Ordering::Relaxed);
        }
    }
    (sampled, by_mip)
}

/// Bindings of each bank's texture since the last report.
static BOUND: [AtomicU32; 256] = [const { AtomicU32::new(0) }; 256];

/// Entry offset and size of the statistics buffer the title reads.
const ENTRIES: u64 = 0x40;
pub const BUFFER_BYTES: u64 = ENTRIES + 8 * 256;

/// Notes a texture bound by a draw or dispatch (T# words).
pub fn note(raw: &[u32; 8]) {
    if (raw[5] >> 25) & 1 != 0 {
        BOUND[(raw[6] & 0xFF) as usize].fetch_add(1, Ordering::Relaxed);
    }
}

/// Writes the statistics into the title's buffer (the CP reached its
/// GetLodStats packet): mode 1 reports every bank as sampled, mode 2 the
/// banks bound since the last report. Returns the banks reported as sampled.
///
/// # Safety
/// `buf` must be writable guest memory of `BUFFER_BYTES`.
pub unsafe fn report(buf: u64, mode: u64, reset: bool) -> u32 {
    let mut sampled = 0;
    for (bank, n) in BOUND.iter().enumerate() {
        let n = if reset { n.swap(0, Ordering::Relaxed) } else { n.load(Ordering::Relaxed) };
        let count = match mode {
            1 => 0xFF_FFFF,
            _ => (n as u64 * 64).min(0xFF_FFFF),
        };
        let entry = if count > 0 {
            sampled += 1;
            count << 32
        } else {
            15u64 << 56
        };
        unsafe { ((buf + ENTRIES + 8 * bank as u64) as *mut u64).write_unaligned(entry) };
    }
    sampled
}

#[cfg(test)]
mod tests {
    use super::*;

    fn report(r: u64, window: u64, banks: &[(usize, u64)], carried: &[(usize, u64)]) -> Carry {
        let mut c = Carry { report: r, window, banks: Box::new([0; 256]), carried: Box::new([0; 256]) };
        for &(b, t) in banks {
            c.banks[b] = t;
        }
        for &(b, t) in carried {
            c.carried[b] = t;
        }
        c
    }

    #[test]
    fn carried_banks_report_the_window() {
        let t = 0x7700_0000_0100u64;
        let (mut mips, mut cnts) = ([15u32; 256], [0u32; 256]);
        // Measured on bank 3 at report 10 (mip 2, 500 lanes) and at report
        // 12 (mip 4, 100 lanes), lent bank 9 at 13 and carried there.
        (mips[3], cnts[3]) = (2, 500);
        assert_eq!(apply(&report(10, 4, &[(3, t)], &[]), &mut mips, &mut cnts), [0, 0, 0]);
        (mips[3], cnts[3]) = (4, 100);
        apply(&report(12, 4, &[(3, t)], &[]), &mut mips, &mut cnts);
        let (mut mips, mut cnts) = ([15u32; 256], [0u32; 256]);
        assert_eq!(apply(&report(13, 4, &[(9, t)], &[(9, t)]), &mut mips, &mut cnts), [1, 0, 0]);
        assert_eq!((mips[9], cnts[9]), (2, 500));
        // With a sample of its own: the finer mip and the larger count.
        let (mut mips, mut cnts) = ([15u32; 256], [0u32; 256]);
        (mips[9], cnts[9]) = (3, 900);
        assert_eq!(apply(&report(13, 4, &[(9, t)], &[(9, t)]), &mut mips, &mut cnts), [0, 1, 0]);
        assert_eq!((mips[9], cnts[9]), (2, 900));
        // Report 10 leaves the window of 4 at 14; 12's and 13's stay.
        let (mut mips, mut cnts) = ([15u32; 256], [0u32; 256]);
        apply(&report(14, 4, &[], &[(5, t)]), &mut mips, &mut cnts);
        assert_eq!((mips[5], cnts[5]), (3, 900));
        // Past every measurement: reported as not sampled.
        let (mut mips, mut cnts) = ([15u32; 256], [0u32; 256]);
        assert_eq!(apply(&report(20, 4, &[], &[(5, t)]), &mut mips, &mut cnts), [0, 0, 1]);
        assert_eq!((mips[5], cnts[5]), (15, 0));
    }
}
