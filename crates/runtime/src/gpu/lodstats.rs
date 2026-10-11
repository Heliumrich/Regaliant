//! Hardware LOD statistics (`sceAgcDcbGetLodStats`), the title's texture
//! streaming feedback. Each frame its streamer lends up to 256 counter banks
//! to textures (T# word 5 bit 25, LOD_HDW_CNT_EN; word 6 bits 0..7, the
//! bank), then reads per bank the 8-byte entry at 0x40 + 8 * bank of the
//! statistics buffer: bits 56..59 the finest mip sampled (15: not sampled),
//! bits 32..55 a count (eboot+0xac90b0, eboot+0xac5b70). With the buffer left
//! as written (zero) the streamer never loaded the high mips (chunk files) of
//! most textures: they stayed at a quarter or an eighth of their resolution.
//!
//! The GPU here has no such counters. Mode 3 (`tune lod_stats`, default)
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

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};

/// Bytes of the measured statistics (mip per bank, then count per bank).
pub const SLOT_BYTES: u64 = rdna::translate::LOD_STATS_DWORDS as u64 * 4;
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
/// then banks per finest mip 0..15] (status line).
static TOTALS: [std::sync::atomic::AtomicU64; 19] = [const { std::sync::atomic::AtomicU64::new(0) }; 19];

/// One status line on the measured statistics ("" before any).
pub fn status() -> String {
    let t: Vec<u64> = TOTALS.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    if t[0] == 0 {
        return String::new();
    }
    let mips: Vec<String> = (0..16).filter(|&m| t[3 + m] > 0).map(|m| format!("{m}:{}", t[3 + m])).collect();
    format!("    LOD statistics measured on the GPU: {} reports, {} with sampled banks, {} banks sampled (finest mip:banks {})
", t[0], t[1], t[2], mips.join(" "))
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
/// into the title's buffer and frees readback slot `slot`. Returns the banks
/// reported as sampled and how many of them per finest mip.
///
/// # Safety
/// `buf` must be writable guest memory of `BUFFER_BYTES`.
pub unsafe fn write_measured(buf: u64, words: &[u32], slot: usize) -> (u32, [u32; 16]) {
    let mut sampled = 0;
    let mut by_mip = [0u32; 16];
    for bank in 0..256 {
        let (mip, count) = (words.get(bank).copied().unwrap_or(15), words.get(256 + bank).copied().unwrap_or(0));
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
