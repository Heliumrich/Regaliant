//! Hardware LOD statistics (`sceAgcDcbGetLodStats`), the title's texture
//! streaming feedback. Each frame its streamer lends up to 256 counter banks
//! to textures (T# word 5 bit 25, LOD_HDW_CNT_EN; word 6 bits 0..7, the
//! bank), then reads per bank the 8-byte entry at 0x40 + 8 * bank of the
//! statistics buffer: bits 56..59 the finest mip sampled (15: not sampled),
//! bits 32..55 a count (eboot+0xac90b0, eboot+0xac5b70). With the buffer left
//! as written (zero) the streamer never loaded the high mips (chunk files) of
//! most textures: they stayed at a quarter or an eighth of their resolution.
//!
//! The GPU here has no such counters: a bank counts as sampled when a draw or
//! dispatch binds its texture, at mip 0 (the finest, rather than the level
//! the sampling would select), counting the bindings (`tune lod_stats`).

use std::sync::atomic::{AtomicU32, Ordering};

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
