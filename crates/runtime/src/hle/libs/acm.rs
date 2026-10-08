//! libSceAcm: the PS5 audio compute manager (DSP jobs run on the audio
//! coprocessor). The title's audio engine uses it for one DSP module,
//! `NCA::ACMConvolver1` (a partitioned convolution reverb), and requires the
//! context to exist (its engine init fails otherwise, eboot+0x1ddea9b).
//!
//! Batch buffers are descriptors {u64 data, u64 used, u64 capacity} holding
//! records that start with a u64 opcode. The title writes one record kind
//! itself: opcode 0x0B {u64 0x0B, u64 status*}, a status/statistics request
//! whose u32 at status+8 it reads back as the batch's processor load
//! (value * 100 / 2^31 percent, eboot+0x1de477d). Job records from the
//! builder below use this runtime's own private opcode (the builder is a
//! library function, so the encoding is ours).
//!
//! Batches are consumed at submission: the title builds its per-tick batch in
//! a stack buffer and returns right after `sceAcmBatchStartBuffers`, so the
//! records cannot be read later. Waits therefore always find the batch done.
//!
//! STAND-IN: convolution jobs are not computed yet. They are recorded and
//! counted, and their outputs are left untouched (the reverb's wet signal is
//! missing). The first jobs' argument structures are dumped to the log for
//! the follow-up implementation.

use crate::hle::Registry;
use crate::{reg, sysv};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

const ERR_INVALID_PARAMETER: i32 = 0x8194_0002u32 as i32;
const ERR_BUFFER_FULL: i32 = 0x8194_0004u32 as i32;

/// Title-written status record (see module docs).
const OP_STATUS: u64 = 0x0B;
/// Private opcode of the convolution job records written by the builder:
/// {u64 OP_DS_CONVOLVE, u64 record bytes, 7 x u64 builder arguments}.
const OP_DS_CONVOLVE: u64 = 0x4453_4143_0000_0001; // "DSAC" | 1
const CONVOLVE_RECORD: u64 = 9 * 8;

static NEXT_BATCH: AtomicU32 = AtomicU32::new(1);
static JOBS: AtomicU64 = AtomicU64::new(0);
static DUMPED: AtomicU32 = AtomicU32::new(0);

fn hexdump(addr: u64, len: usize) -> String {
    if addr < 0x10000 {
        return format!("{addr:#x} (not a pointer)");
    }
    let mut s = format!("{addr:#x}:");
    for i in (0..len).step_by(8) {
        if i % 32 == 0 {
            s.push_str("\n          ");
        }
        let v = unsafe { ((addr + i as u64) as *const u64).read_unaligned() };
        s.push_str(&format!(" {v:016x}"));
    }
    s
}

/// Logs the builder arguments and the structures they point to (first jobs only).
unsafe fn dump_job(args: &[u64; 7]) {
    if DUMPED.fetch_add(1, Ordering::Relaxed) >= 2 {
        return;
    }
    let [batch, n_conv, conv, n_in, inputs, gains, outputs] = *args;
    let mut s = format!("acm: convolution job builder u70oWo92SYQ(batch {batch:#x}, {n_conv}, {conv:#x}, {n_in}, {inputs:#x}, {gains:#x}, {outputs:#x})");
    s.push_str(&format!("\n    convolver[0] {}", hexdump(conv, 0x60)));
    unsafe {
        if inputs >= 0x10000 {
            let p = *(inputs as *const u64);
            s.push_str(&format!("\n    input[0] -> {}", hexdump(p, 0x40)));
        }
        if gains >= 0x10000 {
            s.push_str(&format!("\n    gain[0] = {}", *(gains as *const f32)));
        }
        if outputs >= 0x10000 {
            let p = *(outputs as *const u64);
            s.push_str(&format!("\n    output[0] -> {}", hexdump(p, 0x40)));
        }
    }
    crate::info!("{s}");
}

/// Executes the records of one batch buffer descriptor.
unsafe fn run_buffer(desc: *const u64) -> Result<(), String> {
    let (data, used) = unsafe { (*desc, *desc.add(1)) };
    let mut off = 0u64;
    while off < used {
        let rec = data + off;
        let op = unsafe { *(rec as *const u64) };
        match op {
            OP_STATUS => {
                // status+8: processor load of the batch (no coprocessor time is spent).
                let status = unsafe { *((rec + 8) as *const u64) };
                if status >= 0x10000 {
                    unsafe { *((status + 8) as *mut u32) = 0 };
                }
                off += 16;
            }
            OP_DS_CONVOLVE => {
                let size = unsafe { *((rec + 8) as *const u64) };
                if size < 16 {
                    return Err(format!("bad record size {size:#x} at +{off:#x}"));
                }
                if JOBS.fetch_add(1, Ordering::Relaxed) == 0 {
                    crate::warn!("acm: convolution jobs are not computed yet (reverb wet signal missing; stand-in)");
                }
                off += size;
            }
            other => return Err(format!("unknown record opcode {other:#x} at +{off:#x} (buffer {data:#x}, {used:#x} bytes)")),
        }
    }
    Ok(())
}

sysv! {
    fn sceAcmContextCreate(out: *mut u32) -> i32 {
        if out.is_null() { return ERR_INVALID_PARAMETER; }
        *out = 1;
        crate::info!("sceAcmContextCreate -> 1");
        0
    }
    fn sceAcmContextDestroy(ctx: u32) -> i32 { 0 }
    /// (ctx, buffer count, buffer descriptor pointers, flags, out batch id)
    fn sceAcmBatchStartBuffers(ctx: u32, count: u32, descs: *const u64, flags: u32, out_id: *mut u32) -> i32 {
        if descs.is_null() { return ERR_INVALID_PARAMETER; }
        for i in 0..count as usize {
            let d = *descs.add(i) as *const u64;
            if d.is_null() { return ERR_INVALID_PARAMETER; }
            if let Err(e) = run_buffer(d) {
                crate::error!("acm: batch buffer {i}: {e}");
                return ERR_INVALID_PARAMETER;
            }
        }
        let id = NEXT_BATCH.fetch_add(1, Ordering::Relaxed);
        if !out_id.is_null() { *out_id = id; }
        0
    }
    /// (ctx, batch id, timeout in microseconds; 0 polls, !0 waits forever).
    /// Batches complete at submission, so waits always succeed.
    fn sceAcmBatchWait(ctx: u32, batch: u32, timeout: u32) -> i32 { 0 }
    /// Unnamed libSceAcm export: appends one convolution job.
    /// (batch descriptor, convolver count, convolvers, input count, input
    /// pointers, input gains, output pointers) -- argument roles inferred from
    /// the call at eboot+0x1dfcfa1.
    fn ds_acm_batch_convolution(batch: *mut u64, n_conv: u64, conv: u64, n_in: u64, inputs: u64, gains: u64, outputs: u64) -> i32 {
        if batch.is_null() { return ERR_INVALID_PARAMETER; }
        let args = [batch as u64, n_conv, conv, n_in, inputs, gains, outputs];
        dump_job(&args);
        let (data, used, cap) = (*batch, *batch.add(1), *batch.add(2));
        if used + CONVOLVE_RECORD > cap { return ERR_BUFFER_FULL; }
        let rec = (data + used) as *mut u64;
        *rec = OP_DS_CONVOLVE;
        *rec.add(1) = CONVOLVE_RECORD;
        for (i, a) in args.iter().enumerate().skip(1) {
            *rec.add(1 + i) = *a;
        }
        *batch.add(1) = used + CONVOLVE_RECORD;
        0
    }
}

pub fn register(r: &mut Registry) {
    reg!(r, "libSceAcm", [sceAcmContextCreate, sceAcmContextDestroy, sceAcmBatchStartBuffers, sceAcmBatchWait]);
    r.func_nid("libSceAcm", "u70oWo92SYQ", "u70oWo92SYQ (unnamed: convolution job builder)", ds_acm_batch_convolution as *const () as u64);
}
