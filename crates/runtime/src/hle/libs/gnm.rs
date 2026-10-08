//! libSceGnmDriver: command-buffer builder functions (emit PM4 register writes
//! into guest command buffers, NOP-padded to the reserved size), submission,
//! graphics events and compute queues.

use crate::gpu::pm4::{self, Writer, marker};
use crate::gpu::{self, FlipRequest, Submission};
use crate::hle::{Registry, cstr};
use crate::{fixed, reg, sysv};

const TESS_FACTOR_RING: u64 = 0xFF_0000_0000;
const ERR_INVALID: i32 = 0x8025_0001u32 as i32;

// Shader stage registers (gfx7 SH register offsets).
const SPI_SHADER_PGM_LO_PS: u32 = 0x2C08;
const SPI_SHADER_PGM_RSRC1_PS: u32 = 0x2C0A;
const SPI_SHADER_PGM_LO_VS: u32 = 0x2C48;
const SPI_SHADER_PGM_RSRC1_VS: u32 = 0x2C4A;
const SPI_SHADER_PGM_LO_GS: u32 = 0x2C88;
const SPI_SHADER_PGM_RSRC1_GS: u32 = 0x2C8A;
const SPI_SHADER_PGM_LO_ES: u32 = 0x2CC8;
const SPI_SHADER_PGM_RSRC1_ES: u32 = 0x2CCA;
const SPI_SHADER_PGM_LO_HS: u32 = 0x2D08;
const SPI_SHADER_PGM_RSRC1_HS: u32 = 0x2D0A;
const SPI_SHADER_PGM_LO_LS: u32 = 0x2D48;
const SPI_SHADER_PGM_RSRC1_LS: u32 = 0x2D4A;
const COMPUTE_NUM_THREAD_X: u32 = 0x2E07;
const COMPUTE_PGM_LO: u32 = 0x2E0C;
const COMPUTE_PGM_RSRC1: u32 = 0x2E12;
// Context registers.
const CB_SHADER_MASK: u32 = 0xA08F;
const SPI_VS_OUT_CONFIG: u32 = 0xA1B1;
const SPI_PS_INPUT_ENA: u32 = 0xA1B3;
const SPI_PS_IN_CONTROL: u32 = 0xA1B6;
const SPI_BARYC_CNTL: u32 = 0xA1B8;
const SPI_SHADER_POS_FORMAT: u32 = 0xA1C3;
const SPI_SHADER_Z_FORMAT: u32 = 0xA1C4;
const DB_SHADER_CONTROL: u32 = 0xA203;
const PA_CL_VS_OUT_CNTL: u32 = 0xA207;
const VGT_GS_OUT_PRIM_TYPE: u32 = 0xA29B;

fn modified_rsrc1(rsrc1: u32, modifier: u32) -> u32 {
    if modifier == 0 { rsrc1 } else { (rsrc1 & 0xFCFF_FC3F) | modifier }
}

unsafe fn regs<'a>(p: *const u32, n: usize) -> Option<&'a [u32]> {
    if p.is_null() { None } else { Some(unsafe { std::slice::from_raw_parts(p, n) }) }
}

fn fill_default_state(cmd: *mut u32, n: u32, tag: u32) -> u32 {
    let size = n.min(0x100);
    if cmd.is_null() || size < 4 {
        return 0;
    }
    let mut w = unsafe { Writer::new(cmd, size) };
    w.packet(pm4::op::CONTEXT_CONTROL, &[0x8000_0000, 0x8000_0000]);
    w.marker(tag, "default-state");
    w.finish()
}

sysv! {
    fn sceGnmDrawInitDefaultHardwareState200(cmd: *mut u32, n: u32) -> u32 { fill_default_state(cmd, n, marker::INIT_DEFAULT) }
    fn sceGnmDispatchInitDefaultHardwareState(cmd: *mut u32, n: u32) -> u32 { fill_default_state(cmd, n, marker::INIT_DEFAULT) }

    fn sceGnmSetVsShader(cmd: *mut u32, n: u32, vs: *const u32, modifier: u32) -> i32 {
        let Some(v) = regs(vs, 7) else { return ERR_INVALID };
        let mut w = Writer::new(cmd, n);
        let ok = w.set_sh(SPI_SHADER_PGM_LO_VS, &[v[0], v[1]])
            && w.set_sh(SPI_SHADER_PGM_RSRC1_VS, &[modified_rsrc1(v[2], modifier), v[3]])
            && w.set_ctx(SPI_VS_OUT_CONFIG, &[v[4]])
            && w.set_ctx(SPI_SHADER_POS_FORMAT, &[v[5]])
            && w.set_ctx(PA_CL_VS_OUT_CNTL, &[v[6]]);
        if !ok { return ERR_INVALID; }
        w.finish();
        0
    }
    fn sceGnmUpdateVsShader(cmd: *mut u32, n: u32, vs: *const u32, modifier: u32) -> i32 { sceGnmSetVsShader(cmd, n, vs, modifier) }
    fn sceGnmSetPsShader(cmd: *mut u32, n: u32, ps: *const u32) -> i32 {
        let mut w = Writer::new(cmd, n);
        let ok = match regs(ps, 12) {
            Some(p) => w.set_sh(SPI_SHADER_PGM_LO_PS, &[p[0], p[1]])
                && w.set_sh(SPI_SHADER_PGM_RSRC1_PS, &[p[2], p[3]])
                && w.set_ctx(SPI_SHADER_Z_FORMAT, &[p[4], p[5]])
                && w.set_ctx(SPI_PS_INPUT_ENA, &[p[6], p[7]])
                && w.set_ctx(SPI_PS_IN_CONTROL, &[p[8]])
                && w.set_ctx(SPI_BARYC_CNTL, &[p[9]])
                && w.set_ctx(DB_SHADER_CONTROL, &[p[10]])
                && w.set_ctx(CB_SHADER_MASK, &[p[11]]),
            None => w.set_sh(SPI_SHADER_PGM_LO_PS, &[0, 0])
                && w.set_ctx(SPI_SHADER_Z_FORMAT, &[0, 0])
                && w.set_ctx(CB_SHADER_MASK, &[0]),
        };
        if !ok { return ERR_INVALID; }
        w.finish();
        0
    }
    fn sceGnmUpdatePsShader(cmd: *mut u32, n: u32, ps: *const u32) -> i32 { sceGnmSetPsShader(cmd, n, ps) }
    fn sceGnmSetCsShader(cmd: *mut u32, n: u32, cs: *const u32) -> i32 {
        let Some(c) = regs(cs, 7) else { return ERR_INVALID };
        let mut w = Writer::new(cmd, n);
        let ok = w.set_sh(COMPUTE_PGM_LO, &[c[0], c[1]])
            && w.set_sh(COMPUTE_PGM_RSRC1, &[c[2], c[3]])
            && w.set_sh(COMPUTE_NUM_THREAD_X, &[c[4], c[5], c[6]]);
        if !ok { return ERR_INVALID; }
        w.finish();
        0
    }
    fn sceGnmSetEsShader(cmd: *mut u32, n: u32, es: *const u32, modifier: u32) -> i32 {
        let Some(e) = regs(es, 4) else { return ERR_INVALID };
        let mut w = Writer::new(cmd, n);
        let ok = w.set_sh(SPI_SHADER_PGM_LO_ES, &[e[0], e[1]]) && w.set_sh(SPI_SHADER_PGM_RSRC1_ES, &[modified_rsrc1(e[2], modifier), e[3]]);
        if !ok { return ERR_INVALID; }
        w.finish();
        0
    }
    fn sceGnmSetLsShader(cmd: *mut u32, n: u32, ls: *const u32, modifier: u32) -> i32 {
        let Some(l) = regs(ls, 4) else { return ERR_INVALID };
        let mut w = Writer::new(cmd, n);
        let ok = w.set_sh(SPI_SHADER_PGM_LO_LS, &[l[0], l[1]]) && w.set_sh(SPI_SHADER_PGM_RSRC1_LS, &[modified_rsrc1(l[2], modifier), l[3]]);
        if !ok { return ERR_INVALID; }
        w.finish();
        0
    }
    fn sceGnmSetHsShader(cmd: *mut u32, n: u32, hs: *const u32, tess: *const u32) -> i32 {
        let Some(h) = regs(hs, 4) else { return ERR_INVALID };
        let mut w = Writer::new(cmd, n);
        let ok = w.set_sh(SPI_SHADER_PGM_LO_HS, &[h[0], h[1]]) && w.set_sh(SPI_SHADER_PGM_RSRC1_HS, &[h[2], h[3]]);
        if !ok { return ERR_INVALID; }
        w.finish();
        0
    }
    fn sceGnmUpdateHsShader(cmd: *mut u32, n: u32, hs: *const u32, tess: *const u32) -> i32 { sceGnmSetHsShader(cmd, n, hs, tess) }
    fn sceGnmSetGsShader(cmd: *mut u32, n: u32, gs: *const u32) -> i32 {
        let Some(g) = regs(gs, 7) else { return ERR_INVALID };
        let mut w = Writer::new(cmd, n);
        let ok = w.set_sh(SPI_SHADER_PGM_LO_GS, &[g[0], g[1]])
            && w.set_sh(SPI_SHADER_PGM_RSRC1_GS, &[g[2], g[3]])
            && w.set_ctx(VGT_GS_OUT_PRIM_TYPE, &[g[5]]);
        if !ok { return ERR_INVALID; }
        w.finish();
        0
    }
    fn sceGnmUpdateGsShader(cmd: *mut u32, n: u32, gs: *const u32) -> i32 { sceGnmSetGsShader(cmd, n, gs) }

    fn sceGnmInsertPushMarker(cmd: *mut u32, n: u32, text: *const u8) -> i32 {
        let mut w = Writer::new(cmd, n);
        w.marker(marker::PUSH, cstr(text));
        w.finish();
        0
    }
    fn sceGnmInsertPushColorMarker(cmd: *mut u32, n: u32, text: *const u8, color: u32) -> i32 {
        let mut w = Writer::new(cmd, n);
        w.marker(marker::PUSH, cstr(text));
        w.finish();
        0
    }
    fn sceGnmInsertPopMarker(cmd: *mut u32, n: u32) -> i32 {
        let mut w = Writer::new(cmd, n);
        w.packet(pm4::op::NOP, &[marker::POP]);
        w.finish();
        0
    }
    fn sceGnmInsertSetMarker(cmd: *mut u32, n: u32, text: *const u8) -> i32 {
        let mut w = Writer::new(cmd, n);
        w.marker(marker::SET, cstr(text));
        w.finish();
        0
    }
    fn sceGnmInsertSetColorMarker(cmd: *mut u32, n: u32, text: *const u8, color: u32) -> i32 {
        let mut w = Writer::new(cmd, n);
        w.marker(marker::SET, cstr(text));
        w.finish();
        0
    }
    fn sceGnmInsertWaitFlipDone(cmd: *mut u32, n: u32, handle: i32, index: i32) -> i32 {
        let mut w = Writer::new(cmd, n);
        if !w.packet(pm4::op::NOP, &[marker::WAIT_FLIP, handle as u32, index as u32]) { return ERR_INVALID; }
        w.finish();
        0
    }

    fn sceGnmSubmitCommandBuffers(count: u32, dcb: *const u64, dcb_sizes: *const u32, ccb: *const u64, ccb_sizes: *const u32) -> i32 {
        submit(count, dcb, dcb_sizes, ccb, ccb_sizes, None)
    }
    fn sceGnmSubmitAndFlipCommandBuffers(count: u32, dcb: *const u64, dcb_sizes: *const u32, ccb: *const u64, ccb_sizes: *const u32, handle: i32, index: i32, mode: u32, arg: i64) -> i32 {
        submit(count, dcb, dcb_sizes, ccb, ccb_sizes, Some(FlipRequest { handle, index, mode, arg }))
    }
    fn sceGnmAddEqEvent(eq: u64, id: u64, udata: u64) -> i32 {
        gpu::add_eop_event(eq, id);
        crate::info!("sceGnmAddEqEvent(eq {eq:#x}, id {id:#x})");
        0
    }
    fn sceGnmDeleteEqEvent(eq: u64, id: u64) -> i32 { gpu::remove_eop_event(eq, id); 0 }
    fn sceGnmGetEqEventType(ev: *const u64) -> u32 { if ev.is_null() { 0 } else { *ev as u32 } }
    fn sceGnmGetEqTimeStamp(ev: *const u64) -> u64 { if ev.is_null() { 0 } else { *ev.add(2) } }
    fn sceGnmMapComputeQueue(pipe: u32, queue: u32, base: u64, size_dw: u32, rptr: u64) -> i32 {
        gpu::map_compute_queue(pipe, queue, base, size_dw, rptr)
    }
    fn sceGnmMapComputeQueueWithPriority(pipe: u32, queue: u32, base: u64, size_dw: u32, rptr: u64, prio: u32) -> i32 {
        gpu::map_compute_queue(pipe, queue, base, size_dw, rptr)
    }
    fn sceGnmDingDong(vq: u32, next_dw: u32) { gpu::ding_dong(vq, next_dw) }
    fn sceGnmLogicalCuMaskToPhysicalCuMask(mask: u32) -> u32 { mask }
    fn sceGnmGetTheTessellationFactorRingBufferBaseAddress() -> u64 { TESS_FACTOR_RING }
}

/// Defensive bound on DCBs per submission (guards against garbage counts).
/// Titles submit one DCB per recorded command-list segment; a PS4 title's
/// end-of-frame submitter does not split its list, so the bound must be well
/// above typical counts (an earlier cap of 32 silently dropped whole frames'
/// worth of command lists, leaking the title's command-memory pool).
const MAX_SUBMIT_DCBS: u32 = 4096;

fn submit(count: u32, dcb: *const u64, dcb_sizes: *const u32, ccb: *const u64, ccb_sizes: *const u32, flip: Option<FlipRequest>) -> i32 {
    if dcb.is_null() || dcb_sizes.is_null() || count == 0 || count > MAX_SUBMIT_DCBS {
        crate::error!("sceGnmSubmit*CommandBuffers rejected: count {count}, dcb {dcb:?}, sizes {dcb_sizes:?}");
        return ERR_INVALID;
    }
    let mut s = Submission { dcb: Vec::new(), ccb: Vec::new(), flip, captured: Default::default() };
    unsafe {
        for i in 0..count as usize {
            s.dcb.push((*dcb.add(i), *dcb_sizes.add(i) / 4));
            if !ccb.is_null() && !ccb_sizes.is_null() {
                s.ccb.push((*ccb.add(i), *ccb_sizes.add(i) / 4));
            } else {
                s.ccb.push((0, 0));
            }
        }
    }
    if std::env::var_os("BB_SUBMIT_LOG").is_some() {
        let total: u64 = s.dcb.iter().map(|d| d.1 as u64 * 4).sum();
        crate::info!("submit {} dcb(s) {} bytes{}: {:x?}", s.dcb.len(), total, if s.flip.is_some() { " +flip" } else { "" }, s.dcb.iter().take(8).map(|d| d.0).collect::<Vec<_>>());

    }
    if let Some(f) = &s.flip {
        crate::hle::libs::videoout::gpu_flip_requested(f.handle);
    }
    gpu::submit(s);
    0
}

fixed! {
    sceGnmSubmitDone => 0,
    sceGnmFlushGarlic => 0,
    sceGnmValidateCommandBuffers => 0,
    sceGnmDebugHardwareStatus => 0,
    sceGnmUnmapComputeQueue => 0,
    sceGnmSetupMipStatsReport => 0,
    sceGnmDisableMipStatsReport => 0,
    sceGnmSetGsRingSizes => 0,
    sceGnmIsUserPaEnabled => 0,
    sceGnmDriverCaptureInProgress => 0,
    sceGnmDriverTriggerCapture => 0,
    sceRazorIsLoaded => 0,
    sceRazorCaptureImmediate => 0,
    sceRazorCaptureCommandBuffersOnlyImmediate => 0,
    sceRazorCaptureSinceLastFlip => 0,
    sceRazorCaptureCommandBuffersOnlySinceLastFlip => 0,
}

pub fn register(r: &mut Registry) {
    reg!(r, "libSceGnmDriver", [
        sceGnmDrawInitDefaultHardwareState200, sceGnmDispatchInitDefaultHardwareState, sceGnmSetVsShader,
        sceGnmUpdateVsShader, sceGnmSetPsShader, sceGnmUpdatePsShader, sceGnmSetCsShader, sceGnmSetEsShader,
        sceGnmSetLsShader, sceGnmSetHsShader, sceGnmUpdateHsShader, sceGnmSetGsShader, sceGnmUpdateGsShader,
        sceGnmInsertPushMarker, sceGnmInsertPushColorMarker, sceGnmInsertPopMarker, sceGnmInsertSetMarker,
        sceGnmInsertSetColorMarker, sceGnmInsertWaitFlipDone, sceGnmSubmitCommandBuffers,
        sceGnmSubmitAndFlipCommandBuffers, sceGnmAddEqEvent, sceGnmDeleteEqEvent, sceGnmGetEqEventType,
        sceGnmGetEqTimeStamp, sceGnmMapComputeQueue, sceGnmMapComputeQueueWithPriority, sceGnmDingDong,
        sceGnmLogicalCuMaskToPhysicalCuMask, sceGnmGetTheTessellationFactorRingBufferBaseAddress, sceGnmSubmitDone,
        sceGnmFlushGarlic, sceGnmValidateCommandBuffers, sceGnmDebugHardwareStatus, sceGnmUnmapComputeQueue,
        sceGnmSetupMipStatsReport, sceGnmDisableMipStatsReport, sceGnmSetGsRingSizes, sceGnmIsUserPaEnabled,
        sceGnmDriverCaptureInProgress, sceGnmDriverTriggerCapture, sceRazorIsLoaded, sceRazorCaptureImmediate,
        sceRazorCaptureCommandBuffersOnlyImmediate, sceRazorCaptureSinceLastFlip,
        sceRazorCaptureCommandBuffersOnlySinceLastFlip,
    ]);
}
