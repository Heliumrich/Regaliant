//! Lane masks that must be complete in wave-mask mode.
//!
//! In wave-mask mode (see [`crate::translate`]) SGPR lane masks have the
//! hardware layout: bit i belongs to lane i of the wave. A VALU instruction
//! that writes a mask (a compare, a carry-out) computes only its own lane's
//! bit; the whole mask takes a ballot across the workgroup (a barrier).
//! Masks consumed only lane by lane (v_cndmask selectors, carry-ins) can do
//! without: each invocation keeps a *partial* mask whose own bit is right.
//!
//! This backward analysis finds the mask writes whose value may be read as a
//! whole (scalar arithmetic and compares, branches, SCC, copies to VGPRs or
//! memory) or reach EXEC, through bitwise scalar ops and moves: those writes
//! must produce the complete mask. Reads it does not recognize count as
//! whole reads (more ballots, never wrong masks); writes it does not
//! recognize keep the old value live.

use crate::uniform::Cfg;
use crate::{bits, Enc, Ext, Inst, OP_EXECZ, OP_EXEC_LO, OP_NULL, OP_SCC, OP_VCCZ, OP_VCC_LO, OP_VGPR0};
use crate::HashSet;

/// SGPRs 0..127 and SCC.
const SCC: usize = 128;
const N: usize = 129;
const EXEC: usize = OP_EXEC_LO as usize;

type Live = [bool; N];

/// Register effects of one instruction.
#[derive(Default)]
struct Fx {
    /// Registers whose whole value is read.
    whole: Vec<usize>,
    /// (source, destination): the destination is a bitwise function of the
    /// source, lane by lane (SALU logic ops, moves, SCC of logic ops).
    prop: Vec<(usize, usize)>,
    /// Registers written.
    defs: Vec<usize>,
    /// VALU write of a lane mask (its pc is a candidate).
    mask: bool,
}

fn sgpr(op: u16) -> bool {
    op < 128 && op != OP_NULL
}

impl Fx {
    /// Whole read of operand `op` (`w` consecutive registers for SGPRs).
    fn read(&mut self, op: u16, w: u16) {
        match op {
            _ if sgpr(op) => self.whole.extend((op..(op + w).min(128)).map(usize::from)),
            OP_VCCZ => self.whole.extend([OP_VCC_LO as usize, OP_VCC_LO as usize + 1]),
            OP_EXECZ => self.whole.extend([EXEC, EXEC + 1]),
            OP_SCC => self.whole.push(SCC),
            _ => {}
        }
    }

    fn def(&mut self, op: u16, w: u16) {
        if sgpr(op) {
            self.defs.extend((op..(op + w).min(128)).map(usize::from));
        }
    }

    /// `dst` (w registers) is a bitwise function of `src`.
    fn prop(&mut self, src: u16, dst: u16, w: u16) {
        if !sgpr(src) {
            self.read(src, w);
            return;
        }
        for k in 0..w {
            if src + k < 128 && sgpr(dst) && dst + k < 128 {
                self.prop.push(((src + k) as usize, (dst + k) as usize));
            }
        }
    }

    /// SCC = (bitwise result != 0): a whole read when SCC is used.
    fn prop_scc(&mut self, src: u16, w: u16) {
        if sgpr(src) {
            for k in 0..w.min(128 - src) {
                self.prop.push(((src + k) as usize, SCC));
            }
        }
    }
}

/// Scalar register sources of a VALU instruction (SDWA scalar operands
/// included; DPP sources are VGPRs).
fn valu_srcs(ins: &Inst) -> Vec<u16> {
    match ins.enc {
        Enc::Vop3 | Enc::Vop3p => vec![ins.src(0), ins.src(1), ins.src(2)],
        _ => match ins.ext {
            Ext::Sdwa(d) => {
                let s0 = if bits(d, 23, 1) != 0 { bits(d, 0, 8) as u16 } else { OP_VGPR0 };
                let s1 = if ins.enc != Enc::Vop1 && bits(d, 31, 1) != 0 { bits(ins.w0, 9, 8) as u16 } else { OP_VGPR0 };
                vec![s0, s1]
            }
            Ext::Dpp16(_) | Ext::Dpp8(..) => Vec::new(),
            Ext::None => vec![ins.src(0)],
        },
    }
}

fn width(name: &str) -> u16 {
    if name.contains("64") {
        2
    } else {
        1
    }
}

fn effects(ins: &Inst, wave64: bool) -> Fx {
    let mut fx = Fx::default();
    let name = ins.base_name().unwrap_or("");
    let mw = if wave64 { 2 } else { 1 };
    match ins.enc {
        Enc::Sop2 => {
            let wide = name.ends_with("_b64") || name.ends_with("_u64") || name.ends_with("_i64");
            let w = if wide { 2 } else { 1 };
            let (d, a, b) = (ins.sdst(), ins.ssrc0(), ins.ssrc1());
            let logic = ["s_and_", "s_or_", "s_xor_", "s_andn2_", "s_orn2_", "s_nand_", "s_nor_", "s_xnor_"].iter().any(|p| name.starts_with(p));
            if logic {
                for s in [a, b] {
                    fx.prop(s, d, w);
                    fx.prop_scc(s, w);
                }
                fx.def(d, w);
                fx.defs.push(SCC);
            } else if name.starts_with("s_cselect_") {
                fx.whole.push(SCC);
                fx.prop(a, d, w);
                fx.prop(b, d, w);
                fx.def(d, w);
            } else {
                fx.read(a, w);
                fx.read(b, w);
                if matches!(name, "s_addc_u32" | "s_subb_u32") {
                    fx.whole.push(SCC);
                }
                fx.def(d, w);
                if !crate::uniform::keeps_scc(name) {
                    fx.defs.push(SCC);
                }
            }
        }
        Enc::Sop1 => {
            let (d, s) = (ins.sdst(), ins.ssrc0());
            let w = width(name);
            if name.contains("saveexec") || name.contains("wrexec") {
                // EXEC is always complete: its sources are read whole.
                fx.read(s, w);
                fx.read(OP_EXEC_LO, w);
                fx.def(d, w);
                fx.def(OP_EXEC_LO, w);
                fx.defs.push(SCC);
            } else if matches!(name, "s_mov_b32" | "s_mov_b64") {
                fx.prop(s, d, w);
                fx.def(d, w);
            } else if matches!(name, "s_cmov_b32" | "s_cmov_b64") {
                // Conditional: the old value stays live.
                fx.whole.push(SCC);
                fx.prop(s, d, w);
            } else if matches!(name, "s_not_b32" | "s_not_b64") {
                fx.prop(s, d, w);
                fx.prop_scc(s, w);
                fx.def(d, w);
                fx.defs.push(SCC);
            } else if name.starts_with("s_movrel") {
                // Relatively addressed registers: anything may be read.
                fx.whole.extend(0..128);
            } else {
                fx.read(s, w);
                if name.starts_with("s_bitset") {
                    fx.read(d, w);
                }
                let dw = if name.starts_with("s_bcnt") || name.starts_with("s_ff") || name.starts_with("s_flbit") {
                    1
                } else if matches!(name, "s_setpc_b64" | "s_rfe_b64") {
                    0
                } else {
                    w
                };
                fx.def(d, dw);
                if !crate::uniform::keeps_scc(name) {
                    fx.defs.push(SCC);
                }
            }
        }
        Enc::Sopc => {
            let w = width(name);
            fx.read(ins.ssrc0(), w);
            fx.read(ins.ssrc1(), w);
            if name.starts_with("s_cmp") || name.starts_with("s_bitcmp") {
                fx.defs.push(SCC);
            }
        }
        Enc::Sopk => {
            let d = ins.sdst();
            match name {
                "s_movk_i32" | "s_getreg_b32" => fx.def(d, 1),
                "s_cmovk_i32" => fx.whole.push(SCC),
                "s_addk_i32" => {
                    fx.read(d, 1);
                    fx.def(d, 1);
                    fx.defs.push(SCC);
                }
                "s_mulk_i32" => {
                    fx.read(d, 1);
                    fx.def(d, 1);
                }
                "s_call_b64" => fx.def(d, 2),
                n if n.starts_with("s_cmpk_") => {
                    fx.read(d, 1);
                    fx.defs.push(SCC);
                }
                _ => fx.read(d, 1),
            }
        }
        Enc::Sopp => match name {
            "s_cbranch_scc0" | "s_cbranch_scc1" => fx.whole.push(SCC),
            "s_cbranch_vccz" | "s_cbranch_vccnz" => fx.read(OP_VCC_LO, mw),
            "s_cbranch_execz" | "s_cbranch_execnz" => fx.read(OP_EXEC_LO, mw),
            _ => fx.read(crate::OP_M0, 1),
        },
        Enc::Smem => {
            fx.read(ins.smem_sbase(), 2);
            fx.read(ins.smem_soffset(), 1);
            let n = [("x16", 16), ("x8", 8), ("x4", 4), ("x2", 2)].iter().find(|(s, _)| name.ends_with(s)).map(|x| x.1).unwrap_or(1);
            if name.starts_with("s_load") || name.starts_with("s_buffer_load") {
                fx.def(ins.sdst(), n);
            } else if name.starts_with("s_memtime") || name.starts_with("s_memrealtime") {
                fx.def(ins.sdst(), 2);
            } else {
                fx.read(ins.sdst(), n);
            }
        }
        Enc::Vopc => {
            for s in valu_srcs(ins) {
                fx.read(s, width(name));
            }
            if name.starts_with("v_cmpx") {
                fx.def(OP_EXEC_LO, mw);
            } else {
                fx.def(ins.sdwa_sdst().unwrap_or(OP_VCC_LO), mw);
            }
            fx.mask = true;
        }
        Enc::Vop1 => {
            for s in valu_srcs(ins) {
                fx.read(s, width(name));
            }
            if name == "v_readfirstlane_b32" {
                fx.def(ins.vdst(), 1);
            }
        }
        Enc::Vop2 => {
            for s in valu_srcs(ins) {
                fx.read(s, width(name));
            }
            // v_cndmask_b32 and the carry-ins read VCC lane by lane.
            if matches!(name, "v_addc_co_u32" | "v_subb_co_u32" | "v_subbrev_co_u32") {
                fx.def(OP_VCC_LO, mw);
                fx.mask = true;
            }
        }
        Enc::Vop3 => {
            let s = valu_srcs(ins);
            let w = width(name);
            if crate::vop3_kind(ins.op) == crate::Vop3Kind::FromVopc {
                fx.read(s[0], w);
                fx.read(s[1], w);
                fx.def(if name.starts_with("v_cmpx") { OP_EXEC_LO } else { ins.vdst() }, mw);
                fx.mask = true;
            } else if ins.is_vop3b() {
                fx.read(s[0], w);
                fx.read(s[1], w);
                if !matches!(name, "v_addc_co_u32" | "v_subb_co_u32" | "v_subbrev_co_u32") {
                    // (carry-ins are lane reads)
                    fx.read(s[2], w);
                }
                fx.def(ins.vop3_sdst(), mw);
                fx.mask = true;
            } else {
                let n = if name == "v_cndmask_b32" { 2 } else { 3 };
                for &x in &s[..n] {
                    fx.read(x, w);
                }
                if matches!(name, "v_readlane_b32" | "v_readfirstlane_b32") {
                    fx.def(ins.vdst(), 1);
                }
            }
        }
        Enc::Vop3p => {
            for s in valu_srcs(ins) {
                fx.read(s, 1);
            }
        }
        Enc::Vintrp | Enc::Ds => fx.read(crate::OP_M0, 1),
        Enc::Mubuf | Enc::Mtbuf => {
            fx.read(ins.buf_srsrc(), 4);
            fx.read(ins.buf_soffset(), 1);
        }
        Enc::Mimg => {
            fx.read(ins.mimg_srsrc(), 8);
            fx.read(ins.mimg_ssamp(), 4);
        }
        Enc::Flat | Enc::Global | Enc::Scratch => fx.read(ins.flat_saddr(), 2),
        Enc::Exp => {}
    }
    fx
}

/// The effects as far as reads of mask *values* go: branches, EXEC updates
/// (saveexec) and SCC tests of masks are left out (lane-by-lane emulation
/// keeps them right: each invocation follows its own lane).
fn value_effects(ins: &Inst, wave64: bool) -> Fx {
    let mut fx = effects(ins, wave64);
    let name = ins.base_name().unwrap_or("");
    let exec_update = ins.enc == Enc::Sop1 && (name.contains("saveexec") || name.contains("wrexec"));
    if matches!(ins.enc, Enc::Sopp | Enc::Sopc) || exec_update {
        fx.whole.clear();
    }
    fx.whole.retain(|&r| r != SCC);
    fx.prop.retain(|&(_, d)| d != SCC);
    fx
}

fn step(l: &mut Live, fx: &Fx, exec_live: bool) {
    let after = *l;
    for &d in &fx.defs {
        l[d] = false;
    }
    for &(s, d) in &fx.prop {
        if after[d] {
            l[s] = true;
        }
    }
    for &r in &fx.whole {
        l[r] = true;
    }
    if exec_live {
        l[EXEC] = true;
        l[EXEC + 1] = true;
    }
}

/// Pcs of the VALU mask writes (compares, v_cmpx, carry-outs) whose result
/// must be the complete wave mask.
pub fn complete_masks(code: &[Inst], wave64: bool) -> HashSet<u32> {
    analyze(code, wave64, false)
}

/// Whether a VALU mask write may be read as a value: copied into a VGPR or
/// memory, counted or searched (s_bcnt, s_ff), used in scalar arithmetic.
/// Outside wave-mask mode such reads come out wrong (a ballot that one lane
/// stores holds that lane's bit alone), unlike branches, EXEC updates and
/// SCC tests of masks.
pub fn reads_mask_values(code: &[Inst], wave64: bool) -> bool {
    !mask_value_writes(code, wave64).is_empty()
}

/// Pcs of the VALU mask writes whose value may be read (see
/// [`reads_mask_values`]).
pub fn mask_value_writes(code: &[Inst], wave64: bool) -> HashSet<u32> {
    analyze(code, wave64, true)
}

fn analyze(code: &[Inst], wave64: bool, values_only: bool) -> HashSet<u32> {
    let mut out = HashSet::default();
    if code.is_empty() {
        return out;
    }
    let cfg = Cfg::new(code);
    let fx: Vec<Fx> = code.iter().map(|i| if values_only { value_effects(i, wave64) } else { effects(i, wave64) }).collect();
    let nb = cfg.starts.len();
    let exit = cfg.exit();
    let mut live_in: Vec<Live> = vec![[false; N]; nb];
    let live_out = |live_in: &Vec<Live>, b: usize| {
        let mut l = [false; N];
        for &s in &cfg.succ[b] {
            if s != exit {
                for k in 0..N {
                    l[k] |= live_in[s][k];
                }
            }
        }
        l
    };
    loop {
        let mut changed = false;
        for b in (0..nb).rev() {
            let mut l = live_out(&live_in, b);
            for i in (cfg.starts[b]..cfg.end(b)).rev() {
                step(&mut l, &fx[i], !values_only);
            }
            for k in 0..N {
                if l[k] && !live_in[b][k] {
                    live_in[b][k] = true;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    for b in 0..nb {
        let mut l = live_out(&live_in, b);
        for i in (cfg.starts[b]..cfg.end(b)).rev() {
            if fx[i].mask && fx[i].defs.iter().any(|&d| l[d]) {
                out.insert(code[i].pc);
            }
            step(&mut l, &fx[i], !values_only);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn enc_vopc(op: u32, src0: u32, vsrc1: u32) -> u32 {
        (0x3E << 25) | (op << 17) | (vsrc1 << 9) | src0
    }
    fn enc_vop2(op: u32, vdst: u32, src0: u32, vsrc1: u32) -> u32 {
        (op << 25) | (vdst << 17) | (vsrc1 << 9) | src0
    }
    fn enc_vop1(op: u32, vdst: u32, src0: u32) -> u32 {
        (0x3F << 25) | (vdst << 17) | (op << 9) | src0
    }
    fn enc_sop1(op: u32, sdst: u32, ssrc0: u32) -> u32 {
        (0x17D << 23) | (sdst << 16) | (op << 8) | ssrc0
    }
    fn enc_sop2(op: u32, sdst: u32, s0: u32, s1: u32) -> u32 {
        (0b10 << 30) | (op << 23) | (sdst << 16) | (s1 << 8) | s0
    }
    fn enc_sopp(op: u32, simm: u16) -> u32 {
        (0x17F << 23) | (op << 16) | simm as u32
    }

    fn opcode(table: &[(u16, &str)], name: &str) -> u32 {
        table.iter().find(|(_, n)| n.split('/').next() == Some(name)).unwrap_or_else(|| panic!("{name}")).0 as u32
    }

    fn program(words: &[u32]) -> Vec<Inst> {
        crate::decode_program(words).unwrap()
    }

    #[test]
    fn select_only_masks_stay_partial() {
        use crate::tables::*;
        let cmp = opcode(VOPC, "v_cmp_lt_f32");
        let cnd = opcode(VOP2, "v_cndmask_b32");
        let and_sx = opcode(SOP1, "s_and_saveexec_b64");
        let endpgm = 1;
        // v_cmp_lt_f32 vcc, v0, v1 ; v_cndmask_b32 v2, v0, v1, vcc   -> partial
        // v_cmp_lt_f32 vcc, v1, v0 ; s_and_saveexec_b64 s[4:5], vcc   -> complete
        let code = program(&[
            enc_vopc(cmp, 256, 1),
            enc_vop2(cnd, 2, 256, 1),
            enc_vopc(cmp, 257, 0),
            enc_sop1(and_sx, 4, 106),
            enc_sopp(endpgm, 0),
        ]);
        let c = complete_masks(&code, true);
        assert!(!c.contains(&code[0].pc));
        assert!(c.contains(&code[2].pc));
    }

    #[test]
    fn masks_read_as_values_are_complete() {
        use crate::tables::*;
        let cmp = opcode(VOPC, "v_cmp_lt_f32");
        let and64 = opcode(SOP2, "s_and_b64");
        let bcnt = opcode(SOP1, "s_bcnt1_i32_b64");
        let cnd = opcode(VOP2, "v_cndmask_b32");
        let cbr = opcode(SOPP, "s_cbranch_scc1");
        let endpgm = 1;
        // v_cmp vcc ; s_and_b64 s[2:3], vcc, exec ; s_bcnt1 s6, s[2:3]  -> complete
        let code = program(&[enc_vopc(cmp, 256, 1), enc_sop2(and64, 2, 106, 126), enc_sop1(bcnt, 6, 2), enc_sopp(endpgm, 0)]);
        assert!(complete_masks(&code, true).contains(&code[0].pc));
        // v_cmp vcc ; s_and_b64 vcc, vcc, exec ; v_cndmask (lane read) ; branch on SCC of the and -> complete
        let code = program(&[enc_vopc(cmp, 256, 1), enc_sop2(and64, 106, 106, 126), enc_vop2(cnd, 2, 256, 1), enc_sopp(cbr, 0), enc_sopp(endpgm, 0)]);
        assert!(complete_masks(&code, true).contains(&code[0].pc));
        // Without the SCC branch the and result is only selected from: partial.
        let code = program(&[enc_vopc(cmp, 256, 1), enc_sop2(and64, 106, 106, 126), enc_vop2(cnd, 2, 256, 1), enc_sopp(endpgm, 0)]);
        assert!(!complete_masks(&code, true).contains(&code[0].pc));
    }

    #[test]
    fn mask_values_need_wave_masks() {
        use crate::tables::*;
        let cmp = opcode(VOPC, "v_cmp_lt_f32");
        let mov = opcode(VOP1, "v_mov_b32");
        let and64 = opcode(SOP2, "s_and_b64");
        let and_sx = opcode(SOP1, "s_and_saveexec_b64");
        let bcnt = opcode(SOP1, "s_bcnt1_i32_b64");
        let cbr_scc = opcode(SOPP, "s_cbranch_scc1");
        let cbr_exec = opcode(SOPP, "s_cbranch_execz");
        let endpgm = 1;
        // A ballot copied into a VGPR (to be stored): v_cmp vcc ; v_mov_b32 v0, vcc_lo
        let code = program(&[enc_vopc(cmp, 256, 1), enc_vop1(mov, 0, 106), enc_sopp(endpgm, 0)]);
        assert!(reads_mask_values(&code, true));
        // Counted: v_cmp vcc ; s_and_b64 s[2:3], vcc, exec ; s_bcnt1 s6, s[2:3]
        let code = program(&[enc_vopc(cmp, 256, 1), enc_sop2(and64, 2, 106, 126), enc_sop1(bcnt, 6, 2), enc_sopp(endpgm, 0)]);
        assert!(reads_mask_values(&code, true));
        // Branches only: v_cmp vcc ; s_and_saveexec_b64 s[4:5], vcc ; s_cbranch_execz
        let code = program(&[enc_vopc(cmp, 256, 1), enc_sop1(and_sx, 4, 106), enc_sopp(cbr_exec, 0), enc_sopp(endpgm, 0)]);
        assert!(!reads_mask_values(&code, true));
        assert!(complete_masks(&code, true).contains(&code[0].pc));
        // An SCC test: v_cmp vcc ; s_and_b64 vcc, vcc, exec ; s_cbranch_scc1
        let code = program(&[enc_vopc(cmp, 256, 1), enc_sop2(and64, 106, 106, 126), enc_sopp(cbr_scc, 0), enc_sopp(endpgm, 0)]);
        assert!(!reads_mask_values(&code, true));
    }

    #[test]
    fn loop_carried_mask_is_complete() {
        use crate::tables::*;
        let cmp = opcode(VOPC, "v_cmp_lt_f32");
        let or64 = opcode(SOP2, "s_or_b64");
        let andn2 = opcode(SOP2, "s_andn2_b64");
        let br = opcode(SOPP, "s_cbranch_execnz");
        let endpgm = 1;
        // loop: v_cmp vcc ; s_or_b64 s[2:3], vcc, s[2:3] ; s_andn2_b64 exec, exec, s[2:3] ; s_cbranch_execnz loop
        let code = program(&[enc_vopc(cmp, 256, 1), enc_sop2(or64, 2, 106, 2), enc_sop2(andn2, 126, 126, 2), enc_sopp(br, (-4i16) as u16), enc_sopp(endpgm, 0)]);
        assert!(complete_masks(&code, true).contains(&code[0].pc));
    }
}
