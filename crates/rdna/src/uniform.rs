//! Uniform control flow analysis for the wave-of-one translation model.
//!
//! In the translated code every invocation (lane) takes scalar branches on its
//! own copy of the scalar state. Branches on values that are the same for all
//! lanes of a workgroup (user data, scalar loads, constants, the workgroup id)
//! keep the invocations together; branches on lane masks (EXEC, VCC, compare
//! results) or on SCC derived from them can split them. A block is *uniform*
//! when every invocation of the workgroup executes it the same number of times
//! (it is not control dependent on a divergent branch), so workgroup barriers
//! may be placed there.
//!
//! v_readlane / v_readfirstlane results are treated as uniform (they are on
//! hardware, and the translator makes v_readlane exact in uniform blocks).

use crate::{Enc, Inst};
use std::collections::{HashMap, HashSet};

/// Basic blocks of a program: per instruction its block, and per block
/// whether it is uniform.
pub struct Uniformity {
    pub block_of: HashMap<u32, usize>,
    pub uniform: Vec<bool>,
    /// Conditional branches whose condition depends on lane state.
    pub divergent_branches: HashSet<u32>,
}

impl Uniformity {
    /// Whether the instruction at `pc` runs in a uniform block.
    pub fn is_uniform(&self, pc: u32) -> bool {
        self.block_of.get(&pc).is_some_and(|&b| self.uniform[b])
    }

    /// Every block uniform: the translation decides lane-dependent branches
    /// for the whole workgroup (lockstep execution of a single-wave group).
    pub fn set_lockstep(&mut self) {
        self.uniform.iter_mut().for_each(|u| *u = true);
    }
}

const SCC: usize = 128;
const VCC: usize = crate::OP_VCC_LO as usize;
const EXEC: usize = crate::OP_EXEC_LO as usize;

/// Lane dependence of SGPRs 0..127 and SCC (index 128).
type Dep = [bool; 129];

fn dep_of(st: &Dep, op: u16, wide: bool) -> bool {
    match op {
        0..=127 => st[op as usize] || (wide && op < 127 && st[op as usize + 1]),
        crate::OP_SCC => st[SCC],
        crate::OP_VCCZ => st[VCC] || st[VCC + 1],
        crate::OP_EXECZ => st[EXEC] || st[EXEC + 1],
        _ => false,
    }
}

fn set(st: &mut Dep, r: u16, v: bool, wide: bool) {
    if (r as usize) < 128 && r != crate::OP_NULL {
        st[r as usize] = v;
        if wide && r < 127 {
            st[r as usize + 1] = v;
        }
    }
}

/// Whether a scalar instruction writes SCC.
pub fn writes_scc(ins: &Inst) -> bool {
    let name = ins.base_name().unwrap_or("");
    match ins.enc {
        Enc::Sopc => true,
        Enc::Sop1 | Enc::Sop2 => !keeps_scc(name),
        Enc::Sopk => name.starts_with("s_cmpk_") || name == "s_addk_i32",
        _ => false,
    }
}

/// Scalar ops that do not write SCC.
fn keeps_scc(name: &str) -> bool {
    name.starts_with("s_mov")
        || name.starts_with("s_cmov")
        || name.starts_with("s_cselect")
        || name.starts_with("s_mul")
        || name.starts_with("s_bfm")
        || name.starts_with("s_pack")
        || name.starts_with("s_brev")
        || name.starts_with("s_getpc")
        || name.starts_with("s_sext")
        || name.starts_with("s_bitset")
        || name.starts_with("s_movrel")
        || name.starts_with("s_ff")
        || name.starts_with("s_flbit")
        || name.starts_with("s_bitreplicate")
}

fn step(st: &mut Dep, ins: &Inst) {
    let name = ins.base_name().unwrap_or("");
    let wide_name = name.ends_with("_b64") || name.ends_with("_u64") || name.ends_with("_i64");
    match ins.enc {
        Enc::Sop1 => {
            let d = ins.sdst();
            let s = dep_of(st, ins.ssrc0(), wide_name);
            if name.contains("saveexec") || name.contains("wrexec") {
                let old = st[EXEC] || st[EXEC + 1];
                let new = old || s;
                set(st, d, if name.contains("wrexec") { new } else { old }, wide_name);
                st[EXEC] = new;
                st[EXEC + 1] = new;
                st[SCC] = new;
                return;
            }
            let v = s || (name.starts_with("s_cmov") && st[SCC]);
            let wide_dst = wide_name && !name.starts_with("s_bcnt") && !name.starts_with("s_ff") && !name.starts_with("s_flbit");
            set(st, d, v, wide_dst || name == "s_getpc_b64" || name == "s_bitreplicate_b64_b32");
            if !keeps_scc(name) {
                st[SCC] = s;
            }
        }
        Enc::Sop2 => {
            let d = ins.sdst();
            let shift = name.contains("lsh") || name.contains("ashr") || name.contains("bfe_") || name.starts_with("s_bfm");
            let a = dep_of(st, ins.ssrc0(), wide_name && !name.starts_with("s_bfm"));
            let b = dep_of(st, ins.ssrc1(), wide_name && !shift);
            let carry = name.starts_with("s_cselect") || name == "s_addc_u32" || name == "s_subb_u32";
            let v = a || b || (carry && st[SCC]);
            set(st, d, v, wide_name);
            if !keeps_scc(name) {
                st[SCC] = v;
            }
        }
        Enc::Sopc => {
            let wide = name.ends_with("_u64") || name.ends_with("_b64");
            st[SCC] = dep_of(st, ins.ssrc0(), wide) || dep_of(st, ins.ssrc1(), false);
        }
        Enc::Sopk => {
            let d = ins.sdst();
            match name {
                "s_movk_i32" | "s_getreg_b32" => set(st, d, false, false),
                "s_cmovk_i32" => {
                    let v = st[SCC] || dep_of(st, d, false);
                    set(st, d, v, false);
                }
                n if n.starts_with("s_cmpk_") || n == "s_addk_i32" => st[SCC] = dep_of(st, d, false),
                _ => {}
            }
        }
        Enc::Smem => {
            if name.starts_with("s_load") || name.starts_with("s_buffer_load") {
                let n = [("x16", 16), ("x8", 8), ("x4", 4), ("x2", 2)].iter().find(|(s, _)| name.ends_with(s)).map(|x| x.1).unwrap_or(1);
                let v = dep_of(st, ins.smem_sbase(), true) || (ins.smem_soffset() != crate::OP_NULL && dep_of(st, ins.smem_soffset(), false));
                for k in 0..n {
                    set(st, ins.sdst() + k, v, false);
                }
            } else if name.starts_with("s_memtime") || name.starts_with("s_memrealtime") {
                set(st, ins.sdst(), false, true);
            }
        }
        Enc::Vopc => {
            if name.starts_with("v_cmpx_") {
                st[EXEC] = true;
                st[EXEC + 1] = true;
            } else {
                set(st, ins.sdwa_sdst().unwrap_or(crate::OP_VCC_LO), true, true);
            }
        }
        Enc::Vop2 => {
            // Carry forms write VCC.
            if name.contains("_co_") {
                set(st, crate::OP_VCC_LO, true, true);
            }
        }
        Enc::Vop1 => {
            if name == "v_readfirstlane_b32" {
                set(st, ins.vdst(), false, false);
            }
        }
        Enc::Vop3 => {
            if crate::vop3_kind(ins.op) == crate::Vop3Kind::FromVopc {
                if name.starts_with("v_cmpx_") {
                    st[EXEC] = true;
                    st[EXEC + 1] = true;
                } else {
                    set(st, ins.vdst(), true, true);
                }
            } else if ins.is_vop3b() {
                set(st, ins.vop3_sdst(), true, true);
            } else if matches!(name, "v_readlane_b32" | "v_readfirstlane_b32") {
                set(st, ins.vdst(), false, false);
            }
        }
        _ => {}
    }
}

/// Whether a block-ending conditional branch depends on lane state.
fn branch_divergent(st: &Dep, ins: &Inst) -> bool {
    match ins.name().unwrap_or("") {
        "s_cbranch_execz" | "s_cbranch_execnz" => st[EXEC] || st[EXEC + 1],
        "s_cbranch_vccz" | "s_cbranch_vccnz" => st[VCC] || st[VCC + 1],
        "s_cbranch_scc0" | "s_cbranch_scc1" => st[SCC],
        _ => false,
    }
}

pub fn analyze(code: &[Inst]) -> Uniformity {
    let n = code.len();
    let mut block_of = HashMap::new();
    let mut divergent_branches = HashSet::new();
    if n == 0 {
        return Uniformity { block_of, uniform: Vec::new(), divergent_branches };
    }
    let index: HashMap<u32, usize> = code.iter().enumerate().map(|(i, ins)| (ins.pc, i)).collect();
    let mut starts = std::collections::BTreeSet::new();
    starts.insert(0usize);
    for (i, ins) in code.iter().enumerate() {
        if let Some(t) = ins.branch_target() {
            if let Some(&j) = index.get(&t) {
                starts.insert(j);
            }
            if i + 1 < n {
                starts.insert(i + 1);
            }
        }
        if ins.is_endpgm() && i + 1 < n {
            starts.insert(i + 1);
        }
    }
    let starts: Vec<usize> = starts.into_iter().collect();
    let nb = starts.len();
    let at: HashMap<usize, usize> = starts.iter().enumerate().map(|(b, &s)| (s, b)).collect();
    let end = |b: usize| starts.get(b + 1).copied().unwrap_or(n);
    for b in 0..nb {
        for ins in &code[starts[b]..end(b)] {
            block_of.insert(ins.pc, b);
        }
    }
    // Successors (nb = virtual exit).
    let exit = nb;
    let succ: Vec<Vec<usize>> = (0..nb)
        .map(|b| {
            let e = end(b);
            let last = &code[e - 1];
            if last.is_endpgm() {
                return vec![exit];
            }
            let mut v = Vec::new();
            match last.branch_target() {
                Some(t) => {
                    v.push(index.get(&t).map(|j| at[j]).unwrap_or(exit));
                    if last.name() != Some("s_branch") {
                        v.push(if e < n { at[&e] } else { exit });
                    }
                }
                None => v.push(if e < n { b + 1 } else { exit }),
            }
            v.dedup();
            v
        })
        .collect();
    // Lane dependence at block ends (forward, union at joins).
    let mut entry: Vec<Option<Dep>> = vec![None; nb];
    entry[0] = Some([false; 129]);
    let mut changed = true;
    while changed {
        changed = false;
        for b in 0..nb {
            let Some(mut st) = entry[b] else { continue };
            for ins in &code[starts[b]..end(b)] {
                step(&mut st, ins);
            }
            for &s in &succ[b] {
                if s == exit {
                    continue;
                }
                match &mut entry[s] {
                    None => {
                        entry[s] = Some(st);
                        changed = true;
                    }
                    Some(old) => {
                        for k in 0..129 {
                            if st[k] && !old[k] {
                                old[k] = true;
                                changed = true;
                            }
                        }
                    }
                }
            }
        }
    }
    // Post-dominators (bit sets over blocks + exit).
    let words = (nb + 1).div_ceil(64);
    let full = vec![u64::MAX; words];
    let mut pdom: Vec<Vec<u64>> = vec![full.clone(); nb + 1];
    pdom[exit] = vec![0; words];
    pdom[exit][exit / 64] |= 1 << (exit % 64);
    let mut changed = true;
    while changed {
        changed = false;
        for b in (0..nb).rev() {
            let mut acc = full.clone();
            for &s in &succ[b] {
                for (a, x) in acc.iter_mut().zip(&pdom[s]) {
                    *a &= x;
                }
            }
            acc[b / 64] |= 1 << (b % 64);
            if acc != pdom[b] {
                pdom[b] = acc;
                changed = true;
            }
        }
    }
    let count = |s: &Vec<u64>| s.iter().map(|w| w.count_ones()).sum::<u32>();
    // Immediate post-dominator: the strict post-dominator with the largest set.
    let ipdom: Vec<Option<usize>> = (0..=nb)
        .map(|b| {
            let mut best: Option<(u32, usize)> = None;
            for d in 0..=nb {
                if d != b && pdom[b][d / 64] >> (d % 64) & 1 != 0 {
                    let c = count(&pdom[d]);
                    if best.is_none_or(|(bc, _)| c > bc) {
                        best = Some((c, d));
                    }
                }
            }
            best.map(|x| x.1)
        })
        .collect();
    // Blocks control dependent on divergent branches (transitively).
    let mut divergent = vec![false; nb];
    let mut changed = true;
    while changed {
        changed = false;
        for x in 0..nb {
            if succ[x].len() < 2 || entry[x].is_none() {
                continue;
            }
            let last = &code[end(x) - 1];
            let mut st = entry[x].unwrap();
            for ins in &code[starts[x]..end(x) - 1] {
                step(&mut st, ins);
            }
            let lane_dependent = branch_divergent(&st, last);
            if lane_dependent {
                divergent_branches.insert(last.pc);
            }
            if !(lane_dependent || divergent[x]) {
                continue;
            }
            let stop = ipdom[x];
            for &y in &succ[x] {
                let mut r = Some(y);
                while let Some(c) = r {
                    if Some(c) == stop || c == exit {
                        break;
                    }
                    if !divergent[c] {
                        divergent[c] = true;
                        changed = true;
                    }
                    r = ipdom[c];
                }
            }
        }
    }
    let uniform = (0..nb).map(|b| entry[b].is_some() && !divergent[b]).collect();
    Uniformity { block_of, uniform, divergent_branches }
}
