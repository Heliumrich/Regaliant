//! Self-checks of two simple title kernels (`DS_GPU_VERIFY=1`): after the GPU
//! has run them, their outputs are compared with the expected guest memory
//! contents. Kernels are recognized by their exact instruction sequence (from
//! the captured front-end frame):
//! - fill: `dst[i] = (s4, s5, s6, s7)` for i = group * 64 + thread, V# in s[0:3];
//! - copy: `dst[i] = src[i % n]` for i < count, src V# s[0:3], dst V# s[4:7],
//!   {count, n} read from the constant buffer V# s[8:11].

use super::desc::Vsharp;
use super::shaders::Prepared;
use std::sync::atomic::{AtomicU64, Ordering};

static PASSED: AtomicU64 = AtomicU64::new(0);
static FAILED: AtomicU64 = AtomicU64::new(0);
static UNCHECKED: AtomicU64 = AtomicU64::new(0);
static LOGGED: AtomicU64 = AtomicU64::new(0);

fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_GPU_VERIFY").ok().as_deref() == Some("1"))
}

#[derive(Clone, Copy, Debug)]
enum Kind {
    Fill,
    Copy,
}

pub struct Check {
    kind: Kind,
    addr: u64,
    threads: u64,
    user: Vec<u32>,
}

const FILL: &[&str] = &["v_lshl_add_u32", "v_mov_b32", "v_mov_b32", "v_mov_b32", "v_mov_b32", "buffer_store_format_xyzw", "s_endpgm"];
const COPY: &[&str] = &[
    "s_inst_prefetch",
    "v_lshl_add_u32",
    "s_buffer_load_dword",
    "s_waitcnt",
    "v_cmpx_gt_u32",
    "s_cbranch_execz",
    "s_buffer_load_dword",
    "s_waitcnt",
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
    "s_waitcnt",
    "buffer_store_format_x",
    "s_endpgm",
];

fn matches(insts: &[rdna::Inst], want: &[&str]) -> bool {
    insts.len() == want.len() && insts.iter().zip(want).all(|(i, w)| i.name() == Some(*w))
}

/// A check for a dispatch of a recognized kernel.
pub fn check_for(p: &Prepared, groups: [u32; 3], local: [u32; 3]) -> Option<Check> {
    if !enabled() {
        return None;
    }
    let kind = if matches(&p.insts, FILL) {
        // v_mov_b32 v0..v3 from s4..s7; store v0 at index v4 through s[0:3].
        let movs_ok = (0..4).all(|k| p.insts[1 + k].src(0) == 4 + k as u16 && p.insts[1 + k].vdst() == k as u16);
        let st = &p.insts[5];
        if !(movs_ok && st.buf_srsrc() == 0 && st.vdst() == 0 && st.buf_vaddr() == 4 && st.buf_idxen()) {
            return None;
        }
        Kind::Fill
    } else if matches(&p.insts, COPY) {
        Kind::Copy
    } else {
        return None;
    };
    let threads = groups.iter().map(|&g| g as u64).product::<u64>() * local.iter().map(|&l| l.max(1) as u64).product::<u64>();
    Some(Check { kind, addr: p.addr, threads, user: p.user.clone() })
}

fn rd(a: u64) -> Option<u32> {
    super::cp::guest_readable(a, 4).then(|| unsafe { std::ptr::read_volatile(a as *const u32) })
}

/// Element indices to compare: all of them, or 4096 spread out plus the last.
fn samples(n: u64) -> Vec<u64> {
    if n <= 4096 {
        return (0..n).collect();
    }
    let mut v: Vec<u64> = (0..4096).map(|k| k * n / 4096).collect();
    v.push(n - 1);
    v
}

/// Dword components of a format whose components are all 32-bit.
fn dword_components(format: u8) -> Option<u32> {
    let f = rdna::format::gfx10(format as u16)?;
    let n = f.components() as u32;
    (n > 0 && f.bits.iter().take(n as usize).all(|&b| b == 32)).then_some(n)
}

fn outcome(name: &str, c: &Check, r: Result<String, String>) {
    let first = LOGGED.fetch_add(1, Ordering::Relaxed) < 12;
    match r {
        Ok(msg) => {
            PASSED.fetch_add(1, Ordering::Relaxed);
            if first {
                crate::info!("gpu[{name}] verify: {:?} kernel cs {:#x}: {msg}", c.kind, c.addr);
            }
        }
        Err(msg) if msg.starts_with("unverifiable") => {
            UNCHECKED.fetch_add(1, Ordering::Relaxed);
            if first {
                crate::info!("gpu[{name}] verify: {:?} kernel cs {:#x}: {msg}", c.kind, c.addr);
            }
        }
        Err(msg) => {
            FAILED.fetch_add(1, Ordering::Relaxed);
            crate::warn!("gpu[{name}] verify FAILED: {:?} kernel cs {:#x}: {msg}", c.kind, c.addr);
        }
    }
}

/// Runs a check after the dispatch completed.
pub fn run(name: &str, c: &Check) {
    let r = match c.kind {
        Kind::Fill => fill(c),
        Kind::Copy => copy(c),
    };
    outcome(name, c, r);
}

fn fill(c: &Check) -> Result<String, String> {
    let v = Vsharp::decode(&c.user[0..4]);
    let comps = dword_components(v.format).ok_or_else(|| format!("unverifiable: format {}", v.format))?;
    if v.stride == 0 || v.dst_sel != 0xFAC {
        return Err(format!("unverifiable: stride {} dst_sel {:#x}", v.stride, v.dst_sel));
    }
    let n = c.threads.min(v.num_records as u64);
    let want = &c.user[4..8];
    let mut bad = 0u64;
    let idx = samples(n);
    for &i in &idx {
        for k in 0..comps as u64 {
            let got = rd(v.base + i * v.stride as u64 + 4 * k).ok_or("unverifiable: unmapped")?;
            if got != want[k as usize] {
                if bad == 0 {
                    crate::warn!("verify: fill element {i} component {k} at {:#x} is {got:#x}, expected {:#x}", v.base + i * v.stride as u64 + 4 * k, want[k as usize]);
                }
                bad += 1;
            }
        }
    }
    if bad > 0 {
        return Err(format!("{bad} of {} sampled components differ ({n} elements, format {})", idx.len() as u64 * comps as u64, v.format));
    }
    Ok(format!("{} of {n} elements checked, all equal the fill value {:x?} (format {}, base {:#x})", idx.len(), &want[..comps as usize], v.format, v.base))
}

fn copy(c: &Check) -> Result<String, String> {
    let src = Vsharp::decode(&c.user[0..4]);
    let dst = Vsharp::decode(&c.user[4..8]);
    let cb = Vsharp::decode(&c.user[8..12]);
    let count = rd(cb.base).ok_or("unverifiable: constant buffer unmapped")? as u64;
    let modn = rd(cb.base + 4).ok_or("unverifiable: constant buffer unmapped")? as u64;
    if modn == 0 {
        return Err("unverifiable: modulus 0".into());
    }
    if dword_components(src.format) != Some(1) || src.format != dst.format || src.stride == 0 || dst.stride == 0 || src.dst_sel & 7 != 4 || dst.dst_sel & 7 != 4 {
        return Err(format!("unverifiable: formats {}/{} strides {}/{}", src.format, dst.format, src.stride, dst.stride));
    }
    let n = count.min(c.threads).min(dst.num_records as u64);
    let mut bad = 0u64;
    let idx = samples(n);
    for &i in &idx {
        let s = i % modn;
        let want = if s < src.num_records as u64 { rd(src.base + s * src.stride as u64).ok_or("unverifiable: source unmapped")? } else { 0 };
        let got = rd(dst.base + i * dst.stride as u64).ok_or("unverifiable: destination unmapped")?;
        if got != want {
            if bad == 0 {
                crate::warn!("verify: copy element {i} is {got:#x}, expected {want:#x} (source element {s})");
            }
            bad += 1;
        }
    }
    if bad > 0 {
        return Err(format!("{bad} of {} sampled elements differ (count {count}, modulus {modn})", idx.len()));
    }
    Ok(format!("{} of {n} elements checked, dst[i] == src[i % {modn}] (count {count}, format {})", idx.len(), src.format))
}

pub fn status() -> String {
    format!("verify {} passed, {} failed, {} unverifiable", PASSED.load(Ordering::Relaxed), FAILED.load(Ordering::Relaxed), UNCHECKED.load(Ordering::Relaxed))
}
