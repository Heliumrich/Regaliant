//! Self-checks of two simple title kernels (`DS_GPU_VERIFY=1`): after the GPU
//! has run them, their outputs are compared with the expected guest memory
//! contents. Kernels are recognized by their exact instruction sequence (from
//! the captured front-end frame):
//! - fill: `dst[i] = (s4, s5, s6, s7)` for i = group * 64 + thread, V# in s[0:3];
//! - copy: `dst[i] = src[i % n]` for i < count, src V# s[0:3], dst V# s[4:7],
//!   {count, n} read from the constant buffer V# s[8:11].
//!
//! A copy's expected values are the source as the command processor found
//! it (`snapshot_source`), when no GPU work or deferred CP write still wrote
//! it: read at completion, sources the title updates every frame (animation
//! data) made the check fail spuriously. Others are unverifiable, and so
//! are checks whose destination later work writes again (later in the
//! batch, or a later batch: the title copies into one staging range from
//! many sources).

use super::desc::Vsharp;
use super::shaders::Prepared;
use std::sync::atomic::{AtomicU64, Ordering};

static PASSED: AtomicU64 = AtomicU64::new(0);
static FAILED: AtomicU64 = AtomicU64::new(0);
static UNCHECKED: AtomicU64 = AtomicU64::new(0);
static LOGGED: AtomicU64 = AtomicU64::new(0);

pub fn enabled() -> bool {
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
    /// Copy: (count, modulus, sampled destination indices with the source
    /// values expected there), taken when the dispatch was recorded.
    expected: Option<(u64, u64, Vec<(u64, u32)>)>,
    /// Number of the batch's guest writes recorded up to and including this
    /// dispatch (later ones may overwrite the destination).
    pub writes_mark: usize,
    /// The destination was written again later in the batch.
    pub overwritten: bool,
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
    Some(Check { kind, addr: p.addr, threads, user: p.user.clone(), expected: None, writes_mark: 0, overwritten: false })
}

/// The guest range a check compares (the kernel's destination).
pub fn dest_range(c: &Check) -> (u64, u64) {
    let v = match c.kind {
        Kind::Fill => Vsharp::decode(&c.user[0..4]),
        Kind::Copy => Vsharp::decode(&c.user[4..8]),
    };
    (v.base, v.num_records as u64 * v.stride.max(1) as u64)
}

/// Guest ranges a check's expected values come from (copy: source and
/// constant buffer).
pub fn source_ranges(c: &Check) -> Vec<(u64, u64)> {
    match c.kind {
        Kind::Fill => Vec::new(),
        Kind::Copy => {
            let src = Vsharp::decode(&c.user[0..4]);
            let cb = Vsharp::decode(&c.user[8..12]);
            vec![(src.base, src.num_records as u64 * src.stride.max(1) as u64), (cb.base, 8)]
        }
    }
}

/// Records a copy's expected values now (see the module documentation).
pub fn snapshot_source(c: &mut Check) {
    if !matches!(c.kind, Kind::Copy) {
        return;
    }
    let src = Vsharp::decode(&c.user[0..4]);
    let dst = Vsharp::decode(&c.user[4..8]);
    let cb = Vsharp::decode(&c.user[8..12]);
    let (Some(count), Some(modn)) = (rd(cb.base), rd(cb.base + 4)) else { return };
    let (count, modn) = (count as u64, modn as u64);
    if modn == 0 || src.stride == 0 {
        return;
    }
    let n = count.min(c.threads).min(dst.num_records as u64);
    let mut want = Vec::new();
    for i in samples(n) {
        let s = i % modn;
        let v = if s < src.num_records as u64 {
            match rd(src.base + s * src.stride as u64) {
                Some(v) => v,
                None => return,
            }
        } else {
            0
        };
        want.push((i, v));
    }
    c.expected = Some((count, modn, want));
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
    if c.overwritten {
        outcome(name, c, Err("unverifiable: destination written again by later work".into()));
        return;
    }
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
    let Some((count, modn, want)) = &c.expected else {
        return Err("unverifiable: source still written by GPU work when the copy was recorded".into());
    };
    let (count, modn) = (*count, *modn);
    if dword_components(src.format) != Some(1) || src.format != dst.format || src.stride == 0 || dst.stride == 0 || src.dst_sel & 7 != 4 || dst.dst_sel & 7 != 4 {
        return Err(format!("unverifiable: formats {}/{} strides {}/{}", src.format, dst.format, src.stride, dst.stride));
    }
    let n = count.min(c.threads).min(dst.num_records as u64);
    let mut bad = 0u64;
    for &(i, want) in want {
        let got = rd(dst.base + i * dst.stride as u64).ok_or("unverifiable: destination unmapped")?;
        if got != want {
            if bad == 0 {
                let s = i % modn;
                let now = if s < src.num_records as u64 { rd(src.base + s * src.stride as u64) } else { Some(0) };
                crate::warn!(
                    "verify: copy element {i} is {got:#x}, expected {want:#x} (source element {s} when recorded, {now:#x?} now; source {:#x} stride {}, destination {:#x} stride {})",
                    src.base, src.stride, dst.base, dst.stride
                );
            }
            bad += 1;
        }
    }
    if bad > 0 {
        return Err(format!("{bad} of {} sampled elements differ (count {count}, modulus {modn})", want.len()));
    }
    Ok(format!("{} of {n} elements checked, dst[i] == src[i % {modn}] as recorded (count {count}, format {})", want.len(), src.format))
}

pub fn status() -> String {
    format!("verify {} passed, {} failed, {} unverifiable", PASSED.load(Ordering::Relaxed), FAILED.load(Ordering::Relaxed), UNCHECKED.load(Ordering::Relaxed))
}
