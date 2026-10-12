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

/// Whether a program is the fill kernel: v_mov_b32 v0..v3 from s4..s7, then
/// store v0..v3 at index v4 through the V# in s[0:3].
fn is_fill(p: &Prepared) -> bool {
    if !matches(&p.insts, FILL) {
        return false;
    }
    let movs_ok = (0..4).all(|k| p.insts[1 + k].src(0) == 4 + k as u16 && p.insts[1 + k].vdst() == k as u16);
    let st = &p.insts[5];
    movs_ok && st.buf_srsrc() == 0 && st.vdst() == 0 && st.buf_vaddr() == 4 && st.buf_idxen()
}

/// A dispatch of the fill kernel storing one value in every component
/// (s4..s7 equal, 0 or all ones: the same 32-bit words in any format): its
/// destination range and that word. The title clears DCC and HTILE metadata
/// this way (cs 0x80271be00 in the executable), so the value a fast clear
/// reads back is known when the dispatch is recorded.
pub fn fill_value(p: &Prepared) -> Option<((u64, u64), u32)> {
    if p.user.len() < 8 || !is_fill(p) {
        return None;
    }
    let v = p.user[4];
    if !(v == 0 || v == u32::MAX) || p.user[5..8].iter().any(|&w| w != v) {
        return None;
    }
    let d = Vsharp::decode(&p.user[0..4]);
    Some(((d.base, d.num_records as u64 * d.stride.max(1) as u64), v))
}

/// A dispatch of the fill kernel that stores one 32-bit word over all of
/// [base, base + len): `fill_value`'s word in every component of a 32-bit
/// format whose components fill the stride (identity swizzle, no thread-id
/// addressing or swizzling, indices checked against the records), the
/// element index the flattened thread id (`v_lshl_add_u32 v4, s[tgid_x],
/// log2(local x), v0`; a one-dimensional dispatch, no store offsets): a
/// buffer fill of n = min(`threads_x`, records) elements
/// (`Exec::fill_in_place`). The caller checks the dispatch has no base and
/// no partial groups.
pub fn full_fill(p: &Prepared, groups: [u32; 3], local: [u32; 3], threads_x: u64) -> Option<(u64, u64, u32)> {
    let (_, v) = fill_value(p)?;
    match full_fill_range(p, groups, local, threads_x, v) {
        Ok(r) => Some(r),
        Err(why) => {
            // Once per reason: a fill that stays a kernel dispatch.
            static SEEN: parking_lot::Mutex<Vec<&'static str>> = parking_lot::Mutex::new(Vec::new());
            let mut g = SEEN.lock();
            if !g.contains(&why) {
                g.push(why);
                crate::info!("verify: fill kernel cs {:#x} stays a dispatch: {why} (V# {:08x?}, groups {groups:?}, local {local:?}, rsrc2 {:#x})", p.addr, &p.user[0..4], p.tr.spec.cs_rsrc2);
            }
            None
        }
    }
}

fn full_fill_range(p: &Prepared, groups: [u32; 3], local: [u32; 3], threads_x: u64, v: u32) -> Result<(u64, u64, u32), &'static str> {
    let d = Vsharp::decode(&p.user[0..4]);
    let comps = dword_components(d.format).ok_or("format not of 32-bit components")?;
    if d.dst_sel != 0xFAC {
        return Err("swizzled components");
    }
    if d.add_tid || d.raw[1] >> 30 != 0 {
        return Err("thread-id or swizzled addressing");
    }
    if d.oob_select > 1 {
        return Err("indices not checked against the records");
    }
    if d.stride != 4 * comps {
        return Err("components do not fill the stride");
    }
    let lx = local[0].max(1);
    if groups[1] > 1 || groups[2] > 1 || local[1] > 1 || local[2] > 1 || !lx.is_power_of_two() {
        return Err("not a one-dimensional dispatch of a power-of-two group size");
    }
    // The workgroup id X follows the user SGPRs (COMPUTE_PGM_RSRC2.TGID_X_EN).
    let rsrc2 = p.tr.spec.cs_rsrc2;
    if rsrc2 & (1 << 7) == 0 {
        return Err("no workgroup id X");
    }
    let tgid_x = ((rsrc2 >> 1) & 0x1F) as u16;
    let ix = &p.insts[0];
    if ix.vdst() != 4 || ix.src(0) != tgid_x || ix.src(1) != 128 + lx.trailing_zeros() as u16 || ix.src(2) != rdna::OP_VGPR0 {
        return Err("index not (workgroup id X << log2(group size)) + thread id X");
    }
    if ix.vop3_abs() != 0 || ix.vop3_neg() != 0 || ix.vop3_clamp() || ix.vop3_omod() != 0 || ix.vop3_opsel() != 0 {
        return Err("index instruction modifiers");
    }
    let st = &p.insts[5];
    if st.buf_offset() != 0 || st.buf_offen() || !matches!(st.buf_soffset(), 128 | rdna::OP_NULL) {
        return Err("store offsets");
    }
    let n = threads_x.min(d.num_records as u64);
    Ok((d.base, n * d.stride as u64, v))
}

/// A check for a dispatch of a recognized kernel.
pub fn check_for(p: &Prepared, groups: [u32; 3], local: [u32; 3]) -> Option<Check> {
    if !enabled() {
        return None;
    }
    let kind = if is_fill(p) {
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
