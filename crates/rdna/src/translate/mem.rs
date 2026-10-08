//! Memory, image, interpolation, export and stage-interface semantics.

use super::{e, ok, ImageDim, NumClass, ResSpec, Stage, R, T};
use crate::format::{self, Num};
use crate::{bits, Enc, Inst};
use rspirv::dr::Operand;
use rspirv::spirv::{self, BuiltIn, Decoration, GlslStd450Op as GLOp, ImageOperands, StorageClass, Word};

// ---------------------------------------------------------------------------
// Stage inputs

/// PS input VGPR counts per SPI_PS_INPUT_ADDR bit.
const PS_INPUT_VGPRS: [u32; 16] = [2, 2, 2, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1];

pub fn stage_inputs(tr: &mut T) -> R<()> {
    match tr.spec.stage {
        Stage::Vertex => {
            // NGG (merged ES/GS) wave: s3 = merged wave info (ES threads in
            // [7:0], GS threads in [15:8]); v0 = primitive export data,
            // v5 = vertex id, v8 = instance id.
            let k = tr.u(0x4040);
            tr.set_sreg(3, k)?;
            let vid = tr.input(tr.t.i32, Some(BuiltIn::VertexIndex), None);
            let iid = tr.input(tr.t.i32, Some(BuiltIn::InstanceIndex), None);
            let v = e(tr.b.load(tr.t.i32, None, vid, None, []))?;
            let v = tr.to_u(v)?;
            tr.set_vreg_raw(5, v)?;
            let i = e(tr.b.load(tr.t.i32, None, iid, None, []))?;
            let i = tr.to_u(i)?;
            tr.set_vreg_raw(8, i)?;
        }
        Stage::Fragment => {
            let fc = tr.input(tr.t.v4f, Some(BuiltIn::FragCoord), None);
            tr.frag_coord = Some(fc);
            let ff = tr.input(tr.t.bool_, Some(BuiltIn::FrontFacing), None);
            let fcv = e(tr.b.load(tr.t.v4f, None, fc, None, []))?;
            let mut vg = 0u32;
            let addr = if tr.spec.ps_input_addr == 0 { 0x2 } else { tr.spec.ps_input_addr };
            for bit in 0..16 {
                if addr & (1 << bit) == 0 {
                    continue;
                }
                let n = PS_INPUT_VGPRS[bit];
                for k in 0..n {
                    let val = match bit {
                        8..=11 => {
                            let c = e(tr.b.composite_extract(tr.t.f32, None, fcv, [(bit - 8) as u32]))?;
                            tr.to_u(c)?
                        }
                        12 => {
                            let f = e(tr.b.load(tr.t.bool_, None, ff, None, []))?;
                            let one = tr.f(1.0);
                            let m1 = tr.f(-1.0);
                            let s = e(tr.b.select(tr.t.f32, None, f, one, m1))?;
                            tr.to_u(s)?
                        }
                        14 => tr.u(1),
                        _ => tr.u(0),
                    };
                    tr.set_vreg_raw(vg + k, val)?;
                }
                vg += n;
            }
        }
        Stage::Compute => {
            let lid = tr.input(tr.t.v3u, Some(BuiltIn::LocalInvocationId), None);
            let wid = tr.input(tr.t.v3u, Some(BuiltIn::WorkgroupId), None);
            let lidx = tr.input(tr.t.u32, Some(BuiltIn::LocalInvocationIndex), None);
            let l = e(tr.b.load(tr.t.v3u, None, lid, None, []))?;
            let tid_cnt = (tr.spec.cs_rsrc2 >> 11) & 3;
            for k in 0..=tid_cnt.min(2) {
                let c = e(tr.b.composite_extract(tr.t.u32, None, l, [k]))?;
                tr.set_vreg_raw(k, c)?;
            }
            let w = e(tr.b.load(tr.t.v3u, None, wid, None, []))?;
            let mut s = tr.spec.user_base as u32 + tr.spec.user_sgprs as u32;
            for k in 0..3 {
                if tr.spec.cs_rsrc2 & (1 << (7 + k)) != 0 {
                    let c = e(tr.b.composite_extract(tr.t.u32, None, w, [k]))?;
                    tr.set_sreg(s, c)?;
                    s += 1;
                }
            }
            if tr.spec.cs_rsrc2 & (1 << 10) != 0 {
                // TG_SIZE: wave id in group [5:0], wave count [11:6]
                // (single-lane waves: report one wave).
                let v = tr.u(1 << 6);
                tr.set_sreg(s, v)?;
            }
            let idx = e(tr.b.load(tr.t.u32, None, lidx, None, []))?;
            tr.local_index = Some(idx);
            let km = tr.u(if tr.spec.wave64 { 63 } else { 31 });
            tr.lane_id = Some(e(tr.b.bitwise_and(tr.t.u32, None, idx, km))?);
            let lds_dwords = ((tr.spec.cs_rsrc2 >> 15) & 0x1FF) * 128;
            if lds_dwords > 0 {
                let n = lds_dwords.min(12288);
                if n < lds_dwords {
                    tr.warn(format!("LDS of {lds_dwords} dwords clamped to {n}"));
                }
                let len = tr.u(n);
                let arr = tr.b.type_array(tr.t.u32, len);
                let pt = tr.b.type_pointer(None, StorageClass::Workgroup, arr);
                let v = tr.global_var(pt, StorageClass::Workgroup);
                tr.lds = Some(v);
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Raw buffer access

fn dword_ptr(tr: &mut T, var: Word, idx: Word) -> R<Word> {
    let z = tr.u(0);
    e(tr.b.access_chain(tr.t.sb_u32, None, var, [z, idx]))
}

fn load_dword(tr: &mut T, var: Word, idx: Word) -> R<Word> {
    let p = dword_ptr(tr, var, idx)?;
    e(tr.b.load(tr.t.u32, None, p, None, []))
}

fn store_dword(tr: &mut T, var: Word, idx: Word, v: Word) -> R<()> {
    let p = dword_ptr(tr, var, idx)?;
    ok(tr.b.store(p, v, None, []))
}

/// Loads `n` dwords at byte address `addr`: one 8/16-byte access when the
/// address is aligned to it (n = 2 or 4), else dword by dword (`bytes`:
/// unaligned-safe byte loads instead of dword-truncated addresses).
fn load_dwords(tr: &mut T, var: Word, addr: Word, n: u32, bytes: bool) -> R<Vec<Word>> {
    let u = tr.t.u32;
    let scalar = |tr: &mut T| -> R<Vec<Word>> {
        let mut v = Vec::new();
        if bytes {
            for k in 0..n {
                let o = tr.u(4 * k);
                let a = e(tr.b.i_add(u, None, addr, o))?;
                v.push(load_bytes(tr, var, a, 4)?);
            }
        } else {
            let two = tr.u(2);
            let idx = e(tr.b.shift_right_logical(u, None, addr, two))?;
            for k in 0..n {
                let kk = tr.u(k);
                let i = e(tr.b.i_add(u, None, idx, kk))?;
                v.push(load_dword(tr, var, i)?);
            }
        }
        Ok(v)
    };
    let view = if n == 2 || n == 4 { tr.vec_view(var, n) } else { None };
    let Some(view) = view else { return scalar(tr) };
    let mask = tr.u(4 * n - 1);
    let low = e(tr.b.bitwise_and(u, None, addr, mask))?;
    let z = tr.u(0);
    let aligned = e(tr.b.i_equal(tr.t.bool_, None, low, z))?;
    let (merge, then_b, else_b) = (tr.b.id(), tr.b.id(), tr.b.id());
    ok(tr.b.selection_merge(merge, spirv::SelectionControl::NONE))?;
    ok(tr.b.branch_conditional(aligned, then_b, else_b, []))?;
    e(tr.b.begin_block(Some(then_b)))?;
    let vt = if n == 2 { tr.t.v2u } else { tr.t.v4u };
    let sh = tr.u(if n == 2 { 3 } else { 4 });
    let idx = e(tr.b.shift_right_logical(u, None, addr, sh))?;
    let pt = tr.b.type_pointer(None, StorageClass::StorageBuffer, vt);
    let p = e(tr.b.access_chain(pt, None, view, [z, idx]))?;
    let vv = e(tr.b.load(vt, None, p, None, []))?;
    let mut va = Vec::new();
    for k in 0..n {
        va.push(e(tr.b.composite_extract(u, None, vv, [k]))?);
    }
    ok(tr.b.branch(merge))?;
    e(tr.b.begin_block(Some(else_b)))?;
    let vs = scalar(tr)?;
    // The scalar path creates no blocks: it ends in else_b.
    let else_end = else_b;
    ok(tr.b.branch(merge))?;
    e(tr.b.begin_block(Some(merge)))?;
    let mut out = Vec::new();
    for k in 0..n as usize {
        out.push(e(tr.b.phi(u, None, [(va[k], then_b), (vs[k], else_end)]))?);
    }
    Ok(out)
}

/// Stores dwords at byte address `addr` (one 8/16-byte store when aligned).
fn store_dwords(tr: &mut T, var: Word, addr: Word, vals: &[Word]) -> R<()> {
    let u = tr.t.u32;
    let n = vals.len() as u32;
    let scalar = |tr: &mut T| -> R<()> {
        let two = tr.u(2);
        let idx = e(tr.b.shift_right_logical(u, None, addr, two))?;
        for (k, v) in vals.iter().enumerate() {
            let kk = tr.u(k as u32);
            let i = e(tr.b.i_add(u, None, idx, kk))?;
            store_dword(tr, var, i, *v)?;
        }
        Ok(())
    };
    let view = if n == 2 || n == 4 { tr.vec_view(var, n) } else { None };
    let Some(view) = view else { return scalar(tr) };
    let mask = tr.u(4 * n - 1);
    let low = e(tr.b.bitwise_and(u, None, addr, mask))?;
    let z = tr.u(0);
    let aligned = e(tr.b.i_equal(tr.t.bool_, None, low, z))?;
    let (merge, then_b, else_b) = (tr.b.id(), tr.b.id(), tr.b.id());
    ok(tr.b.selection_merge(merge, spirv::SelectionControl::NONE))?;
    ok(tr.b.branch_conditional(aligned, then_b, else_b, []))?;
    e(tr.b.begin_block(Some(then_b)))?;
    let vt = if n == 2 { tr.t.v2u } else { tr.t.v4u };
    let sh = tr.u(if n == 2 { 3 } else { 4 });
    let idx = e(tr.b.shift_right_logical(u, None, addr, sh))?;
    let pt = tr.b.type_pointer(None, StorageClass::StorageBuffer, vt);
    let p = e(tr.b.access_chain(pt, None, view, [z, idx]))?;
    let vv = e(tr.b.composite_construct(vt, None, vals.to_vec()))?;
    ok(tr.b.store(p, vv, None, []))?;
    ok(tr.b.branch(merge))?;
    e(tr.b.begin_block(Some(else_b)))?;
    scalar(tr)?;
    ok(tr.b.branch(merge))?;
    e(tr.b.begin_block(Some(merge)))?;
    Ok(())
}

/// Loads `nbytes` (1..=4) little-endian bytes at an arbitrary byte address.
fn load_bytes(tr: &mut T, var: Word, addr: Word, nbytes: u32) -> R<Word> {
    let u = tr.t.u32;
    let two = tr.u(2);
    let idx = e(tr.b.shift_right_logical(u, None, addr, two))?;
    let lo = load_dword(tr, var, idx)?;
    let one = tr.u(1);
    let idx1 = e(tr.b.i_add(u, None, idx, one))?;
    let hi = load_dword(tr, var, idx1)?;
    let wide = {
        let v = e(tr.b.composite_construct(tr.t.v2u, None, [lo, hi]))?;
        e(tr.b.bitcast(tr.t.u64, None, v))?
    };
    let k3 = tr.u(3);
    let sub = e(tr.b.bitwise_and(u, None, addr, k3))?;
    let sh = e(tr.b.shift_left_logical(u, None, sub, k3))?;
    let shifted = e(tr.b.shift_right_logical(tr.t.u64, None, wide, sh))?;
    let r = e(tr.b.u_convert(u, None, shifted))?;
    if nbytes >= 4 {
        return Ok(r);
    }
    let m = tr.u((1u32 << (nbytes * 8)) - 1);
    e(tr.b.bitwise_and(u, None, r, m))
}

/// Stores the low `nbytes` (1 or 2) bytes of `v` at a byte address
/// (atomic read-modify-write of the containing dword).
fn store_bytes(tr: &mut T, var: Word, addr: Word, v: Word, nbytes: u32) -> R<()> {
    let u = tr.t.u32;
    let two = tr.u(2);
    let idx = e(tr.b.shift_right_logical(u, None, addr, two))?;
    let k3 = tr.u(3);
    let sub = e(tr.b.bitwise_and(u, None, addr, k3))?;
    let sh = e(tr.b.shift_left_logical(u, None, sub, k3))?;
    let m = tr.u((1u32 << (nbytes * 8)) - 1);
    let mask = e(tr.b.shift_left_logical(u, None, m, sh))?;
    let vm = e(tr.b.bitwise_and(u, None, v, m))?;
    let vs = e(tr.b.shift_left_logical(u, None, vm, sh))?;
    let p = dword_ptr(tr, var, idx)?;
    let scope = tr.u(spirv::Scope::Device as u32);
    let sem = tr.u(0);
    let nm = e(tr.b.not(u, None, mask))?;
    e(tr.b.atomic_and(u, None, p, scope, sem, nm))?;
    e(tr.b.atomic_or(u, None, p, scope, sem, vs))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Scalar memory

fn smem_dwords(name: &str) -> u32 {
    for (suf, n) in [("x16", 16), ("x8", 8), ("x4", 4), ("x2", 2)] {
        if name.ends_with(suf) {
            return n;
        }
    }
    1
}

pub fn smem(tr: &mut T, ins: &Inst) -> R<()> {
    let name = ins.name().unwrap_or("");
    let u = tr.t.u32;
    let d = ins.sdst() as u32;
    if name.starts_with("s_load_dword") {
        let n = smem_dwords(name);
        let Some(&k) = tr.plan.inst_load.get(&ins.pc) else {
            tr.warn(format!("{:#x}: s_load without resolved address (zero)", ins.pc));
            for j in 0..n {
                let z = tr.u(0);
                tr.set_sreg(d + j, z)?;
            }
            return Ok(());
        };
        let base = tr.load_offsets[k];
        for j in 0..n {
            let v = tr.sgpr_data(base + j)?;
            tr.set_sreg(d + j, v)?;
        }
        return Ok(());
    }
    if name.starts_with("s_buffer_load_dword") {
        let n = smem_dwords(name);
        let Some(&(r, _)) = tr.plan.inst_res.get(&ins.pc) else { return Err(format!("{:#x}: s_buffer_load without resource", ins.pc)) };
        let var = tr.buffers[&r];
        let mut off = tr.u(ins.smem_offset() as u32);
        if ins.smem_soffset() != crate::OP_NULL {
            let so = tr.operand(ins, ins.smem_soffset())?;
            off = e(tr.b.i_add(u, None, off, so))?;
        }
        let km = tr.u(!3);
        let off = e(tr.b.bitwise_and(u, None, off, km))?;
        let bias = tr.res_info(r, 0)?;
        let size = tr.res_info(r, 2)?;
        let addr = e(tr.b.i_add(u, None, off, bias))?;
        let two = tr.u(2);
        let idx0 = e(tr.b.shift_right_logical(u, None, addr, two))?;
        for j in 0..n {
            let kj = tr.u(j);
            let idx = e(tr.b.i_add(u, None, idx0, kj))?;
            let v = load_dword(tr, var, idx)?;
            // Out-of-range dwords read 0.
            let kb = tr.u(4 * j);
            let o = e(tr.b.i_add(u, None, off, kb))?;
            let inr = e(tr.b.u_less_than(tr.t.bool_, None, o, size))?;
            let z = tr.u(0);
            let v = e(tr.b.select(u, None, inr, v, z))?;
            tr.set_sreg(d + j, v)?;
        }
        return Ok(());
    }
    match name {
        "s_dcache_inv" | "s_gl1_inv" | "s_dcache_wb" | "s_dcache_discard" | "s_dcache_discard_x2" | "s_atc_probe" | "s_atc_probe_buffer" => {}
        "s_memtime" | "s_memrealtime" | "s_get_waveid_in_workgroup" => {
            let z = tr.u(0);
            tr.set_sreg(d, z)?;
            if name != "s_get_waveid_in_workgroup" {
                tr.set_sreg(d + 1, z)?;
            }
        }
        _ => tr.warn(format!("{:#x}: unsupported {name}", ins.pc)),
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Attribute interpolation

fn attr_input(tr: &mut T, attr: u32) -> R<Option<Word>> {
    let spec_in = tr.spec.ps_inputs.get(attr as usize).copied();
    if let Some(pi) = spec_in {
        if pi.default.is_some() {
            return Ok(None);
        }
    }
    if let Some(&v) = tr.in_attrs.get(&attr) {
        return Ok(Some(v));
    }
    let (loc, flat) = match spec_in {
        Some(pi) => (pi.param as u32, pi.flat),
        None => (attr, false),
    };
    let v = tr.input(tr.t.v4f, None, Some(loc));
    if flat {
        tr.b.decorate(v, Decoration::Flat, []);
    }
    tr.in_attrs.insert(attr, v);
    tr.params = tr.params.max(attr + 1);
    Ok(Some(v))
}

/// v_interp_p1/p2/mov: the interpolated (or flat) attribute value is
/// written by each step; the barycentric arithmetic is done by the host.
pub fn interp(tr: &mut T, ins: &Inst) -> R<()> {
    let attr = ins.intrp_attr();
    let chan = ins.intrp_chan();
    let val = match attr_input(tr, attr)? {
        Some(var) => {
            let v = e(tr.b.load(tr.t.v4f, None, var, None, []))?;
            e(tr.b.composite_extract(tr.t.f32, None, v, [chan]))?
        }
        None => {
            let d = tr.spec.ps_inputs[attr as usize].default.unwrap_or(0);
            let c = [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0]][d as usize & 3][chan as usize];
            tr.f(c)
        }
    };
    let u = tr.to_u(val)?;
    tr.set_vreg(ins.vdst() as u32, u)
}

// ---------------------------------------------------------------------------
// Buffer formats

/// Converts a raw component (`bits` wide) to a 32-bit register value.
fn convert_component(tr: &mut T, raw: Word, bits_: u32, num: Num) -> R<Word> {
    let u = tr.t.u32;
    let f = tr.t.f32;
    let z = tr.u(0);
    let w = tr.u(bits_);
    let sext = |tr: &mut T| -> R<Word> { if bits_ == 32 { Ok(raw) } else { e(tr.b.bit_field_s_extract(u, None, raw, z, w)) } };
    Ok(match num {
        Num::Unorm | Num::Srgb => {
            let x = e(tr.b.convert_u_to_f(f, None, raw))?;
            let k = tr.f(((1u64 << bits_) - 1) as f32);
            let r = e(tr.b.f_div(f, None, x, k))?;
            let r = if num == Num::Srgb && bits_ == 8 { srgb_to_linear(tr, r)? } else { r };
            tr.to_u(r)?
        }
        Num::Snorm => {
            let s = sext(tr)?;
            let si = tr.to_i(s)?;
            let x = e(tr.b.convert_s_to_f(f, None, si))?;
            let k = tr.f(((1u64 << (bits_ - 1)) - 1) as f32);
            let r = e(tr.b.f_div(f, None, x, k))?;
            let m1 = tr.f(-1.0);
            let r = tr.glsl2(f, GLOp::FMax, r, m1)?;
            tr.to_u(r)?
        }
        Num::Uscaled => {
            let x = e(tr.b.convert_u_to_f(f, None, raw))?;
            tr.to_u(x)?
        }
        Num::Sscaled => {
            let s = sext(tr)?;
            let si = tr.to_i(s)?;
            let x = e(tr.b.convert_s_to_f(f, None, si))?;
            tr.to_u(x)?
        }
        Num::Uint => raw,
        Num::Sint => sext(tr)?,
        Num::Float => match bits_ {
            32 => raw,
            16 => {
                let v = tr.glsl1(tr.t.v2f, GLOp::UnpackHalf2x16, raw)?;
                let x = e(tr.b.composite_extract(f, None, v, [0]))?;
                tr.to_u(x)?
            }
            11 | 10 => {
                // Unsigned small float (5-bit exponent, 6/5-bit mantissa):
                // rebias into an f16 (same exponent bias) and convert.
                let mbits = bits_ - 5;
                let mmask = tr.u((1 << mbits) - 1);
                let man = e(tr.b.bitwise_and(u, None, raw, mmask))?;
                let ms = tr.u(mbits);
                let ex = e(tr.b.shift_right_logical(u, None, raw, ms))?;
                let sh = tr.u(10 - mbits);
                let man16 = e(tr.b.shift_left_logical(u, None, man, sh))?;
                let k10 = tr.u(10);
                let ex16 = e(tr.b.shift_left_logical(u, None, ex, k10))?;
                let h = e(tr.b.bitwise_or(u, None, ex16, man16))?;
                let v = tr.glsl1(tr.t.v2f, GLOp::UnpackHalf2x16, h)?;
                let x = e(tr.b.composite_extract(f, None, v, [0]))?;
                tr.to_u(x)?
            }
            _ => {
                tr.warn(format!("float buffer component of {bits_} bits unsupported"));
                raw
            }
        },
    })
}

fn srgb_to_linear(tr: &mut T, c: Word) -> R<Word> {
    let f = tr.t.f32;
    let bt = tr.t.bool_;
    let k = tr.f(0.04045);
    let lin = {
        let d = tr.f(12.92);
        e(tr.b.f_div(f, None, c, d))?
    };
    let pw = {
        let a = tr.f(0.055);
        let s = e(tr.b.f_add(f, None, c, a))?;
        let d = tr.f(1.055);
        let q = e(tr.b.f_div(f, None, s, d))?;
        let g = tr.f(2.4);
        tr.glsl2(f, GLOp::Pow, q, g)?
    };
    let small = e(tr.b.f_ord_less_than_equal(bt, None, c, k))?;
    e(tr.b.select(f, None, small, lin, pw))
}

/// Loads and converts a formatted element: returns 4 register values.
fn load_formatted(tr: &mut T, var: Word, addr: Word, fmt: u16) -> R<[Word; 4]> {
    let Some(fm) = format::gfx10(fmt).filter(|f| !f.block && f.components() > 0) else {
        if fmt != 0 {
            tr.warn(format!("buffer format {fmt} unsupported"));
        }
        // FORMAT_INVALID reads return zero.
        let z = tr.u(0);
        return Ok([z, z, z, z]);
    };
    let u = tr.t.u32;
    let size = fm.bytes();
    let ndw = size.div_ceil(4);
    let dws = if size == 8 || size == 16 {
        load_dwords(tr, var, addr, ndw, true)?
    } else {
        let mut dws = Vec::new();
        for k in 0..ndw {
            let off = tr.u(k * 4);
            let a = e(tr.b.i_add(u, None, addr, off))?;
            dws.push(load_bytes(tr, var, a, size.min(4))?);
        }
        dws
    };
    let one_f = tr.u(1.0f32.to_bits());
    let one_i = tr.u(1);
    let z = tr.u(0);
    let mut out = [z, z, z, if fm.num.is_float() { one_f } else { one_i }];
    let mut bo = 0u32;
    for (c, &bw) in fm.bits.iter().enumerate() {
        let bw = bw as u32;
        if bw == 0 {
            break;
        }
        let dw = dws[(bo / 32) as usize];
        let raw = if bw == 32 {
            dw
        } else {
            let o = tr.u(bo % 32);
            let w = tr.u(bw);
            e(tr.b.bit_field_u_extract(u, None, dw, o, w))?
        };
        out[c] = convert_component(tr, raw, bw, fm.num)?;
        bo += bw;
    }
    Ok(out)
}

/// Converts a register value to a raw component (`bits` wide).
fn pack_component(tr: &mut T, v: Word, bits_: u32, num: Num) -> R<Word> {
    let u = tr.t.u32;
    let f = tr.t.f32;
    let mask = tr.u(if bits_ >= 32 { u32::MAX } else { (1u32 << bits_) - 1 });
    let r = match num {
        Num::Unorm | Num::Srgb | Num::Snorm => {
            let x = tr.to_f(v)?;
            let (lo, k) = if num == Num::Snorm { (-1.0, ((1u64 << (bits_ - 1)) - 1) as f32) } else { (0.0, ((1u64 << bits_) - 1) as f32) };
            let lo = tr.f(lo);
            let one = tr.f(1.0);
            let c = tr.glsl3(f, GLOp::NClamp, x, lo, one)?;
            let kk = tr.f(k);
            let s = e(tr.b.f_mul(f, None, c, kk))?;
            let rn = tr.glsl1(f, GLOp::RoundEven, s)?;
            let i = e(tr.b.convert_f_to_s(tr.t.i32, None, rn))?;
            tr.to_u(i)?
        }
        Num::Uscaled => {
            let x = tr.to_f(v)?;
            e(tr.b.convert_f_to_u(u, None, x))?
        }
        Num::Sscaled => {
            let x = tr.to_f(v)?;
            let i = e(tr.b.convert_f_to_s(tr.t.i32, None, x))?;
            tr.to_u(i)?
        }
        Num::Uint | Num::Sint => v,
        Num::Float => match bits_ {
            32 => v,
            16 => {
                let x = tr.to_f(v)?;
                let z = tr.f(0.0);
                let vv = e(tr.b.composite_construct(tr.t.v2f, None, [x, z]))?;
                tr.glsl1(u, GLOp::PackHalf2x16, vv)?
            }
            _ => {
                tr.warn(format!("float buffer store component of {bits_} bits unsupported"));
                v
            }
        },
    };
    e(tr.b.bitwise_and(u, None, r, mask))
}

/// Packs `n` register values per format and stores the element.
fn store_formatted(tr: &mut T, var: Word, addr: Word, vals: &[Word], fmt: u16) -> R<()> {
    let Some(fm) = format::gfx10(fmt).filter(|f| !f.block && f.components() > 0) else {
        tr.warn(format!("formatted store of format {fmt} unsupported"));
        return Ok(());
    };
    let u = tr.t.u32;
    let size = fm.bytes();
    let ndw = size.div_ceil(4) as usize;
    let mut dws: Vec<Word> = (0..ndw).map(|_| tr.u(0)).collect();
    let mut bo = 0u32;
    for (c, &bw) in fm.bits.iter().enumerate() {
        let bw = bw as u32;
        if bw == 0 || c >= vals.len() {
            break;
        }
        let raw = pack_component(tr, vals[c], bw, fm.num)?;
        let sh = tr.u(bo % 32);
        let placed = e(tr.b.shift_left_logical(u, None, raw, sh))?;
        let k = (bo / 32) as usize;
        dws[k] = e(tr.b.bitwise_or(u, None, dws[k], placed))?;
        bo += bw;
    }
    if size >= 4 {
        store_dwords(tr, var, addr, &dws)
    } else {
        store_bytes(tr, var, addr, dws[0], size)
    }
}

/// Applies V# DST_SEL to formatted-load results.
fn apply_dst_sel(tr: &mut T, v: [Word; 4], dst_sel: u16, float: bool) -> [Word; 4] {
    let one = if float { tr.u(1.0f32.to_bits()) } else { tr.u(1) };
    let z = tr.u(0);
    let mut out = v;
    for (c, o) in out.iter_mut().enumerate() {
        let sel = (dst_sel >> (c * 3)) & 7;
        *o = match sel {
            0 => z,
            1 => one,
            4..=7 => v[(sel - 4) as usize],
            _ => v[c],
        };
    }
    out
}

fn atomic_op(tr: &mut T, op: &str, p: Word, scope: spirv::Scope, v: Word, cmp: Option<Word>) -> R<Option<Word>> {
    let u = tr.t.u32;
    let scope = tr.u(scope as u32);
    let sem = tr.u(0);
    Ok(Some(match op {
        "swap" => e(tr.b.atomic_exchange(u, None, p, scope, sem, v))?,
        "cmpswap" => {
            let c = cmp.unwrap_or(v);
            e(tr.b.atomic_compare_exchange(u, None, p, scope, sem, sem, v, c))?
        }
        "add" => e(tr.b.atomic_i_add(u, None, p, scope, sem, v))?,
        "sub" => e(tr.b.atomic_i_sub(u, None, p, scope, sem, v))?,
        "smin" | "min_i32" => e(tr.b.atomic_s_min(u, None, p, scope, sem, v))?,
        "umin" | "min_u32" => e(tr.b.atomic_u_min(u, None, p, scope, sem, v))?,
        "smax" | "max_i32" => e(tr.b.atomic_s_max(u, None, p, scope, sem, v))?,
        "umax" | "max_u32" => e(tr.b.atomic_u_max(u, None, p, scope, sem, v))?,
        "and" => e(tr.b.atomic_and(u, None, p, scope, sem, v))?,
        "or" => e(tr.b.atomic_or(u, None, p, scope, sem, v))?,
        "xor" => e(tr.b.atomic_xor(u, None, p, scope, sem, v))?,
        _ => return Ok(None),
    }))
}

/// Byte address (relative to the binding) of a MUBUF/MTBUF access.
fn buffer_addr(tr: &mut T, ins: &Inst, r: usize) -> R<Word> {
    let u = tr.t.u32;
    let mut addr = tr.u(ins.buf_offset());
    let mut vreg = ins.buf_vaddr() as u32;
    if ins.buf_idxen() {
        let idx = tr.vreg(vreg)?;
        vreg += 1;
        let st = tr.res_info(r, 1)?;
        let m = e(tr.b.i_mul(u, None, idx, st))?;
        addr = e(tr.b.i_add(u, None, addr, m))?;
    }
    if ins.buf_offen() {
        let off = tr.vreg(vreg)?;
        addr = e(tr.b.i_add(u, None, addr, off))?;
    }
    let so = tr.operand(ins, ins.buf_soffset())?;
    addr = e(tr.b.i_add(u, None, addr, so))?;
    let bias = tr.res_info(r, 0)?;
    e(tr.b.i_add(u, None, addr, bias))
}

pub fn buffer(tr: &mut T, ins: &Inst) -> R<()> {
    let name = ins.name().unwrap_or("");
    let Some(&(r, _)) = tr.plan.inst_res.get(&ins.pc) else { return Err(format!("{:#x}: buffer op without resource", ins.pc)) };
    if bits(ins.w0, 16, 1) != 0 || bits(ins.w1, 23, 1) != 0 {
        tr.warn(format!("{:#x}: buffer lds/tfe unsupported", ins.pc));
    }
    if name.starts_with("buffer_gl") {
        return Ok(()); // cache invalidations
    }
    let var = tr.buffers[&r];
    let addr = buffer_addr(tr, ins, r)?;
    let (vfmt, dst_sel) = match tr.spec.resources.get(r) {
        Some(ResSpec::Buffer { format, dst_sel }) => (*format as u16, *dst_sel),
        _ => (0, 0xFAC), // identity swizzle
    };
    let fmt = if ins.enc == Enc::Mtbuf { ins.mtbuf_format() as u16 } else { vfmt };
    mem_access(tr, ins, name.trim_start_matches('t'), var, addr, fmt, dst_sel)
}

/// GLOBAL instructions with an SGPR base: the base is a bound resource.
pub fn global(tr: &mut T, ins: &Inst) -> R<()> {
    let name = ins.name().unwrap_or("");
    let Some(&(r, _)) = tr.plan.inst_res.get(&ins.pc) else {
        tr.warn(format!("{:#x}: {name} with a 64-bit VGPR address unsupported", ins.pc));
        return Ok(());
    };
    let u = tr.t.u32;
    let var = tr.buffers[&r];
    let off = tr.vreg(ins.flat_addr() as u32)?;
    let k = tr.u(ins.flat_offset() as u32);
    let a = e(tr.b.i_add(u, None, off, k))?;
    let bias = tr.res_info(r, 0)?;
    let addr = e(tr.b.i_add(u, None, a, bias))?;
    let name = name.replacen("global_", "buffer_", 1);
    mem_access(tr, ins, &name, var, addr, 0, 0xFAC)
}

/// Shared MUBUF/GLOBAL semantics on a raw storage buffer at byte `addr`.
fn mem_access(tr: &mut T, ins: &Inst, name: &str, var: Word, addr: Word, fmt: u16, dst_sel: u16) -> R<()> {
    let u = tr.t.u32;
    let vd = match ins.enc {
        Enc::Global | Enc::Flat | Enc::Scratch => {
            if name.contains("_store") || name.contains("_atomic") {
                ins.flat_data() as u32
            } else {
                ins.vdst() as u32
            }
        }
        _ => ins.vdst() as u32,
    };
    let rdst = ins.vdst() as u32;
    let two = tr.u(2);
    if let Some(comps) = name.strip_prefix("buffer_load_format_") {
        if comps.starts_with("d16") {
            tr.warn(format!("{:#x}: {name} unsupported", ins.pc));
            return Ok(());
        }
        let n = comps.len();
        let v = load_formatted(tr, var, addr, fmt)?;
        let float = format::gfx10(fmt).is_none_or(|f| f.num.is_float());
        let v = apply_dst_sel(tr, v, dst_sel, float);
        for (k, x) in v.iter().enumerate().take(n) {
            tr.set_vreg(vd + k as u32, *x)?;
        }
        return Ok(());
    }
    if let Some(comps) = name.strip_prefix("buffer_store_format_") {
        if comps.starts_with("d16") {
            tr.warn(format!("{:#x}: {name} unsupported", ins.pc));
            return Ok(());
        }
        let n = comps.len() as u32;
        let mut vals = Vec::new();
        for k in 0..n {
            vals.push(tr.vreg(vd + k)?);
        }
        let act = tr.active()?;
        return tr.guarded(act, &mut |tr| store_formatted(tr, var, addr, &vals, fmt));
    }
    if let Some(kind) = name.strip_prefix("buffer_load_") {
        match kind {
            "ubyte" | "sbyte" | "ushort" | "sshort" => {
                let nb = if kind.ends_with("byte") { 1 } else { 2 };
                let raw = load_bytes(tr, var, addr, nb)?;
                let r = if kind.starts_with('s') {
                    let z = tr.u(0);
                    let w = tr.u(nb * 8);
                    e(tr.b.bit_field_s_extract(u, None, raw, z, w))?
                } else {
                    raw
                };
                tr.set_vreg(rdst, r)?;
            }
            "dword" | "dwordx2" | "dwordx3" | "dwordx4" => {
                let n = match kind {
                    "dword" => 1,
                    "dwordx2" => 2,
                    "dwordx3" => 3,
                    _ => 4,
                };
                let vals = load_dwords(tr, var, addr, n, false)?;
                for (k, v) in vals.into_iter().enumerate() {
                    tr.set_vreg(rdst + k as u32, v)?;
                }
            }
            "ubyte_d16" | "ubyte_d16_hi" | "sbyte_d16" | "sbyte_d16_hi" | "short_d16" | "short_d16_hi" => {
                let nb = if kind.contains("byte") { 1 } else { 2 };
                let raw = load_bytes(tr, var, addr, nb)?;
                let val = if kind.starts_with('s') && nb == 1 {
                    let z = tr.u(0);
                    let w = tr.u(8);
                    let s = e(tr.b.bit_field_s_extract(u, None, raw, z, w))?;
                    let m = tr.u(0xFFFF);
                    e(tr.b.bitwise_and(u, None, s, m))?
                } else {
                    raw
                };
                let old = tr.vreg(rdst)?;
                let r = if kind.ends_with("_hi") {
                    let k16 = tr.u(16);
                    let hs = e(tr.b.shift_left_logical(u, None, val, k16))?;
                    let m = tr.u(0xFFFF);
                    let lo = e(tr.b.bitwise_and(u, None, old, m))?;
                    e(tr.b.bitwise_or(u, None, lo, hs))?
                } else {
                    let m = tr.u(0xFFFF_0000);
                    let hi = e(tr.b.bitwise_and(u, None, old, m))?;
                    e(tr.b.bitwise_or(u, None, hi, val))?
                };
                tr.set_vreg(rdst, r)?;
            }
            _ => tr.warn(format!("{:#x}: unsupported {name}", ins.pc)),
        }
        return Ok(());
    }
    if let Some(kind) = name.strip_prefix("buffer_store_") {
        let act = tr.active()?;
        let mut unsupported = false;
        tr.guarded(act, &mut |tr| {
            match kind {
                "dword" | "dwordx2" | "dwordx3" | "dwordx4" => {
                    let n = match kind {
                        "dword" => 1,
                        "dwordx2" => 2,
                        "dwordx3" => 3,
                        _ => 4,
                    };
                    let mut vals = Vec::new();
                    for k in 0..n {
                        vals.push(tr.vreg(vd + k)?);
                    }
                    store_dwords(tr, var, addr, &vals)?;
                }
                "byte" | "short" | "byte_d16_hi" | "short_d16_hi" => {
                    let nb = if kind.starts_with("byte") { 1 } else { 2 };
                    let mut v = tr.vreg(vd)?;
                    if kind.ends_with("_hi") {
                        let k16 = tr.u(16);
                        v = e(tr.b.shift_right_logical(u, None, v, k16))?;
                    }
                    store_bytes(tr, var, addr, v, nb)?;
                }
                _ => unsupported = true,
            }
            Ok(())
        })?;
        if unsupported {
            tr.warn(format!("{:#x}: unsupported {name}", ins.pc));
        }
        return Ok(());
    }
    if let Some(op) = name.strip_prefix("buffer_atomic_") {
        if op.ends_with("_x2") || op.starts_with('f') || op == "csub" || op == "inc" || op == "dec" {
            tr.warn(format!("{:#x}: unsupported {name}", ins.pc));
            return Ok(());
        }
        let idx = e(tr.b.shift_right_logical(u, None, addr, two))?;
        let p = dword_ptr(tr, var, idx)?;
        let v = tr.vreg(vd)?;
        let cmp = if op == "cmpswap" { Some(tr.vreg(vd + 1)?) } else { None };
        let glc = match ins.enc {
            Enc::Global => bits(ins.w0, 16, 1) != 0,
            _ => bits(ins.w0, 14, 1) != 0,
        };
        let act = tr.active()?;
        let ret = if ins.enc == Enc::Global { rdst } else { vd };
        tr.guarded(act, &mut |tr| {
            if let Some(old) = atomic_op(tr, op, p, spirv::Scope::Device, v, cmp)? {
                if glc {
                    tr.set_vreg_raw(ret, old)?;
                }
            }
            Ok(())
        })?;
        return Ok(());
    }
    tr.warn(format!("{:#x}: unsupported {name}", ins.pc));
    Ok(())
}

// ---------------------------------------------------------------------------
// Images

/// Coordinate count of an instruction DIM (MIMG dim field).
fn inst_dim(d: u32) -> ImageDim {
    match d {
        0 => ImageDim::D1,
        1 => ImageDim::D2,
        2 => ImageDim::D3,
        3 => ImageDim::Cube,
        4 => ImageDim::D1Array,
        5 => ImageDim::D2Array,
        6 => ImageDim::D2Msaa,
        _ => ImageDim::D2MsaaArray,
    }
}

fn coord_count(dim: ImageDim) -> u32 {
    match dim {
        ImageDim::D1 => 1,
        ImageDim::D2 | ImageDim::D1Array => 2,
        ImageDim::D3 | ImageDim::Cube | ImageDim::D2Array | ImageDim::D2Msaa => 3,
        ImageDim::D2MsaaArray => 4,
    }
}

fn is_array(dim: ImageDim) -> bool {
    matches!(dim, ImageDim::D1Array | ImageDim::D2Array | ImageDim::D2MsaaArray)
}

/// Address VGPR `k` of an image instruction (NSA or contiguous).
fn addr_vgpr(ins: &Inst, k: u32) -> u32 {
    let nsa = bits(ins.w0, 1, 2);
    if nsa == 0 || k == 0 { bits(ins.w1, 0, 8) + if nsa == 0 { k } else { 0 } } else { ins.nsa[(k - 1) as usize] as u32 }
}

struct Addr<'i> {
    ins: &'i Inst,
    next: u32,
}

impl Addr<'_> {
    fn take(&mut self, tr: &mut T) -> R<Word> {
        let r = addr_vgpr(self.ins, self.next);
        self.next += 1;
        tr.vreg(r)
    }
}

/// Integer coordinates (instruction layout) adapted to the host image type:
/// 1D images are 2D on the host (y = 0); array layers are dropped or added.
fn int_coord(tr: &mut T, idim: ImageDim, rdim: ImageDim, cs: Vec<Word>) -> R<Word> {
    let mut xy: Vec<Word> = Vec::new();
    let n_space = match idim {
        ImageDim::D1 | ImageDim::D1Array => 1,
        ImageDim::D3 | ImageDim::Cube => 3,
        _ => 2,
    };
    for c in cs.iter().take(n_space) {
        xy.push(*c);
    }
    let layer = if is_array(idim) { cs.get(n_space).copied() } else { None };
    if matches!(rdim, ImageDim::D1 | ImageDim::D1Array | ImageDim::D2 | ImageDim::D2Array | ImageDim::D2Msaa | ImageDim::D2MsaaArray) {
        while xy.len() < 2 {
            xy.push(tr.i(0));
        }
        xy.truncate(2);
    }
    if is_array(rdim) {
        let l = match layer {
            Some(l) => l,
            None => tr.i(0),
        };
        xy.push(l);
    }
    let cty = match xy.len() {
        1 => tr.t.i32,
        2 => tr.t.v2i,
        3 => tr.t.v3i,
        _ => tr.t.v4i,
    };
    if xy.len() == 1 { Ok(xy[0]) } else { e(tr.b.composite_construct(cty, None, xy)) }
}

/// Cube coordinates (s+1.5, t+1.5, face) -> direction vector.
fn cube_dir(tr: &mut T, s: Word, t: Word, face: Word) -> R<Word> {
    let f = tr.t.f32;
    let bt = tr.t.bool_;
    let k15 = tr.f(1.5);
    let two = tr.f(2.0);
    let sc0 = e(tr.b.f_sub(f, None, s, k15))?;
    let tc0 = e(tr.b.f_sub(f, None, t, k15))?;
    let sc = e(tr.b.f_mul(f, None, sc0, two))?;
    let tc = e(tr.b.f_mul(f, None, tc0, two))?;
    let one = tr.f(1.0);
    let m1 = tr.f(-1.0);
    let nsc = e(tr.b.f_negate(f, None, sc))?;
    let ntc = e(tr.b.f_negate(f, None, tc))?;
    let fr = tr.glsl1(f, GLOp::RoundEven, face)?;
    // face: 0 +X, 1 -X, 2 +Y, 3 -Y, 4 +Z, 5 -Z
    let faces = [[one, ntc, nsc], [m1, ntc, sc], [sc, one, tc], [sc, m1, ntc], [sc, ntc, one], [nsc, ntc, m1]];
    let mut out = Vec::new();
    for c in 0..3 {
        let mut v = faces[5][c];
        for k in (0..5).rev() {
            let kf = tr.f(k as f32);
            let is = e(tr.b.f_ord_equal(bt, None, fr, kf))?;
            v = e(tr.b.select(f, None, is, faces[k][c], v))?;
        }
        out.push(v);
    }
    e(tr.b.composite_construct(tr.t.v3f, None, out))
}

pub fn image(tr: &mut T, ins: &Inst) -> R<()> {
    let name = ins.name().unwrap_or("");
    let u = tr.t.u32;
    let f = tr.t.f32;
    let Some(&(ri, si)) = tr.plan.inst_res.get(&ins.pc) else { return Err(format!("{:#x}: image op without resource", ins.pc)) };
    let (var, img_ty, rdim, class, storage) = {
        let ir = &tr.images[&ri];
        (ir.var, ir.img_ty, ir.dim, ir.class, ir.storage)
    };
    if ins.mimg_a16() || ins.mimg_d16() {
        tr.warn(format!("{:#x}: {name} with a16/d16 unsupported", ins.pc));
        return Ok(());
    }
    let idim = inst_dim(ins.mimg_dim());
    let vd = ins.vdst() as u32;
    let dmask = ins.mimg_dmask();
    let res_ty = match class {
        NumClass::Float => tr.t.v4f,
        NumClass::Sint => tr.t.v4i,
        NumClass::Uint => tr.t.v4u,
    };
    let comp_ty = match class {
        NumClass::Float => f,
        NumClass::Sint => tr.t.i32,
        NumClass::Uint => u,
    };
    let image = e(tr.b.load(img_ty, None, var, None, []))?;
    // Writes dmask-selected components of a vec4 result to consecutive VGPRs.
    let write_result = |tr: &mut T, res: Word| -> R<()> {
        let mut k = 0;
        for c in 0..4 {
            if dmask & (1 << c) != 0 {
                let x = e(tr.b.composite_extract(comp_ty, None, res, [c]))?;
                let xu = tr.to_u(x)?;
                tr.set_vreg(vd + k, xu)?;
                k += 1;
            }
        }
        Ok(())
    };
    let mut ad = Addr { ins, next: 0 };
    match name {
        "image_load" | "image_load_mip" => {
            let mut cs = Vec::new();
            for _ in 0..coord_count(idim) {
                let v = ad.take(tr)?;
                cs.push(tr.to_i(v)?);
            }
            let coord = int_coord(tr, idim, rdim, cs)?;
            let res = if storage {
                e(tr.b.image_read(res_ty, None, image, coord, None, []))?
            } else if matches!(rdim, ImageDim::D2Msaa | ImageDim::D2MsaaArray) {
                let s = tr.i(0);
                e(tr.b.image_fetch(res_ty, None, image, coord, Some(ImageOperands::SAMPLE), [Operand::IdRef(s)]))?
            } else {
                let lod = if name == "image_load_mip" {
                    let v = ad.take(tr)?;
                    tr.to_i(v)?
                } else {
                    tr.i(0)
                };
                e(tr.b.image_fetch(res_ty, None, image, coord, Some(ImageOperands::LOD), [Operand::IdRef(lod)]))?
            };
            write_result(tr, res)
        }
        "image_store" | "image_store_mip" => {
            if !storage {
                return Err(format!("{:#x}: image_store to a non-storage image", ins.pc));
            }
            let mut cs = Vec::new();
            for _ in 0..coord_count(idim) {
                let v = ad.take(tr)?;
                cs.push(tr.to_i(v)?);
            }
            if name == "image_store_mip" {
                let lod = ad.take(tr)?;
                let _ = lod;
                tr.warn(format!("{:#x}: image_store_mip writes the bound mip level", ins.pc));
            }
            let coord = int_coord(tr, idim, rdim, cs)?;
            let mut comps = Vec::new();
            let mut k = 0;
            for c in 0..4 {
                let v = if dmask & (1 << c) != 0 {
                    let x = tr.vreg(vd + k)?;
                    k += 1;
                    x
                } else {
                    tr.u(0)
                };
                comps.push(match class {
                    NumClass::Float => tr.to_f(v)?,
                    NumClass::Sint => tr.to_i(v)?,
                    NumClass::Uint => v,
                });
            }
            let texel = e(tr.b.composite_construct(res_ty, None, comps))?;
            let act = tr.active()?;
            tr.guarded(act, &mut |tr| ok(tr.b.image_write(image, coord, texel, None, [])))
        }
        "image_get_resinfo" => {
            // [width, height, depth or layers, levels]
            let lodv = ad.take(tr)?;
            let lod = tr.to_i(lodv)?;
            let (n, sty) = match rdim {
                ImageDim::D1 | ImageDim::D2 | ImageDim::Cube | ImageDim::D2Msaa => (2, tr.t.v2i),
                _ => (3, tr.t.v3i),
            };
            let ms = matches!(rdim, ImageDim::D2Msaa | ImageDim::D2MsaaArray);
            let size = if storage || ms { e(tr.b.image_query_size(sty, None, image))? } else { e(tr.b.image_query_size_lod(sty, None, image, lod))? };
            let levels = if storage || ms { tr.i(1) } else { e(tr.b.image_query_levels(tr.t.i32, None, image))? };
            // 1D images are 2D on the host: 1D arrays report (width, layers).
            let order: [u32; 3] = if rdim == ImageDim::D1Array { [0, 2, 1] } else { [0, 1, 2] };
            let mut vals = Vec::new();
            for (pos, c) in order.into_iter().enumerate() {
                let one_d = matches!(rdim, ImageDim::D1) && pos == 1;
                if c < n && !one_d {
                    vals.push(e(tr.b.composite_extract(tr.t.i32, None, size, [c]))?);
                } else {
                    vals.push(tr.i(1));
                }
            }
            vals.push(levels);
            let mut k = 0;
            for (c, v) in vals.iter().enumerate() {
                if dmask & (1 << c) != 0 {
                    let x = tr.to_u(*v)?;
                    tr.set_vreg(vd + k, x)?;
                    k += 1;
                }
            }
            Ok(())
        }
        n if n.starts_with("image_sample") || n.starts_with("image_gather4") || n == "image_get_lod" => sample(tr, ins, name, image, img_ty, si, idim, rdim, res_ty, &write_result),
        _ => {
            tr.warn(format!("{:#x}: unsupported {name}", ins.pc));
            Ok(())
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn sample(tr: &mut T, ins: &Inst, name: &str, image: Word, img_ty: Word, si: Option<usize>, idim: ImageDim, rdim: ImageDim, res_ty: Word, write_result: &dyn Fn(&mut T, Word) -> R<()>) -> R<()> {
    let f = tr.t.f32;
    let gather = name.starts_with("image_gather4");
    let get_lod = name == "image_get_lod";
    let mods: Vec<&str> = name.trim_start_matches("image_sample").trim_start_matches("image_gather4").split('_').filter(|s| !s.is_empty()).collect();
    let has = |m: &str| mods.contains(&m);
    let has_o = has("o");
    let has_c = has("c");
    let has_d = has("d") || has("cd");
    let has_l = has("l");
    let has_b = has("b");
    let has_cl = has("cl");
    let has_lz = has("lz");
    let Some(sidx) = si else { return Err(format!("{:#x}: sample without sampler", ins.pc)) };
    let svar = tr.samplers[&sidx];
    let sampler = e(tr.b.load(tr.sampler_ty, None, svar, None, []))?;
    let si_ty = tr.b.type_sampled_image(img_ty);
    let simg = e(tr.b.sampled_image(si_ty, None, image, sampler))?;
    let mut ad = Addr { ins, next: 0 };
    let offset = if has_o { Some(ad.take(tr)?) } else { None };
    let bias = if has_b { Some(ad.take(tr)?) } else { None };
    let zcmp = if has_c { Some(ad.take(tr)?) } else { None };
    let gdims = match idim {
        ImageDim::D1 | ImageDim::D1Array => 1,
        ImageDim::D3 => 3,
        _ => 2,
    };
    let derivs = if has_d {
        let mut dx = Vec::new();
        let mut dy = Vec::new();
        for _ in 0..gdims {
            let v = ad.take(tr)?;
            dx.push(tr.to_f(v)?);
        }
        for _ in 0..gdims {
            let v = ad.take(tr)?;
            dy.push(tr.to_f(v)?);
        }
        Some((dx, dy))
    } else {
        None
    };
    let mut cs = Vec::new();
    for _ in 0..coord_count(idim) {
        let v = ad.take(tr)?;
        cs.push(tr.to_f(v)?);
    }
    let lod_or_clamp = if has_l || has_cl { Some(ad.take(tr)?) } else { None };
    let cube = matches!(rdim, ImageDim::Cube);
    let coord = if cube {
        if idim != ImageDim::Cube {
            tr.warn(format!("{:#x}: cube image sampled with dim {:?}", ins.pc, idim));
        }
        let face = cs.get(2).copied().unwrap_or_else(|| tr.f(0.0));
        cube_dir(tr, cs[0], cs[1], face)?
    } else {
        let n_space = match idim {
            ImageDim::D1 | ImageDim::D1Array => 1,
            ImageDim::D3 | ImageDim::Cube => 3,
            _ => 2,
        };
        let mut v: Vec<Word> = cs.iter().take(n_space).copied().collect();
        let layer = if is_array(idim) { cs.get(n_space).copied() } else { None };
        if matches!(rdim, ImageDim::D1 | ImageDim::D1Array | ImageDim::D2 | ImageDim::D2Array) {
            if v.len() == 1 {
                // Row centre of the height-1 host image.
                v.push(tr.f(0.5));
            }
            v.truncate(2);
        }
        if is_array(rdim) {
            let l = match layer {
                Some(l) => l,
                None => tr.f(0.0),
            };
            v.push(l);
        }
        match v.len() {
            1 => v[0],
            2 => e(tr.b.composite_construct(tr.t.v2f, None, v))?,
            3 => e(tr.b.composite_construct(tr.t.v3f, None, v))?,
            _ => e(tr.b.composite_construct(tr.t.v4f, None, v))?,
        }
    };
    if get_lod {
        if tr.spec.stage != Stage::Fragment {
            tr.warn(format!("{:#x}: image_get_lod outside fragment shaders returns 0", ins.pc));
            let z = tr.f(0.0);
            let r = e(tr.b.composite_construct(tr.t.v4f, None, [z, z, z, z]))?;
            return write_result(tr, r);
        }
        let l = e(tr.b.image_query_lod(tr.t.v2f, None, simg, coord))?;
        let x = e(tr.b.composite_extract(f, None, l, [0]))?;
        let y = e(tr.b.composite_extract(f, None, l, [1]))?;
        let z = tr.f(0.0);
        let r = e(tr.b.composite_construct(tr.t.v4f, None, [x, y, z, z]))?;
        return write_result(tr, r);
    }
    let implicit_ok = tr.spec.stage == Stage::Fragment;
    let mut operands: Vec<Operand> = Vec::new();
    let mut mask = ImageOperands::NONE;
    let explicit;
    if let (Some((dx, dy)), false) = (&derivs, cube) {
        let (vty, dx, dy) = if gdims == 1 {
            // 1D images are 2D on the host: zero y gradient.
            let z = tr.f(0.0);
            (tr.t.v2f, vec![dx[0], z], vec![dy[0], z])
        } else {
            (if gdims == 2 { tr.t.v2f } else { tr.t.v3f }, dx.clone(), dy.clone())
        };
        let gx = e(tr.b.composite_construct(vty, None, dx))?;
        let gy = e(tr.b.composite_construct(vty, None, dy))?;
        mask |= ImageOperands::GRAD;
        operands.push(Operand::IdRef(gx));
        operands.push(Operand::IdRef(gy));
        explicit = true;
    } else if has_l || has_lz || !implicit_ok || (derivs.is_some() && cube) {
        if derivs.is_some() {
            tr.warn(format!("{:#x}: cube derivatives replaced by lod 0", ins.pc));
        }
        let lod = if has_l {
            let v = lod_or_clamp.unwrap();
            tr.to_f(v)?
        } else {
            tr.f(0.0)
        };
        mask |= ImageOperands::LOD;
        operands.push(Operand::IdRef(lod));
        explicit = true;
    } else {
        if let Some(bv) = bias {
            let bf = tr.to_f(bv)?;
            mask |= ImageOperands::BIAS;
            operands.insert(0, Operand::IdRef(bf));
        }
        explicit = false;
    }
    let mut coord = coord;
    if let (Some(ov), false) = (offset, gather) {
        if matches!(rdim, ImageDim::D2 | ImageDim::D2Array) {
            // Vulkan permits the Offset operand only on gathers: shift the
            // coordinates by offset / level size instead.
            let u = tr.t.u32;
            let z = tr.u(0);
            let six = tr.u(6);
            let k8 = tr.u(8);
            let x = e(tr.b.bit_field_s_extract(u, None, ov, z, six))?;
            let y = e(tr.b.bit_field_s_extract(u, None, ov, k8, six))?;
            let xi = tr.to_i(x)?;
            let yi = tr.to_i(y)?;
            let xf = e(tr.b.convert_s_to_f(f, None, xi))?;
            let yf = e(tr.b.convert_s_to_f(f, None, yi))?;
            let level = if has_l {
                let lv = tr.to_f(lod_or_clamp.unwrap())?;
                let r = tr.glsl1(f, GLOp::RoundEven, lv)?;
                e(tr.b.convert_f_to_s(tr.t.i32, None, r))?
            } else if !has_lz && implicit_ok && derivs.is_none() {
                let q = e(tr.b.image_query_lod(tr.t.v2f, None, simg, coord))?;
                let lv = e(tr.b.composite_extract(f, None, q, [0]))?;
                let r = tr.glsl1(f, GLOp::RoundEven, lv)?;
                e(tr.b.convert_f_to_s(tr.t.i32, None, r))?
            } else {
                tr.i(0)
            };
            let sty = if rdim == ImageDim::D2Array { tr.t.v3i } else { tr.t.v2i };
            let size = e(tr.b.image_query_size_lod(sty, None, image, level))?;
            let w = e(tr.b.composite_extract(tr.t.i32, None, size, [0]))?;
            let h = e(tr.b.composite_extract(tr.t.i32, None, size, [1]))?;
            let wf = e(tr.b.convert_s_to_f(f, None, w))?;
            let hf = e(tr.b.convert_s_to_f(f, None, h))?;
            let dx = e(tr.b.f_div(f, None, xf, wf))?;
            let dy = e(tr.b.f_div(f, None, yf, hf))?;
            let cx = e(tr.b.composite_extract(f, None, coord, [0]))?;
            let cy = e(tr.b.composite_extract(f, None, coord, [1]))?;
            let nx = e(tr.b.f_add(f, None, cx, dx))?;
            let ny = e(tr.b.f_add(f, None, cy, dy))?;
            let cty = if rdim == ImageDim::D2Array { tr.t.v3f } else { tr.t.v2f };
            coord = e(tr.b.composite_insert(cty, None, nx, coord, [0]))?;
            coord = e(tr.b.composite_insert(cty, None, ny, coord, [1]))?;
        } else {
            tr.warn(format!("{:#x}: sample offset on non-2D image ignored", ins.pc));
        }
    }
    if let (Some(ov), true) = (offset, gather) {
        if matches!(rdim, ImageDim::D2 | ImageDim::D2Array) {
            let u = tr.t.u32;
            let z = tr.u(0);
            let six = tr.u(6);
            let k8 = tr.u(8);
            let x = e(tr.b.bit_field_s_extract(u, None, ov, z, six))?;
            let y = e(tr.b.bit_field_s_extract(u, None, ov, k8, six))?;
            let xi = tr.to_i(x)?;
            let yi = tr.to_i(y)?;
            let off = e(tr.b.composite_construct(tr.t.v2i, None, [xi, yi]))?;
            mask |= ImageOperands::OFFSET;
            operands.push(Operand::IdRef(off));
        } else {
            tr.warn(format!("{:#x}: sample offset on non-2D image ignored", ins.pc));
        }
    }
    if has_cl && !has_l {
        tr.warn(format!("{:#x}: min-lod clamp ignored", ins.pc));
    }
    let m = if mask == ImageOperands::NONE { None } else { Some(mask) };
    let res = if gather {
        let comp = tr.u(match ins.mimg_dmask() {
            2 => 1,
            4 => 2,
            8 => 3,
            _ => 0,
        });
        // Gather has no LOD/gradient operands (it samples level 0).
        let (m, operands) = if explicit {
            let keep: Vec<Operand> = if mask.contains(ImageOperands::OFFSET) { vec![operands.last().unwrap().clone()] } else { vec![] };
            (if keep.is_empty() { None } else { Some(ImageOperands::OFFSET) }, keep)
        } else {
            (m, operands)
        };
        if let Some(zc) = zcmp {
            let zf = tr.to_f(zc)?;
            e(tr.b.image_dref_gather(res_ty, None, simg, coord, zf, m, operands))?
        } else {
            e(tr.b.image_gather(res_ty, None, simg, coord, comp, m, operands))?
        }
    } else if let Some(zc) = zcmp {
        let zf = tr.to_f(zc)?;
        let r = if explicit { e(tr.b.image_sample_dref_explicit_lod(f, None, simg, coord, zf, mask, operands))? } else { e(tr.b.image_sample_dref_implicit_lod(f, None, simg, coord, zf, m, operands))? };
        let z = tr.f(0.0);
        let one = tr.f(1.0);
        e(tr.b.composite_construct(tr.t.v4f, None, [r, z, z, one]))?
    } else if explicit {
        e(tr.b.image_sample_explicit_lod(res_ty, None, simg, coord, mask, operands))?
    } else {
        e(tr.b.image_sample_implicit_lod(res_ty, None, simg, coord, m, operands))?
    };
    write_result(tr, res)
}

// ---------------------------------------------------------------------------
// Exports

pub fn export(tr: &mut T, ins: &Inst) -> R<()> {
    let tgt = ins.exp_target();
    let en = ins.exp_en();
    let f = tr.t.f32;
    let mut vals = Vec::new();
    for k in 0..4 {
        let v = tr.vreg(ins.exp_src(k) as u32)?;
        vals.push(v);
    }
    let act = tr.active()?;
    // Kill inactive fragments at the valid-mask export.
    if tr.spec.stage == Stage::Fragment && ins.exp_vm() {
        let nact = e(tr.b.logical_not(tr.t.bool_, None, act))?;
        let merge = tr.b.id();
        let kill = tr.b.id();
        ok(tr.b.selection_merge(merge, spirv::SelectionControl::NONE))?;
        ok(tr.b.branch_conditional(nact, kill, merge, []))?;
        e(tr.b.begin_block(Some(kill)))?;
        ok(tr.b.kill())?;
        e(tr.b.begin_block(Some(merge)))?;
    }
    match tgt {
        0..=7 => {
            let mrt = tr.spec.mrt[tgt as usize];
            let comps: Vec<Word> = if ins.exp_compr() {
                let mut c = Vec::new();
                for &w in vals.iter().take(2) {
                    let op = match mrt.export_format {
                        5 => Some(GLOp::UnpackUnorm2x16),
                        6 => Some(GLOp::UnpackSnorm2x16),
                        7 | 8 => None,
                        _ => Some(GLOp::UnpackHalf2x16),
                    };
                    if let Some(op) = op {
                        let v = tr.glsl1(tr.t.v2f, op, w)?;
                        c.push(e(tr.b.composite_extract(f, None, v, [0]))?);
                        c.push(e(tr.b.composite_extract(f, None, v, [1]))?);
                    } else {
                        let u = tr.t.u32;
                        let z = tr.u(0);
                        let k16 = tr.u(16);
                        let (lo, hi) = if mrt.export_format == 8 {
                            (e(tr.b.bit_field_s_extract(u, None, w, z, k16))?, e(tr.b.bit_field_s_extract(u, None, w, k16, k16))?)
                        } else {
                            (e(tr.b.bit_field_u_extract(u, None, w, z, k16))?, e(tr.b.bit_field_u_extract(u, None, w, k16, k16))?)
                        };
                        c.push(tr.to_f(lo)?);
                        c.push(tr.to_f(hi)?);
                    }
                }
                c
            } else {
                let mut c = Vec::new();
                for v in vals.iter().take(4) {
                    c.push(tr.to_f(*v)?);
                }
                c
            };
            let (ty, var) = match tr.out_mrt.get(&tgt) {
                Some(&x) => x,
                None => {
                    let ty = match mrt.class {
                        1 => tr.t.v4i,
                        2 => tr.t.v4u,
                        _ => tr.t.v4f,
                    };
                    let v = tr.output(ty, None, Some(tgt));
                    tr.out_mrt.insert(tgt, (ty, v));
                    (ty, v)
                }
            };
            tr.mrt_mask |= 1 << tgt;
            let mut cs = Vec::new();
            for c in comps {
                cs.push(if ty == tr.t.v4f {
                    c
                } else {
                    let u = tr.to_u(c)?;
                    if ty == tr.t.v4i { tr.to_i(u)? } else { u }
                });
            }
            while cs.len() < 4 {
                cs.push(if ty == tr.t.v4f { tr.f(0.0) } else if ty == tr.t.v4i { tr.i(0) } else { tr.u(0) });
            }
            let v = e(tr.b.composite_construct(ty, None, cs))?;
            tr.guarded(act, &mut |tr| ok(tr.b.store(var, v, None, [])))?;
        }
        8 => {
            if en & 1 != 0 {
                let var = match tr.out_depth {
                    Some(v) => v,
                    None => {
                        let v = tr.output(f, Some(BuiltIn::FragDepth), None);
                        tr.out_depth = Some(v);
                        v
                    }
                };
                let d = tr.to_f(vals[0])?;
                tr.guarded(act, &mut |tr| ok(tr.b.store(var, d, None, [])))?;
            }
            if en & 0xE != 0 {
                tr.warn("mrtz stencil/sample-mask export ignored".into());
            }
        }
        9 | 20 => {} // null; NGG primitive export
        12 => {
            let var = match tr.out_pos {
                Some(v) => v,
                None => {
                    let v = tr.output(tr.t.v4f, Some(BuiltIn::Position), None);
                    tr.out_pos = Some(v);
                    v
                }
            };
            let mut cs = Vec::new();
            for v in vals.iter() {
                cs.push(tr.to_f(*v)?);
            }
            let p = e(tr.b.composite_construct(tr.t.v4f, None, cs))?;
            ok(tr.b.store(var, p, None, []))?;
        }
        13..=15 => tr.warn(format!("export to pos{} ignored", tgt - 12)),
        32..=63 => {
            let p = tgt - 32;
            let var = match tr.out_params.get(&p) {
                Some(&v) => v,
                None => {
                    let v = tr.output(tr.t.v4f, None, Some(p));
                    tr.out_params.insert(p, v);
                    v
                }
            };
            tr.params = tr.params.max(p + 1);
            let mut cs = Vec::new();
            for (k, v) in vals.iter().enumerate() {
                // Disabled channels export 0.
                cs.push(if en >> k & 1 != 0 { tr.to_f(*v)? } else { tr.f(0.0) });
            }
            let pv = e(tr.b.composite_construct(tr.t.v4f, None, cs))?;
            ok(tr.b.store(var, pv, None, []))?;
        }
        _ => tr.warn(format!("export target {tgt} ignored")),
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// LDS / GDS

/// GDS instructions on the shared GDS storage buffer. Address: M0[15:0]
/// (allocation base) + instruction offset (+ VGPR address except for
/// append/consume/ordered count). Lanes are waves of one, so ds_append /
/// ds_consume add / subtract 1 per active lane and return the pre-op value.
fn gds(tr: &mut T, ins: &Inst, name: &str) -> R<()> {
    let u = tr.t.u32;
    let Some(gds) = tr.gds else { return Err(format!("{:#x}: GDS binding missing", ins.pc)) };
    let append = matches!(name, "ds_append" | "ds_consume" | "ds_ordered_count");
    if name.starts_with("ds_gws_") {
        tr.warn(format!("{:#x}: {name} (global wave sync) ignored", ins.pc));
        return Ok(());
    }
    if name == "ds_ordered_count" {
        tr.warn(format!("{:#x}: ds_ordered_count approximated by an unordered atomic add", ins.pc));
    }
    let off16 = ins.ds_offset0() | (ins.ds_offset1() << 8);
    let m0 = tr.sreg(crate::OP_M0 as u32)?;
    let lo16 = tr.u(0xFFFF);
    let base = e(tr.b.bitwise_and(u, None, m0, lo16))?;
    let k = tr.u(off16);
    let mut addr = e(tr.b.i_add(u, None, base, k))?;
    if !append {
        let va = tr.vreg(ins.ds_addr() as u32)?;
        addr = e(tr.b.i_add(u, None, addr, va))?;
    }
    let two = tr.u(2);
    let dw = e(tr.b.shift_right_logical(u, None, addr, two))?;
    let lim = tr.u(super::GDS_BYTES / 4 - 1);
    let dw = e(tr.b.bitwise_and(u, None, dw, lim))?;
    let p = dword_ptr(tr, gds, dw)?;
    let vd = ins.vdst() as u32;
    let d0 = ins.ds_data0() as u32;
    let act = tr.active()?;
    let mut unsupported = false;
    let name = name.to_string();
    tr.guarded(act, &mut |tr| {
        let one = tr.u(1);
        let ret = match name.as_str() {
            "ds_append" => atomic_op(tr, "add", p, spirv::Scope::Device, one, None)?,
            "ds_consume" => atomic_op(tr, "sub", p, spirv::Scope::Device, one, None)?,
            "ds_ordered_count" => {
                let v = tr.vreg(d0)?;
                atomic_op(tr, "add", p, spirv::Scope::Device, v, None)?
            }
            "ds_write_b32" => {
                let v = tr.vreg(d0)?;
                ok(tr.b.store(p, v, None, []))?;
                None
            }
            "ds_read_b32" => Some(e(tr.b.load(u, None, p, None, []))?),
            n => {
                let rtn = n.contains("_rtn_");
                let op = n.trim_start_matches("ds_").replace("_rtn", "");
                let op = op.trim_end_matches("_u32").trim_end_matches("_i32").trim_end_matches("_b32");
                let op = match op {
                    "min" if n.ends_with("i32") => "smin",
                    "max" if n.ends_with("i32") => "smax",
                    "min" => "umin",
                    "max" => "umax",
                    "wrxchg" => "swap",
                    o => o,
                };
                let v = tr.vreg(d0)?;
                match atomic_op(tr, op, p, spirv::Scope::Device, v, None)? {
                    Some(old) => rtn.then_some(old),
                    None => {
                        unsupported = true;
                        None
                    }
                }
            }
        };
        if let Some(r) = ret {
            tr.set_vreg_raw(vd, r)?;
        }
        Ok(())
    })?;
    if unsupported {
        tr.warn(format!("{:#x}: unsupported GDS {}", ins.pc, ins.name().unwrap_or("")));
    }
    Ok(())
}

/// ds_swizzle_b32 (no LDS access): offset bit 15 selects quad permute mode
/// (lane i of each quad reads lane offset[2i+1:2i]); otherwise lanes read
/// lane ((id & and) | or) ^ xor within groups of 32.
fn swizzle(tr: &mut T, ins: &Inst) -> R<()> {
    let u = tr.t.u32;
    let off = ins.ds_offset0() | (ins.ds_offset1() << 8);
    let v = tr.vreg(ins.ds_data0() as u32)?;
    let lane = tr.subgroup_lane()?;
    let src = if off & 0x8000 != 0 {
        let three = tr.u(3);
        let q = e(tr.b.bitwise_and(u, None, lane, three))?;
        let qbase = e(tr.b.i_sub(u, None, lane, q))?;
        let two = tr.u(2);
        let sh = e(tr.b.i_mul(u, None, q, two))?;
        let c = tr.u(off & 0xFF);
        let sel = e(tr.b.shift_right_logical(u, None, c, sh))?;
        let sel = e(tr.b.bitwise_and(u, None, sel, three))?;
        e(tr.b.bitwise_or(u, None, qbase, sel))?
    } else {
        let (and, or, xor) = (off & 0x1F, (off >> 5) & 0x1F, (off >> 10) & 0x1F);
        let m31 = tr.u(31);
        let l = e(tr.b.bitwise_and(u, None, lane, m31))?;
        let hi_mask = tr.u(!31);
        let hi = e(tr.b.bitwise_and(u, None, lane, hi_mask))?;
        let ka = tr.u(and);
        let ko = tr.u(or);
        let kx = tr.u(xor);
        let s = e(tr.b.bitwise_and(u, None, l, ka))?;
        let s = e(tr.b.bitwise_or(u, None, s, ko))?;
        let s = e(tr.b.bitwise_xor(u, None, s, kx))?;
        e(tr.b.bitwise_or(u, None, hi, s))?
    };
    let (r, _) = tr.lane_fetch_active(v, src)?;
    tr.set_vreg(ins.vdst() as u32, r)
}

pub fn ds(tr: &mut T, ins: &Inst) -> R<()> {
    let name = ins.name().unwrap_or("");
    if ins.ds_gds() {
        return gds(tr, ins, name);
    }
    if name == "ds_swizzle_b32" {
        return swizzle(tr, ins);
    }
    let u = tr.t.u32;
    let off16 = ins.ds_offset0() | (ins.ds_offset1() << 8);
    let vd = ins.vdst() as u32;
    let d0 = ins.ds_data0() as u32;
    let d1 = ins.ds_data1() as u32;
    if name == "ds_bpermute_b32" || name == "ds_permute_b32" {
        // Lane-addressed exchange (address / 4 = lane) within 32 lanes.
        let a = tr.vreg(ins.ds_addr() as u32)?;
        let k = tr.u(off16);
        let a = e(tr.b.i_add(u, None, a, k))?;
        let two = tr.u(2);
        let l = e(tr.b.shift_right_logical(u, None, a, two))?;
        let k31 = tr.u(31);
        let l = e(tr.b.bitwise_and(u, None, l, k31))?;
        let v = tr.vreg(d0)?;
        if name == "ds_permute_b32" {
            tr.warn(format!("{:#x}: ds_permute_b32 approximated as a backward permute", ins.pc));
        }
        let (r, _) = tr.lane_fetch_active(v, l)?;
        return tr.set_vreg(vd, r);
    }
    let Some(lds) = tr.lds else {
        tr.warn(format!("{:#x}: {name} without LDS", ins.pc));
        return Ok(());
    };
    let p_wg = tr.b.type_pointer(None, StorageClass::Workgroup, u);
    let mut addr = tr.vreg(ins.ds_addr() as u32)?;
    if name.ends_with("_addtid_b32") {
        // Address = M0 base + offset + lane id * 4.
        let m0 = tr.sreg(crate::OP_M0 as u32)?;
        let lane = tr.lane()?;
        let k4 = tr.u(4);
        let t = e(tr.b.i_mul(u, None, lane, k4))?;
        addr = e(tr.b.i_add(u, None, m0, t))?;
    }
    let two = tr.u(2);
    let elem = |tr: &mut T, byte: Word| -> R<Word> {
        let i = e(tr.b.shift_right_logical(u, None, byte, two))?;
        e(tr.b.access_chain(p_wg, None, lds, [i]))
    };
    let add = |tr: &mut T, a: Word, k: u32| -> R<Word> {
        let kk = tr.u(k);
        e(tr.b.i_add(u, None, a, kk))
    };
    let act = tr.active()?;
    let nwords = |n: &str| -> u32 {
        if n.ends_with("b128") {
            4
        } else if n.ends_with("b96") {
            3
        } else if n.ends_with("b64") {
            2
        } else {
            1
        }
    };
    match name {
        "ds_write_b32" | "ds_write_b64" | "ds_write_b96" | "ds_write_b128" | "ds_write_addtid_b32" => {
            let n = nwords(name);
            tr.guarded(act, &mut |tr| {
                let a = add(tr, addr, off16)?;
                for k in 0..n {
                    let ak = add(tr, a, k * 4)?;
                    let p = elem(tr, ak)?;
                    let v = tr.vreg(d0 + k)?;
                    ok(tr.b.store(p, v, None, []))?;
                }
                Ok(())
            })?;
        }
        "ds_write2_b32" | "ds_write2st64_b32" | "ds_write2_b64" | "ds_write2st64_b64" => {
            let st = if name.contains("st64") { 64 } else { 1 };
            let esz = if name.ends_with("b64") { 8 } else { 4 };
            tr.guarded(act, &mut |tr| {
                for (off, src) in [(ins.ds_offset0(), d0), (ins.ds_offset1(), d1)] {
                    let a = add(tr, addr, off * esz * st)?;
                    for w in 0..(esz / 4) {
                        let aw = add(tr, a, w * 4)?;
                        let p = elem(tr, aw)?;
                        let v = tr.vreg(src + w)?;
                        ok(tr.b.store(p, v, None, []))?;
                    }
                }
                Ok(())
            })?;
        }
        "ds_write_b8" | "ds_write_b16" | "ds_write_b8_d16_hi" | "ds_write_b16_d16_hi" => {
            let nb = if name.contains("b8") { 8 } else { 16 };
            tr.guarded(act, &mut |tr| {
                let a = add(tr, addr, off16)?;
                let p = elem(tr, a)?;
                let mut v = tr.vreg(d0)?;
                if name.ends_with("_hi") {
                    let k16 = tr.u(16);
                    v = e(tr.b.shift_right_logical(u, None, v, k16))?;
                }
                let k3 = tr.u(3);
                let sub = e(tr.b.bitwise_and(u, None, a, k3))?;
                let sh = e(tr.b.shift_left_logical(u, None, sub, k3))?;
                let m = tr.u((1u32 << nb) - 1);
                let mask = e(tr.b.shift_left_logical(u, None, m, sh))?;
                let vm = e(tr.b.bitwise_and(u, None, v, m))?;
                let vs = e(tr.b.shift_left_logical(u, None, vm, sh))?;
                let scope = tr.u(spirv::Scope::Workgroup as u32);
                let sem = tr.u(0);
                let nm = e(tr.b.not(u, None, mask))?;
                e(tr.b.atomic_and(u, None, p, scope, sem, nm))?;
                e(tr.b.atomic_or(u, None, p, scope, sem, vs))?;
                Ok(())
            })?;
        }
        "ds_read_b32" | "ds_read_b64" | "ds_read_b96" | "ds_read_b128" | "ds_read_addtid_b32" => {
            let n = nwords(name);
            let a = add(tr, addr, off16)?;
            for k in 0..n {
                let ak = add(tr, a, k * 4)?;
                let p = elem(tr, ak)?;
                let v = e(tr.b.load(u, None, p, None, []))?;
                tr.set_vreg(vd + k, v)?;
            }
        }
        "ds_read2_b32" | "ds_read2st64_b32" | "ds_read2_b64" | "ds_read2st64_b64" => {
            let st = if name.contains("st64") { 64 } else { 1 };
            let esz = if name.ends_with("b64") { 8 } else { 4 };
            let mut dst = vd;
            for off in [ins.ds_offset0(), ins.ds_offset1()] {
                let a = add(tr, addr, off * esz * st)?;
                for w in 0..(esz / 4) {
                    let aw = add(tr, a, w * 4)?;
                    let p = elem(tr, aw)?;
                    let v = e(tr.b.load(u, None, p, None, []))?;
                    tr.set_vreg(dst, v)?;
                    dst += 1;
                }
            }
        }
        "ds_read_u8" | "ds_read_i8" | "ds_read_u16" | "ds_read_i16" => {
            let nb = if name.ends_with('8') { 8 } else { 16 };
            let a = add(tr, addr, off16)?;
            let p = elem(tr, a)?;
            let w = e(tr.b.load(u, None, p, None, []))?;
            let k3 = tr.u(3);
            let sub = e(tr.b.bitwise_and(u, None, a, k3))?;
            let sh = e(tr.b.shift_left_logical(u, None, sub, k3))?;
            let kw = tr.u(nb);
            let r = if name.contains("_i") { e(tr.b.bit_field_s_extract(u, None, w, sh, kw))? } else { e(tr.b.bit_field_u_extract(u, None, w, sh, kw))? };
            tr.set_vreg(vd, r)?;
        }
        n if n.ends_with("_u32") || n.ends_with("_i32") || n.ends_with("_b32") => {
            // Atomics; only active lanes perform them.
            let rtn = n.contains("_rtn_");
            let op = n.trim_start_matches("ds_").replace("_rtn", "");
            let signed = op.ends_with("_i32");
            let op = op.trim_end_matches("_u32").trim_end_matches("_i32").trim_end_matches("_b32").to_string();
            let op = match op.as_str() {
                "min" if signed => "smin".to_string(),
                "max" if signed => "smax".to_string(),
                "min" => "umin".to_string(),
                "max" => "umax".to_string(),
                "wrxchg" => "swap".to_string(),
                "cmpst" => "cmpswap".to_string(),
                o => o.to_string(),
            };
            let mut unsupported = false;
            tr.guarded(act, &mut |tr| {
                let a = add(tr, addr, off16)?;
                let p = elem(tr, a)?;
                let v = tr.vreg(d0)?;
                // ds_cmpst: data0 = compare value, data1 = new value.
                let (v, cmp) = if op == "cmpswap" { (tr.vreg(d1)?, Some(v)) } else { (v, None) };
                match atomic_op(tr, &op, p, spirv::Scope::Workgroup, v, cmp)? {
                    Some(old) if rtn => tr.set_vreg_raw(vd, old)?,
                    Some(_) => {}
                    None => unsupported = true,
                }
                Ok(())
            })?;
            if unsupported {
                tr.warn(format!("{:#x}: unsupported {name}", ins.pc));
            }
        }
        _ => tr.warn(format!("{:#x}: unsupported {name}", ins.pc)),
    }
    Ok(())
}
