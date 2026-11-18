//! Generated geometry shader that expands rect-list primitives: each input
//! triangle (v0, v1, v2) becomes the strip v0, v1, v2, v3 with
//! v3 = v1 + v2 - v0 (the parallelogram's fourth corner) for the position and
//! every parameter. Ported from the earlier native port's GCN translator
//! (MIT); parameters are passed for the locations in a mask.

use rspirv::binary::Assemble;
use rspirv::dr::{Builder, Operand};
use rspirv::spirv::{self, BuiltIn, Decoration, ExecutionMode, ExecutionModel, StorageClass};

/// Geometry shader passing through the vec4 parameters at the locations set
/// in `param_mask` (bit n = Location n: the vertex stage's outputs) and
/// writing zeros at the other locations of `out_mask` (a pixel stage's inputs
/// the vertex stage does not write: every input needs an output before it).
pub fn rect_list_gs(param_mask: u32, out_mask: u32) -> Vec<u32> {
    let mut b = Builder::new();
    b.set_version(1, 3);
    b.capability(spirv::Capability::Geometry);
    b.capability(spirv::Capability::Shader);
    b.memory_model(spirv::AddressingModel::Logical, spirv::MemoryModel::GLSL450);
    let void = b.type_void();
    let f32t = b.type_float(32, None);
    let u32t = b.type_int(32, 0);
    let v4 = b.type_vector(f32t, 4);
    let three = b.constant_bit32(u32t, 3);
    let per_vertex = b.type_struct([v4]);
    b.decorate(per_vertex, Decoration::Block, []);
    b.member_decorate(per_vertex, 0, Decoration::BuiltIn, [Operand::BuiltIn(BuiltIn::Position)]);
    b.member_decorate(per_vertex, 0, Decoration::Invariant, []);
    let arr_pv = b.type_array(per_vertex, three);
    let p_in_pv = b.type_pointer(None, StorageClass::Input, arr_pv);
    let gl_in = global(&mut b, p_in_pv, StorageClass::Input);
    // The output block has the same (deduplicated) type as gl_in's elements.
    let p_out_block = b.type_pointer(None, StorageClass::Output, per_vertex);
    let gl_out = global(&mut b, p_out_block, StorageClass::Output);
    let p_out_v4 = b.type_pointer(None, StorageClass::Output, v4);
    let arr_v4 = b.type_array(v4, three);
    let p_in_arr = b.type_pointer(None, StorageClass::Input, arr_v4);
    let p_in_v4 = b.type_pointer(None, StorageClass::Input, v4);
    let mut ins = Vec::new();
    let mut outs = Vec::new();
    let mut zeros = Vec::new();
    for i in 0..32 {
        if param_mask & (1 << i) == 0 {
            if out_mask & (1 << i) != 0 {
                let ov = global(&mut b, p_out_v4, StorageClass::Output);
                b.decorate(ov, Decoration::Location, [Operand::LiteralBit32(i)]);
                zeros.push(ov);
            }
            continue;
        }
        let iv = global(&mut b, p_in_arr, StorageClass::Input);
        b.decorate(iv, Decoration::Location, [Operand::LiteralBit32(i)]);
        let ov = global(&mut b, p_out_v4, StorageClass::Output);
        b.decorate(ov, Decoration::Location, [Operand::LiteralBit32(i)]);
        ins.push(iv);
        outs.push(ov);
    }
    let fn_ty = b.type_function(void, []);
    let func = b.begin_function(void, None, spirv::FunctionControl::NONE, fn_ty).unwrap();
    b.begin_block(None).unwrap();
    let k0 = b.constant_bit32(u32t, 0);
    let idx = [b.constant_bit32(u32t, 0), b.constant_bit32(u32t, 1), b.constant_bit32(u32t, 2)];
    let pos: Vec<u32> = (0..3)
        .map(|v| {
            let p = b.access_chain(p_in_v4, None, gl_in, [idx[v], k0]).unwrap();
            b.load(v4, None, p, None, []).unwrap()
        })
        .collect();
    let mut params: Vec<Vec<u32>> = Vec::new();
    for iv in &ins {
        let mut vs = Vec::new();
        for i in idx {
            let p = b.access_chain(p_in_v4, None, *iv, [i]).unwrap();
            vs.push(b.load(v4, None, p, None, []).unwrap());
        }
        params.push(vs);
    }
    let fourth = |b: &mut Builder, v: &[u32]| {
        let s = b.f_add(v4, None, v[1], v[2]).unwrap();
        b.f_sub(v4, None, s, v[0]).unwrap()
    };
    let pos3 = fourth(&mut b, &pos);
    let f0 = b.constant_bit32(f32t, 0);
    let zero4 = b.constant_composite(v4, [f0, f0, f0, f0]);
    let p3: Vec<u32> = params.iter().map(|vs| fourth(&mut b, vs)).collect();
    let out_pos = b.access_chain(p_out_v4, None, gl_out, [k0]).unwrap();
    for v in 0..4 {
        let pv = if v < 3 { pos[v] } else { pos3 };
        b.store(out_pos, pv, None, []).unwrap();
        for (k, o) in outs.iter().enumerate() {
            let val = if v < 3 { params[k][v] } else { p3[k] };
            b.store(*o, val, None, []).unwrap();
        }
        for o in &zeros {
            b.store(*o, zero4, None, []).unwrap();
        }
        b.emit_vertex().unwrap();
    }
    b.end_primitive().unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();
    let mut iface = vec![gl_in, gl_out];
    iface.extend(&ins);
    iface.extend(&outs);
    iface.extend(&zeros);
    b.entry_point(ExecutionModel::Geometry, func, "main", iface);
    b.execution_mode(func, ExecutionMode::Triangles, []);
    b.execution_mode(func, ExecutionMode::OutputTriangleStrip, []);
    b.execution_mode(func, ExecutionMode::OutputVertices, [4]);
    b.execution_mode(func, ExecutionMode::Invocations, [1]);
    b.module().assemble()
}

/// Where the barycentric geometry shader (`bary_gs`) puts what a pixel
/// stage reads per vertex: each per-vertex location keeps vertex 0's value
/// and names the locations of vertex 1's and 2's (all flat), and the
/// perspective-correct and linear barycentrics' locations.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct BaryIo {
    pub per_vertex: Vec<(u32, [u32; 2])>,
    pub bary: [u32; 2],
}

impl BaryIo {
    /// The locations this adds to the pixel stage's inputs (vertex 1's and
    /// 2's values, the barycentrics).
    pub fn added(&self) -> u32 {
        let mut m = (1u32 << self.bary[0]) | (1u32 << self.bary[1]);
        for (_, [a, b]) in &self.per_vertex {
            m |= (1u32 << a) | (1u32 << b);
        }
        m
    }
}

/// Generated geometry shader for devices without fragment barycentrics
/// (`Spec::bary_gs`): each triangle passes through with the vertex stage's
/// parameters at the pixel stage's other input locations (`ps_in_mask`;
/// zeros where the vertex stage writes none), each per-vertex location's
/// three values flat at the locations `io` names, and the barycentrics
/// (1, 0, 0), (0, 1, 0), (0, 0, 1) interpolated perspective-correct and
/// linear. `batch`: location `BATCH_VARYING` carries a batched draw's
/// unsigned SGPR data offset, passed through flat.
pub fn bary_gs(param_mask: u32, ps_in_mask: u32, io: &BaryIo, batch: bool) -> Vec<u32> {
    let mut b = Builder::new();
    b.set_version(1, 3);
    b.capability(spirv::Capability::Geometry);
    b.capability(spirv::Capability::Shader);
    b.memory_model(spirv::AddressingModel::Logical, spirv::MemoryModel::GLSL450);
    let void = b.type_void();
    let f32t = b.type_float(32, None);
    let u32t = b.type_int(32, 0);
    let v4 = b.type_vector(f32t, 4);
    let three = b.constant_bit32(u32t, 3);
    let per_vertex = b.type_struct([v4]);
    b.decorate(per_vertex, Decoration::Block, []);
    b.member_decorate(per_vertex, 0, Decoration::BuiltIn, [Operand::BuiltIn(BuiltIn::Position)]);
    b.member_decorate(per_vertex, 0, Decoration::Invariant, []);
    let arr_pv = b.type_array(per_vertex, three);
    let p_in_pv = b.type_pointer(None, StorageClass::Input, arr_pv);
    let gl_in = global(&mut b, p_in_pv, StorageClass::Input);
    let p_out_block = b.type_pointer(None, StorageClass::Output, per_vertex);
    let gl_out = global(&mut b, p_out_block, StorageClass::Output);
    let p_out_v4 = b.type_pointer(None, StorageClass::Output, v4);
    let arr_v4 = b.type_array(v4, three);
    let p_in_arr = b.type_pointer(None, StorageClass::Input, arr_v4);
    let p_in_v4 = b.type_pointer(None, StorageClass::Input, v4);
    let mut iface = vec![gl_in, gl_out];
    let location = |b: &mut Builder, v: u32, l: u32| b.decorate(v, Decoration::Location, [Operand::LiteralBit32(l)]);
    // The vertex stage's parameters this reads (per location; a batched
    // draw's varying apart).
    let added = io.added();
    let bases: u32 = io.per_vertex.iter().fold(0, |m, (l, _)| m | 1u32.checked_shl(*l).unwrap_or(0));
    let b31 = if batch { 1u32 << super::BATCH_VARYING } else { 0 };
    let wanted = (ps_in_mask & !added & !b31) | bases;
    let mut ins: Vec<(u32, u32)> = Vec::new();
    for l in 0..32 {
        if wanted & param_mask & (1 << l) != 0 {
            let iv = global(&mut b, p_in_arr, StorageClass::Input);
            location(&mut b, iv, l);
            iface.push(iv);
            ins.push((l, iv));
        }
    }
    // Outputs: (variable, location, source: Some((input, vertex)) for a
    // fixed vertex's value, None for the emitted vertex's own).
    let mut pass: Vec<(u32, Option<u32>)> = Vec::new();
    for l in 0..32u32 {
        if ps_in_mask & !added & !bases & !b31 & (1 << l) == 0 {
            continue;
        }
        let ov = global(&mut b, p_out_v4, StorageClass::Output);
        location(&mut b, ov, l);
        iface.push(ov);
        pass.push((ov, ins.iter().find(|(x, _)| *x == l).map(|(_, v)| *v)));
    }
    let mut flat: Vec<(u32, Option<u32>, u32)> = Vec::new();
    for (l, more) in &io.per_vertex {
        let src = ins.iter().find(|(x, _)| x == l).map(|(_, v)| *v);
        for (k, at) in [*l, more[0], more[1]].into_iter().enumerate() {
            let ov = global(&mut b, p_out_v4, StorageClass::Output);
            location(&mut b, ov, at);
            b.decorate(ov, Decoration::Flat, []);
            iface.push(ov);
            flat.push((ov, src, k as u32));
        }
    }
    let bary_p = global(&mut b, p_out_v4, StorageClass::Output);
    location(&mut b, bary_p, io.bary[0]);
    let bary_l = global(&mut b, p_out_v4, StorageClass::Output);
    location(&mut b, bary_l, io.bary[1]);
    b.decorate(bary_l, Decoration::NoPerspective, []);
    iface.push(bary_p);
    iface.push(bary_l);
    let batch_io = if batch {
        let arr_u = b.type_array(u32t, three);
        let p_in_arr_u = b.type_pointer(None, StorageClass::Input, arr_u);
        let iv = global(&mut b, p_in_arr_u, StorageClass::Input);
        location(&mut b, iv, super::BATCH_VARYING);
        let p_out_u = b.type_pointer(None, StorageClass::Output, u32t);
        let ov = global(&mut b, p_out_u, StorageClass::Output);
        location(&mut b, ov, super::BATCH_VARYING);
        b.decorate(ov, Decoration::Flat, []);
        iface.push(iv);
        iface.push(ov);
        Some((iv, ov))
    } else {
        None
    };
    let fn_ty = b.type_function(void, []);
    let func = b.begin_function(void, None, spirv::FunctionControl::NONE, fn_ty).unwrap();
    b.begin_block(None).unwrap();
    let k0 = b.constant_bit32(u32t, 0);
    let idx = [b.constant_bit32(u32t, 0), b.constant_bit32(u32t, 1), b.constant_bit32(u32t, 2)];
    let f0 = b.constant_bit32(f32t, 0);
    let f1 = b.constant_bit32(f32t, 1.0f32.to_bits());
    let zero4 = b.constant_composite(v4, [f0, f0, f0, f0]);
    let corners = [b.constant_composite(v4, [f1, f0, f0, f0]), b.constant_composite(v4, [f0, f1, f0, f0]), b.constant_composite(v4, [f0, f0, f1, f0])];
    let p_in_u = b.type_pointer(None, StorageClass::Input, u32t);
    // Every input loaded once per vertex.
    let pos: Vec<u32> = (0..3)
        .map(|v| {
            let p = b.access_chain(p_in_v4, None, gl_in, [idx[v], k0]).unwrap();
            b.load(v4, None, p, None, []).unwrap()
        })
        .collect();
    let mut vals: std::collections::HashMap<u32, [u32; 3]> = std::collections::HashMap::new();
    for (_, iv) in &ins {
        let mut vs = [0u32; 3];
        for (v, i) in idx.iter().enumerate() {
            let p = b.access_chain(p_in_v4, None, *iv, [*i]).unwrap();
            vs[v] = b.load(v4, None, p, None, []).unwrap();
        }
        vals.insert(*iv, vs);
    }
    let batch_vals: Option<[u32; 3]> = batch_io.map(|(iv, _)| {
        let mut vs = [0u32; 3];
        for (v, i) in idx.iter().enumerate() {
            let p = b.access_chain(p_in_u, None, iv, [*i]).unwrap();
            vs[v] = b.load(u32t, None, p, None, []).unwrap();
        }
        vs
    });
    let out_pos = b.access_chain(p_out_v4, None, gl_out, [k0]).unwrap();
    for v in 0..3 {
        b.store(out_pos, pos[v], None, []).unwrap();
        for (ov, src) in &pass {
            let val = src.map_or(zero4, |iv| vals[&iv][v]);
            b.store(*ov, val, None, []).unwrap();
        }
        for (ov, src, k) in &flat {
            let val = src.map_or(zero4, |iv| vals[&iv][*k as usize]);
            b.store(*ov, val, None, []).unwrap();
        }
        b.store(bary_p, corners[v], None, []).unwrap();
        b.store(bary_l, corners[v], None, []).unwrap();
        if let (Some((_, ov)), Some(bv)) = (batch_io, batch_vals) {
            b.store(ov, bv[v], None, []).unwrap();
        }
        b.emit_vertex().unwrap();
    }
    b.end_primitive().unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();
    b.entry_point(ExecutionModel::Geometry, func, "main", iface);
    b.execution_mode(func, ExecutionMode::Triangles, []);
    b.execution_mode(func, ExecutionMode::OutputTriangleStrip, []);
    b.execution_mode(func, ExecutionMode::OutputVertices, [3]);
    b.execution_mode(func, ExecutionMode::Invocations, [1]);
    b.module().assemble()
}

fn global(b: &mut Builder, ptr: u32, sc: StorageClass) -> u32 {
    let id = b.id();
    let inst = rspirv::dr::Instruction::new(spirv::Op::Variable, Some(ptr), Some(id), vec![Operand::StorageClass(sc)]);
    b.module_mut().types_global_values.push(inst);
    id
}

#[cfg(test)]
mod tests {
    #[test]
    fn rect_gs_assembles() {
        let w = super::rect_list_gs(0b1011, 0b1011);
        assert_eq!(w[0], 0x0723_0203);
        // A pixel input at location 0 the vertex stage does not write.
        let z = super::rect_list_gs(0b1010, 0b1011);
        assert_eq!(z[0], 0x0723_0203);
        assert_ne!(z, super::rect_list_gs(0b1010, 0b1010));
    }

    #[test]
    fn bary_gs_assembles() {
        // Location 1 read per vertex (vertices 1 and 2 at 4 and 5), 0 and 3
        // passed through (3 not written by the vertex stage), the
        // barycentrics at 6 and 7; with and without the batched varying.
        let io = super::BaryIo { per_vertex: vec![(1, [4, 5])], bary: [6, 7] };
        assert_eq!(io.added(), 0b1111_0000);
        let w = super::bary_gs(0b0011, 0b1111_1011, &io, false);
        assert_eq!(w[0], 0x0723_0203);
        let m = rspirv::dr::load_words(&w).expect("parses");
        let ep = &m.entry_points[0];
        // gl_in, gl_out, 2 inputs, 2 passed, 3 flat, 2 barycentrics.
        assert_eq!(ep.operands.len() - 3, 11);
        let wb = super::bary_gs(0b0011, 0b1111_1011 | 1 << super::super::BATCH_VARYING, &io, true);
        assert_ne!(w, wb);
        assert_eq!(rspirv::dr::load_words(&wb).expect("parses").entry_points[0].operands.len() - 3, 13);
        // `RDNA_GS_DUMP=<dir>`: both modules written there for spirv-val.
        if let Some(dir) = std::env::var_os("RDNA_GS_DUMP") {
            let dir = std::path::PathBuf::from(dir);
            for (name, words) in [("bary_gs.spv", &w), ("bary_gs_batch.spv", &wb)] {
                let bytes: Vec<u8> = words.iter().flat_map(|x| x.to_le_bytes()).collect();
                std::fs::write(dir.join(name), bytes).expect("written");
            }
        }
    }
}
