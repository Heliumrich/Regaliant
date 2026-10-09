//! Generated geometry shader that expands rect-list primitives: each input
//! triangle (v0, v1, v2) becomes the strip v0, v1, v2, v3 with
//! v3 = v1 + v2 - v0 (the parallelogram's fourth corner) for the position and
//! every parameter. Ported from the earlier native port's GCN translator
//! (MIT); parameters are passed for the locations in a mask.

use rspirv::binary::Assemble;
use rspirv::dr::{Builder, Operand};
use rspirv::spirv::{self, BuiltIn, Decoration, ExecutionMode, ExecutionModel, StorageClass};

/// Geometry shader passing through the vec4 parameters at the locations set
/// in `param_mask` (bit n = Location n).
pub fn rect_list_gs(param_mask: u32) -> Vec<u32> {
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
    for i in 0..32 {
        if param_mask & (1 << i) == 0 {
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
    let p3: Vec<u32> = params.iter().map(|vs| fourth(&mut b, vs)).collect();
    let out_pos = b.access_chain(p_out_v4, None, gl_out, [k0]).unwrap();
    for v in 0..4 {
        let pv = if v < 3 { pos[v] } else { pos3 };
        b.store(out_pos, pv, None, []).unwrap();
        for (k, o) in outs.iter().enumerate() {
            let val = if v < 3 { params[k][v] } else { p3[k] };
            b.store(*o, val, None, []).unwrap();
        }
        b.emit_vertex().unwrap();
    }
    b.end_primitive().unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();
    let mut iface = vec![gl_in, gl_out];
    iface.extend(&ins);
    iface.extend(&outs);
    b.entry_point(ExecutionModel::Geometry, func, "main", iface);
    b.execution_mode(func, ExecutionMode::Triangles, []);
    b.execution_mode(func, ExecutionMode::OutputTriangleStrip, []);
    b.execution_mode(func, ExecutionMode::OutputVertices, [4]);
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
        let w = super::rect_list_gs(0b1011);
        assert_eq!(w[0], 0x0723_0203);
    }
}
