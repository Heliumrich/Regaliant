//! gfx10 resource descriptors: buffer (V#), image (T#) and sampler (S#).
//!
//! Field layouts from Mesa `src/amd/registers/gfx10-rsrc.json` (MIT; see
//! docs/PROVENANCE.md and docs/RENDERER.md).

fn bits(v: u32, lo: u32, n: u32) -> u32 {
    (v >> lo) & ((1u32 << n) - 1)
}

/// Buffer descriptor (4 dwords).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Vsharp {
    pub base: u64,
    pub stride: u32,
    pub num_records: u32,
    pub dst_sel: u16,
    pub format: u8,
    pub add_tid: bool,
    /// 0 structured with offset, 1 structured, 2 raw, 3 no bounds check.
    pub oob_select: u8,
    pub raw: [u32; 4],
}

impl Vsharp {
    pub fn decode(w: &[u32]) -> Vsharp {
        Vsharp {
            base: w[0] as u64 | ((bits(w[1], 0, 16) as u64) << 32),
            stride: bits(w[1], 16, 14),
            num_records: w[2],
            dst_sel: bits(w[3], 0, 12) as u16,
            format: bits(w[3], 12, 7) as u8,
            add_tid: bits(w[3], 23, 1) != 0,
            oob_select: bits(w[3], 28, 2) as u8,
            raw: [w[0], w[1], w[2], w[3]],
        }
    }

    /// Bytes addressable through the descriptor (None: no bounds check).
    pub fn byte_size(&self) -> Option<u64> {
        if self.oob_select == 3 {
            return None;
        }
        Some(if self.stride == 0 { self.num_records as u64 } else { self.num_records as u64 * self.stride as u64 })
    }
}

/// Image descriptor (8 dwords).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Tsharp {
    pub base: u64,
    pub min_lod: u32,
    pub format: u16,
    pub width: u32,
    pub height: u32,
    pub dst_sel: u16,
    pub base_level: u32,
    pub last_level: u32,
    pub sw_mode: u32,
    /// 8 1D, 9 2D, 10 3D, 11 cube, 12 1D array, 13 2D array, 14 2D MSAA, 15 2D MSAA array.
    pub ty: u32,
    pub depth: u32,
    pub base_array: u32,
    /// Mip count of the resource - 1 (word5[7:4]).
    pub max_mip: u32,
    pub raw: [u32; 8],
}

impl Tsharp {
    pub fn decode(w: &[u32]) -> Tsharp {
        Tsharp {
            base: ((w[0] as u64) << 8) | ((bits(w[1], 0, 8) as u64) << 40),
            min_lod: bits(w[1], 8, 12),
            format: bits(w[1], 20, 9) as u16,
            width: (bits(w[1], 30, 2) | (bits(w[2], 0, 12) << 2)) + 1,
            height: bits(w[2], 14, 14) + 1,
            dst_sel: bits(w[3], 0, 12) as u16,
            base_level: bits(w[3], 12, 4),
            last_level: bits(w[3], 16, 4),
            sw_mode: bits(w[3], 20, 5),
            ty: bits(w[3], 28, 4),
            depth: bits(w[4], 0, 13) + 1,
            base_array: bits(w[4], 16, 13),
            max_mip: bits(w[5], 4, 4),
            raw: [w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn vsharp_from_capture() {
        // Fill kernel V# from the captured frame: 16-byte stride, 0x800
        // records, 32_32_32_32_UINT, identity swizzle.
        let v = Vsharp::decode(&[0xf6670000, 0x00100010, 0x00000800, 0x0004bfac]);
        assert_eq!(v.base, 0x10_f667_0000);
        assert_eq!(v.stride, 16);
        assert_eq!(v.format, 75);
        assert_eq!(v.dst_sel, 0xFAC);
        assert_eq!(v.byte_size(), Some(0x8000));
    }
}
