//! gfx10 surface layouts: block dimensions, mip chains (mip tail first, mip
//! 0 last), and element addresses from the swizzle patterns.
//!
//! Follows Mesa addrlib (`src/amd/addrlib`, MIT; see docs/PROVENANCE.md):
//! `Gfx10Lib::ComputeSurfaceInfoMacroTiled`, `GetSwizzlePatternInfo`
//! (non-RB+ tables, as on gfx10.1 parts), `Lib::ComputeThinBlockDimension`,
//! `Lib::ComputeOffsetFromSwizzlePattern`. Thin (2D, array, cube) surfaces
//! only; 3D (thick) layouts are not implemented yet.

use super::swizzle_tables as t;

/// log2 of the GPU's pipe count (GB_ADDR_CONFIG.NUM_PIPES), used by the
/// XOR swizzle modes. Not known for the PS5 yet (`DS_GPU_PIPES_LOG2`).
pub fn pipes_log2() -> u32 {
    static P: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *P.get_or_init(|| std::env::var("DS_GPU_PIPES_LOG2").ok().and_then(|v| v.parse().ok()).unwrap_or(4).min(6))
}

/// Swizzle modes with an XOR of pipe/bank bits (pipe-count dependent).
pub fn is_xor(sw: u32) -> bool {
    (16..=27).contains(&sw)
}

fn block_log2(sw: u32) -> Option<u32> {
    Some(match sw {
        0 => 8,
        1..=3 => 8,
        4..=7 | 20..=23 => 12,
        8..=11 | 16..=19 | 24..=27 => 16,
        _ => return None,
    })
}

/// Pattern info table of a thin swizzle mode (non-RB+).
fn pattern_table(sw: u32) -> Option<&'static [[u16; 5]]> {
    Some(match sw {
        1 => t::GFX10_SW_256_S_PATINFO,
        2 => t::GFX10_SW_256_D_PATINFO,
        5 => t::GFX10_SW_4K_S_PATINFO,
        6 => t::GFX10_SW_4K_D_PATINFO,
        9 => t::GFX10_SW_64K_S_PATINFO,
        10 => t::GFX10_SW_64K_D_PATINFO,
        17 => t::GFX10_SW_64K_S_T_PATINFO,
        18 => t::GFX10_SW_64K_D_T_PATINFO,
        21 => t::GFX10_SW_4K_S_X_PATINFO,
        22 => t::GFX10_SW_4K_D_X_PATINFO,
        24 => t::GFX10_SW_64K_Z_X_1xaa_PATINFO,
        25 => t::GFX10_SW_64K_S_X_PATINFO,
        26 => t::GFX10_SW_64K_D_X_PATINFO,
        27 => t::GFX10_SW_64K_R_X_1xaa_PATINFO,
        _ => return None,
    })
}

/// Full 20-bit swizzle pattern (ADDR_BIT_SETTING per address bit).
pub fn pattern(sw: u32, elem_log2: u32) -> Option<[u64; 20]> {
    let table = pattern_table(sw)?;
    let index = if is_xor(sw) { pipes_log2() * 5 + elem_log2 } else { elem_log2 } as usize;
    let info = table.get(index)?;
    let mut p = [0u64; 20];
    p[..8].copy_from_slice(t::GFX10_SW_PATTERN_NIBBLE01.get(info[1] as usize)?);
    p[8..12].copy_from_slice(t::GFX10_SW_PATTERN_NIBBLE2.get(info[2] as usize)?);
    p[12..16].copy_from_slice(t::GFX10_SW_PATTERN_NIBBLE3.get(info[3] as usize)?);
    p[16..20].copy_from_slice(t::GFX10_SW_PATTERN_NIBBLE4.get(info[4] as usize)?);
    Some(p)
}

/// Offset contribution of coordinate `v` through the masks of field `field`
/// (0 x, 1 y, 2 z, 3 sample) for the first `bits` address bits.
fn field_offset(p: &[u64; 20], bits: u32, field: u32, v: u32) -> u32 {
    let mut off = 0;
    for (i, &b) in p.iter().take(bits as usize).enumerate() {
        let mask = ((b >> (16 * field)) & 0xFFFF) as u32;
        if (mask & v).count_ones() & 1 != 0 {
            off |= 1 << i;
        }
    }
    off
}

#[derive(Clone, Debug)]
pub struct MipLayout {
    /// Logical size in elements (texels, or 4x4 blocks): what holds data.
    pub width: u32,
    pub height: u32,
    /// Logical size in texels (max(1, size >> mip)).
    pub texel_w: u32,
    pub texel_h: u32,
    /// Row pitch and padded height in elements.
    pub pitch: u32,
    pub padded_height: u32,
    /// Byte offset of the mip within a slice.
    pub offset: u64,
    /// Element coordinates inside the mip-tail block (mips in the tail).
    pub tail: Option<(u32, u32)>,
}

#[derive(Clone, Debug)]
pub struct Layout {
    pub sw: u32,
    pub elem_log2: u32,
    pub block_w: u32,
    pub block_h: u32,
    pub block_log2: u32,
    /// Bytes per array slice (whole mip chain).
    pub slice_size: u64,
    pub mips: Vec<MipLayout>,
    /// Per-element offset tables within a block: offset = xs[x % bw] ^ ys[y % bh].
    xs: Vec<u32>,
    ys: Vec<u32>,
}

fn shift_ceil(v: u32, s: u32) -> u32 {
    (v + (1 << s) - 1) >> s
}

/// `width`/`height` in texels; `block`: 4x4-block compressed format (elements
/// are blocks). Padding follows addrlib (ceil-shifted element sizes); the
/// logical mip sizes are max(1, size >> mip) texels.
pub fn layout(sw: u32, elem_log2: u32, texel_w: u32, texel_h: u32, levels: u32, block: bool) -> Option<Layout> {
    let elems = |t: u32| if block { t.div_ceil(4) } else { t };
    let (width, height) = (elems(texel_w.max(1)), elems(texel_h.max(1)));
    let logical = |i: u32| {
        let (tw, th) = ((texel_w >> i).max(1), (texel_h >> i).max(1));
        (elems(tw), elems(th), tw, th)
    };
    let block_log2 = block_log2(sw)?;
    if sw == 0 {
        return None; // linear: see linear_layout
    }
    let p = pattern(sw, elem_log2)?;
    // Thin block dimensions (1 sample): width takes the extra bit.
    let log2_num = block_log2 - elem_log2;
    let log2_w = (log2_num + 1) / 2;
    let (bw, bh) = (1u32 << log2_w, 1u32 << (log2_num - log2_w));
    let block = 1u64 << block_log2;
    let levels = levels.max(1);
    let mut mips: Vec<MipLayout> = Vec::new();
    let slice_size;
    if levels > 1 {
        // Mip tail: blocks of half the width (thin), the last mips stored at
        // fixed offsets inside one block at the start of the slice.
        let (tail_w, tail_h) = (bw >> 1, bh);
        let max_in_tail = if block_log2 <= 11 { 1 + (1 << block_log2.saturating_sub(9)) } else { block_log2 - 4 };
        let mut first_tail = levels;
        let mut sizes = vec![0u64; levels as usize];
        let mut chain = 0u64;
        for i in 0..levels {
            let (w, h) = (shift_ceil(width.max(1), i), shift_ceil(height.max(1), i));
            if w <= tail_w && h <= tail_h && levels - i <= max_in_tail {
                first_tail = i;
                chain += block;
                break;
            }
            let pitch = w.next_multiple_of(bw);
            let ph = h.next_multiple_of(bh);
            sizes[i as usize] = pitch as u64 * ph as u64 * (1u64 << elem_log2);
            chain += sizes[i as usize];
            let (lw, lh, tw, th) = logical(i);
            mips.push(MipLayout { width: lw, height: lh, texel_w: tw, texel_h: th, pitch, padded_height: ph, offset: 0, tail: None });
        }
        slice_size = chain;
        // Offsets: the tail block first, then mips from the smallest up.
        let mut offset = if first_tail != levels { block } else { 0 };
        for i in (0..first_tail).rev() {
            mips[i as usize].offset = offset;
            offset += sizes[i as usize];
        }
        let b256: [(u32, u32); 5] = [(16, 16), (16, 8), (8, 8), (8, 4), (4, 4)];
        let (mut pitch, mut ph) = (tail_w, tail_h);
        for i in first_tail..levels {
            let m = max_in_tail - 1 - (i - first_tail);
            let mip_off = if m > 6 { 16u32 << m } else { m << 8 };
            let mut mx = ((mip_off >> 9) & 1) | ((mip_off >> 10) & 2) | ((mip_off >> 11) & 4) | ((mip_off >> 12) & 8) | ((mip_off >> 13) & 16) | ((mip_off >> 14) & 32);
            let mut my = ((mip_off >> 8) & 1) | ((mip_off >> 9) & 2) | ((mip_off >> 10) & 4) | ((mip_off >> 11) & 8) | ((mip_off >> 12) & 16) | ((mip_off >> 13) & 32);
            if block_log2 & 1 != 0 {
                std::mem::swap(&mut mx, &mut my);
                if elem_log2 & 1 != 0 {
                    my = (my << 1) | (mx & 1);
                    mx >>= 1;
                }
            }
            let (ew, eh) = b256[elem_log2.min(4) as usize];
            let (lw, lh, tw, th) = logical(i);
            mips.push(MipLayout { width: lw, height: lh, texel_w: tw, texel_h: th, pitch, padded_height: ph, offset: 0, tail: Some((mx * ew, my * eh)) });
            pitch = (pitch >> 1).max(ew);
            ph = (ph >> 1).max(eh);
        }
    } else {
        let pitch = width.max(1).next_multiple_of(bw);
        let ph = height.max(1).next_multiple_of(bh);
        slice_size = pitch as u64 * ph as u64 * (1u64 << elem_log2);
        mips.push(MipLayout { width, height, texel_w: texel_w.max(1), texel_h: texel_h.max(1), pitch, padded_height: ph, offset: 0, tail: None });
    }
    let xs = (0..bw).map(|x| field_offset(&p, block_log2, 0, x)).collect();
    let ys = (0..bh).map(|y| field_offset(&p, block_log2, 1, y)).collect();
    Some(Layout { sw, elem_log2, block_w: bw, block_h: bh, block_log2, slice_size, mips, xs, ys })
}

impl Layout {
    /// Byte offset of element (x, y) of `mip` in array slice `slice`.
    pub fn offset(&self, mip: usize, slice: u32, x: u32, y: u32) -> u64 {
        let m = &self.mips[mip];
        let base = slice as u64 * self.slice_size + m.offset;
        let (x, y) = match m.tail {
            Some((tx, ty)) => (x + tx, y + ty),
            None => (x, y),
        };
        let (bx, by) = (x / self.block_w, y / self.block_h);
        let pitch_blocks = (m.pitch / self.block_w).max(1) as u64;
        let block = (by as u64 * pitch_blocks + bx as u64) << self.block_log2;
        let inner = self.xs[(x % self.block_w) as usize] ^ self.ys[(y % self.block_h) as usize];
        base + block + inner as u64
    }

    /// Copies `mip` of `slice` from tiled `src` into tightly packed rows.
    pub fn detile(&self, src: &[u8], mip: usize, slice: u32, dst: &mut Vec<u8>) {
        let m = &self.mips[mip];
        let e = 1usize << self.elem_log2;
        dst.clear();
        dst.resize(m.width as usize * m.height as usize * e, 0);
        for y in 0..m.height {
            for x in 0..m.width {
                let o = self.offset(mip, slice, x, y) as usize;
                let d = (y as usize * m.width as usize + x as usize) * e;
                if o + e <= src.len() {
                    dst[d..d + e].copy_from_slice(&src[o..o + e]);
                }
            }
        }
    }

    /// Total bytes of `slices` array slices.
    pub fn size(&self, slices: u32) -> u64 {
        self.slice_size * slices.max(1) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_64k_block() {
        // 4-byte elements in 64 KiB blocks: 128 x 128.
        let l = layout(9, 2, 256, 256, 1, false).unwrap();
        assert_eq!((l.block_w, l.block_h), (128, 128));
        assert_eq!(l.slice_size, 256 * 256 * 4);
        // The element map within a block is a bijection onto element slots.
        let mut seen = vec![false; 128 * 128];
        for y in 0..128 {
            for x in 0..128 {
                let o = l.offset(0, 0, x, y) as usize;
                assert_eq!(o % 4, 0);
                assert!(!seen[o / 4]);
                seen[o / 4] = true;
            }
        }
    }

    #[test]
    fn mip_chain_ends_with_mip0() {
        let l = layout(9, 2, 1024, 1024, 11, false).unwrap();
        // mip 0 is the last (largest offset) of the chain.
        assert!(l.mips[0].offset > l.mips[1].offset);
        assert_eq!(l.mips[0].offset + 1024 * 1024 * 4, l.slice_size);
    }
}
