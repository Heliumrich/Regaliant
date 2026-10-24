//! gfx10 surface layouts: block dimensions, mip chains (mip tail first, mip
//! 0 last), and element addresses from the swizzle patterns.
//!
//! Follows Mesa addrlib (`src/amd/addrlib`, MIT; see docs/PROVENANCE.md):
//! `Gfx10Lib::ComputeSurfaceInfoMacroTiled`, `GetSwizzlePatternInfo`
//! (non-RB+ tables, as on gfx10.1 parts), `Lib::ComputeThinBlockDimension`,
//! `Lib::ComputeOffsetFromSwizzlePattern`. Thin (2D, array, cube) surfaces,
//! and thick 3D surfaces with standard swizzles and one mip
//! (`Lib::ComputeThickBlockDimension`, `ComputeSurfaceAddrFromCoordMacroTiled`).

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

/// Pattern info table of a 3D surface with a standard (thick) swizzle
/// (`Gfx10Lib::GetSwizzlePatternInfo`, non-RB+).
fn pattern_table_3d(sw: u32) -> Option<&'static [[u16; 5]]> {
    Some(match sw {
        5 => t::GFX10_SW_4K_S3_PATINFO,
        9 => t::GFX10_SW_64K_S3_PATINFO,
        17 => t::GFX10_SW_64K_S3_T_PATINFO,
        21 => t::GFX10_SW_4K_S3_X_PATINFO,
        25 => t::GFX10_SW_64K_S3_X_PATINFO,
        _ => return None,
    })
}

/// Whether a 3D surface with swizzle mode `sw` is thick (Z and standard
/// swizzles; display and render-target swizzles keep 2D slices).
pub fn is_thick_3d(sw: u32) -> bool {
    matches!(sw, 4 | 5 | 8 | 9 | 16 | 17 | 20 | 21 | 24 | 25)
}

/// Thick 3D layout of one mip: blocks of bw x bh x bd elements; the surface
/// is a stack of slabs of bd slices.
#[derive(Clone, Debug)]
pub struct Layout3d {
    pub elem_log2: u32,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub block_w: u32,
    pub block_h: u32,
    pub block_d: u32,
    pub block_log2: u32,
    pub pitch: u32,
    pub padded_height: u32,
    pub padded_depth: u32,
    p: [u64; 20],
    /// Per-axis offset tables over the whole (padded) extent: the in-block
    /// offset is xs[x] ^ ys[y] ^ zs[z] (XOR swizzles take coordinate bits
    /// above the block's).
    xs: Vec<u32>,
    ys: Vec<u32>,
    zs: Vec<u32>,
}

/// Single-mip thick 3D layout (standard swizzles), or None.
pub fn layout_3d(sw: u32, elem_log2: u32, width: u32, height: u32, depth: u32) -> Option<Layout3d> {
    let block_log2 = block_log2(sw)?;
    let table = pattern_table_3d(sw)?;
    let index = if is_xor(sw) { pipes_log2() * 5 + elem_log2 } else { elem_log2 } as usize;
    let info = table.get(index)?;
    let mut p = [0u64; 20];
    p[..8].copy_from_slice(t::GFX10_SW_PATTERN_NIBBLE01.get(info[1] as usize)?);
    p[8..12].copy_from_slice(t::GFX10_SW_PATTERN_NIBBLE2.get(info[2] as usize)?);
    p[12..16].copy_from_slice(t::GFX10_SW_PATTERN_NIBBLE3.get(info[3] as usize)?);
    p[16..20].copy_from_slice(t::GFX10_SW_PATTERN_NIBBLE4.get(info[4] as usize)?);
    // Lib::ComputeThickBlockDimension: 1 KiB blocks per element size, grown
    // evenly over the three dimensions (depth first, then height).
    const BLOCK_1K_3D: [(u32, u32, u32); 5] = [(16, 8, 8), (8, 8, 8), (8, 8, 4), (8, 4, 4), (4, 4, 4)];
    let (w0, h0, d0) = *BLOCK_1K_3D.get(elem_log2 as usize)?;
    let in_1k = block_log2.checked_sub(10)?;
    let (avg, rest) = (in_1k / 3, in_1k % 3);
    let (bw, bh, bd) = (w0 << avg, h0 << (avg + rest / 2), d0 << (avg + u32::from(rest != 0)));
    let (width, height, depth) = (width.max(1), height.max(1), depth.max(1));
    let xs = (0..width.next_multiple_of(bw)).map(|x| field_offset(&p, block_log2, 0, x)).collect();
    let ys = (0..height.next_multiple_of(bh)).map(|y| field_offset(&p, block_log2, 1, y)).collect();
    let zs = (0..depth.next_multiple_of(bd)).map(|z| field_offset(&p, block_log2, 2, z)).collect();
    Some(Layout3d {
        elem_log2,
        width,
        height,
        depth,
        block_w: bw,
        block_h: bh,
        block_d: bd,
        block_log2,
        pitch: width.next_multiple_of(bw),
        padded_height: height.next_multiple_of(bh),
        padded_depth: depth.next_multiple_of(bd),
        p,
        xs,
        ys,
        zs,
    })
}

impl Layout3d {
    /// Bytes of one slab of block_d slices.
    fn slab_size(&self) -> u64 {
        self.pitch as u64 * self.padded_height as u64 * self.block_d as u64 * (1u64 << self.elem_log2)
    }

    /// Total bytes.
    pub fn size(&self) -> u64 {
        self.slab_size() * (self.padded_depth / self.block_d) as u64
    }

    /// Byte offset of element (x, y, z).
    pub fn offset(&self, x: u32, y: u32, z: u32) -> u64 {
        let pb = (self.pitch / self.block_w) as u64;
        let blk = (y / self.block_h) as u64 * pb + (x / self.block_w) as u64;
        let inner = field_offset(&self.p, self.block_log2, 0, x) ^ field_offset(&self.p, self.block_log2, 1, y) ^ field_offset(&self.p, self.block_log2, 2, z);
        (z / self.block_d) as u64 * self.slab_size() + (blk << self.block_log2) + inner as u64
    }

    /// The whole volume as tightly packed rows and slices.
    pub fn detile(&self, src: &[u8], dst: &mut Vec<u8>) {
        let e = 1usize << self.elem_log2;
        dst.clear();
        dst.resize(self.width as usize * self.height as usize * self.depth as usize * e, 0);
        self.detile_into(src, dst);
    }

    /// `detile` into `dst`, exactly the volume's packed size (the texture
    /// cache's staging memory), from per-axis offset tables and on several
    /// threads (slices) for volumes of 1 MiB and more; elements beyond `src`
    /// are written as zeros. (Per element through the swizzle pattern it
    /// took 100-230 ms for the title's 8-180 MB volumes, on the recording
    /// threads, 2026-10-24.)
    pub fn detile_into(&self, src: &[u8], dst: &mut [u8]) {
        let e = 1usize << self.elem_log2;
        let slice = self.width as usize * self.height as usize * e;
        debug_assert_eq!(dst.len(), slice * self.depth as usize);
        if slice == 0 {
            return;
        }
        let threads = if dst.len() >= 1 << 20 { std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(8) } else { 1 };
        let per = (self.depth as usize).div_ceil(threads.max(1));
        if threads <= 1 || per == 0 {
            self.slices(src, 0, dst);
            return;
        }
        std::thread::scope(|s| {
            for (k, chunk) in dst.chunks_mut(per * slice).enumerate() {
                s.spawn(move || self.slices(src, k * per, chunk));
            }
        });
    }

    /// Detiles whole slices starting at slice `z0` into `out`.
    fn slices(&self, src: &[u8], z0: usize, out: &mut [u8]) {
        match self.elem_log2 {
            0 => self.slices_n::<1>(src, z0, out),
            1 => self.slices_n::<2>(src, z0, out),
            2 => self.slices_n::<4>(src, z0, out),
            3 => self.slices_n::<8>(src, z0, out),
            _ => self.slices_n::<16>(src, z0, out),
        }
    }

    fn slices_n<const N: usize>(&self, src: &[u8], z0: usize, out: &mut [u8]) {
        let (w, h) = (self.width as usize, self.height as usize);
        let (lw, lh, ld) = (self.block_w.trailing_zeros(), self.block_h.trailing_zeros(), self.block_d.trailing_zeros());
        let pitch_blocks = (self.pitch / self.block_w).max(1) as u64;
        let slab = self.slab_size();
        for (r, orow) in out.chunks_exact_mut(w * N).enumerate() {
            let z = z0 + r / h;
            let y = r % h;
            let zy = self.zs[z] ^ self.ys[y];
            let rb = (z >> ld) as u64 * slab + (((y >> lh) as u64 * pitch_blocks) << self.block_log2);
            for x in 0..w {
                let o = (rb + (((x >> lw) as u64) << self.block_log2) + (self.xs[x] ^ zy) as u64) as usize;
                if o + N <= src.len() {
                    orow[x * N..x * N + N].copy_from_slice(&src[o..o + N]);
                } else {
                    orow[x * N..x * N + N].fill(0);
                }
            }
        }
    }
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

    /// Copies `mip` of `slice` from tiled `src` into tightly packed rows
    /// (row ranges on several threads for mips of 1 MiB and more).
    pub fn detile(&self, src: &[u8], mip: usize, slice: u32, dst: &mut Vec<u8>) {
        let m = &self.mips[mip];
        let e = 1usize << self.elem_log2;
        dst.clear();
        dst.resize(m.width as usize * m.height as usize * e, 0);
        self.detile_into(src, mip, slice, dst);
    }

    /// `detile` into `dst`, exactly the mip's packed size (the texture
    /// cache's staging memory: no intermediate copies). Elements beyond
    /// `src` are written as zeros.
    pub fn detile_into(&self, src: &[u8], mip: usize, slice: u32, dst: &mut [u8]) {
        let m = &self.mips[mip];
        let e = 1usize << self.elem_log2;
        let (w, h) = (m.width as usize, m.height as usize);
        debug_assert_eq!(dst.len(), w * h * e);
        if w == 0 || h == 0 {
            return;
        }
        let threads = if dst.len() >= 1 << 20 { std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(8) } else { 1 };
        if threads <= 1 {
            self.detile_rows(src, m, slice, 0, dst);
            return;
        }
        let rows = h.div_ceil(threads);
        std::thread::scope(|s| {
            for (k, chunk) in dst.chunks_mut(rows * w * e).enumerate() {
                s.spawn(move || self.detile_rows(src, m, slice, k * rows, chunk));
            }
        });
    }

    /// Detiles whole rows starting at row `y0` into `out`.
    fn detile_rows(&self, src: &[u8], m: &MipLayout, slice: u32, y0: usize, out: &mut [u8]) {
        match self.elem_log2 {
            0 => self.rows_n::<1>(src, m, slice, y0, out),
            1 => self.rows_n::<2>(src, m, slice, y0, out),
            2 => self.rows_n::<4>(src, m, slice, y0, out),
            3 => self.rows_n::<8>(src, m, slice, y0, out),
            _ => self.rows_n::<16>(src, m, slice, y0, out),
        }
    }

    /// `offset` per element without divisions (block sizes are powers of
    /// two) and fixed-size element copies.
    fn rows_n<const N: usize>(&self, src: &[u8], m: &MipLayout, slice: u32, y0: usize, out: &mut [u8]) {
        let w = m.width as usize;
        let base = slice as u64 * self.slice_size + m.offset;
        let (tx, ty) = m.tail.map(|(a, b)| (a as usize, b as usize)).unwrap_or((0, 0));
        let (lw, lh) = (self.block_w.trailing_zeros(), self.block_h.trailing_zeros());
        let (mw, mh) = (self.block_w as usize - 1, self.block_h as usize - 1);
        let pitch_blocks = (m.pitch / self.block_w).max(1) as u64;
        for (r, orow) in out.chunks_exact_mut(w * N).enumerate() {
            let gy = y0 + r + ty;
            let ysv = self.ys[gy & mh];
            let rb = base + (((gy >> lh) as u64 * pitch_blocks) << self.block_log2);
            for x in 0..w {
                let gx = x + tx;
                let o = (rb + (((gx >> lw) as u64) << self.block_log2) + (self.xs[gx & mw] ^ ysv) as u64) as usize;
                if o + N <= src.len() {
                    orow[x * N..x * N + N].copy_from_slice(&src[o..o + N]);
                } else {
                    orow[x * N..x * N + N].fill(0);
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
    fn fast_detile_matches_offsets() {
        // Every swizzle mode, element size and mip (tails included), and an
        // array slice: the row detiler against the per-element offsets.
        for sw in [1u32, 2, 5, 6, 9, 10, 17, 18, 21, 22, 24, 25, 26, 27] {
            for el in 0..5u32 {
                for (tw, th, levels) in [(300u32, 200u32, 9u32), (1024, 512, 11), (64, 64, 7), (2048, 1024, 1)] {
                    let Some(l) = layout(sw, el, tw, th, levels, false) else { continue };
                    let size = l.size(2) as usize;
                    let src: Vec<u8> = (0..size).map(|i| (i * 2654435761usize >> 7) as u8).collect();
                    for mip in 0..l.mips.len() {
                        for slice in 0..2 {
                            let m = &l.mips[mip];
                            let e = 1usize << el;
                            let mut want = vec![0u8; m.width as usize * m.height as usize * e];
                            for y in 0..m.height {
                                for x in 0..m.width {
                                    let o = l.offset(mip, slice, x, y) as usize;
                                    let d = (y as usize * m.width as usize + x as usize) * e;
                                    if o + e <= src.len() {
                                        want[d..d + e].copy_from_slice(&src[o..o + e]);
                                    }
                                }
                            }
                            let mut got = Vec::new();
                            l.detile(&src, mip, slice, &mut got);
                            assert!(got == want, "sw {sw} elem {el} {tw}x{th} mip {mip} slice {slice}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn fast_detile_3d_matches_offsets() {
        // Thick swizzle modes, element sizes and odd sizes: the table-driven
        // slice detiler against the per-element offsets.
        for sw in [5u32, 9, 17, 21, 25] {
            for el in 0..5u32 {
                for (w, h, d) in [(64u32, 32u32, 16u32), (100, 70, 9), (256, 64, 40), (33, 17, 5)] {
                    let Some(l) = layout_3d(sw, el, w, h, d) else { continue };
                    let size = l.size() as usize;
                    let src: Vec<u8> = (0..size).map(|i| (i * 2654435761usize >> 9) as u8).collect();
                    let e = 1usize << el;
                    let mut want = vec![0u8; w as usize * h as usize * d as usize * e];
                    let mut k = 0;
                    for z in 0..d {
                        for y in 0..h {
                            for x in 0..w {
                                let o = l.offset(x, y, z) as usize;
                                if o + e <= src.len() {
                                    want[k..k + e].copy_from_slice(&src[o..o + e]);
                                }
                                k += e;
                            }
                        }
                    }
                    let mut got = Vec::new();
                    l.detile(&src, &mut got);
                    assert!(got == want, "sw {sw} elem {el} {w}x{h}x{d}");
                }
            }
        }
    }

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
    fn thick_64k_standard() {
        // RGBA8 in 64 KiB thick blocks: 32 x 32 x 16 (addrlib Block1K_3d
        // (8, 8, 4) times 4 in each dimension); a bijection within a block.
        let l = layout_3d(9, 2, 128, 128, 128).unwrap();
        assert_eq!((l.block_w, l.block_h, l.block_d), (32, 32, 16));
        assert_eq!(l.size(), 128 * 128 * 128 * 4);
        let mut seen = vec![false; 32 * 32 * 16];
        for z in 0..16 {
            for y in 0..32 {
                for x in 0..32 {
                    let o = l.offset(x, y, z) as usize;
                    assert_eq!(o % 4, 0);
                    assert!(o < 65536 && !seen[o / 4]);
                    seen[o / 4] = true;
                }
            }
        }
        // RGBA16F: 32 x 16 x 16; 4 KiB blocks of RGBA8: 8 x 16 x 8.
        let l = layout_3d(9, 3, 214, 120, 72).unwrap();
        assert_eq!((l.block_w, l.block_h, l.block_d), (32, 16, 16));
        let l = layout_3d(5, 2, 64, 64, 64).unwrap();
        assert_eq!((l.block_w, l.block_h, l.block_d), (8, 16, 8));
    }

    #[test]
    fn mip_chain_ends_with_mip0() {
        let l = layout(9, 2, 1024, 1024, 11, false).unwrap();
        // mip 0 is the last (largest offset) of the chain.
        assert!(l.mips[0].offset > l.mips[1].offset);
        assert_eq!(l.mips[0].offset + 1024 * 1024 * 4, l.slice_size);
    }
}
