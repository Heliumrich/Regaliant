//! GCN surface tiling (detiling for uploads).
//!
//! Elements are pixels or 4x4 blocks for BC formats. Micro tiles are 8x8
//! elements. 2D (macro) tiling follows the AMD address-library algorithm with
//! explicit pipe/bank parameters (calibrated for the PS4 configuration).

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MicroType {
    Display,
    Thin,
    Depth,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArrayMode {
    Linear,
    Tiled1D,
    Tiled2D,
}

#[derive(Clone, Copy, Debug)]
pub struct TileMode {
    pub array: ArrayMode,
    pub micro: MicroType,
}

/// PS4 Gnm tile mode index -> layout.
pub fn tile_mode(index: u8) -> TileMode {
    use ArrayMode::*;
    use MicroType::*;
    let (array, micro) = match index {
        0..=4 | 6 | 7 => (Tiled2D, Depth),
        5 => (Tiled1D, Depth),
        8 | 31 => (Linear, Display),
        9 => (Tiled1D, Display),
        10 | 11 | 12 => (Tiled2D, Display),
        13 => (Tiled1D, Thin),
        14..=18 => (Tiled2D, Thin),
        19 => (Tiled1D, Thin),
        _ => (Tiled2D, Thin),
    };
    TileMode { array, micro }
}

#[inline]
fn bit(v: u32, b: u32) -> u32 {
    (v >> b) & 1
}

/// Index of element (x, y) within its 8x8 micro tile.
pub fn micro_index(x: u32, y: u32, bpp: u32, micro: MicroType) -> u32 {
    let (x0, x1, x2) = (bit(x, 0), bit(x, 1), bit(x, 2));
    let (y0, y1, y2) = (bit(y, 0), bit(y, 1), bit(y, 2));
    let b: [u32; 6] = match micro {
        MicroType::Display => match bpp {
            8 => [x0, x1, x2, y1, y0, y2],
            16 => [x0, x1, x2, y0, y1, y2],
            32 => [x0, x1, y0, x2, y1, y2],
            64 => [x0, y0, x1, x2, y1, y2],
            _ => [y0, x0, x1, x2, y1, y2],
        },
        _ => [x0, y0, x1, y1, x2, y2],
    };
    b[0] | (b[1] << 1) | (b[2] << 2) | (b[3] << 3) | (b[4] << 4) | (b[5] << 5)
}

#[derive(Clone, Copy, Debug)]
pub struct MacroParams {
    pub num_pipes: u32,
    pub num_banks: u32,
    pub bank_width: u32,
    pub bank_height: u32,
    pub macro_aspect: u32,
    pub pipe_interleave: u32,
}

/// Macro-tile parameters per element size for the PS4 (8 pipes, P8_32x32_16x16).
pub fn ps4_macro_params(bpp: u32) -> MacroParams {
    let (bw, bh, aspect, banks) = match bpp {
        8 => (1, 4, 4, 16),
        16 => (1, 2, 4, 16),
        32 => (1, 1, 2, 16),
        64 => (1, 1, 2, 8),
        _ => (1, 1, 1, 8),
    };
    MacroParams { num_pipes: 8, num_banks: banks, bank_width: bw, bank_height: bh, macro_aspect: aspect, pipe_interleave: 256 }
}

fn pipe_p8_32x32_16x16(x: u32, y: u32) -> u32 {
    let p0 = bit(x, 3) ^ bit(y, 3) ^ bit(x, 4);
    let p1 = bit(x, 4) ^ bit(y, 4);
    let p2 = bit(x, 5) ^ bit(y, 5);
    p0 | (p1 << 1) | (p2 << 2)
}

fn bank(x: u32, y: u32, p: &MacroParams) -> u32 {
    let tx = x / 8 / (p.bank_width * p.num_pipes);
    let ty = y / 8 / p.bank_height;
    let (x3, x4, x5, x6) = (bit(tx, 0), bit(tx, 1), bit(tx, 2), bit(tx, 3));
    let (y3, y4, y5, y6) = (bit(ty, 0), bit(ty, 1), bit(ty, 2), bit(ty, 3));
    match p.num_banks {
        16 => (x3 ^ y6) | ((x4 ^ y5 ^ y6) << 1) | ((x5 ^ y4) << 2) | ((x6 ^ y3) << 3),
        8 => (x3 ^ y5) | ((x4 ^ y4 ^ y5) << 1) | ((x5 ^ y3) << 2),
        4 => (x3 ^ y4) | ((x4 ^ y3) << 1),
        2 => x3 ^ y3,
        _ => 0,
    }
}

/// Byte offset of element (x, y) in slice 0 of a 2D-thin surface.
pub fn macro_offset(x: u32, y: u32, pitch: u32, height: u32, bpp: u32, micro: MicroType, p: &MacroParams) -> u64 {
    let micro_bytes = 64 * bpp / 8;
    let mt_pitch = 8 * p.bank_width * p.num_pipes * p.macro_aspect;
    let mt_height = 8 * p.bank_height * p.num_banks / p.macro_aspect;
    let mt_bytes = (mt_pitch * mt_height * bpp / 8) as u64;
    let mt_per_row = pitch.div_ceil(mt_pitch).max(1);
    let mt_x = x / mt_pitch;
    let mt_y = y / mt_height;
    let mt_offset = (mt_y as u64 * mt_per_row as u64 + mt_x as u64) * mt_bytes;
    let tile_row = (y / 8) % p.bank_height;
    let tile_col = ((x / 8) / p.num_pipes) % p.bank_width;
    let tile_offset = ((tile_row * p.bank_width + tile_col) * micro_bytes) as u64;
    let elem_offset = (micro_index(x, y, bpp, micro) * bpp / 8) as u64;
    let pipe_bits = p.num_pipes.trailing_zeros();
    let bank_bits = p.num_banks.trailing_zeros();
    let group_bits = p.pipe_interleave.trailing_zeros();
    let total = elem_offset + tile_offset + (mt_offset >> (pipe_bits + bank_bits));
    let pipe = pipe_p8_32x32_16x16(x, y) & (p.num_pipes - 1);
    let bk = bank(x, y, p);
    let group_mask = (1u64 << group_bits) - 1;
    let _ = height;
    ((total & !group_mask) << (pipe_bits + bank_bits)) | ((bk as u64) << (group_bits + pipe_bits)) | ((pipe as u64) << group_bits) | (total & group_mask)
}

/// Byte offset of element (x, y) in slice 0 of a 1D-thin surface.
pub fn micro_offset(x: u32, y: u32, pitch: u32, bpp: u32, micro: MicroType) -> u64 {
    let micro_bytes = (64 * bpp / 8) as u64;
    let tiles_per_row = (pitch / 8).max(1) as u64;
    let tile = (y / 8) as u64 * tiles_per_row + (x / 8) as u64;
    tile * micro_bytes + (micro_index(x, y, bpp, micro) * bpp / 8) as u64
}

/// Surface pitch and padded height in elements. `pitch` is the descriptor's
/// PITCH (already aligned by the title's Gnm library and used as-is by the
/// texture unit for addressing); only the height is padded to the tile
/// height, which determines the slice size.
pub fn padded(pitch: u32, height: u32, bpp: u32, mode: TileMode) -> (u32, u32) {
    match mode.array {
        ArrayMode::Linear => (pitch, height),
        ArrayMode::Tiled1D => (pitch, height.next_multiple_of(8)),
        ArrayMode::Tiled2D => {
            let p = ps4_macro_params(bpp);
            let mt_height = 8 * p.bank_height * p.num_banks / p.macro_aspect;
            (pitch, height.next_multiple_of(mt_height))
        }
    }
}

/// Byte offsets of the 64 elements of the 8x8 micro tile whose top-left
/// element is (x0, y0) (both multiples of 8). Within a micro tile the pipe
/// and bank bits are constant, so the macro-tiled address is computed once
/// per tile instead of once per element.
#[allow(clippy::too_many_arguments)]
fn micro_tile_offsets(x0: u32, y0: u32, pitch: u32, bpp: u32, mode: TileMode, p: &MacroParams, lut: &[u32; 64], out: &mut [u64; 64]) {
    let eb = (bpp / 8) as u64;
    if mode.array == ArrayMode::Tiled1D {
        let base = micro_offset(x0, y0, pitch, bpp, mode.micro);
        for (i, o) in out.iter_mut().enumerate() {
            *o = base + lut[i] as u64 * eb;
        }
        return;
    }
    let pipe_bits = p.num_pipes.trailing_zeros();
    let bank_bits = p.num_banks.trailing_zeros();
    let group_bits = p.pipe_interleave.trailing_zeros();
    let group_mask = (1u64 << group_bits) - 1;
    // Same terms as `macro_offset` for the tile's first element (micro index 0).
    let micro_bytes = 64 * bpp / 8;
    let mt_pitch = 8 * p.bank_width * p.num_pipes * p.macro_aspect;
    let mt_height = 8 * p.bank_height * p.num_banks / p.macro_aspect;
    let mt_bytes = (mt_pitch * mt_height * bpp / 8) as u64;
    let mt_per_row = pitch.div_ceil(mt_pitch).max(1);
    let mt_offset = ((y0 / mt_height) as u64 * mt_per_row as u64 + (x0 / mt_pitch) as u64) * mt_bytes;
    let tile_row = (y0 / 8) % p.bank_height;
    let tile_col = ((x0 / 8) / p.num_pipes) % p.bank_width;
    let tile_total = ((tile_row * p.bank_width + tile_col) * micro_bytes) as u64 + (mt_offset >> (pipe_bits + bank_bits));
    let pipe = (pipe_p8_32x32_16x16(x0, y0) & (p.num_pipes - 1)) as u64;
    let swz = ((bank(x0, y0, p) as u64) << (group_bits + pipe_bits)) | (pipe << group_bits);
    for (i, o) in out.iter_mut().enumerate() {
        let total = tile_total + lut[i] as u64 * eb;
        *o = ((total & !group_mask) << (pipe_bits + bank_bits)) | swz | (total & group_mask);
    }
}

/// Calls `f(tiled_offset, linear_offset)` for every element of a tiled
/// (1D or 2D) slice of `width` x `height` elements, a micro tile at a time.
fn for_each_tiled(width: u32, height: u32, pitch: u32, bpp: u32, mode: TileMode, mut f: impl FnMut(usize, usize)) {
    let eb = (bpp / 8) as usize;
    let p = ps4_macro_params(bpp);
    let mut lut = [0u32; 64];
    for (i, l) in lut.iter_mut().enumerate() {
        *l = micro_index(i as u32 % 8, i as u32 / 8, bpp, mode.micro);
    }
    let mut offs = [0u64; 64];
    for ty in (0..height).step_by(8) {
        for tx in (0..width).step_by(8) {
            micro_tile_offsets(tx, ty, pitch, bpp, mode, &p, &lut, &mut offs);
            for ly in 0..8u32.min(height - ty) {
                for lx in 0..8u32.min(width - tx) {
                    let l = ((ty + ly) as usize * width as usize + (tx + lx) as usize) * eb;
                    f(offs[(ly * 8 + lx) as usize] as usize, l);
                }
            }
        }
    }
}

/// Detiles one slice of `width` x `height` elements of `bpp` bits from `src`
/// (surface pitch/height in elements) into a tightly packed linear buffer.
#[allow(clippy::too_many_arguments)]
pub fn detile_slice(src: &[u8], dst: &mut [u8], width: u32, height: u32, pitch: u32, pheight: u32, bpp: u32, mode: TileMode) {
    let eb = (bpp / 8) as usize;
    match mode.array {
        ArrayMode::Linear => {
            for y in 0..height as usize {
                let s = y * pitch as usize * eb;
                let d = y * width as usize * eb;
                let n = width as usize * eb;
                if s + n <= src.len() && d + n <= dst.len() {
                    dst[d..d + n].copy_from_slice(&src[s..s + n]);
                }
            }
        }
        ArrayMode::Tiled1D | ArrayMode::Tiled2D => {
            let _ = pheight;
            for_each_tiled(width, height, pitch, bpp, mode, |t, l| {
                if t + eb <= src.len() && l + eb <= dst.len() {
                    dst[l..l + eb].copy_from_slice(&src[t..t + eb]);
                }
            });
        }
    }
}

/// Tiles a tightly packed linear slice of `width` x `height` elements into
/// `dst` (surface pitch/height in elements); inverse of `detile_slice`.
#[allow(clippy::too_many_arguments)]
pub fn tile_slice(src: &[u8], dst: &mut [u8], width: u32, height: u32, pitch: u32, pheight: u32, bpp: u32, mode: TileMode) {
    let eb = (bpp / 8) as usize;
    let _ = pheight;
    if mode.array == ArrayMode::Linear {
        for y in 0..height as usize {
            let d = y * pitch as usize * eb;
            let s = y * width as usize * eb;
            let n = width as usize * eb;
            if d + n <= dst.len() && s + n <= src.len() {
                dst[d..d + n].copy_from_slice(&src[s..s + n]);
            }
        }
        return;
    }
    for_each_tiled(width, height, pitch, bpp, mode, |t, l| {
        if t + eb <= dst.len() && l + eb <= src.len() {
            dst[t..t + eb].copy_from_slice(&src[l..l + eb]);
        }
    });
}

/// Size in bytes of one slice of a surface with the given padded dimensions.
pub fn slice_bytes(pitch: u32, pheight: u32, bpp: u32) -> u64 {
    pitch as u64 * pheight as u64 * (bpp / 8) as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn micro_index_is_a_permutation() {
        for &bpp in &[8, 16, 32, 64, 128] {
            for m in [MicroType::Display, MicroType::Thin] {
                let mut seen = [false; 64];
                for y in 0..8 {
                    for x in 0..8 {
                        let i = micro_index(x, y, bpp, m) as usize;
                        assert!(!seen[i]);
                        seen[i] = true;
                    }
                }
            }
        }
    }
    #[test]
    fn fast_offsets_match_reference() {
        for (idx, bpp) in [(14u8, 32u32), (14, 64), (14, 8), (14, 128), (0, 32), (10, 32), (13, 32), (5, 16)] {
            let mode = tile_mode(idx);
            let p = ps4_macro_params(bpp);
            let mut lut = [0u32; 64];
            for (i, l) in lut.iter_mut().enumerate() {
                *l = micro_index(i as u32 % 8, i as u32 / 8, bpp, mode.micro);
            }
            let (pitch, ph) = padded(256, 128, bpp, mode);
            let mut offs = [0u64; 64];
            for ty in (0..ph).step_by(8) {
                for tx in (0..pitch).step_by(8) {
                    micro_tile_offsets(tx, ty, pitch, bpp, mode, &p, &lut, &mut offs);
                    for i in 0..64u32 {
                        let (x, y) = (tx + i % 8, ty + i / 8);
                        let r = match mode.array {
                            ArrayMode::Tiled1D => micro_offset(x, y, pitch, bpp, mode.micro),
                            _ => macro_offset(x, y, pitch, ph, bpp, mode.micro, &p),
                        };
                        assert_eq!(offs[i as usize], r, "mode {idx} bpp {bpp} at {x},{y}");
                    }
                }
            }
        }
    }
    #[test]
    fn tile_round_trip() {
        for idx in [0u8, 5, 8, 13, 14] {
            let mode = tile_mode(idx);
            let (w, h) = (96, 40);
            let (pitch, ph) = padded(128, h, 32, mode);
            let lin: Vec<u8> = (0..w * h * 4).map(|i| (i * 7 + idx as u32) as u8).collect();
            let mut tiled = vec![0u8; slice_bytes(pitch, ph, 32) as usize];
            tile_slice(&lin, &mut tiled, w, h, pitch, ph, 32, mode);
            let mut back = vec![0u8; lin.len()];
            detile_slice(&tiled, &mut back, w, h, pitch, ph, 32, mode);
            assert_eq!(lin, back, "tile mode {idx}");
        }
    }
    #[test]
    fn macro_offsets_are_unique() {
        let (w, h) = padded(512, 256, 32, tile_mode(14));
        let p = ps4_macro_params(32);
        let mut seen = std::collections::HashSet::new();
        for y in 0..h {
            for x in 0..w {
                let o = macro_offset(x, y, w, h, 32, MicroType::Thin, &p);
                assert!(o < (w * h * 4) as u64, "offset {o} out of range at {x},{y}");
                assert!(seen.insert(o), "duplicate offset at {x},{y}");
            }
        }
    }
}
