//! Button glyph textures in the layout of the title's own (`ui/textures/buttonicons/_ps5/*.ctxr`):
//! a 0x41-byte header (its image descriptor at 0x21..0x41: 128x128, R8G8B8A8_SRGB, 8 mips,
//! SW_64KB_S; the base address 0, set by the title's loader), 128 KiB of tiled pixels (the mip
//! tail in the first 64 KiB block, mip 0 in the second; premultiplied white glyphs) and a
//! 12-byte footer ("RTXT", a version word, the footer's own offset). The port's prompts
//! (`prompts`) are files of this layout with its own pixels, so the title loads and draws them
//! as its own.

use crate::gpu::tiling;

pub const SIZE: u32 = 128;
pub const MIPS: u32 = 8;
pub const HEADER: usize = 0x41;
/// Tiled bytes of a glyph (the title's files: 0x20000).
pub const DATA: usize = 0x20000;
pub const FOOTER: usize = 12;

fn layout() -> tiling::Layout {
    tiling::layout(9, 2, SIZE, SIZE, MIPS, false).expect("SW_64KB_S layout")
}

/// Mip 0 of a glyph file as packed RGBA rows (None: not of this layout).
pub fn decode(file: &[u8]) -> Option<Vec<u8>> {
    if file.len() != HEADER + DATA + FOOTER || &file[HEADER + DATA..HEADER + DATA + 4] != b"RTXT" {
        return None;
    }
    let l = layout();
    let mut out = Vec::new();
    l.detile(&file[HEADER..HEADER + DATA], 0, 0, &mut out);
    Some(out)
}

/// The mip chain of a SIZE x SIZE RGBA image (premultiplied alpha): each level the mean of its
/// four parents.
pub fn mips(rgba: &[u8]) -> Vec<Vec<u8>> {
    assert_eq!(rgba.len(), (SIZE * SIZE * 4) as usize);
    let mut chain = vec![rgba.to_vec()];
    let mut size = SIZE;
    while chain.len() < MIPS as usize {
        let src = chain.last().unwrap();
        let n = size / 2;
        let mut dst = vec![0u8; (n * n * 4) as usize];
        for y in 0..n {
            for x in 0..n {
                let mut c = [0u32; 4];
                for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    let i = (((2 * y + dy) * size + 2 * x + dx) * 4) as usize;
                    for k in 0..4 {
                        c[k] += src[i + k] as u32;
                    }
                }
                let o = ((y * n + x) * 4) as usize;
                for k in 0..4 {
                    dst[o + k] = ((c[k] + 2) / 4) as u8;
                }
            }
        }
        chain.push(dst);
        size = n;
    }
    chain
}

/// A glyph file: `template`'s header and footer (a title glyph's, kept as they are) around the
/// tiled mip chain.
pub fn file(template: &[u8], chain: &[Vec<u8>]) -> Vec<u8> {
    let l = layout();
    let mut out = template[..HEADER].to_vec();
    out.resize(HEADER + DATA, 0);
    out.extend_from_slice(&template[HEADER + DATA..HEADER + DATA + FOOTER]);
    let data = &mut out[HEADER..HEADER + DATA];
    for (mip, px) in chain.iter().enumerate() {
        let m = &l.mips[mip];
        let (w, h) = (m.width, m.height);
        for y in 0..h {
            for x in 0..w {
                let o = l.offset(mip, 0, x, y) as usize;
                let i = ((y * w + x) * 4) as usize;
                if o + 4 <= data.len() {
                    data[o..o + 4].copy_from_slice(&px[i..i + 4]);
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn game_glyph(name: &str) -> Option<Vec<u8>> {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let p = root.join("Game/Demon-Souls-Remake-PPSA01341-1.07/PPSA01341-app0/ui/textures/buttonicons/_ps5").join(name);
        std::fs::read(p).ok()
    }

    #[test]
    fn layout_matches_the_titles_files() {
        let l = layout();
        assert_eq!(l.slice_size as usize, DATA);
        // A title glyph, if the game tree is here: decoded and tiled again, the same bytes
        // (padding inside the tail block aside: compared where the layout places texels).
        let Some(f) = game_glyph("ps4_button_cross.ctxr") else { return };
        let mut chain = Vec::new();
        for mip in 0..MIPS as usize {
            let mut px = Vec::new();
            l.detile(&f[HEADER..HEADER + DATA], mip, 0, &mut px);
            chain.push(px);
        }
        let again = file(&f, &chain);
        assert_eq!(again.len(), f.len());
        assert_eq!(&again[..HEADER], &f[..HEADER]);
        assert_eq!(&again[HEADER + DATA..], &f[HEADER + DATA..]);
        for (mip, px) in chain.iter().enumerate() {
            let mut back = Vec::new();
            l.detile(&again[HEADER..HEADER + DATA], mip, 0, &mut back);
            assert_eq!(&back, px, "mip {mip}");
        }
        // The title's own mips are its mip 0 averaged (premultiplied): ours match them closely.
        let ours = mips(&chain[0]);
        let diff: u64 = ours[1].iter().zip(&chain[1]).map(|(a, b)| (*a as i32 - *b as i32).unsigned_abs() as u64).sum();
        assert!(diff / (ours[1].len() as u64) < 8, "mip 1 differs by {} per channel", diff / ours[1].len() as u64);
        if let Ok(dir) = std::env::var("DS_DUMP_DIR") {
            let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../Game/Demon-Souls-Remake-PPSA01341-1.07/PPSA01341-app0/ui/textures/buttonicons/_ps5");
            let names: Vec<String> = std::fs::read_dir(root).map(|d| d.filter_map(|e| e.ok()).map(|e| e.file_name().to_string_lossy().into_owned()).collect()).unwrap_or_default();
            for name in names.iter().map(|s| s.as_str()) {
                if let Some(g) = game_glyph(name) {
                    let px = decode(&g).unwrap();
                    let png = crate::gpu::png::encode(SIZE, SIZE, &px);
                    std::fs::write(std::path::Path::new(&dir).join(format!("{name}.png")), png).unwrap();
                }
            }
        }
    }

    #[test]
    fn mips_average_premultiplied() {
        let mut img = vec![0u8; (SIZE * SIZE * 4) as usize];
        for y in 0..SIZE / 2 {
            for x in 0..SIZE {
                let i = ((y * SIZE + x) * 4) as usize;
                img[i..i + 4].copy_from_slice(&[255, 255, 255, 255]);
            }
        }
        let c = mips(&img);
        assert_eq!(c.len(), MIPS as usize);
        assert_eq!(c[7], vec![128, 128, 128, 128]);
    }
}

