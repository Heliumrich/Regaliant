//! gfx10 unified buffer/image formats (the `FORMAT` field of V# and T#).
//!
//! Enumeration values from Mesa `src/amd/registers/gfx10-rsrc.json`
//! (GFX10_FORMAT, MIT). Names list components from the most significant
//! bits down (`10_11_11` = 11-bit x in the low bits, 10-bit z on top).

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Num {
    Unorm,
    Snorm,
    Uscaled,
    Sscaled,
    Uint,
    Sint,
    Float,
    Srgb,
}

impl Num {
    /// Register values are floats (as opposed to raw integers).
    pub fn is_float(self) -> bool {
        !matches!(self, Num::Uint | Num::Sint)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Format {
    pub id: u16,
    pub name: &'static str,
    /// Component bit widths from the least significant bits up (x, y, z, w);
    /// 0 for absent components. Empty for block-compressed and special formats.
    pub bits: [u8; 4],
    pub num: Num,
    /// Block-compressed (BCn) format.
    pub block: bool,
}

impl Format {
    pub fn components(&self) -> usize {
        self.bits.iter().filter(|&&b| b != 0).count()
    }
    /// Element size in bytes (block size for BCn).
    pub fn bytes(&self) -> u32 {
        if self.block {
            return if matches!(self.id, 169 | 170 | 175 | 176) { 8 } else { 16 };
        }
        self.bits.iter().map(|&b| b as u32).sum::<u32>().div_ceil(8)
    }
}

const NAMES: &[(u16, &str)] = &[
    (1, "8_UNORM"),
    (2, "8_SNORM"),
    (3, "8_USCALED"),
    (4, "8_SSCALED"),
    (5, "8_UINT"),
    (6, "8_SINT"),
    (7, "16_UNORM"),
    (8, "16_SNORM"),
    (9, "16_USCALED"),
    (10, "16_SSCALED"),
    (11, "16_UINT"),
    (12, "16_SINT"),
    (13, "16_FLOAT"),
    (14, "8_8_UNORM"),
    (15, "8_8_SNORM"),
    (16, "8_8_USCALED"),
    (17, "8_8_SSCALED"),
    (18, "8_8_UINT"),
    (19, "8_8_SINT"),
    (20, "32_UINT"),
    (21, "32_SINT"),
    (22, "32_FLOAT"),
    (23, "16_16_UNORM"),
    (24, "16_16_SNORM"),
    (25, "16_16_USCALED"),
    (26, "16_16_SSCALED"),
    (27, "16_16_UINT"),
    (28, "16_16_SINT"),
    (29, "16_16_FLOAT"),
    (30, "10_11_11_UNORM"),
    (31, "10_11_11_SNORM"),
    (32, "10_11_11_USCALED"),
    (33, "10_11_11_SSCALED"),
    (34, "10_11_11_UINT"),
    (35, "10_11_11_SINT"),
    (36, "10_11_11_FLOAT"),
    (37, "11_11_10_UNORM"),
    (38, "11_11_10_SNORM"),
    (39, "11_11_10_USCALED"),
    (40, "11_11_10_SSCALED"),
    (41, "11_11_10_UINT"),
    (42, "11_11_10_SINT"),
    (43, "11_11_10_FLOAT"),
    (44, "10_10_10_2_UNORM"),
    (45, "10_10_10_2_SNORM"),
    (46, "10_10_10_2_USCALED"),
    (47, "10_10_10_2_SSCALED"),
    (48, "10_10_10_2_UINT"),
    (49, "10_10_10_2_SINT"),
    (50, "2_10_10_10_UNORM"),
    (51, "2_10_10_10_SNORM"),
    (52, "2_10_10_10_USCALED"),
    (53, "2_10_10_10_SSCALED"),
    (54, "2_10_10_10_UINT"),
    (55, "2_10_10_10_SINT"),
    (56, "8_8_8_8_UNORM"),
    (57, "8_8_8_8_SNORM"),
    (58, "8_8_8_8_USCALED"),
    (59, "8_8_8_8_SSCALED"),
    (60, "8_8_8_8_UINT"),
    (61, "8_8_8_8_SINT"),
    (62, "32_32_UINT"),
    (63, "32_32_SINT"),
    (64, "32_32_FLOAT"),
    (65, "16_16_16_16_UNORM"),
    (66, "16_16_16_16_SNORM"),
    (67, "16_16_16_16_USCALED"),
    (68, "16_16_16_16_SSCALED"),
    (69, "16_16_16_16_UINT"),
    (70, "16_16_16_16_SINT"),
    (71, "16_16_16_16_FLOAT"),
    (72, "32_32_32_UINT"),
    (73, "32_32_32_SINT"),
    (74, "32_32_32_FLOAT"),
    (75, "32_32_32_32_UINT"),
    (76, "32_32_32_32_SINT"),
    (77, "32_32_32_32_FLOAT"),
    (128, "8_SRGB"),
    (129, "8_8_SRGB"),
    (130, "8_8_8_8_SRGB"),
    (131, "6E4_FLOAT"),
    (132, "5_9_9_9_FLOAT"),
    (133, "5_6_5_UNORM"),
    (134, "1_5_5_5_UNORM"),
    (135, "5_5_5_1_UNORM"),
    (136, "4_4_4_4_UNORM"),
    (137, "4_4_UNORM"),
    (138, "1_UNORM"),
    (139, "1_REVERSED_UNORM"),
    (140, "32_FLOAT_CLAMP"),
    (141, "8_24_UNORM"),
    (142, "8_24_UINT"),
    (143, "24_8_UNORM"),
    (144, "24_8_UINT"),
    (145, "X24_8_32_UINT"),
    (146, "X24_8_32_FLOAT"),
    (147, "GB_GR_UNORM"),
    (148, "GB_GR_SNORM"),
    (149, "GB_GR_UINT"),
    (150, "GB_GR_SRGB"),
    (151, "BG_RG_UNORM"),
    (152, "BG_RG_SNORM"),
    (153, "BG_RG_UINT"),
    (154, "BG_RG_SRGB"),
    (169, "BC1_UNORM"),
    (170, "BC1_SRGB"),
    (171, "BC2_UNORM"),
    (172, "BC2_SRGB"),
    (173, "BC3_UNORM"),
    (174, "BC3_SRGB"),
    (175, "BC4_UNORM"),
    (176, "BC4_SNORM"),
    (177, "BC5_UNORM"),
    (178, "BC5_SNORM"),
    (179, "BC6_UFLOAT"),
    (180, "BC6_SFLOAT"),
    (181, "BC7_UNORM"),
    (182, "BC7_SRGB"),
];

fn parse_num(s: &str) -> Option<Num> {
    Some(match s {
        "UNORM" => Num::Unorm,
        "SNORM" => Num::Snorm,
        "USCALED" => Num::Uscaled,
        "SSCALED" => Num::Sscaled,
        "UINT" => Num::Uint,
        "SINT" => Num::Sint,
        "FLOAT" | "UFLOAT" | "SFLOAT" => Num::Float,
        "SRGB" => Num::Srgb,
        _ => return None,
    })
}

/// Looks up a gfx10 FORMAT value.
pub fn gfx10(id: u16) -> Option<Format> {
    let name = NAMES.iter().find(|(i, _)| *i == id)?.1;
    let (layout, num) = name.rsplit_once('_')?;
    let num = parse_num(num)?;
    let mut bits = [0u8; 4];
    let block = layout.starts_with("BC");
    if !block {
        // Plain layouts: widths separated by '_', most significant first.
        let ws: Vec<u8> = layout.split('_').filter_map(|w| w.parse().ok()).collect();
        if ws.len() == layout.split('_').count() {
            for (k, w) in ws.iter().rev().enumerate().take(4) {
                bits[k] = *w;
            }
        }
    }
    Some(Format { id, name, bits, num, block })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn layouts() {
        let f = gfx10(36).unwrap();
        assert_eq!(f.bits, [11, 11, 10, 0]);
        assert_eq!(f.num, Num::Float);
        assert_eq!(gfx10(56).unwrap().bits, [8, 8, 8, 8]);
        assert_eq!(gfx10(50).unwrap().bits, [10, 10, 10, 2]);
        assert_eq!(gfx10(74).unwrap().bytes(), 12);
        assert!(gfx10(181).unwrap().block);
        assert_eq!(gfx10(22).unwrap().components(), 1);
    }
}
