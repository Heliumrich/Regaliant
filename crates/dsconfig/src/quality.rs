//! The Render Quality choice (the PC tab's and the launcher's Display page): the render size as
//! a share of the screen's (or the window's) resolution, named by the upscalers' quality modes.
//! The picture is always output at the screen's resolution: DLSS, FSR or XeSS (or the game's
//! TAA) take the render size up to it, so 100% is native anti-aliasing (DLAA with DLSS). Option
//! 0 keeps the game's own render size (its Visual Mode's).
//!
//! It replaced Render Scale (`Setting_PC_RenderScale`, 50% to 200%) on 2026-11-26: a saved
//! Render Scale is carried over to the nearest option (`from_render_scale`); the shares above
//! 100% are gone (an upscaler cannot take a larger render down to the screen).

/// The setting's id (`Setting_PC_RenderQuality`).
pub const ID: &str = "RenderQuality";
/// The setting it replaced (`Setting_PC_RenderScale`).
pub const OLD_ID: &str = "RenderScale";

/// The shares of the screen's resolution (each side) of options 1-6.
pub const PERCENTS: [u32; 6] = [33, 50, 58, 67, 77, 100];

/// The options' names, option 0 first.
pub const LABELS: [&str; 7] = ["Visual Mode", "Ultra Performance (33%)", "Performance (50%)", "Balanced (58%)", "Quality (67%)", "Ultra Quality (77%)", "DLAA / Native (100%)"];

/// The share in percent of option `v` (0: the game's own render size).
pub fn percent(v: u32) -> u32 {
    v.checked_sub(1).and_then(|i| PERCENTS.get(i as usize).copied()).unwrap_or(0)
}

/// The option nearest to a saved Render Scale option (0 Visual Mode; 1-9 50, 60, 67, 75, 85,
/// 100, 125, 150 and 200%): above 100% becomes 100%.
pub fn from_render_scale(old: u32) -> u32 {
    match old {
        0 => 0,
        1 => 2,
        2 => 3,
        3 => 4,
        4 | 5 => 5,
        _ => 6,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn options() {
        assert_eq!(LABELS.len(), PERCENTS.len() + 1);
        assert_eq!(percent(0), 0);
        assert_eq!(percent(2), 50);
        assert_eq!(percent(6), 100);
        assert_eq!(percent(7), 0);
        // Every old option maps to an option, the 100% and larger ones to 100%.
        for old in 0..10 {
            assert!((from_render_scale(old) as usize) < LABELS.len());
        }
        assert_eq!(percent(from_render_scale(1)), 50);
        assert_eq!(percent(from_render_scale(3)), 67);
        assert_eq!(percent(from_render_scale(9)), 100);
    }
}
