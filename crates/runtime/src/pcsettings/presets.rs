//! Quality presets and the first start's PC defaults. The Graphics list's Quality Preset sets
//! the render scale and the graphics settings together and shows Custom once one of them
//! differs from every preset. On the first start (no pcsettings.txt) the window opens in
//! borderless fullscreen and the preset is chosen from the GPU (its kind and video memory) once
//! the Vulkan device is known; both are written to pcsettings.txt as if chosen in the menu.

use std::sync::atomic::{AtomicBool, Ordering};

/// The Quality Preset choice's options: 0 Custom, then the presets in order.
pub const NAMES: [&str; 5] = ["Custom", "Low", "Medium", "High", "Ultra"];

/// Each preset's settings: (PC setting id, option index). High is the PS5's own quality (the
/// settings' defaults: the true resolution, the Visual Mode's tessellation, its texture
/// filtering); Ultra goes beyond it, Medium and Low trade detail for speed on modest GPUs (a
/// lower render scale first: the largest saving). Shadows stay: the title's only switches turn
/// them off altogether (the torches then light through walls).
const PRESETS: [&[(&str, u32)]; 4] = [
    // Low: 67% render scale, the title's PS4 Pro tessellation.
    &[("RenderScale", 2), ("GeometryDetail", 1), ("TextureFiltering", 0)],
    // Medium: 85%, the Performance mode's tessellation.
    &[("RenderScale", 4), ("GeometryDetail", 2), ("TextureFiltering", 0)],
    // High: as the PS5.
    &[("RenderScale", 5), ("GeometryDetail", 0), ("TextureFiltering", 0)],
    // Ultra: the finest tessellation, 16x anisotropic filtering.
    &[("RenderScale", 5), ("GeometryDetail", 5), ("TextureFiltering", 2)],
];

fn key(id: &str) -> String {
    format!("Setting_PC_{id}")
}

/// The preset the settings match (its option), 0 (Custom) when none.
pub fn current() -> u32 {
    PRESETS
        .iter()
        .position(|p| p.iter().all(|(id, v)| super::store::value(&key(id)) == Some(*v)))
        .map(|i| i as u32 + 1)
        .unwrap_or(0)
}

/// Sets a preset's settings (an option of the Quality Preset choice; Custom changes nothing).
pub fn apply(choice: u32) {
    let Some(p) = choice.checked_sub(1).and_then(|i| PRESETS.get(i as usize)) else { return };
    for (id, v) in p.iter() {
        super::store::set_value(&key(id), *v);
    }
    crate::info!("pc settings: Quality Preset {}", NAMES[choice as usize]);
    super::apply();
}

static FIRST_START: AtomicBool = AtomicBool::new(false);

/// No pcsettings.txt yet: the PC defaults of a first start (the window in borderless
/// fullscreen now; the preset once the GPU is known, `first_start_gpu`).
pub fn first_start() {
    FIRST_START.store(true, Ordering::Relaxed);
    super::store::set_value(&key("WindowMode"), 1);
    crate::info!("pc settings: first start: borderless fullscreen");
}

/// The preset a GPU gets on the first start: High (the PS5's quality) on discrete GPUs with
/// 10 GiB or more, Medium from 6 GiB, Low below or on integrated GPUs.
pub fn preset_for(discrete: bool, vram: u64) -> u32 {
    let gib = vram as f64 / (1u64 << 30) as f64;
    if !discrete || gib < 5.5 {
        1
    } else if gib < 9.5 {
        2
    } else {
        3
    }
}

/// The Vulkan device is known: on a first start, its preset.
pub fn first_start_gpu(name: &str, discrete: bool, vram: u64) {
    if !FIRST_START.swap(false, Ordering::Relaxed) {
        return;
    }
    let choice = preset_for(discrete, vram);
    crate::info!(
        "pc settings: first start on {name} ({:.1} GiB of video memory, {}): the {} preset",
        vram as f64 / (1u64 << 30) as f64,
        if discrete { "discrete" } else { "integrated" },
        NAMES[choice as usize]
    );
    apply(choice);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_by_gpu() {
        const GIB: u64 = 1 << 30;
        assert_eq!(preset_for(true, 24 * GIB), 3);
        assert_eq!(preset_for(true, 12 * GIB), 3);
        assert_eq!(preset_for(true, 8 * GIB), 2);
        assert_eq!(preset_for(true, 6 * GIB), 2);
        assert_eq!(preset_for(true, 4 * GIB), 1);
        assert_eq!(preset_for(false, 16 * GIB), 1);
    }

    #[test]
    fn presets_differ() {
        for (i, a) in PRESETS.iter().enumerate() {
            for b in &PRESETS[i + 1..] {
                assert_ne!(a, b);
            }
            // Every preset sets the same settings (a change of one shows Custom).
            assert_eq!(a.iter().map(|x| x.0).collect::<Vec<_>>(), PRESETS[0].iter().map(|x| x.0).collect::<Vec<_>>());
        }
    }
}
