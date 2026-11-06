//! Quality presets: the Graphics list's Quality Preset sets the render scale and the graphics
//! settings together and shows Custom once one of them differs from every preset. The defaults
//! are the original's (High); no preset is chosen for the player (the user, 2026-11-06 night: a
//! launcher may offer one later).

/// The Quality Preset choice's options: 0 Custom, then the presets in order.
pub const NAMES: [&str; 5] = ["Custom", "Low", "Medium", "High", "Ultra"];

/// Each preset's settings: (PC setting id, option index). High is the PS5's own quality (the
/// settings' defaults: the Visual Mode's render size, tessellation and texture filtering);
/// Ultra goes beyond it (the screen's own resolution, the finest tessellation, 16x filtering),
/// Medium and Low trade detail for speed on modest GPUs (a lower render scale first: the largest
/// saving). Shadows stay: the title's only switches turn them off altogether (the torches then
/// light through walls).
const PRESETS: [&[(&str, u32)]; 4] = [
    // Low: 67% render scale, the title's PS4 Pro tessellation.
    &[("RenderScale", 3), ("GeometryDetail", 1), ("TextureFiltering", 0)],
    // Medium: 85%, the Performance mode's tessellation.
    &[("RenderScale", 5), ("GeometryDetail", 2), ("TextureFiltering", 0)],
    // High: as the PS5.
    &[("RenderScale", 0), ("GeometryDetail", 0), ("TextureFiltering", 0)],
    // Ultra: 100% (the screen's resolution), the finest tessellation, 16x anisotropic filtering.
    &[("RenderScale", 6), ("GeometryDetail", 5), ("TextureFiltering", 2)],
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

#[cfg(test)]
mod tests {
    use super::*;

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
