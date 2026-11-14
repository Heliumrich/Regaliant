// Embeds the launcher's manifest (Common Controls 6 for the system's control styles, per-monitor
// DPI awareness, long paths) and the program icon through the MSVC linker; no resource compiler
// needed.
use std::path::Path;

include!("../icon_res.rs");

fn main() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let manifest = dir.join("app.manifest");
    println!("cargo:rerun-if-changed={}", manifest.display());
    if std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc") {
        println!("cargo:rustc-link-arg-bins=/MANIFEST:EMBED");
        println!("cargo:rustc-link-arg-bins=/MANIFESTINPUT:{}", manifest.display());
    }
    link_icon(&dir.join("../../assets/regaliant.ico"));
}
