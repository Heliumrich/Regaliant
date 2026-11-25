// The program icon (`link_icon`), shared with the launcher's build script.
include!("../icon_res.rs");

fn main() {
    // A profile-instrumented build for PGO passes `--cfg dsport_pgo_gen` with
    // `-Cprofile-generate` (docs/WORKFLOW.md).
    println!("cargo::rustc-check-cfg=cfg(dsport_pgo_gen)");
    // The release version (the workspace's, Cargo.toml at the root): `REGALIANT_VERSION`. The
    // crate's own version stays fixed (its Cargo.toml says why).
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../Cargo.toml");
    println!("cargo:rerun-if-changed={}", root.display());
    let text = std::fs::read_to_string(&root).expect("the workspace's Cargo.toml");
    let version = text
        .split("[workspace.package]")
        .nth(1)
        .and_then(|t| t.lines().find_map(|l| l.trim().strip_prefix("version = \"").and_then(|v| v.strip_suffix('"'))))
        .expect("[workspace.package] version");
    println!("cargo:rustc-env=REGALIANT_VERSION={version}");
    // Keep bottom-up allocations (heaps, stacks) in the low 4 GiB so the guest
    // address range (16 GiB .. 8 TiB) can be reserved at startup. Image ASLR stays on.
    println!("cargo:rustc-link-arg-bins=/HIGHENTROPYVA:NO");
    // The program icon, also the game window's (`gpu::vk::window`).
    link_icon(&std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../assets/regaliant.ico"));

    // Audio decoders (third_party/, see THIRD_PARTY.md).
    let tp = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../third_party");
    let at9 = tp.join("libatrac9/src");
    let mut b = cc::Build::new();
    b.define("COMPILING_DLL", None).warnings(false).opt_level(2);
    for f in std::fs::read_dir(&at9).expect("libatrac9 sources") {
        let p = f.unwrap().path();
        if p.extension().is_some_and(|e| e == "c") {
            println!("cargo:rerun-if-changed={}", p.display());
            b.file(p);
        }
    }
    b.compile("atrac9");
    let mp3 = tp.join("minimp3");
    println!("cargo:rerun-if-changed={}", mp3.join("ds_mp3.c").display());
    cc::Build::new().file(mp3.join("ds_mp3.c")).include(&mp3).warnings(false).opt_level(2).compile("dsmp3");
}
