//! Stand-in patches of the title's own code (PPSA01341 01.007.000) for
//! behavior this port does not reproduce yet. Each fix checks the original
//! bytes first (another title version is left alone and the mismatch logged),
//! every applied fix is logged, and `DS_TITLEFIX=0` turns them all off (to
//! investigate the underlying problem).

struct Fix {
    name: &'static str,
    /// Module-relative address in the main executable.
    offset: usize,
    from: &'static [u8],
    to: &'static [u8],
    why: &'static str,
}

const FIXES: &[Fix] = &[Fix {
    // eboot+0xa21e10 hands a render item to the GPU-driven batches (sets bit 0
    // of the item's byte +0x30, which removes it from the CPU draw lists) when
    // its shader program qualifies ([prog+0x3d8] == 4, no [prog+0x60], no flags
    // in [prog+0x3e4] / bit 4 of [prog+0x3f0]). Here the GPU-driven path draws
    // nothing for such items (the courtyard castle, most architecture), while
    // the CPU path draws them correctly (run 20261010-222845: the bit cleared
    // at run time brings the castle back). `jne` -> `nop; jmp` (same target):
    // every item takes the not-eligible path and stays on the CPU path.
    name: "static meshes on the CPU draw path",
    offset: 0xa21e77,
    from: &[0x0f, 0x85, 0x2b, 0x04, 0x00, 0x00],
    to: &[0x90, 0xe9, 0x2b, 0x04, 0x00, 0x00],
    why: "the GPU-driven batches draw nothing for these items here (STATUS: architecture)",
}];

/// Applies the fixes to the main executable's image (module-relative view).
pub fn apply(image: &mut [u8]) {
    if std::env::var("DS_TITLEFIX").ok().as_deref() == Some("0") {
        crate::info!("title fixes off (DS_TITLEFIX=0)");
        return;
    }
    for f in FIXES {
        match image.get_mut(f.offset..f.offset + f.from.len()) {
            Some(bytes) if bytes == f.from => {
                bytes[..f.to.len()].copy_from_slice(f.to);
                crate::info!("title fix '{}' at eboot+{:#x} (stand-in): {}", f.name, f.offset, f.why);
            }
            _ => crate::warn!("title fix '{}': unexpected bytes at eboot+{:#x} (another title version?); not applied", f.name, f.offset),
        }
    }
}
