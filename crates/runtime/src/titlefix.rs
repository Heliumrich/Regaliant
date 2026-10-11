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

/// None at present. The one used until 2026-10-11 kept static meshes off the
/// GPU-driven batches (eboot+0xa21e77 `jne` -> `nop; jmp`), which drew
/// nothing here: the title's batch culling stores compare ballots, which the
/// shader translation now builds completely (STATUS: architecture).
const FIXES: &[Fix] = &[];

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
