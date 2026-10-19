//! Patches of the title's own code (PPSA01341 01.007.000): stand-ins for
//! behavior this port does not reproduce yet, and fixes of title bugs the
//! port's runs reach. Each fix checks the original bytes first (another
//! title version is left alone and the mismatch logged), every applied fix is
//! logged, and `DS_TITLEFIX=0` turns them all off (to investigate the
//! underlying problem).

use std::sync::atomic::{AtomicU64, Ordering};

enum Change {
    /// Instructions that run before the ones `at` bytes into `from` (in a
    /// trampoline, `patch::Patcher::insert`), built at load time (they may
    /// name the fix's run counter).
    Insert { at: usize, code: fn(counter: u64) -> Vec<u8> },
}

struct Fix {
    name: &'static str,
    /// Module-relative address of `from` in the main executable.
    offset: usize,
    from: &'static [u8],
    change: Change,
    kind: &'static str,
    why: &'static str,
}

/// How many times each fix's inserted code ran (`status`).
static RUNS: [AtomicU64; 1] = [const { AtomicU64::new(0) }];
/// Which fixes were applied.
static APPLIED: [std::sync::atomic::AtomicBool; 1] = [const { std::sync::atomic::AtomicBool::new(false) }];

/// The stand-in used until 2026-10-11 (a byte patch) kept static meshes off
/// the GPU-driven batches (eboot+0xa21e77 `jne` -> `nop; jmp`), which drew
/// nothing here: the title's batch culling stores compare ballots, which the
/// shader translation now builds completely (STATUS: architecture).
const FIXES: &[Fix] = &[Fix {
    name: "collision mesh unload keeps a slot index",
    // BPCollisionMeshManager's unload (eboot+0xd51f0) moves components whose
    // collision body it releases back to the pending list
    // (`mov rcx,[rcx+28h]; cmp r12,[rbx+270h]`, rcx = the component) but
    // leaves their slot index (+0x33C) set. CollisionMeshBodyDestroy
    // (eboot+0xd4780) trusts that index: once the slot belongs to another
    // component, destroying this one releases the other's body and leaves
    // this one in the pending list, freed, for the next level's
    // loading-completed job to read (eboot+0xd4e16: World 8's levels
    // toured with three locations each, then the Nexus or 1-1; FINDINGS.md
    // 2026-10-19 night). The inserted code clears the index there, as the
    // destroy's own slot path does (eboot+0xd48f4).
    offset: 0xd5372,
    from: &[0x48, 0x8b, 0x49, 0x28, 0x4c, 0x3b, 0xa3, 0x70, 0x02, 0x00, 0x00],
    change: Change::Insert { at: 4, code: cmesh_unload_code },
    kind: "title bug",
    why: "components the collision mesh unload returns to the pending list lose their slot index",
}];

/// `mov dword ptr [rcx+33Ch], -1` after counting the run (rax kept; the
/// System V red zone below rsp skipped; the flags the counter changes are set
/// again by the displaced `cmp`).
fn cmesh_unload_code(counter: u64) -> Vec<u8> {
    let mut c = vec![0x48, 0x8d, 0x64, 0x24, 0x80, 0x50, 0x48, 0xb8]; // lea rsp,[rsp-80h]; push rax; mov rax, imm64
    c.extend_from_slice(&counter.to_le_bytes());
    c.extend_from_slice(&[0xf0, 0x48, 0xff, 0x00]); // lock inc qword ptr [rax]
    c.extend_from_slice(&[0x58, 0x48, 0x8d, 0xa4, 0x24, 0x80, 0x00, 0x00, 0x00]); // pop rax; lea rsp,[rsp+80h]
    c.extend_from_slice(&[0xc7, 0x81, 0x3c, 0x03, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff]); // mov dword ptr [rcx+33Ch], -1
    c
}

/// Applies the fixes to the main executable (`ranges`: its functions, for
/// inserted code).
pub fn apply(p: &mut crate::patch::Patcher, ranges: &[(u64, u64)]) {
    if std::env::var("DS_TITLEFIX").ok().as_deref() == Some("0") {
        crate::info!("title fixes off (DS_TITLEFIX=0)");
        return;
    }
    for (i, f) in FIXES.iter().enumerate() {
        if p.image.get(f.offset..f.offset + f.from.len()) != Some(f.from) {
            crate::warn!("title fix '{}': unexpected bytes at eboot+{:#x} (another title version?); not applied", f.name, f.offset);
            continue;
        }
        let done = match &f.change {
            Change::Insert { at, code } => p.insert((f.offset + at) as u64, ranges, &code(RUNS[i].as_ptr() as u64)).map(|_| ()),
        };
        match done {
            Ok(()) => {
                APPLIED[i].store(true, Ordering::Relaxed);
                crate::info!("title fix '{}' at eboot+{:#x} ({}): {}", f.name, f.offset, f.kind, f.why);
            }
            Err(e) => crate::warn!("title fix '{}' at eboot+{:#x}: {e}; not applied", f.name, f.offset),
        }
    }
}

/// For the status report: how often the applied fixes' code ran.
pub fn status() -> String {
    let runs: Vec<String> = FIXES
        .iter()
        .enumerate()
        .filter(|(i, _)| APPLIED[*i].load(Ordering::Relaxed))
        .map(|(i, f)| format!("'{}' ran {} times", f.name, RUNS[i].load(Ordering::Relaxed)))
        .collect();
    if runs.is_empty() { String::new() } else { format!("    title fixes: {}\n", runs.join(", ")) }
}
