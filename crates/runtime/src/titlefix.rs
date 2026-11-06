//! Patches of the title's own code (PPSA01341 01.007.000): stand-ins for
//! behavior this port does not reproduce yet, fixes of title bugs the port's
//! runs reach, adaptations to a shared PC (`jobidle`) and PC controls
//! (`camhook`). Each fix checks the original bytes first (another
//! title version is left alone and the mismatch logged), every applied fix is
//! logged, and `DS_TITLEFIX=0` turns them all off (to investigate the
//! underlying problem).

use std::sync::atomic::{AtomicU64, Ordering};

enum Change {
    /// Instructions that run before the ones `at` bytes into `from` (in a
    /// trampoline, `patch::Patcher::insert`), built at load time (they may
    /// name the fix's run counter).
    Insert { at: usize, code: fn(counter: u64) -> Vec<u8> },
    /// A host function called with the registers before the instructions
    /// `at` bytes into `from` (`patch::Patcher::hook`).
    Hook { at: usize, handler: extern "sysv64" fn(*const u64, u32) },
}

struct Fix {
    name: &'static str,
    /// Module-relative address of `from` in the main executable.
    offset: usize,
    from: &'static [u8],
    change: Change,
    kind: &'static str,
    why: &'static str,
    /// Applied only when this environment variable is set (diagnosis).
    only_with: Option<&'static str>,
}

/// How many times each fix's inserted code ran (`status`; hooks keep their
/// own statistics).
static RUNS: [AtomicU64; FIXES.len()] = [const { AtomicU64::new(0) }; FIXES.len()];
/// Which fixes were applied.
static APPLIED: [std::sync::atomic::AtomicBool; FIXES.len()] = [const { std::sync::atomic::AtomicBool::new(false) }; FIXES.len()];

/// The stand-in used until 2026-10-11 (a byte patch) kept static meshes off
/// the GPU-driven batches (eboot+0xa21e77 `jne` -> `nop; jmp`), which drew
/// nothing here: the title's batch culling stores compare ballots, which the
/// shader translation now builds completely (STATUS: architecture).
const FIXES: &[Fix] = &[
    Fix {
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
        only_with: None,
    },
    // BPE's wait for work (eboot+0x83b640), where its scan of this worker's
    // job counters found nothing: `pause; mov r15,[rbx+r14*8+0E10h]`.
    Fix {
        name: "idle job workers sleep (wait)",
        offset: 0x83b6fd,
        from: &[0xf3, 0x90, 0x4e, 0x8b, 0xbc, 0xf3, 0x10, 0x0e, 0x00, 0x00],
        change: Change::Hook { at: 0, handler: crate::jobidle::wait_hook },
        kind: "shared-PC adaptation, `tune job_idle`",
        why: "the title's 13 job workers poll for work on 13 logical CPUs at all times",
        only_with: None,
    },
    // BPE's take (eboot+0x83b880), at the end of its pass over one queue
    // (`add r15,9220h`; the last queue's end is a take that found nothing).
    Fix {
        name: "idle job workers sleep (take)",
        offset: 0x83bf17,
        from: &[0x49, 0x81, 0xc7, 0x20, 0x92, 0x00, 0x00],
        change: Change::Hook { at: 0, handler: crate::jobidle::take_hook },
        kind: "shared-PC adaptation, `tune job_idle`",
        why: "jobs whose conditions do not hold yet are taken and put back by every worker without pause",
        only_with: None,
    },
    // The worker loop (eboot+0x83b520) running a job it took: `mov rdi,rbx;
    // mov rsi,r15` before the call.
    Fix {
        name: "idle job workers sleep (run)",
        offset: 0x83b570,
        from: &[0x48, 0x89, 0xdf, 0x4c, 0x89, 0xfe],
        change: Change::Hook { at: 0, handler: crate::jobidle::run_hook },
        kind: "shared-PC adaptation, `tune job_idle`",
        why: "a worker that runs a job is busy again",
        only_with: None,
    },
    // Raw mouse look (`camhook`, FINDINGS.md 2026-10-21 evening): the camera
    // stick reader (eboot+0x18bd60) after it loaded the stick into rdx: `mov
    // edi,[rbx+74h]; mov rsi,[rbx]` (rbx the camera's work context).
    Fix {
        name: "raw mouse look (stick)",
        offset: 0x18bd99,
        from: &[0x8b, 0x7b, 0x74, 0x48, 0x8b, 0x33],
        change: Change::Hook { at: 0, handler: crate::camhook::stick_hook },
        kind: "PC controls, `mouse_look_raw`",
        why: "mouse look through the right stick inherits the stick's response curve and top speed",
        only_with: None,
    },
    // ... and the end of the stage that adds the stick to the camera's pitch
    // and yaw (eboot+0x18cf00), which all of its paths reach: `mov
    // rax,[r15]; cmp rax,[rbp-20h]` (its stack guard check; rbx the context).
    Fix {
        name: "raw mouse look (angles)",
        offset: 0x18d1c4,
        from: &[0x49, 0x8b, 0x07, 0x48, 0x3b, 0x45, 0xe0],
        change: Change::Hook { at: 0, handler: crate::camhook::apply_hook },
        kind: "PC controls, `mouse_look_raw`",
        why: "the mouse turns the camera by its counts, not through the stick",
        only_with: None,
    },
    // Diagnosis (`DS_CAMDEBUG=1`): the camera rig update's entry
    // (eboot+0x181b00: `push rbp; mov rbp,rsp; push r15`), rdi the rig: the
    // rigs seen, for the `cam` control command.
    Fix {
        name: "camera rigs (diagnosis)",
        offset: 0x181b00,
        from: &[0x55, 0x48, 0x89, 0xe5, 0x41, 0x57],
        change: Change::Hook { at: 0, handler: crate::camhook::rig_hook },
        kind: "diagnosis",
        why: "the `cam` command lists the camera rigs and their controllers",
        only_with: Some("DS_CAMDEBUG"),
    },
    // The render resolution (`dispmode`, `DS_DISPLAY_MODE`): the per-frame
    // display mode choice (eboot+0x9fe6a3..), `test ecx,ecx; cmovne esi,ecx`
    // before `test eax,eax; cmovne esi,eax` (eax the options' mode).
    Fix {
        name: "display mode held (render resolution)",
        offset: 0x9fe6a8,
        from: &[0x85, 0xc9, 0x0f, 0x45, 0xf1],
        change: Change::Hook { at: 0, handler: crate::dispmode::hook_rax },
        kind: "PC settings, `DS_DISPLAY_MODE`",
        why: "the title's render resolution follows its options (2560x1440 at the gate) and no PC setting",
        only_with: Some("DS_DISPLAY_MODE"),
    },
    // ... the same choice kept as the effective mode (eboot+0xb26030, +0xD28),
    // the same instructions ...
    Fix {
        name: "display mode held (effective mode)",
        offset: 0xb2606e,
        from: &[0x85, 0xc9, 0x0f, 0x45, 0xf1],
        change: Change::Hook { at: 0, handler: crate::dispmode::hook_rax },
        kind: "PC settings, `DS_DISPLAY_MODE`",
        why: "the effective mode follows the options",
        only_with: Some("DS_DISPLAY_MODE"),
    },
    // ... and the UI's 4K output test (eboot+0xde88c0): `test r15d,r15d; mov
    // rcx,[rdx]` before `cmovne eax,r15d` (r15d the options' mode).
    Fix {
        name: "display mode held (UI scale)",
        offset: 0xde88c0,
        from: &[0x45, 0x85, 0xff, 0x48, 0x8b, 0x0a],
        change: Change::Hook { at: 0, handler: crate::dispmode::hook_r15 },
        kind: "PC settings, `DS_DISPLAY_MODE`",
        why: "the UI's scale follows the options' mode (4K for modes 3-5)",
        only_with: Some("DS_DISPLAY_MODE"),
    },
    // ... and the frame's copy of the settings (eboot+0xa4f2dd: memcpy of
    // 0xD8A bytes to r14+0x390), after it: `mov edx,1`.
    Fix {
        name: "display mode held (frame settings)",
        offset: 0xa4f2e9,
        from: &[0xba, 0x01, 0x00, 0x00, 0x00],
        change: Change::Hook { at: 0, handler: crate::dispmode::hook_snapshot },
        kind: "PC settings, `DS_DISPLAY_MODE`",
        why: "the frame's copy of the settings keeps the options' mode",
        only_with: Some("DS_DISPLAY_MODE"),
    },
    // The render size within a mode (`dispmode`, `DS_RENDER_SIZE`; and the
    // output size, `DS_OUTPUT_SIZE`): set display mode (eboot+0x90a600)
    // after it built the sizes for the render resolution manager: `lea
    // rdi,[rbx+0A8h]` before its call.
    Fix {
        name: "render size",
        offset: 0x90a6fc,
        from: &[0x48, 0x8d, 0xbb, 0xa8, 0x00, 0x00, 0x00],
        change: Change::Hook { at: 0, handler: crate::dispmode::hook_render_size },
        kind: "PC settings, `DS_RENDER_SIZE`",
        why: "the title renders at its mode's size (2560x1440 at the gate) and no PC setting",
        only_with: Some("DS_RENDER_SIZE|DS_OUTPUT_SIZE"),
    },
    // The output size (`dispmode`, `DS_OUTPUT_SIZE`, experiment): the
    // video-out constructor's size block before its use, `mov rsi,r15; mov
    // rdi,[rax]` ...
    Fix {
        name: "output size (constructor)",
        offset: 0x90b278,
        from: &[0x4c, 0x89, 0xfe, 0x48, 0x8b, 0x38],
        change: Change::Hook { at: 0, handler: crate::dispmode::hook_ctor_sizes },
        kind: "PC settings, `DS_OUTPUT_SIZE`",
        why: "the title outputs 3840x2160 whatever the display",
        only_with: Some("DS_OUTPUT_SIZE"),
    },
    // ... its two display targets, after `mov edi,0F00h; mov esi,870h`:
    // `mov edx,1` ...
    Fix {
        name: "output size (display target 0)",
        offset: 0x90af7e,
        from: &[0xba, 0x01, 0x00, 0x00, 0x00],
        change: Change::Hook { at: 0, handler: crate::dispmode::hook_targets },
        kind: "PC settings, `DS_OUTPUT_SIZE`",
        why: "the display targets are made at 3840x2160",
        only_with: Some("DS_OUTPUT_SIZE"),
    },
    Fix {
        name: "output size (display target 1)",
        offset: 0x90b06b,
        from: &[0xba, 0x01, 0x00, 0x00, 0x00],
        change: Change::Hook { at: 0, handler: crate::dispmode::hook_targets },
        kind: "PC settings, `DS_OUTPUT_SIZE`",
        why: "the display targets are made at 3840x2160",
        only_with: Some("DS_OUTPUT_SIZE"),
    },
    // ... the current mode's size getter (eboot+0x90ab90) before it stores
    // the output size: `mov [rsi+0Ch],rdx; mov [rsi+14h],r9d` ...
    Fix {
        name: "output size (size getter)",
        offset: 0x90ac57,
        from: &[0x48, 0x89, 0x56, 0x0c, 0x44, 0x89, 0x4e, 0x14],
        change: Change::Hook { at: 0, handler: crate::dispmode::hook_size_getter },
        kind: "PC settings, `DS_OUTPUT_SIZE`",
        why: "the mode's sizes come from per-mode tables (the UI's target among their users)",
        only_with: Some("DS_OUTPUT_SIZE"),
    },
    // ... and the UI's canvas before its layout (eboot+0xde8680): `lea
    // rsi,[rbp-30h]; mov rdi,rbx`.
    Fix {
        name: "output size (UI canvas)",
        offset: 0xde85db,
        from: &[0x48, 0x8d, 0x75, 0xd0, 0x48, 0x89, 0xdf],
        change: Change::Hook { at: 0, handler: crate::dispmode::hook_ui_canvas },
        kind: "PC settings, `DS_OUTPUT_SIZE`",
        why: "the UI is laid out on a 3840x2160 canvas in modes 3-5",
        only_with: Some("DS_OUTPUT_SIZE"),
    },
    // The title's handler of a changed setting (eboot+0xb4700, an observer of
    // its settings registry; rsi the setting): `push rbp; mov rbp,rsp; push
    // rbx; push rax`. The PC settings learn every change there (`pcsettings`).
    Fix {
        name: "setting changes (PC settings)",
        offset: 0xb4700,
        from: &[0x55, 0x48, 0x89, 0xe5, 0x53, 0x50],
        change: Change::Hook { at: 0, handler: crate::pcsettings::changed_hook },
        kind: "PC settings",
        why: "the port applies its settings when the title's SETTINGS screen changes them",
        only_with: None,
    },
    // The PC settings' values at the title's own settings paths (`pcsettings::store`): the
    // menu bindings' scans of the store (`UISetting_<type>::Evaluate`) for the PC keys ...
    Fix {
        name: "PC settings (int lookups)",
        offset: 0xc24245,
        from: &[0x49, 0x8b, 0x4d, 0x40, 0x49, 0x8b, 0x55, 0x48],
        change: Change::Hook { at: 0, handler: crate::pcsettings::store::eval_int_hook },
        kind: "PC settings",
        why: "the title's store holds only the settings it registers",
        only_with: None,
    },
    Fix {
        name: "PC settings (float lookups)",
        offset: 0xc24d15,
        from: &[0x49, 0x8b, 0x85, 0x90, 0x00, 0x00, 0x00],
        change: Change::Hook { at: 0, handler: crate::pcsettings::store::eval_float_hook },
        kind: "PC settings",
        why: "the title's store holds only the settings it registers",
        only_with: None,
    },
    // ... and the menu's setters (`mov qword ptr [rbp-48h],0` after each loads the store).
    Fix {
        name: "PC settings (int changes)",
        offset: 0xfa5cba,
        from: &[0x48, 0xc7, 0x45, 0xb8, 0x00, 0x00, 0x00, 0x00],
        change: Change::Hook { at: 0, handler: crate::pcsettings::store::set_int_hook },
        kind: "PC settings",
        why: "the port keeps and applies the PC settings the menu changes",
        only_with: None,
    },
    Fix {
        name: "PC settings (float changes)",
        offset: 0xfa5ee5,
        from: &[0x48, 0xc7, 0x45, 0xb8, 0x00, 0x00, 0x00, 0x00],
        change: Change::Hook { at: 0, handler: crate::pcsettings::store::set_float_hook },
        kind: "PC settings",
        why: "the port keeps and applies the PC settings the menu changes",
        only_with: None,
    },
];

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
    crate::dispmode::init();
    crate::pcsettings::convar::init(p.base, p.image);
    for (i, f) in FIXES.iter().enumerate() {
        if f.only_with.is_some_and(|v| v.split('|').all(|n| std::env::var_os(n).is_none())) {
            continue;
        }
        if p.image.get(f.offset..f.offset + f.from.len()) != Some(f.from) {
            crate::warn!("title fix '{}': unexpected bytes at eboot+{:#x} (another title version?); not applied", f.name, f.offset);
            continue;
        }
        let done = match &f.change {
            Change::Insert { at, code } => p.insert((f.offset + at) as u64, ranges, &code(RUNS[i].as_ptr() as u64)).map(|_| ()),
            Change::Hook { at, handler } => p.hook((f.offset + at) as u64, ranges, *handler as usize as u64, i as u32).map(|_| ()),
        };
        match done {
            Ok(()) => {
                APPLIED[i].store(true, Ordering::Relaxed);
                crate::info!("title fix '{}' at eboot+{:#x} ({}): {}", f.name, f.offset, f.kind, f.why);
            }
            Err(e) => crate::warn!("title fix '{}' at eboot+{:#x}: {e}; not applied", f.name, f.offset),
        }
    }
    // The idle job workers' sleep needs all three of its hooks.
    if (1..=3).all(|i| APPLIED[i].load(Ordering::Relaxed)) {
        crate::jobidle::enable();
    }
    // Raw mouse look needs both of its hooks.
    if (4..=5).all(|i| APPLIED[i].load(Ordering::Relaxed)) {
        crate::camhook::enable();
    }
}

/// For the status report: how often the applied fixes' code ran.
pub fn status() -> String {
    let runs: Vec<String> = FIXES
        .iter()
        .enumerate()
        .filter(|(i, f)| APPLIED[*i].load(Ordering::Relaxed) && matches!(f.change, Change::Insert { .. }))
        .map(|(i, f)| format!("'{}' ran {} times", f.name, RUNS[i].load(Ordering::Relaxed)))
        .collect();
    let fixes = if runs.is_empty() { String::new() } else { format!("    title fixes: {}\n", runs.join(", ")) };
    fixes + &crate::jobidle::status() + &crate::camhook::status() + &crate::dispmode::status() + &crate::pcsettings::store::status()
}
