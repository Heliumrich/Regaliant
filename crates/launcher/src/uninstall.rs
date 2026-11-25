//! Uninstall: removes what Regaliant added to this PC, asking first, and the saves only when the
//! player says so (issue #19 and the user's request of 2026-11-25).
//!
//! - Removed: the program files the package holds, by name only (`PACKAGE_FILES`: a player may
//!   have put the package in a folder with other files, even the game's), the prepared shaders,
//!   logs and the game's runtime folders (the cache folder), and the settings (the user folder).
//! - Kept always: the game's folder and its .zar archive (Regaliant only reads them), add-on
//!   packages ("Add-ons" folders: the player's own), the player's own `nvngx_dlss.dll`.
//! - Saves (each Save Folder's saves, the save backups, the PS5 Saves import folder, the trophy
//!   progress): kept unless the player chooses to delete them.
//!
//! The launcher's own program file cannot be deleted while it runs: a hidden command deletes it
//! after the launcher has closed, then the package's folders if they are empty.

use crate::{App, TITLE};
use dsconfig::paths::{Layout, Paths};
use std::path::{Path, PathBuf};
use windows_sys::Win32::UI::WindowsAndMessaging::*;

/// The files a package holds besides the launcher itself (`scripts\package.ps1`'s allowlist), and
/// the portable layout's marker.
const PACKAGE_FILES: &[&str] = &[
    "regaliant.exe",
    "README.txt",
    "portable.txt",
    "upscalers\\libxess.dll",
    "upscalers\\libxess.license.txt",
    "upscalers\\libxess.third-party-programs.txt",
    "upscalers\\sl.interposer.dll",
    "upscalers\\sl.common.dll",
    "upscalers\\sl.dlss.dll",
    "upscalers\\streamline.license.txt",
    "upscalers\\streamline.3rd-party-licenses.md",
    "upscalers\\amd_fidelityfx_loader_vk.dll",
    "upscalers\\amd_fidelityfx_upscaler_vk.dll",
    "upscalers\\fidelityfx.license.md",
];

/// Names in the settings folder that hold saves (kept unless the player deletes the saves).
const SAVE_ENTRIES: &[&str] = &["saves", "save-backups", "PS5 Saves", "achievements.txt", "save-folder.txt"];
/// Add-on packages: the player's own, always kept.
const ADDONS: &str = "Add-ons";

fn known_folder(id: &windows::core::GUID) -> Option<PathBuf> {
    use windows::Win32::UI::Shell::{KF_FLAG_DEFAULT, SHGetKnownFolderPath};
    unsafe {
        let p = SHGetKnownFolderPath(id, KF_FLAG_DEFAULT, None).ok()?;
        let s = p.to_string().ok();
        windows::Win32::System::Com::CoTaskMemFree(Some(p.0 as *const _));
        s.map(PathBuf::from)
    }
}

/// The saves folders of the Save Folder's other two choices (Documents, Saved Games) that this
/// install used: the one its save-folder.txt names. Another install may keep its saves in the
/// other one, and a folder this install never used is never touched.
fn other_save_folders(paths: &Paths) -> Vec<PathBuf> {
    use windows::Win32::UI::Shell::{FOLDERID_Documents, FOLDERID_SavedGames};
    let used = std::fs::read_to_string(paths.save_folder_record()).unwrap_or_default();
    let used = used.trim().trim_end_matches(['\\', '/']).to_lowercase();
    let mut v = Vec::new();
    if let Some(d) = known_folder(&FOLDERID_Documents) {
        v.push(d.join("My Games").join("Demon's Souls").join("saves"));
    }
    if let Some(d) = known_folder(&FOLDERID_SavedGames) {
        v.push(d.join("Demon's Souls").join("saves"));
    }
    v.retain(|f| !used.is_empty() && f.to_string_lossy().trim_end_matches(['\\', '/']).to_lowercase() == used);
    v
}

/// Bytes under `p` (files only).
fn size_of(p: &Path) -> u64 {
    let Ok(rd) = std::fs::read_dir(p) else { return std::fs::metadata(p).map_or(0, |m| m.len()) };
    rd.flatten()
        .map(|e| match e.file_type() {
            Ok(t) if t.is_dir() => size_of(&e.path()),
            Ok(_) => e.metadata().map_or(0, |m| m.len()),
            Err(_) => 0,
        })
        .sum()
}

fn human(b: u64) -> String {
    if b >= 1 << 30 { format!("{:.1} GB", b as f64 / (1u64 << 30) as f64) } else { format!("{} MB", (b >> 20).max(1)) }
}

/// Removes `p` (a file or a folder with all it holds); failures are collected.
fn remove(p: &Path, failed: &mut Vec<String>) {
    let r = if p.is_dir() { std::fs::remove_dir_all(p) } else { std::fs::remove_file(p) };
    if let Err(e) = r {
        if p.exists() {
            failed.push(format!("{}: {e}", p.display()));
        }
    }
}

/// Removes the entries of `dir` except `keep` (by name), then `dir` itself when that left it
/// empty.
fn clear_folder(dir: &Path, keep: &[&str], failed: &mut Vec<String>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        if keep.iter().any(|k| k.eq_ignore_ascii_case(&name)) {
            continue;
        }
        remove(&e.path(), failed);
    }
    let _ = std::fs::remove_dir(dir);
}

/// Whether any save is in the places that hold them.
fn saves_found(places: &[PathBuf]) -> bool {
    places.iter().any(|p| std::fs::read_dir(p).is_ok_and(|mut rd| rd.next().is_some()) || p.is_file())
}

/// Whether `p` is one of this layout's own folders: in the installed layout a folder named after
/// the app, in the portable one the `user` folder beside the programs. Nothing else is cleared.
fn ours(paths: &Paths, p: &Path) -> bool {
    match paths.layout {
        Layout::Installed => p.file_name().is_some_and(|n| n.eq_ignore_ascii_case(dsconfig::paths::APP_FOLDER)),
        Layout::Portable => p.parent() == Some(paths.exe_dir.as_path()) && p.file_name().is_some_and(|n| n.eq_ignore_ascii_case("user")),
        Layout::Dev => false,
    }
}

/// Where saves are kept: the settings folder's save entries and `others` (`other_save_folders`).
fn save_places(paths: &Paths, others: &[PathBuf]) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = SAVE_ENTRIES.iter().map(|n| paths.user.join(n)).collect();
    v.extend(others.iter().cloned());
    v
}

/// The Uninstall button.
pub fn run(app: &mut App) {
    let paths = app.paths.clone();
    if paths.layout == Layout::Dev {
        unsafe { crate::ui::message(app.hwnd, "This copy runs from a development checkout. Uninstall is for installed copies.", TITLE, MB_ICONINFORMATION | MB_OK) };
        return;
    }
    if app.running.is_some() {
        app.set_status("Close the game first.");
        return;
    }
    if crate::shaders::running(app) || crate::game::compressing(app) {
        app.set_status("Stop the shader preparation or the compression first.");
        return;
    }
    let data = human(size_of(&paths.cache) + if paths.user != paths.cache { size_of(&paths.user) } else { 0 });
    let text = format!("Uninstall Regaliant from this PC?\n\nThis removes its program files, its prepared shaders, logs and settings ({data}).\n\nYour game files and add-ons are not touched.");
    if unsafe { crate::ui::message(app.hwnd, &text, TITLE, MB_ICONWARNING | MB_OKCANCEL | MB_DEFBUTTON2) } != IDOK {
        return;
    }
    let others = other_save_folders(&paths);
    let mut delete_saves = false;
    if saves_found(&save_places(&paths, &others)) {
        let text = "Delete your saves too?\n\nYes deletes them for good.\nNo keeps them, so a later install continues where you left off.";
        match unsafe { crate::ui::message(app.hwnd, text, TITLE, MB_ICONQUESTION | MB_YESNOCANCEL | MB_DEFBUTTON2) } {
            IDYES => delete_saves = true,
            IDNO => {}
            _ => return,
        }
    }
    let (failed, kept) = perform(&paths, &others, delete_saves);
    let mut text = if failed.is_empty() { "Regaliant was removed.".to_string() } else { format!("Some files could not be removed:\n{}", failed.join("\n")) };
    if !kept.is_empty() {
        text.push_str("\n\n");
        text.push_str(&kept.join("\n"));
    }
    unsafe { crate::ui::message(app.hwnd, &text, TITLE, if failed.is_empty() { MB_ICONINFORMATION | MB_OK } else { MB_ICONWARNING | MB_OK }) };
    remove_self_after_exit(&paths.exe_dir);
    unsafe { PostMessageW(app.hwnd, WM_CLOSE, 0, 0) };
}

/// The removal itself (`run` asked first; `others`: `other_save_folders`): what failed and what
/// was kept, as lines for the player.
fn perform(paths: &Paths, others: &[PathBuf], delete_saves: bool) -> (Vec<String>, Vec<String>) {
    let (user, cache) = (&paths.user, &paths.cache);
    let mut failed = Vec::new();
    // Prepared shaders, logs, the runtime folders (add-ons kept).
    if ours(paths, cache) && cache != user {
        clear_folder(cache, &[ADDONS], &mut failed);
    }
    // Settings, and the saves when asked (add-ons kept).
    if ours(paths, user) {
        let mut keep = vec![ADDONS];
        if !delete_saves {
            keep.extend_from_slice(SAVE_ENTRIES);
        }
        clear_folder(user, &keep, &mut failed);
    }
    if delete_saves {
        for f in others {
            if f.is_dir() {
                remove(f, &mut failed);
                // (Its "Demon's Souls" folder when nothing else is in it.)
                if let Some(parent) = f.parent() {
                    let _ = std::fs::remove_dir(parent);
                }
            }
        }
    }
    // The program files, by name.
    for f in PACKAGE_FILES {
        let p = paths.exe_dir.join(f);
        if p.is_file() {
            remove(&p, &mut failed);
        }
    }
    let up = paths.exe_dir.join("upscalers");
    let _ = std::fs::remove_dir(&up);
    let mut kept = Vec::new();
    if !delete_saves && saves_found(&save_places(paths, others)) {
        kept.push(format!("Your saves are kept in {}.", user.display()));
    }
    if up.join("nvngx_dlss.dll").is_file() {
        kept.push(format!("Your own DLSS file is kept in {}.", up.display()));
    }
    (failed, kept)
}

/// After the launcher has closed: its program file deleted, then the upscalers folder and the
/// package's folder if they are empty (a hidden command that waits two seconds; `rmdir` without
/// `/s` removes only empty folders).
fn remove_self_after_exit(exe_dir: &Path) {
    use std::os::windows::process::CommandExt;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    let Ok(exe) = std::env::current_exe() else { return };
    let line = format!(
        "ping -n 3 127.0.0.1 >nul & del /f /q \"{}\" & rmdir \"{}\" & rmdir \"{}\"",
        exe.display(),
        exe_dir.join("upscalers").display(),
        exe_dir.display()
    );
    let _ = std::process::Command::new("cmd.exe").arg("/d").arg("/c").raw_arg(&line).current_dir(std::env::temp_dir()).creation_flags(CREATE_NO_WINDOW).spawn();
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A portable install in a temporary folder: everything Regaliant adds goes, the saves only
    /// when asked, while add-ons, the player's DLSS file, a game folder, an archive and another
    /// file beside the programs stay.
    #[test]
    fn removes_only_its_own() {
        let root = std::env::temp_dir().join(format!("regaliant-uninstall-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let put = |p: &str| {
            let f = root.join(p);
            std::fs::create_dir_all(f.parent().unwrap()).unwrap();
            std::fs::write(f, b"x").unwrap();
        };
        let programs = ["regaliant.exe", "regaliant-launcher.exe", "README.txt", "portable.txt", "upscalers/libxess.dll", "upscalers/nvngx_dlss.dll", "upscalers/sl.dlss.dll"];
        let data = ["user/pcsettings.txt", "user/shaders/a.bin", "user/logs/x/dsport.log", "user/saves/u/SAVEDATA0PlayerProfile0/USR-DATA", "user/save-backups/b", "user/Add-ons/pkg"];
        let others = ["PPSA01341-app0/eboot.bin", "game.zar", "notes.txt"];
        for f in programs.iter().chain(&data).chain(&others) {
            put(f);
        }
        let paths = Paths::for_exe(&root.join("regaliant-launcher.exe"));
        assert_eq!(paths.layout, Layout::Portable);
        // (A Save Folder choice stands in for Documents' inside the test's own folder: the PC's own
        // folders are never touched by a test.)
        put("docs/My Games/Demon's Souls/saves/u/SAVEDATA0PlayerProfile0/USR-DATA");
        let others = vec![root.join("docs/My Games/Demon's Souls/saves")];
        let (failed, kept) = perform(&paths, &others, false);
        assert!(failed.is_empty(), "{failed:?}");
        for gone in ["regaliant.exe", "README.txt", "portable.txt", "upscalers/libxess.dll", "upscalers/sl.dlss.dll", "user/pcsettings.txt", "user/shaders", "user/logs"] {
            assert!(!root.join(gone).exists(), "{gone} left");
        }
        for stays in ["upscalers/nvngx_dlss.dll", "user/saves/u/SAVEDATA0PlayerProfile0/USR-DATA", "user/save-backups/b", "user/Add-ons/pkg", "PPSA01341-app0/eboot.bin", "game.zar", "notes.txt", "regaliant-launcher.exe"] {
            assert!(root.join(stays).exists(), "{stays} removed");
        }
        assert_eq!(kept.len(), 2, "{kept:?}");
        // Then the saves too; add-ons and the game's files still stay.
        assert!(root.join("docs/My Games/Demon's Souls/saves/u/SAVEDATA0PlayerProfile0/USR-DATA").exists());
        let (failed, _) = perform(&paths, &others, true);
        assert!(failed.is_empty(), "{failed:?}");
        assert!(!root.join("user/saves").exists() && !root.join("user/save-backups").exists() && !root.join("docs/My Games/Demon's Souls").exists());
        assert!(root.join("user/Add-ons/pkg").exists() && root.join("PPSA01341-app0/eboot.bin").exists() && root.join("game.zar").exists());
        let _ = std::fs::remove_dir_all(&root);
    }
}
