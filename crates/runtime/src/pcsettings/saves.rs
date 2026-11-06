//! The save folder (Game & Controls: Save Folder) and the import of decrypted PS5 saves (Import
//! PS5 Saves), both applied when the port starts: the title keeps its saves open while it runs.
//!
//! A changed folder that holds no saves of the game gets a copy of those in the folder used
//! before (local-data/save-folder.txt names it), so the progress moves along; the old folder is
//! left as it was. The import takes the save directories of a decrypted PS5 save of the game
//! (`SAVEDATA0PlayerProfile0`, `SAVEDATA0OptionsProfile0`: each with its `USR-DATA`, directly
//! or under a `PPSA0134x` folder, at any depth) from the folder "PS5 Saves" next to the game,
//! keeps the saves it replaces in save-backups, and renames the folder when done.

use std::path::{Path, PathBuf};

/// The Save Folder choice's options.
pub const FOLDERS: [&str; 3] = ["Game Folder", "Documents", "Saved Games"];
/// The folder the import reads, in the port's root.
pub const IMPORT_DIR: &str = "PS5 Saves";
/// The login user's directory (`savedata`: saves/<user>/<title>/).
fn user() -> String {
    crate::hle::libs::app::USER_ID.to_string()
}

fn known_folder(id: &windows::core::GUID) -> Option<PathBuf> {
    use windows::Win32::UI::Shell::{KF_FLAG_DEFAULT, SHGetKnownFolderPath};
    unsafe {
        let p = SHGetKnownFolderPath(id, KF_FLAG_DEFAULT, None).ok()?;
        let s = p.to_string().ok();
        windows::Win32::System::Com::CoTaskMemFree(Some(p.0 as *const _));
        s.map(PathBuf::from)
    }
}

/// The saves' folder of a Save Folder option (0, or a known folder Windows does not give:
/// local-data/saves in the port's folder).
pub fn folder(option: u32, root: &Path) -> PathBuf {
    use windows::Win32::UI::Shell::{FOLDERID_Documents, FOLDERID_SavedGames};
    let port = root.join("local-data").join("saves");
    match option {
        1 => known_folder(&FOLDERID_Documents).map(|d| d.join("My Games").join("Demon's Souls").join("saves")).unwrap_or(port),
        2 => known_folder(&FOLDERID_SavedGames).map(|d| d.join("Demon's Souls").join("saves")).unwrap_or(port),
        _ => port,
    }
}

/// Save directories (`SAVEDATA*` with a `USR-DATA` file) at most four levels under `dir`.
fn save_dirs(dir: &Path, depth: u32, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let p = e.path();
        if !p.is_dir() {
            continue;
        }
        let name = e.file_name().to_string_lossy().to_string();
        if name.starts_with("SAVEDATA") && p.join("USR-DATA").is_file() {
            out.push(p);
        } else if depth > 0 && !name.starts_with('.') {
            save_dirs(&p, depth - 1, out);
        }
    }
}

fn has_saves(saves: &Path) -> bool {
    let mut v = Vec::new();
    save_dirs(saves, 3, &mut v);
    !v.is_empty()
}

fn copy_dir(from: &Path, to: &Path) -> std::io::Result<u64> {
    std::fs::create_dir_all(to)?;
    let mut n = 0;
    for e in std::fs::read_dir(from)? {
        let e = e?;
        let target = to.join(e.file_name());
        if e.file_type()?.is_dir() {
            n += copy_dir(&e.path(), &target)?;
        } else {
            std::fs::copy(e.path(), &target)?;
            n += 1;
        }
    }
    Ok(n)
}

/// The saves' folder for this start (after `super::init`): the Save Folder setting's, with
/// the saves of the folder used before copied in when it has none; then a pending import.
pub fn start(root: &Path, title: &str) -> PathBuf {
    let option = super::store::value("Setting_PC_SaveFolder").unwrap_or(0);
    let dir = folder(option, root);
    let record = root.join("local-data").join("save-folder.txt");
    let before = std::fs::read_to_string(&record).ok().map(|s| PathBuf::from(s.trim())).filter(|p| !p.as_os_str().is_empty());
    if let Some(prev) = before.filter(|p| *p != dir) {
        if !has_saves(&dir) && has_saves(&prev) {
            match copy_dir(&prev, &dir) {
                Ok(n) => crate::info!("saves: the save folder is now {}: {n} files of the saves copied from {}", dir.display(), prev.display()),
                Err(e) => crate::warn!("saves: copying the saves from {} to {} failed: {e}", prev.display(), dir.display()),
            }
        } else {
            crate::info!("saves: the save folder is now {} (was {})", dir.display(), prev.display());
        }
    }
    if let Err(e) = std::fs::create_dir_all(&dir) {
        crate::warn!("saves: cannot make the save folder {}: {e}", dir.display());
    }
    let _ = std::fs::write(&record, format!("{}\n", dir.display()));
    if super::store::value("Setting_PC_ImportSaves") == Some(1) {
        import(&root.join(IMPORT_DIR), &dir, title);
        super::store::set_value("Setting_PC_ImportSaves", 0);
    }
    dir
}

/// Imports the save directories found in `from` into `saves` (the current ones kept in
/// save-backups beside it first); `from` renamed when anything was imported.
pub fn import(from: &Path, saves: &Path, title: &str) -> usize {
    let mut found = Vec::new();
    save_dirs(from, 4, &mut found);
    if found.is_empty() {
        crate::warn!("saves: PS5 import: no save directory (SAVEDATA... with USR-DATA) in {}", from.display());
        return 0;
    }
    let stamp = crate::utc_stamp();
    let base = saves.join(user()).join(title);
    let mut done = 0;
    for src in &found {
        let name = src.file_name().unwrap_or_default();
        let dst = base.join(name);
        if dst.is_dir() {
            let keep = saves.parent().unwrap_or(saves).join("save-backups").join(user()).join(title).join(name).join(format!("{stamp}-before-import"));
            if let Err(e) = copy_dir(&dst, &keep) {
                crate::warn!("saves: PS5 import: keeping {} failed ({e}): not imported", dst.display());
                continue;
            }
            crate::info!("saves: PS5 import: the current {} kept in {}", name.to_string_lossy(), keep.display());
        }
        // The game's files from the import; the save's metadata (sce_sys) kept when there is
        // one, the import's otherwise.
        let mut n = 0u64;
        let mut ok = true;
        if let Ok(rd) = std::fs::read_dir(src) {
            for e in rd.flatten() {
                let p = e.path();
                let target = dst.join(e.file_name());
                let r = if p.is_dir() {
                    if e.file_name() == "sce_sys" && target.is_dir() {
                        continue;
                    }
                    copy_dir(&p, &target)
                } else {
                    std::fs::create_dir_all(&dst).and_then(|_| std::fs::copy(&p, &target)).map(|_| 1)
                };
                match r {
                    Ok(k) => n += k,
                    Err(e) => {
                        crate::warn!("saves: PS5 import: {} -> {}: {e}", p.display(), target.display());
                        ok = false;
                    }
                }
            }
        }
        if ok {
            done += 1;
            crate::info!("saves: PS5 import: {} imported from {} ({n} files)", name.to_string_lossy(), src.display());
        }
    }
    if done > 0 {
        let to = from.with_file_name(format!("{} (imported {stamp})", from.file_name().unwrap_or_default().to_string_lossy()));
        match std::fs::rename(from, &to) {
            Ok(()) => crate::info!("saves: PS5 import: {} renamed to {}", from.display(), to.display()),
            Err(e) => crate::warn!("saves: PS5 import: renaming {} failed: {e}", from.display()),
        }
    }
    done
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn import_finds_nested_saves_and_keeps_the_current() {
        let base = std::env::temp_dir().join(format!("dsport-ps5import-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let from = base.join(IMPORT_DIR);
        let player = from.join("export").join("PPSA01342").join("SAVEDATA0PlayerProfile0");
        std::fs::create_dir_all(player.join("sce_sys")).unwrap();
        std::fs::write(player.join("USR-DATA"), "ps5").unwrap();
        std::fs::write(player.join("sce_sys").join("param.sfo"), "meta").unwrap();
        let saves = base.join("saves");
        let cur = saves.join(user()).join("PPSA01341").join("SAVEDATA0PlayerProfile0");
        std::fs::create_dir_all(cur.join("sce_sys")).unwrap();
        std::fs::write(cur.join("USR-DATA"), "pc").unwrap();
        std::fs::write(cur.join("sce_sys").join("param.bin"), "ours").unwrap();
        assert_eq!(import(&from, &saves, "PPSA01341"), 1);
        assert_eq!(std::fs::read_to_string(cur.join("USR-DATA")).unwrap(), "ps5");
        // Our metadata kept, the import's not copied over it.
        assert!(cur.join("sce_sys").join("param.bin").is_file());
        assert!(!cur.join("sce_sys").join("param.sfo").exists());
        // The replaced save kept; the import folder renamed.
        let kept = saves.parent().unwrap().join("save-backups").join(user()).join("PPSA01341").join("SAVEDATA0PlayerProfile0");
        let k: Vec<_> = std::fs::read_dir(&kept).unwrap().flatten().collect();
        assert_eq!(k.len(), 1);
        assert_eq!(std::fs::read_to_string(k[0].path().join("USR-DATA")).unwrap(), "pc");
        assert!(!from.exists());
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn import_without_saves_changes_nothing() {
        let base = std::env::temp_dir().join(format!("dsport-ps5import-empty-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let from = base.join(IMPORT_DIR);
        std::fs::create_dir_all(from.join("SAVEDATA0PlayerProfile0")).unwrap();
        assert_eq!(import(&from, &base.join("saves"), "PPSA01341"), 0);
        assert!(from.exists());
        let _ = std::fs::remove_dir_all(&base);
    }
}
