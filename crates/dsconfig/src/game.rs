//! The game folder's checks: Demon's Souls (PS5, PPSA01341) at version 01.007.000, with its
//! executable and its two libraries decrypted, and its data folders. The runtime checks the
//! same at start and the launcher when a folder is chosen; both show `Problem`'s text.

use std::path::{Path, PathBuf};

/// The release the port was made and tested with (Europe).
pub const TITLE_ID: &str = "PPSA01341";

/// Demon's Souls' PS5 title ids (PPSA0134x: the regions' releases). Only PPSA01341 was tested;
/// the others are taken when their version matches (the executable's fingerprint says whether
/// its code is the tested one).
pub fn is_demons_souls(id: &str) -> bool {
    id.len() == 9 && id.starts_with("PPSA0134") && id.as_bytes()[8].is_ascii_digit()
}
/// The one version the port supports.
pub const VERSION: &str = "01.007.000";
/// The application folder's usual name.
pub const APP_DIR: &str = "PPSA01341-app0";
/// The decrypted executable this port was made and tested with (SHA-256).
pub const TESTED_EBOOT: &str = "4a6ca8a2904d02a9ff11aebcb10379c89974cd9bff1a37e09c21a7e782447674";
/// The title's own libraries the port loads (decrypted, in `sce_module`).
pub const MODULES: [&str; 2] = ["libc.prx", "libSceNpCppWebApi.prx"];
/// Data folders every complete copy holds.
const DATA_DIRS: [&str; 8] = ["cp11demonssouls", "levels", "sound", "text", "textures", "ui", "movies", "characters"];

#[derive(Clone, Debug)]
pub struct Game {
    /// The title id (`TITLE_ID`, or another region's release).
    pub id: String,
    /// The application folder (data and `sce_sys`), mounted read-only.
    pub app: PathBuf,
    /// The folder holding the decrypted `eboot.bin` and `sce_module`.
    pub code: PathBuf,
    pub version: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Problem {
    NotFound,
    OtherGame(String),
    Version(String),
    Param(String),
    NoExecutable,
    Encrypted,
    NoModule(String),
    EncryptedModule(String),
    MissingData(String),
}

impl std::fmt::Display for Problem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Problem::NotFound => write!(f, "No Demon's Souls game files were found in this folder. Choose the folder that holds eboot.bin and sce_sys."),
            Problem::OtherGame(id) => write!(f, "This folder holds another game ({id}). This port runs the PS5 game Demon's Souls (PPSA01341)."),
            Problem::Version(v) => write!(f, "This is version {v} of the game. This port needs version {VERSION}."),
            Problem::Param(e) => write!(f, "The game's sce_sys\\param.json could not be read ({e})."),
            Problem::NoExecutable => write!(f, "The game's eboot.bin is missing."),
            Problem::Encrypted => write!(f, "The game's eboot.bin is encrypted. The port needs the decrypted eboot.bin, in the game folder or in its decrypted folder."),
            Problem::NoModule(m) => write!(f, "{m} is missing. The port needs the game's decrypted libraries in sce_module."),
            Problem::EncryptedModule(m) => write!(f, "{m} is encrypted. The port needs the game's decrypted libraries in sce_module."),
            Problem::MissingData(d) => write!(f, "The game's {d} folder is missing. The game files are incomplete."),
        }
    }
}

fn has_param(d: &Path) -> bool {
    d.join("sce_sys").join("param.json").is_file()
}

/// The application folder at or under `dir`: `dir` itself, its `PPSA01341-app0`, or one
/// folder further down (a dump's folder holding the application folder).
pub fn locate(dir: &Path) -> Option<PathBuf> {
    if has_param(dir) {
        return Some(dir.to_path_buf());
    }
    if has_param(&dir.join(APP_DIR)) {
        return Some(dir.join(APP_DIR));
    }
    let mut subs: Vec<PathBuf> = std::fs::read_dir(dir).ok()?.flatten().map(|e| e.path()).filter(|p| p.is_dir()).collect();
    subs.sort();
    subs.into_iter().find_map(|s| [s.clone(), s.join(APP_DIR)].into_iter().find(|d| has_param(d)))
}

/// Whether `path` starts as an ELF file (decrypted); None when it cannot be read.
fn is_elf(path: &Path) -> Option<bool> {
    use std::io::Read;
    let mut b = [0u8; 4];
    let mut f = std::fs::File::open(path).ok()?;
    f.read_exact(&mut b).ok()?;
    Some(b == *b"\x7fELF")
}

/// The title's id and content version from `sce_sys/param.json`.
pub fn param(app: &Path) -> Result<(String, String), Problem> {
    let t = std::fs::read_to_string(app.join("sce_sys").join("param.json")).map_err(|e| Problem::Param(e.to_string()))?;
    let v: serde_json::Value = serde_json::from_str(&t).map_err(|e| Problem::Param(e.to_string()))?;
    let get = |k: &str| v.get(k).and_then(|x| x.as_str()).map(|s| s.trim().to_string());
    let id = get("titleId").ok_or_else(|| Problem::Param("no titleId".into()))?;
    let ver = get("contentVersion").ok_or_else(|| Problem::Param("no contentVersion".into()))?;
    Ok((id, ver))
}

/// Checks the game folder at or under `dir`.
pub fn check(dir: &Path) -> Result<Game, Problem> {
    let app = locate(dir).ok_or(Problem::NotFound)?;
    let (id, version) = param(&app)?;
    if !is_demons_souls(&id) {
        return Err(Problem::OtherGame(id));
    }
    if version != VERSION {
        return Err(Problem::Version(version));
    }
    // The decrypted executable: in `decrypted\` (as dumps keep it beside the encrypted one) or
    // the application folder's own.
    let code = if is_elf(&app.join("decrypted").join("eboot.bin")) == Some(true) {
        app.join("decrypted")
    } else {
        match is_elf(&app.join("eboot.bin")) {
            Some(true) => app.clone(),
            Some(false) => return Err(Problem::Encrypted),
            None => return Err(Problem::NoExecutable),
        }
    };
    for m in MODULES {
        match is_elf(&code.join("sce_module").join(m)) {
            Some(true) => {}
            Some(false) => return Err(Problem::EncryptedModule(m.into())),
            None => return Err(Problem::NoModule(m.into())),
        }
    }
    if let Some(d) = DATA_DIRS.iter().find(|d| !app.join(d).is_dir()) {
        return Err(Problem::MissingData(d.to_string()));
    }
    Ok(Game { id, app, code, version })
}

/// Whether the executable is the one the port was tested with (its SHA-256).
pub fn tested_executable(g: &Game) -> std::io::Result<bool> {
    Ok(crate::sha::file_sha256(&g.code.join("eboot.bin"))? == TESTED_EBOOT)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake(root: &Path, version: &str, decrypted_dir: bool) -> PathBuf {
        let app = root.join(APP_DIR);
        std::fs::create_dir_all(app.join("sce_sys")).unwrap();
        std::fs::write(app.join("sce_sys").join("param.json"), format!("{{\"titleId\": \"PPSA01341\", \"contentVersion\": \"{version}\"}}")).unwrap();
        for d in DATA_DIRS {
            std::fs::create_dir_all(app.join(d)).unwrap();
        }
        let code = if decrypted_dir { app.join("decrypted") } else { app.clone() };
        std::fs::create_dir_all(code.join("sce_module")).unwrap();
        std::fs::write(code.join("eboot.bin"), b"\x7fELF....").unwrap();
        if decrypted_dir {
            std::fs::write(app.join("eboot.bin"), b"\x54\x14\xf5\xee....").unwrap();
        }
        for m in MODULES {
            std::fs::write(code.join("sce_module").join(m), b"\x7fELF....").unwrap();
        }
        app
    }

    #[test]
    fn checks() {
        let tmp = std::env::temp_dir().join(format!("dsconfig-game-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let dump = tmp.join("Demon-Souls-dump");
        let app = fake(&dump, VERSION, true);
        // (The dump's folder, the application folder, and the folder above the dump.)
        for d in [&dump, &app, &tmp] {
            let g = check(d).unwrap();
            assert_eq!(g.app, app);
            assert_eq!(g.code, app.join("decrypted"));
        }
        std::fs::remove_file(app.join("decrypted").join("sce_module").join("libc.prx")).unwrap();
        assert_eq!(check(&dump).unwrap_err(), Problem::NoModule("libc.prx".into()));
        std::fs::remove_dir_all(app.join("decrypted")).unwrap();
        assert_eq!(check(&dump).unwrap_err(), Problem::Encrypted);
        std::fs::remove_dir_all(&tmp).unwrap();

        let app = fake(&tmp, "01.006.000", false);
        assert_eq!(check(&tmp).unwrap_err(), Problem::Version("01.006.000".into()));
        std::fs::remove_dir_all(&tmp).unwrap();
        let app2 = fake(&tmp, VERSION, false);
        assert_eq!(app, app2);
        assert_eq!(check(&tmp).unwrap().code, app2);
        std::fs::remove_dir_all(app2.join("ui")).unwrap();
        assert_eq!(check(&tmp).unwrap_err(), Problem::MissingData("ui".into()));
        std::fs::remove_dir_all(&tmp).unwrap();
        assert_eq!(check(&tmp).unwrap_err(), Problem::NotFound);
        assert!(!Problem::Encrypted.to_string().contains(';'));
    }
}
