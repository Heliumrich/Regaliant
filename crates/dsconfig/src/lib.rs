//! What the port's runtime (`dsport`) and its launcher share: where the port keeps its files
//! (`paths`), the game folder's checks (`game`).

pub mod game;
pub mod paths;
pub mod sha;

/// The port's version (the workspace's).
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
