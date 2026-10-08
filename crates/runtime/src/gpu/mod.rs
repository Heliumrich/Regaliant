//! GPU emulation front end for the PS5 (AGC): submission queues and flips.
//!
//! Status: skeleton. Command buffers submitted through libSceAgcDriver are
//! queued here; the gfx10 PM4 command processor and the RDNA shader recompiler
//! are not implemented yet (see docs/STATUS.md). Nothing here pretends work
//! was executed: every submission is counted and reported as not processed.

pub mod png;
pub mod vk;

use std::sync::atomic::{AtomicBool, AtomicU64};

/// Flips presented so far (VideoOut).
pub static FRAMES: AtomicU64 = AtomicU64::new(0);
/// Command buffer submissions received (DCB/ACB).
pub static SUBMITS: AtomicU64 = AtomicU64::new(0);
/// Debug: dump GPU-visible images at the next flip (control `dump`).
pub static DUMP_REQUEST: AtomicBool = AtomicBool::new(false);
