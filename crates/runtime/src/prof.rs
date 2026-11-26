//! Tracy instrumentation, with the `tracy` feature (`cargo build --release --features tracy`;
//! the Tracy profiler v0.14.1 connects to the running game, `docs/building.md`). Without the
//! feature every macro and function here compiles to nothing.
//!
//! * `zone!("name")`: a zone from here to the end of the enclosing block.
//! * `zone_text!("name", text)`: the same with a text shown on the zone (a program's address).
//! * `plot!("name", value)`: a value plotted over time.
//! * `zone_fine!`: the same, only with `DS_TRACY_FINE=1` (draws, dispatches, HLE calls).
//! * Every HLE function (`hle::sysv!`): a fine zone named after it, on the game's threads.
//! * The GPU's batches: a track per queue (`gpu_batch`, from the batch timestamps).
//! * `frame()`: the end of a frame (the game's flips, `hle::libs::videoout`).
//! * `thread_name(name)`: the calling thread's name in the profile (`log::set_thread_name`).
//!
//! The client runs on demand: nothing is collected until the profiler connects.

/// Starts the client (the first thing `main` does; the zones need it running).
pub fn start() {
    #[cfg(feature = "tracy")]
    {
        let _ = tracy_client::Client::start();
    }
}

/// The calling thread's name in the profile.
pub fn thread_name(_name: &str) {
    #[cfg(feature = "tracy")]
    if let Some(c) = tracy_client::Client::running() {
        c.set_thread_name(_name);
    }
}

/// A frame ended (a flip of the game's).
pub fn frame() {
    #[cfg(feature = "tracy")]
    if let Some(c) = tracy_client::Client::running() {
        c.frame_mark();
    }
}

/// A batch's span on the GPU (its start and end timestamps, the device's ticks) on the track of
/// its queue (`queue`, a GPU context of its own each): `gpu::exec::collect_timestamps`, when the
/// batch completed. Each context is aligned with the CPU's clock once, by a calibrated reading
/// (VK_EXT_calibrated_timestamps) as it is made; without it there is no GPU track.
pub fn gpu_batch(_queue: &str, _name: &str, _start: u64, _end: u64) {
    #[cfg(feature = "tracy")]
    {
        use parking_lot::Mutex;
        use std::collections::HashMap;
        static CTX: Mutex<Option<HashMap<String, Option<tracy_client::GpuContext>>>> = Mutex::new(None);
        let Some(client) = tracy_client::Client::running() else { return };
        if !tracy_client::Client::is_connected() || _end <= _start {
            return;
        }
        let mut g = CTX.lock();
        let map = g.get_or_insert_with(HashMap::new);
        let ctx = map.entry(_queue.to_string()).or_insert_with(|| {
            let c = crate::gpu::vk::try_ctx()?;
            let (gpu, _) = c.calibrate()?;
            client.new_gpu_context(Some(&format!("GPU {_queue}")), tracy_client::GpuContextType::Vulkan, gpu as i64, c.limits.timestamp_period).ok()
        });
        let Some(ctx) = ctx.as_ref() else { return };
        // Each distinct name is a source location of Tracy's (32K at most in a capture, past
        // which the profiler stops with "Too many source locations"): past 8192 names, the
        // queue's name alone.
        static NAMES: Mutex<Option<std::collections::HashSet<String>>> = Mutex::new(None);
        let name = {
            let mut n = NAMES.lock();
            let set = n.get_or_insert_with(Default::default);
            if set.contains(_name) || (set.len() < 8192 && set.insert(_name.to_string())) { _name } else { _queue }
        };
        if let Ok(mut span) = ctx.span_alloc(name, "batch", file!(), line!()) {
            span.end_zone();
            span.upload_timestamp_start(_start as i64);
            span.upload_timestamp_end(_end as i64);
        }
    }
}

/// Whether zones are recorded (the feature).
pub const ON: bool = cfg!(feature = "tracy");

/// Whether the fine zones are recorded (`DS_TRACY_FINE=1`): one per draw, dispatch, stage
/// preparation and HLE call. Millions a second: they slow the game several times over while the
/// profiler is connected, so a capture without them is the one whose timing holds.
#[inline]
pub fn fine() -> bool {
    #[cfg(feature = "tracy")]
    {
        static FINE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *FINE.get_or_init(|| std::env::var("DS_TRACY_FINE").is_ok_and(|v| v == "1"))
    }
    #[cfg(not(feature = "tracy"))]
    false
}

/// A zone to the end of the enclosing block.
#[macro_export]
macro_rules! zone {
    ($name:literal) => {
        #[cfg(feature = "tracy")]
        let _tracy_zone = ::tracy_client::Client::running().map(|c| c.span(::tracy_client::span_location!($name), 0));
    };
    // (A name from another macro: `stringify!`, `concat!`.)
    (@expr $name:expr) => {
        #[cfg(feature = "tracy")]
        let _tracy_zone = ::tracy_client::Client::running().map(|c| c.span(::tracy_client::span_location!($name), 0));
    };
}

/// A fine zone (`fine`): recorded only with `DS_TRACY_FINE=1`.
#[macro_export]
macro_rules! zone_fine {
    ($name:literal) => {
        #[cfg(feature = "tracy")]
        let _tracy_zone = if $crate::prof::fine() { ::tracy_client::Client::running().map(|c| c.span(::tracy_client::span_location!($name), 0)) } else { None };
    };
    (@expr $name:expr) => {
        #[cfg(feature = "tracy")]
        let _tracy_zone = if $crate::prof::fine() { ::tracy_client::Client::running().map(|c| c.span(::tracy_client::span_location!($name), 0)) } else { None };
    };
}

/// A zone as a value, ended when dropped (`drop(guard)` ends it before the block does).
#[macro_export]
macro_rules! zone_guard {
    ($name:literal) => {{
        #[cfg(feature = "tracy")]
        let g = ::tracy_client::Client::running().map(|c| c.span(::tracy_client::span_location!($name), 0));
        #[cfg(not(feature = "tracy"))]
        let g = ();
        g
    }};
}

/// A zone with a text (formatted only while the profiler is connected).
#[macro_export]
macro_rules! zone_text {
    ($name:literal, $($arg:tt)*) => {
        #[cfg(feature = "tracy")]
        let _tracy_zone = ::tracy_client::Client::running().map(|c| {
            let s = c.span(::tracy_client::span_location!($name), 0);
            if c.is_connected() {
                s.emit_text(&format!($($arg)*));
            }
            s
        });
    };
}

/// A value plotted over time.
#[macro_export]
macro_rules! plot {
    ($name:literal, $value:expr) => {
        #[cfg(feature = "tracy")]
        if let Some(c) = ::tracy_client::Client::running() {
            c.plot(::tracy_client::plot_name!($name), $value as f64);
        }
    };
}
