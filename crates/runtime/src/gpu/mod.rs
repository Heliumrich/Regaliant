//! GPU emulation front end for the PS5 (AGC): submission queues, the gfx10
//! PM4 command processor (`cp`) and flips.
//!
//! Command buffers submitted through libSceAgcDriver are queued per hardware
//! queue (graphics, or a compute queue id) and executed in submission order
//! by one command-processor thread per queue, concurrently with the CPU, as
//! on the console. Dispatches run on Vulkan with `DS_GPU=1` (`exec`); draws
//! are decoded but not rendered yet (docs/STATUS.md).

pub mod capture;
pub mod cp;
pub mod desc;
pub mod emucheck;
pub mod frameahead;
pub mod frameprobe;
pub mod hitch;
pub mod keep;
pub mod runprobe;
pub mod smalltrack;
/// Run-time switches for A/B measurements within one run (`tune <name>
/// <value>`; slowly varying load from other GPU users cancels out), each
/// seeded from its environment variable.
pub mod tune {
    use std::sync::atomic::{AtomicU64, Ordering};

    struct Tunable {
        name: &'static str,
        env: &'static str,
        default: u64,
        value: AtomicU64,
    }

    const UNSET: u64 = u64::MAX;

    static ALL: [Tunable; 187] = [
        Tunable { name: "promises", env: "DS_PROMISES", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "lazy", env: "DS_LAZY_BARRIERS", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "shadow_min", env: "DS_GPU_SHADOW_MIN", default: 64 << 10, value: AtomicU64::new(UNSET) },
        Tunable { name: "mirrors", env: "DS_MIRRORS", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "ring_vram", env: "DS_RING_VRAM", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "snap_max", env: "DS_SNAP_MAX", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "snap_index", env: "DS_SNAP_INDEX", default: 4 << 20, value: AtomicU64::new(UNSET) },
        Tunable { name: "ring_lo", env: "DS_RING_LO", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "ring_hi", env: "DS_RING_HI", default: u64::MAX >> 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "shadow_keep", env: "DS_SHADOW_KEEP", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "block_chain", env: "DS_BLOCK_CHAIN", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "gpu_indirect", env: "DS_GPU_INDIRECT", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "lazy_wait_submit", env: "DS_LAZY_WAIT_SUBMIT", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "gpu_cp_writes", env: "DS_GPU_CP_WRITES", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "slots", env: "DS_GPU_SLOTS", default: 4, value: AtomicU64::new(UNSET) },
        Tunable { name: "lod_stats", env: "DS_LOD_STATS", default: 3, value: AtomicU64::new(UNSET) },
        Tunable { name: "cmp_emulate", env: "DS_CMP_EMULATE", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "tex_waits", env: "DS_TEX_WAITS", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "shadow_barrier", env: "DS_SHADOW_BARRIER", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "pre_max", env: "DS_PRE_MAX", default: 8 << 20, value: AtomicU64::new(UNSET) },
        Tunable { name: "scratch_writes", env: "DS_SCRATCH_WRITES", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "trace_sync", env: "DS_TRACE_SYNC_N", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "homes", env: "DS_HOMES", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "gpu_loads", env: "DS_GPU_LOADS", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "reg_vars", env: "DS_REG_VARS", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "prep_cache", env: "DS_PREP_CACHE", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "map_ranges", env: "DS_MAP_RANGES", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "ahead", env: "DS_AHEAD", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "pre_cs", env: "DS_PRE_CS", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "par", env: "DS_PAR", default: 1000, value: AtomicU64::new(UNSET) },
        Tunable { name: "pre_share", env: "DS_PRE_SHARE", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "pending_set", env: "DS_PENDING_SET", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "tex_scrub", env: "DS_TEX_SCRUB", default: 2, value: AtomicU64::new(UNSET) },
        Tunable { name: "pre_share_min", env: "DS_PRE_SHARE_MIN", default: 16384, value: AtomicU64::new(UNSET) },
        Tunable { name: "pre_share_mib", env: "DS_PRE_SHARE_MIB", default: 1024, value: AtomicU64::new(UNSET) },
        Tunable { name: "pre_share_volatile", env: "DS_PRE_SHARE_VOLATILE", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "spec_reads", env: "DS_SPEC_READS", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "submit_thread", env: "DS_SUBMIT_THREAD", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "par_balance", env: "DS_PAR_BALANCE", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "pre_inflight", env: "DS_PRE_INFLIGHT", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "idle_split", env: "DS_IDLE_SPLIT", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "push_desc", env: "DS_PUSH_DESC", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "pre_merge", env: "DS_PRE_MERGE", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "lean_regs", env: "DS_LEAN_REGS", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "span_cache", env: "DS_SPAN_CACHE", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "fill_dma", env: "DS_FILL_DMA", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "lean_cmd", env: "DS_LEAN_CMD", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "exact_big", env: "DS_EXACT_BIG", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "lean_notes", env: "DS_LEAN_NOTES", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "pre_keep", env: "DS_PRE_KEEP", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "pre_big_max", env: "DS_PRE_BIG_MAX", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "pre_big_mib", env: "DS_PRE_BIG_MIB", default: 1024, value: AtomicU64::new(UNSET) },
        Tunable { name: "home_reads", env: "DS_HOME_READS", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "shadow_share", env: "DS_SHADOW_SHARE", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "clean_pages", env: "DS_CLEAN_PAGES", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "eop_light", env: "DS_EOP_LIGHT", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "ts_nowait", env: "DS_TS_NOWAIT", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "upload_parts", env: "DS_UPLOAD_PARTS", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "upload_async", env: "DS_UPLOAD_ASYNC", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "cp_priority", env: "DS_CP_PRIORITY", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "cp_cpus", env: "DS_CP_CPUS", default: u64::MAX - 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "guest_cpus", env: "DS_GUEST_CPUS", default: u64::MAX - 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "lazy_passes", env: "DS_LAZY_PASSES", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "push_partial", env: "DS_PUSH_PARTIAL", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "per_lane", env: "DS_PER_LANE", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "copy_queue", env: "DS_COPY_QUEUE_USE", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "skip_read_waits", env: "DS_SKIP_READ_WAITS", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "par_split", env: "DS_PAR_SPLIT", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "par_weight", env: "DS_PAR_WEIGHT", default: 500, value: AtomicU64::new(UNSET) },
        Tunable { name: "alt_queue", env: "DS_ALT_QUEUE", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "skip_barriers", env: "DS_SKIP_BARRIERS", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "exact_indirect", env: "DS_EXACT_INDIRECT", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "merge_barriers", env: "DS_MERGE_BARRIERS", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "feedback_exact", env: "DS_FEEDBACK_EXACT", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "cp_classes", env: "DS_CP_CLASSES", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "gpu_loads_sel", env: "DS_GPU_LOADS_SEL", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "race_checks", env: "DS_RACE_CHECKS", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "data_snapshots", env: "DS_DATA_SNAPSHOTS", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "par_main_writes", env: "DS_PAR_MAIN_WRITES", default: 2, value: AtomicU64::new(UNSET) },
        Tunable { name: "forward_copies", env: "DS_FORWARD_COPIES", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "label_merge", env: "DS_LABEL_MERGE", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "sync_cs", env: "DS_SYNC_CS_ON", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "par_compute", env: "DS_PAR_COMPUTE", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "par_min", env: "DS_PAR_MIN", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "label_early", env: "DS_LABEL_EARLY", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "par_events", env: "DS_PAR_EVENTS", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "par_min_compute", env: "DS_PAR_MIN_COMPUTE", default: 192, value: AtomicU64::new(UNSET) },
        Tunable { name: "chain_memo", env: "DS_CHAIN_MEMO", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "wait_yields", env: "DS_WAIT_YIELDS", default: 800, value: AtomicU64::new(UNSET) },
        Tunable { name: "timer_sleep", env: "DS_TIMER_SLEEP", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "cp_classes_acb", env: "DS_CP_CLASSES_ACB", default: 0x50, value: AtomicU64::new(UNSET) },
        Tunable { name: "shadow_args", env: "DS_SHADOW_ARGS", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "ahead_queues", env: "DS_AHEAD_QUEUES", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "copy_update", env: "DS_COPY_UPDATE", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "skip_copies", env: "DS_SKIP_COPIES", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "draw_args_gpu", env: "DS_DRAW_ARGS_GPU", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "zero_fill_clears", env: "DS_ZERO_FILL_CLEARS", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "epoch_gpu_waits", env: "DS_EPOCH_GPU_WAITS", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "flip_async", env: "DS_FLIP_ASYNC", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "wait_ahead", env: "DS_WAIT_AHEAD", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "marks_unlocked", env: "DS_MARKS_UNLOCKED", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "lock_relief", env: "DS_LOCK_RELIEF", default: 3, value: AtomicU64::new(UNSET) },
        Tunable { name: "prep_pool", env: "DS_PREP_POOL", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "ts_batch", env: "DS_TS_BATCH", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "ww_probe", env: "DS_WW_PROBE", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "cmdbuf_merge", env: "DS_CMDBUF_MERGE", default: 2, value: AtomicU64::new(UNSET) },
        Tunable { name: "offthread_reset", env: "DS_OFFTHREAD_RESET", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "label_wake", env: "DS_LABEL_WAKE", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "par_spins", env: "DS_PAR_SPINS", default: 512, value: AtomicU64::new(UNSET) },
        Tunable { name: "slots_gfx", env: "DS_GPU_SLOTS_GFX", default: 4, value: AtomicU64::new(UNSET) },
        Tunable { name: "split_every", env: "DS_SPLIT_EVERY", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "xpre_dry", env: "DS_XPRE_DRY", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "par_cost", env: "DS_PAR_COST", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "submit_finish", env: "DS_SUBMIT_FINISH", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "plan_check", env: "DS_PLAN_CHECK", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "homes_watch", env: "DS_HOMES_WATCH", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "unbounded_near", env: "DS_UNBOUNDED_NEAR", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "image_homes", env: "DS_IMAGE_HOMES", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "par_prewake", env: "DS_PAR_PREWAKE", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "attachless_draws", env: "DS_ATTACHLESS_DRAWS", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "block_tails", env: "DS_BLOCK_TAILS", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "mip_views", env: "DS_MIP_VIEWS", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "compile_ahead", env: "DS_COMPILE_AHEAD", default: COMPILE_AHEAD_AUTO, value: AtomicU64::new(UNSET) },
        Tunable { name: "lane_guards", env: "DS_LANE_GUARDS", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "compile_gfx", env: "DS_COMPILE_GFX", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "label_latency", env: "DS_LABEL_LATENCY", default: 2000, value: AtomicU64::new(UNSET) },
        Tunable { name: "precompile", env: "DS_PRECOMPILE", default: 2, value: AtomicU64::new(UNSET) },
        Tunable { name: "precompile_workers", env: "DS_PRECOMPILE_WORKERS", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "depth_views", env: "DS_DEPTH_VIEWS", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "pre_trust", env: "DS_PRE_TRUST", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "draw_repeat", env: "DS_DRAW_REPEAT", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "cpu_args_base", env: "DS_CPU_ARGS_BASE", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "sgpr_pc", env: "DS_SGPR_PC", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "ww_gate", env: "DS_WW_GATE", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "fill_skip", env: "DS_FILL_SKIP", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "image_evict", env: "DS_IMAGE_EVICT", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "vram_high", env: "DS_VRAM_HIGH", default: 85, value: AtomicU64::new(UNSET) },
        Tunable { name: "vram_low", env: "DS_VRAM_LOW", default: 75, value: AtomicU64::new(UNSET) },
        Tunable { name: "pre_window", env: "DS_PRE_WINDOW", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "home_submit_epochs", env: "DS_HOME_SUBMIT_EPOCHS", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "gfx_core", env: "DS_GFX_CORE", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "skip_cmds", env: "DS_SKIP_CMDS", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "xwait_skip", env: "DS_XWAIT_SKIP", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "addresses", env: "DS_ADDRESSES", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "batch_cs", env: "DS_BATCH_CS", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "job_idle", env: "DS_JOB_IDLE", default: 50, value: AtomicU64::new(UNSET) },
        Tunable { name: "job_sleep", env: "DS_JOB_SLEEP", default: 200, value: AtomicU64::new(UNSET) },
        Tunable { name: "job_watch", env: "DS_JOB_WATCH", default: 2000, value: AtomicU64::new(UNSET) },
        Tunable { name: "draw_again", env: "DS_DRAW_AGAIN", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "batch_draws", env: "DS_BATCH_DRAWS", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "sgpr_ubo", env: "DS_SGPR_UBO", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "merge_many", env: "DS_MERGE_MANY", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "power_optout", env: "DS_POWER_OPTOUT", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "flip_grid", env: "DS_FLIP_GRID", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "present_fit", env: "DS_PRESENT_FIT", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "frame_cap", env: "DS_FRAME_CAP", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "scrub_ms", env: "DS_SCRUB_MS", default: 10, value: AtomicU64::new(UNSET) },
        Tunable { name: "skip_steps", env: "DS_SKIP_STEPS", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "stencil_export", env: "DS_STENCIL_EXPORT", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "shadow_share_min", env: "DS_SHADOW_SHARE_MIN", default: 16384, value: AtomicU64::new(UNSET) },
        Tunable { name: "keep_small", env: "DS_KEEP_SMALL", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "frame_ahead", env: "DS_FRAME_AHEAD", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "past_runs", env: "DS_PAST_RUNS", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "past_drop", env: "DS_PAST_DROP", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "acm_reverb", env: "DS_ACM_REVERB", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "trace_waits", env: "DS_TRACE_WAITS_N", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "plan_fast", env: "DS_PLAN_FAST", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "async_particles", env: "DS_ASYNC_PARTICLES", default: 2, value: AtomicU64::new(UNSET) },
        Tunable { name: "shadow_pre", env: "DS_SHADOW_PRE", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "home_read_min", env: "DS_HOME_READ_MIN", default: 1 << 20, value: AtomicU64::new(UNSET) },
        Tunable { name: "shadow_max", env: "DS_SHADOW_MAX", default: 4 << 20, value: AtomicU64::new(UNSET) },
        Tunable { name: "acb_cpus", env: "DS_ACB_CPUS", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "host_census", env: "DS_HOST_CENSUS", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "pre_protect", env: "DS_PRE_PROTECT", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "home_cpu_check", env: "DS_HOME_CPU_CHECK", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "placement", env: "DS_PLACEMENT", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "acb50_cpus", env: "DS_ACB50_CPUS", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "tex_exact", env: "DS_TEX_EXACT", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "home_check_unlocked", env: "DS_HOME_CHECK_UNLOCKED", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "pre_unsized", env: "DS_PRE_UNSIZED", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "home_check_win", env: "DS_HOME_CHECK_WIN", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "gpu_chains", env: "DS_GPU_CHAINS", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "gpu_mirror", env: "DS_GPU_MIRROR", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "past_waits", env: "DS_PAST_WAITS", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "forward_earlier", env: "DS_FORWARD_EARLIER", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "forward_held", env: "DS_FORWARD_HELD", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "past_backoff", env: "DS_PAST_BACKOFF", default: 16, value: AtomicU64::new(UNSET) },
    ];

    /// A switch's value: one relaxed load once set (inlined into every
    /// caller; the per-command paths read dozens of switches).
    #[inline(always)]
    fn get(i: usize) -> u64 {
        let v = ALL[i].value.load(Ordering::Relaxed);
        if v != UNSET {
            return v;
        }
        first_get(i)
    }

    /// The first read of a switch: its environment variable or default.
    #[cold]
    #[inline(never)]
    fn first_get(i: usize) -> u64 {
        let t = &ALL[i];
        let v = std::env::var(t.env).ok().and_then(|x| x.trim().parse().ok()).unwrap_or(t.default);
        t.value.store(v, Ordering::Relaxed);
        v
    }

    /// Label promises (`Cp::wait`).
    #[inline]
    pub fn promises() -> bool {
        get(0) != 0
    }

    /// Dispatches ordered by the title's synchronization only.
    #[inline]
    pub fn lazy_barriers() -> bool {
        get(1) != 0
    }

    /// Written buffers below this size skip VRAM shadows (unless atomic).
    #[inline]
    pub fn shadow_min() -> u64 {
        get(2)
    }

    /// VRAM mirrors of GPU-written buffers (`gpu::mirror`).
    #[inline]
    pub fn mirrors() -> bool {
        get(3) != 0
    }

    /// Which stages read their SGPR data from video memory instead of system
    /// memory (diagnostic: it changes timing, not results): 0 none, 1 all,
    /// 2 the graphics queue's, 3 the compute queues', 4 shaders at addresses
    /// in [`ring_lo`, `ring_hi`).
    #[inline]
    pub fn ring_vram() -> u64 {
        get(4)
    }

    #[inline]
    pub fn ring_range() -> std::ops::Range<u64> {
        get(7)..get(8)
    }

    /// Read-only buffer bindings up to this size are snapshotted into video
    /// memory when recorded (0: off).
    #[inline]
    pub fn snap_max() -> u64 {
        get(5)
    }

    /// Index buffers up to this size are snapshotted into video memory when
    /// recorded (0: off).
    #[inline]
    pub fn snap_index() -> u64 {
        get(6)
    }

    /// VRAM shadows stay resident for the rest of their batch (merged back
    /// once) instead of being merged after each dispatch.
    #[inline]
    pub fn shadow_keep() -> bool {
        get(9) != 0
    }

    /// Structured control flow instead of the translator's dispatcher: 1 for
    /// every shader, 2 for loop-free ones only, 0 never; switching
    /// translates shaders anew.
    #[inline]
    pub fn block_chain() -> u64 {
        get(10)
    }

    /// Read-only buffer bindings and index buffers up to this size are read
    /// from a copy in video memory that the GPU makes at the start of their
    /// batch (`Exec::pre_copy`; 0: off) instead of across PCIe (vertex data,
    /// the per-item culls' meshes): draws took ~45% less GPU time and the
    /// courtyard ran ~10% faster (in-run A/B, run 20261011-063258). 8 MiB
    /// since 2026-10-12: the foliage draws' 6.8 MB instance buffer was read in
    /// place (~9 ms of draw time per courtyard frame); GPU time per frame
    /// ~11% lower in each of three A/B cycles, rate median +3.9% (run
    /// 20261012-132127; 16 MiB was worse earlier).
    #[inline]
    pub fn pre_max() -> u64 {
        get(19)
    }

    /// Diagnosis only (1; breaks rendering): written buffer bindings too large
    /// for a VRAM shadow store into a video-memory scratch buffer and their
    /// results are lost. Measures what GPU stores to imported host memory
    /// cost (the per-item culls' index output).
    #[inline]
    pub fn scratch_writes() -> bool {
        get(20) != 0
    }

    /// Diagnosis only (2; breaks rendering): every written buffer binding
    /// stores into the scratch buffer, none shadowed, merged or written in
    /// guest memory: the upper bound of keeping GPU-written data in video
    /// memory (residency design stage 3), its readers left as they are.
    #[inline]
    pub fn scratch_all() -> bool {
        get(20) == 2
    }

    /// A full barrier after every dispatch that binds a VRAM shadow (1, as
    /// when shadows were merged back after each dispatch), or only the
    /// barriers the title's synchronization asks for (0; with lazy barriers;
    /// merges have their own). Consecutive per-item culls share the shadow
    /// of one atomic argument buffer: with 1 they cannot overlap.
    #[inline]
    pub fn shadow_barrier() -> bool {
        get(18) != 0
    }

    /// Which textures a draw or dispatch waits for GPU writes to (their guest
    /// memory): 1 those a coarse bound finds written, then only the ones read
    /// from guest memory now, with their exact span (`Exec::sync_textures`);
    /// 0 every texture with the coarse bound of 16 bytes per texel (A/B).
    #[inline]
    pub fn tex_waits() -> u64 {
        get(17)
    }

    /// Indirect dispatches whose arguments GPU work still writes read them on
    /// the GPU (else the command processor waits for that work first).
    #[inline]
    pub fn gpu_indirect() -> bool {
        get(11) != 0
    }

    /// A label wait submits the recorded work only when it has to poll (a wait
    /// that passes at once does not end the batch).
    #[inline]
    pub fn lazy_wait_submit() -> bool {
        get(12) != 0
    }

    /// On by default since 2026-10-12: CP memory writes that follow earlier GPU work or
    /// earlier deferred CP writes (WRITE_DATA after a sync event, RELEASE_MEM
    /// values, end-of-pipe timestamps) are performed by the GPU in order with
    /// it instead of by the CPU after the batch completed
    /// (`Exec::gpu_cp_write`, `Exec::gpu_timestamp`). Their promises
    /// are registered only when the batch is submitted, so waits pass later
    /// (and on the latest value only). A batch holding such writes is
    /// submitted only once the CPU performed the queue's earlier deferred
    /// writes (they keep their order); until 2026-10-12 writes behind those
    /// took the CPU path, ~90% of them in the courtyard (run 20261012-040642).
    #[inline]
    pub fn gpu_cp_writes() -> bool {
        // Mirror mode too: their pages become host class, imported exactly
        // (`resident::host_for_cp`).
        get(13) != 0
    }

    /// Batch slots a queue cycles through (`exec::MAX_SLOTS` at most): it
    /// records one batch while the batches of the others execute.
    #[inline]
    pub fn slots() -> u64 {
        get(14)
    }

    /// GetLodStats (hardware LOD statistics, the title's texture streaming
    /// feedback; `lodstats`): 0 leaves the buffer as the title wrote it (no
    /// streaming of high mips), 1 reports every counter bank as sampled at
    /// mip 0, 2 the banks of textures bound since the last report (at mip
    /// 0), 3 what the sampling instructions measured on the GPU (finest mip
    /// and lane count per bank; translations change with it).
    #[inline]
    pub fn lod_stats() -> u64 {
        get(15)
    }

    /// Depth-comparison sampling of color images is emulated in the
    /// translation (Vulkan compares only on depth formats); 0 compares in
    /// hardware everywhere (undefined on color formats; A/B).
    #[inline]
    pub fn cmp_emulate() -> bool {
        get(16) != 0
    }

    /// Index buffers that GPU work wrote through a binding too large for a
    /// VRAM shadow get a home in video memory that later GPU uses of the
    /// range bind (`gpu::homes`): 3.29 against 2.26 flips/s in the
    /// courtyard (in-run A/B, run 20261011-072603). 0 writes the homes back
    /// and drops them.
    #[inline]
    pub fn homes() -> bool {
        get(22) != 0
    }

    /// Scalar loads that only produce data are read by the shader at run
    /// time (`rdna::analysis::analyze_opts`) instead of being resolved on the
    /// CPU when the command is recorded: GPU work that writes them shortly
    /// before then costs no wait. Switching translates programs anew.
    #[inline]
    pub fn gpu_loads() -> bool {
        get(23) != 0
    }

    /// `gpu_loads` mode: 1 every program, 2 all but programs translated per
    /// lane (diagnosis), 3 only stages whose data-only loads GPU work still
    /// writes (prepared again when the command processor would wait).
    #[inline]
    pub fn gpu_loads_mode() -> u64 {
        get(23)
    }

    /// Translations keep each register in its own variable (1) or the
    /// register files in arrays (0, the earlier form): see
    /// `rdna::translate::Spec::reg_vars`. Switching translates programs anew.
    #[inline]
    pub fn reg_vars() -> bool {
        get(24) != 0
    }

    /// Vertex and pixel shader preparations reuse the previous one of their
    /// stage when inputs and the guest words read are unchanged (1): the
    /// copies share the unchanging parts, checked in place (3.5-3.7% less
    /// CPU per command in replay A/Bs, run 20261013-063425; whole clones
    /// cost more than they saved before, run 20261011-092437).
    /// DS_PREP_CHECK=1 compares every reuse with a full preparation.
    #[inline]
    pub fn prep_cache() -> bool {
        get(25) != 0
    }

    /// `prep_cache`'s mode: 1 the latest preparation per stage, 2 a frame's
    /// (faster in replayed frames, slower live: FINDINGS.md 2026-10-13).
    #[inline]
    pub fn prep_cache_mode() -> u64 {
        get(25)
    }

    /// The per-thread caches of readable guest ranges (`mem::readable_span`)
    /// keep the runs a mapping change did not touch; 0 empties them at every
    /// change (the earlier behavior, for A/B).
    #[inline]
    pub fn map_ranges() -> bool {
        get(26) != 0
    }

    /// Worker threads that prepare the stages of synchronization-free runs
    /// ahead of execution (`gpu::ahead`); 0 (default since 2026-10-12)
    /// prepares every stage on the command processor that records it: with
    /// parallel recording, 4 workers cost ~100 ms of CPU per courtyard frame
    /// and saved no frame time (FINDINGS: stage preparation ahead off).
    #[inline]
    pub fn ahead() -> u64 {
        get(27)
    }

    /// Read-only buffer bindings of dispatches copied into video memory at
    /// the batch's start like those of draws (`Exec::pre_copy`); 0: compute
    /// shaders read them from guest memory in place.
    #[inline]
    pub fn pre_cs() -> bool {
        get(28) != 0
    }

    /// Worker command processors that record chunks of long
    /// synchronization-free runs in parallel (`gpu::par`); 0: off; 1000 (the
    /// default): half the logical CPUs, 2 to 8 (`par::workers`).
    #[inline]
    pub fn par() -> u64 {
        get(29)
    }

    /// Batch-start copies of the graphics queue shared across its batches
    /// (`gpu::precache`); 0: each batch copies what it binds.
    #[inline]
    pub fn pre_share() -> bool {
        get(30) != 0
    }

    /// Draws check pending fast clears and overwrites of their images in a
    /// published set (no image-cache lock); 0: under the lock (A/B).
    #[inline]
    pub fn pending_set() -> bool {
        get(31) != 0
    }

    /// Uploaded textures checked for CPU writes by a background thread
    /// (`images::scrubbing`): 2 by their write-watch marks (exact; the pages
    /// go to the write tracker, and re-uploads leave out parts whose bytes
    /// are unchanged: the title rewrites texture memory with the same data),
    /// 1 by sampled hashes (the thread's CPU 5.7-6.4 against 2.3-2.5 ms per
    /// frame, CPU per command +0.4 to +4.4%, run 20261013-083150); 0: by the
    /// first draw binding each per frame (sampled).
    #[inline]
    pub fn tex_scrub() -> bool {
        get(32) != 0
    }

    /// `tex_scrub`'s mode.
    #[inline]
    pub fn tex_scrub_mode() -> u64 {
        get(32)
    }

    /// Smallest batch-start copy kept in the shared video-memory chunks
    /// (`gpu::precache`), bytes.
    #[inline]
    pub fn pre_share_min() -> u64 {
        get(33)
    }

    /// Video memory for shared batch-start copies, MiB.
    #[inline]
    pub fn pre_share_mib() -> u64 {
        get(34)
    }

    /// Frames a range GPU work wrote is not kept in the shared chunks
    /// (copied per batch instead); 0: always kept.
    #[inline]
    pub fn pre_share_volatile() -> u64 {
        get(35)
    }

    /// Stages whose preparation read what submitted GPU work still writes
    /// are used as prepared and checked afterwards (`exec::speculate`): 1
    /// data-only loads copied into their SGPR data by the batch prologue and
    /// descriptor words used as read; 2 only the data-only loads (stages
    /// whose GPU-written words include a descriptor's wait as before).
    #[inline]
    pub fn spec_reads() -> bool {
        get(36) != 0
    }

    #[inline]
    pub fn spec_reads_mode() -> u64 {
        get(36)
    }

    /// Batches are submitted to Vulkan by a thread of their own
    /// (`exec::push_batch`) instead of by the command processors.
    #[inline]
    pub fn submit_thread() -> bool {
        get(37) != 0
    }

    /// The main command processor's share of a split run follows whether it
    /// waited for the workers (`par::main_weight`); 0: equal chunks.
    #[inline]
    pub fn par_balance() -> bool {
        get(38) != 0
    }

    /// A batch's prologue copies are recorded when the prologue ends, one
    /// vkCmdCopyBuffer per source and destination buffer with all their
    /// regions (0: one call per copy as it is made). ~5.6x fewer calls in
    /// the courtyard; default since 2026-10-16: -1.2% CPU per command at the
    /// gate (four cycles 0.974-0.995), flips/s 1.010 (FINDINGS.md).
    #[inline]
    pub fn pre_merge() -> bool {
        get(42) != 0
    }

    /// Batch-start copies of ranges that submitted batches of the same queue
    /// still write (the prologue runs after them); 0: bound in place.
    #[inline]
    pub fn pre_inflight() -> bool {
        get(39) != 0
    }

    /// A command processor's batch is submitted once it has been recording
    /// this many microseconds while the GPU has no work left (0: off).
    #[inline]
    pub fn idle_split() -> u64 {
        get(40)
    }

    /// Pipelines created from now on take their descriptors pushed into the
    /// command buffer (VK_KHR_push_descriptor) when their set layout fits
    /// maxPushDescriptors, instead of a descriptor set allocated, written
    /// and bound per command; pipelines are cached per setting (an in-run
    /// A/B compiles both).
    #[inline]
    pub fn push_desc() -> bool {
        get(41) != 0
    }

    /// Command-processor waits for GPU work still to be traced (`tune
    /// trace_sync <n>`: the next n are logged with the read that collided,
    /// its writer and the shader).
    #[inline]
    pub fn get_trace_sync() -> bool {
        get(21) > 0
    }

    /// Takes one of the waits still to be traced.
    #[inline]
    pub fn take_trace_sync() -> bool {
        get(21) > 0 && ALL[21].value.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| if v != UNSET && v > 0 { Some(v - 1) } else { None }).is_ok()
    }

    /// Register-state snapshots of parallel chunks (and replays) hold only
    /// the registers written so far (`cp::HIGH`); 0 copies whole register
    /// files (100 KiB each), for A/B.
    #[inline]
    pub fn lean_regs() -> bool {
        get(43) != 0
    }

    /// The shader-input reader keeps the mappings of its last four reads and
    /// looks a miss up once; 0: the last read's mapping, two lookups per miss
    /// (the earlier behavior, for A/B).
    #[inline]
    pub fn span_cache() -> bool {
        get(44) != 0
    }

    /// Dispatches of the title's fill kernel over a whole range are buffer
    /// fills of guest memory in place (`verify::full_fill`), without the
    /// kernel and its VRAM shadow; 0: the kernel runs (for A/B).
    #[inline]
    pub fn fill_dma() -> bool {
        get(45) != 0
    }

    /// Per-command steps without needless work: the main command processor
    /// spins on a flag for its workers, and a compute pipeline already bound
    /// in the command buffer is not bound again; 0: as before (for A/B).
    #[inline]
    pub fn lean_cmd() -> bool {
        get(46) != 0
    }

    /// Range unions keep ranges over 64 pages exactly beside their page
    /// filter (`ranges::Ranges`); 0: any such range switches the filter off
    /// (the earlier behavior, for A/B).
    #[inline]
    pub fn exact_big() -> bool {
        get(47) != 0
    }

    /// A dispatch writing the same ranges as its executor's previous command
    /// skips the global write notes (image cache, write tracker, copy cache)
    /// while their generations are unchanged (`Exec::last_notes`); 0: every
    /// dispatch notes them (for A/B).
    #[inline]
    pub fn lean_notes() -> bool {
        get(48) != 0
    }

    /// The shared batch-start copies evict the oldest chunk no batch binds
    /// (not the oldest, often bound by every frame's batches), and a range
    /// another batch's unsubmitted prologue copies is copied privately
    /// instead of replacing that entry; 0: as before (for A/B).
    #[inline]
    pub fn pre_keep() -> bool {
        get(49) != 0
    }

    /// Read-only buffer bindings over `pre_max` and up to this many bytes are
    /// copied into a shared video-memory slot of their own (`precache`,
    /// within `pre_big_mib`) instead of being read in place across PCIe.
    /// Default 0 (off): in the courtyard such ranges are GPU-written every
    /// frame (their copies were dropped) and the replay A/B showed no gain
    /// (GPU time 1.002 and 1.014, run 20261013-025441).
    #[inline]
    pub fn pre_big_max() -> u64 {
        get(50)
    }

    /// Video memory for those copies (MiB).
    #[inline]
    pub fn pre_big_mib() -> u64 {
        get(51)
    }

    /// A read-only buffer binding of `home_read_min` bytes or more inside a
    /// range GPU work writes through large bindings gives that range a
    /// video-memory home (`homes::read_use`), like an index buffer there
    /// does. Default 1 since 2026-10-26 (with `home_read_min` 1 MiB): the
    /// vista +3% (GPU -2.5 ms a frame), a 1-1 walk at 59.94 Hz +2% (GPU p50
    /// -1.5 ms), once texture syncs stopped splitting boots into two speeds
    /// (FINDINGS.md 2026-10-26). In the courtyard of 2026-10-13 it had made
    /// five more homes (427 MiB) without changing their passes' GPU times.
    #[inline]
    pub fn home_reads() -> bool {
        get(52) != 0
    }

    /// A written-buffer shadow's working copy becomes a shared copy of its
    /// window after its merge (`precache`), and the next shadow of the window
    /// is copied in from it instead of from guest memory across PCIe (and
    /// read-only bindings of the window bind it). On by default since
    /// 2026-10-24 night: the GPU's work -3 ms a frame in 1-1, +2.5% at the
    /// vista, tours clean and its self-check (`DS_SHADOW_SHARE_CHECK`) clean
    /// over 31,490 samples. The device losses of 2026-10-13 with it came from
    /// the timestamps' waiting copies (`tune ts_nowait`). Known gap: a write
    /// recorded by another queue after the lookup but executed before the
    /// copy in is not seen by it (the 2026-10-23 census of the gate's small
    /// windows found none merged by a second queue).
    #[inline]
    pub fn shadow_share() -> bool {
        get(53) != 0
    }

    /// The command processors' read checks skip reads in 64 KiB pages found
    /// clean (no pending or in-flight write, no stale home part) since the
    /// write generations last changed (`Exec::clean_pages`); 0: every read is
    /// checked (for A/B).
    #[inline]
    pub fn clean_pages() -> bool {
        get(54) != 0
    }

    /// End-of-pipe writes performed by the GPU (RELEASE_MEM labels and
    /// timestamps) follow the work before them without holding later work
    /// back: labels behind a barrier into transfers only, timestamps as
    /// queries stored at the batch's end or before a later CP write; a wait
    /// passing on such a label in the same batch orders the work after it.
    /// On by default since 2026-10-14 (+2.7% at the gate; the title's ~156
    /// timestamps per flip took ~14 us of GPU time each between full
    /// barriers). It had been off since World 5-1 device losses with it, which
    /// the timestamp copy that waited in the command stream caused (`tune
    /// ts_nowait`; FINDINGS.md 2026-10-13); tours pass with it since.
    #[inline]
    pub fn eop_light() -> bool {
        get(55) != 0
    }

    /// Timestamp queries are copied after a full barrier instead of by a copy
    /// that waits for their availability: the waiting copy stalled the GPU at
    /// level loads (device losses in 4 of 4 tours of the configurations that
    /// showed them, none with this; FINDINGS.md 2026-10-13). 0: the waiting
    /// copy (for A/B).
    #[inline]
    pub fn ts_nowait() -> bool {
        get(56) != 0
    }

    /// Texture re-uploads upload only the (mip, layer)s that the written
    /// ranges (mappings, file reads) or the CPU's writes since the last
    /// upload (write-watch marks, taken through the copy cache) overlap: the
    /// title streams texture arrays a layer at a time (1024x1024x64 BC7,
    /// 1024x1024x256), and every layer it mapped re-uploaded the whole array.
    /// Upload time -32% and the level load to the gate 18.2 -> 13.5 s (runs
    /// 20261013-073959 / -073718); DS_UPLOAD_CHECK=1 found no changed part
    /// left out (run -072814). 0: whole re-uploads (for A/B).
    #[inline]
    pub fn upload_parts() -> bool {
        get(57) != 0
    }

    /// Texture uploads are submitted without waiting for them (1; their
    /// staging buffers freed once the GPU is done with them): -2.3 s of
    /// upload time per gate run. Its import failures (thousands a session,
    /// device losses: FINDINGS.md 2026-10-13) came from the staging buffers
    /// it pooled; unpooled (2026-10-24) it fails 48 like the default, but a
    /// warm 1-1 walk showed no gain (upload hitches 29 against 40, all
    /// hitches 359 against 286): off.
    #[inline]
    pub fn upload_async() -> bool {
        get(58) != 0
    }

    /// Host priority of the GPU front end's threads (`prio`): 0 normal,
    /// 1 above normal (default since 2026-10-13: on 12 of the 5900X's 24
    /// logical CPUs 11.3-11.5 flips/s instead of 2.3, the same at 24; run
    /// 20261013-124535), 2 highest, 3 above normal for the graphics command
    /// processor only (the others at normal: busy front-end threads above
    /// the title's starve its jobs, and World 8-3 loads into a different
    /// state; FINDINGS.md 2026-10-14).
    #[inline]
    pub fn cp_priority() -> u64 {
        get(59)
    }

    /// Logical CPUs the GPU front end's threads may run on (`prio`; a
    /// decimal mask, 0 = all of the process's; default `prio::CPUS_AUTO`:
    /// one last-level-cache group when there are several, `prio::auto_cpus`).
    #[inline]
    pub fn cp_cpus() -> u64 {
        get(60)
    }

    /// Logical CPUs the title's threads may run on (`thread::apply_guest_cpus`;
    /// a decimal mask, 0 = all of the process's; AUTO, the default: the
    /// CPUs the front end leaves, `prio::auto_guest_cpus`).
    #[inline]
    pub fn guest_cpus() -> u64 {
        match get(61) {
            super::prio::CPUS_AUTO => super::prio::auto_guest_cpus(),
            m => m,
        }
    }

    /// A rendering instance ends without a full barrier: later commands that
    /// use its attachments' contents follow the title's own synchronization
    /// (flushes and partial flushes arm the lazy barrier, as for dispatches),
    /// and draws after dispatches wait for them as before.
    #[inline]
    pub fn lazy_passes() -> bool {
        get(62) != 0
    }

    /// A command with the same pipeline layout as the previous push and no
    /// pipeline bound since pushes only the descriptors that changed
    /// (`exec::bind_descriptors`).
    #[inline]
    pub fn push_partial() -> bool {
        get(63) != 0
    }

    /// The listed compute programs are translated per lane
    /// (`shaders::PER_LANE_PROGRAMS`, `Spec::per_lane`): 1 all, 2 those
    /// without subgroup-decided EXEC branches (`Spec::lane_votes`), 0 none;
    /// switching translates them anew.
    #[inline]
    pub fn per_lane() -> u64 {
        get(64)
    }

    /// Batch-start copies of ranges no submitted batch writes run on the
    /// copy queue when there is one (`DS_COPY_QUEUE=0` at boot: none; 0
    /// here: all in the prologues on the main queue).
    #[inline]
    pub fn copy_queue() -> bool {
        get(65) != 0
    }

    /// Diagnosis (timing only, wrong results): stage preparations do not
    /// wait for GPU work writing what they read (they use memory as it is).
    #[inline]
    pub fn skip_read_waits() -> bool {
        get(66) != 0
    }

    /// Chunks per recorder a parallel run is split into (`gpu::par`): the
    /// workers take them in order and submit each when its turn comes; 1:
    /// one each.
    #[inline]
    pub fn par_split() -> u64 {
        get(67)
    }

    /// The main command processor's chunk of a parallel run relative to a
    /// worker's, in 1/1000 (0: `par_balance` decides): half a worker's, as it
    /// records the cull run slower per command than they do (+1.0-1.4%,
    /// FINDINGS.md 2026-10-14).
    #[inline]
    pub fn par_weight() -> u64 {
        get(68)
    }

    /// Diagnosis (timing only, no ordering between the queues): the batches
    /// of the compute queue with this id go to a second Vulkan queue of the
    /// main family (0: none; bit 16: they still share batch-start copies).
    #[inline]
    pub fn alt_queue() -> u64 {
        get(69)
    }

    /// Diagnosis (timing only, wrong results): full barriers left out, by
    /// site: 1 after a render pass, 2 before an indirect dispatch, 4 around
    /// a CP memory write the GPU performs, 8 around a GPU timestamp store,
    /// 16 the title's synchronization (partial flushes, cache actions,
    /// end-of-pipe events) before the next command, 32 between dispatches
    /// and the draws after them, 64 a batch's opening barrier, 128 the
    /// prologue's barrier after earlier work, 256 the barrier before an
    /// end-of-pipe CP write (`eop_light`).
    #[inline]
    pub fn skip_barriers() -> u64 {
        get(70)
    }

    /// An indirect dispatch whose arguments the GPU reads waits behind a full
    /// barrier only when its batch wrote them since its last one (0: always).
    #[inline]
    pub fn exact_indirect() -> bool {
        get(71) != 0
    }

    /// A render pass ended right before a full barrier takes none of its own
    /// (0: one each).
    #[inline]
    pub fn merge_barriers() -> bool {
        get(72) != 0
    }

    /// A draw that samples an attachment of the current rendering instance
    /// ends it only when the instance writes that attachment (or the draw
    /// does, or stores to it); 0: whenever it samples one.
    #[inline]
    pub fn feedback_exact() -> bool {
        get(73) != 0
    }

    /// The graphics command processor's time by packet class, waits apart
    /// (profile report line; costs two clock reads per packet).
    #[inline]
    pub fn cp_classes() -> bool {
        get(74) != 0
    }

    /// Diagnosis: `gpu_loads` only for programs whose code hash ends in a bit
    /// pattern: (bits << 32) | pattern (0: every program). Bisects a program
    /// that renders differently with GPU-side loads.
    #[inline]
    pub fn gpu_loads_sel() -> u64 {
        get(75)
    }

    /// Diagnosis: read-only data the shaders read on the GPU (4 KiB or less
    /// per binding) is compared, once its batch completed, with what guest
    /// memory held when the command was recorded (`exec::check_races`).
    #[inline]
    pub fn race_checks() -> bool {
        get(76) != 0
    }

    /// The largest range the race checks follow: `tune race_checks 1` 4 KiB
    /// (words kept), a larger value that many bytes (ranges over 4 KiB are
    /// compared by hash).
    #[inline]
    pub fn race_max() -> u64 {
        match get(76) {
            0 => 0,
            1 => 4096,
            v => v,
        }
    }

    /// Data-only loads read on the GPU (`gpu_loads`) of data no GPU work
    /// still writes are bound to a snapshot taken when the command is
    /// recorded (0: guest memory as the batch executes).
    #[inline]
    pub fn data_snapshots() -> bool {
        get(77) != 0
    }

    /// Parallel workers reading guest memory that the main command
    /// processor's batch, not yet submitted when the run was planned,
    /// writes (they cannot see it and read the previous contents): 0
    /// counted only, 1 the chunk stops there (the main command processor
    /// records the rest), 2 (default) that batch is submitted before the
    /// workers start (they wait for it like for any submitted work), 3 it is
    /// submitted right after they were handed the run (a worker reading its
    /// writes first waits for that submission: the workers start ~55-65 us
    /// earlier per run).
    #[inline]
    pub fn par_main_writes() -> u64 {
        get(78)
    }

    /// The per-view copy kernel's result forwarded to the graphics queue's
    /// later reads of its destination instead of waiting for the copy
    /// (`gpu::forward`).
    #[inline]
    pub fn forward_copies() -> bool {
        get(79) != 0
    }

    /// A CP write the GPU performs in order (a label the title's CPU or
    /// another queue may watch) first merges the batch's resident shadows
    /// into guest memory: what earlier work wrote is there before the label.
    /// Default 0 (merged when the batch is submitted): ~316k label writes per
    /// gate run precede a merge, merging first cost ~2-3% and changed nothing
    /// seen; the title's CPU reading such data after the label would need it
    /// (FINDINGS.md 2026-10-14 evening).
    #[inline]
    pub fn label_merge() -> bool {
        get(80) != 0
    }

    /// Dispatches of the listed compute shaders (`exec::sync_cs`; by default
    /// two load-time kernels) are completed right after they are recorded
    /// (0: not; switchable while running).
    #[inline]
    pub fn sync_cs() -> bool {
        get(81) != 0
    }

    /// Parallel workers per compute queue's command processor for its
    /// synchronization-free runs (0: compute queues record alone).
    #[inline]
    pub fn par_compute() -> u64 {
        get(82)
    }

    /// The fewest commands a run is split into parallel chunks at (chunks
    /// at least a third of it, at least 16 commands); 0: automatic
    /// (`par::min_run`).
    #[inline]
    pub fn par_min() -> u64 {
        get(83)
    }

    /// A GPU-side label write recorded while another queue blocks on that
    /// label submits its batch at the next packet (0: when another queue
    /// asks before the write is recorded, or after `label_latency`).
    #[inline]
    pub fn label_early() -> bool {
        get(84) != 0
    }

    /// EVENT_WRITE and ACQUIRE_MEM (GPU-side flushes and cache actions:
    /// a barrier in the batch, later CP writes after the GPU work) stay
    /// inside parallel runs instead of ending them: bit 0 on compute
    /// queues, bit 1 on the graphics queue.
    #[inline]
    pub fn par_events() -> u64 {
        get(85)
    }

    /// `par_min` for the compute queues' runs.
    #[inline]
    pub fn par_min_compute() -> u64 {
        get(86)
    }

    /// Stage preparations resolve each of their plan's distinct pointer
    /// chains once (`Plan::chain_ids`; 0: looked up by hash per use).
    #[inline]
    pub fn chain_memo() -> bool {
        get(87) != 0
    }

    /// A command processor's polling wait (labels, REWIND) yields its
    /// time slice this many times after 200 spins before it sleeps 100 us
    /// between checks.
    #[inline]
    pub fn wait_yields() -> u64 {
        get(88)
    }

    /// Short sleeps (polling waits, the title's usleep and nanosleep) on a
    /// waitable timer each thread keeps (`thread::short_sleep`; 0: std's
    /// sleep, which makes and closes a timer per call).
    #[inline]
    pub fn timer_sleep() -> bool {
        get(89) != 0
    }

    /// The compute queue (its id, e.g. 0x50) whose command processor's time
    /// `cp_classes` also gives per packet opcode.
    #[inline]
    pub fn cp_classes_acb() -> u64 {
        get(90)
    }

    /// An indirect dispatch whose arguments a resident shadow holds (an
    /// earlier dispatch of the batch wrote them in its working copy) reads
    /// them there instead of merging the shadow back first: 1 on every
    /// executor, 2 on the graphics queue's (its command processor and
    /// parallel workers) only.
    #[inline]
    pub fn shadow_args() -> u64 {
        get(91)
    }

    /// The command processors whose synchronization-free runs the `ahead`
    /// workers prepare: 0 all, 1 the compute queues only (they record
    /// serially; the graphics queue's runs mostly go to parallel workers),
    /// 2 the graphics queue only.
    #[inline]
    pub fn ahead_queues() -> u64 {
        get(92)
    }

    /// The per-view copy kernel (`gpu::forward`) is recorded by the graphics
    /// command processor as a buffer update of the words it computed for it
    /// instead of a dispatch (no stage binding, SGPR data, descriptors or
    /// shader; the same place in the stream and the same lazy barriers).
    /// Off: during loads the words computed when recording sometimes differ
    /// from what the kernel leaves, and writing them broke meshes (World 4-2
    /// in 3 of 8 boots of the PGO build, FINDINGS.md 2026-10-15).
    #[inline]
    pub fn copy_update() -> bool {
        get(93) != 0
    }

    /// `copy_update 2`: the graphics queue's parallel workers record their
    /// per-view copies as buffer updates too (checked once their batches
    /// completed; not forwarded, as they record out of program order).
    #[inline]
    pub fn copy_update_workers() -> bool {
        get(93) >= 2
    }

    /// Diagnosis (timing only, wrong results): copies between guest memory
    /// and video memory left out, as if the data lived in video memory: 1
    /// the copies into new shadows (use only with 2: the merges would store
    /// garbage), 2 the merges of shadows back to guest memory (guest memory
    /// keeps older values), 4 the batch-start copies (read-only data), 8 the
    /// fill kernel's large zero fills stored in place (guest memory keeps
    /// older values; images they cover are still cleared), 16 / 32 / 64 the
    /// merges of windows of at least 1 MiB / 256 KiB to 1 MiB / under 256 KiB;
    /// windows under 256 KiB written back by a whole copy instead of the
    /// merge kernel (128; CPU-written words in them are lost) or merged into
    /// video memory instead of guest memory (256) or into host memory
    /// nothing reads (2048), into our own write-watched imported memory
    /// nothing reads at one place (4096) or at each window's guest address
    /// modulo 64 MiB (8192); merged into guest memory and then once more into
    /// guest memory (32768) or into the spread scratch (65536) (results
    /// correct: what the stores cost when later work reads the same data;
    /// the others leave GPU-written data in guest memory stale, which can
    /// change the title's GPU-driven work); 512 the title's timestamps stored into video
    /// memory; 1024 GPU-performed end-of-pipe label writes made twice.
    #[inline]
    pub fn skip_copies() -> u64 {
        get(94)
    }

    /// Indexed indirect draws whose arguments GPU work wrote through a VRAM
    /// shadow read them on the GPU from a video-memory home made for that
    /// buffer (the culls' DrawIndexIndirect records): no merge of the
    /// buffer back to guest memory after every batch, no copy in at the
    /// next, no wait for the culls before the draws are recorded.
    #[inline]
    pub fn draw_args_gpu() -> bool {
        get(95) != 0
    }

    /// GPU-owned images whose memory the fill kernel zeroes (recognized
    /// when recorded, `verify::full_fill`) are cleared right after the fill
    /// in its batch (`Images::zero_filled`), instead of their next user
    /// waiting for the GPU and scanning their bytes.
    #[inline]
    pub fn zero_fill_clears() -> bool {
        get(96) != 0
    }

    /// A label wait that passed on a GPU queue's write (a promise, a latched
    /// CP write) ends no copy-cache epoch (`precache::end_scope`): no CPU
    /// write released it. Waits on labels another queue's GPU work writes
    /// were taken for CPU-released ones when the value was not in memory yet
    /// (passed on its promise), ~300 epochs per frame, each making every
    /// shared batch-start copy check its pages for CPU writes again.
    #[inline]
    pub fn epoch_gpu_waits() -> bool {
        get(97) != 0
    }

    /// A GPU queue's FLIP packet is queued at once and shown after the GPU
    /// work before it (a deferred effect: `videoout::reserve_flip`,
    /// `commit_flip`) instead of the command processor waiting for that work
    /// (~0.7 ms per flip on the graphics command processor).
    #[inline]
    pub fn flip_async() -> bool {
        get(98) != 0
    }

    /// A command processor blocked on a label wait prepares the stages of
    /// the commands after it meanwhile (`Cp::scan_through`,
    /// `ahead::prepare_here`; compute queue 0x50 waits ~40 ms per frame on
    /// graphics labels, then records ~800 commands the graphics command
    /// processor waits for), checked when used as any preparation made
    /// ahead (`ahead::take`): 1 the compute queues' command processors, 2
    /// the graphics one too (it wakes later from its waits, ~0.3 ms).
    #[inline]
    pub fn wait_ahead_for(compute: bool) -> bool {
        match get(99) {
            0 => false,
            1 => compute,
            _ => true,
        }
    }

    /// The batch-start copy cache takes write-watch marks without holding
    /// its lock during the query (`precache::take_unlocked`; 0: under the
    /// lock, as before 2026-10-15).
    #[inline]
    pub fn marks_unlocked() -> bool {
        get(100) != 0
    }

    /// Bit 0: writing dispatches skip the image cache's lock when no page
    /// they write holds a GPU-owned image or fast-clear metadata
    /// (`images::may_note`); bit 1: the write logs' lookups share their lock
    /// (`exec::RangeLog`, a reader-writer lock; else taken exclusively).
    #[inline]
    pub fn note_filter() -> bool {
        get(101) & 1 != 0
    }

    #[inline]
    pub fn shared_log_reads() -> bool {
        get(101) & 2 != 0
    }

    /// Stage preparations take their vectors from a per-thread pool that
    /// finished preparations refill (`shaders::pool`; 0: allocate each).
    #[inline]
    pub fn prep_pool() -> bool {
        get(102) != 0
    }

    /// End-of-pipe timestamps stored per dispatch when a batch resolves them
    /// (`Exec::resolve_timestamps`, 1 to 14; 1: one dispatch each, as
    /// before 2026-10-15; 14 saved ~0.4 ms of GPU time per frame, and the
    /// first tour with it showed faults in 5-2 and 8-2 that did not recur:
    /// 1 until understood, FINDINGS.md).
    #[inline]
    pub fn ts_batch() -> u64 {
        get(103)
    }

    /// Write-mark takes query first without resetting (one page at most):
    /// a range found clean needs no resetting query (a query that resets
    /// nothing costs 0.4-0.8 us, the game's takes average 3.8 us; resets
    /// interrupt the process's other threads with TLB shootdowns). Live A/B
    /// 2026-10-15 night: +1.5% flips/s, -1% CPU per command; default 1.
    #[inline]
    pub fn ww_probe() -> bool {
        get(104) != 0
    }

    /// The graphics command processor keeps its batch open at the end of a
    /// title command buffer when another follows at once (more buffers in
    /// the submission, or the next submission queued): ~22 of its ~75
    /// batches per frame end there, each a batch begin (~60 us with slot
    /// waits) and a costlier first command (~100 us) more. Live A/B
    /// 2026-10-15 late night: +3.8% flips/s (1.023-1.050); default 1.
    /// 2 (default): the compute queues too (+1.4% flips/s, -1.3% CPU per
    /// command against 1 in a live A/B).
    #[inline]
    pub fn cmdbuf_merge() -> u64 {
        get(105)
    }

    /// A batch's slot (its command pools, descriptor pool and fence) is
    /// reset by the completion thread once the batch completed, not by the
    /// recording thread when it begins its next batch in that slot (default
    /// 1 since 2026-10-16: -0.5% CPU per command; FINDINGS.md).
    #[inline]
    pub fn offthread_reset() -> bool {
        get(106) != 0
    }

    /// A label wait whose label a GPU queue wrote last sleeps (`WaitOnAddress`,
    /// at most 1 ms at a time) after its spins until a promise, a latched CP
    /// write or a GPU-performed label write wakes it, instead of yielding and
    /// sleeping 100 us between polls (`cp::wake_label`; default since
    /// 2026-10-16: -4.1% CPU per command, the compute queues' CPU -17%).
    #[inline]
    pub fn label_wake() -> bool {
        get(107) != 0
    }

    /// Spins (`pause`) of a parallel run's waits (a worker for its chunk's
    /// turn, the main command processor for a chunk) before they block on
    /// the run's condition variable (512 since 2026-10-16, 4,096 before:
    /// -5% CPU per command with the front end on one CCD).
    #[inline]
    pub fn par_spins() -> u64 {
        get(108)
    }

    /// Batch slots of the graphics queue's command processor (`slots` for
    /// the others): its batches wait for a slot when the GPU is behind. 8
    /// against 4: +0.9% flips/s (noise level), more video memory in use
    /// (FINDINGS.md 2026-10-16): 4.
    #[inline]
    pub fn slots_gfx() -> u64 {
        get(109)
    }

    /// Diagnostics: the graphics queue's command processor ends its batch
    /// after every N commands it records serially (0: off): the marginal
    /// cost of a batch boundary (`cp::SPLITS`; the thread CPU window line).
    #[inline]
    pub fn split_every() -> u64 {
        get(110)
    }

    /// A batch submitted while the GPU has run out of work (every batch
    /// handed over completed) records its copies for the copy queue into its
    /// prologue on the main queue instead: nothing runs that they could
    /// overlap, and the wait across queues (graphics batches with their own
    /// copies there started 65 us after their submission, 27 us without;
    /// FINDINGS.md 2026-10-16) is left out. Neutral on a quiet machine
    /// (flips/s 1.016, 0.997, 0.999, 0.995): off.
    #[inline]
    pub fn xpre_dry() -> bool {
        get(111) != 0
    }

    /// A parallel run is split among the workers at equal recording cost
    /// instead of equal command counts, by the per-command costs its workers
    /// measured when the same run (same packet headers) was recorded before
    /// (`cp::RunCost`): its slowest chunk ended 14% after the mean one
    /// (median; FINDINGS.md 2026-10-16), partly the same chunks every frame.
    /// Default since 2026-10-16: +1.5% flips/s, +0.4% CPU per command.
    #[inline]
    pub fn par_cost() -> bool {
        get(112) != 0
    }

    /// A command processor's batch is finished by the submission thread
    /// (`tune submit_thread`): its prologue's copies recorded and the
    /// prologue, its copy-queue copies and its command buffer ended there,
    /// not on the recording thread (~18 us of a ~50 us submission,
    /// profiled; FINDINGS.md 2026-10-16). Parallel workers' chunk batches
    /// keep finishing theirs.
    #[inline]
    pub fn submit_finish() -> bool {
        get(113) != 0
    }

    /// Diagnostics: the parallel-run planner also computes each chunk's
    /// start state a register at a time (`Cp::apply_state` for every packet)
    /// and logs where it differs from the one-copy register ranges
    /// (`put_regs`).
    #[inline]
    pub fn plan_check() -> bool {
        get(114) != 0
    }

    /// Every video-memory home made from now on is write-watched like the
    /// indirect-argument homes: once per synchronization epoch the pages the
    /// CPU wrote are compared word by word with what the home last took in,
    /// and the words guest code changed are copied in before its next GPU
    /// use (logged). Without it (0; the default until 2026-10-17) a home
    /// sees only the CPU-side writes we make (CP writes, file reads): guest
    /// code writing into its range is not seen and GPU work keeps reading
    /// the home's older copy (level loads writing the new area's meshes into
    /// a homed buffer: the load-time fault class, misplaced and stretched
    /// meshes; -1.8% at the gate).
    #[inline]
    pub fn homes_watch() -> bool {
        get(115) != 0
    }


    /// Bytes from an unbounded pointer's base (a read-only global pointer
    /// of unknown extent, read in place) within which the GPU-written parts
    /// of video-memory homes are written back to guest memory first (0:
    /// none; the pointer is assumed not to read homes, as before).
    #[inline]
    pub fn unbounded_near() -> u64 {
        get(116)
    }

    /// A GPU-owned image overwritten through a buffer is reloaded from guest
    /// memory after the GPU-written parts of video-memory homes over it are
    /// written back (0: as before, for A/B: the reload reads older bytes
    /// where the writers bound a home).
    #[inline]
    pub fn image_homes() -> bool {
        get(117) != 0
    }

    /// Microseconds the workers of a queue spin for a run when its planner
    /// pre-wakes them, once its chunks are cut (`par::Pool::prewake`; 0: off,
    /// the workers are woken when the run is handed over).
    #[inline]
    pub fn par_prewake() -> u64 {
        get(118)
    }

    /// Draws with neither color nor depth attachments whose shaders write
    /// memory (buffer or image stores) are recorded in a rendering instance
    /// without attachments, its area limited by the draw's scissors (0:
    /// skipped, as before 2026-10-16: one each per recording thread in every
    /// session, `draw with only memory writes`).
    #[inline]
    pub fn attachless_draws() -> bool {
        get(119) != 0
    }

    /// Block-compressed textures that GPU work transcodes: the mip tail the
    /// title stores through a view of the tail block alone (a single-level
    /// view one swizzle block in size at the surface's base) is copied into
    /// the tail mips (0: only the view of the whole chain is copied, as
    /// before 2026-10-16: the tail mips kept whatever that view held).
    #[inline]
    pub fn block_tails() -> bool {
        get(120) != 0
    }

    /// Storage images whose programs select the mip level they store to or
    /// load from (image_store_mip / image_load_mip: the title's mip-chain
    /// generators, several levels per dispatch) are bound one view per
    /// level (`rdna::translate::Spec::mip_views`; 0: one view of the T#'s
    /// base level that receives every level, as before 2026-10-16: the
    /// character creator's face, generated mips over the base level).
    /// Switching translates programs anew.
    #[inline]
    pub fn mip_views() -> bool {
        get(121) != 0
    }

    /// Worker threads that compile the pipelines of the commands after a
    /// first use while its command processor compiles its own
    /// (`gpu::compile`; 0: each when execution reaches it. Its timing made
    /// World 8-2's homes race spoil the load until `homes_watch` closed
    /// it). By default (`COMPILE_AHEAD_AUTO`, since 2026-10-17 morning) a
    /// third of the logical CPUs, 2 to 12 (8 on the 24 of the development
    /// machine): first sessions reached the courtyard after 71-73 s with 8
    /// and 70 s with 12 against 77-81 s with 4 (FINDINGS.md 2026-10-17).
    #[inline]
    pub fn compile_ahead() -> u64 {
        match get(122) {
            COMPILE_AHEAD_AUTO => {
                static N: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
                *N.get_or_init(|| (std::thread::available_parallelism().map_or(6, |n| n.get()) as u64 / 3).clamp(2, 12))
            }
            n => n,
        }
    }

    /// `compile_ahead`'s default: workers by the logical CPUs.
    pub const COMPILE_AHEAD_AUTO: u64 = 99;

    /// Runs of lane-local vector instructions translated inside one lane
    /// guard (`rdna::translate::Spec::lane_guards`; default 1 since
    /// 2026-10-17: -22% SPIR-V words, -11% compile time; 0: a select per
    /// register write). Switching translates programs anew.
    #[inline]
    pub fn lane_guards() -> bool {
        get(123) != 0
    }

    /// Graphics pipelines compiled ahead of their first use like compute
    /// ones (`gpu::compile`; needs `compile_ahead`): a draw whose pipeline
    /// is missing scans the draws after it, whose pipelines the compile
    /// workers create from the registers the scan saw.
    #[inline]
    pub fn compile_gfx() -> bool {
        get(124) != 0
    }

    /// A batch holding GPU-side label writes is submitted once it has
    /// recorded this long, us (CPU threads of the title may poll them;
    /// another queue that waits for one asks at once).
    #[inline]
    pub fn label_latency() -> std::time::Duration {
        std::time::Duration::from_micros(get(125))
    }

    /// Pipelines made ahead from learned specs (`gpu::precompile`): 1 learns
    /// each compute program's specs at its first translations and each
    /// graphics pipeline's stages and state, with the programs' code (and
    /// compares the guess from a compute object, statistics); 2 (default
    /// since 2026-10-17) also makes the learned pipelines on
    /// `precompile_workers` threads at the lowest priority, from the start
    /// of a session or when the title creates the programs' objects; 3 a
    /// compute object's guess too; 0 off.
    #[inline]
    pub fn precompile() -> u64 {
        get(126)
    }

    /// Threads that make pipelines from shader objects (`precompile` 2; at
    /// the lowest priority). 0: as many as `compile_ahead`'s workers.
    #[inline]
    pub fn precompile_workers() -> u64 {
        match get(127) {
            0 => compile_ahead(),
            n => n,
        }
    }

    /// Depth attachments over memory that GPU work wrote through a color
    /// view (the half-resolution depth the title's compute shader builds
    /// and its particles test against) get that view's data copied in
    /// before their draws (`exec::sync_depth_views`; 0: they keep their own
    /// contents, as before 2026-10-18: particles showed through walls).
    #[inline]
    pub fn depth_views() -> bool {
        get(128) != 0
    }

    /// Diagnosis only (stale copies): a shared batch-start copy found clean
    /// by a write-mark take is used without another take for this many
    /// flips (0: checked once per synchronization epoch, as it must be).
    /// Bounds what the copy cache's checks cost the frame.
    #[inline]
    pub fn pre_trust() -> u64 {
        get(129)
    }

    /// An indexed indirect draw whose arguments the GPU reads, following one
    /// like it with no other packet between them (the culled clusters' draws:
    /// ~3,700 of the gate frame's ~5,600 indirect draws come in such runs),
    /// is recorded as one more `vkCmdDrawIndexedIndirect` on the state the
    /// previous draw left (pipeline, descriptors, dynamic state, index
    /// buffer, rendering instance), without preparing or binding its stages
    /// again (`Exec::draw_repeat`). Needs `draw_args_gpu`.
    #[inline]
    pub fn draw_repeat() -> bool {
        get(130) != 0
    }

    /// Indexed indirect draws whose arguments the CPU reads hand the start
    /// instance to the vertex stage as `draw_args_gpu` does: as Vulkan's
    /// first instance, the stage taking that user-data word from
    /// `gl_BaseInstance` (`DrawArgs::base_inst`), instead of in its register
    /// (which changes the stage's SGPR data per draw). Draws like the one
    /// before can then be repeated (`draw_repeat`) without GPU-read
    /// arguments; stages that need the word on the CPU keep the register.
    #[inline]
    pub fn cpu_args_base() -> bool {
        get(131) != 0
    }

    /// Stages read their SGPR data at an offset they take from a push
    /// constant (`Spec::sgpr_pc`): the data binding covers the batch's whole
    /// ring and stays the same from command to command, so a command whose
    /// resources equal the one before pushes no descriptors. Read when
    /// stages are specialized: set it at boot (`DS_SGPR_PC`; switched later,
    /// every stage is translated and its pipelines made again). Off: on
    /// the RTX 4090 driver it pushed 17% fewer descriptors per push yet
    /// cost 1.8% more CPU per command (FINDINGS.md 2026-10-19).
    #[inline]
    pub fn sgpr_pc() -> bool {
        get(132) != 0
    }

    /// Write-watch queries taking turns (`mem::take_written`): 1 the
    /// parallel recording workers and the texture scrubber one at a time,
    /// 2 every thread; 0 none.
    #[inline]
    pub fn ww_gate() -> u64 {
        get(133)
    }

    /// A zero fill (the fill kernel over a whole range) of 256 KiB or more
    /// whose range is all the memory of GPU-written images it clears
    /// (`Images::fill_covered`) clears only the images: their guest memory
    /// is not kept current, and the store across PCIe cost the GPU ~0.7 ms
    /// per gate frame (an 18 MB layered render target each frame).
    #[inline]
    pub fn fill_skip() -> bool {
        get(134) != 0
    }

    /// Images are evicted when video memory runs short (`images::maybe_evict`):
    /// once this process uses more than `vram_high` percent of the device-local
    /// heap's budget (VK_EXT_memory_budget), the least recently used images
    /// go until it is under `vram_low` percent. 0: images are kept for the
    /// whole session (the earlier behavior: six level loads filled a 24 GB
    /// GPU, 2026-10-19).
    #[inline]
    pub fn image_evict() -> bool {
        get(135) != 0
    }

    #[inline]
    pub fn vram_high() -> u64 {
        get(136).clamp(10, 100)
    }

    #[inline]
    pub fn vram_low() -> u64 {
        get(137).clamp(5, 100)
    }

    /// Bytes of the aligned window a copy cache check takes write marks
    /// for (`precache::lookup`): one take checks every entry inside it for
    /// the epoch. 0: each entry's own range. A power of two.
    #[inline]
    pub fn pre_window() -> u64 {
        let v = get(138);
        if v == 0 { 0 } else { v.next_power_of_two().max(4096) }
    }

    /// Write-watched video-memory homes check their pages for CPU writes
    /// once per epoch that a title submission or REWIND ended, not once per
    /// epoch of any kind (`homes::check_epoch`; default since 2026-10-20:
    /// +2.4% flips/s at the gate, FINDINGS.md).
    #[inline]
    pub fn home_submit_epochs() -> bool {
        get(139) != 0
    }

    /// The graphics command processor on a physical core of its own among
    /// the front end's CPUs (`prio::own_core`).
    #[inline]
    pub fn gfx_core() -> bool {
        get(140) != 0
    }

    /// Measurement only: draws (bit 0) and dispatches (bit 1) are not
    /// recorded while the command processors still execute their packets,
    /// labels and writes: the title's own frame without our GPU work (another
    /// port's `skip_cmds`). Wrong pictures and GPU-written
    /// data the title reads stay stale.
    #[inline]
    pub fn skip_cmds() -> u64 {
        get(141)
    }

    /// Measurement only (unsafe): the graphics CP skips its waits for labels
    /// compute queue 0x50 wrote last (`Cp::wait`).
    #[inline]
    pub fn xwait_skip() -> bool {
        get(142) != 0
    }

    /// Buffers bound by device address (`Spec::addresses`: no buffer
    /// descriptors; translations made from then on). Needs
    /// bufferDeviceAddress.
    #[inline]
    pub fn addresses() -> bool {
        get(143) != 0 && super::vk::ctx().device_addresses
    }

    /// Consecutive direct dispatches of a program run as one dispatch
    /// (`Spec::batched`; programs seen in runs, `shaders::mark_batchable`).
    /// Needs bufferDeviceAddress (batched translations address their
    /// buffers).
    #[inline]
    pub fn batch_cs() -> bool {
        get(144) != 0 && super::vk::ctx().device_addresses
    }

    /// The title's job workers without work (`jobidle`): microseconds a
    /// worker finds nothing to run before it sleeps between looks (one such
    /// worker, the watcher, keeps looking); 0: the title's own polling.
    /// Default 50: at the gate +7.8% flips/s, -8.5% CPU per command, the
    /// process ~6 logical CPUs instead of ~17; on 8 logical CPUs ~20 flips/s
    /// instead of ~6 and no audio underruns (FINDINGS.md 2026-10-21).
    #[inline]
    pub fn job_idle() -> u64 {
        get(145)
    }

    /// Microseconds such a worker sleeps before each further look
    /// (`job_idle`).
    #[inline]
    pub fn job_sleep() -> u64 {
        get(146)
    }

    /// Microseconds one such worker at a time looks without sleeps (the
    /// watch rotates; 0: no watcher; `job_idle`).
    #[inline]
    pub fn job_watch() -> u64 {
        get(147)
    }

    /// A draw right after the last full draw with its context state and the
    /// same stages (programs, resources, user data and SGPR data) is
    /// recorded as one more draw call on the state that draw left
    /// (`Exec::draw_again`; 37% of the draws, FINDINGS.md 2026-10-21).
    #[inline]
    pub fn draw_again() -> bool {
        get(148) != 0
    }

    /// Batched draws: consecutive draws of a program pair with the same
    /// context state (and images) as one multi-draw whose arguments and
    /// per-draw SGPR data offsets the CPU appends (`exec::draw` draw runs;
    /// the pair's stages translated batched, buffers by device address).
    /// Needs drawIndirectCount and bufferDeviceAddress.
    #[inline]
    pub fn batch_draws() -> bool {
        get(149) != 0 && super::vk::ctx().draw_indirect_count && super::vk::ctx().device_addresses
    }

    /// How translations declare their SGPR data (`Spec::sgpr_ubo`): 0 a
    /// storage buffer, 1 a uniform buffer where the binding is the command's
    /// own range (else a read-only storage buffer), 2 a read-only storage
    /// buffer, 3 as 1 and batched compute translations through a uniform
    /// window of the ring (1 where uniform buffers cannot span 64 KiB).
    /// Takes effect for translations made after a change.
    #[inline]
    pub fn sgpr_ubo() -> u8 {
        let v = get(150).min(3) as u8;
        if v == 3 && super::vk::ctx().limits.max_uniform_buffer_range < 4 * rdna::translate::SGPR_WINDOW_DWORDS {
            1
        } else {
            v
        }
    }

    /// The shadows a write-back flushes merged by one dispatch
    /// (`rdna::kernels::shadow_merge_many`, destinations by device address)
    /// instead of one dispatch and descriptor set each.
    #[inline]
    pub fn merge_many() -> bool {
        get(151) != 0
    }

    /// The process opts out of Windows' power throttling (execution speed
    /// and the timer-resolution rule for hidden or occluded windows) and
    /// raises the timer resolution to 1 ms (`thread::apply_power_optout`,
    /// applied at start and when `tune power_optout` changes).
    #[inline]
    pub fn power_optout() -> bool {
        get(152) != 0
    }

    /// Flips only at the emulated vblanks (59.94 Hz: a frame that misses one
    /// waits up to 16.7 ms more). 0: a ready flip is shown at the vblank
    /// thread's next 1 ms tick once the title's flip rate allows it (as many
    /// vblank periods since the last flip as the rate asks), while the
    /// title still sees its vblanks at 59.94 Hz (`hle::libs::videoout`).
    #[inline]
    pub fn flip_grid() -> bool {
        get(153) != 0
    }

    /// The presenter keeps the frame's aspect and halves a frame more than
    /// twice the window's size before its last blit (`gpu::vk::present`). 0:
    /// one blit stretched over the whole window (comparison).
    #[inline]
    pub fn present_fit() -> bool {
        get(154) != 0
    }

    /// A frame-rate cap in frames per second (0: none): flips are shown at
    /// least 1/cap apart, on the vblank grid a whole number of vblanks
    /// (`hle::libs::videoout`; 30 -> every 2nd vblank at 59.94 Hz, 20 -> 3rd).
    /// The title's game time follows real time, so its speed is kept.
    #[inline]
    pub fn frame_cap() -> u64 {
        get(155)
    }

    /// The texture scrubber's sleep between passes in write-watch mode
    /// (`tex_scrub` 2), ms: each pass takes the marks of the next 64
    /// uploaded textures (diagnosis of its queries' cost to the others).
    #[inline]
    pub fn scrub_ms() -> u64 {
        get(156).clamp(1, 60_000)
    }

    /// Diagnosis (timing only, wrong results; for replay A/Bs of what a
    /// step costs): bit 0 skips draws' and dispatches' texture syncs, bit 1
    /// draws' image-hazard checks (fast clears, overwrites, block copies,
    /// depth views are not resolved).
    #[inline]
    pub fn skip_steps() -> u64 {
        get(157)
    }

    /// Draws whose pixel shader exports the stencil test value (the device
    /// has no stencil export): 0 use the register's test value as the
    /// reference, 1 use 0 (diagnosis: what the marking those draws write
    /// changes on screen).
    #[inline]
    pub fn stencil_export() -> u64 {
        get(158)
    }

    /// Smallest written-buffer shadow whose working copy is shared after its
    /// merge (`shadow_share`), bytes (the copy cache's own floor is
    /// `pre_share_min`).
    #[inline]
    pub fn shadow_share_min() -> u64 {
        get(159)
    }

    /// Small written windows (under 256 KiB, where dispatches got per-batch
    /// shadows) kept in video memory across batches, written back to guest
    /// memory only for readers there (`gpu::keep`); 0 off (windows kept so
    /// far are written back and dropped by their queues).
    #[inline]
    pub fn keep_small() -> bool {
        get(160) != 0
    }

    /// Frame-ahead recording (`gpu::frameahead`): a second graphics
    /// recorder takes the next frame while the graphics command processor
    /// finishes the current one; 1: what it records is dropped at the flip
    /// (stage 1, a measurement); 0 off.
    #[inline]
    pub fn frame_ahead() -> u64 {
        get(161)
    }

    /// The graphics command processor records past its parallel runs
    /// before collecting their chunks (`Cp::past_start`): its batches after
    /// a run held until a stop (at the latest its submission's end), then
    /// submitted after the chunks' or dropped and recorded again (default;
    /// +0.8% at the gate, FINDINGS.md 2026-10-23); 0 off.
    #[inline]
    pub fn past_runs() -> bool {
        get(162) != 0
    }

    /// Diagnosis: what it recorded past a run is always dropped (the path
    /// that records it again, tested).
    #[inline]
    pub fn past_drop() -> bool {
        get(163) != 0
    }

    /// The title's convolution reverb computed (`hle::libs::acm_conv`,
    /// default); 0: its jobs left undone (no reverb, the stand-in before
    /// 2026-10-23).
    #[inline]
    pub fn acm_reverb() -> bool {
        get(164) != 0
    }

    /// The graphics CP's label waits of 1 ms or more still to be logged in
    /// detail (`tune trace_waits <n>`: the next n, wherever they happen;
    /// `DS_TRACE_WAITS=<queue>` logs a queue's first 200 from the start).
    pub fn take_trace_waits() -> bool {
        get(165) > 0 && ALL[165].value.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| if v != UNSET && v > 0 { Some(v - 1) } else { None }).is_ok()
    }

    /// The graphics CP's planning of parallel runs without its costliest
    /// steps (2026-10-23, a vista's profile): indirect-argument ranges
    /// merged as they come instead of sorted, run costs pruned once a frame,
    /// the instance-offset slot's last answer reused; 0: as before (A/B).
    #[inline]
    pub fn plan_fast() -> bool {
        get(166) != 0
    }

    /// Programs compiled in the background at their first use, their
    /// dispatches left out until then, instead of the command processor
    /// stopping for the compile (`exec::compile_in_background`): 2 the
    /// engine programs of per-frame effects named in `DS_ASYNC_ENGINE`
    /// (the particles' lighting atlas passes), 1 also the title's particle
    /// programs (`particles::in_background`), 0 none. Particle programs
    /// left out broke their effects' state in first sessions (an effect's
    /// other programs ran on it: shaders looping until the loop watchdog
    /// stopped them, 43,024 times in a walk of 1-2, and GPU work of 100-190
    /// ms a frame for minutes; FINDINGS.md 2026-10-24).
    #[inline]
    pub fn async_particles() -> u64 {
        get(167)
    }

    /// A dispatch's new VRAM shadows get the guest data in the batch's
    /// prologue (before all its work) when nothing earlier in the batch
    /// writes their window, instead of by copies and two barriers before
    /// the dispatch: those barriers drained the GPU before every such
    /// dispatch, so the title's particle chain (~1,000 small dispatches a
    /// frame, each writing small buffers of its own) ran one dispatch at a
    /// time (`--dispatch-bench`: 256 dispatches overlap in 61 us, 6 ms with
    /// a barrier between). 0: as before.
    #[inline]
    pub fn shadow_pre() -> bool {
        get(168) != 0
    }

    /// The fewest bytes a read-only binding inside a range GPU work writes
    /// through large bindings needs to give that range a home (`home_reads`;
    /// default 1 MiB; 0: over `pre_max`). GPU-written buffers of 4-8 MiB that
    /// later batches bind read-only were written in place across PCIe and
    /// copied back for every reading batch (a vista: ~54 MB a frame).
    #[inline]
    pub fn home_read_min() -> u64 {
        match get(169) {
            0 => pre_max() + 1,
            v => v,
        }
    }

    /// The largest written buffer binding given a VRAM shadow (`Exec`: a
    /// working and a pristine copy in the slot's shadow segment, merged back
    /// at its batch's end, its working copy then shared with later readers,
    /// `tune shadow_share`); larger ones are written in place across PCIe and
    /// copied back for every batch that reads them (`exec::SHADOW_MAX`, at
    /// most 12 MiB: both copies fit a segment of `SHADOW_BYTES`).
    #[inline]
    pub fn shadow_max() -> u64 {
        get(170).min(12 << 20)
    }

    /// Logical CPUs the compute queues' command processors run on (`prio`):
    /// 0 those of the front end (`cp_cpus`), 1 (default) the title's
    /// (`guest_cpus` AUTO: its last-level-cache group has cores to spare
    /// while the front end's runs the graphics command processor and its
    /// workers; +1.7% at 1-1's vista, FINDINGS.md 2026-10-26; nothing
    /// changes where there is one such group), else a decimal mask.
    #[inline]
    pub fn acb_cpus() -> u64 {
        get(171)
    }

    /// Diagnostics, hosted runs (`hostcensus`): bit 1 arms shared copies
    /// clean for 30 flips read-only, bit 2 GPU-written windows of up to 4 MiB
    /// no-access once their batch completed; the host's faults on them are
    /// counted (observation only).
    #[inline]
    pub fn host_census() -> u64 {
        get(172)
    }

    /// Hosted runs (`precache`, stage H4): shared batch-start copies clean
    /// for 30 flips are armed read-only through the host and need no
    /// write-watch take while all their pages stay armed.
    #[inline]
    pub fn pre_protect() -> bool {
        get(173) != 0
    }

    /// Homes (`homes::create`) only over ranges the CPU did not write while
    /// they were watched for 30 frames; refused ones wait 600 frames.
    #[inline]
    pub fn home_cpu_check() -> bool {
        get(174) != 0
    }

    /// Diagnostics (`placement`): sampled logical CPUs of the command
    /// processors and the title's job workers, by physical core.
    #[inline]
    pub fn placement() -> bool {
        get(175) != 0
    }

    /// Logical CPUs (a decimal mask) of queue 0x50's command processor
    /// alone; 0: as the other compute queues (`acb_cpus`).
    #[inline]
    pub fn acb50_cpus() -> u64 {
        get(176)
    }

    /// Texture syncs test the exact guest range of each texture (default;
    /// 0: up to 16 bytes per texel, the old bound: `exec::texture_bytes`).
    #[inline]
    pub fn tex_exact() -> bool {
        get(177) != 0
    }

    /// A write-watched home's check for CPU writes queries its pages with
    /// the homes' lock released (`homes::check_before_lock`; 0: under the
    /// lock, as before).
    #[inline]
    pub fn home_check_unlocked() -> bool {
        get(178) != 0
    }

    /// Read-only unbounded pointer bindings whose mapping ends within this
    /// many bytes of their base are placed as sized read-only buffers of
    /// that size (batch-start copies, mirrors) instead of read in place
    /// across PCIe (0: never, the default).
    #[inline]
    pub fn pre_unsized() -> u64 {
        get(179)
    }

    /// A write-watched home's check for CPU writes covers the 2 MiB windows
    /// of the binding that needs it (`homes::CHECK_WIN`; 0: the whole home
    /// at once, as before).
    #[inline]
    pub fn home_check_win() -> bool {
        get(180) != 0
    }

    /// GPU-resolved chains (RENDERER.md "GPU-resolved item passes"): 0 off,
    /// 1 the guest page table is kept and programs found eligible resolve
    /// their per-item chains in the shader.
    pub fn gpu_chains() -> u64 {
        get(181)
    }

    /// GPU-resolved chains' static ranges in a video-memory mirror armed
    /// through the host (`gmirror`; hosted runs only): 1 on, 0 (default) read
    /// across PCIe from the import windows. Arming ranges as soon as they are
    /// met lost the device during level loads (FINDINGS.md 2026-10-26);
    /// diagnosis modes: 2 armed and copied, not read; 3 copied and read, not
    /// armed; 4 armed only; 5 copied only.
    pub fn gpu_mirror() -> u64 {
        get(182)
    }

    /// The graphics command processor records past a wait on a label a
    /// compute queue wrote last (`Cp::past_wait_start`): held until the
    /// label is promised or a stop, then checked against what the other
    /// queues recorded meanwhile and submitted after the writer's batch, or
    /// dropped and recorded again from the wait: 1 (default since
    /// 2026-10-27) labels queue 0x50 wrote last, 2 any compute queue's; 0
    /// off.
    #[inline]
    pub fn past_waits() -> u64 {
        get(183)
    }

    /// A forwarded copy's entry also accepts, as words memory holds before
    /// it runs, those of the earlier copies into the same destination it
    /// replaced (`forward::Entry::earlier`); 0: only the destination's words
    /// when it was recorded (a reader that met an earlier copy's word ended
    /// the entry as the title's rewrite).
    #[inline]
    pub fn forward_earlier() -> bool {
        get(184) != 0
    }

    /// The graphics CP forwards the per-view copies it records past a
    /// parallel run too, into a view of its own until the stretch commits
    /// (`forward::hold_begin`); 0: not forwarded there (readers after them
    /// waited for the GPU).
    #[inline]
    pub fn forward_held() -> bool {
        get(185) != 0
    }

    /// While most of a label's recent stretches past its waits were
    /// dropped (`past_waits`; each outcome weighs a fourth), only every n-th
    /// wait on it is gone past (a probe); 0: every one.
    #[inline]
    pub fn past_backoff() -> u64 {
        get(186)
    }

    /// `batch_draws` 2: runs of one draw each (diagnosis: the batched stages
    /// without appends).
    #[inline]
    pub fn batch_draws_appends() -> bool {
        get(149) != 2
    }

    /// A switch's value by name; None if unknown.
    pub fn value(name: &str) -> Option<u64> {
        ALL.iter().position(|t| t.name == name).map(get)
    }

    /// Sets a switch by name; false if unknown.
    pub fn set(name: &str, v: u64) -> bool {
        match ALL.iter().find(|t| t.name == name) {
            Some(t) => {
                t.value.store(v, Ordering::Relaxed);
                true
            }
            None => false,
        }
    }
}

pub mod ahead;
pub mod compbench;
pub mod dispbench;
pub mod compile;
pub mod pacing;
pub mod pagetable;
pub mod gpures;
pub mod gmirror;
pub mod drawcensus;
pub mod particles;
pub mod forward;
pub mod par;
pub mod prio;
pub mod precache;
pub mod precompile;
pub mod lockstat;
pub mod exec;

/// Whether a GPU import may cover guest [base, base+len) (`exec`).
pub(crate) fn imports_may_cover(base: u64, len: u64) -> bool {
    exec::imports_may_cover(base, len)
}
pub mod pcache;
pub mod pool;
pub mod lodstats;
pub mod pstats;
pub mod mirror;
pub mod ranges;
pub mod homes;
pub mod resident;
pub mod timeline;
pub mod replay;
pub mod threadcpu;

/// Fast non-cryptographic hashing for maps on the command processors' hot
/// paths (the image cache). Lookups compare keys, so a collision costs time
/// only; keys that stand for content (translation and pipeline hashes) keep
/// SipHash.
pub mod fx {
    use std::hash::{BuildHasherDefault, Hasher};

    #[derive(Default, Clone, Copy)]
    pub struct FxHasher(u64);

    const K: u64 = 0x517c_c1b7_2722_0a95;

    impl FxHasher {
        #[inline]
        fn add(&mut self, w: u64) {
            self.0 = (self.0.rotate_left(5) ^ w).wrapping_mul(K);
        }
    }

    impl Hasher for FxHasher {
        fn write(&mut self, bytes: &[u8]) {
            for c in bytes.chunks(8) {
                let mut b = [0u8; 8];
                b[..c.len()].copy_from_slice(c);
                self.add(u64::from_le_bytes(b));
            }
        }
        fn write_u8(&mut self, i: u8) {
            self.add(i as u64)
        }
        fn write_u16(&mut self, i: u16) {
            self.add(i as u64)
        }
        fn write_u32(&mut self, i: u32) {
            self.add(i as u64)
        }
        fn write_u64(&mut self, i: u64) {
            self.add(i)
        }
        fn write_usize(&mut self, i: usize) {
            self.add(i as u64)
        }
        fn finish(&self) -> u64 {
            self.0
        }
    }

    pub type Map<K, V> = std::collections::HashMap<K, V, BuildHasherDefault<FxHasher>>;
    pub type Set<T> = std::collections::HashSet<T, BuildHasherDefault<FxHasher>>;
}
pub mod images;
pub mod png;
pub mod track;
pub mod shaders;
pub mod swizzle_tables;
pub mod tiling;
pub mod verify;
pub mod vk;
pub mod xwait;
pub mod census;

use parking_lot::{Condvar, Mutex};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Flips presented so far (VideoOut).
pub static FRAMES: AtomicU64 = AtomicU64::new(0);
/// Command buffer submissions received (DCB/ACB).
pub static SUBMITS: AtomicU64 = AtomicU64::new(0);
/// Debug: dump GPU-visible images at the next flip (control `dump`).
pub static DUMP_REQUEST: AtomicBool = AtomicBool::new(false);
static SHUTDOWN: AtomicBool = AtomicBool::new(false);

pub fn shutting_down() -> bool {
    SHUTDOWN.load(Ordering::Relaxed)
}

/// Stops command-processor waits (process shutdown).
pub fn shutdown() {
    SHUTDOWN.store(true, Ordering::SeqCst);
}

/// Queue id of the graphics (DCB) queue; compute queues use their Agc ids.
pub const GFX: u32 = u32::MAX;

/// Submitted command buffers; with `DS_CHECK_SUBMIT=1` also their content
/// hashes and the submission time (see `worker`).
struct Submission {
    bufs: Vec<(u64, u32)>,
    hashes: Vec<u64>,
    /// Copies of the buffers at submission (DS_CHECK_SUBMIT).
    copies: Vec<Vec<u32>>,
    at: std::time::Instant,
    /// Work other than command buffers (frame replay).
    ctl: Option<Ctl>,
    /// The queue's submission count when it was queued (its number).
    index: u64,
}

/// Work for a queue's command processor besides command buffers (frame
/// replay, `gpu::replay`).
#[derive(Clone, Copy, Debug)]
pub(crate) enum Ctl {
    /// Completes the GPU work recorded so far and its deferred effects.
    Finish,
    /// Keeps the command processor's register state in slot 0 or 1.
    Save(usize),
    /// Restores the register state kept in that slot.
    Restore(usize),
}

/// Queues work for a queue's command processor outside the title's
/// submission path (no throttling, no replay hold).
pub(crate) fn enqueue(queue_id: u32, bufs: Vec<(u64, u32)>, ctl: Option<Ctl>) {
    let q = queue(queue_id);
    let mut g = q.pending.lock();
    let index = q.submitted.fetch_add(1, Ordering::SeqCst) + 1;
    g.push_back(Submission { bufs, hashes: Vec::new(), copies: Vec::new(), at: std::time::Instant::now(), ctl, index });
    drop(g);
    q.cv.notify_one();
}

/// The queues created so far.
pub(crate) fn queue_ids() -> Vec<u32> {
    QUEUES.lock().iter().map(|q| q.id).collect()
}

/// Whether every queue has completed all the work queued to it.
pub(crate) fn queues_idle() -> bool {
    QUEUES.lock().iter().all(|q| q.done.load(Ordering::SeqCst) >= q.submitted.load(Ordering::SeqCst))
}

/// The compute queues' command processors: (queue id, host thread id).
pub(crate) fn compute_threads() -> Vec<(u32, u32)> {
    QUEUES.lock().iter().filter(|q| q.id != GFX).map(|q| (q.id, q.tid.load(Ordering::Relaxed))).filter(|x| x.1 != 0).collect()
}

/// The GPU front end's threads by kind ("gfx", "acb", "par", "prep",
/// "submit", "complete", "scrub") and host thread id.
pub(crate) fn cpu_threads() -> Vec<(&'static str, u32)> {
    let mut v: Vec<(&'static str, u32)> = QUEUES.lock().iter().map(|q| (if q.id == GFX { "gfx" } else { "acb" }, q.tid.load(Ordering::Relaxed))).filter(|x| x.1 != 0).collect();
    v.extend(PROFILE_THREADS.lock().iter().copied());
    v
}

/// Whether one of the command buffers (address, dwords) holds a flip (the
/// private FLIP packet) in its top level.
pub(crate) fn has_flip(bufs: &[(u64, u32)]) -> bool {
    use crate::hle::libs::agc_cmd::{pop, DS_MAGIC};
    bufs.iter().any(|&(addr, size)| {
        let mut i = 0u64;
        while i < size as u64 {
            let a = addr + 4 * i;
            if !cp::guest_readable(a, 12) {
                return false;
            }
            let h = unsafe { *(a as *const u32) };
            if h >> 30 != 3 {
                i += 1;
                continue;
            }
            let n = ((h >> 16) & 0x3FFF) as u64 + 1;
            if (h >> 8) & 0xFF == 0x10 && n >= 2 && unsafe { *((a + 4) as *const u32) } == DS_MAGIC && unsafe { *((a + 8) as *const u32) } == pop::FLIP {
                return true;
            }
            i += n + 1;
        }
        false
    })
}

fn buffer_copy(addr: u64, size: u32) -> Vec<u32> {
    if !cp::guest_readable(addr, 4 * size as u64) {
        return Vec::new();
    }
    unsafe { std::slice::from_raw_parts(addr as *const u32, size as usize) }.to_vec()
}

/// `DS_TRACE_SUBMIT=1`: logs submissions and their completion.
fn trace_submit() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_TRACE_SUBMIT").ok().as_deref() == Some("1"))
}

fn check_submit() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_CHECK_SUBMIT").ok().as_deref() == Some("1"))
}

fn buffer_hash(addr: u64, size: u32) -> u64 {
    if !cp::guest_readable(addr, 4 * size as u64) {
        return 0;
    }
    let w = unsafe { std::slice::from_raw_parts(addr as *const u32, size as usize) };
    w.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &x| (h ^ x as u64).wrapping_mul(0x100_0000_01b3))
}

/// Submission backpressure (`DS_SUBMIT_THROTTLE=1`, diagnostic): a submit
/// first waits (up to `THROTTLE_MAX`) until the queue completed all earlier
/// submissions, GPU work included, to test whether a problem depends on how
/// far the command processor lags behind the submitting threads (it can lag
/// by much more than hardware, e.g. during pipeline compilation). Not while
/// the CP polls a label (it may be waiting for the submitting thread).
fn throttle() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_SUBMIT_THROTTLE").ok().as_deref() == Some("1"))
}
const THROTTLE_MAX: std::time::Duration = std::time::Duration::from_millis(250);
/// Submissions up to this many dwords wait for their completion (see `submit`).
const SMALL_SUBMIT: u64 = 512;

/// The packets of a command buffer copy, one token each (diagnostics).
fn pm4_summary(w: &[u32]) -> String {
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < w.len() {
        let h = w[i];
        if h >> 30 != 3 {
            out.push(format!("{i}:t{}", h >> 30));
            i += 1;
            continue;
        }
        let n = ((h >> 16) & 0x3FFF) as usize + 1;
        let p = |k: usize| w.get(i + 1 + k).copied().unwrap_or(0);
        let a64 = |k: usize| p(k) as u64 | (p(k + 1) as u64) << 32;
        let s = match (h >> 8) & 0xFF {
            0x10 if p(0) == crate::hle::libs::agc_cmd::DS_MAGIC => format!("mk{}", p(1)),
            0x93 => format!("wait64[{:#x}]>={:#x}", a64(1), a64(3)),
            0x3C => format!("wait[{:#x}]={:#x}", a64(1), p(3)),
            0x37 => format!("wd[{:#x}]={:#x}", a64(1), p(3)),
            0x49 => format!("rel[{:#x}]={:#x}", a64(2), a64(4)),
            0x15 => format!("disp{}x{}x{}", p(0), p(1), p(2)),
            0x76 => format!("sh{:#x}+{}", p(0), n - 1),
            0x46 => format!("ev{:#x}", p(0) & 0x3F),
            0x58 => "acq".to_string(),
            o => format!("op{o:#x}/{n}"),
        };
        out.push(format!("{i}:{s}"));
        i += n + 1;
    }
    out.join(" ")
}

/// (dispatches, draws, other packets) of a command buffer's top level.
fn packet_mix(addr: u64, size: u32) -> (u32, u32, u32) {
    let (mut d, mut w, mut o) = (0, 0, 0);
    let mut i = 0u64;
    while i < size as u64 {
        let a = addr + 4 * i;
        if !cp::guest_readable(a, 4) {
            break;
        }
        let h = unsafe { *(a as *const u32) };
        if h >> 30 != 3 {
            i += 1;
            continue;
        }
        let n = ((h >> 16) & 0x3FFF) as u64 + 1;
        match (h >> 8) & 0xFF {
            0x15 | 0x16 => d += 1,
            0x24 | 0x25 | 0x27 | 0x2C | 0x2D | 0x30 | 0x35 | 0x38 => w += 1,
            0x76 | 0x69 | 0x79 | 0x10 => {}
            _ => o += 1,
        }
        i += n + 1;
    }
    (d, w, o)
}

/// Whether a submission returns only once its work completed (see `submit`).
fn waits_for_completion(bufs: &[(u64, u32)]) -> bool {
    static SYNC: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    let mode = *SYNC.get_or_init(|| match std::env::var("DS_SUBMIT_SYNC").ok().as_deref() {
        Some("1") => 2,
        Some("small") => 1,
        _ => 0,
    });
    mode == 2 || (mode == 1 && bufs.iter().map(|x| x.1 as u64).sum::<u64>() <= SMALL_SUBMIT)
}
static THROTTLE_WAITS: AtomicU64 = AtomicU64::new(0);
static THROTTLE_NS: AtomicU64 = AtomicU64::new(0);
static THROTTLE_TIMEOUTS: AtomicU64 = AtomicU64::new(0);

/// Submission backpressure statistics.
pub fn throttle_status() -> String {
    format!(
        "submit throttle: {} waits, {} ms total, {} timeouts",
        THROTTLE_WAITS.load(Ordering::Relaxed),
        THROTTLE_NS.load(Ordering::Relaxed) / 1_000_000,
        THROTTLE_TIMEOUTS.load(Ordering::Relaxed)
    )
}

struct Queue {
    id: u32,
    pending: Mutex<VecDeque<Submission>>,
    cv: Condvar,
    /// Submissions completed (diagnostics, backpressure).
    done: AtomicU64,
    /// Submissions queued so far.
    submitted: AtomicU64,
    /// Signalled when a submission completes.
    done_cv: Condvar,
    /// The command processor is polling a label.
    cp_waiting: std::sync::atomic::AtomicBool,
    /// Host thread id of the queue's command processor (stack samples).
    tid: std::sync::atomic::AtomicU32,
}

static QUEUES: Mutex<Vec<&'static Queue>> = Mutex::new(Vec::new());

fn queue(id: u32) -> &'static Queue {
    let mut qs = QUEUES.lock();
    if let Some(q) = qs.iter().find(|q| q.id == id) {
        return q;
    }
    let q: &'static Queue = Box::leak(Box::new(Queue {
        id,
        pending: Mutex::new(VecDeque::new()),
        cv: Condvar::new(),
        done: AtomicU64::new(0),
        submitted: AtomicU64::new(0),
        done_cv: Condvar::new(),
        cp_waiting: std::sync::atomic::AtomicBool::new(false),
        tid: std::sync::atomic::AtomicU32::new(0),
    }));
    qs.push(q);
    let name = if id == GFX { "gpu-gfx".to_string() } else { format!("gpu-acb{id:#x}") };
    crate::info!("gpu: {name} command processor started");
    crate::thread::spawn_host(&name.clone(), 8 << 20, Box::new(move || worker(q, &name))).expect("gpu queue thread");
    q
}

fn worker(q: &'static Queue, name: &str) {
    crate::log::set_thread_name(name);
    q.tid.store(unsafe { windows_sys::Win32::System::Threading::GetCurrentThreadId() }, Ordering::Relaxed);
    crate::exception::arm_watch_current_thread(false);
    let mut cp = cp::Cp::new(name, q.id != GFX);
    if q.id == GFX {
        forward::enable_here();
        prio::set_main();
    }
    shaders::set_compute_queue(q.id != GFX);
    if q.id != GFX {
        prio::set_compute(q.id);
    }
    cp.wait_flag = Some(&q.cp_waiting);
    cp.more_follows = Some(Box::new(move || !q.pending.lock().is_empty()));
    // Register states kept for a frame replay (`Ctl::Save`).
    let mut saved: [Option<cp::ChunkState>; 2] = [None, None];
    // Submissions a stretch past a label wait runs through.
    let mut kept: Vec<Kept> = Vec::new();
    loop {
        prio::apply();
        let batch = {
            let mut g = q.pending.lock();
            let mut idle = None;
            let b = loop {
                if let Some(b) = g.pop_front() {
                    break b;
                }
                idle = idle.or_else(timeline::mark);
                q.cv.wait(&mut g);
            };
            timeline::end(name, idle, || "idle: no title submission".into());
            b
        };
        if let Some(c) = batch.ctl {
            match c {
                Ctl::Finish => cp.finish(),
                Ctl::Save(k) => saved[k] = Some(cp.chunk_state()),
                Ctl::Restore(k) => {
                    if let Some(s) = &saved[k] {
                        cp.set_chunk_state(s);
                    }
                }
            }
            let _g = q.pending.lock();
            q.done.fetch_add(1, Ordering::SeqCst);
            q.done_cv.notify_all();
            continue;
        }
        if q.id == GFX && frameprobe::on() {
            frameprobe::submission(&batch.bufs);
        }
        if q.id == GFX && frameahead::on() {
            frameahead::post(&cp, batch.index, &batch.bufs, batch.at);
        }
        // Recording past a label wait runs on through submissions: they are
        // kept until it settles, to run again from the wait if what was
        // recorded past it is dropped (`Cp::take_rewind_sub`).
        if cp.past_wait_active() {
            kept.push(Kept { index: batch.index, bufs: batch.bufs.clone(), at: batch.at });
        }
        run_submission(&mut cp, &batch, name);
        while let Some((s, buf, addr)) = cp.take_rewind_sub() {
            let Some(k) = kept.iter().position(|x| x.index == s) else {
                crate::error!("gpu[{name}]: a stretch past a label wait was dropped but submission {s} is not kept; going on");
                break;
            };
            let again: Vec<Kept> = kept.drain(k..).collect();
            for (j, x) in again.iter().enumerate() {
                let from = if j == 0 { Some((buf, addr)) } else { None };
                run_kept(&mut cp, x, from, name);
                if cp.rewind_sub_pending() {
                    break;
                }
            }
        }
        if cp.past_wait_active() {
            if !kept.iter().any(|x| x.index == batch.index) {
                kept.push(Kept { index: batch.index, bufs: batch.bufs.clone(), at: batch.at });
            }
        } else {
            kept.clear();
        }
        if throttle() || waits_for_completion(&batch.bufs) {
            cp.finish();
        }
        let _g = q.pending.lock();
        let d = q.done.fetch_add(1, Ordering::SeqCst) + 1;
        q.done_cv.notify_all();
        if trace_submit() {
            crate::info!("submit trace: queue {:#x} #{d} completed ({} ms after submission)", q.id, batch.at.elapsed().as_millis());
        }
    }
}

/// A submission kept while the graphics command processor records past a
/// label wait (its number, command buffers and arrival).
struct Kept {
    index: u64,
    bufs: Vec<(u64, u32)>,
    at: std::time::Instant,
}

/// Runs a kept submission again, from buffer `from.0` at address `from.1`
/// (a stretch past a label wait was dropped), or whole.
fn run_kept(cp: &mut cp::Cp, k: &Kept, from: Option<(usize, u64)>, name: &str) {
    cp.submitted_at = Some(k.at);
    cp.sub_index = k.index;
    cp.new_submission();
    let nbufs = k.bufs.len();
    let (mut i, mut at) = match from {
        Some((b, a)) => (b, Some(a)),
        None => (0, None),
    };
    while i < nbufs {
        let (addr, size) = k.bufs[i];
        cp.next_in_submission = i + 1 < nbufs;
        cp.cur_buf = i;
        match at.take() {
            Some(a) if a >= addr && a < addr + 4 * size as u64 => cp.run(a, size - ((a - addr) / 4) as u32),
            Some(a) => {
                crate::error!("gpu[{name}]: rewind to {a:#x} outside its command buffer {addr:#x}");
                cp.run(addr, size)
            }
            None => cp.run(addr, size),
        }
        if cp.rewind_sub_pending() {
            return;
        }
        if let Some((j, a)) = cp.take_rewind() {
            i = j;
            at = Some(a);
            continue;
        }
        i += 1;
    }
}

/// Runs a title submission's command buffers.
fn run_submission(cp: &mut cp::Cp, batch: &Submission, name: &str) {
    cp.submitted_at = Some(batch.at);
    cp.sub_index = batch.index;
    cp.new_submission();
    let nbufs = batch.bufs.len();
    // (Recording past a parallel run may go back to an earlier buffer of
    // the submission: `Cp::take_rewind`.)
    let mut i = 0usize;
    let mut from: Option<u64> = None;
    {
        while i < nbufs {
            let (addr, size) = batch.bufs[i];
            cp.next_in_submission = i + 1 < nbufs;
            cp.cur_buf = i;
            if trace_submit() && size == 277 {
                // Bink decode command buffers: their counter (dword 272) and
                // dword 159 at submission and now, and the label the first
                // packet waits for.
                let rd = |k: u64| if cp::guest_readable(addr + 4 * k, 4) { unsafe { *((addr + 4 * k) as *const u32) } } else { 0 };
                let sub = batch.copies.get(i).map(|w| (w.get(272).copied().unwrap_or(0), w.get(159).copied().unwrap_or(0))).unwrap_or((0, 0));
                let label = (rd(8) as u64) | ((rd(9) as u64) << 32);
                let lv = if cp::guest_readable(label, 8) { unsafe { *(label as *const u64) } } else { 0 };
                crate::info!("bink trace: {addr:#x} start: [272] {:#x} (submitted {:#x}) [159] {:#x} (submitted {:#x}), waits [{label:#x}] >= {:#x} (now {lv:#x}), {} ms after submission", rd(272), sub.0, rd(159), sub.1, rd(10), batch.at.elapsed().as_millis());
            }
            if let Some(&h) = batch.hashes.get(i) {
                if buffer_hash(addr, size) != h {
                    crate::warn!("gpu[{name}]: command buffer {addr:#x} ({size} dwords) changed between its submission and its execution ({} ms later)", batch.at.elapsed().as_millis());
                }
            }
            match from.take() {
                Some(a) => cp.run(a, size - ((a - addr) / 4) as u32),
                None => cp.run(addr, size),
            }
            if cp.rewind_sub_pending() {
                return;
            }
            if let Some((j, a)) = cp.take_rewind() {
                i = j;
                from = Some(a);
                continue;
            }
            if let Some(&h) = batch.hashes.get(i) {
                if buffer_hash(addr, size) != h {
                    let now = buffer_copy(addr, size);
                    let diffs: Vec<String> = batch.copies.get(i).map(|old| old.iter().zip(&now).enumerate().filter(|(_, (a, b))| a != b).take(6).map(|(k, (a, b))| format!("[{k}] {a:#x}->{b:#x}")).collect()).unwrap_or_default();
                    crate::warn!("gpu[{name}]: command buffer {addr:#x} ({size} dwords) changed during its execution ({} ms after submission): {}", batch.at.elapsed().as_millis(), diffs.join(" "));
                }
            }
            i += 1;
        }
    }
}

/// Queues command buffers (address, size in dwords) for execution on `queue_id`.
/// The graphics queue at each flip it reaches (`profile report`, since
/// `profile reset`): title submissions queued behind the flip and how long
/// the oldest has waited, ms (how far the title runs ahead of recording).
static RUN_AHEAD: parking_lot::Mutex<Vec<(u32, f32)>> = parking_lot::Mutex::new(Vec::new());

pub(crate) fn note_run_ahead() {
    let q = queue(GFX);
    let (n, age) = {
        let g = q.pending.lock();
        (g.len() as u32, g.front().map_or(0.0, |s| s.at.elapsed().as_secs_f32() * 1000.0))
    };
    let mut r = RUN_AHEAD.lock();
    if r.len() < 1 << 16 {
        r.push((n, age));
    }
}

fn run_ahead_line() -> String {
    let mut r = RUN_AHEAD.lock().clone();
    if r.is_empty() {
        return String::new();
    }
    let pct = |v: &mut Vec<f32>, q: f32| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[((v.len() - 1) as f32 * q) as usize]
    };
    let mut n: Vec<f32> = r.iter().map(|x| x.0 as f32).collect();
    let mut a: Vec<f32> = r.iter().map(|x| x.1).collect();
    let line = format!(
        "    graphics queue at its flips ({}): title submissions queued behind p10 {:.0}, p50 {:.0}, p90 {:.0}; the oldest waited p10 {:.1}, p50 {:.1}, p90 {:.1} ms\n",
        r.len(),
        pct(&mut n, 0.1),
        pct(&mut n, 0.5),
        pct(&mut n, 0.9),
        pct(&mut a, 0.1),
        pct(&mut a, 0.5),
        pct(&mut a, 0.9)
    );
    r.clear();
    line
}

pub fn submit(queue_id: u32, buffers: Vec<(u64, u32)>) {
    if buffers.is_empty() {
        return;
    }
    // A frame replay holds the title at the start and end of the frame it
    // records.
    replay::hold();
    SUBMITS.fetch_add(1, Ordering::Relaxed);
    let q = queue(queue_id);
    let hashes = if check_submit() { buffers.iter().map(|&(a, n)| buffer_hash(a, n)).collect() } else { Vec::new() };
    let copies = if check_submit() || trace_submit() { buffers.iter().map(|&(a, n)| buffer_copy(a, n)).collect() } else { Vec::new() };
    let mut g = q.pending.lock();
    if throttle() {
        let t0 = std::time::Instant::now();
        let mut waited = false;
        while q.done.load(Ordering::SeqCst) < q.submitted.load(Ordering::SeqCst) && !q.cp_waiting.load(Ordering::SeqCst) {
            waited = true;
            let left = THROTTLE_MAX.saturating_sub(t0.elapsed());
            if left.is_zero() {
                THROTTLE_TIMEOUTS.fetch_add(1, Ordering::Relaxed);
                break;
            }
            q.done_cv.wait_for(&mut g, left.min(std::time::Duration::from_millis(5)));
        }
        if waited {
            THROTTLE_WAITS.fetch_add(1, Ordering::Relaxed);
            THROTTLE_NS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
    }
    let index = q.submitted.fetch_add(1, Ordering::SeqCst) + 1;
    if trace_submit() {
        let (d, w, o) = buffers.iter().map(|&(a, n)| packet_mix(a, n)).fold((0, 0, 0), |x, y| (x.0 + y.0, x.1 + y.1, x.2 + y.2));
        crate::info!("submit trace: queue {queue_id:#x} #{index} {:#x}+{} (done {}) dispatches {d} draws {w} other {o}", buffers[0].0, buffers[0].1, q.done.load(Ordering::SeqCst));
        // Bink decodes (they start with a "BinkGPU" marker): list the
        // packets of the first one and of ones with other sizes.
        static LISTED: AtomicU64 = AtomicU64::new(0);
        if let Some(w) = copies.first() {
            let bink = w.len() > 3 && w[1] == crate::hle::libs::agc_cmd::DS_MAGIC && w[2] == crate::hle::libs::agc_cmd::pop::PUSH_MARKER && w.get(4) == Some(&u32::from_le_bytes(*b"Bink"));
            if bink && (w.len() != 277 || LISTED.load(Ordering::Relaxed) == 0) && LISTED.fetch_add(1, Ordering::Relaxed) < 10 {
                crate::info!("submit trace: #{index} packets: {}", pm4_summary(w));
            }
        }
    }
    // In queue order (under the queue's lock): the replay keeps each queue's.
    replay::note_submitted(queue_id, &buffers);
    g.push_back(Submission { bufs: buffers, hashes, copies, at: std::time::Instant::now(), ctl: None, index });
    q.cv.notify_one();
    // DS_SUBMIT_SYNC=small (up to SMALL_SUBMIT dwords) or =1 (all),
    // diagnostic: the submit returns once its work completed (no CPU/GPU
    // overlap for those submissions; about halves the frame rate). Bounded,
    // and not while the CP polls a label that may come from the submitter.
    if waits_for_completion(&g.back().map(|b| b.bufs.clone()).unwrap_or_default()) {
        let t0 = std::time::Instant::now();
        while q.done.load(Ordering::SeqCst) < index && !q.cp_waiting.load(Ordering::SeqCst) && t0.elapsed() < THROTTLE_MAX {
            q.done_cv.wait_for(&mut g, std::time::Duration::from_millis(5));
        }
    }
    drop(g);
    q.cv.notify_one();
    if queue_id == GFX {
        frameahead::submitted();
    }
}

/// `DS_PROFILE_GPU=1`: samples the GPU queue threads every 10 ms; the most
/// frequent host stacks (top 3 frames) are listed in hang reports.
static PROFILE: Mutex<Option<std::collections::HashMap<String, u64>>> = Mutex::new(None);

/// Inclusive samples per (queue kind, host function) and samples per queue
/// kind ("gfx" / "acb"): the share of time a function was on the stack.
static PROFILE_FN: Mutex<Option<(std::collections::HashMap<(&'static str, String), u64>, std::collections::HashMap<&'static str, u64>)>> = Mutex::new(None);

/// Samples of a queue kind not waiting or sleeping, by the innermost function
/// of ours on the stack: where command processing spends its CPU time.
static PROFILE_BUSY: Mutex<Option<std::collections::HashMap<(&'static str, String), u64>>> = Mutex::new(None);

/// Samples of a queue kind blocked on a mutex (parking_lot `lock_slow` or an
/// SRW lock), by the innermost function of ours on the stack: who waits for
/// which lock.
static PROFILE_LOCKS: Mutex<Option<std::collections::HashMap<(&'static str, String), u64>>> = Mutex::new(None);

/// Busy samples by the function our innermost function was calling (another
/// crate's, the runtime library's, the driver's: "module!name") and by the
/// innermost frame of all: where per-command time goes below our code.
static PROFILE_CALLS: Mutex<Option<std::collections::HashMap<(&'static str, String), u64>>> = Mutex::new(None);
/// Busy samples by the source line our innermost function was at (its own
/// work or the call it was making): where per-command time goes within a
/// function.
static PROFILE_LINES: Mutex<Option<std::collections::HashMap<(&'static str, String), u64>>> = Mutex::new(None);
static PROFILE_LEAF: Mutex<Option<std::collections::HashMap<(&'static str, String), u64>>> = Mutex::new(None);

/// Threads sampled besides the queues' command processors: (kind, host tid)
/// ("par": parallel recording workers, "prep": stage preparation workers).
static PROFILE_THREADS: Mutex<Vec<(&'static str, u32)>> = Mutex::new(Vec::new());

/// Samples the calling thread as `kind` while profiling (`DS_PROFILE_GPU`).
pub fn profile_this_thread(kind: &'static str) {
    crate::mem::set_ww_gated(matches!(kind, "par" | "scrub" | "ahead"));
    let tid = unsafe { windows_sys::Win32::System::Threading::GetCurrentThreadId() };
    PROFILE_THREADS.lock().push((kind, tid));
}

/// The thread sampler's period in ms (`DS_PROFILE_MS`).
static PROFILE_MS: AtomicU64 = AtomicU64::new(10);

/// Kinds of sampled threads, in report order.
const PROFILE_KINDS: [&str; 5] = ["gfx", "acb", "par", "prep", "submit"];

/// Starts the 10 ms thread sampler: `DS_PROFILE_GPU=1` (with the per-command
/// timers and statistics) or `DS_PROFILE_GPU=2` (the sampler alone: function
/// profiles of an otherwise unprofiled run, whose timers and statistics
/// would show up in them).
pub fn start_profiler() {
    if !matches!(std::env::var("DS_PROFILE_GPU").ok().as_deref(), Some("1" | "2")) {
        return;
    }
    // `DS_PROFILE_MS=<ms>`: the sampling period (default 10).
    PROFILE_MS.store(std::env::var("DS_PROFILE_MS").ok().and_then(|v| v.parse::<u64>().ok()).filter(|&ms| ms > 0).unwrap_or(10), Ordering::Relaxed);
    *PROFILE.lock() = Some(std::collections::HashMap::new());
    *PROFILE_FN.lock() = Some(Default::default());
    *PROFILE_BUSY.lock() = Some(Default::default());
    *PROFILE_LOCKS.lock() = Some(Default::default());
    *PROFILE_CALLS.lock() = Some(Default::default());
    *PROFILE_LINES.lock() = Some(Default::default());
    *PROFILE_LEAF.lock() = Some(Default::default());
    let _ = crate::thread::spawn_host(
        "gpu-profiler",
        1 << 20,
        Box::new(|| {
            crate::log::set_thread_name("gpu-profiler");
            let period = PROFILE_MS.load(Ordering::Relaxed);
            loop {
                std::thread::sleep(std::time::Duration::from_millis(period));
                // `DS_PROFILE_ACB=<text>`: only the compute queues whose name
                // (`acb0x50`) holds the text are sampled.
                static ACB_ONLY: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
                let only = ACB_ONLY.get_or_init(|| std::env::var("DS_PROFILE_ACB").ok());
                let mut targets: Vec<(&'static str, String, u32)> = QUEUES
                    .lock()
                    .iter()
                    .map(|q| if q.id == GFX { ("gfx", "gfx".to_string(), q.tid.load(Ordering::Relaxed)) } else { ("acb", format!("acb{:#x}", q.id), q.tid.load(Ordering::Relaxed)) })
                    .filter(|t| t.0 != "acb" || only.as_ref().is_none_or(|o| t.1.contains(o.as_str())))
                    .collect();
                targets.extend(PROFILE_THREADS.lock().iter().map(|&(kind, tid)| (kind, kind.to_string(), tid)));
                for (kind, name, tid) in targets {
                    let Some((frames, _)) = crate::exception::sample_frames(tid, 0) else { continue };
                    let key: Vec<String> = frames.iter().take(4).map(|&a| crate::exception::host_symbol(a)).collect();
                    let k = format!("{name}: {}", key.join(" <- "));
                    if let Some(m) = PROFILE.lock().as_mut() {
                        *m.entry(k).or_default() += 1;
                    }
                    let labels: Vec<String> = frames.iter().map(|&a| crate::exception::frame_label(a)).collect();
                    let mut fns: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();
                    let waiting = fns.iter().any(|f| f.contains("Wait") || f.contains("Sleep") || f.contains("DelayExecution") || f.contains("SetTimer") || f.contains("CreateWaitableTimer"));
                    let is_ours = |f: &str| (f.starts_with("dsport::") || f.starts_with("rdna::")) && !f.contains("parking_lot");
                    let at = fns.iter().position(|f| is_ours(f));
                    let ours = at.map_or("(none)", |i| fns[i]);
                    if !waiting {
                        if let Some(m) = PROFILE_BUSY.lock().as_mut() {
                            *m.entry((kind, ours.to_string())).or_default() += 1;
                        }
                        // What our innermost function was calling (itself when
                        // it was running its own code).
                        let callee = match at {
                            Some(i) if i > 0 => fns[i - 1],
                            Some(i) => fns[i],
                            None => fns.last().copied().unwrap_or("(none)"),
                        };
                        if let Some(m) = PROFILE_CALLS.lock().as_mut() {
                            *m.entry((kind, format!("{callee} <- {ours}"))).or_default() += 1;
                        }
                        // The line our innermost function was at (a return
                        // address less one: the call's line).
                        if let Some(i) = at {
                            if let Some(line) = crate::exception::host_line(frames[i].saturating_sub((i > 0) as u64)) {
                                if let Some(m) = PROFILE_LINES.lock().as_mut() {
                                    *m.entry((kind, format!("{line} {}", ours.rsplit("::").next().unwrap_or(ours)))).or_default() += 1;
                                }
                            }
                        }
                        if let Some(m) = PROFILE_LEAF.lock().as_mut() {
                            *m.entry((kind, fns.first().copied().unwrap_or("(none)").to_string())).or_default() += 1;
                        }
                    } else if fns.iter().any(|f| f.contains("lock_slow") || f.contains("AcquireSRWLock")) {
                        if let Some(m) = PROFILE_LOCKS.lock().as_mut() {
                            *m.entry((kind, ours.to_string())).or_default() += 1;
                        }
                    }
                    fns.sort_unstable();
                    fns.dedup();
                    if let Some((m, n)) = PROFILE_FN.lock().as_mut() {
                        *n.entry(kind).or_default() += 1;
                        for f in fns {
                            *m.entry((kind, f.to_string())).or_default() += 1;
                        }
                    }
                }
            }
        }),
    );
}

/// Clears the samples and dispatch timings (`profile reset` control command).
pub fn reset_profile() {
    cp::reset_classes();
    if let Some(m) = PROFILE.lock().as_mut() {
        m.clear();
    }
    if let Some((m, n)) = PROFILE_FN.lock().as_mut() {
        m.clear();
        n.clear();
    }
    if let Some(m) = PROFILE_BUSY.lock().as_mut() {
        m.clear();
    }
    if let Some(m) = PROFILE_LOCKS.lock().as_mut() {
        m.clear();
    }
    for p in [&PROFILE_CALLS, &PROFILE_LEAF, &PROFILE_LINES] {
        if let Some(m) = p.lock().as_mut() {
            m.clear();
        }
    }
    exec::reset_shader_profile();
    exec::reset_batch_misses();
    cp::reset_wait_stats();
    par::reset_stats();
    cp::past_wait_reset();
    lockstat::reset_stats();
    images::reset_memo_stats();
    exec::reset_pre_window();
    exec::reset_import_window();
    xwait::reset();
    census::reset();
    smalltrack::reset();
    RUN_AHEAD.lock().clear();
    frameprobe::reset();
    frameahead::reset();
    runprobe::reset();
    crate::mem::reset_host_calls();
    threadcpu::reset_window();
    crate::audio::reset_window();
    crate::guestprof::reset();
}

/// The profile so far (`profile report` control command).
pub fn profile_report() -> String {
    let mut s = profile_summary();
    s.push_str(&exec::shader_profile());
    s.push_str(&drawcensus::lines(exec::profile_flips()));
    s.push_str(&exec::laps_report());
    s.push_str(&cp::wait_stats());
    s.push_str(&cp::classes_line());
    s.push_str(&exec::race_line());
    s.push_str(&forward::status_line());
    s.push_str(&crate::hle::kernel::timeouts_line());
    s.push_str(&cp::long_waits_line());
    s.push_str(&cp::past_wait_status());
    s.push_str(&par::status());
    s.push_str(&images::memo_stats_line());
    s.push_str(&exec::pre_window_line());
    if !exec::profiling_on() {
        s.push_str(&shaders::prep_cache_line());
        // (With DS_PROFILE_GPU=1 the shader profile has them.)
        s.push_str(&crate::mem::ww_lines());
    }
    s.push_str(&exec::push_line());
    s.push_str(&exec::async_line());
    s.push_str(&exec::import_window_line());
    s.push_str(&crate::mem::host_calls_line());
    s.push_str(&xwait::status());
    if census::on() {
        s.push_str(&census::status());
    }
    if smalltrack::on() {
        s.push_str(&smalltrack::report());
    }
    s.push_str(&crate::audio::window_line());
    s.push_str(&crate::hle::libs::acm_conv::status());
    s.push_str(&crate::guestprof::report());
    s.push_str(&threadcpu::window_line());
    s.push_str(&run_ahead_line());
    if frameprobe::on() {
        s.push_str(&frameprobe::report());
    }
    s.push_str(&frameahead::status());
    if runprobe::on() {
        s.push_str(&runprobe::report());
    }
    s
}

fn profile_summary() -> String {
    let g = PROFILE.lock();
    let Some(m) = g.as_ref() else { return String::new() };
    let mut v: Vec<(&String, &u64)> = m.iter().collect();
    v.sort_by(|a, b| b.1.cmp(a.1));
    let mut s = format!("    gpu profile ({} ms samples):\n", PROFILE_MS.load(Ordering::Relaxed));
    for (k, n) in v.iter().take(20) {
        s.push_str(&format!("      {n:6}  {k}\n"));
    }
    drop(g);
    // Inclusive time per function of our crate (and the waits), per queue kind.
    if let Some((m, totals)) = PROFILE_FN.lock().as_ref() {
        let top = std::env::var("DS_PROFILE_FN_TOP").ok().and_then(|x| x.parse().ok()).unwrap_or(40);
        for kind in PROFILE_KINDS {
            let total = totals.get(kind).copied().unwrap_or(0).max(1);
            let mut v: Vec<(&String, &u64)> = m.iter().filter(|((k, f), _)| *k == kind && (f.starts_with("dsport::") || f.contains("Wait") || f.contains("Sleep") || f.contains("Submit"))).map(|((_, f), n)| (f, n)).collect();
            v.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
            s.push_str(&format!("    gpu profile {kind}: {total} samples, inclusive share per function:\n"));
            for (f, n) in v.iter().take(top) {
                s.push_str(&format!("      {:5.1}%  {f}\n", **n as f64 * 100.0 / total as f64));
            }
        }
    }
    if let Some(m) = PROFILE_BUSY.lock().as_ref() {
        for kind in PROFILE_KINDS {
            let mut v: Vec<(&String, &u64)> = m.iter().filter(|((k, _), _)| *k == kind).map(|((_, f), n)| (f, n)).collect();
            let busy: u64 = v.iter().map(|x| *x.1).sum();
            v.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
            s.push_str(&format!("    gpu profile {kind} busy (not waiting): {busy} samples, by innermost function of ours:\n"));
            for (f, n) in v.iter().take(30) {
                s.push_str(&format!("      {:5.1}%  {f}\n", **n as f64 * 100.0 / busy.max(1) as f64));
            }
        }
    }
    for (table, what, n) in [(&PROFILE_CALLS, "by callee of our innermost function", 40), (&PROFILE_LEAF, "by innermost frame", 25), (&PROFILE_LINES, "by source line of our innermost function", 50)] {
        if let Some(m) = table.lock().as_ref() {
            for kind in PROFILE_KINDS {
                let mut v: Vec<(&String, &u64)> = m.iter().filter(|((k, _), _)| *k == kind).map(|((_, f), n)| (f, n)).collect();
                let busy: u64 = v.iter().map(|x| *x.1).sum();
                if busy == 0 {
                    continue;
                }
                v.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
                s.push_str(&format!("    gpu profile {kind} busy (not waiting): {busy} samples, {what}:\n"));
                for (f, c) in v.iter().take(n) {
                    s.push_str(&format!("      {:5.1}%  {f}\n", **c as f64 * 100.0 / busy as f64));
                }
            }
        }
    }
    if let Some(m) = PROFILE_LOCKS.lock().as_ref() {
        for kind in PROFILE_KINDS {
            let mut v: Vec<(&String, &u64)> = m.iter().filter(|((k, _), _)| *k == kind).map(|((_, f), n)| (f, n)).collect();
            let total: u64 = v.iter().map(|x| *x.1).sum();
            v.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
            s.push_str(&format!("    gpu profile {kind} lock waits: {total} samples, by innermost function of ours:\n"));
            for (f, n) in v.iter().take(15) {
                s.push_str(&format!("      {n:6}  {f}\n"));
            }
        }
    }
    s
}

/// One line per queue for hang reports.
pub fn status() -> String {
    status_ex(false)
}

/// Queue lines, with host stack samples of the command processors.
pub fn status_ex(with_stacks: bool) -> String {
    let mut s = String::new();
    let qs: Vec<&'static Queue> = match QUEUES.try_lock() {
        Some(g) => g.clone(),
        None => return s,
    };
    for q in qs {
        let pending = q.pending.try_lock().map(|g| g.len().to_string()).unwrap_or_else(|| "?".into());
        let name = if q.id == GFX { "gpu-gfx".to_string() } else { format!("gpu-acb{:#x}", q.id) };
        let state = cp::STATE.try_lock().and_then(|g| g.iter().find(|(n, _)| *n == name).map(|(_, s)| s.clone())).unwrap_or_else(|| "idle".into());
        s.push_str(&format!("    {name}: {} submission(s) done, {pending} pending; {state}\n", q.done.load(Ordering::Relaxed)));
        if with_stacks {
            if let Some(st) = crate::exception::sample_thread(q.tid.load(Ordering::Relaxed), 16) {
                s.push_str(&format!("        {st}\n"));
            }
        }
    }
    s.push_str(&format!(
        "    gpu totals: {} submits, {} packets, {} draws ({} from {} multi-draw packets, {} of them with a GPU-written count), {} dispatches, {} EOP interrupts, {} flips, {} label waits latched, {} passed on writes made after their submission, {} on counters past their value, {} passed on promises, {} CP reads waited for another queue\n",
        SUBMITS.load(Ordering::Relaxed),
        cp::PACKETS.load(Ordering::Relaxed),
        cp::DRAWS.load(Ordering::Relaxed),
        cp::MDI_DRAWS.load(Ordering::Relaxed),
        cp::MDI_PACKETS.load(Ordering::Relaxed),
        cp::MDI_COUNT_FROM_MEMORY.load(Ordering::Relaxed),
        cp::DISPATCHES.load(Ordering::Relaxed),
        cp::EOP_INTERRUPTS.load(Ordering::Relaxed),
        FRAMES.load(Ordering::Relaxed),
        cp::LATCHED.load(Ordering::Relaxed),
        cp::MISSED.load(Ordering::Relaxed),
        cp::COUNTER_PASSED.load(Ordering::Relaxed),
        cp::PROMISED.load(Ordering::Relaxed),
        exec::FOREIGN_WAITS.load(Ordering::Relaxed)
    ));
    s.push_str(&format!("    {}\n", throttle_status()));
    s.push_str(&exec::status());
    s.push_str(&pagetable::status());
    s.push_str(&gpures::status());
    s.push_str(&gmirror::status());
    s.push_str(&profile_summary());
    s.push_str(&exec::shader_profile());
    s
}
