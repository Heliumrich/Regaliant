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

    static ALL: [Tunable; 49] = [
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
        Tunable { name: "prep_cache", env: "DS_PREP_CACHE", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "map_ranges", env: "DS_MAP_RANGES", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "ahead", env: "DS_AHEAD", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "pre_cs", env: "DS_PRE_CS", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "par", env: "DS_PAR", default: 5, value: AtomicU64::new(UNSET) },
        Tunable { name: "pre_share", env: "DS_PRE_SHARE", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "pending_set", env: "DS_PENDING_SET", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "tex_scrub", env: "DS_TEX_SCRUB", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "pre_share_min", env: "DS_PRE_SHARE_MIN", default: 16384, value: AtomicU64::new(UNSET) },
        Tunable { name: "pre_share_mib", env: "DS_PRE_SHARE_MIB", default: 1024, value: AtomicU64::new(UNSET) },
        Tunable { name: "pre_share_volatile", env: "DS_PRE_SHARE_VOLATILE", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "spec_reads", env: "DS_SPEC_READS", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "submit_thread", env: "DS_SUBMIT_THREAD", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "par_balance", env: "DS_PAR_BALANCE", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "pre_inflight", env: "DS_PRE_INFLIGHT", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "idle_split", env: "DS_IDLE_SPLIT", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "push_desc", env: "DS_PUSH_DESC", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "pre_merge", env: "DS_PRE_MERGE", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "lean_regs", env: "DS_LEAN_REGS", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "span_cache", env: "DS_SPAN_CACHE", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "fill_dma", env: "DS_FILL_DMA", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "lean_cmd", env: "DS_LEAN_CMD", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "exact_big", env: "DS_EXACT_BIG", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "lean_notes", env: "DS_LEAN_NOTES", default: 1, value: AtomicU64::new(UNSET) },
    ];

    #[inline]
    fn get(i: usize) -> u64 {
        let t = &ALL[i];
        let v = t.value.load(Ordering::Relaxed);
        if v != UNSET {
            return v;
        }
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

    /// Translations keep each register in its own variable (1) or the
    /// register files in arrays (0, the earlier form): see
    /// `rdna::translate::Spec::reg_vars`. Switching translates programs anew.
    #[inline]
    pub fn reg_vars() -> bool {
        get(24) != 0
    }

    /// Vertex and pixel shader preparations reuse the previous one of their
    /// stage when inputs and the guest words read are unchanged (1). Off:
    /// cloning whole preparations into and out of the cache cost more than
    /// the resolution it saved (17-20 against 16.5-17 us per draw, run
    /// 20261011-092437); a form that shares the unchanging parts may pay.
    #[inline]
    pub fn prep_cache() -> bool {
        get(25) != 0
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
    /// synchronization-free runs in parallel (`gpu::par`); 0: off.
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

    /// Uploaded textures sampled for CPU writes by a background thread
    /// (`images::scrubbing`); 0: by the first draw binding each per frame.
    #[inline]
    pub fn tex_scrub() -> bool {
        get(32) != 0
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
    /// are used as prepared and checked afterwards (`exec::speculate`).
    #[inline]
    pub fn spec_reads() -> bool {
        get(36) != 0
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
    /// the courtyard; the rate effect was within the noise of a loaded run
    /// (20261012-163645), so it stays off until measured on a quiet machine.
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
pub mod par;
pub mod precache;
pub mod exec;
pub mod pcache;
pub mod lodstats;
pub mod pstats;
pub mod mirror;
pub mod ranges;
pub mod homes;
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
    q.submitted.fetch_add(1, Ordering::SeqCst);
    g.push_back(Submission { bufs, hashes: Vec::new(), copies: Vec::new(), at: std::time::Instant::now(), ctl });
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
    cp.wait_flag = Some(&q.cp_waiting);
    // Register states kept for a frame replay (`Ctl::Save`).
    let mut saved: [Option<cp::ChunkState>; 2] = [None, None];
    loop {
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
        cp.submitted_at = Some(batch.at);
        cp.new_submission();
        for (i, &(addr, size)) in batch.bufs.iter().enumerate() {
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
            cp.run(addr, size);
            if let Some(&h) = batch.hashes.get(i) {
                if buffer_hash(addr, size) != h {
                    let now = buffer_copy(addr, size);
                    let diffs: Vec<String> = batch.copies.get(i).map(|old| old.iter().zip(&now).enumerate().filter(|(_, (a, b))| a != b).take(6).map(|(k, (a, b))| format!("[{k}] {a:#x}->{b:#x}")).collect()).unwrap_or_default();
                    crate::warn!("gpu[{name}]: command buffer {addr:#x} ({size} dwords) changed during its execution ({} ms after submission): {}", batch.at.elapsed().as_millis(), diffs.join(" "));
                }
            }
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

/// Queues command buffers (address, size in dwords) for execution on `queue_id`.
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
    g.push_back(Submission { bufs: buffers, hashes, copies, at: std::time::Instant::now(), ctl: None });
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
    let tid = unsafe { windows_sys::Win32::System::Threading::GetCurrentThreadId() };
    PROFILE_THREADS.lock().push((kind, tid));
}

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
            loop {
                std::thread::sleep(std::time::Duration::from_millis(10));
                let mut targets: Vec<(&'static str, String, u32)> = QUEUES
                    .lock()
                    .iter()
                    .map(|q| if q.id == GFX { ("gfx", "gfx".to_string(), q.tid.load(Ordering::Relaxed)) } else { ("acb", format!("acb{:#x}", q.id), q.tid.load(Ordering::Relaxed)) })
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
    cp::reset_wait_stats();
    par::reset_stats();
    images::reset_memo_stats();
    exec::reset_pre_window();
    threadcpu::reset_window();
}

/// The profile so far (`profile report` control command).
pub fn profile_report() -> String {
    let mut s = profile_summary();
    s.push_str(&exec::shader_profile());
    s.push_str(&cp::wait_stats());
    s.push_str(&par::status());
    s.push_str(&images::memo_stats_line());
    s.push_str(&exec::pre_window_line());
    s.push_str(&threadcpu::window_line());
    s
}

fn profile_summary() -> String {
    let g = PROFILE.lock();
    let Some(m) = g.as_ref() else { return String::new() };
    let mut v: Vec<(&String, &u64)> = m.iter().collect();
    v.sort_by(|a, b| b.1.cmp(a.1));
    let mut s = String::from("    gpu profile (10 ms samples):\n");
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
    s.push_str(&format!("    {}
", throttle_status()));
    s.push_str(&exec::status());
    s.push_str(&profile_summary());
    s.push_str(&exec::shader_profile());
    s
}
