//! Front end 3's C6 (FRONTEND3.md "C6"; `DS_C6=1`, opt-in while it is
//! built): shaders resolve their read-only buffers, and the scalar loads on
//! their chains, from the mirror where the resident set says the mirror
//! holds guest memory, else from guest memory through the page table,
//! reporting the range (`rdna::translate::Spec::c6`). This module keeps the
//! resident set: a bit per 64 KiB granule of the mirror's span on the GPU,
//! and its shadow here.
//! - Granules join from the shaders' reports (`admit_round`, after each
//!   coherence pass): outside the GPU arena they are armed first (a CPU write
//!   then faults and ends the arming; the next pass takes such pages:
//!   `hot_runs`), inside it the pass scans them anyway.
//! - A batch with C6 commands makes the resident set current itself, when
//!   it binds one (`take_cover`, from `Exec::c6_cover` on the recording
//!   thread): it waits for a coherence pass that began in the epoch its
//!   command wants (every CPU write before then was found by a take), then
//!   places every resident page front end 2 holds invalid through front end
//!   2 (`fe2_place`: uploads in its prologue, ordered against CP writes and
//!   GPU work as any binding's), and claims the granules admitted since
//!   (their pages placed with it). Their bits reach the GPU when the
//!   submission thread, in sequence order, has submitted that batch
//!   (`note_submitted`: a small update on the main queue after it); a batch
//!   before it reads them through the page table, one after it from the
//!   mirror (compute batches wait for the main queue's progress).

use super::resident::{MIRROR_BASE, PAGE};
use super::vk::{ctx, Ctx};
use ash::vk;
use parking_lot::Mutex;
use rdna::translate::{C6_FB_BYTES, C6_FB_ENTRIES, C6_GRANULE_SHIFT, C6_RES_WORDS, PT_BASE, PT_END};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

const GRANULE: u64 = 1 << C6_GRANULE_SHIFT;
const WORDS: usize = C6_RES_WORDS as usize;

/// `DS_C6=1` (read once).
pub fn wanted() -> bool {
    static W: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *W.get_or_init(|| std::env::var("DS_C6").ok().as_deref() == Some("1"))
}

/// Stages C6 resolves (`DS_C6_KINDS`: bits 1 compute, 2 vertex, 4 pixel;
/// all by default).
pub fn stage_wanted(stage: rdna::translate::Stage) -> bool {
    static K: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    let k = *K.get_or_init(|| std::env::var("DS_C6_KINDS").ok().and_then(|v| v.parse().ok()).unwrap_or(7));
    let bit = match stage {
        rdna::translate::Stage::Compute => 1,
        rdna::translate::Stage::Fragment => 4,
        _ => 2,
    };
    k & bit != 0
}

/// Whether C6 runs: wanted, front end 2 and the coherence service on, the
/// resident set made.
pub fn on() -> bool {
    wanted() && super::fe2::on() && super::kco::covering() && bufs().is_some()
}

struct Bufs {
    res: vk::Buffer,
    _res_mem: vk::DeviceMemory,
    res_va: u64,
    _fb: vk::Buffer,
    _fb_mem: vk::DeviceMemory,
    fb_va: u64,
    fb_ptr: *mut u32,
}

unsafe impl Send for Bufs {}
unsafe impl Sync for Bufs {}

static BUFS: std::sync::OnceLock<Option<Bufs>> = std::sync::OnceLock::new();

fn bufs() -> Option<&'static Bufs> {
    // (Not before Vulkan is up: front end 2's takes run from the start.)
    if !wanted() || super::vk::try_ctx().is_none() || super::resident::mirror_buffer().is_none() {
        return None;
    }
    BUFS.get_or_init(|| match make() {
        Ok(b) => {
            crate::info!(
                "c6: resident set ({} KiB at device address {:#x}) and feedback ring ({} KiB at {:#x}) made; the mirror at {:#x}, the guest page table at {:#x}",
                WORDS * 4 >> 10,
                b.res_va,
                C6_FB_BYTES >> 10,
                b.fb_va,
                super::resident::mirror_address(),
                super::pagetable::address()
            );
            Some(b)
        }
        Err(e) => {
            crate::error!("c6: {e}; C6 off");
            None
        }
    })
    .as_ref()
}

fn make() -> Result<Bufs, String> {
    let c = ctx();
    if !c.device_addresses {
        return Err("no device addresses".into());
    }
    let bytes = (WORDS * 4) as u64;
    let (res, res_mem) = c.device_buffer(bytes, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST)?;
    super::resident::one_off_now(|cmd| unsafe { c.device.cmd_fill_buffer(cmd, res, 0, bytes, 0) })?;
    let res_va = c.buffer_address(res);
    let (fb, fb_mem, fb_ptr) = c.host_buffer(C6_FB_BYTES, vk::BufferUsageFlags::STORAGE_BUFFER)?;
    unsafe { std::ptr::write_bytes(fb_ptr, 0, C6_FB_BYTES as usize) };
    let fb_va = c.buffer_address(fb);
    Ok(Bufs { res, _res_mem: res_mem, res_va, _fb: fb, _fb_mem: fb_mem, fb_va, fb_ptr: fb_ptr as *mut u32 })
}

/// The address block header's C6 words (`rdna::translate::C6_HDR_*`): the
/// mirror's device address of guest PT_BASE, the resident set's, the
/// feedback ring's.
pub fn header() -> [u32; 6] {
    let Some(b) = bufs() else { return [0; 6] };
    let m = super::resident::mirror_address() + (PT_BASE - MIRROR_BASE);
    [m as u32, (m >> 32) as u32, b.res_va as u32, (b.res_va >> 32) as u32, b.fb_va as u32, (b.fb_va >> 32) as u32]
}

/// The resident set as the batches submitted so far leave it.
static SHADOW: [AtomicU32; WORDS] = [const { AtomicU32::new(0) }; WORDS];
/// Granules admitted, waiting for the copies that make them resident.
static ADMIT: Mutex<Vec<u32>> = Mutex::new(Vec::new());
/// Resident pages front end 2 holds invalid (a take found them written,
/// guest memory changed without a write mark, or an admitted granule's
/// page it never held): every batch with C6 commands places them
/// (`take_cover`); they leave the set when front end 2 validates them (an
/// uploading batch queued: `note_validated`).
static INVALID: Mutex<std::collections::BTreeSet<u64>> = Mutex::new(std::collections::BTreeSet::new());
/// The feedback ring's entries read so far.
static CURSOR: Mutex<u64> = Mutex::new(0);

/// [reports read, reports lost to the ring's capacity, granules admitted,
/// refused (no import window over them), refused (not armed), pages copied
/// by coverage (admissions), pages copied again (resident pages written),
/// pages left out (held or GPU-written in flight), coverages, ns of
/// coverage, waits for a pass, ns waited, granules gone, hot pages taken,
/// reports by kind 1, 2, 3].
pub static STATS: [AtomicU64; 17] = [const { AtomicU64::new(0) }; 17];

fn granule_of(a: u64) -> Option<u32> {
    (PT_BASE..PT_END).contains(&a).then(|| ((a - PT_BASE) >> C6_GRANULE_SHIFT) as u32)
}

/// Whether granule `g` is in the resident set (as submitted).
pub fn resident(g: u32) -> bool {
    SHADOW.get((g >> 5) as usize).is_some_and(|w| w.load(Ordering::Acquire) >> (g & 31) & 1 != 0)
}

fn resident_page(p: u64) -> bool {
    granule_of(p).is_some_and(|g| resident(g) || CLAIMED.lock().contains(&g))
}

/// Front end 2's takes found these pages written (`fe2::cpu_written`):
/// resident ones are copied again before the next batch.
pub fn note_invalidated(pages: &[u64]) {
    if !on() {
        return;
    }
    let v: Vec<u64> = pages.iter().copied().filter(|&p| resident_page(p)).collect();
    if !v.is_empty() {
        INVALID.lock().extend(v);
    }
}

/// Front end 2 validated guest [a, b) (an uploading batch queued:
/// `fe2::uploaded`).
pub fn note_validated(a: u64, b: u64) {
    if !on() {
        return;
    }
    let mut g = INVALID.lock();
    if g.is_empty() {
        return;
    }
    let gone: Vec<u64> = g.range(a..b).copied().collect();
    for p in gone {
        g.remove(&p);
    }
}

/// Guest [base, base+len) changed without a write mark (file reads, CP
/// writes the CPU performs, mapping changes: `fe2::invalidate`): its
/// resident pages are copied again (or leave the set when no import window
/// holds them then).
pub fn note_invalidated_range(base: u64, len: u64) {
    if !on() || len == 0 {
        return;
    }
    let (Some(g0), Some(g1)) = (granule_of(base), granule_of(base + len - 1)) else { return };
    if !(g0..=g1).any(resident) {
        return;
    }
    let mut v = Vec::new();
    let mut p = base & !(PAGE - 1);
    while p < base + len {
        if resident_page(p) {
            v.push(p);
        }
        p += PAGE;
    }
    INVALID.lock().extend(v);
}

/// The mirror's backing over [base, base+len) is dropped (`resident`'s
/// unbacking): the granules leave the set.
pub fn note_gone(base: u64, len: u64) {
    if !on() || len == 0 {
        return;
    }
    // (Not called: the mirror keeps resident granules backed.)
    let _ = (base, len);
}

/// Whether any granule of [base, base+len) is resident (the mirror must
/// keep its backing).
pub fn any_resident(base: u64, len: u64) -> bool {
    if !on() || len == 0 {
        return false;
    }
    match (granule_of(base), granule_of(base + len - 1)) {
        (Some(g0), Some(g1)) => (g0..=g1).any(resident),
        _ => false,
    }
}

/// Granules whose arming failed, and the round they may be tried again.
static REFUSED: Mutex<std::collections::BTreeMap<u32, u64>> = Mutex::new(std::collections::BTreeMap::new());
/// `admit_round`s so far.
static ROUNDS: AtomicU64 = AtomicU64::new(0);
/// Resident granules outside the arena found not armed by the last pass.
static HOT: Mutex<Vec<u32>> = Mutex::new(Vec::new());

/// Granules front end 2's CPU path placed for reading, to join the set
/// (`note_placed`; at most PLACED_MAX waiting).
static PLACED: Mutex<Vec<u32>> = Mutex::new(Vec::new());
const PLACED_MAX: usize = 1 << 16;

/// Front end 2 placed guest [lo, hi) for reading (`Exec::fe2_place`): its
/// granules join the resident set (ranges up to 256 MiB), so the shaders
/// resolving their own buffers find what the CPU path reads.
pub fn note_placed(lo: u64, hi: u64) {
    if !on() || hi <= lo || hi - lo > 256 << 20 {
        return;
    }
    let (Some(g0), Some(g1)) = (granule_of(lo), granule_of(hi - 1)) else { return };
    if (g0..=g1).all(resident) {
        return;
    }
    let mut p = PLACED.lock();
    if p.len() < PLACED_MAX {
        p.extend((g0..=g1).filter(|&g| !resident(g)));
    }
}

/// Reads the shaders' reports: the granules they read through the page
/// table, deduplicated into `out`.
fn drain_feedback(out: &mut Vec<u32>) {
    let Some(b) = bufs() else { return };
    let mut cur = CURSOR.lock();
    let count = unsafe { std::ptr::read_volatile(b.fb_ptr) } as u64;
    // (The counter wraps at 2^32: entries are read modulo the ring.)
    let mut c32 = *cur as u32;
    let lost = count.wrapping_sub(c32 as u64) as u32;
    if lost > C6_FB_ENTRIES {
        STATS[1].fetch_add((lost - C6_FB_ENTRIES) as u64, Ordering::Relaxed);
        c32 = (count as u32).wrapping_sub(C6_FB_ENTRIES);
    }
    while c32 != count as u32 {
        let slot = (c32 & (C6_FB_ENTRIES - 1)) as usize;
        let e = unsafe { b.fb_ptr.add(4 + 4 * slot) };
        let kind = unsafe { std::ptr::read_volatile(e.add(3)) };
        if kind == 0 {
            // (Not written yet: read next time.)
            break;
        }
        let a = unsafe { std::ptr::read_volatile(e) } as u64 | (unsafe { std::ptr::read_volatile(e.add(1)) } as u64) << 32;
        let n = unsafe { std::ptr::read_volatile(e.add(2)) } as u64;
        unsafe { std::ptr::write_volatile(e.add(3), 0) };
        STATS[0].fetch_add(1, Ordering::Relaxed);
        if (1..=3).contains(&kind) {
            STATS[13 + kind as usize].fetch_add(1, Ordering::Relaxed);
        }
        if n > 0 && n <= 256 << 20 {
            if let (Some(g0), Some(g1)) = (granule_of(a), granule_of(a + n - 1)) {
                out.extend(g0..=g1);
            }
        }
        c32 = c32.wrapping_add(1);
    }
    *cur = c32 as u64;
    out.sort_unstable();
    out.dedup();
}

/// After a coherence pass (`kco`): the reported granules are admitted
/// (outside the GPU arena armed through the host first, and taken; a granule
/// with any page not armed then, or without one import window over it, is
/// left out until it is reported again).
pub fn admit_round() {
    if !on() {
        return;
    }
    let mut want = Vec::new();
    drain_feedback(&mut want);
    want.append(&mut PLACED.lock());
    want.sort_unstable();
    want.dedup();
    if want.is_empty() {
        return;
    }
    let arena = crate::engine::arena_owner();
    let pass = ROUNDS.fetch_add(1, Ordering::Relaxed);
    // Resident granules a write took out of the arming last pass: armed
    // again (the next write faults again; a hot one stays taken each pass).
    let hot = std::mem::take(&mut *HOT.lock());
    for g in hot.into_iter().filter(|g| REFUSED.lock().get(g).is_none_or(|&until| pass >= until)).take(256) {
        let a = PT_BASE + ((g as u64) << C6_GRANULE_SHIFT);
        if !super::fe2::arm_granule(a, a + GRANULE) {
            // (Its window not settled yet: tried again 32 rounds on.)
            REFUSED.lock().insert(g, pass + 32);
        }
    }
    let pending: std::collections::HashSet<u32> = ADMIT.lock().iter().copied().collect();
    let mut ok = Vec::new();
    for g in want {
        if resident(g) || pending.contains(&g) || CLAIMED.lock().contains(&g) {
            continue;
        }
        let a = PT_BASE + ((g as u64) << C6_GRANULE_SHIFT);
        // (Not imported here: front end 2 imports from its recording
        // threads, in order with its holds.)
        if !super::exec::import_covers(a, GRANULE) {
            STATS[3].fetch_add(1, Ordering::Relaxed);
            continue;
        }
        // (Outside the arena armed where front end 2 may arm: inside import
        // windows an upload used SETTLE flips ago (arming a younger one raced
        // its import: the device lost to 10 s GPU hangs at level loads,
        // 2026-11-10 morning). Else admitted unarmed: each pass takes it
        // (`hot_runs`) until a later round arms it, its first cover's upload
        // making its window used.)
        if arena == 0 || crate::engine::owner(a) != arena {
            if !super::fe2::arm_granule(a, a + GRANULE) {
                STATS[4].fetch_add(1, Ordering::Relaxed);
            }
        }
        // (The mirror's backing for it: copies into unbacked pages vanish.)
        if super::resident::back(a, GRANULE).is_err() {
            STATS[3].fetch_add(1, Ordering::Relaxed);
            continue;
        }
        ok.push(g);
    }
    if !ok.is_empty() {
        ADMIT.lock().extend(ok);
    }
}

/// Resident pages outside the GPU arena that are not armed now (a write
/// ended their arming, or front end 4 released them): the coherence pass
/// takes them, as runs into `out`.
pub fn hot_runs(out: &mut Vec<(u64, u64)>) {
    out.clear();
    if !on() {
        return;
    }
    let arena = crate::engine::arena_owner();
    let mut hot = Vec::new();
    for (w, word) in SHADOW.iter().enumerate() {
        let mut bits = word.load(Ordering::Acquire);
        while bits != 0 {
            let k = bits.trailing_zeros();
            bits &= bits - 1;
            let g = (w as u32) << 5 | k;
            let a = PT_BASE + ((g as u64) << C6_GRANULE_SHIFT);
            if arena != 0 && crate::engine::owner(a) == arena {
                continue;
            }
            if super::fe2::granule_armed(a, a + GRANULE) {
                continue;
            }
            hot.push(g);
            match out.last_mut() {
                Some(l) if l.1 == a => l.1 = a + GRANULE,
                _ => out.push((a, a + GRANULE)),
            }
        }
    }
    let n: u64 = out.iter().map(|r| (r.1 - r.0) / PAGE).sum();
    STATS[13].fetch_add(n, Ordering::Relaxed);
    *HOT.lock() = hot;
}

/// What a batch with C6 commands must do for its want (`take_cover`): the
/// runs of resident or claimed pages to place (uploaded by its prologue
/// when invalid), and the granules it claimed (their bits set once it is
/// submitted: `note_submitted`).
pub struct Cover {
    pub runs: Vec<(u64, u64)>,
    pub claims: Vec<u32>,
}

/// Granules claimed by batches not submitted yet: their invalidations are
/// tracked as resident pages' are.
static CLAIMED: Mutex<std::collections::BTreeSet<u32>> = Mutex::new(std::collections::BTreeSet::new());

/// For a batch whose C6 command wants epoch `want` (on its recording
/// thread): waits for a coherence pass that began in that epoch or later,
/// claims the granules admitted since (their bits set in the shadow now:
/// their later invalidations are tracked; on the GPU by this batch's
/// prologue), and lists every resident page front end 2 holds invalid.
pub fn take_cover(want: u32) -> Cover {
    let mut out = Cover { runs: Vec::new(), claims: Vec::new() };
    if !on() {
        return out;
    }
    if super::kco::covered_epoch() < want {
        let t = std::time::Instant::now();
        STATS[10].fetch_add(1, Ordering::Relaxed);
        super::kco::trigger();
        while super::kco::covered_epoch() < want {
            if t.elapsed() > std::time::Duration::from_secs(5) {
                crate::warn!("c6: no coherence pass for epoch {want} within 5 s (last {})", super::kco::covered_epoch());
                break;
            }
            std::thread::sleep(std::time::Duration::from_micros(50));
        }
        STATS[11].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
    let admits = std::mem::take(&mut *ADMIT.lock());
    let mut inv = Vec::new();
    // (Claimed first, then their pages listed: an invalidation between the
    // two is tracked.)
    CLAIMED.lock().extend(admits.iter().copied());
    for &g in &admits {
        let a = PT_BASE + ((g as u64) << C6_GRANULE_SHIFT);
        inv.extend((0..GRANULE / PAGE).map(|k| a + k * PAGE).filter(|&p| !super::fe2::page_valid(p)));
    }
    let pages: Vec<u64> = {
        let mut g = INVALID.lock();
        g.extend(inv);
        g.iter().copied().collect()
    };
    for p in pages {
        match out.runs.last_mut() {
            Some(l) if l.1 == p => l.1 = p + PAGE,
            _ => out.runs.push((p, p + PAGE)),
        }
    }
    STATS[5].fetch_add(admits.len() as u64 * (GRANULE / PAGE), Ordering::Relaxed);
    STATS[6].fetch_add(out.runs.iter().map(|r| (r.1 - r.0) / PAGE).sum(), Ordering::Relaxed);
    STATS[8].fetch_add(1, Ordering::Relaxed);
    out.claims = admits;
    out
}

/// The set's updates in flight: (command buffer, fence), freed once done.
static UPDATES: Mutex<Option<(vk::CommandPool, Vec<(vk::CommandBuffer, vk::Fence)>)>> = Mutex::new(None);

/// The submission thread, right after submitting a batch that claimed
/// `claims` (in sequence order): their bits set on the GPU by a small update
/// on the main queue (after the batch's prologue placed their pages), and
/// in the shadow.
pub(super) fn note_submitted(c: &Ctx, claims: &[u32]) -> Result<(), String> {
    if claims.is_empty() {
        return Ok(());
    }
    let b = bufs().ok_or("no resident set")?;
    let mut words: std::collections::BTreeMap<u32, u32> = std::collections::BTreeMap::new();
    for &g in claims {
        let w = g >> 5;
        let v = words.entry(w).or_insert_with(|| SHADOW[w as usize].load(Ordering::Acquire));
        *v |= 1 << (g & 31);
    }
    let mut u = UPDATES.lock();
    let (pool, flights) = match u.as_mut() {
        Some(x) => x,
        None => {
            let pool = unsafe { c.device.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(c.queue_family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER), None) }.map_err(|e| format!("c6 pool: {e}"))?;
            u.insert((pool, Vec::new()))
        }
    };
    let pool = *pool;
    flights.retain(|&(cmd, fence)| {
        let done = unsafe { c.device.get_fence_status(fence) }.unwrap_or(false);
        if done {
            unsafe {
                c.device.destroy_fence(fence, None);
                c.device.free_command_buffers(pool, &[cmd]);
            }
        }
        !done
    });
    let cmd = unsafe { c.device.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(pool).command_buffer_count(1)) }.map_err(|e| format!("c6 command buffer: {e}"))?[0];
    let fence = unsafe { c.device.create_fence(&vk::FenceCreateInfo::default(), None) }.map_err(|e| format!("c6 fence: {e}"))?;
    unsafe {
        c.device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).ok();
        let pre = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::MEMORY_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
        c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[pre], &[], &[]);
        for (&w, &v) in &words {
            c.device.cmd_update_buffer(cmd, b.res, w as u64 * 4, &v.to_le_bytes());
        }
        let post = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::MEMORY_READ);
        c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::ALL_COMMANDS, vk::DependencyFlags::empty(), &[post], &[], &[]);
        c.device.end_command_buffer(cmd).ok();
    }
    // (With the main queue's progress signal: later compute batches wait for it.)
    c.submit_timeline(false, &[cmd], &[], &[], fence)?;
    flights.push((cmd, fence));
    drop(u);
    for (&w, &v) in &words {
        SHADOW[w as usize].fetch_or(v, Ordering::AcqRel);
    }
    let mut cl = CLAIMED.lock();
    for g in claims {
        cl.remove(g);
    }
    STATS[2].fetch_add(claims.len() as u64, Ordering::Relaxed);
    Ok(())
}

/// Granules in the set now.
fn resident_count() -> u64 {
    SHADOW.iter().map(|w| w.load(Ordering::Relaxed).count_ones() as u64).sum()
}

/// The status line ("" when off).
pub fn status() -> String {
    if !on() {
        return String::new();
    }
    let s: Vec<u64> = STATS.iter().map(|x| x.load(Ordering::Relaxed)).collect();
    format!(
        "    C6 resident set: {} stages bound since boot; {} granules now ({} MiB); reports {} (lost {}; through the page table {}, no window in one piece {}, outside the span {}), granules admitted {}, refused (no import window) {}, admitted unarmed {}; batch covers {} (claims {} us), admitted pages to place {} and invalid resident pages placed {}, left out {}; waits for a pass {} ({:.1} ms), granules gone {}, hot pages taken {}\n",
        super::exec::C6_STAGES.load(Ordering::Relaxed),
        resident_count(),
        resident_count() * GRANULE >> 20,
        s[0],
        s[1],
        s[14],
        s[15],
        s[16],
        s[2],
        s[3],
        s[4],
        s[8],
        s[9] as f64 / s[8].max(1) as f64 / 1000.0,
        s[5],
        s[6],
        s[7],
        s[10],
        s[11] as f64 / 1e6,
        s[12],
        s[13]
    )
}

/// Statistics start again (`profile reset`).
pub fn reset() {
    for x in STATS.iter() {
        x.store(0, Ordering::Relaxed);
    }
}
