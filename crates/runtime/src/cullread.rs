//! The title's CPU reads of its GPU culling results. Its job threads read the index counts the
//! index-culling compute (`xprindexculling`) writes into the indirect draw arguments (20-byte
//! records) and drop the objects that read 0 from the depth pre-pass's visibility mask
//! (`cmp dword ptr [r10+rbx*4], 0` at eboot+0xad91de: r10 the records, rbx the record's first
//! dword, both overwritten right after).
//!
//! Each frame the GPU clears the records (`cs_memset32`) and the culls (237 dispatches a frame
//! on the graphics queue at Forlorn Outpost, recorded in parallel chunks) add the counts. The
//! title submits them earlier in the frame and reads them without a fence: the console's GPU has
//! run them by then. Ours runs behind and its write-backs land batch by batch, so a read could
//! find the clear's zeros or part of the counts, and a visible object read 0 for one frame: in
//! the intro cutscene the enclosing box missed the depth pre-pass, its colour pass (depth
//! EQUAL) drew nothing and the sky flashed (FINDINGS 2026-11-14); at Forlorn Outpost wall pieces
//! went missing for a frame, showing black or the beams behind them (the user's videos,
//! 2026-11-15). The wait (`read_wait`, 2026-11-15): a read waits until the GPU work submitted
//! so far that writes the array has completed (an inline check of two counters; the call only
//! while a writer is in flight). The hold: a blank record (the clear's state; a culled object's
//! record keeps its other words: instances, first index and instance, with a count of 0) read
//! non-zero within the last `HOLD_FLIPS` flips, in the same array, reads as non-zero (a static
//! 1; the title only tests it) instead of 0, for the culls not yet submitted at the read. Real
//! culls pass as they are (a hold of every recent zero kept culled objects visible for two
//! more flips: in the four-cutscene script ~270 a flip and the title's GPU memory ~300 MB
//! higher).
//!
//! The view masks (2026-11-15): the title's frustum culling also runs on the GPU. Per object set
//! a dispatch (eboot+0xa1f930) writes four bit masks (a bit an object) and publishes them in the
//! set's block of the render context's table (slots 0-3; shadow views use slots 4+ and 0x18+).
//! The CPU then walks them without a fence: the index culls (eboot+0xa1f560: a dispatch and an
//! argument record per set bit, in bit order), the reads above (eboot+0xa1f020, the same
//! numbering) and the draws (eboot+0xa1f290). The console's GPU has run that small dispatch long
//! before; ours writes the masks back a batch later, so a walk could find part of a mask and bits
//! landed during or after it: the culls numbered fewer objects than the reads and draws (records
//! past them blank, later ones shifted), and walls went missing for a frame or were drawn with
//! other objects' triangles (the user's 0.1.0 videos 3 and 4). Each walk now first waits for the
//! GPU work in flight that writes the masks it reads (`wait_masks`).
//!
//! One array is read (0x1008a9ea10) where this was measured; the table still keys entries by
//! array (a hash of r10 picks a slot of `HOLD_LEN` entries, r10's low half tags the entry), so
//! a view with arrays of its own would not hold another's objects.
//!
//! The hold is an inline insertion (`hold_code`: the read runs ~3,400 times a frame in the
//! tunnel; a hook's register save would cost more). Diagnosis (`DS_CULLREAD_DIAG`, a hook in
//! its place, DIAGNOSTICS.md): 1 the same hold, with reads, zeros and holds logged per flip and
//! array; 0 no hold; 2 the hold with held reads sampled for a refill; 3 the hold with record 0's
//! words logged at its 0 reads; 4 the hold with each read waiting for the GPU work in flight
//! that writes the array (at most 100 ms); 5 that wait without the hold; 7 as 4, and first a
//! wait until the graphics command processor has run every title submission queued before the
//! read (at most 100 ms; not while it polls a label); 8 as 4, with one in 4 reads of a set
//! record at 0 that read non-zero lately read again for up to 3 ms; 9 as 4, counting records
//! whose first-instance field changed since their last read (it is a per-frame offset: ~40%
//! change every flip); 10 as 4, with the graphics queue's backlog of title submissions not yet
//! run at the flip's reads and at its set zero reads of records seen lately.

use std::sync::atomic::{AtomicU64, Ordering};

/// Flips a non-zero read holds a record.
pub const HOLD_FLIPS: u32 = 2;
/// Records a slot covers (dword index of the first word; the title's arrays have 75,000).
const HOLD_BITS: u32 = 17;
pub const HOLD_LEN: u32 = 1 << HOLD_BITS;
/// Slots (one per array, by hash).
const SLOT_BITS: u32 = 2;
const HASH: u64 = 0x9E37_79B9_7F4A_7C15;

/// Per array slot and record (its first dword's index): the flip of its last non-zero read
/// (high half) and the array's address (low half); 0: none.
fn table() -> &'static [AtomicU64] {
    static T: std::sync::OnceLock<Box<[AtomicU64]>> = std::sync::OnceLock::new();
    T.get_or_init(|| (0..(HOLD_LEN as usize) << SLOT_BITS).map(|_| AtomicU64::new(0)).collect())
}

fn entry(r10: u64, rbx: u64) -> &'static AtomicU64 {
    let slot = r10.wrapping_mul(HASH) >> (64 - SLOT_BITS);
    &table()[((slot << HOLD_BITS) + rbx) as usize]
}

static ONE: u32 = 1;

/// The inline hold's addresses: the table, the current flip's counter and the word a held read
/// is pointed at.
fn hold_addrs() -> (u64, u64, u64) {
    (table().as_ptr() as u64, crate::hle::libs::videoout::QUEUED_FLIP.as_ptr() as u64, &ONE as *const u32 as u64)
}

/// The array the title reads (its first byte; 0 until its first read: `read_wait`).
static ARRAY: AtomicU64 = AtomicU64::new(0);
/// The title's arrays: 75,000 dwords.
const ARRAY_BYTES: u64 = 75_000 * 4;
/// The latest batch submitted that writes the array (u64::MAX until the array is known, so the
/// first reads take `read_wait`).
static ARRAY_SEQ: AtomicU64 = AtomicU64::new(u64::MAX);
/// Reads that waited for the GPU's writes to the array, their total wait (us), waits that timed
/// out (100 ms).
static READ_WAITS: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// A batch submitted with its guest writes noted (`exec`): one that writes the array is the
/// latest a read of it waits for.
#[inline]
pub fn note_submitted(seq: u64, writes: &crate::gpu::ranges::Ranges) {
    let a = ARRAY.load(Ordering::Relaxed);
    if a != 0 && writes.overlaps(a, ARRAY_BYTES) {
        ARRAY_SEQ.fetch_max(seq, Ordering::AcqRel);
    }
}

/// The title's read at `r10` when the inline check found the array unknown, another array, or
/// a batch writing it not completed: waits until the GPU work submitted so far that writes the
/// array has completed (its write-backs in guest memory), at most 100 ms. The console's GPU has
/// run the culls the title submitted earlier in the frame by the time it reads them (it reads
/// without a fence); ours runs behind and its write-backs land batch by batch, so a read could
/// find the clear's zeros or part of the culls' counts: a visible object then missed the depth
/// pre-pass for a frame (a wall missing, what is behind it drawn instead: the user's 0.1.0
/// videos of 2026-11-14).
pub extern "sysv64" fn read_wait(r10: u64) {
    if ARRAY.load(Ordering::Acquire) == 0 && ARRAY.compare_exchange(0, r10, Ordering::AcqRel, Ordering::Acquire).is_ok() {
        // (Batches submitted from now on are noted; earlier ones are found by their ranges.)
        let seq = crate::gpu::exec::pending_writes(r10, ARRAY_BYTES).unwrap_or(0);
        let _ = ARRAY_SEQ.compare_exchange(u64::MAX, seq, Ordering::AcqRel, Ordering::Acquire);
    }
    let Some(seq) = crate::gpu::exec::pending_writes(r10, ARRAY_BYTES) else { return };
    if seq <= crate::gpu::exec::done_now() {
        return;
    }
    let t = std::time::Instant::now();
    let ok = crate::gpu::exec::wait_done_for(seq, std::time::Duration::from_millis(100));
    READ_WAITS[0].fetch_add(1, Ordering::Relaxed);
    READ_WAITS[1].fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);
    if !ok {
        READ_WAITS[2].fetch_add(1, Ordering::Relaxed);
    }
}

/// Status line: the reads' waits and the view masks' walks' waits.
pub fn status() -> String {
    let w: Vec<u64> = READ_WAITS.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    let m: Vec<u64> = MASK_WAITS.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    let c: Vec<u64> = MASK_CMP.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    let mut out = String::new();
    if w[0] > 0 {
        out += &format!("    culling results read by the title: {} reads waited for the GPU's writes ({} ms in all, {} after 100 ms)\n", w[0], w[1] / 1000, w[2]);
    }
    if m[0] > 0 {
        out += &format!("    view masks walked by the title: {} walks waited for the GPU's writes ({} ms in all, {} after 100 ms)\n", m[0], m[1] / 1000, m[2]);
    }
    if c.iter().any(|&x| x > 0) {
        out += &format!("    view masks at the reads against the culls' walk: {} the same, {} changed\n", c[0], c[1]);
        out += &format!("    cull masks at the draws against the culls' walk, word for word: {} the same, {} changed\n", MASK_CMP_DRAW[0].load(Ordering::Relaxed), MASK_CMP_DRAW[1].load(Ordering::Relaxed));
    }
    out
}

/// A view's mask slot in a set's block of the table: the title's choice by a pass descriptor's
/// view kind and index (as eboot+0xa1f020, eboot+0xa1f290 and eboot+0xa1f560 make it).
fn mask_slot(kind: u32, idx: i32) -> Option<u64> {
    match kind {
        0 => (idx >= 0).then(|| idx as u64 + 4),
        1 => Some(1),
        3 => Some(2),
        4 => Some(3),
        6 => (idx >= 0).then(|| idx as u64 + 0x18),
        k if k < 7 => Some(0),
        _ => None,
    }
}

/// The source pass mask's slot of a culling read by the pass's kind (the table at eboot+0x2382418,
/// kinds 1-4; the reads leave the others).
const READ_PASS_SLOT: [Option<u64>; 7] = [None, Some(1), Some(0), Some(2), Some(3), None, None];

/// Walks of the view masks that waited for the GPU's writes: [walks, us, waits given up after
/// 100 ms].
static MASK_WAITS: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

fn rd32(a: u64) -> Option<u32> {
    crate::gpu::cp::guest_readable(a, 4).then(|| unsafe { std::ptr::read_volatile(a as *const u32) })
}

fn rd64(a: u64) -> Option<u64> {
    crate::gpu::cp::guest_readable(a, 8).then(|| unsafe { std::ptr::read_volatile(a as *const u64) })
}

/// The view masks of `set` in `ctx`'s table at `slots`: (address, bytes) each, and the set's
/// object count.
fn masks_of(set: u64, ctx: u64, slots: &[Option<u64>]) -> (Vec<(u64, u64)>, u64) {
    let (Some(objs), Some(idx), Some(tab)) = (rd64(set + 0x38), rd32(set + 0x198), rd64(ctx + 0x58)) else { return (Vec::new(), 0) };
    if objs == 0 || tab == 0 || !crate::gpu::cp::guest_readable(objs + 100, 2) {
        return (Vec::new(), 0);
    }
    let count = unsafe { std::ptr::read_volatile((objs + 100) as *const u16) } as u64;
    let bytes = count.div_ceil(32).max(1) * 4;
    let block = idx as u64 * 0x2d0 + tab;
    let v = slots.iter().flatten().filter(|&&k| k < 0x2d0 / 8).filter_map(|&k| rd64(block + 8 * k)).filter(|&m| m != 0).map(|m| (m, bytes)).collect();
    (v, count)
}

/// Whether the walks wait (the `maskwait` control command; on by default).
pub static MASK_WAIT_ON: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// Waits until the GPU work in flight that writes these masks has completed (at most 100 ms).
fn wait_masks(masks: &[(u64, u64)]) {
    if !MASK_WAIT_ON.load(Ordering::Relaxed) {
        return;
    }
    let latest = masks.iter().filter_map(|&(m, n)| crate::gpu::exec::pending_writes(m, n)).max();
    let Some(seq) = latest else { return };
    if seq <= crate::gpu::exec::done_now() {
        return;
    }
    let t = std::time::Instant::now();
    let ok = crate::gpu::exec::wait_done_for(seq, std::time::Duration::from_millis(100));
    MASK_WAITS[0].fetch_add(1, Ordering::Relaxed);
    MASK_WAITS[1].fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);
    if !ok {
        MASK_WAITS[2].fetch_add(1, Ordering::Relaxed);
    }
}

/// Diagnosis (`DS_CULLREAD_DIAG`): per view mask (its address), the bits the culls' walk found;
/// the reads' walks against them: [the same, changed].
static WALKED: parking_lot::Mutex<Option<std::collections::HashMap<u64, u64>>> = parking_lot::Mutex::new(None);
static MASK_CMP: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
/// Diagnosis (`DS_CULLREAD_DIAG`, 2026-11-18 night): per cull mask, its words at the culls' walk
/// (and when); the draws' walks of that mask against them, word for word: [the same, changed,
/// logged].
static WALKED_WORDS: parking_lot::Mutex<Option<std::collections::HashMap<u64, (Vec<u64>, f64)>>> = parking_lot::Mutex::new(None);
static MASK_CMP_DRAW: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

fn mask_words(m: u64, count: u64) -> Option<Vec<u64>> {
    let words = count.div_ceil(64) as usize;
    if words == 0 || !crate::gpu::cp::guest_readable(m, 8 * words as u64) {
        return None;
    }
    let w = unsafe { std::slice::from_raw_parts(m as *const u64, words) };
    Some(w.iter().enumerate().map(|(k, &x)| x & if (k as u64 + 1) * 64 <= count { u64::MAX } else { (1u64 << (count % 64)) - 1 }).collect())
}

fn mask_bits(m: u64, count: u64) -> u64 {
    let words = count.div_ceil(64) as usize;
    if words == 0 || !crate::gpu::cp::guest_readable(m, 8 * words as u64) {
        return 0;
    }
    let w = unsafe { std::slice::from_raw_parts(m as *const u64, words) };
    w.iter().enumerate().map(|(k, &x)| (x & if (k as u64 + 1) * 64 <= count { u64::MAX } else { (1u64 << (count % 64)) - 1 }).count_ones() as u64).sum()
}

fn diag_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("DS_CULLREAD_DIAG").is_some())
}

/// The title's index culls of one object set (eboot+0xa1f560, its entry: rdi the set, rsi the
/// render context, rdx the cull pass's descriptor): the view mask they walk.
pub extern "sysv64" fn mask_wait_cull(regs: *const u64, _arg: u32) {
    let r = |i: usize| unsafe { *regs.add(i) };
    let (set, ctx, desc) = (r(7), r(6), r(2));
    let (Some(kind), Some(idx)) = (rd32(desc + 0x44), rd32(desc + 0x48)) else { return };
    let (masks, count) = masks_of(set, ctx, &[mask_slot(kind, idx as i32)]);
    wait_masks(&masks);
    if diag_on() {
        if let Some(&(m, _)) = masks.first() {
            WALKED.lock().get_or_insert_with(std::collections::HashMap::new).insert(m, mask_bits(m, count));
            if let Some(w) = mask_words(m, count) {
                WALKED_WORDS.lock().get_or_insert_with(std::collections::HashMap::new).insert(m, (w, crate::log::elapsed_ms()));
            }
        }
    }
}

/// The title's culling reads of one object set (eboot+0xa1f020, its entry: rdi the set, rsi the
/// render context, rdx the draw pass's descriptor, whose +8 is the cull pass's): the view mask and
/// the source pass mask they walk.
pub extern "sysv64" fn mask_wait_read(regs: *const u64, _arg: u32) {
    let r = |i: usize| unsafe { *regs.add(i) };
    let (set, ctx, draw) = (r(7), r(6), r(2));
    let Some(cull) = rd64(draw + 8).filter(|&c| c != 0) else { return };
    let (Some(kind), Some(idx), Some(pass)) = (rd32(cull + 0x44), rd32(cull + 0x48), rd32(draw + 0x20)) else { return };
    let (masks, count) = masks_of(set, ctx, &[mask_slot(kind, idx as i32), READ_PASS_SLOT.get(pass as usize).copied().flatten()]);
    wait_masks(&masks);
    if diag_on() {
        if let Some(&(m, _)) = masks.first() {
            if let Some(b) = WALKED.lock().as_mut().and_then(|w| w.remove(&m)) {
                MASK_CMP[(b != mask_bits(m, count)) as usize].fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

/// The title's draws of one object set's pass (eboot+0xa1f290, its entry: rdi the set, rsi the
/// render context, rdx the draw pass's descriptor): the pass's view mask and, with GPU culling,
/// the cull pass's.
pub extern "sysv64" fn mask_wait_draw(regs: *const u64, _arg: u32) {
    let r = |i: usize| unsafe { *regs.add(i) };
    let (set, ctx, draw) = (r(7), r(6), r(2));
    let (Some(kind), Some(idx)) = (rd32(draw + 0x20), rd32(draw + 0x24)) else { return };
    let cull = rd64(draw + 8).unwrap_or(0);
    let cull_slot = if cull != 0 { rd32(cull + 0x44).zip(rd32(cull + 0x48)).and_then(|(k, i)| mask_slot(k, i as i32)) } else { None };
    let (masks, _) = masks_of(set, ctx, &[mask_slot(kind, idx as i32), cull_slot]);
    wait_masks(&masks);
    if diag_on() && cull_slot.is_some() {
        let (cm, count) = masks_of(set, ctx, &[cull_slot]);
        if let Some(&(m, _)) = cm.first() {
            let now = mask_words(m, count);
            let g = WALKED_WORDS.lock();
            if let (Some((then, at)), Some(now)) = (g.as_ref().and_then(|t| t.get(&m)), now) {
                let changed: u32 = then.iter().zip(now.iter()).map(|(a, b)| (a ^ b).count_ones()).sum::<u32>() + (then.len() != now.len()) as u32;
                MASK_CMP_DRAW[(changed != 0) as usize].fetch_add(1, Ordering::Relaxed);
                if changed != 0 && MASK_CMP_DRAW[2].fetch_add(1, Ordering::Relaxed) < 2000 {
                    crate::info!("cull read: the draws walked cull mask {m:#x} with {changed} of its {count} bits changed since the culls' walk ({:.1} ms before)", crate::log::elapsed_ms() - at);
                }
            }
        }
    }
}

/// The code inserted before the read (registers kept, flags clobbered: the displaced `cmp`
/// sets them; the red zone stepped over): first the wait (`read_wait`, called with every
/// caller-saved register and the SSE state kept, only when r10 is not the known array or a
/// batch writing it has not completed), then the hold. `counter` counts the held reads.
pub fn hold_code(counter: u64) -> Vec<u8> {
    use iced_x86::code_asm::*;
    let (tab, flip, one) = hold_addrs();
    let (array, array_seq, done) = (&ARRAY as *const AtomicU64 as u64, &ARRAY_SEQ as *const AtomicU64 as u64, crate::gpu::exec::done_atomic_addr());
    let wait = read_wait as *const () as usize as u64;
    let asm = || -> Result<Vec<u8>, IcedError> {
        let mut a = CodeAssembler::new(64)?;
        let mut zero = a.create_label();
        let mut done_l = a.create_label();
        let mut slow = a.create_label();
        let mut hold = a.create_label();
        a.lea(rsp, ptr(rsp - 0x80))?;
        a.push(rax)?;
        a.push(rcx)?;
        a.push(rdx)?;
        // The wait: nothing to do while r10 is the known array and its latest writer completed.
        a.mov(rax, array)?;
        a.cmp(r10, qword_ptr(rax))?;
        a.jne(slow)?;
        a.mov(rax, array_seq)?;
        a.mov(rax, qword_ptr(rax))?;
        a.mov(rcx, done)?;
        a.cmp(rax, qword_ptr(rcx))?;
        a.jbe(hold)?;
        a.set_label(&mut slow)?;
        // `read_wait(r10)`: the other caller-saved registers, rbx and rbp (the stack's
        // alignment), the SSE state.
        for r in [rsi, rdi, r8, r9, r10, r11, rbx, rbp] {
            a.push(r)?;
        }
        a.mov(rbp, rsp)?;
        a.and(rsp, -16)?;
        a.sub(rsp, 512)?;
        a.fxsave64(ptr(rsp))?;
        a.mov(rdi, r10)?;
        a.mov(rax, wait)?;
        a.call(rax)?;
        a.fxrstor64(ptr(rsp))?;
        a.mov(rsp, rbp)?;
        for r in [rbp, rbx, r11, r10, r9, r8, rdi, rsi] {
            a.pop(r)?;
        }
        a.set_label(&mut hold)?;
        a.cmp(rbx, HOLD_LEN as i32)?;
        a.jae(done_l)?;
        // The entry: slot (r10 * HASH) >> (64 - SLOT_BITS), then the record.
        a.mov(rax, HASH)?;
        a.imul_2(rax, r10)?;
        a.shr(rax, 64 - SLOT_BITS)?;
        a.shl(rax, HOLD_BITS)?;
        a.add(rax, rbx)?;
        a.mov(rdx, tab)?;
        a.lea(rdx, ptr(rdx + rax * 8))?;
        a.mov(rcx, flip)?;
        a.mov(ecx, dword_ptr(rcx))?;
        a.mov(eax, dword_ptr(r10 + rbx * 4))?;
        a.test(eax, eax)?;
        a.jz(zero)?;
        // Seen: (flip << 32) | r10's low half.
        a.shl(rcx, 32)?;
        a.mov(eax, r10d)?;
        a.or(rax, rcx)?;
        a.mov(qword_ptr(rdx), rax)?;
        a.jmp(done_l)?;
        a.set_label(&mut zero)?;
        // Only a blank record (the clear's state; a culled object's record keeps its other
        // words) ...
        a.mov(eax, dword_ptr(r10 + rbx * 4 + 4))?;
        a.or(eax, dword_ptr(r10 + rbx * 4 + 8))?;
        a.or(eax, dword_ptr(r10 + rbx * 4 + 12))?;
        a.or(eax, dword_ptr(r10 + rbx * 4 + 16))?;
        a.jnz(done_l)?;
        // ... read non-zero lately in the same array.
        a.mov(rax, qword_ptr(rdx))?;
        a.test(rax, rax)?;
        a.jz(done_l)?;
        a.cmp(eax, r10d)?;
        a.jne(done_l)?;
        a.shr(rax, 32)?;
        a.sub(ecx, eax)?;
        a.cmp(ecx, HOLD_FLIPS as i32)?;
        a.ja(done_l)?;
        a.mov(rax, counter)?;
        a.lock().inc(qword_ptr(rax))?;
        a.mov(r10, one)?;
        a.lea(rax, ptr(rbx * 4))?;
        a.sub(r10, rax)?;
        a.set_label(&mut done_l)?;
        a.pop(rdx)?;
        a.pop(rcx)?;
        a.pop(rax)?;
        a.lea(rsp, ptr(rsp + 0x80))?;
        a.assemble(0)
    };
    asm().expect("culling read hold: assembly")
}

/// Diagnosis: per flip, each array's reads, zeros and holds; mode 2's sampled holds (still 0,
/// refilled) and the refills' total wait.
static FLIP: AtomicU64 = AtomicU64::new(0);
static SAMPLED: AtomicU64 = AtomicU64::new(0);
static REFILL: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];
static REFILL_US: AtomicU64 = AtomicU64::new(0);
/// Mode 9: per record (its index), the first-instance field last read, and per flip [reads whose
/// record now holds another first instance, of those reading 0, zero reads whose record kept it].
fn last_first() -> &'static [std::sync::atomic::AtomicU32] {
    static T: std::sync::OnceLock<Box<[std::sync::atomic::AtomicU32]>> = std::sync::OnceLock::new();
    T.get_or_init(|| (0..(HOLD_LEN as usize) / 5 + 1).map(|_| std::sync::atomic::AtomicU32::new(0)).collect())
}
static MOVED: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
/// Mode 10, per flip: [the graphics queue's backlog at the first read (u64::MAX: none yet), the
/// largest, set zero reads seen lately with a backlog, without].
static BACKLOG: [AtomicU64; 4] = [AtomicU64::new(u64::MAX), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
/// Mode 8: the lowest and highest first-dword index of the flip's set zero reads seen lately.
static SET_SEEN_IDX: [AtomicU64; 2] = [AtomicU64::new(u64::MAX), AtomicU64::new(0)];
/// Zero reads by (record blank, read non-zero within `HOLD_FLIPS`): index blank << 1 | seen.
static ZERO_KIND: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];
static STATS: parking_lot::Mutex<Vec<(u64, u64, u64, u64)>> = parking_lot::Mutex::new(Vec::new());

fn mode() -> u32 {
    static M: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *M.get_or_init(|| std::env::var("DS_CULLREAD_DIAG").ok().and_then(|v| v.parse().ok()).unwrap_or(1))
}

/// Mode 4 and 5's waits: reads that found GPU writes to the array in flight, their total wait
/// (us), waits that timed out.
static WAITS: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// Mode 7's waits for the graphics command processor to take up the title's submissions made
/// before the read: reads that waited, total us, waits given up (a label poll or 100 ms).
static SUB_WAITS: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// Mode 7: the read waits until the graphics command processor has run every title submission
/// queued before it (the culls the title submitted earlier in the frame recorded and handed to
/// the GPU), unless it polls a label (at most 100 ms).
fn wait_submitted() {
    let t = std::time::Instant::now();
    match crate::gpu::wait_gfx_caught_up(std::time::Duration::from_millis(100)) {
        None => {}
        Some(ok) => {
            SUB_WAITS[0].fetch_add(1, Ordering::Relaxed);
            SUB_WAITS[1].fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);
            if !ok {
                SUB_WAITS[2].fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

/// Modes 4, 5 and 7: the read waits until the GPU work in flight that writes the array (its
/// 75,000 dwords from r10) has completed, at most 100 ms (the title's reads then see the
/// culling's whole result for the work submitted so far, as the console's timing gives them).
fn wait_pending(r10: u64) {
    let Some(seq) = crate::gpu::exec::pending_writes(r10, 75_000 * 4) else { return };
    if seq <= crate::gpu::exec::done_now() {
        return;
    }
    let t = std::time::Instant::now();
    let ok = crate::gpu::exec::wait_done_for(seq, std::time::Duration::from_millis(100));
    WAITS[0].fetch_add(1, Ordering::Relaxed);
    WAITS[1].fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);
    if !ok {
        WAITS[2].fetch_add(1, Ordering::Relaxed);
    }
}

/// Diagnosis: at the read (before it), the hold of `hold_code` (unless mode 0 or 5) and the log.
pub extern "sysv64" fn read_hook(regs: *const u64, _arg: u32) {
    let r = |i: usize| unsafe { *regs.add(i) };
    let (rbx, r10) = (r(3), r(10));
    if mode() == 7 {
        wait_submitted();
    }
    if mode() >= 4 {
        wait_pending(r10);
    }
    let v = unsafe { std::ptr::read_volatile(r10.wrapping_add(rbx.wrapping_mul(4)) as *const u32) };
    let flip = crate::hle::libs::videoout::QUEUED_FLIP.load(Ordering::Relaxed);
    let last = FLIP.swap(flip, Ordering::Relaxed);
    let mut held = false;
    // Zero reads: is the whole 20-byte record zero (the clear's state) or are its other words
    // set (instances, first index or instance: written for a culled object)?
    let w = |k: u64| unsafe { std::ptr::read_volatile(r10.wrapping_add(rbx.wrapping_mul(4) + 4 * k) as *const u32) };
    let blank = v == 0 && (1..5).all(|k| w(k) == 0);
    if v == 0 {
        let seen = rbx < HOLD_LEN as u64 && {
            let e = entry(r10, rbx).load(Ordering::Relaxed);
            e != 0 && e & 0xFFFF_FFFF == r10 & 0xFFFF_FFFF && (flip as u32).wrapping_sub((e >> 32) as u32) <= HOLD_FLIPS
        };
        ZERO_KIND[(blank as usize) << 1 | seen as usize].fetch_add(1, Ordering::Relaxed);
        if mode() == 10 && !blank && seen {
            let b = crate::gpu::gfx_backlog();
            BACKLOG[if b > 0 { 2 } else { 3 }].fetch_add(1, Ordering::Relaxed);
        }
        if mode() == 9 && !blank && rbx < HOLD_LEN as u64 && rbx % 5 == 0 {
            let first = w(4);
            let prev = last_first()[(rbx / 5) as usize].load(Ordering::Relaxed);
            MOVED[if prev != 0 && prev != first { 1 } else { 2 }].fetch_add(1, Ordering::Relaxed);
        }
        // Mode 8: a set record read 0 though non-zero lately: one in 4 is read again for up
        // to 3 ms (a count that lands later, or one the culls computed as 0?), and the record
        // indices of these reads are kept for the flip's line.
        if mode() == 8 && !blank && seen {
            SET_SEEN_IDX[0].fetch_min(rbx, Ordering::Relaxed);
            SET_SEEN_IDX[1].fetch_max(rbx, Ordering::Relaxed);
            if SAMPLED.fetch_add(1, Ordering::Relaxed) % 4 == 0 {
                let t = std::time::Instant::now();
                let a = r10.wrapping_add(rbx.wrapping_mul(4)) as *const u32;
                while unsafe { std::ptr::read_volatile(a) } == 0 && t.elapsed() < std::time::Duration::from_millis(3) {
                    std::hint::spin_loop();
                }
                let filled = unsafe { std::ptr::read_volatile(a) } != 0;
                REFILL[filled as usize].fetch_add(1, Ordering::Relaxed);
                if filled {
                    REFILL_US.fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);
                }
            }
        }
        if rbx == 0 && mode() == 3 {
            crate::info!("cull read: flip {flip}: record 0 reads 0: [{} {} {} {}] after it", w(1), w(2), w(3), w(4));
        }
    }
    if mode() == 10 {
        let b = crate::gpu::gfx_backlog();
        let _ = BACKLOG[0].compare_exchange(u64::MAX, b, Ordering::Relaxed, Ordering::Relaxed);
        BACKLOG[1].fetch_max(b, Ordering::Relaxed);
    }
    // Mode 9: the record's first instance against the one it held at its last read.
    if mode() == 9 && rbx < HOLD_LEN as u64 && rbx % 5 == 0 {
        let first = w(4);
        if first != 0 || v != 0 {
            let prev = last_first()[(rbx / 5) as usize].swap(first, Ordering::Relaxed);
            if prev != 0 && prev != first {
                MOVED[0].fetch_add(1, Ordering::Relaxed);
            }
        }
    }
    if mode() != 0 && mode() != 5 && rbx < HOLD_LEN as u64 {
        let e = entry(r10, rbx);
        if v != 0 {
            e.store((flip << 32) | (r10 & 0xFFFF_FFFF), Ordering::Relaxed);
        } else if blank {
            let seen = e.load(Ordering::Relaxed);
            if seen != 0 && seen & 0xFFFF_FFFF == r10 & 0xFFFF_FFFF && (flip as u32).wrapping_sub((seen >> 32) as u32) <= HOLD_FLIPS {
                held = true;
                // Mode 2: one held read in 16 is read again for up to 2 ms: does the GPU's
                // refill reach it (a read inside the clear's window) or does it stay 0 (a cull)?
                if mode() == 2 && SAMPLED.fetch_add(1, Ordering::Relaxed) % 16 == 0 {
                    let t = std::time::Instant::now();
                    let a = r10.wrapping_add(rbx.wrapping_mul(4)) as *const u32;
                    while unsafe { std::ptr::read_volatile(a) } == 0 && t.elapsed() < std::time::Duration::from_millis(2) {
                        std::hint::spin_loop();
                    }
                    let filled = unsafe { std::ptr::read_volatile(a) } != 0;
                    REFILL[filled as usize].fetch_add(1, Ordering::Relaxed);
                    if filled {
                        REFILL_US.fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);
                    }
                }
                // The displaced `cmp dword ptr [r10+rbx*4], 0` then reads ONE.
                unsafe { *(regs.add(10) as *mut u64) = (&ONE as *const u32 as u64).wrapping_sub(rbx.wrapping_mul(4)) };
            }
        }
    }
    let mut s = STATS.lock();
    if last != flip && last != 0 && !s.is_empty() {
        let line: Vec<String> = s.iter().map(|(a, n, z, h)| format!("{a:#x}: {n} reads, {z} zero, {h} held")).collect();
        let (stay, fill) = (REFILL[0].swap(0, Ordering::Relaxed), REFILL[1].swap(0, Ordering::Relaxed));
        let refill = if stay + fill > 0 {
            if mode() == 8 {
                let (lo, hi) = (SET_SEEN_IDX[0].swap(u64::MAX, Ordering::Relaxed), SET_SEEN_IDX[1].swap(0, Ordering::Relaxed));
                format!("; sampled set zeros seen lately (records {}..{}): {fill} refilled (mean {} us), {stay} still 0 after 3 ms", lo / 5, hi / 5, REFILL_US.swap(0, Ordering::Relaxed) / fill.max(1))
            } else {
                format!("; sampled holds: {fill} refilled (mean {} us), {stay} still 0 after 2 ms", REFILL_US.swap(0, Ordering::Relaxed) / fill.max(1))
            }
        } else {
            String::new()
        };
        let k: Vec<u64> = ZERO_KIND.iter().map(|c| c.swap(0, Ordering::Relaxed)).collect();
        let w: Vec<u64> = WAITS.iter().map(|c| c.swap(0, Ordering::Relaxed)).collect();
        let sw: Vec<u64> = SUB_WAITS.iter().map(|c| c.swap(0, Ordering::Relaxed)).collect();
        let mut waits = if mode() >= 4 { format!("; waits for the GPU's writes: {} ({} us, {} timed out)", w[0], w[1], w[2]) } else { String::new() };
        if mode() == 7 {
            waits += &format!("; waits for the graphics CP to take up earlier submissions: {} ({} us, {} given up)", sw[0], sw[1], sw[2]);
        }
        if mode() == 9 {
            let m: Vec<u64> = MOVED.iter().map(|c| c.swap(0, Ordering::Relaxed)).collect();
            waits += &format!("; records holding another first instance than at their last read: {} (set zero reads: {} moved, {} kept)", m[0], m[1], m[2]);
        }
        if mode() == 10 {
            let b: Vec<u64> = BACKLOG.iter().map(|c| c.swap(0, Ordering::Relaxed)).collect();
            waits += &format!("; graphics submissions not yet run at the reads: first {}, most {}; set zero reads seen lately with a backlog {}, without {}", b[0], b[1], b[2], b[3]);
            BACKLOG[0].store(u64::MAX, Ordering::Relaxed);
        }
        crate::info!(
            "cull read: flip {last}: {}{refill}; zero records: blank {} ({} seen lately), set {} ({} seen lately){waits}",
            line.join("; "),
            k[2] + k[3],
            k[3],
            k[0] + k[1],
            k[1]
        );
        s.clear();
    }
    let i = match s.iter().position(|e| e.0 == r10) {
        Some(i) => i,
        None => {
            s.push((r10, 0, 0, 0));
            s.len() - 1
        }
    };
    s[i].1 += 1;
    s[i].2 += (v == 0) as u64;
    s[i].3 += held as u64;
}
