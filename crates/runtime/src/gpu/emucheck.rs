//! Translation check against the reference interpreter (`rdna::emu`).
//!
//! `DS_EMU_CHECK=<shader>[,<shader>...][:<groups>]` (hex shader addresses):
//! before the first few direct dispatches of those compute shaders run on the GPU, a sample of
//! their workgroups (default 32, spread over the grid) is interpreted on the
//! CPU from the same inputs; after the GPU dispatch, the texels and bytes the
//! interpreter wrote are compared with what the GPU wrote. The result goes
//! to the log. Diagnostic only (slow).

use super::cp::{Cp, Space};
use super::desc::Tsharp;
use std::collections::{BTreeMap, HashMap};

struct Cfg {
    addrs: Vec<u64>,
    groups: u32,
}

fn cfg() -> Option<&'static Cfg> {
    static C: std::sync::OnceLock<Option<Cfg>> = std::sync::OnceLock::new();
    C.get_or_init(|| {
        let v = std::env::var("DS_EMU_CHECK").ok()?;
        let mut it = v.split(':');
        let addrs = it.next()?.split(',').filter_map(|a| u64::from_str_radix(a.trim_start_matches("0x"), 16).ok()).collect();
        let groups = it.next().and_then(|x| x.parse().ok()).unwrap_or(32);
        Some(Cfg { addrs, groups })
    })
    .as_ref()
}

/// Checks done so far per shader (the first 3 dispatches are checked).
static DONE: std::sync::Mutex<Option<HashMap<u64, u32>>> = std::sync::Mutex::new(None);

fn cs_addr(cp: &Cp) -> u64 {
    ((cp.reg(Space::Sh, 0x20C) as u64) | ((cp.reg(Space::Sh, 0x20D) as u64 & 0xFF) << 32)) << 8
}

/// Whether this dispatch is to be checked.
pub fn wanted(cp: &Cp) -> bool {
    let a = cs_addr(cp);
    cfg().is_some_and(|c| c.addrs.contains(&a)) && DONE.lock().unwrap().get_or_insert_with(HashMap::new).get(&a).copied().unwrap_or(0) < 3
}

/// A CPU copy of an image (mip 0, layer 0) in its Vulkan format.
struct Img {
    data: Vec<u8>,
    width: u32,
    height: u32,
    texel: u32,
}

struct Env {
    /// Bytes written by the interpreter (guest memory is not modified).
    writes: BTreeMap<u64, u8>,
    images: HashMap<u64, Option<Img>>,
    /// Texels written: (image base, x, y) -> raw value.
    texels: HashMap<(u64, u32, u32), [u32; 4]>,
    /// T# of each written image.
    written_images: HashMap<u64, [u32; 8]>,
    unmapped: u32,
    missing_images: u32,
}

impl Env {
    fn image(&mut self, t: &[u32; 8]) -> Option<&Img> {
        let ts = Tsharp::decode(t);
        self.images
            .entry(ts.base)
            .or_insert_with(|| super::images::with(|im| im.snapshot_raw(&ts)).ok().map(|(data, width, height, texel)| Img { data, width, height, texel }))
            .as_ref()
    }
}

impl rdna::emu::Env for Env {
    fn read(&mut self, addr: u64, out: &mut [u8]) -> bool {
        if !super::cp::guest_readable(addr, out.len() as u64) {
            self.unmapped += 1;
            return false;
        }
        unsafe { std::ptr::copy_nonoverlapping(addr as *const u8, out.as_mut_ptr(), out.len()) };
        for (k, b) in out.iter_mut().enumerate() {
            if let Some(&x) = self.writes.get(&(addr + k as u64)) {
                *b = x;
            }
        }
        true
    }

    fn write(&mut self, addr: u64, data: &[u8]) {
        for (k, &b) in data.iter().enumerate() {
            self.writes.insert(addr + k as u64, b);
        }
    }

    fn texel(&mut self, t: &[u32; 8], x: u32, y: u32, z: u32, level: u32) -> Option<[u32; 4]> {
        let base = Tsharp::decode(t).base;
        if let Some(v) = self.texels.get(&(base, x, y)) {
            return Some(*v);
        }
        if z != 0 || level != 0 {
            return None;
        }
        let Some(img) = self.image(t) else {
            self.missing_images += 1;
            return None;
        };
        if x >= img.width || y >= img.height {
            return None;
        }
        let o = ((y * img.width + x) * img.texel) as usize;
        let mut raw = [0u32; 4];
        for k in 0..img.texel as usize {
            raw[k / 4] |= (img.data[o + k] as u32) << (8 * (k % 4));
        }
        Some(raw)
    }

    fn store_texel(&mut self, t: &[u32; 8], x: u32, y: u32, z: u32, level: u32, raw: [u32; 4]) {
        if z != 0 || level != 0 {
            return;
        }
        let base = Tsharp::decode(t).base;
        self.written_images.insert(base, *t);
        self.texels.insert((base, x, y), raw);
    }
}

/// Interpreter registers after the dump instruction (`DS_DUMP`), per wave:
/// (VGPRs, SGPRs, executions).
type Dump = HashMap<u32, (Vec<[u32; 64]>, [u32; 128], u32)>;

pub struct Pending {
    env: Env,
    groups: usize,
    errors: Vec<String>,
    steps: u64,
    addr: u64,
    dump: Option<(u32, [u32; 3], Dump, usize)>,
    /// The dispatch's resources as the translation binds them (reported
    /// with differences).
    resources: Vec<String>,
}

/// Interprets a sample of the dispatch's workgroups (call before the GPU
/// dispatch, with earlier GPU work completed).
pub fn prepare(cp: &Cp, groups: [u32; 3], initiator: u32) -> Option<Pending> {
    let c = cfg()?;
    if groups.contains(&0) {
        // An empty (indirect) dispatch: nothing to compare.
        return None;
    }
    let addr = cs_addr(cp);
    *DONE.lock().unwrap().get_or_insert_with(HashMap::new).entry(addr).or_insert(0) += 1;
    let (insts, _) = match super::shaders::program(addr) {
        Ok(p) => p,
        Err(e) => {
            crate::warn!("emu check: {e}");
            return None;
        }
    };
    // The program, for offline disassembly.
    if let (Some(dir), Some(last)) = (crate::run_dir(), insts.last()) {
        let size = (last.pc + 4 * last.len as u32) as u64;
        if super::cp::guest_readable(addr, size) {
            let sdir = dir.join("shaders");
            let _ = std::fs::create_dir_all(&sdir);
            let bytes = unsafe { std::slice::from_raw_parts(addr as *const u8, size as usize) };
            let _ = std::fs::write(sdir.join(format!("{addr:012x}.bin")), bytes);
        }
    }
    let resources = super::shaders::prepare_compute(cp, initiator & (1 << 15) == 0).map(|p| super::shaders::describe_resources(&p)).unwrap_or_default();
    let sh = |i: u32| cp.reg(Space::Sh, i);
    let rsrc2 = sh(0x213);
    let local = [sh(0x207) & 0xFFFF, sh(0x208) & 0xFFFF, sh(0x209) & 0xFFFF];
    let user: Vec<u32> = (0..16).map(|i| sh(0x240 + i)).collect();
    let launch = rdna::emu::Launch {
        wave64: initiator & (1 << 15) == 0,
        local_size: local,
        rsrc2,
        user,
        group: [0; 3],
        lds_bytes: ((rsrc2 >> 15) & 0x1FF) * 512,
        code_base: addr,
        probe: super::shaders::probe_for(addr),
    };
    let total = groups[0] as u64 * groups[1].max(1) as u64 * groups[2].max(1) as u64;
    let n = (c.groups as u64).min(total).max(1);
    let mut env = Env { writes: BTreeMap::new(), images: HashMap::new(), texels: HashMap::new(), written_images: HashMap::new(), unmapped: 0, missing_images: 0 };
    let mut errors = Vec::new();
    let mut steps = 0;
    let t0 = std::time::Instant::now();
    // Register dump of one group (DS_DUMP): interpreted with a trace hook;
    // the GPU writes its dump into the cleared debug buffer.
    let dump = super::shaders::dump_for(addr).map(|(pc, g)| {
        let _ = super::exec::debug_dump(true);
        let mut l = launch.clone();
        l.group = g;
        let lanes = if l.wave64 { 64 } else { 32 };
        let mut regs: Dump = HashMap::new();
        {
            let mut grp = rdna::emu::Group::new(&insts, &mut env, l);
            let r = &mut regs;
            grp.trace = Some(Box::new(move |w, ins: &rdna::Inst, s: &[u32; 128], v: &[[u32; 64]]| {
                if ins.pc == pc {
                    let e = r.entry(w).or_insert_with(|| (Vec::new(), [0; 128], 0));
                    e.0 = v.to_vec();
                    e.1 = *s;
                    e.2 += 1;
                }
            }));
            if let Err(e) = grp.run(50_000_000) {
                errors.push(format!("dump group {g:?}: {e}"));
            }
            steps += grp.steps;
        }
        (pc, g, regs, lanes)
    });
    for k in 0..n {
        let lin = k * total / n + (total / n) / 2;
        let g = [(lin % groups[0] as u64) as u32, ((lin / groups[0] as u64) % groups[1].max(1) as u64) as u32, (lin / (groups[0] as u64 * groups[1].max(1) as u64)) as u32];
        let mut l = launch.clone();
        l.group = g;
        let mut grp = rdna::emu::Group::new(&insts, &mut env, l);
        if let Err(e) = grp.run(50_000_000) {
            errors.push(format!("group {g:?}: {e}"));
        }
        steps += grp.steps;
    }
    crate::info!("emu check: cs {addr:#x}: interpreted {n} of {total} groups ({steps} instructions, {} ms)", t0.elapsed().as_millis());
    Some(Pending { env, groups: n as usize, errors, steps, addr, dump, resources })
}

/// Compares the interpreter's writes with the GPU's (after the dispatch completed).
pub fn compare(p: Pending) {
    let addr = p.addr;
    for e in p.errors.iter().take(5) {
        crate::warn!("emu check: cs {addr:#x}: {e}");
    }
    // Texels.
    for (&base, t) in &p.env.written_images {
        let ts = Tsharp::decode(t);
        let gpu = super::images::with(|im| im.snapshot_raw(&ts));
        let (data, width, _height, texel) = match gpu {
            Ok(x) => x,
            Err(e) => {
                crate::warn!("emu check: cs {addr:#x}: image {base:#x} unreadable: {e}");
                continue;
            }
        };
        let mask = |k: usize| -> u32 {
            let bytes = (texel as usize).saturating_sub(4 * k).min(4);
            if bytes >= 4 { u32::MAX } else { (1u32 << (8 * bytes)) - 1 }
        };
        let (mut n, mut bad) = (0u64, 0u64);
        let mut first = Vec::new();
        let mut keys: Vec<_> = p.env.texels.keys().filter(|k| k.0 == base).copied().collect();
        keys.sort_by_key(|k| (k.2, k.1));
        for key in keys {
            let want = p.env.texels[&key];
            let (x, y) = (key.1, key.2);
            let o = ((y * width + x) * texel) as usize;
            let mut got = [0u32; 4];
            for k in 0..texel as usize {
                got[k / 4] |= (data.get(o + k).copied().unwrap_or(0) as u32) << (8 * (k % 4));
            }
            n += 1;
            let differs = (0..4).any(|k| (want[k] ^ got[k]) & mask(k) != 0);
            if differs {
                bad += 1;
                if first.len() < 12 {
                    first.push(format!("({x},{y}) cpu {:x} gpu {:x}", want[0] & mask(0), got[0] & mask(0)));
                }
            }
        }
        let level = if bad == 0 { "I" } else { "W" };
        let msg = format!("emu check: cs {addr:#x}: image {base:#x}: {n} texels from {} groups, {bad} differ{}", p.groups, if first.is_empty() { String::new() } else { format!(": {}", first.join(" ")) });
        if level == "I" {
            crate::info!("{msg}");
        } else {
            crate::warn!("{msg}");
        }
    }
    // Buffer bytes.
    let (mut n, mut bad) = (0u64, 0u64);
    let mut first = Vec::new();
    for (&a, &b) in &p.env.writes {
        if !super::cp::guest_readable(a, 1) {
            continue;
        }
        let g = unsafe { *(a as *const u8) };
        n += 1;
        if g != b {
            bad += 1;
            if first.len() < 12 {
                first.push(format!("{a:#x}: cpu {b:02x} gpu {g:02x}"));
            }
        }
    }
    if n > 0 {
        let msg = format!("emu check: cs {addr:#x}: {n} buffer bytes from {} groups, {bad} differ{}", p.groups, if first.is_empty() { String::new() } else { format!(": {}", first.join(" ")) });
        if bad == 0 {
            crate::info!("{msg}");
        } else {
            crate::warn!("{msg}; resources:
      {}", p.resources.join("
      "));
        }
    }
    if let Some((pc, g, regs, lanes)) = &p.dump {
        compare_dump(addr, *pc, *g, regs, *lanes);
    }
    if p.env.unmapped > 0 || p.env.missing_images > 0 {
        crate::warn!("emu check: cs {addr:#x}: {} unmapped reads, {} reads of images without a GPU copy ({} instructions)", p.env.unmapped, p.env.missing_images, p.steps);
    }
}

/// Compares the GPU's register dump (debug buffer) with the interpreter's.
fn compare_dump(addr: u64, pc: u32, g: [u32; 3], regs: &Dump, lanes: usize) {
    let gpu = super::exec::debug_dump(false);
    let stride = rdna::translate::DUMP_STRIDE as usize;
    let mut lines = Vec::new();
    let mut executed = 0;
    let mut waves: Vec<_> = regs.keys().copied().collect();
    waves.sort();
    for w in waves {
        let (v, s, count) = &regs[&w];
        let at = |l: usize, k: usize| gpu[(w as usize * lanes + l) * stride + k];
        let gpu_count: Vec<u32> = (0..lanes).map(|l| at(l, 385)).collect();
        executed += gpu_count.iter().filter(|&&c| c > 0).count();
        if gpu_count.iter().any(|&c| c != *count) {
            lines.push(format!("wave {w}: executed {count} times, GPU lanes {:?}", &gpu_count[..lanes.min(8)]));
        }
        for k in 0..256 {
            let bad: Vec<usize> = (0..lanes).filter(|&l| at(l, k) != v[k][l]).collect();
            if !bad.is_empty() {
                let ex: Vec<String> = bad.iter().take(4).map(|&l| format!("lane {l} cpu {:#x} gpu {:#x}", v[k][l], at(l, k))).collect();
                lines.push(format!("wave {w} v{k}: {} lanes differ ({})", bad.len(), ex.join(", ")));
            }
        }
        for k in 0..128 {
            let want = s[k];
            let got: Vec<u32> = (0..lanes).map(|l| at(l, 256 + k)).collect();
            if got.iter().all(|&x| x == want) {
                continue;
            }
            // Lane masks keep this lane's bit in bit 0 on the GPU.
            let pair = if k % 2 == 0 { (want as u64) | ((s[(k + 1).min(127)] as u64) << 32) } else { ((want as u64) << 32) | s[k - 1] as u64 };
            let mask_ok = got.iter().enumerate().all(|(l, &x)| x <= 1 && (x as u64) == (pair >> l & 1)) || (k % 2 == 1 && got.iter().all(|&x| x == 0));
            if mask_ok {
                continue;
            }
            let ex: Vec<String> = got.iter().enumerate().filter(|&(_, &x)| x != want).take(3).map(|(l, x)| format!("lane {l} gpu {x:#x}")).collect();
            lines.push(format!("wave {w} s{k}: cpu {want:#x}, {}", ex.join(", ")));
        }
    }
    if lines.is_empty() {
        crate::info!("emu check: cs {addr:#x}: registers of group {g:?} after {pc:#x} match ({executed} invocations dumped)");
    } else {
        crate::warn!("emu check: cs {addr:#x}: registers of group {g:?} after {pc:#x}: {} difference(s):", lines.len());
        for l in lines.iter().take(40) {
            crate::warn!("    {l}");
        }
    }
}

/// `DS_RACE_WATCH=<shader>`: when that compute shader is dispatched, the
/// first 8 KiB of each buffer it reads are write-protected; the first CPU
/// write to each page is logged with the writer's guest stack (at the next
/// such dispatch). Diagnostic only.
pub fn race_watch(cp: &Cp, initiator: u32) {
    static CFG: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    let Some(want) = *CFG.get_or_init(|| std::env::var("DS_RACE_WATCH").ok().and_then(|v| u64::from_str_radix(v.trim_start_matches("0x"), 16).ok())) else { return };
    let addr = cs_addr(cp);
    if addr != want {
        return;
    }
    for h in crate::exception::race_unwatch() {
        crate::warn!("race watch: cs {addr:#x}: {h}");
    }
    let Ok(p) = super::shaders::prepare_compute(cp, initiator & (1 << 15) == 0) else { return };
    for (i, r) in p.resources.iter().enumerate() {
        if let super::shaders::Resolved::Buffer(v) = r {
            if p.tr.plan.resources[i].read && v.byte_size().is_some_and(|n| n >= 8192) && super::cp::guest_readable(v.base, 8192) {
                crate::exception::race_watch(v.base, 8192);
            }
        }
    }
}

/// `DS_RACE_CHECK=<shader>`: when that compute shader is dispatched, hashes
/// the buffers it reads (per 64 KiB chunk), waits 30 ms and hashes them
/// again: changes mean another agent (CPU threads) was still writing the
/// inputs when the dispatch consumed them. Diagnostic only.
pub fn race_check(cp: &Cp, initiator: u32) {
    static CFG: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    let Some(want) = *CFG.get_or_init(|| std::env::var("DS_RACE_CHECK").ok().and_then(|v| u64::from_str_radix(v.trim_start_matches("0x"), 16).ok())) else { return };
    let addr = cs_addr(cp);
    if addr != want {
        return;
    }
    let Ok(p) = super::shaders::prepare_compute(cp, initiator & (1 << 15) == 0) else { return };
    const CHUNK: u64 = 64 << 10;
    let mut ranges = Vec::new();
    for (i, r) in p.resources.iter().enumerate() {
        if let super::shaders::Resolved::Buffer(v) = r {
            if !p.tr.plan.resources[i].read {
                continue;
            }
            let size = v.byte_size().unwrap_or(0).min(64 << 20);
            if size > 0 && super::cp::guest_readable(v.base, size) {
                ranges.push((i, v.base, size));
            }
        }
    }
    let hash = |a: u64, n: u64| -> u64 {
        let s = unsafe { std::slice::from_raw_parts(a as *const u8, n as usize) };
        s.chunks(8).fold(0xcbf2_9ce4_8422_2325u64, |h, c| {
            let mut w = [0u8; 8];
            w[..c.len()].copy_from_slice(c);
            (h ^ u64::from_le_bytes(w)).wrapping_mul(0x100_0000_01b3)
        })
    };
    let snap: Vec<Vec<u64>> = ranges.iter().map(|&(_, b, n)| (0..n.div_ceil(CHUNK)).map(|k| hash(b + k * CHUNK, CHUNK.min(n - k * CHUNK))).collect()).collect();
    std::thread::sleep(std::time::Duration::from_millis(30));
    // Scalar data (user data and s_load results resolved at record time).
    if let Ok(p2) = super::shaders::prepare_compute(cp, initiator & (1 << 15) == 0) {
        let diff: Vec<usize> = (0..p.sgpr_data.len().min(p2.sgpr_data.len())).filter(|&k| p.sgpr_data[k] != p2.sgpr_data[k]).collect();
        if !diff.is_empty() {
            let ex: Vec<String> = diff.iter().take(8).map(|&k| format!("[{k}] {:#x} -> {:#x}", p.sgpr_data[k], p2.sgpr_data[k])).collect();
            crate::warn!("race check: cs {addr:#x}: {} scalar data dword(s) changed within 30 ms of recording: {}", diff.len(), ex.join(", "));
        }
    }
    for (&(i, b, n), before) in ranges.iter().zip(&snap) {
        let changed: Vec<u64> = (0..n.div_ceil(CHUNK)).filter(|&k| hash(b + k * CHUNK, CHUNK.min(n - k * CHUNK)) != before[k as usize]).collect();
        if !changed.is_empty() {
            crate::warn!("race check: cs {addr:#x} resource {i} ({b:#x}, {n} bytes): {} of {} chunks changed within 30 ms of the dispatch (first at +{:#x})", changed.len(), n.div_ceil(CHUNK), changed[0] * CHUNK);
        }
    }
}
