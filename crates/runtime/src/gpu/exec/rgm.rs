//! Resident GPU-written data, stage W1 (`gpu::rgm`, `tune rgm`): bindings
//! on the working copy W, the copies in that keep it current, and the merge
//! that stores a batch's W writes into guest memory.
//!
//! Copies in (guest memory into R, then R into W: one read of guest memory,
//! so the two copies agree) run in the batch prologue when nothing earlier in
//! the batch, or in a batch not submitted yet that it follows, writes their
//! pages (the prologue runs after all earlier submitted work, whose merges
//! have stored their W writes into guest memory); else in the stream, after
//! this batch's own W writes there went back to guest memory. The merge runs
//! when the batch is submitted, and before anything else in the batch uses
//! guest memory where it wrote W (in-place bindings, shadows, fills, CP
//! writes, indirect arguments, index buffers).

use super::*;
use crate::gpu::rgm::{self, Claim, Look};

/// An executor's merge table: device-local, written in the stream before
/// each merge dispatch (`vkCmdUpdateBuffer`), MERGE_MANY_MAX entries.
pub(super) struct RgmTable {
    buf: vk::Buffer,
    mem: vk::DeviceMemory,
}

impl Drop for RgmTable {
    fn drop(&mut self) {
        unsafe {
            ctx().device.destroy_buffer(self.buf, None);
            ctx().device.free_memory(self.mem, None);
        }
    }
}

const TABLE_BYTES: u64 = (rdna::kernels::MERGE_MANY_MAX * rdna::kernels::RESIDENT_ENTRY_DWORDS * 4) as u64;

impl Exec {
    /// A sized buffer binding of guest [base, base+size) on W: (W, offset
    /// of `base` in it, bytes from `base` to the span's end). A written one
    /// makes its granules resident (copied in whole at their first use); a
    /// read-only one needs them resident. Pages the CPU or our other paths
    /// wrote since their last copy in are copied in first. None: the old
    /// paths (counted by why).
    pub(super) fn rgm_bind(&mut self, base: u64, size: u64, written: bool) -> Option<(vk::Buffer, u64, u64)> {
        let r = rgm::get()?;
        let end = base.checked_add(size)?;
        if size == 0 || size > rgm::MAX_BIND || base < rgm::BASE || end > rgm::END || self.batch_alt {
            rgm::refused(0);
            return None;
        }
        let (g0, g1) = (rgm::granule(base), rgm::granule(end - 1));
        // Read-only: only data GPU work made resident (most read-only
        // bindings in the span are not, and pay nothing more here). A
        // granule this batch copies in whole counts.
        if !written && (g0..=g1).any(|g| !r.usable(g) && !self.rgm_mine.get(&g).is_some_and(|m| m.0 == rgm::ALL_PAGES)) {
            rgm::refused(2);
            return None;
        }
        if (g0..=g1).any(|g| r.is_demoted(g)) {
            rgm::refused(10);
            return None;
        }
        if super::super::homes::overlaps(base, size) || super::super::tune::keep_small() {
            rgm::refused(5);
            return None;
        }
        if self.resident.iter().any(|s| s.guest < end && base < s.guest + s.range) {
            rgm::refused(8);
            return None;
        }
        // Writes earlier in this batch by other paths would land in guest
        // memory after a copy in made here, and the merge would store W's
        // older data over them.
        if self.pending.overlaps(base, size) && !self.pending.intersect(base, size).iter().all(|&(a, n)| self.rgm_written.contains(a, n)) {
            rgm::refused(3);
            return None;
        }
        // The merge stores into guest memory through its import.
        if written && span(base, size).is_none() {
            rgm::refused(4);
            return None;
        }
        // A deferred CP write into the range is performed before this batch
        // is submitted (its copies in then see it).
        if let Some(s) = CP_WRITES.latest(&[(base, size)]) {
            self.submit_after = self.submit_after.max(s);
        }
        // The CPU's writes since the granules' last take, once per epoch
        // (every epoch: CPU writes before a wait the CPU releases are read
        // by the commands after it; with title submissions only, a stale
        // pointer in W lost the device during a 1-1 load, run 20261027-080648).
        let epoch = super::super::precache::epoch();
        if (g0..=g1).any(|g| !r.checked(g, epoch)) {
            r.take_lock(g0, g1);
            let first = (g0..=g1).find(|&g| !r.checked(g, epoch));
            let last = (g0..=g1).rev().find(|&g| !r.checked(g, epoch));
            if let (Some(a), Some(b)) = (first, last) {
                let lo = rgm::granule_addr(a);
                let n = rgm::granule_addr(b) + rgm::GRANULE - lo;
                let mut pages = std::mem::take(&mut self.rgm_pages);
                pages.clear();
                let t = std::time::Instant::now();
                let known = super::super::precache::take_written_for(lo, n, &mut pages, None, crate::mem::WW_WHO_RGM).is_some();
                rgm::STATS[rgm::S_TAKES].fetch_add(1, Ordering::Relaxed);
                rgm::STATS[rgm::S_TAKE_NS].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
                rgm::STATS[rgm::S_TAKE_PAGES].fetch_add(if known { pages.len() as u64 } else { n / rgm::PAGE }, Ordering::Relaxed);
                r.note_taken(a, b, epoch, known.then_some(&pages[..]));
                self.rgm_pages = pages;
            }
            r.take_unlock(g0, g1);
        }
        // The copies in this binding needs (pages this batch copied in
        // already, in the same dirty generation, are not copied again).
        let mut claims: Vec<Claim> = Vec::new();
        for g in g0..=g1 {
            match r.look(g, written) {
                Look::Clean => {}
                Look::Copy(mut c) => {
                    if let Some(&(pages, dgen)) = self.rgm_mine.get(&g) {
                        if pages == rgm::ALL_PAGES || dgen == c.dgen {
                            c.pages &= !pages;
                        }
                    }
                    if c.pages != 0 {
                        claims.push(c);
                    }
                }
                Look::Absent => {
                    // A granule this batch makes resident counts as such.
                    if self.rgm_mine.get(&g).is_some_and(|m| m.0 == rgm::ALL_PAGES) {
                        continue;
                    }
                    rgm::refused(2);
                    return None;
                }
            }
        }
        if !claims.is_empty() {
            match self.rgm_copy_in(r, g0, g1, &claims) {
                Ok(true) => {}
                Ok(false) => return None,
                Err(e) => {
                    self.report(format!("rgm copy in: {e}"));
                    return None;
                }
            }
            for c in claims {
                let e = self.rgm_mine.entry(c.g).or_insert((0, c.dgen));
                if e.1 != c.dgen && e.0 != rgm::ALL_PAGES {
                    *e = (0, c.dgen);
                }
                e.0 |= c.pages;
                self.rgm_claims.push(c);
            }
        }
        if written {
            self.rgm_written.insert(base, size);
            self.rgm_unmerged.insert(base, size);
            rgm::STATS[rgm::S_BIND_W].fetch_add(1, Ordering::Relaxed);
        } else {
            rgm::STATS[rgm::S_BIND_R].fetch_add(1, Ordering::Relaxed);
        }
        Some((r.w, base - rgm::BASE, rgm::END - base))
    }

    /// Records the copies in of `claims` (granules [g0, g1]). False: no
    /// video memory left or a page not importable (counted).
    fn rgm_copy_in(&mut self, r: &'static rgm::Rgm, g0: usize, g1: usize, claims: &[Claim]) -> Result<bool, String> {
        if claims.iter().any(|c| !c.was_resident) && !r.back(g0, g1)? {
            rgm::refused(6);
            return Ok(false);
        }
        // Their pages as runs, with their imports: (import, offset in it,
        // guest address, bytes).
        let mut copies: Vec<(vk::Buffer, u64, u64, u64)> = Vec::new();
        for c in claims {
            let ga = rgm::granule_addr(c.g);
            let mut p = 0u64;
            while p < rgm::GRANULE / rgm::PAGE {
                if c.pages & (1 << p) == 0 {
                    p += 1;
                    continue;
                }
                let mut q = p;
                while q < rgm::GRANULE / rgm::PAGE && c.pages & (1 << q) != 0 {
                    q += 1;
                }
                let (a, n) = (ga + p * rgm::PAGE, (q - p) * rgm::PAGE);
                match span(a, n) {
                    Some((buf, start, _)) => copies.push((buf, a - start, a, n)),
                    None => {
                        rgm::refused(4);
                        return Ok(false);
                    }
                }
                p = q;
            }
        }
        let ranges: Vec<(u64, u64)> = copies.iter().map(|c| (c.2, c.3)).collect();
        let pages: u64 = ranges.iter().map(|r| r.1 / rgm::PAGE).sum();
        rgm::STATS[rgm::S_COPY_PAGES].fetch_add(pages, Ordering::Relaxed);
        let in_stream = self.pending.overlaps_any(&ranges) || self.main_pending(&ranges) || self.closed_overlap(&ranges);
        if !in_stream {
            let _ = self.open_pre();
            for &(buf, off, a, n) in &copies {
                let o = a - rgm::BASE;
                self.pre_copies.push((buf, r.r, vk::BufferCopy { src_offset: off, dst_offset: o, size: n }));
                self.pre_after.push((r.r, r.w, vk::BufferCopy { src_offset: o, dst_offset: o, size: n }));
            }
            return Ok(true);
        }
        // This batch's W writes there go to guest memory first.
        if self.rgm_unmerged.overlaps_any(&ranges) {
            self.rgm_merge(Some(&ranges))?;
        }
        rgm::STATS[rgm::S_COPY_INLINE].fetch_add(1, Ordering::Relaxed);
        self.end_rendering();
        self.barrier();
        let c = ctx();
        let cmd = self.cmd();
        unsafe {
            for &(buf, off, a, n) in &copies {
                c.device.cmd_copy_buffer(cmd, buf, r.r, &[vk::BufferCopy { src_offset: off, dst_offset: a - rgm::BASE, size: n }]);
            }
            let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_READ);
            c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[mb], &[], &[]);
            for &(_, _, a, n) in &copies {
                let o = a - rgm::BASE;
                c.device.cmd_copy_buffer(cmd, r.r, r.w, &[vk::BufferCopy { src_offset: o, dst_offset: o, size: n }]);
            }
        }
        self.barrier();
        Ok(true)
    }

    /// Stores this batch's W writes (all, or those overlapping `only`) into
    /// guest memory and R (`rdna::kernels::resident_merge`), between
    /// barriers.
    pub(super) fn rgm_merge(&mut self, only: Option<&[(u64, u64)]>) -> Result<(), String> {
        use rdna::kernels::{MERGE_LOCAL_SIZE, MERGE_MANY_MAX};
        if self.rgm_unmerged.is_empty() {
            return Ok(());
        }
        let mut ranges: Vec<(u64, u64)> = match only {
            None => self.rgm_unmerged.iter().collect(),
            Some(rs) => rs.iter().flat_map(|&(a, n)| self.rgm_unmerged.intersect(a, n)).collect(),
        };
        if ranges.is_empty() {
            return Ok(());
        }
        ranges.sort_unstable();
        ranges.dedup();
        let r = rgm::get().ok_or("rgm is off")?;
        // `tune rgm 2` (diagnosis, timing only: guest memory then lacks the
        // W writes): no merges.
        if super::super::tune::rgm() == 2 {
            for &(a, n) in &ranges {
                self.rgm_unmerged.remove(a, n);
            }
            return Ok(());
        }
        let &(pipe, layout, _) = rgm::merge_pipe()?;
        let c = ctx();
        // Entries: dword-aligned pieces of one import each.
        let mut entries: Vec<[u32; 8]> = Vec::with_capacity(ranges.len());
        for &(a, n) in &ranges {
            let (lo, hi) = (a & !3, (a + n + 3) & !3);
            for (buf, off, ga, len) in guest_pieces(lo, hi - lo) {
                let va = c.buffer_address(buf);
                if va == 0 {
                    return Err(format!("the import of {ga:#x} has no device address"));
                }
                let dst = va + off;
                let so = ga - rgm::BASE;
                entries.push([0, so as u32, (so >> 32) as u32, dst as u32, (dst >> 32) as u32, (len / 4) as u32, 0, 0]);
            }
        }
        for &(a, n) in &ranges {
            self.rgm_unmerged.remove(a, n);
        }
        if entries.is_empty() {
            return Ok(());
        }
        if super::super::vk::labels_on() {
            let l = format!("{} rgm merges ({})", self.name, entries.len());
            self.label(&l);
        }
        // The batch's writes into W come first.
        self.end_rendering_then_barrier();
        let cmd = self.cmd();
        let mut bytes_total = 0u64;
        unsafe {
            c.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipe);
            for chunk in entries.chunks_mut(MERGE_MANY_MAX as usize) {
                let mut wg = 0u32;
                for e in chunk.iter_mut() {
                    e[0] = wg;
                    wg += e[5].div_ceil(MERGE_LOCAL_SIZE).max(1);
                    bytes_total += 4 * e[5] as u64;
                }
                let data: Vec<u8> = chunk.iter().flat_map(|e| e.iter().flat_map(|x| x.to_le_bytes())).collect();
                // The table: in the slot's ring when it has room (written
                // here), else written in the stream after the previous
                // dispatch read it.
                let ring = self.slots[self.cur].rings[0].and_then(|(rb, rp)| self.ring_alloc(0, data.len() as u64).map(|off| (rb, rp, off)));
                let info = match ring {
                    Some((rb, rp, off)) => {
                        std::ptr::copy_nonoverlapping(data.as_ptr(), rp.add(off as usize), data.len());
                        [vk::DescriptorBufferInfo { buffer: rb, offset: off, range: data.len() as u64 }]
                    }
                    None => {
                        let table = self.rgm_table()?;
                        let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::SHADER_READ).dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
                        c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[mb], &[], &[]);
                        c.device.cmd_update_buffer(cmd, table, 0, &data);
                        let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ);
                        c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &[mb], &[], &[]);
                        rgm::STATS[rgm::S_TABLE_UPDATES].fetch_add(1, Ordering::Relaxed);
                        [vk::DescriptorBufferInfo { buffer: table, offset: 0, range: data.len() as u64 }]
                    }
                };
                let writes = [vk::WriteDescriptorSet::default().dst_binding(0).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&info)];
                c.push_desc.cmd_push_descriptor_set(cmd, vk::PipelineBindPoint::COMPUTE, layout, 0, &writes);
                let pc: Vec<u8> = [0u32, chunk.len() as u32, r.w_addr as u32, (r.w_addr >> 32) as u32, r.r_addr as u32, (r.r_addr >> 32) as u32].iter().flat_map(|x| x.to_le_bytes()).collect();
                c.device.cmd_push_constants(cmd, layout, vk::ShaderStageFlags::COMPUTE, 0, &pc);
                c.device.cmd_dispatch(cmd, wg.max(1), 1, 1);
            }
        }
        self.dyncache.cs_pipeline = pipe;
        rgm::STATS[rgm::S_MERGES].fetch_add(1, Ordering::Relaxed);
        rgm::STATS[rgm::S_MERGE_RANGES].fetch_add(entries.len() as u64, Ordering::Relaxed);
        rgm::STATS[rgm::S_MERGE_BYTES].fetch_add(bytes_total, Ordering::Relaxed);
        self.ops[OP_MERGES] += entries.len() as u64;
        // Later work reads guest memory after the merge.
        self.barrier();
        self.barrier_wanted = false;
        self.dispatched = false;
        Ok(())
    }

    /// Before a use of guest [base, base+size) in place in this batch: its
    /// W writes there go back to guest memory first.
    #[track_caller]
    pub(super) fn rgm_before_in_place(&mut self, ranges: &[(u64, u64)]) -> Result<(), String> {
        if self.rgm_unmerged.is_empty() || !self.rgm_unmerged.overlaps_any(ranges) {
            return Ok(());
        }
        rgm::STATS[rgm::S_EARLY_MERGES].fetch_add(1, Ordering::Relaxed);
        rgm::note_site(0, 0);
        // The W writes there are shared with uses in place: their granules
        // take the old paths for a while.
        if let Some(r) = rgm::get() {
            for &(a, n) in ranges {
                for (wa, wn) in self.rgm_unmerged.intersect(a, n) {
                    r.demote(wa, wn);
                }
            }
        }
        self.rgm_merge(Some(ranges))
    }

    /// The open batch's claims went with it: submitted (`seq` given) or
    /// dropped; its W bookkeeping starts over.
    pub(super) fn rgm_batch_done(&mut self, submitted: bool) {
        if !self.rgm_claims.is_empty() {
            if let Some(r) = rgm::get() {
                if submitted {
                    r.submitted(&self.rgm_claims);
                } else {
                    r.dropped(&self.rgm_claims);
                }
            }
            self.rgm_claims.clear();
        }
        self.rgm_mine.clear();
        self.rgm_written.clear();
        self.rgm_unmerged.clear();
    }

    /// The executor's merge table (made at its first merge).
    fn rgm_table(&mut self) -> Result<vk::Buffer, String> {
        if let Some(t) = &self.rgm_table {
            return Ok(t.buf);
        }
        let c = ctx();
        unsafe {
            let bi = vk::BufferCreateInfo::default().size(TABLE_BYTES).usage(vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST);
            let buf = c.device.create_buffer(&bi, None).map_err(|e| format!("rgm table: {e}"))?;
            let req = c.device.get_buffer_memory_requirements(buf);
            let Some(mt) = c.memory_type(req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL) else {
                c.device.destroy_buffer(buf, None);
                return Err("rgm table: no device-local memory".into());
            };
            let mem = match c.device.allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(mt), None) {
                Ok(m) => m,
                Err(e) => {
                    c.device.destroy_buffer(buf, None);
                    return Err(format!("rgm table memory: {e}"));
                }
            };
            if let Err(e) = c.device.bind_buffer_memory(buf, mem, 0) {
                c.device.destroy_buffer(buf, None);
                c.device.free_memory(mem, None);
                return Err(format!("rgm table bind: {e}"));
            }
            self.rgm_table = Some(RgmTable { buf, mem });
            Ok(buf)
        }
    }
}
