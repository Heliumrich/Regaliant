//! Small GPU-written windows kept in video memory across batches (`gpu::keep`,
//! `tune keep_small`): their bindings, takes and write-backs.
//!
//! Only the queue that made a window writes it and drops it; other queues
//! bind it read-only, take the CPU's writes into it and write it back.
//! Takes run in the batch prologue (after all earlier batches, before this
//! batch's work: no render pass or barrier of their own), or in the stream
//! after a CP write or fill of this batch landed in guest memory there.
//! Write-backs run in the stream, between barriers.

use super::*;
use crate::gpu::keep::{self, Found, Kept};
use std::sync::Arc;

/// The sync kernel (`rdna::kernels::keep_sync`) with its three buffers
/// pushed (no descriptor-pool space: takes and write-backs are not reserved
/// for, and the prologue has no sets).
struct SyncPipe {
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
}

fn sync_pipe() -> Result<&'static SyncPipe, String> {
    static PIPE: std::sync::OnceLock<Result<SyncPipe, String>> = std::sync::OnceLock::new();
    PIPE.get_or_init(|| {
        let c = ctx();
        unsafe {
            let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..3)
                .map(|b| vk::DescriptorSetLayoutBinding::default().binding(b).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE))
                .collect();
            let set_layout = c
                .device
                .create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().flags(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR).bindings(&bindings), None)
                .map_err(|e| format!("keep sync set layout: {e}"))?;
            let layouts = [set_layout];
            let ranges = [vk::PushConstantRange { stage_flags: vk::ShaderStageFlags::COMPUTE, offset: 0, size: 12 }];
            let layout = c.device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts).push_constant_ranges(&ranges), None).map_err(|e| format!("keep sync layout: {e}"))?;
            let code = rdna::kernels::keep_sync();
            let module = c.device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None).map_err(|e| format!("keep sync module: {e}"))?;
            let stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(c"main");
            let ci = vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout);
            let pipeline = super::super::pcache::with_cache('i', 9, |cache| c.device.create_compute_pipelines(cache, &[ci], None)).map_err(|(_, e)| format!("keep sync pipeline: {e}"));
            c.device.destroy_shader_module(module, None);
            Ok(SyncPipe { pipeline: pipeline?[0], layout })
        }
    })
    .as_ref()
    .map_err(|e| e.clone())
}

/// Records the sync kernel over `k` into `cmd`: the CPU's changes since the
/// window last synced into it (a take), and with `give` the GPU's into
/// guest memory as well (see `rdna::kernels::keep_sync`).
fn record_sync(cmd: vk::CommandBuffer, k: &Kept, give: bool) -> Result<(), String> {
    let (hbuf, start, _) = span(k.base, k.size).ok_or_else(|| format!("kept window {:#x}+{:#x} is not mapped GPU memory", k.base, k.size))?;
    let sp = sync_pipe()?;
    let c = ctx();
    let host = vk::DescriptorBufferInfo { buffer: hbuf, offset: k.base - start, range: k.size };
    let work = vk::DescriptorBufferInfo { buffer: k.buf(), offset: k.work(), range: k.size };
    let pristine = vk::DescriptorBufferInfo { buffer: k.buf(), offset: k.pristine(), range: k.size };
    let infos = [host, pristine, work];
    let writes: Vec<vk::WriteDescriptorSet> = infos
        .iter()
        .enumerate()
        .map(|(i, info)| vk::WriteDescriptorSet::default().dst_binding(i as u32).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(std::slice::from_ref(info)))
        .collect();
    let n = (k.size / 4) as u32;
    let pc: Vec<u8> = [0u32, n, give as u32].iter().flat_map(|x| x.to_le_bytes()).collect();
    unsafe {
        c.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, sp.pipeline);
        c.push_desc.cmd_push_descriptor_set(cmd, vk::PipelineBindPoint::COMPUTE, sp.layout, 0, &writes);
        c.device.cmd_push_constants(cmd, sp.layout, vk::ShaderStageFlags::COMPUTE, 0, &pc);
        c.device.cmd_dispatch(cmd, n.div_ceil(rdna::kernels::MERGE_LOCAL_SIZE).max(1), 1, 1);
    }
    Ok(())
}

impl Exec {
    /// Whether this queue's recording batch wrote `k`.
    fn kept_own_write(&self, k: &Kept) -> bool {
        self.kept_written.iter().any(|x| x.id == k.id)
    }

    /// `k` is used by this batch: held until it completes, and the CPU's
    /// writes since the window last synced taken in first (once per batch:
    /// in the prologue, or in the stream when this batch wrote guest memory
    /// there already).
    fn keep_use(&mut self, k: &Arc<Kept>) -> Result<(), String> {
        if self.kept_taken.contains(&k.id) {
            return Ok(());
        }
        // A CP write deferred to the CPU into the window comes first.
        if let Some(s) = CP_WRITES.latest(&[(k.base, k.size)]) {
            self.submit_after = self.submit_after.max(s);
        }
        if self.pending.overlaps(k.base, k.size) {
            self.end_rendering_then_barrier();
            record_sync(self.cmd(), k, false)?;
            self.dyncache.cs_pipeline = sync_pipe()?.pipeline;
            self.barrier();
            self.barrier_wanted = false;
            self.dispatched = false;
        } else {
            let pre = self.open_pre();
            record_sync(pre, k, false)?;
        }
        self.kept_taken.push(k.id);
        self.slots[self.cur].kept.push(k.slot.clone());
        keep::STATS[2].fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Writes `k` back to guest memory in this batch (`why`: a STATS index)
    /// when GPU work wrote it since its last write-back, between barriers
    /// (`trailing`: one after it too, for work of this batch that reads
    /// guest memory there).
    pub(super) fn keep_give(&mut self, k: &Arc<Kept>, why: usize, trailing: bool) -> Result<(), String> {
        if !k.dirty() && !self.kept_own_write(k) {
            return Ok(());
        }
        self.begin();
        if !self.recording {
            return Ok(());
        }
        self.end_rendering_then_barrier();
        record_sync(self.cmd(), k, true)?;
        self.dyncache.cs_pipeline = sync_pipe()?.pipeline;
        if trailing {
            self.barrier();
            self.barrier_wanted = false;
            self.dispatched = false;
        }
        self.slots[self.cur].kept.push(k.slot.clone());
        if !self.kept_given.iter().any(|x| x.id == k.id) {
            self.kept_given.push(k.clone());
        }
        // Guest memory there is written by this batch.
        self.note_write(k.base, k.size);
        keep::STATS[why].fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// The binding of guest [base, base+size) in a kept window (see
    /// `gpu::keep`): (buffer, offset of `base`, bytes from `base` to the
    /// window's end). `create`: a written binding a VRAM shadow would serve
    /// (dispatches), which may make a window. None: the binding goes on as
    /// before (after windows it overlaps were written back where needed).
    pub(super) fn keep_bind(&mut self, base: u64, size: u64, written: bool, unbounded: bool, create: bool) -> Result<Option<(vk::Buffer, u64, u64)>, String> {
        if size == 0 || super::super::resident::mirror_mode() || super::super::fe2::on() {
            return Ok(None);
        }
        let on = super::super::tune::keep_small();
        if !on && keep::any() {
            self.keep_drop_own()?;
        }
        if unbounded {
            // An unbounded pointer (the homes' rule): assumed to use only the
            // window its base lies in, written back first (and taken in
            // again after a write through it).
            if let Some(k) = keep::at(base) {
                self.keep_give(&k, 5, true)?;
                if written {
                    self.kept_taken.retain(|&id| id != k.id);
                }
            }
            return Ok(None);
        }
        match keep::find(base, size) {
            Some(Found::Inside(k)) => {
                let own = k.owner == self.name;
                if written && !own {
                    // Another queue writes it: guest memory gets the
                    // window's data first and the write goes there (the
                    // owner's next take would bring it in, but the owner
                    // drops the window at its next use and the range is not
                    // kept for a while).
                    static LOGGED: AtomicU64 = AtomicU64::new(0);
                    if LOGGED.fetch_add(1, Ordering::Relaxed) < 20 {
                        crate::info!("gpu[{}]: written binding {base:#x}+{size:#x} of shader {:#x} inside kept window {:#x}+{:#x} of {}", self.name, super::super::images::BINDING_SHADER.with(|s| s.get()), k.base, k.size, k.owner);
                    }
                    self.keep_give(&k, 13, true)?;
                    k.retire.store(true, Ordering::Release);
                    keep::refuse(k.base, k.base + k.size);
                    return Ok(None);
                }
                if !on || (own && k.retire.load(Ordering::Acquire)) {
                    // The switch is off, or another queue wrote this one.
                    self.keep_give(&k, if on { 13 } else { 11 }, true)?;
                    if own {
                        keep::remove(&k);
                    }
                    return Ok(None);
                }
                self.keep_use(&k)?;
                if written && !self.kept_own_write(&k) {
                    self.kept_written.push(k.clone());
                }
                keep::note_bound(&k);
                let off = base - k.base;
                Ok(Some((k.buf(), k.work() + off, k.size - off)))
            }
            Some(Found::Overlap(ks)) => {
                static LOGGED: AtomicU64 = AtomicU64::new(0);
                if LOGGED.fetch_add(1, Ordering::Relaxed) < 40 {
                    let w: Vec<String> = ks.iter().map(|k| format!("{:#x}+{:#x} ({}{})", k.base, k.size, k.owner, if k.dirty() { ", dirty" } else { "" })).collect();
                    crate::info!(
                        "gpu[{}]: {} binding {base:#x}+{size:#x} of shader {:#x} overlaps kept windows without being inside one: {}",
                        self.name,
                        if written { "written" } else { "read-only" },
                        super::super::images::BINDING_SHADER.with(|s| s.get()),
                        w.join(", ")
                    );
                }
                // Windows of this queue the binding straddles are written
                // back and dropped; later windows there take their union
                // with the binding (one object bound with other extents).
                // Another queue's are written back; if it writes, they are
                // retired as above.
                let (mut lo, mut hi) = (base, base + size);
                for k in &ks {
                    self.keep_give(k, 5, true)?;
                    if k.owner == self.name {
                        keep::remove(k);
                        (lo, hi) = (lo.min(k.base), hi.max(k.base + k.size));
                    } else if written {
                        k.retire.store(true, Ordering::Release);
                        keep::refuse(k.base, k.base + k.size);
                    }
                }
                if ks.iter().any(|k| k.owner == self.name) {
                    keep::hint(lo, hi);
                }
                Ok(None)
            }
            None => {
                if !(on && written && create && !self.deferred && size <= keep::MAX && keep::lasting(base, size)) {
                    return Ok(None);
                }
                // Not over guest memory this batch wrote or shadows already
                // (the prologue's copies would run before those writes).
                let w0 = base & !(keep::GRAIN - 1);
                let w = (w0, (base + size).next_multiple_of(keep::GRAIN) - w0);
                if self.pending.overlaps(w.0, w.1) || self.resident.iter().any(|r| r.guest < w.0 + w.1 && w.0 < r.guest + r.range) || super::super::homes::overlaps(w.0, w.1) {
                    return Ok(None);
                }
                let Some((_, start, end)) = span(base, size) else { return Ok(None) };
                let mut k = keep::create(base, size, start, end, &self.name);
                if k.as_ref().err() == Some(&keep::Refused::Space) {
                    // Out of video memory for windows: this queue's least
                    // recently bound one goes.
                    if let Some(old) = keep::oldest_of(&self.name) {
                        self.keep_give(&old, 4, true)?;
                        keep::remove(&old);
                        keep::STATS[7].fetch_add(1, Ordering::Relaxed);
                        k = keep::create(base, size, start, end, &self.name);
                    }
                }
                let Ok(k) = k else { return Ok(None) };
                // (A hint may have widened it past the range checked above.)
                let clash = self.pending.overlaps(k.base, k.size) || self.resident.iter().any(|r| r.guest < k.base + k.size && k.base < r.guest + r.range) || super::super::homes::overlaps(k.base, k.size);
                let Some((hbuf, start, _)) = span(k.base, k.size).filter(|_| !clash) else {
                    keep::remove(&k);
                    return Ok(None);
                };
                // Both copies from guest memory, in the prologue.
                let pre = self.open_pre();
                unsafe {
                    let c = ctx();
                    c.device.cmd_copy_buffer(pre, hbuf, k.buf(), &[vk::BufferCopy { src_offset: k.base - start, dst_offset: k.pristine(), size: k.size }]);
                    c.device.cmd_copy_buffer(pre, hbuf, k.buf(), &[vk::BufferCopy { src_offset: k.base - start, dst_offset: k.work(), size: k.size }]);
                }
                if let Some(s) = CP_WRITES.latest(&[(k.base, k.size)]) {
                    self.submit_after = self.submit_after.max(s);
                }
                self.kept_taken.push(k.id);
                self.slots[self.cur].kept.push(k.slot.clone());
                self.kept_written.push(k.clone());
                keep::note_bound(&k);
                let off = base - k.base;
                Ok(Some((k.buf(), k.work() + off, k.size - off)))
            }
        }
    }

    /// Indirect dispatch arguments (12 bytes) inside a kept window: (its
    /// buffer, offset), taken in first.
    pub(super) fn keep_args(&mut self, a: u64) -> Result<Option<(vk::Buffer, u64)>, String> {
        match keep::find(a, 12) {
            Some(Found::Inside(k)) if super::super::tune::keep_small() => {
                self.keep_use(&k)?;
                keep::STATS[9].fetch_add(1, Ordering::Relaxed);
                Ok(Some((k.buf(), k.work() + (a - k.base))))
            }
            Some(Found::Inside(k)) => {
                self.keep_give(&k, 5, true)?;
                Ok(None)
            }
            Some(Found::Overlap(ks)) => {
                for k in &ks {
                    self.keep_give(k, 5, true)?;
                }
                Ok(None)
            }
            None => Ok(None),
        }
    }

    /// CPU-side reads of `reads` (`sync_for_reads`): windows GPU work wrote
    /// there are written back in this batch (the caller submits it and
    /// waits) and written back at the end of every batch that writes them
    /// from now on. Whether any was.
    pub(super) fn keep_reads(&mut self, reads: &[(u64, u64)]) -> Result<bool, String> {
        if !keep::any() {
            return Ok(false);
        }
        let mut ks = keep::dirty_in(reads);
        for k in &self.kept_written {
            if !ks.iter().any(|x| x.id == k.id) && reads.iter().any(|&(a, n)| k.overlaps(a, n)) {
                ks.push(k.clone());
            }
        }
        if ks.is_empty() {
            return Ok(false);
        }
        if self.deferred {
            self.stop("kept window read");
            return Ok(false);
        }
        for k in &ks {
            k.eager.store(true, Ordering::Relaxed);
            self.keep_give(k, 3, false)?;
        }
        Ok(true)
    }

    /// A CP write or fill this batch performs into guest [a, a+n) on the GPU:
    /// later uses of windows there in this batch take it in again (after it,
    /// in the stream).
    pub(super) fn keep_cp_write(&mut self, a: u64, n: u64) {
        if !keep::any() {
            return;
        }
        let ks: Vec<Arc<Kept>> = match keep::find(a, n) {
            Some(Found::Inside(k)) => vec![k],
            Some(Found::Overlap(ks)) => ks,
            None => return,
        };
        for k in ks {
            self.kept_taken.retain(|&id| id != k.id);
            keep::STATS[6].fetch_add(1, Ordering::Relaxed);
        }
    }

    /// The end of the batch: eager windows it wrote are written back.
    pub(super) fn keep_end_batch(&mut self) -> Result<(), String> {
        let eager: Vec<Arc<Kept>> = self.kept_written.iter().filter(|k| k.eager.load(Ordering::Relaxed)).cloned().collect();
        for k in &eager {
            self.keep_give(k, 4, false)?;
        }
        Ok(())
    }

    /// Batch `seq` was submitted: its writes and write-backs count.
    pub(super) fn keep_submitted(&mut self, seq: u64) {
        if !self.kept_written.is_empty() || !self.kept_given.is_empty() {
            keep::submitted(seq, &self.kept_written, &self.kept_given);
        }
        self.kept_written.clear();
        self.kept_given.clear();
        self.kept_taken.clear();
    }

    /// The switch was turned off: this queue's windows are written back and
    /// dropped.
    fn keep_drop_own(&mut self) -> Result<(), String> {
        for k in keep::owned_by(&self.name) {
            self.keep_give(&k, 11, true)?;
            keep::remove(&k);
        }
        Ok(())
    }
}
