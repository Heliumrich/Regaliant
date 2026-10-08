//! Load-time code patching of guest modules.
//!
//! * `mov r64, fs:[0]` (the TCB self pointer) is rewritten in place to
//!   `mov r64, gs:[TEB.TlsSlots[slot]]`, where the slot holds the guest thread
//!   pointer. Same length, no trampolines.
//! * `mov r64, fs:[disp]` with disp != 0 jumps to a trampoline that loads the
//!   thread pointer and then the field.
//! * `syscall` becomes `ud2`, so a raw FreeBSD syscall can never reach the Windows
//!   kernel; the exception handler reports the site if one is ever executed.
//! * Any other FS-relative form is replaced by `ud2` and listed.

use iced_x86::{Decoder, DecoderOptions, Instruction, Mnemonic, OpKind, Register};

#[derive(Default, Debug)]
pub struct PatchStats {
    pub functions: usize,
    pub fs_zero: usize,
    pub fs_raw_extra: usize,
    pub fs_disp: usize,
    pub fs_unknown: Vec<(u64, String, (u64, u64))>,
    pub syscalls: Vec<u64>,
}

/// TEB offset of TlsSlots[slot] on x64.
pub fn teb_tls_slot_offset(slot: u32) -> u32 {
    0x1480 + slot * 8
}

fn gpr64_index(r: Register) -> Option<u8> {
    use Register::*;
    Some(match r {
        RAX => 0,
        RCX => 1,
        RDX => 2,
        RBX => 3,
        RSP => 4,
        RBP => 5,
        RSI => 6,
        RDI => 7,
        R8 => 8,
        R9 => 9,
        R10 => 10,
        R11 => 11,
        R12 => 12,
        R13 => 13,
        R14 => 14,
        R15 => 15,
        _ => return Option::None,
    })
}

/// Encodes `mov reg64, gs:[disp32]`.
fn enc_mov_gs(reg: u8, disp: u32) -> Vec<u8> {
    let rex = 0x48 | if reg >= 8 { 0x04 } else { 0 };
    let mut v = vec![0x65, rex, 0x8B, ((reg & 7) << 3) | 0x04, 0x25];
    v.extend_from_slice(&disp.to_le_bytes());
    v
}

/// Encodes `mov reg64, [reg64 + disp32]`.
fn enc_mov_load(reg: u8, disp: i32) -> Vec<u8> {
    let rex = 0x48 | if reg >= 8 { 0x05 } else { 0 };
    let mut v = vec![rex, 0x8B, 0x80 | ((reg & 7) << 3) | (reg & 7)];
    if reg & 7 == 4 {
        v.push(0x24);
    }
    v.extend_from_slice(&disp.to_le_bytes());
    v
}

fn enc_jmp(from: u64, to: u64) -> Option<[u8; 5]> {
    let rel = to as i64 - (from as i64 + 5);
    let rel32 = i32::try_from(rel).ok()?;
    let mut b = [0xE9, 0, 0, 0, 0];
    b[1..].copy_from_slice(&rel32.to_le_bytes());
    Some(b)
}

pub struct Patcher<'a> {
    /// Host-writable view of the module image, indexed by module-relative vaddr.
    pub image: &'a mut [u8],
    pub base: u64,
    pub tls_slot_disp: u32,
    /// Trampoline space (absolute address + writable slice) near the module.
    pub tramp_base: u64,
    pub tramp: &'a mut [u8],
    pub tramp_used: usize,
}

impl<'a> Patcher<'a> {
    fn emit_trampoline(&mut self, code: &[u8]) -> Option<u64> {
        if self.tramp_used + code.len() > self.tramp.len() {
            return None;
        }
        let at = self.tramp_base + self.tramp_used as u64;
        self.tramp[self.tramp_used..self.tramp_used + code.len()].copy_from_slice(code);
        self.tramp_used = (self.tramp_used + code.len() + 15) & !15;
        Some(at)
    }

    fn handle_fs(&mut self, ins: &Instruction, range: (u64, u64), stats: &mut PatchStats, fmt: &mut iced_x86::FastFormatter) {
        let rel = ins.ip() - self.base;
        let off = rel as usize;
        let len = ins.len();
        let bytes = self.image[off..off + len].to_vec();
        let simple_abs = ins.mnemonic() == Mnemonic::Mov
            && ins.op_count() == 2
            && ins.op0_kind() == OpKind::Register
            && ins.op1_kind() == OpKind::Memory
            && ins.memory_base() == Register::None
            && ins.memory_index() == Register::None
            && gpr64_index(ins.op0_register()).is_some()
            && len == 9
            && bytes[0] == 0x64
            && (bytes[1] & 0xF8) == 0x48
            && bytes[2] == 0x8B
            && (bytes[3] & 0xC7) == 0x04
            && bytes[4] == 0x25;
        if simple_abs {
            let reg = gpr64_index(ins.op0_register()).unwrap();
            let disp = ins.memory_displacement64();
            if disp == 0 {
                let new = enc_mov_gs(reg, self.tls_slot_disp);
                self.image[off..off + 9].copy_from_slice(&new);
                stats.fs_zero += 1;
                return;
            }
            // fs:[disp] -> trampoline: mov reg, gs:[slot]; mov reg, [reg+disp]; jmp back
            let mut code = enc_mov_gs(reg, self.tls_slot_disp);
            code.extend(enc_mov_load(reg, disp as i32));
            let jmp_pos = self.tramp_base + self.tramp_used as u64 + code.len() as u64;
            if let Some(j) = enc_jmp(jmp_pos, ins.next_ip()) {
                code.extend_from_slice(&j);
                if let Some(t) = self.emit_trampoline(&code) {
                    if let Some(j2) = enc_jmp(ins.ip(), t) {
                        self.image[off..off + 5].copy_from_slice(&j2);
                        for b in &mut self.image[off + 5..off + 9] {
                            *b = 0xCC; // never executed: control returns to next_ip
                        }
                        stats.fs_disp += 1;
                        return;
                    }
                }
            }
        }
        let mut text = String::new();
        fmt.format(ins, &mut text);
        // Not patched: these are usually bytes that are not instructions (decoder
        // desynchronized inside a function range). Recorded for fault diagnostics.
        stats.fs_unknown.push((rel, text, range));
    }

    /// Decodes each function range and applies patches. `ranges` are module-relative.
    pub fn patch_functions(&mut self, ranges: &[(u64, u64)], stats: &mut PatchStats) {
        let mut fmt = iced_x86::FastFormatter::new();
        let mut ins = Instruction::default();
        for &(start, end) in ranges {
            if end as usize > self.image.len() || start >= end {
                continue;
            }
            stats.functions += 1;
            let code = self.image[start as usize..end as usize].to_vec();
            let mut dec = Decoder::with_ip(64, &code, self.base + start, DecoderOptions::NONE);
            let mut fs_sites = Vec::new();
            let mut sys_sites = Vec::new();
            while dec.can_decode() {
                dec.decode_out(&mut ins);
                if ins.is_invalid() {
                    continue;
                }
                if ins.segment_prefix() == Register::FS || (ins.op_count() > 0 && has_fs_mem(&ins)) {
                    fs_sites.push(ins);
                } else if ins.mnemonic() == Mnemonic::Syscall {
                    sys_sites.push(ins.ip() - self.base);
                }
            }
            for s in fs_sites {
                self.handle_fs(&s, (start, end), stats, &mut fmt);
            }
            for rel in sys_sites {
                let off = rel as usize;
                self.image[off] = 0x0F;
                self.image[off + 1] = 0x0B;
                stats.syscalls.push(rel);
            }
        }
    }

    /// Byte-pattern pass for `mov r64, fs:[0]` outside FDE-described code.
    pub fn patch_raw_fs_zero(&mut self, code_end: u64, stats: &mut PatchStats) {
        let n = (code_end as usize).min(self.image.len());
        let mut i = 0usize;
        while i + 9 <= n {
            let b = &self.image[i..i + 9];
            if b[0] == 0x64
                && (b[1] == 0x48 || b[1] == 0x4C)
                && b[2] == 0x8B
                && (b[3] & 0xC7) == 0x04
                && b[4] == 0x25
                && b[5..9] == [0, 0, 0, 0]
            {
                let reg = ((b[3] >> 3) & 7) | if b[1] == 0x4C { 8 } else { 0 };
                let new = enc_mov_gs(reg, self.tls_slot_disp);
                self.image[i..i + 9].copy_from_slice(&new);
                stats.fs_raw_extra += 1;
                i += 9;
            } else {
                i += 1;
            }
        }
    }
}

fn has_fs_mem(ins: &Instruction) -> bool {
    (0..ins.op_count()).any(|i| ins.op_kind(i) == OpKind::Memory) && ins.memory_segment() == Register::FS
}

/// Debug aid: disassembles a function range around `site` (module-relative).
pub fn dump_around(base: u64, range: (u64, u64), site: u64) {
    let code = unsafe { std::slice::from_raw_parts((base + range.0) as *const u8, (range.1 - range.0) as usize) };
    let mut dec = Decoder::with_ip(64, code, base + range.0, DecoderOptions::NONE);
    let mut fmt = iced_x86::FastFormatter::new();
    let mut ins = Instruction::default();
    let mut out = String::new();
    while dec.can_decode() {
        dec.decode_out(&mut ins);
        let rel = ins.ip() - base;
        if rel + 96 >= site && rel <= site + 32 {
            let mut t = String::new();
            fmt.format(&ins, &mut t);
            let off = (ins.ip() - base - range.0) as usize;
            let bytes: String = code[off..off + ins.len()].iter().map(|b| format!("{b:02x}")).collect();
            out.push_str(&format!("    {:#x}: {:<24} {}\n", rel, bytes, t));
        }
    }
    crate::info!("function {:#x}..{:#x} around {:#x}:\n{}", range.0, range.1, site, out);
}
