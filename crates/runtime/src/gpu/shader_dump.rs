//! Development aid: disassembles each distinct shader once into the run dir.

use parking_lot::Mutex;
use std::collections::HashSet;
use std::path::PathBuf;

static SEEN: Mutex<Option<HashSet<u64>>> = Mutex::new(None);
pub static DIR: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();

/// Returns the guest code words at `addr`, clamped to its mapping.
pub fn code_slice(addr: u64) -> Option<&'static [u32]> {
    let end = {
        let st = crate::mem::MEM.lock();
        let v = st.find(addr)?;
        v.end()
    };
    let words = ((end - addr) / 4).min(1 << 18) as usize;
    Some(unsafe { std::slice::from_raw_parts(addr as *const u32, words) })
}

pub fn dump(stage: &str, addr: u64, user_data: &[u32]) {
    let Some(dir) = DIR.get() else { return };
    {
        let mut g = SEEN.lock();
        if !g.get_or_insert_with(HashSet::new).insert(addr) {
            return;
        }
    }
    let Some(code) = code_slice(addr) else {
        crate::warn!("{stage} shader at {addr:#x} is not mapped");
        return;
    };
    let mut text = format!("; {stage} shader at {addr:#x}\n; user data: {:x?}\n", user_data);
    match gcn::decode_program(code) {
        Ok(p) => {
            text.push_str(&gcn::listing(&p));
            // Follow s_swappc_b64 into the fetch subroutine when its address is in user data.
            for ins in &p {
                if ins.enc == gcn::Enc::Sop1 && ins.op == 33 {
                    let src = ins.ssrc0() as usize;
                    if src + 1 < user_data.len() {
                        let target = (user_data[src] as u64) | ((user_data[src + 1] as u64) << 32);
                        text.push_str(&format!("; s_swappc target s[{}:{}] = {:#x}\n", src, src + 1, target));
                        if let Some(c2) = code_slice(target) {
                            match gcn::decode_program(c2) {
                                Ok(f) => text.push_str(&gcn::listing(&f)),
                                Err(e) => text.push_str(&format!("; fetch decode error: {}\n", e.0)),
                            }
                        }
                    }
                }
            }
        }
        Err(e) => text.push_str(&format!("; decode error: {}\n", e.0)),
    }
    let _ = std::fs::create_dir_all(dir);
    let _ = std::fs::write(dir.join(format!("{stage}_{addr:x}.txt")), text);
}
