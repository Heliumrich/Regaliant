//! GPU emulation front end for the PS5 (AGC): submission queues, the gfx10
//! PM4 command processor (`cp`) and flips.
//!
//! Command buffers submitted through libSceAgcDriver are queued per hardware
//! queue (graphics, or a compute queue id) and executed in submission order
//! by one command-processor thread per queue, concurrently with the CPU, as
//! on the console. Draws and dispatches are decoded but not rendered yet
//! (docs/STATUS.md).

pub mod capture;
pub mod cp;
pub mod png;
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

struct Queue {
    id: u32,
    pending: Mutex<VecDeque<Vec<(u64, u32)>>>,
    cv: Condvar,
    /// Submissions completed (diagnostics).
    done: AtomicU64,
}

static QUEUES: Mutex<Vec<&'static Queue>> = Mutex::new(Vec::new());

fn queue(id: u32) -> &'static Queue {
    let mut qs = QUEUES.lock();
    if let Some(q) = qs.iter().find(|q| q.id == id) {
        return q;
    }
    let q: &'static Queue = Box::leak(Box::new(Queue { id, pending: Mutex::new(VecDeque::new()), cv: Condvar::new(), done: AtomicU64::new(0) }));
    qs.push(q);
    let name = if id == GFX { "gpu-gfx".to_string() } else { format!("gpu-acb{id:#x}") };
    crate::info!("gpu: {name} command processor started");
    crate::thread::spawn_host(&name.clone(), 8 << 20, Box::new(move || worker(q, &name))).expect("gpu queue thread");
    q
}

fn worker(q: &'static Queue, name: &str) {
    crate::log::set_thread_name(name);
    crate::exception::arm_watch_current_thread();
    let mut cp = cp::Cp::new(name, q.id != GFX);
    loop {
        let batch = {
            let mut g = q.pending.lock();
            loop {
                if let Some(b) = g.pop_front() {
                    break b;
                }
                q.cv.wait(&mut g);
            }
        };
        for (addr, size) in batch {
            cp.run(addr, size);
        }
        q.done.fetch_add(1, Ordering::Relaxed);
    }
}

/// Queues command buffers (address, size in dwords) for execution on `queue_id`.
pub fn submit(queue_id: u32, buffers: Vec<(u64, u32)>) {
    if buffers.is_empty() {
        return;
    }
    SUBMITS.fetch_add(1, Ordering::Relaxed);
    let q = queue(queue_id);
    q.pending.lock().push_back(buffers);
    q.cv.notify_one();
}

/// One line per queue for hang reports.
pub fn status() -> String {
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
    }
    s.push_str(&format!(
        "    gpu totals: {} submits, {} packets, {} draws, {} dispatches, {} EOP interrupts, {} flips\n",
        SUBMITS.load(Ordering::Relaxed),
        cp::PACKETS.load(Ordering::Relaxed),
        cp::DRAWS.load(Ordering::Relaxed),
        cp::DISPATCHES.load(Ordering::Relaxed),
        cp::EOP_INTERRUPTS.load(Ordering::Relaxed),
        FRAMES.load(Ordering::Relaxed)
    ));
    s
}
