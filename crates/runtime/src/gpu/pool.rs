//! Helper threads kept for the data-parallel steps of texture uploads
//! (hashing guest ranges, detiling parts and slices). Those steps started
//! scoped threads for every upload of 1 MiB or more: ~9,000 thread starts
//! and exits in a boot to 1-1 (2026-10-26, `dsdbg --count`), each a few
//! tens of microseconds on the uploading thread's path, and under a
//! debugger each a debug event that stops the whole process.
//!
//! `for_each(n, f)` runs `f(0..n)` on the pool's threads and the caller's
//! and returns when every call has returned (the caller takes indices too,
//! so a call from a pool thread cannot wait on itself).

use parking_lot::{Condvar, Mutex};
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// One `for_each` call: its closure (lifetime erased: the caller does not
/// return before every claimed index has run), the indices handed out and
/// done, and the flag its caller waits on.
struct Job {
    f: *const (dyn Fn(usize) + Sync + 'static),
    n: usize,
    next: AtomicUsize,
    done: AtomicUsize,
    finished: Mutex<bool>,
    cv: Condvar,
}

// SAFETY: `f` is only called while the `for_each` that owns the closure
// waits for `done` to reach `n`; the closure itself is Sync.
unsafe impl Send for Job {}
unsafe impl Sync for Job {}

impl Job {
    /// Runs indices until none are left to hand out.
    fn work(&self) {
        loop {
            let k = self.next.fetch_add(1, Ordering::AcqRel);
            if k >= self.n {
                return;
            }
            // SAFETY: see `Job`.
            unsafe { (*self.f)(k) };
            if self.done.fetch_add(1, Ordering::AcqRel) + 1 == self.n {
                *self.finished.lock() = true;
                self.cv.notify_one();
            }
        }
    }
}

struct Pool {
    queue: Mutex<VecDeque<Arc<Job>>>,
    ready: Condvar,
}

static POOL: std::sync::OnceLock<Pool> = std::sync::OnceLock::new();

/// Helper threads: the CPUs the process may use, at most 8 with the caller
/// (as the scoped threads were).
pub fn threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(8)
}

fn pool() -> &'static Pool {
    POOL.get_or_init(|| {
        for k in 0..threads().saturating_sub(1) {
            let _ = std::thread::Builder::new().name(format!("gpu-pool{k}")).spawn(helper);
        }
        Pool { queue: Mutex::new(VecDeque::new()), ready: Condvar::new() }
    })
}

fn helper() {
    let p = pool();
    loop {
        let job = {
            let mut q = p.queue.lock();
            loop {
                // Jobs whose indices are all handed out leave the queue.
                while q.front().is_some_and(|j| j.next.load(Ordering::Acquire) >= j.n) {
                    q.pop_front();
                }
                if let Some(j) = q.front() {
                    break j.clone();
                }
                p.ready.wait(&mut q);
            }
        };
        job.work();
    }
}

/// Runs `f(k)` for every k in 0..n on the pool's threads and this one;
/// returns when all have returned.
pub fn for_each(n: usize, f: &(dyn Fn(usize) + Sync)) {
    if n == 0 {
        return;
    }
    if n == 1 {
        f(0);
        return;
    }
    let p = pool();
    // SAFETY: lifetime erased; this function does not return before `done`
    // reaches `n`, and no index is run after that.
    let f: *const (dyn Fn(usize) + Sync + 'static) = unsafe { std::mem::transmute(f as *const (dyn Fn(usize) + Sync)) };
    let job = Arc::new(Job { f, n, next: AtomicUsize::new(0), done: AtomicUsize::new(0), finished: Mutex::new(false), cv: Condvar::new() });
    p.queue.lock().push_back(job.clone());
    p.ready.notify_all();
    job.work();
    for _ in 0..2000 {
        if job.done.load(Ordering::Acquire) >= n {
            return;
        }
        std::hint::spin_loop();
    }
    let mut g = job.finished.lock();
    while !*g {
        job.cv.wait(&mut g);
    }
}

/// `f(k, chunk)` for the consecutive `size`-byte chunks of `dst` (the last
/// one shorter), in parallel as `for_each`.
pub fn for_each_chunk_mut(dst: &mut [u8], size: usize, f: &(dyn Fn(usize, &mut [u8]) + Sync)) {
    let size = size.max(1);
    let (ptr, len) = (dst.as_mut_ptr() as usize, dst.len());
    for_each(len.div_ceil(size), &|k| {
        let start = k * size;
        let n = size.min(len - start);
        // SAFETY: the chunks are disjoint parts of `dst`, which stays
        // borrowed until `for_each` returns.
        let chunk = unsafe { std::slice::from_raw_parts_mut((ptr as *mut u8).add(start), n) };
        f(k, chunk);
    });
}
