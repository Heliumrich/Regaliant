//! Mutexes, condition variables, rwlocks, semaphores and event flags.
//!
//! Guest objects are pointer-sized handles; the guest variable holds a pointer to
//! a host object. Zero (and small constants) are static initializers that are
//! created lazily on first use.

use super::errno::*;
use super::sce;
use crate::hle::{Registry, cstr};
use crate::{reg, sysv};
use parking_lot::{Condvar, Mutex};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

const MUTEX_MAGIC: u64 = 0x4242_4D55_5445_5801;
const COND_MAGIC: u64 = 0x4242_434F_4E44_5601;
const RW_MAGIC: u64 = 0x4242_5257_4C4B_0001;
const SEMA_MAGIC: u64 = 0x4242_5345_4D41_0001;
const EF_MAGIC: u64 = 0x4242_4546_4C47_0001;
const MATTR_MAGIC: u64 = 0x4242_4D41_5454_0001;

pub const MUTEX_ERRORCHECK: i32 = 1;
pub const MUTEX_RECURSIVE: i32 = 2;
pub const MUTEX_NORMAL: i32 = 3;
pub const MUTEX_ADAPTIVE: i32 = 4;

static INIT_LOCK: Mutex<()> = Mutex::new(());

fn tid() -> u64 {
    crate::thread::current().map(|t| t.id).unwrap_or(u64::MAX)
}

/// Relative timeout in microseconds -> deadline.
fn deadline_us(us: u64) -> Instant {
    Instant::now() + Duration::from_micros(us)
}

/// Absolute CLOCK_REALTIME timespec -> deadline.
unsafe fn deadline_abs(ts: *const i64) -> Option<Instant> {
    if ts.is_null() {
        return None;
    }
    let (sec, nsec) = unsafe { (*ts, *ts.add(1)) };
    let target = Duration::new(sec.max(0) as u64, nsec.clamp(0, 999_999_999) as u32);
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default();
    Some(Instant::now() + target.saturating_sub(now))
}

// ---------------------------------------------------------------------------
// Mutex

pub struct GMutex {
    magic: u64,
    kind: i32,
    name: String,
    state: Mutex<(u64, u32)>, // (owner thread id, recursion count)
    cv: Condvar,
}

#[repr(C)]
struct MutexAttr {
    magic: u64,
    kind: i32,
    protocol: i32,
}

impl GMutex {
    fn new(kind: i32, name: &str) -> &'static GMutex {
        Box::leak(Box::new(GMutex { magic: MUTEX_MAGIC, kind, name: name.to_string(), state: Mutex::new((0, 0)), cv: Condvar::new() }))
    }

    pub fn lock(&self, deadline: Option<Instant>, try_only: bool) -> i32 {
        let me = tid();
        let mut s = self.state.lock();
        if s.0 == me {
            return match self.kind {
                MUTEX_RECURSIVE => {
                    s.1 += 1;
                    0
                }
                MUTEX_ERRORCHECK => EDEADLK,
                _ => {
                    if try_only {
                        EBUSY
                    } else {
                        crate::error!("self-deadlock on non-recursive mutex '{}'", self.name);
                        EDEADLK
                    }
                }
            };
        }
        let mut _w = None;
        while s.0 != 0 {
            if try_only {
                return EBUSY;
            }
            if _w.is_none() {
                _w = Some(crate::thread::waiting("mutex", self as *const _ as u64, Some(mutex_detail)));
            }
            match deadline {
                Some(d) => {
                    if self.cv.wait_until(&mut s, d).timed_out() && s.0 != 0 {
                        return ETIMEDOUT;
                    }
                }
                None => self.cv.wait(&mut s),
            }
        }
        *s = (me, 1);
        0
    }

    pub fn unlock(&self) -> i32 {
        let me = tid();
        let mut s = self.state.lock();
        if s.0 != me {
            return EPERM;
        }
        s.1 -= 1;
        if s.1 == 0 {
            s.0 = 0;
            drop(s);
            self.cv.notify_one();
        }
        0
    }

    /// Fully releases the mutex (for condvar waits); returns the recursion count.
    fn release_all(&self) -> Result<u32, i32> {
        let me = tid();
        let mut s = self.state.lock();
        if s.0 != me {
            return Err(EPERM);
        }
        let n = s.1;
        *s = (0, 0);
        drop(s);
        self.cv.notify_one();
        Ok(n)
    }

    fn reacquire(&self, count: u32) {
        let me = tid();
        let mut s = self.state.lock();
        let _w = if s.0 != 0 { Some(crate::thread::waiting("mutex (after cond wait)", self as *const _ as u64, Some(mutex_detail))) } else { None };
        while s.0 != 0 {
            self.cv.wait(&mut s);
        }
        *s = (me, count);
    }
}

fn mutex_detail(p: u64) -> String {
    let m = unsafe { &*(p as *const GMutex) };
    match m.state.try_lock() {
        Some(s) if s.0 != 0 => format!("'{}' owned by thread {} (depth {})", m.name, crate::thread::name_of(s.0), s.1),
        Some(_) => format!("'{}' (free)", m.name),
        None => format!("'{}'", m.name),
    }
}

unsafe fn mutex_from(slot: *mut u64) -> Result<&'static GMutex, i32> {
    if slot.is_null() {
        return Err(EINVAL);
    }
    let a = unsafe { AtomicU64::from_ptr(slot) };
    let v = a.load(Ordering::Acquire);
    if v > 0x10 {
        let m = unsafe { &*(v as *const GMutex) };
        return if m.magic == MUTEX_MAGIC { Ok(m) } else { Err(EINVAL) };
    }
    let _g = INIT_LOCK.lock();
    let v = a.load(Ordering::Acquire);
    if v > 0x10 {
        return Ok(unsafe { &*(v as *const GMutex) });
    }
    let kind = match v {
        0 => MUTEX_ERRORCHECK,
        1 => MUTEX_ADAPTIVE,
        2 => MUTEX_RECURSIVE,
        3 => MUTEX_NORMAL,
        _ => MUTEX_ERRORCHECK,
    };
    let m = GMutex::new(kind, "static");
    a.store(m as *const _ as u64, Ordering::Release);
    Ok(m)
}

unsafe fn mutex_init(slot: *mut u64, attr: *mut u64, name: &str) -> i32 {
    if slot.is_null() {
        return EINVAL;
    }
    let mut kind = MUTEX_ERRORCHECK;
    if !attr.is_null() {
        let p = unsafe { *attr } as *const MutexAttr;
        if !p.is_null() && unsafe { (*p).magic } == MATTR_MAGIC {
            kind = unsafe { (*p).kind };
        }
    }
    let m = GMutex::new(kind, name);
    unsafe { *slot = m as *const _ as u64 };
    0
}

unsafe fn mattr_init(slot: *mut u64) -> i32 {
    if slot.is_null() {
        return EINVAL;
    }
    let a = Box::into_raw(Box::new(MutexAttr { magic: MATTR_MAGIC, kind: MUTEX_ERRORCHECK, protocol: 0 }));
    unsafe { *slot = a as u64 };
    0
}

unsafe fn mattr_ref<'a>(slot: *mut u64) -> Option<&'a mut MutexAttr> {
    if slot.is_null() {
        return None;
    }
    let p = unsafe { *slot } as *mut MutexAttr;
    if p.is_null() || unsafe { (*p).magic } != MATTR_MAGIC {
        return None;
    }
    Some(unsafe { &mut *p })
}

// ---------------------------------------------------------------------------
// Condition variable

pub struct GCond {
    magic: u64,
    seq: Mutex<u64>,
    cv: Condvar,
}

unsafe fn cond_from(slot: *mut u64) -> Result<&'static GCond, i32> {
    if slot.is_null() {
        return Err(EINVAL);
    }
    let a = unsafe { AtomicU64::from_ptr(slot) };
    let v = a.load(Ordering::Acquire);
    if v > 0x10 {
        let c = unsafe { &*(v as *const GCond) };
        return if c.magic == COND_MAGIC { Ok(c) } else { Err(EINVAL) };
    }
    let _g = INIT_LOCK.lock();
    let v = a.load(Ordering::Acquire);
    if v > 0x10 {
        return Ok(unsafe { &*(v as *const GCond) });
    }
    let c: &'static GCond = Box::leak(Box::new(GCond { magic: COND_MAGIC, seq: Mutex::new(0), cv: Condvar::new() }));
    a.store(c as *const _ as u64, Ordering::Release);
    Ok(c)
}

impl GCond {
    fn wait(&self, m: &GMutex, deadline: Option<Instant>) -> i32 {
        let mut g = self.seq.lock();
        let start = *g;
        let count = match m.release_all() {
            Ok(n) => n,
            Err(e) => return e,
        };
        let mut timed_out = false;
        let _w = crate::thread::waiting(if deadline.is_some() { "cond (timed)" } else { "cond" }, self as *const _ as u64, None);
        loop {
            match deadline {
                Some(d) => {
                    if self.cv.wait_until(&mut g, d).timed_out() {
                        timed_out = *g == start;
                        break;
                    }
                }
                None => self.cv.wait(&mut g),
            }
            if *g != start {
                break;
            }
        }
        drop(g);
        m.reacquire(count);
        if timed_out { ETIMEDOUT } else { 0 }
    }
    fn signal(&self) {
        let mut g = self.seq.lock();
        *g = g.wrapping_add(1);
        self.cv.notify_one();
    }
    fn broadcast(&self) {
        let mut g = self.seq.lock();
        *g = g.wrapping_add(1);
        self.cv.notify_all();
    }
}

// ---------------------------------------------------------------------------
// Read/write lock

pub struct GRwLock {
    magic: u64,
    state: Mutex<(u32, u64)>, // (readers, writer tid)
    cv: Condvar,
}

fn rw_detail(p: u64) -> String {
    let l = unsafe { &*(p as *const GRwLock) };
    match l.state.try_lock() {
        Some(s) if s.1 != 0 => format!("write-locked by thread {}", crate::thread::name_of(s.1)),
        Some(s) => format!("{} reader(s)", s.0),
        None => String::new(),
    }
}

unsafe fn rw_from(slot: *mut u64) -> Result<&'static GRwLock, i32> {
    if slot.is_null() {
        return Err(EINVAL);
    }
    let a = unsafe { AtomicU64::from_ptr(slot) };
    let v = a.load(Ordering::Acquire);
    if v > 0x10 {
        let c = unsafe { &*(v as *const GRwLock) };
        return if c.magic == RW_MAGIC { Ok(c) } else { Err(EINVAL) };
    }
    let _g = INIT_LOCK.lock();
    let v = a.load(Ordering::Acquire);
    if v > 0x10 {
        return Ok(unsafe { &*(v as *const GRwLock) });
    }
    let c: &'static GRwLock = Box::leak(Box::new(GRwLock { magic: RW_MAGIC, state: Mutex::new((0, 0)), cv: Condvar::new() }));
    a.store(c as *const _ as u64, Ordering::Release);
    Ok(c)
}

impl GRwLock {
    fn rdlock(&self, deadline: Option<Instant>, try_only: bool) -> i32 {
        let mut s = self.state.lock();
        let mut _w = None;
        while s.1 != 0 {
            if s.1 == tid() {
                return EDEADLK;
            }
            if try_only {
                return EBUSY;
            }
            if _w.is_none() {
                _w = Some(crate::thread::waiting("rwlock (read)", self as *const _ as u64, Some(rw_detail)));
            }
            match deadline {
                Some(d) => {
                    if self.cv.wait_until(&mut s, d).timed_out() && s.1 != 0 {
                        return ETIMEDOUT;
                    }
                }
                None => self.cv.wait(&mut s),
            }
        }
        s.0 += 1;
        0
    }
    fn wrlock(&self, deadline: Option<Instant>, try_only: bool) -> i32 {
        let me = tid();
        let mut s = self.state.lock();
        if s.1 == me {
            return EDEADLK;
        }
        let mut _w = None;
        while s.0 != 0 || s.1 != 0 {
            if try_only {
                return EBUSY;
            }
            if _w.is_none() {
                _w = Some(crate::thread::waiting("rwlock (write)", self as *const _ as u64, Some(rw_detail)));
            }
            match deadline {
                Some(d) => {
                    if self.cv.wait_until(&mut s, d).timed_out() && (s.0 != 0 || s.1 != 0) {
                        return ETIMEDOUT;
                    }
                }
                None => self.cv.wait(&mut s),
            }
        }
        s.1 = me;
        0
    }
    fn unlock(&self) -> i32 {
        let mut s = self.state.lock();
        if s.1 != 0 {
            if s.1 != tid() {
                return EPERM;
            }
            s.1 = 0;
        } else if s.0 > 0 {
            s.0 -= 1;
        } else {
            return EPERM;
        }
        drop(s);
        self.cv.notify_all();
        0
    }
}

// ---------------------------------------------------------------------------
// Semaphore

pub struct GSema {
    magic: u64,
    name: String,
    count: Mutex<i32>,
    max: i32,
    cv: Condvar,
}

fn sema_detail(p: u64) -> String {
    let s = unsafe { &*(p as *const GSema) };
    match s.count.try_lock() {
        Some(c) => format!("'{}' count {}/{}", s.name, *c, s.max),
        None => format!("'{}'", s.name),
    }
}

unsafe fn sema_from<'a>(h: u64) -> Option<&'a GSema> {
    if h == 0 {
        return None;
    }
    let s = unsafe { &*(h as *const GSema) };
    if s.magic == SEMA_MAGIC { Some(s) } else { None }
}

// ---------------------------------------------------------------------------
// Event flag

pub struct GEventFlag {
    magic: u64,
    name: String,
    bits: Mutex<u64>,
    cv: Condvar,
}

const EVF_WAITMODE_AND: u32 = 0x01;
const EVF_WAITMODE_OR: u32 = 0x02;
const EVF_WAITMODE_CLEAR_ALL: u32 = 0x10;
const EVF_WAITMODE_CLEAR_PAT: u32 = 0x20;

fn ef_detail(p: u64) -> String {
    let e = unsafe { &*(p as *const GEventFlag) };
    match e.bits.try_lock() {
        Some(b) => format!("'{}' bits {:#x}", e.name, *b),
        None => format!("'{}'", e.name),
    }
}

unsafe fn ef_from<'a>(h: u64) -> Option<&'a GEventFlag> {
    if h == 0 {
        return None;
    }
    let s = unsafe { &*(h as *const GEventFlag) };
    if s.magic == EF_MAGIC { Some(s) } else { None }
}

fn ef_satisfied(bits: u64, pat: u64, mode: u32) -> bool {
    if mode & EVF_WAITMODE_AND != 0 { bits & pat == pat } else { bits & pat != 0 }
}

fn sce_ret(e: i32) -> i32 {
    if e == 0 { 0 } else { sce(e) }
}

sysv! {
    fn scePthreadMutexattrInit(attr: *mut u64) -> i32 { sce_ret(mattr_init(attr)) }
    fn scePthreadMutexattrDestroy(attr: *mut u64) -> i32 {
        if let Some(a) = mattr_ref(attr) { a.magic = 0; drop(Box::from_raw(a as *mut MutexAttr)); *attr = 0; 0 } else { sce(EINVAL) }
    }
    fn scePthreadMutexattrSettype(attr: *mut u64, kind: i32) -> i32 {
        match mattr_ref(attr) { Some(a) => { a.kind = kind; 0 } None => sce(EINVAL) }
    }
    fn scePthreadMutexattrGettype(attr: *mut u64, kind: *mut i32) -> i32 {
        match mattr_ref(attr) { Some(a) => { *kind = a.kind; 0 } None => sce(EINVAL) }
    }
    fn scePthreadMutexattrSetprotocol(attr: *mut u64, p: i32) -> i32 {
        match mattr_ref(attr) { Some(a) => { a.protocol = p; 0 } None => sce(EINVAL) }
    }
    fn scePthreadMutexattrSetpshared(attr: *mut u64, p: i32) -> i32 {
        match mattr_ref(attr) { Some(_) => 0, None => sce(EINVAL) }
    }
    fn scePthreadMutexInit(m: *mut u64, attr: *mut u64, name: *const u8) -> i32 {
        sce_ret(mutex_init(m, attr, if name.is_null() { "mutex" } else { cstr(name) }))
    }
    fn scePthreadMutexDestroy(m: *mut u64) -> i32 {
        if m.is_null() { return sce(EINVAL); }
        *m = 0;
        0
    }
    fn scePthreadMutexLock(m: *mut u64) -> i32 {
        match mutex_from(m) { Ok(m) => sce_ret(m.lock(None, false)), Err(e) => sce(e) }
    }
    fn scePthreadMutexTrylock(m: *mut u64) -> i32 {
        match mutex_from(m) { Ok(m) => sce_ret(m.lock(None, true)), Err(e) => sce(e) }
    }
    fn scePthreadMutexTimedlock(m: *mut u64, usec: u32) -> i32 {
        match mutex_from(m) { Ok(m) => sce_ret(m.lock(Some(deadline_us(usec as u64)), false)), Err(e) => sce(e) }
    }
    fn scePthreadMutexUnlock(m: *mut u64) -> i32 {
        match mutex_from(m) { Ok(m) => sce_ret(m.unlock()), Err(e) => sce(e) }
    }
    fn scePthreadCondattrInit(attr: *mut u64) -> i32 { if !attr.is_null() { *attr = 1; } 0 }
    fn scePthreadCondattrDestroy(attr: *mut u64) -> i32 { 0 }
    fn scePthreadCondInit(c: *mut u64, attr: *mut u64, name: *const u8) -> i32 {
        if c.is_null() { return sce(EINVAL); }
        *c = 0;
        match cond_from(c) { Ok(_) => 0, Err(e) => sce(e) }
    }
    fn scePthreadCondDestroy(c: *mut u64) -> i32 { if !c.is_null() { *c = 0; } 0 }
    fn scePthreadCondWait(c: *mut u64, m: *mut u64) -> i32 {
        match (cond_from(c), mutex_from(m)) { (Ok(c), Ok(m)) => sce_ret(c.wait(m, None)), _ => sce(EINVAL) }
    }
    fn scePthreadCondTimedwait(c: *mut u64, m: *mut u64, usec: u32) -> i32 {
        match (cond_from(c), mutex_from(m)) { (Ok(c), Ok(m)) => sce_ret(c.wait(m, Some(deadline_us(usec as u64)))), _ => sce(EINVAL) }
    }
    fn scePthreadCondSignal(c: *mut u64) -> i32 { match cond_from(c) { Ok(c) => { c.signal(); 0 } Err(e) => sce(e) } }
    fn scePthreadCondBroadcast(c: *mut u64) -> i32 { match cond_from(c) { Ok(c) => { c.broadcast(); 0 } Err(e) => sce(e) } }
    fn scePthreadRwlockInit(rw: *mut u64, attr: *mut u64, name: *const u8) -> i32 {
        if rw.is_null() { return sce(EINVAL); }
        *rw = 0;
        match rw_from(rw) { Ok(_) => 0, Err(e) => sce(e) }
    }
    fn scePthreadRwlockDestroy(rw: *mut u64) -> i32 { if !rw.is_null() { *rw = 0; } 0 }
    fn scePthreadRwlockRdlock(rw: *mut u64) -> i32 { match rw_from(rw) { Ok(l) => sce_ret(l.rdlock(None, false)), Err(e) => sce(e) } }
    fn scePthreadRwlockWrlock(rw: *mut u64) -> i32 { match rw_from(rw) { Ok(l) => sce_ret(l.wrlock(None, false)), Err(e) => sce(e) } }
    fn scePthreadRwlockTryrdlock(rw: *mut u64) -> i32 { match rw_from(rw) { Ok(l) => sce_ret(l.rdlock(None, true)), Err(e) => sce(e) } }
    fn scePthreadRwlockTrywrlock(rw: *mut u64) -> i32 { match rw_from(rw) { Ok(l) => sce_ret(l.wrlock(None, true)), Err(e) => sce(e) } }
    fn scePthreadRwlockTimedrdlock(rw: *mut u64, usec: u32) -> i32 {
        match rw_from(rw) { Ok(l) => sce_ret(l.rdlock(Some(deadline_us(usec as u64)), false)), Err(e) => sce(e) }
    }
    fn scePthreadRwlockTimedwrlock(rw: *mut u64, usec: u32) -> i32 {
        match rw_from(rw) { Ok(l) => sce_ret(l.wrlock(Some(deadline_us(usec as u64)), false)), Err(e) => sce(e) }
    }
    fn scePthreadRwlockUnlock(rw: *mut u64) -> i32 { match rw_from(rw) { Ok(l) => sce_ret(l.unlock()), Err(e) => sce(e) } }

    // POSIX
    fn pthread_mutexattr_init(attr: *mut u64) -> i32 { mattr_init(attr) }
    fn pthread_mutexattr_destroy(attr: *mut u64) -> i32 {
        if let Some(a) = mattr_ref(attr) { a.magic = 0; drop(Box::from_raw(a as *mut MutexAttr)); *attr = 0; 0 } else { EINVAL }
    }
    fn pthread_mutexattr_settype(attr: *mut u64, kind: i32) -> i32 {
        match mattr_ref(attr) { Some(a) => { a.kind = kind; 0 } None => EINVAL }
    }
    fn pthread_mutexattr_setprotocol(attr: *mut u64, p: i32) -> i32 {
        match mattr_ref(attr) { Some(a) => { a.protocol = p; 0 } None => EINVAL }
    }
    fn pthread_mutex_init(m: *mut u64, attr: *mut u64) -> i32 { mutex_init(m, attr, "pmutex") }
    fn pthread_mutex_destroy(m: *mut u64) -> i32 { if m.is_null() { EINVAL } else { *m = 0; 0 } }
    fn pthread_mutex_lock(m: *mut u64) -> i32 { match mutex_from(m) { Ok(m) => m.lock(None, false), Err(e) => e } }
    fn pthread_mutex_trylock(m: *mut u64) -> i32 { match mutex_from(m) { Ok(m) => m.lock(None, true), Err(e) => e } }
    fn pthread_mutex_timedlock(m: *mut u64, ts: *const i64) -> i32 {
        match mutex_from(m) { Ok(m) => m.lock(deadline_abs(ts), false), Err(e) => e }
    }
    fn pthread_mutex_unlock(m: *mut u64) -> i32 { match mutex_from(m) { Ok(m) => m.unlock(), Err(e) => e } }
    fn pthread_cond_init(c: *mut u64, attr: *mut u64) -> i32 {
        if c.is_null() { return EINVAL; }
        *c = 0;
        match cond_from(c) { Ok(_) => 0, Err(e) => e }
    }
    fn pthread_cond_destroy(c: *mut u64) -> i32 { if !c.is_null() { *c = 0; } 0 }
    fn pthread_cond_wait(c: *mut u64, m: *mut u64) -> i32 {
        match (cond_from(c), mutex_from(m)) { (Ok(c), Ok(m)) => c.wait(m, None), _ => EINVAL }
    }
    fn pthread_cond_timedwait(c: *mut u64, m: *mut u64, ts: *const i64) -> i32 {
        match (cond_from(c), mutex_from(m)) { (Ok(c), Ok(m)) => c.wait(m, deadline_abs(ts)), _ => EINVAL }
    }
    fn pthread_cond_signal(c: *mut u64) -> i32 { match cond_from(c) { Ok(c) => { c.signal(); 0 } Err(e) => e } }
    fn pthread_cond_broadcast(c: *mut u64) -> i32 { match cond_from(c) { Ok(c) => { c.broadcast(); 0 } Err(e) => e } }
    fn pthread_rwlock_init(rw: *mut u64, attr: *mut u64) -> i32 {
        if rw.is_null() { return EINVAL; }
        *rw = 0;
        match rw_from(rw) { Ok(_) => 0, Err(e) => e }
    }
    fn pthread_rwlock_destroy(rw: *mut u64) -> i32 { if !rw.is_null() { *rw = 0; } 0 }
    fn pthread_rwlock_rdlock(rw: *mut u64) -> i32 { match rw_from(rw) { Ok(l) => l.rdlock(None, false), Err(e) => e } }
    fn pthread_rwlock_wrlock(rw: *mut u64) -> i32 { match rw_from(rw) { Ok(l) => l.wrlock(None, false), Err(e) => e } }
    fn pthread_rwlock_unlock(rw: *mut u64) -> i32 { match rw_from(rw) { Ok(l) => l.unlock(), Err(e) => e } }

    // Semaphores
    fn sceKernelCreateSema(out: *mut u64, name: *const u8, attr: u32, init: i32, max: i32, opt: u64) -> i32 {
        if out.is_null() || init < 0 || max <= 0 || init > max { return sce(EINVAL); }
        let s: &'static GSema = Box::leak(Box::new(GSema { magic: SEMA_MAGIC, name: cstr(name).to_string(), count: Mutex::new(init), max, cv: Condvar::new() }));
        *out = s as *const _ as u64;
        0
    }
    fn sceKernelDeleteSema(h: u64) -> i32 { match sema_from(h) { Some(s) => { s.cv.notify_all(); 0 } None => sce(ESRCH) } }
    fn sceKernelWaitSema(h: u64, need: i32, timeout: *mut u32) -> i32 {
        let Some(s) = sema_from(h) else { return sce(ESRCH) };
        if need < 1 || need > s.max { return sce(EINVAL); }
        let start = Instant::now();
        let deadline = if timeout.is_null() { None } else { Some(deadline_us(*timeout as u64)) };
        let mut c = s.count.lock();
        let _w = if *c < need { Some(crate::thread::waiting("sema", h, Some(sema_detail))) } else { None };
        while *c < need {
            match deadline {
                Some(d) => {
                    if s.cv.wait_until(&mut c, d).timed_out() && *c < need {
                        *timeout = 0;
                        return sce(ETIMEDOUT);
                    }
                }
                None => s.cv.wait(&mut c),
            }
        }
        *c -= need;
        if !timeout.is_null() {
            let el = start.elapsed().as_micros() as u32;
            *timeout = (*timeout).saturating_sub(el);
        }
        0
    }
    fn sceKernelPollSema(h: u64, need: i32) -> i32 {
        let Some(s) = sema_from(h) else { return sce(ESRCH) };
        let mut c = s.count.lock();
        if *c < need { return sce(EBUSY); }
        *c -= need;
        0
    }
    fn sceKernelSignalSema(h: u64, n: i32) -> i32 {
        let Some(s) = sema_from(h) else { return sce(ESRCH) };
        let mut c = s.count.lock();
        if *c + n > s.max { return sce(EINVAL); }
        *c += n;
        s.cv.notify_all();
        0
    }
    fn sceKernelCancelSema(h: u64, count: i32, waiters: *mut i32) -> i32 {
        let Some(s) = sema_from(h) else { return sce(ESRCH) };
        if !waiters.is_null() { *waiters = 0; }
        if count >= 0 { *s.count.lock() = count; }
        s.cv.notify_all();
        0
    }

    // Event flags
    fn sceKernelCreateEventFlag(out: *mut u64, name: *const u8, attr: u32, init: u64, opt: u64) -> i32 {
        if out.is_null() { return sce(EINVAL); }
        let e: &'static GEventFlag = Box::leak(Box::new(GEventFlag { magic: EF_MAGIC, name: cstr(name).to_string(), bits: Mutex::new(init), cv: Condvar::new() }));
        *out = e as *const _ as u64;
        0
    }
    fn sceKernelDeleteEventFlag(h: u64) -> i32 { match ef_from(h) { Some(_) => 0, None => sce(ESRCH) } }
    fn sceKernelSetEventFlag(h: u64, bits: u64) -> i32 {
        let Some(e) = ef_from(h) else { return sce(ESRCH) };
        *e.bits.lock() |= bits;
        e.cv.notify_all();
        0
    }
    fn sceKernelClearEventFlag(h: u64, bits: u64) -> i32 {
        let Some(e) = ef_from(h) else { return sce(ESRCH) };
        *e.bits.lock() &= bits;
        0
    }
    fn sceKernelPollEventFlag(h: u64, pat: u64, mode: u32, result: *mut u64) -> i32 {
        let Some(e) = ef_from(h) else { return sce(ESRCH) };
        let mut b = e.bits.lock();
        if !result.is_null() { *result = *b; }
        if !ef_satisfied(*b, pat, mode) { return sce(EBUSY); }
        if mode & EVF_WAITMODE_CLEAR_ALL != 0 { *b = 0; } else if mode & EVF_WAITMODE_CLEAR_PAT != 0 { *b &= !pat; }
        0
    }
    fn sceKernelWaitEventFlag(h: u64, pat: u64, mode: u32, result: *mut u64, timeout: *mut u32) -> i32 {
        let Some(e) = ef_from(h) else { return sce(ESRCH) };
        if mode & (EVF_WAITMODE_AND | EVF_WAITMODE_OR) == 0 { return sce(EINVAL); }
        let deadline = if timeout.is_null() { None } else { Some(deadline_us(*timeout as u64)) };
        let mut b = e.bits.lock();
        let _w = if !ef_satisfied(*b, pat, mode) { Some(crate::thread::waiting("event flag", h, Some(ef_detail))) } else { None };
        while !ef_satisfied(*b, pat, mode) {
            match deadline {
                Some(d) => {
                    if e.cv.wait_until(&mut b, d).timed_out() && !ef_satisfied(*b, pat, mode) {
                        if !result.is_null() { *result = *b; }
                        *timeout = 0;
                        return sce(ETIMEDOUT);
                    }
                }
                None => e.cv.wait(&mut b),
            }
        }
        if !result.is_null() { *result = *b; }
        if mode & EVF_WAITMODE_CLEAR_ALL != 0 { *b = 0; } else if mode & EVF_WAITMODE_CLEAR_PAT != 0 { *b &= !pat; }
        0
    }
    fn sceKernelCancelEventFlag(h: u64, set: u64, waiters: *mut i32) -> i32 {
        let Some(e) = ef_from(h) else { return sce(ESRCH) };
        *e.bits.lock() = set;
        if !waiters.is_null() { *waiters = 0; }
        e.cv.notify_all();
        0
    }
}

pub fn register(r: &mut Registry) {
    reg!(r, "libkernel", [
        scePthreadMutexattrInit, scePthreadMutexattrDestroy, scePthreadMutexattrSettype, scePthreadMutexattrGettype,
        scePthreadMutexattrSetprotocol, scePthreadMutexattrSetpshared, scePthreadMutexInit, scePthreadMutexDestroy,
        scePthreadMutexLock, scePthreadMutexTrylock, scePthreadMutexTimedlock, scePthreadMutexUnlock,
        scePthreadCondattrInit, scePthreadCondattrDestroy, scePthreadCondInit, scePthreadCondDestroy,
        scePthreadCondWait, scePthreadCondTimedwait, scePthreadCondSignal, scePthreadCondBroadcast,
        scePthreadRwlockInit, scePthreadRwlockDestroy, scePthreadRwlockRdlock, scePthreadRwlockWrlock,
        scePthreadRwlockTryrdlock, scePthreadRwlockTrywrlock, scePthreadRwlockTimedrdlock,
        scePthreadRwlockTimedwrlock, scePthreadRwlockUnlock, sceKernelCreateSema, sceKernelDeleteSema,
        sceKernelWaitSema, sceKernelPollSema, sceKernelSignalSema, sceKernelCancelSema, sceKernelCreateEventFlag,
        sceKernelDeleteEventFlag, sceKernelSetEventFlag, sceKernelClearEventFlag, sceKernelPollEventFlag,
        sceKernelWaitEventFlag, sceKernelCancelEventFlag,
    ]);
    reg!(r, "libScePosix", [
        pthread_mutexattr_init, pthread_mutexattr_destroy, pthread_mutexattr_settype, pthread_mutexattr_setprotocol,
        pthread_mutex_init, pthread_mutex_destroy, pthread_mutex_lock, pthread_mutex_trylock, pthread_mutex_timedlock,
        pthread_mutex_unlock, pthread_cond_init, pthread_cond_destroy, pthread_cond_wait, pthread_cond_timedwait,
        pthread_cond_signal, pthread_cond_broadcast, pthread_rwlock_init, pthread_rwlock_destroy,
        pthread_rwlock_rdlock, pthread_rwlock_wrlock, pthread_rwlock_unlock,
    ]);
}
