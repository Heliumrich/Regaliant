//! Event queues (kqueue-like `SceKernelEqueue`).

use super::errno::*;
use super::sce;
use crate::hle::{Registry, cstr};
use crate::{reg, sysv};
use parking_lot::{Condvar, Mutex};
use std::time::{Duration, Instant};

const EQ_MAGIC: u64 = 0x4242_4551_5545_0001;

pub const EVFILT_TIMER: i16 = -7;
pub const EVFILT_USER: i16 = -11;
pub const EVFILT_DISPLAY: i16 = -13;
pub const EVFILT_GRAPHICS: i16 = -14;
pub const EVFILT_HRTIMER: i16 = -15;

pub const EV_CLEAR: u16 = 0x20;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Kevent {
    pub ident: u64,
    pub filter: i16,
    pub flags: u16,
    pub fflags: u32,
    pub data: u64,
    pub udata: u64,
}

#[derive(Clone, Debug)]
struct Entry {
    ev: Kevent,
    triggered: bool,
}

pub struct Equeue {
    magic: u64,
    pub name: String,
    events: Mutex<Vec<Entry>>,
    cv: Condvar,
}

impl Equeue {
    pub fn add(&self, ident: u64, filter: i16, flags: u16, udata: u64) {
        let mut e = self.events.lock();
        e.retain(|x| !(x.ev.ident == ident && x.ev.filter == filter));
        e.push(Entry { ev: Kevent { ident, filter, flags, fflags: 0, data: 0, udata }, triggered: false });
    }
    pub fn remove(&self, ident: u64, filter: i16) -> bool {
        let mut e = self.events.lock();
        let n = e.len();
        e.retain(|x| !(x.ev.ident == ident && x.ev.filter == filter));
        n != e.len()
    }
    /// Triggers matching events; returns whether any matched.
    pub fn trigger(&self, ident: u64, filter: i16, data: Option<u64>, udata: Option<u64>) -> bool {
        let mut e = self.events.lock();
        let mut any = false;
        for x in e.iter_mut().filter(|x| x.ev.ident == ident && x.ev.filter == filter) {
            x.triggered = true;
            if let Some(d) = data {
                x.ev.data = d;
            }
            if let Some(u) = udata {
                x.ev.udata = u;
            }
            any = true;
        }
        drop(e);
        if any {
            self.cv.notify_all();
        }
        any
    }
    pub fn has(&self, ident: u64, filter: i16) -> bool {
        self.events.lock().iter().any(|x| x.ev.ident == ident && x.ev.filter == filter)
    }
    fn wait(&self, out: &mut [Kevent], deadline: Option<Instant>) -> usize {
        let mut e = self.events.lock();
        let mut _w = None;
        loop {
            let mut n = 0;
            for x in e.iter_mut() {
                if n == out.len() {
                    break;
                }
                if x.triggered {
                    out[n] = x.ev;
                    n += 1;
                    x.triggered = false;
                }
            }
            if n > 0 {
                return n;
            }
            if _w.is_none() {
                _w = Some(crate::thread::waiting(if deadline.is_some() { "equeue (timed)" } else { "equeue" }, self as *const _ as u64, Some(equeue_detail)));
            }
            match deadline {
                Some(d) => {
                    if self.cv.wait_until(&mut e, d).timed_out() {
                        // One last check after the timeout.
                        let mut n = 0;
                        for x in e.iter_mut() {
                            if n < out.len() && x.triggered {
                                out[n] = x.ev;
                                n += 1;
                                x.triggered = false;
                            }
                        }
                        return n;
                    }
                }
                None => self.cv.wait(&mut e),
            }
        }
    }
}

fn equeue_detail(p: u64) -> String {
    let q = unsafe { &*(p as *const Equeue) };
    match q.events.try_lock() {
        Some(e) => {
            let evs: Vec<String> = e.iter().take(8).map(|x| format!("({},{})", x.ev.ident, x.ev.filter)).collect();
            format!("'{}' {} event(s) {}", q.name, e.len(), evs.join(" "))
        }
        None => format!("'{}'", q.name),
    }
}

pub unsafe fn from_handle<'a>(h: u64) -> Option<&'a Equeue> {
    if h == 0 {
        return None;
    }
    let q = unsafe { &*(h as *const Equeue) };
    if q.magic == EQ_MAGIC { Some(q) } else { None }
}

sysv! {
    fn sceKernelCreateEqueue(out: *mut u64, name: *const u8) -> i32 {
        if out.is_null() { return sce(EINVAL); }
        let q: &'static Equeue = Box::leak(Box::new(Equeue { magic: EQ_MAGIC, name: cstr(name).to_string(), events: Mutex::new(Vec::new()), cv: Condvar::new() }));
        *out = q as *const _ as u64;
        crate::debug!("equeue '{}' created {:#x}", q.name, *out);
        0
    }
    fn sceKernelDeleteEqueue(h: u64) -> i32 { if from_handle(h).is_some() { 0 } else { sce(EBADF) } }
    fn sceKernelWaitEqueue(h: u64, ev: *mut Kevent, num: i32, out: *mut i32, timeout: *const u32) -> i32 {
        let Some(q) = from_handle(h) else { return sce(EBADF) };
        if ev.is_null() || num < 1 { return sce(EINVAL); }
        let deadline = if timeout.is_null() { None } else { Some(Instant::now() + Duration::from_micros(*timeout as u64)) };
        let buf = std::slice::from_raw_parts_mut(ev, num as usize);
        let n = q.wait(buf, deadline);
        if !out.is_null() { *out = n as i32; }
        if n == 0 { sce(ETIMEDOUT) } else { 0 }
    }
    fn sceKernelAddUserEvent(h: u64, id: i32) -> i32 {
        let Some(q) = from_handle(h) else { return sce(EBADF) };
        q.add(id as u64, EVFILT_USER, 0, 0);
        0
    }
    fn sceKernelAddUserEventEdge(h: u64, id: i32) -> i32 {
        let Some(q) = from_handle(h) else { return sce(EBADF) };
        q.add(id as u64, EVFILT_USER, EV_CLEAR, 0);
        0
    }
    fn sceKernelTriggerUserEvent(h: u64, id: i32, udata: u64) -> i32 {
        let Some(q) = from_handle(h) else { return sce(EBADF) };
        if q.trigger(id as u64, EVFILT_USER, None, Some(udata)) { 0 } else { sce(ENOENT) }
    }
    fn sceKernelDeleteUserEvent(h: u64, id: i32) -> i32 {
        let Some(q) = from_handle(h) else { return sce(EBADF) };
        if q.remove(id as u64, EVFILT_USER) { 0 } else { sce(ENOENT) }
    }
    fn sceKernelGetEventId(ev: *const Kevent) -> u64 { (*ev).ident }
    fn sceKernelGetEventFilter(ev: *const Kevent) -> i32 { (*ev).filter as i32 }
    fn sceKernelGetEventData(ev: *const Kevent) -> u64 { (*ev).data }
    fn sceKernelGetEventUserData(ev: *const Kevent) -> u64 { (*ev).udata }
    fn sceKernelGetEventFflags(ev: *const Kevent) -> u32 { (*ev).fflags }
    fn sceKernelGetEventError(ev: *const Kevent) -> i32 { 0 }
}

pub fn register(r: &mut Registry) {
    reg!(r, "libkernel", [
        sceKernelCreateEqueue, sceKernelDeleteEqueue, sceKernelWaitEqueue, sceKernelAddUserEvent,
        sceKernelAddUserEventEdge, sceKernelTriggerUserEvent, sceKernelDeleteUserEvent, sceKernelGetEventId,
        sceKernelGetEventFilter, sceKernelGetEventData, sceKernelGetEventUserData, sceKernelGetEventFflags,
        sceKernelGetEventError,
    ]);
}
