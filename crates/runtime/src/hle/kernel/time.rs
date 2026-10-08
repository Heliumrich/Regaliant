//! Time sources: process time, TSC, clocks and sleeping.

use super::errno::*;
use super::sce;
use crate::hle::Registry;
use crate::{reg, sysv};
use std::sync::OnceLock;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

struct Clock {
    start: Instant,
    tsc_hz: u64,
}

static CLOCK: OnceLock<Clock> = OnceLock::new();

/// Measures the TSC frequency against the monotonic clock (guest code may read
/// the TSC directly with rdtsc, so the reported frequency must be the host's).
pub fn init() {
    CLOCK.get_or_init(|| {
        let t0 = Instant::now();
        let c0 = unsafe { core::arch::x86_64::_rdtsc() };
        std::thread::sleep(Duration::from_millis(100));
        let c1 = unsafe { core::arch::x86_64::_rdtsc() };
        let dt = t0.elapsed().as_secs_f64();
        let hz = ((c1 - c0) as f64 / dt) as u64;
        crate::info!("TSC frequency calibrated at {:.3} MHz", hz as f64 / 1e6);
        Clock { start: Instant::now(), tsc_hz: hz }
    });
}

fn clock() -> &'static Clock {
    CLOCK.get().expect("time::init not called")
}

pub fn process_time_us() -> u64 {
    clock().start.elapsed().as_micros() as u64
}

pub fn tsc_hz() -> u64 {
    clock().tsc_hz
}

fn realtime() -> Duration {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default()
}

const CLOCK_REALTIME: i32 = 0;
const CLOCK_VIRTUAL: i32 = 1;
const CLOCK_PROF: i32 = 2;
const CLOCK_MONOTONIC: i32 = 4;
const CLOCK_UPTIME: i32 = 5;
const CLOCK_UPTIME_PRECISE: i32 = 7;
const CLOCK_UPTIME_FAST: i32 = 8;
const CLOCK_REALTIME_PRECISE: i32 = 9;
const CLOCK_REALTIME_FAST: i32 = 10;
const CLOCK_MONOTONIC_PRECISE: i32 = 11;
const CLOCK_MONOTONIC_FAST: i32 = 12;
const CLOCK_SECOND: i32 = 13;
const CLOCK_THREAD_CPUTIME_ID: i32 = 14;
const CLOCK_PROCESS_CPUTIME_ID: i32 = 15;

fn clock_value(id: i32) -> Option<Duration> {
    Some(match id {
        CLOCK_REALTIME | CLOCK_REALTIME_PRECISE | CLOCK_REALTIME_FAST => realtime(),
        CLOCK_SECOND => Duration::from_secs(realtime().as_secs()),
        CLOCK_MONOTONIC | CLOCK_MONOTONIC_PRECISE | CLOCK_MONOTONIC_FAST | CLOCK_UPTIME | CLOCK_UPTIME_PRECISE
        | CLOCK_UPTIME_FAST => clock().start.elapsed() + Duration::from_secs(1),
        CLOCK_VIRTUAL | CLOCK_PROF | CLOCK_THREAD_CPUTIME_ID | CLOCK_PROCESS_CPUTIME_ID => clock().start.elapsed(),
        _ => return None,
    })
}

unsafe fn write_ts(ts: *mut i64, d: Duration) {
    unsafe {
        *ts = d.as_secs() as i64;
        *ts.add(1) = d.subsec_nanos() as i64;
    }
}

pub fn sleep_us(us: u64) {
    if us == 0 {
        std::thread::yield_now();
    } else {
        std::thread::sleep(Duration::from_micros(us));
    }
}

/// time_t <-> SYSTEMTIME through FILETIME (100 ns since 1601).
fn to_systemtime(t: i64) -> Option<windows_sys::Win32::Foundation::SYSTEMTIME> {
    use windows_sys::Win32::Foundation::{FILETIME, SYSTEMTIME};
    let ft100 = (t.checked_add(11_644_473_600)?).checked_mul(10_000_000)?;
    let ft = FILETIME { dwLowDateTime: ft100 as u32, dwHighDateTime: (ft100 >> 32) as u32 };
    let mut st: SYSTEMTIME = unsafe { std::mem::zeroed() };
    (unsafe { windows_sys::Win32::System::Time::FileTimeToSystemTime(&ft, &mut st) } != 0).then_some(st)
}

fn from_systemtime(st: &windows_sys::Win32::Foundation::SYSTEMTIME) -> Option<i64> {
    use windows_sys::Win32::Foundation::FILETIME;
    let mut ft = FILETIME { dwLowDateTime: 0, dwHighDateTime: 0 };
    if unsafe { windows_sys::Win32::System::Time::SystemTimeToFileTime(st, &mut ft) } == 0 {
        return None;
    }
    let v = ((ft.dwHighDateTime as i64) << 32) | ft.dwLowDateTime as i64;
    Some(v / 10_000_000 - 11_644_473_600)
}

/// Host time zone: (standard offset west of UTC in seconds, DST seconds at UTC time `t`).
fn zone_at_utc(t: i64) -> Option<(i32, i32)> {
    use windows_sys::Win32::System::Time::{GetTimeZoneInformation, SystemTimeToTzSpecificLocalTime, TIME_ZONE_INFORMATION};
    let mut tzi: TIME_ZONE_INFORMATION = unsafe { std::mem::zeroed() };
    unsafe { GetTimeZoneInformation(&mut tzi) };
    let utc = to_systemtime(t)?;
    let mut local = unsafe { std::mem::zeroed() };
    if unsafe { SystemTimeToTzSpecificLocalTime(std::ptr::null(), &utc, &mut local) } == 0 {
        return None;
    }
    let offset = from_systemtime(&local)? - t; // local - utc
    let west = tzi.Bias * 60; // UTC = local + bias (standard time)
    Some((west, (offset + west as i64) as i32))
}

sysv! {
    /// SceKernelTimesec {time_t t; u32 westsec; u32 dstsec} (layout confirmed by libc's use).
    fn sceKernelConvertUtcToLocaltime(t: i64, local: *mut i64, ts: *mut u8, dst: *mut i32) -> i32 {
        let Some((west, dstsec)) = zone_at_utc(t) else { return sce(EINVAL) };
        let l = t - west as i64 + dstsec as i64;
        if !local.is_null() { *local = l; }
        if !ts.is_null() {
            *(ts as *mut i64) = t;
            *(ts.add(8) as *mut u32) = west as u32;
            *(ts.add(12) as *mut u32) = dstsec as u32;
        }
        if !dst.is_null() { *dst = dstsec; }
        0
    }
    fn sceKernelConvertLocaltimeToUtc(local: i64, isdst: i64, utc: *mut i64, ts: *mut u8, dst: *mut i32) -> i32 {
        // Find the UTC instant whose local time is `local` (DST at that instant).
        let Some((west, _)) = zone_at_utc(local) else { return sce(EINVAL) };
        let mut u = local + west as i64;
        if let Some((_, d)) = zone_at_utc(u) {
            u -= d as i64;
        }
        let (west, dstsec) = zone_at_utc(u).unwrap_or((west, 0));
        if !utc.is_null() { *utc = u; }
        if !ts.is_null() {
            *(ts as *mut i64) = u;
            *(ts.add(8) as *mut u32) = west as u32;
            *(ts.add(12) as *mut u32) = dstsec as u32;
        }
        if !dst.is_null() { *dst = dstsec; }
        0
    }
    fn sceKernelGetProcessTime() -> u64 { process_time_us() }
    fn sceKernelGetProcessTimeCounter() -> u64 { unsafe { core::arch::x86_64::_rdtsc() } }
    fn sceKernelGetProcessTimeCounterFrequency() -> u64 { tsc_hz() }
    fn sceKernelReadTsc() -> u64 { unsafe { core::arch::x86_64::_rdtsc() } }
    fn sceKernelGetTscFrequency() -> u64 { tsc_hz() }
    fn sceKernelUsleep(us: u32) -> i32 { sleep_us(us as u64); 0 }
    fn sceKernelSleep(s: u32) -> i32 { std::thread::sleep(Duration::from_secs(s as u64)); 0 }
    fn sceKernelNanosleep(req: *const i64, rem: *mut i64) -> i32 {
        if req.is_null() { return sce(EINVAL); }
        std::thread::sleep(Duration::new((*req).max(0) as u64, (*req.add(1)).clamp(0, 999_999_999) as u32));
        if !rem.is_null() { *rem = 0; *rem.add(1) = 0; }
        0
    }
    fn sceKernelClockGettime(id: i32, ts: *mut i64) -> i32 {
        if ts.is_null() { return sce(EFAULT); }
        match clock_value(id) { Some(d) => { write_ts(ts, d); 0 } None => sce(EINVAL) }
    }
    fn sceKernelClockGetres(id: i32, ts: *mut i64) -> i32 {
        if ts.is_null() { return sce(EFAULT); }
        write_ts(ts, Duration::from_nanos(1));
        0
    }
    fn sceKernelGettimeofday(tv: *mut i64) -> i32 {
        if tv.is_null() { return sce(EFAULT); }
        let d = realtime();
        *tv = d.as_secs() as i64;
        *tv.add(1) = d.subsec_micros() as i64;
        0
    }
    fn sceKernelGettimezone(tz: *mut i32) -> i32 {
        if !tz.is_null() { *tz = 0; *tz.add(1) = 0; }
        0
    }
    fn clock_gettime(id: i32, ts: *mut i64) -> i32 {
        match clock_value(id) { Some(d) if !ts.is_null() => { write_ts(ts, d); 0 } _ => { super::set_errno(EINVAL); -1 } }
    }
    fn clock_getres(id: i32, ts: *mut i64) -> i32 {
        if !ts.is_null() { write_ts(ts, Duration::from_nanos(1)); }
        0
    }
    fn gettimeofday(tv: *mut i64, tz: *mut i32) -> i32 {
        if !tv.is_null() {
            let d = realtime();
            *tv = d.as_secs() as i64;
            *tv.add(1) = d.subsec_micros() as i64;
        }
        if !tz.is_null() { *tz = 0; *tz.add(1) = 0; }
        0
    }
    fn nanosleep(req: *const i64, rem: *mut i64) -> i32 {
        if req.is_null() { super::set_errno(EINVAL); return -1; }
        std::thread::sleep(Duration::new((*req).max(0) as u64, (*req.add(1)).clamp(0, 999_999_999) as u32));
        if !rem.is_null() { *rem = 0; *rem.add(1) = 0; }
        0
    }
    fn usleep(us: u32) -> i32 { sleep_us(us as u64); 0 }
    fn sleep(s: u32) -> u32 { std::thread::sleep(Duration::from_secs(s as u64)); 0 }
}

pub fn register(r: &mut Registry) {
    reg!(r, "libkernel", [
        sceKernelConvertUtcToLocaltime, sceKernelConvertLocaltimeToUtc,
        sceKernelGetProcessTime, sceKernelGetProcessTimeCounter, sceKernelGetProcessTimeCounterFrequency,
        sceKernelReadTsc, sceKernelGetTscFrequency, sceKernelUsleep, sceKernelSleep, sceKernelNanosleep,
        sceKernelClockGettime, sceKernelClockGetres, sceKernelGettimeofday, sceKernelGettimezone,
    ]);
    reg!(r, "libScePosix", [clock_gettime, clock_getres, gettimeofday, nanosleep, usleep, sleep]);
}
