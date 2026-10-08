//! HLE for PS5 system libraries other than libkernel.

pub mod ajm;
pub mod ampr;
pub mod app;
pub mod devices;
pub mod fiber;
pub mod ime_dialog;
pub mod libc_internal;
pub mod online;
pub mod ps5svc;
pub mod rtc;
pub mod savedata;
pub mod sysmodule;
pub mod videoout;

use crate::hle::Registry;

pub fn register(r: &mut Registry) {
    sysmodule::register(r);
    libc_internal::register(r);
    ampr::register(r);
    fiber::register(r);
    ps5svc::register(r);
    app::register(r);
    online::register(r);
    ime_dialog::register(r);
    devices::register(r);
    ajm::register(r);
    rtc::register(r);
    savedata::register(r);
    videoout::register(r);
}
