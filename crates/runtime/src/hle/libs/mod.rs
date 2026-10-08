//! HLE for PS5 system libraries other than libkernel.

pub mod ajm;
pub mod app;
pub mod devices;
pub mod ime_dialog;
pub mod online;
pub mod rtc;
pub mod savedata;
pub mod sysmodule;
pub mod videoout;

use crate::hle::Registry;

pub fn register(r: &mut Registry) {
    sysmodule::register(r);
    app::register(r);
    online::register(r);
    ime_dialog::register(r);
    devices::register(r);
    ajm::register(r);
    rtc::register(r);
    savedata::register(r);
    videoout::register(r);
}
