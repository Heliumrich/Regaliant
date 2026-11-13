//! System IME dialog (libSceImeDialog): text entry such as the character name. The console's
//! on-screen keyboard is replaced by a text box the port draws in the game's own window
//! (`crate::textbox`, put on each frame by the presenter): the window gives its keys to the box
//! while it is open (`char_typed`, `key_down`), Enter confirms, Escape cancels, Backspace erases,
//! Ctrl+V pastes; a controller's A confirms and B cancels (`pad_input`). The game sees no input
//! meanwhile, as on the console while its keyboard shows. (Until 2026-11-13 a separate Windows
//! text box took the foreground: the game left exclusive fullscreen for the desktop.) Automated
//! runs supply the text with `DS_IME_TEXT` or the `ime <text>` control command. The entered text
//! is written to the title's UTF-16 buffer when it next polls the dialog status.

use crate::hle::Registry;
use crate::{reg, sysv};
use parking_lot::Mutex;
use std::sync::atomic::{AtomicU32, Ordering::Relaxed};
use std::time::Instant;

const STATUS_NONE: i32 = 0;
const STATUS_RUNNING: i32 = 1;
const STATUS_FINISHED: i32 = 2;
const END_OK: i32 = 0;
const END_CANCELED: i32 = 1;
const END_ABORTED: i32 = 2;

struct Dialog {
    /// Guest char16 buffer receiving the text (SceImeDialogParam.inputTextBuffer).
    buf: u64,
    max_len: u32,
    status: i32,
    end: i32,
    text: Option<Vec<u16>>,
    delivered: bool,
    /// The text being typed, the title and placeholder the box shows, when it opened.
    edit: Vec<u16>,
    title: String,
    placeholder: String,
    opened: Instant,
}

static DIALOG: Mutex<Option<Dialog>> = Mutex::new(None);
/// Text supplied through the control channel (`ime <text>`).
pub static CONTROL_TEXT: Mutex<Option<String>> = Mutex::new(None);
/// Edits since start (the box is drawn again when they or the caret change).
static EDITS: AtomicU32 = AtomicU32::new(0);

unsafe fn read_u16z(p: u64, max: usize) -> Vec<u16> {
    let mut v = Vec::new();
    if p == 0 {
        return v;
    }
    for i in 0..max {
        let c = unsafe { *((p + 2 * i as u64) as *const u16) };
        if c == 0 {
            break;
        }
        v.push(c);
    }
    v
}

fn finish(d: &mut Dialog, text: Option<Vec<u16>>, end: i32) {
    if let Some(mut t) = text {
        t.truncate(d.max_len as usize);
        d.text = Some(t);
    }
    d.end = end;
    d.status = STATUS_FINISHED;
    EDITS.fetch_add(1, Relaxed);
}

/// The box is open (the window's keys and the pad are its).
pub fn active() -> bool {
    DIALOG.lock().as_ref().is_some_and(|d| d.status == STATUS_RUNNING)
}

/// A character typed in the game's window (WM_CHAR): true when the box took it.
pub fn char_typed(c: u16) -> bool {
    let mut g = DIALOG.lock();
    let Some(d) = g.as_mut().filter(|d| d.status == STATUS_RUNNING) else { return false };
    if c >= 0x20 && c != 0x7F && (d.edit.len() as u32) < d.max_len.max(1) {
        d.edit.push(c);
        EDITS.fetch_add(1, Relaxed);
    }
    true
}

/// A key pressed in the game's window (WM_KEYDOWN; `repeat`: held down): true when the box took
/// it (every key while it is open: the game sees none).
pub fn key_down(vk: u8, repeat: bool, ctrl: bool) -> bool {
    let mut g = DIALOG.lock();
    let Some(d) = g.as_mut().filter(|d| d.status == STATUS_RUNNING) else { return false };
    match vk {
        // Backspace (Ctrl: the last word).
        0x08 => {
            if ctrl {
                while d.edit.last() == Some(&(b' ' as u16)) {
                    d.edit.pop();
                }
                while d.edit.last().is_some_and(|&c| c != b' ' as u16) {
                    d.edit.pop();
                }
            } else {
                d.edit.pop();
            }
            EDITS.fetch_add(1, Relaxed);
        }
        0x0D if !repeat => {
            let t = d.edit.clone();
            crate::info!("ime dialog: confirmed in the game's window");
            finish(d, Some(t), END_OK);
        }
        0x1B if !repeat => {
            crate::info!("ime dialog: canceled in the game's window");
            finish(d, None, END_CANCELED);
        }
        b'V' if ctrl => {
            if let Some(t) = clipboard_text() {
                for c in t.encode_utf16().take_while(|&c| c != b'\r' as u16 && c != b'\n' as u16) {
                    if c >= 0x20 && c != 0x7F && (d.edit.len() as u32) < d.max_len.max(1) {
                        d.edit.push(c);
                    }
                }
                EDITS.fetch_add(1, Relaxed);
            }
        }
        _ => {}
    }
    true
}

/// The pad's buttons (bit i: `input::PAD_BUTTONS[i]`, A 0, B 1) while the box is open: A
/// confirms, B cancels (each on its press; the button is then kept from the game until it is
/// released). True while the box is open (the pad is its alone).
pub fn pad_input(phys: u32) -> bool {
    static LAST: AtomicU32 = AtomicU32::new(0);
    let was = LAST.swap(phys, Relaxed);
    let mut g = DIALOG.lock();
    let Some(d) = g.as_mut().filter(|d| d.status == STATUS_RUNNING) else { return false };
    let pressed = phys & !was;
    if pressed & 1 != 0 {
        let t = d.edit.clone();
        crate::info!("ime dialog: confirmed with the controller");
        finish(d, Some(t), END_OK);
        crate::input::swallow_pad(1);
    } else if pressed & 2 != 0 {
        crate::info!("ime dialog: canceled with the controller");
        finish(d, None, END_CANCELED);
        crate::input::swallow_pad(2);
    }
    true
}

/// What the box shows now and a generation that changes with it (None: no box).
pub fn overlay() -> Option<(u64, crate::textbox::Content)> {
    let g = DIALOG.lock();
    let d = g.as_ref().filter(|d| d.status == STATUS_RUNNING)?;
    // The caret blinks every 530 ms (Windows' default).
    let phase = (d.opened.elapsed().as_millis() / 530) as u64;
    let pad = crate::input::last_device() == crate::input::DEVICE_PAD;
    let content = crate::textbox::Content {
        title: if d.title.is_empty() { "Enter Text".into() } else { d.title.clone() },
        text: String::from_utf16_lossy(&d.edit),
        placeholder: d.placeholder.clone(),
        max_len: d.max_len,
        caret: phase % 2 == 0,
        hint: if pad { "Type with the keyboard.   (A) Confirm   (B) Cancel".into() } else { "Enter  Confirm      Esc  Cancel".into() },
    };
    let generation = (EDITS.load(Relaxed) as u64) << 33 | (phase & 1) << 32 | pad as u64;
    Some((generation, content))
}

fn clipboard_text() -> Option<String> {
    use windows_sys::Win32::System::DataExchange::{CloseClipboard, GetClipboardData, OpenClipboard};
    use windows_sys::Win32::System::Memory::{GlobalLock, GlobalUnlock};
    const CF_UNICODETEXT: u32 = 13;
    unsafe {
        if OpenClipboard(std::ptr::null_mut()) == 0 {
            return None;
        }
        let h = GetClipboardData(CF_UNICODETEXT);
        let mut out = None;
        if !h.is_null() {
            let p = GlobalLock(h as _) as *const u16;
            if !p.is_null() {
                let mut n = 0;
                while n < 4096 && *p.add(n) != 0 {
                    n += 1;
                }
                out = Some(String::from_utf16_lossy(std::slice::from_raw_parts(p, n)));
                GlobalUnlock(h as _);
            }
        }
        CloseClipboard();
        out
    }
}

sysv! {
    fn sceImeDialogInit(param: *const u8, extended: *const u8) -> i32 {
        if param.is_null() {
            return crate::hle::kernel::sce(22); // EINVAL
        }
        let max_len = *(param.add(0x24) as *const u32);
        let buf = *(param.add(0x28) as *const u64);
        let placeholder = *(param.add(0x40) as *const u64);
        let title = *(param.add(0x48) as *const u64);
        let s = |v: Vec<u16>| String::from_utf16_lossy(&v);
        let title_s = s(read_u16z(title, 256));
        let placeholder_s = s(read_u16z(placeholder, 256));
        let initial = read_u16z(buf, max_len.min(1024) as usize);
        crate::info!("ime dialog: title '{}', placeholder '{}', initial '{}', max {} chars", title_s, placeholder_s, s(initial.clone()), max_len);
        let mut d = Dialog {
            buf,
            max_len,
            status: STATUS_RUNNING,
            end: END_CANCELED,
            text: None,
            delivered: false,
            edit: initial,
            title: title_s,
            placeholder: placeholder_s,
            opened: Instant::now(),
        };
        if let Ok(t) = std::env::var("DS_IME_TEXT") {
            crate::info!("ime dialog: text supplied by DS_IME_TEXT");
            finish(&mut d, Some(t.encode_utf16().collect()), END_OK);
        } else {
            crate::info!("ime dialog: a text box in the game's window (Enter confirms, Esc cancels)");
            EDITS.fetch_add(1, Relaxed);
        }
        *DIALOG.lock() = Some(d);
        0
    }
    fn sceImeDialogGetStatus() -> i32 {
        let mut g = DIALOG.lock();
        let Some(d) = g.as_mut() else { return STATUS_NONE };
        if d.status == STATUS_RUNNING {
            if let Some(t) = CONTROL_TEXT.lock().take() {
                crate::info!("ime dialog: text supplied by control command");
                finish(d, Some(t.encode_utf16().collect()), END_OK);
            }
        }
        if d.status == STATUS_FINISHED && !d.delivered {
            d.delivered = true;
            if let (Some(t), true) = (&d.text, d.end == END_OK && d.buf != 0) {
                for (i, c) in t.iter().chain(std::iter::once(&0)).enumerate() {
                    *((d.buf + 2 * i as u64) as *mut u16) = *c;
                }
                crate::info!("ime dialog: entered '{}'", String::from_utf16_lossy(t));
            }
        }
        d.status
    }
    fn sceImeDialogGetResult(result: *mut i32) -> i32 {
        let g = DIALOG.lock();
        let end = g.as_ref().map(|d| d.end).unwrap_or(END_CANCELED);
        if !result.is_null() {
            std::ptr::write_bytes(result as *mut u8, 0, 16);
            *result = end;
        }
        0
    }
    fn sceImeDialogTerm() -> i32 {
        DIALOG.lock().take();
        EDITS.fetch_add(1, Relaxed);
        0
    }
    fn sceImeDialogAbort() -> i32 {
        if let Some(d) = DIALOG.lock().as_mut() {
            finish(d, None, END_ABORTED);
        }
        0
    }
}

pub fn register(r: &mut Registry) {
    reg!(r, "libSceImeDialog", [sceImeDialogInit, sceImeDialogGetStatus, sceImeDialogGetResult, sceImeDialogTerm, sceImeDialogAbort]);
}
