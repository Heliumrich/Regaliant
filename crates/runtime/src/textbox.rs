//! The text box the port draws in the game's window while the title's system text entry is open
//! (the character's name: `hle::libs::ime_dialog`). The console shows its own keyboard over the
//! game; here the keyboard types into a box drawn over the frame, in the game's window, so the
//! game keeps its window and its fullscreen (a separate Windows text box took the foreground and
//! threw exclusive fullscreen back to the desktop: the user, 2026-11-13). Drawn with GDI into a
//! bitmap the presenter puts on the frame (`gpu::vk::present`).

use windows_sys::Win32::Foundation::{RECT, SIZE};
use windows_sys::Win32::Graphics::Gdi::*;

/// What the box shows.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Content {
    pub title: String,
    pub text: String,
    /// Shown in the empty field (the title's placeholder).
    pub placeholder: String,
    pub max_len: u32,
    pub caret: bool,
    pub hint: String,
}

fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

const fn rgb(r: u8, g: u8, b: u8) -> u32 {
    r as u32 | (g as u32) << 8 | (b as u32) << 16
}

/// The box's size at a frame `frame_h` pixels high (designed at 1080 lines).
pub fn size(frame_h: u32) -> (u32, u32) {
    let s = (frame_h as f32 / 1080.0).max(0.5);
    ((820.0 * s).round() as u32, (206.0 * s).round() as u32)
}

/// The box as top-down BGRA pixels (alpha 255) for a frame `frame_h` pixels high.
pub fn render(c: &Content, frame_h: u32) -> (Vec<u8>, u32, u32) {
    let s = (frame_h as f32 / 1080.0).max(0.5);
    let px = |v: f32| (v * s).round() as i32;
    let (w, h) = size(frame_h);
    let mut out = vec![0u8; (w * h * 4) as usize];
    unsafe {
        let dc = CreateCompatibleDC(std::ptr::null_mut());
        if dc.is_null() {
            return (out, w, h);
        }
        let mut bmi: BITMAPINFO = std::mem::zeroed();
        bmi.bmiHeader.biSize = std::mem::size_of::<BITMAPINFOHEADER>() as u32;
        bmi.bmiHeader.biWidth = w as i32;
        bmi.bmiHeader.biHeight = -(h as i32);
        bmi.bmiHeader.biPlanes = 1;
        bmi.bmiHeader.biBitCount = 32;
        bmi.bmiHeader.biCompression = BI_RGB;
        let mut bits: *mut core::ffi::c_void = std::ptr::null_mut();
        let bmp = CreateDIBSection(dc, &bmi, DIB_RGB_COLORS, &mut bits, std::ptr::null_mut(), 0);
        if bmp.is_null() || bits.is_null() {
            DeleteDC(dc);
            return (out, w, h);
        }
        let old_bmp = SelectObject(dc, bmp as _);
        let fill = |r: RECT, color: u32| {
            let b = CreateSolidBrush(color);
            FillRect(dc, &r, b);
            DeleteObject(b as _);
        };
        let frame = |r: RECT, color: u32, t: i32| {
            for i in 0..t.max(1) {
                let b = CreateSolidBrush(color);
                let rr = RECT { left: r.left + i, top: r.top + i, right: r.right - i, bottom: r.bottom - i };
                FrameRect(dc, &rr, b);
                DeleteObject(b as _);
            }
        };
        let (wi, hi) = (w as i32, h as i32);
        // The panel: near black, a bronze frame and a darker line inside it (the game's menus).
        fill(RECT { left: 0, top: 0, right: wi, bottom: hi }, rgb(11, 11, 13));
        frame(RECT { left: 0, top: 0, right: wi, bottom: hi }, rgb(150, 134, 104), px(2.0));
        frame(RECT { left: px(5.0), top: px(5.0), right: wi - px(5.0), bottom: hi - px(5.0) }, rgb(58, 53, 44), 1);
        let font = |height: f32, face: &str| {
            let name = wide(face);
            CreateFontW(-px(height), 0, 0, 0, 400, 0, 0, 0, DEFAULT_CHARSET as _, OUT_TT_PRECIS as _, CLIP_DEFAULT_PRECIS as _, ANTIALIASED_QUALITY as _, (VARIABLE_PITCH as u32 | FF_ROMAN as u32) as _, name.as_ptr())
        };
        let title_font = font(30.0, "Georgia");
        let text_font = font(32.0, "Georgia");
        let hint_font = font(21.0, "Segoe UI");
        SetBkMode(dc, TRANSPARENT as _);
        let draw = |text: &str, f: HFONT, color: u32, mut r: RECT, flags: DRAW_TEXT_FORMAT| {
            let mut t: Vec<u16> = text.encode_utf16().collect();
            if t.is_empty() {
                return;
            }
            let old = SelectObject(dc, f as _);
            SetTextColor(dc, color);
            DrawTextW(dc, t.as_mut_ptr(), t.len() as i32, &mut r, flags);
            SelectObject(dc, old);
        };
        let single = DT_SINGLELINE | DT_VCENTER | DT_NOPREFIX;
        draw(&c.title, title_font, rgb(226, 216, 194), RECT { left: px(24.0), top: px(14.0), right: wi - px(24.0), bottom: px(60.0) }, single | DT_CENTER | DT_END_ELLIPSIS);
        // The field.
        let field = RECT { left: px(40.0), top: px(70.0), right: wi - px(40.0), bottom: px(128.0) };
        fill(field, rgb(4, 4, 5));
        frame(field, rgb(96, 88, 72), 1);
        let count = format!("{} / {}", c.text.encode_utf16().count(), c.max_len);
        let count_r = RECT { left: field.right - px(110.0), top: field.top, right: field.right - px(14.0), bottom: field.bottom };
        draw(&count, hint_font, rgb(118, 112, 100), count_r, single | DT_RIGHT);
        let text_r = RECT { left: field.left + px(16.0), top: field.top, right: count_r.left - px(8.0), bottom: field.bottom };
        if c.text.is_empty() {
            draw(&c.placeholder, text_font, rgb(110, 104, 94), text_r, single | DT_LEFT | DT_END_ELLIPSIS);
        } else {
            draw(&c.text, text_font, rgb(250, 248, 244), text_r, single | DT_LEFT | DT_END_ELLIPSIS);
        }
        if c.caret {
            // After the text (its width as drawn in the text font).
            let old = SelectObject(dc, text_font as _);
            let t: Vec<u16> = c.text.encode_utf16().collect();
            let mut sz = SIZE { cx: 0, cy: 0 };
            if !t.is_empty() {
                GetTextExtentPoint32W(dc, t.as_ptr(), t.len() as i32, &mut sz);
            }
            SelectObject(dc, old);
            let x = (text_r.left + sz.cx + px(2.0)).min(text_r.right);
            fill(RECT { left: x, top: field.top + px(12.0), right: x + px(2.0).max(1), bottom: field.bottom - px(12.0) }, rgb(236, 226, 200));
        }
        draw(&c.hint, hint_font, rgb(168, 160, 146), RECT { left: px(24.0), top: px(140.0), right: wi - px(24.0), bottom: hi - px(12.0) }, single | DT_CENTER | DT_END_ELLIPSIS);
        GdiFlush();
        let src = std::slice::from_raw_parts(bits as *const u8, (w * h * 4) as usize);
        out.copy_from_slice(src);
        for p in out.chunks_exact_mut(4) {
            p[3] = 255;
        }
        SelectObject(dc, old_bmp);
        DeleteObject(title_font as _);
        DeleteObject(text_font as _);
        DeleteObject(hint_font as _);
        DeleteObject(bmp as _);
        DeleteDC(dc);
    }
    (out, w, h)
}

/// HDR10 frames: the box's display values (sRGB) as PQ values of the same colours at
/// `nits` (BT.709 greys and near-greys: the primaries are not converted).
pub fn to_pq(bgra: &mut [u8], nits: f32) {
    let lut: Vec<u8> = (0..256)
        .map(|v| {
            let c = v as f32 / 255.0;
            let lin = if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) };
            let y = (lin * nits / 10000.0).clamp(0.0, 1.0);
            let (m1, m2, c1, c2, c3) = (0.1593017578125f32, 78.84375f32, 0.8359375f32, 18.8515625f32, 18.6875f32);
            let yp = y.powf(m1);
            let pq = ((c1 + c2 * yp) / (1.0 + c3 * yp)).powf(m2);
            (pq * 255.0).round() as u8
        })
        .collect();
    for p in bgra.chunks_exact_mut(4) {
        for c in &mut p[..3] {
            *c = lut[*c as usize];
        }
    }
}
