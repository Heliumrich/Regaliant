//! Small Win32 helpers: controls laid out in DPI-independent units, text, check boxes, list
//! views, a progress bar, folder pickers, message boxes.

use std::path::PathBuf;
use windows_sys::Win32::Foundation::{HWND, LPARAM, WPARAM};
use windows_sys::Win32::Graphics::Gdi::{CreateFontIndirectW, HFONT, LOGFONTW};
use windows_sys::Win32::UI::Controls::*;
use windows_sys::Win32::UI::HiDpi::SystemParametersInfoForDpi;
use windows_sys::Win32::UI::WindowsAndMessaging::*;

pub fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

pub fn from_wide(b: &[u16]) -> String {
    let n = b.iter().position(|&c| c == 0).unwrap_or(b.len());
    String::from_utf16_lossy(&b[..n])
}

/// The layout scale: device pixels per 1/96 inch.
#[derive(Clone, Copy)]
pub struct Scale(pub u32);

impl Scale {
    pub fn px(self, v: i32) -> i32 {
        (v as i64 * self.0 as i64 / 96) as i32
    }
}

/// The message font at a DPI (normal, and bold when `bold`).
pub unsafe fn message_font(dpi: u32, bold: bool, size_pct: i32) -> HFONT {
    let mut ncm: NONCLIENTMETRICSW = unsafe { std::mem::zeroed() };
    ncm.cbSize = std::mem::size_of::<NONCLIENTMETRICSW>() as u32;
    unsafe { SystemParametersInfoForDpi(SPI_GETNONCLIENTMETRICS, ncm.cbSize, &mut ncm as *mut _ as *mut _, 0, dpi) };
    let mut lf: LOGFONTW = ncm.lfMessageFont;
    lf.lfHeight = lf.lfHeight * size_pct / 100;
    if bold {
        lf.lfWeight = 600;
    }
    unsafe { CreateFontIndirectW(&lf) }
}

pub struct Builder {
    pub parent: HWND,
    pub scale: Scale,
    pub font: windows_sys::Win32::Graphics::Gdi::HFONT,
}

impl Builder {
    /// A child control at (x, y, w, h) in 1/96-inch units.
    pub unsafe fn add(&self, class: &str, text: &str, style: u32, ex: u32, r: (i32, i32, i32, i32), id: i32) -> HWND {
        let s = self.scale;
        let c = wide(class);
        let t = wide(text);
        let h = unsafe {
            CreateWindowExW(ex, c.as_ptr(), t.as_ptr(), WS_CHILD | WS_VISIBLE | style, s.px(r.0), s.px(r.1), s.px(r.2), s.px(r.3), self.parent, id as isize as _, std::ptr::null_mut(), std::ptr::null())
        };
        unsafe { SendMessageW(h, WM_SETFONT, self.font as WPARAM, 1) };
        h
    }

    pub unsafe fn label(&self, text: &str, r: (i32, i32, i32, i32), id: i32) -> HWND {
        unsafe { self.add("STATIC", text, 0x80 /* SS_NOPREFIX */, 0, r, id) }
    }

    pub unsafe fn button(&self, text: &str, r: (i32, i32, i32, i32), id: i32) -> HWND {
        unsafe { self.add("BUTTON", text, WS_TABSTOP | BS_PUSHBUTTON as u32, 0, r, id) }
    }

    pub unsafe fn check(&self, text: &str, r: (i32, i32, i32, i32), id: i32) -> HWND {
        unsafe { self.add("BUTTON", text, WS_TABSTOP | BS_AUTOCHECKBOX as u32 | BS_MULTILINE as u32, 0, r, id) }
    }

    pub unsafe fn group(&self, text: &str, r: (i32, i32, i32, i32)) -> HWND {
        unsafe { self.add("BUTTON", text, BS_GROUPBOX as u32, 0, r, -1) }
    }

    pub unsafe fn edit(&self, text: &str, r: (i32, i32, i32, i32), id: i32) -> HWND {
        unsafe { self.add("EDIT", text, WS_TABSTOP | ES_AUTOHSCROLL as u32, WS_EX_CLIENTEDGE, r, id) }
    }

    /// A read-only text field (selectable, copyable).
    pub unsafe fn field(&self, text: &str, r: (i32, i32, i32, i32), id: i32) -> HWND {
        unsafe { self.add("EDIT", text, WS_TABSTOP | ES_AUTOHSCROLL as u32 | ES_READONLY as u32, WS_EX_CLIENTEDGE, r, id) }
    }

    pub unsafe fn progress(&self, r: (i32, i32, i32, i32), id: i32) -> HWND {
        let h = unsafe { self.add("msctls_progress32", "", 0, 0, r, id) };
        unsafe { SendMessageW(h, PBM_SETRANGE32, 0, 1000) };
        h
    }
}

pub unsafe fn set_text(h: HWND, s: &str) {
    let w = wide(s);
    unsafe { SetWindowTextW(h, w.as_ptr()) };
}

pub unsafe fn get_text(h: HWND) -> String {
    let n = unsafe { GetWindowTextLengthW(h) }.max(0) as usize;
    let mut b = vec![0u16; n + 1];
    unsafe { GetWindowTextW(h, b.as_mut_ptr(), b.len() as i32) };
    from_wide(&b)
}

pub unsafe fn set_check(h: HWND, on: bool) {
    unsafe { SendMessageW(h, BM_SETCHECK, on as WPARAM, 0) };
}

pub unsafe fn checked(h: HWND) -> bool {
    unsafe { SendMessageW(h, BM_GETCHECK, 0, 0) == 1 }
}

pub unsafe fn enable(h: HWND, on: bool) {
    unsafe { windows_sys::Win32::UI::Input::KeyboardAndMouse::EnableWindow(h, on as i32) };
}

pub unsafe fn show(h: HWND, on: bool) {
    unsafe { ShowWindow(h, if on { SW_SHOW } else { SW_HIDE }) };
}

/// Repaints a window and all its controls now. The main window does not clip its children, so
/// an erase of part of it can leave controls there unpainted (the Game page's lower half was
/// blank in a first start, 2026-11-11).
pub unsafe fn redraw(h: HWND) {
    use windows_sys::Win32::Graphics::Gdi::{RedrawWindow, RDW_ALLCHILDREN, RDW_ERASE, RDW_INVALIDATE, RDW_UPDATENOW};
    unsafe { RedrawWindow(h, std::ptr::null(), std::ptr::null_mut(), RDW_INVALIDATE | RDW_ERASE | RDW_ALLCHILDREN | RDW_UPDATENOW) };
}

/// The progress bar at `permille` (0-1000).
pub unsafe fn set_progress(h: HWND, permille: u32) {
    unsafe { SendMessageW(h, PBM_SETPOS, permille.min(1000) as WPARAM, 0) };
}

pub unsafe fn message(parent: HWND, text: &str, title: &str, flags: MESSAGEBOX_STYLE) -> i32 {
    let (t, m) = (wide(title), wide(text));
    unsafe { MessageBoxW(parent, m.as_ptr(), t.as_ptr(), flags) }
}

/// A report-style list view with columns (title, width).
pub unsafe fn list_view(b: &Builder, r: (i32, i32, i32, i32), id: i32, cols: &[(&str, i32)]) -> HWND {
    let h = unsafe { b.add("SysListView32", "", WS_TABSTOP | LVS_REPORT | LVS_SINGLESEL | LVS_SHOWSELALWAYS, WS_EX_CLIENTEDGE, r, id) };
    unsafe {
        SendMessageW(h, LVM_SETEXTENDEDLISTVIEWSTYLE, 0, (LVS_EX_FULLROWSELECT | LVS_EX_DOUBLEBUFFER) as LPARAM);
        for (i, (t, w)) in cols.iter().enumerate() {
            let mut tw = wide(t);
            let col = LVCOLUMNW { mask: LVCF_TEXT | LVCF_WIDTH | LVCF_SUBITEM, cx: b.scale.px(*w), pszText: tw.as_mut_ptr(), iSubItem: i as i32, ..Default::default() };
            SendMessageW(h, LVM_INSERTCOLUMNW, i, &col as *const _ as LPARAM);
        }
    }
    h
}

pub unsafe fn lv_clear(h: HWND) {
    unsafe { SendMessageW(h, LVM_DELETEALLITEMS, 0, 0) };
}

pub unsafe fn lv_add(h: HWND, cells: &[String]) {
    let n = unsafe { SendMessageW(h, LVM_GETITEMCOUNT, 0, 0) } as i32;
    for (c, text) in cells.iter().enumerate() {
        let mut w = wide(text);
        let mut it: LVITEMW = unsafe { std::mem::zeroed() };
        it.mask = LVIF_TEXT;
        it.iItem = n;
        it.iSubItem = c as i32;
        it.pszText = w.as_mut_ptr();
        unsafe {
            if c == 0 {
                SendMessageW(h, LVM_INSERTITEMW, 0, &it as *const _ as LPARAM);
            } else {
                SendMessageW(h, LVM_SETITEMTEXTW, n as WPARAM, &it as *const _ as LPARAM);
            }
        }
    }
}

pub unsafe fn lv_selected(h: HWND) -> Option<usize> {
    let r = unsafe { SendMessageW(h, LVM_GETNEXTITEM, usize::MAX, LVNI_SELECTED as LPARAM) };
    (r >= 0).then_some(r as usize)
}

/// A folder picker (the shell's file dialog in folder mode).
pub fn pick_folder(parent: HWND, title: &str) -> Option<PathBuf> {
    use windows::Win32::System::Com::{CLSCTX_INPROC_SERVER, CoCreateInstance, CoTaskMemFree};
    use windows::Win32::UI::Shell::{FOS_FORCEFILESYSTEM, FOS_PICKFOLDERS, FileOpenDialog, IFileOpenDialog, SIGDN_FILESYSPATH};
    unsafe {
        let d: IFileOpenDialog = CoCreateInstance(&FileOpenDialog, None, CLSCTX_INPROC_SERVER).ok()?;
        let opts = d.GetOptions().ok()?;
        d.SetOptions(opts | FOS_PICKFOLDERS | FOS_FORCEFILESYSTEM).ok()?;
        let t = wide(title);
        let _ = d.SetTitle(windows::core::PCWSTR(t.as_ptr()));
        d.Show(Some(windows::Win32::Foundation::HWND(parent as _))).ok()?;
        let item = d.GetResult().ok()?;
        let p = item.GetDisplayName(SIGDN_FILESYSPATH).ok()?;
        let s = p.to_string().ok();
        CoTaskMemFree(Some(p.0 as *const _));
        s.map(PathBuf::from)
    }
}

/// A file picker (the shell's file dialog) for `kind` files matching `spec` (e.g. "*.zar"); with
/// `save`, a save dialog starting in that folder with that file name.
pub fn pick_file(parent: HWND, title: &str, kind: &str, spec: &str, save: Option<(&std::path::Path, &str)>) -> Option<PathBuf> {
    use windows::Win32::System::Com::{CLSCTX_INPROC_SERVER, CoCreateInstance, CoTaskMemFree};
    use windows::Win32::UI::Shell::Common::COMDLG_FILTERSPEC;
    use windows::Win32::UI::Shell::{FOS_FILEMUSTEXIST, FOS_FORCEFILESYSTEM, FOS_OVERWRITEPROMPT, FileOpenDialog, FileSaveDialog, IFileDialog, IFileOpenDialog, IFileSaveDialog, IShellItem, SHCreateItemFromParsingName, SIGDN_FILESYSPATH};
    use windows::core::{Interface, PCWSTR};
    unsafe {
        let d: IFileDialog = match save {
            Some(_) => CoCreateInstance::<_, IFileSaveDialog>(&FileSaveDialog, None, CLSCTX_INPROC_SERVER).ok()?.cast().ok()?,
            None => CoCreateInstance::<_, IFileOpenDialog>(&FileOpenDialog, None, CLSCTX_INPROC_SERVER).ok()?.cast().ok()?,
        };
        let opts = d.GetOptions().ok()?;
        d.SetOptions(opts | FOS_FORCEFILESYSTEM | if save.is_some() { FOS_OVERWRITEPROMPT } else { FOS_FILEMUSTEXIST }).ok()?;
        let (t, k, s, ext) = (wide(title), wide(kind), wide(spec), wide(spec.trim_start_matches("*.")));
        let _ = d.SetTitle(PCWSTR(t.as_ptr()));
        let _ = d.SetFileTypes(&[COMDLG_FILTERSPEC { pszName: PCWSTR(k.as_ptr()), pszSpec: PCWSTR(s.as_ptr()) }]);
        let _ = d.SetDefaultExtension(PCWSTR(ext.as_ptr()));
        if let Some((folder, name)) = save {
            let (f, n) = (wide(&folder.display().to_string()), wide(name));
            if let Ok(item) = SHCreateItemFromParsingName::<_, _, IShellItem>(PCWSTR(f.as_ptr()), None) {
                let _ = d.SetFolder(&item);
            }
            let _ = d.SetFileName(PCWSTR(n.as_ptr()));
        }
        d.Show(Some(windows::Win32::Foundation::HWND(parent as _))).ok()?;
        let item = d.GetResult().ok()?;
        let p = item.GetDisplayName(SIGDN_FILESYSPATH).ok()?;
        let s = p.to_string().ok();
        CoTaskMemFree(Some(p.0 as *const _));
        s.map(PathBuf::from)
    }
}

/// Opens a folder (made first when missing) with Explorer.
pub fn open_path(p: &std::path::Path) {
    let _ = std::fs::create_dir_all(p);
    let (op, f) = (wide("open"), wide(&p.display().to_string()));
    unsafe { windows_sys::Win32::UI::Shell::ShellExecuteW(std::ptr::null_mut(), op.as_ptr(), f.as_ptr(), std::ptr::null(), std::ptr::null(), SW_SHOWNORMAL) };
}

/// Opens Explorer at the folder holding `item`, with `item` selected.
pub fn select_path(item: &std::path::Path) {
    let (op, exe, args) = (wide("open"), wide("explorer.exe"), wide(&format!("/select,\"{}\"", item.display())));
    unsafe { windows_sys::Win32::UI::Shell::ShellExecuteW(std::ptr::null_mut(), op.as_ptr(), exe.as_ptr(), args.as_ptr(), std::ptr::null(), SW_SHOWNORMAL) };
}

/// Opens a web address in the default browser.
pub fn open_url(u: &str) {
    let (op, f) = (wide("open"), wide(u));
    unsafe { windows_sys::Win32::UI::Shell::ShellExecuteW(std::ptr::null_mut(), op.as_ptr(), f.as_ptr(), std::ptr::null(), std::ptr::null(), SW_SHOWNORMAL) };
}

/// A path for one line of a page (its middle left out when long).
pub fn short(p: &std::path::Path, max: usize) -> String {
    let s = p.display().to_string();
    let c: Vec<char> = s.chars().collect();
    if c.len() <= max {
        return s;
    }
    let head = max / 3;
    let tail = max - head - 3;
    format!("{}...{}", c[..head].iter().collect::<String>(), c[c.len() - tail..].iter().collect::<String>())
}
