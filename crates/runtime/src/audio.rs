//! Host audio output: 48 kHz stereo f32 streams mixed into a WASAPI
//! shared-mode render client (event driven). Producers (AudioOut2 contexts,
//! AudioOut ports) push fixed-size grains into their stream and block while
//! the stream already holds `depth` grains, which paces the title's audio
//! threads by the real output clock. Without an output device the same
//! queues drain on a timer (null device, logged).

use parking_lot::{Condvar, Mutex};
use std::collections::VecDeque;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

pub const RATE: u32 = 48_000;

#[derive(Default)]
struct Stream {
    grains: VecDeque<Vec<f32>>,
    /// Frames already consumed from the front grain.
    offset: usize,
    /// Frames pushed so far (diagnostics).
    pushed: u64,
    /// Peak absolute sample value pushed (diagnostics: 0 = silence).
    peak: f32,
}

struct Mixer {
    streams: Mutex<Vec<Stream>>,
    cv: Condvar,
}

static MIXER: OnceLock<Mixer> = OnceLock::new();
pub static UNDERRUN_FRAMES: AtomicU64 = AtomicU64::new(0);
pub static PLAYED_FRAMES: AtomicU64 = AtomicU64::new(0);
/// UNDERRUN_FRAMES and PLAYED_FRAMES at the last `profile reset`.
static AT_RESET: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];

/// Starts a profile window (`profile reset`).
pub fn reset_window() {
    AT_RESET[0].store(UNDERRUN_FRAMES.load(Ordering::Relaxed), Ordering::Relaxed);
    AT_RESET[1].store(PLAYED_FRAMES.load(Ordering::Relaxed), Ordering::Relaxed);
}

/// Played and underrun seconds since `profile reset` (profile report).
pub fn window_line() -> String {
    let u = UNDERRUN_FRAMES.load(Ordering::Relaxed).saturating_sub(AT_RESET[0].load(Ordering::Relaxed));
    let p = PLAYED_FRAMES.load(Ordering::Relaxed).saturating_sub(AT_RESET[1].load(Ordering::Relaxed));
    format!("    audio in the window: {:.1} s played, {:.2} s underrun\n", p as f64 / RATE as f64, u as f64 / RATE as f64)
}

/// A capture of the mix as played (`audio capture <seconds>`): 32-bit float
/// stereo WAV in the run directory.
struct Capture {
    path: std::path::PathBuf,
    frames_left: usize,
    samples: Vec<f32>,
    /// Samples the soft knee compressed (above `SOFT_KNEE` in the mix).
    knee: u64,
}

static CAPTURE: Mutex<Option<Capture>> = Mutex::new(None);
static CAPTURING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Starts capturing the next `seconds` of the mix to `<run dir>/audio-<n>.wav`.
pub fn start_capture(seconds: f64) -> Result<(), String> {
    let dir = crate::run_dir().ok_or("no run directory")?;
    let mut c = CAPTURE.lock();
    if c.is_some() {
        return Err("a capture is in progress".into());
    }
    let path = (0..).map(|n| dir.join(format!("audio-{n}.wav"))).find(|p| !p.exists()).unwrap();
    let frames = (seconds * RATE as f64) as usize;
    crate::info!("audio: capturing {seconds} s of the mix to {}", path.display());
    *c = Some(Capture { path, frames_left: frames, samples: Vec::with_capacity(2 * frames), knee: 0 });
    CAPTURING.store(true, Ordering::Relaxed);
    Ok(())
}

/// Appends mixed frames (before the soft knee) to the capture in progress
/// and writes the file once it is complete.
fn capture(mix: &[f32]) {
    let mut g = CAPTURE.lock();
    let Some(c) = g.as_mut() else { return };
    let n = (mix.len() / 2).min(c.frames_left);
    for &v in &mix[..2 * n] {
        if v.abs() > SOFT_KNEE {
            c.knee += 1;
        }
        c.samples.push(soft_knee(v));
    }
    c.frames_left -= n;
    if c.frames_left > 0 {
        return;
    }
    let c = g.take().unwrap();
    CAPTURING.store(false, Ordering::Relaxed);
    drop(g);
    let peak = c.samples.iter().fold(0f32, |m, v| m.max(v.abs()));
    let rms = (c.samples.iter().map(|&v| v as f64 * v as f64).sum::<f64>() / c.samples.len().max(1) as f64).sqrt();
    match write_wav(&c.path, &c.samples) {
        Ok(()) => crate::info!(
            "audio: capture written: {} ({:.1} s, rms {rms:.4}, peak {peak:.3}, {} samples above the knee)",
            c.path.display(),
            c.samples.len() as f64 / 2.0 / RATE as f64,
            c.knee
        ),
        Err(e) => crate::warn!("audio: capture {}: {e}", c.path.display()),
    }
}

fn write_wav(path: &std::path::Path, samples: &[f32]) -> std::io::Result<()> {
    let data = 4 * samples.len() as u32;
    let mut b = Vec::with_capacity(58 + data as usize);
    b.extend_from_slice(b"RIFF");
    b.extend_from_slice(&(50 + data).to_le_bytes());
    b.extend_from_slice(b"WAVEfmt ");
    b.extend_from_slice(&18u32.to_le_bytes());
    b.extend_from_slice(&3u16.to_le_bytes()); // IEEE float
    b.extend_from_slice(&2u16.to_le_bytes());
    b.extend_from_slice(&RATE.to_le_bytes());
    b.extend_from_slice(&(RATE * 8).to_le_bytes());
    b.extend_from_slice(&8u16.to_le_bytes());
    b.extend_from_slice(&32u16.to_le_bytes());
    b.extend_from_slice(&0u16.to_le_bytes());
    b.extend_from_slice(b"fact");
    b.extend_from_slice(&4u32.to_le_bytes());
    b.extend_from_slice(&(samples.len() as u32 / 2).to_le_bytes());
    b.extend_from_slice(b"data");
    b.extend_from_slice(&data.to_le_bytes());
    for v in samples {
        b.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, b)
}

/// Stream 0 belongs to AudioOut2 (one context mixes everything it pushes).
pub const STREAM_AUDIOOUT2: usize = 0;

fn mixer() -> &'static Mixer {
    MIXER.get_or_init(|| {
        let m = Mixer { streams: Mutex::new(Vec::new()), cv: Condvar::new() };
        crate::thread::spawn_host("audio-out", 1 << 20, Box::new(output_thread)).expect("audio thread");
        m
    })
}

pub fn init() {
    mixer();
}

/// Allocates a producer stream id.
pub fn new_stream() -> usize {
    let mut s = mixer().streams.lock();
    s.push(Stream::default());
    s.len() - 1
}

fn ensure(streams: &mut Vec<Stream>, id: usize) {
    while streams.len() <= id {
        streams.push(Stream::default());
    }
}

/// Queues one stereo interleaved grain on stream `id`, blocking while the
/// stream already holds `depth` grains.
pub fn push_stream(id: usize, grain: Vec<f32>, depth: usize) {
    // (The title mixes in its job system: any of its 13 job workers pushes
    // here, so raising producers raised the spinning workers to the GPU
    // front end's level and undid `gpu::prio`, run 20261013-123910.)
    let m = mixer();
    let mut s = m.streams.lock();
    ensure(&mut s, id);
    let _w = if s[id].grains.len() >= depth.max(1) { Some(crate::thread::waiting("audio output queue", id as u64, None)) } else { None };
    while s[id].grains.len() >= depth.max(1) {
        m.cv.wait(&mut s);
    }
    s[id].pushed += (grain.len() / 2) as u64;
    let peak = grain.iter().fold(0f32, |m, v| m.max(v.abs()));
    if peak > s[id].peak {
        s[id].peak = peak;
    }
    s[id].grains.push_back(grain);
}

/// One line per stream for status reports.
pub fn status() -> String {
    let Some(m) = MIXER.get() else { return String::new() };
    let Some(s) = m.streams.try_lock() else { return String::new() };
    let mut out = format!(
        "    audio: {:.1} s played, {:.2} s underrun\n",
        PLAYED_FRAMES.load(Ordering::Relaxed) as f64 / RATE as f64,
        UNDERRUN_FRAMES.load(Ordering::Relaxed) as f64 / RATE as f64
    );
    for (i, st) in s.iter().enumerate() {
        out.push_str(&format!("    audio stream {i}: {:.1} s pushed, {} grain(s) queued, peak {:.3}\n", st.pushed as f64 / RATE as f64, st.grains.len(), st.peak));
    }
    out
}

pub fn push_grain(grain: Vec<f32>, depth: usize) {
    push_stream(STREAM_AUDIOOUT2, grain, depth);
}

pub fn queued_grains() -> usize {
    let s = mixer().streams.lock();
    s.get(STREAM_AUDIOOUT2).map(|x| x.grains.len()).unwrap_or(0)
}

/// Mixes `frames` stereo frames from all streams into `out` (zero-filled first).
fn mix_into(out: &mut [f32], frames: usize) {
    out[..2 * frames].fill(0.0);
    let m = mixer();
    let mut s = m.streams.lock();
    let mut starved = false;
    for st in s.iter_mut() {
        let mut done = 0usize;
        while done < frames {
            let Some(front) = st.grains.front() else {
                if done > 0 || !st.grains.is_empty() {
                    starved = true;
                }
                break;
            };
            let avail = front.len() / 2 - st.offset;
            let n = avail.min(frames - done);
            for i in 0..n {
                out[2 * (done + i)] += front[2 * (st.offset + i)];
                out[2 * (done + i) + 1] += front[2 * (st.offset + i) + 1];
            }
            done += n;
            st.offset += n;
            if st.offset * 2 >= front.len() {
                st.grains.pop_front();
                st.offset = 0;
            }
        }
    }
    drop(s);
    m.cv.notify_all();
    if starved {
        UNDERRUN_FRAMES.fetch_add(frames as u64, Ordering::Relaxed);
    }
    if CAPTURING.load(Ordering::Relaxed) {
        capture(&out[..2 * frames]);
    }
    for v in out[..2 * frames].iter_mut() {
        *v = soft_knee(*v);
    }
    PLAYED_FRAMES.fetch_add(frames as u64, Ordering::Relaxed);
}

/// A soft knee above 0.8 instead of clipping at full scale (loud transients
/// of the 3D mix reached ~1.2 in combat).
fn soft_knee(v: f32) -> f32 {
    let a = v.abs();
    if a > SOFT_KNEE { v.signum() * (SOFT_KNEE + (1.0 - SOFT_KNEE) * ((a - SOFT_KNEE) / (1.0 - SOFT_KNEE)).tanh()) } else { v }
}

/// Output level from which samples are compressed smoothly toward full
/// scale (`mix_into`).
const SOFT_KNEE: f32 = 0.8;

fn output_thread() {
    crate::log::set_thread_name("audio-out");
    // Little work per wake-up, but late work is heard: above the title's
    // threads and the GPU front end's (`gpu::prio`).
    unsafe {
        use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_HIGHEST};
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
    }
    match wasapi::run() {
        Ok(()) => {}
        Err(e) => {
            crate::warn!("audio: no host output device ({e}); using a paced null device (silent)");
            null_device();
        }
    }
}

/// Drains the streams in real time without producing sound.
fn null_device() {
    let frames = 480usize; // 10 ms
    let mut buf = vec![0f32; 2 * frames];
    let start = std::time::Instant::now();
    let mut played = 0u64;
    loop {
        mix_into(&mut buf, frames);
        played += frames as u64;
        let due = start + std::time::Duration::from_secs_f64(played as f64 / RATE as f64);
        let now = std::time::Instant::now();
        if due > now {
            std::thread::sleep(due - now);
        }
    }
}

mod wasapi {
    use windows::Win32::Foundation::{CloseHandle, WAIT_OBJECT_0};
    use windows::Win32::Media::Audio::*;
    use windows::Win32::Media::KernelStreaming::{SPEAKER_FRONT_LEFT, SPEAKER_FRONT_RIGHT, WAVE_FORMAT_EXTENSIBLE};
    use windows::Win32::Media::Multimedia::KSDATAFORMAT_SUBTYPE_IEEE_FLOAT;
    use windows::Win32::System::Com::{CLSCTX_ALL, COINIT_MULTITHREADED, CoCreateInstance, CoInitializeEx};
    use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};

    pub fn run() -> Result<(), String> {
        unsafe {
            CoInitializeEx(None, COINIT_MULTITHREADED).ok().map_err(|e| format!("CoInitializeEx: {e}"))?;
            let en: IMMDeviceEnumerator = CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL).map_err(|e| format!("device enumerator: {e}"))?;
            let dev = en.GetDefaultAudioEndpoint(eRender, eConsole).map_err(|e| format!("default endpoint: {e}"))?;
            let client: IAudioClient = dev.Activate(CLSCTX_ALL, None).map_err(|e| format!("activate: {e}"))?;
            let mut fmt: WAVEFORMATEXTENSIBLE = std::mem::zeroed();
            fmt.Format.wFormatTag = WAVE_FORMAT_EXTENSIBLE as u16;
            fmt.Format.nChannels = 2;
            fmt.Format.nSamplesPerSec = super::RATE;
            fmt.Format.wBitsPerSample = 32;
            fmt.Format.nBlockAlign = 8;
            fmt.Format.nAvgBytesPerSec = super::RATE * 8;
            fmt.Format.cbSize = 22;
            fmt.Samples.wValidBitsPerSample = 32;
            fmt.dwChannelMask = SPEAKER_FRONT_LEFT | SPEAKER_FRONT_RIGHT;
            fmt.SubFormat = KSDATAFORMAT_SUBTYPE_IEEE_FLOAT;
            let flags = AUDCLNT_STREAMFLAGS_EVENTCALLBACK | AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM | AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY;
            // 30 ms device buffer (100 ns units).
            client
                .Initialize(AUDCLNT_SHAREMODE_SHARED, flags, 300_000, 0, &fmt as *const _ as *const WAVEFORMATEX, None)
                .map_err(|e| format!("initialize: {e}"))?;
            let event = CreateEventW(None, false, false, None).map_err(|e| format!("event: {e}"))?;
            client.SetEventHandle(event).map_err(|e| format!("event handle: {e}"))?;
            let render: IAudioRenderClient = client.GetService().map_err(|e| format!("render client: {e}"))?;
            let total = client.GetBufferSize().map_err(|e| format!("buffer size: {e}"))?;
            client.Start().map_err(|e| format!("start: {e}"))?;
            crate::info!("audio: WASAPI shared-mode output, 48 kHz stereo float, device buffer {total} frames");
            let mut scratch = vec![0f32; 2 * total as usize];
            loop {
                let w = WaitForSingleObject(event, 200);
                if w != WAIT_OBJECT_0 && w.0 != 0x102 {
                    break;
                }
                let padding = match client.GetCurrentPadding() {
                    Ok(p) => p,
                    Err(e) => {
                        crate::warn!("audio: device lost ({e}); switching to the null device");
                        let _ = CloseHandle(event);
                        return Err(e.to_string());
                    }
                };
                let avail = (total - padding) as usize;
                if avail == 0 {
                    continue;
                }
                super::mix_into(&mut scratch, avail);
                if let Ok(p) = render.GetBuffer(avail as u32) {
                    std::ptr::copy_nonoverlapping(scratch.as_ptr(), p as *mut f32, 2 * avail);
                    let _ = render.ReleaseBuffer(avail as u32, 0);
                }
            }
            let _ = CloseHandle(event);
            Err("event wait failed".into())
        }
    }
}
