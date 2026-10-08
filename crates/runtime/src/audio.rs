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
        "    audio: {:.1} s played, {:.2} s underrun
",
        PLAYED_FRAMES.load(Ordering::Relaxed) as f64 / RATE as f64,
        UNDERRUN_FRAMES.load(Ordering::Relaxed) as f64 / RATE as f64
    );
    for (i, st) in s.iter().enumerate() {
        out.push_str(&format!("    audio stream {i}: {:.1} s pushed, {} grain(s) queued, peak {:.3}
", st.pushed as f64 / RATE as f64, st.grains.len(), st.peak));
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
    for v in out[..2 * frames].iter_mut() {
        *v = v.clamp(-1.0, 1.0);
    }
    PLAYED_FRAMES.fetch_add(frames as u64, Ordering::Relaxed);
}

fn output_thread() {
    crate::log::set_thread_name("audio-out");
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
