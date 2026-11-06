//! The 3D audio renderer the PS5's system provides (its part the title relies on): the title
//! renders its 3D sound to ambisonics (ACN channel order, SN3D normalization, 36 object ports up
//! to 5th order: `hle::libs::audio2`) and the system turns that sound field into the output. Here
//! (FINDINGS.md 2026-11-06 night):
//!
//! - a max-rE sampling decoder of order 3 (ACN 0..15) to 26 virtual speakers (12 around the
//!   listener, 6 above and 6 below at 45 degrees, one overhead and one below), normalized for
//!   unit energy over the sphere;
//! - headphones: each virtual speaker reaches each ear with a broadband level by the angle to the
//!   ear (down to -5.4 dB facing away) and Brown and Duda's head-shadow filter (a first-order
//!   shelf by that angle: +6 dB treble facing the ear, down to -20 dB behind the head): a sound to
//!   one side is louder and brighter in the near ear and still heard, darker, in the far one. No
//!   interaural delays: delayed copies summed over neighbouring virtual speakers cut a frontal
//!   source's treble by 11-19 dB from 2 kHz, and delaying the low band alone left a 9 dB dip at
//!   its crossover;
//! - stereo speakers: amplitude panning between speakers at +-30 degrees (sources further out at
//!   the speaker, rear ones folded forward at -3 dB, the far channel never below -12 dB);
//! - 5.1 and 7.1: pairwise panning between the layout's speakers around the listener;
//! - 7.1.4 for Windows' spatial sound (Windows Sonic, Dolby Atmos: their own head model or
//!   speakers): the 7.1 ring as above and four height speakers at 45 degrees, the virtual
//!   speakers above the ear plane crossfaded into the height layer by elevation.
//!
//! Axes as AmbiX: x forward, y left, z up; azimuth counterclockwise from the front.

use std::f32::consts::PI;

/// The decoder's order (ACN 0..15); the title's 4th and 5th orders are left out.
pub const ORDER: usize = 3;
pub const CHANNELS: usize = (ORDER + 1) * (ORDER + 1);

/// Real spherical harmonics, ACN order, SN3D (no Condon-Shortley phase), up to order 3, at the
/// unit vector (x, y, z).
pub fn sh(x: f32, y: f32, z: f32) -> [f32; CHANNELS] {
    let s3 = 3f32.sqrt();
    let s15 = 15f32.sqrt();
    let s58 = (5.0f32 / 8.0).sqrt();
    let s38 = (3.0f32 / 8.0).sqrt();
    [
        1.0,
        y,
        z,
        x,
        s3 * x * y,
        s3 * y * z,
        (3.0 * z * z - 1.0) / 2.0,
        s3 * x * z,
        s3 / 2.0 * (x * x - y * y),
        s58 * y * (3.0 * x * x - y * y),
        s15 * x * y * z,
        s38 * y * (5.0 * z * z - 1.0),
        z * (5.0 * z * z - 3.0) / 2.0,
        s38 * x * (5.0 * z * z - 1.0),
        s15 / 2.0 * z * (x * x - y * y),
        s58 * x * (x * x - 3.0 * y * y),
    ]
}

fn order_of(acn: usize) -> usize {
    (acn as f32).sqrt() as usize
}

fn unit(az: f32, el: f32) -> (f32, f32, f32) {
    (el.cos() * az.cos(), el.cos() * az.sin(), el.sin())
}

/// The virtual speakers (azimuth, elevation; radians).
fn speakers() -> Vec<(f32, f32)> {
    let d = PI / 180.0;
    let mut v = Vec::new();
    for k in 0..12 {
        v.push((k as f32 * 30.0 * d, 0.0));
    }
    for k in 0..6 {
        v.push((k as f32 * 60.0 * d, 45.0 * d));
        v.push(((30.0 + k as f32 * 60.0) * d, -45.0 * d));
    }
    v.push((0.0, 90.0 * d));
    v.push((0.0, -90.0 * d));
    v
}

/// The output the field is rendered for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    Headphones,
    Stereo,
    /// 5.1: FL FR FC LFE BL BR (WASAPI order; surrounds at +-110 degrees).
    Surround51,
    /// 7.1: FL FR FC LFE BL BR SL SR (backs at +-150, sides at +-90 degrees).
    Surround71,
    /// 7.1.4 for Windows' spatial sound: the 7.1 order, then TFL TFR TBL TBR (+-45 and +-135
    /// degrees at 45 degrees up).
    Spatial714,
}

impl Layout {
    pub fn channels(self) -> usize {
        match self {
            Layout::Headphones | Layout::Stereo => 2,
            Layout::Surround51 => 6,
            Layout::Surround71 => 8,
            Layout::Spatial714 => 12,
        }
    }
}

/// One ear's path from a virtual speaker: a level and a first-order head-shadow filter
/// (`y = b0 x + b1 x1 - a1 y1`, the level folded into b0 and b1).
#[derive(Clone, Copy, Default)]
struct Ear {
    b0: f32,
    b1: f32,
    a1: f32,
    x1: f32,
    y1: f32,
}

pub struct Renderer {
    layout: Layout,
    /// Per virtual speaker: the decoder's weights per ACN channel.
    decode: Vec<[f32; CHANNELS]>,
    /// Headphones: per virtual speaker, the left and right ear paths.
    ears: Vec<[Ear; 2]>,
    /// Per virtual speaker: this frame's sample.
    cur: Vec<f32>,
    /// Speakers: per virtual speaker, its gain into each output channel.
    pans: Vec<Vec<f32>>,
    /// The output gain making a frontal plane wave of amplitude 1 come out at 0.707 per side
    /// (as the earlier decode did), measured at construction.
    gain: f32,
}

impl Renderer {
    pub fn new(layout: Layout) -> Renderer {
        let spk = speakers();
        let n = ORDER + 1;
        // max-rE weights per order (cosine of 137.9 degrees / (order + 1.51)).
        let ce = (137.9f32 / (ORDER as f32 + 1.51)).to_radians().cos();
        let legendre = |l: usize, x: f32| -> f32 {
            let (mut p0, mut p1) = (1.0f32, x);
            if l == 0 {
                return p0;
            }
            for k in 1..l {
                let p2 = ((2 * k + 1) as f32 * x * p1 - k as f32 * p0) / (k + 1) as f32;
                p0 = p1;
                p1 = p2;
            }
            p1
        };
        let g: Vec<f32> = (0..n).map(|l| legendre(l, ce)).collect();
        let mut decode: Vec<[f32; CHANNELS]> = spk
            .iter()
            .map(|&(az, el)| {
                let (x, y, z) = unit(az, el);
                let y_ = sh(x, y, z);
                let mut w = [0f32; CHANNELS];
                for (acn, w) in w.iter_mut().enumerate() {
                    let l = order_of(acn);
                    *w = (2 * l + 1) as f32 * g[l] * y_[acn];
                }
                w
            })
            .collect();
        // Unit energy over the sphere: plane waves from a dense set of directions.
        let mut e = 0f64;
        let mut count = 0;
        for i in 0..36 {
            for j in -8..=8 {
                let (az, el) = (i as f32 * 10f32.to_radians(), j as f32 * 10f32.to_radians());
                let (x, y, z) = unit(az, el);
                let b = sh(x, y, z);
                let w = el.cos().max(0.05) as f64;
                e += w * decode.iter().map(|d| (0..CHANNELS).map(|k| d[k] * b[k]).sum::<f32>().powi(2) as f64).sum::<f64>();
                count += 1;
                let _ = count;
            }
        }
        let wsum: f64 = (0..36).flat_map(|_| -8..=8).map(|j: i32| (j as f32 * 10f32.to_radians()).cos().max(0.05) as f64).sum();
        let c = (1.0 / (e / wsum)).sqrt() as f32;
        for d in &mut decode {
            for w in d.iter_mut() {
                *w *= c;
            }
        }
        let (ears, pans) = match layout {
            Layout::Headphones => (spk.iter().map(|&(az, el)| head_paths(az, el)).collect(), Vec::new()),
            _ => (Vec::new(), spk.iter().map(|&(az, el)| pan(layout, az, el)).collect()),
        };
        let mut r = Renderer { layout, decode, cur: vec![0f32; spk.len()], ears, pans, gain: 1.0 };
        r.gain = r.front_gain();
        r
    }

    pub fn layout(&self) -> Layout {
        self.layout
    }

    /// The gain making a frontal plane wave come out at 0.707 per side (DC: the head filters'
    /// gain is 1 there; delays do not matter).
    fn front_gain(&self) -> f32 {
        let b = sh(1.0, 0.0, 0.0);
        let s: Vec<f32> = self.decode.iter().map(|d| (0..CHANNELS).map(|k| d[k] * b[k]).sum()).collect();
        let left: f32 = match self.layout {
            // Each ear path's gain at DC: (b0 + b1) / (1 + a1).
            Layout::Headphones => s.iter().zip(&self.ears).map(|(v, e)| v * (e[0].b0 + e[0].b1) / (1.0 + e[0].a1)).sum(),
            _ => s.iter().zip(&self.pans).map(|(v, p)| v * p[0]).sum::<f32>() + s.iter().zip(&self.pans).map(|(v, p)| v * p.get(2).copied().unwrap_or(0.0) * 0.7071).sum::<f32>(),
        };
        if left.abs() > 1e-6 { 0.7071 / left } else { 1.0 }
    }

    /// Adds `n` frames of the field (`acn[k]`: ACN channel k's samples, None when silent) into
    /// `out` (`n` x the layout's channels, interleaved).
    pub fn render(&mut self, acn: &[Option<&[f32]>; CHANNELS], n: usize, out: &mut [f32]) {
        let ch = self.layout.channels();
        let present: Vec<(usize, &[f32])> = acn.iter().enumerate().filter_map(|(k, s)| s.map(|s| (k, s))).collect();
        if present.is_empty() {
            return;
        }
        let gain = self.gain;
        for f in 0..n {
            // Each virtual speaker's sample.
            for (i, d) in self.decode.iter().enumerate() {
                let mut v = 0f32;
                for &(k, s) in &present {
                    v += d[k] * s[f];
                }
                self.cur[i] = v;
            }
            match self.layout {
                Layout::Headphones => {
                    let (mut l, mut r) = (0f32, 0f32);
                    for (ears, &x) in self.ears.iter_mut().zip(&self.cur) {
                        for (e, acc) in ears.iter_mut().zip([&mut l, &mut r]) {
                            let y = e.b0 * x + e.b1 * e.x1 - e.a1 * e.y1;
                            e.x1 = x;
                            e.y1 = y;
                            *acc += y;
                        }
                    }
                    out[2 * f] += l * gain;
                    out[2 * f + 1] += r * gain;
                }
                _ => {
                    for (p, &v) in self.pans.iter().zip(&self.cur) {
                        let v = v * gain;
                        for c in 0..ch {
                            out[f * ch + c] += v * p[c];
                        }
                    }
                }
            }
        }
    }
}

/// The two ear paths of a virtual speaker at (az, el): a level (1 facing the ear, 0.54 facing
/// away: (1 + 0.3 cos) / 1.3) and Brown and Duda's head shadow for a head of radius 8.75 cm, at
/// 48 kHz.
fn head_paths(az: f32, el: f32) -> [Ear; 2] {
    let (_, y, _) = unit(az, el);
    let (a, c, fs) = (0.0875f32, 343.0f32, 48000.0f32);
    let ear = |cos_to_ear: f32| -> Ear {
        let cos_to_ear = cos_to_ear.clamp(-1.0, 1.0);
        let g = cos_to_ear.acos();
        let level = (1.0 + 0.3 * cos_to_ear) / 1.3;
        let (amin, tmin) = (0.1f32, 150f32.to_radians());
        let alpha = (1.0 + amin / 2.0) + (1.0 - amin / 2.0) * (g / tmin * PI).cos();
        let tau = a / (2.0 * c);
        let k = 2.0 * fs;
        let den = 1.0 + tau * k;
        Ear { b0: level * (1.0 + alpha * tau * k) / den, b1: level * (1.0 - alpha * tau * k) / den, a1: (1.0 - tau * k) / den, x1: 0.0, y1: 0.0 }
    };
    // The left ear's axis is +y, the right's -y.
    [ear(y), ear(-y)]
}

/// A virtual speaker's gains into the layout's channels.
fn pan(layout: Layout, az: f32, el: f32) -> Vec<f32> {
    let deg = |r: f32| r.to_degrees();
    // Azimuth in (-180, 180], positive left; elevated speakers count at their azimuth.
    let mut a = deg(az);
    if a > 180.0 {
        a -= 360.0;
    }
    let _ = el;
    match layout {
        Layout::Stereo | Layout::Headphones => {
            // Rear sources fold forward at -3 dB.
            let (mut a, mut g) = (a, 1.0f32);
            if a.abs() > 90.0 {
                a = a.signum() * (180.0 - a.abs());
                g = 0.7071;
            }
            // Within +-30 degrees between the speakers (tangent law), beyond at the speaker; the
            // far channel at least -12 dB.
            let p = (a / 30.0).clamp(-1.0, 1.0);
            let th = (p + 1.0) * PI / 4.0;
            let (l, r) = (th.sin(), th.cos());
            let floor = 0.25 * l.max(r);
            vec![g * l.max(floor), g * r.max(floor)]
        }
        Layout::Spatial714 => {
            // Above the ear plane: crossfaded into the height layer by elevation (all of it at 45
            // degrees and above, spread over the four at the top).
            let mut gains = pan(Layout::Surround71, az, 0.0);
            gains.resize(12, 0.0);
            let e = el.to_degrees();
            if e > 0.0 {
                let t = (e / 45.0).min(1.0) * PI / 2.0;
                let (ring, top) = (t.cos(), t.sin());
                for g in gains.iter_mut().take(8) {
                    *g *= ring;
                }
                if e > 80.0 {
                    for g in gains.iter_mut().skip(8) {
                        *g = top * 0.5;
                    }
                } else {
                    let heights: [(f32, usize); 4] = [(45.0, 8), (-45.0, 9), (135.0, 10), (-135.0, 11)];
                    let h = pan_ring(&heights, a, 12);
                    for k in 8..12 {
                        gains[k] = top * h[k];
                    }
                }
            }
            gains
        }
        Layout::Surround51 | Layout::Surround71 => {
            // The layout's speakers around the listener (degrees, positive left) and their channels.
            let ring: &[(f32, usize)] = if layout == Layout::Surround71 {
                &[(30.0, 0), (0.0, 2), (-30.0, 1), (-90.0, 7), (-150.0, 5), (150.0, 4), (90.0, 6)]
            } else {
                &[(30.0, 0), (0.0, 2), (-30.0, 1), (-110.0, 5), (110.0, 4)]
            };
            let mut gains = vec![0f32; layout.channels()];
            // The pair of neighbours around `a` (sorted by angle), constant-power panning between them.
            let mut sorted: Vec<(f32, usize)> = ring.to_vec();
            sorted.sort_by(|x, y| x.0.total_cmp(&y.0));
            let norm = |x: f32| if x < 0.0 { x + 360.0 } else { x };
            let an = norm(a);
            let m = sorted.len();
            for k in 0..m {
                let (a0, c0) = sorted[k];
                let (a1, c1) = sorted[(k + 1) % m];
                let (s0, mut s1) = (norm(a0), norm(a1));
                if s1 <= s0 {
                    s1 += 360.0;
                }
                let mut t = an;
                if t < s0 {
                    t += 360.0;
                }
                if t >= s0 && t <= s1 {
                    let f = (t - s0) / (s1 - s0);
                    gains[c0] += ((1.0 - f) * PI / 2.0).sin();
                    gains[c1] += (f * PI / 2.0).sin();
                    break;
                }
            }
            gains
        }
    }
}

/// Constant-power panning of azimuth `a` (degrees, positive left) between the neighbours on a
/// ring of (azimuth, channel) speakers, into `n` channels.
fn pan_ring(ring: &[(f32, usize)], a: f32, n: usize) -> Vec<f32> {
    let mut gains = vec![0f32; n];
    let mut sorted: Vec<(f32, usize)> = ring.to_vec();
    sorted.sort_by(|x, y| x.0.total_cmp(&y.0));
    let norm = |x: f32| if x < 0.0 { x + 360.0 } else { x };
    let an = norm(a);
    let m = sorted.len();
    for k in 0..m {
        let (a0, c0) = sorted[k];
        let (a1, c1) = sorted[(k + 1) % m];
        let (s0, mut s1) = (norm(a0), norm(a1));
        if s1 <= s0 {
            s1 += 360.0;
        }
        let mut t = an;
        if t < s0 {
            t += 360.0;
        }
        if t >= s0 && t <= s1 {
            let f = (t - s0) / (s1 - s0);
            gains[c0] += ((1.0 - f) * PI / 2.0).sin();
            gains[c1] += (f * PI / 2.0).sin();
            break;
        }
    }
    gains
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plane(az_deg: f32) -> [Vec<f32>; CHANNELS] {
        let (x, y, z) = unit(az_deg.to_radians(), 0.0);
        let b = sh(x, y, z);
        // A 1 kHz tone of amplitude 1 over 4800 samples.
        let tone: Vec<f32> = (0..4800).map(|i| (2.0 * PI * 1000.0 * i as f32 / 48000.0).sin()).collect();
        std::array::from_fn(|k| tone.iter().map(|v| v * b[k]).collect())
    }

    fn rms(v: impl Iterator<Item = f32>) -> f32 {
        let (s, n) = v.fold((0f64, 0u32), |(s, n), x| (s + (x * x) as f64, n + 1));
        (s / n.max(1) as f64).sqrt() as f32
    }

    fn render(layout: Layout, az: f32) -> Vec<f32> {
        let mut r = Renderer::new(layout);
        let field = plane(az);
        let acn: [Option<&[f32]>; CHANNELS] = std::array::from_fn(|k| Some(field[k].as_slice()));
        let ch = layout.channels();
        let mut out = vec![0f32; 4800 * ch];
        r.render(&acn, 4800, &mut out);
        out
    }

    #[test]
    fn headphones_keep_both_ears() {
        // Front: both ears alike.
        let o = render(Layout::Headphones, 0.0);
        let (l, r) = (rms(o.iter().step_by(2).copied().skip(200)), rms(o.iter().skip(1).step_by(2).copied().skip(200)));
        assert!((l / r - 1.0).abs() < 0.05, "front {l} {r}");
        // Hard left: the right ear still hears it (a few to ~20 dB lower at 1 kHz, not silence).
        let o = render(Layout::Headphones, 90.0);
        let (l, r) = (rms(o.iter().step_by(2).copied().skip(200)), rms(o.iter().skip(1).step_by(2).copied().skip(200)));
        let ild = 20.0 * (l / r).log10();
        assert!(ild > 2.0 && ild < 20.0, "left source ILD {ild} dB ({l} {r})");
        // Hard right: mirrored.
        let o = render(Layout::Headphones, -90.0);
        let (l, r) = (rms(o.iter().step_by(2).copied().skip(200)), rms(o.iter().skip(1).step_by(2).copied().skip(200)));
        assert!(r > l * 1.25, "right source {l} {r}");
    }

    #[test]
    fn stereo_pans_without_silencing_a_side() {
        let o = render(Layout::Stereo, 90.0);
        let (l, r) = (rms(o.iter().step_by(2).copied()), rms(o.iter().skip(1).step_by(2).copied()));
        assert!(l > r && r > 0.0, "{l} {r}");
        assert!(20.0 * (l / r).log10() < 14.0, "{l} {r}");
        let o = render(Layout::Stereo, 0.0);
        let (l, r) = (rms(o.iter().step_by(2).copied()), rms(o.iter().skip(1).step_by(2).copied()));
        assert!((l / r - 1.0).abs() < 0.05 && (l - 0.5).abs() < 0.15, "front {l} {r}");
    }

    #[test]
    fn surround_puts_sources_at_their_speakers() {
        // 7.1: a source at the back left lands mostly in BL (channel 4).
        let o = render(Layout::Surround71, 150.0);
        let e: Vec<f32> = (0..8).map(|c| rms(o.iter().skip(c).step_by(8).copied())).collect();
        let max = e.iter().copied().fold(0.0, f32::max);
        assert_eq!(e.iter().position(|&v| v == max), Some(4), "{e:?}");
        assert_eq!(e[3], 0.0, "LFE fed: {e:?}");
    }

    #[test]
    fn report_levels() {
        // (Run with --nocapture: the ear levels by direction, for the findings.)
        for az in [0.0f32, 30.0, 60.0, 90.0, 135.0, 180.0, -90.0] {
            let o = render(Layout::Headphones, az);
            let (l, r) = (rms(o.iter().step_by(2).copied().skip(200)), rms(o.iter().skip(1).step_by(2).copied().skip(200)));
            let s = render(Layout::Stereo, az);
            let (sl, sr) = (rms(s.iter().step_by(2).copied()), rms(s.iter().skip(1).step_by(2).copied()));
            eprintln!("az {az:6.1}: headphones L {l:.3} R {r:.3} ({:+.1} dB); stereo L {sl:.3} R {sr:.3} ({:+.1} dB)", 20.0 * (l / r).log10(), 20.0 * (sl / sr).log10());
        }
    }

    #[test]
    fn report_response() {
        // (Run with --nocapture: the headphones' response by frequency for a frontal and a side source.)
        for az in [0.0f32, 90.0] {
            let mut line = format!("az {az:5.1}:");
            for hz in [125.0f32, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 12000.0] {
                let (x, y, z) = unit(az.to_radians(), 0.0);
                let b = sh(x, y, z);
                let tone: Vec<f32> = (0..9600).map(|i| (2.0 * PI * hz * i as f32 / 48000.0).sin()).collect();
                let field: [Vec<f32>; CHANNELS] = std::array::from_fn(|k| tone.iter().map(|v| v * b[k]).collect());
                let acn: [Option<&[f32]>; CHANNELS] = std::array::from_fn(|k| Some(field[k].as_slice()));
                let mut r = Renderer::new(Layout::Headphones);
                let mut out = vec![0f32; 9600 * 2];
                r.render(&acn, 9600, &mut out);
                let (l, rr) = (rms(out.iter().step_by(2).copied().skip(2000)), rms(out.iter().skip(1).step_by(2).copied().skip(2000)));
                line += &format!(" {hz}: {:+.1}/{:+.1}", 20.0 * (l / 0.5).log10(), 20.0 * (rr / 0.5).log10());
            }
            eprintln!("{line}");
        }
    }

    #[test]
    fn spatial_bed_lifts_sources_above() {
        // A source 60 degrees up in front: mostly the front height pair.
        let (x, y, z) = unit(0.0, 60f32.to_radians());
        let b = sh(x, y, z);
        let tone: Vec<f32> = (0..4800).map(|i| (2.0 * PI * 1000.0 * i as f32 / 48000.0).sin()).collect();
        let field: [Vec<f32>; CHANNELS] = std::array::from_fn(|k| tone.iter().map(|v| v * b[k]).collect());
        let acn: [Option<&[f32]>; CHANNELS] = std::array::from_fn(|k| Some(field[k].as_slice()));
        let mut r = Renderer::new(Layout::Spatial714);
        let mut out = vec![0f32; 4800 * 12];
        r.render(&acn, 4800, &mut out);
        let e: Vec<f32> = (0..12).map(|c| rms(out.iter().skip(c).step_by(12).copied())).collect();
        let tops = e[8] + e[9];
        let ring: f32 = e[..8].iter().sum();
        assert!(tops > ring && (e[8] / e[9] - 1.0).abs() < 0.1, "{e:?}");
    }

    #[test]
    fn energy_is_even_over_directions() {
        let r = Renderer::new(Layout::Stereo);
        let mut es = Vec::new();
        for az in (0..360).step_by(15) {
            let (x, y, z) = unit((az as f32).to_radians(), 0.0);
            let b = sh(x, y, z);
            let e: f32 = r.decode.iter().map(|d| (0..CHANNELS).map(|k| d[k] * b[k]).sum::<f32>().powi(2)).sum();
            es.push(e);
        }
        let (lo, hi) = (es.iter().copied().fold(f32::MAX, f32::min), es.iter().copied().fold(0.0, f32::max));
        assert!(hi / lo < 1.6, "energy {lo}..{hi}");
    }
}
