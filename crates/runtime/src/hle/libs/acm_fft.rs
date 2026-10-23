//! A real FFT of 2048 points for the convolution reverb (`acm`): a complex
//! FFT of 1024 points over the even and odd samples, then the split step.
//! Spectra are packed as the title's impulse responses are (FINDINGS.md
//! 2026-10-23): 1024 complex values, bin 0 holding the DC term as its real
//! part and the Nyquist term as its imaginary part.

pub const N: usize = 2048;
const H: usize = N / 2;

#[derive(Clone, Copy, Default, Debug)]
#[repr(C)]
pub struct C {
    pub re: f32,
    pub im: f32,
}

impl C {
    #[inline]
    fn mul(self, o: C) -> C {
        C { re: self.re * o.re - self.im * o.im, im: self.re * o.im + self.im * o.re }
    }
}

pub struct Fft {
    /// Twiddles of the complex FFT of H points: e^(-2 pi i k / H).
    tw: Vec<C>,
    /// e^(-2 pi i k / N) for the split step.
    split: Vec<C>,
    rev: Vec<u32>,
}

impl Fft {
    pub fn new() -> Fft {
        let tw = (0..H / 2).map(|k| {
            let a = -2.0 * std::f64::consts::PI * k as f64 / H as f64;
            C { re: a.cos() as f32, im: a.sin() as f32 }
        });
        let split = (0..H).map(|k| {
            let a = -2.0 * std::f64::consts::PI * k as f64 / N as f64;
            C { re: a.cos() as f32, im: a.sin() as f32 }
        });
        let bits = H.trailing_zeros();
        let rev = (0..H as u32).map(|i| i.reverse_bits() >> (32 - bits)).collect();
        Fft { tw: tw.collect(), split: split.collect(), rev }
    }

    /// In-place complex FFT of H points (forward: e^-; the inverse is the
    /// conjugated transform, unscaled).
    fn cfft(&self, a: &mut [C], inverse: bool) {
        for i in 0..H {
            let j = self.rev[i] as usize;
            if i < j {
                a.swap(i, j);
            }
        }
        let mut len = 2;
        while len <= H {
            let step = H / len;
            for start in (0..H).step_by(len) {
                for k in 0..len / 2 {
                    let mut w = self.tw[k * step];
                    if inverse {
                        w.im = -w.im;
                    }
                    let u = a[start + k];
                    let v = a[start + k + len / 2].mul(w);
                    a[start + k] = C { re: u.re + v.re, im: u.im + v.im };
                    a[start + k + len / 2] = C { re: u.re - v.re, im: u.im - v.im };
                }
            }
            len *= 2;
        }
    }

    /// The packed spectrum of `x` (N samples).
    pub fn forward(&self, x: &[f32], out: &mut [C]) {
        for k in 0..H {
            out[k] = C { re: x[2 * k], im: x[2 * k + 1] };
        }
        self.cfft(out, false);
        // Split: X[k] = (Z[k] + conj Z[H-k]) / 2 + w^k (Z[k] - conj Z[H-k]) / 2i.
        let z0 = out[0];
        out[0] = C { re: z0.re + z0.im, im: z0.re - z0.im };
        for k in 1..=H / 2 {
            let (a, b) = (out[k], out[H - k]);
            let e = C { re: 0.5 * (a.re + b.re), im: 0.5 * (a.im - b.im) };
            let o = C { re: 0.5 * (a.im + b.im), im: -0.5 * (a.re - b.re) };
            let t = o.mul(self.split[k]);
            out[k] = C { re: e.re + t.re, im: e.im + t.im };
            if k != H - k {
                // X[H-k] = conj(e) - conj(t) rotated: from the same pair.
                let t2 = C { re: o.re, im: -o.im }.mul(self.split[H - k]);
                out[H - k] = C { re: e.re + t2.re, im: -e.im + t2.im };
            }
        }
    }

    /// The N samples of the packed spectrum `x` (the inverse of `forward`,
    /// scaled by 1/N).
    pub fn inverse(&self, x: &[C], out: &mut [f32], scratch: &mut [C]) {
        // Z[k] = E[k] + i O[k], E[k] = (X[k] + conj X[H-k]) / 2,
        // O[k] = (X[k] - conj X[H-k]) / 2 * conj(w^k).
        let x0 = x[0];
        scratch[0] = C { re: 0.5 * (x0.re + x0.im), im: 0.5 * (x0.re - x0.im) };
        for k in 1..H {
            let (a, b) = (x[k], x[H - k]);
            let e = C { re: 0.5 * (a.re + b.re), im: 0.5 * (a.im - b.im) };
            let d = C { re: 0.5 * (a.re - b.re), im: 0.5 * (a.im + b.im) };
            let w = self.split[k];
            let o = d.mul(C { re: w.re, im: -w.im });
            scratch[k] = C { re: e.re - o.im, im: e.im + o.re };
        }
        self.cfft(scratch, true);
        let s = 1.0 / H as f32;
        for k in 0..H {
            out[2 * k] = scratch[k].re * s;
            out[2 * k + 1] = scratch[k].im * s;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_and_known_spectrum() {
        let f = Fft::new();
        let mut x = vec![0f32; N];
        for (i, v) in x.iter_mut().enumerate() {
            *v = ((i * 7919) % 101) as f32 / 50.0 - 1.0;
        }
        let mut spec = vec![C::default(); H];
        f.forward(&x, &mut spec);
        // Against a direct DFT at a few bins (packed bin 0: DC, Nyquist).
        let dft = |k: usize| -> (f64, f64) {
            let mut s = (0f64, 0f64);
            for (n, &v) in x.iter().enumerate() {
                let a = -2.0 * std::f64::consts::PI * (k * n) as f64 / N as f64;
                s.0 += v as f64 * a.cos();
                s.1 += v as f64 * a.sin();
            }
            s
        };
        let (dc, _) = dft(0);
        let (ny, _) = dft(H);
        assert!((spec[0].re as f64 - dc).abs() < 1e-2, "dc {} {}", spec[0].re, dc);
        assert!((spec[0].im as f64 - ny).abs() < 1e-2, "nyquist {} {}", spec[0].im, ny);
        for k in [1usize, 2, 17, 511, 512, 513, 1000, 1023] {
            let (re, im) = dft(k);
            assert!((spec[k].re as f64 - re).abs() < 1e-2 && (spec[k].im as f64 - im).abs() < 1e-2, "bin {k}: {:?} against ({re}, {im})", spec[k]);
        }
        let mut back = vec![0f32; N];
        let mut scratch = vec![C::default(); H];
        f.inverse(&spec, &mut back, &mut scratch);
        for (a, b) in x.iter().zip(&back) {
            assert!((a - b).abs() < 1e-4, "{a} {b}");
        }
    }
}
