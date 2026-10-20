//! Deterministic CPU-only alternating optimization. This intentionally does not
//! reproduce the reference SOAP optimizer, presets, soft rounding, or IFCE.
//! The distortion + L1 latent-rate surrogate is optimized by scalar coordinate
//! updates and small ridge regressions; no tensor framework/autograd is needed.
use crate::{
    Error, Result, check_size, entropy,
    model::{Model, context, solve},
    stream,
};

pub struct Encoder {
    source: Vec<[f32; 3]>,
    model: Model,
    grids: [Vec<i32>; 3],
    passes: u32,
    completed: u32,
    penalty: f64,
}
impl Encoder {
    pub fn new(
        rgba: &[u8],
        width: usize,
        height: usize,
        quality: u32,
        effort: u32,
    ) -> Result<Self> {
        check_size(width, height)?;
        if rgba.len() != width * height * 4 {
            return Err(Error("RGBA input length does not match dimensions"));
        }
        if !(1..=100).contains(&quality) || !(1..=9).contains(&effort) {
            return Err(Error("Quality must be 1–100 and effort 1–9"));
        }
        let source: Vec<[f32; 3]> = rgba
            .chunks_exact(4)
            .map(|v| {
                [
                    v[0] as f32 / 255.0,
                    v[1] as f32 / 255.0,
                    v[2] as f32 / 255.0,
                ]
            })
            .collect();
        let gain = 4.0 + quality as f32 * 0.44;
        let mut model = Model::initial(width, height, gain);
        model.quantize();
        let grids = std::array::from_fn(|i| {
            let (w, h) = model.size(i);
            let block = 1 << i;
            let mut grid = Vec::with_capacity(w * h);
            for gy in 0..h {
                for gx in 0..w {
                    let mut sum = 0.0;
                    let mut n = 0;
                    for y in gy * block..((gy + 1) * block).min(height) {
                        for x in gx * block..((gx + 1) * block).min(width) {
                            let [r, g, b] = source[y * width + x];
                            let luma = 0.299 * r + 0.587 * g + 0.114 * b;
                            sum += match i {
                                0 => luma - 0.5,
                                1 => b - luma,
                                _ => r - luma,
                            };
                            n += 1;
                        }
                    }
                    grid.push((gain * sum / n as f32).round_ties_even().clamp(-64.0, 63.0) as i32);
                }
            }
            grid
        });
        Ok(Self {
            source,
            model,
            grids,
            passes: 2 + effort * 2,
            completed: 0,
            penalty: 0.0001 * ((101 - quality) as f64 / 100.0).powi(2),
        })
    }
    /// One bounded optimization pass. The browser yields between calls so it
    /// can display progress. Cancellation terminates the enclosing worker.
    pub fn step(&mut self) -> bool {
        if self.completed >= self.passes {
            return true;
        }
        self.refine_latents();
        self.fit_synthesis();
        self.completed += 1;
        self.completed >= self.passes
    }
    pub fn completed(&self) -> u32 {
        self.completed
    }
    pub fn passes(&self) -> u32 {
        self.passes
    }
    fn refine_latents(&mut self) {
        let m = &self.model;
        for i in 0..3 {
            let (w, h) = m.size(i);
            let block = 1 << i;
            let norm: f64 = m.synthesis.iter().map(|w| f64::from(w[i]).powi(2)).sum();
            if norm < 1e-12 {
                continue;
            }
            for gy in 0..h {
                for gx in 0..w {
                    let mut numerator = 0.0;
                    let mut n = 0;
                    for y in gy * block..((gy + 1) * block).min(m.height) {
                        for x in gx * block..((gx + 1) * block).min(m.width) {
                            let idx = m.indices(x, y);
                            let z = std::array::from_fn(|j| self.grids[j][idx[j]] as f32);
                            let prediction = m.rgb(z);
                            let target = self.source[y * m.width + x];
                            for c in 0..3 {
                                numerator += f64::from(m.synthesis[c][i])
                                    * f64::from(
                                        target[c] - prediction[c] + m.synthesis[c][i] * z[i],
                                    );
                            }
                            n += 1;
                        }
                    }
                    let value = numerator.signum()
                        * (numerator.abs() - self.penalty / 2.0).max(0.0)
                        / (norm * n as f64);
                    self.grids[i][gy * w + gx] = value.round_ties_even().clamp(-64.0, 63.0) as i32;
                }
            }
        }
    }
    fn fit_synthesis(&mut self) {
        let mut a = [[0.0; 4]; 4];
        let mut b = [[0.0; 4]; 3];
        let stride = (self.source.len() / 65536).max(1);
        for p in (0..self.source.len()).step_by(stride) {
            let idx = self
                .model
                .indices(p % self.model.width, p / self.model.width);
            let z = [
                self.grids[0][idx[0]] as f64,
                self.grids[1][idx[1]] as f64,
                self.grids[2][idx[2]] as f64,
                1.0,
            ];
            for j in 0..4 {
                for k in 0..4 {
                    a[j][k] += z[j] * z[k];
                }
                for (c, bc) in b.iter_mut().enumerate() {
                    bc[j] += z[j] * f64::from(self.source[p][c]);
                }
            }
        }
        // Anchor rank-deficient/constant images to their preceding model.
        for (j, row) in a.iter_mut().enumerate() {
            row[j] += 0.1;
            for (c, bc) in b.iter_mut().enumerate() {
                bc[j] += 0.1 * f64::from(self.model.synthesis[c][j]);
            }
        }
        for (c, bc) in b.into_iter().enumerate() {
            if let Some(w) = solve(a, bc) {
                self.model.synthesis[c] = w.map(|v| v.clamp(-4.0, 4.0) as f32);
            }
        }
        self.model.quantize();
    }
    fn fit_arm(&mut self) {
        let mut a = [[0.0; 7]; 7];
        let mut b = [0.0; 7];
        for i in 0..3 {
            let (w, h) = self.model.size(i);
            let stride = (w * h / 65536).max(1);
            for p in (0..w * h).step_by(stride) {
                let ctx = context(&self.grids[i], w, h, p % w, p / w);
                let z: [f64; 7] =
                    std::array::from_fn(|j| if j == 6 { 1.0 } else { f64::from(ctx[j]) });
                for j in 0..7 {
                    b[j] += z[j] * f64::from(self.grids[i][p]);
                    for k in 0..7 {
                        a[j][k] += z[j] * z[k];
                    }
                }
            }
        }
        for (j, row) in a.iter_mut().enumerate() {
            row[j] += 1.0;
        }
        if let Some(w) = solve(a, b) {
            for (j, &v) in w[..6].iter().enumerate() {
                self.model.arm_w[j] = (v.clamp(-4.0, 4.0) * 256.0).round_ties_even() as i32;
            }
            self.model.arm_b[0] = (w[6].clamp(-64.0, 63.0) * 65536.0).round_ties_even() as i32;
            // Keep all parameter magnitudes within the declared profile bound.
            self.model.arm_b[0] = self.model.arm_b[0].clamp(-(1 << 20), 1 << 20);
        }
        let mut abs_error = 0.0;
        let mut n = 0;
        for i in 0..3 {
            let (w, h) = self.model.size(i);
            let stride = (w * h / 65536).max(1);
            for p in (0..w * h).step_by(stride) {
                let (mu, _) = self
                    .model
                    .probability(context(&self.grids[i], w, h, p % w, p / w));
                abs_error += (f64::from(self.grids[i][p]) - mu).abs();
                n += 1;
            }
        }
        let log_scale = (abs_error / n as f64 + 0.05).ln().clamp(-5.0, 5.0);
        self.model.arm_b[1] = ((log_scale + 4.0) * 65536.0).round_ties_even() as i32;
    }
    pub fn finish(mut self) -> Result<Vec<u8>> {
        if self.completed < self.passes {
            return Err(Error("Cool-chic training is not complete"));
        }
        self.fit_arm();
        let latent = entropy::encode(&self.model, &self.grids)?;
        Ok(stream::pack(&self.model, &latent))
    }
}
