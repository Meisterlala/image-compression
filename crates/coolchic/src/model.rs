//! Small per-image model: three hierarchical grids, a linear RGB synthesis,
//! nearest-neighbor separable upsampling, and a learned six-context linear ARM.
use crate::{Error, Result};

pub(crate) const CONTEXT: [(isize, isize); 6] =
    [(0, -1), (-1, 0), (-1, -1), (-1, 1), (0, -2), (-2, 0)];

pub(crate) struct Model {
    pub width: usize,
    pub height: usize,
    pub synthesis: [[f32; 4]; 3],
    // Quantized reference ARM parameters, q_w = 2^-8, q_b = 2^-16.
    pub arm_w: [i32; 12],
    pub arm_b: [i32; 2],
}
impl Model {
    pub fn initial(width: usize, height: usize, gain: f32) -> Self {
        Self {
            width,
            height,
            synthesis: [
                [1.0 / gain, 0.0, 1.0 / gain, 0.5],
                [
                    1.0 / gain,
                    -0.114 / 0.587 / gain,
                    -0.299 / 0.587 / gain,
                    0.5,
                ],
                [1.0 / gain, 1.0 / gain, 0.0, 0.5],
            ],
            arm_w: [0; 12],
            arm_b: [0, 4 * 65536],
        }
    }
    pub fn size(&self, i: usize) -> (usize, usize) {
        (self.width.div_ceil(1 << i), self.height.div_ceil(1 << i))
    }
    pub fn indices(&self, x: usize, y: usize) -> [usize; 3] {
        std::array::from_fn(|i| (y >> i) * self.width.div_ceil(1 << i) + (x >> i))
    }
    pub fn rgb(&self, z: [f32; 3]) -> [f32; 3] {
        self.synthesis
            .map(|w| ((w[0] * z[0] + w[1] * z[1]) + w[2] * z[2]) + w[3])
    }
    pub fn quantize(&mut self) {
        for w in &mut self.synthesis {
            for a in &mut w[..3] {
                *a = (*a * 4096.0).round_ties_even() / 4096.0;
            }
            w[3] = (w[3] * 65536.0).round_ties_even() / 65536.0;
        }
    }
    pub fn parameter_groups(&self) -> [Vec<i32>; 8] {
        let mut syn_w = vec![4096, 0, 0, 0, 4096, 0, 0, 0, 4096]; // Output transform identity first
        syn_w.extend(
            self.synthesis
                .iter()
                .flat_map(|w| w[..3].iter().map(|x| (x * 4096.0).round_ties_even() as i32)),
        );
        let mut syn_b = vec![0; 3];
        syn_b.extend(
            self.synthesis
                .iter()
                .map(|w| (w[3] * 65536.0).round_ties_even() as i32),
        );
        [
            self.arm_w.to_vec(),
            self.arm_b.to_vec(),
            vec![],
            vec![],
            vec![0, 4096, 0, 4096, 0, 0],
            vec![0; 4],
            syn_w,
            syn_b,
        ]
    }
    pub fn from_groups(width: usize, height: usize, g: [Vec<i32>; 8]) -> Result<Self> {
        if g[4] != [0, 4096, 0, 4096, 0, 0]
            || g[5] != [0; 4]
            || g[6][..9] != [4096, 0, 0, 0, 4096, 0, 0, 0, 4096]
            || g[7][..3] != [0; 3]
        {
            return Err(Error(
                "Unsupported upsampling or output transform in Cool-chic profile",
            ));
        }
        Ok(Self {
            width,
            height,
            synthesis: std::array::from_fn(|c| {
                [
                    g[6][9 + c * 3] as f32 / 4096.0,
                    g[6][10 + c * 3] as f32 / 4096.0,
                    g[6][11 + c * 3] as f32 / 4096.0,
                    g[7][3 + c] as f32 / 65536.0,
                ]
            }),
            arm_w: g[0]
                .as_slice()
                .try_into()
                .map_err(|_| Error("Invalid ARM weights"))?,
            arm_b: g[1]
                .as_slice()
                .try_into()
                .map_err(|_| Error("Invalid ARM biases"))?,
        })
    }
    pub fn probability(&self, context: [i32; 6]) -> (f64, f64) {
        // Reference fixed_point_arm: x<<16; w=q_w<<8; b=q_b<<16;
        // last log-scale bias subtracts four; output shifts by 24.
        let out: [i64; 2] = std::array::from_fn(|c| {
            let b = i64::from(self.arm_b[c]) - if c == 1 { 4 * 65536 } else { 0 };
            let sum = context.iter().enumerate().fold(b * 65536, |sum, (j, &x)| {
                sum + i64::from(x) * 65536 * i64::from(self.arm_w[c * 6 + j]) * 256
            });
            sum >> 24
        });
        let mu = out[0].clamp(-16384, 16383) as f64 / 256.0;
        let log_index = (out[1] + 1280).clamp(0, 2560) as usize;
        let table = include_bytes!("log_scales.f32le");
        let p = log_index * 4;
        let scale = f32::from_le_bytes(table[p..p + 4].try_into().unwrap());
        (mu, f64::from(scale))
    }
    pub fn render(&self, grids: &[Vec<i32>; 3]) -> Vec<u8> {
        let mut rgba = Vec::with_capacity(self.width * self.height * 4);
        for y in 0..self.height {
            for x in 0..self.width {
                let idx = self.indices(x, y);
                let rgb = self.rgb(std::array::from_fn(|i| grids[i][idx[i]] as f32));
                rgba.extend(rgb.map(|v| (v * 255.0).round_ties_even().clamp(0.0, 255.0) as u8));
                rgba.push(255);
            }
        }
        rgba
    }
}

pub(crate) fn context(grid: &[i32], width: usize, height: usize, x: usize, y: usize) -> [i32; 6] {
    CONTEXT.map(|(dy, dx)| {
        let xx = x as isize + dx;
        let yy = y as isize + dy;
        if xx < 0 || yy < 0 || xx >= width as isize || yy >= height as isize {
            0
        } else {
            grid[yy as usize * width + xx as usize]
        }
    })
}

// Small positive-definite normal systems. Ridge is supplied by the caller;
// pivoted elimination keeps flat/constant images from making the fit singular.
pub(crate) fn solve<const N: usize>(mut a: [[f64; N]; N], mut b: [f64; N]) -> Option<[f64; N]> {
    for col in 0..N {
        let pivot = (col..N).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))?;
        if a[pivot][col].abs() < 1e-12 {
            return None;
        }
        a.swap(col, pivot);
        b.swap(col, pivot);
        let scale = a[col][col];
        for value in &mut a[col][col..] {
            *value /= scale;
        }
        b[col] /= scale;
        let pivot_row = a[col];
        for (i, row) in a.iter_mut().enumerate() {
            if i != col {
                let s = row[col];
                for (value, pivot) in row[col..].iter_mut().zip(&pivot_row[col..]) {
                    *value -= s * pivot;
                }
                b[i] -= s * b[col];
            }
        }
    }
    b.iter().all(|v| v.is_finite()).then_some(b)
}
