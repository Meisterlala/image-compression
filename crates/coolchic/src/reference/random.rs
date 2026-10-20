//! Training randomness is local/reproducible, not a server dependency. The
//! separate common-noise generator is bitstream-defined and matches upstream.
use super::Tensor;
use crate::Result;

#[derive(Clone, Debug)]
pub struct Random {
    state: u64,
    spare: Option<f32>,
}
impl Random {
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed.max(1),
            spare: None,
        }
    }
    pub fn uniform(&mut self) -> f32 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        ((x.wrapping_mul(2685821657736338717) >> 40) as f32 + 0.5) / 16777216.0
    }
    pub fn normal(&mut self) -> f32 {
        if let Some(value) = self.spare.take() {
            return value;
        }
        let radius = (-2.0 * self.uniform().ln()).sqrt();
        let angle = std::f32::consts::TAU * self.uniform();
        self.spare = Some(radius * angle.sin());
        radius * angle.cos()
    }
}

pub fn common_noise(shapes: &[(usize, usize)]) -> Result<Vec<Tensor>> {
    let mut seed = 18101995_u64;
    let mut tensors = Vec::with_capacity(shapes.len());
    for &(height, width) in shapes {
        let mut tensor = Tensor::zeros(1, height, width)?;
        for value in &mut tensor.data {
            seed = (16807 * seed) % 2147483647;
            let u1 = seed as f64 / 2147483647.0;
            seed = (16807 * seed) % 2147483647;
            let u2 = seed as f64 / 2147483647.0;
            // The format specifies this literal, not the full-precision pi.
            #[allow(clippy::approx_constant)]
            let angle = 2.0 * 3.14159265359 * u2;
            *value = ((-2.0 * u1.ln()).sqrt() * angle.cos()) as f32;
        }
        tensors.push(tensor);
    }
    Ok(tensors)
}
