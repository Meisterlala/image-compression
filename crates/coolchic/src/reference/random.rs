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
    /// Preserve the scalar RNG stream, but evaluate independent Box-Muller
    /// pairs in parallel. The hot transcendental work no longer serializes
    /// the entire training worker. Odd counts keep the unscaled spare sample.
    pub fn normal_values(&mut self, count: usize) -> Vec<f32> {
        if count == 0 {
            return Vec::new();
        }
        let spare = self.spare.take();
        let offset = usize::from(spare.is_some());
        let extra = (count - offset) % 2;
        let mut values = vec![0.0; count + extra];
        if let Some(value) = spare {
            values[0] = value;
        }
        for pair in values[offset..].chunks_exact_mut(2) {
            pair[0] = self.uniform();
            pair[1] = self.uniform();
        }
        let transform = |pair: &mut [f32]| {
            let radius = (-2.0 * pair[0].ln()).sqrt();
            let angle = std::f32::consts::TAU * pair[1];
            pair[0] = radius * angle.cos();
            pair[1] = radius * angle.sin();
        };
        #[cfg(feature = "parallel")]
        let parallel = super::parallel::worthwhile(count * 32);
        #[cfg(not(feature = "parallel"))]
        let parallel = false;
        if parallel {
            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;
                values[offset..].par_chunks_mut(2048).for_each(|chunk| {
                    for pair in chunk.chunks_exact_mut(2) {
                        transform(pair);
                    }
                });
            }
        } else {
            for pair in values[offset..].chunks_exact_mut(2) {
                transform(pair);
            }
        }
        if extra != 0 {
            self.spare = values.pop();
        }
        values
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

#[cfg(test)]
mod tests {
    use super::Random;
    #[test]
    fn batched_normals_preserve_the_exact_scalar_stream_and_spare() {
        let mut scalar = Random::new(123);
        let mut batched = scalar.clone();
        for count in [0, 1, 1, 3, 32769, 32768, 2, 1, 0, 17] {
            let expected: Vec<_> = (0..count).map(|_| scalar.normal().to_bits()).collect();
            let actual: Vec<_> = batched
                .normal_values(count)
                .iter()
                .map(|v| v.to_bits())
                .collect();
            assert_eq!(actual, expected, "batch of {count}");
            assert_eq!(batched.state, scalar.state);
            assert_eq!(
                batched.spare.map(f32::to_bits),
                scalar.spare.map(f32::to_bits)
            );
        }
    }
}
