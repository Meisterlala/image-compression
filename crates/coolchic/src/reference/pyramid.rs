//! Reference Upsampling.forward over the entire latent pyramid, including
//! learned pre-concatenation, kernel reuse, odd-size crops and reverse mode.
use super::{Filter, FilterMode, Tensor};
use crate::{Error, Result};

#[derive(Clone, Debug)]
pub struct Pyramid {
    pub upsampling: Vec<Filter>,
    pub preconcat: Vec<Filter>,
}

pub struct PyramidGradient {
    /// In original highest-to-lowest resolution order.
    pub latents: Vec<Tensor>,
    pub upsampling: Vec<Vec<f32>>,
    pub preconcat: Vec<Vec<f32>>,
}

impl Pyramid {
    fn states(&self, latents: &[Tensor]) -> Result<Vec<Tensor>> {
        if latents.is_empty()
            || (latents.len() > 1 && (self.upsampling.is_empty() || self.preconcat.is_empty()))
            || self
                .upsampling
                .iter()
                .any(|f| f.mode != FilterMode::Upsample)
            || self
                .preconcat
                .iter()
                .any(|f| f.mode != FilterMode::Residual)
        {
            return Err(Error("Invalid reference latent pyramid"));
        }
        let mut states = vec![latents.last().unwrap().clone()];
        for (index, target) in latents.iter().rev().skip(1).enumerate() {
            if target.channels == 0 {
                break;
            }
            let up =
                self.upsampling[index % self.upsampling.len()].forward(states.last().unwrap())?;
            let up = up.crop(target.height, target.width)?;
            let high = self.preconcat[index % self.preconcat.len()].forward(target)?;
            states.push(high.concat(&up)?);
        }
        Ok(states)
    }

    pub fn forward(&self, latents: &[Tensor]) -> Result<Tensor> {
        Ok(self.states(latents)?.pop().unwrap())
    }

    pub fn backward(&self, latents: &[Tensor], gradient: &Tensor) -> Result<PyramidGradient> {
        let states = self.states(latents)?;
        if !gradient.same_shape(states.last().unwrap()) {
            return Err(Error("Invalid reference pyramid gradient shape"));
        }
        let mut result = PyramidGradient {
            latents: latents
                .iter()
                .map(|l| Tensor::zeros(l.channels, l.height, l.width))
                .collect::<Result<_>>()?,
            upsampling: self
                .upsampling
                .iter()
                .map(|f| vec![0.0; f.half.len()])
                .collect(),
            preconcat: self
                .preconcat
                .iter()
                .map(|f| vec![0.0; f.half.len()])
                .collect(),
        };
        let mut current = gradient.clone();
        for index in (0..states.len() - 1).rev() {
            let latent_index = latents.len() - 2 - index;
            let target = &latents[latent_index];
            let up_index = index % self.upsampling.len();
            let pre_index = index % self.preconcat.len();
            let high_gradient = current.first_channels(target.channels)?;
            let high = self.preconcat[pre_index].backward(target, &high_gradient)?;
            result.latents[latent_index] = high.input;
            for (a, b) in result.preconcat[pre_index].iter_mut().zip(high.half) {
                *a += b;
            }

            let low = &states[index];
            let mut expanded = Tensor::zeros(low.channels, low.height * 2, low.width * 2)?;
            // Reverse the top-left crop with zero gradients for cropped borders.
            for c in 0..low.channels {
                for y in 0..target.height {
                    for x in 0..target.width {
                        let dst = expanded.index(c, y, x);
                        expanded.data[dst] = current.data[current.index(c + target.channels, y, x)];
                    }
                }
            }
            let up = self.upsampling[up_index].backward(low, &expanded)?;
            current = up.input;
            for (a, b) in result.upsampling[up_index].iter_mut().zip(up.half) {
                *a += b;
            }
        }
        *result.latents.last_mut().unwrap() = current;
        Ok(result)
    }
}
