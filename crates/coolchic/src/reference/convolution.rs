//! Replicate-padded synthesis convolutions, residual/ReLU trunk, linear
//! stabiliser and output transform from component/core/synthesis.py.
use super::Tensor;
use crate::{Error, Result};

#[derive(Clone, Debug)]
pub struct Conv2d {
    pub input_channels: usize,
    pub output_channels: usize,
    pub kernel: usize,
    /// PyTorch OIHW order.
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
    pub residual: bool,
    pub relu: bool,
}

#[derive(Debug)]
pub struct ConvGradient {
    pub input: Tensor,
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
}

impl Conv2d {
    fn validate(&self, input: &Tensor) -> Result<()> {
        let count = self
            .input_channels
            .checked_mul(self.output_channels)
            .and_then(|n| n.checked_mul(self.kernel))
            .and_then(|n| n.checked_mul(self.kernel));
        if self.kernel == 0
            || self.kernel.is_multiple_of(2)
            || self.output_channels == 0
            || input.channels != self.input_channels
            || count != Some(self.weight.len())
            || self.bias.len() != self.output_channels
            || (self.residual && self.input_channels != self.output_channels)
        {
            return Err(Error("Invalid reference synthesis convolution"));
        }
        Ok(())
    }

    fn weight_index(&self, out: usize, input: usize, y: usize, x: usize) -> usize {
        ((out * self.input_channels + input) * self.kernel + y) * self.kernel + x
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.validate(input)?;
        let mut output = Tensor::zeros(self.output_channels, input.height, input.width)?;
        if super::acceleration::convolution(self, input, None, &mut output.data) {
            return Ok(output);
        }
        #[cfg(feature = "parallel")]
        if super::parallel::worthwhile(
            input.data.len() * self.output_channels * self.kernel * self.kernel,
        ) {
            return super::parallel::forward(self, input);
        }
        // CHW pointwise GEMM: stream contiguous planes so LLVM can SIMD-
        // vectorize in WASM, retaining the channel accumulation order.
        if self.kernel == 1 {
            let pixels = input.height * input.width;
            for out in 0..self.output_channels {
                let plane = &mut output.data[out * pixels..(out + 1) * pixels];
                for channel in 0..input.channels {
                    let weight = self.weight[out * input.channels + channel];
                    let source = &input.data[channel * pixels..(channel + 1) * pixels];
                    for (value, &sample) in plane.iter_mut().zip(source) {
                        *value += sample * weight;
                    }
                }
                for (i, value) in plane.iter_mut().enumerate() {
                    *value += self.bias[out];
                    if self.residual {
                        *value += input.data[out * pixels + i];
                    }
                    if self.relu {
                        *value = value.max(0.0);
                    }
                }
            }
            return Ok(output);
        }
        let pad = self.kernel / 2;
        for out in 0..self.output_channels {
            for y in 0..input.height {
                for x in 0..input.width {
                    let mut value = 0.0;
                    for channel in 0..input.channels {
                        for ky in 0..self.kernel {
                            let iy = (y + ky).saturating_sub(pad).min(input.height - 1);
                            for kx in 0..self.kernel {
                                let ix = (x + kx).saturating_sub(pad).min(input.width - 1);
                                value += input.data[input.index(channel, iy, ix)]
                                    * self.weight[self.weight_index(out, channel, ky, kx)];
                            }
                        }
                    }
                    value += self.bias[out];
                    if self.residual {
                        value += input.data[input.index(out, y, x)];
                    }
                    if self.relu {
                        value = value.max(0.0);
                    }
                    let index = output.index(out, y, x);
                    output.data[index] = value;
                }
            }
        }
        Ok(output)
    }

    /// Exact reverse-mode derivative, including repeated border samples caused
    /// by replicate padding. No finite-difference approximations in training.
    pub fn backward(&self, input: &Tensor, gradient: &Tensor) -> Result<ConvGradient> {
        self.validate(input)?;
        if (gradient.channels, gradient.height, gradient.width)
            != (self.output_channels, input.height, input.width)
        {
            return Err(Error("Invalid reference convolution gradient shape"));
        }
        if super::acceleration::eligible(self, input, true) {
            let mut accelerated = vec![0.0; input.data.len() + self.weight.len() + self.bias.len()];
            if super::acceleration::convolution(self, input, Some(gradient), &mut accelerated) {
                let bias = accelerated.split_off(input.data.len() + self.weight.len());
                let weight = accelerated.split_off(input.data.len());
                return Ok(ConvGradient {
                    input: Tensor::new(input.channels, input.height, input.width, accelerated)?,
                    weight,
                    bias,
                });
            }
        }
        #[cfg(feature = "parallel")]
        if super::parallel::worthwhile(
            input.data.len() * self.output_channels * self.kernel * self.kernel,
        ) {
            return super::parallel::backward(self, input, gradient);
        }
        let activated = if self.relu {
            Some(self.forward(input)?)
        } else {
            None
        };
        let mut result = ConvGradient {
            input: Tensor::zeros(input.channels, input.height, input.width)?,
            weight: vec![0.0; self.weight.len()],
            bias: vec![0.0; self.bias.len()],
        };
        if self.kernel == 1 {
            let pixels = input.height * input.width;
            let mut gated = vec![0.0; pixels];
            for out in 0..self.output_channels {
                for (i, g) in gated.iter_mut().enumerate() {
                    let oi = out * pixels + i;
                    *g = if activated.as_ref().is_some_and(|a| a.data[oi] <= 0.0) {
                        0.0
                    } else {
                        gradient.data[oi]
                    };
                    result.bias[out] += *g;
                }
                if self.residual {
                    for (value, &g) in result.input.data[out * pixels..(out + 1) * pixels]
                        .iter_mut()
                        .zip(&gated)
                    {
                        *value += g;
                    }
                }
                for channel in 0..input.channels {
                    let wi = out * input.channels + channel;
                    let source = &input.data[channel * pixels..(channel + 1) * pixels];
                    let target = &mut result.input.data[channel * pixels..(channel + 1) * pixels];
                    let mut dw = 0.0;
                    for i in 0..pixels {
                        target[i] += gated[i] * self.weight[wi];
                        dw += gated[i] * source[i];
                    }
                    result.weight[wi] = dw;
                }
            }
            return Ok(result);
        }
        let pad = self.kernel / 2;
        for out in 0..self.output_channels {
            for y in 0..input.height {
                for x in 0..input.width {
                    let oi = gradient.index(out, y, x);
                    let g = if activated.as_ref().is_some_and(|a| a.data[oi] <= 0.0) {
                        0.0
                    } else {
                        gradient.data[oi]
                    };
                    result.bias[out] += g;
                    if self.residual {
                        let index = result.input.index(out, y, x);
                        result.input.data[index] += g;
                    }
                    for channel in 0..input.channels {
                        for ky in 0..self.kernel {
                            let iy = (y + ky).saturating_sub(pad).min(input.height - 1);
                            for kx in 0..self.kernel {
                                let ix = (x + kx).saturating_sub(pad).min(input.width - 1);
                                let ii = input.index(channel, iy, ix);
                                let wi = self.weight_index(out, channel, ky, kx);
                                result.input.data[ii] += g * self.weight[wi];
                                result.weight[wi] += g * input.data[ii];
                            }
                        }
                    }
                }
            }
        }
        Ok(result)
    }
}

#[derive(Clone, Debug)]
pub struct Synthesis {
    pub trunk: Vec<Conv2d>,
    /// When common randomness is enabled the stabiliser takes only the first
    /// half of the input channels; its input_channels encodes that distinction.
    pub stabiliser: Option<Conv2d>,
    pub output_transform: Conv2d,
}

#[derive(Debug)]
pub struct SynthesisGradient {
    pub input: Tensor,
    /// In forward order, ready for per-parameter optimizer state.
    pub trunk: Vec<ConvGradient>,
    pub stabiliser: Option<ConvGradient>,
    pub output_transform: ConvGradient,
}

impl Synthesis {
    fn intermediates(&self, input: &Tensor) -> Result<Vec<Tensor>> {
        if self.trunk.is_empty() {
            return Err(Error("Empty reference synthesis trunk"));
        }
        let mut states = vec![input.clone()];
        for layer in &self.trunk {
            states.push(layer.forward(states.last().unwrap())?);
        }
        Ok(states)
    }

    fn merged(&self, input: &Tensor, trunk: &Tensor) -> Result<Tensor> {
        let mut merged = trunk.clone();
        if let Some(stabiliser) = &self.stabiliser {
            let branch = stabiliser.forward(&input.first_channels(stabiliser.input_channels)?)?;
            merged.add_assign(&branch)?;
        }
        Ok(merged)
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let states = self.intermediates(input)?;
        self.output_transform
            .forward(&self.merged(input, states.last().unwrap())?)
    }

    pub fn backward(&self, input: &Tensor, gradient: &Tensor) -> Result<SynthesisGradient> {
        let states = self.intermediates(input)?;
        let merged = self.merged(input, states.last().unwrap())?;
        let output_transform = self.output_transform.backward(&merged, gradient)?;
        let mut current = output_transform.input.clone();
        let mut trunk = Vec::with_capacity(self.trunk.len());
        for (layer, state) in self.trunk.iter().zip(&states).rev() {
            let g = layer.backward(state, &current)?;
            current = g.input.clone();
            trunk.push(g);
        }
        trunk.reverse();
        let stabiliser = if let Some(layer) = &self.stabiliser {
            let g = layer.backward(
                &input.first_channels(layer.input_channels)?,
                &output_transform.input,
            )?;
            // Stabiliser may omit common-randomness channels.
            for (a, b) in current.data.iter_mut().zip(&g.input.data) {
                *a += b;
            }
            Some(g)
        } else {
            None
        };
        Ok(SynthesisGradient {
            input: current,
            trunk,
            stabiliser,
            output_transform,
        })
    }
}
