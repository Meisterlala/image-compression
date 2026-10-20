//! Cross-resolution IFCE context graph from CoolChicEncoder.forward. Context
//! comes only from already-decoded lower-priority features, never the current
//! latent. The IFCE runs BEFORE the final nearest-neighbour interpolation.
use super::{
    Tensor,
    arm::{Arm, ArmGradient},
};
use crate::{Error, Result};

pub struct InterFeature {
    /// One optional network per latent, in highest-to-lowest resolution order.
    pub networks: Vec<Option<Arm>>,
    pub output_channels: usize,
}

pub struct InterFeatureGradient {
    pub latents: Vec<Tensor>,
    pub networks: Vec<Option<ArmGradient>>,
}

fn expand(input: &Tensor, height: usize, width: usize, allow_identity: bool) -> Result<Tensor> {
    if allow_identity && (input.height, input.width) == (height, width) {
        return Ok(input.clone());
    }
    if height
        > input
            .height
            .checked_mul(2)
            .ok_or(Error("Reference IFCE dimensions overflow"))?
        || width
            > input
                .width
                .checked_mul(2)
                .ok_or(Error("Reference IFCE dimensions overflow"))?
    {
        return Err(Error("Invalid reference IFCE hierarchy"));
    }
    let mut output = Tensor::zeros(input.channels, height, width)?;
    for c in 0..input.channels {
        for y in 0..height {
            for x in 0..width {
                let index = output.index(c, y, x);
                output.data[index] = input.data[input.index(c, y / 2, x / 2)];
            }
        }
    }
    Ok(output)
}

fn expand_backward(input: &Tensor, gradient: &Tensor, allow_identity: bool) -> Result<Tensor> {
    if allow_identity && input.same_shape(gradient) {
        return Ok(gradient.clone());
    }
    if input.channels != gradient.channels
        || gradient.height > input.height * 2
        || gradient.width > input.width * 2
    {
        return Err(Error("Invalid reference IFCE resize gradient"));
    }
    let mut output = Tensor::zeros(input.channels, input.height, input.width)?;
    for c in 0..gradient.channels {
        for y in 0..gradient.height {
            for x in 0..gradient.width {
                let index = output.index(c, y / 2, x / 2);
                output.data[index] += gradient.data[gradient.index(c, y, x)];
            }
        }
    }
    Ok(output)
}

// States of fixed_upsampling on the suffix AFTER the requested latent. Its
// final result is one resolution lower than the current latent. Empty suffix
// uses the reference's dummy all-zero single-channel feature.
fn coarse_states(latents: &[Tensor], index: usize) -> Result<Vec<Tensor>> {
    if index >= latents.len() || latents.iter().any(|l| l.channels != 1) {
        return Err(Error("Invalid reference IFCE latent list"));
    }
    if index == latents.len() - 1 {
        let last = &latents[index];
        return Ok(vec![Tensor::zeros(last.channels, last.height, last.width)?]);
    }
    let mut states = vec![latents.last().unwrap().clone()];
    for target in latents[index + 1..].iter().rev().skip(1) {
        let up = expand(states.last().unwrap(), target.height, target.width, true)?;
        states.push(target.concat(&up)?);
    }
    Ok(states)
}

impl InterFeature {
    pub fn forward(&self, latents: &[Tensor]) -> Result<Vec<Tensor>> {
        if self.networks.len() != latents.len() || latents.is_empty() {
            return Err(Error("Invalid reference IFCE network list"));
        }
        let mut outputs = Vec::with_capacity(latents.len());
        for (index, (latent, network)) in latents.iter().zip(&self.networks).enumerate() {
            let output = if let Some(network) = network {
                let states = coarse_states(latents, index)?;
                let transformed = network.forward(states.last().unwrap())?;
                if transformed.channels != self.output_channels {
                    return Err(Error("Invalid IFCE output channels"));
                }
                // Always x2 then crop, even if neighbouring latent grids have
                // identical shapes (e.g. small images or hyperlatent grids).
                expand(&transformed, latent.height, latent.width, false)?
            } else {
                Tensor::zeros(self.output_channels, latent.height, latent.width)?
            };
            outputs.push(output);
        }
        Ok(outputs)
    }

    pub fn backward(
        &self,
        latents: &[Tensor],
        gradients: &[Tensor],
    ) -> Result<InterFeatureGradient> {
        if self.networks.len() != latents.len()
            || gradients.len() != latents.len()
            || latents.is_empty()
        {
            return Err(Error("Invalid reference IFCE gradient list"));
        }
        let mut output = InterFeatureGradient {
            latents: latents
                .iter()
                .map(|l| Tensor::zeros(l.channels, l.height, l.width))
                .collect::<Result<_>>()?,
            networks: Vec::with_capacity(latents.len()),
        };
        for (index, network) in self.networks.iter().enumerate() {
            if let Some(network) = network {
                let states = coarse_states(latents, index)?;
                let coarse = states.last().unwrap();
                let transformed = network.forward(coarse)?;
                let gradient = &gradients[index];
                let target = &latents[index];
                if (gradient.channels, gradient.height, gradient.width)
                    != (self.output_channels, target.height, target.width)
                {
                    return Err(Error("Invalid reference IFCE output gradient"));
                }
                let gradient = expand_backward(&transformed, gradient, false)?;
                let network_gradient = network.backward(coarse, &gradient)?;
                let mut current = network_gradient.input.clone();
                if index + 1 < latents.len() {
                    for stage in (0..states.len() - 1).rev() {
                        let target_index = latents.len() - 2 - stage;
                        let high_gradient = current.first_channels(1)?;
                        output.latents[target_index].add_assign(&high_gradient)?;
                        let n = current.height * current.width;
                        let lower_gradient = Tensor::new(
                            current.channels - 1,
                            current.height,
                            current.width,
                            current.data[n..].to_vec(),
                        )?;
                        current = expand_backward(&states[stage], &lower_gradient, true)?;
                    }
                    output.latents.last_mut().unwrap().add_assign(&current)?;
                }
                // The last level uses zeros; its context gradient must not
                // leak into that level's actual latent values.
                output.networks.push(Some(network_gradient));
            } else {
                output.networks.push(None);
            }
        }
        Ok(output)
    }
}
