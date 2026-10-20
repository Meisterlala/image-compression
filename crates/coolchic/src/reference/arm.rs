//! Trainable reference residual MLP / IFCE and causal spatial context extraction.
//! Floating-point training path only: the bitstream's integer ARM is separate.
use super::{Conv2d, ConvGradient, Tensor};
use crate::{Error, Result};

#[derive(Clone, Debug)]
pub struct Arm {
    /// Residual/ReLU hidden layers followed by a non-residual linear output.
    /// All layers are 1x1 convolutions (a shared pointwise affine transform).
    pub layers: Vec<Conv2d>,
    pub stabiliser: Option<Conv2d>,
}

#[derive(Debug)]
pub struct ArmGradient {
    pub input: Tensor,
    pub layers: Vec<ConvGradient>,
    pub stabiliser: Option<ConvGradient>,
}

impl Arm {
    fn states(&self, input: &Tensor) -> Result<Vec<Tensor>> {
        if self.layers.is_empty()
            || self.layers.iter().any(|layer| layer.kernel != 1)
            || self
                .stabiliser
                .as_ref()
                .is_some_and(|layer| layer.kernel != 1)
        {
            return Err(Error("Invalid reference ARM layers"));
        }
        let mut states = vec![input.clone()];
        for layer in &self.layers {
            states.push(layer.forward(states.last().unwrap())?);
        }
        Ok(states)
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        Ok(self.forward_cached(input)?.0)
    }

    pub(super) fn forward_cached(&self, input: &Tensor) -> Result<(Tensor, Vec<Tensor>)> {
        let states = self.states(input)?;
        let mut output = states.last().unwrap().clone();
        if let Some(stabiliser) = &self.stabiliser {
            output.add_assign(&stabiliser.forward(input)?)?;
        }
        Ok((output, states))
    }

    pub fn backward(&self, input: &Tensor, gradient: &Tensor) -> Result<ArmGradient> {
        let states = self.states(input)?;
        self.backward_cached(input, gradient, &states)
    }

    pub(super) fn backward_cached(
        &self,
        input: &Tensor,
        gradient: &Tensor,
        states: &[Tensor],
    ) -> Result<ArmGradient> {
        if self.layers.is_empty()
            || !input.valid_layout()
            || states.len() != self.layers.len() + 1
            || states.iter().any(|state| !state.valid_layout())
            || !states[0].same_shape(input)
            || self.layers.iter().zip(&states[1..]).any(|(layer, state)| {
                (state.channels, state.height, state.width)
                    != (layer.output_channels, input.height, input.width)
            })
        {
            return Err(Error("Invalid ARM activation tape"));
        }
        let mut current = gradient.clone();
        let mut layers = Vec::with_capacity(self.layers.len());
        for (index, layer) in self.layers.iter().enumerate().rev() {
            let g = layer.backward_cached(&states[index], &current, Some(&states[index + 1]))?;
            current = g.input.clone();
            layers.push(g);
        }
        layers.reverse();
        let stabiliser = if let Some(layer) = &self.stabiliser {
            let g = layer.backward(input, gradient)?;
            current.add_assign(&g.input)?;
            Some(g)
        } else {
            None
        };
        Ok(ArmGradient {
            input: current,
            layers,
            stabiliser,
        })
    }
}

// The position in this array is the row-major index in the top 40 elements
// of a 9x9 mask; the value is its priority. Do NOT sort into raster order.
const PRIORITIES: [usize; 40] = [
    38, 35, 30, 25, 23, 31, 36, 37, 39, 33, 28, 21, 20, 6, 15, 22, 29, 34, 32, 18, 12, 10, 5, 9,
    14, 19, 27, 24, 13, 8, 2, 1, 3, 11, 17, 26, 16, 7, 4, 0,
];

pub(crate) fn neighbours(count: usize) -> Result<Vec<(isize, isize)>> {
    if count > PRIORITIES.len() {
        return Err(Error("Invalid reference spatial context count"));
    }
    let mut positions: Vec<usize> = (0..PRIORITIES.len()).collect();
    positions.sort_by_key(|&index| PRIORITIES[index]);
    Ok(positions[..count]
        .iter()
        .map(|&index| (index as isize / 9 - 4, index as isize % 9 - 4))
        .collect())
}

/// Reference _get_neighbor, one channel per latent, with constant-zero borders.
pub fn spatial_context(input: &Tensor, count: usize) -> Result<Tensor> {
    if input.channels != 1 {
        return Err(Error("Reference context needs one latent channel"));
    }
    let positions = neighbours(count)?;
    let mut output = Tensor::zeros(count, input.height, input.width)?;
    for (channel, &(dy, dx)) in positions.iter().enumerate() {
        for y in 0..input.height {
            for x in 0..input.width {
                let iy = y as isize + dy;
                let ix = x as isize + dx;
                if iy >= 0 && ix >= 0 && iy < input.height as isize && ix < input.width as isize {
                    let index = output.index(channel, y, x);
                    output.data[index] = input.data[input.index(0, iy as usize, ix as usize)];
                }
            }
        }
    }
    Ok(output)
}

/// Scatter-add context gradients to their causal latent pixels, excluding pads.
pub fn spatial_context_backward(gradient: &Tensor) -> Result<Tensor> {
    let positions = neighbours(gradient.channels)?;
    let mut output = Tensor::zeros(1, gradient.height, gradient.width)?;
    for (channel, &(dy, dx)) in positions.iter().enumerate() {
        for y in 0..gradient.height {
            for x in 0..gradient.width {
                let iy = y as isize + dy;
                let ix = x as isize + dx;
                if iy >= 0
                    && ix >= 0
                    && iy < gradient.height as isize
                    && ix < gradient.width as isize
                {
                    let index = output.index(0, iy as usize, ix as usize);
                    output.data[index] += gradient.data[gradient.index(channel, y, x)];
                }
            }
        }
    }
    Ok(output)
}
