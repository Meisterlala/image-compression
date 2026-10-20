//! Integer ARM/IFCE used by actual reference bitstreams. Q16 activations,
//! Q32 accumulators, Q8 probability/context outputs. Never substitute the
//! floating-point training network for this path: range-coder drift corrupts files.
use super::{Conv2d, arm::Arm};
use crate::{Error, Result};

#[derive(Clone, Debug)]
struct Layer {
    inputs: usize,
    weights: Vec<i64>,
    bias: Vec<i64>,
}
#[derive(Clone, Debug)]
pub struct FixedArm {
    layers: Vec<Layer>,
    stabiliser: Option<Layer>,
}

fn shifted(value: f32, q_shift: i32, target: i32) -> Result<i64> {
    let integer = (value / 2.0_f32.powi(q_shift)).round_ties_even();
    if !integer.is_finite() || integer.abs() > (1 << 20) as f32 {
        return Err(Error("Invalid quantized reference ARM parameter"));
    }
    let result = (f64::from(integer) * 2.0_f64.powi(q_shift + target)).round_ties_even();
    if result < i64::MIN as f64 || result >= i64::MAX as f64 {
        return Err(Error("Reference ARM parameter overflow"));
    }
    Ok(result as i64)
}

fn layer(
    source: &Conv2d,
    q_weight: i32,
    q_bias: i32,
    inter_features: usize,
    residual: bool,
    subtract_scale: bool,
) -> Result<Layer> {
    let mut weights = Vec::with_capacity(source.weight.len());
    let spatial = source
        .input_channels
        .checked_sub(inter_features)
        .ok_or(Error("Invalid integer ARM inter-feature count"))?;
    for output in 0..source.output_channels {
        for input in 0..source.input_channels {
            let target = 16 - if input >= spatial { 8 } else { 0 };
            let mut value = shifted(
                source.weight[output * source.input_channels + input],
                q_weight,
                target,
            )?;
            if residual && input == output {
                value += 1 << target;
            }
            weights.push(value);
        }
    }
    let mut bias = source
        .bias
        .iter()
        .map(|&b| shifted(b, q_bias, 32))
        .collect::<Result<Vec<_>>>()?;
    if subtract_scale {
        if bias.len() != 2 {
            return Err(Error("Reference probability ARM requires two outputs"));
        }
        bias[1] -= 4_i64 << 32;
    }
    Ok(Layer {
        inputs: source.input_channels,
        weights,
        bias,
    })
}

impl Layer {
    fn forward(&self, input: &[i64]) -> Result<Vec<i64>> {
        if input.len() != self.inputs {
            return Err(Error("Invalid reference integer ARM input dimensions"));
        }
        let mut output = self.bias.clone();
        for (row, value) in output.iter_mut().enumerate() {
            for (column, &input) in input.iter().enumerate() {
                let product = input
                    .checked_mul(self.weights[row * self.inputs + column])
                    .ok_or(Error("Cool-chic integer ARM accumulator overflow"))?;
                *value = value
                    .checked_add(product)
                    .ok_or(Error("Cool-chic integer ARM accumulator overflow"))?;
            }
        }
        Ok(output)
    }
}

impl FixedArm {
    pub fn new(
        network: &Arm,
        q_weight: i32,
        q_bias: i32,
        inter_features: usize,
        probability: bool,
    ) -> Result<Self> {
        if network.layers.is_empty() {
            return Err(Error("Empty reference integer ARM"));
        }
        let mut layers = Vec::with_capacity(network.layers.len());
        for (index, source) in network.layers.iter().enumerate() {
            let last = index == network.layers.len() - 1;
            // Upstream folds residuals whenever dimensions match, rather than
            // consulting the floating layer's residual flag.
            layers.push(layer(
                source,
                q_weight,
                q_bias,
                if index == 0 { inter_features } else { 0 },
                source.input_channels == source.output_channels,
                last && probability,
            )?);
        }
        let stabiliser = network
            .stabiliser
            .as_ref()
            .map(|source| layer(source, q_weight, q_bias, inter_features, false, false))
            .transpose()?;
        Ok(Self { layers, stabiliser })
    }

    pub fn forward(&self, context: &[i64]) -> Result<Vec<i64>> {
        let input = context
            .iter()
            .map(|&v| {
                v.checked_mul(1 << 16)
                    .ok_or(Error("Reference integer context overflow"))
            })
            .collect::<Result<Vec<_>>>()?;
        let stabiliser = self
            .stabiliser
            .as_ref()
            .map(|layer| layer.forward(&input))
            .transpose()?;
        let mut current = input;
        for (index, layer) in self.layers.iter().enumerate() {
            current = layer.forward(&current)?;
            if index + 1 < self.layers.len() {
                for value in &mut current {
                    *value = (*value).max(0) >> 16;
                }
            }
        }
        if let Some(stabiliser) = stabiliser {
            for (value, extra) in current.iter_mut().zip(stabiliser) {
                *value = value
                    .checked_add(extra)
                    .ok_or(Error("Reference integer stabiliser overflow"))?;
            }
        }
        for value in &mut current {
            *value >>= 24;
        }
        Ok(current)
    }

    pub fn probability(&self, context: &[i64]) -> Result<(f64, f64)> {
        let output = self.forward(context)?;
        if output.len() != 2 {
            return Err(Error("Invalid integer probability ARM output"));
        }
        let mu = output[0].clamp(-16384, 16383) as f64 / 256.0;
        let index = (output[1] + 1280).clamp(0, 2560) as usize;
        let table = include_bytes!("../log_scales.f32le");
        let scale = f32::from_le_bytes(table[index * 4..index * 4 + 4].try_into().unwrap());
        Ok((mu, f64::from(scale)))
    }
}
