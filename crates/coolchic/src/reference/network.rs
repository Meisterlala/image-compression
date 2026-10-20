//! Complete differentiable RGB intra graph: quantization, entropy model,
//! hyperlatents, learned synthesis, common noise, interpolation and MSE+lambda R.
use super::{
    Filter, FilterMode, Tensor,
    architecture::Architecture,
    arm::{Arm, ArmGradient, spatial_context, spatial_context_backward},
    convolution::{Conv2d, ConvGradient, Synthesis, SynthesisGradient},
    inter_feature::{InterFeature, InterFeatureGradient},
    pyramid::{Pyramid, PyramidGradient},
    quantizer::{self, Quantizer},
    random::{Random, common_noise},
    rate::{Rate, laplace_rate},
    resize::{resize, resize_backward},
};
use crate::{Error, Result};

#[derive(Clone, Debug)]
pub struct Network {
    pub architecture: Architecture,
    pub latents: Vec<Tensor>,
    pub arm: Arm,
    pub ifce: Option<InterFeature>,
    pub pyramid: Pyramid,
    pub synthesis: Synthesis,
    /// Reference filter biases are sent but not used in convolution.
    pub up_bias: Vec<f32>,
    pub pre_bias: Vec<f32>,
    pub common: Vec<Tensor>,
}

pub struct Gradient {
    pub latents: Vec<Tensor>,
    pub arm: ArmGradient,
    pub ifce: Option<InterFeatureGradient>,
    pub pyramid: PyramidGradient,
    pub synthesis: SynthesisGradient,
}

pub struct Evaluation {
    pub image: Tensor,
    pub bits: Vec<Tensor>,
    pub mse: f32,
    pub rate_bpp: f32,
    pub loss: f32,
}

struct Reconstruction {
    dense: Tensor,
    synthesis_input: Tensor,
    native: Tensor,
    raw: Tensor,
    synthesis_states: Vec<Tensor>,
    pyramid_states: Vec<Tensor>,
}

struct Tape {
    arm_states: Vec<Vec<Tensor>>,
    synthesis_states: Vec<Tensor>,
    pyramid_states: Vec<Tensor>,
    quantized: Vec<Tensor>,
    derivatives: Vec<Vec<f32>>,
    contexts: Vec<Tensor>,
    raw_arms: Vec<Tensor>,
    rates: Vec<Rate>,
    dense: Tensor,
    synthesis_input: Tensor,
    native: Tensor,
    raw: Tensor,
    evaluation: Evaluation,
}

fn convolution(
    inputs: usize,
    outputs: usize,
    kernel: usize,
    residual: bool,
    relu: bool,
    rng: &mut Random,
) -> Conv2d {
    let n = inputs * outputs * kernel * kernel;
    let bound = 1.0 / ((inputs * kernel * kernel) as f32).sqrt() / (outputs * outputs) as f32;
    Conv2d {
        input_channels: inputs,
        output_channels: outputs,
        kernel,
        residual,
        relu,
        weight: (0..n)
            .map(|_| {
                if residual {
                    0.0
                } else {
                    (rng.uniform() - 0.5) * 2.0 * bound
                }
            })
            .collect(),
        bias: vec![0.0; outputs],
    }
}

fn linear(inputs: usize, outputs: usize, residual: bool, relu: bool, rng: &mut Random) -> Conv2d {
    Conv2d {
        input_channels: inputs,
        output_channels: outputs,
        kernel: 1,
        residual,
        relu,
        weight: (0..inputs * outputs)
            .map(|_| {
                if residual {
                    0.0
                } else {
                    rng.normal() / (outputs * outputs) as f32
                }
            })
            .collect(),
        bias: vec![0.0; outputs],
    }
}
fn arm(inputs: usize, outputs: usize, hidden: usize, stabiliser: bool, rng: &mut Random) -> Arm {
    let mut layers: Vec<_> = (0..hidden)
        .map(|_| linear(inputs, inputs, true, true, rng))
        .collect();
    layers.push(linear(inputs, outputs, false, false, rng));
    Arm {
        layers,
        stabiliser: stabiliser.then(|| linear(inputs, outputs, false, false, rng)),
    }
}

fn fixed_pyramid(latents: &[Tensor]) -> Result<Tensor> {
    let mut current = latents
        .last()
        .ok_or(Error("Empty common-noise pyramid"))?
        .clone();
    for target in latents.iter().rev().skip(1) {
        let up = if (current.height, current.width) == (target.height, target.width) {
            current.clone()
        } else {
            resize(
                &current,
                current.height * 2,
                current.width * 2,
                super::architecture::Interpolation::Bicubic,
            )?
            .crop(target.height, target.width)?
        };
        current = target.concat(&up)?;
    }
    Ok(current)
}

fn add_conv(target: &mut ConvGradient, source: &ConvGradient) {
    for (a, b) in target.weight.iter_mut().zip(&source.weight) {
        *a += b;
    }
    for (a, b) in target.bias.iter_mut().zip(&source.bias) {
        *a += b;
    }
}
fn add_arm(target: &mut ArmGradient, source: &ArmGradient) {
    for (a, b) in target.layers.iter_mut().zip(&source.layers) {
        add_conv(a, b);
    }
    if let (Some(a), Some(b)) = (&mut target.stabiliser, &source.stabiliser) {
        add_conv(a, b);
    }
}

impl Network {
    pub fn new(architecture: Architecture, rng: &mut Random) -> Result<Self> {
        architecture.validate()?;
        if architecture.estimated_working_bytes(false) > 3 * 1024 * 1024 * 1024 {
            return Err(Error(
                "Cool-chic reference reconstruction exceeds the 3 GiB browser memory budget. Choose a smaller Downscale image size.",
            ));
        }
        let specs = architecture.latent_specs();
        let latents = specs
            .iter()
            .map(|s| Tensor::zeros(1, s.height, s.width))
            .collect::<Result<Vec<_>>>()?;
        let arm = arm(
            architecture.spatial_context + architecture.ifce_outputs,
            2,
            architecture.arm_hidden,
            architecture.arm_stabiliser,
            rng,
        );
        let ifce = if architecture.ifce_outputs > 0 {
            Some(InterFeature {
                output_channels: architecture.ifce_outputs,
                networks: architecture
                    .ifce_inputs()
                    .into_iter()
                    .map(|n| {
                        (n > 0).then(|| {
                            crate::reference::network::arm(
                                n,
                                architecture.ifce_outputs,
                                0,
                                false,
                                rng,
                            )
                        })
                    })
                    .collect(),
            })
        } else {
            None
        };
        let n = architecture.latent_max;
        let mut upsampling = Vec::with_capacity(n);
        let mut preconcat = Vec::with_capacity(n);
        for _ in 0..n {
            let mut half = vec![0.0; architecture.up_kernel / 2];
            let core: &[f32] = if architecture.up_kernel < 8 {
                &[0.25, 0.75]
            } else {
                &[0.0351562, 0.1054687, -0.2617187, -0.8789063]
            };
            let start = half.len() - core.len();
            half[start..].copy_from_slice(core);
            upsampling.push(Filter {
                kernel: architecture.up_kernel,
                half,
                mode: FilterMode::Upsample,
            });
            let mut half = vec![0.0; architecture.pre_kernel.div_ceil(2)];
            *half.last_mut().unwrap() = 1.0;
            preconcat.push(Filter {
                kernel: architecture.pre_kernel,
                half,
                mode: FilterMode::Residual,
            });
        }
        let mut inputs = architecture.synthesis_inputs();
        let mut trunk = Vec::new();
        for layer in &architecture.synthesis {
            trunk.push(convolution(
                inputs,
                layer.outputs,
                layer.kernel,
                layer.residual,
                layer.relu,
                rng,
            ));
            inputs = layer.outputs;
        }
        let stabiliser = architecture.synthesis_stabiliser.then(|| {
            convolution(
                architecture.latent_max - architecture.latent_min + 1,
                3,
                1,
                false,
                false,
                rng,
            )
        });
        let mut output_transform = convolution(3, 3, 1, false, false, rng);
        output_transform.weight.fill(0.0);
        for c in 0..3 {
            output_transform.weight[c * 3 + c] = 1.0;
        }
        let synthesis = Synthesis {
            trunk,
            stabiliser,
            output_transform,
        };
        let shapes: Vec<_> = specs
            .iter()
            .filter(|s| !s.hyper)
            .map(|s| (s.height, s.width))
            .collect();
        let common = if architecture.common_randomness {
            common_noise(&shapes)?
        } else {
            Vec::new()
        };
        Ok(Self {
            architecture,
            latents,
            arm,
            ifce,
            pyramid: Pyramid {
                upsampling,
                preconcat,
            },
            synthesis,
            up_bias: vec![0.0; n],
            pre_bias: vec![0.0; n],
            common,
        })
    }

    pub fn initialize_output_transform(&mut self, target: &Tensor) -> Result<()> {
        if target.channels != 3 {
            return Err(Error("Reference training requires RGB pixels"));
        }
        let pixels = target.height * target.width;
        for c in 0..3 {
            let channel = &target.data[c * pixels..(c + 1) * pixels];
            let min = channel.iter().copied().fold(f32::INFINITY, f32::min);
            let max = channel.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            self.synthesis.output_transform.weight[c * 3 + c] = max - min;
            self.synthesis.output_transform.bias[c] = min;
        }
        Ok(())
    }

    fn reconstruction(&self, quantized: &[Tensor], capture: bool) -> Result<Reconstruction> {
        let arch = &self.architecture;
        let specs = arch.latent_specs();
        if quantized.len() != specs.len()
            || quantized
                .iter()
                .zip(&specs)
                .any(|(q, s)| q.channels != 1 || q.height != s.height || q.width != s.width)
        {
            return Err(Error("Invalid reference reconstruction latent dimensions"));
        }
        let synthesis_latents: Vec<_> = quantized
            .iter()
            .zip(&specs)
            .filter(|(_, s)| !s.hyper)
            .map(|(q, _)| q.clone())
            .collect();
        let (dense, pyramid_states) = if capture {
            self.pyramid.forward_cached(&synthesis_latents)?
        } else {
            (self.pyramid.forward(&synthesis_latents)?, Vec::new())
        };
        let synthesis_input = if arch.common_randomness {
            let common = fixed_pyramid(&self.common)?;
            let common = resize(
                &common,
                arch.height,
                arch.width,
                super::architecture::Interpolation::Bicubic,
            )?;
            dense.concat(&common)?
        } else {
            dense.clone()
        };
        let (native, synthesis_states) = if capture {
            self.synthesis.forward_cached(&synthesis_input)?
        } else {
            (self.synthesis.forward(&synthesis_input)?, Vec::new())
        };
        let raw = resize(&native, arch.height, arch.width, arch.interpolation)?;
        Ok(Reconstruction {
            dense,
            synthesis_input,
            native,
            raw,
            synthesis_states,
            pyramid_states,
        })
    }

    pub fn reconstruct(&self, quantized: &[Tensor]) -> Result<Tensor> {
        Ok(self.reconstruction(quantized, false)?.raw)
    }

    fn tape(
        &self,
        target: &Tensor,
        lambda: f32,
        mode: Quantizer,
        temperature: f32,
        noise: &[f32],
        evaluate: bool,
    ) -> Result<Tape> {
        let arch = &self.architecture;
        if (target.channels, target.height, target.width) != (3, arch.height, arch.width)
            || !lambda.is_finite()
            || lambda <= 0.0
            || noise.len() != self.latents.iter().map(|l| l.data.len()).sum::<usize>()
        {
            return Err(Error("Invalid reference training input"));
        }
        let mut quantized = Vec::new();
        let mut derivatives = Vec::new();
        let mut offset = 0;
        for latent in &self.latents {
            let scaled: Vec<_> = latent.data.iter().map(|v| v * 16.0).collect();
            let (values, derivative) = quantizer::quantize(
                &scaled,
                &noise[offset..offset + scaled.len()],
                mode,
                temperature,
            )?;
            quantized.push(Tensor::new(1, latent.height, latent.width, values)?);
            derivatives.push(derivative);
            offset += scaled.len();
        }
        let inter = if let Some(ifce) = &self.ifce {
            Some(ifce.forward(&quantized)?)
        } else {
            None
        };
        let mut contexts = Vec::new();
        let mut raw_arms = Vec::new();
        let mut arm_states = Vec::new();
        let mut rates = Vec::new();
        for (index, latent) in quantized.iter().enumerate() {
            let mut context = spatial_context(latent, arch.spatial_context)?;
            if let Some(inter) = &inter {
                context = context.concat(&inter[index])?;
            }
            let raw = if evaluate {
                self.arm.forward(&context)?
            } else {
                let (raw, states) = self.arm.forward_cached(&context)?;
                arm_states.push(states);
                raw
            };
            rates.push(laplace_rate(latent, &raw)?);
            contexts.push(context);
            raw_arms.push(raw);
        }
        let Reconstruction {
            dense,
            synthesis_input,
            native,
            raw,
            synthesis_states,
            pyramid_states,
        } = self.reconstruction(&quantized, !evaluate)?;
        let mut image = raw.clone();
        for v in &mut image.data {
            *v = v.clamp(0.0, 1.0);
            if evaluate {
                *v = (*v * 255.0).round_ties_even() / 255.0;
            }
        }
        // Accumulate reductions precisely: a serial f32 sum develops visible
        // size-dependent loss bias on megapixel images. Tensor operations and
        // the returned loss remain f32, like the reference tensor reductions.
        let mse = image
            .data
            .iter()
            .zip(&target.data)
            .map(|(a, b)| f64::from((a - b).powi(2)))
            .sum::<f64>() as f32
            / image.data.len() as f32;
        let bits: Vec<_> = rates.iter().map(|r| r.bits.clone()).collect();
        let rate_bpp = bits
            .iter()
            .flat_map(|t| &t.data)
            .map(|&v| f64::from(v))
            .sum::<f64>() as f32
            / (arch.height * arch.width) as f32;
        let evaluation = Evaluation {
            image,
            bits,
            mse,
            rate_bpp,
            loss: mse + lambda * rate_bpp,
        };
        Ok(Tape {
            arm_states,
            synthesis_states,
            pyramid_states,
            quantized,
            derivatives,
            contexts,
            raw_arms,
            rates,
            dense,
            synthesis_input,
            native,
            raw,
            evaluation,
        })
    }

    pub fn evaluate(&self, target: &Tensor, lambda: f32) -> Result<Evaluation> {
        let n = self.latents.iter().map(|l| l.data.len()).sum();
        Ok(self
            .tape(target, lambda, Quantizer::Hard, 0.0, &vec![0.0; n], true)?
            .evaluation)
    }

    pub fn loss_gradient(
        &self,
        target: &Tensor,
        lambda: f32,
        mode: Quantizer,
        temperature: f32,
        noise: &[f32],
    ) -> Result<(Evaluation, Gradient)> {
        let tape = self.tape(target, lambda, mode, temperature, noise, false)?;
        let mut output_gradient = tape.raw.clone();
        let n = output_gradient.data.len() as f32;
        for ((gradient, (&raw, &decoded)), &target) in output_gradient
            .data
            .iter_mut()
            .zip(tape.raw.data.iter().zip(&tape.evaluation.image.data))
            .zip(&target.data)
        {
            *gradient = if raw > 0.0 && raw < 1.0 {
                2.0 * (decoded - target) / n
            } else {
                0.0
            };
        }
        let native_gradient = resize_backward(
            &tape.native,
            &output_gradient,
            self.architecture.interpolation,
        )?;
        let synthesis = self.synthesis.backward_cached(
            &tape.synthesis_input,
            &native_gradient,
            &tape.synthesis_states,
        )?;
        let dense_gradient = synthesis.input.first_channels(tape.dense.channels)?;
        let specs = self.architecture.latent_specs();
        let synthesis_latents: Vec<_> = tape
            .quantized
            .iter()
            .zip(&specs)
            .filter(|(_, s)| !s.hyper)
            .map(|(q, _)| q.clone())
            .collect();
        let pyramid = self.pyramid.backward_cached(
            &synthesis_latents,
            &dense_gradient,
            &tape.pyramid_states,
        )?;
        let mut quantized_gradient: Vec<_> = tape
            .quantized
            .iter()
            .map(|t| Tensor::zeros(1, t.height, t.width))
            .collect::<Result<_>>()?;
        let mut syn_index = 0;
        for (index, spec) in specs.iter().enumerate() {
            if !spec.hyper {
                quantized_gradient[index].add_assign(&pyramid.latents[syn_index])?;
                syn_index += 1;
            }
        }
        let rate_weight = lambda / (target.height * target.width) as f32;
        let mut arm_gradient: Option<ArmGradient> = None;
        let mut inter_gradients = Vec::new();
        for (index, rate) in tape.rates.iter().enumerate() {
            let mut raw_gradient = tape.raw_arms[index].clone();
            for (a, b) in raw_gradient.data.iter_mut().zip(&rate.arm_derivative.data) {
                *a = b * rate_weight;
            }
            let gradient = self.arm.backward_cached(
                &tape.contexts[index],
                &raw_gradient,
                &tape.arm_states[index],
            )?;
            let spatial = gradient
                .input
                .first_channels(self.architecture.spatial_context)?;
            quantized_gradient[index].add_assign(&spatial_context_backward(&spatial)?)?;
            for (a, b) in quantized_gradient[index]
                .data
                .iter_mut()
                .zip(&rate.latent_derivative.data)
            {
                *a += b * rate_weight;
            }
            if self.ifce.is_some() {
                let start = self.architecture.spatial_context * rate.bits.height * rate.bits.width;
                inter_gradients.push(Tensor::new(
                    self.architecture.ifce_outputs,
                    rate.bits.height,
                    rate.bits.width,
                    gradient.input.data[start..].to_vec(),
                )?);
            }
            if let Some(total) = &mut arm_gradient {
                add_arm(total, &gradient);
            } else {
                arm_gradient = Some(gradient);
            }
        }
        let ifce = if let Some(ifce) = &self.ifce {
            let gradient = ifce.backward(&tape.quantized, &inter_gradients)?;
            for (a, b) in quantized_gradient.iter_mut().zip(&gradient.latents) {
                a.add_assign(b)?;
            }
            Some(gradient)
        } else {
            None
        };
        for (gradient, derivative) in quantized_gradient.iter_mut().zip(&tape.derivatives) {
            for (a, b) in gradient.data.iter_mut().zip(derivative) {
                *a *= 16.0 * b;
            }
        }
        let gradient = Gradient {
            latents: quantized_gradient,
            arm: arm_gradient.unwrap(),
            ifce,
            pyramid,
            synthesis,
        };
        Ok((tape.evaluation, gradient))
    }
}
