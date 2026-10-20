//! Data-parallel kernels with disjoint output ownership: no locks, atomics or
//! per-pixel allocations. Each scalar reduction retains a fixed order.
use super::{Conv2d, ConvGradient, Tensor};
use crate::Result;
use rayon::prelude::*;

pub fn worthwhile(operations: usize) -> bool {
    operations >= 1_000_000 && rayon::current_num_threads() > 1
}

pub fn forward(layer: &Conv2d, input: &Tensor) -> Result<Tensor> {
    let pixels = input.height * input.width;
    let mut output = Tensor::zeros(layer.output_channels, input.height, input.width)?;
    output
        .data
        .par_chunks_mut(pixels)
        .enumerate()
        .for_each(|(out, plane)| {
            plane
                .par_chunks_mut(2048)
                .enumerate()
                .for_each(|(tile, values)| {
                    let start = tile * 2048;
                    if layer.kernel == 1 {
                        for c in 0..input.channels {
                            let weight = layer.weight[out * input.channels + c];
                            let source =
                                &input.data[c * pixels + start..c * pixels + start + values.len()];
                            for (v, &s) in values.iter_mut().zip(source) {
                                *v += s * weight;
                            }
                        }
                        for (i, v) in values.iter_mut().enumerate() {
                            *v += layer.bias[out];
                            if layer.residual {
                                *v += input.data[out * pixels + start + i];
                            }
                            if layer.relu {
                                *v = v.max(0.0);
                            }
                        }
                    } else {
                        let pad = layer.kernel / 2;
                        for (i, value) in values.iter_mut().enumerate() {
                            let y = (start + i) / input.width;
                            let x = (start + i) % input.width;
                            let mut v = 0.0;
                            for c in 0..input.channels {
                                for ky in 0..layer.kernel {
                                    let iy = (y + ky).saturating_sub(pad).min(input.height - 1);
                                    for kx in 0..layer.kernel {
                                        let ix = (x + kx).saturating_sub(pad).min(input.width - 1);
                                        let wi = ((out * input.channels + c) * layer.kernel + ky)
                                            * layer.kernel
                                            + kx;
                                        v += input.data[(c * input.height + iy) * input.width + ix]
                                            * layer.weight[wi];
                                    }
                                }
                            }
                            v += layer.bias[out];
                            if layer.residual {
                                v += input.data[out * pixels + start + i];
                            }
                            if layer.relu {
                                v = v.max(0.0);
                            }
                            *value = v;
                        }
                    }
                });
        });
    Ok(output)
}

pub fn backward(layer: &Conv2d, input: &Tensor, gradient: &Tensor) -> Result<ConvGradient> {
    let pixels = input.height * input.width;
    let mut gated = gradient.data.clone();
    if layer.relu {
        let activated = forward(layer, input)?;
        gated
            .par_iter_mut()
            .zip(&activated.data)
            .for_each(|(g, &a)| {
                if a <= 0.0 {
                    *g = 0.0;
                }
            });
    }
    let mut result = ConvGradient {
        input: Tensor::zeros(input.channels, input.height, input.width)?,
        weight: vec![0.0; layer.weight.len()],
        bias: vec![0.0; layer.bias.len()],
    };
    // Gather dInput rather than racing scatter-adds between output tiles.
    result
        .input
        .data
        .par_chunks_mut(pixels)
        .enumerate()
        .for_each(|(c, plane)| {
            plane
                .par_chunks_mut(2048)
                .enumerate()
                .for_each(|(tile, values)| {
                    let start = tile * 2048;
                    if layer.kernel == 1 {
                        for out in 0..layer.output_channels {
                            let source =
                                &gated[out * pixels + start..out * pixels + start + values.len()];
                            let weight = layer.weight[out * input.channels + c];
                            for (i, (v, &g)) in values.iter_mut().zip(source).enumerate() {
                                if layer.residual && out == c {
                                    *v += gated[c * pixels + start + i];
                                }
                                *v += g * weight;
                            }
                        }
                    } else {
                        let pad = (layer.kernel / 2) as isize;
                        for (i, value) in values.iter_mut().enumerate() {
                            let iy = ((start + i) / input.width) as isize;
                            let ix = ((start + i) % input.width) as isize;
                            let mut v = if layer.residual {
                                gated[c * pixels + start + i]
                            } else {
                                0.0
                            };
                            for out in 0..layer.output_channels {
                                for ky in 0..layer.kernel {
                                    let y = iy + pad - ky as isize;
                                    let lo_y = if iy == 0 { 0 } else { y.max(0) };
                                    let hi_y = if iy == input.height as isize - 1 {
                                        iy
                                    } else {
                                        y.min(input.height as isize - 1)
                                    };
                                    for kx in 0..layer.kernel {
                                        let x = ix + pad - kx as isize;
                                        let lo_x = if ix == 0 { 0 } else { x.max(0) };
                                        let hi_x = if ix == input.width as isize - 1 {
                                            ix
                                        } else {
                                            x.min(input.width as isize - 1)
                                        };
                                        let wi = ((out * input.channels + c) * layer.kernel + ky)
                                            * layer.kernel
                                            + kx;
                                        for yy in lo_y..=hi_y {
                                            for xx in lo_x..=hi_x {
                                                v += gated[(out * input.height + yy as usize)
                                                    * input.width
                                                    + xx as usize]
                                                    * layer.weight[wi];
                                            }
                                        }
                                    }
                                }
                            }
                            *value = v;
                        }
                    }
                });
        });
    result
        .weight
        .par_iter_mut()
        .enumerate()
        .for_each(|(wi, value)| {
            let out = wi / (input.channels * layer.kernel * layer.kernel);
            let c = wi / (layer.kernel * layer.kernel) % input.channels;
            let ky = wi / layer.kernel % layer.kernel;
            let kx = wi % layer.kernel;
            let mut sum = 0.0;
            if layer.kernel == 1 {
                for (&g, &v) in gated[out * pixels..(out + 1) * pixels]
                    .iter()
                    .zip(&input.data[c * pixels..(c + 1) * pixels])
                {
                    sum += g * v;
                }
            } else {
                let pad = layer.kernel / 2;
                for y in 0..input.height {
                    let iy = (y + ky).saturating_sub(pad).min(input.height - 1);
                    for x in 0..input.width {
                        let ix = (x + kx).saturating_sub(pad).min(input.width - 1);
                        sum += gated[(out * input.height + y) * input.width + x]
                            * input.data[(c * input.height + iy) * input.width + ix];
                    }
                }
            }
            *value = sum;
        });
    result
        .bias
        .par_iter_mut()
        .enumerate()
        .for_each(|(out, value)| {
            *value = gated[out * pixels..(out + 1) * pixels].iter().sum();
        });
    Ok(result)
}
