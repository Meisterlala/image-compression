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
    if layer.kernel != 1 {
        // Partition the output allocation (co * height rows), never the input
        // channel count; rectangular ci != co convolutions use the same path.
        output
            .data
            .par_chunks_mut(input.width)
            .enumerate()
            .for_each(|(index, values)| {
                layer.forward_row(input, index / input.height, index % input.height, values);
            });
        return Ok(output);
    }
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
                    }
                });
        });
    Ok(output)
}

fn input_gradient_row(
    layer: &Conv2d,
    input: &Tensor,
    gated: &[f32],
    c: usize,
    iy: usize,
    values: &mut [f32],
) {
    let width = input.width;
    let pad = (layer.kernel / 2) as isize;
    if layer.residual {
        values.copy_from_slice(
            &gated[(c * input.height + iy) * width..(c * input.height + iy + 1) * width],
        );
    }
    for out in 0..layer.output_channels {
        for ky in 0..layer.kernel {
            let y = iy as isize + pad - ky as isize;
            let lo_y = if iy == 0 { 0 } else { y.max(0) };
            let hi_y = if iy == input.height - 1 {
                iy as isize
            } else {
                y.min(input.height as isize - 1)
            };
            for kx in 0..layer.kernel {
                let shift = pad - kx as isize;
                let weight = layer.weight
                    [((out * input.channels + c) * layer.kernel + ky) * layer.kernel + kx];
                for yy in lo_y..=hi_y {
                    let source = &gated[(out * input.height + yy as usize) * width
                        ..(out * input.height + yy as usize + 1) * width];
                    if width == 1 {
                        values[0] += source[0] * weight;
                        continue;
                    }
                    // Only first/last columns have multiple replicate-padding
                    // preimages. The entire interior is a contiguous SIMD axpy.
                    let left_end = shift.min(width as isize - 1);
                    for x in 0..=left_end {
                        values[0] += source[x as usize] * weight;
                    }
                    let right_start = (width as isize - 1 + shift).max(0);
                    for x in right_start..width as isize {
                        values[width - 1] += source[x as usize] * weight;
                    }
                    let begin = 1.max(-shift).min(width as isize - 1) as usize;
                    let end = (width as isize - 1)
                        .min(width as isize - shift)
                        .max(begin as isize) as usize;
                    if end > begin {
                        let start = (begin as isize + shift) as usize;
                        for (v, &g) in values[begin..end]
                            .iter_mut()
                            .zip(&source[start..start + end - begin])
                        {
                            *v += g * weight;
                        }
                    }
                }
            }
        }
    }
}

pub fn backward(
    layer: &Conv2d,
    input: &Tensor,
    gradient: &Tensor,
    output: Option<&Tensor>,
) -> Result<ConvGradient> {
    let pixels = input.height * input.width;
    let mut gated = gradient.data.clone();
    if layer.relu {
        let recomputed = if output.is_none() {
            Some(forward(layer, input)?)
        } else {
            None
        };
        let activated = output.or(recomputed.as_ref()).unwrap();
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
    if layer.kernel != 1 {
        result
            .input
            .data
            .par_chunks_mut(input.width)
            .enumerate()
            .for_each(|(index, values)| {
                input_gradient_row(
                    layer,
                    input,
                    &gated,
                    index / input.height,
                    index % input.height,
                    values,
                );
            });
    } else {
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
                                let source = &gated
                                    [out * pixels + start..out * pixels + start + values.len()];
                                let weight = layer.weight[out * input.channels + c];
                                for (i, (v, &g)) in values.iter_mut().zip(source).enumerate() {
                                    if layer.residual && out == c {
                                        *v += gated[c * pixels + start + i];
                                    }
                                    *v += g * weight;
                                }
                            }
                        }
                    });
            });
    }
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
