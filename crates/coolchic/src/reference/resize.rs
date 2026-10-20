//! PyTorch interpolate with align_corners=False (RGB still-image path).
use super::{Tensor, architecture::Interpolation};
use crate::{Error, Result};

fn cubic1(x: f32) -> f32 {
    ((1.25 * x - 2.25) * x) * x + 1.0
}
fn cubic2(x: f32) -> f32 {
    ((-0.75 * x + 3.75) * x - 6.0) * x + 3.0
}

fn taps(input: usize, output: usize, target: usize, mode: Interpolation) -> Vec<(usize, f32)> {
    let scale = input as f32 / output as f32;
    if mode == Interpolation::Nearest {
        return vec![(
            ((target as f32 * scale).floor() as usize).min(input - 1),
            1.0,
        )];
    }
    let source = scale * (target as f32 + 0.5) - 0.5;
    if mode == Interpolation::Bilinear {
        let source = source.max(0.0);
        let low = (source.floor() as usize).min(input - 1);
        let t = source - low as f32;
        return vec![(low, 1.0 - t), ((low + 1).min(input - 1), t)];
    }
    let base = source.floor() as isize;
    let t = source - base as f32;
    [cubic2(t + 1.0), cubic1(t), cubic1(1.0 - t), cubic2(2.0 - t)]
        .into_iter()
        .enumerate()
        .map(|(i, w)| {
            (
                (base + i as isize - 1).clamp(0, input as isize - 1) as usize,
                w,
            )
        })
        .collect()
}

pub fn resize(input: &Tensor, height: usize, width: usize, mode: Interpolation) -> Result<Tensor> {
    if (height, width) == (input.height, input.width) {
        return Ok(input.clone());
    }
    let mut output = Tensor::zeros(input.channels, height, width)?;
    let horizontal: Vec<_> = (0..width)
        .map(|x| taps(input.width, width, x, mode))
        .collect();
    let vertical: Vec<_> = (0..height)
        .map(|y| taps(input.height, height, y, mode))
        .collect();
    for c in 0..input.channels {
        for (y, rows) in vertical.iter().enumerate() {
            for (x, cols) in horizontal.iter().enumerate() {
                let mut value = 0.0;
                for &(iy, wy) in rows {
                    let mut row = 0.0;
                    for &(ix, wx) in cols {
                        row += input.data[input.index(c, iy, ix)] * wx;
                    }
                    value += row * wy;
                }
                let index = output.index(c, y, x);
                output.data[index] = value;
            }
        }
    }
    Ok(output)
}

pub fn resize_backward(input: &Tensor, gradient: &Tensor, mode: Interpolation) -> Result<Tensor> {
    if input.channels != gradient.channels {
        return Err(Error("Reference interpolation channel mismatch"));
    }
    if input.same_shape(gradient) {
        return Ok(gradient.clone());
    }
    let mut output = Tensor::zeros(input.channels, input.height, input.width)?;
    let horizontal: Vec<_> = (0..gradient.width)
        .map(|x| taps(input.width, gradient.width, x, mode))
        .collect();
    let vertical: Vec<_> = (0..gradient.height)
        .map(|y| taps(input.height, gradient.height, y, mode))
        .collect();
    for c in 0..gradient.channels {
        for (y, rows) in vertical.iter().enumerate() {
            for (x, cols) in horizontal.iter().enumerate() {
                let gradient = gradient.data[gradient.index(c, y, x)];
                for &(iy, wy) in rows {
                    for &(ix, wx) in cols {
                        let index = output.index(c, iy, ix);
                        output.data[index] += gradient * wy * wx;
                    }
                }
            }
        }
    }
    Ok(output)
}
