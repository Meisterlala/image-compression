//! Learned separable/symmetric filters from component/core/upsampling.py.
//! Includes learned weights and their derivatives. Transposed convolution uses
//! reference replicate padding/crop.
use super::Tensor;
use crate::{Error, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FilterMode {
    Upsample,
    Residual,
}

#[derive(Clone, Debug)]
pub struct Filter {
    pub kernel: usize,
    /// Only the left half (including the middle value for odd kernels).
    pub half: Vec<f32>,
    pub mode: FilterMode,
}

#[derive(Debug)]
pub struct FilterGradient {
    pub input: Tensor,
    pub half: Vec<f32>,
}

impl Filter {
    fn validate(&self) -> Result<()> {
        let valid = match self.mode {
            FilterMode::Upsample => self.kernel >= 4 && self.kernel.is_multiple_of(2),
            FilterMode::Residual => self.kernel % 2 == 1,
        };
        if !valid || self.half.len() != self.kernel.div_ceil(2) {
            return Err(Error("Invalid reference upsampling filter"));
        }
        Ok(())
    }

    fn half_index(&self, k: usize) -> usize {
        k.min(self.kernel - k - 1)
    }

    fn source(&self, target: usize, k: usize, size: usize) -> Option<usize> {
        match self.mode {
            FilterMode::Residual => {
                let source = target.checked_add(k)?.checked_sub(self.kernel / 2)?;
                (source < size).then_some(source)
            }
            FilterMode::Upsample => {
                let padding = self.kernel / 2;
                let crop = 2 * padding - 1 + self.kernel / 2;
                let coordinate = target.checked_add(crop)?.checked_sub(k)?;
                if coordinate % 2 != 0 {
                    return None;
                }
                let padded = coordinate / 2;
                if padded >= size.checked_add(2 * padding)? {
                    return None;
                }
                Some(padded.saturating_sub(padding).min(size - 1))
            }
        }
    }

    fn axis_forward(&self, input: &Tensor, horizontal: bool) -> Result<Tensor> {
        let factor = if self.mode == FilterMode::Upsample {
            2
        } else {
            1
        };
        let height = input
            .height
            .checked_mul(if horizontal { 1 } else { factor })
            .ok_or(Error("Reference upsampling dimensions overflow"))?;
        let width = input
            .width
            .checked_mul(if horizontal { factor } else { 1 })
            .ok_or(Error("Reference upsampling dimensions overflow"))?;
        let mut output = Tensor::zeros(input.channels, height, width)?;
        let row = |index: usize, values: &mut [f32]| {
            let channel = index / height;
            let y = index % height;
            for (x, value) in values.iter_mut().enumerate() {
                let mut sum = 0.0;
                for k in 0..self.kernel {
                    let position = self.source(
                        if horizontal { x } else { y },
                        k,
                        if horizontal {
                            input.width
                        } else {
                            input.height
                        },
                    );
                    if let Some(position) = position {
                        let i = input.index(
                            channel,
                            if horizontal { y } else { position },
                            if horizontal { position } else { x },
                        );
                        sum += input.data[i] * self.half[self.half_index(k)];
                    }
                }
                *value = sum;
            }
        };
        #[cfg(feature = "parallel")]
        if super::parallel::worthwhile(output.data.len() * self.kernel) {
            use rayon::prelude::*;
            output
                .data
                .par_chunks_mut(width)
                .enumerate()
                .for_each(|(i, values)| row(i, values));
            return Ok(output);
        }
        output
            .data
            .chunks_mut(width)
            .enumerate()
            .for_each(|(i, values)| row(i, values));
        Ok(output)
    }

    fn axis_backward(
        &self,
        input: &Tensor,
        gradient: &Tensor,
        horizontal: bool,
    ) -> Result<FilterGradient> {
        #[cfg(feature = "parallel")]
        if super::parallel::worthwhile(gradient.data.len() * self.kernel) {
            return self.parallel_axis_backward(input, gradient, horizontal);
        }
        let mut result = FilterGradient {
            input: Tensor::zeros(input.channels, input.height, input.width)?,
            half: vec![0.0; self.half.len()],
        };
        for channel in 0..gradient.channels {
            for y in 0..gradient.height {
                for x in 0..gradient.width {
                    let g = gradient.data[gradient.index(channel, y, x)];
                    for k in 0..self.kernel {
                        let position = self.source(
                            if horizontal { x } else { y },
                            k,
                            if horizontal {
                                input.width
                            } else {
                                input.height
                            },
                        );
                        if let Some(position) = position {
                            let i = input.index(
                                channel,
                                if horizontal { y } else { position },
                                if horizontal { position } else { x },
                            );
                            let ki = self.half_index(k);
                            result.input.data[i] += g * self.half[ki];
                            result.half[ki] += g * input.data[i];
                        }
                    }
                }
            }
        }
        Ok(result)
    }

    #[cfg(feature = "parallel")]
    fn parallel_axis_backward(
        &self,
        input: &Tensor,
        gradient: &Tensor,
        horizontal: bool,
    ) -> Result<FilterGradient> {
        use rayon::prelude::*;
        // Independent rows/columns own their input gradients. Collect in raster
        // order for deterministic filter-gradient reduction, never float atomics.
        let lines = input.channels
            * if horizontal {
                input.height
            } else {
                input.width
            };
        let size = if horizontal {
            input.width
        } else {
            input.height
        };
        let targets = if horizontal {
            gradient.width
        } else {
            gradient.height
        };
        let parts: Vec<_> = (0..lines)
            .into_par_iter()
            .map(|line| {
                let c = line
                    / if horizontal {
                        input.height
                    } else {
                        input.width
                    };
                let fixed = line
                    % if horizontal {
                        input.height
                    } else {
                        input.width
                    };
                let mut values = vec![0.0; size];
                let mut half = vec![0.0; self.half.len()];
                for target in 0..targets {
                    let g = gradient.data[gradient.index(
                        c,
                        if horizontal { fixed } else { target },
                        if horizontal { target } else { fixed },
                    )];
                    for k in 0..self.kernel {
                        if let Some(position) = self.source(target, k, size) {
                            let ki = self.half_index(k);
                            values[position] += g * self.half[ki];
                            let i = input.index(
                                c,
                                if horizontal { fixed } else { position },
                                if horizontal { position } else { fixed },
                            );
                            half[ki] += g * input.data[i];
                        }
                    }
                }
                (values, half)
            })
            .collect();
        let mut result = FilterGradient {
            input: Tensor::zeros(input.channels, input.height, input.width)?,
            half: vec![0.0; self.half.len()],
        };
        for (line, (values, half)) in parts.into_iter().enumerate() {
            if horizontal {
                result.input.data[line * input.width..(line + 1) * input.width]
                    .copy_from_slice(&values);
            } else {
                let c = line / input.width;
                let x = line % input.width;
                for (y, value) in values.into_iter().enumerate() {
                    result.input.data[(c * input.height + y) * input.width + x] = value;
                }
            }
            for (a, b) in result.half.iter_mut().zip(half) {
                *a += b;
            }
        }
        Ok(result)
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.validate()?;
        let horizontal = self.axis_forward(input, true)?;
        let mut output = self.axis_forward(&horizontal, false)?;
        if self.mode == FilterMode::Residual {
            output.add_assign(input)?;
        }
        Ok(output)
    }

    pub fn backward(&self, input: &Tensor, gradient: &Tensor) -> Result<FilterGradient> {
        self.validate()?;
        let horizontal = self.axis_forward(input, true)?;
        let factor = if self.mode == FilterMode::Upsample {
            2
        } else {
            1
        };
        if (gradient.channels, gradient.height, gradient.width)
            != (input.channels, horizontal.height * factor, horizontal.width)
        {
            return Err(Error("Invalid reference filter gradient shape"));
        }
        let vertical = self.axis_backward(&horizontal, gradient, false)?;
        let mut result = self.axis_backward(input, &vertical.input, true)?;
        for (a, b) in result.half.iter_mut().zip(vertical.half) {
            *a += b;
        }
        if self.mode == FilterMode::Residual {
            result.input.add_assign(gradient)?;
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_neighbour_is_a_special_case_not_a_substitute() {
        let filter = Filter {
            kernel: 4,
            half: vec![0.0, 1.0],
            mode: FilterMode::Upsample,
        };
        let input = Tensor::new(1, 1, 2, vec![2.0, 7.0]).unwrap();
        assert_eq!(
            filter.forward(&input).unwrap().data,
            vec![2.0, 2.0, 7.0, 7.0, 2.0, 2.0, 7.0, 7.0]
        );
    }

    #[test]
    fn preconcat_has_a_residual_even_for_one_tap() {
        let filter = Filter {
            kernel: 1,
            half: vec![2.0],
            mode: FilterMode::Residual,
        };
        let input = Tensor::new(1, 1, 1, vec![3.0]).unwrap();
        assert_eq!(filter.forward(&input).unwrap().data, vec![15.0]);
        let gradient = Tensor::new(1, 1, 1, vec![1.0]).unwrap();
        let g = filter.backward(&input, &gradient).unwrap();
        assert_eq!(g.input.data, vec![5.0]);
        assert_eq!(g.half, vec![12.0]);
    }
}
