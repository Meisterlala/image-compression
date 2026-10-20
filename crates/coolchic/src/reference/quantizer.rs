//! Reference soft rounding and its reverse-mode derivative. Noise is supplied
//! explicitly, so a saved random sample can be replayed for parity/gradient tests.
use crate::{Error, Result};

#[derive(Clone, Copy, Debug)]
pub enum Quantizer {
    None,
    SoftAlone,
    Soft,
    StraightThrough,
    Hard,
}

pub fn soft_round(x: f32, temperature: f32) -> f32 {
    let floor = x.floor();
    floor
        + 0.5 * ((x - floor - 0.5) / temperature).tanh() / (1.0 / (2.0 * temperature)).tanh()
        + 0.5
}

fn soft_pair(x: f32, temperature: f32, normalizer: f32, derivative_denominator: f32) -> (f32, f32) {
    let floor = x.floor();
    let tanh = ((x - floor - 0.5) / temperature).tanh();
    (
        floor + 0.5 * tanh / normalizer + 0.5,
        0.5 * (1.0 - tanh * tanh) / derivative_denominator,
    )
}

/// Returns forward values and elementwise dy/dx. `Hard` has zero gradient,
/// matching PyTorch round. In pinned v5, the STE branch also detaches its
/// result inside torch.no_grad (despite its docstring claiming a surrogate
/// derivative); retain actual upstream behavior, not that intended behavior.
pub fn quantize(
    input: &[f32],
    noise: &[f32],
    mode: Quantizer,
    temperature: f32,
) -> Result<(Vec<f32>, Vec<f32>)> {
    if noise.len() != input.len()
        || (!matches!(mode, Quantizer::None | Quantizer::Hard)
            && (!temperature.is_finite() || temperature <= 0.0))
    {
        return Err(Error("Invalid reference quantizer parameters"));
    }
    // The temperature normalizer is constant across the tensor. Forward and
    // derivative share the same tanh, rather than evaluating it twice/sample.
    let normalizer = if matches!(mode, Quantizer::Soft | Quantizer::SoftAlone) {
        (1.0 / (2.0 * temperature)).tanh()
    } else {
        1.0
    };
    let derivative_denominator = temperature * normalizer;
    let apply = |x: f32, n: f32| match mode {
        Quantizer::None => (x + n, 1.0),
        Quantizer::Hard => (x.round_ties_even(), 0.0),
        Quantizer::StraightThrough => (x.round_ties_even(), 0.0),
        Quantizer::SoftAlone => soft_pair(x, temperature, normalizer, derivative_denominator),
        Quantizer::Soft => {
            let (inner, dx) = soft_pair(x, temperature, normalizer, derivative_denominator);
            let (value, dy) = soft_pair(inner + n, temperature, normalizer, derivative_denominator);
            (value, dy * dx)
        }
    };
    let mut output = vec![0.0; input.len()];
    let mut derivative = vec![0.0; input.len()];
    #[cfg(feature = "parallel")]
    if super::parallel::worthwhile(input.len() * 32) {
        use rayon::prelude::*;
        output
            .par_iter_mut()
            .zip(derivative.par_iter_mut())
            .zip(input.par_iter().zip(noise.par_iter()))
            .with_min_len(2048)
            .for_each(|((value, gradient), (&x, &n))| {
                (*value, *gradient) = apply(x, n);
            });
        return Ok((output, derivative));
    }
    for ((value, gradient), (&x, &n)) in output
        .iter_mut()
        .zip(&mut derivative)
        .zip(input.iter().zip(noise))
    {
        (*value, *gradient) = apply(x, n);
    }
    Ok((output, derivative))
}

/// Transform an externally supplied U(0,1) sample exactly as the reference;
/// Gaussian sampling/RNG state belongs to the trainer, not this operator.
pub fn kumaraswamy(uniform: f32, a: f32) -> Result<f32> {
    if !uniform.is_finite() || !(0.0..=1.0).contains(&uniform) || !a.is_finite() || a <= 0.0 {
        return Err(Error("Invalid reference quantization noise parameters"));
    }
    let b = (2.0_f32.powf(a) * (a - 1.0) + 1.0) / a;
    if !b.is_finite() || b <= 0.0 {
        return Err(Error("Invalid Kumaraswamy distribution"));
    }
    Ok((1.0 - (1.0 - uniform).powf(1.0 / b)).powf(1.0 / a) - 0.5)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_tanh_and_parallel_soft_quantization_match_the_scalar_formulas() {
        let input: Vec<_> = (0..65537).map(|i| (i % 811) as f32 / 31.0 - 12.0).collect();
        let noise: Vec<_> = (0..input.len())
            .map(|i| (i % 37) as f32 / 101.0 - 0.2)
            .collect();
        for temperature in [0.08, 0.35, 1.2] {
            let old_derivative = |x: f32| {
                let t = ((x - x.floor() - 0.5) / temperature).tanh();
                0.5 * (1.0 - t * t) / (temperature * (1.0 / (2.0 * temperature)).tanh())
            };
            for mode in [Quantizer::SoftAlone, Quantizer::Soft] {
                let (values, derivatives) = quantize(&input, &noise, mode, temperature).unwrap();
                for (i, (&x, &n)) in input.iter().zip(&noise).enumerate() {
                    let (value, derivative) = if matches!(mode, Quantizer::Soft) {
                        let inner = soft_round(x, temperature) + n;
                        (
                            soft_round(inner, temperature),
                            old_derivative(inner) * old_derivative(x),
                        )
                    } else {
                        (soft_round(x, temperature), old_derivative(x))
                    };
                    assert_eq!(values[i].to_bits(), value.to_bits());
                    assert_eq!(derivatives[i].to_bits(), derivative.to_bits());
                }
            }
        }
    }

    #[test]
    fn ties_are_even_and_ste_does_not_change_forward() {
        let input = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5];
        let (hard, dhard) = quantize(&input, &[0.0; 6], Quantizer::Hard, 0.0).unwrap();
        let (ste, dste) = quantize(&input, &[0.0; 6], Quantizer::StraightThrough, 0.35).unwrap();
        assert_eq!(hard, vec![-2.0, -2.0, -0.0, 0.0, 2.0, 2.0]);
        assert_eq!(ste, hard);
        assert!(dhard.iter().all(|&x| x == 0.0));
        assert!(dste.iter().all(|&x| x == 0.0));
    }
}
