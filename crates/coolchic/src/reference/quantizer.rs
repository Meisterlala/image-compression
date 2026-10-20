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

fn soft_derivative(x: f32, temperature: f32) -> f32 {
    let tanh = ((x - x.floor() - 0.5) / temperature).tanh();
    0.5 * (1.0 - tanh * tanh) / (temperature * (1.0 / (2.0 * temperature)).tanh())
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
    let mut output = Vec::with_capacity(input.len());
    let mut derivative = Vec::with_capacity(input.len());
    for (&x, &n) in input.iter().zip(noise) {
        let (value, gradient) = match mode {
            Quantizer::None => (x + n, 1.0),
            Quantizer::Hard => (x.round_ties_even(), 0.0),
            Quantizer::StraightThrough => (x.round_ties_even(), 0.0),
            Quantizer::SoftAlone => (soft_round(x, temperature), soft_derivative(x, temperature)),
            Quantizer::Soft => {
                let inner = soft_round(x, temperature) + n;
                (
                    soft_round(inner, temperature),
                    soft_derivative(inner, temperature) * soft_derivative(x, temperature),
                )
            }
        };
        output.push(value);
        derivative.push(gradient);
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
