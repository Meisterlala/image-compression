//! Laplace rate and analytical reverse-mode derivatives from the reference
//! CoolChicEncoder.forward. The probability floor is a differentiable clamp,
//! not a detached L1 proxy as used by the legacy reduced encoder.
use super::Tensor;
use crate::{Error, Result};

pub struct Rate {
    pub bits: Tensor,
    pub latent_derivative: Tensor,
    /// Derivatives w.r.t. raw ARM [mu, log_scale] output, before shift/clamp/exp.
    pub arm_derivative: Tensor,
}

fn cdf(delta: f32, scale: f32) -> f32 {
    let sign = if delta == 0.0 { 0.0 } else { delta.signum() };
    0.5 - 0.5 * sign * (-delta.abs() / scale).exp_m1()
}

fn density(delta: f32, scale: f32) -> f32 {
    // The actual upstream expression sign(x)*expm1(-abs(x)) has a zero
    // autograd derivative at x=0. Preserve this source-level kink for parity.
    if delta == 0.0 {
        0.0
    } else {
        (-delta.abs() / scale).exp() / (2.0 * scale)
    }
}

pub fn laplace_rate(latent: &Tensor, raw_arm: &Tensor) -> Result<Rate> {
    if latent.channels != 1
        || raw_arm.channels != 2
        || (latent.height, latent.width) != (raw_arm.height, raw_arm.width)
    {
        return Err(Error("Invalid reference Laplace rate shapes"));
    }
    let mut result = Rate {
        bits: Tensor::zeros(1, latent.height, latent.width)?,
        latent_derivative: Tensor::zeros(1, latent.height, latent.width)?,
        arm_derivative: Tensor::zeros(2, latent.height, latent.width)?,
    };
    let n = latent.data.len();
    for i in 0..n {
        let mu = raw_arm.data[i];
        let shifted = raw_arm.data[n + i] - 4.0;
        let scale = shifted.clamp(-5.0, 5.0).exp();
        let upper = latent.data[i] + 0.5 - mu;
        let lower = latent.data[i] - 0.5 - mu;
        let probability = cdf(upper, scale) - cdf(lower, scale);
        let floor = 2.0_f32.powi(-16);
        result.bits.data[i] = -probability.max(floor).log2();
        if probability < floor {
            continue;
        }
        let multiplier = -1.0 / (probability * std::f32::consts::LN_2);
        let dlatent = density(upper, scale) - density(lower, scale);
        result.latent_derivative.data[i] = multiplier * dlatent;
        result.arm_derivative.data[i] = -multiplier * dlatent;
        if shifted > -5.0 && shifted < 5.0 {
            let dscale = (-upper * (-upper.abs() / scale).exp()
                + lower * (-lower.abs() / scale).exp())
                / (2.0 * scale * scale);
            result.arm_derivative.data[n + i] = multiplier * dscale * scale;
        }
    }
    Ok(result)
}
