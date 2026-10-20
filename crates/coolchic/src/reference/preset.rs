//! Standard intra schedule from training/presets.py (MSE tuning), with an
//! explicitly non-reference shortened schedule for 500–1999-iteration fast runs.
use super::quantizer::Quantizer;
use crate::{Error, Result};

#[derive(Clone, Debug)]
pub struct Phase {
    pub iterations: usize,
    pub learning_rate: f64,
    pub validation_frequency: usize,
    pub patience: usize,
    pub cosine: bool,
    pub temperature: (f32, f32),
    pub noise: (f32, f32),
    pub quantizer: Quantizer,
}

impl Phase {
    /// The reference updates temperatures only after validation, using the
    /// zero-based iteration index, not a continuously changing per-step value.
    pub fn scheduled_values(&self, iteration: usize) -> (f32, f32) {
        let last_validation = if iteration < self.validation_frequency {
            0
        } else {
            iteration / self.validation_frequency * self.validation_frequency - 1
        };
        let t = last_validation as f32 / self.iterations as f32;
        (
            self.temperature.0 + t * (self.temperature.1 - self.temperature.0),
            self.noise.0 + t * (self.noise.1 - self.noise.0),
        )
    }
    pub fn learning_rate_at(&self, validations: usize) -> f64 {
        if !self.cosine {
            return self.learning_rate;
        }
        let max = self.iterations as f64 / self.validation_frequency as f64;
        1e-5 + (self.learning_rate - 1e-5)
            * (1.0 + (std::f64::consts::PI * validations as f64 / max).cos())
            / 2.0
    }
}

#[derive(Clone, Debug)]
pub struct Intra {
    pub warmup: Phase,
    pub core: Phase,
    pub hard: Phase,
    pub iterations: usize,
}
impl Intra {
    pub fn new(iterations: usize, lambda: f32, start_lr: f64) -> Result<Self> {
        if !(500..=100_000).contains(&iterations)
            || !lambda.is_finite()
            || lambda <= 0.0
            || !start_lr.is_finite()
            || start_lr <= 0.0
        {
            return Err(Error(
                "Intra training needs 500–100000 iterations and a positive rate constraint",
            ));
        }
        // Below the official 2000 minimum, scale all stages and validation
        // cadence proportionally. Keep the reference schedule unchanged above it.
        let short = iterations < 2000;
        let hard = if short { iterations / 4 } else { 500 };
        let validation = if short { iterations / 20 } else { 100 };
        let warm = if short {
            iterations / 40
        } else if iterations < 4000 {
            50
        } else if iterations < 9000 {
            100
        } else if iterations < 12000 {
            200
        } else {
            400
        };
        let noise = ((-0.432 * f64::from(lambda).log10() + 0.747) / 10.0) as f32;
        Ok(Self {
            warmup: Phase {
                iterations: warm,
                learning_rate: start_lr,
                validation_frequency: validation,
                patience: warm,
                cosine: false,
                temperature: (0.35, 0.35),
                noise: (noise, noise),
                quantizer: Quantizer::Soft,
            },
            core: Phase {
                iterations: iterations - hard - 7 * warm,
                learning_rate: 0.01,
                validation_frequency: validation,
                patience: 5000,
                cosine: true,
                temperature: (0.35, 0.08),
                noise: (0.22, 0.15),
                quantizer: Quantizer::Soft,
            },
            hard: Phase {
                iterations: hard,
                learning_rate: 1e-4,
                validation_frequency: validation,
                patience: 10000,
                cosine: true,
                temperature: (0.3, 0.3),
                noise: (1.0, 1.0),
                quantizer: Quantizer::Hard,
            },
            iterations,
        })
    }
}
