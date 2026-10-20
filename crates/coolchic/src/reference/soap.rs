//! Cool-Chic v5 SOAP per-parameter state and reference training configurations.
//! Port of Nikhil Vyas's SOAP (MIT), modified by Orange; see SOAP-LICENSE.txt.
//! Scope: channels-first, unmerged dimensions, no gradient normalization, with
//! bias correction, as used by training/train.py. Not a different Adam proxy.
use crate::{Error, Result};
use nalgebra::{DMatrix, linalg::SymmetricEigen};

#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub learning_rate: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    pub weight_decay: f64,
    pub precondition_frequency: usize,
    pub max_precondition_dim: usize,
}

impl Config {
    pub fn network(learning_rate: f64) -> Self {
        Self {
            learning_rate,
            beta1: 0.95,
            beta2: 0.95,
            epsilon: 1e-8,
            weight_decay: 0.01,
            precondition_frequency: 10,
            max_precondition_dim: 256,
        }
    }
    pub fn latent(learning_rate: f64) -> Self {
        Self {
            beta1: 0.9,
            beta2: 0.999,
            weight_decay: 0.0,
            max_precondition_dim: 0,
            precondition_frequency: 1,
            ..Self::network(learning_rate)
        }
    }
    fn validate(&self) -> Result<()> {
        if !self.learning_rate.is_finite()
            || self.learning_rate < 0.0
            || !(0.0..1.0).contains(&self.beta1)
            || !(0.0..1.0).contains(&self.beta2)
            || !self.epsilon.is_finite()
            || self.epsilon <= 0.0
            || !self.weight_decay.is_finite()
            || self.weight_decay < 0.0
            || self.precondition_frequency == 0
        {
            return Err(Error("Invalid reference SOAP settings"));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct Soap {
    shape: Vec<usize>,
    initialized: bool,
    pub step: usize,
    pub first_moment: Vec<f32>,
    pub second_moment: Vec<f32>,
    gram: Vec<Option<DMatrix<f32>>>,
    basis: Vec<Option<DMatrix<f32>>>,
    config: Config,
}

impl Soap {
    pub fn new(shape: Vec<usize>, config: Config) -> Result<Self> {
        config.validate()?;
        let size = shape
            .iter()
            .try_fold(1usize, |n, d| n.checked_mul(*d))
            .ok_or(Error("Reference SOAP dimensions overflow"))?;
        if shape.is_empty() || shape.contains(&0) {
            return Err(Error("Invalid reference SOAP shape"));
        }
        let gram: Vec<_> = shape
            .iter()
            .map(|&dimension| {
                if shape.len() > 1 && dimension <= config.max_precondition_dim {
                    Some(DMatrix::<f32>::zeros(dimension, dimension))
                } else {
                    None
                }
            })
            .collect();
        let basis = vec![None; shape.len()];
        Ok(Self {
            shape,
            initialized: false,
            step: 0,
            first_moment: vec![0.0; size],
            second_moment: vec![0.0; size],
            gram,
            basis,
            config,
        })
    }

    /// Match the reference scheduler without resetting moments or eigenbases.
    pub fn set_learning_rate(&mut self, learning_rate: f64) -> Result<()> {
        let next = Config {
            learning_rate,
            ..self.config
        };
        next.validate()?;
        self.config = next;
        Ok(())
    }

    // Mode multiplication retains original tensor axes. The Python source
    // rotates axes after each tensordot; after a full sweep its result is the
    // same CHW/OIHW layout as this implementation.
    fn project(&self, input: &[f32], back: bool) -> Vec<f32> {
        let mut values = input.to_vec();
        for (axis, matrix) in self.basis.iter().enumerate() {
            if let Some(matrix) = matrix {
                let dimension = self.shape[axis];
                let stride: usize = self.shape[axis + 1..].iter().product();
                let blocks = values.len() / (dimension * stride);
                let mut output = vec![0.0; values.len()];
                for block in 0..blocks {
                    for target in 0..dimension {
                        for suffix in 0..stride {
                            let mut sum = 0.0;
                            for source in 0..dimension {
                                let weight = if back {
                                    matrix[(target, source)]
                                } else {
                                    matrix[(source, target)]
                                };
                                sum +=
                                    values[(block * dimension + source) * stride + suffix] * weight;
                            }
                            output[(block * dimension + target) * stride + suffix] = sum;
                        }
                    }
                }
                values = output;
            }
        }
        values
    }

    fn update_gram(&mut self, gradient: &[f32]) {
        let coefficient = (1.0 - self.config.beta2) as f32;
        for (axis, gram) in self.gram.iter_mut().enumerate() {
            if let Some(gram) = gram {
                let dimension = self.shape[axis];
                let stride: usize = self.shape[axis + 1..].iter().product();
                let blocks = gradient.len() / (dimension * stride);
                for row in 0..dimension {
                    for col in 0..dimension {
                        let mut dot = 0.0;
                        for block in 0..blocks {
                            for suffix in 0..stride {
                                dot += gradient[(block * dimension + row) * stride + suffix]
                                    * gradient[(block * dimension + col) * stride + suffix];
                            }
                        }
                        gram[(row, col)] += coefficient * (dot - gram[(row, col)]);
                    }
                }
            }
        }
    }

    fn initialize_basis(&mut self) -> Result<()> {
        for (axis, gram) in self.gram.iter().enumerate() {
            if let Some(gram) = gram {
                let mut matrix = gram.clone();
                for i in 0..matrix.nrows() {
                    matrix[(i, i)] += 1e-30;
                }
                let eig = SymmetricEigen::try_new(matrix, f32::EPSILON, 10_000)
                    .ok_or(Error("Reference SOAP eigendecomposition did not converge"))?;
                let mut order: Vec<usize> = (0..gram.nrows()).collect();
                order.sort_by(|&a, &b| {
                    eig.eigenvalues[b]
                        .total_cmp(&eig.eigenvalues[a])
                        .then(b.cmp(&a))
                });
                let mut basis = DMatrix::zeros(gram.nrows(), gram.nrows());
                for (target, &source) in order.iter().enumerate() {
                    basis.set_column(target, &eig.eigenvectors.column(source));
                }
                self.basis[axis] = Some(basis);
            }
        }
        Ok(())
    }

    fn reorder_second_moment(&mut self, axis: usize, order: &[usize]) {
        let dimension = self.shape[axis];
        let stride: usize = self.shape[axis + 1..].iter().product();
        let blocks = self.second_moment.len() / (dimension * stride);
        let previous = self.second_moment.clone();
        for block in 0..blocks {
            for (target, &source) in order.iter().enumerate() {
                for suffix in 0..stride {
                    self.second_moment[(block * dimension + target) * stride + suffix] =
                        previous[(block * dimension + source) * stride + suffix];
                }
            }
        }
    }

    fn refresh_basis(&mut self) {
        for axis in 0..self.shape.len() {
            if let (Some(gram), Some(basis)) = (&self.gram[axis], &self.basis[axis]) {
                let dimension = self.shape[axis];
                let estimates = basis.transpose() * gram * basis;
                let mut order: Vec<usize> = (0..dimension).collect();
                order.sort_by(|&a, &b| {
                    estimates[(b, b)]
                        .total_cmp(&estimates[(a, a)])
                        .then(a.cmp(&b))
                });
                let mut sorted = DMatrix::zeros(dimension, dimension);
                for (target, &source) in order.iter().enumerate() {
                    sorted.set_column(target, &basis.column(source));
                }
                let next = (gram * sorted).qr().q();
                self.reorder_second_moment(axis, &order);
                self.basis[axis] = Some(next);
            }
        }
    }

    pub fn update(&mut self, parameter: &mut [f32], gradient: &[f32]) -> Result<()> {
        if parameter.len() != self.first_moment.len()
            || gradient.len() != parameter.len()
            || !gradient.iter().all(|x| x.is_finite())
            || !parameter.iter().all(|x| x.is_finite())
        {
            return Err(Error("Invalid reference SOAP parameter or gradient"));
        }
        if !self.initialized {
            // Even the latent Adam-like group skips its first update in the
            // actual reference. Replacing this with ordinary Adam is incorrect.
            self.update_gram(gradient);
            self.initialize_basis()?;
            self.initialized = true;
            return Ok(());
        }
        let projected = self.project(gradient, false);
        self.step += 1;
        let beta1 = self.config.beta1 as f32;
        let beta2 = self.config.beta2 as f32;
        let mut normalized = vec![0.0; gradient.len()];
        for (index, value) in projected.iter().enumerate() {
            self.first_moment[index] =
                beta1 * self.first_moment[index] + (1.0 - self.config.beta1) as f32 * value;
            self.second_moment[index] = beta2 * self.second_moment[index]
                + (1.0 - self.config.beta2) as f32 * value * value;
            normalized[index] = self.first_moment[index]
                / (self.second_moment[index].sqrt() + self.config.epsilon as f32);
        }
        let step_size = self.config.learning_rate
            * (1.0 - self.config.beta2.powf(self.step as f64)).sqrt()
            / (1.0 - self.config.beta1.powf(self.step as f64));
        let normalized = self.project(&normalized, true);
        for (p, g) in parameter.iter_mut().zip(normalized) {
            *p -= step_size as f32 * g;
            // Reference decays the UPDATED parameter, not its previous value.
            *p -= (self.config.learning_rate * self.config.weight_decay) as f32 * *p;
        }
        self.first_moment = self.project(&self.first_moment, true);
        self.update_gram(gradient);
        if self.step.is_multiple_of(self.config.precondition_frequency) {
            self.refresh_basis();
        }
        self.first_moment = self.project(&self.first_moment, false);
        Ok(())
    }
}
