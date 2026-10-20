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
                // LAPACK scales tiny matrices internally; scalar Householder
                // normalization otherwise underflows for zero-gradient Gram
                // matrices regularized by 1e-30 I. Positive scalar scaling
                // preserves the eigenbasis and keeps float32 tie behavior.
                let magnitude = matrix.amax();
                matrix /= magnitude;
                let primary = SymmetricEigen::try_new(matrix.clone(), f32::EPSILON, 10_000);
                let (values, vectors) = if let Some(eig) = primary.filter(|e| {
                    e.eigenvalues
                        .iter()
                        .chain(e.eigenvectors.iter())
                        .all(|v| v.is_finite())
                }) {
                    (eig.eigenvalues, eig.eigenvectors)
                } else {
                    // Match the reference's double-precision fallback when
                    // f32 decomposition fails on a singular/tiny covariance.
                    let eig =
                        SymmetricEigen::try_new(matrix.cast::<f64>(), f32::EPSILON as f64, 10_000)
                            .ok_or(Error("Reference SOAP eigendecomposition did not converge"))?;
                    (
                        eig.eigenvalues.cast::<f32>(),
                        eig.eigenvectors.cast::<f32>(),
                    )
                };
                if vectors.iter().any(|v| !v.is_finite()) {
                    return Err(Error("Non-finite reference SOAP eigenbasis"));
                }
                let mut order: Vec<usize> = (0..gram.nrows()).collect();
                order.sort_by(|&a, &b| values[b].total_cmp(&values[a]).then(b.cmp(&a)));
                let mut basis = DMatrix::zeros(gram.nrows(), gram.nrows());
                for (target, &source) in order.iter().enumerate() {
                    basis.set_column(target, &vectors.column(source));
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
                let mut power = gram * sorted;
                let magnitude = power.amax();
                if magnitude > 0.0 {
                    power /= magnitude;
                }
                let mut next = power.clone().qr().q();
                if next.iter().any(|v| !v.is_finite()) {
                    next = power.cast::<f64>().qr().q().cast::<f32>();
                }
                self.reorder_second_moment(axis, &order);
                self.basis[axis] = Some(next);
            }
        }
    }

    /// Compact state shared with the GPU optimizer. Matrix storage on the wire
    /// is row-major; nalgebra's internal column-major order is never exposed.
    #[cfg(any(feature = "wasm", test))]
    pub(crate) fn gpu_pack(
        &self,
        offset: usize,
        latent: bool,
        state: &mut Vec<f32>,
    ) -> serde_json::Value {
        use serde_json::json;
        let first = state.len();
        state.extend(&self.first_moment);
        let second = state.len();
        state.extend(&self.second_moment);
        let mut axes = Vec::new();
        for (axis, gram) in self.gram.iter().enumerate() {
            if let Some(gram) = gram {
                let dimension = self.shape[axis];
                let gram_offset = state.len();
                for row in 0..dimension {
                    for col in 0..dimension {
                        state.push(gram[(row, col)]);
                    }
                }
                let basis_offset = state.len();
                for row in 0..dimension {
                    for col in 0..dimension {
                        state.push(self.basis[axis].as_ref().map_or(0.0, |q| q[(row, col)]));
                    }
                }
                let order = state.len();
                state.resize(order + dimension, 0.0);
                axes.push(json!({"dimension":dimension,"stride":self.shape[axis+1..].iter().product::<usize>(),"gram":gram_offset,"basis":basis_offset,"order":order}));
            }
        }
        json!({"offset":offset,"length":self.first_moment.len(),"first":first,"second":second,"latent":latent,"axes":axes})
    }

    #[cfg(any(feature = "wasm", test))]
    pub(crate) fn gpu_unpack(&mut self, state: &[f32], cursor: &mut usize, step: usize) {
        let n = self.first_moment.len();
        self.first_moment
            .copy_from_slice(&state[*cursor..*cursor + n]);
        *cursor += n;
        self.second_moment
            .copy_from_slice(&state[*cursor..*cursor + n]);
        *cursor += n;
        for axis in 0..self.shape.len() {
            if let Some(gram) = &mut self.gram[axis] {
                let d = self.shape[axis];
                *gram = DMatrix::from_row_slice(d, d, &state[*cursor..*cursor + d * d]);
                *cursor += d * d;
                self.basis[axis] = Some(DMatrix::from_row_slice(
                    d,
                    d,
                    &state[*cursor..*cursor + d * d],
                ));
                *cursor += d * d + d;
            }
        }
        self.initialized = true;
        self.step = step;
    }

    #[cfg(any(feature = "wasm", test))]
    pub(crate) fn gpu_step(&self) -> i32 {
        if self.initialized {
            self.step as i32
        } else {
            -1
        }
    }

    /// Use exactly the reference eigensolver/QR for the GPU's small basis
    /// refreshes. Returns a row-major basis followed by the variance permutation.
    #[cfg(any(feature = "wasm", test))]
    pub fn gpu_basis(gram: &[f32], previous: Option<&[f32]>) -> Result<Vec<f32>> {
        let d = (gram.len() as f64).sqrt() as usize;
        if d == 0
            || d > 256
            || d * d != gram.len()
            || gram.iter().any(|v| !v.is_finite())
            || previous.is_some_and(|q| q.len() != gram.len() || q.iter().any(|v| !v.is_finite()))
        {
            return Err(Error("Invalid GPU SOAP basis input"));
        }
        let mut soap = Self::new(vec![d], Config::network(0.01))?;
        soap.gram[0] = Some(DMatrix::from_row_slice(d, d, gram));
        soap.second_moment = (0..d).map(|i| i as f32).collect();
        if let Some(q) = previous {
            soap.basis[0] = Some(DMatrix::from_row_slice(d, d, q));
            soap.refresh_basis();
        } else {
            soap.initialize_basis()?;
        }
        let q = soap.basis[0]
            .as_ref()
            .ok_or(Error("GPU SOAP basis missing"))?;
        let mut result = Vec::with_capacity(d * d + d);
        for row in 0..d {
            for col in 0..d {
                result.push(q[(row, col)]);
            }
        }
        result.extend(soap.second_moment);
        Ok(result)
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

#[cfg(test)]
mod gpu_state_tests {
    use super::*;

    #[test]
    fn packed_gpu_state_resumes_the_reference_optimizer() {
        for (shape, config) in [
            (vec![3, 3], Config::network(0.01)),
            (vec![1, 1, 7, 9], Config::latent(0.01)),
        ] {
            let mut reference = Soap::new(shape.clone(), config).unwrap();
            let mut restored = Soap::new(shape, config).unwrap();
            let mut parameters = vec![0.25; reference.first_moment.len()];
            for i in 0..23 {
                let gradient: Vec<_> = (0..parameters.len())
                    .map(|j| ((j * 7 + i * 13) % 31) as f32 / 31.0 - 0.5)
                    .collect();
                reference.update(&mut parameters, &gradient).unwrap();
                let mut state = parameters.clone();
                reference.gpu_pack(0, config.max_precondition_dim == 0, &mut state);
                let mut cursor = parameters.len();
                restored.gpu_unpack(&state, &mut cursor, reference.step);
                assert_eq!(cursor, state.len());
                assert_eq!(restored.gpu_step(), reference.gpu_step());
                let mut actual = parameters.clone();
                let mut expected = parameters.clone();
                restored.update(&mut actual, &gradient).unwrap();
                reference.clone().update(&mut expected, &gradient).unwrap();
                assert_eq!(actual, expected);
            }
        }
    }

    #[test]
    fn basis_helper_matches_reference_initialization_and_refresh() {
        let mut soap = Soap::new(vec![2], Config::network(0.01)).unwrap();
        soap.gram[0] = Some(DMatrix::from_row_slice(2, 2, &[4.0, 1.0, 1.0, 2.0]));
        soap.initialize_basis().unwrap();
        let initial = Soap::gpu_basis(&[4.0, 1.0, 1.0, 2.0], None).unwrap();
        let q = soap.basis[0].as_ref().unwrap();
        assert_eq!(&initial[..4], &[q[(0, 0)], q[(0, 1)], q[(1, 0)], q[(1, 1)]]);
        soap.second_moment = vec![0.0, 1.0];
        soap.refresh_basis();
        let refreshed = Soap::gpu_basis(&[4.0, 1.0, 1.0, 2.0], Some(&initial[..4])).unwrap();
        let q = soap.basis[0].as_ref().unwrap();
        assert_eq!(
            &refreshed[..4],
            &[q[(0, 0)], q[(0, 1)], q[(1, 0)], q[(1, 1)]]
        );
        assert_eq!(&refreshed[4..], &soap.second_moment);
        assert!(Soap::gpu_basis(&[f32::NAN], None).is_err());
        assert!(Soap::gpu_basis(&[1.0, 2.0], None).is_err());
    }
}
