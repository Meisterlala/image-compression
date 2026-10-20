//! Reference greedy network-quantization / exp-Golomb RD search. Each step
//! tests one genuine weight/bias quantization pair, so the browser can yield.
use super::{
    Tensor,
    network::Network,
    stream::{Descriptors, MAX_Q_INDEX, MIN_SHIFT},
};
use crate::{Error, Result};

/// The reference search uses this estimator, not the exact signed-code length
/// (notably a zero at order 0 has estimated cost 0). Keep that distinction.
pub fn estimated_bits(values: &[f32], step: f32, order: u32) -> u64 {
    values
        .iter()
        .map(|&v| {
            let symbol = (v / step).round_ties_even().abs() as u64;
            let high = (2 * symbol + (1_u64 << order)).ilog2();
            u64::from(2 * (high - order) + order + u32::from(symbol != 0))
        })
        .sum()
}

pub fn best_order(values: &[f32], step: f32) -> (u32, u64) {
    (0..=12)
        .map(|order| (order, estimated_bits(values, step, order)))
        .min_by_key(|&(_, bits)| bits)
        .unwrap()
}

pub struct Search {
    pub network: Network,
    pub descriptors: Descriptors,
    pub finished: bool,
    order: Vec<usize>,
    module_index: usize,
    weight_index: u32,
    bias_index: u32,
    full_precision: [Vec<f32>; 8],
    best_pair: Option<(u32, u32, u32, u32)>,
    best_loss: f32,
    quantized: [bool; 4],
    pub tested: usize,
    pub total: usize,
}

impl Search {
    pub fn new(network: Network) -> Self {
        let order: Vec<usize> = [0, 1, 3, 2]
            .into_iter()
            .filter(|&m| m != 1 || network.ifce.is_some())
            .collect();
        let (full_precision, _) = network.parameters();
        let total = order
            .iter()
            .map(|&m| (MAX_Q_INDEX[m * 2] + 1) as usize * (MAX_Q_INDEX[m * 2 + 1] + 1) as usize)
            .sum();
        Self {
            network,
            descriptors: Descriptors {
                q_index: [0; 8],
                orders: [0; 8],
            },
            finished: false,
            order,
            module_index: 0,
            weight_index: 0,
            bias_index: 0,
            full_precision,
            best_pair: None,
            best_loss: f32::INFINITY,
            quantized: [false; 4],
            tested: 0,
            total,
        }
    }
    pub fn module(&self) -> &'static str {
        if self.finished {
            return "complete";
        }
        ["ARM", "IFCE", "upsampling", "synthesis"][self.order[self.module_index]]
    }

    fn pair(&self, module: usize, weight: u32, bias: u32) -> Option<[Vec<f32>; 8]> {
        let (mut groups, _) = self.network.parameters();
        for (group, index) in [(module * 2, weight), (module * 2 + 1, bias)] {
            let step = 2.0_f32.powi(MIN_SHIFT[group] + index as i32);
            let values: Vec<_> = self.full_precision[group]
                .iter()
                .map(|&v| (v / step).round_ties_even())
                .collect();
            if values.iter().any(|v| !v.is_finite() || v.abs() > 65535.0) {
                return None;
            }
            groups[group] = values.iter().map(|v| v * step).collect();
        }
        Some(groups)
    }

    pub fn step(&mut self, target: &Tensor, lambda: f32) -> Result<bool> {
        if self.finished {
            return Ok(true);
        }
        let module = self.order[self.module_index];
        let w = module * 2;
        let b = w + 1;
        if let Some(groups) = self.pair(module, self.weight_index, self.bias_index) {
            self.network.set_parameters(&groups)?;
            let evaluation_loss = match super::graph::evaluation(&self.network, target, lambda) {
                Some(loss) => loss,
                None => self.network.evaluate(target, lambda)?.loss,
            };
            let w_step = 2.0_f32.powi(MIN_SHIFT[w] + self.weight_index as i32);
            let b_step = 2.0_f32.powi(MIN_SHIFT[b] + self.bias_index as i32);
            let (w_order, w_bits) = best_order(&groups[w], w_step);
            let (b_order, b_bits) = best_order(&groups[b], b_step);
            let mut bits = w_bits + b_bits;
            for (other, &quantized) in self.quantized.iter().enumerate() {
                if quantized && other != module {
                    for group in [other * 2, other * 2 + 1] {
                        bits += estimated_bits(
                            &groups[group],
                            self.descriptors.step(group),
                            self.descriptors.orders[group],
                        );
                    }
                }
            }
            let loss =
                evaluation_loss + lambda * bits as f32 / (target.width * target.height) as f32;
            if loss.is_finite() && loss < self.best_loss {
                self.best_loss = loss;
                self.best_pair = Some((self.weight_index, self.bias_index, w_order, b_order));
            }
        }
        self.tested += 1;
        self.bias_index += 1;
        if self.bias_index > MAX_Q_INDEX[b] {
            self.bias_index = 0;
            self.weight_index += 1;
        }
        if self.weight_index > MAX_Q_INDEX[w] {
            let (wi, bi, wo, bo) = self
                .best_pair
                .ok_or(Error("No valid reference neural quantization candidate"))?;
            let groups = self
                .pair(module, wi, bi)
                .ok_or(Error("Selected neural quantization exceeds bounds"))?;
            self.network.set_parameters(&groups)?;
            self.descriptors.q_index[w] = wi;
            self.descriptors.q_index[b] = bi;
            self.descriptors.orders[w] = wo;
            self.descriptors.orders[b] = bo;
            self.quantized[module] = true;
            self.module_index += 1;
            self.weight_index = 0;
            self.bias_index = 0;
            self.best_pair = None;
            self.best_loss = f32::INFINITY;
            if self.module_index == self.order.len() {
                self.finished = true;
            }
        }
        Ok(self.finished)
    }
}
