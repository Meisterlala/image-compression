//! Reference intra training orchestration: 5→2→1 warm-up, SOAP, clipping,
//! cosine schedules, validation/best checkpoints, hard-round refinement and
//! full greedy neural RD quantization. Each step yields at an iteration/trial.
use super::{
    Tensor,
    architecture::{Architecture, OperatingPoint},
    codec,
    network::Network,
    parameters::Chunk,
    preset::{Intra, Phase},
    quantization::Search,
    quantizer::Quantizer,
    random::Random,
    soap::{Config, Soap},
};
use crate::{Error, Result};

#[derive(Clone, Debug)]
pub struct Settings {
    pub point: OperatingPoint,
    pub iterations: usize,
    pub lambda: f32,
    pub seed: u64,
}
impl Default for Settings {
    fn default() -> Self {
        Self {
            point: OperatingPoint::Lop,
            iterations: 2000,
            lambda: 0.001,
            seed: 18101995,
        }
    }
}

#[derive(Clone)]
struct Snapshot {
    groups: [Vec<f32>; 8],
    latents: Vec<Tensor>,
}
impl Snapshot {
    fn capture(network: &Network) -> Self {
        Self {
            groups: network.parameters().0,
            latents: network.latents.clone(),
        }
    }
    fn restore(&self, network: &mut Network) -> Result<()> {
        network.set_parameters(&self.groups)?;
        network.latents.clone_from(&self.latents);
        Ok(())
    }
}

struct Optimizers {
    chunks: Vec<Chunk>,
    neural: Vec<Option<Soap>>,
    latents: Vec<Soap>,
}
impl Optimizers {
    fn new(network: &Network, learning_rate: f64) -> Result<Self> {
        let (_, chunks) = network.parameters();
        let neural = chunks
            .iter()
            .map(|chunk| {
                if chunk.trainable {
                    Soap::new(chunk.shape.clone(), Config::network(learning_rate)).map(Some)
                } else {
                    Ok(None)
                }
            })
            .collect::<Result<_>>()?;
        let latents = network
            .latents
            .iter()
            .map(|l| Soap::new(vec![1, 1, l.height, l.width], Config::latent(learning_rate)))
            .collect::<Result<_>>()?;
        Ok(Self {
            chunks,
            neural,
            latents,
        })
    }

    fn update(
        &mut self,
        network: &mut Network,
        gradient: &super::network::Gradient,
        learning_rate: f64,
    ) -> Result<()> {
        self.update_parameters(
            network,
            gradient.parameter_groups(network),
            &gradient.latents,
            learning_rate,
        )
    }

    fn update_parameters(
        &mut self,
        network: &mut Network,
        mut gradients: [Vec<f32>; 8],
        latent_gradients: &[Tensor],
        learning_rate: f64,
    ) -> Result<()> {
        let (mut groups, _) = network.parameters();
        // clip_grad_norm_(weight_parameters, 0.1), excluding latents, frozen
        // output transforms and unused filter biases (which have no gradient).
        let mut squared_norm = 0.0_f32;
        for chunk in self.chunks.iter().filter(|c| c.trainable) {
            let values = &gradients[chunk.group][chunk.offset..chunk.offset + chunk.len()];
            let norm = values.iter().map(|v| v * v).sum::<f32>().sqrt();
            squared_norm += norm * norm;
        }
        let coefficient = (0.1 / (squared_norm.sqrt() + 1e-6)).min(1.0);
        for (chunk, optimizer) in self.chunks.iter().zip(&mut self.neural) {
            if let Some(optimizer) = optimizer {
                let values = &mut gradients[chunk.group][chunk.offset..chunk.offset + chunk.len()];
                for value in values.iter_mut() {
                    *value *= coefficient;
                }
                optimizer.set_learning_rate(learning_rate)?;
                optimizer.update(
                    &mut groups[chunk.group][chunk.offset..chunk.offset + chunk.len()],
                    values,
                )?;
            }
        }
        network.set_parameters(&groups)?;
        for ((latent, gradient), optimizer) in network
            .latents
            .iter_mut()
            .zip(latent_gradients)
            .zip(&mut self.latents)
        {
            optimizer.set_learning_rate(learning_rate)?;
            optimizer.update(&mut latent.data, &gradient.data)?;
        }
        Ok(())
    }
}

struct Runner {
    network: Network,
    phase: Phase,
    iteration: usize,
    validations: usize,
    last_record: usize,
    best: Snapshot,
    best_loss: f32,
    optimizers: Optimizers,
}
impl Runner {
    fn new(network: Network, phase: Phase, target: &Tensor, lambda: f32) -> Result<Self> {
        let best_loss = network.evaluate(target, lambda)?.loss;
        let best = Snapshot::capture(&network);
        let optimizers = Optimizers::new(&network, phase.learning_rate)?;
        Ok(Self {
            network,
            phase,
            iteration: 0,
            validations: 0,
            last_record: 0,
            best,
            best_loss,
            optimizers,
        })
    }
    #[cfg(all(feature = "wasm", target_arch = "wasm32"))]
    fn gpu_batch(&mut self, target: &Tensor, lambda: f32, random: &mut Random) -> Result<usize> {
        if !super::graph::available() {
            return Ok(0);
        }
        // Do not cross validation, schedule, early-restore or phase boundaries.
        let count = 8
            .min(self.phase.iterations - self.iteration)
            .min(self.phase.validation_frequency - self.iteration % self.phase.validation_frequency)
            .min(
                self.phase
                    .patience
                    .saturating_sub(self.iteration - self.last_record)
                    + 1,
            );
        let (temperature, sigma) = self.phase.scheduled_values(self.iteration);
        let mut rng = random.clone();
        let n: usize = self.network.latents.iter().map(|l| l.data.len()).sum();
        let groups = self.network.parameters().0;
        let mut state: Vec<f32> = groups
            .iter()
            .flatten()
            .copied()
            .chain(
                self.network
                    .latents
                    .iter()
                    .flat_map(|l| l.data.iter().copied()),
            )
            .collect();
        let parameters = state.len();
        let mut offsets = [0; 8];
        for i in 1..8 {
            offsets[i] = offsets[i - 1] + groups[i - 1].len();
        }
        let mut tensors = Vec::new();
        for (chunk, optimizer) in self.optimizers.chunks.iter().zip(&self.optimizers.neural) {
            if let Some(optimizer) = optimizer {
                tensors.push(optimizer.gpu_pack(
                    offsets[chunk.group] + chunk.offset,
                    false,
                    &mut state,
                ));
            }
        }
        let mut offset = groups.iter().map(Vec::len).sum();
        for (latent, optimizer) in self.network.latents.iter().zip(&self.optimizers.latents) {
            tensors.push(optimizer.gpu_pack(offset, true, &mut state));
            offset += latent.data.len();
        }
        // Include fresh target/common uploads and the returned optimizer state
        // when fitting the bridge's 256 MiB budget. Large jobs use fewer steps.
        let fixed = 2 * state.len()
            + target.data.len()
            + self
                .network
                .common
                .iter()
                .map(|l| l.data.len())
                .sum::<usize>();
        let capacity = (256 * 1024 * 1024 / 4usize).saturating_sub(256 + fixed);
        let count = count.min(capacity / n);
        if count == 0 {
            return Ok(0);
        }
        let mut noise = Vec::with_capacity(n * count);
        for _ in 0..count {
            if matches!(self.phase.quantizer, Quantizer::Hard) {
                noise.resize(noise.len() + n, 0.0);
            } else {
                noise.extend(rng.normal_values(n)?.into_iter().map(|v| v * sigma));
            }
        }
        let step = self.optimizers.latents[0].gpu_step();
        let kind = match self.phase.quantizer {
            Quantizer::None => 0.0,
            Quantizer::SoftAlone => 1.0,
            Quantizer::Soft => 2.0,
            Quantizer::StraightThrough => 3.0,
            Quantizer::Hard => 4.0,
        };
        let normalizer = if matches!(self.phase.quantizer, Quantizer::Soft | Quantizer::SoftAlone) {
            (1.0 / (2.0 * temperature)).tanh()
        } else {
            1.0
        };
        let options = [
            6.0,
            temperature,
            lambda,
            normalizer,
            kind,
            self.phase.learning_rate_at(self.validations) as f32,
            step as f32,
            count as f32,
        ];
        let layout =
            serde_json::json!({"size":state.len(),"parameters":parameters,"tensors":tensors});
        let Some(values) =
            super::graph::training_batch(&self.network, target, &state, &noise, &options, layout)
        else {
            // Failed batches are transactional: CPU parameters, moments and RNG
            // still describe the last completed batch, so CPU fallback is safe.
            return Ok(0);
        };
        let mut cursor = 0;
        let next_groups = std::array::from_fn(|i| {
            let begin = cursor;
            cursor += groups[i].len();
            values[begin..cursor].to_vec()
        });
        self.network.set_parameters(&next_groups)?;
        for latent in &mut self.network.latents {
            let end = cursor + latent.data.len();
            latent.data.copy_from_slice(&values[cursor..end]);
            cursor = end;
        }
        let next_step = (step + count as i32) as usize;
        for optimizer in self
            .optimizers
            .neural
            .iter_mut()
            .flatten()
            .chain(&mut self.optimizers.latents)
        {
            optimizer.gpu_unpack(&values, &mut cursor, next_step);
        }
        *random = rng;
        Ok(count)
    }

    fn step(&mut self, target: &Tensor, lambda: f32, random: &mut Random) -> Result<bool> {
        if self.iteration >= self.phase.iterations {
            return Ok(true);
        }
        if self.iteration - self.last_record > self.phase.patience {
            if !self.phase.cosine {
                self.best.restore(&mut self.network)?;
                return Ok(true);
            }
            self.best.restore(&mut self.network)?;
            self.last_record = self.iteration;
        }
        #[cfg(all(feature = "wasm", target_arch = "wasm32"))]
        let batched = self.gpu_batch(target, lambda, random)?;
        #[cfg(not(all(feature = "wasm", target_arch = "wasm32")))]
        let batched = 0usize;
        if batched == 0 {
            let (temperature, sigma) = self.phase.scheduled_values(self.iteration);
            let n = self.network.latents.iter().map(|l| l.data.len()).sum();
            let noise: Vec<f32> = if matches!(self.phase.quantizer, Quantizer::Hard) {
                vec![0.0; n]
            } else {
                let mut values = random.normal_values(n)?;
                for value in &mut values {
                    *value *= sigma;
                }
                values
            };
            let learning_rate = self.phase.learning_rate_at(self.validations);
            if let Some(gradient) = super::graph::gradient(
                &self.network,
                target,
                lambda,
                self.phase.quantizer,
                temperature,
                &noise,
            ) {
                self.optimizers.update_parameters(
                    &mut self.network,
                    gradient.groups,
                    &gradient.latents,
                    learning_rate,
                )?;
            } else {
                let (_, gradient) = self.network.loss_gradient(
                    target,
                    lambda,
                    self.phase.quantizer,
                    temperature,
                    &noise,
                )?;
                self.optimizers
                    .update(&mut self.network, &gradient, learning_rate)?;
            }
        }
        self.iteration += batched.max(1);
        let count = self.iteration - 1;
        if self
            .iteration
            .is_multiple_of(self.phase.validation_frequency)
            || self.iteration == self.phase.iterations
        {
            let loss = self.network.evaluate(target, lambda)?.loss;
            if loss < self.best_loss {
                self.best_loss = loss;
                self.best = Snapshot::capture(&self.network);
                self.last_record = count;
            }
            self.validations += 1;
        }
        let done = self.iteration == self.phase.iterations;
        if done {
            self.best.restore(&mut self.network)?;
        }
        Ok(done)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stage {
    Warm1,
    Warm2,
    Core,
    Hard,
    Quantize,
    Done,
}
struct Candidate {
    network: Option<Network>,
    loss: f32,
}

pub struct Trainer {
    target: Tensor,
    settings: Settings,
    preset: Intra,
    random: Random,
    candidates: Vec<Candidate>,
    candidate: usize,
    runner: Option<Runner>,
    search: Option<Search>,
    stage: Stage,
    pub completed: usize,
    pub estimated_memory: u64,
}

impl Trainer {
    pub fn new(rgba: &[u8], width: usize, height: usize, settings: Settings) -> Result<Self> {
        let architecture = Architecture::preset(width, height, settings.point)?;
        architecture.validate()?;
        let estimated_memory = architecture.estimated_working_bytes(true);
        if estimated_memory > 3 * 1024 * 1024 * 1024 {
            return Err(Error(
                "Cool-chic reference training exceeds the 3 GiB browser memory budget. Choose a smaller Downscale image size or a lighter operating point.",
            ));
        }
        if rgba.len() != width * height * 4 {
            return Err(Error("Reference RGB input size mismatch"));
        }
        let preset = Intra::new(settings.iterations, settings.lambda, 0.01)?;
        let mut target = Tensor::zeros(3, height, width)?;
        for (i, rgba) in rgba.chunks_exact(4).enumerate() {
            for (c, &value) in rgba[..3].iter().enumerate() {
                target.data[c * width * height + i] = f32::from(value) / 255.0;
            }
        }
        let mut random = Random::new(settings.seed);
        let mut candidates = Vec::with_capacity(5);
        for _ in 0..5 {
            let mut network = Network::new(architecture.clone(), &mut random)?;
            network.initialize_output_transform(&target)?;
            candidates.push(Candidate {
                network: Some(network),
                loss: f32::INFINITY,
            });
        }
        Ok(Self {
            target,
            settings,
            preset,
            random,
            candidates,
            candidate: 0,
            runner: None,
            search: None,
            stage: Stage::Warm1,
            completed: 0,
            estimated_memory,
        })
    }

    pub fn total_iterations(&self) -> usize {
        self.preset.iterations
    }
    /// Counts for the current major stage: training, then quantization search.
    pub fn progress(&self) -> (usize, usize) {
        self.search
            .as_ref()
            .map_or((self.completed, self.preset.iterations), |search| {
                (search.tested, search.total)
            })
    }
    pub fn status(&self) -> String {
        if let Some(search) = &self.search {
            return if self.stage == Stage::Done {
                "Writing the .cool bitstream…".to_string()
            } else {
                format!("Quantizing {}…", search.module())
            };
        }
        match self.stage {
            Stage::Warm1 => format!("Warm-up 1/2 · candidate {}/5", self.candidate + 1),
            Stage::Warm2 => format!("Warm-up 2/2 · candidate {}/2", self.candidate + 1),
            Stage::Core => "Training".to_string(),
            Stage::Hard => "Hard-round refinement".to_string(),
            _ => "Finishing".to_string(),
        }
    }

    pub fn step(&mut self) -> Result<bool> {
        if self.stage == Stage::Done {
            return Ok(true);
        }
        if self.stage == Stage::Quantize {
            if self
                .search
                .as_mut()
                .unwrap()
                .step(&self.target, self.settings.lambda)?
            {
                self.stage = Stage::Done;
            }
            return Ok(self.stage == Stage::Done);
        }
        if self.runner.is_none() {
            let network = self.candidates[self.candidate]
                .network
                .take()
                .ok_or(Error("Missing reference candidate"))?;
            let phase = match self.stage {
                Stage::Warm1 | Stage::Warm2 => self.preset.warmup.clone(),
                Stage::Core => self.preset.core.clone(),
                Stage::Hard => self.preset.hard.clone(),
                _ => unreachable!(),
            };
            self.runner = Some(Runner::new(
                network,
                phase,
                &self.target,
                self.settings.lambda,
            )?);
        }
        let runner = self.runner.as_mut().unwrap();
        let before = runner.iteration;
        let done = runner.step(&self.target, self.settings.lambda, &mut self.random)?;
        self.completed += runner.iteration - before;
        if done {
            let runner = self.runner.take().unwrap();
            self.candidates[self.candidate] = Candidate {
                network: Some(runner.network),
                loss: runner.best_loss,
            };
            match self.stage {
                Stage::Warm1 | Stage::Warm2 => {
                    self.candidate += 1;
                    if self.candidate == self.candidates.len() {
                        self.candidates.sort_by(|a, b| a.loss.total_cmp(&b.loss));
                        self.candidates
                            .truncate(if self.stage == Stage::Warm1 { 2 } else { 1 });
                        self.candidate = 0;
                        self.stage = if self.stage == Stage::Warm1 {
                            Stage::Warm2
                        } else {
                            Stage::Core
                        };
                    }
                }
                Stage::Core => self.stage = Stage::Hard,
                Stage::Hard => {
                    let network = self.candidates[0].network.take().unwrap();
                    self.search = Some(Search::new(network));
                    self.candidates.clear();
                    self.stage = Stage::Quantize;
                }
                _ => unreachable!(),
            }
        }
        Ok(false)
    }

    pub fn finish(self) -> Result<Vec<u8>> {
        if self.stage != Stage::Done {
            return Err(Error("Reference training/quantization is not complete"));
        }
        let search = self.search.unwrap();
        let grids: Vec<Vec<i32>> = search
            .network
            .latents
            .iter()
            .map(|latent| {
                latent
                    .data
                    .iter()
                    .map(|&v| (v * 16.0).round_ties_even().clamp(-64.0, 63.0) as i32)
                    .collect()
            })
            .collect();
        codec::encode(&search.network, &search.descriptors, &grids)
    }
}
