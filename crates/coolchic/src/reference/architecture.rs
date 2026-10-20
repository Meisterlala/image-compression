//! Still-image architecture descriptors from v5.0.0 cfg/dec/intra/*.cfg and
//! CoolChicEncoderParameter. No reduced-profile substitutions.
use crate::{Error, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Interpolation {
    Nearest,
    Bilinear,
    Bicubic,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OperatingPoint {
    Lop,
    Mop,
    Hop,
    Vhop,
}
#[derive(Clone, Debug)]
pub struct Layer {
    pub outputs: usize,
    pub kernel: usize,
    pub residual: bool,
    pub relu: bool,
}
#[derive(Clone, Debug)]
pub struct Architecture {
    pub height: usize,
    pub width: usize,
    pub latent_min: usize,
    pub latent_max: usize,
    pub hyperlatent: Option<(usize, usize)>,
    pub ifce_range: Option<(usize, usize)>,
    pub ifce_outputs: usize,
    pub spatial_context: usize,
    pub arm_hidden: usize,
    pub arm_stabiliser: bool,
    pub synthesis_stabiliser: bool,
    pub synthesis: Vec<Layer>,
    pub up_kernel: usize,
    pub pre_kernel: usize,
    pub common_randomness: bool,
    pub interpolation: Interpolation,
}
#[derive(Clone, Debug)]
pub struct LatentSpec {
    pub height: usize,
    pub width: usize,
    pub hyper: bool,
}

impl Architecture {
    pub fn preset(width: usize, height: usize, point: OperatingPoint) -> Result<Self> {
        crate::check_size(width, height)?;
        let pixels = width * height;
        let max = if pixels < 1_000_000 {
            6
        } else if pixels < 3_000_000 {
            7
        } else {
            8
        };
        let (spatial, ifce, hidden) = match point {
            OperatingPoint::Lop => (6, 2, 3),
            OperatingPoint::Mop => (10, 4, 16),
            OperatingPoint::Hop => (14, 6, 48),
            OperatingPoint::Vhop => (20, 6, 64),
        };
        let mut synthesis = vec![
            Layer {
                outputs: hidden,
                kernel: 1,
                residual: false,
                relu: true,
            },
            Layer {
                outputs: 3,
                kernel: 1,
                residual: false,
                relu: false,
            },
        ];
        if point != OperatingPoint::Lop {
            synthesis.push(Layer {
                outputs: 3,
                kernel: 3,
                residual: true,
                relu: true,
            });
        }
        synthesis.push(Layer {
            outputs: 3,
            kernel: 3,
            residual: true,
            relu: false,
        });
        Ok(Self {
            height,
            width,
            latent_min: 0,
            latent_max: max,
            hyperlatent: Some((4, max)),
            ifce_range: Some((0, 2)),
            ifce_outputs: ifce,
            spatial_context: spatial,
            arm_hidden: 2,
            arm_stabiliser: true,
            synthesis_stabiliser: true,
            synthesis,
            up_kernel: 8,
            pre_kernel: 7,
            common_randomness: false,
            interpolation: Interpolation::Bicubic,
        })
    }

    pub fn validate(&self) -> Result<()> {
        crate::check_size(self.width, self.height)?;
        let valid_range = |range: (usize, usize)| range.0 <= range.1 && range.1 <= 15;
        if !valid_range((self.latent_min, self.latent_max))
            || self.hyperlatent.is_some_and(|r| !valid_range(r))
            || self.ifce_range.is_some_and(|r| !valid_range(r))
            || self.ifce_outputs > 31
            || self.spatial_context > 40
            || self.arm_hidden > 7
            || (self.ifce_range.is_none() && self.ifce_outputs != 0)
            || (self.ifce_range.is_some() && self.ifce_outputs == 0)
            || self.up_kernel < 4
            || self.up_kernel > 14
            || !self.up_kernel.is_multiple_of(2)
            || self.pre_kernel == 0
            || self.pre_kernel > 15
            || self.pre_kernel.is_multiple_of(2)
            || self.synthesis.is_empty()
            || self.synthesis.len() > 7
            || self.synthesis.last().unwrap().outputs != 3
            || (self.common_randomness && self.latent_min != 0)
        {
            return Err(Error(
                "Invalid or unsupported RGB Cool-chic reference architecture",
            ));
        }
        let mut inputs = self.synthesis_inputs();
        for layer in &self.synthesis {
            if layer.outputs == 0
                || layer.outputs > 127
                || layer.kernel == 0
                || layer.kernel > 15
                || layer.kernel.is_multiple_of(2)
                || (layer.residual && layer.outputs != inputs)
            {
                return Err(Error("Invalid reference synthesis layer descriptor"));
            }
            inputs = layer.outputs;
        }
        let specs = self.latent_specs();
        if specs
            .windows(2)
            .any(|s| s[0].height > s[1].height * 2 || s[0].width > s[1].width * 2)
        {
            return Err(Error(
                "Reference latent hierarchy contains unsupported resolution gaps",
            ));
        }
        if specs.len() > 31 {
            return Err(Error("Too many reference latent grids"));
        }
        Ok(())
    }

    /// Conservative live-buffer estimate for browser guardrails, not a claim
    /// about measured process RSS. Call only after validating this descriptor.
    pub fn estimated_working_bytes(&self, training: bool) -> u64 {
        let pixels = (self.width * self.height) as u64;
        let outputs: usize = self.synthesis.iter().map(|l| l.outputs).sum();
        let contexts = self.spatial_context + self.ifce_outputs;
        let channels = 2 * outputs
            + 2 * self.synthesis_inputs()
            + 12
            + if training {
                3 * contexts * (self.arm_hidden + 1) + 4 * self.ifce_outputs + 20
            } else {
                2 * self.ifce_outputs
            };
        let latent_values: u64 = self
            .latent_specs()
            .iter()
            .map(|s| (s.width * s.height) as u64)
            .sum();
        pixels * channels as u64 * 4 + latent_values * if training { 40 } else { 8 }
    }

    pub fn synthesis_inputs(&self) -> usize {
        (self.latent_max - self.latent_min + 1) * if self.common_randomness { 2 } else { 1 }
    }

    pub fn latent_specs(&self) -> Vec<LatentSpec> {
        let min = self
            .hyperlatent
            .map_or(self.latent_min, |r| r.0.min(self.latent_min));
        let max = self
            .hyperlatent
            .map_or(self.latent_max, |r| r.1.max(self.latent_max));
        let mut latents = Vec::new();
        for level in min..=max {
            let height = self.height.div_ceil(1 << level);
            let width = self.width.div_ceil(1 << level);
            if (self.latent_min..=self.latent_max).contains(&level) {
                latents.push(LatentSpec {
                    height,
                    width,
                    hyper: false,
                });
            }
            if self
                .hyperlatent
                .is_some_and(|r| (r.0..=r.1).contains(&level))
            {
                latents.push(LatentSpec {
                    height,
                    width,
                    hyper: true,
                });
            }
        }
        latents
    }

    pub fn ifce_inputs(&self) -> Vec<usize> {
        let specs = self.latent_specs();
        specs
            .iter()
            .enumerate()
            .map(|(index, latent)| {
                // Match upstream's image-height-derived ratio, including its
                // behavior on tiny images and duplicated hyperlatent resolutions.
                let level = (self.height as f64 / latent.height as f64).log2().ceil() as usize;
                if self
                    .ifce_range
                    .is_some_and(|r| (r.0..=r.1).contains(&level))
                {
                    (specs.len() - 1 - index).max(1)
                } else {
                    0
                }
            })
            .collect()
    }
}
