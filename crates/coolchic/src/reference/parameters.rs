//! Reference neural parameter ordering shared by the optimizer, quantization
//! search and bitstream writer. Groups are ARM/IFCE/upsampling/synthesis, each
//! with all weights then all biases, matching DescriptorCoolChic/DescriptorNN.
use super::{
    Conv2d, ConvGradient,
    arm::{Arm, ArmGradient},
    network::{Gradient, Network},
};
use crate::{Error, Result};

#[derive(Clone, Debug)]
pub struct Chunk {
    pub group: usize,
    pub offset: usize,
    pub shape: Vec<usize>,
    pub trainable: bool,
}
impl Chunk {
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

fn add(
    groups: &mut [Vec<f32>; 8],
    chunks: &mut Vec<Chunk>,
    group: usize,
    values: &[f32],
    shape: Vec<usize>,
    trainable: bool,
) {
    chunks.push(Chunk {
        group,
        offset: groups[group].len(),
        shape,
        trainable,
    });
    groups[group].extend_from_slice(values);
}
fn conv(
    groups: &mut [Vec<f32>; 8],
    chunks: &mut Vec<Chunk>,
    group: usize,
    layer: &Conv2d,
    linear: bool,
    trainable: bool,
) {
    let shape = if linear {
        vec![layer.output_channels, layer.input_channels]
    } else {
        vec![
            layer.output_channels,
            layer.input_channels,
            layer.kernel,
            layer.kernel,
        ]
    };
    add(groups, chunks, group, &layer.weight, shape, trainable);
    add(
        groups,
        chunks,
        group + 1,
        &layer.bias,
        vec![layer.output_channels],
        trainable,
    );
}
fn arm(groups: &mut [Vec<f32>; 8], chunks: &mut Vec<Chunk>, group: usize, network: &Arm) {
    for layer in &network.layers {
        conv(groups, chunks, group, layer, true, true);
    }
    if let Some(layer) = &network.stabiliser {
        conv(groups, chunks, group, layer, true, true);
    }
}
fn conv_grad(groups: &mut [Vec<f32>; 8], group: usize, gradient: &ConvGradient) {
    groups[group].extend_from_slice(&gradient.weight);
    groups[group + 1].extend_from_slice(&gradient.bias);
}
fn arm_grad(groups: &mut [Vec<f32>; 8], group: usize, gradient: &ArmGradient) {
    for layer in &gradient.layers {
        conv_grad(groups, group, layer);
    }
    if let Some(layer) = &gradient.stabiliser {
        conv_grad(groups, group, layer);
    }
}

impl Network {
    pub fn parameters(&self) -> ([Vec<f32>; 8], Vec<Chunk>) {
        let mut groups: [Vec<f32>; 8] = std::array::from_fn(|_| Vec::new());
        let mut chunks = Vec::new();
        arm(&mut groups, &mut chunks, 0, &self.arm);
        if let Some(ifce) = &self.ifce {
            for network in ifce.networks.iter().flatten() {
                arm(&mut groups, &mut chunks, 2, network);
            }
        }
        // Per-layer reference registration order is bias then parametrized
        // weight. Filtering by weight/bias still gives the group order below.
        for (filters, biases) in [
            (&self.pyramid.upsampling, &self.up_bias),
            (&self.pyramid.preconcat, &self.pre_bias),
        ] {
            for (filter, &bias) in filters.iter().zip(biases) {
                add(
                    &mut groups,
                    &mut chunks,
                    4,
                    &filter.half,
                    vec![filter.half.len()],
                    true,
                );
                add(&mut groups, &mut chunks, 5, &[bias], vec![1], false);
            }
        }
        conv(
            &mut groups,
            &mut chunks,
            6,
            &self.synthesis.output_transform,
            false,
            false,
        );
        if let Some(layer) = &self.synthesis.stabiliser {
            conv(&mut groups, &mut chunks, 6, layer, false, true);
        }
        for layer in &self.synthesis.trunk {
            conv(&mut groups, &mut chunks, 6, layer, false, true);
        }
        (groups, chunks)
    }

    pub fn set_parameters(&mut self, groups: &[Vec<f32>; 8]) -> Result<()> {
        let (current, _) = self.parameters();
        if groups
            .iter()
            .zip(&current)
            .any(|(a, b)| a.len() != b.len() || a.iter().any(|v| !v.is_finite()))
        {
            return Err(Error("Invalid reference neural parameter groups"));
        }
        let mut offsets = [0usize; 8];
        let mut assign = |group: usize, destination: &mut [f32]| {
            let start = offsets[group];
            destination.copy_from_slice(&groups[group][start..start + destination.len()]);
            offsets[group] += destination.len();
        };
        fn layer(layer: &mut Conv2d, group: usize, assign: &mut impl FnMut(usize, &mut [f32])) {
            assign(group, &mut layer.weight);
            assign(group + 1, &mut layer.bias);
        }
        fn network(network: &mut Arm, group: usize, assign: &mut impl FnMut(usize, &mut [f32])) {
            for l in &mut network.layers {
                layer(l, group, assign);
            }
            if let Some(l) = &mut network.stabiliser {
                layer(l, group, assign);
            }
        }
        network(&mut self.arm, 0, &mut assign);
        if let Some(ifce) = &mut self.ifce {
            for n in ifce.networks.iter_mut().flatten() {
                network(n, 2, &mut assign);
            }
        }
        for (filters, biases) in [
            (&mut self.pyramid.upsampling, &mut self.up_bias),
            (&mut self.pyramid.preconcat, &mut self.pre_bias),
        ] {
            for (filter, bias) in filters.iter_mut().zip(biases) {
                assign(4, &mut filter.half);
                assign(5, std::slice::from_mut(bias));
            }
        }
        layer(&mut self.synthesis.output_transform, 6, &mut assign);
        if let Some(l) = &mut self.synthesis.stabiliser {
            layer(l, 6, &mut assign);
        }
        for l in &mut self.synthesis.trunk {
            layer(l, 6, &mut assign);
        }
        Ok(())
    }
}

impl Gradient {
    pub fn parameter_groups(&self, network: &Network) -> [Vec<f32>; 8] {
        let mut groups: [Vec<f32>; 8] = std::array::from_fn(|_| Vec::new());
        arm_grad(&mut groups, 0, &self.arm);
        if let Some(ifce) = &self.ifce {
            for n in ifce.networks.iter().flatten() {
                arm_grad(&mut groups, 2, n);
            }
        }
        for half in self
            .pyramid
            .upsampling
            .iter()
            .chain(&self.pyramid.preconcat)
        {
            groups[4].extend_from_slice(half);
        }
        groups[5].resize(network.up_bias.len() + network.pre_bias.len(), 0.0);
        conv_grad(&mut groups, 6, &self.synthesis.output_transform);
        if let Some(layer) = &self.synthesis.stabiliser {
            conv_grad(&mut groups, 6, layer);
        }
        for layer in &self.synthesis.trunk {
            conv_grad(&mut groups, 6, layer);
        }
        groups
    }
}
