//! General v5 RGB still-image headers and neural parameter serialization.
use super::{
    architecture::{Architecture, Interpolation, Layer},
    network::Network,
    random::Random,
};
use crate::{
    Error, Result,
    bits::{Reader, Writer},
};

pub const MIN_SHIFT: [i32; 8] = [-8, -16, -8, -16, -12, 0, -12, -24];
pub const MAX_Q_INDEX: [u32; 8] = [8, 16, 8, 16, 12, 0, 12, 24];

#[derive(Clone, Debug)]
pub struct Descriptors {
    pub q_index: [u32; 8],
    pub orders: [u32; 8],
}
impl Descriptors {
    pub fn validate(&self) -> Result<()> {
        if self
            .q_index
            .iter()
            .zip(MAX_Q_INDEX)
            .any(|(&index, max)| index > max)
            || self.orders.iter().any(|&order| order > 12)
        {
            return Err(Error(
                "Invalid reference neural quantization or exp-Golomb descriptor",
            ));
        }
        Ok(())
    }
    pub fn shift(&self, group: usize) -> i32 {
        MIN_SHIFT[group] + self.q_index[group] as i32
    }
    pub fn step(&self, group: usize) -> f32 {
        2.0_f32.powi(self.shift(group))
    }
}

fn finish(mut writer: Writer, offset: usize) -> Result<Vec<u8>> {
    let bytes = writer.bits.len().div_ceil(8);
    if bytes >= 65536 {
        return Err(Error("Reference header exceeds format size"));
    }
    writer.patch(offset, bytes as u32, 16);
    Ok(writer.bytes())
}

pub fn neural_bytes(network: &Network, descriptors: &Descriptors) -> Result<(Vec<u8>, usize)> {
    descriptors.validate()?;
    let (groups, _) = network.parameters();
    let mut writer = Writer::default();
    for (group, values) in groups.iter().enumerate() {
        for &value in values {
            let integer = (value / descriptors.step(group)).round_ties_even();
            if !integer.is_finite() || integer.abs() > 65535.0 {
                return Err(Error("Reference neural parameter exceeds 16-bit magnitude"));
            }
            writer.signed_golomb(integer as i32, descriptors.orders[group]);
        }
    }
    let pad = (8 - writer.bits.len() % 8) % 8;
    writer.bits.splice(0..0, std::iter::repeat_n(false, pad));
    let bytes = writer.bytes();
    if bytes.len() >= 1 << 14 {
        return Err(Error(
            "Reference neural data exceeds the 14-bit format length",
        ));
    }
    Ok((bytes, pad))
}

pub fn pack(network: &Network, descriptors: &Descriptors, latent: &[u8]) -> Result<Vec<u8>> {
    network.architecture.validate()?;
    let (neural, pad) = neural_bytes(network, descriptors)?;
    if latent.is_empty() || !latent.len().is_multiple_of(4) || latent.len() >= 1 << 28 {
        return Err(Error("Invalid reference latent payload length"));
    }
    let arch = &network.architecture;
    let mut writer = Writer::default();
    writer.put(1, 12);
    writer.put(1, 12);
    writer.put(0, 12);
    writer.put(0, 16);
    writer.put(0, 12);
    let mut output = finish(writer, 36)?;
    let mut writer = Writer::default();
    writer.put(0, 12);
    writer.put(0, 2);
    writer.put(0, 2);
    writer.put(0, 4);
    writer.put(0, 16);
    output.extend(finish(writer, 20)?);
    let mut writer = Writer::default();
    writer.put(u32::from(arch.synthesis_stabiliser), 1);
    writer.put(arch.synthesis.len() as u32, 3);
    writer.put(arch.up_kernel as u32, 4);
    writer.put(arch.pre_kernel as u32, 4);
    writer.put(arch.ifce_outputs as u32, 5);
    writer.put(arch.spatial_context as u32, 6);
    writer.put(u32::from(arch.arm_stabiliser), 1);
    writer.put(arch.arm_hidden as u32, 3);
    writer.put(arch.height as u32, 14);
    writer.put(arch.width as u32, 14);
    writer.put(arch.latent_min as u32, 4);
    writer.put(arch.latent_max as u32, 4);
    writer.put(arch.latent_specs().len() as u32, 5);
    writer.put(u32::from(arch.hyperlatent.is_some()), 1);
    writer.put(u32::from(arch.common_randomness), 1);
    writer.put(
        match arch.interpolation {
            Interpolation::Nearest => 0,
            Interpolation::Bilinear => 1,
            Interpolation::Bicubic => 2,
        },
        2,
    );
    for index in descriptors.q_index {
        writer.put(index, 5);
    }
    for order in descriptors.orders {
        writer.put(order, 4);
    }
    writer.put(neural.len() as u32, 14);
    writer.put(pad as u32, 3);
    writer.put(latent.len() as u32, 28);
    let offset = writer.bits.len();
    writer.put(0, 16);
    if let Some((min, max)) = arch.ifce_range {
        writer.put(min as u32, 4);
        writer.put(max as u32, 4);
    }
    if let Some((min, max)) = arch.hyperlatent {
        writer.put(min as u32, 4);
        writer.put(max as u32, 4);
    }
    for layer in &arch.synthesis {
        writer.put(layer.outputs as u32, 7);
        writer.put(layer.kernel as u32, 4);
        writer.put(u32::from(layer.residual), 1);
        writer.put(u32::from(layer.relu), 1);
    }
    output.extend(finish(writer, offset)?);
    output.extend(neural);
    output.extend(latent);
    Ok(output)
}

pub struct Header<'a> {
    pub architecture: Architecture,
    pub descriptors: Descriptors,
    pub neural: &'a [u8],
    pub neural_pad: usize,
    pub latent: &'a [u8],
}

pub fn header(bytes: &[u8]) -> Result<Header<'_>> {
    if bytes.len() > 32 * 1024 * 1024 {
        return Err(Error("Cool-chic file exceeds the 32 MiB browser limit"));
    }
    let mut reader = Reader::new(bytes);
    // No video/YUV masquerading as a single RGB still image.
    for (value, bits) in [(1, 12), (1, 12), (0, 12)] {
        reader.expect(value, bits)?;
    }
    let size = reader.get(16)? as usize;
    reader.expect(0, 12)?;
    let mut reader = Reader::new(reader.finish_header(size)?);
    for bits in [12, 2, 2, 4] {
        reader.expect(0, bits)?;
    }
    let size = reader.get(16)? as usize;
    let mut reader = Reader::new(reader.finish_header(size)?);
    let synthesis_stabiliser = reader.get(1)? != 0;
    let layers = reader.get(3)? as usize;
    let up_kernel = reader.get(4)? as usize;
    let pre_kernel = reader.get(4)? as usize;
    let ifce_outputs = reader.get(5)? as usize;
    let spatial_context = reader.get(6)? as usize;
    let arm_stabiliser = reader.get(1)? != 0;
    let arm_hidden = reader.get(3)? as usize;
    let height = reader.get(14)? as usize;
    let width = reader.get(14)? as usize;
    let latent_min = reader.get(4)? as usize;
    let latent_max = reader.get(4)? as usize;
    let n_latents = reader.get(5)? as usize;
    let hyper = reader.get(1)? != 0;
    let common_randomness = reader.get(1)? != 0;
    let interpolation = match reader.get(2)? {
        0 => Interpolation::Nearest,
        1 => Interpolation::Bilinear,
        2 => Interpolation::Bicubic,
        _ => return Err(Error("Invalid reference interpolation descriptor")),
    };
    let mut descriptors = Descriptors {
        q_index: [0; 8],
        orders: [0; 8],
    };
    for index in &mut descriptors.q_index {
        *index = reader.get(5)?;
    }
    for order in &mut descriptors.orders {
        *order = reader.get(4)?;
    }
    descriptors.validate()?;
    let neural_bytes = reader.get(14)? as usize;
    let neural_pad = reader.get(3)? as usize;
    let latent_bytes = reader.get(28)? as usize;
    let size = reader.get(16)? as usize;
    let ifce_range = if ifce_outputs > 0 {
        Some((reader.get(4)? as usize, reader.get(4)? as usize))
    } else {
        None
    };
    let hyperlatent = if hyper {
        Some((reader.get(4)? as usize, reader.get(4)? as usize))
    } else {
        None
    };
    let mut synthesis = Vec::with_capacity(layers);
    for _ in 0..layers {
        synthesis.push(Layer {
            outputs: reader.get(7)? as usize,
            kernel: reader.get(4)? as usize,
            residual: reader.get(1)? != 0,
            relu: reader.get(1)? != 0,
        });
    }
    let architecture = Architecture {
        height,
        width,
        latent_min,
        latent_max,
        hyperlatent,
        ifce_range,
        ifce_outputs,
        spatial_context,
        arm_hidden,
        arm_stabiliser,
        synthesis_stabiliser,
        synthesis,
        up_kernel,
        pre_kernel,
        common_randomness,
        interpolation,
    };
    architecture.validate()?;
    if n_latents != architecture.latent_specs().len() {
        return Err(Error("Inconsistent reference latent grid count"));
    }
    let data = reader.finish_header(size)?;
    if data.len() != neural_bytes + latent_bytes
        || latent_bytes == 0
        || !latent_bytes.is_multiple_of(4)
    {
        return Err(Error("Invalid reference payload lengths"));
    }
    Ok(Header {
        architecture,
        descriptors,
        neural: &data[..neural_bytes],
        neural_pad,
        latent: &data[neural_bytes..],
    })
}

pub fn neural_network(header: &Header<'_>) -> Result<Network> {
    let mut network = Network::new(header.architecture.clone(), &mut Random::new(1))?;
    let (mut groups, _) = network.parameters();
    if groups.iter().map(Vec::len).sum::<usize>() + header.neural_pad > header.neural.len() * 8 {
        return Err(Error(
            "Reference neural payload is too small for its architecture",
        ));
    }
    let mut reader = Reader::new(header.neural);
    reader.expect(0, header.neural_pad)?;
    for (group, values) in groups.iter_mut().enumerate() {
        for value in values {
            let integer = reader.signed_golomb(header.descriptors.orders[group])?;
            if integer.unsigned_abs() > 1 << 20 {
                return Err(Error("Reference neural parameter magnitude exceeds limit"));
            }
            *value = integer as f32 * header.descriptors.step(group);
        }
    }
    if reader.pos != header.neural.len() * 8 {
        return Err(Error("Unexpected trailing reference neural parameters"));
    }
    network.set_parameters(&groups)?;
    Ok(network)
}
