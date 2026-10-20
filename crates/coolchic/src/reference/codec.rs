//! Full RGB reference range coding and reconstruction, separate from training.
use super::{
    Tensor,
    arm::neighbours,
    fixed::FixedArm,
    inter_feature::coarse_states,
    network::Network,
    stream::{self, Descriptors},
};
use crate::{Error, Image, Result};
use constriction::stream::{
    Decode, Encode,
    model::DefaultLeakyQuantizer,
    queue::{DefaultRangeDecoder, DefaultRangeEncoder},
};
use probability::distribution::Laplace;

fn run(
    network: &Network,
    descriptors: &Descriptors,
    mut symbol: impl FnMut(f64, f64, usize, usize) -> Result<i32>,
) -> Result<Vec<Tensor>> {
    let arch = &network.architecture;
    let fixed = FixedArm::new(
        &network.arm,
        descriptors.shift(0),
        descriptors.shift(1),
        arch.ifce_outputs,
        true,
    )?;
    let inter_arms = network
        .ifce
        .as_ref()
        .map(|ifce| {
            ifce.networks
                .iter()
                .map(|n| {
                    n.as_ref()
                        .map(|n| {
                            FixedArm::new(n, descriptors.shift(2), descriptors.shift(3), 0, false)
                        })
                        .transpose()
                })
                .collect::<Result<Vec<_>>>()
        })
        .transpose()?;
    let specs = arch.latent_specs();
    let mut grids: Vec<_> = specs
        .iter()
        .map(|s| Tensor::zeros(1, s.height, s.width))
        .collect::<Result<_>>()?;
    let offsets = neighbours(arch.spatial_context)?;
    let mut context = vec![0_i64; arch.spatial_context + arch.ifce_outputs];
    for index in (0..grids.len()).rev() {
        let coarse = if arch.ifce_outputs > 0 {
            Some(coarse_states(&grids, index)?.pop().unwrap())
        } else {
            None
        };
        let mut inter = Vec::new();
        if let (Some(coarse), Some(Some(fixed))) =
            (&coarse, inter_arms.as_ref().map(|arms| &arms[index]))
        {
            inter.resize(arch.ifce_outputs * coarse.height * coarse.width, 0_i64);
            let mut values = vec![0_i64; coarse.channels];
            for y in 0..coarse.height {
                for x in 0..coarse.width {
                    for (c, value) in values.iter_mut().enumerate() {
                        *value = coarse.data[coarse.index(c, y, x)] as i64;
                    }
                    let output = fixed.forward(&values)?;
                    if output.len() != arch.ifce_outputs {
                        return Err(Error("Invalid fixed IFCE output"));
                    }
                    for (c, value) in output.into_iter().enumerate() {
                        // Upstream interpolates by casting integer IFCE outputs
                        // through float32. Preserve that conversion, not f64.
                        inter[(c * coarse.height + y) * coarse.width + x] = (value as f32) as i64;
                    }
                }
            }
        }
        let width = specs[index].width;
        let height = specs[index].height;
        crate::wavefront::visit(width, height, |x, y| {
            for (c, &(dy, dx)) in offsets.iter().enumerate() {
                let iy = y as isize + dy;
                let ix = x as isize + dx;
                context[c] = if iy < 0 || ix < 0 || iy >= height as isize || ix >= width as isize {
                    0
                } else {
                    grids[index].data[iy as usize * width + ix as usize] as i64
                };
            }
            if let Some(coarse) = &coarse {
                for c in 0..arch.ifce_outputs {
                    context[arch.spatial_context + c] = if inter.is_empty() {
                        0
                    } else {
                        inter[(c * coarse.height + y / 2) * coarse.width + x / 2]
                    };
                }
            }
            let (mu, scale) = fixed.probability(&context)?;
            let value = symbol(mu, scale, index, y * width + x)?;
            grids[index].data[y * width + x] = value as f32;
            Ok(())
        })?;
    }
    Ok(grids)
}

pub fn encode_latents(
    network: &Network,
    descriptors: &Descriptors,
    grids: &[Vec<i32>],
) -> Result<Vec<u8>> {
    descriptors.validate()?;
    let specs = network.architecture.latent_specs();
    if grids.len() != specs.len()
        || grids.iter().zip(&specs).any(|(g, s)| {
            g.len() != s.height * s.width || g.iter().any(|&x| !(-64..64).contains(&x))
        })
    {
        return Err(Error(
            "Reference entropy values must be in -64..63 with matching grid sizes",
        ));
    }
    let quantizer = DefaultLeakyQuantizer::new(-64..=63);
    let mut coder = DefaultRangeEncoder::new();
    run(network, descriptors, |mu, scale, grid, index| {
        let value = grids[grid][index];
        coder
            .encode_symbol(value, quantizer.quantize(Laplace::new(mu, scale)))
            .map_err(|_| Error("Reference range encoding failed"))?;
        Ok(value)
    })?;
    let words = coder
        .into_compressed()
        .map_err(|_| Error("Reference range finalization failed"))?;
    Ok(words.iter().flat_map(|w| w.to_le_bytes()).collect())
}

pub fn decode_latents(
    network: &Network,
    descriptors: &Descriptors,
    bytes: &[u8],
) -> Result<Vec<Tensor>> {
    if bytes.is_empty() || !bytes.len().is_multiple_of(4) {
        return Err(Error("Invalid reference range data length"));
    }
    let words: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let mut coder = DefaultRangeDecoder::from_compressed(words)
        .map_err(|_| Error("Invalid reference range stream"))?;
    let quantizer = DefaultLeakyQuantizer::new(-64..=63);
    let grids = run(network, descriptors, |mu, scale, _, _| {
        coder
            .decode_symbol(quantizer.quantize(Laplace::new(mu, scale)))
            .map_err(|_| Error("Reference entropy decoding failed"))
    })?;
    let integers: Vec<_> = grids
        .iter()
        .map(|g| g.data.iter().map(|&x| x as i32).collect())
        .collect();
    if encode_latents(network, descriptors, &integers)? != bytes {
        return Err(Error("Non-canonical or truncated reference range stream"));
    }
    Ok(grids)
}

pub fn encode(network: &Network, descriptors: &Descriptors, grids: &[Vec<i32>]) -> Result<Vec<u8>> {
    let latent = encode_latents(network, descriptors, grids)?;
    stream::pack(network, descriptors, &latent)
}

pub fn decode(bytes: &[u8]) -> Result<Image> {
    let header = stream::header(bytes)?;
    let network = stream::neural_network(&header)?;
    let grids = decode_latents(&network, &header.descriptors, header.latent)?;
    let image = network.reconstruct(&grids)?;
    let pixels = image.height * image.width;
    if image.channels != 3 {
        return Err(Error("Reference output must be RGB"));
    }
    let mut rgba = Vec::with_capacity(pixels * 4);
    for index in 0..pixels {
        for c in 0..3 {
            let value = image.data[c * pixels + index];
            if value.is_nan() {
                return Err(Error("Non-finite reference reconstruction"));
            }
            rgba.push((value.clamp(0.0, 1.0) * 255.0).round_ties_even() as u8);
        }
        rgba.push(255);
    }
    Ok(Image {
        width: image.width,
        height: image.height,
        rgba,
    })
}
