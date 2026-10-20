//! Reference constriction queue coder, 24-bit leaky Laplace probabilities,
//! little-endian u32 words, and the 9x9 ARM wavefront ordering.
use crate::{
    Error, Result,
    model::{Model, context},
};
use constriction::stream::{
    Decode, Encode,
    model::DefaultLeakyQuantizer,
    queue::{DefaultRangeDecoder, DefaultRangeEncoder},
};
use probability::distribution::Laplace;

pub(crate) fn visit(width: usize, height: usize, mut f: impl FnMut(usize, usize) -> Result<()>) -> Result<()> {
    if width <= 9 {
        for y in 0..height {
            for x in 0..width {
                f(x, y)?;
            }
        }
    } else {
        for diagonal in 0..width + (height - 1) * 10 {
            let first = if diagonal < width {
                0
            } else {
                (diagonal - width) / 10 + 1
            };
            let last = (diagonal / 10).min(height - 1);
            for y in first..=last {
                f(diagonal - y * 10, y)?;
            }
        }
    }
    Ok(())
}

pub(crate) fn encode(model: &Model, grids: &[Vec<i32>; 3]) -> Result<Vec<u8>> {
    let quantizer = DefaultLeakyQuantizer::new(-64..=63);
    let mut coder = DefaultRangeEncoder::new();
    for i in (0..3).rev() {
        let (w, h) = model.size(i);
        visit(w, h, |x, y| {
            let (mu, scale) = model.probability(context(&grids[i], w, h, x, y));
            coder
                .encode_symbol(
                    grids[i][y * w + x],
                    quantizer.quantize(Laplace::new(mu, scale)),
                )
                .map_err(|_| Error("Cool-chic range encoding failed"))
        })?;
    }
    let words = coder
        .into_compressed()
        .map_err(|_| Error("Cool-chic range finalization failed"))?;
    Ok(words.iter().flat_map(|w| w.to_le_bytes()).collect())
}

pub(crate) fn decode(model: &Model, bytes: &[u8]) -> Result<[Vec<i32>; 3]> {
    let words: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let mut coder = DefaultRangeDecoder::from_compressed(words)
        .map_err(|_| Error("Invalid Cool-chic range stream"))?;
    let quantizer = DefaultLeakyQuantizer::new(-64..=63);
    let mut grids: [Vec<i32>; 3] = std::array::from_fn(|i| {
        let (w, h) = model.size(i);
        vec![0; w * h]
    });
    for i in (0..3).rev() {
        let (w, h) = model.size(i);
        visit(w, h, |x, y| {
            let (mu, scale) = model.probability(context(&grids[i], w, h, x, y));
            grids[i][y * w + x] = coder
                .decode_symbol(quantizer.quantize(Laplace::new(mu, scale)))
                .map_err(|_| Error("Invalid Cool-chic entropy data"))?;
            Ok(())
        })?;
    }
    // Range decoders may implicitly extend a stream with zeros. Require canonical
    // re-encoding to reject truncated or extra payload words, not display garbage.
    if encode(model, &grids)? != bytes {
        return Err(Error("Non-canonical or truncated Cool-chic entropy data"));
    }
    Ok(grids)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn wavefront_visits_every_cell_once() {
        for w in [1, 2, 8, 9, 10, 11, 19, 31] {
            for h in [1, 2, 5, 17] {
                let mut seen = vec![false; w * h];
                visit(w, h, |x, y| {
                    assert!(x < w && y < h);
                    assert!(!seen[y * w + x]);
                    seen[y * w + x] = true;
                    Ok(())
                })
                .unwrap();
                assert!(seen.iter().all(|v| *v));
            }
        }
    }
}
