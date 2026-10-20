//! CPU-only, reduced-profile Cool-chic 5.0 codec. See README.md for limitations.
#![forbid(unsafe_code)]
mod bits;
mod entropy;
mod model;
mod stream;
mod train;
pub use train::Encoder;

/// Deliberately bounded first browser profile, independent of other codec limits.
pub const MAX_PIXELS: usize = 4_000_000;
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Error(pub &'static str);
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}
impl std::error::Error for Error {}
pub type Result<T> = std::result::Result<T, Error>;

fn check_size(width: usize, height: usize) -> Result<()> {
    if width == 0
        || height == 0
        || width >= 16384
        || height >= 16384
        || width.checked_mul(height).is_none_or(|n| n > MAX_PIXELS)
    {
        return Err(Error(
            "Cool-chic supports up to 4 million pixels and dimensions below 16384. Choose a smaller Downscale image size.",
        ));
    }
    Ok(())
}

pub struct Image {
    pub width: usize,
    pub height: usize,
    pub rgba: Vec<u8>,
}
/// Reads just the declared dimensions and validates this implementation's profile.
/// Does not allocate image-size buffers or decode the entropy stream.
pub fn dimensions(bytes: &[u8]) -> Result<(usize, usize)> {
    let (m, _) = stream::unpack(bytes)?;
    Ok((m.width, m.height))
}
pub fn decode(bytes: &[u8]) -> Result<Image> {
    if bytes.len() > 32 * 1024 * 1024 {
        return Err(Error(
            "Cool-chic bitstream exceeds the 32 MiB profile limit",
        ));
    }
    let (model, latent) = stream::unpack(bytes)?;
    let grids = entropy::decode(&model, latent)?;
    Ok(Image {
        width: model.width,
        height: model.height,
        rgba: model.render(&grids),
    })
}
pub fn encode(
    rgba: &[u8],
    width: usize,
    height: usize,
    quality: u32,
    effort: u32,
) -> Result<Vec<u8>> {
    let mut encoder = Encoder::new(rgba, width, height, quality, effort)?;
    while !encoder.step() {}
    encoder.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    fn pixels(w: usize, h: usize) -> Vec<u8> {
        (0..w * h)
            .flat_map(|p| {
                [
                    (p % w * 255 / w) as u8,
                    (p / w * 255 / h) as u8,
                    ((p * 17) % 256) as u8,
                    255,
                ]
            })
            .collect()
    }
    #[test]
    fn real_roundtrips_and_determinism() {
        for (w, h) in [(1, 1), (2, 3), (9, 11), (10, 1), (13, 17), (32, 24)] {
            let rgba = pixels(w, h);
            let bytes = encode(&rgba, w, h, 75, 1).unwrap();
            assert_eq!(dimensions(&bytes).unwrap(), (w, h));
            let image = decode(&bytes).unwrap();
            assert_eq!(image.rgba.len(), rgba.len());
            assert!(image.rgba.chunks_exact(4).all(|p| p[3] == 255));
            assert_eq!(encode(&rgba, w, h, 75, 1).unwrap(), bytes);
        }
    }
    #[test]
    fn constant_colors_and_quality() {
        for rgb in [[0, 0, 0, 255], [255, 255, 255, 255], [180, 33, 60, 255]] {
            let source = rgb.repeat(64);
            let image = decode(&encode(&source, 8, 8, 100, 2).unwrap()).unwrap();
            assert!(
                source
                    .iter()
                    .zip(image.rgba)
                    .all(|(&a, b)| (i32::from(a) - i32::from(b)).abs() <= 2)
            );
        }
    }
    #[test]
    fn invalid_inputs_and_incomplete_training() {
        assert!(Encoder::new(&[], 0, 1, 75, 1).is_err());
        assert!(Encoder::new(&[], 4000, 4000, 75, 1).is_err());
        assert!(Encoder::new(&[0; 4], 1, 1, 0, 1).is_err());
        assert!(Encoder::new(&[0; 4], 1, 1, 75, 10).is_err());
        assert!(
            Encoder::new(&[0; 4], 1, 1, 75, 1)
                .unwrap()
                .finish()
                .is_err()
        );
        let bytes = encode(&pixels(13, 17), 13, 17, 75, 1).unwrap();
        for end in 0..bytes.len() {
            assert!(
                decode(&bytes[..end]).is_err(),
                "accepted truncated stream at {end}"
            );
        }
        let mut trailing = bytes.clone();
        trailing.push(0);
        assert!(decode(&trailing).is_err());
        let mut unsupported = bytes;
        unsupported[0] ^= 128;
        assert!(decode(&unsupported).is_err());
    }
}
