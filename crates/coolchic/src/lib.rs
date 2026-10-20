//! Browser-portable Cool-chic v5 RGB still-image codec and reference MSE trainer.
//! No server, Python or pretrained weights at runtime. See README.md.
#![forbid(unsafe_code)]
mod bits;
pub mod reference;
#[cfg(feature = "wasm")]
mod wasm;
mod wavefront;
#[cfg(all(feature = "threads", target_arch = "wasm32"))]
pub use wasm_bindgen_rayon::init_thread_pool;
pub use reference::training::{Settings as EncoderSettings, Trainer as Encoder};

/// Browser guardrail, separate from the reference format's dimension fields.
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

/// Validate the header and declared dimensions before image-size allocation.
pub fn dimensions(bytes: &[u8]) -> Result<(usize, usize)> {
    let header = reference::stream::header(bytes)?;
    Ok((header.architecture.width, header.architecture.height))
}
pub fn decode(bytes: &[u8]) -> Result<Image> {
    reference::codec::decode(bytes)
}
pub fn encode(
    rgba: &[u8],
    width: usize,
    height: usize,
    settings: EncoderSettings,
) -> Result<Vec<u8>> {
    let mut encoder = Encoder::new(rgba, width, height, settings)?;
    while !encoder.step()? {}
    encoder.finish()
}
