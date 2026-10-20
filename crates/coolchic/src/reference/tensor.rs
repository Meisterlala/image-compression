use crate::{Error, Result};

/// A batch-one CHW tensor, matching the reference codec's per-image networks.
#[derive(Clone, Debug)]
pub struct Tensor {
    pub channels: usize,
    pub height: usize,
    pub width: usize,
    pub data: Vec<f32>,
}

impl Tensor {
    pub fn new(channels: usize, height: usize, width: usize, data: Vec<f32>) -> Result<Self> {
        let len = channels
            .checked_mul(height)
            .and_then(|n| n.checked_mul(width));
        if height == 0 || width == 0 || len != Some(data.len()) {
            return Err(Error("Invalid reference tensor shape"));
        }
        Ok(Self {
            channels,
            height,
            width,
            data,
        })
    }

    pub fn zeros(channels: usize, height: usize, width: usize) -> Result<Self> {
        let len = channels
            .checked_mul(height)
            .and_then(|n| n.checked_mul(width))
            .ok_or(Error("Reference tensor dimensions overflow"))?;
        let mut data = Vec::new();
        data.try_reserve_exact(len)
            .map_err(|_| Error("Insufficient memory for reference tensor"))?;
        data.resize(len, 0.0);
        Self::new(channels, height, width, data)
    }

    pub fn index(&self, channel: usize, y: usize, x: usize) -> usize {
        (channel * self.height + y) * self.width + x
    }

    pub fn same_shape(&self, other: &Self) -> bool {
        (self.channels, self.height, self.width) == (other.channels, other.height, other.width)
    }

    pub fn add_assign(&mut self, other: &Self) -> Result<()> {
        if !self.same_shape(other) {
            return Err(Error("Reference tensor addition shape mismatch"));
        }
        for (a, b) in self.data.iter_mut().zip(&other.data) {
            *a += b;
        }
        Ok(())
    }

    pub fn first_channels(&self, channels: usize) -> Result<Self> {
        if channels > self.channels {
            return Err(Error("Invalid reference channel slice"));
        }
        Self::new(
            channels,
            self.height,
            self.width,
            self.data[..channels * self.height * self.width].to_vec(),
        )
    }
}
