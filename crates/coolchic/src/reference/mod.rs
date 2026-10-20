//! Reference-v5 RGB intra MSE training, neural operators and bitstream I/O.
//!
//! This is the application's Cool-chic encoder and decoder.
//! Ported from Orange Cool-Chic v5.0.0 (BSD-3-Clause); see UPSTREAM-LICENSE.txt.

mod acceleration;
pub mod architecture;
pub mod arm;
pub mod codec;
pub mod convolution;
pub mod fixed;
pub mod inter_feature;
pub mod network;
pub mod parameters;
pub mod preset;
pub mod pyramid;
pub mod quantization;
pub mod quantizer;
pub mod random;
pub mod rate;
pub mod resize;
pub mod soap;
pub mod stream;
pub mod tensor;
pub mod training;
pub mod upsampling;

pub use convolution::{Conv2d, ConvGradient, Synthesis, SynthesisGradient};
pub use tensor::Tensor;
pub use upsampling::{Filter, FilterGradient, FilterMode};
