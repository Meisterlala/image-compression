//! Reference-v5 neural operators and derivatives, under construction.
//!
//! This is NOT yet the application's encoder/decoder. The legacy reduced codec
//! stays separate until reference bitstream and complete training parity pass.
//! Ported from Orange Cool-Chic v5.0.0 (BSD-3-Clause); see UPSTREAM-LICENSE.txt.

pub mod arm;
pub mod convolution;
pub mod pyramid;
pub mod quantizer;
pub mod rate;
pub mod tensor;
pub mod upsampling;

pub use convolution::{Conv2d, ConvGradient, Synthesis, SynthesisGradient};
pub use tensor::Tensor;
pub use upsampling::{Filter, FilterGradient, FilterMode};
