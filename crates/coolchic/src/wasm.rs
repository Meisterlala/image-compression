//! Thin browser bindings over the same reference trainer/decoder as native tests.
use crate::reference::architecture::{Architecture, OperatingPoint};
use wasm_bindgen::prelude::*;

fn error(e: crate::Error) -> JsError {
    JsError::new(e.0)
}
fn point(value: &str) -> Result<OperatingPoint, JsError> {
    match value {
        "lop" => Ok(OperatingPoint::Lop),
        "mop" => Ok(OperatingPoint::Mop),
        "hop" => Ok(OperatingPoint::Hop),
        "vhop" => Ok(OperatingPoint::Vhop),
        _ => Err(JsError::new("Unknown Cool-chic operating point")),
    }
}

#[wasm_bindgen(js_name = estimateCoolChicMemory)]
pub fn estimate_memory(width: usize, height: usize, operating_point: &str) -> Result<f64, JsError> {
    let architecture =
        Architecture::preset(width, height, point(operating_point)?).map_err(error)?;
    architecture.validate().map_err(error)?;
    Ok(architecture.estimated_working_bytes(true) as f64)
}

#[wasm_bindgen]
pub struct CoolChicEncoder(crate::Encoder);
#[wasm_bindgen]
impl CoolChicEncoder {
    #[wasm_bindgen(constructor)]
    pub fn new(
        rgba: &[u8],
        width: usize,
        height: usize,
        operating_point: &str,
        iterations: u32,
        lambda: f32,
    ) -> Result<CoolChicEncoder, JsError> {
        let settings = crate::EncoderSettings {
            point: point(operating_point)?,
            iterations: iterations as usize,
            lambda,
            ..crate::EncoderSettings::default()
        };
        crate::Encoder::new(rgba, width, height, settings)
            .map(Self)
            .map_err(error)
    }
    pub fn step(&mut self) -> Result<bool, JsError> {
        self.0.step().map_err(error)
    }
    #[wasm_bindgen(getter)]
    pub fn completed(&self) -> u32 {
        self.0.completed as u32
    }
    #[wasm_bindgen(getter)]
    pub fn iterations(&self) -> u32 {
        self.0.total_iterations() as u32
    }
    #[wasm_bindgen(getter, js_name = progressCompleted)]
    pub fn progress_completed(&self) -> u32 {
        self.0.progress().0 as u32
    }
    #[wasm_bindgen(getter, js_name = progressTotal)]
    pub fn progress_total(&self) -> u32 {
        self.0.progress().1 as u32
    }
    #[wasm_bindgen(getter)]
    pub fn status(&self) -> String {
        self.0.status()
    }
    pub fn finish(self) -> Result<Vec<u8>, JsError> {
        self.0.finish().map_err(error)
    }
}

/// Equal-input diagnostics for full-graph CPU/GPU parity and throughput. This
/// uses the real reference network, not a reduced benchmark architecture.
#[wasm_bindgen]
pub struct CoolChicGraphProblem {
    network: crate::reference::network::Network,
    target: crate::reference::Tensor,
    noise: Vec<f32>,
}
#[wasm_bindgen]
impl CoolChicGraphProblem {
    #[wasm_bindgen(constructor)]
    pub fn new(
        rgba: &[u8],
        width: usize,
        height: usize,
        operating_point: &str,
    ) -> Result<CoolChicGraphProblem, JsError> {
        use crate::reference::{Tensor, network::Network, random::Random};
        let architecture =
            Architecture::preset(width, height, point(operating_point)?).map_err(error)?;
        architecture.validate().map_err(error)?;
        if architecture.estimated_working_bytes(true) > 3 * 1024 * 1024 * 1024 {
            return Err(JsError::new("Graph problem exceeds browser memory budget"));
        }
        if rgba.len() != width * height * 4 {
            return Err(JsError::new("RGB input length mismatch"));
        }
        let mut target = Tensor::zeros(3, height, width).map_err(error)?;
        for (i, pixel) in rgba.chunks_exact(4).enumerate() {
            for c in 0..3 {
                target.data[c * width * height + i] = f32::from(pixel[c]) / 255.0;
            }
        }
        let mut random = Random::new(18101995);
        let mut network = Network::new(architecture, &mut random).map_err(error)?;
        network
            .initialize_output_transform(&target)
            .map_err(error)?;
        let mut noise = random
            .normal_values(network.latents.iter().map(|l| l.data.len()).sum())
            .map_err(error)?;
        for value in &mut noise {
            *value *= 0.22;
        }
        Ok(Self {
            network,
            target,
            noise,
        })
    }
    pub fn descriptor(&self) -> String {
        crate::reference::graph::descriptor(&self.network)
    }
    pub fn parameters(&self) -> Vec<f32> {
        self.network.parameters().0.into_iter().flatten().collect()
    }
    pub fn latents(&self) -> Vec<f32> {
        self.network
            .latents
            .iter()
            .flat_map(|l| l.data.iter().copied())
            .collect()
    }
    pub fn common(&self) -> Vec<f32> {
        self.network
            .common
            .iter()
            .flat_map(|l| l.data.iter().copied())
            .collect()
    }
    pub fn target(&self) -> Vec<f32> {
        self.target.data.clone()
    }
    pub fn noise(&self) -> Vec<f32> {
        self.noise.clone()
    }
    #[wasm_bindgen(js_name = cpuLoss)]
    pub fn cpu_loss(&self, lambda: f32) -> Result<f32, JsError> {
        Ok(self
            .network
            .evaluate(&self.target, lambda)
            .map_err(error)?
            .loss)
    }
    #[wasm_bindgen(js_name = cpuGradient)]
    pub fn cpu_gradient(
        &self,
        mode: u32,
        temperature: f32,
        lambda: f32,
    ) -> Result<Vec<f32>, JsError> {
        use crate::reference::quantizer::Quantizer;
        let quantizer = match mode {
            0 => Quantizer::None,
            1 => Quantizer::SoftAlone,
            2 => Quantizer::Soft,
            3 => Quantizer::StraightThrough,
            4 => Quantizer::Hard,
            _ => return Err(JsError::new("Invalid quantizer")),
        };
        let (_, gradient) = self
            .network
            .loss_gradient(&self.target, lambda, quantizer, temperature, &self.noise)
            .map_err(error)?;
        Ok(crate::reference::graph::pack_reference_gradient(
            &self.network,
            &gradient,
        ))
    }
}

#[wasm_bindgen]
pub struct CoolChicImage(crate::Image);
#[wasm_bindgen]
impl CoolChicImage {
    #[wasm_bindgen(getter)]
    pub fn width(&self) -> usize {
        self.0.width
    }
    #[wasm_bindgen(getter)]
    pub fn height(&self) -> usize {
        self.0.height
    }
    pub fn rgba(&self) -> Vec<u8> {
        self.0.rgba.clone()
    }
}
#[wasm_bindgen(js_name = decodeCoolChic)]
pub fn decode(bytes: &[u8]) -> Result<CoolChicImage, JsError> {
    crate::decode(bytes).map(CoolChicImage).map_err(error)
}
