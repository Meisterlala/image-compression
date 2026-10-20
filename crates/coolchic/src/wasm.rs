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
