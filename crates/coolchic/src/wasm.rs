//! Thin ownership-only browser bindings. No browser APIs in the codec itself.
use wasm_bindgen::prelude::*;
fn error(e: crate::Error) -> JsError {
    JsError::new(e.0)
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
        quality: u32,
        effort: u32,
    ) -> Result<CoolChicEncoder, JsError> {
        crate::Encoder::new(rgba, width, height, quality, effort)
            .map(Self)
            .map_err(error)
    }
    pub fn step(&mut self) -> bool {
        self.0.step()
    }
    #[wasm_bindgen(getter)]
    pub fn completed(&self) -> u32 {
        self.0.completed()
    }
    #[wasm_bindgen(getter)]
    pub fn passes(&self) -> u32 {
        self.0.passes()
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
