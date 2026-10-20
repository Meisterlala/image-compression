//! Local JPEG AI bindings and explicitly defined image-quality metrics.
#![forbid(unsafe_code)]

use serde::Serialize;
use std::{
    borrow::Cow,
    sync::{Arc, RwLock},
};
use wasm_bindgen::prelude::*;
use zenjpegai::encoder::{EncodeLimits, EncodeParams};
use zenjpegai::header::OperatingPoint;
use zenjpegai::model::ModelSource;
use zenjpegai::nn::fast::{Engine, Tier};
use zenjpegai::weights::packed::PackedBundle;
use zenjpegai::{Decoder, Encoder, RgbImage};

const MAX_PIXELS: usize = 40_000_000;
// Expensive jobs require browser confirmation above 1.5 GiB; the hard cap is 3 GiB.
const MEMORY_LIMIT: u64 = 3072 << 20;

#[derive(Clone, Default)]
struct Models(Arc<RwLock<PackedBundle>>);
impl ModelSource for Models {
    fn read(&self, path: &str) -> Result<Cow<'_, [u8]>, zenjpegai::Error> {
        let store = self
            .0
            .read()
            .map_err(|_| zenjpegai::Error::Model("model store poisoned".into()))?;
        Ok(Cow::Owned(store.read(path)?.into_owned()))
    }
}
fn error(e: impl std::fmt::Debug) -> JsError {
    JsError::new(&format!("{e:?}"))
}

#[wasm_bindgen]
pub struct JpegAi {
    op: OperatingPoint,
    models: Models,
    encoder: Encoder,
    decoder: Decoder,
}

#[wasm_bindgen]
impl JpegAi {
    #[wasm_bindgen(constructor)]
    pub fn new(operating_point: &str) -> Result<JpegAi, JsError> {
        let op = parse_operating_point(operating_point).map_err(JsError::new)?;
        let models = Models::default();
        // One CPU worker, SIMD when compiled with +simd128. No WebGPU or server fallback.
        let engine = Engine::with(Tier::detect(), false);
        zenjpegai::nn::fast::set_pool_limit(32 << 20);
        Ok(Self {
            op,
            encoder: Encoder::with_source(Box::new(models.clone()), engine).limits(
                EncodeLimits::default()
                    .with_max_pixels(MAX_PIXELS as u64)
                    .with_max_memory(MEMORY_LIMIT),
            ),
            decoder: Decoder::with_source(Box::new(models.clone()), engine)
                .operating_point(Some(op))
                .limits(
                    zenjpegai::Limits::default()
                        .with_max_pixels(MAX_PIXELS as u64)
                        .with_max_memory(MEMORY_LIMIT),
                ),
            models,
        })
    }
    #[wasm_bindgen(js_name = addModels)]
    pub fn add_models(&self, bundle: Vec<u8>) -> Result<String, JsError> {
        let mut store = self.models.0.write().map_err(error)?;
        store.add(bundle).map_err(error)?;
        Ok(store.notice().to_owned())
    }
    /// Selected SOP/BOP/HOP model with quantisation, or target-bpp search within that point.
    pub fn encode(
        &self,
        rgba: &[u8],
        width: u32,
        height: u32,
        model: u8,
        displacement: i32,
        target_bpp: f64,
    ) -> Result<Vec<u8>, JsError> {
        validate(rgba, width as usize, height as usize, MAX_PIXELS).map_err(error)?;
        if model > 3
            || !(-1069..=702).contains(&displacement)
            || !target_bpp.is_finite()
            || !(0.0..=8.0).contains(&target_bpp)
        {
            return Err(JsError::new("invalid JPEG AI settings"));
        }
        let src = RgbImage {
            width: width as usize,
            height: height as usize,
            bit_depth: 8,
            data: rgba
                .chunks_exact(4)
                .flat_map(|p| p[..3].iter().map(|&v| u16::from(v)))
                .collect(),
        };
        let params = EncodeParams {
            model_id: model,
            op: self.op,
            beta_displacement_log: [displacement; 2],
            ..Default::default()
        };
        let result = if target_bpp > 0.0 {
            self.encoder
                .encode_to_bpp(src, target_bpp, params)
                .map(|(bytes, _)| bytes)
        } else {
            self.encoder.encode(src, params)
        };
        self.encoder.release_buffers();
        let bytes = result.map_err(error)?;
        let headers = self.decoder.read_headers(&bytes).map_err(error)?;
        if headers.picture.synthesis_transforms.first() != Some(&self.op) {
            return Err(JsError::new(
                "encoded JPEG AI operating point does not match",
            ));
        }
        Ok(bytes)
    }
    pub fn decode(&self, stream: &[u8]) -> Result<Vec<u8>, JsError> {
        let image = self.decoder.decode(stream).map_err(error)?;
        let rgba = image
            .data
            .chunks_exact(3)
            .flat_map(|p| [p[0] as u8, p[1] as u8, p[2] as u8, 255])
            .collect();
        self.decoder.release_buffers();
        Ok(rgba)
    }
    #[wasm_bindgen(js_name = estimateMemory)]
    pub fn estimate_memory(&self, width: u32, height: u32, rate_matched: bool) -> f64 {
        estimate_job_memory(width.into(), height.into(), self.op, rate_matched) as f64
    }
}
impl Default for JpegAi {
    fn default() -> Self {
        Self::new("bop").expect("BOP is a supported operating point")
    }
}

fn parse_operating_point(value: &str) -> Result<OperatingPoint, &'static str> {
    match value {
        "sop" => Ok(OperatingPoint::Sop),
        "bop" => Ok(OperatingPoint::Bop),
        "hop" => Ok(OperatingPoint::Hop),
        _ => Err("JPEG AI operating point must be sop, bop or hop"),
    }
}

fn estimate_job_memory(width: u64, height: u64, op: OperatingPoint, rate_matched: bool) -> u64 {
    let encode =
        zenjpegai::encoder::estimate_encode_memory(width, height, op, rate_matched).live_bytes;
    if op != OperatingPoint::Hop {
        return encode;
    }
    // HOP synthesis can peak above analysis. The pinned encoder uses <=1024² synthesis
    // tiles, decoded sequentially. Match the upstream decoder's 970 B/tile-pixel and
    // 20 B/picture-pixel bound, with 128 MiB/model for retained bundles/encoder networks.
    let retained = (if rate_matched { 4 } else { 1 }) * (128 << 20);
    let decode = (44u64 << 20)
        .saturating_add(20u64.saturating_mul(width.saturating_mul(height)))
        .saturating_add(970 * width.min(1024) * height.min(1024))
        .saturating_add(retained);
    // Upstream's target-search model allowance is calibrated for BOP. Pad HOP's
    // three extra model sets from 28 to 64 MiB each.
    let encode = encode.saturating_add(if rate_matched { 3 * (36 << 20) } else { 0 });
    encode.max(decode)
}

fn validate(rgba: &[u8], width: usize, height: usize, limit: usize) -> Result<(), &'static str> {
    let pixels = width.checked_mul(height).ok_or("dimension overflow")?;
    if pixels == 0 || pixels > limit || pixels.checked_mul(4) != Some(rgba.len()) {
        return Err("invalid image dimensions or pixel limit exceeded");
    }
    Ok(())
}

#[derive(Debug, Serialize)]
pub struct Metrics {
    pub mse: f64,
    pub psnr: Option<f64>,
    pub ssim: f64,
}

/// RGB MSE/PSNR; SSIM on Rec.709 luma, non-overlapping 8x8 windows (including edge windows).
/// Population variance, K1=.01, K2=.03, L=255. Not MS-SSIM or a perceptual guarantee.
pub fn measure(
    original: &[u8],
    decoded: &[u8],
    width: usize,
    height: usize,
) -> Result<Metrics, &'static str> {
    validate(original, width, height, MAX_PIXELS)?;
    validate(decoded, width, height, MAX_PIXELS)?;
    let mut squared = 0.0;
    for (a, b) in original.chunks_exact(4).zip(decoded.chunks_exact(4)) {
        for c in 0..3 {
            squared += (f64::from(a[c]) - f64::from(b[c])).powi(2);
        }
    }
    let mse = squared / (width * height * 3) as f64;
    let luma =
        |p: &[u8]| 0.2126 * f64::from(p[0]) + 0.7152 * f64::from(p[1]) + 0.0722 * f64::from(p[2]);
    let mut total = 0.0;
    let mut windows = 0;
    for y in (0..height).step_by(8) {
        for x in (0..width).step_by(8) {
            let (mut sa, mut sb, mut saa, mut sbb, mut sab, mut n) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            for yy in y..(y + 8).min(height) {
                for xx in x..(x + 8).min(width) {
                    let i = (yy * width + xx) * 4;
                    let (a, b) = (luma(&original[i..i + 4]), luma(&decoded[i..i + 4]));
                    sa += a;
                    sb += b;
                    saa += a * a;
                    sbb += b * b;
                    sab += a * b;
                    n += 1.0;
                }
            }
            let (ma, mb) = (sa / n, sb / n);
            let (va, vb, cov) = (
                (saa / n - ma * ma).max(0.0),
                (sbb / n - mb * mb).max(0.0),
                sab / n - ma * mb,
            );
            let (c1, c2) = (6.5025, 58.5225);
            total += ((2.0 * ma * mb + c1) * (2.0 * cov + c2))
                / ((ma * ma + mb * mb + c1) * (va + vb + c2));
            windows += 1;
        }
    }
    Ok(Metrics {
        mse,
        psnr: (mse > 0.0).then(|| 10.0 * (65025.0 / mse).log10()),
        ssim: (total / f64::from(windows)).clamp(-1.0, 1.0),
    })
}

#[wasm_bindgen]
pub fn metrics(
    original: &[u8],
    decoded: &[u8],
    width: u32,
    height: u32,
) -> Result<JsValue, JsError> {
    let measured = measure(original, decoded, width as usize, height as usize).map_err(error)?;
    measured
        .serialize(&serde_wasm_bindgen::Serializer::new().serialize_missing_as_null(true))
        .map_err(error)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn identical() {
        let p = vec![128; 9 * 11 * 4];
        let m = measure(&p, &p, 9, 11).unwrap();
        assert_eq!(m.mse, 0.0);
        assert_eq!(m.psnr, None);
        assert!((m.ssim - 1.0).abs() < 1e-10);
    }
    #[test]
    fn full_resolution_metrics_are_not_limited_to_four_megapixels() {
        let pixels = vec![127; 2200 * 2000 * 4];
        let measured = measure(&pixels, &pixels, 2200, 2000).unwrap();
        assert_eq!(measured.mse, 0.0);
        assert!((measured.ssim - 1.0).abs() < 1e-9);
    }
    #[test]
    fn two_k_encode_fits_the_browser_budget() {
        for rate_matched in [false, true] {
            let estimate = zenjpegai::encoder::estimate_encode_memory(
                2048,
                2048,
                OperatingPoint::Bop,
                rate_matched,
            );
            assert!(estimate.live_bytes <= MEMORY_LIMIT);
        }
    }
    #[test]
    fn larger_encodes_fit_the_doubled_budget_but_need_confirmation() {
        let estimate =
            zenjpegai::encoder::estimate_encode_memory(3600, 2400, OperatingPoint::Bop, false);
        assert!(estimate.live_bytes > 1536 << 20);
        assert!(estimate.live_bytes <= MEMORY_LIMIT);
        assert_eq!(MEMORY_LIMIT, 3072 << 20);
    }
    #[test]
    fn operating_points_and_hop_memory_are_explicit() {
        assert_eq!(parse_operating_point("bop"), Ok(OperatingPoint::Bop));
        assert_eq!(parse_operating_point("hop"), Ok(OperatingPoint::Hop));
        assert_eq!(parse_operating_point("sop"), Ok(OperatingPoint::Sop));
        assert!(parse_operating_point("invalid").is_err());
        let bop = estimate_job_memory(1024, 1024, OperatingPoint::Bop, false);
        let hop = estimate_job_memory(1024, 1024, OperatingPoint::Hop, false);
        let analysis =
            zenjpegai::encoder::estimate_encode_memory(1024, 1024, OperatingPoint::Hop, false);
        assert!(hop > bop && hop > analysis.live_bytes);
        assert!(estimate_job_memory(2048, 2048, OperatingPoint::Hop, true) > 1536 << 20);
        assert!(estimate_job_memory(2048, 2048, OperatingPoint::Hop, true) <= MEMORY_LIMIT);
        assert!(estimate_job_memory(6000, 4000, OperatingPoint::Hop, false) > MEMORY_LIMIT);
    }
    #[test]
    fn known_error() {
        let a = vec![0; 8 * 8 * 4];
        let b = vec![10; 8 * 8 * 4];
        let m = measure(&a, &b, 8, 8).unwrap();
        assert!((m.mse - 100.0).abs() < 1e-10);
        assert!((m.psnr.unwrap() - 28.1308).abs() < 0.001);
        assert!(m.ssim < 0.1);
    }
    #[test]
    fn malformed() {
        assert!(measure(&[0; 4], &[0; 4], 0, 1).is_err());
        assert!(measure(&[0; 4], &[0; 3], 1, 1).is_err());
        assert!(validate(&[], usize::MAX, 2, MAX_PIXELS).is_err());
    }
    #[test]
    fn alpha_excluded() {
        let m = measure(&[12, 34, 56, 0], &[12, 34, 56, 255], 1, 1).unwrap();
        assert_eq!(m.mse, 0.0);
        assert_eq!(m.ssim, 1.0);
    }
}
