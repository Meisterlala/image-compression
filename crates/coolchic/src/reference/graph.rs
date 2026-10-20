//! One GPU forward/backward graph per iteration, not one host synchronization
//! per operator. The reference optimizer, schedule and bitstream remain Rust.
use super::{Tensor, network::Network, quantizer::Quantizer};

pub struct ParameterGradient {
    pub groups: [Vec<f32>; 8],
    pub latents: Vec<Tensor>,
}

#[cfg(feature = "wasm")]
pub(crate) fn descriptor(network: &Network) -> String {
    use serde_json::json;
    let a = &network.architecture;
    let interpolation = match a.interpolation {
        super::architecture::Interpolation::Nearest => "nearest",
        super::architecture::Interpolation::Bilinear => "bilinear",
        super::architecture::Interpolation::Bicubic => "bicubic",
    };
    json!({ "architecture": {
        "height": a.height, "width": a.width, "latent_min": a.latent_min, "latent_max": a.latent_max,
        "hyperlatent": a.hyperlatent, "ifce_outputs": a.ifce_outputs, "spatial_context": a.spatial_context,
        "arm_hidden": a.arm_hidden, "arm_stabiliser": a.arm_stabiliser, "synthesis_stabiliser": a.synthesis_stabiliser,
        "up_kernel": a.up_kernel, "pre_kernel": a.pre_kernel, "common_randomness": a.common_randomness,
        "interpolation": interpolation,
        "synthesis": a.synthesis.iter().map(|l| json!({"outputs":l.outputs,"kernel":l.kernel,"residual":l.residual,"relu":l.relu})).collect::<Vec<_>>()
    }, "latents": a.latent_specs().iter().map(|l| json!({"height":l.height,"width":l.width,"hyper":l.hyper})).collect::<Vec<_>>(),
    "ifce_inputs": a.ifce_inputs(), "neighbours": super::arm::neighbours(a.spatial_context).expect("validated architecture"),
    "group_lengths": network.parameters().0.iter().map(Vec::len).collect::<Vec<_>>() }).to_string()
}

#[cfg(feature = "wasm")]
pub(crate) fn pack_reference_gradient(
    network: &Network,
    gradient: &super::network::Gradient,
) -> Vec<f32> {
    let mut groups = gradient.parameter_groups(network);
    for chunk in network.parameters().1.iter().filter(|c| !c.trainable) {
        groups[chunk.group][chunk.offset..chunk.offset + chunk.len()].fill(0.0);
    }
    groups
        .into_iter()
        .flatten()
        .chain(gradient.latents.iter().flat_map(|l| l.data.iter().copied()))
        .collect()
}

#[cfg(all(feature = "wasm", target_arch = "wasm32"))]
mod browser {
    use super::*;
    use js_sys::{Float32Array, Function};
    use std::cell::RefCell;
    use wasm_bindgen::prelude::*;
    thread_local! { static BACKEND: RefCell<Option<Function>> = const { RefCell::new(None) }; }
    #[wasm_bindgen(js_name = setCoolChicGraphAccelerator)]
    pub fn configure(callback: Option<Function>) {
        BACKEND.with_borrow_mut(|backend| *backend = callback);
    }
    fn array(data: &[f32]) -> Float32Array {
        let memory = wasm_bindgen::memory().unchecked_into::<js_sys::WebAssembly::Memory>();
        let buffer = memory.buffer();
        if buffer.is_instance_of::<js_sys::SharedArrayBuffer>() {
            let begin = data.as_ptr() as u32 / 4;
            Float32Array::new(&buffer).subarray(begin, begin + data.len() as u32)
        } else {
            Float32Array::from(data)
        }
    }
    pub fn gradient(
        network: &Network,
        target: &Tensor,
        lambda: f32,
        mode: Quantizer,
        temperature: f32,
        noise: &[f32],
    ) -> Option<ParameterGradient> {
        let callback = BACKEND.with_borrow(Clone::clone)?;
        let gradient = (|| {
            let groups = network.parameters().0;
            let group_lengths: Vec<_> = groups.iter().map(Vec::len).collect();
            let parameters: Vec<_> = groups.into_iter().flatten().collect();
            let latents: Vec<_> = network
                .latents
                .iter()
                .flat_map(|l| l.data.iter().copied())
                .collect();
            let common: Vec<_> = network
                .common
                .iter()
                .flat_map(|l| l.data.iter().copied())
                .collect();
            let kind = match mode {
                Quantizer::None => 0.0,
                Quantizer::SoftAlone => 1.0,
                Quantizer::Soft => 2.0,
                Quantizer::StraightThrough => 3.0,
                Quantizer::Hard => 4.0,
            };
            let normalizer = if matches!(mode, Quantizer::Soft | Quantizer::SoftAlone) {
                (1.0 / (2.0 * temperature)).tanh()
            } else {
                1.0
            };
            let result = callback.call7(
                &JsValue::NULL,
                &JsValue::from_str(&descriptor(network)),
                &array(&parameters),
                &array(&latents),
                &array(&target.data),
                &array(noise),
                &array(&common),
                &Float32Array::from([kind, temperature, lambda, normalizer].as_slice()),
            );
            let values = result
                .ok()
                .and_then(|v| v.dyn_into::<Float32Array>().ok())?;
            if values.length() as usize != parameters.len() + latents.len() {
                return None;
            }
            let mut packed = vec![0.0; values.length() as usize];
            values.copy_to(&mut packed);
            if packed.iter().any(|v| !v.is_finite()) {
                return None;
            }
            let mut offset = 0;
            let groups = std::array::from_fn(|i| {
                let values = packed[offset..offset + group_lengths[i]].to_vec();
                offset += group_lengths[i];
                values
            });
            let mut gradients = Vec::with_capacity(network.latents.len());
            for latent in &network.latents {
                let size = latent.data.len();
                gradients.push(
                    Tensor::new(
                        latent.channels,
                        latent.height,
                        latent.width,
                        packed[offset..offset + size].to_vec(),
                    )
                    .ok()?,
                );
                offset += size;
            }
            Some(ParameterGradient {
                groups,
                latents: gradients,
            })
        })();
        if gradient.is_none() {
            BACKEND.with_borrow_mut(|backend| *backend = None);
        }
        gradient
    }
}

pub fn gradient(
    network: &Network,
    target: &Tensor,
    lambda: f32,
    mode: Quantizer,
    temperature: f32,
    noise: &[f32],
) -> Option<ParameterGradient> {
    #[cfg(all(feature = "wasm", target_arch = "wasm32"))]
    {
        browser::gradient(network, target, lambda, mode, temperature, noise)
    }
    #[cfg(not(all(feature = "wasm", target_arch = "wasm32")))]
    {
        let _ = (network, target, lambda, mode, temperature, noise);
        None
    }
}
