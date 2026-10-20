//! Optional browser compute bridge. Only floating-point training convolutions
//! are offloaded; integer entropy coding and portable stream decoding stay CPU.
use super::{Conv2d, Tensor};

#[cfg(all(feature = "wasm", target_arch = "wasm32"))]
mod browser {
    use super::*;
    use js_sys::{Float32Array, Function, Uint32Array};
    use std::{
        cell::{Cell, RefCell},
        collections::HashSet,
    };
    use wasm_bindgen::prelude::*;

    thread_local! {
        static BACKEND: RefCell<Option<(Function, u32)>> = const { RefCell::new(None) };
        static CPU_ONLY: Cell<bool> = const { Cell::new(false) };
        static REJECTED: RefCell<HashSet<[u32; 8]>> = RefCell::new(HashSet::new());
    }

    /// Install a synchronous worker bridge, not a Promise-returning callback.
    /// Clearing it guarantees CPU-only operation (also used before decoding).
    #[wasm_bindgen(js_name = setCoolChicAccelerator)]
    pub fn configure(callback: Option<Function>, minimum_operations: u32) {
        BACKEND.with_borrow_mut(|backend| *backend = callback.map(|f| (f, minimum_operations)));
        REJECTED.with_borrow_mut(HashSet::clear);
    }

    /// CPU reference/calibration entry point. The bridge uses this once per
    /// shape to measure the actual local WASM/GPU crossover, including copies.
    #[wasm_bindgen(js_name = coolChicCpuConvolution)]
    pub fn cpu_convolution(
        params: &[u32],
        input: &[f32],
        weight: &[f32],
        bias: &[f32],
        gradient: Option<Vec<f32>>,
    ) -> Result<Vec<f32>, JsError> {
        if params.len() != 8 {
            return Err(JsError::new("Invalid convolution parameters"));
        }
        let layer = Conv2d {
            input_channels: params[0] as usize,
            output_channels: params[1] as usize,
            kernel: params[4] as usize,
            residual: params[5] != 0,
            relu: params[6] != 0,
            weight: weight.to_vec(),
            bias: bias.to_vec(),
        };
        let input = Tensor::new(
            params[0] as usize,
            params[2] as usize,
            params[3] as usize,
            input.to_vec(),
        )
        .map_err(|e| JsError::new(&e.to_string()))?;
        let old = CPU_ONLY.replace(true);
        let result = (|| {
            if let Some(gradient) = gradient {
                let gradient =
                    Tensor::new(params[1] as usize, input.height, input.width, gradient)?;
                let g = layer.backward(&input, &gradient)?;
                Ok([g.input.data, g.weight, g.bias].concat())
            } else {
                Ok(layer.forward(&input)?.data)
            }
        })();
        CPU_ONLY.set(old);
        result.map_err(|e: crate::Error| JsError::new(&e.to_string()))
    }

    fn key(layer: &Conv2d, input: &Tensor, backward: bool) -> [u32; 8] {
        [
            input.channels as u32,
            layer.output_channels as u32,
            input.height as u32,
            input.width as u32,
            layer.kernel as u32,
            u32::from(layer.residual),
            u32::from(layer.relu),
            u32::from(backward),
        ]
    }

    pub fn eligible(layer: &Conv2d, input: &Tensor, backward: bool) -> bool {
        if CPU_ONLY.get() || REJECTED.with_borrow(|r| r.contains(&key(layer, input, backward))) {
            return false;
        }
        BACKEND.with_borrow(|backend| {
            backend.as_ref().is_some_and(|(_, minimum)| {
                input.data.len() as u64
                    * layer.output_channels as u64
                    * (layer.kernel * layer.kernel) as u64
                    >= u64::from(*minimum)
            })
        })
    }

    pub fn convolution(
        layer: &Conv2d,
        input: &Tensor,
        gradient: Option<&Tensor>,
        output: &mut [f32],
    ) -> bool {
        if !eligible(layer, input, gradient.is_some()) {
            return false;
        }
        let shape = key(layer, input, gradient.is_some());
        let accelerated = BACKEND.with_borrow(|backend| {
            let Some((callback, _)) = backend else {
                return false;
            };
            let params = Uint32Array::from(shape.as_slice());
            let Ok(value) = callback.call5(
                &JsValue::NULL,
                &params,
                &Float32Array::from(input.data.as_slice()),
                &Float32Array::from(layer.weight.as_slice()),
                &Float32Array::from(layer.bias.as_slice()),
                &gradient.map_or(JsValue::NULL, |g| {
                    Float32Array::from(g.data.as_slice()).into()
                }),
            ) else {
                return false;
            };
            let Ok(values) = value.dyn_into::<Float32Array>() else {
                return false;
            };
            if values.length() as usize != output.len() {
                return false;
            }
            values.copy_to(output);
            if output.iter().all(|v| v.is_finite()) {
                true
            } else {
                output.fill(0.0);
                false
            }
        });
        if !accelerated {
            REJECTED.with_borrow_mut(|r| {
                r.insert(shape);
            });
        }
        accelerated
    }
}

pub fn eligible(layer: &Conv2d, input: &Tensor, backward: bool) -> bool {
    #[cfg(all(feature = "wasm", target_arch = "wasm32"))]
    {
        browser::eligible(layer, input, backward)
    }
    #[cfg(not(all(feature = "wasm", target_arch = "wasm32")))]
    {
        let _ = (layer, input, backward);
        false
    }
}

pub fn convolution(
    layer: &Conv2d,
    input: &Tensor,
    gradient: Option<&Tensor>,
    output: &mut [f32],
) -> bool {
    #[cfg(all(feature = "wasm", target_arch = "wasm32"))]
    {
        browser::convolution(layer, input, gradient, output)
    }
    #[cfg(not(all(feature = "wasm", target_arch = "wasm32")))]
    {
        let _ = (layer, input, gradient, output);
        false
    }
}
