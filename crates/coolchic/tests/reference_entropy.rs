//! Trainable entropy-model parity; the existing integer bitstream remains separate.
use coolchic::reference::{
    arm::{Arm, spatial_context, spatial_context_backward},
    inter_feature::InterFeature,
    pyramid::Pyramid,
    rate::laplace_rate,
};

// Keep fixture readers shared without pulling any serialization dependency into WASM.
#[path = "support/operators.rs"]
mod support;
use support::*;

#[test]
fn residual_arm_and_ifce_linear_networks_forward_and_backward() {
    for case in fixtures()["arm"].as_array().unwrap() {
        let model = Arm {
            layers: case["layers"]
                .as_array()
                .unwrap()
                .iter()
                .map(conv)
                .collect(),
            stabiliser: (!case["stabiliser"].is_null()).then(|| conv(&case["stabiliser"])),
        };
        let input = tensor(&case["input"]);
        tensor_close(
            &model.forward(&input).unwrap(),
            &case["output"],
            "ARM output",
        );
        let gradient = model.backward(&input, &tensor(&case["upstream"])).unwrap();
        tensor_close(&gradient.input, &case["input_grad"], "ARM input gradient");
        for (g, expected) in gradient
            .layers
            .iter()
            .zip(case["layer_grad"].as_array().unwrap())
        {
            conv_gradient(g, expected);
        }
        if let Some(g) = &gradient.stabiliser {
            conv_gradient(g, &case["stabiliser_grad"]);
        }
    }
}

#[test]
fn causal_context_priority_and_zero_padded_gradient_scatter() {
    for case in fixtures()["context"].as_array().unwrap() {
        let output = spatial_context(
            &tensor(&case["input"]),
            case["count"].as_u64().unwrap() as usize,
        )
        .unwrap();
        tensor_close(&output, &case["output"], "spatial contexts");
        tensor_close(
            &spatial_context_backward(&tensor(&case["upstream"])).unwrap(),
            &case["input_grad"],
            "causal context input gradient",
        );
    }
}

#[test]
fn laplace_rate_probability_floor_scale_clamp_and_gradients() {
    for case in fixtures()["rate"].as_array().unwrap() {
        let result = laplace_rate(&tensor(&case["latent"]), &tensor(&case["raw_arm"])).unwrap();
        tensor_close(&result.bits, &case["bits"], "Laplace rate bits");
        tensor_close(
            &result.latent_derivative,
            &case["latent_grad"],
            "Laplace latent derivative",
        );
        tensor_close(
            &result.arm_derivative,
            &case["arm_grad"],
            "Laplace ARM derivative",
        );
    }
}

#[test]
fn cross_resolution_ifce_forward_backward_and_disabled_levels() {
    for case in fixtures()["ifce"].as_array().unwrap() {
        let model = InterFeature {
            output_channels: case["outputs"].as_u64().unwrap() as usize,
            networks: case["networks"]
                .as_array()
                .unwrap()
                .iter()
                .map(|network| {
                    if network.is_null() {
                        None
                    } else {
                        Some(Arm {
                            layers: network["layers"]
                                .as_array()
                                .unwrap()
                                .iter()
                                .map(conv)
                                .collect(),
                            stabiliser: None,
                        })
                    }
                })
                .collect(),
        };
        let latents: Vec<_> = case["latents"]
            .as_array()
            .unwrap()
            .iter()
            .map(tensor)
            .collect();
        let outputs = model.forward(&latents).unwrap();
        for (actual, expected) in outputs.iter().zip(case["output"].as_array().unwrap()) {
            tensor_close(actual, expected, "IFCE output");
        }
        let upstreams: Vec<_> = case["upstream"]
            .as_array()
            .unwrap()
            .iter()
            .map(tensor)
            .collect();
        let gradient = model.backward(&latents, &upstreams).unwrap();
        for (actual, expected) in gradient
            .latents
            .iter()
            .zip(case["latent_grad"].as_array().unwrap())
        {
            tensor_close(actual, expected, "IFCE latent gradient");
        }
        for (actual, expected) in gradient
            .networks
            .iter()
            .zip(case["parameter_grad"].as_array().unwrap())
        {
            if let Some(actual) = actual {
                for (layer, expected) in actual.layers.iter().zip(expected.as_array().unwrap()) {
                    conv_gradient(layer, expected);
                }
            } else {
                assert!(expected.is_null());
            }
        }
    }
}

#[test]
fn full_learned_pyramid_odd_sizes_and_shared_kernel_gradients() {
    for case in fixtures()["pyramid"].as_array().unwrap() {
        let model = Pyramid {
            upsampling: case["upsampling"]
                .as_array()
                .unwrap()
                .iter()
                .map(filter)
                .collect(),
            preconcat: case["preconcat"]
                .as_array()
                .unwrap()
                .iter()
                .map(filter)
                .collect(),
        };
        let latents: Vec<_> = case["latents"]
            .as_array()
            .unwrap()
            .iter()
            .map(tensor)
            .collect();
        tensor_close(
            &model.forward(&latents).unwrap(),
            &case["output"],
            "pyramid output",
        );
        let gradient = model
            .backward(&latents, &tensor(&case["upstream"]))
            .unwrap();
        for (actual, expected) in gradient
            .latents
            .iter()
            .zip(case["latent_grad"].as_array().unwrap())
        {
            tensor_close(actual, expected, "pyramid latent gradient");
        }
        for (actual, expected) in gradient
            .upsampling
            .iter()
            .zip(case["upsampling"].as_array().unwrap())
        {
            close(
                actual,
                &floats(&expected["half_grad"]),
                "pyramid upsampling gradient",
            );
        }
        for (actual, expected) in gradient
            .preconcat
            .iter()
            .zip(case["preconcat"].as_array().unwrap())
        {
            close(
                actual,
                &floats(&expected["half_grad"]),
                "pyramid preconcat gradient",
            );
        }
    }
}
