use coolchic::reference::{
    architecture::{Architecture, OperatingPoint},
    network::Network,
    quantizer::Quantizer,
    random::{Random, common_noise},
    resize::{resize, resize_backward},
};
use serde_json::Value;

#[path = "support/operators.rs"]
#[allow(dead_code)]
mod operators;
#[path = "support/network.rs"]
mod support;
use operators::{close, floats, tensor, tensor_close};

fn fixtures() -> Value {
    serde_json::from_str(include_str!("fixtures/reference-graph.json")).unwrap()
}

// Entropy gradients divide by CDF differences as small as 2^-16. Small
// float32 BLAS/libm differences are amplified through the whole graph; retain
// tighter per-operator tests while bounding full-chain absolute/relative error.
fn gradient_close(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label} length");
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        let tolerance = 3e-5 + e.abs() * 1e-3;
        assert!(
            a.is_finite() && (a - e).abs() <= tolerance,
            "{label}[{i}]: Rust={a}, PyTorch={e}, tolerance={tolerance}"
        );
    }
}

#[test]
fn complete_reference_graph_forward_loss_and_all_parameter_gradients() {
    for case in fixtures()["graphs"].as_array().unwrap() {
        let architecture = support::architecture(&case["architecture"]);
        let expected_inputs: Vec<usize> = case["ifce_inputs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        assert_eq!(
            architecture.ifce_inputs(),
            expected_inputs,
            "{} IFCE shape rules",
            case["name"]
        );
        let mut network = Network::new(architecture, &mut Random::new(42)).unwrap();
        let groups = std::array::from_fn(|i| floats(&case["groups"][i]));
        network.set_parameters(&groups).unwrap();
        let (roundtrip, chunks) = network.parameters();
        assert_eq!(
            roundtrip, groups,
            "{} neural parameter ordering",
            case["name"]
        );
        assert!(chunks.iter().any(|c| !c.trainable));
        network.latents = case["latents"]
            .as_array()
            .unwrap()
            .iter()
            .map(tensor)
            .collect();
        let target = tensor(&case["target"]);
        let (evaluation, gradient) = network
            .loss_gradient(
                &target,
                0.001,
                Quantizer::Soft,
                0.35,
                &floats(&case["noise"]),
            )
            .unwrap();
        close(
            &[evaluation.loss],
            &[case["loss"].as_f64().unwrap() as f32],
            &format!("{} loss", case["name"]),
        );
        tensor_close(
            &evaluation.image,
            &case["decoded"],
            &format!("{} decoded image", case["name"]),
        );
        let rates: Vec<_> = evaluation
            .bits
            .iter()
            .flat_map(|t| t.data.iter().copied())
            .collect();
        close(
            &rates,
            &floats(&case["rate"]),
            &format!("{} latent rates", case["name"]),
        );
        for (index, (actual, expected)) in gradient
            .latents
            .iter()
            .zip(case["latent_grad"].as_array().unwrap())
            .enumerate()
        {
            gradient_close(
                &actual.data,
                &tensor(expected).data,
                &format!("{} latent {index} gradient", case["name"]),
            );
        }
        for (index, group) in gradient.parameter_groups(&network).iter().enumerate() {
            gradient_close(
                group,
                &floats(&case["parameter_grad"][index]),
                &format!("{} parameter group {index} gradient", case["name"]),
            );
        }
        let evaluation = network.evaluate(&target, 0.001).unwrap();
        tensor_close(
            &evaluation.image,
            &case["evaluation"]["image"],
            &format!("{} eval reconstruction", case["name"]),
        );
        close(
            &[evaluation.loss],
            &[case["evaluation"]["loss"].as_f64().unwrap() as f32],
            &format!("{} eval loss", case["name"]),
        );
    }
}

#[test]
fn final_image_interpolation_and_gradients_match_reference() {
    for case in fixtures()["resize"].as_array().unwrap() {
        let mode = support::interpolation(&case["mode"]);
        let input = tensor(&case["input"]);
        let expected = tensor(&case["output"]);
        tensor_close(
            &resize(&input, expected.height, expected.width, mode).unwrap(),
            &case["output"],
            "resize output",
        );
        tensor_close(
            &resize_backward(&input, &tensor(&case["upstream"]), mode).unwrap(),
            &case["input_grad"],
            "resize input gradient",
        );
    }
}

#[test]
fn bitstream_defined_common_noise_matches_reference() {
    let result = common_noise(&[(3, 5), (2, 3), (1, 2)]).unwrap();
    for (actual, expected) in result
        .iter()
        .zip(fixtures()["common_noise"].as_array().unwrap())
    {
        assert_eq!(
            actual.data,
            tensor(expected).data,
            "Common noise must be deterministic across runtimes"
        );
    }
}

#[test]
fn standard_intra_presets_and_adaptive_resolutions() {
    for (point, spatial, outputs, hidden) in [
        (OperatingPoint::Lop, 6, 2, 3),
        (OperatingPoint::Mop, 10, 4, 16),
        (OperatingPoint::Hop, 14, 6, 48),
        (OperatingPoint::Vhop, 20, 6, 64),
    ] {
        let config = Architecture::preset(96, 64, point).unwrap();
        config.validate().unwrap();
        assert_eq!(
            (
                config.spatial_context,
                config.ifce_outputs,
                config.synthesis[0].outputs
            ),
            (spatial, outputs, hidden)
        );
        assert_eq!(config.latent_specs().len(), 10);
        assert_eq!(config.latent_max, 6);
    }
    assert_eq!(
        Architecture::preset(1000, 1000, OperatingPoint::Lop)
            .unwrap()
            .latent_max,
        7
    );
    assert_eq!(
        Architecture::preset(2000, 1500, OperatingPoint::Lop)
            .unwrap()
            .latent_max,
        8
    );
}
