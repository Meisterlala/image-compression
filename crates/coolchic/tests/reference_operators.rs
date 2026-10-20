//! Forward and reverse-mode parity against the actual pinned Python modules.
//! Fixtures are generated independently by scripts/verify-coolchic-operators.py.
use coolchic::reference::{
    Conv2d, ConvGradient, Filter, FilterMode, Synthesis, Tensor,
    quantizer::{self, Quantizer},
};
use serde_json::Value;

fn fixtures() -> Value {
    serde_json::from_str(include_str!("fixtures/reference-operators.json")).unwrap()
}
fn floats(value: &Value) -> Vec<f32> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect()
}
fn tensor(value: &Value) -> Tensor {
    let shape: Vec<usize> = value["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as usize)
        .collect();
    Tensor::new(shape[0], shape[1], shape[2], floats(&value["data"])).unwrap()
}
fn conv(value: &Value) -> Conv2d {
    Conv2d {
        input_channels: value["input_channels"].as_u64().unwrap() as usize,
        output_channels: value["output_channels"].as_u64().unwrap() as usize,
        kernel: value["kernel"].as_u64().unwrap() as usize,
        residual: value["residual"].as_bool().unwrap(),
        relu: value["relu"].as_bool().unwrap(),
        weight: floats(&value["weight"]),
        bias: floats(&value["bias"]),
    }
}
fn close(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label} length");
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        let tolerance = 3e-6 + e.abs() * 2e-5;
        assert!(
            a.is_finite() && (a - e).abs() <= tolerance,
            "{label}[{i}]: Rust={a}, PyTorch={e}, tolerance={tolerance}"
        );
    }
}
fn tensor_close(actual: &Tensor, expected: &Value, label: &str) {
    let expected = tensor(expected);
    assert!(actual.same_shape(&expected), "{label} shape");
    close(&actual.data, &expected.data, label);
}
fn conv_gradient(actual: &ConvGradient, expected: &Value) {
    close(
        &actual.weight,
        &floats(&expected["weight"]),
        "weight gradient",
    );
    close(&actual.bias, &floats(&expected["bias"]), "bias gradient");
}

#[test]
fn replicate_convolution_residual_relu_forward_and_backward() {
    for case in fixtures()["convolution"].as_array().unwrap() {
        let layer = conv(&case["layer"]);
        let input = tensor(&case["input"]);
        tensor_close(
            &layer.forward(&input).unwrap(),
            &case["output"],
            "convolution output",
        );
        let g = layer.backward(&input, &tensor(&case["upstream"])).unwrap();
        tensor_close(&g.input, &case["input_grad"], "convolution input gradient");
        conv_gradient(&g, &case["parameter_grad"]);
    }
}

#[test]
fn nonlinear_synthesis_stabiliser_and_output_transform_forward_and_backward() {
    for case in fixtures()["synthesis"].as_array().unwrap() {
        let model = Synthesis {
            trunk: case["trunk"].as_array().unwrap().iter().map(conv).collect(),
            stabiliser: (!case["stabiliser"].is_null()).then(|| conv(&case["stabiliser"])),
            output_transform: conv(&case["output_transform"]),
        };
        let input = tensor(&case["input"]);
        tensor_close(
            &model.forward(&input).unwrap(),
            &case["output"],
            "synthesis output",
        );
        let gradient = model.backward(&input, &tensor(&case["upstream"])).unwrap();
        tensor_close(
            &gradient.input,
            &case["input_grad"],
            "synthesis input gradient",
        );
        for (g, expected) in gradient
            .trunk
            .iter()
            .zip(case["trunk_grad"].as_array().unwrap())
        {
            conv_gradient(g, expected);
        }
        if let Some(g) = &gradient.stabiliser {
            conv_gradient(g, &case["stabiliser_grad"]);
        }
        conv_gradient(&gradient.output_transform, &case["output_transform_grad"]);
    }
}

#[test]
fn learned_upsampling_and_preconcat_forward_and_backward() {
    for case in fixtures()["filter"].as_array().unwrap() {
        let filter = Filter {
            kernel: case["kernel"].as_u64().unwrap() as usize,
            half: floats(&case["half"]),
            mode: if case["mode"] == "upsample" {
                FilterMode::Upsample
            } else {
                FilterMode::Residual
            },
        };
        let input = tensor(&case["input"]);
        let output = filter.forward(&input).unwrap();
        tensor_close(&output, &case["output"], "filter evaluation output");
        tensor_close(&output, &case["training_output"], "filter training output");
        let gradient = filter.backward(&input, &tensor(&case["upstream"])).unwrap();
        tensor_close(
            &gradient.input,
            &case["input_grad"],
            "filter input gradient",
        );
        close(
            &gradient.half,
            &floats(&case["half_grad"]),
            "symmetric filter parameter gradient",
        );
    }
}

#[test]
fn reference_quantization_and_noise_forward_and_backward() {
    let fixture = fixtures();
    for case in fixture["quantizer"].as_array().unwrap() {
        let mode = match case["mode"].as_str().unwrap() {
            "none" => Quantizer::None,
            "softround_alone" => Quantizer::SoftAlone,
            "softround" => Quantizer::Soft,
            "hardround" => Quantizer::Hard,
            "ste" => Quantizer::StraightThrough,
            _ => unreachable!(),
        };
        let (output, derivative) = quantizer::quantize(
            &floats(&case["input"]),
            &floats(&case["noise"]),
            mode,
            case["temperature"].as_f64().unwrap() as f32,
        )
        .unwrap();
        close(&output, &floats(&case["output"]), "quantizer output");
        close(
            &derivative,
            &floats(&case["derivative"]),
            "quantizer derivative",
        );
    }
    for case in fixture["noise"].as_array().unwrap() {
        let a = case["a"].as_f64().unwrap() as f32;
        let result: Vec<f32> = floats(&case["uniform"])
            .into_iter()
            .map(|u| quantizer::kumaraswamy(u, a).unwrap())
            .collect();
        close(&result, &floats(&case["output"]), "Kumaraswamy noise");
    }
}

#[test]
fn rejects_invalid_shapes_without_executing_kernels() {
    assert!(Tensor::new(1, 2, 2, vec![0.0]).is_err());
    assert!(Tensor::zeros(usize::MAX, 2, 2).is_err());
    let x = Tensor::zeros(1, 1, 1).unwrap();
    let filter = Filter {
        kernel: 3,
        half: vec![0.0, 1.0],
        mode: FilterMode::Upsample,
    };
    assert!(filter.forward(&x).is_err());
    assert!(quantizer::quantize(&[0.0], &[], Quantizer::None, 0.0).is_err());
    assert!(quantizer::quantize(&[0.0], &[0.0], Quantizer::Soft, 0.0).is_err());
}
