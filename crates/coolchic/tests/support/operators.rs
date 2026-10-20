use coolchic::reference::{Conv2d, ConvGradient, Filter, FilterMode, Tensor};
use serde_json::Value;

pub fn fixtures() -> Value {
    serde_json::from_str(include_str!("../fixtures/reference-operators.json")).unwrap()
}
pub fn floats(value: &Value) -> Vec<f32> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect()
}
pub fn tensor(value: &Value) -> Tensor {
    let shape: Vec<usize> = value["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as usize)
        .collect();
    Tensor::new(shape[0], shape[1], shape[2], floats(&value["data"])).unwrap()
}
pub fn conv(value: &Value) -> Conv2d {
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
pub fn filter(value: &Value) -> Filter {
    Filter {
        kernel: value["kernel"].as_u64().unwrap() as usize,
        half: floats(&value["half"]),
        mode: if value["mode"] == "upsample" {
            FilterMode::Upsample
        } else {
            FilterMode::Residual
        },
    }
}
pub fn close(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label} length");
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        let tolerance = 3e-6 + e.abs() * 2e-5;
        assert!(
            a.is_finite() && (a - e).abs() <= tolerance,
            "{label}[{i}]: Rust={a}, PyTorch={e}, tolerance={tolerance}"
        );
    }
}
pub fn tensor_close(actual: &Tensor, expected: &Value, label: &str) {
    let expected = tensor(expected);
    assert!(actual.same_shape(&expected), "{label} shape");
    close(&actual.data, &expected.data, label);
}
pub fn conv_gradient(actual: &ConvGradient, expected: &Value) {
    close(
        &actual.weight,
        &floats(&expected["weight"]),
        "weight gradient",
    );
    close(&actual.bias, &floats(&expected["bias"]), "bias gradient");
}
