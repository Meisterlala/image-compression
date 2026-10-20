use coolchic::reference::architecture::{Architecture, Interpolation, Layer};
use serde_json::Value;

pub fn interpolation(value: &Value) -> Interpolation {
    match value.as_str().unwrap() {
        "nearest" => Interpolation::Nearest,
        "bilinear" => Interpolation::Bilinear,
        "bicubic" => Interpolation::Bicubic,
        _ => panic!("Invalid fixture interpolation"),
    }
}
pub fn architecture(value: &Value) -> Architecture {
    let number = |name: &str| value[name].as_u64().unwrap() as usize;
    let flag = |name: &str| value[name].as_bool().unwrap();
    let range = |name: &str| {
        value[name].as_array().map(|r| {
            (
                r[0].as_u64().unwrap() as usize,
                r[1].as_u64().unwrap() as usize,
            )
        })
    };
    Architecture {
        height: number("height"),
        width: number("width"),
        latent_min: number("latent_min"),
        latent_max: number("latent_max"),
        hyperlatent: range("hyperlatent"),
        ifce_range: range("ifce_range"),
        ifce_outputs: number("ifce_outputs"),
        spatial_context: number("spatial_context"),
        arm_hidden: number("arm_hidden"),
        arm_stabiliser: flag("arm_stabiliser"),
        synthesis_stabiliser: flag("synthesis_stabiliser"),
        up_kernel: number("up_kernel"),
        pre_kernel: number("pre_kernel"),
        common_randomness: flag("common_randomness"),
        interpolation: interpolation(&value["interpolation"]),
        synthesis: value["synthesis"]
            .as_array()
            .unwrap()
            .iter()
            .map(|layer| Layer {
                outputs: layer["outputs"].as_u64().unwrap() as usize,
                kernel: layer["kernel"].as_u64().unwrap() as usize,
                residual: layer["residual"].as_bool().unwrap(),
                relu: layer["relu"].as_bool().unwrap(),
            })
            .collect(),
    }
}
