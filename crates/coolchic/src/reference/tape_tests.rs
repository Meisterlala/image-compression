use super::{
    Conv2d, Filter, FilterMode, Tensor, arm::Arm, convolution::Synthesis, pyramid::Pyramid,
};

fn tensor(c: usize, h: usize, w: usize) -> Tensor {
    Tensor::new(c, h, w, vec![0.2; c * h * w]).unwrap()
}
fn conv(ci: usize, co: usize, relu: bool) -> Conv2d {
    Conv2d {
        input_channels: ci,
        output_channels: co,
        kernel: 1,
        weight: vec![0.1; ci * co],
        bias: vec![0.0; co],
        residual: false,
        relu,
    }
}

#[test]
fn convolution_rejects_invalid_activation_layouts_before_reading_them() {
    let input = tensor(2, 2, 3);
    let gradient = tensor(3, 2, 3);
    for relu in [false, true] {
        let layer = conv(2, 3, relu);
        assert!(
            layer
                .backward_cached(&input, &gradient, Some(&layer.forward(&input).unwrap()))
                .is_ok()
        );
        for bad in [
            tensor(2, 2, 3),
            tensor(3, 1, 6),
            Tensor {
                channels: 3,
                height: 2,
                width: 3,
                data: vec![],
            },
        ] {
            assert!(
                layer
                    .backward_cached(&input, &gradient, Some(&bad))
                    .is_err()
            );
        }
    }
}

#[test]
fn arm_rejects_mismatched_intermediate_activations() {
    let arm = Arm {
        layers: vec![conv(2, 2, true), conv(2, 1, false)],
        stabiliser: None,
    };
    let input = tensor(2, 2, 3);
    let (output, mut states) = arm.forward_cached(&input).unwrap();
    assert!(arm.backward_cached(&input, &output, &states).is_ok());
    states[1] = tensor(1, 2, 3);
    assert!(arm.backward_cached(&input, &output, &states).is_err());
}

#[test]
fn synthesis_rejects_invalid_trunk_and_merge_tapes() {
    let synthesis = Synthesis {
        trunk: vec![conv(2, 3, true)],
        stabiliser: Some(conv(2, 3, false)),
        output_transform: conv(3, 3, false),
    };
    let input = tensor(2, 2, 3);
    let (output, states) = synthesis.forward_cached(&input).unwrap();
    assert!(synthesis.backward_cached(&input, &output, &states).is_ok());
    for index in [1, 2] {
        let mut bad = states.clone();
        bad[index] = tensor(3, 1, 6);
        assert!(synthesis.backward_cached(&input, &output, &bad).is_err());
    }
}

#[test]
fn pyramid_checks_all_stage_shapes_and_exact_active_length() {
    let pyramid = Pyramid {
        upsampling: vec![Filter {
            kernel: 4,
            half: vec![0.0, 1.0],
            mode: FilterMode::Upsample,
        }],
        preconcat: vec![Filter {
            kernel: 1,
            half: vec![1.0],
            mode: FilterMode::Residual,
        }],
    };
    let latents = vec![tensor(1, 5, 9), tensor(1, 3, 5), tensor(1, 2, 3)];
    let (output, states) = pyramid.forward_cached(&latents).unwrap();
    assert!(pyramid.backward_cached(&latents, &output, &states).is_ok());
    let mut extra = states.clone();
    extra.push(states.last().unwrap().clone());
    assert!(pyramid.backward_cached(&latents, &output, &extra).is_err());
    let mut intermediate = states;
    intermediate[1] = tensor(2, 2, 8);
    assert!(
        pyramid
            .backward_cached(&latents, &output, &intermediate)
            .is_err()
    );
    let stopped = vec![tensor(0, 5, 9), tensor(1, 3, 5), tensor(1, 2, 3)];
    let (output, states) = pyramid.forward_cached(&stopped).unwrap();
    assert!(pyramid.backward_cached(&stopped, &output, &states).is_ok());
}
