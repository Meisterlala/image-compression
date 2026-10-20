#![cfg(feature = "parallel")]
use coolchic::reference::{Conv2d, Tensor};

fn close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && (a - e).abs() <= 3e-5 + e.abs() * 3e-4,
            "derivative {i}: parallel={a}, serial={e}"
        );
    }
}

#[test]
fn real_parallel_convolutions_match_single_thread_forward_and_derivatives() {
    let serial = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let parallel = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    for (ci, co, h, w, k, residual, relu) in [
        (8, 8, 129, 131, 1, true, true),
        (3, 3, 129, 131, 3, true, true),
        // Oversized kernels, degenerate axes and repeated replicate samples.
        (2, 2, 1, 157, 41, true, false),
        (2, 3, 157, 1, 41, false, true),
    ] {
        let input = Tensor::new(
            ci,
            h,
            w,
            (0..ci * h * w)
                .map(|i| ((i * 7 % 23) as f32 - 11.0) / 31.0)
                .collect(),
        )
        .unwrap();
        let gradient = Tensor::new(
            co,
            h,
            w,
            (0..co * h * w)
                .map(|i| ((i * 5 % 17) as f32 - 8.0) / 29.0)
                .collect(),
        )
        .unwrap();
        let layer = Conv2d {
            input_channels: ci,
            output_channels: co,
            kernel: k,
            residual,
            relu,
            weight: (0..ci * co * k * k)
                .map(|i| ((i * 3 % 19) as f32 - 9.0) / 47.0)
                .collect(),
            bias: (0..co).map(|i| (i as f32 - 1.0) / 13.0).collect(),
        };
        let expected = serial.install(|| layer.forward(&input)).unwrap();
        let actual = parallel.install(|| layer.forward(&input)).unwrap();
        close(&actual.data, &expected.data);
        let expected = serial
            .install(|| layer.backward(&input, &gradient))
            .unwrap();
        let actual = parallel
            .install(|| layer.backward(&input, &gradient))
            .unwrap();
        close(&actual.input.data, &expected.input.data);
        close(&actual.weight, &expected.weight);
        close(&actual.bias, &expected.bias);
    }
}

#[test]
fn parallel_learned_filters_match_serial_gradients() {
    use coolchic::reference::{Filter, FilterMode};
    let serial = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let parallel = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let input = Tensor::new(
        2,
        257,
        263,
        (0..2 * 257 * 263)
            .map(|i| ((i * 7 % 23) as f32 - 11.0) / 31.0)
            .collect(),
    )
    .unwrap();
    for (kernel, mode) in [(7, FilterMode::Residual), (8, FilterMode::Upsample)] {
        let filter = Filter {
            kernel,
            mode,
            half: vec![0.1; kernel.div_ceil(2)],
        };
        let expected = serial.install(|| filter.forward(&input)).unwrap();
        let actual = parallel.install(|| filter.forward(&input)).unwrap();
        close(&actual.data, &expected.data);
        let gradient = Tensor::new(
            actual.channels,
            actual.height,
            actual.width,
            (0..actual.data.len())
                .map(|i| ((i * 5 % 17) as f32 - 8.0) / 29.0)
                .collect(),
        )
        .unwrap();
        let expected = serial
            .install(|| filter.backward(&input, &gradient))
            .unwrap();
        let actual = parallel
            .install(|| filter.backward(&input, &gradient))
            .unwrap();
        close(&actual.input.data, &expected.input.data);
        close(&actual.half, &expected.half);
    }
}
