use coolchic::reference::{
    codec,
    network::Network,
    preset::{Intra, Phase},
    quantization::{Search, estimated_bits},
    random::Random,
    training::{Settings, Trainer},
};
use serde_json::Value;

#[path = "support/operators.rs"]
#[allow(dead_code)]
mod operators;
#[path = "support/network.rs"]
mod support;
use operators::{close, floats, tensor};

fn fixtures() -> Value {
    serde_json::from_str(include_str!("fixtures/reference-training.json")).unwrap()
}

fn phase(actual: &Phase, expected: &Value) {
    assert_eq!(
        actual.iterations,
        expected["iterations"].as_u64().unwrap() as usize
    );
    assert_eq!(
        actual.validation_frequency,
        expected["validation_frequency"].as_u64().unwrap() as usize
    );
    assert_eq!(
        actual.patience,
        expected["patience"].as_u64().unwrap() as usize
    );
    assert_eq!(actual.cosine, expected["cosine"].as_bool().unwrap());
    assert_eq!(
        actual.learning_rate,
        expected["learning_rate"].as_f64().unwrap()
    );
    close(
        &[actual.temperature.0, actual.temperature.1],
        &floats(&expected["temperature"]),
        "preset temperature",
    );
    close(
        &[actual.noise.0, actual.noise.1],
        &floats(&expected["noise"]),
        "preset noise",
    );
}

#[test]
fn exact_standard_intra_iteration_budgets_and_schedules() {
    for case in fixtures()["schedules"].as_array().unwrap() {
        let iterations = case["iterations"].as_u64().unwrap() as usize;
        let preset = Intra::new(iterations, 0.001, 0.01).unwrap();
        phase(&preset.warmup, &case["warmup"]);
        phase(&preset.core, &case["core"]);
        phase(&preset.hard, &case["hard"]);
        assert_eq!(
            7 * preset.warmup.iterations + preset.core.iterations + preset.hard.iterations,
            iterations
        );
        assert_eq!(
            preset.core.scheduled_values(99),
            preset.core.scheduled_values(0)
        );
        assert!(preset.core.scheduled_values(100).0 < 0.35);
        assert!(preset.core.learning_rate_at(1) < preset.core.learning_rate);
    }
}

#[test]
fn actual_reference_exp_golomb_search_estimator() {
    let f = fixtures();
    let values = floats(&f["estimated_bits"]["values"]);
    for (order, expected) in f["estimated_bits"]["orders"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        assert_eq!(
            estimated_bits(&values, 1.0, order as u32),
            expected.as_u64().unwrap()
        );
    }
    assert_eq!(estimated_bits(&[0.0], 1.0, 0), 0); // Intentional difference from actual file bits.
}

#[test]
fn full_greedy_quantization_search_matches_unmodified_reference() {
    let graphs: Value =
        serde_json::from_str(include_str!("fixtures/reference-graph.json")).unwrap();
    for case in fixtures()["quantization"].as_array().unwrap() {
        let graph = graphs["graphs"]
            .as_array()
            .unwrap()
            .iter()
            .find(|g| g["name"] == case["graph"])
            .unwrap();
        let mut network = Network::new(
            support::architecture(&graph["architecture"]),
            &mut Random::new(1),
        )
        .unwrap();
        network
            .set_parameters(&std::array::from_fn(|i| floats(&graph["groups"][i])))
            .unwrap();
        network.latents = graph["latents"]
            .as_array()
            .unwrap()
            .iter()
            .map(tensor)
            .collect();
        let target = tensor(&graph["target"]);
        let mut search = Search::new(network);
        while !search.step(&target, 0.001).unwrap() {}
        assert_eq!(search.tested, search.total);
        let indices: Vec<u32> = case["q_index"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let orders: Vec<u32> = case["orders"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        assert_eq!(
            search.descriptors.q_index.as_slice(),
            indices,
            "{} quantization indices",
            case["graph"]
        );
        assert_eq!(
            search.descriptors.orders.as_slice(),
            orders,
            "{} exp-Golomb orders",
            case["graph"]
        );
        for (i, group) in search.network.parameters().0.iter().enumerate() {
            assert_eq!(
                *group,
                floats(&case["groups"][i]),
                "{} quantized group {i}",
                case["graph"]
            );
        }
    }
}

#[test]
fn full_training_runs_all_stages_and_emits_a_real_reference_file() {
    let mut trainer = Trainer::new(&[0, 128, 255, 255], 1, 1, Settings::default()).unwrap();
    let mut stages = [false; 5];
    let mut calls = 0;
    loop {
        let status = trainer.status();
        for (index, name) in [
            "Warm-up 1/2",
            "Warm-up 2/2",
            "Training",
            "Hard-round",
            "Quantizing",
        ]
        .iter()
        .enumerate()
        {
            if status.starts_with(name) {
                stages[index] = true;
            }
        }
        calls += 1;
        if trainer.step().unwrap() {
            break;
        }
        assert!(calls < 3000, "Reference trainer did not finish");
    }
    assert!(stages.iter().all(|&s| s));
    assert_eq!(trainer.completed, 2000);
    let file = trainer.finish().unwrap();
    let image = codec::decode(&file).unwrap();
    assert_eq!(image.rgba, [0, 128, 255, 255]);
}

#[test]
fn invalid_or_incomplete_training_is_not_exported() {
    assert!(Intra::new(20, 0.001, 0.01).is_err());
    assert!(Intra::new(2000, f32::NAN, 0.01).is_err());
    assert!(Trainer::new(&[], 1, 1, Settings::default()).is_err());
    assert!(
        Trainer::new(&[0; 4], 1, 1, Settings::default())
            .unwrap()
            .finish()
            .is_err()
    );
}
