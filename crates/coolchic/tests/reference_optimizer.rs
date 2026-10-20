use coolchic::reference::soap::{Config, Soap};

#[path = "support/operators.rs"]
#[allow(dead_code)]
mod support;
use support::{fixtures, floats};

#[test]
fn soap_network_and_adam_like_latent_updates_match_reference_trajectories() {
    for case in fixtures()["soap"].as_array().unwrap() {
        let shape: Vec<usize> = case["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let mut config = if case["group"] == "latent" {
            Config::latent(0.01)
        } else {
            Config::network(0.01)
        };
        config.max_precondition_dim = case["max_dimension"].as_u64().unwrap() as usize;
        let mut optimizer = Soap::new(shape.clone(), config).unwrap();
        let initial = floats(&case["initial"]);
        let mut parameter = initial.clone();
        for (step, data) in case["steps"].as_array().unwrap().iter().enumerate() {
            optimizer
                .set_learning_rate(data["learning_rate"].as_f64().unwrap())
                .unwrap();
            optimizer
                .update(&mut parameter, &floats(&data["gradient"]))
                .unwrap();
            if step == 0 {
                assert_eq!(
                    parameter, initial,
                    "First reference update initializes bases only"
                );
            }
            let expected = floats(&data["parameter"]);
            for (i, (&a, &e)) in parameter.iter().zip(&expected).enumerate() {
                let tolerance = 1e-5 + e.abs() * 1e-4;
                assert!(
                    (a - e).abs() <= tolerance,
                    "shape={shape:?}, group={}, cap={}, step={step}, parameter[{i}]: Rust={a}, Python={e}, tolerance={tolerance}",
                    case["group"],
                    config.max_precondition_dim
                );
            }
        }
        assert_eq!(optimizer.step, 24);
    }
}

#[test]
fn rejects_invalid_settings_and_non_finite_updates() {
    assert!(Soap::new(vec![0], Config::network(0.01)).is_err());
    assert!(Soap::new(vec![1], Config::network(f64::NAN)).is_err());
    assert!(Soap::new(vec![usize::MAX, 3], Config::network(0.01)).is_err());
    let mut state = Soap::new(vec![1], Config::latent(0.01)).unwrap();
    let mut parameter = [1.0];
    assert!(state.update(&mut parameter, &[f32::NAN]).is_err());
    assert_eq!(parameter, [1.0]);
    assert!(state.update(&mut parameter, &[]).is_err());
    assert!(state.set_learning_rate(-1.0).is_err());
}
