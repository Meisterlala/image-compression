#!/usr/bin/env python3
"""Verify standard intra schedules and greedy quantization against real v5 code."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

REVISION = "a2f9f84f9e1ab31cacc0a882fdba0cd900add4d6"
ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--write-fixtures", action="store_true")
    args = parser.parse_args()
    reference = args.reference.resolve()
    if subprocess.check_output(["git", "-C", str(reference), "rev-parse", "HEAD"], text=True).strip() != REVISION:
        raise SystemExit("Wrong reference revision")
    if subprocess.check_output(["git", "-C", str(reference), "status", "--porcelain", "--untracked-files=no"], text=True).strip():
        raise SystemExit("Reference checkout must be unmodified")
    sys.path.insert(0, str(reference))
    import torch
    from coolchic.component.core.coolchic import CoolChicEncoderParameter, exp_golomb_nbins
    from coolchic.component.frame import FrameEncoder
    from coolchic.training.presets import PresetIntra
    from coolchic.nnquant.quantizemodel import quantize_model
    from coolchic.nnquant.quantstep import POSSIBLE_Q_STEP
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    result = {"upstream_revision": REVISION, "pytorch_version": str(torch.__version__), "schedules": [], "quantization": []}
    for iterations in [2000, 4000, 9000, 10000, 12000, 20000]:
        with contextlib.redirect_stdout(io.StringIO()):
            preset = PresetIntra(lmbda=0.001, start_lr=0.01, itr_main_training=iterations)
        def phase(p):
            return {"iterations": p.max_itr, "learning_rate": p.lr, "validation_frequency": p.freq_valid,
                "patience": p.patience, "cosine": p.schedule_lr, "temperature": p.softround_temperature,
                "noise": p.noise_parameter, "quantizer": p.quantizer_type}
        result["schedules"].append({"iterations": iterations, "warmup": phase(preset.warmup.phases[0].training_phase),
            "core": phase(preset.training_phases[0]), "hard": phase(preset.training_phases[1])})
    graph_cases = json.loads((ROOT / "crates/coolchic/tests/fixtures/reference-graph.json").read_text())["graphs"]
    for case in graph_cases[:2]:
        a = case["architecture"]
        layers = [f"{l['outputs']}-{l['kernel']}-{'residual' if l['residual'] else 'linear'}-{'relu' if l['relu'] else 'none'}" for l in a["synthesis"]]
        parameter = CoolChicEncoderParameter(layers_synthesis=layers, linear_stabiliser_synth=a["synthesis_stabiliser"],
            ups_k_size=a["up_kernel"], ups_preconcat_k_size=a["pre_kernel"], ifce_resolution=tuple(a["ifce_range"]),
            output_feature_ifce=a["ifce_outputs"], spatial_context_arm=a["spatial_context"],
            linear_stabiliser_arm=a["arm_stabiliser"], n_hidden_layers_arm=a["arm_hidden"],
            latent_resolution=(a["latent_min"], a["latent_max"]),
            hyperlatent_resolution=tuple(a["hyperlatent"]) if a["hyperlatent"] else None,
            flag_common_randomness=a["common_randomness"], img_size=(a["height"], a["width"]), final_upsampling_type=a["interpolation"])
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            frame_encoder = FrameEncoder({"residue": parameter})
        cc = frame_encoder.coolchic_enc["residue"]
        with torch.no_grad():
            for i, source in enumerate(case["latents"]):
                cc.latent_grids[i].copy_(torch.tensor(source["data"]).reshape_as(cc.latent_grids[i]))
            for m, name in enumerate(["arm", "ifce", "upsampling", "synthesis"]):
                module = getattr(cc, name)
                for kind, label in enumerate(["weight", "bias"]):
                    offset = 0
                    for key, value in module.named_parameters():
                        if label not in key:
                            continue
                        values = case["groups"][m * 2 + kind][offset:offset + value.numel()]
                        value.copy_(torch.tensor(values).reshape_as(value))
                        offset += value.numel()
        target = torch.tensor(case["target"]["data"]).reshape(1, *case["target"]["shape"])
        frame = SimpleNamespace(refs_data=[], data=SimpleNamespace(data=target))
        with contextlib.redirect_stdout(io.StringIO()):
            quantize_model(frame_encoder, frame, {"mse": 1.0}, 0.001)
        groups, indices, orders = [], [], []
        for name in ["arm", "ifce", "upsampling", "synthesis"]:
            module = getattr(cc, name)
            for kind in ["weight", "bias"]:
                groups.append([float(x) for key, value in module.named_parameters() if kind in key for x in value.detach().reshape(-1)])
                step = cc.nn_q_step.get_value(name, kind)
                indices.append(int((POSSIBLE_Q_STEP.get_value(name, kind) == step).nonzero()[0]))
                orders.append(cc.nn_expgol_cnt.get_value(name, kind))
        result["quantization"].append({"graph": case["name"], "q_index": indices, "orders": orders, "groups": groups})
    values = torch.tensor([0., -1., 1., -2., 2., 63., -64., 65535.])
    result["estimated_bits"] = {"values": values.tolist(), "orders": [int(exp_golomb_nbins(values, order)) for order in range(13)]}
    path = ROOT / "crates/coolchic/tests/fixtures/reference-training.json"
    serialized = json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n"
    if args.write_fixtures:
        path.write_text(serialized)
        print("Wrote reference intra schedules and complete neural quantization-search fixtures")
    elif path.read_text() != serialized:
        raise SystemExit("Reference training fixtures differ")
    else:
        print("PASS: schedules, quantization choices and exp-Golomb estimator match reference")


if __name__ == "__main__":
    main()
