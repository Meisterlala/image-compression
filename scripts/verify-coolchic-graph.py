#!/usr/bin/env python3
"""Check/regenerate full reference training-graph fixtures (development only).

Uses unmodified pinned CoolChicEncoder.forward and training.loss.loss_function.
PyTorch is not needed to compile or run the browser app or Rust tests.
"""
import argparse
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys

REVISION = "a2f9f84f9e1ab31cacc0a882fdba0cd900add4d6"
ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--write-fixtures", action="store_true")
    args = parser.parse_args()
    reference = args.reference.resolve()
    if subprocess.check_output(["git", "-C", str(reference), "rev-parse", "HEAD"], text=True).strip() != REVISION:
        raise SystemExit("Wrong upstream reference revision")
    if subprocess.check_output(["git", "-C", str(reference), "status", "--porcelain", "--untracked-files=no"], text=True).strip():
        raise SystemExit("Reference checkout has modified tracked files")
    sys.path.insert(0, str(reference))
    import torch
    from coolchic.component.core.coolchic import CoolChicEncoder, CoolChicEncoderParameter
    from coolchic.component.core.noise import CommonGaussianNoiseGenerator
    from coolchic.training.loss import loss_function

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    def flat(t):
        return t.detach().reshape(-1).tolist()

    def tensor(t):
        return {"shape": list(t.shape[1:]), "data": flat(t)}

    def sequence(shape, offset=0, scale=1 / 64):
        n = 1
        for d in shape:
            n *= d
        return (((torch.arange(n) * 37 + offset) % 31 - 15) * scale).reshape(shape)

    def groups(model, gradients=False):
        result = []
        for name in ["arm", "ifce", "upsampling", "synthesis"]:
            module = getattr(model, name)
            for kind in ["weight", "bias"]:
                values = []
                if module is not None:
                    for key, value in module.named_parameters():
                        if kind not in key:
                            continue
                        values.extend(flat(value.grad if value.grad is not None else torch.zeros_like(value)) if gradients else flat(value))
                result.append(values)
        return result

    result = {"upstream_revision": REVISION, "pytorch_version": str(torch.__version__), "graphs": [], "resize": []}
    variants = [
        ("small", 0, 2, None, False, ["4-1-linear-relu", "4-3-residual-relu", "3-1-linear-none"], 6, 2),
        ("hyper_common", 0, 2, (1, 2), True, ["4-1-linear-relu", "4-3-residual-relu", "3-1-linear-none"], 6, 2),
        ("final_resize", 1, 3, (2, 3), False, ["4-1-linear-relu", "4-3-residual-relu", "3-1-linear-none"], 6, 2),
        ("lop", 0, 6, (4, 6), False, ["3-1-linear-relu", "3-1-linear-none", "3-3-residual-none"], 6, 2),
        ("mop", 0, 6, (4, 6), False, ["16-1-linear-relu", "3-1-linear-none", "3-3-residual-relu", "3-3-residual-none"], 10, 4),
        ("hop", 0, 6, (4, 6), False, ["48-1-linear-relu", "3-1-linear-none", "3-3-residual-relu", "3-3-residual-none"], 14, 6),
    ]
    for name, low, high, hyper, common, layers, spatial, outputs in variants:
        parameter = CoolChicEncoderParameter(layers_synthesis=layers, linear_stabiliser_synth=True,
            ups_k_size=8, ups_preconcat_k_size=7, ifce_resolution=(0, 2), output_feature_ifce=outputs,
            spatial_context_arm=spatial, linear_stabiliser_arm=True, n_hidden_layers_arm=2,
            latent_resolution=(low, high), hyperlatent_resolution=hyper, flag_common_randomness=common,
            img_size=(5, 7), final_upsampling_type="bicubic")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            model = CoolChicEncoder(parameter)
        with torch.no_grad():
            for index, value in enumerate(model.parameters()):
                value.copy_(sequence(value.shape, offset=index * 11))
                value.requires_grad_(True)
            model.synthesis.output_transform.weight.copy_(torch.diag(torch.tensor([0.5, 0.6, 0.7])).reshape(3, 3, 1, 1))
            model.synthesis.output_transform.bias.copy_(torch.tensor([0.4, 0.5, 0.6]))
        target = (sequence((1, 3, 5, 7)) + 0.5).clamp(0., 1.)
        n = sum(x.numel() for x in model.latent_grids)
        torch.manual_seed(27)
        noise = torch.randn(n) * 0.22
        torch.manual_seed(27)
        model.train()
        forward = model(quantizer_noise_type="gaussian", quantizer_type="softround",
            soft_round_temperature=torch.tensor(0.35), noise_parameter=torch.tensor(0.22))
        decoded = forward["raw_out"].clamp(0., 1.)
        loss = loss_function(decoded, {"residue": forward["rate"]}, target, {"mse": 1.0}, lmbda=0.001)
        loss.loss.backward()
        case = {"name": name, "target": tensor(target), "groups": groups(model), "parameter_grad": groups(model, True),
            "latents": [tensor(x) for x in model.latent_grids], "latent_grad": [tensor(x.grad) for x in model.latent_grids],
            "noise": flat(noise), "loss": loss.loss.item(), "decoded": tensor(decoded), "rate": flat(forward["rate"]),
            "architecture": {"height": 5, "width": 7, "latent_min": low, "latent_max": high,
                "hyperlatent": hyper, "ifce_range": [0, 2], "ifce_outputs": outputs, "spatial_context": spatial,
                "arm_hidden": 2, "arm_stabiliser": True, "synthesis_stabiliser": True, "up_kernel": 8,
                "pre_kernel": 7, "common_randomness": common, "interpolation": "bicubic",
                "synthesis": [{"outputs": int(layer.split("-")[0]), "kernel": int(layer.split("-")[1]),
                    "residual": layer.split("-")[2] == "residual", "relu": layer.split("-")[3] == "relu"} for layer in layers]},
            "ifce_inputs": parameter.input_features_ifce,
        }
        model.eval()
        with torch.no_grad():
            forward = model(quantizer_noise_type="none", quantizer_type="hardround")
            decoded = torch.round(forward["raw_out"].clamp(0., 1.) * 255) / 255
            loss = loss_function(decoded, {"residue": forward["rate"]}, target, {"mse": 1.0}, lmbda=0.001)
        case["evaluation"] = {"image": tensor(decoded), "loss": loss.loss.item()}
        result["graphs"].append(case)

    for mode in ["nearest", "bilinear", "bicubic"]:
        for target_shape in [(5, 7), (1, 2)]:
            x = sequence((1, 2, 3, 4), offset=5).requires_grad_()
            options = {} if mode == "nearest" else {"align_corners": False}
            y = torch.nn.functional.interpolate(x, size=target_shape, mode=mode, **options)
            upstream = sequence(y.shape, offset=9)
            (y * upstream).sum().backward()
            result["resize"].append({"mode": mode, "input": tensor(x), "output": tensor(y),
                "upstream": tensor(upstream), "input_grad": tensor(x.grad)})
    generator = CommonGaussianNoiseGenerator()
    result["common_noise"] = [tensor(generator.sample((1, 1, h, w))) for h, w in [(3, 5), (2, 3), (1, 2)]]
    path = ROOT / "crates/coolchic/tests/fixtures/reference-graph.json"
    serialized = json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n"
    if args.write_fixtures:
        path.write_text(serialized)
        print(f"Wrote {path.relative_to(ROOT)}")
    elif path.read_text() != serialized:
        raise SystemExit("Graph fixtures differ; inspect changes before regenerating")
    else:
        print("PASS: graph, interpolation and common-noise fixtures match unmodified v5 reference")


if __name__ == "__main__":
    main()
