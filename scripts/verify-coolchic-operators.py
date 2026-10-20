#!/usr/bin/env python3
"""Generate/check neural forward+backward fixtures using the unmodified v5 reference.

Offline development only; PyTorch is NOT an application dependency.
  .cache/coolchic-venv/bin/python scripts/verify-coolchic-operators.py \
      --reference /path/to/Cool-Chic --write-fixtures
  cargo test --manifest-path crates/coolchic/Cargo.toml --test reference_operators
"""
import argparse
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
    actual = subprocess.check_output(["git", "-C", str(reference), "rev-parse", "HEAD"], text=True).strip()
    if actual != REVISION:
        raise SystemExit(f"Expected upstream {REVISION}, found {actual}")
    if subprocess.check_output(["git", "-C", str(reference), "status", "--porcelain", "--untracked-files=no"], text=True).strip():
        raise SystemExit("Reference checkout must not have modified tracked files")
    sys.path.insert(0, str(reference))
    import torch
    from coolchic.component.core.synthesis import Synthesis, SynthesisConv2d
    from coolchic.component.core.upsampling import (
        UpsamplingSeparableSymmetricConv2d,
        UpsamplingSeparableSymmetricConvTranspose2d,
    )
    from coolchic.component.core.quantizer import quantize, softround, generate_kumaraswamy_noise

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    def sequence(shape, offset=0, scale=1 / 32):
        n = 1
        for dimension in shape:
            n *= dimension
        return (((torch.arange(n) * 37 + offset) % 31 - 15) * scale).reshape(shape)

    def flat(tensor):
        return tensor.detach().reshape(-1).tolist()

    def tensor(tensor):
        return {"shape": list(tensor.shape[1:]), "data": flat(tensor)}

    def fill(module):
        with torch.no_grad():
            for i, parameter in enumerate(module.parameters()):
                parameter.copy_(sequence(parameter.shape, offset=i * 11, scale=1 / 64))
                parameter.requires_grad_(True)

    def conv(module, relu=False):
        return {
            "input_channels": module.in_channels, "output_channels": module.out_channels,
            "kernel": module.kernel_size, "residual": module.residual, "relu": relu,
            "weight": flat(module.weight), "bias": flat(module.bias),
        }

    def conv_grad(module):
        return {"weight": flat(module.weight.grad), "bias": flat(module.bias.grad)}

    cases = {"upstream_revision": REVISION, "convolution": [], "synthesis": [], "filter": [], "quantizer": []}
    for height, width in [(1, 1), (3, 5)]:
        for kernel, residual, relu in [(1, False, False), (3, False, True), (3, True, True)]:
            module = SynthesisConv2d(3, 3 if residual else 2, kernel, residual)
            fill(module)
            x = sequence((1, 3, height, width), offset=7).requires_grad_()
            y = module(x)
            if relu:
                y = torch.relu(y)
            upstream = sequence(y.shape, offset=13)
            (y * upstream).sum().backward()
            cases["convolution"].append({"layer": conv(module, relu), "input": tensor(x),
                "output": tensor(y), "upstream": tensor(upstream), "input_grad": tensor(x.grad),
                "parameter_grad": conv_grad(module)})

    for stabiliser, common_randomness in [(False, False), (True, False), (True, True)]:
        channels = 6 if common_randomness else 3
        module = Synthesis(channels, ["4-1-linear-relu", "4-3-residual-relu", "3-1-linear-none"],
            flag_linear_stabiliser=stabiliser, flag_common_randomness=common_randomness)
        fill(module)
        x = sequence((1, channels, 3, 5), offset=17).requires_grad_()
        y = module(x)
        upstream = sequence(y.shape, offset=5)
        (y * upstream).sum().backward()
        convolutions = [layer for layer in module.main_branch if isinstance(layer, SynthesisConv2d)]
        cases["synthesis"].append({
            "trunk": [conv(layer, i < len(convolutions) - 1) for i, layer in enumerate(convolutions)],
            "stabiliser": conv(module.stabiliser_branch) if stabiliser else None,
            "output_transform": conv(module.output_transform),
            "input": tensor(x), "output": tensor(y), "upstream": tensor(upstream),
            "input_grad": tensor(x.grad), "trunk_grad": [conv_grad(layer) for layer in convolutions],
            "stabiliser_grad": conv_grad(module.stabiliser_branch) if stabiliser else None,
            "output_transform_grad": conv_grad(module.output_transform),
        })

    for height, width in [(1, 1), (3, 5)]:
        for mode, kernels, klass in [
            ("residual", [1, 3, 5], UpsamplingSeparableSymmetricConv2d),
            ("upsample", [4, 6, 8, 10], UpsamplingSeparableSymmetricConvTranspose2d),
        ]:
            for kernel in kernels:
                module = klass(kernel)
                fill(module)
                # Upstream uses one feature per grid, merged into batch for upsampling.
                x = sequence((1, 1, height, width), offset=3).requires_grad_()
                module.train()
                training_output = module(x)
                upstream = sequence(training_output.shape, offset=9)
                (training_output * upstream).sum().backward()
                module.eval()
                output = module(x.detach())
                original = module.parametrizations.weight.original
                cases["filter"].append({"mode": mode, "kernel": kernel, "half": flat(original),
                    "input": tensor(x), "output": tensor(output), "training_output": tensor(training_output),
                    "upstream": tensor(upstream), "input_grad": tensor(x.grad), "half_grad": flat(original.grad)})

    # Capture reference noise before quantize consumes it, then replay the seed.
    # This tests the actual upstream implementation, not a reimplementation of its formula.
    for mode in ["none", "softround_alone", "softround", "ste", "hardround"]:
        for temperature in [0.08, 0.35, 1.0]:
            x = (torch.arange(21, dtype=torch.float32) / 4 - 2.5).requires_grad_()
            t = torch.tensor(temperature)
            torch.manual_seed(27)
            noise = torch.randn_like(x) * 0.22
            torch.manual_seed(27)
            output = quantize(x, "gaussian", mode, t, torch.tensor(0.22))
            # v5's STE branch assigns inside no_grad, detaching the output.
            # Record actual behavior, not just the derivative claimed in its docs.
            if output.requires_grad:
                output.sum().backward()
            cases["quantizer"].append({"mode": mode, "temperature": temperature, "input": flat(x),
                "noise": flat(noise), "output": flat(output),
                "derivative": flat(x.grad) if x.grad is not None else [0.0] * x.numel()})

    uniform = torch.tensor([0.0, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0])
    cases["noise"] = [{"a": a, "uniform": flat(uniform),
        "output": flat(generate_kumaraswamy_noise(uniform, torch.tensor(a)))} for a in [1.0, 2.0, 5.0]]
    path = ROOT / "crates/coolchic/tests/fixtures/reference-operators.json"
    serialized = json.dumps(cases, separators=(",", ":"), allow_nan=False) + "\n"
    if args.write_fixtures:
        path.write_text(serialized)
        print(f"Wrote {path.relative_to(ROOT)} from unmodified upstream {REVISION}")
    elif path.read_text() != serialized:
        raise SystemExit("Fixtures differ; inspect changes before regenerating")
    else:
        print(f"PASS: committed operator fixtures match upstream {REVISION}")


if __name__ == "__main__":
    main()
