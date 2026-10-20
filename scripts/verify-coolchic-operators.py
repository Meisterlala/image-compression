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
    from coolchic.training.soap import SOAP
    from coolchic.component.core.arm import Arm, ArmLinear, Ifce, _get_neighbor, _get_non_zero_pixel_ctx_index, _laplace_cdf
    from coolchic.component.core.upsampling import (
        Upsampling, fixed_upsampling, UpsamplingSeparableSymmetricConv2d,
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

    cases = {"upstream_revision": REVISION, "pytorch_version": str(torch.__version__), "convolution": [], "synthesis": [], "filter": [],
        "quantizer": [], "arm": [], "context": [], "rate": [], "pyramid": [], "soap": [], "ifce": []}
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

    def affine(module, relu):
        return {"input_channels": module.in_channels, "output_channels": module.out_channels,
            "kernel": 1, "residual": module.residual, "relu": relu,
            "weight": flat(module.weight), "bias": flat(module.bias)}

    def pointwise(value):
        return tensor(value.T.reshape(1, value.shape[1], 1, value.shape[0]))

    for dimension, hidden, stabiliser, outputs in [(6, 0, False, 2), (6, 2, True, 2), (16, 2, True, 2), (3, 0, False, 4)]:
        module = Arm(dimension, hidden, outputs, stabiliser)
        fill(module)
        x = sequence((15, dimension), offset=5).requires_grad_()
        y = module(x)
        upstream = sequence(y.shape, offset=9)
        (y * upstream).sum().backward()
        layers = [layer for layer in module.mlp if isinstance(layer, ArmLinear)]
        cases["arm"].append({"layers": [affine(layer, i < len(layers) - 1) for i, layer in enumerate(layers)],
            "stabiliser": affine(module.stabiliser_branch, False) if stabiliser else None,
            "input": pointwise(x), "output": pointwise(y), "upstream": pointwise(upstream),
            "input_grad": pointwise(x.grad), "layer_grad": [conv_grad(layer) for layer in layers],
            "stabiliser_grad": conv_grad(module.stabiliser_branch) if stabiliser else None})

    for count in [0, 6, 16, 40]:
        x = sequence((1, 1, 3, 5), offset=9).requires_grad_()
        neighbours = _get_neighbor(x, _get_non_zero_pixel_ctx_index(count), 9)
        y = neighbours.reshape(15, count).T.reshape(1, count, 3, 5)
        upstream = sequence(y.shape, offset=13)
        (y * upstream).sum().backward()
        cases["context"].append({"input": tensor(x), "count": count, "output": tensor(y),
            "upstream": tensor(upstream), "input_grad": tensor(x.grad)})

    x = (torch.arange(15, dtype=torch.float32) / 4 - 2.0).requires_grad_()
    raw = torch.stack([x.detach() + torch.tensor([0.5, -0.5, 0.0, 0.25, -0.25] * 3),
        torch.tensor([-2., -1., 0., 1., 2., 3., 4., 5., 6., 8., 9., 10., 11., 14., -6.])], dim=1).requires_grad_()
    arm = Arm(6, 0)
    mu, scale = arm.reparameterize_output(raw)
    probability = torch.clamp_min(_laplace_cdf(x + 0.5, mu, scale) - _laplace_cdf(x - 0.5, mu, scale), 2**-16)
    bits = -torch.log2(probability)
    bits.sum().backward()
    cases["rate"].append({"latent": tensor(x.reshape(1, 1, 3, 5)), "raw_arm": pointwise(raw),
        "bits": tensor(bits.reshape(1, 1, 3, 5)), "latent_grad": tensor(x.grad.reshape(1, 1, 3, 5)),
        "arm_grad": pointwise(raw.grad)})
    # Keep raw ARM spatial shape aligned with latent spatial shape.
    cases["rate"][-1]["raw_arm"]["shape"] = [2, 3, 5]
    cases["rate"][-1]["arm_grad"]["shape"] = [2, 3, 5]

    for shared in [False, True]:
        module = Upsampling(8, 3, 1 if shared else 2, 1 if shared else 2)
        fill(module)
        module.eval()
        latents = [sequence((1, 1, h, w), offset=i * 7).requires_grad_()
            for i, (h, w) in enumerate([(5, 7), (3, 4), (2, 2)])]
        y = module(latents)
        upstream = sequence(y.shape, offset=13)
        (y * upstream).sum().backward()
        def filter_params(module, mode):
            original = module.parametrizations.weight.original
            return {"kernel": module.target_k_size, "half": flat(original), "half_grad": flat(original.grad), "mode": mode}
        cases["pyramid"].append({"latents": [tensor(x) for x in latents],
            "latent_grad": [tensor(x.grad) for x in latents], "output": tensor(y), "upstream": tensor(upstream),
            "upsampling": [filter_params(f, "upsample") for f in module.conv_transpose2ds],
            "preconcat": [filter_params(f, "residual") for f in module.conv2ds]})

    for shapes, enabled in [([(5, 7), (3, 4), (2, 2)], [True, True, True]),
            ([(3, 5), (2, 3), (1, 2), (1, 1)], [True, False, True, True])]:
        sizes = [max(len(shapes) - i - 1, 1) if flag else 0 for i, flag in enumerate(enabled)]
        module = Ifce(sizes, 4)
        fill(module)
        latents = [sequence((1, 1, h, w), offset=i * 9).requires_grad_() for i, (h, w) in enumerate(shapes)]
        _, intermediate = fixed_upsampling(latents, mode="nearest")
        results, upstreams = [], []
        total = torch.tensor(0.)
        for index, (latent, flag) in enumerate(zip(latents, enabled)):
            if flag:
                coarse = intermediate[len(latents) - 1 - index]
                coarse_flat = coarse.permute(0, 2, 3, 1).reshape(-1, coarse.shape[1])
                raw = module(coarse_flat, index)
                raw = raw.reshape(1, coarse.shape[2], coarse.shape[3], 4).permute(0, 3, 1, 2)
                result = torch.nn.functional.interpolate(raw, scale_factor=2., mode="nearest")[:, :, :latent.shape[2], :latent.shape[3]]
            else:
                result = torch.zeros(1, 4, *latent.shape[-2:])
            upstream = sequence(result.shape, offset=index * 5)
            total = total + (result * upstream).sum()
            results.append(result)
            upstreams.append(upstream)
        total.backward()
        networks, gradients = [], []
        for index, flag in enumerate(enabled):
            if flag:
                arm = module.arms[module.index_to_arm[index]]
                layers = [layer for layer in arm.mlp if isinstance(layer, ArmLinear)]
                networks.append({"layers": [affine(layer, False) for layer in layers]})
                gradients.append([conv_grad(layer) for layer in layers])
            else:
                networks.append(None)
                gradients.append(None)
        cases["ifce"].append({"outputs": 4, "networks": networks, "parameter_grad": gradients,
            "latents": [tensor(x) for x in latents], "output": [tensor(x) for x in results],
            "upstream": [tensor(x) for x in upstreams],
            "latent_grad": [tensor(x.grad if x.grad is not None else torch.zeros_like(x)) for x in latents]})

    uniform = torch.tensor([0.0, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0])
    cases["noise"] = [{"a": a, "uniform": flat(uniform),
        "output": flat(generate_kumaraswamy_noise(uniform, torch.tensor(a)))} for a in [1.0, 2.0, 5.0]]
    for shape, group, max_dimension in [([3, 2], "network", 256), ([2, 3, 3, 3], "network", 256),
            ([5], "network", 256), ([3, 2], "network", 2), ([1, 1, 3, 5], "latent", 0)]:
        torch.manual_seed(873)
        initial = torch.randn(shape) * 0.2
        parameter = torch.nn.Parameter(initial.clone())
        config = {"lr": 0.01, "betas": (0.9, 0.999) if group == "latent" else (0.95, 0.95),
            "precondition_frequency": 1 if group == "latent" else 10,
            "max_precond_dim": max_dimension, "merge_dims": False, "precondition_1d": False,
            "weight_decay": 0.0 if group == "latent" else 0.01}
        optimizer = SOAP([parameter], **config)
        steps = []
        for index in range(25):
            gradient = torch.randn(shape) * 0.1
            parameter.grad = gradient
            learning_rate = 0.01 if index < 12 else 0.002
            optimizer.param_groups[0]["lr"] = learning_rate
            optimizer.step()
            steps.append({"learning_rate": learning_rate, "gradient": flat(gradient), "parameter": flat(parameter)})
        cases["soap"].append({"shape": shape, "group": group, "max_dimension": max_dimension,
            "initial": flat(initial), "steps": steps})

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
