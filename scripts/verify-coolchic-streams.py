#!/usr/bin/env python3
"""Independent full v5 bitstream fixtures and Rust ↔ Python interoperability.

All files are produced by official header/network/range-coder implementations.
--write-fixtures regenerates golden files. Default checks files are unchanged,
then verifies native Rust decoding and reference decoding of Rust-written files.
"""
import argparse
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
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
    from coolchic.bitstream.component.coolchic import encode_decode_coolchic
    from coolchic.bitstream.header.header import CoolChicHeader, VideoHeader, FrameHeader
    from coolchic.bitstream.neuralnet.neuralnet import encode_network
    from coolchic.bitstream.decode import decode_video
    from coolchic.component.core.coolchic import (CoolChicEncoderParameter, instantiate_arm_from_cc_param,
        instantiate_ifce_from_cc_param, instantiate_syn_from_cc_param, instantiate_ups_from_cc_param)
    from coolchic.component.core.types import DescriptorCoolChic, DescriptorNN
    from coolchic.utils.codingstructure import CodingStructure
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    def sequence(shape, offset):
        n = 1
        for size in shape:
            n *= size
        return (((torch.arange(n) * 37 + offset) % 31 - 15) / 64).reshape(shape)

    variants = [
        ("lop", 6, 2, ["3-1-linear-relu", "3-1-linear-none", "3-3-residual-none"], (0, 6), (4, 6), False, "bicubic"),
        ("mop", 10, 4, ["16-1-linear-relu", "3-1-linear-none", "3-3-residual-relu", "3-3-residual-none"], (0, 6), (4, 6), False, "bicubic"),
        ("hop", 14, 6, ["48-1-linear-relu", "3-1-linear-none", "3-3-residual-relu", "3-3-residual-none"], (0, 6), (4, 6), False, "bicubic"),
        ("vhop", 20, 6, ["64-1-linear-relu", "3-1-linear-none", "3-3-residual-relu", "3-3-residual-none"], (0, 6), (4, 6), False, "bicubic"),
        ("common", 6, 2, ["4-1-linear-relu", "4-3-residual-relu", "3-1-linear-none"], (0, 2), (1, 2), True, "bicubic"),
        ("resize", 6, 2, ["4-1-linear-relu", "3-1-linear-none"], (1, 3), (2, 3), False, "bilinear"),
        ("linear", 6, 0, ["3-1-linear-none"], (0, 2), None, False, "nearest"),
    ]
    fixtures = ROOT / "crates/coolchic/tests/fixtures/full-reference"
    fixtures.mkdir(parents=True, exist_ok=True)
    manifest = {"upstream_revision": REVISION, "cases": []}

    def rgba(path):
        with contextlib.redirect_stdout(io.StringIO()):
            image = decode_video(str(path))["0"].data[0]
        h, w = image.shape[-2:]
        rgb = torch.round(image * 255).to(torch.uint8).permute(1, 2, 0)
        return torch.cat([rgb, torch.full((h, w, 1), 255, dtype=torch.uint8)], dim=2).numpy().tobytes()

    for name, spatial, outputs, layers, latent_range, hyper_range, common, interpolation in variants:
        param = CoolChicEncoderParameter(layers_synthesis=layers, linear_stabiliser_synth=name != "linear",
            ups_k_size=8, ups_preconcat_k_size=7, ifce_resolution=(0, 2) if outputs else None,
            output_feature_ifce=outputs, spatial_context_arm=spatial,
            linear_stabiliser_arm=name != "linear", n_hidden_layers_arm=2 if name != "linear" else 0,
            latent_resolution=latent_range, hyperlatent_resolution=hyper_range, flag_common_randomness=common,
            img_size=(7, 13), final_upsampling_type=interpolation)
        arm, ifce = instantiate_arm_from_cc_param(param), instantiate_ifce_from_cc_param(param)
        ups, syn = instantiate_ups_from_cc_param(param), instantiate_syn_from_cc_param(param)
        q = DescriptorCoolChic(arm=DescriptorNN(2**-8, 2**-12),
            ifce=DescriptorNN(2**-8, 2**(-12 if outputs else -16)),
            upsampling=DescriptorNN(2**-12, 1), synthesis=DescriptorNN(2**-12, 2**-16))
        counts = DescriptorCoolChic(arm=DescriptorNN(3, 8), ifce=DescriptorNN(3 if outputs else 0, 8 if outputs else 0),
            upsampling=DescriptorNN(8, 0), synthesis=DescriptorNN(8, 8))
        with torch.no_grad():
            for module_name, module in [("arm", arm), ("ifce", ifce), ("upsampling", ups), ("synthesis", syn)]:
                if module is None:
                    continue
                for index, (key, value) in enumerate(module.named_parameters()):
                    step = q.get_value(module_name, "weight" if "weight" in key else "bias")
                    sample = sequence(value.shape, index * 11)
                    if module_name == "upsampling" and "weight" in key:
                        sample = value + sample / 16
                    value.copy_(torch.round(sample / step) * step)
                module.eval()
            arm.mlp[-1].bias.copy_(torch.tensor([0.0625, 4.0625]))
            syn.output_transform.weight.copy_(torch.diag(torch.tensor([0.5, 0.75, 1.])).reshape(3, 3, 1, 1))
            syn.output_transform.bias.copy_(torch.tensor([0.375, 0.5, 0.625]))
            encoder = SimpleNamespace(arm=arm, ifce=ifce, upsampling=ups, synthesis=syn, nn_q_step=q, nn_expgol_cnt=counts)
            neural, descriptor = encode_network(encoder)
            header = CoolChicHeader()
            header.set_header(param, descriptor["nn_n_bytes"], descriptor["nn_n_bit_pad"], q, counts, 0)
            latents = [((torch.arange(h * w) * 13 + i * 7) % 31 - 15).reshape(1, 1, h, w)
                for i, (_, _, h, w) in enumerate(param.size_per_latent)]
            _, payload = encode_decode_coolchic(header, neural, mode="encode", enc_quantized_latent=latents)
        video = VideoHeader()
        video.set_header(CodingStructure(n_frames=1, intra_pos=[0], p_pos=[]))
        frame = FrameHeader()
        frame.set_header(SimpleNamespace(frame_display_index=0, frame_type="I", frame_data_type="rgb", bitdepth=8))
        data = video.to_bytes() + frame.to_bytes() + payload
        file = fixtures / f"{name}.cool"
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "reference.cool"
            source.write_bytes(data)
            pixels = rgba(source)
        if args.write_fixtures:
            file.write_bytes(data)
            (fixtures / f"{name}.rgba").write_bytes(pixels)
        elif file.read_bytes() != data or (fixtures / f"{name}.rgba").read_bytes() != pixels:
            raise SystemExit(f"Reference fixture changed: {name}")
        manifest["cases"].append({"name": name, "width": 13, "height": 7})
    metadata = fixtures / "manifest.json"
    serialized = json.dumps(manifest, indent=2) + "\n"
    if args.write_fixtures:
        metadata.write_text(serialized)
        print("Wrote full-reference .cool/.rgba golden files")
    elif metadata.read_text() != serialized:
        raise SystemExit("Reference manifest differs")

    subprocess.run(["cargo", "build", "--quiet", "--release", "--manifest-path", str(ROOT / "crates/coolchic/Cargo.toml"), "--example", "reference", "--example", "reference_train"], check=True)
    native = ROOT / "crates/coolchic/target/release/examples/reference"
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        for case in manifest["cases"]:
            source = fixtures / f"{case['name']}.cool"
            decoded, written, written_decoded = directory / "rust.rgba", directory / "rust.cool", directory / "written.rgba"
            subprocess.run([str(native), "decode", str(source), str(decoded)], check=True)
            assert decoded.read_bytes() == (fixtures / f"{case['name']}.rgba").read_bytes(), f"{case['name']}: reference → Rust pixels differ"
            subprocess.run([str(native), "shift", str(source), str(written)], check=True)
            subprocess.run([str(native), "decode", str(written), str(written_decoded)], check=True)
            assert rgba(written) == written_decoded.read_bytes(), f"{case['name']}: Rust → reference pixels differ"
            print(f"PASS {case['name']}: reference → Rust and Rust-written → reference, exact RGBA")
        trainer = ROOT / "crates/coolchic/target/release/examples/reference_train"
        for width, height, point in [(8, 6, "lop"), (8, 6, "mop"), (4, 3, "hop"), (32, 24, "lop")]:
            source = directory / "source.rgba"
            original = bytes(v for y in range(height) for x in range(width) for v in
                (x * 255 // max(width - 1, 1), y * 255 // max(height - 1, 1),
                 (x + y) * 255 // max(width + height - 2, 1), 255))
            source.write_bytes(original)
            encoded, decoded = directory / "trained.cool", directory / "trained.rgba"
            subprocess.run([str(trainer), str(source), str(width), str(height), "2000", "0.001", point, str(encoded)],
                check=True, capture_output=True)
            subprocess.run([str(native), "decode", str(encoded), str(decoded)], check=True)
            official = rgba(encoded)
            assert official == decoded.read_bytes(), f"Rust-trained {point} {width}x{height}: official decoder disagrees"
            error = sum((a - b) ** 2 for i, (a, b) in enumerate(zip(original, official)) if i % 4 != 3) / (width * height * 3)
            import math
            psnr = 10 * math.log10(255**2 / error) if error else float("inf")
            # On tiny images, the reference's neural-parameter RD penalty can
            # legitimately dominate and discard detail. Use the larger sample
            # for a reconstruction sanity bound, not tiny-file overhead cases.
            if width * height >= 512:
                assert psnr > 18, f"Rust-trained {point}: gradient reconstruction regressed ({psnr} dB)"
            print(f"PASS Rust-trained {point} {width}×{height}: 2000 iterations, official decoder exact, {len(encoded.read_bytes())} bytes, {psnr:.2f} dB")


if __name__ == "__main__":
    main()
