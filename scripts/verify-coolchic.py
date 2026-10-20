#!/usr/bin/env python3
"""Optional offline interoperability test against the pinned official Python codec.

Not needed to build or run the application. Requires the upstream Python deps.
Run from repo root with an appropriately provisioned Python interpreter:
  python scripts/verify-coolchic.py --reference /path/to/Cool-Chic
Pass --write-fixtures to regenerate the small reference-generated golden files.
"""
import argparse
import contextlib
import io
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
    ref = args.reference.resolve()
    actual = subprocess.check_output(["git", "-C", str(ref), "rev-parse", "HEAD"], text=True).strip()
    if actual != REVISION:
        raise SystemExit(f"Expected reference commit {REVISION}, got {actual}")
    sys.path.insert(0, str(ref))
    import torch
    from coolchic.bitstream.decode import decode_video
    from coolchic.bitstream.component.coolchic import encode_decode_coolchic
    from coolchic.bitstream.header.header import CoolChicHeader, FrameHeader, VideoHeader
    from coolchic.bitstream.neuralnet.neuralnet import encode_network
    from coolchic.component.core.coolchic import (
        CoolChicEncoderParameter, instantiate_arm_from_cc_param,
        instantiate_syn_from_cc_param, instantiate_ups_from_cc_param,
    )
    from coolchic.component.core.types import DescriptorCoolChic, DescriptorNN
    from coolchic.utils.codingstructure import CodingStructure

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    subprocess.run(["cargo", "build", "--quiet", "--release", "--manifest-path", str(ROOT / "crates/coolchic/Cargo.toml"), "--example", "rgba"], check=True)
    native = ROOT / "crates/coolchic/target/release/examples/rgba"
    fixtures = ROOT / "crates/coolchic/tests/fixtures"

    def official_rgba(path):
        with contextlib.redirect_stdout(io.StringIO()):
            image = decode_video(str(path))["0"].data[0]
        h, w = image.shape[-2:]
        rgb = torch.round(image * 255).to(torch.uint8).permute(1, 2, 0)
        alpha = torch.full((h, w, 1), 255, dtype=torch.uint8)
        return torch.cat([rgb, alpha], dim=2).numpy().tobytes()

    # Independent reference writer: configure actual upstream modules, not Rust
    # headers or a repacked Rust stream. Nonzero ARM scale weights exercise the
    # fixed-point arithmetic and probability table over a range of indices.
    def official_stream(w, h):
        param = CoolChicEncoderParameter(
            layers_synthesis=["3-1-linear-none"], linear_stabiliser_synth=False,
            ups_k_size=4, ups_preconcat_k_size=1, ifce_resolution=None,
            output_feature_ifce=0, spatial_context_arm=6, linear_stabiliser_arm=False,
            n_hidden_layers_arm=0, latent_resolution=(0, 2), hyperlatent_resolution=None,
            flag_common_randomness=False, img_size=(h, w), final_upsampling_type="nearest",
        )
        arm = instantiate_arm_from_cc_param(param)
        ups = instantiate_ups_from_cc_param(param)
        syn = instantiate_syn_from_cc_param(param)
        q = DescriptorCoolChic(
            arm=DescriptorNN(2**-8, 2**-16), ifce=DescriptorNN(2**-8, 2**-16),
            upsampling=DescriptorNN(2**-12, 1), synthesis=DescriptorNN(2**-12, 2**-16),
        )
        counts = DescriptorCoolChic(
            arm=DescriptorNN(2, 8), ifce=DescriptorNN(0, 0),
            upsampling=DescriptorNN(4, 0), synthesis=DescriptorNN(8, 8),
        )
        values = {
            "arm": ([192, 64, -64, 0, 32, 0, 1, -2, 0, 0, 0, 0], [4096, 6 * 65536]),
            "upsampling": ([0, 4096, 0, 4096, 0, 0], [0] * 4),
            "synthesis": ([4096, 0, 0, 0, 4096, 0, 0, 0, 4096,
                           128, 0, 128, 128, -25, -65, 128, 128, 0], [0, 0, 0, 32768, 32768, 32768]),
        }
        with torch.no_grad():
            for name, module in [("arm", arm), ("upsampling", ups), ("synthesis", syn)]:
                loaded = {}
                for idx, kind in enumerate(["weight", "bias"]):
                    ptr = 0
                    for key, tensor in module.get_param(which=kind).items():
                        size = tensor.numel()
                        loaded[key] = torch.tensor(values[name][idx][ptr:ptr + size], dtype=torch.float32).reshape_as(tensor) * q.get_value(name, kind)
                        ptr += size
                    assert ptr == len(values[name][idx])
                module.load_state_dict(loaded, strict=True)
                module.eval()
            enc = SimpleNamespace(arm=arm, ifce=None, upsampling=ups, synthesis=syn, nn_q_step=q, nn_expgol_cnt=counts)
            neural, descriptor = encode_network(enc)
            header = CoolChicHeader()
            header.set_header(param, descriptor["nn_n_bytes"], descriptor["nn_n_bit_pad"], q, counts, 0)
            latents = []
            for i, shape in enumerate(param.size_per_latent):
                n = shape[-2] * shape[-1]
                z = ((torch.arange(n) * 37 + i * 11) % 64 - 32).reshape(shape)
                latents.append(z)
            _, payload = encode_decode_coolchic(header, neural, mode="encode", enc_quantized_latent=latents)
        video = VideoHeader()
        video.set_header(CodingStructure(n_frames=1, intra_pos=[0], p_pos=[]))
        frame = FrameHeader()
        frame.set_header(SimpleNamespace(frame_display_index=0, frame_type="I", frame_data_type="rgb", bitdepth=8))
        return video.to_bytes() + frame.to_bytes() + payload

    with tempfile.TemporaryDirectory(prefix="coolchic-interop-") as temp:
        temp = Path(temp)
        for w, h in [(1, 1), (3, 5), (9, 11), (13, 17), (32, 24)]:
            raw = bytes(v for p in range(w * h) for v in [p % w * 255 // w, p // w * 255 // h, p * 17 % 256, 255])
            src = temp / "source.rgba"
            src.write_bytes(raw)
            for quality in [1, 75, 100]:
                rust = temp / "rust.cool"
                decoded = temp / "rust.rgba"
                subprocess.run([str(native), "encode", str(src), str(w), str(h), str(quality), "1", str(rust)], check=True)
                subprocess.run([str(native), "decode", str(rust), str(decoded)], check=True, capture_output=True)
                assert official_rgba(rust) == decoded.read_bytes(), (w, h, quality, "Rust → reference pixel mismatch")
            official = temp / "reference.cool"
            official.write_bytes(official_stream(w, h))
            expected = official_rgba(official)
            decoded = temp / "reference.rgba"
            subprocess.run([str(native), "decode", str(official), str(decoded)], check=True, capture_output=True)
            assert expected == decoded.read_bytes(), (w, h, "reference → Rust pixel mismatch")
            if args.write_fixtures and (w, h) in [(3, 5), (13, 17)]:
                fixtures.mkdir(parents=True, exist_ok=True)
                (fixtures / f"reference-{w}x{h}.cool").write_bytes(official.read_bytes())
                (fixtures / f"reference-{w}x{h}.rgba").write_bytes(expected)
            print(f"PASS {w}×{h}: Rust → official (three qualities), official → Rust; exact RGBA agreement")


if __name__ == "__main__":
    main()
