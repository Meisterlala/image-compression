# Cool-chic v5

Rust encoder/decoder for RGB still images, ported from Orange Cool-Chic v5.0.0,
commit `a2f9f84f9e1ab31cacc0a882fdba0cd900add4d6`. Runs natively and in browser WASM;
no pretrained weights or Python at runtime. Python/PyTorch are used for parity
checks only.

## Scope

- LOP/MOP/HOP/VHOP architectures, latent/hyperlatent pyramids, learned symmetric
  upsampling, nonlinear synthesis, causal ARM and cross-resolution IFCE.
- Full analytical MSE/entropy gradients; gain-16 soft/noisy latent quantization.
- Reference 5→2→1 warm-up, SOAP network and Adam-like latent optimizers, clipping,
  cosine rates, validation/checkpoint selection and 500 hard-round refinement steps.
- Greedy neural RD quantization, exp-Golomb-order search and reference `.cool`
  bitstreams with Q16/Q32 integer probability inference and range coding.
- Decoding supports common randomness and final nearest/bilinear/bicubic resizing.

Only 8-bit RGB still-image MSE training is exposed, not video, YUV, HDR or
Wasserstein training. The Rust API ignores alpha; callers must composite it first.
Training uses a deterministic local RNG, not PyTorch's generators; learned
parameters need not match across implementations or devices.

The app exposes architecture, λ (larger favors smaller files) and 100–100,000
iterations. Below 2,000 it warns and shortens stages/validation cadence; at 2,000
and above the reference schedule is unchanged. App default: 2,000 iterations;
reference CLI default: 10,000.

Browser limits: 4 million pixels, dimensions below 16384, 3 GiB estimated working
memory, 32 MiB input streams. Estimates are not measured peak RSS. Header/bounds
checks and canonical entropy re-encoding reject malformed streams, but this is
not a hardened hostile-file parser.

## API

```rust,ignore
use coolchic::{encode, decode, EncoderSettings};
let bytes = encode(&rgba, width, height, EncoderSettings::default())?;
let image = decode(&bytes)?;
```

For incremental work, use `Encoder::new`, `step`, `status`, `progress` and `finish`.
Each step performs a training iteration or quantization trial; `finish` rejects
incomplete work. Browser cancellation terminates the worker.

CPU SIMD/threading and optional full-graph GPU training are described in
[acceleration](../../docs/coolchic-acceleration.md). Optimizer/schedule/bitstream
control stays in Rust; previews decode actual bytes on the portable CPU backend.

## Test

From the repository root:

```sh
cargo test --manifest-path crates/coolchic/Cargo.toml
cargo test --manifest-path crates/coolchic/Cargo.toml --features parallel
cargo clippy --manifest-path crates/coolchic/Cargo.toml --all-targets --features parallel -- -D warnings
cargo check --manifest-path crates/coolchic/Cargo.toml --features wasm --target wasm32-unknown-unknown
```

[Fixtures](tests/fixtures/README.md) cover operators, full gradients, optimizer
trajectories, schedules, quantization and seven official streams. Cargo tests
need no Python or network. To verify against upstream, install its requirements
in a development Python environment, then run:

```sh
git clone --depth 1 --branch v5.0.0 https://github.com/Orange-OpenSource/Cool-Chic.git /tmp/coolchic-reference-v5
for check in operators graph training streams; do
  python scripts/verify-coolchic-$check.py --reference /tmp/coolchic-reference-v5
done
```

Use `--write-fixtures` only to regenerate reviewed golden data. Native helpers:

```sh
cargo run --release --manifest-path crates/coolchic/Cargo.toml --example reference_train -- input.rgba 32 24 2000 0.001 lop output.cool
cargo run --release --manifest-path crates/coolchic/Cargo.toml --example reference -- decode output.cool output.rgba
```

## Provenance

Port/application: AGPL-3.0-only (`LICENSE`). Reference algorithms and probability
tables: Copyright 2023–2026 Orange, BSD-3-Clause (`UPSTREAM-LICENSE.txt`). SOAP:
Copyright 2024 Nikhil Vyas, MIT (`SOAP-LICENSE.txt`), with Orange modifications.
Matrix factorization uses nalgebra (Apache-2.0). Deployed notices: `public/legal/`.

`src/log_scales.f32le` stores the final 2,561 float32 entries of upstream
`mu_scale.npy` (`exp(k/256)`, k = −1280…1280, rounded by PyTorch). Exact stored
values prevent encoder/decoder probability-table drift. Dependencies are pinned
in `Cargo.lock`.
