# Cool-chic v5 RGB codec and reference intra MSE trainer

Browser-portable Rust implementation of the **v5.0.0 RGB still-image reference
encoder and decoder**, pinned to Orange-OpenSource/Cool-Chic commit
`a2f9f84f9e1ab31cacc0a882fdba0cd900add4d6`.

The same code runs natively and in browser WASM. No server, Python, pretrained
weights or tensor runtime is needed at runtime. Python/PyTorch are used only for
independent development-time parity checks.

## Encoder

- Official LOP, MOP, HOP and VHOP intra architectures and adaptive latent and
  hyperlatent resolutions; learned separable/symmetric upsampling, nonlinear
  synthesis, residual ARM, IFCE and linear stabilisers.
- Full analytical forward/backward graph, gain-16 soft/noisy latent quantization
  and actual Laplace entropy loss with the reference probability floor.
- Standard intra MSE preset: 5→2→1 warm-up candidates, SOAP network optimizer,
  Adam-like latent group, global gradient clipping, validation/checkpoint
  selection, cosine learning-rate and validation-gated noise/temperature schedules,
  followed by 500 hard-round refinement iterations.
- Complete greedy neural weight/bias RD quantization and exp-Golomb-order search.
- Native and browser encode the actual reference `.cool` headers, quantized neural
  parameters and causal range-coded latent payload. The preview is decoded from
  these bytes, not copied from the training graph or input image.

The browser exposes the actual architecture, rate constraint λ and training
iteration budget. The browser allows 100–100,000 iterations. Below the reference
minimum of 2,000, it warns and proportionally
shortens training stages and validation cadence. At 2,000 and above the reference
schedule is unchanged. The browser defaults to 2,000; the reference CLI to 10,000.
Larger λ favors smaller files; smaller λ favors detail. This codec is lossy.

Training initialization and noise use a deterministic local RNG, not PyTorch's
platform-specific generators. Learned parameters are not promised to be identical
across Rust, PyTorch, CPUs or GPUs. Module outputs/derivatives, optimizer
trajectories, preset schedules and quantization decisions are independently
validated against the pinned source.

## Decoder and scope

The decoder implements integer Q16/Q32 ARM/IFCE, reference probability tables and
constriction queue coding, hyperlatents, deterministic common randomness,
learned synthesis and final nearest/bilinear/bicubic interpolation. Golden
reference-produced files and actual Rust/WASM-trained files are independently
decoded by the official Python decoder.

Scope is **single 8-bit RGB still images and MSE training**. Video/inter prediction,
YUV/high-bit-depth I/O, and perceptual Wasserstein training are not exposed by this
image application. Common-randomness reconstruction is supported by the decoder.
Inputs are composited on white in the app before encoding; the raw Rust API
expects already-composited sRGB and ignores alpha.

Browser guardrails: at most **4,000,000 pixels**, each dimension below 16384,
3 GiB conservative estimated working-buffer budget and 32 MiB input bitstreams.
Large training jobs require confirmation before candidate allocation. These are
browser resource limits, not claims about format limits or measured peak RSS.
Malformed headers, parameter bounds and arithmetic are checked; entropy payloads
are checked by canonical re-encoding. This is not a hardened hostile-file parser.

## Rust API

```rust,ignore
use coolchic::{encode, decode, EncoderSettings};
let bytes = encode(&rgba, width, height, EncoderSettings::default())?;
let image = decode(&bytes)?;
```

For bounded browser progress/cancellation, use `Encoder::new`, `step`, `status`,
`completed`, `total_iterations` and `finish`. Each step performs one real training
iteration or quantization trial. `finish` rejects incomplete work. Browser cancel
terminates the worker, including an in-flight WASM call.

## Validation

```sh
cargo test --manifest-path crates/coolchic/Cargo.toml
cargo clippy --manifest-path crates/coolchic/Cargo.toml --all-targets -- -D warnings
cargo check --manifest-path crates/coolchic/Cargo.toml --features wasm --target wasm32-unknown-unknown
```

Native helpers have no image-I/O dependencies:

```sh
cargo run --release --manifest-path crates/coolchic/Cargo.toml --example reference_train -- input.rgba 32 24 2000 0.001 lop output.cool
cargo run --release --manifest-path crates/coolchic/Cargo.toml --example reference -- decode output.cool output.rgba
```

To reproduce the independent Python checks, install the pinned upstream's
requirements (including CPU PyTorch) in a development-only virtual environment:

```sh
git clone --depth 1 --branch v5.0.0 https://github.com/Orange-OpenSource/Cool-Chic.git /tmp/coolchic-reference-v5
# Run with a Python interpreter containing the upstream requirements:
python scripts/verify-coolchic-operators.py --reference /tmp/coolchic-reference-v5
python scripts/verify-coolchic-graph.py --reference /tmp/coolchic-reference-v5
python scripts/verify-coolchic-training.py --reference /tmp/coolchic-reference-v5
python scripts/verify-coolchic-streams.py --reference /tmp/coolchic-reference-v5
```

Add `--write-fixtures` only when deliberately regenerating reviewed golden data.
Normal Rust tests need no Python or network. See `tests/fixtures/README.md`.

## Provenance and licenses

Reference algorithms and probability-table data: Orange Cool-Chic v5.0.0,
Copyright 2023–2026 Orange, BSD-3-Clause (`UPSTREAM-LICENSE.txt`).
SOAP: Copyright 2024 Nikhil Vyas, MIT (`SOAP-LICENSE.txt`), with upstream Orange
modifications. Portable eigendecomposition/QR uses nalgebra (Apache-2.0).
New port/application code is AGPL-3.0-only (`LICENSE`). Runtime dependency
versions are pinned by `Cargo.lock`; notices are deployed under `public/legal/`.

`src/log_scales.f32le` contains the final 2561 float32 entries from upstream
`coolchic/bitstream/component/mu_scale.npy`, generated as `exp(k/256)` for
`k=-1280..1280` and rounded by upstream PyTorch. Exact stored values prevent
probability-table drift between independent encoders and decoders.
