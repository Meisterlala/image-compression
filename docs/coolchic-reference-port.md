# Cool-chic reference implementation

The browser uses the Rust/WASM **v5.0 RGB intra MSE encoder and decoder**, pinned
to Orange-OpenSource/Cool-Chic commit
`a2f9f84f9e1ab31cacc0a882fdba0cd900add4d6`. No server, Python or downloaded
weights are needed at runtime. See `crates/coolchic/README.md` for scope/API.

## Algorithm

- Official LOP/MOP/HOP/VHOP architectures, adaptive latent/hyperlatent grids,
  learned separable/symmetric upsampling, nonlinear synthesis and stabilisers.
- Gain-16 quantization, causal residual ARM and cross-resolution IFCE, actual
  Laplace rate and analytical gradients through the complete graph.
- Reference intra MSE schedule: 5→2→1 warm-up, SOAP network and Adam-like latent
  groups, gradient clipping, cosine rates, validation-gated temperatures/noise,
  checkpoint selection and 500 hard-round refinement iterations.
- Greedy neural RD quantization and reference exp-Golomb-order search.
- Full RGB bitstream headers/neural groups, Q16/Q32 integer ARM/IFCE, constriction
  causal range coding, hyperlatents, common-noise reconstruction and final image
  interpolation. Preview pixels come from decoding actual output bytes.

Training uses reproducible local randomness, not PyTorch's CPU/GPU RNG streams.
Identical learned parameters across implementations/devices are not promised.
Bitstream-defined common randomness and integer probability tables match exactly.
SOAP's small-matrix factorizations use scaling and a float64 fallback to avoid
underflow with zero/tiny-gradient covariances while retaining trajectory checks.

## Browser controls and lifecycle

Four adjacent preset choices select the actual reference architectures. The rate
slider shows actual λ, increasing left-to-right from 1e-6 to 0.1; higher values
prioritize smaller files. Training budgets range from 100 to 100,000 iterations.
Below 2,000, a visible warning explains the reference minimum;
training stages and validation cadence are proportionally shortened. The reference
schedule remains unchanged at 2,000 and above. The dedicated worker posts progress
without artificial timer pauses and throttles UI updates to avoid excess rendering.
No quality percentage is substituted for the algorithm inputs.
Tooltips describe practical trade-offs; detailed scheduling stays in source/docs.

Per-image training is measured as encoding time. WASM startup, actual byte-stream
decoding and metrics have separate timings. Worker termination cancels jobs.
Source images, settings, bytes and decoded output remain in browser IndexedDB
across reloads. No job is resumed automatically. Downloads use the settings of
that output, not any controls changed afterwards.

The app rejects more than 4 million pixels or dimensions >=16384. A conservative
buffer estimate rejects training above 3 GiB and asks for confirmation above
1.5 GiB before allocating candidates. These are browser guardrails, not reference
format constraints or measured process RSS. It never silently downscales.

## Acceleration

The browser defaults to adaptive WebGPU and CPU multithreading. Shared-memory
features require cross-origin isolation, with portable single-thread SIMD WASM
as the fallback. Only training floating-point kernels are accelerated; output
still uses the reference `.cool` format and portable CPU decoding. Backends and
limits, independent derivative checks and reproducible measurements are described
in [coolchic-acceleration.md](coolchic-acceleration.md).

## Independent validation

The committed fixtures come from unmodified pinned Python modules and check
individual operators/derivatives, complete forward/loss/gradient graphs,
optimizer trajectories, exact preset schedules and final quantization choices.
Full-chain entropy gradients allow 3e-5 absolute / 1e-3 relative float32 error:
CDF differences near the probability floor amplify BLAS/libm rounding. Individual
operators have tighter tolerances.

Seven official streams cover LOP/MOP/HOP/VHOP, hyperlatents, common randomness,
final resizing and a linear/no-IFCE architecture. Native re-encoding is byte-exact,
and reconstructions match exact official RGBA in both directions. Actual Rust
training runs also produce files independently decoded by the reference. Browser
tests check golden files in WASM, train/download/decode, reload integrity,
segmented controls, sliders/tooltips, cancellation, pixel/memory errors and mobile
layout. Native samples are sanity tests, not published codec benchmarks.

```sh
git clone --depth 1 --branch v5.0.0 https://github.com/Orange-OpenSource/Cool-Chic.git /tmp/coolchic-reference-v5
# Use a development-only Python environment with upstream requirements installed:
python scripts/verify-coolchic-operators.py --reference /tmp/coolchic-reference-v5
python scripts/verify-coolchic-graph.py --reference /tmp/coolchic-reference-v5
python scripts/verify-coolchic-training.py --reference /tmp/coolchic-reference-v5
python scripts/verify-coolchic-streams.py --reference /tmp/coolchic-reference-v5
cargo test --manifest-path crates/coolchic/Cargo.toml
cargo clippy --manifest-path crates/coolchic/Cargo.toml --all-targets -- -D warnings
npm run build:web
npx playwright test tests/coolchic.spec.ts
```

Scope is RGB 8-bit still-image MSE training; not video/YUV/high-bit-depth I/O or
perceptual Wasserstein training. Copyright/license notices: Orange BSD-3-Clause,
SOAP MIT, nalgebra Apache-2.0, application/port AGPL-3.0-only, deployed in `/legal/`.
