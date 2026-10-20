# Cool-chic v5 reference port — in progress

Target: full **RGB still-image** reference encoding and decoding, locally in the browser. Upstream: Orange-OpenSource/Cool-Chic v5.0.0, commit `a2f9f84f9e1ab31cacc0a882fdba0cd900add4d6`. Upstream BSD-3-Clause notice is in `crates/coolchic/UPSTREAM-LICENSE.txt`. SOAP's MIT notice is in `crates/coolchic/SOAP-LICENSE.txt`; the portable Rust eigendecomposition/QR backend is nalgebra.

**Not complete. The app still runs the old reduced encoder.** New reference operators live separately in `crates/coolchic/src/reference/`; nothing silently substitutes the reduced implementation for full reference behavior.

## Verified foundation

Forward and analytical reverse-mode implementations, checked against fixtures produced by the unmodified pinned PyTorch source:

- Replicate-padded OIHW synthesis convolutions, residual layers and ReLU.
- Nonlinear synthesis trunk, optional linear stabiliser (including omission of common-randomness channels), and output transform.
- Learned symmetric/separable transposed-convolution upsampling with the reference border padding/crop; residual pre-concatenation filters.
- Soft-round quantization, double soft-round with supplied Gaussian noise, hard rounding, and Kumaraswamy noise transform.
- Full learned latent pyramid, including odd-size cropping, shared/per-level kernels and gradient accumulation.
- Residual ARM MLP, optional stabiliser, causal spatial contexts and zero-padded gradient scatter.
- Cross-resolution IFCE graph: nearest-neighbour context pyramid, pointwise transformation before final interpolation, odd-size crops, disabled levels, dummy zero context on the first-decoded feature, and gradients into all already-decoded features.
- Actual Laplace rate (not an L1 proxy), probability floor, scale clamp and analytical gradients into latents and ARM outputs.
- SOAP in the reference trainer's configuration: per-axis Shampoo matrices/eigenbases, projected moments, periodic QR refresh, second-moment reorder, bias correction and post-update weight decay. Latent group uses the reference's unpreconditioned Adam-like update, including its skipped first step. Tests replay 25-step PyTorch trajectories across matrices, convolutions, biases, capped dimensions and latent grids, with a mid-run learning-rate change.

The fixtures cover single-pixel and odd-sized tensors, learned nonzero filters, gradients of inputs and every network/filter parameter, and multiple soft-round temperatures. Numerical parity uses absolute/relative float32 tolerances, not bit-exact claims. Full-chain entropy gradients have a separate 3e-5 absolute / 1e-3 relative tolerance: CDF differences near the 2^-16 probability floor amplify small BLAS/libm rounding differences. Individual operator checks remain tighter.

Upstream v5's `ste` branch assigns its result inside `torch.no_grad`, detaching its output despite the documented surrogate derivative. Tests preserve actual source behavior. This is separate from the `softround`/`hardround` modes used by the intra preset.

- Complete RGB MSE training graph: gain-16 latent quantization, rate derivatives through spatial/IFCE contexts, hyperlatent filtering, learned synthesis, image clamp and distortion loss, deterministic common-noise reconstruction, and final nearest/bilinear/bicubic interpolation. Full forward/loss/gradient fixtures use the actual upstream CoolChicEncoder and loss function, including LOP/MOP/HOP architectures, hyperlatents, common randomness and a reduced-resolution synthesis output.
- Standard LOP/MOP/HOP/VHOP architecture descriptors and adaptive latent/hyperlatent resolutions, and neural parameter group ordering shared by serialization and optimization.

- General reference v5 RGB bitstream headers and neural parameter serialization, Q16/Q32 integer ARM/IFCE, full causal range coding, and reconstruction. Seven official golden streams cover LOP/MOP/HOP/VHOP, common randomness, hyperlatents, reduced-resolution output and a linear/no-IFCE architecture. Native Rust re-encoding is byte-exact; both directions decode to exact official RGBA in the independent Python check. Old reduced-profile files remain decodable.

- Complete standard intra MSE trainer: 5→2→1 warm-up candidates, reference iteration budgets (minimum 2000), validation-gated temperature/noise updates, cosine learning rates, global network-gradient clipping, best-checkpoint restoration, 500 hard-round refinement iterations and greedy ARM→IFCE→synthesis→upsampling RD quantization. Quantization choices and exp-Golomb orders match the unmodified reference in independent fixtures.
- Actual Rust-trained LOP/MOP/HOP files independently decode to identical pixels in the official decoder, including a 32×24 gradient at 2000 iterations (390 bytes, 34.57 dB in the tested native build). This is a sanity sample, not a general benchmark or reference-quality guarantee.

Training initialization/noise use a reproducible local RNG, not PyTorch's platform-specific CPU/GPU generator. Identical learned weights across implementations/devices are not promised. Bitstream-defined common randomness matches exactly. SOAP handles tiny/singular Gram matrices with positive scaling and a float64 decomposition fallback, retaining reference trajectory checks including zero/tiny first gradients.

## Remaining before replacing the app codec

- Browser WASM integration, actual rate/iteration/architecture controls, persisted-setting defaults/migration, cancellation/progress, memory guardrails and honest encode/decode benchmarks. No Python or server dependency at runtime.
- Replace/remove the old reduced encoder as the default path; retain backward decoding of its saved .cool outputs.
- Update licenses, source packaging and user/developer documentation; run browser regression and reload/download integrity tests.

## Reproduce operator parity

Python/PyTorch are needed only to regenerate/check independent test fixtures:

```sh
git clone --depth 1 --branch v5.0.0 https://github.com/Orange-OpenSource/Cool-Chic.git /tmp/coolchic-reference-v5
.cache/coolchic-venv/bin/python scripts/verify-coolchic-operators.py --reference /tmp/coolchic-reference-v5
.cache/coolchic-venv/bin/python scripts/verify-coolchic-graph.py --reference /tmp/coolchic-reference-v5
.cache/coolchic-venv/bin/python scripts/verify-coolchic-streams.py --reference /tmp/coolchic-reference-v5
.cache/coolchic-venv/bin/python scripts/verify-coolchic-training.py --reference /tmp/coolchic-reference-v5
# Only after inspecting changes: add --write-fixtures to regenerate.
cargo test --manifest-path crates/coolchic/Cargo.toml
cargo clippy --manifest-path crates/coolchic/Cargo.toml --all-targets -- -D warnings
```
