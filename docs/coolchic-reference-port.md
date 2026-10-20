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

The fixtures cover single-pixel and odd-sized tensors, learned nonzero filters, gradients of inputs and every network/filter parameter, and multiple soft-round temperatures. Numerical parity uses absolute/relative float32 tolerances, not bit-exact claims.

Upstream v5's `ste` branch assigns its result inside `torch.no_grad`, detaching its output despite the documented surrogate derivative. Tests preserve actual source behavior. This is separate from the `softround`/`hardround` modes used by the intra preset.

## Remaining before replacing the app codec

- Integrate the ported neural modules into a complete training graph, including latent gains, distortion loss, hyperlatent/common-randomness handling and final output interpolation.
- Gradient clipping, warm-up candidate selection, schedules, validation/checkpoint selection and quantization search around the ported optimizer.
- Reference architecture/configuration choices and parameter initialization; wire rate constraint and iteration controls to actual reference settings, not the old quality/pass shortcuts.
- Generalized bitstream headers/neural parameter parsing, integer ARM/IFCE and entropy coding for reference architectures; complete image reconstruction.
- Independent reference-produced bitstreams and end-to-end forward/backward interoperability tests, including LOP/MOP/HOP configurations.
- Browser WASM integration with cancellation/progress, memory limits and honest encode/decode benchmarks. No Python or server dependency at runtime.

## Reproduce operator parity

Python/PyTorch are needed only to regenerate/check independent test fixtures:

```sh
git clone --depth 1 --branch v5.0.0 https://github.com/Orange-OpenSource/Cool-Chic.git /tmp/coolchic-reference-v5
.cache/coolchic-venv/bin/python scripts/verify-coolchic-operators.py --reference /tmp/coolchic-reference-v5
# Only after inspecting changes: add --write-fixtures to regenerate.
cargo test --manifest-path crates/coolchic/Cargo.toml
cargo clippy --manifest-path crates/coolchic/Cargo.toml --all-targets -- -D warnings
```
