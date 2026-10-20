# Cool-chic v5 reference port — in progress

Target: full **RGB still-image** reference encoding and decoding, locally in the browser. Upstream: Orange-OpenSource/Cool-Chic v5.0.0, commit `a2f9f84f9e1ab31cacc0a882fdba0cd900add4d6`. Upstream BSD-3-Clause notice is in `crates/coolchic/UPSTREAM-LICENSE.txt`.

**Not complete. The app still runs the old reduced encoder.** New reference operators live separately in `crates/coolchic/src/reference/`; nothing silently substitutes the reduced implementation for full reference behavior.

## Verified foundation

Forward and analytical reverse-mode implementations, checked against fixtures produced by the unmodified pinned PyTorch source:

- Replicate-padded OIHW synthesis convolutions, residual layers and ReLU.
- Nonlinear synthesis trunk, optional linear stabiliser (including omission of common-randomness channels), and output transform.
- Learned symmetric/separable transposed-convolution upsampling with the reference border padding/crop; residual pre-concatenation filters.
- Soft-round quantization, double soft-round with supplied Gaussian noise, hard rounding, and Kumaraswamy noise transform.

The fixtures cover single-pixel and odd-sized tensors, learned nonzero filters, gradients of inputs and every network/filter parameter, and multiple soft-round temperatures. Numerical parity uses absolute/relative float32 tolerances, not bit-exact claims.

Upstream v5's `ste` branch assigns its result inside `torch.no_grad`, detaching its output despite the documented surrogate derivative. Tests preserve actual source behavior. This is separate from the `softround`/`hardround` modes used by the intra preset.

## Remaining before replacing the app codec

- Assemble full latent hierarchy, including odd-size cropping and backward propagation.
- Trainable residual ARM, IFCE, causal context extraction and Laplace rate derivatives.
- Reference SOAP network optimizer and Adam-like latent updates, clipping, warm-up candidate selection, schedules, validation/checkpoint selection and quantization search.
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
