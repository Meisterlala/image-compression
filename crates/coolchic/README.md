# Experimental CPU-only Cool-chic 5.0 profile

Self-contained Rust encoder **and** decoder. No Python, GPU, pretrained weights,
backend, tensor runtime, unsafe code, or architecture-specific intrinsics. Native
and `wasm32-unknown-unknown` use the same code. The only runtime dependencies are
`constriction` 0.4.2 (the reference entropy coder) and `probability`.

This is a **reduced-profile implementation**, not a port of the full reference
encoder. It produces real `.cool` files compatible with the official **v5.0.0**
Python decoder; all per-image network parameters are included in the file.
The Rust decoder accepts this profile only, not arbitrary reference bitstreams.
There is no private envelope, PNG fallback, or model sidecar.

## Implemented profile

- Single RGB 8-bit intra image, at most 4,000,000 pixels, each dimension < 16384.
- Three integer latent grids at full, half and quarter resolution; range -64..63.
- Separable symmetric nearest-neighbor upsampling (4-tap transpose kernels),
  identity residual pre-concatenation filters (1 tap).
- Learned 3-input/3-output linear synthesis and identity output transform.
- Learned six-spatial-context linear Laplace ARM. Fixed-point arithmetic,
  clipping, probability table and wavefront traversal follow the reference.
- Reference headers, neural quantization, signed order-k exponential-Golomb
  coding, and little-endian constriction range stream.
- Deterministic per-image alternating optimization of latents and synthesis,
  followed by fitting/quantizing the entropy model. Quality changes latent gain
  and an L1 rate surrogate; effort changes optimization passes (4–20).

**Not implemented:** reference SOAP/autograd training, soft rounding, IFCE
(inter-feature context extraction), learned upsampling, nonlinear synthesis,
reference LOP/MOP/HOP presets, hyperlatents, common randomness, video, YUV, or
high bit depth. No reference RD/perceptual-quality or speed claims apply.
The restricted synthesis can lose fine color detail even at quality 100; quality
100 is not lossless. More effort is not guaranteed to improve every metric.
This is a small interoperable starting point, not the full 5.0 compression system.

## API and validation

```rust,ignore
let bytes = coolchic::encode(&rgba, width, height, 75, 5)?;
let image = coolchic::decode(&bytes)?;
```

`Encoder::new`, `step`, and `finish` expose individual optimization passes for
browser progress. `finish` consumes the encoder and rejects incomplete training.
Input sizes, settings, header/payload lengths, profile parameters, and neural
parameter bounds are validated before large decoder allocations. Canonical
range re-encoding rejects truncated/extended entropy payloads (included in decode
timing). This is not an authenticated format or a hardened hostile-file parser.

```sh
cargo test --manifest-path crates/coolchic/Cargo.toml
cargo clippy --manifest-path crates/coolchic/Cargo.toml --all-targets -- -D warnings
cargo build --manifest-path crates/coolchic/Cargo.toml --target wasm32-unknown-unknown --release
```

`examples/rgba.rs` provides dependency-free native file I/O for raw RGBA buffers.

## Provenance

Format and decoder algorithms ported/adapted from Orange-OpenSource/Cool-Chic
**v5.0.0**, commit `a2f9f84f9e1ab31cacc0a882fdba0cd900add4d6`:
https://github.com/Orange-OpenSource/Cool-Chic/tree/a2f9f84f9e1ab31cacc0a882fdba0cd900add4d6

Copyright 2023–2026 Orange. Upstream BSD-3-Clause notice is preserved in
`UPSTREAM-LICENSE.txt` and in the deployed application's legal directory. New
application/port code is AGPL-3.0-only, consistent with the enclosing project.

`src/log_scales.f32le` is the final 2561 float32 values (little endian) from the
upstream `coolchic/bitstream/component/mu_scale.npy`: `exp(k/256)` for k=-1280..1280,
**as rounded by upstream PyTorch**. Keeping the exact table avoids entropy
probability mismatches from different implementations of exp. Means are exact
multiples of 1/256 and are computed directly. The upstream `_generate_possible_mu_scale.py`
documents construction of the table. No trained model weights are bundled.
