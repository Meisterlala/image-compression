# Cool-chic browser acceleration

## What runs where

- The per-image trainer, full gradients, SOAP, preset schedule, RD quantization
  and actual `.cool` bitstream remain Rust/WASM. No server or pretrained model.
- **CPU multithreading** is automatic for images >=16,384 pixels. SIMD is always
  enabled; neither CPU optimization has an off switch. A separate
  SIMD + atomics WASM module uses up to 8 Rayon workers (leaving at least one
  advertised hardware thread free). Large convolution forward/backward and
  learned upsampling/filter passes use disjoint output tiles/lines. Small jobs
  avoid pool startup and use the portable SIMD module.
- **GPU training** is independently optional and on by default. WebGPU computes
  convolution/ARM forward and analytical input/weight/bias derivatives. Padding,
  residual connections and ReLU gradients match the reference operators. Fixed
  partial reductions avoid floating-point atomic races.
- Rust training is synchronous; a separate GPU worker performs asynchronous
  dispatch/readback while the training worker sleeps in `Atomics.wait`. This is
  not busy-spinning, nor a shared GPU tensor graph. Transfers and synchronization
  can dominate. SOAP, other graph operations and entropy coding remain on CPU.
- The preview always decodes the actual output bytes with the **portable CPU
  decoder**. GPU rounding may change trained weights/bytes, just as training on
  different CPUs/GPUs can; the reference algorithm/format is unchanged.

Low GPU utilization is not itself an error. Qualifying kernels (>=8 million MACs)
are warmed up and timed against **actual local CPU WASM**, including the bridge's
packing/readback. GPU must take less than half the measured CPU time and pass a
numerical sanity check. Decisions are cached by shape/direction in Rust: rejected
operations do not keep copying data or retrying. An 8-core CPU can beat this GPU
bridge, especially on pointwise layers. Progress reports the CPU pool and GPU use;
completed results show thread count, GPU job count, fallback, or CPU-faster notices.
Turning the GPU toggle off skips GPU initialization/calibration entirely.

## Compatibility and failure behavior

The threaded module and GPU bridge need a secure context and `crossOriginIsolated`
(`Cross-Origin-Opener-Policy: same-origin`, `Cross-Origin-Embedder-Policy:
require-corp`). Vite dev/preview and `deploy/nginx.conf` already supply these.
Workers and WASM must remain same-origin under the supplied CSP.

Without isolation, SharedArrayBuffer, WebGPU or multiple hardware threads, the
portable single-thread SIMD module still works. GPU initialization is bounded to
15 seconds and compute waits to 30 seconds. Device/worker errors, over-budget
buffers, non-finite output and timeout fall back to CPU. GPU transfer and device
working buffers have separate 256 MiB caps. CPU image/memory guardrails still
apply. Cancel terminates the training worker and its descendants, including
Rayon/GPU workers. Results retain their backend notices across reloads.

## Build

Stable Rust builds the portable codec; only the threaded artifact uses a pinned
nightly and rebuilt thread-safe stdlib:

```sh
rustup toolchain install nightly-2026-02-28 --profile minimal \
  --component rust-src --target wasm32-unknown-unknown
npm run build:wasm
npm run build:web
```

`scripts/build-wasm.mjs` builds `src/wasm-coolchic/` with SIMD and
`src/wasm-coolchic-threads/` with SIMD, atomics, bulk memory, imported shared memory,
TLS exports, a 4 GiB address-space ceiling, and `-Z build-std=panic_abort,std`.
Generated assets are ignored and excluded from the source archive; sources and
lockfiles are included. GPU WGSL and worker code are ordinary Vite build inputs.

## Validation

```sh
cargo test --manifest-path crates/coolchic/Cargo.toml
cargo test --manifest-path crates/coolchic/Cargo.toml --features parallel
cargo clippy --manifest-path crates/coolchic/Cargo.toml --all-targets --features parallel -- -D warnings
npm run check
npx vitest run
node scripts/verify-coolchic-gpu.mjs --require-gpu --benchmark
npx playwright test tests/coolchic.spec.ts
```

The standalone verifier starts its own Vite dev server and real Chromium. It checks
explicit opt-out, an actual four-thread WASM pool executing nontrivial derivatives,
real parallel training, independent PyTorch convolution golden outputs/gradients,
additional degenerate/border/multi-chunk GPU cases, the worker bridge, GPU-trained
streams decoded by a separate native process, missing WebGPU, and missing
isolation. `--require-gpu` makes unavailable GPU hardware a failure rather than a
skip. `--benchmark` also compares portable WASM, adaptive GPU, 4/8 CPU threads and
GPU + 8 threads. Unit tests inject timeout/device failure and check crossover
selection/disposal. Native parallel tests compare large kernels and learned-filter
derivatives with a genuine one-thread pool, not just pool startup.

### Local measurements (not codec-wide benchmarks)

Chromium headless, Ryzen 7 7700X, RTX 2080 Ti. LOP, synthetic square RGB input,
100-iteration preset; **candidate initialization plus first 10 real training
steps**, not a complete encode. Module loading and worker/pool initialization are
outside these timing samples; GPU calibration is inside. Single samples:

| Image | Portable SIMD | 4 CPU threads | 8 CPU threads | Adaptive GPU + 8 threads |
|---|---:|---:|---:|---:|
| 128² | 369 ms | 322 ms | 284 ms | 279 ms |
| 384² | 3,186 ms | 1,783 ms | 1,423 ms | 1,591 ms |
| 512² | 5,832 ms | 3,164 ms | 2,473 ms | 2,770 ms |

On this machine, threaded CPU won the GPU crossover checks, so low GPU utilization
is expected in the last column. Initial calibration costs time, amortized over
long runs. Adaptive GPU with **one** CPU thread measured 5,468 ms at 512² versus
5,832 ms portable. Individual 512² spatial forward/backward GPU kernels including
bridge took about 11/16 ms versus 29/51 ms portable CPU; the rest of training is
still CPU work.

A captured pre-planar-optimization browser artifact took 1,033 / 9,188 / 16,465 ms
for the same 128² / 384² / 512² initialization + 10-step smoke benchmark. Thus the
combined planar/SIMD + eight-thread path was roughly 3.6–6.7× faster in these
samples. This is not a claim of full-encode speedup, better compression, or stable
performance on other hardware. Re-run the verifier on the target browser/device.
