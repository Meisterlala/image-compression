# Cool-chic browser acceleration

## What runs where

- The per-image trainer, full gradients, SOAP, preset schedule, RD quantization
  and actual `.cool` bitstream remain Rust/WASM. No server or pretrained model.
- **CPU multithreading** is automatic for images >=16,384 pixels. SIMD is always
  enabled; neither CPU optimization has an off switch. A separate
  SIMD + atomics WASM module uses up to 16 Rayon workers (leaving one
  advertised hardware thread free on systems with more than two). Large
  convolution forward/backward and learned filter passes use disjoint output
  tiles/lines. Large soft quantization and Box-Muller noise transforms also run
  in parallel, preserving the exact RNG stream and odd-count spare samples. Small jobs
  avoid pool startup and use the portable SIMD module.
- **GPU training** is independently optional and on by default. WebGPU computes
  convolution/ARM forward and analytical input/weight/bias derivatives. Padding,
  residual connections and ReLU gradients match the reference operators. Fixed
  partial reductions avoid floating-point atomic races. Shared WASM uses
  read-only tensor views during the synchronous bridge call, and GPU readback
  writes directly into the bridge response, eliminating redundant host copies.
- Rust training is synchronous; a separate GPU worker performs asynchronous
  dispatch/readback while the training worker sleeps in `Atomics.wait`. This is
  not busy-spinning, nor a shared GPU tensor graph. Transfers and synchronization
  can dominate. SOAP, other graph operations and entropy coding remain on CPU.
- The preview always decodes the actual output bytes with the **portable CPU
  decoder**. GPU rounding may change trained weights/bytes, just as training on
  different CPUs/GPUs can; the reference algorithm/format is unchanged.
- Training retains ARM, synthesis and pyramid forward activations for reverse
  mode instead of rebuilding them (and recomputing ReLUs) in backward passes.
  Filter geometry is planned once per axis; spatial convolutions separate
  replicate borders from contiguous SIMD interiors. Soft quantization shares
  forward/derivative tanh evaluations and hoists the temperature normalizer.
  The conservative memory estimate includes additional retained-buffer headroom.

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
the automatic pool and GPU + automatic threads. Unit tests inject timeout/device failure and check crossover
selection/disposal. Native parallel tests compare large kernels and learned-filter
derivatives with a genuine one-thread pool, not just pool startup. Regression
checks also cover malformed activation tapes, noise-length overflow without RNG
advancement, safe-integer GPU sizing, and bounded explicit pool sizes. The test
worker transfers typed buffers; only the Playwright-to-native JSON boundary
materializes arrays for independent byte/pixel checks.

The compute callback is synchronous even though its factory is asynchronous.
Its response is borrowed until the next invocation: Rust calls `copy_to` on the
same stack before another convolution can run. No Promise or `await` separates
callback return from that copy. Tests cover this consumption contract and reuse.

### Local measurements (not codec-wide benchmarks)

Chromium headless, Ryzen 7 7700X, RTX 2080 Ti. LOP, synthetic square RGB input,
100-iteration preset; **candidate initialization plus first 10 real training
steps**, not a complete encode. Module loading and worker/pool initialization are
outside these timing samples; GPU calibration is inside. Single samples:

| Image | Portable SIMD | 4 CPU threads | 8 CPU threads | Automatic (15 threads) | GPU + automatic |
|---|---:|---:|---:|---:|---:|
| 128² | 207 ms | 168 ms | 137 ms | 140 ms | 141 ms |
| 384² | 1,691 ms | 937 ms | 777 ms | 733 ms | 871 ms |
| 512² | 3,033 ms | 1,631 ms | 1,315 ms | 1,247 ms | 1,481 ms |

On this machine, threaded CPU won the GPU crossover checks, so low GPU utilization
is expected in the last column. Initial calibration costs time, amortized over
long runs. Adaptive GPU with **one** CPU thread measured 3,111 ms at 512² versus
3,033 ms portable: faster spatial backward kernels did not repay calibration
within this short sample. A 512² backward GPU job including bridge took about
12 ms versus 49 ms portable CPU, but the threaded CPU won that comparison.

A captured pre-planar-optimization browser artifact took 1,033 / 9,188 / 16,465 ms
for the same 128² / 384² / 512² initialization + 10-step smoke benchmark. Thus the
combined SIMD/caching + automatic-thread path was roughly 7.4–13.2× faster in
these samples. This is not a claim of full-encode speedup or stable performance
on other hardware. Re-run the verifier on the target browser/device.

A separate **complete 2,000-iteration** 96×64 synthetic RGB encode measured
60,631 ms on the captured original artifact and 11,160 ms on the optimized
portable WASM: about 5.4× faster, with the same 512-byte file size and
44.668127 dB decoded PSNR. This small job does not need a CPU pool. These are
local sanity measurements, not a compression benchmark or a guarantee for
larger images.

## Remaining GPU bottleneck

This is still a hybrid trainer, **not a GPU-resident training graph**. Every
GPU convolution crosses a host synchronization boundary, and the CPU owns the
rest of the graph and optimizer state. Parallel shader invocations alone cannot
remove that boundary. Batching network stages and keeping activations/gradients
on-device would be the next architectural step toward sustained GPU utilization;
it is not implemented by the per-operator bridge described here. Neither 100%
GPU utilization nor instantaneous 2,000-iteration encoding is promised.
