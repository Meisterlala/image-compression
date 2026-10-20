# Cool-chic acceleration

## Runtime

The browser trains in Rust/WASM, with an optional WebGPU tensor graph enabled by
default for images of at least 16,384 pixels. Preview decoding always uses the
portable CPU decoder on the emitted `.cool` bytes.

- **CPU:** SIMD is always enabled. Larger images use a separate shared-memory
  WASM module with up to 16 Rayon workers, leaving one hardware thread free when
  more than two are available. Small images skip pool/GPU startup.
- **GPU:** forward/backward graphs, clipping, SOAP/Adam updates and hard-round
  rate-distortion evaluations run on-device. Training queues up to eight complete
  iterations with weights and optimizer moments resident on the GPU. Only small
  SOAP eigendecomposition/QR problems visit WASM (initially and every ten updates).
  Rust retains schedules, RNG, checkpoints, greedy decisions and entropy coding.
  The GPU toggle disables this path without disabling CPU threading.

`src/lib/coolchic.ts` selects the backend. `coolchic-graph-acceleration.ts` bridges
synchronous WASM to `coolchic-tensor.worker.ts` through shared buffers and bounded
`Atomics.wait` calls: one synchronization per training batch or RD evaluation.
Batches stop at validation/schedule boundaries and shrink to fit the 256 MiB
transfer cap. Initialization allows 30 seconds; compute allows 60 seconds.
Complete optimizer state returns at batch boundaries; failed batches leave CPU
weights, moments and RNG untouched, allowing safe CPU continuation.
Progress and saved result notices report actual backend use.

## GPU implementation

`coolchic-tensor-graph.ts` uses TensorFlow.js 4.22 WebGPU kernels and reverse mode;
CPU forwarding is disabled. `coolchic-tensor-fused.ts` supplies fused operators
and exact adjoints. Padding, causal contexts, hyperlatents, common randomness,
quantization, probability floors and clipping derivatives follow the reference.
Frozen parameter groups stay frozen; fixture checks can request raw derivatives.

`coolchic-tensor-replay.ts` records differentiated command plans. Dead storage
intermediates are recycled within each capture, then reserved for that plan;
uniforms and host-written constants remain immutable. CPU tensors upload through
`queue.writeBuffer`, without retaining mapped upload buffers in the replay. Later iterations update inputs and replay pipelines/bind
groups without rebuilding the autodiff graph. Optimizer commands are cached too;
noise and learning-rate data upload once per batch. Replays preserve the backend's
bounded command-buffer submissions instead of merging whole training batches into
one oversized command buffer. These submissions are asynchronous; weights and
moments remain GPU-resident between them. Switching between training and evaluation releases the previous command plan
and its optimizer bind groups, rather than retaining both GPU working sets. Target/common tensors stay cached
until the graph descriptor changes. Gradient sums use bounded four-input groups
to respect WebGPU storage-binding limits. Device validation errors are checked.

GPU rounding can change trained weights and output bytes. This is not a promise
of identical training across devices or faster encoding on every machine.

## Build and browser requirements

`npm run build:wasm` builds portable SIMD and threaded SIMD/atomics modules. The
threaded build uses `nightly-2026-02-28` and `-Z build-std=panic_abort,std`; see
[setup](../README.md#run) and `scripts/build-wasm.mjs` for toolchain/flags.

Shared-memory paths need HTTPS or localhost, SharedArrayBuffer and cross-origin
isolation (`COOP: same-origin`, `COEP: require-corp`). Vite and nginx provide these
headers. Workers/WASM stay same-origin under the supplied CSP. Missing isolation
or thread support selects portable single-thread SIMD WASM; missing WebGPU alone
still allows CPU threading. Cancel terminates the worker and its descendants.

## Validate

After rebuilding WASM:

```sh
npm test
node scripts/verify-coolchic-tensor-graph.mjs
node scripts/verify-coolchic-gpu.mjs --require-gpu
# Optional warmed full-gradient timings (not complete encode times):
node scripts/verify-coolchic-gpu.mjs --require-gpu --benchmark
```

The tensor verifier checks full images, rates, losses and all gradients against
six pinned Python fixtures (3e-5 absolute / 1e-3 relative gradient tolerance).
The runtime verifier checks threaded derivative parity, actual CPU/GPU training,
production backend selection, native decoding of emitted streams, and missing
WebGPU/isolation fallbacks. Without `--require-gpu`, absent hardware is a skip.
Unit tests cover graph buffer reuse, validation, timeouts, failure and disposal.
