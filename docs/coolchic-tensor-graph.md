# Full WebGPU tensor graph

`src/lib/coolchic-tensor-graph.ts` is the next GPU path, separate from the existing
per-convolution bridge. Integration with the WASM training controller is in
progress; the existing app path has not yet been switched to this implementation.

The entire reference forward and backward graph executes with GPU-resident
tensors, using TensorFlow.js 4.22's WebGPU kernel runtime and automatic reverse
mode. CPU forwarding is disabled. Matmul/convolution workgroups run in parallel;
dispatches are batched instead of synchronizing after each convolution. The
intended boundary is one gradient readback per iteration, preserving the existing
Rust SOAP/Adam updates, clipping, schedule, RNG, validation/checkpoint selection,
500-step hard refinement and final RD/bitstream writer.

This is not a replacement architecture or approximate training objective:
replicate padding, learned symmetric separable filters, transpose convolution
padding/crop, causal ARM contexts, IFCE and hyperlatent ordering, common randomness,
final interpolation, gain-16 quantization, differentiable Laplace probability
floor, source-level kinks and strict output/log-scale clipping derivatives are
retained. Frozen parameter groups remain frozen in production; the debug oracle
also returns their raw derivatives for independent full-gradient checks.

TensorFlow.js's unbounded AddN kernel can exceed WebGPU's default storage-binding
limit when summing rates/shared gradients over the full latent hierarchy. This
backend reduces in bounded four-operand groups, entirely on GPU. No higher device
limit or CPU handoff is required. Validation/device errors are checked rather
than accepting finite all-zero results from invalid command buffers.

## Independent validation

```sh
node scripts/verify-coolchic-tensor-graph.mjs
```

A real Chromium WebGPU device checks all six full Python-generated fixtures:
small, hyper/common-randomness, final bicubic resizing, LOP, MOP and HOP. It compares
complete decoded tensors, entropy rates, losses, all parameter gradients and all
latent gradients, using the existing full-chain float32 tolerances (3e-5 absolute
/ 1e-3 relative for gradients). Current maximum absolute image error is below
4.2e-7 and gradient error below 1.8e-5. No tolerance has been loosened and no
reference fixture has been regenerated.

Passing those fixtures does not establish faster complete training. Actual
WASM-integrated, equal-work CPU/GPU benchmarks and complete stream/quality checks
remain required before declaring this backend successful.
