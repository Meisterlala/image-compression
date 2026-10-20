# Official v5 reference fixtures

Source: unmodified Orange Cool-Chic **v5.0.0**, commit
`a2f9f84f9e1ab31cacc0a882fdba0cd900add4d6`. The generators assert that revision
and reject modified tracked source files. Synthetic test pixels only; no dataset.

- `reference-operators.json`: actual PyTorch forward/backward results for
  convolutions, synthesis, learned filters, quantizers, causal contexts, ARM/IFCE,
  complete latent pyramids and 25-step SOAP trajectories (including zero/tiny
  initial gradients).
- `reference-graph.json`: actual `CoolChicEncoder.forward` and reference MSE loss,
  including all network/latent gradients, hyperlatents, common randomness, final
  interpolation and official LOP/MOP/HOP architectures.
- `reference-training.json`: standard intra schedules, the reference neural-rate
  estimator and final greedy quantization choices/parameters.
- `full-reference/*.cool` and `.rgba`: seven files written and decoded by the
  official Python codec, covering LOP/MOP/HOP/VHOP, common randomness,
  hyperlatents, reduced-resolution synthesis and a linear/no-IFCE configuration.
  Rust must reproduce exact RGBA and byte-identical header/network/range encoding.

Generators live in `scripts/verify-coolchic-{operators,graph,training,streams}.py`.
`--write-fixtures` is an explicit regeneration operation; default mode verifies
committed data. The stream checker also trains actual Rust encoders for 2,000
iterations and checks their output with the official decoder. Browser tests
check official files directly in WASM and decode browser-trained downloads in an
independent native process.

Normal Cargo tests use only committed data, without Python or model downloads.
Floating graph/gradient checks have explicit tolerances for CPU/libm rounding;
integer entropy streams and the golden file reconstructions are checked exactly.
