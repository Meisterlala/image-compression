# Official-reference golden fixtures

Generated with `scripts/verify-coolchic.py --write-fixtures` using the official
Cool-Chic **v5.0.0** source at `a2f9f84f9e1ab31cacc0a882fdba0cd900add4d6`.

The Python script independently instantiates upstream modules, installs explicit
quantized parameters (including nonzero spatial mean **and scale** ARM weights),
and encodes deterministic synthetic latent grids using the official writer.
`.rgba` contains the corresponding official decoder output, including opaque
alpha. The Rust encoder is **not** used to generate either golden file.

- 3×5 exercises narrow-image raster ordering, tiny and odd-size latent grids.
- 13×17 exercises wavefront ordering and odd-size cropping at multiple scales.
- Tests require neither Python nor model downloads; fixtures total <2 KiB.

The same script also checks the other direction: Rust encodes five sizes at
three qualities, then the reference and Rust decoders must produce identical
RGBA bytes. See the crate README for optional Python environment setup.
