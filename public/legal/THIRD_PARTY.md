# Third-party components

This application is AGPL-3.0-only. The following licenses remain applicable to
third-party code and data. Open-source terms are not patent clearance.

## Runtime

- Cool-chic v5 RGB reference encoder/decoder port: neural, training, bitstream
  algorithms and probability-table data adapted from Orange-OpenSource/Cool-Chic,
  commit a2f9f84f9e1ab31cacc0a882fdba0cd900add4d6. Copyright 2023–2026 Orange,
  BSD-3-Clause (coolchic-LICENSE.txt). New port/application code is AGPL-3.0-only.
  Source: https://github.com/Orange-OpenSource/Cool-Chic/tree/a2f9f84f9e1ab31cacc0a882fdba0cd900add4d6
- SOAP optimizer: Copyright 2024 Nikhil Vyas, MIT (soap-LICENSE.txt), with upstream
  Orange modifications. Source: https://github.com/nikhilvyas/SOAP
- nalgebra 0.34.2: Apache-2.0 (nalgebra-LICENSE.txt), portable matrix factorization.
  Source: https://github.com/dimforge/nalgebra
- constriction 0.4.2: MIT license selected (constriction-LICENSE.txt).
  Source: https://github.com/bamler-lab/constriction
- probability 0.20.3: MIT (probability-LICENSE.txt).
  Source: https://github.com/stainless-steel/probability
- zenjpegai, pinned to 1192fca6b6c2fdc4dcf29a6c8064c53a987caf59:
  AGPL-3.0-only or commercial. This build uses AGPL-3.0-only.
  Source: https://github.com/imazen/zenjpegai/tree/1192fca6b6c2fdc4dcf29a6c8064c53a987caf59
  Full AGPL text: AGPL-3.0.txt.
- JPEG AI reference software and weights, pinned to
  b9e573fef1dd90408270da23aae678c823f15525: BSD-style ITU/ISO/IEC notice.
  Full text: jpeg-ai-reference-LICENSE.txt, also embedded in each model bundle.
  Source: https://gitlab.com/wg1/jpeg-ai/jpeg-ai-reference-software
  The notice explicitly reserves third-party and patent rights.
- @jsquash/jpeg 1.6.0, @jsquash/avif 2.1.1, @jsquash/jxl 1.3.0:
  wrappers Copyright Google Inc. and contributors, Apache-2.0 (jsquash-LICENSE.txt).
  Source: https://github.com/jamsinclair/jSquash
  JPEG codec notices: mozjpeg-LICENSE.md.
  Codec-specific dependencies retain their original licenses:
  - MozJPEG: https://github.com/mozilla/mozjpeg
  - libavif: https://github.com/AOMediaCodec/libavif/blob/main/LICENSE
  - libaom: https://aomedia.googlesource.com/aom/+/refs/heads/main/LICENSE
  - dav1d: https://code.videolan.org/videolan/dav1d/-/blob/master/COPYING
  - libjxl: https://github.com/libjxl/libjxl/blob/main/LICENSE
- Vue: Copyright (c) 2018-present Yuxi (Evan) You and Vue contributors, MIT.
  Full text: vue-LICENSE.txt. Source: https://github.com/vuejs/core

## Rust transitive dependencies

Exact versions and checksums are in crates/codec/Cargo.lock and
crates/coolchic/Cargo.lock in the source archive.
Their complete sources and license files are available through Cargo (`cargo vendor`).
These include archmage, magetypes, safe_unaligned_simd, libm, enough, whereat,
wasm-bindgen, js-sys, serde and serde-wasm-bindgen and their transitive dependencies.
Use the dependency lockfiles rather than floating upstream branches for reproduction.

## Build and source

The source archive linked by the application includes its source, tests, lockfiles,
license notices, model packer, scripts and deployment configuration. Generated model
bundles are reconstructed by scripts/prepare-models.mjs from SHA-256 verified reference
checkpoints. Generated WASM is reconstructed by scripts/build-wasm.mjs.
