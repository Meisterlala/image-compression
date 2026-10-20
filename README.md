# Image compression playground

Compare **JPEG AI, JPEG, AVIF, JPEG XL and Cool-chic v5** in your browser.
Encoding, decoding and measurements run locally; images are not uploaded.
There is no backend. Codecs use Rust or jSquash WebAssembly, with optional
WebGPU decoding for JPEG AI and adaptive WebGPU training for Cool-chic.

## Run locally

Requires Node 22.12+, npm, Rust 1.89+, Git and `wasm-pack`.

```sh
rustup target add wasm32-unknown-unknown
cargo install wasm-pack --locked
# The separate threaded Cool-chic WASM build needs a pinned nightly stdlib:
rustup toolchain install nightly-2026-02-28 --profile minimal --component rust-src --target wasm32-unknown-unknown
npm ci
npm run build:wasm
npm run models
npm run dev
```

Open http://localhost:5173.

Model preparation downloads about 440 MiB of pinned JPEG AI checkpoints,
verifies them and creates encoder/decoder bundles. It requires no Python.
These assets are needed only for JPEG AI. In the app, explicitly download the
models you want into browser storage before encoding; automatic bitrate search
needs all four models for the selected operating point.

**After changing Rust, rebuild WASM.** Vite and `npm run build:web` do not
recompile Rust. Use `npm run build:wasm`, then rebuild or reload the frontend.
Generated WASM and model bundles are not committed.

Cool-chic defaults to adaptive GPU training and CPU multithreading (up to 8
workers for larger images). GPU can be disabled; CPU SIMD/threading are automatic,
with no off switch. Threading and the
GPU bridge require a secure context and cross-origin isolation; the supplied Vite
and nginx configs provide COOP/COEP headers. Unsupported browsers fall back to
portable single-thread SIMD WASM. Progress and result notices report the backend.
GPU utilization may stay low when measured CPU execution is faster; calibration
includes transfers/readback, not just shader time. Training is still expensive.
See [acceleration validation and measurements](docs/coolchic-acceleration.md).

## Use

Choose an image, optionally downscale it, select a codec and encode.
Compare decoded outputs side by side and download the compressed file or decoded
PNG. The table shows actual file size, encode/decode time, PSNR, luma SSIM and
bits per pixel. Previews and quality measurements come from decoding the output
bytes, not from an encoder approximation.

Settings, the source image and completed outputs are stored locally across
reloads. Jobs run one at a time; cancelling terminates the worker. Browser storage
limits or eviction can remove saved data and downloaded models.

### Codecs

- **JPEG AI:** SOP/BOP/HOP models, quantization and target-bitrate search.
  Encoding uses CPU WASM. Optional WebGPU decoding falls back to CPU when
  unavailable; the preview identifies the decoder actually used. GPU and CPU
  pixels can differ, and metrics measure the actual preview.
- **JPEG:** MozJPEG, including progressive encoding and chroma subsampling.
- **AVIF:** libavif/libaom, including lossless mode.
- **JPEG XL:** libjxl, including lossless and progressive encoding.
- **Cool-chic v5:** Rust implementation of the reference RGB intra MSE encoder
  and decoder, with LOP/MOP/HOP/VHOP architectures, rate constraint and training
  budget controls. Short runs are available but can hurt compression or fidelity.
  Algorithm details, scope and provenance: [crate README](crates/coolchic/README.md).

JPEG AI target search and Cool-chic training can take substantial time. Target
bitrate and quality controls are not interchangeable across codecs.

## Production build

```sh
npm run models
npm run build
npm run preview          # http://localhost:4173
```

Deploy `dist/` on an HTTPS static host with the correct WASM MIME type. Production
needs no Node or Rust service. The included nginx configuration supplies security
headers, compression and caching. An optional container is available:

```sh
docker build -t image-compression-demo .
docker run --rm -p 8080:80 image-compression-demo
```

The app targets the origin root (`/`). Subpath deployments require matching
changes to Vite's base, asset URLs and server configuration. Use a modern browser
with WASM SIMD; model storage and WebGPU require HTTPS or localhost.

## Limits

- 8-bit sRGB still images. Alpha is composited on white; metadata is discarded.
  No HDR, animation, ICC preservation or transparent output.
- Input files: 40 MiB; decoded input: 40 million pixels. Browser image decoding
  can allocate before these checks, so hostile images are not safely sandboxed.
- Cool-chic: 4 million pixels and dimensions below 16384. Scope is RGB MSE
  still-image training, not video, YUV or perceptual Wasserstein training.
- Neural-codec jobs estimated above 3 GiB are rejected; above 1.5 GiB they require
  confirmation. Estimates are not measured whole-browser peaks and cannot
  guarantee that a device has enough memory. Downscaling is always explicit.
- File-size comparisons exclude shared JPEG AI model downloads. Metrics compare
  the decoded output with its encode-time input. These measurements are not a
  standardized codec benchmark or conformance certification.

## Development

- `src/`: Vue UI, local persistence and worker orchestration.
- `crates/codec/`: JPEG AI adapter, model packer and image metrics.
- `crates/coolchic/`: native/browser Cool-chic encoder and decoder.
- `scripts/`: WASM builds, model preparation and source packaging.
- `public/legal/`: upstream pins, licenses and notices.

```sh
npm run check
npm test
npm run test:e2e         # uses the production build; Chromium required
```

## License

Application code is **AGPL-3.0-only**. Production builds include corresponding
source and build scripts at `/source.tar.gz`; keep it and `/legal/` with the site.
Third-party code and model terms also apply—see
[third-party notices](public/legal/THIRD_PARTY.md). Open-source licenses do not
provide patent clearance.
