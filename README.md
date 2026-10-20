# Image compression playground

View it all online at https://compress.meisterlala.dev

Compare JPEG AI, JPEG, AVIF, JPEG XL and Cool-chic v5 in your browser.
Images stay on your device; there is no backend. Previews and metrics use decoded
output bytes. Download compressed files or decoded PNGs; work is saved locally
across reloads.

## Run

Requires Node 22.12+, npm, Git, Rust 1.89+ and `wasm-pack`.

```sh
rustup target add wasm32-unknown-unknown
cargo install wasm-pack --locked
rustup toolchain install nightly-2026-02-28 --profile minimal --component rust-src --target wasm32-unknown-unknown
npm ci
npm run build:wasm
npm run dev
```

Open http://localhost:5173. For JPEG AI, first run `npm run models` (about 440 MiB
of pinned checkpoints), then download the desired models in the app. Target-bitrate
search needs all four models for the selected operating point.

**Rebuild WASM after Rust changes.** Vite and `npm run build:web` do not compile
Rust. Generated WASM and model bundles are not committed.

## Codecs and limits

- **JPEG AI:** SOP/BOP/HOP, CPU encoding, optional WebGPU decoding.
- **JPEG:** MozJPEG, progressive encoding and chroma subsampling.
- **AVIF / JPEG XL:** lossy and lossless encoding.
- **Cool-chic v5:** per-image training, LOP/MOP/HOP/VHOP, rate and iteration
  controls. Optional GPU training; automatic CPU SIMD/threading. See the
  [codec README](crates/coolchic/README.md) and [acceleration docs](docs/coolchic-acceleration.md).

Neural encoding can be slow. Quality and bitrate controls differ across codecs.
Metrics are PSNR, luma SSIM and bits per pixel; file sizes exclude shared JPEG AI
models. This is a playground, not a standardized benchmark.

Inputs are 8-bit sRGB still images: alpha is composited on white, metadata is
removed. No HDR or animation. Limits: 40 MiB input, 40 million pixels; Cool-chic
allows 4 million pixels and dimensions below 16384. Neural jobs estimated above
3 GiB are rejected; above 1.5 GiB require confirmation. Estimates cannot guarantee
available memory, and browser decoding may allocate before checks. Downscaling is
explicit. Browser storage can be evicted; saved work is not a backup.

## Build and test

```sh
npm run check
npm test
npm run models
npm run build           # WASM, source archive, then frontend → dist/
npm run test:e2e        # production build; Playwright Chromium required
npm run preview        # http://localhost:4173
```

Serve `dist/` at the origin root over HTTPS with WASM MIME support and COOP/COEP
headers for shared-memory acceleration. Vite and `deploy/nginx.conf` supply these.
Without shared-memory support, Cool-chic uses single-thread SIMD WASM. Missing
WebGPU disables GPU training, not CPU threading. Docker serves a prebuilt `dist/`:

```sh
docker build -t image-compression-demo .
docker run --rm -p 8080:80 image-compression-demo
```

### Build and push a multi-architecture image

With the build prerequisites above and Docker Buildx installed:

```sh
docker login registry.meisterlala.dev
./scripts/build-and-push.sh          # :latest
./scripts/build-and-push.sh v0.1.0   # explicit tag
```

The script runs `npm ci`, prepares models, and builds WASM and the frontend on
the host, then pushes `registry.meisterlala.dev/image-compression` for
`linux/amd64` and `linux/arm64`. Override `IMAGE`, `PLATFORMS`, or `BUILDER` via
environment variables. The first model download is about 440 MiB.

Code: `src/` (Vue/workers), `crates/codec/` (JPEG AI/metrics), `crates/coolchic/`
(Cool-chic), `scripts/` (builds/verification). See [persistence](docs/persistence.md)
for local state APIs.

## License

AGPL-3.0-only. Keep `/source.tar.gz` and `/legal/` with production deployments.
[Third-party terms](public/legal/THIRD_PARTY.md) also apply; open-source licenses
do not provide patent clearance.
