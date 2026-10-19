# Image compression demo

A static Vue 3 / TypeScript image-compression playground, focused on JPEG AI.
All encoding, output decoding and metrics execute locally in CPU WebAssembly.
No backend, upload endpoint, analytics or remote model inference.

## Run locally

Requirements: Node 22.12+ (tested with 26), npm, Rust 1.89+ (tested with 1.96),
`wasm32-unknown-unknown`, Git and `wasm-pack`.

```sh
rustup target add wasm32-unknown-unknown
cargo install wasm-pack --locked
npm ci
npm run build:wasm
npm run models
npm run dev
```

Open http://localhost:5173. Initial model preparation downloads pinned reference
checkpoints at build time (~300 MiB), validates their LFS SHA-256 hashes, and packs
**encoder and decoder tensors**. No Python/PyTorch is needed. The resulting four
bundles total ~104 MiB. Raw checkpoints remain in `.cache/`; model bundles and
WASM are generated, not committed. The manifest and dependency lockfiles are committed.

`npm run models` is required for JPEG AI, not JPEG/AVIF/JPEG XL. Missing assets
produce a visible error rather than simulated compression. `npm run build` refuses
to publish an incomplete model set.

## Production: Vite builds, nginx serves

Vite is the development server and build tool, **not** the production server.
nginx is optional: any HTTPS static host with correct WASM MIME types works.
There is no Node or Rust process in production.

```sh
npm run models
npm run build
npm run preview          # inspect the production build at http://localhost:4173

# Optional self-hosted static deployment:
docker build -t image-compression-demo .
docker run --rm -p 8080:80 image-compression-demo
```

`deploy/nginx.conf` supplies WASM MIME types, COOP/COEP, same-origin CSP,
compression, cache policy and real 404s for missing assets. Deploy behind HTTPS
(or use localhost) for crypto/Cache Storage. All runtime requests are same-origin.
Vite and preview provide the same isolation headers locally. The current codecs
are deliberately single-threaded WASM, so SharedArrayBuffer is not required;
headers leave room for a future threaded build without a service-worker shim.

The app currently targets an origin root (`/`). To deploy under a subpath, update
Vite's base, static legal/favicon URLs, and nginx paths together.

`npm run build` includes an archive of the application's corresponding source and
build scripts at `/source.tar.gz` and license notices at `/legal/`. Keep them with
the deployed site. This application is AGPL-3.0-only; review the license before
modifying or redistributing it. External library and model terms also apply.

## Features

- Local file picker and drag/drop; explicit working size (256–1024 px maximum edge).
- JPEG AI: all four BOP models, adjustable quantisation, actual target-bpp search.
- **Inline JPEG AI model weights:** list sizes/digests, download with progress and cancel, use a
  cached model, remove cached weights, export `.zjb`, import an exact verified
  `.zjb` from this build. Uses Cache Storage, shared with the encoding worker.
- JPEG: quality, progressive mode and chroma subsampling (MozJPEG).
- AVIF: quality, effort, lossless and chroma subsampling (jSquash/libavif).
- JPEG XL: quality, effort, lossless and progressive encoding (jSquash/libjxl).
- Four codec panes, each with its own original-versus-decoded preview and independent settings.
  No sidebar, modal model manager or runs/history table.
- Preview comes from decoding the actual output bytes using WASM, including JPEG AI/JXL.
- Original/output split, device-pixel zoom and scroll-to-pan per pane.
- Encode time, decode time and compression ratio are equally prominent in every pane.
  Ratio = raw working RGB bytes / encoded file bytes; upload/file ratio is also available
  in the file details. Startup/download time is reported separately.
- RGB MSE/PSNR, 8x8 luma SSIM, encoded size/bpp and bitstream/preview/comparison-JSON downloads.
- Encode one codec or queue all four. Jobs run sequentially to avoid CPU/memory contention,
  with separate settings, progress, cancellation and one retained result per format.
- Automatic light/dark mode follows browser `prefers-color-scheme`, including live changes.
- Cancel terminates the worker, freeing its WASM heap. Each new job is a fresh
  worker; downloaded model weights persist independently in browser storage.
- Readable, component-scoped Vue styles. Global CSS is reset/tokens only.

## Architecture

```text
src/App.vue                  shared input and four codec panes
src/components/CodecPane.vue independent settings, measurements and comparison per format
src/components/              input, controls, viewer, measurements, inline model weights
src/lib/comparisons.ts       serial job queue and independent per-codec state/cancellation
src/base.css                 reset and automatic light/dark design tokens only
src/lib/models.ts            verified downloads, streaming progress, cache/import
src/lib/codecs.ts            lazy-loaded single-threaded jSquash WASM codecs
src/lib/runner.ts            one cancellable worker per experiment
src/codec.worker.ts          orchestration; no UI-thread codec work
crates/codec/src/lib.rs      Rust JPEG AI WASM adapter + metrics and unit tests
crates/codec/src/bin/        native encoder+decoder model packer
scripts/                    WASM/model preparation and source archive
public/models/manifest.json  pinned weights and integrity metadata
```

The JPEG AI implementation is pinned to zenjpegai commit
`1192fca6b6c2fdc4dcf29a6c8064c53a987caf59`. Reference weights are pinned to
`b9e573fef1dd90408270da23aae678c823f15525`. Upstream demo weights are decoder-only,
so this project packs its own bundles using the upstream tensor-access Recorder,
including analysis, hyper-encoder, common and synthesis tensors.

## Validation

```sh
npm test                       # Rust unit tests + Vitest
npm run check                  # TypeScript + Vue templates
npx playwright install chromium
npm run build
npm run test:e2e                # real browser WASM; no mocked codec outputs
cargo fmt --manifest-path crates/codec/Cargo.toml --check
cargo clippy --manifest-path crates/codec/Cargo.toml --all-targets --features model-packer -- -D warnings
```

If using an already installed Chromium, set
`PLAYWRIGHT_CHROMIUM_EXECUTABLE=/path/to/chromium`.
The browser suite covers four-codec roundtrips, downloaded bitstreams, report
metrics, lossless equality, JPEG AI target search, cache-only model reuse,
model export/import, independent settings/results, queue cancellation, live color-scheme
changes without altering image pixels, broken files, and mobile overflow. It operates
against `dist/` using Vite preview. It does not certify JPEG AI conformance.

## Limits and honest comparisons

- 8-bit sRGB still images only. Browser APIs import, resize and composite alpha on
  white **before** WASM; output transforms/entropy coding/metrics are WASM. Metadata
  is discarded. No HDR, animation, ICC preservation, or transparent output.
- Max input file 40 MiB, source 40 MP (checked after browser decode), working 1 MP.
  Browser decoding can allocate before the source-size check; not hardened against
  hostile decompression bombs. Use a smaller source on memory-constrained devices.
- JPEG AI is CPU-only BOP, advanced optional coding tools off. No WebGPU or hidden
  server fallback. Modern browsers with WASM SIMD are required for this build.
- Encode working-set estimate has a 768 MiB limit. It is not a whole-tab measured
  peak and excludes browser copies, model cache storage and other tabs.
- JPEG AI target-bpp search can take minutes and uses all four model bundles.
  Requested bpp is a target, not a guarantee; header overhead dominates tiny images.
- The primary compression ratio compares uncompressed RGB working bytes with encoded
  file bytes, excluding model downloads. Upload/file ratio is separate, because a resized
  upload is not a codec-only comparison. Quality
  metrics always compare normalized working pixels against decoded output pixels.
  Settings are snapshotted per result; quality scales differ across formats.
- SSIM: Rec.709 luma, non-overlapping 8x8 windows (including smaller edges), population
  variance, K1=.01/K2=.03/L=255. Not MS-SSIM. PSNR uses RGB MSE, with `null` in JSON
  for infinite PSNR. These scores do not establish perceptual equivalence.
- Model storage may be evicted, denied or out of quota. Compression still works
  by downloading weights directly if caching fails. Export models for portable use.
- `.jai` is an experimental extension; downloading one does not make browsers support it.
- No standardized reference-decoder interoperability or performance promise is made
  by the UI tests. Keep upstream conformance and patent review separate from UI validation.

See `/legal/` and `public/legal/THIRD_PARTY.md`. The reference notice explicitly
reserves third-party/patent rights; an open-source license is not patent clearance.
