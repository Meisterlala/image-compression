/// <reference lib="webworker" />
import init, { JpegAi, metrics } from './wasm/codec';
import { loadCodec } from './lib/codecs';
import { decodeJpegAi } from './lib/jpegai-decode';
import type { Job, Result, WorkerMessage } from './types';
import { getCachedModel, loadManifest } from './lib/models';
import { JPEG_AI_CONFIRM_THRESHOLD, JPEG_AI_MEMORY_LIMIT, MAX_PIXELS } from './lib/limits';

const progress = (text: string) => self.postMessage({ type: 'progress', text } satisfies WorkerMessage);
let resume: (() => void) | undefined;
self.onmessage = async ({ data }: MessageEvent<Job | { type: 'confirm' }>) => {
  if ('type' in data) {
    const proceed = resume;
    resume = undefined;
    proceed?.();
    return;
  }
  const job = data;
  try {
    const { width, height, pixels, settings: s } = job;
    if (!Number.isInteger(width) || !Number.isInteger(height) || width < 1 || height < 1 || width * height > MAX_PIXELS || pixels.length !== width * height * 4) throw new Error('Image must contain between 1 and 40 million pixels.');
    if (s.format === 'coolchic' && (width * height > 4_000_000 || width >= 16384 || height >= 16384)) {
      throw new Error('Cool-chic supports up to 4 million pixels and dimensions below 16384. Choose a smaller Downscale image size.');
    }
    progress('Loading local WASM runtime…');
    const start = performance.now();
    await init();
    let encoded: Uint8Array<ArrayBuffer>;
    let decoded: Uint8ClampedArray<ArrayBuffer>;
    let decodeBackend: Result['decodeBackend'];
    let decodeWarning: string | undefined;
    let modelSize = 0;
    let estimatedMemory: number | null = null;
    const notices: string[] = [];
    let encodeMs: number, decodeMs: number, startupMs: number;
    let confirmationWaitMs = 0;
    if (s.format === 'jpegai') {
      const ai = new JpegAi(s.operatingPoint);
      try {
      const point = s.operatingPoint.toUpperCase();
      estimatedMemory = ai.estimateMemory(width, height, s.rateMode === 'target');
      if (estimatedMemory > JPEG_AI_MEMORY_LIMIT) throw new Error('Estimated JPEG AI working memory exceeds the 3 GiB cap. Choose a smaller size under Downscale image.');
      if (estimatedMemory > JPEG_AI_CONFIRM_THRESHOLD) {
        const memoryBytes = estimatedMemory;
        const waiting = performance.now();
        await new Promise<void>(resolve => {
          resume = resolve;
          self.postMessage({ type: 'confirmation-required', memoryBytes } satisfies WorkerMessage);
        });
        confirmationWaitMs = performance.now() - waiting;
      }
      const manifest = await loadManifest(s.operatingPoint);
      const entries = s.rateMode === 'target' ? manifest.models : manifest.models.filter(m => m.id === s.model);
      if (entries.length !== (s.rateMode === 'target' ? 4 : 1)) throw new Error('Incomplete JPEG AI model manifest');
      for (const entry of entries) {
        progress(`Reading cached JPEG AI ${point} model ${entry.id + 1} · ${(entry.bytes / 1048576).toFixed(1)} MiB…`);
        const { bytes } = await getCachedModel(entry, (received, total) => {
          progress(`Reading cached model ${entry.id + 1} · ${Math.round(received / total * 100)}%…`);
        });
        notices.push(ai.addModels(bytes)); modelSize += bytes.length;
      }
      startupMs = performance.now() - start - confirmationWaitMs;
      progress(s.rateMode === 'target' ? `Searching all four ${point} models for the target bitrate. This may take several minutes…` : `Encoding JPEG AI ${point} on your CPU…`);
      let tick = performance.now();
      encoded = new Uint8Array(ai.encode(new Uint8Array(pixels.buffer), width, height, s.model, s.displacement, s.rateMode === 'target' ? s.targetBpp : 0));
      encodeMs = performance.now() - tick;
      progress(s.gpuDecode !== false ? `Decoding JPEG AI ${point} with WebGPU (CPU fallback if needed)…` : `Decoding JPEG AI ${point} on CPU…`);
      tick = performance.now();
      const reconstruction = await decodeJpegAi(ai, encoded, s.gpuDecode !== false);
      decoded = new Uint8ClampedArray(reconstruction.pixels);
      decodeBackend = reconstruction.backend;
      decodeWarning = reconstruction.warning;
      decodeMs = performance.now() - tick;
      } finally {
        ai.free();
      }
    } else {
      progress(`Loading ${s.format.toUpperCase()} encoder and decoder…`);
      const codec = await loadCodec(s.format, s, progress);
      if ('estimateMemory' in codec) {
        estimatedMemory = codec.estimateMemory(width, height);
        if (estimatedMemory > 3 * 1024 ** 3) throw new Error('Cool-chic reference training exceeds the 3 GiB browser memory budget. Choose a smaller Downscale image size or a lighter operating point.');
        if (estimatedMemory > JPEG_AI_CONFIRM_THRESHOLD) {
          const memoryBytes = estimatedMemory;
          const waiting = performance.now();
          await new Promise<void>(resolve => {
            resume = resolve;
            self.postMessage({ type: 'confirmation-required', memoryBytes } satisfies WorkerMessage);
          });
          confirmationWaitMs = performance.now() - waiting;
        }
      }
      startupMs = performance.now() - start - confirmationWaitMs;
      progress(`Encoding ${s.format.toUpperCase()} locally…`);
      let tick = performance.now();
      encoded = await codec.encode(new ImageData(pixels, width, height));
      encodeMs = performance.now() - tick;
      progress('Decoding compressed output for an honest preview…');
      tick = performance.now();
      const image = await codec.decode(encoded.buffer);
      if (!image || image.width !== width || image.height !== height) throw new Error('Decoded dimensions do not match the source');
      decoded = new Uint8ClampedArray(image.data);
      decodeMs = performance.now() - tick;
    }
    progress('Measuring quality in Rust/WASM…');
    const tick = performance.now();
    const measured = metrics(new Uint8Array(pixels.buffer), new Uint8Array(decoded.buffer), width, height) as Result['metrics'];
    const result: Result = { referenceBytes: job.referenceBytes, referenceDownscaled: job.referenceDownscaled, settings: s, width, height, encoded, pixels: decoded, metrics: measured, startupMs, encodeMs, decodeMs, metricsMs: performance.now() - tick, estimatedMemory, decodeBackend, decodeWarning, modelBytes: modelSize, notices: [...new Set(notices)] };
    self.postMessage({ type: 'result', result } satisfies WorkerMessage, { transfer: [encoded.buffer, decoded.buffer] });
  } catch (e) {
    self.postMessage({ type: 'error', error: e instanceof Error ? e.message : String(e) } satisfies WorkerMessage);
  }
};
