import type { EncodeProgress, Settings } from '../types';
import { createCoolChicAcceleration, type Acceleration } from './coolchic-acceleration';

// Full reference MSE intra training and real bitstream decoding, in a dedicated
// WASM module/worker, with optional WebGPU convolution acceleration.
export async function loadCoolChic(settings: Settings, progress: (text: string, info?: EncodeProgress) => void) {
  const point = settings.coolchicPoint ?? 'lop';
  const iterations = settings.coolchicIterations ?? 2000;
  const lambda = settings.coolchicLambda ?? 0.001;
  if (!['lop', 'mop', 'hop', 'vhop'].includes(point)) throw new Error('Unknown Cool-chic operating point.');
  if (!Number.isInteger(iterations) || iterations < 100 || iterations > 100000) {
    throw new Error('Cool-chic training requires 100–100000 iterations.');
  }
  if (!Number.isFinite(lambda) || lambda < 0.000001 || lambda > 0.1) {
    throw new Error('Cool-chic rate constraint must be between 0.000001 and 0.1.');
  }
  const { default: init, CoolChicEncoder, decodeCoolChic, estimateCoolChicMemory, setCoolChicAccelerator, coolChicCpuConvolution } = await import('../wasm-coolchic/coolchic');
  await init();
  const notices: string[] = [];
  return {
    notices,
    estimateMemory: (width: number, height: number) => estimateCoolChicMemory(width, height, point),
    async encode(image: ImageData): Promise<Uint8Array<ArrayBuffer>> {
      let encoder: InstanceType<typeof CoolChicEncoder> | undefined;
      let acceleration: Acceleration | undefined;
      try {
        if (settings.coolchicGpu !== false) {
          try {
            progress('Initializing Cool-chic WebGPU training…');
            acceleration = await createCoolChicAcceleration((p, x, w, b, g) => coolChicCpuConvolution(p, x, w, b, g ?? undefined));
            setCoolChicAccelerator(acceleration.convolution, 8_000_000);
          } catch (error) {
            notices.push(`WebGPU training unavailable; using CPU WASM. ${error instanceof Error ? error.message : String(error)}`);
          }
        }
        const started = performance.now();
        encoder = new CoolChicEncoder(new Uint8Array(image.data.buffer, image.data.byteOffset, image.data.byteLength), image.width, image.height, point, iterations, lambda);
        let lastProgress = -Infinity;
        while (!encoder.step()) {
          const now = performance.now();
          if (now - lastProgress >= 100) {
            const quantizing = encoder.completed >= iterations;
            progress(`${quantizing ? 'Final optimization' : 'Training'} Cool-chic ${point.toUpperCase()} · ${encoder.status}`, {
              stage: quantizing ? 'quantizing' : 'training',
              completed: encoder.progressCompleted,
              total: encoder.progressTotal,
              elapsedMs: now - started,
            });
            lastProgress = now;
          }
          // This runs only in a dedicated worker: postMessage updates the UI
          // without timer pauses. Cancellation terminates the worker directly,
          // including an in-flight WASM call.
        }
        progress('Entropy-coding the .cool bitstream…');
        const trained = encoder;
        encoder = undefined; // finish consumes the Rust trainer, including on error.
        return new Uint8Array(trained.finish());
      } finally {
        setCoolChicAccelerator(undefined, 0);
        encoder?.free();
        if (acceleration?.failure) notices.push(`WebGPU training fell back to CPU WASM: ${acceleration.failure}`);
        notices.push(acceleration?.calls
          ? `Training used WebGPU (${acceleration.calls} convolution jobs) + CPU WASM. Portable decoding stays on CPU.`
          : 'Training used CPU WASM SIMD (GPU disabled, unavailable, or operations below the offload threshold).');
        acceleration?.dispose();
      }
    },
    decode(buffer: ArrayBuffer): ImageData {
      const image = decodeCoolChic(new Uint8Array(buffer));
      try { return new ImageData(new Uint8ClampedArray(image.rgba()), image.width, image.height); }
      finally { image.free(); }
    },
  };
}
