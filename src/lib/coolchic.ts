import type { Settings } from '../types';

// Full reference MSE intra training and real bitstream decoding, in a dedicated
// WASM module/worker. No Python, GPU runtime, downloaded weights or backend.
export async function loadCoolChic(settings: Settings, progress: (text: string) => void) {
  const point = settings.coolchicPoint ?? 'lop';
  const iterations = settings.coolchicIterations ?? 2000;
  const lambda = settings.coolchicLambda ?? 0.001;
  if (!['lop', 'mop', 'hop', 'vhop'].includes(point)) throw new Error('Unknown Cool-chic operating point.');
  if (!Number.isInteger(iterations) || iterations < 500 || iterations > 100000) {
    throw new Error('Cool-chic training requires 500–100000 iterations.');
  }
  if (!Number.isFinite(lambda) || lambda < 0.000001 || lambda > 0.1) {
    throw new Error('Cool-chic rate constraint must be between 0.000001 and 0.1.');
  }
  const { default: init, CoolChicEncoder, decodeCoolChic, estimateCoolChicMemory } = await import('../wasm-coolchic/coolchic');
  await init();
  return {
    estimateMemory: (width: number, height: number) => estimateCoolChicMemory(width, height, point),
    async encode(image: ImageData): Promise<Uint8Array<ArrayBuffer>> {
      let encoder: InstanceType<typeof CoolChicEncoder> | undefined;
      try {
        encoder = new CoolChicEncoder(new Uint8Array(image.data.buffer, image.data.byteOffset, image.data.byteLength), image.width, image.height, point, iterations, lambda);
        let lastProgress = -Infinity;
        while (!encoder.step()) {
          const now = performance.now();
          if (now - lastProgress >= 100) {
            progress(`Training Cool-chic ${point.toUpperCase()} · ${encoder.status}`);
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
      } finally { encoder?.free(); }
    },
    decode(buffer: ArrayBuffer): ImageData {
      const image = decodeCoolChic(new Uint8Array(buffer));
      try { return new ImageData(new Uint8ClampedArray(image.rgba()), image.width, image.height); }
      finally { image.free(); }
    },
  };
}
