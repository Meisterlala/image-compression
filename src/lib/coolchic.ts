import type { EncodeProgress, Settings } from '../types';
import { createCoolChicGraphAcceleration, type GraphAcceleration } from './coolchic-graph-acceleration';
import { loadCoolChicRuntime } from './coolchic-runtime';

// Full reference MSE intra training and real bitstream decoding, in a dedicated
// WASM module/worker, with optional GPU-resident forward/backward execution.
export async function loadCoolChic(settings: Settings, progress: (text: string, info?: EncodeProgress) => void, onGpuFailure: () => void = () => {}) {
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
  const { bindings: portable } = await loadCoolChicRuntime(false);
  const notices: string[] = [];
  return {
    notices,
    estimateMemory: (width: number, height: number) => portable.estimateCoolChicMemory(width, height, point),
    async encode(image: ImageData): Promise<Uint8Array<ArrayBuffer>> {
      const runtime = await loadCoolChicRuntime(image.width * image.height >= 16384);
      const { CoolChicEncoder, setCoolChicGraphAccelerator } = runtime.bindings;
      if (runtime.warning) notices.push(runtime.warning);
      notices.push(`CPU training backend: WASM SIMD · ${runtime.threads} thread${runtime.threads === 1 ? '' : 's'}.`);
      let encoder: InstanceType<typeof CoolChicEncoder> | undefined;
      let acceleration: GraphAcceleration | undefined;
      let gpuFailed = false;
      const reportGpuFailure = () => {
        if (!gpuFailed) onGpuFailure();
        gpuFailed = true;
      };
      try {
        if (settings.coolchicGpu !== false && image.width * image.height >= 16384) {
          try {
            progress('Initializing Cool-chic WebGPU training…');
            acceleration = await createCoolChicGraphAcceleration();
            setCoolChicGraphAccelerator(acceleration.gradient);
          } catch (error) {
            reportGpuFailure();
            notices.push(`WebGPU training unavailable; using CPU WASM. ${error instanceof Error ? error.message : String(error)}`);
          }
        }
        const started = performance.now();
        encoder = new CoolChicEncoder(new Uint8Array(image.data.buffer, image.data.byteOffset, image.data.byteLength), image.width, image.height, point, iterations, lambda);
        let lastProgress = -Infinity;
        while (!encoder.step()) {
          if (acceleration?.failure) reportGpuFailure();
          const now = performance.now();
          if (now - lastProgress >= 100) {
            const quantizing = encoder.completed >= iterations;
            progress(`${quantizing ? 'Final optimization' : 'Training'} Cool-chic ${point.toUpperCase()} · ${runtime.threads} CPU thread${runtime.threads === 1 ? '' : 's'}${acceleration?.calls ? ' + WebGPU' : ''} · ${encoder.status}`, {
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
        setCoolChicGraphAccelerator(undefined);
        encoder?.free();
        if (acceleration?.failure) {
          reportGpuFailure();
          notices.push(`WebGPU training fell back to CPU WASM: ${acceleration.failure}`);
        }
        if (acceleration?.calls) notices.push(`WebGPU tensor graph: ${acceleration.calls} training iterations${acceleration.batches ? ` in ${acceleration.batches} GPU batches` : ''}, ${acceleration.evaluations} RD evaluations. Portable decoding uses WASM.`);
        else if (settings.coolchicGpu === false) notices.push('WebGPU training disabled.');
        else if (image.width * image.height < 16384) notices.push('Small image: CPU SIMD avoids GPU graph startup overhead.');
        acceleration?.dispose();
      }
    },
    decode(buffer: ArrayBuffer): ImageData {
      const image = portable.decodeCoolChic(new Uint8Array(buffer));
      try { return new ImageData(new Uint8ClampedArray(image.rgba()), image.width, image.height); }
      finally { image.free(); }
    },
  };
}
