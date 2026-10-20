// The Rust trainer is synchronous. A second worker owns WebGPU and completes
// async readbacks while the training worker sleeps (not spins) in Atomics.wait.
// No global cross-codec GPU state. Input views may borrow shared WASM memory.
import { convolutionShape } from './coolchic-shape';

/** Synchronous callback. Results may borrow the response buffer and are valid
 * until the next call; Rust copies them before returning from this call stack.
 * The async factory does not make convolution() asynchronous. */
export type ConvCallback = (params: Uint32Array, input: Float32Array, weight: Float32Array, bias: Float32Array, gradient: Float32Array | null) => Float32Array | null;
export interface Acceleration {
  convolution: ConvCallback;
  calls: number;
  calibrations: { shape: string; cpuMs: number; gpuMs: number; useGpu: boolean }[];
  failure?: string;
  dispose(): void;
}
export async function createCoolChicAcceleration(cpu?: ConvCallback): Promise<Acceleration> {
  if (!globalThis.crossOriginIsolated || typeof SharedArrayBuffer === 'undefined') {
    throw new Error('GPU training needs cross-origin isolation (COOP/COEP headers)');
  }
  if (typeof WorkerGlobalScope === 'undefined' || !(globalThis instanceof WorkerGlobalScope)) {
    throw new Error('GPU training must run in a dedicated worker');
  }
  if (!navigator.gpu) throw new Error('WebGPU is not available in this worker');
  const worker = new Worker(new URL('../coolchic-gpu.worker.ts', import.meta.url), { type: 'module' });
  try {
    await new Promise<void>((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error('GPU initialization timed out')), 15000);
      worker.onmessage = ({ data }) => {
        clearTimeout(timer);
        if (data.type === 'ready') resolve(); else reject(new Error(data.message || 'GPU initialization failed'));
      };
      worker.onerror = event => { clearTimeout(timer); reject(new Error(event.message)); };
      worker.postMessage({ type: 'init' });
    });
  } catch (error) { worker.terminate(); throw error; }
  let shared: SharedArrayBuffer | undefined;
  let disposed = false;
  const decisions = new Map<string, boolean>();
  const raw: ConvCallback = (params, input, weight, bias, gradient) => {
      if (disposed || backend.failure) return null;
      const shape = convolutionShape(params);
      if (input.length !== shape.inputCount || weight.length !== shape.weights || bias.length !== shape.co
          || !!gradient !== !!shape.backward || (gradient?.length ?? 0) !== (shape.backward ? shape.outputCount : 0)) {
        throw new RangeError('Invalid GPU convolution payload');
      }
      const inputLength = shape.inputLength, outputLength = shape.resultCount, bytes = shape.transferBytes;
      if (bytes > 256 * 1024 ** 2) {
        backend.failure = 'GPU transfer exceeds the 256 MiB budget';
        return null;
      }
      if (!shared || shared.byteLength < bytes) shared = new SharedArrayBuffer(bytes);
      const control = new Int32Array(shared, 0, 8);
      const packed = new Float32Array(shared, 32, inputLength);
      let offset = 0;
      for (const values of [input, weight, bias, gradient]) {
        if (values) { packed.set(values, offset); offset += values.length; }
      }
      Atomics.store(control, 0, 0);
      worker.postMessage({ type: 'convolution', params, shared, inputLength, outputLength });
      // Bounded wait means device/worker failure cannot deadlock a training job.
      const wait = Atomics.wait(control, 0, 0, 30000);
      if (wait === 'timed-out' || Atomics.load(control, 0) !== 1) {
        backend.failure = wait === 'timed-out' ? 'GPU compute timed out' : 'GPU compute failed';
        worker.terminate();
        return null;
      }
      // Rust copies this completed response synchronously before the next
      // request. Returning the view avoids another full-size host copy.
      return new Float32Array(shared, 32 + inputLength * 4, outputLength);
  };
  const backend: Acceleration = {
    calls: 0,
    calibrations: [],
    convolution(params, input, weight, bias, gradient) {
      const key = Array.from(params).join(',');
      if (decisions.get(key) === false) return null;
      let result: Float32Array | null;
      if (cpu && !decisions.has(key)) {
        // Ignore first-dispatch warm-up, then measure the complete bridge cost,
        // including host packing/readback. Require headroom for WASM copies.
        if (!raw(params, input, weight, bias, gradient)) return null;
        let start = performance.now();
        result = raw(params, input, weight, bias, gradient);
        const gpuMs = performance.now() - start;
        start = performance.now();
        const reference = cpu(params, input, weight, bias, gradient);
        const cpuMs = performance.now() - start;
        const correct = result && reference && result.length === reference.length
          && result.every((v, i) => Math.abs(v - reference[i]) <= 3e-5 + Math.abs(reference[i]) * 1e-3);
        const useGpu = !!correct && gpuMs < cpuMs * 0.5;
        backend.calibrations.push({ shape: key, cpuMs, gpuMs, useGpu });
        decisions.set(key, useGpu);
        if (!useGpu) return null; // Rust caches rejection: no more copies for this shape.
      } else result = raw(params, input, weight, bias, gradient);
      if (result) backend.calls++;
      return result;
    },
    dispose() { disposed = true; worker.terminate(); shared = undefined; },
  };
  worker.onmessage = ({ data }) => { if (data.type === 'error') backend.failure = data.message; };
  worker.onerror = event => { backend.failure = event.message || 'GPU worker failed'; };
  return backend;
}
