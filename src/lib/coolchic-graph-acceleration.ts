import type { TensorGraphSpec } from './coolchic-tensor-graph';
export type GraphCallback = (spec: string, parameters: Float32Array, latents: Float32Array, target: Float32Array,
  noise: Float32Array, common: Float32Array, options: Float32Array) => Float32Array | null;
export interface GraphAcceleration { gradient: GraphCallback; calls: number; evaluations: number; failure?: string; dispose(): void }
export const GRAPH_HEADER = 1024;

/** One bounded worker sleep per complete forward/backward graph. */
export async function createCoolChicGraphAcceleration(): Promise<GraphAcceleration> {
  if (!globalThis.crossOriginIsolated || typeof SharedArrayBuffer === 'undefined' || !navigator.gpu) throw new Error('GPU tensor training requires isolated WebGPU workers');
  const worker = new Worker(new URL('../coolchic-tensor.worker.ts', import.meta.url), { type: 'module' });
  try {
    await new Promise<void>((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error('GPU tensor initialization timed out')), 30000);
      worker.onmessage = ({ data }) => { clearTimeout(timer); data.type === 'ready' ? resolve() : reject(new Error(data.error)); };
      worker.onerror = event => { clearTimeout(timer); reject(new Error(event.message)); };
      worker.postMessage({ type: 'init' });
    });
  } catch (error) { worker.terminate(); throw error; }
  let shared: SharedArrayBuffer | undefined, currentSpec: string | undefined, descriptor: TensorGraphSpec | undefined;
  let disposed = false;
  const backend: GraphAcceleration = {
    calls: 0, evaluations: 0,
    gradient(spec, parameters, latents, target, noise, common, options) {
      if (disposed || backend.failure) return null;
      try {
      const fresh = spec !== currentSpec;
      if (fresh) descriptor = JSON.parse(spec) as TensorGraphSpec;
      const expectedParameters = descriptor!.group_lengths.reduce((a,b) => a+b, 0);
      const expectedLatents = descriptor!.latents.reduce((n,l) => n+l.height*l.width, 0);
      const evaluation = options[0] === 5;
      if (parameters.length !== expectedParameters || latents.length !== expectedLatents || noise.length !== (evaluation ? 0 : expectedLatents) || options.length !== 4) throw new Error('GPU graph input length mismatch');
      const inputLength = parameters.length + latents.length + noise.length + (fresh ? target.length + common.length : 0);
      const outputLength = evaluation ? 1 : parameters.length + latents.length;
      const bytes = GRAPH_HEADER + (inputLength + outputLength) * 4;
      if (!Number.isSafeInteger(bytes) || bytes > 256 * 1024 ** 2) { backend.failure = 'GPU graph transfer exceeds 256 MiB'; return null; }
      if (!shared || shared.byteLength < bytes) shared = new SharedArrayBuffer(bytes);
      const control = new Int32Array(shared, 0, 8), packed = new Float32Array(shared, GRAPH_HEADER, inputLength);
      let offset = 0;
      for (const values of [parameters, latents, noise, ...(fresh ? [target, common] : [])]) { packed.set(values, offset); offset += values.length; }
      Atomics.store(control, 0, 0); Atomics.store(control, 1, 0);
      worker.postMessage({ type: 'gradient', shared, spec, fresh, parameters: parameters.length, latents: latents.length,
        target: fresh ? target.length : 0, common: fresh ? common.length : 0, options, inputLength, outputLength });
      const wait = Atomics.wait(control, 0, 0, 60000);
      if (wait === 'timed-out' || Atomics.load(control, 0) !== 1) {
        const length = Math.min(Atomics.load(control, 1), GRAPH_HEADER - 32);
        backend.failure = length > 0 ? new TextDecoder().decode(new Uint8Array(shared, 32, length).slice()) : 'GPU tensor graph timed out or failed';
        worker.terminate(); return null;
      }
      currentSpec = spec;
      if (evaluation) backend.evaluations++; else backend.calls++;
      return new Float32Array(shared, GRAPH_HEADER + inputLength * 4, outputLength);
      } catch (error) { backend.failure = String(error); worker.terminate(); return null; }
    },
    dispose() { disposed = true; worker.terminate(); shared = undefined; },
  };
  worker.onerror = event => { backend.failure = event.message || 'GPU tensor worker failed'; };
  return backend;
}
