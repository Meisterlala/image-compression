/// <reference lib="webworker" />
import { CoolChicTensorGraph, initializeTensorGpu, type TensorGraphSpec } from './lib/coolchic-tensor-graph';
import type { SoapLayout, SoapBasis } from './lib/coolchic-gpu-optimizer';
let soapBasis: SoapBasis | undefined;
const HEADER = 1024; // Matches the bounded control/error header in the bridge.
let graph: CoolChicTensorGraph | undefined;
let currentSpec: string | undefined;
interface Request {
  type: 'init' | 'gradient'; shared: SharedArrayBuffer; spec: string; fresh: boolean;
  parameters: number; latents: number; target: number; common: number;
  inputLength: number; outputLength: number; options: Float32Array;
}
self.onmessage = async ({ data }: MessageEvent<Request>) => {
  if (data.type === 'init') {
    try { await initializeTensorGpu(); self.postMessage({ type: 'ready' }); }
    catch (error) { self.postMessage({ type: 'error', error: String(error) }); }
    return;
  }
  const control = new Int32Array(data.shared, 0, 8);
  try {
    const { optimizer: layout, ...spec } = JSON.parse(data.spec) as TensorGraphSpec & { optimizer?: SoapLayout };
    const signature = JSON.stringify(spec);
    const input = new Float32Array(data.shared, HEADER, data.inputLength);
    let offset = 0;
    const take = (length: number) => { const values = input.subarray(offset, offset + length); offset += length; return values; };
    const evaluation = data.options[0] === 5, training = data.options[0] === 6;
    const parameters = take(data.parameters), latents = take(data.latents), noise = take(evaluation ? 0 : training ? spec.latents.reduce((n,l) => n+l.height*l.width,0) * data.options[7] : data.latents);
    if (data.fresh) {
      graph?.dispose(); graph = undefined;
      const target = take(data.target), common = take(data.common);
      graph = new CoolChicTensorGraph(spec, target, common);
      currentSpec = signature;
    }
    if (!graph || currentSpec !== signature) throw new Error('GPU tensor graph not initialized');
    const [mode, temperature, lambda, normalizer] = data.options;
    let result: Float32Array;
    if (training) {
      if (!soapBasis) {
        const wasm = await import('./wasm-coolchic/coolchic');
        await wasm.default(); soapBasis = wasm.coolChicSoapBasis;
      }
      if (!layout) throw new Error('Missing GPU optimizer layout');
      result = await graph.train(parameters, noise, { mode: data.options[4], temperature, lambda, normalizer }, layout, data.options[6], data.options[7], data.options[5], soapBasis);
    } else result = evaluation ? new Float32Array([await graph.evaluate(parameters, latents, lambda)])
      : await graph.gradient(parameters, latents, noise, { mode, temperature, lambda, normalizer });
    if (result.length !== data.outputLength) throw new Error('GPU gradient output length mismatch');
    new Float32Array(data.shared, HEADER + data.inputLength * 4, data.outputLength).set(result);
    Atomics.store(control, 0, 1);
  } catch (error) {
    const message = new TextEncoder().encode(error instanceof Error ? error.stack ?? error.message : String(error)).subarray(0, HEADER - 32);
    new Uint8Array(data.shared, 32, message.length).set(message);
    Atomics.store(control, 1, message.length); Atomics.store(control, 0, -1);
  } finally { Atomics.notify(control, 0); }
};
