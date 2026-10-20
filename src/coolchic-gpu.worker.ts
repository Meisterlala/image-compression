/// <reference lib="webworker" />
import { CoolChicGpu } from './lib/coolchic-gpu';

let gpu: CoolChicGpu | undefined;
self.onmessage = async ({ data }: MessageEvent<{ type: 'init' | 'dispose' | 'convolution'; params?: Uint32Array; shared?: SharedArrayBuffer; inputLength?: number; outputLength?: number }>) => {
  if (data.type === 'init') {
    try { gpu = await CoolChicGpu.create(); self.postMessage({ type: 'ready' }); }
    catch (error) { self.postMessage({ type: 'error', message: String(error) }); }
    return;
  }
  if (data.type === 'dispose') { gpu?.dispose(); self.close(); return; }
  if (!data.shared) return;
  const control = new Int32Array(data.shared, 0, 8);
  try {
    if (!gpu || !data.params || !data.inputLength || !data.outputLength) throw new Error('GPU worker not initialized');
    const input = new Float32Array(data.shared, 32, data.inputLength);
    const result = await gpu.convolution(data.params, input);
    if (result.length !== data.outputLength) throw new Error('GPU output shape mismatch');
    new Float32Array(data.shared, 32 + data.inputLength * 4, result.length).set(result);
    Atomics.store(control, 0, 1);
  } catch (error) {
    self.postMessage({ type: 'error', message: String(error) });
    Atomics.store(control, 0, -1);
  } finally { Atomics.notify(control, 0); }
};
