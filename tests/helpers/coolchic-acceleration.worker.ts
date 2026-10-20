/// <reference lib="webworker" />
// Development-only harness served by the GPU verifier's Vite server.
import init, { CoolChicEncoder, decodeCoolChic, setCoolChicAccelerator, coolChicCpuConvolution } from '../../src/wasm-coolchic/coolchic';
import { createCoolChicAcceleration } from '../../src/lib/coolchic-acceleration';
import { loadCoolChic } from '../../src/lib/coolchic';
import { defaultSettings } from '../../src/types';
import { loadCoolChicRuntime } from '../../src/lib/coolchic-runtime';
self.onmessage = async ({ data }) => {
  try {
    if (data.type === 'bridge') {
      const gpu = await createCoolChicAcceleration();
      try {
        const result = gpu.convolution(new Uint32Array([1, 1, 1, 2, 1, 0, 0, 0]), new Float32Array([1, 2]), new Float32Array([3]), new Float32Array([4]), null);
        self.postMessage({ result: Array.from(result ?? []), calls: gpu.calls, failure: gpu.failure });
      } finally { gpu.dispose(); }
      return;
    }
    const w = data.width ?? 32, h = data.height ?? 24;
    const rgba = new Uint8Array(w * h * 4);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const i = (y * w + x) * 4;
      rgba.set([x * 255 / w, y * 255 / h, (x + y) * 255 / (w + h), 255], i);
    }
    await init();
    if (data.type === 'loader') {
      if (data.unavailable) Object.defineProperty(navigator, 'gpu', { value: undefined });
      const codec = await loadCoolChic({ ...defaultSettings, coolchicGpu: data.gpu, coolchicIterations: 100 }, () => {});
      const start = performance.now();
      const bytes = await codec.encode(new ImageData(new Uint8ClampedArray(rgba), w, h));
      if (!(bytes.buffer instanceof ArrayBuffer)) throw new Error('Expected owned encoded bytes');
      self.postMessage({ bytes, notices: codec.notices, ms: performance.now() - start }, [bytes.buffer]);
      return;
    }
    const runtime = await loadCoolChicRuntime(!!data.threads, typeof data.threads === 'number' ? data.threads : undefined);
    const { CoolChicEncoder, setCoolChicAccelerator, coolChicCpuConvolution } = runtime.bindings;
    if (data.type === 'kernel') {
      const result = coolChicCpuConvolution(new Uint32Array(data.params), new Float32Array(data.input), new Float32Array(data.weight), new Float32Array(data.bias), data.gradient ? new Float32Array(data.gradient) : undefined);
      if (!(result.buffer instanceof ArrayBuffer)) throw new Error('Expected owned diagnostic output');
      self.postMessage({ result, threads: runtime.threads, warning: runtime.warning }, [result.buffer]);
      return;
    }
    // Force offload on a tiny real job to test the entire WASM callback path,
    // not just direct shader invocations. Production uses a crossover threshold.
    const gpu = data.gpu ? await createCoolChicAcceleration(data.calibrate
      ? (p, x, w, b, g) => coolChicCpuConvolution(p, x, w, b, g ?? undefined) : undefined) : undefined;
    let encoder: CoolChicEncoder | undefined;
    try {
      setCoolChicAccelerator(gpu?.convolution, data.threshold ?? 0);
      const start = performance.now();
      encoder = new CoolChicEncoder(rgba, w, h, data.point ?? 'lop', data.iterations ?? 100, 0.001);
      const steps = data.steps ?? Infinity;
      let count = 0;
      while (!encoder.step() && ++count < steps) {}
      if (count >= steps) {
        self.postMessage({ ms: performance.now() - start, calls: gpu?.calls ?? 0, failure: gpu?.failure, calibrations: gpu?.calibrations, threads: runtime.threads, warning: runtime.warning });
        return;
      }
      const trained = encoder; encoder = undefined;
      const bytes = trained.finish();
      const ms = performance.now() - start;
      setCoolChicAccelerator(undefined, 0);
      const image = decodeCoolChic(bytes);
      try {
        const pixels = image.rgba();
        let squaredError = 0;
        for (let i = 0; i < pixels.length; i++) if (i % 4 !== 3) squaredError += (pixels[i] - rgba[i]) ** 2;
        const psnr = 10 * Math.log10(255 ** 2 / (squaredError / (w * h * 3)));
        if (!(bytes.buffer instanceof ArrayBuffer) || !(pixels.buffer instanceof ArrayBuffer)) throw new Error('Expected owned codec outputs');
        self.postMessage({ bytes, pixels, ms, psnr, calls: gpu?.calls ?? 0, failure: gpu?.failure, threads: runtime.threads }, [bytes.buffer, pixels.buffer]);
      }
      finally { image.free(); }
    } finally { encoder?.free(); setCoolChicAccelerator(undefined, 0); gpu?.dispose(); }
  } catch (error) { self.postMessage({ error: String(error) }); }
};
