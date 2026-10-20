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
      let trainingMs: number | undefined;
      const codec = await loadCoolChic({ ...defaultSettings, coolchicPoint: data.point ?? 'lop', coolchicGpu: data.gpu, coolchicIterations: data.iterations ?? 100 }, (_text, info) => {
        if (info?.stage === 'quantizing' && trainingMs === undefined) trainingMs = info.elapsedMs;
      });
      const start = performance.now();
      const bytes = await codec.encode(new ImageData(new Uint8ClampedArray(rgba), w, h));
      const ms = performance.now() - start, decoded = codec.decode(bytes.buffer);
      let squaredError = 0;
      for (let i = 0; i < rgba.length; i++) if (i % 4 !== 3) squaredError += (rgba[i] - decoded.data[i]) ** 2;
      const psnr = 10 * Math.log10(255 ** 2 / (squaredError / (w * h * 3)));
      if (!(bytes.buffer instanceof ArrayBuffer)) throw new Error('Expected owned encoded bytes');
      self.postMessage({ bytes, notices: codec.notices, ms, psnr, trainingMs }, [bytes.buffer]);
      return;
    }
    const runtime = await loadCoolChicRuntime(!!data.threads, typeof data.threads === 'number' ? data.threads : undefined);
    const { CoolChicEncoder, setCoolChicAccelerator, coolChicCpuConvolution } = runtime.bindings;
    if (data.type === 'graph-benchmark') {
      const { CoolChicTensorGraph, initializeTensorGpu } = await import('../../src/lib/coolchic-tensor-graph');
      const problem = new runtime.bindings.CoolChicGraphProblem(rgba, w, h, data.point ?? 'lop');
      await initializeTensorGpu();
      const spec = JSON.parse(problem.descriptor());
      const graph = new CoolChicTensorGraph(spec, problem.target(), problem.common());
      try {
        const parameters = problem.parameters(), latents = problem.latents(), noise = problem.noise();
        const options = { mode: 2, temperature: Math.fround(0.35), lambda: Math.fround(0.001) };
        await graph.gradient(parameters, latents, noise, options);
        const gpu = await graph.gradient(parameters, latents, noise, options);
        const cpu = problem.cpuGradient(2, 0.35, 0.001);
        let maxError = 0;
        for (let i = 0; i < cpu.length; i++) {
          maxError = Math.max(maxError, Math.abs(cpu[i] - gpu[i]));
          if (Math.abs(cpu[i] - gpu[i]) > 3e-5 + Math.abs(cpu[i]) * 1e-3) throw new Error(`Graph gradient mismatch ${i}: GPU ${gpu[i]}, CPU ${cpu[i]}`);
        }
        if (data.profile) {
          const tf = await import('@tensorflow/tfjs-core');
          const profile = await tf.profile(async () => { await graph.gradient(parameters, latents, noise, options); return tf.scalar(0); });
          const result = profile.result as import('@tensorflow/tfjs-core').Tensor;
          result.dispose();
          const totals = new Map<string, { count: number; ms: number }>();
          for (const kernel of profile.kernels) {
            const value = totals.get(kernel.name) ?? { count: 0, ms: 0 };
            value.count++; value.ms += Number(await kernel.kernelTimeMs); totals.set(kernel.name, value);
          }
          self.postMessage({ profile: [...totals].sort((a, b) => b[1].ms - a[1].ms), kernels: profile.kernels.length });
          return;
        }
        const iterations = data.repeats ?? 3;
        let start = performance.now();
        for (let i = 0; i < iterations; i++) problem.cpuGradient(2, 0.35, 0.001);
        const cpuMs = (performance.now() - start) / iterations;
        start = performance.now();
        for (let i = 0; i < iterations; i++) await graph.gradient(parameters, latents, noise, options);
        const gpuMs = (performance.now() - start) / iterations;
        self.postMessage({ cpuMs, gpuMs, speedup: cpuMs / gpuMs, maxError, threads: runtime.threads });
      } finally { graph.dispose(); problem.free(); }
      return;
    }
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
