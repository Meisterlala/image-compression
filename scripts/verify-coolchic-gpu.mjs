// Real-browser shader, synchronous WASM bridge, training and portability checks.
// Requires generated Cool-chic WASM (npm run build:wasm).
// --require-gpu makes absence a failure; --benchmark also times real WASM steps.
import { chromium } from '@playwright/test';
import { createServer } from 'vite';
import { readFileSync, writeFileSync, mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { execFileSync } from 'node:child_process';
import assert from 'node:assert/strict';

const server = await createServer({ server: { host: '127.0.0.1', port: 0, strictPort: false } });
await server.listen();
const browser = await chromium.launch({ headless: true,
  ...(process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE ? { executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE } : {}),
  args: ['--enable-unsafe-webgpu', '--use-angle=vulkan', '--enable-features=Vulkan', '--disable-vulkan-surface'],
});
const dir = mkdtempSync(join(tmpdir(), 'coolchic-gpu-'));
try {
  const page = await browser.newPage();
  await page.goto(`http://127.0.0.1:${server.httpServer.address().port}`);
  const runWorker = data => page.evaluate(data => new Promise((resolve, reject) => {
    const worker = new Worker('/tests/helpers/coolchic-acceleration.worker.ts', { type: 'module' });
    const timer = setTimeout(() => { worker.terminate(); reject(new Error('Training test timed out')); }, 180000);
    worker.onmessage = ({ data }) => { clearTimeout(timer); worker.terminate(); data.error ? reject(new Error(data.error)) : resolve(data); };
    worker.onerror = event => { clearTimeout(timer); worker.terminate(); reject(new Error(event.message)); };
    worker.postMessage(data);
  }), data);
  const verifyStream = (result, width = 32, height = 24) => {
    assert(result.bytes.length > 41);
    writeFileSync(join(dir, 'output.cool'), Buffer.from(result.bytes));
    execFileSync('cargo', ['run', '--quiet', '--manifest-path', 'crates/coolchic/Cargo.toml', '--example', 'rgba', '--', 'decode', join(dir, 'output.cool'), join(dir, 'output.rgba')]);
    const pixels = readFileSync(join(dir, 'output.rgba'));
    assert.equal(pixels.length, width * height * 4);
    if (result.pixels) assert.deepEqual([...pixels], result.pixels);
  };
  const cpu = await runWorker({ type: 'loader', gpu: false });
  verifyStream(cpu);
  assert(cpu.notices.some(n => n.includes('WASM SIMD')));
  console.log('PASS: explicit CPU opt-out trains a portable file');
  // Force actual parallel kernels, not merely successful pool initialization.
  for (const [ci, co, h, w, k, residual, relu] of [[8,8,129,131,1,1,1], [3,3,129,131,3,1,1]]) {
    const params = [ci,co,h,w,k,residual,relu,1];
    const input = Array.from({length: ci*h*w}, (_,i) => ((i*7)%23-11)/31);
    const weight = Array.from({length: ci*co*k*k}, (_,i) => ((i*3)%19-9)/47);
    const bias = Array.from({length: co}, (_,i) => (i-1)/13);
    const gradient = Array.from({length: co*h*w}, (_,i) => ((i*5)%17-8)/29);
    const args = { type: 'kernel', params, input, weight, bias, gradient };
    const serial = await runWorker({ ...args, threads: false });
    const parallel = await runWorker({ ...args, threads: 4 });
    assert.equal(parallel.threads, 4); assert(!parallel.warning);
    assert.equal(serial.result.length, parallel.result.length);
    serial.result.forEach((v, i) => assert(Math.abs(v-parallel.result[i]) <= 3e-5 + Math.abs(v)*3e-4, `Threaded derivative ${i}`));
  }
  console.log('PASS: real 4-thread WASM pointwise/spatial forward and backward parity');
  const threaded = await runWorker({ type: 'train', threads: 4, width: 128, height: 128 });
  assert.equal(threaded.threads, 4); verifyStream(threaded, 128, 128);
  console.log('PASS: actual parallel WASM training emits a portable native-decodable stream');
  const noGpu = await runWorker({ type: 'loader', gpu: true, unavailable: true });
  verifyStream(noGpu);
  assert(noGpu.notices.some(n => n.includes('WebGPU training unavailable')));
  console.log('PASS: missing WebGPU falls back with isolation still enabled');
  const hasGpu = await page.evaluate(async () => !!(await navigator.gpu?.requestAdapter()));
  if (!hasGpu && process.argv.includes('--require-gpu')) throw new Error('WebGPU adapter required but unavailable');
  if (hasGpu) {
    const fixtures = JSON.parse(readFileSync('crates/coolchic/tests/fixtures/reference-operators.json', 'utf8')).convolution;
    const parity = await page.evaluate(async fixtures => {
      const { CoolChicGpu } = await import('/src/lib/coolchic-gpu.ts');
      const gpu = await CoolChicGpu.create();
      let checked = 0;
      const close = (actual, expected) => {
        if (actual.length !== expected.length) throw new Error('Length mismatch');
        actual.forEach((a, i) => { if (!Number.isFinite(a) || Math.abs(a - expected[i]) > 3e-5 + Math.abs(expected[i]) * 3e-4) throw new Error(`Mismatch at ${i}: ${a} vs ${expected[i]}`); });
      };
      try {
        for (const f of fixtures) {
          const l = f.layer, [, h, w] = f.input.shape;
          const params = new Uint32Array([l.input_channels, l.output_channels, h, w, l.kernel, +l.residual, +l.relu, 0]);
          const data = [...f.input.data, ...l.weight, ...l.bias];
          close(await gpu.convolution(params, new Float32Array(data)), f.output.data);
          params[7] = 1;
          close(await gpu.convolution(params, new Float32Array([...data, ...f.upstream.data])), [...f.input_grad.data, ...f.parameter_grad.weight, ...f.parameter_grad.bias]);
          checked++;
        }
        // Multi-chunk reduction + dispatch Y dimension, non-square images,
        // negative values, degenerate axes, and padding larger than the input.
        for (const [ci, co, h, w, k, residual, relu] of [[2,3,17,19,3,0,1], [2,2,1,5,5,1,1], [2,2,5,1,3,1,0], [2,3,129,131,1,0,0]]) {
          const n = h*w, nw = ci*co*k*k;
          const input = Float32Array.from({length: ci*n}, (_,i) => ((i*7)%23-11)/31);
          const weights = Float32Array.from({length: nw}, (_,i) => ((i*3)%19-9)/47);
          const bias = Float32Array.from({length: co}, (_,i) => (i-1)/13);
          const grad = Float32Array.from({length: co*n}, (_,i) => ((i*5)%17-8)/29);
          const output = new Float32Array(co*n), dx = new Float32Array(ci*n), dw = new Float32Array(nw), db = new Float32Array(co);
          const source = (c,y,x,ky,kx) => (c*h + Math.max(0,Math.min(h-1,y+ky-(k>>1))))*w + Math.max(0,Math.min(w-1,x+kx-(k>>1)));
          const wi = (o,c,ky,kx) => ((o*ci+c)*k+ky)*k+kx;
          for (let o=0;o<co;o++) for(let y=0;y<h;y++) for(let x=0;x<w;x++) {
            const i=(o*h+y)*w+x;
            let v=0;
            for(let c=0;c<ci;c++) for(let ky=0;ky<k;ky++) for(let kx=0;kx<k;kx++) v=Math.fround(v+Math.fround(input[source(c,y,x,ky,kx)]*weights[wi(o,c,ky,kx)]));
            v=Math.fround(v+bias[o]); if(residual) v=Math.fround(v+input[i]);
            output[i]=relu?Math.max(0,v):v;
            const g=relu && output[i]<=0 ? 0 : grad[i]; db[o]=Math.fround(db[o]+g);
            if(residual) dx[i]=Math.fround(dx[i]+g);
            for(let c=0;c<ci;c++) for(let ky=0;ky<k;ky++) for(let kx=0;kx<k;kx++) {
              const ii=source(c,y,x,ky,kx), j=wi(o,c,ky,kx);
              dx[ii]=Math.fround(dx[ii]+Math.fround(g*weights[j])); dw[j]=Math.fround(dw[j]+Math.fround(g*input[ii]));
            }
          }
          const params=new Uint32Array([ci,co,h,w,k,residual,relu,0]);
          close(await gpu.convolution(params,new Float32Array([...input,...weights,...bias])), output);
          params[7]=1;
          close(await gpu.convolution(params,new Float32Array([...input,...weights,...bias,...grad])), [...dx,...dw,...db]);
          checked++;
        }
        return checked;
      } finally { gpu.dispose(); }
    }, fixtures);
    console.log(`PASS: ${parity} forward/backward GPU cases match independent PyTorch/scalar derivatives`);
    const bridge = await runWorker({ type: 'bridge' });
    assert.deepEqual(bridge.result, [7, 10]); assert.equal(bridge.calls, 1); assert(!bridge.failure);
    console.log('PASS: dedicated-worker shared-memory bridge');
    const trained = await runWorker({ type: 'train', gpu: true, threshold: 50_000 });
    assert(trained.calls > 0); assert(!trained.failure); verifyStream(trained);
    console.log(`PASS: GPU/WASM-trained stream independently decoded by native Rust (${trained.calls} GPU jobs, ${Math.round(trained.ms)} ms)`);
    if (process.argv.includes('--benchmark')) {
      for (const width of [128, 384, 512]) {
        const results = [];
        for (const [gpu, threads] of [[false,false], [true,false], [false,4], [false,8], [true,8]]) results.push(await runWorker({ type: 'train', gpu, threads, calibrate: true, threshold: 8_000_000, width, height: width, steps: 10 }));
        console.log(JSON.stringify({ benchmark: `${width}x${width}, init + 10 real WASM training steps`, cpu: results[0], gpu: results[1], threads4: results[2], threads8: results[3], gpuThreads8: results[4] }));
      }
    }
  } else console.log('SKIP: WebGPU parity (no adapter); fallback checks still run');
  // Missing COOP/COEP must fall back without SharedArrayBuffer or a GPU worker.
  await page.route('**/*', async route => {
    const response = await route.fetch();
    const headers = { ...response.headers() };
    delete headers['cross-origin-opener-policy']; delete headers['cross-origin-embedder-policy'];
    await route.fulfill({ response, headers });
  });
  await page.goto(`http://127.0.0.1:${server.httpServer.address().port}`);
  assert.equal(await page.evaluate(() => crossOriginIsolated), false);
  const fallback = await runWorker({ type: 'loader', gpu: true });
  verifyStream(fallback);
  assert(fallback.notices.some(n => n.includes('WebGPU training unavailable')));
  console.log('PASS: default-on GPU falls back to portable CPU WASM without isolation');
} finally { await browser.close(); await server.close(); rmSync(dir, { recursive: true, force: true }); }
