// Production graph bridge, threaded WASM and CPU fallback checks.
// Requires npm run build:wasm; --require-gpu rejects missing hardware.
// --benchmark also times warmed full gradients at larger image sizes.
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
    worker.onmessage = ({ data }) => {
      clearTimeout(timer); worker.terminate();
      for (const key of ['bytes', 'pixels', 'result']) if (ArrayBuffer.isView(data[key])) data[key] = Array.from(data[key]);
      data.error ? reject(new Error(data.error)) : resolve(data);
    };
    worker.onerror = event => { clearTimeout(timer); worker.terminate(); reject(new Error(event.message)); };
    worker.postMessage(data);
  }), data);
  const verifyStream = (result, width = 32, height = 24) => {
    assert(result.bytes.length > 41);
    writeFileSync(join(dir, 'output.cool'), Buffer.from(result.bytes));
    execFileSync('cargo', ['run', '--quiet', '--manifest-path', 'crates/coolchic/Cargo.toml', '--example', 'rgba', '--', 'decode', join(dir, 'output.cool'), join(dir, 'output.rgba')]);
    const pixels = readFileSync(join(dir, 'output.rgba'));
    assert.equal(pixels.length, width * height * 4);
    assert.deepEqual([...pixels], result.pixels);
  };
  const cpu = await runWorker({ type: 'loader', gpu: false });
  verifyStream(cpu);
  assert(cpu.notices.some(n => n.includes('WebGPU training disabled')));
  console.log('PASS: explicit CPU opt-out trains a portable file');
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
  console.log('PASS: 4-thread WASM pointwise/spatial derivative parity');
  const threaded = await runWorker({ type: 'train', threads: 4, width: 256, height: 256 });
  assert.equal(threaded.threads, 4); verifyStream(threaded, 256, 256);
  console.log('PASS: parallel WASM training emits a native-decodable stream');
  const noGpu = await runWorker({ type: 'loader', gpu: true, unavailable: true, width: 128, height: 128 });
  verifyStream(noGpu, 128, 128);
  assert(noGpu.notices.some(n => n.includes('WebGPU training unavailable')));
  console.log('PASS: missing WebGPU falls back to CPU');
  const hasGpu = await page.evaluate(async () => !!(await navigator.gpu?.requestAdapter()));
  if (!hasGpu && process.argv.includes('--require-gpu')) throw new Error('WebGPU adapter required but unavailable');
  if (hasGpu) {
    const parity = await runWorker({ type: 'graph-benchmark', repeats: 2 });
    console.log('PASS: complete GPU gradients and RD loss match CPU WASM', parity);
    const trained = await runWorker({ type: 'train', gpu: true });
    assert(trained.calls > 0); assert(trained.evaluations > 0); assert(!trained.failure); verifyStream(trained);
    console.log(`PASS: GPU graph/WASM training emits a native-decodable stream (${trained.calls} gradients, ${trained.evaluations} RD evaluations)`);
    const automatic = await runWorker({ type: 'loader', gpu: true, width: 128, height: 128 });
    verifyStream(automatic, 128, 128);
    assert(automatic.notices.some(n => n.includes('WebGPU tensor graph:')));
    assert(!automatic.notices.some(n => n.includes('unavailable') || n.includes('fell back')));
    console.log('PASS: production loader selects the GPU graph');
    if (process.argv.includes('--benchmark')) {
      for (const width of [256,512,1024]) {
        const result = await runWorker({type:'graph-benchmark',width,height:width,threads:true,repeats:5});
        console.log(JSON.stringify({benchmark:`${width}x${width}, warmed complete gradients vs automatic CPU`,...result}));
      }
    }
  } else console.log('SKIP: WebGPU graph checks (no adapter)');
  await page.route('**/*', async route => {
    const response = await route.fetch();
    const headers = { ...response.headers() };
    delete headers['cross-origin-opener-policy']; delete headers['cross-origin-embedder-policy'];
    await route.fulfill({ response, headers });
  });
  await page.goto(`http://127.0.0.1:${server.httpServer.address().port}`);
  assert.equal(await page.evaluate(() => crossOriginIsolated), false);
  const fallback = await runWorker({ type: 'loader', gpu: true, width: 128, height: 128 });
  verifyStream(fallback, 128, 128);
  assert(fallback.notices.some(n => n.includes('WebGPU training unavailable')));
  console.log('PASS: missing isolation falls back to portable CPU WASM');
} finally { await browser.close(); await server.close(); rmSync(dir, { recursive: true, force: true }); }
