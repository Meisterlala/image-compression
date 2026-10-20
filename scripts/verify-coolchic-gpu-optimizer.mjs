// Exact SOAP algorithms and resumable state, independent of the neural graph.
import { chromium } from '@playwright/test';
import { createServer } from 'vite';
import assert from 'node:assert/strict';
const server = await createServer({ server: { host: '127.0.0.1', port: 0, strictPort: false } });
await server.listen();
const browser = await chromium.launch({ headless: true,
  ...(process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE ? { executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE } : {}),
  args: ['--enable-unsafe-webgpu', '--use-angle=vulkan', '--enable-features=Vulkan', '--disable-vulkan-surface'],
});
try {
  const page = await browser.newPage();
  page.on('console', message => { if (message.type() === 'error') console.error(message.text()); });
  await page.goto(`http://127.0.0.1:${server.httpServer.address().port}`);
  for (const [shape, latent] of [[[3,3,3,3],false], [[8,4],false], [[16],false], [[1,1,33,31],true]]) {
    const result = await page.evaluate(async ({shape,latent}) => {
      const wasm = await import('/src/wasm-coolchic/coolchic.js'); await wasm.default();
      const { GpuSoap } = await import('/src/lib/coolchic-gpu-optimizer.ts');
      const adapter = await navigator.gpu.requestAdapter(); if (!adapter) throw new Error('A GPU adapter is required');
      const device = await adapter.requestDevice();
      let failure;
      device.addEventListener('uncapturederror', e => { failure ??= e.error.message; });
      const cpu = new wasm.CoolChicSoapProblem(new Uint32Array(shape),latent);
      const layout = JSON.parse(cpu.descriptor());
      const gpu = new GpuSoap(device,layout,wasm.coolChicSoapBasis); gpu.upload(cpu.state());
      const buffer = device.createBuffer({size:layout.parameters*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
      let maxError=0, firstError=0;
      try {
        for (let i=0;i<32;i++) {
          const gradient = Float32Array.from({length:layout.parameters},(_,j) => Math.sin((j+1)*(i+3)*1.618)/13 + Math.cos((j+3)*(i+1)*0.77)/17);
          // Exercise scheduler changes as well as refreshes at 10, 20 and 30.
          const lr = i<17 ? Math.fround(0.01) : Math.fround(0.0001);
          device.queue.writeBuffer(buffer,0,gradient);
          await gpu.update(buffer,i-1,lr); cpu.update(gradient,lr);
          const actual = await gpu.download(), expected = cpu.state();
          if (failure) throw new Error(failure);
          for (let j=0;j<layout.parameters;j++) {
            const error = Math.abs(actual[j]-expected[j]); maxError=Math.max(maxError,error);
            if (i===0) firstError=Math.max(firstError,error);
            if (!Number.isFinite(actual[j]) || error>0.001+Math.abs(expected[j])*0.002) throw new Error(`step ${i}, parameter ${j}: GPU ${actual[j]}, CPU ${expected[j]}`);
          }
          // A rank-deficient Gram matrix has no unique null-space basis;
          // align that initial basis before comparing subsequent arithmetic.
          if (i===0 && !latent && shape.length===2) cpu.restore(actual,0);
          // The production bridge restores this complete state at each batch.
          if (i%8===7) gpu.upload(actual);
        }
        await device.queue.onSubmittedWorkDone(); if (failure) throw new Error(failure);
        return {maxError,firstError};
      } finally {gpu.dispose();buffer.destroy();cpu.free();device.destroy();}
    }, {shape,latent});
    assert.equal(result.firstError,0,'Reference SOAP skips its first update');
    console.log(`PASS ${latent?'latent':'network'} ${shape.join('x')}: 32 updates, three QR refreshes, state roundtrips, max parameter error ${result.maxError}`);
  }
} finally {await browser.close();await server.close();}
