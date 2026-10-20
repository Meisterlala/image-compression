// Independent full GPU forward/backward parity against pinned Python fixtures.
import { chromium } from '@playwright/test';
import { createServer } from 'vite';
import { readFileSync } from 'node:fs';
import assert from 'node:assert/strict';
const server = await createServer({ server: { host: '127.0.0.1', port: 0, strictPort: false } });
await server.listen();
const browser = await chromium.launch({ headless: true,
  ...(process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE ? { executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE } : {}),
  args: ['--enable-unsafe-webgpu', '--use-angle=vulkan', '--enable-features=Vulkan', '--disable-vulkan-surface'],
});
try {
  const page = await browser.newPage();
  page.on('pageerror', error => console.error('PAGE ERROR', error));
  page.on('console', message => { if (message.type() === 'error' || message.type() === 'warning') console.error('GPU:', message.text()); });
  await page.goto(`http://127.0.0.1:${server.httpServer.address().port}`);
  const fixtures = JSON.parse(readFileSync('crates/coolchic/tests/fixtures/reference-graph.json', 'utf8')).graphs;
  const names = process.argv.filter(arg => arg.startsWith('--case=')).map(arg => arg.slice(7));
  for (const fixture of fixtures.filter(g => !names.length || names.includes(g.name))) {
    const result = await page.evaluate(async g => {
      const { CoolChicTensorGraph, initializeTensorGpu } = await import('/src/lib/coolchic-tensor-graph.ts');
      const { specForFixture, commonForSpec } = await import('/tests/helpers/coolchic-tensor-fixtures.ts');
      await initializeTensorGpu();
      const spec = specForFixture(g), graph = new CoolChicTensorGraph(spec, new Float32Array(g.target.data), commonForSpec(spec));
      try {
        const params = new Float32Array(g.groups.flat()), latents = new Float32Array(g.latents.flatMap(l => l.data)), noise = new Float32Array(g.noise);
        const packed = await graph.gradient(params, latents, noise, { mode: 2, temperature: Math.fround(0.35), lambda: Math.fround(0.001) }, true);
        let offset = 0;
        const take = n => { const values = [...packed.subarray(offset, offset + n)]; offset += n; return values; };
        const stats = take(3), image = take(g.decoded.data.length), rate = take(g.rate.length), gradients = take(params.length + latents.length);
        return { stats, image, rate, gradients, total: packed.length, consumed: offset };
      } finally { graph.dispose(); }
    }, fixture);
    assert.equal(result.total, result.consumed);
    const close = (actual, expected, label, absolute = 3e-5, relative = 1e-3) => {
      assert.equal(actual.length, expected.length, label);
      let maximum = 0;
      actual.forEach((a, i) => { maximum = Math.max(maximum, Math.abs(a - expected[i]));
        assert(Number.isFinite(a) && Math.abs(a - expected[i]) <= absolute + Math.abs(expected[i]) * relative,
          `${fixture.name} ${label}[${i}]: GPU=${a}, Python=${expected[i]}, tolerance=${absolute + Math.abs(expected[i]) * relative}`); });
      return maximum;
    };
    close([result.stats[0]], [fixture.loss], 'loss');
    const image = close(result.image, fixture.decoded.data, 'decoded', 3e-5, 1e-4);
    const rate = close(result.rate, fixture.rate, 'rate', 3e-4, 1e-4);
    const gradient = close(result.gradients, [...fixture.parameter_grad.flat(), ...fixture.latent_grad.flatMap(l => l.data)], 'gradient');
    console.log(`PASS ${fixture.name}: full GPU graph matches Python; max abs image=${image}, rate=${rate}, gradient=${gradient}`);
  }
} finally { await browser.close(); await server.close(); }
