import { test, expect, type Page } from '@playwright/test';

test.beforeEach(({ page }) => {
  page.on('pageerror', error => { throw error; });
});

async function upload(page: Page) {
  const png = await page.evaluate(() => {
    const canvas = document.createElement('canvas'); canvas.width = 96; canvas.height = 64;
    const ctx = canvas.getContext('2d')!;
    const gradient = ctx.createLinearGradient(0, 0, 96, 64);
    gradient.addColorStop(0, '#cb381d'); gradient.addColorStop(0.5, '#32b58b'); gradient.addColorStop(1, '#395ad5');
    ctx.fillStyle = gradient; ctx.fillRect(0, 0, 96, 64);
    ctx.fillStyle = '#f3d14b'; ctx.fillRect(11, 8, 31, 29);
    return canvas.toDataURL('image/png').split(',')[1]!;
  });
  await page.getByLabel('Choose image file', { exact: true }).setInputFiles({ name: 'gradient.png', mimeType: 'image/png', buffer: Buffer.from(png, 'base64') });
  await expect(page.getByLabel('Original image', { exact: true })).toHaveAttribute('width', '96');
  return Buffer.byteLength(png, 'base64');
}

function pane(page: Page, format: string) {
  return page.locator(`[data-codec="${format}"]`);
}

function resultRow(page: Page, format: string) {
  return page.locator(`[data-result-format="${format}"]`);
}

async function settings(page: Page, format: string) {
  const labels: Record<string, string> = { jpegai: 'JPEG AI', jpeg: 'JPEG', avif: 'AVIF', jxl: 'JPEG XL', coolchic: 'Cool-chic 5.0' };
  await page.getByRole('tab', { name: labels[format], exact: true }).click();
}

async function run(page: Page, format: string) {
  await settings(page, format);
  const panel = pane(page, format);
  await panel.getByRole('button', { name: /^Encode(?: again)?$/ }).click();
  await expect(panel).toHaveAttribute('data-phase', /done|error/, { timeout: 150_000 });
  await expect(panel.getByRole('alert')).toHaveCount(0);
  await expect(resultRow(page, format).locator('[data-metric="encode"]')).toHaveAttribute('data-ms', /[0-9]/);
}

test('codec tabs retain five real WASM outputs with one visible panel', async ({ page }) => {
  const requests: string[] = [];
  page.on('request', request => { expect(request.method()).toBe('GET'); requests.push(request.url()); });
  await page.goto('/');
  await expect(page.getByRole('tab')).toHaveCount(5);
  await expect(page.getByRole('tabpanel')).toHaveCount(1);
  await expect(page.locator('.codec-pane:visible')).toHaveCount(1);
  await expect(page.locator('aside, dialog')).toHaveCount(0);
  await expect(page.getByRole('table', { name: 'Image format comparison', exact: true })).toBeVisible();
  await expect(page.locator('tbody tr')).toHaveCount(6);
  for (const format of ['jpegai', 'jpeg', 'avif', 'jxl', 'coolchic']) {
    await expect(resultRow(page, format).locator('td')).toHaveText(Array(6).fill(''));
  }
  await expect(page.locator('.page-header a')).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Select image', exact: true })).toBeVisible();
  await expect(page.locator('.file-description')).toHaveText('Your image is processed only locally in your browser.');
  const originalBytes = await upload(page);
  await expect(page.getByLabel('Original image', { exact: true })).toBeVisible();
  await expect(page.locator('.comparison canvas')).toHaveCount(0);
  await expect(page.getByRole('button', { name: /Encode all|comparison JSON/i })).toHaveCount(0);
  for (const [id, label] of [['jpegai', 'JPEG AI'], ['jpeg', 'JPEG'], ['avif', 'AVIF'], ['jxl', 'JPEG XL'], ['coolchic', 'Cool-chic 5.0']]) {
    await run(page, id!);
    const panel = pane(page, id!);
    await expect(panel).toHaveAttribute('data-phase', 'done', { timeout: 150_000 });
    await expect(panel.getByRole('alert')).toHaveCount(0);
    const event = page.waitForEvent('download');
    await panel.getByRole('button', { name: `Download ${label}`, exact: true }).click();
    const download = await event;
    expect(download.suggestedFilename()).toMatch(/\.(jai|jpg|avif|jxl|cool)$/);
    const stream = await download.createReadStream();
    const chunks: Buffer[] = []; for await (const chunk of stream!) chunks.push(chunk);
    const encoded = Buffer.concat(chunks);
    expect(encoded.length).toBeGreaterThan(16);
    const reduction = (1 - encoded.length / originalBytes) * 100;
    const size = resultRow(page, id!).locator('[data-metric="size"]');
    await expect(size).toHaveAttribute('data-percent', String(reduction));
    await expect(size).not.toContainText('%');
    await size.getByRole('button').hover();
    await expect(page.getByRole('tooltip')).toContainText(`${Math.abs(reduction).toFixed(1)}% ${reduction >= 0 ? 'smaller' : 'larger'}`);
    await page.keyboard.press('Escape');
    expect(Number(await resultRow(page, id!).locator('[data-metric="encode"]').getAttribute('data-ms'))).toBeGreaterThan(0);
    expect(Number(await resultRow(page, id!).locator('[data-metric="decode"]').getAttribute('data-ms'))).toBeGreaterThan(0);
    if (id === 'jpeg') expect(encoded.subarray(0, 2).toString('hex')).toBe('ffd8');
    if (id === 'avif') expect(encoded.subarray(4, 8).toString()).toBe('ftyp');
    const decoded = page.getByLabel(`Right ${label} image`, { exact: true });
    await expect(decoded).toBeVisible();
    await expect(decoded).toHaveAttribute('width', '96');
    await expect(decoded).toHaveAttribute('height', '64');
  }
  for (const metric of ['size', 'encode', 'decode', 'psnr', 'ssim', 'bpp']) {
    const cells = await page.locator(`tbody tr:not(.original-row) [data-metric="${metric}"]`).evaluateAll(elements => elements.map(element => ({
      score: Number((element as HTMLElement).dataset.score),
      rank: (element as HTMLElement).dataset.rank,
      color: getComputedStyle(element.querySelector('button') ?? element).color,
    })));
    const scores = cells.map(cell => cell.score);
    const low = Math.min(...scores), high = Math.max(...scores);
    const higher = metric === 'psnr' || metric === 'ssim';
    for (const cell of cells) {
      const expected = low === high ? undefined : cell.score === (higher ? high : low) ? 'best' : cell.score === (higher ? low : high) ? 'worst' : undefined;
      expect(cell.rank).toBe(expected);
      const [red, green] = cell.color.match(/\d+/g)!.map(Number);
      if (expected === 'best') expect(green!).toBeGreaterThan(red!);
      if (expected === 'worst') expect(red!).toBeGreaterThan(green!);
    }
  }
  await expect(resultRow(page, 'original').locator('[data-rank]')).toHaveCount(0);
  await expect(page.locator('.comparison:visible')).toHaveCount(1);
  await expect(page.locator('[role="tabpanel"]')).toHaveCount(5);
  await settings(page, 'jpegai');
  await expect(page.getByLabel('Right JPEG AI image', { exact: true })).toBeVisible();
  expect(requests.every(url => new URL(url).hostname === 'localhost' || url.startsWith('blob:'))).toBeTruthy();
});

test('four compact model cards select and download into cache without file exports', async ({ page }) => {
  let fileDownloads = 0;
  page.on('download', () => fileDownloads++);
  await page.goto('/');
  const models = pane(page, 'jpegai').getByRole('group', { name: 'JPEG AI models', exact: true });
  await expect(models.locator('.model-card')).toHaveCount(4);
  const card = models.locator('[data-model="0"]');
  await card.locator('.model-select').click();
  await expect(card).toHaveAttribute('data-ready', 'true', { timeout: 60_000 });
  await expect(card.locator('.model-select')).toHaveAttribute('aria-pressed', 'true');
  await card.getByRole('button', { name: 'Remove', exact: true }).click();
  await expect(card).toHaveAttribute('data-ready', 'false');
  await card.getByRole('button', { name: 'Download', exact: true }).click();
  await expect(card).toHaveAttribute('data-ready', 'true');
  await models.locator('[data-model="1"] .model-select').click();
  await expect(models.locator('[data-model="1"]')).toHaveAttribute('data-ready', 'true');
  await card.locator('.model-select').click();
  await expect(card.locator('.model-select')).toHaveAttribute('aria-pressed', 'true');
  expect(fileDownloads).toBe(0);
  await expect(page.locator('details, dialog')).toHaveCount(0);
  await expect(models.locator('select')).toHaveCount(0);
  await expect(models).not.toContainText(/SHA|\.zjb|Not downloaded/);
  await upload(page);
  await page.route('**/models/*.zjb*', route => route.abort());
  await run(page, 'jpegai');
});

test('BOP/HOP selector explains the trade-off and uses real independent models and cache entries', async ({ page }) => {
  await page.goto('/'); await upload(page);
  const panel = pane(page, 'jpegai');
  const point = panel.getByLabel('Operating point', { exact: true });
  const help = panel.getByRole('button', { name: 'About Operating point', exact: true });
  await expect(point).toHaveValue('bop');
  await help.hover();
  await expect(page.getByRole('tooltip')).toContainText('Base Operation Point');
  await expect(page.getByRole('tooltip')).toContainText('High Operation Point');
  await expect(page.getByRole('tooltip')).toContainText('slower decoding');
  await page.keyboard.press('Escape');
  await run(page, 'jpegai');
  await expect(resultRow(page, 'jpegai')).toHaveAttribute('data-operating-point', 'bop');
  const bop = await page.getByLabel('Right JPEG AI image', { exact: true }).evaluate(canvas => (canvas as HTMLCanvasElement).toDataURL());
  await point.selectOption('hop');
  await expect(panel.locator('.model-control')).toHaveAttribute('data-operating-point', 'hop');
  await expect(panel.locator('[data-model="1"]')).toHaveAttribute('data-ready', 'false');
  await expect(panel.locator('.changed-note')).toBeVisible();
  await expect(resultRow(page, 'jpegai')).toHaveAttribute('data-operating-point', 'bop');
  await run(page, 'jpegai');
  await expect(resultRow(page, 'jpegai')).toHaveAttribute('data-operating-point', 'hop');
  await expect(resultRow(page, 'jpegai').locator('.dimensions')).toContainText('HOP');
  const hop = await page.getByLabel('Right JPEG AI image', { exact: true }).evaluate(canvas => (canvas as HTMLCanvasElement).toDataURL());
  expect(hop).not.toBe(bop);
  await point.selectOption('bop');
  await expect(panel.locator('[data-model="1"]')).toHaveAttribute('data-ready', 'true');
  await expect(resultRow(page, 'jpegai')).toHaveAttribute('data-operating-point', 'hop');
  const download = page.waitForEvent('download');
  await panel.getByRole('button', { name: 'Download JPEG AI', exact: true }).click();
  expect((await download).suggestedFilename()).toContain('-hop-'); // Saved output, not current control.
  await page.route('**/models/*.zjb*', route => route.abort());
  await run(page, 'jpegai'); // BOP cache is intact after HOP use.
  expect(await page.getByLabel('Right JPEG AI image', { exact: true }).evaluate(canvas => (canvas as HTMLCanvasElement).toDataURL())).toBe(bop);
  await point.selectOption('hop');
  await expect(panel.locator('[data-model="1"]')).toHaveAttribute('data-ready', 'true');
  await run(page, 'jpegai'); // HOP cache is also independent and reusable.
  expect(await page.getByLabel('Right JPEG AI image', { exact: true }).evaluate(canvas => (canvas as HTMLCanvasElement).toDataURL())).toBe(hop);
  await page.setViewportSize({ width: 390, height: 844 });
  await help.focus();
  await expect(page.getByRole('tooltip')).toBeInViewport();
  await page.keyboard.press('Escape');
  await help.click();
  await expect(page.getByRole('tooltip')).toContainText('higher memory');
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBeTruthy();
});

test('HOP target-bpp search loads only HOP bundles and decodes its actual output', async ({ page }) => {
  const weights: string[] = [];
  page.on('request', request => { if (request.url().includes('.zjb')) weights.push(request.url()); });
  await page.goto('/'); await upload(page);
  await pane(page, 'jpegai').getByLabel('Operating point', { exact: true }).selectOption('hop');
  await pane(page, 'jpegai').getByLabel('Model selection', { exact: true }).selectOption('target');
  await run(page, 'jpegai');
  await expect(resultRow(page, 'jpegai')).toHaveAttribute('data-operating-point', 'hop');
  await expect(page.getByLabel('Right JPEG AI image', { exact: true })).toHaveAttribute('width', '96');
  expect(weights).toHaveLength(4);
  expect(weights.every(url => /\/hop-m[0-3]\.zjb/.test(url))).toBeTruthy();
});

test('switching operating point cancels a model download without leaving Encode disabled', async ({ page }) => {
  let release!: () => void;
  const pending = new Promise<void>(resolve => { release = resolve; });
  await page.route('**/models/bop-m0.zjb*', async route => { await pending; await route.abort().catch(() => {}); });
  await page.goto('/'); await upload(page);
  const request = page.waitForRequest('**/models/bop-m0.zjb*');
  await pane(page, 'jpegai').locator('[data-model="0"] .model-select').click();
  await request;
  await pane(page, 'jpegai').getByLabel('Operating point', { exact: true }).selectOption('hop');
  await expect(pane(page, 'jpegai').locator('.model-control')).toHaveAttribute('data-operating-point', 'hop');
  await expect(pane(page, 'jpegai').getByRole('button', { name: 'Encode', exact: true })).toBeEnabled();
  release();
  await expect(pane(page, 'jpegai').getByRole('alert')).toHaveCount(0);
});

test('lossless modes really reconstruct identical working pixels', async ({ page }) => {
  await page.goto('/'); await upload(page);
  for (const format of ['avif', 'jxl']) {
    await settings(page, format);
    const panel = pane(page, format);
    await panel.getByRole('checkbox', { name: 'Lossless' }).check();
    await run(page, format);
    await expect(resultRow(page, format).locator('[data-metric="ssim"]')).toHaveText('1.0000');
    await expect(resultRow(page, format).locator('[data-metric="psnr"]')).toContainText('∞');
  }
});

test('target-bpp search produces a real JPEG AI result', async ({ page }) => {
  await page.goto('/'); await upload(page);
  await settings(page, 'jpegai');
  await pane(page, 'jpegai').getByLabel('Model selection', { exact: true }).selectOption('target');
  await run(page, 'jpegai');
  await expect(pane(page, 'jpegai').getByLabel('Model selection', { exact: true })).toHaveValue('target');
  await expect(page.getByLabel('Right JPEG AI image', { exact: true })).toBeVisible();
});

test('cancel stops work and allows another run', async ({ page }) => {
  await page.goto('/'); await upload(page);
  await page.route('**/models/*.zjb*', async route => { await new Promise(resolve => setTimeout(resolve, 2000)); await route.abort(); });
  await pane(page, 'jpegai').getByRole('button', { name: 'Encode', exact: true }).click();
  await pane(page, 'jpegai').getByRole('button', { name: 'Cancel encoding', exact: true }).click();
  await expect(pane(page, 'jpegai')).toHaveAttribute('data-phase', 'cancelled');
  await page.unroute('**/models/*.zjb*');
  await run(page, 'jpeg');
});

test('single-threaded WASM also works on static hosts without isolation headers', async ({ page }) => {
  await page.route('**/*', async route => {
    const response = await route.fetch();
    const headers = { ...response.headers() };
    delete headers['cross-origin-opener-policy'];
    delete headers['cross-origin-embedder-policy'];
    await route.fulfill({ response, headers });
  });
  await page.goto('/');
  expect(await page.evaluate(() => crossOriginIsolated)).toBe(false);
  await upload(page);
  await run(page, 'jpeg');
});

test('bad files give a visible error; mobile layout has no horizontal page overflow', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/');
  await page.getByLabel('Choose image file', { exact: true }).setInputFiles({ name: 'broken.png', mimeType: 'image/png', buffer: Buffer.from('not an image') });
  await expect(page.getByRole('alert')).toBeVisible();
  await upload(page);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBeTruthy();
  await settings(page, 'jpegai');
  await expect(pane(page, 'jpegai').getByRole('group', { name: 'JPEG AI models', exact: true })).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBeTruthy();
});

test('pane settings and results are independent, with no history after re-encoding', async ({ page }) => {
  await page.goto('/'); await upload(page);
  for (const [id, value] of [['jpeg', '34'], ['avif', '82']]) {
    await settings(page, id!);
    await pane(page, id!).locator('input[type="range"]').first().evaluate((input, value) => {
      (input as HTMLInputElement).value = value;
      input.dispatchEvent(new Event('input', { bubbles: true }));
    }, value!);
    await run(page, id!);
  }
  const avifTime = await resultRow(page, 'avif').locator('[data-metric="encode"]').getAttribute('data-ms');
  await expect(pane(page, 'jpeg').locator('#jpeg-quality')).toHaveValue('34');
  await expect(pane(page, 'avif').locator('#avif-quality')).toHaveValue('82');
  await settings(page, 'jpeg');
  await pane(page, 'jpeg').locator('#jpeg-quality').evaluate(input => {
    (input as HTMLInputElement).value = '50';
    input.dispatchEvent(new Event('input', { bubbles: true }));
  });
  await expect(pane(page, 'jpeg').locator('.changed-note')).toContainText('Settings or image size changed');
  await expect(pane(page, 'avif').locator('.changed-note')).toHaveCount(0);
  await run(page, 'jpeg');
  await expect(pane(page, 'jpeg').locator('#jpeg-quality')).toHaveValue('50');
  await expect(pane(page, 'avif').locator('#avif-quality')).toHaveValue('82');
  await expect(resultRow(page, 'avif').locator('[data-metric="encode"]')).toHaveAttribute('data-ms', avifTime!);
  await expect(page.locator('[data-metric="encode"][data-ms]')).toHaveCount(2);
  await expect(page.locator('aside, dialog')).toHaveCount(0);
  const ids = await page.locator('[id]').evaluateAll(elements => elements.map(element => element.id));
  expect(new Set(ids).size).toBe(ids.length);
  await upload(page);
  await expect(page.locator('[data-metric="encode"][data-ms]')).toHaveCount(0);
});

test('queued panes can be cancelled independently without stopping another codec', async ({ page }) => {
  await page.goto('/'); await upload(page);
  await page.route('**/models/*.zjb*', async route => {
    await new Promise(resolve => setTimeout(resolve, 2500));
    await route.abort();
  });
  await pane(page, 'jpegai').getByRole('button', { name: 'Encode', exact: true }).click();
  await settings(page, 'jpeg');
  await pane(page, 'jpeg').getByRole('button', { name: 'Encode', exact: true }).click();
  await settings(page, 'jxl');
  await pane(page, 'jxl').getByRole('button', { name: 'Encode', exact: true }).click();
  await expect(pane(page, 'jpeg')).toHaveAttribute('data-phase', 'queued');
  await pane(page, 'jxl').getByRole('button', { name: 'Cancel queued job' }).click();
  await expect(pane(page, 'jxl')).toHaveAttribute('data-phase', 'cancelled');
  await settings(page, 'jpegai');
  await pane(page, 'jpegai').getByRole('button', { name: 'Cancel encoding' }).click();
  await expect(pane(page, 'jpeg')).toHaveAttribute('data-phase', 'done');
  await expect(pane(page, 'jpegai')).toHaveAttribute('data-phase', 'cancelled');
  await expect(resultRow(page, 'jxl').locator('[data-metric="encode"]')).not.toHaveAttribute('data-ms');
});

test('primary action stays by the output and tabs preserve the result without premature previews', async ({ page }) => {
  await page.goto('/'); await upload(page);
  const uploadButton = await page.getByRole('button', { name: 'Select image', exact: true }).boundingBox();
  const encodeButton = await pane(page, 'jpegai').getByRole('button', { name: 'Encode', exact: true }).boundingBox();
  expect(encodeButton!.height).toBeGreaterThan(uploadButton!.height);
  await expect(page.getByRole('button', { name: /comparison JSON|Encode all/i })).toHaveCount(0);
  await expect(page.getByText('Not encoded yet.', { exact: true })).toHaveCount(0);
  await expect(page.locator('.comparison canvas')).toHaveCount(0);
  await page.getByRole('tab', { name: 'JPEG AI', exact: true }).focus();
  await page.keyboard.press('ArrowRight');
  await expect(page.getByRole('tab', { name: 'JPEG', exact: true })).toHaveAttribute('aria-selected', 'true');
  await run(page, 'jpeg');
  await page.getByLabel('Image comparison zoom', { exact: true }).selectOption('2');
  await settings(page, 'avif');
  await expect(pane(page, 'avif').locator('.comparison')).toHaveCount(0);
  await expect(page.getByLabel('Original image', { exact: true })).toBeVisible();
  await settings(page, 'jpeg');
  await expect(page.getByLabel('Image comparison zoom', { exact: true })).toHaveValue('2');
  await expect(page.getByLabel('Right JPEG image', { exact: true })).toBeVisible();
});

test('floating comparison has independent selectors and follows tabs and completed encodes', async ({ page }) => {
  await page.goto('/'); await upload(page); await run(page, 'jpeg');
  const card = page.getByRole('region', { name: 'Image comparison', exact: true });
  const left = card.getByLabel('Left image', { exact: true });
  const right = card.getByLabel('Right image', { exact: true });
  await expect(left).toHaveValue('source');
  await expect(right).toHaveValue('jpeg');
  await expect(card.locator('.image-choices')).not.toContainText('×');
  await expect(card.locator('.viewer-toolbar')).not.toContainText(/\d+\s*×\s*\d+/);
  await expect(right.locator('option[value="avif"]')).toHaveJSProperty('disabled', true);
  await expect(pane(page, 'jpeg').getByRole('button', { name: 'Download JPEG', exact: true })).toBeVisible();
  await expect(pane(page, 'jpeg').getByRole('button', { name: 'Preview PNG', exact: true })).toBeVisible();
  await expect(card.getByRole('button', { name: /Download|Preview PNG/ })).toHaveCount(0);
  await expect(page.locator('.codec-pane .comparison')).toHaveCount(0);
  const position = await page.evaluate(() => ({
    encoder: document.querySelector('[data-codec="jpeg"]')!.getBoundingClientRect().bottom,
    viewerTop: document.querySelector('.image-comparison-card')!.getBoundingClientRect().top,
    viewerBottom: document.querySelector('.image-comparison-card')!.getBoundingClientRect().bottom,
    table: document.querySelector('.comparison-card')!.getBoundingClientRect().top,
  }));
  expect(position.viewerTop).toBeGreaterThan(position.encoder);
  expect(position.table).toBeGreaterThan(position.viewerBottom);
  await run(page, 'avif');
  await expect(right).toHaveValue('avif');
  await left.selectOption('jpeg'); await right.selectOption('source');
  await expect(page.getByLabel('Left JPEG image', { exact: true })).toBeVisible();
  await expect(page.getByLabel('Right Original image', { exact: true })).toBeVisible();
  const original = await page.getByLabel('Original image', { exact: true }).evaluate(canvas => (canvas as HTMLCanvasElement).toDataURL());
  expect(await page.getByLabel('Right Original image', { exact: true }).evaluate(canvas => (canvas as HTMLCanvasElement).toDataURL())).toBe(original);
  await settings(page, 'jpeg');
  await expect(right).toHaveValue('jpeg');
  await expect(left).toHaveValue('jpeg');
  await settings(page, 'jpegai'); // No output for this tab: fall back to the last encode.
  await expect(right).toHaveValue('avif');
  await expect(left).toHaveValue('jpeg');
  await settings(page, 'avif');
  await right.selectOption('source');
  await run(page, 'avif');
  await expect(right).toHaveValue('avif');
  await expect(left).toHaveValue('jpeg');
  await page.setViewportSize({ width: 390, height: 844 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBeTruthy();
});

test('file-size percentage reports larger output honestly', async ({ page }) => {
  await page.goto('/');
  const png = await page.evaluate(() => {
    const canvas = document.createElement('canvas');
    canvas.width = 1; canvas.height = 1;
    return canvas.toDataURL('image/png').split(',')[1]!;
  });
  await page.getByLabel('Choose image file', { exact: true }).setInputFiles({ name: 'one-pixel.png', mimeType: 'image/png', buffer: Buffer.from(png, 'base64') });
  await expect(page.getByLabel('Original image', { exact: true })).toHaveAttribute('width', '1');
  await run(page, 'jpeg');
  const size = resultRow(page, 'jpeg').locator('[data-metric="size"]');
  expect(Number(await size.getAttribute('data-percent'))).toBeLessThan(0);
  await expect(size).not.toContainText('%');
  await size.getByRole('button').hover();
  await expect(page.getByRole('tooltip')).toContainText('% larger');
});

test('every comparison metric has a tooltip and improvement direction', async ({ page }) => {
  await page.goto('/');
  const table = page.getByRole('table', { name: 'Image format comparison', exact: true });
  await expect(table.locator('tbody tr')).toHaveCount(6);
  const headings = table.locator('thead th button');
  await expect(headings).toHaveCount(6);
  await expect(table.locator('.direction')).toHaveCount(0);
  await expect(table).not.toContainText('Startup');
  await expect(table).not.toContainText('Size reduction');
  const directions = ['Lower', 'Lower', 'Lower', 'Higher', 'Higher', 'Lower'];
  for (const [index, heading] of (await headings.all()).entries()) {
    await heading.focus();
    await expect(page.getByRole('tooltip')).toBeVisible();
    await expect(page.getByRole('tooltip')).toContainText(`${directions[index]} is better`);
    await page.keyboard.press('Escape');
    await expect(page.getByRole('tooltip')).toHaveCount(0);
  }
  const positions = await page.evaluate(() => ({
    tabs: document.querySelector('.codec-tabs')!.getBoundingClientRect().bottom,
    tableTop: document.querySelector('.comparison-card')!.getBoundingClientRect().top,
    panelBottom: document.querySelector('[role="tabpanel"]:not([style*="display: none"])')!.getBoundingClientRect().bottom,
  }));
  expect(positions.tableTop).toBeGreaterThanOrEqual(positions.tabs);
  expect(positions.tableTop).toBeGreaterThanOrEqual(positions.panelBottom);
  await expect(page.locator('.codec-pane .comparison-card')).toHaveCount(0);
});

test('comparison format labels instantly jump to the controls or original', async ({ page }) => {
  await page.goto('/');
  for (const [id, label] of [['avif', 'AVIF'], ['jxl', 'JPEG XL'], ['jpeg', 'JPEG'], ['jpegai', 'JPEG AI']]) {
    await resultRow(page, id!).getByRole('button', { name: label, exact: true }).click();
    await expect(page.getByRole('tab', { name: label, exact: true })).toHaveAttribute('aria-selected', 'true');
    await expect(page.locator(`#panel-${id}`)).toBeFocused();
    const box = await page.locator(`#panel-${id}`).boundingBox();
    expect(box!.y).toBeGreaterThanOrEqual(0);
    expect(box!.y).toBeLessThanOrEqual(24);
  }
  await resultRow(page, 'original').getByRole('button', { name: 'Original', exact: true }).click();
  await expect(page.locator('#original-image')).toBeFocused();
  expect(await page.evaluate(() => scrollY)).toBe(0);
});

test('expensive JPEG AI jobs pause for confirmation before loading any model weights', async ({ page }) => {
  let weightRequests = 0;
  await page.route('**/models/*.zjb*', route => { weightRequests++; return route.abort(); });
  await page.goto('/');
  const png = await page.evaluate(() => {
    const canvas = document.createElement('canvas'); canvas.width = 3600; canvas.height = 2400;
    const context = canvas.getContext('2d')!;
    context.fillStyle = '#548d6a'; context.fillRect(0, 0, canvas.width, canvas.height);
    return canvas.toDataURL('image/png').split(',')[1]!;
  });
  await page.getByLabel('Choose image file', { exact: true }).setInputFiles({ name: 'large.png', mimeType: 'image/png', buffer: Buffer.from(png, 'base64') });
  const panel = pane(page, 'jpegai');
  await expect(panel.getByRole('button', { name: 'Encode', exact: true })).toBeEnabled();
  await panel.getByRole('button', { name: 'Encode', exact: true }).click();
  await expect(panel).toHaveAttribute('data-phase', 'confirming');
  await expect(panel.getByRole('alert')).toContainText('Estimated working memory');
  expect(weightRequests).toBe(0);
  await expect(resultRow(page, 'jpegai').locator('td')).toHaveText(Array(6).fill(''));
  await panel.getByRole('button', { name: 'Cancel', exact: true }).click();
  await expect(panel).toHaveAttribute('data-phase', 'cancelled');
  expect(weightRequests).toBe(0);
  await panel.getByRole('button', { name: 'Encode', exact: true }).click();
  await expect(panel).toHaveAttribute('data-phase', 'confirming');
  await panel.getByRole('button', { name: 'Continue encoding', exact: true }).click();
  // Block the intentional weight request so this tests the gate, not a multi-GiB encode.
  await expect.poll(() => weightRequests).toBe(1);
  await expect(panel).toHaveAttribute('data-phase', 'error');
  await expect(panel.locator('.memory-warning')).toHaveCount(0);
});

test('HOP target search checks synthesis memory before downloading weights', async ({ page }) => {
  let weights = 0;
  await page.route('**/models/*.zjb*', route => { weights++; return route.abort(); });
  await page.goto('/');
  const png = await page.evaluate(() => {
    const canvas = document.createElement('canvas'); canvas.width = 2048; canvas.height = 2048;
    const context = canvas.getContext('2d')!; context.fillStyle = '#548d6a'; context.fillRect(0, 0, 2048, 2048);
    return canvas.toDataURL('image/png').split(',')[1]!;
  });
  await page.getByLabel('Choose image file', { exact: true }).setInputFiles({ name: 'hop-memory.png', mimeType: 'image/png', buffer: Buffer.from(png, 'base64') });
  const panel = pane(page, 'jpegai');
  await panel.getByLabel('Operating point', { exact: true }).selectOption('hop');
  await panel.getByLabel('Model selection', { exact: true }).selectOption('target');
  await panel.getByRole('button', { name: 'Encode', exact: true }).click();
  await expect(panel).toHaveAttribute('data-phase', 'confirming');
  await expect(panel.locator('.memory-warning')).toContainText('1.57 GiB');
  expect(weights).toBe(0);
  await expect(panel.getByLabel('Operating point', { exact: true })).toBeDisabled();
  await panel.getByRole('button', { name: 'Cancel', exact: true }).click();
  await expect(panel).toHaveAttribute('data-phase', 'cancelled');
  expect(weights).toBe(0);
});

test('PSNR and SSIM tooltips explain the scores on hover, focus and tap', async ({ page }) => {
  await page.goto('/'); await upload(page); await run(page, 'jpeg');
  const psnr = page.getByRole('button', { name: 'About PSNR', exact: true });
  const ssim = page.getByRole('button', { name: 'About SSIM · luma', exact: true });
  await psnr.hover();
  await expect(page.getByRole('tooltip')).toContainText('Peak signal-to-noise ratio');
  await expect(page.getByRole('tooltip')).toContainText('Higher is better');
  await page.getByRole('heading', { name: 'Image compression', exact: true }).hover();
  await expect(page.getByRole('tooltip')).toHaveCount(0);
  await ssim.focus();
  await expect(page.getByRole('tooltip')).toContainText('Structural similarity');
  await expect(page.getByRole('tooltip')).toContainText('Closer to 1');
  await expect(page.getByRole('tooltip')).toContainText('colour differences');
  await page.keyboard.press('Escape');
  await expect(page.getByRole('tooltip')).toHaveCount(0);
  await page.setViewportSize({ width: 390, height: 844 });
  await page.emulateMedia({ colorScheme: 'dark' });
  await psnr.click();
  const tooltip = page.getByRole('tooltip');
  await expect(tooltip).toBeVisible();
  const box = await tooltip.boundingBox();
  expect(box!.x).toBeGreaterThanOrEqual(0);
  expect(box!.x + box!.width).toBeLessThanOrEqual(390);
  await page.keyboard.press('Escape');
  await expect(page.getByRole('tooltip')).toHaveCount(0);
});

test('Full resolution is the default and downscale presets restore the original file', async ({ page }) => {
  await page.goto('/');
  const sizes = page.getByRole('group', { name: 'Downscale image', exact: true });
  await expect(sizes.getByRole('button')).toHaveText(['Full resolution', '2048', '1024', '512', '256']);
  await expect(sizes.getByRole('button', { name: 'Full resolution', exact: true })).toHaveAttribute('aria-pressed', 'true');
  const png = await page.evaluate(() => {
    const canvas = document.createElement('canvas'); canvas.width = 3072; canvas.height = 1536;
    const ctx = canvas.getContext('2d')!;
    const gradient = ctx.createLinearGradient(0, 0, 3072, 1536);
    gradient.addColorStop(0, '#bf423d'); gradient.addColorStop(1, '#294dda');
    ctx.fillStyle = gradient; ctx.fillRect(0, 0, 3072, 1536);
    ctx.fillStyle = '#f0d030'; ctx.fillRect(100, 100, 400, 500);
    return canvas.toDataURL('image/png').split(',')[1]!;
  });
  await page.getByLabel('Choose image file', { exact: true }).setInputFiles({ name: 'large.png', mimeType: 'image/png', buffer: Buffer.from(png, 'base64') });
  const input = page.getByRole('region', { name: 'Image input', exact: true });
  await expect(input).toHaveAttribute('data-width', '3072');
  await expect(input).toHaveAttribute('data-height', '1536');
  await expect(sizes.getByRole('button').first()).toBeEnabled();
  await expect(page.getByLabel('Original image', { exact: true })).toHaveAttribute('width', '96');
  const preview = await page.getByLabel('Original image', { exact: true }).evaluate(canvas => (canvas as HTMLCanvasElement).toDataURL());
  await run(page, 'jpeg');
  await expect(page.getByLabel('Right JPEG image', { exact: true })).toHaveAttribute('width', '3072');
  const jpegTime = await resultRow(page, 'jpeg').locator('[data-metric="encode"]').getAttribute('data-ms');
  const jpegReduction = await resultRow(page, 'jpeg').locator('[data-metric="size"]').getAttribute('data-percent');
  const originalBytes = Buffer.from(png, 'base64').length;
  const reference = resultRow(page, 'original');
  await expect(reference.getByRole('button', { name: 'Original', exact: true })).toBeVisible();
  await expect(reference.locator('[data-metric="bpp"]')).toHaveText((originalBytes * 8 / (3072 * 1536)).toFixed(3));
  for (const edge of [2048, 1024, 512, 256]) {
    await sizes.getByRole('button', { name: String(edge), exact: true }).click();
    await expect(input).toHaveAttribute('data-width', String(edge));
    await expect(input).toHaveAttribute('data-height', String(edge / 2));
    await expect(resultRow(page, 'jpeg').locator('[data-metric="encode"]')).toHaveAttribute('data-ms', jpegTime!);
    await expect(resultRow(page, 'jpeg').locator('[data-metric="size"]')).toHaveAttribute('data-percent', jpegReduction!);
    await expect(reference.getByRole('button', { name: 'Downscaled', exact: true })).toBeVisible();
    const resizedBytes = Number(await reference.getAttribute('data-reference-bytes'));
    expect(resizedBytes).toBeGreaterThan(0);
    expect(resizedBytes).not.toBe(originalBytes);
    await expect(reference.locator('[data-metric="bpp"]')).toHaveText((resizedBytes * 8 / (edge * edge / 2)).toFixed(3));
    await expect(resultRow(page, 'jpeg').locator('.dimensions')).toHaveText('3072 × 1536');
    await expect(page.getByLabel('Left Downscaled image', { exact: true })).toHaveAttribute('width', String(edge));
    await expect(page.getByLabel('Right JPEG image', { exact: true })).toHaveAttribute('width', '3072');
    await expect(sizes.getByRole('button', { name: String(edge), exact: true })).toHaveAttribute('aria-pressed', 'true');
    await expect(sizes.getByRole('button').first()).toBeEnabled();
  }
  await run(page, 'avif');
  await expect(resultRow(page, 'avif').locator('.dimensions')).toHaveText('256 × 128');
  const avifReduction = await resultRow(page, 'avif').locator('[data-metric="size"]').getAttribute('data-percent');
  await sizes.getByRole('button', { name: 'Full resolution', exact: true }).click();
  await expect(input).toHaveAttribute('data-width', '3072');
  await expect(input).toHaveAttribute('data-height', '1536');
  await expect(sizes.getByRole('button').first()).toBeEnabled();
  await expect.poll(() => page.getByLabel('Original image', { exact: true }).evaluate(canvas => (canvas as HTMLCanvasElement).toDataURL())).toBe(preview);
  await expect(reference.getByRole('button', { name: 'Original', exact: true })).toBeVisible();
  await expect(reference).toHaveAttribute('data-reference-bytes', String(originalBytes));
  await expect(resultRow(page, 'jpeg').locator('[data-metric="encode"]')).toHaveAttribute('data-ms', jpegTime!);
  await expect(resultRow(page, 'avif').locator('[data-metric="size"]')).toHaveAttribute('data-percent', avifReduction!);
  await expect(page.getByLabel('Left Original image', { exact: true })).toHaveAttribute('width', '3072');
  await expect(page.getByLabel('Right AVIF image', { exact: true })).toHaveAttribute('width', '256');
  await upload(page);
  await expect(page.locator('tbody [data-ms]')).toHaveCount(0);
  await expect(page.locator('tbody [data-rank]')).toHaveCount(0);
  await expect(page.locator('.comparison')).toHaveCount(0);
  await page.setViewportSize({ width: 390, height: 844 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBeTruthy();
});

test('2048 square JPEG AI output is encoded, decoded and measured without the old pixel cap', async ({ page }) => {
  await page.goto('/');
  const png = await page.evaluate(() => {
    const canvas = document.createElement('canvas'); canvas.width = 2048; canvas.height = 2048;
    const ctx = canvas.getContext('2d')!;
    ctx.fillStyle = '#448c72'; ctx.fillRect(0, 0, 2048, 2048);
    ctx.fillStyle = '#d0a64a'; ctx.fillRect(250, 200, 900, 1100);
    return canvas.toDataURL('image/png').split(',')[1]!;
  });
  await page.getByLabel('Choose image file', { exact: true }).setInputFiles({ name: '2k.png', mimeType: 'image/png', buffer: Buffer.from(png, 'base64') });
  const input = page.getByRole('region', { name: 'Image input', exact: true });
  await expect(input).toHaveAttribute('data-width', '2048');
  await expect(input).toHaveAttribute('data-height', '2048');
  await run(page, 'jpegai');
  const decoded = page.getByLabel('Right JPEG AI image', { exact: true });
  await expect(decoded).toHaveAttribute('width', '2048');
  await expect(decoded).toHaveAttribute('height', '2048');
  await expect(resultRow(page, 'jpegai').locator('[data-metric="ssim"]')).toHaveText(/\d+\.\d+/);
});

test('browser color preference switches every pane and native controls without changing image pixels', async ({ page }) => {
  await page.emulateMedia({ colorScheme: 'light' });
  await page.goto('/'); await upload(page);
  await settings(page, 'jpegai');
  const light = await page.evaluate(() => ({
    background: getComputedStyle(document.documentElement).backgroundColor,
    pane: getComputedStyle(document.querySelector('.codec-pane')!).backgroundColor,
    canvas: (document.querySelector('.source-preview canvas') as HTMLCanvasElement).toDataURL(),
  }));
  await page.emulateMedia({ colorScheme: 'dark' });
  expect(await page.evaluate(() => getComputedStyle(document.documentElement).colorScheme)).toBe('dark');
  for (const panel of await page.locator('.codec-pane').all()) {
    expect(await panel.evaluate(element => getComputedStyle(element).backgroundColor)).not.toBe(light.pane);
  }
  expect(await page.getByRole('group', { name: 'Downscale image', exact: true }).getByRole('button', { name: 'Full resolution', exact: true }).evaluate(element => getComputedStyle(element).colorScheme)).toBe('dark');
  expect(await page.evaluate(() => getComputedStyle(document.documentElement).backgroundColor)).not.toBe(light.background);
  expect(await page.locator('.source-preview canvas').evaluate(element => (element as HTMLCanvasElement).toDataURL())).toBe(light.canvas);
  await page.setViewportSize({ width: 390, height: 844 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBeTruthy();
  await page.emulateMedia({ colorScheme: 'light' });
  expect(await page.evaluate(() => getComputedStyle(document.documentElement).backgroundColor)).toBe(light.background);
});
