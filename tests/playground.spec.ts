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
  await expect(page.getByRole('button', { name: 'Encode all four formats' })).toBeEnabled();
}

function pane(page: Page, format: string) {
  return page.locator(`[data-codec="${format}"]`);
}

async function settings(page: Page, format: string) {
  const details = pane(page, format).locator('.pane-settings');
  if (!await details.evaluate(element => element.hasAttribute('open'))) await details.locator(':scope > summary').click();
}

async function run(page: Page, format: string) {
  const panel = pane(page, format);
  await panel.getByRole('button', { name: /^Encode(?: again)?$/ }).click();
  await expect(panel.locator('.pane-status')).toContainText(/Done\.|failed/, { timeout: 150_000 });
  await expect(panel.getByRole('alert')).toHaveCount(0);
  await expect(panel.locator('[data-metric="encode"]')).toHaveAttribute('data-ms', /[0-9]/);
}

async function report(page: Page) {
  const event = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Download comparison JSON' }).click();
  const stream = await (await event).createReadStream();
  const chunks: Buffer[] = [];
  for await (const chunk of stream!) chunks.push(chunk);
  return JSON.parse(Buffer.concat(chunks).toString());
}

test('four simultaneous comparison panes retain real WASM outputs and separate metrics', async ({ page }) => {
  const requests: string[] = [];
  page.on('request', request => { expect(request.method()).toBe('GET'); requests.push(request.url()); });
  await page.goto('/');
  await expect(page.locator('.codec-pane')).toHaveCount(4);
  await expect(page.locator('aside, dialog, table')).toHaveCount(0);
  await upload(page);
  await page.getByRole('button', { name: 'Encode all four formats' }).click();
  for (const [id, label] of [['jpegai', 'JPEG AI'], ['jpeg', 'JPEG'], ['avif', 'AVIF'], ['jxl', 'JPEG XL']]) {
    const panel = pane(page, id!);
    await expect(panel.locator('.pane-status')).toHaveText('Done.', { timeout: 150_000 });
    await expect(panel.getByRole('alert')).toHaveCount(0);
    const event = page.waitForEvent('download');
    await panel.getByRole('button', { name: `Download ${label}`, exact: true }).click();
    const download = await event;
    expect(download.suggestedFilename()).toMatch(/\.(jai|jpg|avif|jxl)$/);
    const stream = await download.createReadStream();
    const chunks: Buffer[] = []; for await (const chunk of stream!) chunks.push(chunk);
    const encoded = Buffer.concat(chunks);
    expect(encoded.length).toBeGreaterThan(16);
    if (id === 'jpeg') expect(encoded.subarray(0, 2).toString('hex')).toBe('ffd8');
    if (id === 'avif') expect(encoded.subarray(4, 8).toString()).toBe('ftyp');
    const decoded = panel.getByLabel(`Decoded ${label} image`, { exact: true });
    await expect(decoded).toBeVisible();
    await expect(decoded).toHaveAttribute('width', '96');
    await expect(decoded).toHaveAttribute('height', '64');
  }
  const previewTops = await page.locator('.comparison').evaluateAll(elements => elements.map(element => element.getBoundingClientRect().top));
  expect(Math.max(...previewTops) - Math.min(...previewTops)).toBeLessThan(1);
  const saved = await report(page);
  expect(saved.results).toHaveLength(4);
  for (const result of saved.results) {
    expect(result.encodeMs).toBeGreaterThan(0);
    expect(result.decodeMs).toBeGreaterThan(0);
    expect(result.startupMs).toBeGreaterThan(0);
    expect(result.metrics.ssim).toBeGreaterThan(0.5);
    const panel = pane(page, result.settings.format);
    await expect(panel.locator('[data-metric="encode"]')).toHaveAttribute('data-ms', String(result.encodeMs));
    await expect(panel.locator('[data-metric="decode"]')).toHaveAttribute('data-ms', String(result.decodeMs));
    expect(result.compressionRatio).toBeCloseTo(result.width * result.height * 3 / result.encodedBytes);
    await expect(panel.locator('[data-metric="ratio"]')).toHaveAttribute('data-ratio', String(result.compressionRatio));
  }
  expect(requests.every(url => new URL(url).hostname === 'localhost' || url.startsWith('blob:'))).toBeTruthy();
});

test('model library downloads, caches, selects, exports, imports and reuses weights', async ({ page }) => {
  await page.goto('/');
  await settings(page, 'jpegai');
  await expect(page.locator('dialog')).toHaveCount(0);
  const models = pane(page, 'jpegai').getByRole('region', { name: 'JPEG AI models', exact: true });
  await expect(models).toBeVisible();
  await expect(models.locator('.model-card')).toHaveCount(4);
  const card = models.locator('[data-model="0"]');
  await card.getByRole('button', { name: 'Download weights' }).click();
  await expect(card.getByText('Downloaded · ready')).toBeVisible({ timeout: 60_000 });
  const event = page.waitForEvent('download');
  await card.getByRole('button', { name: 'Save .zjb' }).click();
  const saved = await event;
  expect(saved.suggestedFilename()).toBe('bop-m0.zjb');
  const path = await saved.path();
  await expect(card.getByRole('button', { name: 'Remove' })).toBeEnabled();
  await card.getByRole('button', { name: 'Remove' }).click();
  await expect(card.getByText('Not downloaded')).toBeVisible();
  await page.getByLabel('Import model file').setInputFiles(path!);
  await expect(card.getByText('Downloaded · ready')).toBeVisible();
  await card.getByRole('button', { name: 'Use model', exact: false }).click();
  await expect(models).toBeVisible();
  await expect(pane(page, 'jpegai').getByLabel('Trained model', { exact: true })).toHaveValue('0');
  await upload(page);
  // Block every weight request: a downloaded model must be usable from Cache Storage.
  await page.route('**/models/*.zjb*', route => route.abort());
  await run(page, 'jpegai');
});

test('lossless modes really reconstruct identical working pixels', async ({ page }) => {
  await page.goto('/'); await upload(page);
  for (const format of ['avif', 'jxl']) {
    await settings(page, format);
    const panel = pane(page, format);
    await panel.getByRole('checkbox', { name: 'Lossless' }).check();
    await run(page, format);
    await expect(panel.locator('.secondary-stats')).toContainText('1.0000');
    await expect(panel.locator('.secondary-stats')).toContainText('∞');
  }
});

test('target-bpp search produces a real JPEG AI result', async ({ page }) => {
  await page.goto('/'); await upload(page);
  await settings(page, 'jpegai');
  await pane(page, 'jpegai').getByLabel('Rate control', { exact: true }).selectOption('target');
  await run(page, 'jpegai');
  const saved = await report(page);
  expect(saved.results[0].settings.rateMode).toBe('target');
});

test('cancel stops work and allows another run', async ({ page }) => {
  await page.goto('/'); await upload(page);
  await page.route('**/models/*.zjb*', async route => { await new Promise(resolve => setTimeout(resolve, 2000)); await route.abort(); });
  await pane(page, 'jpegai').getByRole('button', { name: 'Encode', exact: true }).click();
  await pane(page, 'jpegai').getByRole('button', { name: 'Cancel encoding', exact: true }).click();
  await expect(pane(page, 'jpegai').locator('.pane-status')).toContainText('Cancelled.');
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
  await expect(pane(page, 'jpegai').getByRole('region', { name: 'JPEG AI models', exact: true })).toBeVisible();
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
  let saved = await report(page);
  expect(saved.results.map((r: any) => [r.settings.format, r.settings.quality])).toEqual([['jpeg', 34], ['avif', 82]]);
  await pane(page, 'jpeg').locator('#jpeg-quality').evaluate(input => {
    (input as HTMLInputElement).value = '50';
    input.dispatchEvent(new Event('input', { bubbles: true }));
  });
  await expect(pane(page, 'jpeg').locator('.changed-note')).toContainText('Settings changed');
  await expect(pane(page, 'avif').locator('.changed-note')).toHaveCount(0);
  await run(page, 'jpeg');
  saved = await report(page);
  expect(saved.results).toHaveLength(2);
  expect(saved.results.map((r: any) => [r.settings.format, r.settings.quality])).toEqual([['jpeg', 50], ['avif', 82]]);
  await expect(page.locator('table, aside, dialog')).toHaveCount(0);
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
  await pane(page, 'jpeg').getByRole('button', { name: 'Encode', exact: true }).click();
  await pane(page, 'jxl').getByRole('button', { name: 'Encode', exact: true }).click();
  await expect(pane(page, 'jpeg').locator('.pane-status')).toContainText('Queued');
  await pane(page, 'jxl').getByRole('button', { name: 'Cancel queued job' }).click();
  await expect(pane(page, 'jxl').locator('.pane-status')).toContainText('Cancelled before starting');
  await pane(page, 'jpegai').getByRole('button', { name: 'Cancel encoding' }).click();
  await expect(pane(page, 'jpeg').locator('.pane-status')).toHaveText('Done.');
  await expect(pane(page, 'jpegai').locator('.pane-status')).toContainText('Cancelled');
  await expect(pane(page, 'jxl').locator('[data-metric="encode"]')).not.toHaveAttribute('data-ms');
});

test('browser color preference switches every pane and native controls without changing image pixels', async ({ page }) => {
  await page.emulateMedia({ colorScheme: 'light' });
  await page.goto('/'); await upload(page);
  await settings(page, 'jpegai');
  const light = await page.evaluate(() => ({
    background: getComputedStyle(document.documentElement).backgroundColor,
    pane: getComputedStyle(document.querySelector('.codec-pane')!).backgroundColor,
    canvas: (document.querySelector('.image-canvas') as HTMLCanvasElement).toDataURL(),
  }));
  await page.emulateMedia({ colorScheme: 'dark' });
  expect(await page.evaluate(() => getComputedStyle(document.documentElement).colorScheme)).toBe('dark');
  for (const panel of await page.locator('.codec-pane').all()) {
    expect(await panel.evaluate(element => getComputedStyle(element).backgroundColor)).not.toBe(light.pane);
  }
  expect(await page.locator('#working-size').evaluate(element => getComputedStyle(element).colorScheme)).toBe('dark');
  expect(await page.evaluate(() => getComputedStyle(document.documentElement).backgroundColor)).not.toBe(light.background);
  expect(await page.locator('.image-canvas').first().evaluate(element => (element as HTMLCanvasElement).toDataURL())).toBe(light.canvas);
  await page.setViewportSize({ width: 390, height: 844 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBeTruthy();
  await page.emulateMedia({ colorScheme: 'light' });
  expect(await page.evaluate(() => getComputedStyle(document.documentElement).backgroundColor)).toBe(light.background);
});
