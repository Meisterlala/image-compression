import { test, expect, type Page } from '@playwright/test';
import { execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';

async function upload(page: Page, width = 96, height = 64) {
  const png = await page.evaluate(({ width, height }) => {
    const canvas = document.createElement('canvas'); canvas.width = width; canvas.height = height;
    const ctx = canvas.getContext('2d')!;
    const gradient = ctx.createLinearGradient(0, 0, width, height);
    gradient.addColorStop(0, '#ce201b'); gradient.addColorStop(0.5, '#28b579'); gradient.addColorStop(1, '#403cd0');
    ctx.fillStyle = gradient; ctx.fillRect(0, 0, width, height);
    ctx.fillStyle = '#f5d050'; ctx.fillRect(5, 7, width / 3, height / 3);
    return canvas.toDataURL('image/png').split(',')[1]!;
  }, { width, height });
  await page.getByLabel('Choose image file', { exact: true }).setInputFiles({ name: 'coolchic-test.png', mimeType: 'image/png', buffer: Buffer.from(png, 'base64') });
  await expect(page.getByRole('region', { name: 'Image input', exact: true })).toHaveAttribute('data-width', String(width));
}

async function open(page: Page) {
  await page.goto('/');
  await page.getByRole('tab', { name: 'Cool-chic 5.0', exact: true }).click();
  return page.locator('[data-codec="coolchic"]');
}

test('Cool-chic trains, downloads a portable bitstream and retains its real decoded result', async ({ page }, info) => {
  const requests: string[] = [];
  page.on('request', req => { requests.push(req.url()); expect(req.method()).toBe('GET'); });
  const pane = await open(page);
  await expect(pane.locator('.pane-settings .hint')).toHaveCount(0);
  await expect(pane.getByRole('slider')).toHaveCount(2);
  await expect(pane.getByRole('radiogroup', { name: 'Compression preset' }).getByRole('radio')).toHaveCount(4);
  await expect(pane.getByRole('radio', { name: 'LOP', exact: true })).toBeChecked();
  await expect(pane.getByLabel('Training iterations', { exact: true })).toHaveAttribute('aria-valuetext', '2000 iterations');
  await expect(pane.getByLabel('Rate constraint (λ)', { exact: true })).toHaveAttribute('aria-valuetext', '0.001');
  await expect(pane.getByRole('checkbox')).toHaveCount(0);
  await expect(pane.locator('.model-card')).toHaveCount(0);
  await upload(page);
  await pane.getByRole('button', { name: 'Encode', exact: true }).click();
  await expect(pane).toHaveAttribute('data-phase', /done|error/, { timeout: 150000 });
  await expect(pane).toHaveAttribute('data-phase', 'done');
  await expect(pane.getByRole('alert')).toHaveCount(0);
  const row = page.locator('[data-result-format="coolchic"]');
  await expect(row.locator('[data-metric="size"]')).not.toBeEmpty();
  await expect(row.locator('[data-metric="psnr"]')).toContainText('dB');
  await expect(page.getByLabel('Right Cool-chic 5.0 image', { exact: true })).toHaveAttribute('width', '96');
  const download = page.waitForEvent('download');
  await pane.getByRole('button', { name: 'Download Cool-chic 5.0', exact: true }).click();
  const file = await download;
  expect(file.suggestedFilename()).toMatch(/-lop-96x64\.cool$/);
  const encodedPath = info.outputPath('output.cool');
  const rgbaPath = info.outputPath('decoded.rgba');
  await file.saveAs(encodedPath);
  const encoded = readFileSync(encodedPath);
  expect(encoded.length).toBeGreaterThan(41);
  // Independent native process decodes the browser-downloaded file; no sidecar
  // model or original image is supplied. Python is not needed for browser tests.
  execFileSync('cargo', ['run', '--quiet', '--manifest-path', 'crates/coolchic/Cargo.toml', '--example', 'rgba', '--', 'decode', encodedPath, rgbaPath]);
  const nativePixels = [...readFileSync(rgbaPath)];
  const browserPixels = await page.getByLabel('Right Cool-chic 5.0 image', { exact: true }).evaluate((el) => {
    const source = el as HTMLCanvasElement | HTMLImageElement;
    const canvas = document.createElement('canvas'); canvas.width = 96; canvas.height = 64;
    const ctx = canvas.getContext('2d')!; ctx.drawImage(source, 0, 0);
    return [...ctx.getImageData(0, 0, 96, 64).data];
  });
  expect(browserPixels).toEqual(nativePixels);
  expect(requests.some(url => url.endsWith('.zjb'))).toBeFalsy();
  expect(requests.every(url => new URL(url).origin === new URL(page.url()).origin)).toBeTruthy();
  expect(requests.some(url => /coolchic_bg.*\.wasm/.test(url))).toBeTruthy();
  const goldenCases = ['lop', 'mop', 'hop', 'vhop', 'common', 'resize', 'linear'].map(name => ({
    name,
    encoded: Array.from(readFileSync(`crates/coolchic/tests/fixtures/full-reference/${name}.cool`)),
    expected: Array.from(readFileSync(`crates/coolchic/tests/fixtures/full-reference/${name}.rgba`)),
  }));
  // Decode official golden streams in browser WASM too, including common-noise
  // math and all network architectures (not only the native interoperability path).
  const goldenPixels = await page.evaluate(async ({ urls, cases }) => {
    let bindings;
    for (const url of urls) {
      const module = await import(url);
      if ('decodeCoolChic' in module) { bindings = module; break; }
    }
    if (!bindings) throw new Error('Cool-chic WASM bindings were not loaded');
    await bindings.default();
    return cases.map(item => {
      const image = bindings.decodeCoolChic(new Uint8Array(item.encoded));
      try { return { name: item.name, pixels: Array.from(image.rgba()) }; }
      finally { image.free(); }
    });
  }, { urls: requests.filter(url => /\/coolchic-[^/]+\.js$/.test(url)), cases: goldenCases });
  goldenPixels.forEach((actual, index) => expect(actual.pixels, actual.name).toEqual(goldenCases[index]!.expected));
  await page.getByRole('tab', { name: 'JPEG', exact: true }).click();
  await page.getByRole('tab', { name: 'Cool-chic 5.0', exact: true }).click();
  await expect(pane.getByRole('button', { name: 'Download Cool-chic 5.0', exact: true })).toBeVisible();
  await pane.getByRole('radio', { name: 'MOP', exact: true }).check();
  await pane.getByLabel('Rate constraint (λ)', { exact: true }).fill('80');
  await pane.getByLabel('Training iterations', { exact: true }).fill('18');
  await expect(page.locator('[data-persistence]')).toContainText('Saved locally');
  await page.reload();
  await expect(pane).toHaveAttribute('data-phase', 'done');
  await expect(pane.getByRole('radio', { name: 'MOP', exact: true })).toBeChecked();
  await expect(pane.getByLabel('Rate constraint (λ)', { exact: true })).toHaveAttribute('aria-valuetext', '0.01');
  await expect(pane.getByLabel('Training iterations', { exact: true })).toHaveAttribute('aria-valuetext', '4000 iterations');
  const restored = page.waitForEvent('download');
  await pane.getByRole('button', { name: 'Download Cool-chic 5.0', exact: true }).click();
  const restoredFile = await restored;
  expect(restoredFile.suggestedFilename()).toContain('-lop-'); // Saved result, not changed controls.
  const stream = await restoredFile.createReadStream();
  const chunks: Buffer[] = []; for await (const chunk of stream!) chunks.push(chunk);
  expect(Buffer.concat(chunks)).toEqual(encoded);
});

test('Cool-chic rejects oversize input explicitly instead of silently resizing', async ({ page }) => {
  const requests: string[] = [];
  page.on('request', request => requests.push(request.url()));
  const pane = await open(page);
  await upload(page, 2048, 2048);
  await pane.getByRole('button', { name: 'Encode', exact: true }).click();
  await expect(pane).toHaveAttribute('data-phase', 'error');
  await expect(pane.getByRole('alert')).toContainText('up to 4 million pixels');
  await expect(pane.getByRole('alert')).toContainText('Downscale image');
  expect(requests.some(url => /coolchic_bg.*\.wasm/.test(url))).toBeFalsy();
  await expect(pane.getByRole('button', { name: 'Download Cool-chic 5.0', exact: true })).toHaveCount(0);
});

test('Cool-chic reports training progress and cancellation leaves other codecs usable', async ({ page }) => {
  const pane = await open(page);
  await upload(page, 768, 512);
  await pane.getByLabel('Training iterations', { exact: true }).fill('100');
  await pane.getByRole('button', { name: 'Encode', exact: true }).click();
  await expect(pane.locator('.pane-status')).toContainText('Training Cool-chic', { timeout: 60000 });
  await pane.getByRole('button', { name: 'Cancel encoding', exact: true }).click();
  await expect(pane).toHaveAttribute('data-phase', 'cancelled');
  await expect(pane.getByRole('button', { name: 'Download Cool-chic 5.0', exact: true })).toHaveCount(0);
  await page.getByRole('tab', { name: 'JPEG', exact: true }).click();
  const jpeg = page.locator('[data-codec="jpeg"]');
  await jpeg.getByRole('button', { name: 'Encode', exact: true }).click();
  await expect(jpeg).toHaveAttribute('data-phase', 'done');
});

test('Cool-chic rejects a reference training job above its memory budget', async ({ page }) => {
  const pane = await open(page);
  await pane.getByRole('radio', { name: 'HOP', exact: true }).check();
  await upload(page, 2000, 2000); // Within the hard pixel cap, but too large for HOP training.
  await pane.getByRole('button', { name: 'Encode', exact: true }).click();
  await expect(pane).toHaveAttribute('data-phase', 'error');
  await expect(pane.getByRole('alert')).toContainText('3 GiB');
  await expect(pane.getByRole('alert')).toContainText('Downscale image');
});

test('Cool-chic asks before a large reference job and can cancel before training', async ({ page }) => {
  const pane = await open(page);
  await pane.getByRole('radio', { name: 'HOP', exact: true }).check();
  await upload(page, 1280, 1024);
  await pane.getByRole('button', { name: 'Encode', exact: true }).click();
  await expect(pane).toHaveAttribute('data-phase', 'confirming');
  await expect(pane.getByRole('alert')).toContainText('Large Cool-chic 5.0 encode');
  await pane.getByRole('button', { name: 'Cancel', exact: true }).click();
  await expect(pane).toHaveAttribute('data-phase', 'cancelled');
});

test('Cool-chic segmented choices, intuitive sliders and practical tooltips are usable', async ({ page }) => {
  const pane = await open(page);
  const highest = pane.getByRole('radio', { name: 'VHOP', exact: true });
  await highest.check();
  await expect(highest).toBeChecked();
  await expect.poll(() => pane.locator('.point-options').evaluate(track => {
    const box = track.getBoundingClientRect();
    const pill = getComputedStyle(track, '::before');
    const selected = track.querySelector('input:checked')!.getBoundingClientRect();
    const x = box.left + parseFloat(getComputedStyle(track).borderLeftWidth) + parseFloat(pill.left) + new DOMMatrix(pill.transform).m41;
    return Math.max(Math.abs(x - selected.left), Math.abs(parseFloat(pill.width) - selected.width));
  })).toBeLessThan(0.5);
  await expect(pane.locator('.control-description')).toContainText('takes the longest');
  await highest.press('ArrowLeft');
  await expect(pane.getByRole('radio', { name: 'HOP', exact: true })).toBeChecked();
  await pane.getByRole('button', { name: 'About Compression preset', exact: true }).focus();
  await expect(page.getByRole('tooltip')).toContainText('fastest');
  await expect(page.getByRole('tooltip')).toContainText('memory');
  await page.keyboard.press('Escape');
  await expect(page.getByRole('tooltip')).toHaveCount(0);
  const rate = pane.getByRole('slider', { name: 'Rate constraint (λ)', exact: true });
  await rate.fill('0'); await expect(rate).toHaveAttribute('aria-valuetext', '0.000001');
  await rate.fill('100'); await expect(rate).toHaveAttribute('aria-valuetext', '0.1');
  const iterations = pane.getByRole('slider', { name: 'Training iterations', exact: true });
  await iterations.fill('100'); await expect(iterations).toHaveAttribute('aria-valuetext', '100000 iterations');
  await page.setViewportSize({ width: 390, height: 844 });
  for (const label of ['Rate constraint (λ)', 'Training iterations']) {
    await pane.getByRole('button', { name: `About ${label}`, exact: true }).click();
    await expect(page.getByRole('tooltip')).toBeInViewport();
    await page.keyboard.press('Escape');
  }
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBeTruthy();
});

test('Cool-chic controls and five-tab layout fit a narrow viewport', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  const pane = await open(page);
  await expect(pane.getByLabel('Training iterations', { exact: true })).toBeVisible();
  await expect(pane.getByRole('radiogroup', { name: 'Compression preset' })).toBeVisible();
  await expect(pane.getByLabel('Rate constraint (λ)', { exact: true })).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBeTruthy();
});
