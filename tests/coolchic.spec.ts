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
  await expect(pane).toContainText('Experimental reduced encoder');
  await expect(pane).toContainText('not the full Cool-chic reference encoder');
  await expect(pane).toContainText('quality 100 is not lossless');
  await expect(pane).not.toContainText('model is included');
  await expect(pane.getByRole('slider')).toHaveCount(2);
  await expect(pane.getByRole('checkbox')).toHaveCount(0);
  await expect(pane.locator('.model-card')).toHaveCount(0);
  await upload(page);
  await pane.getByRole('button', { name: 'Encode', exact: true }).click();
  await expect(pane).toHaveAttribute('data-phase', 'done');
  await expect(pane.getByRole('alert')).toHaveCount(0);
  const row = page.locator('[data-result-format="coolchic"]');
  await expect(row.locator('[data-metric="size"]')).not.toBeEmpty();
  await expect(row.locator('[data-metric="psnr"]')).toContainText('dB');
  await expect(page.getByLabel('Right Cool-chic 5.0 image', { exact: true })).toHaveAttribute('width', '96');
  const download = page.waitForEvent('download');
  await pane.getByRole('button', { name: 'Download Cool-chic 5.0', exact: true }).click();
  const file = await download;
  expect(file.suggestedFilename()).toMatch(/\.cool$/);
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
  await page.getByRole('tab', { name: 'JPEG', exact: true }).click();
  await page.getByRole('tab', { name: 'Cool-chic 5.0', exact: true }).click();
  await expect(pane.getByRole('button', { name: 'Download Cool-chic 5.0', exact: true })).toBeVisible();
});

test('Cool-chic rejects oversize input explicitly instead of silently resizing', async ({ page }) => {
  const pane = await open(page);
  await upload(page, 2048, 2048);
  await pane.getByRole('button', { name: 'Encode', exact: true }).click();
  await expect(pane).toHaveAttribute('data-phase', 'error');
  await expect(pane.getByRole('alert')).toContainText('up to 4 million pixels');
  await expect(pane.getByRole('alert')).toContainText('Downscale image');
  await expect(pane.getByRole('button', { name: 'Download Cool-chic 5.0', exact: true })).toHaveCount(0);
});

test('Cool-chic reports training progress and cancellation leaves other codecs usable', async ({ page }) => {
  const pane = await open(page);
  await upload(page, 768, 512);
  await pane.getByLabel('Training effort', { exact: false }).fill('9');
  await pane.getByRole('button', { name: 'Encode', exact: true }).click();
  await expect(pane.locator('.pane-status')).toContainText('Training Cool-chic');
  await pane.getByRole('button', { name: 'Cancel encoding', exact: true }).click();
  await expect(pane).toHaveAttribute('data-phase', 'cancelled');
  await expect(pane.getByRole('button', { name: 'Download Cool-chic 5.0', exact: true })).toHaveCount(0);
  await page.getByRole('tab', { name: 'JPEG', exact: true }).click();
  const jpeg = page.locator('[data-codec="jpeg"]');
  await jpeg.getByRole('button', { name: 'Encode', exact: true }).click();
  await expect(jpeg).toHaveAttribute('data-phase', 'done');
});

test('Cool-chic controls and five-tab layout fit a narrow viewport', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  const pane = await open(page);
  await expect(pane.getByLabel('Training effort', { exact: false })).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBeTruthy();
});
