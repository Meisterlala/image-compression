import { test, expect, type Page } from '@playwright/test';

async function upload(page: Page, name = 'saved.png') {
  const png = await page.evaluate(() => {
    const canvas = document.createElement('canvas');
    canvas.width = 32; canvas.height = 24;
    const context = canvas.getContext('2d')!;
    context.fillStyle = '#c63821'; context.fillRect(0, 0, 32, 24);
    return canvas.toDataURL().split(',')[1]!;
  });
  await page.getByLabel('Choose image file', { exact: true }).setInputFiles({
    name, mimeType: 'image/png', buffer: Buffer.from(png, 'base64'),
  });
  await expect(page.getByLabel('Original image', { exact: true })).toHaveAttribute('width', '32');
}

async function saved(page: Page, expected: { quality: number; filename?: string; formats: string[] }) {
  await expect.poll(() => page.evaluate(async () => {
    const database = await new Promise<IDBDatabase>((resolve, reject) => {
      const request = indexedDB.open('image-compression-state', 1);
      request.onsuccess = () => resolve(request.result); request.onerror = () => reject(request.error);
    });
    try {
      type Stored = { jpeg?: { quality: number }; file?: File; results?: Record<string, unknown> };
      const records = await new Promise<Record<string, Stored>>((resolve, reject) => {
        const transaction = database.transaction('state');
        const request = transaction.objectStore('state').openCursor();
        const records: Record<string, Stored> = {};
        request.onsuccess = () => {
          const cursor = request.result;
          if (cursor) { records[String(cursor.key)] = cursor.value; cursor.continue(); }
        };
        transaction.oncomplete = () => resolve(records);
        transaction.onabort = transaction.onerror = () => reject(transaction.error);
      });
      return { quality: records['codec-settings']?.jpeg?.quality,
        filename: records.workspace?.file?.name, formats: Object.keys(records.workspace?.results ?? {}).sort() };
    } finally { database.close(); }
  })).toEqual(expected);
}

test('reload restores image, binary outputs, settings and viewer preferences without encoding again', async ({ page }) => {
  page.on('pageerror', error => { throw error; });
  await page.goto('/');
  await upload(page);
  await page.getByRole('tab', { name: 'JPEG', exact: true }).click();
  const panel = page.locator('[data-codec="jpeg"]');
  await panel.getByRole('button', { name: 'Encode', exact: true }).click();
  await expect(panel).toHaveAttribute('data-phase', 'done');
  const pixels = await page.getByLabel('Right JPEG image', { exact: true }).evaluate(canvas => (canvas as HTMLCanvasElement).toDataURL());
  const metrics = await page.locator('[data-result-format="jpeg"]').innerText();
  await page.getByLabel('Image comparison zoom').selectOption('2');
  await page.getByLabel('Image comparison split').fill('72');
  await panel.getByLabel(/^Quality/).fill('61');
  await saved(page, { quality: 61, filename: 'saved.png', formats: ['jpeg'] });
  const binary = async () => page.evaluate(async () => {
    const database = await new Promise<IDBDatabase>((resolve, reject) => {
      const request = indexedDB.open('image-compression-state', 1);
      request.onsuccess = () => resolve(request.result); request.onerror = () => reject(request.error);
    });
    try {
      return await new Promise<number[]>((resolve, reject) => {
        const request = database.transaction('state').objectStore('state').get('workspace');
        request.onsuccess = () => resolve(Array.from(request.result.results.jpeg.encoded));
        request.onerror = () => reject(request.error);
      });
    } finally { database.close(); }
  });
  const encoded = await binary();
  expect(encoded.slice(0, 2)).toEqual([255, 216]);
  await page.reload();
  await expect(page.getByRole('tab', { name: 'JPEG', exact: true })).toHaveAttribute('aria-selected', 'true');
  await expect(panel).toHaveAttribute('data-phase', 'done');
  await expect(panel.getByLabel(/^Quality/)).toHaveValue('61');
  await expect(page.getByLabel('Image comparison zoom')).toHaveValue('2');
  await expect(page.getByLabel('Image comparison split')).toHaveValue('72');
  expect(await page.locator('[data-result-format="jpeg"]').innerText()).toBe(metrics);
  expect(await binary()).toEqual(encoded);
  expect(await page.getByLabel('Right JPEG image', { exact: true }).evaluate(canvas => (canvas as HTMLCanvasElement).toDataURL())).toBe(pixels);
  const downloadEvent = page.waitForEvent('download');
  await panel.getByRole('button', { name: 'Download JPEG', exact: true }).click();
  const download = await downloadEvent;
  const stream = await download.createReadStream();
  const chunks: Buffer[] = []; for await (const chunk of stream!) chunks.push(chunk);
  expect(Array.from(Buffer.concat(chunks))).toEqual(encoded);

  // New images invalidate old outputs atomically but retain settings.
  await upload(page, 'replacement.png');
  await saved(page, { quality: 61, filename: 'replacement.png', formats: [] });
  await page.reload();
  await expect(panel).toHaveAttribute('data-phase', 'idle');
  await expect(panel.getByLabel(/^Quality/)).toHaveValue('61');
  await expect(panel.getByRole('button', { name: 'Download JPEG', exact: true })).toHaveCount(0);
  await expect(page.getByLabel('Original image', { exact: true })).toHaveAttribute('width', '32');
});

test('unavailable browser storage warns but does not prevent editing or uploading', async ({ page }) => {
  await page.addInitScript(() => {
    Object.defineProperty(window, 'indexedDB', { get() { throw new Error('Storage denied'); } });
  });
  await page.goto('/');
  await expect(page.getByRole('alert')).toContainText('Browser storage is unavailable');
  await upload(page);
  await page.getByRole('tab', { name: 'JPEG', exact: true }).click();
  await expect(page.locator('[data-codec="jpeg"]').getByRole('button', { name: 'Encode', exact: true })).toBeEnabled();
});

test('failed writes warn without breaking controls', async ({ page }) => {
  await page.addInitScript(() => {
    const put = IDBObjectStore.prototype.put;
    IDBObjectStore.prototype.put = function (...args) {
      if (this.transaction.db.name === 'image-compression-state') {
        throw new DOMException('Quota exceeded', 'QuotaExceededError');
      }
      return put.apply(this, args);
    };
  });
  await page.goto('/');
  await page.getByRole('tab', { name: 'JPEG', exact: true }).click();
  await expect(page.getByRole('alert')).toContainText('Browser storage is unavailable or full');
  await upload(page);
  await expect(page.locator('[data-codec="jpeg"]').getByRole('button', { name: 'Encode', exact: true })).toBeEnabled();
});

test('preferences survive reload even without an image', async ({ page }) => {
  await page.goto('/');
  await page.getByRole('tab', { name: 'JPEG', exact: true }).click();
  await page.locator('[data-codec="jpeg"]').getByLabel(/^Quality/).fill('43');
  await saved(page, { quality: 43, formats: [] });
  await page.reload();
  await expect(page.getByRole('tab', { name: 'JPEG', exact: true })).toHaveAttribute('aria-selected', 'true');
  await expect(page.locator('[data-codec="jpeg"]').getByLabel(/^Quality/)).toHaveValue('43');
});
