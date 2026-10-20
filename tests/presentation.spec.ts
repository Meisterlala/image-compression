import { expect, test, type Locator } from '@playwright/test';

async function actionStyle(actions: Locator) {
  return actions.evaluate(element => {
    const style = getComputedStyle(element);
    return {
      border: style.border,
      radius: style.borderRadius,
      background: style.backgroundColor,
      padding: style.padding,
      minHeight: style.minHeight,
    };
  });
}

test('metadata and icons are available without JavaScript', async ({ request }) => {
  const response = await request.get('/');
  const html = await response.text();
  expect(html).toContain('<title>Image compression — Compare JPEG, AVIF, JPEG XL &amp; AI</title>');
  expect(html).toContain('name="description"');
  expect(html).toContain('rel="canonical" href="https://compress.meisterlala.dev/"');
  expect(html).not.toMatch(/Compression Lab|og:|twitter:|social-preview/);
  const structured = JSON.parse(html.match(/<script type="application\/ld\+json">([\s\S]*?)<\/script>/)![1]!);
  expect(structured['@type']).toBe('WebApplication');
  expect(structured.name).toBe('Image compression');
  expect(structured.url).toBe('https://compress.meisterlala.dev/');
  const svg = await request.get('/icon.svg');
  expect(svg.ok()).toBeTruthy();
  expect(svg.headers()['content-type']).toContain('image/svg+xml');
  const ico = await request.get('/favicon.ico');
  expect((await ico.body()).subarray(0, 4)).toEqual(Buffer.from([0, 0, 1, 0]));
  const touch = await request.get('/apple-touch-icon.png');
  const png = await touch.body();
  expect(png.subarray(0, 8).toString('hex')).toBe('89504e470d0a1a0a');
  expect(png.readUInt32BE(16)).toBe(180);
  expect(png.readUInt32BE(20)).toBe(180);
  expect(await (await request.get('/robots.txt')).text()).toContain('Sitemap: https://compress.meisterlala.dev/sitemap.xml');
  expect(await (await request.get('/sitemap.xml')).text()).toContain('<loc>https://compress.meisterlala.dev/</loc>');
});

test('all codecs use the same action container on small and large screens in both themes', async ({ page }) => {
  await page.goto('/');
  await expect(page.getByRole('heading', { level: 1 })).toHaveText('Image compression');
  await expect(page.getByRole('button', { name: /sample/i })).toHaveCount(0);
  await expect(page.getByText('Runs locally in your browser. No image is uploaded to a server.', { exact: true })).toBeVisible();
  await expect(page.getByRole('link', { name: /GitHub|source|legal|licenses/i })).toHaveCount(0);
  await expect(page.getByText('Your image is processed only locally in your browser.', { exact: true })).toHaveCount(0);
  await expect(page.getByText('or drag and drop it here', { exact: true })).toHaveCount(0);
  for (const colorScheme of ['light', 'dark'] as const) {
    await page.emulateMedia({ colorScheme });
    for (const width of [320, 390, 768, 1440]) {
      await page.setViewportSize({ width, height: 950 });
      let expectedStyle: Awaited<ReturnType<typeof actionStyle>> | undefined;
      for (const name of ['JPEG AI', 'JPEG', 'AVIF', 'JPEG XL', 'Cool-chic 5.0']) {
        const tab = page.getByRole('tab', { name, exact: true });
        await tab.click();
        const actions = page.getByRole('group', { name: `${name} encoding actions`, exact: true });
        const style = await actionStyle(actions);
        expectedStyle ??= style;
        expect(style).toEqual(expectedStyle);
        expect(style.border).not.toContain('0px');
        await expect(actions.getByRole('button', { name: 'Encode', exact: true })).toBeDisabled();
        const box = (await tab.boundingBox())!;
        expect(box.height).toBeGreaterThanOrEqual(44);
        expect(box.x).toBeGreaterThanOrEqual(0);
        expect(box.x + box.width).toBeLessThanOrEqual(width);
        expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBeTruthy();
      }
    }
  }
});

test('encoding, cancellation and downloads stay inside the same styled action area', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/');
  await page.getByRole('tab', { name: 'JPEG', exact: true }).click();
  const actions = page.getByRole('group', { name: 'JPEG encoding actions', exact: true });
  const initialStyle = await actionStyle(actions);
  const image = await page.evaluate(() => {
    const canvas = document.createElement('canvas');
    canvas.width = 96; canvas.height = 64;
    const context = canvas.getContext('2d')!;
    context.fillStyle = '#528467'; context.fillRect(0, 0, 96, 64);
    context.fillStyle = '#e8bf63'; context.fillRect(12, 10, 30, 28);
    return canvas.toDataURL('image/png').split(',')[1]!;
  });
  await page.getByLabel('Choose image file', { exact: true }).setInputFiles({ name: 'test-image.png', mimeType: 'image/png', buffer: Buffer.from(image, 'base64') });
  await expect(actions.getByRole('button', { name: 'Encode', exact: true })).toBeEnabled();
  const description = (await page.locator('.file-description').boundingBox())!;
  const uploadButton = (await page.getByRole('button', { name: 'Select image', exact: true }).boundingBox())!;
  expect(description.y + description.height).toBeLessThanOrEqual(uploadButton.y);

  let release!: () => void;
  const pending = new Promise<void>(resolve => { release = resolve; });
  await page.route('**/assets/codec.worker-*.js', async route => { await pending; await route.abort().catch(() => {}); });
  await actions.getByRole('button', { name: 'Encode', exact: true }).click();
  await expect(actions.getByRole('button', { name: 'Cancel encoding', exact: true })).toBeVisible();
  expect(await actionStyle(actions)).toEqual(initialStyle);
  await actions.getByRole('button', { name: 'Cancel encoding', exact: true }).click();
  await expect(actions.getByRole('button', { name: 'Encode', exact: true })).toBeVisible();
  expect(await actionStyle(actions)).toEqual(initialStyle);
  release();
  await page.unroute('**/assets/codec.worker-*.js');

  await actions.getByRole('button', { name: 'Encode', exact: true }).click();
  await expect(actions.getByRole('button', { name: 'Encode again', exact: true })).toBeVisible();
  await expect(actions.getByRole('button', { name: 'Download JPEG', exact: true })).toBeVisible();
  await expect(actions.getByRole('button', { name: 'Download lossless PNG', exact: true })).toBeVisible();
  expect(await actionStyle(actions)).toEqual(initialStyle);
  await expect(page.getByLabel('Right JPEG image', { exact: true })).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBeTruthy();
  await page.reload();
  await expect(actions.getByRole('button', { name: 'Encode again', exact: true })).toBeVisible();
  expect(await actionStyle(actions)).toEqual(initialStyle);
  expect(errors).toEqual([]);
});

test('keyboard users can skip to controls and navigate the codec tabs', async ({ page }) => {
  await page.goto('/');
  await expect(page.getByRole('heading', { level: 1 })).toBeVisible();
  await page.keyboard.press('Tab');
  await expect(page.getByRole('link', { name: 'Skip to the playground' })).toBeFocused();
  await page.keyboard.press('Enter');
  await expect(page.locator('#playground')).toBeFocused();
  await page.keyboard.press('Tab');
  await expect(page.getByRole('tab', { name: 'JPEG AI', exact: true })).toBeFocused();
  await page.keyboard.press('End');
  await expect(page.getByRole('tab', { name: 'Cool-chic 5.0', exact: true })).toBeFocused();
  await page.keyboard.press('Home');
  await expect(page.getByRole('tab', { name: 'JPEG AI', exact: true })).toHaveAttribute('aria-selected', 'true');
});
