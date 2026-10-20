import { test, expect } from '@playwright/test';

for (const [width, height] of [[160, 90], [90, 160]]) {
  test(`fit enlarges a small ${width}×${height} image to the available space`, async ({ page }) => {
    await page.setViewportSize({ width: 1280, height: 900 });
    await page.goto('/');
    const png = await page.evaluate(({ width, height }) => {
      const canvas = document.createElement('canvas');
      canvas.width = width;
      canvas.height = height;
      canvas.getContext('2d')!.fillRect(0, 0, width, height);
      return canvas.toDataURL().split(',')[1]!;
    }, { width, height });
    await page.getByLabel('Choose image file', { exact: true }).setInputFiles({
      name: 'small.png', mimeType: 'image/png', buffer: Buffer.from(png, 'base64'),
    });
    await page.getByRole('tab', { name: 'JPEG', exact: true }).click();
    await page.locator('[data-codec="jpeg"]').getByRole('button', { name: /^Encode(?: again)?$/ }).click();
    const zoom = page.getByLabel('Image comparison zoom', { exact: true });
    await expect(zoom).toBeVisible();
    await zoom.selectOption('fit');

    for (const viewportSize of [{ width: 1280, height: 900 }, { width: 390, height: 844 }]) {
      await page.setViewportSize(viewportSize);
      await expect.poll(async () => page.locator('.viewer-scroll').evaluate(el => {
        const style = getComputedStyle(el);
        const availableWidth = el.clientWidth - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight);
        const availableHeight = el.clientHeight - parseFloat(style.paddingTop) - parseFloat(style.paddingBottom);
        const canvas = el.querySelector('canvas')!;
        const bounds = canvas.getBoundingClientRect();
        const scale = Math.min(availableWidth / canvas.width, availableHeight / canvas.height);
        return Math.max(Math.abs(bounds.width - canvas.width * scale), Math.abs(bounds.height - canvas.height * scale));
      })).toBeLessThan(1);
    }
  });
}

test('comparison supports drag pan, anchored pinch zoom and independent split dragging', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/');
  const png = await page.evaluate(() => {
    const canvas = document.createElement('canvas');
    canvas.width = 1200; canvas.height = 900;
    canvas.getContext('2d')!.fillRect(0, 0, 1200, 900);
    return canvas.toDataURL().split(',')[1]!;
  });
  await page.getByLabel('Choose image file', { exact: true }).setInputFiles({
    name: 'large.png', mimeType: 'image/png', buffer: Buffer.from(png, 'base64'),
  });
  await page.getByRole('tab', { name: 'JPEG', exact: true }).click();
  await page.locator('[data-codec="jpeg"]').getByRole('button', { name: /^Encode(?: again)?$/ }).click();
  const zoom = page.getByLabel('Image comparison zoom', { exact: true });
  await expect(zoom).toBeVisible();
  await zoom.selectOption('2');
  const viewport = page.locator('.viewer-scroll');
  await viewport.scrollIntoViewIfNeeded();
  const bounds = (await viewport.boundingBox())!;
  const x = bounds.x + 100, y = bounds.y + 100;
  const split = page.getByLabel('Image comparison split', { exact: true });
  const initialSplit = await split.inputValue();
  await page.mouse.move(x, y);
  await page.mouse.down();
  await page.mouse.move(x - 60, y - 50, { steps: 5 });
  await page.mouse.up();
  expect(await viewport.evaluate(el => el.scrollLeft)).toBeGreaterThan(40);
  expect(await viewport.evaluate(el => el.scrollTop)).toBeGreaterThan(30);
  await expect(split).toHaveValue(initialSplit);

  const widthBeforeWheel = await page.locator('.image-stage').evaluate(el => el.getBoundingClientRect().width);
  await page.mouse.move(x, y);
  await page.mouse.wheel(0, -40);
  await expect.poll(() => page.locator('.image-stage').evaluate(el => el.getBoundingClientRect().width)).toBeGreaterThan(widthBeforeWheel);
  await expect(split).toHaveValue(initialSplit);
  const beforeLargeWheel = Number(await zoom.inputValue());
  await page.mouse.wheel(0, -2000);
  await expect.poll(async () => Number(await zoom.inputValue())).toBeGreaterThan(beforeLargeWheel);
  expect(Number(await zoom.inputValue())).toBeLessThan(beforeLargeWheel * 1.07);

  await zoom.selectOption('fit');
  // Real touch events exercise pointer capture and the browser's touch-action behavior.
  const cdp = await page.context().newCDPSession(page);
  const touchBounds = (await viewport.boundingBox())!;
  const tx = touchBounds.x + 45;
  const touch = (id: number, px: number) => ({ id, x: px, y: touchBounds.y + touchBounds.height / 2 });
  const initialWidth = await page.locator('.image-stage').evaluate(el => el.getBoundingClientRect().width);
  await cdp.send('Input.dispatchTouchEvent', { type: 'touchStart', touchPoints: [touch(1, tx), touch(2, tx + 80)] });
  await cdp.send('Input.dispatchTouchEvent', { type: 'touchMove', touchPoints: [touch(1, tx - 25), touch(2, tx + 110)] });
  await cdp.send('Input.dispatchTouchEvent', { type: 'touchEnd', touchPoints: [] });
  await expect.poll(() => page.locator('.image-stage').evaluate(el => el.getBoundingClientRect().width)).toBeGreaterThan(initialWidth * 1.4);
  await expect(split).toHaveValue(initialSplit);
  await expect(viewport).not.toHaveClass(/panning/);

  await zoom.selectOption('fit');
  const handle = page.locator('.image-divider span');
  const handleBounds = (await handle.boundingBox())!;
  await page.mouse.move(handleBounds.x + 14, handleBounds.y + 14);
  await page.mouse.down();
  await page.mouse.move(handleBounds.x + 55, handleBounds.y + 14, { steps: 5 });
  await page.mouse.up();
  expect(Number(await split.inputValue())).toBeGreaterThan(Number(initialSplit) + 5);
  expect(await viewport.evaluate(el => el.scrollLeft)).toBe(0);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
});
