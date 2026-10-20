import { expect, test } from '@playwright/test';

test('lossless PNG download preserves the decoded image pixels', async ({ page }) => {
  await page.goto('/');
  const png = await page.evaluate(() => {
    const canvas = document.createElement('canvas');
    canvas.width = 16; canvas.height = 16;
    const context = canvas.getContext('2d')!;
    context.fillStyle = '#b35c29';
    context.fillRect(0, 0, 16, 16);
    return canvas.toDataURL('image/png').split(',')[1]!;
  });
  await page.getByLabel('Choose image file', { exact: true }).setInputFiles({
    name: 'sample.png', mimeType: 'image/png', buffer: Buffer.from(png, 'base64'),
  });
  await page.getByRole('tab', { name: 'JPEG', exact: true }).click();
  const pane = page.locator('[data-codec="jpeg"]');
  await pane.getByRole('button', { name: 'Encode', exact: true }).click();
  const button = pane.getByRole('button', { name: 'Download lossless PNG', exact: true });
  await expect(button).toBeVisible();
  const expected = await page.getByLabel('Right JPEG image', { exact: true }).evaluate(element => {
    const canvas = element as HTMLCanvasElement;
    return { width: canvas.width, height: canvas.height, pixels: Array.from(canvas.getContext('2d')!.getImageData(0, 0, canvas.width, canvas.height).data) };
  });
  const pending = page.waitForEvent('download');
  await button.click();
  const download = await pending;
  expect(download.suggestedFilename()).toBe('decoded-lossless.png');
  const stream = await download.createReadStream();
  const chunks: Buffer[] = [];
  for await (const chunk of stream) chunks.push(Buffer.from(chunk));
  const bytes = Buffer.concat(chunks);
  expect(bytes.subarray(0, 8)).toEqual(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]));
  const actual = await page.evaluate(async base64 => {
    const bytes = Uint8Array.from(atob(base64), char => char.charCodeAt(0));
    const blob = new Blob([bytes], { type: 'image/png' });
    const image = await createImageBitmap(blob);
    const canvas = document.createElement('canvas');
    canvas.width = image.width; canvas.height = image.height;
    const context = canvas.getContext('2d')!;
    context.drawImage(image, 0, 0);
    return { width: canvas.width, height: canvas.height, pixels: Array.from(context.getImageData(0, 0, canvas.width, canvas.height).data) };
  }, bytes.toString('base64'));
  expect(actual).toEqual(expected);
});
