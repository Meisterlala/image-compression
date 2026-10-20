import { test, expect } from '@playwright/test';

for (const mode of ['missing', 'no-adapter', 'device-failure', 'available'] as const) {
  test(`GPU controls reflect ${mode} availability in both codecs`, async ({ page }) => {
    await page.addInitScript(mode => {
      Object.defineProperty(navigator, 'gpu', { configurable: true, value: mode === 'missing' ? undefined : {
        requestAdapter: async () => mode === 'no-adapter' ? null : {
          requestDevice: async () => {
            if (mode === 'device-failure') throw new Error('Device unavailable');
            return { destroy() {} };
          },
        },
      } });
    }, mode);
    await page.emulateMedia({ colorScheme: 'light' });
    await page.goto('/');
    for (const [tab, label] of [['JPEG AI', 'GPU decoding'], ['Cool-chic 5.0', 'GPU training']]) {
      await page.getByRole('tab', { name: tab, exact: true }).click();
      const toggle = page.getByRole('checkbox', { name: label, exact: true });
      if (mode === 'available') {
        await expect(toggle).toBeEnabled();
        await expect(toggle).toBeChecked();
        await expect(page.locator('.decode-setting > span:visible')).toContainText('Will try WebGPU');
        await toggle.uncheck();
        await expect(page.locator('.decode-setting > span:visible')).toHaveText('Using CPU only');
        await toggle.check();
      } else {
        await expect(toggle).toBeDisabled();
        await expect(toggle).not.toBeChecked();
        const warning = page.locator('.decode-setting > .gpu-unavailable:visible');
        await expect(warning).toHaveText('WebGPU is disabled or unavailable in your browser. Using CPU only.');
        await expect(warning).toHaveCSS('color', 'rgb(138, 58, 46)');
      }
    }
    await expect(page.getByRole('region', { name: 'Training backend' })).toHaveCount(0);
  });
}
