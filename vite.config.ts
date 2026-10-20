import { defineConfig } from 'vitest/config';
import vue from '@vitejs/plugin-vue';

const isolationHeaders = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
};

export default defineConfig({
  plugins: [vue()],
  optimizeDeps: { include: ['@tensorflow/tfjs-core', '@tensorflow/tfjs-core/dist/public/chained_ops/register_all_chained_ops', '@tensorflow/tfjs-core/dist/register_all_gradients', '@tensorflow/tfjs-backend-webgpu', '@tensorflow/tfjs-backend-webgpu/dist/kernels/AddN'] },
  server: { headers: isolationHeaders, port: 5173 },
  preview: { headers: isolationHeaders, port: 4173 },
  worker: { format: 'es' },
  build: { target: 'es2022', assetsInlineLimit: 0 },
  test: { include: ['src/**/*.test.ts'] },
});
