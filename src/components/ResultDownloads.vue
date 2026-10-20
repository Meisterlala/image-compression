<script setup lang="ts">
import { formats, type Result } from '../types';
import { download, draw } from '../lib/image';

const props = defineProps<{ result: Result; file?: File }>();
function saveCompressed() {
  const result = props.result;
  const format = formats.find(format => format.id === result.settings.format)!;
  const name = props.file?.name.replace(/\.[^.]+$/, '') ?? 'image';
  const point = result.settings.format === 'jpegai' ? `-${result.settings.operatingPoint}`
    : result.settings.format === 'coolchic' && result.settings.coolchicPoint ? `-${result.settings.coolchicPoint}` : '';
  download(new Blob([result.encoded], { type: format.mime }), `${name}${point}-${result.width}x${result.height}.${format.extension}`);
}
function saveLosslessPng() {
  const result = props.result;
  const canvas = document.createElement('canvas');
  draw(canvas, new ImageData(result.pixels, result.width, result.height));
  canvas.toBlob(blob => { if (blob) download(blob, 'decoded-lossless.png'); }, 'image/png');
}
</script>

<template>
  <div class="download-row" role="group" aria-label="Image downloads">
    <button @click="saveCompressed">Download {{ formats.find(format => format.id === result.settings.format)?.label }}</button>
    <button @click="saveLosslessPng">Download lossless PNG</button>
  </div>
</template>

<style scoped>
.download-row {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 8px;
  width: 100%;
  margin-top: 8px;
  padding-top: 16px;
  border-top: 1px solid var(--line);
}

button {
  padding: 10px 14px;
  min-height: 44px;
  border: 1px solid var(--line);
  border-radius: 8px;
  color: var(--ink);
  background: var(--paper);
  font-size: 12px;
}
</style>
