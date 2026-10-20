<script setup lang="ts">
import { nextTick, ref, watch } from 'vue';
import { downscaleOptions } from '../lib/limits';

const props = defineProps<{
  source?: ImageData;
  downscaled?: boolean;
  filename?: string;
  width?: number;
  height?: number;
  disabled: boolean;
}>();
const size = defineModel<number>('size', { required: true });
const emit = defineEmits<{ choose: [file: File]; resize: [] }>();
const input = ref<HTMLInputElement>();
const dragging = ref(false);
const preview = ref<HTMLCanvasElement | null>(null);
watch(() => props.source, async (image, _previous, onCleanup) => {
  let stale = false;
  onCleanup(() => { stale = true; });
  await nextTick();
  if (!image || !preview.value) return;
  const scale = Math.min(1, 96 * window.devicePixelRatio / Math.max(image.width, image.height));
  const bitmap = await createImageBitmap(image, {
    resizeWidth: Math.max(1, Math.round(image.width * scale)),
    resizeHeight: Math.max(1, Math.round(image.height * scale)),
    resizeQuality: 'high',
  });
  try {
    if (stale || !preview.value) return;
    preview.value.width = bitmap.width;
    preview.value.height = bitmap.height;
    preview.value.getContext('2d')!.drawImage(bitmap, 0, 0);
  } finally { bitmap.close(); }
}, { immediate: true });

function chooseSize(value: number) {
  if (props.disabled || size.value === value) return;
  size.value = value;
  emit('resize');
}

function choose(files?: FileList | null) {
  if (!props.disabled && files?.[0]) emit('choose', files[0]);
  if (input.value) input.value.value = '';
}
function drop(event: DragEvent) {
  dragging.value = false;
  choose(event.dataTransfer?.files);
}
</script>

<template>
  <section class="input-bar" aria-label="Image input" :data-width="width" :data-height="height" :class="{ dragging }"
    @dragover.prevent="dragging = !disabled"
    @dragleave.prevent="dragging = false"
    @drop.prevent="drop"
  >
    <div class="dropzone">
      <div v-if="filename || (width && height)" class="file-description">
        <strong v-if="filename">{{ filename }}</strong>
        <span v-if="width && height">{{ width }} × {{ height }}</span>
      </div>
      <button :disabled="disabled" @click="input?.click()"><svg viewBox="0 0 20 20" aria-hidden="true"><path d="M10 13V3m-4 4 4-4 4 4M3 13v4h14v-4" /></svg>Select image</button>
      <input
        ref="input"
        class="file-input"
        type="file"
        accept="image/*"
        aria-label="Choose image file"
        :disabled="disabled"
        @change="choose(($event.target as HTMLInputElement).files)"
      >
    </div>
    <figure class="source-preview">
      <canvas v-if="source" ref="preview" :aria-label="downscaled ? 'Downscaled image' : 'Original image'"></canvas>
      <div v-else class="preview-placeholder" aria-hidden="true"><svg viewBox="0 0 48 48"><rect x="7" y="8" width="34" height="32" rx="5" /><circle cx="18" cy="19" r="4" /><path d="m8 34 10-10 9 9 6-6 8 9" /></svg></div>
    </figure>
    <div class="downscale-control" role="group" aria-labelledby="downscale-label">
      <span id="downscale-label">Downscale image</span>
      <div class="size-options">
        <button v-for="option in downscaleOptions" :key="option.value" :aria-pressed="size === option.value" :disabled="disabled" @click="chooseSize(option.value)">{{ option.label }}</button>
      </div>
    </div>
  </section>
</template>

<style scoped>
.input-bar { display: grid; grid-template-columns: 80px minmax(0, 1fr); align-items: center; width: 100%; gap: 16px; padding: 16px; background: var(--paper); border: 1px solid var(--line); border-radius: var(--radius); box-shadow: var(--card-shadow); }
.input-bar.dragging { border-color: var(--accent); outline: 3px dashed var(--accent); outline-offset: 4px; background: var(--accent-soft); }
.dropzone { grid-column: 2; grid-row: 1; display: grid; align-content: center; gap: 8px; min-width: 0; }
.dropzone button { display: flex; align-items: center; justify-content: center; gap: 8px; padding: 12px 14px; color: var(--button-ink); background: var(--button-bg); border: 1px solid transparent; border-radius: 8px; font-size: 13px; font-weight: 600; }
.dropzone button:hover:not(:disabled) { filter: brightness(1.1); }
.dropzone button svg { width: 16px; height: 16px; fill: none; stroke: currentColor; stroke-width: 1.5; stroke-linecap: round; stroke-linejoin: round; }
.file-description { display: grid; gap: 5px; min-width: 0; }
.file-description strong { font-size: 12px; font-weight: 500; overflow-wrap: anywhere; }
.file-description span { color: var(--muted); font-size: 11px; line-height: 1.6; }
.file-input { position: absolute; width: 1px; height: 1px; overflow: hidden; clip-path: inset(50%); }
.source-preview { grid-column: 1; grid-row: 1; margin: 0; text-align: center; }
.source-preview canvas, .preview-placeholder { display: block; width: 80px; height: 90px; object-fit: contain; border: 1px dashed var(--line); border-radius: 9px; background: var(--surface); }
.preview-placeholder { display: grid; place-items: center; color: var(--muted); }
.preview-placeholder svg { width: 36px; height: 36px; fill: none; stroke: var(--accent); stroke-width: 1.4; stroke-linejoin: round; }
.downscale-control { grid-column: 1 / -1; display: grid; gap: 9px; font-size: 11px; color: var(--muted); }
.size-options { display: grid; grid-template-columns: 1.8fr repeat(4, 1fr); overflow: hidden; padding: 3px; gap: 2px; border: 1px solid var(--line); border-radius: 8px; background: var(--surface); }
.size-options button { min-width: 0; min-height: 38px; padding: 8px 3px; border: 0; border-radius: 5px; color: var(--muted); background: transparent; font-size: 11px; }
.size-options button[aria-pressed="true"] { color: var(--accent); background: var(--paper); box-shadow: 0 1px 3px #20352c10; font-weight: 600; }
@media (max-width: 380px) { .input-bar { padding: 16px; gap: 16px 12px; grid-template-columns: 64px minmax(0, 1fr); } .source-preview canvas, .preview-placeholder { width: 64px; height: 80px; } .size-options button { font-size: 10px; } }
</style>
