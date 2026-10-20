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
  if (files?.[0]) emit('choose', files[0]);
  if (input.value) input.value.value = '';
}
function drop(event: DragEvent) {
  dragging.value = false;
  choose(event.dataTransfer?.files);
}
</script>

<template>
  <section class="input-bar" aria-label="Image input" :data-width="width" :data-height="height">
    <div
      class="dropzone"
      :class="{ dragging }"
      @dragover.prevent="dragging = true"
      @dragleave.prevent="dragging = false"
      @drop.prevent="drop"
    >
      <button :disabled="disabled" @click="input?.click()">Select image</button>
      <div class="file-description">
        <strong v-if="filename">{{ filename }}</strong>
        <span v-if="width && height">{{ width }} × {{ height }}</span>
        <span>Your image is processed only locally in your browser.</span>
      </div>
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
      <div v-else class="preview-placeholder" aria-hidden="true">＋</div>
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
.input-bar {
  display: grid;
  grid-template-columns: 96px minmax(0, 1fr);
  align-items: center;
  width: 440px;
  max-width: 100%;
  gap: 14px;
  padding: 14px;
  background: var(--paper);
  border: 1px solid var(--line);
  border-radius: var(--radius);
}

.dropzone {
  grid-column: 2;
  grid-row: 1;
  display: grid;
  align-content: center;
  gap: 10px;
  max-width: 220px;
  min-width: 0;
  padding: 0;
  border-radius: 5px;
}

.dropzone.dragging {
  background: var(--accent-soft);
  border-color: var(--accent);
}

.dropzone button {
  flex-shrink: 0;
  padding: 12px 16px;
  color: var(--button-ink);
  background: var(--button-bg);
  border: 1px solid var(--line);
  border-radius: 5px;
  font-size: 13px;
}

.file-description {
  display: grid;
  gap: 5px;
  min-width: 0;
}

.file-description strong {
  font-size: 13px;
  font-weight: 500;
  overflow-wrap: anywhere;
}

.file-description span {
  color: var(--muted);
  font-size: 11px;
}

.file-input {
  position: absolute;
  width: 1px;
  height: 1px;
  overflow: hidden;
  clip-path: inset(50%);
}

.source-preview {
  grid-column: 1;
  grid-row: 1;
  margin: 0;
  text-align: center;
}

.source-preview canvas,
.preview-placeholder {
  display: block;
  width: 96px;
  height: 96px;
  object-fit: contain;
  border: 1px solid var(--line);
  border-radius: 5px;
  background: var(--surface);
}

.preview-placeholder {
  display: grid;
  place-items: center;
  color: var(--muted);
  font-size: 28px;
}

.downscale-control {
  grid-column: 1 / -1;
  display: grid;
  gap: 8px;
  font-size: 12px;
}

.size-options {
  display: grid;
  grid-template-columns: 1.8fr repeat(4, 1fr);
  overflow: hidden;
  border: 1px solid var(--line);
  border-radius: 5px;
}

.size-options button {
  min-width: 0;
  padding: 9px 5px;
  border: 0;
  border-right: 1px solid var(--line);
  color: var(--muted);
  background: var(--surface);
  font-size: 12px;
}

.size-options button:last-child {
  border-right: 0;
}

.size-options button[aria-pressed="true"] {
  color: var(--button-ink);
  background: var(--button-bg);
  font-weight: 600;
}

@media (max-width: 560px) {
  .input-bar {
    width: 100%;
  }

}
</style>
