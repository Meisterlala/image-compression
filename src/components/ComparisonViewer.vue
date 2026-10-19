<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue';
import { draw } from '../lib/image';
import type { Result } from '../types';

const props = defineProps<{ source?: ImageData; result?: Result; format: string }>();
const original = ref<HTMLCanvasElement | null>(null);
const decoded = ref<HTMLCanvasElement | null>(null);
const divider = ref(50);
const splitting = ref(false);
function updateSplit(event: PointerEvent) {
  const bounds = original.value?.getBoundingClientRect();
  if (bounds) divider.value = Math.max(0, Math.min(100, (event.clientX - bounds.left) / bounds.width * 100));
}
function startSplit(event: PointerEvent) {
  splitting.value = true;
  (event.currentTarget as HTMLElement).setPointerCapture(event.pointerId);
  updateSplit(event);
}
const zoom = ref('fit');
const dpr = ref(window.devicePixelRatio);
const stageStyle = computed(() => {
  if (!props.source) return {};
  return zoom.value === 'fit'
    ? { width: `${Math.min(props.source.width, 252 * props.source.width / props.source.height)}px`, maxWidth: '100%' }
    : { width: `${props.source.width * Number(zoom.value) / dpr.value}px`, maxWidth: 'none' };
});
async function render() {
  await nextTick();
  if (props.source) draw(original.value, props.source);
  if (props.result) draw(decoded.value, new ImageData(props.result.pixels, props.result.width, props.result.height));
}
const updateDpr = () => { dpr.value = window.devicePixelRatio; };
watch([() => props.source, () => props.result], render, { immediate: true });
onMounted(() => window.addEventListener('resize', updateDpr));
onBeforeUnmount(() => window.removeEventListener('resize', updateDpr));
</script>

<template>
  <section class="comparison" :aria-label="`Original versus ${format}`">
    <div class="preview-top">
      <h3>Original / {{ format }}</h3>
      <span v-if="result" class="preview-badge">Decoded output</span>
    </div>
    <div v-if="!source" class="empty-preview">
      <p>Choose an image to preview it.</p>
      <p>Compression results will appear here.</p>
    </div>
    <template v-else>
      <div class="viewer-toolbar">
        <span>{{ source.width }} × {{ source.height }} px</span>
        <label>View
          <select v-model="zoom" :aria-label="`${format} preview zoom`">
            <option value="fit">Fit</option>
            <option value="1">1:1 device pixels</option>
            <option value="2">2× device pixels</option>
            <option value="4">4× device pixels</option>
          </select>
        </label>
      </div>
      <div class="viewer-scroll" :class="{ fitting: zoom === 'fit' }">
        <div class="image-stage" :class="{ magnified: Number(zoom) >= 2 }" :style="stageStyle">
          <canvas ref="original" class="image-canvas" :aria-label="`Original working image for ${format}`"></canvas>
          <canvas
            v-show="result"
            ref="decoded"
            class="image-canvas decoded-canvas"
            :style="{ clipPath: `inset(0 0 0 ${divider}%)` }"
            :aria-label="`Decoded ${format} image`"
          ></canvas>
          <div
            v-if="result"
            class="image-divider"
            :style="{ left: `${divider}%` }"
            @pointerdown.prevent="startSplit"
            @pointermove="splitting && updateSplit($event)"
            @pointerup="splitting = false"
            @pointercancel="splitting = false"
          ><span>↔</span></div>
          <span class="image-label original-label">Original</span>
          <span v-if="result" class="image-label decoded-label">{{ format }}</span>
        </div>
      </div>
      <label v-if="result" class="divider-control">
        Comparison split
        <input v-model.number="divider" type="range" min="0" max="100" :aria-label="`${format} comparison split`">
      </label>
      <p v-else class="preview-hint">This is the normalized working image. Compress it to reveal the comparison.</p>
      <p v-if="zoom !== 'fit'" class="preview-hint">Scroll horizontally or vertically to pan. Zoom is measured in device pixels.</p>
    </template>
  </section>
</template>

<style scoped>
.comparison {
  overflow: hidden;
  border: 1px solid var(--line);
  border-radius: var(--radius);
  background: var(--paper);
}

.preview-top {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 16px;
  padding: 10px 12px;
  border-bottom: 1px solid var(--line);
}

h3 {
  margin: 0;
  font-size: 12px;
  font-weight: 500;
}

.preview-badge {
  color: var(--muted);
  font-size: 11px;
}

.empty-preview {
  display: flex;
  min-height: 328px;
  align-items: center;
  justify-content: center;
  flex-direction: column;
  background: var(--surface);
  color: var(--muted);
  font-size: 13px;
}

.empty-preview p {
  margin: 5px 16px;
}

.viewer-toolbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 12px 16px;
  font: 11px ui-monospace, monospace;
  color: var(--muted);
}

.viewer-toolbar select {
  max-width: 150px;
  margin-left: 5px;
  padding: 5px;
  border: 1px solid var(--line);
  border-radius: 4px;
  background: var(--surface);
  color: var(--ink);
  font-size: 11px;
}

.viewer-scroll {
  height: 280px;
  overflow: auto;
  padding: 14px;
  background: var(--viewer-background);
}

.viewer-scroll.fitting {
  display: flex;
  align-items: center;
}

.fitting .image-stage {
  flex-shrink: 0;
}

.image-stage {
  position: relative;
  margin: auto;
  line-height: 0;
}

.image-canvas {
  display: block;
  width: 100%;
  height: auto;
}

.magnified .image-canvas {
  image-rendering: pixelated;
}

.decoded-canvas {
  position: absolute;
  inset: 0;
}

.image-divider {
  position: absolute;
  top: 0;
  bottom: 0;
  width: 16px;
  transform: translateX(-50%);
  cursor: ew-resize;
  touch-action: none;
}

.image-divider::before {
  content: "";
  position: absolute;
  inset: 0 auto 0 7px;
  width: 2px;
  background: var(--divider);
  box-shadow: 0 0 3px #0008;
}

.image-divider span {
  position: absolute;
  top: 50%;
  left: -6px;
  display: grid;
  place-items: center;
  width: 28px;
  height: 28px;
  border-radius: 50%;
  background: var(--divider);
  color: var(--divider-ink);
  font-size: 17px;
  line-height: 1;
  box-shadow: 0 1px 5px #0004;
}

.image-label {
  position: absolute;
  top: 10px;
  padding: 6px 8px;
  border-radius: 4px;
  background: #142b24c9;
  color: white;
  font-size: 10px;
  line-height: 1;
  pointer-events: none;
}

.original-label {
  left: 10px;
}

.decoded-label {
  right: 10px;
}

.divider-control {
  display: flex;
  align-items: center;
  gap: 15px;
  padding: 12px 18px;
  color: var(--muted);
  font-size: 11px;
}

.divider-control input {
  flex: 1;
  min-width: 0;
  accent-color: var(--accent);
}

.preview-hint {
  margin: 12px 18px;
  color: var(--muted);
  font-size: 11px;
  line-height: 1.5;
}

@container (max-width: 420px) {
  .viewer-toolbar {
    flex-wrap: wrap;
    gap: 8px;
    padding: 10px;
  }

  .viewer-toolbar label {
    display: flex;
    align-items: center;
    gap: 5px;
  }

  .viewer-toolbar select {
    max-width: 120px;
    margin-left: 0;
  }
}
</style>
