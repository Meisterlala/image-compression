<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue';
import { draw } from '../lib/image';
import { usePersistentRef } from '../lib/persistence';

const props = defineProps<{ left: ImageData; right: ImageData; leftLabel: string; rightLabel: string }>();
const original = ref<HTMLCanvasElement | null>(null);
const decoded = ref<HTMLCanvasElement | null>(null);
const divider = usePersistentRef('comparison-divider', 50);
const splitPointer = ref<number | null>(null);
const viewport = ref<HTMLDivElement | null>(null);
const stage = ref<HTMLDivElement | null>(null);
const pointers = new Map<number, { x: number; y: number }>();
const panning = ref(false);
function updateSplit(event: PointerEvent) {
  const bounds = original.value?.getBoundingClientRect();
  if (bounds) divider.value = Math.max(0, Math.min(100, (event.clientX - bounds.left) / bounds.width * 100));
}
function startSplit(event: PointerEvent) {
  if (splitPointer.value !== null) return;
  splitPointer.value = event.pointerId;
  (event.currentTarget as HTMLElement).setPointerCapture(event.pointerId);
  updateSplit(event);
}
const zoom = usePersistentRef('comparison-zoom', 'fit');
const dpr = ref(window.devicePixelRatio);
const viewportHeight = ref(window.innerHeight);
const stageStyle = computed(() => {
  return zoom.value === 'fit'
    ? { width: `${(Math.max(320, Math.min(viewportHeight.value * 0.52, 660)) - 28) * props.left.width / props.left.height}px`, maxWidth: '100%' }
    : { width: `${props.left.width * Number(zoom.value) / dpr.value}px`, maxWidth: 'none' };
});
const customZoom = computed(() => !['fit', '1', '2', '4'].includes(zoom.value));

// Keep the image point beneath the gesture stationary as its rendered size changes.
async function zoomAt(scale: number, from: { x: number; y: number }, to = from) {
  const bounds = stage.value?.getBoundingClientRect();
  if (!bounds || !viewport.value) return;
  const x = (from.x - bounds.left) / bounds.width;
  const y = (from.y - bounds.top) / bounds.height;
  zoom.value = String(Math.max(0.05, Math.min(16, scale)));
  await nextTick();
  const resized = stage.value?.getBoundingClientRect();
  if (!resized || !viewport.value) return;
  viewport.value.scrollLeft += resized.left + x * resized.width - to.x;
  viewport.value.scrollTop += resized.top + y * resized.height - to.y;
}
function startPan(event: PointerEvent) {
  if (event.button !== 0 || splitPointer.value !== null) return;
  pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
  (event.currentTarget as HTMLElement).setPointerCapture(event.pointerId);
  panning.value = true;
}
function movePan(event: PointerEvent) {
  const previous = pointers.get(event.pointerId);
  if (!previous || !viewport.value) return;
  const before = [...pointers.values()];
  pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
  const after = [...pointers.values()];
  if (before.length === 1) {
    viewport.value.scrollLeft += previous.x - event.clientX;
    viewport.value.scrollTop += previous.y - event.clientY;
  } else if (before.length === 2) {
    const distance = (points: typeof before) => Math.hypot(points[1]!.x - points[0]!.x, points[1]!.y - points[0]!.y);
    const center = (points: typeof before) => ({ x: (points[0]!.x + points[1]!.x) / 2, y: (points[0]!.y + points[1]!.y) / 2 });
    const bounds = stage.value?.getBoundingClientRect();
    if (bounds && distance(before) > 0) {
      const scale = bounds.width * dpr.value / props.left.width;
      void zoomAt(scale * distance(after) / distance(before), center(before), center(after));
    }
  }
}
function endPan(event: PointerEvent) {
  pointers.delete(event.pointerId);
  panning.value = pointers.size > 0;
}
function endSplit(event: PointerEvent) {
  if (splitPointer.value === event.pointerId) splitPointer.value = null;
}
function wheelZoom(event: WheelEvent) {
  // Both mouse wheels and trackpad pinch zoom around the pointer.
  if (!stage.value) return;
  event.preventDefault();
  const scale = stage.value.getBoundingClientRect().width * dpr.value / props.left.width;
  const delta = event.deltaY * (event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? viewport.value?.clientHeight ?? 400 : 1);
  // Wheel deltas vary wildly between devices. Cap each event to a small step
  // rather than letting a large delta multiply the zoom several times over.
  const step = Math.max(-40, Math.min(40, delta));
  void zoomAt(scale * Math.exp(-step * 0.0015), { x: event.clientX, y: event.clientY });
}
async function render() {
  await nextTick();
  draw(original.value, props.left);
  draw(decoded.value, props.right);
}
const updateDpr = () => {
  dpr.value = window.devicePixelRatio;
  viewportHeight.value = window.innerHeight;
};
watch([() => props.left, () => props.right], render, { immediate: true });
onMounted(() => window.addEventListener('resize', updateDpr));
onBeforeUnmount(() => window.removeEventListener('resize', updateDpr));
</script>

<template>
  <section class="comparison" :aria-label="`${leftLabel} versus ${rightLabel}`">
      <div class="viewer-toolbar">
        <label>View
          <select v-model="zoom" aria-label="Image comparison zoom">
            <option value="fit">Fit</option>
            <option value="1">1:1 left-image pixels</option>
            <option value="2">2× left-image pixels</option>
            <option value="4">4× left-image pixels</option>
            <option v-if="customZoom" :value="zoom">{{ Math.round(Number(zoom) * 100) }}% (custom)</option>
          </select>
        </label>
      </div>
      <div
        ref="viewport"
        class="viewer-scroll"
        :class="{ fitting: zoom === 'fit', panning }"
        @pointerdown.prevent="startPan"
        @pointermove="movePan"
        @pointerup="endPan"
        @pointercancel="endPan"
        @lostpointercapture="endPan"
        @wheel="wheelZoom"
      >
        <div ref="stage" class="image-stage" :class="{ magnified: Number(zoom) >= 2 }" :style="stageStyle">
          <canvas ref="original" class="image-canvas left-canvas" :aria-label="`Left ${leftLabel} image`"></canvas>
          <canvas
            ref="decoded"
            class="image-canvas decoded-canvas right-canvas"
            :style="{ clipPath: `inset(0 0 0 ${divider}%)` }"
            :aria-label="`Right ${rightLabel} image`"
          ></canvas>
          <div
            class="image-divider"
            :style="{ left: `${divider}%` }"
            @pointerdown.stop.prevent="startSplit"
            @pointermove.stop="splitPointer === $event.pointerId && updateSplit($event)"
            @pointerup.stop="endSplit"
            @pointercancel.stop="endSplit"
            @lostpointercapture.stop="endSplit"
          ><span>↔</span></div>
          <span class="image-label original-label">{{ leftLabel }}</span>
          <span class="image-label decoded-label">{{ rightLabel }}</span>
        </div>
      </div>
      <label class="divider-control">
        Comparison split
        <input v-model.number="divider" type="range" min="0" max="100" aria-label="Image comparison split">
      </label>
      <p class="preview-hint">Drag the image to pan; scroll or pinch with two fingers to zoom. Drag the ↔ handle to move the comparison split. Zoom is measured in device pixels.</p>
  </section>
</template>

<style scoped>
.comparison {
  overflow: hidden;
  background: var(--paper);
}

.viewer-toolbar {
  display: flex;
  align-items: center;
  justify-content: flex-end;
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
  height: clamp(320px, 52vh, 660px);
  overflow: auto;
  padding: 14px;
  background: var(--viewer-background);
  touch-action: none;
  user-select: none;
  cursor: grab;
  overscroll-behavior: contain;
}

.viewer-scroll.panning {
  cursor: grabbing;
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
