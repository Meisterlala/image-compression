<script setup lang="ts">
import { computed, watch } from 'vue';
import { usePersistentRef } from '../lib/persistence';
import { formats, type Format, type Result } from '../types';
import ComparisonViewer from './ComparisonViewer.vue';

const props = defineProps<{ source: ImageData; downscaled: boolean; results: Partial<Record<Format, Result>>; activeFormat: Format }>();
type Selection = 'source' | Format;
const left = usePersistentRef<Selection>('comparison-left', 'source');
const right = usePersistentRef<Selection>('comparison-right', 'source');
// An old selection for a cleared codec is not a valid selection for a new image.
const savedRight = right.value !== 'source' && props.results[right.value] ? right.value : 'source';
for (const selection of [left, right]) {
  if (selection.value !== 'source' && !props.results[selection.value]) selection.value = 'source';
}
let lastEncoded: Format | undefined;
const sourceLabel = computed(() => props.downscaled ? 'Downscaled' : 'Original');

// A completed encode selects its output. Manual selections otherwise remain independent.
watch(() => formats.map(format => props.results[format.id]), (next, previous) => {
  const index = next.findIndex((result, index) => result && result !== previous?.[index]);
  if (index !== -1) {
    lastEncoded = formats[index]!.id;
    if (previous || savedRight === 'source') right.value = lastEncoded;
  }
}, { immediate: true });
watch(() => props.activeFormat, format => {
  if (props.results[format]) right.value = format;
  else if (lastEncoded) right.value = lastEncoded;
});

function label(selection: Selection) {
  return selection === 'source' ? sourceLabel.value : formats.find(format => format.id === selection)!.label;
}
function image(selection: Selection) {
  if (selection === 'source') return props.source;
  const result = props.results[selection];
  return result ? new ImageData(result.pixels, result.width, result.height) : undefined;
}
const leftImage = computed(() => image(left.value));
const rightImage = computed(() => image(right.value));
const choices = computed(() => [
  { id: 'source', label: sourceLabel.value, disabled: false },
  ...formats.map(format => {
    const result = props.results[format.id];
    return { id: format.id, label: format.label, disabled: !result };
  }),
]);
</script>

<template>
  <section class="image-comparison-card" aria-labelledby="image-comparison-title">
    <h2 id="image-comparison-title">Image comparison</h2>
    <div class="image-choices">
      <label>Left image
        <select v-model="left" aria-label="Left image">
          <option v-for="choice in choices" :key="choice.id" :value="choice.id" :disabled="choice.disabled">{{ choice.label }}</option>
        </select>
      </label>
      <label>Right image
        <select v-model="right" aria-label="Right image">
          <option v-for="choice in choices" :key="choice.id" :value="choice.id" :disabled="choice.disabled">{{ choice.label }}</option>
        </select>
      </label>
    </div>
    <ComparisonViewer v-if="leftImage && rightImage" :left="leftImage" :right="rightImage" :left-label="label(left)" :right-label="label(right)" />
  </section>
</template>

<style scoped>
.image-comparison-card {
  container-type: inline-size;
  margin-top: 22px;
  overflow: hidden;
  border: 1px solid var(--line);
  border-radius: var(--radius);
  background: var(--paper);
}

h2 {
  margin: 0;
  padding: 16px 16px 0;
  font-size: 16px;
  font-weight: 600;
}

.image-choices {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 16px;
  padding: 16px;
  border-bottom: 1px solid var(--line);
}

label {
  display: grid;
  gap: 7px;
  min-width: 0;
  color: var(--muted);
  font-size: 12px;
}

select {
  width: 100%;
  min-width: 0;
  padding: 10px;
  border: 1px solid var(--line);
  border-radius: 5px;
  color: var(--ink);
  background: var(--surface);
}

@media (max-width: 560px) {
  .image-choices { gap: 10px; padding: 14px; }
  select { padding: 9px 5px; font-size: 16px; min-height: 44px; }
}
</style>
