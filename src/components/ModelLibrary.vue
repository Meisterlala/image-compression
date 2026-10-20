<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref, shallowRef, watch } from 'vue';
import { cachedModels, getModel, loadManifest, modelLabels, removeModel, type ModelManifest } from '../lib/models';

import type { OperatingPoint } from '../types';

const props = defineProps<{ operatingPoint: OperatingPoint; selected: number; disabled: boolean; refreshKey?: object }>();
const emit = defineEmits<{ use: [id: number]; busy: [value: boolean] }>();
const manifest = shallowRef<ModelManifest>();
const cached = shallowRef(new Set<number>());
const downloading = ref<number | null>(null);
const progress = ref(0);
const loading = ref(false);
const error = ref('');
let abort: AbortController | undefined;

async function refresh() {
  loading.value = true;
  try {
    manifest.value = await loadManifest(props.operatingPoint);
    cached.value = await cachedModels(manifest.value.models);
  } catch (e) { error.value = e instanceof Error ? e.message : String(e); }
  finally { loading.value = false; }
}
async function fetchModel(id: number) {
  const model = manifest.value?.models.find(model => model.id === id);
  if (!model || props.disabled || downloading.value !== null || cached.value.has(id)) return;
  error.value = '';
  downloading.value = id;
  progress.value = 0;
  abort = new AbortController();
  try {
    const result = await getModel(model, (received, total) => progress.value = Math.round(received / total * 100), abort.signal);
    cached.value = await cachedModels(manifest.value!.models);
    if (!result.persisted) error.value = 'Could not store model. Check browser storage.';
  } catch (e) {
    if (!(e instanceof DOMException && e.name === 'AbortError')) error.value = e instanceof Error ? e.message : String(e);
  } finally { downloading.value = null; abort = undefined; }
}
function choose(id: number) {
  emit('use', id);
  void fetchModel(id);
}
async function remove(id: number) {
  const model = manifest.value?.models.find(model => model.id === id);
  if (!model || props.disabled) return;
  try {
    await removeModel(model);
    cached.value = await cachedModels(manifest.value!.models);
  } catch (e) { error.value = e instanceof Error ? e.message : String(e); }
}
onMounted(refresh);
watch(downloading, value => emit('busy', value !== null));
watch(() => props.refreshKey, refresh);
onBeforeUnmount(() => { abort?.abort(); emit('busy', false); });
</script>

<template>
  <div class="model-control" role="group" aria-label="JPEG AI models" :data-operating-point="operatingPoint">
    <span class="field-label">{{ operatingPoint.toUpperCase() }} models</span>
    <div class="model-grid">
      <div v-for="(label, index) in modelLabels" :key="index" class="model-card" :class="{ ready: cached.has(index), selected: selected === index }" :data-model="index" :data-ready="cached.has(index)">
        <button class="model-select" :aria-pressed="selected === index" :disabled="disabled || loading || downloading !== null" @click="choose(index)">
          <span class="model-name">Model {{ index + 1 }} <span class="ready-dot" :aria-label="cached.has(index) ? 'Ready' : 'Not cached'"></span></span>
          <span class="model-label">{{ label }}</span>
        </button>
        <button v-if="downloading === index" class="model-action" @click="abort?.abort()">Cancel · {{ progress }}%</button>
        <button v-else-if="cached.has(index)" class="model-action" :disabled="disabled || loading || downloading !== null" @click="remove(index)">Remove</button>
        <button v-else class="model-action" :disabled="disabled || loading || downloading !== null" @click="choose(index)">Download</button>
      </div>
    </div>
    <p v-if="error" class="error" role="alert">{{ error }}</p>
  </div>
</template>

<style scoped>
.model-control {
  margin-top: 20px;
}

.field-label {
  display: block;
  margin-bottom: 8px;
  font-size: 12px;
}

.model-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 10px;
}

.model-card {
  padding: 10px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: var(--surface);
}

.model-card.selected {
  border-color: var(--accent);
  box-shadow: 0 0 0 1px var(--accent);
}

.model-card.ready {
  background: var(--accent-soft);
}

.model-select {
  display: grid;
  width: 100%;
  gap: 6px;
  padding: 4px 0 10px;
  border: 0;
  background: transparent;
  color: var(--ink);
  text-align: left;
}

.model-name {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  font-size: 13px;
  font-weight: 600;
}

.model-label {
  color: var(--muted);
  font-size: 12px;
}

.ready-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--line);
}

.ready .ready-dot {
  background: var(--accent);
}

.model-action {
  padding: 7px 12px;
  border: 1px solid var(--line);
  border-radius: 5px;
  color: var(--ink);
  background: var(--paper);
  font-size: 12px;
}

@container (max-width: 560px) {
  .model-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

.error {
  padding: 10px;
  color: var(--error-ink);
  background: var(--error-bg);
  border-radius: 5px;
  font-size: 12px;
}
</style>
