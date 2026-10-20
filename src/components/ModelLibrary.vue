<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, shallowRef, watch } from 'vue';
import { cachedModels, getModel, loadManifest, modelLabels, removeModel, type ModelManifest } from '../lib/models';

import { operatingPoints, type OperatingPoint } from '../types';
import { bytes as formatBytes } from '../lib/image';

const props = defineProps<{ operatingPoint: OperatingPoint; rateMode: 'model' | 'target'; selected: number; disabled: boolean; refreshKey?: string }>();
const emit = defineEmits<{ use: [point: OperatingPoint, id: number]; busy: [value: boolean]; ready: [value: boolean] }>();
const manifests = shallowRef<Partial<Record<OperatingPoint, ModelManifest>>>({});
const cached = shallowRef<Record<OperatingPoint, Set<number>>>({ sop: new Set(), bop: new Set(), hop: new Set() });
const downloading = shallowRef<{ point: OperatingPoint; id: number } | null>(null);
const progress = ref(0);
const loading = ref(false);
const error = ref('');
let abort: AbortController | undefined;
let refreshVersion = 0;
const ready = computed(() => !loading.value && !!manifests.value[props.operatingPoint] && (props.rateMode === 'target'
  ? manifests.value[props.operatingPoint]!.models.every(model => cached.value[props.operatingPoint].has(model.id))
  : cached.value[props.operatingPoint].has(props.selected)));
const isDownloading = (point: OperatingPoint, id: number) => downloading.value?.point === point && downloading.value.id === id;

async function refresh() {
  const version = ++refreshVersion;
  loading.value = true;
  error.value = '';
  try {
    const groups = await Promise.all(operatingPoints.map(async ({ id }) => {
      const manifest = await loadManifest(id);
      return { point: id, manifest, cached: await cachedModels(manifest.models) };
    }));
    if (version !== refreshVersion) return;
    manifests.value = Object.fromEntries(groups.map(group => [group.point, group.manifest]));
    cached.value = Object.fromEntries(groups.map(group => [group.point, group.cached])) as Record<OperatingPoint, Set<number>>;
  } catch (e) { if (version === refreshVersion) error.value = e instanceof Error ? e.message : String(e); }
  finally { if (version === refreshVersion) loading.value = false; }
}
async function fetchModel(point: OperatingPoint, id: number) {
  const model = manifests.value[point]?.models.find(model => model.id === id);
  if (!model || props.disabled || downloading.value !== null || cached.value[point].has(id)) return;
  error.value = '';
  downloading.value = { point, id };
  progress.value = 0;
  abort = new AbortController();
  try {
    const result = await getModel(model, (received, total) => progress.value = Math.round(received / total * 100), abort.signal);
    await refresh();
    if (!result.persisted) error.value = 'Could not store model. Browser storage is required before encoding.';
  } catch (e) {
    if (!(e instanceof DOMException && e.name === 'AbortError')) error.value = e instanceof Error ? e.message : String(e);
  } finally { downloading.value = null; abort = undefined; }
}
async function remove(point: OperatingPoint, id: number) {
  const model = manifests.value[point]?.models.find(model => model.id === id);
  if (!model || props.disabled) return;
  try {
    await removeModel(model);
    await refresh();
  } catch (e) { error.value = e instanceof Error ? e.message : String(e); }
}
onMounted(refresh);
watch(downloading, value => emit('busy', value !== null));
watch(ready, value => emit('ready', value), { immediate: true });
watch(() => props.refreshKey, refresh);
onBeforeUnmount(() => { refreshVersion++; abort?.abort(); emit('busy', false); emit('ready', false); });
</script>

<template>
  <div class="model-control" role="group" aria-label="JPEG AI models" :data-operating-point="operatingPoint">
    <section v-for="point in operatingPoints" :key="point.id" class="model-group" :class="{ current: point.id === operatingPoint, 'is-downloading': downloading?.point === point.id }" :data-operating-point="point.id" :aria-label="`${point.label} models`">
      <span class="field-label">{{ point.label }} models</span>
      <div class="model-grid">
        <div v-for="(label, index) in modelLabels" :key="index" class="model-card" :class="{ ready: cached[point.id].has(index), selected: point.id === operatingPoint && selected === index }" :data-model="index" :data-ready="cached[point.id].has(index)">
          <button class="model-select" :aria-pressed="point.id === operatingPoint && selected === index" :disabled="disabled || loading" @click="emit('use', point.id, index)">
            <span class="model-name">Model {{ index + 1 }} <span class="ready-dot" :aria-label="cached[point.id].has(index) ? 'Ready' : 'Not cached'"></span></span>
            <span class="model-label">{{ label }}</span>
            <span class="model-size">{{ manifests[point.id] ? formatBytes(manifests[point.id]!.models[index]!.bytes) : '…' }}</span>
          </button>
          <button v-if="isDownloading(point.id, index)" class="model-action" @click="abort?.abort()">Cancel · {{ progress }}%</button>
          <button v-else-if="cached[point.id].has(index)" class="model-action" :disabled="disabled || loading || downloading !== null" @click="remove(point.id, index)">Remove</button>
          <button v-else class="model-action" :disabled="disabled || loading || downloading !== null" @click="fetchModel(point.id, index)">Download</button>
        </div>
      </div>
    </section>
    <p v-if="error" class="error" role="alert">{{ error }}</p>
  </div>
</template>

<style scoped>
.model-control {
  margin-top: 20px;
}

.model-group + .model-group {
  margin-top: 12px;
}

.current .field-label {
  color: var(--accent);
}

.field-label {
  display: block;
  margin-bottom: 6px;
  font-size: 12px;
}

.model-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 10px;
}

.model-card {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  align-items: center;
  gap: 8px;
  padding: 8px;
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
  gap: 2px;
  padding: 0;
  border: 0;
  line-height: 1.35;
  background: transparent;
  color: var(--ink);
  text-align: left;
}

.model-name {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  font-size: 12px;
  font-weight: 600;
}

.model-label {
  color: var(--muted);
  font-size: 11px;
}

.model-size {
  color: var(--muted);
  font-family: ui-monospace, monospace;
  font-size: 11px;
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
  padding: 6px 8px;
  border: 1px solid var(--line);
  border-radius: 5px;
  color: var(--ink);
  background: var(--paper);
  font-size: 12px;
}

@container (max-width: 680px) {
  .model-group:not(.current):not(.is-downloading) {
    display: none;
  }

  .model-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .model-card {
    grid-template-columns: minmax(0, 1fr);
  }

  .model-action {
    justify-self: start;
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
