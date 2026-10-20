<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, shallowRef, watch } from 'vue';
import { cachedModels, getModel, loadManifest, modelLabels, removeModel, type ModelManifest } from '../lib/models';

import { operatingPoints, type OperatingPoint } from '../types';
import { bytes as formatBytes } from '../lib/image';
import MetricTooltip from './MetricTooltip.vue';

const props = defineProps<{ operatingPoint: OperatingPoint; rateMode: 'model' | 'target'; selected: number; disabled: boolean; refreshKey?: string }>();
const emit = defineEmits<{ use: [point: OperatingPoint, id: number]; point: [point: OperatingPoint]; busy: [value: boolean]; ready: [value: boolean] }>();
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
const storedBytes = computed(() => operatingPoints.reduce((total, point) => total +
  (manifests.value[point.id]?.models.reduce((sum, model) => sum + (cached.value[point.id].has(model.id) ? model.bytes : 0), 0) ?? 0), 0));
const isDownloading = (point: OperatingPoint, id: number) => downloading.value?.point === point && downloading.value.id === id;
function sharedSize(point: OperatingPoint) {
  const models = manifests.value[point]?.models;
  return models?.length && models.every(model => model.bytes === models[0]!.bytes) ? formatBytes(models[0]!.bytes) : undefined;
}

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
  <div class="model-control" role="group" aria-label="JPEG AI models" :data-operating-point="operatingPoint" :data-selected-model="selected">
    <p class="selection-summary" aria-live="polite">Using <strong>{{ operatingPoints.find(point => point.id === operatingPoint)?.label }}</strong> · {{ rateMode === 'target' ? 'Automatic model selection' : `Model ${selected + 1}` }}</p>
    <div class="model-scroll" role="region" aria-label="Scroll model table" tabindex="0">
      <table aria-label="JPEG AI model downloads">
        <thead>
          <tr>
            <th scope="col">Model</th>
            <th v-for="point in operatingPoints" :key="point.id" scope="col" class="point-column" :class="{ current: point.id === operatingPoint }" :data-operating-point="point.id">
              <MetricTooltip v-slot="{ tooltipId, showTooltip, hideTooltip }" :label="point.label" :text="point.description" :as-child="true">
                <button class="point-select" :class="{ active: point.id === operatingPoint }" :data-operating-point="point.id" :aria-label="`Use ${point.label}`" :aria-describedby="tooltipId" :aria-pressed="point.id === operatingPoint" :disabled="disabled || loading" @focus="showTooltip" @blur="hideTooltip" @click="emit('point', point.id); showTooltip()">
                  <span class="point-name">{{ point.label }}</span>
                  <small class="model-size">{{ sharedSize(point.id) ? `${sharedSize(point.id)} each` : 'Size per model' }}</small>
                  <span v-if="rateMode === 'target' && point.id === operatingPoint" class="automatic-label">Automatic</span>
                </button>
              </MetricTooltip>
            </th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="(label, index) in modelLabels" :key="index" :data-model="index" :class="{ selected: rateMode === 'model' && selected === index }">
            <th scope="row">
              <span class="model-name">Model {{ index + 1 }}</span>
              <span class="model-label">{{ label }}</span>
            </th>
            <td v-for="point in operatingPoints" :key="point.id" class="model-cell point-column" :class="{ active: point.id === operatingPoint && rateMode === 'model' && selected === index, automatic: point.id === operatingPoint && rateMode === 'target' }" :data-operating-point="point.id" :data-model="index" :data-ready="cached[point.id].has(index)">
              <button class="model-select" :aria-label="`Use ${point.label} model ${index + 1}`" :aria-pressed="point.id === operatingPoint && rateMode === 'model' && selected === index" :disabled="disabled || loading" @click="emit('use', point.id, index)">
                <span v-if="point.id === operatingPoint && rateMode === 'model' && selected === index" class="selected-label">✓ Selected</span>
                <span v-else class="select-hint">Select</span>
              </button>
              <div class="cell-actions">
              <small v-if="!sharedSize(point.id)" class="model-size">{{ manifests[point.id] ? formatBytes(manifests[point.id]!.models[index]!.bytes) : '…' }}</small>
              <button v-if="isDownloading(point.id, index)" class="model-action cancel" aria-label="Cancel download" :title="`Cancel download (${progress}%)`" @click="abort?.abort()">
                <svg viewBox="0 0 24 24" aria-hidden="true"><path d="m6 6 12 12M18 6 6 18" /></svg>
                <span class="action-label">Cancel · {{ progress }}%</span>
              </button>
              <button v-else-if="cached[point.id].has(index)" class="model-action remove" aria-label="Remove" :disabled="disabled || loading || downloading !== null" @click="remove(point.id, index)">
                <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 6h18M9 6V3h6v3M6 6l1 15h10l1-15M10 10v7m4-7v7" /></svg>
                <span class="action-label">Remove</span>
              </button>
              <button v-else class="model-action download" aria-label="Download" :disabled="disabled || loading || downloading !== null" @click="fetchModel(point.id, index)">
                <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3v12m-4-4 4 4 4-4M5 17v4h14v-4" /></svg>
                <span class="action-label">Download</span>
              </button>
              </div>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
    <p class="stored-size" aria-live="polite">Stored models use {{ formatBytes(storedBytes) }}</p>
    <p v-if="error" class="error" role="alert">{{ error }}</p>
  </div>
</template>

<style scoped>
.model-control {
  margin-top: 20px;
}

.model-scroll {
  overflow-x: auto;
  border: 1px solid var(--line);
  border-radius: 6px;
}

.selection-summary {
  margin: 0 0 10px;
  font-size: 12px;
  color: var(--muted);
}

.selection-summary strong {
  color: var(--ink);
  font-weight: 600;
}

table {
  width: 100%;
  table-layout: fixed;
  border-collapse: collapse;
  font-size: 12px;
}

th, td {
  padding: 10px 14px;
  border-bottom: 1px solid var(--line);
  text-align: left;
  vertical-align: middle;
}

thead th {
  background: var(--surface);
  font-weight: 500;
}

thead th:first-child {
  width: 150px;
}

thead .point-column {
  position: relative;
  padding: 0;
  border-left: 1px solid var(--line);
}

.point-select {
  position: relative;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: flex-start;
  width: 100%;
  min-height: 70px;
  padding: 10px;
  border: 0;
  background: transparent;
  color: var(--ink);
  font: inherit;
  text-align: center;
}

.point-select:focus-visible {
  outline-offset: -4px;
}

.point-select.active {
  background: var(--accent-soft);
  box-shadow: inset 0 0 0 1px var(--accent);
  font-weight: 600;
}

.point-column :deep(.metric-help) {
  display: block;
}

.point-name {
  min-height: 18px;
  line-height: 18px;
  text-decoration: underline dotted var(--muted);
  text-underline-offset: 4px;
}

.automatic-label {
  position: absolute;
  bottom: 4px;
  left: 0;
  right: 0;
  line-height: 12px;
  font-size: 10px;
  color: var(--accent);
}

tbody tr:last-child th,
tbody tr:last-child td {
  border-bottom: 0;
}

.model-cell {
  position: relative;
  height: 68px;
  border-left: 1px solid var(--line);
}

.model-cell.active {
  background: var(--accent-soft);
  box-shadow: inset 0 0 0 2px var(--accent);
}

.model-cell.automatic {
  background: var(--accent-soft);
  box-shadow: inset 2px 0 var(--accent), inset -2px 0 var(--accent);
}

.model-cell:not(.active):not(.automatic):hover {
  background: var(--surface);
}

.model-select {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  padding: 0;
  border: 0;
  background: transparent;
  color: var(--ink);
}

.model-select:focus-visible {
  outline-offset: -5px;
}

.cell-actions {
  position: absolute;
  left: 50%;
  top: 50%;
  transform: translate(-50%, -50%);
  z-index: 1;
  display: inline-flex;
  align-items: center;
  gap: 8px;
  pointer-events: none;
}

.cell-actions button {
  pointer-events: auto;
}

.model-name {
  font-size: 12px;
  font-weight: 600;
}

.selected-label,
.select-hint {
  position: absolute;
  top: 4px;
  right: 8px;
  color: var(--accent);
  font-size: 10px;
  line-height: 12px;
  font-weight: 600;
}

.select-hint {
  opacity: 0;
  color: var(--muted);
  font-weight: 400;
}

.model-select:hover .select-hint,
.model-select:focus-visible .select-hint {
  opacity: 1;
}

.model-label {
  display: block;
  margin-top: 3px;
  color: var(--muted);
  font-size: 11px;
  font-weight: 400;
}

.model-size {
  display: block;
  margin-top: 4px;
  color: var(--muted);
  font-family: ui-monospace, monospace;
  font-size: 11px;
  font-weight: 400;
  white-space: nowrap;
}

.model-action {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  min-width: 106px;
  min-height: 28px;
  padding: 5px 10px;
  line-height: 16px;
  border: 1px solid var(--line);
  border-radius: 5px;
  font-size: 12px;
}

.model-action svg {
  width: 14px;
  height: 14px;
  fill: none;
  stroke: currentColor;
  stroke-width: 1.7;
  stroke-linecap: round;
  stroke-linejoin: round;
}

.download {
  background: var(--paper);
  color: var(--ink);
}

.remove,
.cancel {
  background: transparent;
  color: var(--danger);
}

.remove:hover:not(:disabled),
.cancel:hover {
  background: var(--error-bg);
}

@container (max-width: 680px) {
  th, td {
    padding: 8px;
  }

  thead th:first-child {
    width: 82px;
  }

  .point-select {
    min-height: 112px;
    padding: 8px 4px;
    line-height: 16px;
  }

  .point-name {
    display: block;
    min-height: 48px;
    line-height: 16px;
    font-size: 11px;
  }

  .model-size {
    white-space: normal;
    font-size: 10px;
    line-height: 14px;
  }

  .model-action {
    min-width: 26px;
    width: 26px;
    padding: 5px;
  }

  .action-label {
    display: none;
  }

  .selected-label,
  .select-hint {
    right: 3px;
    font-size: 9px;
  }
}

.stored-size {
  margin: 8px 0 0;
  color: var(--muted);
  font-size: 11px;
}

.error {
  padding: 10px;
  color: var(--error-ink);
  background: var(--error-bg);
  border-radius: 5px;
  font-size: 12px;
}
</style>
