<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref, shallowRef, watch } from 'vue';
import { cachedModels, getModel, importModel, loadManifest, modelLabels, removeModel, type ModelEntry, type ModelManifest } from '../lib/models';
import { bytes, download } from '../lib/image';

const props = defineProps<{ selected: number; disabled: boolean; refreshKey?: object }>();
const emit = defineEmits<{ use: [id: number] }>();
const importInput = ref<HTMLInputElement>();
const manifest = shallowRef<ModelManifest>();
const cached = shallowRef(new Set<number>());
const downloading = ref<number | null>(null);
const progress = ref(0);
const loading = ref(false);
const error = ref('');
const message = ref('');
let abort: AbortController | undefined;

async function refresh() {
  loading.value = true;
  try {
    manifest.value = await loadManifest();
    cached.value = await cachedModels(manifest.value.models);
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e);
  } finally { loading.value = false; }
}
async function fetchModel(entry: ModelEntry, save = false) {
  if (props.disabled || downloading.value !== null) return;
  error.value = ''; message.value = ''; downloading.value = entry.id; progress.value = 0;
  abort = new AbortController();
  try {
    const model = await getModel(entry, (received, total) => progress.value = Math.round(received / total * 100), abort.signal);
    if (save) download(new Blob([model.bytes], { type: 'application/octet-stream' }), entry.file);
    cached.value = await cachedModels(manifest.value!.models);
    message.value = model.persisted
      ? `Model ${entry.id + 1} is ready. Select “Use model” to choose it.`
      : 'Your browser could not cache these weights. Compression can still load them directly; try freeing storage.';
  } catch (e) {
    if (!(e instanceof DOMException && e.name === 'AbortError')) error.value = e instanceof Error ? e.message : String(e);
  } finally { downloading.value = null; abort = undefined; }
}
async function remove(entry: ModelEntry) {
  try {
    await removeModel(entry);
    cached.value = await cachedModels(manifest.value!.models);
    message.value = `Model ${entry.id + 1} removed from the app cache.`;
  } catch (e) { error.value = e instanceof Error ? e.message : String(e); }
}
async function importFile(files: FileList | null) {
  if (!files?.[0] || !manifest.value) return;
  loading.value = true; error.value = '';
  try {
    const entry = await importModel(files[0], manifest.value.models);
    cached.value = await cachedModels(manifest.value.models);
    message.value = `Imported model ${entry.id + 1}. SHA-256 verified; ready to use.`;
  } catch (e) { error.value = e instanceof Error ? e.message : String(e); }
  finally { loading.value = false; if (importInput.value) importInput.value.value = ''; }
}
onMounted(refresh);
watch(() => props.refreshKey, refresh);
onBeforeUnmount(() => abort?.abort());
</script>

<template>
  <section class="model-library" aria-label="JPEG AI models">
    <h3>Model weights</h3>
    <p class="intro">BOP models · ~26 MiB each. Target-bitrate search needs all four. Downloaded weights are cached and reused for encoding and decoding.</p>
    <p v-if="loading" role="status">Checking local models…</p>
    <div class="model-list">
      <article v-for="model in manifest?.models" :key="model.id" :data-model="model.id" class="model-card" :class="{ selected: selected === model.id }">
        <div class="model-description">
          <strong>Model {{ model.id + 1 }} · {{ modelLabels[model.id] }}</strong>
          <span>{{ bytes(model.bytes) }} · {{ cached.has(model.id) ? 'Downloaded · ready' : 'Not downloaded' }}</span>
        </div>
        <div class="model-actions">
          <button v-if="!cached.has(model.id)" class="download-button" :disabled="disabled || downloading !== null || loading" @click="fetchModel(model)">Download weights</button>
          <button v-else class="use-button" :disabled="disabled || downloading !== null || loading" @click="emit('use', model.id)">{{ selected === model.id ? 'Selected ✓' : 'Use model' }}</button>
          <button class="text-button" :disabled="disabled || downloading !== null || loading" @click="fetchModel(model, true)">Save .zjb</button>
          <button v-if="cached.has(model.id)" class="text-button remove-button" :disabled="disabled || downloading !== null || loading" @click="remove(model)">Remove</button>
        </div>
        <div v-if="downloading === model.id" class="progress-area" role="status">
          <progress :value="progress" max="100"></progress>
          <span>{{ progress }}% · download / verify</span><button @click="abort?.abort()">Cancel download</button>
        </div>
        <details class="integrity"><summary>SHA-256</summary><code>{{ model.sha256 }}</code></details>
      </article>
    </div>
    <div class="import-row">
      <button class="import-button" :disabled="disabled || loading || downloading !== null || !manifest" @click="importInput?.click()">Import .zjb file</button>
      <input ref="importInput" class="file-input" type="file" accept=".zjb" aria-label="Import model file" :disabled="disabled || loading || downloading !== null" @change="importFile(($event.target as HTMLInputElement).files)">
      <span>Only SHA-256 verified bundles from this build are accepted.</span>
    </div>
    <p v-if="message" class="message" role="status">{{ message }}</p>
    <p v-if="error" class="error" role="alert">{{ error }}</p>
    <p class="footnote">Model bytes are not included in the encoded file or compression ratio. Missing models also download on encode. <a href="./legal/" target="_blank" rel="noopener">License & patent notice</a></p>
  </section>
</template>

<style scoped>
.model-library {
  margin-top: 20px;
  padding-top: 18px;
  border-top: 1px solid var(--line);
}

h3 {
  margin: 0;
  font-size: 14px;
  font-weight: 600;
}

.intro {
  margin: 8px 0 12px;
  color: var(--muted);
  font-size: 12px;
  line-height: 1.6;
}

.model-list {
  display: grid;
  gap: 8px;
}

.model-card {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 8px 12px;
  padding: 10px;
  border: 1px solid var(--line);
  border-radius: 5px;
  background: var(--paper);
}

.model-card.selected {
  border-color: var(--accent);
}

.model-description {
  display: grid;
  gap: 5px;
  font-size: 12px;
  line-height: 1.5;
}

.model-description strong {
  font-weight: 500;
}

.model-description span {
  color: var(--muted);
  font-size: 11px;
}

.integrity,
.progress-area {
  grid-column: 1 / -1;
}

.model-actions {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 6px;
  justify-content: flex-end;
}

button {
  font-size: 10px;
}

.download-button,
.use-button,
.import-button {
  padding: 9px 11px;
  border: 1px solid var(--accent);
  border-radius: 5px;
  color: var(--button-ink);
  background: var(--button-bg);
}

.text-button {
  padding: 7px 4px;
  border: 0;
  color: var(--accent);
  background: none;
}

.remove-button {
  color: var(--danger);
}

details {
  color: var(--muted);
  font-size: 9px;
}

summary {
  cursor: pointer;
}

code {
  display: block;
  padding-top: 8px;
  overflow-wrap: anywhere;
  line-height: 1.6;
}

.progress-area {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 6px;
  color: var(--muted);
  font-size: 10px;
}

progress {
  width: 100%;
  height: 7px;
  accent-color: var(--accent);
}

.progress-area button {
  border: 0;
  color: var(--danger);
  background: none;
}

.import-row {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 12px;
  margin-top: 22px;
}

.import-button {
  color: var(--accent);
  background: transparent;
}

.import-row span,
.footnote {
  color: var(--muted);
  font-size: 10px;
  line-height: 1.7;
}

.file-input {
  position: absolute;
  width: 1px;
  height: 1px;
  clip-path: inset(50%);
}

.message,
.error {
  padding: 12px;
  font-size: 12px;
  line-height: 1.5;
  border-radius: 6px;
}

.message {
  background: var(--accent-soft);
  color: var(--accent);
}

.error {
  background: var(--error-bg);
  color: var(--error-ink);
}

.footnote {
  margin: 20px 0 0;
}

.footnote a {
  color: var(--accent);
}

@container (max-width: 520px) {
  .model-card {
    grid-template-columns: minmax(0, 1fr);
  }

  .model-actions {
    justify-content: flex-start;
  }
}
</style>
