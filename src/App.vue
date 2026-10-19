<script setup lang="ts">
import { computed, ref, shallowRef } from 'vue';
import ImageInput from './components/ImageInput.vue';
import CodecPane from './components/CodecPane.vue';
import { formats } from './types';
import { compressionStats, download, prepareImage } from './lib/image';
import { useComparisons } from './lib/comparisons';

const source = shallowRef<ImageData>();
const file = shallowRef<File>();
const workingSize = ref(512);
const importing = ref(false);
const error = ref('');
const { settings, results, status, busy, run, runAll, cancel, cancelAll, reset } = useComparisons(source);
const hasResults = computed(() => Object.keys(results).length > 0);

async function loadFile(chosen: File) {
  if (busy.value || importing.value) return;
  importing.value = true;
  error.value = '';
  try {
    const image = await prepareImage(chosen, workingSize.value);
    reset();
    source.value = image;
    file.value = chosen;
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e);
  } finally { importing.value = false; }
}

function saveReport() {
  const report = formats.flatMap(({ id }) => {
    const result = results[id];
    if (!result) return [];
    const { pixels: _p, encoded, notices: _n, ...rest } = result;
    return [{ ...rest, encodedBytes: encoded.length, ...compressionStats(result) }];
  });
  download(new Blob([JSON.stringify({
    source: file.value?.name,
    originalBytes: file.value?.size,
    normalization: '8-bit sRGB, alpha flattened on white, browser resize',
    compressionRatioDefinition: 'raw working RGB bytes / encoded file bytes',
    ssim: '8x8 block SSIM on Rec.709 luma, population variance',
    results: report,
  }, null, 2)], { type: 'application/json' }), 'compression-report.json');
}
</script>

<template>
  <div class="app-shell">
    <header class="page-header">
      <div>
        <h1>Image compression demo</h1>
        <p>JPEG AI, JPEG, AVIF and JPEG XL · CPU WebAssembly · no image uploads</p>
      </div>
      <nav aria-label="Documentation"><a href="./legal/" target="_blank" rel="noopener">Implementation & licenses</a><a href="./source.tar.gz" download>Source</a></nav>
    </header>
    <main>
      <ImageInput
        v-model:size="workingSize"
        :filename="file?.name"
        :width="source?.width"
        :height="source?.height"
        :disabled="busy || importing"
        @choose="loadFile"
        @resize="file && loadFile(file)"
      />
      <div class="comparison-actions">
        <button class="primary-button" :disabled="!source || busy || importing" @click="runAll">Encode all four formats</button>
        <button v-if="busy" class="secondary-button" @click="cancelAll">Cancel all</button>
        <button class="secondary-button" :disabled="!hasResults" @click="saveReport">Download comparison JSON</button>
        <span v-if="importing" role="status">Preparing working image…</span>
        <span v-else>Jobs run sequentially to avoid CPU contention.</span>
      </div>
      <p class="measurement-note">
        Compression ratio = raw working RGB bytes / encoded file bytes. Startup and model downloads are measured separately from encoding and decoding.
      </p>
      <p v-if="error" class="error-panel" role="alert">{{ error }}</p>
      <section class="codec-grid" aria-label="Codec comparisons">
        <CodecPane
          v-for="format in formats"
          :key="format.id"
          v-model="settings[format.id]"
          :source="source"
          :file="file"
          :result="results[format.id]"
          :status="status[format.id]"
          :importing="importing"
          @encode="run(format.id)"
          @cancel="cancel(format.id)"
        />
      </section>
    </main>
  </div>
</template>

<style scoped>
.app-shell {
  max-width: 1920px;
  padding: 0 28px 28px;
  margin: auto;
}

.page-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 24px;
  padding: 22px 0;
  margin-bottom: 20px;
  border-bottom: 1px solid var(--line);
}

h1 {
  margin: 0;
  font-size: 22px;
  font-weight: 600;
}

.page-header p {
  margin: 8px 0 0;
  color: var(--muted);
  font-size: 12px;
}

nav {
  display: flex;
  flex-wrap: wrap;
  gap: 16px;
  font-size: 12px;
}

nav a {
  color: var(--accent);
  text-underline-offset: 3px;
}

.codec-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 22px;
  align-items: start;
}

.comparison-actions {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 12px;
  margin: 18px 0 12px;
}

.comparison-actions > span,
.measurement-note {
  color: var(--muted);
  font-size: 12px;
  line-height: 1.6;
}

.measurement-note {
  margin: 0 0 18px;
}

.comparison-actions button {
  padding: 10px 14px;
  border: 1px solid var(--line);
  border-radius: 6px;
  font-size: 12px;
}

.primary-button {
  color: var(--button-ink);
  background: var(--button-bg);
}

.secondary-button {
  color: var(--ink);
  background: var(--paper);
}

.error-panel {
  margin: 10px 0;
  padding: 15px;
  color: var(--error-ink);
  background: var(--error-bg);
  border: 1px solid var(--line);
  border-radius: 7px;
  font-size: 12px;
  line-height: 1.7;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}

@media (min-width: 1280px) {
  .codec-grid {
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 16px;
  }
}

@media (max-width: 720px) {
  .app-shell {
    padding: 0 14px 20px;
  }

  .page-header {
    flex-direction: column;
    align-items: flex-start;
    gap: 14px;
  }

  .codec-grid {
    grid-template-columns: minmax(0, 1fr);
    gap: 18px;
  }
}
</style>
