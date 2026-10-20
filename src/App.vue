<script setup lang="ts">
import { nextTick, ref, shallowRef } from 'vue';
import ImageInput from './components/ImageInput.vue';
import CodecPane from './components/CodecPane.vue';
import ComparisonTable from './components/ComparisonTable.vue';
import ImageComparison from './components/ImageComparison.vue';
import { formats, type Format, type Result } from './types';
import { prepareImage } from './lib/image';
import { useComparisons } from './lib/comparisons';

import { usePersistence, usePersistentRef, usePersistentState } from './lib/persistence';

const { error: storageError, pending: saving } = usePersistence();
const activeFormat = usePersistentRef<Format>('active-format', 'jpegai');
function tabKey(event: KeyboardEvent, index: number) {
  let next: number;
  if (event.key === 'ArrowRight') next = (index + 1) % formats.length;
  else if (event.key === 'ArrowLeft') next = (index + formats.length - 1) % formats.length;
  else if (event.key === 'Home') next = 0;
  else if (event.key === 'End') next = formats.length - 1;
  else return;
  event.preventDefault();
  activeFormat.value = formats[next]!.id;
  void nextTick(() => document.getElementById(`tab-${activeFormat.value}`)?.focus());
}

async function jumpTo(format: Format | 'original') {
  if (format !== 'original') activeFormat.value = format;
  await nextTick();
  const target = document.getElementById(format === 'original' ? 'original-image' : `panel-${format}`);
  target?.focus({ preventScroll: true });
  target?.scrollIntoView({ behavior: 'instant', block: 'start' });
}

const source = shallowRef<ImageData>();
const file = shallowRef<File>();
const referenceBytes = ref(0);
const downscaled = ref(false);
const downscaleSize = usePersistentRef('downscale-size', 0);
let appliedSize = 0;
const importing = ref(false);
const error = ref('');
const { settings, results, status, busy, run, confirm, cancel, reset } = useComparisons(source, referenceBytes, downscaled);

// Source and outputs share one transaction, so a new upload cannot restore old results.
interface Workspace {
  file?: File;
  image?: { pixels: Uint8ClampedArray<ArrayBuffer>; width: number; height: number };
  referenceBytes: number;
  downscaled: boolean;
  appliedSize: number;
  results: Partial<Record<Format, Result>>;
}
usePersistentState<Workspace>('workspace', {
  snapshot: () => ({
    file: file.value,
    image: source.value ? { pixels: source.value.data, width: source.value.width, height: source.value.height } : undefined,
    referenceBytes: referenceBytes.value, downscaled: downscaled.value, appliedSize,
    results: { ...results },
  }),
  restore: saved => {
    const image = saved.image ? new ImageData(saved.image.pixels, saved.image.width, saved.image.height) : undefined;
    source.value = image;
    file.value = saved.file;
    referenceBytes.value = saved.referenceBytes;
    downscaled.value = saved.downscaled;
    appliedSize = saved.appliedSize;
    if (image) for (const { id } of formats) {
      if (!saved.results[id]) continue;
      results[id] = saved.results[id];
      Object.assign(status[id], { phase: 'done', message: 'Restored from browser storage.' });
    }
  },
});
usePersistentState('codec-settings', {
  snapshot: () => settings,
  restore: saved => {
    for (const { id } of formats) Object.assign(settings[id], saved[id], { format: id });
  },
  deep: true,
});

async function loadFile(chosen: File, resizing = false) {
  if (busy.value || importing.value) return;
  importing.value = true;
  error.value = '';
  try {
    const requestedSize = downscaleSize.value;
    const image = await prepareImage(chosen, requestedSize);
    appliedSize = requestedSize;
    if (!resizing) reset();
    source.value = image.image;
    referenceBytes.value = image.referenceBytes;
    downscaled.value = image.downscaled;
    file.value = chosen;
  } catch (e) {
    if (source.value) downscaleSize.value = appliedSize;
    error.value = e instanceof Error ? e.message : String(e);
  } finally { importing.value = false; }
}
</script>

<template>
  <div class="app-shell">
    <header id="original-image" class="page-header" tabindex="-1">
      <div class="intro">
        <h1>Image compression</h1>
        <p>Compare JPEG AI, JPEG, AVIF, JPEG XL and Cool-chic 5.0 on your own image. Adjust compression and compare image quality, file size, and encoding and decoding times.</p>
      </div>
      <ImageInput
        v-model:size="downscaleSize"
        :filename="file?.name"
        :source="source"
        :downscaled="downscaled"
        :width="source?.width"
        :height="source?.height"
        :disabled="busy || importing"
        @choose="loadFile"
        @resize="file && loadFile(file, true)"
      />
    </header>
    <main>
      <p v-if="storageError" class="error-panel" role="alert">{{ storageError }}</p>
      <span v-else class="import-status" role="status" data-persistence>{{ saving ? 'Saving locally…' : 'Saved locally in this browser. Clearing site data removes saved work.' }}</span>
      <span v-if="importing" class="import-status" role="status">Opening image…</span>
      <p v-if="error" class="error-panel" role="alert">{{ error }}</p>
      <div class="codec-tabs" role="tablist" aria-label="Image format">
        <button
          v-for="(format, index) in formats"
          :id="`tab-${format.id}`"
          :key="format.id"
          role="tab"
          :aria-selected="activeFormat === format.id"
          :aria-controls="`panel-${format.id}`"
          :tabindex="activeFormat === format.id ? 0 : -1"
          @click="activeFormat = format.id"
          @keydown="tabKey($event, index)"
        >{{ format.label }}</button>
      </div>
      <div
        v-for="format in formats"
        v-show="activeFormat === format.id"
        :id="`panel-${format.id}`"
        :key="format.id"
        role="tabpanel"
        tabindex="-1"
        :aria-labelledby="`tab-${format.id}`"
      >
        <CodecPane
          v-model="settings[format.id]"
          :source="source"
          :file="file"
          :result="results[format.id]"
          :status="status[format.id]"
          :importing="importing"
          @encode="run(format.id)"
          @cancel="cancel(format.id)"
          @confirm="confirm(format.id)"
        />
      </div>
      <ImageComparison v-if="source && Object.keys(results).length" :source="source" :downscaled="downscaled" :results="results" :active-format="activeFormat" />
      <ComparisonTable :results="results" :source="source" :reference-bytes="referenceBytes" :downscaled="downscaled" @select="jumpTo" />
    </main>
  </div>
</template>

<style scoped>
.app-shell {
  max-width: 1440px;
  padding: 0 28px 28px;
  margin: auto;
}

.page-header {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  align-items: center;
  gap: 28px;
  padding: 22px 0;
  margin-bottom: 20px;
  border-bottom: 1px solid var(--line);
}

.intro {
  max-width: 520px;
}

.intro p {
  margin: 12px 0 0;
  color: var(--muted);
  font-size: 13px;
  line-height: 1.7;
}

h1 {
  margin: 0;
  font-size: 22px;
  font-weight: 600;
}

[role="tabpanel"] {
  scroll-margin-top: 16px;
}

.codec-tabs {
  display: grid;
  grid-template-columns: repeat(5, minmax(0, 1fr));
  margin: 20px 0 12px;
  border-bottom: 1px solid var(--line);
}

.codec-tabs button {
  padding: 14px 8px;
  border: 0;
  border-bottom: 3px solid transparent;
  color: var(--muted);
  background: transparent;
  font-size: 14px;
}

.codec-tabs button[aria-selected="true"] {
  color: var(--accent);
  border-bottom-color: var(--accent);
  background: var(--accent-soft);
  font-weight: 600;
}

.import-status {
  display: block;
  margin-top: 12px;
  color: var(--muted);
  font-size: 12px;
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

@media (max-width: 720px) {
  .codec-tabs { grid-template-columns: repeat(3, minmax(0, 1fr)); }

  .app-shell {
    padding: 0 14px 20px;
  }

  .page-header {
    grid-template-columns: minmax(0, 1fr);
    gap: 18px;
  }

}
</style>
