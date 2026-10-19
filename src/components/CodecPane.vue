<script setup lang="ts">
import { computed } from 'vue';
import CodecControls from './CodecControls.vue';
import ComparisonViewer from './ComparisonViewer.vue';
import ModelLibrary from './ModelLibrary.vue';
import ResultsPanel from './ResultsPanel.vue';
import { formats, type Result, type Settings } from '../types';
import type { PaneStatus } from '../lib/comparisons';

const settings = defineModel<Settings>({ required: true });
const props = defineProps<{
  source?: ImageData;
  file?: File;
  result?: Result;
  status: PaneStatus;
  importing: boolean;
}>();
const emit = defineEmits<{ encode: []; cancel: [] }>();
const format = computed(() => formats.find(format => format.id === settings.value.format)!);
const working = computed(() => props.status.phase === 'running' || props.status.phase === 'queued');
const changed = computed(() => !!props.result && JSON.stringify(settings.value) !== JSON.stringify(props.result.settings));
function selectModel(id: number) {
  settings.value = { ...settings.value, model: id, rateMode: 'model' };
}
</script>

<template>
  <section class="codec-pane" :data-codec="format.id" :aria-labelledby="`${format.id}-title`" :aria-busy="working">
    <header class="pane-heading">
      <div><h2 :id="`${format.id}-title`">{{ format.label }}</h2><p>{{ format.description }}</p></div>
      <button v-if="working" class="cancel-button" @click="emit('cancel')">{{ status.phase === 'queued' ? 'Cancel queued job' : 'Cancel encoding' }}</button>
      <button v-else class="encode-button" :disabled="!source || importing" @click="emit('encode')">{{ result ? 'Encode again' : 'Encode' }}</button>
    </header>
    <ResultsPanel :result="result" :file="file" />
    <p class="pane-status" role="status" aria-live="polite">{{ status.message }}</p>
    <p v-if="status.error" class="error" role="alert">{{ status.error }}</p>
    <p v-if="changed" class="changed-note">Settings changed. Encode again to update this pane.</p>
    <p v-else-if="result && working" class="changed-note">Showing the previous result while this job runs.</p>
    <ComparisonViewer :source="source" :result="result" :format="format.label" />
    <ResultsPanel v-if="result" :result="result" :file="file" part="details" />
    <details class="pane-settings">
      <summary>{{ format.id === 'jpegai' ? 'Settings & model weights' : 'Settings' }}</summary>
      <div class="settings-content">
        <CodecControls v-model="settings" :disabled="working || importing" />
        <ModelLibrary
          v-if="format.id === 'jpegai'"
          :selected="settings.model"
          :disabled="working || importing"
          :refresh-key="result"
          @use="selectModel"
        />
      </div>
    </details>
  </section>
</template>

<style scoped>
.codec-pane {
  container-type: inline-size;
  min-width: 0;
  padding: 18px;
  border: 1px solid var(--line);
  border-radius: var(--radius);
  background: var(--paper);
}

.pane-heading {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
  min-height: 68px;
  margin-bottom: 14px;
}

h2 {
  margin: 0;
  font-size: 20px;
  font-weight: 600;
}

.pane-heading p {
  margin: 6px 0 0;
  color: var(--muted);
  font-size: 12px;
}

.pane-heading button {
  flex-shrink: 0;
  padding: 10px 12px;
  border: 1px solid var(--line);
  border-radius: 5px;
  font-size: 12px;
}

.encode-button {
  color: var(--button-ink);
  background: var(--button-bg);
}

.cancel-button {
  color: var(--danger);
  background: var(--surface);
}

.pane-status,
.changed-note {
  margin: 12px 0;
  color: var(--muted);
  font-size: 12px;
  line-height: 1.5;
  overflow-wrap: anywhere;
}

.changed-note {
  color: var(--accent);
}

.error {
  padding: 12px;
  border-radius: 5px;
  color: var(--error-ink);
  background: var(--error-bg);
  font-size: 12px;
  line-height: 1.6;
  overflow-wrap: anywhere;
}

.pane-settings {
  margin-top: 16px;
  padding-top: 14px;
  border-top: 1px solid var(--line);
}

.pane-settings > summary {
  cursor: pointer;
  font-size: 13px;
  font-weight: 500;
}

.settings-content {
  padding-top: 18px;
}

@media (max-width: 420px) {
  .codec-pane {
    padding: 12px;
  }

  .pane-heading {
    gap: 10px;
  }
}
</style>
