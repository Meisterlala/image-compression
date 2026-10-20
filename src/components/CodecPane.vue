<script setup lang="ts">
import { computed, ref } from 'vue';
import CodecControls from './CodecControls.vue';
import ModelLibrary from './ModelLibrary.vue';
import ResultDownloads from './ResultDownloads.vue';
import { formats, type OperatingPoint, type Result, type Settings } from '../types';
import type { PaneStatus } from '../lib/comparisons';

const settings = defineModel<Settings>({ required: true });
const props = defineProps<{
  source?: ImageData;
  file?: File;
  result?: Result;
  status: PaneStatus;
  importing: boolean;
}>();
const emit = defineEmits<{ encode: []; confirm: []; cancel: [] }>();
const modelBusy = ref(false);
const modelsReady = ref(false);
const format = computed(() => formats.find(format => format.id === settings.value.format)!);
const working = computed(() => props.status.phase === 'running' || props.status.phase === 'queued' || props.status.phase === 'confirming');
const changed = computed(() => !!props.result && (
  props.source?.width !== props.result.width || props.source?.height !== props.result.height ||
  JSON.stringify(settings.value) !== JSON.stringify(props.result.settings)
));
function selectPoint(operatingPoint: OperatingPoint) {
  settings.value = { ...settings.value, operatingPoint, rateMode: 'target' };
}
function selectModel(operatingPoint: OperatingPoint, id: number) {
  settings.value = { ...settings.value, operatingPoint, model: id, rateMode: 'model' };
}
</script>

<template>
  <section class="codec-pane" :data-codec="format.id" :data-phase="status.phase" :aria-labelledby="`${format.id}-title`" :aria-busy="status.phase === 'running' || status.phase === 'queued'">
    <header class="pane-heading">
      <h2 :id="`${format.id}-title`">{{ format.label }}</h2>
    </header>
    <section class="pane-settings" aria-label="Settings">
      <CodecControls v-model="settings" :disabled="working || importing" />
      <ModelLibrary
        v-if="format.id === 'jpegai'"
        :operating-point="settings.operatingPoint"
        :rate-mode="settings.rateMode"
        :selected="settings.model"
        :disabled="working || importing"
        :refresh-key="status.phase"
        @use="selectModel"
        @point="selectPoint"
        @busy="modelBusy = $event"
        @ready="modelsReady = $event"
      />
    </section>
    <p v-if="status.error" class="error" role="alert">{{ status.error }}</p>
    <p v-if="changed" class="changed-note">Settings or image size changed. Showing the previous result; encode again to update it.</p>
    <p v-else-if="result && working" class="changed-note">Showing the previous result while this job runs.</p>
    <div class="output-actions" :class="{ 'no-output': !result }">
      <div v-if="status.phase === 'confirming'" class="memory-warning" role="alert">
        <strong>Large JPEG AI encode</strong>
        <p>Estimated working memory: {{ ((status.confirmationBytes ?? 0) / 1073741824).toFixed(2) }} GiB. This may be slow or cause the browser to run out of memory.</p>
        <div class="warning-actions">
          <button class="encode-button" @click="emit('confirm')">Continue encoding</button>
          <button class="cancel-button" @click="emit('cancel')">Cancel</button>
        </div>
      </div>
      <template v-else>
        <button v-if="working" class="cancel-button" @click="emit('cancel')">{{ status.phase === 'queued' ? 'Cancel queued job' : 'Cancel encoding' }}</button>
        <button v-else class="encode-button" :disabled="!source || importing || modelBusy || (format.id === 'jpegai' && !modelsReady)" @click="emit('encode')">{{ result ? 'Encode again' : 'Encode' }}</button>
        <p v-if="source && format.id === 'jpegai' && !modelsReady && !modelBusy && !working" class="pane-status">{{ settings.rateMode === 'target' ? 'Download all four models to use automatic model selection.' : 'Download the selected model before encoding.' }}</p>
        <p v-if="working" class="pane-status" role="status" aria-live="polite">{{ status.message }}</p>
      </template>
    </div>
    <ResultDownloads v-if="result" :result="result" :file="file" />
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
  margin-bottom: 14px;
}

h2 {
  margin: 0;
  font-size: 20px;
  font-weight: 600;
}

.output-actions {
  display: grid;
  justify-items: center;
  gap: 10px;
  margin: 16px 0;
  padding: 12px;
}

.output-actions.no-output {
  min-height: 128px;
  align-content: center;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: var(--surface);
}

.output-actions button {
  min-width: 200px;
  padding: 16px 24px;
  border: 1px solid var(--line);
  border-radius: 6px;
  font-size: 16px;
  font-weight: 600;
}

.encode-button {
  color: var(--button-ink);
  background: var(--button-bg);
}

.cancel-button {
  color: var(--danger);
  background: var(--surface);
}

.memory-warning {
  max-width: 620px;
  padding: 18px;
  border: 1px solid var(--danger);
  border-radius: 6px;
  background: var(--error-bg);
  color: var(--error-ink);
  font-size: 14px;
  line-height: 1.6;
}

.warning-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
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
  margin-bottom: 20px;
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
