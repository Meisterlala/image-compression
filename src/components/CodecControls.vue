<script setup lang="ts">
import { operatingPoints, type Settings } from '../types';
import MetricTooltip from './MetricTooltip.vue';
const settings = defineModel<Settings>({ required: true });
defineProps<{ disabled: boolean }>();
const id = (field: string) => `${settings.value.format}-${field}`;
</script>

<template>
  <fieldset class="settings-grid" :class="{ 'jpegai-settings': settings.format === 'jpegai' }" :disabled="disabled">
    <legend class="sr-only">Encoding settings</legend>
    <template v-if="settings.format === 'jpegai'">
      <div class="field">
        <div class="field-heading">
          <MetricTooltip label="Operating point" text="SOP — Simple Operation Point: the lightest decoder, useful for fast previews or resource-constrained devices; reconstruction quality can be lower. BOP — Base Operation Point: the balanced default for general-purpose use. HOP — High Operation Point: higher-complexity transforms for compression-priority work, at the cost of slower decoding and higher memory use. Better compression is an aim, not a guarantee for every image. Each point uses its own model downloads; only Download fetches weights." />
        </div>
        <div class="point-options" role="radiogroup" aria-label="Operating point">
          <label v-for="point in operatingPoints" :key="point.id" class="point-choice" :class="{ selected: settings.operatingPoint === point.id }">
            <input v-model="settings.operatingPoint" class="sr-only" type="radio" :name="id('operating-point')" :value="point.id">
            <span>{{ point.label }}</span>
          </label>
        </div>
      </div>
      <div class="field">
        <label :for="id('rate-mode')">Model selection</label>
        <select :id="id('rate-mode')" v-model="settings.rateMode">
          <option value="model">Use selected model</option>
          <option value="target">Choose model automatically</option>
        </select>
      </div>
      <template v-if="settings.rateMode === 'model'">
        <div class="field full-width">
          <label :for="id('displacement')">Quantisation shift <output>{{ settings.displacement }}</output></label>
          <input :id="id('displacement')" v-model.number="settings.displacement" type="range" min="-1069" max="702" step="1">
          <div class="range-captions"><span>Smaller file</span><span>Higher fidelity</span></div>
        </div>
      </template>
      <div v-else class="field">
        <label :for="id('target-bpp')">Target bits / pixel <output>{{ settings.targetBpp.toFixed(2) }}</output></label>
        <input :id="id('target-bpp')" v-model.number="settings.targetBpp" type="range" min="0.1" max="2" step="0.05">
      </div>
    </template>
    <template v-else-if="settings.format === 'coolchic'">
      <div class="field">
        <label :for="id('quality')">Quality <output>{{ settings.quality }}</output></label>
        <input :id="id('quality')" v-model.number="settings.quality" type="range" min="1" max="100">
      </div>
      <div class="field">
        <label :for="id('effort')">Training effort <output>{{ settings.effort }} · {{ 2 + settings.effort * 2 }} passes</output></label>
        <input :id="id('effort')" v-model.number="settings.effort" type="range" min="1" max="9">
      </div>
    </template>
    <template v-else>
      <div class="field">
        <label :for="id('quality')">Quality <output>{{ settings.quality }}</output></label>
        <input :id="id('quality')" v-model.number="settings.quality" :disabled="settings.lossless && settings.format !== 'jpeg'" type="range" min="1" max="100">
      </div>
      <div v-if="settings.format !== 'jpeg'" class="field">
        <label :for="id('effort')">Encoding effort <output>{{ settings.effort }}</output></label>
        <input :id="id('effort')" v-model.number="settings.effort" type="range" min="1" max="9">
      </div>
      <div v-if="settings.format !== 'jxl'" class="field">
        <label :for="id('chroma')">Chroma subsampling</label>
        <select :id="id('chroma')" v-model="settings.chroma" :disabled="settings.lossless && settings.format === 'avif'">
          <option value="444">4:4:4</option><option value="420">4:2:0</option>
        </select>
      </div>
      <label v-if="settings.format !== 'jpeg'" class="checkbox-row"><input v-model="settings.lossless" type="checkbox"> Lossless</label>
      <label v-if="settings.format !== 'avif'" class="checkbox-row"><input v-model="settings.progressive" type="checkbox"> Progressive encoding</label>
    </template>
  </fieldset>
</template>

<style scoped>
.settings-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 16px 20px;
  padding: 0;
  margin: 0;
  border: 0;
  min-width: 0;
}

.field {
  min-width: 0;
}

.field > label,
.field-heading {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 8px;
  min-height: 20px;
  margin-bottom: 8px;
  font-size: 12px;
  line-height: 20px;
}

.field-heading :deep(.metric-help) {
  display: inline-flex;
}

.field-heading :deep(button) {
  padding: 0;
  border-bottom: 0;
  color: var(--ink);
  text-decoration: underline dotted var(--muted);
  text-underline-offset: 4px;
}

.point-options {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  height: 56px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: var(--surface);
  overflow: hidden;
}

.point-choice {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  min-width: 0;
  padding: 0 8px;
  color: var(--ink);
  font-size: 12px;
  line-height: 16px;
  text-align: center;
  cursor: pointer;
}

.point-choice + .point-choice {
  border-left: 1px solid var(--line);
}

.point-choice.selected {
  font-weight: 600;
  background: var(--accent-soft);
  box-shadow: inset 0 0 0 1px var(--accent);
}

.point-choice:has(input:focus-visible) {
  box-shadow: inset 0 0 0 2px var(--accent);
}

fieldset:disabled .point-choice {
  cursor: default;
  opacity: 0.55;
}

.full-width {
  grid-column: 1 / -1;
}

.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  overflow: hidden;
  clip-path: inset(50%);
}

output {
  color: var(--accent);
  font-family: ui-monospace, monospace;
}

select {
  width: 100%;
  padding: 10px 8px;
  font-size: 12px;
  line-height: 20px;
  color: var(--ink);
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 6px;
}

.jpegai-settings select {
  height: 56px;
}

input[type="range"] {
  width: 100%;
  margin: 8px 0;
  accent-color: var(--accent);
}

.range-captions {
  display: flex;
  justify-content: space-between;
  font-size: 10px;
  color: var(--muted);
}

.hint {
  color: var(--muted);
  font-size: 11px;
  line-height: 1.6;
}

.checkbox-row {
  display: flex;
  align-items: center;
  gap: 8px;
  margin: 0;
  font-size: 12px;
}

input[type="checkbox"] {
  accent-color: var(--accent);
}

@container (max-width: 600px) {
  .jpegai-settings {
    grid-template-columns: minmax(0, 1fr);
  }
}

@container (max-width: 400px) {
  .settings-grid {
    grid-template-columns: minmax(0, 1fr);
  }
}
</style>
