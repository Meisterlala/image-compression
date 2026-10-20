<script setup lang="ts">
import type { Settings } from '../types';
import MetricTooltip from './MetricTooltip.vue';
const settings = defineModel<Settings>({ required: true });
defineProps<{ disabled: boolean }>();
const id = (field: string) => `${settings.value.format}-${field}`;
</script>

<template>
  <fieldset class="settings-grid" :disabled="disabled">
    <legend class="sr-only">Encoding settings</legend>
    <template v-if="settings.format === 'jpegai'">
      <div class="field">
        <label class="sr-only" :for="id('operating-point')">Operating point</label>
        <div class="field-heading">
          <MetricTooltip label="Operating point" text="BOP (Base Operation Point) is the faster, lower-complexity option. HOP (High Operation Point) uses more complex neural transforms, aiming for better compression at similar quality, at the cost of slower decoding and higher memory use. Each uses its own model downloads; HOP is not guaranteed to produce a smaller file for every image or setting." />
        </div>
        <select :id="id('operating-point')" v-model="settings.operatingPoint">
          <option value="bop">BOP · Faster</option>
          <option value="hop">HOP · Higher complexity</option>
        </select>
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
  gap: 8px;
  margin-bottom: 8px;
  font-size: 12px;
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
  color: var(--ink);
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 6px;
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

@container (max-width: 400px) {
  .settings-grid {
    grid-template-columns: minmax(0, 1fr);
  }
}
</style>
