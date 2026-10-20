<script setup lang="ts">
import MetricTooltip from './MetricTooltip.vue';
import { computed } from 'vue';
import type { Settings } from '../types';
import { coolchicPoints, coolchicPointDescriptions, rateFromSlider, rateToSlider, iterationsFromSlider, iterationsToSlider } from '../lib/coolchic-controls';
const settings = defineModel<Settings>({ required: true });
defineProps<{ disabled: boolean }>();
const id = (field: string) => `${settings.value.format}-${field}`;
const coolchicPoint = computed(() => Math.max(0, coolchicPoints.indexOf(settings.value.coolchicPoint)));
const coolchicRate = computed({
  get: () => rateToSlider(settings.value.coolchicLambda),
  set: (value: number) => { settings.value.coolchicLambda = rateFromSlider(value); },
});
const coolchicIterations = computed({
  get: () => iterationsToSlider(settings.value.coolchicIterations),
  set: (value: number) => { settings.value.coolchicIterations = iterationsFromSlider(value); },
});
</script>

<template>
  <fieldset class="settings-grid" :class="{ 'jpegai-settings': settings.format === 'jpegai' }" :disabled="disabled">
    <legend class="sr-only">Encoding settings</legend>
    <template v-if="settings.format === 'jpegai'">
      <div class="field">
        <label :for="id('rate-mode')">Model selection</label>
        <select :id="id('rate-mode')" v-model="settings.rateMode">
          <option value="model">Use selected model</option>
          <option value="target">Choose model automatically</option>
        </select>
      </div>
      <template v-if="settings.rateMode === 'model'">
        <div class="field">
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
      <div class="field coolchic-point-field">
        <div class="control-heading">
          <span :id="id('complexity-label')">Compression preset</span>
          <MetricTooltip label="Compression preset" text="LOP is the fastest and uses the least memory. MOP, HOP and VHOP spend progressively more time and memory trying to improve compression. They can also take longer to decode. A higher preset does not guarantee a smaller file or a better image." />
        </div>
        <p :id="id('complexity-description')" class="control-description" aria-live="polite">{{ coolchicPointDescriptions[coolchicPoint] }}</p>
        <div class="point-options" role="radiogroup" :aria-labelledby="id('complexity-label')" :aria-describedby="id('complexity-description')" :style="{ '--point-index': coolchicPoint }">
          <label v-for="point in coolchicPoints" :key="point" class="point-choice" :class="{ selected: settings.coolchicPoint === point }">
            <input v-model="settings.coolchicPoint" type="radio" :name="id('operating-point')" :value="point" :aria-label="point.toUpperCase()">
            <span>{{ point.toUpperCase() }}</span>
          </label>
        </div>
      </div>
      <div class="field">
        <div class="control-heading">
          <label :for="id('lambda')">Quality</label>
          <MetricTooltip label="Quality" text="Higher settings preserve more detail but usually make larger files. Lower settings make smaller files and can lose detail. Even the highest setting is not lossless." />
          <output :for="id('lambda')">{{ Math.round(coolchicRate) }}</output>
        </div>
        <input :id="id('lambda')" v-model.number="coolchicRate" type="range" min="0" max="100" step="1" aria-label="Quality" :aria-valuetext="`${Math.round(coolchicRate)} out of 100`">
        <div class="range-captions"><span>Smaller file</span><span>Higher fidelity</span></div>
      </div>
      <div class="field">
        <div class="control-heading">
          <label :for="id('iterations')">Training iterations</label>
          <MetricTooltip label="Training iterations" text="More iterations can improve quality or reduce file size, but encoding takes longer. Minimum: 2,000." />
          <output :for="id('iterations')">{{ settings.coolchicIterations.toLocaleString('en-US') }}</output>
        </div>
        <input :id="id('iterations')" v-model.number="coolchicIterations" type="range" min="0" max="100" step="1" aria-label="Training iterations" :aria-valuetext="`${settings.coolchicIterations} iterations`">
        <div class="range-captions"><span>Faster</span><span>More effort</span></div>
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

.field > label {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 8px;
  min-height: 20px;
  margin-bottom: 8px;
  font-size: 12px;
  line-height: 20px;
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

.control-heading {
  display: flex;
  align-items: center;
  gap: 6px;
  min-height: 20px;
  margin-bottom: 8px;
  font-size: 12px;
}

.coolchic-point-field { grid-column: 1 / -1; }

.control-description {
  margin: 0 0 12px;
  color: var(--muted);
  font-size: 12px;
  line-height: 1.6;
}

.point-options {
  position: relative;
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  padding: 3px;
  border: 1px solid var(--line);
  border-radius: 7px;
  background: var(--surface);
}

.point-options::before {
  content: '';
  box-sizing: border-box;
  position: absolute;
  top: 3px;
  bottom: 3px;
  left: 3px;
  width: calc((100% - 6px) / 4);
  border: 1px solid var(--accent);
  border-radius: 4px;
  background: var(--accent-soft);
  transform: translateX(calc(var(--point-index) * 100%));
  transition: transform 160ms ease;
  pointer-events: none;
}

.point-choice {
  position: relative;
  z-index: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 4px;
  min-height: 54px;
  padding: 8px 4px;
  color: var(--muted);
  font-size: 12px;
  white-space: nowrap;
}

.point-choice.selected { color: var(--accent); font-weight: 600; }
.point-choice input {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  margin: 0;
  opacity: 0;
  cursor: pointer;
}
.point-choice:has(input:focus-visible) { outline: 2px solid var(--accent); outline-offset: 2px; border-radius: 4px; }
fieldset:disabled .point-options { opacity: 0.55; }
fieldset:disabled .point-choice input { cursor: default; }
@media (prefers-reduced-motion: reduce) { .point-options::before { transition: none; } }

.control-heading > output { margin-left: auto; }
.control-heading :deep(.tooltip-trigger) { font-size: 0; border: 0; }
.control-heading :deep(.info-icon) { font-size: 12px; margin-left: 0; }


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
