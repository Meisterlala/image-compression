<script setup lang="ts">
import { formats, type Format, type Result } from '../types';
import { bytes, duration } from '../lib/image';
import { rankValue } from '../lib/ranking';
import MetricTooltip from './MetricTooltip.vue';

const props = defineProps<{ results: Partial<Record<Format, Result>>; source?: ImageData; referenceBytes: number; downscaled: boolean }>();
const emit = defineEmits<{ select: [format: Format | 'original'] }>();
const metrics = [
  { id: 'size', label: 'File size', direction: 'lower', text: 'Size of the image file, excluding codec and model downloads. Compare image quality too.' },
  { id: 'encode', label: 'Encode time', direction: 'lower', text: 'Time spent compressing the image. Model downloads, codec startup and time awaiting confirmation are not included.' },
  { id: 'decode', label: 'Decode time', direction: 'lower', text: 'Time spent reconstructing the image from the compressed file, measured separately from encoding.' },
  { id: 'psnr', label: 'PSNR', direction: 'higher', text: 'Peak signal-to-noise ratio: pixel differences from that encode’s input, in decibels (dB). ∞ means identical pixels. This is independent of the comparison dropdowns and does not always match perceived quality.' },
  { id: 'ssim', label: 'SSIM · luma', direction: 'higher', text: 'Structural similarity: compares brightness, contrast and structure against that encode’s input. Closer to 1 is better. This score uses brightness (luma), so it may miss colour differences.' },
  { id: 'bpp', label: 'Bits / pixel', direction: 'lower', text: 'Bits stored per pixel in the encoded output. Lower means fewer bytes at the same dimensions. Compare quality alongside size.' },
] as const;
type Metric = typeof metrics[number]['id'];

// Quality and bitrate ties use displayed precision, not invisible decimal differences.
function score(metric: Metric, result: Result): number {
  switch (metric) {
    case 'size': return result.encoded.length;
    case 'encode': return result.encodeMs;
    case 'decode': return result.decodeMs;
    case 'psnr': return result.metrics.psnr === null ? Infinity : Number(result.metrics.psnr.toFixed(2));
    case 'ssim': return Number(result.metrics.ssim.toFixed(4));
    case 'bpp': return Number((result.encoded.length * 8 / (result.width * result.height)).toFixed(3));
  }
}
function rank(metric: typeof metrics[number], result?: Result) {
  if (!result) return;
  const values = formats.flatMap(format => {
    const encoded = props.results[format.id];
    return encoded ? [score(metric.id, encoded)] : [];
  });
  return rankValue(score(metric.id, result), values, metric.direction);
}

function reduction(result?: Result) {
  return result?.referenceBytes ? (1 - result.encoded.length / result.referenceBytes) * 100 : undefined;
}
function value(metric: Metric, result?: Result): string {
  if (!result) return '';
  switch (metric) {
    case 'size': return bytes(result.encoded.length);
    case 'encode': return duration(result.encodeMs);
    case 'decode': return duration(result.decodeMs);
    case 'psnr': return `${result.metrics.psnr === null ? '∞' : result.metrics.psnr.toFixed(2)} dB`;
    case 'ssim': return result.metrics.ssim.toFixed(4);
    case 'bpp': return (result.encoded.length * 8 / (result.width * result.height)).toFixed(3);
  }
}
function savings(result: Result) {
  const saved = reduction(result);
  if (saved === undefined) return '';
  const input = result.referenceDownscaled ? 'downscaled PNG input' : 'uploaded file';
  return `${Math.abs(saved).toFixed(1)}% ${saved >= 0 ? 'smaller' : 'larger'} than the ${input} used for this encode (${result.width} × ${result.height}).`;
}
function original(metric: Metric): string {
  if (!props.source) return '';
  if (metric === 'size') return bytes(props.referenceBytes);
  if (metric === 'bpp') return (props.referenceBytes * 8 / (props.source.width * props.source.height)).toFixed(3);
  if (metric === 'psnr') return '∞ dB';
  if (metric === 'ssim') return '1.0000';
  // Historical encode/decode timings of the reference file are unknown.
  return '';
}
</script>

<template>
  <section class="comparison-card" aria-labelledby="comparison-title">
    <h2 id="comparison-title">Comparison table</h2>
    <div class="table-scroll" tabindex="0" aria-label="Scroll comparison table">
      <table aria-label="Image format comparison">
        <thead>
          <tr>
            <th scope="col">Format</th>
            <th v-for="metric in metrics" :key="metric.id" scope="col">
              <MetricTooltip :label="metric.label" :text="metric.text" :direction="metric.direction" />
            </th>
          </tr>
        </thead>
        <tbody>
          <tr class="original-row" data-result-format="original" :data-reference-bytes="referenceBytes">
            <th scope="row">
              <button class="format-link" @click="emit('select', 'original')">{{ downscaled ? 'Downscaled' : 'Original' }}</button>
              <small v-if="source" class="dimensions">{{ source.width }} × {{ source.height }}</small>
            </th>
            <td v-for="metric in metrics" :key="metric.id" :data-metric="metric.id">
              <MetricTooltip
                v-if="source && metric.id === 'size'"
                :label="original(metric.id)"
                :text="`${downscaled ? 'Downscaled reference measured as PNG' : 'Original uploaded file'} (${source.width} × ${source.height}).`"
                :info-icon="false"
              />
              <template v-else>{{ original(metric.id) }}</template>
            </td>
          </tr>
          <tr v-for="format in formats" :key="format.id" :data-result-format="format.id">
            <th scope="row">
              <button class="format-link" @click="emit('select', format.id)">{{ format.label }}</button>
              <small v-if="results[format.id]" class="dimensions">{{ results[format.id]!.width }} × {{ results[format.id]!.height }}</small>
            </th>
            <td
              v-for="metric in metrics"
              :key="metric.id"
              :data-metric="metric.id"
              :class="rank(metric, results[format.id])"
              :data-rank="rank(metric, results[format.id])"
              :data-score="results[format.id] ? score(metric.id, results[format.id]!) : undefined"
              :aria-label="rank(metric, results[format.id]) ? `${value(metric.id, results[format.id])}, ${rank(metric, results[format.id])} ${metric.label}` : undefined"
              :data-ms="metric.id === 'encode' ? results[format.id]?.encodeMs : metric.id === 'decode' ? results[format.id]?.decodeMs : undefined"
              :data-percent="metric.id === 'size' ? reduction(results[format.id]) : undefined"
            >
              <MetricTooltip v-if="metric.id === 'size' && results[format.id]" :label="value(metric.id, results[format.id])" :text="savings(results[format.id]!)" :info-icon="false" />
              <template v-else>{{ value(metric.id, results[format.id]) }}</template>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  </section>
</template>

<style scoped>
.comparison-card {
  margin-top: 22px;
  overflow: hidden;
  border: 1px solid var(--line);
  border-radius: var(--radius);
  background: var(--paper);
}

h2 {
  margin: 0;
  padding: 16px;
  font-size: 16px;
  font-weight: 600;
}

.table-scroll {
  overflow-x: auto;
}

table {
  width: 100%;
  min-width: 850px;
  border-collapse: collapse;
  font-size: 12px;
}

th,
td {
  height: 42px;
  padding: 11px 14px;
  border-top: 1px solid var(--line);
  text-align: left;
  white-space: nowrap;
}

thead th {
  padding-top: 12px;
  padding-bottom: 12px;
  background: var(--surface);
  font-weight: 500;
}

tbody td {
  font-family: ui-monospace, monospace;
  font-variant-numeric: tabular-nums;
}

.dimensions {
  display: block;
  margin-top: 5px;
  color: var(--muted);
  font-size: 10px;
  font-weight: 400;
}

.format-link {
  padding: 0;
  border: 0;
  background: none;
  color: inherit;
  font: inherit;
  text-decoration: underline;
  text-decoration-color: var(--muted);
  text-underline-offset: 4px;
}

.format-link:hover {
  color: var(--accent);
}

.best {
  color: var(--best-value);
}

.worst {
  color: var(--worst-value);
}

th:first-child {
  position: sticky;
  left: 0;
  z-index: 1;
  min-width: 100px;
  background: var(--paper);
}

thead th:first-child,
.original-row th,
.original-row td {
  background: var(--surface);
}
</style>
