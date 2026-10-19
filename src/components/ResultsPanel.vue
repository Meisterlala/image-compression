<script setup lang="ts">
import { computed } from 'vue';
import { formats, type Result } from '../types';
import { bytes, compressionStats, download, draw, duration, runLabel } from '../lib/image';
const props = withDefaults(defineProps<{ result?: Result; file?: File; part?: 'primary' | 'details' }>(), { part: 'primary' });
const stats = computed(() => props.result ? compressionStats(props.result) : undefined);
function saveCompressed() {
  const r = props.result;
  if (!r) return;
  const f = formats.find(f => f.id === r.settings.format)!;
  download(new Blob([r.encoded], { type: f.mime }), `${props.file?.name.replace(/\.[^.]+$/, '') ?? 'image'}-${r.width}x${r.height}.${f.extension}`);
}
function savePreview() {
  const r = props.result;
  if (!r) return;
  const canvas = document.createElement('canvas');
  draw(canvas, new ImageData(r.pixels, r.width, r.height));
  canvas.toBlob(blob => { if (blob) download(blob, 'decoded-preview.png'); }, 'image/png');
}
</script>

<template>
  <section class="measurements" :aria-label="part === 'primary' ? 'Measurements' : 'File details'">
    <dl v-if="part === 'primary'" class="primary-stats">
      <div class="stat" data-metric="encode" :data-ms="result?.encodeMs">
        <dt>Encode time</dt><dd>{{ result ? duration(result.encodeMs) : '—' }}</dd>
      </div>
      <div class="stat" data-metric="decode" :data-ms="result?.decodeMs">
        <dt>Decode time</dt><dd>{{ result ? duration(result.decodeMs) : '—' }}</dd>
      </div>
      <div class="stat" data-metric="ratio" :data-ratio="stats?.compressionRatio">
        <dt>Compression ratio</dt><dd>{{ stats ? `${stats.compressionRatio.toFixed(2)}:1` : '—' }}</dd>
      </div>
    </dl>
    <template v-if="part === 'details' && result && stats">
      <dl class="secondary-stats">
        <div><dt>Encoded file</dt><dd>{{ bytes(result.encoded.length) }}</dd></div>
        <div><dt>Bits / pixel</dt><dd>{{ stats.bpp.toFixed(3) }}</dd></div>
        <div><dt>SSIM · luma</dt><dd>{{ result.metrics.ssim.toFixed(4) }}</dd></div>
        <div><dt>PSNR</dt><dd>{{ result.metrics.psnr === null ? '∞' : result.metrics.psnr.toFixed(2) }} dB</dd></div>
      </dl>
      <details class="measurement-details">
        <summary>Startup, quality & file details</summary>
        <dl class="runtime-details">
          <dt>Result settings</dt><dd>{{ runLabel(result.settings) }}</dd>
          <dt>Runtime / model startup</dt><dd>{{ duration(result.startupMs) }}</dd>
          <dt>Metrics computation</dt><dd>{{ duration(result.metricsMs) }}</dd>
          <dt>Raw working RGB</dt><dd>{{ bytes(stats.rawBytes) }}</dd>
          <template v-if="file">
            <dt>Original upload</dt><dd>{{ bytes(file.size) }}</dd>
            <dt>Upload / encoded</dt><dd>{{ (file.size / result.encoded.length).toFixed(2) }}:1 (dimensions may differ)</dd>
          </template>
          <dt>RGB mean squared error</dt><dd>{{ result.metrics.mse.toFixed(4) }}</dd>
          <template v-if="result.estimatedMemory">
            <dt>Estimated encode memory</dt><dd>{{ bytes(result.estimatedMemory) }} (not measured peak)</dd>
            <dt>Separate model bundles</dt><dd>{{ bytes(result.modelBytes) }}</dd>
          </template>
        </dl>
        <p class="hint">Encode and decode are timed separately around the actual WASM calls. Downloading/loading models and initializing WASM are startup work. SSIM uses 8×8 luma blocks; quality metrics compare the same working pixels.</p>
      </details>
      <div class="download-row">
        <button class="secondary-button" @click="saveCompressed">Download {{ formats.find(f => f.id === result?.settings.format)?.label }}</button>
        <button class="secondary-button" @click="savePreview">Preview PNG</button>
      </div>
    </template>
  </section>
</template>

<style scoped>
.primary-stats {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
  margin: 0;
}

.stat {
  min-width: 0;
  padding: 12px;
  border: 1px solid var(--line);
  border-radius: 5px;
  background: var(--surface);
}

.stat dt {
  font-size: 12px;
  line-height: 1.4;
}

.stat dd {
  margin-top: 8px;
  font: 500 22px/1.3 ui-monospace, monospace;
  font-variant-numeric: tabular-nums;
}

dt {
  color: var(--muted);
}

dd {
  margin: 0;
  overflow-wrap: anywhere;
}

.secondary-stats {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
  margin: 16px 0;
  font-size: 12px;
}

.secondary-stats dd {
  margin-top: 5px;
  font-family: ui-monospace, monospace;
}

.measurement-details {
  margin-top: 12px;
  font-size: 12px;
}

summary {
  cursor: pointer;
  color: var(--muted);
}

.runtime-details {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px;
  padding: 12px 0;
  font-size: 12px;
}

.runtime-details dd {
  text-align: right;
}

.hint {
  color: var(--muted);
  font-size: 11px;
  line-height: 1.6;
}

.download-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 14px;
}

button {
  padding: 8px 10px;
  border: 1px solid var(--line);
  border-radius: 5px;
  color: var(--ink);
  background: var(--paper);
  font-size: 12px;
}

@container (max-width: 420px) {
  .primary-stats {
    grid-template-columns: minmax(0, 1fr);
    gap: 0;
    overflow: hidden;
    border: 1px solid var(--line);
    border-radius: 5px;
  }

  .stat {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    padding: 10px;
    border: 0;
    border-bottom: 1px solid var(--line);
    border-radius: 0;
  }

  .stat:last-child {
    border-bottom: 0;
  }

  .stat dt {
    font-size: 12px;
  }

  .stat dd {
    margin: 0;
    font-size: 17px;
    white-space: nowrap;
  }

  .secondary-stats {
    grid-template-columns: 1fr 1fr;
  }
}
</style>
