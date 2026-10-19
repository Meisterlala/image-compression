<script setup lang="ts">
import { ref } from 'vue';

defineProps<{
  filename?: string;
  width?: number;
  height?: number;
  disabled: boolean;
}>();
const size = defineModel<number>('size', { required: true });
const emit = defineEmits<{ choose: [file: File]; resize: [] }>();
const input = ref<HTMLInputElement>();
const dragging = ref(false);

function choose(files?: FileList | null) {
  if (files?.[0]) emit('choose', files[0]);
  if (input.value) input.value.value = '';
}
function drop(event: DragEvent) {
  dragging.value = false;
  choose(event.dataTransfer?.files);
}
</script>

<template>
  <section class="input-bar" aria-label="Image input">
    <div
      class="dropzone"
      :class="{ dragging }"
      @dragover.prevent="dragging = true"
      @dragleave.prevent="dragging = false"
      @drop.prevent="drop"
    >
      <button :disabled="disabled" @click="input?.click()">{{ filename ? 'Change image' : 'Choose image' }}</button>
      <div class="file-description">
        <strong>{{ filename ?? 'Or drop an image here' }}</strong>
        <span>{{ width && height ? `${width} × ${height} working pixels` : 'PNG, JPEG, WebP, AVIF… · up to 40 MiB' }}</span>
      </div>
      <input
        ref="input"
        class="file-input"
        type="file"
        accept="image/*"
        aria-label="Choose image file"
        :disabled="disabled"
        @change="choose(($event.target as HTMLInputElement).files)"
      >
    </div>
    <div class="working-size">
      <label for="working-size">Working image max edge</label>
      <select id="working-size" v-model.number="size" :disabled="disabled" @change="emit('resize')">
        <option :value="256">256 px</option>
        <option :value="512">512 px</option>
        <option :value="768">768 px</option>
        <option :value="1024">1024 px</option>
      </select>
    </div>
    <p class="hint">Same 8-bit sRGB pixels for every codec. Alpha is flattened on white; large images are resized and metadata is removed.</p>
  </section>
</template>

<style scoped>
.input-bar {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 12px 24px;
  padding: 16px;
  background: var(--paper);
  border: 1px solid var(--line);
  border-radius: var(--radius);
}

.dropzone {
  display: flex;
  align-items: center;
  gap: 16px;
  min-width: 0;
  padding: 8px;
  border: 1px dashed var(--line);
  border-radius: 5px;
}

.dropzone.dragging {
  background: var(--accent-soft);
  border-color: var(--accent);
}

.dropzone button {
  flex-shrink: 0;
  padding: 10px 12px;
  color: var(--ink);
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 5px;
  font-size: 13px;
}

.file-description {
  display: grid;
  gap: 5px;
  min-width: 0;
}

.file-description strong {
  font-size: 13px;
  font-weight: 500;
  overflow-wrap: anywhere;
}

.file-description span {
  color: var(--muted);
  font-size: 11px;
}

.file-input {
  position: absolute;
  width: 1px;
  height: 1px;
  overflow: hidden;
  clip-path: inset(50%);
}

.working-size {
  display: grid;
  gap: 8px;
}

label {
  font-size: 12px;
}

select {
  padding: 8px;
  color: var(--ink);
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 5px;
  font-size: 12px;
}

.hint {
  grid-column: 1 / -1;
  margin: 0;
  color: var(--muted);
  font-size: 11px;
  line-height: 1.6;
}

@media (max-width: 560px) {
  .input-bar {
    grid-template-columns: minmax(0, 1fr);
  }

  .working-size {
    grid-template-columns: 1fr auto;
    align-items: center;
  }

  .dropzone {
    gap: 10px;
  }
}
</style>
