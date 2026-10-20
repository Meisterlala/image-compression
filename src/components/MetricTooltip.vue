<script setup lang="ts">
import { nextTick, onBeforeUnmount, onMounted, ref, useId, type CSSProperties } from 'vue';

withDefaults(defineProps<{ label: string; text: string; direction?: 'higher' | 'lower'; infoIcon?: boolean; asChild?: boolean }>(), { infoIcon: true, asChild: false });
const id = useId();
const visible = ref(false);
const trigger = ref<HTMLElement>();
const popup = ref<HTMLSpanElement>();
const position = ref<CSSProperties>({ visibility: 'hidden' });
let timer: ReturnType<typeof setTimeout> | undefined;
let version = 0;
function keepOpen() { clearTimeout(timer); }
function hide() {
  keepOpen();
  version++;
  visible.value = false;
}
function place() {
  if (!visible.value || !trigger.value || !popup.value) return;
  const bounds = trigger.value.getBoundingClientRect();
  if (bounds.bottom < 0 || bounds.top > innerHeight || bounds.right < 0 || bounds.left > innerWidth) { hide(); return; }
  const width = Math.min(260, innerWidth - 24);
  const height = popup.value.offsetHeight;
  const above = bounds.top - height - 8;
  position.value = {
    width: `${width}px`,
    left: `${Math.max(12, Math.min(bounds.left, innerWidth - width - 12))}px`,
    top: `${above >= 12 ? above : Math.max(12, Math.min(bounds.bottom + 8, innerHeight - height - 12))}px`,
    visibility: 'visible',
  };
}
async function show() {
  keepOpen();
  const current = ++version;
  position.value = { ...position.value, width: `${Math.min(260, innerWidth - 24)}px`, visibility: 'hidden' };
  visible.value = true;
  await nextTick();
  if (current === version) place();
}
function leave() { keepOpen(); timer = setTimeout(hide, 120); }
function outside(event: PointerEvent) {
  const target = event.target as Node;
  if (!trigger.value?.contains(target) && !popup.value?.contains(target)) hide();
}
function escape(event: KeyboardEvent) { if (event.key === 'Escape') hide(); }
onMounted(() => {
  document.addEventListener('pointerdown', outside);
  document.addEventListener('keydown', escape);
  window.addEventListener('scroll', place, true);
  window.addEventListener('resize', place);
});
onBeforeUnmount(() => {
  keepOpen();
  document.removeEventListener('pointerdown', outside);
  document.removeEventListener('keydown', escape);
  window.removeEventListener('scroll', place, true);
  window.removeEventListener('resize', place);
});
</script>

<template>
  <span ref="trigger" class="metric-help" @mouseenter="show" @mouseleave="leave">
    <slot v-if="asChild" v-bind="{ tooltipId: id, showTooltip: show, hideTooltip: hide }" />
    <button
      v-else
      class="tooltip-trigger"
      :class="{ 'metric-value': !infoIcon }"
      type="button"
      :aria-label="`About ${label}`"
      :aria-describedby="id"
      @focus="show"
      @blur="hide"
      @click="show"
      @keydown.esc.stop.prevent="hide"
    >
      {{ label }} <span v-if="infoIcon" class="info-icon" aria-hidden="true">ⓘ</span>
    </button>
    <Teleport to="body">
      <span v-show="visible" :id="id" ref="popup" role="tooltip" class="tooltip" :style="position" @mouseenter="keepOpen" @mouseleave="leave">
        <strong v-if="direction" class="direction">{{ direction === 'higher' ? 'Higher is better' : 'Lower is better' }}</strong>
        {{ text }}
      </span>
    </Teleport>
  </span>
</template>

<style scoped>
.metric-help {
  position: relative;
  display: inline-block;
}

.tooltip-trigger {
  padding: 2px 0;
  border: 0;
  border-bottom: 1px dotted var(--muted);
  background: transparent;
  color: var(--muted);
  font: inherit;
  cursor: help;
}

.tooltip-trigger.metric-value {
  color: inherit;
  border-bottom-color: currentColor;
}

.info-icon {
  margin-left: 3px;
  color: var(--accent);
}

.direction {
  display: block;
  margin-bottom: 5px;
  font-size: 12px;
  font-weight: 600;
}

.tooltip {
  position: fixed;
  z-index: 1000;
  width: 260px;
  max-width: calc(100vw - 24px);
  padding: 12px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: var(--paper);
  color: var(--ink);
  box-shadow: 0 4px 16px #0003;
  font-size: 12px;
  font-weight: 400;
  line-height: 1.6;
  white-space: pre-line;
}

</style>
