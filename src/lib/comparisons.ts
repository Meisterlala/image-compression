import { computed, onBeforeUnmount, reactive, shallowReactive, type ShallowRef } from 'vue';
import { defaultSettings, formats, type Format, type Result, type Settings } from '../types';
import { startJob } from './runner';

export type Phase = 'idle' | 'queued' | 'running' | 'done' | 'cancelled' | 'error';
export interface PaneStatus { phase: Phase; message: string; error: string }

export function useComparisons(source: ShallowRef<ImageData | undefined>) {
  const settings = reactive(Object.fromEntries(formats.map(format => [
    format.id, { ...defaultSettings, format: format.id },
  ])) as Record<Format, Settings>);
  const results = shallowReactive<Partial<Record<Format, Result>>>({});
  const status = reactive(Object.fromEntries(formats.map(format => [
    format.id, { phase: 'idle', message: 'Not encoded yet.', error: '' },
  ])) as Record<Format, PaneStatus>);
  const busy = computed(() => formats.some(({ id }) => ['queued', 'running'].includes(status[id].phase)));
  const queue: { format: Format; settings: Settings }[] = [];
  let draining = false;
  let active: { format: Format; job: ReturnType<typeof startJob>; cancelled: boolean } | undefined;

  async function drain() {
    if (draining) return;
    draining = true;
    try {
      // Never benchmark competing encoders: one worker at a time, separate result per pane.
      while (queue.length) {
        const next = queue.shift()!;
        const state = status[next.format];
        const image = source.value;
        if (!image) { state.phase = 'idle'; continue; }
        state.phase = 'running';
        state.message = 'Starting WASM worker…';
        try {
          const job = startJob({ pixels: image.data, width: image.width, height: image.height, settings: next.settings }, text => state.message = text);
          const current = { format: next.format, job, cancelled: false };
          active = current;
          const output = await job.promise;
          if (!current.cancelled) {
            results[next.format] = output;
            state.phase = 'done';
            state.message = 'Done.';
          }
        } catch (error) {
          if (active?.cancelled) {
            state.phase = 'cancelled';
            state.message = 'Cancelled. Worker stopped.';
          } else {
            state.phase = 'error';
            state.message = 'Encoding failed.';
            state.error = error instanceof Error ? error.message : String(error);
          }
        } finally { active = undefined; }
      }
    } finally { draining = false; }
  }

  function run(format: Format) {
    const state = status[format];
    if (!source.value || state.phase === 'queued' || state.phase === 'running') return;
    state.phase = 'queued';
    state.error = '';
    state.message = 'Queued. Waiting for the current codec.';
    queue.push({ format, settings: { ...settings[format] } });
    void drain();
  }

  function runAll() {
    // Show fast baseline results before a potentially long JPEG AI rate search.
    for (const format of ['jpeg', 'avif', 'jxl', 'jpegai'] as const) run(format);
  }

  function cancel(format: Format) {
    const index = queue.findIndex(item => item.format === format);
    if (index !== -1) {
      queue.splice(index, 1);
      status[format].phase = 'cancelled';
      status[format].message = 'Cancelled before starting.';
    }
    if (active?.format === format) {
      active.cancelled = true;
      active.job.cancel();
    }
  }

  function cancelAll() { for (const { id } of formats) cancel(id); }

  function reset() {
    for (const { id } of formats) {
      delete results[id];
      Object.assign(status[id], { phase: 'idle', message: 'Not encoded yet.', error: '' });
    }
  }

  onBeforeUnmount(cancelAll);
  return { settings, results, status, busy, run, runAll, cancel, cancelAll, reset };
}
