import { computed, onBeforeUnmount, reactive, shallowReactive, type Ref, type ShallowRef } from 'vue';
import { defaultSettings, formats, type EncodeProgress, type Format, type Result, type Settings } from '../types';
import { startJob } from './runner';

export type Phase = 'idle' | 'queued' | 'running' | 'confirming' | 'done' | 'cancelled' | 'error';
export interface PaneStatus { progress?: EncodeProgress; phase: Phase; message: string; error: string; confirmationBytes: number | null }

export function useComparisons(source: ShallowRef<ImageData | undefined>, referenceBytes: Ref<number>, downscaled: Ref<boolean>) {
  const settings = reactive(Object.fromEntries(formats.map(format => [
    format.id, { ...defaultSettings, format: format.id },
  ])) as Record<Format, Settings>);
  const results = shallowReactive<Partial<Record<Format, Result>>>({});
  const status = reactive(Object.fromEntries(formats.map(format => [
    format.id, { phase: 'idle', message: '', error: '', confirmationBytes: null },
  ])) as Record<Format, PaneStatus>);
  const busy = computed(() => formats.some(({ id }) => ['queued', 'running', 'confirming'].includes(status[id].phase)));
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
          const job = startJob(
            { pixels: image.data, width: image.width, height: image.height, referenceBytes: referenceBytes.value, referenceDownscaled: downscaled.value, settings: next.settings },
            (text, progress) => { state.message = text; state.progress = progress; },
            memoryBytes => {
              state.confirmationBytes = memoryBytes;
              state.phase = 'confirming';
              state.message = '';
            },
          );
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
        } finally { state.confirmationBytes = null; state.progress = undefined; active = undefined; }
      }
    } finally { draining = false; }
  }

  function run(format: Format) {
    const state = status[format];
    if (!source.value || state.phase === 'queued' || state.phase === 'running' || state.phase === 'confirming') return;
    state.progress = undefined;
    state.phase = 'queued';
    state.error = '';
    state.message = 'Queued. Waiting for the current codec.';
    queue.push({ format, settings: { ...settings[format] } });
    void drain();
  }

  function confirm(format: Format) {
    if (active?.format !== format || status[format].phase !== 'confirming') return;
    status[format].phase = 'running';
    status[format].confirmationBytes = null;
    status[format].message = format === 'coolchic' ? 'Starting Cool-chic reference training…' : 'Loading JPEG AI models…';
    active.job.confirm();
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
      Object.assign(status[id], { phase: 'idle', message: '', error: '', confirmationBytes: null, progress: undefined });
    }
  }

  onBeforeUnmount(cancelAll);
  return { settings, results, status, busy, run, confirm, cancel, reset };
}
