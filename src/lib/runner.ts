import type { Job, Result, WorkerMessage } from '../types';
export function startJob(job: Job, onProgress: (text: string) => void, onConfirmation: (memoryBytes: number) => void) {
  const worker = new Worker(new URL('../codec.worker.ts', import.meta.url), { type: 'module' });
  let rejectJob: (reason: Error) => void;
  let settled = false;
  const promise = new Promise<Result>((resolve, reject) => {
    rejectJob = reject;
    const fail = (message: string) => { settled = true; worker.terminate(); reject(new Error(message)); };
    worker.onmessage = ({ data }: MessageEvent<WorkerMessage>) => {
      if (data.type === 'progress') onProgress(data.text);
      if (data.type === 'confirmation-required') onConfirmation(data.memoryBytes);
      if (data.type === 'error') fail(data.error);
      if (data.type === 'result') { settled = true; worker.terminate(); resolve(data.result); }
    };
    worker.onerror = (e) => { e.preventDefault(); fail(e.message || 'WASM worker stopped. Try a smaller working image or a current browser with WASM SIMD.'); };
    worker.onmessageerror = () => fail('Could not receive worker result');
    // Copy: retaining the source is necessary for repeat experiments and comparisons.
    const pixels = new Uint8ClampedArray(job.pixels);
    worker.postMessage({ ...job, pixels }, [pixels.buffer]);
  });
  return {
    promise,
    confirm() { if (!settled) worker.postMessage({ type: 'confirm' }); },
    cancel() { if (!settled) { settled = true; worker.terminate(); rejectJob(new DOMException('Compression cancelled', 'AbortError')); } },
  };
}
