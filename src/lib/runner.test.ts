import { afterEach, expect, it, vi } from 'vitest';
import { startJob } from './runner';
import { defaultSettings, type WorkerMessage } from '../types';

afterEach(() => vi.unstubAllGlobals());

it.each(['jpegai', 'coolchic'] as const)('reports runtime GPU failure for %s without stopping the job', async format => {
  let receive: (event: { data: WorkerMessage }) => void;
  const terminate = vi.fn();
  vi.stubGlobal('Worker', class {
    set onmessage(handler: typeof receive) { receive = handler; }
    postMessage() {}
    terminate = terminate;
  });
  const onGpuFailure = vi.fn();
  const onProgress = vi.fn();
  const job = startJob({
    pixels: new Uint8ClampedArray(4), width: 1, height: 1,
    referenceBytes: 4, referenceDownscaled: false,
    settings: { ...defaultSettings, format },
  }, onProgress, vi.fn(), onGpuFailure);
  receive!({ data: { type: 'gpu-failure' } });
  expect(onGpuFailure).toHaveBeenCalledOnce();
  expect(terminate).not.toHaveBeenCalled();
  receive!({ data: { type: 'progress', text: 'Continuing on CPU' } });
  expect(onProgress).toHaveBeenCalledWith('Continuing on CPU', undefined);
  const cancelled = expect(job.promise).rejects.toThrow('Compression cancelled');
  job.cancel();
  await cancelled;
});
