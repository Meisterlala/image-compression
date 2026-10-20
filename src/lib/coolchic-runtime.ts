import type * as Portable from '../wasm-coolchic/coolchic';
type CodecBindings = Pick<typeof Portable, 'CoolChicEncoder' | 'decodeCoolChic' | 'estimateCoolChicMemory' | 'setCoolChicAccelerator' | 'coolChicCpuConvolution' | 'CoolChicGraphProblem' | 'setCoolChicGraphAccelerator'>;
let threaded: Promise<{ bindings: CodecBindings; threads: number }> | undefined;
async function initThreads(count: number) {
  const bindings = await import('../wasm-coolchic-threads/coolchic');
  await bindings.default();
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    await Promise.race([
      bindings.initThreadPool(count),
      new Promise<never>((_, reject) => { timer = setTimeout(() => reject(new Error('CPU thread pool initialization timed out')), 15000); }),
    ]);
  } finally { clearTimeout(timer); }
  return { bindings, threads: count };
}

export function cpuThreadCount(cores: number, requested?: number) {
  if (requested !== undefined) {
    if (!Number.isInteger(requested) || requested < 1 || requested > 16) throw new RangeError('CPU thread count must be an integer from 1 to 16');
    return requested;
  }
  const available = Number.isSafeInteger(cores) && cores > 0 ? cores : 1;
  return Math.min(16, Math.max(1, available > 2 ? available - 1 : available));
}

/** Both modules expose the same codec API. Shared-memory WASM is a separate
 * lazy asset, so lack of isolation/threads never prevents portable encoding. */
export async function loadCoolChicRuntime(useThreads: boolean, requestedThreads?: number): Promise<{
  bindings: CodecBindings; threads: number; warning?: string;
}> {
  const count = cpuThreadCount(navigator.hardwareConcurrency, requestedThreads);
  let warning: string | undefined;
  if (useThreads && globalThis.crossOriginIsolated && typeof SharedArrayBuffer !== 'undefined') {
    if (count > 1) {
      try {
        // Feature test before fetching the threaded module.
        new WebAssembly.Memory({ initial: 1, maximum: 1, shared: true });
        threaded ??= initThreads(count);
        return await threaded;
      } catch (error) {
        warning = `CPU multithreading unavailable; using single-thread WASM. ${error instanceof Error ? error.message : String(error)}`;
      }
    }
  }
  const bindings = await import('../wasm-coolchic/coolchic');
  await bindings.default();
  return { bindings, threads: 1, warning };
}
