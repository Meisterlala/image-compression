import { afterEach, describe, expect, it, vi } from 'vitest';
import { createCoolChicAcceleration } from './coolchic-acceleration';

class MockWorker {
  static instances: MockWorker[] = [];
  onmessage?: (event: { data: { type: string; message?: string } }) => void;
  onerror?: (event: { message: string }) => void;
  terminated = false;
  fail = false;
  jobs = 0;
  output = new Float32Array([7, 10]);
  constructor() { MockWorker.instances.push(this); }
  terminate() { this.terminated = true; }
  postMessage(data: { type: string; shared?: SharedArrayBuffer; inputLength?: number; outputLength?: number }) {
    if (data.type === 'init') queueMicrotask(() => this.onmessage?.({ data: { type: 'ready' } }));
    if (data.type === 'convolution') {
      this.jobs++;
      const shared = data.shared!;
      if (!this.fail) new Float32Array(shared, 32 + data.inputLength! * 4, data.outputLength).set(this.output);
      Atomics.store(new Int32Array(shared, 0, 8), 0, this.fail ? -1 : 1);
    }
  }
}
function environment() {
  MockWorker.instances = [];
  vi.stubGlobal('crossOriginIsolated', true);
  vi.stubGlobal('WorkerGlobalScope', class { static [Symbol.hasInstance]() { return true; } });
  vi.stubGlobal('Worker', MockWorker);
  vi.stubGlobal('navigator', { gpu: {} });
}
const params = new Uint32Array([1, 1, 1, 2, 1, 0, 0, 0]);
const input = new Float32Array([1, 2]), weight = new Float32Array([3]), bias = new Float32Array([4]);
afterEach(() => { vi.restoreAllMocks(); vi.unstubAllGlobals(); });

describe('Cool-chic WASM compute bridge', () => {
  it('rejects without isolation before allocating workers', async () => {
    environment(); vi.stubGlobal('crossOriginIsolated', false);
    await expect(createCoolChicAcceleration()).rejects.toThrow('cross-origin isolation');
    expect(MockWorker.instances).toHaveLength(0);
  });
  it('returns real shared output and disposes the helper', async () => {
    environment(); const backend = await createCoolChicAcceleration();
    expect(backend.convolution(params, input, weight, bias, null)).toEqual(new Float32Array([7, 10]));
    expect(backend.calls).toBe(1);
    backend.dispose();
    expect(MockWorker.instances[0]!.terminated).toBe(true);
    expect(backend.convolution(params, input, weight, bias, null)).toBeNull();
  });
  it('has a synchronous borrowed-response contract and copied results survive buffer reuse', async () => {
    environment(); const backend = await createCoolChicAcceleration();
    const borrowed = backend.convolution(params, input, weight, bias, null)!;
    expect(borrowed).toBeInstanceOf(Float32Array); // Not a Promise: no await/re-entry before Rust copies.
    const consumed = new Float32Array(borrowed); // Equivalent to Rust's immediate copy_to(output).
    MockWorker.instances[0]!.output.set([13, 16]);
    const later = backend.convolution(params, input, weight, bias, null);
    expect(later).toEqual(new Float32Array([13, 16]));
    expect(consumed).toEqual(new Float32Array([7, 10]));
    backend.dispose();
  });
  it('rejects malformed payloads before posting compute work', async () => {
    environment(); const backend = await createCoolChicAcceleration();
    expect(() => backend.convolution(params, new Float32Array(0), weight, bias, null)).toThrow('payload');
    expect(MockWorker.instances[0]!.jobs).toBe(0);
    backend.dispose();
  });
  it('bounds hangs and disables the accelerator after timeout', async () => {
    environment(); const backend = await createCoolChicAcceleration();
    vi.spyOn(Atomics, 'wait').mockReturnValue('timed-out');
    expect(backend.convolution(params, input, weight, bias, null)).toBeNull();
    expect(backend.failure).toContain('timed out');
    expect(MockWorker.instances[0]!.terminated).toBe(true);
    expect(backend.convolution(params, input, weight, bias, null)).toBeNull();
    expect(MockWorker.instances[0]!.jobs).toBe(1);
  });
  it('falls back after device/worker computation failure', async () => {
    environment(); const backend = await createCoolChicAcceleration();
    MockWorker.instances[0]!.fail = true;
    expect(backend.convolution(params, input, weight, bias, null)).toBeNull();
    expect(backend.failure).toContain('failed');
    expect(backend.calls).toBe(0);
  });
  it.each([true, false])('calibrates once per shape and selects GPU only with headroom (%s)', async faster => {
    environment(); const cpu = vi.fn(() => new Float32Array([7, 10]));
    const backend = await createCoolChicAcceleration(cpu);
    vi.spyOn(performance, 'now').mockReturnValueOnce(0).mockReturnValueOnce(faster ? 4 : 6)
      .mockReturnValueOnce(10).mockReturnValueOnce(20);
    const first = backend.convolution(params, input, weight, bias, null);
    expect(first).toEqual(faster ? new Float32Array([7, 10]) : null);
    const second = backend.convolution(params, input, weight, bias, null);
    expect(second).toEqual(first);
    expect(cpu).toHaveBeenCalledTimes(1);
    expect(backend.calibrations).toHaveLength(1);
    expect(backend.calls).toBe(faster ? 2 : 0);
    expect(MockWorker.instances[0]!.jobs).toBe(faster ? 3 : 2);
    backend.dispose();
  });
});
