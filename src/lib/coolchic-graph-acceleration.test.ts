import { afterEach, describe, expect, it, vi } from 'vitest';
import { createCoolChicGraphAcceleration, GRAPH_HEADER, type GraphAcceleration } from './coolchic-graph-acceleration';

class MockWorker {
  static instances: MockWorker[] = [];
  onmessage?: (event: { data: { type: string; error?: string } }) => void;
  onerror?: (event: { message: string }) => void;
  terminated = false;
  fail = false;
  requests: { fresh: boolean; target: number; common: number }[] = [];
  output = new Float32Array([7, 10]);
  constructor() { MockWorker.instances.push(this); }
  terminate() { this.terminated = true; }
  postMessage(data: { type: string; shared?: SharedArrayBuffer; inputLength?: number; outputLength?: number; fresh: boolean; target: number; common: number }) {
    if (data.type === 'init') queueMicrotask(() => this.onmessage?.({ data: { type: 'ready' } }));
    if (data.type === 'gradient') {
      this.requests.push(data);
      const shared = data.shared!;
      if (!this.fail) new Float32Array(shared, GRAPH_HEADER + data.inputLength! * 4, data.outputLength).set(this.output.subarray(0, data.outputLength));
      Atomics.store(new Int32Array(shared, 0, 8), 0, this.fail ? -1 : 1);
    }
  }
}
function environment() {
  MockWorker.instances = [];
  vi.stubGlobal('crossOriginIsolated', true);
  vi.stubGlobal('Worker', MockWorker);
  vi.stubGlobal('navigator', { gpu: {} });
}
const spec = JSON.stringify({ group_lengths: [1], latents: [{ height: 1, width: 1 }] });
const parameters = new Float32Array([3]), latents = new Float32Array([1]);
const target = new Float32Array([1, 2, 3]), noise = new Float32Array([0]), common = new Float32Array(0);
const options = new Float32Array([2, 0.35, 0.001, 0]);
const gradient = (backend: GraphAcceleration) => backend.gradient(spec, parameters, latents, target, noise, common, options);
afterEach(() => { vi.restoreAllMocks(); vi.unstubAllGlobals(); });

describe('Cool-chic GPU graph bridge', () => {
  it.each(['isolation', 'gpu'])('rejects missing %s before allocating workers', async missing => {
    environment();
    if (missing === 'isolation') vi.stubGlobal('crossOriginIsolated', false);
    else vi.stubGlobal('navigator', {});
    await expect(createCoolChicGraphAcceleration()).rejects.toThrow('isolated WebGPU workers');
    expect(MockWorker.instances).toHaveLength(0);
  });
  it('returns shared gradients and disposes the worker', async () => {
    environment(); const backend = await createCoolChicGraphAcceleration();
    expect(gradient(backend)).toEqual(new Float32Array([7, 10]));
    expect(backend.calls).toBe(1);
    backend.dispose();
    expect(MockWorker.instances[0]!.terminated).toBe(true);
    expect(gradient(backend)).toBeNull();
  });
  it('copies borrowed responses synchronously before buffer reuse', async () => {
    environment(); const backend = await createCoolChicGraphAcceleration();
    const borrowed = gradient(backend)!;
    expect(borrowed).toBeInstanceOf(Float32Array);
    const consumed = new Float32Array(borrowed);
    MockWorker.instances[0]!.output.set([13, 16]);
    expect(gradient(backend)).toEqual(new Float32Array([13, 16]));
    expect(consumed).toEqual(new Float32Array([7, 10]));
    backend.dispose();
  });
  it('sends fixed target/common data only when the graph changes', async () => {
    environment(); const backend = await createCoolChicGraphAcceleration();
    gradient(backend); gradient(backend);
    backend.gradient(JSON.stringify({ group_lengths: [1], latents: [{ height: 1, width: 1 }], name: 'another' }), parameters, latents, target, noise, common, options);
    expect(MockWorker.instances[0]!.requests.map(({ fresh, target }) => [fresh, target])).toEqual([[true, 3], [false, 0], [true, 3]]);
    backend.dispose();
  });
  it('tracks hard-round RD evaluations separately from gradients', async () => {
    environment(); const backend = await createCoolChicGraphAcceleration();
    expect(backend.gradient(spec, parameters, latents, target, new Float32Array(0), common, new Float32Array([5, 0, 0.001, 0]))).toEqual(new Float32Array([7]));
    expect(backend.evaluations).toBe(1);
    expect(backend.calls).toBe(0);
    backend.dispose();
  });
  it('rejects malformed payloads before posting compute work', async () => {
    environment(); const backend = await createCoolChicGraphAcceleration();
    expect(backend.gradient(spec, new Float32Array(0), latents, target, noise, common, options)).toBeNull();
    expect(backend.failure).toContain('input length mismatch');
    expect(MockWorker.instances[0]!.requests).toHaveLength(0);
    expect(MockWorker.instances[0]!.terminated).toBe(true);
  });
  it('bounds hangs and disables the graph after timeout', async () => {
    environment(); const backend = await createCoolChicGraphAcceleration();
    vi.spyOn(Atomics, 'wait').mockReturnValue('timed-out');
    expect(gradient(backend)).toBeNull();
    expect(backend.failure).toContain('timed out');
    expect(MockWorker.instances[0]!.terminated).toBe(true);
    expect(gradient(backend)).toBeNull();
    expect(MockWorker.instances[0]!.requests).toHaveLength(1);
  });
  it('falls back after device/worker computation failure', async () => {
    environment(); const backend = await createCoolChicGraphAcceleration();
    MockWorker.instances[0]!.fail = true;
    expect(gradient(backend)).toBeNull();
    expect(backend.failure).toContain('failed');
    expect(backend.calls).toBe(0);
    expect(MockWorker.instances[0]!.terminated).toBe(true);
  });
});
