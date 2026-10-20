import { describe, expect, it, vi } from 'vitest';
import { canUseWebGpu } from './gpu-availability';

function gpuWithAdapter(adapter: unknown) {
  return { requestAdapter: vi.fn().mockResolvedValue(adapter) } as unknown as GPU;
}

describe('WebGPU availability', () => {
  it('rejects a missing API or unavailable adapter', async () => {
    expect(await canUseWebGpu(undefined)).toBe(false);
    expect(await canUseWebGpu(gpuWithAdapter(null))).toBe(false);
  });

  it('checks device creation and releases the probe device', async () => {
    const destroy = vi.fn();
    expect(await canUseWebGpu(gpuWithAdapter({ requestDevice: vi.fn().mockResolvedValue({ destroy }) }))).toBe(true);
    expect(destroy).toHaveBeenCalledOnce();
  });

  it('handles adapter and device failures without rejecting', async () => {
    expect(await canUseWebGpu({ requestAdapter: vi.fn().mockRejectedValue(new Error('blocked')) } as unknown as GPU)).toBe(false);
    expect(await canUseWebGpu(gpuWithAdapter({ requestDevice: vi.fn().mockRejectedValue(new Error('device unavailable')) }))).toBe(false);
  });
});
