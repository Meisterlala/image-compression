import { describe, expect, it, vi } from 'vitest';
import { decodeJpegAi } from './jpegai-decode';

describe('GPU-first JPEG AI decoding', () => {
  const stream = new Uint8Array([1, 2]);
  const pixels = new Uint8Array([10, 20, 30, 255]);
  function decoder() {
    return {
      initGpu: vi.fn().mockResolvedValue(undefined),
      decodeGpu: vi.fn().mockResolvedValue(pixels),
      decode: vi.fn().mockReturnValue(pixels),
    };
  }
  it('always attempts WebGPU and reports success without a warning', async () => {
    const ai = decoder();
    expect(await decodeJpegAi(ai, stream)).toEqual({ pixels, backend: 'webgpu' });
    expect(ai.initGpu).toHaveBeenCalledOnce();
    expect(ai.decodeGpu).toHaveBeenCalledWith(stream);
    expect(ai.decode).not.toHaveBeenCalled();
  });
  it.each(['initialisation', 'decode'])('falls back after GPU %s failure', async stage => {
    const ai = decoder();
    (stage === 'initialisation' ? ai.initGpu : ai.decodeGpu).mockRejectedValue(new Error('device lost'));
    const result = await decodeJpegAi(ai, stream);
    expect(result.backend).toBe('cpu');
    expect(result.pixels).toBe(pixels);
    expect(result.warning).toContain('device lost');
    expect(ai.decode).toHaveBeenCalledWith(stream);
    if (stage === 'initialisation') expect(ai.decodeGpu).not.toHaveBeenCalled();
  });
  it('uses CPU without warning when GPU decoding is disabled', async () => {
    const ai = decoder();
    expect(await decodeJpegAi(ai, stream, false)).toEqual({ pixels, backend: 'cpu', warning: undefined });
    expect(ai.initGpu).not.toHaveBeenCalled();
    expect(ai.decodeGpu).not.toHaveBeenCalled();
    expect(ai.decode).toHaveBeenCalledWith(stream);
  });
  it('propagates CPU errors rather than returning a fake preview', async () => {
    const ai = decoder();
    ai.initGpu.mockRejectedValue(new Error('no adapter'));
    ai.decode.mockImplementation(() => { throw new Error('bad stream'); });
    await expect(decodeJpegAi(ai, stream)).rejects.toThrow('bad stream');
  });
});
