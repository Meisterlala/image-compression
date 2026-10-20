interface Decoder {
  initGpu(): Promise<void>;
  decodeGpu(stream: Uint8Array): Promise<Uint8Array>;
  decode(stream: Uint8Array): Uint8Array;
}

/** GPU-first by default; an explicit opt-out uses CPU without a fallback warning. */
export async function decodeJpegAi(ai: Decoder, stream: Uint8Array, useGpu = true) {
  if (!useGpu) return { pixels: ai.decode(stream), backend: 'cpu' as const, warning: undefined };
  try {
    await ai.initGpu();
    const pixels = await ai.decodeGpu(stream);
    return { pixels, backend: 'webgpu' as const };
  } catch (error) {
    const reason = error instanceof Error ? error.message : String(error);
    return {
      pixels: ai.decode(stream),
      backend: 'cpu' as const,
      warning: `WebGPU decoding unavailable; used CPU instead. ${reason}`,
    };
  }
}
