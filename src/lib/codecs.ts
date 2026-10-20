import type { Format, Settings } from '../types';
import jpegEncoderWasm from '@jsquash/jpeg/codec/enc/mozjpeg_enc.wasm?url';
import jpegDecoderWasm from '@jsquash/jpeg/codec/dec/mozjpeg_dec.wasm?url';
import avifEncoderWasm from '@jsquash/avif/codec/enc/avif_enc.wasm?url';
import avifDecoderWasm from '@jsquash/avif/codec/dec/avif_dec.wasm?url';
import jxlEncoderWasm from '@jsquash/jxl/codec/enc/jxl_enc.wasm?url';
import jxlDecoderWasm from '@jsquash/jxl/codec/dec/jxl_dec.wasm?url';

// Dependency prebundling changes import.meta.url in dev. Resolve binaries
// explicitly instead of guessing paths relative to the prebundled module.
const wasmOptions = (url: string) => ({ locateFile: () => new URL(url, import.meta.url).href });

// Explicit single-threaded factories avoid nested Emscripten workers and work without
// SharedArrayBuffer/cross-origin isolation. All transforms still execute in WASM.
export async function loadCodec(format: Exclude<Format, 'jpegai'>, s: Settings, progress: (text: string) => void = () => {}) {
  if (format === 'coolchic') {
    const { loadCoolChic } = await import('./coolchic');
    return loadCoolChic(s, progress);
  }
  if (format === 'jpeg') {
    const [{ default: factory }, { defaultOptions }, { default: decode, init: initDecoder }] = await Promise.all([
      import('@jsquash/jpeg/codec/enc/mozjpeg_enc.js'), import('@jsquash/jpeg/meta.js'), import('@jsquash/jpeg/decode.js'),
    ]);
    const [module] = await Promise.all([factory(wasmOptions(jpegEncoderWasm)), initDecoder(wasmOptions(jpegDecoderWasm))]);
    return { encode: (image: ImageData) => {
      const out = module.encode(image.data, image.width, image.height, { ...defaultOptions, quality: s.quality, progressive: s.progressive, auto_subsample: false, chroma_subsample: s.chroma === '420' ? 2 : 1 });
      if (!out) throw new Error('JPEG encoding failed');
      return new Uint8Array(out); // Copy out of the Emscripten heap before the next call.
    }, decode };
  }
  if (format === 'avif') {
    const [{ default: factory }, { defaultOptions }, { default: decode, init: initDecoder }] = await Promise.all([
      import('@jsquash/avif/codec/enc/avif_enc.js'), import('@jsquash/avif/meta.js'), import('@jsquash/avif/decode.js'),
    ]);
    const [module] = await Promise.all([factory(wasmOptions(avifEncoderWasm)), initDecoder(wasmOptions(avifDecoderWasm))]);
    return { encode: (image: ImageData) => {
      const out = module.encode(image.data, image.width, image.height, { ...defaultOptions, quality: s.lossless ? 100 : s.quality, speed: 10 - s.effort, subsample: s.lossless || s.chroma === '444' ? 3 : 1, qualityAlpha: -1 });
      if (!out) throw new Error('AVIF encoding failed');
      return new Uint8Array(out);
    }, decode };
  }
  const [{ default: factory }, { defaultOptions }, { default: decode, init: initDecoder }] = await Promise.all([
    import('@jsquash/jxl/codec/enc/jxl_enc.js'), import('@jsquash/jxl/meta.js'), import('@jsquash/jxl/decode.js'),
  ]);
  const [module] = await Promise.all([factory(wasmOptions(jxlEncoderWasm)), initDecoder(wasmOptions(jxlDecoderWasm))]);
  return { encode: (image: ImageData) => {
    const out = module.encode(image.data, image.width, image.height, { ...defaultOptions, quality: s.lossless ? 100 : s.quality, effort: s.effort, lossless: s.lossless, lossyModular: false, lossyPalette: false, progressive: s.progressive });
    if (!out) throw new Error('JPEG XL encoding failed');
    return new Uint8Array(out);
  }, decode };
}
