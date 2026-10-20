import { downscaleOptions, MAX_PIXELS } from './limits';

export const MAX_FILE_BYTES = 40 * 1024 * 1024;

export function resizedDimensions(width: number, height: number, downscale: number) {
  if (!Number.isSafeInteger(width) || !Number.isSafeInteger(height) || width < 1 || height < 1) throw new Error('Invalid image dimensions.');
  if (width * height > MAX_PIXELS) throw new Error('Source exceeds 40 megapixels. Resize it before opening.');
  if (!downscaleOptions.some(option => option.value === downscale)) throw new Error('Invalid downscale option.');
  const ratio = downscale === 0 ? 1 : Math.min(1, downscale / Math.max(width, height));
  return {
    width: Math.max(1, Math.round(width * ratio)),
    height: Math.max(1, Math.round(height * ratio)),
  };
}

export interface PreparedImage { image: ImageData; referenceBytes: number; downscaled: boolean }

export async function prepareImage(file: File, downscale: number): Promise<PreparedImage> {
  if (!file.size || file.size > MAX_FILE_BYTES) throw new Error('Choose an image smaller than 40 MiB.');
  // Browser decoding is used only for importing user files. Every output codec and metric is WASM.
  const image = await createImageBitmap(file, { imageOrientation: 'from-image', premultiplyAlpha: 'none', colorSpaceConversion: 'default' });
  try {
    const { width, height } = resizedDimensions(image.width, image.height, downscale);
    const canvas = document.createElement('canvas'); canvas.width = width; canvas.height = height;
    const context = canvas.getContext('2d', { colorSpace: 'srgb', willReadFrequently: true });
    if (!context) throw new Error('A 2D canvas is required');
    context.fillStyle = '#ffffff'; context.fillRect(0, 0, width, height);
    context.imageSmoothingEnabled = true; context.imageSmoothingQuality = 'high';
    context.drawImage(image, 0, 0, width, height);
    const downscaled = width !== image.width || height !== image.height;
    // Measure an actual PNG of the resized reference, not its raw RGBA allocation.
    const referenceBytes = downscaled ? (await new Promise<Blob>((resolve, reject) => {
      canvas.toBlob(blob => blob ? resolve(blob) : reject(new Error('Could not measure the downscaled image.')), 'image/png');
    })).size : file.size;
    return { image: context.getImageData(0, 0, width, height), referenceBytes, downscaled };
  } finally { image.close(); }
}
export function draw(canvas: HTMLCanvasElement | null, image: ImageData) {
  if (!canvas) return;
  canvas.width = image.width; canvas.height = image.height;
  canvas.getContext('2d')!.putImageData(image, 0, 0);
}
export function download(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob); const link = document.createElement('a');
  link.href = url; link.download = filename; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 30_000);
}
export const bytes = (value: number) => value < 1024 ? `${value} B` : value < 1048576 ? `${(value / 1024).toFixed(1)} KiB` : `${(value / 1048576).toFixed(1)} MiB`;
export const duration = (ms: number) => ms < 1000 ? `${ms.toFixed(ms < 10 ? 2 : 1)} ms` : `${(ms / 1000).toFixed(2)} s`;

export function compressionStats(result: Pick<import('../types').Result, 'width' | 'height' | 'encoded'>) {
  const rawBytes = result.width * result.height * 3;
  return {
    rawBytes,
    compressionRatio: rawBytes / result.encoded.length,
    bpp: result.encoded.length * 8 / (result.width * result.height),
  };
}
export function runLabel(s: import('../types').Settings) {
  if (s.format === 'jpegai') return s.rateMode === 'target' ? `${s.targetBpp} bpp target · BOP` : `model ${s.model + 1} · shift ${s.displacement} · BOP`;
  return `${s.lossless && s.format !== 'jpeg' ? 'lossless' : `quality ${s.quality}`} · ${s.format === 'jpeg' ? (s.progressive ? 'progressive' : 'baseline') : `effort ${s.effort}`}`;
}
