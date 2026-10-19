export const MAX_FILE_BYTES = 40 * 1024 * 1024;
export const MAX_SOURCE_PIXELS = 40_000_000;
export async function prepareImage(file: File, maxEdge: number): Promise<ImageData> {
  if (!file.size || file.size > MAX_FILE_BYTES) throw new Error('Choose an image smaller than 40 MiB.');
  if (!Number.isFinite(maxEdge) || maxEdge < 32 || maxEdge > 1024) throw new Error('Working size must be between 32 and 1024 px.');
  // Browser decoding is used only for importing user files. Every output codec and metric is WASM.
  const image = await createImageBitmap(file, { imageOrientation: 'from-image', premultiplyAlpha: 'none', colorSpaceConversion: 'default' });
  try {
    if (image.width * image.height > MAX_SOURCE_PIXELS) throw new Error('Source exceeds 40 megapixels. Resize it before opening.');
    const ratio = Math.min(1, maxEdge / Math.max(image.width, image.height));
    const width = Math.max(1, Math.round(image.width * ratio));
    const height = Math.max(1, Math.round(image.height * ratio));
    const canvas = document.createElement('canvas'); canvas.width = width; canvas.height = height;
    const context = canvas.getContext('2d', { colorSpace: 'srgb', willReadFrequently: true });
    if (!context) throw new Error('A 2D canvas is required');
    context.fillStyle = '#ffffff'; context.fillRect(0, 0, width, height);
    context.imageSmoothingEnabled = true; context.imageSmoothingQuality = 'high';
    context.drawImage(image, 0, 0, width, height);
    return context.getImageData(0, 0, width, height);
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
