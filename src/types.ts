export type OperatingPoint = 'bop' | 'hop';
export type Format = 'jpegai' | 'jpeg' | 'avif' | 'jxl';
export const formats: { id: Format; label: string; extension: string; mime: string; description: string }[] = [
  { id: 'jpegai', label: 'JPEG AI', extension: 'jai', mime: 'application/octet-stream', description: 'zenjpegai · BOP / HOP · experimental' },
  { id: 'jpeg', label: 'JPEG', extension: 'jpg', mime: 'image/jpeg', description: 'MozJPEG' },
  { id: 'avif', label: 'AVIF', extension: 'avif', mime: 'image/avif', description: 'libavif / libaom' },
  { id: 'jxl', label: 'JPEG XL', extension: 'jxl', mime: 'image/jxl', description: 'libjxl' },
];
export interface Settings { operatingPoint: OperatingPoint; format: Format; quality: number; effort: number; lossless: boolean; progressive: boolean; chroma: '420' | '444'; model: number; displacement: number; rateMode: 'model' | 'target'; targetBpp: number }
export const defaultSettings: Settings = { operatingPoint: 'bop', format: 'jpegai', quality: 75, effort: 5, lossless: false, progressive: true, chroma: '444', model: 1, displacement: 0, rateMode: 'model', targetBpp: 0.5 };
export interface Metrics { mse: number; psnr: number | null; ssim: number }
export interface Job { referenceBytes: number; referenceDownscaled: boolean; pixels: Uint8ClampedArray<ArrayBuffer>; width: number; height: number; settings: Settings }
export interface Result { referenceBytes: number; referenceDownscaled: boolean; settings: Settings; width: number; height: number; encoded: Uint8Array<ArrayBuffer>; pixels: Uint8ClampedArray<ArrayBuffer>; metrics: Metrics; startupMs: number; encodeMs: number; decodeMs: number; metricsMs: number; estimatedMemory: number | null; modelBytes: number; notices: string[] }
export type WorkerMessage = { type: 'progress'; text: string } | { type: 'result'; result: Result } | { type: 'error'; error: string } | { type: 'confirmation-required'; memoryBytes: number };
