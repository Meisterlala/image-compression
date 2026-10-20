export type OperatingPoint = 'sop' | 'bop' | 'hop';
export const operatingPoints: { id: OperatingPoint; label: string; description: string }[] = [
  { id: 'sop', label: 'Simple Operation Point', description: 'Simple Operation Point (SOP): Fast previews and resource-constrained devices. Lightest decoder; reconstruction quality can be lower.' },
  { id: 'bop', label: 'Base Operation Point', description: 'Base Operation Point (BOP): Balanced default for general-purpose use.' },
  { id: 'hop', label: 'High Operation Point', description: 'High Operation Point (HOP): Compression-priority work; slower decoding and higher memory. Smaller files are not guaranteed.' },
];
export type Format = 'jpegai' | 'jpeg' | 'avif' | 'jxl' | 'coolchic';
export const formats: { id: Format; label: string; extension: string; mime: string; description: string }[] = [
  { id: 'jpegai', label: 'JPEG AI', extension: 'jai', mime: 'application/octet-stream', description: 'zenjpegai · SOP / BOP / HOP · experimental' },
  { id: 'jpeg', label: 'JPEG', extension: 'jpg', mime: 'image/jpeg', description: 'MozJPEG' },
  { id: 'avif', label: 'AVIF', extension: 'avif', mime: 'image/avif', description: 'libavif / libaom' },
  { id: 'jxl', label: 'JPEG XL', extension: 'jxl', mime: 'image/jxl', description: 'libjxl' },
  { id: 'coolchic', label: 'Cool-chic 5.0', extension: 'cool', mime: 'application/octet-stream', description: 'Rust/WASM · reference intra MSE training' },
];
export type CoolChicPoint = 'lop' | 'mop' | 'hop' | 'vhop';
export interface Settings { coolchicPoint: CoolChicPoint; coolchicLambda: number; coolchicIterations: number; coolchicGpu?: boolean; gpuDecode?: boolean; operatingPoint: OperatingPoint; format: Format; quality: number; effort: number; lossless: boolean; progressive: boolean; chroma: '420' | '444'; model: number; displacement: number; rateMode: 'model' | 'target'; targetBpp: number }
export const defaultSettings: Settings = { coolchicPoint: 'lop', coolchicLambda: 0.001, coolchicIterations: 2000, coolchicGpu: true, gpuDecode: true, operatingPoint: 'bop', format: 'jpegai', quality: 75, effort: 5, lossless: false, progressive: true, chroma: '444', model: 1, displacement: 0, rateMode: 'model', targetBpp: 0.5 };
export interface Metrics { mse: number; psnr: number | null; ssim: number }
export interface Job { referenceBytes: number; referenceDownscaled: boolean; pixels: Uint8ClampedArray<ArrayBuffer>; width: number; height: number; settings: Settings }
export interface Result { referenceBytes: number; referenceDownscaled: boolean; settings: Settings; width: number; height: number; encoded: Uint8Array<ArrayBuffer>; pixels: Uint8ClampedArray<ArrayBuffer>; metrics: Metrics; startupMs: number; encodeMs: number; decodeMs: number; metricsMs: number; estimatedMemory: number | null; modelBytes: number; decodeBackend?: 'webgpu' | 'cpu'; decodeWarning?: string; notices: string[] }
export interface EncodeProgress { stage: 'training' | 'quantizing'; completed: number; total: number; elapsedMs: number }
export type WorkerMessage = { type: 'progress'; text: string; progress?: EncodeProgress } | { type: 'result'; result: Result } | { type: 'error'; error: string } | { type: 'confirmation-required'; memoryBytes: number };
