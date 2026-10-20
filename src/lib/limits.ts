export const MAX_PIXELS = 40_000_000;
export const JPEG_AI_CONFIRM_THRESHOLD = 1536 * 1024 * 1024;
export const JPEG_AI_MEMORY_LIMIT = 3072 * 1024 * 1024;

export const downscaleOptions = [
  { value: 0, label: 'Full resolution' },
  { value: 2048, label: '2048' },
  { value: 1024, label: '1024' },
  { value: 512, label: '512' },
  { value: 256, label: '256' },
] as const;
