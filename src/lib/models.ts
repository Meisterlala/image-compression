import type { OperatingPoint } from '../types';

export const MODEL_CACHE = 'compression-models-v1';
export interface ModelEntry { id: number; file: string; bytes: number; sha256: string }
export interface ModelManifest { revision: string; operatingPoint: OperatingPoint; models: ModelEntry[] }
export const modelLabels = ['Lowest bitrate', 'Low bitrate', 'Medium bitrate', 'Highest fidelity'];

function modelUrl(entry: ModelEntry) {
  const url = new URL(`${import.meta.env.BASE_URL}models/${entry.file}`, globalThis.location.origin);
  url.searchParams.set('sha256', entry.sha256);
  return url.href;
}

export async function loadManifest(operatingPoint: OperatingPoint = 'bop'): Promise<ModelManifest> {
  if (operatingPoint !== 'bop' && operatingPoint !== 'hop') throw new Error('Invalid JPEG AI operating point');
  const file = operatingPoint === 'bop' ? 'manifest.json' : 'manifest-hop.json';
  const response = await fetch(`${import.meta.env.BASE_URL}models/${file}`, { cache: 'no-cache' });
  if (!response.ok || !response.headers.get('content-type')?.includes('json')) {
    throw new Error('JPEG AI models are missing. Run npm run models, then rebuild the static site.');
  }
  const manifest = await response.json() as ModelManifest;
  if (manifest.operatingPoint !== operatingPoint || manifest.models?.length !== 4 || manifest.models.some((m, i) =>
    m.id !== i || m.file !== `${operatingPoint}-m${i}.zjb` || !Number.isInteger(m.bytes) || m.bytes <= 0 || m.bytes > 128 * 1048576 || !/^[a-f0-9]{64}$/.test(m.sha256))) {
    throw new Error('Invalid JPEG AI model manifest');
  }
  return manifest;
}

export async function cachedModels(entries: ModelEntry[]): Promise<Set<number>> {
  const found = new Set<number>();
  try {
    const cache = await caches.open(MODEL_CACHE);
    for (const entry of entries) {
      const response = await cache.match(modelUrl(entry));
      if (response && response.headers.get('X-Model-SHA256') === entry.sha256) found.add(entry.id);
    }
  } catch { /* Cache API may be unavailable in private browsing. */ }
  return found;
}

export async function removeModel(entry: ModelEntry) {
  const cache = await caches.open(MODEL_CACHE);
  await cache.delete(modelUrl(entry));
}

/** The library and WASM worker share the same content-addressed, verified cache. */
export async function getModel(entry: ModelEntry, onProgress?: (received: number, total: number) => void, signal?: AbortSignal): Promise<{ bytes: Uint8Array<ArrayBuffer>; persisted: boolean }> {
  let cache: Cache | undefined;
  try { cache = await caches.open(MODEL_CACHE); } catch { /* Direct use still works. */ }
  const key = modelUrl(entry);
  let response = await cache?.match(key);
  const fromCache = !!response;
  if (!response) {
    response = await fetch(key, { signal });
    if (!response.ok) throw new Error(`Model download failed (${response.status}). Run npm run models before serving the site.`);
  }
  const buffer = new Uint8Array(entry.bytes);
  let received = 0;
  if (!response.body) throw new Error('The model response has no data');
  const reader = response.body.getReader();
  const abort = () => { void reader.cancel(); };
  signal?.addEventListener('abort', abort, { once: true });
  try {
    for (;;) {
      signal?.throwIfAborted();
      const { done, value } = await reader.read();
      if (done) break;
      if (received + value.length > entry.bytes) throw new Error('Model exceeds its declared size');
      buffer.set(value, received);
      received += value.length;
      onProgress?.(received, entry.bytes);
    }
    signal?.throwIfAborted();
  } finally {
    signal?.removeEventListener('abort', abort);
    await reader.cancel();
  }
  const digest = await crypto.subtle.digest('SHA-256', buffer);
  const hex = Array.from(new Uint8Array(digest), b => b.toString(16).padStart(2, '0')).join('');
  if (received !== entry.bytes || hex !== entry.sha256) {
    await cache?.delete(key);
    throw new Error('JPEG AI model integrity check failed. Download it again.');
  }
  let persisted = fromCache;
  if (!fromCache && cache) {
    try {
      await cache.put(key, new Response(buffer, { headers: { 'Content-Type': 'application/octet-stream', 'X-Model-SHA256': entry.sha256 } }));
      persisted = true;
    } catch { /* Quota full: verified bytes are still usable by the caller. */ }
  }
  return { bytes: buffer, persisted };
}
