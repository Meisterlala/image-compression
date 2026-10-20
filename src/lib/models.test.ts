import { afterEach, describe, expect, it, vi } from 'vitest';
import { loadManifest, type ModelManifest } from './models';
import type { OperatingPoint } from '../types';

function manifest(op: OperatingPoint): ModelManifest {
  return { revision: 'pinned', operatingPoint: op, models: Array.from({ length: 4 }, (_, id) => ({
    id, file: `${op}-m${id}.zjb`, bytes: op === 'hop' ? 57_069_952 : 27_000_000, sha256: 'a'.repeat(64),
  })) };
}
function serve(data: unknown) {
  const fetcher = vi.fn().mockResolvedValue(new Response(JSON.stringify(data), { headers: { 'Content-Type': 'application/json' } }));
  vi.stubGlobal('fetch', fetcher);
  return fetcher;
}
afterEach(() => vi.unstubAllGlobals());

describe('operating-point model manifests', () => {
  it('defaults to BOP, preserving its existing URLs and cache keys', async () => {
    const data = manifest('bop'); const fetcher = serve(data);
    expect(await loadManifest()).toEqual(data);
    expect(fetcher.mock.calls[0]![0]).toMatch(/\/models\/manifest\.json$/);
  });
  it('loads the separate HOP manifest with its larger bundles', async () => {
    const data = manifest('hop'); const fetcher = serve(data);
    expect(await loadManifest('hop')).toEqual(data);
    expect(fetcher.mock.calls[0]![0]).toMatch(/\/models\/manifest-hop\.json$/);
  });
  it('rejects a different point rather than silently falling back', async () => {
    serve(manifest('bop'));
    await expect(loadManifest('hop')).rejects.toThrow('Invalid JPEG AI model manifest');
  });
  it('rejects cross-point model URLs, incomplete sets and invalid integrity metadata', async () => {
    const crossPoint = manifest('hop'); crossPoint.models[0]!.file = 'bop-m0.zjb';
    const incomplete = manifest('hop'); incomplete.models.pop();
    const oversized = manifest('hop'); oversized.models[0]!.bytes = 129 * 1048576;
    const badHash = manifest('hop'); badHash.models[0]!.sha256 = 'bad';
    for (const data of [crossPoint, incomplete, oversized, badHash]) {
      serve(data);
      await expect(loadManifest('hop')).rejects.toThrow('Invalid JPEG AI model manifest');
    }
  });
});
