import { describe, expect, it } from 'vitest';
import { bytes, compressionStats, duration, resizedDimensions, runLabel } from './image';
import { defaultSettings } from '../types';

describe('downscale dimensions', () => {
  it('keeps the original resolution when Full is selected', () => {
    expect(resizedDimensions(6000, 4000, 0)).toEqual({ width: 6000, height: 4000 });
  });
  it('fits every preset without stretching landscape or portrait images', () => {
    for (const edge of [2048, 1024, 512, 256]) {
      expect(resizedDimensions(4096, 2048, edge)).toEqual({ width: edge, height: edge / 2 });
      expect(resizedDimensions(2048, 4096, edge)).toEqual({ width: edge / 2, height: edge });
    }
  });
  it('never enlarges a small source or rounds a thin image down to zero', () => {
    expect(resizedDimensions(96, 64, 2048)).toEqual({ width: 96, height: 64 });
    expect(resizedDimensions(1, 4000, 256)).toEqual({ width: 1, height: 256 });
  });
  it('rejects unsupported sizes and invalid or excessive source dimensions', () => {
    for (const edge of [-1, 768, NaN, Infinity]) expect(() => resizedDimensions(100, 100, edge)).toThrow();
    expect(() => resizedDimensions(0, 100, 0)).toThrow();
    expect(() => resizedDimensions(10000, 10000, 0)).toThrow();
  });
});

describe('display helpers', () => {
  it('uses binary file sizes', () => {
    expect(bytes(100)).toBe('100 B');
    expect(bytes(1024)).toBe('1.0 KiB');
    expect(bytes(1048576)).toBe('1.0 MiB');
  });
  it('separates milliseconds from seconds', () => {
    expect(duration(12.4)).toBe('12.4 ms');
    expect(duration(0.456)).toBe('0.46 ms');
    expect(duration(1234)).toBe('1.23 s');
  });
  it('defines compression ratio against identical raw RGB working pixels', () => {
    const stats = compressionStats({ width: 100, height: 80, encoded: new Uint8Array(2000) });
    expect(stats.rawBytes).toBe(24000);
    expect(stats.compressionRatio).toBe(12);
    expect(stats.bpp).toBe(2);
  });
  it('labels the saved experiment settings rather than current controls', () => {
    expect(runLabel(defaultSettings)).toContain('model 2');
    expect(runLabel({ ...defaultSettings, rateMode: 'target', targetBpp: 0.75 })).toContain('0.75 bpp target');
    expect(runLabel({ ...defaultSettings, format: 'jxl', lossless: true })).toContain('lossless');
  });
});
