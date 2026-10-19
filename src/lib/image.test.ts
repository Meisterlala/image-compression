import { describe, expect, it } from 'vitest';
import { bytes, compressionStats, duration, runLabel } from './image';
import { defaultSettings } from '../types';

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
