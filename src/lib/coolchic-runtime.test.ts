import { describe, expect, it } from 'vitest';
import { cpuThreadCount } from './coolchic-runtime';

describe('automatic WASM CPU pool size', () => {
  it.each([[1, 1], [2, 2], [3, 2], [8, 7], [16, 15], [32, 16], [NaN, 1], [0, 1]])('bounds %s reported cores to %s workers', (cores, expected) => {
    expect(cpuThreadCount(cores)).toBe(expected);
  });
  it.each([-1, 0, 1.5, 17, Infinity, NaN])('rejects unsupported explicit counts: %s', count => {
    expect(() => cpuThreadCount(16, count)).toThrow(RangeError);
  });
  it('allows supported diagnostic pool sizes', () => {
    expect(cpuThreadCount(16, 1)).toBe(1);
    expect(cpuThreadCount(16, 4)).toBe(4);
    expect(cpuThreadCount(16, 16)).toBe(16);
  });
});
