import { describe, expect, it } from 'vitest';
import { rankValue } from './ranking';

describe('comparison ranks', () => {
  it('marks only the minimum and maximum for size and timing', () => {
    const values = [24, 10, 35, 16];
    expect(values.map(value => rankValue(value, values, 'lower'))).toEqual([undefined, 'best', 'worst', undefined]);
  });
  it('reverses the order for quality', () => {
    const values = [0.94, 0.83, 0.99, 0.96];
    expect(values.map(value => rankValue(value, values, 'higher'))).toEqual([undefined, 'worst', 'best', undefined]);
  });
  it('keeps absent, solitary and identical results neutral', () => {
    expect(rankValue(10, [], 'lower')).toBeUndefined();
    expect(rankValue(10, [10], 'lower')).toBeUndefined();
    expect(rankValue(10, [10, 10, 10], 'higher')).toBeUndefined();
  });
  it('marks all ties at a distinct best or worst', () => {
    const values = [10, 10, 20, 30, 30];
    expect(values.map(value => rankValue(value, values, 'lower'))).toEqual(['best', 'best', undefined, 'worst', 'worst']);
  });
  it('supports infinite PSNR for identical pixels', () => {
    const values = [32, Infinity, 40, Infinity];
    expect(values.map(value => rankValue(value, values, 'higher'))).toEqual(['worst', 'best', undefined, 'best']);
    expect(rankValue(Infinity, [Infinity, Infinity], 'higher')).toBeUndefined();
  });
  it('does not let an invalid value contaminate valid ranks', () => {
    expect(rankValue(10, [10, NaN, 20], 'lower')).toBe('best');
    expect(rankValue(NaN, [10, NaN, 20], 'lower')).toBeUndefined();
  });
});
