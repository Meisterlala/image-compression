import { describe, expect, it } from 'vitest';
import { defaultSettings } from '../types';
import { coolchicPoints, rateFromSlider, rateToSlider, iterationsFromSlider, iterationsToSlider } from './coolchic-controls';

describe('Cool-chic nonlinear sliders', () => {
  it('covers the reference control ranges and defaults', () => {
    expect(coolchicPoints).toEqual(['lop', 'mop', 'hop', 'vhop']);
    expect(rateFromSlider(0)).toBe(0.000001);
    expect(rateFromSlider(100)).toBe(0.1);
    expect(rateFromSlider(rateToSlider(defaultSettings.coolchicLambda))).toBe(0.001);
    expect(iterationsFromSlider(0)).toBe(100);
    expect(iterationsFromSlider(1)).toBeGreaterThan(100);
    expect(iterationsFromSlider(25)).toBe(2000);
    expect(iterationsFromSlider(100)).toBe(100000);
    expect(iterationsFromSlider(iterationsToSlider(defaultSettings.coolchicIterations))).toBe(2000);
  });
  it('uses monotonic logarithmic spacing rather than linear rate/budget steps', () => {
    expect(rateFromSlider(20)).toBe(0.00001);
    expect(rateFromSlider(40)).toBe(0.0001);
    expect(rateFromSlider(60)).toBe(0.001);
    for (let position = 1; position <= 100; position++) {
      expect(rateFromSlider(position)).toBeGreaterThan(rateFromSlider(position - 1));
      expect(iterationsFromSlider(position)).toBeGreaterThanOrEqual(iterationsFromSlider(position - 1));
      expect(rateToSlider(rateFromSlider(position))).toBeCloseTo(position, 3);
    }
  });
  it('round-trips real iteration values and clamps invalid persisted values', () => {
    for (const iterations of [500, 501, 700, 1000, 1500, 1999, 2000, 2100, 4000, 10000, 20000, 100000]) {
      expect(iterationsFromSlider(iterationsToSlider(iterations))).toBe(iterations);
    }
    expect(rateFromSlider(-100)).toBe(0.000001);
    expect(rateFromSlider(200)).toBe(0.1);
    expect(rateToSlider(Number.NaN)).toBe(60);
    expect(iterationsToSlider(Number.NaN)).toBe(25);
    expect(iterationsFromSlider(200)).toBe(100000);
  });
});
