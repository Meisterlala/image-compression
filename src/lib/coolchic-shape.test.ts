import { describe, expect, it } from 'vitest';
import { convolutionShape } from './coolchic-shape';

describe('GPU convolution size arithmetic', () => {
  it('accounts for input, parameters, gradients, readback and partials', () => {
    const shape = convolutionShape(new Uint32Array([2, 3, 17, 19, 3, 0, 1, 1]));
    expect(shape.weights).toBe(54);
    expect(shape.inputCount).toBe(646);
    expect(shape.outputCount).toBe(969);
    expect(shape.inputLength).toBe(646 + 54 + 3 + 969);
    expect(shape.resultCount).toBe(646 + 54 + 3);
    expect(shape.partialCount).toBe(57 * 2);
    expect(shape.transferBytes).toBe(32 + (shape.inputLength + shape.resultCount) * 4);
    expect(shape.workingBytes).toBe(shape.sizes.reduce((a, b) => a + b, shape.resultBytes));
  });
  it.each([
    [1, 1, 1, 1, 1, 0, 0],
    [0, 1, 1, 1, 1, 0, 0, 0],
    [1, 1, NaN, 1, 1, 0, 0, 0],
    [1, 1, Infinity, 1, 1, 0, 0, 0],
    [1, 1, 1.5, 1, 1, 0, 0, 0],
    [1, 1, 1, 1, 2, 0, 0, 0],
    [1, 2, 1, 1, 1, 1, 0, 0],
    [1, 1, 1, 1, 1, 0, 2, 0],
    [1, 1, 0xffffffff, 0xffffffff, 1, 0, 0, 0],
    [0xffffffff, 0xffffffff, 1, 1, 0xffffffff, 0, 0, 1],
  ])('rejects invalid/unsafe dimensions before allocation: %j', (...params) => {
    expect(() => convolutionShape(params)).toThrow(RangeError);
  });
  it('checks length before trying to materialize an enormous array-like', () => {
    expect(() => convolutionShape({ length: Number.MAX_SAFE_INTEGER })).toThrow(RangeError);
  });
});
