const sum = (...values: number[]) => {
  const result = values.reduce((a, b) => a + b, 0);
  if (!Number.isSafeInteger(result)) throw new RangeError('GPU convolution size overflow');
  return result;
};
const product = (...values: number[]) => {
  const result = values.reduce((a, b) => a * b, 1);
  if (!Number.isSafeInteger(result)) throw new RangeError('GPU convolution size overflow');
  return result;
};

/** Check all host arithmetic before buffer allocation or u32 shader indexing. */
export function convolutionShape(params: ArrayLike<number>) {
  if (params.length !== 8) throw new RangeError('Invalid GPU convolution parameters');
  const values = Array.from(params);
  if (values.some(v => !Number.isSafeInteger(v) || v < 0 || v > 0xffffffff)) {
    throw new RangeError('Invalid GPU convolution parameters');
  }
  const [ci, co, h, w, k, residual, relu, backward] = values;
  if (!ci || !co || !h || !w || !k || k % 2 !== 1 || [residual, relu, backward].some(v => v > 1)
      || (residual && ci !== co)) throw new RangeError('Invalid GPU convolution');
  const pixels = product(h, w);
  const weights = product(ci, co, k, k);
  const inputCount = product(ci, pixels), outputCount = product(co, pixels);
  const parameters = sum(weights, co), chunks = Math.ceil(pixels / 256);
  const inputLength = sum(inputCount, parameters, backward ? outputCount : 0);
  const resultCount = backward ? sum(inputCount, parameters) : outputCount;
  const partialCount = backward ? product(parameters, chunks) : 1;
  const sizes = [product(inputLength, 4), product(outputCount, 4), backward ? product(resultCount, 4) : 4, product(partialCount, 4)];
  const resultBytes = product(resultCount, 4);
  return { ci, co, h, w, k, residual, relu, backward, pixels, weights, chunks,
    inputCount, outputCount, inputLength, resultCount, parameters, partialCount, sizes, resultBytes,
    transferBytes: sum(32, sizes[0], resultBytes), workingBytes: sum(...sizes, resultBytes) };
}
