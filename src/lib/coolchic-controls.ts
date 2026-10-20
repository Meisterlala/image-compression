export const coolchicPoints = ['lop', 'mop', 'hop', 'vhop'] as const;
export const coolchicPointDescriptions = [
  'Fastest encoding and decoding, with the lowest memory use.',
  'Uses more time and memory to try to reduce the file size.',
  'Slower encoding and decoding, with potential file-size savings.',
  'Highest effort and memory use. May make smaller files, but takes the longest.',
] as const;
export const coolchicRateMin = 0.000001;
export const coolchicRateMax = 0.1;
export const coolchicIterationsMin = 2000;
export const coolchicIterationsMax = 100000;

function clamp(value: number, min: number, max: number, fallback: number) {
  return Number.isFinite(value) ? Math.min(max, Math.max(min, value)) : fallback;
}

// Show the actual algorithm parameter, increasing numerically to the right.
export function rateFromSlider(position: number): number {
  return Number((10 ** (-6 + clamp(position, 0, 100, 60) / 20)).toPrecision(5));
}
export function rateToSlider(value: number): number {
  const lambda = clamp(value, coolchicRateMin, coolchicRateMax, 0.001);
  return (Math.log10(lambda) + 6) * 20;
}

// Logarithmic spacing keeps 2k–10k useful while still reaching a 100k budget.
export function iterationsFromSlider(position: number): number {
  const value = coolchicIterationsMin * 50 ** (clamp(position, 0, 100, 0) / 100);
  return Math.round(value / 100) * 100;
}
export function iterationsToSlider(value: number): number {
  const iterations = clamp(value, coolchicIterationsMin, coolchicIterationsMax, coolchicIterationsMin);
  return Math.log(iterations / coolchicIterationsMin) / Math.log(50) * 100;
}
