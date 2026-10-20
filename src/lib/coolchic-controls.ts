export const coolchicPoints = ['lop', 'mop', 'hop', 'vhop'] as const;
export const coolchicPointDescriptions = [
  'Fastest encoding and decoding, with the lowest memory use.',
  'Uses more time and memory to try to reduce the file size.',
  'Slower encoding and decoding, with potential file-size savings.',
  'Highest effort and memory use. May make smaller files, but takes the longest.',
] as const;
export const coolchicRateMin = 0.000001;
export const coolchicRateMax = 0.1;
export const coolchicIterationsMin = 500;
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

// Ease into the exponential range: most of the track controls short runs.
// The small linear term avoids a dead zone at the minimum.
export function iterationsFromSlider(position: number): number {
  const p = clamp(position, 0, 100, 0) / 100;
  const ratio = coolchicIterationsMax / coolchicIterationsMin;
  return Math.round(coolchicIterationsMin * ratio ** (0.15 * p + 0.85 * p * p));
}
export function iterationsToSlider(value: number): number {
  const iterations = clamp(value, coolchicIterationsMin, coolchicIterationsMax, 2000);
  const exponent = Math.log(iterations / coolchicIterationsMin) / Math.log(coolchicIterationsMax / coolchicIterationsMin);
  return (Math.sqrt(0.15 ** 2 + 3.4 * exponent) - 0.15) / 1.7 * 100;
}
