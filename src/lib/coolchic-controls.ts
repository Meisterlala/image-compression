export const coolchicPoints = ['lop', 'mop', 'hop', 'vhop'] as const;
export const coolchicPointDescriptions = [
  'Low Operating Point — fastest encoding and decoding, with the lowest memory use.',
  'Medium Operating Point — uses more time and memory to try to reduce the file size.',
  'High Operating Point — slower encoding and decoding, with potential file-size savings.',
  'Very High Operating Point — highest effort and memory use. May make smaller files, but takes the longest.',
] as const;
export const coolchicRateMin = 0.000001;
export const coolchicRateMax = 0.1;
export const coolchicIterationsMin = 100;
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

// Exact anchors: 100 at the left, 2000 at 25%, 100000 at the right.
export function iterationsFromSlider(position: number): number {
  const p = clamp(position, 0, 100, 0);
  return Math.round(p <= 25 ? 100 * 20 ** (p / 25) : 2000 * 50 ** ((p - 25) / 75));
}
export function iterationsToSlider(value: number): number {
  const iterations = clamp(value, coolchicIterationsMin, coolchicIterationsMax, 2000);
  return iterations <= 2000
    ? Math.log(iterations / 100) / Math.log(20) * 25
    : 25 + Math.log(iterations / 2000) / Math.log(50) * 75;
}
