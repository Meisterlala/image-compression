export type Rank = 'best' | 'worst';

/** Compare only encoded results. Equal or solitary values remain neutral. */
export function rankValue(value: number, values: readonly number[], direction: 'higher' | 'lower'): Rank | undefined {
  const valid = values.filter(value => !Number.isNaN(value));
  if (valid.length < 2 || Number.isNaN(value)) return;
  const low = Math.min(...valid);
  const high = Math.max(...valid);
  if (low === high) return;
  if (value === (direction === 'higher' ? high : low)) return 'best';
  if (value === (direction === 'higher' ? low : high)) return 'worst';
  return;
}
