import type { TensorGraphSpec, GraphArchitecture } from '../../src/lib/coolchic-tensor-graph';

// This test-only adapter uses the public pinned fixture architecture; production
// obtains causal offsets and latent descriptors directly from Rust.
export function specForFixture(g: { architecture: GraphArchitecture; groups: number[][]; ifce_inputs: number[] }): TensorGraphSpec {
  const a = g.architecture;
  const min = Math.min(a.latent_min, a.hyperlatent?.[0] ?? a.latent_min);
  const max = Math.max(a.latent_max, a.hyperlatent?.[1] ?? a.latent_max);
  const latents: TensorGraphSpec['latents'] = [];
  for (let level = min; level <= max; level++) {
    const shape = { height: Math.ceil(a.height / 2 ** level), width: Math.ceil(a.width / 2 ** level) };
    if (level >= a.latent_min && level <= a.latent_max) latents.push({ ...shape, hyper: false });
    if (a.hyperlatent && level >= a.hyperlatent[0] && level <= a.hyperlatent[1]) latents.push({ ...shape, hyper: true });
  }
  const priorities = [38,35,30,25,23,31,36,37,39,33,28,21,20,6,15,22,29,34,32,18,12,10,5,9,14,19,27,24,13,8,2,1,3,11,17,26,16,7,4,0];
  const neighbours = Array.from({ length: 40 }, (_, i) => i).sort((a, b) => priorities[a] - priorities[b])
    .slice(0, a.spatial_context).map(i => [Math.floor(i / 9) - 4, i % 9 - 4] as [number, number]);
  return { architecture: a, latents, ifce_inputs: g.ifce_inputs, neighbours, group_lengths: g.groups.map(values => values.length) };
}
export function commonForSpec(spec: TensorGraphSpec) {
  if (!spec.architecture.common_randomness) return new Float32Array(0);
  const count = spec.latents.filter(l => !l.hyper).reduce((n, l) => n + l.height * l.width, 0);
  const values = new Float32Array(count);
  let seed = 18101995;
  for (let i = 0; i < count; i++) {
    seed = 16807 * seed % 2147483647; const u1 = seed / 2147483647;
    seed = 16807 * seed % 2147483647; const u2 = seed / 2147483647;
    values[i] = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * 3.14159265359 * u2);
  }
  return values;
}
