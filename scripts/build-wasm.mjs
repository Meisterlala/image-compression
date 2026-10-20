import { spawnSync } from 'node:child_process';
for (const [crate, output, features] of [
  ['codec', 'wasm', []],
  ['coolchic', 'wasm-coolchic', ['--features', 'wasm']],
]) {
  const result = spawnSync('wasm-pack', ['build', `crates/${crate}`, '--target', 'web', '--out-dir', `../../src/${output}`, '--out-name', crate, '--release', ...features], {
    stdio: 'inherit', env: { ...process.env, RUSTFLAGS: `${process.env.RUSTFLAGS ?? ''} -C target-feature=+simd128` },
  });
  if (result.status !== 0) process.exit(result.status ?? 1);
}
