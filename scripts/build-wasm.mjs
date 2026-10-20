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
// Keep a portable SIMD module as the fallback. Only this separate artifact
// requires atomics/shared memory and a rebuilt, thread-safe Rust stdlib.
const threadFlags = '-C target-feature=+simd128,+atomics,+bulk-memory -C link-arg=--shared-memory -C link-arg=--max-memory=4294967296 -C link-arg=--import-memory -C link-arg=--export=__wasm_init_tls -C link-arg=--export=__tls_size -C link-arg=--export=__tls_align -C link-arg=--export=__tls_base';
const threaded = spawnSync('rustup', ['run', 'nightly-2026-02-28', 'wasm-pack', 'build', 'crates/coolchic', '--target', 'web', '--out-dir', '../../src/wasm-coolchic-threads', '--out-name', 'coolchic', '--release', '--features', 'threads', '-Z', 'build-std=panic_abort,std'], {
  stdio: 'inherit', env: { ...process.env, RUSTFLAGS: `${process.env.RUSTFLAGS ?? ''} ${threadFlags}` },
});
if (threaded.status !== 0) process.exit(threaded.status ?? 1);
