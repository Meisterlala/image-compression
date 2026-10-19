import { spawnSync } from 'node:child_process';
const result = spawnSync('wasm-pack', ['build', 'crates/codec', '--target', 'web', '--out-dir', '../../src/wasm', '--out-name', 'codec', '--release'], {
  stdio: 'inherit', env: { ...process.env, RUSTFLAGS: `${process.env.RUSTFLAGS ?? ''} -C target-feature=+simd128` },
});
process.exit(result.status ?? 1);
