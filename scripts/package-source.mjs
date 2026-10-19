import { spawnSync } from 'node:child_process';
import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';

// A production build must not silently omit the main codec's weights.
const manifest = JSON.parse(await readFile('public/models/manifest.json', 'utf8'));
if (manifest.models?.length !== 4) throw new Error('Run npm run models before building.');
for (const model of manifest.models) {
  const bytes = await readFile(`public/models/${model.file}`);
  if (bytes.length !== model.bytes || createHash('sha256').update(bytes).digest('hex') !== model.sha256) {
    throw new Error(`Model ${model.file} missing or corrupt. Run npm run models.`);
  }
}
const args = [
  '-czf', 'public/source.tar.gz',
  '--exclude=crates/codec/target',
  '--exclude=src/wasm',
  'src', 'crates', 'scripts', 'tests', 'deploy', 'public/legal',
  'public/models/manifest.json', 'index.html', 'package.json', 'package-lock.json',
  'tsconfig.json', 'vite.config.ts', 'playwright.config.ts',
  'Dockerfile', '.dockerignore', '.gitignore', 'LICENSE', 'README.md',
];
const result = spawnSync('tar', args, { stdio: 'inherit' });
if (result.status !== 0) throw new Error('Failed to package corresponding source.');
console.log('Packaged source, tests, lockfiles and build/deployment scripts.');
