import { spawnSync } from 'node:child_process';
import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';

// A production build must not silently omit the main codec's weights.
for (const [op, file] of [['sop', 'manifest-sop.json'], ['bop', 'manifest.json'], ['hop', 'manifest-hop.json']]) {
  const manifest = JSON.parse(await readFile(`public/models/${file}`, 'utf8'));
  if (manifest.operatingPoint !== op || !Array.isArray(manifest.models) || manifest.models.length !== 4 ||
      manifest.models.some((model, id) => model.id !== id || model.file !== `${op}-m${id}.zjb`)) {
    throw new Error(`Invalid or incomplete ${op.toUpperCase()} manifest. Run npm run models before building.`);
  }
  for (const model of manifest.models) {
    const bytes = await readFile(`public/models/${model.file}`);
    if (bytes.length !== model.bytes || createHash('sha256').update(bytes).digest('hex') !== model.sha256) {
      throw new Error(`Model ${model.file} missing or corrupt. Run npm run models.`);
    }
  }
}
const args = [
  '-czf', 'public/source.tar.gz',
  '--exclude=crates/codec/target',
  '--exclude=crates/coolchic/target',
  '--exclude=src/wasm',
  '--exclude=src/wasm-coolchic',
  '--exclude=src/wasm-coolchic-threads',
  'src', 'crates', 'scripts', 'tests', 'docs', 'deploy', 'public/legal',
  'public/icon.svg', 'public/favicon.ico', 'public/apple-touch-icon.png',
  'public/robots.txt', 'public/sitemap.xml',
  'public/models/manifest.json', 'public/models/manifest-hop.json', 'public/models/manifest-sop.json', 'index.html', 'package.json', 'package-lock.json',
  'tsconfig.json', 'vite.config.ts', 'playwright.config.ts',
  'Dockerfile', '.dockerignore', '.gitignore', 'LICENSE', 'README.md',
];
const result = spawnSync('tar', args, { stdio: 'inherit' });
if (result.status !== 0) throw new Error('Failed to package corresponding source.');
console.log('Packaged source, tests, lockfiles and build/deployment scripts.');
