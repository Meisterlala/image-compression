// Build-time only. No GitLab requests or credentials are used by the website.
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { existsSync } from 'node:fs';
import { mkdir, readFile, writeFile, rename } from 'node:fs/promises';
const revision = 'b9e573fef1dd90408270da23aae678c823f15525';
const gitUrl = 'https://gitlab.com/wg1/jpeg-ai/jpeg-ai-reference-software.git';
const cache = '.cache/jpeg-ai-reference';
const output = 'public/models';
const run = (cmd, args, capture = false) => {
  const r = spawnSync(cmd, args, { encoding: 'utf8', stdio: capture ? 'pipe' : 'inherit', env: { ...process.env, GIT_LFS_SKIP_SMUDGE: '1' } });
  if (r.status !== 0) throw new Error(`${cmd} failed: ${r.stderr ?? ''}`);
  return r.stdout;
};
const hash = (b) => createHash('sha256').update(b).digest('hex');
await mkdir('.cache', { recursive: true });
await mkdir(output, { recursive: true });
if (!existsSync(`${cache}/.git`)) run('git', ['clone', '--no-checkout', '--depth', '1', gitUrl, cache]);
if (run('git', ['-C', cache, 'rev-parse', 'HEAD'], true).trim() !== revision) {
  run('git', ['-C', cache, 'fetch', '--depth', '1', 'origin', revision]);
}
run('cargo', ['build', '--release', '--manifest-path', 'crates/codec/Cargo.toml', '--features', 'model-packer', '--bin', 'pack-models']);
const betas = ['0.002', '0.012', '0.075', '0.5'];
const models = [];
for (let id = 0; id < 4; id++) {
  const beta = betas[id];
  const paths = ['Y', 'UV'].flatMap(c => [`VM_common_int/${c}_${beta}.pth`, `VM_bop/encoder_${c}_${beta}.pth`, `VM_bop/decoder_${c}_${beta}.pth`]);
  await Promise.all(paths.map(async path => {
    const pointer = run('git', ['-C', cache, 'show', `${revision}:models/${path}`], true);
    const sha = /oid sha256:([a-f0-9]{64})/.exec(pointer)?.[1];
    const size = Number(/size (\d+)/.exec(pointer)?.[1]);
    if (!sha || !size) throw new Error(`Not a pinned LFS pointer: ${path}`);
    const dest = `${cache}/models/${path}`;
    if (existsSync(dest)) { const b = await readFile(dest); if (b.length === size && hash(b) === sha) return; }
    console.log(`Downloading verified weights: ${path} (${(size/1048576).toFixed(1)} MiB)`);
    const url = `https://gitlab.com/wg1/jpeg-ai/jpeg-ai-reference-software/-/raw/${revision}/models/${path}`;
    let bytes;
    for (let attempt = 0; attempt < 3; attempt++) {
      const response = await fetch(url, { signal: AbortSignal.timeout(180_000) });
      if (response.ok) { bytes = Buffer.from(await response.arrayBuffer()); if (bytes.length === size && hash(bytes) === sha) break; }
      bytes = undefined;
    }
    if (!bytes) throw new Error(`Download or SHA-256 verification failed: ${path}`);
    await mkdir(dest.slice(0, dest.lastIndexOf('/')), { recursive: true });
    await writeFile(`${dest}.tmp`, bytes); await rename(`${dest}.tmp`, dest);
  }));
  const file = `bop-m${id}.zjb`;
  run('crates/codec/target/release/pack-models', [`${cache}/models`, String(id), 'public/legal/jpeg-ai-reference-LICENSE.txt', `${output}/${file}`]);
  const bytes = await readFile(`${output}/${file}`);
  models.push({ id, file, bytes: bytes.length, sha256: hash(bytes) });
}
await writeFile(`${output}/manifest.json`, JSON.stringify({ revision, operatingPoint: 'bop', models }, null, 2) + '\n');
console.log('All encoder + decoder bundles ready. Serve public/models with the site.');
