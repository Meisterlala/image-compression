# Local state

`src/lib/persistence.ts` saves workspace and preferences in IndexedDB
(`image-compression-state`, schema 1). Reload restores the source/prepared image,
completed outputs, settings, metrics, tabs and viewer preferences. Jobs are not
resumed. A new upload clears results but keeps preferences; resizing keeps results.

Writes start immediately; storage failures show a warning. Clearing site data,
private-session expiry or browser eviction can delete work and cached models.
Nothing is uploaded; this is not a backup.

## Vue API

The plugin loads before mounting in `src/main.ts`. Use stable feature keys:

```ts
const selected = usePersistentRef('feature-selection', 'default');
usePersistentState('feature-state', {
  snapshot: () => ({ /* durable fields */ }),
  restore: saved => { /* apply fields */ },
  deep: true, // nested settings only, not large pixel buffers
});
```

Restoration is synchronous during setup; scoped Vue watches batch later writes.
Snapshots unwrap proxies and preserve File, Blob and typed arrays. Store related
state together for atomic updates (source and outputs share a workspace record),
but keep binaries apart from frequently changed controls. Persist bytes, not
workers, queues, progress, DOM nodes or object URLs. Store ImageData as dimensions
and pixels. Use a new key for incompatible record changes.

Test: `npm run build:web && npx playwright test tests/persistence.spec.ts`.
