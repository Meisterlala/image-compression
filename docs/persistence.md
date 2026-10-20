# Browser-only persistence

The workspace is saved automatically in IndexedDB (`image-compression-state`, schema version 1). Nothing is uploaded. Reloading restores the original file, prepared image, completed codec outputs (encoded bytes, decoded pixels, settings and metrics), current codec controls, selected tab, resize setting and comparison viewer preferences. Downloads continue to work without encoding again.

The UI shows when writes are pending and warns when storage is unavailable or full. Wait for “Saved locally” before closing the page. Browser site-data clearing, private-session expiry, or automatic storage eviction can remove saved work; this is not a backup. In-progress jobs are deliberately not restored. Importing another image clears previous results while retaining preferences. Resizing retains previous outputs as before.

## Vue integration

`src/lib/persistence.ts` provides a Vue plugin, initialized in `src/main.ts` before mounting, and two composables:

```ts
const selected = usePersistentRef('feature-selection', 'default');

usePersistentState('feature-state', {
  snapshot: () => ({ /* durable fields */ }),
  restore: saved => { /* apply saved fields, merge defaults as needed */ },
  deep: true, // for nested reactive settings, not large pixel buffers
});
```

Use stable, feature-specific keys. State is restored synchronously during setup, then watched using Vue's batched updates. Snapshots unwrap Vue proxies and IndexedDB preserves File, Blob and typed-array values. Store related state together when it must be updated atomically; the source image and all results are one workspace record. Keep large binaries separate from frequently changed settings to avoid rewriting them with each slider movement.

Do not persist transient workers, job queues, progress, DOM nodes, or object URLs. Store underlying bytes instead. ImageData is represented explicitly by width, height and pixels because browser structured-clone support for ImageData varies. For incompatible future record changes, migrate the record or use a new key. Reactive watches are disposed with their Vue component scope.

Run `npx playwright test tests/persistence.spec.ts` after `npm run build:web` for reload, download integrity, image replacement and denied-storage coverage.
