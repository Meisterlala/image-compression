import { inject, ref, toRaw, watch, type App, type InjectionKey, type Ref, type WatchStopHandle } from 'vue';

const DATABASE = 'image-compression-state';
const STORE = 'state';

// IndexedDB structured clones preserve files, blobs and typed arrays without base64.
function snapshot<T>(value: T): T {
  const raw = toRaw(value);
  if (Array.isArray(raw)) return raw.map(snapshot) as T;
  if (raw && Object.getPrototypeOf(raw) === Object.prototype) {
    return Object.fromEntries(Object.entries(raw).map(([key, item]) => [key, snapshot(item)])) as T;
  }
  return raw;
}

export interface Persistence {
  error: Ref<string>;
  get<T>(key: string): T | undefined;
  save(key: string, value: unknown): void;
  install(app: App): void;
}
const persistenceKey: InjectionKey<Persistence> = Symbol('browser persistence');

export async function createPersistence(): Promise<Persistence> {
  const error = ref('');
  const values = new Map<string, unknown>();
  let database: IDBDatabase | undefined;
  const fail = () => {
    error.value = 'Browser storage is unavailable or full. Recent changes may not survive a reload. Free browser storage or allow site storage to save your work.';
  };
  try {
    database = await new Promise<IDBDatabase>((resolve, reject) => {
      const request = indexedDB.open(DATABASE, 1);
      request.onupgradeneeded = () => request.result.createObjectStore(STORE);
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
      request.onblocked = () => reject(new Error('Storage upgrade blocked'));
    });
    database.onversionchange = () => { database?.close(); fail(); };
    await new Promise<void>((resolve, reject) => {
      const transaction = database!.transaction(STORE, 'readonly');
      const request = transaction.objectStore(STORE).openCursor();
      request.onsuccess = () => {
        const cursor = request.result;
        if (cursor) { values.set(String(cursor.key), cursor.value); cursor.continue(); }
      };
      transaction.oncomplete = () => resolve();
      transaction.onabort = () => reject(transaction.error);
      transaction.onerror = () => reject(transaction.error);
    });
  } catch { fail(); }

  const persistence: Persistence = {
    error,
    get: <T>(key: string) => values.get(key) as T | undefined,
    save(key, value) {
      let copy: unknown;
      try { copy = snapshot(value); } catch { fail(); return; }
      values.set(key, copy);
      if (!database) return;
      // Start immediately: debouncing can lose the last edit on reload.
      try {
        const transaction = database.transaction(STORE, 'readwrite');
        transaction.onabort = transaction.onerror = fail;
        transaction.objectStore(STORE).put(copy, key);
      } catch { fail(); }
    },
    install: app => { app.provide(persistenceKey, persistence); },
  };
  return persistence;
}

export function usePersistence(): Persistence {
  const persistence = inject(persistenceKey);
  if (!persistence) throw new Error('Install the browser persistence plugin first');
  return persistence;
}

/** Register durable Vue state. Keep transient jobs, DOM nodes and object URLs out of snapshots. */
export function usePersistentState<T>(key: string, options: {
  snapshot: () => T;
  restore: (saved: T) => void;
  deep?: boolean;
}): { restored: boolean; stop: WatchStopHandle } {
  const persistence = usePersistence();
  const saved = persistence.get<T>(key);
  let restored = false;
  if (saved !== undefined) {
    try { options.restore(saved); restored = true; }
    catch { persistence.error.value = 'Some saved state could not be restored. You can continue using the app.'; }
  }
  const stop = watch(options.snapshot, value => persistence.save(key, value), { deep: options.deep });
  return { restored, stop };
}

export function usePersistentRef<T>(key: string, initial: T): Ref<T> {
  const state = ref(initial) as Ref<T>;
  usePersistentState(key, { snapshot: () => state.value, restore: saved => { state.value = saved; }, deep: true });
  return state;
}
