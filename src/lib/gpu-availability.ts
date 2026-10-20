import { readonly, ref } from 'vue';

export type GpuAvailability = 'checking' | 'available' | 'unavailable';
const availability = ref<GpuAvailability>('checking');
let pending: Promise<void> | undefined;

/** An exposed API alone does not guarantee an accessible GPU. */
export async function canUseWebGpu(gpu: GPU | undefined): Promise<boolean> {
  try {
    const adapter = await gpu?.requestAdapter();
    if (!adapter) return false;
    const device = await adapter.requestDevice();
    device.destroy();
    return true;
  } catch {
    return false;
  }
}

export function useGpuAvailability() {
  pending ??= canUseWebGpu(globalThis.navigator?.gpu).then(available => {
    availability.value = available ? 'available' : 'unavailable';
  });
  return readonly(availability);
}
