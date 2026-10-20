import type { WebGPUBackend } from '@tensorflow/tfjs-backend-webgpu';

type Encode = (encoder: GPUCommandEncoder) => void;
interface Batch { commands: Encode[] }
export interface GraphInputBuffer { kind: 'parameters' | 'latents' | 'noise'; offset: number; length: number; buffer: GPUBuffer }

/** CPU tensor uploads do not need persistent mapped/staging buffers. Keeping
 * those buffers in a replay plan can exhaust a browser's host-visible heap even
 * when the card still has plenty of device-local VRAM. */
export function useQueueTensorUploads(backend: WebGPUBackend) {
  const original = backend.uploadToGPU.bind(backend);
  backend.uploadToGPU = dataId => {
    const tensor = backend.tensorMap.get(dataId);
    if (tensor.resource || !tensor.values || !['float32','int32','bool'].includes(tensor.dtype)) return original(dataId);
    const values = tensor.dtype === 'float32'
      ? new Float32Array(tensor.values as ArrayLike<number>)
      : new Int32Array(tensor.values as ArrayLike<number>);
    const resource = backend.bufferManager.acquireBuffer(values.byteLength,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST,false,false);
    backend.device.queue.writeBuffer(resource,0,values);
    Object.assign(tensor,{resource,values:null});
  };
}

/** Record the already differentiated GPU program once. Re-encode its fixed
 * pipelines/bind groups directly thereafter; no TensorFlow tensor allocation,
 * autodiff traversal or bind-group rebuilding on the iteration hot path. */
export class TensorReplay {
  private pins = new Map<GPUBuffer, number>();
  private deferred = new Map<GPUBuffer, boolean | undefined>();
  private collecting?: Set<GPUBuffer>;
  private available?: Map<string,GPUBuffer[]>;
  private availableSet?: Set<GPUBuffer>;
  private immutable = new WeakSet<GPUBuffer>();
  private acquire: WebGPUBackend['bufferManager']['acquireBuffer'];
  private release: WebGPUBackend['bufferManager']['releaseBuffer'];
  // Bind groups retain these buffers: do not recycle intermediates/uniforms
  // until their recorded plan is released.
  constructor(private backend: WebGPUBackend) {
    const manager = backend.bufferManager;
    this.acquire = manager.acquireBuffer; this.release = manager.releaseBuffer;
    manager.acquireBuffer = (...args) => {
      const [size,usage,mapped,reuse] = args;
      const immutable = mapped || reuse === false || !!(usage & GPUBufferUsage.UNIFORM);
      const free = !immutable ? this.available?.get(`${size}:${usage}`) : undefined;
      if (free?.length) {
        const buffer = free.pop()!;
        this.availableSet!.delete(buffer); this.deferred.delete(buffer);
        return buffer;
      }
      const buffer = this.acquire.apply(manager,args);
      if (immutable) this.immutable.add(buffer); else this.immutable.delete(buffer);
      this.pin(buffer); return buffer;
    };
    manager.releaseBuffer = (buffer,reuse) => {
      if (this.pins.has(buffer)) {
        this.deferred.set(buffer,reuse);
        // Recycle dead intermediates inside this command plan, but never let
        // another plan steal them. Uniforms/host-written constants must retain
        // their original contents, since host writes are not replayed.
        if (reuse !== false && this.collecting?.has(buffer) && this.pins.get(buffer) === 1 && !this.immutable.has(buffer) && !this.availableSet!.has(buffer)) {
          const key = `${buffer.size}:${buffer.usage}`;
          let free = this.available!.get(key);
          if (!free) {free=[];this.available!.set(key,free);}
          free.push(buffer); this.availableSet!.add(buffer);
        }
      } else this.release.call(manager,buffer,reuse);
    };
  }
  private pin(buffer: GPUBuffer) {
    const set = this.collecting;
    if (!set || set.has(buffer)) return;
    set.add(buffer); this.pins.set(buffer, (this.pins.get(buffer) ?? 0) + 1);
  }
  private unpin(buffers: Set<GPUBuffer>) {
    for (const buffer of buffers) {
      const count = this.pins.get(buffer)! - 1;
      if (count) this.pins.set(buffer, count);
      else {
        this.pins.delete(buffer); this.immutable.delete(buffer);
        if (this.deferred.has(buffer)) {
          this.release.call(this.backend.bufferManager, buffer, this.deferred.get(buffer));
          this.deferred.delete(buffer);
        }
      }
    }
  }
  private flush() {
    const backend = this.backend as unknown as { commandEncoder?: GPUCommandEncoder; endComputePassEncoder(): void; submitQueue(): void };
    if (backend.commandEncoder) { backend.endComputePassEncoder(); backend.submitQueue(); }
  }
  capture<T>(build: () => T, inputs: GPUBuffer[] = []): { value: T; batches: Batch[]; release: () => void } {
    if (this.collecting) throw new Error('Nested GPU capture');
    this.flush();
    const device = this.backend.device, queue = device.queue;
    const originalEncoder = device.createCommandEncoder, originalBindGroup = device.createBindGroup, originalSubmit = queue.submit;
    const buffers = new Set<GPUBuffer>(), recorded = new Map<GPUCommandBuffer, Encode[]>(), batches: Batch[] = [];
    this.collecting = buffers; this.available = new Map(); this.availableSet = new Set();
    // Even unused inputs are written during replay. Keep them out of the pool
    // while another cached plan reassigns its TensorFlow variables.
    for (const input of inputs) this.pin(input);
    device.createBindGroup = descriptor => {
      const entries = Array.from(descriptor.entries);
      for (const entry of entries) if ('buffer' in entry.resource) this.pin(entry.resource.buffer);
      return originalBindGroup.call(device, { ...descriptor, entries });
    };
    device.createCommandEncoder = descriptor => {
      const actual = originalEncoder.call(device, descriptor), commands: Encode[] = [];
      return new Proxy(actual, { get: (target, property) => {
        if (property === 'beginComputePass') return (description?: GPUComputePassDescriptor) => {
          const pass = target.beginComputePass(description), steps: ((pass: GPUComputePassEncoder) => void)[] = [];
          return new Proxy(pass, { get: (p, key) => {
            if (key === 'setPipeline') return (pipeline: GPUComputePipeline) => { steps.push(p => p.setPipeline(pipeline)); p.setPipeline(pipeline); };
            if (key === 'setBindGroup') return (index: number, group: GPUBindGroup, offsets?: number[]) => {
              const copy = offsets ? Array.from(offsets) : [];
              steps.push(p => p.setBindGroup(index, group, copy)); p.setBindGroup(index, group, copy);
            };
            if (key === 'dispatchWorkgroups') return (x: number, y?: number, z?: number) => { steps.push(p => p.dispatchWorkgroups(x, y, z)); p.dispatchWorkgroups(x, y, z); };
            if (key === 'end') return () => { p.end(); commands.push(encoder => { const replay = encoder.beginComputePass(description); for (const step of steps) step(replay); replay.end(); }); };
            const value = Reflect.get(p, key, p); return typeof value === 'function' ? value.bind(p) : value;
          } });
        };
        if (property === 'copyBufferToBuffer') return (source: GPUBuffer, so: number, destination: GPUBuffer, to: number, size: number) => {
          this.pin(source); this.pin(destination);
          commands.push(e => e.copyBufferToBuffer(source, so, destination, to, size));
          target.copyBufferToBuffer(source, so, destination, to, size);
        };
        if (property === 'finish') return (description?: GPUCommandBufferDescriptor) => {
          const buffer = target.finish(description); recorded.set(buffer, commands); return buffer;
        };
        const value = Reflect.get(target, property, target); return typeof value === 'function' ? value.bind(target) : value;
      } });
    };
    queue.submit = commandBuffers => {
      const array = Array.from(commandBuffers);
      for (const buffer of array) {
        const commands = recorded.get(buffer);
        if (!commands) throw new Error('GPU capture encountered an external command buffer');
        batches.push({ commands });
      }
      originalSubmit.call(queue, array);
    };
    try {
      const value = build(); this.flush();
      return { value, batches, release: () => this.unpin(buffers) };
    } catch (error) { this.unpin(buffers); throw error; }
    finally {
      device.createCommandEncoder = originalEncoder; device.createBindGroup = originalBindGroup; queue.submit = originalSubmit;
      this.collecting = undefined; this.available = undefined; this.availableSet = undefined;
    }
  }
  dispose() {
    if (this.pins.size) throw new Error('Dispose GPU plans before releasing their recorder');
    this.backend.bufferManager.acquireBuffer = this.acquire;
    this.backend.bufferManager.releaseBuffer = this.release;
  }
}

export class CapturedTensorGradient {
  private staging?: GPUBuffer;
  constructor(private device: GPUDevice, private batches: Batch[], readonly output: GPUBuffer,
    private count: number, private inputs: GraphInputBuffer[], private release: () => void) {
    if (!batches.length) throw new Error('Empty GPU execution graph');
  }
  async run(parameters: Float32Array, latents: Float32Array, noise: Float32Array): Promise<Float32Array> {
    this.staging ??= this.device.createBuffer({ size:this.count*4,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ });
    const sources = { parameters, latents, noise };
    for (const input of this.inputs) {
      let values = sources[input.kind].subarray(input.offset, input.offset + input.length);
      if (!(values.buffer instanceof ArrayBuffer)) values = new Float32Array(values);
      this.device.queue.writeBuffer(input.buffer, 0, values as Float32Array<ArrayBuffer>);
    }
    const encoder = this.device.createCommandEncoder();
    this.encode(encoder);
    encoder.copyBufferToBuffer(this.output, 0, this.staging, 0, this.count * 4);
    this.device.queue.submit([encoder.finish()]);
    await this.staging.mapAsync(GPUMapMode.READ);
    try { return new Float32Array(this.staging.getMappedRange(0, this.count * 4).slice(0)); }
    finally { this.staging.unmap(); }
  }
  private encode(encoder: GPUCommandEncoder) {
    for (const batch of this.batches) for (const command of batch.commands) command(encoder);
  }
  /** No mapAsync/readback: the optimizer consumes gradients directly. */
  encodeDevice(encoder: GPUCommandEncoder, state: GPUBuffer, parameterCount: number, noise: GPUBuffer, noiseOffset: number) {
    for (const input of this.inputs) {
      if (input.kind === 'noise') {
        encoder.copyBufferToBuffer(noise, (noiseOffset + input.offset) * 4, input.buffer, 0, input.length * 4);
      } else {
        const offset = input.offset + (input.kind === 'latents' ? parameterCount : 0);
        encoder.copyBufferToBuffer(state, offset * 4, input.buffer, 0, input.length * 4);
      }
    }
    this.encode(encoder);
    return this.output;
  }
  dispose() { this.staging?.destroy(); this.release(); }
}
