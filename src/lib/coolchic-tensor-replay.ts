import type { WebGPUBackend } from '@tensorflow/tfjs-backend-webgpu';

type Encode = (encoder: GPUCommandEncoder) => void;
interface Batch { commands: Encode[] }
export interface GraphInputBuffer { kind: 'parameters' | 'latents' | 'noise'; offset: number; length: number; buffer: GPUBuffer }

/** Record the already differentiated GPU program once. Re-encode its fixed
 * pipelines/bind groups directly thereafter; no TensorFlow tensor allocation,
 * autodiff traversal or bind-group rebuilding on the iteration hot path. */
export class TensorReplay {
  private pins = new Map<GPUBuffer, number>();
  private deferred = new Map<GPUBuffer, boolean | undefined>();
  private collecting?: Set<GPUBuffer>;
  private acquire: WebGPUBackend['bufferManager']['acquireBuffer'];
  private release: WebGPUBackend['bufferManager']['releaseBuffer'];
  // Bind groups retain these buffers: do not recycle intermediates/uniforms
  // until their recorded plan is released.
  constructor(private backend: WebGPUBackend) {
    const manager = backend.bufferManager;
    this.acquire = manager.acquireBuffer; this.release = manager.releaseBuffer;
    manager.acquireBuffer = (...args) => { const buffer = this.acquire.apply(manager, args); this.pin(buffer); return buffer; };
    manager.releaseBuffer = (buffer, reuse) => {
      if (this.pins.has(buffer)) this.deferred.set(buffer, reuse);
      else this.release.call(manager, buffer, reuse);
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
        this.pins.delete(buffer);
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
  capture<T>(build: () => T): { value: T; batches: Batch[]; release: () => void } {
    if (this.collecting) throw new Error('Nested GPU capture');
    this.flush();
    const device = this.backend.device, queue = device.queue;
    const originalEncoder = device.createCommandEncoder, originalBindGroup = device.createBindGroup, originalSubmit = queue.submit;
    const buffers = new Set<GPUBuffer>(), recorded = new Map<GPUCommandBuffer, Encode[]>(), batches: Batch[] = [];
    this.collecting = buffers;
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
      this.collecting = undefined;
    }
  }
  dispose() {
    if (this.pins.size) throw new Error('Dispose GPU plans before releasing their recorder');
    this.backend.bufferManager.acquireBuffer = this.acquire;
    this.backend.bufferManager.releaseBuffer = this.release;
  }
}

export class CapturedTensorGradient {
  private staging: GPUBuffer;
  constructor(private device: GPUDevice, private batches: Batch[], private output: GPUBuffer,
    private count: number, private inputs: GraphInputBuffer[], private release: () => void) {
    if (!batches.length) throw new Error('Empty GPU execution graph');
    this.staging = device.createBuffer({ size: count * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  }
  async run(parameters: Float32Array, latents: Float32Array, noise: Float32Array): Promise<Float32Array> {
    const sources = { parameters, latents, noise };
    for (const input of this.inputs) {
      let values = sources[input.kind].subarray(input.offset, input.offset + input.length);
      if (!(values.buffer instanceof ArrayBuffer)) values = new Float32Array(values);
      this.device.queue.writeBuffer(input.buffer, 0, values as Float32Array<ArrayBuffer>);
    }
    for (let index = 0; index < this.batches.length; index++) {
      const encoder = this.device.createCommandEncoder();
      for (const command of this.batches[index].commands) command(encoder);
      if (index === this.batches.length - 1) encoder.copyBufferToBuffer(this.output, 0, this.staging, 0, this.count * 4);
      this.device.queue.submit([encoder.finish()]);
    }
    await this.staging.mapAsync(GPUMapMode.READ);
    try { return new Float32Array(this.staging.getMappedRange(0, this.count * 4).slice(0)); }
    finally { this.staging.unmap(); }
  }
  dispose() { this.staging.destroy(); this.release(); }
}
