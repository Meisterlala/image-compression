/// <reference types="@webgpu/types" />
import shader from './coolchic-kernels.wgsl?raw';
import { convolutionShape } from './coolchic-shape';

/** Reusable WebGPU kernels. Training remains in Rust/WASM; only floating-point
 * convolutions and their analytical derivatives cross this boundary. */
export class CoolChicGpu {
  private buffers = new Map<string, GPUBuffer>();
  private pipelines: GPUComputePipeline[] = [];
  private layout!: GPUBindGroupLayout;
  private lost?: string;
  private constructor(private device: GPUDevice) {
    void device.lost.then(info => { this.lost = info.message || 'GPU device lost'; });
  }
  static async create() {
    if (!navigator.gpu) throw new Error('WebGPU is not available');
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('No WebGPU adapter');
    const device = await adapter.requestDevice();
    const gpu = new CoolChicGpu(device);
    try {
      const module = device.createShaderModule({ code: shader });
      const info = await module.getCompilationInfo();
      const errors = info.messages.filter(m => m.type === 'error');
      if (errors.length) throw new Error(errors.map(e => e.message).join('; '));
      gpu.layout = device.createBindGroupLayout({ entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        ...[1, 2, 3, 4].map(binding => ({ binding, visibility: GPUShaderStage.COMPUTE,
          buffer: { type: binding === 1 ? 'read-only-storage' as const : 'storage' as const } })),
      ] });
      const layout = device.createPipelineLayout({ bindGroupLayouts: [gpu.layout] });
      gpu.pipelines = await Promise.all(['forward', 'inputGradient', 'parameterPartials', 'parameterReduce']
        .map(entryPoint => device.createComputePipelineAsync({ layout, compute: { module, entryPoint } })));
      return gpu;
    } catch (error) { gpu.dispose(); throw error; }
  }
  private buffer(name: string, bytes: number, usage: GPUBufferUsageFlags) {
    if (!Number.isSafeInteger(bytes) || bytes < 0 || bytes > 256 * 1024 ** 2) throw new RangeError('Invalid GPU buffer size');
    const size = Math.max(4, Math.ceil(bytes / 256) * 256);
    let buffer = this.buffers.get(name);
    if (!buffer || buffer.size < size) {
      const liveBytes = [...this.buffers.values()].reduce((sum, value) => sum + value.size, 0);
      if (liveBytes - (buffer?.size ?? 0) + size > 256 * 1024 ** 2) throw new Error('GPU buffer cache exceeds 256 MiB');
      buffer?.destroy();
      buffer = this.device.createBuffer({ size, usage });
      this.buffers.set(name, buffer);
    }
    return buffer;
  }
  async convolution(params: Uint32Array, data: Float32Array, destination?: Float32Array): Promise<Float32Array> {
    if (this.lost) throw new Error(this.lost);
    const { ci, co, h, w, k, residual, relu, backward, chunks, inputCount, outputCount,
      inputLength, resultCount, parameters, partialCount, sizes, resultBytes, workingBytes } = convolutionShape(params);
    if (!(data instanceof Float32Array) || data.length !== inputLength) throw new Error('Invalid GPU payload');
    if (destination && (!(destination instanceof Float32Array) || destination.length !== resultCount)) throw new Error('GPU output shape mismatch');
    // Bound auxiliary buffers as well as the host transfer; never exceed the
    // conservative browser memory budget just to try an accelerator.
    if (sizes.some(n => n > this.device.limits.maxStorageBufferBindingSize) || workingBytes > 256 * 1024 ** 2) {
      throw new Error('GPU convolution exceeds buffer budget');
    }
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const uniform = this.buffer('uniform', 48, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    const buffers = sizes.map((size, i) => this.buffer(`storage${i}`, size, storage));
    const staging = this.buffer('readback', resultBytes, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
    this.device.queue.writeBuffer(uniform, 0, new Uint32Array([ci, co, h, w, k, residual, relu, backward, chunks, 0, 0, 0]));
    // WebGPU intentionally disallows SharedArrayBuffer-backed BufferSource.
    this.device.queue.writeBuffer(buffers[0], 0, new Float32Array(data));
    const group = this.device.createBindGroup({ layout: this.layout, entries: [uniform, ...buffers]
      .map((buffer, binding) => ({ binding, resource: { buffer } })) });
    this.device.pushErrorScope('validation');
    try {
      const encoder = this.device.createCommandEncoder();
      const dispatch = (pipeline: number, count: number) => {
        const groups = Math.ceil(count / 64);
        const pass = encoder.beginComputePass();
        pass.setPipeline(this.pipelines[pipeline]);
        pass.setBindGroup(0, group);
        pass.dispatchWorkgroups(Math.min(groups, 256), Math.ceil(groups / 256));
        pass.end();
      };
      dispatch(0, outputCount);
      if (backward) {
        dispatch(1, inputCount);
        dispatch(2, partialCount);
        dispatch(3, parameters);
      }
      encoder.copyBufferToBuffer(buffers[backward ? 2 : 1], 0, staging, 0, resultBytes);
      this.device.queue.submit([encoder.finish()]);
      await staging.mapAsync(GPUMapMode.READ, 0, resultBytes);
      try {
        const result = destination ?? new Float32Array(resultCount);
        result.set(new Float32Array(staging.getMappedRange(0, resultBytes)));
        if (!result.every(Number.isFinite)) throw new Error('Non-finite GPU result');
        if (this.lost) throw new Error(this.lost);
        return result;
      } finally { staging.unmap(); }
    } finally {
      const error = await this.device.popErrorScope();
      if (error) throw new Error(error.message);
    }
  }
  dispose() {
    for (const buffer of this.buffers.values()) buffer.destroy();
    this.buffers.clear();
    this.device.destroy();
  }
}
