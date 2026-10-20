/// <reference types="@webgpu/types" />
import shader from './coolchic-kernels.wgsl?raw';

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
    const [ci, co, h, w, k, residual, relu, backward] = params;
    if (!ci || !co || !h || !w || !k || k % 2 !== 1 || (residual && ci !== co)) throw new Error('Invalid GPU convolution');
    const pixels = h * w, weights = ci * co * k * k;
    if (data.length !== ci * pixels + weights + co + (backward ? co * pixels : 0)) throw new Error('Invalid GPU payload');
    const chunks = Math.ceil(pixels / 256);
    const resultCount = backward ? ci * pixels + weights + co : co * pixels;
    if (destination && destination.length !== resultCount) throw new Error('GPU output shape mismatch');
    const sizes = [data.byteLength, co * pixels * 4, backward ? resultCount * 4 : 4, backward ? (weights + co) * chunks * 4 : 4];
    // Bound auxiliary buffers as well as the host transfer; never exceed the
    // conservative browser memory budget just to try an accelerator.
    if (sizes.some(n => n > this.device.limits.maxStorageBufferBindingSize) || sizes.reduce((a, b) => a + b, resultCount * 4) > 256 * 1024 ** 2) {
      throw new Error('GPU convolution exceeds buffer budget');
    }
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const uniform = this.buffer('uniform', 48, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    const buffers = sizes.map((size, i) => this.buffer(`storage${i}`, size, storage));
    const staging = this.buffer('readback', resultCount * 4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
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
      dispatch(0, co * pixels);
      if (backward) {
        dispatch(1, ci * pixels);
        dispatch(2, (weights + co) * chunks);
        dispatch(3, weights + co);
      }
      encoder.copyBufferToBuffer(buffers[backward ? 2 : 1], 0, staging, 0, resultCount * 4);
      this.device.queue.submit([encoder.finish()]);
      await staging.mapAsync(GPUMapMode.READ, 0, resultCount * 4);
      try {
        const result = destination ?? new Float32Array(resultCount);
        result.set(new Float32Array(staging.getMappedRange(0, resultCount * 4)));
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
