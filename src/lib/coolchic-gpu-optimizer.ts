// GPU-resident reference SOAP, including its first-update skip and latent
// Adam-like groups. Only the small eigendecomposition/QR problems visit WASM.
export interface SoapAxis { dimension: number; stride: number; gram: number; basis: number; order: number }
export interface SoapTensor {
  offset: number; length: number; first: number; second: number; latent: boolean; axes: SoapAxis[];
}
export interface SoapLayout { size: number; parameters: number; tensors: SoapTensor[] }
export type SoapBasis = (gram: Float32Array, previous?: Float32Array) => Float32Array;

type Encode = (encoder: GPUCommandEncoder) => void;

export class GpuSoap {
  readonly state: GPUBuffer;
  private scratch: GPUBuffer[];
  private clip: GPUBuffer;
  private scalars: GPUBuffer;
  private staging: GPUBuffer;
  private pipelines = new Map<string, GPUComputePipeline>();
  private groups = new Map<GPUComputePipeline, { buffers: GPUBuffer[]; group: GPUBindGroup }[]>();
  private bases = new Map<SoapAxis, Float32Array>();
  private recording?: Encode[];
  private recorded?: { gradient: GPUBuffer; commands: Encode[] };
  constructor(private device: GPUDevice, readonly layout: SoapLayout, private basis: SoapBasis) {
    const allocate = (size: number, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST) => device.createBuffer({ size: Math.max(4, size * 4), usage });
    this.state = allocate(layout.size);
    const preconditioned = layout.tensors.filter(t => t.axes.length);
    const largest = Math.max(1, ...preconditioned.map(t => t.length));
    const moments = Math.max(1, ...preconditioned.map(t => t.offset + t.length));
    this.scratch = [allocate(largest), allocate(largest), allocate(largest), allocate(moments)];
    this.clip = allocate(1);
    this.scalars = allocate(3);
    this.staging = allocate(layout.size, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
  }
  releasePlanBindings() {
    this.recorded = undefined; this.groups.clear();
  }
  get memoryBytes() {
    return this.state.size + this.clip.size + this.scalars.size + this.staging.size + this.scratch.reduce((n,buffer) => n+buffer.size,0);
  }
  upload(values: Float32Array) {
    if (values.length !== this.layout.size) throw new Error('Invalid SOAP state length');
    this.device.queue.writeBuffer(this.state, 0, new Float32Array(values));
    for (const tensor of this.layout.tensors) for (const axis of tensor.axes) {
      this.bases.set(axis, values.slice(axis.basis, axis.basis + axis.dimension ** 2));
    }
  }
  private dispatch(encoder: GPUCommandEncoder, size: number, code: string, buffers: GPUBuffer[], write: number[]) {
    const active = buffers.map((buffer,i) => ({buffer,i})).filter(({i}) => new RegExp(`\\bb${i}\\b`).test(code));
    const declarations = active.map(({i}) => `@group(0) @binding(${i}) var<storage, ${write.includes(i) ? 'read_write' : 'read'}> b${i}: array<f32>;`).join('\n');
    const source = `${declarations}\n@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid: vec3<u32>) { let i = gid.x; if (i >= ${size}u) { return; } ${code} }`;
    let pipeline = this.pipelines.get(source);
    if (!pipeline) {
      pipeline = this.device.createComputePipeline({ layout: 'auto', compute: { module: this.device.createShaderModule({ label: 'CoolChicSoap', code: source }), entryPoint: 'main' } });
      this.pipelines.set(source, pipeline);
    }
    let groups = this.groups.get(pipeline);
    if (!groups) { groups = []; this.groups.set(pipeline, groups); }
    let cached = groups.find(g => g.buffers.every((buffer,i) => buffer === active[i].buffer));
    if (!cached) {
      cached = { buffers: active.map(({buffer}) => buffer), group: this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: active.map(({buffer,i}) => ({ binding: i, resource: { buffer } })) }) };
      groups.push(cached);
    }
    const selected = pipeline, group = cached.group;
    const command = (encoder: GPUCommandEncoder) => {
      const pass = encoder.beginComputePass(); pass.setPipeline(selected); pass.setBindGroup(0,group);
      pass.dispatchWorkgroups(Math.ceil(size / 64)); pass.end();
    };
    this.recording?.push(command); command(encoder);
  }
  private copy(encoder: GPUCommandEncoder, source: GPUBuffer, sourceOffset: number, destination: GPUBuffer, destinationOffset: number, size: number) {
    const command = (encoder: GPUCommandEncoder) => encoder.copyBufferToBuffer(source,sourceOffset,destination,destinationOffset,size);
    this.recording?.push(command); command(encoder);
  }
  private project(encoder: GPUCommandEncoder, tensor: SoapTensor, input: GPUBuffer, offset: number, back: boolean) {
    let current = this.scratch[0], next = this.scratch[1];
    this.copy(encoder,input,offset*4,current,0,tensor.length*4);
    // Every size-one axis has the same scalar Gram and ±1 basis. Pairs
    // cancel exactly; avoid eight redundant projection passes per 1x1 weight.
    const unitAxes = tensor.axes.filter(axis => axis.dimension === 1);
    const axes = tensor.axes.filter(axis => axis.dimension !== 1);
    if (unitAxes.length % 2) axes.push(unitAxes[0]);
    for (const axis of axes) {
      const { dimension: d, stride: s, basis: q } = axis;
      this.dispatch(encoder, tensor.length, `
        let block = i / ${d * s}u; let destination = (i / ${s}u) % ${d}u; let suffix = i % ${s}u;
        var value = 0.0;
        for (var source = 0u; source < ${d}u; source++) {
          let weight = b0[${q}u + ${back ? `destination * ${d}u + source` : `source * ${d}u + destination`}];
          value += b1[block * ${d * s}u + source * ${s}u + suffix] * weight;
        }
        b2[i] = value;`, [this.state, current, next], [2]);
      [current, next] = [next, current];
    }
    return current;
  }
  /** Gradient is already packed in the same CHW/OIHW order as Rust. */
  async encodeUpdate(encoder: GPUCommandEncoder, gradient: GPUBuffer, step: number, configurations: GPUBuffer, configurationOffset: number) {
    const f = (value: number) => `${Math.fround(value).toExponential(12)}`;
    // Copy rates in command order. Host writeBuffer calls would overwrite the
    // rates for earlier iterations queued in this same command encoder.
    encoder.copyBufferToBuffer(configurations, configurationOffset * 4, this.scalars, 0, 12);
    const initial = step < 0, refresh = !initial && (step + 1) % 10 === 0;
    if (!initial && !refresh) {
      if (this.recorded?.gradient === gradient) {
        for (const command of this.recorded.commands) command(encoder);
        return encoder;
      }
      this.recording = [];
    }
    // The network has few parameters relative to its latents. Reproduce the
    // reference's per-tensor norm order, without reading a norm back to JS.
    const norm = this.layout.tensors.filter(t => !t.latent).map(t => `{
      var squared = 0.0; for (var j = 0u; j < ${t.length}u; j++) { let g = b0[${t.offset}u+j]; squared += g*g; }
      let n = sqrt(squared); total += n*n;
    }`).join('\n');
    this.dispatch(encoder, 1, `var total = 0.0; ${norm} b1[0] = min(1.0, 0.1 / (sqrt(total) + 0.000001));`, [gradient, this.clip], [1]);
    for (const tensor of this.layout.tensors) {
      const { offset: p, length: n, first: m, second: v, latent } = tensor;
      if (!initial && !tensor.axes.length) {
        // Latents dominate the parameter count, but have no SOAP projections.
        // Fuse their Adam-like moments and parameter update in one pass.
        const beta1 = latent ? 0.9 : 0.95, beta2 = latent ? 0.999 : 0.95;
        this.dispatch(encoder, n, `
          let g = b1[${p}u+i] * ${latent ? '1.0' : 'b3[0]'};
          let first = ${f(beta1)} * b0[${m}u+i] + ${f(1-beta1)} * g;
          let second = ${f(beta2)} * b0[${v}u+i] + ${f(1-beta2)} * g * g;
          b0[${m}u+i] = first; b0[${v}u+i] = second;
          var value = b0[${p}u+i] - b2[${latent ? 1 : 0}] * (first / (sqrt(second) + 1.0e-8));
          value -= ${latent ? '0.0' : 'b2[2]'} * value; b0[${p}u+i] = value;
        `, [this.state, gradient, this.scalars, this.clip], [0]);
        continue;
      }
      if (!initial) {
        this.dispatch(encoder, n, `b2[i] = b0[${p}u+i] * ${latent ? '1.0' : 'b1[0]'};`, [gradient, this.clip, this.scratch[2]], [2]);
        // project() uses the first two scratch buffers; retain the clipped
        // gradient in the third only until its moment update is encoded.
        const projected = this.project(encoder, tensor, this.scratch[2], 0, false);
        const normalized = projected === this.scratch[0] ? this.scratch[1] : this.scratch[0];
        const beta1 = latent ? 0.9 : 0.95, beta2 = latent ? 0.999 : 0.95;
        this.dispatch(encoder, n, `
          let g = b1[i];
          let first = ${f(beta1)} * b0[${m}u+i] + ${f(1-beta1)} * g;
          let second = ${f(beta2)} * b0[${v}u+i] + ${f(1-beta2)} * g * g;
          b0[${m}u+i] = first; b0[${v}u+i] = second;
          b2[i] = first / (sqrt(second) + 1.0e-8);`, [this.state, projected, normalized], [0, 2]);
        // A copy into scratch[2] avoids aliasing the ping-pong projection.
        this.copy(encoder,normalized,0,this.scratch[2],0,n*4);
        const update = this.project(encoder, tensor, this.scratch[2], 0, true);
        this.dispatch(encoder, n, `var value = b0[${p}u+i] - b2[${latent ? 1 : 0}] * b1[i]; value -= ${latent ? '0.0' : 'b2[2]'} * value; b0[${p}u+i] = value;`, [this.state, update, this.scalars], [0]);
        if (tensor.axes.length) {
          const moment = this.project(encoder, tensor, this.state, m, true);
          this.copy(encoder,moment,0,this.scratch[3],p*4,n*4);
        }
      }
      for (const axis of tensor.axes) {
        const { dimension: d, stride: s, gram } = axis;
        const blocks = n / (d * s);
        this.dispatch(encoder, d * d, `
          let row = i / ${d}u; let col = i % ${d}u; var dot = 0.0;
          for (var block = 0u; block < ${blocks}u; block++) { for (var suffix = 0u; suffix < ${s}u; suffix++) {
            let a = b1[${p}u + block * ${d*s}u + row * ${s}u + suffix] * b2[0];
            let b = b1[${p}u + block * ${d*s}u + col * ${s}u + suffix] * b2[0]; dot += a*b;
          } }
          let old = b0[${gram}u+i]; b0[${gram}u+i] = old + ${f(0.05)} * (dot - old);`, [this.state, gradient, this.clip], [0]);
      }
      // Reprojection without a basis refresh can be queued immediately. The
      // refresh case must first read the small Gram matrices for Rust's QR.
      if (!initial && !refresh && tensor.axes.length) {
        const moment = this.project(encoder, tensor, this.scratch[3], p, false);
        this.copy(encoder,moment,0,this.state,m*4,n*4);
      }
    }
    if (initial || refresh) {
      // Read only Gram matrices, not all parameters, gradients or latents.
      const axes = this.layout.tensors.flatMap(t => t.axes);
      const count = axes.reduce((n, a) => n + a.dimension ** 2, 0);
      if (count) {
        let offset = 0;
        for (const axis of axes) { encoder.copyBufferToBuffer(this.state, axis.gram * 4, this.staging, offset * 4, axis.dimension ** 2 * 4); offset += axis.dimension ** 2; }
        this.device.queue.submit([encoder.finish()]);
        await this.staging.mapAsync(GPUMapMode.READ, 0, count * 4);
        const values = new Float32Array(this.staging.getMappedRange(0, count * 4).slice(0)); this.staging.unmap();
        offset = 0;
        for (const axis of axes) {
          const length = axis.dimension ** 2;
          const result = this.basis(values.subarray(offset, offset + length), initial ? undefined : this.bases.get(axis));
          const basis = result.slice(0, length); this.bases.set(axis, basis);
          this.device.queue.writeBuffer(this.state, axis.basis * 4, basis);
          if (refresh) {
            this.device.queue.writeBuffer(this.state, axis.order * 4, result.slice(length));
          }
          offset += length;
        }
        encoder = this.device.createCommandEncoder();
        if (refresh) for (const tensor of this.layout.tensors) {
          for (const axis of tensor.axes.filter(axis => axis.dimension !== 1)) {
            this.dispatch(encoder, tensor.length, `
              let block = i / ${axis.dimension*axis.stride}u; let destination = (i / ${axis.stride}u) % ${axis.dimension}u; let suffix = i % ${axis.stride}u;
              let source = u32(b0[${axis.order}u + destination]);
              b1[i] = b0[${tensor.second}u + block * ${axis.dimension*axis.stride}u + source * ${axis.stride}u + suffix];`, [this.state, this.scratch[0]], [1]);
            encoder.copyBufferToBuffer(this.scratch[0], 0, this.state, tensor.second * 4, tensor.length * 4);
          }
          if (tensor.axes.length) {
            const moment = this.project(encoder, tensor, this.scratch[3], tensor.offset, false);
            encoder.copyBufferToBuffer(moment, 0, this.state, tensor.first * 4, tensor.length * 4);
          }
        }
      }
    }
    if (this.recording) { this.recorded = {gradient,commands:this.recording}; this.recording = undefined; }
    return encoder;
  }
  async download() {
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(this.state, 0, this.staging, 0, this.layout.size * 4);
    this.device.queue.submit([encoder.finish()]);
    await this.staging.mapAsync(GPUMapMode.READ);
    try { return new Float32Array(this.staging.getMappedRange().slice(0)); }
    finally { this.staging.unmap(); }
  }
  dispose() {
    this.state.destroy(); this.clip.destroy(); this.scalars.destroy(); this.staging.destroy(); this.scratch.forEach(b => b.destroy()); this.pipelines.clear(); this.groups.clear(); this.recorded = undefined;
  }
}
