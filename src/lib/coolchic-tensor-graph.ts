import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-core/dist/public/chained_ops/register_all_chained_ops';
import '@tensorflow/tfjs-core/dist/register_all_gradients';
import '@tensorflow/tfjs-backend-webgpu';
import type { WebGPUBackend } from '@tensorflow/tfjs-backend-webgpu';
import { addN as gpuAddN } from '@tensorflow/tfjs-backend-webgpu/dist/kernels/AddN';
import { TensorReplay, CapturedTensorGradient, type GraphInputBuffer } from './coolchic-tensor-replay';
import { registerFusedCoolChicKernels, fusedQuantize, fusedRate, fusedContext, fusedFilter, fusedConvolution } from './coolchic-tensor-fused';

export interface GraphArchitecture {
  height: number; width: number; latent_min: number; latent_max: number;
  hyperlatent: [number, number] | null; ifce_outputs: number; spatial_context: number;
  arm_hidden: number; arm_stabiliser: boolean; synthesis_stabiliser: boolean;
  up_kernel: number; pre_kernel: number; common_randomness: boolean;
  interpolation: 'nearest' | 'bilinear' | 'bicubic';
  synthesis: { outputs: number; kernel: number; residual: boolean; relu: boolean }[];
}
export interface TensorGraphSpec {
  architecture: GraphArchitecture;
  latents: { height: number; width: number; hyper: boolean }[];
  ifce_inputs: number[];
  neighbours: [number, number][];
  group_lengths: number[];
}
interface Parameter { key: string; group: number; offset: number; length: number; trainable: boolean }
interface Conv { ci: number; co: number; k: number; residual: boolean; relu: boolean; weight: Parameter; bias: Parameter }
interface Arm { layers: Conv[]; stabiliser?: Conv }
type T4 = tf.Tensor4D;
export type GraphOptions = { mode: number; temperature: number; lambda: number; normalizer?: number };

let initialization: Promise<void> | undefined;
let gpuFailure: string | undefined;
export function initializeTensorGpu() {
  return initialization ??= initialize();
}
async function initialize() {
  // Never silently forward small tensors to CPU: that would introduce hidden
  // synchronization inside the forward/backward graph.
  tf.env().set('WEBGPU_CPU_FORWARD', false);
  tf.env().set('WEBGPU_DEFERRED_SUBMIT_BATCH_SIZE', 256);
  if (!await tf.setBackend('webgpu')) throw new Error('WebGPU tensor backend initialization failed');
  await tf.ready();
  registerFusedCoolChicKernels();
  const device = (tf.backend() as WebGPUBackend).device;
  device.addEventListener('uncapturederror', event => { gpuFailure = event.error.message; });
  void device.lost.then(info => { gpuFailure = info.message || 'WebGPU device lost'; });
  // TFJS 4.22's AddN binds every operand at once, exceeding WebGPU's default
  // 8-storage-buffer limit for shared ARM gradients / >7 latent rates. Reduce
  // in bounded groups instead; no CPU handoff and no larger adapter required.
  tf.unregisterKernel('AddN', 'webgpu');
  tf.registerKernel({ kernelName: 'AddN', backendName: 'webgpu', kernelFunc: ({ inputs, backend }) => {
    const gpu = backend as WebGPUBackend;
    let tensors = inputs as unknown as tf.TensorInfo[];
    const temporary: tf.TensorInfo[] = [];
    try {
      while (tensors.length > 4) {
        const next: tf.TensorInfo[] = [];
        for (let i = 0; i < tensors.length; i += 4) {
          const sum = gpuAddN({ inputs: tensors.slice(i, i + 4), backend: gpu });
          temporary.push(sum); next.push(sum);
        }
        tensors = next;
      }
      return gpuAddN({ inputs: tensors, backend: gpu });
    } finally { for (const tensor of temporary) gpu.disposeData(tensor.dataId); }
  } });
}

/** The full reference forward/backward graph is queued on the GPU. Only raw
 * parameter/latent gradients return to Rust, once per iteration. The existing
 * Rust SOAP/Adam, clipping, RNG, checkpoints and RD search are unchanged. */
export class CoolChicTensorGraph {
  private refs: Parameter[] = [];
  private vars = new Map<string, tf.Variable>();
  private constants = new Map<string, tf.Tensor>();
  private latentVars: tf.Variable[];
  private noiseVars: tf.Variable[];
  private recorder!: TensorReplay;
  private plan?: { key: string; graph: CapturedTensorGradient };
  private running = false;
  private target: T4;
  private common?: T4;
  private arm: Arm;
  private ifce: (Arm | undefined)[];
  private up: Parameter[] = [];
  private pre: Parameter[] = [];
  private trunk: Conv[];
  private stabiliser?: Conv;
  private output: Conv;
  private debugParameterGradients = false;
  readonly parameterCount: number;
  readonly latentCount: number;
  constructor(readonly spec: TensorGraphSpec, target: Float32Array, common: Float32Array = new Float32Array(0)) {
    const a = spec.architecture;
    if (!a.height || !a.width || spec.group_lengths.length !== 8 || !spec.latents.length) throw new Error('Invalid tensor graph descriptor');
    const offsets = Array<number>(8).fill(0);
    const param = (group: number, length: number, trainable: boolean): Parameter => {
      if (!Number.isSafeInteger(length) || length < 0) throw new Error('Invalid parameter length');
      const p = { key: `${group}:${offsets[group]}`, group, offset: offsets[group], length, trainable };
      offsets[group] += length; this.refs.push(p);
      this.vars.set(p.key, tf.tidy(() => tf.variable(tf.zeros([length]), group !== 5)));
      return p;
    };
    const conv = (ci: number, co: number, k: number, residual: boolean, relu: boolean, group: number, trainable = true): Conv => ({
      ci, co, k, residual, relu, weight: param(group, ci * co * k * k, trainable), bias: param(group + 1, co, trainable),
    });
    const arm = (ci: number, co: number, hidden: number, stabiliser: boolean, group: number): Arm => ({
      layers: [...Array.from({ length: hidden }, () => conv(ci, ci, 1, true, true, group)), conv(ci, co, 1, false, false, group)],
      stabiliser: stabiliser ? conv(ci, co, 1, false, false, group) : undefined,
    });
    this.arm = arm(a.spatial_context + a.ifce_outputs, 2, a.arm_hidden, a.arm_stabiliser, 0);
    this.ifce = spec.ifce_inputs.map(n => n > 0 && a.ifce_outputs ? arm(n, a.ifce_outputs, 0, false, 2) : undefined);
    for (const destination of [this.up, this.pre]) {
      const k = destination === this.up ? a.up_kernel : a.pre_kernel;
      for (let i = 0; i < a.latent_max; i++) { destination.push(param(4, Math.ceil(k / 2), true)); param(5, 1, false); }
    }
    this.output = conv(3, 3, 1, false, false, 6, false);
    if (a.synthesis_stabiliser) this.stabiliser = conv(a.latent_max - a.latent_min + 1, 3, 1, false, false, 6);
    let ci = (a.latent_max - a.latent_min + 1) * (a.common_randomness ? 2 : 1);
    this.trunk = a.synthesis.map(l => { const result = conv(ci, l.outputs, l.kernel, l.residual, l.relu, 6); ci = l.outputs; return result; });
    if (offsets.some((length, i) => length !== spec.group_lengths[i])) { this.dispose(); throw new Error('Reference parameter layout mismatch'); }
    this.parameterCount = offsets.reduce((a, b) => a + b, 0);
    this.latentCount = spec.latents.reduce((n, l) => n + l.height * l.width, 0);
    this.latentVars = spec.latents.map(l => tf.tidy(() => tf.variable(tf.zeros([1, l.height, l.width, 1]))));
    this.noiseVars = spec.latents.map(l => tf.tidy(() => tf.variable(tf.zeros([1, l.height, l.width, 1]), false)));
    this.target = tf.tidy(() => tf.tensor4d(new Float32Array(target), [1, 3, a.height, a.width]).transpose([0, 2, 3, 1]) as T4);
    if (a.common_randomness) {
      this.common = tf.tidy(() => {
        let offset = 0;
        const grids = spec.latents.filter(l => !l.hyper).map(l => {
          const length = l.height * l.width;
          const grid = tf.tensor4d(new Float32Array(common.subarray(offset, offset + length)), [1, l.height, l.width, 1]);
          offset += length; return grid;
        });
        if (offset !== common.length) throw new Error('Invalid common-noise payload');
        let dense = grids[grids.length - 1];
        for (let i = grids.length - 2; i >= 0; i--) {
          const high = grids[i];
          if (dense.shape[1] !== high.shape[1] || dense.shape[2] !== high.shape[2]) dense = this.crop(this.resize(dense, dense.shape[1] * 2, dense.shape[2] * 2, 'bicubic'), high.shape[1], high.shape[2]);
          dense = tf.concat([high, dense], 3) as T4;
        }
        return this.resize(dense, a.height, a.width, 'bicubic');
      });
    }
    this.refs.sort((a, b) => a.group - b.group || a.offset - b.offset);
    this.recorder = new TensorReplay(tf.backend() as WebGPUBackend);
  }
  private constant(key: string, create: () => tf.Tensor) {
    let value = this.constants.get(key);
    if (!value) { value = tf.keep(create()); this.constants.set(key, value); }
    return value;
  }
  private parameter(p: Parameter) { return this.vars.get(p.key)!; }
  private convolution(x: T4, layer: Conv): T4 {
    return fusedConvolution(x, this.parameter(layer.weight), this.parameter(layer.bias), layer.co, layer.k, layer.residual, layer.relu,
      this.debugParameterGradients || layer.weight.trainable);
  }
  private runArm(x: T4, arm: Arm) {
    let y = x;
    for (const layer of arm.layers) y = this.convolution(y, layer);
    return arm.stabiliser ? y.add(this.convolution(x, arm.stabiliser)) as T4 : y;
  }
  private crop(x: T4, h: number, w: number) { return x.slice([0, 0, 0, 0], [1, h, w, x.shape[3]]) as T4; }
  private expand(x: T4, h: number, w: number, identity: boolean) {
    if (identity && x.shape[1] === h && x.shape[2] === w) return x;
    return this.crop(tf.image.resizeNearestNeighbor(x, [x.shape[1] * 2, x.shape[2] * 2], false), h, w);
  }
  private filter(x: T4, p: Parameter, kernel: number, up: boolean): T4 {
    return fusedFilter(x, this.parameter(p), kernel, up);
  }
  private resize(x: T4, h: number, w: number, mode: GraphArchitecture['interpolation']): T4 {
    if (x.shape[1] === h && x.shape[2] === w) return x;
    // Constant interpolation plans reproduce align_corners=False, including
    // replicated bicubic borders. The ordinary presets use the identity path.
    const axis = (input: T4, size: number, dimension: 1 | 2): T4 => {
      const n = input.shape[dimension];
      const count = mode === 'bicubic' ? 4 : mode === 'bilinear' ? 2 : 1;
      const key = `resize:${n}:${size}:${mode}`;
      let indices: number[] = [], weights: number[] = [];
      const f = Math.fround;
      const cubic1 = (x: number) => f(f(f(f(f(1.25 * x) - 2.25) * x) * x) + 1);
      const cubic2 = (x: number) => f(f(f(f(f(f(-0.75 * x) + 3.75) * x) - 6) * x) + 3);
      for (let target = 0; target < size; target++) {
        const scale = f(n / size), source = f(f(scale * f(target + 0.5)) - 0.5);
        if (mode === 'nearest') { indices.push(Math.min(n - 1, Math.floor(f(target * scale)))); weights.push(1); }
        else if (mode === 'bilinear') {
          const pos = Math.max(0, source), low = Math.min(n - 1, Math.floor(pos)), t = f(pos - low);
          indices.push(low, Math.min(low + 1, n - 1)); weights.push(f(1 - t), t);
        } else {
          const base = Math.floor(source), t = f(source - base);
          indices.push(...[0, 1, 2, 3].map(i => Math.max(0, Math.min(n - 1, base + i - 1))));
          weights.push(cubic2(f(t + 1)), cubic1(t), cubic1(f(1 - t)), cubic2(f(2 - t)));
        }
      }
      const idx = this.constant(key + ':i', () => tf.tensor2d(indices, [size, count], 'int32'));
      const wt = this.constant(key + `:${dimension}:w`, () => tf.tensor(weights, dimension === 1 ? [1, size, count, 1, 1] : [1, 1, size, count, 1]));
      return tf.gather(input, idx, dimension).mul(wt).sum(dimension + 1) as T4;
    };
    return axis(axis(x, w, 2), h, 1);
  }
  private quantize(x: T4, noise: T4, options: GraphOptions): T4 {
    return fusedQuantize(x, noise, options);
  }
  private rate(x: T4, raw: T4): T4 {
    return fusedRate(x, raw);
  }
  private forward(noise: Float32Array, options: GraphOptions) {
    const a = this.spec.architecture;
    const quantized = this.latentVars.map((variable, i) => this.quantize(variable as T4, this.noiseVars[i] as T4, options));
    const rates = quantized.map((latent, index) => {
      const h = latent.shape[1], w = latent.shape[2];
      const contexts = [fusedContext(latent, this.spec.neighbours)];
      if (a.ifce_outputs) {
        const network = this.ifce[index];
        let inter: T4;
        if (network) {
          let coarse: T4 = quantized[quantized.length - 1];
          if (index === quantized.length - 1) coarse = tf.zerosLike(latent);
          else for (let i = quantized.length - 2; i > index; i--) coarse = tf.concat([quantized[i], this.expand(coarse, quantized[i].shape[1], quantized[i].shape[2], true)], 3) as T4;
          inter = this.expand(this.runArm(coarse, network), h, w, false);
        } else inter = tf.zeros([1, h, w, a.ifce_outputs]);
        contexts.push(inter);
      }
      return this.rate(latent, this.runArm(tf.concat(contexts, 3) as T4, this.arm));
    });
    const synthesis = quantized.filter((_, i) => !this.spec.latents[i].hyper);
    let dense = synthesis[synthesis.length - 1];
    for (let i = synthesis.length - 2, stage = 0; i >= 0; i--, stage++) {
      const high = synthesis[i];
      const up = this.crop(this.filter(dense, this.up[stage % this.up.length], a.up_kernel, true), high.shape[1], high.shape[2]);
      dense = tf.concat([this.filter(high, this.pre[stage % this.pre.length], a.pre_kernel, false), up], 3) as T4;
    }
    const input = this.common ? tf.concat([dense, this.common], 3) as T4 : dense;
    let native = input;
    for (const layer of this.trunk) native = this.convolution(native, layer);
    if (this.stabiliser) native = native.add(this.convolution(input.slice([0, 0, 0, 0], [-1, -1, -1, this.stabiliser.ci]) as T4, this.stabiliser)) as T4;
    const raw = this.resize(this.convolution(native, this.output), a.height, a.width, a.interpolation);
    const clip = tf.customGrad<T4>((...args) => {
      const x = args[0] as T4, save = args[1] as tf.GradSaveFunc;
      save([x.greater(0).logicalAnd(x.less(1)).cast('float32')]);
      return { value: x.clipByValue(0, 1) as T4, gradFunc: (dy: T4, saved: tf.Tensor[]) => dy.mul(saved[0]) as T4 };
    });
    const image = clip(raw), mse = image.sub(this.target).square().mean();
    const rate = tf.addN(rates.map(r => r.sum())).div(a.height * a.width);
    return { image, rates, mse, rate, loss: mse.add(rate.mul(options.lambda)) as tf.Scalar };
  }
  async gradient(parameters: Float32Array, latents: Float32Array, noise: Float32Array, options: GraphOptions, debug = false): Promise<Float32Array> {
    if (gpuFailure) throw new Error(gpuFailure);
    if (this.running) throw new Error('Concurrent GPU gradient calls are not supported');
    this.running = true;
    const device = (tf.backend() as WebGPUBackend).device;
    device.pushErrorScope('validation');
    try {
      const result = await this.computeGradient(parameters, latents, noise, options, debug);
      if (!result.every(Number.isFinite)) throw new Error('Non-finite GPU graph gradient');
      return result;
    } finally {
      this.running = false;
      const error = await device.popErrorScope();
      if (error) throw new Error(error.message);
      if (gpuFailure) throw new Error(gpuFailure);
    }
  }
  private async computeGradient(parameters: Float32Array, latents: Float32Array, noise: Float32Array, options: GraphOptions, debug: boolean): Promise<Float32Array> {
    if (parameters.length !== this.parameterCount || latents.length !== this.latentCount || noise.length !== this.latentCount) throw new Error('Tensor graph payload length mismatch');
    const key = JSON.stringify([options.mode, options.temperature, options.normalizer, options.lambda]);
    if (!debug && this.plan?.key === key) return this.plan.graph.run(parameters, latents, noise);
    if (!debug) { this.plan?.graph.dispose(); this.plan = undefined; }
    tf.tidy(() => {
      const offsets = [0]; for (const n of this.spec.group_lengths) offsets.push(offsets[offsets.length - 1] + n);
      for (const p of this.refs) this.parameter(p).assign(tf.tensor1d(new Float32Array(parameters.subarray(offsets[p.group] + p.offset, offsets[p.group] + p.offset + p.length))));
      let offset = 0;
      for (let i = 0; i < this.latentVars.length; i++) {
        const variable = this.latentVars[i], n = variable.size;
        variable.assign(tf.tensor(new Float32Array(latents.subarray(offset, offset + n)), variable.shape));
        this.noiseVars[i].assign(tf.tensor(new Float32Array(noise.subarray(offset, offset + n)), variable.shape));
        offset += n;
      }
    });
    this.debugParameterGradients = debug;
    const build = () => tf.tidy(() => {
      const trainable = [...this.refs.filter(p => debug ? p.group !== 5 : p.trainable).map(p => this.parameter(p)), ...this.latentVars];
      const { grads } = tf.variableGrads(() => this.forward(noise, options).loss, trainable);
      const tensors = [...this.refs.map(p => grads[this.parameter(p).name]?.reshape([-1]) ?? tf.zeros([p.length])), ...this.latentVars.map(v => grads[v.name].reshape([-1]))];
      if (debug) {
        const result = this.forward(noise, options);
        tensors.unshift(result.loss.reshape([1]), result.mse.reshape([1]), result.rate.reshape([1]), result.image.transpose([0, 3, 1, 2]).reshape([-1]), ...result.rates.map(r => r.reshape([-1])));
      }
      return tf.concat(tensors) as tf.Tensor1D;
    });
    let packed: tf.Tensor1D;
    if (debug) packed = build();
    else {
      const backend = tf.backend() as WebGPUBackend;
      const inputs: GraphInputBuffer[] = [];
      const prepare = (tensor: tf.Tensor, kind: GraphInputBuffer['kind'], offset: number) => {
        (backend as unknown as { uploadToGPU(id: object): void }).uploadToGPU(tensor.dataId);
        const resource = backend.tensorMap.get(tensor.dataId).resource as GPUBuffer;
        inputs.push({ kind, offset, length: tensor.size, buffer: resource });
      };
      const offsets = [0]; for (const n of this.spec.group_lengths) offsets.push(offsets[offsets.length - 1] + n);
      for (const p of this.refs) prepare(this.parameter(p), 'parameters', offsets[p.group] + p.offset);
      let offset = 0;
      for (let i = 0; i < this.latentVars.length; i++) {
        prepare(this.latentVars[i], 'latents', offset); prepare(this.noiseVars[i], 'noise', offset); offset += this.latentVars[i].size;
      }
      const capture = this.recorder.capture(build);
      packed = capture.value;
      const output = backend.tensorMap.get(packed.dataId).resource as GPUBuffer;
      this.plan = { key, graph: new CapturedTensorGradient(backend.device, capture.batches, output, packed.size, inputs, capture.release) };
    }
    try {
      const result = await packed.data() as Float32Array;
      if ((tf.backend() as WebGPUBackend).memory().numBytesAllocatedInGPU > 1536 * 1024 ** 2) throw new Error('Tensor graph exceeded GPU memory budget');
      return result;
    } finally { packed.dispose(); }
  }
  dispose() {
    for (const variable of this.vars.values()) variable.dispose(); this.vars.clear();
    this.latentVars?.forEach(v => v.dispose()); this.noiseVars?.forEach(v => v.dispose()); this.target?.dispose(); this.common?.dispose();
    for (const tensor of this.constants.values()) tensor.dispose(); this.constants.clear();
    this.plan?.graph.dispose(); this.plan = undefined; this.recorder?.dispose();
  }
}
