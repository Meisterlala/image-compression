import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-core/dist/public/chained_ops/register_all_chained_ops';
import '@tensorflow/tfjs-core/dist/register_all_gradients';
import '@tensorflow/tfjs-backend-webgpu';
import type { WebGPUBackend } from '@tensorflow/tfjs-backend-webgpu';
import { addN as gpuAddN } from '@tensorflow/tfjs-backend-webgpu/dist/kernels/AddN';

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
  private target: T4;
  private common?: T4;
  private arm: Arm;
  private ifce: (Arm | undefined)[];
  private up: Parameter[] = [];
  private pre: Parameter[] = [];
  private trunk: Conv[];
  private stabiliser?: Conv;
  private output: Conv;
  private transformed = new Map<string, tf.Tensor>();
  readonly parameterCount: number;
  readonly latentCount: number;
  constructor(readonly spec: TensorGraphSpec, target: Float32Array, common = new Float32Array(0)) {
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
  }
  private constant(key: string, create: () => tf.Tensor) {
    let value = this.constants.get(key);
    if (!value) { value = tf.keep(create()); this.constants.set(key, value); }
    return value;
  }
  private parameter(p: Parameter) { return this.vars.get(p.key)!; }
  private convolution(x: T4, layer: Conv): T4 {
    let weight = this.transformed.get(layer.weight.key);
    if (!weight) {
      weight = this.parameter(layer.weight).reshape([layer.co, layer.ci, layer.k, layer.k]).transpose([2, 3, 1, 0]);
      this.transformed.set(layer.weight.key, weight);
    }
    let y = tf.conv2d(this.replicate(x, layer.k >> 1, layer.k >> 1), weight as tf.Tensor4D, 1, 'valid').add(this.parameter(layer.bias)) as T4;
    if (layer.residual) y = y.add(x) as T4;
    if (layer.relu) y = y.relu() as T4;
    return y;
  }
  private runArm(x: T4, arm: Arm) {
    let y = x;
    for (const layer of arm.layers) y = this.convolution(y, layer);
    return arm.stabiliser ? y.add(this.convolution(x, arm.stabiliser)) as T4 : y;
  }
  private replicate(x: T4, py: number, px: number): T4 {
    let y = x;
    if (px) y = tf.concat([x.slice([0, 0, 0, 0], [-1, -1, 1, -1]).tile([1, 1, px, 1]), x,
      x.slice([0, 0, x.shape[2] - 1, 0], [-1, -1, 1, -1]).tile([1, 1, px, 1])], 2) as T4;
    if (py) y = tf.concat([y.slice([0, 0, 0, 0], [-1, 1, -1, -1]).tile([1, py, 1, 1]), y,
      y.slice([0, y.shape[1] - 1, 0, 0], [-1, 1, -1, -1]).tile([1, py, 1, 1])], 1) as T4;
    return y;
  }
  private crop(x: T4, h: number, w: number) { return x.slice([0, 0, 0, 0], [1, h, w, x.shape[3]]) as T4; }
  private expand(x: T4, h: number, w: number, identity: boolean) {
    if (identity && x.shape[1] === h && x.shape[2] === w) return x;
    return this.crop(tf.image.resizeNearestNeighbor(x, [x.shape[1] * 2, x.shape[2] * 2], false), h, w);
  }
  private filter(x: T4, p: Parameter, kernel: number, up: boolean): T4 {
    let full = this.transformed.get(p.key);
    if (!full) {
      const half = this.parameter(p);
      full = tf.concat([half, half.slice([0], [Math.floor(kernel / 2)]).reverse()], 0);
      this.transformed.set(p.key, full);
    }
    const c = x.shape[3];
    if (!up) {
      const horizontal = full.reshape([1, kernel, 1, 1]).tile([1, 1, c, 1]) as T4;
      const vertical = full.reshape([kernel, 1, 1, 1]).tile([1, 1, c, 1]) as T4;
      return tf.depthwiseConv2d(tf.depthwiseConv2d(x, horizontal, 1, 'same'), vertical, 1, 'same').add(x) as T4;
    }
    const diagonal = this.constant(`eye:${c}`, () => tf.eye(c));
    const matrix = full.reshape([kernel, 1, 1]).mul(diagonal.reshape([1, c, c]));
    const pad = kernel / 2, crop = 3 * pad - 1;
    const paddedX = this.replicate(x, 0, pad);
    const width = (paddedX.shape[2] - 1) * 2 + kernel;
    const h = tf.conv2dTranspose(paddedX, matrix.reshape([1, kernel, c, c]) as T4, [1, x.shape[1], width, c], [1, 2], 'valid')
      .slice([0, 0, crop, 0], [1, x.shape[1], x.shape[2] * 2, c]) as T4;
    const paddedY = this.replicate(h, pad, 0);
    const height = (paddedY.shape[1] - 1) * 2 + kernel;
    return tf.conv2dTranspose(paddedY, matrix.reshape([kernel, 1, c, c]) as T4, [1, height, h.shape[2], c], [2, 1], 'valid')
      .slice([0, crop, 0, 0], [1, x.shape[1] * 2, h.shape[2], c]) as T4;
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
    const { mode, temperature } = options;
    const normalizer = options.normalizer ?? Math.fround(Math.tanh(Math.fround(1 / Math.fround(2 * temperature))));
    const pair = (value: tf.Tensor) => {
      const floor = value.floor();
      const t = value.sub(floor).sub(0.5).div(temperature).tanh();
      return { value: floor.add(t.mul(0.5).div(normalizer)).add(0.5), derivative: tf.scalar(1).sub(t.square()).mul(0.5).div(Math.fround(temperature * normalizer)) };
    };
    const op = tf.customGrad<T4>((...args) => {
      const input = args[0] as T4, save = args[1] as tf.GradSaveFunc;
      if (mode === 0) return { value: input.add(noise) as T4, gradFunc: (dy: T4) => dy };
      if (mode === 3 || mode === 4) return { value: input.round() as T4, gradFunc: () => tf.zerosLike(input) };
      const first = pair(input);
      const second = mode === 2 ? pair(first.value.add(noise)) : first;
      save([mode === 2 ? second.derivative.mul(first.derivative) : first.derivative]);
      return { value: second.value as T4, gradFunc: (dy: T4, saved: tf.Tensor[]) => dy.mul(saved[0]) as T4 };
    });
    return op(x);
  }
  private rate(x: T4, raw: T4): T4 {
    const calculate = (latent: T4, arm: T4) => {
      const mu = arm.slice([0, 0, 0, 0], [-1, -1, -1, 1]);
      const shifted = arm.slice([0, 0, 0, 1], [-1, -1, -1, 1]).sub(4);
      const scale = shifted.clipByValue(-5, 5).exp();
      const upper = latent.add(0.5).sub(mu), lower = latent.sub(0.5).sub(mu);
      const cdf = (d: tf.Tensor) => tf.scalar(0.5).sub(d.sign().mul(d.abs().neg().div(scale).expm1()).mul(0.5));
      return { upper, lower, scale, shifted, probability: cdf(upper).sub(cdf(lower)) };
    };
    const op = tf.customGrad<T4>((...args) => {
      const latent = args[0] as T4, arm = args[1] as T4, save = args[2] as tf.GradSaveFunc;
      const { probability } = calculate(latent, arm);
      save([latent, arm]);
      return { value: probability.maximum(2 ** -16).log().div(-Math.fround(Math.LN2)) as T4,
        gradFunc: (dy: T4, saved: tf.Tensor[]) => {
          const { upper, lower, scale, shifted, probability: p } = calculate(saved[0] as T4, saved[1] as T4);
          const eu = upper.abs().neg().div(scale).exp(), el = lower.abs().neg().div(scale).exp();
          const density = (d: tf.Tensor, e: tf.Tensor) => tf.where(d.equal(0), tf.zerosLike(d), e.div(scale.mul(2)));
          const multiplier = tf.where(p.greaterEqual(2 ** -16), p.maximum(2 ** -16).mul(Math.fround(Math.LN2)).reciprocal().neg(), tf.zerosLike(p)).mul(dy);
          const dx = density(upper, eu).sub(density(lower, el)).mul(multiplier);
          const ds = upper.neg().mul(eu).add(lower.mul(el)).div(scale.mul(2).mul(scale)).mul(multiplier).mul(scale);
          const logGradient = tf.where(shifted.greater(-5).logicalAnd(shifted.less(5)), ds, tf.zerosLike(ds));
          return [dx as T4, tf.concat([dx.neg(), logGradient], 3) as T4];
        } };
    });
    return op(x, raw);
  }
  private forward(noise: Float32Array, options: GraphOptions) {
    const a = this.spec.architecture;
    this.transformed.clear();
    let offset = 0;
    const quantized = this.latentVars.map((variable, i) => {
      const { height: h, width: w } = this.spec.latents[i], size = h * w;
      const sample = tf.tensor4d(new Float32Array(noise.subarray(offset, offset + size)), [1, h, w, 1]); offset += size;
      return this.quantize(variable.mul(16) as T4, sample, options);
    });
    const rates = quantized.map((latent, index) => {
      const h = latent.shape[1], w = latent.shape[2];
      const padded = tf.pad(latent, [[0, 0], [4, 4], [4, 4], [0, 0]]);
      const contexts = this.spec.neighbours.map(([dy, dx]) => padded.slice([0, 4 + dy, 4 + dx, 0], [1, h, w, 1]));
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
    const device = (tf.backend() as WebGPUBackend).device;
    device.pushErrorScope('validation');
    try { return await this.computeGradient(parameters, latents, noise, options, debug); }
    finally {
      const error = await device.popErrorScope();
      if (error) throw new Error(error.message);
      if (gpuFailure) throw new Error(gpuFailure);
    }
  }
  private async computeGradient(parameters: Float32Array, latents: Float32Array, noise: Float32Array, options: GraphOptions, debug: boolean): Promise<Float32Array> {
    if (parameters.length !== this.parameterCount || latents.length !== this.latentCount || noise.length !== this.latentCount) throw new Error('Tensor graph payload length mismatch');
    tf.tidy(() => {
      const offsets = [0]; for (const n of this.spec.group_lengths) offsets.push(offsets[offsets.length - 1] + n);
      for (const p of this.refs) this.parameter(p).assign(tf.tensor1d(new Float32Array(parameters.subarray(offsets[p.group] + p.offset, offsets[p.group] + p.offset + p.length))));
      let offset = 0;
      for (const variable of this.latentVars) { const n = variable.size; variable.assign(tf.tensor(new Float32Array(latents.subarray(offset, offset + n)), variable.shape)); offset += n; }
    });
    const packed = tf.tidy(() => {
      const trainable = [...this.refs.filter(p => debug ? p.group !== 5 : p.trainable).map(p => this.parameter(p)), ...this.latentVars];
      const { grads } = tf.variableGrads(() => this.forward(noise, options).loss, trainable);
      const tensors = [...this.refs.map(p => grads[this.parameter(p).name]?.reshape([-1]) ?? tf.zeros([p.length])), ...this.latentVars.map(v => grads[v.name].reshape([-1]))];
      if (debug) {
        const result = this.forward(noise, options);
        tensors.unshift(result.loss.reshape([1]), result.mse.reshape([1]), result.rate.reshape([1]), result.image.transpose([0, 3, 1, 2]).reshape([-1]), ...result.rates.map(r => r.reshape([-1])));
      }
      return tf.concat(tensors) as tf.Tensor1D;
    });
    try {
      const result = await packed.data() as Float32Array;
      if (!result.every(Number.isFinite)) throw new Error('Non-finite GPU graph gradient');
      if ((tf.backend() as WebGPUBackend).memory().numBytesAllocatedInGPU > 1536 * 1024 ** 2) throw new Error('Tensor graph exceeded GPU memory budget');
      return result;
    } finally { packed.dispose(); this.transformed.clear(); }
  }
  dispose() {
    for (const variable of this.vars.values()) variable.dispose(); this.vars.clear();
    this.latentVars?.forEach(v => v.dispose()); this.target?.dispose(); this.common?.dispose();
    for (const tensor of this.constants.values()) tensor.dispose(); this.constants.clear();
  }
}
