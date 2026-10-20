import * as tf from '@tensorflow/tfjs-core';
import type { WebGPUBackend } from '@tensorflow/tfjs-backend-webgpu';
import type { WebGPUProgram } from '@tensorflow/tfjs-backend-webgpu/dist/webgpu_program';
import { getMainHeaderString as main } from '@tensorflow/tfjs-backend-webgpu/dist/webgpu_program';
import { computeDispatch, flatDispatchLayout } from '@tensorflow/tfjs-backend-webgpu/dist/webgpu_util';

type T4 = tf.Tensor4D;
type Attrs = Record<string, tf.Attribute>;
const float = (n: number) => Math.fround(n).toExponential(9);
class Program implements WebGPUProgram {
  size = true; workgroupSize: [number, number, number] = [64, 1, 1];
  dispatchLayout; dispatch; shaderKey: string;
  constructor(readonly outputShape: number[], readonly variableNames: string[], private code: string, prefix = '') {
    this.dispatchLayout = flatDispatchLayout(outputShape);
    this.dispatch = computeDispatch(this.dispatchLayout, outputShape, this.workgroupSize);
    this.shaderKey = `coolchic_${prefix}_${code}`;
  }
  getUserCode() { return this.code; }
}
const run = (name: string, inputs: tf.NamedTensorMap, attrs: Attrs) => tf.engine().runKernel(name, inputs, attrs) as tf.Tensor;
const body = (code: string) => `${main('index')} { if (index >= uniforms.size) { return; } ${code} }`;
function kernel(name: string, create: (inputs: tf.NamedTensorInfoMap, attrs: Attrs, backend: WebGPUBackend) => tf.TensorInfo) {
  tf.registerKernel({ kernelName: name, backendName: 'webgpu', kernelFunc: args => create(args.inputs as tf.NamedTensorInfoMap, args.attrs as Attrs, args.backend as WebGPUBackend) });
}

export function registerFusedCoolChicKernels() {
  for (const gradient of [false, true]) kernel(gradient ? 'CoolQuantGrad' : 'CoolQuant', (inputs, attrs, backend) => {
    const x = inputs.x!, mode = attrs.mode as number, temp = attrs.temperature as number, normalizer = attrs.normalizer as number;
    const t = float(mode === 1 || mode === 2 ? temp : 1), norm = float(mode === 1 || mode === 2 ? normalizer : 1);
    const pair = `fn softPair(x: f32) -> vec2<f32> {
      let f = floor(x); let v = tanh((x - f - 0.5) / ${t});
      return vec2<f32>(f + 0.5 * v / ${norm} + 0.5, 0.5 * (1.0 - v*v) / (${t} * ${norm}));
    }`;
    const expression = mode === 0 ? (gradient ? '16.0 * getDyByOutputIndex(index)' : 'getXByOutputIndex(index) * 16.0 + getNoiseByOutputIndex(index)')
      : mode >= 3 ? (gradient ? '0.0' : 'round(getXByOutputIndex(index) * 16.0)')
      : gradient ? (mode === 2 ? 'getDyByOutputIndex(index) * softPair(first.x + getNoiseByOutputIndex(index)).y * first.y * 16.0' : 'getDyByOutputIndex(index) * first.y * 16.0')
      : mode === 2 ? 'softPair(first.x + getNoiseByOutputIndex(index)).x' : 'first.x';
    const code = pair + body(`${mode === 1 || mode === 2 ? 'let first = softPair(getXByOutputIndex(index) * 16.0);' : ''} setOutputAtIndex(index, ${expression});`);
    if (gradient && mode >= 3) return backend.makeTensorInfo(x.shape, 'float32', new Float32Array(x.shape.reduce((a,b) => a*b, 1)));
    const names = gradient ? (mode === 0 ? ['dy'] : mode === 1 ? ['x','dy'] : ['x','noise','dy'])
      : (mode === 1 || mode >= 3 ? ['x'] : ['x','noise']);
    return backend.runWebGPUProgram(new Program(x.shape, names, code), names.map(name => inputs[name]!), 'float32');
  });
  tf.registerKernel({ kernelName: 'CoolRate_prepare', backendName: 'webgpu', kernelFunc: args => {
    const { x, raw } = args.inputs as tf.NamedTensorInfoMap, backend = args.backend as WebGPUBackend;
    const shape = x!.shape, packedShape = (channels: number) => [...shape.slice(0, 3), channels];
    const base = backend.runWebGPUProgram(new Program(packedShape(4), ['x','raw'], body(`
      let c = getCoordsFromIndex(index); let v = getX(c[0],c[1],c[2],0); let mu = getRaw(c[0],c[1],c[2],0);
      let shift = getRaw(c[0],c[1],c[2],1) - 4.0; var value = shift;
      if (c[3] == 0) { value = v + 0.5 - mu; }
      if (c[3] == 1) { value = v - 0.5 - mu; }
      if (c[3] == 2) { value = exp(clamp(shift,-5.0,5.0)); }
      setOutputAtIndex(index,value);`)), [x!,raw!], 'float32');
    let expm1: tf.TensorInfo | undefined, cdf: tf.TensorInfo | undefined, probability: tf.TensorInfo | undefined;
    try {
      expm1 = backend.runWebGPUProgram(new Program(packedShape(2), ['base'], body(`let c = getCoordsFromIndex(index);
        let d = getBase(c[0],c[1],c[2],c[3]); let s = getBase(c[0],c[1],c[2],2);
        setOutputAtIndex(index,exp(-abs(d)/s)-1.0);`)), [base], 'float32');
      cdf = backend.runWebGPUProgram(new Program(packedShape(2), ['base','expm1'], body(`let c = getCoordsFromIndex(index);
        setOutputAtIndex(index,0.5 - 0.5 * sign(getBase(c[0],c[1],c[2],c[3])) * getExpm1(c[0],c[1],c[2],c[3]));`)), [base,expm1], 'float32');
      probability = backend.runWebGPUProgram(new Program(shape, ['cdf'], body(`let c = getCoordsFromIndex(index);
        setOutputAtIndex(index,getCdf(c[0],c[1],c[2],0)-getCdf(c[0],c[1],c[2],1));`)), [cdf], 'float32');
      const bits = backend.runWebGPUProgram(new Program(shape, ['p'], body(`setOutputAtIndex(index,-log(max(getPByOutputIndex(index),0.0000152587890625))/${float(Math.LN2)});`)), [probability], 'float32');
      return [bits,base,probability];
    } catch (error) { backend.disposeData(base.dataId); if (probability) backend.disposeData(probability.dataId); throw error; }
    finally { if (expm1) backend.disposeData(expm1.dataId); if (cdf) backend.disposeData(cdf.dataId); }
  } });
  for (const part of ['latent', 'arm']) kernel(`CoolRate_${part}`, (inputs, _attrs, backend) => {
    const shape = inputs.probability!.shape, output = part === 'arm' ? [...shape.slice(0, 3), 2] : shape;
    const names = ['base', 'probability', 'dy'];
    const code = `fn density(d: f32, scale: f32) -> f32 { return select(exp(-abs(d) / scale) / (2.0 * scale), 0.0, d == 0.0); }` + body(`
      let c = getCoordsFromIndex(index);
      let upper = getBase(c[0],c[1],c[2],0); let lower = getBase(c[0],c[1],c[2],1);
      let scale = getBase(c[0],c[1],c[2],2); let shifted = getBase(c[0],c[1],c[2],3);
      let p = getProbability(c[0],c[1],c[2],0);
      if (p < 0.0000152587890625) { setOutputAtIndex(index, 0.0); return; }
      let m = -getDy(c[0], c[1], c[2], 0) / (p * ${float(Math.LN2)});
      let dx = (density(upper, scale) - density(lower, scale)) * m;
      ${part === 'latent' ? 'setOutputAtIndex(index, dx);' : `
      if (c[3] == 0) { setOutputAtIndex(index, -dx); }
      else {
        var ds = 0.0;
        if (shifted > -5.0 && shifted < 5.0) { ds = m * (-upper * exp(-abs(upper)/scale) + lower * exp(-abs(lower)/scale)) / (2.0 * scale * scale) * scale; }
        setOutputAtIndex(index, ds);
      }`}
    `);
    return backend.runWebGPUProgram(new Program(output, names, code), names.map(name => inputs[name]!), 'float32');
  });
  for (const gradient of [false, true]) kernel(gradient ? 'CoolContextGrad' : 'CoolContext', (inputs, attrs, backend) => {
    const shape = attrs.shape as number[], offsets = attrs.offsets as number[], n = offsets.length / 2;
    const [, h, w] = shape;
    const output = gradient ? shape : [shape[0], h, w, n];
    const lookup = `const neighbours = array<vec2<i32>, ${n}>(${Array.from({ length: n }, (_, i) => `vec2<i32>(${offsets[i * 2]}, ${offsets[i * 2 + 1]})`).join(',')});`;
    const code = lookup + body(gradient ? `
      let c = getCoordsFromIndex(index); var value = 0.0;
      for (var k = 0; k < ${n}; k++) {
        let y = c[1] - neighbours[k].x; let x = c[2] - neighbours[k].y;
        if (y >= 0 && y < ${h} && x >= 0 && x < ${w}) { value += getDy(c[0], y, x, k); }
      }
      setOutputAtIndex(index, value);
    ` : `
      let c = getCoordsFromIndex(index); let y = c[1] + neighbours[c[3]].x; let x = c[2] + neighbours[c[3]].y;
      var value = 0.0; if (y >= 0 && y < ${h} && x >= 0 && x < ${w}) { value = getX(c[0], y, x, 0); }
      setOutputAtIndex(index, value);
    `);
    const names = gradient ? ['dy'] : ['x'];
    return backend.runWebGPUProgram(new Program(output, names, code), names.map(name => inputs[name]!), 'float32');
  });
  registerAxis();
  registerConvolution();
}

function registerAxis() {
  for (const part of ['value', 'input', 'half']) kernel(`CoolAxis_${part}`, (inputs, attrs, backend) => {
    const shape = attrs.shape as number[], horizontal = attrs.horizontal as boolean, up = attrs.up as boolean, k = attrs.kernel as number;
    const [batch, h, w, channels] = shape, oh = h * (up && !horizontal ? 2 : 1), ow = w * (up && horizontal ? 2 : 1);
    const sourceSize = horizontal ? w : h, targetSize = horizontal ? ow : oh, pad = k >> 1, crop = 3 * pad - 1, half = Math.ceil(k / 2);
    const source = `fn source(position: i32, tap: i32) -> i32 {
      ${up ? `let coordinate = position + ${crop} - tap;
      if (coordinate < 0 || coordinate % 2 != 0) { return -1; }
      let padded = coordinate / 2; if (padded >= ${sourceSize + 2 * pad}) { return -1; }
      return clamp(padded - ${pad}, 0, ${sourceSize - 1});` : `let value = position + tap - ${pad}; return select(-1, value, value >= 0 && value < ${sourceSize});`}
    }`;
    if (part === 'value') {
      const code = source + body(`let c = getCoordsFromIndex(index); var value = 0.0;
        for (var tap = 0; tap < ${k}; tap++) {
          let s = source(c[${horizontal ? 2 : 1}], tap);
          if (s >= 0) { value += getX(c[0], ${horizontal ? 'c[1], s' : 's, c[2]'}, c[3]) * getHalf(min(tap, ${k - 1} - tap)); }
        }
        setOutputAtIndex(index, value);`);
      return backend.runWebGPUProgram(new Program([batch, oh, ow, channels], ['x', 'half'], code), [inputs.x!, inputs.half!], 'float32');
    }
    if (part === 'input') {
      const code = body(`let c = getCoordsFromIndex(index); let s = c[${horizontal ? 2 : 1}]; var value = 0.0;
        for (var tap = 0; tap < ${k}; tap++) {
          ${up ? `let lo = select(s + ${pad}, 0, s == 0);
          let hi = select(s + ${pad + 1}, ${sourceSize + 2 * pad}, s == ${sourceSize - 1});
          for (var padded = lo; padded < hi; padded++) {
            let t = 2 * padded + tap - ${crop};
            if (t >= 0 && t < ${targetSize}) { value += getDy(c[0], ${horizontal ? 'c[1], t' : 't, c[2]'}, c[3]) * getHalf(min(tap, ${k - 1} - tap)); }
          }` : `let t = s + ${pad} - tap;
          if (t >= 0 && t < ${targetSize}) { value += getDy(c[0], ${horizontal ? 'c[1], t' : 't, c[2]'}, c[3]) * getHalf(min(tap, ${k - 1} - tap)); }`}
        }
        setOutputAtIndex(index, value);`);
      return backend.runWebGPUProgram(new Program(shape, ['half', 'dy'], code), [inputs.half!, inputs.dy!], 'float32');
    }
    const size = batch * oh * ow * channels, chunks = Math.ceil(size / 256);
    const code = source + body(`let chunk = index / ${half}; let hi = index % ${half}; var value = 0.0;
      for (var pos = chunk * 256; pos < min((chunk + 1) * 256, ${size}); pos++) {
        let channel = pos % ${channels}; let x = (pos / ${channels}) % ${ow}; let y = (pos / ${channels * ow}) % ${oh}; let b = pos / ${channels * ow * oh};
        let g = getDy(b, y, x, channel);
        for (var side = 0; side < 2; side++) {
          let tap = select(hi, ${k - 1} - hi, side == 1);
          if (side == 1 && tap == hi) { continue; }
          let s = source(${horizontal ? 'x' : 'y'}, tap);
          if (s >= 0) { value += g * getX(b, ${horizontal ? 'y, s' : 's, x'}, channel); }
        }
      }
      setOutputAtIndex(index, value);`);
    const partials = backend.runWebGPUProgram(new Program([chunks, half], ['x', 'dy'], code), [inputs.x!, inputs.dy!], 'float32');
    const reduce = body(`var value = 0.0; for (var chunk = 0; chunk < ${chunks}; chunk++) { value += getPartials(chunk, index); } setOutputAtIndex(index, value);`);
    try { return backend.runWebGPUProgram(new Program([half], ['partials'], reduce), [partials], 'float32'); }
    finally { backend.disposeData(partials.dataId); }
  });
}

export function fusedQuantize(x: T4, noise: T4, options: { mode: number; temperature: number; normalizer?: number }): T4 {
  const attrs = { ...options, normalizer: options.normalizer ?? Math.fround(Math.tanh(Math.fround(1 / Math.fround(2 * options.temperature)))) };
  const op = tf.customGrad<T4>((...args) => {
    const input = args[0] as T4, save = args[1] as tf.GradSaveFunc;
    save([input, noise]);
    return { value: run('CoolQuant', { x: input, noise }, attrs) as T4,
      gradFunc: (dy: T4, saved: tf.Tensor[]) => run('CoolQuantGrad', { x: saved[0], noise: saved[1], dy }, attrs) };
  });
  return op(x);
}
export function fusedRate(x: T4, raw: T4): T4 {
  const op = tf.customGrad<T4>((...args) => {
    const latent = args[0] as T4, arm = args[1] as T4, save = args[2] as tf.GradSaveFunc;
    // Explicit GPU rounding boundaries around expm1/CDF subtraction preserve
    // the reference probabilities near the 2^-16 floor without host roundtrips.
    const [value, base, probability] = tf.engine().runKernel('CoolRate_prepare', { x: latent, raw: arm }, {}) as tf.Tensor[];
    save([base, probability]);
    return { value: value as T4, gradFunc: (dy: T4, saved: tf.Tensor[]) => {
      const inputs = { base: saved[0], probability: saved[1], dy };
      return [run('CoolRate_latent', inputs, {}), run('CoolRate_arm', inputs, {})];
    } };
  });
  return op(x, raw);
}
export function fusedContext(x: T4, neighbours: [number, number][]): T4 {
  if (!neighbours.length) return tf.zeros([1, x.shape[1], x.shape[2], 0]);
  const attrs = { shape: x.shape, offsets: neighbours.flat() };
  const op = tf.customGrad<T4>((...args) => ({ value: run('CoolContext', { x: args[0] as T4 }, attrs) as T4,
    gradFunc: (dy: T4) => run('CoolContextGrad', { dy }, attrs) }));
  return op(x);
}
export function fusedFilter(x: T4, half: tf.Tensor, kernel: number, up: boolean): T4 {
  const axis = (input: T4, horizontal: boolean): T4 => {
    const attrs = { shape: input.shape, horizontal, kernel, up };
    const op = tf.customGrad<T4>((...args) => {
      const value = args[0] as T4, weights = args[1] as tf.Tensor, save = args[2] as tf.GradSaveFunc;
      save([value, weights]);
      return { value: run('CoolAxis_value', { x: value, half: weights }, attrs) as T4,
        gradFunc: (dy: T4, saved: tf.Tensor[]) => [run('CoolAxis_input', { half: saved[1], dy }, attrs), run('CoolAxis_half', { x: saved[0], dy }, attrs)] };
    });
    return op(input, half);
  };
  const output = axis(axis(x, true), false);
  return up ? output : output.add(x) as T4;
}

function registerConvolution() {
  for (const part of ['value', 'input', 'parameters']) tf.registerKernel({ kernelName: `CoolConv_${part}`, backendName: 'webgpu', kernelFunc: args => {
    const inputs = args.inputs as tf.NamedTensorInfoMap, backend = args.backend as WebGPUBackend;
    const a = args.attrs as { shape: number[]; co: number; kernel: number; residual: boolean; relu: boolean };
    const [batch, h, w, ci] = a.shape, co = a.co, k = a.kernel, pad = k >> 1, nw = ci * co * k * k;
    const gate = `fn gate(b: i32, y: i32, x: i32, c: i32) -> f32 {
      ${a.relu ? 'if (getAct(b,y,x,c) <= 0.0) { return 0.0; }' : ''}
      return getDy(b,y,x,c);
    }`;
    if (part === 'value') {
      const code = body(`let c = getCoordsFromIndex(index); var value = 0.0;
        for (var ic = 0; ic < ${ci}; ic++) {
          for (var ky = 0; ky < ${k}; ky++) { for (var kx = 0; kx < ${k}; kx++) {
            let y = clamp(c[1] + ky - ${pad}, 0, ${h - 1}); let x = clamp(c[2] + kx - ${pad}, 0, ${w - 1});
            value += getX(c[0], y, x, ic) * getWeights(((c[3] * ${ci} + ic) * ${k} + ky) * ${k} + kx);
          } }
        }
        value += getBias(c[3]);
        ${a.residual ? 'value += getX(c[0],c[1],c[2],c[3]);' : ''}
        ${a.relu ? 'value = max(value, 0.0);' : ''}
        setOutputAtIndex(index, value);`);
      return backend.runWebGPUProgram(new Program([batch, h, w, co], ['x','weights','bias'], code), [inputs.x!, inputs.weights!, inputs.bias!], 'float32');
    }
    if (part === 'input') {
      const code = gate + body(`let c = getCoordsFromIndex(index); var value = ${a.residual ? 'gate(c[0],c[1],c[2],c[3])' : '0.0'};
        for (var oc = 0; oc < ${co}; oc++) {
          for (var ky = 0; ky < ${k}; ky++) {
            let yy = c[1] + ${pad} - ky;
            let loY = max(0, select(yy, 0, c[1] == 0));
            let hiY = min(${h - 1}, select(yy, ${h - 1}, c[1] == ${h - 1}));
            for (var kx = 0; kx < ${k}; kx++) {
              let xx = c[2] + ${pad} - kx;
              let loX = max(0, select(xx, 0, c[2] == 0));
              let hiX = min(${w - 1}, select(xx, ${w - 1}, c[2] == ${w - 1}));
              let weight = getWeights(((oc * ${ci} + c[3]) * ${k} + ky) * ${k} + kx);
              for (var y = loY; y <= hiY; y++) { for (var x = loX; x <= hiX; x++) { value += gate(c[0],y,x,oc) * weight; } }
            }
          }
        }
        setOutputAtIndex(index, value);`);
      const names = a.relu ? ['weights','act','dy'] : ['weights','dy'];
      return backend.runWebGPUProgram(new Program(a.shape, names, code), names.map(name => inputs[name]!), 'float32');
    }
    const size = batch * h * w, chunks = Math.ceil(size / 256), parameters = nw + co;
    const code = gate + body(`let param = index % ${parameters}; let chunk = index / ${parameters};
      let isBias = param >= ${nw}; var oc = param - ${nw}; var ic = 0; var ky = 0; var kx = 0;
      if (!isBias) { oc = param / ${ci * k * k}; ic = (param / ${k * k}) % ${ci}; ky = (param / ${k}) % ${k}; kx = param % ${k}; }
      var value = 0.0;
      for (var pos = chunk * 256; pos < min((chunk + 1) * 256, ${size}); pos++) {
        let x = pos % ${w}; let y = (pos / ${w}) % ${h}; let b = pos / ${w * h};
        var sample = 1.0;
        if (!isBias) { sample = getX(b, clamp(y + ky - ${pad}, 0, ${h - 1}), clamp(x + kx - ${pad}, 0, ${w - 1}), ic); }
        value += gate(b,y,x,oc) * sample;
      }
      setOutputAtIndex(index, value);`);
    const names = a.relu ? ['x','act','dy'] : ['x','dy'];
    const partials = backend.runWebGPUProgram(new Program([chunks, parameters], names, code), names.map(name => inputs[name]!), 'float32');
    const reduce = (length: number, start: number) => new Program([length], ['partials'], body(`var value = 0.0;
      for (var chunk = 0; chunk < ${chunks}; chunk++) { value += getPartials(chunk, index + ${start}); }
      setOutputAtIndex(index, value);`));
    try { return [backend.runWebGPUProgram(reduce(nw, 0), [partials], 'float32'), backend.runWebGPUProgram(reduce(co, nw), [partials], 'float32')]; }
    finally { backend.disposeData(partials.dataId); }
  } });
}

export function fusedConvolution(x: T4, weights: tf.Tensor, bias: tf.Tensor,
  co: number, kernel: number, residual: boolean, relu: boolean, parameters = true): T4 {
  const attrs = { shape: x.shape, co, kernel, residual, relu };
  const op = tf.customGrad<T4>((...args) => {
    const input = args[0] as T4, w = args[1] as tf.Tensor, b = args[2] as tf.Tensor, save = args[3] as tf.GradSaveFunc;
    const value = run('CoolConv_value', { x: input, weights: w, bias: b }, attrs) as T4;
    save([input, w, value]);
    return { value, gradFunc: (dy: T4, saved: tf.Tensor[]) => {
      const dx = run('CoolConv_input', { weights: saved[1], act: saved[2], dy }, attrs);
      const grads = parameters ? tf.engine().runKernel('CoolConv_parameters', { x: saved[0], act: saved[2], dy }, attrs) as tf.Tensor[]
        : [tf.zerosLike(w), tf.zerosLike(b)];
      return [dx, ...grads];
    } };
  });
  return op(x, weights, bias);
}
