// CHW / OIHW replicate-padded convolution and exact analytical derivatives.
// No floating-point atomics: each invocation owns its result. Weight/bias
// gradients use fixed-order partial reductions (float32 rounding may differ).
struct Params {
  ci: u32, co: u32, h: u32, w: u32,
  k: u32, residual: u32, relu: u32, backward: u32,
  chunks: u32, unused0: u32, unused1: u32, unused2: u32,
}
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> data: array<f32>;
@group(0) @binding(2) var<storage, read_write> gated: array<f32>;
@group(0) @binding(3) var<storage, read_write> result: array<f32>;
@group(0) @binding(4) var<storage, read_write> partial: array<f32>;

fn pixels() -> u32 { return p.h * p.w; }
fn nw() -> u32 { return p.co * p.ci * p.k * p.k; }
fn weight(o: u32, c: u32, ky: u32, kx: u32) -> f32 {
  return data[p.ci * pixels() + ((o * p.ci + c) * p.k + ky) * p.k + kx];
}
fn sample(c: u32, y: u32, x: u32, ky: u32, kx: u32) -> f32 {
  let iy = u32(clamp(i32(y + ky) - i32(p.k / 2u), 0, i32(p.h) - 1));
  let ix = u32(clamp(i32(x + kx) - i32(p.k / 2u), 0, i32(p.w) - 1));
  return data[(c * p.h + iy) * p.w + ix];
}
fn index(id: vec3<u32>) -> u32 { return id.x + id.y * 16384u; }

@compute @workgroup_size(64)
fn forward(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = index(id);
  if i >= p.co * pixels() { return; }
  let o = i / pixels();
  let pos = i % pixels();
  var v = 0.0;
  if p.backward == 0u || p.relu != 0u {
    for (var c = 0u; c < p.ci; c++) {
      for (var ky = 0u; ky < p.k; ky++) {
        for (var kx = 0u; kx < p.k; kx++) {
          v += sample(c, pos / p.w, pos % p.w, ky, kx) * weight(o, c, ky, kx);
        }
      }
    }
    v += data[p.ci * pixels() + nw() + o];
    if p.residual != 0u { v += data[i]; }
  }
  if p.backward != 0u {
    let g = data[p.ci * pixels() + nw() + p.co + i];
    gated[i] = select(g, 0.0, p.relu != 0u && v <= 0.0);
  } else {
    gated[i] = select(v, max(v, 0.0), p.relu != 0u);
  }
}

@compute @workgroup_size(64)
fn inputGradient(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = index(id);
  if i >= p.ci * pixels() { return; }
  let c = i / pixels();
  let iy = i32((i % pixels()) / p.w);
  let ix = i32(i % p.w);
  var v = 0.0;
  if p.residual != 0u { v = gated[i]; }
  for (var o = 0u; o < p.co; o++) {
    for (var ky = 0u; ky < p.k; ky++) {
      // All preimages of replicate padding, including degenerate 1px axes.
      let y = iy + i32(p.k / 2u) - i32(ky);
      let loY = max(0, select(y, 0, iy == 0));
      let hiY = min(i32(p.h) - 1, select(y, i32(p.h) - 1, iy == i32(p.h) - 1));
      for (var kx = 0u; kx < p.k; kx++) {
        let x = ix + i32(p.k / 2u) - i32(kx);
        let loX = max(0, select(x, 0, ix == 0));
        let hiX = min(i32(p.w) - 1, select(x, i32(p.w) - 1, ix == i32(p.w) - 1));
        let w = weight(o, c, ky, kx);
        for (var yy = loY; yy <= hiY; yy++) {
          for (var xx = loX; xx <= hiX; xx++) {
            v += gated[(o * p.h + u32(yy)) * p.w + u32(xx)] * w;
          }
        }
      }
    }
  }
  result[i] = v;
}

@compute @workgroup_size(64)
fn parameterPartials(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = index(id);
  if i >= (nw() + p.co) * p.chunks { return; }
  let param = i / p.chunks;
  let chunk = i % p.chunks;
  let isBias = param >= nw();
  var o = param - nw();
  var c = 0u;
  var ky = 0u;
  var kx = 0u;
  if !isBias {
    o = param / (p.ci * p.k * p.k);
    c = (param / (p.k * p.k)) % p.ci;
    ky = (param / p.k) % p.k;
    kx = param % p.k;
  }
  var v = 0.0;
  for (var pos = chunk * 256u; pos < min((chunk + 1u) * 256u, pixels()); pos++) {
    var s = 1.0;
    if !isBias { s = sample(c, pos / p.w, pos % p.w, ky, kx); }
    v += gated[o * pixels() + pos] * s;
  }
  partial[i] = v;
}

@compute @workgroup_size(64)
fn parameterReduce(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = index(id);
  if i >= nw() + p.co { return; }
  var v = 0.0;
  for (var chunk = 0u; chunk < p.chunks; chunk++) {
    v += partial[i * p.chunks + chunk];
  }
  result[p.ci * pixels() + i] = v;
}
