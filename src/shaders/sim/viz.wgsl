// 断面図の描画（検証ページ用）。state バッファを直接読む。

struct V {
  nx: u32, ny: u32, nz: u32, n: u32,
  mode: u32, slice: u32, field: u32, cmap: u32,   // mode 0: x–z 断面(j=slice), 1: x–y 断面(k=slice)
  vmin: f32, vmax: f32, lx: f32, ly: f32,
  ztop: f32, contour: f32, pad0: f32, pad1: f32,
  rect: vec4f,                                     // 描画領域（px）: x0, y0, x1, y1
};

struct Col {
  zc: f32, zf: f32, dzc: f32, dzw: f32,
  rho0: f32, rhoF: f32, th0: f32, qv0: f32,
  pi0: f32, p0: f32, u0: f32, v0: f32,
  ug: f32, vg: f32, wsub: f32, thLs: f32,       // 地衡風、大規模沈降 [m/s]、θ の大規模・放射強制 [K/s]
  qvLs: f32, nudge: f32, pad0: f32, pad1: f32,  // qv の大規模強制 [1/s]、基本場へのナッジング率 [1/s]
  pad2: f32, pad3: f32, pad4: f32, pad5: f32,
};

@group(0) @binding(0) var<uniform> U: V;
@group(0) @binding(1) var<storage, read> st: array<f32>;
@group(0) @binding(2) var<storage, read> col: array<Col>;

struct VsOut { @builtin(position) pos: vec4f };

@vertex
fn vsMain(@builtin(vertex_index) i: u32) -> VsOut {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u)) * 2.0 - 1.0;
  var o: VsOut;
  o.pos = vec4f(p, 0.0, 1.0);
  return o;
}

fn wrapI(i: i32, n: u32) -> u32 { let m = i32(n); return u32(((i % m) + m) % m); }

// セル中心の値（W は中心へ平均、TH と QV は基本場からの偏差）
fn cellValue(f: u32, i: i32, j: i32, k: i32) -> f32 {
  let kk = u32(clamp(k, 0, i32(U.nz) - 1));
  let id = wrapI(i, U.nx) + U.nx * (wrapI(j, U.ny) + U.ny * kk);
  if (f == 2u) {
    let w0 = st[2u * U.n + id];
    var w1 = 0.0;
    if (kk + 1u < U.nz) { w1 = st[2u * U.n + id + U.nx * U.ny]; }
    return 0.5 * (w0 + w1);
  }
  if (f == 0u) { // u：x 面の値を中心へ
    let ip = wrapI(i + 1, U.nx) + U.nx * (wrapI(j, U.ny) + U.ny * kk);
    return 0.5 * (st[id] + st[ip]);
  }
  let v = st[f * U.n + id];
  if (f == 3u) { return v - col[kk].th0; }
  if (f == 4u) { return v - col[kk].qv0; }
  return v;
}

// 高さ z に対応する（小数の）層番号
fn kOf(z: f32) -> f32 {
  let nz = i32(U.nz);
  if (z <= col[0].zc) { return 0.0; }
  for (var k = 1; k < nz; k++) {
    if (z <= col[k].zc) {
      return f32(k - 1) + (z - col[k - 1].zc) / (col[k].zc - col[k - 1].zc);
    }
  }
  return f32(nz - 1);
}

fn sample(f: u32, x: f32, y: f32, z: f32) -> f32 {
  let fx = x / U.lx * f32(U.nx) - 0.5;
  let i0 = i32(floor(fx)); let tx = fx - floor(fx);
  if (U.mode == 0u) {
    let kf = kOf(z);
    let k0 = i32(floor(kf)); let tz = kf - floor(kf);
    let j = i32(U.slice);
    let a = mix(cellValue(f, i0, j, k0), cellValue(f, i0 + 1, j, k0), tx);
    let b = mix(cellValue(f, i0, j, k0 + 1), cellValue(f, i0 + 1, j, k0 + 1), tx);
    return mix(a, b, tz);
  }
  let fy = y / U.ly * f32(U.ny) - 0.5;
  let j0 = i32(floor(fy)); let ty = fy - floor(fy);
  let k = i32(U.slice);
  let a = mix(cellValue(f, i0, j0, k), cellValue(f, i0 + 1, j0, k), tx);
  let b = mix(cellValue(f, i0, j0 + 1, k), cellValue(f, i0 + 1, j0 + 1, k), tx);
  return mix(a, b, ty);
}

fn diverging(t: f32) -> vec3f {
  // 青 − 白 − 赤
  let c0 = vec3f(0.12, 0.25, 0.65);
  let c1 = vec3f(0.97, 0.97, 0.97);
  let c2 = vec3f(0.70, 0.12, 0.12);
  if (t < 0.5) { return mix(c0, c1, smoothstep(0.0, 1.0, t * 2.0)); }
  return mix(c1, c2, smoothstep(0.0, 1.0, t * 2.0 - 1.0));
}
fn sequential(t: f32) -> vec3f {
  // 濃紺 → 青 → 水色 → 白
  let a = vec3f(0.05, 0.07, 0.18);
  let b = vec3f(0.15, 0.35, 0.75);
  let c = vec3f(0.55, 0.80, 0.95);
  let d = vec3f(1.0, 1.0, 1.0);
  if (t < 0.33) { return mix(a, b, t / 0.33); }
  if (t < 0.66) { return mix(b, c, (t - 0.33) / 0.33); }
  return mix(c, d, (t - 0.66) / 0.34);
}

@fragment
fn fsMain(in: VsOut) -> @location(0) vec4f {
  let r = U.rect;
  let uv = (in.pos.xy - r.xy) / (r.zw - r.xy);
  let inside = all(uv >= vec2f(0.0)) && all(uv <= vec2f(1.0));
  let x = clamp(uv.x, 0.0, 1.0) * U.lx;
  var y = 0.0; var z = 0.0;
  if (U.mode == 0u) { z = (1.0 - clamp(uv.y, 0.0, 1.0)) * U.ztop; } else { y = (1.0 - clamp(uv.y, 0.0, 1.0)) * U.ly; }
  let v = sample(U.field, x, y, z);
  // 雲の輪郭（qc = 閾値の等値線）
  let qc = sample(5u, x, y, z);
  let e = qc - U.contour;
  let line = 1.0 - smoothstep(0.0, 1.2 * fwidth(e) + 1e-12, abs(e));
  if (!inside) { return vec4f(0.08, 0.08, 0.09, 1.0); }
  let t = clamp((v - U.vmin) / (U.vmax - U.vmin), 0.0, 1.0);
  var c = select(sequential(t), diverging(t), U.cmap == 0u);
  if (U.contour > 0.0) { c = mix(c, vec3f(0.0), line * 0.85); }
  return vec4f(c, 1.0);
}
