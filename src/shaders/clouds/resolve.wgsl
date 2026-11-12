// 雲の時間方向の蓄積（TAA）：前フレームの結果を再投影して今フレームの1サンプルと混ぜる。
//  1. 今フレームの画素の「雲までの代表距離」から世界の点を求め、前フレームのカメラで見た画面位置へ投影
//  2. その位置の履歴を双線形で読み、今フレームの近傍 3×3 の平均 ± 1.25σ に収める（分散クリップ：ゴースト防止）
//  3. 履歴と今フレームを混合率 alpha で混ぜる
// 雲の無い画素（距離 0）は無限遠とみなし、方向だけで再投影する。
// （common.wgsl の後ろに連結）

struct ResolveU {
  fwd: vec3f, tanHalf: f32,
  right: vec3f, aspect: f32,
  up: vec3f, camH: f32,          // 今フレームのカメラ（camH: 地表からの高さ [m]）
  pfwd: vec3f, ptanHalf: f32,
  pright: vec3f, paspect: f32,
  pup: vec3f, pcamH: f32,        // 前フレームのカメラ
  alpha: f32, valid: f32, pad0: f32, pad1: f32,  // 混合率、履歴が使えるか（0 = 捨てる）
};

@group(0) @binding(0) var<uniform> R: ResolveU;
@group(0) @binding(1) var raw: texture_2d<f32>;
@group(0) @binding(2) var depthTex: texture_2d<f32>;
@group(0) @binding(3) var history: texture_2d<f32>;
@group(0) @binding(4) var smp: sampler;

fn finite4(v: vec4f) -> bool { return !(any(v != v) || any(abs(v) > vec4f(1e30))); }

@fragment
fn fsMain(in: VsOut) -> @location(0) vec4f {
  let size = vec2i(textureDimensions(raw));
  let ip = vec2i(in.pos.xy);
  let cur = textureLoad(raw, ip, 0);
  if (R.valid < 0.5) { return cur; }

  // 近傍の統計（分散クリップ用）
  var m1 = vec4f(0.0);
  var m2 = vec4f(0.0);
  for (var y = -1; y <= 1; y++) {
    for (var x = -1; x <= 1; x++) {
      let q = clamp(ip + vec2i(x, y), vec2i(0), size - 1);
      let s = textureLoad(raw, q, 0);
      m1 += s;
      m2 += s * s;
    }
  }
  m1 /= 9.0;
  m2 /= 9.0;
  let sigma = sqrt(max(m2 - m1 * m1, vec4f(0.0)));
  let lo = m1 - 1.25 * sigma;
  let hi = m1 + 1.25 * sigma;

  // 再投影
  let x = in.uv.x * R.tanHalf * R.aspect;
  let y = in.uv.y * R.tanHalf;
  let d = normalize(R.fwd + x * R.right + y * R.up);
  let t = textureLoad(depthTex, ip, 0).r;
  var dp = d;
  if (t > 0.0) {
    dp = normalize(vec3f(0.0, 0.0, R.camH) + d * t - vec3f(0.0, 0.0, R.pcamH));
  }
  let zf = dot(dp, R.pfwd);
  if (zf <= 1e-4) { return cur; }
  let px = dot(dp, R.pright) / zf / (R.ptanHalf * R.paspect);
  let py = dot(dp, R.pup) / zf / R.ptanHalf;
  if (abs(px) > 1.0 || abs(py) > 1.0) { return cur; }
  let uv = vec2f(0.5 + 0.5 * px, 0.5 - 0.5 * py);
  var prev = textureSampleLevel(history, smp, uv, 0.0);
  if (!finite4(prev)) { return cur; }
  prev = clamp(prev, lo, hi);
  return mix(prev, cur, R.alpha);
}
