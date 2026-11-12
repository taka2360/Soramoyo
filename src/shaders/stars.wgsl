// 恒星の描画（1星 = 四角形1枚のインスタンス）。HDR に加算合成する。
// 明るさ：V 等級 → 照度 E = 2.54×10⁻⁶ × 10^(−0.4 V) [lx]
// 大気による減光（視線方向の透過率）、雲による遮り（雲テクスチャの透過率）、瞬き（大気のゆらぎ）を含む。
// 星は点光源なので、目の分解能（約 1′）または画素の大きさに広げたガウス分布で描く。
// （atmosphere/functions.wgsl + common.wgsl の後ろに連結）

struct Star { pos: vec3f, vmag: f32, color: vec3f, rnd: f32 };

@group(0) @binding(9) var<uniform> F: Frame;
@group(0) @binding(10) var<storage, read> expo: array<f32, 4>;
@group(0) @binding(11) var cloudTex: texture_2d<f32>;
@group(0) @binding(20) var<storage, read> stars: array<Star>;

struct SOut {
  @builtin(position) pos: vec4f,
  @location(0) local: vec2f,       // 星の中心からの位置（px）
  @location(1) flux: vec3f,        // 画素の輝度に掛ける係数
  @location(2) sigma: f32,
  @location(3) uv: vec2f,
};

fn refractTrueToApparent(v: vec3f) -> vec3f {
  let h = asin(clamp(v.z, -1.0, 1.0)) * 180.0 / PI;
  let hc = max(h, -2.0);
  let R = 1.02 / tan((hc + 10.3 / (hc + 5.11)) * PI / 180.0) / 60.0 * F.refractionScale;
  let ha = (h + max(R, 0.0)) * PI / 180.0;
  let horiz = length(v.xy);
  if (horiz < 1e-6) { return v; }
  return vec3f(v.xy * (cos(ha) / horiz), sin(ha));
}

@vertex
fn vsStar(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> SOut {
  var o: SOut;
  o.pos = vec4f(2.0, 2.0, 0.0, 1.0); // 既定は画面外（描かない）
  let s = stars[ii];
  // J2000 → ENU（行列の各行との内積）→ 大気差で見かけの方向へ
  let v = refractTrueToApparent(normalize(vec3f(dot(F.eqX, s.pos), dot(F.eqY, s.pos), dot(F.eqZ, s.pos))));
  if (v.z < -0.01) { return o; }
  let z = dot(v, F.fwd);
  if (z <= 0.0) { return o; }
  let ndc = vec2f(dot(v, F.right) / (z * F.tanHalfFov * F.aspect), dot(v, F.up) / (z * F.tanHalfFov));
  if (any(abs(ndc) > vec2f(1.05))) { return o; }

  // 画素の立体角と、星像の広がり。ズームは「双眼鏡で拡大して見る」とみなし、星像は画面上でほぼ一定（0.8 画素）。
  // 強く拡大したときは大気のゆらぎ（シーイング ~2″）で広がる
  let pxAngle = 2.0 * F.tanHalfFov / F.resolution.y;
  let sigma = max(0.8, (2.0 / 3600.0 * PI / 180.0) / pxAngle);
  let omegaPx = pxAngle * pxAngle;

  // 大気の透過率（視線方向）
  let r = length(F.camPos);
  let T = getTransmittanceToTop(r, v.z);
  // 瞬き：大気のゆらぎ。地平線に近い（空気の層が厚い）ほど強い
  let air = 1.0 / max(v.z + 0.15 * pow(v.z + 0.0885, -1.253), 0.03);
  let amp = clamp(0.06 * air, 0.0, 0.6);
  let tw = 1.0 + amp * (sin(F.time * 9.1 + s.rnd * 50.0) * 0.6 + sin(F.time * 23.7 + s.rnd * 113.0) * 0.4);
  let E = 2.54e-6 * pow(10.0, -0.4 * s.vmag) * tw;
  let pre = 1.0 / max(expo[0], 1e-4);
  let flux = s.color * T * E / (2.0 * PI * sigma * sigma * omegaPx) * pre;
  // 画面で見えないほど暗い星は描かない（塗りつぶしの節約。値はプリエクスポージャー後）
  if (luminance(flux) < 1e-4) { return o; }

  // 三角形2枚：角の番号 0,1,2 / 2,1,3
  let corners = array<u32, 6>(0u, 1u, 2u, 2u, 1u, 3u);
  let idx = corners[vi % 6u];
  let c = vec2f(f32(idx & 1u), f32((idx >> 1u) & 1u)) * 2.0 - 1.0;
  let half = 3.0 * sigma;
  let offs = c * half / (F.resolution * 0.5);
  o.pos = vec4f(ndc + offs, 0.0, 1.0);
  o.local = c * half;
  o.flux = flux;
  o.sigma = sigma;
  o.uv = (ndc * vec2f(0.5, -0.5) + 0.5);
  return o;
}

@fragment
fn fsStar(in: SOut) -> @location(0) vec4f {
  let r2 = dot(in.local, in.local);
  let g = exp(-r2 / (2.0 * in.sigma * in.sigma));
  let cloudT = textureSampleLevel(cloudTex, smp, in.uv, 0.0).a;
  return vec4f(in.flux * g * cloudT, 0.0);
}
