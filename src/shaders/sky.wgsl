// 空と地面の描画（atmosphere/functions.wgsl + common.wgsl の後ろに連結）。
// 出力は輝度 [cd/m²]（線形 sRGB）。

@group(0) @binding(9) var<uniform> F: Frame;
// [0] = 前フレームまでの順応輝度。HDR バッファ（半精度, 上限 65504）に収めるためのプリエクスポージャーに使う
@group(0) @binding(10) var<storage, read> expo: array<f32, 4>;

fn preExposure() -> f32 { return 1.0 / max(expo[0], 1e-4); }

// 雲（clouds.wgsl の出力：rgb = 散乱光（プリエクスポージャー済み）, a = 透過率）と地面の雲の影
@group(0) @binding(11) var cloudTex: texture_2d<f32>;
@group(0) @binding(12) var shadowVol: texture_3d<f32>;
@group(0) @binding(13) var zmap: texture_2d<f32>;
@group(0) @binding(15) var<uniform> C: CloudU;
@group(0) @binding(16) var smpRep: sampler;

// 地点 p（地球中心基準 km）の地表での、雲による太陽光の透過率
fn cloudShadowAtGround(p: vec3f) -> f32 {
  if (C.enabled < 0.5) { return 1.0; }
  let lx = p.x * 1000.0 - C.origin.x;
  let ly = p.y * 1000.0 - C.origin.y;
  let uvw = vec3f(lx / C.size.x, ly / C.size.y, zToW(zmap, 1.0, C.ztop, C.nz));
  return exp(-textureSampleLevel(shadowVol, smpRep, uvw, 0.0).r);
}

// 半精度の上限を超えないよう、色相を保ったまま上限で頭打ちにする
fn toHdr(L: vec3f) -> vec3f {
  let v = L * preExposure();
  let m = max(v.r, max(v.g, v.b));
  return select(v, v * (60000.0 / m), m > 60000.0);
}

// 地表の照度（直達・天空）[lx]
fn sunAndSkyIrradiance(p: vec3f, n: vec3f, sun: vec3f) -> array<vec3f, 2> {
  let r = length(p);
  let muS = dot(p, sun) / r;
  let sky = getIrradiance(r, muS) * (1.0 + dot(n, p) / r) * 0.5;
  let direct = F.sunIllum * getTransmittanceToSun(r, muS) * max(dot(n, sun), 0.0) * cloudShadowAtGround(p);
  return array<vec3f, 2>(direct, sky);
}

// ---------------------------------------------------------------- 大気差（見かけ → 真）
fn apparentToTrue(d: vec3f) -> vec3f {
  let h = asin(clamp(d.z, -1.0, 1.0)) * 180.0 / PI;
  let hc = max(h, -1.9);
  let R = 1.0 / tan((hc + 7.31 / (hc + 4.4)) * PI / 180.0) / 60.0 * F.refractionScale; // 度
  let ht = (h - max(R, 0.0)) * PI / 180.0;
  let horiz = length(d.xy);
  if (horiz < 1e-6) { return d; }
  return vec3f(d.xy * (cos(ht) / horiz), sin(ht));
}

// 太陽面の周辺減光（Hestroffer & Magnan 1998 の冪乗則, 波長ごと）。円盤平均が 1 になるよう正規化
fn limbDarkening(cosTheta: f32) -> vec3f {
  let alpha = vec3f(0.397, 0.503, 0.652);
  let mu = max(cosTheta, 0.0);
  return pow(vec3f(mu), alpha) * (alpha + 2.0) * 0.5;
}

@fragment
fn fsMain(in: VsOut) -> @location(0) vec4f {
  let view = viewDir(F, in.uv);
  let cam = F.camPos;
  let sun = F.sunDir;
  let r = length(cam);
  let rmu = dot(cam, view);

  // 地面（地球の球面）との交差
  let disc = rmu * rmu - r * r + A.bottom_radius * A.bottom_radius;
  var tGround = -1.0;
  if (disc >= 0.0) { tGround = -rmu - sqrt(disc); }

  // 太陽面判定用の角度（fwidth は一様な制御フローで呼ぶ必要があるため分岐の外で計算）
  let dTrue = apparentToTrue(view);
  let ang = 2.0 * asin(clamp(length(dTrue - sun) * 0.5, 0.0, 1.0)); // 小角度でも精度が落ちない形
  let angPx = fwidth(ang) + 1e-7;

  var L: vec3f;
  if (tGround > 0.0) {
    let p = cam + view * tGround;
    let n = normalize(p);
    let e = sunAndSkyIrradiance(p, n, sun);
    let groundL = F.groundAlbedo / PI * (e[0] + e[1]);
    let a = getSkyRadianceToPoint(cam, p, sun);
    L = groundL * a.transmittance + a.radiance;
  } else {
    let s = getSkyRadiance(cam, view, sun);
    L = s.radiance;

    // 太陽面：見かけの視線を真の方向に戻して比べる → 地平線近くで自然に扁平になる
    let rad = F.sunAngRadius;
    if (ang < rad * 1.5) {
      let cover = clamp((rad - ang) / angPx + 0.5, 0.0, 1.0);
      let x = clamp(ang / rad, 0.0, 1.0);
      let cosTheta = sqrt(1.0 - x * x);
      let sunL = F.sunIllum / (PI * rad * rad) * limbDarkening(cosTheta);
      L += sunL * s.transmittance * cover;
    }
  }

  if (F.gridOn > 0.5) {
    let alt = asin(clamp(view.z, -1.0, 1.0)) * 180.0 / PI;
    let az = atan2(view.x, view.y) * 180.0 / PI;
    let g = gridMask(alt, az);
    // 目盛りは画面の明るさに合わせて見える強さで重ねる（露出に追従させるため相対値）
    L = mix(L, vec3f(luminance(L) * 2.5 + 1e-4), g * 0.35);
  }
  // 雲を合成（雲テクスチャは半解像度。双線形で拡大）
  let cuv = in.pos.xy / F.resolution;
  let cl = textureSampleLevel(cloudTex, smp, cuv, 0.0);
  var outc = toHdr(L) * cl.a + cl.rgb;
  let m = max(outc.r, max(outc.g, outc.b));
  outc = select(outc, outc * (60000.0 / m), m > 60000.0);
  return vec4f(outc, 1.0);
}

fn gridLine(v: f32, spacing: f32, widthPx: f32) -> f32 {
  let d = abs(fract(v / spacing + 0.5) - 0.5) * spacing;
  let w = fwidth(v) * widthPx;
  return 1.0 - smoothstep(0.0, w, d);
}
fn gridMask(alt: f32, az: f32) -> f32 {
  let altLine = gridLine(alt, 10.0, 1.0);
  let azLine = gridLine(az, 10.0, 1.0) * smoothstep(88.0, 75.0, abs(alt));
  let horizon = gridLine(alt, 360.0, 1.5);
  return max(max(altLine, azLine) * 0.6, horizon);
}
