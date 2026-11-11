// 空と地面の描画（atmosphere/functions.wgsl + common.wgsl の後ろに連結）。
// 出力は輝度 [cd/m²]（線形 sRGB）。

@group(0) @binding(9) var<uniform> F: Frame;
// [0] = 前フレームまでの順応輝度。HDR バッファ（半精度, 上限 65504）に収めるためのプリエクスポージャーに使う
@group(0) @binding(10) var<storage, read> expo: array<f32, 4>;

fn preExposure() -> f32 { return 1.0 / max(expo[0], 1e-4); }

// 半精度の上限を超えないよう、色相を保ったまま上限で頭打ちにする
fn toHdr(L: vec3f) -> vec3f {
  let v = L * preExposure();
  let m = max(v.r, max(v.g, v.b));
  return select(v, v * (60000.0 / m), m > 60000.0);
}

// ---------------------------------------------------------------- Bruneton の描画関数
struct SkyOut { radiance: vec3f, transmittance: vec3f };

fn skyScattering(r: f32, mu: f32, muS: f32, nu: f32, hitGround: bool) -> vec3f {
  let ray = sampleScattering(rayleighTex, r, mu, muS, nu, hitGround);
  let mie = sampleScattering(mieTex, r, mu, muS, nu, hitGround);
  return ray * rayleighPhase(nu) + mie * miePhase(A.mie_g, nu);
}

fn getSkyRadiance(camIn: vec3f, view: vec3f, sun: vec3f) -> SkyOut {
  var o: SkyOut;
  var cam = camIn;
  var r = length(cam);
  var rmu = dot(cam, view);
  let distToTop = -rmu - sqrt(rmu * rmu - r * r + A.top_radius * A.top_radius);
  if (distToTop > 0.0) {
    cam = cam + view * distToTop;
    r = A.top_radius;
    rmu += distToTop;
  } else if (r > A.top_radius) {
    o.radiance = vec3f(0.0);
    o.transmittance = vec3f(1.0);
    return o;
  }
  let mu = rmu / r;
  let muS = dot(cam, sun) / r;
  let nu = dot(view, sun);
  let hitGround = rayIntersectsGround(r, mu);
  o.transmittance = select(getTransmittanceToTop(r, mu), vec3f(0.0), hitGround);
  o.radiance = skyScattering(r, mu, muS, nu, hitGround);
  return o;
}

fn getSkyRadianceToPoint(camIn: vec3f, point: vec3f, sun: vec3f) -> SkyOut {
  var o: SkyOut;
  let view = normalize(point - camIn);
  var cam = camIn;
  var r = length(cam);
  var rmu = dot(cam, view);
  let distToTop = -rmu - sqrt(rmu * rmu - r * r + A.top_radius * A.top_radius);
  if (distToTop > 0.0) {
    cam = cam + view * distToTop;
    r = A.top_radius;
    rmu += distToTop;
  }
  let mu = rmu / r;
  let muS = dot(cam, sun) / r;
  let nu = dot(view, sun);
  let d = length(point - cam);
  let hitGround = rayIntersectsGround(r, mu);
  o.transmittance = getTransmittance(r, mu, d, hitGround);

  let ray0 = sampleScattering(rayleighTex, r, mu, muS, nu, hitGround);
  let mie0 = sampleScattering(mieTex, r, mu, muS, nu, hitGround);
  let rP = clampRadius(sqrt(d * d + 2.0 * r * mu * d + r * r));
  let muP = (r * mu + d) / rP;
  let muSP = (r * muS + d * nu) / rP;
  let ray1 = sampleScattering(rayleighTex, rP, muP, muSP, nu, hitGround);
  let mie1 = sampleScattering(mieTex, rP, muP, muSP, nu, hitGround);
  let ray = max(ray0 - o.transmittance * ray1, vec3f(0.0));
  var mie = max(mie0 - o.transmittance * mie1, vec3f(0.0));
  mie *= smoothstep(0.0, 0.01, muS); // Bruneton と同じ：地平線下の太陽での Mie の誤差を抑える
  o.radiance = ray * rayleighPhase(nu) + mie * miePhase(A.mie_g, nu);
  return o;
}

// 地表の照度（直達・天空）[lx]
fn sunAndSkyIrradiance(p: vec3f, n: vec3f, sun: vec3f) -> array<vec3f, 2> {
  let r = length(p);
  let muS = dot(p, sun) / r;
  let sky = getIrradiance(r, muS) * (1.0 + dot(n, p) / r) * 0.5;
  let direct = F.sunIllum * getTransmittanceToSun(r, muS) * max(dot(n, sun), 0.0);
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
  return vec4f(toHdr(L), 1.0);
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
