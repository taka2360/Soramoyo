// =============================================================================
// 大気散乱：Bruneton 2017 "Precomputed Atmospheric Scattering: a New Implementation"
// (https://github.com/ebruneton/precomputed_atmospheric_scattering, BSD ライセンス) の
// 関数群を WGSL に移植したもの。長さの単位は km。
//
// 事前計算は「3波長ずつ」行い、CIE 等色関数で輝度（線形 sRGB, cd/m²）に変換して足し合わせる。
// テクスチャに入る値の単位：
//   - 事前計算中の delta テクスチャ：その3波長の分光放射輝度
//   - 最終テクスチャ：輝度 [cd/m²]（scattering, single_mie）、照度 [lx]（irradiance）
// =============================================================================

const TRANSMITTANCE_W: f32 = 256.0;
const TRANSMITTANCE_H: f32 = 64.0;
const SCATTERING_R: f32 = 32.0;
const SCATTERING_MU: f32 = 128.0;
const SCATTERING_MU_S: f32 = 32.0;
const SCATTERING_NU: f32 = 8.0;
const IRRADIANCE_W: f32 = 64.0;
const IRRADIANCE_H: f32 = 16.0;
const PI_A: f32 = 3.14159265358979;

struct Atmo {
  solar_irradiance: vec3f, sun_angular_radius: f32,
  rayleigh_scattering: vec3f, bottom_radius: f32,
  mie_scattering: vec3f, top_radius: f32,
  mie_extinction: vec3f, mie_g: f32,
  absorption_extinction: vec3f, mu_s_min: f32,
  ground_albedo: f32, mie_scale_height: f32, scattering_order: f32, pad0: f32,
  // luminance_from_radiance（3x3, 行ごと）。delta(3波長) → 線形 sRGB 輝度
  lum0: vec4f, lum1: vec4f, lum2: vec4f,
};

@group(0) @binding(0) var<uniform> A: Atmo;
@group(0) @binding(1) var smp: sampler;
@group(0) @binding(2) var transmittanceTex: texture_2d<f32>;
@group(0) @binding(3) var irradianceTex: texture_2d<f32>;
@group(0) @binding(4) var rayleighTex: texture_3d<f32>;   // 単散乱 Rayleigh（最終: 多重散乱込み）
@group(0) @binding(5) var mieTex: texture_3d<f32>;        // 単散乱 Mie
@group(0) @binding(6) var multipleTex: texture_3d<f32>;   // 多重散乱（delta）
@group(0) @binding(7) var densityTex: texture_3d<f32>;    // 散乱密度（delta）

// ---------------------------------------------------------------- 基本
fn clampCosine(mu: f32) -> f32 { return clamp(mu, -1.0, 1.0); }
fn clampDistance(d: f32) -> f32 { return max(d, 0.0); }
fn clampRadius(r: f32) -> f32 { return clamp(r, A.bottom_radius, A.top_radius); }
fn safeSqrt(a: f32) -> f32 { return sqrt(max(a, 0.0)); }

fn distanceToTopAtmosphereBoundary(r: f32, mu: f32) -> f32 {
  let disc = r * r * (mu * mu - 1.0) + A.top_radius * A.top_radius;
  return clampDistance(-r * mu + safeSqrt(disc));
}
fn distanceToBottomAtmosphereBoundary(r: f32, mu: f32) -> f32 {
  let disc = r * r * (mu * mu - 1.0) + A.bottom_radius * A.bottom_radius;
  return clampDistance(-r * mu - safeSqrt(disc));
}
fn rayIntersectsGround(r: f32, mu: f32) -> bool {
  return mu < 0.0 && r * r * (mu * mu - 1.0) + A.bottom_radius * A.bottom_radius >= 0.0;
}
fn distanceToNearestAtmosphereBoundary(r: f32, mu: f32, hitGround: bool) -> f32 {
  if (hitGround) { return distanceToBottomAtmosphereBoundary(r, mu); }
  return distanceToTopAtmosphereBoundary(r, mu);
}

// ---------------------------------------------------------------- 密度分布
fn rayleighDensity(h: f32) -> f32 { return clamp(exp(-h / 8.0), 0.0, 1.0); }
fn mieDensity(h: f32) -> f32 { return clamp(exp(-h / A.mie_scale_height), 0.0, 1.0); }
fn ozoneDensity(h: f32) -> f32 {
  // 高度25kmを頂点とするテント型（Bruneton のデモと同じ）
  if (h < 25.0) { return clamp(h / 15.0 - 2.0 / 3.0, 0.0, 1.0); }
  return clamp(-h / 15.0 + 8.0 / 3.0, 0.0, 1.0);
}

// ---------------------------------------------------------------- 位相関数
fn rayleighPhase(nu: f32) -> f32 {
  return 3.0 / (16.0 * PI_A) * (1.0 + nu * nu);
}
fn miePhase(g: f32, nu: f32) -> f32 {
  // Cornette–Shanks
  let k = 3.0 / (8.0 * PI_A) * (1.0 - g * g) / (2.0 + g * g);
  return k * (1.0 + nu * nu) / pow(max(1.0 + g * g - 2.0 * g * nu, 1e-6), 1.5);
}

// ---------------------------------------------------------------- テクスチャ座標
fn texCoordFromUnit(x: f32, size: f32) -> f32 { return 0.5 / size + x * (1.0 - 1.0 / size); }
fn unitFromTexCoord(u: f32, size: f32) -> f32 { return (u - 0.5 / size) / (1.0 - 1.0 / size); }

// ---------------------------------------------------------------- 透過率
fn transmittanceUvFromRMu(r: f32, mu: f32) -> vec2f {
  let H = sqrt(A.top_radius * A.top_radius - A.bottom_radius * A.bottom_radius);
  let rho = safeSqrt(r * r - A.bottom_radius * A.bottom_radius);
  let d = distanceToTopAtmosphereBoundary(r, mu);
  let dMin = A.top_radius - r;
  let dMax = rho + H;
  let xMu = (d - dMin) / (dMax - dMin);
  let xR = rho / H;
  return vec2f(texCoordFromUnit(xMu, TRANSMITTANCE_W), texCoordFromUnit(xR, TRANSMITTANCE_H));
}

fn rMuFromTransmittanceUv(uv: vec2f) -> vec2f {
  let xMu = unitFromTexCoord(uv.x, TRANSMITTANCE_W);
  let xR = unitFromTexCoord(uv.y, TRANSMITTANCE_H);
  let H = sqrt(A.top_radius * A.top_radius - A.bottom_radius * A.bottom_radius);
  let rho = H * xR;
  let r = sqrt(rho * rho + A.bottom_radius * A.bottom_radius);
  let dMin = A.top_radius - r;
  let dMax = rho + H;
  let d = dMin + xMu * (dMax - dMin);
  var mu = 1.0;
  if (d != 0.0) { mu = (H * H - rho * rho - d * d) / (2.0 * r * d); }
  return vec2f(r, clampCosine(mu));
}

fn getTransmittanceToTop(r: f32, mu: f32) -> vec3f {
  return textureSampleLevel(transmittanceTex, smp, transmittanceUvFromRMu(r, mu), 0.0).rgb;
}

fn getTransmittance(r: f32, mu: f32, d: f32, hitGround: bool) -> vec3f {
  let rD = clampRadius(sqrt(d * d + 2.0 * r * mu * d + r * r));
  let muD = clampCosine((r * mu + d) / rD);
  if (hitGround) {
    return min(getTransmittanceToTop(rD, -muD) / getTransmittanceToTop(r, -mu), vec3f(1.0));
  }
  return min(getTransmittanceToTop(r, mu) / getTransmittanceToTop(rD, muD), vec3f(1.0));
}

fn getTransmittanceToSun(r: f32, muS: f32) -> vec3f {
  let sinH = A.bottom_radius / r;
  let cosH = -sqrt(max(1.0 - sinH * sinH, 0.0));
  return getTransmittanceToTop(r, muS) *
    smoothstep(-sinH * A.sun_angular_radius, sinH * A.sun_angular_radius, muS - cosH);
}

// ---------------------------------------------------------------- 散乱テクスチャ座標（4D → 3D）
fn scatteringUvwzFromRMuMuSNu(r: f32, mu: f32, muS: f32, nu: f32, hitGround: bool) -> vec4f {
  let H = sqrt(A.top_radius * A.top_radius - A.bottom_radius * A.bottom_radius);
  let rho = safeSqrt(r * r - A.bottom_radius * A.bottom_radius);
  let uR = texCoordFromUnit(rho / H, SCATTERING_R);

  let rMu = r * mu;
  let disc = rMu * rMu - r * r + A.bottom_radius * A.bottom_radius;
  var uMu: f32;
  if (hitGround) {
    let d = -rMu - safeSqrt(disc);
    let dMin = r - A.bottom_radius;
    let dMax = rho;
    var x = 0.0;
    if (dMax != dMin) { x = (d - dMin) / (dMax - dMin); }
    uMu = 0.5 - 0.5 * texCoordFromUnit(x, SCATTERING_MU / 2.0);
  } else {
    let d = -rMu + safeSqrt(disc + H * H);
    let dMin = A.top_radius - r;
    let dMax = rho + H;
    uMu = 0.5 + 0.5 * texCoordFromUnit((d - dMin) / (dMax - dMin), SCATTERING_MU / 2.0);
  }

  let d = distanceToTopAtmosphereBoundary(A.bottom_radius, muS);
  let dMin = A.top_radius - A.bottom_radius;
  let dMax = H;
  let a = (d - dMin) / (dMax - dMin);
  let D = distanceToTopAtmosphereBoundary(A.bottom_radius, A.mu_s_min);
  let AA = (D - dMin) / (dMax - dMin);
  let uMuS = texCoordFromUnit(max(1.0 - a / AA, 0.0) / (1.0 + a), SCATTERING_MU_S);
  let uNu = (nu + 1.0) / 2.0;
  return vec4f(uNu, uMuS, uMu, uR);
}

struct RMuMuSNu { r: f32, mu: f32, muS: f32, nu: f32, hitGround: bool };

fn rMuMuSNuFromScatteringUvwz(uvwz: vec4f) -> RMuMuSNu {
  var o: RMuMuSNu;
  let H = sqrt(A.top_radius * A.top_radius - A.bottom_radius * A.bottom_radius);
  let rho = H * unitFromTexCoord(uvwz.w, SCATTERING_R);
  o.r = sqrt(rho * rho + A.bottom_radius * A.bottom_radius);

  if (uvwz.z < 0.5) {
    let dMin = o.r - A.bottom_radius;
    let dMax = rho;
    let d = dMin + (dMax - dMin) * unitFromTexCoord(1.0 - 2.0 * uvwz.z, SCATTERING_MU / 2.0);
    o.mu = -1.0;
    if (d != 0.0) { o.mu = clampCosine(-(rho * rho + d * d) / (2.0 * o.r * d)); }
    o.hitGround = true;
  } else {
    let dMin = A.top_radius - o.r;
    let dMax = rho + H;
    let d = dMin + (dMax - dMin) * unitFromTexCoord(2.0 * uvwz.z - 1.0, SCATTERING_MU / 2.0);
    o.mu = 1.0;
    if (d != 0.0) { o.mu = clampCosine((H * H - rho * rho - d * d) / (2.0 * o.r * d)); }
    o.hitGround = false;
  }

  let xMuS = unitFromTexCoord(uvwz.y, SCATTERING_MU_S);
  let dMin = A.top_radius - A.bottom_radius;
  let dMax = H;
  let D = distanceToTopAtmosphereBoundary(A.bottom_radius, A.mu_s_min);
  let AA = (D - dMin) / (dMax - dMin);
  let a = (AA - xMuS * AA) / (1.0 + xMuS * AA);
  let d = dMin + min(a, AA) * (dMax - dMin);
  o.muS = 1.0;
  if (d != 0.0) { o.muS = clampCosine((H * H - d * d) / (2.0 * A.bottom_radius * d)); }
  o.nu = clampCosine(uvwz.x * 2.0 - 1.0);
  return o;
}

fn rMuMuSNuFromScatteringFragCoord(fc: vec3f) -> RMuMuSNu {
  let SIZE = vec4f(SCATTERING_NU - 1.0, SCATTERING_MU_S, SCATTERING_MU, SCATTERING_R);
  let fcNu = floor(fc.x / SCATTERING_MU_S);
  let fcMuS = fc.x - fcNu * SCATTERING_MU_S;
  let uvwz = vec4f(fcNu, fcMuS, fc.y, fc.z) / SIZE;
  var o = rMuMuSNuFromScatteringUvwz(uvwz);
  let s = sqrt((1.0 - o.mu * o.mu) * (1.0 - o.muS * o.muS));
  o.nu = clamp(o.nu, o.mu * o.muS - s, o.mu * o.muS + s);
  return o;
}

// nu 方向は手動で線形補間（3Dテクスチャの x に nu と mu_s を詰めているため）
fn sampleScattering(tex: texture_3d<f32>, r: f32, mu: f32, muS: f32, nu: f32, hitGround: bool) -> vec3f {
  let uvwz = scatteringUvwzFromRMuMuSNu(r, mu, muS, nu, hitGround);
  let tcx = uvwz.x * (SCATTERING_NU - 1.0);
  let tx = floor(tcx);
  let l = tcx - tx;
  let uvw0 = vec3f((tx + uvwz.y) / SCATTERING_NU, uvwz.z, uvwz.w);
  let uvw1 = vec3f((tx + 1.0 + uvwz.y) / SCATTERING_NU, uvwz.z, uvwz.w);
  return textureSampleLevel(tex, smp, uvw0, 0.0).rgb * (1.0 - l) +
         textureSampleLevel(tex, smp, uvw1, 0.0).rgb * l;
}

// ---------------------------------------------------------------- 照度テクスチャ座標
fn irradianceUvFromRMuS(r: f32, muS: f32) -> vec2f {
  let xR = (r - A.bottom_radius) / (A.top_radius - A.bottom_radius);
  let xMuS = muS * 0.5 + 0.5;
  return vec2f(texCoordFromUnit(xMuS, IRRADIANCE_W), texCoordFromUnit(xR, IRRADIANCE_H));
}
fn rMuSFromIrradianceUv(uv: vec2f) -> vec2f {
  let xMuS = unitFromTexCoord(uv.x, IRRADIANCE_W);
  let xR = unitFromTexCoord(uv.y, IRRADIANCE_H);
  return vec2f(A.bottom_radius + xR * (A.top_radius - A.bottom_radius), clampCosine(2.0 * xMuS - 1.0));
}
fn getIrradiance(r: f32, muS: f32) -> vec3f {
  return textureSampleLevel(irradianceTex, smp, irradianceUvFromRMuS(r, muS), 0.0).rgb;
}
