// 大気散乱 LUT の事前計算パス（functions.wgsl の後ろに連結して使う）。
// __FMT__ は TS 側で rgba32float / rgba16float に置換される。

@group(0) @binding(8) var transmittanceOut: texture_storage_2d<__FMT__, write>;
@group(0) @binding(9) var deltaIrradianceOut: texture_storage_2d<__FMT__, write>;
@group(0) @binding(10) var deltaRayleighOut: texture_storage_3d<__FMT__, write>;
@group(0) @binding(11) var deltaMieOut: texture_storage_3d<__FMT__, write>;
@group(0) @binding(12) var<storage, read_write> scatterAcc: array<vec4f>;
@group(0) @binding(13) var<storage, read_write> mieAcc: array<vec4f>;
@group(0) @binding(14) var densityOut: texture_storage_3d<__FMT__, write>;
@group(0) @binding(15) var<storage, read_write> irradianceAcc: array<vec4f>;
@group(0) @binding(16) var multipleOut: texture_storage_3d<__FMT__, write>;
@group(0) @binding(17) var finalScatterOut: texture_storage_3d<__FMT__, write>;
@group(0) @binding(18) var finalMieOut: texture_storage_3d<__FMT__, write>;
@group(0) @binding(19) var finalIrradianceOut: texture_storage_2d<__FMT__, write>;
@group(0) @binding(20) var<storage, read> scatterAccR: array<vec4f>;
@group(0) @binding(21) var<storage, read> mieAccR: array<vec4f>;
@group(0) @binding(22) var<storage, read> irradianceAccR: array<vec4f>;

const SCAT_W: u32 = 256u; // NU * MU_S
const SCAT_H: u32 = 128u;
const SCAT_D: u32 = 32u;

fn toLuminance(v: vec3f) -> vec3f {
  return vec3f(dot(A.lum0.xyz, v), dot(A.lum1.xyz, v), dot(A.lum2.xyz, v));
}
fn scatIndex(p: vec3u) -> u32 { return p.x + SCAT_W * (p.y + SCAT_H * p.z); }

// ---------------------------------------------------------------- 透過率
fn opticalLengthToTop(r: f32, mu: f32, which: i32) -> f32 {
  let N = 500;
  let dx = distanceToTopAtmosphereBoundary(r, mu) / f32(N);
  var sum = 0.0;
  for (var i = 0; i <= N; i++) {
    let di = f32(i) * dx;
    let ri = sqrt(di * di + 2.0 * r * mu * di + r * r);
    let h = ri - A.bottom_radius;
    var y: f32;
    if (which == 0) { y = rayleighDensity(h); } else if (which == 1) { y = mieDensity(h); } else { y = ozoneDensity(h); }
    let w = select(1.0, 0.5, i == 0 || i == N);
    sum += y * w * dx;
  }
  return sum;
}

@compute @workgroup_size(8, 8)
fn csTransmittance(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= u32(TRANSMITTANCE_W) || id.y >= u32(TRANSMITTANCE_H)) { return; }
  let uv = (vec2f(id.xy) + 0.5) / vec2f(TRANSMITTANCE_W, TRANSMITTANCE_H);
  let rm = rMuFromTransmittanceUv(uv);
  let tau = A.rayleigh_scattering * opticalLengthToTop(rm.x, rm.y, 0) +
            A.mie_extinction * opticalLengthToTop(rm.x, rm.y, 1) +
            A.absorption_extinction * opticalLengthToTop(rm.x, rm.y, 2);
  textureStore(transmittanceOut, id.xy, vec4f(exp(-tau), 1.0));
}

// ---------------------------------------------------------------- 直達照度（地表の多重散乱計算用。最終照度には含めない）
@compute @workgroup_size(8, 8)
fn csDirectIrradiance(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= u32(IRRADIANCE_W) || id.y >= u32(IRRADIANCE_H)) { return; }
  let uv = (vec2f(id.xy) + 0.5) / vec2f(IRRADIANCE_W, IRRADIANCE_H);
  let rm = rMuSFromIrradianceUv(uv);
  let r = rm.x; let muS = rm.y;
  let a = A.sun_angular_radius;
  var f = 0.0;
  if (muS > a) { f = muS; } else if (muS > -a) { f = (muS + a) * (muS + a) / (4.0 * a); }
  let e = A.solar_irradiance * getTransmittanceToTop(r, muS) * f;
  textureStore(deltaIrradianceOut, id.xy, vec4f(e, 1.0));
}

// ---------------------------------------------------------------- 単散乱
@compute @workgroup_size(4, 4, 4)
fn csSingleScattering(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= SCAT_W || id.y >= SCAT_H || id.z >= SCAT_D) { return; }
  let p = rMuMuSNuFromScatteringFragCoord(vec3f(id) + 0.5);
  let N = 50;
  let dx = distanceToNearestAtmosphereBoundary(p.r, p.mu, p.hitGround) / f32(N);
  var sumR = vec3f(0.0);
  var sumM = vec3f(0.0);
  for (var i = 0; i <= N; i++) {
    let d = f32(i) * dx;
    let rD = clampRadius(sqrt(d * d + 2.0 * p.r * p.mu * d + p.r * p.r));
    let muSD = clampCosine((p.r * p.muS + d * p.nu) / rD);
    let t = getTransmittance(p.r, p.mu, d, p.hitGround) * getTransmittanceToSun(rD, muSD);
    let w = select(1.0, 0.5, i == 0 || i == N);
    sumR += t * rayleighDensity(rD - A.bottom_radius) * w;
    sumM += t * mieDensity(rD - A.bottom_radius) * w;
  }
  let ray = sumR * dx * A.solar_irradiance * A.rayleigh_scattering;
  let mie = sumM * dx * A.solar_irradiance * A.mie_scattering;
  textureStore(deltaRayleighOut, id, vec4f(ray, 1.0));
  textureStore(deltaMieOut, id, vec4f(mie, 1.0));
  let i = scatIndex(id);
  scatterAcc[i] += vec4f(toLuminance(ray), 0.0);
  mieAcc[i] += vec4f(toLuminance(mie), 0.0);
}

// ---------------------------------------------------------------- 散乱密度（order 次の散乱源）
fn scatteringForOrder(r: f32, mu: f32, muS: f32, nu: f32, hitGround: bool, order: i32) -> vec3f {
  if (order == 1) {
    let ray = sampleScattering(rayleighTex, r, mu, muS, nu, hitGround);
    let mie = sampleScattering(mieTex, r, mu, muS, nu, hitGround);
    return ray * rayleighPhase(nu) + mie * miePhase(A.mie_g, nu);
  }
  return sampleScattering(multipleTex, r, mu, muS, nu, hitGround);
}

@compute @workgroup_size(4, 4, 4)
fn csScatteringDensity(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= SCAT_W || id.y >= SCAT_H || id.z >= SCAT_D) { return; }
  let p = rMuMuSNuFromScatteringFragCoord(vec3f(id) + 0.5);
  let order = i32(A.scattering_order);
  let r = p.r; let mu = p.mu; let muS = p.muS; let nu = p.nu;

  let zenith = vec3f(0.0, 0.0, 1.0);
  let omega = vec3f(sqrt(1.0 - mu * mu), 0.0, mu);
  var sunX = 0.0;
  if (omega.x != 0.0) { sunX = (nu - mu * muS) / omega.x; }
  let sunY = sqrt(max(1.0 - sunX * sunX - muS * muS, 0.0));
  let omegaS = vec3f(sunX, sunY, muS);

  let N = 16;
  let dphi = PI_A / f32(N);
  let dtheta = PI_A / f32(N);
  var acc = vec3f(0.0);
  let rhoR = rayleighDensity(r - A.bottom_radius);
  let rhoM = mieDensity(r - A.bottom_radius);

  for (var l = 0; l < N; l++) {
    let theta = (f32(l) + 0.5) * dtheta;
    let cT = cos(theta); let sT = sin(theta);
    let hitG = rayIntersectsGround(r, cT);
    var distG = 0.0;
    var transG = vec3f(0.0);
    var albedo = 0.0;
    if (hitG) {
      distG = distanceToBottomAtmosphereBoundary(r, cT);
      transG = getTransmittance(r, cT, distG, true);
      albedo = A.ground_albedo;
    }
    for (var m = 0; m < 2 * N; m++) {
      let phi = (f32(m) + 0.5) * dphi;
      let wi = vec3f(cos(phi) * sT, sin(phi) * sT, cT);
      let dw = dtheta * dphi * sT;
      let nu1 = dot(omegaS, wi);
      var incident = scatteringForOrder(r, wi.z, muS, nu1, hitG, order - 1);
      let gN = normalize(zenith * r + wi * distG);
      let gE = getIrradiance(A.bottom_radius, dot(gN, omegaS));
      incident += transG * albedo * (1.0 / PI_A) * gE;
      let nu2 = dot(omega, wi);
      acc += incident * (A.rayleigh_scattering * rhoR * rayleighPhase(nu2) +
                         A.mie_scattering * rhoM * miePhase(A.mie_g, nu2)) * dw;
    }
  }
  textureStore(densityOut, id, vec4f(acc, 1.0));
}

// ---------------------------------------------------------------- 間接照度（order 次）
@compute @workgroup_size(8, 8)
fn csIndirectIrradiance(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= u32(IRRADIANCE_W) || id.y >= u32(IRRADIANCE_H)) { return; }
  let uv = (vec2f(id.xy) + 0.5) / vec2f(IRRADIANCE_W, IRRADIANCE_H);
  let rm = rMuSFromIrradianceUv(uv);
  let r = rm.x; let muS = rm.y;
  let order = i32(A.scattering_order);
  let N = 32;
  let dphi = PI_A / f32(N);
  let dtheta = PI_A / f32(N);
  let omegaS = vec3f(sqrt(1.0 - muS * muS), 0.0, muS);
  var e = vec3f(0.0);
  for (var j = 0; j < N / 2; j++) {
    let theta = (f32(j) + 0.5) * dtheta;
    for (var i = 0; i < 2 * N; i++) {
      let phi = (f32(i) + 0.5) * dphi;
      let w = vec3f(cos(phi) * sin(theta), sin(phi) * sin(theta), cos(theta));
      let dw = dtheta * dphi * sin(theta);
      let nu = dot(w, omegaS);
      e += scatteringForOrder(r, w.z, muS, nu, false, order) * w.z * dw;
    }
  }
  textureStore(deltaIrradianceOut, id.xy, vec4f(e, 1.0));
  let k = id.x + id.y * u32(IRRADIANCE_W);
  irradianceAcc[k] += vec4f(toLuminance(e), 0.0);
}

// ---------------------------------------------------------------- 多重散乱
@compute @workgroup_size(4, 4, 4)
fn csMultipleScattering(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= SCAT_W || id.y >= SCAT_H || id.z >= SCAT_D) { return; }
  let p = rMuMuSNuFromScatteringFragCoord(vec3f(id) + 0.5);
  let N = 50;
  let dx = distanceToNearestAtmosphereBoundary(p.r, p.mu, p.hitGround) / f32(N);
  var sum = vec3f(0.0);
  for (var i = 0; i <= N; i++) {
    let d = f32(i) * dx;
    let rI = clampRadius(sqrt(d * d + 2.0 * p.r * p.mu * d + p.r * p.r));
    let muI = clampCosine((p.r * p.mu + d) / rI);
    let muSI = clampCosine((p.r * p.muS + d * p.nu) / rI);
    let s = sampleScattering(densityTex, rI, muI, muSI, p.nu, p.hitGround) *
            getTransmittance(p.r, p.mu, d, p.hitGround) * dx;
    sum += s * select(1.0, 0.5, i == 0 || i == N);
  }
  textureStore(multipleOut, id, vec4f(sum, 1.0));
  scatterAcc[scatIndex(id)] += vec4f(toLuminance(sum) / rayleighPhase(p.nu), 0.0);
}

// ---------------------------------------------------------------- 累積バッファ → 最終テクスチャ
@compute @workgroup_size(4, 4, 4)
fn csFinalizeScattering(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= SCAT_W || id.y >= SCAT_H || id.z >= SCAT_D) { return; }
  let i = scatIndex(id);
  textureStore(finalScatterOut, id, vec4f(scatterAccR[i].rgb, 1.0));
  textureStore(finalMieOut, id, vec4f(mieAccR[i].rgb, 1.0));
}

@compute @workgroup_size(8, 8)
fn csFinalizeIrradiance(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= u32(IRRADIANCE_W) || id.y >= u32(IRRADIANCE_H)) { return; }
  textureStore(finalIrradianceOut, id.xy, vec4f(irradianceAccR[id.x + id.y * u32(IRRADIANCE_W)].rgb, 1.0));
}
