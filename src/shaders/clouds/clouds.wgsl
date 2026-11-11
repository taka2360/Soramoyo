// 雲のレイマーチング（半解像度）。出力：rgb = 雲と、雲までの空気の散乱光（プリエクスポージャー済み）、a = 雲の透過率
// 合成は sky.wgsl：最終 = 空 × a + rgb
// （atmosphere/functions.wgsl + common.wgsl + cloudcommon.wgsl の後ろに連結）

@group(0) @binding(9) var<uniform> F: Frame;
@group(0) @binding(10) var<storage, read> expo: array<f32, 4>;
@group(0) @binding(11) var vol: texture_3d<f32>;
@group(0) @binding(12) var shadowVol: texture_3d<f32>;
@group(0) @binding(13) var zmap: texture_2d<f32>;
@group(0) @binding(14) var noiseTex: texture_3d<f32>;
@group(0) @binding(15) var<uniform> C: CloudU;
@group(0) @binding(16) var smpRep: sampler;
@group(0) @binding(17) var history: texture_2d<f32>;

const EARTH_R_M: f32 = 6360000.0;

// ---------------------------------------------------------------- 位相関数
// Jendersie & d'Eon 2023 "An Approximate Mie Scattering Function for Fog and Cloud Rendering"
// 雲粒の直径 d [µm]（5〜50）に対する HG + Draine の混合
fn hg(g: f32, c: f32) -> f32 {
  return (1.0 - g * g) / (4.0 * PI * pow(max(1.0 + g * g - 2.0 * g * c, 1e-6), 1.5));
}
fn draine(g: f32, a: f32, c: f32) -> f32 {
  return hg(g, c) * (1.0 + a * c * c) / (1.0 + a * (1.0 + 2.0 * g * g) / 3.0);
}
fn cloudPhase(c: f32, d: f32) -> f32 {
  let gHG = exp(-0.0990567 / (d - 1.67154));
  let gD = exp(-2.20679 / (d + 3.91029) - 0.428934);
  let a = exp(3.62489 - 8.29288 / (d + 5.52825));
  let wD = exp(-0.599085 / (d - 0.641583) - 0.665888);
  return (1.0 - wD) * hg(gHG, c) + wD * draine(gD, a, c);
}

// ---------------------------------------------------------------- 幾何
// 地球中心 (0,0,-R) の球殻との交差。戻り値 (t0, t1)、交差なしは (1e30, -1e30)
fn sphere(o: vec3f, d: vec3f, rad: f32) -> vec2f {
  let oc = o - vec3f(0.0, 0.0, -EARTH_R_M);
  let b = dot(oc, d);
  let c = dot(oc, oc) - rad * rad;
  let h = b * b - c;
  if (h < 0.0) { return vec2f(1e30, -1e30); }
  let s = sqrt(h);
  return vec2f(-b - s, -b + s);
}

fn altitude(p: vec3f) -> f32 { return length(p - vec3f(0.0, 0.0, -EARTH_R_M)) - EARTH_R_M; }

// ---------------------------------------------------------------- 密度
struct Medium { sigma: f32, odSun: f32, odUp: f32 };

fn domainUvw(p: vec3f, h: f32) -> vec3f {
  return vec3f((p.x - C.origin.x) / C.size.x, (p.y - C.origin.y) / C.size.y, zToW(zmap, h, C.ztop, C.nz));
}

// 太陽方向の光学的厚さ：近い範囲（格子2つ分）は細かく辿り、その先は前計算した影テクスチャ。
// 影テクスチャだけだと、粗い格子の補間で雲の縁に内部の大きな値が混ざり、日の当たる面が暗くなるため。
fn sunOpticalDepth(p: vec3f, sun: vec3f) -> f32 {
  let reach = 2.0 * C.cellSize;
  var od = 0.0;
  var t = 0.0;
  for (var i = 0; i < 6; i++) {
    let ds = reach * (f32(i) + 1.0) / 21.0; // 合計 = reach
    let q = p + sun * (t + 0.5 * ds);
    od += sampleMedium(q, altitude(q), false).sigma * ds;
    t += ds;
  }
  let q = p + sun * reach;
  let hq = altitude(q);
  if (hq > C.slabTop) { return od; }
  return od + textureSampleLevel(shadowVol, smpRep, domainUvw(q, hq), 0.0).r;
}

fn sampleMedium(p: vec3f, h: f32, withShadow: bool) -> Medium {
  var m: Medium;
  let lx = p.x - C.origin.x;
  let ly = p.y - C.origin.y;
  let uvw = vec3f(lx / C.size.x, ly / C.size.y, zToW(zmap, h, C.ztop, C.nz));
  let v = textureSampleLevel(vol, smpRep, uvw, 0.0);
  var lwc = v.r; // g/m³
  // ---- 細部合成（試作）：格子より細かい縁の凹凸。強さは渦粘性（乱流の強さ）で変える
  if (C.detail > 0.0 && lwc > 0.0) {
    let q = vec3f(lx, ly, h) / C.noisePeriod + vec3f(0.0, 0.0, C.time * 0.0005);
    let n = textureSampleLevel(noiseTex, smpRep, q, 0.0);
    let n2 = textureSampleLevel(noiseTex, smpRep, q * 3.7, 0.0);
    let turb = clamp(v.b / 20.0, 0.0, 1.0);
    let shape = n.r * 0.6 + n.g * 0.4;
    let fine = n2.g * 0.5 + n2.b * 0.5;
    let erosion = C.detail * (0.15 + 0.25 * turb) * (1.0 - mix(shape, fine, 0.4));
    lwc = max(lwc - erosion, 0.0) * (1.0 + 0.3 * C.detail * (shape - 0.5));
  }
  m.sigma = sigmaCloud(lwc * 1e-3, C.nDrop) + sigmaRain(v.g * 1e-3);
  if (withShadow && m.sigma > 0.0) {
    m.odUp = textureSampleLevel(shadowVol, smpRep, uvw, 0.0).g;
  }
  return m;
}

fn ignNoise(p: vec2f) -> f32 {
  return fract(52.9829189 * fract(dot(p, vec2f(0.06711056, 0.00583715))));
}

@fragment
fn fsMain(in: VsOut) -> @location(0) vec4f {
  let cur = march(in);
  // 時間方向の蓄積（簡易 TAA。カメラが動いたフレームは histAlpha=1 で履歴を捨てる）
  let prev = textureLoad(history, vec2i(in.pos.xy), 0);
  return mix(prev, cur, C.histAlpha);
}

fn march(in: VsOut) -> vec4f {
  if (C.enabled < 0.5) { return vec4f(0.0, 0.0, 0.0, 1.0); }
  let d = viewDir(F, in.uv);
  let camKm = F.camPos;
  let camH = (length(camKm) - A.bottom_radius) * 1000.0; // 地表（球）からの高さ [m]
  let o = vec3f(0.0, 0.0, camH);

  // 雲の層（球殻）との交差区間
  let rb = EARTH_R_M + C.slabBot;
  let rt = EARTH_R_M + C.slabTop;
  let st = sphere(o, d, rt);
  let sb = sphere(o, d, rb);
  let sg = sphere(o, d, EARTH_R_M);
  var t0: f32; var t1: f32;
  if (camH < C.slabBot) {
    t0 = sb.y; t1 = st.y;
  } else if (camH <= C.slabTop) {
    t0 = 0.0;
    t1 = st.y;
    if (sb.x > 0.0 && sb.x < t1) { t1 = sb.x; }
  } else {
    t0 = max(st.x, 0.0); t1 = st.y;
    if (sb.x > 0.0 && sb.x < t1) { t1 = sb.x; }
  }
  if (sg.x > 0.0) { t1 = min(t1, sg.x); } // 地面に当たる手前まで
  t1 = min(t1, t0 + C.maxDist);
  if (!(t1 > t0)) { return vec4f(0.0, 0.0, 0.0, 1.0); }

  let sun = F.sunDir;
  let cosT = dot(d, sun);
  let dropD = C.dropDiamUm;
  // 多重散乱のオクターブ近似（Wrenninge 2013）
  var phases: array<f32, 4>;
  let iso = 1.0 / (4.0 * PI);
  let p0 = cloudPhase(cosT, dropD);
  for (var i = 0; i < 4; i++) { phases[i] = mix(p0, iso, 1.0 - pow(0.5, f32(i))); }

  var T = 1.0;
  var L = vec3f(0.0);
  var depthW = 0.0;
  var wsum = 0.0;
  var t = t0;
  let jitter = ignNoise(in.pos.xy + C.frame * vec2f(5.588238, 3.1416));
  var ds = max(40.0, t * 0.015);
  t += ds * jitter;
  for (var i = 0; i < 220; i++) {
    if (t >= t1 || T < 0.01) { break; }
    let p = o + d * t;
    let h = altitude(p);
    var m = sampleMedium(p, h, true);
    ds = max(40.0, t * 0.015);
    if (m.sigma > 1e-7) {
      m.odSun = sunOpticalDepth(p, sun);
      // 点での太陽光（大気の透過込み）と空からの光
      let rKm = A.bottom_radius + h / 1000.0;
      let up = normalize(p - vec3f(0.0, 0.0, -EARTH_R_M));
      let muS = dot(up, sun);
      let sunE = F.sunIllum * getTransmittanceToSun(rKm, muS);
      let skyE = getIrradiance(rKm, muS);
      var S = vec3f(0.0);
      var a = 1.0; var b = 1.0;
      for (var oc = 0; oc < 4; oc++) {
        S += b * sunE * phases[oc] * exp(-a * m.odSun);
        a *= 0.5; b *= 0.5;
      }
      // 環境光：上半球は空、下半球は地面の照り返し（雲による遮りを上下の光学的厚さで近似）
      let groundE = (skyE + F.sunIllum * getTransmittanceToSun(A.bottom_radius, muS) * max(muS, 0.0)) * C.groundAlbedo;
      let amb = (skyE * exp(-0.25 * m.odUp) * 0.5 + groundE * 0.5 * 0.6) / PI;
      let src = S + amb;
      let Tseg = exp(-m.sigma * ds);
      // 区間内の積分（Hillaire 2016 のエネルギー保存形、散乱アルベド≈1）
      L += T * src * (1.0 - Tseg);
      let wgt = T * (1.0 - Tseg);
      depthW += t * wgt;
      wsum += wgt;
      T *= Tseg;
    } else {
      ds *= 1.6; // 空いている所は大きく進む
    }
    t += ds;
  }

  if (wsum <= 0.0) { return vec4f(0.0, 0.0, 0.0, 1.0); }
  // 空気遠近：雲の平均的な距離までの大気の散乱と透過
  let dist = depthW / wsum;
  let pKm = camKm + d * (dist / 1000.0);
  let ap = getSkyRadianceToPoint(camKm, pKm, sun);
  let col = ap.radiance * (1.0 - T) + ap.transmittance * L;
  let pre = 1.0 / max(expo[0], 1e-4);
  return vec4f(col * pre, T);
}
