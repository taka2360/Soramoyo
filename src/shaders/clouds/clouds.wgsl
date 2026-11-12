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
struct Medium { sigma: f32, odSun: f32, odUp: f32, iceFrac: f32 };

// 氷晶の位相関数：強い前方散乱（HG, g=0.75）＋ 22° ハロ（六角柱の氷晶の屈折）。
// ハロは全散乱の 3% を 22° ± 0.6° のガウス形に割り当てる簡易表現（46° ハロ・幻日などは無し）
fn icePhase(c: f32) -> f32 {
  let th = acos(clamp(c, -1.0, 1.0));
  let d = (th - 0.3840) / 0.01047;
  let halo = exp(-d * d) / 0.0437;
  return 0.97 * hg(0.75, c) + 0.03 * halo;
}

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
  let v = sampleVolSmoothZ(vol, smpRep, uvw);
  var lwc = v.r; // 雲水 g/m³
  var iwc = v.g; // 雲氷 g/m³
  // ---- 細部合成（試作）：格子より細かい縁の凹凸。
  // 雲水量に比例して削る（固定量で削ると、薄い雲がノイズの形だけ残って繊維状に見えるため）。
  // 縁（雲水量の少ない所）ほど強く、渦粘性（乱流の強さ）が大きいほど強く削る。
  if (C.detail > 0.0 && lwc > 0.0) {
    let q = vec3f(lx, ly, h) / C.noisePeriod + vec3f(0.0, 0.0, C.time * 0.0005);
    let n = textureSampleLevel(noiseTex, smpRep, q, 0.0);
    let n2 = textureSampleLevel(noiseTex, smpRep, q * 3.7, 0.0);
    let turb = clamp(v.a / 20.0, 0.0, 1.0);
    let shape = mix(n.r * 0.6 + n.g * 0.4, n2.g * 0.5 + n2.b * 0.5, 0.35);
    let edge = 1.0 - clamp(lwc / 0.6, 0.0, 1.0);
    let cut = C.detail * (0.5 + 0.5 * turb) * edge * (1.0 - shape);
    let keep = clamp(1.0 - 1.6 * cut, 0.0, 1.0);
    lwc *= keep;
    iwc *= keep;
  }
  let sI = sigmaIce(iwc * 1e-3);
  m.sigma = sigmaCloud(lwc * 1e-3, C.nDrop) + sI + sigmaRain(v.b * 1e-3);
  m.iceFrac = select(0.0, sI / m.sigma, m.sigma > 0.0);
  if (withShadow && m.sigma > 0.0) {
    m.odUp = textureSampleLevel(shadowVol, smpRep, uvw, 0.0).g;
  }
  return m;
}

// ---------------------------------------------------------------- 薄明光線
// 雲の影に入った空気は太陽光を散乱しない。視線上で「影のせいで欠けた単散乱光」を見積もって差し引く。
// （空の輝度 LUT は雲の無い大気の値なので、その差分として扱う。多重散乱の欠けは無視）
fn shaftDeficit(o: vec3f, d: vec3f, tEnd: f32, sun: vec3f, jitter: f32) -> vec3f {
  if (sun.z <= 0.01) { return vec3f(0.0); }
  let tmax = min(tEnd, 40000.0);
  if (tmax <= 0.0) { return vec3f(0.0); }
  let camR = A.bottom_radius + o.z / 1000.0;
  let nu = dot(d, sun);
  let pr = rayleighPhase(nu);
  let pm = miePhase(A.mie_g, nu);
  var acc = vec3f(0.0);
  let N = 24;
  var tPrev = 0.0;
  for (var i = 0; i < N; i++) {
    let x = (f32(i) + jitter) / f32(N);
    let t = tmax * x * x; // 近くを細かく
    let dt = t - tPrev;
    tPrev = t;
    let p = o + d * t;
    let h = altitude(p);
    if (h > C.slabTop) { continue; }
    var q = p;
    if (h < C.slabBot) { q = p + sun * ((C.slabBot + 1.0 - h) / sun.z); }
    let hq = altitude(q);
    let od = textureSampleLevel(shadowVol, smpRep, domainUvw(q, hq), 0.0).r;
    let shadowFrac = 1.0 - exp(-od);
    if (shadowFrac <= 0.001) { continue; }
    let rKm = A.bottom_radius + h / 1000.0;
    let up = normalize(p - vec3f(0.0, 0.0, -EARTH_R_M));
    let sunE = F.lightIllum * getTransmittanceToSun(rKm, dot(up, sun));
    let hk = h / 1000.0;
    let scat = A.rayleigh_scattering * exp(-hk / 8.0) * pr + A.mie_scattering * exp(-hk / A.mie_scale_height) * pm; // [1/km]
    let tv = getTransmittance(camR, d.z, t / 1000.0, rayIntersectsGround(camR, d.z));
    acc += tv * sunE * scat * (dt / 1000.0) * shadowFrac;
  }
  // 数値の破綻（NaN/Inf）が露出計算に波及しないように
  if (any(acc != acc) || any(acc > vec3f(1e9))) { return vec3f(0.0); }
  return acc;
}

fn ignNoise(p: vec2f) -> f32 {
  return fract(52.9829189 * fract(dot(p, vec2f(0.06711056, 0.00583715))));
}

@fragment
fn fsMain(in: VsOut) -> @location(0) vec4f {
  var cur = march(in);
  if (any(cur != cur)) { cur = vec4f(0.0, 0.0, 0.0, 1.0); }
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

  // 主光源（昼は太陽、太陽が地平線下 6° より低い夜は月）
  let sun = F.lightDir;
  let cosT = dot(d, sun);
  let dropD = C.dropDiamUm;
  // 多重散乱のオクターブ近似（Wrenninge 2013）
  var phases: array<f32, 4>;
  let iso = 1.0 / (4.0 * PI);
  let p0 = cloudPhase(cosT, dropD);
  var phasesI: array<f32, 4>;
  let pI = icePhase(cosT);
  for (var i = 0; i < 4; i++) {
    phases[i] = mix(p0, iso, 1.0 - pow(0.5, f32(i)));
    phasesI[i] = mix(pI, iso, 1.0 - pow(0.5, f32(i)));
  }

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
      let sunE = F.lightIllum * getTransmittanceToSun(rKm, muS);
      let skyE = getIrradiance(rKm, muS) * F.lightScale;
      var S = vec3f(0.0);
      var a = 1.0; var b = 1.0;
      for (var oc = 0; oc < 4; oc++) {
        S += b * sunE * mix(phases[oc], phasesI[oc], m.iceFrac) * exp(-a * m.odSun);
        a *= 0.5; b *= 0.5;
      }
      // 環境光：上半球は空、下半球は地面の照り返し（雲による遮りを上下の光学的厚さで近似）
      let groundE = (skyE + F.lightIllum * getTransmittanceToSun(A.bottom_radius, muS) * max(muS, 0.0)) * C.groundAlbedo;
      // 夜は大気光・星明かり（上から）と光害（地上の照明。下から雲底を照らす）も効く
      let nightUp = (F.nightGlow + 4.0e-5) * vec3f(0.9, 1.0, 0.75);
      let cityUp = F.lpZenith * 4.0 * vec3f(1.0, 0.82, 0.6) / 0.86;
      let amb = (skyE * exp(-0.25 * m.odUp) * 0.5 + groundE * 0.5 * 0.6) / PI
              + nightUp * exp(-0.25 * m.odUp) * 0.5 + cityUp * 0.5;
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

  let pre = 1.0 / max(expo[0], 1e-4);
  var tShaft = 40000.0;
  if (sg.x > 0.0) { tShaft = sg.x; }
  if (wsum > 0.0) { tShaft = depthW / wsum; }
  let deficit = shaftDeficit(o, d, tShaft, sun, jitter);
  if (wsum <= 0.0) { return vec4f(-deficit * pre, 1.0); }
  // 空気遠近：雲の平均的な距離までの大気の散乱と透過
  let dist = depthW / wsum;
  let pKm = camKm + d * (dist / 1000.0);
  let ap = getSkyRadianceToPoint(camKm, pKm, sun);
  let col = ap.radiance * (1.0 - T) + ap.transmittance * L - deficit;
  return vec4f(col * pre, T);
}
