// 雲のレイマーチング（半解像度）。出力0：rgb = 雲と、雲までの空気の散乱光（プリエクスポージャー済み）、a = 雲の透過率
// 出力1：雲までの代表距離 [m]（時間方向の蓄積の再投影用。雲が無ければ 0）
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
@group(0) @binding(18) var velVol: texture_3d<f32>;
@group(0) @binding(19) var smpNoise: sampler;  // ノイズ用（3方向とも繰り返し）

// 今サンプルしている点までのカメラからの距離 [m]（細部の遠方での間引き用）
var<private> gDist: f32 = 0.0;
// 主サンプル点のドメインワープのずれ [m]（影テクスチャを同じ形で読むため）
var<private> gOff: vec3f = vec3f(0.0);

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
  let q = p + sun * reach + vec3f(gOff.xy, 0.0);
  let hq = max(altitude(q) + gOff.z, 0.0);
  if (hq > C.slabTop) { return od; }
  // 影テクスチャも細部合成と同じ位置のずれで、近くは3次 B スプラインで読む（線形補間の四角い影を避ける）
  if (gDist < 15000.0 * C.detailHQ) { return od + sampleVolSmooth3(shadowVol, smpRep, domainUvw(q, hq)).r; }
  return od + textureSampleLevel(shadowVol, smpRep, domainUvw(q, hq), 0.0).r;
}

// 細部合成のノイズ。雲水（積雲）は上昇流で「もこもこ」（Worley の粒）、下降流・蒸発域で「すじ・ちぎれ」（Perlin–Worley）。
// 雲氷（巻雲・かなとこ）は鉛直に縮めた座標で、水平に伸びた繊維状に。
fn detailNoise(p: vec3f, up: f32, ice: f32, turb: f32) -> f32 {
  let stretch = mix(vec3f(1.0), vec3f(1.0, 1.0, 3.0), ice);  // 水平の周波数は領域を割り切るよう整数倍のみ
  let q = p * stretch / C.noisePeriod;
  let n1 = textureSampleLevel(noiseTex, smpNoise, q, 0.0);
  var billow = n1.b;
  var wisp = n1.r;
  // 細かい段（〜25〜90 m）は近くだけ（遠くはエイリアシングになるため）
  // 乱流が強いほど細かい段の割合を増やす
  let hf = (1.0 - smoothstep(2500.0, 9000.0, gDist)) * C.detailHQ * (0.5 + 0.5 * turb);
  if (hf > 0.0) {
    let n2 = textureSampleLevel(noiseTex, smpNoise, q * vec3f(4.0, 4.0, 3.7) + vec3f(0.31, 0.17, 0.53), 0.0);
    billow = mix(billow, 0.55 * billow + 0.45 * n2.a, hf);
    wisp = mix(wisp, 0.6 * wisp + 0.4 * n2.g, hf);
  }
  return mix(wisp, billow, up);
}

// 細部合成の倍率（格子以下の雲量としての解釈）。
// 格子の値 = 雲量 cf ×（雲の中の雲水量 qIn）とみなし、各点が雲の中か外かをノイズで決める：
//   中なら qIn、外なら 0。ノイズが一様分布なら平均は格子の値に一致する（雲の量は計算どおり）。
// qIn は 0.3 g/m³（雲氷は 0.05）か格子の値の大きい方。格子の値がそれ以上なら雲量 1（削らない）。
// 境界の幅 w は近くで細く（鋭い縁）、遠くで太く（エイリアシング防止。w が大きいと格子の値そのものに近づく）。
// ノイズは各点の風（計算領域に対する相対速度）で流す。長く流すと歪むので、
// 位相を半周期ずらした2組を交互にリセットして重ねる（flow map の手法）。
fn detailKeep(p: vec3f, uvw: vec3f, lwc: f32, iwc: f32, nu: f32) -> f32 {
  let ice = select(0.0, 1.0, iwc > lwc);
  let q = max(lwc, iwc);
  let cf = q / mix(0.3, 0.05, ice);
  if (cf >= 1.0) { return 1.0; }
  let vel = textureSampleLevel(velVol, smpRep, uvw, 0.0).xyz;
  let up = smoothstep(-0.3, 1.2, vel.z);
  let Tp = 30.0;  // 長いと風の差で模様が引き伸ばされて筋になる
  let ph = fract(C.time / Tp);
  let ph2 = fract(ph + 0.5);
  let wA = 1.0 - abs(2.0 * ph - 1.0);
  let turb = clamp(nu / 20.0, 0.0, 1.0);
  let eA = detailNoise(p - vel * (ph * Tp), up, ice, turb);
  let eB = detailNoise(p - vel * (ph2 * Tp) + vec3f(437.0, 811.0, 173.0), up, ice, turb);
  let e = smoothstep(0.2, 0.8, mix(eB, eA, wA));  // おおよそ一様分布に
  let w = mix(0.1, 0.35, smoothstep(5000.0, 60000.0, gDist));
  let inside = smoothstep(e - w, e + w, cf);
  return mix(1.0, inside / max(cf, 1e-4), clamp(C.detail, 0.0, 1.0));
}

fn sampleMedium(p: vec3f, h: f32, withShadow: bool) -> Medium {
  var m: Medium;
  let lx = p.x - C.origin.x;
  let ly = p.y - C.origin.y;
  var uvw = vec3f(lx / C.size.x, ly / C.size.y, zToW(zmap, h, C.ztop, C.nz));
  var v = sampleVolSmoothZ(vol, smpRep, uvw);
  if (withShadow) { gOff = vec3f(0.0); }
  // 雲の近くでは読む位置をノイズで格子1個分ほどずらす（ドメインワープ）。
  // 積雲は格子数個分の太さしかないので、縁を鋭くするだけでは格子の形（平らな側面の柱）がそのまま見える。
  // 格子幅程度の渦が表面をこぶ状に押し出す効果として、こぶの大きさ（100〜400 m）のノイズで位置をずらす。
  // ずらす量は乱流の強さ（渦粘性）で増やす。
  if (C.detail > 0.0 && v.r + v.g > 1e-4) {
    let q0 = vec3f(lx, ly, h) / C.noisePeriod;
    let nw = textureSampleLevel(noiseTex, smpNoise, q0 * vec3f(2.0, 2.0, 2.5) + vec3f(0.5, 0.21, 0.77), 0.0);
    let nw2 = textureSampleLevel(noiseTex, smpNoise, q0 * vec3f(2.0, 2.0, 2.5) + vec3f(0.13, 0.62, 0.35), 0.0);
    let amp = C.detail * C.cellSize * (0.7 + 0.5 * clamp(v.a / 20.0, 0.0, 1.0));
    let off = (smoothstep(vec3f(0.3), vec3f(0.8), vec3f(nw.b, nw2.b, nw.a)) - vec3f(0.5)) * 2.0 * amp;
    let pw = vec3f(lx + off.x, ly + off.y, max(h + 0.5 * off.z, 0.0));
    if (withShadow) { gOff = vec3f(off.xy, 0.5 * off.z); }
    uvw = vec3f(pw.x / C.size.x, pw.y / C.size.y, zToW(zmap, pw.z, C.ztop, C.nz));
    if (gDist < 15000.0 * C.detailHQ) { v = sampleVolSmooth3(vol, smpRep, uvw); }
    else { v = sampleVolSmoothZ(vol, smpRep, uvw); }
  }
  var lwc = v.r; // 雲水 g/m³
  var iwc = v.g; // 雲氷 g/m³
  // ごく薄い値（数値拡散のにじみ）を晴れとみなす判定は、細部合成の前の格子の値で行う
  let mL = cloudMask(lwc);
  let mI = iceMask(iwc);
  // ---- 細部合成：格子（50〜250 m）より細かい縁の凹凸。雲の量・大きさ・寿命は計算どおりで、縁の形だけを足す。
  var pr = v.b;
  if (C.detail > 0.0 && mL + mI > 0.0) {
    let keep = detailKeep(vec3f(lx, ly, h), uvw, lwc * mL, iwc * mI, v.a);
    // 雲の中にある降水（積乱雲の中の雪・あられ）も雲と同じ形で。雲の外の雨脚・尾流雲はなめらかなまま
    let inCloud = clamp((lwc * mL + iwc * mI) / max(pr * 0.3, 1e-6), 0.0, 1.0);
    pr *= mix(1.0, keep, inCloud);
    lwc *= keep;
    iwc *= keep;
  }
  let sI = sigmaIce(iwc * 1e-3) * mI;
  m.sigma = sigmaCloud(lwc * 1e-3, C.nDrop) * mL + sI + sigmaRain(pr * 1e-3);
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
  let N = i32(C.shaftSamples);
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

struct CloudOut {
  @location(0) color: vec4f,
  @location(1) depth: f32,
};

var<private> gDepth: f32 = 0.0;

@fragment
fn fsMain(in: VsOut) -> CloudOut {
  var cur = march(in);
  if (any(cur != cur)) { cur = vec4f(0.0, 0.0, 0.0, 1.0); }
  // 半精度の上限（65504）を超えると Inf になり、合成で NaN になる。色相を保ったまま頭打ちにする
  let mx = max(abs(cur.r), max(abs(cur.g), abs(cur.b)));
  if (!(mx < 30000.0)) { cur = vec4f(select(vec3f(0.0), cur.rgb * (30000.0 / mx), mx < 3.0e38), cur.a); }
  var o: CloudOut;
  o.color = cur;
  o.depth = select(0.0, gDepth, gDepth == gDepth);
  return o;
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
  // 刻みの方針：雲の無い所は粗く（基本刻みの2倍）進み、雲に当たったら1歩戻って細かく（基本の 0.35 倍）進む。
  // 一定の刻みのままだと、薄い雲頂の面と刻みが干渉して縞模様（エイリアシング）が出るため。
  var ds = 2.0 * max(C.minStep, t * C.stepScale);
  t += ds * jitter;
  var inCloud = false;
  var emptyRun = 0;
  let maxSteps = i32(C.maxSteps);
  for (var i = 0; i < maxSteps; i++) {
    if (t >= t1 || T < 0.01) { break; }
    let p = o + d * t;
    let h = altitude(p);
    gDist = t;
    var m = sampleMedium(p, h, true);
    let base = max(C.minStep, t * C.stepScale);
    if (m.sigma > 1e-7 && !inCloud) {
      // 粗い刻みで雲に入った：1歩戻って細かい刻みでやり直す（ジッターで戻り幅もばらつかせる）
      inCloud = true;
      emptyRun = 0;
      t = max(t - ds * (0.5 + 0.5 * jitter), t0);
      ds = 0.35 * base;
      continue;
    }
    if (m.sigma > 1e-7) {
      emptyRun = 0;
      ds = 0.35 * base;
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
      // 雲を抜けたら、しばらく細かいまま進んでから粗い刻みに戻る
      emptyRun++;
      if (emptyRun > 7) { inCloud = false; } // 戻った距離（粗い刻み1歩 ≤ 基本の2倍）より長く細かく進んでから戻す
      ds = select(2.0 * base, 0.35 * base, inCloud);
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
  gDepth = dist;
  let pKm = camKm + d * (dist / 1000.0);
  let ap = getSkyRadianceToPoint(camKm, pKm, sun);
  let col = ap.radiance * (1.0 - T) + ap.transmittance * L - deficit;
  return vec4f(col * pre, T);
}
