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

// ---------------------------------------------------------------- 夜空の光
@group(0) @binding(17) var mwTex: texture_2d<f32>;

// ENU → J2000 赤道座標（回転行列の転置）
fn enuToJ2000(v: vec3f) -> vec3f { return F.eqX * v.x + F.eqY * v.y + F.eqZ * v.z; }

// 天の川（輪郭カタログの5段階を、ぼかした上で表面輝度に換算）[cd/m²]
// 段階 → 表面輝度は目安の対応（最も外側 ≈ 22.6、最も明るい部分 ≈ 20.4 mag/arcsec²）
fn milkyWay(vTrue: vec3f) -> vec3f {
  let e = enuToJ2000(vTrue);
  let ra = atan2(e.y, e.x);
  let dec = asin(clamp(e.z, -1.0, 1.0));
  let uv = vec2f(fract(ra / (2.0 * PI) + 1.0), 0.5 - dec / PI);
  let v = textureSampleLevel(mwTex, smpRep, uv, 0.0).r;
  if (v < 0.005) { return vec3f(0.0); }
  let mu = 23.2 - 2.75 * v;
  let l = 1.08e5 * pow(10.0, -0.4 * mu) * smoothstep(0.0, 0.06, v);
  // 積分光の色はやや暖色（G〜K 型星が主体）
  return l * vec3f(1.08, 0.98, 0.86) * F.mwScale;
}

// 大気光（高度 ~90 km の薄い層の発光）。van Rhijn 関数で地平線付近ほど明るい
fn airglow(vTrue: vec3f) -> vec3f {
  let ratio = 6371.0 / 6461.0;
  let s2 = 1.0 - vTrue.z * vTrue.z;
  let vr = 1.0 / sqrt(max(1.0 - ratio * ratio * s2, 0.02));
  // 557.7 nm の酸素の緑線と OH の赤い帯 → 黄緑寄り（輝度 1 に正規化した色）
  return F.nightGlow * vr * vec3f(0.85, 1.0, 0.55) / 0.937;
}

// 光害：地上の照明が大気で散乱された光。地平線に近いほど明るい（経験的な形）
fn lightPollution(view: vec3f) -> vec3f {
  let altDeg = max(asin(clamp(view.z, -1.0, 1.0)) * 180.0 / PI, 0.0);
  let shape = 1.0 + 5.0 * exp(-altDeg / 10.0);
  return F.lpZenith * shape * vec3f(1.0, 0.82, 0.6) / 0.86;
}

// 月面：Lommel–Seeliger 則（反射の明るさが入射・出射角の余弦で決まる）。アルベドの模様（海）は未実装
fn moonDisk(dTrue: vec3f, angM: f32) -> vec3f {
  let md = F.moonDir;
  let rho = (dTrue - md * dot(dTrue, md)) / sin(F.moonAngRadius);
  let r2 = min(dot(rho, rho), 1.0);
  let n = rho - md * sqrt(1.0 - r2);
  let mu = sqrt(1.0 - r2);
  let mu0 = dot(n, F.sunDir);
  var ls = 0.0;
  if (mu0 > 0.0) { ls = mu0 / (mu0 + mu); }
  let lit = F.moonIllum * F.moonLumScale * ls;
  // 地球照（地球で反射した太陽光。やや青みがかる）
  let es = F.earthshine * vec3f(0.85, 0.95, 1.2) * mu;
  return lit + es;
}

@fragment
fn fsMain(in: VsOut) -> @location(0) vec4f {
  let view = viewDir(F, in.uv);
  let cam = F.camPos;
  let sun = F.sunDir;
  let moon = F.moonDir;
  let r = length(cam);
  let rmu = dot(cam, view);
  let moonRatio = luminance(F.moonIllum) / luminance(F.sunIllum);

  // 地面（地球の球面）との交差
  let disc = rmu * rmu - r * r + A.bottom_radius * A.bottom_radius;
  var tGround = -1.0;
  if (disc >= 0.0) { tGround = -rmu - sqrt(disc); }

  // 太陽面・月面の判定用の角度（fwidth は一様な制御フローで呼ぶ必要があるため分岐の外で計算）
  let dTrue = apparentToTrue(view);
  let ang = 2.0 * asin(clamp(length(dTrue - sun) * 0.5, 0.0, 1.0)); // 小角度でも精度が落ちない形
  let angPx = fwidth(ang) + 1e-7;
  let angM = 2.0 * asin(clamp(length(dTrue - moon) * 0.5, 0.0, 1.0));
  let angMPx = fwidth(angM) + 1e-7;

  var L: vec3f;
  if (tGround > 0.0) {
    let p = cam + view * tGround;
    let n = normalize(p);
    let e = sunAndSkyIrradiance(p, n, sun);
    // 月明かり（直達＋月による天空光）
    let rp = length(p);
    let muM = dot(p, moon) / rp;
    let eMoon = F.moonIllum * getTransmittanceToSun(rp, muM) * max(dot(n, moon), 0.0)
              + getIrradiance(rp, muM) * moonRatio;
    let groundL = F.groundAlbedo / PI * (e[0] + e[1] + eMoon);
    let a = getSkyRadianceToPoint(cam, p, sun);
    let am = getSkyRadianceToPoint(cam, p, moon);
    L = groundL * a.transmittance + a.radiance + am.radiance * moonRatio + lightPollution(vec3f(view.xy, 0.0)) * 0.3;
  } else {
    let s = getSkyRadiance(cam, view, sun);
    let sm = getSkyRadiance(cam, view, moon);
    L = s.radiance + sm.radiance * moonRatio + lightPollution(view);

    // 月面の覆い（月の後ろの星・天の川・太陽を隠す。日食も自然に起きる）
    let radM = F.moonAngRadius;
    let coverM = clamp((radM - angM) / angMPx + 0.5, 0.0, 1.0);

    // 大気の外からの光（大気で減光される）
    var outer = (milkyWay(dTrue) + airglow(dTrue) + vec3f(4.0e-5)) * (1.0 - coverM);

    // 太陽面：見かけの視線を真の方向に戻して比べる → 地平線近くで自然に扁平になる
    let rad = F.sunAngRadius;
    if (ang < rad * 1.5) {
      let cover = clamp((rad - ang) / angPx + 0.5, 0.0, 1.0);
      let x = clamp(ang / rad, 0.0, 1.0);
      let cosTheta = sqrt(1.0 - x * x);
      outer += F.sunIllum / (PI * rad * rad) * limbDarkening(cosTheta) * cover * (1.0 - coverM);
    }
    if (angM < radM * 1.5) {
      outer += moonDisk(dTrue, angM) * coverM;
    }
    L += outer * s.transmittance;
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
  var outc = max(toHdr(L) * cl.a + cl.rgb, vec3f(0.0)); // rgb は薄明光線の分だけ負になりうる
  if (any(outc != outc)) { outc = vec3f(0.0); }
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
