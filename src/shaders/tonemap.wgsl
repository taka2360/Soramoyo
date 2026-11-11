// HDR（輝度 cd/m²）→ 表示。
// 白色点で割る（太陽光を白とみなす色順応）→ 露出（自動 or 固定）→ ACES 近似 → sRGB。
// 段階2で暗所視（プルキンエ効果）を追加する予定。

@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var hdrTex: texture_2d<f32>;
@group(0) @binding(2) var<storage, read> expo: array<f32, 4>;

fn acesFitted(x: vec3f) -> vec3f {
  // Narkowicz 2015 の近似
  let a = 2.51; let b = 0.03; let c = 2.43; let d = 0.59; let e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3f(0.0), vec3f(1.0));
}

// 色相を保つトーンマップ：輝度に ACES 曲線をかけて RGB を比例縮小し、
// 表示範囲を超えた分は少しだけ白に寄せる。チャンネルごとの ACES だと、
// 沈む太陽（実際は輝度がまだ極端に高い）が真っ白に飛んでしまい、橙色に見えないため。
fn toneMap(x: vec3f) -> vec3f {
  let l = max(luminance(x), 1e-9);
  let lm = acesFitted(vec3f(l)).x;
  var c = x * (lm / l);
  let m = max(c.r, max(c.g, c.b));
  var over = 1.0;
  if (m > 1.0) {
    c = mix(c / m, vec3f(1.0), 0.12 * (1.0 - 1.0 / m));
    over = 1.0 / m;
  }
  // 通常の明るさではチャンネルごとの圧縮も少し混ぜ、明部の自然な色の抜けを残す
  return clamp(mix(c, acesFitted(x), 0.25 * over), vec3f(0.0), vec3f(1.0));
}

fn linearToSrgb(c: vec3f) -> vec3f {
  let lo = c * 12.92;
  let hi = 1.055 * pow(c, vec3f(1.0 / 2.4)) - 0.055;
  return select(hi, lo, c <= vec3f(0.0031308));
}

fn ign(p: vec2f) -> f32 {
  return fract(52.9829189 * fract(dot(p, vec2f(0.06711056, 0.00583715))));
}

@fragment
fn fsMain(in: VsOut) -> @location(0) vec4f {
  let p = vec2i(in.pos.xy);
  let hdr = textureLoad(hdrTex, p, 0).rgb / (F.whitePoint * max(expo[1], 1e-12));
  var exposure: f32;
  if (F.autoExposure > 0.5) {
    // 順応輝度に下限を設け、暗い場面は暗く見えるようにする。
    // キー値は Krawczyk 2005：暗い場面ほど小さい（＝暗く表示）
    let La = max(expo[0], 0.02);
    // 0.2 倍は ACES 曲線と組み合わせたときの見た目の調整（晴天の天頂が濃い青に見える程度）
    let key = 0.2 * (1.03 - 2.0 / (2.0 + log(La + 1.0) / log(10.0)));
    exposure = key / La;
  } else {
    exposure = 1.0e-4; // 固定：EV 0 で晴天の空がほどよく見える程度
  }
  exposure *= exp2(F.exposureEV);
  let mapped = toneMap(hdr * exposure);
  var s = linearToSrgb(mapped);
  s += (ign(in.pos.xy) - 0.5) / 255.0;
  return vec4f(s, 1.0);
}
