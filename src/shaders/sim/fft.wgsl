// 1次元複素 FFT（Stockham 基数2、共有メモリ内、N ≤ 1024）。1ワークグループ = 1本の線。
// TS 側で __N__（2の冪, ≤512）、__LOG2N__、__DIR__（-1 順変換 / +1 逆変換）、__AXIS__（0=x, 1=y）を置換。
// 逆変換では 1/N を掛ける。（common.wgsl の後ろに連結）

const FN: u32 = __N__u;
const FHALF: u32 = FN / 2u;
const FLOG2N: u32 = __LOG2N__u;
const FDIR: f32 = __DIR__;
const FAXIS: u32 = __AXIS__u;
const FWG: u32 = min(FHALF, 256u); // ワークグループのスレッド数（N>512 では1スレッドが複数の蝶演算を担当）

var<workgroup> bufA: array<vec2f, FN>;
var<workgroup> bufB: array<vec2f, FN>;

fn cmul(a: vec2f, b: vec2f) -> vec2f { return vec2f(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }

@compute @workgroup_size(FWG)
fn csFft(@builtin(local_invocation_id) lid: vec3u, @builtin(workgroup_id) wid: vec3u) {
  let t = lid.x;
  var base: u32;
  var stride: u32;
  if (FAXIS == 0u) {
    base = P.nx * (wid.x + P.ny * wid.y); // wid = (j, k)
    stride = 1u;
  } else {
    base = wid.x + P.nx * P.ny * wid.y;   // wid = (i, k)
    stride = P.nx;
  }
  for (var q = t; q < FN; q += FWG) { bufA[q] = cplx[base + q * stride]; }
  workgroupBarrier();

  var ns = 1u;
  for (var s = 0u; s < FLOG2N; s++) {
    for (var b = t; b < FHALF; b += FWG) {
      let k = b % ns;
      let ang = FDIR * 6.283185307 * f32(k) / f32(2u * ns);
      let tw = vec2f(cos(ang), sin(ang));
      let d = (b / ns) * ns * 2u + k;
      if ((s & 1u) == 0u) {
        let v0 = bufA[b];
        let v1 = cmul(bufA[b + FHALF], tw);
        bufB[d] = v0 + v1;
        bufB[d + ns] = v0 - v1;
      } else {
        let v0 = bufB[b];
        let v1 = cmul(bufB[b + FHALF], tw);
        bufA[d] = v0 + v1;
        bufA[d + ns] = v0 - v1;
      }
    }
    workgroupBarrier();
    ns *= 2u;
  }
  var scale = 1.0;
  if (FDIR > 0.0) { scale = 1.0 / f32(FN); }
  for (var q = t; q < FN; q += FWG) {
    if ((FLOG2N & 1u) == 1u) { cplx[base + q * stride] = bufB[q] * scale; }
    else { cplx[base + q * stride] = bufA[q] * scale; }
  }
}
