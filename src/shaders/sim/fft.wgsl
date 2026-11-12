// 圧力の水平 FFT（Stockham 基数2、共有メモリ内、N ≤ 1024）。（common.wgsl の後ろに連結）
// TS 側で __N__（2の冪）、__LOG2N__、__DIR__（-1 順変換 / +1 逆変換）、__MODE__ を置換する。
//
// 右辺（発散）は実数なので、x 方向は「2本の実数の線を1本の複素 FFT で」変換する（実数 FFT）。
//   線の番号 L = j + ny·k（行 = 連続した nx 個）。ワークグループ m は線 2m と 2m+1 を受け持つ（奇数本なら相方は 0）。
//   MODE 0（x 順変換）：z = a + i·b を FFT し、A_k = (Z_k + conj Z_{N−k})/2、B_k = (Z_k − conj Z_{N−k})/(2i) に分ける。
//                       各行の 0..nx/2 の位置に半分のスペクトルを書く（残りは使わない）。
//   MODE 1（x 逆変換）：Z_q = A_q + i·B_q を全 q に復元（q > N/2 は共役）して逆 FFT。実部→線 2m、虚部→線 2m+1。
//   MODE 2（y 方向）  ：kx = 0..nx/2 の列だけ複素 FFT。隣り合う FLINES 列をまとめて処理し、読み書きを連続アドレスにする。
// 逆変換では 1/N を掛ける。

const FN: u32 = __N__u;
const FHALF: u32 = FN / 2u;
const FLOG2N: u32 = __LOG2N__u;
const FDIR: f32 = __DIR__;
const FMODE: u32 = __MODE__u;
// 1ワークグループで同時に処理する線の数（y 方向のみ複数。共有メモリ 16 KB に収める）
const FLINES: u32 = select(1u, clamp(2048u / FN, 1u, 8u), FMODE == 2u);
const FWORK: u32 = FLINES * FHALF;            // 1段の蝶演算の総数
const FWG: u32 = min(FWORK, 256u);            // ワークグループのスレッド数
const FBPT: u32 = FWORK / FWG;                // 1スレッドが受け持つ蝶演算の数
const XGROUPS: u32 = 32768u;                  // x 方向の線の組は (wid.x + XGROUPS·wid.y) 番目

// buf[q·FLINES + line]：同じ q の FLINES 本が隣り合う
var<workgroup> buf: array<vec2f, FLINES * FN>;

fn cmul(a: vec2f, b: vec2f) -> vec2f { return vec2f(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }
fn conj2(a: vec2f) -> vec2f { return vec2f(a.x, -a.y); }
fn mulI(a: vec2f) -> vec2f { return vec2f(-a.y, a.x); } // i·a

// 共有メモリ上で FLINES 本の FFT をその場で行う（1本のバッファ：読み→同期→書き）
fn fftShared(t: u32) {
  var ns = 1u;
  for (var s = 0u; s < FLOG2N; s++) {
    var v0: array<vec2f, FBPT>;
    var v1: array<vec2f, FBPT>;
    for (var r = 0u; r < FBPT; r++) {
      let e = t + r * FWG;
      let line = e % FLINES;
      let b = e / FLINES;
      let k = b % ns;
      let ang = FDIR * 3.14159265359 * f32(k) / f32(ns);
      let tw = vec2f(cos(ang), sin(ang));
      v0[r] = buf[b * FLINES + line];
      v1[r] = cmul(buf[(b + FHALF) * FLINES + line], tw);
    }
    workgroupBarrier();
    for (var r = 0u; r < FBPT; r++) {
      let e = t + r * FWG;
      let line = e % FLINES;
      let b = e / FLINES;
      let k = b % ns;
      let d = (b / ns) * ns * 2u + k;
      buf[d * FLINES + line] = v0[r] + v1[r];
      buf[(d + ns) * FLINES + line] = v0[r] - v1[r];
    }
    workgroupBarrier();
    ns *= 2u;
  }
}

@compute @workgroup_size(FWG)
fn csFft(@builtin(local_invocation_id) lid: vec3u, @builtin(workgroup_id) wid: vec3u) {
  let t = lid.x;
  let nh = P.nx / 2u; // 半分のスペクトルは 0..nh（nh+1 個）

  if (FMODE == 2u) {
    // ---- y 方向：列 i0..i0+FLINES-1（i ≤ nh のみ）、層 k = wid.y
    let i0 = wid.x * FLINES;
    let k = wid.y;
    let pl = P.nx * P.ny;
    for (var e = t; e < FLINES * FN; e += FWG) {
      let line = e % FLINES;
      let q = e / FLINES;
      let i = i0 + line;
      var v = vec2f(0.0);
      if (i <= nh) { v = cplx[i + P.nx * q + pl * k]; }
      buf[e] = v;
    }
    workgroupBarrier();
    fftShared(t);
    var scale = 1.0;
    if (FDIR > 0.0) { scale = 1.0 / f32(FN); }
    for (var e = t; e < FLINES * FN; e += FWG) {
      let line = e % FLINES;
      let q = e / FLINES;
      let i = i0 + line;
      if (i <= nh) { cplx[i + P.nx * q + pl * k] = buf[e] * scale; }
    }
    return;
  }

  // ---- x 方向（実数 FFT）：線 2m と 2m+1
  let nLines = P.ny * P.nz;
  let m = wid.x + XGROUPS * wid.y;
  if (2u * m >= nLines) { return; } // ワークグループ全体で同じ判定（同期の前なので安全）
  let baseA = P.nx * (2u * m);
  let hasB = 2u * m + 1u < nLines;
  let baseB = baseA + P.nx;

  if (FMODE == 0u) {
    for (var q = t; q < FN; q += FWG) {
      var b = 0.0;
      if (hasB) { b = cplx[baseB + q].x; }
      buf[q] = vec2f(cplx[baseA + q].x, b);
    }
    workgroupBarrier();
    fftShared(t);
    for (var q = t; q <= nh; q += FWG) {
      let zk = buf[q];
      let zc = conj2(buf[(FN - q) % FN]);
      cplx[baseA + q] = 0.5 * (zk + zc);
      if (hasB) { let d = zk - zc; cplx[baseB + q] = 0.5 * vec2f(d.y, -d.x); }
    }
  } else {
    for (var q = t; q < FN; q += FWG) {
      var a: vec2f; var b = vec2f(0.0);
      if (q <= nh) {
        a = cplx[baseA + q];
        if (hasB) { b = cplx[baseB + q]; }
        if (q == 0u || q == nh) { a.y = 0.0; b.y = 0.0; } // 実数の線の 0 と N/2 の係数は実数
      } else {
        a = conj2(cplx[baseA + FN - q]);
        if (hasB) { b = conj2(cplx[baseB + FN - q]); }
      }
      buf[q] = a + mulI(b);
    }
    workgroupBarrier();
    fftShared(t);
    let scale = 1.0 / f32(FN);
    for (var q = t; q < FN; q += FWG) {
      let z = buf[q] * scale;
      cplx[baseA + q] = vec2f(z.x, 0.0);
      if (hasB) { cplx[baseB + q] = vec2f(z.y, 0.0); }
    }
  }
}
