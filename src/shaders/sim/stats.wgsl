// 層ごとの統計（1ワークグループ = 1層）。sO（最新の状態）から集計する。
// stats[k*16 + m]：
//  0 Σθ  1 Σqv  2 Σqc  3 Σqr  4 max|u|  5 max|v|  6 max w  7 min w
//  8 max qc  9 max qr  10 雲セル数(qc>1e-5)  11 Σθl（液水温位）  12 min θ'  13 max θ'
// 層 k = nz のワークグループは地表面（diag）の集計：0 Σ降水  1 Σ切り詰め水量

const NSTAT: u32 = 16u;
var<workgroup> red: array<array<f32, 14>, 256>;

@compute @workgroup_size(256)
fn csStats(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wid: vec3u) {
  let k = wid.x;
  let plane = P.nx * P.ny;
  var a: array<f32, 14>;
  for (var m = 0u; m < 14u; m++) { a[m] = 0.0; }
  a[7] = 1e9; a[12] = 1e9; a[13] = -1e9;

  if (k < P.nz) {
    let cl = col[k];
    for (var p = li; p < plane; p += 256u) {
      let id = p + plane * k;
      let th = sO[F_TH * P.n + id];
      let qv = sO[F_QV * P.n + id];
      let qc = sO[F_QC * P.n + id];
      let qr = sO[F_QR * P.n + id];
      let w = sO[F_W * P.n + id];
      a[0] += th; a[1] += qv; a[2] += qc; a[3] += qr;
      a[4] = max(a[4], abs(sO[F_U * P.n + id]));
      a[5] = max(a[5], abs(sO[F_V * P.n + id]));
      a[6] = max(a[6], w); a[7] = min(a[7], w);
      a[8] = max(a[8], qc); a[9] = max(a[9], qr);
      a[10] += select(0.0, 1.0, qc > 1e-5);
      a[11] += th - LV / (CP * cl.pi0) * (qc + qr);
      a[12] = min(a[12], th - cl.th0); a[13] = max(a[13], th - cl.th0);
    }
  } else {
    for (var p = li; p < plane; p += 256u) {
      a[0] += diag[D_PRECIP * P.n + p];
      a[1] += diag[D_PRECIP * P.n + plane + p];
    }
  }
  red[li] = a;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (li < s) {
      var x = red[li];
      let y = red[li + s];
      for (var m = 0u; m < 14u; m++) {
        if (m == 4u || m == 5u || m == 6u || m == 8u || m == 9u || m == 13u) { x[m] = max(x[m], y[m]); }
        else if (m == 7u || m == 12u) { x[m] = min(x[m], y[m]); }
        else { x[m] += y[m]; }
      }
      red[li] = x;
    }
    workgroupBarrier();
  }
  if (li == 0u) {
    for (var m = 0u; m < 14u; m++) { stats[k * NSTAT + m] = red[0][m]; }
  }
}
