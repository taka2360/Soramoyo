// 層ごとの統計（1ワークグループ = 1層）。sO（最新の状態）から集計する。
// stats[k*16 + m]：
//  0 Σθ  1 Σqv  2 Σqc（雲水）  3 Σ降水（雨＋雪＋あられ）  4 max|u|  5 max|v|  6 max w  7 min w
//  8 max（雲水＋雲氷）  9 max 降水  10 雲セル数（雲水＋雲氷 > 1e-5）  11 Σθil（氷・液水温位）  12 min θ'  13 max θ'
//  14 Σqi（雲氷）
// 層 k = nz のワークグループは地表面（diag）の集計：0 Σ降水  1 Σ切り詰め水量  2 ΣTs  3 ΣH  4 ΣLE  5 Σ日射
//  6 Σ蒸発の累計  7 Σ大規模強制で入った水の累計

const NSTAT: u32 = 16u;
const NACC: u32 = 15u;
var<workgroup> red: array<array<f32, 15>, 256>;

@compute @workgroup_size(256)
fn csStats(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wid: vec3u) {
  let k = wid.x;
  let plane = P.nx * P.ny;
  var a: array<f32, 15>;
  for (var m = 0u; m < NACC; m++) { a[m] = 0.0; }
  if (k < P.nz) { a[7] = 1e9; a[12] = 1e9; a[13] = -1e9; }

  if (k < P.nz) {
    let cl = col[k];
    for (var p = li; p < plane; p += 256u) {
      let id = p + plane * k;
      let th = sO[F_TH * P.n + id];
      let qv = sO[F_QV * P.n + id];
      let qc = sO[F_QC * P.n + id];
      let qi = sO[F_QI * P.n + id];
      let qp = sO[F_QR * P.n + id] + sO[F_QS * P.n + id] + sO[F_QG * P.n + id];
      let w = sO[F_W * P.n + id];
      a[0] += th; a[1] += qv; a[2] += qc; a[3] += qp;
      a[4] = max(a[4], abs(sO[F_U * P.n + id]));
      a[5] = max(a[5], abs(sO[F_V * P.n + id]));
      a[6] = max(a[6], w); a[7] = min(a[7], w);
      a[8] = max(a[8], qc + qi); a[9] = max(a[9], qp);
      a[10] += select(0.0, 1.0, qc + qi > 1e-5);
      a[11] += th - (LV * (qc + sO[F_QR * P.n + id]) + LS * (qi + sO[F_QS * P.n + id] + sO[F_QG * P.n + id])) / (CP * cl.pi0);
      a[12] = min(a[12], th - cl.th0); a[13] = max(a[13], th - cl.th0);
      a[14] += qi;
    }
  } else {
    for (var p = li; p < plane; p += 256u) {
      a[0] += diag[dSurf(0u, p)];
      a[1] += diag[dSurf(1u, p)];
      a[2] += diag[dSurf(2u, p)];
      a[3] += diag[dSurf(3u, p)];
      a[4] += diag[dSurf(4u, p)];
      a[5] += diag[dSurf(5u, p)];
      a[6] += diag[dSurf(8u, p)];
      a[7] += diag[dSurf(9u, p)];
    }
  }
  red[li] = a;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (li < s) {
      var x = red[li];
      let y = red[li + s];
      for (var m = 0u; m < NACC; m++) {
        if (k == P.nz) { x[m] += y[m]; }
        else if (m == 4u || m == 5u || m == 6u || m == 8u || m == 9u || m == 13u) { x[m] = max(x[m], y[m]); }
        else if (m == 7u || m == 12u) { x[m] = min(x[m], y[m]); }
        else { x[m] += y[m]; }
      }
      red[li] = x;
    }
    workgroupBarrier();
  }
  if (li == 0u) {
    for (var m = 0u; m < NACC; m++) { stats[k * NSTAT + m] = red[0][m]; }
  }
}
