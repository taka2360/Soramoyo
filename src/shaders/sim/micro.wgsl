// 雲微物理（暖かい雨）：飽和調整 + Kessler（自動変換・捕捉・雨の蒸発）+ 雨の落下。
// 1スレッド = 1柱。1ステップに1回、RK3 の後に sO（新しい状態）に対して適用する。
// 式は Klemp & Wilhelmson (1978) / WRF の Kessler スキームに準拠。
// 負の値（5次風上移流のアンダーシュート）はここで 0 に切り詰め、量を stats で監視する。

const MAXNZ_M: u32 = 256u;

fn so(f: u32, id: u32) -> f32 { return sO[f * P.n + id]; }

@compute @workgroup_size(8, 8, 1)
fn csMicro(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= P.nx || g.y >= P.ny) { return; }
  if ((P.flags & FLAG_MOIST) == 0u) { return; }
  let plane = P.nx * P.ny;
  let base = g.x + P.nx * g.y;
  let nz = P.nz;
  let dt = P.dtFull;
  let rain = (P.flags & FLAG_RAIN) != 0u;

  var qrCol: array<f32, MAXNZ_M>;
  var negAcc = 0.0; // 切り詰めで増やした水量（kg/m²）

  for (var k = 0u; k < nz; k++) {
    let id = base + plane * k;
    let cl = col[k];
    var th = so(F_TH, id);
    var qv = so(F_QV, id);
    var qc = so(F_QC, id);
    var qr = so(F_QR, id);
    let m = cl.rho0 * cl.dzc;
    if (qv < 0.0) { negAcc -= qv * m; qv = 0.0; }
    if (qc < 0.0) { negAcc -= qc * m; qc = 0.0; }
    if (qr < 0.0) { negAcc -= qr * m; qr = 0.0; }
    let p = cl.p0;
    let pi = cl.pi0;

    // --- 雨：自動変換と捕捉（陰的、WRF と同じ形）
    if (rain) {
      let qcOld = qc;
      let autoc = max(0.001 * (qc - 0.001), 0.0);
      qc = (qc - dt * autoc) / (1.0 + dt * 2.2 * pow(max(qr, 0.0), 0.875));
      qc = max(qc, 0.0);
      qr += qcOld - qc;
    }

    // --- 飽和調整（Newton 法 3 回）
    for (var it = 0; it < 3; it++) {
      let T = th * pi;
      let qs = qvsatLiq(T, p);
      if (qv <= qs && qc <= 0.0) { break; }
      let tc = T - 273.15;
      let dlnes = 17.67 * 243.5 / ((tc + 243.5) * (tc + 243.5));
      var dq = (qv - qs) / (1.0 + (LV / CP) * qs * dlnes);
      dq = max(dq, -qc);
      th += LV / (CP * pi) * dq;
      qv -= dq;
      qc += dq;
    }

    // --- 雨の蒸発（未飽和のとき）
    if (rain && qr > 0.0) {
      let T = th * pi;
      let qs = qvsatLiq(T, p);
      if (qv < qs) {
        let rhoqr = 0.001 * cl.rho0 * qr; // [g/cm³]
        let cvent = 1.6 + 124.9 * pow(rhoqr, 0.2046);
        var ern = dt * (cvent * pow(rhoqr, 0.525)) / (2.55e8 / (p * qs) + 5.4e5) * ((qs - qv) / (0.001 * cl.rho0 * qs));
        let tc = T - 273.15;
        let dlnes = 17.67 * 243.5 / ((tc + 243.5) * (tc + 243.5));
        let deficit = (qs - qv) / (1.0 + (LV / CP) * qs * dlnes);
        ern = min(min(ern, deficit), qr);
        th -= LV / (CP * pi) * ern;
        qv += ern;
        qr -= ern;
      }
    }

    sO[F_TH * P.n + id] = th;
    sO[F_QV * P.n + id] = qv;
    sO[F_QC * P.n + id] = qc;
    qrCol[k] = qr;
  }

  // --- 雨の落下（風上差分、CFL に応じて小分け）
  var precip = 0.0;
  if (rain) {
    let rhoS = col[0].rho0;
    var maxC = 0.0;
    for (var k = 0u; k < nz; k++) {
      let cl = col[k];
      let vt = 36.34 * pow(max(0.001 * cl.rho0 * qrCol[k], 0.0), 0.1364) * sqrt(rhoS / cl.rho0);
      maxC = max(maxC, vt * dt / cl.dzc);
    }
    let nsub = u32(clamp(ceil(maxC / 0.8), 1.0, 50.0));
    let dts = dt / f32(nsub);
    for (var s = 0u; s < nsub; s++) {
      var fluxAbove = 0.0; // 上のセルから流れ込む ρ qr vt
      for (var kk = i32(nz) - 1; kk >= 0; kk--) {
        let k = u32(kk);
        let cl = col[k];
        let q = qrCol[k];
        let vt = 36.34 * pow(max(0.001 * cl.rho0 * q, 0.0), 0.1364) * sqrt(rhoS / cl.rho0);
        let fluxOut = cl.rho0 * q * vt;
        qrCol[k] = max(q + dts * (fluxAbove - fluxOut) / (cl.rho0 * cl.dzc), 0.0);
        fluxAbove = fluxOut;
      }
      precip += fluxAbove * dts; // 地表に落ちた量 [kg/m²]
    }
  }
  for (var k = 0u; k < nz; k++) {
    sO[F_QR * P.n + base + plane * k] = qrCol[k];
  }
  diag[D_PRECIP * P.n + base] += precip;
  // 切り詰めた水量の累計（保存性の監視用）を diag の別領域に
  diag[D_PRECIP * P.n + plane + base] += negAcc;
}
