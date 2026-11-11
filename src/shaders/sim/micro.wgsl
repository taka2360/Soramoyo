// 柱ごとの物理過程（1スレッド = 1柱、RK3 の後に sO に対して1回適用）：
//  1. 負値の切り詰め（量は監視用に記録）
//  2. 大規模強制：沈降による移流、θ・qv の強制、基本場へのナッジング
//  3. 地表：エネルギー収支（地温の予報）から顕熱・潜熱、または固定フラックス。地表摩擦
//  4. 雲微物理（暖かい雨）：飽和調整 + Kessler（自動変換・捕捉・蒸発）
//  5. 雨の落下
// 微物理の式は Klemp & Wilhelmson (1978) / WRF の Kessler スキームに準拠。

const MAXNZ_M: u32 = 256u;
const SIGMA_SB: f32 = 5.670e-8;
const KARMAN: f32 = 0.4;

fn so(f: u32, id: u32) -> f32 { return sO[f * P.n + id]; }

@compute @workgroup_size(8, 8, 1)
fn csMicro(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= P.nx || g.y >= P.ny) { return; }
  let pl = P.nx * P.ny;
  let base = g.x + P.nx * g.y;
  let nz = P.nz;
  let dt = P.dtFull;
  let moist = (P.flags & FLAG_MOIST) != 0u;
  let rain = (P.flags & FLAG_RAIN) != 0u;

  var thC: array<f32, MAXNZ_M>;
  var qvC: array<f32, MAXNZ_M>;
  var qrCol: array<f32, MAXNZ_M>;
  var negAcc = 0.0;

  // ---- 1. 読み込みと負値の切り詰め
  var tauCloud = 0.0; // 柱の雲の光学的厚さ（地表日射用）
  for (var k = 0u; k < nz; k++) {
    let id = base + pl * k;
    let cl = col[k];
    let m = cl.rho0 * cl.dzc;
    thC[k] = so(F_TH, id);
    var qv = so(F_QV, id);
    var qc = so(F_QC, id);
    var qr = so(F_QR, id);
    if (qv < 0.0) { negAcc -= qv * m; qv = 0.0; }
    if (qc < 0.0) { negAcc -= qc * m; qc = 0.0; }
    if (qr < 0.0) { negAcc -= qr * m; qr = 0.0; }
    qvC[k] = qv;
    qrCol[k] = qr;
    sO[F_QC * P.n + id] = qc;
    let lwc = qc * cl.rho0;
    if (lwc > 1e-9) { tauCloud += 0.02198 * pow(lwc, 2.0 / 3.0) * pow(P.nDrop, 1.0 / 3.0) * cl.dzc; }
    tauCloud += 3.0 * qr * cl.rho0 * cl.dzc;
  }

  // ---- 2. 大規模強制（沈降は風上差分。下降流なら上の層の値を運んでくる）
  // 下の層から順に更新する。下降流は上の層（未更新）の値しか参照しないので正しい。
  // 上昇流（ws>0）では下の層（更新済み）を参照するが、強制は小さいので許容する。
  for (var k = 0u; k < nz; k++) {
    let cl = col[k];
    var wTop = 0.0;
    if (k + 1u < nz) { wTop = col[k + 1u].wsub; }
    let ws = 0.5 * (cl.wsub + wTop);
    var dth = 0.0; var dqv = 0.0;
    if (ws < 0.0 && k + 1u < nz) {
      let dz = col[k + 1u].dzw;
      dth = -ws * (thC[k + 1u] - thC[k]) / dz;
      dqv = -ws * (qvC[k + 1u] - qvC[k]) / dz;
    } else if (ws > 0.0 && k > 0u) {
      dth = -ws * (thC[k] - thC[k - 1u]) / cl.dzw;
      dqv = -ws * (qvC[k] - qvC[k - 1u]) / cl.dzw;
    }
    dth += cl.thLs - cl.nudge * (thC[k] - cl.th0);
    dqv += cl.qvLs - cl.nudge * (qvC[k] - cl.qv0);
    thC[k] += dt * dth;
    qvC[k] = max(qvC[k] + dt * dqv, 0.0);
  }

  // ---- 3. 地表
  let c0 = col[0];
  let id0 = base;
  // 地面に対する風（計算座標系の移動速度を足す）。u, v はセル中心へ平均
  let uE = so(F_U, g.y * P.nx + (g.x + 1u) % P.nx);
  let uw = 0.5 * (so(F_U, id0) + uE) + P.frameU;
  var vw = P.frameV;
  if (P.ny > 1u) { vw += 0.5 * (so(F_V, id0) + so(F_V, ((g.y + 1u) % P.ny) * P.nx + g.x)); }
  let speed = sqrt(uw * uw + vw * vw);
  let z1 = c0.zc;
  let cn = KARMAN / log(max(z1 / max(P.z0, 1e-4), 1.5));
  let chn = cn * cn; // 中立のバルク係数（安定度補正は未実装）
  // 対流時の弱風でも乱流で熱が運ばれるよう、実効風速に下限（自由対流の速度スケール ~1 m/s）
  let ueff = sqrt(speed * speed + 1.0);
  let t1 = thC[0] * c0.pi0;
  var H = P.shf;
  var LE = P.lhf;
  if ((P.flags & FLAG_SURF_ENERGY) != 0u) {
    var ts = diag[dSurf(2u, base)];
    // 日射：晴天値 × 雲の透過（2流近似の目安：T ≈ 1/(1 + 0.75(1−g)τ/μ)、g=0.85）
    let mu = max(P.cosZen, 0.05);
    var sw = 0.0;
    if (P.cosZen > 0.0) { sw = P.swClear / (1.0 + 0.1125 * tauCloud / mu); }
    // 下向き長波：晴天は Brutsaert (1975)、雲があれば黒体に近づける
    let e = qvC[0] * c0.p0 / (EPSV + qvC[0]) / 100.0; // 水蒸気圧 [hPa]
    let epsClear = 1.24 * pow(max(e, 0.01) / t1, 1.0 / 7.0);
    let epsAir = epsClear + (1.0 - epsClear) * (1.0 - exp(-tauCloud));
    let lwDown = epsAir * SIGMA_SB * t1 * t1 * t1 * t1;
    let lwUp = 0.98 * SIGMA_SB * ts * ts * ts * ts;
    let rn = (1.0 - P.albedo) * sw + 0.98 * lwDown - lwUp;
    let rho = c0.rho0;
    H = rho * CP * chn * ueff * (ts - t1);
    let qs = qvsatLiq(ts, c0.p0);
    if (qs >= qvC[0]) {
      // 植物の気孔は暗いと閉じる（Jarvis 型の日射応答を簡略化）。夜の蒸散を抑える
      let beta = P.soilWet * (0.1 + 0.9 * clamp(sw / 400.0, 0.0, 1.0));
      LE = rho * LV * chn * ueff * beta * (qs - qvC[0]);
    } else {
      LE = rho * LV * chn * ueff * (qs - qvC[0]); // 結露（露）
    }
    let gflux = rn - H - LE;
    // force-restore 法（Deardorff 1978）
    ts += dt * (gflux / P.cSoil - (6.2831853 / P.tauSoil) * (ts - P.tDeep));
    diag[dSurf(2u, base)] = ts;
    diag[dSurf(5u, base)] = sw;
  }
  diag[dSurf(3u, base)] = H;
  diag[dSurf(4u, base)] = LE;
  thC[0] += dt * H / (c0.rho0 * CP * c0.pi0 * c0.dzc);
  if (moist) { qvC[0] = max(qvC[0] + dt * LE / (c0.rho0 * LV * c0.dzc), 0.0); }

  // 地表摩擦（最下層の u, v を減速。陰的）
  var us2: f32;
  if (P.ustar > 0.0) { us2 = P.ustar * P.ustar; } else { us2 = chn * speed * speed; }
  if (speed > 1e-3 && P.z0 > 0.0) { // z0 = 0 は摩擦なし（自由すべり）
    let k0 = 1.0 / (1.0 + dt * us2 / (speed * c0.dzc));
    sO[F_U * P.n + id0] = (so(F_U, id0) + P.frameU) * k0 - P.frameU;
    if (P.ny > 1u) { sO[F_V * P.n + id0] = (so(F_V, id0) + P.frameV) * k0 - P.frameV; }
  }

  // ---- 4. 雲微物理
  for (var k = 0u; k < nz; k++) {
    let id = base + pl * k;
    let cl = col[k];
    var th = thC[k];
    var qv = qvC[k];
    var qc = so(F_QC, id);
    var qr = qrCol[k];
    let p = cl.p0;
    let pi = cl.pi0;

    if (moist) {
      if (rain) {
        let qcOld = qc;
        let autoc = max(0.001 * (qc - 0.001), 0.0);
        qc = (qc - dt * autoc) / (1.0 + dt * 2.2 * pow(max(qr, 0.0), 0.875));
        qc = max(qc, 0.0);
        qr += qcOld - qc;
      }
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
    }
    sO[F_TH * P.n + id] = th;
    sO[F_QV * P.n + id] = qv;
    sO[F_QC * P.n + id] = qc;
    qrCol[k] = qr;
  }

  // ---- 5. 雨の落下（風上差分、CFL に応じて小分け）
  var precip = 0.0;
  if (moist && rain) {
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
      var fluxAbove = 0.0;
      for (var kk = i32(nz) - 1; kk >= 0; kk--) {
        let k = u32(kk);
        let cl = col[k];
        let q = qrCol[k];
        let vt = 36.34 * pow(max(0.001 * cl.rho0 * q, 0.0), 0.1364) * sqrt(rhoS / cl.rho0);
        let fluxOut = cl.rho0 * q * vt;
        qrCol[k] = max(q + dts * (fluxAbove - fluxOut) / (cl.rho0 * cl.dzc), 0.0);
        fluxAbove = fluxOut;
      }
      precip += fluxAbove * dts;
    }
  }
  for (var k = 0u; k < nz; k++) {
    sO[F_QR * P.n + base + pl * k] = qrCol[k];
  }
  diag[dSurf(0u, base)] += precip;
  diag[dSurf(1u, base)] += negAcc;
}
