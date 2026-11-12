// 柱ごとの物理過程（1スレッド = 1柱、RK3 の後に sO に対して1回適用）：
//  1. 負値の切り詰め（量は監視用に記録）
//  2. 大規模強制：沈降による移流、θ・qv の強制、基本場へのナッジング
//  3. 地表：エネルギー収支（地温の予報）から顕熱・潜熱、または固定フラックス。地表摩擦
//  4. 雲の長波放射（雲頂の冷却）
//  5. 雲微物理（混合相の1モーメント）：飽和調整、自動変換・捕捉・蒸発（雨は Kessler / KW78）
//  6. 降水の落下（雨と雪）

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

  // ---- 4. 長波放射（雲頂での冷却、雲底での加熱）
  // Stevens et al. (2005) の DYCOMS-II の式：F(z) = F0·exp(−Q(z,∞)) + F1·exp(−Q(0,z)) [+ 第3項]
  //   Q = κ ∫ρ q_n dz（κ = 85 m²/kg）。雲が厚ければ雲頂で約 70 W/m² の冷却になる。
  // 雲の無いところの冷却（晴天の放射冷却）は大規模強制 thLs で別に与える。
  if (moist && (P.flags & FLAG_LW) != 0u) {
    var qAbove: array<f32, MAXNZ_M>;
    var acc = 0.0;
    for (var kk = i32(nz) - 1; kk >= 0; kk--) {
      let k = u32(kk);
      qAbove[k] = acc; // 上面より上の積算
      acc += 85.0 * col[k].rho0 * so(F_QC, base + pl * k) * col[k].dzc;
    }
    let qTotal = acc;
    // 逆転層の高さ（DYCOMS 用）：qt が 8 g/kg を下回る最初の高さ
    var zi = 1e9;
    if ((P.flags & FLAG_DYCOMS) != 0u) {
      for (var k = 0u; k < nz; k++) {
        if (qvC[k] + so(F_QC, base + pl * k) < 0.008) { zi = col[k].zf; break; }
      }
    }
    var below = 0.0;
    var fPrev = 70.0 * exp(-qTotal) + 22.0; // 地表面での正味の上向きフラックス
    for (var k = 0u; k < nz; k++) {
      let cl = col[k];
      below += 85.0 * cl.rho0 * so(F_QC, base + pl * k) * cl.dzc;
      let zTop = cl.zf + cl.dzc;
      var fTop = 70.0 * exp(-qAbove[k]) + 22.0 * exp(-below);
      if (zTop > zi) {
        let dz = zTop - zi;
        fTop += 1.12 * CP * 3.75e-6 * (0.25 * pow(dz, 4.0 / 3.0) + zi * pow(dz, 1.0 / 3.0));
      }
      thC[k] -= dt * (fTop - fPrev) / (cl.rho0 * CP * cl.pi0 * cl.dzc);
      fPrev = fTop;
    }
  }

  // ---- 5. 雲微物理（1モーメント、混合相。SAM と同じ考え方）
  // 雲の凝結物 qn（= 雲水 + 雲氷）と降水 qp（= 雨 + 雪）を予報し、水と氷の割合は温度で決める。
  // 相変化の熱は「氷・液水温位」θil を保存したまま温度を解き直すことで、凍結・融解の熱も含めて自動で入る。
  let ice = (P.flags & FLAG_ICE) != 0u;
  for (var k = 0u; k < nz; k++) {
    let id = base + pl * k;
    let cl = col[k];
    var th = thC[k];
    var qv = qvC[k];
    var qn = so(F_QC, id);
    var qp = qrCol[k];
    let p = cl.p0;
    let pi = cl.pi0;

    if (moist) {
      let T0 = th * pi;
      var wn = 1.0; var wp = 1.0;
      if (ice) { wn = liqFracCloud(T0); wp = liqFracPrecip(T0); }
      // 保存量 θil（このステップの相変化で変わらない）
      let thil = th - (LV * (wn * qn + wp * qp) + LS * ((1.0 - wn) * qn + (1.0 - wp) * qp)) / (CP * pi);

      if (rain) {
        // 自動変換：雲水 → 雨（Kessler）、雲氷 → 雪（併合、温度依存。Lin et al. 1983 の形）
        let ql = wn * qn;
        let qi = (1.0 - wn) * qn;
        let autL = 0.001 * max(ql - 0.001, 0.0);
        let autI = 0.001 * exp(0.025 * (T0 - 273.16)) * max(qi - 1e-4, 0.0);
        // 捕捉：降水が雲粒を集める（雨は Kessler、雪は効率を下げた同じ形）
        let effAcc = wp + (1.0 - wp) * 0.5;
        let rate = autL + autI;
        let qnOld = qn;
        qn = (qn - dt * rate) / (1.0 + dt * 2.2 * effAcc * pow(max(qp, 0.0), 0.875));
        qn = max(qn, 0.0);
        qp += qnOld - qn;

        // 降水の蒸発・昇華（未飽和のとき）。雨は Klemp–Wilhelmson、雪は同じ形を半分の効率で
        if (qp > 0.0) {
          let qsl = qvsatLiq(T0, p);
          let qsi = qvsatIce(T0, p);
          let qsm = wp * qsl + (1.0 - wp) * qsi;
          if (qv < qsm) {
            let rhoqp = 0.001 * cl.rho0 * qp;
            let cvent = 1.6 + 124.9 * pow(rhoqp, 0.2046);
            var ern = dt * (cvent * pow(rhoqp, 0.525)) / (2.55e8 / (p * qsm) + 5.4e5) * ((qsm - qv) / (0.001 * cl.rho0 * qsm));
            ern *= wp + (1.0 - wp) * 0.5;
            ern = min(min(ern, 0.5 * (qsm - qv)), qp);
            qv += ern;
            qp -= ern;
          }
        }
      }

      // 飽和調整：qt_n = qv + qn と θil を保ったまま、qv = 混合相の飽和混合比 となる温度を Newton 法で解く
      let qtn = qv + qn;
      let tl = thil * pi;
      var T = T0;
      for (var it = 0; it < 4; it++) {
        var f0: f32; var f1: f32;
        for (var j = 0; j < 2; j++) {
          let Tj = T + f32(j) * 0.05;
          var a = 1.0; var b = 1.0;
          if (ice) { a = liqFracCloud(Tj); b = liqFracPrecip(Tj); }
          let qs = a * qvsatLiq(Tj, p) + (1.0 - a) * qvsatIce(Tj, p);
          let qnj = max(qtn - qs, 0.0);
          let fj = Tj - tl - (LV * (a * qnj + b * qp) + LS * ((1.0 - a) * qnj + (1.0 - b) * qp)) / CP;
          if (j == 0) { f0 = fj; } else { f1 = fj; }
        }
        let d = (f1 - f0) / 0.05;
        T -= f0 / max(d, 0.1);
      }
      var a = 1.0;
      if (ice) { a = liqFracCloud(T); }
      let qs = a * qvsatLiq(T, p) + (1.0 - a) * qvsatIce(T, p);
      qn = max(qtn - qs, 0.0);
      qv = qtn - qn;
      th = T / pi;
    }
    sO[F_TH * P.n + id] = th;
    sO[F_QV * P.n + id] = qv;
    sO[F_QC * P.n + id] = qn;
    qrCol[k] = qp;
  }

  // ---- 6. 降水の落下（風上差分、CFL に応じて小分け）。雨と雪の割合で落下速度を混ぜる
  var precip = 0.0;
  if (moist && rain) {
    let rhoS = col[0].rho0;
    var vtC: array<f32, MAXNZ_M>;
    var maxC = 0.0;
    for (var k = 0u; k < nz; k++) {
      let cl = col[k];
      let rq = max(0.001 * cl.rho0 * qrCol[k], 0.0);
      let vr = 36.34 * pow(rq, 0.1364) * sqrt(rhoS / cl.rho0);
      var v = vr;
      if (ice) {
        let wp = liqFracPrecip(sO[F_TH * P.n + base + pl * k] * cl.pi0);
        // 雪（あられを含めた平均的な値）：おおむね 0.5〜2 m/s
        let vs = clamp(3.0 * pow(rq * 1000.0, 0.11), 0.3, 2.5) * sqrt(rhoS / cl.rho0);
        v = wp * vr + (1.0 - wp) * vs;
      }
      vtC[k] = v;
      maxC = max(maxC, v * dt / cl.dzc);
    }
    let nsub = u32(clamp(ceil(maxC / 0.8), 1.0, 50.0));
    let dts = dt / f32(nsub);
    for (var s = 0u; s < nsub; s++) {
      var fluxAbove = 0.0;
      for (var kk = i32(nz) - 1; kk >= 0; kk--) {
        let k = u32(kk);
        let cl = col[k];
        let q = qrCol[k];
        let fluxOut = cl.rho0 * q * vtC[k];
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
