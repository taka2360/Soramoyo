// RK3 の後に sO に対して1回適用する物理過程。スレッド内に柱の配列を持たない（ローカルメモリに溢れて遅い）よう、3つのカーネルに分ける。
//  csColumnPre（1スレッド = 1柱）：
//    1. 負値の切り詰め（量は監視用に記録）
//    2. 大規模強制：沈降による移流、θ・qv の強制、基本場へのナッジング
//    3. 地表：エネルギー収支（地温の予報）から顕熱・潜熱、または固定フラックス。地表摩擦
//    4. 雲の長波放射（雲頂の冷却）
//  csMicroCell（1スレッド = 1セル）：
//    5. 雲微物理（混合相の1モーメント）：飽和調整、自動変換・捕捉・蒸発（雨は Kessler / KW78）、あられ（着氷・融解）
//  csSediment（1スレッド = 1柱）：
//    6. 降水の落下（雨と雪、あられ）。「柱に凝結物あり」の印と地表の降水の積算
// 途中の値はすべて sO に書き戻して次のカーネルに渡す。

const SIGMA_SB: f32 = 5.670e-8;
const KARMAN: f32 = 0.4;

fn so(f: u32, id: u32) -> f32 { return sO[f * P.n + id]; }
fn soSet(f: u32, id: u32, v: f32) { sO[f * P.n + id] = v; }

@compute @workgroup_size(8, 8, 1)
fn csColumnPre(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= P.nx || g.y >= P.ny) { return; }
  let pl = P.nx * P.ny;
  let base = g.x + P.nx * g.y;
  let nz = P.nz;
  let dt = P.dtFull;
  let moist = (P.flags & FLAG_MOIST) != 0u;

  // ---- 1. 負値の切り詰め
  var negAcc = 0.0;
  var tauCloud = 0.0; // 柱の雲の光学的厚さ（地表日射用）
  var qTotal = 0.0;   // 長波放射の柱全体の κ∫ρ q_n dz
  for (var k = 0u; k < nz; k++) {
    let id = base + pl * k;
    let cl = col[k];
    let m = cl.rho0 * cl.dzc;
    var qv = so(F_QV, id);
    var qc = so(F_QC, id);
    var qr = so(F_QR, id);
    var qg = so(F_QG, id);
    if (qv < 0.0) { negAcc -= qv * m; qv = 0.0; soSet(F_QV, id, qv); }
    if (qc < 0.0) { negAcc -= qc * m; qc = 0.0; soSet(F_QC, id, qc); }
    if (qr < 0.0) { negAcc -= qr * m; qr = 0.0; soSet(F_QR, id, qr); }
    if (qg < 0.0) { negAcc -= qg * m; qg = 0.0; soSet(F_QG, id, qg); }
    let lwc = qc * cl.rho0;
    if (lwc > 1e-9) { tauCloud += 0.02198 * pow(lwc, 2.0 / 3.0) * pow(P.nDrop, 1.0 / 3.0) * cl.dzc; }
    tauCloud += 3.0 * (qr + qg) * cl.rho0 * cl.dzc;
    qTotal += 85.0 * cl.rho0 * qc * cl.dzc;
  }

  // ---- 2. 大規模強制（沈降は風上差分。下降流なら上の層の値を運んでくる）
  // 下の層から順に更新する。下降流は上の層（未更新）の値しか参照しないので正しい。
  // 上昇流（ws>0）では下の層（更新済み、レジスタに持ち越し）を参照するが、強制は小さいので許容する。
  var thPrev = 0.0; var qvPrev = 0.0;      // 更新済みの k−1
  var thCur = so(F_TH, base); var qvCur = so(F_QV, base);
  for (var k = 0u; k < nz; k++) {
    let cl = col[k];
    var thNext = 0.0; var qvNext = 0.0;    // 未更新の k+1
    var wTop = 0.0;
    if (k + 1u < nz) {
      wTop = col[k + 1u].wsub;
      thNext = so(F_TH, base + pl * (k + 1u));
      qvNext = so(F_QV, base + pl * (k + 1u));
    }
    let ws = 0.5 * (cl.wsub + wTop);
    var dth = 0.0; var dqv = 0.0;
    if (ws < 0.0 && k + 1u < nz) {
      let dz = col[k + 1u].dzw;
      dth = -ws * (thNext - thCur) / dz;
      dqv = -ws * (qvNext - qvCur) / dz;
    } else if (ws > 0.0 && k > 0u) {
      dth = -ws * (thCur - thPrev) / cl.dzw;
      dqv = -ws * (qvCur - qvPrev) / cl.dzw;
    }
    dth += cl.thLs - cl.nudge * (thCur - cl.th0);
    dqv += cl.qvLs - cl.nudge * (qvCur - cl.qv0);
    thPrev = thCur + dt * dth;
    qvPrev = max(qvCur + dt * dqv, 0.0);
    let id = base + pl * k;
    soSet(F_TH, id, thPrev);
    soSet(F_QV, id, qvPrev);
    thCur = thNext; qvCur = qvNext;
  }

  // ---- 3. 地表
  let c0 = col[0];
  let id0 = base;
  var th0 = so(F_TH, id0);
  var qv0 = so(F_QV, id0);
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
  let t1 = th0 * c0.pi0;
  var H = P.shf;
  var LE = P.lhf;
  if ((P.flags & FLAG_SURF_ENERGY) != 0u) {
    var ts = diag[dSurf(2u, base)];
    // 日射：晴天値 × 雲の透過（2流近似の目安：T ≈ 1/(1 + 0.75(1−g)τ/μ)、g=0.85）
    let mu = max(P.cosZen, 0.05);
    var sw = 0.0;
    if (P.cosZen > 0.0) { sw = P.swClear / (1.0 + 0.1125 * tauCloud / mu); }
    // 下向き長波：晴天は Brutsaert (1975)、雲があれば黒体に近づける
    let e = qv0 * c0.p0 / (EPSV + qv0) / 100.0; // 水蒸気圧 [hPa]
    let epsClear = 1.24 * pow(max(e, 0.01) / t1, 1.0 / 7.0);
    let epsAir = epsClear + (1.0 - epsClear) * (1.0 - exp(-tauCloud));
    let lwDown = epsAir * SIGMA_SB * t1 * t1 * t1 * t1;
    let lwUp = 0.98 * SIGMA_SB * ts * ts * ts * ts;
    let rn = (1.0 - P.albedo) * sw + 0.98 * lwDown - lwUp;
    let rho = c0.rho0;
    H = rho * CP * chn * ueff * (ts - t1);
    let qs = qvsatLiq(ts, c0.p0);
    if (qs >= qv0) {
      // 植物の気孔は暗いと閉じる（Jarvis 型の日射応答を簡略化）。夜の蒸散を抑える
      let beta = P.soilWet * (0.1 + 0.9 * clamp(sw / 400.0, 0.0, 1.0));
      LE = rho * LV * chn * ueff * beta * (qs - qv0);
    } else {
      LE = rho * LV * chn * ueff * (qs - qv0); // 結露（露）
    }
    let gflux = rn - H - LE;
    // force-restore 法（Deardorff 1978）
    ts += dt * (gflux / P.cSoil - (6.2831853 / P.tauSoil) * (ts - P.tDeep));
    diag[dSurf(2u, base)] = ts;
    diag[dSurf(5u, base)] = sw;
  }
  diag[dSurf(3u, base)] = H;
  diag[dSurf(4u, base)] = LE;
  th0 += dt * H / (c0.rho0 * CP * c0.pi0 * c0.dzc);
  soSet(F_TH, id0, th0);
  if (moist) { qv0 = max(qv0 + dt * LE / (c0.rho0 * LV * c0.dzc), 0.0); soSet(F_QV, id0, qv0); }

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
    // 逆転層の高さ（DYCOMS 用）：qt が 8 g/kg を下回る最初の高さ
    var zi = 1e9;
    if ((P.flags & FLAG_DYCOMS) != 0u) {
      for (var k = 0u; k < nz; k++) {
        let id = base + pl * k;
        if (so(F_QV, id) + so(F_QC, id) < 0.008) { zi = col[k].zf; break; }
      }
    }
    var below = 0.0;
    var fPrev = 70.0 * exp(-qTotal) + 22.0; // 地表面での正味の上向きフラックス
    // 雲の無い柱はフラックスが高さによらず一定（加熱・冷却 0）なので省く（DYCOMS の第3項がある場合を除く）
    let lwLoops = qTotal > 0.0 || zi < 1e8;
    for (var k = 0u; k < select(0u, nz, lwLoops); k++) {
      let cl = col[k];
      let id = base + pl * k;
      below += 85.0 * cl.rho0 * so(F_QC, id) * cl.dzc;
      let qAbove = max(qTotal - below, 0.0); // 上面より上の積算
      let zTop = cl.zf + cl.dzc;
      var fTop = 70.0 * exp(-qAbove) + 22.0 * exp(-below);
      if (zTop > zi) {
        let dz = zTop - zi;
        fTop += 1.12 * CP * 3.75e-6 * (0.25 * pow(dz, 4.0 / 3.0) + zi * pow(dz, 1.0 / 3.0));
      }
      soSet(F_TH, id, so(F_TH, id) - dt * (fTop - fPrev) / (cl.rho0 * CP * cl.pi0 * cl.dzc));
      fPrev = fTop;
    }
  }
  diag[dSurf(1u, base)] += negAcc;
}

// ---- 5. 雲微物理（1モーメント、混合相。SAM と同じ考え方）
// 雲の凝結物 qn（= 雲水 + 雲氷）と降水 qp（= 雨 + 雪）を予報し、水と氷の割合は温度で決める。
// 相変化の熱は「氷・液水温位」θil を保存したまま温度を解き直すことで、凍結・融解の熱も含めて自動で入る。
@compute @workgroup_size(8, 8, 1)
fn csMicroCell(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= P.nx || g.y >= P.ny || g.z >= P.nz) { return; }
  if ((P.flags & FLAG_MOIST) == 0u) { return; }
  let k = g.z;
  let id = g.x + P.nx * (g.y + P.ny * k);
  let dt = P.dtFull;
  let rain = (P.flags & FLAG_RAIN) != 0u;
  let ice = (P.flags & FLAG_ICE) != 0u;
  let cl = col[k];
  var th = so(F_TH, id);
  var qv = so(F_QV, id);
  var qn = so(F_QC, id);
  var qp = so(F_QR, id);
  var qg = so(F_QG, id);
  let p = cl.p0;
  let pi = cl.pi0;

  // 凝結物が無く未飽和なら相変化は起きない（飽和調整の Newton 法も元の温度に収束する）ので省く。
  // 晴れた格子が大半なので、これで微物理の計算量が大きく減る。
  if (qn == 0.0 && qp == 0.0 && qg == 0.0) {
    let T0 = th * pi;
    var a0 = 1.0;
    if (ice) { a0 = liqFracCloud(T0); }
    var qs0 = qvsatLiq(T0, p);
    if (a0 < 1.0) { qs0 = a0 * qs0 + (1.0 - a0) * qvsatIce(T0, p); }
    if (qv < qs0) { return; }
  }

  let T0 = th * pi;
  var wn = 1.0; var wp = 1.0;
  if (ice) { wn = liqFracCloud(T0); wp = liqFracPrecip(T0); }
  // 保存量 θil（このステップの相変化で変わらない）。あられ qg は常に氷
  let thil = th - (LV * (wn * qn + wp * qp) + LS * ((1.0 - wn) * qn + (1.0 - wp) * qp + qg)) / (CP * pi);

  if (rain) {
    // 自動変換：雲水 → 雨（Kessler）、雲氷 → 雪（併合、温度依存。Lin et al. 1983 の形）
    let ql = wn * qn;
    let qi = (1.0 - wn) * qn;
    let autL = 0.001 * max(ql - 0.001, 0.0);
    let autI = 0.001 * exp(0.025 * (T0 - 273.16)) * max(qi - 1e-4, 0.0);
    // 捕捉：降水（雨・雪）とあられが雲粒を集める（陰的に解く）。
    //  降水：Kessler 形 2.2 qp^0.875。雲氷に対する効率は雪で exp(0.025(T−T0))
    //  あられ：Lin 型（N0 = 4×10⁵ m⁻⁴, 密度 500 kg/m³）を同じ形に直すと 2.2 ρ^0.375 qg^0.875。雲氷への効率 0.1
    let cP = 2.2 * pow(max(qp, 0.0), 0.875);
    let cG = 2.2 * pow(cl.rho0, 0.375) * pow(max(qg, 0.0), 0.875);
    let effI = wp + (1.0 - wp) * exp(0.025 * min(T0 - 273.16, 0.0));  // 雪の雲氷に対する効率（Lin et al. 1983）
    let aL = min(dt * autL, ql);
    let aI = min(dt * autI, qi);
    let qlN = max((ql - aL) / (1.0 + dt * (cP + cG)), 0.0);
    let qiN = max((qi - aI) / (1.0 + dt * (cP * effI + 0.1 * cG)), 0.0);
    let colL = max(ql - aL - qlN, 0.0);
    let colI = max(qi - aI - qiN, 0.0);
    // 凍った降水（雪）やあられが過冷却の雲水を集める（着氷）と、その分はあられになる
    let toGL = colL * (cG + (1.0 - wp) * cP) / max(cP + cG, 1e-12);
    let toGI = colI * (0.1 * cG) / max(cP * effI + 0.1 * cG, 1e-12);
    qn = qlN + qiN;
    qg += toGL + toGI;
    qp += aL + aI + (colL - toGL) + (colI - toGI);

    if (ice) {
      // 雪 → あられ：過冷却の雲水の中で強く着氷した雪・持ち上げられて凍った雨はあられになる
      //  （1モーメントでは着氷の度合いを持たないので、雲水量に比例する時定数 300 s で近似）
      //  + Lin et al. (1983) の雪の併合による変換 1e-3 exp(0.09(T−T0)) (qs − 0.6 g/kg)
      let qs = (1.0 - wp) * qp;
      if (qs > 0.0 && T0 < 273.16) {
        let rRime = min(qlN / 1e-3, 1.0) / 300.0;
        let rAgg = 1e-3 * exp(0.09 * (T0 - 273.16)) * max(qs - 6e-4, 0.0) / qs;
        let dq = qs * (1.0 - exp(-dt * (rRime + rAgg)));
        qp -= dq;
        qg += dq;
      }
      // あられの融解：0℃ より暖かいと溶けて降水（雨）に（時定数 1000 s/K。融解熱は θil の保存で入る）
      if (qg > 0.0 && T0 > 273.16) {
        let dm = qg * (1.0 - exp(-dt * 1e-3 * (T0 - 273.16)));
        qg -= dm;
        qp += dm;
      }
    }

    // 降水の蒸発・昇華（未飽和のとき）。雨は Klemp–Wilhelmson、雪は同じ形を半分の効率で
    let qsl = qvsatLiq(T0, p);
    let qsi = qvsatIce(T0, p);
    if (qp > 0.0) {
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
    // あられの昇華（粒が大きく数が少ないので、同じ形を 0.3 倍の効率で）
    if (qg > 0.0 && qv < qsi) {
      let rhoqg = 0.001 * cl.rho0 * qg;
      let cvent = 1.6 + 124.9 * pow(rhoqg, 0.2046);
      var eg = dt * 0.3 * (cvent * pow(rhoqg, 0.525)) / (2.55e8 / (p * qsi) + 5.4e5) * ((qsi - qv) / (0.001 * cl.rho0 * qsi));
      eg = min(min(eg, 0.5 * (qsi - qv)), qg);
      qv += eg;
      qg -= eg;
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
      let fj = Tj - tl - (LV * (a * qnj + b * qp) + LS * ((1.0 - a) * qnj + (1.0 - b) * qp + qg)) / CP;
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

  soSet(F_TH, id, th);
  soSet(F_QV, id, qv);
  soSet(F_QC, id, qn);
  soSet(F_QR, id, qp);
  soSet(F_QG, id, qg);
}

// ---- 6. 降水の落下（風上差分、CFL に応じて小分け）。雨と雪の割合で落下速度を混ぜる
// 落下速度は小分けの前の値で固定し、cplx（圧力の計算が終わって空いている）に (雨・雪, あられ) を置く。
@compute @workgroup_size(8, 8, 1)
fn csSediment(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= P.nx || g.y >= P.ny) { return; }
  let pl = P.nx * P.ny;
  let base = g.x + P.nx * g.y;
  let nz = P.nz;
  let dt = P.dtFull;
  let moist = (P.flags & FLAG_MOIST) != 0u;
  let rain = (P.flags & FLAG_RAIN) != 0u;
  let ice = (P.flags & FLAG_ICE) != 0u;

  var precip = 0.0;
  var anyPrecip = false;
  for (var k = 0u; k < nz; k++) {
    let id = base + pl * k;
    if (so(F_QR, id) > 0.0 || so(F_QG, id) > 0.0) { anyPrecip = true; break; }
  }
  if (moist && rain && anyPrecip) {
    let rhoS = col[0].rho0;
    var maxC = 0.0;
    for (var k = 0u; k < nz; k++) {
      let cl = col[k];
      let id = base + pl * k;
      let rq = max(0.001 * cl.rho0 * so(F_QR, id), 0.0);
      let vr = 36.34 * pow(rq, 0.1364) * sqrt(rhoS / cl.rho0);
      var v = vr;
      if (ice) {
        let wp = liqFracPrecip(so(F_TH, id) * cl.pi0);
        // 雪（あられを含めた平均的な値）：おおむね 0.5〜2 m/s
        let vs = clamp(3.0 * pow(rq * 1000.0, 0.11), 0.3, 2.5) * sqrt(rhoS / cl.rho0);
        v = wp * vr + (1.0 - wp) * vs;
      }
      maxC = max(maxC, v * dt / cl.dzc);
      // あられ・ひょう：V = 6.8 m/s ×(ρqg / 1 g m⁻³)^(1/8) ×(1 kg m⁻³ / ρ)^(1/2)
      //  （N0 = 4×10⁵ m⁻⁴, 密度 500 kg/m³, 抗力係数 0.6 の指数分布の質量加重平均。1〜15 m/s に制限）
      let gq = max(cl.rho0 * so(F_QG, id), 0.0);
      let vg = clamp(6.8 * pow(gq * 1000.0, 0.125) / sqrt(cl.rho0), 1.0, 15.0);
      if (gq > 0.0) { maxC = max(maxC, vg * dt / cl.dzc); }
      cplx[id] = vec2f(v, vg);
    }
    let nsub = u32(clamp(ceil(maxC / 0.8), 1.0, 50.0));
    let dts = dt / f32(nsub);
    for (var s = 0u; s < nsub; s++) {
      var fluxAbove = 0.0;
      var fluxAboveG = 0.0;
      for (var kk = i32(nz) - 1; kk >= 0; kk--) {
        let k = u32(kk);
        let cl = col[k];
        let id = base + pl * k;
        let vt = cplx[id];
        let q = so(F_QR, id);
        let fluxOut = cl.rho0 * q * vt.x;
        soSet(F_QR, id, max(q + dts * (fluxAbove - fluxOut) / (cl.rho0 * cl.dzc), 0.0));
        fluxAbove = fluxOut;
        let qgk = so(F_QG, id);
        let fluxOutG = cl.rho0 * qgk * vt.y;
        soSet(F_QG, id, max(qgk + dts * (fluxAboveG - fluxOutG) / (cl.rho0 * cl.dzc), 0.0));
        fluxAboveG = fluxOutG;
      }
      precip += (fluxAbove + fluxAboveG) * dts;
    }
  }
  var anyCond = false;
  for (var k = 0u; k < nz; k++) {
    let id = base + pl * k;
    if (so(F_QR, id) > 0.0 || so(F_QG, id) > 0.0 || so(F_QC, id) > 0.0) { anyCond = true; break; }
  }
  diag[dSurf(0u, base)] += precip;
  diag[dSurf(6u, base)] = select(0.0, 1.0, anyCond);
}
