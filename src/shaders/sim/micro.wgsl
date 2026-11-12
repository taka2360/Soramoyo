// RK3 の後に sO に対して1回適用する物理過程。スレッド内に柱の配列を持たない（ローカルメモリに溢れて遅い）よう、4つのカーネルに分ける。
//  csRadiation（1スレッド = 太陽方向に傾いた1本の光線。地表の柱 (i, j) に届く光線）：
//    雲の光学的厚さを太陽方向に傾けて積算し、地表の日射と、雲の日射加熱（cplx.x に置く）を求める
//  csColumnPre（1スレッド = 1柱）：
//    1. 負値の切り詰め（量は監視用に記録）
//    2. 大規模強制：沈降による移流、θ・qv の強制、基本場へのナッジング（入った水を記録）
//    3. 地表：安定度補正つきのバルク式、気孔抵抗、エネルギー収支（地温の予報）、地表摩擦
//    4. 雲の長波放射（雲頂の冷却）
//  csMicroCell（1スレッド = 1セル）：
//    5. 雲微物理（1モーメント、6種類：水蒸気・雲水・雲氷・雨・雪・あられ）
//  csSediment（1スレッド = 1柱）：
//    6. 雲氷・雨・雪・あられの落下。「柱に凝結物あり」の印と地表の降水の積算
// 途中の値はすべて sO（と cplx・diag）に書き戻して次のカーネルに渡す。
// どのカーネルも、他のスレッドが同じカーネルの中で書き換える値は読まない（結果が実行ごとに変わらないように）。

const SIGMA_SB: f32 = 5.670e-8;
const KARMAN: f32 = 0.4;
const RV: f32 = 461.5;
const RHO_W: f32 = 1000.0;
const KA: f32 = 2.43e-2;    // 空気の熱伝導率 [W/m/K]
const CW: f32 = 4187.0;     // 水の比熱 [J/kg/K]
const MU_AIR: f32 = 1.718e-5; // 空気の粘性係数 [kg/m/s]
// 長波の質量吸収係数 [m²/kg]：雲水は DYCOMS-II の値、雲氷は Ebert & Curry (1992)（有効径 ~50 µm）
const KAPPA_LW_LIQ: f32 = 85.0;
const KAPPA_LW_ICE: f32 = 50.0;
// 雲の日射吸収（近赤外）：厚い雲で入射の A_INF、吸収は雲頂から光学的厚さ TAU_ABS 程度に集中
const SW_A_INF: f32 = 0.1;
const SW_TAU_ABS: f32 = 5.0;
const G_ASYM: f32 = 0.85;   // 雲粒の非対称因子

fn so(f: u32, id: u32) -> f32 { return sO[f * P.n + id]; }
fn soSet(f: u32, id: u32, v: f32) { sO[f * P.n + id] = v; }

// ---------------------------------------------------------------- 光学的厚さ（可視）
// 雲水は雲粒数濃度から有効半径を決める（描画の sigmaCloud と同じ）。雲氷は有効半径 30 µm、
// 雨・あられは 0.5 mm、雪は見通しを悪くする効果として雨の 5 倍（描画と同じ）。
fn cellTau(id: u32, k: u32) -> f32 {
  let cl = col[k];
  let rho = cl.rho0;
  let lwc = max(so(F_QC, id), 0.0) * rho;
  var s = 0.0;
  if (lwc > 1e-9) { s += 0.02198 * pow(lwc, 2.0 / 3.0) * pow(P.nDrop, 1.0 / 3.0); }
  s += rho * (54.5 * max(so(F_QI, id), 0.0) + 3.0 * (max(so(F_QR, id), 0.0) + max(so(F_QG, id), 0.0)) + 15.0 * max(so(F_QS, id), 0.0));
  return s * cl.dzc;
}

// δ-Eddington（散乱のみ、Shettle & Weinman 1970）の雲層の反射率
fn cloudReflectance(tau: f32, mu: f32) -> f32 {
  if (tau <= 0.0) { return 0.0; }
  let t1 = (1.0 - G_ASYM) * tau;
  let tauD = (1.0 - G_ASYM * G_ASYM) * tau; // δ スケーリングした光学的厚さ
  let r = (t1 + (2.0 / 3.0 - mu) * (1.0 - exp(-tauD / mu))) / (4.0 / 3.0 + t1);
  return clamp(r, 0.0, 1.0);
}
// 雲頂から斜めの光学的厚さ tauS までに吸収される割合（入射に対する）
fn cloudAbsorbed(tauS: f32) -> f32 { return SW_A_INF * (1.0 - exp(-tauS / SW_TAU_ABS)); }

// ============================================================ 日射（太陽方向に傾けた光線）
// 地表の柱 (i, j) から太陽の方向へ伸ばした光線が、高さ zc_k で通る柱は (i + round(zc_k tanZ cosA / dx), …)。
// 高さごとに一定のずれなので、光線と（柱, 高さ）は1対1に対応し、各セルにはちょうど1本の光線が書き込む。
@compute @workgroup_size(8, 8, 1)
fn csRadiation(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= P.nx || g.y >= P.ny) { return; }
  let pl = P.nx * P.ny;
  let base = g.x + P.nx * g.y;
  let nz = P.nz;
  let S = P.swClear;
  if (P.cosZen <= 0.0 || S <= 0.0) {
    for (var k = 0u; k < nz; k++) { cplx[base + pl * k] = vec2f(0.0); }
    diag[dSurf(5u, base)] = 0.0;
    return;
  }
  let mu = max(P.cosZen, 0.05);
  let tanZ = sqrt(max(1.0 - mu * mu, 0.0)) / mu;
  let hl = sqrt(P.sunX * P.sunX + P.sunY * P.sunY);
  var dir = vec2f(0.0);
  if (hl > 1e-6) { dir = vec2f(P.sunX, P.sunY) / hl; }
  // 1回目：光線上のセルの光学的厚さを cplx.y に置き、全体を積算
  var tauTot = 0.0;
  for (var k = 0u; k < nz; k++) {
    let off = col[k].zc * tanZ * dir;
    let ci = wrapX(i32(g.x) + i32(round(off.x / P.dx)));
    var cj = 0u;
    if (P.ny > 1u) { cj = wrapY(i32(g.y) + i32(round(off.y / P.dy))); }
    let cId = ci + P.nx * cj;
    let id = cId + pl * k;
    var t = 0.0;
    if (diag[dSurf(7u, cId)] > 0.0) { t = cellTau(id, k); }
    cplx[id] = vec2f(0.0, t);
    tauTot += t;
  }
  // 2回目：上から順に、吸収の増分をそのセルの加熱 [K/s] にする
  if (tauTot > 0.0) {
    var above = 0.0;
    var aPrev = 0.0;
    for (var kk = i32(nz) - 1; kk >= 0; kk--) {
      let k = u32(kk);
      let cl = col[k];
      let off = cl.zc * tanZ * dir;
      let ci = wrapX(i32(g.x) + i32(round(off.x / P.dx)));
      var cj = 0u;
      if (P.ny > 1u) { cj = wrapY(i32(g.y) + i32(round(off.y / P.dy))); }
      let id = ci + P.nx * cj + pl * k;
      let t = cplx[id].y;
      if (t > 0.0) {
        above += t;
        let a = cloudAbsorbed(above / mu);
        cplx[id] = vec2f(S * (a - aPrev) / (cl.rho0 * CP * cl.pi0 * cl.dzc), 0.0);
        aPrev = a;
      } else {
        cplx[id] = vec2f(0.0);
      }
    }
  }
  // 地表の日射：δ-Eddington の透過から雲の吸収を引き、地表と雲の間の多重反射を足す
  let R = cloudReflectance(tauTot, mu);
  let A = cloudAbsorbed(tauTot / mu);
  let T = max(1.0 - R - A, 0.0);
  diag[dSurf(5u, base)] = S * T / (1.0 - P.albedo * R);
}

// ============================================================ 柱ごとの前処理
// 大規模強制：沈降による移流（風上差分）＋ θ・qv の強制・ナッジング。
// 下の層から順に更新する。風上側の値は更新前の値を使う（上昇流のとき下の層、下降流のとき上の層）。
// 水物質について、強制で柱に入った水の量 [kg/m²] を返す。
fn columnForcing(f: u32, base: u32, pl: u32, dt: f32) -> f32 {
  let nz = P.nz;
  var prevOld = 0.0;
  var cur = so(f, base);
  var src = 0.0;
  for (var k = 0u; k < nz; k++) {
    let cl = col[k];
    var next = 0.0;
    var wTop = 0.0;
    if (k + 1u < nz) {
      wTop = col[k + 1u].wsub;
      next = so(f, base + pl * (k + 1u));
    }
    let ws = 0.5 * (cl.wsub + wTop);
    var d = 0.0;
    if (ws < 0.0 && k + 1u < nz) {
      d = -ws * (next - cur) / col[k + 1u].dzw;
    } else if (ws > 0.0 && k > 0u) {
      d = -ws * (cur - prevOld) / cl.dzw;
    }
    if (f == F_TH) { d += cl.thLs - cl.nudge * (cur - cl.th0); }
    if (f == F_QV) { d += cl.qvLs - cl.nudge * (cur - cl.qv0); }
    var nv = cur + dt * d;
    if (f != F_TH) {
      nv = max(nv, 0.0);
      src += (nv - cur) * cl.rho0 * cl.dzc;
    }
    if (nv != cur) { soSet(f, base + pl * k, nv); }
    prevOld = cur;
    cur = next;
  }
  return src;
}

// 地表層の安定度補正（Louis, Tiedtke & Geleyn 1982）。戻り値は中立の係数に掛ける (Fm, Fh)
fn louis(rib: f32, zOverZ0: f32, a2: f32) -> vec2f {
  if (rib < 0.0) {
    let c = 1.0 + 75.0 * a2 * sqrt(-rib * zOverZ0);
    return vec2f(1.0 - 10.0 * rib / c, 1.0 - 15.0 * rib / c);
  }
  let s = sqrt(1.0 + 5.0 * rib);
  return vec2f(1.0 / (1.0 + 10.0 * rib / s), 1.0 / (1.0 + 15.0 * rib * s));
}

@compute @workgroup_size(8, 8, 1)
fn csColumnPre(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= P.nx || g.y >= P.ny) { return; }
  let pl = P.nx * P.ny;
  let base = g.x + P.nx * g.y;
  let nz = P.nz;
  let dt = P.dtFull;
  let moist = (P.flags & FLAG_MOIST) != 0u;
  // この柱とその近くに凝結物が無ければ（csActDilate の印）、凝結物は 0 のまま（読まずに済ませる）
  let act = moist && diag[dSurf(7u, base)] > 0.0;

  // ---- 1. 負値の切り詰め
  var negAcc = 0.0;
  var tauLw = 0.0;   // 長波の κ∫ρ q dz（柱全体）
  var tauVis = 0.0;  // 可視の光学的厚さ（真上方向。地表の下向き長波用）
  if (moist) {
    for (var k = 0u; k < nz; k++) {
      let id = base + pl * k;
      let cl = col[k];
      let m = cl.rho0 * cl.dzc;
      let qv = so(F_QV, id);
      if (qv < 0.0) { negAcc -= qv * m; soSet(F_QV, id, 0.0); }
      if (act) {
        for (var f = F_QC; f <= F_QG; f++) {
          let q = so(f, id);
          if (q < 0.0) { negAcc -= q * m; soSet(f, id, 0.0); }
        }
        tauLw += (KAPPA_LW_LIQ * max(so(F_QC, id), 0.0) + KAPPA_LW_ICE * max(so(F_QI, id), 0.0)) * m;
        tauVis += cellTau(id, k);
      }
    }
  }

  // ---- 2. 大規模強制
  _ = columnForcing(F_TH, base, pl, dt);
  var lsWater = 0.0;
  if (moist) {
    lsWater += columnForcing(F_QV, base, pl, dt);
    if (act && (P.flags & FLAG_SUBS) != 0u) {
      for (var f = F_QC; f <= F_QG; f++) { lsWater += columnForcing(f, base, pl, dt); }
    }
  }
  diag[dSurf(9u, base)] += lsWater;

  // ---- 3. 地表
  let c0 = col[0];
  let id0 = base;
  var th0 = so(F_TH, id0);
  var qv0 = so(F_QV, id0);
  // 地面に対する風（計算座標系の移動速度を足す）。ステップ開始時の状態（sN）でセル中心へ平均
  // （sO の u, v は隣の柱のスレッドが摩擦で書き換えるので読まない）
  let uE = sN[F_U * P.n + g.y * P.nx + (g.x + 1u) % P.nx];
  let uw = 0.5 * (sN[F_U * P.n + id0] + uE) + P.frameU;
  var vw = P.frameV;
  if (P.ny > 1u) { vw += 0.5 * (sN[F_V * P.n + id0] + sN[F_V * P.n + ((g.y + 1u) % P.ny) * P.nx + g.x]); }
  let speed = sqrt(uw * uw + vw * vw);
  let z1 = c0.zc;
  let zr = max(z1 / max(P.z0, 1e-4), 1.5);
  let cn = KARMAN / log(zr);
  let a2 = cn * cn; // 中立のバルク係数
  // 対流時の弱風でも乱流で熱が運ばれるよう、自由対流の速度スケール w* を実効風速に足す（Beljaars 1995、zi = 1 km）
  let hPrev = diag[dSurf(3u, base)];
  let lePrev = diag[dSurf(4u, base)];
  let bflux = GRAV / th0 * (hPrev / (c0.rho0 * CP * c0.pi0) + 0.608 * th0 * lePrev / (c0.rho0 * LV));
  let wstar = pow(max(bflux, 0.0) * 1000.0, 1.0 / 3.0);
  let ueff = sqrt(speed * speed + 1.44 * wstar * wstar + 0.01);
  let t1 = th0 * c0.pi0;
  var H = P.shf;
  var LE = P.lhf;
  var fs = vec2f(1.0, 1.0); // 安定度補正（固定フラックスのときは中立）
  if ((P.flags & FLAG_SURF_ENERGY) != 0u) {
    var ts = diag[dSurf(2u, base)];
    let sw = diag[dSurf(5u, base)]; // csRadiation が求めた地表の日射
    // 地表の温位（地表のエクスナー関数は最下層から静水圧で外挿）
    let piS = c0.pi0 + GRAV * z1 / (CP * th0);
    let thS = ts / piS;
    let thv1 = th0 * (1.0 + 0.608 * qv0);
    let thvS = thS * (1.0 + 0.608 * qv0);
    let rib = GRAV * z1 * (thv1 - thvS) / (thv1 * ueff * ueff);
    fs = louis(clamp(rib, -10.0, 10.0), zr, a2);
    let ch = a2 * fs.y;
    // 下向き長波：晴天は Brutsaert (1975)、雲があれば黒体に近づける
    let e = qv0 * c0.p0 / (EPSV + qv0) / 100.0; // 水蒸気圧 [hPa]
    let epsClear = 1.24 * pow(max(e, 0.01) / t1, 1.0 / 7.0);
    let epsAir = epsClear + (1.0 - epsClear) * (1.0 - exp(-tauVis));
    let lwDown = epsAir * SIGMA_SB * t1 * t1 * t1 * t1;
    let lwUp = 0.98 * SIGMA_SB * ts * ts * ts * ts;
    let rn = (1.0 - P.albedo) * sw + 0.98 * lwDown - lwUp;
    let rho = c0.rho0;
    let ra = 1.0 / (ch * ueff); // 空気力学的抵抗 [s/m]
    H = rho * CP * (thS - th0) * piS / ra;
    let qs = qvsatLiq(ts, c0.p0);
    if (qs >= qv0) {
      // 気孔抵抗（Jarvis 1976 型。係数は Noilhan & Planton 1989）：日射・土壌水分・飽差・気温で決まる
      let f = 0.55 * (sw / 100.0) * (2.0 / P.lai);
      let F1 = (1.0 + f) / (f + P.rsMin / 5000.0);
      let F2 = clamp(P.soilWet, 1e-3, 1.0);
      let F3 = max(1.0 - 25.0 * max(qvsatLiq(t1, c0.p0) - qv0, 0.0), 0.25);
      let F4 = max(1.0 - 0.0016 * (298.0 - t1) * (298.0 - t1), 0.05);
      let rs = min(P.rsMin / P.lai * F1 / (F2 * F3 * F4), 5000.0);
      LE = rho * LV * (qs - qv0) / (ra + rs);
    } else {
      LE = rho * LV * (qs - qv0) / ra; // 結露（露）
    }
    let gflux = rn - H - LE;
    // force-restore 法（Deardorff 1978）
    ts += dt * (gflux / P.cSoil - (6.2831853 / P.tauSoil) * (ts - P.tDeep));
    diag[dSurf(2u, base)] = ts;
  }
  diag[dSurf(3u, base)] = H;
  diag[dSurf(4u, base)] = LE;
  th0 += dt * H / (c0.rho0 * CP * c0.pi0 * c0.dzc);
  soSet(F_TH, id0, th0);
  if (moist) {
    let qNew = max(qv0 + dt * LE / (c0.rho0 * LV * c0.dzc), 0.0);
    diag[dSurf(8u, base)] += (qNew - qv0) * c0.rho0 * c0.dzc; // 地表からの蒸発（負なら結露）
    soSet(F_QV, id0, qNew);
  }

  // 地表摩擦（最下層の u, v を減速。陰的）
  var us2: f32;
  if (P.ustar > 0.0) { us2 = P.ustar * P.ustar; } else { us2 = a2 * fs.x * ueff * speed; }
  if (speed > 1e-3 && P.z0 > 0.0) { // z0 = 0 は摩擦なし（自由すべり）
    let k0 = 1.0 / (1.0 + dt * us2 / (speed * c0.dzc));
    sO[F_U * P.n + id0] = (so(F_U, id0) + P.frameU) * k0 - P.frameU;
    if (P.ny > 1u) { sO[F_V * P.n + id0] = (so(F_V, id0) + P.frameV) * k0 - P.frameV; }
  }

  // ---- 4. 長波放射（雲頂での冷却、雲底での加熱）
  // Stevens et al. (2005) の DYCOMS-II の式：F(z) = F0·exp(−Q(z,∞)) + F1·exp(−Q(0,z)) [+ 第3項]
  //   Q = ∫κ ρ q dz（κ は雲水 85、雲氷 50 m²/kg）。雲が厚ければ雲頂で約 70 W/m² の冷却になる。
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
    var fPrev = 70.0 * exp(-tauLw) + 22.0; // 地表面での正味の上向きフラックス
    // 雲の無い柱はフラックスが高さによらず一定（加熱・冷却 0）なので省く（DYCOMS の第3項がある場合を除く）
    let lwLoops = tauLw > 0.0 || zi < 1e8;
    for (var k = 0u; k < select(0u, nz, lwLoops); k++) {
      let cl = col[k];
      let id = base + pl * k;
      if (act) { below += (KAPPA_LW_LIQ * so(F_QC, id) + KAPPA_LW_ICE * so(F_QI, id)) * cl.rho0 * cl.dzc; }
      let qAbove = max(tauLw - below, 0.0); // 上面より上の積算
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

// ============================================================ 雲微物理
// 1モーメント・6種類（Lin et al. 1983、Rutledge & Hobbs 1984、Hong et al. 2004 (WSM) を基にした簡略版）。
//  雲水 qc：水飽和に合わせる飽和調整（0℃以下でも過冷却水として残る）。−40℃で均質凍結、それより暖かい所は
//           Bigg (1953) の不均質凍結（ゆっくり）。
//  雲氷 qi：氷晶核（Cooper 1986）から生まれ、拡散成長（Hong et al. 2004）。水飽和の雲の中では氷の方が
//           飽和水蒸気圧が低いので、氷が育ち雲水が蒸発する（Bergeron–Findeisen）。0℃より暖かいと融ける。
//  雨 qr  ：雲水の自動変換・捕捉（Kessler）、蒸発（Klemp & Wilhelmson 1978）。0℃以下では Bigg の凍結であられに。
//  雪 qs  ：雲氷の併合・捕捉、昇華成長、雲水の捕捉（着氷）。雪の成長のうち着氷が占める割合が大きいとあられに変わる。
//  あられ qg：着氷・凍った雨・強く着氷した雪。0℃より暖かいと融ける。
// 粒径分布は指数分布（切片 N0 固定：雪 3×10⁶、あられ 4×10⁵ m⁻⁴）。
// 相変化の熱は「氷・液水温位」θil を保存したまま温度を解き直すことで、凍結・融解の熱も含めて自動で入る。

fn dvAir(T: f32, p: f32) -> f32 { return 2.11e-5 * pow(T / T_0, 1.94) * (101325.0 / p); }
// 氷の拡散成長の熱・拡散の抵抗 A + B [m s/kg]
fn abIce(T: f32, p: f32) -> f32 {
  return LS / (KA * T) * (LS / (RV * T) - 1.0) + RV * T / (esatIce(T) * dvAir(T, p));
}
// 指数分布の粒子の、通風を含む拡散成長の積分 N0 [0.78 λ⁻² + 0.31 Sc^⅓ (a/ν)^½ Γ((b+5)/2) λ^−(b+5)/2]
fn ventInt(n0: f32, lam: f32, a: f32, b: f32, gam: f32, nuK: f32) -> f32 {
  return n0 * (0.78 / (lam * lam) + 0.31 * 0.843 * sqrt(a / nuK) * gam * pow(lam, -(b + 5.0) * 0.5));
}

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
  var qc = so(F_QC, id);
  var qi = so(F_QI, id);
  var qr = so(F_QR, id);
  var qs = so(F_QS, id);
  var qg = so(F_QG, id);
  let p = cl.p0;
  let pi = cl.pi0;
  let rho = cl.rho0;
  // 雲の日射加熱（csRadiation。凝結物の無いセルでは 0）
  if ((P.flags & FLAG_RAD) != 0u) { th += dt * cplx[id].x; }

  // 凝結物が無く、水に対しても（0℃以下なら氷晶核ができる過飽和に対しても）未飽和なら何も起きない。
  // 晴れた格子が大半なので、これで微物理の計算量が大きく減る。
  if (qc == 0.0 && qi == 0.0 && qr == 0.0 && qs == 0.0 && qg == 0.0) {
    let T0 = th * pi;
    var quiet = qv < qvsatLiq(T0, p);
    if (ice && T0 < T_0) { quiet = quiet && qv < 1.05 * qvsatIce(T0, p); }
    if (quiet) { return; }
  }

  let T0 = th * pi;
  let Tc = T0 - T_0;
  let cold = ice && T0 < T_0;
  // 保存量 θil（このステップの相変化で変わらない）
  let thil = th - (LV * (qc + qr) + LS * (qi + qs + qg)) / (CP * pi);

  if (ice) {
    // 一瞬で起きる相変化：0℃より暖かい雲氷は融ける、−40℃より冷たい雲水・雨は凍る
    if (T0 > T_0 && qi > 0.0) { qc += qi; qi = 0.0; }
    if (T0 < T_HOM) { qi += qc; qc = 0.0; qg += qr; qr = 0.0; }
  }

  if (rain) {
    let rhoS = col[0].rho0;
    let fallFac = sqrt(rhoS / rho);           // 落下速度の密度補正 (ρ0/ρ)^½
    let nuK = MU_AIR / rho;                    // 動粘性係数
    // ---- 自動変換：雲水 → 雨（Kessler、閾値 1 g/kg）、雲氷 → 雪（Lin et al. 1983 の形、閾値 0.1 g/kg）
    let aL = min(dt * 0.001 * max(qc - 0.001, 0.0), qc);
    var aI = 0.0;
    if (ice) { aI = min(dt * 0.001 * exp(0.025 * min(Tc, 0.0)) * max(qi - 1e-4, 0.0), qi); }
    // ---- 捕捉の係数 [1/s]（陰的に解く）
    //  雨：Kessler 2.2 qr^0.875
    //  雪：Lin（N0 = 3×10⁶, 密度 100, V = 4.84 D^0.25）→ 1.484 (ρqs)^0.8125 (ρ0/ρ)^½
    //  あられ：Lin 型（N0 = 4×10⁵, 密度 500, 抗力係数 0.6）→ 2.2 ρ^0.375 qg^0.875
    let kR = 2.2 * pow(max(qr, 0.0), 0.875);
    var kS = 0.0; var kG = 0.0;
    if (ice) {
      kS = 1.484 * pow(max(rho * qs, 0.0), 0.8125) * fallFac;
      kG = 2.2 * pow(rho, 0.375) * pow(max(qg, 0.0), 0.875);
    }
    // 雲水
    let qcA = qc - aL;
    let sumL = kR + kS + kG;
    let qcN = qcA / (1.0 + dt * sumL);
    let colL = max(qcA - qcN, 0.0);
    let inv = 1.0 / max(sumL, 1e-20);
    let colLr = colL * kR * inv;
    let colLs = colL * kS * inv;
    let colLg = colL * kG * inv;
    // 雲氷（雪の捕捉効率は exp(0.025 Tc)、あられ 0.1、雨 1）
    let effSI = exp(0.025 * min(Tc, 0.0));
    let qiA = qi - aI;
    let sumI = kR + kS * effSI + 0.1 * kG;
    let qiN0 = qiA / (1.0 + dt * sumI);
    let colI = max(qiA - qiN0, 0.0);
    let invI = 1.0 / max(sumI, 1e-20);
    let colIr = colI * kR * invI;          // 雨滴が氷晶を捕えると凍る → あられ
    let colIs = colI * kS * effSI * invI;
    let colIg = colI * 0.1 * kG * invI;

    qc = qcN;
    qi = qiN0;
    qr += aL + colLr;
    qs += aI + colIs;
    qg += colIr + colIg;
    if (cold) {
      qs += colLs;   // 着氷した雪
      qg += colLg;
    } else {
      qr += colLs + colLg; // 0℃以上では集めた雲水は水のまま落ちる
    }

    // 雪・あられの粒径分布の傾き λ と、拡散成長の積分
    var iS = 0.0; var iG = 0.0;
    if (ice && qs > 1e-12) {
      let lamS = pow(9.4248e8 / (rho * qs), 0.25);
      iS = ventInt(3e6, lamS, 4.836 * fallFac, 0.25, 1.4569, nuK);
    }
    if (ice && qg > 1e-12) {
      let lamG = pow(6.2832e8 / (rho * qg), 0.25);
      iG = ventInt(4e5, lamG, sqrt(10900.0 / rho), 0.5, 1.6084, nuK);
    }

    // ---- 氷と水蒸気のやりとり（0℃以下）：氷晶核の活性化、雲氷・雪・あられの昇華成長（負なら昇華）
    var depS = 0.0;
    if (cold) {
      let qsi = qvsatIce(T0, p);
      // 氷晶核（Cooper 1986：N = 5 exp(0.304 (T0 − T)) m⁻³、上限 2.5×10⁵）。
      // 水飽和の雲の中、または氷に対して 5% 以上の過飽和で、氷晶（1 個 10⁻¹² kg）を作る
      let nIn = min(5.0 * exp(0.304 * (T_0 - T0)), 2.5e5);
      if (qv > qsi && (qc > 0.0 || qv > 1.05 * qsi)) {
        let qnuc = min(max(1e-12 * nIn / rho - qi, 0.0), 0.5 * (qv - qsi));
        qi += qnuc;
        qv -= qnuc;
      }
      // 雲氷の数濃度（Hong et al. 2004：5.38×10⁷ (ρqi)^0.75）と粒径（D = 11.9 m^½）
      var cI = 0.0;
      if (qi > 0.0) {
        let ni = clamp(max(nIn, 5.38e7 * pow(rho * qi, 0.75)), 1e3, 1e6);
        let di = min(11.9 * sqrt(rho * qi / ni), 5e-4);
        cI = 4.0 * di * ni;
      }
      let ab = abIce(T0, p);
      // 成長速度 = c (Si − 1)、c [1/s]（qv − qsi に対する係数）
      let cAll = (cI + 6.2832 * (iS + iG)) / (rho * ab * qsi);
      if (cAll > 0.0) {
        // 過飽和は成長で減るので、その分を含めて解析的に積分する（大きな dt でも行き過ぎない）
        let gam = 1.0 + LS * LS * qsi / (CP * RV * T0 * T0);
        let sat = qv - qsi;
        let dep = sat / gam * (1.0 - exp(-gam * cAll * dt));
        let invC = 1.0 / (cAll * rho * ab * qsi);
        var dI = dep * cI * invC;
        var dS = dep * 6.2832 * iS * invC;
        var dG = dep * 6.2832 * iG * invC;
        if (dep < 0.0) { dI = max(dI, -qi); dS = max(dS, -qs); dG = max(dG, -qg); }
        qi += dI; qs += dS; qg += dG;
        qv -= dI + dS + dG;
        depS = dS;
      }

      // ---- 雪 → あられ
      //  着氷率：雪の成長のうち雲水の捕捉が占める割合 Fr。Fr が大きい（着氷が主）とき、
      //  着氷した量の 2 倍（着氷が粒の半分を占めたらあられとみなす）の雪をあられに移す。
      //  + Lin et al. (1983) の併合による変換 1e-3 exp(0.09 Tc)(qs − 0.6 g/kg)
      if (qs > 0.0) {
        let fr = colLs / (colLs + max(depS, 0.0) + colIs + 1e-20);
        var dq = min(qs, 2.0 * colLs * smoothstep(0.5, 0.9, fr));
        let rAgg = 1e-3 * exp(0.09 * Tc) * max(qs - 6e-4, 0.0) / qs;
        dq += (qs - dq) * (1.0 - exp(-dt * rAgg));
        qs -= dq;
        qg += dq;
      }
      // ---- 雨の凍結（Bigg 1953：B' = 100 m⁻³s⁻¹, A' = 0.66 K⁻¹）→ あられ
      if (qr > 0.0) {
        let lr = log(max(rho * qr, 1e-20) / 2.513e10); // ln(ρqr / (π ρw N0r))
        let pgfr = exp(log(1.579e15 / rho) + 1.75 * lr) * (exp(0.66 * (T_0 - T0)) - 1.0);
        let dq = min(qr, dt * pgfr);
        qr -= dq;
        qg += dq;
      }
      // ---- 雲水の不均質凍結（Bigg 1953 の雲粒版）→ 雲氷
      if (qc > 0.0) {
        let kh = 100.0 * (exp(0.66 * (T_0 - T0)) - 1.0) * rho * qc / (RHO_W * P.nDrop);
        let dq = qc * (1.0 - exp(-dt * kh));
        qc -= dq;
        qi += dq;
      }
    }

    // ---- 雪・あられの融解（0℃より暖かい。Lin et al. 1983：熱伝導と、集めた雲水の熱。蒸発の冷却で遅れる）
    if (ice && T0 > T_0 && (qs > 0.0 || qg > 0.0)) {
      let qs0 = qvsatLiq(T_0, p);
      let gm = 6.2832 / (rho * LF) * (KA * Tc - LV * dvAir(T0, p) * rho * (qs0 - qv));
      let mS = max(gm * iS + CW * Tc / LF * colLs / dt, 0.0);
      let mG = max(gm * iG + CW * Tc / LF * colLg / dt, 0.0);
      let dS = min(qs, dt * mS);
      let dG = min(qg, dt * mG);
      qs -= dS; qg -= dG;
      qr += dS + dG;
    }

    // ---- 雨の蒸発（水に対して未飽和のとき。Klemp & Wilhelmson 1978）
    let qsl = qvsatLiq(T0, p);
    if (qr > 0.0 && qv < qsl) {
      let rhoqr = 0.001 * rho * qr;
      let cvent = 1.6 + 124.9 * pow(rhoqr, 0.2046);
      var ern = dt * (cvent * pow(rhoqr, 0.525)) / (2.55e8 / (p * qsl) + 5.4e5) * ((qsl - qv) / (0.001 * rho * qsl));
      ern = min(min(ern, 0.5 * (qsl - qv)), qr);
      qv += ern;
      qr -= ern;
    }
  }

  // ---- 飽和調整（雲水のみ）：qtn = qv + qc と θil を保ったまま、qv = 水の飽和混合比 となる温度を Newton 法で解く
  let qtn = qv + qc;
  let tl = thil * pi;
  let iceSum = qi + qs + qg;
  var T = T0;
  for (var it = 0; it < 4; it++) {
    var f0: f32; var f1: f32;
    for (var j = 0; j < 2; j++) {
      let Tj = T + f32(j) * 0.05;
      let qnj = max(qtn - qvsatLiq(Tj, p), 0.0);
      let fj = Tj - tl - (LV * (qnj + qr) + LS * iceSum) / CP;
      if (j == 0) { f0 = fj; } else { f1 = fj; }
    }
    let d = (f1 - f0) / 0.05;
    T -= f0 / max(d, 0.1);
  }
  qc = max(qtn - qvsatLiq(T, p), 0.0);
  qv = qtn - qc;
  if (ice && T < T_HOM && qc > 0.0) {
    // −40℃より冷たい所で水の過飽和になった分は、そのまま凍る（均質凍結）
    qi += qc;
    qc = 0.0;
    T = tl + (LV * qr + LS * (qi + qs + qg)) / CP;
  }
  th = T / pi;

  soSet(F_TH, id, th);
  soSet(F_QV, id, qv);
  soSet(F_QC, id, qc);
  soSet(F_QI, id, qi);
  soSet(F_QR, id, qr);
  soSet(F_QS, id, qs);
  soSet(F_QG, id, qg);
}

// ============================================================ 落下
// 風上差分、CFL に応じて小分け。落下速度は小分けの前の値で固定し、diag の正値保存リミッタの係数の場所
// （次のステップの最終段まで使わない）に置く：dPd(F_QV) 雨、dPd(F_QC) 雲氷、dPd(F_QI) 雪、dPd(F_QR) あられ。
fn vSlot(s: u32, id: u32) -> u32 { return dPd(F_QV + s, id); }

@compute @workgroup_size(8, 8, 1)
fn csSediment(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= P.nx || g.y >= P.ny) { return; }
  let pl = P.nx * P.ny;
  let base = g.x + P.nx * g.y;
  let nz = P.nz;
  let dt = P.dtFull;
  let moist = (P.flags & FLAG_MOIST) != 0u;
  let rain = (P.flags & FLAG_RAIN) != 0u;

  var precip = 0.0;
  var anyFall = false;
  if (moist && rain) {
    for (var k = 0u; k < nz; k++) {
      let id = base + pl * k;
      if (so(F_QR, id) > 0.0 || so(F_QS, id) > 0.0 || so(F_QG, id) > 0.0 || so(F_QI, id) > 0.0) { anyFall = true; break; }
    }
  }
  if (anyFall) {
    let rhoS = col[0].rho0;
    var maxC = 0.0;
    for (var k = 0u; k < nz; k++) {
      let cl = col[k];
      let id = base + pl * k;
      let rho = cl.rho0;
      let fallFac = sqrt(rhoS / rho);
      // 雨：Klemp & Wilhelmson (1978)
      let rq = max(0.001 * rho * so(F_QR, id), 0.0);
      let vr = 36.34 * pow(rq, 0.1364) * fallFac;
      // 雲氷：Heymsfield & Donner (1990) 3.29 (ρqi)^0.16
      let vi = 3.29 * pow(max(rho * so(F_QI, id), 1e-12), 0.16);
      // 雪：Lin（質量加重平均）1.836 (ρqs)^0.0625 (ρ0/ρ)^½
      let vs = 1.836 * pow(max(rho * so(F_QS, id), 1e-12), 0.0625) * fallFac;
      // あられ・ひょう：V = 6.8 m/s ×(ρqg / 1 g m⁻³)^(1/8) ×(1 kg m⁻³ / ρ)^(1/2)（1〜15 m/s に制限）
      let gq = max(rho * so(F_QG, id), 0.0);
      let vg = clamp(6.8 * pow(gq * 1000.0, 0.125) / sqrt(rho), 1.0, 15.0);
      var vmax = 0.0;
      if (rq > 0.0) { vmax = max(vmax, vr); }
      if (so(F_QI, id) > 0.0) { vmax = max(vmax, vi); }
      if (so(F_QS, id) > 0.0) { vmax = max(vmax, vs); }
      if (gq > 0.0) { vmax = max(vmax, vg); }
      maxC = max(maxC, vmax * dt / cl.dzc);
      diag[vSlot(0u, id)] = vr;
      diag[vSlot(1u, id)] = vi;
      diag[vSlot(2u, id)] = vs;
      diag[vSlot(3u, id)] = vg;
    }
    let nsub = u32(clamp(ceil(maxC / 0.8), 1.0, 50.0));
    let dts = dt / f32(nsub);
    for (var s = 0u; s < nsub; s++) {
      var fluxAbove = vec4f(0.0); // (雨, 雲氷, 雪, あられ)
      for (var kk = i32(nz) - 1; kk >= 0; kk--) {
        let k = u32(kk);
        let cl = col[k];
        let id = base + pl * k;
        let q = vec4f(so(F_QR, id), so(F_QI, id), so(F_QS, id), so(F_QG, id));
        let v = vec4f(diag[vSlot(0u, id)], diag[vSlot(1u, id)], diag[vSlot(2u, id)], diag[vSlot(3u, id)]);
        let fluxOut = cl.rho0 * q * v;
        let qn = max(q + dts * (fluxAbove - fluxOut) / (cl.rho0 * cl.dzc), vec4f(0.0));
        soSet(F_QR, id, qn.x);
        soSet(F_QI, id, qn.y);
        soSet(F_QS, id, qn.z);
        soSet(F_QG, id, qn.w);
        fluxAbove = fluxOut;
      }
      precip += (fluxAbove.x + fluxAbove.y + fluxAbove.z + fluxAbove.w) * dts;
    }
  }
  var anyCond = false;
  if (moist) {
    for (var k = 0u; k < nz; k++) {
      let id = base + pl * k;
      if (so(F_QC, id) > 0.0 || so(F_QI, id) > 0.0 || so(F_QR, id) > 0.0 || so(F_QS, id) > 0.0 || so(F_QG, id) > 0.0) { anyCond = true; break; }
    }
  }
  diag[dSurf(0u, base)] += precip;
  diag[dSurf(6u, base)] = select(0.0, 1.0, anyCond);
}
