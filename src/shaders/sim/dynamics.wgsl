// 力学：渦粘性、スカラー・運動量の RK 段更新、発散、射影。
// （common.wgsl の後ろに連結）

// 凝結物の総量（浮力の荷重）
fn condensate(i: i32, j: i32, k: i32) -> f32 {
  return c(F_QC, i, j, k) + c(F_QI, i, j, k) + c(F_QR, i, j, k) + c(F_QS, i, j, k) + c(F_QG, i, j, k);
}

// ============================================================ 渦粘性（Smagorinsky–Lilly）
@compute @workgroup_size(8, 8, 1)
fn csNu(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= P.nx || g.y >= P.ny || g.z >= P.nz) { return; }
  let i = i32(g.x); let j = i32(g.y); let k = i32(g.z);
  let id = idx(i, j, k);
  var nu = P.nuConst;
  if ((P.flags & FLAG_SMAG) != 0u) {
    let cl = col[k];
    let km = max(k - 1, 0); let kp = min(k + 1, i32(P.nz) - 1);
    let dzz = max(col[kp].zc - col[km].zc, 1e-3);
    let dudx = (c(F_U, i + 1, j, k) - c(F_U, i, j, k)) / P.dx;
    let dvdy = (c(F_V, i, j + 1, k) - c(F_V, i, j, k)) / P.dy;
    let dwdz = (cw(i, j, k + 1) - cw(i, j, k)) / cl.dzc;
    let uc = 0.5 * (c(F_U, i, j, kp) + c(F_U, i + 1, j, kp)) - 0.5 * (c(F_U, i, j, km) + c(F_U, i + 1, j, km));
    let vc = 0.5 * (c(F_V, i, j, kp) + c(F_V, i, j + 1, kp)) - 0.5 * (c(F_V, i, j, km) + c(F_V, i, j + 1, km));
    let dudz = uc / dzz;
    let dvdz = vc / dzz;
    let dudy = (0.5 * (c(F_U, i, j + 1, k) + c(F_U, i + 1, j + 1, k)) - 0.5 * (c(F_U, i, j - 1, k) + c(F_U, i + 1, j - 1, k))) / (2.0 * P.dy);
    let dvdx = (0.5 * (c(F_V, i + 1, j, k) + c(F_V, i + 1, j + 1, k)) - 0.5 * (c(F_V, i - 1, j, k) + c(F_V, i - 1, j + 1, k))) / (2.0 * P.dx);
    let dwdx = (0.5 * (cw(i + 1, j, k) + cw(i + 1, j, k + 1)) - 0.5 * (cw(i - 1, j, k) + cw(i - 1, j, k + 1))) / (2.0 * P.dx);
    let dwdy = (0.5 * (cw(i, j + 1, k) + cw(i, j + 1, k + 1)) - 0.5 * (cw(i, j - 1, k) + cw(i, j - 1, k + 1))) / (2.0 * P.dy);
    let s2 = 2.0 * (dudx * dudx + dvdy * dvdy + dwdz * dwdz) +
             (dudy + dvdx) * (dudy + dvdx) + (dudz + dwdx) * (dudz + dwdx) + (dvdz + dwdy) * (dvdz + dwdy);
    // 浮力による抑制（Lilly 1962）：N² は仮温位から
    let thvp = c(F_TH, i, j, kp) * (1.0 + 0.608 * c(F_QV, i, j, kp) - condensate(i, j, kp));
    let thvm = c(F_TH, i, j, km) * (1.0 + 0.608 * c(F_QV, i, j, km) - condensate(i, j, km));
    var n2 = GRAV / cl.th0 * (thvp - thvm) / dzz;
    // 雲水のある（水に飽和した）セルでは湿潤の N²（Durran & Klemp 1982）。乾燥の N² のままだと、
    // 湿潤断熱に近い雲の中を「安定」とみなして渦粘性を消してしまう
    if (c(F_QC, i, j, k) > 0.0) {
      let T = c(F_TH, i, j, k) * cl.pi0;
      let qs = qvsatLiq(T, cl.p0);
      let a = (1.0 + LV * qs / (RD * T)) / (1.0 + EPSV * LV * LV * qs / (CP * RD * T * T));
      let thp = c(F_TH, i, j, kp); let thm = c(F_TH, i, j, km);
      let qsp = qvsatLiq(thp * col[kp].pi0, col[kp].p0);
      let qsm = qvsatLiq(thm * col[km].pi0, col[km].p0);
      let qwp = c(F_QV, i, j, kp) + c(F_QC, i, j, kp) + c(F_QI, i, j, kp);
      let qwm = c(F_QV, i, j, km) + c(F_QC, i, j, km) + c(F_QI, i, j, km);
      n2 = GRAV * (a * ((log(thp) - log(thm)) + LV / (CP * T) * (qsp - qsm)) - (qwp - qwm)) / dzz;
    }
    let fri = sqrt(max(0.0, 1.0 - n2 / (P.prandtl * max(s2, 1e-10))));
    var delta: f32;
    if (P.ny == 1u) { delta = sqrt(P.dx * cl.dzc); } else { delta = pow(P.dx * P.dy * cl.dzc, 1.0 / 3.0); }
    let l = P.cs * delta;
    nu += l * l * sqrt(s2) * fri;
    // 陽解法の拡散の安定条件を超えないよう上限
    var dmin = min(P.dx, cl.dzc);
    if (P.ny > 1u) { dmin = min(dmin, P.dy); }
    nu = min(nu, 0.12 * dmin * dmin / P.dtFull);
  }
  diag[D_NU * P.n + id] = nu;
}

fn nuAt(i: i32, j: i32, k: i32) -> f32 { return diag[D_NU * P.n + idx(i, j, k)]; }

// ============================================================ スカラー
fn sFluxX(f: u32, i: i32, j: i32, k: i32) -> f32 {
  let u = c(F_U, i, j, k);
  let s = interp5(c(f, i - 3, j, k), c(f, i - 2, j, k), c(f, i - 1, j, k), c(f, i, j, k), c(f, i + 1, j, k), c(f, i + 2, j, k), u);
  return u * s;
}
fn sFluxY(f: u32, i: i32, j: i32, k: i32) -> f32 {
  let v = c(F_V, i, j, k);
  let s = interp5(c(f, i, j - 3, k), c(f, i, j - 2, k), c(f, i, j - 1, k), c(f, i, j, k), c(f, i, j + 1, k), c(f, i, j + 2, k), v);
  return v * s;
}
// 下面 k での値（不等間隔の重み。境界付近は次数を下げる。範囲外の点の重みは 0）
fn zFaceValue(f: u32, i: i32, j: i32, k: i32, vel: f32) -> f32 {
  let v0 = vec4f(c(f, i, j, k - 3), c(f, i, j, k - 2), c(f, i, j, k - 1), c(f, i, j, k));
  let v1 = vec2f(c(f, i, j, k + 1), c(f, i, j, k + 2));
  return zInterpS(k, v0, v1, vel);
}
// 下面 k の鉛直フラックス ρ w s
fn sFluxZ(f: u32, i: i32, j: i32, k: i32) -> f32 {
  if (k <= 0 || k >= i32(P.nz)) { return 0.0; }
  let w = cw(i, j, k);
  return col[k].rhoF * w * zFaceValue(f, i, j, k, w);
}

fn baseOf(f: u32, k: i32) -> f32 {
  if (f == F_TH) { return col[k].th0; }
  if (f == F_QV) { return col[k].qv0; }
  return 0.0;
}

// ---- 正値保存リミッタ（RK3 の最終段のみ。Skamarock 2006 の考え方）
// 各面のフラックスは「質量を減らす側のセル」の係数 r で縮める。
// r = min(1, そのセルにある量 / そのセルから出ていく量)。面ごとに両側で同じフラックスを使うので保存的。
fn pdRatio(f: u32, i: i32, j: i32, k: i32) -> f32 { return diag[dPd(f, idx(i, j, k))]; }
fn limX(f: u32, i: i32, j: i32, k: i32) -> f32 {
  let F = sFluxX(f, i, j, k);
  return F * select(pdRatio(f, i, j, k), pdRatio(f, i - 1, j, k), F > 0.0);
}
fn limY(f: u32, i: i32, j: i32, k: i32) -> f32 {
  let F = sFluxY(f, i, j, k);
  return F * select(pdRatio(f, i, j, k), pdRatio(f, i, j - 1, k), F > 0.0);
}
fn limZ(f: u32, i: i32, j: i32, k: i32) -> f32 {
  if (k <= 0 || k >= i32(P.nz)) { return 0.0; }
  let F = sFluxZ(f, i, j, k);
  return F * select(pdRatio(f, i, j, k), pdRatio(f, i, j, k - 1), F > 0.0);
}

@compute @workgroup_size(8, 8, 1)
fn csPdRatio(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= P.nx || g.y >= P.ny || g.z >= P.nz) { return; }
  let i = i32(g.x); let j = i32(g.y); let k = i32(g.z);
  let id = idx(i, j, k);
  let cl = col[k];
  let act = condActive(i, j);
  for (var f = F_QV; f <= F_QG; f++) {
    if (f >= F_QC && !act) { diag[dPd(f, id)] = 1.0; continue; }
    let out = (max(sFluxX(f, i + 1, j, k), 0.0) - min(sFluxX(f, i, j, k), 0.0)) / P.dx
            + (max(sFluxY(f, i, j + 1, k), 0.0) - min(sFluxY(f, i, j, k), 0.0)) / P.dy
            + (max(sFluxZ(f, i, j, k + 1), 0.0) - min(sFluxZ(f, i, j, k), 0.0)) / (cl.rho0 * cl.dzc);
    var r = 1.0;
    let avail = max(sN[f * P.n + id], 0.0);
    if (out * P.dt > avail) { r = avail / (out * P.dt); }
    diag[dPd(f, id)] = r;
  }
}

// ---- 凝結物の無い列の省略
// 微物理が付けた「柱に凝結物あり」の印を水平 ±9 列に広げる（RK3 の3段で、移流の幅 ±3 列ずつ値が広がりうるため）。
// 印の無い列では、雲水・雨・あられはステップ中ずっと 0 のままなので、その移流・拡散・リミッタを計算しない。
@compute @workgroup_size(8, 8, 1)
fn csActDilate(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= P.nx || g.y >= P.ny) { return; }
  var a = 0.0;
  let rj = select(0, 9, P.ny > 1u);
  for (var dj = -rj; dj <= rj; dj++) {
    for (var di = -9; di <= 9; di++) {
      a = max(a, diag[dSurf(6u, wrapX(i32(g.x) + di) + P.nx * wrapY(i32(g.y) + dj))]);
    }
  }
  diag[dSurf(7u, g.x + P.nx * g.y)] = a;
}
fn condActive(i: i32, j: i32) -> bool { return diag[dSurf(7u, wrapX(i) + P.nx * wrapY(j))] > 0.0; }

fn scalarTendency(f: u32, i: i32, j: i32, k: i32, limited: bool) -> f32 {
  let cl = col[k];
  var t: f32;
  if (limited) {
    t = -(limX(f, i + 1, j, k) - limX(f, i, j, k)) / P.dx
        -(limY(f, i, j + 1, k) - limY(f, i, j, k)) / P.dy
        -(limZ(f, i, j, k + 1) - limZ(f, i, j, k)) / (cl.rho0 * cl.dzc);
  } else {
    t = -(sFluxX(f, i + 1, j, k) - sFluxX(f, i, j, k)) / P.dx
        -(sFluxY(f, i, j + 1, k) - sFluxY(f, i, j, k)) / P.dy
        -(sFluxZ(f, i, j, k + 1) - sFluxZ(f, i, j, k)) / (cl.rho0 * cl.dzc);
  }

  // 拡散（K = ν/Pr）。勾配は場そのもので取る（−K∂s/∂z）。基本場からの偏差で取ると、逆転層が
  // 初期の位置から少しずれただけで偽の大きな勾配ができ、雲頂の層に水と冷たさが溜まる（DYCOMS-II で確認）。
  let sc = c(f, i, j, k);
  let nu0 = nuAt(i, j, k);
  let kx1 = 0.5 * (nu0 + nuAt(i + 1, j, k)); let kx0 = 0.5 * (nu0 + nuAt(i - 1, j, k));
  var diff = (kx1 * (c(f, i + 1, j, k) - sc) - kx0 * (sc - c(f, i - 1, j, k))) / (P.dx * P.dx);
  if (P.ny > 1u) {
    let ky1 = 0.5 * (nu0 + nuAt(i, j + 1, k)); let ky0 = 0.5 * (nu0 + nuAt(i, j - 1, k));
    diff += (ky1 * (c(f, i, j + 1, k) - sc) - ky0 * (sc - c(f, i, j - 1, k))) / (P.dy * P.dy);
  }
  let nz = i32(P.nz);
  var fz1 = 0.0; var fz0 = 0.0;
  if (k + 1 < nz) {
    fz1 = col[k + 1].rhoF * 0.5 * (nu0 + nuAt(i, j, k + 1)) * (c(f, i, j, k + 1) - sc) / col[k + 1].dzw;
  }
  if (k > 0) {
    fz0 = cl.rhoF * 0.5 * (nu0 + nuAt(i, j, k - 1)) * (sc - c(f, i, j, k - 1)) / cl.dzw;
  }
  diff += (fz1 - fz0) / (cl.rho0 * cl.dzc);
  t += diff / P.prandtl;

  let s0 = sc - baseOf(f, k);
  if (f == F_TH) { t -= spongeCoef(cl.zc) * s0; }
  return t;
}

@compute @workgroup_size(8, 8, 2)
fn csScalars(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= P.nx || g.y >= P.ny || g.z >= P.nz) { return; }
  let i = i32(g.x); let j = i32(g.y); let k = i32(g.z);
  let id = idx(i, j, k);
  sO[F_TH * P.n + id] = sN[F_TH * P.n + id] + P.dt * scalarTendency(F_TH, i, j, k, false);
  if ((P.flags & FLAG_MOIST) != 0u) {
    let lim = P.stage == 2u && (P.flags & FLAG_PD) != 0u;
    sO[F_QV * P.n + id] = sN[F_QV * P.n + id] + P.dt * scalarTendency(F_QV, i, j, k, lim);
    let act = condActive(i, j);
    for (var f = F_QC; f <= F_QG; f++) {
      if (act) { sO[f * P.n + id] = sN[f * P.n + id] + P.dt * scalarTendency(f, i, j, k, lim); }
      else { sO[f * P.n + id] = sN[f * P.n + id]; }
    }
  } else {
    sO[F_QV * P.n + id] = sN[F_QV * P.n + id];
    for (var f = F_QC; f <= F_QG; f++) { sO[f * P.n + id] = 0.0; }
  }
}

// ============================================================ 運動量

// --- u
fn uFluxX(i: i32, j: i32, k: i32) -> f32 { // セル中心 i-1（u[i-1] と u[i] の間）
  let vel = 0.5 * (c(F_U, i - 1, j, k) + c(F_U, i, j, k));
  return vel * interp5(c(F_U, i - 3, j, k), c(F_U, i - 2, j, k), c(F_U, i - 1, j, k), c(F_U, i, j, k), c(F_U, i + 1, j, k), c(F_U, i + 2, j, k), vel);
}
fn uFluxY(i: i32, j: i32, k: i32) -> f32 { // 角 (i-1/2, j-1/2)
  let vel = 0.5 * (c(F_V, i - 1, j, k) + c(F_V, i, j, k));
  return vel * interp5(c(F_U, i, j - 3, k), c(F_U, i, j - 2, k), c(F_U, i, j - 1, k), c(F_U, i, j, k), c(F_U, i, j + 1, k), c(F_U, i, j + 2, k), vel);
}
fn uFluxZ(i: i32, j: i32, k: i32) -> f32 { // (i-1/2, 面 k)
  if (k <= 0 || k >= i32(P.nz)) { return 0.0; }
  let vel = 0.5 * (cw(i - 1, j, k) + cw(i, j, k));
  return col[k].rhoF * vel * zFaceValue(F_U, i, j, k, vel);
}
// --- v
fn vFluxX(i: i32, j: i32, k: i32) -> f32 {
  let vel = 0.5 * (c(F_U, i, j - 1, k) + c(F_U, i, j, k));
  return vel * interp5(c(F_V, i - 3, j, k), c(F_V, i - 2, j, k), c(F_V, i - 1, j, k), c(F_V, i, j, k), c(F_V, i + 1, j, k), c(F_V, i + 2, j, k), vel);
}
fn vFluxY(i: i32, j: i32, k: i32) -> f32 {
  let vel = 0.5 * (c(F_V, i, j - 1, k) + c(F_V, i, j, k));
  return vel * interp5(c(F_V, i, j - 3, k), c(F_V, i, j - 2, k), c(F_V, i, j - 1, k), c(F_V, i, j, k), c(F_V, i, j + 1, k), c(F_V, i, j + 2, k), vel);
}
fn vFluxZ(i: i32, j: i32, k: i32) -> f32 {
  if (k <= 0 || k >= i32(P.nz)) { return 0.0; }
  let vel = 0.5 * (cw(i, j - 1, k) + cw(i, j, k));
  return col[k].rhoF * vel * zFaceValue(F_V, i, j, k, vel);
}
// --- w（w 点 = 面 k）
fn wFluxX(i: i32, j: i32, k: i32) -> f32 { // (i-1/2, 面 k)
  let vel = 0.5 * (c(F_U, i, j, k - 1) + c(F_U, i, j, k));
  return vel * interp5(cw(i - 3, j, k), cw(i - 2, j, k), cw(i - 1, j, k), cw(i, j, k), cw(i + 1, j, k), cw(i + 2, j, k), vel);
}
fn wFluxY(i: i32, j: i32, k: i32) -> f32 {
  let vel = 0.5 * (c(F_V, i, j, k - 1) + c(F_V, i, j, k));
  return vel * interp5(cw(i, j - 3, k), cw(i, j - 2, k), cw(i, j - 1, k), cw(i, j, k), cw(i, j + 1, k), cw(i, j + 2, k), vel);
}
fn wFluxZ(i: i32, j: i32, kc: i32) -> f32 { // セル中心 kc（w[kc] と w[kc+1] の間）
  let vel = 0.5 * (cw(i, j, kc) + cw(i, j, kc + 1));
  let v0 = vec4f(cw(i, j, kc - 2), cw(i, j, kc - 1), cw(i, j, kc), cw(i, j, kc + 1));
  let v1 = vec2f(cw(i, j, kc + 2), cw(i, j, kc + 3));
  return col[kc].rho0 * vel * zInterpW(kc, v0, v1, vel);
}

fn buoyancyCenter(i: i32, j: i32, k: i32) -> f32 {
  let cl = col[k];
  return GRAV * ((c(F_TH, i, j, k) - cl.th0) / cl.th0 + 0.608 * (c(F_QV, i, j, k) - cl.qv0) - condensate(i, j, k));
}

// ---- 速度の拡散：応力テンソル τij = ν (∂ui/∂xj + ∂uj/∂xi) の発散（鉛直は密度で重み付け）
// 地表と上端では τ13 = τ23 = 0（地表の摩擦は csColumnPre で別に与える）。
// 渦粘性 ν はセル中心の値で、u 点・角の点では隣の平均を使う。
fn nuEdgeXZ(i: i32, j: i32, k: i32) -> f32 { // (i−1/2, j, 面 k)
  return 0.25 * (nuAt(i - 1, j, k - 1) + nuAt(i, j, k - 1) + nuAt(i - 1, j, k) + nuAt(i, j, k));
}
fn nuEdgeYZ(i: i32, j: i32, k: i32) -> f32 { // (i, j−1/2, 面 k)
  return 0.25 * (nuAt(i, j - 1, k - 1) + nuAt(i, j, k - 1) + nuAt(i, j - 1, k) + nuAt(i, j, k));
}
fn nuEdgeXY(i: i32, j: i32, k: i32) -> f32 { // (i−1/2, j−1/2, k)
  return 0.25 * (nuAt(i - 1, j - 1, k) + nuAt(i, j - 1, k) + nuAt(i - 1, j, k) + nuAt(i, j, k));
}
// 法線応力（セル中心）
fn tau11(i: i32, j: i32, k: i32) -> f32 { return 2.0 * nuAt(i, j, k) * (c(F_U, i + 1, j, k) - c(F_U, i, j, k)) / P.dx; }
fn tau22(i: i32, j: i32, k: i32) -> f32 { return 2.0 * nuAt(i, j, k) * (c(F_V, i, j + 1, k) - c(F_V, i, j, k)) / P.dy; }
fn tau33(i: i32, j: i32, k: i32) -> f32 { return 2.0 * nuAt(i, j, k) * (cw(i, j, k + 1) - cw(i, j, k)) / col[k].dzc; }
// τ12 at (i−1/2, j−1/2, k)
fn tau12(i: i32, j: i32, k: i32) -> f32 {
  return nuEdgeXY(i, j, k) * ((c(F_U, i, j, k) - c(F_U, i, j - 1, k)) / P.dy + (c(F_V, i, j, k) - c(F_V, i - 1, j, k)) / P.dx);
}
// ρ τ13 at (i−1/2, j, 面 k)、ρ τ23 at (i, j−1/2, 面 k)。面 0 と nz では 0
fn rtau13(i: i32, j: i32, k: i32) -> f32 {
  if (k <= 0 || k >= i32(P.nz)) { return 0.0; }
  return col[k].rhoF * nuEdgeXZ(i, j, k) * ((c(F_U, i, j, k) - c(F_U, i, j, k - 1)) / col[k].dzw + (cw(i, j, k) - cw(i - 1, j, k)) / P.dx);
}
fn rtau23(i: i32, j: i32, k: i32) -> f32 {
  if (k <= 0 || k >= i32(P.nz)) { return 0.0; }
  return col[k].rhoF * nuEdgeYZ(i, j, k) * ((c(F_V, i, j, k) - c(F_V, i, j, k - 1)) / col[k].dzw + (cw(i, j, k) - cw(i, j - 1, k)) / P.dy);
}

@compute @workgroup_size(8, 8, 2)
fn csMomentum(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= P.nx || g.y >= P.ny || g.z >= P.nz) { return; }
  let i = i32(g.x); let j = i32(g.y); let k = i32(g.z);
  let id = idx(i, j, k);
  let cl = col[k];
  let rdz = 1.0 / (cl.rho0 * cl.dzc);
  let sp = spongeCoef(cl.zc);

  // u（x 面 i-1/2）
  {
    var t = -(uFluxX(i + 1, j, k) - uFluxX(i, j, k)) / P.dx
            -(uFluxY(i, j + 1, k) - uFluxY(i, j, k)) / P.dy
            -(uFluxZ(i, j, k + 1) - uFluxZ(i, j, k)) * rdz;
    t += (tau11(i, j, k) - tau11(i - 1, j, k)) / P.dx + (rtau13(i, j, k + 1) - rtau13(i, j, k)) * rdz;
    if (P.ny > 1u) { t += (tau12(i, j + 1, k) - tau12(i, j, k)) / P.dy; }
    t -= sp * (c(F_U, i, j, k) - cl.u0);
    if (P.ny > 1u && P.fcor != 0.0) {
      // コリオリ力と大規模気圧傾度（地衡風）。地面に対する風で評価
      let vAt = 0.25 * (c(F_V, i - 1, j, k) + c(F_V, i, j, k) + c(F_V, i - 1, j + 1, k) + c(F_V, i, j + 1, k));
      t += P.fcor * (vAt + P.frameV - cl.vg);
    }
    sO[F_U * P.n + id] = sN[F_U * P.n + id] + P.dt * t;
  }
  // v（y 面 j-1/2）
  if (P.ny > 1u) {
    var t = -(vFluxX(i + 1, j, k) - vFluxX(i, j, k)) / P.dx
            -(vFluxY(i, j + 1, k) - vFluxY(i, j, k)) / P.dy
            -(vFluxZ(i, j, k + 1) - vFluxZ(i, j, k)) * rdz;
    t += (tau12(i + 1, j, k) - tau12(i, j, k)) / P.dx + (tau22(i, j, k) - tau22(i, j - 1, k)) / P.dy
       + (rtau23(i, j, k + 1) - rtau23(i, j, k)) * rdz;
    t -= sp * (c(F_V, i, j, k) - cl.v0);
    if (P.fcor != 0.0) {
      let uAt = 0.25 * (c(F_U, i, j - 1, k) + c(F_U, i + 1, j - 1, k) + c(F_U, i, j, k) + c(F_U, i + 1, j, k));
      t -= P.fcor * (uAt + P.frameU - cl.ug);
    }
    sO[F_V * P.n + id] = sN[F_V * P.n + id] + P.dt * t;
  } else {
    sO[F_V * P.n + id] = sN[F_V * P.n + id];
  }
  // w（下面 k-1/2）
  if (k == 0) {
    sO[F_W * P.n + id] = 0.0;
  } else {
    let rdzw = 1.0 / (cl.rhoF * cl.dzw);
    var t = -(wFluxX(i + 1, j, k) - wFluxX(i, j, k)) / P.dx
            -(wFluxY(i, j + 1, k) - wFluxY(i, j, k)) / P.dy
            -(wFluxZ(i, j, k) - wFluxZ(i, j, k - 1)) * rdzw;
    // 浮力：上下のセル中心から面の高さへ線形補間
    let a = (cl.zf - col[k - 1].zc) / cl.dzw;
    t += mix(buoyancyCenter(i, j, k - 1), buoyancyCenter(i, j, k), a);
    // 拡散（応力テンソルの発散）
    let w0 = cw(i, j, k);
    t += (rtau13(i + 1, j, k) - rtau13(i, j, k)) / (P.dx * cl.rhoF);
    if (P.ny > 1u) { t += (rtau23(i, j + 1, k) - rtau23(i, j, k)) / (P.dy * cl.rhoF); }
    t += (cl.rho0 * tau33(i, j, k) - col[k - 1].rho0 * tau33(i, j, k - 1)) * rdzw;
    t -= spongeCoef(cl.zf) * w0;
    sO[F_W * P.n + id] = sN[F_W * P.n + id] + P.dt * t;
  }
}

// ============================================================ 圧力：発散と射影
// ∇·(ρ0 ∇φ) = ∇·(ρ0 u*) / dt を解き、u = u* − dt ∇φ（φ = p'/ρ0）
fn o(f: u32, i: i32, j: i32, k: i32) -> f32 { return sO[f * P.n + idx(i, j, k)]; }
fn ow(i: i32, j: i32, k: i32) -> f32 {
  if (k <= 0 || k >= i32(P.nz)) { return 0.0; }
  return sO[F_W * P.n + idx(i, j, k)];
}

@compute @workgroup_size(8, 8, 1)
fn csDivergence(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= P.nx || g.y >= P.ny || g.z >= P.nz) { return; }
  let i = i32(g.x); let j = i32(g.y); let k = i32(g.z);
  let cl = col[k];
  var dv = cl.rho0 * (o(F_U, i + 1, j, k) - o(F_U, i, j, k)) / P.dx;
  if (P.ny > 1u) { dv += cl.rho0 * (o(F_V, i, j + 1, k) - o(F_V, i, j, k)) / P.dy; }
  var rhoTop = 0.0;
  if (k + 1 < i32(P.nz)) { rhoTop = col[k + 1].rhoF; }
  dv += (rhoTop * ow(i, j, k + 1) - cl.rhoF * ow(i, j, k)) / cl.dzc;
  cplx[idx(i, j, k)] = vec2f(dv / P.dt, 0.0);
}

fn phi(i: i32, j: i32, k: i32) -> f32 { return cplx[idx(i, j, k)].x; }

@compute @workgroup_size(32, 2, 1)
fn csProject(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= P.nx || g.y >= P.ny || g.z >= P.nz) { return; }
  let i = i32(g.x); let j = i32(g.y); let k = i32(g.z);
  let id = idx(i, j, k);
  let p0 = phi(i, j, k);
  sO[F_U * P.n + id] -= P.dt * (p0 - phi(i - 1, j, k)) / P.dx;
  if (P.ny > 1u) { sO[F_V * P.n + id] -= P.dt * (p0 - phi(i, j - 1, k)) / P.dy; }
  if (k > 0) { sO[F_W * P.n + id] -= P.dt * (p0 - phi(i, j, k - 1)) / col[k].dzw; }
}
