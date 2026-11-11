// 力学：渦粘性、スカラー・運動量の RK 段更新、発散、射影。
// （common.wgsl の後ろに連結）

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
    let thvp = c(F_TH, i, j, kp) * (1.0 + 0.608 * c(F_QV, i, j, kp) - c(F_QC, i, j, kp) - c(F_QR, i, j, kp));
    let thvm = c(F_TH, i, j, km) * (1.0 + 0.608 * c(F_QV, i, j, km) - c(F_QC, i, j, km) - c(F_QR, i, j, km));
    let n2 = GRAV / cl.th0 * (thvp - thvm) / dzz;
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
// 下面 k の鉛直フラックス ρ w s（境界付近は次数を下げる）
fn sFluxZ(f: u32, i: i32, j: i32, k: i32) -> f32 {
  let nz = i32(P.nz);
  if (k <= 0 || k >= nz) { return 0.0; }
  let w = cw(i, j, k);
  var s: f32;
  if (k >= 3 && k <= nz - 3) {
    s = interp5(c(f, i, j, k - 3), c(f, i, j, k - 2), c(f, i, j, k - 1), c(f, i, j, k), c(f, i, j, k + 1), c(f, i, j, k + 2), w);
  } else if (k >= 2 && k <= nz - 2) {
    s = interp3(c(f, i, j, k - 2), c(f, i, j, k - 1), c(f, i, j, k), c(f, i, j, k + 1), w);
  } else {
    s = 0.5 * (c(f, i, j, k - 1) + c(f, i, j, k));
  }
  return col[k].rhoF * w * s;
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
  for (var f = F_QV; f <= F_QR; f++) {
    let out = (max(sFluxX(f, i + 1, j, k), 0.0) - min(sFluxX(f, i, j, k), 0.0)) / P.dx
            + (max(sFluxY(f, i, j + 1, k), 0.0) - min(sFluxY(f, i, j, k), 0.0)) / P.dy
            + (max(sFluxZ(f, i, j, k + 1), 0.0) - min(sFluxZ(f, i, j, k), 0.0)) / (cl.rho0 * cl.dzc);
    var r = 1.0;
    let avail = max(sN[f * P.n + id], 0.0);
    if (out * P.dt > avail) { r = avail / (out * P.dt); }
    diag[dPd(f, id)] = r;
  }
}

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

  // 拡散（基本場からの偏差に対して。K = ν/Pr）
  let s0 = c(f, i, j, k) - baseOf(f, k);
  let nu0 = nuAt(i, j, k);
  let ip = c(f, i + 1, j, k) - baseOf(f, k); let im = c(f, i - 1, j, k) - baseOf(f, k);
  let jp = c(f, i, j + 1, k) - baseOf(f, k); let jm = c(f, i, j - 1, k) - baseOf(f, k);
  let kx1 = 0.5 * (nu0 + nuAt(i + 1, j, k)); let kx0 = 0.5 * (nu0 + nuAt(i - 1, j, k));
  let ky1 = 0.5 * (nu0 + nuAt(i, j + 1, k)); let ky0 = 0.5 * (nu0 + nuAt(i, j - 1, k));
  var diff = (kx1 * (ip - s0) - kx0 * (s0 - im)) / (P.dx * P.dx);
  if (P.ny > 1u) { diff += (ky1 * (jp - s0) - ky0 * (s0 - jm)) / (P.dy * P.dy); }
  let nz = i32(P.nz);
  var fz1 = 0.0; var fz0 = 0.0;
  if (k + 1 < nz) {
    let sp = c(f, i, j, k + 1) - baseOf(f, k + 1);
    fz1 = col[k + 1].rhoF * 0.5 * (nu0 + nuAt(i, j, k + 1)) * (sp - s0) / col[k + 1].dzw;
  }
  if (k > 0) {
    let sm = c(f, i, j, k - 1) - baseOf(f, k - 1);
    fz0 = cl.rhoF * 0.5 * (nu0 + nuAt(i, j, k - 1)) * (s0 - sm) / cl.dzw;
  }
  diff += (fz1 - fz0) / (cl.rho0 * cl.dzc);
  t += diff / P.prandtl;

  if (f == F_TH) { t -= spongeCoef(cl.zc) * s0; }
  return t;
}

@compute @workgroup_size(8, 8, 1)
fn csScalars(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= P.nx || g.y >= P.ny || g.z >= P.nz) { return; }
  let i = i32(g.x); let j = i32(g.y); let k = i32(g.z);
  let id = idx(i, j, k);
  sO[F_TH * P.n + id] = sN[F_TH * P.n + id] + P.dt * scalarTendency(F_TH, i, j, k, false);
  if ((P.flags & FLAG_MOIST) != 0u) {
    let lim = P.stage == 2u && (P.flags & FLAG_PD) != 0u;
    sO[F_QV * P.n + id] = sN[F_QV * P.n + id] + P.dt * scalarTendency(F_QV, i, j, k, lim);
    sO[F_QC * P.n + id] = sN[F_QC * P.n + id] + P.dt * scalarTendency(F_QC, i, j, k, lim);
    sO[F_QR * P.n + id] = sN[F_QR * P.n + id] + P.dt * scalarTendency(F_QR, i, j, k, lim);
  } else {
    sO[F_QV * P.n + id] = sN[F_QV * P.n + id];
    sO[F_QC * P.n + id] = 0.0;
    sO[F_QR * P.n + id] = 0.0;
  }
}

// ============================================================ 運動量
// 鉛直に u（または v）を面 k で補間（境界付近は次数を下げる）
fn zInterpCenter(f: u32, i: i32, j: i32, k: i32, vel: f32) -> f32 {
  let nz = i32(P.nz);
  if (k >= 3 && k <= nz - 3) {
    return interp5(c(f, i, j, k - 3), c(f, i, j, k - 2), c(f, i, j, k - 1), c(f, i, j, k), c(f, i, j, k + 1), c(f, i, j, k + 2), vel);
  } else if (k >= 2 && k <= nz - 2) {
    return interp3(c(f, i, j, k - 2), c(f, i, j, k - 1), c(f, i, j, k), c(f, i, j, k + 1), vel);
  }
  return 0.5 * (c(f, i, j, k - 1) + c(f, i, j, k));
}

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
  return col[k].rhoF * vel * zInterpCenter(F_U, i, j, k, vel);
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
  return col[k].rhoF * vel * zInterpCenter(F_V, i, j, k, vel);
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
  let nz = i32(P.nz);
  let vel = 0.5 * (cw(i, j, kc) + cw(i, j, kc + 1));
  var s: f32;
  if (kc - 2 >= 0 && kc + 3 <= nz) {
    s = interp5(cw(i, j, kc - 2), cw(i, j, kc - 1), cw(i, j, kc), cw(i, j, kc + 1), cw(i, j, kc + 2), cw(i, j, kc + 3), vel);
  } else if (kc - 1 >= 0 && kc + 2 <= nz) {
    s = interp3(cw(i, j, kc - 1), cw(i, j, kc), cw(i, j, kc + 1), cw(i, j, kc + 2), vel);
  } else {
    s = 0.5 * (cw(i, j, kc) + cw(i, j, kc + 1));
  }
  return col[kc].rho0 * vel * s;
}

fn buoyancyCenter(i: i32, j: i32, k: i32) -> f32 {
  let cl = col[k];
  return GRAV * ((c(F_TH, i, j, k) - cl.th0) / cl.th0 + 0.608 * (c(F_QV, i, j, k) - cl.qv0)
                 - c(F_QC, i, j, k) - c(F_QR, i, j, k));
}

// 速度成分の拡散（ν∇²、鉛直は自由すべり）
fn momDiff(f: u32, i: i32, j: i32, k: i32, nu: f32) -> f32 {
  let s0 = c(f, i, j, k);
  var d = (c(f, i + 1, j, k) - 2.0 * s0 + c(f, i - 1, j, k)) / (P.dx * P.dx);
  if (P.ny > 1u) { d += (c(f, i, j + 1, k) - 2.0 * s0 + c(f, i, j - 1, k)) / (P.dy * P.dy); }
  let nz = i32(P.nz);
  var g1 = 0.0; var g0 = 0.0;
  if (k + 1 < nz) { g1 = (c(f, i, j, k + 1) - s0) / col[k + 1].dzw; }
  if (k > 0) { g0 = (s0 - c(f, i, j, k - 1)) / col[k].dzw; }
  d += (g1 - g0) / col[k].dzc;
  return nu * d;
}

@compute @workgroup_size(8, 8, 1)
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
    t += momDiff(F_U, i, j, k, 0.5 * (nuAt(i - 1, j, k) + nuAt(i, j, k)));
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
    t += momDiff(F_V, i, j, k, 0.5 * (nuAt(i, j - 1, k) + nuAt(i, j, k)));
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
    // 拡散（水平のみ＋鉛直）
    let w0 = cw(i, j, k);
    let nu = 0.5 * (nuAt(i, j, k - 1) + nuAt(i, j, k));
    var d = (cw(i + 1, j, k) - 2.0 * w0 + cw(i - 1, j, k)) / (P.dx * P.dx);
    if (P.ny > 1u) { d += (cw(i, j + 1, k) - 2.0 * w0 + cw(i, j - 1, k)) / (P.dy * P.dy); }
    d += ((cw(i, j, k + 1) - w0) / cl.dzc - (w0 - cw(i, j, k - 1)) / col[k - 1].dzc) / cl.dzw;
    t += nu * d;
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

@compute @workgroup_size(8, 8, 1)
fn csProject(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= P.nx || g.y >= P.ny || g.z >= P.nz) { return; }
  let i = i32(g.x); let j = i32(g.y); let k = i32(g.z);
  let id = idx(i, j, k);
  let p0 = phi(i, j, k);
  sO[F_U * P.n + id] -= P.dt * (p0 - phi(i - 1, j, k)) / P.dx;
  if (P.ny > 1u) { sO[F_V * P.n + id] -= P.dt * (p0 - phi(i, j - 1, k)) / P.dy; }
  if (k > 0) { sO[F_W * P.n + id] -= P.dt * (p0 - phi(i, j, k - 1)) / col[k].dzw; }
}
