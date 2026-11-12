// 水平フーリエ空間での鉛直三重対角ソルバ（Thomas 法）。1スレッド = 1波数の柱。
//   a_k φ_{k-1} + b_k φ_k + c_k φ_{k+1} = r_k
//   a_k = ρf[k]/(dzc[k] dzw[k]), c_k = ρf[k+1]/(dzc[k] dzw[k+1]), b_k = −a_k − c_k − ρ0[k] λ
// λ は離散ラプラシアンの固有値（2−2cos θ）/Δ²。λ=0（水平平均）は φ[0]=0 で固定。

const MAXNZ: u32 = 256u;

@compute @workgroup_size(8, 8, 1)
fn csTridiag(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= P.nx || g.y >= P.ny) { return; }
  let i = g.x; let j = g.y;
  // 右辺は実数なので、波数 (i, j) と (−i, −j) の係数は互いに複素共役で、解も共役になる。
  // 半分の波数だけ解き、相方には共役を書く（計算量が半分）。
  let pi_ = (P.nx - i) % P.nx;
  let pj = (P.ny - j) % P.ny;
  if (i > P.nx / 2u) { return; }
  if ((i == 0u || i == P.nx / 2u) && j > P.ny / 2u) { return; }
  let partner = pi_ + P.nx * pj;
  let hasPartner = partner != i + P.nx * j;
  let nz = P.nz;
  let TWO_PI = 6.283185307;
  var lam = (2.0 - 2.0 * cos(TWO_PI * f32(i) / f32(P.nx))) / (P.dx * P.dx);
  if (P.ny > 1u) { lam += (2.0 - 2.0 * cos(TWO_PI * f32(j) / f32(P.ny))) / (P.dy * P.dy); }
  let zeroMode = (i == 0u && j == 0u);

  var cp: array<f32, MAXNZ>;
  var dp: array<vec2f, MAXNZ>;
  let plane = P.nx * P.ny;
  let base = i + P.nx * j;

  for (var k = 0u; k < nz; k++) {
    let cl = col[k];
    var a = 0.0;
    if (k > 0u) { a = cl.rhoF / (cl.dzc * cl.dzw); }
    var cc = 0.0;
    if (k + 1u < nz) { cc = col[k + 1u].rhoF / (cl.dzc * col[k + 1u].dzw); }
    var b = -a - cc - cl.rho0 * lam;
    var r = cplx[base + plane * k];
    if (zeroMode && k == 0u) { a = 0.0; b = 1.0; cc = 0.0; r = vec2f(0.0); }
    if (k == 0u) {
      cp[0] = cc / b;
      dp[0] = r / b;
    } else {
      let m = b - a * cp[k - 1u];
      cp[k] = cc / m;
      dp[k] = (r - a * dp[k - 1u]) / m;
    }
  }
  var x = dp[nz - 1u];
  cplx[base + plane * (nz - 1u)] = x;
  if (hasPartner) { cplx[partner + plane * (nz - 1u)] = vec2f(x.x, -x.y); }
  for (var kk = i32(nz) - 2; kk >= 0; kk--) {
    let k = u32(kk);
    x = dp[k] - cp[k] * x;
    cplx[base + plane * k] = x;
    if (hasPartner) { cplx[partner + plane * k] = vec2f(x.x, -x.y); }
  }
}
