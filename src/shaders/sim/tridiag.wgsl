// 水平フーリエ空間での鉛直三重対角ソルバ（Thomas 法）。1スレッド = 1波数の柱。
//   a_k φ_{k-1} + b_k φ_k + c_k φ_{k+1} = r_k
//   a_k = ρf[k]/(dzc[k] dzw[k]), c_k = ρf[k+1]/(dzc[k] dzw[k+1]), b_k = −a_k − c_k − ρ0[k] λ
// λ は離散ラプラシアンの固有値（2−2cos θ）/Δ²。λ=0（水平平均）は φ[0]=0 で固定。
// x 方向は実数 FFT なので kx = 0..nx/2 の列だけを解く（残りの半分は共役で、逆変換で復元する）。
// 前進消去の係数はスレッド内の配列に持たず（ローカルメモリに溢れて遅い）、
// d' は cplx の自分の位置に上書き、c' は diag の作業領域 [6n, 7n) に置く。

@compute @workgroup_size(8, 8, 1)
fn csTridiag(@builtin(global_invocation_id) g: vec3u) {
  if (g.x > P.nx / 2u || g.y >= P.ny) { return; }
  let i = g.x; let j = g.y;
  let nz = P.nz;
  let TWO_PI = 6.283185307;
  var lam = (2.0 - 2.0 * cos(TWO_PI * f32(i) / f32(P.nx))) / (P.dx * P.dx);
  if (P.ny > 1u) { lam += (2.0 - 2.0 * cos(TWO_PI * f32(j) / f32(P.ny))) / (P.dy * P.dy); }
  let zeroMode = (i == 0u && j == 0u);

  let plane = P.nx * P.ny;
  let base = i + P.nx * j;
  let cpBase = 6u * P.n;

  var cpPrev = 0.0;
  var dpPrev = vec2f(0.0);
  for (var k = 0u; k < nz; k++) {
    let cl = col[k];
    var a = 0.0;
    if (k > 0u) { a = cl.rhoF / (cl.dzc * cl.dzw); }
    var cc = 0.0;
    if (k + 1u < nz) { cc = col[k + 1u].rhoF / (cl.dzc * col[k + 1u].dzw); }
    var b = -a - cc - cl.rho0 * lam;
    let id = base + plane * k;
    var r = cplx[id];
    if (zeroMode && k == 0u) { a = 0.0; b = 1.0; cc = 0.0; r = vec2f(0.0); }
    let m = b - a * cpPrev;
    cpPrev = cc / m;
    dpPrev = (r - a * dpPrev) / m;
    diag[cpBase + id] = cpPrev;
    cplx[id] = dpPrev;
  }
  var x = dpPrev;
  for (var kk = i32(nz) - 2; kk >= 0; kk--) {
    let id = base + plane * u32(kk);
    x = cplx[id] - diag[cpBase + id] * x;
    cplx[id] = x;
  }
}
