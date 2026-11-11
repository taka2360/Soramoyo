// 細部合成用のタイル可能な 3D ノイズ（128³, rgba8unorm）
//  r: Perlin–Worley（低周波）  g, b, a: Worley FBM（周波数 4, 8, 16 セル）
// 雲の「形」はシミュレーションが決め、このノイズは格子より細かい縁の凹凸にだけ使う。

@group(0) @binding(0) var noiseOut: texture_storage_3d<rgba8unorm, write>;

const NS: f32 = 128.0;

fn hash3(p: vec3u) -> vec3f {
  var x = p * vec3u(1664525u, 1013904223u, 2654435761u);
  x = (x.yzx ^ (x >> vec3u(16u))) * vec3u(2246822519u, 3266489917u, 668265263u);
  x = x ^ (x >> vec3u(13u));
  return vec3f(x & vec3u(0xffffffu)) / f32(0xffffff);
}

fn worley(p: vec3f, cells: f32) -> f32 {
  let q = p * cells;
  let c = floor(q);
  var dmin = 1e9;
  for (var z = -1; z <= 1; z++) {
    for (var y = -1; y <= 1; y++) {
      for (var x = -1; x <= 1; x++) {
        let cc = c + vec3f(f32(x), f32(y), f32(z));
        let w = (cc % cells + cells) % cells; // 周期
        let fp = cc + hash3(vec3u(w));
        dmin = min(dmin, length(fp - q));
      }
    }
  }
  return clamp(1.0 - dmin, 0.0, 1.0);
}

fn grad(c: vec3f, cells: f32) -> vec3f {
  let w = (c % cells + cells) % cells;
  return normalize(hash3(vec3u(w) + vec3u(17u, 31u, 47u)) * 2.0 - 1.0);
}
fn perlin(p: vec3f, cells: f32) -> f32 {
  let q = p * cells;
  let i = floor(q);
  let f = q - i;
  let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
  var r = 0.0;
  for (var z = 0; z < 2; z++) {
    for (var y = 0; y < 2; y++) {
      for (var x = 0; x < 2; x++) {
        let o = vec3f(f32(x), f32(y), f32(z));
        let d = dot(grad(i + o, cells), f - o);
        let wgt = mix(1.0 - u, u, o);
        r += d * wgt.x * wgt.y * wgt.z;
      }
    }
  }
  return r * 0.5 + 0.5;
}

@compute @workgroup_size(4, 4, 4)
fn csNoise(@builtin(global_invocation_id) g: vec3u) {
  if (any(g >= vec3u(128u))) { return; }
  let p = (vec3f(g) + 0.5) / NS;
  let pf = perlin(p, 4.0) * 0.5 + perlin(p, 8.0) * 0.3 + perlin(p, 16.0) * 0.2;
  let w1 = worley(p, 4.0) * 0.625 + worley(p, 8.0) * 0.25 + worley(p, 16.0) * 0.125;
  let w2 = worley(p, 8.0) * 0.625 + worley(p, 16.0) * 0.25 + worley(p, 32.0) * 0.125;
  let w3 = worley(p, 16.0) * 0.625 + worley(p, 32.0) * 0.25 + worley(p, 64.0) * 0.125;
  // Perlin–Worley：Perlin を Worley で再マップ
  let pw = clamp((pf - (1.0 - w1)) / (1.0 - (1.0 - w1) + 1e-4), 0.0, 1.0);
  textureStore(noiseOut, g, vec4f(pw, w1, w2, w3));
}
