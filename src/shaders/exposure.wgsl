// 自動露出：画面の対数平均輝度を求め、目の順応のように時間をかけて追従させる。
// 結果は storage バッファ（[0] = 順応輝度 cd/m², [1] = このフレームのプリエクスポージャー）に保持し、トーンマップが直接読む（CPU 読み戻しなし）。

@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var hdrTex: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> expo: array<f32, 4>;

var<workgroup> sh: array<f32, 256>;

@compute @workgroup_size(16, 16)
fn csMain(@builtin(local_invocation_id) lid: vec3u, @builtin(local_invocation_index) li: u32) {
  // このフレームの HDR に掛かっているプリエクスポージャー（sky.wgsl の preExposure と同じ式）
  let pre = 1.0 / max(expo[0], 1e-4);
  let dims = vec2f(textureDimensions(hdrTex));
  var s = 0.0;
  var wsum = 0.0;
  for (var j = 0u; j < 4u; j++) {
    for (var i = 0u; i < 4u; i++) {
      let uv = (vec2f(lid.xy * 4u + vec2u(i, j)) + 0.5) / 64.0;
      let c = textureLoad(hdrTex, vec2i(uv * dims), 0).rgb / (F.whitePoint * pre);
      // 画面中央をやや重視（中央重点測光）
      let w = 1.0 - 0.5 * length(uv - 0.5);
      s += log(max(luminance(c), 1e-6)) * w;
      wsum += w;
    }
  }
  sh[li] = s / wsum;
  workgroupBarrier();
  for (var k = 128u; k > 0u; k >>= 1u) {
    if (li < k) { sh[li] += sh[li + k]; }
    workgroupBarrier();
  }
  if (li == 0u) {
    let tgt = sh[0] / 256.0;            // log 輝度
    let prev = expo[0];
    var cur = tgt;
    if (prev > 0.0) {
      let lp = log(prev);
      // 明るくなる方向は速く、暗くなる方向はゆっくり順応（実時間）
      let tau = select(3.0, 0.6, tgt > lp);
      cur = mix(lp, tgt, 1.0 - exp(-F.dtReal / tau));
    }
    expo[0] = exp(cur);
    expo[1] = pre;  // トーンマップがこのフレームの HDR を絶対輝度に戻すのに使う
  }
}
