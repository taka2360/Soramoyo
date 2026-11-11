/**
 * WebGPU 初期化。
 * - 雲シミュレーションで大きなバッファを使うため、アダプタが許す上限を明示的に要求する
 *   （既定の maxStorageBufferBindingSize は 128MB しかない）。
 * - 任意機能（timestamp-query, float32-filterable）は使えれば有効化する。
 */
export interface Gpu {
  adapter: GPUAdapter;
  device: GPUDevice;
  context: GPUCanvasContext;
  canvasFormat: GPUTextureFormat;
  hasTimestamps: boolean;
  hasFloat32Filterable: boolean;
}

export async function initGpu(canvas: HTMLCanvasElement): Promise<Gpu> {
  if (!('gpu' in navigator)) {
    throw new Error('このブラウザは WebGPU に対応していません（Chrome 113 以降を使ってください）。');
  }
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('WebGPU アダプタを取得できませんでした。');

  const wanted: GPUFeatureName[] = ['timestamp-query', 'float32-filterable'];
  const requiredFeatures = wanted.filter((f) => adapter.features.has(f));

  const L = adapter.limits;
  const requiredLimits: Record<string, number> = {
    maxStorageBufferBindingSize: L.maxStorageBufferBindingSize,
    maxBufferSize: L.maxBufferSize,
    maxStorageBuffersPerShaderStage: L.maxStorageBuffersPerShaderStage,
    maxComputeWorkgroupStorageSize: L.maxComputeWorkgroupStorageSize,
    maxComputeInvocationsPerWorkgroup: L.maxComputeInvocationsPerWorkgroup,
    maxComputeWorkgroupSizeX: L.maxComputeWorkgroupSizeX,
    maxComputeWorkgroupSizeY: L.maxComputeWorkgroupSizeY,
    maxComputeWorkgroupSizeZ: L.maxComputeWorkgroupSizeZ,
    maxTextureDimension3D: L.maxTextureDimension3D,
  };

  const device = await adapter.requestDevice({ requiredFeatures, requiredLimits });
  device.lost.then((info) => {
    console.error(`[Soramoyo] GPU device lost: ${info.reason} ${info.message}`);
    showFatal(`GPU デバイスが失われました（${info.reason}）。ページを再読み込みしてください。\n${info.message}`);
  });
  device.addEventListener('uncapturederror', (e) => {
    console.error('[Soramoyo] WebGPU error:', (e as GPUUncapturedErrorEvent).error.message);
  });

  const context = canvas.getContext('webgpu');
  if (!context) throw new Error('WebGPU キャンバスコンテキストを取得できませんでした。');
  const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format: canvasFormat, alphaMode: 'opaque' });

  return {
    adapter,
    device,
    context,
    canvasFormat,
    hasTimestamps: requiredFeatures.includes('timestamp-query'),
    hasFloat32Filterable: requiredFeatures.includes('float32-filterable'),
  };
}

export function showFatal(msg: string): void {
  const el = document.getElementById('fatal');
  if (!el) return;
  el.style.display = 'flex';
  el.textContent = msg;
}
