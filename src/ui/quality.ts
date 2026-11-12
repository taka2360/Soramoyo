import type { RenderSettings } from '../render/renderer';
import type { CloudSettings } from '../render/clouds';
import type { SimManager } from '../sim/manager';

/**
 * 画質プリセット。雲の計算そのもの（格子・物理）は変えず、描画の細かさと1フレームの計算量だけを変える。
 * 計算領域の大きさ（格子数）は「大気の状態」の選択肢で選ぶ（128² / 256²）。
 */
export interface QualityPreset {
  renderScale: number;
  cloudScale: number;
  stepScale: number;
  minStepM: number;
  maxSteps: number;
  maxDistKm: number;
  shaftSamples: number;
  detailHQ: boolean;
  simStepsPerFrame: number;
  /** 1フレームに雲の計算へ割く GPU 時間 [ms] */
  simBudgetMs: number;
}

export const QUALITY: Record<string, QualityPreset> = {
  '低（軽い）': {
    renderScale: 0.75, cloudScale: 0.333, stepScale: 0.025, minStepM: 60, maxSteps: 160,
    maxDistKm: 60, shaftSamples: 10, detailHQ: false, simStepsPerFrame: 2, simBudgetMs: 4,
  },
  '中': {
    renderScale: 1, cloudScale: 0.5, stepScale: 0.02, minStepM: 50, maxSteps: 220,
    maxDistKm: 90, shaftSamples: 16, detailHQ: false, simStepsPerFrame: 3, simBudgetMs: 6,
  },
  '高（既定）': {
    renderScale: 1, cloudScale: 0.5, stepScale: 0.015, minStepM: 40, maxSteps: 320,
    maxDistKm: 120, shaftSamples: 24, detailHQ: true, simStepsPerFrame: 3, simBudgetMs: 8,
  },
  '最高（重い）': {
    renderScale: 1, cloudScale: 1, stepScale: 0.01, minStepM: 25, maxSteps: 480,
    maxDistKm: 160, shaftSamples: 32, detailHQ: true, simStepsPerFrame: 6, simBudgetMs: 14,
  },
};

export const DEFAULT_QUALITY = '高（既定）';

export function applyQuality(name: string, rs: RenderSettings, cs: CloudSettings, sim: SimManager): void {
  const q = QUALITY[name];
  if (!q) return;
  rs.renderScale = q.renderScale;
  rs.cloudScale = q.cloudScale;
  cs.stepScale = q.stepScale;
  cs.minStepM = q.minStepM;
  cs.maxSteps = q.maxSteps;
  cs.maxDistKm = q.maxDistKm;
  cs.shaftSamples = q.shaftSamples;
  cs.detailHQ = q.detailHQ;
  sim.maxStepsPerFrame = q.simStepsPerFrame;
  sim.simBudgetMs = q.simBudgetMs;
}
