/**
 * timestamp-query によるパスごとの GPU 時間計測。機能が無い環境では何もしない。
 * 読み戻しは非同期で、前回の読み戻しが終わるまで次は発行しない。
 */
export class GpuTimer {
  private querySet?: GPUQuerySet;
  private resolveBuf?: GPUBuffer;
  private readBuf?: GPUBuffer;
  private busy = false;
  private index = new Map<string, number>();
  /** パス名 → 平滑化した時間 [ms] */
  readonly ms = new Map<string, number>();

  constructor(device: GPUDevice, enabled: boolean, names: string[]) {
    if (!enabled) return;
    names.forEach((n, i) => this.index.set(n, i));
    const count = names.length * 2;
    this.querySet = device.createQuerySet({ type: 'timestamp', count });
    this.resolveBuf = device.createBuffer({ size: count * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    this.readBuf = device.createBuffer({ size: count * 8, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  }

  get enabled(): boolean {
    return !!this.querySet;
  }

  writes(name: string): GPURenderPassTimestampWrites | undefined {
    const i = this.index.get(name);
    if (!this.querySet || i === undefined || this.busy) return undefined;
    return { querySet: this.querySet, beginningOfPassWriteIndex: i * 2, endOfPassWriteIndex: i * 2 + 1 };
  }

  computeWrites(name: string): GPUComputePassTimestampWrites | undefined {
    return this.writes(name) as GPUComputePassTimestampWrites | undefined;
  }

  private pending = false;
  resolve(enc: GPUCommandEncoder): void {
    if (!this.querySet || this.busy) return;
    enc.resolveQuerySet(this.querySet, 0, this.querySet.count, this.resolveBuf!, 0);
    enc.copyBufferToBuffer(this.resolveBuf!, 0, this.readBuf!, 0, this.readBuf!.size);
    this.pending = true;
  }

  readback(): void {
    if (!this.pending || this.busy) return;
    this.pending = false;
    this.busy = true;
    this.readBuf!.mapAsync(GPUMapMode.READ).then(() => {
      const t = new BigInt64Array(this.readBuf!.getMappedRange().slice(0));
      this.readBuf!.unmap();
      for (const [name, i] of this.index) {
        const dt = Number(t[i * 2 + 1] - t[i * 2]) / 1e6;
        if (dt >= 0 && dt < 1000) {
          const prev = this.ms.get(name) ?? dt;
          this.ms.set(name, prev * 0.9 + dt * 0.1);
        }
      }
      this.busy = false;
    });
  }
}
