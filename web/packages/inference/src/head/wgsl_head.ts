import { Kernel, createStorageBuffer, uploadStorage } from "@gaussifier/wgsl";
import type { HeadBackend } from "./types.js";
import { DEFAULT_CONV, OUT1X1_WGSL, POOL_WGSL, UPSAMPLE_WGSL, convShader, convTile, type ConvConfig } from "./shaders.js";

export interface HeadLayout {
  format: number;
  dtype: string;
  arch: { in_channels: number; out_channels: number; hidden: number; aux_hidden: number; depth: number; kernel_size: number };
  tensors: Array<{ name: string; shape: number[]; offset: number; count: number }>;
}

/** One layer for both nets: weights of net 0 then net 1 in each buffer, net 1 at `off` floats. */
interface PairLayer {
  convW: GPUBuffer; convB: GPUBuffer;
  /** Per net: channel counts and the float offset of that net's slice in each weight buffer. */
  cin: number[]; cout: number[]; wOff: number[]; bOff: number[]; res: number;
}
interface Nets { hidden: number[]; layers: PairLayer[]; outW: GPUBuffer[]; outB: GPUBuffer[] }

interface Step { kernel: Kernel; buffers: (GPUBuffer | "features")[]; x: number; y: number; z: number }

interface Plan {
  width: number; height: number;
  steps: Step[];
  buffers: GPUBuffer[];
  delta: GPUBuffer;
}

/** 1-D work split into a 2-D dispatch that respects maxComputeWorkgroupsPerDimension. */
function grid(n: number, size = 256): { x: number; y: number } {
  const total = Math.max(1, Math.ceil(n / size));
  const x = Math.min(total, 65535);
  return { x, y: Math.ceil(total / x) };
}

/** A buffer holding both nets' copies of one activation: net 0 at 0, net 1 at off[1] floats. */
interface Shared { buf: GPUBuffer; off: number[] }
/** Stands in for the engine's features buffer, bound at record time. */
const FEATURES_INPUT: Shared = { buf: null as unknown as GPUBuffer, off: [0, 0] };

/** Collects the buffers, uniforms and dispatches of a plan while prepare() wires the UNets. */
class PlanBuilder {
  readonly steps: Step[] = [];
  readonly buffers: GPUBuffer[] = [];
  readonly uniforms: GPUBuffer[] = [];
  constructor(private readonly device: GPUDevice, readonly width: number, readonly height: number, private readonly hidden: number[]) {}
  dims(res: number): { w: number; h: number; hw: number } {
    return { w: this.width / res, h: this.height / res, hw: (this.width / res) * (this.height / res) };
  }
  alloc(floats: number, label: string): GPUBuffer {
    const b = createStorageBuffer(this.device, floats * 4, label);
    this.buffers.push(b);
    return b;
  }
  /** An activation of `mult` times the hidden width per net at resolution `res`, for both nets. */
  shared(mult: number, res: number, label: string): Shared {
    const hw = this.dims(res).hw;
    const sizes = this.hidden.map((hn) => mult * hn * hw);
    return { buf: this.alloc(sizes[0] + sizes[1], label), off: [0, sizes[0]] };
  }
  /** A uniform buffer of u32 words, padded to 16 bytes. */
  uniform(values: number[]): GPUBuffer {
    const size = Math.ceil(values.length / 4) * 16;
    const buf = new Uint32Array(size / 4);
    buf.set(values);
    const gpu = this.device.createBuffer({ size, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(gpu, 0, buf);
    this.uniforms.push(gpu);
    return gpu;
  }
}

/**
 * Custom WGSL implementation of CorrectionHead (two UNets of depth 2 summed).
 * Reads the [1,17,H,W] features buffer, writes a [1,8,H,W] delta buffer.
 */
export class WgslHead implements HeadBackend {
  readonly kind = "wgsl" as const;
  private conv3x3: Kernel;
  private pool2x2: Kernel;
  private upsample2x: Kernel;
  private out1x1: Kernel;
  private nets: Nets;
  private plan: Plan | null = null;
  private uniforms: GPUBuffer[] = [];
  readonly tile: { tileW: number; tileH: number; threads: number };

  constructor(readonly device: GPUDevice, readonly layout: HeadLayout, weights: Float32Array, readonly convCfg: ConvConfig = DEFAULT_CONV) {
    if (layout.format !== 2 || layout.dtype !== "float32") throw new Error("unsupported head layout");
    const a = layout.arch;
    if (a.kernel_size !== 3 || a.depth !== 2 || a.in_channels !== 17 || a.out_channels !== 8) throw new Error("head architecture not supported by the WGSL head");
    this.tile = convTile(convCfg);
    const S = "storage" as const, R = "read-only-storage" as const, U = "uniform" as const;
    this.conv3x3 = Kernel.create(device, { label: "head.conv3x3", code: convShader(convCfg), entryPoint: "conv3x3", bindings: [U, R, R, R, S] });
    this.pool2x2 = Kernel.create(device, { label: "head.pool2x2", code: POOL_WGSL, entryPoint: "pool2x2", bindings: [U, R, S] });
    this.upsample2x = Kernel.create(device, { label: "head.upsample2x", code: UPSAMPLE_WGSL, entryPoint: "upsample2x", bindings: [U, R, S] });
    this.out1x1 = Kernel.create(device, { label: "head.out1x1", code: OUT1X1_WGSL, entryPoint: "out1x1", bindings: [U, R, R, R, R, R, S] });
    this.nets = this.loadWeights(layout, weights);
  }

  /** One row per folded 3x3 convolution layer of a net of hidden width h, in execution order. */
  private static layerSpec(h: number): Array<{ conv: string; cin: number; cout: number; res: number }> {
    return [
      { conv: "enc_blocks.0.0", cin: 17, cout: h, res: 1 }, { conv: "enc_blocks.0.3", cin: h, cout: h, res: 1 },
      { conv: "enc_blocks.1.0", cin: h, cout: 2 * h, res: 2 }, { conv: "enc_blocks.1.3", cin: 2 * h, cout: 2 * h, res: 2 },
      { conv: "bottleneck.0", cin: 2 * h, cout: 4 * h, res: 4 }, { conv: "bottleneck.3", cin: 4 * h, cout: 4 * h, res: 4 },
      { conv: "merge_blocks.0.0", cin: 6 * h, cout: 2 * h, res: 2 }, { conv: "merge_blocks.0.3", cin: 2 * h, cout: 2 * h, res: 2 },
      { conv: "merge_blocks.1.0", cin: 3 * h, cout: h, res: 1 }, { conv: "merge_blocks.1.3", cin: h, cout: h, res: 1 },
    ];
  }

  /** Upload the weights of both nets, each layer's tensors packed net 0 then net 1 into one buffer. */
  private loadWeights(layout: HeadLayout, weights: Float32Array): Nets {
    const byName = new Map(layout.tensors.map((t) => [t.name, t]));
    const tensor = (name: string, expected: number[]): Float32Array => {
      const t = byName.get(name);
      if (!t) throw new Error(`head weights missing ${name}`);
      if (t.shape.length !== expected.length || t.shape.some((v, i) => v !== expected[i])) throw new Error(`head weight ${name} has shape ${t.shape} expected ${expected}`);
      return weights.subarray(t.offset, t.offset + t.count);
    };
    const pair = (parts: Float32Array[], label: string): { buf: GPUBuffer; off: number[] } => {
      const packed = new Float32Array(parts.reduce((s, p) => s + p.length, 0));
      const off: number[] = [];
      let o = 0;
      for (const p of parts) { off.push(o); packed.set(p, o); o += p.length; }
      return { buf: uploadStorage(this.device, packed, label), off };
    };
    const prefixes = ["net", "aux_net"];
    const hidden = [layout.arch.hidden, layout.arch.aux_hidden];
    const specs = hidden.map(WgslHead.layerSpec);
    const layers: PairLayer[] = specs[0].map((_, i) => {
      const per = prefixes.map((prefix, n) => {
        const L = specs[n][i];
        return { cin: L.cin, cout: L.cout, w: tensor(`${prefix}.${L.conv}.weight`, [L.cout, L.cin, 3, 3]), b: tensor(`${prefix}.${L.conv}.bias`, [L.cout]) };
      });
      const w = pair(per.map((l) => l.w), `conv${i}.w`), b = pair(per.map((l) => l.b), `conv${i}.b`);
      return { convW: w.buf, convB: b.buf, cin: per.map((l) => l.cin), cout: per.map((l) => l.cout), wOff: w.off, bOff: b.off, res: specs[0][i].res };
    });
    return {
      hidden, layers,
      outW: prefixes.map((prefix, n) => uploadStorage(this.device, tensor(`${prefix}.out.weight`, [8, hidden[n], 1, 1]), `${prefix}.out.w`)),
      outB: prefixes.map((prefix) => uploadStorage(this.device, tensor(`${prefix}.out.bias`, [8]), `${prefix}.out.b`)),
    };
  }

  /** Buffers and dispatch list for one frame size; both nets share every buffer, net 1 at `off[1]` floats. */
  prepare(width: number, height: number): void {
    if (this.plan && this.plan.width === width && this.plan.height === height) return;
    this.disposePlan();
    if (width % 4 || height % 4) throw new Error("head input size must be a multiple of 4");
    const b = new PlanBuilder(this.device, width, height, this.nets.hidden);
    const actA1 = b.shared(1, 1, "actA1"), actB1 = b.shared(1, 1, "actB1"), cat1 = b.shared(3, 1, "cat1");
    const actA2 = b.shared(2, 2, "actA2"), actB2 = b.shared(2, 2, "actB2"), cat0 = b.shared(6, 2, "cat0"), pooled0 = b.shared(1, 2, "pooled0");
    const actA4 = b.shared(4, 4, "actA4"), actB4 = b.shared(4, 4, "actB4"), pooled1 = b.shared(2, 4, "pooled1");
    const h = (k: number) => this.nets.hidden.map((hn) => k * hn);
    const L = this.nets.layers;
    // Encoder at full resolution, then half, bottleneck at quarter; decoder back up with skip concatenation.
    this.conv(b, L[0], FEATURES_INPUT, actA1);
    this.conv(b, L[1], actA1, cat1, h(2));
    this.pool(b, cat1, h(2), h(1), 1, pooled0);
    this.conv(b, L[2], pooled0, actA2);
    this.conv(b, L[3], actA2, cat0, h(4));
    this.pool(b, cat0, h(4), h(2), 2, pooled1);
    this.conv(b, L[4], pooled1, actA4);
    this.conv(b, L[5], actA4, actB4);
    this.upsample(b, actB4, h(4), 4, cat0, h(0));
    this.conv(b, L[6], cat0, actA2);
    this.conv(b, L[7], actA2, actB2);
    this.upsample(b, actB2, h(2), 2, cat1, h(0));
    this.conv(b, L[8], cat1, actA1);
    this.conv(b, L[9], actA1, actB1);
    const hw = width * height;
    const delta = b.alloc(8 * hw, "head.delta");
    b.steps.push({ kernel: this.out1x1, buffers: [b.uniform([hw, this.nets.hidden[0], this.nets.hidden[1], actB1.off[1]]), actB1.buf, this.nets.outW[0], this.nets.outB[0], this.nets.outW[1], this.nets.outB[1], delta], ...grid(hw), z: 1 });
    this.plan = { width, height, steps: b.steps, buffers: b.buffers, delta };
    this.uniforms = b.uniforms;
  }

  /** 3x3 conv of both nets: z slices below `zsplit` belong to net 0. */
  private conv(b: PlanBuilder, L: PairLayer, input: Shared, dst: Shared, dstCh = [0, 0]): void {
    const d = b.dims(L.res);
    const zs = L.cout.map((c) => Math.ceil(c / this.convCfg.CO));
    const u = b.uniform([
      d.w, d.h, L.cin[0], L.cout[0], input.off[0], L.wOff[0], L.bOff[0], dst.off[0] + dstCh[0] * d.hw,
      d.w, d.h, L.cin[1], L.cout[1], input.off[1], L.wOff[1], L.bOff[1], dst.off[1] + dstCh[1] * d.hw,
      zs[0], 0, 0, 0,
    ]);
    b.steps.push({ kernel: this.conv3x3, buffers: [u, input === FEATURES_INPUT ? "features" : input.buf, L.convW, L.convB, dst.buf], x: Math.ceil(d.w / this.tile.tileW), y: Math.ceil(d.h / this.tile.tileH), z: zs[0] + zs[1] });
  }

  /** 2x2 max pool of C channels starting at channel srcCh of src, at resolution `res`. */
  private pool(b: PlanBuilder, src: Shared, srcCh: number[], C: number[], res: number, dst: Shared): void {
    const d = b.dims(res);
    b.steps.push({ kernel: this.pool2x2, buffers: [b.uniform([
      d.w, d.h, C[0], srcCh[0], src.off[0], dst.off[0], 0, 0,
      d.w, d.h, C[1], srcCh[1], src.off[1], dst.off[1], 0, 0,
      0, 0, 0, 0,
    ]), src.buf, dst.buf], ...grid(Math.max(C[0], C[1]) * (d.hw / 4)), z: 2 });
  }

  /** Bilinear x2 upsample of C channels from resolution lowRes into channel offset dstCh of dst. */
  private upsample(b: PlanBuilder, src: Shared, C: number[], lowRes: number, dst: Shared, dstCh: number[]): void {
    const d = b.dims(lowRes);
    b.steps.push({ kernel: this.upsample2x, buffers: [b.uniform([
      d.w, d.h, C[0], dstCh[0], src.off[0], dst.off[0], 0, 0,
      d.w, d.h, C[1], dstCh[1], src.off[1], dst.off[1], 0, 0,
      0, 0, 0, 0,
    ]), src.buf, dst.buf], ...grid(Math.max(C[0], C[1]) * d.hw * 4), z: 2 });
  }

  get dispatchCount(): number { return this.plan?.steps.length ?? 0; }

  record(encoder: GPUCommandEncoder, features: GPUBuffer, width: number, height: number): GPUBuffer {
    this.prepare(width, height);
    const plan = this.plan!;
    const pass = encoder.beginComputePass({ label: "wgsl-head" });
    for (const s of plan.steps) {
      const buffers = s.buffers.map((b) => (b === "features" ? features : b));
      s.kernel.dispatch(pass, buffers, s.x, s.y, s.z);
    }
    pass.end();
    return plan.delta;
  }

  async run(features: GPUBuffer, width: number, height: number): Promise<GPUBuffer> {
    const enc = this.device.createCommandEncoder();
    const delta = this.record(enc, features, width, height);
    this.device.queue.submit([enc.finish()]);
    return delta;
  }

  release(): void { /* the delta buffer is owned by the plan */ }

  private disposePlan(): void {
    if (this.plan) for (const b of this.plan.buffers) b.destroy();
    this.plan = null;
    for (const u of this.uniforms) u.destroy();
    this.uniforms = [];
  }

  dispose(): void {
    this.disposePlan();
    for (const L of this.nets.layers) for (const b of [L.convW, L.convB]) b.destroy();
    for (const b of [...this.nets.outW, ...this.nets.outB]) b.destroy();
  }
}

export function parseHeadWeights(layoutBytes: Uint8Array, weightBytes: Uint8Array): { layout: HeadLayout; weights: Float32Array } {
  const layout = JSON.parse(new TextDecoder().decode(layoutBytes)) as HeadLayout;
  const copy = weightBytes.slice();
  const weights = new Float32Array(copy.buffer, 0, copy.byteLength >>> 2);
  const expected = layout.tensors.reduce((a, t) => Math.max(a, t.offset + t.count), 0);
  if (weights.length !== expected) throw new Error(`head weights: ${weights.length} floats, layout expects ${expected}`);
  return { layout, weights };
}
