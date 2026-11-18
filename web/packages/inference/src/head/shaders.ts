/**
 * WGSL generators for the custom correction head.
 *
 * Layouts: activations are planar f32 [C, H, W]. Conv weights are PyTorch order
 * [C_out, C_in, 3, 3] flattened, so element (co, ci, ky, kx) sits at
 * ((co * cin + ci) * 3 + ky) * 3 + kx. Zero padding of one pixel.
 *
 * The 3x3 convolution stages an input halo tile and a weight slab in workgroup
 * memory. Each thread owns PX by PY output pixels for CO output channels, so
 * every weight loaded from workgroup memory feeds PX*PY multiply-adds. The store
 * applies SiLU to the convolution with the calibrated affine folded in. All loops
 * over channels, pixels and taps are unrolled at generation time so nothing
 * lives in dynamically indexed private arrays.
 */

export interface ConvConfig {
  /** Input channels staged per chunk. */
  CI: number;
  /** Output channels per workgroup (dispatch z). */
  CO: number;
  /** Threads per workgroup in x and y. */
  TX: number;
  TY: number;
  /** Output pixels per thread in x and y. */
  PX: number;
  PY: number;
}

/** Default tiling: 64 threads, a 16x16 output tile, and 8 output channels. */
export const DEFAULT_CONV: ConvConfig = { CI: 4, CO: 8, TX: 16, TY: 4, PX: 1, PY: 4 };

export function convTile(cfg: ConvConfig): { tileW: number; tileH: number; threads: number } {
  return { tileW: cfg.TX * cfg.PX, tileH: cfg.TY * cfg.PY, threads: cfg.TX * cfg.TY };
}

/**
 * The head is two UNets with identical structure and different widths. Every kernel runs both
 * nets' matching layers in one dispatch: parameters come as a two-entry array and the dispatch
 * z coordinate selects the net (conv: z < zsplit is net 0; the others: z is the net), which
 * halves the dispatch count and lets the small net's work overlap the large net's.
 */
const PAIR_WGSL = `
fn net_of(z: u32, zsplit: u32) -> u32 { return select(0u, 1u, z >= zsplit); }
`;

export function convShader(cfg: ConvConfig): string {
  const { CI, CO, TX, TY, PX, PY } = cfg;
  const TILE_W = TX * PX, TILE_H = TY * PY, HALO_W = TILE_W + 2, HALO_H = TILE_H + 2, THREADS = TX * TY;
  const HALO = HALO_H * HALO_W;
  const S_IN = CI * HALO, S_W = CO * CI * 9;
  const acc: string[] = [];
  for (let co = 0; co < CO; co++) for (let j = 0; j < PY; j++) for (let i = 0; i < PX; i++) acc.push(`var a${co}_${j}_${i}: f32 = 0.0;`);
  const win: string[] = [];
  for (let j = 0; j < PY + 2; j++) for (let i = 0; i < PX + 2; i++) win.push(`let x${j}_${i} = s_in[sb + ${j * HALO_W + i}u];`);
  const mac: string[] = [];
  for (let co = 0; co < CO; co++) {
    const lines = [`{ let wb = ${co * CI * 9}u + c * 9u;`];
    for (let k = 0; k < 9; k++) lines.push(`let k${k} = s_w[wb + ${k}u];`);
    for (let j = 0; j < PY; j++) for (let i = 0; i < PX; i++) {
      const terms: string[] = [];
      for (let ky = 0; ky < 3; ky++) for (let kx = 0; kx < 3; kx++) terms.push(`k${ky * 3 + kx} * x${j + ky}_${i + kx}`);
      lines.push(`a${co}_${j}_${i} += ${terms.join(" + ")};`);
    }
    lines.push("}");
    mac.push(lines.join("\n      "));
  }
  const stores: string[] = [];
  for (let co = 0; co < CO; co++) {
    const lines = [`{ let gco = co0 + ${co}u; if (gco < q.cout) { let b = bias[q.b_off + gco];`];
    for (let j = 0; j < PY; j++) for (let i = 0; i < PX; i++) {
      lines.push(`{ let x = px + ${i}u; let y = py + ${j}u; if (x < q.W && y < q.H) { let v = a${co}_${j}_${i} + b; outp[q.out_off + gco * HW + y * q.W + x] = v / (1.0 + exp(-v)); } }`);
    }
    lines.push("} }");
    stores.push(lines.join("\n    "));
  }
  return `
${PAIR_WGSL}
struct NP { W: u32, H: u32, cin: u32, cout: u32, in_off: u32, w_off: u32, b_off: u32, out_off: u32 };
struct P { n: array<NP, 2>, zsplit: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> inp: array<f32>;
@group(0) @binding(2) var<storage, read> wgt: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> outp: array<f32>;
var<workgroup> s_in: array<f32, ${S_IN}>;
var<workgroup> s_w: array<f32, ${S_W}>;
@compute @workgroup_size(${TX}, ${TY}, 1)
fn conv3x3(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let net = net_of(wg.z, p.zsplit); let q = p.n[net]; let zl = wg.z - net * p.zsplit;
  let HW = q.W * q.H;
  let tx0 = wg.x * ${TILE_W}u; let ty0 = wg.y * ${TILE_H}u; let co0 = zl * ${CO}u;
  let px = tx0 + lid.x * ${PX}u; let py = ty0 + lid.y * ${PY}u;
  ${acc.join("\n  ")}
  for (var ci0 = 0u; ci0 < q.cin; ci0 += ${CI}u) {
    for (var e = li; e < ${S_IN}u; e += ${THREADS}u) {
      let c = e / ${HALO}u; let r = e - c * ${HALO}u; let hy = r / ${HALO_W}u; let hx = r - hy * ${HALO_W}u;
      let gx = i32(tx0 + hx) - 1; let gy = i32(ty0 + hy) - 1; let ci = ci0 + c;
      var v = 0.0;
      if (ci < q.cin && gx >= 0 && gx < i32(q.W) && gy >= 0 && gy < i32(q.H)) { v = inp[q.in_off + ci * HW + u32(gy) * q.W + u32(gx)]; }
      s_in[e] = v;
    }
    for (var e = li; e < ${S_W}u; e += ${THREADS}u) {
      let co = e / ${CI * 9}u; let r = e - co * ${CI * 9}u; let c = r / 9u; let k = r - c * 9u;
      let gco = co0 + co; let ci = ci0 + c;
      var v = 0.0;
      if (gco < q.cout && ci < q.cin) { v = wgt[q.w_off + (gco * q.cin + ci) * 9u + k]; }
      s_w[e] = v;
    }
    workgroupBarrier();
    for (var c = 0u; c < ${CI}u; c++) {
      let sb = c * ${HALO}u + (lid.y * ${PY}u) * ${HALO_W}u + lid.x * ${PX}u;
      ${win.join("\n      ")}
      ${mac.join("\n      ")}
    }
    workgroupBarrier();
  }
  ${stores.join("\n  ")}
}
`;
}

export const POOL_WGSL = `
struct NP { W: u32, H: u32, C: u32, src_ch: u32, src_off: u32, dst_off: u32, pad0: u32, pad1: u32 };
struct P { n: array<NP, 2>, pad0: u32, pad1: u32, pad2: u32, pad3: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(256)
fn pool2x2(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
  let q = p.n[gid.z];
  let w2 = q.W / 2u; let h2 = q.H / 2u;
  let e = gid.x + gid.y * nw.x * 256u;
  if (e >= q.C * w2 * h2) { return; }
  let c = e / (w2 * h2); let r = e - c * w2 * h2; let y = r / w2; let x = r - y * w2;
  let base = q.src_off + (q.src_ch + c) * q.W * q.H + (2u * y) * q.W + 2u * x;
  let v = max(max(src[base], src[base + 1u]), max(src[base + q.W], src[base + q.W + 1u]));
  dst[q.dst_off + e] = v;
}
`;

/** Bilinear x2 upsample (align_corners=false, PyTorch semantics) into a channel slice of a full-resolution destination. */
export const UPSAMPLE_WGSL = `
struct NP { w: u32, h: u32, C: u32, dst_ch: u32, src_off: u32, dst_off: u32, pad0: u32, pad1: u32 };
struct P { n: array<NP, 2>, pad0: u32, pad1: u32, pad2: u32, pad3: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(256)
fn upsample2x(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
  let q = p.n[gid.z];
  let W = 2u * q.w; let H = 2u * q.h;
  let e = gid.x + gid.y * nw.x * 256u;
  if (e >= q.C * W * H) { return; }
  let c = e / (W * H); let r = e - c * W * H; let y = r / W; let x = r - y * W;
  let sx = max((f32(x) + 0.5) * 0.5 - 0.5, 0.0);
  let sy = max((f32(y) + 0.5) * 0.5 - 0.5, 0.0);
  let x0 = u32(floor(sx)); let y0 = u32(floor(sy));
  let x1 = min(x0 + 1u, q.w - 1u); let y1 = min(y0 + 1u, q.h - 1u);
  let lx = sx - f32(x0); let ly = sy - f32(y0);
  let base = q.src_off + c * q.w * q.h;
  let top = (1.0 - lx) * src[base + y0 * q.w + x0] + lx * src[base + y0 * q.w + x1];
  let bot = (1.0 - lx) * src[base + y1 * q.w + x0] + lx * src[base + y1 * q.w + x1];
  dst[q.dst_off + (q.dst_ch + c) * W * H + y * W + x] = (1.0 - ly) * top + ly * bot;
}
`;

/** Final 1x1 convolutions of both nets summed: out = W_n a_n + b_n + W_a a_a + b_a. Both activations live in one buffer. */
export const OUT1X1_WGSL = `
struct P { hw: u32, cn: u32, ca: u32, a_off: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> act: array<f32>;
@group(0) @binding(2) var<storage, read> w_n: array<f32>;
@group(0) @binding(3) var<storage, read> b_n: array<f32>;
@group(0) @binding(4) var<storage, read> w_a: array<f32>;
@group(0) @binding(5) var<storage, read> b_a: array<f32>;
@group(0) @binding(6) var<storage, read_write> outp: array<f32>;
@compute @workgroup_size(256)
fn out1x1(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
  let px = gid.x + gid.y * nw.x * 256u;
  if (px >= p.hw) { return; }
  for (var o = 0u; o < 8u; o++) {
    var s = b_n[o] + b_a[o];
    for (var c = 0u; c < p.cn; c++) { s += w_n[o * p.cn + c] * act[c * p.hw + px]; }
    for (var c = 0u; c < p.ca; c++) { s += w_a[o * p.ca + c] * act[p.a_off + c * p.hw + px]; }
    outp[o * p.hw + px] = s;
  }
}
`;
