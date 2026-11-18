// Fused implicit-GEMM convolution for the GroupNorm-free Gaussifier networks.
//
// One kernel covers every conv in the frozen-GN forward map and correction head:
//
//   out[p, n] = act( bias[n] + residual[p, n] + sum_{tap, c} W[tap, c, n] * X(p + tap, c)
//                    [+ sum_c Wp[c, n] * Xp(p, c)] )
//
// where X is produced on the fly while staging the input tile into shared memory:
//
//   kPlain   X = in0                                         (same resolution as out)
//   kPool    X = maxpool2x2(in0)                             (in0 at 2H x 2W)
//   kUpCat   X = concat(bilinear_up2x(in0), in1)             (in0 at H/2 x W/2, in1 at H x W)
//   kUpCatRs X = concat(resize(bilinear_up2x(in0), H x W), in1)  (in0 at floor(H/2) x
//            floor(W/2), H or W odd: the reference U-Net's up2x + F.interpolate(size=skip))
//
// so neither the pooled nor the upsampled/concatenated tensor is ever written to global
// memory. Layout is NHWC FP16 (batch 1), accumulation FP32, weights [tap][Cin][Cout] FP16.
// The optional second K segment (ProjParams) is a 1x1 conv over a second input Xp with its
// own input mode; it lets a residual block's projection shortcut accumulate into conv2
// instead of round-tripping through memory.
//
// ---------------------------------------------------------------------------------------
// API
//
//   launch_conv2<InMode, KS, ProjMode, BK, TH, BN, WM, WN, Epi>(ConvParams, ProjParams,
//                                                            Epi::Params, stream)
//       explicit tile config, optional fused 1x1 projection segment (ProjMode =
//       InMode::kNoProj disables it) and a pluggable epilogue (DefaultEpilogue = bias +
//       residual + SiLU + NHWC FP16 store; see epilogues.cuh for the others).
//   launch_conv2<..., Epi, /*kF8=*/true>(...)
//       FP8 (e4m3) operands for the main segment: mma m16n8k32 e4m3, FP32 accumulate,
//       activations quantised while staging with the static per-tensor ConvParams::a_scale,
//       weights from ConvParams::weight8 ([taps][Cout][Cin] e4m3, Cin contiguous) with
//       per-output-channel ConvParams::w_scale; dequantised before an (FP16) projection
//       segment and before the epilogue, so Epi sees true-scale sums. BK 32 or 64.
//   InMode::kImageF32: in0 is a const float* H x W x 3 image read as 16 channels (stem).
//   BK may also be an odd multiple of 8 (8, 24, 40, 56; FP16 only): channel counts then only
//   need to be multiples of BK, e.g. native 24-channel layers with BK 24 (no zero padding to
//   32). Each tap row's KS * BK / 8 eight-channel units are paired into m16n8k16 MMAs across
//   tap boundaries (ldmatrix takes an independent address per 8-lane group), the odd unit
//   left over runs as m16n8k8. Weights keep the [taps][Cin][Cout] layout.
//   ConvParams::row0 / rows (optional): compute only output rows [row0, row0 + rows) (a
//   band). row0 must be a multiple of TH and rows a multiple of TH unless the band ends at H;
//   halo rows outside the band are read from the input as usual. rows = 0: to the bottom.
//   ConvParams::single_stage = 1 (optional): single-chunk launches (plain / pool input with
//   Cin == BK, no projection) request only one pipeline stage of shared memory
//   (max(stage, epilogue tile)), which can double occupancy (e.g. BK 24 / TH 16 / BN 24).
//   BN may be any multiple of 8 per warp (e.g. 48); non-power-of-two row widths are padded
//   instead of swizzled.
//   ConvParams::acc16 = 1 / 2 (optional, FP16 operand path): FP16-accumulate MMAs per K chunk /
//   over the whole K (see ConvParams). Selected per launch; loader default: default_acc16().
//   InMode::kPlainF8 + F8StoreEpilogue (epilogues.cuh): FP8 activation storage between two
//   convs (the consumer reads e4m3 written by the producer instead of quantising FP16).
//   conv_block.cuh: launch_block<...> runs a whole residual block in one kernel.
//
// Core: mma.sync m16n8k16 (FP32 accumulate) fed by ldmatrix from XOR-swizzled shared
// memory, two-stage pipeline: cp.async (zero-fill for padding) for plain inputs and
// weights, register prefetch for the computed pool / upsample inputs (global loads are
// issued before the current chunk's MMAs and combined + stored after them).
// Block tile: TH rows x 16 columns of output pixels by BN output channels; WM x WN warps,
// each owning TH/WM tile rows and BN/WN channels. K is consumed in chunks of BK channels
// (all taps of a chunk at once, so the halo tile is staged once per chunk).
// Requirements: every channel count of every segment a multiple of BK (for kUpCat this
// includes c0, so a chunk never straddles the concat boundary). H and W are arbitrary: the
// last row/column tiles may be partial (halo staging zero-fills, epilogues must mask
// y >= H and x >= W). kPool needs in0 at exactly 2H x 2W; kUpCat needs in0 at H/2 x W/2
// (H, W even), kUpCatRs at floor(H/2) x floor(W/2) (any H, W; same channel rules as kUpCat).

#pragma once

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace gfused {

// kImageF32: in0 is really a const float* H x W x 3 image (reinterpret_cast), read
// as a 16-channel input (3 real + 13 zero channels); c0 must be 16, BK 16, FP16 path.
// kPlainF8: in0 is really a const uint8_t* NHWC e4m3 tensor (reinterpret_cast) that already
// holds e4m3(X / a_scale) for this layer's a_scale (F8StoreEpilogue, epilogues.cuh); FP8 path
// only. It is staged with plain cp.async (no convert), and gives bit-identical results to
// kPlain on the FP16 tensor the producer would otherwise have stored.
// kUpCatRs: kUpCat whose upsampled half is the align_corners=False bilinear up2x of in0
// resized (align_corners=False bilinear) to H x W, the two interpolations composed in FP32
// without rounding the 2x intermediate. That is what the reference does at an odd decoder
// level; for even H and W it equals kUpCat up to rounding, so callers use it only for odd ones.
enum class InMode : int {
  kPlain = 0, kPool = 1, kUpCat = 2, kNoProj = 3, kImageF32 = 4, kPlainF8 = 5, kUpCatRs = 6,
  kUpCatPh = 7
};

// kUpCatPh: kUpCat computed by upsample folding (conv_phase.cuh): the upsampled half runs as
// four phase kernels on the low-resolution input (ConvParams::weight_ph / weight_edge), no
// interpolated halo. Only through launch_conv2 with KS 3, no projection, FP16 operands.

// Either upsample + concat mode (in0 low resolution, in1 skip).
constexpr bool is_upcat(InMode m) { return m == InMode::kUpCat || m == InMode::kUpCatRs; }

struct ConvParams {
  const __half* in0;       // NHWC; resolution depends on mode
  const __half* in1;       // kUpCat skip tensor, NHWC at (H, W); else nullptr
  int c0;                  // channels of in0 (multiple of 16)
  int c1;                  // channels of in1 (multiple of 16, or 0)
  int H, W;                // output (= conv input after prologue) resolution
  const __half* weight;    // [taps][Cin][Cout], Cin = c0 + c1
  const float* bias;       // [Cout] or nullptr
  const __half* residual;  // NHWC [H, W, Cout] or nullptr
  __half* out;             // NHWC [H, W, Cout]
  int cout;                // multiple of BN
  int silu;                // apply SiLU after bias + residual
  // FP8 operand path only (launch_conv2<..., kF8 = true>); ignored otherwise. The main
  // segment then computes sum_k e4m3(X / a_scale) * e4m3(W[n] / w_scale[n]) * a_scale *
  // w_scale[n] with FP32 accumulation; a projection segment stays FP16.
  const uint8_t* weight8 = nullptr;  // e4m3 [taps][Cout][Cin] (Cin contiguous)
  const float* w_scale = nullptr;    // [Cout] per-output-channel weight scales
  float a_scale = 1.0f;              // per-tensor (static) activation scale
  // Row band (launch_conv2 / launch_block): output rows [row0, row0 + rows); rows = 0 means
  // up to H. row0 % TH == 0, and rows % TH == 0 unless row0 + rows >= H.
  int row0 = 0;
  int rows = 0;
  // 1: a single-chunk launch (plain / pool input, Cin == BK, no projection) requests only one
  // pipeline stage of shared memory (max(stage, epilogue tile)) instead of two, which can
  // raise blocks/SM. Off by default (occupancy changes can go either way at DRAM-bound sizes).
  int single_stage = 0;
  // FP16-accumulate MMAs (FP16 operand path only; built with GFUSED_ACC16=1):
  //   0  FP32 accumulation (mma ...f32.f16.f16.f32), the default
  //   1  each K chunk (KS * KS * BK products) accumulates in FP16 (mma ...f16.f16.f16.f16),
  //      the chunk partials are added in FP32
  //   2  FP16 accumulation over the whole K (KS * KS * Cin products, projection segment
  //      included), converted to FP32 for the epilogue
  // Meant for GPUs whose FP16-accumulate MMA rate is higher (GeForce; default_acc16). Values
  // that overflow FP16 (|x| > 65504) become inf.
  int acc16 = 0;
  // 1: the fused bilinear upsample (kUpCat) interpolates in packed half2 arithmetic (lerp4_h2,
  // about one FP16 ulp from the FP32 blend, 5-12% faster upsample convs on sm_89); 0: FP32
  // blend rounded once (lerp4), the default. The correction head sets it.
  int lerp16 = 0;
  // Optional (conv_v2_kernel, H and W even): NHWC [H/2][W/2][cout] FP16 map whose bilinear 2x
  // upsample (align_corners=False, as kUpCat) is added to the FP32 sums before the epilogue.
  // A 1x1 conv commutes with the upsample, so an up block's projection of its upsampled input
  // runs at low resolution (Z = proj_up(low)) and conv2 adds up2x(Z) here.
  const __half* up_add = nullptr;
  // InMode::kUpCatPh only (conv_phase.cuh): phase kernels of the upsampled segment,
  // [4 phases (py * 2 + px)][9 taps (dy * 3 + dx)][c0][cout] FP16, and that segment's original
  // 3x3 weights as k pairs, [9 taps][c0 / 2][cout] half2 (border lines). See upfold.h.
  const __half* weight_ph = nullptr;
  const uint32_t* weight_edge = nullptr;
  int dbg = 0;  // ablation bits (conv_phase.cuh)
};

// Second K segment: 1x1 conv over Xp (same output resolution H x W, input mode ProjMode).
struct ProjParams {
  const __half* in0 = nullptr;
  const __half* in1 = nullptr;
  int c0 = 0, c1 = 0;
  const __half* weight = nullptr;  // [Cin_p][Cout]
};

namespace detail {

__device__ __forceinline__ float silu(float x) { return x / (1.0f + __expf(-x)); }

// Bilinear 2x upsample, align_corners=False: src = (dst + 0.5) / 2 - 0.5, clamped at 0.
__device__ __forceinline__ void up_coord(int dst, int src_size, int& i0, int& i1, float& w1) {
  float s = fmaxf((dst + 0.5f) * 0.5f - 0.5f, 0.0f);
  i0 = min(static_cast<int>(s), src_size - 1);
  i1 = min(i0 + 1, src_size - 1);
  w1 = s - static_cast<float>(i0);
}

// One axis of kUpCatRs: output index dst of `output` samples resized from the 2x upsample
// of `src` samples (both align_corners=False). The 2x intermediate's two adjacent samples
// involve at most three distinct source samples: first, first + 1, first + 2 (clamp the
// indices at src - 1; zero weights mark unused taps).
struct AxisWeights {
  int first;
  float w[3];
};

__device__ __forceinline__ AxisWeights up2x_resize_weights(int dst, int src, int output) {
  const float resized =
      fminf(2 * src - 1, fmaxf(0.f, (dst + 0.5f) * (2.f * src / output) - 0.5f));
  const int i = static_cast<int>(resized);
  const float blend = resized - i;
  const float left = fmaxf(0.f, (i + 0.5f) * 0.5f - 0.5f);
  const float right = fmaxf(0.f, (min(i + 1, 2 * src - 1) + 0.5f) * 0.5f - 0.5f);
  const int first = static_cast<int>(left), second = static_cast<int>(right);
  // Explicit rounding (each weight: a rounded product, then one fused multiply-add) so the
  // weights do not depend on how the compiler contracts them in a given kernel.
  AxisWeights result{first, {__fmul_rn(1.f - blend, 1.f - (left - first)),
                             __fmul_rn(1.f - blend, left - first), 0.f}};
  // second is first or first + 1 (no dynamic index: it would put the weights on the stack)
  const float near = 1.f - (right - second), far = right - second;
  if (second == first) {
    result.w[0] = __fmaf_rn(blend, near, result.w[0]);
    result.w[1] = __fmaf_rn(blend, far, result.w[1]);
  } else {
    result.w[1] = __fmaf_rn(blend, near, result.w[1]);
    result.w[2] = __fmaf_rn(blend, far, result.w[2]);
  }
  return result;
}

__device__ __forceinline__ uint4 max4(uint4 a, uint4 b, uint4 d, uint4 e) {
  __half2* ha = reinterpret_cast<__half2*>(&a);
  const __half2* hb = reinterpret_cast<const __half2*>(&b);
  const __half2* hd = reinterpret_cast<const __half2*>(&d);
  const __half2* he = reinterpret_cast<const __half2*>(&e);
#pragma unroll
  for (int i = 0; i < 4; ++i) ha[i] = __hmax2(__hmax2(ha[i], hb[i]), __hmax2(hd[i], he[i]));
  return a;
}

__device__ __forceinline__ uint4 lerp4(uint4 v00, uint4 v01, uint4 v10, uint4 v11, float wx,
                                       float wy) {
  const __half2* a = reinterpret_cast<const __half2*>(&v00);
  const __half2* b = reinterpret_cast<const __half2*>(&v01);
  const __half2* d = reinterpret_cast<const __half2*>(&v10);
  const __half2* e = reinterpret_cast<const __half2*>(&v11);
  uint4 r;
  __half2* hr = reinterpret_cast<__half2*>(&r);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    float2 fa = __half22float2(a[i]), fb = __half22float2(b[i]);
    float2 fd = __half22float2(d[i]), fe = __half22float2(e[i]);
    float2 top = make_float2(fa.x + (fb.x - fa.x) * wx, fa.y + (fb.y - fa.y) * wx);
    float2 bot = make_float2(fd.x + (fe.x - fd.x) * wx, fd.y + (fe.y - fd.y) * wx);
    hr[i] = __floats2half2_rn(top.x + (bot.x - top.x) * wy, top.y + (bot.y - top.y) * wy);
  }
  return r;
}

// lerp4 in packed half2 arithmetic (ConvParams::lerp16): top = a + (b - a) wx, bottom =
// d + (e - d) wx, top + (bottom - top) wy; 12 HSUB2 / HFMA2 per 8 channels instead of 32
// conversions and 24 FP32 operations. Within about one FP16 ulp of lerp4 (which rounds once).
__device__ __forceinline__ uint4 lerp4_h2(uint4 v00, uint4 v01, uint4 v10, uint4 v11,
                                          __half2 w) {
  const __half2* a = reinterpret_cast<const __half2*>(&v00);
  const __half2* b = reinterpret_cast<const __half2*>(&v01);
  const __half2* d = reinterpret_cast<const __half2*>(&v10);
  const __half2* e = reinterpret_cast<const __half2*>(&v11);
  const __half2 hx = __low2half2(w), hy = __high2half2(w);
  uint4 r;
  __half2* hr = reinterpret_cast<__half2*>(&r);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const __half2 top = __hfma2(__hsub2(b[i], a[i]), hx, a[i]);
    const __half2 bot = __hfma2(__hsub2(e[i], d[i]), hx, d[i]);
    hr[i] = __hfma2(__hsub2(bot, top), hy, top);
  }
  return r;
}

// ------------------------------------------------------------------ PTX helpers

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, bool valid) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src),
               "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void cp_async_wait_all() {
  asm volatile("cp.async.wait_group 0;\n" ::);
}

__device__ __forceinline__ void st_shared16(uint32_t addr, uint4 v) {
  asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};\n" ::"r"(addr), "r"(v.x), "r"(v.y),
               "r"(v.z), "r"(v.w));
}

__device__ __forceinline__ uint4 ld_shared16(uint32_t addr) {
  uint4 v;
  asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "r"(addr));
  return v;
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t addr, uint32_t (&r)[4]) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t addr, uint32_t (&r)[4]) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}
__device__ __forceinline__ void ldmatrix_x2(uint32_t addr, uint32_t (&r)[2]) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
               : "=r"(r[0]), "=r"(r[1])
               : "r"(addr));
}
__device__ __forceinline__ void mma16832_e4m3(float (&c)[4], const uint32_t (&a)[4], uint32_t b0,
                                              uint32_t b1) {
  asm("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, "
      "{%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void ldmatrix_x2_trans(uint32_t addr, uint32_t (&r)[2]) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
               : "=r"(r[0]), "=r"(r[1])
               : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x1_trans(uint32_t addr, uint32_t& r) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];\n"
               : "=r"(r)
               : "r"(addr));
}
__device__ __forceinline__ void mma1688(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t b0) {
  asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, "
      "{%0, %1, %2, %3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(b0));
}

// FP16-accumulate variants (ConvParams::acc16): c holds the same fragment elements as the
// FP32 forms' c[0..3], packed as two half2 (c[0] = {c0, c1}, c[1] = {c2, c3}).
__device__ __forceinline__ void mma1688(uint32_t (&c)[2], uint32_t a0, uint32_t a1, uint32_t b0) {
  asm("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3}, {%4}, {%0, %1};\n"
      : "+r"(c[0]), "+r"(c[1])
      : "r"(a0), "r"(a1), "r"(b0));
}
__device__ __forceinline__ void mma16816(uint32_t (&c)[2], const uint32_t (&a)[4], uint32_t b0,
                                         uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, "
      "{%0, %1};\n"
      : "+r"(c[0]), "+r"(c[1])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma16816(float (&c)[4], const uint32_t (&a)[4], uint32_t b0,
                                         uint32_t b1) {
  asm(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
      "{%8, %9}, {%0, %1, %2, %3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// XOR swizzle of 16-byte chunks so that ldmatrix reads of 8 consecutive rows (same logical
// chunk) hit 8 distinct bank groups. kChunks = chunks per row (power of two).
// Rows with a non-power-of-two chunk count (e.g. BN = 48 -> 6) are not swizzled but padded
// to an odd chunk stride, which also maps 8 consecutive rows to 8 distinct bank groups.
template <int kChunks>
__host__ __device__ constexpr int row_chunks() {
  return (kChunks & (kChunks - 1)) ? (kChunks | 1) : kChunks;
}

template <int kChunks>
__device__ __forceinline__ int swz(int row, int chunk) {
  if constexpr (kChunks & (kChunks - 1)) {
    return chunk;
  } else if constexpr (kChunks >= 8) {
    return chunk ^ (row & 7);
  } else if constexpr (kChunks == 1) {
    return chunk;
  } else {
    return chunk ^ ((row / (8 / kChunks)) & (kChunks - 1));
  }
}

// Byte offset of (row, chunk) in a swizzled [rows][kChunks x 16 B] array.
template <int kChunks>
__device__ __forceinline__ uint32_t swz_off(int row, int chunk) {
  return static_cast<uint32_t>((row * row_chunks<kChunks>() + swz<kChunks>(row, chunk)) * 16);
}

// ------------------------------------------------------------------ input stager
//
// Stages one BK-channel chunk of X over the (TH+2R) x (16+2R) halo pixels into shared memory
// (swizzled [pixel][BK] FP16, or [pixel][BK] e4m3 bytes in the FP8 path). Staging kinds:
//   kAsync  plain FP16 source (kPlain, or the skip half of kUpCat): cp.async with zero-fill;
//           also any chunk that was combined + stored immediately at issue time.
//   kReg    computed sources (kPool, kImageF32, and every plain source in the FP8 path,
//           which needs a convert): the raw vectors are loaded into registers by issue() and
//           combined / converted + stored by finish_reg() after the current chunk's MMAs.
//           Falls back to immediate when they would exceed the register budget.
//   kRaw    upsampled half of kUpCat: the low-resolution source patch ((TH/2+2) x 10 pixels,
//           origin (ty0/2 - 1, tx0/2 - 1), which covers every bilinear tap of the halo) is
//           cp.async'ed into a raw smem slot two chunks ahead, and transform() interpolates
//           it into the halo tile one chunk ahead (no extra barrier).

enum class StageKind : int { kAsync = 0, kReg = 1, kRaw = 2 };

struct SrcDesc {
  const __half* in0;
  const __half* in1;
  int c0, c1;
};

// 8 halves -> 8 e4m3 bytes of x * inv (saturating, round to nearest even).
__device__ __forceinline__ uint2 to_e4m3(uint4 v, float inv) {
  const __half2* h = reinterpret_cast<const __half2*>(&v);
  uint32_t q[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    float2 f = __half22float2(h[i]);
    f.x *= inv;
    f.y *= inv;
    q[i] = __nv_cvt_float2_to_fp8x2(f, __NV_SATFINITE, __NV_E4M3);
  }
  return make_uint2(q[0] | (q[1] << 16), q[2] | (q[3] << 16));
}

template <InMode kMode, int R, int TH, int BK, int NT, int kRawBudget, bool kF8>
struct Stager {
  static constexpr int HH = TH + 2 * R, HW = 16 + 2 * R;
  // staged units per pixel: 8 channels (16 bytes FP16 source), or 16 e4m3 channels (kPlainF8)
  static constexpr int kUnitCh = kMode == InMode::kPlainF8 ? 16 : 8;
  static constexpr int C8 = BK / kUnitCh;
  static constexpr int CA = kF8 ? BK / 16 : BK / 8;  // 16-byte smem chunks per pixel row
  static constexpr int UNITS = HH * HW * C8;
  static constexpr int UPT = (UNITS + NT - 1) / NT;
  // upsample source patch: origin (ty0/2 - PO, tx0/2 - PO) covers every bilinear tap of the
  // halo (R <= 1: (TH/2 + 2) x 10; R = 2: (TH/2 + 4) x 12)
  // kUpCatRs (R <= 1): the resized taps of a halo span one more source row / column, and
  // the patch origin is the first tap of the tile's first in-image halo row / column.
  static constexpr bool kUp = is_upcat(kMode), kRs = kMode == InMode::kUpCatRs;
  static_assert(!kRs || R <= 1, "kUpCatRs: R <= 1");
  static constexpr int PO = 1 + R / 2;
  static constexpr int PH = TH / 2 + 2 * PO + kRs, PW = 8 + 2 * PO + kRs;
  static constexpr int PUNITS = PH * PW * C8;
  static constexpr int kRawBytes = kUp ? PH * PW * BK * 2 : 0;
  // kUpCatRs axis table: HH row then HW column entries of 16 bytes (tap patch indices, weights)
  static constexpr int kTabBytes = kRs ? (HH + HW) * 16 : 0;
  static constexpr int kPerUnit = kMode == InMode::kPool ? 4 : 1;
  static constexpr bool kRegCapable =
      kMode == InMode::kPool || kMode == InMode::kImageF32 || (kF8 && kMode != InMode::kPlainF8);
  static constexpr bool kRegs = kRegCapable && UPT * kPerUnit <= kRawBudget;
  static_assert(TH % 2 == 0, "TH must be even");
  static_assert(!kF8 || BK % 32 == 0, "FP8 path needs BK multiple of 32");
  static_assert(kMode != InMode::kImageF32 || (BK == 16 && !kF8), "kImageF32: BK 16, FP16");
  static_assert(kMode != InMode::kPlainF8 || kF8, "kPlainF8: FP8 path only");

  uint4 raw[kRegs ? UPT : 1][kRegs ? kPerUnit : 1];
  float a_inv = 1.0f;  // FP8: 1 / activation scale
  uint32_t s_tab = 0;  // kUpCatRs: shared-memory axis table (kTabBytes), set before prepare()
  bool lerp16 = false;  // ConvParams::lerp16

  __device__ __forceinline__ static bool is_raw(const SrcDesc& s, int k0) {
    return kUp && k0 < s.c0;
  }

  __device__ __forceinline__ void store(uint32_t s_a, int pix, int c8, uint4 v) const {
    if constexpr (kF8) {
      const uint2 q = to_e4m3(v, a_inv);
      asm volatile("st.shared.v2.u32 [%0], {%1, %2};\n" ::"r"(s_a + swz_off<CA>(pix, c8 >> 1) +
                                                             (c8 & 1) * 8),
                   "r"(q.x), "r"(q.y));
    } else {
      st_shared16(s_a + swz_off<CA>(pix, c8), v);
    }
  }

  // Raw vectors of one unit (zero outside the image). For kUpCat this is the skip half.
  __device__ __forceinline__ void load_unit(const SrcDesc& s, int H, int W, int y, int x, int k0,
                                            int c8, uint4 (&r)[kPerUnit]) const {
#pragma unroll
    for (int j = 0; j < kPerUnit; ++j) r[j] = make_uint4(0, 0, 0, 0);
    if (y < 0 || y >= H || x < 0 || x >= W) return;
    if constexpr (kMode == InMode::kPool) {
      const int W2 = 2 * W;
      const __half* b = s.in0 + (static_cast<size_t>(2 * y) * W2 + 2 * x) * s.c0 + k0 + c8 * 8;
      const size_t row = static_cast<size_t>(W2) * s.c0;
      r[0] = __ldg(reinterpret_cast<const uint4*>(b));
      r[kPerUnit - 3] = __ldg(reinterpret_cast<const uint4*>(b + s.c0));
      r[kPerUnit - 2] = __ldg(reinterpret_cast<const uint4*>(b + row));
      r[kPerUnit - 1] = __ldg(reinterpret_cast<const uint4*>(b + row + s.c0));
    } else if constexpr (kMode == InMode::kImageF32) {
      if (c8 == 0) {
        const float* f = reinterpret_cast<const float*>(s.in0) + (static_cast<size_t>(y) * W + x) * 3;
        r[0] = make_uint4(__float_as_uint(__ldg(f)), __float_as_uint(__ldg(f + 1)),
                          __float_as_uint(__ldg(f + 2)), 0u);
      }
    } else {
      const bool skip = kUp;
      const __half* src = skip ? s.in1 : s.in0;
      const int c = skip ? s.c1 : s.c0;
      const int kk = skip ? k0 - s.c0 : k0;
      r[0] = __ldg(reinterpret_cast<const uint4*>(src + (static_cast<size_t>(y) * W + x) * c +
                                                   kk + c8 * 8));
    }
  }

  __device__ __forceinline__ static uint4 combine(const uint4 (&r)[kPerUnit]) {
    if constexpr (kMode == InMode::kPool) {
      return max4(r[0], r[kPerUnit - 3], r[kPerUnit - 2], r[kPerUnit - 1]);
    } else if constexpr (kMode == InMode::kImageF32) {
      uint4 v;
      __half2* h = reinterpret_cast<__half2*>(&v);
      h[0] = __floats2half2_rn(__uint_as_float(r[0].x), __uint_as_float(r[0].y));
      h[1] = __floats2half2_rn(__uint_as_float(r[0].z), 0.0f);
      h[2] = h[3] = __floats2half2_rn(0.0f, 0.0f);
      return v;
    } else {
      return r[0];
    }
  }

  // Per-thread staging descriptors, computed once per tile (prepare()) so the per-chunk
  // cp.async issue is just an add: element offset of the unit's 8 channels (-1: zero-fill
  // padding, -2: no unit) and its swizzled smem offset.
  static constexpr bool kAsyncPlain =
      ((kMode == InMode::kPlain || kUp) && !kF8) || kMode == InMode::kPlainF8;
  static constexpr int PUPT = (PUNITS + NT - 1) / NT;
  int pg[kAsyncPlain ? UPT : 1];
  uint32_t ps[kAsyncPlain ? UPT : 1];
  int rg[kUp ? PUPT : 1];
  // kUpCat transform descriptors per halo unit: patch pixel indices of the 4 bilinear taps
  // packed in 16-bit pairs (ti0 = i00 | i01 << 16, ti1 = i10 | i11 << 16; ti0 = ~0u: zero
  // padding, ~1u: no unit) and the weights (wx, wy) as a half2 (exact: 0, 0.25 or 0.75).
  static constexpr int kTU = (kMode == InMode::kUpCat && kAsyncPlain) ? UPT : 1;
  uint32_t ti0[kTU], ti1[kTU];
  __half2 tw[kTU];

  __device__ __forceinline__ void prepare(const SrcDesc& s, int H, int W, int ty0, int tx0) {
    const int tid = threadIdx.x;
    if constexpr (kAsyncPlain) {
      const int c = kUp ? s.c1 : s.c0;
#pragma unroll
      for (int i = 0; i < UPT; ++i) {
        const int u = tid + i * NT;
        const int pix = u / C8, c8 = u % C8;
        const int y = ty0 + pix / HW - R, x = tx0 + pix % HW - R;
        const bool ok = y >= 0 && y < H && x >= 0 && x < W;
        pg[i] = u >= UNITS ? -2 : ok ? (y * W + x) * c + c8 * kUnitCh : -1;
        ps[i] = swz_off<CA>(pix, c8);
      }
    }
    if constexpr (kMode == InMode::kUpCat && kAsyncPlain) {
      const int sy0 = ty0 / 2 - PO, sx0 = tx0 / 2 - PO;
#pragma unroll
      for (int i = 0; i < UPT; ++i) {
        const int u = tid + i * NT;
        const int pix = u / C8;
        const int y = ty0 + pix / HW - R, x = tx0 + pix % HW - R;
        ti1[i] = 0;
        tw[i] = __floats2half2_rn(0.0f, 0.0f);
        if (u >= UNITS) {
          ti0[i] = ~1u;
        } else if (y < 0 || y >= H || x < 0 || x >= W) {
          ti0[i] = ~0u;
        } else {
          int y0, y1, x0, x1;
          float wy, wx;
          up_coord(y, H / 2, y0, y1, wy);
          up_coord(x, W / 2, x0, x1, wx);
          ti0[i] = ((y0 - sy0) * PW + x0 - sx0) | (((y0 - sy0) * PW + x1 - sx0) << 16);
          ti1[i] = ((y1 - sy0) * PW + x0 - sx0) | (((y1 - sy0) * PW + x1 - sx0) << 16);
          tw[i] = __floats2half2_rn(wx, wy);
        }
      }
    }
    if constexpr (kUp) {
      const int Hs = H / 2, Ws = W / 2;
      int sy0 = ty0 / 2 - PO, sx0 = tx0 / 2 - PO;
      if constexpr (kRs) {
        sy0 = up2x_resize_weights(max(ty0 - R, 0), Hs, H).first;
        sx0 = up2x_resize_weights(max(tx0 - R, 0), Ws, W).first;
        // axis table: entry t < HH is halo row t, else halo column t - HH. Tap indices are
        // patch rows / columns (clamped at the source edge like the reference); 0xff marks a
        // halo row / column outside the image (zero padding).
        for (int t = tid; t < HH + HW; t += NT) {
          const bool row = t < HH;
          const int d = row ? ty0 + t - R : tx0 + t - HH - R;
          const int n = row ? H : W, src = row ? Hs : Ws, o = row ? sy0 : sx0;
          uint4 e = make_uint4(0xffu, 0u, 0u, 0u);
          if (d >= 0 && d < n) {
            const AxisWeights a = up2x_resize_weights(d, src, n);
            e.x = 0;
#pragma unroll
            for (int k = 0; k < 3; ++k) {  // unused (zero-weight) taps read tap 0
              const int idx = a.w[k] != 0.f ? min(a.first + k, src - 1) - o : a.first - o;
              e.x |= static_cast<uint32_t>(idx) << (8 * k);
            }
            e.y = __float_as_uint(a.w[0]);
            e.z = __float_as_uint(a.w[1]);
            e.w = __float_as_uint(a.w[2]);
          }
          st_shared16(s_tab + t * 16, e);
        }
      }
#pragma unroll
      for (int j = 0; j < PUPT; ++j) {
        const int u = tid + j * NT;
        const int pp = u / C8, c8 = u % C8;
        const int sy = sy0 + pp / PW, sx = sx0 + pp % PW;
        const bool ok = sy >= 0 && sy < Hs && sx >= 0 && sx < Ws;
        rg[j] = u >= PUNITS ? -2 : ok ? (sy * Ws + sx) * s.c0 + c8 * 8 : -1;
      }
    }
  }

  // Non-raw chunk. Returns kAsync (cp.async issued or stored immediately) or kReg.
  __device__ __forceinline__ StageKind issue(const SrcDesc& s, int H, int W, int ty0, int tx0,
                                             int k0, uint32_t s_a) {
    const int tid = threadIdx.x;
    if constexpr (kMode == InMode::kPlainF8) {  // e4m3 bytes, 16 channels per unit
      const uint8_t* src = reinterpret_cast<const uint8_t*>(s.in0) + k0;
#pragma unroll
      for (int i = 0; i < UPT; ++i) {
        if (UNITS % NT == 0 || pg[i] != -2)
          cp_async16(s_a + ps[i], src + max(pg[i], 0), pg[i] >= 0);
      }
      return StageKind::kAsync;
    } else if constexpr (kAsyncPlain) {
      const bool skip = kUp;
      const __half* src = (skip ? s.in1 : s.in0) + (skip ? k0 - s.c0 : k0);
#pragma unroll
      for (int i = 0; i < UPT; ++i) {
        if (UNITS % NT == 0 || pg[i] != -2) cp_async16(s_a + ps[i], src + max(pg[i], 0), pg[i] >= 0);
      }
      return StageKind::kAsync;
    } else if constexpr (kRegs) {
#pragma unroll
      for (int i = 0; i < UPT; ++i) {
        const int u = tid + i * NT;
        const int pix = u / C8, c8 = u % C8;
        const int y = ty0 + pix / HW - R, x = tx0 + pix % HW - R;
        load_unit(s, H, W, u < UNITS ? y : -1, x, k0, c8, raw[i]);
      }
      return StageKind::kReg;
    } else {  // immediate
      for (int u = tid; u < UNITS; u += NT) {
        const int pix = u / C8, c8 = u % C8;
        uint4 r[kPerUnit];
        load_unit(s, H, W, ty0 + pix / HW - R, tx0 + pix % HW - R, k0, c8, r);
        store(s_a, pix, c8, combine(r));
      }
      return StageKind::kAsync;
    }
  }

  __device__ __forceinline__ void finish_reg(uint32_t s_a) {
    if constexpr (kRegs) {
      const int tid = threadIdx.x;
#pragma unroll
      for (int i = 0; i < UPT; ++i) {
        const int u = tid + i * NT;
        if (UNITS % NT == 0 || u < UNITS) store(s_a, u / C8, u % C8, combine(raw[i]));
      }
    }
  }

  // kUpCat upsampled half: low-resolution source patch -> raw slot (cp.async, FP16).
  __device__ __forceinline__ void issue_raw(const SrcDesc& s, int, int, int, int, int k0,
                                            uint32_t s_raw) const {
    if constexpr (kUp) {
      const __half* src = s.in0 + k0;
#pragma unroll
      for (int j = 0; j < PUPT; ++j) {
        if (PUNITS % NT == 0 || rg[j] != -2)
          cp_async16(s_raw + (threadIdx.x + j * NT) * 16, src + max(rg[j], 0), rg[j] >= 0);
      }
    }
  }

  __device__ __forceinline__ void transform(int H, int W, int ty0, int tx0, uint32_t s_a,
                                            uint32_t s_raw) const {
    if constexpr (kRs) {  // composed weights from the axis table, summed like the reference
      for (int u = threadIdx.x; u < UNITS; u += NT) {
        const int pix = u / C8, c8 = u % C8;
        const uint4 ey = ld_shared16(s_tab + (pix / HW) * 16);
        const uint4 ex = ld_shared16(s_tab + (HH + pix % HW) * 16);
        uint4 v = make_uint4(0, 0, 0, 0);
        if (ey.x != 0xffu && ex.x != 0xffu) {
          const float wy[3] = {__uint_as_float(ey.y), __uint_as_float(ey.z),
                               __uint_as_float(ey.w)};
          const float wx[3] = {__uint_as_float(ex.y), __uint_as_float(ex.z),
                               __uint_as_float(ex.w)};
          float acc[8] = {};
#pragma unroll
          for (int dy = 0; dy < 3; ++dy) {
            const uint32_t row = s_raw + (((ey.x >> (8 * dy)) & 0xffu) * PW * C8 + c8) * 16;
#pragma unroll
            for (int dx = 0; dx < 3; ++dx) {
              const uint4 q = ld_shared16(row + ((ex.x >> (8 * dx)) & 0xffu) * C8 * 16);
              const __half2* h = reinterpret_cast<const __half2*>(&q);
              const float weight = wy[dy] * wx[dx];
#pragma unroll
              for (int k = 0; k < 4; ++k) {
                const float2 f = __half22float2(h[k]);
                acc[2 * k] += f.x * weight;
                acc[2 * k + 1] += f.y * weight;
              }
            }
          }
          __half2* vh = reinterpret_cast<__half2*>(&v);
#pragma unroll
          for (int k = 0; k < 4; ++k) vh[k] = __floats2half2_rn(acc[2 * k], acc[2 * k + 1]);
        }
        store(s_a, pix, c8, v);
      }
    } else if constexpr (kMode == InMode::kUpCat && kAsyncPlain) {  // precomputed descriptors
#pragma unroll
      for (int i = 0; i < UPT; ++i) {
        if (UNITS % NT != 0 && ti0[i] == ~1u) continue;
        const int c8 = (threadIdx.x + i * NT) % C8;
        uint4 v = make_uint4(0, 0, 0, 0);
        if (ti0[i] != ~0u) {
          const uint32_t b = s_raw + c8 * 16;
          const uint4 q00 = ld_shared16(b + (ti0[i] & 0xffff) * (C8 * 16));
          const uint4 q01 = ld_shared16(b + (ti0[i] >> 16) * (C8 * 16));
          const uint4 q10 = ld_shared16(b + (ti1[i] & 0xffff) * (C8 * 16));
          const uint4 q11 = ld_shared16(b + (ti1[i] >> 16) * (C8 * 16));
          if (lerp16) {
            v = lerp4_h2(q00, q01, q10, q11, tw[i]);
          } else {
            const float2 w = __half22float2(tw[i]);
            v = lerp4(q00, q01, q10, q11, w.x, w.y);
          }
        }
        st_shared16(s_a + ps[i], v);
      }
    } else if constexpr (kMode == InMode::kUpCat) {
      const int Hs = H / 2, Ws = W / 2;
      const int sy0 = ty0 / 2 - PO, sx0 = tx0 / 2 - PO;
      for (int u = threadIdx.x; u < UNITS; u += NT) {
        const int pix = u / C8, c8 = u % C8;
        const int y = ty0 + pix / HW - R, x = tx0 + pix % HW - R;
        uint4 v = make_uint4(0, 0, 0, 0);
        if (y >= 0 && y < H && x >= 0 && x < W) {
          int y0, y1, x0, x1;
          float wy, wx;
          up_coord(y, Hs, y0, y1, wy);
          up_coord(x, Ws, x0, x1, wx);
          const uint32_t r0 = ((y0 - sy0) * PW) * C8 + c8, r1 = ((y1 - sy0) * PW) * C8 + c8;
          const uint32_t q0 = (x0 - sx0) * C8, q1 = (x1 - sx0) * C8;
          v = lerp4(ld_shared16(s_raw + (r0 + q0) * 16), ld_shared16(s_raw + (r0 + q1) * 16),
                    ld_shared16(s_raw + (r1 + q0) * 16), ld_shared16(s_raw + (r1 + q1) * 16),
                    wx, wy);
        }
        store(s_a, pix, c8, v);
      }
    }
  }
};

// Weights [taps][cin][cout] rows (tap, k0..k0+BK) x cols (n0..n0+BN) -> swizzled smem.
template <int TAPS, int BK, int BN, int NT>
__device__ __forceinline__ void stage_weights(const __half* w, int cin, int cout, int k0, int n0,
                                              uint32_t s_b) {
  constexpr int C8 = BN / 8;
  constexpr int UNITS = TAPS * BK * C8;
#pragma unroll 4
  for (int u = threadIdx.x; u < UNITS; u += NT) {
    const int row = u / C8, c8 = u % C8;  // row = tap * BK + k
    const int tap = row / BK, k = row % BK;
    const __half* g = w + (static_cast<size_t>(tap) * cin + k0 + k) * cout + n0 + c8 * 8;
    cp_async16(s_b + swz_off<C8>(row, c8), g, true);
  }
}

// Same, with per-thread descriptors precomputed once per block (element offset at k0 = 0
// and swizzled smem offset per unit); per chunk only k0 * cout is added.
template <int TAPS, int BK, int BN, int NT, bool kF8>
struct WeightStager {
  static constexpr int C8 = kF8 ? BK / 16 : BN / 8;  // 16-byte units per smem row
  static constexpr int ROWS = kF8 ? TAPS * BN : TAPS * BK;
  static constexpr int UNITS = ROWS * C8;
  static constexpr int UPT = (UNITS + NT - 1) / NT;
  int g[UPT];
  uint32_t so[UPT];
  __device__ __forceinline__ void prepare(int cin, int cout, int n0) {
#pragma unroll
    for (int i = 0; i < UPT; ++i) {
      const int u = threadIdx.x + i * NT;
      const int row = u / C8, c = u % C8;
      if constexpr (kF8) {  // row = tap * BN + n, bytes k0 + c * 16
        const int tap = row / BN, n = row % BN;
        g[i] = (tap * cout + n0 + n) * cin + c * 16;
      } else {  // row = tap * BK + k
        const int tap = row / BK, k = row % BK;
        g[i] = (tap * cin + k) * cout + n0 + c * 8;
      }
      so[i] = swz_off<C8>(row, c);
    }
  }
  // base: weight pointer (bytes for FP8); step: k0 * cout (FP16) or k0 (FP8) elements
  template <class T>
  __device__ __forceinline__ void issue(const T* base, int step, uint32_t s_b) const {
#pragma unroll
    for (int i = 0; i < UPT; ++i) {
      if (UNITS % NT == 0 || threadIdx.x + i * NT < UNITS)
        cp_async16(s_b + so[i], base + g[i] + step, true);
    }
  }
};

// B fragments (k16 x n8 tiles, this warp's NT8 of them) for weight row block `row0`.
template <int NT8, int CB>
__device__ __forceinline__ void load_b(uint32_t s_b, int row0, int wn, uint32_t (&b)[NT8][2]) {
  const int lane = threadIdx.x & 31;
  const int row = row0 + (lane & 7) + ((lane >> 3) & 1) * 8, nc = lane >> 4;
#pragma unroll
  for (int j = 0; j + 1 < NT8; j += 2) {
    uint32_t r[4];
    ldmatrix_x4_trans(s_b + swz_off<CB>(row, wn * NT8 + j + nc), r);
    b[j][0] = r[0];
    b[j][1] = r[1];
    b[j + 1][0] = r[2];
    b[j + 1][1] = r[3];
  }
  if constexpr (NT8 % 2) {
    uint32_t r[2];
    ldmatrix_x2_trans(s_b + swz_off<CB>(row, wn * NT8 + NT8 - 1), r);
    b[NT8 - 1][0] = r[0];
    b[NT8 - 1][1] = r[1];
  }
}

// One chunk of MMAs: A = halo tile [(TH+2R) x (16+2R)][BK], B = [KS*KS][BK][BN].
// For 3x3, each A fragment (halo row hr, column shift kx) is loaded once and reused for the
// up-to-three output rows hr - ky it contributes to.
// Acc: float [TH / WM][BN / WN / 8][4] (FP32 accumulate) or uint32_t [..][..][2] (FP16).
template <int KS, int TH, int BK, int BN, int WM, int WN, class Acc>
__device__ __forceinline__ void mma_chunk(uint32_t s_a, uint32_t s_b, Acc& acc) {
  constexpr int R = KS / 2, HW = 16 + 2 * R;
  constexpr int WR = TH / WM, NT8 = BN / WN / 8;
  constexpr int CA = BK / 8, CB = BN / 8;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int wm = warp % WM, wn = warp / WM;
  // ldmatrix lane -> (row within 16, k8 half) for A
  const int a_row = (lane & 7) + ((lane >> 3) & 1) * 8, a_kc = lane >> 4;
#pragma unroll
  for (int s = 0; s < BK / 16; ++s) {
#pragma unroll
    for (int kx = 0; kx < KS; ++kx) {
      uint32_t b[KS][NT8][2];
#pragma unroll
      for (int ky = 0; ky < KS; ++ky) load_b<NT8, CB>(s_b, (ky * KS + kx) * BK + s * 16, wn, b[ky]);
#pragma unroll
      for (int hr = 0; hr < WR + 2 * R; ++hr) {
        const int pix = (wm * WR + hr) * HW + a_row + kx;
        uint32_t a[4];
        ldmatrix_x4(s_a + swz_off<CA>(pix, s * 2 + a_kc), a);
#pragma unroll
        for (int ky = 0; ky < KS; ++ky) {
          const int r = hr - ky;
          if (r >= 0 && r < WR) {
#pragma unroll
            for (int j = 0; j < NT8; ++j) mma16816(acc[r][j], a, b[ky][j][0], b[ky][j][1]);
          }
        }
      }
    }
  }
}

// k8 x n8 B fragments (this warp's NT8 of them) for weight rows row0 .. row0 + 7.
template <int NT8, int CB>
__device__ __forceinline__ void load_b_k8(uint32_t s_b, int row0, int wn, uint32_t (&b)[NT8]) {
  const int lane = threadIdx.x & 31;
  const int row = row0 + (lane & 7), m = lane >> 3;
  int j = 0;
#pragma unroll
  for (; j + 3 < NT8; j += 4) {
    uint32_t r[4];
    ldmatrix_x4_trans(s_b + swz_off<CB>(row, wn * NT8 + j + m), r);
    b[j] = r[0];
    b[j + 1] = r[1];
    b[j + 2] = r[2];
    b[j + 3] = r[3];
  }
  if constexpr (NT8 % 4 >= 2) {
    constexpr int j2 = NT8 / 4 * 4;
    uint32_t r[2];
    ldmatrix_x2_trans(s_b + swz_off<CB>(row, wn * NT8 + j2 + (m & 1)), r);
    b[j2] = r[0];
    b[j2 + 1] = r[1];
  }
  if constexpr (NT8 % 2) ldmatrix_x1_trans(s_b + swz_off<CB>(row, wn * NT8 + NT8 - 1), b[NT8 - 1]);
}

// One chunk of MMAs for BK an odd multiple of 8 (FP16). The A halo rows hold CA = BK / 8
// sixteen-byte units per pixel, unswizzled (CA odd): unit q = kx * CA + c8 of halo pixel
// (row, col) lives at ((row * HW + col) * CA + c8) * 16 = (row * HW * CA + col * CA + q) * 16
// relative to column kx, so the U = KS * CA units of one tap row are contiguous per output
// pixel. Weight rows (tap * BK + k) are likewise q * 8 + k within a tap row. Units are
// consumed in pairs (one m16n8k16: lanes 0-15 address unit 2i, lanes 16-31 unit 2i + 1, which
// may sit at the next kx), and the trailing unit of an odd U as one m16n8k8. Each A fragment
// is reused for the up to KS output rows it contributes to, as in mma_chunk.
template <int KS, int TH, int BK, int BN, int WM, int WN, class Acc>
__device__ __forceinline__ void mma_chunk_k8(uint32_t s_a, uint32_t s_b, Acc& acc) {
  constexpr int R = KS / 2, HW = 16 + 2 * R;
  constexpr int WR = TH / WM, NT8 = BN / WN / 8;
  constexpr int CA = BK / 8, CB = BN / 8;
  constexpr int U = KS * CA;  // 8-channel units per tap row
  static_assert(CA % 2 == 1, "mma_chunk_k8: BK must be an odd multiple of 8");
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int wm = warp % WM, wn = warp / WM;
  const int a_row = (lane & 7) + ((lane >> 3) & 1) * 8, a_kc = lane >> 4;
  // byte address of unit 0 of this lane's pixel in halo row 0 of the warp
  const uint32_t a0 = s_a + ((wm * WR) * HW + a_row) * CA * 16;
#pragma unroll
  for (int i = 0; i < U / 2; ++i) {
    uint32_t b[KS][NT8][2];
#pragma unroll
    for (int ky = 0; ky < KS; ++ky) load_b<NT8, CB>(s_b, ky * KS * BK + i * 16, wn, b[ky]);
#pragma unroll
    for (int hr = 0; hr < WR + 2 * R; ++hr) {
      uint32_t a[4];
      ldmatrix_x4(a0 + (hr * HW * CA + 2 * i + a_kc) * 16, a);
#pragma unroll
      for (int ky = 0; ky < KS; ++ky) {
        const int r = hr - ky;
        if (r >= 0 && r < WR) {
#pragma unroll
          for (int j = 0; j < NT8; ++j) mma16816(acc[r][j], a, b[ky][j][0], b[ky][j][1]);
        }
      }
    }
  }
  if constexpr (U % 2) {
    uint32_t b[KS][NT8];
#pragma unroll
    for (int ky = 0; ky < KS; ++ky) load_b_k8<NT8, CB>(s_b, ky * KS * BK + (U - 1) * 8, wn, b[ky]);
#pragma unroll
    for (int hr = 0; hr < WR + 2 * R; ++hr) {
      uint32_t a[2];
      ldmatrix_x2(a0 + (hr * HW * CA + U - 1) * 16, a);
#pragma unroll
      for (int ky = 0; ky < KS; ++ky) {
        const int r = hr - ky;
        if (r >= 0 && r < WR) {
#pragma unroll
          for (int j = 0; j < NT8; ++j) mma1688(acc[r][j], a[0], a[1], b[ky][j]);
        }
      }
    }
  }
}

// FP16 chunk dispatch on the K granularity.
template <int KS, int TH, int BK, int BN, int WM, int WN, class Acc>
__device__ __forceinline__ void mma_chunk_h(uint32_t s_a, uint32_t s_b, Acc& acc) {
  if constexpr (BK % 16) {
    mma_chunk_k8<KS, TH, BK, BN, WM, WN>(s_a, s_b, acc);
  } else {
    mma_chunk<KS, TH, BK, BN, WM, WN>(s_a, s_b, acc);
  }
}

// FP8 B fragments (k32 x n8 e4m3) from [rows = tap * BN + n][BK bytes] smem, k chunk pair kc0.
template <int NT8, int CB>
__device__ __forceinline__ void load_b8(uint32_t s_b, int row0, int kc0, int wn,
                                        uint32_t (&b)[NT8][2]) {
  const int lane = threadIdx.x & 31, m = lane >> 3;
  const int row = row0 + wn * NT8 * 8 + (lane & 7) + (m >> 1) * 8, kc = kc0 + (m & 1);
#pragma unroll
  for (int j = 0; j + 1 < NT8; j += 2) {
    uint32_t r[4];
    ldmatrix_x4(s_b + swz_off<CB>(row + j * 8, kc), r);
    b[j][0] = r[0];
    b[j][1] = r[1];
    b[j + 1][0] = r[2];
    b[j + 1][1] = r[3];
  }
  if constexpr (NT8 % 2) {
    uint32_t r[2];
    ldmatrix_x2(s_b + swz_off<CB>(row0 + (wn * NT8 + NT8 - 1) * 8 + (lane & 7), kc), r);
    b[NT8 - 1][0] = r[0];
    b[NT8 - 1][1] = r[1];
  }
}

// FP8 chunk: A = halo [(TH+2R) x (16+2R)][BK bytes], B = [KS*KS][BN][BK bytes].
template <int KS, int TH, int BK, int BN, int WM, int WN>
__device__ __forceinline__ void mma_chunk_f8(uint32_t s_a, uint32_t s_b,
                                             float (&acc)[TH / WM][BN / WN / 8][4]) {
  constexpr int R = KS / 2, HW = 16 + 2 * R;
  constexpr int WR = TH / WM, NT8 = BN / WN / 8;
  constexpr int CA = BK / 16;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int wm = warp % WM, wn = warp / WM;
  const int a_row = (lane & 7) + ((lane >> 3) & 1) * 8, a_kc = lane >> 4;
#pragma unroll
  for (int s = 0; s < BK / 32; ++s) {
#pragma unroll
    for (int kx = 0; kx < KS; ++kx) {
      uint32_t b[KS][NT8][2];
#pragma unroll
      for (int ky = 0; ky < KS; ++ky) load_b8<NT8, CA>(s_b, (ky * KS + kx) * BN, s * 2, wn, b[ky]);
#pragma unroll
      for (int hr = 0; hr < WR + 2 * R; ++hr) {
        const int pix = (wm * WR + hr) * HW + a_row + kx;
        uint32_t a[4];
        ldmatrix_x4(s_a + swz_off<CA>(pix, s * 2 + a_kc), a);
#pragma unroll
        for (int ky = 0; ky < KS; ++ky) {
          const int r = hr - ky;
          if (r >= 0 && r < WR) {
#pragma unroll
            for (int j = 0; j < NT8; ++j) mma16832_e4m3(acc[r][j], a, b[ky][j][0], b[ky][j][1]);
          }
        }
      }
    }
  }
}

}  // namespace detail

// ------------------------------------------------------------------ epilogues
//
// Epilogue contract (stable): an epilogue sees the block's FP32 accumulator tile in shared
// memory, laid out [TH * 16 pixels][ld] (pixel = row * 16 + col within the tile, channel
// n0 + j at column j), after a __syncthreads(); every thread of the block calls
//   Epi::apply<TH, BN, NT>(tile, ld, ty0, tx0, n0, const ConvParams&, const Epi::Params&)
// and must skip pixels with y >= p.H or x >= p.W (edge tiles are partial).

struct DefaultEpilogue {
  struct Params {};
  template <int TH, int BN, int NT>
  __device__ __forceinline__ static void apply(const float* tile, int ld, int ty0, int tx0,
                                               int n0, const ConvParams& p, const Params&) {
    constexpr int C8 = BN / 8;
    for (int u = threadIdx.x; u < TH * 16 * C8; u += NT) {
      const int pix = u / C8, cg = (u % C8) * 8;
      const int y = ty0 + pix / 16, x = tx0 + pix % 16;
      if (y >= p.H || x >= p.W) continue;
      const int n = n0 + cg;
      const size_t o = (static_cast<size_t>(y) * p.W + x) * p.cout + n;
      const float4 t0 = *reinterpret_cast<const float4*>(tile + pix * ld + cg);
      const float4 t1 = *reinterpret_cast<const float4*>(tile + pix * ld + cg + 4);
      float v[8] = {t0.x, t0.y, t0.z, t0.w, t1.x, t1.y, t1.z, t1.w};
      if (p.bias) {
        const float4 b0 = __ldg(reinterpret_cast<const float4*>(p.bias + n));
        const float4 b1 = __ldg(reinterpret_cast<const float4*>(p.bias + n + 4));
        v[0] += b0.x; v[1] += b0.y; v[2] += b0.z; v[3] += b0.w;
        v[4] += b1.x; v[5] += b1.y; v[6] += b1.z; v[7] += b1.w;
      }
      if (p.residual) {
        uint4 rv = __ldg(reinterpret_cast<const uint4*>(p.residual + o));
        const __half* rh = reinterpret_cast<const __half*>(&rv);
#pragma unroll
        for (int j = 0; j < 8; ++j) v[j] += __half2float(rh[j]);
      }
      if (p.silu) {
#pragma unroll
        for (int j = 0; j < 8; ++j) v[j] = detail::silu(v[j]);
      }
      uint4 ov;
      __half2* oh = reinterpret_cast<__half2*>(&ov);
#pragma unroll
      for (int j = 0; j < 4; ++j) oh[j] = __floats2half2_rn(v[2 * j], v[2 * j + 1]);
      *reinterpret_cast<uint4*>(p.out + o) = ov;
    }
  }
};

// ------------------------------------------------------------------ kernel

template <InMode kMode, int KS, InMode kProj, int BK, int TH, int BN, int WM, int WN,
          bool kF8 = false>
struct ConvV2Cfg {
  static constexpr int NT = WM * WN * 32;
  static constexpr int R = KS / 2;
  static constexpr bool kHasProj = kProj != InMode::kNoProj;
  static constexpr int kEl = kF8 ? 1 : 2;  // bytes per main-segment operand element
  static constexpr int kA1 = (TH + 2 * R) * (16 + 2 * R) * BK * kEl;
  static constexpr int kB1 =
      kF8 ? KS * KS * BK * BN : KS * KS * BK * detail::row_chunks<BN / 8>() * 16;
  static constexpr int kA2 = kHasProj ? TH * 16 * BK * 2 : 0;  // projection segment (FP16)
  static constexpr int kB2 = kHasProj ? BK * detail::row_chunks<BN / 8>() * 16 : 0;
  static constexpr int kA = kA1 > kA2 ? kA1 : kA2;  // bytes per stage
  static constexpr int kStage = kA + (kB1 > kB2 ? kB1 : kB2);
  // raw upsample-source patch slots (two: one chunk being transformed, one in flight)
  // (kUpCatRs: one more source row and column, plus the stager's axis table after the slots)
  static constexpr bool kRs = kMode == InMode::kUpCatRs || kProj == InMode::kUpCatRs;
  static constexpr int kRaw = (is_upcat(kMode) || is_upcat(kProj))
                                  ? (TH / 2 + 2 + kRs) * (10 + kRs) * BK * 2 : 0;
  static constexpr int kTab = kRs ? (TH + 2 * R + 16 + 2 * R) * 16 : 0;
  static constexpr int kEpiLd = BN + 4;
  static constexpr int kEpi = TH * 16 * kEpiLd * 4;
  static constexpr int kMain = 2 * kStage + 2 * kRaw + kTab;
  static constexpr int kSmem = kMain > kEpi ? kMain : kEpi;
  // ConvParams::up_add: low-res patch ((TH/2 + 2) x 10 pixels x BN) after the epilogue tile
  // in its own slot after everything, loaded with the first chunk, when that keeps the blocks
  // per SM (100 KB, 1 KB reserved per block); else after the epilogue tile, loaded after the
  // main loop.
  // (pixel stride BN + 8 halves: the 8 lanes of a fragment row read distinct banks)
  static constexpr int kZPix = (TH / 2 + 2) * 10, kZLd = BN + 8, kZBytes = kZPix * kZLd * 2;
  static constexpr int kZEnd = (kSmem + 127) / 128 * 128 + kZBytes;
  static constexpr bool kZEarly = 102400 / (kZEnd + 1024) >= 102400 / (kSmem + 1024);
  static constexpr int kZOff = kZEarly ? (kSmem + 127) / 128 * 128 : (kEpi + 127) / 128 * 128;
  static constexpr int kSmemZ = kZOff + kZBytes > kSmem ? kZOff + kZBytes : kSmem;
  // Single-chunk launch (Cin == BK, no projection, no upsample): only stage 0 is used.
  static constexpr int kSmem1 = kStage > kEpi ? kStage : kEpi;
  // Aim for <= 128 registers per thread (2 x 256-thread or 4 x 128-thread blocks per SM),
  // unless only one block fits in shared memory anyway or the accumulators alone need more
  // than 32 registers (e.g. TH = 32 tiles), where the cap would only cause spills.
  static constexpr int kAccRegs = (TH / WM) * (BN / WN / 8) * 4;
  static constexpr int kMinBlocks =
      (NT >= 512 || kAccRegs > 32 || kMain > 50 * 1024) ? 1 : 65536 / (NT * 128);
  static_assert(TH % WM == 0, "tile rows must split across M warps");
  static_assert(BN % (WN * 8) == 0, "BN must split into n8 tiles across N warps");
  static_assert(BK % 8 == 0 && BK <= 64 && (BK % 16 == 0 || (BK / 8) % 2 == 1),
                "BK must be a multiple of 8 up to 64");
  static_assert(BK % 16 == 0 || !kF8, "BK an odd multiple of 8: FP16 only");
  static_assert(!kF8 || BK % 32 == 0, "FP8 needs BK = 32 or 64");
  static_assert(KS == 1 || KS == 3, "KS must be 1 or 3");
  static_assert(!(is_upcat(kMode) && is_upcat(kProj)), "one upsample segment per kernel");
};

template <InMode kMode, int KS, InMode kProj, int BK, int TH, int BN, int WM, int WN, class Epi,
          bool kF8 = false, int kAcc = 0>
__global__ void __launch_bounds__(WM * WN * 32,
                                  (ConvV2Cfg<kMode, KS, kProj, BK, TH, BN, WM, WN, kF8>::kMinBlocks))
    conv_v2_kernel(ConvParams p, ProjParams q, typename Epi::Params ep) {
  static_assert(kAcc == 0 || !kF8, "FP16 accumulation: FP16 operand path only");
  using Cfg = ConvV2Cfg<kMode, KS, kProj, BK, TH, BN, WM, WN, kF8>;
  constexpr int NT = Cfg::NT;
  constexpr int R = Cfg::R;
  constexpr int WR = TH / WM, NT8 = BN / WN / 8;
  constexpr bool kHasProj = Cfg::kHasProj;
  constexpr InMode kPM = kHasProj ? kProj : InMode::kPlain;
  extern __shared__ __align__(128) unsigned char smem_raw[];
  const uint32_t s_base = detail::smem_u32(smem_raw);

  const int tiles_x = (p.W + 15) / 16;  // last column tile may be partial
  const int tile = blockIdx.y;
  const int ty0 = p.row0 + (tile / tiles_x) * TH;  // row band (row0 = 0: whole image)
  const int tx0 = (tile % tiles_x) * 16;
  const int n0 = blockIdx.x * BN;

  const int cin1 = p.c0 + p.c1;
  const int n1 = cin1 / BK;
  const int n2 = kHasProj ? (q.c0 + q.c1) / BK : 0;
  const int n_chunks = n1 + n2;

  const detail::SrcDesc s1{p.in0, p.in1, p.c0, p.c1};
  const detail::SrcDesc s2{q.in0, q.in1, q.c0, q.c1};
  constexpr int kRawBudget = 8;  // uint4s; 12+ spills under the 128-register cap
  detail::Stager<kMode, R, TH, BK, NT, kRawBudget, kF8> st1;
  detail::Stager<kPM, 0, TH, BK, NT, kRawBudget, false> st2;
  if constexpr (kF8) st1.a_inv = 1.0f / p.a_scale;
  st1.s_tab = st2.s_tab = s_base + 2 * Cfg::kStage + 2 * Cfg::kRaw;
  static_assert(decltype(st1)::kRawBytes <= Cfg::kRaw && decltype(st2)::kRawBytes <= Cfg::kRaw &&
                    decltype(st1)::kTabBytes <= Cfg::kTab && decltype(st2)::kTabBytes <= Cfg::kTab,
                "upsample patch / table exceeds its shared-memory slot");
  st1.lerp16 = st2.lerp16 = p.lerp16 != 0;
  st1.prepare(s1, p.H, p.W, ty0, tx0);
  if constexpr (kHasProj) st2.prepare(s2, p.H, p.W, ty0, tx0);
  detail::WeightStager<KS * KS, BK, BN, NT, kF8> wst;
  wst.prepare(cin1, p.cout, n0);

  float acc[WR][NT8][4];
#pragma unroll
  for (int r = 0; r < WR; ++r)
#pragma unroll
    for (int j = 0; j < NT8; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[r][j][e] = 0.0f;
  // FP16 accumulators (kAcc 1: per chunk, kAcc 2: whole K), same fragment layout as acc
  uint32_t hacc[kAcc ? WR : 1][kAcc ? NT8 : 1][2];
  auto hzero = [&]() {
#pragma unroll
    for (int r = 0; r < (kAcc ? WR : 1); ++r)
#pragma unroll
      for (int j = 0; j < (kAcc ? NT8 : 1); ++j) hacc[r][j][0] = hacc[r][j][1] = 0u;
  };
  auto hflush = [&]() {  // acc += float(hacc)
    if constexpr (kAcc != 0) {
#pragma unroll
      for (int r = 0; r < WR; ++r)
#pragma unroll
        for (int j = 0; j < NT8; ++j) {
          const float2 lo = __half22float2(*reinterpret_cast<const __half2*>(&hacc[r][j][0]));
          const float2 hi = __half22float2(*reinterpret_cast<const __half2*>(&hacc[r][j][1]));
          acc[r][j][0] += lo.x;
          acc[r][j][1] += lo.y;
          acc[r][j][2] += hi.x;
          acc[r][j][3] += hi.y;
        }
    }
  };
  if constexpr (kAcc != 0) hzero();
  // one FP16 chunk of MMAs into acc (FP32), or into hacc (flushed per chunk for kAcc 1)
  auto mma_h = [&](auto chunk_fn) {
    if constexpr (kAcc == 0) {
      chunk_fn(acc);
    } else {
      if constexpr (kAcc == 1) hzero();
      chunk_fn(hacc);
      if constexpr (kAcc == 1) hflush();
    }
  };

  using detail::StageKind;
  auto buf = [&](int kc) { return s_base + (kc & 1) * Cfg::kStage; };
  auto raw_slot = [&](int kc) { return s_base + 2 * Cfg::kStage + (kc & 1) * Cfg::kRaw; };
  auto is_raw = [&](int kc) {
    if (kc >= n_chunks) return false;
    if (kc < n1) return st1.is_raw(s1, kc * BK);
    return st2.is_raw(s2, (kc - n1) * BK);
  };
  auto issue_raw = [&](int kc) {
    if (kc < n1) {
      st1.issue_raw(s1, p.H, p.W, ty0, tx0, kc * BK, raw_slot(kc));
    } else if constexpr (kHasProj) {
      st2.issue_raw(s2, p.H, p.W, ty0, tx0, (kc - n1) * BK, raw_slot(kc));
    }
  };
  auto transform = [&](int kc) {
    if (kc < n1) {
      st1.transform(p.H, p.W, ty0, tx0, buf(kc), raw_slot(kc));
    } else if constexpr (kHasProj) {
      st2.transform(p.H, p.W, ty0, tx0, buf(kc), raw_slot(kc));
    }
  };
  auto issue_a = [&](int kc) -> StageKind {  // non-raw chunk
    if (kc < n1) return st1.issue(s1, p.H, p.W, ty0, tx0, kc * BK, buf(kc));
    if constexpr (kHasProj) return st2.issue(s2, p.H, p.W, ty0, tx0, (kc - n1) * BK, buf(kc));
    return StageKind::kAsync;
  };
  auto finish_reg = [&](int kc) {
    if (kc < n1) {
      st1.finish_reg(buf(kc));
    } else if constexpr (kHasProj) {
      st2.finish_reg(buf(kc));
    }
  };
  auto stage_b = [&](int kc) {
    const uint32_t s_b = buf(kc) + Cfg::kA;
    if (kc < n1) {
      if constexpr (kF8) {
        wst.issue(p.weight8, kc * BK, s_b);
      } else {
        wst.issue(p.weight, kc * BK * p.cout, s_b);
      }
    } else if constexpr (kHasProj) {
      detail::stage_weights<1, BK, BN, NT>(q.weight, q.c0 + q.c1, p.cout, (kc - n1) * BK, n0,
                                           s_b);
    }
  };
  // FP8: fold the dequantisation scale into the accumulators once the main segment is done
  // (before an FP16 projection segment accumulates true-scale values).
  auto rescale = [&]() {
    if constexpr (kF8) {
      const int lane = threadIdx.x & 31, wn = (threadIdx.x >> 5) / WM;
#pragma unroll
      for (int j = 0; j < NT8; ++j) {
        const int n = n0 + (wn * NT8 + j) * 8 + 2 * (lane & 3);
        const float s0 = p.a_scale * __ldg(p.w_scale + n);
        const float s1 = p.a_scale * __ldg(p.w_scale + n + 1);
#pragma unroll
        for (int r = 0; r < WR; ++r) {
          acc[r][j][0] *= s0;
          acc[r][j][1] *= s1;
          acc[r][j][2] *= s0;
          acc[r][j][3] *= s1;
        }
      }
    }
  };

  // ConvParams::up_add: cp.async of the low-res patch rows ty0/2 - 1 .. + TH/2 + 1, columns
  // tx0/2 - 1 .. + 9 (clamped), [pixel][BN] FP16 at Cfg::kZOff
  auto stage_z = [&]() {
    const int Hs = p.H / 2, Ws = p.W / 2, sy0 = ty0 / 2 - 1, sx0 = tx0 / 2 - 1;
    constexpr int C8 = BN / 8;
    for (int u = threadIdx.x; u < Cfg::kZPix * C8; u += NT) {
      const int pp = u / C8, c8 = u % C8;
      const int sy = min(max(sy0 + pp / 10, 0), Hs - 1), sx = min(max(sx0 + pp % 10, 0), Ws - 1);
      detail::cp_async16(s_base + Cfg::kZOff + (pp * Cfg::kZLd + c8 * 8) * 2,
                         p.up_add + (static_cast<size_t>(sy) * Ws + sx) * p.cout + n0 + c8 * 8,
                         true);
    }
  };
  // Pipeline: one barrier per chunk. Iteration kc: issue the raw upsample patch of chunk
  // kc+2, the plain / register loads and weights of chunk kc+1; run chunk kc's MMAs; then
  // complete chunk kc+1 (transform its raw patch, or combine its registers).
  {
    if (Cfg::kZEarly && p.up_add) stage_z();  // committed with chunk 0's group
    if (is_raw(0)) issue_raw(0);
    if (is_raw(1)) issue_raw(1);
    const StageKind k = is_raw(0) ? StageKind::kAsync : issue_a(0);
    stage_b(0);
    detail::cp_async_commit();
    if (is_raw(0)) {
      detail::cp_async_wait_all();
      __syncthreads();
      transform(0);
    }
    if (k == StageKind::kReg) finish_reg(0);
  }
  for (int kc = 0; kc < n_chunks; ++kc) {
    detail::cp_async_wait_all();
    __syncthreads();
    StageKind k = StageKind::kAsync;
    const bool next_raw = is_raw(kc + 1);
    if (is_raw(kc + 2)) issue_raw(kc + 2);
    if (kc + 1 < n_chunks) {
      if (!next_raw) k = issue_a(kc + 1);
      stage_b(kc + 1);
    }
    detail::cp_async_commit();
    const uint32_t s_a = buf(kc);
    if (kc < n1) {
      if constexpr (kF8) {
        detail::mma_chunk_f8<KS, TH, BK, BN, WM, WN>(s_a, s_a + Cfg::kA, acc);
      } else {
        mma_h([&](auto& a) { detail::mma_chunk_h<KS, TH, BK, BN, WM, WN>(s_a, s_a + Cfg::kA, a); });
      }
      if (kc == n1 - 1) rescale();
    } else if constexpr (kHasProj) {
      mma_h([&](auto& a) { detail::mma_chunk_h<1, TH, BK, BN, WM, WN>(s_a, s_a + Cfg::kA, a); });
    }
    if (next_raw) transform(kc + 1);
    if (k == StageKind::kReg) finish_reg(kc + 1);
  }
  if constexpr (kAcc == 2) hflush();
  __syncthreads();

  // up_add: acc += bilinear_up2x(up_add) from the low-res patch (late slot: staged now, after
  // the epilogue tile, the stages being free)
  if (p.up_add) {
    if constexpr (!Cfg::kZEarly) {
      stage_z();
      detail::cp_async_commit();
      detail::cp_async_wait_all();
      __syncthreads();
    }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int wm = warp % WM, wn = warp / WM;
    const int g = lane >> 2, t = lane & 3;
    const int Hs = p.H / 2, Ws = p.W / 2, sy0 = ty0 / 2 - 1, sx0 = tx0 / 2 - 1;
    const __half2* z_s = reinterpret_cast<const __half2*>(smem_raw + Cfg::kZOff);
    int xo[2][2];
    float wx[2];
#pragma unroll
    for (int hf = 0; hf < 2; ++hf) {
      int x0, x1;
      detail::up_coord(min(tx0 + g + 8 * hf, p.W - 1), Ws, x0, x1, wx[hf]);
      xo[hf][0] = x0 - sx0;
      xo[hf][1] = x1 - sx0;
    }
#pragma unroll
    for (int r = 0; r < WR; ++r) {
      int y0, y1;
      float wy;
      detail::up_coord(min(ty0 + wm * WR + r, p.H - 1), Hs, y0, y1, wy);
#pragma unroll
      for (int j = 0; j < NT8; ++j) {
        const int col = (wn * NT8 + j) * 8 + 2 * t;
#pragma unroll
        for (int hf = 0; hf < 2; ++hf) {
          auto ld = [&](int yy, int xx) {
            return z_s[(((yy - sy0) * 10 + xx) * Cfg::kZLd + col) / 2];
          };
          const float2 a = __half22float2(ld(y0, xo[hf][0])), b = __half22float2(ld(y0, xo[hf][1]));
          const float2 c = __half22float2(ld(y1, xo[hf][0])), d = __half22float2(ld(y1, xo[hf][1]));
          const float tx = a.x + (b.x - a.x) * wx[hf], ty = a.y + (b.y - a.y) * wx[hf];
          const float bx = c.x + (d.x - c.x) * wx[hf], by = c.y + (d.y - c.y) * wx[hf];
          acc[r][j][2 * hf] += tx + (bx - tx) * wy;
          acc[r][j][2 * hf + 1] += ty + (by - ty) * wy;
        }
      }
    }
  }
  // Accumulators -> FP32 tile in shared memory.
  float* tile_s = reinterpret_cast<float*>(smem_raw);
  {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int wm = warp % WM, wn = warp / WM;
    const int g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int r = 0; r < WR; ++r)
#pragma unroll
      for (int j = 0; j < NT8; ++j) {
        const int pix = (wm * WR + r) * 16 + g;
        const int col = (wn * NT8 + j) * 8 + 2 * t;
        *reinterpret_cast<float2*>(tile_s + pix * Cfg::kEpiLd + col) =
            make_float2(acc[r][j][0], acc[r][j][1]);
        *reinterpret_cast<float2*>(tile_s + (pix + 8) * Cfg::kEpiLd + col) =
            make_float2(acc[r][j][2], acc[r][j][3]);
      }
  }
  __syncthreads();
  Epi::template apply<TH, BN, NT>(tile_s, Cfg::kEpiLd, ty0, tx0, n0, p, ep);
}

namespace detail {

// Host launch attributes and occupancy belong to the current CUDA device.
// Keep independent caches for concurrent callers and reinitialize on a device switch.
// kSlot separates the caches of kernel variants selected at run time from one call site.
template <int kSlot = 0, class Init>
inline cudaError_t device_launch_value(int& value, Init init) {
  static thread_local int cached_device = -1;
  static thread_local int cached_value = 0;
  int device = 0;
  cudaError_t status = cudaGetDevice(&device);
  if (status != cudaSuccess) return status;
  if (cached_device != device) {
    status = init(device, cached_value);
    if (status != cudaSuccess) return status;
    cached_device = device;
  }
  value = cached_value;
  return cudaSuccess;
}

template <class Kernel>
inline cudaError_t resident_blocks(Kernel kernel, int device, int threads, int smem, int& blocks) {
  if (smem > 48 * 1024) {
    const auto status = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                             smem);
    if (status != cudaSuccess) return status;
  }
  int sms = 0, per_sm = 0;
  auto status = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
  if (status == cudaSuccess)
    status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, threads, smem);
  if (status != cudaSuccess) return status;
  if (per_sm < 1) return cudaErrorInvalidConfiguration;
  blocks = sms * per_sm;
  return cudaSuccess;
}
// Output rows of the band [row0, row0 + rows) clipped to H; false if the band is not aligned
// to TH (row0 % TH, or rows % TH unless the band reaches the bottom).
inline bool row_band(const ConvParams& p, int TH, int& band) {
  if (p.row0 < 0 || p.rows < 0 || p.row0 % TH) return false;
  const int left = p.H - p.row0;
  band = p.rows == 0 || p.rows >= left ? (left > 0 ? left : 0) : p.rows;
  return band == left || band % TH == 0;
}
}  // namespace detail

}  // namespace gfused

#include "conv_phase.cuh"

namespace gfused {

// Launcher. Requirements: every channel count a multiple of BK, cout multiple of BN,
// pointers 16-byte aligned. Any H, W. kF8 = true selects the e4m3 operand path for the main
// segment (needs p.weight8 / p.w_scale / p.a_scale, BK 32 or 64, channels multiple of BK).
// The FP16-accumulate kernels are compiled only with GFUSED_ACC16=1 (CMake option
// GFUSED_ACC16; they roughly double the conv kernel count); without it ConvParams::acc16 != 0
// fails with cudaErrorNotSupported.
#ifndef GFUSED_ACC16
#define GFUSED_ACC16 0
#endif

// Compute capability of the current device as major * 10 + minor (89 for an L40S, 120 for an
// RTX 5090), or 0 if it cannot be queried. Per-architecture tile tables and defaults key on it.
inline int device_arch() {
  int device = 0, major = 0, minor = 0;
  if (cudaGetDevice(&device) != cudaSuccess ||
      cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess ||
      cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device) != cudaSuccess)
    return 0;
  return major * 10 + minor;
}

// ConvParams::acc16 for every FP16 conv-core layer when GFUSED_FM_ACC16 / GFUSED_HEAD_ACC16
// are unset: mode 1 (FP16 accumulation per K chunk, FP32 chunk sums) on sm_120, whose
// FP16-accumulate mma.sync rate is twice the FP32-accumulate one (RTX 5090: 510 vs 256 dense
// TFLOP/s); 0 on every other device (the L40S runs both at one rate) and in builds without
// the FP16-accumulate kernels.
inline int default_acc16() {
#if GFUSED_ACC16
  return device_arch() == 120 ? 1 : 0;
#else
  return 0;
#endif
}

template <InMode kMode, int KS, InMode kProj, int BK, int TH, int BN, int WM, int WN,
          class Epi = DefaultEpilogue, bool kF8 = false>
inline cudaError_t launch_conv2(const ConvParams& p, const ProjParams& q,
                                const typename Epi::Params& ep, cudaStream_t stream) {
  if constexpr (kMode == InMode::kUpCatPh) {
    static_assert(KS == 3 && kProj == InMode::kNoProj && !kF8, "kUpCatPh: 3x3, FP16, no proj");
    return detail::launch_ph<TH, BK, BN, WM, WN, Epi>(p, ep, stream);
  } else {
  using Cfg = ConvV2Cfg<kMode, KS, kProj, BK, TH, BN, WM, WN, kF8>;
  auto kern = conv_v2_kernel<kMode, KS, kProj, BK, TH, BN, WM, WN, Epi, kF8, 0>;
  if (p.acc16 < 0 || p.acc16 > 2) return cudaErrorInvalidValue;
  if (p.acc16) {
#if GFUSED_ACC16
    if constexpr (kF8) {
      return cudaErrorNotSupported;
    } else {
      kern = p.acc16 == 1 ? conv_v2_kernel<kMode, KS, kProj, BK, TH, BN, WM, WN, Epi, kF8, 1>
                          : conv_v2_kernel<kMode, KS, kProj, BK, TH, BN, WM, WN, Epi, kF8, 2>;
    }
#else
    return cudaErrorNotSupported;
#endif
  }
  if constexpr (Cfg::kSmemZ > 48 * 1024) {
    int ignored = 0;
    const auto set_smem = [kern](int, int&) {
      return cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, Cfg::kSmemZ);
    };
    // One device-aware cache per accumulation mode.
    const auto status = p.acc16 == 0   ? detail::device_launch_value<0>(ignored, set_smem)
                        : p.acc16 == 1 ? detail::device_launch_value<1>(ignored, set_smem)
                                       : detail::device_launch_value<2>(ignored, set_smem);
    if (status != cudaSuccess) return status;
  }
  int band = 0;
  if (!detail::row_band(p, TH, band)) return cudaErrorInvalidValue;
  if (band == 0) return cudaSuccess;
  const int tiles_x = (p.W + 15) / 16;
  if (tiles_x < 1 || tiles_x > 65535) return cudaErrorInvalidValue;
  // grid.y is limited to 65535 even when the tensor itself fits in memory.
  const int max_rows = (65535 / tiles_x) * TH;
  const bool one_chunk =
      p.single_stage && !is_upcat(kMode) && !Cfg::kHasProj && p.c0 + p.c1 == BK;
  ConvParams part = p;
  for (int row = 0; row < band; row += max_rows) {
    part.row0 = p.row0 + row;
    part.rows = max_rows < band - row ? max_rows : band - row;
    const int tiles = ((part.rows + TH - 1) / TH) * tiles_x;
    const dim3 grid(p.cout / BN, tiles);
    const int smem = one_chunk ? Cfg::kSmem1 : p.up_add ? Cfg::kSmemZ : Cfg::kSmem;
    kern<<<grid, Cfg::NT, smem, stream>>>(part, q, ep);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return e;
  }
  return cudaSuccess;
  }
}

}  // namespace gfused
