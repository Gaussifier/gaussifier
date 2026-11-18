// Reusable conv_fused v2 epilogues (Epi::apply contract, see conv_fused.cuh).
//
// PoolEpilogue: DefaultEpilogue's store (bias + residual + SiLU, NHWC FP16 to p.out) plus a
// second, 2x2 max-pooled copy of the same FP16 values to ep.pooled. A later layer that would
// read p.out through InMode::kPool (4 loads + max per staged unit, from a tensor 4x the size
// it needs) reads ep.pooled through InMode::kPlain instead (cp.async path, 1 load).
//
// Usage (producer, any launch_conv2 / launch_block instantiation; FP16 or FP8):
//
//     PoolEpilogue::Params ep{pooled};   // __half*, NHWC [H/2][W/2][p.cout], 16-byte aligned
//     launch_conv2<M, KS, PM, BK, TH, BN, WM, WN, PoolEpilogue>(p, q, ep, stream);
//
//   consumer: InMode::kPool on p.out at (H/2, W/2)  ->  InMode::kPlain on `pooled`, same
//   ConvParams otherwise (c0, H, W are the pooled layer's, exactly as with kPool).
//
// Semantics: pooled[yp][xp][n] = max over the 2x2 window of the FP16 values written to p.out,
// computed with the same __hmax2 as the kPool staging (detail::max4), so a kPlain consumer of
// `pooled` is bit-identical to a kPool consumer of p.out. Windows are tile local because ty0
// and tx0 are even (TH even, tile width 16). Pooled size is floor(H/2) x floor(W/2) (as
// max_pool2d); for odd H or W the trailing row/column is stored but not pooled.
//
// Optional halves: ep.pooled == nullptr behaves exactly like DefaultEpilogue; p.out == nullptr
// writes only the pooled tensor (when nothing reads the full-resolution output).
//
// Per-thread work unit: one 2x2 window x 8 channels (bias loaded once for the four pixels).
// Store coalescing: lanes cover consecutive 8-channel groups of a pixel, then every other
// pixel of a tile row; the four pixel stores of a window are four instructions, each writing
// whole 32-byte sectors whenever BN >= 16.
//
// F8StoreEpilogue: FP8 activation storage. Computes exactly DefaultEpilogue's FP16 values and
// stores them quantised for the consumer, e4m3(h / ep.a_scale) (saturating, round to nearest
// even; the same detail::to_e4m3 as the FP8 staging), to ep.out8, NHWC [H][W][p.cout] bytes.
// The consumer is an FP8 conv (launch_conv2<..., kF8 = true>) with InMode::kPlainF8 on out8
// and the same a_scale: it stages the bytes with cp.async instead of quantising the FP16
// tensor while staging, with bit-identical results (the quantisation is the same operation
// at the same point; 1 / a_scale is computed on the device as the conv kernel does). Halves
// the bytes of the intermediate tensor (written once, read ~1.3x with the halo). p.out is
// not written.

#pragma once

#include "conv_fused.cuh"

namespace gfused {

namespace epi_detail {

// bias + residual + SiLU on 8 consecutive channels of tile pixel `pix`, rounded to FP16.
// Same operation order as DefaultEpilogue (bit-identical stored values).
__device__ __forceinline__ uint4 finish8(const float* tile, int ld, int pix, int cg, const float* b,
                                         const ConvParams& p, size_t o) {
  const float4 t0 = *reinterpret_cast<const float4*>(tile + pix * ld + cg);
  const float4 t1 = *reinterpret_cast<const float4*>(tile + pix * ld + cg + 4);
  float v[8] = {t0.x, t0.y, t0.z, t0.w, t1.x, t1.y, t1.z, t1.w};
  if (p.bias) {
#pragma unroll
    for (int j = 0; j < 8; ++j) v[j] += b[j];
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
  return ov;
}

}  // namespace epi_detail

struct PoolEpilogue {
  struct Params {
    __half* pooled = nullptr;  // NHWC [H/2][W/2][p.cout] (channel stride p.cout, like p.out)
  };
  template <int TH, int BN, int NT>
  __device__ __forceinline__ static void apply(const float* tile, int ld, int ty0, int tx0,
                                               int n0, const ConvParams& p, const Params& ep) {
    static_assert(TH % 2 == 0, "pooling windows must be tile local (even TH)");
    static_assert(BN % 8 == 0, "8-channel units");
    constexpr int C8 = BN / 8;
    const int Wp = p.W >> 1;
    for (int u = threadIdx.x; u < (TH / 2) * 8 * C8; u += NT) {
      const int win = u / C8, cg = (u % C8) * 8;
      const int wy = win >> 3, wx = win & 7;
      const int n = n0 + cg;
      float b[8];
      if (p.bias) {
        const float4 b0 = __ldg(reinterpret_cast<const float4*>(p.bias + n));
        const float4 b1 = __ldg(reinterpret_cast<const float4*>(p.bias + n + 4));
        b[0] = b0.x; b[1] = b0.y; b[2] = b0.z; b[3] = b0.w;
        b[4] = b1.x; b[5] = b1.y; b[6] = b1.z; b[7] = b1.w;
      }
      uint4 h[4];
      bool full = true;
#pragma unroll
      for (int d = 0; d < 4; ++d) {
        const int ly = 2 * wy + (d >> 1), lx = 2 * wx + (d & 1);
        const int y = ty0 + ly, x = tx0 + lx;
        if (y >= p.H || x >= p.W) {
          full = false;
          continue;
        }
        const size_t o = (static_cast<size_t>(y) * p.W + x) * p.cout + n;
        h[d] = epi_detail::finish8(tile, ld, ly * 16 + lx, cg, b, p, o);
        if (p.out) *reinterpret_cast<uint4*>(p.out + o) = h[d];
      }
      if (ep.pooled && full) {
        const int yp = (ty0 >> 1) + wy, xp = (tx0 >> 1) + wx;
        *reinterpret_cast<uint4*>(ep.pooled + (static_cast<size_t>(yp) * Wp + xp) * p.cout + n) =
            detail::max4(h[0], h[1], h[2], h[3]);
      }
    }
  }
};

struct F8StoreEpilogue {
  struct Params {
    uint8_t* out8 = nullptr;  // NHWC e4m3 [H][W][p.cout] (16-byte aligned rows: cout % 16 == 0)
    float a_scale = 1.0f;     // the consumer's static activation scale
  };
  template <int TH, int BN, int NT>
  __device__ __forceinline__ static void apply(const float* tile, int ld, int ty0, int tx0,
                                               int n0, const ConvParams& p, const Params& ep) {
    static_assert(BN % 8 == 0, "8-channel units");
    constexpr int C8 = BN / 8;
    const float inv = 1.0f / ep.a_scale;  // as the conv kernel's Stager::a_inv
    for (int u = threadIdx.x; u < TH * 16 * C8; u += NT) {
      const int pix = u / C8, cg = (u % C8) * 8;
      const int y = ty0 + pix / 16, x = tx0 + pix % 16;
      if (y >= p.H || x >= p.W) continue;
      const int n = n0 + cg;
      const size_t o = (static_cast<size_t>(y) * p.W + x) * p.cout + n;
      float b[8];
      if (p.bias) {
        const float4 b0 = __ldg(reinterpret_cast<const float4*>(p.bias + n));
        const float4 b1 = __ldg(reinterpret_cast<const float4*>(p.bias + n + 4));
        b[0] = b0.x; b[1] = b0.y; b[2] = b0.z; b[3] = b0.w;
        b[4] = b1.x; b[5] = b1.y; b[6] = b1.z; b[7] = b1.w;
      }
      const uint2 q = detail::to_e4m3(epi_detail::finish8(tile, ld, pix, cg, b, p, o), inv);
      *reinterpret_cast<uint2*>(ep.out8 + o) = q;
    }
  }
};

}  // namespace gfused
