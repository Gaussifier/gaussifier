// Torch-free fused forward map: weight loading, workspace, and the layer schedule.
#include "forward_map.cuh"

#include <math_constants.h>

#include <cstdlib>
#include <cstring>
#include <fstream>

#include "conv_block.cuh"
#include "epilogues.cuh"
#include "pipeline_common.cuh"
#include "upfold.h"

namespace gfused {
namespace {

constexpr uint32_t kMagic = 0x4d464647;  // "GFFM"
constexpr uint32_t kVersion = 1, kVersionFp8 = 2;
constexpr int kHeadOut = 8;

size_t align16(size_t n) { return (n + 15) & ~size_t(15); }

// ---------------------------------------------------------------- tile configurations

// v2 tile shapes (BK, TH, BN, WM x WN warps), picked per layer from the sweep in
// tests/fused/test_conv_fused.py on sm_89 (L40S); BK = 16 wherever a 16-channel input
// (stem) shares the kernel.
template <int BK_, int TH_, int BN_, int WM_, int WN_>
struct Tile {
  static constexpr int BK = BK_, TH = TH_, BN = BN_, WM = WM_, WN = WN_;
};
using TileR32 = Tile<16, 16, 32, 8, 1>;  // full-resolution 32-channel layers
using TileM64 = Tile<16, 8, 64, 4, 2>;   // 64/128-channel layers
using TileL64 = Tile<16, 16, 64, 8, 2>;  // larger M tile, 64-wide N
using TileS32 = Tile<16, 8, 32, 4, 1>;   // small maps, many N blocks
// Upsample + concat convs: 4 tile rows per warp (WR = 4) with 32 channels per warp halve the
// B-fragment loads per MMA. L40S, 512 x 768 isolated bursts: up1.conv1 113.2 -> 107.7 us,
// up2.conv1 119.0 -> 115.7 us (up0.conv1 keeps TileR32: 115.5 vs 117.0 us).
using TileU32 = Tile<16, 16, 32, 4, 1>;  // up2.conv1
using TileU64 = Tile<16, 16, 64, 4, 2>;  // up1.conv1

template <class T, InMode M, int KS, InMode PM = InMode::kNoProj, class Epi = DefaultEpilogue>
cudaError_t launch(const ConvParams& p, const ProjParams& q, const typename Epi::Params& ep,
                   cudaStream_t s) {
  return launch_conv2<M, KS, PM, T::BK, T::TH, T::BN, T::WM, T::WN, Epi>(p, q, ep, s);
}

ConvParams make_params(const ConvLayer& L, const __half* in0, int c0, const __half* in1, int c1,
                       int H, int W, const __half* residual, bool silu, __half* out) {
  ConvParams p{};
  p.in0 = in0;
  p.c0 = c0;
  p.in1 = in1;
  p.c1 = c1;
  p.H = H;
  p.W = W;
  p.weight = L.w;
  p.bias = L.b;
  p.residual = residual;
  p.out = out;
  p.cout = L.cout;
  p.silu = silu ? 1 : 0;
  p.acc16 = L.acc16;
  return p;
}

// Order-preserving float <-> int mapping for atomicMax on floats.
__device__ __forceinline__ int f2ord(float f) {
  int i = __float_as_int(f);
  return i >= 0 ? i : i ^ 0x7fffffff;
}
__device__ __forceinline__ float ord2f(int i) {
  return __int_as_float(i >= 0 ? i : i ^ 0x7fffffff);
}

// Output heads fused into the last conv's epilogue: p2 = silu(acc + bias [+ residual]) is
// rounded to FP16 exactly as if it had been stored, then the per-pixel 1x1 heads (density
// logit, log-cov xx/xy/yy, sigmoid rgb) are evaluated and the block's max logit is folded
// into a global atomicMax. p2 itself never touches memory.
struct HeadEpilogue {
  struct Params {
    const __half* w;  // [32][kHeadOut]
    const float* b;   // [kHeadOut]
    float* logits;    // H x W
    float* log_cov;   // 3 x H x W
    float* rgb;       // 3 x H x W
    int* max_ord;
  };
  template <int TH, int BN, int NT>
  __device__ __forceinline__ static void apply(const float* tile, int ld, int ty0, int tx0, int,
                                               const ConvParams& p, const Params& ep) {
    static_assert(BN == 32, "head epilogue needs all 32 channels of a pixel in one block");
    __shared__ float s_w[32 * kHeadOut];
    __shared__ float s_cb[32];
    __shared__ float s_max[NT / 32];
    for (int i = threadIdx.x; i < 32 * kHeadOut; i += NT) s_w[i] = __half2float(ep.w[i]);
    for (int i = threadIdx.x; i < 32; i += NT) s_cb[i] = p.bias ? p.bias[i] : 0.0f;
    __syncthreads();
    const int n_pix = p.H * p.W;
    float lmax = -INFINITY;
    for (int pix = threadIdx.x; pix < TH * 16; pix += NT) {
      const int y = ty0 + pix / 16, x = tx0 + pix % 16;
      if (y >= p.H || x >= p.W) continue;
      const size_t i = static_cast<size_t>(y) * p.W + x;
      float acc[7];
#pragma unroll
      for (int o = 0; o < 7; ++o) acc[o] = ep.b[o];
#pragma unroll 1  // fully unrolled, ptxas hoists all 224 weight loads and spills
      for (int c4 = 0; c4 < 8; ++c4) {
        const float4 t = *reinterpret_cast<const float4*>(tile + pix * ld + c4 * 4);
        float v[4] = {t.x, t.y, t.z, t.w};
        if (p.residual) {
          const __half* r = p.residual + i * 32 + c4 * 4;
#pragma unroll
          for (int j = 0; j < 4; ++j) v[j] += __half2float(r[j]);
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const int c = c4 * 4 + j;
          float a = v[j] + s_cb[c];
          if (p.silu) a = detail::silu(a);
          a = __half2float(__float2half_rn(a));  // p2 is FP16
#pragma unroll
          for (int o = 0; o < 7; ++o) acc[o] += a * s_w[c * kHeadOut + o];
        }
      }
      lmax = fmaxf(lmax, acc[0]);
      ep.logits[i] = acc[0];
      ep.log_cov[i] = acc[1];
      ep.log_cov[n_pix + i] = acc[2];
      ep.log_cov[2 * n_pix + i] = acc[3];
#pragma unroll
      for (int c = 0; c < 3; ++c) ep.rgb[c * n_pix + i] = 1.0f / (1.0f + __expf(-acc[4 + c]));
    }
    for (int off = 16; off > 0; off >>= 1)
      lmax = fmaxf(lmax, __shfl_xor_sync(0xffffffff, lmax, off));
    if ((threadIdx.x & 31) == 0) s_max[threadIdx.x >> 5] = lmax;
    __syncthreads();
    if (threadIdx.x < 32) {
      float m = threadIdx.x < NT / 32 ? s_max[threadIdx.x] : -INFINITY;
      for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, off));
      if (threadIdx.x == 0) atomicMax(ep.max_ord, f2ord(m));
    }
  }
};

// FP8 (e4m3) tiles for the planned layers, from the FP8 sweep in test_conv_fused.py.
using F8M64 = Tile<32, 8, 64, 4, 2>;
using F8L64 = Tile<32, 16, 64, 8, 2>;
using F8R32 = Tile<32, 16, 32, 8, 1>;

// One conv: FP8 tile F when use8 (and the input modes allow FP8), else FP16 tile T.
// M = kPlainF8 (e4m3 input written by F8StoreEpilogue) exists only as FP8.
template <class T, class F, InMode M, int KS, InMode PM = InMode::kNoProj,
          class Epi = DefaultEpilogue>
cudaError_t launch_layer(const ConvLayer& L, bool use8, ConvParams p, const ProjParams& q,
                         const typename Epi::Params& ep, cudaStream_t s) {
  constexpr bool kOnly8 = M == InMode::kPlainF8;
  if constexpr (M != InMode::kImageF32 && PM != InMode::kImageF32) {
    if (use8 || kOnly8) {
      if (!use8) return cudaErrorInvalidValue;
      p.acc16 = 0;  // FP8 operands accumulate in FP32
      p.weight8 = L.w8;
      p.w_scale = L.w_scale;
      p.a_scale = L.a_scale;
      return launch_conv2<M, KS, PM, F::BK, F::TH, F::BN, F::WM, F::WN, Epi, true>(p, q, ep, s);
    }
  }
  if constexpr (!kOnly8) return launch<T, M, KS, PM, Epi>(p, q, ep, s);
  return cudaErrorInvalidValue;
}

// Does layer L run FP8? (the plan says so and every K segment of the kernel has channel
// counts the BK-32 FP8 kernel takes; the fused projection segment included).
bool runs_fp8(const ConvLayer& L, int c0, int c1, bool with_proj, int pc0, int pc1, bool image) {
  if (!L.fp8 || image) return false;
  if (c0 % 32 || c1 % 32) return false;
  return !with_proj || (pc0 % 32 == 0 && pc1 % 32 == 0);
}

// block(X) = silu(conv2(silu(conv1(X))) + proj(X)), X defined by (mode, in0, in1).
// T1/T2: FP16 conv1 / conv2 (+ fused projection) tiles, F1/F2: FP8 tiles for planned FP8
// layers. The 1x1 projection is an extra K segment of conv2 (FP16 even when conv2 is FP8).
// part: 1 = conv1 only, 2 = conv2 (+ projection) only, 3 = both. [row0, row0 + rows) is the
// output row band (rows = 0: to the bottom); see ConvParams::row0.
// When conv2 runs FP8, conv1 stores its output (t_y1, read only by conv2) as e4m3 with
// conv2's activation scale (F8StoreEpilogue) and conv2 reads it as InMode::kPlainF8: half the
// bytes, bit-identical to quantising the FP16 tensor in conv2's staging. kFwdNoF8Storage
// keeps the FP16 t_y1.
// z (kUpCat only): the projection's upsampled half already applied at low resolution
// (Z = proj_up(in0), [H/2][W/2][cout], see low_res_proj); conv2 then projects only the skip
// (in1, InMode::kPlain segment) and adds up2x(Z) before its epilogue (ConvParams::up_add).
template <InMode M, class T1, class T2, class F1, class F2, class Epi = DefaultEpilogue>
cudaError_t res_block(const ResBlock& B, const __half* in0, int c0, const __half* in1, int c1,
                      int H, int W, ForwardMapWorkspace& ws, __half* out,
                      const typename Epi::Params& ep, cudaStream_t s, int part = 3,
                      int row0 = 0, int rows = 0, const __half* z = nullptr) {
  cudaError_t e;
  constexpr bool kImg = M == InMode::kImageF32;
  const int cm = B.conv1.cout;
  auto band = [&](ConvParams p) {
    p.row0 = row0;
    p.rows = rows;
    return p;
  };
  const bool f8 = runs_fp8(B.conv2, cm, 0, true, c0, c1, kImg);
  // (never for the image stem: its projection reads the FP32 image, which FP8 cannot take)
  const bool store8 = !kImg && f8 && !(ws.flags & kFwdNoF8Storage);
  uint8_t* y1_8 = reinterpret_cast<uint8_t*>(ws.t_y1);
  if (part & 1) {
    const ConvParams p1 =
        band(make_params(B.conv1, in0, c0, in1, c1, H, W, nullptr, true, ws.t_y1));
    const bool c1f8 = runs_fp8(B.conv1, c0, c1, false, 0, 0, kImg);
    if constexpr (!kImg) {
      if (store8) {
        e = launch_layer<T1, F1, M, 3, InMode::kNoProj, F8StoreEpilogue>(
            B.conv1, c1f8, p1, ProjParams{}, {y1_8, B.conv2.a_scale}, s);
        if (e) return e;
      }
    }
    bool folded = false;  // conv1 already ran through the experimental phase-fold path
    if constexpr (M == InMode::kUpCat) {
      static const bool fold = [] {
        const char* v = std::getenv("GFUSED_FM_UPFOLD");
        return v && v[0] == '1';
      }();
      if (fold && !store8 && !c1f8 && B.conv1.wph && !p1.acc16) {
        ConvParams pf = p1;
        pf.weight_ph = B.conv1.wph;
        pf.weight_edge = B.conv1.wedge;
        if ((e = detail::launch_ph<32, 16, 32, 8, 1, DefaultEpilogue, 16>(pf, {}, s))) return e;
        folded = true;
      }
    }
    if (!folded && !store8 &&
        (e = launch_layer<T1, F1, M, 3>(B.conv1, c1f8, p1, ProjParams{}, {}, s)))
      return e;
  }
  if (!(part & 2)) return cudaSuccess;
  if constexpr (M == InMode::kUpCat) {
    if (z) {  // skip-only projection (kPlain segment over in1) + up2x(Z)
      const ProjParams qs{in1, nullptr, c1, 0, B.proj.w + static_cast<size_t>(c0) * B.proj.cout};
      if (store8) {
        ConvParams p2 = band(make_params(B.conv2, reinterpret_cast<const __half*>(y1_8), cm,
                                         nullptr, 0, H, W, nullptr, true, out));
        p2.up_add = z;
        return launch_layer<T2, F2, InMode::kPlainF8, 3, InMode::kPlain, Epi>(B.conv2, true, p2,
                                                                              qs, ep, s);
      }
      ConvParams p2 = band(make_params(B.conv2, ws.t_y1, cm, nullptr, 0, H, W, nullptr, true, out));
      p2.up_add = z;
      return launch_layer<T2, F2, InMode::kPlain, 3, InMode::kPlain, Epi>(B.conv2, f8, p2, qs, ep,
                                                                         s);
    }
  }
  const ProjParams q{in0, in1, c0, c1, B.proj.w};
  if constexpr (!kImg) {
    if (store8) {
      const ConvParams p2 = band(make_params(B.conv2, reinterpret_cast<const __half*>(y1_8), cm,
                                             nullptr, 0, H, W, nullptr, true, out));
      return launch_layer<T2, F2, InMode::kPlainF8, 3, M, Epi>(B.conv2, true, p2, q, ep, s);
    }
  }
  const ConvParams p2 =
      band(make_params(B.conv2, ws.t_y1, cm, nullptr, 0, H, W, nullptr, true, out));
  return launch_layer<T2, F2, InMode::kPlain, 3, M, Epi>(B.conv2, f8, p2, q, ep, s);
}

// The same block in row bands of band_rows (L2 cache blocking): conv1 band b, then conv2
// band b - 1 (its halo row comes from conv1 band b), so conv2 reads t_y1 and the
// projection's inputs while they are still in L2. Bit-identical to res_block.
// band_rows must be a multiple of every tile height used (16).
template <InMode M, class T1, class T2, class F1, class F2, class Epi = DefaultEpilogue>
cudaError_t res_block_banded(const ResBlock& B, const __half* in0, int c0, const __half* in1,
                             int c1, int H, int W, ForwardMapWorkspace& ws, __half* out,
                             const typename Epi::Params& ep, int band_rows, cudaStream_t s,
                             const __half* z = nullptr) {
  if (band_rows <= 0 || band_rows >= H)
    return res_block<M, T1, T2, F1, F2, Epi>(B, in0, c0, in1, c1, H, W, ws, out, ep, s, 3, 0, 0,
                                             z);
  cudaError_t e;
  for (int r = 0; r < H + band_rows; r += band_rows) {
    if (r < H && (e = res_block<M, T1, T2, F1, F2, Epi>(B, in0, c0, in1, c1, H, W, ws, out, ep,
                                                        s, 1, r, band_rows)))
      return e;
    if (r > 0 && (e = res_block<M, T1, T2, F1, F2, Epi>(B, in0, c0, in1, c1, H, W, ws, out, ep,
                                                        s, 2, r - band_rows, band_rows, z)))
      return e;
  }
  return cudaSuccess;
}

// Low-resolution projection (default on; ForwardMapWorkspace flag kFwdFullResProj or
// GFUSED_FM_LOWPROJ=0 turn it off): proj(concat(up2x(low), skip)) = up2x(proj_up(low)) +
// proj_skip(skip), as a 1x1 conv commutes with the bilinear upsample (its taps sum to 1).
// Z = proj_up(low) is a 1x1 conv at H/2 x W/2 (a quarter of the pixels, no interpolation)
// into ws.pooled[0] (free after down0; holds H/2 x W/2 x 32 >= every Z). Even H and W only.
bool low_res_proj_enabled(const ForwardMapWorkspace& ws) {
  static const bool env_off = [] {
    const char* e = std::getenv("GFUSED_FM_LOWPROJ");
    return e && e[0] == '0';
  }();
  return !env_off && !(ws.flags & kFwdFullResProj);
}

template <class TZ>
cudaError_t low_res_proj(const ResBlock& B, const __half* low, int c0, int H, int W,
                         __half* z, cudaStream_t s) {
  ConvParams p = make_params(B.proj, low, c0, nullptr, 0, H / 2, W / 2, nullptr, false, z);
  p.bias = nullptr;
  p.acc16 = 0;
  return launch<TZ, InMode::kPlain, 1>(p, ProjParams{}, {}, s);
}

template <class T1, class T2, class F1, class F2, class TZ, class Epi = DefaultEpilogue>
cudaError_t up_block(const ResBlock& B, const __half* low, int c0, const __half* skip, int c1,
                     int H, int W, ForwardMapWorkspace& ws, __half* out,
                     const typename Epi::Params& ep, int band_rows, cudaStream_t s) {
  // An odd extent: the reference resizes its up2x to the skip's size (kUpCatRs folds it in).
  if (H % 2 || W % 2)
    return res_block_banded<InMode::kUpCatRs, T1, T2, F1, F2, Epi>(
        B, low, c0, skip, c1, H, W, ws, out, ep, band_rows, s);
  const __half* z = nullptr;
  if (low_res_proj_enabled(ws)) {
    if (const cudaError_t e = low_res_proj<TZ>(B, low, c0, H, W, ws.pooled[0], s)) return e;
    z = ws.pooled[0];
  }
  return res_block_banded<InMode::kUpCat, T1, T2, F1, F2, Epi>(B, low, c0, skip, c1, H, W, ws,
                                                               out, ep, band_rows, s, z);
}

// Encoder block whose output feeds a down block: with `pooled` its conv2 epilogue also
// writes maxpool2x2(out) (PoolEpilogue), so the next block reads a plain input 4x smaller.
template <InMode M, class T1, class T2, class F1, class F2>
cudaError_t enc_block(const ResBlock& B, const __half* in0, int c0, int H, int W,
                      ForwardMapWorkspace& ws, __half* out, __half* pooled, cudaStream_t s) {
  if (pooled)
    return res_block<M, T1, T2, F1, F2, PoolEpilogue>(B, in0, c0, nullptr, 0, H, W, ws, out,
                                                      PoolEpilogue::Params{pooled}, s);
  return res_block<M, T1, T2, F1, F2>(B, in0, c0, nullptr, 0, H, W, ws, out, {}, s);
}

// Row-band schedule for up2: band height and the image size from which it is used.
// L40S, 1024x1536, up2 kernel sum: FP16 992 -> 910 us at 128 rows (64: 963, 256: 1041,
// 32: 1197); FP8 899 -> 907 us (no gain: its register-staged loads are latency, not DRAM,
// bound), so the default bands only an FP16 up2.
constexpr long long kBandedMinPixels = 1024LL * 1024;
constexpr int kBandRows = 128;

// Single-kernel residual block (conv_block.cuh): TH 8, 4 warps (stem) / 8 warps (down0).
constexpr long long kFusedBlockMinPixels = 1024LL * 1024;

template <InMode M, int CM>
cudaError_t fused_block(const ResBlock& B, const __half* in0, int c0, const __half* in1, int c1,
                        int H, int W, __half* out, __half* pooled, cudaStream_t s) {
  BlockParams bp{};
  bp.p = make_params(B.conv2, nullptr, 0, nullptr, 0, H, W, nullptr, true, out);
  bp.q = ProjParams{in0, in1, c0, c1, B.proj.w};
  bp.w1 = B.conv1.w;
  bp.b1 = B.conv1.b;
  if (pooled) {
    const PoolEpilogue::Params ep{pooled};
    if constexpr (CM == 32) return launch_block<M, 8, 4, 16, 16, CM, CM, PoolEpilogue>(bp, ep, s);
    return launch_block<M, 8, 8, 16, 16, CM, CM, PoolEpilogue>(bp, ep, s);
  }
  if constexpr (CM == 32) return launch_block<M, 8, 4, 16, 16, CM, CM>(bp, {}, s);
  return launch_block<M, 8, 8, 16, 16, CM, CM>(bp, {}, s);
}

// ---------------------------------------------------------------- small kernels

__global__ void reset_reduce_kernel(float* reduce) {
  reinterpret_cast<int*>(reduce)[0] = static_cast<int>(0x807fffff);  // f2ord(-inf)
  reduce[1] = 0.0f;
}

__global__ void exp_sum_kernel(const float* __restrict__ logits, int n_pix,
                               const int* __restrict__ max_ord, float* __restrict__ sum) {
  const float m = ord2f(*max_ord);
  float s = 0.0f;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_pix; i += gridDim.x * blockDim.x)
    s += __expf(logits[i] - m);
  for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffff, s, off);
  __shared__ float s_sum[32];
  if ((threadIdx.x & 31) == 0) s_sum[threadIdx.x >> 5] = s;
  __syncthreads();
  if (threadIdx.x < 32) {
    float t = threadIdx.x < (blockDim.x >> 5) ? s_sum[threadIdx.x] : 0.0f;
    for (int off = 16; off > 0; off >>= 1) t += __shfl_xor_sync(0xffffffff, t, off);
    if (threadIdx.x == 0) atomicAdd(sum, t);
  }
}

__global__ void density_kernel(const float* __restrict__ logits, int n_pix,
                               const int* __restrict__ max_ord, const float* __restrict__ sum,
                               float* __restrict__ density, float* __restrict__ rate_sum) {
  const float m = ord2f(*max_ord);
  const float mean = *sum / static_cast<float>(n_pix);
  const float inv = 1.0f / (mean + 1e-8f);
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n_pix) density[i] = __expf(logits[i] - m) * inv;
  if (i == 0) *rate_sum = __expf(m) * *sum;  // sum exp(l) = exp(max) * sum exp(l - max)
}

__global__ void crop_maps_kernel(const float* net_logits, const float* net_cov,
                                 const float* net_rgb, float* logits, float* cov, float* rgb,
                                 int P, int W, int NP, int NW, int* max_ord) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  float v = -CUDART_INF_F;
  if (i < P) {
    const int j = (i / W) * NW + i % W;
    v = net_logits[j];
    logits[i] = v;
    for (int c = 0; c < 3; ++c) {
      cov[c * P + i] = net_cov[c * NP + j];
      rgb[c * P + i] = net_rgb[c * NP + j];
    }
  }
  for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffff, v, off));
  if ((threadIdx.x & 31) == 0) atomicMax(max_ord, f2ord(v));
}

}  // namespace

// ---------------------------------------------------------------- weights

cudaError_t load_forward_map_weights_from_memory(const void* data, size_t size,
                                                 ForwardMapWeights* out, std::string* error) {
  const auto* bytes = static_cast<const uint8_t*>(data);
  size_t off = 0;
  auto rd = [&](uint32_t* v) {
    if (off + 4 > size) return false;
    std::memcpy(v, bytes + off, 4);
    off += 4;
    return true;
  };
  uint32_t magic, version, n;
  if (!rd(&magic) || !rd(&version) || !rd(&n) || magic != kMagic ||
      (version != kVersion && version != kVersionFp8)) {
    if (error) *error = "not a GFFM v1/v2 weight blob";
    return cudaErrorInvalidValue;
  }
  struct Entry {
    uint32_t dtype, count;
    size_t src, dst;
  };
  std::vector<Entry> entries;
  size_t dev_bytes = 0;
  for (uint32_t t = 0; t < n; ++t) {
    Entry e{};
    if (!rd(&e.dtype) || !rd(&e.count)) {
      if (error) *error = "truncated blob header";
      return cudaErrorInvalidValue;
    }
    off = align16(off);
    const size_t nb = static_cast<size_t>(e.count) * (e.dtype == 0 ? 2 : e.dtype == 1 ? 4 : 1);
    if (off + nb > size) {
      if (error) *error = "truncated blob payload";
      return cudaErrorInvalidValue;
    }
    e.src = off;
    e.dst = dev_bytes;
    off += nb;
    dev_bytes = align16(dev_bytes + nb);
    entries.push_back(e);
  }
  uint8_t* dev = nullptr;
  cudaError_t err = cudaMalloc(&dev, dev_bytes);
  if (err) return err;
  for (const auto& e : entries) {
    err = cudaMemcpy(dev + e.dst, bytes + e.src,
                     static_cast<size_t>(e.count) * (e.dtype == 0 ? 2 : e.dtype == 1 ? 4 : 1),
                     cudaMemcpyHostToDevice);
    if (err) return err;
  }
  // Fixed order: for stem, down0..2, up0..2: conv1 (w, b), conv2 (w, b), proj (w);
  // then head_w, head_b. Shapes are implied by the topology.
  size_t k = 0;
  auto h16 = [&]() { return reinterpret_cast<const __half*>(dev + entries[k++].dst); };
  auto f32 = [&]() { return reinterpret_cast<const float*>(dev + entries[k++].dst); };
  const int cin_[7] = {16, 32, 64, 128, 256 + 128, 128 + 64, 64 + 32};
  const int cout_[7] = {32, 64, 128, 256, 128, 64, 32};
  ResBlock* blocks[7] = {&out->stem, &out->down[0], &out->down[1], &out->down[2],
                         &out->up[0],  &out->up[1],   &out->up[2]};
  const size_t n_expect = 7 * 5 + 2 + (version == kVersionFp8 ? 14 * 3 : 0);
  if (entries.size() != n_expect) {
    if (error) *error = "unexpected tensor count " + std::to_string(entries.size());
    cudaFree(dev);
    return cudaErrorInvalidValue;
  }
  for (int i = 0; i < 7; ++i) {
    ResBlock& B = *blocks[i];
    B.conv1 = {h16(), f32(), 3, cin_[i], cout_[i]};
    B.conv2 = {h16(), f32(), 3, cout_[i], cout_[i]};
    B.proj = {h16(), nullptr, 1, cin_[i], cout_[i]};
  }
  out->head_w = h16();
  out->head_b = f32();
  if (version == kVersionFp8) {
    for (int i = 0; i < 7; ++i) {
      for (ConvLayer* L : {&blocks[i]->conv1, &blocks[i]->conv2}) {
        const Entry& meta = entries[k];
        float m[2];
        std::memcpy(m, bytes + meta.src, sizeof(m));
        ++k;
        L->w8 = reinterpret_cast<const uint8_t*>(dev + entries[k].dst);
        const bool has8 = entries[k++].count == static_cast<uint32_t>(9 * L->cin * L->cout);
        L->w_scale = f32();
        L->a_scale = m[1];
        L->fp8 = m[0] > 0.0f && has8 && L->cin % 32 == 0 && L->cout % 32 == 0;
      }
    }
  }
  out->device_blob = dev;
  {  // upsample-folding weights of the up blocks' conv1 (derived from the FP16 weights)
    std::vector<std::vector<__half>> ph(3);
    std::vector<std::vector<uint32_t>> edge(3);
    size_t fold_bytes = 0;
    for (int u = 0; u < 3; ++u) {
      const Entry& e = entries[(4 + u) * 5];  // conv1 weights [9][cin][cout]
      std::vector<__half> w(e.count);
      std::memcpy(w.data(), bytes + e.src, e.count * 2);
      const int c0 = cout_[3 + u], cin = cin_[4 + u], co = cout_[4 + u];
      fold_upsample_phases(w.data(), c0, cin, co, ph[u], edge[u]);
      fold_bytes = align16(fold_bytes + ph[u].size() * 2);
      fold_bytes = align16(fold_bytes + edge[u].size() * 4);
    }
    uint8_t* fd = nullptr;
    if ((err = cudaMalloc(&fd, fold_bytes))) return err;
    size_t o = 0;
    for (int u = 0; u < 3; ++u) {
      ConvLayer& L = blocks[4 + u]->conv1;
      cudaMemcpy(fd + o, ph[u].data(), ph[u].size() * 2, cudaMemcpyHostToDevice);
      L.wph = reinterpret_cast<const __half*>(fd + o);
      o = align16(o + ph[u].size() * 2);
      cudaMemcpy(fd + o, edge[u].data(), edge[u].size() * 4, cudaMemcpyHostToDevice);
      L.wedge = reinterpret_cast<const uint32_t*>(fd + o);
      o = align16(o + edge[u].size() * 4);
    }
    out->fold_blob = fd;
  }
  // FP16 accumulation: GFUSED_FM_ACC16 if set (see header), else default_acc16().
  if (const char* env = std::getenv("GFUSED_FM_ACC16")) {
    const int n_env = static_cast<int>(std::strlen(env));
    if (n_env == 1) set_forward_map_acc16(out, -1, env[0] - '0');
    for (int i = 0; n_env > 1 && i < n_env && i < 14; ++i)
      set_forward_map_acc16(out, i, env[i] - '0');
  } else if (const int mode = default_acc16()) {
    set_forward_map_acc16(out, -1, mode);
  }
  return cudaSuccess;
}

bool set_forward_map_acc16(ForwardMapWeights* w, int conv, int mode) {
  if (mode < 0 || mode > 2 || conv < -1 || conv >= 14) return false;
  ResBlock* blocks[7] = {&w->stem, &w->down[0], &w->down[1], &w->down[2],
                         &w->up[0], &w->up[1],   &w->up[2]};
  for (int i = 0; i < 14; ++i) {
    if (conv >= 0 && i != conv) continue;
    ConvLayer& L = i % 2 ? blocks[i / 2]->conv2 : blocks[i / 2]->conv1;
    L.acc16 = mode;
  }
  return true;
}

cudaError_t load_forward_map_weights(const std::string& path, ForwardMapWeights* out,
                                     std::string* error) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    if (error) *error = "cannot open " + path;
    return cudaErrorInvalidValue;
  }
  std::vector<char> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  return load_forward_map_weights_from_memory(buf.data(), buf.size(), out, error);
}

void free_forward_map_weights(ForwardMapWeights* w) {
  if (w->device_blob) cudaFree(w->device_blob);
  if (w->fold_blob) cudaFree(w->fold_blob);
  *w = ForwardMapWeights{};
}

// ---------------------------------------------------------------- workspace

cudaError_t alloc_forward_map_workspace(int H, int W, ForwardMapWorkspace* ws) {
  if (H % 16 || W % 16) return cudaErrorInvalidValue;
  const size_t P = static_cast<size_t>(H) * W;
  const size_t halves[] = {
      P * 32,        // skip0 (stem)
      P / 4 * 64,    // skip1 (down0)
      P / 16 * 128,  // skip2 (down1)
      P / 64 * 256,  // bottom (down2)
      P / 16 * 128,  // up0
      P / 4 * 64,    // up1
      P * 32,        // t_y1 (every block's conv1 out is P*32)
      P / 4 * 32,    // pooled0 = maxpool(skip0)
      P / 16 * 64,   // pooled1 = maxpool(skip1)
      P / 64 * 128,  // pooled2 = maxpool(skip2)
  };
  size_t bytes = 0;
  for (size_t h : halves) bytes = align16(bytes + h * 2);
  bytes = align16(bytes + P * 4);  // logits
  bytes += 16;                     // reduce
  uint8_t* dev = nullptr;
  cudaError_t e = cudaMalloc(&dev, bytes);
  if (e) return e;
  size_t off = 0;
  auto take = [&](size_t nh) {
    auto* p = reinterpret_cast<__half*>(dev + off);
    off = align16(off + nh * 2);
    return p;
  };
  ws->H = H;
  ws->W = W;
  ws->skip[0] = take(halves[0]);
  ws->skip[1] = take(halves[1]);
  ws->skip[2] = take(halves[2]);
  ws->bottom = take(halves[3]);
  ws->up_out[0] = take(halves[4]);
  ws->up_out[1] = take(halves[5]);
  ws->t_y1 = take(halves[6]);
  for (int i = 0; i < 3; ++i) ws->pooled[i] = take(halves[7 + i]);
  ws->logits = reinterpret_cast<float*>(dev + off);
  off = align16(off + P * 4);
  ws->reduce = reinterpret_cast<float*>(dev + off);
  ws->device_blob = dev;
  return cudaSuccess;
}

void free_forward_map_workspace(ForwardMapWorkspace* ws) {
  if (ws->device_blob) cudaFree(ws->device_blob);
  const unsigned flags = ws->flags;
  *ws = ForwardMapWorkspace{};
  ws->flags = flags;
}

// ---------------------------------------------------------------- schedule

cudaError_t run_forward_map(const ForwardMapWeights& w, const float* image_hwc3,
                            ForwardMapWorkspace& ws, const ForwardMapOutputs& out,
                            cudaStream_t s, int H, int W) {
  H = H ? H : ws.H;
  W = W ? W : ws.W;
  if (H <= 0 || W <= 0 || H > ws.H || W > ws.W) return cudaErrorInvalidValue;
  const int P = H * W;
  cudaError_t e;
  // The stem reads the FP32 HWC3 image directly (InMode::kImageF32, 16 virtual channels).
  const __half* img = reinterpret_cast<const __half*>(image_hwc3);

  // pooled[i] = maxpool2x2(skip[i]), written by skip[i]'s producer
  __half* const* pooled = ws.pooled;
  const DefaultEpilogue::Params dp{};
  // encoder
  // Whole-block fusion (conv1 -> conv2 + projection in one kernel, y1 kept in shared
  // memory) pays once the full-resolution activations no longer fit in L2.
  const bool acc16_early = w.stem.conv1.acc16 || w.stem.conv2.acc16 || w.down[0].conv1.acc16 ||
                           w.down[0].conv2.acc16;  // the fused blocks accumulate in FP32
  const bool fuse_blocks = !acc16_early && ((ws.flags & kFwdFusedBlocks) ||
                                            static_cast<long long>(H) * W >= kFusedBlockMinPixels);
  if (fuse_blocks) {
    if ((e = fused_block<InMode::kImageF32, 32>(w.stem, img, 16, nullptr, 0, H, W, ws.skip[0],
                                                 pooled[0], s)))
      return e;
  } else {
    if ((e = enc_block<InMode::kImageF32, TileR32, TileR32, F8R32, F8R32>(
             w.stem, img, 16, H, W, ws, ws.skip[0], pooled[0], s)))
      return e;
  }
  // down0 as one kernel only while it runs FP16 (the FP8 two-kernel path is faster)
  const bool down0_fp8 = runs_fp8(w.down[0].conv1, 32, 0, false, 0, 0, false);
  if (fuse_blocks && !down0_fp8) {
    if ((e = fused_block<InMode::kPlain, 64>(w.down[0], pooled[0], 32, nullptr, 0, H / 2, W / 2,
                                             ws.skip[1], pooled[1], s)))
      return e;
  } else {
    if ((e = enc_block<InMode::kPlain, TileM64, TileM64, F8M64, F8M64>(
             w.down[0], pooled[0], 32, H / 2, W / 2, ws, ws.skip[1], pooled[1], s)))
      return e;
  }
  if ((e = enc_block<InMode::kPlain, TileM64, TileM64, F8M64, F8M64>(
           w.down[1], pooled[1], 64, H / 4, W / 4, ws, ws.skip[2], pooled[2], s)))
    return e;
  if ((e = enc_block<InMode::kPlain, TileL64, TileL64, F8L64, F8M64>(
           w.down[2], pooled[2], 128, H / 8, W / 8, ws, ws.bottom, nullptr, s)))
    return e;
  // decoder
  // Z tiles (1x1, K = c0): L40S sweep at 512 x 768 / 1024 x 1536.
  if ((e = up_block<TileR32, TileM64, F8M64, F8M64, Tile<64, 8, 64, 4, 2>>(
           w.up[0], ws.bottom, 256, ws.skip[2], 128, H / 4, W / 4, ws, ws.up_out[0], dp, 0, s)))
    return e;
  if ((e = up_block<TileU64, TileL64, F8L64, F8M64, Tile<32, 16, 64, 8, 2>>(
           w.up[1], ws.up_out[0], 128, ws.skip[1], 64, H / 2, W / 2, ws, ws.up_out[1], dp, 0, s)))
    return e;

  // last block + heads + density normalization
  int* max_ord = reinterpret_cast<int*>(ws.reduce);
  float* sum = ws.reduce + 1;
  reset_reduce_kernel<<<1, 1, 0, s>>>(ws.reduce);  // graph-capturable, no host copies
  // up2 in row bands once its 32-channel full-resolution tensors no longer fit in L2
  const bool up2_fp8 = runs_fp8(w.up[2].conv1, 64, 32, false, 0, 0, false);
  const bool banded = !(ws.flags & kFwdUnbanded) &&
                      ((ws.flags & kFwdBanded) ||
                       (static_cast<long long>(H) * W >= kBandedMinPixels && !up2_fp8));
  const HeadEpilogue::Params hp{w.head_w, w.head_b, ws.logits, out.log_cov, out.rgb, max_ord};
  if ((e = up_block<TileU32, TileR32, F8R32, F8R32, Tile<32, 16, 32, 8, 1>, HeadEpilogue>(
           w.up[2], ws.up_out[1], 64, ws.skip[0], 32, H, W, ws, nullptr, hp,
           banded ? kBandRows : 0, s)))
    return e;
  if (out.density) {
    exp_sum_kernel<<<256, 256, 0, s>>>(ws.logits, P, max_ord, sum);
    density_kernel<<<(P + 255) / 256, 256, 0, s>>>(ws.logits, P, max_ord, sum, out.density,
                                                   out.rate_sum);
  }
  return cudaGetLastError();
}

cudaError_t crop_forward_map(ForwardMapWorkspace& ws, const float* net_cov, const float* net_rgb,
                             const ForwardMapOutputs& out, float* logits, int H, int W,
                             cudaStream_t s) {
  const int P = H * W;
  int* max_ord = reinterpret_cast<int*>(ws.reduce);
  float* sum = ws.reduce + 1;
  reset_reduce_kernel<<<1, 1, 0, s>>>(ws.reduce);
  crop_maps_kernel<<<grid_for(P), 256, 0, s>>>(ws.logits, net_cov, net_rgb, logits, out.log_cov,
                                               out.rgb, P, W, ws.H * ws.W, ws.W, max_ord);
  exp_sum_kernel<<<256, 256, 0, s>>>(logits, P, max_ord, sum);
  density_kernel<<<grid_for(P), 256, 0, s>>>(logits, P, max_ord, sum, out.density, out.rate_sum);
  return cudaGetLastError();
}

}  // namespace gfused
