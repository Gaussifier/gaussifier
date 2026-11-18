// Correction-head specific fused conv pieces, layered on conv_fused.cuh.
//
//   head_in_conv_kernel  first conv of BOTH U-Nets in one launch. The 14-channel head input
//                        (image 3 | density | log_cov 3 | rgb 3 | rendered 3 | count; the
//                        diff channels are folded into the weights by the exporter) is read
//                        from the pipeline's packed FP16 input (contract C1) or gathered from
//                        its FP32 buffers, in the C1 channel order (stage_head_input). Output
//                        channels [0, 32) go to the main net's tensor, [32, 64) to the aux
//                        net's, each + bias + SiLU.
//   head_enc0_kernel     first conv + enc0.b of one net in one kernel (large images).
//   OutEpilogue          conv_fused epilogue that, instead of storing the conv output,
//                        applies bias + SiLU and the U-Net's final 1x1 conv (C -> 8) per
//                        pixel and writes / accumulates the 8 x H x W planar FP32 output.

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "conv_fused.cuh"
#include "correction_head.cuh"

namespace gfused {
namespace headk {

// MMA / copy helpers shared with the conv core.
using detail::cp_async_commit;
using detail::cp_async_wait_all;
using detail::ldmatrix_x4;
using detail::ldmatrix_x4_trans;
using detail::mma16816;
using detail::silu;
using detail::smem_u32;
using detail::st_shared16;
using detail::swz_off;
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src) {
  detail::cp_async16(dst, src, true);
}
__device__ __forceinline__ void cp_async16_zfill(uint32_t dst, const void* src, bool valid) {
  detail::cp_async16(dst, src, valid);
}

// 4x4 transpose of 32-bit values across the 4 lanes of an MMA quad (lanes 4g .. 4g+3, t =
// lane & 3): on entry lane t holds v[j] = channels (8j + 2t, 8j + 2t + 1) of one pixel (the
// accumulator fragment layout of four n8 tiles); on exit it holds channels 8t .. 8t + 7 in
// order, so the quad writes the pixel's 64 bytes as four coalesced 16-byte stores.
__device__ __forceinline__ void quad_transpose(uint32_t (&v)[4], int t) {
  const bool b1 = t & 2, b0 = t & 1;
  uint32_t x0 = b1 ? v[0] : v[2], x1 = b1 ? v[1] : v[3];
  x0 = __shfl_xor_sync(0xffffffffu, x0, 2);
  x1 = __shfl_xor_sync(0xffffffffu, x1, 2);
  if (b1) {
    v[0] = x0, v[1] = x1;
  } else {
    v[2] = x0, v[3] = x1;
  }
  x0 = b0 ? v[0] : v[1];
  x1 = b0 ? v[2] : v[3];
  x0 = __shfl_xor_sync(0xffffffffu, x0, 1);
  x1 = __shfl_xor_sync(0xffffffffu, x1, 1);
  if (b0) {
    v[0] = x0, v[2] = x1;
  } else {
    v[1] = x0, v[3] = x1;
  }
}
__device__ __forceinline__ uint32_t silu_pack(float a, float b) {
  const __half2 h = __floats2half2_rn(a / (1.0f + __expf(-a)), b / (1.0f + __expf(-b)));
  return *reinterpret_cast<const uint32_t*>(&h);
}
// Stores 32 channels of one pixel held in fragment layout (see quad_transpose) to dst
// (the pixel's first channel); every lane of the warp must call it (ok masks the store).
__device__ __forceinline__ void store_quad32(__half* dst, uint32_t (&v)[4], int t, bool ok) {
  quad_transpose(v, t);
  if (ok) *reinterpret_cast<uint4*>(dst + 8 * t) = make_uint4(v[0], v[1], v[2], v[3]);
}

// ------------------------------------------------------------------ first conv (both nets)

// The head input as two 16-byte units per pixel (contract C1 order; the first conv's weight
// rows are permuted to it at load):
//   unit 0: image r, g, b | density | log_cov 0, 1, 2 | rgb 0
//   unit 1: rendered r, g, b | count | rgb 1, rgb 2 | 0, 0
// Stages halo pixel (y, x) into smem units (s0, s1): from the packed [2][H][W][8] FP16 input
// when present (cp.async; unit 1's halves 0..3 re-gathered from FP32 unless dynamic_in_packed),
// else gathered from the FP32 buffers. Zero outside the image.
__device__ __forceinline__ void stage_head_input(const CorrectionHeadInputs& in, int H, int W,
                                                 int y, int x, uint32_t s0, uint32_t s1) {
  const bool ok = y >= 0 && y < H && x >= 0 && x < W;
  const int P = H * W, i = ok ? y * W + x : 0;
  if (in.packed16) {
    const uint4* pk = reinterpret_cast<const uint4*>(in.packed16);
    cp_async16_zfill(s0, pk + i, ok);
    if (in.dynamic_in_packed || !ok) {
      cp_async16_zfill(s1, pk + P + i, ok);
      return;
    }
    uint4 u = __ldg(pk + P + i);
    __half2* hu = reinterpret_cast<__half2*>(&u);
    hu[0] = __floats2half2_rn(__ldg(in.rendered + i), __ldg(in.rendered + P + i));
    hu[1] = __floats2half2_rn(__ldg(in.rendered + 2 * P + i), __ldg(in.count + i));
    st_shared16(s1, u);
    return;
  }
  __align__(16) __half v[16];
#pragma unroll
  for (int c = 0; c < 16; ++c) v[c] = __float2half(0.0f);
  if (ok) {
    v[0] = __float2half(__ldg(in.image_hwc3 + 3 * i));
    v[1] = __float2half(__ldg(in.image_hwc3 + 3 * i + 1));
    v[2] = __float2half(__ldg(in.image_hwc3 + 3 * i + 2));
    v[3] = __float2half(__ldg(in.density + i));
#pragma unroll
    for (int c = 0; c < 3; ++c) {
      v[4 + c] = __float2half(__ldg(in.log_cov + c * P + i));
      v[8 + c] = __float2half(__ldg(in.rendered + c * P + i));
    }
    v[7] = __float2half(__ldg(in.rgb + i));
    v[11] = __float2half(__ldg(in.count + i));
    v[12] = __float2half(__ldg(in.rgb + P + i));
    v[13] = __float2half(__ldg(in.rgb + 2 * P + i));
  }
  st_shared16(s0, reinterpret_cast<uint4*>(v)[0]);
  st_shared16(s1, reinterpret_cast<uint4*>(v)[1]);
}

constexpr int kInCh = 16;       // 14 real + 2 zero
constexpr int kFirstCout = 64;  // main 32 | aux 32 (aux real 24, zero padded)

// MMAs + epilogue of one warp for one tile: rows wm * WR .. + WR of the TH x 16 tile, the NJ
// n8 tiles of one net (NJ = 4: 32 channels; NJ = 3: the aux net at native width 24, whose
// zero-padded columns 24..31 are neither computed nor stored). Taps are accumulated in order
// 0..8, one m16n8k16 each, so the result does not depend on NJ.
// s_a: halo tile as two unit planes [2][HH * HW][16 B] (unswizzled: 8 consecutive pixels are 128
// contiguous bytes, conflict free), so every A address is a lane base + an immediate offset.
// s_b: weights [9 * 16 rows][8 chunks] swizzled (chunk ^ row & 7); the rows a lane reads in tap
// `tap` are tap * 16 + (lane & 15), so the swizzle is tap invariant and B addresses are a lane
// base + tap * 2048 as well.
template <int NJ, int WR, int HW, int kPlane>
__device__ __forceinline__ void head_in_warp_tile(uint32_t s_a, uint32_t s_b, int wm, int nb,
                                                  const float* __restrict__ bias, __half* dst,
                                                  int oc, int H, int W, int y0, int x0) {
  const int lane = threadIdx.x & 31;
  float acc[WR][NJ][4];
#pragma unroll
  for (int r = 0; r < WR; ++r)
#pragma unroll
    for (int j = 0; j < NJ; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[r][j][e] = 0.0f;
  const int l15 = lane & 15, sw = lane & 7;
  const uint32_t a0 = s_a + (lane >> 4) * kPlane + ((wm * WR) * HW + l15) * 16;
  const uint32_t bq0 = s_b + (l15 * 8 + ((nb + (lane >> 4)) ^ sw)) * 16;
  const uint32_t bq1 = s_b + (l15 * 8 + ((nb + 2 + (lane >> 4)) ^ sw)) * 16;
  const uint32_t bq2 = s_b + (l15 * 8 + ((nb + 2) ^ sw)) * 16;  // x2: lanes 0..15
#pragma unroll
  for (int tap = 0; tap < 9; ++tap) {
    const int ky = tap / 3, kx = tap % 3;
    uint32_t b[NJ][2];
    {
      uint32_t r[4];
      ldmatrix_x4_trans(bq0 + tap * 2048, r);
      b[0][0] = r[0], b[0][1] = r[1], b[1][0] = r[2], b[1][1] = r[3];
    }
    if constexpr (NJ == 4) {
      uint32_t r[4];
      ldmatrix_x4_trans(bq1 + tap * 2048, r);
      b[2][0] = r[0], b[2][1] = r[1], b[3][0] = r[2], b[3][1] = r[3];
    } else {
      uint32_t r[2];
      detail::ldmatrix_x2_trans(bq2 + tap * 2048, r);
      b[2][0] = r[0], b[2][1] = r[1];
    }
#pragma unroll
    for (int r = 0; r < WR; ++r) {
      uint32_t a[4];
      ldmatrix_x4(a0 + ((r + ky) * HW + kx) * 16, a);
#pragma unroll
      for (int j = 0; j < NJ; ++j) mma16816(acc[r][j], a, b[j][0], b[j][1]);
    }
  }
  // epilogue straight from the fragments: bias + SiLU, quad transpose, 16-byte stores
  const int g = lane >> 2, t = lane & 3;
  float b0[NJ], b1[NJ];
#pragma unroll
  for (int j = 0; j < NJ; ++j) {
    b0[j] = __ldg(bias + (nb + j) * 8 + 2 * t);
    b1[j] = __ldg(bias + (nb + j) * 8 + 2 * t + 1);
  }
#pragma unroll
  for (int r = 0; r < WR; ++r) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int y = y0 + r, x = x0 + g + 8 * h;
      const size_t px = static_cast<size_t>(y) * W + x;
      uint32_t v[4];
#pragma unroll
      for (int j = 0; j < 4; ++j)
        v[j] = j < NJ ? silu_pack(acc[r][j < NJ ? j : 0][2 * h] + b0[j < NJ ? j : 0],
                                  acc[r][j < NJ ? j : 0][2 * h + 1] + b1[j < NJ ? j : 0])
                      : 0u;
      store_quad32(dst + px * oc, v, t, 8 * t < oc && y < H && x < W);  // partial edge tiles
    }
  }
}

// Block: TH x 16 output pixels, WM x 2 warps: warp column 0 computes the main net's 32
// channels, column 1 the aux net's (AUX8 n8 tiles: 3 at native width 24, 4 when padded to 32).
// Persistent: weights resident, input halo double buffered across the block's tiles.
template <int TH, int WM, int AUX8, bool kPacked>
__global__ void __launch_bounds__(WM * 64, 2)
    head_in_conv_kernel(CorrectionHeadInputs in, int H, int W, const __half* __restrict__ wgt,
                        const float* __restrict__ bias, __half* __restrict__ out_main,
                        __half* __restrict__ out_aux, int aux_c) {
  constexpr int NT = WM * 64;
  constexpr int HH = TH + 2, HW = 18;
  constexpr int BN = kFirstCout, CB = BN / 8;
  constexpr int WR = TH / WM;
  constexpr int kPlane = HH * HW * 16;  // one 16-byte unit per halo pixel
  constexpr int kA = 2 * kPlane, kB = 9 * kInCh * BN * 2;
  static_assert(TH % WM == 0 && (AUX8 == 3 || AUX8 == 4) && CB == 8, "tile config");
  __shared__ __align__(128) unsigned char smem[2 * kA + kB];
  const uint32_t s_b = smem_u32(smem) + 2 * kA;

  // partial edge tiles (any H, W): staging zero-fills, stores are masked
  const int tiles_x = (W + 15) / 16, n_tiles = ((H + TH - 1) / TH) * tiles_x;

  // weights [9][16][64] -> swizzled smem (async, with the first tile's input)
  for (int u = threadIdx.x; u < 9 * kInCh * CB; u += NT) {
    const int row = u / CB, c8 = u % CB;
    cp_async16(s_b + swz_off<CB>(row, c8), wgt + row * BN + c8 * 8);
  }
  // stage the 14-channel halo tile (zero outside the image)
  auto stage = [&](int tile, uint32_t s_a) {
    const int ty0 = (tile / tiles_x) * TH, tx0 = (tile % tiles_x) * 16;
    for (int pix = threadIdx.x; pix < HH * HW; pix += NT) {
      const int y = ty0 + pix / HW - 1, x = tx0 + pix % HW - 1;
      const uint32_t d = s_a + pix * 16;
      if constexpr (kPacked) {  // both units straight from the packed planes
        const bool ok = y >= 0 && y < H && x >= 0 && x < W;
        const uint4* pk = reinterpret_cast<const uint4*>(in.packed16) + (ok ? y * W + x : 0);
        cp_async16_zfill(d, pk, ok);
        cp_async16_zfill(d + kPlane, pk + H * W, ok);
      } else {
        stage_head_input(in, H, W, y, x, d, d + kPlane);
      }
    }
  };
  stage(blockIdx.x, smem_u32(smem));
  cp_async_commit();

  const int warp = threadIdx.x >> 5;
  const int wm = warp % WM, net = warp / WM;  // warp-uniform
  const int nb = net * 4;                     // first n8 tile (weight chunk) of this warp's net
  __half* dst = net ? out_aux : out_main;
  const int oc = net ? aux_c : 32;  // output channel stride (aux native: 24)
  int buf = 0;
  for (int tile = blockIdx.x; tile < n_tiles; tile += gridDim.x, buf ^= 1) {
    const int ty0 = (tile / tiles_x) * TH, tx0 = (tile % tiles_x) * 16;
    const uint32_t s_a = smem_u32(smem) + buf * kA;
    cp_async_wait_all();
    __syncthreads();  // this tile staged; the other buffer's previous tile is fully consumed
    if (tile + static_cast<int>(gridDim.x) < n_tiles)
      stage(tile + gridDim.x, smem_u32(smem) + (buf ^ 1) * kA);
    cp_async_commit();
    if (net == 0 || AUX8 == 4) {
      head_in_warp_tile<4, WR, HW, kPlane>(s_a, s_b, wm, nb, bias, dst, oc, H, W,
                                           ty0 + wm * WR, tx0);
    } else {
      head_in_warp_tile<AUX8, WR, HW, kPlane>(s_a, s_b, wm, nb, bias, dst, oc, H, W,
                                              ty0 + wm * WR, tx0);
    }
  }
}

// AUX8 is picked from aux_c (24: native, 3 n8 tiles; 32: padded), kPacked from the input
// (packed planes with the per-state unit valid: cp.async only; else stage_head_input).
template <int TH, int WM>
inline cudaError_t launch_head_in(const CorrectionHeadInputs& in, int H, int W, const __half* w,
                                  const float* b, __half* out_main, __half* out_aux, int aux_c,
                                  cudaStream_t s) {
  const bool packed = in.packed16 && in.dynamic_in_packed, native = aux_c == 24;
  auto kern = native ? (packed ? head_in_conv_kernel<TH, WM, 3, true>
                               : head_in_conv_kernel<TH, WM, 3, false>)
                     : (packed ? head_in_conv_kernel<TH, WM, 4, true>
                               : head_in_conv_kernel<TH, WM, 4, false>);
  int res = 0;  // persistent grid size, one device-aware cache per instantiation
  const auto occupancy = [kern](int device, int& value) {
    return detail::resident_blocks(kern, device, WM * 64, 0, value);
  };
  const int slot = native * 2 + packed;
  const auto status = slot == 0   ? detail::device_launch_value<0>(res, occupancy)
                      : slot == 1 ? detail::device_launch_value<1>(res, occupancy)
                      : slot == 2 ? detail::device_launch_value<2>(res, occupancy)
                                  : detail::device_launch_value<3>(res, occupancy);
  if (status != cudaSuccess) return status;
  const int tiles = ((H + TH - 1) / TH) * ((W + 15) / 16);
  kern<<<tiles < res ? tiles : res, WM * 64, 0, s>>>(in, H, W, w, b, out_main, out_aux, aux_c);
  return cudaGetLastError();
}

// ------------------------------------------------------------------ fused enc0 double conv

// Both level-0 encoder convs of one net in one kernel: X (14-ch gathered input) -> conv3x3
// -> bias + SiLU -> conv3x3 -> bias + SiLU -> NHWC FP16 [H, W, 32]. The first conv is
// evaluated on the (TH+2) x 18 halo region of the output tile and kept in shared memory, so
// the intermediate feature map never touches global memory (the halo recompute costs
// ~1.4x of a K=16 conv). blockIdx.y selects the net (0 main, 1 aux); both nets read the same
// inputs and the first conv's weights are the 32-column halves of the shared [9][16][64].
template <int TH, int WARPS>
struct Enc0Cfg {
  static constexpr int NT = WARPS * 32;
  static constexpr int IH = TH + 4, IW = 20;                 // input halo
  static constexpr int MH = TH + 2, MW = 18;                 // mid halo
  static constexpr int M1 = MH * MW, MT1 = (M1 + 15) / 16;   // phase-1 rows / m16 tiles
  static constexpr int kIn = IH * IW * 16 * 2;               // [pix][16] fp16
  static constexpr int kMid = MT1 * 16 * 32 * 2;             // [pix][32] fp16
  static constexpr int kW0 = 9 * 16 * 32 * 2;
  static constexpr int kW1 = 9 * 32 * 32 * 2;
  static constexpr int kSmem = kIn + kMid + kW0 + kW1;
  static_assert(TH % WARPS == 0 || WARPS % TH == 0, "tile rows vs warps");
};

template <int TH, int WARPS>
__global__ void __launch_bounds__(WARPS * 32)
    head_enc0_kernel(CorrectionHeadInputs in, int H, int W, const __half* __restrict__ w0,
                     const float* __restrict__ b0, const __half* __restrict__ w1_main,
                     const float* __restrict__ b1_main, const __half* __restrict__ w1_aux,
                     const float* __restrict__ b1_aux, __half* __restrict__ out_main,
                     __half* __restrict__ out_aux, int aux_c) {
  using C = Enc0Cfg<TH, WARPS>;
  constexpr int NT = C::NT;
  extern __shared__ __align__(128) unsigned char smem[];
  const uint32_t s_in = smem_u32(smem), s_mid = s_in + C::kIn, s_w0 = s_mid + C::kMid,
                 s_w1 = s_w0 + C::kW0;
  const int net = blockIdx.y;
  const __half* w1 = net ? w1_aux : w1_main;
  const float* b1 = net ? b1_aux : b1_main;
  __half* out = net ? out_aux : out_main;
  const int oc = net ? aux_c : 32;  // output channel stride (aux native: 24 of the 32 padded)

  // partial edge tiles (any H, W): staging zero-fills, stores are masked
  const int tiles_x = (W + 15) / 16, n_tiles = ((H + TH - 1) / TH) * tiles_x;

  // weights (async): w0 columns [net*32, net*32+32) of [9*16][64]; w1 [9*32][32]
  for (int u = threadIdx.x; u < 9 * 16 * 4; u += NT) {
    const int row = u / 4, c8 = u % 4;
    cp_async16(s_w0 + swz_off<4>(row, c8), w0 + row * 64 + net * 32 + c8 * 8);
  }
  for (int u = threadIdx.x; u < 9 * 32 * 4; u += NT) {
    const int row = u / 4, c8 = u % 4;
    cp_async16(s_w1 + swz_off<4>(row, c8), w1 + row * 32 + c8 * 8);
  }
  // persistent over tiles: the weights stay resident in shared memory
  for (int tile = blockIdx.x; tile < n_tiles; tile += gridDim.x) {
  const int ty0 = (tile / tiles_x) * TH, tx0 = (tile % tiles_x) * 16;
  __syncthreads();  // previous tile's phase 2 is done with s_mid / s_in
  // gather the 14-channel input halo
  for (int pix = threadIdx.x; pix < C::IH * C::IW; pix += NT) {
    const int y = ty0 + pix / C::IW - 2, x = tx0 + pix % C::IW - 2;
    stage_head_input(in, H, W, y, x, s_in + swz_off<2>(pix, 0), s_in + swz_off<2>(pix, 1));
  }
  cp_async_commit();
  cp_async_wait_all();
  __syncthreads();

  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int a_row = (lane & 7) + ((lane >> 3) & 1) * 8, a_kc = lane >> 4;
  const int b_k = a_row, b_nc = lane >> 4;
  const int g = lane >> 2, t = lane & 3;

  // phase 1: mid = SiLU(conv(X) + b0) on the halo region; zero outside the image
  for (int mt = warp; mt < C::MT1; mt += WARPS) {
    float acc[4][4] = {};
    int q = mt * 16 + a_row;
    if (q >= C::M1) q = 0;
    const int qy = q / C::MW, qx = q % C::MW;
#pragma unroll
    for (int tap = 0; tap < 9; ++tap) {
      const int ky = tap / 3, kx = tap % 3;
      uint32_t a[4], bl[4], bh[4];
      ldmatrix_x4(s_in + swz_off<2>((qy + ky) * C::IW + qx + kx, a_kc), a);
      ldmatrix_x4_trans(s_w0 + swz_off<4>(tap * 16 + b_k, 0 + b_nc), bl);
      ldmatrix_x4_trans(s_w0 + swz_off<4>(tap * 16 + b_k, 2 + b_nc), bh);
      mma16816(acc[0], a, bl[0], bl[1]);
      mma16816(acc[1], a, bl[2], bl[3]);
      mma16816(acc[2], a, bh[0], bh[1]);
      mma16816(acc[3], a, bh[2], bh[3]);
    }
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int r = mt * 16 + g + 8 * h;
      if (r >= C::M1) continue;
      const int y = ty0 - 1 + r / C::MW, x = tx0 - 1 + r % C::MW;
      const bool inside = y >= 0 && y < H && x >= 0 && x < W;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int n = j * 8 + 2 * t;
        const float bb0 = __ldg(b0 + net * 32 + n), bb1 = __ldg(b0 + net * 32 + n + 1);
        const __half2 v = inside ? __floats2half2_rn(silu(acc[j][2 * h] + bb0),
                                                     silu(acc[j][2 * h + 1] + bb1))
                                 : __floats2half2_rn(0.0f, 0.0f);
        asm volatile("st.shared.b32 [%0], %1;\n" ::"r"(s_mid + swz_off<4>(r, j) + 4 * t),
                     "r"(*reinterpret_cast<const uint32_t*>(&v)));
      }
    }
  }
  __syncthreads();

  // phase 2: out = SiLU(conv(mid) + b1) on the TH x 16 tile; warp owns TH / WARPS rows
  constexpr int WR = TH >= WARPS ? TH / WARPS : 1;
  if (warp < TH) {
    float acc[WR][4][4] = {};
#pragma unroll
    for (int tap = 0; tap < 9; ++tap) {
      const int ky = tap / 3, kx = tap % 3;
#pragma unroll
      for (int s = 0; s < 2; ++s) {
        uint32_t bl[4], bh[4];
        ldmatrix_x4_trans(s_w1 + swz_off<4>(tap * 32 + s * 16 + b_k, 0 + b_nc), bl);
        ldmatrix_x4_trans(s_w1 + swz_off<4>(tap * 32 + s * 16 + b_k, 2 + b_nc), bh);
#pragma unroll
        for (int r = 0; r < WR; ++r) {
          const int row = warp * WR + r;
          uint32_t a[4];
          ldmatrix_x4(s_mid + swz_off<4>((row + ky) * C::MW + a_row + kx, s * 2 + a_kc), a);
          mma16816(acc[r][0], a, bl[0], bl[1]);
          mma16816(acc[r][1], a, bl[2], bl[3]);
          mma16816(acc[r][2], a, bh[0], bh[1]);
          mma16816(acc[r][3], a, bh[2], bh[3]);
        }
      }
    }
    float bb0[4], bb1[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      bb0[j] = __ldg(b1 + j * 8 + 2 * t);
      bb1[j] = __ldg(b1 + j * 8 + 2 * t + 1);
    }
#pragma unroll
    for (int r = 0; r < WR; ++r) {
      const int y = ty0 + warp * WR + r;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int x = tx0 + g + 8 * h;
        uint32_t v[4];
#pragma unroll
        for (int j = 0; j < 4; ++j)
          v[j] = silu_pack(acc[r][j][2 * h] + bb0[j], acc[r][j][2 * h + 1] + bb1[j]);
        store_quad32(out + (static_cast<size_t>(y) * W + x) * oc, v, t,
                     8 * t < oc && y < H && x < W);
      }
    }
  }
  }  // tile loop
}

template <int TH, int WARPS>
inline cudaError_t launch_head_enc0(const CorrectionHeadInputs& in, int H, int W,
                                    const __half* w0, const float* b0, const __half* w1m,
                                    const float* b1m, const __half* w1a, const float* b1a,
                                    __half* out_main, __half* out_aux, int aux_c,
                                    cudaStream_t s) {
  using C = Enc0Cfg<TH, WARPS>;
  auto kern = head_enc0_kernel<TH, WARPS>;
  int blocks_per_net = 0;
  const auto status = detail::device_launch_value(blocks_per_net, [kern](int device, int& value) {
    return detail::resident_blocks(kern, device, C::NT, C::kSmem, value);
  });
  if (status != cudaSuccess) return status;
  blocks_per_net = (blocks_per_net + 1) / 2;
  const int tiles = ((H + TH - 1) / TH) * ((W + 15) / 16);
  dim3 grid(tiles < blocks_per_net ? tiles : blocks_per_net, 2);
  kern<<<grid, C::NT, C::kSmem, s>>>(in, H, W, w0, b0, w1m, b1m, w1a, b1a, out_main, out_aux,
                                     aux_c);
  return cudaGetLastError();
}

// ------------------------------------------------------------------ fused 1x1 out epilogue

// Requires the block to own every output channel (BN == cout, n0 == 0).
struct OutEpilogue {
  struct Params {
    const float* w;  // [cout][8] FP32 (final 1x1 conv)
    const float* b;  // [8] FP32 (final bias; for accumulate, the caller's other half adds its own)
    float* out;      // 8 x H x W planar FP32
    int accumulate;  // out += result instead of out = result
  };
  template <int TH, int BN, int NT>
  __device__ __forceinline__ static void apply(const float* tile, int ld, int ty0, int tx0, int,
                                               const ConvParams& p, const Params& e) {
    const size_t P = static_cast<size_t>(p.H) * p.W;
    for (int pix = threadIdx.x; pix < TH * 16; pix += NT) {
      const int y = ty0 + pix / 16, x = tx0 + pix % 16;
      if (y >= p.H || x >= p.W) continue;  // partial edge tiles
      float acc[8];
#pragma unroll
      for (int o = 0; o < 8; ++o) acc[o] = __ldg(e.b + o);
      const float* row = tile + pix * ld;
      // float4 tile reads (ld = BN + 4: conflict-free per quarter warp); the weight and bias
      // addresses are warp-uniform, so the __ldg loads broadcast.
#pragma unroll 2
      for (int c4 = 0; c4 < BN; c4 += 4) {
        const float4 t4 = *reinterpret_cast<const float4*>(row + c4);
        const float4 b4 = __ldg(reinterpret_cast<const float4*>(p.bias + c4));
        const float v[4] = {silu(t4.x + b4.x), silu(t4.y + b4.y), silu(t4.z + b4.z),
                            silu(t4.w + b4.w)};
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const float4 w0 = __ldg(reinterpret_cast<const float4*>(e.w + (c4 + q) * 8));
          const float4 w1 = __ldg(reinterpret_cast<const float4*>(e.w + (c4 + q) * 8 + 4));
          acc[0] += v[q] * w0.x; acc[1] += v[q] * w0.y; acc[2] += v[q] * w0.z;
          acc[3] += v[q] * w0.w; acc[4] += v[q] * w1.x; acc[5] += v[q] * w1.y;
          acc[6] += v[q] * w1.z; acc[7] += v[q] * w1.w;
        }
      }
      const size_t px = static_cast<size_t>(y) * p.W + x;
      float* dst = e.out + px;
#pragma unroll
      for (int o = 0; o < 8; ++o) {
        if (e.accumulate) acc[o] += dst[o * P];
        dst[o * P] = acc[o];
      }
    }
  }
};

}  // namespace headk
}  // namespace gfused
