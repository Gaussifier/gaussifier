// Lean persistent full-resolution kernels of the correction head (one U-Net per launch).
//
//   head_fullres_kernel<C, TH, WM, kOut = false>   "pair": a plain 3x3 conv C -> C + bias +
//       SiLU, used for enc0.b. Optionally also writes the 2x2-max-pooled output, so the next
//       layer (enc1.a) reads a 4x smaller tensor through plain staging instead of pooling on
//       the fly.
//   head_fullres_kernel<C, TH, WM, kOut = true>    "out": dec1.b + SiLU + the final 1x1 conv
//       (C -> 8) for one pixel tile, plus the out bias, written (or accumulated) as the planar
//       8 x H x W FP32 correction. The C-channel feature map is never stored.
//
// Structure (vs conv_fused): the 3x3 weights stay resident in shared memory; the block walks
// a static-stride list of TH x 16 pixel tiles and pipelines 16-channel halo chunks through NS
// buffers across tile boundaries (one barrier per chunk); staging descriptors are computed
// once per block; the epilogue works on the accumulator fragments (no shared-memory FP32
// tile). The out kernel's 1x1 conv is an m16n8k16 MMA whose A operand is the conv
// accumulator fragment itself (FP16-rounded post-SiLU activations, as a PyTorch FP16 head
// computes them).
//
// Requirements: W a multiple of 16, H a multiple of 2 (partial row tiles are masked), C a
// multiple of 16, 16-byte aligned NHWC tensors.

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "head_conv.cuh"

namespace gfused {
namespace headk {

struct FullresNet {
  const __half* in = nullptr;    // NHWC [H][W][C]
  const __half* w = nullptr;     // [9][C][C]
  const float* b = nullptr;      // [C]
  __half* out = nullptr;         // pair: NHWC [H][W][C]
  __half* pooled = nullptr;      // pair (optional): NHWC [H/2][W/2][C], 2x2 max of out
  const float* w_out = nullptr;  // out kernel: [C][8] FP32 final 1x1 weights
  const float* b_out = nullptr;  // out kernel: [8]
};

struct FullresParams {
  FullresNet net;
  int H = 0, W = 0;
  float* out = nullptr;  // out kernel: 8 x H x W planar
  int accumulate = 0;    // out kernel: out += result
};

__device__ __forceinline__ uint32_t pack_half2(float a, float b) {
  const __half2 h = __floats2half2_rn(a, b);
  return *reinterpret_cast<const uint32_t*>(&h);
}

template <int C, int TH, int WM, int NS = 2>
struct FullresCfg {
  static constexpr int NT = WM * 32;
  static constexpr int WR = TH / WM;           // tile rows per warp
  static constexpr int NT8 = C / 8;            // n8 tiles per warp (all channels)
  static constexpr int KC = C / 16;            // 16-channel chunks per tile
  static constexpr int HP = (TH + 2) * 18;     // halo pixels
  static constexpr int UNITS = HP * 2;         // 16-byte units per chunk
  static constexpr int UPT = (UNITS + NT - 1) / NT;
  static constexpr int kA = HP * 32;           // bytes per chunk buffer
  static constexpr int kW = 9 * C * C * 2;     // bytes of the resident weights
  static constexpr int kSmem = kW + NS * kA;   // NS-stage chunk pipeline
  static_assert(NS >= 2 && NS <= 4, "2..4 stages");
  static constexpr int kMinBlocks = 2 * kSmem <= 98 * 1024 && NT <= 256 ? 2 : 1;
  static_assert(C % 16 == 0 && C <= 64, "C must be 16, 32, 48 or 64");
  static_assert(TH % WM == 0, "tile rows must split across warps");
};

template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

template <int C, int TH, int WM, bool kOut, bool kPool, int NS>
__global__ void __launch_bounds__(WM * 32, (FullresCfg<C, TH, WM, NS>::kMinBlocks))
    head_fullres_kernel(FullresParams p) {
  using Cfg = FullresCfg<C, TH, WM, NS>;
  constexpr int NT = Cfg::NT, WR = Cfg::WR, NT8 = Cfg::NT8, KC = Cfg::KC;
  constexpr int CB = C / 8;  // 16-byte chunks per weight row
  static_assert(!kPool || (!kOut && WR % 2 == 0), "pooling needs an even number of warp rows");
  extern __shared__ __align__(128) unsigned char smem[];
  const uint32_t s_w = smem_u32(smem), s_a = s_w + Cfg::kW;
  const FullresNet& N = p.net;

  const int H = p.H, W = p.W;
  const int tiles_x = W / 16, n_tiles = ((H + TH - 1) / TH) * tiles_x;
  if (static_cast<int>(blockIdx.x) >= n_tiles) return;

  // resident weights: [9 * C rows][CB chunks], swizzled
  for (int u = threadIdx.x; u < 9 * C * CB; u += NT) {
    const int row = u / CB, c8 = u % CB;
    cp_async16(s_w + swz_off<CB>(row, c8), N.w + row * C + c8 * 8);
  }

  // staging descriptors (tile relative): halo pixel (dy, dx), 8-channel half c of the chunk
  int goff[Cfg::UPT], dyx[Cfg::UPT];
  uint32_t soff[Cfg::UPT];
#pragma unroll
  for (int k = 0; k < Cfg::UPT; ++k) {
    const int u = threadIdx.x + k * NT;
    const int pix = u >> 1, c = u & 1;
    const int dy = pix / 18, dx = pix % 18;
    goff[k] = ((dy - 1) * W + (dx - 1)) * C + c * 8;
    dyx[k] = u < Cfg::UNITS ? (dy << 8) | dx : -1;
    soff[k] = swz_off<2>(pix, c);
  }
  auto issue = [&](int tile, int ch, uint32_t buf) {
    const __half* src = N.in + ch * 16;
    const int ty0 = (tile / tiles_x) * TH, tx0 = (tile % tiles_x) * 16;
    const __half* base = src + (static_cast<size_t>(ty0) * W + tx0) * C;
    const bool interior = ty0 > 0 && ty0 + TH < H && tx0 > 0 && tx0 + 16 < W;
#pragma unroll
    for (int k = 0; k < Cfg::UPT; ++k) {
      if (Cfg::UNITS % NT == 0 || dyx[k] >= 0) {
        bool ok = true;
        if (!interior) {
          const int y = ty0 + (dyx[k] >> 8) - 1, x = tx0 + (dyx[k] & 255) - 1;
          ok = y >= 0 && y < H && x >= 0 && x < W;
        }
        cp_async16_zfill(buf + soff[k], ok ? base + goff[k] : src, ok);
      }
    }
  };

  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int a_row = (lane & 7) + ((lane >> 3) & 1) * 8, a_kc = lane >> 4;
  const int b_k = a_row, b_nc = lane >> 4;
  // C channels of one pixel from fragment layout v[j] = channels (8j + 2t, +1); all lanes call
  auto store_c = [&](__half* dst, uint32_t (&v)[NT8], bool ok) {
    if constexpr (NT8 == 4) {
      store_quad32(dst, v, t, ok);
    } else {
#pragma unroll
      for (int j = 0; j < NT8; ++j)
        if (ok) *reinterpret_cast<uint32_t*>(dst + 8 * j + 2 * t) = v[j];
    }
  };

  // out kernel: B fragments of the 1x1 weights (k = channel, n = output channel g)
  uint32_t bo[KC][2];
  float out_bias0 = 0.0f, out_bias1 = 0.0f;
  if constexpr (kOut) {
#pragma unroll
    for (int s = 0; s < KC; ++s) {
      const float* wo = N.w_out;
      const int k0 = s * 16 + 2 * t;
      bo[s][0] = pack_half2(__ldg(wo + k0 * 8 + g), __ldg(wo + (k0 + 1) * 8 + g));
      bo[s][1] = pack_half2(__ldg(wo + (k0 + 8) * 8 + g), __ldg(wo + (k0 + 9) * 8 + g));
    }
    out_bias0 = __ldg(N.b_out + 2 * t);
    out_bias1 = __ldg(N.b_out + 2 * t + 1);
  }

  float acc[WR][NT8][4];
  float oacc[kOut ? WR : 1][4];
  auto zero_acc = [&]() {
#pragma unroll
    for (int r = 0; r < WR; ++r)
#pragma unroll
      for (int j = 0; j < NT8; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[r][j][e] = 0.0f;
  };
  zero_acc();
  if constexpr (kOut) {
#pragma unroll
    for (int r = 0; r < WR; ++r)
#pragma unroll
      for (int e = 0; e < 4; ++e) oacc[r][e] = 0.0f;
  }

  // NS-stage pipeline over the block's chunk sequence (tile-major, KC chunks per tile): chunk
  // q + NS - 1 is issued while chunk q is computed; one commit group per iteration.
  int iss_tile = blockIdx.x, iss_c = 0;
  auto issue_next = [&](int slot) {
    if (iss_tile < n_tiles) issue(iss_tile, iss_c, s_a + slot * Cfg::kA);
    if (++iss_c == KC) {
      iss_c = 0;
      iss_tile += gridDim.x;
    }
  };
#pragma unroll
  for (int st = 0; st < NS - 1; ++st) {
    issue_next(st);
    cp_async_commit();
  }
  int slot = 0;
  for (int tile = blockIdx.x; tile < n_tiles; tile += gridDim.x) {
    const int ty0 = (tile / tiles_x) * TH, tx0 = (tile % tiles_x) * 16;
#pragma unroll
    for (int ch = 0; ch < KC; ++ch) {
      cp_async_wait_group<NS - 2>();
      __syncthreads();
      issue_next(slot == 0 ? NS - 1 : slot - 1);
      cp_async_commit();

      // MMAs of chunk ch: A halo [(TH+2) x 18][16], B rows (tap * C + ch * 16 + k)
      const uint32_t sa = s_a + slot * Cfg::kA;
      slot = slot == NS - 1 ? 0 : slot + 1;
#pragma unroll
      for (int kx = 0; kx < 3; ++kx) {
        uint32_t b[3][NT8][2];
#pragma unroll
        for (int ky = 0; ky < 3; ++ky)
#pragma unroll
          for (int j = 0; j < NT8; j += 2) {
            uint32_t r[4];
            ldmatrix_x4_trans(s_w + swz_off<CB>((ky * 3 + kx) * C + ch * 16 + b_k, j + b_nc), r);
            b[ky][j][0] = r[0];
            b[ky][j][1] = r[1];
            b[ky][j + 1][0] = r[2];
            b[ky][j + 1][1] = r[3];
          }
#pragma unroll
        for (int hr = 0; hr < WR + 2; ++hr) {
          uint32_t a[4];
          ldmatrix_x4(sa + swz_off<2>((warp * WR + hr) * 18 + a_row + kx, a_kc), a);
#pragma unroll
          for (int ky = 0; ky < 3; ++ky) {
            const int r = hr - ky;
            if (r >= 0 && r < WR) {
#pragma unroll
              for (int j = 0; j < NT8; ++j) mma16816(acc[r][j], a, b[ky][j][0], b[ky][j][1]);
            }
          }
        }
      }
      if (ch != KC - 1) continue;

      // ---- epilogue for this tile (from the fragments)
      float bb[NT8][2];
#pragma unroll
      for (int j = 0; j < NT8; ++j) {
        bb[j][0] = __ldg(N.b + j * 8 + 2 * t);
        bb[j][1] = __ldg(N.b + j * 8 + 2 * t + 1);
      }
      if constexpr (kOut) {
#pragma unroll
        for (int r = 0; r < WR; ++r)
#pragma unroll
          for (int s = 0; s < KC; ++s) {
            const int j0 = 2 * s, j1 = 2 * s + 1;
            const uint32_t a[4] = {
                pack_half2(silu(acc[r][j0][0] + bb[j0][0]), silu(acc[r][j0][1] + bb[j0][1])),
                pack_half2(silu(acc[r][j0][2] + bb[j0][0]), silu(acc[r][j0][3] + bb[j0][1])),
                pack_half2(silu(acc[r][j1][0] + bb[j1][0]), silu(acc[r][j1][1] + bb[j1][1])),
                pack_half2(silu(acc[r][j1][2] + bb[j1][0]), silu(acc[r][j1][3] + bb[j1][1]))};
            mma16816(oacc[r], a, bo[s][0], bo[s][1]);
          }
        // store: lane holds output channels 2t, 2t+1 of pixels x = tx0 + g (+ 8)
        const size_t P = static_cast<size_t>(H) * W;
#pragma unroll
        for (int r = 0; r < WR; ++r) {
          const int y = ty0 + warp * WR + r;
          if (y < H) {
#pragma unroll
            for (int h = 0; h < 2; ++h) {
              const size_t px = static_cast<size_t>(y) * W + tx0 + g + 8 * h;
              float v0 = oacc[r][2 * h] + out_bias0, v1 = oacc[r][2 * h + 1] + out_bias1;
              if (p.accumulate) {
                v0 += p.out[(2 * t) * P + px];
                v1 += p.out[(2 * t + 1) * P + px];
              }
              p.out[(2 * t) * P + px] = v0;
              p.out[(2 * t + 1) * P + px] = v1;
            }
          }
#pragma unroll
          for (int e = 0; e < 4; ++e) oacc[r][e] = 0.0f;
        }
      } else {
        // bias + SiLU in FP16 fragment layout; 16-byte stores per pixel (quad transpose)
        uint32_t prev[2][NT8];
#pragma unroll
        for (int r = 0; r < WR; ++r) {
          const int y = ty0 + warp * WR + r;
          uint32_t v[2][NT8];
#pragma unroll
          for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int j = 0; j < NT8; ++j)
              v[h][j] = silu_pack(acc[r][j][2 * h] + bb[j][0], acc[r][j][2 * h + 1] + bb[j][1]);
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const size_t px = static_cast<size_t>(y) * W + tx0 + g + 8 * h;
            uint32_t sv[NT8];  // store_c transposes in place
#pragma unroll
            for (int j = 0; j < NT8; ++j) sv[j] = v[h][j];
            store_c(N.out + px * C, sv, y < H);
          }
          if constexpr (kPool) {
            // 2x2 max of the stored (FP16-rounded) values: rows r - 1, r of this warp (even
            // ty0 and WR), x pair in lanes g, g ^ 1 (lane ^ 4)
            if (r % 2 == 1) {
              const int W2 = W / 2;
#pragma unroll
              for (int h = 0; h < 2; ++h) {
                uint32_t m[NT8];
#pragma unroll
                for (int j = 0; j < NT8; ++j) {
                  __half2 a2 = __hmax2(*reinterpret_cast<const __half2*>(&v[h][j]),
                                       *reinterpret_cast<const __half2*>(&prev[h][j]));
                  uint32_t au = *reinterpret_cast<uint32_t*>(&a2);
                  const uint32_t ou = __shfl_xor_sync(0xffffffffu, au, 4);
                  a2 = __hmax2(a2, *reinterpret_cast<const __half2*>(&ou));
                  m[j] = *reinterpret_cast<uint32_t*>(&a2);
                }
                const size_t pp = static_cast<size_t>(y / 2) * W2 + (tx0 + g + 8 * h) / 2;
                store_c(N.pooled + pp * C, m, (g & 1) == 0 && y < H);
              }
            }
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
              for (int j = 0; j < NT8; ++j) prev[h][j] = v[h][j];
          }
        }
      }
      zero_acc();
    }
  }
}

// Launch over one net, persistent: min(tiles, resident blocks). Returns cudaErrorInvalidValue
// if the shapes do not fit the kernel.
template <int C, int TH, int WM, bool kOut, bool kPool = false, int NS = 2>
inline cudaError_t launch_head_fullres(const FullresParams& p, cudaStream_t s) {
  using Cfg = FullresCfg<C, TH, WM, NS>;
  if (p.W % 16 || p.H % 2 || p.H <= 0 || p.W <= 0) return cudaErrorInvalidValue;
  if (kPool && TH % 2) return cudaErrorInvalidValue;
  auto kern = head_fullres_kernel<C, TH, WM, kOut, kPool, NS>;
  int resident = 0;
  const auto status = detail::device_launch_value(resident, [kern](int device, int& value) {
    return detail::resident_blocks(kern, device, Cfg::NT, Cfg::kSmem, value);
  });
  if (status != cudaSuccess) return status;
  const int tiles = ((p.H + TH - 1) / TH) * (p.W / 16);
  kern<<<tiles < resident ? tiles : resident, Cfg::NT, Cfg::kSmem, s>>>(p);
  return cudaGetLastError();
}

}  // namespace headk
}  // namespace gfused
