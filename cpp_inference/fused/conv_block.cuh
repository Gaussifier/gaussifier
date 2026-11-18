// Fused residual block: out = act(conv2(silu(conv1(X) + b1)) + b2 + proj1x1(X)) in ONE kernel.
//
// conv1's output y1 never touches global memory: each block computes y1 on its output tile's
// (TH+2) x 18 halo directly into shared memory (FP16, zero outside the image = conv2's zero
// padding), then runs conv2 over it and the 1x1 projection over X as further K segments.
// The y1 halo costs (TH+2)*18 / (TH*16) extra conv1 work (1.41x at TH = 8) but removes the
// y1 write + read, which dominates the full-resolution, low-channel blocks.
//
// X is any conv_fused input mode (kPlain / kPool / kUpCat / kImageF32), staged with the
// same Stager as the single-conv kernel (halo R = 2 for conv1, R = 0 for the projection).
// conv1's M fragments are 16 consecutive pixels of the y1 halo in raster order, addressed
// per lane through ldmatrix, so the 18-wide halo needs no padding to a multiple of 16.
//
//   launch_block<Mode, TH, NW, BK1, BK2, CM, BN, Epi>(BlockParams, Epi::Params, stream)
//
// Requirements: X channel counts (and kUpCat c0) multiples of BK1; CM multiple of BK2;
// one block computes all CM conv1 and all BN = cout output channels.
// BlockParams::p.row0 / rows select an output row band as for launch_conv2.

#pragma once

#include "conv_fused.cuh"

namespace gfused {

struct BlockParams {
  ConvParams p;      // conv2: H, W, weight = W2 [9][CM][cout], bias = b2, out, cout, silu
  ProjParams q;      // X (in0, in1, c0, c1) and the projection weight [c0 + c1][cout]
  const __half* w1;  // conv1 weight [9][c0 + c1][CM]
  const float* b1;   // conv1 bias [CM] (SiLU always applied to y1)
};

template <InMode kMode, int TH, int NW, int BK1, int BK2, int CM, int BN>
struct BlockCfg {
  static constexpr int NT = NW * 32;
  static constexpr int YH = TH + 2, YW = 18, NY = YH * YW;  // y1 halo
  static constexpr int NF = (NY + 15) / 16;                 // conv1 M fragments
  static constexpr int FPW = (NF + NW - 1) / NW;            // fragments per warp
  static constexpr int WM = NW < TH ? NW : TH;              // conv2 warp layout
  static constexpr int WN = NW / WM;
  static constexpr int A1 = (TH + 4) * 20 * BK1 * 2;
  static constexpr int B1 = 9 * BK1 * detail::row_chunks<CM / 8>() * 16;
  static constexpr int B2 = 9 * BK2 * detail::row_chunks<BN / 8>() * 16;
  static constexpr int A3 = TH * 16 * BK1 * 2, B3 = BK1 * detail::row_chunks<BN / 8>() * 16;
  static constexpr int kA = A1 > A3 ? A1 : A3;
  static constexpr int kB = (B1 > B2 ? B1 : B2) > B3 ? (B1 > B2 ? B1 : B2) : B3;
  static constexpr int kStage = kA + kB;
  static constexpr int kRaw =
      kMode == InMode::kUpCat ? (TH / 2 + 4) * 12 * BK1 * 2 : 0;  // R = 2 patch (>= R = 0)
  static constexpr int kY1 = NY * CM * 2;
  static constexpr int kEpiLd = BN + 4;
  static constexpr int kEpi = TH * 16 * kEpiLd * 4;
  static constexpr int kPipe = 2 * kStage + 2 * kRaw;
  static constexpr int kY1Off = kPipe > kEpi ? kPipe : kEpi;
  static constexpr int kSmem = kY1Off + kY1;
  static_assert(TH % WM == 0 && BN % (WN * 8) == 0, "bad warp layout");
  static_assert(CM % BK2 == 0 && BK1 % 16 == 0 && BK2 % 16 == 0, "bad K chunking");
};

namespace detail {

// conv1 chunk: A = X halo [(TH+4) x 20][BK1], B = [9][BK1][CM]; this warp's FPW fragments.
template <int TH, int BK1, int CM, int FPW>
__device__ __forceinline__ void mma_y1(uint32_t s_a, uint32_t s_b, const int (&xbase)[FPW],
                                       int nf_valid, float (&acc)[FPW][CM / 8][4]) {
  constexpr int NT8 = CM / 8, CA = BK1 / 8, CB = CM / 8;
  const int lane = threadIdx.x & 31;
  const int a_kc = lane >> 4;
#pragma unroll
  for (int s = 0; s < BK1 / 16; ++s) {
#pragma unroll
    for (int tap = 0; tap < 9; ++tap) {
      const int ky = tap / 3, kx = tap % 3;
      uint32_t b[NT8][2];
      load_b<NT8, CB>(s_b, tap * BK1 + s * 16, 0, b);
#pragma unroll
      for (int i = 0; i < FPW; ++i) {
        if (i < nf_valid) {  // warp-uniform
          uint32_t a[4];
          ldmatrix_x4(s_a + swz_off<CA>(xbase[i] + ky * 20 + kx, s * 2 + a_kc), a);
#pragma unroll
          for (int j = 0; j < NT8; ++j) mma16816(acc[i][j], a, b[j][0], b[j][1]);
        }
      }
    }
  }
}

}  // namespace detail

template <InMode kMode, int TH, int NW, int BK1, int BK2, int CM, int BN, class Epi>
__global__ void __launch_bounds__(NW * 32)
    conv_block_kernel(BlockParams bp, typename Epi::Params ep) {
  using Cfg = BlockCfg<kMode, TH, NW, BK1, BK2, CM, BN>;
  constexpr int NT = Cfg::NT, NY = Cfg::NY, FPW = Cfg::FPW, NF = Cfg::NF;
  constexpr int WM = Cfg::WM, WN = Cfg::WN, WR = TH / WM, NT8 = BN / WN / 8;
  constexpr int CA2 = BK2 / 8;
  extern __shared__ __align__(128) unsigned char smem_raw[];
  const uint32_t s_base = detail::smem_u32(smem_raw);
  const uint32_t s_y1 = s_base + Cfg::kY1Off;
  const ConvParams& p = bp.p;
  const ProjParams& q = bp.q;

  const int tiles_x = (p.W + 15) / 16;
  const int ty0 = p.row0 + (blockIdx.x / tiles_x) * TH;  // row band (ConvParams::row0)
  const int tx0 = (blockIdx.x % tiles_x) * 16;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;

  const int xc = q.c0 + q.c1;
  const int n1 = xc / BK1, n2 = CM / BK2, n3 = q.weight ? xc / BK1 : 0;
  const int n_chunks = n1 + n2 + n3;

  const detail::SrcDesc sx{q.in0, q.in1, q.c0, q.c1};
  detail::Stager<kMode, 2, TH, BK1, NT, 8, false> st1;  // conv1 X halo
  detail::Stager<kMode, 0, TH, BK1, NT, 8, false> st3;  // projection X tile
  st1.prepare(sx, p.H, p.W, ty0, tx0);
  st3.prepare(sx, p.H, p.W, ty0, tx0);

  // conv1 fragment rows: y1 halo pixel qq (raster over (TH+2) x 18) -> X halo pixel at tap 0
  int xbase[FPW];
  const int a_row = (lane & 7) + ((lane >> 3) & 1) * 8;
#pragma unroll
  for (int i = 0; i < FPW; ++i) {
    const int qq = min((warp * FPW + i) * 16 + a_row, NY - 1);
    xbase[i] = (qq / 18) * 20 + qq % 18;
  }
  const int nf_valid = min(FPW, NF - warp * FPW);  // may be <= 0 for the last warps

  float acc1[FPW][CM / 8][4];
  float acc[WR][NT8][4];
#pragma unroll
  for (int i = 0; i < FPW; ++i)
#pragma unroll
    for (int j = 0; j < CM / 8; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc1[i][j][e] = 0.0f;
#pragma unroll
  for (int r = 0; r < WR; ++r)
#pragma unroll
    for (int j = 0; j < NT8; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[r][j][e] = 0.0f;

  using detail::StageKind;
  const int seg3 = n1 + n2;
  auto buf = [&](int kc) { return s_base + (kc & 1) * Cfg::kStage; };
  auto raw_slot = [&](int kc) { return s_base + 2 * Cfg::kStage + (kc & 1) * Cfg::kRaw; };
  auto is_raw = [&](int kc) {
    if (kc >= n_chunks || (kc >= n1 && kc < seg3)) return false;
    return st1.is_raw(sx, (kc < n1 ? kc : kc - seg3) * BK1);
  };
  auto issue_raw = [&](int kc) {
    if (kc < n1) {
      st1.issue_raw(sx, p.H, p.W, ty0, tx0, kc * BK1, raw_slot(kc));
    } else {
      st3.issue_raw(sx, p.H, p.W, ty0, tx0, (kc - seg3) * BK1, raw_slot(kc));
    }
  };
  auto transform = [&](int kc) {
    if (kc < n1) {
      st1.transform(p.H, p.W, ty0, tx0, buf(kc), raw_slot(kc));
    } else {
      st3.transform(p.H, p.W, ty0, tx0, buf(kc), raw_slot(kc));
    }
  };
  auto issue_a = [&](int kc) -> StageKind {
    if (kc < n1) return st1.issue(sx, p.H, p.W, ty0, tx0, kc * BK1, buf(kc));
    if (kc >= seg3) return st3.issue(sx, p.H, p.W, ty0, tx0, (kc - seg3) * BK1, buf(kc));
    return StageKind::kAsync;  // conv2 chunks read the resident y1 tile
  };
  auto finish_reg = [&](int kc) {
    if (kc < n1) {
      st1.finish_reg(buf(kc));
    } else if (kc >= seg3) {
      st3.finish_reg(buf(kc));
    }
  };
  auto stage_b = [&](int kc) {
    const uint32_t s_b = buf(kc) + Cfg::kA;
    if (kc < n1) {
      detail::stage_weights<9, BK1, CM, NT>(bp.w1, xc, CM, kc * BK1, 0, s_b);
    } else if (kc < seg3) {
      detail::stage_weights<9, BK2, BN, NT>(p.weight, CM, BN, (kc - n1) * BK2, 0, s_b);
    } else {
      detail::stage_weights<1, BK1, BN, NT>(q.weight, xc, BN, (kc - seg3) * BK1, 0, s_b);
    }
  };
  // y1 = silu(acc1 + b1), zero outside the image, FP16 into the resident [CM/BK2][NY][BK2]
  // swizzled tile (the halo layout conv2's mma_chunk<3> expects).
  auto store_y1 = [&]() {
    const int g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int i = 0; i < FPW; ++i) {
      if (i >= nf_valid) continue;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int qq = (warp * FPW + i) * 16 + g + h * 8;
        if (qq >= NY) continue;
        const int y = ty0 - 1 + qq / 18, x = tx0 - 1 + qq % 18;
        const bool in = y >= 0 && y < p.H && x >= 0 && x < p.W;
#pragma unroll
        for (int j = 0; j < CM / 8; ++j) {
          const int n = j * 8 + 2 * t;
          float v0 = 0.0f, v1 = 0.0f;
          if (in) {
            v0 = detail::silu(acc1[i][j][2 * h] + __ldg(bp.b1 + n));
            v1 = detail::silu(acc1[i][j][2 * h + 1] + __ldg(bp.b1 + n + 1));
          }
          const __half2 hv = __floats2half2_rn(v0, v1);
          const uint32_t addr = s_y1 + (n / BK2) * (NY * BK2 * 2) +
                                detail::swz_off<CA2>(qq, (n % BK2) / 8) + (n % 8) * 2;
          asm volatile("st.shared.b32 [%0], %1;\n" ::"r"(addr),
                       "r"(*reinterpret_cast<const uint32_t*>(&hv)));
        }
      }
    }
  };

  {
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
      detail::mma_y1<TH, BK1, CM, FPW>(s_a, s_a + Cfg::kA, xbase, nf_valid, acc1);
      if (kc == n1 - 1) store_y1();  // visible to conv2 after the next barrier
    } else if (kc < seg3) {
      detail::mma_chunk<3, TH, BK2, BN, WM, WN>(s_y1 + (kc - n1) * (NY * BK2 * 2),
                                                s_a + Cfg::kA, acc);
    } else {
      detail::mma_chunk<1, TH, BK1, BN, WM, WN>(s_a, s_a + Cfg::kA, acc);
    }
    if (next_raw) transform(kc + 1);
    if (k == StageKind::kReg) finish_reg(kc + 1);
  }
  __syncthreads();

  float* tile_s = reinterpret_cast<float*>(smem_raw);
  {
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
  Epi::template apply<TH, BN, NT>(tile_s, Cfg::kEpiLd, ty0, tx0, 0, p, ep);
}

template <InMode kMode, int TH, int NW, int BK1, int BK2, int CM, int BN,
          class Epi = DefaultEpilogue>
inline cudaError_t launch_block(const BlockParams& bp, const typename Epi::Params& ep,
                                cudaStream_t stream) {
  using Cfg = BlockCfg<kMode, TH, NW, BK1, BK2, CM, BN>;
  auto kern = conv_block_kernel<kMode, TH, NW, BK1, BK2, CM, BN, Epi>;
  if constexpr (Cfg::kSmem > 48 * 1024) {
    int ignored = 0;
    const auto status = detail::device_launch_value(ignored, [kern](int, int&) {
      return cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, Cfg::kSmem);
    });
    if (status != cudaSuccess) return status;
  }
  if (bp.p.cout != BN) return cudaErrorInvalidValue;
  int band = 0;
  if (!detail::row_band(bp.p, TH, band)) return cudaErrorInvalidValue;
  if (band == 0) return cudaSuccess;
  const int tiles = ((band + TH - 1) / TH) * ((bp.p.W + 15) / 16);
  kern<<<tiles, Cfg::NT, Cfg::kSmem, stream>>>(bp, ep);
  return cudaGetLastError();
}

}  // namespace gfused
