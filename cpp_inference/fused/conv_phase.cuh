// Upsample folding for the decoder merge convs (InMode::kUpCatPh, included by conv_fused.cuh).
//
//   out = conv3x3(concat(bilinear_up2x(L), S))   (align_corners=False, edge clamp, zero pad)
//
// The bilinear 2x upsample and the 3x3 conv are both linear, so for every output phase
// (py, px) = (y & 1, x & 1) the upsampled half equals a 3x3 conv applied directly to the
// low-resolution map L:
//
//   out_up(2i + py, 2j + px) = sum_{dy, dx in 0..2} L(i + dy - 1, j + dx - 1) . P[py, px][dy][dx]
//   P[py, px][dy][dx] = sum_{ky, kx} a_py[dy][ky] a_px[dx][kx] W[ky][kx]
//
// with the per-axis bilinear taps (rows: L offset dy - 1, columns: conv tap k - 1)
//   a_0 = [[.75, .25, 0], [.25, .75, .75], [0, 0, .25]]   (even outputs)
//   a_1 = [[.25, 0, 0], [.75, .75, .25], [0, .25, .75]]    (odd outputs)
// Every phase kernel needs the full 3x3 low-resolution support (the even phase reads
// L(i - 1) through U(2i - 1) and L(i + 1) through U(2i + 1)), so the four phase kernels hold
// 4 x 9 taps (host: fold_upsample_phases, upfold.h). The FLOPs equal the unfolded conv; the
// gain is that the A operand is the raw low-resolution patch: no per-pixel 4-tap blend and
// no write + re-read of the upsampled halo tile through shared memory.
//
// Borders. Reading L with clamped (replicated) indices reproduces the upsample's edge clamp
// exactly for every in-image U, so the folded sum equals the 3x3 conv over U padded by
// *replication* (U(-1, x) := U(0, x), ...). The reference pads U with zeros, so the output
// pixels on the image border subtract the outside taps' replicated values:
//   top    (y = 0):     C = sum_kx W[0][kx] . U(0, clamp(x + kx - 1))          (corners incl.)
//   bottom (y = H - 1): C = sum_kx W[2][kx] . U(H - 1, clamp(x + kx - 1))
//   left   (x = 0):     C = sum_ky W[ky][0] . U0(y + ky - 1, 0)  (U0: zero outside the rows)
//   right  (x = W - 1): C = sum_ky W[ky][2] . U0(y + ky - 1, W - 1)
// (the row lines cover the corner taps, so the column lines skip rows outside the image).
// Each line is a 16-position x (3 x c0) x BN GEMM; warps 0..3 of a border tile run it with
// m16n8k8 MMAs from the low-resolution patch already in shared memory (U interpolated in
// registers) and the original FP16 weights (ConvParams::weight_edge, k-pair packed), and the
// results are subtracted from the FP32 tile before the epilogue.
//
// Tiling. Output tile TH x 16 (as conv_v2_kernel; same epilogue contract), seen as 4 phases x
// (TH/2 x 8) low-resolution positions. An MMA M fragment is 16 positions of one phase: column
// 0..7 of the low-resolution rows i and i + TH/4 (so that row offsets dy reuse fragments, as
// mma_chunk does). Warp wm owns fragments wm * WR .. + WR - 1 of the virtual order
// (phase, i), all of one phase (WR divides TH / 4), so its B fragments are that phase's.
// Upsampled-segment chunks are BKU = 8 channels (m16n8k8; 36-tap B, the low-resolution patch
// (TH/2 + 2) x 10 as A, staged with cp.async and clamped indices); skip chunks are BK
// channels of the (TH + 2) x 18 halo, staged in four parity planes so that one phase's
// stride-2 pixels are contiguous (conflict-free ldmatrix rows).
#pragma once

namespace gfused {
namespace detail {

template <int TH, int BK, int BN, int WM, int WN, int BKU_ = 8>
struct PhCfg {
  static constexpr int BKU = BKU_;
  static constexpr int NT = WM * WN * 32;
  static constexpr int Q = TH / 4;  // low-resolution rows per phase / 2
  static constexpr int WR = TH / WM, NT8 = BN / WN / 8;
  static constexpr int HH = TH + 2, HW = 18;  // hi-res halo
  static constexpr int PPC = 9, PP = (TH / 2 + 1) * PPC;  // parity plane: rows x cols
  static constexpr int PR = TH / 2 + 2, PC = 10;          // low-resolution patch
  static constexpr int CA = BK / 8, CU = BKU / 8, CB = BN / 8;
  static constexpr int kASkip = HH * HW * row_chunks<CA>() * 16;
  static constexpr int kAUp = PR * PC * row_chunks<CU>() * 16;
  static constexpr int kBSkip = 9 * BK * row_chunks<CB>() * 16;
  static constexpr int kBUp = 36 * BKU * row_chunks<CB>() * 16;
  static constexpr int kStageS = kASkip + kBSkip, kStageU = kAUp + kBUp;
  static constexpr int kStage = kStageS > kStageU ? kStageS : kStageU;
  static constexpr int kEpiLd = BN + 4;
  static constexpr int kEpi = TH * 16 * kEpiLd * 4;
  static constexpr int kSmem = 2 * kStage > kEpi ? 2 * kStage : kEpi;
  static constexpr int kAccRegs = WR * NT8 * 4;
  static constexpr int kMinBlocks =
      (NT >= 512 || kAccRegs > 32 || 2 * kStage > 50 * 1024) ? 1 : 65536 / (NT * 128);
  static_assert(TH % 8 == 0, "kUpCatPh: TH multiple of 8");
  static_assert(Q % WR == 0, "kUpCatPh: a warp's fragments must share one phase");
  static constexpr int NB = (TH + 15) / 16;  // 16-row blocks of a left / right border line
  static_assert(WM >= 2 + 2 * NB, "kUpCatPh: one M warp per border line block");
  static_assert(BN % (WN * 8) == 0, "BN must split into n8 tiles across N warps");
  static_assert(BK % 8 == 0 && BK <= 64, "BK multiple of 8 up to 64");
  static_assert(BKU == 8 || BKU == 16, "upsampled chunks of 8 or 16 channels");
};

// One skip chunk of MMAs (A = parity-plane halo, B = [9][BK][BN]) for a warp of phase
// (PY, px). Fragment row m reads plane ((PY + ky) & 1, (px + kx) & 1), plane row
// arow + h, column acol + ((px + kx) >> 1), h = r + ((PY + ky) >> 1).
template <int PY, class C>
__device__ __forceinline__ void ph_mma_skip(uint32_t s_a, uint32_t s_b, int px, int arow, int acol,
                                            int wn, float (&acc)[C::WR][C::NT8][4]) {
  constexpr int WR = C::WR, NT8 = C::NT8, CA = C::CA, CB = C::CB, BK = C::CA * 8;
  const int lane = threadIdx.x & 31, a_kc = lane >> 4;
#pragma unroll
  for (int s = 0; s < BK / 16; ++s) {
#pragma unroll
    for (int kx = 0; kx < 3; ++kx) {
      const int cpar = (px + kx) & 1, co = (px + kx) >> 1;
      uint32_t b[3][NT8][2];
#pragma unroll
      for (int ky = 0; ky < 3; ++ky) load_b<NT8, CB>(s_b, (ky * 3 + kx) * BK + s * 16, wn, b[ky]);
#pragma unroll
      for (int rp = 0; rp < 2; ++rp) {
#pragma unroll
        for (int h = 0; h <= WR; ++h) {
          bool used = false;
#pragma unroll
          for (int ky = 0; ky < 3; ++ky) {
            const int r = h - ((PY + ky) >> 1);
            used |= ((PY + ky) & 1) == rp && r >= 0 && r < WR;
          }
          if (!used) continue;
          const int pl = (rp * 2 + cpar) * C::PP + (arow + h) * C::PPC + acol + co;
          uint32_t a[4];
          ldmatrix_x4(s_a + swz_off<CA>(pl, s * 2 + a_kc), a);
#pragma unroll
          for (int ky = 0; ky < 3; ++ky) {
            const int r = h - ((PY + ky) >> 1);
            if (((PY + ky) & 1) == rp && r >= 0 && r < WR) {
#pragma unroll
              for (int j = 0; j < NT8; ++j) mma16816(acc[r][j], a, b[ky][j][0], b[ky][j][1]);
            }
          }
        }
      }
    }
  }
  if constexpr (BK % 16) {  // trailing 8-channel unit: m16n8k8
    constexpr int c = BK / 8 - 1;
#pragma unroll
    for (int kx = 0; kx < 3; ++kx) {
      const int cpar = (px + kx) & 1, co = (px + kx) >> 1;
      uint32_t b[3][NT8];
#pragma unroll
      for (int ky = 0; ky < 3; ++ky)
        load_b_k8<NT8, CB>(s_b, (ky * 3 + kx) * BK + c * 8, wn, b[ky]);
#pragma unroll
      for (int rp = 0; rp < 2; ++rp) {
#pragma unroll
        for (int h = 0; h <= WR; ++h) {
          bool used = false;
#pragma unroll
          for (int ky = 0; ky < 3; ++ky) {
            const int r = h - ((PY + ky) >> 1);
            used |= ((PY + ky) & 1) == rp && r >= 0 && r < WR;
          }
          if (!used) continue;
          const int pl = (rp * 2 + cpar) * C::PP + (arow + h) * C::PPC + acol + co;
          uint32_t a[2];
          ldmatrix_x2(s_a + swz_off<CA>(pl, c), a);
#pragma unroll
          for (int ky = 0; ky < 3; ++ky) {
            const int r = h - ((PY + ky) >> 1);
            if (((PY + ky) & 1) == rp && r >= 0 && r < WR) {
#pragma unroll
              for (int j = 0; j < NT8; ++j) mma1688(acc[r][j], a[0], a[1], b[ky][j]);
            }
          }
        }
      }
    }
  }
}

// One upsampled-segment chunk (BKU channels): A = low-resolution patch rows arow + h,
// columns acol + dx; B = this warp's phase kernel taps [ph][dy][dx].
template <class C>
__device__ __forceinline__ void ph_mma_up(uint32_t s_a, uint32_t s_b, int ph, int arow, int acol,
                                          int wn, float (&acc)[C::WR][C::NT8][4]) {
  constexpr int WR = C::WR, NT8 = C::NT8, CB = C::CB, BKU = C::BKU;
  const int a_kc = (threadIdx.x & 31) >> 4;
#pragma unroll
  for (int dx = 0; dx < 3; ++dx) {
    if constexpr (BKU == 8) {
      uint32_t b[3][NT8];
#pragma unroll
      for (int dy = 0; dy < 3; ++dy)
        load_b_k8<NT8, CB>(s_b, (ph * 9 + dy * 3 + dx) * 8, wn, b[dy]);
#pragma unroll
      for (int h = 0; h < WR + 2; ++h) {
        uint32_t a[2];
        ldmatrix_x2(s_a + ((arow + h) * C::PC + acol + dx) * 16, a);
#pragma unroll
        for (int dy = 0; dy < 3; ++dy) {
          const int r = h - dy;
          if (r >= 0 && r < WR) {
#pragma unroll
            for (int j = 0; j < NT8; ++j) mma1688(acc[r][j], a[0], a[1], b[dy][j]);
          }
        }
      }
    } else {
      uint32_t b[3][NT8][2];
#pragma unroll
      for (int dy = 0; dy < 3; ++dy) load_b<NT8, CB>(s_b, (ph * 9 + dy * 3 + dx) * 16, wn, b[dy]);
#pragma unroll
      for (int h = 0; h < WR + 2; ++h) {
        uint32_t a[4];
        ldmatrix_x4(s_a + swz_off<2>((arow + h) * C::PC + acol + dx, a_kc), a);
#pragma unroll
        for (int dy = 0; dy < 3; ++dy) {
          const int r = h - dy;
          if (r >= 0 && r < WR) {
#pragma unroll
            for (int j = 0; j < NT8; ++j) mma16816(acc[r][j], a, b[dy][j][0], b[dy][j][1]);
          }
        }
      }
    }
  }
}

}  // namespace detail

// ConvParams fields used: in0 (low res, H/2 x W/2 x c0), in1 (skip, H x W x c1), c0 % 8 == 0,
// c1 % BK == 0, H and W even, weight [9][c0 + c1][cout] (skip rows used), weight_ph
// [4][9][c0][cout], weight_edge [9][c0 / 2][cout] (half2 k pairs of weight's upsampled rows),
// bias / residual / out / cout / silu as usual; row0 / rows bands (row0 % TH == 0).
template <int TH, int BK, int BN, int WM, int WN, int BKU, class Epi>
__global__ void __launch_bounds__(WM * WN * 32,
                                  (detail::PhCfg<TH, BK, BN, WM, WN, BKU>::kMinBlocks))
    conv_ph_kernel(ConvParams p, typename Epi::Params ep) {
  using C = detail::PhCfg<TH, BK, BN, WM, WN, BKU>;
  using namespace detail;
  constexpr int NT = C::NT, WR = C::WR, NT8 = C::NT8, Q = C::Q;
  extern __shared__ __align__(128) unsigned char smem_raw[];
  const uint32_t s_base = smem_u32(smem_raw);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int wm = warp % WM, wn = warp / WM;

  const int tiles_x = (p.W + 15) / 16;
  const int ty0 = p.row0 + (blockIdx.y / tiles_x) * TH;
  const int tx0 = (blockIdx.y % tiles_x) * 16;
  const int n0 = blockIdx.x * BN;
  const int Hs = p.H / 2, Ws = p.W / 2;
  const int nu = p.c0 / BKU, n_chunks = nu + p.c1 / BK;

  // staging descriptors: skip halo (parity planes, zero padding) and the low-res patch
  // (clamped indices, always valid)
  constexpr int US = C::HH * C::HW * C::CA, UPT_S = (US + NT - 1) / NT;
  constexpr int UU = C::PR * C::PC * C::CU, UPT_U = (UU + NT - 1) / NT;
  int sg[UPT_S], ug[UPT_U];
  uint32_t so[UPT_S], uo[UPT_U];
#pragma unroll
  for (int i = 0; i < UPT_S; ++i) {
    const int u = tid + i * NT;
    const int pix = u / C::CA, c8 = u % C::CA;
    const int hy = pix / C::HW, hx = pix % C::HW;
    const int y = ty0 + hy - 1, x = tx0 + hx - 1;
    const bool ok = y >= 0 && y < p.H && x >= 0 && x < p.W;
    sg[i] = u >= US ? -2 : ok ? (y * p.W + x) * p.c1 + c8 * 8 : -1;
    const int pl = ((hy & 1) * 2 + (hx & 1)) * C::PP + (hy >> 1) * C::PPC + (hx >> 1);
    so[i] = swz_off<C::CA>(pl, c8);
  }
#pragma unroll
  for (int i = 0; i < UPT_U; ++i) {
    const int u = tid + i * NT;
    const int pp = u / C::CU, c8 = u % C::CU;
    const int sy = min(max(ty0 / 2 - 1 + pp / C::PC, 0), Hs - 1);
    const int sx = min(max(tx0 / 2 - 1 + pp % C::PC, 0), Ws - 1);
    ug[i] = u >= UU ? -2 : (sy * Ws + sx) * p.c0 + c8 * 8;
    uo[i] = swz_off<C::CU>(pp, c8);
  }

  auto buf = [&](int kc) { return s_base + (kc & 1) * C::kStage; };
  auto issue = [&](int kc) {
    const uint32_t sa = buf(kc);
    if (kc < nu) {
      const int k0 = kc * BKU;
      const __half* src = p.in0 + k0;
#pragma unroll
      for (int i = 0; i < UPT_U; ++i)
        if (UU % NT == 0 || ug[i] != -2) cp_async16(sa + uo[i], src + ug[i], true);
      if (!(p.dbg & 2)) stage_weights<36, BKU, BN, NT>(p.weight_ph, p.c0, p.cout, k0, n0, sa + C::kAUp);
    } else {
      const int k = (kc - nu) * BK;
      const __half* src = p.in1 + k;
#pragma unroll
      for (int i = 0; i < UPT_S; ++i)
        if (US % NT == 0 || sg[i] != -2) cp_async16(sa + so[i], src + max(sg[i], 0), sg[i] >= 0);
      if (!(p.dbg & 16)) stage_weights<9, BK, BN, NT>(p.weight, p.c0 + p.c1, p.cout, p.c0 + k, n0, sa + C::kASkip);
    }
  };

  float acc[WR][NT8][4];
#pragma unroll
  for (int r = 0; r < WR; ++r)
#pragma unroll
    for (int j = 0; j < NT8; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[r][j][e] = 0.0f;

  // this warp's phase and fragment rows
  const int v0 = wm * WR, ph = v0 / Q, i0 = v0 % Q, py = ph >> 1, px = ph & 1;
  const int m = (lane & 7) + ((lane >> 3) & 1) * 8;
  const int arow = i0 + (m >> 3) * Q, acol = m & 7;

  // border lines: warp wm 0 top, 1 bottom, 2 .. 1 + NB left rows lb * 16 .., 2 + NB ..
  // 1 + 2 NB right (when the tile holds that edge)
  const int yb = p.H - 1 - ty0, xr = p.W - 1 - tx0;
  const int lkind = wm < 2 ? wm : wm < 2 + C::NB ? 2 : 3;  // 0 top 1 bottom 2 left 3 right
  const int lb = wm < 2 ? 0 : (wm - 2) % C::NB;           // 16-row block of a column line
  const bool line = wm < 2 + 2 * C::NB &&
                    ((lkind == 0 && ty0 == 0) || (lkind == 1 && yb >= 0 && yb < TH) ||
                     (lkind == 2 && tx0 == 0) || (lkind == 3 && xr >= 0 && xr < 16));
  float corr[NT8][4];
#pragma unroll
  for (int j = 0; j < NT8; ++j)
#pragma unroll
    for (int e = 0; e < 4; ++e) corr[j][e] = 0.0f;
  // One upsampled chunk of a border line: A[pos][k3 * 8 + c] = U at line position
  // pos + k3 - 1 (interpolated from the patch in s_a), B = weight_edge taps.
  auto line_chunk = [&](uint32_t s_a, int k0) {
    const int g = lane >> 2, t = lane & 3;
    const bool row_line = lkind < 2;
    const int sy0 = ty0 / 2 - 1, sx0 = tx0 / 2 - 1;
#pragma unroll
    for (int k3 = 0; k3 < 3; ++k3) {
      const int tap = lkind == 0 ? k3 : lkind == 1 ? 6 + k3 : lkind == 2 ? 3 * k3 : 3 * k3 + 2;
      int i0p[2], i1p[2];
      float w1[2];
      bool ok[2];
#pragma unroll
      for (int hlf = 0; hlf < 2; ++hlf) {
        const int pos = g + hlf * 8 + (row_line ? 0 : lb * 16);
        int j0, j1;
        if (row_line) {
          up_coord(min(max(tx0 + pos + k3 - 1, 0), p.W - 1), Ws, j0, j1, w1[hlf]);
          const int prow = (lkind == 0 ? 0 : Hs - 1) - sy0;
          i0p[hlf] = prow * C::PC + j0 - sx0;
          i1p[hlf] = prow * C::PC + j1 - sx0;
          ok[hlf] = true;
        } else {
          const int yq = ty0 + pos + k3 - 1;
          ok[hlf] = pos < TH && yq >= 0 && yq < p.H;
          up_coord(min(max(yq, 0), p.H - 1), Hs, j0, j1, w1[hlf]);
          const int pcol = (lkind == 2 ? 0 : Ws - 1) - sx0;
          i0p[hlf] = (j0 - sy0) * C::PC + pcol;
          i1p[hlf] = (j1 - sy0) * C::PC + pcol;
        }
      }
#pragma unroll
      for (int s8 = 0; s8 < BKU / 8; ++s8) {
        uint32_t a[2];
#pragma unroll
        for (int hlf = 0; hlf < 2; ++hlf) {
          float2 v = make_float2(0.f, 0.f);
          if (ok[hlf]) {
            uint32_t q0, q1;
            asm volatile("ld.shared.u32 %0, [%1];\n"
                         : "=r"(q0)
                         : "r"(s_a + swz_off<C::CU>(i0p[hlf], s8) + t * 4));
            asm volatile("ld.shared.u32 %0, [%1];\n"
                         : "=r"(q1)
                         : "r"(s_a + swz_off<C::CU>(i1p[hlf], s8) + t * 4));
            const float2 f0 = __half22float2(*reinterpret_cast<const __half2*>(&q0));
            const float2 f1 = __half22float2(*reinterpret_cast<const __half2*>(&q1));
            v = make_float2(f0.x + (f1.x - f0.x) * w1[hlf], f0.y + (f1.y - f0.y) * w1[hlf]);
          }
          const __half2 hv = __floats2half2_rn(v.x, v.y);
          a[hlf] = *reinterpret_cast<const uint32_t*>(&hv);
        }
        const uint32_t* wb = p.weight_edge +
                             (static_cast<size_t>(tap) * (p.c0 / 2) + k0 / 2 + s8 * 4 + t) *
                                 p.cout +
                             n0 + wn * NT8 * 8 + g;
#pragma unroll
        for (int j = 0; j < NT8; ++j) mma1688(corr[j], a[0], a[1], __ldg(wb + j * 8));
      }
    }
  };

  issue(0);
  cp_async_commit();
  for (int kc = 0; kc < n_chunks; ++kc) {
    cp_async_wait_all();
    __syncthreads();
    if (kc + 1 < n_chunks) issue(kc + 1);
    cp_async_commit();
    const uint32_t sa = buf(kc);
    if (kc < nu) {
      if (!(p.dbg & 1)) ph_mma_up<C>(sa, sa + C::kAUp, ph, arow, acol, wn, acc);
      if (line && !(p.dbg & 4)) line_chunk(sa, kc * BKU);
    } else if (p.dbg & 8) {
    } else if (py) {
      ph_mma_skip<1, C>(sa, sa + C::kASkip, px, arow, acol, wn, acc);
    } else {
      ph_mma_skip<0, C>(sa, sa + C::kASkip, px, arow, acol, wn, acc);
    }
  }
  __syncthreads();

  float* tile_s = reinterpret_cast<float*>(smem_raw);
  {
    const int g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int r = 0; r < WR; ++r) {
      const int i = i0 + r;
      const int pix0 = (2 * i + py) * 16 + 2 * g + px;
      const int pix1 = (2 * (i + Q) + py) * 16 + 2 * g + px;
#pragma unroll
      for (int j = 0; j < NT8; ++j) {
        const int col = (wn * NT8 + j) * 8 + 2 * t;
        *reinterpret_cast<float2*>(tile_s + pix0 * C::kEpiLd + col) =
            make_float2(acc[r][j][0], acc[r][j][1]);
        *reinterpret_cast<float2*>(tile_s + pix1 * C::kEpiLd + col) =
            make_float2(acc[r][j][2], acc[r][j][3]);
      }
    }
  }
  __syncthreads();
  const bool any_line = ty0 == 0 || (yb >= 0 && yb < TH) || tx0 == 0 || (xr >= 0 && xr < 16);
  if (any_line) {
    if (line) {
      const int g = lane >> 2, t = lane & 3;
#pragma unroll
      for (int hlf = 0; hlf < 2; ++hlf) {
        const int pos = g + hlf * 8 + (lkind < 2 ? 0 : lb * 16);
        if (lkind >= 2 && pos >= TH) continue;
        const int pix = lkind == 0   ? pos
                        : lkind == 1 ? yb * 16 + pos
                        : lkind == 2 ? pos * 16
                                     : pos * 16 + xr;
#pragma unroll
        for (int j = 0; j < NT8; ++j) {
          const int col = (wn * NT8 + j) * 8 + 2 * t;
          atomicAdd(tile_s + pix * C::kEpiLd + col, -corr[j][2 * hlf]);
          atomicAdd(tile_s + pix * C::kEpiLd + col + 1, -corr[j][2 * hlf + 1]);
        }
      }
    }
    __syncthreads();
  }
  Epi::template apply<TH, BN, NT>(tile_s, C::kEpiLd, ty0, tx0, n0, p, ep);
}

namespace detail {
template <int TH, int BK, int BN, int WM, int WN, class Epi, int BKU = 8>
inline cudaError_t launch_ph(const ConvParams& p, const typename Epi::Params& ep,
                             cudaStream_t stream) {
  using C = PhCfg<TH, BK, BN, WM, WN, BKU>;
  auto kern = conv_ph_kernel<TH, BK, BN, WM, WN, BKU, Epi>;
  if (p.acc16 || !p.weight_ph || !p.weight_edge || p.H % 2 || p.W % 2 || p.c0 % BKU ||
      p.c1 % BK || p.cout % BN)
    return cudaErrorInvalidValue;
  if constexpr (C::kSmem > 48 * 1024) {
    int ignored = 0;
    const auto status = device_launch_value(ignored, [kern](int, int&) {
      return cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, C::kSmem);
    });
    if (status != cudaSuccess) return status;
  }
  int band = 0;
  if (!row_band(p, TH, band)) return cudaErrorInvalidValue;
  if (band == 0) return cudaSuccess;
  const int tiles_x = (p.W + 15) / 16;
  if (tiles_x < 1 || tiles_x > 65535) return cudaErrorInvalidValue;
  const int max_rows = (65535 / tiles_x) * TH;
  ConvParams part = p;
  for (int row = 0; row < band; row += max_rows) {
    part.row0 = p.row0 + row;
    part.rows = max_rows < band - row ? max_rows : band - row;
    const dim3 grid(p.cout / BN, ((part.rows + TH - 1) / TH) * tiles_x);
    kern<<<grid, C::NT, C::kSmem, stream>>>(part, ep);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return e;
  }
  return cudaSuccess;
}
}  // namespace detail

}  // namespace gfused
