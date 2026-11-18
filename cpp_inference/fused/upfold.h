// Host-side weights for upsample folding (InMode::kUpCatPh, conv_phase.cuh), derived at load
// time from a merge conv's FP16 weights, so the weight blobs need no new format.
#pragma once

#include <cuda_fp16.h>

#include <cstdint>
#include <cstring>
#include <vector>

namespace gfused {

// Bilinear 2x (align_corners=False) taps of one axis: kUpTap[parity][d][k] is the weight of
// L(i + d - 1) in U(2i + parity + k - 1) (conv tap k over U).
inline constexpr double kUpTap[2][3][3] = {
    {{0.75, 0.25, 0.0}, {0.25, 0.75, 0.75}, {0.0, 0.0, 0.25}},
    {{0.25, 0.0, 0.0}, {0.75, 0.75, 0.25}, {0.0, 0.25, 0.75}}};

// w: [9][cin][cout] FP16 (host), upsampled channels 0 .. c0 - 1.
//   ph:   [4 (py * 2 + px)][9 (dy * 3 + dx)][c0][cout] = sum_{ky,kx} a_py[dy][ky] a_px[dx][kx] w
//         (FP64 sums of the FP16 weights, rounded once to FP16)
//   edge: [9][c0 / 2][cout] half2 {w[t][2q][n], w[t][2q + 1][n]} (c0 even)
inline void fold_upsample_phases(const __half* w, int c0, int cin, int cout,
                                 std::vector<__half>& ph, std::vector<uint32_t>& edge) {
  const size_t plane = static_cast<size_t>(c0) * cout;
  ph.assign(36 * plane, __float2half(0.0f));
  edge.assign(9 * plane / 2, 0u);
  std::vector<double> wd(9 * plane);
  for (int t = 0; t < 9; ++t)
    for (int c = 0; c < c0; ++c)
      for (int n = 0; n < cout; ++n)
        wd[t * plane + static_cast<size_t>(c) * cout + n] =
            __half2float(w[(static_cast<size_t>(t) * cin + c) * cout + n]);
  for (int py = 0; py < 2; ++py)
    for (int px = 0; px < 2; ++px)
      for (int dy = 0; dy < 3; ++dy)
        for (int dx = 0; dx < 3; ++dx) {
          __half* dst = ph.data() + ((py * 2 + px) * 9 + dy * 3 + dx) * plane;
          for (size_t e = 0; e < plane; ++e) {
            double s = 0.0;
            for (int ky = 0; ky < 3; ++ky)
              for (int kx = 0; kx < 3; ++kx) {
                const double a = kUpTap[py][dy][ky] * kUpTap[px][dx][kx];
                if (a != 0.0) s += a * wd[(ky * 3 + kx) * plane + e];
              }
            dst[e] = __float2half_rn(static_cast<float>(s));
          }
        }
  for (int t = 0; t < 9; ++t)
    for (int q = 0; q < c0 / 2; ++q)
      for (int n = 0; n < cout; ++n) {
        const __half lo = w[(static_cast<size_t>(t) * cin + 2 * q) * cout + n];
        const __half hi = w[(static_cast<size_t>(t) * cin + 2 * q + 1) * cout + n];
        uint16_t l, h;
        static_assert(sizeof(__half) == 2);
        std::memcpy(&l, &lo, 2);
        std::memcpy(&h, &hi, 2);
        edge[(static_cast<size_t>(t) * (c0 / 2) + q) * cout + n] =
            static_cast<uint32_t>(l) | (static_cast<uint32_t>(h) << 16);
      }
}

}  // namespace gfused
