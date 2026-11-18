// Torch-free EMV sampler with device-resident counts. See pipeline_sampler.cuh.
//
// Semantics follow the native sampler used by the Python reference
// (gaussifier.voronoi_samplers.equal_mass_voronoi -> equal_mass_voronoi_native):
//   * error-diffusion oversample to max(n + 1, round(1.5 n)) points (tile top-k emit,
//     systematic rounding of tile masses);
//   * merge rounds: Voronoi assignment + cell masses -> 8 lowest-mass Voronoi neighbours ->
//     parallel priority matching of the lowest-mass candidates -> exact-count completion ->
//     mass-weighted merge -> order-preserving compaction;
//   * 3 Lloyd iterations (assignment + density-weighted centroids);
//   * final assignment -> cell masses -> weights = mass * n / sum(mass).
// The Voronoi assignment is an exact nearest-seed search over a grid hash (see "Voronoi
// assignment" below) where the native sampler uses jump flooding + lost-point recovery, which
// is approximate (~7e-6 of the pixels on Kodak point sets get a farther seed). The lowest-mass
// candidates are found by a radix select on (mass, index) keys instead of a full argsort, and
// the matching runs one kernel per column (see "candidate selection" / "priority matching");
// both give the argsort-based result exactly (checked by sampler_assign_check.cu).
// Other deviations from the libtorch implementation: the systematic rounding uses a parallel
// scan, empty cells are removed by the matcher instead of a separate pre-pass, unmatched
// candidates complete via first live adjacent cell then nearest survivor (the libtorch exact
// path recomputes the assignment first), owners are int32, distances use explicit
// round-to-nearest products and sums (no FMA contraction), and floating-point accumulation
// order differs.

#include <cub/cub.cuh>
#include <thrust/iterator/counting_iterator.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <numeric>
#include <vector>

#include "pipeline_sampler.cuh"

namespace gfused {
namespace {

constexpr int kMaxTileArea = 32 * 32;

// ------------------------------------------------------------------ RNG (error diffusion)
__host__ __device__ inline uint64_t splitmix64_next(uint64_t state) {
  uint64_t v = state;
  v = (v ^ (v >> 30)) * 0xBF58476D1CE4E5B9ULL;
  v = (v ^ (v >> 27)) * 0x94D049BB133111EBULL;
  return v ^ (v >> 31);
}
__host__ __device__ inline uint64_t splitmix64_step(uint64_t& state) {
  state += 0x9E3779B97F4A7C15ULL;
  return splitmix64_next(state);
}
__host__ __device__ inline double unit_interval_from_seed(int64_t seed, int64_t salt) {
  uint64_t state =
      static_cast<uint64_t>(seed) + 0x9E3779B97F4A7C15ULL * (static_cast<uint64_t>(salt) + 1ULL);
  const uint64_t v = splitmix64_next(state);
  return static_cast<double>(v >> 11) * (1.0 / static_cast<double>(1ULL << 53));
}
__host__ __device__ inline double fract_unit(double v) {
  v = v - floor(v);
  return v < 0.0 ? v + 1.0 : v;
}

// ------------------------------------------------------------------ error diffusion
__global__ void k_ed_setup(SamplerCtrl* c, int H, int W, int64_t seed, double oversample) {
  const int n = max(1, c->n);
  const long long by_factor = llround(static_cast<double>(n) * oversample);
  const int n_init = static_cast<int>(max(static_cast<long long>(n) + 1, by_factor));
  const double size = static_cast<double>(H) * W;
  const double spacing = sqrt(size / fmax(1.0, static_cast<double>(n_init)));
  long long ts = max(4LL, min(static_cast<long long>(min(H, W)), llround(spacing * 4.0)));
  if (ts > 32) ts = 32;
  if (ts < 4) ts = 4;
  const int t = static_cast<int>(ts);
  uint64_t st = static_cast<uint64_t>(seed) ^ 0xA24BAED4963EE407ULL;
  const int sy = static_cast<int>(splitmix64_step(st) % static_cast<uint64_t>(t));
  const int sx = static_cast<int>(splitmix64_step(st) % static_cast<uint64_t>(t));
  c->n = n;
  c->n_init = n_init;
  c->tile_size = t;
  c->tiles_y = (H + t - 1) / t;
  c->tiles_x = (W + t - 1) / t;
  c->tile_total = c->tiles_x * c->tiles_y;
  c->shift_y = sy;
  c->shift_x = sx;
  c->n_cur = n_init;
  c->stat_matched = c->stat_forced_adj = c->stat_forced_nn = 0;
  c->stat_count_error = 0;
  c->weight_sum = 0.f;
}

// Also zeroes, once per run, the buffers whose consumers re-zero them after use (select
// histogram, grid cell counts).
__global__ void k_ed_aggregate(const float* __restrict__ density, const SamplerCtrl* c, int H,
                               int W, double* tile_mass, int* zero_a, int na, int* zero_b,
                               int nb) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  {
    const int lin = (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) +
                    threadIdx.y * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * gridDim.y * blockDim.x * blockDim.y;
    for (int j = lin; j < na; j += stride) zero_a[j] = 0;
    for (int j = lin; j < nb; j += stride) zero_b[j] = 0;
  }
  if (x >= W || y >= H) return;
  const int t = c->tile_size, tx = c->tiles_x, ty = c->tiles_y;
  const int syy = (y + c->shift_y) % H;
  const int sxx = (x + c->shift_x) % W;
  const int tile = min(syy / t, ty - 1) * tx + min(sxx / t, tx - 1);
  const float raw = density[y * W + x];
  const double v = (isfinite(raw) && raw > 0.f) ? static_cast<double>(raw) : 0.0;
  atomicAdd(&tile_mass[tile], v);
}

struct EdNorm {
  double scale, uniform, offset;
  bool valid;
};
__device__ inline EdNorm ed_norm(const SamplerCtrl* c, const double* cum, int max_tiles,
                                 double offset) {
  EdNorm e;
  const double total = cum[max_tiles - 1];
  e.valid = isfinite(total) && total > 0.0;
  e.scale = e.valid ? static_cast<double>(c->n_init) / total : 1.0;
  e.uniform = static_cast<double>(c->n_init) / static_cast<double>(c->tile_total);
  e.offset = offset;
  return e;
}
__device__ inline long long ed_edge(const EdNorm& e, const double* cum, int i) {
  const double nc = e.valid ? cum[i] * e.scale : e.uniform * static_cast<double>(i + 1);
  return static_cast<long long>(floor(nc + e.offset + 1e-12));
}

// Parallel systematic rounding: count_i = edge_i - edge_{i-1} on the scanned tile masses.
__global__ void k_ed_counts(SamplerCtrl* c, const double* __restrict__ cum, int max_tiles,
                            double offset, int* tile_count) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= max_tiles) return;
  if (i >= c->tile_total) {
    tile_count[i] = 0;
    return;
  }
  const EdNorm e = ed_norm(c, cum, max_tiles, offset);
  const long long prev = i == 0 ? static_cast<long long>(floor(offset)) : ed_edge(e, cum, i - 1);
  tile_count[i] = static_cast<int>(max(0LL, ed_edge(e, cum, i) - prev));
  if (i == 0) c->valid_total = e.valid ? 1 : 0;
}

// Exact-count repair: add/remove single points at the largest/smallest rounding residual.
__global__ void k_ed_fix(SamplerCtrl* c, const double* __restrict__ tile_mass,
                         const double* __restrict__ cum, int max_tiles, double offset,
                         int* tile_count) {
  __shared__ double s_val[32];
  __shared__ int s_idx[32];
  __shared__ long long s_diff;
  const int tid = threadIdx.x;
  const EdNorm e = ed_norm(c, cum, max_tiles, offset);
  const int total_tiles = c->tile_total;
  if (tid == 0) {
    const long long sum = ed_edge(e, cum, total_tiles - 1) - static_cast<long long>(floor(offset));
    s_diff = static_cast<long long>(c->n_init) - sum;
    c->ed_rounded_sum = static_cast<int>(sum);
  }
  __syncthreads();
  for (int guard = 0; guard < total_tiles && s_diff != 0; ++guard) {
    const bool add = s_diff > 0;
    double best = add ? -1.0 : 1e300;
    int best_i = INT_MAX;
    for (int i = tid; i < total_tiles; i += blockDim.x) {
      const double v = tile_mass[i];
      const double nv = e.valid ? ((isfinite(v) && v > 0.0) ? v * e.scale : 0.0) : e.uniform;
      const double r = nv - floor(nv);
      const bool ok = add ? (r > best) : (tile_count[i] > 0 && r < best);
      if (ok) {
        best = r;
        best_i = i;
      }
    }
    // warp then block arg-reduction (ties -> smallest index)
    for (int o = 16; o > 0; o >>= 1) {
      const double ov = __shfl_down_sync(0xffffffffu, best, o);
      const int oi = __shfl_down_sync(0xffffffffu, best_i, o);
      const bool better = add ? (ov > best || (ov == best && oi < best_i))
                              : (ov < best || (ov == best && oi < best_i));
      if (oi != INT_MAX && (best_i == INT_MAX || better)) {
        best = ov;
        best_i = oi;
      }
    }
    if ((tid & 31) == 0) {
      s_val[tid >> 5] = best;
      s_idx[tid >> 5] = best_i;
    }
    __syncthreads();
    if (tid == 0) {
      double bv = s_val[0];
      int bi = s_idx[0];
      for (int w = 1; w < static_cast<int>(blockDim.x >> 5); ++w) {
        const bool better = add ? (s_val[w] > bv || (s_val[w] == bv && s_idx[w] < bi))
                                : (s_val[w] < bv || (s_val[w] == bv && s_idx[w] < bi));
        if (s_idx[w] != INT_MAX && (bi == INT_MAX || better)) {
          bv = s_val[w];
          bi = s_idx[w];
        }
      }
      if (bi == INT_MAX) {
        s_diff = 0;
      } else if (add) {
        tile_count[bi] += 1;
        s_diff -= 1;
      } else {
        tile_count[bi] -= 1;
        s_diff += 1;
      }
    }
    __syncthreads();
  }
}

// Small-tile variant of k_ed_emit (tiles up to 16x16, the common case): one warp per tile,
// kEdWarps tiles per block, same selection and emit as k_ed_emit, in less time:
//  * the priority p = log(u) / w is first bracketed in FP32 (q = -p in [ql, qh], half-width
//    4e-6 q + 2e-6 / w, several times the FP32 error of __logf, the conversion and the division);
//  * the cells are bitonic-sorted by (ql, slot) instead of an O(cells^2) rank count;
//  * cells whose bracket overlaps another (found with a prefix max of qh in ql order) form
//    contiguous groups, which get the exact FP64 priority and are re-ordered inside the group
//    by (exact q, slot). Disjoint brackets order like the exact values, so the final order is
//    the FP64 (bits, slot) order of k_ed_emit; only the tie order of exactly equal priorities
//    can differ (slots follow pixel order here, atomic arrival there).
constexpr int kEdWarps = 4;

__device__ __forceinline__ double ed_exact_prio(const float* __restrict__ density, int idx,
                                                bool valid_total, int64_t seed) {
  const float raw = density[idx];
  double w = (isfinite(raw) && raw > 0.f) ? static_cast<double>(raw) : 0.0;
  if (!valid_total) w = 1.0;
  if (w > 0.0 && isfinite(w)) {
    double u = unit_interval_from_seed(seed, 1000003LL + idx);
    if (u < 1e-300) u = 1e-300;
    return log(u) / w;
  }
  return unit_interval_from_seed(seed, 2000003LL + idx) - 1.0;
}

__global__ void __launch_bounds__(32 * kEdWarps)
    k_ed_emit_small(const float* __restrict__ density, const SamplerCtrl* c,
                    const int* __restrict__ tile_count, const int* __restrict__ tile_off, int H,
                    int W, int64_t seed, float2* out) {
  constexpr int kArea = 256;
  // Shared memory bounds the occupancy here, so the per-warp arrays are reused:
  //   s_key: (ql bits, slot) sorted, then the exact q of grouped positions (double)
  //   s_ql:  ql by slot, then the selected cells (int, by rank)
  __shared__ unsigned long long s_key_all[kEdWarps][kArea];
  __shared__ float s_ql_all[kEdWarps][kArea], s_qh_all[kEdWarps][kArea];  // by slot
  __shared__ unsigned char s_loc_all[kEdWarps][kArea];   // tile-local pixel index, by slot
  __shared__ unsigned char s_slot_all[kEdWarps][kArea];  // slot by sorted position
  __shared__ unsigned char s_start_all[kEdWarps][kArea]; // group start, by position
  const int wid = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int tile = blockIdx.x * kEdWarps + wid;
  if (c->tile_size > 16 || tile >= c->tile_total) return;
  const int count = tile_count[tile];
  if (count <= 0) return;
  unsigned long long* s_key = s_key_all[wid];
  double* s_qx = reinterpret_cast<double*>(s_key_all[wid]);
  float* s_ql = s_ql_all[wid];
  int* s_sel = reinterpret_cast<int*>(s_ql_all[wid]);
  float* s_qh = s_qh_all[wid];
  unsigned char* s_loc = s_loc_all[wid];
  unsigned char* s_slot = s_slot_all[wid];
  unsigned char* s_start = s_start_all[wid];
  const bool valid_total = c->valid_total != 0;
  const int t = c->tile_size, tiles_x = c->tiles_x, tiles_y = c->tiles_y;
  const int shift_y = c->shift_y, shift_x = c->shift_x;
  const int ty = tile / tiles_x, tx = tile - ty * tiles_x;
  const int sy_lo = ty * t, sy_hi = min((ty + 1) * t, H);
  const int sx_lo = tx * t, sx_hi = min((tx + 1) * t, W);
  const int tw = sx_hi - sx_lo;
  const int tp = (sy_hi - sy_lo) * tw;
  auto pixel_of = [&](int l) {
    const int sy = sy_lo + l / tw, sx = sx_lo + l % tw;
    return ((sy - shift_y + H) % H) * W + (sx - shift_x + W) % W;
  };
  int n_cells = 0;
  for (int l0 = 0; l0 < tp; l0 += 32) {
    const int l = l0 + lane;
    bool ok = false;
    float ql = 0.f, qh = 0.f;
    if (l < tp) {
      const int sy = sy_lo + l / tw, sx = sx_lo + l % tw;
      ok = min(sy / t, tiles_y - 1) * tiles_x + min(sx / t, tiles_x - 1) == tile;
      if (ok) {
        const int idx = pixel_of(l);
        const float raw = density[idx];
        double w = (isfinite(raw) && raw > 0.f) ? static_cast<double>(raw) : 0.0;
        if (!valid_total) w = 1.0;
        bool exact = true;
        if (w > 0.0 && isfinite(w)) {
          const double u = unit_interval_from_seed(seed, 1000003LL + idx);
          const float wf = static_cast<float>(w);
          if (u > 1e-30 && wf > 1e-30f && wf < 1e30f) {
            const float qf = -__logf(static_cast<float>(u)) / wf;
            const float e = 4e-6f * qf + 2e-6f / wf;
            ql = __fsub_rd(qf, e);
            qh = __fadd_ru(qf, e);
            exact = false;
          }
        }
        if (exact) {
          const double q = -ed_exact_prio(density, idx, valid_total, seed);
          ql = __double2float_rd(q);
          qh = __double2float_ru(q);
        }
        ql = fmaxf(ql, 0.f);  // q > 0; keeps the float bits ordered as unsigned
      }
    }
    const unsigned m = __ballot_sync(0xffffffffu, ok);
    if (ok) {
      const int slot = n_cells + __popc(m & ((1u << lane) - 1u));
      s_key[slot] = (static_cast<unsigned long long>(__float_as_uint(ql)) << 8) |
                    static_cast<unsigned>(slot);
      s_ql[slot] = ql;
      s_qh[slot] = qh;
      s_loc[slot] = static_cast<unsigned char>(l);
    }
    n_cells += __popc(m);
  }
  if (n_cells <= 0) return;
  int np = 32;
  while (np < n_cells) np <<= 1;
  for (int j = n_cells + lane; j < np; j += 32) s_key[j] = ~0ULL;
  __syncwarp();
  for (int k = 2; k <= np; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      for (int x = lane; x < np; x += 32) {
        const int y = x ^ j;
        if (y > x) {
          const unsigned long long ka = s_key[x], kb = s_key[y];
          if ((ka > kb) == ((x & k) == 0)) {
            s_key[x] = kb;
            s_key[y] = ka;
          }
        }
      }
      __syncwarp();
    }
  }
  // group starts: position r starts a group unless an earlier bracket reaches ql[r]
  float carry = -INFINITY;
  bool any_amb = false;
  for (int base = 0; base < n_cells; base += 32) {
    const int r = base + lane;
    const int slot = r < n_cells ? static_cast<int>(s_key[r] & 0xffULL) : 0;
    const float qh = r < n_cells ? s_qh[slot] : -INFINITY;
    float incl = qh;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const float v = __shfl_up_sync(0xffffffffu, incl, o);
      if (lane >= o) incl = fmaxf(incl, v);
    }
    float excl = __shfl_up_sync(0xffffffffu, incl, 1);
    excl = lane == 0 ? carry : fmaxf(excl, carry);
    const bool st = !(r > 0 && excl >= s_ql[slot]);
    if (r < n_cells) {
      s_start[r] = st;
      s_slot[r] = static_cast<unsigned char>(slot);
    }
    any_amb |= __any_sync(0xffffffffu, r < n_cells && r > 0 && !st);
    carry = fmaxf(carry, __shfl_sync(0xffffffffu, incl, 31));
  }
  __syncwarp();  // s_key and s_ql are free from here on
  const int take = min(count, n_cells);
  if (!any_amb) {
    for (int r = lane; r < take; r += 32) s_sel[r] = pixel_of(s_loc[s_slot[r]]);
  } else {
    // exact priorities for every member of a multi-cell group
    for (int r = lane; r < n_cells; r += 32) {
      const bool grouped = !s_start[r] || (r + 1 < n_cells && !s_start[r + 1]);
      if (grouped) s_qx[r] = -ed_exact_prio(density, pixel_of(s_loc[s_slot[r]]), valid_total, seed);
    }
    __syncwarp();
    for (int r = lane; r < n_cells; r += 32) {
      const int slot = s_slot[r];
      int pos = r;
      const bool grouped = !s_start[r] || (r + 1 < n_cells && !s_start[r + 1]);
      if (grouped) {
        int g0 = r;
        while (!s_start[g0]) --g0;
        const double qr = s_qx[r];
        pos = g0;
        for (int j = g0; j < n_cells && (j == g0 || !s_start[j]); ++j) {
          const double qj = s_qx[j];
          pos += qj < qr || (qj == qr && s_slot[j] < slot);
        }
      }
      if (pos < take) s_sel[pos] = pixel_of(s_loc[slot]);
    }
  }
  __syncwarp();
  const int start = tile_off[tile];
  for (int k = lane; k < take; k += 32) {
    const int cell = s_sel[k];
    const int y = cell / W, x = cell - (cell / W) * W;
    const double ox = 0.25 + 0.5 * unit_interval_from_seed(seed, 3000003LL + 2LL * cell);
    const double oy = 0.25 + 0.5 * unit_interval_from_seed(seed, 3000004LL + 2LL * cell);
    out[start + k] = make_float2(static_cast<float>((x + ox) / W), static_cast<float>((y + oy) / H));
  }
  for (int dup = lane; dup < count - take; dup += 32) {
    const int cell = s_sel[dup % take];
    const int y = cell / W, x = cell - (cell / W) * W;
    const double bx = 0.25 + 0.5 * unit_interval_from_seed(seed, 3000003LL + 2LL * cell);
    const double byy = 0.25 + 0.5 * unit_interval_from_seed(seed, 3000004LL + 2LL * cell);
    const double slot = static_cast<double>(dup + 1);
    const double ox = 0.1 + 0.8 * fract_unit(bx + slot * 0.7548776662466927);
    const double oy = 0.1 + 0.8 * fract_unit(byy + slot * 0.5698402909980532);
    out[start + take + dup] =
        make_float2(static_cast<float>((x + ox) / W), static_cast<float>((y + oy) / H));
  }
}

// Per-tile weighted top-k (exponential-race priorities) + jittered emit.
// kArea bounds the tile area handled by this instantiation; two instantiations (tiles up to
// 16x16 and up to 32x32) are launched and each returns unless the device tile size is its own,
// so the common small-tile case gets small shared arrays and high occupancy.
template <int kArea>
__global__ void k_ed_emit(const float* __restrict__ density, const SamplerCtrl* c,
                          const int* __restrict__ tile_count, const int* __restrict__ tile_off,
                          int H, int W, int64_t seed, float2* out) {
  const int tile = blockIdx.x;
  const int ts_dev = c->tile_size;
  if ((kArea == 256) != (ts_dev <= 16)) return;
  if (tile >= c->tile_total) return;
  const int count = tile_count[tile];
  if (count <= 0) return;
  const bool valid_total = c->valid_total != 0;
  const int t = c->tile_size, tiles_x = c->tiles_x, tiles_y = c->tiles_y;
  const int shift_y = c->shift_y, shift_x = c->shift_x;

  __shared__ double s_prio[kArea];
  __shared__ int s_cell[kArea];
  __shared__ int s_n;
  const int tid = threadIdx.x;
  if (tid == 0) s_n = 0;
  __syncthreads();
  const int ty = tile / tiles_x, tx = tile - ty * tiles_x;
  const int sy_lo = ty * t, sy_hi = min((ty + 1) * t, H);
  const int sx_lo = tx * t, sx_hi = min((tx + 1) * t, W);
  const int tw = sx_hi - sx_lo;
  const int tp = (sy_hi - sy_lo) * tw;
  for (int l = tid; l < tp; l += blockDim.x) {
    const int sy = sy_lo + l / tw, sx = sx_lo + l % tw;
    const int y = (sy - shift_y + H) % H, x = (sx - shift_x + W) % W;
    if (min(sy / t, tiles_y - 1) * tiles_x + min(sx / t, tiles_x - 1) != tile) continue;
    const int idx = y * W + x;
    const float raw = density[idx];
    double w = (isfinite(raw) && raw > 0.f) ? static_cast<double>(raw) : 0.0;
    if (!valid_total) w = 1.0;
    double prio;
    if (w > 0.0 && isfinite(w)) {
      double u = unit_interval_from_seed(seed, 1000003LL + idx);
      if (u < 1e-300) u = 1e-300;
      prio = log(u) / w;
    } else {
      prio = unit_interval_from_seed(seed, 2000003LL + idx) - 1.0;
    }
    const int slot = atomicAdd(&s_n, 1);
    if (slot < kArea) {
      s_prio[slot] = prio;
      s_cell[slot] = idx;
    }
  }
  __syncthreads();
  const int n_cells = min(s_n, kArea);
  if (n_cells <= 0) return;
  const int take = min(count, n_cells);
  // Top-`take` by priority, in descending order (rank = number of higher priorities; ties by
  // slot), equivalent to the native thread-0 selection sort for distinct priorities.
  // Priorities are strictly negative (or -0), so p_j > p_a <=> bits(p_j) < bits(p_a) as
  // unsigned integers: integer compares instead of (1/64-rate) FP64 compares.
  __shared__ int s_sel[kArea];
  for (int a = tid; a < n_cells; a += blockDim.x) {
    const unsigned long long ka = __double_as_longlong(s_prio[a]);
    int rank = 0;
    for (int j = 0; j < n_cells; ++j) {
      const unsigned long long kj = __double_as_longlong(s_prio[j]);
      rank += (kj < ka) || (kj == ka && j < a);
    }
    if (rank < take) s_sel[rank] = s_cell[a];
  }
  __syncthreads();
  const int start = tile_off[tile];
  for (int k = tid; k < take; k += blockDim.x) {
    const int cell = s_sel[k];
    const int y = cell / W, x = cell - (cell / W) * W;
    const double ox = 0.25 + 0.5 * unit_interval_from_seed(seed, 3000003LL + 2LL * cell);
    const double oy = 0.25 + 0.5 * unit_interval_from_seed(seed, 3000004LL + 2LL * cell);
    out[start + k] = make_float2(static_cast<float>((x + ox) / W), static_cast<float>((y + oy) / H));
  }
  if (count > take) {
    for (int dup = tid; dup < count - take; dup += blockDim.x) {
      const int cell = s_sel[dup % take];
      const int y = cell / W, x = cell - (cell / W) * W;
      const double bx = 0.25 + 0.5 * unit_interval_from_seed(seed, 3000003LL + 2LL * cell);
      const double byy = 0.25 + 0.5 * unit_interval_from_seed(seed, 3000004LL + 2LL * cell);
      const double slot = static_cast<double>(dup + 1);
      const double ox = 0.1 + 0.8 * fract_unit(bx + slot * 0.7548776662466927);
      const double oy = 0.1 + 0.8 * fract_unit(byy + slot * 0.5698402909980532);
      out[start + take + dup] =
          make_float2(static_cast<float>((x + ox) / W), static_cast<float>((y + oy) / H));
    }
  }
}

// ------------------------------------------------------------------ Voronoi assignment
// Exact nearest-seed assignment over a grid hash (replaces jump flooding + lost-point
// recovery). Every pixel centre (x + 0.5, y + 0.5) gets
//     argmin_i (d_i, i),  d_i = fl(fl(dx_i^2) + fl(dy_i^2)),  dx_i = fl(x + 0.5 - px_i),
// in round-to-nearest single precision (seed_d2), with px_i the clamped pixel-space seed position
// of the JFA code. Ties go to the lowest seed index, so the result is unique and independent of
// launch order; JFA's d < best scan order is not, and JFA itself is approximate.
//
// Layout: seeds are binned by their pixel floor into square cells of side s pixels (s from the
// device-side count, ~ mean seed spacing). Cells are numbered tile-major over 64x64-cell tiles
// and in Morton order inside a tile, so every aligned block of 4^l x 4^l cells (l = 0..3) is one
// contiguous range of the counting-sorted seed array: a 4-level quadtree from one sort.
//
// Query (k_grid_query): one thread per 2x2 pixel quad scans rings 0..2 of level-0 cells around
// its cell; a pixel is resolved once its best distance is strictly below the distance to the
// part of the image outside the covered square. Pixels still open (sparse regions) are queued
// by 8x4 tile and finished by one warp per tile (k_grid_query_open): rings at levels 1 (4s),
// 2 (16s) and 3 (64s, unbounded), nearest-first quadtree descent into non-empty nodes. Nodes and
// cells whose box lower bound exceeds the best distance are skipped. The stopping and pruning
// tests use the same rounded arithmetic as seed_d2, which is monotone in |dx| and |dy|, so
// neither can drop a seed that is closer or tied: the result equals a brute-force argmin over
// all seeds bit for bit (0 mismatches over 77M pixel assignments on Kodak point sets and on
// synthetic maps with large empty regions).
//
// The query kernels also accumulate the per-cell density sums (masses, Lloyd centroids), so the
// owner map is only materialised where the merge rounds need it (adjacency).
//
// All kernels launch over the planned extents, take the active count by pointer and gate on
// device counts (`gate`, nullable, skips the work when <= 0): no host sync, graph-capturable.
__device__ inline bool gated(const int* gate) { return gate != nullptr && *gate <= 0; }

constexpr int kGridMinCell = 2;   // smallest cell side (pixels): bounds the planned key count
constexpr int kGridTileLog = 6;   // 64x64 cells per tile
constexpr int kGridTile = 1 << kGridTileLog;
constexpr int kGridLevels = 4;    // cell sides s, 4s, 16s, 64s (= one tile)
constexpr int kGridRingCap = 2;   // rings per level before moving up (top level: unbounded)
constexpr float kGridCellScale = 1.0f;  // cell side / mean seed spacing (rounded up to even)

__device__ __forceinline__ float seed_d2(float dx, float dy) {
  return __fadd_rn(__fmul_rn(dx, dx), __fmul_rn(dy, dy));
}

// Spread the low 6 bits of v to the even bit positions.
__device__ __forceinline__ unsigned morton_spread6(unsigned v) {
  v &= 0x3fu;
  v = (v | (v << 4)) & 0x30fu;  // 00 0011 0000 1111
  v = (v | (v << 2)) & 0x333u;  // 0011 0011 0011
  v = (v | (v << 1)) & 0x555u;  // 0101 0101 0101
  return v;
}

__device__ __forceinline__ int grid_key(const GridCtrl& g, int fx, int fy) {
  const int tx = fx >> kGridTileLog, ty = fy >> kGridTileLog;
  const unsigned lx = static_cast<unsigned>(fx) & (kGridTile - 1);
  const unsigned ly = static_cast<unsigned>(fy) & (kGridTile - 1);
  return ((ty * g.tiles_x + tx) << (2 * kGridTileLog)) +
         static_cast<int>(morton_spread6(lx) | (morton_spread6(ly) << 1));
}

// Grid for n seeds: cell side ~ mean seed spacing, clamped to [kGridMinCell, max(H, W)].
__device__ __forceinline__ GridCtrl grid_params(int n, int H, int W) {
  GridCtrl g;
  g.n = max(0, n);
  const float spacing = sqrtf(static_cast<float>(H) * W / fmaxf(1.f, static_cast<float>(g.n)));
  // even sides keep every 2x2 pixel quad inside one cell (k_grid_query)
  int s = 2 * static_cast<int>(ceilf(0.5f * spacing * kGridCellScale));
  s = max(kGridMinCell, min(s, 2 * ((max(H, W) + 1) / 2)));
  g.s = s;
  g.gx = (W + s - 1) / s;
  g.gy = (H + s - 1) / s;
  g.tiles_x = (g.gx + kGridTile - 1) / kGridTile;
  g.tiles_y = (g.gy + kGridTile - 1) / kGridTile;
  g.n_keys = g.tiles_x * g.tiles_y * kGridTile * kGridTile;
  return g;
}

// Counting sort of the seeds by cell key, in three kernels: count (per-cell ranks + per-tile
// totals), scan (one block per 64x64-cell tile: tile offset from the tile totals + block scan
// of its 4096 cells), scatter. The scatter re-zeroes the counts it consumed, so the next
// assignment starts from zero without a reset kernel (k_ed_aggregate zeroes them once per run).
// The count kernel also zeroes this assignment's accumulators and open-tile state, and, in the
// merge rounds, resets the neighbour sets and the first two matching tmin buffers.
// Distinct Voronoi neighbours per point, collected lock-free into a small open-addressed set
// (kNbrSlots per point, -1 = empty); k_rows_and_keys keeps the knn_k lowest-mass ones. Same
// result as the native locked top-k insertion (dedup, keep lowest masses, ascending order,
// pad with self) unless a cell has more than kNbrSlots distinct neighbours.
constexpr int kNbrSlots = 16;

constexpr int kScanThreads = 1024;
constexpr int kTileKeys = kGridTile * kGridTile;

struct RoundReset {
  int* nbr;                                   // nullable: even merge rounds only
  unsigned long long* tmin0, *tmin1, *tmin2;  // nullable: merge rounds only
};

__global__ void k_grid_scan(const int* n_ptr, const int* gate, const int* __restrict__ cnt,
                            const int* __restrict__ tile_tot, int* start, int max_keys) {
  using BlockScan = cub::BlockScan<int, kScanThreads>;
  __shared__ typename BlockScan::TempStorage tmp;
  __shared__ int s_off;
  if (gated(gate)) return;
  const int t = blockIdx.x, tid = threadIdx.x;
  if (tid < 32) {
    int v = 0;
    for (int j = tid; j < t; j += 32) v += tile_tot[j];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (tid == 0) s_off = v;
  }
  constexpr int kPer = kTileKeys / kScanThreads;
  const int base = t * kTileKeys + tid * kPer;
  int c[kPer], sum = 0;
#pragma unroll
  for (int q = 0; q < kPer; ++q) {
    c[q] = cnt[base + q];
    sum += c[q];
  }
  int before;
  BlockScan(tmp).ExclusiveSum(sum, before);
  __syncthreads();
  before += s_off;
#pragma unroll
  for (int q = 0; q < kPer; ++q) {
    start[base + q] = before;
    before += c[q];
  }
  if (t == gridDim.x - 1 && tid == kScanThreads - 1) start[max_keys] = before;
}

__global__ void k_grid_count(const float2* __restrict__ pts, const int* n_ptr, const int* gate,
                             GridCtrl* gc, int* cnt, int* tile_tot, int* key, int* rank,
                             float2* pix, int H, int W, int cap, float* z0, float* z1, float* z2,
                             int* open_count, unsigned* open_mask, int open_tiles, RoundReset rr) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (gated(gate)) return;
  if (i == 0) *open_count = 0;
  if (i < open_tiles) open_mask[i] = 0u;
  if (i < cap) {
    if (z0) z0[i] = 0.f;
    if (z1) z1[i] = 0.f;
    if (z2) z2[i] = 0.f;
  }
  const GridCtrl g = grid_params(*n_ptr, H, W);
  if (i == 0) *gc = g;  // diagnostics only
  const bool active = i < g.n && i < cap;
  int k = -1;
  if (active) {
    if (rr.tmin0) {
      rr.tmin0[i] = ~0ULL;
      rr.tmin1[i] = ~0ULL;
      rr.tmin2[i] = ~0ULL;
    }
    if (rr.nbr) {
      int4* row = reinterpret_cast<int4*>(rr.nbr + static_cast<size_t>(i) * kNbrSlots);
#pragma unroll
      for (int q = 0; q < kNbrSlots / 4; ++q) row[q] = make_int4(-1, -1, -1, -1);
    }
    const float2 p = pts[i];
    const float2 q = make_float2(fminf(fmaxf(p.x * W, 0.f), W - 1e-3f),
                                 fminf(fmaxf(p.y * H, 0.f), H - 1e-3f));
    const int px = min(max(static_cast<int>(floorf(q.x)), 0), W - 1);
    const int py = min(max(static_cast<int>(floorf(q.y)), 0), H - 1);
    k = grid_key(g, px / g.s, py / g.s);
    key[i] = k;
    rank[i] = atomicAdd(&cnt[k], 1);
    pix[i] = q;
  }
  // per-tile totals, warp-aggregated (consecutive seeds are spatially close)
  const unsigned am = __ballot_sync(0xffffffffu, active);
  if (active) {
    const int tile = k / kTileKeys;
    const unsigned peers = __match_any_sync(am, tile);
    if ((threadIdx.x & 31) == __ffs(peers) - 1) atomicAdd(&tile_tot[tile], __popc(peers));
  }
}

__global__ void k_grid_scatter(const int* n_ptr, const int* gate, const int* __restrict__ start,
                               const int* __restrict__ key, const int* __restrict__ rank,
                               const float2* __restrict__ pix, float2* spos, int* sidx, int* cnt,
                               int n_cnt, int cap) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (gated(gate)) return;
  if (i < n_cnt) cnt[i] = 0;  // counts + tile totals, consumed by k_grid_scan
  if (i >= cap || i >= *n_ptr) return;
  const int j = start[key[i]] + rank[i];
  spos[j] = pix[i];
  sidx[j] = i;
}

// Lower bound of seed_d2 over seeds whose pixel floor lies in [x0, x1) x [y0, y1).
__device__ __forceinline__ float box_lb(float mx, float my, float x0, float x1, float y0,
                                        float y1) {
  const float bx = fmaxf(fmaxf(x0 - mx, mx - x1), 0.f);
  const float by = fmaxf(fmaxf(y0 - my, my - y1), 0.f);
  return seed_d2(bx, by);
}

// What an assignment produces: the owner map (merge rounds: adjacency) and/or per-cell sums of a
// pixel weight, accumulated by the query kernels themselves (no separate pass over the owner
// map). Lloyd mode (acc_x != nullptr) skips weights <= 0 and also sums weight * pixel centre in
// normalised coordinates, as the former k_lloyd_accumulate; mass mode adds every weight, as the
// former k_mass_scatter.
struct AssignOut {
  int* owner;           // nullable
  const float* dens;    // nullable: no accumulation
  float* acc_w;
  float* acc_x;         // nullable: mass mode
  float* acc_y;
};

__device__ __forceinline__ bool pixel_weight(const AssignOut& o, int px, int py, int W, int H,
                                             float& w, float& xw, float& yw) {
  w = o.dens[py * W + px];
  if (o.acc_x) {
    if (w <= 0.f) return false;
    xw = w * ((px + 0.5f) / W);
    yw = w * ((py + 0.5f) / H);
  } else if (w == 0.f) {
    return false;
  }
  return true;
}

__device__ __forceinline__ void add_weight(const AssignOut& o, int owner, float w, float xw,
                                           float yw) {
  atomicAdd(&o.acc_w[owner], w);
  if (o.acc_x) {
    atomicAdd(&o.acc_x[owner], xw);
    atomicAdd(&o.acc_y[owner], yw);
  }
}

// One thread per 2x2 pixel quad. Cell sides are even, so a quad lies in a single level-0
// cell and its four pixels share the ring sequence: every staged seed is loaded once for four
// pixels. Each pixel keeps its own best and its own stopping test; pixels still open after
// kQuadRings level-0 rings are queued, by 8x4 pixel tile, for k_grid_query_open (sparse
// regions).
constexpr int kQuadRings = 2;
constexpr int kQBX = 16, kQBY = 8;  // quads per block
constexpr int kOpenBlocks = 1024;    // k_grid_query_open grid (warp-stride over open tiles)
constexpr int kOpenTW = 8, kOpenTH = 4;  // open tile: one warp, one lane per pixel
// The open search starts at level 1: its rings 0..1 already cover the level-0 rings the quad
// pass gave up on (any start level is exact; this one is the fastest on Kodak and synthetic
// sparse maps).
constexpr int kOpenStartLevel = 1;

__global__ void __launch_bounds__(kQBX * kQBY)
    k_grid_query(const int* n_ptr, const int* gate, const int* __restrict__ start,
                 const float2* __restrict__ spos, const int* __restrict__ sidx, AssignOut out,
                 int* open_count, int* open_tiles, unsigned* open_mask, float2* open_best,
                 int H, int W, int* stats) {
  if (gated(gate)) return;
  const int qx = blockIdx.x * kQBX + threadIdx.x, qy = blockIdx.y * kQBY + threadIdx.y;
  const int x = 2 * qx, y = 2 * qy;
  if (x >= W || y >= H) return;
  const GridCtrl g = grid_params(*n_ptr, H, W);
  const bool has_x1 = x + 1 < W, has_y1 = y + 1 < H;
  if (g.n <= 0) {  // no seeds: nothing owned, nothing accumulated
    for (int p = 0; p < 4; ++p) {
      const int px = x + (p & 1), py = y + (p >> 1);
      if (out.owner && px < W && py < H) out.owner[py * W + px] = -1;
    }
    return;
  }
  const unsigned in_image =
      1u | (has_x1 ? 2u : 0u) | (has_y1 ? 4u : 0u) | (has_x1 && has_y1 ? 8u : 0u);
  const float mx0 = x + 0.5f, mx1 = x + 1.5f, my0 = y + 0.5f, my1 = y + 1.5f;
  float bd[4] = {INFINITY, INFINITY, INFINITY, INFINITY};
  int bo[4] = {INT_MAX, INT_MAX, INT_MAX, INT_MAX};
  unsigned open = in_image;  // pixels outside the image count as resolved
  const int cx = x / g.s, cy = y / g.s;
  for (int r = 0; r <= kQuadRings && open; ++r) {
    const int x0 = cx - r, x1 = cx + r, y0 = cy - r, y1 = cy + r;
    const float bmax = fmaxf(fmaxf(bd[0], bd[1]), fmaxf(bd[2], bd[3]));
    for (int yy = max(y0, 0); yy <= min(y1, g.gy - 1); ++yy) {
      const int step = (yy == y0 || yy == y1) ? 1 : (x1 - x0);
      for (int xx = x0; xx <= x1; xx += step) {
        if (xx < 0 || xx >= g.gx) continue;
        const float fx0 = static_cast<float>(xx * g.s), fy0 = static_cast<float>(yy * g.s);
        // lower bound over the quad's pixel centres [mx0, mx1] x [my0, my1]
        const float gx = fmaxf(fmaxf(fx0 - mx1, mx0 - (fx0 + g.s)), 0.f);
        const float gy = fmaxf(fmaxf(fy0 - my1, my0 - (fy0 + g.s)), 0.f);
        if (seed_d2(gx, gy) > bmax) continue;
        const int k = grid_key(g, xx, yy);
        const int e = start[k + 1];
        for (int j = start[k]; j < e; ++j) {
          const float2 q = spos[j];
          const int id = sidx[j];
          const float ax = __fmul_rn(mx0 - q.x, mx0 - q.x), bx = __fmul_rn(mx1 - q.x, mx1 - q.x);
          const float ay = __fmul_rn(my0 - q.y, my0 - q.y), by = __fmul_rn(my1 - q.y, my1 - q.y);
          const float d[4] = {__fadd_rn(ax, ay), __fadd_rn(bx, ay), __fadd_rn(ax, by),
                              __fadd_rn(bx, by)};
#pragma unroll
          for (int p = 0; p < 4; ++p) {
            if (d[p] < bd[p] || (d[p] == bd[p] && id < bo[p])) {
              bd[p] = d[p];
              bo[p] = id;
            }
          }
        }
      }
    }
    // per-pixel stopping test against the uncovered part of the image
#pragma unroll
    for (int p = 0; p < 4; ++p) {
      if (!(open & (1u << p))) continue;
      const float mx = (p & 1) ? mx1 : mx0, my = (p >> 1) ? my1 : my0;
      float bnd = INFINITY;
      if (x0 > 0) bnd = fminf(bnd, mx - static_cast<float>(x0 * g.s));
      if (x1 < g.gx - 1) bnd = fminf(bnd, static_cast<float>((x1 + 1) * g.s) - mx);
      if (y0 > 0) bnd = fminf(bnd, my - static_cast<float>(y0 * g.s));
      if (y1 < g.gy - 1) bnd = fminf(bnd, static_cast<float>((y1 + 1) * g.s) - my);
      if (bnd == INFINITY || bd[p] < __fmul_rn(bnd, bnd)) {
        open &= ~(1u << p);
        if (stats) atomicAdd(&stats[min(r, 7)], 1);
      }
    }
  }
  unsigned bits = 0;  // open pixels of this quad in its 8x4 open tile
#pragma unroll
  for (int p = 0; p < 4; ++p) {
    const int px = x + (p & 1), py = y + (p >> 1);
    if (!(in_image & (1u << p))) continue;
    if (out.owner) out.owner[py * W + px] = bo[p];
    if (open & (1u << p)) {
      open_best[py * W + px] = make_float2(bd[p], __int_as_float(bo[p]));
      bits |= 1u << ((py & (kOpenTH - 1)) * kOpenTW + (px & (kOpenTW - 1)));
    }
  }
  if (bits) {
    const int tile = (y / kOpenTH) * ((W + kOpenTW - 1) / kOpenTW) + x / kOpenTW;
    if (atomicOr(&open_mask[tile], bits) == 0u) open_tiles[atomicAdd(open_count, 1)] = tile;
  }
  if (!out.dens) return;
  // Per-cell sums of the resolved pixels; pixels of the quad with the same owner are summed
  // before the atomics (open pixels accumulate in k_grid_query_open).
  float w[4], xw[4], yw[4];
  unsigned todo = 0;
#pragma unroll
  for (int p = 0; p < 4; ++p) {
    if (((in_image & ~open) >> p) & 1u) {
      if (pixel_weight(out, x + (p & 1), y + (p >> 1), W, H, w[p], xw[p], yw[p]))
        todo |= 1u << p;
    }
  }
#pragma unroll
  for (int p = 0; p < 4; ++p) {
    if (!(todo & (1u << p))) continue;
    float sw = w[p], sx = xw[p], sy = yw[p];
#pragma unroll
    for (int q = p + 1; q < 4; ++q) {
      if ((todo & (1u << q)) && bo[q] == bo[p]) {
        sw += w[q];
        sx += xw[q];
        sy += yw[q];
        todo &= ~(1u << q);
      }
    }
    add_weight(out, bo[p], sw, sx, sy);
  }
}

// Pixels left open by k_grid_query (sparse regions, far from every seed): one warp per 8x4
// pixel tile, one lane per pixel. The warp walks one shared, warp-uniform search for the whole
// tile (rings around the tile's cells at levels 0..3, nearest-first quadtree descent), pruning
// nodes against the tile's pixel-centre box and the largest best distance of the lanes still
// searching; every scanned seed is broadcast to all lanes, which keep their own (bd, bo) and
// their own stopping test. Same arithmetic and tie-break as everywhere else: exact.
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}

__device__ __forceinline__ void tile_visit(const GridCtrl& g, const int* __restrict__ start,
                                           const float2* __restrict__ spos,
                                           const int* __restrict__ sidx, float mx, float my,
                                           float bx0, float bx1, float by0, float by1,
                                           bool live, int level, int cx, int cy, float& bd,
                                           int& bo) {
  constexpr int kStack = 1 + 3 * 2 * (kGridLevels - 1);
  int st_m[kStack], st_x[kStack], st_y[kStack];
  st_m[0] = 2 * level;
  st_x[0] = cx << (2 * level);
  st_y[0] = cy << (2 * level);
  int sp = 1;
  float bmax = warp_max(live ? bd : -INFINITY);
  const float tcx = 0.5f * (bx0 + bx1), tcy = 0.5f * (by0 + by1);
  while (sp > 0) {
    --sp;
    const int m = st_m[sp], fx = st_x[sp], fy = st_y[sp];
    const int side = 1 << m;
    if (fx >= g.gx || fy >= g.gy) continue;
    const float x0 = static_cast<float>(fx * g.s), y0 = static_cast<float>(fy * g.s);
    const float x1 = static_cast<float>((fx + side) * g.s);
    const float y1 = static_cast<float>((fy + side) * g.s);
    const float gx = fmaxf(fmaxf(x0 - bx1, bx0 - x1), 0.f);
    const float gy = fmaxf(fmaxf(y0 - by1, by0 - y1), 0.f);
    if (seed_d2(gx, gy) > bmax) continue;
    const int k0 = grid_key(g, fx, fy);
    const int a = start[k0];
    const int b = start[k0 + side * side];
    if (b - a <= 32 || m == 0) {
      for (int j = a; j < b; ++j) {
        const float2 q = spos[j];
        const int id = sidx[j];
        const float d = seed_d2(mx - q.x, my - q.y);
        if (d < bd || (d == bd && id < bo)) {
          bd = d;
          bo = id;
        }
      }
      if (a < b) bmax = warp_max(live ? bd : -INFINITY);
      continue;
    }
    // children: lane c < 4 probes child c (count + box bound) in parallel
    const int h = side >> 1;
    const int lane = threadIdx.x & 31;
    const int ccx = fx + (lane & 1) * h, ccy = fy + ((lane >> 1) & 1) * h;
    bool keep = false;
    float key = INFINITY;
    if (lane < 4 && ccx < g.gx && ccy < g.gy) {
      const float cx0 = static_cast<float>(ccx * g.s), cy0 = static_cast<float>(ccy * g.s);
      const float cx1 = cx0 + h * g.s, cy1 = cy0 + h * g.s;
      const float cgx = fmaxf(fmaxf(cx0 - bx1, bx0 - cx1), 0.f);
      const float cgy = fmaxf(fmaxf(cy0 - by1, by0 - cy1), 0.f);
      if (seed_d2(cgx, cgy) <= bmax) {
        const int ck = grid_key(g, ccx, ccy);
        keep = start[ck] < start[ck + h * h];
        key = box_lb(tcx, tcy, cx0, cx1, cy0, cy1);  // nearest-first for the tile centre
      }
    }
    // push kept children farthest first
    unsigned todo = __ballot_sync(0xffffffffu, keep);
    while (todo) {
      float kk = (todo >> lane) & 1u ? key : -INFINITY;
      int kl = lane;
#pragma unroll
      for (int o = 2; o > 0; o >>= 1) {  // argmax over lanes 0..3
        const float ok = __shfl_xor_sync(0xffffffffu, kk, o);
        const int ol = __shfl_xor_sync(0xffffffffu, kl, o);
        if (ok > kk || (ok == kk && ol < kl)) {
          kk = ok;
          kl = ol;
        }
      }
      kl = __shfl_sync(0xffffffffu, kl, 0);
      st_m[sp] = m - 1;
      st_x[sp] = fx + (kl & 1) * h;
      st_y[sp] = fy + ((kl >> 1) & 1) * h;
      ++sp;
      todo &= ~(1u << kl);
    }
  }
}

// Cell c of the ring r around the cell rectangle [xa, xb] x [ya, yb] (r = 0: the rectangle):
// top row, bottom row, then the left and right columns without the corners.
__device__ __forceinline__ void rect_ring_cell(int c, int r, int xa, int xb, int ya, int yb,
                                               int& xx, int& yy) {
  const int x0 = xa - r, y0 = ya - r, wd = xb - xa + 1 + 2 * r, hd = yb - ya + 1 + 2 * r;
  if (r == 0) {
    xx = x0 + c % wd;
    yy = y0 + c / wd;
  } else if (c < wd) {
    xx = x0 + c;
    yy = y0;
  } else if (c < 2 * wd) {
    xx = x0 + c - wd;
    yy = y0 + hd - 1;
  } else if (c < 2 * wd + hd - 2) {
    xx = x0;
    yy = y0 + 1 + (c - 2 * wd);
  } else {
    xx = x0 + wd - 1;
    yy = y0 + 1 + (c - 2 * wd - (hd - 2));
  }
}

__global__ void k_grid_query_open(const int* n_ptr, const int* gate, const int* __restrict__ start,
                                  const float2* __restrict__ spos, const int* __restrict__ sidx,
                                  const int* open_count, const int* __restrict__ open_tiles,
                                  const unsigned* __restrict__ open_mask,
                                  const float2* __restrict__ open_best, AssignOut out, int H,
                                  int W, int* stats) {
  if (gated(gate)) return;
  const int count = *open_count;
  if (count == 0) return;
  const GridCtrl g = grid_params(*n_ptr, H, W);
  const int lane = threadIdx.x & 31;
  const int tiles_w = (W + kOpenTW - 1) / kOpenTW;
  const int warps = gridDim.x * (blockDim.x >> 5);
  for (int e = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5); e < count; e += warps) {
    const int tile = open_tiles[e];
    const unsigned mask = open_mask[tile];
    const int tx0 = (tile % tiles_w) * kOpenTW, ty0 = (tile / tiles_w) * kOpenTH;
    const int px = tx0 + (lane & (kOpenTW - 1)), py = ty0 + lane / kOpenTW;
    const bool mine = (mask >> lane) & 1u;
    float bd = -INFINITY;  // lanes without an open pixel never update and never bound the walk
    int bo = INT_MAX;
    if (mine) {
      const float2 b = open_best[py * W + px];
      bd = b.x;
      bo = __float_as_int(b.y);
    }
    bool live = mine;
    const float mx = px + 0.5f, my = py + 0.5f;
    const int qx1 = min(tx0 + kOpenTW, W) - 1, qy1 = min(ty0 + kOpenTH, H) - 1;
    const float bx0 = tx0 + 0.5f, bx1 = qx1 + 0.5f, by0 = ty0 + 0.5f, by1 = qy1 + 0.5f;
    for (int level = kOpenStartLevel; level < kGridLevels && __any_sync(0xffffffffu, live); ++level) {
      const int cs = g.s << (2 * level);
      const int gx = (g.gx + (1 << (2 * level)) - 1) >> (2 * level);
      const int gy = (g.gy + (1 << (2 * level)) - 1) >> (2 * level);
      const int cxa = tx0 / cs, cxb = qx1 / cs, cya = ty0 / cs, cyb = qy1 / cs;
      const int rmax = level + 1 == kGridLevels ? INT_MAX : kGridRingCap;
      for (int r = 0; r <= rmax; ++r) {
        const int x0 = cxa - r, x1 = cxb + r, y0 = cya - r, y1 = cyb + r;
        const int wd = x1 - x0 + 1, hd = y1 - y0 + 1;
        const int ncell = r == 0 ? wd * hd : 2 * wd + 2 * (hd - 2);
        const float bmax = warp_max(live ? bd : -INFINITY);
        for (int base = 0; base < ncell; base += 32) {
          // lanes probe 32 ring cells at once: non-empty and within reach?
          int xx = -1, yy = -1;
          bool keep = false;
          float key = INFINITY;
          if (base + lane < ncell) {
            rect_ring_cell(base + lane, r, cxa, cxb, cya, cyb, xx, yy);
            if (xx >= 0 && xx < gx && yy >= 0 && yy < gy) {
              const float cx0 = static_cast<float>(xx * cs), cy0 = static_cast<float>(yy * cs);
              const float cgx = fmaxf(fmaxf(cx0 - bx1, bx0 - (cx0 + cs)), 0.f);
              const float cgy = fmaxf(fmaxf(cy0 - by1, by0 - (cy0 + cs)), 0.f);
              if (seed_d2(cgx, cgy) <= bmax) {
                const int fx = xx << (2 * level), fy = yy << (2 * level);
                const int k0 = grid_key(g, fx, fy);
                keep = start[k0] < start[k0 + (1 << (4 * level))];
                key = box_lb(0.5f * (bx0 + bx1), 0.5f * (by0 + by1), cx0, cx0 + cs, cy0,
                             cy0 + cs);
              }
            }
          }
          unsigned todo = __ballot_sync(0xffffffffu, keep);
          while (todo) {  // nearest kept cell first
            float kk = (todo >> lane) & 1u ? key : INFINITY;
            int kl = lane;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
              const float ok = __shfl_xor_sync(0xffffffffu, kk, o);
              const int ol = __shfl_xor_sync(0xffffffffu, kl, o);
              if (ok < kk || (ok == kk && ol < kl)) {
                kk = ok;
                kl = ol;
              }
            }
            const int vx = __shfl_sync(0xffffffffu, xx, kl), vy = __shfl_sync(0xffffffffu, yy, kl);
            tile_visit(g, start, spos, sidx, mx, my, bx0, bx1, by0, by1, live, level, vx, vy, bd,
                       bo);
            todo &= ~(1u << kl);
          }
        }
        if (live) {
          float bnd = INFINITY;
          if (x0 > 0) bnd = fminf(bnd, mx - static_cast<float>(x0 * cs));
          if (x1 < gx - 1) bnd = fminf(bnd, static_cast<float>((x1 + 1) * cs) - mx);
          if (y0 > 0) bnd = fminf(bnd, my - static_cast<float>(y0 * cs));
          if (y1 < gy - 1) bnd = fminf(bnd, static_cast<float>((y1 + 1) * cs) - my);
          if (bnd == INFINITY || bd < __fmul_rn(bnd, bnd)) {
            live = false;
            if (stats) atomicAdd(&stats[8 * (level + 1) + min(r, 7)], 1);
          }
        }
        if (!__any_sync(0xffffffffu, live)) break;
      }
    }
    if (mine) {
      if (out.owner) out.owner[py * W + px] = bo;
      float w, xw, yw;
      if (out.dens && pixel_weight(out, px, py, W, H, w, xw, yw)) add_weight(out, bo, w, xw, yw);
    }
  }
}

// Exact Voronoi assignment of pts[0 : *n_ptr]: zeros the accumulators of `out` (planned
// capacity), then fills out.owner and / or the per-cell weight sums.
cudaError_t assign(SamplerBuffers& b, const float2* pts, const int* n_ptr, const int* gate,
                   int cap, const AssignOut& out, cudaStream_t s,
                   RoundReset rr = RoundReset{nullptr, nullptr, nullptr, nullptr}) {
  const int H = b.cfg.H, W = b.cfg.W;
  const int open_tiles = ((W + kOpenTW - 1) / kOpenTW) * ((H + kOpenTH - 1) / kOpenTH);
  const int key_tiles = b.grid_max_keys / kTileKeys;
  int* tile_tot = b.grid_cnt + b.grid_max_keys + 1;
  const int n_cnt = b.grid_max_keys + 1 + key_tiles;
  k_grid_count<<<grid_for(std::max(cap, open_tiles)), 256, 0, s>>>(
      pts, n_ptr, gate, b.grid, b.grid_cnt, tile_tot, b.grid_key, b.grid_rank, b.pts_pixel, H, W,
      cap, out.dens ? out.acc_w : nullptr, out.acc_x, out.acc_y, b.grid_open_count,
      b.grid_open_mask, open_tiles, rr);
  k_grid_scan<<<key_tiles, kScanThreads, 0, s>>>(n_ptr, gate, b.grid_cnt, tile_tot, b.grid_start,
                                                  b.grid_max_keys);
  k_grid_scatter<<<grid_for(std::max(cap, n_cnt)), 256, 0, s>>>(
      n_ptr, gate, b.grid_start, b.grid_key, b.grid_rank, b.pts_pixel, b.grid_pos, b.grid_idx,
      b.grid_cnt, n_cnt, cap);
  const dim3 qg(((W + 1) / 2 + kQBX - 1) / kQBX, ((H + 1) / 2 + kQBY - 1) / kQBY);
  k_grid_query<<<qg, dim3(kQBX, kQBY), 0, s>>>(n_ptr, gate, b.grid_start, b.grid_pos, b.grid_idx,
                                               out, b.grid_open_count, b.grid_open_tiles,
                                               b.grid_open_mask, b.grid_open_best, H, W, nullptr);
  k_grid_query_open<<<kOpenBlocks, 128, 0, s>>>(n_ptr, gate, b.grid_start, b.grid_pos, b.grid_idx,
                                                b.grid_open_count, b.grid_open_tiles,
                                                b.grid_open_mask, b.grid_open_best, out, H, W,
                                                nullptr);
#ifdef GFUSED_SAMPLER_ASSIGN_HOOK
  GFUSED_SAMPLER_ASSIGN_HOOK(b, pts, n_ptr, gate, out, s);
#endif
  return cudaSuccess;
}

// ------------------------------------------------------------------ merge round
constexpr int kSelDigit = 12;
constexpr int kSelBins = 1 << kSelDigit;
constexpr int kSelList = 4096;
constexpr int kSelThreads = 1024;
constexpr int kSelPasses = 3;  // launched select passes (digit, digit, gather typically)
constexpr unsigned long long kNoProp = ~0ULL;  // matching: no proposal

__global__ void k_round_schedule(SamplerCtrl* c, SelCtrl* sc, int round, int rounds, double frac,
                                 int key_bits) {
  const int n_in = round == 0 ? c->n_init : c->n_cur;
  const int excess = max(0, n_in - c->n);
  int cap;
  if (round + 1 == rounds) {
    cap = excess;
  } else {
    cap = excess > 0 ? max(1, static_cast<int>(static_cast<double>(n_in) * frac)) : 0;
  }
  c->round_n_in = n_in;
  const int remove = min(min(excess, cap), max(0, n_in - 1));
  c->round_remove = remove;
  c->nn_count = 0;
  sc->prefix = 0ULL;
  sc->thr = 0ULL;  // no candidates until the select finishes
  sc->rank = remove;
  sc->bits = key_bits;
  sc->list = n_in <= kSelList;
  sc->done = remove <= 0;
  sc->list_n = 0;
}


// Set entries carry the round parity in bit 30 (index | tag << 30): a slot is free unless it
// holds an entry of the current tag, so odd rounds reuse the table without the reset that
// even rounds do (k_grid_count). Indices are < 2^30.
__device__ __forceinline__ bool nbr_live(int e, int tag) { return e >= 0 && (e >> 30) == tag; }

__device__ inline void nbr_insert(int* nbr, int src, int dst, int tag) {
  int* row = nbr + static_cast<size_t>(src) * kNbrSlots;
  const int enc = dst | (tag << 30);
  const unsigned h = (static_cast<unsigned>(dst) * 2654435761u) >> 28;
  for (int p = 0; p < kNbrSlots; ++p) {
    int* slot = row + ((h + p) & (kNbrSlots - 1));
    int old = *slot;
    while (true) {
      if (old == enc) return;
      if (nbr_live(old, tag)) break;  // taken by another neighbour: probe on
      const int prev = atomicCAS(slot, old, enc);
      if (prev == old) return;
      old = prev;  // lost the race: re-examine the new value
    }
  }
}

// Pixel-adjacency of the owner map: every 4-neighbour pixel pair with different owners links
// the two cells (both directions). Pairs repeat along a cell boundary; each warp inserts every
// distinct pair once (match_any on the packed pair), which removes most hash probes.
__device__ __forceinline__ void adj_edge(int* nbr, unsigned am, bool has, int a, int b, int tag) {
  const unsigned long long pair =
      has ? (static_cast<unsigned long long>(static_cast<unsigned>(min(a, b))) << 32) |
                static_cast<unsigned>(max(a, b))
          : ~0ULL;
  const unsigned peers = __match_any_sync(am, pair);
  const int lane = (threadIdx.y * blockDim.x + threadIdx.x) & 31;
  if (has && lane == __ffs(peers) - 1) {
    nbr_insert(nbr, a, b, tag);
    nbr_insert(nbr, b, a, tag);
  }
}

__global__ void k_adjacency(const int* __restrict__ owner, const SamplerCtrl* c, int* nbr, int H,
                            int W, int tag) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
  if (c->round_remove <= 0) return;  // uniform
  const bool in = x < W && y < H;
  const unsigned am = __ballot_sync(0xffffffffu, in);
  if (!in) return;
  const int pix = y * W + x;
  const int src = owner[pix];
  const int dr = x + 1 < W ? owner[pix + 1] : -1;
  const int dd = y + 1 < H ? owner[pix + W] : -1;
  adj_edge(nbr, am, src >= 0 && dr >= 0 && dr != src, src, dr, tag);
  adj_edge(nbr, am, src >= 0 && dd >= 0 && dd != src, src, dd, tag);
}

// ---- candidate selection
// The matching needs (a) a strict priority order over the points, ascending mass with ties to
// the lower index (the order of the former stable argsort), and (b) the candidate set = the
// round_remove first points in that order. Both come from one 64-bit key per point,
//     key(i) = twiddle(mass_i) << ib | i      (ib = index bits of the planned capacity),
// where twiddle is CUB's order-preserving float transform (-0 folded onto +0, as CUB does), so
// comparing keys is comparing argsort ranks. Instead of sorting, a radix select finds the key
// of rank round_remove: 12-bit digit histograms from the top, narrowing a prefix, until the
// bucket holding that rank has <= kSelList keys, which are gathered and bitonic-sorted in one
// block. cand(i) = key(i) < threshold. Every pass launches over fixed extents and returns
// early once the threshold is known.

__device__ __forceinline__ unsigned long long sel_key(float m, int i, int ib) {
  unsigned u = __float_as_uint(m);
  if (u == 0x80000000u) u = 0u;
  const unsigned t = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
  return (static_cast<unsigned long long>(t) << ib) | static_cast<unsigned>(i);
}

// Block-cooperative feed of one key per thread into the current select pass (histogram of the
// next digit, or the gather list). Must be reached by every thread of the block.
__device__ __forceinline__ void sel_feed(const SelCtrl* __restrict__ sc, bool valid,
                                         unsigned long long k, int* s_hist, unsigned* g_hist,
                                         unsigned long long* list, int* list_n) {
  if (sc->done) return;  // uniform
  const int bits = sc->bits;
  const bool match = valid && (k >> bits) == sc->prefix;
  if (sc->list) {
    if (match) list[atomicAdd(list_n, 1)] = k;
    return;
  }
  const int w = min(kSelDigit, bits);
  for (int j = threadIdx.x; j < kSelBins; j += blockDim.x) s_hist[j] = 0;
  __syncthreads();
  if (match) atomicAdd(&s_hist[(k >> (bits - w)) & ((1u << w) - 1u)], 1);
  __syncthreads();
  for (int j = threadIdx.x; j < kSelBins; j += blockDim.x) {
    const int v = s_hist[j];
    if (v) atomicAdd(&g_hist[j], static_cast<unsigned>(v));
  }
}

constexpr int kSelFeedBlocks = 256;  // k_sel_hist grid (block-stride)

__global__ void __launch_bounds__(256)
    k_sel_hist(const SamplerCtrl* c, const SelCtrl* sc, const unsigned long long* __restrict__ key,
               unsigned* g_hist, unsigned long long* list, int* list_n) {
  __shared__ int s_hist[kSelBins];
  if (c->round_remove <= 0 || sc->done) return;
  const int n = c->round_n_in;
  for (int base = blockIdx.x * blockDim.x; base < n; base += gridDim.x * blockDim.x) {
    const int i = base + threadIdx.x;
    sel_feed(sc, i < n, i < n ? key[i] : 0ULL, s_hist, g_hist, list, list_n);
    __syncthreads();
  }
}

// Bitonic sort of s_k[0, m) (m <= kSelList) by one block; returns with s_k sorted.
__device__ void sel_block_sort(unsigned long long* s_k, int m) {
  const int tid = threadIdx.x;
  int np = 2;
  while (np < m) np <<= 1;
  for (int j = m + tid; j < np; j += kSelThreads) s_k[j] = ~0ULL;
  __syncthreads();
  for (int k = 2; k <= np; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      for (int x = tid; x < np; x += kSelThreads) {
        const int y = x ^ j;
        if (y > x) {
          const unsigned long long a = s_k[x], b = s_k[y];
          if ((a > b) == ((x & k) == 0)) {
            s_k[x] = b;
            s_k[y] = a;
          }
        }
      }
      __syncthreads();
    }
  }
}

// Picks the digit bin that holds the sought rank from the histogram `hist` (kSelBins
// entries, read through `get`), and updates the select state. Block-wide.
template <typename Get>
__device__ void sel_pick_bin(SelCtrl* sc, Get get, int* list_n) {
  using BlockScan = cub::BlockScan<int, kSelThreads>;
  __shared__ typename BlockScan::TempStorage scan_tmp;
  const int tid = threadIdx.x;
  const int rank = sc->rank;
  constexpr int kPer = kSelBins / kSelThreads;
  int cnt[kPer], tot = 0;
#pragma unroll
  for (int q = 0; q < kPer; ++q) {
    cnt[q] = get(tid * kPer + q);
    tot += cnt[q];
  }
  int before;
  BlockScan(scan_tmp).ExclusiveSum(tot, before);
  __syncthreads();  // everyone has read sc->rank / the bins
#pragma unroll
  for (int q = 0; q < kPer; ++q) {
    if (before <= rank && rank < before + cnt[q]) {  // exactly one bin
      const int w = min(kSelDigit, sc->bits);
      const unsigned long long prefix = (sc->prefix << w) | static_cast<unsigned>(tid * kPer + q);
      sc->prefix = prefix;
      sc->rank = rank - before;
      sc->bits -= w;
      if (sc->bits == 0) {
        sc->thr = prefix;
        sc->done = 1;
      } else if (cnt[q] <= kSelList) {
        sc->list = 1;
        *list_n = 0;
      }
    }
    before += cnt[q];
  }
  __syncthreads();
}

// One select step: picks the next digit from the global histogram, or sorts the gathered
// list and fixes the threshold. With `finish`, a block-local loop then completes the select
// over all keys (histogram passes and the final gather in this one block), so the launch
// sequence is bounded at three passes; that loop only runs when a bucket of more than
// kSelList keys survives 24 key bits (masses equal to 7 significant digits).
__global__ void __launch_bounds__(kSelThreads)
    k_sel_pick(const SamplerCtrl* c, SelCtrl* sc, unsigned* g_hist,
               const unsigned long long* __restrict__ list, int* list_n,
               const unsigned long long* __restrict__ key, bool finish) {
  __shared__ unsigned long long s_k[kSelList];  // also the block histogram (int[kSelBins])
  if (c->round_remove <= 0 || sc->done) return;
  const int tid = threadIdx.x;
  if (sc->list) {
    const int m = *list_n;
    for (int j = tid; j < m; j += kSelThreads) s_k[j] = list[j];
    sel_block_sort(s_k, m);
    if (tid == 0) {
      sc->thr = s_k[sc->rank];
      sc->done = 1;
    }
    return;
  }
  sel_pick_bin(sc, [&](int j) { return static_cast<int>(g_hist[j]); }, list_n);
  for (int j = tid; j < kSelBins; j += kSelThreads) g_hist[j] = 0u;
  if (!finish) return;
  int* s_hist = reinterpret_cast<int*>(s_k);
  __shared__ int s_m;
  const int n = c->round_n_in;
  while (!sc->done) {  // uniform: sc is written between barriers only
    const int bits = sc->bits;
    const unsigned long long prefix = sc->prefix;
    if (sc->list) {
      if (tid == 0) s_m = 0;
      __syncthreads();
      for (int i = tid; i < n; i += kSelThreads) {
        const unsigned long long k = key[i];
        if ((k >> bits) == prefix) s_k[atomicAdd(&s_m, 1)] = k;
      }
      __syncthreads();
      sel_block_sort(s_k, s_m);
      if (tid == 0) {
        sc->thr = s_k[sc->rank];
        sc->done = 1;
      }
      __syncthreads();
      break;
    }
    const int w = min(kSelDigit, bits);
    for (int j = tid; j < kSelBins; j += kSelThreads) s_hist[j] = 0;
    __syncthreads();
    for (int i = tid; i < n; i += kSelThreads) {
      const unsigned long long k = key[i];
      if ((k >> bits) == prefix) atomicAdd(&s_hist[(k >> (bits - w)) & ((1u << w) - 1u)], 1);
    }
    __syncthreads();
    sel_pick_bin(sc, [&](int j) { return s_hist[j]; }, list_n);
  }
}

// Keeps each point's K lowest-mass neighbours (ascending, padded with self), writes the
// selection key, initialises the matching state and the merge accumulators, and feeds the
// first select pass.
template <int K>
__global__ void __launch_bounds__(256)
    k_rows_and_keys(const SamplerCtrl* c, const SelCtrl* sc, const float2* __restrict__ pts,
                    const float* __restrict__ mass, const int* __restrict__ nbr, int* top_idx,
                    unsigned long long* key, uint8_t* alive, uint8_t* matched, int* target,
                    uint8_t* pending, unsigned long long* myp0, unsigned long long* myp1,
                    float2* wpos, float* mpost, unsigned* g_hist, unsigned long long* list,
                    int* list_n, int ib, int tag, int cap) {
  __shared__ int s_hist[kSelBins];
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (c->round_remove <= 0) return;  // uniform
  const int n_in = c->round_n_in;
  const bool valid = i < n_in && i < cap;
  unsigned long long k = 0ULL;
  if (valid) {
    float m[K];
    int id[K];
#pragma unroll
    for (int q = 0; q < K; ++q) {
      m[q] = INFINITY;
      id[q] = i;
    }
    int row[kNbrSlots];
    const int4* row4 = reinterpret_cast<const int4*>(nbr + static_cast<size_t>(i) * kNbrSlots);
#pragma unroll
    for (int q = 0; q < kNbrSlots / 4; ++q) {
      const int4 v = row4[q];
      row[4 * q] = v.x;
      row[4 * q + 1] = v.y;
      row[4 * q + 2] = v.z;
      row[4 * q + 3] = v.w;
    }
#pragma unroll
    for (int sl = 0; sl < kNbrSlots; ++sl) {
      if (!nbr_live(row[sl], tag)) continue;
      const int j = row[sl] & 0x3fffffff;
      const float mj = mass[j];
      if (!(mj < m[K - 1])) continue;
      // register-resident insertion: shift every slot above the insertion point
#pragma unroll
      for (int b = K - 1; b > 0; --b) {
        if (m[b - 1] > mj) {
          m[b] = m[b - 1];
          id[b] = id[b - 1];
        } else if (m[b] > mj) {
          m[b] = mj;
          id[b] = j;
        }
      }
      if (m[0] > mj) {
        m[0] = mj;
        id[0] = j;
      }
    }
#pragma unroll
    for (int q = 0; q < K; ++q) top_idx[i * K + q] = id[q];
    const float mi = mass[i];
    k = sel_key(mi, i, ib);
    key[i] = k;
    alive[i] = 1;
    matched[i] = 0;
    target[i] = -1;
    pending[i] = 0;
    myp0[i] = kNoProp;
    myp1[i] = kNoProp;
    const float2 p = pts[i];
    wpos[i] = make_float2(p.x * mi, p.y * mi);
    mpost[i] = mi;
  }
  sel_feed(sc, valid, k, s_hist, g_hist, list, list_n);
}

// ---- priority matching, K columns
// Column c: phase 1 = every live unmatched candidate proposes to its c-th lowest-mass neighbour
// (if alive) with its key; phase 2 = a proposal wins if it is the smallest the target received.
// Phase 2 of column c and phase 1 of column c+1 run in one kernel: a thread needs the
// post-phase-2 state of its proposal target, which it evaluates itself from the phase-1 outputs
// (myp, tmin) of column c instead of reading the target's alive flag after a grid barrier.
// Only candidates ever propose or match, so every other thread returns at once: myp of
// non-candidates stays at "no proposal" from k_rows_and_keys, and the tmin entries a column
// used are reset by their proposers two kernels later, before the column that reuses the
// buffer (tmin is triple-buffered: column c read, c+1 written, c+2 reset; myp
// double-buffered). All three tmin buffers start the round reset by k_grid_count.

__device__ __forceinline__ bool won_col(int j, int col, int K, const int* __restrict__ table,
                                        const unsigned long long* __restrict__ myp,
                                        const unsigned long long* __restrict__ tmin) {
  const unsigned long long p = myp[j];
  return p != kNoProp && p == tmin[table[j * K + col]];
}

__device__ __forceinline__ void propose(int i, int n, int K, int col, bool live_i,
                                        unsigned long long key_i, const int* __restrict__ table,
                                        bool prop_alive, unsigned long long* myp,
                                        unsigned long long* tmin) {
  const int prop = table[i * K + col];
  const bool valid = live_i && prop != i && prop >= 0 && prop < n && prop_alive;
  myp[i] = valid ? key_i : kNoProp;
  if (valid) atomicMin(&tmin[prop], key_i);
}

// phase 1 of column 0; also fixes the candidate set from the select threshold
__global__ void k_match_first(const SamplerCtrl* c, const SelCtrl* sc,
                              const unsigned long long* __restrict__ key,
                              const int* __restrict__ table, const uint8_t* __restrict__ alive,
                              uint8_t* cand, unsigned long long* myp, unsigned long long* tmin,
                              int K, int cap) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= cap || c->round_remove <= 0) return;
  const int n = c->round_n_in;
  if (i >= n) return;
  const unsigned long long k = key[i];
  const bool cd = k < sc->thr;
  cand[i] = cd;
  if (!cd) return;
  const int prop = table[i * K];
  const bool pa = prop >= 0 && prop < n && alive[prop];
  propose(i, n, K, 0, true, k, table, pa, myp, tmin);
}

// phase 2 of column col + phase 1 of column col + 1 (+ reset of column col - 1's tmin entries)
__global__ void k_match_step(const SamplerCtrl* c, const unsigned long long* __restrict__ key,
                             const int* __restrict__ table, const uint8_t* __restrict__ cand,
                             uint8_t* alive, uint8_t* matched, int* target,
                             const unsigned long long* __restrict__ myp_cur,
                             const unsigned long long* __restrict__ tmin_cur,
                             unsigned long long* myp_next, unsigned long long* tmin_next,
                             unsigned long long* tmin_prev, int K, int col, int cap) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= cap || c->round_remove <= 0) return;
  const int n = c->round_n_in;
  if (i >= n || !cand[i]) return;
  // myp_next holds this thread's column col - 1 proposal (same buffer parity)
  if (col >= 1 && col + 2 < K) {
    const unsigned long long pp = myp_next[i];
    if (pp != kNoProp) tmin_prev[table[i * K + col - 1]] = kNoProp;
  }
  bool live = alive[i] && !matched[i];
  if (live && won_col(i, col, K, table, myp_cur, tmin_cur)) {
    target[i] = table[i * K + col];
    matched[i] = 1;
    alive[i] = 0;
    live = false;
  }
  const int prop = table[i * K + col + 1];
  bool pa = false;
  if (prop >= 0 && prop < n) {
    // alive after phase 2 of this column; alive[prop] may already hold prop's own update,
    // which agrees with the evaluation below
    pa = alive[prop] && !won_col(prop, col, K, table, myp_cur, tmin_cur);
  }
  propose(i, n, K, col + 1, live, key[i], table, pa, myp_next, tmin_next);
}

// phase 2 of the last column + exact-count completion step 1 (every unmatched candidate is
// removed) + merge of the matched points into their targets
__global__ void k_match_last(SamplerCtrl* c, const float2* __restrict__ pts,
                             const float* __restrict__ mass, const int* __restrict__ table,
                             const uint8_t* __restrict__ cand, uint8_t* alive, uint8_t* matched,
                             int* target, uint8_t* pending,
                             const unsigned long long* __restrict__ myp_cur,
                             const unsigned long long* __restrict__ tmin_cur, float2* wpos,
                             float* mpost, int K, int cap) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  bool won = false;
  if (i < cap && c->round_remove > 0 && i < c->round_n_in && cand[i]) {
    bool m = matched[i];
    if (alive[i] && !m && won_col(i, K - 1, K, table, myp_cur, tmin_cur)) {
      target[i] = table[i * K + K - 1];
      matched[i] = 1;
      m = true;
    }
    pending[i] = !m;
    alive[i] = 0;
    if (m) {
      won = true;
      const int t = target[i];
      const float mi = mass[i];
      atomicAdd(&wpos[t].x, pts[i].x * mi);
      atomicAdd(&wpos[t].y, pts[i].y * mi);
      atomicAdd(&mpost[t], mi);
    }
  }
  const unsigned bal = __ballot_sync(0xffffffffu, won);
  if ((threadIdx.x & 31) == 0 && bal) atomicAdd(&c->stat_matched, __popc(bal));
}

// Step 2: merge into the lowest-mass surviving Voronoi neighbour, else queue a
// nearest-survivor search.
__global__ void k_fallback_adj(SamplerCtrl* c, const float2* __restrict__ pts,
                               const float* __restrict__ mass, const int* __restrict__ table,
                               const uint8_t* __restrict__ pending,
                               const uint8_t* __restrict__ alive, int* target, uint8_t* matched,
                               int* nn_list, float2* wpos, float* mpost, int K, int cap) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= cap || c->round_remove <= 0) return;
  const int n = c->round_n_in;
  if (i >= n || !pending[i]) return;
  for (int k = 0; k < K; ++k) {
    const int j = table[i * K + k];
    if (j != i && j >= 0 && j < n && alive[j]) {
      target[i] = j;
      matched[i] = 1;
      const float mi = mass[i];
      atomicAdd(&wpos[j].x, pts[i].x * mi);
      atomicAdd(&wpos[j].y, pts[i].y * mi);
      atomicAdd(&mpost[j], mi);
      atomicAdd(&c->stat_forced_adj, 1);
      return;
    }
  }
  const int slot = atomicAdd(&c->nn_count, 1);
  nn_list[slot] = i;
}

// Step 3: exact nearest survivor in normalised coordinates (d = fl(fl(dx^2) + fl(dy^2)),
// ties -> lowest index), searched on this round's Voronoi grid hash (built over the same
// points by assign): one warp per queued source, lanes split the cells of each ring, rings at
// levels 0..3 as in the assignment. Cells are binned in clamped pixel space while the metric is
// normalised, so ring / node bounds are converted with a 0.01 px margin and a 1e-5 relative
// shrink; both only ever make the search look further, never skip a closer or tied survivor.
constexpr int kNnBlocks = 512;  // k_fallback_nn grid (warp-stride over the queue)

__device__ __forceinline__ float nn_gap(float gap_px, int extent) {
  return fmaxf(gap_px - 0.01f, 0.f) / static_cast<float>(extent);
}

__global__ void k_fallback_nn(SamplerCtrl* c, const float2* __restrict__ pts,
                              const uint8_t* __restrict__ alive, const int* __restrict__ nn_list,
                              const int* __restrict__ start, const int* __restrict__ sidx,
                              int* target, uint8_t* matched, const float* __restrict__ mass,
                              float2* wpos, float* mpost, int H, int W) {
  if (c->round_remove <= 0) return;
  const int count = c->nn_count;
  if (count == 0) return;
  const GridCtrl g = grid_params(c->round_n_in, H, W);
  const int lane = threadIdx.x & 31;
  const int warps = gridDim.x * (blockDim.x >> 5);
  for (int e = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5); e < count; e += warps) {
    const int i = nn_list[e];
    const float2 p = pts[i];
    const float qx = fminf(fmaxf(p.x * W, 0.f), W - 1e-3f), qy = fminf(fmaxf(p.y * H, 0.f), H - 1e-3f);
    const int px = min(max(static_cast<int>(floorf(qx)), 0), W - 1);
    const int py = min(max(static_cast<int>(floorf(qy)), 0), H - 1);
    unsigned long long best = ~0ULL;
    bool done = false;
    for (int level = 0; level < kGridLevels && !done; ++level) {
      const int cs = g.s << (2 * level);
      const int gx = (g.gx + (1 << (2 * level)) - 1) >> (2 * level);
      const int gy = (g.gy + (1 << (2 * level)) - 1) >> (2 * level);
      const int cx = px / cs, cy = py / cs;
      const int rmax = level + 1 == kGridLevels ? INT_MAX : kGridRingCap;
      for (int r = 0; r <= rmax; ++r) {
        const int ncell = r == 0 ? 1 : 8 * r;
        const float bd = __uint_as_float(static_cast<unsigned>(best >> 32));
        for (int cc = lane; cc < ncell; cc += 32) {
          int xx, yy;
          rect_ring_cell(cc, r, cx, cx, cy, cy, xx, yy);
          if (xx < 0 || xx >= gx || yy < 0 || yy >= gy) continue;
          const float x0 = static_cast<float>(xx * cs), y0 = static_cast<float>(yy * cs);
          const float gxn = nn_gap(fmaxf(fmaxf(x0 - qx, qx - (x0 + cs)), 0.f), W);
          const float gyn = nn_gap(fmaxf(fmaxf(y0 - qy, qy - (y0 + cs)), 0.f), H);
          if (best != ~0ULL && 0.99999f * seed_d2(gxn, gyn) > bd) continue;
          const int k0 = grid_key(g, xx << (2 * level), yy << (2 * level));
          const int a = start[k0], b = start[k0 + (1 << (4 * level))];
          for (int j = a; j < b; ++j) {
            const int id = sidx[j];
            if (!alive[id]) continue;
            const float2 o = pts[id];
            const float d = seed_d2(o.x - p.x, o.y - p.y);
            const unsigned long long key =
                (static_cast<unsigned long long>(__float_as_uint(d)) << 32) |
                static_cast<unsigned>(id);
            best = key < best ? key : best;
          }
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
          const unsigned long long other = __shfl_xor_sync(0xffffffffu, best, o);
          best = other < best ? other : best;
        }
        // lower bound on the distance of survivors outside the covered square
        const int x0 = cx - r, x1 = cx + r, y0 = cy - r, y1 = cy + r;
        float bnd = INFINITY;
        if (x0 > 0) bnd = fminf(bnd, nn_gap(qx - static_cast<float>(x0 * cs), W));
        if (x1 < gx - 1) bnd = fminf(bnd, nn_gap(static_cast<float>((x1 + 1) * cs) - qx, W));
        if (y0 > 0) bnd = fminf(bnd, nn_gap(qy - static_cast<float>(y0 * cs), H));
        if (y1 < gy - 1) bnd = fminf(bnd, nn_gap(static_cast<float>((y1 + 1) * cs) - qy, H));
        if (bnd == INFINITY ||
            (best != ~0ULL &&
             __uint_as_float(static_cast<unsigned>(best >> 32)) < 0.99999f * __fmul_rn(bnd, bnd))) {
          done = true;
          break;
        }
      }
    }
    if (lane == 0 && best != ~0ULL) {
      const int t = static_cast<int>(best & 0xffffffffULL);
      target[i] = t;
      matched[i] = 1;
      if (wpos) {
        const float mi = mass[i];
        atomicAdd(&wpos[t].x, p.x * mi);
        atomicAdd(&wpos[t].y, p.y * mi);
        atomicAdd(&mpost[t], mi);
      }
      atomicAdd(&c->stat_forced_nn, 1);
    }
  }
}

__global__ void k_merge_finalize(const SamplerCtrl* c, const float2* __restrict__ pts,
                                 const float2* __restrict__ wpos, const float* __restrict__ mpost,
                                 const uint8_t* __restrict__ alive, float2* out, uint8_t* flag,
                                 int cap) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= cap) return;
  const int n = c->round_n_in;
  if (i >= n) {
    flag[i] = 0;
    return;
  }
  if (c->round_remove <= 0) {
    out[i] = pts[i];
    flag[i] = 1;
    return;
  }
  const float mp = mpost[i];
  const float2 p = pts[i];
  const float d = fmaxf(mp, 1e-8f);
  out[i] = mp > 1e-8f ? make_float2(wpos[i].x / d, wpos[i].y / d) : p;
  flag[i] = alive[i];
}

// ------------------------------------------------------------------ Lloyd
__global__ void k_lloyd_finalize(float2* pts, const float* __restrict__ wsum,
                                 const float* __restrict__ xacc, const float* __restrict__ yacc,
                                 const int* n_ptr, int cap) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= cap || i >= *n_ptr) return;
  const float w = wsum[i];
  if (w > 1e-8f)
    pts[i] = make_float2(fminf(fmaxf(xacc[i] / w, 0.f), 1.f), fminf(fmaxf(yacc[i] / w, 0.f), 1.f));
}

// ------------------------------------------------------------------ final weights
__global__ void k_count_check(SamplerCtrl* c, int n_cap) {
  c->stat_count_error = abs(c->n_cur - c->n);
  c->n_cur = min(c->n_cur, n_cap);
  c->weight_sum = 0.f;
}

__global__ void k_weight_sum(SamplerCtrl* c, const float* __restrict__ mass, int cap) {
  __shared__ float s[32];
  const int n = c->n_cur;
  float v = 0.f;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < min(n, cap); i += gridDim.x * blockDim.x)
    v += mass[i];
  for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
  if ((threadIdx.x & 31) == 0) s[threadIdx.x >> 5] = v;
  __syncthreads();
  if (threadIdx.x == 0) {
    float t = 0.f;
    for (int w = 0; w < static_cast<int>(blockDim.x >> 5); ++w) t += s[w];
    atomicAdd(&c->weight_sum, t);
  }
}

__global__ void k_weights_out(const SamplerCtrl* c, const float2* __restrict__ pts,
                              const float* __restrict__ mass, float2* pts_out, float* w_out,
                              int n_cap) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_cap || i >= c->n_cur) return;
  const float scale = static_cast<float>(c->n_cur) / fmaxf(c->weight_sum, 1e-8f);
  pts_out[i] = pts[i];
  w_out[i] = mass[i] * scale;
}

// ------------------------------------------------------------------ host helpers
}  // namespace

// ------------------------------------------------------------------ API
void plan_sampler(const SamplerConfig& cfg, Arena& a, SamplerBuffers* b) {
  b->cfg = cfg;
  const int H = cfg.H, W = cfg.W, P = H * W;
  const int n = std::max(1, cfg.n_cap);
  b->init_cap = static_cast<int>(
      std::max<long long>(n + 1LL, std::llround(static_cast<double>(n) * cfg.oversample)));
  b->max_tiles = ((H + 3) / 4) * ((W + 3) / 4);
  const int cap = b->init_cap, K = cfg.knn_k, T = b->max_tiles;
  b->ctrl = a.take<SamplerCtrl>(1);
  b->density_s = a.take<float>(P);
  b->pts[0] = a.take<float2>(cap);
  b->pts[1] = a.take<float2>(cap);
  b->pts_tmp = a.take<float2>(cap);
  b->pts_pixel = a.take<float2>(cap);
  b->owner = a.take<int>(P);
  {
    const int gx = (W + kGridMinCell - 1) / kGridMinCell, gy = (H + kGridMinCell - 1) / kGridMinCell;
    const long long keys = static_cast<long long>((gx + kGridTile - 1) / kGridTile) *
                           ((gy + kGridTile - 1) / kGridTile) * kGridTile * kGridTile;
    b->grid_max_keys = static_cast<int>(keys);
  }
  b->grid = a.take<GridCtrl>(1);
  b->grid_cnt = a.take<int>(b->grid_max_keys + 1 + b->grid_max_keys / kTileKeys);
  b->grid_start = a.take<int>(b->grid_max_keys + 1);
  b->grid_key = a.take<int>(cap);
  b->grid_rank = a.take<int>(cap);
  b->grid_pos = a.take<float2>(cap);
  b->grid_idx = a.take<int>(cap);
  b->grid_open_count = a.take<int>(1);
  {
    const int open_tiles = ((W + kOpenTW - 1) / kOpenTW) * ((H + kOpenTH - 1) / kOpenTH);
    b->grid_open_tiles = a.take<int>(open_tiles);
    b->grid_open_mask = a.take<unsigned>(open_tiles);
  }
  b->grid_open_best = a.take<float2>(P);
  b->mass = a.take<float>(cap);
  b->nbr = a.take<int>(static_cast<size_t>(cap) * kNbrSlots);
  b->top_idx = a.take<int>(static_cast<size_t>(cap) * K);
  b->key = a.take<unsigned long long>(cap);
  b->sel = a.take<SelCtrl>(1);
  b->sel_hist = a.take<unsigned>(kSelBins);
  b->sel_list = a.take<unsigned long long>(cap);
  b->cand = a.take<uint8_t>(cap);
  b->alive = a.take<uint8_t>(cap);
  b->matched = a.take<uint8_t>(cap);
  b->flag = a.take<uint8_t>(cap);
  b->target = a.take<int>(cap);
  b->myp[0] = a.take<unsigned long long>(cap);
  b->myp[1] = a.take<unsigned long long>(cap);
  for (int q = 0; q < 3; ++q) b->tmin[q] = a.take<unsigned long long>(cap);
  b->nn_list = a.take<int>(cap);
  b->wsum = a.take<float>(cap);
  b->xacc = a.take<float>(cap);
  b->yacc = a.take<float>(cap);
  b->wpos = a.take<float2>(cap);
  b->mpost = a.take<float>(cap);
  b->tile_mass = a.take<double>(T);
  b->tile_cum = a.take<double>(T);
  b->tile_count = a.take<int>(T);
  b->tile_off = a.take<int>(T);

  size_t need = 0, t = 0;
  cub::DeviceSelect::Flagged(nullptr, t, b->pts_tmp, b->flag, b->pts[0], &b->ctrl->n_cur, cap);
  need = std::max(need, t);
  cub::DeviceScan::InclusiveSum(nullptr, t, b->tile_mass, b->tile_cum, T);
  need = std::max(need, t);
  cub::DeviceScan::ExclusiveSum(nullptr, t, b->tile_count, b->tile_off, T);
  need = std::max(need, t);
  b->cub_temp_bytes = need;
  b->cub_temp = a.take<uint8_t>(need);
}

cudaError_t run_sampler(SamplerBuffers& b, const float* density_w, float2* points_out,
                        float* weights_out, cudaStream_t s) {
  const SamplerConfig& cfg = b.cfg;
  const int H = cfg.H, W = cfg.W, cap = b.init_cap, K = cfg.knn_k, T = b.max_tiles;
  if (K != 8) return cudaErrorInvalidValue;  // k_rows_and_keys is instantiated for knn_k = 8
  SamplerCtrl* c = b.ctrl;
  const double offset = unit_interval_from_seed(cfg.seed, 101);
  size_t tb = b.cub_temp_bytes;

  // ---- error-diffusion oversample into pts[0]
  k_ed_setup<<<1, 1, 0, s>>>(c, H, W, cfg.seed, cfg.oversample);
  GF_TRY(cudaMemsetAsync(b.tile_mass, 0, sizeof(double) * T, s));
  k_ed_aggregate<<<dim3((W + 15) / 16, (H + 15) / 16), dim3(16, 16), 0, s>>>(
      b.density_s, c, H, W, b.tile_mass, reinterpret_cast<int*>(b.sel_hist), kSelBins, b.grid_cnt,
      b.grid_max_keys + 1 + b.grid_max_keys / kTileKeys);
  GF_TRY(cub::DeviceScan::InclusiveSum(b.cub_temp, tb, b.tile_mass, b.tile_cum, T, s));
  k_ed_counts<<<grid_for(T), 256, 0, s>>>(c, b.tile_cum, T, offset, b.tile_count);
  k_ed_fix<<<1, 1024, 0, s>>>(c, b.tile_mass, b.tile_cum, T, offset, b.tile_count);
  tb = b.cub_temp_bytes;
  GF_TRY(cub::DeviceScan::ExclusiveSum(b.cub_temp, tb, b.tile_count, b.tile_off, T, s));
  k_ed_emit_small<<<(T + kEdWarps - 1) / kEdWarps, 32 * kEdWarps, 0, s>>>(
      b.density_s, c, b.tile_count, b.tile_off, H, W, cfg.seed, b.pts[0]);
  // tiles larger than 16 px exist only when the tile size is >= 17: at most this many
  const int big_tiles = ((H + 16) / 17) * ((W + 16) / 17);
  k_ed_emit<kMaxTileArea><<<big_tiles, 64, 0, s>>>(b.density_s, c, b.tile_count, b.tile_off, H,
                                                   W, cfg.seed, b.pts[0]);

  // ---- merge rounds pts[r % 2] -> pts[(r + 1) % 2]
  const int R = std::max(1, cfg.merge_rounds);
  int key_ib = 1;  // index bits of the selection keys
  while ((1LL << key_ib) < cap) ++key_ib;
  const int* gate = &c->round_remove;
  const int* n_in = &c->round_n_in;
  for (int r = 0; r < R; ++r) {
    const float2* pin = b.pts[r % 2];
    float2* pout = b.pts[(r + 1) % 2];
    k_round_schedule<<<1, 1, 0, s>>>(c, b.sel, r, R, cfg.per_round_remove_fraction,
                                     32 + key_ib);
    // owner map for the adjacency + cell masses of the sampling density (+ reset of the
    // neighbour sets and matching buffers)
    GF_TRY(assign(b, pin, n_in, gate, cap,
                  AssignOut{b.owner, b.density_s, b.mass, nullptr, nullptr}, s,
                  RoundReset{(r & 1) ? nullptr : b.nbr, b.tmin[0], b.tmin[1], b.tmin[2]}));
    k_adjacency<<<dim3((W + 15) / 16, (H + 15) / 16), dim3(16, 16), 0, s>>>(b.owner, c, b.nbr,
                                                                             H, W, r & 1);
    k_rows_and_keys<8><<<grid_for(cap), 256, 0, s>>>(
        c, b.sel, pin, b.mass, b.nbr, b.top_idx, b.key, b.alive, b.matched, b.target, b.flag,
        b.myp[0], b.myp[1], b.wpos, b.mpost, b.sel_hist, b.sel_list, &b.sel->list_n, key_ib, r & 1, cap);
    k_sel_pick<<<1, kSelThreads, 0, s>>>(c, b.sel, b.sel_hist, b.sel_list, &b.sel->list_n,
                                         b.key, kSelPasses == 1);
    for (int pass = 1; pass < kSelPasses; ++pass) {
      k_sel_hist<<<kSelFeedBlocks, 256, 0, s>>>(c, b.sel, b.key, b.sel_hist, b.sel_list,
                                                 &b.sel->list_n);
      k_sel_pick<<<1, kSelThreads, 0, s>>>(c, b.sel, b.sel_hist, b.sel_list, &b.sel->list_n,
                                           b.key, pass + 1 == kSelPasses);
    }
    k_match_first<<<grid_for(cap), 256, 0, s>>>(c, b.sel, b.key, b.top_idx, b.alive, b.cand,
                                                 b.myp[0], b.tmin[0], K, cap);
    for (int col = 0; col + 1 < K; ++col) {
      k_match_step<<<grid_for(cap), 256, 0, s>>>(
          c, b.key, b.top_idx, b.cand, b.alive, b.matched, b.target, b.myp[col % 2],
          b.tmin[col % 3], b.myp[(col + 1) % 2], b.tmin[(col + 1) % 3], b.tmin[(col + 2) % 3], K,
          col, cap);  // tmin[(col + 2) % 3] == tmin[(col - 1) % 3]
    }
    k_match_last<<<grid_for(cap), 256, 0, s>>>(c, pin, b.mass, b.top_idx, b.cand, b.alive,
                                                b.matched, b.target, b.flag, b.myp[(K - 1) % 2],
                                                b.tmin[(K - 1) % 3], b.wpos, b.mpost, K, cap);
#ifdef GFUSED_SAMPLER_MATCH_HOOK
    GFUSED_SAMPLER_MATCH_HOOK(b, s);
#endif
    k_fallback_adj<<<grid_for(cap), 256, 0, s>>>(c, pin, b.mass, b.top_idx, b.flag, b.alive,
                                                  b.target, b.matched, b.nn_list, b.wpos,
                                                  b.mpost, K, cap);
    k_fallback_nn<<<kNnBlocks, 128, 0, s>>>(c, pin, b.alive, b.nn_list, b.grid_start, b.grid_idx,
                                            b.target, b.matched, b.mass, b.wpos, b.mpost, H, W);
    k_merge_finalize<<<grid_for(cap), 256, 0, s>>>(c, pin, b.wpos, b.mpost, b.alive, b.pts_tmp,
                                                    b.flag, cap);
    tb = b.cub_temp_bytes;
    GF_TRY(cub::DeviceSelect::Flagged(b.cub_temp, tb, b.pts_tmp, b.flag, pout, &c->n_cur, cap, s));
  }
  float2* fin = b.pts[R % 2];

  // ---- Lloyd polish (sampling density)
  for (int it = 0; it < cfg.lloyd_iters; ++it) {
    // density-weighted centroids
    GF_TRY(assign(b, fin, &c->n_cur, nullptr, cap,
                  AssignOut{nullptr, b.density_s, b.wsum, b.xacc, b.yacc}, s));
    k_lloyd_finalize<<<grid_for(cap), 256, 0, s>>>(fin, b.wsum, b.xacc, b.yacc, &c->n_cur, cap);
  }

  // ---- final assignment -> cell masses -> weights
  k_count_check<<<1, 1, 0, s>>>(c, cfg.n_cap);
  GF_TRY(assign(b, fin, &c->n_cur, nullptr, cap,
                AssignOut{nullptr, density_w, b.mass, nullptr, nullptr}, s));
  k_weight_sum<<<128, 256, 0, s>>>(c, b.mass, cap);
  k_weights_out<<<grid_for(cfg.n_cap), 256, 0, s>>>(c, fin, b.mass, points_out, weights_out,
                                                     cfg.n_cap);
  return cudaGetLastError();
}

namespace {

// The enclosing power-of-two square gives the same order for rectangular frames
// as gaussifier.hilbert_sampler. Compute the permutation once at prepare time.
uint64_t hilbert_distance(int x, int y, int side) {
  uint64_t distance = 0;
  for (int s = side / 2; s > 0; s /= 2) {
    const bool rx = x & s, ry = y & s;
    distance += uint64_t(s) * s * ((3 * rx) ^ ry);
    if (!ry) {
      if (rx) {
        x = s - 1 - x;
        y = s - 1 - y;
      }
      std::swap(x, y);
    }
  }
  return distance;
}

struct AddMoments {
  __device__ double3 operator()(double3 a, double3 b) const {
    return make_double3(a.x + b.x, a.y + b.y, a.z + b.z);
  }
};

__global__ void k_video_count(SamplerCtrl* c, VideoTrackCtrl* v, double ema, bool first) {
  const int raw = c->n;
  if (first) {
    *v = {};
    v->smoothed_count = raw;
    v->count = raw;
    v->next_id = raw;
  } else {
    v->smoothed_count = ema * v->smoothed_count + (1.0 - ema) * raw;
  }
  v->stats.target = max(1, __double2int_rn(v->smoothed_count));
  v->stats.survivors = v->kept;
  c->n = c->n_cur = v->count;
  c->n_init = first ? c->n_cur : 0;
  c->tile_size = c->stat_matched = c->stat_forced_adj = c->stat_forced_nn = 0;
  c->stat_count_error = 0;
}

__global__ void k_hilbert_pixels(VideoTracks v, const float* density, int H, int W) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= H * W) return;
  const int p = v.hilbert_order[i];
  const double mass = fmaxf(density[p], 0.f);
  v.hilbert_prefix[i] = make_double3(mass, mass * (p % W + 0.5) / W,
                                    mass * (p / W + 0.5) / H);
}

__device__ int hilbert_cut(const double3* prefix, int P, double total, int cut, int n) {
  if (cut == 0) return 0;
  if (cut == n) return P;
  if (total <= 1e-12) return static_cast<int>(static_cast<int64_t>(P) * cut / n);
  const double target = total * cut / n;
  int lo = 0, hi = P;
  while (lo < hi) {
    const int mid = lo + (hi - lo) / 2;
    if (prefix[mid].x <= target) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

__global__ void k_hilbert_tracks(VideoTracks v, float2* points, int cap, int H, int W) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int n = v.ctrl->count, P = H * W;
  if (i >= cap) return;
  if (i >= n) {
    v.free_slots[cap - 1 - i] = i;
    return;
  }
  if (i == 0) v.ctrl->free_count = cap - n;
  const double3* prefix = v.hilbert_prefix;
  const double total = prefix[P - 1].x;
  const int a = hilbert_cut(prefix, P, total, i, n);
  const int b = hilbert_cut(prefix, P, total, i + 1, n);
  const double3 start = a ? prefix[a - 1] : make_double3(0, 0, 0);
  const double3 end = b ? prefix[b - 1] : make_double3(0, 0, 0);
  const int p = v.hilbert_order[min(a, P - 1)];
  double x = (p % W + 0.5) / W;
  double y = (p / W + 0.5) / H;
  if (end.x - start.x > 1e-12) {
    x = (end.y - start.y) / (end.x - start.x);
    y = (end.z - start.z) / (end.x - start.x);
  } else if (total <= 1e-12 && b > a) {
    x = y = 0.0;
    for (int j = a; j < b; ++j) {
      const int pixel = v.hilbert_order[j];
      x += pixel % W + 0.5;
      y += pixel / W + 0.5;
    }
    x /= (b - a) * static_cast<double>(W);
    y /= (b - a) * static_cast<double>(H);
  }
  points[i] = make_float2(fmin(fmax(x, 0.0), 1.0), fmin(fmax(y, 0.0), 1.0));
  v.tracks[i] = VideoTrack{i, points[i], i, 0, false};
  v.ids[i] = i;
  v.slots[i] = i;
  v.history_slots[i] = -1;
}

__global__ void k_restore_tracks(VideoTracks v, float2* points, int cap) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= cap || i >= v.ctrl->count) return;
  const VideoTrack t = v.tracks[i];
  points[i] = t.xy;
  v.ids[i] = t.id;
  v.slots[i] = t.slot;
  v.history_slots[i] = t.has_history ? t.slot : -1;
}

__global__ void k_video_weights(const SamplerCtrl* c, const float* mass, float* weights, int cap) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= cap || i >= c->n_cur) return;
  weights[i] = mass[i] * (static_cast<float>(c->n_cur) / fmaxf(c->weight_sum, 1e-8f));
}

__global__ void k_repair_begin(VideoTrackCtrl* v, VideoTrackOptions opts) {
  const int n = v->count;
  const int lower = static_cast<int>(n * (1.0 - opts.max_count_rate));
  const int upper = max(n + 1, __double2int_ru(n * (1.0 + opts.max_count_rate)));
  v->stats.limited_target = max(1, min(max(v->stats.target, lower), upper));
  v->hot = v->cold = v->eligible = 0;
}

// Positive floats sort as unsigned bits. The high word ranks the priority tier
// and mass. The low word carries the row through the stable radix sort.
__device__ unsigned long long track_rank_key(unsigned priority, int row) {
  return (static_cast<unsigned long long>(priority) << 32) | static_cast<unsigned>(row);
}

__global__ void k_rank_kills(VideoTracks v, const SamplerCtrl* sc, VideoTrackOptions opts,
                              const float* mass, const float2* xy, int cap) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= cap) return;
  if (i >= v.ctrl->count) {
    v.rank_keys[0][i] = ULLONG_MAX;
    return;
  }
  const float rel = mass[i] * (v.ctrl->stats.limited_target / fmaxf(sc->weight_sum, 1e-12f));
  const bool hot = rel > 1.f + opts.tau_hot, cold = rel < 1.f - opts.tau_cold;
  VideoTrack t = v.tracks[i];
  t.xy = xy[i];
  t.has_history = true;
  t.strikes = hot || cold ? min(t.strikes, opts.patience - 1) + 1 : 0;
  v.updated[i] = t;
  const bool ripe = t.strikes >= opts.patience;
  if (ripe && hot) atomicAdd(&v.ctrl->hot, 1);
  if (ripe && cold) atomicAdd(&v.ctrl->cold, 1);
  v.rank_keys[0][i] = track_rank_key(__float_as_uint(rel) | (ripe && cold ? 0 : 0x80000000u), i);
}

__global__ void k_repair_budget(VideoTrackCtrl* v, VideoTrackOptions opts) {
  const int n = v->count, delta = n - v->stats.limited_target;
  const int cap = opts.churn_fraction > 0 ? max(1, static_cast<int>(opts.churn_fraction * n)) : 0;
  const int budget = min(min(cap, max(v->hot, v->cold)), max(0, (n - abs(delta)) / 2));
  v->tentative_kills = max(0, delta) + budget;
  v->tentative_splits = max(0, -delta) + budget;
}

__global__ void k_index_kills(VideoTracks v, int cap) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < cap && i < v.ctrl->count)
    v.kill_rank[static_cast<unsigned>(v.rank_keys[1][i])] = i;
}

__global__ void k_link_pixels(VideoTracks v, const int* owner, int P) {
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= P || !v.ctrl->tentative_splits) return;
  const int i = owner[p];
  if (i < 0 || v.kill_rank[i] < v.ctrl->tentative_kills) return;
  v.pixel_next[p] = atomicExch(&v.pixel_head[i], p);
  atomicAdd(&v.pixel_count[i], 1);
}

__global__ void k_rank_splits(VideoTracks v, const float* mass, int cap) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= cap) return;
  const bool eligible = i < v.ctrl->count && v.ctrl->tentative_splits && v.pixel_count[i] >= 2;
  if (eligible) atomicAdd(&v.ctrl->eligible, 1);
  v.rank_keys[1][i] = eligible ? track_rank_key(0x7fffffffu - __float_as_uint(mass[i]), i)
                               : ULLONG_MAX;
}

__global__ void k_repair_counts(VideoTrackCtrl* v) {
  VideoTrackStats& stats = v->stats;
  stats.born = min(v->tentative_splits, v->eligible);
  stats.killed = max(0, v->count - stats.limited_target + stats.born);
  stats.shortfall = max(0, stats.limited_target - (v->count - stats.killed + stats.born));
}

struct CellMoments {
  double mass = 0, x = 0, y = 0, xx = 0, xy = 0, yy = 0;

  __device__ void add(double px, double py, double w) {
    mass += w;
    x += w * px;
    y += w * py;
    xx += w * px * px;
    xy += w * px * py;
    yy += w * py * py;
  }
};

__device__ int nearest_owned_pixel(VideoTracks v, int head, int W, double x, double y,
                                    int forbid) {
  int best = INT_MAX;
  double min_dist = INFINITY;
  for (int p = head; p >= 0; p = v.pixel_next[p]) {
    if (p == forbid) continue;
    const double dx = p % W + 0.5 - x, dy = p / W + 0.5 - y;
    const double dist = dx * dx + dy * dy;
    if (dist < min_dist || (dist == min_dist && p < best)) {
      min_dist = dist;
      best = p;
    }
  }
  return best;
}

__global__ void k_split_cells(VideoTracks v, const float* density, int H, int W, int cap) {
  const int rank = blockIdx.x * blockDim.x + threadIdx.x;
  if (rank >= cap || rank >= v.ctrl->stats.born) return;
  const int i = static_cast<unsigned>(v.rank_keys[0][rank]);
  const int head = v.pixel_head[i];
  CellMoments weighted, uniform;
  for (int p = head; p >= 0; p = v.pixel_next[p]) {
    const double x = p % W + 0.5, y = p / W + 0.5;
    weighted.add(x, y, density[p]);
    uniform.add(x, y, 1.0);
  }
  const CellMoments m = weighted.mass > 1e-12 ? weighted : uniform;
  const double x = m.x / m.mass, y = m.y / m.mass;
  const double var_x = fmax(m.xx / m.mass - x * x, 0.0);
  const double var_y = fmax(m.yy / m.mass - y * y, 0.0);
  const double cov = m.xy / m.mass - x * y;
  const double half = 0.5 * (var_x - var_y);
  const double eigen = 0.5 * (var_x + var_y) + sqrt(half * half + cov * cov);
  const double norm = hypot(cov, eigen - var_x);
  const double offset = fmax(sqrt(eigen), 0.5);
  const double dx = norm > 1e-30 ? offset * cov / norm : offset;
  const double dy = norm > 1e-30 ? offset * (eigen - var_x) / norm : 0.0;
  const int parent = nearest_owned_pixel(v, head, W, x - dx, y - dy, -1);
  const int child = nearest_owned_pixel(v, head, W, x + dx, y + dy, parent);
  v.updated[i].xy = make_float2((parent % W + 0.5) / W, (parent / W + 0.5) / H);
  v.updated[i].strikes = 0;
  v.child_xy[rank] = make_float2((child % W + 0.5) / W, (child / W + 0.5) / H);
}

__global__ void k_retire_tracks(VideoTracks v, int cap) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= cap) return;
  v.keep[i] = 0;
  if (i >= v.ctrl->count) return;
  const int rank = v.kill_rank[i];
  if (rank < v.ctrl->stats.killed) {
    v.free_slots[v.ctrl->free_count + rank] = v.updated[i].slot;
  } else {
    v.keep[i] = 1;
  }
}

__global__ void k_write_repaired_tracks(VideoTracks v, int cap) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= cap) return;
  if (i < v.ctrl->kept) {
    v.tracks[i] = v.updated[v.kept_rows[i]];
  } else {
    const int child = i - v.ctrl->kept;
    if (child >= v.ctrl->stats.born) return;
    const int slot = v.free_slots[v.ctrl->free_count + v.ctrl->stats.killed - 1 - child];
    v.tracks[i] = VideoTrack{v.ctrl->next_id + child, v.child_xy[child], slot, 0, false};
  }
}

__global__ void k_repair_finish(VideoTrackCtrl* v) {
  v->count = v->kept + v->stats.born;
  v->free_count += v->stats.killed - v->stats.born;
  v->next_id += v->stats.born;
}

}  // namespace

void plan_video_tracks(const SamplerConfig& cfg, Arena& a, VideoTracks* v) {
  const int cap = cfg.n_cap, P = cfg.H * cfg.W;
  v->ctrl = a.take<VideoTrackCtrl>(1);
  v->tracks = a.take<VideoTrack>(cap);
  v->updated = a.take<VideoTrack>(cap);
  v->ids = a.take<int64_t>(cap);
  v->slots = a.take<int>(cap);
  v->history_slots = a.take<int>(cap);
  v->free_slots = a.take<int>(cap);
  v->hilbert_order = a.take<int>(P);
  v->hilbert_prefix = a.take<double3>(P);
  v->pixel_head = a.take<int>(cap);
  v->pixel_next = a.take<int>(P);
  v->pixel_count = a.take<int>(cap);
  v->kill_rank = a.take<int>(cap);
  v->kept_rows = a.take<int>(cap);
  for (auto& keys : v->rank_keys) keys = a.take<unsigned long long>(cap);
  v->child_xy = a.take<float2>(cap);
  v->keep = a.take<uint8_t>(cap);
  size_t scan = 0, sort = 0, select = 0;
  cub::DeviceScan::InclusiveScan(nullptr, scan, v->hilbert_prefix, v->hilbert_prefix,
                                 AddMoments{}, P);
  cub::DeviceRadixSort::SortKeys(nullptr, sort, v->rank_keys[0], v->rank_keys[1], cap, 32, 64);
  cub::DeviceSelect::Flagged(nullptr, select, thrust::make_counting_iterator(0),
                             v->keep, v->kept_rows, &v->ctrl->kept, cap);
  v->cub_temp_bytes = std::max({scan, sort, select});
  v->cub_temp = a.take<uint8_t>(v->cub_temp_bytes);
}

cudaError_t init_video_tracks(const SamplerConfig& cfg, VideoTracks& v) {
  const int P = cfg.H * cfg.W;
  int side = 1;
  while (side < std::max(cfg.H, cfg.W)) side *= 2;
  std::vector<uint64_t> distances(P);
  std::vector<int> order(P);
  std::iota(order.begin(), order.end(), 0);
  for (int i = 0; i < P; ++i)
    distances[i] = hilbert_distance(i % cfg.W, i / cfg.W, side);
  std::sort(order.begin(), order.end(), [&](int a, int b) { return distances[a] < distances[b]; });
  // CUB's multi-word look-back scan can read padding lanes whose values it discards.
  GF_TRY(cudaMemset(v.cub_temp, 0, v.cub_temp_bytes));
  return cudaMemcpy(v.hilbert_order, order.data(), sizeof(int) * P, cudaMemcpyHostToDevice);
}

cudaError_t begin_video_frame(SamplerBuffers& b, VideoTracks& v, const VideoTrackOptions& opts,
                              bool first, cudaStream_t s) {
  k_video_count<<<1, 1, 0, s>>>(b.ctrl, v.ctrl, opts.count_ema, first);
  return cudaGetLastError();
}

cudaError_t sample_video_frame(SamplerBuffers& b, VideoTracks& v, const float* density,
                                bool first, float2* points, float* weights, cudaStream_t s) {
  const int H = b.cfg.H, W = b.cfg.W, cap = b.cfg.n_cap;
  const int* n = &b.ctrl->n_cur;
  if (first) {
    const int grid_entries = b.grid_max_keys + 1 + b.grid_max_keys / kTileKeys;
    GF_TRY(cudaMemsetAsync(b.grid_cnt, 0, sizeof(int) * grid_entries, s));
    k_hilbert_pixels<<<grid_for(H * W), 256, 0, s>>>(v, b.density_s, H, W);
    size_t temp = v.cub_temp_bytes;
    GF_TRY(cub::DeviceScan::InclusiveScan(v.cub_temp, temp, v.hilbert_prefix, v.hilbert_prefix,
                                         AddMoments{}, H * W, s));
    k_hilbert_tracks<<<grid_for(cap), 256, 0, s>>>(v, points, cap, H, W);
  } else {
    k_restore_tracks<<<grid_for(cap), 256, 0, s>>>(v, points, cap);
  }
  // initial_tracks() polishes the Hilbert anchors before the render loop's own polish.
  const int iters = b.cfg.lloyd_iters * (first ? 2 : 1);
  for (int it = 0; it < iters; ++it) {
    GF_TRY(assign(b, points, n, nullptr, cap,
                  AssignOut{nullptr, b.density_s, b.wsum, b.xacc, b.yacc}, s));
    k_lloyd_finalize<<<grid_for(cap), 256, 0, s>>>(points, b.wsum, b.xacc, b.yacc, n, cap);
  }
  k_count_check<<<1, 1, 0, s>>>(b.ctrl, cap);
  GF_TRY(assign(b, points, n, nullptr, cap, AssignOut{nullptr, density, b.mass, nullptr, nullptr}, s));
  k_weight_sum<<<128, 256, 0, s>>>(b.ctrl, b.mass, cap);
  k_video_weights<<<grid_for(cap), 256, 0, s>>>(b.ctrl, b.mass, weights, cap);
  return cudaGetLastError();
}

cudaError_t repair_video_tracks(SamplerBuffers& b, VideoTracks& v, const VideoTrackOptions& opts,
                                 const float* density, const float2* xy, cudaStream_t s) {
  const int H = b.cfg.H, W = b.cfg.W, cap = b.cfg.n_cap;
  k_count_check<<<1, 1, 0, s>>>(b.ctrl, cap);
  GF_TRY(assign(b, xy, &b.ctrl->n_cur, nullptr, cap,
                AssignOut{b.owner, density, b.mass, nullptr, nullptr}, s));
  k_weight_sum<<<128, 256, 0, s>>>(b.ctrl, b.mass, cap);
  k_repair_begin<<<1, 1, 0, s>>>(v.ctrl, opts);
  k_rank_kills<<<grid_for(cap), 256, 0, s>>>(v, b.ctrl, opts, b.mass, xy, cap);
  k_repair_budget<<<1, 1, 0, s>>>(v.ctrl, opts);
  size_t temp = v.cub_temp_bytes;
  GF_TRY(cub::DeviceRadixSort::SortKeys(v.cub_temp, temp, v.rank_keys[0], v.rank_keys[1],
                                        cap, 32, 64, s));
  k_index_kills<<<grid_for(cap), 256, 0, s>>>(v, cap);
  GF_TRY(cudaMemsetAsync(v.pixel_head, -1, sizeof(int) * cap, s));
  GF_TRY(cudaMemsetAsync(v.pixel_count, 0, sizeof(int) * cap, s));
  k_link_pixels<<<grid_for(H * W), 256, 0, s>>>(v, b.owner, H * W);
  k_rank_splits<<<grid_for(cap), 256, 0, s>>>(v, b.mass, cap);
  k_repair_counts<<<1, 1, 0, s>>>(v.ctrl);
  temp = v.cub_temp_bytes;
  GF_TRY(cub::DeviceRadixSort::SortKeys(v.cub_temp, temp, v.rank_keys[1], v.rank_keys[0],
                                        cap, 32, 64, s));
  k_split_cells<<<grid_for(cap), 256, 0, s>>>(v, density, H, W, cap);
  k_retire_tracks<<<grid_for(cap), 256, 0, s>>>(v, cap);
  temp = v.cub_temp_bytes;
  GF_TRY(cub::DeviceSelect::Flagged(v.cub_temp, temp, thrust::make_counting_iterator(0),
                                    v.keep, v.kept_rows, &v.ctrl->kept, cap, s));
  k_write_repaired_tracks<<<grid_for(cap), 256, 0, s>>>(v, cap);
  k_repair_finish<<<1, 1, 0, s>>>(v.ctrl);
  return cudaGetLastError();
}

}  // namespace gfused
