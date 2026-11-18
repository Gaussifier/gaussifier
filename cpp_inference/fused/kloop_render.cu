// Raw-pointer simple-sum renderer and K-loop point kernels. See kloop_render.cuh.

#include <cub/cub.cuh>

#include <algorithm>

#include "config.h"   // gaussifier native_simple_sum_backend: BLOCK_X/BLOCK_Y/BLOCK_SIZE/...
#include "helpers.cuh"
#include "kloop_render.cuh"

namespace gfused {
namespace {

static_assert(BLOCK_X == 16 && BLOCK_Y == 16, "renderer tiles are 16x16");


// alpha = exp(-sigma) < 1/255 <=> sigma > ln(255) = 5.5452; the renderer skips those terms.
// kSigmaCut leaves a margin for __expf's error so every culled term is one the reference
// renderer also skips (culling is exact, not an approximation).
constexpr float kSigmaCut = 5.6f;

constexpr int kMaxSlots = 8;

// Fused: log-cov -> (scale, rotation); colour = clamp(rgb * w); project; tile range; count
// tile hits; optional hard point-count map. Tile range = the reference 3-sigma-circle tile
// box intersected with the axis-aligned box of the sigma = kSigmaCut ellipse.
// Tile counting is warp-aggregated (spatially coherent points hit the same tiles): every lane
// of the warp runs the same loop with a full-mask __match_any_sync, so no early returns.
__global__ void __launch_bounds__(256)
k_project(const int* n_ptr, int n_cap, const float2* __restrict__ xy,
          const float* __restrict__ cov, const float* __restrict__ rgb,
          const float* __restrict__ weights, dim3 img_size, dim3 tile_bounds, float4* proj_a,
          float4* proj_b, int4* proj_box, int* proj_slot, int* tile_counts, int* large_count,
          int* large_ids, int4* large_box, int* wide_ids, int* wide_count, float* point_counts) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  int4 box = make_int4(0, 0, 0, 0);
  int mode = 0;  // 0: not binned, 1: slot-recorded, 2: wide (cursor-filled from the end)
  if (i < n_cap) {
    if (i >= *n_ptr) {
      proj_b[i] = make_float4(0.f, 0.f, 0.f, 0.f);  // overflow path: contributes exactly 0
      proj_a[i] = make_float4(0.f, 0.f, 0.f, 0.f);
      proj_box[i] = box;
    } else {
      const float2 p = xy[i];
      if (point_counts) {
        const int W = img_size.x, H = img_size.y;
        const int px = max(0, min(W - 1, static_cast<int>(floorf(p.x * W))));
        const int py = max(0, min(H - 1, static_cast<int>(floorf(p.y * H))));
        atomicAdd(&point_counts[py * W + px], 1.f);
      }
      const float a = cov[i * 3 + 0], b = cov[i * 3 + 1], c = cov[i * 3 + 2];
      const float center = 0.5f * (a + c);
      const float half_diff = 0.5f * (a - c);
      const float radius_l = sqrtf(half_diff * half_diff + b * b + 1e-12f);
      const float2 scale =
          make_float2(expf(0.5f * (center + radius_l)), expf(0.5f * (center - radius_l)));
      const float rot = -0.5f * atan2f(2.0f * b, a - c);
      const float w = weights[i];
      float col[3];
#pragma unroll
      for (int k = 0; k < 3; ++k) col[k] = fminf(1.f, fmaxf(0.f, rgb[i * 3 + k] * w));
      float2 pc;
      float3 conic;
      float radius;
      const bool ok = project_gaussian_2d_scale_rot(p, scale, rot, img_size, pc, conic, radius);
      if (!ok) {
        proj_a[i] = make_float4(0.f, 0.f, 0.f, 0.f);
        proj_b[i] = make_float4(0.f, 0.f, 0.f, 0.f);
      } else {
        proj_a[i] = make_float4(pc.x, pc.y, conic.x, conic.y);
        proj_b[i] = make_float4(conic.z, col[0], col[1], col[2]);
        if (radius > 0.f) {
          uint2 tmin, tmax;
          get_tile_bbox(pc, radius, tile_bounds, tmin, tmax);
          const bool everywhere = static_cast<int>((tmax.x - tmin.x) * (tmax.y - tmin.y)) >
                                  SIMPLE_SUM_LARGE_TILE_THRESHOLD;  // reference "large" rule
          if (everywhere) {
            large_ids[atomicAdd(large_count, 1)] = i;
            large_box[i] = make_int4(-1, 0, 0, 0);
          } else {
            // Ellipse {0.5 d^T C d <= cut}: half extents sqrt(2 cut Sigma_xx/yy), Sigma = C^-1.
            // Pixels sample integer coordinates (px = j).
            const float det = conic.x * conic.z - conic.y * conic.y;
            box = make_int4(tmin.x, tmin.y, tmax.x, tmax.y);
            if (det > 0.f && conic.x > 0.f) {
              const float ex = sqrtf(2.f * kSigmaCut * conic.z / det);
              const float ey = sqrtf(2.f * kSigmaCut * conic.x / det);
              box.x = max(box.x, static_cast<int>(floorf(ceilf(pc.x - ex) / BLOCK_X)));
              box.z = min(box.z, static_cast<int>(floorf(floorf(pc.x + ex) / BLOCK_X)) + 1);
              box.y = max(box.y, static_cast<int>(floorf(ceilf(pc.y - ey) / BLOCK_Y)));
              box.w = min(box.w, static_cast<int>(floorf(floorf(pc.y + ey) / BLOCK_Y)) + 1);
            }
            if (box.z <= box.x || box.w <= box.y) box = make_int4(0, 0, 0, 0);
          }
        }
      }
      const int nt = (box.z - box.x) * (box.w - box.y);
      mode = nt == 0 ? 0 : (nt <= kMaxSlots ? 1 : 2);
      proj_box[i] = box;
      if (mode == 2) {
        proj_slot[i * kMaxSlots] = -1;
        wide_ids[atomicAdd(wide_count, 1)] = i;  // counted by k_count_wide afterwards
        mode = 0;
      }
    }
  }
  const int bw = box.z - box.x;
  const int nt = mode ? bw * (box.w - box.y) : 0;
  const int lane = threadIdx.x & 31;
  const int max_nt = __reduce_max_sync(0xffffffffu, nt);
  for (int q = 0; q < max_nt; ++q) {
    const int key = q < nt ? (box.y + q / bw) * static_cast<int>(tile_bounds.x) + box.x + q % bw : -1;
    const unsigned peers = __match_any_sync(0xffffffffu, key);
    const int leader = __ffs(peers) - 1;
    int base = 0;
    if (key >= 0 && lane == leader) base = atomicAdd(&tile_counts[key], __popc(peers));
    base = __shfl_sync(0xffffffffu, base, leader);
    if (key >= 0 && mode == 1) proj_slot[i * kMaxSlots + q] = base + __popc(peers & ((1u << lane) - 1u));
  }
}

// Wide points (> kMaxSlots tiles) are counted after every slot-recorded point has its slot, so
// their entries occupy the tail [slot-recorded count, total) of each tile list.
__global__ void k_count_wide(const int* __restrict__ wide_ids, const int* wide_count,
                             const int4* __restrict__ proj_box, int tiles_x, int* tile_counts) {
  const int n = *wide_count;
  for (int e = blockIdx.x * blockDim.x + threadIdx.x; e < n; e += gridDim.x * blockDim.x) {
    const int4 box = proj_box[wide_ids[e]];
    for (int ty = box.y; ty < box.w; ++ty)
      for (int tx = box.x; tx < box.z; ++tx) atomicAdd(&tile_counts[ty * tiles_x + tx], 1);
  }
}

__global__ void k_scatter_tiles(const int* n_ptr, int n_cap, const int4* __restrict__ proj_box,
                                const int* __restrict__ proj_slot, int tiles_x,
                                const int* __restrict__ tile_offsets, int* tile_cursors,
                                int capacity, int* overflow, int* tile_ids) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_cap || i >= *n_ptr) return;
  const int4 box = proj_box[i];
  if (box.z <= box.x) return;
  const int4 s0 = reinterpret_cast<const int4*>(proj_slot)[2 * i];
  if (s0.x < 0) {
    for (int ty = box.y; ty < box.w; ++ty)
      for (int tx = box.x; tx < box.z; ++tx) {
        const int tile = ty * tiles_x + tx;
        const int at = tile_offsets[tile + 1] - 1 - atomicAdd(&tile_cursors[tile], 1);
        if (at < capacity) tile_ids[at] = i;
        else atomicExch(overflow, 1);
      }
    return;
  }
  const int4 s1 = reinterpret_cast<const int4*>(proj_slot)[2 * i + 1];
  const int slots[kMaxSlots] = {s0.x, s0.y, s0.z, s0.w, s1.x, s1.y, s1.z, s1.w};
  int q = 0;
  for (int ty = box.y; ty < box.w; ++ty)
    for (int tx = box.x; tx < box.z; ++tx) {
      const int at = tile_offsets[ty * tiles_x + tx] + slots[q++];
      if (at < capacity) tile_ids[at] = i;
      else atomicExch(overflow, 1);
    }
}

// Simple-sum raster (nd_rasterize_simple_sum_forward_raw math) for a 16x16 tile with 16x8
// threads, two pixels (rows y and y + 8) per thread so every shared-memory Gaussian load feeds
// two evaluations; packed 128-bit shared loads, exact sigma pre-cull before the exp, planar
// clamped output and fused squared error.
constexpr int kRasterThreads = BLOCK_X * BLOCK_Y / 2;

// sigma = 0.5 (cx dx^2 + cz dy^2) + cy dx dy with the exact rounding sequence nvcc emits for the
// reference kernel's expression (so terms at the alpha = 1/255 cut flip identically).
__device__ __forceinline__ float sigma_ref(float cx, float cy, float cz, float dx, float dy) {
  const float r = __fmul_rn(dy, __fmul_rn(dy, cz));
  const float s = __fmul_rn(__fmaf_rn(dx, __fmul_rn(dx, cx), r), 0.5f);
  return __fmaf_rn(dy, __fmul_rn(dx, cy), s);
}
__global__ void __launch_bounds__(kRasterThreads)
k_raster(int tiles_x, int W, int H, const int* __restrict__ tile_ids,
         const int* __restrict__ tile_offsets, const int* __restrict__ large_ids,
         const int* __restrict__ large_count, const int4* __restrict__ large_box,
         const float4* __restrict__ proj_a, const float4* __restrict__ proj_b, const int* n_ptr,
         int capacity, const int* __restrict__ overflow_flag, float* __restrict__ out,
         const float* __restrict__ image, double* sq_err, const float* __restrict__ counts,
         __half* __restrict__ packed16) {
  const int tile_id = blockIdx.y * tiles_x + blockIdx.x;
  const int j = blockIdx.x * BLOCK_X + threadIdx.x;
  const int i0 = blockIdx.y * BLOCK_Y + threadIdx.y;
  const int i1 = i0 + BLOCK_Y / 2;
  const float px = static_cast<float>(j), py0 = static_cast<float>(i0), py1 = static_cast<float>(i1);
  const bool overflow = __ldg(overflow_flag) != 0;
  const int large_total = __ldg(large_count);
  const int range_start = overflow ? 0 : tile_offsets[tile_id];
  const int range_end = overflow ? *n_ptr : min(tile_offsets[tile_id + 1], capacity);
  constexpr int kBatch = BLOCK_X * BLOCK_Y;
  const int nb = (range_end - range_start + kBatch - 1) / kBatch;
  const int nlb = overflow ? 0 : (large_total + kBatch - 1) / kBatch;
  const int tr = threadIdx.y * BLOCK_X + threadIdx.x;

  __shared__ float4 s_a[kBatch];
  __shared__ float4 s_b[kBatch];
  float a0 = 0.f, a1 = 0.f, a2 = 0.f, b0 = 0.f, b1 = 0.f, b2 = 0.f;

  for (int pass = 0; pass < 2; ++pass) {
    const int batches = pass == 0 ? nb : nlb;
    for (int bi = 0; bi < batches; ++bi) {
      const int bstart = (pass == 0 ? range_start : 0) + kBatch * bi;
      const int bend = pass == 0 ? range_end : large_total;
      __syncthreads();
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int slot = tr + h * kRasterThreads;
        const int idx = bstart + slot;
        if (idx < bend) {
          const int g = pass == 0 ? (overflow ? idx : tile_ids[idx]) : large_ids[idx];
          s_a[slot] = proj_a[g];
          float4 bv = proj_b[g];
          if (pass == 1) {  // boxed large entries contribute only inside their tile box
            const int4 lb = large_box[g];
            if (lb.x >= 0 && !(static_cast<int>(blockIdx.x) >= lb.x &&
                               static_cast<int>(blockIdx.x) < lb.z &&
                               static_cast<int>(blockIdx.y) >= lb.y &&
                               static_cast<int>(blockIdx.y) < lb.w))
              bv = make_float4(INFINITY, 0.f, 0.f, 0.f);  // sigma -> inf/NaN: skipped
          }
          s_b[slot] = bv;
        }
      }
      __syncthreads();
      const int bs = min(kBatch, bend - bstart);
      for (int t = 0; t < bs; ++t) {
        const float4 A = s_a[t];
        const float4 B = s_b[t];
        const float dx = A.x - px;
        {
          const float dy = A.y - py0;
          const float sigma = sigma_ref(A.z, A.w, B.x, dx, dy);
          if (sigma <= kSigmaCut && sigma >= 0.f) {  // also drops NaN
            const float alpha = __expf(-sigma);
            if (alpha >= 1.f / 255.f) {
              a0 += B.y * alpha;
              a1 += B.z * alpha;
              a2 += B.w * alpha;
            }
          }
        }
        {
          const float dy = A.y - py1;
          const float sigma = sigma_ref(A.z, A.w, B.x, dx, dy);
          if (sigma <= kSigmaCut && sigma >= 0.f) {
            const float alpha = __expf(-sigma);
            if (alpha >= 1.f / 255.f) {
              b0 += B.y * alpha;
              b1 += B.z * alpha;
              b2 += B.w * alpha;
            }
          }
        }
      }
    }
  }

  float se = 0.f;
  const int P = W * H;
  auto emit = [&](int i, float c0, float c1, float c2) {
    if (i >= H || j >= W) return;
    const int pix = i * W + j;
    const float v[3] = {fminf(1.f, fmaxf(0.f, c0)), fminf(1.f, fmaxf(0.f, c1)),
                        fminf(1.f, fmaxf(0.f, c2))};
#pragma unroll
    for (int c = 0; c < 3; ++c) {
      out[c * P + pix] = v[c];
      if (sq_err) {
        const float d = v[c] - image[c * P + pix];
        se += d * d;
      }
    }
    if (packed16) {  // C1 unit 1: rendered r, g, b, count | rgb 1, 2, 0, 0 (kept)
      uint4* unit = reinterpret_cast<uint4*>(packed16) + P + pix;
      const uint2 keep = reinterpret_cast<const uint2*>(unit)[1];
      const __half2 h01 = __halves2half2(__float2half(v[0]), __float2half(v[1]));
      const __half2 h23 = __halves2half2(__float2half(v[2]), __float2half(__ldg(counts + pix)));
      *unit = make_uint4(*reinterpret_cast<const unsigned*>(&h01),
                         *reinterpret_cast<const unsigned*>(&h23), keep.x, keep.y);
    }
  };
  emit(i0, a0, a1, a2);
  emit(i1, b0, b1, b2);
  if (sq_err) {
    for (int o = 16; o > 0; o >>= 1) se += __shfl_down_sync(0xffffffffu, se, o);
    __shared__ float s_se[kRasterThreads / 32];
    if ((tr & 31) == 0) s_se[tr >> 5] = se;
    __syncthreads();
    if (tr == 0) {
      double t = 0.0;
      for (int w = 0; w < kRasterThreads / 32; ++w) t += s_se[w];
      atomicAdd(sq_err, t);
    }
  }
}

struct Bilinear {
  int i00, i01, i10, i11;  // -1 = outside (zero padding)
  float w00, w01, w10, w11;
};
__device__ inline Bilinear bilinear(float2 p, int H, int W) {
  const float xp = p.x * W - 0.5f, yp = p.y * H - 0.5f;
  const int x0 = static_cast<int>(floorf(xp)), y0 = static_cast<int>(floorf(yp));
  const float fx = xp - x0, fy = yp - y0;
  const bool x0v = x0 >= 0 && x0 < W, x1v = x0 + 1 >= 0 && x0 + 1 < W;
  const bool y0v = y0 >= 0 && y0 < H, y1v = y0 + 1 >= 0 && y0 + 1 < H;
  Bilinear b;
  b.i00 = (x0v && y0v) ? y0 * W + x0 : -1;
  b.i01 = (x0v && y1v) ? (y0 + 1) * W + x0 : -1;
  b.i10 = (x1v && y0v) ? y0 * W + x0 + 1 : -1;
  b.i11 = (x1v && y1v) ? (y0 + 1) * W + x0 + 1 : -1;
  b.w00 = (1.f - fx) * (1.f - fy);
  b.w01 = (1.f - fx) * fy;
  b.w10 = fx * (1.f - fy);
  b.w11 = fx * fy;
  return b;
}
__device__ inline float bsample(const Bilinear& b, const float* m) {
  const float v00 = b.i00 >= 0 ? m[b.i00] : 0.f, v01 = b.i01 >= 0 ? m[b.i01] : 0.f;
  const float v10 = b.i10 >= 0 ? m[b.i10] : 0.f, v11 = b.i11 >= 0 ? m[b.i11] : 0.f;
  return v00 * b.w00 + v01 * b.w01 + v10 * b.w10 + v11 * b.w11;
}

__global__ void k_sample(const int* n_ptr, int n_cap, const float2* __restrict__ xy,
                         const float* __restrict__ map, int C, int H, int W, bool clamp01,
                         const float* diag_shift, float* out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_cap || i >= *n_ptr) return;
  const Bilinear b = bilinear(xy[i], H, W);
  const float shift = diag_shift ? *diag_shift : 0.f;
  const size_t P = static_cast<size_t>(H) * W;
  for (int c = 0; c < C; ++c) {
    float v = bsample(b, map + c * P);
    if (c == 0 || c == 2) v += shift;
    if (clamp01) v = fminf(1.f, fmaxf(0.f, v));
    out[i * C + c] = v;
  }
}

__global__ void k_apply(const int* n_ptr, int n_cap, const float* __restrict__ corr, int H, int W,
                        float xy_step, float2* xy, float* cov, float* rgb) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_cap || i >= *n_ptr) return;
  const float2 p = xy[i];
  const Bilinear b = bilinear(p, H, W);
  const size_t P = static_cast<size_t>(H) * W;
  float d[8];
#pragma unroll
  for (int c = 0; c < 8; ++c) d[c] = bsample(b, corr + c * P);
#pragma unroll
  for (int k = 0; k < 3; ++k) cov[i * 3 + k] += d[k];
#pragma unroll
  for (int k = 0; k < 3; ++k) rgb[i * 3 + k] = fminf(1.f, fmaxf(0.f, rgb[i * 3 + k] + d[3 + k]));
  xy[i] = make_float2(fminf(1.f, fmaxf(0.f, p.x + xy_step * d[6])),
                      fminf(1.f, fmaxf(0.f, p.y + xy_step * d[7])));
}

__global__ void k_blend_history(const int* n, int cap, const int* slots, float blend,
                                const float* history, float* cov, float* rgb) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= cap || i >= *n || slots[i] < 0) return;
  const float* prev = history + 6 * slots[i];
#pragma unroll
  for (int c = 0; c < 3; ++c) {
    cov[3 * i + c] = blend * prev[c] + (1.f - blend) * cov[3 * i + c];
    rgb[3 * i + c] = fminf(1.f, fmaxf(0.f, blend * prev[3 + c] + (1.f - blend) * rgb[3 * i + c]));
  }
}

__global__ void k_store_history(const int* n, int cap, const int* slots, const float* cov,
                                const float* rgb, float* history) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= cap || i >= *n) return;
  float* dst = history + 6 * slots[i];
#pragma unroll
  for (int c = 0; c < 3; ++c) {
    dst[c] = cov[3 * i + c];
    dst[3 + c] = rgb[3 * i + c];
  }
}

__global__ void k_premultiply(const int* n_ptr, int n_cap, const float* __restrict__ rgb,
                              const float* __restrict__ w, float* out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_cap || i >= *n_ptr) return;
#pragma unroll
  for (int k = 0; k < 3; ++k) out[i * 3 + k] = fminf(1.f, fmaxf(0.f, rgb[i * 3 + k] * w[i]));
}

}  // namespace

void plan_render(int H, int W, int n_cap, Arena& a, RenderBuffers* r) {
  r->H = H;
  r->W = W;
  r->n_cap = n_cap;
  r->tiles_x = (W + BLOCK_X - 1) / BLOCK_X;
  r->tiles_y = (H + BLOCK_Y - 1) / BLOCK_Y;
  r->num_tiles = r->tiles_x * r->tiles_y;
  r->capacity = std::max(8 * n_cap, 4096);
  const int nt = r->num_tiles;
  r->zero_bytes = sizeof(int) * (3 + 2 * static_cast<size_t>(nt));
  r->zero_block = a.take<int>(3 + 2 * static_cast<size_t>(nt));
  if (r->zero_block) {
    r->overflow = r->zero_block;
    r->large_count = r->zero_block + 1;
    r->wide_count = r->zero_block + 2 + 2 * nt;
    r->tile_counts = r->zero_block + 2;
    r->tile_cursors = r->zero_block + 2 + nt;
  }
  r->tile_offsets = a.take<int>(nt + 1);
  r->tile_ids = a.take<int>(r->capacity);
  r->large_ids = a.take<int>(n_cap);
  r->proj_a = a.take<float4>(n_cap);
  r->proj_b = a.take<float4>(n_cap);
  r->proj_box = a.take<int4>(n_cap);
  r->proj_slot = a.take<int>(8 * static_cast<size_t>(n_cap));
  r->large_box = a.take<int4>(n_cap);
  r->wide_ids = a.take<int>(n_cap);
  size_t t = 0;
  cub::DeviceScan::InclusiveSum(nullptr, t, r->tile_counts, r->tile_offsets, nt);
  r->cub_temp_bytes = t;
  r->cub_temp = a.take<uint8_t>(t);
}

cudaError_t init_render(RenderBuffers& r, cudaStream_t s) {
  return cudaMemsetAsync(r.tile_offsets, 0, sizeof(int) * (r.num_tiles + 1), s);
}

cudaError_t render_points(RenderBuffers& r, const int* n, const float2* xy, const float* cov,
                          const float* rgb, const float* weights, float* out_chw,
                          const float* image_chw, double* sq_err, cudaStream_t s,
                          float* point_counts, __half* packed16) {
  if (packed16 && !point_counts) return cudaErrorInvalidValue;
  const dim3 img(r.W, r.H, 1), tb(r.tiles_x, r.tiles_y, 1);
  GF_TRY(cudaMemsetAsync(r.zero_block, 0, r.zero_bytes, s));
  if (point_counts) GF_TRY(cudaMemsetAsync(point_counts, 0, sizeof(float) * r.H * r.W, s));
  k_project<<<grid_for(r.n_cap), 256, 0, s>>>(n, r.n_cap, xy, cov, rgb, weights, img, tb,
                                               r.proj_a, r.proj_b, r.proj_box, r.proj_slot,
                                               r.tile_counts, r.large_count, r.large_ids,
                                               r.large_box, r.wide_ids, r.wide_count,
                                               point_counts);
  k_count_wide<<<32, 256, 0, s>>>(r.wide_ids, r.wide_count, r.proj_box, r.tiles_x,
                                  r.tile_counts);
  size_t t = r.cub_temp_bytes;
  GF_TRY(cub::DeviceScan::InclusiveSum(r.cub_temp, t, r.tile_counts, r.tile_offsets + 1,
                                       r.num_tiles, s));
  k_scatter_tiles<<<grid_for(r.n_cap), 256, 0, s>>>(n, r.n_cap, r.proj_box, r.proj_slot,
                                                     r.tiles_x, r.tile_offsets, r.tile_cursors,
                                                     r.capacity, r.overflow, r.tile_ids);
  k_raster<<<tb, dim3(BLOCK_X, BLOCK_Y / 2), 0, s>>>(r.tiles_x, r.W, r.H, r.tile_ids, r.tile_offsets,
                                                 r.large_ids, r.large_count, r.large_box,
                                                 r.proj_a, r.proj_b,
                                                 n, r.capacity, r.overflow, out_chw, image_chw,
                                                 sq_err, point_counts, packed16);
  return cudaGetLastError();
}

cudaError_t sample_point_features(const int* n, const float2* xy, int n_cap, const float* map,
                                  int C, int H, int W, bool clamp01, const float* diag_shift,
                                  float* out, cudaStream_t s) {
  k_sample<<<grid_for(n_cap), 256, 0, s>>>(n, n_cap, xy, map, C, H, W, clamp01, diag_shift, out);
  return cudaGetLastError();
}

cudaError_t apply_correction(const int* n, int n_cap, const float* corr, int H, int W,
                             float xy_step, float2* xy, float* cov, float* rgb, cudaStream_t s) {
  k_apply<<<grid_for(n_cap), 256, 0, s>>>(n, n_cap, corr, H, W, xy_step, xy, cov, rgb);
  return cudaGetLastError();
}

cudaError_t blend_point_history(const int* n, int cap, const int* history_slots, float blend,
                                const float* history, float* cov, float* rgb, cudaStream_t s) {
  k_blend_history<<<grid_for(cap), 256, 0, s>>>(n, cap, history_slots, blend, history, cov, rgb);
  return cudaGetLastError();
}

cudaError_t store_point_history(const int* n, int cap, const int* slots, const float* cov,
                                const float* rgb, float* history, cudaStream_t s) {
  k_store_history<<<grid_for(cap), 256, 0, s>>>(n, cap, slots, cov, rgb, history);
  return cudaGetLastError();
}

cudaError_t premultiply_colour(const int* n, int n_cap, const float* rgb, const float* weights,
                               float* colour_out, cudaStream_t s) {
  k_premultiply<<<grid_for(n_cap), 256, 0, s>>>(n, n_cap, rgb, weights, colour_out);
  return cudaGetLastError();
}

}  // namespace gfused
