// Raw-pointer simple-sum renderer and K-loop point kernels with a device-resident count.
//
// The renderer reproduces gaussifier's native simple-sum forward
// (native_simple_sum_backend/cuda/csrc: simple_sum_project_partition_tile_hits ->
// cub inclusive scan -> simple_sum_project_scatter_small_to_tiles_bounded ->
// nd_rasterize_simple_sum_forward_raw), reusing its device helpers (helpers.cuh), with:
//   * the K-loop pre-render step (log-cov -> scale/rotation, colour * weight clamped) fused
//     into the projection kernel;
//   * launch extents fixed at the planned capacity, gated on the device count;
//   * output written planar (3 x H x W) and clamped to [0, 1] (what the head consumes), and
//     an optional fused squared-error accumulation against the input image.

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "pipeline_common.cuh"

namespace gfused {

struct RenderBuffers {
  int H = 0, W = 0, n_cap = 0;
  int tiles_x = 0, tiles_y = 0, num_tiles = 0;
  int capacity = 0;              // tile-id list capacity (8 * n_cap, at least 4096)
  int* zero_block = nullptr;     // [overflow, large_count, tile_counts(nt), tile_cursors(nt), wide_count]
  int* overflow = nullptr;
  int* large_count = nullptr;
  int* tile_counts = nullptr;
  int* tile_cursors = nullptr;
  int* tile_offsets = nullptr;   // nt + 1, [0] stays 0
  int* tile_ids = nullptr;       // capacity
  int* large_ids = nullptr;      // n_cap
  float4* proj_a = nullptr;      // n_cap: pixel centre x, y, conic xx, conic xy
  float4* proj_b = nullptr;      // n_cap: conic yy, weight-premultiplied clamped colour r, g, b
  int4* proj_box = nullptr;      // n_cap: tile range [x0, y0, x1, y1) (empty if not binned)
  int* proj_slot = nullptr;      // 8 * n_cap: slot of the point in each of its (<= 8) tile lists
  int* wide_ids = nullptr;       // n_cap: points binned into more than 8 tiles
  int* wide_count = nullptr;     // in zero_block
  int4* large_box = nullptr;     // n_cap: tile box of a large-list entry (x = -1: every tile)
  void* cub_temp = nullptr;
  size_t cub_temp_bytes = 0;
  size_t zero_bytes = 0;
};

void plan_render(int H, int W, int n_cap, Arena& arena, RenderBuffers* r);
// Zeroes the persistent parts (tile_offsets[0]); call once after binding, outside capture.
cudaError_t init_render(RenderBuffers& r, cudaStream_t stream);

// Renders the first *n points. cov: (N, 3) log-covariance [a, b, c]; rgb: (N, 3) latent colour;
// weights: (N,). out_chw: 3 x H x W clamped to [0, 1]. If image_chw and sq_err are non-null,
// atomically adds sum((out - image)^2) to *sq_err (double).
cudaError_t render_points(RenderBuffers& r, const int* n, const float2* xy, const float* cov,
                          const float* rgb, const float* weights, float* out_chw,
                          const float* image_chw, double* sq_err, cudaStream_t stream,
                          float* point_counts = nullptr, __half* packed16 = nullptr);
// point_counts (nullable, H x W): also writes the hard point-count map of these positions
// (the head's count channel), fused into the projection pass.
// packed16 (nullable, requires point_counts; contract C1, layout [2][H][W][8] halves): the
// raster also rewrites each pixel's dynamic unit packed16[(P + pix) * 8 ..+8) =
// __float2half(clamped render r, g, b, point count) | its existing halves 4..7 (rgb 1, 2, 0, 0),
// one full 16-byte store (a partial-sector 8-byte store made the head's next kernel pay
// ~30 us/call of L2 read-modify-write at 512x768).

// Bilinear (align_corners=False, zero padding) sample of a C x H x W map at the points.
// Writes (N, C); `clamp01` clamps the result; `diag_shift` (device scalar, nullable) is added to
// channels 0 and 2 (count-override covariance rescale).
cudaError_t sample_point_features(const int* n, const float2* xy, int n_cap, const float* map,
                                  int C, int H, int W, bool clamp01, const float* diag_shift,
                                  float* out, cudaStream_t stream);

// Per-slot history holds the previous frame's converged log-covariance and latent RGB.
cudaError_t blend_point_history(const int* n, int cap, const int* history_slots, float blend,
                                const float* history, float* cov, float* rgb, cudaStream_t stream);
cudaError_t store_point_history(const int* n, int cap, const int* slots, const float* cov,
                                const float* rgb, float* history, cudaStream_t stream);

// Applies the 8-channel correction (8 x H x W planar) sampled at each point: cov += d[0:3],
// rgb = clamp(rgb + d[3:6]), xy = clamp(xy + xy_step * d[6:8]).
cudaError_t apply_correction(const int* n, int n_cap, const float* corr, int H, int W,
                             float xy_step, float2* xy, float* cov, float* rgb,
                             cudaStream_t stream);

// colour_out = clamp(rgb * weight, 0, 1) for the first *n points.
cudaError_t premultiply_colour(const int* n, int n_cap, const float* rgb, const float* weights,
                               float* colour_out, cudaStream_t stream);

}  // namespace gfused
