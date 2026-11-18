// Torch-free equal-mass Voronoi (EMV) sampler whose point count lives on the device.
//
// Port of the native EMV sampler (src/gaussifier_sampler/native_voronoi): error-diffusion
// oversample -> greedy mass-preserving merge rounds with an exact-count contract -> Lloyd
// polish, then the Voronoi cell masses that become the per-point render weights. Voronoi cells
// are exact nearest-seed assignments from a grid-hash search (no jump flooding).
//
// Unlike the libtorch implementation, every quantity that depends on the requested count
// (error-diffusion tile size, merge schedule, launch extents) is computed on the device, so a
// single launch sequence serves every count up to the planned capacity and the whole sampler
// can be captured once in a CUDA graph and replayed for any image of the same resolution.
// All kernels launch over the planned capacity and gate on the device-side counts.

#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "pipeline_common.cuh"

namespace gfused {

// Device-resident scalar state shared by the sampler kernels.
struct SamplerCtrl {
  int n;            // requested final count (written by the caller before run_sampler)
  int n_init;       // error-diffusion count: max(n + 1, round(oversample * n))
  int tile_size, tiles_x, tiles_y, tile_total, shift_x, shift_y;
  int valid_total;  // density has positive finite mass
  int round_n_in;   // current merge round: input count
  int round_remove; // current merge round: points to remove
  int n_cur;        // alive count after the last compaction (CUB DeviceSelect output)
  int nn_count;     // merge fallback: candidates needing a nearest-survivor search
  int ed_rounded_sum;
  // diagnostics (accumulated over the run; reset at the start)
  int stat_matched;       // removed by adjacency matching
  int stat_forced_adj;    // unmatched candidates merged into first live adjacent cell
  int stat_forced_nn;     // unmatched candidates merged into nearest survivor
  int stat_count_error;   // |final alive count - n| (must be 0)
  float weight_sum;       // sum of final cell masses
};

// Device-resident Voronoi grid-hash parameters (rebuilt by every assignment).
struct GridCtrl {
  int n;                 // seeds in this assignment
  int s;                 // cell side in pixels
  int gx, gy;            // level-0 cells
  int tiles_x, tiles_y;  // 64x64-cell tiles
  int n_keys;            // tiles_x * tiles_y * 64 * 64
};

// Device-resident radix-select state of the merge-round candidate selection.
struct SelCtrl {
  unsigned long long prefix;  // key bits fixed so far
  unsigned long long thr;     // key of rank round_remove (candidates: key < thr)
  int rank;                   // rank still sought inside the prefix bucket
  int bits;                   // key bits below the prefix
  int list;                   // next pass gathers the bucket instead of a histogram
  int done;
  int list_n;
  int pad;
};

struct SamplerConfig {
  int H = 0, W = 0;
  int n_cap = 0;               // largest count the plan supports
  int64_t seed = 12345;
  double oversample = 1.5;
  int merge_rounds = 2;        // 1.5x oversample with 1/3 per-round cap needs at most 2
  double per_round_remove_fraction = 1.0 / 3.0;
  int knn_k = 8;
  int lloyd_iters = 3;
};

struct SamplerBuffers {
  SamplerConfig cfg;
  int init_cap = 0;    // capacity for the oversampled point set
  int max_tiles = 0;   // error-diffusion tiles at the smallest tile size (4)
  SamplerCtrl* ctrl = nullptr;
  float* density_s = nullptr;   // sampling density (caller-filled, H*W)
  float2* pts[2] = {};          // double-buffered (init_cap)
  float2* pts_tmp = nullptr;    // merged positions before compaction
  float2* pts_pixel = nullptr;  // clamped pixel-space seed positions (init_cap)
  int* owner = nullptr;         // H*W nearest seed per pixel (merge-round assignments only)
  // Voronoi grid hash (see pipeline_sampler.cu): counting sort of the seeds by cell key
  GridCtrl* grid = nullptr;
  int grid_max_keys = 0;        // key capacity at the smallest cell side
  int* grid_cnt = nullptr;      // grid_max_keys + 1
  int* grid_start = nullptr;    // grid_max_keys + 1 (exclusive scan of grid_cnt)
  int* grid_key = nullptr;      // init_cap
  int* grid_rank = nullptr;     // init_cap
  float2* grid_pos = nullptr;   // init_cap, sorted by key
  int* grid_idx = nullptr;      // init_cap, sorted by key
  int* grid_open_count = nullptr;   // 8x4 pixel tiles the quad pass left to the full search
  int* grid_open_tiles = nullptr;   // list of those tiles
  unsigned* grid_open_mask = nullptr;  // per tile: open pixels
  float2* grid_open_best = nullptr; // H*W (best distance, best index bits) of open pixels
  float* mass = nullptr;        // init_cap
  int* nbr = nullptr;           // init_cap * 16 distinct-neighbour sets
  int* top_idx = nullptr;       // init_cap * knn_k (ascending neighbour mass)
  unsigned long long* key = nullptr;   // init_cap selection / matching priority keys
  SelCtrl* sel = nullptr;
  unsigned* sel_hist = nullptr;        // 4096 digit bins (zero between passes)
  unsigned long long* sel_list = nullptr;  // gathered bucket (<= 4096 used)
  uint8_t* cand = nullptr, *alive = nullptr, *matched = nullptr, *flag = nullptr;
  int* target = nullptr;
  unsigned long long* myp[2] = {};   // matching: proposal key per column (~0 = none)
  unsigned long long* tmin[3] = {};  // matching: smallest proposal received per column
  int* nn_list = nullptr;
  float* wsum = nullptr, *xacc = nullptr, *yacc = nullptr;  // Lloyd accumulators
  float2* wpos = nullptr;     // merge accumulators
  float* mpost = nullptr;
  // error diffusion
  double* tile_mass = nullptr;  // max_tiles
  double* tile_cum = nullptr;   // max_tiles
  int* tile_count = nullptr;    // max_tiles
  int* tile_off = nullptr;      // max_tiles
  void* cub_temp = nullptr;
  size_t cub_temp_bytes = 0;
};

// Plans (base == nullptr: measure) or binds all sampler buffers in `arena`.
void plan_sampler(const SamplerConfig& cfg, Arena& arena, SamplerBuffers* b);

// Runs the sampler. Reads ctrl->n (set by the caller on the stream) and b.density_s.
//   density_w : density whose cell masses become the point weights (H*W)
//   points_out: final points, n_cap rows, first ctrl->n valid
//   weights_out: per-point weight = cell_mass * n / sum(cell_mass)
// Everything is enqueued on `stream` without host synchronisation or allocation.
cudaError_t run_sampler(SamplerBuffers& b, const float* density_w, float2* points_out,
                        float* weights_out, cudaStream_t stream);

struct VideoTrackOptions {
  double count_ema = 0.9;
  double churn_fraction = 0.002;
  double max_count_rate = 0.05;
  float tau_hot = 0.9f;
  float tau_cold = 0.5f;
  int patience = 12;
};

struct VideoTrackStats {
  int target, limited_target, survivors, killed, born, shortfall;
};

struct VideoTrackCtrl {
  VideoTrackStats stats;
  double smoothed_count;
  int64_t next_id;
  int count, free_count, kept, hot, cold, eligible, tentative_kills, tentative_splits;
};

struct VideoTrack {
  int64_t id;
  float2 xy;
  int slot, strikes;
  bool has_history;
};

struct VideoTracks {
  VideoTrackCtrl* ctrl = nullptr;
  VideoTrack* tracks = nullptr;   // repaired state for the next frame
  VideoTrack* updated = nullptr;
  int64_t* ids = nullptr;         // current frame, aligned with Gaussian rows
  int* slots = nullptr;
  int* history_slots = nullptr;   // -1 for a birth (even when its slot was reused)
  int* free_slots = nullptr;
  int* hilbert_order = nullptr;
  double3* hilbert_prefix = nullptr;
  int* pixel_head = nullptr;
  int* pixel_next = nullptr;
  int* pixel_count = nullptr;
  int* kill_rank = nullptr;
  int* kept_rows = nullptr;
  unsigned long long* rank_keys[2] = {};
  float2* child_xy = nullptr;
  uint8_t* keep = nullptr;
  void* cub_temp = nullptr;
  size_t cub_temp_bytes = 0;
};

void plan_video_tracks(const SamplerConfig& cfg, Arena& arena, VideoTracks* v);
cudaError_t init_video_tracks(const SamplerConfig& cfg, VideoTracks& v);
cudaError_t begin_video_frame(SamplerBuffers& b, VideoTracks& v, const VideoTrackOptions& opts,
                              bool first, cudaStream_t stream);
cudaError_t sample_video_frame(SamplerBuffers& b, VideoTracks& v, const float* density_w,
                                bool first, float2* points, float* weights, cudaStream_t stream);
cudaError_t repair_video_tracks(SamplerBuffers& b, VideoTracks& v, const VideoTrackOptions& opts,
                                 const float* density, const float2* final_xy, cudaStream_t stream);

}  // namespace gfused
