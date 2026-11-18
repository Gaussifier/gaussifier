// Torch-free end-to-end Gaussifier inference: forward map -> point count -> EMV sampler ->
// K-state correction loop, all enqueued on one stream from one device arena, with no host
// synchronisation, allocation or host<->device copies inside run(), so the whole run is
// captured as a single CUDA graph per (H, W, K) and replayed for every image of that size.
// Video captures separate graphs for the first frame and for subsequent frames.
//
//   FusedPipeline p;
//   p.load(opts, &err);                  // forward-map GFFM + correction-head GFCH blobs
//   p.prepare(H, W);                     // one arena for everything the pipeline owns
//   cudaMemcpyAsync(p.input_chw(), host, 3*H*W*4, H2D, stream);
//   p.run_graph(K, stream);              // capture once, then replay
//   ... read p.outputs() after synchronising; the point count is p.outputs().count[0].
//
// Semantics: gaussifier render_loop._correction_render_loss with the frozen-GN release config
// (n = floor(sum rate), capped at 0.40 points/pixel and 1e6; EMV placement with seed; hard
// points; per-point log-cov / rgb sampled from the forward-map maps; weights from Voronoi cell
// masses; K rendered states with K-1 head updates; xy step 2.56 / max(H, W); colour
// premultiplied by weight and clamped).

#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>

#include "correction_head.cuh"
#include "forward_map.cuh"
#include "kloop_render.cuh"
#include "pipeline_sampler.cuh"

namespace gfused {

enum class SizeMode { Pad = 0, Exact = 1 };

struct PipelineOptions {
  std::string forward_map_path;  // GFFM blob from scripts/export_fused_weights.py
  std::string head_path;         // GFCH blob from scripts/export_fused_head.py
  int64_t seed = 12345;
  int n_override = 0;            // 0 = adaptive floor(sum rate); > 0 = fixed count
  // With n_override above the model's own count, shift the log-cov diagonal so the splatted
  // mass is kept (production semantics). false = pin the count only (validation).
  bool count_rescale = true;
  double points_per_pixel_max = 0.40;
  int render_n_max = 1000000;
  int lloyd_iters = 3;
  int knn_k = 8;
  int merge_rounds = 2;
  float xy_step_pixels = 2.56f;
  int max_k = 16;
  // Adaptive K (both <= 0: fixed K, the default). The K passed to run()/run_graph() is the cap.
  // After each rendered state k (1-based) the loop stops when
  //   psnr_k >= psnr_target                          (psnr_target > 0), or
  //   k >= 2 and psnr_k - psnr_{k-1} < min_gain      (min_gain > 0).
  // In run_graph() the loop is a CUDA conditional WHILE node decided on the GPU; in run()
  // (eager) it is a host loop that reads the decision back after each state.
  float psnr_target = 0.f;
  float min_gain = 0.f;
  bool keep_initial = false;  // retain initial points and attributes for host consumers
  // Validation hook: skip the forward map; the caller fills external_maps() before each run.
  bool external_maps = false;
  // Debug: before every head call verify the packed head input (contract C1, written by the
  // pipeline and the renderer) against __float2half of the FP32 sources, counting mismatching
  // values in PipelineStats::io_mismatch. Not for timing.
  bool check_head_io = false;
  SizeMode size_mode = SizeMode::Exact;
  bool video = false;
  VideoTrackOptions video_tracks;
  // A small history contribution anchors the correction loop to the current feature maps.
  float warm_attribute_blend = 0.1f;
};

// Forward-map output buffers (device), for external_maps mode.
struct PipelineMaps {
  float* density;   // H x W
  float* log_cov;   // 3 x H x W
  float* rgb;       // 3 x H x W
  float* rate_sum;  // [1]
};

// Device pointers to the final state (valid after the stream completes).
struct PipelineOutputs {
  const int* count;        // [1] number of Gaussians N
  const float2* xy;        // (N, 2) in [0, 1]
  const float* log_cov;    // (N, 3) [a, b, c]
  const float* rgb;        // (N, 3) latent colour (un-premultiplied)
  const float* colour;     // (N, 3) clamp(rgb * weight, 0, 1): what the renderer consumes
  const float* weights;    // (N,)
  const float* render;     // 3 x H x W final render, clamped to [0, 1]
  const float* density;    // H x W raw learned density
  const float2* initial_xy;
  const float* initial_log_cov;
  const float* initial_rgb;
  const double* sq_err;    // [K] sum of squared error of each rendered state vs the input
  const int64_t* ids;      // [N] stream-local IDs (video only)
  const int* slots;        // [N] reusable, stable slots (video only)
};

// Host-side diagnostics (synchronous read).
struct PipelineStats {
  int n = 0, n_requested = 0, n_auto = 0, n_init = 0;
  float rate_sum = 0.f;
  int ed_tile_size = 0, matched = 0, forced_adj = 0, forced_nn = 0, count_error = 0;
  int k_used = 0;        // rendered states of the last run (== K in fixed mode)
  int io_mismatch = -1;  // check_head_io: mismatching values over the run (-1: not checked)
  VideoTrackStats tracks{};
};

// Device-resident pipeline scalars.
struct PipelineCtrl {
  float rate_sum;
  int n_auto;
  float diag_shift;
  float density_sum;
  int k_done;     // adaptive: states rendered so far
  int stop;       // adaptive: last decision
  double sq_cur;  // adaptive: squared error of the state being rendered
  int io_mismatch;  // check_head_io
};

class FusedPipeline {
 public:
  FusedPipeline() = default;
  ~FusedPipeline();
  FusedPipeline(const FusedPipeline&) = delete;
  FusedPipeline& operator=(const FusedPipeline&) = delete;

  cudaError_t load(const PipelineOptions& opts, std::string* error);
  // Creates an unprepared pipeline sharing the immutable model and options.
  std::unique_ptr<FusedPipeline> make_workspace() const;
  std::unique_ptr<FusedPipeline> make_video_workspace(double count_ema,
                                                      float warm_attribute_blend) const;
  // The neural nets use padded workspaces. All caller-visible buffers use the real dimensions.
  // Re-preparing releases the previous workspace and graphs.
  static bool supports_size(int H, int W);
  cudaError_t prepare(int H, int W, int max_points = 0);

  float* input_chw() { return in_chw_; }  // 3 x H x W float32 [0, 1], caller-filled
  PipelineMaps external_maps() {
    return {density_, external_cov_ ? external_cov_ : log_cov_, rgb_map_, &ctrl_->rate_sum};
  }
  // Enqueues one full run eagerly (also what the graph captures). In video mode each
  // call advances one frame. Keep the same stream across frames and prepare() to reset.
  cudaError_t run(int K, cudaStream_t stream);
  // Captures run(K) on first use for this K (in video mode: the graph of the next frame, first
  // or warm). run_graph also launches the graph.
  cudaError_t capture_graph(int K, cudaStream_t stream);
  cudaError_t run_graph(int K, cudaStream_t stream);
  // Changing sampling options invalidates the arena and its graphs, while keeping weights.
  void set_sampling(int64_t seed, int count);
  // Changing size mode invalidates a prepared odd-size arena and its graphs.
  void set_size_mode(SizeMode mode);
  // Capture both video graphs without advancing the stream, e.g. before benchmarking.
  cudaError_t prepare_video_graphs(int K, cudaStream_t stream);
  PipelineOutputs outputs() const;
  cudaError_t read_stats(PipelineStats* stats, cudaStream_t stream = nullptr) const;  // synchronous

  int H() const { return H_; }
  int W() const { return W_; }
  int n_cap() const { return n_cap_; }
  size_t arena_bytes() const { return arena_bytes_; }
  int graph_nodes(int K) const;

 private:
  void release();
  bool adaptive() const { return opts_.psnr_target > 0.f || opts_.min_gain > 0.f; }
  enum class NetSchedule { Aligned, ReflectPad, Exact };
  cudaError_t enqueue_forward_map(cudaStream_t s);
  cudaError_t run_frame(int K, bool warm, cudaStream_t s);
  cudaError_t capture_graph(int K, bool warm, cudaStream_t s);
  cudaError_t enqueue_initial_state(bool warm, bool pack, cudaStream_t s);
  cudaError_t enqueue_step(cudaStream_t s);              // head update + next render
  cudaError_t enqueue_decide(int K, cudaStream_t s, bool set_handle);
  cudaError_t run_adaptive(int K, bool warm, cudaStream_t s);
  cudaError_t enqueue_head(cudaStream_t s);              // head + correction apply
  cudaError_t enqueue_pixel_prep(bool warm, cudaStream_t s);  // shift, sampling density
  cudaError_t save_initial(cudaStream_t s);
  cudaError_t enqueue_pack(cudaStream_t s);              // static half of the packed head input
  int k_fixed_ = 0;
  unsigned long long cond_handle_ = 0;  // cudaGraphConditionalHandle while capturing
  void plan(Arena& a);

  PipelineOptions opts_;
  struct Model;
  std::shared_ptr<const Model> model_;
  ForwardMapWorkspace fm_ws_{};
  CorrectionHeadWorkspace head_ws_{};

  int H_ = 0, W_ = 0, net_h_ = 0, net_w_ = 0, n_cap_ = 0;
  NetSchedule schedule_ = NetSchedule::Aligned;
  uint8_t* arena_ = nullptr;
  size_t arena_bytes_ = 0;
  SamplerBuffers sb_{};
  VideoTracks tracks_{};
  float* attribute_history_ = nullptr;
  bool video_started_ = false;
  bool last_frame_warm_ = false;
  RenderBuffers rb_{};
  PipelineCtrl* ctrl_ = nullptr;
  float *in_chw_ = nullptr, *in_hwc_ = nullptr;
  float *density_ = nullptr, *log_cov_ = nullptr, *rgb_map_ = nullptr;
  float* external_cov_ = nullptr;
  float *net_image_ = nullptr, *net_cov_ = nullptr, *net_rgb_ = nullptr;
  float* real_logits_ = nullptr;
  float2* xy_ = nullptr;
  float *cov_ = nullptr, *rgb_ = nullptr, *weights_ = nullptr, *colour_ = nullptr;
  float *render_ = nullptr, *count_map_ = nullptr, *corr_ = nullptr;
  float2* initial_xy_ = nullptr;
  float *initial_cov_ = nullptr, *initial_rgb_ = nullptr;
  double* sq_err_ = nullptr;
  __half* packed16_ = nullptr;  // [2][H][W][8] packed head input (C1)
  std::map<std::pair<int, bool>, cudaGraphExec_t> graphs_;
  std::map<std::pair<int, bool>, size_t> graph_nodes_;
};

}  // namespace gfused
