// Torch-free end-to-end Gaussifier pipeline. See pipeline.cuh.

#include "pipeline.cuh"

#include <algorithm>
#include <cmath>
#include <limits>

namespace gfused {
namespace {

__global__ void k_chw_to_hwc(const float* __restrict__ chw, float* __restrict__ hwc, int P) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= P) return;
  hwc[i * 3 + 0] = chw[i];
  hwc[i * 3 + 1] = chw[P + i];
  hwc[i * 3 + 2] = chw[2 * P + i];
}

__device__ int reflect_index(int i, int size) {
  // Padding is at most 15 pixels per axis and the real extent is at least 16.
  return i < size ? i : 2 * (size - 1) - i;
}

__global__ void k_reflect_image(const float* real, float* padded, int H, int W, int NH, int NW) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= NH * NW) return;
  const int j = reflect_index(i / NW, H) * W + reflect_index(i % NW, W);
  for (int c = 0; c < 3; ++c) padded[3 * i + c] = real[3 * j + c];
}

// Reference Poisson rates include clamping and epsilon. The forward map has already
// reduced density over the logical image, without including the aligned workspace padding.
__global__ void k_real_rate_sum(const float* logits, int P, float* rate) {
  float r = 0.f;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < P; i += gridDim.x * blockDim.x)
    r += __expf(fminf(8.f, fmaxf(-20.f, logits[i]))) + 1e-8f;
  for (int o = 16; o > 0; o >>= 1) r += __shfl_down_sync(0xffffffffu, r, o);
  if ((threadIdx.x & 31) == 0) atomicAdd(rate, r);
}

// Point count (render_loop._resolve_n_render_points) and the count-override covariance shift
// (GaussifierSampler.sample(count=...)); resets the density sum for the next kernel.
__global__ void k_point_count_resolve(const float* rate_sum, PipelineCtrl* ctrl, SamplerCtrl* sc,
                                      int n_cap, int n_override) {
  const float rate = *rate_sum;
  const double r = static_cast<double>(rate);
  const int n_auto = (isfinite(r) && r >= 1.0) ? static_cast<int>(fmin(r, 2.0e9)) : 1;
  int n;
  float shift = 0.f;
  if (n_override > 0) {
    n = n_override;
    if (n > n_auto) shift = static_cast<float>(log(static_cast<double>(n_auto) / n));
  } else {
    n = n_auto;
  }
  n = max(1, min(n, n_cap));
  ctrl->rate_sum = rate;
  ctrl->n_auto = n_auto;
  ctrl->diag_shift = shift;
  ctrl->density_sum = 0.f;
  ctrl->io_mismatch = 0;
  sc->n = n;
}

__global__ void k_density_sum(const float* __restrict__ d, int P, PipelineCtrl* ctrl) {
  __shared__ float s[32];
  float v = 0.f;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < P; i += gridDim.x * blockDim.x) v += d[i];
  for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
  if ((threadIdx.x & 31) == 0) s[threadIdx.x >> 5] = v;
  __syncthreads();
  if (threadIdx.x == 0) {
    float t = 0.f;
    for (int w = 0; w < static_cast<int>(blockDim.x >> 5); ++w) t += s[w];
    atomicAdd(&ctrl->density_sum, t);
  }
}

// Per-pixel prologue after the density sum, one pass: the count-override covariance shift of
// the log-cov map (in place; GaussifierSampler.sample) and density_for_sampling =
// density * n / sum(density) (render_loop._correction_render_loss).
__global__ void k_pixel_prep(const float* __restrict__ d, float* log_cov, const PipelineCtrl* ctrl,
                             const SamplerCtrl* sc, float* density_s, int P, int shift) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= P) return;
  const float total = ctrl->density_sum;
  const float dv = d[i];
  density_s[i] = total > 1e-8f ? dv * (static_cast<float>(sc->n) / total) : dv;
  if (shift) {
    const float sh = ctrl->diag_shift;
    log_cov[i] += sh;
    log_cov[2 * P + i] += sh;
  }
}

// Packed head input (contract C1), layout [2][H][W][8] halves (two 16-byte units per pixel,
// unit u of pixel i at packed + (u * P + i) * 8), each value __float2half of the FP32 value
// the head's gather path reads:
//   unit 0 (static):  image r, g, b, density, log_cov 0, 1, 2, rgb 0
//   unit 1 (dynamic): rendered r, g, b, count, rgb 1, rgb 2, 0, 0
// Written once per image (after the sampler, so it is fresh in L2 for the first render and
// head call), unit 1 with zeros in the four dynamic halves; the raster rewrites the whole of
// unit 1 for every state that feeds the head. Full 16-byte stores only: partial sectors cost
// an L2 read-modify-write on eviction.
__global__ void k_pack_static(const float* __restrict__ image_chw, const float* __restrict__ d,
                              const float* __restrict__ log_cov, const float* __restrict__ rgb,
                              __half* packed, int P) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= P) return;
  auto h2 = [](float a, float b) {
    const __half2 h = __halves2half2(__float2half(a), __float2half(b));
    return *reinterpret_cast<const unsigned*>(&h);
  };
  uint4 lo;
  lo.x = h2(image_chw[i], image_chw[P + i]);
  lo.y = h2(image_chw[2 * P + i], d[i]);
  lo.z = h2(log_cov[i], log_cov[P + i]);
  lo.w = h2(log_cov[2 * P + i], rgb[i]);
  uint4* q = reinterpret_cast<uint4*>(packed);
  q[i] = lo;
  q[P + i] = make_uint4(0u, 0u, h2(rgb[P + i], rgb[2 * P + i]), 0u);
}

// check_head_io: packed vs __float2half of the FP32 sources (what the gather path reads).
__global__ void k_check_packed(const __half* __restrict__ packed, const float* __restrict__ img_hwc,
                               const float* __restrict__ density, const float* __restrict__ log_cov,
                               const float* __restrict__ rgb, const float* __restrict__ rendered,
                               const float* __restrict__ count, int P, PipelineCtrl* ctrl) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= P) return;
  float ref[16] = {img_hwc[3 * i], img_hwc[3 * i + 1], img_hwc[3 * i + 2], density[i],
                   log_cov[i], log_cov[P + i], log_cov[2 * P + i], rgb[i],
                   rendered[i], rendered[P + i], rendered[2 * P + i], count[i],
                   rgb[P + i], rgb[2 * P + i], 0.f, 0.f};
  int bad = 0;
  for (int c = 0; c < 16; ++c) {
    const __half want = __float2half(ref[c]);
    const __half got = packed[(static_cast<size_t>(c / 8) * P + i) * 8 + c % 8];
    bad += *reinterpret_cast<const unsigned short*>(&want) !=
           *reinterpret_cast<const unsigned short*>(&got);
  }
  if (bad) atomicAdd(&ctrl->io_mismatch, bad);
}

// Adaptive K: records the squared error of the state just rendered, then decides whether to
// render another one; in graph mode it also drives the WHILE node's condition.
__global__ void k_adaptive_reset(PipelineCtrl* c) {
  c->k_done = 0;
  c->stop = 0;
  c->sq_cur = 0.0;
}

__global__ void k_adaptive_decide(PipelineCtrl* c, double* sq_err, double inv_count, int k_max,
                                  float target, float min_gain, unsigned long long handle,
                                  int set_handle) {
  const int k = c->k_done;
  sq_err[k] = c->sq_cur;
  c->sq_cur = 0.0;
  const int done = k + 1;
  c->k_done = done;
  auto psnr = [&](double sq) { return -10.0 * log10(fmax(sq * inv_count, 1e-10)); };
  const double p = psnr(sq_err[k]);
  bool stop = done >= k_max;
  if (target > 0.f && p >= target) stop = true;
  if (min_gain > 0.f && done >= 2 && p - psnr(sq_err[k - 1]) < min_gain) stop = true;
  c->stop = stop;
#if CUDART_VERSION >= 12040
  if (set_handle)
    cudaGraphSetConditional(static_cast<cudaGraphConditionalHandle>(handle), stop ? 0 : 1);
#endif
}

}  // namespace

struct FusedPipeline::Model {
  ForwardMapWeights fm{};
  CorrectionHeadWeights head{};

  ~Model() {
    free_forward_map_weights(&fm);
    free_correction_head_weights(&head);
  }
};

FusedPipeline::~FusedPipeline() { release(); }

void FusedPipeline::release() {
  for (auto& g : graphs_) cudaGraphExecDestroy(g.second);
  graphs_.clear();
  graph_nodes_.clear();
  if (arena_) cudaFree(arena_);
  arena_ = nullptr;
  if (fm_ws_.device_blob) free_forward_map_workspace(&fm_ws_);
  if (head_ws_.impl || head_ws_.H) free_correction_head_workspace(&head_ws_);
  H_ = W_ = net_h_ = net_w_ = 0;
  schedule_ = NetSchedule::Aligned;
  arena_bytes_ = 0;
  video_started_ = last_frame_warm_ = false;
}

cudaError_t FusedPipeline::load(const PipelineOptions& opts, std::string* error) {
  if (opts.video) {
    const auto& t = opts.video_tracks;
    const float blend = opts.warm_attribute_blend;
    if (!(blend >= 0.f && blend <= 1.f && t.count_ema >= 0.0 && t.count_ema <= 1.0 &&
          t.churn_fraction >= 0.0 && t.churn_fraction <= 1.0 && t.max_count_rate >= 0.0 &&
          t.max_count_rate <= 1.0 && t.tau_cold > 0.f && t.tau_cold < 1.f &&
          t.tau_hot > 0.f && std::isfinite(t.tau_hot) && t.patience >= 1)) {
      if (error) *error = "invalid video count, repair, or attribute-blend options";
      return cudaErrorInvalidValue;
    }
  }
  auto model = std::make_shared<Model>();
  GF_TRY(load_forward_map_weights(opts.forward_map_path, &model->fm, error));
  GF_TRY(load_correction_head_weights(opts.head_path, &model->head, error));
  release();
  opts_ = opts;
  model_ = std::move(model);
  return cudaSuccess;
}

std::unique_ptr<FusedPipeline> FusedPipeline::make_workspace() const {
  auto p = std::make_unique<FusedPipeline>();
  p->model_ = model_;
  p->opts_ = opts_;
  return p;
}

std::unique_ptr<FusedPipeline> FusedPipeline::make_video_workspace(
    double count_ema, float warm_attribute_blend) const {
  auto p = make_workspace();
  p->opts_.video = true;
  p->opts_.video_tracks.count_ema = count_ema;
  p->opts_.warm_attribute_blend = warm_attribute_blend;
  return p;
}

void FusedPipeline::plan(Arena& a) {
  const size_t P = static_cast<size_t>(H_) * W_;
  ctrl_ = a.take<PipelineCtrl>(1);
  in_chw_ = a.take<float>(3 * P);
  in_hwc_ = a.take<float>(3 * P);
  density_ = a.take<float>(P);
  external_cov_ = opts_.external_maps && opts_.n_override > 0 && opts_.count_rescale
                      ? a.take<float>(3 * P) : nullptr;
  const size_t NP = static_cast<size_t>(net_h_) * net_w_;
  if (schedule_ == NetSchedule::ReflectPad && !opts_.external_maps) {
    net_image_ = a.take<float>(3 * NP);
    net_cov_ = a.take<float>(3 * NP);
    net_rgb_ = a.take<float>(3 * NP);
    real_logits_ = a.take<float>(P);
  }
  log_cov_ = a.take<float>(3 * P);
  rgb_map_ = a.take<float>(3 * P);
  xy_ = a.take<float2>(n_cap_);
  cov_ = a.take<float>(3 * static_cast<size_t>(n_cap_));
  rgb_ = a.take<float>(3 * static_cast<size_t>(n_cap_));
  if (opts_.keep_initial) {
    initial_xy_ = a.take<float2>(n_cap_);
    initial_cov_ = a.take<float>(3 * static_cast<size_t>(n_cap_));
    initial_rgb_ = a.take<float>(3 * static_cast<size_t>(n_cap_));
  }
  weights_ = a.take<float>(n_cap_);
  colour_ = a.take<float>(3 * static_cast<size_t>(n_cap_));
  render_ = a.take<float>(3 * P);
  count_map_ = a.take<float>(P);
  corr_ = a.take<float>(8 * P);
  sq_err_ = a.take<double>(opts_.max_k);
  packed16_ = a.take<__half>(16 * P);
  // rate_sum lives in Ctrl
  SamplerConfig sc;
  sc.H = H_;
  sc.W = W_;
  sc.n_cap = n_cap_;
  sc.seed = opts_.seed;
  sc.merge_rounds = opts_.merge_rounds;
  sc.knn_k = opts_.knn_k;
  sc.lloyd_iters = opts_.lloyd_iters;
  plan_sampler(sc, a, &sb_);
  if (opts_.video) {
    plan_video_tracks(sc, a, &tracks_);
    attribute_history_ = a.take<float>(6 * static_cast<size_t>(n_cap_));
  }
  plan_render(H_, W_, n_cap_, a, &rb_);
}

bool FusedPipeline::supports_size(int H, int W) {
  // The kernels use int offsets (up to 32 values per full-resolution pixel), and
  // a two-dimensional CUDA grid for the renderer and sampler.
  if (H < 16 || W < 16 || H > 16 * 65535 || W > 16 * 65535) return false;
  const long long NH = (H + 15) / 16 * 16, NW = (W + 15) / 16 * 16;
  return NH * NW <= std::numeric_limits<int>::max() / 32;
}

cudaError_t FusedPipeline::prepare(int H, int W, int max_points) {
  if (!model_ || !supports_size(H, W) || max_points < 0 || max_points > opts_.render_n_max ||
      opts_.n_override < 0 || opts_.n_override > opts_.render_n_max)
    return cudaErrorInvalidValue;
  release();
  H_ = H;
  W_ = W;
  net_h_ = (H + 15) / 16 * 16;
  net_w_ = (W + 15) / 16 * 16;
  if (H != net_h_ || W != net_w_)
    schedule_ = opts_.size_mode == SizeMode::Exact ? NetSchedule::Exact : NetSchedule::ReflectPad;
  const long long P = static_cast<long long>(H) * W;
  long long cap = std::min<long long>(
      opts_.render_n_max,
      std::max<long long>(1, static_cast<long long>(opts_.points_per_pixel_max * P)));
  if (opts_.n_override > 0) cap = std::max<long long>(cap, opts_.n_override);
  if (max_points > 0) cap = max_points;
  n_cap_ = static_cast<int>(cap);

  Arena measure;
  plan(measure);
  arena_bytes_ = measure.used();
  cudaError_t e = cudaMalloc(&arena_, arena_bytes_);
  if (!e) {
    Arena bind(arena_);
    plan(bind);
    if (!opts_.external_maps) e = alloc_forward_map_workspace(net_h_, net_w_, &fm_ws_);
    if (!e) e = alloc_correction_head_workspace(net_h_, net_w_, &head_ws_);
    if (!e) e = init_render(rb_, nullptr);
    if (!e && opts_.video) {
      // Video bypasses the EMV setup and fixed K bypasses the adaptive counters.
      // read_stats copies both structs in full, including the inactive fields.
      e = cudaMemset(sb_.ctrl, 0, sizeof(SamplerCtrl));
      if (!e) e = cudaMemset(ctrl_, 0, sizeof(PipelineCtrl));
      if (!e) e = init_video_tracks(sb_.cfg, tracks_);
    }
    if (!e) e = cudaStreamSynchronize(nullptr);
  }
  if (e) release();
  return e;
}

cudaError_t FusedPipeline::enqueue_forward_map(cudaStream_t s) {
  const int P = H_ * W_;
  k_chw_to_hwc<<<grid_for(P), 256, 0, s>>>(in_chw_, in_hwc_, P);
  if (opts_.external_maps) {
    if (external_cov_)
      GF_TRY(cudaMemcpyAsync(log_cov_, external_cov_, 3 * P * sizeof(float),
                             cudaMemcpyDeviceToDevice, s));
    return cudaGetLastError();
  }
  const ForwardMapOutputs fo{density_, log_cov_, rgb_map_, &ctrl_->rate_sum};
  const float* logits = fm_ws_.logits;
  switch (schedule_) {
    case NetSchedule::Aligned:
      return run_forward_map(model_->fm, in_hwc_, fm_ws_, fo, s);
    case NetSchedule::Exact:
      GF_TRY(run_forward_map(model_->fm, in_hwc_, fm_ws_, fo, s, H_, W_));
      break;
    case NetSchedule::ReflectPad: {
      k_reflect_image<<<grid_for(net_h_ * net_w_), 256, 0, s>>>(
          in_hwc_, net_image_, H_, W_, net_h_, net_w_);
      const ForwardMapOutputs net_out{nullptr, net_cov_, net_rgb_, nullptr};
      GF_TRY(run_forward_map(model_->fm, net_image_, fm_ws_, net_out, s));
      GF_TRY(crop_forward_map(fm_ws_, net_cov_, net_rgb_, fo, real_logits_, H_, W_, s));
      logits = real_logits_;
      break;
    }
  }
  GF_TRY(cudaMemsetAsync(fo.rate_sum, 0, sizeof(float), s));
  k_real_rate_sum<<<128, 256, 0, s>>>(logits, P, fo.rate_sum);
  return cudaGetLastError();
}

cudaError_t FusedPipeline::run(int K, cudaStream_t s) {
  if (!arena_ || K < 1 || K > opts_.max_k) return cudaErrorInvalidValue;
  const bool warm = opts_.video && video_started_;
  GF_TRY(run_frame(K, warm, s));
  last_frame_warm_ = warm;
  video_started_ = opts_.video;
  return cudaSuccess;
}

// Forward map, point count, placement (EMV, or the video tracks) and the initial per-point
// attributes from the forward-map maps; warm video frames blend in the tracked history.
cudaError_t FusedPipeline::enqueue_initial_state(bool warm, bool pack, cudaStream_t s) {
  const int H = H_, W = W_;
  GF_TRY(enqueue_forward_map(s));
  GF_TRY(enqueue_pixel_prep(warm, s));
  if (opts_.video) {
    GF_TRY(sample_video_frame(sb_, tracks_, density_, !warm, xy_, weights_, s));
  } else {
    GF_TRY(run_sampler(sb_, density_, xy_, weights_, s));
  }
  if (pack) GF_TRY(enqueue_pack(s));  // head input, only if a head call follows
  const int* n = &sb_.ctrl->n_cur;
  GF_TRY(sample_point_features(n, xy_, n_cap_, log_cov_, 3, H, W, false, nullptr, cov_, s));
  GF_TRY(sample_point_features(n, xy_, n_cap_, rgb_map_, 3, H, W, true, nullptr, rgb_, s));
  if (warm && opts_.warm_attribute_blend > 0.f)
    GF_TRY(blend_point_history(n, n_cap_, tracks_.history_slots, opts_.warm_attribute_blend,
                                attribute_history_, cov_, rgb_, s));
  // the state the correction loop starts from (after the warm blend in video mode)
  if (opts_.keep_initial) GF_TRY(save_initial(s));
  return cudaSuccess;
}

cudaError_t FusedPipeline::run_frame(int K, bool warm, cudaStream_t s) {
  GF_TRY(cudaMemsetAsync(sq_err_, 0, sizeof(double) * opts_.max_k, s));
  const int* n = &sb_.ctrl->n_cur;
  if (adaptive()) {
    GF_TRY(run_adaptive(K, warm, s));
  } else {
    GF_TRY(enqueue_initial_state(warm, K > 1, s));
    // K rendered states, K-1 head updates
    for (int k = 0; k < K; ++k) {
      // Non-final states also produce the head's hard point-count map in the projection pass
      // and the packed dynamic head-input channels in the raster.
      const bool head_next = k + 1 < K;
      GF_TRY(render_points(rb_, n, xy_, cov_, rgb_, weights_, render_, in_chw_, sq_err_ + k, s,
                           head_next ? count_map_ : nullptr, head_next ? packed16_ : nullptr));
      if (head_next) GF_TRY(enqueue_head(s));
    }
  }
  GF_TRY(premultiply_colour(n, n_cap_, rgb_, weights_, colour_, s));
  if (opts_.video) {
    if (opts_.warm_attribute_blend > 0.f)
      GF_TRY(store_point_history(n, n_cap_, tracks_.slots, cov_, rgb_, attribute_history_, s));
    GF_TRY(repair_video_tracks(sb_, tracks_, opts_.video_tracks, density_, xy_, s));
  }
  k_fixed_ = K;
  return cudaGetLastError();
}

cudaError_t FusedPipeline::enqueue_step(cudaStream_t s) {
  const int* n = &sb_.ctrl->n_cur;
  GF_TRY(enqueue_head(s));
  return render_points(rb_, n, xy_, cov_, rgb_, weights_, render_, in_chw_, &ctrl_->sq_cur, s,
                       count_map_, packed16_);
}

cudaError_t FusedPipeline::enqueue_pixel_prep(bool warm, cudaStream_t s) {
  const int P = H_ * W_;
  k_point_count_resolve<<<1, 1, 0, s>>>(&ctrl_->rate_sum, ctrl_, sb_.ctrl, n_cap_,
                                        opts_.n_override);
  if (opts_.video) GF_TRY(begin_video_frame(sb_, tracks_, opts_.video_tracks, !warm, s));
  k_density_sum<<<128, 256, 0, s>>>(density_, P, ctrl_);
  k_pixel_prep<<<grid_for(P), 256, 0, s>>>(density_, log_cov_, ctrl_, sb_.ctrl, sb_.density_s,
                                           P, opts_.n_override > 0 && opts_.count_rescale);
  return cudaGetLastError();
}

cudaError_t FusedPipeline::save_initial(cudaStream_t s) {
  const size_t bytes = static_cast<size_t>(n_cap_) * sizeof(float);
  GF_TRY(cudaMemcpyAsync(initial_xy_, xy_, 2 * bytes, cudaMemcpyDeviceToDevice, s));
  GF_TRY(cudaMemcpyAsync(initial_cov_, cov_, 3 * bytes, cudaMemcpyDeviceToDevice, s));
  return cudaMemcpyAsync(initial_rgb_, rgb_, 3 * bytes, cudaMemcpyDeviceToDevice, s);
}

cudaError_t FusedPipeline::enqueue_pack(cudaStream_t s) {
  const int P = H_ * W_;
  k_pack_static<<<grid_for(P), 256, 0, s>>>(in_chw_, density_, log_cov_, rgb_map_, packed16_, P);
  return cudaGetLastError();
}

// Head update for the state just rendered: correction head -> per-point correction apply.
cudaError_t FusedPipeline::enqueue_head(cudaStream_t s) {
  const int H = H_, W = W_, P = H * W;
  const int* n = &sb_.ctrl->n_cur;
  const float xy_step = opts_.xy_step_pixels / static_cast<float>(std::max(H, W));
  CorrectionHeadInputs hin{in_hwc_, density_, log_cov_, rgb_map_, render_, count_map_};
  hin.packed16 = packed16_;
  hin.dynamic_in_packed = true;  // the raster wrote this state's unit 1
  if (opts_.check_head_io)
    k_check_packed<<<grid_for(P), 256, 0, s>>>(packed16_, in_hwc_, density_, log_cov_, rgb_map_,
                                               render_, count_map_, P, ctrl_);
  // The head runs at the real extent in every mode (zero padding at the real border, as in
  // the reference); see run_correction_head.
  GF_TRY(run_correction_head(model_->head, hin, head_ws_, corr_, s, H, W));
  GF_TRY(apply_correction(n, n_cap_, corr_, H, W, xy_step, xy_, cov_, rgb_, s));
  return cudaGetLastError();
}

cudaError_t FusedPipeline::enqueue_decide(int K, cudaStream_t s, bool set_handle) {
  k_adaptive_decide<<<1, 1, 0, s>>>(ctrl_, sq_err_, 1.0 / (3.0 * H_ * W_), K, opts_.psnr_target,
                                    opts_.min_gain, cond_handle_, set_handle ? 1 : 0);
  return cudaGetLastError();
}

cudaError_t FusedPipeline::run_adaptive(int K, bool warm, cudaStream_t s) {
  cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
  GF_TRY(cudaStreamIsCapturing(s, &status));
  k_adaptive_reset<<<1, 1, 0, s>>>(ctrl_);
  GF_TRY(enqueue_initial_state(warm, true, s));
  const int* n = &sb_.ctrl->n_cur;
  GF_TRY(render_points(rb_, n, xy_, cov_, rgb_, weights_, render_, in_chw_, &ctrl_->sq_cur, s,
                       count_map_, packed16_));
  if (status != cudaStreamCaptureStatusActive) {
    // Eager: host-driven loop, one decision readback per state.
    for (;;) {
      GF_TRY(enqueue_decide(K, s, false));
      int stop = 0;
      GF_TRY(cudaMemcpyAsync(&stop, &ctrl_->stop, sizeof(int), cudaMemcpyDeviceToHost, s));
      GF_TRY(cudaStreamSynchronize(s));
      if (stop) break;
      GF_TRY(enqueue_step(s));
    }
    return cudaSuccess;
  }
#if CUDART_VERSION >= 12040
  // Graph: decide(state 1) -> WHILE(cond) { head update; render; decide }.
  cudaGraph_t graph = nullptr;
  const cudaGraphNode_t* deps = nullptr;
  size_t n_deps = 0;
#if CUDART_VERSION >= 13000
  GF_TRY(cudaStreamGetCaptureInfo(s, &status, nullptr, &graph, &deps, nullptr, &n_deps));
#else
  GF_TRY(cudaStreamGetCaptureInfo(s, &status, nullptr, &graph, &deps, &n_deps));
#endif
  cudaGraphConditionalHandle handle;
  GF_TRY(cudaGraphConditionalHandleCreate(&handle, graph, 0, 0));
  cond_handle_ = static_cast<unsigned long long>(handle);
  GF_TRY(enqueue_decide(K, s, true));
#if CUDART_VERSION >= 13000
  GF_TRY(cudaStreamGetCaptureInfo(s, &status, nullptr, &graph, &deps, nullptr, &n_deps));
#else
  GF_TRY(cudaStreamGetCaptureInfo(s, &status, nullptr, &graph, &deps, &n_deps));
#endif
  cudaGraphNodeParams params = {};
  params.type = cudaGraphNodeTypeConditional;
  params.conditional.handle = handle;
  params.conditional.type = cudaGraphCondTypeWhile;
  params.conditional.size = 1;
  cudaGraphNode_t node = nullptr;
#if CUDART_VERSION >= 13000
  GF_TRY(cudaGraphAddNode(&node, graph, deps, nullptr, n_deps, &params));
  GF_TRY(cudaStreamUpdateCaptureDependencies(s, &node, nullptr, 1,
                                             cudaStreamSetCaptureDependencies));
#else
  GF_TRY(cudaGraphAddNode(&node, graph, deps, n_deps, &params));
  GF_TRY(cudaStreamUpdateCaptureDependencies(s, &node, 1, cudaStreamSetCaptureDependencies));
#endif
  cudaGraph_t body = params.conditional.phGraph_out[0];
  cudaStream_t bs = nullptr;
  GF_TRY(cudaStreamCreateWithFlags(&bs, cudaStreamNonBlocking));
  cudaError_t e = cudaStreamBeginCaptureToGraph(bs, body, nullptr, nullptr, 0,
                                                cudaStreamCaptureModeThreadLocal);
  if (!e) e = enqueue_step(bs);
  if (!e) e = enqueue_decide(K, bs, true);
  cudaGraph_t captured = nullptr;
  const cudaError_t end = cudaStreamEndCapture(bs, &captured);
  cudaStreamDestroy(bs);
  GF_TRY(e);
  GF_TRY(end);
  return cudaSuccess;
#else
  return cudaErrorNotSupported;
#endif
}

cudaError_t FusedPipeline::capture_graph(int K, bool warm, cudaStream_t s) {
  const auto key = std::make_pair(K, warm);
  if (!graphs_.count(key)) {
    cudaGraph_t g = nullptr;
    GF_TRY(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    const int previous_k = k_fixed_;
    const cudaError_t run_err = run_frame(K, warm, s);
    k_fixed_ = previous_k;
    const cudaError_t end_err = cudaStreamEndCapture(s, &g);
    if (run_err) {
      if (g) cudaGraphDestroy(g);
      return run_err;
    }
    GF_TRY(end_err);
    size_t nodes = 0;
    cudaGraphGetNodes(g, nullptr, &nodes);
    cudaGraphExec_t exec = nullptr;
    const cudaError_t inst = cudaGraphInstantiate(&exec, g, 0);
    cudaGraphDestroy(g);
    GF_TRY(inst);
    graphs_.emplace(key, exec);
    graph_nodes_[key] = nodes;
  }
  return cudaSuccess;
}

cudaError_t FusedPipeline::capture_graph(int K, cudaStream_t s) {
  if (!arena_ || K < 1 || K > opts_.max_k) return cudaErrorInvalidValue;
  return capture_graph(K, opts_.video && video_started_, s);
}

cudaError_t FusedPipeline::prepare_video_graphs(int K, cudaStream_t s) {
  if (!opts_.video || !arena_ || K < 1 || K > opts_.max_k) return cudaErrorInvalidValue;
  GF_TRY(capture_graph(K, false, s));
  return capture_graph(K, true, s);
}

cudaError_t FusedPipeline::run_graph(int K, cudaStream_t s) {
  if (!arena_ || K < 1 || K > opts_.max_k) return cudaErrorInvalidValue;
  const bool warm = opts_.video && video_started_;
  GF_TRY(capture_graph(K, warm, s));
  GF_TRY(cudaGraphLaunch(graphs_.at({K, warm}), s));
  if (!adaptive()) k_fixed_ = K;
  last_frame_warm_ = warm;
  video_started_ = opts_.video;
  return cudaSuccess;
}

int FusedPipeline::graph_nodes(int K) const {
  auto it = graph_nodes_.find({K, last_frame_warm_});
  return it == graph_nodes_.end() ? 0 : static_cast<int>(it->second);
}

void FusedPipeline::set_sampling(int64_t seed, int count) {
  if (opts_.seed == seed && opts_.n_override == count) return;
  release();
  opts_.seed = seed;
  opts_.n_override = count;
}

void FusedPipeline::set_size_mode(SizeMode mode) {
  if (opts_.size_mode == mode) return;
  if (schedule_ != NetSchedule::Aligned) release();
  opts_.size_mode = mode;
}

PipelineOutputs FusedPipeline::outputs() const {
  return {&sb_.ctrl->n_cur,
          xy_,
          cov_,
          rgb_,
          colour_,
          weights_,
          render_,
          density_,
          initial_xy_,
          initial_cov_,
          initial_rgb_,
          sq_err_,
          opts_.video ? tracks_.ids : nullptr,
          opts_.video ? tracks_.slots : nullptr};
}

cudaError_t FusedPipeline::read_stats(PipelineStats* st, cudaStream_t s) const {
  SamplerCtrl sc{};
  PipelineCtrl c{};
  GF_TRY(cudaMemcpyAsync(&sc, sb_.ctrl, sizeof(sc), cudaMemcpyDeviceToHost, s));
  GF_TRY(cudaMemcpyAsync(&c, ctrl_, sizeof(c), cudaMemcpyDeviceToHost, s));
  if (opts_.video)
    GF_TRY(cudaMemcpyAsync(&st->tracks, &tracks_.ctrl->stats, sizeof(st->tracks),
                           cudaMemcpyDeviceToHost, s));
  GF_TRY(cudaStreamSynchronize(s));
  st->n = sc.n_cur;
  st->n_requested = sc.n;
  st->n_auto = c.n_auto;
  st->n_init = sc.n_init;
  st->rate_sum = c.rate_sum;
  st->ed_tile_size = sc.tile_size;
  st->matched = sc.stat_matched;
  st->forced_adj = sc.stat_forced_adj;
  st->forced_nn = sc.stat_forced_nn;
  st->count_error = sc.stat_count_error;
  st->k_used = adaptive() ? c.k_done : k_fixed_;
  st->io_mismatch = opts_.check_head_io ? c.io_mismatch : -1;
  return cudaSuccess;
}

}  // namespace gfused
