// Synchronous host-array interface for the Python binding.
#include "python_api.h"

#include "pipeline.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>

struct GfusedSession {
  explicit GfusedSession(int device, std::unique_ptr<gfused::FusedPipeline> pipeline)
      : device(device), pipeline(std::move(pipeline)) {}
  ~GfusedSession() {
    if (stream) cudaStreamDestroy(stream);
  }

  int device;
  cudaStream_t stream = nullptr;
  std::unique_ptr<gfused::FusedPipeline> pipeline;
  int count = 0;
  // A nonzero state count identifies a video session.
  int video_states = 0;
  uint64_t frame = 0;
  bool video_failed = false;
};

namespace {

thread_local char last_error[1024]{};
// Arena preparation uses device synchronization, which cannot overlap graph capture.
std::mutex runtime_mutex;

void check(cudaError_t result) {
  if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}

struct DeviceGuard {
  explicit DeviceGuard(int device) {
    check(cudaGetDevice(&previous));
    check(cudaSetDevice(device));
  }
  ~DeviceGuard() { cudaSetDevice(previous); }
  int previous;
};

gfused::SizeMode parse_size_mode(int mode) {
  if (mode != static_cast<int>(gfused::SizeMode::Pad) &&
      mode != static_cast<int>(gfused::SizeMode::Exact))
    throw std::invalid_argument("invalid fused size mode");
  return static_cast<gfused::SizeMode>(mode);
}

void validate_run(int height, int width, int states, int count) {
  if (!gfused::FusedPipeline::supports_size(height, width) ||
      states < 1 || states > GFUSED_MAX_STATES || count < 0 || count > 1000000)
    throw std::invalid_argument("invalid fused image size, state count or point count");
}

gfused::PipelineStats run_frame(GfusedSession& session, const float* input, int states) {
  auto& pipe = *session.pipeline;
  const auto stream = session.stream;
  const size_t bytes = 3 * static_cast<size_t>(pipe.H()) * pipe.W() * sizeof(float);
  check(cudaMemcpyAsync(pipe.input_chw(), input, bytes, cudaMemcpyHostToDevice, stream));
  const auto run = pipe.run_graph(states, stream);
  const auto sync = cudaStreamSynchronize(stream);
  check(run);
  check(sync);
  gfused::PipelineStats stats;
  check(pipe.read_stats(&stats, stream));
  if (stats.count_error || stats.n < 1 || stats.n > pipe.n_cap())
    throw std::runtime_error("fused sampler failed its point count contract");
  return stats;
}

template <typename T>
void download(T* dest, const void* src, size_t elements, cudaStream_t stream) {
  auto status = cudaMemcpyAsync(dest, src, elements * sizeof(T), cudaMemcpyDeviceToHost, stream);
  if (status != cudaSuccess) {
    cudaStreamSynchronize(stream);
    check(status);
  }
}

template <typename Fn>
int call(Fn fn) noexcept {
  try {
    std::lock_guard<std::mutex> lock(runtime_mutex);
    return fn();
  } catch (const std::exception& e) {
    std::snprintf(last_error, sizeof(last_error), "%s", e.what());
    return -1;
  }
}

}  // namespace

#define GFUSED_API extern "C" __attribute__((visibility("default")))

GFUSED_API const char* gfused_error() noexcept { return last_error; }

GFUSED_API int gfused_create(const char* fm, const char* head, int device,
                             GfusedSession** output) noexcept {
  return call([&] {
    if (!fm || !head || !output)
      throw std::invalid_argument("expected fused weight paths and output handle");
    DeviceGuard guard(device);
    auto session = std::make_unique<GfusedSession>(device, std::make_unique<gfused::FusedPipeline>());
    check(cudaStreamCreateWithFlags(&session->stream, cudaStreamNonBlocking));
    gfused::PipelineOptions options;
    options.forward_map_path = fm;
    options.head_path = head;
    options.keep_initial = true;
    std::string error;
    if (auto status = session->pipeline->load(options, &error); status != cudaSuccess) {
      if (!error.empty()) throw std::runtime_error(error);
      check(status);
    }
    *output = session.release();
    return 0;
  });
}

GFUSED_API void gfused_destroy(GfusedSession* session) noexcept {
  if (!session) return;
  call([&] {
    DeviceGuard guard(session->device);
    delete session;
    return 0;
  });
}

GFUSED_API int gfused_run_with_size_mode(GfusedSession* session, const float* input, int height,
                                         int width, int states, int64_t seed, int count,
                                         int size_mode) noexcept {
  return call([&] {
    if (!session || session->video_states || !input)
      throw std::invalid_argument("expected an image session and input");
    DeviceGuard guard(session->device);
    auto& pipe = *session->pipeline;
    session->count = 0;
    validate_run(height, width, states, count);
    pipe.set_size_mode(parse_size_mode(size_mode));
    pipe.set_sampling(seed, count);
    if (pipe.H() != height || pipe.W() != width) check(pipe.prepare(height, width));
    return session->count = run_frame(*session, input, states).n;
  });
}

GFUSED_API int gfused_run(GfusedSession* session, const float* input, int height, int width,
                          int states, int64_t seed, int count) noexcept {
  return gfused_run_with_size_mode(session, input, height, width, states, seed, count,
                                   static_cast<int>(gfused::SizeMode::Exact));
}

GFUSED_API int gfused_read(GfusedSession* session, float* density, float* render, float* initial_xy,
                           float* initial_cov, float* initial_rgb, float* xy, float* cov,
                           float* rgb, float* colour, float* weights) noexcept {
  return call([&] {
    if (!session) throw std::invalid_argument("expected a fused session");
    DeviceGuard guard(session->device);
    auto& pipe = *session->pipeline;
    const size_t n = session->count;
    if (!n) throw std::logic_error("no fused result available");
    const size_t pixels = static_cast<size_t>(pipe.H()) * pipe.W();
    const auto out = pipe.outputs();
    const auto stream = session->stream;
    download(density, out.density, pixels, stream);
    download(render, out.render, 3 * pixels, stream);
    download(initial_xy, out.initial_xy, 2 * n, stream);
    download(initial_cov, out.initial_log_cov, 3 * n, stream);
    download(initial_rgb, out.initial_rgb, 3 * n, stream);
    download(xy, out.xy, 2 * n, stream);
    download(cov, out.log_cov, 3 * n, stream);
    download(rgb, out.rgb, 3 * n, stream);
    download(colour, out.colour, 3 * n, stream);
    download(weights, out.weights, n, stream);
    check(cudaStreamSynchronize(stream));
    return 0;
  });
}

GFUSED_API int gfused_video_create(GfusedSession* source, int height, int width, int states,
                                   int64_t seed, int count, int size_mode, double count_ema,
                                   double warm_attribute_blend, GfusedSession** output) noexcept {
  return call([&] {
    if (!source || source->video_states || !output)
      throw std::invalid_argument("expected an image session and video output handle");
    validate_run(height, width, states, count);
    const auto mode = parse_size_mode(size_mode);
    if (!(count_ema >= 0 && count_ema <= 1 &&
          warm_attribute_blend >= 0 && warm_attribute_blend <= 1))
      throw std::invalid_argument("video count_ema and warm_attribute_blend must be in [0, 1]");
    DeviceGuard guard(source->device);
    auto session = std::make_unique<GfusedSession>(
        source->device, source->pipeline->make_video_workspace(count_ema, warm_attribute_blend));
    auto& pipe = *session->pipeline;
    pipe.set_size_mode(mode);
    pipe.set_sampling(seed, count);
    check(cudaStreamCreateWithFlags(&session->stream, cudaStreamNonBlocking));
    check(pipe.prepare(height, width));
    check(pipe.prepare_video_graphs(states, session->stream));
    session->video_states = states;
    *output = session.release();
    return 0;
  });
}

GFUSED_API int gfused_video_push(GfusedSession* session, const float* input, int height,
                                 int width, GfusedVideoStats* result) noexcept {
  return call([&] {
    if (!session || !session->video_states || !input || !result)
      throw std::invalid_argument("expected a video session, frame and stats output");
    auto& pipe = *session->pipeline;
    if (height != pipe.H() || width != pipe.W())
      throw std::invalid_argument("all video frames must have the session dimensions");
    if (session->video_failed) throw std::logic_error("fused video session failed");
    DeviceGuard guard(session->device);
    const int previous_count = session->count;
    session->count = 0;
    session->video_failed = true;
    const auto stats = run_frame(*session, input, session->video_states);
    const auto& tracks = stats.tracks;
    GfusedVideoStats frame{};
    frame.frame = session->frame;
    frame.n_requested = stats.n_requested;
    frame.n_target = tracks.target;
    frame.n_target_limited = tracks.limited_target;
    frame.survivors = tracks.survivors;
    frame.kills_for_next = tracks.killed;
    frame.births_for_next = tracks.born;
    frame.shortfall_for_next = tracks.shortfall;
    frame.k_used = stats.k_used;
    frame.retained_fraction = previous_count ? static_cast<double>(tracks.survivors) / previous_count
                                            : 0.0;
    frame.rate_sum = stats.rate_sum;
    double sq[GFUSED_MAX_STATES];
    download(sq, pipe.outputs().sq_err, frame.k_used, session->stream);
    check(cudaStreamSynchronize(session->stream));
    for (int k = 0; k < frame.k_used; ++k) {
      const double mse = sq[k] / (3.0 * height * width);
      frame.psnr[k] = -10.0 * std::log10(std::max(mse, 1e-10));
    }
    *result = frame;
    ++session->frame;
    session->video_failed = false;
    return session->count = stats.n;
  });
}

GFUSED_API int gfused_video_read_tracks(GfusedSession* session, int64_t* ids,
                                        int32_t* slots) noexcept {
  return call([&] {
    if (!session || !session->video_states || !ids || !slots)
      throw std::invalid_argument("expected a video session and track output buffers");
    if (!session->count) throw std::logic_error("no fused video result available");
    DeviceGuard guard(session->device);
    const auto out = session->pipeline->outputs();
    const auto stream = session->stream;
    static_assert(sizeof(*out.slots) == sizeof(*slots));
    download(ids, out.ids, session->count, stream);
    download(slots, out.slots, session->count, stream);
    check(cudaStreamSynchronize(stream));
    return 0;
  });
}

GFUSED_API void gfused_video_destroy(GfusedSession* session) noexcept {
  gfused_destroy(session);
}
