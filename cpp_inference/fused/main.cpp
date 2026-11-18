// gaussifier_fused_infer: torch-free end-to-end Gaussifier inference (CUDA runtime only).
//
//   gaussifier_fused_infer --fm forward_map.gffm --head head.gfch [--K 7] [--seed 12345]
//       [--n-points N] [--repeat R] [--no-graph] [--gaussians OUT.npy]
//       input.npy output.npy
//   gaussifier_fused_infer --fm ... --head ... --manifest pairs.txt   (lines: <input.npy>
//       <render.npy> [<gaussians.npy> [<maps stem>]]; one model load, bounded workspace cache per
//       resolution and --inflight N streams; render "DUMMY" and gaussians "-" skip writing)
//   gaussifier_fused_infer --fm ... --head ... --video FRAMES --video-out DIR   (GPU video warm
//       start: one workspace, frames in order; Gaussians per frame plus .ids.npy / .slots.npy)
//
// Input: C-contiguous float32 NPY [1, 3, H, W] or [3, H, W] in [0, 1], H and W >= 16. Output: the final render as float32 [3, H, W]; Gaussians as
// <stem>.points.npy (N, 2), <stem>.cov.npy (N, 3) log-covariance, <stem>.rgb.npy (N, 3) latent
// colour, <stem>.color.npy (N, 3) weight-premultiplied colour (what the renderer consumes) and
// <stem>.weights.npy (N,). One JSON line per image goes to stdout (count, per-state PSNR,
// timings); logs go to stderr. See README.md for the remaining (validation / timing) flags.

#include <cuda_runtime.h>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include "pipeline.cuh"

namespace {

void check_cuda(cudaError_t e, const std::string& context) {
  if (e == cudaSuccess) return;
  const std::string detail = e == cudaErrorMemoryAllocation ? "out of memory" : cudaGetErrorString(e);
  throw std::runtime_error(context + ": " + detail + " (" + cudaGetErrorName(e) + ")");
}

#define CHECK_CUDA(expr) check_cuda((expr), #expr)

[[noreturn]] void die(const std::string& msg) { throw std::runtime_error(msg); }

bool ends_with(const std::string& s, const std::string& suf) {
  return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

// ---- NPY (float32, little-endian, C order only)
struct Npy {
  std::vector<long> shape;
  std::vector<float> data;
};

std::string_view trim_header(std::string_view text) {
  const auto first = text.find_first_not_of(" \t\r\n");
  if (first == std::string_view::npos) return {};
  return text.substr(first, text.find_last_not_of(" \t\r\n") - first + 1);
}

std::string_view npy_field(std::string_view header, const std::string& name) {
  size_t p = header.find("'" + name + "'");
  if (p == std::string_view::npos) p = header.find("\"" + name + "\"");
  if (p == std::string_view::npos) return {};
  const auto rest = trim_header(header.substr(p + name.size() + 2));
  if (rest.empty() || rest.front() != ':') return {};
  return trim_header(rest.substr(1));
}

bool npy_field_end(std::string_view suffix) {
  suffix = trim_header(suffix);
  return !suffix.empty() && (suffix.front() == ',' || suffix.front() == '}');
}

Npy load_npy(const std::string& path, const std::vector<long>& expected = {}) {
  if (!ends_with(path, ".npy")) die("input must be .npy: " + path);
  std::ifstream f(path, std::ios::binary);
  if (!f) die("cannot open " + path);
  char magic[6];
  if (!f.read(magic, 6) || std::memcmp(magic, "\x93NUMPY", 6) != 0)
    die("not an NPY file: " + path);
  unsigned char ver[2];
  if (!f.read(reinterpret_cast<char*>(ver), 2)) die("truncated NPY version: " + path);
  if (ver[0] < 1 || ver[0] > 3 || ver[1] != 0) die("unsupported NPY version: " + path);
  const int len_bytes = ver[0] == 1 ? 2 : 4;
  unsigned char b[4] = {};
  if (!f.read(reinterpret_cast<char*>(b), len_bytes)) die("truncated NPY header length: " + path);
  const uint32_t hlen = b[0] | (b[1] << 8) | (b[2] << 16) | (static_cast<uint32_t>(b[3]) << 24);
  if (hlen == 0 || hlen > 65536) die("invalid NPY header length: " + path);
  std::string header(hlen, '\0');
  if (!f.read(header.data(), hlen)) die("truncated NPY header: " + path);

  const auto dict = trim_header(header);
  if (dict.empty() || dict.front() != '{' || dict.back() != '}')
    die("invalid NPY header: " + path);
  const auto dtype = npy_field(dict, "descr");
  if (dtype.size() < 5 || (dtype.substr(0, 5) != "'<f4'" && dtype.substr(0, 5) != "\"<f4\"") ||
      !npy_field_end(dtype.substr(5)))
    die("NPY must be little-endian float32: " + path);
  const auto order = npy_field(dict, "fortran_order");
  if (order.substr(0, 5) != "False" || !npy_field_end(order.substr(5)))
    die("NPY must be C order: " + path);
  const auto dims = npy_field(dict, "shape");
  const auto end_dims = dims.find(')');
  if (dims.empty() || dims.front() != '(' || end_dims == std::string_view::npos ||
      !npy_field_end(dims.substr(end_dims + 1)))
    die("invalid NPY shape: " + path);
  std::stringstream ss(std::string(dims.substr(1, end_dims - 1)));
  std::string tok;
  Npy npy;
  while (std::getline(ss, tok, ',')) {
    const auto token = trim_header(tok);
    if (token.empty()) {
      if (ss.eof() && !npy.shape.empty()) break;  // whitespace after a trailing comma
      die("invalid NPY shape: " + path);
    }
    long dim = 0;
    const char* end = token.data() + token.size();
    const auto parsed = std::from_chars(token.data(), end, dim);
    if (parsed.ec != std::errc{} || parsed.ptr != end || dim < 1)
      die("invalid NPY shape: " + path);
    npy.shape.push_back(dim);
  }

  auto& shape = npy.shape;
  if (expected.empty()) {
    if (shape.size() == 4 && shape[0] == 1) shape.erase(shape.begin());
    if (shape.size() != 3 || shape[0] != 3)
      die("expected NPY shape [1,3,H,W] or [3,H,W]: " + path);
    const long H = shape[1], W = shape[2];
    if (H < 16 || W < 16) die("NPY height and width must be at least 16: " + path);
    if (std::max(H, W) > std::numeric_limits<int>::max() ||
        !gfused::FusedPipeline::supports_size(static_cast<int>(H), static_cast<int>(W)))
      die("NPY dimensions exceed CUDA indexing limits: " + path);
  } else if (shape != expected) {
    die("NPY map shape mismatch: " + path);
  }

  size_t count = 1;
  for (long dim : shape) count *= dim;  // bounded by the image or expected map shape above
  const auto bytes = static_cast<std::streamoff>(count * sizeof(float));
  const auto start = f.tellg();
  f.seekg(0, std::ios::end);
  if (f.tellg() - start != bytes) die("NPY payload size does not match shape: " + path);
  f.seekg(start);
  npy.data.resize(count);
  if (!f.read(reinterpret_cast<char*>(npy.data.data()), bytes)) die("cannot read NPY payload: " + path);
  return npy;
}

void save_npy(const std::string& path, const void* data, const std::vector<long>& shape,
              const char* dtype = "<f4", size_t item_size = sizeof(float)) {
  std::string sh = "(";
  size_t count = 1;
  for (size_t i = 0; i < shape.size(); ++i) {
    sh += std::to_string(shape[i]) + (shape.size() == 1 || i + 1 < shape.size() ? "," : "");
    if (i + 1 < shape.size()) sh += " ";
    count *= shape[i];
  }
  sh += ")";
  std::string header = std::string("{'descr': '") + dtype +
                       "', 'fortran_order': False, 'shape': " + sh + ", }";
  const size_t total = 10 + header.size() + 1;
  header += std::string((64 - total % 64) % 64, ' ') + "\n";
  std::ofstream f(path, std::ios::binary);
  if (!f) die("cannot write " + path);
  f.write("\x93NUMPY\x01\x00", 8);
  const uint16_t hl = static_cast<uint16_t>(header.size());
  f.write(reinterpret_cast<const char*>(&hl), 2);
  f.write(header.data(), header.size());
  f.write(static_cast<const char*>(data), count * item_size);
  f.close();
  if (!f) die("could not finish writing " + path);
}

struct Args {
  gfused::PipelineOptions opts;
  int K = 7;
  int repeat = 1;
  int inflight = 1;
  int sustained = 0;  // > 0: also time this many back-to-back graph replays (no host sync)
  bool graph = true;
  std::string input, output, gaussians, manifest;
  std::string video, video_out;  // --video: frame directory or manifest, per-frame output dir
  bool maps = false;  // manifest 4th column / --maps: <stem> of external forward-map outputs
  std::string maps_stem;
};

void usage() {
  std::fprintf(stderr,
               "usage: gaussifier_fused_infer --fm FM.gffm --head HEAD.gfch [--K|--k-max 7]\n"
               "         [--size-mode pad|exact]  (default: exact)\n"
               "         [--seed S] [--psnr-target DB] [--min-gain DB]  (adaptive K, K = cap)\n"
               "         [--n-points N | --pin-count N] [--repeat R] [--sustained R] [--no-graph]\n"
               "         [--gaussians OUT.npy] [--maps STEM | --external-maps] [--check-head-io]\n"
               "         (input.npy output.npy | --manifest FILE [--inflight N])\n"
               "         input: little-endian float32 [3,H,W] or [1,3,H,W], H,W >= 16\n"
               "         manifest: INPUT RENDER [GAUSSIANS [MAPS_STEM]], one image per line\n"
               "       gaussifier_fused_infer --fm FM.gffm --head HEAD.gfch [--K 7]\n"
               "         [--size-mode pad|exact] --video NPY_DIR_OR_MANIFEST --video-out DIR\n"
               "         [--count-ema 0.9] [--warm-attribute-blend 0.1] [--no-graph]\n"
               "         Video frames share one resolution. Directory entries run in lexical order.\n"
               "         A video manifest lists one .npy path per line, relative to the manifest.\n"
               "         Output includes row-aligned .ids.npy (int64) and .slots.npy (int32).\n");
}

template <typename T>
T number_arg(const std::string& value, const std::string& flag) {
  T result{};
  const char* begin = value.data();
  const char* end = begin + value.size();
  if (begin != end && *begin == '+') ++begin;
  const auto parsed = std::from_chars(begin, end, result);
  if (parsed.ec != std::errc{} || parsed.ptr != end ||
      (std::is_floating_point_v<T> && !std::isfinite(result)))
    die("invalid number for " + flag + ": " + value);
  return result;
}

Args parse(int argc, char** argv) {
  Args a;
  std::vector<std::string> pos;
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    auto val = [&]() -> std::string {
      if (i + 1 >= argc) die("missing value for " + k);
      return argv[++i];
    };
    if (k == "--fm") a.opts.forward_map_path = val();
    else if (k == "--head") a.opts.head_path = val();
    else if (k == "--size-mode") {
      const auto mode = val();
      if (mode == "pad") a.opts.size_mode = gfused::SizeMode::Pad;
      else if (mode == "exact") a.opts.size_mode = gfused::SizeMode::Exact;
      else die("--size-mode must be pad or exact");
    }
    else if (k == "--K" || k == "--k-max") a.K = number_arg<int>(val(), k);
    else if (k == "--psnr-target") a.opts.psnr_target = number_arg<float>(val(), k);
    else if (k == "--min-gain") a.opts.min_gain = number_arg<float>(val(), k);
    else if (k == "--seed") a.opts.seed = number_arg<int64_t>(val(), k);
    else if (k == "--n-points") a.opts.n_override = number_arg<int>(val(), k);
    else if (k == "--pin-count") {  // fixed count, no covariance rescale (validation)
      a.opts.n_override = number_arg<int>(val(), k);
      a.opts.count_rescale = false;
    }
    else if (k == "--inflight") a.inflight = number_arg<int>(val(), k);
    else if (k == "--repeat") a.repeat = std::max(1, number_arg<int>(val(), k));
    else if (k == "--check-head-io") a.opts.check_head_io = true;
    else if (k == "--sustained") a.sustained = std::max(0, number_arg<int>(val(), k));
    else if (k == "--no-graph") a.graph = false;
    else if (k == "--gaussians") a.gaussians = val();
    else if (k == "--manifest") a.manifest = val();
    else if (k == "--video") a.video = val();
    else if (k == "--video-out") a.video_out = val();
    else if (k == "--count-ema") a.opts.video_tracks.count_ema = number_arg<double>(val(), k);
    else if (k == "--warm-attribute-blend")
      a.opts.warm_attribute_blend = number_arg<float>(val(), k);
    else if (k == "--maps") {
      a.maps = true;
      a.maps_stem = val();
    } else if (k == "--external-maps") a.maps = true;
    else if (k == "-h" || k == "--help") {
      usage();
      std::exit(0);
    } else if (!k.empty() && k[0] == '-') die("unknown option " + k);
    else pos.push_back(k);
  }
  a.opts.external_maps = a.maps;
  if (a.opts.forward_map_path.empty() || a.opts.head_path.empty()) {
    usage();
    die("--fm and --head are required");
  }
  if (a.K < 1 || a.K > a.opts.max_k) die("--K must be between 1 and " + std::to_string(a.opts.max_k));
  if (a.opts.n_override < 0 || a.opts.n_override > a.opts.render_n_max)
    die("point count override must be between 0 and " + std::to_string(a.opts.render_n_max));
  if (a.inflight < 1) die("--inflight must be positive");
  if (a.sustained > std::numeric_limits<int>::max() - 11) die("--sustained is too large");
  if (a.inflight > 1 && a.manifest.empty()) die("--inflight > 1 requires --manifest");
  if (a.inflight > 1 && (a.repeat > 1 || a.sustained > 0))
    die("--repeat and --sustained require --inflight 1");
  a.opts.video = !a.video.empty();
  if (a.opts.video) {
    // One workspace carries the tracks from frame to frame: no concurrency, no repeats.
    if (a.video_out.empty() || !pos.empty() || !a.manifest.empty() || !a.gaussians.empty() ||
        a.repeat != 1 || a.sustained || a.maps || a.inflight != 1)
      die("--video requires --video-out and cannot be combined with image inputs, "
          "external maps, --inflight, or repeated runs");
  } else if (!a.video_out.empty()) {
    die("--video-out requires --video");
  } else if (a.manifest.empty()) {
    if (pos.size() != 2) {
      usage();
      die("expected input.npy output.npy");
    }
    a.input = pos[0];
    a.output = pos[1];
  } else if (!pos.empty()) {
    die("positional input and output cannot be combined with --manifest");
  }
  return a;
}

double median(std::vector<double> v) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  const size_t m = v.size() / 2;
  return v.size() % 2 ? v[m] : 0.5 * (v[m - 1] + v[m]);
}

struct Job {
  std::string input, render, gaussians, maps;
};

std::vector<Job> video_jobs(const Args& a) {
  namespace fs = std::filesystem;
  const fs::path source(a.video), out(a.video_out);
  std::vector<fs::path> frames;
  if (fs::is_directory(source)) {
    if (fs::weakly_canonical(source) == fs::weakly_canonical(out))
      die("--video-out must differ from the input frame directory");
    for (const auto& entry : fs::directory_iterator(source))
      if (entry.is_regular_file() && entry.path().extension() == ".npy")
        frames.push_back(entry.path());
    std::sort(frames.begin(), frames.end());
  } else {
    std::ifstream manifest(source);
    if (!manifest) die("cannot open video manifest " + a.video);
    std::string line;
    while (std::getline(manifest, line)) {
      const size_t start = line.find_first_not_of(" \t\r");
      if (start == std::string::npos) continue;
      const size_t end = line.find_last_not_of(" \t\r");
      frames.push_back(source.parent_path() / line.substr(start, end - start + 1));
    }
    if (manifest.bad()) die("cannot read video manifest " + a.video);
  }
  if (frames.empty()) die("no video frames in " + a.video);
  fs::create_directories(out);
  std::vector<Job> jobs;
  for (size_t i = 0; i < frames.size(); ++i) {
    char name[32];
    std::snprintf(name, sizeof(name), "%06zu.npy", i);
    jobs.push_back({frames[i].string(), "DUMMY", (out / name).string(), {}});
  }
  return jobs;
}

// Validation: uploads <stem>.density.npy (H, W), .log_cov.npy (3, H, W), .rgb.npy (3, H, W) and
// .rate.npy (1,) into the pipeline's forward-map buffers.
void upload_maps(gfused::FusedPipeline& p, const std::string& stem, cudaStream_t s) {
  const gfused::PipelineMaps m = p.external_maps();
  const long H = p.H(), W = p.W();
  struct Map {
    const char* suffix;
    float* device;
    std::vector<long> shape;
  };
  const Map items[] = {{".density.npy", m.density, {H, W}},
                       {".log_cov.npy", m.log_cov, {3, H, W}},
                       {".rgb.npy", m.rgb, {3, H, W}},
                       {".rate.npy", m.rate_sum, {1}}};
  for (const auto& item : items) {
    const auto data = load_npy(stem + item.suffix, item.shape).data;
    CHECK_CUDA(cudaMemcpyAsync(item.device, data.data(), data.size() * sizeof(float),
                               cudaMemcpyHostToDevice, s));
    CHECK_CUDA(cudaStreamSynchronize(s));
  }
}

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

struct Slot {
  explicit Slot(const gfused::FusedPipeline& model) : pipe(model.make_workspace()) {}
  ~Slot() {
    if (stream) cudaStreamSynchronize(stream);
    pipe.reset();
    if (pinned) cudaFreeHost(pinned);
    if (uploaded) cudaEventDestroy(uploaded);
    if (start) cudaEventDestroy(start);
    if (done) cudaEventDestroy(done);
    if (stream) cudaStreamDestroy(stream);
  }

  cudaError_t prepare(int H, int W) {
    GF_TRY(pipe->prepare(H, W));
    GF_TRY(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    GF_TRY(cudaEventCreate(&uploaded));
    GF_TRY(cudaEventCreate(&start));
    GF_TRY(cudaEventCreate(&done));
    return cudaMallocHost(&pinned, 3 * static_cast<size_t>(H) * W * sizeof(float));
  }

  std::unique_ptr<gfused::FusedPipeline> pipe;
  cudaStream_t stream = nullptr;
  cudaEvent_t uploaded = nullptr, start = nullptr, done = nullptr;
  float* pinned = nullptr;
  std::optional<Job> job;
  size_t last_used = 0;
  size_t workspace_bytes = 0;
  Clock::time_point submitted, first;
  std::vector<double> gpu_ms, wall_ms, h2d_ms;
  double first_ms = 0;
  bool captured = false;
  size_t frame = 0;    // video: frames run in this workspace
  int prev_count = 0;  // video: Gaussians of the previous frame
};

class WorkspaceCache {
 public:
  WorkspaceCache(const gfused::FusedPipeline& model, size_t limit)
      : model_(model), limit_(limit) {}

  Slot& acquire(int H, int W, const Job& job) {
    for (auto& entry : entries_) {
      if (!entry->job && entry->pipe->H() == H && entry->pipe->W() == W) {
        entry->last_used = ++sequence_;
        return *entry;
      }
    }
    if (entries_.size() >= limit_) evict_oldest();
    auto slot = std::make_unique<Slot>(model_);
    size_t before, after, total;
    const auto tp = Clock::now();
    CHECK_CUDA(cudaMemGetInfo(&before, &total));
    cudaError_t e = slot->prepare(H, W);
    // An idle cached resolution should never prevent an allocation that fits on its own.
    while (e == cudaErrorMemoryAllocation) {
      slot.reset();
      cudaGetLastError();  // Clear the failed allocation before retrying on this thread.
      if (!evict_oldest()) break;
      slot = std::make_unique<Slot>(model_);
      CHECK_CUDA(cudaMemGetInfo(&before, &total));
      e = slot->prepare(H, W);
    }
    check_cuda(e, "preparing " + std::to_string(H) + "x" + std::to_string(W) + " for " + job.input);
    CHECK_CUDA(cudaMemGetInfo(&after, &total));
    slot->workspace_bytes = before > after ? before - after : slot->pipe->arena_bytes();
    slot->last_used = ++sequence_;
    std::fprintf(stderr, "[prepare] %dx%d n_cap=%d arena=%.1f MiB device_delta=%.1f MiB (%.1f ms)\n",
                 H, W, slot->pipe->n_cap(), slot->pipe->arena_bytes() / 1048576.0,
                 slot->workspace_bytes / 1048576.0, elapsed_ms(tp));
    entries_.push_back(std::move(slot));
    return *entries_.back();
  }

 private:
  bool evict_oldest() {
    auto oldest = entries_.end();
    for (auto it = entries_.begin(); it != entries_.end(); ++it) {
      if (!(*it)->job && (oldest == entries_.end() || (*it)->last_used < (*oldest)->last_used))
        oldest = it;
    }
    if (oldest == entries_.end()) return false;
    entries_.erase(oldest);
    return true;
  }

  const gfused::FusedPipeline& model_;
  size_t limit_, sequence_ = 0;
  std::vector<std::unique_ptr<Slot>> entries_;
};

void enqueue_replay(const Args& a, Slot& slot) {
  auto& pipe = *slot.pipe;
  const auto s = slot.stream;
  const size_t bytes = 3 * static_cast<size_t>(pipe.H()) * pipe.W() * sizeof(float);
  slot.submitted = Clock::now();
  CHECK_CUDA(cudaEventRecord(slot.uploaded, s));
  CHECK_CUDA(cudaMemcpyAsync(pipe.input_chw(), slot.pinned, bytes, cudaMemcpyHostToDevice, s));
  CHECK_CUDA(cudaEventRecord(slot.start, s));
  check_cuda(a.graph ? pipe.run_graph(a.K, s) : pipe.run(a.K, s), "running " + slot.job->input);
  CHECK_CUDA(cudaEventRecord(slot.done, s));
}

void collect_timing(Slot& slot, bool record) {
  check_cuda(cudaEventSynchronize(slot.done), "finishing " + slot.job->input);
  if (!record) return;
  float upload, compute;
  CHECK_CUDA(cudaEventElapsedTime(&upload, slot.uploaded, slot.start));
  CHECK_CUDA(cudaEventElapsedTime(&compute, slot.start, slot.done));
  slot.h2d_ms.push_back(upload);
  slot.gpu_ms.push_back(compute);
  slot.wall_ms.push_back(elapsed_ms(slot.submitted));
}

void start_job(const Args& a, Slot& slot, Job job, const Npy& image) {
  slot.job = std::move(job);
  const auto& current = *slot.job;
  slot.gpu_ms.clear();
  slot.wall_ms.clear();
  slot.h2d_ms.clear();
  std::memcpy(slot.pinned, image.data.data(), image.data.size() * sizeof(float));
  if (a.maps) {
    const auto& stem = current.maps.empty() ? a.maps_stem : current.maps;
    if (stem.empty()) die("external maps requested but no maps stem for " + current.input);
    upload_maps(*slot.pipe, stem, slot.stream);
  }
  slot.first = Clock::now();
  slot.captured = a.graph && slot.pipe->graph_nodes(a.K) == 0;
  // Video: capture the first-frame and warm-frame graphs up front (without advancing).
  if (a.graph && a.opts.video && slot.frame == 0)
    check_cuda(slot.pipe->prepare_video_graphs(a.K, slot.stream),
               "capturing video graphs for " + current.input);
  if (a.graph)
    check_cuda(slot.pipe->capture_graph(a.K, slot.stream), "capturing graph for " + current.input);
  if (a.repeat > 1) {
    enqueue_replay(a, slot);
    collect_timing(slot, false);
    slot.first_ms = elapsed_ms(slot.first);
    for (int rep = 1; rep < a.repeat; ++rep) {
      enqueue_replay(a, slot);
      collect_timing(slot, true);
    }
  }
  enqueue_replay(a, slot);
}

std::vector<double> sustained_timings(const Args& a, Slot& slot) {
  if (a.sustained == 0) return {};
  const int warm = 10, n = a.sustained + warm;
  std::vector<cudaEvent_t> events(n + 1, nullptr);
  std::vector<double> times;
  const auto s = slot.stream;
  for (auto& event : events) CHECK_CUDA(cudaEventCreate(&event));
  CHECK_CUDA(cudaEventRecord(events[0], s));
  for (int r = 0; r < n; ++r) {
    CHECK_CUDA(a.graph ? slot.pipe->run_graph(a.K, s) : slot.pipe->run(a.K, s));
    CHECK_CUDA(cudaEventRecord(events[r + 1], s));
  }
  CHECK_CUDA(cudaStreamSynchronize(s));
  for (int r = warm; r < n; ++r) {
    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, events[r], events[r + 1]));
    times.push_back(ms);
  }
  for (auto event : events) CHECK_CUDA(cudaEventDestroy(event));
  return times;
}

void read_device(void* host, const void* device, size_t size, cudaStream_t s) {
  CHECK_CUDA(cudaMemcpyAsync(host, device, size, cudaMemcpyDeviceToHost, s));
  CHECK_CUDA(cudaStreamSynchronize(s));
}

void json_string(std::ostream& out, const std::string& s) {
  static constexpr char hex[] = "0123456789abcdef";
  out << '"';
  for (unsigned char c : s) {
    if (c == '"' || c == '\\') out << '\\' << c;
    else if (c < 32) out << "\\u00" << hex[c >> 4] << hex[c & 15];
    else out << c;
  }
  out << '"';
}

void finish_job(const Args& a, Slot& slot) {
  collect_timing(slot, true);
  if (a.repeat == 1) slot.first_ms = elapsed_ms(slot.first);
  const auto sus_ms = sustained_timings(a, slot);
  auto& pipe = *slot.pipe;
  const auto s = slot.stream;
  const Job& job = *slot.job;
  const int H = pipe.H(), W = pipe.W();
  const size_t P = static_cast<size_t>(H) * W;

  gfused::PipelineStats st;
  CHECK_CUDA(pipe.read_stats(&st, s));
  const gfused::PipelineOutputs o = pipe.outputs();
  const int N = st.n, k_used = st.k_used;
  std::vector<double> sq(k_used);
  read_device(sq.data(), o.sq_err, sizeof(double) * k_used, s);
  if (job.render != "DUMMY") {
    read_device(slot.pinned, o.render, 3 * P * sizeof(float), s);
    save_npy(job.render, slot.pinned, {3, H, W});
  }
  if (!job.gaussians.empty()) {
    if (!ends_with(job.gaussians, ".npy")) die("--gaussians must end in .npy");
    const std::string stem = job.gaussians.substr(0, job.gaussians.size() - 4);
    std::vector<float> buf(3 * static_cast<size_t>(N));
    auto dump = [&](const void* src, int C, const std::string& suffix) {
      read_device(buf.data(), src, sizeof(float) * C * N, s);
      if (C == 1) save_npy(stem + suffix, buf.data(), {N});
      else save_npy(stem + suffix, buf.data(), {N, C});
    };
    dump(o.xy, 2, ".points.npy");
    dump(o.log_cov, 3, ".cov.npy");
    dump(o.rgb, 3, ".rgb.npy");
    dump(o.colour, 3, ".color.npy");
    dump(o.weights, 1, ".weights.npy");
    if (a.opts.video) {
      std::vector<int64_t> ids(N);
      std::vector<int32_t> slots(N);
      read_device(ids.data(), o.ids, ids.size() * sizeof(int64_t), s);
      read_device(slots.data(), o.slots, slots.size() * sizeof(int32_t), s);
      save_npy(stem + ".ids.npy", ids.data(), {N}, "<i8", sizeof(int64_t));
      save_npy(stem + ".slots.npy", slots.data(), {N}, "<i4", sizeof(int32_t));
    }
  }

  std::ostringstream js;
  js.precision(6);
  js << std::fixed << "{\"input\":";
  json_string(js, job.input);
  js << ",\"H\":" << H << ",\"W\":" << W << ",\"K\":" << a.K << ",\"k_used\":" << k_used
     << ",\"n\":" << N << ",\"n_requested\":" << st.n_requested
     << ",\"rate_sum\":" << st.rate_sum << ",\"n_init\":" << st.n_init
     << ",\"ed_tile\":" << st.ed_tile_size << ",\"matched\":" << st.matched
     << ",\"forced_adj\":" << st.forced_adj << ",\"forced_nn\":" << st.forced_nn
     << ",\"count_error\":" << st.count_error << ",\"io_mismatch\":" << st.io_mismatch
     << ",\"psnr\":[";
  for (int k = 0; k < k_used; ++k) {
    const double mse = sq[k] / (3.0 * P);
    js << (k ? "," : "") << -10.0 * std::log10(std::max(mse, 1e-10));
  }
  js << "],\"graph\":" << (a.graph ? "true" : "false") << ",\"graph_nodes\":"
     << pipe.graph_nodes(a.K) << ",\"first_ms\":" << slot.first_ms
     << ",\"graph_captured\":" << (slot.captured ? "true" : "false")
     << ",\"arena_bytes\":" << pipe.arena_bytes()
     << ",\"workspace_device_delta_bytes\":" << slot.workspace_bytes
     << ",\"gpu_ms_median\":" << median(slot.gpu_ms) << ",\"wall_ms_median\":" << median(slot.wall_ms)
     << ",\"h2d_ms_median\":" << median(slot.h2d_ms) << ",\"reps\":" << slot.gpu_ms.size();
  if (a.opts.video) {
    const auto& t = st.tracks;
    js << ",\"frame\":" << slot.frame << ",\"gaussians\":";
    json_string(js, job.gaussians);
    js << ",\"n_target\":" << t.target << ",\"n_target_limited\":" << t.limited_target
       << ",\"survivors\":" << t.survivors << ",\"retained_fraction\":"
       << (slot.prev_count ? static_cast<double>(t.survivors) / slot.prev_count : 0.0)
       << ",\"kills_for_next\":" << t.killed << ",\"births_for_next\":" << t.born
       << ",\"shortfall_for_next\":" << t.shortfall;
    slot.prev_count = N;
    ++slot.frame;
  }
  if (!sus_ms.empty()) {
    std::vector<double> sorted = sus_ms;
    std::sort(sorted.begin(), sorted.end());
    const double mean = std::accumulate(sorted.begin(), sorted.end(), 0.0) / sorted.size();
    double squared_deviation = 0.0;
    for (double ms : sorted) squared_deviation += (ms - mean) * (ms - mean);
    const size_t p99 = static_cast<size_t>(std::ceil(0.99 * sorted.size())) - 1;
    js << ",\"sustained_latency\":{\"mean_ms\":" << mean << ",\"min_ms\":" << sorted.front()
       << ",\"max_ms\":" << sorted.back()
       << ",\"stddev_ms\":" << std::sqrt(squared_deviation / sorted.size())
       << ",\"p99_ms\":" << sorted[p99] << "}";
    js << ",\"sustained_ms_median\":" << median(sus_ms)
       << ",\"sustained_ms_mean\":"
       << std::accumulate(sus_ms.begin(), sus_ms.end(), 0.0) / sus_ms.size()
       << ",\"sustained_ms_p10\":" << sorted[sorted.size() / 10]
       << ",\"sustained_ms_p90\":" << sorted[sorted.size() * 9 / 10]
       << ",\"sustained_reps\":" << sus_ms.size();
  }
  std::cout << js.str() << "}" << std::endl;
  slot.job.reset();
}

int run(const Args& a) {
  CHECK_CUDA(cudaSetDevice(0));
  std::ifstream manifest;
  if (!a.manifest.empty()) {
    manifest.open(a.manifest);
    if (!manifest) die("cannot open manifest " + a.manifest);
  }

  gfused::FusedPipeline model;
  std::string err;
  const auto t0 = Clock::now();
  const cudaError_t loaded = model.load(a.opts, &err);
  check_cuda(loaded, "loading weights " + a.opts.forward_map_path + " and " + a.opts.head_path +
                         (err.empty() ? "" : ": " + err));
  std::fprintf(stderr, "[load] %.1f ms\n", elapsed_ms(t0));

  const auto work_start = Clock::now();
  WorkspaceCache cache(model, std::max(2, a.inflight));
  std::deque<Slot*> pending;
  auto finish_oldest = [&]() {
    finish_job(a, *pending.front());
    pending.pop_front();
  };

  size_t jobs = 0;
  int video_h = 0, video_w = 0;
  auto submit = [&](Job job) {
    // The serial path also allows a later job to read the previous job's output.
    if (a.inflight == 1 && !pending.empty()) finish_oldest();
    const auto image = load_npy(job.input);
    if (!std::all_of(image.data.begin(), image.data.end(),
                     [](float v) { return v >= 0.f && v <= 1.f; }))
      die("NPY image values must be finite and in [0,1]: " + job.input);
    if (pending.size() == static_cast<size_t>(a.inflight)) finish_oldest();
    if (a.opts.video) {  // every frame reuses the first frame's workspace and tracks
      if (jobs == 0) video_h = image.shape[1], video_w = image.shape[2];
      if (image.shape[1] != video_h || image.shape[2] != video_w)
        die("all video frames must have the first frame's dimensions: " + job.input);
    }
    auto& slot = cache.acquire(image.shape[1], image.shape[2], job);
    start_job(a, slot, std::move(job), image);
    pending.push_back(&slot);
    ++jobs;
  };
  if (a.opts.video) {
    for (auto& job : video_jobs(a)) submit(std::move(job));
  } else if (a.manifest.empty()) {
    submit({a.input, a.output, a.gaussians, a.maps_stem});
  } else {
    std::string line;
    for (size_t lineno = 1; std::getline(manifest, line); ++lineno) {
      std::stringstream ss(line);
      Job j;
      if (!(ss >> j.input) || j.input.front() == '#') continue;
      std::string extra;
      if (!(ss >> j.render)) die("missing render path at manifest line " + std::to_string(lineno));
      ss >> j.gaussians >> j.maps;
      if (ss >> extra) die("extra field at manifest line " + std::to_string(lineno));
      if (j.gaussians == "-") j.gaussians.clear();
      submit(std::move(j));
    }
    if (manifest.bad()) die("cannot read manifest " + a.manifest);
    if (jobs == 0) die("no images in manifest " + a.manifest);
  }
  while (!pending.empty()) finish_oldest();
  if (!a.manifest.empty()) {
    const double ms = elapsed_ms(work_start);
    std::fprintf(stderr, "[manifest] images=%zu inflight=%d wall_ms=%.3f images_per_sec=%.3f\n",
                 jobs, a.inflight, ms, 1000.0 * jobs / ms);
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run(parse(argc, argv));
  } catch (const std::bad_alloc&) {
    std::fprintf(stderr, "ERROR: out of host memory\n");
  } catch (const std::exception& e) {
    std::fprintf(stderr, "ERROR: %s\n", e.what());
  }
  return 1;
}
