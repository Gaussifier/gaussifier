// Per-kernel isolated-burst benchmark of the fused pipeline (GFUSED_BUILD_CHECKS tool).
//
//   layer_bench --fm fm.gffm --head head.gfch [--H 512 --W 768] [--K 2] [--input img.f32]
//               [--burst-us 1500] [--reps 5] [--filter conv] [--all]
//
// Captures one FusedPipeline::run(K) as a CUDA graph, then for every kernel node builds a
// graph that replays that node alone R times (exact production kernel, launch config and
// arguments) between two event-record nodes, framed by a clock probe. Each measurement is a
// short burst (about --burst-us) after an idle gap, so the SM clock is the boost clock unless
// the card is already power-limited. The probe kernel measures the SM clock right after the
// burst (delta %clock64 / delta %globaltimer over 20 us), which gives the clock-normalised
// efficiency: achieved FLOP/s / (clock x SMs x 1024 FP16 FLOP/clk/SM (mma.sync, FP32 acc)).
//
// FLOPs (2 x MAC, dense-equivalent, no halo recompute counted) are derived from the kernel's
// own arguments for the conv kernels (conv_v2_kernel, conv_block_kernel, head_in / head_enc0 /
// head_fullres); other kernels report time only. Output: one CSV line per kernel node of the
// forward map and the first head call (--all: every node).
#include <cuda_runtime.h>
#include <cxxabi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "conv_block.cuh"
#include "head_fullres.cuh"
#include "pipeline.cuh"

using namespace gfused;

namespace {

__global__ void clock_probe_kernel(unsigned long long* out) {
  unsigned long long t0, t1;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
  const long long c0 = clock64();
  do {
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t1));
  } while (t1 - t0 < 20000);
  const long long c1 = clock64();
  if (threadIdx.x == 0) {
    out[0] = static_cast<unsigned long long>(c1 - c0);
    out[1] = t1 - t0;
  }
}

#define CK(x)                                                                         \
  do {                                                                                \
    cudaError_t e_ = (x);                                                             \
    if (e_ != cudaSuccess) {                                                          \
      std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x,                  \
                   cudaGetErrorString(e_));                                           \
      std::exit(1);                                                                   \
    }                                                                                 \
  } while (0)

std::string demangle(const char* m) {
  int st = 0;
  char* d = abi::__cxa_demangle(m, nullptr, nullptr, &st);
  std::string s = st == 0 && d ? d : m;
  std::free(d);
  return s;
}

// template argument i (0-based, top level) of a demangled "name<a, b<c>, d>(...)"
std::string targ(const std::string& s, int i) {
  size_t p = s.find('<');
  if (p == std::string::npos) return "";
  int depth = 0, idx = 0;
  size_t start = p + 1;
  for (size_t k = p + 1; k < s.size(); ++k) {
    const char c = s[k];
    if (c == '<' || c == '(') ++depth;
    else if ((c == '>' || c == ')') && depth > 0) --depth;
    else if ((c == ',' && depth == 0) || (c == '>' && depth == 0)) {
      if (idx == i) {
        std::string a = s.substr(start, k - start);
        while (!a.empty() && a[0] == ' ') a.erase(0, 1);
        return a;
      }
      ++idx;
      start = k + 1;
      if (c == '>') break;
    }
  }
  return "";
}
int targi(const std::string& s, int i) { return std::atoi(targ(s, i).c_str()); }

struct Info {
  std::string kind;   // short label
  double flops = 0;   // 2 x MAC
  double rd = 0, wr = 0;  // compulsory global bytes (each input and output once, no halo)
  bool fp8 = false;
};

Info classify(const std::string& name, void** args) {
  Info r;
  if (name.find("conv_v2_kernel") != std::string::npos) {
    const ConvParams& p = *static_cast<ConvParams*>(args[0]);
    const ProjParams& q = *static_cast<ProjParams*>(args[1]);
    const int ks = targi(name, 1);
    const int rows = p.rows ? std::min(p.rows, p.H - p.row0) : p.H - p.row0;
    const double pix = static_cast<double>(rows) * p.W;
    const int cin = p.c0 + p.c1;
    const int pc = q.weight ? q.c0 + q.c1 : 0;
    r.flops = 2.0 * pix * p.cout * (ks * ks * cin + pc);
    const int mode = targ(name, 0).back() - '0';  // "(gfused::InMode)2"
    const int pmode = targ(name, 2).back() - '0';
    const double esz = mode == 5 ? 1 : 2;  // kPlainF8 input bytes
    auto in_bytes = [&](int m, int a, int b) {  // a: in0 channels, b: in1 channels
      if (m == 1) return 4.0 * pix * a * 2;          // pool reads 2H x 2W
      if (m == 2) return pix / 4 * a * 2 + pix * b * 2.0;
      if (m == 4) return pix * 12.0;                 // FP32 RGB image
      return pix * (a + b) * esz;
    };
    r.rd = in_bytes(mode, p.c0, p.c1) + (pc && pmode != 3 ? in_bytes(pmode, q.c0, q.c1) : 0.0);
    const bool f8out = name.find("F8StoreEpilogue") != std::string::npos;
    const bool head = name.find("HeadEpilogue") != std::string::npos ||
                      name.find("OutEpilogue") != std::string::npos;
    const bool pool = name.find("PoolEpilogue") != std::string::npos;
    r.wr = head ? pix * 32.0 : pix * p.cout * (f8out ? 1.0 : 2.0) * (pool ? 1.25 : 1.0);
    r.fp8 = targ(name, 9) == "true";
    char buf[160];
    std::snprintf(buf, sizeof buf, "conv m%s %dx%d %d+%d->%d%s%s bk%d th%d bn%d w%dx%d",
                  targ(name, 0).substr(targ(name, 0).size() - 1).c_str(), p.H, p.W, p.c0, p.c1,
                  p.cout, pc ? (" +proj" + std::to_string(pc)).c_str() : "",
                  r.fp8 ? " FP8" : "", targi(name, 3), targi(name, 4), targi(name, 5),
                  targi(name, 6), targi(name, 7));
    r.kind = buf;
  } else if (name.find("conv_ph_kernel") != std::string::npos) {  // upsample folding
    const ConvParams& p = *static_cast<ConvParams*>(args[0]);
    const int rows = p.rows ? std::min(p.rows, p.H - p.row0) : p.H - p.row0;
    const double pix = static_cast<double>(rows) * p.W;
    r.flops = 2.0 * pix * p.cout * 9.0 * (p.c0 + p.c1);
    r.rd = pix / 4 * p.c0 * 2 + pix * p.c1 * 2.0;
    r.wr = pix * p.cout * 2.0;
    char buf[160];
    std::snprintf(buf, sizeof buf, "conv ph %dx%d %d+%d->%d bk%d th%d bn%d w%dx%d bku%d", p.H,
                  p.W, p.c0, p.c1, p.cout, targi(name, 1), targi(name, 0), targi(name, 2),
                  targi(name, 3), targi(name, 4), targi(name, 5));
    r.kind = buf;
  } else if (name.find("conv_block_kernel") != std::string::npos) {
    const BlockParams& bp = *static_cast<BlockParams*>(args[0]);
    const int rows = bp.p.rows ? std::min(bp.p.rows, bp.p.H - bp.p.row0) : bp.p.H - bp.p.row0;
    const double pix = static_cast<double>(rows) * bp.p.W;
    const int xc = bp.q.c0 + bp.q.c1, cm = targi(name, 5), co = bp.p.cout;
    r.flops = 2.0 * pix * (9.0 * xc * cm + 9.0 * cm * co + (bp.q.weight ? xc * co : 0));
    const int mode = targ(name, 0).back() - '0';
    r.rd = mode == 1 ? 4.0 * pix * xc * 2 : mode == 2 ? pix / 4 * bp.q.c0 * 2 + pix * bp.q.c1 * 2.0
         : mode == 4 ? pix * 12.0 : pix * xc * 2.0;
    r.wr = pix * co * 2.0 * (name.find("PoolEpilogue") != std::string::npos ? 1.25 : 1.0);
    char buf[128];
    std::snprintf(buf, sizeof buf, "block %dx%d %d->%d->%d", bp.p.H, bp.p.W, xc, cm, co);
    r.kind = buf;
  } else if (name.find("head_in_conv_kernel") != std::string::npos) {
    const int H = *static_cast<int*>(args[1]), W = *static_cast<int*>(args[2]);
    const int aux8 = targi(name, 2);
    r.flops = 2.0 * H * W * 9 * 16 * (32 + aux8 * 8);
    r.rd = 32.0 * H * W;  // packed FP16 input, two 16-byte planes
    r.wr = 2.0 * H * W * (32 + aux8 * 8);
    char buf[96];
    std::snprintf(buf, sizeof buf, "head_in %dx%d 16->32+%d", H, W, aux8 * 8);
    r.kind = buf;
  } else if (name.find("head_enc0_kernel") != std::string::npos) {
    const int H = *static_cast<int*>(args[1]), W = *static_cast<int*>(args[2]);
    r.flops = 2.0 * H * W * 2 * (9 * 16 * 32 + 9 * 32 * 32);
    r.rd = 32.0 * H * W;
    r.wr = 2.0 * H * W * 64;
    char buf[96];
    std::snprintf(buf, sizeof buf, "head_enc0 %dx%d x2 nets", H, W);
    r.kind = buf;
  } else if (name.find("head_fullres_kernel") != std::string::npos) {
    const headk::FullresParams& p = *static_cast<headk::FullresParams*>(args[0]);
    const int C = targi(name, 0);
    const bool out = targ(name, 3) == "true";
    r.flops = 2.0 * p.H * p.W * (9.0 * C * C + (out ? C * 8 : 0));
    r.rd = 2.0 * p.H * p.W * C;
    r.wr = out ? 32.0 * p.H * p.W : 2.0 * p.H * p.W * C * (targ(name, 4) == "true" ? 1.25 : 1.0);
    char buf[96];
    std::snprintf(buf, sizeof buf, "fullres%s %dx%d %d->%d", out ? "+out" : "", p.H, p.W, C, C);
    r.kind = buf;
  } else {
    std::string base = name.substr(0, name.find_first_of("<("));
    r.kind = base;
  }
  return r;
}

void idle(int ms) { std::this_thread::sleep_for(std::chrono::milliseconds(ms)); }

}  // namespace

int main(int argc, char** argv) {
  std::string fm, head, input;
  int H = 512, W = 768, K = 2, reps = 5;
  double burst_us = 1500;
  bool all = false;
  std::string filter;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto nxt = [&]() { return std::string(argv[++i]); };
    if (a == "--fm") fm = nxt();
    else if (a == "--head") head = nxt();
    else if (a == "--input") input = nxt();
    else if (a == "--H") H = std::stoi(nxt());
    else if (a == "--W") W = std::stoi(nxt());
    else if (a == "--K") K = std::stoi(nxt());
    else if (a == "--reps") reps = std::stoi(nxt());
    else if (a == "--burst-us") burst_us = std::stod(nxt());
    else if (a == "--filter") filter = nxt();
    else if (a == "--all") all = true;
    else {
      std::fprintf(stderr, "unknown arg %s\n", a.c_str());
      return 2;
    }
  }
  PipelineOptions o;
  o.forward_map_path = fm;
  o.head_path = head;
  FusedPipeline pl;
  std::string err;
  if (pl.load(o, &err) != cudaSuccess) {
    std::fprintf(stderr, "load: %s\n", err.c_str());
    return 1;
  }
  CK(pl.prepare(H, W));
  std::vector<float> img(static_cast<size_t>(3) * H * W);
  if (!input.empty()) {
    std::ifstream f(input, std::ios::binary);
    f.read(reinterpret_cast<char*>(img.data()), img.size() * 4);
    if (!f) {
      std::fprintf(stderr, "short input %s\n", input.c_str());
      return 1;
    }
  } else {
    for (int c = 0; c < 3; ++c)
      for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
          img[(static_cast<size_t>(c) * H + y) * W + x] =
              0.5f + 0.4f * std::sin(0.013f * x * (c + 1) + 0.021f * y) *
                         std::cos(0.0071f * (x + y) * (3 - c));
  }
  CK(cudaMemcpy(pl.input_chw(), img.data(), img.size() * 4, cudaMemcpyHostToDevice));
  cudaStream_t s;
  CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  CK(pl.run(K, s));
  CK(cudaStreamSynchronize(s));
  cudaGraph_t g;
  CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
  CK(pl.run(K, s));
  CK(cudaStreamEndCapture(s, &g));
  // topological order
  size_t n = 0;
  CK(cudaGraphGetNodes(g, nullptr, &n));
  std::vector<cudaGraphNode_t> nodes(n);
  CK(cudaGraphGetNodes(g, nodes.data(), &n));
  std::vector<cudaGraphNode_t> order;
  {
    std::vector<int> indeg(n, 0);
    std::vector<std::vector<int>> succ(n);
    for (size_t i = 0; i < n; ++i) {
      size_t m = 0;
      CK(cudaGraphNodeGetDependentNodes(nodes[i], nullptr, nullptr, &m));
      std::vector<cudaGraphNode_t> d(m);
      if (m) CK(cudaGraphNodeGetDependentNodes(nodes[i], d.data(), nullptr, &m));
      for (auto x : d) {
        const int j = static_cast<int>(std::find(nodes.begin(), nodes.end(), x) - nodes.begin());
        succ[i].push_back(j);
        ++indeg[j];
      }
    }
    std::vector<int> ready;
    for (size_t i = 0; i < n; ++i)
      if (!indeg[i]) ready.push_back(static_cast<int>(i));
    while (!ready.empty()) {  // lowest creation index first (stream order)
      auto it = std::min_element(ready.begin(), ready.end());
      const int i = *it;
      ready.erase(it);
      order.push_back(nodes[i]);
      for (int j : succ[i])
        if (--indeg[j] == 0) ready.push_back(j);
    }
  }
  unsigned long long* d_probe;
  CK(cudaMalloc(&d_probe, 16));
  int dev = 0, sms = 0;
  CK(cudaGetDevice(&dev));
  CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
  std::printf("# H=%d W=%d K=%d sms=%d nodes=%zu\n", H, W, K, sms, order.size());
  // Memory roofline: compulsory bytes at the L2-resident rates (read 4.0 TB/s, write 2.5 TB/s,
  // measured with streaming kernels on this L40S) while the layer's footprint fits the
  // 96 MB L2 (<= 80 MB), else at DRAM rates (0.74 / 0.54 TB/s).
  // roof_us = max(FLOPs at the measured clock's peak, memory time); roof = roof_us / us.
  std::printf("idx,kernel,grid,block,smem,us,gflop,tflops,mhz,eff,mem_us,roof_us,roof,limiter\n");
  int idx = 0, conv_seen = 0;
  double tot_us = 0, tot_fl = 0;
  bool first_head_done = false;
  int head_outs = 0;
  for (auto nd : order) {
    cudaGraphNodeType t;
    CK(cudaGraphNodeGetType(nd, &t));
    if (t != cudaGraphNodeTypeKernel) continue;
    cudaKernelNodeParams kp{};
    if (cudaGraphKernelNodeGetParams(nd, &kp) != cudaSuccess) {
      cudaGetLastError();
      continue;
    }
    const char* mname = nullptr;
    if (cudaFuncGetName(&mname, kp.func) != cudaSuccess || !mname) {
      cudaGetLastError();
      mname = "?";
    }
    const std::string name = demangle(mname);
    const int my = idx++;
    const Info inf = classify(name, kp.kernelParams);
    // Default scope: everything up to (and including) the first head's final layer.
    const bool head_out = (name.find("head_fullres_kernel") != std::string::npos &&
                           targ(name, 3) == "true") ||
                          name.find("OutEpilogue") != std::string::npos;
    if (!all && first_head_done) continue;
    if (inf.flops > 0) ++conv_seen;
    if (!filter.empty() && inf.kind.find(filter) == std::string::npos) continue;
    if (!all && inf.flops == 0) continue;  // sampler / renderer kernels need pipeline state
    // burst graph: event, R copies of the node, event, clock probe
    auto build = [&](int R, cudaGraphExec_t& ex, cudaEvent_t a, cudaEvent_t b) {
      cudaGraph_t bg;
      CK(cudaGraphCreate(&bg, 0));
      cudaGraphNode_t prev, cur;
      CK(cudaGraphAddEventRecordNode(&prev, bg, nullptr, 0, a));
      for (int r = 0; r < R; ++r) {
        CK(cudaGraphAddKernelNode(&cur, bg, &prev, 1, &kp));
        prev = cur;
      }
      CK(cudaGraphAddEventRecordNode(&cur, bg, &prev, 1, b));
      prev = cur;
      cudaKernelNodeParams pp{};
      void* pa[] = {&d_probe};
      pp.func = reinterpret_cast<void*>(clock_probe_kernel);
      pp.gridDim = dim3(1);
      pp.blockDim = dim3(32);
      pp.kernelParams = pa;
      CK(cudaGraphAddKernelNode(&cur, bg, &prev, 1, &pp));
      CK(cudaGraphInstantiate(&ex, bg, 0));
      CK(cudaGraphDestroy(bg));
    };
    cudaEvent_t ea, eb;
    CK(cudaEventCreate(&ea));
    CK(cudaEventCreate(&eb));
    cudaGraphExec_t ex1;
    build(3, ex1, ea, eb);
    CK(cudaGraphLaunch(ex1, s));
    CK(cudaStreamSynchronize(s));
    CK(cudaGraphLaunch(ex1, s));
    CK(cudaStreamSynchronize(s));
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, ea, eb));
    CK(cudaGraphExecDestroy(ex1));
    const double est_us = std::max(1.0, ms * 1000.0 / 3);
    const int R = std::max(3, std::min(2000, static_cast<int>(burst_us / est_us)));
    cudaGraphExec_t ex;
    build(R, ex, ea, eb);
    std::vector<double> us, mhz;
    CK(cudaGraphLaunch(ex, s));  // warm
    CK(cudaStreamSynchronize(s));
    for (int rep = 0; rep < reps; ++rep) {
      idle(25);
      CK(cudaGraphLaunch(ex, s));
      CK(cudaStreamSynchronize(s));
      CK(cudaEventElapsedTime(&ms, ea, eb));
      unsigned long long h[2];
      CK(cudaMemcpy(h, d_probe, 16, cudaMemcpyDeviceToHost));
      us.push_back(ms * 1000.0 / R);
      mhz.push_back(h[1] ? 1000.0 * h[0] / h[1] : 0);
    }
    CK(cudaGraphExecDestroy(ex));
    CK(cudaEventDestroy(ea));
    CK(cudaEventDestroy(eb));
    std::sort(us.begin(), us.end());
    std::sort(mhz.begin(), mhz.end());
    const double t_us = us[us.size() / 2], f = mhz[mhz.size() / 2];
    const double tf = inf.flops / (t_us * 1e-6) / 1e12;
    const double peak = f * 1e6 * sms * (inf.fp8 ? 2048.0 : 1024.0) / 1e12;
    tot_us += t_us;
    tot_fl += inf.flops;
    const bool l2 = inf.rd + inf.wr <= 80e6;
    const double mem_us = (inf.rd / (l2 ? 4.0e12 : 0.74e12) + inf.wr / (l2 ? 2.5e12 : 0.54e12)) * 1e6;
    const double comp_us = inf.flops / (peak * 1e12) * 1e6;
    const double roof_us = std::max(mem_us, comp_us);
    // latency: fewer blocks than one full wave of the SMs (tail / fill dominated)
    const unsigned blocks = kp.gridDim.x * kp.gridDim.y;
    const char* lim = blocks < static_cast<unsigned>(sms) ? "latency(<1 wave)"
                      : mem_us > comp_us ? (l2 ? "memory(L2)" : "memory(DRAM)") : "compute";
    std::printf("%d,\"%s\",%ux%u,%u,%zu,%.2f,%.3f,%.1f,%.0f,%.3f,%.2f,%.2f,%.3f,%s\n", my,
                inf.kind.c_str(), kp.gridDim.x, kp.gridDim.y, kp.blockDim.x,
                static_cast<size_t>(kp.sharedMemBytes), t_us, inf.flops / 1e9, tf, f,
                inf.flops > 0 ? tf / peak : 0.0, mem_us, roof_us, roof_us / t_us, lim);
    std::fflush(stdout);
    if (head_out && ++head_outs == 2) first_head_done = true;
  }
  std::printf("# total %.1f us, %.2f GFLOP\n", tot_us, tot_fl / 1e9);
  return 0;
}
