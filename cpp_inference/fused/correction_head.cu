// Torch-free fused correction head: weight loading, workspace, and the layer schedule.
//
// Schedule per call. Two variants, chosen by image size (knob kHeadSchedule in head_tuning.cuh
// forces one); "small" is up to 512 x 1024 pixels, where the working set stays in L2:
//   1. first conv of both U-Nets in one launch (head_conv.cuh). Small: head_in_conv_kernel
//      (persistent, resident weights). Large: the fused enc0 kernel (first conv + enc0.b per
//      net, the intermediate kept in shared memory). Input from the packed FP16 planes of
//      contract C1, or gathered from the FP32 buffers.
//   2. small only: fork; the aux net runs on a second stream.
//   3. per net: enc0.b (small: lean persistent head_fullres kernel for 32-wide nets, else
//      conv_fused), whose epilogue also writes the 2x2-max-pooled skip so enc1.a reads it
//      plainly (likewise enc1.b -> mid.a via epilogues.cuh PoolEpilogue). Then enc1 .. dec1.a
//      on conv_fused (maxpool and bilinear-up2x + concat fused into the staging).
//   4. dec1.b + bias + SiLU + final 1x1 conv: the aux net writes out = aux + b_aux, the main
//      net (after the join) accumulates. Small: lean head_fullres out kernel (the 1x1 as an
//      MMA on the accumulator fragments). Large: conv_fused + OutEpilogue (the lean kernels
//      stream ~15% worse once the layer is DRAM bound). Output planar 8 x H x W FP32.
//
// Channel widths: the aux net (24/48/96) runs at native width (conv core BK 8 / 24) unless
// its enc1.a conv is FP8, which needs 32-multiple input segments; it then runs the copy padded
// to 32 at level 0. Padded channels carry exact zeros.
#include "correction_head.cuh"

#include <cuda_fp16.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

#include "conv_fused.cuh"
#include "epilogues.cuh"
#include "head_conv.cuh"
#include "head_fullres.cuh"
#include "head_tuning.cuh"

namespace gfused {
namespace {

constexpr uint32_t kMagic = 0x48434647;  // "GFCH"
constexpr uint32_t kVersion = 2;  // v1 (no FP8 layers) also accepted
constexpr int kOut = 8;
constexpr int kInReal = 14;  // image3 density1 logcov3 rgb3 rendered3 count1 (diff folded)
constexpr int kInPad = headk::kInCh;
constexpr int kConvsPerNet = 10;
constexpr int kL0 = 32;  // padded level-0 width of both nets (first conv writes 32 + 32)

size_t align16(size_t n) { return (n + 15) & ~size_t(15); }
int pad16(int c) { return (c + 15) & ~15; }
int pad8(int c) { return (c + 7) & ~7; }

struct Conv {
  const __half* w = nullptr;  // [taps][cin_p][cout_p]
  const float* b = nullptr;   // [cout_p]
  int cin = 0, cout = 0;      // padded
  // FP8 (e4m3) layer: weights [taps][cout_p][cin_p], per-output-channel scales, static
  // per-tensor input scale. w8 == nullptr -> FP16.
  const uint8_t* w8 = nullptr;
  const float* w_scale = nullptr;
  float a_scale = 1.0f;
  int acc16 = 0;  // knob kHeadAcc16: ConvParams::acc16 for this FP16 layer (conv core only)
};

struct Net {
  Conv c[kConvsPerNet];  // c[0] unused at run time (merged into first_w)
  int ch[3] = {};        // padded channel widths per level
  const float* out_w = nullptr;  // [ch0][8] FP32
  const float* out_b = nullptr;  // [8]
};

struct WeightsImpl {
  void* blob = nullptr;
  Net net[2];
  const __half* first_w = nullptr;  // [9][16][64]
  const float* first_b = nullptr;   // [64]
  int cfg[2][kConvsPerNet];  // knob kHeadLayerCfg: tile config per layer, -1 = tuned default
  Net aux24;                 // aux net at native widths (24 / 48 / 96), if they differ
  bool has_aux24 = false;
  int aux_native = -1;  // knob kHeadAuxNative: -1 auto, 0 padded, 1 native
  int schedule = -1;    // knob kHeadSchedule: -1 by image size, 0 small, 1 large
};

struct NetBufs {
  // p0 / p1: 2x2-max-pooled s0 [H/2][W/2][c0] / s1 [H/4][W/4][c1]
  __half *t0, *s0, *t1, *s1, *tb, *bot, *m0, *p0, *p1;
};

struct WorkspaceImpl {
  void* blob = nullptr;
  NetBufs nb[2];
  cudaStream_t aux = nullptr;
  cudaEvent_t fork = nullptr, join = nullptr;
};

// ---------------------------------------------------------------- tile configurations

// (BK, TH, BN, WM, WN) candidates; a config is valid for a layer when BK divides every input
// segment and BN divides cout (FP8 layers additionally need BK >= 32). Chosen per layer by the
// sweep in the test script.
struct CfgDesc {
  int bk, th, bn, wm, wn;
};
#define GF_HEAD_CFGS(X)                                                                  \
  X(0, 32, 16, 32, 8, 1) X(1, 16, 16, 32, 8, 1) X(2, 32, 8, 32, 4, 1)                   \
  X(3, 32, 8, 64, 4, 2) X(4, 32, 16, 64, 8, 1) X(5, 16, 16, 16, 8, 1)                   \
  X(6, 16, 8, 16, 4, 1) X(7, 64, 8, 32, 4, 1) X(8, 32, 4, 32, 4, 1)                     \
  X(9, 16, 8, 32, 4, 1) X(10, 64, 8, 64, 4, 2) X(11, 32, 8, 16, 4, 1)                   \
  X(12, 16, 16, 64, 8, 1) X(13, 16, 8, 64, 4, 2) X(14, 16, 16, 32, 4, 2)                \
  X(15, 16, 32, 32, 8, 1) X(16, 16, 16, 48, 8, 1) X(17, 16, 8, 48, 4, 1)                \
  X(18, 32, 8, 48, 4, 1) X(19, 32, 16, 48, 8, 1)                                         \
  /* native 24-wide aux (BK an odd multiple of 8) */                                     \
  X(20, 24, 16, 24, 4, 1) X(21, 24, 8, 24, 8, 1) X(22, 24, 8, 24, 4, 1)                 \
  X(23, 24, 16, 48, 8, 1) X(24, 24, 8, 48, 4, 2) X(25, 8, 16, 24, 8, 1)                 \
  X(26, 8, 8, 24, 4, 1) X(27, 24, 16, 24, 8, 1) X(28, 24, 12, 24, 6, 1)                 \
  /* 4 tile rows per warp (fewer shared-memory fragment loads per MMA) */                \
  X(29, 16, 16, 32, 4, 1) X(30, 16, 16, 64, 4, 2)
#define GF_DESC(i, bk, th, bn, wm, wn) {bk, th, bn, wm, wn},
constexpr CfgDesc kCfgs[kHeadNumCfgs] = {GF_HEAD_CFGS(GF_DESC)};
#undef GF_DESC

template <InMode M, class Epi, bool kF8, int BK, int TH, int BN, int WM, int WN>
cudaError_t launch_one(const ConvParams& p, const typename Epi::Params& ep, cudaStream_t s) {
  if constexpr (kF8 && BK < 32) {
    return cudaErrorInvalidValue;
  } else {
    return launch_conv2<M, 3, InMode::kNoProj, BK, TH, BN, WM, WN, Epi, kF8>(p, ProjParams{},
                                                                           ep, s);
  }
}

template <InMode M, class Epi, bool kF8 = false>
cudaError_t launch_idx(int idx, const ConvParams& p, const typename Epi::Params& ep,
                       cudaStream_t s) {
  switch (idx) {
#define GF_CASE(i, bk, th, bn, wm, wn) \
  case i: return launch_one<M, Epi, kF8, bk, th, bn, wm, wn>(p, ep, s);
    GF_HEAD_CFGS(GF_CASE)
#undef GF_CASE
  }
  return cudaErrorInvalidValue;
}

// Dynamic shared memory of a config (depends on the input mode: kUpCat adds the raw upsample
// patch slots), so configs that exceed the device's opt-in limit are never selected.
// single: the launch is a single-chunk one with ConvParams::single_stage (one smem stage).
template <InMode M, bool kF8, int BK, int TH, int BN, int WM, int WN>
constexpr int smem_one(bool single) {
  if constexpr (kF8 && BK < 32) {
    return 1 << 30;
  } else {
    using Cfg = ConvV2Cfg<M, 3, InMode::kNoProj, BK, TH, BN, WM, WN, kF8>;
    return single ? Cfg::kSmem1 : Cfg::kSmem;
  }
}
template <InMode M, bool kF8>
int smem_idx(int idx, bool single) {
  switch (idx) {
#define GF_SMEM(i, bk, th, bn, wm, wn) \
  case i: return smem_one<M, kF8, bk, th, bn, wm, wn>(single);
    GF_HEAD_CFGS(GF_SMEM)
#undef GF_SMEM
  }
  return 1 << 30;
}
int max_smem() {
  int size = 0;
  const auto status = detail::device_launch_value(size, [](int device, int& value) {
    return cudaDeviceGetAttribute(&value, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
  });
  return status == cudaSuccess ? size : 99 * 1024;
}

bool cfg_valid(int idx, InMode mode, int c0, int c1, int cout, bool out_epi, bool f8,
               bool single_stage = false) {
  if (idx < 0 || idx >= kHeadNumCfgs) return false;
  const CfgDesc d = kCfgs[idx];
  const bool one = single_stage && !is_upcat(mode) && c1 == 0 && c0 == d.bk;
  int smem = 0;
  switch (mode) {
    case InMode::kPlain: smem = f8 ? smem_idx<InMode::kPlain, true>(idx, one)
                                   : smem_idx<InMode::kPlain, false>(idx, one); break;
    case InMode::kPool: smem = f8 ? smem_idx<InMode::kPool, true>(idx, one)
                                  : smem_idx<InMode::kPool, false>(idx, one); break;
    case InMode::kUpCatRs: smem = f8 ? smem_idx<InMode::kUpCatRs, true>(idx, one)
                                     : smem_idx<InMode::kUpCatRs, false>(idx, one); break;
    default: smem = f8 ? smem_idx<InMode::kUpCat, true>(idx, one)
                       : smem_idx<InMode::kUpCat, false>(idx, one); break;
  }
  if (smem > max_smem()) return false;
  if (c0 % d.bk || c1 % d.bk || cout % d.bn) return false;
  if (out_epi && (d.bn != cout || f8)) return false;
  if (f8 && d.bk < 32) return false;
  return true;
}

int default_cfg(InMode mode, int c0, int c1, int cout, bool out_epi, bool f8,
                bool single_stage = false) {
  for (int idx : {0, 1, 5, 2, 18, 11, 8, 28, 22, 26}) {
    if (cfg_valid(idx, mode, c0, c1, cout, out_epi, f8, single_stage)) return idx;
  }
  return -1;
}

// Per-layer configs from the sweep (tests/fused/test_correction_head_fused.py --tune) on L40S,
// for the v1.5 head (main 32/64/128, aux 24/48/96). Index = layer 1..9 (0 unused). Small
// images (<= 512 x 1024 pixels) keep the working set in L2 and prefer short 8-row tiles with
// 64-wide N (more blocks, fewer weight reloads); larger ones prefer 16-row tiles.
// Re-tuned (17:5x core: upsample-patch prefetch, BN = 48, FP8; main enc1.a small: config 0,
// 22.6 vs 24.6 us at 512 x 768 and 16.7 vs 17.6 us at 512 x 512 after the packed-half2
// upsample staging). Configs that exceed the
// device's opt-in shared memory for a layer's mode are rejected by cfg_valid (e.g. config 3 in
// kUpCat FP16), and FP8 config 2 spills in kUpCat, so it is not used for merge.a layers.
// The tables are per architecture; any device other than sm_120 uses the sm_89 (L40S) set.
struct HeadTables {
  int small[2][kConvsPerNet], large[2][kConvsPerNet];  // FP16 layers (aux: 32-padded)
  // FP8 layers (BK >= 32 configs only); -1 = default.
  int f8_small[2][kConvsPerNet], f8_large[2][kConvsPerNet];
  int a24_small[kConvsPerNet], a24_large[kConvsPerNet];  // native 24-wide aux (configs 20..28)
};
constexpr HeadTables kTablesSm89 = {
    {{-1, 14, 0, 14, 13, 13, 1, 13, 1, 1}, {-1, 14, 16, 5, 9, 9, 16, 5, 1, 1}},
    {{-1, 11, 13, 1, 13, 1, 1, 13, 1, 14}, {-1, 11, 18, 5, 16, 1, 17, 5, 1, 9}},
    // FP8 halves the operand shared memory, so the 16-row, 64-wide config 4 fits for the
    // upsample merge.
    {{-1, 0, 3, 3, 3, 3, 4, 0, 0, 0}, {-1, 0, 18, 0, 18, 18, 0, 0, 0, 0}},
    {{-1, 0, 3, 3, 3, 3, 4, 3, 0, 0}, {-1, 0, 18, 0, 18, 18, 0, 0, 0, 0}},
    // native aux, from the --tune sweep with pooled skips
    {-1, 20, 27, 5, 9, 9, 16, 5, 25, 20},
    {-1, 21, 23, 5, 1, 1, 17, 5, 22, 28},
};
// sm_120 (RTX 5090), from --tune with FP16 accumulation (the sm_120 default, which also moves
// enc0.b and dec1.b off the lean kernels): small = 512 x 512 + 512 x 768, large = 1024 x 1536
// + 1280 x 720, each layer's config minimising the summed time relative to the per-size best.
// FP16 main net and native aux from the FP16 blobs; the FP8 layers and the 32-padded aux net's
// FP16 layers from the budget_0.03dB blobs (where enc1.a FP8 forces the padded aux).
constexpr HeadTables kTablesSm120 = {
    {{-1, 1, 14, 13, 14, 14, 1, 13, 1, 2}, {-1, 1, 16, 5, 14, 9, 5, 5, 1, 14}},
    {{-1, 14, 13, 1, 13, 14, 14, 13, 1, 14}, {-1, 14, 18, 16, 14, 1, 5, 16, 1, 14}},
    {{-1, 0, 3, 3, 7, 2, 2, 0, 0, 0}, {-1, 0, 11, 0, 18, 11, 0, 0, 0, 0}},
    {{-1, 0, 3, 3, 3, 2, 4, 3, 0, 0}, {-1, 0, 18, 0, 18, 2, 0, 0, 0, 0}},
    {-1, 27, 27, 5, 14, 14, 5, 5, 25, 20},
    {-1, 28, 23, 16, 14, 14, 17, 16, 25, 28},
};
int tuned_cfg(int k, int i, long long pixels, bool f8, bool native = false) {
  const HeadTables& t = device_arch() == 120 ? kTablesSm120 : kTablesSm89;
  const bool small = pixels <= 512LL * 1024;
  if (f8) return small ? t.f8_small[k][i] : t.f8_large[k][i];
  if (native && k == 1) return small ? t.a24_small[i] : t.a24_large[i];
  return small ? t.small[k][i] : t.large[k][i];
}

// Native-width aux net: auto = on unless the aux enc1.a conv runs FP8 (its 24-channel input
// then needs the 32-padded layout).
bool aux_native(const WeightsImpl& w) {
  if (!w.has_aux24) return false;
  if (w.aux_native >= 0) return w.aux_native != 0;
  return w.net[1].c[2].w8 == nullptr;
}
const Net& net_of(const WeightsImpl& w, int k) {
  return k == 1 && aux_native(w) ? w.aux24 : w.net[k];
}

struct LayerSpec {
  InMode mode;
  const __half* in0;
  int c0;
  const __half* in1;
  int c1;
  int H, W;
  __half* out;  // nullptr for the OutEpilogue layer
};

// pool: bit 0 = enc1.a reads the pooled p0, bit 1 = mid.a reads the pooled p1.
LayerSpec layer_spec(const Net& n, const NetBufs& b, int i, int H, int W, int pool = 0) {
  const int c0 = n.ch[0], c1 = n.ch[1], c2 = n.ch[2];
  const int H2 = H / 2, W2 = W / 2, H4 = H / 4, W4 = W / 4;
  switch (i) {
    case 1: return {InMode::kPlain, b.t0, c0, nullptr, 0, H, W, b.s0};
    case 2:
      if (pool & 1) return {InMode::kPlain, b.p0, c0, nullptr, 0, H2, W2, b.t1};
      return {InMode::kPool, b.s0, c0, nullptr, 0, H2, W2, b.t1};
    case 3: return {InMode::kPlain, b.t1, c1, nullptr, 0, H2, W2, b.s1};
    case 4:
      if (pool & 2) return {InMode::kPlain, b.p1, c1, nullptr, 0, H4, W4, b.tb};
      return {InMode::kPool, b.s1, c1, nullptr, 0, H4, W4, b.tb};
    case 5: return {InMode::kPlain, b.tb, c2, nullptr, 0, H4, W4, b.bot};
    case 6: return {InMode::kUpCat, b.bot, c2, b.s1, c1, H2, W2, b.t1};
    case 7: return {InMode::kPlain, b.t1, c1, nullptr, 0, H2, W2, b.m0};
    case 8: return {InMode::kUpCat, b.m0, c1, b.s0, c0, H, W, b.t0};
    default: return {InMode::kPlain, b.t0, c0, nullptr, 0, H, W, nullptr};  // 9
  }
}

cudaError_t run_layer(const WeightsImpl& w, int k, int i, const NetBufs& b, int H, int W,
                      float* out, bool accumulate, cudaStream_t s, int pool = 0) {
  const Net& n = net_of(w, k);
  const Conv& L = n.c[i];
  LayerSpec sp = layer_spec(n, b, i, H, W, pool);
  // An odd decoder level: the reference resizes its up2x to the skip's size.
  if (sp.mode == InMode::kUpCat && (sp.H % 2 || sp.W % 2)) sp.mode = InMode::kUpCatRs;
  const bool small = static_cast<long long>(H) * W <= 512LL * 1024;
  // producer of a pooled skip (PoolEpilogue: stores the layer output and its 2x2 max)
  __half* pooled = (i == 1 && (pool & 1)) ? b.p0 : (i == 3 && (pool & 2)) ? b.p1 : nullptr;
  ConvParams p{};
  p.in0 = sp.in0;
  p.c0 = sp.c0;
  p.in1 = sp.in1;
  p.c1 = sp.c1;
  p.H = sp.H;
  p.W = sp.W;
  p.weight = L.w;
  p.bias = L.b;
  p.residual = nullptr;
  p.out = sp.out;
  p.cout = L.cout;
  p.silu = 1;
  p.single_stage = small ? 1 : 0;  // one smem stage for single-chunk launches (BK 24 layers)
  p.lerp16 = 1;  // half2 upsample interpolation in the merge layers (dec0.a, dec1.a)
  const bool f8 = L.w8 != nullptr;
  p.acc16 = f8 ? 0 : L.acc16;
  if (f8) {
    p.weight8 = L.w8;
    p.w_scale = L.w_scale;
    p.a_scale = L.a_scale;
  }
  const bool out_epi = i == 9;
  int idx = w.cfg[k][i];
  if (idx < 0) idx = tuned_cfg(k, i, static_cast<long long>(H) * W, f8, &n == &w.aux24);
  const bool ss = p.single_stage != 0;
  if (!cfg_valid(idx, sp.mode, sp.c0, sp.c1, L.cout, out_epi, f8, ss))
    idx = default_cfg(sp.mode, sp.c0, sp.c1, L.cout, out_epi, f8, ss);
  if (!cfg_valid(idx, sp.mode, sp.c0, sp.c1, L.cout, out_epi, f8, ss))
    return cudaErrorInvalidValue;
  if (pooled) {  // enc0.b / enc1.b: plain input
    if (sp.mode != InMode::kPlain) return cudaErrorInvalidValue;
    const PoolEpilogue::Params ep{pooled};
    return f8 ? launch_idx<InMode::kPlain, PoolEpilogue, true>(idx, p, ep, s)
              : launch_idx<InMode::kPlain, PoolEpilogue>(idx, p, ep, s);
  }
  if (f8) {
    const DefaultEpilogue::Params ep{};
    switch (sp.mode) {
      case InMode::kPlain: return launch_idx<InMode::kPlain, DefaultEpilogue, true>(idx, p, ep, s);
      case InMode::kPool: return launch_idx<InMode::kPool, DefaultEpilogue, true>(idx, p, ep, s);
      case InMode::kUpCat: return launch_idx<InMode::kUpCat, DefaultEpilogue, true>(idx, p, ep, s);
      case InMode::kUpCatRs:
        return launch_idx<InMode::kUpCatRs, DefaultEpilogue, true>(idx, p, ep, s);
      default: return cudaErrorInvalidValue;
    }
  }
  if (out_epi) {
    const headk::OutEpilogue::Params ep{n.out_w, n.out_b, out, accumulate ? 1 : 0};
    return launch_idx<InMode::kPlain, headk::OutEpilogue>(idx, p, ep, s);
  }
  const DefaultEpilogue::Params ep{};
  switch (sp.mode) {
    case InMode::kPlain: return launch_idx<InMode::kPlain, DefaultEpilogue>(idx, p, ep, s);
    case InMode::kPool: return launch_idx<InMode::kPool, DefaultEpilogue>(idx, p, ep, s);
    case InMode::kUpCat: return launch_idx<InMode::kUpCat, DefaultEpilogue>(idx, p, ep, s);
    case InMode::kUpCatRs: return launch_idx<InMode::kUpCatRs, DefaultEpilogue>(idx, p, ep, s);
    default: return cudaErrorInvalidValue;
  }
}

// Small-image schedule (see the header comment); the knob overrides the size rule.
bool small_schedule(const WeightsImpl& w, int H, int W) {
  if (w.schedule >= 0) return w.schedule == 0;
  return static_cast<long long>(H) * W <= 512LL * 1024;
}

// ---------------------------------------------------------------- full-res head kernels

// head_fullres tile config: 16-row tiles, 4 warps, two-stage chunk pipeline.
constexpr int kFullresTH = 16, kFullresWarps = 4;

// The lean kernels are 32-channel FP16 only (16-channel chunks): a native 24-wide aux net or
// an FP8 layer runs its enc0.b / dec1.b + out on the generic core, as does a layer with FP16
// accumulation (knob kHeadAcc16; the lean kernels accumulate in FP32).
bool fullres_ok(const WeightsImpl& w, int k, int i) {
  const Net& n = net_of(w, k);
  return n.ch[0] == 32 && n.c[i].w8 == nullptr && n.c[i].acc16 == 0;
}
struct Sched {
  bool small = false;    // head_in first conv + dual stream (else fused enc0, one stream)
  bool lean_enc0b[2] = {};  // per-net lean enc0.b kernel
  bool lean_out[2] = {};    // per-net lean dec1.b + out kernel
  int pool = 0;  // pooled skips (LayerSpec `pool` bits): bit 0 enc0.b -> enc1.a, bit 1 enc1.b
                 // -> mid.a. The fused enc0 kernel does enc0.b, so it stores no pooled copy.
  bool enc0b_done(int k) const { return !small || lean_enc0b[k]; }
};
Sched schedule(const WeightsImpl& w, int H, int W) {
  Sched c;
  c.small = small_schedule(w, H, W);
  c.pool = c.small ? 3 : 2;
  for (int k = 0; k < 2; ++k) {
    c.lean_enc0b[k] = c.small && fullres_ok(w, k, 1);
    c.lean_out[k] = c.small && fullres_ok(w, k, 9);
  }
  return c;
}

// Lean enc0.b of net k (with the pooled skip when `pool`).
cudaError_t run_pair(const WeightsImpl& w, WorkspaceImpl& S, int H, int W, bool pool, int k,
                     cudaStream_t s) {
  headk::FullresParams p{};
  p.H = H;
  p.W = W;
  const Conv& L = net_of(w, k).c[1];
  p.net.in = S.nb[k].t0;
  p.net.w = L.w;
  p.net.b = L.b;
  p.net.out = S.nb[k].s0;
  p.net.pooled = S.nb[k].p0;
  return pool ? headk::launch_head_fullres<32, kFullresTH, kFullresWarps, false, true>(p, s)
              : headk::launch_head_fullres<32, kFullresTH, kFullresWarps, false, false>(p, s);
}

// Lean dec1.b + out of net k (accumulate: out += net, else out = net).
cudaError_t run_outk(const WeightsImpl& w, WorkspaceImpl& S, int H, int W, float* out, int k,
                     bool accumulate, cudaStream_t s) {
  headk::FullresParams p{};
  p.H = H;
  p.W = W;
  p.out = out;
  p.accumulate = accumulate ? 1 : 0;
  const Net& n = net_of(w, k);
  const Conv& L = n.c[9];
  p.net.in = S.nb[k].t0;
  p.net.w = L.w;
  p.net.b = L.b;
  p.net.w_out = n.out_w;
  p.net.b_out = n.out_b;
  return headk::launch_head_fullres<32, kFullresTH, kFullresWarps, true, false>(p, s);
}

cudaError_t run_first(const WeightsImpl& w, const CorrectionHeadInputs& in, WorkspaceImpl& S,
                      int H, int W, bool small, cudaStream_t s) {
  const int ac = net_of(w, 1).ch[0];  // aux level-0 channel stride (24 native, 32 padded)
  if (!small) {  // fused enc0 double conv: writes the level-0 skips directly
    const Conv &m1 = w.net[0].c[1], &a1 = w.net[1].c[1];
    return headk::launch_head_enc0<8, 8>(in, H, W, w.first_w, w.first_b, m1.w, m1.b, a1.w, a1.b,
                                         S.nb[0].s0, S.nb[1].s0, ac, s);
  }
  return headk::launch_head_in<16, 4>(in, H, W, w.first_w, w.first_b, S.nb[0].t0, S.nb[1].t0,
                                      ac, s);
}

}  // namespace

// ---------------------------------------------------------------- weights

cudaError_t load_correction_head_weights(const std::string& path, CorrectionHeadWeights* out,
                                         std::string* error) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    if (error) *error = "cannot open " + path;
    return cudaErrorInvalidValue;
  }
  std::vector<char> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  const auto* bytes = reinterpret_cast<const uint8_t*>(buf.data());
  const size_t size = buf.size();
  size_t off = 0;
  auto rd = [&](uint32_t* v) {
    if (off + 4 > size) return false;
    std::memcpy(v, bytes + off, 4);
    off += 4;
    return true;
  };
  auto fail = [&](const std::string& msg) {
    if (error) *error = msg;
    return cudaErrorInvalidValue;
  };
  uint32_t magic, version, n;
  if (!rd(&magic) || !rd(&version) || !rd(&n) || magic != kMagic || version < 1 ||
      version > kVersion)
    return fail("not a GFCH v1/v2 weight blob");
  if (n != 2 * (kConvsPerNet + 1)) return fail("unexpected conv count " + std::to_string(n));

  struct Raw {
    uint32_t ks, cin, cout;
    std::vector<__half> w;  // copied: the file buffer has no alignment guarantee
    std::vector<float> b;
    uint32_t prec = 0;           // 1 = FP8
    float a_scale = 1.0f;
    std::vector<uint8_t> w8;     // [taps][cout][cin] e4m3
    std::vector<float> w_scale;  // [cout]
  };
  std::vector<Raw> raw(n);
  for (auto& r : raw) {
    if (!rd(&r.ks) || !rd(&r.cin) || !rd(&r.cout) || !rd(&r.prec))
      return fail("truncated conv header");
    if (version < 2) r.prec = 0;
    off = align16(off);
    const size_t nw = static_cast<size_t>(r.ks) * r.ks * r.cin * r.cout;
    if (off + nw * 2 > size) return fail("truncated weights");
    r.w.resize(nw);
    std::memcpy(r.w.data(), bytes + off, nw * 2);
    off = align16(off + nw * 2);
    if (off + r.cout * 4 > size) return fail("truncated bias");
    r.b.resize(r.cout);
    std::memcpy(r.b.data(), bytes + off, r.cout * 4);
    off += r.cout * 4;
    if (r.prec == 1) {
      off = align16(off);
      if (off + 4 > size) return fail("truncated a_scale");
      std::memcpy(&r.a_scale, bytes + off, 4);
      off = align16(off + 4);
      r.w8.resize(nw);
      if (off + nw > size) return fail("truncated FP8 weights");
      std::memcpy(r.w8.data(), bytes + off, nw);
      off = align16(off + nw);
      if (off + r.cout * 4 > size) return fail("truncated w_scale");
      r.w_scale.resize(r.cout);
      std::memcpy(r.w_scale.data(), bytes + off, r.cout * 4);
      off += r.cout * 4;
    } else if (r.prec != 0) {
      return fail("unknown conv precision");
    }
  }

  // Host staging of the padded device layout.
  std::vector<uint8_t> host;
  auto put = [&](const void* src, size_t nb) {
    const size_t o = align16(host.size());
    host.resize(o + nb, 0);
    if (src) std::memcpy(host.data() + o, src, nb);
    return o;
  };
  struct Seg {
    int real, pad;
  };
  // Pad a conv into [taps][cin_p][ld] at column offset col0 (weights) / bias[col0 + o].
  auto pad_into = [&](const Raw& r, const std::vector<Seg>& segs, int ld, int col0,
                      std::vector<__half>& w, std::vector<float>& b) {
    int cin_p = 0, cin_r = 0;
    for (auto s : segs) cin_p += s.pad, cin_r += s.real;
    if (cin_r != static_cast<int>(r.cin)) return false;
    const int taps = r.ks * r.ks;
    if (w.empty()) w.assign(static_cast<size_t>(taps) * cin_p * ld, __float2half(0.0f));
    if (b.empty()) b.assign(ld, 0.0f);
    for (int t = 0; t < taps; ++t) {
      int src_c = 0, dst_c = 0;
      for (auto s : segs) {
        for (int c = 0; c < s.real; ++c)
          for (int o = 0; o < static_cast<int>(r.cout); ++o)
            w[(static_cast<size_t>(t) * cin_p + dst_c + c) * ld + col0 + o] =
                r.w[(static_cast<size_t>(t) * r.cin + src_c + c) * r.cout + o];
        src_c += s.real;
        dst_c += s.pad;
      }
    }
    for (int o = 0; o < static_cast<int>(r.cout); ++o) b[col0 + o] = r.b[o];
    return true;
  };

  auto* impl = new WeightsImpl();
  for (auto& row : impl->cfg)
    for (int& c : row) c = -1;
  auto bail = [&](const std::string& msg) {
    delete impl;
    return fail(msg);
  };
  // Slots: 0 main, 1 aux (level widths padded to 16: 24 -> 32), 2 aux at native widths
  // (padded to 8 only; conv core BK 8 / 24). Slot 2 has no FP8 layer whose input segments
  // are not multiples of 32 (that layer runs FP16 there).
  size_t wofs[3][kConvsPerNet] = {}, bofs[3][kConvsPerNet] = {}, owo[3] = {}, obo[3] = {};
  size_t w8ofs[3][kConvsPerNet] = {}, wsofs[3][kConvsPerNet] = {};
  bool f8[3][kConvsPerNet] = {};
  std::vector<__half> fw;
  std::vector<float> fb;
  {
    const Raw* A = raw.data() + (kConvsPerNet + 1);
    impl->has_aux24 = pad8(static_cast<int>(A[0].cout)) != pad16(static_cast<int>(A[0].cout));
  }
  for (int slot = 0; slot < 3; ++slot) {
    if (slot == 2 && !impl->has_aux24) break;
    const int k = slot == 0 ? 0 : 1;
    const Raw* R = raw.data() + k * (kConvsPerNet + 1);
    const int r0 = R[0].cout, r1 = R[2].cout, r2 = R[4].cout;
    auto padw = [&](int c) { return slot == 2 ? pad8(c) : pad16(c); };
    const int p0 = padw(r0), p1 = padw(r1), p2 = padw(r2);
    if ((slot < 2 && p0 != kL0) || p0 > 32 || p1 > 64 || p2 > 128)
      return bail("unsupported head widths");
    Net& net = slot == 2 ? impl->aux24 : impl->net[k];
    net.ch[0] = p0, net.ch[1] = p1, net.ch[2] = p2;
    const std::vector<Seg> segs[kConvsPerNet] = {
        {{kInReal, kInPad}}, {{r0, p0}}, {{r0, p0}}, {{r1, p1}}, {{r1, p1}},
        {{r2, p2}},          {{r2, p2}, {r1, p1}},   {{r1, p1}}, {{r1, p1}, {r0, p0}}, {{r0, p0}}};
    const int couts[kConvsPerNet] = {p0, p0, p1, p1, p2, p2, p1, p1, p0, p0};
    for (int i = 0; i < kConvsPerNet; ++i) {
      if (R[i].ks != 3) return bail("conv " + std::to_string(k * 11 + i) + " is not 3x3");
      if (R[i].prec == 1 && (i == 0 || i == 9))
        return bail("FP8 is not supported for conv " + std::to_string(k * 11 + i));
      if (i == 0) {  // both nets' first convs share one [9][16][64] weight
        if (slot < 2 && !pad_into(R[0], segs[0], headk::kFirstCout, k * kL0, fw, fb))
          return bail("unexpected first conv shape");
        continue;
      }
      std::vector<__half> w;
      std::vector<float> b;
      if (!pad_into(R[i], segs[i], couts[i], 0, w, b))
        return bail("unexpected shape for conv " + std::to_string(k * 11 + i));
      wofs[slot][i] = put(w.data(), w.size() * 2);
      bofs[slot][i] = put(b.data(), b.size() * 4);
      net.c[i].cin = 0;
      for (auto s : segs[i]) net.c[i].cin += s.pad;
      net.c[i].cout = couts[i];
      bool seg32 = true;
      for (auto sg : segs[i]) seg32 = seg32 && sg.pad % 32 == 0;
      if (R[i].prec == 1 && slot == 2 && !seg32) continue;  // native aux: this layer FP16
      if (R[i].prec == 1) {  // e4m3 [taps][cout_p][cin_p] (zero = 0.0 in e4m3), scales 1.0 pad
        const int cin_p = net.c[i].cin, cout_p = couts[i];
        if (!seg32)
          return bail("FP8 conv " + std::to_string(k * 11 + i) +
                      " needs input segments that are multiples of 32");
        f8[slot][i] = true;
        std::vector<uint8_t> w8(static_cast<size_t>(9) * cout_p * cin_p, 0);
        for (int t = 0; t < 9; ++t)
          for (int o = 0; o < static_cast<int>(R[i].cout); ++o) {
            int src_c = 0, dst_c = 0;
            for (auto sg : segs[i]) {
              for (int c = 0; c < sg.real; ++c)
                w8[(static_cast<size_t>(t) * cout_p + o) * cin_p + dst_c + c] =
                    R[i].w8[(static_cast<size_t>(t) * R[i].cout + o) * R[i].cin + src_c + c];
              src_c += sg.real;
              dst_c += sg.pad;
            }
          }
        std::vector<float> ws(cout_p, 1.0f);
        std::memcpy(ws.data(), R[i].w_scale.data(), R[i].cout * 4);
        w8ofs[slot][i] = put(w8.data(), w8.size());
        wsofs[slot][i] = put(ws.data(), ws.size() * 4);
        net.c[i].a_scale = R[i].a_scale;
      }
    }
    const Raw& ro = R[kConvsPerNet];
    if (ro.ks != 1 || ro.cout != kOut || static_cast<int>(ro.cin) != r0)
      return bail("unexpected out conv shape");
    std::vector<float> ow(static_cast<size_t>(p0) * kOut, 0.0f);
    for (int c = 0; c < r0; ++c)
      for (int o = 0; o < kOut; ++o) ow[c * kOut + o] = __half2float(ro.w[c * kOut + o]);
    owo[slot] = put(ow.data(), ow.size() * 4);
    obo[slot] = put(ro.b.data(), kOut * 4);
  }
  // First conv rows: exported order image 3 | density | log_cov 3 | rgb 3 | rendered 3 | count
  // | 0 0 -> the C1 unit order (rows 8..15 = rendered 3, count, rgb 1, rgb 2, 0, 0).
  {
    constexpr int kPerm[16] = {0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 8, 9, 14, 15};
    std::vector<__half> pw(fw.size());
    for (int t = 0; t < 9; ++t)
      for (int k = 0; k < kInPad; ++k)
        std::memcpy(&pw[(static_cast<size_t>(t) * kInPad + k) * headk::kFirstCout],
                    &fw[(static_cast<size_t>(t) * kInPad + kPerm[k]) * headk::kFirstCout],
                    headk::kFirstCout * sizeof(__half));
    fw.swap(pw);
  }
  const size_t fwo = put(fw.data(), fw.size() * 2), fbo = put(fb.data(), fb.size() * 4);

  uint8_t* dev = nullptr;
  cudaError_t err = cudaMalloc(&dev, host.size());
  if (!err) err = cudaMemcpy(dev, host.data(), host.size(), cudaMemcpyHostToDevice);
  if (err) {
    if (dev) cudaFree(dev);
    delete impl;
    return err;
  }
  impl->blob = dev;
  for (int slot = 0; slot < (impl->has_aux24 ? 3 : 2); ++slot) {
    Net& net = slot == 2 ? impl->aux24 : impl->net[slot];
    for (int i = 1; i < kConvsPerNet; ++i) {
      net.c[i].w = reinterpret_cast<const __half*>(dev + wofs[slot][i]);
      net.c[i].b = reinterpret_cast<const float*>(dev + bofs[slot][i]);
      if (f8[slot][i]) {
        net.c[i].w8 = dev + w8ofs[slot][i];
        net.c[i].w_scale = reinterpret_cast<const float*>(dev + wsofs[slot][i]);
      }
    }
    net.out_w = reinterpret_cast<const float*>(dev + owo[slot]);
    net.out_b = reinterpret_cast<const float*>(dev + obo[slot]);
  }
  impl->first_w = reinterpret_cast<const __half*>(dev + fwo);
  impl->first_b = reinterpret_cast<const float*>(dev + fbo);
  out->impl = impl;
  // FP16 accumulation: GFUSED_HEAD_ACC16 if set (head_tuning.cuh), else default_acc16().
  const char* env = std::getenv("GFUSED_HEAD_ACC16");
  const int n_env = env ? static_cast<int>(std::strlen(env)) : 0;
  const int def = default_acc16();
  for (int k = 0; k < 2; ++k)
    for (int i = 1; i < kConvsPerNet; ++i) {
      const int pos = n_env == 1 ? 0 : k * 9 + i - 1;
      const int mode = !env ? def : pos < n_env ? env[pos] - '0' : 0;
      set_correction_head_knob(out, kHeadAcc16 + k * 10 + i, mode);
    }
  return cudaSuccess;
}

void free_correction_head_weights(CorrectionHeadWeights* w) {
  if (auto* impl = static_cast<WeightsImpl*>(w->impl)) {
    cudaFree(impl->blob);
    delete impl;
  }
  *w = CorrectionHeadWeights{};
}

// ---------------------------------------------------------------- workspace

cudaError_t alloc_correction_head_workspace(int H, int W, CorrectionHeadWorkspace* ws) {
  // Sized for a multiple-of-16 extent; run_correction_head can run any smaller logical
  // extent in it.
  if (H % 16 || W % 16 || H <= 0 || W <= 0) return cudaErrorInvalidValue;
  const size_t P = static_cast<size_t>(H) * W;
  // Sized for level widths <= 32 / 64 / 128 (checked at weight load).
  const size_t net_halves[] = {P * 32,       P * 32,       P / 4 * 64, P / 4 * 64,
                               P / 16 * 128, P / 16 * 128, P / 4 * 64, P / 4 * 32,
                               P / 16 * 64};
  size_t bytes = 0;
  for (int k = 0; k < 2; ++k)
    for (size_t h : net_halves) bytes = align16(bytes + h * 2);
  uint8_t* dev = nullptr;
  cudaError_t e = cudaMalloc(&dev, bytes);
  if (e) return e;
  auto* impl = new WorkspaceImpl();
  impl->blob = dev;
  size_t off = 0;
  for (int k = 0; k < 2; ++k) {
    NetBufs& b = impl->nb[k];
    __half** slots[] = {&b.t0, &b.s0, &b.t1, &b.s1, &b.tb, &b.bot, &b.m0, &b.p0, &b.p1};
    for (int i = 0; i < 9; ++i) {
      *slots[i] = reinterpret_cast<__half*>(dev + off);
      off = align16(off + net_halves[i] * 2);
    }
  }
  if ((e = cudaStreamCreateWithFlags(&impl->aux, cudaStreamNonBlocking)) ||
      (e = cudaEventCreateWithFlags(&impl->fork, cudaEventDisableTiming)) ||
      (e = cudaEventCreateWithFlags(&impl->join, cudaEventDisableTiming))) {
    CorrectionHeadWorkspace tmp;
    tmp.impl = impl;
    free_correction_head_workspace(&tmp);
    return e;
  }
  ws->H = H;
  ws->W = W;
  ws->impl = impl;
  return cudaSuccess;
}

void free_correction_head_workspace(CorrectionHeadWorkspace* ws) {
  if (auto* impl = static_cast<WorkspaceImpl*>(ws->impl)) {
    if (impl->join) cudaEventDestroy(impl->join);
    if (impl->fork) cudaEventDestroy(impl->fork);
    if (impl->aux) cudaStreamDestroy(impl->aux);
    cudaFree(impl->blob);
    delete impl;
  }
  *ws = CorrectionHeadWorkspace{};
}

// ---------------------------------------------------------------- schedule

cudaError_t run_correction_head(const CorrectionHeadWeights& w, const CorrectionHeadInputs& in,
                                CorrectionHeadWorkspace& ws, float* out, cudaStream_t s,
                                int H, int W) {
  const auto* Wt = static_cast<const WeightsImpl*>(w.impl);
  auto* S = static_cast<WorkspaceImpl*>(ws.impl);
  if (!Wt || !S) return cudaErrorInvalidValue;
  H = H ? H : ws.H;
  W = W ? W : ws.W;
  if (H <= 0 || W <= 0 || H > ws.H || W > ws.W) return cudaErrorInvalidValue;
  // A logical extent below the workspace runs every layer at that extent (partial edge
  // tiles); decoder levels with an odd extent resize their up2x (InMode::kUpCatRs).
  const bool odd = H % 4 || W % 4;
  Sched c = schedule(*Wt, H, W);
  if (odd) {  // odd levels: generic layers with pooled skips (floor pooling)
    c = Sched{};
    c.small = true;
    c.pool = 3;
  }
  // the lean full-resolution kernels take whole 16-wide column tiles
  if (W % 16)
    for (int k = 0; k < 2; ++k) c.lean_enc0b[k] = c.lean_out[k] = false;
  cudaError_t e;
  if ((e = run_first(*Wt, in, *S, H, W, c.small, s))) return e;
  const bool fork = c.small;
  // small: the aux net runs on the side stream, forked after the shared first conv
  cudaStream_t sa = fork ? S->aux : s;
  if (fork) {
    if ((e = cudaEventRecord(S->fork, s))) return e;
    if ((e = cudaStreamWaitEvent(sa, S->fork, 0))) return e;
  }
  // one net's layers enc0.b .. dec1.a
  auto net_layers = [&](int k, cudaStream_t st) -> cudaError_t {
    cudaError_t r;
    if (c.lean_enc0b[k] && (r = run_pair(*Wt, *S, H, W, c.pool & 1, k, st))) return r;
    for (int i = c.enc0b_done(k) ? 2 : 1; i < kConvsPerNet - 1; ++i)
      if ((r = run_layer(*Wt, k, i, S->nb[k], H, W, out, false, st, c.pool))) return r;
    return cudaSuccess;
  };
  // dec1.b + out: the aux net writes out = aux + b_aux, the main net accumulates
  auto out_layer = [&](int k, cudaStream_t st) -> cudaError_t {
    const bool acc = k == 0;
    return c.lean_out[k] ? run_outk(*Wt, *S, H, W, out, k, acc, st)
                         : run_layer(*Wt, k, kConvsPerNet - 1, S->nb[k], H, W, out, acc, st);
  };
  if ((e = net_layers(1, sa)) || (e = out_layer(1, sa))) return e;
  if (fork && (e = cudaEventRecord(S->join, sa))) return e;
  if ((e = net_layers(0, s))) return e;
  if (fork && (e = cudaStreamWaitEvent(s, S->join, 0))) return e;
  return out_layer(0, s);
}

// ---------------------------------------------------------------- tuning hooks

bool set_correction_head_knob(CorrectionHeadWeights* w, int key, int value) {
  auto* impl = static_cast<WeightsImpl*>(w->impl);
  if (!impl) return false;
  if (key == kHeadSchedule) {
    impl->schedule = value;
    return value >= -1 && value <= 1;
  }
  if (key == kHeadAuxNative) {
    impl->aux_native = value;
    return value >= -1 && value <= 1;
  }
  if (key >= kHeadAcc16 && key < kHeadAcc16 + 20) {
    const int k = (key - kHeadAcc16) / kConvsPerNet, i = (key - kHeadAcc16) % kConvsPerNet;
    if (i == 0 || value < 0 || value > 2) return false;
    impl->net[k].c[i].acc16 = value;
    if (k == 1) impl->aux24.c[i].acc16 = value;
    return true;
  }
  const int idx = key - kHeadLayerCfg, k = idx / kConvsPerNet, i = idx % kConvsPerNet;
  if (idx < 0 || k > 1 || i == 0) return false;
  if (value < 0) {
    impl->cfg[k][i] = -1;
    return true;
  }
  const Net& n = net_of(*impl, k);
  NetBufs dummy{};
  const LayerSpec sp = layer_spec(n, dummy, i, 64, 64);
  const bool f8 = n.c[i].w8 != nullptr;
  if (!cfg_valid(value, sp.mode, sp.c0, sp.c1, n.c[i].cout, i == 9, f8, true)) return false;
  impl->cfg[k][i] = value;
  return true;
}

float time_correction_head_layer(const CorrectionHeadWeights& w, CorrectionHeadWorkspace& ws,
                                 const CorrectionHeadInputs& in, int net, int layer, int iters,
                                 cudaStream_t stream) {
  const auto* Wt = static_cast<const WeightsImpl*>(w.impl);
  auto* S = static_cast<WorkspaceImpl*>(ws.impl);
  float* out = reinterpret_cast<float*>(S->nb[1].s0);  // scratch, large enough for 8 x H x W
  cudaGetLastError();  // clear a stale error from an earlier failed config
  const int H = ws.H, W = ws.W;
  const Sched c = schedule(*Wt, H, W);
  auto launch = [&]() {
    if (layer == 0) return run_first(*Wt, in, *S, H, W, c.small, stream);
    if (layer == 1 && c.lean_enc0b[net]) return run_pair(*Wt, *S, H, W, c.pool & 1, net, stream);
    if (layer == 9 && c.lean_out[net]) return run_outk(*Wt, *S, H, W, out, net, false, stream);
    return run_layer(*Wt, net, layer, S->nb[net], H, W, out, false, stream, c.pool);
  };
  if (launch() != cudaSuccess) {
    cudaGetLastError();
    return -1.0f;
  }
  cudaEvent_t a, b;
  cudaEventCreate(&a);
  cudaEventCreate(&b);
  float best = 1e30f;
  for (int rep = 0; rep < 5; ++rep) {
    cudaEventRecord(a, stream);
    for (int it = 0; it < iters; ++it) launch();
    cudaEventRecord(b, stream);
    cudaEventSynchronize(b);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, a, b);
    best = ms / iters < best ? ms / iters : best;
  }
  cudaEventDestroy(a);
  cudaEventDestroy(b);
  return best;
}

}  // namespace gfused
