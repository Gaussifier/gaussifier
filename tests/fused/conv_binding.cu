// Test-only PyTorch binding for the fused conv kernel. The runtime itself is torch-free.
//
//   conv_v2(...)      launch_conv2 with tile config `cfg` (index into GF_CFGS below) and an
//                     optional fused 1x1 projection segment (pmode < 0: none)
//   conv_v2_f8(...)   FP8 (e4m3) main segment, config index into GF_F8_CFGS
//   configs(), f8_configs()   the config tables as (BK, TH, BN, WM, WN) rows
//   conv_v2_band(...)  conv_v2 (no projection) restricted to output rows [row0, row0 + rows),
//                     written into an existing `out`; returns false if the launcher rejects
//                     the band
//   smem(mode, cfg, one_chunk)  dynamic shared memory of a config (FP16, no projection)
//   conv_block(...), conv_block_band(...)  fused residual block (conv_block.cuh)
//   Configs 20-31: native widths (BK an odd multiple of 8: 24 / 8; BN 24 / 48 / 56).
//   pmode: -1 none, 0 plain, 1 pool, 2 upcat, 4 image (pin0 = FP32 H x W x 3), 6 upcat-resize;
//   mode 4 = kImageF32 main input (in0 = FP32 H x W x 3, c0 = 16), mode 6 = kUpCatRs (in0 at
//   floor(H/2) x floor(W/2), any H, W)
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "../../cpp_inference/fused/conv_block.cuh"
#include "../../cpp_inference/fused/conv_fused.cuh"

using gfused::InMode;

// Like C10_CUDA_CHECK, but first clears the runtime's last-error state: tests probe configs
// that cannot launch (e.g. over the shared-memory limit), and a stale error would otherwise
// surface in the next unrelated launch check of the process.
static void check_launch(cudaError_t e) {
  if (e != cudaSuccess) {
    (void)cudaGetLastError();
    C10_CUDA_CHECK(e);
  }
}

// (index, BK, TH, BN, WM, WN)
#define GF_CFGS(X)        \
  X(0, 16, 8, 32, 4, 1)   \
  X(1, 32, 8, 32, 4, 1)   \
  X(2, 16, 8, 64, 4, 2)   \
  X(3, 32, 8, 64, 4, 2)   \
  X(4, 32, 16, 32, 8, 1)  \
  X(5, 16, 16, 32, 8, 1)  \
  X(6, 32, 4, 64, 2, 2)   \
  X(7, 16, 4, 64, 2, 2)   \
  X(8, 32, 8, 64, 8, 1)   \
  X(9, 16, 8, 128, 4, 2)  \
  X(10, 16, 16, 32, 4, 1) \
  X(11, 16, 16, 64, 8, 2) \
  X(12, 32, 8, 32, 8, 1)  \
  X(13, 16, 8, 32, 2, 1)  \
  X(14, 32, 4, 32, 4, 1)  \
  X(15, 16, 16, 16, 8, 1) \
  X(16, 16, 8, 16, 4, 1)  \
  X(17, 16, 16, 48, 8, 1) \
  X(18, 16, 8, 48, 4, 1)  \
  X(19, 16, 8, 48, 8, 1)   \
  X(20, 24, 8, 24, 4, 1)   \
  X(21, 24, 16, 24, 8, 1)  \
  X(22, 24, 8, 48, 4, 1)   \
  X(23, 24, 16, 48, 8, 1)  \
  X(24, 24, 16, 24, 4, 1)  \
  X(25, 24, 8, 24, 8, 1)   \
  X(26, 8, 16, 24, 8, 1)   \
  X(27, 8, 8, 24, 4, 1)    \
  X(28, 24, 4, 24, 4, 1)   \
  X(29, 16, 8, 56, 4, 1)   \
  X(30, 24, 12, 24, 6, 1)  \
  X(31, 24, 8, 48, 4, 2)   \
  X(32, 16, 16, 64, 4, 2)

#define GF_F8_CFGS(X)    \
  X(0, 32, 16, 32, 8, 1) \
  X(1, 32, 8, 64, 4, 2)  \
  X(2, 32, 16, 64, 8, 2) \
  X(3, 64, 8, 32, 4, 1)  \
  X(4, 32, 16, 32, 4, 1) \
  X(5, 32, 8, 32, 4, 1)  \
  X(6, 64, 8, 64, 4, 2)  \
  X(7, 32, 16, 64, 4, 2)

template <InMode M, int KS, InMode PM, bool F8 = false>
static void dispatch_v2(const gfused::ConvParams& p, const gfused::ProjParams& q,
                        cudaStream_t s, int cfg) {
  const int cin = p.c0 + p.c1;
  const int cinp = q.c0 + q.c1;
#define GF_CASE(I, BK, TH, BN, WM, WN)                                                       \
  if (cfg == I) {                                                                            \
    TORCH_CHECK(cin % BK == 0 && p.c0 % BK == 0 && cinp % BK == 0 && q.c0 % BK == 0,         \
                "cfg ", I, ": channels not a multiple of BK ", BK);                          \
    TORCH_CHECK(p.cout % BN == 0, "cfg ", I, ": cout not a multiple of BN ", BN);            \
    check_launch((gfused::launch_conv2<M, KS, PM, BK, TH, BN, WM, WN,                        \
                                       gfused::DefaultEpilogue, F8>(p, q, {}, s)));          \
    return;                                                                                  \
  }
  if constexpr (F8) {
    GF_F8_CFGS(GF_CASE)
  } else if constexpr (M != InMode::kImageF32 && PM != InMode::kImageF32) {
    GF_CFGS(GF_CASE)
  } else {  // image input needs BK 16
    if (cfg == 5) {
      check_launch((gfused::launch_conv2<M, KS, PM, 16, 16, 32, 8, 1>(p, q, {}, s)));
      return;
    }
    if (cfg == 10) {
      check_launch((gfused::launch_conv2<M, KS, PM, 16, 16, 32, 4, 1>(p, q, {}, s)));
      return;
    }
  }
#undef GF_CASE
  TORCH_CHECK(false, "unknown cfg ", cfg);
}

template <InMode M, int KS, InMode PM, bool F8 = false>
static cudaError_t dispatch_v2_raw(const gfused::ConvParams& p, const gfused::ProjParams& q,
                                   cudaStream_t s, int cfg) {
#define GF_CASE(I, BK, TH, BN, WM, WN)                                                     \
  if (cfg == I) return gfused::launch_conv2<M, KS, PM, BK, TH, BN, WM, WN>(p, q, {}, s);
  GF_CFGS(GF_CASE)
#undef GF_CASE
  return cudaErrorInvalidValue;
}

static int g_single_stage = 0;  // ConvParams::single_stage for every test launch
void set_single_stage(bool on) { g_single_stage = on ? 1 : 0; }
static int g_acc16 = 0;  // ConvParams::acc16 for every test launch (FP16 operand configs)
void set_acc16(int64_t mode) { g_acc16 = static_cast<int>(mode); }

static gfused::ConvParams make_params(const torch::Tensor& in0,
                                      const c10::optional<torch::Tensor>& in1,
                                      const torch::Tensor& weight,
                                      const c10::optional<torch::Tensor>& bias,
                                      const c10::optional<torch::Tensor>& residual, bool silu,
                                      int64_t H, int64_t W, torch::Tensor& out) {
  TORCH_CHECK(in0.is_cuda() && in0.is_contiguous());
  const bool image = in0.scalar_type() == torch::kFloat;  // kImageF32: H x W x 3
  const int64_t cout = weight.size(2);
  out = torch::empty({H, W, cout}, in0.options().dtype(torch::kHalf));
  gfused::ConvParams p{};
  p.in0 = reinterpret_cast<const __half*>(in0.data_ptr());
  p.c0 = image ? 16 : static_cast<int>(in0.size(2));
  p.in1 = in1 ? reinterpret_cast<const __half*>(in1->data_ptr<at::Half>()) : nullptr;
  p.c1 = in1 ? static_cast<int>(in1->size(2)) : 0;
  p.H = static_cast<int>(H);
  p.W = static_cast<int>(W);
  p.weight = reinterpret_cast<const __half*>(weight.data_ptr<at::Half>());
  p.bias = bias ? bias->data_ptr<float>() : nullptr;
  p.residual =
      residual ? reinterpret_cast<const __half*>(residual->data_ptr<at::Half>()) : nullptr;
  p.out = reinterpret_cast<__half*>(out.data_ptr<at::Half>());
  p.cout = static_cast<int>(cout);
  p.silu = silu ? 1 : 0;
  p.single_stage = g_single_stage;
  p.acc16 = g_acc16;
  return p;
}

static gfused::ProjParams make_proj(const c10::optional<torch::Tensor>& pin0,
                                    const c10::optional<torch::Tensor>& pin1,
                                    const c10::optional<torch::Tensor>& pw, int64_t pmode,
                                    int64_t mode, int64_t ks) {
  gfused::ProjParams q{};
  if (pmode >= 0) {
    TORCH_CHECK(pin0 && pw && mode == 0 && ks == 3, "proj fusion is tested on 3x3 kPlain only");
    q.in0 = reinterpret_cast<const __half*>(pin0->data_ptr());
    q.c0 = pmode == 4 ? 16 : static_cast<int>(pin0->size(2));
    q.in1 = pin1 ? reinterpret_cast<const __half*>(pin1->data_ptr<at::Half>()) : nullptr;
    q.c1 = pin1 ? static_cast<int>(pin1->size(2)) : 0;
    q.weight = reinterpret_cast<const __half*>(pw->data_ptr<at::Half>());
  }
  return q;
}

torch::Tensor conv_v2(torch::Tensor in0, c10::optional<torch::Tensor> in1, torch::Tensor weight,
                      c10::optional<torch::Tensor> bias, c10::optional<torch::Tensor> residual,
                      int64_t mode, int64_t ks, bool silu, int64_t H, int64_t W, int64_t cfg,
                      c10::optional<torch::Tensor> pin0, c10::optional<torch::Tensor> pin1,
                      c10::optional<torch::Tensor> pw, int64_t pmode) {
  torch::Tensor out;
  auto p = make_params(in0, in1, weight, bias, residual, silu, H, W, out);
  gfused::ProjParams q = make_proj(pin0, pin1, pw, pmode, mode, ks);
  auto s = at::cuda::getCurrentCUDAStream();
  const int c = static_cast<int>(cfg);
  if (pmode == 0) {
    dispatch_v2<InMode::kPlain, 3, InMode::kPlain>(p, q, s, c);
  } else if (pmode == 1) {
    dispatch_v2<InMode::kPlain, 3, InMode::kPool>(p, q, s, c);
  } else if (pmode == 2) {
    dispatch_v2<InMode::kPlain, 3, InMode::kUpCat>(p, q, s, c);
  } else if (pmode == 4) {
    dispatch_v2<InMode::kPlain, 3, InMode::kImageF32>(p, q, s, c);
  } else if (pmode == 6) {
    dispatch_v2<InMode::kPlain, 3, InMode::kUpCatRs>(p, q, s, c);
  } else if (ks == 3) {
    if (mode == 4) dispatch_v2<InMode::kImageF32, 3, InMode::kNoProj>(p, q, s, c);
    else if (mode == 0) dispatch_v2<InMode::kPlain, 3, InMode::kNoProj>(p, q, s, c);
    else if (mode == 1) dispatch_v2<InMode::kPool, 3, InMode::kNoProj>(p, q, s, c);
    else if (mode == 6) dispatch_v2<InMode::kUpCatRs, 3, InMode::kNoProj>(p, q, s, c);
    else dispatch_v2<InMode::kUpCat, 3, InMode::kNoProj>(p, q, s, c);
  } else {
    if (mode == 0) dispatch_v2<InMode::kPlain, 1, InMode::kNoProj>(p, q, s, c);
    else if (mode == 1) dispatch_v2<InMode::kPool, 1, InMode::kNoProj>(p, q, s, c);
    else if (mode == 6) dispatch_v2<InMode::kUpCatRs, 1, InMode::kNoProj>(p, q, s, c);
    else dispatch_v2<InMode::kUpCat, 1, InMode::kNoProj>(p, q, s, c);
  }
  return out;
}

bool conv_v2_band(torch::Tensor in0, c10::optional<torch::Tensor> in1, torch::Tensor weight,
                  c10::optional<torch::Tensor> bias, c10::optional<torch::Tensor> residual,
                  int64_t mode, int64_t ks, bool silu, int64_t H, int64_t W, int64_t cfg,
                  int64_t row0, int64_t rows, torch::Tensor out) {
  torch::Tensor tmp;
  auto p = make_params(in0, in1, weight, bias, residual, silu, H, W, tmp);
  TORCH_CHECK(out.sizes() == tmp.sizes() && out.scalar_type() == torch::kHalf);
  p.out = reinterpret_cast<__half*>(out.data_ptr<at::Half>());
  p.row0 = static_cast<int>(row0);
  p.rows = static_cast<int>(rows);
  auto s = at::cuda::getCurrentCUDAStream();
  const int c = static_cast<int>(cfg);
  const gfused::ProjParams q{};
  cudaError_t e;
  if (ks == 3) {
    if (mode == 0) e = dispatch_v2_raw<InMode::kPlain, 3, InMode::kNoProj>(p, q, s, c);
    else if (mode == 1) e = dispatch_v2_raw<InMode::kPool, 3, InMode::kNoProj>(p, q, s, c);
    else if (mode == 6) e = dispatch_v2_raw<InMode::kUpCatRs, 3, InMode::kNoProj>(p, q, s, c);
    else e = dispatch_v2_raw<InMode::kUpCat, 3, InMode::kNoProj>(p, q, s, c);
  } else {
    if (mode == 0) e = dispatch_v2_raw<InMode::kPlain, 1, InMode::kNoProj>(p, q, s, c);
    else if (mode == 1) e = dispatch_v2_raw<InMode::kPool, 1, InMode::kNoProj>(p, q, s, c);
    else e = dispatch_v2_raw<InMode::kUpCat, 1, InMode::kNoProj>(p, q, s, c);
  }
  if (e == cudaErrorInvalidValue) {
    (void)cudaGetLastError();
    return false;
  }
  check_launch(e);
  return true;
}

template <InMode M>
static int64_t smem_of(int cfg, bool one_chunk) {
#define GF_CASE(I, BK, TH, BN, WM, WN)                                                    \
  if (cfg == I) {                                                                         \
    using C = gfused::ConvV2Cfg<M, 3, InMode::kNoProj, BK, TH, BN, WM, WN>;               \
    return one_chunk ? C::kSmem1 : C::kSmem;                                              \
  }
  GF_CFGS(GF_CASE)
#undef GF_CASE
  return -1;
}

int64_t smem(int64_t mode, int64_t cfg, bool one_chunk) {
  const int c = static_cast<int>(cfg);
  if (mode == 0) return smem_of<InMode::kPlain>(c, one_chunk);
  if (mode == 1) return smem_of<InMode::kPool>(c, one_chunk);
  if (mode == 6) return smem_of<InMode::kUpCatRs>(c, false);
  return smem_of<InMode::kUpCat>(c, false);
}

// FP8 main segment. w8: e4m3 bytes (uint8) [taps][cout][cin]; w_scale [cout]; weight is only
// used for its shape ([taps][cin][cout]).
torch::Tensor conv_v2_f8(torch::Tensor in0, c10::optional<torch::Tensor> in1,
                         torch::Tensor weight, torch::Tensor w8, torch::Tensor w_scale,
                         double a_scale, c10::optional<torch::Tensor> bias,
                         c10::optional<torch::Tensor> residual, int64_t mode, int64_t ks,
                         bool silu, int64_t H, int64_t W, int64_t cfg,
                         c10::optional<torch::Tensor> pin0, c10::optional<torch::Tensor> pin1,
                         c10::optional<torch::Tensor> pw, int64_t pmode) {
  torch::Tensor out;
  auto p = make_params(in0, in1, weight, bias, residual, silu, H, W, out);
  TORCH_CHECK(w8.scalar_type() == torch::kUInt8 && w8.is_contiguous());
  p.weight8 = w8.data_ptr<uint8_t>();
  p.w_scale = w_scale.data_ptr<float>();
  p.a_scale = static_cast<float>(a_scale);
  p.acc16 = 0;  // FP8 operands accumulate in FP32
  gfused::ProjParams q = make_proj(pin0, pin1, pw, pmode, mode, ks);
  auto s = at::cuda::getCurrentCUDAStream();
  const int c = static_cast<int>(cfg);
  if (pmode == 0) {
    dispatch_v2<InMode::kPlain, 3, InMode::kPlain, true>(p, q, s, c);
  } else if (pmode == 1) {
    dispatch_v2<InMode::kPlain, 3, InMode::kPool, true>(p, q, s, c);
  } else if (pmode == 2) {
    dispatch_v2<InMode::kPlain, 3, InMode::kUpCat, true>(p, q, s, c);
  } else if (pmode == 6) {
    dispatch_v2<InMode::kPlain, 3, InMode::kUpCatRs, true>(p, q, s, c);
  } else {
    TORCH_CHECK(ks == 3, "FP8 test path: 3x3 only");
    if (mode == 0) dispatch_v2<InMode::kPlain, 3, InMode::kNoProj, true>(p, q, s, c);
    else if (mode == 1) dispatch_v2<InMode::kPool, 3, InMode::kNoProj, true>(p, q, s, c);
    else if (mode == 6) dispatch_v2<InMode::kUpCatRs, 3, InMode::kNoProj, true>(p, q, s, c);
    else dispatch_v2<InMode::kUpCat, 3, InMode::kNoProj, true>(p, q, s, c);
  }
  return out;
}

// Fused residual block (conv_block.cuh). mode: 0 plain, 1 pool, 2 upcat, 4 image (x0 FP32).
// cfg: (TH, NW, BK1, BK2) index into GF_BLOCK_CFGS.
#define GF_BLOCK_CFGS(X) X(0, 8, 4, 16, 16) X(1, 8, 8, 16, 16) X(2, 16, 8, 16, 16) X(3, 8, 4, 16, 32)

template <InMode M, int CM>
static void dispatch_block(const gfused::BlockParams& bp, cudaStream_t s, int cfg) {
#define GF_CASE(I, TH, NW, BK1, BK2)                                                        \
  if (cfg == I) {                                                                           \
    check_launch((gfused::launch_block<M, TH, NW, BK1, BK2, CM, CM>(bp, {}, s)));           \
    return;                                                                                 \
  }
  GF_BLOCK_CFGS(GF_CASE)
#undef GF_CASE
  TORCH_CHECK(false, "unknown block cfg ", cfg);
}

torch::Tensor conv_block_band(torch::Tensor x0, c10::optional<torch::Tensor> x1,
                              torch::Tensor w1, torch::Tensor b1, torch::Tensor w2,
                              torch::Tensor b2, torch::Tensor wp, int64_t mode, int64_t H,
                              int64_t W, int64_t cfg, int64_t row0, int64_t rows,
                              c10::optional<torch::Tensor> out_opt) {
  const int64_t cm = w1.size(2);
  auto out = out_opt ? *out_opt : torch::empty({H, W, cm}, x0.options().dtype(torch::kHalf));
  TORCH_CHECK(out.size(0) == H && out.size(1) == W && out.size(2) == cm);
  gfused::BlockParams bp{};
  bp.q.in0 = reinterpret_cast<const __half*>(x0.data_ptr());
  bp.q.c0 = mode == 4 ? 16 : static_cast<int>(x0.size(2));
  bp.q.in1 = x1 ? reinterpret_cast<const __half*>(x1->data_ptr<at::Half>()) : nullptr;
  bp.q.c1 = x1 ? static_cast<int>(x1->size(2)) : 0;
  bp.q.weight = reinterpret_cast<const __half*>(wp.data_ptr<at::Half>());
  bp.w1 = reinterpret_cast<const __half*>(w1.data_ptr<at::Half>());
  bp.b1 = b1.data_ptr<float>();
  bp.p.H = static_cast<int>(H);
  bp.p.W = static_cast<int>(W);
  bp.p.weight = reinterpret_cast<const __half*>(w2.data_ptr<at::Half>());
  bp.p.bias = b2.data_ptr<float>();
  bp.p.out = reinterpret_cast<__half*>(out.data_ptr<at::Half>());
  bp.p.cout = static_cast<int>(cm);
  bp.p.silu = 1;
  bp.p.row0 = static_cast<int>(row0);
  bp.p.rows = static_cast<int>(rows);
  auto s = at::cuda::getCurrentCUDAStream();
  const int c = static_cast<int>(cfg);
  if (mode == 4 && cm == 32) dispatch_block<InMode::kImageF32, 32>(bp, s, c);
  else if (mode == 0 && cm == 32) dispatch_block<InMode::kPlain, 32>(bp, s, c);
  else if (mode == 1 && cm == 64) dispatch_block<InMode::kPool, 64>(bp, s, c);
  else if (mode == 2 && cm == 32) dispatch_block<InMode::kUpCat, 32>(bp, s, c);
  else if (mode == 2 && cm == 64) dispatch_block<InMode::kUpCat, 64>(bp, s, c);
  else TORCH_CHECK(false, "block mode/cm not instantiated");
  return out;
}

torch::Tensor conv_block(torch::Tensor x0, c10::optional<torch::Tensor> x1, torch::Tensor w1,
                         torch::Tensor b1, torch::Tensor w2, torch::Tensor b2, torch::Tensor wp,
                         int64_t mode, int64_t H, int64_t W, int64_t cfg) {
  return conv_block_band(x0, x1, w1, b1, w2, b2, wp, mode, H, W, cfg, 0, 0, c10::nullopt);
}

std::vector<std::vector<int64_t>> f8_configs() {
  std::vector<std::vector<int64_t>> v;
#define GF_ROW(I, BK, TH, BN, WM, WN) v.push_back({BK, TH, BN, WM, WN});
  GF_F8_CFGS(GF_ROW)
#undef GF_ROW
  return v;
}

std::vector<std::vector<int64_t>> configs() {
  std::vector<std::vector<int64_t>> v;
#define GF_ROW(I, BK, TH, BN, WM, WN) v.push_back({BK, TH, BN, WM, WN});
  GF_CFGS(GF_ROW)
#undef GF_ROW
  return v;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("conv_v2", &conv_v2);
  m.def("conv_v2_f8", &conv_v2_f8);
  m.def("conv_block", &conv_block);
  m.def("conv_block_band", &conv_block_band);
  m.def("conv_v2_band", &conv_v2_band);
  m.def("smem", &smem);
  m.def("set_single_stage", &set_single_stage);
  m.def("set_acc16", &set_acc16);
  m.def("configs", &configs);
  m.def("f8_configs", &f8_configs);
}
