// Test-only PyTorch binding for the torch-free fused correction head.
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "../../cpp_inference/fused/correction_head.cuh"
#include "../../cpp_inference/fused/head_tuning.cuh"

static void check_f32(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == torch::kFloat && t.is_contiguous(), name,
              " must be a contiguous CUDA float32 tensor");
}

struct Runner {
  gfused::CorrectionHeadWeights w{};
  gfused::CorrectionHeadWorkspace ws{};
  explicit Runner(const std::string& path) {
    std::string err;
    TORCH_CHECK(gfused::load_correction_head_weights(path, &w, &err) == cudaSuccess, err);
  }
  ~Runner() {
    gfused::free_correction_head_workspace(&ws);
    gfused::free_correction_head_weights(&w);
  }
  // Allocates/reallocates the workspace for (H, W). Call before graph capture.
  void prepare(int64_t H, int64_t W) {
    if (ws.H != H || ws.W != W) {
      gfused::free_correction_head_workspace(&ws);
      C10_CUDA_CHECK(gfused::alloc_correction_head_workspace(static_cast<int>(H),
                                                             static_cast<int>(W), &ws));
    }
  }
  // image [H, W, 3]; density [H, W]; log_cov/rgb/rendered [3, H, W]; count [H, W];
  // out [8, H, W] (written in place; no allocation, so it is graph-capturable).
  static gfused::CorrectionHeadInputs inputs(const std::vector<torch::Tensor>& t) {
    TORCH_CHECK(t.size() == 6);
    for (auto& x : t) check_f32(x, "input");
    return {t[0].data_ptr<float>(), t[1].data_ptr<float>(), t[2].data_ptr<float>(),
            t[3].data_ptr<float>(), t[4].data_ptr<float>(), t[5].data_ptr<float>()};
  }
  bool set_knob(int64_t key, int64_t value) {
    return gfused::set_correction_head_knob(&w, static_cast<int>(key), static_cast<int>(value));
  }
  double time_layer(std::vector<torch::Tensor> ins, int64_t net, int64_t layer, int64_t iters) {
    return gfused::time_correction_head_layer(w, ws, inputs(ins), static_cast<int>(net),
                                              static_cast<int>(layer), static_cast<int>(iters),
                                              at::cuda::getCurrentCUDAStream());
  }
  void run(torch::Tensor image, torch::Tensor density, torch::Tensor log_cov, torch::Tensor rgb,
           torch::Tensor rendered, torch::Tensor count, torch::Tensor out) {
    for (auto* t : {&image, &density, &log_cov, &rgb, &rendered, &count, &out})
      check_f32(*t, "input");
    const int64_t H = image.size(0), W = image.size(1);
    TORCH_CHECK(out.size(0) == 8 && out.size(1) == H && out.size(2) == W);
    TORCH_CHECK(ws.H == H && ws.W == W, "call prepare(H, W) first");
    gfused::CorrectionHeadInputs in{image.data_ptr<float>(),   density.data_ptr<float>(),
                                    log_cov.data_ptr<float>(), rgb.data_ptr<float>(),
                                    rendered.data_ptr<float>(), count.data_ptr<float>()};
    C10_CUDA_CHECK(gfused::run_correction_head(w, in, ws, out.data_ptr<float>(),
                                               at::cuda::getCurrentCUDAStream()));
  }
};

// Packed input variant (contract C1): packed [2, H, W, 8] float16 unit planes; dynamic = unit
// 1 halves 0..3 valid (else rendered / count are read from the FP32 tensors).
static void run_packed(Runner& r, torch::Tensor image, torch::Tensor density,
                       torch::Tensor log_cov, torch::Tensor rgb, torch::Tensor rendered,
                       torch::Tensor count, torch::Tensor packed, bool dynamic,
                       torch::Tensor out) {
  TORCH_CHECK(packed.is_cuda() && packed.scalar_type() == torch::kHalf && packed.is_contiguous()
              && packed.dim() == 4 && packed.size(0) == 2 && packed.size(3) == 8);
  const int64_t H = image.size(0), W = image.size(1);
  TORCH_CHECK(packed.size(1) == H && packed.size(2) == W && r.ws.H == H && r.ws.W == W);
  gfused::CorrectionHeadInputs in = Runner::inputs({image, density, log_cov, rgb, rendered, count});
  in.packed16 = reinterpret_cast<const __half*>(packed.data_ptr<at::Half>());
  in.dynamic_in_packed = dynamic;
  check_f32(out, "out");
  TORCH_CHECK(out.size(0) == 8 && out.size(1) == H && out.size(2) == W);
  C10_CUDA_CHECK(gfused::run_correction_head(r.w, in, r.ws, out.data_ptr<float>(),
                                             at::cuda::getCurrentCUDAStream()));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  pybind11::class_<Runner>(m, "CorrectionHead")
      .def(pybind11::init<const std::string&>())
      .def("prepare", &Runner::prepare)
      .def("run", &Runner::run)
      .def("set_knob", &Runner::set_knob)
      .def("run_packed", &run_packed)
      .def("time_layer", &Runner::time_layer);
}
