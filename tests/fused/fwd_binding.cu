// Test-only PyTorch binding for the torch-free fused forward map.
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "../../cpp_inference/fused/forward_map.cuh"

struct Runner {
  gfused::ForwardMapWeights w{};
  gfused::ForwardMapWorkspace ws{};
  explicit Runner(const std::string& path) {
    std::string err;
    TORCH_CHECK(gfused::load_forward_map_weights(path, &w, &err) == cudaSuccess, err);
  }
  ~Runner() {
    gfused::free_forward_map_workspace(&ws);
    gfused::free_forward_map_weights(&w);
  }
  // Schedule overrides (gfused::kFwd* in forward_map.cuh); reallocates the workspace on the
  // next run.
  void set_flags(int64_t flags) {
    gfused::free_forward_map_workspace(&ws);
    ws.flags = static_cast<unsigned>(flags);
  }
  // FP16-accumulate experiment (forward_map.cuh set_forward_map_acc16).
  bool set_acc16(int64_t conv, int64_t mode) {
    return gfused::set_forward_map_acc16(&w, static_cast<int>(conv), static_cast<int>(mode));
  }
  std::vector<torch::Tensor> run(torch::Tensor image_hwc3) {
    TORCH_CHECK(image_hwc3.is_cuda() && image_hwc3.scalar_type() == torch::kFloat &&
                image_hwc3.is_contiguous() && image_hwc3.size(2) == 3);
    const int H = static_cast<int>(image_hwc3.size(0)), W = static_cast<int>(image_hwc3.size(1));
    if (ws.H != H || ws.W != W) {
      gfused::free_forward_map_workspace(&ws);
      C10_CUDA_CHECK(gfused::alloc_forward_map_workspace(H, W, &ws));
    }
    auto opt = image_hwc3.options();
    auto density = torch::empty({H, W}, opt);
    auto log_cov = torch::empty({3, H, W}, opt);
    auto rgb = torch::empty({3, H, W}, opt);
    auto rate = torch::empty({1}, opt);
    gfused::ForwardMapOutputs o{density.data_ptr<float>(), log_cov.data_ptr<float>(),
                                rgb.data_ptr<float>(), rate.data_ptr<float>()};
    C10_CUDA_CHECK(gfused::run_forward_map(w, image_hwc3.data_ptr<float>(), ws, o,
                                           at::cuda::getCurrentCUDAStream()));
    return {density, log_cov, rgb, rate};
  }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  pybind11::class_<Runner>(m, "ForwardMap", pybind11::module_local())
      .def(pybind11::init<const std::string&>())
      .def("run", &Runner::run)
      .def("set_flags", &Runner::set_flags)
      .def("set_acc16", &Runner::set_acc16);
}
