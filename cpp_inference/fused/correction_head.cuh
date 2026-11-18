// Torch-free fused correction head (frozen-GroupNorm v0.11.1 head architecture).
//
// Interface between the head implementation (correction_head.cu) and the end-to-end pipeline
// (pipeline.cu); the weight and workspace structs are opaque to the pipeline.
//
// Architecture: CorrectionHead = main CorrectionUNet(hidden 32, depth 2)
//                              + aux  CorrectionUNet(hidden 24, depth 2), outputs summed.
//   CorrectionUNet: enc[i] = double_conv(conv3x3 -> GN -> SiLU, x2), maxpool 2x2 between
//   levels, bottleneck double_conv, decoder: bilinear up2x (align_corners=False),
//   concat([up, skip]), double_conv; final 1x1 conv to 8 channels (with bias).
//   Every GroupNorm is frozen (calibrated mean/var) and folded into its conv.
//
// Input (17 channels, in this order), all at H x W:
//   image (3) | density (1) | log_cov map (3) | rgb map (3) | rendered (3) |
//   diff = image - rendered (3) | hard point count (1)
// Output: 8 channels per pixel = log-cov delta (3), rgb delta (3), xy delta (2).
// The K-loop bilinearly samples the output at point positions (align_corners=False).

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <string>

namespace gfused {

struct CorrectionHeadWeights {
  void* impl = nullptr;  // owned by the implementation
};

struct CorrectionHeadWorkspace {
  int H = 0, W = 0;
  void* impl = nullptr;  // owned by the implementation
};

struct CorrectionHeadInputs {
  const float* image_hwc3;  // H x W x 3, float32 [0, 1] (same buffer the forward map reads)
  const float* density;     // H x W (forward-map density)
  const float* log_cov;     // 3 x H x W planar (forward-map log-covariance map)
  const float* rgb;         // 3 x H x W planar (forward-map sigmoid rgb map)
  const float* rendered;    // 3 x H x W planar (current K-loop render, clamped to [0, 1])
  const float* count;       // H x W hard point-count map (points per pixel)
  // Packed input (contract C1; optional, nullptr = gather from the FP32 buffers above): the
  // head input as two FP16 16-byte unit planes [2][H][W][8]; unit u of pixel i at
  // packed16 + (u * H * W + i) * 8.
  //   unit 0 (static per image): image r, g, b, density, log_cov 0, 1, 2, rgb 0
  //   unit 1 (per K-loop state): rendered r, g, b, count, rgb 1, rgb 2, 0, 0
  // Values must equal __float2half of the FP32 buffers (the pipeline packs unit 0 once per
  // image and the renderer rewrites unit 1 for every state; see kloop_render.cuh).
  // dynamic_in_packed: unit 1 halves 0..3 are valid for this call (what the pipeline passes);
  // false = they are placeholders and rendered / count are read from the FP32 pointers.
  // (head_in_conv_kernel has a separate cp.async-only instantiation for the packed + dynamic
  // case.)
  const __half* packed16 = nullptr;
  bool dynamic_in_packed = false;
};

cudaError_t load_correction_head_weights(const std::string& path, CorrectionHeadWeights* out,
                                         std::string* error);
void free_correction_head_weights(CorrectionHeadWeights* w);

cudaError_t alloc_correction_head_workspace(int H, int W, CorrectionHeadWorkspace* ws);
void free_correction_head_workspace(CorrectionHeadWorkspace* ws);

// Writes the 8-channel correction map, 8 x H x W planar float32, to `out`.
// Must be CUDA-graph capturable (no host sync, no allocation, no host->device copies).
// The workspace is aligned to 16. H, W (default: the workspace's) may be any smaller logical
// extent; inputs and output then use that extent. When H or W is not a multiple of 4 a
// decoder level has an odd extent; its up2x is resized to the skip's size inside the conv's
// input staging (InMode::kUpCatRs), as the reference resizes it.
cudaError_t run_correction_head(const CorrectionHeadWeights& w, const CorrectionHeadInputs& in,
                                CorrectionHeadWorkspace& ws, float* out, cudaStream_t stream,
                                int H = 0, int W = 0);

}  // namespace gfused
