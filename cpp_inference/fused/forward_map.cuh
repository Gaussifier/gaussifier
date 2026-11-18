// Torch-free fused forward map for the frozen-GroupNorm Gaussifier backbone.
//
// Topology (UNetBackbone2D, residual blocks, base 32, depth 4), every GroupNorm folded into
// the preceding conv's weights and bias:
//
//   block(X):  y1  = silu(conv3x3(X) + b1)
//              out = silu(conv3x3(y1) + b2 + proj1x1(X))
//   stem:  X = image (FP32 HWC3 read directly by the conv staging as 16 channels, 3 real)
//   down:  X = maxpool2x2(prev)                       -> 64, 128, 256 channels
//   up:    X = concat(bilinear_up2x(prev), skip)      -> 128, 64, 32 channels
//   heads: per-pixel 1x1 on p2 (32 ch): density logit, log-cov (3), rgb logits (3)
//   density = exp(l - max l) / (mean(exp(l - max l)) + eps); rate sum = sum exp(l)
//
// Upsampling and concatenation happen inside the conv kernel's input staging (conv_fused.cuh);
// pooling happens in the producing conv's epilogue (PoolEpilogue, epilogues.cuh), which
// writes skip[i] and maxpool2x2(skip[i]) side by side.
// From 1024 x 1024 pixels an FP16 up2 runs in 128-row bands (conv1 band b, then conv2 + heads
// band b - 1) so conv2 reads t_y1 from L2, and the stem and down0 run as single-kernel
// residual blocks (conv_block.cuh). Each block's 1x1 projection is an extra K segment of its
// conv2 (up blocks: only its skip half; the upsampled half runs at low resolution and conv2
// adds its up2x, see low_res_proj in forward_map.cu), and the heads run in the epilogue of
// up2.conv2, so only block outputs and conv1's output (t_y1) touch memory; p2 is never
// materialised.

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

#include "conv_fused.cuh"

namespace gfused {

// One conv layer's device parameters (weights [taps][cin][cout] half, bias fp32 or null).
struct ConvLayer {
  const __half* w = nullptr;
  const float* b = nullptr;
  int ks = 3, cin = 0, cout = 0;
  // FP8 precision plan (GFFM v2): e4m3 weights [taps][cout][cin], per-cout scales, static
  // activation scale. fp8 = false runs the FP16 weights above.
  const uint8_t* w8 = nullptr;
  const float* w_scale = nullptr;
  float a_scale = 0.0f;
  bool fp8 = false;
  // ConvParams::acc16 for this layer's FP16 launches (0 FP32 accumulation; 1 / 2 FP16, see
  // conv_fused.cuh). Ignored by FP8 layers and by the fused single-kernel blocks, which are
  // not used while a stem / down0 conv has it set.
  int acc16 = 0;
  // Up blocks' conv1 (upsample folding, InMode::kUpCatPh, upfold.h; GFUSED_FM_UPFOLD=1):
  // phase kernels [4][9][c0][cout] and k-pair packed upsampled-segment weights.
  const __half* wph = nullptr;
  const uint32_t* wedge = nullptr;
};

struct ResBlock {
  ConvLayer conv1, conv2, proj;
};

struct ForwardMapWeights {
  ResBlock stem, down[3], up[3];
  const __half* head_w = nullptr;  // [32][8]: density, logcov xx/xy/yy, rgb r/g/b, pad
  const float* head_b = nullptr;   // [8]
  void* device_blob = nullptr;     // owns everything above
  void* fold_blob = nullptr;       // up blocks' conv1 phase weights (wph / wedge)
};

// ForwardMapWorkspace::flags: schedule overrides for tests (0 = size-based default). None of
// them changes the arithmetic except kFwdFusedBlocks (the fused block keeps conv1's output in
// shared memory, so its accumulation order differs slightly).
constexpr unsigned kFwdFusedBlocks = 1;  // single-kernel stem / down0 at any size
constexpr unsigned kFwdBanded = 2;       // run up2 in 128-row bands at any size
constexpr unsigned kFwdUnbanded = 4;     // never band up2
constexpr unsigned kFwdNoF8Storage = 8;  // FP16 t_y1 even when conv2 is FP8 (see res_block)
// Up blocks project their upsampled input at full resolution (inside conv2) instead of at
// low resolution plus an upsampled add (default; see low_res_proj in forward_map.cu).
constexpr unsigned kFwdFullResProj = 16;

// Activations for one resolution. All NHWC half.
struct ForwardMapWorkspace {
  int H = 0, W = 0;
  unsigned flags = 0;
  __half* skip[3] = {};               // stem (32), down0 (64), down1 (128)
  __half* bottom = nullptr;           // down2 output (256) at H/8
  __half* up_out[2] = {};             // up0 (128) at H/4, up1 (64) at H/2
  __half* t_y1 = nullptr;             // block temporary (largest: H x W x 32)
  float* logits = nullptr;            // H x W
  float* reduce = nullptr;            // [0] max (as ordered int), [1] sum exp(l - max)
  void* device_blob = nullptr;
  // maxpool2x2(skip[i]) (32 @ H/2, 64 @ H/4, 128 @ H/8), written by the producer's
  // PoolEpilogue next to skip[i] and read by down[i] as a plain input.
  __half* pooled[3] = {};
};

struct ForwardMapOutputs {
  float* density;    // H x W; nullptr leaves logits in the workspace for cropping
  float* log_cov;    // 3 x H x W (planar)
  float* rgb;        // 3 x H x W (planar, sigmoid applied)
  float* rate_sum;   // [1] sum exp(logits) = predicted count before flooring/caps
};

// Weight blob: "GFFM" magic, u32 version, u32 n_tensors, then per tensor
// u32 dtype (0 = f16, 1 = f32, 2 = u8), u32 n_elems, payload (16-byte aligned).
// Tensor order is fixed by export_fused_weights.py and consumed by load_forward_map_weights.
// v1: FP16 network (37 tensors). v2: v1 + per 3x3 conv (stem..up2, conv1 then conv2)
// {meta f32 [is_fp8, act_scale], e4m3 u8 [9][cout][cin], w_scale f32 [cout]} (FP8 plan).
// A planned FP8 layer whose channel counts the FP8 kernel cannot take (multiples of 32)
// silently runs FP16.
cudaError_t load_forward_map_weights(const std::string& path, ForwardMapWeights* out,
                                     std::string* error);
cudaError_t load_forward_map_weights_from_memory(const void* data, size_t size,
                                                 ForwardMapWeights* out, std::string* error);
void free_forward_map_weights(ForwardMapWeights* w);

// FP16-accumulate setting per conv: index 0..13 = stem.conv1, stem.conv2, down0.conv1, ...,
// up2.conv2 (conv2 includes its fused projection); mode 0 / 1 / 2 as ConvParams::acc16; index
// -1 sets every conv. Returns false for an invalid index / mode. The loaders set
// default_acc16() (conv_fused.cuh), or GFUSED_FM_ACC16 when it is set: one digit for every
// conv, or 14 digits in conv index order ("0" turns it off).
bool set_forward_map_acc16(ForwardMapWeights* w, int conv, int mode);

cudaError_t alloc_forward_map_workspace(int H, int W, ForwardMapWorkspace* ws);
void free_forward_map_workspace(ForwardMapWorkspace* ws);

// The workspace is aligned to 16. H, W (default: the workspace's) may be any smaller logical
// extent; image and outputs then use that extent. When H or W is not a multiple of 8 a
// decoder level has an odd extent; its up2x is resized to the skip's size inside the conv's
// input staging (InMode::kUpCatRs), as the reference resizes it.
cudaError_t run_forward_map(const ForwardMapWeights& w, const float* image_hwc3,
                            ForwardMapWorkspace& ws, const ForwardMapOutputs& out,
                            cudaStream_t stream, int H = 0, int W = 0);

// Extract real pixels from the aligned output and normalize density over that crop.
// Writes logits separately so the caller can sum the Poisson rates over the same crop.
cudaError_t crop_forward_map(ForwardMapWorkspace& ws, const float* net_cov, const float* net_rgb,
                             const ForwardMapOutputs& out, float* logits, int H, int W,
                             cudaStream_t stream);

}  // namespace gfused
