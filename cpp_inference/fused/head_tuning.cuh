// Test/tuning hooks for the fused correction head (not part of the pipeline interface).
#pragma once

#include <cuda_runtime.h>

#include "correction_head.cuh"

namespace gfused {

// Schedule knobs (value -1 restores the default):
//   kHeadSchedule    0 small-image schedule, 1 large-image schedule (default: by image size,
//                    see correction_head.cu); lets tests run both on the same inputs.
//   kHeadAuxNative   0 aux net padded to 32 at level 0 (the layout FP8 aux layers force),
//                    1 native 24 (default unless the aux enc1.a conv is FP8).
//   kHeadLayerCfg + net * 10 + layer (net 0 main / 1 aux, layer 1..9): conv_fused tile config
//                    index (0 .. kHeadNumCfgs - 1, see correction_head.cu); the per-layer sweep
//                    behind the tuned tables (tests/fused/test_correction_head_fused.py --tune).
//   kHeadAcc16 + net * 10 + layer (layer 1..9): ConvParams::acc16 (0 FP32 accumulation, 1 / 2
//                    FP16, see conv_fused.cuh) for that FP16 layer; such a layer runs on the
//                    conv core (not the lean full-resolution kernels). FP8 layers and the first
//                    conv (layer 0) always accumulate in FP32. The loader sets
//                    default_acc16() (conv_fused.cuh), or GFUSED_HEAD_ACC16 when it is set:
//                    one digit for every layer, or 18 digits (main layers 1..9, then aux).
// Returns false for an invalid key / value.
constexpr int kHeadSchedule = 0;
constexpr int kHeadAuxNative = 1;
constexpr int kHeadLayerCfg = 100;
constexpr int kHeadAcc16 = 200;
constexpr int kHeadNumCfgs = 31;  // 20..28: native 24-wide aux configs (BK 8 / 24)
bool set_correction_head_knob(CorrectionHeadWeights* w, int key, int value);

// Times `iters` launches of one layer (net 0 main / 1 aux, layer 0..9; layer 0 = the shared
// first conv) with its current config on `stream`, against the workspace's buffers. Returns
// milliseconds per launch (min over 5 bursts), or a negative value if the config is invalid.
float time_correction_head_layer(const CorrectionHeadWeights& w, CorrectionHeadWorkspace& ws,
                                 const CorrectionHeadInputs& in, int net, int layer, int iters,
                                 cudaStream_t stream);

}  // namespace gfused
