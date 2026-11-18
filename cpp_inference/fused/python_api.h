// Synchronous host-array C ABI for libgaussifier_fused.
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#define GFUSED_NOEXCEPT noexcept
#else
#define GFUSED_NOEXCEPT
#endif

typedef struct GfusedSession GfusedSession;

const char* gfused_error(void) GFUSED_NOEXCEPT;
int gfused_create(const char* fm, const char* head, int device,
                  GfusedSession** output) GFUSED_NOEXCEPT;
void gfused_destroy(GfusedSession* session) GFUSED_NOEXCEPT;
int gfused_run(GfusedSession* session, const float* input, int height, int width,
               int states, int64_t seed, int count) GFUSED_NOEXCEPT;
int gfused_run_with_size_mode(GfusedSession* session, const float* input, int height, int width,
                              int states, int64_t seed, int count, int size_mode) GFUSED_NOEXCEPT;

// Read the latest result for an image or video session. Supply H*W floats for density,
// 3*H*W for render, 2*N for each xy array, N for weights, and 3*N for all other arrays.
int gfused_read(GfusedSession* session, float* density, float* render, float* initial_xy,
                float* initial_cov, float* initial_rgb, float* xy, float* cov, float* rgb,
                float* colour, float* weights) GFUSED_NOEXCEPT;

enum { GFUSED_MAX_STATES = 16 };

typedef struct GfusedVideoStats {
  uint64_t frame;
  int32_t n_requested, n_target, n_target_limited, survivors;
  int32_t kills_for_next, births_for_next, shortfall_for_next, k_used;
  double retained_fraction, rate_sum;
  double psnr[GFUSED_MAX_STATES];  // First k_used entries.
} GfusedVideoStats;

// Opens an ordered, fixed-size stream sharing the source model. The stream can outlive the
// source session. size_mode is 0 for pad, 1 for exact. count=0 selects adaptive count.
// Release with gfused_video_destroy. Calls return -1 on error (see gfused_error).
int gfused_video_create(GfusedSession* source, int height, int width, int states, int64_t seed,
                        int count, int size_mode, double count_ema, double warm_attribute_blend,
                        GfusedSession** output) GFUSED_NOEXCEPT;
// Runs a single CHW float32 frame and returns N. Read the result before pushing another frame.
// A run failure invalidates the stream. An argument validation failure leaves it usable.
int gfused_video_push(GfusedSession* session, const float* input, int height, int width,
                      GfusedVideoStats* stats) GFUSED_NOEXCEPT;
// Read N IDs and slots in the same row order as gfused_read.
int gfused_video_read_tracks(GfusedSession* session, int64_t* ids, int32_t* slots) GFUSED_NOEXCEPT;
void gfused_video_destroy(GfusedSession* session) GFUSED_NOEXCEPT;

#ifdef __cplusplus
}
#endif
#undef GFUSED_NOEXCEPT
