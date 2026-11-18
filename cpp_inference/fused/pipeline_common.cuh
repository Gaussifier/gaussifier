// Shared helpers for the torch-free fused pipeline: a bump arena over one cudaMalloc and
// launch/error helpers. Every component plans its buffers twice through the same code path:
// once against a null base (to measure) and once against the real allocation.

#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace gfused {

class Arena {
 public:
  explicit Arena(uint8_t* base = nullptr) : base_(base) {}

  template <typename T>
  T* take(size_t count, size_t align = 256) {
    off_ = (off_ + align - 1) / align * align;
    T* p = base_ ? reinterpret_cast<T*>(base_ + off_) : nullptr;
    off_ += count * sizeof(T);
    return p;
  }
  size_t used() const { return off_; }

 private:
  uint8_t* base_;
  size_t off_ = 0;
};

inline unsigned grid_for(long long n, int block = 256) {
  return static_cast<unsigned>((n + block - 1) / block);
}

#define GF_TRY(expr)                        \
  do {                                      \
    cudaError_t gf_err_ = (expr);           \
    if (gf_err_ != cudaSuccess) return gf_err_; \
  } while (0)

}  // namespace gfused
