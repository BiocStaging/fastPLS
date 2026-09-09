// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Stefano Cacciatore

#ifdef FASTPLS_CORE_ONLY

#include "accelerator_core_backend.h"

#include <stdexcept>

namespace fastpls_svd {

#ifndef FASTPLS_HAS_CUDA
bool has_cuda_backend() {
  return false;
}

bool cuda_lda_native_available() {
  return false;
}

fastpls::core::Matrix<float> cuda_core_gemm_f32(
    fastpls::core::ConstMatrixView<float>,
    fastpls::core::ConstMatrixView<float>, bool, bool) {
  throw std::runtime_error(
    "CUDA backend requested but this fastPLS build has no CUDA support"
  );
}
#endif

#ifndef FASTPLS_HAS_METAL
bool has_metal_backend() {
  return false;
}

fastpls::core::Matrix<float> metal_core_gemm_f32(
    fastpls::core::ConstMatrixView<float>,
    fastpls::core::ConstMatrixView<float>, bool, bool) {
  throw std::runtime_error(
    "Metal backend requested but this fastPLS build has no Metal support"
  );
}
#endif

}  // namespace fastpls_svd

#endif
