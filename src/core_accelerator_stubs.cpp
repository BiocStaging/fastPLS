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

fastpls::core::Matrix<float> cuda_core_gemm_f32(
    fastpls::core::ConstMatrixView<float>,
    fastpls::core::ConstMatrixView<float>, bool, bool) {
  throw std::runtime_error(
    "CUDA backend requested but this fastPLS build has no CUDA support"
  );
}

void* cuda_core_workspace_create_f32() {
  throw std::runtime_error(
    "CUDA backend requested but this fastPLS build has no CUDA support"
  );
}

void cuda_core_workspace_destroy_f32(void*) noexcept {}

bool cuda_core_gemm_into_f32(
    void*, fastpls::core::ConstMatrixView<float>,
    fastpls::core::ConstMatrixView<float>, bool, bool,
    fastpls::core::MatrixView<float>) {
  throw std::runtime_error(
    "CUDA backend requested but this fastPLS build has no CUDA support"
  );
}

bool cuda_core_self_gram_into_f32(
    void*, fastpls::core::ConstMatrixView<float>, bool,
    fastpls::core::MatrixView<float>, bool) {
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

bool metal_core_gemm_into_f32(
    fastpls::core::ConstMatrixView<float>,
    fastpls::core::ConstMatrixView<float>, bool, bool,
    fastpls::core::MatrixView<float>) {
  return false;
}

bool metal_core_gemm_accumulate_into_f32(
    fastpls::core::ConstMatrixView<float>,
    fastpls::core::ConstMatrixView<float>, bool, bool,
    fastpls::core::MatrixView<float>) {
  return false;
}

bool metal_core_rank1_subtract_f32(
    fastpls::core::MatrixView<float>,
    fastpls::core::ConstMatrixView<float>,
    fastpls::core::ConstMatrixView<float>) {
  throw std::runtime_error(
    "Metal backend requested but this fastPLS build has no Metal support"
  );
}

void* metal_sample_gram_workspace_create_f32(
    fastpls::core::ConstMatrixView<float>,
    fastpls::core::ConstMatrixView<float>) {
  throw std::runtime_error(
    "Metal backend requested but this fastPLS build has no Metal support"
  );
}

void metal_sample_gram_workspace_destroy_f32(void*) noexcept {}

bool metal_sample_gram_apply_f32(
    void*, fastpls::core::ConstMatrixView<float>,
    fastpls::core::MatrixView<float>) {
  throw std::runtime_error(
    "Metal backend requested but this fastPLS build has no Metal support"
  );
}

bool metal_sample_geometry_f32(
    void*, fastpls::core::ConstMatrixView<float>,
    fastpls::core::MatrixView<float>,
    fastpls::core::MatrixView<float>) {
  throw std::runtime_error(
    "Metal backend requested but this fastPLS build has no Metal support"
  );
}
#endif

}  // namespace fastpls_svd

#endif
