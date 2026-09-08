// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Stefano Cacciatore
#ifndef FASTPLS_ACCELERATOR_CORE_BACKEND_H
#define FASTPLS_ACCELERATOR_CORE_BACKEND_H

#include <fastpls/core/matrix.hpp>

namespace fastpls_svd {

fastpls::core::Matrix<float> cuda_core_gemm_f32(
  fastpls::core::ConstMatrixView<float> left,
  fastpls::core::ConstMatrixView<float> right,
  bool transpose_left,
  bool transpose_right
);

fastpls::core::Matrix<float> metal_core_gemm_f32(
  fastpls::core::ConstMatrixView<float> left,
  fastpls::core::ConstMatrixView<float> right,
  bool transpose_left,
  bool transpose_right
);

}  // namespace fastpls_svd

#endif
