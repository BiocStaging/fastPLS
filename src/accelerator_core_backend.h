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

bool metal_core_gemm_into_f32(
  fastpls::core::ConstMatrixView<float> left,
  fastpls::core::ConstMatrixView<float> right,
  bool transpose_left,
  bool transpose_right,
  fastpls::core::MatrixView<float> output
);

bool metal_core_gemm_accumulate_into_f32(
  fastpls::core::ConstMatrixView<float> left,
  fastpls::core::ConstMatrixView<float> right,
  bool transpose_left,
  bool transpose_right,
  fastpls::core::MatrixView<float> output
);

bool metal_core_rank1_subtract_f32(
  fastpls::core::MatrixView<float> target,
  fastpls::core::ConstMatrixView<float> column,
  fastpls::core::ConstMatrixView<float> row
);

void* metal_sample_gram_workspace_create_f32(
  fastpls::core::ConstMatrixView<float> predictors,
  fastpls::core::ConstMatrixView<float> sample_gram
);

void metal_sample_gram_workspace_destroy_f32(void* workspace) noexcept;

bool metal_sample_gram_apply_f32(
  void* workspace,
  fastpls::core::ConstMatrixView<float> direction,
  fastpls::core::MatrixView<float> output
);

bool metal_sample_geometry_f32(
  void* workspace,
  fastpls::core::ConstMatrixView<float> direction,
  fastpls::core::MatrixView<float> score,
  fastpls::core::MatrixView<float> loading
);

}  // namespace fastpls_svd

#endif
