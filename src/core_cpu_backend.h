// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Stefano Cacciatore
#ifndef FASTPLS_CORE_CPU_BACKEND_H
#define FASTPLS_CORE_CPU_BACKEND_H

#include <fastpls/core/matrix.hpp>

namespace fastpls {
namespace runtime {

void cpu_gemm_f32(core::ConstMatrixView<float> left,
                  core::ConstMatrixView<float> right,
                  bool transpose_left,
                  bool transpose_right,
                  core::MatrixView<float> output);

void cpu_gemm_f64(core::ConstMatrixView<double> left,
                  core::ConstMatrixView<double> right,
                  bool transpose_left,
                  bool transpose_right,
                  core::MatrixView<double> output);

}  // namespace runtime
}  // namespace fastpls

#endif
