// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Stefano Cacciatore
#ifndef FASTPLS_CORE_CPU_BACKEND_H
#define FASTPLS_CORE_CPU_BACKEND_H

#include <fastpls/core/matrix.hpp>

#include <string>
#include <vector>

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

class CpuLinearAlgebraF32 {
 public:
  void gemm(core::ConstMatrixView<float> left,
            core::ConstMatrixView<float> right,
            bool transpose_left,
            bool transpose_right,
            core::MatrixView<float> output) const;

  bool qr_economy(core::ConstMatrixView<float> input,
                  core::Matrix<float>& q) const;

  bool symmetric_eigen(core::Matrix<float>& matrix,
                       std::vector<float>& eigenvalues) const;

  bool svd_economy(core::ConstMatrixView<float> input,
                   bool left_only,
                   core::Matrix<float>& u,
                   std::vector<float>& singular_values,
                   core::Matrix<float>& vt) const;
};

class CpuLinearAlgebraF64 {
 public:
  void gemm(core::ConstMatrixView<double> left,
            core::ConstMatrixView<double> right,
            bool transpose_left,
            bool transpose_right,
            core::MatrixView<double> output) const;

  bool qr_economy(core::ConstMatrixView<double> input,
                  core::Matrix<double>& q) const;

  bool symmetric_eigen(core::Matrix<double>& matrix,
                       std::vector<double>& eigenvalues) const;

  bool svd_economy(core::ConstMatrixView<double> input,
                   bool left_only,
                   core::Matrix<double>& u,
                   std::vector<double>& singular_values,
                   core::Matrix<double>& vt) const;
};

std::string cpu_backend_description();

}  // namespace runtime
}  // namespace fastpls

#endif
