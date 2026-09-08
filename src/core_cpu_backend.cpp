// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Stefano Cacciatore
#include "core_cpu_backend.h"

#include <fastpls/core/linalg.hpp>

#include <algorithm>
#include <cstdlib>
#include <stdexcept>

#if defined(FASTPLS_USE_ACCELERATE)
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#elif defined(FASTPLS_USE_OPENBLAS)
#include <cblas.h>
#include <openblas_config.h>
#else
#include <R_ext/BLAS.h>
#include <R_ext/RS.h>
#endif

namespace fastpls {
namespace runtime {
namespace {

#if defined(FASTPLS_USE_OPENBLAS)
int requested_threads() {
  const char* raw = std::getenv("OPENBLAS_NUM_THREADS");
  if (raw == nullptr) return 1;
  char* end = nullptr;
  const long parsed = std::strtol(raw, &end, 10);
  if (end == raw) return 1;
  return static_cast<int>(std::max(1L, std::min(parsed, 1024L)));
}

void configure_openblas_threads() {
  const int requested = requested_threads();
  static int configured = -1;
  if (configured != requested) {
    openblas_set_num_threads(requested);
    configured = requested;
  }
}
#endif

}  // namespace

void cpu_gemm_f32(core::ConstMatrixView<float> left,
                  core::ConstMatrixView<float> right,
                  bool transpose_left,
                  bool transpose_right,
                  core::MatrixView<float> output) {
  const std::size_t rows = transpose_left ? left.columns() : left.rows();
  const std::size_t inner_left = transpose_left ? left.rows() : left.columns();
  const std::size_t inner_right = transpose_right ? right.columns() : right.rows();
  const std::size_t columns = transpose_right ? right.rows() : right.columns();
  if (inner_left != inner_right || output.rows() != rows ||
      output.columns() != columns) {
    throw std::invalid_argument("fastPLS CPU matrix-product dimensions are inconsistent");
  }

#if defined(FASTPLS_USE_ACCELERATE) || defined(FASTPLS_USE_OPENBLAS)
#if defined(FASTPLS_USE_OPENBLAS)
  configure_openblas_threads();
#endif
  cblas_sgemm(
    CblasColMajor,
    transpose_left ? CblasTrans : CblasNoTrans,
    transpose_right ? CblasTrans : CblasNoTrans,
    static_cast<int>(rows), static_cast<int>(columns),
    static_cast<int>(inner_left), 1.0f, left.data(),
    static_cast<int>(left.leading_dimension()), right.data(),
    static_cast<int>(right.leading_dimension()), 0.0f, output.data(),
    static_cast<int>(output.leading_dimension())
  );
#else
  core::reference_gemm(
    left, right, transpose_left, transpose_right, output
  );
#endif
}

void cpu_gemm_f64(core::ConstMatrixView<double> left,
                  core::ConstMatrixView<double> right,
                  bool transpose_left,
                  bool transpose_right,
                  core::MatrixView<double> output) {
  const std::size_t rows = transpose_left ? left.columns() : left.rows();
  const std::size_t inner_left = transpose_left ? left.rows() : left.columns();
  const std::size_t inner_right = transpose_right ? right.columns() : right.rows();
  const std::size_t columns = transpose_right ? right.rows() : right.columns();
  if (inner_left != inner_right || output.rows() != rows ||
      output.columns() != columns) {
    throw std::invalid_argument("fastPLS CPU matrix-product dimensions are inconsistent");
  }

#if defined(FASTPLS_USE_ACCELERATE) || defined(FASTPLS_USE_OPENBLAS)
#if defined(FASTPLS_USE_OPENBLAS)
  configure_openblas_threads();
#endif
  cblas_dgemm(
    CblasColMajor,
    transpose_left ? CblasTrans : CblasNoTrans,
    transpose_right ? CblasTrans : CblasNoTrans,
    static_cast<int>(rows), static_cast<int>(columns),
    static_cast<int>(inner_left), 1.0, left.data(),
    static_cast<int>(left.leading_dimension()), right.data(),
    static_cast<int>(right.leading_dimension()), 0.0, output.data(),
    static_cast<int>(output.leading_dimension())
  );
#else
  const char trans_left = transpose_left ? 'T' : 'N';
  const char trans_right = transpose_right ? 'T' : 'N';
  const BLAS_INT m = static_cast<BLAS_INT>(rows);
  const BLAS_INT n = static_cast<BLAS_INT>(columns);
  const BLAS_INT k = static_cast<BLAS_INT>(inner_left);
  const BLAS_INT lda = static_cast<BLAS_INT>(left.leading_dimension());
  const BLAS_INT ldb = static_cast<BLAS_INT>(right.leading_dimension());
  const BLAS_INT ldc = static_cast<BLAS_INT>(output.leading_dimension());
  const double alpha = 1.0;
  const double beta = 0.0;
  F77_CALL(dgemm)(
    &trans_left, &trans_right, &m, &n, &k, &alpha, left.data(), &lda,
    right.data(), &ldb, &beta, output.data(), &ldc FCONE FCONE
  );
#endif
}

}  // namespace runtime
}  // namespace fastpls
