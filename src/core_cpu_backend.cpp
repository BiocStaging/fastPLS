// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Stefano Cacciatore
#include "core_cpu_backend.h"

#include <fastpls/core/linalg.hpp>

#include <algorithm>
#include <cstdlib>
#include <stdexcept>

#if !defined(_WIN32)
#include <dlfcn.h>
#endif

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

#if !defined(_WIN32)
using CblasSgemm = void (*)(int, int, int, int, int, int, float,
                            const float*, int, const float*, int, float,
                            float*, int);
using CblasDgemm = void (*)(int, int, int, int, int, int, double,
                            const double*, int, const double*, int, double,
                            double*, int);
#endif

#if !defined(FASTPLS_USE_ACCELERATE) && \
    !defined(FASTPLS_USE_OPENBLAS) && \
    !defined(_WIN32) && !defined(__APPLE__)
CblasSgemm system_cblas_sgemm() {
  static CblasSgemm function = reinterpret_cast<CblasSgemm>(
    dlsym(RTLD_DEFAULT, "cblas_sgemm")
  );
  return function;
}
#endif

#if defined(FASTPLS_USE_OPENBLAS) && !defined(_WIN32)
void* linked_openblas_library() {
  static void* library = [] {
    Dl_info information{};
    if (dladdr(reinterpret_cast<void*>(openblas_get_config), &information) == 0 ||
        information.dli_fname == nullptr) {
      return static_cast<void*>(nullptr);
    }
    return dlopen(information.dli_fname, RTLD_LAZY | RTLD_LOCAL);
  }();
  return library;
}

template<class Function>
Function linked_openblas_function(const char* name) {
  void* library = linked_openblas_library();
  return library == nullptr ? nullptr :
    reinterpret_cast<Function>(dlsym(library, name));
}

CblasSgemm linked_openblas_sgemm() {
  static CblasSgemm function =
    linked_openblas_function<CblasSgemm>("cblas_sgemm");
  return function;
}

CblasDgemm linked_openblas_dgemm() {
  static CblasDgemm function =
    linked_openblas_function<CblasDgemm>("cblas_dgemm");
  return function;
}
#endif

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

std::string cpu_backend_description() {
#if defined(FASTPLS_USE_ACCELERATE)
  return "Apple Accelerate";
#elif defined(FASTPLS_USE_OPENBLAS)
  configure_openblas_threads();
  const char* configuration = openblas_get_config();
  return configuration == nullptr ? "OpenBLAS" :
    std::string("OpenBLAS: ") + configuration;
#elif defined(_WIN32)
  return "R BLAS/LAPACK with reference float32 products";
#elif defined(__APPLE__)
  return "Apple system BLAS";
#else
  return system_cblas_sgemm() == nullptr ?
    "R BLAS with reference float32 products" : "system CBLAS";
#endif
}

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

#if defined(FASTPLS_USE_ACCELERATE)
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
#elif defined(FASTPLS_USE_OPENBLAS)
  configure_openblas_threads();
  const CblasSgemm sgemm = linked_openblas_sgemm();
  if (sgemm == nullptr) {
    throw std::runtime_error("fastPLS could not resolve OpenBLAS SGEMM");
  }
  sgemm(
    102,
    transpose_left ? 112 : 111,
    transpose_right ? 112 : 111,
    static_cast<int>(rows), static_cast<int>(columns),
    static_cast<int>(inner_left), 1.0f, left.data(),
    static_cast<int>(left.leading_dimension()), right.data(),
    static_cast<int>(right.leading_dimension()), 0.0f, output.data(),
    static_cast<int>(output.leading_dimension())
  );
#elif !defined(_WIN32) && !defined(__APPLE__)
  const CblasSgemm sgemm = system_cblas_sgemm();
  if (sgemm != nullptr) {
    // CBLAS uses stable integer values for column-major and transpose flags.
    constexpr int column_major = 102;
    constexpr int no_transpose = 111;
    constexpr int transpose = 112;
    sgemm(
      column_major,
      transpose_left ? transpose : no_transpose,
      transpose_right ? transpose : no_transpose,
      static_cast<int>(rows), static_cast<int>(columns),
      static_cast<int>(inner_left), 1.0f, left.data(),
      static_cast<int>(left.leading_dimension()), right.data(),
      static_cast<int>(right.leading_dimension()), 0.0f, output.data(),
      static_cast<int>(output.leading_dimension())
    );
    return;
  }
  core::reference_gemm(
    left, right, transpose_left, transpose_right, output
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

#if defined(FASTPLS_USE_ACCELERATE)
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
#elif defined(FASTPLS_USE_OPENBLAS)
  configure_openblas_threads();
  const CblasDgemm dgemm = linked_openblas_dgemm();
  if (dgemm == nullptr) {
    throw std::runtime_error("fastPLS could not resolve OpenBLAS DGEMM");
  }
  dgemm(
    102,
    transpose_left ? 112 : 111,
    transpose_right ? 112 : 111,
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
