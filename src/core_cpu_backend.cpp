// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Stefano Cacciatore
#include "core_cpu_backend.h"

#include <fastpls/core/linalg.hpp>

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <type_traits>

#if !defined(_WIN32)
#include <dlfcn.h>
#else
#include <windows.h>
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
using CblasSgemv = void (*)(int, int, int, int, float, const float*, int,
                            const float*, int, float, float*, int);
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

CblasSgemv system_cblas_sgemv() {
  static CblasSgemv function = reinterpret_cast<CblasSgemv>(
    dlsym(RTLD_DEFAULT, "cblas_sgemv")
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

CblasSgemv linked_openblas_sgemv() {
  static CblasSgemv function =
    linked_openblas_function<CblasSgemv>("cblas_sgemv");
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

bool cpu_gemv_f32(core::ConstMatrixView<float> matrix,
                  bool transpose,
                  const float* vector,
                  float* output) {
  if (vector == nullptr || output == nullptr) return false;
#if defined(FASTPLS_USE_ACCELERATE)
  cblas_sgemv(
    CblasColMajor, transpose ? CblasTrans : CblasNoTrans,
    static_cast<int>(matrix.rows()),
    static_cast<int>(matrix.columns()), 1.0f, matrix.data(),
    static_cast<int>(matrix.leading_dimension()), vector, 1, 0.0f,
    output, 1
  );
  return true;
#elif defined(FASTPLS_USE_OPENBLAS)
  configure_openblas_threads();
#if defined(_WIN32)
  cblas_sgemv(
    CblasColMajor, transpose ? CblasTrans : CblasNoTrans,
    static_cast<int>(matrix.rows()),
    static_cast<int>(matrix.columns()), 1.0f, matrix.data(),
    static_cast<int>(matrix.leading_dimension()), vector, 1, 0.0f,
    output, 1
  );
#else
  const CblasSgemv sgemv = linked_openblas_sgemv();
  if (sgemv == nullptr) return false;
  sgemv(
    102, transpose ? 112 : 111,
    static_cast<int>(matrix.rows()),
    static_cast<int>(matrix.columns()), 1.0f, matrix.data(),
    static_cast<int>(matrix.leading_dimension()), vector, 1, 0.0f,
    output, 1
  );
#endif
  return true;
#elif !defined(_WIN32) && !defined(__APPLE__)
  const CblasSgemv sgemv = system_cblas_sgemv();
  if (sgemv == nullptr) return false;
  sgemv(
    102, transpose ? 112 : 111,
    static_cast<int>(matrix.rows()),
    static_cast<int>(matrix.columns()), 1.0f, matrix.data(),
    static_cast<int>(matrix.leading_dimension()), vector, 1, 0.0f,
    output, 1
  );
  return true;
#else
  return false;
#endif
}

template<class T>
void mirror_lower_to_upper(core::MatrixView<T> output) {
  constexpr std::size_t block_size = 32;
  for (std::size_t column_block = 0; column_block < output.columns();
       column_block += block_size) {
    const std::size_t column_end = std::min(
      output.columns(), column_block + block_size
    );
    for (std::size_t row_block = column_block; row_block < output.rows();
         row_block += block_size) {
      const std::size_t row_end = std::min(
        output.rows(), row_block + block_size
      );
      for (std::size_t column = column_block; column < column_end; ++column) {
        const std::size_t first_row = std::max(row_block, column + 1);
        for (std::size_t row = first_row; row < row_end; ++row) {
          output(column, row) = output(row, column);
        }
      }
    }
  }
}

template<class T>
bool is_self_gram(core::ConstMatrixView<T> left,
                  core::ConstMatrixView<T> right,
                  bool transpose_left,
                  bool transpose_right,
                  core::MatrixView<T> output) {
  if (left.data() != right.data() || left.rows() != right.rows() ||
      left.columns() != right.columns() ||
      left.leading_dimension() != right.leading_dimension() ||
      transpose_left == transpose_right || output.rows() != output.columns()) {
    return false;
  }
  const std::size_t dimension = transpose_left ? left.columns() : left.rows();
  return output.rows() == dimension;
}

template<class T>
bool use_symmetric_kernel(std::size_t dimension, std::size_t rank,
                          bool transpose_input) {
#if defined(FASTPLS_USE_ACCELERATE)
  // Accelerate's float32 GEMM overtakes SYRK for large sample-space outputs,
  // whereas float64 SYRK regains an advantage near n=1000. For transposed
  // products, SYRK wins in the measured medium-dimension, moderate-rank range;
  // GEMM remains faster for tiny outputs and very tall input matrices.
  if (transpose_input) {
    return dimension >= 128 && rank <= 8192;
  }
  if constexpr (std::is_same<T, float>::value) {
    return dimension <= 128;
  }
  return dimension >= 768;
#else
  // Avoid multithreaded SYRK launch overhead only for truly tiny products.
#if defined(_WIN32)
  if constexpr (std::is_same<T, float>::value) {
    return dimension >= 32;
  }
#endif
  return dimension >= 32 || rank >= 256;
#endif
}

}  // namespace

std::vector<std::string> set_cpu_threads(const int threads) {
  if (threads < 1) {
    throw std::invalid_argument("fastPLS CPU thread count must be positive");
  }
  std::vector<std::string> configured;
#if defined(FASTPLS_USE_OPENBLAS)
  openblas_set_num_threads(threads);
  configured.emplace_back("OpenBLAS");
#endif
#if !defined(_WIN32)
  using ThreadSetter = void (*)(int);
  const auto invoke = [&](const char* symbol, const char* runtime) {
    ThreadSetter setter = reinterpret_cast<ThreadSetter>(
      dlsym(RTLD_DEFAULT, symbol)
    );
    if (setter == nullptr) return;
    setter(threads);
    if (std::find(configured.begin(), configured.end(), runtime) ==
        configured.end()) {
      configured.emplace_back(runtime);
    }
  };
#if !defined(FASTPLS_USE_OPENBLAS)
  invoke("openblas_set_num_threads", "OpenBLAS");
#endif
  invoke("MKL_Set_Num_Threads", "MKL");
  invoke("mkl_set_num_threads", "MKL");
  invoke("bli_thread_set_num_threads", "BLIS");
  invoke("omp_set_num_threads", "OpenMP");
#elif !defined(FASTPLS_USE_OPENBLAS)
  using ThreadSetter = void (*)(int);
  const auto invoke = [&](const char* const* libraries,
                          const std::size_t library_count,
                          const char* symbol,
                          const char* runtime) {
    for (std::size_t index = 0; index < library_count; ++index) {
      HMODULE module = GetModuleHandleA(libraries[index]);
      if (module == nullptr) continue;
      ThreadSetter setter = reinterpret_cast<ThreadSetter>(
        GetProcAddress(module, symbol)
      );
      if (setter == nullptr) continue;
      setter(threads);
      if (std::find(configured.begin(), configured.end(), runtime) ==
          configured.end()) {
        configured.emplace_back(runtime);
      }
      return;
    }
  };
  const char* openblas_libraries[] = {
    "libopenblas.dll", "openblas.dll", "Rblas.dll"
  };
  const char* mkl_libraries[] = {"mkl_rt.dll"};
  const char* blis_libraries[] = {"libblis.dll", "blis.dll"};
  const char* openmp_libraries[] = {
    "libgomp-1.dll", "libomp.dll", "vcomp140.dll"
  };
  invoke(openblas_libraries, 3, "openblas_set_num_threads", "OpenBLAS");
  invoke(mkl_libraries, 1, "MKL_Set_Num_Threads", "MKL");
  invoke(mkl_libraries, 1, "mkl_set_num_threads", "MKL");
  invoke(blis_libraries, 2, "bli_thread_set_num_threads", "BLIS");
  invoke(openmp_libraries, 3, "omp_set_num_threads", "OpenMP");
#endif
  return configured;
}

void cpu_gemm_f32(core::ConstMatrixView<float> left,
                  core::ConstMatrixView<float> right,
                  bool transpose_left,
                  bool transpose_right,
                  core::MatrixView<float> output,
                  bool accumulate,
                  bool dispatch_symmetric) {
  const std::size_t rows = transpose_left ? left.columns() : left.rows();
  const std::size_t inner_left = transpose_left ? left.rows() : left.columns();
  const std::size_t inner_right = transpose_right ? right.columns() : right.rows();
  const std::size_t columns = transpose_right ? right.rows() : right.columns();
  if (inner_left != inner_right || output.rows() != rows ||
      output.columns() != columns) {
    throw std::invalid_argument("fastPLS CPU matrix-product dimensions are inconsistent");
  }

  if (!accumulate && !transpose_right && right.columns() == 1 &&
      cpu_gemv_f32(left, transpose_left, right.data(), output.data())) {
    return;
  }
  if (!accumulate && transpose_left && !transpose_right &&
      left.columns() == 1 &&
      output.rows() == 1 && cpu_gemv_f32(
        right, true, left.data(), output.data())) {
    return;
  }
  if (dispatch_symmetric && !accumulate &&
      is_self_gram(left, right, transpose_left, transpose_right, output)) {
    const bool transpose_input = transpose_left;
    const std::size_t dimension = transpose_input ? left.columns() : left.rows();
    const std::size_t rank = transpose_input ? left.rows() : left.columns();
    if (use_symmetric_kernel<float>(dimension, rank, transpose_input)) {
      cpu_self_gram_f32(left, transpose_input, output, true);
      return;
    }
  }

#if defined(FASTPLS_USE_ACCELERATE)
  cblas_sgemm(
    CblasColMajor,
    transpose_left ? CblasTrans : CblasNoTrans,
    transpose_right ? CblasTrans : CblasNoTrans,
    static_cast<int>(rows), static_cast<int>(columns),
    static_cast<int>(inner_left), 1.0f, left.data(),
    static_cast<int>(left.leading_dimension()), right.data(),
    static_cast<int>(right.leading_dimension()),
    accumulate ? 1.0f : 0.0f, output.data(),
    static_cast<int>(output.leading_dimension())
  );
#elif defined(FASTPLS_USE_OPENBLAS)
  configure_openblas_threads();
#if defined(_WIN32)
  cblas_sgemm(
    CblasColMajor,
    transpose_left ? CblasTrans : CblasNoTrans,
    transpose_right ? CblasTrans : CblasNoTrans,
    static_cast<int>(rows), static_cast<int>(columns),
    static_cast<int>(inner_left), 1.0f, left.data(),
    static_cast<int>(left.leading_dimension()), right.data(),
    static_cast<int>(right.leading_dimension()),
    accumulate ? 1.0f : 0.0f, output.data(),
    static_cast<int>(output.leading_dimension())
  );
#else
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
    static_cast<int>(right.leading_dimension()),
    accumulate ? 1.0f : 0.0f, output.data(),
    static_cast<int>(output.leading_dimension())
  );
#endif
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
      static_cast<int>(right.leading_dimension()),
      accumulate ? 1.0f : 0.0f, output.data(),
      static_cast<int>(output.leading_dimension())
    );
    return;
  }
  if (!accumulate) {
    core::reference_gemm(
      left, right, transpose_left, transpose_right, output
    );
  } else {
    core::Matrix<float> temporary(rows, columns);
    core::reference_gemm(
      left, right, transpose_left, transpose_right, temporary.view()
    );
    for (std::size_t column = 0; column < columns; ++column) {
      for (std::size_t row = 0; row < rows; ++row) {
        output(row, column) += temporary(row, column);
      }
    }
  }
#else
  if (!accumulate) {
    core::reference_gemm(
      left, right, transpose_left, transpose_right, output
    );
  } else {
    core::Matrix<float> temporary(rows, columns);
    core::reference_gemm(
      left, right, transpose_left, transpose_right, temporary.view()
    );
    for (std::size_t column = 0; column < columns; ++column) {
      for (std::size_t row = 0; row < rows; ++row) {
        output(row, column) += temporary(row, column);
      }
    }
  }
#endif
}

void cpu_gemm_f64(core::ConstMatrixView<double> left,
                  core::ConstMatrixView<double> right,
                  bool transpose_left,
                  bool transpose_right,
                  core::MatrixView<double> output,
                  bool accumulate,
                  bool dispatch_symmetric) {
  const std::size_t rows = transpose_left ? left.columns() : left.rows();
  const std::size_t inner_left = transpose_left ? left.rows() : left.columns();
  const std::size_t inner_right = transpose_right ? right.columns() : right.rows();
  const std::size_t columns = transpose_right ? right.rows() : right.columns();
  if (inner_left != inner_right || output.rows() != rows ||
      output.columns() != columns) {
    throw std::invalid_argument("fastPLS CPU matrix-product dimensions are inconsistent");
  }
  if (dispatch_symmetric && !accumulate &&
      is_self_gram(left, right, transpose_left, transpose_right, output)) {
    const bool transpose_input = transpose_left;
    const std::size_t dimension = transpose_input ? left.columns() : left.rows();
    const std::size_t rank = transpose_input ? left.rows() : left.columns();
    if (use_symmetric_kernel<double>(dimension, rank, transpose_input)) {
      cpu_self_gram_f64(left, transpose_input, output, true);
      return;
    }
  }

#if defined(FASTPLS_USE_ACCELERATE)
  cblas_dgemm(
    CblasColMajor,
    transpose_left ? CblasTrans : CblasNoTrans,
    transpose_right ? CblasTrans : CblasNoTrans,
    static_cast<int>(rows), static_cast<int>(columns),
    static_cast<int>(inner_left), 1.0, left.data(),
    static_cast<int>(left.leading_dimension()), right.data(),
    static_cast<int>(right.leading_dimension()),
    accumulate ? 1.0 : 0.0, output.data(),
    static_cast<int>(output.leading_dimension())
  );
#elif defined(FASTPLS_USE_OPENBLAS)
  configure_openblas_threads();
#if defined(_WIN32)
  cblas_dgemm(
    CblasColMajor,
    transpose_left ? CblasTrans : CblasNoTrans,
    transpose_right ? CblasTrans : CblasNoTrans,
    static_cast<int>(rows), static_cast<int>(columns),
    static_cast<int>(inner_left), 1.0, left.data(),
    static_cast<int>(left.leading_dimension()), right.data(),
    static_cast<int>(right.leading_dimension()),
    accumulate ? 1.0 : 0.0, output.data(),
    static_cast<int>(output.leading_dimension())
  );
#else
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
    static_cast<int>(right.leading_dimension()),
    accumulate ? 1.0 : 0.0, output.data(),
    static_cast<int>(output.leading_dimension())
  );
#endif
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
  const double beta = accumulate ? 1.0 : 0.0;
  F77_CALL(dgemm)(
    &trans_left, &trans_right, &m, &n, &k, &alpha, left.data(), &lda,
    right.data(), &ldb, &beta, output.data(), &ldc FCONE FCONE
  );
#endif
}

void cpu_self_gram_f32(core::ConstMatrixView<float> input,
                       bool transpose_input,
                       core::MatrixView<float> output,
                       bool full_output) {
  const std::size_t dimension = transpose_input ? input.columns() : input.rows();
  const std::size_t rank = transpose_input ? input.rows() : input.columns();
  if (output.rows() != dimension || output.columns() != dimension) {
    throw std::invalid_argument(
      "fastPLS CPU float32 self-Gram dimensions are inconsistent"
    );
  }
  if (!use_symmetric_kernel<float>(dimension, rank, transpose_input)) {
    cpu_gemm_f32(
      input, input, transpose_input, !transpose_input, output, false, false
    );
    return;
  }
#if defined(FASTPLS_USE_ACCELERATE)
  cblas_ssyrk(
    CblasColMajor, CblasLower,
    transpose_input ? CblasTrans : CblasNoTrans,
    static_cast<int>(dimension), static_cast<int>(rank),
    1.0f, input.data(), static_cast<int>(input.leading_dimension()),
    0.0f, output.data(), static_cast<int>(output.leading_dimension())
  );
#elif defined(FASTPLS_USE_OPENBLAS)
  configure_openblas_threads();
  cblas_ssyrk(
    CblasColMajor, CblasLower,
    transpose_input ? CblasTrans : CblasNoTrans,
    static_cast<int>(dimension), static_cast<int>(rank),
    1.0f, input.data(), static_cast<int>(input.leading_dimension()),
    0.0f, output.data(), static_cast<int>(output.leading_dimension())
  );
#else
  core::reference_gemm(
    input, input, transpose_input, !transpose_input, output
  );
#endif
  if (full_output) mirror_lower_to_upper(output);
}

void cpu_self_gram_f64(core::ConstMatrixView<double> input,
                       bool transpose_input,
                       core::MatrixView<double> output,
                       bool full_output) {
  const std::size_t dimension = transpose_input ? input.columns() : input.rows();
  const std::size_t rank = transpose_input ? input.rows() : input.columns();
  if (output.rows() != dimension || output.columns() != dimension) {
    throw std::invalid_argument(
      "fastPLS CPU float64 self-Gram dimensions are inconsistent"
    );
  }
  if (!use_symmetric_kernel<double>(dimension, rank, transpose_input)) {
    cpu_gemm_f64(
      input, input, transpose_input, !transpose_input, output, false, false
    );
    return;
  }
#if defined(FASTPLS_USE_ACCELERATE)
  cblas_dsyrk(
    CblasColMajor, CblasLower,
    transpose_input ? CblasTrans : CblasNoTrans,
    static_cast<int>(dimension), static_cast<int>(rank),
    1.0, input.data(), static_cast<int>(input.leading_dimension()),
    0.0, output.data(), static_cast<int>(output.leading_dimension())
  );
#elif defined(FASTPLS_USE_OPENBLAS)
  configure_openblas_threads();
  cblas_dsyrk(
    CblasColMajor, CblasLower,
    transpose_input ? CblasTrans : CblasNoTrans,
    static_cast<int>(dimension), static_cast<int>(rank),
    1.0, input.data(), static_cast<int>(input.leading_dimension()),
    0.0, output.data(), static_cast<int>(output.leading_dimension())
  );
#else
  const char lower = 'L';
  const char transpose = transpose_input ? 'T' : 'N';
  const BLAS_INT n = static_cast<BLAS_INT>(dimension);
  const BLAS_INT k = static_cast<BLAS_INT>(rank);
  const BLAS_INT lda = static_cast<BLAS_INT>(input.leading_dimension());
  const BLAS_INT ldc = static_cast<BLAS_INT>(output.leading_dimension());
  const double alpha = 1.0;
  const double beta = 0.0;
  F77_CALL(dsyrk)(
    &lower, &transpose, &n, &k, &alpha, input.data(), &lda, &beta,
    output.data(), &ldc FCONE FCONE
  );
#endif
  if (full_output) mirror_lower_to_upper(output);
}

void cpu_crossprod_f32(core::ConstMatrixView<float> input,
                       core::MatrixView<float> output) {
  cpu_gemm_f32(input, input, true, false, output);
}

void CpuLinearAlgebraF64::gemm(core::ConstMatrixView<double> left,
                               core::ConstMatrixView<double> right,
                               bool transpose_left,
                               bool transpose_right,
                               core::MatrixView<double> output) const {
  cpu_gemm_f64(
    left, right, transpose_left, transpose_right, output
  );
}

void CpuLinearAlgebraF64::gemm_accumulate(
    core::ConstMatrixView<double> left,
    core::ConstMatrixView<double> right,
    bool transpose_left,
    bool transpose_right,
    core::MatrixView<double> output) const {
  cpu_gemm_f64(
    left, right, transpose_left, transpose_right, output, true
  );
}

void CpuLinearAlgebraF64::self_gram(
    core::ConstMatrixView<double> input, bool transpose_input,
    core::MatrixView<double> output, bool full_output) const {
  cpu_self_gram_f64(input, transpose_input, output, full_output);
}

void CpuLinearAlgebraF32::gemm(core::ConstMatrixView<float> left,
                               core::ConstMatrixView<float> right,
                               bool transpose_left,
                               bool transpose_right,
                               core::MatrixView<float> output) const {
  cpu_gemm_f32(
    left, right, transpose_left, transpose_right, output
  );
}

void CpuLinearAlgebraF32::gemm_accumulate(
    core::ConstMatrixView<float> left,
    core::ConstMatrixView<float> right,
    bool transpose_left,
    bool transpose_right,
    core::MatrixView<float> output) const {
  cpu_gemm_f32(
    left, right, transpose_left, transpose_right, output, true
  );
}

void CpuLinearAlgebraF32::self_gram(
    core::ConstMatrixView<float> input, bool transpose_input,
    core::MatrixView<float> output, bool full_output) const {
  cpu_self_gram_f32(input, transpose_input, output, full_output);
}

}  // namespace runtime
}  // namespace fastpls
