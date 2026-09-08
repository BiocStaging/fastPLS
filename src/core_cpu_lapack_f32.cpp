// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Stefano Cacciatore
#include "core_cpu_backend.h"

#include <R_ext/Lapack.h>
#include <R_ext/RS.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(FASTPLS_USE_ACCELERATE) || defined(FASTPLS_USE_OPENBLAS)
extern "C" {
void F77_NAME(sgeqrf)(const La_INT*, const La_INT*, float*, const La_INT*,
                      float*, float*, const La_INT*, La_INT*);
void F77_NAME(sorgqr)(const La_INT*, const La_INT*, const La_INT*, float*,
                      const La_INT*, const float*, float*, const La_INT*,
                      La_INT*);
void F77_NAME(ssyevd)(const char*, const char*, const La_INT*, float*,
                      const La_INT*, float*, float*, const La_INT*, La_INT*,
                      const La_INT*, La_INT* FCLEN FCLEN);
void F77_NAME(sgesvd)(const char*, const char*, const La_INT*, const La_INT*,
                      float*, const La_INT*, float*, float*, const La_INT*,
                      float*, const La_INT*, float*, const La_INT*, La_INT*
                      FCLEN FCLEN);
void F77_NAME(sgesdd)(const char*, const La_INT*, const La_INT*, float*,
                      const La_INT*, float*, float*, const La_INT*, float*,
                      const La_INT*, float*, const La_INT*, La_INT*, La_INT*
                      FCLEN);
}
#endif

namespace fastpls {
namespace runtime {
namespace {

#if defined(FASTPLS_USE_ACCELERATE) || defined(FASTPLS_USE_OPENBLAS)
template<class T>
core::Matrix<T> contiguous_copy(core::ConstMatrixView<T> input) {
  core::Matrix<T> output(input.rows(), input.columns());
  for (std::size_t column = 0; column < input.columns(); ++column) {
    std::copy_n(
      input.data() + column * input.leading_dimension(), input.rows(),
      output.data() + column * output.rows()
    );
  }
  return output;
}

La_INT lapack_dimension(std::size_t value, const char* operation) {
  if (value > static_cast<std::size_t>(std::numeric_limits<La_INT>::max())) {
    throw std::overflow_error(
      std::string("fastPLS float32 ") + operation +
      " dimension exceeds LAPACK range"
    );
  }
  return static_cast<La_INT>(value);
}

La_INT workspace_size(float query) {
  if (!std::isfinite(query) || query < 1.0f ||
      query > static_cast<float>(std::numeric_limits<La_INT>::max())) {
    throw std::runtime_error(
      "fastPLS received an invalid float32 LAPACK workspace size"
    );
  }
  return static_cast<La_INT>(query);
}
#endif

#if !defined(FASTPLS_USE_ACCELERATE) && !defined(FASTPLS_USE_OPENBLAS)
float column_dot(const core::Matrix<float>& matrix,
                 std::size_t first, std::size_t second) {
  float value = 0.0f;
  for (std::size_t row = 0; row < matrix.rows(); ++row) {
    value += matrix(row, first) * matrix(row, second);
  }
  return value;
}

float column_norm(const core::Matrix<float>& matrix, std::size_t column) {
  return std::sqrt(std::max(column_dot(matrix, column, column), 0.0f));
}

bool portable_qr(core::ConstMatrixView<float> input,
                 core::Matrix<float>& q) {
  const std::size_t rank = std::min(input.rows(), input.columns());
  q.resize(input.rows(), rank);
  const float threshold = std::numeric_limits<float>::epsilon() *
    static_cast<float>(std::max(input.rows(), input.columns()));
  for (std::size_t column = 0; column < rank; ++column) {
    for (std::size_t row = 0; row < input.rows(); ++row) {
      q(row, column) = input(row, column);
    }
    for (int pass = 0; pass < 2; ++pass) {
      for (std::size_t previous = 0; previous < column; ++previous) {
        const float projection = column_dot(q, previous, column);
        for (std::size_t row = 0; row < q.rows(); ++row) {
          q(row, column) -= projection * q(row, previous);
        }
      }
    }
    float norm = column_norm(q, column);
    if (!std::isfinite(norm) || norm <= threshold) {
      bool completed = false;
      for (std::size_t basis = 0; basis < q.rows() && !completed; ++basis) {
        for (std::size_t row = 0; row < q.rows(); ++row) {
          q(row, column) = row == basis ? 1.0f : 0.0f;
        }
        for (int pass = 0; pass < 2; ++pass) {
          for (std::size_t previous = 0; previous < column; ++previous) {
            const float projection = column_dot(q, previous, column);
            for (std::size_t row = 0; row < q.rows(); ++row) {
              q(row, column) -= projection * q(row, previous);
            }
          }
        }
        norm = column_norm(q, column);
        completed = std::isfinite(norm) && norm > threshold;
      }
      if (!completed) return false;
    }
    const float inverse = 1.0f / norm;
    for (std::size_t row = 0; row < q.rows(); ++row) {
      q(row, column) *= inverse;
    }
  }
  return true;
}

bool portable_symmetric_eigen(core::Matrix<float>& matrix,
                              std::vector<float>& eigenvalues) {
  if (matrix.rows() != matrix.columns()) return false;
  const std::size_t n = matrix.rows();
  core::Matrix<float> vectors(n, n);
  for (std::size_t column = 0; column < n; ++column) {
    vectors(column, column) = 1.0f;
  }
  const std::size_t maximum_iterations = std::max<std::size_t>(32, 64 * n * n);
  for (std::size_t iteration = 0; iteration < maximum_iterations; ++iteration) {
    std::size_t p = 0;
    std::size_t q = 0;
    float maximum = 0.0f;
    float diagonal_scale = 1.0f;
    for (std::size_t column = 0; column < n; ++column) {
      diagonal_scale = std::max(diagonal_scale, std::abs(matrix(column, column)));
      for (std::size_t row = 0; row < column; ++row) {
        const float candidate = std::abs(matrix(row, column));
        if (candidate > maximum) {
          maximum = candidate;
          p = row;
          q = column;
        }
      }
    }
    const float tolerance = std::numeric_limits<float>::epsilon() *
      static_cast<float>(std::max<std::size_t>(n, 1)) * diagonal_scale;
    if (maximum <= tolerance) {
      eigenvalues.resize(n);
      for (std::size_t index = 0; index < n; ++index) {
        eigenvalues[index] = matrix(index, index);
      }
      std::vector<std::size_t> order(n);
      std::iota(order.begin(), order.end(), 0);
      std::sort(order.begin(), order.end(), [&](std::size_t left,
                                                std::size_t right) {
        return eigenvalues[left] < eigenvalues[right];
      });
      core::Matrix<float> sorted_vectors(n, n);
      std::vector<float> sorted_values(n);
      for (std::size_t column = 0; column < n; ++column) {
        sorted_values[column] = eigenvalues[order[column]];
        for (std::size_t row = 0; row < n; ++row) {
          sorted_vectors(row, column) = vectors(row, order[column]);
        }
      }
      matrix = std::move(sorted_vectors);
      eigenvalues = std::move(sorted_values);
      return true;
    }

    const float app = matrix(p, p);
    const float aqq = matrix(q, q);
    const float apq = matrix(p, q);
    const float tau = (aqq - app) / (2.0f * apq);
    const float tangent = (tau >= 0.0f ? 1.0f : -1.0f) /
      (std::abs(tau) + std::sqrt(1.0f + tau * tau));
    const float cosine = 1.0f / std::sqrt(1.0f + tangent * tangent);
    const float sine = tangent * cosine;
    for (std::size_t index = 0; index < n; ++index) {
      if (index == p || index == q) continue;
      const float aip = matrix(index, p);
      const float aiq = matrix(index, q);
      matrix(index, p) = matrix(p, index) = cosine * aip - sine * aiq;
      matrix(index, q) = matrix(q, index) = sine * aip + cosine * aiq;
    }
    matrix(p, p) = app - tangent * apq;
    matrix(q, q) = aqq + tangent * apq;
    matrix(p, q) = matrix(q, p) = 0.0f;
    for (std::size_t row = 0; row < n; ++row) {
      const float vip = vectors(row, p);
      const float viq = vectors(row, q);
      vectors(row, p) = cosine * vip - sine * viq;
      vectors(row, q) = sine * vip + cosine * viq;
    }
  }
  return false;
}

bool portable_svd(core::ConstMatrixView<float> input,
                  bool left_only,
                  core::Matrix<float>& u,
                  std::vector<float>& singular_values,
                  core::Matrix<float>& vt) {
  const std::size_t rank = std::min(input.rows(), input.columns());
  if (rank == 0) {
    u.resize(input.rows(), 0);
    singular_values.clear();
    vt.resize(0, input.columns());
    return true;
  }
  CpuLinearAlgebraF32 backend;
  core::Matrix<float> gram;
  if (input.rows() >= input.columns()) {
    gram.resize(input.columns(), input.columns());
    backend.gemm(input, input, true, false, gram.view());
  } else {
    gram.resize(input.rows(), input.rows());
    backend.gemm(input, input, false, true, gram.view());
  }
  std::vector<float> values;
  if (!portable_symmetric_eigen(gram, values)) return false;
  singular_values.resize(rank);
  for (std::size_t index = 0; index < rank; ++index) {
    singular_values[index] = std::sqrt(std::max(values[rank - 1 - index], 0.0f));
  }
  const float tolerance = std::numeric_limits<float>::epsilon() *
    static_cast<float>(std::max(input.rows(), input.columns())) *
    std::max(singular_values.front(), 1.0f);

  if (input.rows() >= input.columns()) {
    core::Matrix<float> right(input.columns(), rank);
    for (std::size_t column = 0; column < rank; ++column) {
      const std::size_t source = rank - 1 - column;
      for (std::size_t row = 0; row < right.rows(); ++row) {
        right(row, column) = gram(row, source);
      }
    }
    u.resize(input.rows(), rank);
    backend.gemm(input, right.view(), false, false, u.view());
    for (std::size_t column = 0; column < rank; ++column) {
      if (singular_values[column] > tolerance) {
        const float inverse = 1.0f / singular_values[column];
        for (std::size_t row = 0; row < u.rows(); ++row) {
          u(row, column) *= inverse;
        }
      }
    }
    if (!left_only) {
      vt.resize(rank, input.columns());
      for (std::size_t column = 0; column < right.rows(); ++column) {
        for (std::size_t row = 0; row < rank; ++row) {
          vt(row, column) = right(column, row);
        }
      }
    }
  } else {
    u.resize(input.rows(), rank);
    for (std::size_t column = 0; column < rank; ++column) {
      const std::size_t source = rank - 1 - column;
      for (std::size_t row = 0; row < u.rows(); ++row) {
        u(row, column) = gram(row, source);
      }
    }
    if (!left_only) {
      vt.resize(rank, input.columns());
      backend.gemm(u.view(), input, true, false, vt.view());
      for (std::size_t row = 0; row < rank; ++row) {
        if (singular_values[row] > tolerance) {
          const float inverse = 1.0f / singular_values[row];
          for (std::size_t column = 0; column < vt.columns(); ++column) {
            vt(row, column) *= inverse;
          }
        }
      }
    }
  }
  if (left_only) vt.resize(0, input.columns());
  return true;
}
#endif

}  // namespace

bool CpuLinearAlgebraF32::qr_economy(core::ConstMatrixView<float> input,
                                     core::Matrix<float>& q) const {
#if defined(FASTPLS_USE_ACCELERATE) || defined(FASTPLS_USE_OPENBLAS)
  if (input.empty()) {
    q.resize(input.rows(), 0);
    return true;
  }
  const La_INT m = lapack_dimension(input.rows(), "QR");
  const La_INT n = lapack_dimension(input.columns(), "QR");
  const La_INT rank = std::min(m, n);
  const La_INT lda = std::max<La_INT>(1, m);
  core::Matrix<float> factor = contiguous_copy(input);
  std::vector<float> tau(static_cast<std::size_t>(rank));
  La_INT info = 0;
  La_INT lwork = -1;
  float query = 0.0f;
  F77_CALL(sgeqrf)(
    &m, &n, factor.data(), &lda, tau.data(), &query, &lwork, &info
  );
  if (info != 0) return false;
  lwork = workspace_size(query);
  std::vector<float> workspace(static_cast<std::size_t>(lwork));
  F77_CALL(sgeqrf)(
    &m, &n, factor.data(), &lda, tau.data(), workspace.data(), &lwork, &info
  );
  if (info != 0) return false;
  q.resize(static_cast<std::size_t>(m), static_cast<std::size_t>(rank));
  for (La_INT column = 0; column < rank; ++column) {
    std::copy_n(
      factor.data() + static_cast<std::size_t>(column) * factor.rows(),
      static_cast<std::size_t>(m),
      q.data() + static_cast<std::size_t>(column) * q.rows()
    );
  }
  lwork = -1;
  query = 0.0f;
  F77_CALL(sorgqr)(
    &m, &rank, &rank, q.data(), &lda, tau.data(), &query, &lwork, &info
  );
  if (info != 0) return false;
  lwork = workspace_size(query);
  workspace.resize(static_cast<std::size_t>(lwork));
  F77_CALL(sorgqr)(
    &m, &rank, &rank, q.data(), &lda, tau.data(), workspace.data(),
    &lwork, &info
  );
  return info == 0;
#else
  return portable_qr(input, q);
#endif
}

bool CpuLinearAlgebraF32::symmetric_eigen(
    core::Matrix<float>& matrix,
    std::vector<float>& eigenvalues) const {
  if (matrix.rows() != matrix.columns()) {
    throw std::invalid_argument(
      "fastPLS float32 symmetric eigendecomposition requires a square matrix"
    );
  }
#if defined(FASTPLS_USE_ACCELERATE) || defined(FASTPLS_USE_OPENBLAS)
  const La_INT n = lapack_dimension(matrix.rows(), "eigendecomposition");
  if (n == 0) {
    eigenvalues.clear();
    return true;
  }
  const La_INT lda = std::max<La_INT>(1, n);
  eigenvalues.resize(static_cast<std::size_t>(n));
  const char vectors = 'V';
  const char lower = 'L';
  La_INT info = 0;
  La_INT lwork = -1;
  La_INT liwork = -1;
  float query = 0.0f;
  La_INT integer_query = 0;
  F77_CALL(ssyevd)(
    &vectors, &lower, &n, matrix.data(), &lda, eigenvalues.data(),
    &query, &lwork, &integer_query, &liwork, &info FCONE FCONE
  );
  if (info != 0) return false;
  lwork = workspace_size(query);
  liwork = std::max<La_INT>(1, integer_query);
  std::vector<float> workspace(static_cast<std::size_t>(lwork));
  std::vector<La_INT> integer_workspace(static_cast<std::size_t>(liwork));
  F77_CALL(ssyevd)(
    &vectors, &lower, &n, matrix.data(), &lda, eigenvalues.data(),
    workspace.data(), &lwork, integer_workspace.data(), &liwork, &info
    FCONE FCONE
  );
  return info == 0;
#else
  return portable_symmetric_eigen(matrix, eigenvalues);
#endif
}

bool CpuLinearAlgebraF32::svd_economy(
    core::ConstMatrixView<float> input,
    bool left_only,
    core::Matrix<float>& u,
    std::vector<float>& singular_values,
    core::Matrix<float>& vt) const {
#if defined(FASTPLS_USE_ACCELERATE) || defined(FASTPLS_USE_OPENBLAS)
  const La_INT m = lapack_dimension(input.rows(), "SVD");
  const La_INT n = lapack_dimension(input.columns(), "SVD");
  const La_INT rank = std::min(m, n);
  if (rank == 0) {
    u.resize(input.rows(), 0);
    singular_values.clear();
    vt.resize(0, input.columns());
    return true;
  }
  const La_INT lda = std::max<La_INT>(1, m);
  const La_INT ldu = std::max<La_INT>(1, m);
  core::Matrix<float> factor = contiguous_copy(input);
  u.resize(static_cast<std::size_t>(m), static_cast<std::size_t>(rank));
  singular_values.resize(static_cast<std::size_t>(rank));
  La_INT info = 0;
  La_INT lwork = -1;
  float query = 0.0f;
  if (left_only) {
    const char compute_left = 'S';
    const char no_right = 'N';
    const La_INT ldvt = 1;
    float right_dummy = 0.0f;
    F77_CALL(sgesvd)(
      &compute_left, &no_right, &m, &n, factor.data(), &lda,
      singular_values.data(), u.data(), &ldu, &right_dummy, &ldvt,
      &query, &lwork, &info FCONE FCONE
    );
    if (info != 0) return false;
    lwork = workspace_size(query);
    std::vector<float> workspace(static_cast<std::size_t>(lwork));
    factor = contiguous_copy(input);
    F77_CALL(sgesvd)(
      &compute_left, &no_right, &m, &n, factor.data(), &lda,
      singular_values.data(), u.data(), &ldu, &right_dummy, &ldvt,
      workspace.data(), &lwork, &info FCONE FCONE
    );
    vt.resize(0, input.columns());
    return info == 0;
  }
  const char economy = 'S';
  const La_INT ldvt = std::max<La_INT>(1, rank);
  vt.resize(static_cast<std::size_t>(rank), static_cast<std::size_t>(n));
  std::vector<La_INT> integer_workspace(
    static_cast<std::size_t>(8 * rank)
  );
  F77_CALL(sgesdd)(
    &economy, &m, &n, factor.data(), &lda, singular_values.data(),
    u.data(), &ldu, vt.data(), &ldvt, &query, &lwork,
    integer_workspace.data(), &info FCONE
  );
  if (info != 0) return false;
  lwork = workspace_size(query);
  std::vector<float> workspace(static_cast<std::size_t>(lwork));
  factor = contiguous_copy(input);
  F77_CALL(sgesdd)(
    &economy, &m, &n, factor.data(), &lda, singular_values.data(),
    u.data(), &ldu, vt.data(), &ldvt, workspace.data(), &lwork,
    integer_workspace.data(), &info FCONE
  );
  return info == 0;
#else
  return portable_svd(input, left_only, u, singular_values, vt);
#endif
}

}  // namespace runtime
}  // namespace fastpls
