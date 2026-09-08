// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Stefano Cacciatore
#include "core_cpu_backend.h"

#include <R_ext/Lapack.h>
#include <R_ext/RS.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace fastpls {
namespace runtime {
namespace {

La_INT lapack_dimension(std::size_t value, const char* operation) {
  if (value > static_cast<std::size_t>(std::numeric_limits<La_INT>::max())) {
    throw std::overflow_error(
      std::string("fastPLS ") + operation + " dimension exceeds LAPACK range"
    );
  }
  return static_cast<La_INT>(value);
}

core::Matrix<double> contiguous_copy(core::ConstMatrixView<double> input) {
  core::Matrix<double> output(input.rows(), input.columns());
  for (std::size_t column = 0; column < input.columns(); ++column) {
    std::copy_n(
      input.data() + column * input.leading_dimension(), input.rows(),
      output.data() + column * output.rows()
    );
  }
  return output;
}

La_INT workspace_size(double query) {
  if (!std::isfinite(query) || query < 1.0 ||
      query > static_cast<double>(std::numeric_limits<La_INT>::max())) {
    throw std::runtime_error("fastPLS received an invalid LAPACK workspace size");
  }
  return static_cast<La_INT>(query);
}

}  // namespace

bool CpuLinearAlgebraF64::qr_economy(core::ConstMatrixView<double> input,
                                     core::Matrix<double>& q) const {
  if (input.empty()) {
    q.resize(input.rows(), 0);
    return true;
  }
  const La_INT m = lapack_dimension(input.rows(), "QR");
  const La_INT n = lapack_dimension(input.columns(), "QR");
  const La_INT rank = std::min(m, n);
  const La_INT lda = std::max<La_INT>(1, m);
  core::Matrix<double> factor = contiguous_copy(input);
  std::vector<double> tau(static_cast<std::size_t>(rank));
  La_INT info = 0;
  La_INT lwork = -1;
  double query = 0.0;
  F77_CALL(dgeqrf)(
    &m, &n, factor.data(), &lda, tau.data(), &query, &lwork, &info
  );
  if (info != 0) return false;
  lwork = workspace_size(query);
  std::vector<double> workspace(static_cast<std::size_t>(lwork));
  F77_CALL(dgeqrf)(
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
  query = 0.0;
  F77_CALL(dorgqr)(
    &m, &rank, &rank, q.data(), &lda, tau.data(), &query, &lwork, &info
  );
  if (info != 0) return false;
  lwork = workspace_size(query);
  workspace.resize(static_cast<std::size_t>(lwork));
  F77_CALL(dorgqr)(
    &m, &rank, &rank, q.data(), &lda, tau.data(), workspace.data(),
    &lwork, &info
  );
  return info == 0;
}

bool CpuLinearAlgebraF64::symmetric_eigen(
    core::Matrix<double>& matrix,
    std::vector<double>& eigenvalues) const {
  if (matrix.rows() != matrix.columns()) {
    throw std::invalid_argument(
      "fastPLS symmetric eigendecomposition requires a square matrix"
    );
  }
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
  double query = 0.0;
  La_INT integer_query = 0;
  F77_CALL(dsyevd)(
    &vectors, &lower, &n, matrix.data(), &lda, eigenvalues.data(),
    &query, &lwork, &integer_query, &liwork, &info FCONE FCONE
  );
  if (info != 0) return false;
  lwork = workspace_size(query);
  liwork = std::max<La_INT>(1, integer_query);
  std::vector<double> workspace(static_cast<std::size_t>(lwork));
  std::vector<La_INT> integer_workspace(static_cast<std::size_t>(liwork));
  F77_CALL(dsyevd)(
    &vectors, &lower, &n, matrix.data(), &lda, eigenvalues.data(),
    workspace.data(), &lwork, integer_workspace.data(), &liwork, &info
    FCONE FCONE
  );
  return info == 0;
}

bool CpuLinearAlgebraF64::svd_economy(
    core::ConstMatrixView<double> input,
    bool left_only,
    core::Matrix<double>& u,
    std::vector<double>& singular_values,
    core::Matrix<double>& vt) const {
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
  core::Matrix<double> factor = contiguous_copy(input);
  u.resize(static_cast<std::size_t>(m), static_cast<std::size_t>(rank));
  singular_values.resize(static_cast<std::size_t>(rank));
  La_INT info = 0;
  La_INT lwork = -1;
  double query = 0.0;

  if (left_only) {
    const char compute_left = 'S';
    const char no_right = 'N';
    const La_INT ldvt = 1;
    double right_dummy = 0.0;
    F77_CALL(dgesvd)(
      &compute_left, &no_right, &m, &n, factor.data(), &lda,
      singular_values.data(), u.data(), &ldu, &right_dummy, &ldvt,
      &query, &lwork, &info FCONE FCONE
    );
    if (info != 0) return false;
    lwork = workspace_size(query);
    std::vector<double> workspace(static_cast<std::size_t>(lwork));
    factor = contiguous_copy(input);
    F77_CALL(dgesvd)(
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
  F77_CALL(dgesdd)(
    &economy, &m, &n, factor.data(), &lda, singular_values.data(),
    u.data(), &ldu, vt.data(), &ldvt, &query, &lwork,
    integer_workspace.data(), &info FCONE
  );
  if (info != 0) return false;
  lwork = workspace_size(query);
  std::vector<double> workspace(static_cast<std::size_t>(lwork));
  factor = contiguous_copy(input);
  F77_CALL(dgesdd)(
    &economy, &m, &n, factor.data(), &lda, singular_values.data(),
    u.data(), &ldu, vt.data(), &ldvt, workspace.data(), &lwork,
    integer_workspace.data(), &info FCONE
  );
  return info == 0;
}

}  // namespace runtime
}  // namespace fastpls
