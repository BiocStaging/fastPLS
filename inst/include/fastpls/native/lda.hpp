// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Stefano Cacciatore
#ifndef FASTPLS_NATIVE_LDA_HPP
#define FASTPLS_NATIVE_LDA_HPP
#include <armadillo>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace fastpls { namespace native {

struct LDAFloatCholeskyResult {
  arma::fmat linear;
  float lambda = 0.0f;
  float relative_ridge = 0.0f;
};

struct LDAFloatCPUWorkspace {
  arma::fmat covariance;
  arma::fmat lower;
  arma::fmat solution;

  void prepare(arma::uword n, arma::uword rhs_cols) {
    covariance.set_size(n, n);
    lower.zeros(n, n);
    solution.set_size(n, rhs_cols);
  }
};

inline thread_local LDAFloatCPUWorkspace g_lda_float_cpu_workspace;

inline bool lda_cholesky_solve_float_once(const arma::fmat& pooled,
                                   const arma::fmat& rhs,
                                   float lambda,
                                   LDAFloatCPUWorkspace& workspace) {
  const arma::uword n = pooled.n_rows;
  workspace.prepare(n, rhs.n_cols);
  workspace.covariance = pooled;
  workspace.covariance.diag() += lambda;
  arma::fmat& lower = workspace.lower;
  for (arma::uword row = 0; row < n; ++row) {
    for (arma::uword col = 0; col <= row; ++col) {
      float value = workspace.covariance(row, col);
      for (arma::uword inner = 0; inner < col; ++inner) {
        value -= lower(row, inner) * lower(col, inner);
      }
      if (row == col) {
        if (!std::isfinite(value) || value <= 0.0f) {
          return false;
        }
        lower(row, col) = std::sqrt(value);
      } else {
        const float diagonal = lower(col, col);
        if (!std::isfinite(diagonal) || diagonal <= 0.0f) {
          return false;
        }
        lower(row, col) = value / diagonal;
      }
    }
  }

  workspace.solution = rhs;
  arma::fmat& solution = workspace.solution;
  for (arma::uword column = 0; column < rhs.n_cols; ++column) {
    for (arma::uword row = 0; row < n; ++row) {
      float value = solution(row, column);
      for (arma::uword inner = 0; inner < row; ++inner) {
        value -= lower(row, inner) * solution(inner, column);
      }
      solution(row, column) = value / lower(row, row);
    }
    for (arma::sword row = static_cast<arma::sword>(n) - 1; row >= 0; --row) {
      float value = solution(static_cast<arma::uword>(row), column);
      for (arma::uword inner = static_cast<arma::uword>(row) + 1; inner < n; ++inner) {
        value -= lower(inner, static_cast<arma::uword>(row)) * solution(inner, column);
      }
      solution(static_cast<arma::uword>(row), column) =
        value / lower(static_cast<arma::uword>(row), static_cast<arma::uword>(row));
    }
  }
  return solution.is_finite();
}

inline LDAFloatCholeskyResult lda_cholesky_solve_float(const arma::fmat& pooled,
                                                const arma::fmat& means) {
  const arma::uword k = pooled.n_rows;
  float scale = arma::trace(pooled) / static_cast<float>(std::max<arma::uword>(1, k));
  if (!std::isfinite(scale) || scale <= 0.0f) {
    scale = 1.0f;
  }
  const arma::fmat rhs = means.t();
  constexpr float ridge_grid[] = {
    1e-8f, 1e-6f, 1e-5f, 1e-4f, 1e-3f, 1e-2f
  };
  for (float rho : ridge_grid) {
    const float lambda = rho * scale;
    if (lda_cholesky_solve_float_once(
          pooled, rhs, lambda, g_lda_float_cpu_workspace
        )) {
      LDAFloatCholeskyResult out;
      out.linear = g_lda_float_cpu_workspace.solution.t();
      out.lambda = lambda;
      out.relative_ridge = rho;
      return out;
    }
  }
  throw std::runtime_error(
    "float32 PLS-LDA Cholesky factorization failed for every deterministic regularization level"
  );
}

constexpr double kLdaRelativeRidge[] = {
  1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2
};

struct LDACholeskyResult {
  arma::mat linear;
  double lambda = 0.0;
  double relative_ridge = 0.0;
};

inline LDACholeskyResult lda_cholesky_solve(const arma::mat& pooled,
                                     const arma::mat& means) {
  const arma::uword k = pooled.n_rows;
  double scale = arma::trace(pooled) /
    static_cast<double>(std::max<arma::uword>(1, k));
  if (!std::isfinite(scale) || scale <= 0.0) {
    scale = 1.0;
  }

  const arma::mat rhs = means.t();
  for (double rho : kLdaRelativeRidge) {
    arma::mat covariance = pooled;
    const double lambda = rho * scale;
    covariance.diag() += lambda;

    arma::mat lower;
    if (!arma::chol(lower, covariance, "lower")) {
      continue;
    }
    arma::mat intermediate;
    arma::mat solution;
    const bool forward_ok = arma::solve(
      intermediate, arma::trimatl(lower), rhs, arma::solve_opts::fast
    );
    const bool backward_ok = forward_ok && arma::solve(
      solution, arma::trimatu(lower.t()), intermediate, arma::solve_opts::fast
    );
    if (backward_ok && solution.is_finite()) {
      LDACholeskyResult out;
      out.linear = solution.t();
      out.lambda = lambda;
      out.relative_ridge = rho;
      return out;
    }
  }
  throw std::runtime_error(
    "PLS-LDA Cholesky factorization failed for every deterministic regularization level"
  );
}

} }
#endif
