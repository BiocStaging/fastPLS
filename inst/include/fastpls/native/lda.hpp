// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Stefano Cacciatore
#ifndef FASTPLS_NATIVE_LDA_HPP
#define FASTPLS_NATIVE_LDA_HPP
#include <armadillo>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace fastpls { namespace native {

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
