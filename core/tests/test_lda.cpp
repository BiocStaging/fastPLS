// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Stefano Cacciatore
#include <fastpls/native/lda.hpp>
#include <type_traits>

template<class T>
auto solve(const arma::Mat<T>& covariance, const arma::Mat<T>& means) {
  if constexpr (std::is_same<T, float>::value) {
    return fastpls::native::lda_cholesky_solve_float(covariance, means);
  } else {
    return fastpls::native::lda_cholesky_solve(covariance, means);
  }
}

template<class T>
void check() {
  for (int width : {3, 7, 3}) {
    arma::Mat<T> means(2, width, arma::fill::ones);
    means.row(1) *= T(2);
    for (int condition : {0, 1, 2, 3}) {
      arma::Mat<T> covariance(width, width, arma::fill::eye);
      if (condition == 1) covariance.ones();
      if (condition == 2) covariance(0, 0) = T(-1e-4);
      if (condition == 3) covariance.zeros();
      const auto result = solve(covariance, means);
      T scale = arma::trace(covariance) / T(width);
      if (scale <= T(0)) scale = T(1);
      if (result.lambda != result.relative_ridge * scale ||
          (condition == 0 && result.relative_ridge != T(1e-8)) ||
          (condition == 2 && result.relative_ridge <= T(1e-4))) {
        throw std::runtime_error("LDA regularization sequence changed");
      }
      covariance.diag() += result.lambda;
      arma::Mat<T> weights = result.linear.t();
      const T residual = arma::norm(covariance * weights - means.t(), "fro") /
        std::max(T(1), arma::norm(covariance, "fro") * arma::norm(weights, "fro"));
      const T tolerance = std::is_same<T, float>::value ? T(2e-6) : T(1e-12);
      if (!weights.is_finite() || residual > tolerance) {
        throw std::runtime_error("LDA triangular-solve residual exceeded tolerance");
      }
    }
    arma::Mat<T> indefinite(width, width, arma::fill::eye);
    indefinite *= T(-1);
    bool failed = false;
    try { solve(indefinite, means); }
    catch (const std::runtime_error&) { failed = true; }
    if (!failed) throw std::runtime_error("Invalid LDA covariance did not fail");
  }
}

int main() {
  check<float>();
  check<double>();
}
