// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Stefano Cacciatore
#include <fastpls/native/opls.hpp>
#include <type_traits>

template<class T>
void check() {
  arma::Mat<T> X(80, 12), Y(80, 4);
  std::mt19937 rng(981);
  std::normal_distribution<T> normal(T(0), T(1));
  for (auto& x : X) x = normal(rng) + T(2);
  for (auto& y : Y) y = normal(rng);
  Y += X.cols(0, 3);
  const auto original = X;
  const T tolerance = std::is_same<T, float>::value ? T(2e-5) : T(1e-12);
  for (int scaling : {1, 2, 3}) for (int north : {0, 1, 3}) {
    auto model = fastpls::native::fit_opls_filter(X, Y, north, scaling);
    auto predicted = fastpls::native::apply_opls_filter(
      X, model.x_mean, model.x_scale, model.W, model.P);
    if (!arma::approx_equal(X, original, "absdiff", T(0)) ||
        model.completed != north || !model.X.is_finite() ||
        arma::norm(predicted - model.X, "fro") > tolerance *
          std::max(T(1), arma::norm(model.X, "fro"))) {
      throw std::runtime_error("Native OPLS filtering contract failed");
    }
    for (arma::uword a = 0; a < model.W.n_cols; ++a) {
      if (std::abs(arma::norm(model.W.col(a), 2) - T(1)) > tolerance) {
        throw std::runtime_error("OPLS filter direction is not normalized");
      }
    }
  }
}

int main() {
  check<float>();
  check<double>();
}
