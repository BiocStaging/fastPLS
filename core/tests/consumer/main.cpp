// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Stefano Cacciatore
#include <fastpls/native/plssvd.hpp>
#include <type_traits>

template<class T>
void check() {
  arma::Mat<T> x(40, 5), y(40, 2);
  std::mt19937 rng(23);
  std::normal_distribution<T> normal(T(0), T(1));
  for (auto& value : x) value = normal(rng);
  y.col(0) = x.col(0) + T(2) * x.col(1);
  y.col(1) = x.col(2) - x.col(3);
  fastpls::native::SimplsOptions controls;
  controls.fitted = true;
  auto model = fastpls::native::fit_simpls(x, y, arma::ivec{1, 2}, controls);
  auto predicted = fastpls::native::predict_simpls(model, x, 2);
  const T tolerance = std::is_same<T, float>::value ? T(1e-4) : T(1e-10);
  if (!predicted.is_finite() || model.completed_components != 2 ||
      arma::norm(predicted - model.fitted.slice(1), "fro") > tolerance) {
    throw std::runtime_error("Installed consumer predictions differ from fitted values");
  }
  auto same = [&](const arma::Mat<T>& actual, const arma::Mat<T>& expected) {
    if (!actual.is_finite() || arma::norm(actual - expected, "fro") >
        tolerance * std::max(T(1), arma::norm(expected, "fro"))) {
      throw std::runtime_error("Installed composed model predictions differ from fitted values");
    }
  };
  fastpls::native::PlssvdOptions svd_controls;
  svd_controls.fitted = true;
  auto svd_model = fastpls::native::fit_plssvd(x, y, arma::ivec{1, 2}, svd_controls);
  same(fastpls::native::predict_plssvd(svd_model, x, 1), svd_model.fitted.slice(1));
}

int main() {
  check<float>();
  check<double>();
}
