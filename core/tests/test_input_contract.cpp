// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Stefano Cacciatore
#include <fastpls/native/plssvd.hpp>

template<class Function>
void invalid(Function&& function) {
  try {
    function();
  } catch (const std::invalid_argument&) {
    return;
  }
  throw std::runtime_error("Invalid native input did not raise invalid_argument");
}

template<class T>
void check() {
  arma::Mat<T> x(20, 4, arma::fill::ones), y(19, 2, arma::fill::ones);
  invalid([&] { fastpls::native::fit_simpls(x, y, arma::ivec{1}); });
  invalid([&] { fastpls::native::fit_plssvd(x, y, arma::ivec{1}); });
  y.set_size(20, 0);
  invalid([&] { fastpls::native::fit_simpls(x, y, arma::ivec{1}); });
  invalid([&] { fastpls::native::fit_plssvd(x, y, arma::ivec{1}); });
  x.set_size(1, 4);
  y.set_size(1, 2);
  invalid([&] { fastpls::native::fit_simpls(x, y, arma::ivec{1}); });
  invalid([&] { fastpls::native::fit_plssvd(x, y, arma::ivec{1}); });
  x.set_size(20, 4);
  y.set_size(20, 2);
  std::mt19937 rng(913);
  std::normal_distribution<T> normal(T(0), T(1));
  for (auto& value : x) value = normal(rng);
  for (auto& value : y) value = normal(rng);
  auto simpls = fastpls::native::fit_simpls(x, y, arma::ivec{1, 2});
  auto plssvd = fastpls::native::fit_plssvd(x, y, arma::ivec{1, 2});
  invalid([&] { fastpls::native::predict_simpls(simpls, x, 3); });
  invalid([&] { fastpls::native::predict_plssvd(plssvd, x, 2); });
  simpls.Q.reset();
  invalid([&] { fastpls::native::predict_simpls(simpls, x, 1); });
  plssvd.prediction_weights.reset();
  invalid([&] { fastpls::native::predict_plssvd(plssvd, x, 0); });
  simpls = fastpls::native::fit_simpls(x, y, arma::ivec{1});
  simpls.x_scale.reset();
  invalid([&] { fastpls::native::predict_simpls(simpls, x, 1); });
}

int main() {
  check<float>();
  check<double>();
}
