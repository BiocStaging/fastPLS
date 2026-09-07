// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Stefano Cacciatore
#include <fastpls/native/plssvd.hpp>
#include <iostream>
#include <type_traits>

void require(bool value, const char* text) {
  if (!value) throw std::runtime_error(text);
}

template<typename T>
void check() {
  const T tolerance = std::is_same<T, float>::value ? T(2e-4) : T(1e-10);
  std::mt19937 random(200);
  std::normal_distribution<T> normal(T(0), T(1));
  arma::Mat<T> X(140, 24), beta(24, 6), noise(140, 6);
  for (auto& x : X) x = normal(random);
  for (auto& x : beta) x = normal(random);
  for (auto& x : noise) x = normal(random) * T(0.01);
  arma::Mat<T> Y = X * beta + noise;
  const auto original = X;
  for (int scaling : {1, 2, 3}) for (bool cached : {false, true}) {
    fastpls::native::PlssvdOptions options;
    options.scaling = scaling;
    options.fitted = true;
    options.store_coefficients = true;
    options.cache_score_gram = cached;
    auto fit = fastpls::native::fit_plssvd(X, Y, arma::ivec{1, 3, 6}, options);
    require(arma::approx_equal(X, original, "absdiff", T(0)), "X was mutated");
    arma::Mat<T> scaled = X;
    scaled.each_row() -= fit.x_mean;
    scaled.each_row() /= fit.x_scale;
    for (arma::uword prefix = 0; prefix < fit.components.n_elem; ++prefix) {
      auto predicted = fastpls::native::predict_plssvd(fit, X, prefix);
      arma::Mat<T> dense = scaled * fit.coefficients.slice(prefix);
      dense.each_row() += fit.y_mean;
      T divisor = std::max(T(1), arma::norm(dense, "fro"));
      require(arma::norm(dense - predicted, "fro") / divisor < tolerance,
              "dense and latent predictions disagree");
      require(arma::norm(fit.fitted.slice(prefix) - predicted, "fro") / divisor < tolerance,
              "training and prediction paths disagree");
    }
    options.fitted = options.store_coefficients = false;
    auto compact = fastpls::native::fit_plssvd(X, Y, arma::ivec{1, 3, 6}, options);
    require(compact.coefficients.is_empty() && compact.fitted.is_empty(),
            "unrequested dense outputs were retained");
    require(arma::approx_equal(fit.R, compact.R, "absdiff", T(0)),
            "output policy changed PLS-SVD fit");
  }
  bool rejected = false;
  try { fastpls::native::fit_plssvd(X, arma::Mat<T>(2, 3), arma::ivec{1}); }
  catch (const std::invalid_argument&) { rejected = true; }
  require(rejected, "incompatible training dimensions were accepted");
  std::cout << (std::is_same<T, float>::value ? "float32" : "float64") << " PLS-SVD OK\n";
}

int main() { check<float>(); check<double>(); }
