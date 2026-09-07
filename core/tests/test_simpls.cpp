// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Stefano Cacciatore
#include <fastpls/native/simpls.hpp>
#include <iostream>
#include <type_traits>

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

template<typename T>
void check() {
  const T tolerance = std::is_same<T, float>::value ? T(2e-4) : T(1e-10);
  arma::Mat<T> X(240, 24), coefficients(24, 6), noise(240, 6);
  std::mt19937 rng(901);
  std::normal_distribution<T> normal(T(0), T(1));
  for (auto& x : X) x = normal(rng);
  for (auto& x : coefficients) x = normal(rng);
  for (auto& x : noise) x = normal(rng) * T(0.01);
  arma::Mat<T> Y = X * coefficients + noise;
  const arma::Mat<T> original = X;
  for (int scaling : {1, 2, 3}) {
    for (bool cached : {false, true}) {
      fastpls::native::SimplsOptions controls;
      controls.scaling = scaling;
      controls.fitted = true;
      controls.store_scores = true;
      controls.store_coefficients = true;
      controls.crossprod_min_components = 1;
      controls.cache_crossprod = cached;
      arma::ivec counts{1, 3, 6};
      auto model = fastpls::native::fit_simpls(X, Y, counts, controls);
      require(model.completed_components == 6, "component path ended early");
      require(arma::approx_equal(X, original, "absdiff", T(0)), "input was mutated");
      arma::Mat<T> standardized = X;
      standardized.each_row() -= model.x_mean;
      standardized.each_row() /= model.x_scale;
      for (arma::uword i = 0; i < counts.n_elem; ++i) {
        auto predicted = fastpls::native::predict_simpls(model, X, counts(i));
        arma::Mat<T> dense = standardized * model.coefficients.slice(i);
        dense.each_row() += model.y_mean;
        T scale = std::max(T(1), arma::norm(dense, "fro"));
        require(arma::norm(predicted - dense, "fro") / scale < tolerance,
                "compact and dense prediction differ");
        require(arma::norm(predicted - model.fitted.slice(i), "fro") / scale < tolerance,
                "fitted and predicted values differ");
      }
      controls.fitted = false;
      controls.store_coefficients = false;
      auto compact = fastpls::native::fit_simpls(X, Y, counts, controls);
      require(compact.coefficients.n_elem == 0 && compact.fitted.n_elem == 0,
              "unrequested response workspaces were retained");
      require(arma::approx_equal(model.R, compact.R, "absdiff", T(0)),
              "output policy changed the fitted directions");
    }
  }
  arma::Mat<T> dummy(X.n_rows, 3, arma::fill::zeros);
  for (arma::uword i = 0; i < X.n_rows; ++i) dummy(i, i % 3) = T(1);
  auto classification = fastpls::native::fit_simpls(X, dummy, arma::ivec{1, 3, 6});
  require(classification.completed_components == 6 && classification.R.is_finite(),
          "dummy response path failed");
  std::cout << (std::is_same<T, float>::value ? "float32" : "float64")
            << " SIMPLS extraction OK\n";
}

int main() {
  check<float>();
  check<double>();
}
