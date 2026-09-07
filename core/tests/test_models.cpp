// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Stefano Cacciatore
#include <fastpls/native/models.hpp>

template<class T>
void check() {
  arma::Mat<T> x(45, 8), y(45, 3), test(7, 8);
  std::mt19937 rng(653);
  std::normal_distribution<T> normal(T(0), T(1));
  for (auto& value : x) value = normal(rng) + T(2);
  for (auto& value : y) value = normal(rng);
  for (auto& value : test) value = normal(rng) + T(2);
  y += x.cols(0, 2);
  const T tolerance = std::is_same<T, float>::value ? T(1e-4) : T(1e-11);
  auto same = [&](const arma::Mat<T>& a, const arma::Mat<T>& b) {
    if (!a.is_finite() || arma::norm(a - b, "fro") >
          tolerance * std::max(T(1), arma::norm(b, "fro"))) {
      throw std::runtime_error("Native model composition changed predictions");
    }
  };
  for (int scaling : {1, 2, 3}) {
    fastpls::native::SimplsOptions options;
    options.scaling = scaling;
    options.fitted = true;
    for (int north : {0, 1, 2}) {
      auto model = fastpls::native::fit_opls(x, y, arma::ivec{1, 3}, north, options);
      if (model.filter.X.n_elem != 0) {
        throw std::runtime_error("OPLS retained its unused filtered training matrix");
      }
      same(fastpls::native::predict_opls(model, x, 3), model.inner.fitted.slice(1));
      auto filtered = fastpls::native::apply_opls_filter(test, model.filter.x_mean,
        model.filter.x_scale, model.filter.W, model.filter.P);
      same(fastpls::native::predict_opls(model, test, 3),
           fastpls::native::predict_simpls(model.inner, filtered, 3));
    }
    for (int kernel : {1, 2, 3}) {
      fastpls::native::KernelPlsOptions controls;
      controls.pls = options;
      controls.kernel = kernel;
      controls.gamma = 0.2;
      auto model = fastpls::native::fit_kernelpls(x, y, arma::ivec{1, 3}, controls);
      same(fastpls::native::predict_kernelpls(model, x, 3), model.inner.fitted.slice(1));
      if (kernel == 1) {
        if (model.reference.n_elem) throw std::runtime_error("Linear kernel retained training data");
        auto direct = fastpls::native::fit_simpls(x, y, arma::ivec{1, 3}, options);
        same(fastpls::native::predict_kernelpls(model, test, 3),
             fastpls::native::predict_simpls(direct, test, 3));
      } else {
        arma::Mat<T> standardized = test;
        standardized.each_row() -= model.x_mean;
        standardized.each_row() /= model.x_scale;
        auto matrix = fastpls::native::kernel_matrix(standardized, model.reference,
          model.kernel, model.gamma, model.degree, model.coef0);
        auto centered = fastpls::native::center_kernel_test(matrix,
          model.kernel_mean, model.kernel_grand);
        same(fastpls::native::predict_kernelpls(model, test, 3),
             fastpls::native::predict_simpls(model.inner, centered, 3));
      }
    }
  }
}

int main() {
  check<float>();
  check<double>();
}
