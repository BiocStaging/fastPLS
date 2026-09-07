// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Stefano Cacciatore
#ifndef FASTPLS_NATIVE_MODELS_HPP
#define FASTPLS_NATIVE_MODELS_HPP
#include <fastpls/native/simpls.hpp>
#include <fastpls/native/opls.hpp>
#include <fastpls/native/kernels.hpp>

namespace fastpls { namespace native {

template<class T>
struct OplsModel {
  OplsFilter<T> filter;
  SimplsModel<T> inner;
};

template<class T>
OplsModel<T> fit_opls(arma::Mat<T> X, arma::Mat<T> Y, arma::ivec components,
                      int north = 1, SimplsOptions options = {}) {
  if (X.n_rows < 2 || X.n_cols == 0 || Y.n_cols == 0 || X.n_rows != Y.n_rows ||
      components.is_empty() || components.min() < 1 || north < 0 ||
      north >= static_cast<int>(std::min(X.n_rows, X.n_cols)) ||
      components.max() > static_cast<int>(std::min(
        X.n_rows - 1, X.n_cols - static_cast<arma::uword>(north)))) {
    throw std::invalid_argument("Invalid OPLS training dimensions or component bound");
  }
  auto filter = fit_opls_filter(std::move(X), Y, north, options.scaling, options.svd);
  options.scaling = 3;
  auto inner = fit_simpls(filter.X, Y, std::move(components), options);
  filter.X.reset();
  return {std::move(filter), std::move(inner)};
}

template<class T>
arma::Mat<T> predict_opls(const OplsModel<T>& model, arma::Mat<T> X, int components) {
  auto filtered = apply_opls_filter(std::move(X), model.filter.x_mean,
                                    model.filter.x_scale, model.filter.W, model.filter.P);
  return predict_simpls(model.inner, std::move(filtered), components);
}

struct KernelPlsOptions {
  int kernel = 1;
  double gamma = 1.0;
  int degree = 2;
  double coef0 = 1.0;
  SimplsOptions pls;
};

template<class T>
struct KernelPlsModel {
  SimplsModel<T> inner;
  arma::Mat<T> reference;
  arma::Row<T> x_mean, x_scale, kernel_mean;
  T kernel_grand = T(0), gamma = T(1), coef0 = T(1);
  int kernel = 1, degree = 2;
};

template<class T>
KernelPlsModel<T> fit_kernelpls(arma::Mat<T> X, const arma::Mat<T>& Y,
                                arma::ivec components, KernelPlsOptions options = {}) {
  if (X.n_rows < 2 || X.n_cols == 0 || Y.n_cols == 0 || X.n_rows != Y.n_rows ||
      options.kernel < 1 || options.kernel > 3 || components.is_empty() ||
      components.min() < 1 || components.max() > static_cast<int>(
        options.kernel == 1 ? std::min(X.n_rows - 1, X.n_cols) : X.n_rows - 1)) {
    throw std::invalid_argument("Invalid kernel PLS training dimensions or kernel id");
  }
  KernelPlsModel<T> model;
  model.kernel = options.kernel;
  model.degree = options.degree;
  model.gamma = static_cast<T>(options.gamma);
  model.coef0 = static_cast<T>(options.coef0);
  if (options.kernel == 1) {
    model.inner = fit_simpls(X, Y, std::move(components), options.pls);
    return model;
  }
  if (!std::isfinite(model.gamma) || model.gamma <= T(0) ||
      !std::isfinite(model.coef0) || model.degree < 1) {
    throw std::invalid_argument("Invalid nonlinear kernel parameters");
  }
  model.x_mean.zeros(X.n_cols);
  model.x_scale.ones(X.n_cols);
  if (options.pls.scaling < 3) {
    model.x_mean = arma::mean(X, 0);
    X.each_row() -= model.x_mean;
  }
  if (options.pls.scaling == 2) {
    model.x_scale = column_standard_deviation(X);
    X.each_row() /= model.x_scale;
  }
  auto centered = center_kernel_train(kernel_matrix(
    X, X, model.kernel, model.gamma, model.degree, model.coef0));
  model.kernel_mean = std::move(centered.column_means);
  model.kernel_grand = centered.grand_mean;
  options.pls.scaling = 3;
  model.inner = fit_simpls(centered.K, Y, std::move(components), options.pls);
  model.reference = std::move(X);
  return model;
}

template<class T>
arma::Mat<T> predict_kernelpls(const KernelPlsModel<T>& model, arma::Mat<T> X,
                               int components) {
  if (model.kernel == 1) return predict_simpls(model.inner, std::move(X), components);
  if (X.n_cols != model.reference.n_cols || X.n_cols != model.x_mean.n_elem ||
      X.n_cols != model.x_scale.n_elem) {
    throw std::invalid_argument("Kernel PLS prediction dimensions differ from training");
  }
  X.each_row() -= model.x_mean;
  X.each_row() /= model.x_scale;
  auto kernel = kernel_matrix(X, model.reference, model.kernel,
                               model.gamma, model.degree, model.coef0);
  auto centered = center_kernel_test(std::move(kernel), model.kernel_mean, model.kernel_grand);
  return predict_simpls(model.inner, std::move(centered), components);
}

} }
#endif
