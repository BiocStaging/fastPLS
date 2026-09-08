// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Stefano Cacciatore
#ifndef FASTPLS_NATIVE_KERNELS_HPP
#define FASTPLS_NATIVE_KERNELS_HPP
#include <fastpls/core/kernels.hpp>
#include <armadillo>
#include <stdexcept>

namespace fastpls { namespace native {

template<class T>
arma::Mat<T> kernel_from_dots(const arma::Mat<T>& X1, const arma::Mat<T>& X2,
                             arma::Mat<T> dots, int kernel, T gamma,
                             int degree, T coef0) {
  if (X1.n_cols != X2.n_cols) {
    throw std::invalid_argument("X1 and X2 must have the same number of columns");
  }
  if (dots.n_rows != X1.n_rows || dots.n_cols != X2.n_rows) {
    throw std::invalid_argument("Kernel dot-product dimensions do not match inputs");
  }
  core::kernel_from_dots(
    core::make_const_view(X1.memptr(), X1.n_rows, X1.n_cols, X1.n_rows),
    core::make_const_view(X2.memptr(), X2.n_rows, X2.n_cols, X2.n_rows),
    core::make_view(dots.memptr(), dots.n_rows, dots.n_cols, dots.n_rows),
    static_cast<core::KernelType>(kernel), gamma, degree, coef0
  );
  return dots;
}

template<class T>
arma::Mat<T> kernel_matrix(const arma::Mat<T>& X1, const arma::Mat<T>& X2,
                          int kernel, T gamma, int degree, T coef0) {
  if (X1.n_cols != X2.n_cols) {
    throw std::invalid_argument("X1 and X2 must have the same number of columns");
  }
  arma::Mat<T> dots = X1 * X2.t();
  return kernel_from_dots(X1, X2, std::move(dots), kernel, gamma, degree, coef0);
}

template<class T>
struct CenteredKernel {
  arma::Mat<T> K;
  arma::Row<T> column_means;
  T grand_mean;
};

template<class T>
CenteredKernel<T> center_kernel_train(arma::Mat<T> K) {
  const auto centered = core::center_kernel_train(
    core::make_view(K.memptr(), K.n_rows, K.n_cols, K.n_rows)
  );
  arma::Row<T> means(centered.column_means);
  return {std::move(K), std::move(means), centered.grand_mean};
}

template<class T>
arma::Mat<T> center_kernel_test(arma::Mat<T> K, const arma::Row<T>& train_means,
                               T train_grand_mean) {
  if (K.n_cols != train_means.n_cols) {
    throw std::invalid_argument("Ktest columns must match the training kernel size");
  }
  core::center_kernel_test(
    core::make_view(K.memptr(), K.n_rows, K.n_cols, K.n_rows),
    train_means.memptr(), train_means.n_elem, train_grand_mean
  );
  return K;
}

} }
#endif
