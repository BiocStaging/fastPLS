// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Stefano Cacciatore
#ifndef FASTPLS_NATIVE_KERNELS_HPP
#define FASTPLS_NATIVE_KERNELS_HPP
#include <armadillo>
#include <cmath>
#include <stdexcept>
#include <type_traits>

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
  if (kernel == 1) return dots;
  if (kernel == 3) {
    if constexpr (std::is_same<T, float>::value) {
      dots.transform([gamma, coef0, degree](T value) {
        return static_cast<T>(std::pow(gamma * value + coef0, degree));
      });
      return dots;
    } else {
      return arma::pow(gamma * dots + coef0, degree);
    }
  }
  if (kernel != 2) throw std::invalid_argument("Unknown kernel id");
  arma::Col<T> n1 = arma::sum(arma::square(X1), 1);
  arma::Row<T> n2 = arma::sum(arma::square(X2), 1).t();
  dots *= T(-2);
  dots.each_col() += n1;
  dots.each_row() += n2;
  arma::Mat<T>& dist2 = dots;
  if constexpr (std::is_same<T, float>::value) {
    dist2.transform([gamma](T value) {
      if (value < T(0) && value > T(-1e-5)) value = T(0);
      return std::exp(-gamma * value);
    });
    return dist2;
  } else {
    dist2.transform([](T value) {
      return value < T(0) && value > T(-1e-10) ? T(0) : value;
    });
    return arma::exp(-gamma * dist2);
  }
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
  arma::Row<T> means = arma::mean(K, 0);
  arma::Col<T> rows = arma::mean(K, 1);
  const T grand = arma::mean(means);
  K.each_row() -= means;
  K.each_col() -= rows;
  K += grand;
  return {std::move(K), std::move(means), grand};
}

template<class T>
arma::Mat<T> center_kernel_test(arma::Mat<T> K, const arma::Row<T>& train_means,
                               T train_grand_mean) {
  if (K.n_cols != train_means.n_cols) {
    throw std::invalid_argument("Ktest columns must match the training kernel size");
  }
  arma::Col<T> rows = arma::mean(K, 1);
  K.each_row() -= train_means;
  K.each_col() -= rows;
  K += train_grand_mean;
  return K;
}

} }
#endif
