// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Stefano Cacciatore
#include <fastpls/native/kernels.hpp>
#include <random>

template<class T>
void check() {
  arma::Mat<T> x(30, 7), heldout(9, 7);
  std::mt19937 rng(294);
  std::normal_distribution<T> normal(T(0), T(1));
  for (auto& value : x) value = normal(rng);
  for (auto& value : heldout) value = normal(rng);
  const T tolerance = std::is_same<T, float>::value ? T(2e-5) : T(1e-12);
  for (int kind : {1, 2, 3}) {
    const T gamma = T(0.2), offset = T(0.5);
    auto kernel = fastpls::native::kernel_matrix(x, x, kind, gamma, 3, offset);
    auto centered = fastpls::native::center_kernel_train(kernel);
    auto new_kernel = fastpls::native::kernel_matrix(heldout, x, kind, gamma, 3, offset);
    arma::Mat<T> direct(new_kernel.n_rows, new_kernel.n_cols);
    for (arma::uword row = 0; row < heldout.n_rows; ++row) {
      for (arma::uword col = 0; col < x.n_rows; ++col) {
        T dot = arma::dot(heldout.row(row), x.row(col));
        direct(row, col) = kind == 1 ? dot : kind == 3 ?
          static_cast<T>(std::pow(gamma * dot + offset, 3)) :
          std::exp(-gamma * arma::accu(arma::square(heldout.row(row) - x.row(col))));
      }
    }
    auto prediction = fastpls::native::center_kernel_test(
      new_kernel, centered.column_means, centered.grand_mean);
    arma::Mat<T> explicit_center = direct;
    explicit_center.each_row() -= centered.column_means;
    explicit_center.each_col() -= arma::mean(direct, 1);
    explicit_center += centered.grand_mean;
    if (arma::norm(prediction - explicit_center, "fro") >
          tolerance * std::max(T(1), arma::norm(explicit_center, "fro")) ||
        arma::norm(arma::mean(centered.K, 0), 2) >
          tolerance * std::max(T(1), arma::norm(centered.K, "fro"))) {
      throw std::runtime_error("Kernel formula or training-only centering mismatch");
    }
  }
}

int main() {
  check<float>();
  check<double>();
}
