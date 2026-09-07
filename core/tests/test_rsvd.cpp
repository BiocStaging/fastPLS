// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Stefano Cacciatore
#include <fastpls/native/rsvd.hpp>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <type_traits>

void require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}

template<typename T>
void check_precision() {
  static_assert(std::is_same<typename arma::Mat<T>::elem_type, T>::value, "precision");
  const T tolerance = std::is_same<T, float>::value ? T(2e-3) : T(1e-8);
  arma::Mat<T> left(120, 5), right(85, 5);
  for (arma::uword i = 0; i < right.n_elem; ++i) right[i] = std::cos(T(i * i + 1));
  // Distinct frequencies give a known low-rank matrix without an external RNG.
  for (arma::uword j = 0; j < 5; ++j) {
    for (arma::uword i = 0; i < left.n_rows; ++i) {
      left(i, j) = std::sin(T(i + 1) * T(j + 1) / T(13));
    }
  }
  arma::Mat<T> A = left * right.t();
  fastpls::native::RsvdControls options;
  options.oversample = 8;
  options.power = 2;
  options.seed = 19;
  auto fit = fastpls::native::rsvd(A, 5, options);
  auto repeat = fastpls::native::rsvd(A, 5, options);
  require(fit.s.n_elem == 5, "requested rank was not retained");
  require(arma::approx_equal(fit.U, repeat.U, "absdiff", T(0)), "fixed seed differs");
  arma::Mat<T> reconstructed = fit.U * arma::diagmat(fit.s) * fit.Vt;
  require(arma::norm(A - reconstructed, "fro") / arma::norm(A, "fro") < tolerance,
          "low-rank reconstruction failed");
  arma::Mat<T> identity = arma::eye<arma::Mat<T>>(5, 5);
  require(arma::norm(fit.U.t() * fit.U - identity, "fro") < tolerance,
          "left orthogonality failed");
  options.left_only = true;
  auto compact = fastpls::native::rsvd(A, 5, options);
  require(compact.Vt.n_elem == 0, "unrequested right factors were retained");
  require(arma::approx_equal(fit.U, compact.U, "absdiff", T(0)), "left-only changed U");
  arma::Mat<T> small = A.submat(0, 0, 8, 5);
  auto dense = fastpls::native::rsvd(small, 6, options);
  require(dense.s.n_elem == 6, "full-width recovery lost components");
  arma::Mat<T> empty(0, 4);
  require(fastpls::native::rsvd(empty, 2, options).s.n_elem == 0,
          "empty matrix was not handled");
  std::cout << (std::is_same<T, float>::value ? "float32" : "float64") << " OK\n";
}

int main() {
  check_precision<float>();
  check_precision<double>();
}
