// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Stefano Cacciatore
#include <fastpls/native/operators.hpp>
#include <iostream>
#include <type_traits>

void require(bool value, const char* text) {
  if (!value) throw std::runtime_error(text);
}

template<typename T>
void check() {
  const T tolerance = std::is_same<T, float>::value ? T(2e-5) : T(1e-12);
  std::mt19937 random(531);
  std::normal_distribution<T> normal(T(0), T(1));
  for (arma::uword rows : {13u, 81u}) for (arma::uword width : {1u, 3u, 11u}) {
    arma::Mat<T> X(rows, 43), Y(rows, 37), B(37, width), C(43, width);
    for (auto& x : X) x = normal(random);
    for (auto& x : Y) x = normal(random);
    for (auto& x : B) x = normal(random);
    for (auto& x : C) x = normal(random);
    fastpls::native::CrosscovOperator<T> op(X, Y, 5);
    arma::Mat<T> left = X;
    for (int step = 0; step <= 5; ++step) {
      arma::Mat<T> S = left.t() * Y;
      for (bool transpose : {false, true}) {
        arma::Mat<T> expected = transpose ? arma::Mat<T>(S.t() * C) : arma::Mat<T>(S * B);
        arma::Mat<T> actual = op.multiply(transpose ? C : B, transpose);
        require(arma::norm(expected - actual, "fro") /
                  std::max(T(1), arma::norm(expected, "fro")) < tolerance,
                "implicit product changed the operator");
      }
      if (step < 5) {
        arma::Col<T> v(43);
        for (auto& x : v) x = normal(random);
        v /= arma::norm(v, 2);
        op.deflate(v);
        arma::Col<T> score = left * v;
        for (arma::uword column = 0; column < left.n_cols; ++column) {
          left.col(column) -= score * v(column);
        }
      }
    }
    arma::Mat<T> U, V;
    arma::Col<T> singular;
    op.full_svd(U, singular, V, false);
    arma::Mat<T> reconstructed = U * arma::diagmat(singular) * V.t();
    arma::Mat<T> expected = left.t() * Y;
    require(arma::norm(reconstructed - expected, "fro") /
              std::max(T(1), arma::norm(expected, "fro")) < tolerance,
            "factor-product reconstruction failed");
    bool rejected = false;
    try { op.deflate(arma::Col<T>(43, arma::fill::ones)); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "capacity guard failed");
    arma::Mat<T> empty(37, 0);
    require(op.multiply(empty).n_cols == 0, "empty product has incorrect dimensions");
  }
  std::cout << (std::is_same<T, float>::value ? "float32" : "float64")
            << " matrix-free operators OK\n";
}

int main() { check<float>(); check<double>(); }
