// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Stefano Cacciatore
#include <fastpls/native/operator_rsvd.hpp>
#include <iostream>
#include <type_traits>

void require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}

template<typename T>
struct CountingOperator : fastpls::native::MatrixOperator<T> {
  using Base = fastpls::native::MatrixOperator<T>;
  using Base::Base;
  int products = 0, full_calls = 0;
  arma::Mat<T> multiply(const arma::Mat<T>& X, bool transpose = false) {
    ++products;
    return Base::multiply(X, transpose);
  }
  void full_svd(arma::Mat<T>& U, arma::Col<T>& d, arma::Mat<T>& V, bool left_only) {
    ++full_calls;
    Base::full_svd(U, d, V, left_only);
  }
};

template<typename T>
T residual(const arma::Mat<T>& A, const fastpls::native::SingularTriplets<T>& sv) {
  arma::Mat<T> left = A * sv.Vt.t() - sv.U * arma::diagmat(sv.s);
  arma::Mat<T> right = A.t() * sv.U - sv.Vt.t() * arma::diagmat(sv.s);
  return std::max(arma::norm(left, "fro"), arma::norm(right, "fro")) /
    std::max(T(1), arma::norm(A, "fro"));
}

template<typename T>
void check() {
  const T tolerance = std::is_same<T, float>::value ? T(2e-5) : T(1e-10);
  std::mt19937 generator(614);
  std::normal_distribution<T> normal(T(0), T(1));
  arma::Mat<T> left(160, 100), right(125, 100), U, V, R;
  for (auto& v : left) v = normal(generator);
  for (auto& v : right) v = normal(generator);
  arma::qr_econ(U, R, left);
  arma::qr_econ(V, R, right);
  for (int shape : {0, 1, 2}) {
    arma::Col<T> singular(100);
    for (arma::uword i = 0; i < singular.n_elem; ++i) {
      singular(i) = shape == 0 ? T(1) / T(1 + i) :
        shape == 1 ? T(1) - T(0.005) * T(i) : (i < 12 ? T(1) / T(i + 1) : T(0));
    }
    arma::Mat<T> A = U * arma::diagmat(singular) * V.t();
    CountingOperator<T> op(A);
    fastpls::native::RsvdControls controls;
    controls.oversample = 1;
    controls.power = 1;
    fastpls::native::OperatorRsvdWorkspace<T> work;
    auto acceptable = [&](const fastpls::native::SingularTriplets<T>& candidate) {
      return candidate.U.is_finite() && candidate.Vt.is_finite() &&
        residual(A, candidate) < tolerance;
    };
    auto recovered = fastpls::native::recover_operator_rsvd<T>(op, 6, controls, work, acceptable);
    require(acceptable(recovered.result), "unqualified recovery returned");
    require(op.full_calls == 0, "recovery materialized a full operator decomposition");
    require(arma::max(arma::abs(recovered.result.s - singular.head(6))) < 10 * tolerance,
            "recovery missed leading singular values");
    require(recovered.estimated_workspace_bytes <= 256.L * 1024 * 1024,
            "recovery exceeded workspace estimate");
    auto repeated = fastpls::native::recover_operator_rsvd<T>(op, 6, controls, work, acceptable);
    require(arma::approx_equal(recovered.result.U, repeated.result.U, "absdiff", T(0)),
            "workspace reuse changed the fresh-seed result");
    bool rejected = false;
    const int before = op.products;
    try { fastpls::native::recover_operator_rsvd<T>(op, 6, controls, work, acceptable, 1); }
    catch (const std::runtime_error&) { rejected = true; }
    require(rejected && op.products == before, "budget error allocated numerical work");
    int checked = 0;
    rejected = false;
    try {
      fastpls::native::recover_operator_rsvd<T>(op, 6, controls, work,
        [&](const auto&) { ++checked; return false; });
    } catch (const std::runtime_error&) { rejected = true; }
    require(rejected && checked == 4, "failed recovery was silently accepted");
  }
  arma::Mat<T> X(30, 140), Y(30, 130);
  for (auto& v : X) v = normal(generator);
  for (auto& v : Y) v = normal(generator);
  arma::Mat<T> S = X.t() * Y;
  fastpls::native::CrosscovOperator<T> op(X, Y, 1);
  fastpls::native::OperatorRsvdWorkspace<T> work;
  fastpls::native::RsvdControls controls;
  controls.oversample = 0;
  auto recovered = fastpls::native::recover_operator_rsvd<T>(op, 5, controls, work,
    [&](const auto& candidate) { return residual(S, candidate) < tolerance; });
  require(residual(S, recovered.result) < tolerance, "implicit recovery failed");
  std::cout << (std::is_same<T, float>::value ? "float32" : "float64")
            << " native rSVD recovery OK\n";
}

int main() { check<float>(); check<double>(); }
