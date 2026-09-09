// SPDX-License-Identifier: MIT
#include <fastpls/core.hpp>

#include "reference_backend.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>

template<class T>
void check_dense_preprocessing() {
  fastpls::core::Matrix<T> x(5, 3);
  fastpls::core::Matrix<T> y(5, 2);
  for (std::size_t row = 0; row < x.rows(); ++row) {
    x(row, 0) = static_cast<T>(row + 1);
    x(row, 1) = static_cast<T>(2 * static_cast<int>(row) - 3);
    x(row, 2) = row % 2 == 0 ? T(-1) : T(1);
    y(row, 0) = T(1.5) * x(row, 0) - T(0.25) * x(row, 1) + T(4);
    y(row, 1) = T(-0.5) * x(row, 0) + T(0.75) * x(row, 2) - T(2);
  }
  auto scaled_x = x;
  ReferenceBackend<T> backend;
  const auto prepared = fastpls::core::prepare_scaled_dense_crossprod(
    scaled_x.view(), y.view(),
    fastpls::core::PredictorScaling::centering, backend
  );
  assert(prepared.crossprod.rows() == x.columns());
  assert(prepared.crossprod.columns() == y.columns());
  for (std::size_t column = 0; column < scaled_x.columns(); ++column) {
    T sum = T(0);
    for (std::size_t row = 0; row < scaled_x.rows(); ++row) {
      sum += scaled_x(row, column);
    }
    assert(std::abs(sum) < T(1e-5));
  }

  fastpls::core::Matrix<T> centered_y(y.rows(), y.columns());
  for (std::size_t column = 0; column < y.columns(); ++column) {
    for (std::size_t row = 0; row < y.rows(); ++row) {
      centered_y(row, column) = y(row, column) -
        prepared.response_mean[column];
    }
  }
  fastpls::core::Matrix<T> expected(x.columns(), y.columns());
  backend.gemm(
    scaled_x.view(), centered_y.view(), true, false, expected.view()
  );
  for (std::size_t index = 0; index < expected.size(); ++index) {
    assert(std::abs(expected.data()[index] - prepared.crossprod.data()[index]) <
      T(2e-5));
  }
  const double perfect = fastpls::core::dense_response_r2(
    y.view(), prepared.response_mean.data(), y.columns(), centered_y.view()
  );
  assert(std::abs(perfect - 1.0) < 1e-10);

  const auto unscaled = fastpls::core::unscaled_dense_crossprod(
    x.view(), y.view(), backend
  );
  fastpls::core::Matrix<T> expected_unscaled(x.columns(), y.columns());
  backend.gemm(x.view(), centered_y.view(), true, false,
               expected_unscaled.view());
  for (std::size_t index = 0; index < expected_unscaled.size(); ++index) {
    assert(std::abs(expected_unscaled.data()[index] -
      unscaled.crossprod.data()[index]) < T(2e-5));
  }
}

int main() {
  check_dense_preprocessing<float>();
  check_dense_preprocessing<double>();
  return 0;
}
