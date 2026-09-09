// SPDX-License-Identifier: MIT

#include <fastpls/core/kernels.hpp>

#include <cassert>
#include <cmath>

namespace {

template<class T>
bool close(T left, T right, T tolerance = T(1e-5)) {
  return std::abs(left - right) <= tolerance;
}

}  // namespace

int main() {
  fastpls::core::Matrix<double> left(2, 2);
  left(0, 0) = 1.0;
  left(1, 0) = 0.0;
  left(0, 1) = 0.0;
  left(1, 1) = 2.0;
  const auto polynomial = fastpls::core::kernel_matrix_reference(
    left.view(), left.view(), fastpls::core::KernelType::polynomial,
    0.5, 2, 1.0
  );
  assert(close(polynomial(0, 0), 2.25));
  assert(close(polynomial(0, 1), 1.0));
  assert(close(polynomial(1, 1), 9.0));

  const auto radial = fastpls::core::kernel_matrix_reference(
    left.view(), left.view(), fastpls::core::KernelType::radial_basis,
    0.25, 1, 0.0
  );
  assert(close(radial(0, 0), 1.0));
  assert(close(radial(1, 1), 1.0));
  assert(close(radial(0, 1), std::exp(-1.25)));

  fastpls::core::Matrix<double> training(3, 3);
  for (std::size_t column = 0; column < 3; ++column) {
    for (std::size_t row = 0; row < 3; ++row) {
      training(row, column) = static_cast<double>(1 + row + 2 * column);
    }
  }
  const auto centering = fastpls::core::center_kernel_train(training.view());
  for (std::size_t row = 0; row < 3; ++row) {
    double sum = 0.0;
    for (std::size_t column = 0; column < 3; ++column) {
      sum += training(row, column);
    }
    assert(close(sum, 0.0));
  }
  for (std::size_t column = 0; column < 3; ++column) {
    double sum = 0.0;
    for (std::size_t row = 0; row < 3; ++row) sum += training(row, column);
    assert(close(sum, 0.0));
  }

  fastpls::core::Matrix<double> test(1, 3);
  test(0, 0) = 2.0;
  test(0, 1) = 4.0;
  test(0, 2) = 7.0;
  fastpls::core::center_kernel_test(
    test.view(), centering.column_means.data(),
    centering.column_means.size(), centering.grand_mean
  );
  assert(close(test(0, 0) + test(0, 1) + test(0, 2), 0.0));
  return 0;
}
