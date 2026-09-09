// SPDX-License-Identifier: MIT
#include <fastpls/core.hpp>

#include <cassert>
#include <cmath>

int main() {
  fastpls::core::Matrix<double> left(3, 2);
  fastpls::core::Matrix<double> right(2, 4);
  for (std::size_t column = 0; column < left.columns(); ++column) {
    for (std::size_t row = 0; row < left.rows(); ++row) {
      left(row, column) = static_cast<double>(1 + row + 3 * column);
    }
  }
  for (std::size_t column = 0; column < right.columns(); ++column) {
    for (std::size_t row = 0; row < right.rows(); ++row) {
      right(row, column) = static_cast<double>(1 + row + 2 * column);
    }
  }
  fastpls::core::Matrix<double> output(3, 4);
  fastpls::core::reference_gemm(
    left.view(), right.view(), false, false, output.view()
  );
  for (std::size_t column = 0; column < output.columns(); ++column) {
    for (std::size_t row = 0; row < output.rows(); ++row) {
      const double expected =
        left(row, 0) * right(0, column) +
        left(row, 1) * right(1, column);
      assert(std::abs(output(row, column) - expected) < 1e-12);
    }
  }

  fastpls::core::Matrix<double> gram(2, 2);
  fastpls::core::reference_gemm(
    left.view(), left.view(), true, false, gram.view()
  );
  assert(std::abs(gram(0, 1) - gram(1, 0)) < 1e-12);

  fastpls::core::Matrix<double> system(2, 2);
  system(0, 0) = 4.0;
  system(1, 0) = 1.0;
  system(0, 1) = 1.0;
  system(1, 1) = 3.0;
  fastpls::core::Matrix<double> rhs(2, 1);
  rhs(0, 0) = 1.0;
  rhs(1, 0) = 2.0;
  fastpls::core::Matrix<double> solution;
  assert(fastpls::core::solve_symmetric_system(
    system.view(), rhs.view(), solution
  ));
  assert(std::abs(solution(0, 0) - 1.0 / 11.0) < 1e-12);
  assert(std::abs(solution(1, 0) - 7.0 / 11.0) < 1e-12);

  system(0, 0) = 0.0;
  system(1, 0) = 1.0;
  system(0, 1) = 1.0;
  system(1, 1) = 1.0;
  rhs(0, 0) = 1.0;
  rhs(1, 0) = 2.0;
  assert(fastpls::core::solve_symmetric_system(
    system.view(), rhs.view(), solution
  ));
  assert(std::abs(solution(0, 0) - 1.0) < 1e-12);
  assert(std::abs(solution(1, 0) - 1.0) < 1e-12);
  return 0;
}
