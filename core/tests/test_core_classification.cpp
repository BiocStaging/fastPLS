// SPDX-License-Identifier: MIT
#include <fastpls/core.hpp>

#include <cassert>
#include <cmath>
#include <cstddef>
#include <stdexcept>

int main() {
  fastpls::core::Matrix<float> x(4, 3);
  for (std::size_t column = 0; column < x.columns(); ++column) {
    for (std::size_t row = 0; row < x.rows(); ++row) {
      x(row, column) = static_cast<float>(1 + row + 10 * column);
    }
  }
  const std::size_t labels[] = {0, 1, 0, 1};
  const float offset[] = {-0.5f, -0.5f};
  fastpls::core::Matrix<float> crossprod(3, 2);
  fastpls::core::centered_label_crossprod(
    x.view(), labels, 4, offset, 2, crossprod.view()
  );

  for (std::size_t column = 0; column < x.columns(); ++column) {
    const float class_zero = x(0, column) + x(2, column);
    const float class_one = x(1, column) + x(3, column);
    const float total = class_zero + class_one;
    assert(std::abs(crossprod(column, 0) - (class_zero - 0.5f * total)) < 1e-6f);
    assert(std::abs(crossprod(column, 1) - (class_one - 0.5f * total)) < 1e-6f);
  }

  fastpls::core::Matrix<double> scores(2, 3);
  scores(0, 0) = 0.2;
  scores(0, 1) = 0.8;
  scores(0, 2) = 0.1;
  scores(1, 0) = 2.0;
  scores(1, 1) = 1.0;
  scores(1, 2) = 0.0;
  assert(fastpls::core::row_argmax(scores.view(), 0) == 1);
  assert(fastpls::core::row_argmax(scores.view(), 1) == 0);

  bool rejected = false;
  const std::size_t invalid_labels[] = {0, 1, 2, 1};
  try {
    fastpls::core::centered_label_crossprod(
      x.view(), invalid_labels, 4, offset, 2, crossprod.view()
    );
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  assert(rejected);
  return 0;
}
