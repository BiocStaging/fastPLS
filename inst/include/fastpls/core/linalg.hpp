// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Stefano Cacciatore
#ifndef FASTPLS_CORE_LINALG_HPP
#define FASTPLS_CORE_LINALG_HPP

#include <fastpls/core/matrix.hpp>

#include <cstddef>
#include <stdexcept>

namespace fastpls {
namespace core {

template<class T>
void reference_gemm(ConstMatrixView<T> left,
                    ConstMatrixView<T> right,
                    bool transpose_left,
                    bool transpose_right,
                    MatrixView<T> output) {
  const std::size_t rows = transpose_left ? left.columns() : left.rows();
  const std::size_t inner_left = transpose_left ? left.rows() : left.columns();
  const std::size_t inner_right = transpose_right ? right.columns() : right.rows();
  const std::size_t columns = transpose_right ? right.rows() : right.columns();
  if (inner_left != inner_right || output.rows() != rows ||
      output.columns() != columns) {
    throw std::invalid_argument("fastPLS matrix-product dimensions are inconsistent");
  }

  for (std::size_t column = 0; column < columns; ++column) {
    for (std::size_t row = 0; row < rows; ++row) {
      T value = T(0);
      for (std::size_t inner = 0; inner < inner_left; ++inner) {
        const T lhs = transpose_left ? left(inner, row) : left(row, inner);
        const T rhs = transpose_right ? right(column, inner) : right(inner, column);
        value += lhs * rhs;
      }
      output(row, column) = value;
    }
  }
}

template<class T>
void reference_gemm(MatrixView<T> left,
                    MatrixView<T> right,
                    bool transpose_left,
                    bool transpose_right,
                    MatrixView<T> output) {
  reference_gemm(
    ConstMatrixView<T>(left), ConstMatrixView<T>(right), transpose_left,
    transpose_right, output
  );
}

}  // namespace core
}  // namespace fastpls

#endif
