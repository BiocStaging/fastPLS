// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Stefano Cacciatore
#ifndef FASTPLS_CORE_OPERATORS_HPP
#define FASTPLS_CORE_OPERATORS_HPP

#include <fastpls/core/matrix.hpp>

#include <algorithm>
#include <cstddef>
#include <stdexcept>

namespace fastpls {
namespace core {

template<class T, class Backend>
class ExplicitOperator {
 public:
  ExplicitOperator(ConstMatrixView<T> matrix, Backend& backend)
      : matrix_(matrix), backend_(backend) {
    if (matrix_.data() == nullptr || matrix_.empty()) {
      throw std::invalid_argument(
        "fastPLS explicit operator requires a nonempty matrix"
      );
    }
  }

  std::size_t rows() const noexcept { return matrix_.rows(); }
  std::size_t columns() const noexcept { return matrix_.columns(); }
  std::size_t workspace_rows() const noexcept { return 0; }

  void multiply(ConstMatrixView<T> right, bool transpose,
                Matrix<T>& output) {
    const std::size_t output_rows = transpose ? columns() : rows();
    output.resize(output_rows, right.columns());
    multiply(right, transpose, output.view());
  }

  void multiply(ConstMatrixView<T> right, bool transpose,
                MatrixView<T> output) {
    backend_.gemm(matrix_, right, transpose, false, output);
  }

  void materialize(Matrix<T>& output) const {
    output.resize(rows(), columns());
    for (std::size_t column = 0; column < columns(); ++column) {
      std::copy_n(
        matrix_.data() + column * matrix_.leading_dimension(), rows(),
        output.data() + column * rows()
      );
    }
  }

 private:
  ConstMatrixView<T> matrix_;
  Backend& backend_;
};

// Represents S_a = F_a'Y with F_a = F_{a-1}(I-v_a v_a'). Deflation
// updates the smaller predictor-side factor instead of materializing S_a.
template<class T, class Backend>
class CrosscovOperator {
 public:
  CrosscovOperator(ConstMatrixView<T> predictors,
                   ConstMatrixView<T> responses,
                   std::size_t component_capacity,
                   Backend& backend)
      : responses_(responses), capacity_(component_capacity),
        backend_(backend) {
    if (predictors.data() == nullptr || responses_.data() == nullptr ||
        predictors.empty() || responses_.empty() ||
        predictors.rows() != responses_.rows()) {
      throw std::invalid_argument(
        "fastPLS cross-covariance operator requires matching nonempty matrices"
      );
    }
    left_.resize(predictors.rows(), predictors.columns());
    for (std::size_t column = 0; column < predictors.columns(); ++column) {
      std::copy_n(
        predictors.data() + column * predictors.leading_dimension(),
        predictors.rows(), left_.data() + column * left_.rows()
      );
    }
  }

  std::size_t rows() const noexcept { return left_.columns(); }
  std::size_t columns() const noexcept { return responses_.columns(); }
  std::size_t workspace_rows() const noexcept { return responses_.rows(); }
  std::size_t deflations() const noexcept { return active_; }

  void multiply(ConstMatrixView<T> right, bool transpose,
                Matrix<T>& output) {
    const std::size_t output_rows = transpose ? columns() : rows();
    output.resize(output_rows, right.columns());
    multiply(right, transpose, output.view());
  }

  void multiply(ConstMatrixView<T> right, bool transpose,
                MatrixView<T> output) {
    const std::size_t expected_rows = transpose ? rows() : columns();
    const std::size_t expected_output_rows = transpose ? columns() : rows();
    if (right.rows() != expected_rows ||
        output.rows() != expected_output_rows ||
        output.columns() != right.columns()) {
      throw std::invalid_argument(
        "fastPLS cross-covariance product dimensions are inconsistent"
      );
    }
    intermediate_.resize(left_.rows(), right.columns());
    if (transpose) {
      backend_.gemm(
        left_.view(), right, false, false, intermediate_.view()
      );
      backend_.gemm(
        responses_, intermediate_.view(), true, false, output
      );
    } else {
      backend_.gemm(
        responses_, right, false, false, intermediate_.view()
      );
      backend_.gemm(
        left_.view(), intermediate_.view(), true, false, output
      );
    }
  }

  void deflate(ConstMatrixView<T> direction) {
    if (direction.rows() != rows() || direction.columns() != 1 ||
        active_ >= capacity_) {
      throw std::invalid_argument(
        "fastPLS cross-covariance deflation dimensions are inconsistent"
      );
    }
    score_.resize(left_.rows(), 1);
    backend_.gemm(
      left_.view(), direction, false, false, score_.view()
    );
    for (std::size_t column = 0; column < left_.columns(); ++column) {
      const T weight = direction(column, 0);
      T* values = left_.data() + column * left_.rows();
      for (std::size_t row = 0; row < left_.rows(); ++row) {
        values[row] -= score_(row, 0) * weight;
      }
    }
    ++active_;
  }

  void materialize(Matrix<T>& output) {
    output.resize(rows(), columns());
    materialize(output.view());
  }

  void materialize(MatrixView<T> output) {
    if (output.rows() != rows() || output.columns() != columns()) {
      throw std::invalid_argument(
        "fastPLS cross-covariance materialization dimensions are inconsistent"
      );
    }
    backend_.gemm(
      left_.view(), responses_, true, false, output
    );
  }

  ConstMatrixView<T> left_factor() const noexcept { return left_.view(); }
  ConstMatrixView<T> right_factor() const noexcept { return responses_; }

 private:
  Matrix<T> left_;
  ConstMatrixView<T> responses_;
  const std::size_t capacity_;
  Backend& backend_;
  Matrix<T> intermediate_;
  Matrix<T> score_;
  std::size_t active_ = 0;
};

}  // namespace core
}  // namespace fastpls

#endif
