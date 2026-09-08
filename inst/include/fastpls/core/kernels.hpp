// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Stefano Cacciatore
#ifndef FASTPLS_CORE_KERNELS_HPP
#define FASTPLS_CORE_KERNELS_HPP

#include <fastpls/core/linalg.hpp>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace fastpls {
namespace core {

enum class KernelType { linear = 1, radial_basis = 2, polynomial = 3 };

template<class T>
void kernel_from_dots(ConstMatrixView<T> left,
                      ConstMatrixView<T> right,
                      MatrixView<T> dots,
                      KernelType kernel,
                      T gamma,
                      int degree,
                      T offset) {
  if (left.columns() != right.columns() ||
      dots.rows() != left.rows() || dots.columns() != right.rows()) {
    throw std::invalid_argument("Kernel dimensions are inconsistent");
  }
  if (kernel == KernelType::linear) return;
  if (kernel == KernelType::polynomial) {
    for (std::size_t column = 0; column < dots.columns(); ++column) {
      for (std::size_t row = 0; row < dots.rows(); ++row) {
        dots(row, column) = std::pow(gamma * dots(row, column) + offset,
                                     degree);
      }
    }
    return;
  }
  if (kernel != KernelType::radial_basis) {
    throw std::invalid_argument("Unknown kernel type");
  }

  std::vector<T> left_norm(left.rows(), T(0));
  std::vector<T> right_norm(right.rows(), T(0));
  for (std::size_t feature = 0; feature < left.columns(); ++feature) {
    for (std::size_t row = 0; row < left.rows(); ++row) {
      left_norm[row] += left(row, feature) * left(row, feature);
    }
    for (std::size_t row = 0; row < right.rows(); ++row) {
      right_norm[row] += right(row, feature) * right(row, feature);
    }
  }
  const T tolerance = std::is_same<T, float>::value ? T(1e-5) : T(1e-10);
  for (std::size_t column = 0; column < dots.columns(); ++column) {
    for (std::size_t row = 0; row < dots.rows(); ++row) {
      T distance = left_norm[row] + right_norm[column] - T(2) * dots(row, column);
      if (distance < T(0) && distance > -tolerance) distance = T(0);
      dots(row, column) = std::exp(-gamma * distance);
    }
  }
}

template<class T>
Matrix<T> kernel_matrix_reference(ConstMatrixView<T> left,
                                  ConstMatrixView<T> right,
                                  KernelType kernel,
                                  T gamma,
                                  int degree,
                                  T offset) {
  Matrix<T> result(left.rows(), right.rows());
  reference_gemm(left, right, false, true, result.view());
  kernel_from_dots(left, right, result.view(), kernel, gamma, degree, offset);
  return result;
}

template<class T>
Matrix<T> kernel_matrix_reference(MatrixView<T> left,
                                  MatrixView<T> right,
                                  KernelType kernel,
                                  T gamma,
                                  int degree,
                                  T offset) {
  return kernel_matrix_reference(
    ConstMatrixView<T>(left), ConstMatrixView<T>(right), kernel, gamma,
    degree, offset
  );
}

template<class T>
struct KernelCentering {
  std::vector<T> column_means;
  T grand_mean = T(0);
};

template<class T>
KernelCentering<T> center_kernel_train(MatrixView<T> kernel) {
  if (kernel.empty()) {
    throw std::invalid_argument("Training kernel must be non-empty");
  }
  KernelCentering<T> result;
  result.column_means.assign(kernel.columns(), T(0));
  std::vector<T> row_means(kernel.rows(), T(0));
  for (std::size_t column = 0; column < kernel.columns(); ++column) {
    for (std::size_t row = 0; row < kernel.rows(); ++row) {
      const T value = kernel(row, column);
      result.column_means[column] += value;
      row_means[row] += value;
    }
    result.column_means[column] /= static_cast<T>(kernel.rows());
    result.grand_mean += result.column_means[column];
  }
  result.grand_mean /= static_cast<T>(kernel.columns());
  for (T& value : row_means) value /= static_cast<T>(kernel.columns());
  for (std::size_t column = 0; column < kernel.columns(); ++column) {
    for (std::size_t row = 0; row < kernel.rows(); ++row) {
      kernel(row, column) += result.grand_mean - row_means[row] -
        result.column_means[column];
    }
  }
  return result;
}

template<class T>
void center_kernel_test(MatrixView<T> kernel,
                        const T* training_column_means,
                        std::size_t mean_count,
                        T training_grand_mean) {
  if (kernel.empty() || training_column_means == nullptr ||
      mean_count != kernel.columns()) {
    throw std::invalid_argument(
      "Test-kernel columns must match the training kernel size"
    );
  }
  for (std::size_t row = 0; row < kernel.rows(); ++row) {
    T row_mean = T(0);
    for (std::size_t column = 0; column < kernel.columns(); ++column) {
      row_mean += kernel(row, column);
    }
    row_mean /= static_cast<T>(kernel.columns());
    for (std::size_t column = 0; column < kernel.columns(); ++column) {
      kernel(row, column) += training_grand_mean - row_mean -
        training_column_means[column];
    }
  }
}

}  // namespace core
}  // namespace fastpls

#endif
