// SPDX-License-Identifier: MIT
#ifndef FASTPLS_TEST_REFERENCE_BACKEND_HPP
#define FASTPLS_TEST_REFERENCE_BACKEND_HPP

#include <fastpls/core.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <vector>

template<class T>
class ReferenceBackend {
 public:
  void gemm(fastpls::core::ConstMatrixView<T> left,
            fastpls::core::ConstMatrixView<T> right,
            bool transpose_left,
            bool transpose_right,
            fastpls::core::MatrixView<T> output) {
    fastpls::core::reference_gemm(
      left, right, transpose_left, transpose_right, output
    );
  }

  bool qr_economy(fastpls::core::ConstMatrixView<T> input,
                  fastpls::core::Matrix<T>& q) {
    const std::size_t rank = std::min(input.rows(), input.columns());
    q.resize(input.rows(), rank);
    for (std::size_t column = 0; column < rank; ++column) {
      for (std::size_t row = 0; row < input.rows(); ++row) {
        q(row, column) = input(row, column);
      }
      for (int pass = 0; pass < 2; ++pass) {
        for (std::size_t previous = 0; previous < column; ++previous) {
          T projection = T(0);
          for (std::size_t row = 0; row < q.rows(); ++row) {
            projection += q(row, previous) * q(row, column);
          }
          for (std::size_t row = 0; row < q.rows(); ++row) {
            q(row, column) -= projection * q(row, previous);
          }
        }
      }
      T norm = T(0);
      for (std::size_t row = 0; row < q.rows(); ++row) {
        norm += q(row, column) * q(row, column);
      }
      norm = std::sqrt(norm);
      if (!std::isfinite(norm) ||
          norm <= std::numeric_limits<T>::epsilon()) return false;
      for (std::size_t row = 0; row < q.rows(); ++row) {
        q(row, column) /= norm;
      }
    }
    return true;
  }

  bool symmetric_eigen(fastpls::core::Matrix<T>& matrix,
                       std::vector<T>& eigenvalues) {
    const std::size_t n = matrix.rows();
    if (n != matrix.columns()) return false;
    fastpls::core::Matrix<T> vectors(n, n);
    for (std::size_t column = 0; column < n; ++column) {
      vectors(column, column) = T(1);
    }
    for (std::size_t iteration = 0; iteration < 128 * n * n; ++iteration) {
      std::size_t p = 0;
      std::size_t q = 0;
      T maximum = T(0);
      T scale = T(1);
      for (std::size_t column = 0; column < n; ++column) {
        scale = std::max(scale, std::abs(matrix(column, column)));
        for (std::size_t row = 0; row < column; ++row) {
          if (std::abs(matrix(row, column)) > maximum) {
            maximum = std::abs(matrix(row, column));
            p = row;
            q = column;
          }
        }
      }
      if (maximum <= T(8) * std::numeric_limits<T>::epsilon() * scale) {
        eigenvalues.resize(n);
        for (std::size_t index = 0; index < n; ++index) {
          eigenvalues[index] = matrix(index, index);
        }
        std::vector<std::size_t> order(n);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](std::size_t left,
                                                  std::size_t right) {
          return eigenvalues[left] < eigenvalues[right];
        });
        fastpls::core::Matrix<T> sorted(n, n);
        std::vector<T> values(n);
        for (std::size_t column = 0; column < n; ++column) {
          values[column] = eigenvalues[order[column]];
          for (std::size_t row = 0; row < n; ++row) {
            sorted(row, column) = vectors(row, order[column]);
          }
        }
        matrix = std::move(sorted);
        eigenvalues = std::move(values);
        return true;
      }
      const T app = matrix(p, p);
      const T aqq = matrix(q, q);
      const T apq = matrix(p, q);
      const T tau = (aqq - app) / (T(2) * apq);
      const T tangent = (tau >= T(0) ? T(1) : T(-1)) /
        (std::abs(tau) + std::sqrt(T(1) + tau * tau));
      const T cosine = T(1) / std::sqrt(T(1) + tangent * tangent);
      const T sine = tangent * cosine;
      for (std::size_t index = 0; index < n; ++index) {
        if (index == p || index == q) continue;
        const T aip = matrix(index, p);
        const T aiq = matrix(index, q);
        matrix(index, p) = matrix(p, index) = cosine * aip - sine * aiq;
        matrix(index, q) = matrix(q, index) = sine * aip + cosine * aiq;
      }
      matrix(p, p) = app - tangent * apq;
      matrix(q, q) = aqq + tangent * apq;
      matrix(p, q) = matrix(q, p) = T(0);
      for (std::size_t row = 0; row < n; ++row) {
        const T vip = vectors(row, p);
        const T viq = vectors(row, q);
        vectors(row, p) = cosine * vip - sine * viq;
        vectors(row, q) = sine * vip + cosine * viq;
      }
    }
    return false;
  }

  bool svd_economy(fastpls::core::ConstMatrixView<T>, bool,
                   fastpls::core::Matrix<T>&, std::vector<T>&,
                   fastpls::core::Matrix<T>&) {
    return false;
  }
};

#endif
