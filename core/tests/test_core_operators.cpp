// SPDX-License-Identifier: MIT
#include <fastpls/core.hpp>

#include <cassert>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace {

template<class T>
struct ReferenceBackend {
  void gemm(fastpls::core::ConstMatrixView<T> left,
            fastpls::core::ConstMatrixView<T> right,
            bool transpose_left,
            bool transpose_right,
            fastpls::core::MatrixView<T> output) {
    fastpls::core::reference_gemm(
      left, right, transpose_left, transpose_right, output
    );
  }
};

template<class T>
T relative_error(fastpls::core::ConstMatrixView<T> observed,
                 fastpls::core::ConstMatrixView<T> expected) {
  T difference = T(0);
  T scale = T(0);
  for (std::size_t column = 0; column < observed.columns(); ++column) {
    for (std::size_t row = 0; row < observed.rows(); ++row) {
      const T residual = observed(row, column) - expected(row, column);
      difference += residual * residual;
      scale += expected(row, column) * expected(row, column);
    }
  }
  return std::sqrt(difference / std::max(scale, T(1)));
}

template<class T>
T relative_error(fastpls::core::MatrixView<T> observed,
                 fastpls::core::MatrixView<T> expected) {
  return relative_error(
    fastpls::core::ConstMatrixView<T>(observed),
    fastpls::core::ConstMatrixView<T>(expected)
  );
}

template<class T>
void check() {
  using fastpls::core::CrosscovOperator;
  using fastpls::core::CenteredCrosscovOperator;
  using fastpls::core::ExplicitOperator;
  using fastpls::core::Matrix;
  using fastpls::core::ProjectedOperator;
  ReferenceBackend<T> backend;
  Matrix<T> x(17, 9);
  Matrix<T> y(17, 7);
  for (std::size_t column = 0; column < x.columns(); ++column) {
    for (std::size_t row = 0; row < x.rows(); ++row) {
      x(row, column) = static_cast<T>(1 + row + 3 * column) / T(19);
    }
  }
  for (std::size_t column = 0; column < y.columns(); ++column) {
    for (std::size_t row = 0; row < y.rows(); ++row) {
      y(row, column) = static_cast<T>(2 + 2 * row - column) / T(23);
    }
  }

  CrosscovOperator<T, ReferenceBackend<T>> implicit(
    x.view(), y.view(), 3, backend
  );
  Matrix<T> explicit_crosscov(9, 7);
  backend.gemm(x.view(), y.view(), true, false, explicit_crosscov.view());
  Matrix<T> right(7, 3);
  Matrix<T> left(9, 2);
  for (std::size_t index = 0; index < right.size(); ++index) {
    right.data()[index] = static_cast<T>(index + 1) / T(13);
  }
  for (std::size_t index = 0; index < left.size(); ++index) {
    left.data()[index] = static_cast<T>(index + 2) / T(17);
  }

  const T tolerance = sizeof(T) == sizeof(float) ? T(2e-5) : T(1e-12);
  std::vector<T> response_mean(y.columns(), T(0));
  Matrix<T> centered_y(y.rows(), y.columns());
  for (std::size_t column = 0; column < y.columns(); ++column) {
    for (std::size_t row = 0; row < y.rows(); ++row) {
      response_mean[column] += y(row, column);
    }
    response_mean[column] /= static_cast<T>(y.rows());
    for (std::size_t row = 0; row < y.rows(); ++row) {
      centered_y(row, column) = y(row, column) - response_mean[column];
    }
  }
  Matrix<T> centered_crosscov(x.columns(), y.columns());
  backend.gemm(
    x.view(), centered_y.view(), true, false, centered_crosscov.view()
  );
  CenteredCrosscovOperator<T, ReferenceBackend<T>> centered_operator(
    x.view(), y.view(), response_mean.data(), response_mean.size(), backend
  );
  Matrix<T> actual_centered;
  Matrix<T> expected_centered(x.columns(), right.columns());
  centered_operator.multiply(right.view(), false, actual_centered);
  backend.gemm(
    centered_crosscov.view(), right.view(), false, false,
    expected_centered.view()
  );
  assert(relative_error(
    actual_centered.view(), expected_centered.view()) < tolerance
  );
  expected_centered.resize(y.columns(), left.columns());
  centered_operator.multiply(left.view(), true, actual_centered);
  backend.gemm(
    centered_crosscov.view(), left.view(), true, false,
    expected_centered.view()
  );
  assert(relative_error(
    actual_centered.view(), expected_centered.view()) < tolerance
  );
  centered_operator.materialize(actual_centered);
  assert(relative_error(
    actual_centered.view(), centered_crosscov.view()) < tolerance
  );

  ProjectedOperator<
    T, CenteredCrosscovOperator<T, ReferenceBackend<T>>, ReferenceBackend<T>
  > projected(centered_operator, 3, backend);
  Matrix<T> projected_crosscov = centered_crosscov;
  for (std::size_t step = 0; step <= 3; ++step) {
    Matrix<T> actual;
    Matrix<T> expected(projected_crosscov.rows(), right.columns());
    projected.multiply(right.view(), false, actual);
    backend.gemm(
      projected_crosscov.view(), right.view(), false, false, expected.view()
    );
    assert(relative_error(actual.view(), expected.view()) < tolerance);
    expected.resize(projected_crosscov.columns(), left.columns());
    projected.multiply(left.view(), true, actual);
    backend.gemm(
      projected_crosscov.view(), left.view(), true, false, expected.view()
    );
    assert(relative_error(actual.view(), expected.view()) < tolerance);
    projected.materialize(actual);
    assert(relative_error(
      actual.view(), projected_crosscov.view()) < tolerance
    );
    if (step == 3) break;

    Matrix<T> direction(9, 1);
    for (std::size_t row = 0; row < direction.rows(); ++row) {
      direction(row, 0) = static_cast<T>(row == step ? 1 : 0);
    }
    projected.deflate(direction.view());
    for (std::size_t column = 0;
         column < projected_crosscov.columns(); ++column) {
      projected_crosscov(step, column) = T(0);
    }
  }
  assert(projected.deflations() == 3);
  assert(projected.basis().columns() == 3);

  for (std::size_t step = 0; step <= 3; ++step) {
    ExplicitOperator<T, ReferenceBackend<T>> explicit_operator(
      explicit_crosscov.view(), backend
    );
    Matrix<T> actual;
    Matrix<T> expected;
    implicit.multiply(right.view(), false, actual);
    explicit_operator.multiply(right.view(), false, expected);
    assert(relative_error(actual.view(), expected.view()) < tolerance);
    implicit.multiply(left.view(), true, actual);
    explicit_operator.multiply(left.view(), true, expected);
    assert(relative_error(actual.view(), expected.view()) < tolerance);
    if (step == 3) break;

    Matrix<T> direction(9, 1);
    T norm = T(0);
    for (std::size_t row = 0; row < direction.rows(); ++row) {
      direction(row, 0) = static_cast<T>(row + step + 1);
      norm += direction(row, 0) * direction(row, 0);
    }
    norm = std::sqrt(norm);
    for (std::size_t row = 0; row < direction.rows(); ++row) {
      direction(row, 0) /= norm;
    }
    implicit.deflate(direction.view());

    Matrix<T> projection(9, 9);
    for (std::size_t column = 0; column < 9; ++column) {
      for (std::size_t row = 0; row < 9; ++row) {
        projection(row, column) = (row == column ? T(1) : T(0)) -
          direction(row, 0) * direction(column, 0);
      }
    }
    Matrix<T> updated(9, 7);
    backend.gemm(
      projection.view(), explicit_crosscov.view(), false, false,
      updated.view()
    );
    explicit_crosscov = std::move(updated);
  }

  Matrix<T> materialized;
  implicit.materialize(materialized);
  assert(relative_error(materialized.view(), explicit_crosscov.view()) < tolerance);
  Matrix<T> empty(7, 0);
  Matrix<T> empty_result;
  implicit.multiply(empty.view(), false, empty_result);
  assert(empty_result.rows() == 9 && empty_result.columns() == 0);
  bool rejected = false;
  try {
    Matrix<T> direction(9, 1);
    implicit.deflate(direction.view());
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  assert(rejected);
}

}  // namespace

int main() {
  check<float>();
  check<double>();
  return 0;
}
