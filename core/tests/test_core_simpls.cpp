// SPDX-License-Identifier: MIT
#include "reference_backend.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>

namespace {

template<class T>
void check() {
  using fastpls::core::Matrix;
  using fastpls::core::SimplsControls;
  using fastpls::core::SimplsWorkspace;

  ReferenceBackend<T> backend;
  Matrix<T> predictors(8, 3);
  const T values[24] = {
    T(-3), T(-2), T(-1), T(0), T(0), T(1), T(2), T(3),
    T(1), T(-1), T(1), T(-1), T(1), T(-1), T(1), T(-1),
    T(-1), T(2), T(-2), T(1), T(1), T(-2), T(2), T(-1)
  };
  for (std::size_t index = 0; index < 24; ++index) {
    predictors.data()[index] = values[index];
  }

  Matrix<T> coefficients(3, 3);
  coefficients(0, 0) = T(1.5);
  coefficients(1, 0) = T(-0.5);
  coefficients(0, 1) = T(-0.75);
  coefficients(1, 1) = T(2);
  coefficients(0, 2) = T(0.25);
  coefficients(1, 2) = T(1.25);
  Matrix<T> response(8, 3);
  backend.gemm(
    predictors.view(), coefficients.view(), false, false, response.view()
  );
  Matrix<T> crosscov(3, 3);
  backend.gemm(
    predictors.view(), response.view(), true, false, crosscov.view()
  );

  SimplsControls controls;
  controls.components = 2;
  controls.maximum_block = 1;
  controls.cache_predictor_crossprod = false;
  controls.reorthogonalize = true;
  controls.store_scores = true;
  controls.rsvd.oversample = 0;
  controls.rsvd.power = 5;
  controls.rsvd.seed = 19;
  SimplsWorkspace<T> workspace;
  auto model = fastpls::core::fit_simpls_preprocessed<T>(
    predictors.view(), crosscov.view(), controls, backend, workspace
  );
  assert(model.completed_components == 2);

  Matrix<T> prediction = fastpls::core::predict_simpls_preprocessed<T>(
    predictors.view(), model, 2, backend
  );
  T squared_error = T(0);
  T response_sum_squares = T(0);
  for (std::size_t index = 0; index < response.size(); ++index) {
    const T difference = prediction.data()[index] - response.data()[index];
    squared_error += difference * difference;
    response_sum_squares += response.data()[index] * response.data()[index];
  }
  const T relative_error = std::sqrt(squared_error / response_sum_squares);
  const T tolerance = sizeof(T) == sizeof(float) ? T(2e-4) : T(1e-8);
  assert(relative_error < tolerance);

  for (std::size_t left = 0; left < 2; ++left) {
    for (std::size_t right = 0; right < 2; ++right) {
      T score_product = T(0);
      T basis_product = T(0);
      for (std::size_t row = 0; row < predictors.rows(); ++row) {
        score_product += model.scores(row, left) * model.scores(row, right);
      }
      for (std::size_t row = 0; row < predictors.columns(); ++row) {
        basis_product += model.deflation_basis(row, left) *
          model.deflation_basis(row, right);
      }
      const T expected = left == right ? T(1) : T(0);
      assert(std::abs(score_product - expected) < tolerance);
      assert(std::abs(basis_product - expected) < tolerance);
    }
  }
}

}  // namespace

int main() {
  check<float>();
  check<double>();
  return 0;
}
