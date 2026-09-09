// SPDX-License-Identifier: MIT
#include "reference_backend.hpp"

#include <fastpls/core/oplspls.hpp>

#include <cassert>
#include <cmath>
#include <cstddef>

namespace {

template<class T>
void check(bool randomized_filter) {
  fastpls::core::Matrix<T> predictors(48, 8);
  fastpls::core::Matrix<T> responses(48, 3);
  fastpls::core::Matrix<T> test(9, 8);
  for (std::size_t row = 0; row < predictors.rows(); ++row) {
    for (std::size_t column = 0; column < predictors.columns(); ++column) {
      predictors(row, column) = static_cast<T>(
        2.0 + std::sin(0.09 * (row + 1) * (column + 1))
      );
    }
    responses(row, 0) = predictors(row, 0) + predictors(row, 1);
    responses(row, 1) = predictors(row, 2) - predictors(row, 3);
    responses(row, 2) = predictors(row, 4) + T(0.3) * predictors(row, 5);
  }
  for (std::size_t row = 0; row < test.rows(); ++row) {
    for (std::size_t column = 0; column < test.columns(); ++column) {
      test(row, column) = static_cast<T>(
        2.0 + std::cos(0.14 * (row + 1) * (column + 1))
      );
    }
  }

  ReferenceBackend<T> backend;
  fastpls::core::OplsControls controls;
  controls.orthogonal_components = 2;
  controls.scaling = fastpls::core::PredictorScaling::autoscaling;
  controls.randomized_filter = randomized_filter;
  controls.filter_rsvd.oversample = 4;
  controls.filter_rsvd.power = 5;
  controls.filter_rsvd.seed = 47;
  controls.simpls.components = 3;
  controls.simpls.rsvd.oversample = 3;
  controls.simpls.rsvd.power = 5;
  controls.simpls.rsvd.seed = 53;

  const auto model = fastpls::core::fit_opls(
    predictors, responses.view(), controls, backend
  );
  assert(model.filter.predictors.size() == 0);
  assert(model.filter.completed_components == 2);
  assert(model.inner.completed_components == 3);
  const auto training = fastpls::core::predict_opls(
    model, predictors, 3, backend
  );
  const auto held_out = fastpls::core::predict_opls(
    model, test, 3, backend
  );
  assert(training.rows() == predictors.rows());
  assert(training.columns() == responses.columns());
  assert(held_out.rows() == test.rows());
  assert(held_out.columns() == responses.columns());
  for (std::size_t index = 0; index < training.size(); ++index) {
    assert(std::isfinite(training.data()[index]));
  }
  for (std::size_t index = 0; index < held_out.size(); ++index) {
    assert(std::isfinite(held_out.data()[index]));
  }
}

}  // namespace

int main() {
  check<float>(false);
  check<double>(false);
  check<float>(true);
  check<double>(true);
  return 0;
}
