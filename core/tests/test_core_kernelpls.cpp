// SPDX-License-Identifier: MIT
#include "reference_backend.hpp"

#include <fastpls/core/kernelpls.hpp>

#include <cassert>
#include <cmath>
#include <cstddef>
#include <type_traits>

namespace {

template<class T>
void check() {
  fastpls::core::Matrix<T> predictors(40, 6);
  fastpls::core::Matrix<T> responses(40, 3);
  fastpls::core::Matrix<T> test(7, 6);
  for (std::size_t row = 0; row < predictors.rows(); ++row) {
    for (std::size_t column = 0; column < predictors.columns(); ++column) {
      predictors(row, column) = static_cast<T>(
        1.5 + std::sin(0.13 * (row + 1) * (column + 1))
      );
    }
    responses(row, 0) = predictors(row, 0) + predictors(row, 1);
    responses(row, 1) = predictors(row, 2) - predictors(row, 3);
    responses(row, 2) = predictors(row, 4) * predictors(row, 5);
  }
  for (std::size_t row = 0; row < test.rows(); ++row) {
    for (std::size_t column = 0; column < test.columns(); ++column) {
      test(row, column) = static_cast<T>(
        1.5 + std::cos(0.19 * (row + 1) * (column + 1))
      );
    }
  }

  ReferenceBackend<T> backend;
  for (const auto kernel : {
         fastpls::core::KernelType::linear,
         fastpls::core::KernelType::radial_basis,
         fastpls::core::KernelType::polynomial}) {
    fastpls::core::KernelPlsControls controls;
    controls.kernel = kernel;
    controls.gamma = 0.2;
    controls.scaling = fastpls::core::PredictorScaling::autoscaling;
    controls.simpls.components = 3;
    controls.simpls.rsvd.oversample = 3;
    controls.simpls.rsvd.power = 5;
    controls.simpls.rsvd.seed = 37;
    const auto model = fastpls::core::fit_kernelpls(
      predictors, responses.view(), controls, backend
    );
    assert(model.inner.completed_components == 3);
    const auto training = fastpls::core::predict_kernelpls(
      model, predictors, 3, backend
    );
    const auto held_out = fastpls::core::predict_kernelpls(
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
    if (kernel == fastpls::core::KernelType::linear) {
      assert(model.reference.size() == 0);
      assert(model.kernel_column_means.empty());
    } else {
      assert(model.reference.rows() == predictors.rows());
      assert(model.kernel_column_means.size() == predictors.rows());
    }
  }
}

}  // namespace

int main() {
  check<float>();
  check<double>();
  return 0;
}
