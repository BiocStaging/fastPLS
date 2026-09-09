// SPDX-License-Identifier: MIT
#include <fastpls/core.hpp>

#include "reference_backend.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

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

  const auto scaled = fastpls::core::scaled_label_crossprod(
    fastpls::core::ConstMatrixView<float>(x.view()), labels, 4, 2,
    fastpls::core::PredictorScaling::autoscaling
  );
  assert(scaled.crossprod.rows() == 3);
  assert(scaled.crossprod.columns() == 2);
  assert(std::abs(scaled.predictor_center[0] - 2.5f) < 1e-6f);
  assert(std::abs(scaled.predictor_scale[0] - 1.2909944f) < 1e-6f);
  assert(std::abs(scaled.response_mean[0] - 0.5f) < 1e-6f);
  assert(std::abs(scaled.class_counts[1] - 2.0f) < 1e-6f);
  assert(std::abs(scaled.crossprod(0, 0) + 0.7745967f) < 1e-6f);
  assert(std::abs(scaled.crossprod(0, 1) - 0.7745967f) < 1e-6f);

  fastpls::core::Matrix<double> x_double(4, 1);
  for (std::size_t row = 0; row < x_double.rows(); ++row) {
    x_double(row, 0) = static_cast<double>(1 + row);
  }
  const fastpls::core::Matrix<double>& x_double_const = x_double;
  const auto centered_double = fastpls::core::scaled_label_crossprod(
    x_double_const.view(), labels, 4, 2,
    fastpls::core::PredictorScaling::centering
  );
  assert(std::abs(centered_double.crossprod(0, 0) + 1.0) < 1e-15);
  assert(std::abs(centered_double.crossprod(0, 1) - 1.0) < 1e-15);

  fastpls::core::Matrix<float> x_grouped(4, 1);
  for (std::size_t row = 0; row < x_grouped.rows(); ++row) {
    x_grouped(row, 0) = static_cast<float>(1 + row);
  }
  const std::size_t grouped_labels[] = {0, 0, 1, 1};
  ReferenceBackend<float> reference_backend;
  const auto grouped = fastpls::core::prepare_scaled_label_crossprod(
    x_grouped.view(), grouped_labels, 4, 2,
    fastpls::core::PredictorScaling::none, reference_backend
  );
  assert(std::abs(grouped.crossprod(0, 0) + 2.0f) < 1e-6f);
  assert(std::abs(grouped.crossprod(0, 1) - 2.0f) < 1e-6f);

  fastpls::core::Matrix<double> scores(2, 3);
  scores(0, 0) = 0.2;
  scores(0, 1) = 0.8;
  scores(0, 2) = 0.1;
  scores(1, 0) = 2.0;
  scores(1, 1) = 1.0;
  scores(1, 2) = 0.0;
  assert(fastpls::core::row_argmax(scores.view(), 0) == 1);
  assert(fastpls::core::row_argmax(scores.view(), 1) == 0);

  std::vector<std::size_t> workspace;
  std::size_t top_indices[2] = {0, 0};
  double top_scores[2] = {0.0, 0.0};
  fastpls::core::row_top_k(
    scores.view(), 0, 2, workspace, top_indices, top_scores
  );
  assert(top_indices[0] == 1);
  assert(top_indices[1] == 0);
  assert(std::abs(top_scores[0] - 0.8) < 1e-15);
  assert(std::abs(top_scores[1] - 0.2) < 1e-15);

  scores(1, 1) = 2.0;
  fastpls::core::row_top_k(
    scores.view(), 1, 2, workspace, top_indices, top_scores
  );
  assert(top_indices[0] == 0);
  assert(top_indices[1] == 1);

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
