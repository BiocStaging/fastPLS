// SPDX-License-Identifier: MIT
#include <fastpls/core.hpp>

#include <cassert>
#include <cmath>

int main() {
  fastpls::core::Matrix<float> scores(8, 3);
  const int labels[] = {1, 1, 1, 1, 2, 2, 2, 2};
  for (std::size_t row = 0; row < scores.rows(); ++row) {
    const float group = row < 4 ? -2.0f : 2.0f;
    scores(row, 0) = group + 0.1f * static_cast<float>(row % 4);
    scores(row, 1) = 0.2f * static_cast<float>(row);
    scores(row, 2) = scores(row, 1);  // Singular without regularization.
  }
  const int prefixes[] = {1, 3};
  const auto models = fastpls::core::train_lda_prefixes(
    scores.view(), labels, 8, 2, prefixes, 2
  );
  assert(models.size() == 2);
  assert(models[0].linear.columns() == 1);
  assert(models[1].linear.columns() == 3);
  assert(models[1].relative_ridge >= 1e-8f);

  const auto predictions = fastpls::core::lda_predict(
    scores.view(), models[1]
  );
  for (std::size_t row = 0; row < predictions.size(); ++row) {
    assert(predictions[row] == labels[row]);
  }
  const auto discriminants = fastpls::core::lda_scores(
    scores.view(), models[1]
  );
  assert(discriminants.rows() == scores.rows());
  assert(discriminants.columns() == 2);
  for (std::size_t index = 0; index < discriminants.size(); ++index) {
    assert(std::isfinite(discriminants.data()[index]));
  }

  bool rejected = false;
  const int invalid_labels[] = {1, 1, 1, 1, 3, 2, 2, 2};
  try {
    fastpls::core::train_lda_prefixes(
      scores.view(), invalid_labels, 8, 2, prefixes, 2
    );
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  assert(rejected);
  return 0;
}
