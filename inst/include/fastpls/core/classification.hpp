// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Stefano Cacciatore
#ifndef FASTPLS_CORE_CLASSIFICATION_HPP
#define FASTPLS_CORE_CLASSIFICATION_HPP

#include <fastpls/core/matrix.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace fastpls {
namespace core {

enum class PredictorScaling {
  centering = 1,
  autoscaling = 2,
  none = 3
};

template<class T>
struct LabelCrossprodResult {
  Matrix<T> crossprod;
  std::vector<T> predictor_center;
  std::vector<T> predictor_scale;
  std::vector<T> response_mean;
  std::vector<T> class_counts;
};

template<class T, class Label>
LabelCrossprodResult<T> scaled_label_crossprod(
    ConstMatrixView<T> predictors, const Label* labels,
    std::size_t label_count, std::size_t class_count,
    PredictorScaling scaling) {
  if (predictors.data() == nullptr || labels == nullptr ||
      predictors.empty() || predictors.rows() != label_count ||
      class_count < 2) {
    throw std::invalid_argument(
      "fastPLS scaled label cross-product dimensions are invalid"
    );
  }
  LabelCrossprodResult<T> result;
  result.crossprod.resize(predictors.columns(), class_count);
  result.predictor_center.assign(predictors.columns(), T(0));
  result.predictor_scale.assign(predictors.columns(), T(1));
  result.response_mean.assign(class_count, T(0));
  result.class_counts.assign(class_count, T(0));

  for (std::size_t sample = 0; sample < predictors.rows(); ++sample) {
    const std::size_t label = static_cast<std::size_t>(labels[sample]);
    if (label >= class_count) {
      throw std::invalid_argument(
        "fastPLS scaled label cross-product contains an invalid class index"
      );
    }
    result.class_counts[label] += T(1);
  }
  for (std::size_t response = 0; response < class_count; ++response) {
    result.response_mean[response] = result.class_counts[response] /
      static_cast<T>(predictors.rows());
  }

  std::vector<T> class_sums(class_count);
  for (std::size_t predictor = 0;
       predictor < predictors.columns(); ++predictor) {
    T sum = T(0);
    T sum_squares = T(0);
    for (std::size_t sample = 0; sample < predictors.rows(); ++sample) {
      const T value = predictors(sample, predictor);
      sum += value;
      sum_squares += value * value;
    }
    if (scaling != PredictorScaling::none) {
      result.predictor_center[predictor] =
        sum / static_cast<T>(predictors.rows());
    }
    if (scaling == PredictorScaling::autoscaling) {
      const T centered_sum_squares = std::max(
        T(0), sum_squares - static_cast<T>(predictors.rows()) *
          result.predictor_center[predictor] *
          result.predictor_center[predictor]
      );
      T standard_deviation = std::sqrt(
        centered_sum_squares /
        static_cast<T>(std::max<std::size_t>(predictors.rows() - 1, 1))
      );
      if (!std::isfinite(standard_deviation) || standard_deviation <= T(0)) {
        standard_deviation = T(1);
      }
      result.predictor_scale[predictor] = standard_deviation;
    }

    std::fill(class_sums.begin(), class_sums.end(), T(0));
    T total = T(0);
    for (std::size_t sample = 0; sample < predictors.rows(); ++sample) {
      const std::size_t label = static_cast<std::size_t>(labels[sample]);
      const T standardized =
        (predictors(sample, predictor) -
         result.predictor_center[predictor]) /
        result.predictor_scale[predictor];
      class_sums[label] += standardized;
      total += standardized;
    }
    for (std::size_t response = 0; response < class_count; ++response) {
      result.crossprod(predictor, response) = class_sums[response] -
        total * result.response_mean[response];
    }
  }
  return result;
}

template<class T, class Label>
LabelCrossprodResult<T> scaled_label_crossprod(
    MatrixView<T> predictors, const Label* labels,
    std::size_t label_count, std::size_t class_count,
    PredictorScaling scaling) {
  return scaled_label_crossprod(
    ConstMatrixView<T>(predictors), labels, label_count, class_count, scaling
  );
}

template<class T, class Label>
void centered_label_crossprod(ConstMatrixView<T> predictors,
                              const Label* labels,
                              std::size_t label_count,
                              const T* center_offset,
                              std::size_t class_count,
                              MatrixView<T> output) {
  if (predictors.data() == nullptr || labels == nullptr ||
      center_offset == nullptr || output.data() == nullptr) {
    throw std::invalid_argument(
      "fastPLS label cross-product received a null buffer"
    );
  }
  if (predictors.rows() != label_count ||
      output.rows() != predictors.columns() ||
      output.columns() != class_count) {
    throw std::invalid_argument(
      "fastPLS label cross-product dimensions are inconsistent"
    );
  }

  std::vector<T> class_sums(class_count);
  for (std::size_t predictor = 0;
       predictor < predictors.columns(); ++predictor) {
    std::fill(class_sums.begin(), class_sums.end(), T(0));
    T total = T(0);
    for (std::size_t sample = 0; sample < predictors.rows(); ++sample) {
      const std::size_t label = static_cast<std::size_t>(labels[sample]);
      if (label >= class_count) {
        throw std::invalid_argument(
          "fastPLS label cross-product contains an invalid class index"
        );
      }
      const T value = predictors.data()[
        sample + predictor * predictors.leading_dimension()
      ];
      total += value;
      class_sums[label] += value;
    }
    for (std::size_t response = 0; response < class_count; ++response) {
      output.data()[predictor + response * output.leading_dimension()] =
        class_sums[response] + total * center_offset[response];
    }
  }
}

template<class T, class Label>
void centered_label_crossprod(MatrixView<T> predictors,
                              const Label* labels,
                              std::size_t label_count,
                              const T* center_offset,
                              std::size_t class_count,
                              MatrixView<T> output) {
  centered_label_crossprod(
    ConstMatrixView<T>(predictors), labels, label_count, center_offset,
    class_count, output
  );
}

template<class T>
std::size_t row_argmax(ConstMatrixView<T> values, std::size_t row) {
  if (values.empty() || row >= values.rows()) {
    throw std::invalid_argument("fastPLS argmax dimensions are invalid");
  }
  std::size_t best = 0;
  T best_value = values.data()[row];
  for (std::size_t column = 1; column < values.columns(); ++column) {
    const T value = values.data()[
      row + column * values.leading_dimension()
    ];
    if (value > best_value) {
      best = column;
      best_value = value;
    }
  }
  return best;
}

template<class T>
std::size_t row_argmax(MatrixView<T> values, std::size_t row) {
  return row_argmax(ConstMatrixView<T>(values), row);
}

template<class T>
void row_top_k(ConstMatrixView<T> values, std::size_t row,
               std::size_t keep, std::vector<std::size_t>& workspace,
               std::size_t* indices, T* scores) {
  if (values.empty() || row >= values.rows() || keep == 0 ||
      keep > values.columns() || indices == nullptr || scores == nullptr) {
    throw std::invalid_argument("fastPLS top-rank dimensions are invalid");
  }
  workspace.resize(values.columns());
  for (std::size_t column = 0; column < values.columns(); ++column) {
    workspace[column] = column;
  }
  const auto before = [&values, row](std::size_t left, std::size_t right) {
    const T lhs = values(row, left);
    const T rhs = values(row, right);
    if (std::isnan(lhs)) return false;
    if (std::isnan(rhs)) return true;
    return lhs == rhs ? left < right : lhs > rhs;
  };
  std::partial_sort(
    workspace.begin(), workspace.begin() + keep, workspace.end(), before
  );
  for (std::size_t rank = 0; rank < keep; ++rank) {
    indices[rank] = workspace[rank];
    scores[rank] = values(row, workspace[rank]);
  }
}

template<class T>
void row_top_k(MatrixView<T> values, std::size_t row,
               std::size_t keep, std::vector<std::size_t>& workspace,
               std::size_t* indices, T* scores) {
  row_top_k(
    ConstMatrixView<T>(values), row, keep, workspace, indices, scores
  );
}

}  // namespace core
}  // namespace fastpls

#endif
