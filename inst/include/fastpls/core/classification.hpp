// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Stefano Cacciatore
#ifndef FASTPLS_CORE_CLASSIFICATION_HPP
#define FASTPLS_CORE_CLASSIFICATION_HPP

#include <fastpls/core/matrix.hpp>

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace fastpls {
namespace core {

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

}  // namespace core
}  // namespace fastpls

#endif
