// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Stefano Cacciatore
#ifndef FASTPLS_CORE_LDA_HPP
#define FASTPLS_CORE_LDA_HPP

#include <fastpls/core/classification.hpp>
#include <fastpls/core/matrix.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace fastpls {
namespace core {

template<class T>
struct LdaModel {
  Matrix<T> means;
  Matrix<T> linear;
  std::vector<T> constants;
  std::vector<T> priors;
  T ridge = T(0);
  T relative_ridge = T(0);
};

namespace detail {

template<class T>
bool lda_cholesky_solve(ConstMatrixView<T> pooled,
                        ConstMatrixView<T> means, T ridge,
                        Matrix<T>& linear) {
  const std::size_t components = pooled.rows();
  const std::size_t classes = means.rows();
  Matrix<T> lower(components, components);
  for (std::size_t row = 0; row < components; ++row) {
    for (std::size_t column = 0; column <= row; ++column) {
      T value = pooled(row, column);
      if (row == column) value += ridge;
      for (std::size_t inner = 0; inner < column; ++inner) {
        value -= lower(row, inner) * lower(column, inner);
      }
      if (row == column) {
        if (!std::isfinite(value) || value <= T(0)) return false;
        lower(row, column) = std::sqrt(value);
      } else {
        const T diagonal = lower(column, column);
        if (!std::isfinite(diagonal) || diagonal <= T(0)) return false;
        lower(row, column) = value / diagonal;
      }
    }
  }

  Matrix<T> solution(components, classes);
  for (std::size_t class_index = 0; class_index < classes; ++class_index) {
    for (std::size_t row = 0; row < components; ++row) {
      T value = means(class_index, row);
      for (std::size_t inner = 0; inner < row; ++inner) {
        value -= lower(row, inner) * solution(inner, class_index);
      }
      solution(row, class_index) = value / lower(row, row);
    }
    for (std::size_t reverse = components; reverse > 0; --reverse) {
      const std::size_t row = reverse - 1;
      T value = solution(row, class_index);
      for (std::size_t inner = row + 1; inner < components; ++inner) {
        value -= lower(inner, row) * solution(inner, class_index);
      }
      value /= lower(row, row);
      if (!std::isfinite(value)) return false;
      solution(row, class_index) = value;
    }
  }
  linear.resize(classes, components);
  for (std::size_t class_index = 0; class_index < classes; ++class_index) {
    for (std::size_t component = 0; component < components; ++component) {
      linear(class_index, component) = solution(component, class_index);
    }
  }
  return true;
}

template<class T>
std::vector<LdaModel<T>> finalize_lda_prefixes(
    ConstMatrixView<T> pooled, ConstMatrixView<T> means,
    const std::vector<T>& counts, std::size_t sample_count,
    const int* prefixes, std::size_t prefix_count) {
  constexpr T ridge_grid[] = {
    T(1e-8), T(1e-6), T(1e-5), T(1e-4), T(1e-3), T(1e-2)
  };
  std::vector<LdaModel<T>> models;
  models.reserve(prefix_count);
  for (std::size_t index = 0; index < prefix_count; ++index) {
    const std::size_t retained = static_cast<std::size_t>(prefixes[index]);
    LdaModel<T> model;
    model.means.resize(means.rows(), retained);
    Matrix<T> covariance(retained, retained);
    T scale = T(0);
    for (std::size_t component = 0; component < retained; ++component) {
      scale += pooled(component, component);
      for (std::size_t row = 0; row < retained; ++row) {
        covariance(row, component) = pooled(row, component);
      }
      for (std::size_t class_index = 0;
           class_index < means.rows(); ++class_index) {
        model.means(class_index, component) =
          means(class_index, component);
      }
    }
    scale /= static_cast<T>(retained);
    if (!std::isfinite(scale) || scale <= T(0)) scale = T(1);

    bool solved = false;
    for (const T relative : ridge_grid) {
      const T ridge = relative * scale;
      if (lda_cholesky_solve(
          ConstMatrixView<T>(covariance.view()),
          ConstMatrixView<T>(model.means.view()), ridge, model.linear)) {
        model.ridge = ridge;
        model.relative_ridge = relative;
        solved = true;
        break;
      }
    }
    if (!solved) {
      throw std::runtime_error(
        "fastPLS LDA Cholesky factorization failed for every regularization level"
      );
    }

    model.constants.resize(means.rows());
    model.priors.resize(means.rows());
    for (std::size_t class_index = 0;
         class_index < means.rows(); ++class_index) {
      model.priors[class_index] = counts[class_index] /
        static_cast<T>(sample_count);
      T product = T(0);
      for (std::size_t component = 0; component < retained; ++component) {
        product += model.means(class_index, component) *
          model.linear(class_index, component);
      }
      model.constants[class_index] = T(-0.5) * product +
        std::log(std::max(model.priors[class_index],
                          std::numeric_limits<T>::min()));
    }
    models.push_back(std::move(model));
  }
  return models;
}

}  // namespace detail

template<class T>
std::vector<LdaModel<T>> train_lda_prefixes(
    ConstMatrixView<T> scores, const int* labels, std::size_t label_count,
    std::size_t class_count, const int* prefixes, std::size_t prefix_count) {
  if (scores.empty() || labels == nullptr || prefixes == nullptr ||
      scores.rows() != label_count || class_count < 2 || prefix_count < 1) {
    throw std::invalid_argument("fastPLS LDA training dimensions are invalid");
  }
  std::size_t maximum = 0;
  for (std::size_t index = 0; index < prefix_count; ++index) {
    if (prefixes[index] < 1 ||
        static_cast<std::size_t>(prefixes[index]) > scores.columns()) {
      throw std::invalid_argument(
        "fastPLS LDA component counts must be within the score dimension"
      );
    }
    maximum = std::max(maximum, static_cast<std::size_t>(prefixes[index]));
  }

  std::vector<T> counts(class_count, T(0));
  Matrix<T> means(class_count, maximum);
  for (std::size_t sample = 0; sample < scores.rows(); ++sample) {
    const int encoded = labels[sample] - 1;
    if (encoded < 0 || static_cast<std::size_t>(encoded) >= class_count) {
      throw std::invalid_argument(
        "fastPLS LDA labels must be encoded as 1..n_classes"
      );
    }
    const std::size_t class_index = static_cast<std::size_t>(encoded);
    counts[class_index] += T(1);
    for (std::size_t component = 0; component < maximum; ++component) {
      means(class_index, component) += scores(sample, component);
    }
  }
  for (std::size_t class_index = 0; class_index < class_count; ++class_index) {
    if (counts[class_index] <= T(0)) {
      throw std::invalid_argument("fastPLS LDA received an empty class");
    }
    for (std::size_t component = 0; component < maximum; ++component) {
      means(class_index, component) /= counts[class_index];
    }
  }

  Matrix<T> pooled(maximum, maximum);
  const T denominator = static_cast<T>(std::max<std::size_t>(
    1, scores.rows() > class_count ? scores.rows() - class_count : 1
  ));
  for (std::size_t column = 0; column < maximum; ++column) {
    for (std::size_t row = 0; row <= column; ++row) {
      T value = T(0);
      for (std::size_t sample = 0; sample < scores.rows(); ++sample) {
        value += scores(sample, row) * scores(sample, column);
      }
      for (std::size_t class_index = 0;
           class_index < class_count; ++class_index) {
        value -= counts[class_index] * means(class_index, row) *
          means(class_index, column);
      }
      value /= denominator;
      pooled(row, column) = value;
      pooled(column, row) = value;
    }
  }

  return detail::finalize_lda_prefixes<T>(
    pooled.view(), means.view(), counts, scores.rows(), prefixes, prefix_count
  );
}

template<class T>
std::vector<LdaModel<T>> train_lda_prefixes(
    MatrixView<T> scores, const int* labels, std::size_t label_count,
    std::size_t class_count, const int* prefixes, std::size_t prefix_count) {
  return train_lda_prefixes(
    ConstMatrixView<T>(scores), labels, label_count, class_count,
    prefixes, prefix_count
  );
}

template<class T>
std::vector<LdaModel<T>> train_lda_prefixes_from_moments(
    ConstMatrixView<T> gram, ConstMatrixView<T> class_sums,
    const T* counts, std::size_t count_size, std::size_t sample_count,
    const int* prefixes, std::size_t prefix_count) {
  if (gram.empty() || gram.rows() != gram.columns() || class_sums.empty() ||
      class_sums.columns() != gram.columns() || class_sums.rows() < 2 ||
      counts == nullptr || count_size != class_sums.rows() ||
      sample_count < 1 || prefixes == nullptr || prefix_count < 1) {
    throw std::invalid_argument("fastPLS LDA moment dimensions are invalid");
  }
  std::size_t maximum = 0;
  for (std::size_t index = 0; index < prefix_count; ++index) {
    if (prefixes[index] < 1 ||
        static_cast<std::size_t>(prefixes[index]) > gram.columns()) {
      throw std::invalid_argument(
        "fastPLS LDA component counts must be within the moment dimension"
      );
    }
    maximum = std::max(maximum, static_cast<std::size_t>(prefixes[index]));
  }

  std::vector<T> count_values(counts, counts + count_size);
  Matrix<T> means(class_sums.rows(), maximum);
  T total_count = T(0);
  for (std::size_t class_index = 0;
       class_index < class_sums.rows(); ++class_index) {
    if (!std::isfinite(count_values[class_index]) ||
        count_values[class_index] <= T(0)) {
      throw std::invalid_argument("fastPLS LDA received an empty class");
    }
    total_count += count_values[class_index];
    for (std::size_t component = 0; component < maximum; ++component) {
      const T value = class_sums(class_index, component);
      if (!std::isfinite(value)) {
        throw std::invalid_argument("fastPLS LDA moments must be finite");
      }
      means(class_index, component) = value / count_values[class_index];
    }
  }
  const T expected = static_cast<T>(sample_count);
  if (std::abs(total_count - expected) >
      T(1e-8) * std::max(T(1), expected)) {
    throw std::invalid_argument("fastPLS LDA class counts do not sum to n");
  }

  Matrix<T> pooled(maximum, maximum);
  const T denominator = static_cast<T>(std::max<std::size_t>(
    1, sample_count > count_size ? sample_count - count_size : 1
  ));
  for (std::size_t column = 0; column < maximum; ++column) {
    for (std::size_t row = 0; row <= column; ++row) {
      T value = gram(row, column);
      if (!std::isfinite(value)) {
        throw std::invalid_argument("fastPLS LDA moments must be finite");
      }
      for (std::size_t class_index = 0;
           class_index < class_sums.rows(); ++class_index) {
        value -= count_values[class_index] * means(class_index, row) *
          means(class_index, column);
      }
      value /= denominator;
      pooled(row, column) = value;
      pooled(column, row) = value;
    }
  }
  return detail::finalize_lda_prefixes<T>(
    pooled.view(), means.view(), count_values, sample_count,
    prefixes, prefix_count
  );
}

template<class T>
Matrix<T> lda_scores(ConstMatrixView<T> scores, const LdaModel<T>& model) {
  if (scores.empty() || scores.columns() != model.linear.columns() ||
      model.linear.rows() != model.constants.size()) {
    throw std::invalid_argument("fastPLS LDA prediction dimensions are invalid");
  }
  Matrix<T> output(scores.rows(), model.linear.rows());
  for (std::size_t class_index = 0;
       class_index < model.linear.rows(); ++class_index) {
    for (std::size_t sample = 0; sample < scores.rows(); ++sample) {
      T value = model.constants[class_index];
      for (std::size_t component = 0;
           component < scores.columns(); ++component) {
        value += scores(sample, component) *
          model.linear(class_index, component);
      }
      output(sample, class_index) = value;
    }
  }
  return output;
}

template<class T>
Matrix<T> lda_scores(MatrixView<T> scores, const LdaModel<T>& model) {
  return lda_scores(ConstMatrixView<T>(scores), model);
}

template<class T>
std::vector<int> lda_predict(ConstMatrixView<T> scores,
                             const LdaModel<T>& model) {
  const Matrix<T> discriminants = lda_scores(scores, model);
  std::vector<int> labels(discriminants.rows());
  for (std::size_t row = 0; row < discriminants.rows(); ++row) {
    labels[row] = static_cast<int>(row_argmax(discriminants.view(), row) + 1);
  }
  return labels;
}

template<class T>
std::vector<int> lda_predict(MatrixView<T> scores,
                             const LdaModel<T>& model) {
  return lda_predict(ConstMatrixView<T>(scores), model);
}

}  // namespace core
}  // namespace fastpls

#endif
