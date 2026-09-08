// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Stefano Cacciatore
#ifndef FASTPLS_CORE_STATISTICS_HPP
#define FASTPLS_CORE_STATISTICS_HPP

#include <fastpls/core/matrix.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace fastpls::core {

template<typename Scalar>
double observed_mean_r2(BasicMatrixView<const Scalar> observed,
                        BasicMatrixView<const Scalar> predicted) {
  if (observed.rows() != predicted.rows() ||
      observed.columns() != predicted.columns()) {
    throw std::invalid_argument("R2 matrices must have matching dimensions");
  }
  double total_sum_squares = 0.0;
  double prediction_sum_squares = 0.0;
  for (std::size_t column = 0; column < observed.columns(); ++column) {
    double mean = 0.0;
    for (std::size_t row = 0; row < observed.rows(); ++row) {
      mean += static_cast<double>(observed(row, column));
    }
    mean /= static_cast<double>(observed.rows());
    for (std::size_t row = 0; row < observed.rows(); ++row) {
      const double value = static_cast<double>(observed(row, column));
      const double error = value - static_cast<double>(predicted(row, column));
      const double centered = value - mean;
      prediction_sum_squares += error * error;
      total_sum_squares += centered * centered;
    }
  }
  return 1.0 - prediction_sum_squares / total_sum_squares;
}

enum class CorrelationStatus {
  success,
  insufficient_pairs,
  no_complete_pairs,
  constant_input
};

struct CorrelationResult {
  double value;
  std::size_t complete_pairs;
  CorrelationStatus status;
};

namespace detail {

struct RankEntry {
  double value;
  std::size_t pair;
};

inline bool rank_values(std::vector<RankEntry>& entries,
                        std::vector<double>& ranks) {
  std::sort(entries.begin(), entries.end(),
            [](const RankEntry& left, const RankEntry& right) {
              return left.value < right.value;
            });
  if (entries.front().value == entries.back().value) return false;
  for (std::size_t begin = 0; begin < entries.size();) {
    std::size_t end = begin + 1;
    while (end < entries.size() &&
           entries[end].value == entries[begin].value) {
      ++end;
    }
    const double rank =
      (static_cast<double>(begin) + static_cast<double>(end) + 1.0) / 2.0;
    for (std::size_t index = begin; index < end; ++index) {
      ranks[entries[index].pair] = rank;
    }
    begin = end;
  }
  return true;
}

}  // namespace detail

inline CorrelationResult spearman_correlation(const double* observed,
                                              const double* predicted,
                                              std::size_t size) {
  std::vector<detail::RankEntry> observed_entries;
  std::vector<double> paired_prediction;
  observed_entries.reserve(size);
  paired_prediction.reserve(size);
  for (std::size_t index = 0; index < size; ++index) {
    if (!std::isnan(observed[index]) && !std::isnan(predicted[index])) {
      observed_entries.push_back({observed[index], observed_entries.size()});
      paired_prediction.push_back(predicted[index]);
    }
  }
  const std::size_t pairs = observed_entries.size();
  if (pairs == 0) {
    return {NAN, 0, CorrelationStatus::no_complete_pairs};
  }
  if (pairs < 2) {
    return {NAN, pairs, CorrelationStatus::insufficient_pairs};
  }

  std::vector<double> observed_ranks(pairs);
  if (!detail::rank_values(observed_entries, observed_ranks)) {
    return {NAN, pairs, CorrelationStatus::constant_input};
  }
  std::vector<detail::RankEntry> predicted_entries(pairs);
  for (std::size_t index = 0; index < pairs; ++index) {
    predicted_entries[index] = {paired_prediction[index], index};
  }
  std::vector<double> predicted_ranks(pairs);
  if (!detail::rank_values(predicted_entries, predicted_ranks)) {
    return {NAN, pairs, CorrelationStatus::constant_input};
  }

  const long double mean = (static_cast<long double>(pairs) + 1.0L) / 2.0L;
  long double cross = 0.0L;
  long double observed_sum = 0.0L;
  long double predicted_sum = 0.0L;
  for (std::size_t index = 0; index < pairs; ++index) {
    const long double left = observed_ranks[index] - mean;
    const long double right = predicted_ranks[index] - mean;
    cross += left * right;
    observed_sum += left * left;
    predicted_sum += right * right;
  }
  const long double denominator = std::sqrt(observed_sum * predicted_sum);
  if (!(denominator > 0.0L)) {
    return {NAN, pairs, CorrelationStatus::constant_input};
  }
  return {static_cast<double>(cross / denominator), pairs,
          CorrelationStatus::success};
}

}  // namespace fastpls::core

#endif
