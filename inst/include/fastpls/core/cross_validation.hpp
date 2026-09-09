// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Stefano Cacciatore
#ifndef FASTPLS_CORE_CROSS_VALIDATION_HPP
#define FASTPLS_CORE_CROSS_VALIDATION_HPP

#include <fastpls/core/classification.hpp>
#include <fastpls/core/lda.hpp>
#include <fastpls/core/plssvd.hpp>
#include <fastpls/core/simpls.hpp>
#include <fastpls/core/supervised.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace fastpls {
namespace core {

enum class LinearPlsFamily {
  plssvd = 1,
  simpls = 3
};

enum class ClassificationHead {
  argmax = 0,
  lda = 1
};

enum class RegressionMetric {
  r2 = 2,
  q2 = 3,
  rmsd = 4
};

template<class T>
struct ClassificationCvResult {
  std::vector<int> folds;
  std::vector<int> status;
  std::vector<double> metrics;
  Matrix<int> predictions;
};

template<class T>
struct RegressionCvResult {
  std::vector<int> folds;
  std::vector<int> status;
  std::vector<double> metrics;
  std::vector<Matrix<T>> predictions;
};

namespace cv_detail {

struct FoldPartition {
  std::vector<std::size_t> train;
  std::vector<std::size_t> test;
};

inline std::vector<FoldPartition> fold_partitions(
    const int* folds, std::size_t sample_count) {
  if (folds == nullptr || sample_count < 2) {
    throw std::invalid_argument("cross-validation folds are invalid");
  }
  int maximum = 0;
  for (std::size_t sample = 0; sample < sample_count; ++sample) {
    if (folds[sample] < 1) {
      throw std::invalid_argument("cross-validation folds must start at one");
    }
    maximum = std::max(maximum, folds[sample]);
  }
  std::vector<FoldPartition> output(static_cast<std::size_t>(maximum));
  for (std::size_t sample = 0; sample < sample_count; ++sample) {
    for (int fold = 1; fold <= maximum; ++fold) {
      auto& partition = output[static_cast<std::size_t>(fold - 1)];
      if (folds[sample] == fold) {
        partition.test.push_back(sample);
      } else {
        partition.train.push_back(sample);
      }
    }
  }
  return output;
}

template<class T>
Matrix<T> gather_rows(ConstMatrixView<T> source,
                      const std::vector<std::size_t>& rows) {
  Matrix<T> output(rows.size(), source.columns());
  for (std::size_t column = 0; column < source.columns(); ++column) {
    for (std::size_t row = 0; row < rows.size(); ++row) {
      output(row, column) = source(rows[row], column);
    }
  }
  return output;
}

template<class T>
void standardize(MatrixView<T> values, const std::vector<T>& center,
                 const std::vector<T>& scale) {
  if (values.columns() != center.size() || center.size() != scale.size()) {
    throw std::invalid_argument(
      "cross-validation predictor preprocessing is inconsistent"
    );
  }
  for (std::size_t column = 0; column < values.columns(); ++column) {
    if (!std::isfinite(scale[column]) || scale[column] <= T(0)) {
      throw std::runtime_error("cross-validation predictor scale is invalid");
    }
    for (std::size_t row = 0; row < values.rows(); ++row) {
      values(row, column) =
        (values(row, column) - center[column]) / scale[column];
    }
  }
}

template<class T, class Backend>
Matrix<T> project_scores(ConstMatrixView<T> predictors,
                         ConstMatrixView<T> weights,
                         std::size_t components, Backend& backend) {
  if (components < 1 || components > weights.columns() ||
      predictors.columns() != weights.rows()) {
    throw std::invalid_argument(
      "cross-validation score projection dimensions are invalid"
    );
  }
  ConstMatrixView<T> prefix(
    weights.data(), weights.rows(), components, weights.leading_dimension()
  );
  Matrix<T> scores(predictors.rows(), components);
  backend.gemm(predictors, prefix, false, false, scores.view());
  return scores;
}

template<class T, class Backend>
Matrix<T> predict_plssvd(const PlssvdModel<T>& model,
                         ConstMatrixView<T> predictors,
                         std::size_t prefix_index,
                         const std::vector<T>& response_mean,
                         Backend& backend) {
  if (prefix_index >= model.components.size() ||
      prefix_index >= model.prediction_weights.size()) {
    throw std::invalid_argument("PLS-SVD prediction prefix is invalid");
  }
  const std::size_t components = static_cast<std::size_t>(
    model.components[prefix_index]
  );
  Matrix<T> scores = project_scores(
    predictors, model.weights.view(), components, backend
  );
  Matrix<T> prediction(predictors.rows(), response_mean.size());
  backend.gemm(
    scores.view(), model.prediction_weights[prefix_index].view(),
    false, false, prediction.view()
  );
  for (std::size_t column = 0; column < prediction.columns(); ++column) {
    for (std::size_t row = 0; row < prediction.rows(); ++row) {
      prediction(row, column) += response_mean[column];
    }
  }
  return prediction;
}

template<class T, class Backend>
Matrix<T> predict_simpls(const SimplsModel<T>& model,
                         ConstMatrixView<T> predictors,
                         std::size_t components,
                         const std::vector<T>& response_mean,
                         Backend& backend) {
  Matrix<T> prediction = predict_simpls_preprocessed(
    predictors, model, components, backend
  );
  for (std::size_t column = 0; column < prediction.columns(); ++column) {
    for (std::size_t row = 0; row < prediction.rows(); ++row) {
      prediction(row, column) += response_mean[column];
    }
  }
  return prediction;
}

template<class T>
std::vector<int> active_classes(const int* labels,
                                const std::vector<std::size_t>& rows,
                                std::size_t class_count) {
  std::vector<int> counts(class_count, 0);
  for (const std::size_t row : rows) {
    const int label = labels[row];
    if (label < 1 || static_cast<std::size_t>(label) > class_count) {
      throw std::invalid_argument(
        "cross-validation labels must be encoded as 1..n_classes"
      );
    }
    ++counts[static_cast<std::size_t>(label - 1)];
  }
  std::vector<int> active;
  for (std::size_t label = 0; label < class_count; ++label) {
    if (counts[label] > 0) active.push_back(static_cast<int>(label) + 1);
  }
  return active;
}

inline std::vector<std::size_t> compact_labels(
    const int* labels, const std::vector<std::size_t>& rows,
    const std::vector<int>& active) {
  std::vector<int> map(active.empty() ? 0 : active.back() + 1, -1);
  for (std::size_t index = 0; index < active.size(); ++index) {
    if (active[index] >= static_cast<int>(map.size())) {
      map.resize(static_cast<std::size_t>(active[index]) + 1, -1);
    }
    map[static_cast<std::size_t>(active[index])] = static_cast<int>(index);
  }
  std::vector<std::size_t> output(rows.size());
  for (std::size_t index = 0; index < rows.size(); ++index) {
    const int label = labels[rows[index]];
    if (label < 0 || static_cast<std::size_t>(label) >= map.size() ||
        map[static_cast<std::size_t>(label)] < 0) {
      throw std::invalid_argument("cross-validation class mapping failed");
    }
    output[index] = static_cast<std::size_t>(
      map[static_cast<std::size_t>(label)]
    );
  }
  return output;
}

template<class T>
std::vector<int> predicted_classes(ConstMatrixView<T> scores,
                                   const std::vector<int>& active) {
  if (scores.columns() != active.size()) {
    throw std::invalid_argument(
      "cross-validation class-score dimensions are invalid"
    );
  }
  std::vector<int> output(scores.rows());
  for (std::size_t row = 0; row < scores.rows(); ++row) {
    output[row] = active[row_argmax(scores, row)];
  }
  return output;
}

template<class T>
std::vector<int> lda_predictions(ConstMatrixView<T> scores,
                                 const LdaModel<T>& model,
                                 const std::vector<int>& active) {
  Matrix<T> discriminants = lda_scores(scores, model);
  return predicted_classes<T>(
    ConstMatrixView<T>(discriminants.view()), active
  );
}

template<class T>
void store_classes(Matrix<int>& destination,
                   const std::vector<std::size_t>& rows,
                   std::size_t prefix,
                   const std::vector<int>& values) {
  if (rows.size() != values.size()) {
    throw std::invalid_argument(
      "cross-validation class prediction size is invalid"
    );
  }
  for (std::size_t index = 0; index < rows.size(); ++index) {
    destination(rows[index], prefix) = values[index];
  }
}

template<class T>
void store_values(Matrix<T>& destination,
                  const std::vector<std::size_t>& rows,
                  ConstMatrixView<T> values) {
  if (rows.size() != values.rows() ||
      destination.columns() != values.columns()) {
    throw std::invalid_argument(
      "cross-validation response prediction size is invalid"
    );
  }
  for (std::size_t column = 0; column < values.columns(); ++column) {
    for (std::size_t index = 0; index < rows.size(); ++index) {
      destination(rows[index], column) = values(index, column);
    }
  }
}

}  // namespace cv_detail

template<class T, class Backend>
ClassificationCvResult<T> cross_validate_classification(
    ConstMatrixView<T> predictors, const int* labels,
    std::size_t class_count, const int* folds,
    const int* components, std::size_t prefix_count,
    PredictorScaling scaling, LinearPlsFamily family,
    ClassificationHead head, const PlssvdControls& plssvd_controls,
    const SimplsControls& simpls_controls, Backend& backend,
    bool store_predictions) {
  if (predictors.empty() || labels == nullptr || class_count < 2 ||
      components == nullptr || prefix_count < 1) {
    throw std::invalid_argument(
      "classification cross-validation inputs are invalid"
    );
  }
  const auto partitions = cv_detail::fold_partitions(
    folds, predictors.rows()
  );
  ClassificationCvResult<T> result;
  result.folds.assign(folds, folds + predictors.rows());
  result.status.assign(partitions.size(), 0);
  result.metrics.assign(prefix_count, 0.0);
  if (store_predictions) {
    result.predictions.resize(predictors.rows(), prefix_count);
  }
  std::vector<double> totals(prefix_count, 0.0);

  for (std::size_t fold = 0; fold < partitions.size(); ++fold) {
    const auto& partition = partitions[fold];
    if (partition.test.empty()) {
      result.status[fold] = 2;
      continue;
    }
    const auto active = cv_detail::active_classes<T>(
      labels, partition.train, class_count
    );
    if (active.size() <= 1) {
      const int fallback = active.empty() ? 1 : active.front();
      for (std::size_t prefix = 0; prefix < prefix_count; ++prefix) {
        std::vector<int> predicted(partition.test.size(), fallback);
        if (store_predictions) {
          cv_detail::store_classes<T>(
            result.predictions, partition.test, prefix, predicted
          );
        }
        for (const std::size_t row : partition.test) {
          result.metrics[prefix] += labels[row] == fallback ? 1.0 : 0.0;
          totals[prefix] += 1.0;
        }
      }
      result.status[fold] = 4;
      continue;
    }

    Matrix<T> train = cv_detail::gather_rows(predictors, partition.train);
    Matrix<T> test = cv_detail::gather_rows(predictors, partition.test);
    const auto compact = cv_detail::compact_labels(
      labels, partition.train, active
    );
    const auto prepared = prepare_scaled_label_crossprod(
      train.view(), compact.data(), compact.size(), active.size(),
      scaling, backend
    );
    cv_detail::standardize(
      test.view(), prepared.predictor_center, prepared.predictor_scale
    );

    if (family == LinearPlsFamily::plssvd) {
      PlssvdControls controls = plssvd_controls;
      controls.rsvd.seed += static_cast<unsigned int>(fold);
      auto model = fit_plssvd_preprocessed<T>(
        train.view(), prepared.crossprod.view(), components, prefix_count,
        controls, backend
      );
      std::vector<LdaModel<T>> lda_models;
      Matrix<T> test_scores;
      if (head == ClassificationHead::lda) {
        std::vector<int> lda_labels(compact.size());
        for (std::size_t i = 0; i < compact.size(); ++i) {
          lda_labels[i] = static_cast<int>(compact[i]) + 1;
        }
        lda_models = train_lda_prefixes(
          model.scores.view(), lda_labels.data(), lda_labels.size(),
          active.size(), components, prefix_count
        );
        test_scores = cv_detail::project_scores<T>(
          ConstMatrixView<T>(test.view()),
          ConstMatrixView<T>(model.weights.view()),
          model.completed_components, backend
        );
      }
      for (std::size_t prefix = 0; prefix < prefix_count; ++prefix) {
        std::vector<int> predicted;
        if (head == ClassificationHead::lda) {
          ConstMatrixView<T> score_prefix(
            test_scores.data(), test_scores.rows(),
            static_cast<std::size_t>(components[prefix]), test_scores.rows()
          );
          predicted = cv_detail::lda_predictions<T>(
            score_prefix, lda_models[prefix], active
          );
        } else {
          const auto scores = cv_detail::predict_plssvd<T>(
            model, ConstMatrixView<T>(test.view()), prefix,
            prepared.response_mean, backend
          );
          predicted = cv_detail::predicted_classes<T>(
            ConstMatrixView<T>(scores.view()), active
          );
        }
        if (store_predictions) {
          cv_detail::store_classes<T>(
            result.predictions, partition.test, prefix, predicted
          );
        }
        for (std::size_t i = 0; i < partition.test.size(); ++i) {
          result.metrics[prefix] +=
            predicted[i] == labels[partition.test[i]] ? 1.0 : 0.0;
          totals[prefix] += 1.0;
        }
      }
    } else if (family == LinearPlsFamily::simpls) {
      SimplsControls controls = simpls_controls;
      controls.rsvd.seed += static_cast<unsigned int>(fold);
      controls.store_scores = head == ClassificationHead::lda;
      SimplsWorkspace<T> workspace;
      auto model = fit_simpls_preprocessed<T>(
        train.view(), prepared.crossprod.view(), controls, backend, workspace
      );
      std::vector<LdaModel<T>> lda_models;
      Matrix<T> test_scores;
      if (head == ClassificationHead::lda) {
        std::vector<int> lda_labels(compact.size());
        for (std::size_t i = 0; i < compact.size(); ++i) {
          lda_labels[i] = static_cast<int>(compact[i]) + 1;
        }
        lda_models = train_lda_prefixes(
          model.scores.view(), lda_labels.data(), lda_labels.size(),
          active.size(), components, prefix_count
        );
        test_scores = cv_detail::project_scores<T>(
          ConstMatrixView<T>(test.view()),
          ConstMatrixView<T>(model.weights.view()),
          model.completed_components, backend
        );
      }
      for (std::size_t prefix = 0; prefix < prefix_count; ++prefix) {
        std::vector<int> predicted;
        if (head == ClassificationHead::lda) {
          ConstMatrixView<T> score_prefix(
            test_scores.data(), test_scores.rows(),
            static_cast<std::size_t>(components[prefix]), test_scores.rows()
          );
          predicted = cv_detail::lda_predictions<T>(
            score_prefix, lda_models[prefix], active
          );
        } else {
          const auto scores = cv_detail::predict_simpls<T>(
            model, ConstMatrixView<T>(test.view()),
            static_cast<std::size_t>(components[prefix]),
            prepared.response_mean, backend
          );
          predicted = cv_detail::predicted_classes<T>(
            ConstMatrixView<T>(scores.view()), active
          );
        }
        if (store_predictions) {
          cv_detail::store_classes<T>(
            result.predictions, partition.test, prefix, predicted
          );
        }
        for (std::size_t i = 0; i < partition.test.size(); ++i) {
          result.metrics[prefix] +=
            predicted[i] == labels[partition.test[i]] ? 1.0 : 0.0;
          totals[prefix] += 1.0;
        }
      }
    } else {
      throw std::invalid_argument(
        "classification CV supports PLS-SVD and SIMPLS"
      );
    }
    result.status[fold] = 1;
  }
  for (std::size_t prefix = 0; prefix < prefix_count; ++prefix) {
    result.metrics[prefix] = totals[prefix] > 0.0 ?
      result.metrics[prefix] / totals[prefix] :
      std::numeric_limits<double>::quiet_NaN();
  }
  return result;
}

template<class T, class Backend>
RegressionCvResult<T> cross_validate_regression(
    ConstMatrixView<T> predictors, ConstMatrixView<T> responses,
    const int* folds, const int* components, std::size_t prefix_count,
    PredictorScaling scaling, LinearPlsFamily family,
    RegressionMetric metric, const PlssvdControls& plssvd_controls,
    const SimplsControls& simpls_controls, Backend& backend,
    bool store_predictions) {
  if (predictors.empty() || responses.empty() ||
      predictors.rows() != responses.rows() || components == nullptr ||
      prefix_count < 1) {
    throw std::invalid_argument("regression cross-validation inputs are invalid");
  }
  const auto partitions = cv_detail::fold_partitions(
    folds, predictors.rows()
  );
  RegressionCvResult<T> result;
  result.folds.assign(folds, folds + predictors.rows());
  result.status.assign(partitions.size(), 0);
  result.metrics.assign(prefix_count, 0.0);
  std::vector<double> counts(prefix_count, 0.0);
  if (store_predictions) {
    result.predictions.reserve(prefix_count);
    for (std::size_t prefix = 0; prefix < prefix_count; ++prefix) {
      result.predictions.emplace_back(predictors.rows(), responses.columns());
    }
  }
  long double total_sum = 0.0L;
  long double total_square = 0.0L;
  for (std::size_t column = 0; column < responses.columns(); ++column) {
    for (std::size_t row = 0; row < responses.rows(); ++row) {
      const long double value = responses(row, column);
      total_sum += value;
      total_square += value * value;
    }
  }
  const long double observation_count = static_cast<long double>(
    responses.rows() * responses.columns()
  );
  const long double total_ss = total_square -
    total_sum * total_sum / observation_count;

  for (std::size_t fold = 0; fold < partitions.size(); ++fold) {
    const auto& partition = partitions[fold];
    if (partition.test.empty()) {
      result.status[fold] = 2;
      continue;
    }
    Matrix<T> train = cv_detail::gather_rows(predictors, partition.train);
    Matrix<T> test = cv_detail::gather_rows(predictors, partition.test);
    Matrix<T> train_response = cv_detail::gather_rows(
      responses, partition.train
    );
    const auto prepared = prepare_scaled_dense_crossprod(
      train.view(), train_response.view(), scaling, backend
    );
    cv_detail::standardize(
      test.view(), prepared.predictor_center, prepared.predictor_scale
    );

    if (family == LinearPlsFamily::plssvd) {
      PlssvdControls controls = plssvd_controls;
      controls.rsvd.seed += static_cast<unsigned int>(fold);
      auto model = fit_plssvd_preprocessed<T>(
        train.view(), prepared.crossprod.view(), components, prefix_count,
        controls, backend
      );
      for (std::size_t prefix = 0; prefix < prefix_count; ++prefix) {
        const auto prediction = cv_detail::predict_plssvd<T>(
          model, ConstMatrixView<T>(test.view()), prefix,
          prepared.response_mean, backend
        );
        for (std::size_t column = 0; column < prediction.columns(); ++column) {
          for (std::size_t index = 0; index < partition.test.size(); ++index) {
            const long double error = prediction(index, column) -
              responses(partition.test[index], column);
            result.metrics[prefix] += static_cast<double>(error * error);
            counts[prefix] += 1.0;
          }
        }
        if (store_predictions) {
          cv_detail::store_values(
            result.predictions[prefix], partition.test, prediction.view()
          );
        }
      }
    } else if (family == LinearPlsFamily::simpls) {
      SimplsControls controls = simpls_controls;
      controls.rsvd.seed += static_cast<unsigned int>(fold);
      controls.store_scores = false;
      SimplsWorkspace<T> workspace;
      auto model = fit_simpls_preprocessed<T>(
        train.view(), prepared.crossprod.view(), controls, backend, workspace
      );
      for (std::size_t prefix = 0; prefix < prefix_count; ++prefix) {
        const auto prediction = cv_detail::predict_simpls<T>(
          model, ConstMatrixView<T>(test.view()),
          static_cast<std::size_t>(components[prefix]),
          prepared.response_mean, backend
        );
        for (std::size_t column = 0; column < prediction.columns(); ++column) {
          for (std::size_t index = 0; index < partition.test.size(); ++index) {
            const long double error = prediction(index, column) -
              responses(partition.test[index], column);
            result.metrics[prefix] += static_cast<double>(error * error);
            counts[prefix] += 1.0;
          }
        }
        if (store_predictions) {
          cv_detail::store_values(
            result.predictions[prefix], partition.test, prediction.view()
          );
        }
      }
    } else {
      throw std::invalid_argument("regression CV supports PLS-SVD and SIMPLS");
    }
    result.status[fold] = 1;
  }

  for (std::size_t prefix = 0; prefix < prefix_count; ++prefix) {
    if (metric == RegressionMetric::rmsd) {
      result.metrics[prefix] = counts[prefix] > 0.0 ?
        std::sqrt(result.metrics[prefix] / counts[prefix]) :
        std::numeric_limits<double>::quiet_NaN();
    } else {
      result.metrics[prefix] = total_ss > 0.0L ?
        1.0 - result.metrics[prefix] / static_cast<double>(total_ss) :
        std::numeric_limits<double>::quiet_NaN();
    }
  }
  return result;
}

}  // namespace core
}  // namespace fastpls

#endif
